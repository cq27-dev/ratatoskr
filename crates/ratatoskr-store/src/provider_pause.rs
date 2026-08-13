//! Durable delivery state for automatic provider pauses.
//!
//! The dashboard owns this instance-level SQLite state because a provider pause must outlive a
//! dashboard process, while project checkpoint stores remain single-writer files owned by runs.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ratatoskr_core::normalized_node_name;
use rusqlite::{Connection, OptionalExtension, params};

use crate::StoreError;

/// The instance-wide identity of one run.
///
/// A run id is operator-chosen and unique only within its project, so provider pause delivery may
/// never use the id alone.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProviderPauseKey {
    project: String,
    run_id: String,
}

impl ProviderPauseKey {
    pub fn new(project: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            project: project.into(),
            run_id: run_id.into(),
        }
    }

    pub fn project(&self) -> &str {
        &self.project
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }
}

/// What a provider-pause waiter should do after registering or polling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderPauseDisposition {
    Hold,
    Continue,
    /// An operator stopped this node while it was waiting for provider recovery.
    Stop,
    /// The run's process has exited, so no provider retry may be made.
    Exited,
}

/// The server's durable registration for one provider-pause waiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderPauseRegistration {
    pub disposition: ProviderPauseDisposition,
    pub generation: i64,
}

/// The durable, instance-wide half of provider pause delivery. Cheap to clone.
#[derive(Clone)]
pub struct ProviderPauseStore {
    conn: Arc<Mutex<Connection>>,
}

impl ProviderPauseStore {
    /// Open the instance database at `path`, creating the provider-pause schema if needed.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::from_connection(crate::open_sqlite(path.as_ref())?)
    }

    /// An in-memory provider-pause ledger, for router tests.
    pub fn open_in_memory() -> Result<Self, StoreError> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(conn: Connection) -> Result<Self, StoreError> {
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        migrate_schema(&conn)?;
        conn.execute_batch(include_str!("provider_pause_schema.sql"))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Register `waiter` idempotently and return its durable pause generation.
    ///
    /// A waiter that arrives after an operator resume joins the retained tombstone and receives
    /// `Continue`; it must acknowledge that delivery before the tombstone may be removed. A child
    /// that has acknowledged that generation asks for a fresh one on its next provider failure,
    /// even while another old waiter still has not acknowledged. A confirmed exit is a durable
    /// fence: it rejects every late registration that races process death.
    pub async fn register(
        &self,
        key: &ProviderPauseKey,
        waiter: &str,
        acknowledged_generation: Option<i64>,
    ) -> Result<ProviderPauseRegistration, StoreError> {
        self.register_for_node(key, waiter, "", acknowledged_generation)
            .await
    }

    /// Register `waiter` for `node` idempotently and return its durable pause generation.
    ///
    /// Node identity lets a durable [`ProviderPauseDisposition::Stop`] override a run-wide
    /// resume only for the node the operator selected.
    pub async fn register_for_node(
        &self,
        key: &ProviderPauseKey,
        waiter: &str,
        node: &str,
        acknowledged_generation: Option<i64>,
    ) -> Result<ProviderPauseRegistration, StoreError> {
        let conn = Arc::clone(&self.conn);
        let (key, waiter, node) = (key.clone(), waiter.to_string(), normalized_node_name(node));
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("provider pause store mutex poisoned");
            let tx = conn.transaction()?;
            let exited: Option<i64> = tx
                .query_row(
                    "SELECT exited FROM provider_pause_runs WHERE project = ?1 AND run_id = ?2",
                    params![key.project(), key.run_id()],
                    |row| row.get(0),
                )
                .optional()?;
            if exited == Some(1) {
                tx.commit()?;
                return Ok::<_, StoreError>(ProviderPauseRegistration {
                    disposition: ProviderPauseDisposition::Exited,
                    generation: 0,
                });
            }

            let existing: Option<(i64, i64)> = tx
                .query_row(
                    "SELECT waiter.generation, pause.resumed
                     FROM provider_pause_waiters waiter
                     JOIN provider_pause_generations pause
                       ON pause.project = waiter.project
                      AND pause.run_id = waiter.run_id
                      AND pause.generation = waiter.generation
                     WHERE waiter.project = ?1 AND waiter.run_id = ?2 AND waiter.waiter = ?3",
                    params![key.project(), key.run_id(), &waiter],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if let Some((generation, resumed)) = existing {
                tx.execute(
                    "UPDATE provider_pause_waiters SET node = ?5
                     WHERE project = ?1 AND run_id = ?2 AND generation = ?3 AND waiter = ?4",
                    params![key.project(), key.run_id(), generation, &waiter, &node],
                )?;
                let stopped = is_node_stopped(&tx, &key, &node)?;
                touch(&tx, &key)?;
                tx.commit()?;
                return Ok::<_, StoreError>(ProviderPauseRegistration {
                    disposition: disposition(resumed, stopped),
                    generation,
                });
            }

            let latest_generation: Option<i64> = tx
                .query_row(
                    "SELECT latest_generation FROM provider_pause_runs
                     WHERE project = ?1 AND run_id = ?2",
                    params![key.project(), key.run_id()],
                    |row| row.get(0),
                )
                .optional()?;
            let (generation, resumed) = match latest_generation {
                None => {
                    let generation = 1;
                    tx.execute(
                        "INSERT INTO provider_pause_runs
                            (project, run_id, latest_generation, last_seen_ms)
                         VALUES (?1, ?2, ?3, ?4)",
                        params![key.project(), key.run_id(), generation, now_ms()],
                    )?;
                    tx.execute(
                        "INSERT INTO provider_pause_generations (project, run_id, generation)
                         VALUES (?1, ?2, ?3)",
                        params![key.project(), key.run_id(), generation],
                    )?;
                    (generation, 0)
                }
                Some(latest_generation) => {
                    let resumed: Option<i64> = tx
                        .query_row(
                            "SELECT resumed FROM provider_pause_generations
                             WHERE project = ?1 AND run_id = ?2 AND generation = ?3",
                            params![key.project(), key.run_id(), latest_generation],
                            |row| row.get(0),
                        )
                        .optional()?;
                    match (resumed, acknowledged_generation == Some(latest_generation)) {
                        (None | Some(1), true) | (None, false) => {
                            let generation = latest_generation
                                .checked_add(1)
                                .expect("provider pause generation overflow");
                            tx.execute(
                                "UPDATE provider_pause_runs
                                 SET latest_generation = ?3, last_seen_ms = ?4
                                 WHERE project = ?1 AND run_id = ?2",
                                params![key.project(), key.run_id(), generation, now_ms()],
                            )?;
                            tx.execute(
                                "INSERT INTO provider_pause_generations (project, run_id, generation)
                                 VALUES (?1, ?2, ?3)",
                                params![key.project(), key.run_id(), generation],
                            )?;
                            (generation, 0)
                        }
                        (Some(resumed), _) => {
                            touch(&tx, &key)?;
                            (latest_generation, resumed)
                        }
                    }
                }
            };
            tx.execute(
                "INSERT INTO provider_pause_waiters (project, run_id, generation, waiter, node)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![key.project(), key.run_id(), generation, &waiter, &node],
            )?;
            let stopped = is_node_stopped(&tx, &key, &node)?;
            tx.commit()?;
            Ok::<_, StoreError>(ProviderPauseRegistration {
                disposition: disposition(resumed, stopped),
                generation,
            })
        })
        .await?
    }

    /// Whether an unacknowledged provider waiter is still holding this run.
    pub async fn is_holding(&self, key: &ProviderPauseKey) -> Result<bool, StoreError> {
        let conn = Arc::clone(&self.conn);
        let key = key.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("provider pause store mutex poisoned");
            let holding: i64 = conn.query_row(
                "SELECT EXISTS(
                    SELECT 1
                    FROM provider_pause_runs run
                    JOIN provider_pause_generations pause
                      ON pause.project = run.project AND pause.run_id = run.run_id
                    JOIN provider_pause_waiters waiter
                      ON waiter.project = pause.project
                     AND waiter.run_id = pause.run_id
                     AND waiter.generation = pause.generation
                    WHERE run.project = ?1 AND run.run_id = ?2 AND run.exited = 0
                      AND pause.resumed = 0 AND waiter.acknowledged = 0
                )",
                params![key.project(), key.run_id()],
                |row| row.get(0),
            )?;
            Ok::<_, StoreError>(holding != 0)
        })
        .await?
    }

    /// Whether a confirmed child exit prevents a provider retry for this run.
    pub async fn is_exited(&self, key: &ProviderPauseKey) -> Result<bool, StoreError> {
        let conn = Arc::clone(&self.conn);
        let key = key.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("provider pause store mutex poisoned");
            let exited: Option<i64> = conn
                .query_row(
                    "SELECT exited FROM provider_pause_runs WHERE project = ?1 AND run_id = ?2",
                    params![key.project(), key.run_id()],
                    |row| row.get(0),
                )
                .optional()?;
            Ok::<_, StoreError>(exited == Some(1))
        })
        .await?
    }

    /// Whether `node` has a durable Stop that survives provider-pause acknowledgement.
    ///
    /// A stopped node continues polling while it is parked, which renews the lifecycle heartbeat
    /// and prevents restart reconciliation from mistaking the live child for an abandoned pause.
    pub async fn is_node_stopped(
        &self,
        key: &ProviderPauseKey,
        node: &str,
    ) -> Result<bool, StoreError> {
        let conn = Arc::clone(&self.conn);
        let (key, node) = (key.clone(), normalized_node_name(node));
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("provider pause store mutex poisoned");
            let tx = conn.transaction()?;
            let exited: Option<i64> = tx
                .query_row(
                    "SELECT exited FROM provider_pause_runs WHERE project = ?1 AND run_id = ?2",
                    params![key.project(), key.run_id()],
                    |row| row.get(0),
                )
                .optional()?;
            if exited == Some(1) {
                tx.commit()?;
                return Ok::<_, StoreError>(false);
            }
            let stopped = is_node_stopped(&tx, &key, &node)?;
            if stopped {
                touch(&tx, &key)?;
            }
            tx.commit()?;
            Ok::<_, StoreError>(stopped)
        })
        .await?
    }

    /// Durable nodes that remain stopped after a dashboard restart.
    pub async fn stopped_nodes(&self, key: &ProviderPauseKey) -> Result<Vec<String>, StoreError> {
        let conn = Arc::clone(&self.conn);
        let key = key.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("provider pause store mutex poisoned");
            let mut statement = conn.prepare(
                "SELECT stop.node
                 FROM provider_pause_stops stop
                 JOIN provider_pause_runs run
                   ON run.project = stop.project AND run.run_id = stop.run_id
                 WHERE stop.project = ?1 AND stop.run_id = ?2 AND run.exited = 0
                 ORDER BY stop.node",
            )?;
            statement
                .query_map(params![key.project(), key.run_id()], |row| row.get(0))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(StoreError::from)
        })
        .await?
    }

    /// Persist an operator resume before any waiter can observe it.
    pub async fn release(&self, key: &ProviderPauseKey) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let key = key.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("provider pause store mutex poisoned");
            conn.execute(
                "UPDATE provider_pause_generations SET resumed = 1
                 WHERE project = ?1 AND run_id = ?2 AND resumed = 0
                   AND EXISTS (
                       SELECT 1 FROM provider_pause_runs run
                       WHERE run.project = ?1 AND run.run_id = ?2 AND run.exited = 0
                   )",
                params![key.project(), key.run_id()],
            )?;
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// Persist an operator Stop before, during, or after provider delivery for `node`.
    ///
    /// The generation-zero run row stages a Stop that reaches the store just before its waiter is
    /// registered. A Stop may also follow Resume before a child receives its response, so it must
    /// cover retained resumed generations as well as active holds. Unlike a resume tombstone, Stop
    /// remains after acknowledgement until [`Self::clear_stop`] observes Start.
    pub async fn stop(&self, key: &ProviderPauseKey, node: &str) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let (key, node) = (key.clone(), normalized_node_name(node));
        let stopped_at_ms = now_ms();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("provider pause store mutex poisoned");
            let tx = conn.transaction()?;
            tx.execute(
                "INSERT INTO provider_pause_runs
                    (project, run_id, latest_generation, exited, last_seen_ms)
                 VALUES (?1, ?2, 0, 0, ?3)
                 ON CONFLICT(project, run_id) DO NOTHING",
                params![key.project(), key.run_id(), stopped_at_ms],
            )?;
            tx.execute(
                "INSERT INTO provider_pause_stops (project, run_id, node)
                 SELECT ?1, ?2, ?3
                 WHERE EXISTS (
                     SELECT 1 FROM provider_pause_runs run
                     WHERE run.project = ?1 AND run.run_id = ?2 AND run.exited = 0
                 )
                 ON CONFLICT(project, run_id, node) DO NOTHING",
                params![key.project(), key.run_id(), &node],
            )?;
            tx.commit()?;
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// Remove a durable Stop after the operator starts `node` again.
    pub async fn clear_stop(&self, key: &ProviderPauseKey, node: &str) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let (key, node) = (key.clone(), normalized_node_name(node));
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("provider pause store mutex poisoned");
            let tx = conn.transaction()?;
            tx.execute(
                "DELETE FROM provider_pause_stops
                 WHERE project = ?1 AND run_id = ?2 AND node = ?3",
                params![key.project(), key.run_id(), &node],
            )?;
            tx.execute(
                "DELETE FROM provider_pause_runs
                 WHERE project = ?1 AND run_id = ?2
                   AND NOT EXISTS (
                       SELECT 1 FROM provider_pause_generations
                       WHERE project = ?1 AND run_id = ?2
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM provider_pause_stops
                       WHERE project = ?1 AND run_id = ?2
                   )",
                params![key.project(), key.run_id()],
            )?;
            tx.commit()?;
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// Fence a confirmed child exit, clearing deliveries while retaining the terminal lifecycle.
    ///
    /// The fence makes an exit cleanup race-safe: a pause request that reaches the dashboard after
    /// process death receives [`ProviderPauseDisposition::Exited`] instead of recreating a hold.
    pub async fn record_exit(&self, key: &ProviderPauseKey) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let key = key.clone();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("provider pause store mutex poisoned");
            let tx = conn.transaction()?;
            let updated = tx.execute(
                "UPDATE provider_pause_runs
                 SET exited = 1, last_seen_ms = ?3
                 WHERE project = ?1 AND run_id = ?2",
                params![key.project(), key.run_id(), now_ms()],
            )?;
            if updated == 0 {
                tx.execute(
                    "INSERT INTO provider_pause_runs
                        (project, run_id, latest_generation, exited, last_seen_ms)
                     VALUES (?1, ?2, 0, 1, ?3)",
                    params![key.project(), key.run_id(), now_ms()],
                )?;
            }
            tx.execute(
                "DELETE FROM provider_pause_generations WHERE project = ?1 AND run_id = ?2",
                params![key.project(), key.run_id()],
            )?;
            tx.execute(
                "DELETE FROM provider_pause_stops WHERE project = ?1 AND run_id = ?2",
                params![key.project(), key.run_id()],
            )?;
            tx.commit()?;
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// List runs with an outstanding provider delivery older than `cutoff_ms`.
    ///
    /// Every paused poll and acknowledgement refreshes `last_seen_ms`. A stale heartbeat is not
    /// proof that the child exited: host suspension, scheduler delay, and dashboard downtime can
    /// all produce it. Only an observed process exit may call [`Self::record_exit`]; callers use
    /// this query to report a suspect pause while retaining its recoverable delivery state.
    pub async fn list_unresponsive_before(
        &self,
        cutoff_ms: i64,
    ) -> Result<Vec<ProviderPauseKey>, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("provider pause store mutex poisoned");
            let mut statement = conn.prepare(
                "SELECT run.project, run.run_id
                 FROM provider_pause_runs run
                 WHERE run.exited = 0 AND run.last_seen_ms < ?1
                   AND EXISTS (
                       SELECT 1 FROM provider_pause_waiters waiter
                       WHERE waiter.project = run.project AND waiter.run_id = run.run_id
                         AND waiter.acknowledged = 0
                   )",
            )?;
            let keys = statement
                .query_map(params![cutoff_ms], |row| {
                    Ok(ProviderPauseKey::new(
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok::<_, StoreError>(keys)
        })
        .await?
    }

    /// List pauses whose child has not renewed its liveness heartbeat within `grace`.
    pub async fn list_unresponsive_for(
        &self,
        grace: Duration,
    ) -> Result<Vec<ProviderPauseKey>, StoreError> {
        let grace_ms: i64 = grace
            .as_millis()
            .try_into()
            .expect("provider pause liveness grace fits in i64 milliseconds");
        self.list_unresponsive_before(now_ms().saturating_sub(grace_ms))
            .await
    }

    /// Acknowledge a delivered `Continue` or `Stop` and return the current durable directive.
    ///
    /// Repeating an acknowledgement is a success. A response may be lost after this transaction
    /// commits; keeping the acknowledged waiter through confirmed child exit lets a later retry
    /// observe a Stop that arrived in that response-loss window.
    pub async fn acknowledge(
        &self,
        key: &ProviderPauseKey,
        generation: i64,
        waiter: &str,
        node: &str,
    ) -> Result<ProviderPauseDisposition, StoreError> {
        let conn = Arc::clone(&self.conn);
        let (key, waiter, node) = (key.clone(), waiter.to_string(), normalized_node_name(node));
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("provider pause store mutex poisoned");
            let tx = conn.transaction()?;
            let exited: Option<i64> = tx
                .query_row(
                    "SELECT exited FROM provider_pause_runs WHERE project = ?1 AND run_id = ?2",
                    params![key.project(), key.run_id()],
                    |row| row.get(0),
                )
                .optional()?;
            if exited == Some(1) {
                tx.commit()?;
                return Ok::<_, StoreError>(ProviderPauseDisposition::Exited);
            }
            // An acknowledgement may be retried after its first response is lost. Renew this
            // child's liveness before recording it so restart reconciliation cannot fence a live
            // waiter while it is still completing the delivery protocol.
            touch(&tx, &key)?;
            tx.execute(
                "UPDATE provider_pause_waiters
                 SET acknowledged = 1
                 WHERE project = ?1 AND run_id = ?2 AND generation = ?3 AND waiter = ?4",
                params![key.project(), key.run_id(), generation, &waiter],
            )?;
            let disposition = if is_node_stopped(&tx, &key, &node)? {
                ProviderPauseDisposition::Stop
            } else {
                ProviderPauseDisposition::Continue
            };
            tx.commit()?;
            Ok::<_, StoreError>(disposition)
        })
        .await?
    }
}

fn disposition(resumed: i64, stopped: bool) -> ProviderPauseDisposition {
    if stopped {
        ProviderPauseDisposition::Stop
    } else if resumed == 0 {
        ProviderPauseDisposition::Hold
    } else {
        ProviderPauseDisposition::Continue
    }
}

fn is_node_stopped(
    tx: &rusqlite::Transaction<'_>,
    key: &ProviderPauseKey,
    node: &str,
) -> Result<bool, StoreError> {
    tx.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM provider_pause_stops
            WHERE project = ?1 AND run_id = ?2 AND node = ?3
        )",
        params![key.project(), key.run_id(), node],
        |row| row.get::<_, i64>(0),
    )
    .map(|stopped| stopped != 0)
    .map_err(StoreError::from)
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is before the Unix epoch")
        .as_millis()
        .try_into()
        .expect("milliseconds since the Unix epoch fit in i64")
}

fn touch(tx: &rusqlite::Transaction<'_>, key: &ProviderPauseKey) -> Result<(), StoreError> {
    tx.execute(
        "UPDATE provider_pause_runs SET last_seen_ms = ?3
         WHERE project = ?1 AND run_id = ?2 AND exited = 0",
        params![key.project(), key.run_id(), now_ms()],
    )?;
    Ok(())
}

/// The node-key spelling this ledger is written in, stamped on the instance database.
///
/// `PRAGMA user_version` is the whole marker: the rows themselves cannot say which rule wrote them,
/// because the old spelling and the new one overlap for every name without an underscore.
const NODE_KEY_SPELLING: i64 = 1;

fn migrate_schema(conn: &Connection) -> Result<(), StoreError> {
    let spelling: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if spelling < NODE_KEY_SPELLING {
        // Node keys were once spelled with every underscore stripped, so a Stop recorded then
        // addresses no node now: a stage stopped as `security_review` sits under `securityreview`,
        // the child asks under its current name and is told it may run, and the Start that would
        // clear the row deletes under the current spelling and matches nothing. The old spelling
        // cannot be turned back into the new one — underscores were dropped, not encoded, so
        // `securityreview` is as much `securityreview` as `security_review` — so the rows are
        // cleared rather than rewritten, which leaves the ledger saying what the child already
        // believes: that nothing is stopped.
        conn.execute_batch(&format!(
            "DROP TABLE IF EXISTS provider_pause_stops;
             PRAGMA user_version = {NODE_KEY_SPELLING};"
        ))?;
    }
    if !table_exists(conn, "provider_pause_runs")? {
        return Ok(());
    }
    if !has_column(conn, "provider_pause_runs", "project")? {
        // Pre-scoped pause rows cannot be assigned to a project safely. Live children reconstruct
        // a correct hold on their next paused poll; retaining the old row would permit a
        // cross-project resume, which is worse than requiring that reconstruction.
        conn.execute_batch(
            "DROP TABLE IF EXISTS provider_pause_waiters;
             DROP TABLE IF EXISTS provider_pause_generations;
             DROP TABLE IF EXISTS provider_pause_runs;",
        )?;
        return Ok(());
    }
    if !has_column(conn, "provider_pause_runs", "exited")? {
        conn.execute_batch(
            "ALTER TABLE provider_pause_runs
             ADD COLUMN exited INTEGER NOT NULL DEFAULT 0 CHECK (exited IN (0, 1));",
        )?;
    }
    if !has_column(conn, "provider_pause_runs", "last_seen_ms")? {
        conn.execute_batch(
            "ALTER TABLE provider_pause_runs
             ADD COLUMN last_seen_ms INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if table_exists(conn, "provider_pause_waiters")?
        && !has_column(conn, "provider_pause_waiters", "node")?
    {
        conn.execute_batch(
            "ALTER TABLE provider_pause_waiters
             ADD COLUMN node TEXT NOT NULL DEFAULT '';",
        )?;
    }
    if table_exists(conn, "provider_pause_stops")?
        && has_column(conn, "provider_pause_stops", "generation")?
    {
        // A generation-scoped Stop disappears after its waiter acknowledges it, so it cannot keep
        // a parked node stopped across a dashboard restart. Recreate the durable node schema.
        conn.execute_batch("DROP TABLE provider_pause_stops;")?;
    }
    Ok(())
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool, StoreError> {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        params![table],
        |row| row.get::<_, i64>(0),
    )
    .map(|exists| exists != 0)
    .map_err(StoreError::from)
}

fn has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, StoreError> {
    let mut statement = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()
        .map(|columns| columns.iter().any(|name| name == column))
        .map_err(StoreError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> ProviderPauseKey {
        ProviderPauseKey::new("project", "run")
    }

    async fn store() -> ProviderPauseStore {
        ProviderPauseStore::open_in_memory().expect("in-memory provider pause store")
    }

    #[tokio::test]
    async fn a_resume_is_retained_until_every_waiter_acknowledges() {
        let pauses = store().await;
        let key = key();
        assert_eq!(
            pauses.register(&key, "first", None).await.unwrap(),
            ProviderPauseRegistration {
                disposition: ProviderPauseDisposition::Hold,
                generation: 1,
            }
        );
        assert_eq!(
            pauses.register(&key, "second", None).await.unwrap(),
            ProviderPauseRegistration {
                disposition: ProviderPauseDisposition::Hold,
                generation: 1,
            }
        );
        pauses.release(&key).await.unwrap();

        assert_eq!(
            pauses
                .register(&key, "late", None)
                .await
                .unwrap()
                .disposition,
            ProviderPauseDisposition::Continue,
            "a late waiter belongs to the resumed pause rather than re-arming it"
        );
        pauses.acknowledge(&key, 1, "first", "").await.unwrap();
        pauses.acknowledge(&key, 1, "second", "").await.unwrap();
        assert_eq!(
            pauses
                .register(&key, "late", None)
                .await
                .unwrap()
                .disposition,
            ProviderPauseDisposition::Continue,
            "the retained tombstone survives acknowledgement by the other waiters"
        );
        pauses.acknowledge(&key, 1, "late", "").await.unwrap();

        assert_eq!(
            pauses.register(&key, "next", Some(1)).await.unwrap(),
            ProviderPauseRegistration {
                disposition: ProviderPauseDisposition::Hold,
                generation: 2,
            },
            "only every acknowledgement permits the next provider failure to pause again"
        );
    }

    #[tokio::test]
    async fn a_new_pause_after_acknowledgement_gets_a_fresh_generation() {
        let pauses = store().await;
        let key = key();
        pauses.register(&key, "first", None).await.unwrap();
        pauses.register(&key, "second", None).await.unwrap();
        pauses.release(&key).await.unwrap();
        pauses.acknowledge(&key, 1, "first", "").await.unwrap();

        assert_eq!(
            pauses.register(&key, "fresh", Some(1)).await.unwrap(),
            ProviderPauseRegistration {
                disposition: ProviderPauseDisposition::Hold,
                generation: 2,
            },
            "a new provider failure must not reuse a resume intended for generation one"
        );
        assert_eq!(
            pauses.register(&key, "second", None).await.unwrap(),
            ProviderPauseRegistration {
                disposition: ProviderPauseDisposition::Continue,
                generation: 1,
            },
            "the old waiter's lost response still resolves against its original generation"
        );
    }

    #[tokio::test]
    async fn a_lost_acknowledgement_response_can_be_retried_without_rearming() {
        let pauses = store().await;
        let key = key();
        pauses.register(&key, "analyst", None).await.unwrap();
        pauses.release(&key).await.unwrap();

        pauses.acknowledge(&key, 1, "analyst", "").await.unwrap();
        pauses.acknowledge(&key, 1, "analyst", "").await.unwrap();

        assert!(!pauses.is_holding(&key).await.unwrap());
        assert_eq!(
            pauses.register(&key, "next", Some(1)).await.unwrap(),
            ProviderPauseRegistration {
                disposition: ProviderPauseDisposition::Hold,
                generation: 2,
            },
            "only a later provider failure may create the next hold"
        );
    }

    #[tokio::test]
    async fn a_stop_after_a_lost_acknowledgement_response_is_delivered_on_retry() {
        let pauses = store().await;
        let key = key();
        pauses
            .register_for_node(&key, "analyst-waiter", "analyst", None)
            .await
            .unwrap();
        pauses.release(&key).await.unwrap();

        assert_eq!(
            pauses
                .acknowledge(&key, 1, "analyst-waiter", "analyst")
                .await
                .unwrap(),
            ProviderPauseDisposition::Continue,
            "the first reply is assumed lost after this durable acknowledgement commits"
        );
        pauses.stop(&key, "analyst").await.unwrap();

        assert_eq!(
            pauses
                .acknowledge(&key, 1, "analyst-waiter", "analyst")
                .await
                .unwrap(),
            ProviderPauseDisposition::Stop,
            "the retry observes Stop instead of allowing another provider request"
        );
    }

    #[tokio::test]
    async fn a_stop_before_registration_is_delivered_to_the_later_waiter() {
        let pauses = store().await;
        let key = key();

        pauses.stop(&key, "analyst").await.unwrap();
        assert_eq!(
            pauses
                .register_for_node(&key, "analyst-waiter", "analyst", None)
                .await
                .unwrap(),
            ProviderPauseRegistration {
                disposition: ProviderPauseDisposition::Stop,
                generation: 1,
            },
            "registration must consume the Stop staged before a waiter existed"
        );
        assert_eq!(
            pauses
                .acknowledge(&key, 1, "analyst-waiter", "analyst")
                .await
                .unwrap(),
            ProviderPauseDisposition::Stop,
            "acknowledgement cannot turn the staged Stop into Continue"
        );
    }

    #[tokio::test]
    async fn acknowledging_a_stopped_waiter_releases_an_empty_pause() {
        let pauses = store().await;
        let key = key();
        pauses.register(&key, "analyst", None).await.unwrap();
        assert!(pauses.is_holding(&key).await.unwrap());

        pauses.acknowledge(&key, 1, "analyst", "").await.unwrap();
        assert!(!pauses.is_holding(&key).await.unwrap());
        assert_eq!(
            pauses
                .register(&key, "retry", Some(1))
                .await
                .unwrap()
                .disposition,
            ProviderPauseDisposition::Hold
        );
    }

    #[tokio::test]
    async fn a_confirmed_exit_fences_late_provider_pause_registration() {
        let pauses = store().await;
        let key = key();
        pauses.register(&key, "analyst", None).await.unwrap();
        assert!(pauses.is_holding(&key).await.unwrap());

        pauses.record_exit(&key).await.unwrap();

        assert!(pauses.is_exited(&key).await.unwrap());
        assert!(!pauses.is_holding(&key).await.unwrap());
        assert_eq!(
            pauses
                .register(&key, "late", None)
                .await
                .unwrap()
                .disposition,
            ProviderPauseDisposition::Exited,
            "an in-flight pause request cannot recreate a hold after the child exits"
        );
    }

    #[tokio::test]
    async fn a_pause_is_scoped_to_its_project_even_when_run_ids_match() {
        let pauses = store().await;
        let open = ProviderPauseKey::new("open", "same-id");
        let shut = ProviderPauseKey::new("shut", "same-id");
        pauses.register(&open, "analyst", None).await.unwrap();
        pauses.register(&shut, "analyst", None).await.unwrap();

        pauses.release(&open).await.unwrap();

        assert!(!pauses.is_holding(&open).await.unwrap());
        assert!(pauses.is_holding(&shut).await.unwrap());
    }

    #[tokio::test]
    async fn a_stop_spelled_by_the_old_rule_is_cleared_rather_than_left_unaddressable() {
        // Node keys once had their underscores stripped, so a stage stopped as `security_review`
        // was recorded as `securityreview`. Under the current rule nothing addresses that row: the
        // child asks whether `security_review` is stopped and is told it may run, while the
        // dashboard keeps listing a stopped node whose Start deletes under the current spelling.
        // Clearing it makes the ledger agree with the child; keeping it would leave the operator
        // starting a node the store never stops answering for.
        let conn = Connection::open_in_memory().expect("in-memory SQLite connection");
        conn.execute_batch(include_str!("provider_pause_schema.sql"))
            .expect("pause schema");
        conn.execute_batch(
            "INSERT INTO provider_pause_runs
                (project, run_id, latest_generation, last_seen_ms)
             VALUES ('project', 'run', 0, 0);
             INSERT INTO provider_pause_stops (project, run_id, node)
             VALUES ('project', 'run', 'securityreview');",
        )
        .expect("a ledger written under the old spelling");

        let pauses = ProviderPauseStore::from_connection(conn).expect("migrated provider pauses");

        assert!(
            pauses.stopped_nodes(&key()).await.unwrap().is_empty(),
            "a stop nothing can address is cleared, not carried forward"
        );

        // And only once: a Stop written under the current spelling is durable, which is the whole
        // point of the table.
        pauses.stop(&key(), "security_review").await.unwrap();
        {
            let conn = pauses
                .conn
                .lock()
                .expect("provider pause store mutex poisoned");
            migrate_schema(&conn).expect("re-opening a migrated ledger");
        }
        assert_eq!(
            pauses.stopped_nodes(&key()).await.unwrap(),
            ["security_review"]
        );
    }

    #[tokio::test]
    async fn an_unscoped_ledger_is_dropped_before_it_can_cross_projects() {
        let conn = Connection::open_in_memory().expect("in-memory SQLite connection");
        conn.execute_batch(
            "CREATE TABLE provider_pause_runs (
                run_id TEXT PRIMARY KEY,
                latest_generation INTEGER NOT NULL
            );
            INSERT INTO provider_pause_runs (run_id, latest_generation) VALUES ('same-id', 1);",
        )
        .expect("legacy pause schema");

        let pauses = ProviderPauseStore::from_connection(conn).expect("migrated provider pauses");
        let open = ProviderPauseKey::new("open", "same-id");
        let shut = ProviderPauseKey::new("shut", "same-id");
        pauses.register(&open, "analyst", None).await.unwrap();
        pauses.register(&shut, "analyst", None).await.unwrap();
        pauses.release(&open).await.unwrap();

        assert!(!pauses.is_holding(&open).await.unwrap());
        assert!(pauses.is_holding(&shut).await.unwrap());
    }

    #[tokio::test]
    async fn liveness_reconciliation_reports_but_does_not_fence_a_stale_provider_pause() {
        let pauses = store().await;
        let key = key();
        pauses.register(&key, "analyst", None).await.unwrap();

        assert_eq!(
            pauses.list_unresponsive_before(i64::MAX).await.unwrap(),
            vec![key.clone()]
        );
        assert!(!pauses.is_exited(&key).await.unwrap());
        assert_eq!(
            pauses
                .register(&key, "late", None)
                .await
                .unwrap()
                .disposition,
            ProviderPauseDisposition::Hold,
            "host suspension cannot turn a recoverable pause into an exit fence"
        );
    }

    #[tokio::test]
    async fn acknowledging_a_delivery_renews_liveness_until_other_waiters_finish() {
        let pauses = store().await;
        let key = key();
        pauses.register(&key, "first", None).await.unwrap();
        pauses.register(&key, "second", None).await.unwrap();
        pauses.release(&key).await.unwrap();

        // Model a long dashboard restart immediately before this child retries its lost ACK.
        // Another waiter still needs the generation, so the acknowledgement leaves this run row
        // in place for liveness reconciliation to inspect.
        {
            let conn = pauses
                .conn
                .lock()
                .expect("provider pause store mutex poisoned");
            conn.execute(
                "UPDATE provider_pause_runs SET last_seen_ms = 0
                 WHERE project = ?1 AND run_id = ?2",
                params![key.project(), key.run_id()],
            )
            .unwrap();
        }

        pauses.acknowledge(&key, 1, "first", "").await.unwrap();

        assert!(
            pauses.list_unresponsive_before(1).await.unwrap().is_empty(),
            "an acknowledgement retry proves the child remains alive while delivery completes"
        );
        assert!(!pauses.is_exited(&key).await.unwrap());
    }

    #[tokio::test]
    async fn an_acknowledged_delivery_is_not_reported_as_unresponsive() {
        let pauses = store().await;
        let key = key();
        pauses.register(&key, "analyst", None).await.unwrap();
        pauses.release(&key).await.unwrap();
        pauses.acknowledge(&key, 1, "analyst", "").await.unwrap();

        {
            let conn = pauses
                .conn
                .lock()
                .expect("provider pause store mutex poisoned");
            conn.execute(
                "UPDATE provider_pause_runs SET last_seen_ms = 0
                 WHERE project = ?1 AND run_id = ?2",
                params![key.project(), key.run_id()],
            )
            .unwrap();
        }

        assert!(
            pauses.list_unresponsive_before(1).await.unwrap().is_empty(),
            "a normally resumed child has no outstanding provider delivery to warn about"
        );
    }

    #[tokio::test]
    async fn an_unacknowledged_resume_survives_reopening_the_instance_database() {
        let path = std::env::temp_dir().join(format!(
            "ratatoskr-provider-pause-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let key = key();
        let before_restart = ProviderPauseStore::open(&path).expect("provider pause store");
        before_restart
            .register(&key, "analyst", None)
            .await
            .unwrap();
        before_restart.release(&key).await.unwrap();
        drop(before_restart);

        let after_restart = ProviderPauseStore::open(&path).expect("reopened provider pause store");
        assert_eq!(
            after_restart
                .register(&key, "analyst", None)
                .await
                .unwrap()
                .disposition,
            ProviderPauseDisposition::Continue
        );
        after_restart
            .acknowledge(&key, 1, "analyst", "")
            .await
            .unwrap();
        drop(after_restart);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn a_node_stop_survives_restart_and_overrides_a_lost_resume() {
        let path = std::env::temp_dir().join(format!(
            "ratatoskr-provider-pause-stop-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let key = key();
        let before_restart = ProviderPauseStore::open(&path).expect("provider pause store");
        before_restart
            .register_for_node(&key, "analyst-waiter", "analyst", None)
            .await
            .unwrap();
        before_restart
            .register_for_node(&key, "implementer-waiter", "implementer", None)
            .await
            .unwrap();
        before_restart.release(&key).await.unwrap();
        before_restart.stop(&key, "analyst").await.unwrap();
        drop(before_restart);

        let after_restart = ProviderPauseStore::open(&path).expect("reopened provider pause store");
        assert_eq!(
            after_restart
                .register_for_node(&key, "analyst-waiter", "analyst", None)
                .await
                .unwrap()
                .disposition,
            ProviderPauseDisposition::Stop,
            "Stop is delivered after a dashboard restart instead of exposing the lost Continue"
        );
        assert_eq!(
            after_restart
                .register_for_node(&key, "implementer-waiter", "implementer", None)
                .await
                .unwrap()
                .disposition,
            ProviderPauseDisposition::Continue,
            "a node-specific Stop does not terminate another provider waiter"
        );

        after_restart.clear_stop(&key, "analyst").await.unwrap();
        assert_eq!(
            after_restart
                .register_for_node(&key, "analyst-waiter", "analyst", None)
                .await
                .unwrap()
                .disposition,
            ProviderPauseDisposition::Continue,
            "Start clears the durable Stop without re-arming a provider pause"
        );
        drop(after_restart);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn a_stopped_node_remains_stopped_after_acknowledging_its_delivery() {
        let pauses = store().await;
        let key = key();
        pauses
            .register_for_node(&key, "redteam-waiter", "redteam", None)
            .await
            .unwrap();
        pauses.stop(&key, "red_team").await.unwrap();
        pauses
            .acknowledge(&key, 1, "redteam-waiter", "redteam")
            .await
            .unwrap();

        assert!(
            pauses.is_node_stopped(&key, "redteam").await.unwrap(),
            "acknowledging Stop delivers it; it does not restart the parked node"
        );
        pauses.clear_stop(&key, "red_team").await.unwrap();
        assert!(
            !pauses.is_node_stopped(&key, "redteam").await.unwrap(),
            "the control-name aliases share one durable Stop"
        );
    }
}
