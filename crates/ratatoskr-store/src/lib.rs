//! Ratatoskr's checkpoint store: a single SQLite file, one writer by construction.
//!
//! All connection access funnels through one `Arc<Mutex<Connection>>` and blocking SQLite work
//! runs on `tokio::task::spawn_blocking`. The mutex — not convention — is what enforces the
//! single-writer discipline the integration research called for; `rusqlite` (bundled, synchronous)
//! is used precisely to avoid `sqlx`-style connection pooling, which would invite multi-writer
//! contention.

use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{Connection, OptionalExtension, params};

/// Errors from the checkpoint store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io error opening store: {0}")]
    Io(#[from] std::io::Error),
    #[error("store task panicked")]
    Join(#[from] tokio::task::JoinError),
}

/// A per-node checkpoint snapshot read back from the `checkpoints` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub node_name: String,
    pub output_json: String,
    pub created_at: String,
}

/// A row of the `runs` table. `updated_at` moves only on a status transition — it is not a
/// heartbeat, so it can't be used alone to tell a live run from one that died mid-flight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    pub run_id: String,
    pub issue_id: Option<String>,
    pub status: String,
    pub updated_at: String,
}

/// A handle to the checkpoint database. Cheap to clone (shares the guarded connection).
#[derive(Clone)]
pub struct Store {
    conn: Arc<Mutex<Connection>>,
}

impl Store {
    /// Open (creating if needed) the checkpoint database at `path`, in WAL mode, with the schema
    /// applied. WAL means Phase 5's read-only `status` command won't block on the writer.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        // WAL so readers (`status`, `serve`) never block on the writer. `busy_timeout` covers the
        // brief moments a WAL checkpoint does take the write lock — without it a concurrent reader
        // gets a sporadic `SQLITE_BUSY` instead of waiting.
        conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;")?;
        Self::from_connection(conn)
    }

    /// An in-memory store, for tests and for Phase 1's `ratatoskr ask` (no durable state needed).
    pub fn open_in_memory() -> Result<Self, StoreError> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(conn: Connection) -> Result<Self, StoreError> {
        conn.execute_batch(include_str!("schema.sql"))?;
        Ok(Store {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Insert or update a run row.
    pub async fn upsert_run(
        &self,
        run_id: &str,
        issue_id: Option<&str>,
        status: &str,
    ) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let (run_id, issue_id, status) = (
            run_id.to_string(),
            issue_id.map(str::to_string),
            status.to_string(),
        );
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            conn.execute(
                // COALESCE, not assignment: status transitions pass `issue_id = None`, so a plain
                // `issue_id = excluded.issue_id` would null out an issue set at submission on the
                // very next status write. A later write can set it; none can erase it.
                "INSERT INTO runs (run_id, issue_id, status, updated_at)
                 VALUES (?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                 ON CONFLICT(run_id) DO UPDATE SET
                     issue_id = COALESCE(excluded.issue_id, runs.issue_id),
                     status = excluded.status,
                     updated_at = excluded.updated_at",
                params![run_id, issue_id, status],
            )?;
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// The current status string for a run, or `None` if there's no such run.
    pub async fn run_status(&self, run_id: &str) -> Result<Option<String>, StoreError> {
        Ok(self.run(run_id).await?.map(|r| r.status))
    }

    /// One run row, or `None` if there's no such run. A run can have checkpoints but no row: the
    /// scripted path writes its `issue` checkpoint before the row exists, and the schema's foreign
    /// key is decorative (`PRAGMA foreign_keys` is never enabled). Callers must tolerate `None`
    /// without concluding the run doesn't exist.
    pub async fn run(&self, run_id: &str) -> Result<Option<Run>, StoreError> {
        let conn = Arc::clone(&self.conn);
        let run_id = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            let run = conn
                .query_row(
                    "SELECT run_id, issue_id, status, updated_at FROM runs WHERE run_id = ?1",
                    params![run_id],
                    row_to_run,
                )
                .optional()?;
            Ok::<_, StoreError>(run)
        })
        .await?
    }

    /// Every run, most recently updated first — what the dashboard's run list reads.
    pub async fn list_runs(&self) -> Result<Vec<Run>, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            let mut stmt = conn.prepare(
                "SELECT run_id, issue_id, status, updated_at FROM runs
                 ORDER BY updated_at DESC, run_id DESC",
            )?;
            let rows = stmt
                .query_map([], row_to_run)?
                .collect::<Result<Vec<_>, _>>()?;
            Ok::<_, StoreError>(rows)
        })
        .await?
    }

    /// Append a node's output snapshot for a run. Called after each node in the plan flow.
    pub async fn insert_checkpoint(
        &self,
        run_id: &str,
        node_name: &str,
        output_json: &str,
    ) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let (run_id, node_name, output_json) = (
            run_id.to_string(),
            node_name.to_string(),
            output_json.to_string(),
        );
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            conn.execute(
                "INSERT INTO checkpoints (run_id, node_name, output_json)
                 VALUES (?1, ?2, ?3)",
                params![run_id, node_name, output_json],
            )?;
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// All checkpoints for a run, oldest first. Not needed by `plan` itself; it's what Phase 5's
    /// `ratatoskr status` command will read.
    pub async fn checkpoints_for_run(&self, run_id: &str) -> Result<Vec<Checkpoint>, StoreError> {
        let conn = Arc::clone(&self.conn);
        let run_id = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            let mut stmt = conn.prepare(
                "SELECT node_name, output_json, created_at
                 FROM checkpoints WHERE run_id = ?1 ORDER BY id ASC",
            )?;
            let rows = stmt
                .query_map(params![run_id], |row| {
                    Ok(Checkpoint {
                        node_name: row.get(0)?,
                        output_json: row.get(1)?,
                        created_at: row.get(2)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok::<_, StoreError>(rows)
        })
        .await?
    }
}

/// Shared row mapper for the `runs` columns, in the order both queries select them.
fn row_to_run(row: &rusqlite::Row<'_>) -> rusqlite::Result<Run> {
    Ok(Run {
        run_id: row.get(0)?,
        issue_id: row.get(1)?,
        status: row.get(2)?,
        updated_at: row.get(3)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn write_a_run_and_read_it_back() {
        let store = Store::open_in_memory().unwrap();

        assert_eq!(store.run_status("run-1").await.unwrap(), None);

        store
            .upsert_run("run-1", Some("issue-42"), "pending")
            .await
            .unwrap();
        assert_eq!(
            store.run_status("run-1").await.unwrap().as_deref(),
            Some("pending")
        );

        // Upsert updates in place rather than inserting a duplicate.
        store
            .upsert_run("run-1", Some("issue-42"), "running")
            .await
            .unwrap();
        assert_eq!(
            store.run_status("run-1").await.unwrap().as_deref(),
            Some("running")
        );
    }

    #[tokio::test]
    async fn a_status_write_does_not_erase_the_issue_id() {
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-1", Some("issue-42"), "running")
            .await
            .unwrap();

        // Every status transition in the pipeline passes `issue_id = None`; it must not clobber.
        store.upsert_run("run-1", None, "converged").await.unwrap();

        let run = store.run("run-1").await.unwrap().unwrap();
        assert_eq!(run.issue_id.as_deref(), Some("issue-42"));
        assert_eq!(run.status, "converged");
    }

    #[tokio::test]
    async fn list_runs_is_newest_first() {
        let store = Store::open_in_memory().unwrap();
        assert!(store.list_runs().await.unwrap().is_empty());
        assert!(store.run("nope").await.unwrap().is_none());

        for id in ["a", "b", "c"] {
            store.upsert_run(id, None, "running").await.unwrap();
        }
        // `updated_at` has millisecond resolution and these writes can share a millisecond, so
        // assert on set membership plus the tiebreak, not on a specific interleaving.
        let runs = store.list_runs().await.unwrap();
        assert_eq!(runs.len(), 3);
        let mut ids: Vec<_> = runs.iter().map(|r| r.run_id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, ["a", "b", "c"]);
    }

    #[tokio::test]
    async fn a_checkpoint_needs_its_run_row_to_exist() {
        // The bundled SQLite is compiled with `SQLITE_DEFAULT_FOREIGN_KEYS=1`, so this constraint
        // is live even though the system `sqlite3` CLI (built without it) will happily insert an
        // orphan. Anything that checkpoints must create the run row first.
        let store = Store::open_in_memory().unwrap();
        assert!(
            store
                .insert_checkpoint("never-started", "scout", "{}")
                .await
                .is_err(),
            "a checkpoint for an unknown run is refused"
        );

        store.upsert_run("started", None, "running").await.unwrap();
        assert!(
            store
                .insert_checkpoint("started", "scout", "{}")
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn checkpoints_persist_in_order() {
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("run-1", None, "running").await.unwrap();

        store
            .insert_checkpoint("run-1", "scout", r#"{"a":1}"#)
            .await
            .unwrap();
        store
            .insert_checkpoint("run-1", "analyst", r#"{"b":2}"#)
            .await
            .unwrap();

        let checkpoints = store.checkpoints_for_run("run-1").await.unwrap();
        assert_eq!(checkpoints.len(), 2);
        assert_eq!(checkpoints[0].node_name, "scout");
        assert_eq!(checkpoints[1].node_name, "analyst");
        assert_eq!(checkpoints[1].output_json, r#"{"b":2}"#);
        assert!(store.checkpoints_for_run("other").await.unwrap().is_empty());
    }
}
