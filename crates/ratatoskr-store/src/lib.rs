//! Ratatoskr's checkpoint store: a single SQLite file, one writer by construction.
//!
//! All connection access funnels through one `Arc<Mutex<Connection>>` and blocking SQLite work
//! runs on `tokio::task::spawn_blocking`. The mutex — not convention — is what enforces the
//! single-writer discipline the integration research called for; `rusqlite` (bundled, synchronous)
//! is used precisely to avoid `sqlx`-style connection pooling, which would invite multi-writer
//! contention.

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ratatoskr_core::NodeTelemetry;
use rusqlite::{Connection, OptionalExtension, params};

/// Columns added to the two tables after the first schema shipped, as `(table, column, decl)`.
///
/// `CREATE TABLE IF NOT EXISTS` is a no-op on a database that already has the table, so a store
/// created before these columns existed would silently keep the narrow shape. Every entry must be
/// nullable and have no default beyond `NULL`: SQLite rewrites nothing on `ADD COLUMN`, so this
/// stays an O(1) metadata change no matter how many runs are already recorded.
const ADDED_COLUMNS: &[(&str, &str, &str)] = &[
    ("runs", "config_json", "TEXT"),
    ("runs", "graph_hash", "TEXT"),
    ("runs", "repo_sha", "TEXT"),
    ("checkpoints", "input_json", "TEXT"),
    ("checkpoints", "model", "TEXT"),
    ("checkpoints", "iteration", "INTEGER"),
    ("checkpoints", "duration_ms", "INTEGER"),
    ("checkpoints", "turns", "INTEGER"),
    ("checkpoints", "input_tokens", "INTEGER"),
    ("checkpoints", "output_tokens", "INTEGER"),
    ("checkpoints", "cached_input_tokens", "INTEGER"),
    ("checkpoints", "cache_creation_input_tokens", "INTEGER"),
    ("checkpoints", "reasoning_tokens", "INTEGER"),
    ("checkpoints", "tools_json", "TEXT"),
    ("checkpoints", "reuses_session", "INTEGER"),
    ("checkpoints", "thinking", "INTEGER"),
    ("checkpoints", "tools_used_json", "TEXT"),
    ("checkpoints", "error", "TEXT"),
];

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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Checkpoint {
    pub node_name: String,
    pub output_json: String,
    pub created_at: String,
    /// What the node was given. `None` for rows written before the column existed.
    pub input_json: Option<String>,
    /// Which pass of the converge loop this row came from; `None` for a node that runs once.
    pub iteration: Option<u32>,
    pub telemetry: NodeTelemetry,
}

/// One checkpoint write: the node's identity, what it was given, what it produced, and what that
/// cost. A parameter struct because a positional list of one required and eleven optional values —
/// most of them same-typed — is one a caller can silently transpose.
#[derive(Debug, Clone, Default)]
pub struct CheckpointWrite<'a> {
    pub run_id: &'a str,
    pub node_name: &'a str,
    pub output_json: &'a str,
    /// What the node was given, serialized. Recording it is what makes a run replayable: without
    /// the input, a checkpoint shows what came out and gives no way to ask why.
    pub input_json: Option<&'a str>,
    pub iteration: Option<u32>,
    pub telemetry: NodeTelemetry,
}

/// A row of the `runs` table. `updated_at` moves only on a status transition — it is not a
/// heartbeat, so it can't be used alone to tell a live run from one that died mid-flight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    pub run_id: String,
    pub issue_id: Option<String>,
    pub status: String,
    pub updated_at: String,
    /// The resolved config the run started under, serialized.
    pub config_json: Option<String>,
    /// Identifies the graph that ran — the orchestration script and the rulesets that shaped it.
    /// Two runs with different hashes are not comparable however alike their configs look.
    pub graph_hash: Option<String>,
    /// The commit the run started from.
    pub repo_sha: Option<String>,
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
        migrate(&conn)?;
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
                    "SELECT run_id, issue_id, status, updated_at, config_json, graph_hash, repo_sha FROM runs WHERE run_id = ?1",
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
                "SELECT run_id, issue_id, status, updated_at, config_json, graph_hash, repo_sha FROM runs
                 ORDER BY updated_at DESC, run_id DESC",
            )?;
            let rows = stmt
                .query_map([], row_to_run)?
                .collect::<Result<Vec<_>, _>>()?;
            Ok::<_, StoreError>(rows)
        })
        .await?
    }

    /// Record the provenance of a run: the config it started under, the graph that ran, and the
    /// commit it ran against. Written once at run start, separately from [`Store::upsert_run`] —
    /// that one fires on every status transition, and this is not something a transition knows.
    pub async fn record_run_provenance(
        &self,
        run_id: &str,
        config_json: Option<&str>,
        graph_hash: Option<&str>,
        repo_sha: Option<&str>,
    ) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let (run_id, config_json, graph_hash, repo_sha) = (
            run_id.to_string(),
            config_json.map(str::to_string),
            graph_hash.map(str::to_string),
            repo_sha.map(str::to_string),
        );
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            // COALESCE on the stored side, for the same reason `upsert_run` uses it on the incoming
            // side: provenance is written once, and a later call that knows less must not erase it.
            conn.execute(
                "UPDATE runs SET
                     config_json = COALESCE(config_json, ?2),
                     graph_hash  = COALESCE(graph_hash, ?3),
                     repo_sha    = COALESCE(repo_sha, ?4)
                 WHERE run_id = ?1",
                params![run_id, config_json, graph_hash, repo_sha],
            )?;
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// Append a node's checkpoint for a run. Called after each node in the plan flow, including
    /// after one that failed — `telemetry.error` is why, and a failed node's cost was still billed.
    pub async fn insert_checkpoint(&self, write: CheckpointWrite<'_>) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let CheckpointWrite {
            run_id,
            node_name,
            output_json,
            input_json,
            iteration,
            telemetry,
        } = write;
        let (run_id, node_name, output_json, input_json) = (
            run_id.to_string(),
            node_name.to_string(),
            output_json.to_string(),
            input_json.map(str::to_string),
        );
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            let usage = telemetry.usage;
            conn.execute(
                "INSERT INTO checkpoints (
                     run_id, node_name, output_json, input_json, iteration,
                     model, duration_ms, turns, error,
                     input_tokens, output_tokens, cached_input_tokens,
                     cache_creation_input_tokens, reasoning_tokens, tools_json, reuses_session,
                     thinking, tools_used_json
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                           ?17, ?18)",
                params![
                    run_id,
                    node_name,
                    output_json,
                    input_json,
                    iteration,
                    telemetry.model,
                    telemetry.duration_ms,
                    telemetry.turns,
                    telemetry.error,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cached_input_tokens,
                    usage.cache_creation_input_tokens,
                    usage.reasoning_tokens,
                    serde_json::to_string(&telemetry.tools).unwrap_or_default(),
                    telemetry.reuses_session,
                    telemetry.thinking,
                    serde_json::to_string(&telemetry.tools_used).unwrap_or_default(),
                ],
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
                "SELECT node_name, output_json, created_at, input_json, iteration,
                        model, duration_ms, turns, error,
                        input_tokens, output_tokens, cached_input_tokens,
                        cache_creation_input_tokens, reasoning_tokens, tools_json, reuses_session,
                        thinking, tools_used_json
                 FROM checkpoints WHERE run_id = ?1 ORDER BY id ASC",
            )?;
            let rows = stmt
                .query_map(params![run_id], |row| {
                    Ok(Checkpoint {
                        node_name: row.get(0)?,
                        output_json: row.get(1)?,
                        created_at: row.get(2)?,
                        input_json: row.get(3)?,
                        iteration: row.get(4)?,
                        telemetry: NodeTelemetry {
                            model: row.get(5)?,
                            duration_ms: row.get(6)?,
                            turns: row.get(7)?,
                            error: row.get(8)?,
                            // A pre-migration row has NULL for both: an empty tool list and a
                            // fresh session are the honest reading of "not recorded".
                            tools: row
                                .get::<_, Option<String>>(14)?
                                .and_then(|j| serde_json::from_str(&j).ok())
                                .unwrap_or_default(),
                            reuses_session: row.get::<_, Option<bool>>(15)?.unwrap_or(false),
                            thinking: row.get::<_, Option<bool>>(16)?.unwrap_or(false),
                            tools_used: row
                                .get::<_, Option<String>>(17)?
                                .and_then(|j| serde_json::from_str(&j).ok())
                                .unwrap_or_default(),
                            usage: ratatoskr_core::TokenUsage {
                                // A pre-migration row has NULL here, which is "unknown", not zero.
                                // Reading it as 0 is the honest projection for a sum; the `model`
                                // column being NULL is what distinguishes the two cases.
                                input_tokens: row.get::<_, Option<u64>>(9)?.unwrap_or(0),
                                output_tokens: row.get::<_, Option<u64>>(10)?.unwrap_or(0),
                                cached_input_tokens: row.get::<_, Option<u64>>(11)?.unwrap_or(0),
                                cache_creation_input_tokens: row
                                    .get::<_, Option<u64>>(12)?
                                    .unwrap_or(0),
                                reasoning_tokens: row.get::<_, Option<u64>>(13)?.unwrap_or(0),
                            },
                        },
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
        config_json: row.get(4)?,
        graph_hash: row.get(5)?,
        repo_sha: row.get(6)?,
    })
}

/// Add any column in [`ADDED_COLUMNS`] the database does not already have.
///
/// This is the whole migration story, and it works because every added column is nullable: SQLite's
/// `ADD COLUMN` only writes the schema record, so this costs the same on a store with one run as on
/// one with ten thousand. Adding a *non*-nullable column, or one needing a backfill, would need a
/// real versioned migration — do that rather than stretching this.
fn migrate(conn: &Connection) -> Result<(), StoreError> {
    for table in ["runs", "checkpoints"] {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let existing: HashSet<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<_, _>>()?;
        for (_, column, decl) in ADDED_COLUMNS.iter().filter(|(t, ..)| *t == table) {
            if !existing.contains(*column) {
                conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"))?;
            }
        }
    }
    Ok(())
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
                .insert_checkpoint(CheckpointWrite {
                    run_id: "never-started",
                    node_name: "scout",
                    output_json: "{}",
                    ..Default::default()
                })
                .await
                .is_err(),
            "a checkpoint for an unknown run is refused"
        );

        store.upsert_run("started", None, "running").await.unwrap();
        assert!(
            store
                .insert_checkpoint(CheckpointWrite {
                    run_id: "started",
                    node_name: "scout",
                    output_json: "{}",
                    ..Default::default()
                })
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn checkpoints_persist_in_order() {
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("run-1", None, "running").await.unwrap();

        for (node, output) in [("scout", r#"{"a":1}"#), ("analyst", r#"{"b":2}"#)] {
            store
                .insert_checkpoint(CheckpointWrite {
                    run_id: "run-1",
                    node_name: node,
                    output_json: output,
                    ..Default::default()
                })
                .await
                .unwrap();
        }

        let checkpoints = store.checkpoints_for_run("run-1").await.unwrap();
        assert_eq!(checkpoints.len(), 2);
        assert_eq!(checkpoints[0].node_name, "scout");
        assert_eq!(checkpoints[1].node_name, "analyst");
        assert_eq!(checkpoints[1].output_json, r#"{"b":2}"#);
        assert!(store.checkpoints_for_run("other").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_checkpoint_carries_its_input_cost_and_model() {
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("run-1", None, "running").await.unwrap();

        store
            .insert_checkpoint(CheckpointWrite {
                run_id: "run-1",
                node_name: "analyst",
                output_json: r#"{"out":1}"#,
                input_json: Some(r#"{"issue":"issue-6"}"#),
                iteration: Some(2),
                telemetry: NodeTelemetry {
                    model: Some("anthropic/claude-opus-4".into()),
                    duration_ms: Some(4200),
                    turns: Some(7),
                    error: None,
                    usage: ratatoskr_core::TokenUsage {
                        input_tokens: 1000,
                        output_tokens: 250,
                        cached_input_tokens: 800,
                        cache_creation_input_tokens: 200,
                        reasoning_tokens: 4_000,
                    },
                    tools: vec!["Read".to_string(), "semantic_search".to_string()],
                    tools_used: vec!["Read".to_string()],
                    reuses_session: true,
                    thinking: true,
                },
            })
            .await
            .unwrap();

        let cp = &store.checkpoints_for_run("run-1").await.unwrap()[0];
        assert_eq!(cp.input_json.as_deref(), Some(r#"{"issue":"issue-6"}"#));
        assert_eq!(cp.iteration, Some(2));
        assert_eq!(
            cp.telemetry.model.as_deref(),
            Some("anthropic/claude-opus-4")
        );
        assert_eq!(cp.telemetry.duration_ms, Some(4200));
        assert_eq!(cp.telemetry.turns, Some(7));
        assert_eq!(cp.telemetry.usage.input_tokens, 1000);
        assert_eq!(cp.telemetry.usage.cached_input_tokens, 800);
        assert_eq!(cp.telemetry.usage.cache_creation_input_tokens, 200);
        // Billed as output and reported apart from it: a node that thinks before every tool call
        // reads as nearly free when this is dropped, and it is most of what the node spent.
        assert_eq!(cp.telemetry.usage.reasoning_tokens, 4_000);
        // What the node could reach, and whether its memory carried over — neither is
        // reconstructable later from a config that has since changed.
        assert_eq!(cp.telemetry.tools, ["Read", "semantic_search"]);
        // Given two, reached for one — the gap is the point.
        assert_eq!(cp.telemetry.tools_used, ["Read"]);
        assert!(cp.telemetry.reuses_session);
        assert!(cp.telemetry.thinking);
    }

    #[tokio::test]
    async fn a_failed_node_still_records_what_it_cost() {
        // The most useful row in a failed run is the one that says why, and the calls it made
        // before failing were billed like any other.
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("run-1", None, "running").await.unwrap();
        store
            .insert_checkpoint(CheckpointWrite {
                run_id: "run-1",
                node_name: "scout",
                output_json: "{}",
                telemetry: NodeTelemetry {
                    error: Some("output failed schema validation".into()),
                    usage: ratatoskr_core::TokenUsage {
                        input_tokens: 99,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .unwrap();

        let cp = &store.checkpoints_for_run("run-1").await.unwrap()[0];
        assert_eq!(
            cp.telemetry.error.as_deref(),
            Some("output failed schema validation")
        );
        assert_eq!(cp.telemetry.usage.input_tokens, 99);
    }

    #[tokio::test]
    async fn provenance_is_written_once_and_never_erased() {
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("run-1", None, "running").await.unwrap();
        store
            .record_run_provenance("run-1", Some("{}"), Some("deadbeef"), Some("abc123"))
            .await
            .unwrap();

        // Every status transition passes `issue_id = None`; none of them may drop provenance.
        store.upsert_run("run-1", None, "converged").await.unwrap();
        // And a later provenance write that knows less must not erase what the first one knew.
        store
            .record_run_provenance("run-1", None, None, None)
            .await
            .unwrap();

        let run = store.run("run-1").await.unwrap().unwrap();
        assert_eq!(run.config_json.as_deref(), Some("{}"));
        assert_eq!(run.graph_hash.as_deref(), Some("deadbeef"));
        assert_eq!(run.repo_sha.as_deref(), Some("abc123"));
        assert_eq!(run.status, "converged");
    }

    #[test]
    fn a_narrow_database_gains_the_columns_it_is_missing() {
        // The shape a store created before these columns existed has. `CREATE TABLE IF NOT EXISTS`
        // is a no-op against it, so without `migrate` every widened write would fail.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE runs (
                 run_id TEXT PRIMARY KEY, issue_id TEXT, status TEXT NOT NULL,
                 updated_at TEXT NOT NULL);
             CREATE TABLE checkpoints (
                 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
                 node_name TEXT NOT NULL, output_json TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')));
             INSERT INTO runs (run_id, status, updated_at) VALUES ('old', 'converged', 'then');
             INSERT INTO checkpoints (run_id, node_name, output_json)
                 VALUES ('old', 'scout', '{}');",
        )
        .unwrap();

        let store = Store::from_connection(conn).unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();

        // The pre-existing row survives, reading as "nothing was recorded" rather than failing.
        let cp = rt.block_on(store.checkpoints_for_run("old")).unwrap();
        assert_eq!(cp.len(), 1);
        assert_eq!(cp[0].telemetry.model, None);
        assert_eq!(cp[0].input_json, None);
        assert_eq!(cp[0].telemetry.usage.input_tokens, 0);
        assert_eq!(
            rt.block_on(store.run("old")).unwrap().unwrap().repo_sha,
            None
        );

        // And a widened write against the migrated table works.
        rt.block_on(store.insert_checkpoint(CheckpointWrite {
            run_id: "old",
            node_name: "analyst",
            output_json: "{}",
            iteration: Some(1),
            telemetry: NodeTelemetry {
                model: Some("anthropic/x".into()),
                ..Default::default()
            },
            ..Default::default()
        }))
        .unwrap();
        let cp = rt.block_on(store.checkpoints_for_run("old")).unwrap();
        assert_eq!(cp[1].telemetry.model.as_deref(), Some("anthropic/x"));
    }

    #[test]
    fn migrating_twice_is_a_no_op() {
        // `from_connection` runs it on every open, so it has to be safe to run against a database
        // that already has every column.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(include_str!("schema.sql")).unwrap();
        migrate(&conn).unwrap();
        migrate(&conn).unwrap();
    }
}
