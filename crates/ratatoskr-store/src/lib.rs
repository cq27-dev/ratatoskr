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
        conn.execute_batch("PRAGMA journal_mode = WAL;")?;
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
                "INSERT INTO runs (run_id, issue_id, status, updated_at)
                 VALUES (?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                 ON CONFLICT(run_id) DO UPDATE SET
                     issue_id = excluded.issue_id,
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
        let conn = Arc::clone(&self.conn);
        let run_id = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            let status = conn
                .query_row(
                    "SELECT status FROM runs WHERE run_id = ?1",
                    params![run_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            Ok::<_, StoreError>(status)
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
