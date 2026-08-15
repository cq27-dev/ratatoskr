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
    // Where an imported run came from. Null for one this machine produced, which is what makes
    // "only mine" answerable after somebody else's runs have been imported alongside.
    ("runs", "origin", "TEXT"),
    // The graph the run executed, so it can be drawn afterwards by something whose own pipeline
    // has changed — or that never had this one.
    ("runs", "shape_json", "TEXT"),
    ("runs", "config_json", "TEXT"),
    ("runs", "graph_hash", "TEXT"),
    ("runs", "repo_sha", "TEXT"),
    ("runs", "image_digest", "TEXT"),
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
    // Which execution wrote this row, and which execution invoked that one. Null for a row written
    // before an execution had an identity, and null in `parent_span_id` for a stage the run itself
    // drove — which is most of them, and not a gap.
    ("checkpoints", "span_id", "TEXT"),
    ("checkpoints", "parent_span_id", "TEXT"),
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
    #[error("run bundle: {0}")]
    Bundle(String),
    #[error("`{0}` names more than one run — use more of the id")]
    AmbiguousRun(String),
    #[error("no run `{0}` to record provenance against")]
    NoSuchRun(String),
    #[error(
        "this bundle is format version {found}; this build reads up to {}. Update ratatoskr to read it",
        crate::bundle::FORMAT_VERSION
    )]
    Unsupported { found: u32 },
    /// A bundle whose executions do not form a graph: an id naming two of them, or one invoked by
    /// itself.
    #[error("run {run_id} in this bundle has an unusable execution graph: {problem}")]
    BadExecutionGraph { run_id: String, problem: String },
}

pub mod auth;
pub mod bundle;
pub mod provider_pause;

/// A per-node checkpoint snapshot read back from the `checkpoints` table.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    pub node_name: String,
    pub output_json: String,
    pub created_at: String,
    /// What the node was given. `None` for rows written before the column existed.
    pub input_json: Option<String>,
    /// Which pass of the converge loop this row came from; `None` for a node that runs once.
    pub iteration: Option<u32>,
    /// Which execution wrote this row, and what invoked it.
    ///
    /// `None` for a row written before executions had identities. A row that HAS an identity and no
    /// parent is a stage the run itself drove — the ordinary case, and not the same statement.
    pub invocation: Option<ratatoskr_core::span::Invocation>,
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
    /// Which execution is writing this row, and what invoked it. `None` only where there is no
    /// execution to name — an import replaying a row that was written without one.
    pub invocation: Option<ratatoskr_core::span::Invocation>,
    pub telemetry: NodeTelemetry,
}

/// A row of the `runs` table. `updated_at` moves only on a status transition — it is not a
/// heartbeat, so it can't be used alone to tell a live run from one that died mid-flight.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    /// Immutable OCI image ID used for this run's sandboxed work, when it had any.
    pub image_digest: Option<String>,
    /// Where this run came from, when it was not produced here. `None` for a local run.
    pub origin: Option<String>,
    /// The graph that ran, serialized. `None` for runs recorded before shapes were stored, which
    /// fall back to the reader's built-in.
    pub shape_json: Option<String>,
    /// What it is for, as recorded by whoever ran it. Empty unless tagged.
    pub tags: Vec<String>,
}

/// One event of a run's history: the raw log record, plus what it takes to order and filter them.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EventRow {
    pub seq: i64,
    pub at: String,
    pub kind: String,
    pub node: Option<String>,
    /// The log record verbatim. Kept whole so reading it back is the same parse as reading the log.
    pub payload_json: String,
}

/// Open a SQLite file the way every database in this crate wants it, creating its directory.
///
/// WAL so readers (`status`, `serve`) never block on the writer. `busy_timeout` covers the brief
/// moments a WAL checkpoint does take the write lock — without it a concurrent reader gets a
/// sporadic `SQLITE_BUSY` instead of waiting.
pub(crate) fn open_sqlite(path: &Path) -> Result<Connection, StoreError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path)?;
    conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;")?;
    Ok(conn)
}

/// The length of a run id in full: a hyphenated uuid. Anything shorter is treated as a prefix.
const UUID_LEN: usize = 36;

/// A handle to the checkpoint database. Cheap to clone (shares the guarded connection).
#[derive(Clone)]
pub struct Store {
    conn: Arc<Mutex<Connection>>,
}

impl Store {
    /// Open (creating if needed) the checkpoint database at `path`, in WAL mode, with the schema
    /// applied. WAL means Phase 5's read-only `status` command won't block on the writer.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::from_connection(open_sqlite(path.as_ref())?)
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
                    "SELECT run_id, issue_id, status, updated_at, config_json, graph_hash, repo_sha, image_digest, origin, shape_json FROM runs WHERE run_id = ?1",
                    params![run_id],
                    row_to_run,
                )
                .optional()?;
            Ok::<_, StoreError>(run)
        })
        .await?
    }

    /// The full run id a prefix names, the way git resolves a short hash.
    ///
    /// `Ok(None)` for a prefix nothing starts with; [`StoreError::AmbiguousRun`] when more than one
    /// does — never a silent pick, because the two runs a prefix could mean are usually the two
    /// you are trying to tell apart.
    ///
    /// A prefix scan on the primary key, so this is a range scan of an index rather than a table
    /// scan. `LIMIT 2` because one more than one is all the answer needs.
    pub async fn resolve_run(&self, prefix: &str) -> Result<Option<String>, StoreError> {
        // An exact id needs no resolving, and a full uuid is the common case — the dashboard
        // shortens for display but every internal caller has the whole thing.
        if prefix.len() >= UUID_LEN {
            return Ok(Some(prefix.to_string()));
        }
        // `LIKE` treats these as wildcards, so a prefix carrying one would match far too much.
        // Refused rather than escaped: no real run id contains either.
        if prefix.is_empty() || prefix.contains('%') || prefix.contains('_') {
            return Ok(None);
        }
        let conn = Arc::clone(&self.conn);
        let prefix = prefix.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            let mut stmt =
                conn.prepare("SELECT run_id FROM runs WHERE run_id LIKE ?1 || '%' LIMIT 2")?;
            let found = stmt
                .query_map(params![prefix], |row| row.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            match found.len() {
                0 => Ok(None),
                1 => Ok(Some(found.into_iter().next().expect("one row"))),
                _ => Err(StoreError::AmbiguousRun(prefix)),
            }
        })
        .await?
    }

    /// Every run, most recently updated first — what the dashboard's run list reads.
    pub async fn list_runs(&self) -> Result<Vec<Run>, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            let mut stmt = conn.prepare(
                "SELECT run_id, issue_id, status, updated_at, config_json, graph_hash, repo_sha, image_digest, origin, shape_json FROM runs
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
    /// commit it ran against. The image digest arrives when container-backed work first begins;
    /// each fact is still write-once. This is separate from [`Store::upsert_run`] — that one fires
    /// on every status transition, and this is not something a transition knows.
    pub async fn record_run_provenance(
        &self,
        run_id: &str,
        config_json: Option<&str>,
        graph_hash: Option<&str>,
        repo_sha: Option<&str>,
        shape_json: Option<&str>,
        image_digest: Option<&str>,
    ) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let (run_id, config_json, graph_hash, repo_sha, shape_json, image_digest) = (
            run_id.to_string(),
            config_json.map(str::to_string),
            graph_hash.map(str::to_string),
            repo_sha.map(str::to_string),
            shape_json.map(str::to_string),
            image_digest.map(str::to_string),
        );
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            // COALESCE on the stored side, for the same reason `upsert_run` uses it on the incoming
            // side: provenance is written once, and a later call that knows less must not erase it.
            let changed = conn.execute(
                "UPDATE runs SET
                     config_json = COALESCE(config_json, ?2),
                     graph_hash  = COALESCE(graph_hash, ?3),
                     repo_sha    = COALESCE(repo_sha, ?4),
                     shape_json  = COALESCE(shape_json, ?5),
                     image_digest = COALESCE(image_digest, ?6)
                 WHERE run_id = ?1",
                params![
                    run_id,
                    config_json,
                    graph_hash,
                    repo_sha,
                    shape_json,
                    image_digest
                ],
            )?;
            // `UPDATE ... WHERE run_id = ?` matches nothing and reports success when the run row is
            // absent, so the row count is the only thing separating provenance that landed from
            // provenance that was merely sent. Answered here rather than by each caller reading the
            // run back: there is no legitimate provenance for a run that does not exist, and a
            // caller that has to notice on its own is a caller that can forget to.
            match changed {
                0 => Err(StoreError::NoSuchRun(run_id)),
                _ => Ok(()),
            }
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
            invocation,
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
                     thinking, tools_used_json, span_id, parent_span_id
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                           ?17, ?18, ?19, ?20)",
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
                    // Written as the sixteen hex characters they are read back from, so the column
                    // holds what an exporter and a human both expect to see.
                    invocation.map(|i| i.span_id.to_string()),
                    invocation
                        .and_then(|i| i.parent_span_id)
                        .map(|p| p.to_string()),
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
                        thinking, tools_used_json, span_id, parent_span_id
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
                        // A row with no identity reports none. Nothing is invented: a reader that
                        // cannot tell two executions apart must be told so, not given a plausible
                        // answer.
                        //
                        // And a parent that is PRESENT but unreadable takes the identity down with
                        // it. Reading it as absent would turn a nested execution whose parentage
                        // cannot be recovered into a top-level one — a claim about the run's shape,
                        // made out of a value nobody could parse.
                        invocation: read_invocation(row.get(18)?, row.get(19)?),
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

    /// Store a run's event history. Returns how many rows were new.
    ///
    /// Idempotent: `(run_id, seq)` is the key, so re-ingesting a log that was already read adds
    /// nothing. That matters because the obvious way to use this is to run it again whenever you
    /// are unsure whether it ran.
    pub async fn ingest_events(
        &self,
        run_id: &str,
        events: Vec<EventRow>,
    ) -> Result<usize, StoreError> {
        let conn = Arc::clone(&self.conn);
        let run_id = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("store mutex poisoned");
            let tx = conn.transaction()?;
            let mut added = 0;
            {
                let mut stmt = tx.prepare(
                    "INSERT OR IGNORE INTO events (run_id, seq, at, kind, node, payload_json)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )?;
                for e in &events {
                    added +=
                        stmt.execute(params![run_id, e.seq, e.at, e.kind, e.node, e.payload_json])?;
                }
            }
            tx.commit()?;
            Ok::<_, StoreError>(added)
        })
        .await?
    }

    /// A run's stored history, in order. Empty for a run never ingested.
    pub async fn events_for_run(&self, run_id: &str) -> Result<Vec<EventRow>, StoreError> {
        let conn = Arc::clone(&self.conn);
        let run_id = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            let mut stmt = conn.prepare(
                "SELECT seq, at, kind, node, payload_json FROM events
                 WHERE run_id = ?1 ORDER BY seq",
            )?;
            let rows = stmt
                .query_map(params![run_id], |row| {
                    Ok(EventRow {
                        seq: row.get(0)?,
                        at: row.get(1)?,
                        kind: row.get(2)?,
                        node: row.get(3)?,
                        payload_json: row.get(4)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok::<_, StoreError>(rows)
        })
        .await?
    }

    /// Add tags to a run. Re-tagging with one it already has is a no-op.
    pub async fn tag_run(&self, run_id: &str, tags: Vec<String>) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let run_id = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            for tag in tags {
                conn.execute(
                    "INSERT OR IGNORE INTO run_tags (run_id, tag) VALUES (?1, ?2)",
                    params![run_id, tag.trim()],
                )?;
            }
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// Remove tags from a run. Removing one it does not have is a no-op.
    pub async fn untag_run(&self, run_id: &str, tags: Vec<String>) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let run_id = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            for tag in tags {
                conn.execute(
                    "DELETE FROM run_tags WHERE run_id = ?1 AND tag = ?2",
                    params![run_id, tag.trim()],
                )?;
            }
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// Fill in each run's tags. Separate from listing because most readers do not need them.
    pub async fn attach_tags(&self, runs: &mut [Run]) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let ids: Vec<String> = runs.iter().map(|r| r.run_id.clone()).collect();
        let found = tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            let mut stmt =
                conn.prepare("SELECT tag FROM run_tags WHERE run_id = ?1 ORDER BY tag")?;
            let mut out = Vec::with_capacity(ids.len());
            for id in ids {
                let tags = stmt
                    .query_map(params![id], |row| row.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                out.push(tags);
            }
            Ok::<_, StoreError>(out)
        })
        .await??;
        for (run, tags) in runs.iter_mut().zip(found) {
            run.tags = tags;
        }
        Ok(())
    }

    /// Delete a run and everything recorded about it.
    ///
    /// In one transaction and children first, because `checkpoints.run_id` is a real foreign key
    /// and this database enforces them — deleting the run row first fails rather than orphaning.
    pub async fn delete_run(&self, run_id: &str) -> Result<bool, StoreError> {
        let conn = Arc::clone(&self.conn);
        let run_id = run_id.to_string();
        tokio::task::spawn_blocking(move || {
            let mut conn = conn.lock().expect("store mutex poisoned");
            let tx = conn.transaction()?;
            tx.execute("DELETE FROM events WHERE run_id = ?1", params![run_id])?;
            tx.execute("DELETE FROM run_tags WHERE run_id = ?1", params![run_id])?;
            tx.execute("DELETE FROM checkpoints WHERE run_id = ?1", params![run_id])?;
            let gone = tx.execute("DELETE FROM runs WHERE run_id = ?1", params![run_id])?;
            tx.commit()?;
            Ok::<_, StoreError>(gone > 0)
        })
        .await?
    }

    /// Insert an imported run whole, preserving what it recorded rather than restating it.
    ///
    /// `upsert_run` writes the fields a live run knows and stamps `updated_at` as now; an import
    /// has to keep the run's own timestamps and provenance, or every imported run would claim to
    /// have finished at the moment it was imported.
    async fn insert_imported_run(&self, run: &Run, origin: &str) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let run = run.clone();
        let origin = origin.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            conn.execute(
                "INSERT INTO runs
                   (run_id, issue_id, status, updated_at, config_json, graph_hash, repo_sha, image_digest, origin, shape_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    run.run_id,
                    run.issue_id,
                    run.status,
                    run.updated_at,
                    run.config_json,
                    run.graph_hash,
                    run.repo_sha,
                    run.image_digest,
                    // The bundle's own origin wins when it has one: a run that has already been
                    // passed along keeps saying where it started, not who forwarded it.
                    run.origin.clone().unwrap_or(origin),
                    run.shape_json,
                ],
            )?;
            Ok::<_, StoreError>(())
        })
        .await?
    }

    /// The current time, as this database writes timestamps.
    ///
    /// Taken from SQLite rather than the process clock so an exported bundle's `exported_at` is
    /// the same kind of string, and in the same zone, as every timestamp beside it.
    pub async fn now(&self) -> Result<String, StoreError> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            let now =
                conn.query_row("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')", [], |row| {
                    row.get::<_, String>(0)
                })?;
            Ok::<_, StoreError>(now)
        })
        .await?
    }

    /// Record where a run came from. Set on import; never set for a run produced here.
    pub async fn set_origin(&self, run_id: &str, origin: &str) -> Result<(), StoreError> {
        let conn = Arc::clone(&self.conn);
        let run_id = run_id.to_string();
        let origin = origin.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().expect("store mutex poisoned");
            conn.execute(
                "UPDATE runs SET origin = ?2 WHERE run_id = ?1",
                params![run_id, origin],
            )?;
            Ok::<_, StoreError>(())
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
        image_digest: row.get(7)?,
        origin: row.get(8)?,
        shape_json: row.get(9)?,
        // Filled by the caller when it wants them: a join on every listing would cost every reader
        // for a column most of them do not look at.
        tags: Vec::new(),
    })
}

/// The execution a row names, from its two columns.
///
/// `None` unless the identity is readable and the parentage is unambiguous: absent, or present and
/// readable. A present-but-unreadable parent invalidates the whole thing, because the alternatives
/// are both worse — reporting it as a root asserts a shape the row does not carry, and reporting the
/// identity without the parent hides that something was lost.
fn read_invocation(
    span_id: Option<String>,
    parent_span_id: Option<String>,
) -> Option<ratatoskr_core::span::Invocation> {
    use ratatoskr_core::span::SpanId;
    let span_id = SpanId::parse(span_id.as_deref()?)?;
    let parent = match parent_span_id.as_deref() {
        None => None,
        Some(hex) => Some(SpanId::parse(hex)?),
    };
    Some(ratatoskr_core::span::Invocation {
        span_id,
        parent_span_id: parent,
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
    async fn a_prefix_resolves_to_one_run_the_way_a_short_hash_does() {
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("358e8441-fa9a-4ab4-bbbe-46a826455b20", None, "running")
            .await
            .unwrap();
        store
            .upsert_run("6402ccea-650f-4472-bff5-24e34466fe6d", None, "running")
            .await
            .unwrap();

        assert_eq!(
            store.resolve_run("358e8441").await.unwrap().as_deref(),
            Some("358e8441-fa9a-4ab4-bbbe-46a826455b20")
        );
        // A full id resolves to itself without touching the database.
        assert_eq!(
            store
                .resolve_run("6402ccea-650f-4472-bff5-24e34466fe6d")
                .await
                .unwrap()
                .as_deref(),
            Some("6402ccea-650f-4472-bff5-24e34466fe6d")
        );
        assert_eq!(store.resolve_run("deadbeef").await.unwrap(), None);
    }

    #[tokio::test]
    async fn an_ambiguous_prefix_is_an_error_rather_than_a_guess() {
        // The two runs a prefix could mean are usually the two you are trying to tell apart, so
        // picking one is the worst available answer.
        let store = Store::open_in_memory().unwrap();
        for id in [
            "abc11111-0000-0000-0000-000000000000",
            "abc22222-0000-0000-0000-000000000000",
        ] {
            store.upsert_run(id, None, "running").await.unwrap();
        }
        assert!(matches!(
            store.resolve_run("abc").await,
            Err(StoreError::AmbiguousRun(p)) if p == "abc"
        ));
        // One more character tells them apart.
        assert_eq!(
            store.resolve_run("abc1").await.unwrap().as_deref(),
            Some("abc11111-0000-0000-0000-000000000000")
        );
    }

    #[tokio::test]
    async fn a_prefix_carrying_a_like_wildcard_matches_nothing() {
        // `%` and `_` are wildcards to LIKE. Unescaped, `%` would match every run and resolve to
        // whichever two rows came back first — an ambiguity error at best, the wrong run at worst.
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("358e8441-fa9a-4ab4-bbbe-46a826455b20", None, "running")
            .await
            .unwrap();
        assert_eq!(store.resolve_run("%").await.unwrap(), None);
        assert_eq!(store.resolve_run("358e____").await.unwrap(), None);
        assert_eq!(store.resolve_run("").await.unwrap(), None);
    }

    #[tokio::test]
    async fn a_runs_history_survives_being_ingested_twice() {
        // Ingest is the obvious thing to re-run when unsure whether it ran, so running it again
        // must cost nothing rather than double a run's history.
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("r1", None, "running").await.unwrap();
        let events = vec![
            EventRow {
                seq: 0,
                at: "2026-08-07T10:00:00Z".into(),
                kind: "node_start".into(),
                node: Some("context".into()),
                payload_json: r#"{"kind":"node_start"}"#.into(),
            },
            EventRow {
                seq: 1,
                at: "2026-08-07T10:00:01Z".into(),
                kind: "tool_call".into(),
                node: Some("context".into()),
                payload_json: r#"{"kind":"tool_call"}"#.into(),
            },
        ];
        assert_eq!(store.ingest_events("r1", events.clone()).await.unwrap(), 2);
        assert_eq!(store.ingest_events("r1", events).await.unwrap(), 0);
        let back = store.events_for_run("r1").await.unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].kind, "node_start");
        assert_eq!(back[1].seq, 1);
    }

    #[tokio::test]
    async fn deleting_a_run_takes_everything_recorded_about_it() {
        // `checkpoints.run_id` is an enforced foreign key, so anything left behind would either
        // block the delete or outlive the run it describes.
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("r1", None, "converged").await.unwrap();
        store
            .insert_checkpoint(CheckpointWrite {
                run_id: "r1",
                node_name: "analyst",
                output_json: "{}",
                ..Default::default()
            })
            .await
            .unwrap();
        store
            .ingest_events(
                "r1",
                vec![EventRow {
                    seq: 0,
                    at: "t".into(),
                    kind: "checkpoint".into(),
                    node: Some("analyst".into()),
                    payload_json: "{}".into(),
                }],
            )
            .await
            .unwrap();
        store.tag_run("r1", vec!["baseline".into()]).await.unwrap();

        assert!(store.delete_run("r1").await.unwrap());
        assert!(store.run("r1").await.unwrap().is_none());
        assert!(store.events_for_run("r1").await.unwrap().is_empty());
        assert!(store.checkpoints_for_run("r1").await.unwrap().is_empty());
        // Deleting one that is already gone is not an error, so a prune can be re-run.
        assert!(!store.delete_run("r1").await.unwrap());
    }

    #[tokio::test]
    async fn tags_are_a_set_and_travel_with_the_run() {
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("r1", None, "converged").await.unwrap();
        store
            .tag_run(
                "r1",
                vec!["arm-a".into(), "baseline".into(), "arm-a".into()],
            )
            .await
            .unwrap();
        store.tag_run("r1", vec!["arm-a".into()]).await.unwrap();

        let mut runs = store.list_runs().await.unwrap();
        store.attach_tags(&mut runs).await.unwrap();
        assert_eq!(runs[0].tags, ["arm-a", "baseline"]);

        store.untag_run("r1", vec!["arm-a".into()]).await.unwrap();
        let mut runs = store.list_runs().await.unwrap();
        store.attach_tags(&mut runs).await.unwrap();
        assert_eq!(runs[0].tags, ["baseline"]);
    }

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
    async fn a_record_whose_parent_cannot_be_read_is_not_promoted_to_a_root() {
        // Present-but-unreadable and absent are different states, and the difference is the run's
        // shape: absent means the run drove this execution, so reading a parent nobody can parse as
        // absent asserts a top-level execution out of a value that was lost. The identity goes with
        // it — reporting one without its parentage hides that anything was missing.
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("run-1", None, "running").await.unwrap();
        store
            .insert_checkpoint(CheckpointWrite {
                run_id: "run-1",
                node_name: "referee",
                output_json: "{}",
                invocation: Some(ratatoskr_core::span::Invocation::root(
                    ratatoskr_core::span::SpanId::parse("00000000000000a1").unwrap(),
                )),
                ..Default::default()
            })
            .await
            .unwrap();
        // Corrupt the parent the way a hand-edited store or a half-written value would.
        {
            let conn = store.conn.lock().expect("store mutex poisoned");
            conn.execute(
                "UPDATE checkpoints SET parent_span_id = 'not-a-span' WHERE node_name = 'referee'",
                [],
            )
            .unwrap();
        }

        let rows = store.checkpoints_for_run("run-1").await.unwrap();
        assert_eq!(
            rows[0].invocation, None,
            "an unreadable parent invalidates the identity rather than becoming no parent at all"
        );
    }

    #[tokio::test]
    async fn a_checkpoint_says_which_execution_wrote_it_and_what_invoked_that_one() {
        // A name is not an execution: a stage is invoked once per converge pass, and may be invoked
        // concurrently, so a reader with only names cannot tell two live invocations apart or say
        // what a nested one belongs to. The row has to carry the identity, and carry the absence of
        // one as an absence.
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("run-1", None, "running").await.unwrap();
        let parent = ratatoskr_core::span::SpanId::parse("00000000000000a1").unwrap();
        let child = ratatoskr_core::span::SpanId::parse("fedcba9876543210").unwrap();

        // A stage the run drove: an identity of its own, no parent.
        store
            .insert_checkpoint(CheckpointWrite {
                run_id: "run-1",
                node_name: "implementer",
                output_json: "{}",
                invocation: Some(ratatoskr_core::span::Invocation::root(parent)),
                ..Default::default()
            })
            .await
            .unwrap();
        // An execution invoked from inside it.
        store
            .insert_checkpoint(CheckpointWrite {
                run_id: "run-1",
                node_name: "referee",
                output_json: "{}",
                invocation: Some(ratatoskr_core::span::Invocation::root(parent).child(child)),
                ..Default::default()
            })
            .await
            .unwrap();
        // And a row written with no execution to name, which must not read as one.
        store
            .insert_checkpoint(CheckpointWrite {
                run_id: "run-1",
                node_name: "issue",
                output_json: "{}",
                ..Default::default()
            })
            .await
            .unwrap();

        let rows = store.checkpoints_for_run("run-1").await.unwrap();
        let of = |name: &str| {
            rows.iter()
                .find(|c| c.node_name == name)
                .unwrap()
                .invocation
        };
        assert_eq!(of("implementer").unwrap().span_id, parent);
        assert_eq!(of("implementer").unwrap().parent_span_id, None);
        assert_eq!(of("referee").unwrap().span_id, child);
        assert_eq!(
            of("referee").unwrap().parent_span_id,
            Some(parent),
            "a nested execution names the one that invoked it, not its own name"
        );
        assert_eq!(of("issue"), None, "no identity is not the invalid identity");
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
                invocation: Some(
                    ratatoskr_core::span::Invocation::root(
                        ratatoskr_core::span::SpanId::parse("00000000000000a1").unwrap(),
                    )
                    .child(ratatoskr_core::span::SpanId::parse("00000000000000b2").unwrap()),
                ),
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
            .record_run_provenance(
                "run-1",
                Some("{}"),
                Some("deadbeef"),
                Some("abc123"),
                None,
                None,
            )
            .await
            .unwrap();

        // Every status transition passes `issue_id = None`; none of them may drop provenance.
        store.upsert_run("run-1", None, "converged").await.unwrap();
        // And a later provenance write that knows less must not erase what the first one knew.
        store
            .record_run_provenance("run-1", None, None, None, None, None)
            .await
            .unwrap();

        let run = store.run("run-1").await.unwrap().unwrap();
        assert_eq!(run.config_json.as_deref(), Some("{}"));
        assert_eq!(run.graph_hash.as_deref(), Some("deadbeef"));
        assert_eq!(run.repo_sha.as_deref(), Some("abc123"));
        assert_eq!(run.status, "converged");
    }

    #[tokio::test]
    async fn a_runs_image_digest_is_recorded_with_the_rest_of_its_provenance() {
        // A container-backed run pins its execution environment by digest, and that pin is part
        // of the provenance every reader sees — the single-row read and the listing alike.
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("run-1", None, "running").await.unwrap();
        store
            .record_run_provenance(
                "run-1",
                Some("{}"),
                Some("deadbeef"),
                Some("abc123"),
                None,
                Some("sha256:aaa"),
            )
            .await
            .unwrap();

        let run = store.run("run-1").await.unwrap().unwrap();
        assert_eq!(run.image_digest.as_deref(), Some("sha256:aaa"));
        assert_eq!(run.config_json.as_deref(), Some("{}"));
        assert_eq!(run.graph_hash.as_deref(), Some("deadbeef"));
        let listed = store.list_runs().await.unwrap();
        assert_eq!(listed[0].image_digest.as_deref(), Some("sha256:aaa"));
    }

    #[tokio::test]
    async fn an_image_digest_cannot_be_erased_or_overwritten_once_recorded() {
        // One run is one immutable execution environment. A later provenance write that knows
        // less must not erase the digest, and a later resolution that disagrees must not move
        // it — either would record provenance for an image that did not run.
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("run-1", None, "running").await.unwrap();
        store
            .record_run_provenance("run-1", None, None, None, None, Some("sha256:first"))
            .await
            .unwrap();

        store
            .record_run_provenance("run-1", Some("{}"), None, None, None, None)
            .await
            .unwrap();
        store
            .record_run_provenance("run-1", None, None, None, None, Some("sha256:second"))
            .await
            .unwrap();

        let run = store.run("run-1").await.unwrap().unwrap();
        assert_eq!(run.image_digest.as_deref(), Some("sha256:first"));
        assert_eq!(run.config_json.as_deref(), Some("{}"));
    }

    #[tokio::test]
    async fn a_run_without_a_container_image_records_no_digest() {
        // The landlock path: no image is inspected, so no digest is recorded, and the run reads
        // as `None` rather than as some placeholder that would imitate a pin.
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("run-1", None, "running").await.unwrap();
        store
            .record_run_provenance("run-1", Some("{}"), Some("deadbeef"), None, None, None)
            .await
            .unwrap();

        let run = store.run("run-1").await.unwrap().unwrap();
        assert!(run.image_digest.is_none());
        assert_eq!(run.graph_hash.as_deref(), Some("deadbeef"));
    }

    #[test]
    fn a_database_from_before_image_digests_reads_them_as_absent() {
        // The narrow shape a store created before the column existed has. Migration adds it
        // nullable, an old row reads as "no image was pinned", and a widened write against the
        // migrated table works.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE runs (
                 run_id TEXT PRIMARY KEY, issue_id TEXT, status TEXT NOT NULL,
                 updated_at TEXT NOT NULL);
             CREATE TABLE checkpoints (
                 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
                 node_name TEXT NOT NULL, output_json TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')));
             INSERT INTO runs (run_id, status, updated_at) VALUES ('old', 'converged', 'then');",
        )
        .unwrap();

        let store = Store::from_connection(conn).unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();

        let run = rt.block_on(store.run("old")).unwrap().unwrap();
        assert!(run.image_digest.is_none());

        rt.block_on(store.record_run_provenance(
            "old",
            None,
            None,
            None,
            None,
            Some("sha256:late"),
        ))
        .unwrap();
        let run = rt.block_on(store.run("old")).unwrap().unwrap();
        assert_eq!(run.image_digest.as_deref(), Some("sha256:late"));
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
