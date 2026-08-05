//! The observability dashboard's HTTP layer: a small, read-only local server over the checkpoint
//! store.
//!
//! It opens the same SQLite file the run writes to and only ever calls read methods — the store's
//! single-writer discipline is preserved because this process never writes. WAL means these reads
//! don't block a run in progress.

pub mod launch;
pub mod pipeline;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use ratatoskr_store::{Checkpoint, Store};
use serde::Serialize;
use tower_http::services::{ServeDir, ServeFile};

use crate::launch::{LaunchError, Launcher};
use crate::pipeline::{ISSUE_NODE, NodeView};

/// Errors starting the server.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error(
        "no checkpoint store at {0} — run `ratatoskr plan` or `ratatoskr run` first, or point \
         --config at the right ratatoskr.toml"
    )]
    NoStore(PathBuf),
    #[error("store error: {0}")]
    Store(#[from] ratatoskr_store::StoreError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Clone)]
struct AppState {
    store: Store,
    launcher: Arc<Launcher>,
}

/// What `serve` needs: where to read, where to listen, and what to launch runs against.
pub struct ServeOptions<'a> {
    pub store_path: &'a Path,
    pub addr: SocketAddr,
    /// Working directory for runs started from the dashboard — the project's repo root.
    pub project: &'a Path,
    /// Config path handed to those runs, so they resolve the same settings this server did.
    pub config_path: &'a Path,
    /// How many runs may be in flight at once.
    pub max_runs: usize,
}

/// Serve the dashboard API on `addr`, reading the store at `store_path`.
///
/// Fails if the store file is absent rather than creating an empty one: `Store::open` would
/// happily produce a fresh database, and a typo'd path would then show an empty dashboard that
/// looks like "no runs yet" instead of a mistake.
pub async fn serve(opts: ServeOptions<'_>) -> Result<(), ServeError> {
    let ServeOptions {
        store_path,
        addr,
        project,
        config_path,
        max_runs,
    } = opts;
    if !store_path.exists() {
        return Err(ServeError::NoStore(store_path.to_path_buf()));
    }
    let store = Store::open(store_path)?;
    let launcher = Arc::new(Launcher::new(project, config_path, max_runs));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let web = web_dir();

    match &web {
        Some(dir) => {
            tracing::info!("serving dashboard from {}", dir.display());
            println!("dashboard on http://{bound}");
        }
        None => {
            // The UI is a separate build artifact, so a Rust-only checkout still gets a working
            // API instead of a hard failure.
            println!("dashboard API on http://{bound} (no UI build found — see {WEB_HINT})");
        }
    }

    axum::serve(listener, router(store, launcher, web)).await?;
    Ok(())
}

const WEB_HINT: &str = "crates/ratatoskr-serve/web: `bun install && bun run build`";

/// Where the built dashboard assets live, if they've been built.
///
/// `RATATOSKR_WEB_DIR` wins so a packaged build can point elsewhere; otherwise this is the
/// in-repo build output. Returning `None` is normal, not an error — the API stands alone.
fn web_dir() -> Option<PathBuf> {
    let candidate = match std::env::var_os("RATATOSKR_WEB_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/web/dist")),
    };
    candidate.join("index.html").is_file().then_some(candidate)
}

fn router(store: Store, launcher: Arc<Launcher>, web: Option<PathBuf>) -> Router {
    let api = Router::new()
        .route("/api/runs", get(list_runs).post(start_run))
        .route("/api/runs/{run_id}", get(run_detail))
        .route("/api/runs/{run_id}/nodes/{node}", get(node_checkpoints))
        .with_state(AppState { store, launcher });

    match web {
        // Unmatched paths fall back to index.html so the client owns its own routing.
        Some(dir) => {
            let index = dir.join("index.html");
            api.fallback_service(ServeDir::new(dir).fallback(ServeFile::new(index)))
        }
        None => api,
    }
}

/// A run as the list view shows it.
#[derive(Debug, Serialize)]
struct RunSummary {
    run_id: String,
    issue_id: Option<String>,
    status: String,
    updated_at: String,
}

/// A run's full pipeline view.
#[derive(Debug, Serialize)]
struct RunDetail {
    run_id: String,
    /// `None` for a run with checkpoints but no `runs` row — possible because the schema's
    /// foreign key isn't enforced and the scripted path checkpoints the issue first.
    status: Option<String>,
    issue_id: Option<String>,
    updated_at: Option<String>,
    /// The run's subject, from the `issue` pseudo-checkpoint. `runs.issue_id` is unset by the
    /// built-in flows, so this is normally the only record of what a run is about.
    issue: Option<String>,
    /// The most recent thing that happened, checkpoint or status change. `updated_at` alone moves
    /// only on status transitions, so it can't distinguish a live run from one killed mid-flight;
    /// this can, by how stale it is.
    last_activity: Option<String>,
    nodes: Vec<NodeView>,
    worktree: Option<WorktreeView>,
}

/// The implementer's worktree — the reviewable deliverable, kept on `converged` and
/// `max_iterations_reached` and removed by a hard error or `ratatoskr clean`. Reported separately
/// from node state on purpose: a converged run's worktree is usually still on disk.
#[derive(Debug, Serialize)]
struct WorktreeView {
    path: String,
    exists: bool,
}

/// One stored checkpoint, with its JSON parsed so the client gets structure rather than a string.
#[derive(Debug, Serialize)]
struct CheckpointView {
    node_name: String,
    created_at: String,
    output: serde_json::Value,
}

async fn list_runs(State(state): State<AppState>) -> Result<Json<Vec<RunSummary>>, ApiError> {
    let runs = state.store.list_runs().await?;
    Ok(Json(
        runs.into_iter()
            .map(|r| RunSummary {
                run_id: r.run_id,
                issue_id: r.issue_id,
                status: r.status,
                updated_at: r.updated_at,
            })
            .collect(),
    ))
}

async fn run_detail(
    State(state): State<AppState>,
    AxumPath(run_id): AxumPath<String>,
) -> Result<Json<RunDetail>, ApiError> {
    let run = state.store.run(&run_id).await?;
    let checkpoints = state.store.checkpoints_for_run(&run_id).await?;
    if run.is_none() && checkpoints.is_empty() {
        return Err(ApiError::NotFound(format!("no run {run_id}")));
    }

    let status = run.as_ref().map(|r| r.status.clone());
    let nodes = pipeline::derive(status.as_deref(), &checkpoints);
    let last_activity = checkpoints
        .iter()
        .map(|c| c.created_at.as_str())
        .chain(run.as_ref().map(|r| r.updated_at.as_str()))
        .max()
        .map(str::to_string);

    Ok(Json(RunDetail {
        run_id,
        status,
        issue_id: run.as_ref().and_then(|r| r.issue_id.clone()),
        updated_at: run.as_ref().map(|r| r.updated_at.clone()),
        issue: issue_text(&checkpoints),
        last_activity,
        nodes,
        worktree: worktree_view(&checkpoints),
    }))
}

async fn node_checkpoints(
    State(state): State<AppState>,
    AxumPath((run_id, node)): AxumPath<(String, String)>,
) -> Result<Json<Vec<CheckpointView>>, ApiError> {
    let all = state.store.checkpoints_for_run(&run_id).await?;
    // Every checkpoint, not just the latest: the implementer writes one per converge iteration and
    // the diagnostic progression between them is the interesting part.
    let views: Vec<CheckpointView> = all
        .into_iter()
        .filter(|c| c.node_name == node)
        .map(|c| CheckpointView {
            created_at: c.created_at,
            output: parse_or_raw(&c.output_json),
            node_name: c.node_name,
        })
        .collect();
    if views.is_empty() {
        return Err(ApiError::NotFound(format!(
            "no checkpoints for node {node} in run {run_id}"
        )));
    }
    Ok(Json(views))
}

/// Pull the run's issue text out of the `issue` pseudo-checkpoint.
fn issue_text(checkpoints: &[Checkpoint]) -> Option<String> {
    let raw = checkpoints
        .iter()
        .find(|c| c.node_name == ISSUE_NODE)?
        .output_json
        .as_str();
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    value
        .get("issue")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// The implementer records an absolute `worktree_path`; iterations reuse it, so the latest
/// checkpoint is authoritative. Whether it's still on disk is a filesystem question, not a
/// store one — `ratatoskr clean` removes worktrees without touching checkpoints.
fn worktree_view(checkpoints: &[Checkpoint]) -> Option<WorktreeView> {
    let raw = checkpoints
        .iter()
        .rev()
        .find(|c| c.node_name == "implementer")?
        .output_json
        .as_str();
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    let path = value.get("worktree_path")?.as_str()?.to_string();
    let exists = Path::new(&path).exists();
    Some(WorktreeView { path, exists })
}

/// Parse stored JSON, falling back to the raw text so a malformed checkpoint is still visible
/// rather than swallowing the whole response.
fn parse_or_raw(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

/// A request to start a run.
#[derive(Debug, serde::Deserialize)]
struct StartRun {
    issue: String,
}

/// The id of a run that has been started. Returned as soon as the process is spawned — a run
/// takes minutes, so the client follows it through the normal endpoints rather than waiting.
#[derive(Debug, Serialize)]
struct StartedRun {
    run_id: String,
}

async fn start_run(
    State(state): State<AppState>,
    Json(body): Json<StartRun>,
) -> Result<(StatusCode, Json<StartedRun>), ApiError> {
    let run_id = state.launcher.spawn(&body.issue)?;
    tracing::info!(kind = "run_started", run_id = %run_id, "started run from the dashboard");
    Ok((StatusCode::ACCEPTED, Json(StartedRun { run_id })))
}

/// API error responses.
#[derive(Debug, thiserror::Error)]
enum ApiError {
    #[error("{0}")]
    NotFound(String),
    #[error("store error: {0}")]
    Store(#[from] ratatoskr_store::StoreError),
    #[error("{0}")]
    Launch(#[from] LaunchError),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let code = match self {
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::Store(_) => StatusCode::INTERNAL_SERVER_ERROR,
            // Capacity is a "try again", a bad issue is the caller's fault, and a spawn failure
            // is ours.
            ApiError::Launch(LaunchError::AtCapacity(_)) => StatusCode::CONFLICT,
            ApiError::Launch(LaunchError::EmptyIssue) => StatusCode::BAD_REQUEST,
            ApiError::Launch(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (code, Json(serde_json::json!({ "error": self.to_string() }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cp(node: &str, json: &str) -> Checkpoint {
        Checkpoint {
            node_name: node.to_string(),
            output_json: json.to_string(),
            created_at: "t".to_string(),
        }
    }

    #[test]
    fn reads_the_issue_text_out_of_the_pseudo_node() {
        let cps = vec![cp(ISSUE_NODE, r#"{"issue":"fix the flaky retry"}"#)];
        assert_eq!(issue_text(&cps).as_deref(), Some("fix the flaky retry"));
        assert!(issue_text(&[]).is_none());
        // A malformed record is absent, not a panic.
        assert!(issue_text(&[cp(ISSUE_NODE, "not json")]).is_none());
    }

    #[test]
    fn takes_the_worktree_from_the_latest_implementer_checkpoint() {
        let cps = vec![
            cp("implementer", r#"{"worktree_path":"/tmp/old"}"#),
            cp(
                "implementer",
                r#"{"worktree_path":"/tmp/ratatoskr-definitely-absent"}"#,
            ),
        ];
        let wt = worktree_view(&cps).unwrap();
        assert_eq!(wt.path, "/tmp/ratatoskr-definitely-absent");
        assert!(!wt.exists);
        assert!(worktree_view(&[]).is_none());
    }

    #[test]
    fn a_malformed_checkpoint_still_renders() {
        assert_eq!(parse_or_raw(r#"{"a":1}"#), serde_json::json!({"a": 1}));
        assert_eq!(parse_or_raw("garbage"), serde_json::json!("garbage"));
    }
}
