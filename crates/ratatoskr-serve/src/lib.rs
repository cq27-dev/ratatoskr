//! The observability dashboard's HTTP layer: a small, read-only local server over the checkpoint
//! store.
//!
//! It opens the same SQLite file the run writes to and only ever calls read methods — the store's
//! single-writer discipline is preserved because this process never writes. WAL means these reads
//! don't block a run in progress.

pub mod clarify;
pub mod events;
pub mod launch;
pub mod pipeline;
pub mod project;

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use ratatoskr_store::Checkpoint;
use serde::Serialize;
use tokio_stream::StreamExt as _;
use tower_http::services::{ServeDir, ServeFile};

use crate::clarify::{AnswerError, AskReply, AskRequest, Desk};
use crate::launch::LaunchError;
use crate::pipeline::{ISSUE_NODE, NodeView};
pub use crate::project::ProjectSpec;
use crate::project::{Project, ProjectError, ProjectView};

/// Errors starting the server.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error(transparent)]
    Project(#[from] ProjectError),
    #[error("store error: {0}")]
    Store(#[from] ratatoskr_store::StoreError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Clone)]
struct AppState {
    projects: Arc<BTreeMap<String, Project>>,
    /// Questions from runs waiting on a human, and who is watching. Shared across projects: run
    /// ids are unique, so a question needs no further scoping.
    desk: Arc<Desk>,
}

impl AppState {
    /// The project a request names, or a 404 that says which one was asked for.
    fn project(&self, slug: &str) -> Result<&Project, ApiError> {
        self.projects
            .get(slug)
            .ok_or_else(|| ApiError::NotFound(format!("no project `{slug}`")))
    }
}

/// What `serve` needs: where to listen, and which projects to watch.
pub struct ServeOptions {
    pub addr: SocketAddr,
    /// One or more projects. Each keeps its own store, worktrees, and logs — nothing is merged.
    pub projects: Vec<ProjectSpec>,
    /// How many runs may be in flight at once, per project.
    pub max_runs: usize,
}

/// Serve the dashboard for one or more projects.
pub async fn serve(opts: ServeOptions) -> Result<(), ServeError> {
    let ServeOptions {
        addr,
        projects,
        max_runs,
    } = opts;
    // Bind before opening the projects: a spawned run is told where to reach this server, and with
    // port 0 the real port isn't known until the listener exists.
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let projects = Arc::new(project::open_all(
        projects,
        max_runs,
        &dashboard_url(bound),
    )?);
    let desk = Arc::new(Desk::default());
    let web = web_dir();

    for slug in projects.keys() {
        tracing::info!("watching project {slug}");
    }
    match &web {
        Some(dir) => {
            tracing::info!("serving dashboard from {}", dir.display());
            println!(
                "dashboard on http://{bound} ({} project(s))",
                projects.len()
            );
        }
        None => {
            // The UI is a separate build artifact, so a Rust-only checkout still gets a working
            // API instead of a hard failure.
            println!("dashboard API on http://{bound} (no UI build found — see {WEB_HINT})");
        }
    }

    axum::serve(listener, router(projects, desk, web)).await?;
    Ok(())
}

/// Where a spawned run should reach this server.
///
/// A wildcard bind (`0.0.0.0`) is an address to listen on, not one to connect to, so the child is
/// pointed at loopback on the same port.
fn dashboard_url(bound: SocketAddr) -> String {
    if bound.ip().is_unspecified() {
        format!("http://127.0.0.1:{}", bound.port())
    } else {
        format!("http://{bound}")
    }
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

fn router(
    projects: Arc<BTreeMap<String, Project>>,
    desk: Arc<Desk>,
    web: Option<PathBuf>,
) -> Router {
    let api = Router::new()
        .route("/api/projects", get(list_projects))
        .route(
            "/api/projects/{project}/runs",
            get(list_runs).post(start_run),
        )
        .route("/api/projects/{project}/runs/{run_id}", get(run_detail))
        .route(
            "/api/projects/{project}/runs/{run_id}/nodes/{node}",
            get(node_checkpoints),
        )
        .route(
            "/api/projects/{project}/runs/{run_id}/events",
            get(run_events),
        )
        // Answering is keyed by question id alone — unique across every project.
        .route(
            "/api/clarifications/{question_id}",
            axum::routing::post(answer_question),
        )
        // Not for browsers: the waiting end of the rendezvous, called by a run process.
        .route(
            "/internal/clarifications",
            axum::routing::post(await_answer),
        )
        .with_state(AppState { projects, desk });

    match web {
        // Unmatched paths fall back to index.html so the client owns its own routing.
        Some(dir) => {
            let index = dir.join("index.html");
            api.fallback_service(ServeDir::new(dir).fallback(ServeFile::new(index)))
        }
        None => api,
    }
}

async fn list_projects(State(state): State<AppState>) -> Json<Vec<ProjectView>> {
    Json(
        state
            .projects
            .values()
            .map(|p| ProjectView {
                slug: p.slug.clone(),
                dir: p.dir.display().to_string(),
            })
            .collect(),
    )
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

async fn list_runs(
    State(state): State<AppState>,
    AxumPath(project): AxumPath<String>,
) -> Result<Json<Vec<RunSummary>>, ApiError> {
    let runs = state.project(&project)?.store.list_runs().await?;
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
    AxumPath((project, run_id)): AxumPath<(String, String)>,
) -> Result<Json<RunDetail>, ApiError> {
    let found = state.project(&project)?;
    let store = &found.store;
    let config_path = found.config_path.clone();
    let run = store.run(&run_id).await?;
    let checkpoints = store.checkpoints_for_run(&run_id).await?;
    if run.is_none() && checkpoints.is_empty() {
        return Err(ApiError::NotFound(format!("no run {run_id}")));
    }

    let status = run.as_ref().map(|r| r.status.clone());
    // Best-effort: an unreadable or missing config costs the planned facts and nothing else, and a
    // dashboard that refused to show a run because its config moved would be worse than one that
    // shows the run without them.
    let config = std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|t| ratatoskr_core::RatatoskrConfig::from_toml_str(&t).ok());
    let nodes = pipeline::derive_with(status.as_deref(), &checkpoints, config.as_ref());
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
    AxumPath((project, run_id, node)): AxumPath<(String, String, String)>,
) -> Result<Json<Vec<CheckpointView>>, ApiError> {
    let all = state
        .project(&project)?
        .store
        .checkpoints_for_run(&run_id)
        .await?;
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
    AxumPath(project): AxumPath<String>,
    Json(body): Json<StartRun>,
) -> Result<(StatusCode, Json<StartedRun>), ApiError> {
    let run_id = state.project(&project)?.launcher.spawn(&body.issue)?;
    tracing::info!(kind = "run_started", run_id = %run_id, "started run from the dashboard");
    Ok((StatusCode::ACCEPTED, Json(StartedRun { run_id })))
}

/// Stream a run's activity as it happens.
///
/// Checkpoints only tell you a node *finished*; this is what it is doing in between. The stream
/// replays the run's recent history on connect, then follows the log, and ends when the client
/// disconnects — the tailing task is owned by the channel and dies with it.
async fn run_events(
    State(state): State<AppState>,
    AxumPath((project, run_id)): AxumPath<(String, String)>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError>
{
    let dir = state.project(&project)?.log_dir.clone();
    let (tx, rx) = tokio::sync::mpsc::channel(256);
    // Watching this run *is* holding an event stream open, so attendance is exactly this task's
    // lifetime — no disconnect handling to get wrong.
    let attending = state.desk.attend(&project, &run_id);
    tokio::spawn(async move {
        events::follow(dir, run_id, tx).await;
        drop(attending);
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(|event| {
        Ok(Event::default()
            .json_data(event)
            .unwrap_or_else(|_| Event::default().data("{}")))
    });
    // A keep-alive comment holds the connection open through the quiet stretch while a node waits
    // on a model, which can outlast an idle-timeout proxy.
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// A run asking whether a human will answer. Blocks until one does, or until it's clear none
/// will — which is immediately when nobody is watching, so an unattended run is never delayed.
async fn await_answer(
    State(state): State<AppState>,
    Json(req): Json<AskRequest>,
) -> Json<AskReply> {
    let answer = state
        .desk
        .wait_for_answer(&req.project, &req.run_id, &req.question_id)
        .await;
    Json(AskReply { answer })
}

/// A human answering a parked question.
#[derive(Debug, serde::Deserialize)]
struct Answer {
    answer: String,
}

async fn answer_question(
    State(state): State<AppState>,
    AxumPath(question_id): AxumPath<String>,
    Json(body): Json<Answer>,
) -> Result<StatusCode, ApiError> {
    match state.desk.answer(&question_id, body.answer) {
        Ok(()) => Ok(StatusCode::NO_CONTENT),
        // Already answered, timed out, or the run moved on — all the same to a viewer, and all
        // reachable by clicking twice or answering a question replayed from history.
        Err(AnswerError::NotPending) => Err(ApiError::Gone(
            "that question is no longer waiting for an answer".to_string(),
        )),
    }
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
    #[error("{0}")]
    Gone(String),
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
            ApiError::Gone(_) => StatusCode::GONE,
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
            ..Default::default()
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
