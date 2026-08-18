//! The observability dashboard's HTTP layer, over the checkpoint store.
//!
//! It opens the same SQLite file the run writes to and only ever calls read methods — the store's
//! single-writer discipline is preserved because this process never writes to a *project's* store.
//! WAL means these reads don't block a run in progress. The instance's own identity database is a
//! different file and is written here; see [`ratatoskr_store::auth`].
//!
//! Two listeners, not one. The public one carries the dashboard and its API, and is the one that
//! may face a network. The internal one carries the rendezvous a run process calls to ask a human
//! a question, binds loopback, and is therefore unreachable from outside by construction rather
//! than by a rule someone has to keep enforcing.

pub mod auth;
pub mod clarify;
pub mod events;
pub mod github;
pub mod launch;
pub mod pipeline;
pub mod project;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{FromRef, Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ratatoskr_core::auth::{Access, Role};
use ratatoskr_core::{Command, Control, ControlView, RunControl};
use ratatoskr_store::Checkpoint;
use ratatoskr_store::auth::AuthStore;
use ratatoskr_store::provider_pause::{
    ProviderPauseDisposition, ProviderPauseKey, ProviderPauseRegistration, ProviderPauseStore,
};
use serde::Serialize;
use tokio_stream::StreamExt as _;
use tower_http::services::{ServeDir, ServeFile};

use crate::auth::Caller;
use crate::clarify::{AnswerError, AskReply, AskRequest, Desk};
use crate::launch::LaunchError;
use crate::pipeline::{ISSUE_NODE, NodeView};
pub use crate::project::ProjectSpec;
use crate::project::{Project, ProjectError, ProjectView};

/// Errors starting the server.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error("internal address must be loopback, got {0}")]
    InternalAddressNotLoopback(SocketAddr),
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
    /// Questions from runs waiting on a human, and who is watching. Questions have their own
    /// unique ids; [`Desk`] scopes viewer attendance by project and run id.
    desk: Arc<Desk>,
    /// What operators have asked of the runs in flight, keyed by project and run id.
    ///
    /// This remains in memory: a command is advice to a process that is running right now, so
    /// persisting it would resurrect a pause for a run that is long finished. A child waiting on a
    /// provider pause lives in `provider_pauses` instead, because its resume tombstone must survive
    /// dashboard restart. Neither belongs in a project's store, which only a run process writes.
    control: Arc<Mutex<HashMap<RunKey, RunControlState>>>,
    /// Durable, idempotent delivery records for provider pauses in the instance database.
    provider_pauses: ProviderPauseStore,
    /// Who may use this instance. Instance-wide, not per project — see `ratatoskr_store::auth`.
    auth: AuthStore,
    /// Failed sign-ins, so a password cannot be guessed at network speed.
    throttle: Arc<auth::LoginThrottle>,
    /// The GitHub integration, when one is configured. `None` leaves the webhook route refusing
    /// everything, which is what an instance that has not set it up should do.
    github: Option<Arc<github::GitHubConfig>>,
    /// Whether to mark the session cookie `Secure` and give it the `__Host-` prefix.
    ///
    /// Off for a loopback instance, where a browser would discard such a cookie outright, and on
    /// for anything reachable over TLS. Not derived from the request, because a reverse proxy
    /// terminates TLS and the request arrives here over plain HTTP either way.
    secure_cookies: bool,
}

/// Volatile control needs the same scope as the durable provider-pause ledger: a run id is only
/// unique within a project.
#[derive(Clone, Hash, PartialEq, Eq)]
struct RunKey {
    project: String,
    run_id: String,
}

impl RunKey {
    fn new(project: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            project: project.into(),
            run_id: run_id.into(),
        }
    }
}

/// The dashboard's volatile controls for one run.
///
/// Volatile operator control for one run.
#[derive(Default)]
struct RunControlState {
    control: RunControl,
}

impl RunControlState {
    fn poll(&mut self, node: &str) -> Control {
        self.control.poll(node)
    }

    fn is_empty(&self) -> bool {
        self.control.is_empty()
    }
}

/// So `Caller` can be extracted: the extractor needs the identity database and nothing else.
impl FromRef<AppState> for AuthStore {
    fn from_ref(state: &AppState) -> Self {
        state.auth.clone()
    }
}

impl AppState {
    /// The project a request names and the full id of the run it names, resolving a short one.
    ///
    /// The dashboard puts an eight-character run id in its URLs, the way a git short hash works,
    /// so a request can arrive naming a prefix. Resolved here — one place — rather than in each of
    /// the four handlers that take a run id.
    async fn project_and_run(
        &self,
        slug: &str,
        run_id: &str,
        caller: &Caller,
        access: Access,
    ) -> Result<(&Project, String), ApiError> {
        let project = self.project(slug, caller, access)?;
        let resolved = project
            .store
            .resolve_run(run_id)
            .await?
            // A prefix nothing starts with is kept as-is rather than refused here: the handlers
            // below already answer an unknown run, and this way they answer it the same way for a
            // short id as for a long one.
            .unwrap_or_else(|| run_id.to_string());
        Ok((project, resolved))
    }

    /// The project a request names, if this caller may do `access` to it.
    ///
    /// Authorization lives here rather than in each handler because every project-scoped handler
    /// already had to call this to get anywhere — so a route that forgets to check does not
    /// compile, instead of quietly serving.
    ///
    /// A caller who may not read a private project gets the same 404 as one naming a project that
    /// does not exist. Anything else answers "does this repository exist here?" for free.
    fn project(&self, slug: &str, caller: &Caller, access: Access) -> Result<&Project, ApiError> {
        let missing = || ApiError::NotFound(format!("no project `{slug}`"));
        let project = self.projects.get(slug).ok_or_else(missing)?;
        if !caller.may(project.visibility, access) {
            return match (access, caller) {
                // Reading is hidden: a stranger learns nothing about what this instance watches.
                (Access::Read, _) => Err(missing()),
                // Acting is not: the caller can already see the project, so refusing with a reason
                // tells them what to do about it.
                (Access::Act, caller) => Err(caller.denied(&format!("act on `{slug}`"))),
            };
        }
        Ok(project)
    }
}

/// What `serve` needs: where to listen, and which projects to watch.
pub struct ServeOptions {
    pub addr: SocketAddr,
    /// Where the rendezvous a run process calls is bound. Loopback, always — a run is a child of
    /// this process on this host, and reaching this address is what stands in for a credential.
    pub internal_addr: SocketAddr,
    /// One or more projects. Each keeps its own store, worktrees, and logs — nothing is merged.
    pub projects: Vec<ProjectSpec>,
    /// How many runs may be in flight at once, per project.
    pub max_runs: usize,
    /// The instance database for identities and durable provider-pause delivery. Created on first
    /// use.
    pub auth_db: PathBuf,
    /// Whether this instance is reached over TLS, which decides the session cookie's attributes.
    /// See [`AppState::secure_cookies`].
    pub secure_cookies: bool,
    /// The GitHub integration, if this instance has one.
    pub github: Option<github::GitHubConfig>,
}

/// Keep the unauthenticated child rendezvous unreachable from the network.
///
/// Reaching the internal listener is the child's credential for clarification and control
/// requests. The listener must therefore reject wildcard and routable addresses before the public
/// listener is opened.
fn require_loopback_internal_address(address: SocketAddr) -> Result<(), ServeError> {
    if address.ip().is_loopback() {
        Ok(())
    } else {
        Err(ServeError::InternalAddressNotLoopback(address))
    }
}

/// Serve the dashboard for one or more projects.
pub async fn serve(opts: ServeOptions) -> Result<(), ServeError> {
    let ServeOptions {
        addr,
        internal_addr,
        projects,
        max_runs,
        auth_db,
        secure_cookies,
        github,
    } = opts;
    require_loopback_internal_address(internal_addr)?;
    // Bind before opening the projects: a spawned run is told where to reach this server, and with
    // port 0 the real port isn't known until the listener exists.
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let internal = tokio::net::TcpListener::bind(internal_addr).await?;
    let internal_bound = internal.local_addr()?;
    let projects = Arc::new(project::open_all(
        projects,
        max_runs,
        // The INTERNAL address, not the public one. This is the URL a run process is handed to
        // ask a human a question, and the route that answers it only exists on that listener.
        &dashboard_url(internal_bound),
    )?);
    let desk = Arc::new(Desk::default());
    let web = web_dir();
    let auth = AuthStore::open(&auth_db)?;
    let provider_pauses = ProviderPauseStore::open(&auth_db)?;
    for project in projects.values() {
        project
            .launcher
            .set_provider_pause_store(provider_pauses.clone());
    }
    report_unresponsive_provider_pauses(provider_pauses.clone());
    if auth.is_empty().await? {
        // Not fatal: an instance with no principals still serves its public projects, and the
        // loopback case that needs no accounts at all is the common one. Said once, loudly,
        // because on a hosted instance it means nobody can log in.
        tracing::warn!(
            "no principals in {} — nothing can be started or answered until one exists \
             (`ratatoskr users add <name> --role operator`)",
            auth_db.display()
        );
    }
    if let Some(config) = &github {
        let addressable: Vec<&str> = projects
            .values()
            .filter_map(|p| p.repository.as_deref())
            .collect();
        if addressable.is_empty() {
            // Configured but useless: nothing here has a GitHub origin, so every delivery would be
            // about a repository this instance does not serve.
            tracing::warn!(
                "the GitHub integration is configured as /{}, but no project has a GitHub origin \
                 — mentions will be ignored",
                config.trigger
            );
        } else {
            tracing::info!(
                "@{} can start runs on {}",
                config.trigger,
                addressable.join(", ")
            );
        }
    }
    let state = AppState {
        projects: Arc::clone(&projects),
        desk,
        control: Arc::default(),
        provider_pauses,
        auth,
        throttle: Arc::new(auth::LoginThrottle::default()),
        secure_cookies,
        github: github.map(Arc::new),
    };

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

    tracing::info!("clarification rendezvous on http://{internal_bound}");
    // Both, concurrently, and either finishing ends the process: a dashboard whose internal
    // listener has died cannot answer a question, and would sit there looking healthy.
    let public = axum::serve(listener, router(state.clone(), web));
    let private = axum::serve(internal, internal_router(state));
    tokio::select! {
        result = public => result?,
        result = private => result?,
    }
    Ok(())
}

/// A child waiting for an operator proves liveness through paused polls and acknowledgements.
///
/// Heartbeats cannot prove death: a suspended host can pause both the child and dashboard longer
/// than the grace period. Report an unresponsive pause once, but retain it until the launcher
/// observes a real child exit and records the only irreversible lifecycle fence.
fn report_unresponsive_provider_pauses(pauses: ProviderPauseStore) {
    const RECONNECT_GRACE: Duration = Duration::from_secs(30);
    const RECONCILE_EVERY: Duration = Duration::from_secs(5);
    tokio::spawn(async move {
        tokio::time::sleep(RECONNECT_GRACE).await;
        let mut interval = tokio::time::interval(RECONCILE_EVERY);
        let mut reported = HashSet::new();
        loop {
            interval.tick().await;
            match pauses.list_unresponsive_for(RECONNECT_GRACE).await {
                Ok(keys) => {
                    let current: HashSet<_> = keys.into_iter().collect();
                    for key in current.difference(&reported) {
                        tracing::warn!(
                            kind = "run_pause_suspect",
                            project = key.project(),
                            run_id = key.run_id(),
                            "provider-paused child has not renewed its heartbeat; retaining pause until confirmed exit"
                        );
                    }
                    reported = current;
                }
                Err(error) => tracing::error!(
                    kind = "provider_pause_reconcile_failed",
                    %error,
                    "could not reconcile provider pause liveness"
                ),
            }
        }
    });
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

/// The public router: the dashboard, its API, and the session routes.
///
/// The rendezvous a run process calls is deliberately absent — it lives on [`internal_router`],
/// which binds loopback. Splitting them means the internal endpoint is not exposed and then
/// guarded; it is simply not there to reach.
fn router(state: AppState, web: Option<PathBuf>) -> Router {
    let api = Router::new()
        .route("/api/auth/login", post(login))
        .route("/api/auth/logout", post(logout))
        .route("/api/auth/me", get(whoami))
        // Public by necessity: GitHub has to reach it. Its signature is what makes a delivery
        // trustworthy — see the module docs.
        .route("/api/integrations/github", post(github_webhook))
        .route("/api/projects", get(list_projects))
        .route(
            "/api/projects/{project}/runs",
            get(list_runs).post(start_run),
        )
        .route("/api/projects/{project}/runs/{run_id}", get(run_detail))
        .route(
            "/api/projects/{project}/runs/{run_id}/control",
            post(control_run),
        )
        .route(
            "/api/projects/{project}/runs/{run_id}/nodes/{node}",
            get(node_checkpoints),
        )
        .route(
            "/api/projects/{project}/runs/{run_id}/events",
            get(run_events),
        )
        .route(
            "/api/projects/{project}/runs/{run_id}/history",
            get(run_history),
        )
        // Answering is keyed by question id alone — unique across every project.
        .route(
            "/api/clarifications/{question_id}",
            axum::routing::post(answer_question),
        )
        .with_state(state);

    match web {
        // Unmatched paths fall back to index.html so the client owns its own routing.
        Some(dir) => {
            let index = dir.join("index.html");
            api.fallback_service(ServeDir::new(dir).fallback(ServeFile::new(index)))
        }
        None => api,
    }
}

/// The loopback-only router: the waiting end of the rendezvous, called by a run process.
///
/// No authentication, because reaching it is the credential — it is bound to 127.0.0.1 and a run
/// is a child of this process on the same host. If this ever needs to face a network, it needs a
/// machine credential first; do not simply move the route back.
fn internal_router(state: AppState) -> Router {
    Router::new()
        .route("/internal/clarifications", post(await_answer))
        .route("/internal/control", post(node_control))
        .route("/internal/control/pause", post(pause_run))
        .route(
            "/internal/control/pause/ack",
            post(acknowledge_provider_pause),
        )
        .with_state(state)
}

/// What a run process asks for: which node, of which run.
#[derive(serde::Deserialize)]
struct ControlAsk {
    project: String,
    run_id: String,
    node: String,
    /// A provider-paused child includes its stable waiter identity on every wait poll. If the
    /// dashboard restarted, each returning child re-establishes its own acknowledged hold without
    /// making ordinary turn-boundary polls hang.
    #[serde(default)]
    provider_pause_waiter: Option<String>,
    /// The newest provider pause generation this child has acknowledged. It lets a subsequent
    /// provider failure start a new pause while an older concurrent waiter is still acknowledging.
    #[serde(default)]
    known_provider_resume_generation: Option<i64>,
}

/// What a run process needs to record an automatic, run-wide pause.
#[derive(serde::Deserialize)]
struct PauseAsk {
    project: String,
    run_id: String,
    node: String,
    provider_pause_waiter: String,
    #[serde(default)]
    known_provider_resume_generation: Option<i64>,
}

/// The durable disposition of a provider pause registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct ProviderPauseReply {
    directive: ProviderPauseDirective,
    generation: i64,
}

/// The durable directive that remains in force when a child retries an acknowledgement.
#[derive(serde::Serialize, serde::Deserialize)]
struct ProviderPauseAcknowledgementReply {
    directive: ProviderPauseDirective,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProviderPauseDirective {
    Hold,
    Continue,
    Stop,
}

impl From<ProviderPauseRegistration> for ProviderPauseReply {
    fn from(registration: ProviderPauseRegistration) -> Self {
        let directive = match registration.disposition {
            ProviderPauseDisposition::Hold => ProviderPauseDirective::Hold,
            ProviderPauseDisposition::Continue => ProviderPauseDirective::Continue,
            ProviderPauseDisposition::Stop | ProviderPauseDisposition::Exited => {
                ProviderPauseDirective::Stop
            }
        };
        Self {
            directive,
            generation: registration.generation,
        }
    }
}

impl From<ProviderPauseDisposition> for ProviderPauseAcknowledgementReply {
    fn from(disposition: ProviderPauseDisposition) -> Self {
        let directive = match disposition {
            ProviderPauseDisposition::Continue => ProviderPauseDirective::Continue,
            ProviderPauseDisposition::Stop | ProviderPauseDisposition::Exited => {
                ProviderPauseDirective::Stop
            }
            ProviderPauseDisposition::Hold => {
                unreachable!("a provider pause acknowledgement cannot retain a hold")
            }
        };
        Self { directive }
    }
}

/// One internal control reply, including a paused waiter's server-assigned generation.
#[derive(serde::Serialize)]
struct ProviderControlReply {
    #[serde(flatten)]
    control: Control,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_pause_generation: Option<i64>,
}

/// Acknowledge one delivery against the exact generation that produced it.
#[derive(serde::Deserialize)]
struct PauseAcknowledgement {
    project: String,
    run_id: String,
    node: String,
    provider_pause_waiter: String,
    provider_pause_generation: i64,
}

/// Answer one node's "what should I do now?".
///
/// Loopback only, like the clarification rendezvous beside it, and unauthenticated for the same
/// reason: reaching this port is the credential. It takes the operator's text with it, so a reply
/// is delivered exactly once — a message re-read on every poll would be the operator saying it
/// again on every turn.
async fn node_control(
    State(state): State<AppState>,
    Json(ask): Json<ControlAsk>,
) -> Json<ProviderControlReply> {
    let key = ProviderPauseKey::new(&ask.project, &ask.run_id);
    let provider_registration = match ask.provider_pause_waiter.as_deref() {
        Some(waiter) => state
            .provider_pauses
            .register_for_node(
                &key,
                waiter,
                &ask.node,
                ask.known_provider_resume_generation,
            )
            .await
            .map(Some),
        None => Ok(None),
    };
    let (provider_holding, provider_stopped, provider_exited, provider_pause_generation) =
        match provider_registration {
            Ok(Some(registration)) => (
                registration.disposition == ProviderPauseDisposition::Hold,
                registration.disposition == ProviderPauseDisposition::Stop,
                registration.disposition == ProviderPauseDisposition::Exited,
                Some(registration.generation),
            ),
            Ok(None) => match state.provider_pauses.is_exited(&key).await {
                Ok(true) => (false, false, true, None),
                Ok(false) => match state.provider_pauses.is_node_stopped(&key, &ask.node).await {
                    Ok(true) => (false, true, false, None),
                    Ok(false) => match state.provider_pauses.is_holding(&key).await {
                        Ok(holding) => (holding, false, false, None),
                        Err(error) => {
                            tracing::warn!(project = ask.project, run_id = ask.run_id, %error, "could not read provider pause delivery state");
                            // A durable pause we cannot read must never become an implicit provider retry.
                            (true, false, false, None)
                        }
                    },
                    Err(error) => {
                        tracing::warn!(project = ask.project, run_id = ask.run_id, %error, "could not read durable node stop state");
                        // An ordinary poll remains available when a Stop that was never written
                        // cannot be queried. A paused waiter uses the fail-closed branch above.
                        (false, false, false, None)
                    }
                },
                Err(error) => {
                    tracing::warn!(project = ask.project, run_id = ask.run_id, %error, "could not read provider pause lifecycle state");
                    // A durable pause we cannot read must never become an implicit provider retry.
                    (true, false, false, None)
                }
            },
            Err(error) => {
                tracing::warn!(project = ask.project, run_id = ask.run_id, %error, "could not read provider pause delivery state");
                // A durable pause we cannot read must never become an implicit provider retry.
                (true, false, false, None)
            }
        };

    let mut control = state.control.lock().expect("control mutex poisoned");
    let run_key = RunKey::new(&ask.project, &ask.run_id);
    let mut reply = control
        .get_mut(&run_key)
        .map(|run| run.poll(&ask.node))
        .unwrap_or_else(Control::carry_on);
    if provider_stopped || provider_exited {
        reply.directive = ratatoskr_core::Directive::Stop;
    } else if provider_holding && reply.directive == ratatoskr_core::Directive::Continue {
        reply.directive = ratatoskr_core::Directive::Hold;
    }
    if control.get(&run_key).is_some_and(RunControlState::is_empty) {
        control.remove(&run_key);
    }
    Json(ProviderControlReply {
        control: reply,
        provider_pause_generation,
    })
}

/// Persist a provider pause before a run waits for an operator.
///
/// Loopback-only like [`node_control`]. The narrow endpoint deliberately accepts no command or
/// text: an untrusted provider failure may only put its own run on hold, never stop another node
/// or inject content into a model conversation.
async fn pause_run(
    State(state): State<AppState>,
    Json(ask): Json<PauseAsk>,
) -> Result<Json<ProviderPauseReply>, StatusCode> {
    let key = ProviderPauseKey::new(&ask.project, &ask.run_id);
    state
        .provider_pauses
        .register_for_node(
            &key,
            &ask.provider_pause_waiter,
            &ask.node,
            ask.known_provider_resume_generation,
        )
        .await
        .map(ProviderPauseReply::from)
        .map(Json)
        .map_err(|error| {
            tracing::warn!(project = ask.project, run_id = ask.run_id, %error, "could not persist provider pause");
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

/// Acknowledge a provider pause delivery before the child resumes or stops.
async fn acknowledge_provider_pause(
    State(state): State<AppState>,
    Json(ask): Json<PauseAcknowledgement>,
) -> Result<Json<ProviderPauseAcknowledgementReply>, StatusCode> {
    let key = ProviderPauseKey::new(&ask.project, &ask.run_id);
    state
        .provider_pauses
        .acknowledge(
            &key,
            ask.provider_pause_generation,
            &ask.provider_pause_waiter,
            &ask.node,
        )
        .await
        .map(ProviderPauseAcknowledgementReply::from)
        .map(Json)
        .map_err(|error| {
            tracing::warn!(project = ask.project, run_id = ask.run_id, %error, "could not acknowledge provider pause");
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

/// Pause, resume, stop, start or steer a run in flight.
///
/// Needs `Act`, the same as starting a run: these change what a run does, and a reader of a public
/// project must not be able to stop it. Recorded here and nowhere else — the run picks it up at its
/// next turn boundary, which is why the answer is what the *operator asked for* rather than what
/// the run has since done about it.
async fn control_run(
    State(state): State<AppState>,
    caller: Caller,
    AxumPath((project, run_id)): AxumPath<(String, String)>,
    Json(command): Json<Command>,
) -> Result<Json<ControlView>, ApiError> {
    let (found, run_id) = state
        .project_and_run(&project, &run_id, &caller, Access::Act)
        .await?;
    let provider_key = ProviderPauseKey::new(&found.slug, &run_id);
    let run_key = RunKey::new(&found.slug, &run_id);
    if state.provider_pauses.is_exited(&provider_key).await? {
        state
            .control
            .lock()
            .expect("control mutex poisoned")
            .remove(&run_key);
        return Ok(Json(ControlView::default()));
    }
    match &command {
        Command::Resume => state.provider_pauses.release(&provider_key).await?,
        Command::Stop { node } => state.provider_pauses.stop(&provider_key, node).await?,
        Command::Start { node } => {
            state
                .provider_pauses
                .clear_stop(&provider_key, node)
                .await?
        }
        Command::Pause | Command::Steer { .. } => {}
    }
    let mut view = {
        let mut control = state.control.lock().expect("control mutex poisoned");
        let run = control.entry(run_key.clone()).or_default();
        tracing::info!(project = found.slug, run_id, ?command, "operator control");
        run.control.apply(command);
        let view = run.control.view();
        // Nothing asked for and nothing outstanding: drop the entry rather than accumulate one per
        // run this instance has ever watched.
        if run.is_empty() {
            control.remove(&run_key);
        }
        view
    };
    view.paused |= state.provider_pauses.is_holding(&provider_key).await?;
    append_durable_stops(
        &mut view,
        state.provider_pauses.stopped_nodes(&provider_key).await?,
    );
    Ok(Json(view))
}

/// Merge durable provider-pause Stops with ordinary in-memory controls for one dashboard view.
fn append_durable_stops(view: &mut ControlView, stopped: Vec<String>) {
    view.stopped.extend(stopped);
    view.stopped.sort_unstable();
    view.stopped.dedup();
}

/// Start a run because someone mentioned the bot in an issue.
///
/// Answers 200 to anything correctly signed, whether or not it did something. GitHub retries a
/// delivery that fails, and there is nothing to retry about a comment that was not addressed to us
/// or came from someone we do not know — a 4xx there would turn one ignored comment into a stream
/// of them. What did or did not happen goes to the log, which is where an operator looks.
async fn github_webhook(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<StatusCode, ApiError> {
    let Some(config) = state.github.clone() else {
        // Not configured. A 404 rather than a 501, so an instance that has not set this up does not
        // advertise that it could.
        return Err(ApiError::NotFound("no such endpoint".to_string()));
    };

    let request = match github::read(&headers, &body, &config) {
        Ok(request) => request,
        Err(github::Refusal::Unsigned) => {
            // The only refusal that gets a status: an unsigned caller is not GitHub, and telling
            // them apart from a caller we simply ignored is the point.
            tracing::warn!(
                kind = "github_unsigned",
                "refused an unsigned webhook delivery"
            );
            return Err(ApiError::Unauthorized("bad signature".to_string()));
        }
        Err(_) => return Ok(StatusCode::OK),
    };

    let principal = match github::principal_for(&state.auth, &request).await {
        Ok(principal) => principal,
        Err(_) => {
            // Named, because "why did the bot ignore me" is the question this log line answers.
            tracing::info!(
                kind = "github_unauthorized",
                login = %request.sender_login,
                id = %request.sender_id,
                "ignored a mention from someone without operator"
            );
            return Ok(StatusCode::OK);
        }
    };

    let Some(project) = state
        .projects
        .values()
        .find(|p| p.repository.as_deref() == Some(request.repository.as_str()))
    else {
        tracing::info!(
            kind = "github_unknown_repository",
            repository = %request.repository,
            "ignored a mention on a repository this instance does not serve"
        );
        return Ok(StatusCode::OK);
    };

    // The issue number travels with the instruction, because everything downstream — the branch
    // name, the commit subject, the pull request — is built from it.
    let issue = format!(
        "GitHub issue #{}: {}\n\nRepository: {}",
        request.issue_number, request.instruction, request.repository
    );
    match project.launcher.spawn(&issue) {
        Ok(run_id) => tracing::info!(
            kind = "run_started",
            run_id = %run_id,
            principal = %principal.principal_id,
            login = %request.sender_login,
            issue_number = request.issue_number,
            "started run from a GitHub mention"
        ),
        // Capacity and a bad instruction are both "not now" from GitHub's point of view, and
        // neither is worth a retry storm.
        Err(error) => tracing::warn!(
            kind = "github_launch_failed",
            %error,
            issue_number = request.issue_number,
            "could not start a run from a GitHub mention"
        ),
    }
    Ok(StatusCode::OK)
}

/// What the browser posts to log in.
#[derive(Debug, serde::Deserialize)]
struct Login {
    username: String,
    password: String,
}

/// Who the caller is, for the dashboard to render.
#[derive(Debug, Serialize)]
struct Me {
    /// `None` when nobody is logged in — not an error, because an anonymous caller is a valid
    /// caller on a public project and the dashboard has to draw something for them.
    #[serde(skip_serializing_if = "Option::is_none")]
    principal_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<Role>,
}

/// Exchange a username and password for a session cookie.
async fn login(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Login>,
) -> Result<Response, ApiError> {
    if !state.throttle.may_try(&body.username) {
        return Err(ApiError::TooManyAttempts);
    }
    let Some(principal) = state
        .auth
        .authenticate(&body.username, &body.password)
        .await
        .map_err(|e| ApiError::Auth(e.to_string()))?
    else {
        state.throttle.failed(&body.username);
        // One message for every way this fails. Saying "no such user" would enumerate the
        // account list for anyone who asks.
        return Err(ApiError::Unauthorized(
            "that username and password do not match".to_string(),
        ));
    };
    state.throttle.succeeded(&body.username);

    let agent = headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok());
    let token = state
        .auth
        .create_session(&principal.principal_id, agent)
        .await?;
    tracing::info!(
        kind = "login",
        principal = %principal.principal_id,
        "session started"
    );

    let mut response = Json(Me {
        principal_id: Some(principal.principal_id),
        display_name: Some(principal.display_name),
        role: Some(principal.role),
    })
    .into_response();
    response.headers_mut().insert(
        auth::SET_COOKIE_HEADER,
        auth::set_cookie(&token, state.secure_cookies),
    );
    Ok(response)
}

/// End this session. Succeeds whether or not there was one, so a stale tab can always get clean.
async fn logout(State(state): State<AppState>, headers: axum::http::HeaderMap) -> Response {
    if let Some(token) = auth::token_from_headers(&headers)
        && let Err(error) = state.auth.revoke_session(&token).await
    {
        tracing::warn!(%error, "could not revoke a session");
    }
    let mut response = StatusCode::NO_CONTENT.into_response();
    response.headers_mut().insert(
        auth::SET_COOKIE_HEADER,
        auth::clear_cookie(state.secure_cookies),
    );
    response
}

/// Who the caller is. Always 200: "nobody" is an answer, not a failure.
async fn whoami(caller: Caller) -> Json<Me> {
    Json(match caller {
        Caller::Anonymous => Me {
            principal_id: None,
            display_name: None,
            role: None,
        },
        Caller::Session(p) => Me {
            principal_id: Some(p.principal_id),
            display_name: Some(p.display_name),
            role: Some(p.role),
        },
    })
}

/// The projects this caller may see.
///
/// Filtered, not merely marked: the list of repositories an instance watches is itself worth
/// keeping back, and a private project that appears greyed out has still told a stranger it
/// exists.
async fn list_projects(State(state): State<AppState>, caller: Caller) -> Json<Vec<ProjectView>> {
    let signed_in = matches!(caller, Caller::Session(_));
    Json(
        state
            .projects
            .values()
            .filter(|p| caller.may(p.visibility, Access::Read))
            .map(|p| ProjectView {
                slug: p.slug.clone(),
                dir: signed_in.then(|| p.dir.display().to_string()),
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
    /// Whether `status` is one a run stops at. Answered here because [`RunStatus`] is where the
    /// classification lives and is exhaustive over it — a client keeping its own list has to be
    /// told about every new status, and is wrong until it is. Open-world in the same direction the
    /// enum is: a status this build does not know reads as still in flight, which shows a stale run
    /// rather than declaring a live one finished.
    terminal: bool,
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
    /// The run's own record of its registry: every stage it could execute, and the node whose work
    /// each is. What folds a member's events into its box, and what a box's feed is filtered by.
    ///
    /// Shipped alongside `nodes` rather than inside them because it is a property of the RUN, not
    /// of what has finished. `nodes` is derived from checkpoints, so a box whose stage is executing
    /// right now is not in it yet — and that is precisely the window an operator reaches for Stop
    /// in. Read from `nodes`, the mapping is empty exactly then, the client draws
    /// `context_distillation` as its own box, and the Stop it offers is addressed to a name the
    /// runtime never polls.
    ///
    /// Empty for a run recorded before the registry travelled with it, where every node is exactly
    /// its own stage — which is what such a run's nodes were.
    stages: Vec<StageMembership>,
    worktree: Option<WorktreeView>,
    /// The pull request the run opened, if it opened one. Absent for comment-only runs, for runs
    /// that published nothing, and for runs that never reached the publisher.
    pull_request: Option<PullRequestView>,
    /// What the operator has asked of this run — what the controls should show.
    ///
    /// What was asked for, not what the run has done about it: a node acts at its next turn
    /// boundary, which can be a minute away, and a button that sprang back until the run noticed
    /// would read as the click having been lost.
    control: ControlView,
}

/// One stage of a run's registry as the dashboard needs it: what it is called, and which box its
/// records are drawn in.
///
/// A projection of what the run recorded, not the recorded row. `shape_json` gains fields as a run
/// records more about a stage — its governance identity, the session scope it declared — and none
/// of that is the dashboard's business. Shipping the stored row would put each of them on the wire
/// as it arrived and make every storage change a client change.
#[derive(Debug, Serialize)]
struct StageMembership {
    id: String,
    node: String,
}

/// The implementer's worktree — the reviewable deliverable, kept on `converged` and
/// `max_iterations_reached` and removed by a hard error or `ratatoskr clean`. Reported separately
/// from node state on purpose: a converged run's worktree is usually still on disk.
#[derive(Debug, Serialize)]
struct WorktreeView {
    path: String,
    exists: bool,
}

/// A pull request a run opened. The publisher's `url` is only a PR for `action` `pull_request`
/// or `both`; `#number` is the URL's last path segment (`/pull/139` → `139`).
#[derive(Debug, Serialize)]
struct PullRequestView {
    number: u64,
    url: String,
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
    caller: Caller,
    AxumPath(project): AxumPath<String>,
) -> Result<Json<Vec<RunSummary>>, ApiError> {
    let runs = state
        .project(&project, &caller, Access::Read)?
        .store
        .list_runs()
        .await?;
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
    caller: Caller,
    AxumPath((project, run_id)): AxumPath<(String, String)>,
) -> Result<Json<RunDetail>, ApiError> {
    let (found, run_id) = state
        .project_and_run(&project, &run_id, &caller, Access::Read)
        .await?;
    let store = &found.store;
    let config_path = found.config_path.clone();
    let run = store.run(&run_id).await?;
    let checkpoints = store.checkpoints_for_run(&run_id).await?;
    if run.is_none() && checkpoints.is_empty() {
        return Err(ApiError::NotFound(format!("no run {run_id}")));
    }

    let status = run.as_ref().map(|r| r.status.clone());
    // A run that happened here is drawn against the config as it stands. That is deliberate: a node
    // that has not run yet should say what it WILL run on, and the live file is that answer even
    // when it has drifted since the run started.
    //
    // A run that came from somewhere else has no such future here. Its nodes will never run on this
    // machine's routes, so reading them made a foreign run claim models it never had — or report
    // `planned: None`, "this node has nowhere to run", when the far machine routed it perfectly
    // well. Its own recorded config is the only honest answer, and `origin` is what separates the
    // two: `None` for a run produced here.
    //
    // Failing to read a config costs the planned facts and nothing else — a dashboard that refused
    // to show a run because its config moved would be worse than one that shows the run without
    // them. But for an imported run, recording NOTHING and recording something unreadable are not
    // the same failure. Nothing makes no claim, so the live file is a guess that displaces no
    // answer. A recording this build cannot parse — a corrupted bundle, a config shape from a
    // newer one — is an answer that exists and was not read, and substituting another machine's
    // for it recreates exactly the false attribution above. That one reports no planned facts.
    let live = || {
        std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|t| ratatoskr_core::RatatoskrConfig::from_toml_str(&t).ok())
    };
    let config = match run.as_ref().filter(|run| run.origin.is_some()) {
        Some(imported) => match imported.config_json.as_deref() {
            Some(recorded) => {
                serde_json::from_str::<ratatoskr_core::RatatoskrConfig>(recorded).ok()
            }
            None => live(),
        },
        None => live(),
    };
    let shape_json = run.as_ref().and_then(|r| r.shape_json.as_deref());
    let recorded = ratatoskr_core::shape::recorded(shape_json);
    let nodes = pipeline::derive_from(status.as_deref(), &checkpoints, config.as_ref(), &recorded);
    let stages = recorded
        .stages
        .iter()
        .map(|stage| StageMembership {
            id: stage.id.clone(),
            node: stage.node.clone(),
        })
        .collect();
    let last_activity = checkpoints
        .iter()
        .map(|c| c.created_at.as_str())
        .chain(run.as_ref().map(|r| r.updated_at.as_str()))
        .max()
        .map(str::to_string);

    let provider_key = ProviderPauseKey::new(&found.slug, &run_id);
    let run_key = RunKey::new(&found.slug, &run_id);
    let exited = state.provider_pauses.is_exited(&provider_key).await?;
    let mut control = state
        .control
        .lock()
        .expect("control mutex poisoned")
        .get(&run_key)
        .map(|state| state.control.view())
        .unwrap_or_default();
    if exited {
        state
            .control
            .lock()
            .expect("control mutex poisoned")
            .remove(&run_key);
        control = ControlView::default();
    } else {
        control.paused |= state.provider_pauses.is_holding(&provider_key).await?;
        append_durable_stops(
            &mut control,
            state.provider_pauses.stopped_nodes(&provider_key).await?,
        );
    }

    Ok(Json(RunDetail {
        control,
        run_id,
        terminal: pipeline::is_terminal(status.as_deref()),
        status,
        issue_id: run.as_ref().and_then(|r| r.issue_id.clone()),
        updated_at: run.as_ref().map(|r| r.updated_at.clone()),
        issue: issue_text(&checkpoints),
        last_activity,
        nodes,
        stages,
        worktree: worktree_view(&checkpoints),
        pull_request: pull_request_view(&checkpoints),
    }))
}

async fn node_checkpoints(
    State(state): State<AppState>,
    caller: Caller,
    AxumPath((project, run_id, node)): AxumPath<(String, String, String)>,
) -> Result<Json<Vec<CheckpointView>>, ApiError> {
    let (found, run_id) = state
        .project_and_run(&project, &run_id, &caller, Access::Read)
        .await?;
    let all = found.store.checkpoints_for_run(&run_id).await?;
    // Asked with the name of a *box*, which is what the graph draws and what a selection names. Its
    // records are its members' as well as its own, so the run's registry resolves the name rather
    // than matching it: a box whose members ran directly has no row under its own name at all, and
    // matching would 404 a box the graph draws as done.
    let shape_json = found.store.run(&run_id).await?.and_then(|r| r.shape_json);
    let recorded = ratatoskr_core::shape::recorded(shape_json.as_deref());
    let registry = recorded.index();
    // Indexed, not scanned: this is asked once per checkpoint, and an imported run brings both its
    // registry and its log with it — a box of many members against a long log is the same quadratic
    // exposure the derivation's own lookups are indexed against.
    let names: std::collections::HashSet<&str> = registry.records_of(&node).into_iter().collect();
    // Every checkpoint, not just the latest: the implementer writes one per converge iteration and
    // the diagnostic progression between them is the interesting part.
    let views: Vec<CheckpointView> = all
        .into_iter()
        .filter(|c| names.contains(&c.node_name.as_str()))
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

/// The pull request the latest `publisher` checkpoint opened, if any.
///
/// A checkpoint records `pull_request_url` separately from `comment_url`, and the action keeps
/// comment-only checkpoints out of the dashboard. The parser accepts a URL only when it has
/// GitHub's pull-request path shape, so a malformed field is never used as an anchor verbatim.
fn pull_request_view(checkpoints: &[Checkpoint]) -> Option<PullRequestView> {
    let raw = checkpoints
        .iter()
        .rev()
        .find(|c| c.node_name == "publisher")?
        .output_json
        .as_str();
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    match value.get("action").and_then(|v| v.as_str()) {
        Some("pull_request") | Some("both") => {}
        _ => return None,
    }
    value
        .get("pull_request_url")
        .and_then(serde_json::Value::as_str)
        .and_then(pull_request_url)
}

/// Extract one GitHub pull-request URL from a checkpoint field.
///
/// The field is meant to hold the URL alone, but a model writes it. Splitting on whitespace
/// recovers the URL from a labelled or otherwise chatty value without letting the dashboard use
/// the surrounding text as its `href`.
fn pull_request_url(value: &str) -> Option<PullRequestView> {
    value
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|c| matches!(c, '"' | '\'' | '(' | ')' | '[' | ']' | ',' | '.'))
        })
        .find_map(parse_github_pull_request_url)
}

fn parse_github_pull_request_url(url: &str) -> Option<PullRequestView> {
    let mut segments = url.strip_prefix("https://github.com/")?.split('/');
    let owner = segments.next()?;
    let repository = segments.next()?;
    if owner.is_empty() || repository.is_empty() || segments.next()? != "pull" {
        return None;
    }
    let number_and_suffix = segments.next()?;
    if segments.next().is_some() {
        return None;
    }

    let number_end = number_and_suffix
        .find(['?', '#'])
        .unwrap_or(number_and_suffix.len());
    let (digits, suffix) = number_and_suffix.split_at(number_end);
    if digits.is_empty() || (!suffix.is_empty() && !suffix.starts_with(['?', '#'])) {
        return None;
    }
    let number = digits.parse().ok()?;
    Some(PullRequestView {
        number,
        url: url.to_string(),
    })
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
    caller: Caller,
    AxumPath(project): AxumPath<String>,
    Json(body): Json<StartRun>,
) -> Result<(StatusCode, Json<StartedRun>), ApiError> {
    let run_id = state
        .project(&project, &caller, Access::Act)?
        .launcher
        .spawn(&body.issue)?;
    // Who, not just what: on an instance more than one person can reach, "a run started" is not
    // an answerable audit line.
    tracing::info!(
        kind = "run_started",
        run_id = %run_id,
        principal = caller.id(),
        "started run from the dashboard"
    );
    Ok((StatusCode::ACCEPTED, Json(StartedRun { run_id })))
}

/// Stream a run's activity as it happens.
///
/// Checkpoints only tell you a node *finished*; this is what it is doing in between. The stream
/// replays the run's recent history on connect, then follows the log, and ends when the client
/// disconnects — the tailing task is owned by the channel and dies with it.
/// Every event a run produced, for moving through it after the fact.
async fn run_history(
    State(state): State<AppState>,
    caller: Caller,
    AxumPath((project, run_id)): AxumPath<(String, String)>,
) -> Result<Json<Vec<events::LiveEvent>>, ApiError> {
    let (project, run_id) = state
        .project_and_run(&project, &run_id, &caller, Access::Read)
        .await?;
    Ok(Json(
        events::history(&project.store, &project.log_dir, &run_id).await,
    ))
}

async fn run_events(
    State(state): State<AppState>,
    caller: Caller,
    AxumPath((project, run_id)): AxumPath<(String, String)>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError>
{
    let (found, run_id) = state
        .project_and_run(&project, &run_id, &caller, Access::Read)
        .await?;
    let dir = found.log_dir.clone();
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
    caller: Caller,
    AxumPath(question_id): AxumPath<String>,
    Json(body): Json<Answer>,
) -> Result<StatusCode, ApiError> {
    // Checked against the role directly rather than through a project, because a question is
    // keyed by id alone — it carries no project in its path. An answer goes into the prompt of an
    // agent that holds tools, which makes this every bit as much "acting" as starting a run.
    if !caller.role().is_some_and(|r| r >= Role::Operator) {
        return Err(caller.denied("answer a clarification"));
    }
    tracing::info!(
        kind = "clarification_answered",
        question_id = %question_id,
        principal = caller.id(),
        "answered a node's question"
    );
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
    /// Nobody is logged in and this needed someone to be. Distinct from `Forbidden` because the
    /// dashboard renders them differently: this one opens the login form.
    #[error("{0}")]
    Unauthorized(String),
    /// Somebody is logged in and still may not. Logging in again would not help, so the dashboard
    /// must not offer it.
    #[error("{0}")]
    Forbidden(String),
    #[error("too many failed sign-in attempts for that account — try again later")]
    TooManyAttempts,
    #[error("{0}")]
    Auth(String),
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
            ApiError::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            ApiError::Forbidden(_) => StatusCode::FORBIDDEN,
            ApiError::TooManyAttempts => StatusCode::TOO_MANY_REQUESTS,
            ApiError::Auth(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (code, Json(serde_json::json!({ "error": self.to_string() }))).into_response()
    }
}

/// End-to-end authorization, driven through the real router.
///
/// Separate from the unit tests on [`auth::Caller`] because they answer different questions. Those
/// prove the predicate is right; these prove a route actually calls it. A guard that is correct
/// and unreachable is the failure mode worth testing for, and it is invisible from either side
/// alone.
#[cfg(test)]
mod access_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use ratatoskr_core::auth::Visibility;
    use ratatoskr_store::auth::AuthStore;
    use tower::ServiceExt as _;

    #[test]
    fn the_internal_listener_requires_a_loopback_address() {
        for address in ["127.0.0.1:0", "[::1]:0"] {
            let address: SocketAddr = address.parse().expect("a socket address");
            assert!(
                require_loopback_internal_address(address).is_ok(),
                "{address} is loopback"
            );
        }

        for address in ["0.0.0.0:0", "[::]:0", "192.0.2.1:0"] {
            let address: SocketAddr = address.parse().expect("a socket address");
            assert!(
                matches!(
                    require_loopback_internal_address(address),
                    Err(ServeError::InternalAddressNotLoopback(found)) if found == address
                ),
                "{address} must not expose unauthenticated control"
            );
        }
    }

    /// One project of each visibility, and an empty identity database.
    async fn state() -> AppState {
        state_with_config(PathBuf::from("ratatoskr.toml")).await
    }

    /// [`state`], with the projects' `ratatoskr.toml` at a path of the caller's choosing — for the
    /// cases that turn on what the reader's own config says, rather than on there being none.
    async fn state_with_config(config_path: PathBuf) -> AppState {
        let mut projects = BTreeMap::new();
        for (slug, visibility) in [("open", Visibility::Public), ("shut", Visibility::Private)] {
            projects.insert(
                slug.to_string(),
                Project {
                    slug: slug.to_string(),
                    repository: Some(format!("cq27-dev/{slug}")),
                    dir: PathBuf::from("/tmp").join(slug),
                    visibility,
                    config_path: config_path.clone(),
                    store: ratatoskr_store::Store::open_in_memory().expect("in-memory store"),
                    log_dir: PathBuf::from("/tmp").join(slug).join("logs"),
                    launcher: Arc::new(launch::Launcher::new(
                        Path::new("/tmp"),
                        Path::new("ratatoskr.toml"),
                        1,
                        "http://127.0.0.1:1",
                        slug,
                    )),
                },
            );
        }
        AppState {
            projects: Arc::new(projects),
            desk: Arc::new(Desk::default()),
            control: Arc::default(),
            provider_pauses: ProviderPauseStore::open_in_memory()
                .expect("in-memory provider pause store"),
            auth: AuthStore::open_in_memory().expect("in-memory identity database"),
            throttle: Arc::new(auth::LoginThrottle::default()),
            secure_cookies: false,
            github: None,
        }
    }

    /// A session cookie for a principal of `role`, and the state that knows about it.
    async fn signed_in(state: &AppState, role: Role) -> String {
        let principal = state
            .auth
            .create_local("kk", "hunter2", "KK", role)
            .await
            .expect("a new principal");
        let token = state
            .auth
            .create_session(&principal.principal_id, None)
            .await
            .expect("a session");
        format!("{}={token}", auth::COOKIE_NAME_INSECURE)
    }

    async fn send(state: AppState, request: Request<Body>) -> StatusCode {
        router(state, None)
            .oneshot(request)
            .await
            .expect("the router to answer")
            .status()
    }

    fn get(path: &str, cookie: Option<&str>) -> Request<Body> {
        let builder = Request::builder().uri(path);
        let builder = match cookie {
            Some(c) => builder.header("cookie", c),
            None => builder,
        };
        builder.body(Body::empty()).expect("a valid request")
    }

    fn post(path: &str, json: &str, cookie: Option<&str>) -> Request<Body> {
        let builder = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json");
        let builder = match cookie {
            Some(c) => builder.header("cookie", c),
            None => builder,
        };
        builder
            .body(Body::from(json.to_string()))
            .expect("a valid request")
    }

    /// The controls, end to end: an operator issues a command on the public API and a run process
    /// picks it up on the internal one.
    mod control {
        use super::*;

        const PROJECT: &str = "open";

        /// Ask the internal rendezvous what a node should do, the way a run process does.
        async fn ask(state: AppState, run_id: &str, node: &str) -> Control {
            ask_for(state, PROJECT, run_id, node).await
        }

        async fn ask_for(state: AppState, project: &str, run_id: &str, node: &str) -> Control {
            let body = serde_json::json!({
                "project": project,
                "run_id": run_id,
                "node": node,
            })
            .to_string();
            let response = internal_router(state)
                .oneshot(post("/internal/control", &body, None))
                .await
                .expect("the rendezvous to answer");
            let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .expect("a body");
            serde_json::from_slice(&bytes).expect("a control")
        }

        /// Ask while the child has already persisted a provider pause. Its stable identity lets a
        /// restarted dashboard reconstruct every concurrent hold without changing ordinary polls.
        async fn ask_while_provider_paused(
            state: AppState,
            run_id: &str,
            node: &str,
            waiter: &str,
        ) -> Control {
            let body = serde_json::json!({
                "project": PROJECT,
                "run_id": run_id,
                "node": node,
                "provider_pause_waiter": waiter,
            })
            .to_string();
            let response = internal_router(state)
                .oneshot(post("/internal/control", &body, None))
                .await
                .expect("the rendezvous to answer");
            let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .expect("a body");
            serde_json::from_slice(&bytes).expect("a control")
        }

        /// A fresh process against the same projects. Unlike `Clone`, it discards every volatile
        /// rendezvous map while retaining the store handles and identity database a real restart
        /// reopens.
        fn restarted(state: &AppState) -> AppState {
            AppState {
                projects: Arc::clone(&state.projects),
                desk: Arc::new(Desk::default()),
                control: Arc::default(),
                provider_pauses: state.provider_pauses.clone(),
                auth: state.auth.clone(),
                throttle: Arc::clone(&state.throttle),
                github: state.github.clone(),
                secure_cookies: state.secure_cookies,
            }
        }

        /// Record the automatic pause a run asks for after a provider failure.
        async fn automatic_pause(
            state: AppState,
            run_id: &str,
            waiter: &str,
        ) -> ProviderPauseDirective {
            automatic_pause_reply(state, run_id, waiter, None)
                .await
                .directive
        }

        /// Record a provider pause with the latest resume generation this child acknowledged.
        async fn automatic_pause_reply(
            state: AppState,
            run_id: &str,
            waiter: &str,
            known_provider_resume_generation: Option<i64>,
        ) -> ProviderPauseReply {
            let body = serde_json::json!({
                "project": PROJECT,
                "run_id": run_id,
                "node": waiter,
                "provider_pause_waiter": waiter,
                "known_provider_resume_generation": known_provider_resume_generation,
            })
            .to_string();
            let response = internal_router(state)
                .oneshot(post("/internal/control/pause", &body, None))
                .await
                .expect("the rendezvous to accept the pause");
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .expect("a body");
            serde_json::from_slice(&bytes).expect("a provider pause reply")
        }

        /// Confirm a received provider pause delivery, as the child does before it resumes.
        async fn acknowledge(
            state: AppState,
            run_id: &str,
            generation: i64,
            waiter: &str,
        ) -> ProviderPauseDirective {
            let body = serde_json::json!({
                "project": PROJECT,
                "run_id": run_id,
                "node": waiter,
                "provider_pause_waiter": waiter,
                "provider_pause_generation": generation,
            })
            .to_string();
            let response = internal_router(state)
                .oneshot(post("/internal/control/pause/ack", &body, None))
                .await
                .expect("the rendezvous to accept the acknowledgement");
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .expect("an acknowledgement reply body");
            serde_json::from_slice::<ProviderPauseAcknowledgementReply>(&bytes)
                .expect("a provider pause acknowledgement reply")
                .directive
        }

        /// Issue a command as an operator.
        async fn command(state: AppState, cookie: &str, json: &str) -> StatusCode {
            command_for(state, cookie, PROJECT, json).await
        }

        async fn command_for(
            state: AppState,
            cookie: &str,
            project: &str,
            json: &str,
        ) -> StatusCode {
            send(
                state,
                post(
                    &format!("/api/projects/{project}/runs/r1/control"),
                    json,
                    Some(cookie),
                ),
            )
            .await
        }

        #[tokio::test]
        async fn a_pause_reaches_the_run_that_asks_for_it() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;
            assert_eq!(
                command(state.clone(), &cookie, r#"{"command":"pause"}"#).await,
                StatusCode::OK
            );
            assert_eq!(
                ask(state.clone(), "r1", "analyst").await.directive,
                ratatoskr_core::Directive::Hold
            );

            command(state.clone(), &cookie, r#"{"command":"resume"}"#).await;
            assert_eq!(
                ask(state, "r1", "analyst").await.directive,
                ratatoskr_core::Directive::Continue
            );
        }

        #[tokio::test]
        async fn an_automatic_pause_uses_the_same_continue_control_as_the_dashboard() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;
            assert_eq!(
                automatic_pause(state.clone(), "r1", "analyst").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                ask_while_provider_paused(state.clone(), "r1", "analyst", "analyst")
                    .await
                    .directive,
                ratatoskr_core::Directive::Hold
            );

            // The public control is the button's path. Once it continues, the automatic hold is
            // gone too, so its icon and its next internal poll cannot disagree.
            assert_eq!(
                command(state.clone(), &cookie, r#"{"command":"resume"}"#).await,
                StatusCode::OK
            );
            assert_eq!(
                ask_while_provider_paused(state.clone(), "r1", "analyst", "analyst")
                    .await
                    .directive,
                ratatoskr_core::Directive::Continue
            );
            assert_eq!(
                acknowledge(state, "r1", 1, "analyst").await,
                ProviderPauseDirective::Continue
            );
        }

        #[tokio::test]
        async fn a_stop_that_precedes_provider_pause_registration_remains_durable() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;

            assert_eq!(
                command(
                    state.clone(),
                    &cookie,
                    r#"{"command":"stop","node":"analyst"}"#
                )
                .await,
                StatusCode::OK
            );
            assert_eq!(
                automatic_pause(state.clone(), "r1", "analyst").await,
                ProviderPauseDirective::Stop,
                "a provider waiter arriving after Stop must not register as Hold"
            );
            assert_eq!(
                acknowledge(state, "r1", 1, "analyst").await,
                ProviderPauseDirective::Stop,
                "the acknowledgement keeps Stop authoritative over provider retry"
            );
        }

        #[tokio::test]
        async fn one_resume_releases_every_concurrent_provider_pause() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;
            // `RedTeam::run_and_author` joins two provider calls. They can both pause before the
            // dashboard is next observed, and each needs the same one resume.
            assert_eq!(
                automatic_pause(state.clone(), "r1", "red-team").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                automatic_pause(state.clone(), "r1", "author").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                command(state.clone(), &cookie, r#"{"command":"resume"}"#).await,
                StatusCode::OK
            );

            assert_eq!(
                ask_while_provider_paused(state.clone(), "r1", "redteam", "red-team")
                    .await
                    .directive,
                ratatoskr_core::Directive::Continue
            );
            assert_eq!(
                acknowledge(state.clone(), "r1", 1, "red-team").await,
                ProviderPauseDirective::Continue
            );
            assert_eq!(
                ask_while_provider_paused(state.clone(), "r1", "redteam", "author")
                    .await
                    .directive,
                ratatoskr_core::Directive::Continue,
                "the second provider waiter consumes the same resume, not a new hold"
            );
            assert_eq!(
                acknowledge(state, "r1", 1, "author").await,
                ProviderPauseDirective::Continue
            );
        }

        #[tokio::test]
        async fn a_provider_pause_survives_a_dashboard_restart_until_an_operator_resumes() {
            let before_restart = state().await;
            let cookie = signed_in(&before_restart, Role::Operator).await;
            assert_eq!(
                automatic_pause(before_restart.clone(), "r1", "analyst").await,
                ProviderPauseDirective::Hold
            );

            let after_restart = restarted(&before_restart);
            assert_eq!(
                ask_while_provider_paused(after_restart.clone(), "r1", "analyst", "analyst")
                    .await
                    .directive,
                ratatoskr_core::Directive::Hold,
                "a waiting child re-establishes its acknowledged provider pause"
            );

            assert_eq!(
                command(after_restart.clone(), &cookie, r#"{"command":"resume"}"#).await,
                StatusCode::OK
            );
            assert_eq!(
                ask_while_provider_paused(after_restart.clone(), "r1", "analyst", "analyst")
                    .await
                    .directive,
                ratatoskr_core::Directive::Continue,
                "only the operator's resume releases the reconstructed hold"
            );
            assert_eq!(
                acknowledge(after_restart, "r1", 1, "analyst").await,
                ProviderPauseDirective::Continue
            );
        }

        #[tokio::test]
        async fn a_provider_stop_survives_a_dashboard_restart_until_acknowledged() {
            let before_restart = state().await;
            let cookie = signed_in(&before_restart, Role::Operator).await;
            assert_eq!(
                automatic_pause(before_restart.clone(), "r1", "analyst").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                command(
                    before_restart.clone(),
                    &cookie,
                    r#"{"command":"stop","node":"analyst"}"#
                )
                .await,
                StatusCode::OK
            );

            let after_restart = restarted(&before_restart);
            assert_eq!(
                ask_while_provider_paused(after_restart.clone(), "r1", "analyst", "analyst")
                    .await
                    .directive,
                ratatoskr_core::Directive::Stop,
                "a restarted dashboard delivers the selected node's durable Stop"
            );
            assert_eq!(
                acknowledge(after_restart.clone(), "r1", 1, "analyst").await,
                ProviderPauseDirective::Stop
            );
            assert_eq!(
                ask(after_restart.clone(), "r1", "analyst").await.directive,
                ratatoskr_core::Directive::Stop,
                "the parked node keeps receiving its durable Stop after acknowledging it"
            );
            assert_eq!(
                command(
                    after_restart.clone(),
                    &cookie,
                    r#"{"command":"start","node":"analyst"}"#
                )
                .await,
                StatusCode::OK
            );
            assert_eq!(
                ask(after_restart, "r1", "analyst").await.directive,
                ratatoskr_core::Directive::Continue,
                "only Start clears the durable Stop"
            );
        }

        #[tokio::test]
        async fn one_resume_releases_every_provider_waiter_reconstructed_after_a_restart() {
            let before_restart = state().await;
            let cookie = signed_in(&before_restart, Role::Operator).await;
            assert_eq!(
                automatic_pause(before_restart.clone(), "r1", "red-team").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                automatic_pause(before_restart.clone(), "r1", "author").await,
                ProviderPauseDirective::Hold
            );

            let after_restart = restarted(&before_restart);
            for (node, waiter) in [("redteam", "red-team"), ("author", "author")] {
                assert_eq!(
                    ask_while_provider_paused(after_restart.clone(), "r1", node, waiter)
                        .await
                        .directive,
                    ratatoskr_core::Directive::Hold,
                    "each returning child re-establishes its own pause claim"
                );
            }

            assert_eq!(
                command(after_restart.clone(), &cookie, r#"{"command":"resume"}"#).await,
                StatusCode::OK
            );
            for (node, waiter) in [("redteam", "red-team"), ("author", "author")] {
                assert_eq!(
                    ask_while_provider_paused(after_restart.clone(), "r1", node, waiter)
                        .await
                        .directive,
                    ratatoskr_core::Directive::Continue,
                    "one resume reaches every provider waiter reconstructed after the restart"
                );
                assert_eq!(
                    acknowledge(after_restart.clone(), "r1", 1, waiter).await,
                    ProviderPauseDirective::Continue
                );
            }
        }

        #[tokio::test]
        async fn a_lost_continue_reply_cannot_rearm_a_provider_pause_after_a_restart() {
            let before_reply_loss = state().await;
            let cookie = signed_in(&before_reply_loss, Role::Operator).await;
            assert_eq!(
                automatic_pause(before_reply_loss.clone(), "r1", "analyst").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                command(
                    before_reply_loss.clone(),
                    &cookie,
                    r#"{"command":"resume"}"#
                )
                .await,
                StatusCode::OK
            );

            // The dashboard processed this reply, but the child never received it and therefore
            // cannot acknowledge it yet.
            assert_eq!(
                ask_while_provider_paused(before_reply_loss.clone(), "r1", "analyst", "analyst")
                    .await
                    .directive,
                ratatoskr_core::Directive::Continue
            );

            let after_restart = restarted(&before_reply_loss);
            assert_eq!(
                ask_while_provider_paused(after_restart.clone(), "r1", "analyst", "analyst")
                    .await
                    .directive,
                ratatoskr_core::Directive::Continue,
                "the durable resume tombstone survives until this child acknowledges it"
            );
            assert_eq!(
                acknowledge(after_restart.clone(), "r1", 1, "analyst").await,
                ProviderPauseDirective::Continue
            );
            assert_eq!(
                automatic_pause_reply(after_restart, "r1", "next", Some(1))
                    .await
                    .directive,
                ProviderPauseDirective::Hold,
                "a child that received the acknowledgement records its generation before a later provider failure pauses anew"
            );
        }

        #[tokio::test]
        async fn a_stop_after_a_lost_acknowledgement_reply_is_returned_by_the_retry() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;
            assert_eq!(
                automatic_pause(state.clone(), "r1", "analyst").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                command(state.clone(), &cookie, r#"{"command":"resume"}"#).await,
                StatusCode::OK
            );
            assert_eq!(
                ask_while_provider_paused(state.clone(), "r1", "analyst", "analyst")
                    .await
                    .directive,
                ratatoskr_core::Directive::Continue
            );

            assert_eq!(
                acknowledge(state.clone(), "r1", 1, "analyst").await,
                ProviderPauseDirective::Continue,
                "the child loses this committed acknowledgement reply"
            );
            assert_eq!(
                command(
                    state.clone(),
                    &cookie,
                    r#"{"command":"stop","node":"analyst"}"#
                )
                .await,
                StatusCode::OK
            );
            assert_eq!(
                acknowledge(state, "r1", 1, "analyst").await,
                ProviderPauseDirective::Stop,
                "retrying the lost acknowledgement delivers the intervening Stop"
            );
        }

        #[tokio::test]
        async fn a_late_provider_waiter_consumes_the_existing_resume_tombstone() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;
            assert_eq!(
                automatic_pause(state.clone(), "r1", "first").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                command(state.clone(), &cookie, r#"{"command":"resume"}"#).await,
                StatusCode::OK
            );

            assert_eq!(
                automatic_pause(state.clone(), "r1", "late").await,
                ProviderPauseDirective::Continue,
                "a concurrent pause request that arrives after Resume must not re-arm Hold"
            );
            for (node, waiter) in [("first", "first"), ("late", "late")] {
                assert_eq!(
                    ask_while_provider_paused(state.clone(), "r1", node, waiter)
                        .await
                        .directive,
                    ratatoskr_core::Directive::Continue
                );
                assert_eq!(
                    acknowledge(state.clone(), "r1", 1, waiter).await,
                    ProviderPauseDirective::Continue
                );
            }
        }

        #[tokio::test]
        async fn a_new_provider_failure_after_acknowledgement_uses_a_fresh_generation() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;
            assert_eq!(
                automatic_pause(state.clone(), "r1", "first").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                automatic_pause(state.clone(), "r1", "second").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                command(state.clone(), &cookie, r#"{"command":"resume"}"#).await,
                StatusCode::OK
            );
            assert_eq!(
                ask_while_provider_paused(state.clone(), "r1", "first", "first")
                    .await
                    .directive,
                ratatoskr_core::Directive::Continue
            );
            assert_eq!(
                acknowledge(state.clone(), "r1", 1, "first").await,
                ProviderPauseDirective::Continue
            );

            assert_eq!(
                automatic_pause_reply(state.clone(), "r1", "fresh", Some(1)).await,
                ProviderPauseReply {
                    directive: ProviderPauseDirective::Hold,
                    generation: 2,
                },
                "a fresh failure after generation one was acknowledged needs its own resume"
            );
            assert_eq!(
                ask_while_provider_paused(state.clone(), "r1", "second", "second")
                    .await
                    .directive,
                ratatoskr_core::Directive::Continue,
                "a late generation-one waiter still receives the prior resume"
            );
            assert_eq!(
                acknowledge(state.clone(), "r1", 1, "second").await,
                ProviderPauseDirective::Continue
            );
            assert_eq!(
                ask_while_provider_paused(state.clone(), "r1", "fresh", "fresh")
                    .await
                    .directive,
                ratatoskr_core::Directive::Hold
            );
        }

        #[tokio::test]
        async fn a_stopped_provider_waiter_keeps_the_other_nodes_held_until_acknowledged() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;
            assert_eq!(
                automatic_pause(state.clone(), "r1", "analyst").await,
                ProviderPauseDirective::Hold
            );
            assert_eq!(
                command(
                    state.clone(),
                    &cookie,
                    r#"{"command":"stop","node":"analyst"}"#
                )
                .await,
                StatusCode::OK
            );
            assert_eq!(
                ask_while_provider_paused(state.clone(), "r1", "analyst", "analyst")
                    .await
                    .directive,
                ratatoskr_core::Directive::Stop
            );
            assert_eq!(
                ask(state.clone(), "r1", "implementer").await.directive,
                ratatoskr_core::Directive::Hold,
                "Stop has not yet been acknowledged by the waiting child"
            );
            assert_eq!(
                acknowledge(state.clone(), "r1", 1, "analyst").await,
                ProviderPauseDirective::Stop
            );
            assert_eq!(
                ask(state, "r1", "implementer").await.directive,
                ratatoskr_core::Directive::Continue
            );
        }

        #[tokio::test]
        async fn a_confirmed_exit_clears_an_unacknowledged_provider_pause() {
            let state = state().await;
            assert_eq!(
                automatic_pause(state.clone(), "r1", "analyst").await,
                ProviderPauseDirective::Hold
            );
            let key = ProviderPauseKey::new(PROJECT, "r1");
            assert!(state.provider_pauses.is_holding(&key).await.unwrap());
            state.provider_pauses.record_exit(&key).await.unwrap();

            assert_eq!(
                automatic_pause(state.clone(), "r1", "late").await,
                ProviderPauseDirective::Stop,
                "a late pause request after confirmed process exit cannot recreate a hold"
            );
            assert!(
                !state.provider_pauses.is_holding(&key).await.unwrap(),
                "a confirmed child exit clears its outstanding provider delivery"
            );
        }

        #[tokio::test]
        async fn controls_with_matching_run_ids_do_not_cross_projects() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;
            assert_eq!(
                command_for(state.clone(), &cookie, "open", r#"{"command":"pause"}"#).await,
                StatusCode::OK
            );
            assert_eq!(
                ask_for(state.clone(), "open", "r1", "analyst")
                    .await
                    .directive,
                ratatoskr_core::Directive::Hold
            );
            assert_eq!(
                ask_for(state, "shut", "r1", "analyst").await.directive,
                ratatoskr_core::Directive::Continue,
                "volatile control uses the same project/run identity as durable pause delivery"
            );
        }

        #[tokio::test]
        async fn text_reaches_the_node_it_names_exactly_once() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;
            command(
                state.clone(),
                &cookie,
                r#"{"command":"steer","node":"implementer","text":"use the existing helper"}"#,
            )
            .await;

            assert_eq!(
                ask(state.clone(), "r1", "implementer").await.steer,
                ["use the existing helper"]
            );
            // Twice would be the operator appearing to repeat themselves on every turn.
            assert!(
                ask(state.clone(), "r1", "implementer")
                    .await
                    .steer
                    .is_empty()
            );
            // And it was never for the other node in the fork.
            assert!(ask(state, "r1", "redteam").await.steer.is_empty());
        }

        #[tokio::test]
        async fn a_run_nobody_has_touched_is_told_to_carry_on() {
            // The common case by far: every node of every uncontrolled run asks this at every turn
            // boundary, and must not be held up by the answer.
            let state = state().await;
            assert_eq!(
                ask(state, "never-heard-of-it", "analyst").await,
                Control::carry_on()
            );
        }

        #[tokio::test]
        async fn a_viewer_cannot_touch_a_run() {
            // Reading a public project is open to anyone; stopping its run is not.
            let state = state().await;
            let viewer = signed_in(&state, Role::Viewer).await;
            assert_eq!(
                command(state.clone(), &viewer, r#"{"command":"pause"}"#).await,
                StatusCode::FORBIDDEN
            );
            assert_eq!(
                send(
                    state.clone(),
                    post(
                        "/api/projects/open/runs/r1/control",
                        r#"{"command":"pause"}"#,
                        None
                    ),
                )
                .await,
                StatusCode::UNAUTHORIZED
            );
            // Neither of them changed anything.
            assert_eq!(ask(state, "r1", "analyst").await, Control::carry_on());
        }

        #[tokio::test]
        async fn stopping_and_starting_a_node_is_one_round_trip_each() {
            let state = state().await;
            let cookie = signed_in(&state, Role::Operator).await;
            command(
                state.clone(),
                &cookie,
                r#"{"command":"stop","node":"implementer"}"#,
            )
            .await;
            assert_eq!(
                ask(state.clone(), "r1", "implementer").await.directive,
                ratatoskr_core::Directive::Stop
            );

            command(
                state.clone(),
                &cookie,
                r#"{"command":"start","node":"implementer"}"#,
            )
            .await;
            assert_eq!(
                ask(state, "r1", "implementer").await.directive,
                ratatoskr_core::Directive::Continue
            );
        }
    }

    #[tokio::test]
    async fn an_imported_runs_unreadable_recording_reports_nothing_rather_than_the_readers_config()
    {
        // Recording NOTHING and recording something unreadable are different failures. Nothing
        // makes no claim, so the reader's own config displaces no answer. A recording this build
        // cannot parse — a corrupted bundle, a shape from a newer one — is an answer that exists
        // and was not read, and showing this machine's models in its place is the false
        // attribution the whole change exists to remove, restored by the fallback.
        //
        // The reader's config is READABLE here, which is what makes the assertion mean anything:
        // there is a wrong answer available to fall through to.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-reader-config-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("ratatoskr.toml");
        // A whole config, because that is what one is: `from_toml_str` reads the shape the file
        // has, not a fragment of it.
        let mut ours = ratatoskr_core::RatatoskrConfig::default();
        ours.models.insert(
            "analyst".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "ours".to_string(),
                model: "our-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        std::fs::write(&config_path, toml::to_string(&ours).unwrap()).unwrap();
        let state = state_with_config(config_path).await;
        let store = &state.projects["open"].store;
        let shape = serde_json::json!({
            "nodes": [],
            "stages": [{"id": "analyst", "node": "analyst"}],
        })
        .to_string();
        for (id, recorded) in [("unreadable", "{\"models\":"), ("silent", "")] {
            store.upsert_run(id, None, "converged").await.unwrap();
            store
                .record_run_provenance(
                    id,
                    Some(recorded).filter(|raw| !raw.is_empty()),
                    None,
                    None,
                    Some(&shape),
                    None,
                )
                .await
                .unwrap();
            store
                .insert_checkpoint(ratatoskr_store::CheckpointWrite {
                    run_id: id,
                    node_name: "analyst",
                    output_json: "{}",
                    input_json: None,
                    iteration: None,
                    cause: None,
                    invocation: None,
                    telemetry: Default::default(),
                })
                .await
                .unwrap();
            store.set_origin(id, "someone-else").await.unwrap();
        }

        let planned = |id: &'static str| {
            let state = state.clone();
            async move {
                let response = router(state, None)
                    .oneshot(get(&format!("/api/projects/open/runs/{id}"), None))
                    .await
                    .expect("the router to answer");
                assert_eq!(response.status(), StatusCode::OK);
                let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
                    .await
                    .expect("a run detail body");
                let detail: serde_json::Value = serde_json::from_slice(&bytes).expect("run detail");
                detail["nodes"][0]["planned"].clone()
            }
        };

        assert!(
            planned("unreadable").await.is_null(),
            "a recording that could not be read was replaced by the reader's own config"
        );
        // And the absent case still falls through, which is the behaviour a run recorded before
        // provenance travelled with it has always had.
        assert_eq!(
            planned("silent").await["model"],
            serde_json::json!("ours/our-model")
        );
    }

    #[tokio::test]
    async fn an_imported_run_plans_against_the_config_it_recorded_not_the_readers() {
        // Two runs identical but for `origin`. The reader's own `ratatoskr.toml` does not exist in
        // this harness, so the live config contributes nothing — which is what makes the assertion
        // sharp in both directions: the imported run's planned route can only have come from its
        // own recording, and the local run's absence proves the recording was not read for it.
        let state = state().await;
        let store = &state.projects["open"].store;
        // Serialized the same way `record_provenance` writes it: the run's whole resolved config,
        // not a fragment, so what this reads back is what a real recording round-trips.
        let mut theirs = ratatoskr_core::RatatoskrConfig::default();
        theirs.models.insert(
            "analyst".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "elsewhere".to_string(),
                model: "their-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                // Not the default, so what the reader reports about the scope can only have come
                // from this route: the recorded declaration is resolved against the recorded
                // route, which is the other half of the same problem.
                session: ratatoskr_core::SessionScope::Reuse,
            },
        );
        let recorded = serde_json::to_string(&theirs).expect("a serializable config");
        let shape = serde_json::json!({
            "nodes": [],
            "stages": [{"id": "analyst", "node": "analyst"}],
        })
        .to_string();
        for id in ["imported", "local"] {
            store.upsert_run(id, None, "converged").await.unwrap();
            store
                .record_run_provenance(id, Some(&recorded), None, None, Some(&shape), None)
                .await
                .unwrap();
            store
                .insert_checkpoint(ratatoskr_store::CheckpointWrite {
                    run_id: id,
                    node_name: "analyst",
                    output_json: "{}",
                    input_json: None,
                    iteration: None,
                    cause: None,
                    invocation: None,
                    telemetry: Default::default(),
                })
                .await
                .unwrap();
        }
        store.set_origin("imported", "someone-else").await.unwrap();

        let planned = |id: &'static str| {
            let state = state.clone();
            async move {
                let response = router(state, None)
                    .oneshot(get(&format!("/api/projects/open/runs/{id}"), None))
                    .await
                    .expect("the router to answer");
                assert_eq!(response.status(), StatusCode::OK);
                let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
                    .await
                    .expect("a run detail body");
                let detail: serde_json::Value = serde_json::from_slice(&bytes).expect("run detail");
                detail["nodes"][0]["planned"].clone()
            }
        };

        assert_eq!(
            planned("imported").await["model"],
            serde_json::json!("elsewhere/their-model"),
            "a foreign run's boxes must not claim this machine's routes"
        );
        assert_eq!(
            planned("imported").await["sessions"],
            serde_json::json!(["reuse"]),
            "the recorded session declaration resolves against the recorded route"
        );
        assert!(
            planned("local").await.is_null(),
            "a local run is drawn against the config as it stands, not the one it recorded"
        );
    }

    #[tokio::test]
    async fn a_boxs_records_are_read_under_the_boxs_name_however_its_members_ran() {
        // The graph draws boxes, so a selection names a box — and `redteam` is not a stage id. Two
        // ways a box's rows are not under its own name: a member wrote them (a workflow invoked the
        // member directly, which is the same thing that makes the box read Done), or the aggregate
        // and the member both wrote and only one matches. Matching the name instead of resolving it
        // 404s the first case, on a box the graph draws as finished.
        let state = state().await;
        let store = &state.projects["open"].store;
        store
            .upsert_run("r9", None, "converged")
            .await
            .expect("a run row");
        let shape = serde_json::json!({
            "nodes": [],
            "stages": [{"id": "redteam_classifier", "node": "redteam"}],
        })
        .to_string();
        store
            .record_run_provenance("r9", None, None, None, Some(&shape), None)
            .await
            .expect("the registry to record");
        store
            .insert_checkpoint(ratatoskr_store::CheckpointWrite {
                run_id: "r9",
                node_name: "redteam_classifier",
                output_json: r#"{"verdict":"needs tests"}"#,
                input_json: None,
                iteration: None,
                cause: None,
                invocation: None,
                telemetry: Default::default(),
            })
            .await
            .expect("the member's checkpoint");

        let response = router(state, None)
            .oneshot(get("/api/projects/open/runs/r9/nodes/redteam", None))
            .await
            .expect("the router to answer");
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "the red team's box has records; only their name differs from the box's"
        );
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("a checkpoint body");
        let views: Vec<serde_json::Value> =
            serde_json::from_slice(&bytes).expect("checkpoint views");
        assert_eq!(
            views
                .iter()
                .map(|v| v["node_name"].as_str().unwrap_or_default())
                .collect::<Vec<_>>(),
            ["redteam_classifier"],
            "the member's row is the box's row"
        );
    }

    #[tokio::test]
    async fn a_stranger_reads_a_public_project() {
        let state = state().await;
        assert_eq!(
            send(state, get("/api/projects/open/runs", None)).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn a_stranger_cannot_tell_a_private_project_from_one_that_does_not_exist() {
        // Both 404. A 403 here would confirm the project exists, which is a fact about what this
        // machine works on that a stranger has no business learning.
        let state = state().await;
        let private = send(state.clone(), get("/api/projects/shut/runs", None)).await;
        let absent = send(state, get("/api/projects/nope/runs", None)).await;
        assert_eq!(private, StatusCode::NOT_FOUND);
        assert_eq!(absent, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn every_read_route_of_a_private_project_is_closed_to_a_stranger() {
        // Route by route, because the guard lives in a helper each one has to remember to call —
        // and "one handler forgot" is precisely the bug this catches.
        let state = state().await;
        for path in [
            "/api/projects/shut/runs",
            "/api/projects/shut/runs/r1",
            "/api/projects/shut/runs/r1/nodes/scout",
            "/api/projects/shut/runs/r1/history",
            "/api/projects/shut/runs/r1/events",
        ] {
            assert_eq!(
                send(state.clone(), get(path, None)).await,
                StatusCode::NOT_FOUND,
                "{path} leaked a private project"
            );
        }
    }

    #[tokio::test]
    async fn a_viewer_reads_a_private_project() {
        let state = state().await;
        let cookie = signed_in(&state, Role::Viewer).await;
        assert_eq!(
            send(state, get("/api/projects/shut/runs", Some(&cookie))).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn starting_a_run_needs_an_operator_even_on_a_public_project() {
        // The distinction the whole design rests on: public means readable, never runnable.
        let state = state().await;
        let anonymous = send(
            state.clone(),
            post("/api/projects/open/runs", r#"{"issue":"go"}"#, None),
        )
        .await;
        assert_eq!(anonymous, StatusCode::UNAUTHORIZED);

        let viewer = signed_in(&state, Role::Viewer).await;
        let refused = send(
            state,
            post(
                "/api/projects/open/runs",
                r#"{"issue":"go"}"#,
                Some(&viewer),
            ),
        )
        .await;
        // 403, not 401: they are logged in, so offering a login form would be a dead end.
        assert_eq!(refused, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn answering_a_clarification_needs_an_operator() {
        // Keyed by question id with no project in the path, so it cannot go through the project
        // guard and is checked separately — which is exactly why it needs its own test.
        let state = state().await;
        let anonymous = send(
            state.clone(),
            post("/api/clarifications/q1", r#"{"answer":"yes"}"#, None),
        )
        .await;
        assert_eq!(anonymous, StatusCode::UNAUTHORIZED);

        let viewer = signed_in(&state, Role::Viewer).await;
        assert_eq!(
            send(
                state,
                post(
                    "/api/clarifications/q1",
                    r#"{"answer":"yes"}"#,
                    Some(&viewer)
                )
            )
            .await,
            StatusCode::FORBIDDEN
        );
    }

    #[tokio::test]
    async fn an_operator_gets_past_the_guard_on_both_acting_routes() {
        let state = state().await;
        let cookie = signed_in(&state, Role::Operator).await;

        // An empty issue, deliberately. The handler's own validation rejects it *after* the
        // authorization check and *before* the launcher, so a 400 proves the operator got through
        // without this test starting a real child process — which a valid issue would do, against
        // the test binary, on every `cargo test`.
        assert_eq!(
            send(
                state.clone(),
                post(
                    "/api/projects/open/runs",
                    r#"{"issue":"  "}"#,
                    Some(&cookie)
                )
            )
            .await,
            StatusCode::BAD_REQUEST
        );

        // Likewise: no such question is pending, and `Gone` is the answer only a caller who was
        // allowed to ask can receive.
        assert_eq!(
            send(
                state,
                post(
                    "/api/clarifications/q1",
                    r#"{"answer":"yes"}"#,
                    Some(&cookie)
                )
            )
            .await,
            StatusCode::GONE
        );
    }

    #[tokio::test]
    async fn the_rendezvous_is_not_on_the_public_router() {
        // The reason it is a second listener rather than a guarded route: it is not reachable
        // here at all, so no future middleware mistake can expose it.
        //
        // 404 or 405, not one or the other: with the built UI mounted, unknown paths fall through
        // to the static-file service, which answers a POST with 405 rather than 404. Both mean the
        // handler is absent — a registered route would answer the body, not the method.
        let state = state().await;
        let status = send(
            state,
            post(
                "/internal/clarifications",
                r#"{"run_id":"r","question_id":"q","project":"open"}"#,
                None,
            ),
        )
        .await;
        assert!(
            status == StatusCode::NOT_FOUND || status == StatusCode::METHOD_NOT_ALLOWED,
            "the rendezvous answered {status} on the public router"
        );
    }

    #[tokio::test]
    async fn the_project_list_hides_private_projects_and_host_paths() {
        let state = state().await;
        let body = router(state.clone(), None)
            .oneshot(get("/api/projects", None))
            .await
            .expect("the router to answer")
            .into_body();
        let bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .expect("a body");
        let listed = String::from_utf8(bytes.to_vec()).expect("utf-8");
        assert!(listed.contains("open"));
        // Not merely marked private — absent. A greyed-out entry still says it exists.
        assert!(!listed.contains("shut"), "{listed}");
        // And no absolute path, which says more about the machine than a stranger needs.
        assert!(!listed.contains("/tmp"), "{listed}");
    }

    #[tokio::test]
    async fn signing_in_sets_a_session_and_signing_out_clears_it() {
        let state = state().await;
        state
            .auth
            .create_local("kk", "hunter2", "KK", Role::Operator)
            .await
            .expect("a principal");

        let response = router(state.clone(), None)
            .oneshot(post(
                "/api/auth/login",
                r#"{"username":"kk","password":"hunter2"}"#,
                None,
            ))
            .await
            .expect("the router to answer");
        assert_eq!(response.status(), StatusCode::OK);
        let cookie = response
            .headers()
            .get("set-cookie")
            .expect("a session cookie")
            .to_str()
            .expect("ascii")
            .to_string();
        assert!(cookie.contains("HttpOnly"));

        let token = cookie.split(';').next().expect("a name=value pair");
        assert_eq!(
            send(state.clone(), get("/api/projects/shut/runs", Some(token))).await,
            StatusCode::OK,
            "the cookie the server set has to be one it accepts back"
        );

        let out = router(state.clone(), None)
            .oneshot(post("/api/auth/logout", "", Some(token)))
            .await
            .expect("the router to answer");
        assert_eq!(out.status(), StatusCode::NO_CONTENT);
        assert_eq!(
            send(state, get("/api/projects/shut/runs", Some(token))).await,
            StatusCode::NOT_FOUND,
            "a revoked session must stop working immediately"
        );
    }

    #[tokio::test]
    async fn a_wrong_password_is_refused_and_says_nothing_about_the_username() {
        let state = state().await;
        state
            .auth
            .create_local("kk", "hunter2", "KK", Role::Operator)
            .await
            .expect("a principal");
        for body in [
            r#"{"username":"kk","password":"wrong"}"#,
            r#"{"username":"nobody","password":"wrong"}"#,
        ] {
            let response = router(state.clone(), None)
                .oneshot(post("/api/auth/login", body, None))
                .await
                .expect("the router to answer");
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            assert!(response.headers().get("set-cookie").is_none());
        }
    }

    #[tokio::test]
    async fn a_short_run_id_reaches_the_same_run_as_the_long_one() {
        // What the dashboard's URLs carry. Every run-scoped route has to resolve it, and "one
        // handler forgot" is invisible until someone opens a link to that pane.
        let state = state().await;
        let project = state.projects.get("open").expect("the public project");
        project
            .store
            .upsert_run("358e8441-fa9a-4ab4-bbbe-46a826455b20", None, "converged")
            .await
            .expect("a run");

        // Compared against the full id rather than asserted to be 200: whether a route answers
        // depends on what the run contains — `/nodes/scout` is a 404 for a run with no scout
        // checkpoint — and the claim being made is that the short form reaches the *same run*, not
        // that every route has something to say about it.
        for suffix in ["", "/nodes/scout", "/history"] {
            let short = send(
                state.clone(),
                get(&format!("/api/projects/open/runs/358e8441{suffix}"), None),
            )
            .await;
            let full = send(
                state.clone(),
                get(
                    &format!(
                        "/api/projects/open/runs/358e8441-fa9a-4ab4-bbbe-46a826455b20{suffix}"
                    ),
                    None,
                ),
            )
            .await;
            assert_eq!(
                short, full,
                "the short id took a different path at {suffix:?}"
            );
        }

        // And a prefix nothing starts with still reads as a missing run.
        assert_eq!(
            send(state, get("/api/projects/open/runs/deadbeef", None)).await,
            StatusCode::NOT_FOUND
        );
    }

    /// The webhook, through the real router.
    ///
    /// Every one of these is about a request that reaches a public endpoint from the internet, so
    /// what matters is not that the happy path works but that each way of being wrong stops.
    mod github_webhook {
        use super::*;
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        const SECRET: &str = "a webhook secret of some length";

        async fn wired() -> AppState {
            let mut state = state().await;
            state.github = Some(Arc::new(github::GitHubConfig {
                trigger: "ratatoskr".to_string(),
                account: None,
                secret: SECRET.to_string(),
            }));
            state
        }

        fn body(login: &str, id: u64, repo: &str) -> String {
            serde_json::json!({
                "action": "created",
                "issue": { "number": 164 },
                "comment": { "body": "@ratatoskr fix the retry", "user": { "id": id, "login": login } },
                "repository": { "full_name": repo },
            })
            .to_string()
        }

        fn delivery(payload: &str, secret: &str) -> Request<Body> {
            let mut mac =
                Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("any key length");
            mac.update(payload.as_bytes());
            let signature = format!(
                "sha256={}",
                mac.finalize()
                    .into_bytes()
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            );
            Request::builder()
                .method("POST")
                .uri("/api/integrations/github")
                .header("content-type", "application/json")
                .header("x-github-event", "issue_comment")
                .header("x-hub-signature-256", signature)
                .body(Body::from(payload.to_string()))
                .expect("a valid request")
        }

        #[tokio::test]
        async fn an_unsigned_delivery_is_refused() {
            let state = wired().await;
            let payload = body("kk", 1234, "cq27-dev/open");
            assert_eq!(
                send(state, delivery(&payload, "not the secret")).await,
                StatusCode::UNAUTHORIZED
            );
        }

        #[tokio::test]
        async fn a_signed_mention_from_a_stranger_is_ignored_rather_than_refused() {
            // 200 on purpose. GitHub retries anything else, and there is nothing to retry about a
            // comment from someone this instance has never heard of — which is most comments on a
            // public repository.
            let state = wired().await;
            let payload = body("stranger", 9999, "cq27-dev/open");
            let launcher = Arc::clone(&state.projects["open"].launcher);
            assert_eq!(
                send(state, delivery(&payload, SECRET)).await,
                StatusCode::OK
            );
            assert_eq!(launcher.in_flight(), 0, "a stranger started a run");
        }

        #[tokio::test]
        async fn a_linked_viewer_still_cannot_start_a_run() {
            // Being known is not being trusted: the role is checked exactly as it is in the
            // browser, and a viewer with a GitHub identity is still a viewer.
            let state = wired().await;
            let principal = state
                .auth
                .create_local("kk", "hunter2", "KK", Role::Viewer)
                .await
                .expect("a principal");
            state
                .auth
                .attach_identity(&principal.principal_id, github::GITHUB, "1234")
                .await
                .expect("a link");

            let payload = body("kk", 1234, "cq27-dev/open");
            let launcher = Arc::clone(&state.projects["open"].launcher);
            assert_eq!(
                send(state, delivery(&payload, SECRET)).await,
                StatusCode::OK
            );
            // The assertion that matters. 200 is also what a *started* run answers, so the status
            // alone cannot tell a refusal from a success.
            assert_eq!(launcher.in_flight(), 0, "a viewer started a run");
        }

        #[tokio::test]
        async fn a_repository_this_instance_does_not_serve_is_ignored() {
            let state = wired().await;
            let principal = state
                .auth
                .create_local("kk", "hunter2", "KK", Role::Operator)
                .await
                .expect("a principal");
            state
                .auth
                .attach_identity(&principal.principal_id, github::GITHUB, "1234")
                .await
                .expect("a link");

            let payload = body("kk", 1234, "someone-else/private-thing");
            let launcher = Arc::clone(&state.projects["open"].launcher);
            assert_eq!(
                send(state, delivery(&payload, SECRET)).await,
                StatusCode::OK
            );
            assert_eq!(
                launcher.in_flight(),
                0,
                "a mention on another repository reached a project here"
            );
        }

        #[tokio::test]
        async fn the_endpoint_does_not_exist_when_the_integration_is_not_configured() {
            // A 404 rather than a 501: an instance that has not set this up should not advertise
            // that it could.
            let state = state().await;
            let payload = body("kk", 1234, "cq27-dev/open");
            assert_eq!(
                send(state, delivery(&payload, SECRET)).await,
                StatusCode::NOT_FOUND
            );
        }
    }

    #[tokio::test]
    async fn a_stale_cookie_logs_you_out_rather_than_erroring() {
        // A public project has to keep working for someone whose session has lapsed — otherwise a
        // month-old tab shows an error page instead of the run it is pointed at.
        let state = state().await;
        let stale = format!("{}=not-a-real-token", auth::COOKIE_NAME_INSECURE);
        assert_eq!(
            send(state, get("/api/projects/open/runs", Some(&stale))).await,
            StatusCode::OK
        );
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

    // --- pull_request_view -------------------------------------------------
    // The contract leaves `PullRequestView::number` as u64/i64/String; these tests read it as an
    // integer (the `139` literal compiles for u64 or i64), which is the reading that makes the
    // "last URL segment is numeric" requirement checkable. `url` is asserted as a `&str`.

    #[test]
    fn reads_the_pull_request_from_a_publisher_checkpoint() {
        let cps = vec![cp(
            "publisher",
            r#"{"action":"pull_request","pull_request_url":"https://github.com/o/r/pull/139","reasoning":"..."}"#,
        )];
        let pr = pull_request_view(&cps).unwrap();
        assert_eq!(pr.number, 139);
        assert_eq!(pr.url, "https://github.com/o/r/pull/139");
    }

    #[test]
    fn action_both_still_yields_the_pull_request() {
        let cps = vec![cp(
            "publisher",
            r#"{"action":"both","pull_request_url":"https://github.com/o/r/pull/42","comment_url":"https://github.com/o/r/issues/1#issuecomment-2","reasoning":"x"}"#,
        )];
        let pr = pull_request_view(&cps).unwrap();
        assert_eq!(pr.number, 42);
        assert_eq!(pr.url, "https://github.com/o/r/pull/42");
    }

    #[test]
    fn takes_the_pull_request_from_the_latest_publisher_checkpoint() {
        // Latest-wins, like worktree_view: the later publisher checkpoint is authoritative.
        let cps = vec![
            cp(
                "publisher",
                r#"{"action":"pull_request","pull_request_url":"https://github.com/o/r/pull/1"}"#,
            ),
            cp(
                "publisher",
                r#"{"action":"pull_request","pull_request_url":"https://github.com/o/r/pull/2"}"#,
            ),
        ];
        let pr = pull_request_view(&cps).unwrap();
        assert_eq!(pr.number, 2);
        assert_eq!(pr.url, "https://github.com/o/r/pull/2");
    }

    #[test]
    fn a_comment_is_never_presented_as_a_pull_request() {
        let cps = vec![cp(
            "publisher",
            r#"{"action":"comment","comment_url":"https://github.com/o/r/issues/12#issuecomment-999"}"#,
        )];
        assert!(pull_request_view(&cps).is_none());
    }

    #[test]
    fn action_none_with_no_url_is_absent() {
        let cps = vec![cp("publisher", r#"{"action":"none"}"#)];
        assert!(pull_request_view(&cps).is_none());
    }

    #[test]
    fn no_publisher_checkpoint_is_absent() {
        assert!(pull_request_view(&[]).is_none());
        let cps = vec![cp("implementer", r#"{"worktree_path":"/tmp/x"}"#)];
        assert!(pull_request_view(&cps).is_none());
    }

    #[test]
    fn a_malformed_publisher_checkpoint_is_absent_not_a_panic() {
        assert!(pull_request_view(&[cp("publisher", "not json")]).is_none());
    }

    #[test]
    fn a_pull_request_with_a_non_numeric_last_segment_is_absent() {
        let cps = vec![cp(
            "publisher",
            r#"{"action":"pull_request","pull_request_url":"https://github.com/o/r/pull/not-a-number"}"#,
        )];
        assert!(pull_request_view(&cps).is_none());
        let empty_url = vec![cp(
            "publisher",
            r#"{"action":"pull_request","pull_request_url":""}"#,
        )];
        assert!(pull_request_view(&empty_url).is_none());
    }

    #[test]
    fn a_labelled_pull_request_field_yields_only_the_url() {
        // The publisher is told to write the URL alone, and a model writes it. A value that came
        // back labelled still anchors the dashboard at the pull request, not at the label.
        let cps = vec![cp(
            "publisher",
            r#"{"action":"both","pull_request_url":"PR: https://github.com/cq27-dev/ratatoskr/pull/214","comment_url":"https://github.com/cq27-dev/ratatoskr/issues/210#issuecomment-5231512849"}"#,
        )];

        let pr = pull_request_view(&cps).unwrap();
        assert_eq!(pr.number, 214);
        assert_eq!(pr.url, "https://github.com/cq27-dev/ratatoskr/pull/214");
    }

    #[test]
    fn a_non_github_pull_request_url_is_absent() {
        let cps = vec![cp(
            "publisher",
            r#"{"action":"pull_request","pull_request_url":"https://example.com/o/r/pull/214"}"#,
        )];
        assert!(pull_request_view(&cps).is_none());
    }

    #[test]
    fn a_pull_request_missing_the_url_field_is_absent() {
        let cps = vec![cp("publisher", r#"{"action":"pull_request"}"#)];
        assert!(pull_request_view(&cps).is_none());
    }

    #[test]
    fn a_pull_request_view_serializes_number_and_url() {
        // Mirrors `RunDetail.pull_request`: the JSON the API/api.ts consumer sees.
        let cps = vec![cp(
            "publisher",
            r#"{"action":"pull_request","pull_request_url":"https://github.com/o/r/pull/139"}"#,
        )];
        let pr = pull_request_view(&cps).unwrap();
        let json = serde_json::to_value(&pr).unwrap();
        assert_eq!(json["number"], serde_json::json!(139));
        assert_eq!(
            json["url"],
            serde_json::json!("https://github.com/o/r/pull/139")
        );
    }
}
