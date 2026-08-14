//! Scriptable orchestration: run `.ratatoskr/workflow.ts` (issue #18) instead of the hardcoded
//! `run_plan`/`run_full` flow. The script composes host bindings — one per node call site — but
//! every gate stays Rust-enforced: schema validation and checkpointing happen inside each binding,
//! the false-convergence guard lives in `redTeam`, `max_iterations` is capped in `iterate`, the one
//! ceiling re-plan is owned by `replanAtCeiling`, and the terminal status is inferred from
//! checkpoints after the script returns (never trusted from the script).
//!
//! Gates the script cannot weaken: schema validation and checkpointing per binding, the
//! false-convergence guard in `redTeam`, `max_iterations` in `iterate`, the referee check, the
//! acceptance frozen on first use, and the review — a run that called `verify()` and left blocking
//! findings standing is not converged, because the terminal status is read from the checkpoint
//! rather than from what the script returned.
//!
//! `ratatoskr-script` can't call the nodes (it depends only on `ratatoskr-core`), so the bindings
//! live here and plug into `WorkflowRuntime` as `HostFn`s.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ratatoskr_core::{RatatoskrConfig, RunState, RunStatus};
use ratatoskr_exec::{WorktreePath, remove_worktree};
use ratatoskr_graph::NodeError;
use ratatoskr_mcp::{RagRatClient, ServerTools};
use ratatoskr_script::{HostFn, ScriptEngine, WorkflowRuntime};
use ratatoskr_store::Store;
use rmcp::service::ServerSink;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::json;

use crate::{
    AnalystOutput, BookkeeperInput, BookkeeperOutput, ChildTask, ImplementerNode,
    ImplementerOutput, MemoryOutput, PlanError, PlanOutcome, RedTeamNode, RedTeamOutput,
    RunOutcome, ScoutOutput, Stage, bookkeeper, checkpoint, converge,
    publisher::{PublisherInput, PublisherOutput},
    redteam, referee, verifier,
};

/// Backstop on total node-running binding calls per run — a runaway-loop guard, far above any real
/// workflow. `max_iterations` and the false-convergence guard are the precise limits; this only
/// catches a script that ignores them and loops.
const INVOCATION_CEILING: usize = 500;

const STANDARD_WORKFLOW_NAME: &str = "ratatoskr-standard-v1";
pub(crate) const STANDARD_WORKFLOW_V1: &str = include_str!("../workflows/standard-v1.ts");
/// What a workflow imports the standard node definitions from.
pub(crate) const STANDARD_DEFINITIONS_MODULE: &str = "ratatoskr/nodes";
pub(crate) const STANDARD_DEFINITIONS: &str = include_str!("../workflows/nodes.ts");
pub(crate) const STANDARD_WORKFLOW_INCLUDES: &[(&str, &str)] = &[
    ("prompts/analyst.md", include_str!("../prompts/analyst.md")),
    (
        "prompts/bookkeeper.md",
        include_str!("../prompts/bookkeeper.md"),
    ),
    (
        "prompts/characterizer.md",
        include_str!("../prompts/characterizer.md"),
    ),
    ("prompts/context.md", include_str!("../prompts/context.md")),
    (
        "prompts/implementer.md",
        include_str!("../prompts/implementer.md"),
    ),
    (
        "prompts/overseer.md",
        include_str!("../prompts/overseer.md"),
    ),
    (
        "prompts/publisher.md",
        include_str!("../prompts/publisher.md"),
    ),
    (
        "prompts/redteam-author.md",
        include_str!("../prompts/redteam-author.md"),
    ),
    (
        "prompts/redteam-classifier.md",
        include_str!("../prompts/redteam-classifier.md"),
    ),
    ("prompts/scout.md", include_str!("../prompts/scout.md")),
    (
        "prompts/verifier.md",
        include_str!("../prompts/verifier.md"),
    ),
];

/// Everything the bindings need, cloned as an `Arc` into every host closure. Holds the run's shared
/// mutable state (the worktree handle and the invocation/iteration counters) behind atomics/a mutex.
pub struct WorkflowContext {
    config: RatatoskrConfig,
    store: Store,
    engine: Arc<ScriptEngine>,
    run_id: String,
    issue: String,
    /// `None` without rag-rat. The nodes that call it outside the agent — the memory baseline and
    /// the bookkeeper — check before reaching for it.
    sink: Option<ServerSink>,
    /// rag-rat's whole offer, followed by other configured server offers, for every node's pool.
    servers: Vec<ServerTools>,
    /// Configured offers without rag-rat, retained for fresh terminal-stage contexts.
    configured_servers: Vec<ServerTools>,
    repo_path: PathBuf,
    /// Prepared by `redTeam` (or `implement` when no red team runs), then reused by implementation,
    /// iteration, and cleanup. The script never sees a raw path.
    worktree: Mutex<Option<WorktreePath>>,
    /// Red-team authoring must finish before the implementer can edit the same worktree.
    red_team_started: AtomicBool,
    red_team_completed: AtomicBool,
    implement_started: AtomicBool,
    /// Serializes `iterate` calls — two implementers editing one worktree would corrupt it.
    iterate_lock: tokio::sync::Mutex<()>,
    /// The iteration ceiling has one Rust-owned recovery: revise the plan from accumulated review
    /// evidence and make one final attempt. A script can decline it but cannot mint a second one.
    ceiling_replan_started: AtomicBool,
    invocations: AtomicUsize,
    iterations: AtomicU32,
    /// What plugins contributed for this run, prefixed to each node's preamble.
    plugin_context: crate::PluginContext,
    /// Where this run's nodes report what their turns cost. A scripted run records the same
    /// telemetry as a built-in one — the script chooses the order, not what gets measured.
    ledger: Arc<ratatoskr_agent::RunLedger>,
    /// One clarification rendezvous for the whole run, shared by every host that can ask.
    clarifier: Arc<crate::clarify::NodeClarifier>,
    /// The acceptance this run is judged by, resolved once and reused.
    ///
    /// The built-in flow resolves it before the fork and freezes it for the same reason it matters
    /// more here: a script can re-analyse between iterations, and if each binding resolved its own
    /// the plan could move the bar it is judged against mid-run. Whichever binding runs acceptance
    /// first decides it; everything after gets that.
    acceptance: Mutex<Option<Vec<ratatoskr_core::AcceptanceStep>>>,
    /// OCI image identity frozen when this run first executes container-backed work.
    container_image: tokio::sync::OnceCell<Option<String>>,
    /// The one stage registry this run executes: the standard stages with the running workflow's
    /// declarations laid over them.
    ///
    /// Every consumer reads it from here — the JavaScript host table *and* the Rust-owned lifecycle
    /// adapters that run `implementer_attempt`, `redteam_author`, `redteam_classifier` and
    /// `characterizer`. Rebuilding it per path is what let an override validate at startup and then
    /// be ignored by the model turn that actually ran.
    stages: ExecutionStages,
}

/// The one stage registry a run executes, resolved once and shared by everything that has to answer
/// *about* that run — the executor, and the clarifier that speaks for a stage without running it.
///
/// Shared rather than re-derived: a second resolution is a second answer, and the clarifier
/// answering out of the compiled-in table while the executor ran the overlaid registry is exactly
/// how one run came to route the same node two different ways.
pub(crate) type ExecutionStages = Arc<tokio::sync::OnceCell<Arc<Vec<Stage>>>>;

/// This run's registry, falling back to the bundled standard one. See [`WorkflowContext::stages`]
/// for when that fallback is the right answer.
pub(crate) async fn execution_stages(
    stages: &ExecutionStages,
) -> Result<Arc<Vec<Stage>>, PlanError> {
    stages
        .get_or_try_init(|| async { Ok(Arc::new(standard_stages().await?)) })
        .await
        .cloned()
}

pub(crate) struct WorkflowContextParams<'a> {
    pub client: Option<&'a RagRatClient>,
    pub configured: &'a [ServerTools],
    pub config: &'a RatatoskrConfig,
    pub store: &'a Store,
    pub run_id: &'a str,
    pub issue: &'a str,
    pub engine: &'a Arc<ScriptEngine>,
    pub plugin_context: crate::PluginContext,
    pub ledger: Arc<ratatoskr_agent::RunLedger>,
}

impl WorkflowContext {
    pub fn new(
        client: Option<&RagRatClient>,
        config: &RatatoskrConfig,
        store: &Store,
        run_id: &str,
        issue: &str,
        engine: &Arc<ScriptEngine>,
        plugin_context: crate::PluginContext,
    ) -> Result<Arc<Self>, PlanError> {
        Self::new_with_ledger(WorkflowContextParams {
            client,
            configured: &[],
            config,
            store,
            run_id,
            issue,
            engine,
            plugin_context,
            ledger: Arc::new(ratatoskr_agent::RunLedger::default()),
        })
    }

    pub(crate) fn new_with_ledger(
        params: WorkflowContextParams<'_>,
    ) -> Result<Arc<Self>, PlanError> {
        let WorkflowContextParams {
            client,
            configured,
            config,
            store,
            run_id,
            issue,
            engine,
            plugin_context,
            ledger,
        } = params;
        let repo_path = std::env::current_dir()
            .map_err(|e| PlanError::node("workflow", NodeError::Failed(format!("cwd: {e}"))))?;
        // One registry cell, shared with the clarifier: it must answer for the stage this run
        // executes, not for the compiled-in stage of the same name.
        let stages: ExecutionStages = Arc::default();
        let clarifier = crate::clarify::NodeClarifier::new(
            config,
            store,
            engine,
            run_id,
            issue,
            Arc::clone(&stages),
        );
        let configured_servers = configured.to_vec();
        Ok(Arc::new(Self {
            ledger,
            clarifier,
            acceptance: Mutex::new(None),
            container_image: tokio::sync::OnceCell::new(),
            plugin_context,
            stages,
            config: config.clone(),
            store: store.clone(),
            engine: Arc::clone(engine),
            run_id: run_id.to_string(),
            issue: issue.to_string(),
            sink: client.map(|c| c.sink()),
            servers: client
                .map(|c| c.offer())
                .into_iter()
                .chain(configured_servers.iter().cloned())
                .collect(),
            configured_servers,
            repo_path,
            worktree: Mutex::new(None),
            red_team_started: AtomicBool::new(false),
            red_team_completed: AtomicBool::new(false),
            implement_started: AtomicBool::new(false),
            iterate_lock: tokio::sync::Mutex::new(()),
            ceiling_replan_started: AtomicBool::new(false),
            invocations: AtomicUsize::new(0),
            iterations: AtomicU32::new(0),
        }))
    }

    /// The acceptance steps for this run, resolved from `proposed` the first time and frozen.
    fn acceptance(
        &self,
        proposed: &[ratatoskr_core::AcceptanceStep],
    ) -> Vec<ratatoskr_core::AcceptanceStep> {
        let mut slot = self.acceptance.lock().expect("acceptance mutex poisoned");
        slot.get_or_insert_with(|| self.config.sandbox.acceptance(proposed))
            .clone()
    }

    /// Resolve the container tag once, only when the full run actually reaches sandboxed work.
    /// Landlock and MicroVM runs retain their configured image behavior and require no OCI runtime.
    pub(crate) async fn resolved_container_image(&self) -> Result<Option<String>, String> {
        if self.config.sandbox.backend != "container" {
            return Ok(None);
        }
        self.container_image
            .get_or_try_init(|| async {
                let image = ratatoskr_exec::resolve_container_image(&self.config.sandbox.image)
                    .await
                    .map_err(|error| error.to_string())?;
                self.store
                    .record_run_provenance(&self.run_id, None, None, None, None, Some(&image))
                    .await
                    .map_err(|error| error.to_string())?;
                Ok(Some(image))
            })
            .await
            .cloned()
    }

    fn sandbox_config(&self) -> ratatoskr_core::SandboxConfig {
        let mut sandbox = self.config.sandbox.clone();
        if let Some(Some(image)) = self.container_image.get() {
            sandbox.image.clone_from(image);
        }
        sandbox
    }

    /// Count one node-running binding call and refuse past the ceiling.
    fn guard(&self) -> Result<(), String> {
        if self.invocations.fetch_add(1, Ordering::Relaxed) >= INVOCATION_CEILING {
            return Err(format!(
                "workflow exceeded {INVOCATION_CEILING} node invocations — runaway loop?"
            ));
        }
        Ok(())
    }

    pub(crate) fn ledger(&self) -> &Arc<ratatoskr_agent::RunLedger> {
        &self.ledger
    }

    /// This run's stage registry.
    ///
    /// `install_execution_stages` puts the running workflow's overlaid registry here before the
    /// first host call. A context that never runs a workflow — the overseer's selection turn, and
    /// the terminal bookkeeper/publisher adapters, both of which run outside one — falls back to the
    /// bundled standard registry, which is what those turns are defined against. Every stage that
    /// can reach this fallback is refused to workflows at the load-time gate, so the fallback can
    /// never stand in for a declaration someone made and expected to run.
    pub(crate) async fn stages(&self) -> Result<Arc<Vec<Stage>>, PlanError> {
        execution_stages(&self.stages).await
    }
}

/// Resolve the registry this run executes and install it on the context, once.
///
/// The workflow's declarations are laid over the standard stages rather than appended: an override
/// replaces the standard stage where it sat, so the by-id scans (delegation resolution among them)
/// find the override and not the original.
async fn install_execution_stages(
    ctx: &WorkflowContext,
    runtime: &WorkflowRuntime,
) -> Result<Arc<Vec<Stage>>, PlanError> {
    ctx.stages
        .get_or_try_init(|| async { Ok(Arc::new(overlaid_stages(runtime).await?)) })
        .await
        .cloned()
}

async fn overlaid_stages(runtime: &WorkflowRuntime) -> Result<Vec<Stage>, PlanError> {
    let mut stages = standard_stages().await?;
    // The bundled runtime declares the base stages itself; laying them over themselves would be a
    // no-op with a duplicate-work cost. Decided by provenance, not by name: a repository workflow
    // may take any name, including the bundled one, and its declarations still have to be honored.
    if !runtime.is_bundled() {
        crate::stage::overlay(
            &mut stages,
            crate::stage::stages_from_workflow(runtime.meta()),
        );
    }
    Ok(stages)
}

// --- reconstruction helpers -------------------------------------------------

/// The most recent checkpoint for `node`, deserialized. `MissingCheckpoint` is the gate: a script
/// can't claim a terminal run without having actually run the node that writes this checkpoint.
async fn latest_checkpoint<T: DeserializeOwned>(
    store: &Store,
    run_id: &str,
    node: &'static str,
) -> Result<T, PlanError> {
    let checkpoints = store.checkpoints_for_run(run_id).await?;
    let cp = checkpoints
        .iter()
        .rev()
        .find(|c| c.node_name == node)
        .ok_or_else(|| PlanError::MissingCheckpoint(run_id.to_string(), node))?;
    Ok(serde_json::from_str(&cp.output_json)?)
}

async fn count_checkpoints(store: &Store, run_id: &str, node: &str) -> Result<u32, PlanError> {
    let checkpoints = store.checkpoints_for_run(run_id).await?;
    Ok(checkpoints.iter().filter(|c| c.node_name == node).count() as u32)
}

fn same_json<T: serde::Serialize, U: serde::Serialize>(left: &T, right: &U) -> bool {
    match (serde_json::to_value(left), serde_json::to_value(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn previous_verifier_findings(
    checkpoints: &[ratatoskr_store::Checkpoint],
    threshold: verifier::Severity,
) -> Vec<verifier::Finding> {
    checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.node_name == "verifier")
        .filter_map(|checkpoint| {
            serde_json::from_str::<verifier::VerifierOutput>(&checkpoint.output_json).ok()
        })
        // This is the same history the built-in loop carries: a clean or below-threshold review
        // does not produce a correction, so it is not part of a later correction chain.
        .filter(|output| !output.blocking(threshold).is_empty())
        .flat_map(|output| output.findings)
        .collect()
}

enum ScriptedReview {
    NotRun,
    Available(verifier::VerifierOutput),
    Unavailable,
}

fn scripted_review(checkpoints: &[ratatoskr_store::Checkpoint]) -> ScriptedReview {
    let Some(reviewed_at) = checkpoints
        .iter()
        .rposition(|checkpoint| checkpoint.node_name == "verifier")
    else {
        return ScriptedReview::NotRun;
    };
    // An implementer checkpoint written after the review means the tree moved on since it was
    // judged: that review describes a state this run no longer has, so it cannot be the review
    // terminal status rests on. Unreviewed, not reviewed-clean. The bundled flow always verifies
    // after its last iterate, so it never lands here.
    if checkpoints[reviewed_at + 1..]
        .iter()
        .any(|checkpoint| checkpoint.node_name == "implementer")
    {
        return ScriptedReview::NotRun;
    }
    let checkpoint = &checkpoints[reviewed_at];
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&checkpoint.output_json) else {
        return ScriptedReview::Unavailable;
    };
    if value.get("error").is_some() {
        return ScriptedReview::Unavailable;
    }
    match serde_json::from_value(value) {
        Ok(output) => ScriptedReview::Available(output),
        Err(_) => ScriptedReview::Unavailable,
    }
}

fn status_with_review_availability(status: RunStatus, review: &ScriptedReview) -> RunStatus {
    if status == RunStatus::Converged && matches!(review, ScriptedReview::Unavailable) {
        RunStatus::Unreviewed
    } else {
        status
    }
}

/// Terminal status, inferred from the baseline, final implementer output, and the async referee
/// judgement — never trusted from the script. The exemption is applied before the judgement runs,
/// so this pure function only decides whether its violations block the test result.
fn infer_status(
    red_team: &RedTeamOutput,
    implementer: &ImplementerOutput,
    referee: &[referee::Violation],
    review: Option<&verifier::VerifierOutput>,
    threshold: verifier::Severity,
) -> RunStatus {
    if !referee.is_empty() {
        tracing::warn!(violations = ?referee, "run weakened the referee; not converged");
        return RunStatus::MaxIterationsReached;
    }
    // A script chooses when to review; it does not get to ignore what a review said. Calling
    // `verify()`, leaving blocking findings standing and returning is not convergence — and
    // inferring the status from the checkpoint rather than the script is what makes that true.
    if let Some(review) = review {
        let blocking = review.blocking(threshold);
        if !blocking.is_empty() {
            tracing::warn!(
                blocking = blocking.len(),
                "the review left blocking findings; not converged"
            );
            return RunStatus::MaxIterationsReached;
        }
    }
    // The tests written for this change, before it existed. They fail in the baseline by
    // construction, so `is_converged` alone would pass a change that satisfied none of them.
    let authored = red_team
        .authored
        .as_ref()
        .map(|a| a.tests.as_slice())
        .unwrap_or_default();
    let unsatisfied = converge::unsatisfied(authored, &implementer.failing_tests);
    if !unsatisfied.is_empty() {
        tracing::warn!(
            tests = ?unsatisfied,
            "the tests written for this change are still failing; not converged"
        );
        return RunStatus::MaxIterationsReached;
    }
    let post_ran = converge::test_command_ran(
        &implementer.failing_tests,
        implementer.passed_tests,
        implementer.exit_code,
    );
    if post_ran && converge::is_converged(&red_team.failing_tests, &implementer.failing_tests) {
        RunStatus::Converged
    } else {
        RunStatus::MaxIterationsReached
    }
}

/// Say what a `plan` entry left undone, rather than asking whether the run converged.
///
/// A `plan` entry composes freely, but the plan itself is reconstructed in Rust from what the run
/// checkpointed — so `context()` and `analyst()` are a requirement of the entry, not a suggestion.
/// A workflow that composed something else got "not a converged run?" about a command with no
/// converge loop, naming neither itself nor the calls it skipped.
fn plan_entry_omitted(workflow: &str, error: PlanError) -> PlanError {
    match error {
        PlanError::MissingCheckpoint(_, node @ ("scout" | "memory" | "analyst")) => {
            PlanError::Configuration(format!(
                "workflow `{workflow}` returned from its `plan` entry with no `{node}` \
                 checkpoint. A `plan` entry must drive `context()` and `analyst()`: the plan is \
                 reconstructed from the checkpoints those two write, and nothing else composes it."
            ))
        }
        other => other,
    }
}

async fn reconstruct_plan(store: &Store, run_id: &str) -> Result<PlanOutcome, PlanError> {
    // A run may have gathered context either way: one `context` checkpoint, or the separate
    // `scout` and `memory` ones a script composing the older bindings still writes. Preferring the
    // merged one and falling back keeps both replayable.
    let merged: Option<crate::ContextOutput> =
        latest_checkpoint(store, run_id, "context").await.ok();
    let (scout, memory, brief, constraints) = match merged {
        Some(c) => (c.scout, c.memory, c.brief, c.constraints),
        None => (
            latest_checkpoint::<ScoutOutput>(store, run_id, "scout").await?,
            latest_checkpoint::<MemoryOutput>(store, run_id, "memory").await?,
            String::new(),
            Vec::new(),
        ),
    };
    let analyst: AnalystOutput = latest_checkpoint(store, run_id, "analyst").await?;

    let mut state = RunState::new(run_id, None);
    state.scout_report = Some(serde_json::to_value(&scout)?);
    state.memories = memory
        .memories
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()?;
    state.analysis = Some(serde_json::to_value(&analyst)?);
    state.status = RunStatus::Planned;
    Ok(PlanOutcome {
        state,
        scout,
        memory,
        analyst,
        brief,
        constraints,
    })
}

// --- host bindings ----------------------------------------------------------
//
// Each returns a `HostFn`: `Arc<dyn Fn(String) -> Future<Output = Result<String, String>>>`. The
// argument and return are JSON (the seam's contract); errors become thrown JS `Error`s.

/// JSON shape shared by `RedTeamOutput` and `ImplementerOutput` for the pure converge helpers.
#[derive(Deserialize)]
struct RunShape {
    #[serde(default)]
    failing_tests: Vec<String>,
    #[serde(default)]
    passed_tests: usize,
    #[serde(default)]
    exit_code: i32,
}

#[derive(Deserialize)]
struct BaselinePost {
    baseline: RunShape,
    post: RunShape,
}

fn binding<F, Fut>(ctx: Arc<WorkflowContext>, f: F) -> HostFn
where
    F: Fn(Arc<WorkflowContext>, String) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<String, String>> + Send + 'static,
{
    let f = Arc::new(f);
    Arc::new(move |arg| {
        let ctx = Arc::clone(&ctx);
        let f = Arc::clone(&f);
        Box::pin(async move { f(ctx, arg).await })
            as std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>>
    })
}

/// `acceptance` is passed in rather than read here because the baseline and the post-change run
/// must execute the same steps — a red team that resolved its own would drift from the implementer
/// the moment a plan proposed anything.
async fn build_red_team(
    ctx: &Arc<WorkflowContext>,
    acceptance: Vec<ratatoskr_core::AcceptanceStep>,
) -> Result<RedTeamNode, PlanError> {
    let short: String = ctx.run_id.chars().take(8).collect();
    let stages = ctx.stages().await?;
    // Enablement only, and each half on its own stage. Each drives its turn through the stage
    // executor, which resolves route, tools, ceiling and prompt from the run's registry — the
    // classifier from `redteam_classifier`, the author from `redteam_author`, whose own `write`
    // ceiling is what keeps the classifier's read ceiling from disarming it. So the gate has to
    // ask about the same stage the turn will run: a single answer under the shared `redteam`
    // governance name decides for whichever stage it reached first, and is wrong for the other.
    let enabled =
        |stage_id| crate::red_team_half_enabled(&ctx.engine, &ctx.config, &stages, stage_id);
    let classifier = enabled("redteam_classifier").then(|| redteam::RedTeamClassifier {
        declared_context: Arc::clone(ctx),
    });
    let author = enabled("redteam_author").then(|| redteam::TestAuthor {
        declared_context: Arc::clone(ctx),
    });
    Ok(RedTeamNode {
        author,
        acceptance,
        characterizer: crate::build_characterizer(
            &ctx.engine,
            &ctx.config,
            &stages,
            Some(Arc::clone(ctx)),
        )?,
        repo_path: ctx.repo_path.clone(),
        worktree_root: ctx.config.worktree.root.clone(),
        baseline_branch: format!("ratatoskr/{short}-baseline"),
        sandbox: ctx.sandbox_config(),
        name: format!("ratatoskr-redteam-{short}"),
        classifier,
    })
}

/// Checkpoint a scripted node's output, claiming whatever its turn cost from the run's ledger.
///
/// A binding's `arg` is already the node's serialized input — the script hands it across the seam as
/// JSON — so recording it costs nothing beyond passing it along.
async fn note<T: serde::Serialize>(
    ctx: &WorkflowContext,
    node: &str,
    out: &T,
    input: Option<String>,
) -> Result<(), String> {
    // Implementer attempts are the only repeated model checkpoints whose ordinal is durable
    // friction evidence. Derive it from the persisted sequence rather than trusting the script's
    // loop variable, which has no authority over bookkeeping semantics.
    let iteration = if node == "implementer" {
        Some(
            count_checkpoints(&ctx.store, &ctx.run_id, node)
                .await
                .map_err(|error| error.to_string())?
                + 1,
        )
    } else {
        None
    };
    crate::record(crate::Record {
        store: &ctx.store,
        run_id: &ctx.run_id,
        node,
        output: out,
        input,
        iteration,
        ledger: Some(&ctx.ledger),
    })
    .await
    .map_err(|e| e.to_string())
}

/// Say so when a run ends still holding model turns nobody claimed.
///
/// A turn is claimed by the checkpoint it belongs to. Whatever is left is cost the run paid that no
/// row accounts for — a node whose model turn ran under one name and was checkpointed under
/// another, or a turn whose checkpoint never happened. Nothing else would say: a dropped number
/// reads exactly like a node that never called a model (#262).
/// Stage identities whose model turn no checkpoint claims, with the reason each is allowed to.
///
/// A turn is claimed by the checkpoint written under the same name in the same claim scope
/// (`crate::record` -> `RunLedger::take`), and `execute_after_guard` writes one when the invocation
/// checkpoints OR the stage belongs to another node. A stage that is folded into someone else's
/// record as evidence AND is a node of its own therefore has nothing to claim its turn, and its
/// cost lands in the bin — invisible, because a dropped number reads exactly like a node that never
/// called a model.
///
/// Written out and bolted both ways by `nothing_records_under_a_name_nobody_claims`: a stage that
/// acquires the property without being listed fails, and a listed name that no longer has it fails.
/// The second direction is the one that matters when a fix lands — the list shrinks and the case
/// says so.
///
/// The other shape this admits is a delegation target: the executor invokes one at the evidence
/// disposition, so a target that is its own node is unclaimed for the same reason (#283). The
/// bundled registry declares no delegation, so nothing is listed for it here; the rule below covers
/// one the moment it appears.
pub(crate) const UNCLAIMED_BY_DESIGN: &[(&str, &str)] = &[(
    "characterizer",
    "folded into another stage's record as evidence and declares no node, because which node ran \
     it depends on the invocation (#244)",
)];

fn warn_about_unclaimed_turns(ctx: &WorkflowContext) {
    // The by-design residents are filtered out rather than reported, so what is left is always
    // something to act on. A warning an operator learns to expect is a warning nobody reads, and
    // `nothing_records_under_a_name_nobody_claims` is what keeps the filter from hiding a real one.
    let unclaimed: Vec<String> = ctx
        .ledger
        .unclaimed()
        .into_iter()
        .filter(|name| {
            !UNCLAIMED_BY_DESIGN
                .iter()
                .any(|(known, _)| known == &name.as_str())
        })
        .collect();
    if unclaimed.is_empty() {
        return;
    }
    tracing::warn!(
        nodes = %unclaimed.join(", "),
        "these model turns cost the run and reached no checkpoint, so nothing reports what they \
         spent; a stage whose output is folded into another stage's record as evidence, while \
         being a node of its own, is the usual cause — see UNCLAIMED_BY_DESIGN for the ones that \
         are meant to be here"
    );
}

async fn red_team_host(ctx: Arc<WorkflowContext>, _arg: String) -> Result<String, String> {
    ctx.guard()?;
    if ctx.red_team_started.swap(true, Ordering::SeqCst) {
        return Err("redTeam() called more than once in a workflow".to_string());
    }
    if ctx.implement_started.load(Ordering::SeqCst) {
        return Err(
            "redTeam() must run and finish before implement() starts, so test authoring cannot race implementation"
                .to_string(),
        );
    }
    let analyst = latest_checkpoint::<AnalystOutput>(&ctx.store, &ctx.run_id, "analyst")
        .await
        .map_err(|error| error.to_string())?;
    ctx.resolved_container_image().await?;
    let acceptance = ctx.acceptance(&analyst.acceptance);
    let implementer = build_implementer(&ctx, analyst.clone())
        .await
        .map_err(|e| e.to_string())?;
    let worktree = implementer.prepare().await.map_err(|e| e.to_string())?;
    *ctx.worktree.lock().unwrap() = Some(worktree.clone());
    let node = build_red_team(&ctx, acceptance)
        .await
        .map_err(|e| e.to_string())?;
    let out = node
        .run_and_author(worktree.as_path(), &ctx.issue, &analyst.interface)
        .await
        .map_err(|e| e.to_string())?;
    // Checkpoint before the guard so a failed baseline stays inspectable.
    note(&ctx, crate::policy::REDTEAM_NODE, &out, None).await?;
    // The false-convergence guard is enforced here — the script cannot skip it.
    if !converge::test_command_ran(&out.failing_tests, out.passed_tests, out.exit_code) {
        return Err(format!(
            "the baseline acceptance run produced no checks (exit {}); check the analyst's acceptance, [sandbox] test_command and the sandbox backend",
            out.exit_code
        ));
    }
    ctx.red_team_completed.store(true, Ordering::SeqCst);
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

async fn build_implementer(
    ctx: &Arc<WorkflowContext>,
    analyst: AnalystOutput,
) -> Result<ImplementerNode, PlanError> {
    let stages = ctx.stages().await?;
    Ok(ImplementerNode {
        clarifier: Some(ctx.clarifier.as_dyn()),
        acceptance: ctx.acceptance(&analyst.acceptance),
        characterizer: crate::build_characterizer(
            &ctx.engine,
            &ctx.config,
            &stages,
            Some(Arc::clone(ctx)),
        )
        .ok()
        .flatten(),
        repo_path: ctx.repo_path.clone(),
        worktree_root: ctx.config.worktree.root.clone(),
        sandbox: ctx.sandbox_config(),
        run_id: ctx.run_id.clone(),
        issue: ctx.issue.clone(),
        analyst,
        declared_context: Arc::clone(ctx),
    })
}

#[derive(Deserialize)]
struct ImplementArg {
    analyst: AnalystOutput,
}

async fn implement_host(ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    ctx.guard()?;
    // Atomically refuse a second `implement` (would leak the first worktree + branch).
    if ctx.implement_started.swap(true, Ordering::SeqCst) {
        return Err("implement() called more than once in a workflow".to_string());
    }
    if ctx.red_team_started.load(Ordering::SeqCst) && !ctx.red_team_completed.load(Ordering::SeqCst)
    {
        return Err(
            "implement() cannot start until the awaited redTeam() call has finished".to_string(),
        );
    }
    let input: ImplementArg =
        serde_json::from_str(&arg).map_err(|e| format!("implement arg: {e}"))?;
    ctx.resolved_container_image().await?;
    let node = build_implementer(&ctx, input.analyst)
        .await
        .map_err(|e| e.to_string())?;
    let prepared = { ctx.worktree.lock().unwrap().clone() };
    let worktree = match prepared {
        Some(worktree) => worktree,
        None => {
            let worktree = node.prepare().await.map_err(|e| e.to_string())?;
            *ctx.worktree.lock().unwrap() = Some(worktree.clone());
            worktree
        }
    };
    let out = match node.work(&worktree).await {
        Ok(out) => out,
        Err(error) => {
            node.discard(&worktree).await;
            ctx.worktree.lock().unwrap().take();
            return Err(error.to_string());
        }
    };
    note(&ctx, crate::policy::IMPLEMENTER_NODE, &out, Some(arg)).await?;
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

async fn referee_judgement(
    ctx: &Arc<WorkflowContext>,
    worktree: &WorktreePath,
    analyst: &AnalystOutput,
    implementer: &ImplementerOutput,
) -> Vec<referee::Violation> {
    let stages = match ctx.stages().await {
        Ok(stages) => stages,
        Err(error) => {
            tracing::warn!("the referee could not resolve this run's stages: {error}");
            return Vec::new();
        }
    };
    let violations = match referee::judge(referee::Judgement {
        engine: &ctx.engine,
        config: &ctx.config,
        stages: &stages,
        ledger: &ctx.ledger,
        issue: &ctx.issue,
        requirements: &analyst.requirements,
        implementer,
        worktree,
    })
    .await
    {
        Ok(Some(violations)) => violations,
        Ok(None) => return Vec::new(),
        Err(error) => {
            tracing::warn!(
                "the referee could not judge this change; trusting test results: {error}"
            );
            if let Err(record_error) = note(
                ctx,
                "referee",
                &serde_json::json!({ "error": error.to_string() }),
                None,
            )
            .await
            {
                tracing::warn!("failed to record referee failure: {record_error}");
            }
            return Vec::new();
        }
    };
    if let Err(error) = note(
        ctx,
        "referee",
        &referee::RefereeOutput {
            violations: violations.clone(),
        },
        None,
    )
    .await
    {
        tracing::warn!("failed to record referee judgement: {error}");
    }
    violations
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct IterateArg {
    /// The result returned by `verify()`, when review rather than tests caused this correction.
    /// It is only a correlation token: the host recomputes it from the checkpoint before use.
    #[serde(default)]
    review: Option<VerifyResult>,
}

fn review_correction(
    checkpoints: &[ratatoskr_store::Checkpoint],
    supplied: &VerifyResult,
    threshold: verifier::Severity,
) -> Result<String, String> {
    let implementer_position = checkpoints
        .iter()
        .rposition(|checkpoint| checkpoint.node_name == "implementer")
        .ok_or_else(|| "iterate() called before implement()".to_string())?;
    let (review_position, checkpoint) = checkpoints
        .iter()
        .enumerate()
        .skip(implementer_position + 1)
        .rev()
        .find(|(_, checkpoint)| checkpoint.node_name == "verifier")
        .ok_or_else(|| {
            "iterate() received a review that was not checkpointed after the current implementation"
                .to_string()
        })?;
    let output_value: serde_json::Value = serde_json::from_str(&checkpoint.output_json)
        .map_err(|_| "iterate() cannot correct an unavailable verifier result".to_string())?;
    if output_value.get("error").is_some() {
        return Err("iterate() cannot correct an unavailable verifier result".to_string());
    }
    let output: verifier::VerifierOutput = serde_json::from_value(output_value)
        .map_err(|_| "iterate() cannot correct an unavailable verifier result".to_string())?;
    let expected = verification_result(output.clone(), threshold);
    if !same_json(supplied, &expected) {
        return Err(
            "iterate() review does not match the latest Rust-validated verifier checkpoint"
                .to_string(),
        );
    }

    let blocking = output.blocking(threshold);
    if blocking.is_empty() {
        return Err("iterate() review has no blocking findings to correct".to_string());
    }
    let plan_faults: Vec<verifier::Finding> = blocking
        .iter()
        .filter(|finding| finding.kind == verifier::FindingKind::Plan)
        .map(|finding| (*finding).clone())
        .collect();
    if plan_faults.is_empty() {
        return Ok(verifier::correction(&blocking));
    }

    // A workflow drives the declared analyst stage explicitly, but a plan finding is usable only
    // when that turn happened after this review and was actually a revision of the reviewed plan
    // for the blocking plan findings. This keeps routing compositional without trusting the script.
    let revision_checkpoint = checkpoints
        .iter()
        .skip(review_position + 1)
        .rev()
        .find(|checkpoint| checkpoint.node_name == "analyst")
        .ok_or_else(|| {
            "iterate() must run analyst() after a blocking plan finding before implementation"
                .to_string()
        })?;
    let revision_input: crate::analyst::AnalystInput = revision_checkpoint
        .input_json
        .as_deref()
        .ok_or_else(|| "the analyst revision checkpoint has no input".to_string())
        .and_then(|input| serde_json::from_str(input).map_err(|error| error.to_string()))?;
    let verifier_input: verifier::VerifierInput = checkpoint
        .input_json
        .as_deref()
        .ok_or_else(|| "the verifier checkpoint has no input".to_string())
        .and_then(|input| serde_json::from_str(input).map_err(|error| error.to_string()))?;
    if revision_input
        .previous
        .as_deref()
        .is_none_or(|previous| !same_json(previous, &verifier_input.analyst))
        || !same_json(&revision_input.findings, &plan_faults)
    {
        return Err(
            "the analyst checkpoint after review is not a revision of the reviewed plan and findings"
                .to_string(),
        );
    }
    let revised: AnalystOutput = serde_json::from_str(&revision_checkpoint.output_json)
        .map_err(|error| error.to_string())?;
    Ok(crate::replan(&revised, &blocking))
}

async fn iterate_host(ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    ctx.guard()?;
    let input: IterateArg =
        serde_json::from_str(&arg).map_err(|error| format!("iterate arg: {error}"))?;
    // One iterate at a time — reject overlapping calls (e.g. `Promise.all([iterate(), iterate()])`)
    // that would drive two implementers against the same worktree. Held for the whole call.
    let _iterate = ctx
        .iterate_lock
        .try_lock()
        .map_err(|_| "iterate() is already in progress".to_string())?;
    // Backstop mirroring today's loop: only `max_iterations - 1` iterate calls are legitimate.
    let n = ctx.iterations.fetch_add(1, Ordering::SeqCst) + 1;
    if n >= ctx.config.implementer.max_iterations {
        return Err(format!(
            "iterate() exceeded max_iterations ({})",
            ctx.config.implementer.max_iterations
        ));
    }
    let worktree = ctx
        .worktree
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "iterate() called before implement()".to_string())?;

    // Rebuild the diagnostic Rust-side (identical wording to the hardcoded converge loop) from the
    // baseline and the latest implementer output — the script doesn't get to author it.
    let red_team: RedTeamOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "redteam")
        .await
        .map_err(|e| e.to_string())?;
    let prev: ImplementerOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "implementer")
        .await
        .map_err(|e| e.to_string())?;
    let post_ran =
        converge::test_command_ran(&prev.failing_tests, prev.passed_tests, prev.exit_code);
    let analyst: AnalystOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "analyst")
        .await
        .map_err(|e| e.to_string())?;
    let referee = referee_judgement(&ctx, &worktree, &analyst, &prev).await;
    // Referee first, same as the built-in loop: a moved referee makes the test sets meaningless,
    // so reverting it is what this iteration has to be told to do.
    let authored = red_team
        .authored
        .as_ref()
        .map(|authored| authored.tests.as_slice())
        .unwrap_or_default();
    let unsatisfied = converge::unsatisfied(authored, &prev.failing_tests);
    let diagnostic = if !referee.is_empty() {
        referee::correction(&referee)
    } else if !post_ran {
        format!(
            "The test command did not run to completion (exit {}) — your change likely does not \
             compile. Fix it so the tests run and pass.",
            prev.exit_code
        )
    } else if !unsatisfied.is_empty() {
        format!(
            "These tests were written for this change, from the interface, before any code existed \
             to satisfy them — making them pass is what the change is for, and they are still \
             failing: {}. They are not yours to edit; implement what they describe. If one of \
             them is wrong about the contract rather than about your code, say so in your summary \
             and implement the rest.",
            unsatisfied.join(", ")
        )
    } else if !converge::is_converged(&red_team.failing_tests, &prev.failing_tests) {
        let new_failures =
            converge::newly_introduced_failures(&red_team.failing_tests, &prev.failing_tests);
        format!(
            "Your change introduced NEW failing tests not present in the baseline: {}. Fix them \
             without breaking other tests.",
            new_failures.join(", ")
        )
    } else if let Some(review) = input.review.as_ref() {
        let checkpoints = ctx
            .store
            .checkpoints_for_run(&ctx.run_id)
            .await
            .map_err(|error| error.to_string())?;
        review_correction(
            &checkpoints,
            review,
            crate::parse_threshold(&ctx.config.implementer.verify_threshold),
        )?
    } else {
        return Err(
            "iterate() has no referee, test, or checkpointed review correction to apply"
                .to_string(),
        );
    };

    ctx.resolved_container_image().await?;
    let node = build_implementer(&ctx, analyst)
        .await
        .map_err(|e| e.to_string())?;
    let out = node
        .iterate(&worktree, &diagnostic)
        .await
        .map_err(|e| e.to_string())?;
    // The diagnostic, not the binding's argument: the script does not author it, so it is the one
    // thing that explains what this iteration was actually asked to fix.
    note(
        &ctx,
        crate::policy::IMPLEMENTER_NODE,
        &out,
        Some(diagnostic),
    )
    .await?;
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

// Pure converge helpers — no work, no checkpoint, not counted toward the invocation ceiling.
async fn is_converged_host(_ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    let bp: BaselinePost =
        serde_json::from_str(&arg).map_err(|e| format!("isConverged arg: {e}"))?;
    let v = converge::is_converged(&bp.baseline.failing_tests, &bp.post.failing_tests);
    serde_json::to_string(&v).map_err(|e| e.to_string())
}

async fn test_command_ran_host(_ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    let s: RunShape = serde_json::from_str(&arg).map_err(|e| format!("testCommandRan arg: {e}"))?;
    let v = converge::test_command_ran(&s.failing_tests, s.passed_tests, s.exit_code);
    serde_json::to_string(&v).map_err(|e| e.to_string())
}

/// What `verify()` hands the script.
///
/// `blocking` is the part that matters and the part the script does not get to compute: Rust reads
/// `[implementer] verify_threshold` and decides what clears it. A workflow chooses *whether* to
/// review and what to do about findings; it cannot decide that a P1 is not a P1.
#[derive(serde::Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct VerifyResult {
    /// False when no `[models.verifier]` route is configured — the run is unreviewed and the
    /// script should be able to tell that apart from a clean review.
    configured: bool,
    /// True when the verifier could not be reached. Not a finding: it says nothing about the
    /// change, and treating it as a clean review would claim a review nobody performed.
    unavailable: bool,
    /// Everything found, including what fell below the threshold — recorded, not blocking.
    findings: Vec<verifier::Finding>,
    /// What blocks, worst first.
    blocking: Vec<verifier::Finding>,
    /// Whether any blocking finding faults the PLAN rather than the code. The script's cue to
    /// re-analyse before re-driving the implementer, instead of sending it back at a requirement
    /// already shown to be wrong.
    needs_replan: bool,
}

fn verification_result(
    out: verifier::VerifierOutput,
    threshold: verifier::Severity,
) -> VerifyResult {
    let blocking: Vec<verifier::Finding> = out.blocking(threshold).into_iter().cloned().collect();
    let needs_replan = blocking
        .iter()
        .any(|finding| finding.kind == verifier::FindingKind::Plan);
    VerifyResult {
        configured: true,
        unavailable: false,
        findings: out.findings,
        blocking,
        needs_replan,
    }
}

/// `verify({ analyst })` — read the worktree's diff against the plan.
///
/// Mirrors the built-in flow's second gate. The script decides when to call it; every judgement
/// inside stays here.
async fn verify_host(
    ctx: Arc<WorkflowContext>,
    executor: Arc<StageExecutor>,
    arg: String,
) -> Result<String, String> {
    #[derive(Deserialize)]
    struct Arg {
        analyst: AnalystOutput,
    }
    ctx.guard()?;
    let input: Arg = serde_json::from_str(&arg).map_err(|e| format!("verify arg: {e}"))?;
    // Review excludes editing. A verifier that runs beside `iterate()` reads the worktree while an
    // implementer is writing it, and its checkpoint is what terminal status rests on — so the
    // review would be of a half-written tree that never existed. Held for the whole call, and
    // taken before any early return so the exclusion does not depend on the verifier's config.
    let _iterate = ctx
        .iterate_lock
        .try_lock()
        .map_err(|_| "verify() cannot overlap iterate()".to_string())?;

    let none = |configured, unavailable| {
        serde_json::to_string(&VerifyResult {
            configured,
            unavailable,
            findings: Vec::new(),
            blocking: Vec::new(),
            needs_replan: false,
        })
        .map_err(|e| e.to_string())
    };
    if !crate::verifier_enabled(&ctx.engine, &ctx.config, executor.stages.as_slice()) {
        return none(false, false);
    }
    let worktree = ctx
        .worktree
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "verify() called before implement()".to_string())?;
    let implementer: ImplementerOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "implementer")
        .await
        .map_err(|e| e.to_string())?;

    // The patch, not the `--stat` the implementer records: a summary cannot show a weakened
    // assertion, which is one of the things this gate exists to catch.
    let diff = ratatoskr_exec::diff_text(&worktree)
        .await
        .unwrap_or_default();
    let checkpoints = ctx
        .store
        .checkpoints_for_run(&ctx.run_id)
        .await
        .map_err(|error| error.to_string())?;
    let threshold = crate::parse_threshold(&ctx.config.implementer.verify_threshold);
    let checkpointed_analyst: AnalystOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "analyst")
        .await
        .map_err(|error| error.to_string())?;
    if !same_json(&input.analyst, &checkpointed_analyst) {
        return Err("verify() analyst does not match the latest analyst checkpoint".to_string());
    }
    let verifier_input = verifier::VerifierInput {
        previous_findings: previous_verifier_findings(&checkpoints, threshold),
        issue: ctx.issue.clone(),
        analyst: checkpointed_analyst,
        diff,
        touched_files: implementer.touched_files.clone(),
    };
    let input_json = serde_json::to_string(&verifier_input).map_err(|e| e.to_string())?;
    let raw = if verifier_input.diff.trim().is_empty() {
        let out = verifier::VerifierOutput {
            findings: Vec::new(),
            assessment: "there was no diff to review".to_string(),
        };
        note(&ctx, "verifier", &out, Some(input_json)).await?;
        serde_json::to_string(&out).map_err(|e| e.to_string())?
    } else {
        let stage = executor
            .stages
            .iter()
            .find(|stage| stage.id == "verifier")
            .cloned()
            .ok_or_else(|| "standard verifier stage is not registered".to_string())?;
        match execute_standard_stage(
            &executor,
            stage,
            input_json.clone(),
            StandardStageInvocation {
                resource_root: Some(worktree.0.clone()),
                // The review reads the change in place; it never gets to be the last writer in the
                // tree it judges, whatever a workflow declares for the `verifier` stage.
                capability_ceiling: ratatoskr_core::Capability::Read,
                rag_rat_worktree: Some(worktree.0.clone()),
                shell: None,
                publish: None,
                clarifier: None,
                invocation_guidance: None,
                output: StageOutput::Checkpoint,
                after_guard: true,
            },
        )
        .await
        {
            Ok(raw) => raw,
            Err(error) => {
                // A verifier that cannot run must not fail a change that was made and passed. Recorded
                // and reported as unavailable, exactly as the built-in flow treats it.
                tracing::warn!("the verifier could not review this change: {error}");
                note(
                    &ctx,
                    "verifier",
                    &serde_json::json!({ "error": error }),
                    Some(input_json),
                )
                .await?;
                return none(true, true);
            }
        }
    };
    let out: verifier::VerifierOutput = serde_json::from_str(&raw).map_err(|e| e.to_string())?;

    serde_json::to_string(&verification_result(out, threshold)).map_err(|e| e.to_string())
}

#[derive(serde::Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CeilingReplanResult {
    analyst: AnalystOutput,
    implementation: ImplementerOutput,
}

trait CeilingRecovery: Sync {
    async fn revise(
        &self,
        _ctx: &Arc<WorkflowContext>,
        executor: &Arc<StageExecutor>,
        input: &crate::analyst::AnalystInput,
    ) -> Result<Option<AnalystOutput>, String>;

    async fn iterate(
        &self,
        ctx: &Arc<WorkflowContext>,
        worktree: &WorktreePath,
        revised: &AnalystOutput,
        diagnostic: &str,
    ) -> Result<ImplementerOutput, String>;
}

struct LiveCeilingRecovery;

impl CeilingRecovery for LiveCeilingRecovery {
    async fn revise(
        &self,
        _ctx: &Arc<WorkflowContext>,
        executor: &Arc<StageExecutor>,
        input: &crate::analyst::AnalystInput,
    ) -> Result<Option<AnalystOutput>, String> {
        let input_json = serde_json::to_string(input).map_err(|error| error.to_string())?;
        let stage = executor
            .stages
            .iter()
            .find(|stage| stage.id == "analyst")
            .cloned()
            .ok_or_else(|| "standard analyst stage is not registered".to_string())?;
        let raw = match execute_standard_stage(
            executor,
            stage,
            input_json,
            StandardStageInvocation {
                resource_root: None,
                capability_ceiling: ratatoskr_core::Capability::Read,
                rag_rat_worktree: None,
                shell: None,
                publish: None,
                clarifier: None,
                invocation_guidance: None,
                output: StageOutput::Checkpoint,
                after_guard: true,
            },
        )
        .await
        {
            Ok(raw) => raw,
            Err(error) => {
                tracing::warn!("the analyst could not re-plan at the ceiling: {error}");
                return Ok(None);
            }
        };
        serde_json::from_str(&raw)
            .map(Some)
            .map_err(|error| error.to_string())
    }

    async fn iterate(
        &self,
        ctx: &Arc<WorkflowContext>,
        worktree: &WorktreePath,
        revised: &AnalystOutput,
        diagnostic: &str,
    ) -> Result<ImplementerOutput, String> {
        ctx.resolved_container_image().await?;
        build_implementer(ctx, revised.clone())
            .await
            .map_err(|error| error.to_string())?
            .iterate(worktree, diagnostic)
            .await
            .map_err(|error| error.to_string())
    }
}

/// Spend the one recovery the built-in convergence loop historically allowed after the ordinary
/// attempt budget was exhausted.
///
/// The workflow supplies no evidence or plan. Rust reconstructs both from checkpoints, proves the
/// current result still needs correction, and owns the analyst + implementer calls as one bounded
/// operation. Returning `null` means the ceiling is final. This is deliberately not split into a
/// script-visible "authorize" token and a later iteration: such a token would let a workflow replay
/// or reorder the extra attempt.
async fn replan_at_ceiling_host(
    ctx: Arc<WorkflowContext>,
    executor: Arc<StageExecutor>,
    _arg: String,
) -> Result<String, String> {
    replan_at_ceiling_with(ctx, executor, &LiveCeilingRecovery).await
}

async fn replan_at_ceiling_with<R: CeilingRecovery>(
    ctx: Arc<WorkflowContext>,
    executor: Arc<StageExecutor>,
    recovery: &R,
) -> Result<String, String> {
    ctx.guard()?;
    let _iterate = ctx
        .iterate_lock
        .try_lock()
        .map_err(|_| "replanAtCeiling() cannot overlap iterate()".to_string())?;
    if ctx.ceiling_replan_started.load(Ordering::SeqCst) {
        return Ok("null".to_string());
    }

    let checkpoints = ctx
        .store
        .checkpoints_for_run(&ctx.run_id)
        .await
        .map_err(|error| error.to_string())?;
    let attempts = checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.node_name == "implementer")
        .count() as u32;
    if attempts < ctx.config.implementer.max_iterations {
        return Ok("null".to_string());
    }
    let already_replanned = checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.node_name == "analyst")
        .filter_map(|checkpoint| checkpoint.input_json.as_deref())
        .filter_map(|input| serde_json::from_str::<crate::analyst::AnalystInput>(input).ok())
        .any(|input| input.previous.is_some());
    if already_replanned {
        return Ok("null".to_string());
    }

    let threshold = crate::parse_threshold(&ctx.config.implementer.verify_threshold);
    let findings = previous_verifier_findings(&checkpoints, threshold);
    if findings.is_empty() {
        return Ok("null".to_string());
    }

    let red_team: RedTeamOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "redteam")
        .await
        .map_err(|error| error.to_string())?;
    let implementation: ImplementerOutput =
        latest_checkpoint(&ctx.store, &ctx.run_id, "implementer")
            .await
            .map_err(|error| error.to_string())?;
    let current_plan: AnalystOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "analyst")
        .await
        .map_err(|error| error.to_string())?;
    let worktree = ctx
        .worktree
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "replanAtCeiling() called before implement()".to_string())?;

    // A stale finding from an earlier iteration is not authority to extend a now-clean run. The
    // current attempt must still need a correction under the same Rust-owned referee/test/review
    // gates as the convergence loop.
    let referee = referee_judgement(&ctx, &worktree, &current_plan, &implementation).await;
    let authored = red_team
        .authored
        .as_ref()
        .map(|authored| authored.tests.as_slice())
        .unwrap_or_default();
    let tests_clean = converge::test_command_ran(
        &implementation.failing_tests,
        implementation.passed_tests,
        implementation.exit_code,
    ) && converge::unsatisfied(authored, &implementation.failing_tests)
        .is_empty()
        && converge::is_converged(&red_team.failing_tests, &implementation.failing_tests);
    let current_review_blocks = checkpoints
        .iter()
        .rposition(|checkpoint| checkpoint.node_name == "implementer")
        .and_then(|position| {
            checkpoints
                .iter()
                .skip(position + 1)
                .rev()
                .find(|checkpoint| checkpoint.node_name == "verifier")
        })
        .and_then(|checkpoint| {
            serde_json::from_str::<verifier::VerifierOutput>(&checkpoint.output_json).ok()
        })
        .is_some_and(|output| !output.blocking(threshold).is_empty());
    if referee.is_empty() && tests_clean && !current_review_blocks {
        return Ok("null".to_string());
    }

    // Consume the recovery before either model turn. A failed best-effort analyst re-plan stops at
    // the wall exactly as a run without this recovery does; it never earns a retry of the extra
    // budget.
    if ctx.ceiling_replan_started.swap(true, Ordering::SeqCst) {
        return Ok("null".to_string());
    }
    tracing::warn!(
        attempts,
        findings = findings.len(),
        "the iteration budget is spent; asking the analyst to look at the plan before one final attempt"
    );

    let gathered: crate::ContextOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "context")
        .await
        .map_err(|error| error.to_string())?;
    let revision = crate::analyst::AnalystInput {
        issue: ctx.issue.clone(),
        scout: gathered.scout,
        memory: gathered.memory,
        brief: gathered.brief,
        constraints: gathered.constraints,
        previous: Some(Box::new(current_plan)),
        findings,
    };
    let Some(revised) = recovery.revise(&ctx, &executor, &revision).await? else {
        return Ok("null".to_string());
    };
    let borrowed = revision.findings.iter().collect::<Vec<_>>();
    let diagnostic = crate::replan(&revised, &borrowed);
    ctx.iterations.fetch_add(1, Ordering::SeqCst);
    let implementation = recovery
        .iterate(&ctx, &worktree, &revised, &diagnostic)
        .await?;
    note(
        &ctx,
        "implementer",
        &implementation,
        Some(serde_json::to_string(&revised).map_err(|error| error.to_string())?),
    )
    .await?;
    serde_json::to_string(&CeilingReplanResult {
        analyst: revised,
        implementation,
    })
    .map_err(|error| error.to_string())
}

/// `context(issue)` — the merged gather step: distilled findings plus the memories unmodified.
///
/// The one operation that guarantees the ranked memory search happened, so it is what both bundled
/// entries call before anything reasons about the issue.
async fn context_host(
    ctx: Arc<WorkflowContext>,
    executor: Arc<StageExecutor>,
    arg: String,
) -> Result<String, String> {
    ctx.guard()?;
    let issue: String = serde_json::from_str(&arg).map_err(|e| format!("context arg: {e}"))?;
    let memory = match &ctx.sink {
        Some(sink) => crate::memory::search(sink, &issue, "")
            .await
            .map_err(|error| error.to_string())?,
        None => crate::MemoryOutput::default(),
    };
    let input = crate::context::distillation_input(&issue, memory.clone(), ctx.sink.is_some());
    let input_json = serde_json::to_string(&input).map_err(|error| error.to_string())?;
    let stage = executor
        .stages
        .iter()
        .find(|stage| stage.id == "context_distillation")
        .cloned()
        .ok_or_else(|| "standard stage `context_distillation` is not registered".to_string())?;
    let raw = execute_standard_stage(
        &executor,
        stage,
        input_json,
        StandardStageInvocation {
            resource_root: None,
            capability_ceiling: ratatoskr_core::Capability::Read,
            rag_rat_worktree: None,
            shell: None,
            publish: None,
            clarifier: None,
            invocation_guidance: None,
            output: StageOutput::Evidence,
            after_guard: true,
        },
    )
    .await?;
    let distilled: crate::context::Distillation =
        serde_json::from_str(&raw).map_err(|error| error.to_string())?;
    let out = crate::context::attach_evidence(distilled, memory);
    note(&ctx, crate::policy::CONTEXT_NODE, &out, Some(arg)).await?;
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

fn validate_declared_output(stage: &Stage, output: &serde_json::Value) -> Result<(), String> {
    let Some(schema) = stage.output_schema.as_ref() else {
        if stage.output_contract.is_empty() {
            return Ok(());
        }
        return Err(format!(
            "stage `{}` declares output contract `{}` without outputSchema",
            stage.id, stage.output_contract
        ));
    };
    ratatoskr_graph::validate_value(output, schema).map_err(|e| {
        format!(
            "stage `{}` returned invalid `{}` output: {e}",
            stage.id, stage.output_contract
        )
    })?;
    // The declared schema is the stage's own; for the identifiers Rust reads back as a concrete
    // type, passing it is not the same as being readable. Checked here rather than at load because
    // this is the one gate every stage's output passes through on its way to a checkpoint — and
    // failing here names the stage and the contract, which the serde error at the eventual
    // `latest_checkpoint` no longer can.
    crate::policy::check_typed_output(&stage.id, output)
}

fn normalize_declared_output(stage: &Stage, output: &mut serde_json::Value) -> Result<(), String> {
    if stage.array_normalization.is_empty() {
        return Ok(());
    }
    let object = output
        .as_object_mut()
        .ok_or_else(|| format!("stage `{}` normalization requires object output", stage.id))?;
    for normalization in &stage.array_normalization {
        if !object.contains_key(&normalization.field) && normalization.default_empty {
            object.insert(
                normalization.field.clone(),
                serde_json::Value::Array(Vec::new()),
            );
        }
        let Some(value) = object.get_mut(&normalization.field) else {
            continue;
        };
        let array = value.as_array_mut().ok_or_else(|| {
            format!(
                "stage `{}` normalization field `{}` is not an array",
                stage.id, normalization.field
            )
        })?;
        if normalization.retain_when_any_non_blank.is_empty() {
            continue;
        }
        array.retain(|item| {
            item.as_object().is_some_and(|item| {
                normalization.retain_when_any_non_blank.iter().any(|field| {
                    item.get(field)
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|value| !value.trim().is_empty())
                })
            })
        });
    }
    Ok(())
}

/// Build a declared stage's cached guidance in the order that governs its runtime input.
///
/// Runtime data deliberately stays out of this preamble: [`declared_stage_question`] puts it in
/// the user message after platform, agent, stage, repository, plugin, and skill guidance.
fn declared_stage_preamble(
    stage: &Stage,
    governance_id: &str,
    profile: &crate::AgentProfile,
    system_prompt: Option<&str>,
    repository_guidance: &str,
    plugin_context: Option<&str>,
    skills: &[ratatoskr_plugin::Skill],
) -> String {
    let stage_guidance = match system_prompt {
        Some(instructions) => [
            "Return JSON matching the declared output contract.",
            profile.base_prompt.as_str(),
            instructions,
            if stage.append_repository_guidance {
                repository_guidance
            } else {
                ""
            },
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n"),
        None => stage.prompt(
            "Return JSON matching the declared output contract.",
            profile,
            repository_guidance,
        ),
    };
    let base = [stage_guidance.as_str(), plugin_context.unwrap_or_default()]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    crate::effective_preamble(governance_id, &base, None, None, skills)
}

fn declared_stage_question(stage: &Stage, runtime_question: &str) -> String {
    format!(
        "Input contract: {}\nOutput contract: {}\n\n{runtime_question}",
        stage.input_contract, stage.output_contract
    )
}

pub(crate) trait StageTurn: Send + Sync {
    fn run<'a>(
        &'a self,
        run: ratatoskr_agent::NodeRun<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<String, ratatoskr_agent::AgentError>> + Send + 'a>>;
}

struct LiveStageTurn;

impl StageTurn for LiveStageTurn {
    fn run<'a>(
        &'a self,
        run: ratatoskr_agent::NodeRun<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<String, ratatoskr_agent::AgentError>> + Send + 'a>>
    {
        Box::pin(ratatoskr_agent::run_structured(run))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StageOutput {
    Checkpoint,
    Evidence,
}

struct StageInvocation {
    stage: Stage,
    input_json: String,
    rendered_question: Option<String>,
    resource_root: Option<PathBuf>,
    /// What Rust grants THIS invocation, independent of where its file tools resolve.
    ///
    /// `resource_root` answers "where", never "may mutate": `verify_host` hands the review turn the
    /// implementer's worktree so it can read the change, and a workflow is free to override the
    /// verifier's declared capabilities. The offer takes the lower of this and the stage's own
    /// ceiling, so an override can only ever narrow what the caller granted.
    capability_ceiling: ratatoskr_core::Capability,
    rag_rat_worktree: Option<PathBuf>,
    shell: Option<ratatoskr_agent::shell::ShellAccess>,
    publish: Option<StandardStagePublishResources>,
    clarifier: Option<Arc<dyn ratatoskr_agent::Clarifier>>,
    invocation_guidance: Option<String>,
    output: StageOutput,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RenderedStageQuestion {
    input: serde_json::Value,
    question: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RenderedStageEnvelope {
    #[serde(rename = "__ratatoskrRenderedQuestion")]
    rendered: RenderedStageQuestion,
}

fn stage_invocation(stage: Stage, host_input_json: String) -> Result<StageInvocation, String> {
    let Some(_) = stage.question_renderer.as_ref() else {
        return Ok(StageInvocation {
            stage,
            input_json: host_input_json,
            rendered_question: None,
            resource_root: None,
            capability_ceiling: ratatoskr_core::Capability::Read,
            rag_rat_worktree: None,
            shell: None,
            publish: None,
            clarifier: None,
            invocation_guidance: None,
            output: StageOutput::Checkpoint,
        });
    };
    let envelope: RenderedStageEnvelope = serde_json::from_str(&host_input_json)
        .map_err(|error| format!("stage `{}` rendered-question envelope: {error}", stage.id))?;
    Ok(StageInvocation {
        stage,
        input_json: serde_json::to_string(&envelope.rendered.input)
            .map_err(|error| error.to_string())?,
        rendered_question: Some(envelope.rendered.question),
        resource_root: None,
        capability_ceiling: ratatoskr_core::Capability::Read,
        rag_rat_worktree: None,
        shell: None,
        publish: None,
        clarifier: None,
        invocation_guidance: None,
        output: StageOutput::Checkpoint,
    })
}

/// Generic execution boundary for every declarative stage.
///
/// It owns model/profile resolution, authority narrowing, delegation, schema validation,
/// normalization, telemetry and checkpointing. Workflow operation adapters stay outside it so a
/// stage never acquires worktree, iteration or terminal-status powers by sharing an identifier.
struct StageExecutor {
    ctx: Arc<WorkflowContext>,
    stages: Arc<Vec<Stage>>,
    turn: Arc<dyn StageTurn>,
}

impl StageExecutor {
    fn new(
        ctx: Arc<WorkflowContext>,
        stages: Arc<Vec<Stage>>,
        turn: Arc<dyn StageTurn>,
    ) -> Arc<Self> {
        Arc::new(Self { ctx, stages, turn })
    }

    fn host(self: &Arc<Self>, stage: Stage) -> HostFn {
        let executor = Arc::clone(self);
        Arc::new(move |input_json| {
            let executor = Arc::clone(&executor);
            let stage = stage.clone();
            Box::pin(async move {
                let invocation = stage_invocation(stage, input_json)?;
                executor.execute(invocation).await
            })
        })
    }

    async fn execute(self: &Arc<Self>, invocation: StageInvocation) -> Result<String, String> {
        self.ctx.guard()?;
        self.execute_after_guard(invocation).await
    }

    async fn execute_after_guard(
        self: &Arc<Self>,
        invocation: StageInvocation,
    ) -> Result<String, String> {
        let StageInvocation {
            stage,
            input_json,
            rendered_question,
            resource_root,
            capability_ceiling,
            rag_rat_worktree,
            shell,
            publish,
            clarifier,
            invocation_guidance,
            output: disposition,
        } = invocation;
        let input: serde_json::Value =
            serde_json::from_str(&input_json).map_err(|e| format!("{} arg: {e}", stage.id))?;
        let governance_id = stage.governance_id().to_string();
        let plugins = self.ctx.plugin_context.for_node(&governance_id);
        let default_tools = stage.tools.iter().map(String::as_str).collect::<Vec<_>>();
        let mut offered = self
            .ctx
            .plugin_context
            .pool_for(&governance_id, &self.ctx.servers);
        // A file-mutation tool is offered only against the root Rust supplied for this invocation,
        // never on the strength of the declaration alone. A declared stage host owns no worktree
        // lifecycle, and its file root otherwise falls back to the process's working directory —
        // the operator's checkout — so a stage that declared `Write` would be editing that.
        //
        // The root says *where* the file tools resolve and nothing more: the caller's ceiling says
        // whether this invocation may mutate there. Reading the two out of one field is what let a
        // `capabilities: ["write"]` override of the read-only `verifier` hold Edit/Write inside the
        // implementer's worktree — as the last writer after every gate had already passed.
        let granted = ratatoskr_core::Capability::ceiling(&stage.capabilities)
            .map(|declared| declared.min(capability_ceiling));
        if resource_root.is_some()
            && stage.tools.iter().any(|tool| {
                tool == ratatoskr_agent::files::WRITE || tool == ratatoskr_agent::files::EDIT
            })
            && granted.is_some_and(|granted| granted.permits(ratatoskr_core::Capability::Write))
        {
            offered.add_local_tools(ratatoskr_agent::files::edit_declarations());
        }
        // The grant is what puts an implementation behind `Bash`, exactly as `gh` needs its
        // publish grant. Offered without one it is a tool whose every call is refused, and the
        // model spends turns discovering that.
        if shell.is_some()
            && stage
                .tools
                .iter()
                .any(|tool| tool == ratatoskr_agent::shell::BASH)
        {
            offered.add_local(ratatoskr_agent::shell::declaration());
        }
        // Same rule again: without the clarifier grant, `ask` reaches a stub that errors on every
        // call. JS-host invocations pass none, so a repository stage that declares `ask` would hold
        // a tool it can only discover is broken.
        if clarifier.is_some()
            && stage
                .tools
                .iter()
                .any(|tool| tool == ratatoskr_agent::ASK_TOOL_NAME)
        {
            offered.add_local(crate::clarify::ask_tool());
        }
        if publish.is_some()
            && stage
                .tools
                .iter()
                .any(|tool| tool == ratatoskr_agent::publish::GH)
        {
            offered.add_local(ratatoskr_agent::publish::declaration());
        }
        if publish
            .as_ref()
            .and_then(|publish| publish.push.as_ref())
            .is_some()
            && stage
                .tools
                .iter()
                .any(|tool| tool == ratatoskr_agent::publish::PUSH)
        {
            offered.add_local(ratatoskr_agent::publish::push_declaration());
        }
        let (mut cfg, profile) = crate::plugins::declared_stage_agent_config(
            &self.ctx.engine,
            &self.ctx.config,
            offered,
            &stage,
            &default_tools,
            &plugins,
            capability_ceiling,
        )
        .map_err(|e| e.to_string())?;
        cfg.route.session = stage.session_scope(cfg.route.session);

        // Delegation folds the child's evidence into the parent's runtime input on the way to the
        // parent's checkpoint, so only a checkpointed invocation can honour it. `characterizer`,
        // `redteam_classifier` and `context_distillation` are each *both* a global a workflow may
        // call directly and a stage a Rust adapter invokes as evidence — one stage id, two
        // dispositions — so the load-time refusal in `validate` cannot speak for this one. Refuse
        // here instead of running the turn with the declaration quietly omitted.
        if let Some(delegation) = stage
            .delegation
            .as_ref()
            .filter(|_| disposition == StageOutput::Evidence)
        {
            return Err(format!(
                "stage `{}` delegates to `{}`, but this invocation folds its output into another \
                 stage's record instead of checkpointing it, and cannot honour the delegation; \
                 drop the delegation or call `{}` directly from the workflow",
                stage.id, delegation.target, stage.id
            ));
        }
        // A child is evidence within its parent's call, never a second checkpointed graph stage.
        let runtime_input = if let Some(delegation) = stage.delegation.as_ref() {
            let target = self
                .stages
                .iter()
                .find(|candidate| candidate.id == delegation.target)
                .ok_or_else(|| {
                    format!(
                        "stage `{}` delegates to missing `{}`",
                        stage.id, delegation.target
                    )
                })?
                .clone();
            let target_profile = crate::agent_profiles(&self.ctx.config)
                .into_iter()
                .find(|candidate| candidate.id == target.agent)
                .ok_or_else(|| format!("stage `{}` has no agent `{}`", target.id, target.agent))?;
            let task = ChildTask::spawn(&stage, &profile, &target, &target_profile, input.clone())
                .map_err(|e| e.to_string())?;
            let child = Box::pin(self.execute(StageInvocation {
                stage: target,
                input_json: serde_json::to_string(&task.input).map_err(|e| e.to_string())?,
                rendered_question: None,
                resource_root: resource_root.clone(),
                capability_ceiling,
                rag_rat_worktree: rag_rat_worktree.clone(),
                shell: None,
                publish: None,
                clarifier: None,
                invocation_guidance: None,
                output: StageOutput::Evidence,
            }))
            .await?;
            let child_output: serde_json::Value =
                serde_json::from_str(&child).map_err(|e| e.to_string())?;
            let evidence: serde_json::Value =
                task.evidence(child_output).map_err(|e| e.to_string())?;
            json!({ "input": input, "child_evidence": evidence })
        } else {
            input
        };
        let runtime_input_json =
            serde_json::to_string(&runtime_input).map_err(|e| e.to_string())?;
        let repository_guidance = crate::repo_conventions(&self.ctx.repo_path).unwrap_or_default();
        let mut preamble = declared_stage_preamble(
            &stage,
            &governance_id,
            &profile,
            cfg.system_prompt.as_deref(),
            &repository_guidance,
            plugins.context.as_deref(),
            &plugins.skills,
        );
        if let Some(guidance) = invocation_guidance.as_deref() {
            preamble.push_str("\n\n");
            preamble.push_str(guidance);
        }
        if stage.question_renderer.is_some() && rendered_question.is_none() {
            return Err(format!(
                "stage `{}` has renderQuestion but was invoked without its workflow host",
                stage.id
            ));
        }
        let question = declared_stage_question(
            &stage,
            rendered_question.as_deref().unwrap_or(&runtime_input_json),
        );
        let output_schema = stage
            .output_schema
            .clone()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| format!("stage `{}` has invalid outputSchema: {e}", stage.id))?;
        // Stable within this run and stage, unique across runs. Endpoint reuse needs the uniqueness;
        // local compaction uses the same key so both continuation modes agree on stage identity.
        //
        // Per STAGE, not per governance identity. Two stages that share a route are still two
        // pieces of work with different capabilities, and a shared key would continue one of them
        // into the other's session — handing a read-only half the write-capable half's context the
        // moment a route stops being `fresh`.
        let conversation = format!("{}-{}", self.ctx.run_id, stage.id);
        let raw = self
            .turn
            .run(ratatoskr_agent::NodeRun {
                // The stage, so the span, the `node_start` event, the ledger claim and a
                // clarification's `from` all name the work that actually ran. `governance_id`
                // stays what it is documented to be — the route, the ruleset, the plugins and the
                // skills, all resolved above under it.
                node: &stage.id,
                // The operator, though, acts on the box the graph draws. A stage that is its own
                // node is that box; one that belongs to another answers at the box's address.
                controlled_as: Some(stage.node_id()),
                route: &cfg.route,
                preamble: &preamble,
                question: &question,
                tools: cfg.tools,
                output_schema: output_schema
                    .unwrap_or_else(|| schemars::schema_for!(serde_json::Value)),
                policy: cfg.policy,
                max_turns: cfg.max_turns,
                clarifier,
                observer: plugins.observer.clone(),
                skills: crate::skills::loaded(&plugins.skills, &governance_id),
                files: resource_root.or(cfg.files),
                rag_rat_worktree,
                shell,
                push: publish.and_then(|publish| publish.push),
                conversation: Some(&conversation),
                ledger: Some(Arc::clone(&self.ctx.ledger)),
                produces: Some(&stage.output_contract),
            })
            .await
            .map_err(|e| format!("stage `{}` agent failed: {e}", stage.id))?;
        let mut output: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("stage `{}` returned invalid JSON: {e}", stage.id))?;
        validate_declared_output(&stage, &output)?;
        normalize_declared_output(&stage, &mut output)?;
        // Evidence is folded into the enclosing host's aggregate record rather than checkpointed —
        // but the model turn behind it is this stage's, and its cost has to land somewhere. A stage
        // that belongs to a node writes its own row inside that node's box: the aggregate becomes
        // the parent, and each child row describes exactly one turn. That is the only honest place
        // for `model` and `tools`, because the halves of a box run on different profiles with
        // different tool sets, and a row naming both is true of the box and useless about either.
        //
        // Without it the turn would be recorded under the stage and claimed under the box, which is
        // exactly the cost-in-the-bin that `RunLedger::unclaimed` exists to catch.
        if disposition == StageOutput::Checkpoint || !stage.is_own_node() {
            note(&self.ctx, &stage.id, &output, Some(input_json)).await?;
        }
        serde_json::to_string(&output).map_err(|e| e.to_string())
    }
}

/// A Rust-owned operation a workflow calls as a bare host.
///
/// These are not stages and are not migrating away: each owns run state or a gate a declared stage
/// cannot hold — the worktree lifecycle, the iteration ceiling, the acceptance baseline, the
/// deterministic memory search behind `context`. The bundled `plan` and `full` entries call every
/// one of them, and so may a repository workflow.
#[derive(Clone, Copy)]
pub(crate) enum OperationHost {
    Context,
    RedTeam,
    Implement,
    Iterate,
    ReplanAtCeiling,
    Verify,
    IsConverged,
    TestCommandRan,
}

impl OperationHost {
    fn host(self, ctx: &Arc<WorkflowContext>, executor: &Arc<StageExecutor>) -> HostFn {
        match self {
            Self::Context => {
                let ctx = Arc::clone(ctx);
                let executor = Arc::clone(executor);
                Arc::new(move |arg| {
                    let ctx = Arc::clone(&ctx);
                    let executor = Arc::clone(&executor);
                    Box::pin(async move { context_host(ctx, executor, arg).await })
                })
            }
            Self::RedTeam => binding(Arc::clone(ctx), red_team_host),
            Self::Implement => binding(Arc::clone(ctx), implement_host),
            Self::Iterate => binding(Arc::clone(ctx), iterate_host),
            Self::ReplanAtCeiling => {
                let ctx = Arc::clone(ctx);
                let executor = Arc::clone(executor);
                Arc::new(move |arg| {
                    let ctx = Arc::clone(&ctx);
                    let executor = Arc::clone(&executor);
                    Box::pin(async move { replan_at_ceiling_host(ctx, executor, arg).await })
                })
            }
            Self::Verify => {
                let ctx = Arc::clone(ctx);
                let executor = Arc::clone(executor);
                Arc::new(move |arg| {
                    let ctx = Arc::clone(&ctx);
                    let executor = Arc::clone(&executor);
                    Box::pin(async move { verify_host(ctx, executor, arg).await })
                })
            }
            Self::IsConverged => binding(Arc::clone(ctx), is_converged_host),
            Self::TestCommandRan => binding(Arc::clone(ctx), test_command_ran_host),
        }
    }
}

/// The one authority on which names are operation hosts. Both the host table below and the
/// startup check that reserves these names against stage declarations read this list, so a host
/// added here cannot be silently shadowed by a declared stage of the same name.
pub(crate) const OPERATION_HOSTS: &[(&str, OperationHost)] = &[
    ("context", OperationHost::Context),
    ("redTeam", OperationHost::RedTeam),
    ("implement", OperationHost::Implement),
    ("iterate", OperationHost::Iterate),
    ("replanAtCeiling", OperationHost::ReplanAtCeiling),
    ("verify", OperationHost::Verify),
    ("isConverged", OperationHost::IsConverged),
    ("testCommandRan", OperationHost::TestCommandRan),
];

fn build_operation_hosts(
    ctx: &Arc<WorkflowContext>,
    executor: &Arc<StageExecutor>,
) -> HashMap<String, HostFn> {
    OPERATION_HOSTS
        .iter()
        .map(|(name, operation)| ((*name).to_string(), operation.host(ctx, executor)))
        .collect()
}

pub(crate) use crate::policy::SELECTION_STAGE_ID;

/// The stages a workflow can call, as globals under their own ids.
///
/// [`crate::policy::is_js_host`] decides, so the filter is the classification rather than a list
/// beside it. Filtered here, at the one place where a stage becomes a repository-JS global, rather
/// than out of the registry — the Rust adapters still have to resolve these stages, and a filter
/// applied before an overlay is a filter an override walks past.
fn build_declared_stage_hosts(executor: &Arc<StageExecutor>) -> HashMap<String, HostFn> {
    executor
        .stages
        .iter()
        .filter(|stage| crate::policy::is_js_host(&stage.id))
        .map(|stage| (stage.id.clone(), executor.host(stage.clone())))
        .collect()
}

fn build_hosts_with_turn(
    ctx: &Arc<WorkflowContext>,
    stages: &[Stage],
    turn: Arc<dyn StageTurn>,
) -> Result<HashMap<String, HostFn>, PlanError> {
    let stages = Arc::new(stages.to_vec());
    let executor = StageExecutor::new(Arc::clone(ctx), Arc::clone(&stages), turn);
    let declared = build_declared_stage_hosts(&executor);
    let mut hosts = build_operation_hosts(ctx, &executor);
    if let Some(stage) = stages.iter().find(|stage| hosts.contains_key(&stage.id)) {
        return Err(PlanError::Configuration(format!(
            "stage `{}` conflicts with a workflow operation host",
            stage.id
        )));
    }
    hosts.extend(declared);
    // One host call is one claim scope. This is the boundary a workflow can cross twice at once —
    // `Promise.all([probe(a), probe(b)])` runs one stage's host twice, under one name — so it is
    // where the identity a checkpoint claims against has to be minted. Everything an invocation
    // runs inside itself is inside its scope: the halves a composite host folds into one record,
    // and a nested stage whose own checkpoint claims under its own name.
    Ok(hosts
        .into_iter()
        .map(|(name, host)| (name, claiming(host)))
        .collect())
}

/// Wrap a host so its call is one claim scope.
fn claiming(host: HostFn) -> HostFn {
    Arc::new(move |arg| Box::pin(ratatoskr_agent::claim_scope(host(arg))))
}

fn build_hosts(
    ctx: &Arc<WorkflowContext>,
    stages: &[Stage],
) -> Result<HashMap<String, HostFn>, PlanError> {
    build_hosts_with_turn(ctx, stages, Arc::new(LiveStageTurn))
}

fn stage_question_renderers(stages: &[Stage]) -> HashMap<String, String> {
    stages
        .iter()
        .filter_map(|stage| {
            stage
                .question_renderer
                .as_ref()
                .map(|source| (stage.id.clone(), source.clone()))
        })
        .collect()
}

pub(crate) async fn standard_stages() -> Result<Vec<Stage>, PlanError> {
    let runtime = standard_runtime().await?;
    let stages = crate::stage::stages_from_workflow(runtime.meta());
    crate::validate::validate_declared_contracts(&stages)?;
    Ok(stages)
}

pub(crate) async fn standard_runtime() -> Result<WorkflowRuntime, PlanError> {
    let definitions = standard_definitions()?;
    WorkflowRuntime::bundled_with_includes(
        STANDARD_WORKFLOW_NAME,
        STANDARD_WORKFLOW_V1,
        STANDARD_WORKFLOW_INCLUDES,
        &[(STANDARD_DEFINITIONS_MODULE, &definitions)],
    )
    .await
    .map_err(|error| PlanError::node("workflow", NodeError::Failed(error.to_string())))
}

/// The standard node definitions as importable JavaScript.
///
/// Transpiled through the include-resolving entry rather than plain type stripping: the definitions
/// carry `LOAD("prompts/..")` calls, which are compile-time inclusions with no runtime equivalent.
/// Every workflow gets the same map — a repository's own workflow imports these exactly as the
/// bundled one does.
pub(crate) fn standard_definitions() -> Result<String, PlanError> {
    ratatoskr_script::transpile_with_includes(
        STANDARD_DEFINITIONS_MODULE,
        STANDARD_DEFINITIONS,
        STANDARD_WORKFLOW_INCLUDES,
        // The definitions module is the leaf of the import graph: it imports nothing.
        &[],
    )
    .map_err(|error| PlanError::node("workflow", NodeError::Failed(error.to_string())))
}

/// Evaluate one bundled standard stage outside a repository workflow script.
///
/// Selection runs before a workflow exists, so the overseer cannot be invoked through a script
/// host. It still uses the same executor boundary; the caller retains the semantic gate and owns
/// checkpointing only after that gate accepts the stage output.
pub(crate) async fn evaluate_standard_stage(
    ctx: Arc<WorkflowContext>,
    stage_id: &str,
    input_json: String,
) -> Result<String, String> {
    evaluate_standard_stage_with_turn(ctx, stage_id, input_json, Arc::new(LiveStageTurn)).await
}

/// Rust-owned resources granted to one bundled evidence turn.
///
/// A stage may use these resources but cannot create, replace, retain, or clean them up. That
/// keeps worktree and sandbox lifecycle in the operation adapter while the model call remains a
/// generic declared-stage execution.
#[derive(Clone)]
pub(crate) struct StandardStageResources {
    pub resource_root: PathBuf,
    /// What this invocation may do in `resource_root`. See [`StageInvocation::capability_ceiling`].
    pub capability_ceiling: ratatoskr_core::Capability,
    /// A linked worktree that rag-rat queries must see as an overlay over the base index.
    ///
    /// This is deliberately separate from the file-tool root: a Rust host selects the worktree
    /// and the agent binding keeps the absolute path out of model-visible tool arguments.
    pub rag_rat_worktree: Option<PathBuf>,
    pub shell: Option<ratatoskr_agent::shell::ShellAccess>,
    pub publish: Option<StandardStagePublishResources>,
    pub clarifier: Option<Arc<dyn ratatoskr_agent::Clarifier>>,
    pub guidance: Option<String>,
}

/// External publish authority granted by Rust to one terminal stage invocation.
///
/// Merely declaring `gh` or `git_push` never installs their host implementations. This grant is
/// what keeps a repository workflow from publishing before Rust has accepted a terminal outcome.
#[derive(Clone)]
pub(crate) struct StandardStagePublishResources {
    pub push: Option<ratatoskr_agent::publish::PushAccess>,
}

#[derive(Clone)]
struct StandardStageInvocation {
    resource_root: Option<PathBuf>,
    /// See [`StageInvocation::capability_ceiling`]: the mutation grant is per invocation, not a
    /// second meaning read out of `resource_root`.
    capability_ceiling: ratatoskr_core::Capability,
    rag_rat_worktree: Option<PathBuf>,
    shell: Option<ratatoskr_agent::shell::ShellAccess>,
    publish: Option<StandardStagePublishResources>,
    clarifier: Option<Arc<dyn ratatoskr_agent::Clarifier>>,
    invocation_guidance: Option<String>,
    output: StageOutput,
    after_guard: bool,
}

async fn execute_standard_stage(
    executor: &Arc<StageExecutor>,
    stage: Stage,
    input_json: String,
    settings: StandardStageInvocation,
) -> Result<String, String> {
    let stage_id = stage.id.clone();
    let host_executor = Arc::clone(executor);
    let host_stage = stage.clone();
    let host_settings = settings.clone();
    let host: HostFn = Arc::new(move |rendered_input| {
        let executor = Arc::clone(&host_executor);
        let stage = host_stage.clone();
        let settings = host_settings.clone();
        Box::pin(async move {
            let mut invocation = stage_invocation(stage, rendered_input)?;
            invocation.resource_root = settings.resource_root;
            invocation.capability_ceiling = settings.capability_ceiling;
            invocation.rag_rat_worktree = settings.rag_rat_worktree;
            invocation.shell = settings.shell;
            invocation.publish = settings.publish;
            invocation.clarifier = settings.clarifier;
            invocation.invocation_guidance = settings.invocation_guidance;
            invocation.output = settings.output;
            if settings.after_guard {
                executor.execute_after_guard(invocation).await
            } else {
                executor.execute(invocation).await
            }
        })
    });
    let input: serde_json::Value =
        serde_json::from_str(&input_json).map_err(|error| format!("{stage_id} arg: {error}"))?;
    let runtime = standard_runtime()
        .await
        .map_err(|error| error.to_string())?;
    runtime
        .run_with_question_renderers(
            "standardStageTurn",
            json!({ "stage": stage_id, "input": input }).to_string(),
            HashMap::from([(stage.id, host)]),
            stage_question_renderers(executor.stages.as_slice()),
        )
        .await
        .map_err(|error| error.to_string())
}

pub(crate) async fn evaluate_standard_stage_with_resources(
    ctx: Arc<WorkflowContext>,
    stage_id: &str,
    input_json: String,
    resources: StandardStageResources,
) -> Result<String, String> {
    evaluate_standard_stage_with_resources_and_turn(
        ctx,
        stage_id,
        input_json,
        resources,
        Arc::new(LiveStageTurn),
    )
    .await
}

pub(crate) async fn evaluate_standard_stage_with_turn(
    ctx: Arc<WorkflowContext>,
    stage_id: &str,
    input_json: String,
    turn: Arc<dyn StageTurn>,
) -> Result<String, String> {
    evaluate_standard_stage_with_turn_and_resources(ctx, stage_id, input_json, None, turn).await
}

pub(crate) async fn evaluate_standard_stage_with_resources_and_turn(
    ctx: Arc<WorkflowContext>,
    stage_id: &str,
    input_json: String,
    resources: StandardStageResources,
    turn: Arc<dyn StageTurn>,
) -> Result<String, String> {
    evaluate_standard_stage_with_turn_and_resources(
        ctx,
        stage_id,
        input_json,
        Some(resources),
        turn,
    )
    .await
}

async fn evaluate_standard_stage_with_turn_and_resources(
    ctx: Arc<WorkflowContext>,
    stage_id: &str,
    input_json: String,
    resources: Option<StandardStageResources>,
    turn: Arc<dyn StageTurn>,
) -> Result<String, String> {
    let (
        resource_root,
        capability_ceiling,
        rag_rat_worktree,
        shell,
        publish,
        clarifier,
        invocation_guidance,
    ) = match resources {
        Some(resources) => (
            Some(resources.resource_root),
            resources.capability_ceiling,
            resources.rag_rat_worktree,
            resources.shell,
            resources.publish,
            resources.clarifier,
            resources.guidance,
        ),
        None => (
            None,
            ratatoskr_core::Capability::Read,
            None,
            None,
            None,
            None,
            None,
        ),
    };
    // The run's registry, not a fresh standard one: a workflow that overrides `implementer_attempt`
    // (or any other adapter-invoked stage) must have that override be what this turn runs.
    let stages = ctx.stages().await.map_err(|error| error.to_string())?;
    let stage = stages
        .iter()
        .find(|stage| stage.id == stage_id)
        .cloned()
        .ok_or_else(|| format!("standard stage `{stage_id}` is not registered"))?;
    let executor = StageExecutor::new(ctx, stages, turn);
    execute_standard_stage(
        &executor,
        stage,
        input_json,
        StandardStageInvocation {
            resource_root,
            capability_ceiling,
            rag_rat_worktree,
            shell,
            publish,
            clarifier,
            invocation_guidance,
            output: StageOutput::Evidence,
            after_guard: false,
        },
    )
    .await
}

// --- wrappers (own every status write, gate, and cleanup) -------------------

/// Scripted `plan`: scout → memory → analyst, composed by the script's `plan(input)` entry.
pub async fn run_plan_scripted(
    runtime: WorkflowRuntime,
    ctx: Arc<WorkflowContext>,
) -> Result<PlanOutcome, PlanError> {
    run_plan_scripted_with_turn(runtime, ctx, Arc::new(LiveStageTurn)).await
}

async fn run_plan_scripted_with_turn(
    runtime: WorkflowRuntime,
    ctx: Arc<WorkflowContext>,
    turn: Arc<dyn StageTurn>,
) -> Result<PlanOutcome, PlanError> {
    // The run row first: the issue checkpoint references it, and the schema enforces that.
    ctx.store
        .upsert_run(&ctx.run_id, None, RunStatus::Running.as_str())
        .await?;
    // The registry first: a run's shape says which stages compose each of its nodes, and that is a
    // property of what the run will execute rather than of what the workflow wrote down — a layout
    // may name a node whose stages it never redeclared.
    let stages = install_execution_stages(&ctx, &runtime).await?;
    // A scripted run is measured the same way a built-in one is; the script picks the order, not
    // whether the run is comparable to another afterwards. Failing here fails the run: the shape
    // carries the registry every control is addressed through.
    crate::record_provenance(
        &ctx.store,
        &ctx.run_id,
        &ctx.config,
        &crate::stage::shape_from_workflow(runtime.meta(), &stages),
    )
    .await?;
    checkpoint(
        &ctx.store,
        &ctx.run_id,
        "issue",
        &json!({ "issue": ctx.issue }),
    )
    .await?;

    let stages = install_execution_stages(&ctx, &runtime).await?;
    let hosts = build_hosts_with_turn(&ctx, &stages, turn)?;
    let input = json!({ "issue": ctx.issue }).to_string();
    let result = runtime
        .run_with_question_renderers("plan", input, hosts, stage_question_renderers(&stages))
        .await;

    let outcome = match result {
        Ok(_) => reconstruct_plan(&ctx.store, &ctx.run_id)
            .await
            .map_err(|error| plan_entry_omitted(runtime.meta().name.as_str(), error)),
        Err(e) => Err(PlanError::node(
            "workflow",
            NodeError::Failed(e.to_string()),
        )),
    };
    let status = if outcome.is_ok() {
        RunStatus::Planned
    } else {
        RunStatus::Failed
    };
    if let Err(e) = ctx
        .store
        .upsert_run(&ctx.run_id, None, status.as_str())
        .await
    {
        tracing::warn!("failed to record final run status: {e}");
    }
    warn_about_unclaimed_turns(&ctx);
    ctx.plugin_context.session_end(status.as_str()).await;
    outcome
}

/// Scripted `run`: the full flow via the script's `run(input)` entry. Rust infers the terminal
/// status from checkpoints and does the bookkeeping — the script only sequences.
pub async fn run_full_scripted(
    runtime: WorkflowRuntime,
    ctx: Arc<WorkflowContext>,
) -> Result<RunOutcome, PlanError> {
    let actions = LiveTerminalActions::default();
    run_full_scripted_with_actions(runtime, ctx, &actions).await
}

async fn run_full_scripted_with_actions<A: FullTerminalActions>(
    runtime: WorkflowRuntime,
    ctx: Arc<WorkflowContext>,
    actions: &A,
) -> Result<RunOutcome, PlanError> {
    // The run row first: the issue checkpoint references it, and the schema enforces that.
    ctx.store
        .upsert_run(&ctx.run_id, None, RunStatus::Running.as_str())
        .await?;
    // The registry first: a run's shape says which stages compose each of its nodes, and that is a
    // property of what the run will execute rather than of what the workflow wrote down — a layout
    // may name a node whose stages it never redeclared.
    let stages = install_execution_stages(&ctx, &runtime).await?;
    // A scripted run is measured the same way a built-in one is; the script picks the order, not
    // whether the run is comparable to another afterwards. Failing here fails the run: the shape
    // carries the registry every control is addressed through.
    crate::record_provenance(
        &ctx.store,
        &ctx.run_id,
        &ctx.config,
        &crate::stage::shape_from_workflow(runtime.meta(), &stages),
    )
    .await?;
    checkpoint(
        &ctx.store,
        &ctx.run_id,
        "issue",
        &json!({ "issue": ctx.issue }),
    )
    .await?;

    let hosts = build_hosts(&ctx, &stages)?;
    let input = json!({
        "issue": ctx.issue,
        "maxIterations": ctx.config.implementer.max_iterations,
        "alwaysFork": ctx.config.implementer.always_fork,
    })
    .to_string();

    // Run the script, then reconstruct the outcome. EITHER failing is a run failure: on any error
    // (a script/binding error, or a reconstruction error like a missing checkpoint) the worktree is
    // cleaned up and the run is marked `Failed` — never left orphaned or stuck at `Running`.
    let result = match runtime
        .run_with_question_renderers("run", input, hosts, stage_question_renderers(&stages))
        .await
    {
        Ok(_) => finish_full(&ctx, actions).await,
        Err(e) => Err(PlanError::node(
            "workflow",
            NodeError::Failed(e.to_string()),
        )),
    };

    // Before the cleanup below, not at the end of the function: an abandoned host future is frozen
    // in the runtime's spawner rather than cancelled, and dropping the runtime is what drops it —
    // which is what kills a sandbox child still writing the tree. Removing the worktree first
    // would remove it out from under that child.
    drop(runtime);

    if result.is_err() {
        // Take the handle out before awaiting so the mutex guard isn't held across the await.
        let leftover = ctx.worktree.lock().unwrap().take();
        if let Some(wt) = leftover
            && let Err(rm) = remove_worktree(&ctx.repo_path, &wt).await
        {
            tracing::warn!("failed to clean up worktree after workflow error: {rm}");
        }
        if let Err(e) = ctx
            .store
            .upsert_run(&ctx.run_id, None, RunStatus::Failed.as_str())
            .await
        {
            tracing::warn!("failed to record final run status: {e}");
        }
    }
    // Closed here whichever way the script went, so a plugin cannot tell a scripted run from a
    // built-in one by whether its session ever ended.
    let reason = match &result {
        Ok(outcome) => outcome.status,
        Err(_) => RunStatus::Failed,
    };
    warn_about_unclaimed_turns(&ctx);
    ctx.plugin_context.session_end(reason.as_str()).await;
    result
}

trait FullTerminalActions: Sync {
    async fn commit(
        &self,
        ctx: &WorkflowContext,
        worktree: &WorktreePath,
        implementer: &ImplementerOutput,
    );

    async fn publish(
        &self,
        ctx: &Arc<WorkflowContext>,
        input: PublisherInput,
        terminal: bool,
        worktree: Option<&WorktreePath>,
    ) -> Option<PublisherOutput>;

    async fn bookkeep(
        &self,
        ctx: &Arc<WorkflowContext>,
        input: BookkeeperInput,
    ) -> Option<BookkeeperOutput>;
}

struct LiveTerminalActions {
    publisher_turn: Arc<dyn StageTurn>,
}

impl Default for LiveTerminalActions {
    fn default() -> Self {
        Self {
            publisher_turn: Arc::new(LiveStageTurn),
        }
    }
}

#[cfg(test)]
impl LiveTerminalActions {
    fn with_publisher_turn(publisher_turn: Arc<dyn StageTurn>) -> Self {
        Self { publisher_turn }
    }
}

fn terminal_run(ctx: &WorkflowContext) -> crate::Run<'_> {
    crate::Run {
        client: None,
        configured: &ctx.configured_servers,
        config: &ctx.config,
        store: &ctx.store,
        run_id: &ctx.run_id,
        issue: &ctx.issue,
        engine: &ctx.engine,
        clarifier: &ctx.clarifier,
        context: &ctx.plugin_context,
        ledger: &ctx.ledger,
    }
}

impl FullTerminalActions for LiveTerminalActions {
    async fn commit(
        &self,
        ctx: &WorkflowContext,
        worktree: &WorktreePath,
        implementer: &ImplementerOutput,
    ) {
        crate::commit_worktree(
            &ctx.config,
            &ctx.issue,
            worktree,
            &implementer.branch,
            implementer,
        )
        .await;
    }

    async fn publish(
        &self,
        ctx: &Arc<WorkflowContext>,
        input: PublisherInput,
        terminal: bool,
        worktree: Option<&WorktreePath>,
    ) -> Option<PublisherOutput> {
        let run = terminal_run(ctx);
        crate::publish_if_enabled(
            &run,
            input,
            terminal,
            &ctx.repo_path,
            worktree,
            Arc::clone(&self.publisher_turn),
        )
        .await
    }

    async fn bookkeep(
        &self,
        ctx: &Arc<WorkflowContext>,
        input: BookkeeperInput,
    ) -> Option<BookkeeperOutput> {
        match bookkeep_scripted(ctx, input).await {
            Ok(bookkeeper) => Some(bookkeeper),
            Err(error) => {
                tracing::warn!("bookkeeping failed: {error}");
                None
            }
        }
    }
}

/// Reconstruct the `RunOutcome` from the store after a successful script run, write the Rust-inferred
/// terminal status, commit and deliver the result. Any error here is handled by the caller's cleanup
/// path.
async fn finish_full<A: FullTerminalActions>(
    ctx: &Arc<WorkflowContext>,
    actions: &A,
) -> Result<RunOutcome, PlanError> {
    // The store is the source of truth the script can't fake; a missing checkpoint is a hard error.
    let plan = reconstruct_plan(&ctx.store, &ctx.run_id).await?;
    if !crate::fork_is_needed(&plan.analyst, &ctx.config) {
        let status = RunStatus::NoCodeChange;
        let published = actions
            .publish(
                ctx,
                PublisherInput {
                    issue: ctx.issue.clone(),
                    analyst: plan.analyst.clone(),
                    implementer: None,
                    status: status.as_str().to_string(),
                    iterations: 0,
                    unresolved: Vec::new(),
                },
                true,
                None,
            )
            .await;
        // Publishing may pause for a provider. Do not advertise completion until that last stage
        // returns, or the dashboard would hide the control that can resume the still-live child.
        if let Err(error) = ctx
            .store
            .upsert_run(&ctx.run_id, None, status.as_str())
            .await
        {
            tracing::warn!("failed to record the run's final status: {error}");
        }
        let mut state = plan.state.clone();
        state.status = status;
        state.clarifications.extend(ctx.clarifier.drain());
        if let Some(published) = &published {
            state.artifacts.push(serde_json::to_value(published)?);
        }
        return Ok(RunOutcome {
            state,
            plan,
            red_team: None,
            implementer: None,
            worktree: None,
            iterations: 0,
            status,
            bookkeeper: None,
        });
    }

    let red_team: RedTeamOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "redteam").await?;
    let implementer: ImplementerOutput =
        latest_checkpoint(&ctx.store, &ctx.run_id, "implementer").await?;
    let iterations = count_checkpoints(&ctx.store, &ctx.run_id, "implementer").await?;
    // The worktree is Rust-owned lifecycle state. The implementer checkpoint's rendered path is
    // report data and must never select a terminal file or Git root.
    let worktree = ctx.worktree.lock().unwrap().clone().ok_or_else(|| {
        PlanError::node(
            "publisher",
            NodeError::Failed("run has no worktree".to_string()),
        )
    })?;

    // Terminal status is Rust-inferred, never trusted from the script.
    // The last review, if the script ran one. Absent is not the same as clean: a workflow that
    // never verified simply has no verifier checkpoint, and the warning at run start already said
    // the change would be accepted on its tests alone.
    let checkpoints = ctx.store.checkpoints_for_run(&ctx.run_id).await?;
    let review = scripted_review(&checkpoints);
    let referee = referee_judgement(ctx, &worktree, &plan.analyst, &implementer).await;
    let status = infer_status(
        &red_team,
        &implementer,
        &referee,
        match &review {
            ScriptedReview::Available(output) => Some(output),
            ScriptedReview::NotRun | ScriptedReview::Unavailable => None,
        },
        crate::parse_threshold(&ctx.config.implementer.verify_threshold),
    );
    let status = status_with_review_availability(status, &review);
    actions.commit(ctx, &worktree, &implementer).await;

    let terminal = matches!(
        status,
        RunStatus::Converged | RunStatus::MaxIterationsReached | RunStatus::Unreviewed
    );
    let (bookkeeper, published) = tokio::join!(
        async {
            if !terminal {
                return None;
            }
            actions
                .bookkeep(
                    ctx,
                    BookkeeperInput {
                        issue: ctx.issue.clone(),
                        analyst: plan.analyst.clone(),
                        implementer: implementer.clone(),
                        iterations,
                        converged: status == RunStatus::Converged,
                        friction: crate::friction_of(&ctx.store, &ctx.run_id).await,
                    },
                )
                .await
        },
        actions.publish(
            ctx,
            PublisherInput {
                issue: ctx.issue.clone(),
                analyst: plan.analyst.clone(),
                implementer: Some(implementer.clone()),
                status: status.as_str().to_string(),
                iterations,
                unresolved: crate::unresolved_of(&ctx.store, &ctx.run_id).await,
            },
            terminal,
            Some(&worktree),
        )
    );

    // Publisher and bookkeeper can each make provider requests. The terminal status only means
    // the child has finished every such stage, so keep the stored run resumable until both return.
    if let Err(error) = ctx
        .store
        .upsert_run(&ctx.run_id, None, status.as_str())
        .await
    {
        tracing::warn!("failed to record final run status: {error}");
    }

    let mut state = plan.state.clone();
    state.red_team = Some(serde_json::to_value(&red_team)?);
    state.implementer = Some(serde_json::to_value(&implementer)?);
    state.status = status;
    state.clarifications.extend(ctx.clarifier.drain());
    if let Some(bk) = &bookkeeper {
        state.artifacts = vec![serde_json::to_value(bk)?];
    }
    if let Some(published) = &published {
        state.artifacts.push(serde_json::to_value(published)?);
    }

    Ok(RunOutcome {
        state,
        plan,
        red_team: Some(red_team),
        implementer: Some(implementer),
        worktree: Some(worktree),
        iterations,
        status,
        bookkeeper,
    })
}

/// Compose + write the run-back memory (the `bookkeep_and_checkpoint` body, using the context's
/// tools/sink instead of a `&RagRatClient`).
async fn bookkeep_scripted(
    ctx: &Arc<WorkflowContext>,
    input: BookkeeperInput,
) -> Result<BookkeeperOutput, PlanError> {
    let input_json = serde_json::to_string(&input)?;
    let out = if let Some(output) = bookkeeper::skipped_before_compose(&input, ctx.sink.is_some()) {
        output
    } else {
        let raw = evaluate_standard_stage_with_resources(
            Arc::clone(ctx),
            "bookkeeper",
            input_json,
            StandardStageResources {
                resource_root: ctx.repo_path.clone(),
                capability_ceiling: ratatoskr_core::Capability::Read,
                rag_rat_worktree: None,
                shell: None,
                publish: None,
                // The bundled bookkeeper declares `ask`, and this run's clarifier is right here —
                // the same one whose exchanges this path already drains into the run state.
                clarifier: Some(ctx.clarifier.as_dyn()),
                guidance: None,
            },
        )
        .await
        .map_err(|error| {
            PlanError::node(
                "bookkeeper",
                NodeError::Failed(format!("bookkeeper compose failed: {error}")),
            )
        })?;
        let decisions: bookkeeper::MemoryDecisions =
            serde_json::from_str(&raw).map_err(|error| {
                PlanError::node(
                    "bookkeeper",
                    NodeError::Failed(format!(
                        "bookkeeper decisions could not be reconstructed: {error}"
                    )),
                )
            })?;
        bookkeeper::apply_decisions(
            ctx.sink
                .as_ref()
                .expect("bookkeeper preflight requires a memory sink"),
            decisions.decisions,
            &input,
        )
        .await
        .map_err(|error| PlanError::node("bookkeeper", error))?
    };
    note(ctx, "bookkeeper", &out, None)
        .await
        .map_err(|e| PlanError::node("bookkeeper", NodeError::Failed(e)))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::analyst;

    fn red(failing: &[&str], passing: &[&str], exit: i32) -> RedTeamOutput {
        RedTeamOutput {
            authored: None,
            failing_tests: failing.iter().map(|s| s.to_string()).collect(),
            passed_tests: passing.len(),
            exit_code: exit,
            classifications: vec![],
        }
    }

    fn imp(failing: &[&str], passing: &[&str], exit: i32) -> ImplementerOutput {
        ImplementerOutput {
            branch: "ratatoskr/test".into(),
            worktree_path: "/wt".to_string(),
            diff_summary: String::new(),
            touched_files: vec![],
            rewritten_files: Vec::new(),
            commit_kind: String::new(),
            commit_scope: String::new(),
            commit_subject: String::new(),
            failing_tests: failing.iter().map(|s| s.to_string()).collect(),
            passed_tests: passing.len(),
            exit_code: exit,
            narrative: None,
        }
    }

    fn model_route() -> ratatoskr_core::ModelRoute {
        ratatoskr_core::ModelRoute {
            provider: "test".to_string(),
            model: "test-model".to_string(),
            max_tokens: None,
            context_window: None,
            temperature: None,
            params: None,
            session: ratatoskr_core::SessionScope::Fresh,
        }
    }

    #[tokio::test]
    async fn terminal_run_retains_configured_server_offers() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-terminal-configured-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let config = RatatoskrConfig::default();
        let mut tool = rmcp::model::Tool::default();
        tool.name = "RemotePublish".to_string().into();
        let configured = [ServerTools {
            origin: "remote".to_string(),
            sink: None,
            tools: vec![tool],
            prefix: None,
            renames: std::collections::BTreeMap::new(),
            capabilities: std::collections::BTreeMap::from([(
                "RemotePublish".to_string(),
                ratatoskr_core::Capability::Publish,
            )]),
            provenance: ratatoskr_mcp::ServerProvenance::Configured,
        }];
        let ctx = WorkflowContext::new_with_ledger(WorkflowContextParams {
            client: None,
            configured: &configured,
            config: &config,
            store: &store,
            run_id: "terminal-configured",
            issue: "publish through a configured server",
            engine: &engine,
            plugin_context: crate::PluginContext::default(),
            ledger: Arc::new(ratatoskr_agent::RunLedger::default()),
        })
        .unwrap();

        assert_eq!(
            terminal_run(&ctx).configured[0].display_names(),
            ["RemotePublish"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum TerminalCall {
        Commit {
            branch: String,
            worktree: PathBuf,
        },
        Publish {
            status: String,
            has_implementer: bool,
            terminal: bool,
            iterations: u32,
            unresolved: usize,
        },
        Bookkeep {
            converged: bool,
            iterations: u32,
        },
    }

    struct RecordingTerminalActions {
        calls: Mutex<Vec<TerminalCall>>,
        publisher_worktrees: Mutex<Vec<Option<PathBuf>>>,
        delivery_statuses: Mutex<Vec<Option<String>>>,
        published: Option<PublisherOutput>,
        bookkeeper: Option<BookkeeperOutput>,
    }

    impl RecordingTerminalActions {
        fn new(publish: bool, bookkeep: bool) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                publisher_worktrees: Mutex::new(Vec::new()),
                delivery_statuses: Mutex::new(Vec::new()),
                published: publish.then(|| PublisherOutput {
                    action: crate::publisher::PublisherAction::Comment,
                    pull_request_url: String::new(),
                    comment_url: "https://example.test/comment".to_string(),
                    reasoning: "delivered by the Rust terminal adapter".to_string(),
                }),
                bookkeeper: bookkeep.then(|| BookkeeperOutput {
                    memories_written: Vec::new(),
                    memories_revised: Vec::new(),
                    skipped: Some("nothing durable".to_string()),
                    iterations: 0,
                    residual_risk_accepted: false,
                }),
            }
        }

        fn publisher_worktrees(&self) -> Vec<Option<PathBuf>> {
            self.publisher_worktrees
                .lock()
                .expect("terminal publisher worktree mutex poisoned")
                .clone()
        }

        fn calls(&self) -> Vec<TerminalCall> {
            self.calls
                .lock()
                .expect("terminal calls mutex poisoned")
                .clone()
        }

        fn delivery_statuses(&self) -> Vec<Option<String>> {
            self.delivery_statuses
                .lock()
                .expect("terminal delivery statuses mutex poisoned")
                .clone()
        }
    }

    impl FullTerminalActions for RecordingTerminalActions {
        async fn commit(
            &self,
            _ctx: &WorkflowContext,
            worktree: &WorktreePath,
            implementer: &ImplementerOutput,
        ) {
            self.calls
                .lock()
                .expect("terminal calls mutex poisoned")
                .push(TerminalCall::Commit {
                    branch: implementer.branch.clone(),
                    worktree: worktree.as_path().to_path_buf(),
                });
        }

        async fn publish(
            &self,
            ctx: &Arc<WorkflowContext>,
            input: PublisherInput,
            terminal: bool,
            worktree: Option<&WorktreePath>,
        ) -> Option<PublisherOutput> {
            let stored_status = ctx
                .store
                .run_status(&ctx.run_id)
                .await
                .expect("terminal run status");
            self.delivery_statuses
                .lock()
                .expect("terminal delivery statuses mutex poisoned")
                .push(stored_status);
            self.publisher_worktrees
                .lock()
                .expect("terminal publisher worktree mutex poisoned")
                .push(worktree.map(|worktree| worktree.as_path().to_path_buf()));
            self.calls
                .lock()
                .expect("terminal calls mutex poisoned")
                .push(TerminalCall::Publish {
                    status: input.status,
                    has_implementer: input.implementer.is_some(),
                    terminal,
                    iterations: input.iterations,
                    unresolved: input.unresolved.len(),
                });
            self.published.clone()
        }

        async fn bookkeep(
            &self,
            ctx: &Arc<WorkflowContext>,
            input: BookkeeperInput,
        ) -> Option<BookkeeperOutput> {
            let stored_status = ctx
                .store
                .run_status(&ctx.run_id)
                .await
                .expect("terminal run status");
            self.delivery_statuses
                .lock()
                .expect("terminal delivery statuses mutex poisoned")
                .push(stored_status);
            self.calls
                .lock()
                .expect("terminal calls mutex poisoned")
                .push(TerminalCall::Bookkeep {
                    converged: input.converged,
                    iterations: input.iterations,
                });
            self.bookkeeper.clone().map(|mut output| {
                output.iterations = input.iterations;
                output
            })
        }
    }

    async fn terminal_plan(store: &Store, run_id: &str, changes_code: bool) -> AnalystOutput {
        store
            .upsert_run(run_id, None, RunStatus::Running.as_str())
            .await
            .unwrap();
        checkpoint(
            store,
            run_id,
            "scout",
            &ScoutOutput {
                related_items: Vec::new(),
                papertrail_summary: String::new(),
            },
        )
        .await
        .unwrap();
        checkpoint(store, run_id, "memory", &MemoryOutput::default())
            .await
            .unwrap();
        let analyst = AnalystOutput {
            impact_summary: "exercise terminal parity".to_string(),
            touched: Vec::new(),
            risks: Vec::new(),
            requirements: vec!["keep external effects Rust-owned".to_string()],
            residual_risk: String::new(),
            changes_code,
            acceptance: Vec::new(),
            interface: Vec::new(),
        };
        checkpoint(store, run_id, "analyst", &analyst)
            .await
            .unwrap();
        analyst
    }

    async fn init_test_repo(root: &std::path::Path) -> PathBuf {
        let repo = root.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::write(repo.join("tracked.txt"), "baseline\n").unwrap();
        for args in [
            &["init", "-q", "-b", "main"][..],
            &["config", "user.email", "test@example.com"],
            &["config", "user.name", "Test"],
            &["add", "."],
            &["commit", "-qm", "initial"],
        ] {
            let output = tokio::process::Command::new("git")
                .arg("-C")
                .arg(&repo)
                .args(args)
                .output()
                .await
                .unwrap();
            assert!(output.status.success(), "git {args:?}: {output:?}");
        }
        repo
    }

    #[test]
    fn declared_contracts_validate_all_json_root_values() {
        let mut stage = crate::stage::stage_fixture("publisher", "publish");
        stage.id = "security_evidence".into();
        stage.output_contract = "SecurityEvidence".into();
        stage.output_schema = Some(json!({
            "type": "object",
            "required": ["finding"],
            "properties": { "finding": { "type": "string" } }
        }));

        assert!(validate_declared_output(&stage, &json!({})).is_err());
        assert!(validate_declared_output(&stage, &json!("evidence")).is_err());
        assert!(validate_declared_output(&stage, &json!({ "unrelated": true })).is_err());
        assert!(validate_declared_output(&stage, &json!({ "finding": "validated" })).is_ok());

        stage.output_schema = Some(json!({
            "type": "array",
            "items": { "type": "string" }
        }));
        assert!(validate_declared_output(&stage, &json!(["validated"])).is_ok());
        assert!(validate_declared_output(&stage, &json!("not an array")).is_err());

        stage.output_schema = Some(json!({ "type": "integer" }));
        assert!(validate_declared_output(&stage, &json!(42)).is_ok());
        assert!(validate_declared_output(&stage, &json!([42])).is_err());
    }

    #[test]
    fn declared_stage_guidance_precedes_runtime_input() {
        let mut stage = crate::stage::stage_fixture("analyst", "reason");
        stage.instructions = "stage instructions".to_string();
        stage.context = "stage context".to_string();
        let mut profile = crate::built_in_agents()
            .into_iter()
            .find(|profile| profile.id == "reason")
            .unwrap();
        profile.base_prompt = "agent prompt".to_string();
        let skills = [ratatoskr_plugin::Skill {
            name: "review-skill".to_string(),
            description: "review declared outputs".to_string(),
            body: String::new(),
            dir: PathBuf::new(),
        }];
        let preamble = declared_stage_preamble(
            &stage,
            stage.governance_id(),
            &profile,
            None,
            "repository guidance",
            Some("plugin context"),
            &skills,
        );
        let question = declared_stage_question(&stage, r#"{"runtime":"input"}"#);
        let full_prompt = format!("{preamble}\n\n{question}");

        for (earlier, later) in [
            ("Return JSON", "agent prompt"),
            ("agent prompt", "stage instructions"),
            ("stage instructions", "stage context"),
            ("stage context", "repository guidance"),
            ("repository guidance", "plugin context"),
            ("plugin context", "Available skills:"),
            ("Available skills:", r#"{"runtime":"input"}"#),
        ] {
            assert!(
                full_prompt.find(earlier) < full_prompt.find(later),
                "expected `{earlier}` before `{later}` in {full_prompt}"
            );
        }
        assert!(!preamble.contains(r#"{"runtime":"input"}"#));
    }

    struct RecordingStageTurn {
        sessions: Mutex<Vec<ratatoskr_core::SessionScope>>,
        models: Mutex<Vec<String>>,
        nodes: Mutex<Vec<String>>,
        controls: Mutex<Vec<Option<String>>>,
        conversations: Mutex<Vec<Option<String>>>,
        ledger_ids: Mutex<Vec<Option<usize>>>,
        tools: Mutex<Vec<Vec<String>>>,
        files: Mutex<Vec<Option<std::path::PathBuf>>>,
        rag_rat_worktrees: Mutex<Vec<Option<std::path::PathBuf>>>,
        has_shell: Mutex<Vec<bool>>,
        has_push: Mutex<Vec<bool>>,
        has_clarifier: Mutex<Vec<bool>>,
        preambles: Mutex<Vec<String>>,
        questions: Mutex<Vec<String>>,
        has_ledger: Mutex<Vec<bool>>,
        output: String,
    }

    impl Default for RecordingStageTurn {
        fn default() -> Self {
            Self {
                sessions: Mutex::new(Vec::new()),
                models: Mutex::new(Vec::new()),
                nodes: Mutex::new(Vec::new()),
                controls: Mutex::new(Vec::new()),
                conversations: Mutex::new(Vec::new()),
                ledger_ids: Mutex::new(Vec::new()),
                tools: Mutex::new(Vec::new()),
                files: Mutex::new(Vec::new()),
                rag_rat_worktrees: Mutex::new(Vec::new()),
                has_shell: Mutex::new(Vec::new()),
                has_push: Mutex::new(Vec::new()),
                has_clarifier: Mutex::new(Vec::new()),
                preambles: Mutex::new(Vec::new()),
                questions: Mutex::new(Vec::new()),
                has_ledger: Mutex::new(Vec::new()),
                output: "{}".to_string(),
            }
        }
    }

    impl StageTurn for RecordingStageTurn {
        fn run<'a>(
            &'a self,
            run: ratatoskr_agent::NodeRun<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<String, ratatoskr_agent::AgentError>> + Send + 'a>>
        {
            self.sessions
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.route.session);
            self.models
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.route.model.clone());
            self.nodes
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.node.to_string());
            self.controls
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.controlled_as.map(str::to_string));
            self.conversations
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.conversation.map(str::to_string));
            self.ledger_ids
                .lock()
                .expect("recording runner mutex poisoned")
                .push(
                    run.ledger
                        .as_ref()
                        .map(|ledger| Arc::as_ptr(ledger) as usize),
                );
            self.tools
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.tools.names());
            self.files
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.files.clone());
            self.rag_rat_worktrees
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.rag_rat_worktree.clone());
            self.has_shell
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.shell.is_some());
            self.has_push
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.push.is_some());
            self.has_clarifier
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.clarifier.is_some());
            self.preambles
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.preamble.to_string());
            self.questions
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.question.to_string());
            self.has_ledger
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.ledger.is_some());
            let output = self.output.clone();
            Box::pin(async move { Ok(output) })
        }
    }

    /// A stage turn that charges the run's ledger the way a live one does, so what a checkpoint
    /// reports about cost can be asserted without spending a model turn.
    struct ChargingStageTurn {
        output: String,
        telemetry: ratatoskr_core::NodeTelemetry,
        /// Holds every turn open until they have all recorded, so what a claim sees is a ledger
        /// with another live invocation's turn still standing in it. Without it a turn that
        /// completes synchronously is claimed before the next one starts, and no ordering a
        /// concurrent workflow can produce is exercised at all.
        barrier: Option<Arc<tokio::sync::Barrier>>,
    }

    impl StageTurn for ChargingStageTurn {
        fn run<'a>(
            &'a self,
            run: ratatoskr_agent::NodeRun<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<String, ratatoskr_agent::AgentError>> + Send + 'a>>
        {
            run.ledger
                .as_ref()
                .expect("the executor charges every turn to the run's ledger")
                .record(run.node, self.telemetry.clone());
            let output = self.output.clone();
            let barrier = self.barrier.clone();
            Box::pin(async move {
                if let Some(barrier) = barrier {
                    barrier.wait().await;
                }
                Ok(output)
            })
        }
    }

    struct SequencedStageTurn {
        outputs: Mutex<VecDeque<String>>,
        runs: Mutex<Vec<ObservedStageRun>>,
    }

    struct ObservedStageRun {
        node: String,
        session: ratatoskr_core::SessionScope,
        question: String,
        ledger_id: Option<usize>,
    }

    impl SequencedStageTurn {
        fn new(outputs: impl IntoIterator<Item = serde_json::Value>) -> Self {
            Self {
                outputs: Mutex::new(outputs.into_iter().map(|value| value.to_string()).collect()),
                runs: Mutex::new(Vec::new()),
            }
        }
    }

    impl StageTurn for SequencedStageTurn {
        fn run<'a>(
            &'a self,
            run: ratatoskr_agent::NodeRun<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<String, ratatoskr_agent::AgentError>> + Send + 'a>>
        {
            self.runs
                .lock()
                .expect("sequenced runner mutex poisoned")
                .push(ObservedStageRun {
                    node: run.node.to_string(),
                    session: run.route.session,
                    question: run.question.to_string(),
                    ledger_id: run
                        .ledger
                        .as_ref()
                        .map(|ledger| Arc::as_ptr(ledger) as usize),
                });
            // What the live turn does at the end of a model call, and the reason a claim has
            // anything to take: a turn is recorded under the name it RAN as, and claimed by the
            // checkpoint written under that same name in the same scope.
            if let Some(ledger) = run.ledger.as_ref() {
                ledger.record(
                    run.node,
                    ratatoskr_core::NodeTelemetry {
                        model: Some(run.route.model.clone()),
                        ..Default::default()
                    },
                );
            }
            let output = self
                .outputs
                .lock()
                .expect("sequenced runner mutex poisoned")
                .pop_front()
                .expect("one staged output per model turn");
            Box::pin(async move { Ok(output) })
        }
    }

    struct RecordingCeilingRecovery {
        revised: AnalystOutput,
        implementation: ImplementerOutput,
        revisions: Mutex<Vec<crate::analyst::AnalystInput>>,
        diagnostics: Mutex<Vec<String>>,
    }

    impl CeilingRecovery for RecordingCeilingRecovery {
        async fn revise(
            &self,
            ctx: &Arc<WorkflowContext>,
            _executor: &Arc<StageExecutor>,
            input: &crate::analyst::AnalystInput,
        ) -> Result<Option<AnalystOutput>, String> {
            self.revisions.lock().unwrap().push(input.clone());
            note(
                ctx,
                "analyst",
                &self.revised,
                Some(serde_json::to_string(input).unwrap()),
            )
            .await?;
            Ok(Some(self.revised.clone()))
        }

        async fn iterate(
            &self,
            _ctx: &Arc<WorkflowContext>,
            _worktree: &WorktreePath,
            _revised: &AnalystOutput,
            diagnostic: &str,
        ) -> Result<ImplementerOutput, String> {
            self.diagnostics
                .lock()
                .unwrap()
                .push(diagnostic.to_string());
            Ok(self.implementation.clone())
        }
    }

    #[tokio::test]
    async fn bundled_standard_plan_sequences_typed_checkpointed_stages() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-plan-entry-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "context".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "test".to_string(),
                model: "context-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-standard-plan",
            "preserve the declared plan path",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let turn = Arc::new(SequencedStageTurn::new([
            json!({
                "brief": "the generic stage boundary is load-bearing",
                "constraints": [{ "says": "checkpoint the original input" }],
                "prior_art": [],
                "papertrail_summary": "standard-v1 owns sequencing"
            }),
            json!({
                "impact_summary": "route built-in plan through the bundled runtime",
                "changes_code": true,
                "requirements": ["preserve checkpoint reconstruction"]
            }),
        ]));

        let outcome = run_plan_scripted_with_turn(
            standard_runtime().await.unwrap(),
            Arc::clone(&ctx),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .await
        .unwrap();

        assert_eq!(outcome.state.status, RunStatus::Planned);
        assert_eq!(
            outcome.analyst.impact_summary,
            "route built-in plan through the bundled runtime"
        );
        assert_eq!(outcome.brief, "the generic stage boundary is load-bearing");

        let checkpoints = store
            .checkpoints_for_run("run-standard-plan")
            .await
            .unwrap();
        assert_eq!(
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.node_name.as_str())
                .collect::<Vec<_>>(),
            // The distillation is the context node's model turn and writes its own row inside
            // that node's box; `context` is the box's own aggregate, written by the operation host.
            ["issue", "context_distillation", "context", "analyst"]
        );
        // And the run wrote down the layout the workflow it ran declared, which is what anything
        // drawing this run afterwards places its records against.
        let shape: ratatoskr_core::shape::Recorded = serde_json::from_str(
            store
                .run("run-standard-plan")
                .await
                .unwrap()
                .unwrap()
                .shape_json
                .as_deref()
                .expect("a run records its shape"),
        )
        .unwrap();
        assert_eq!(
            shape,
            crate::stage::shape_from_workflow(
                standard_runtime().await.unwrap().meta(),
                &standard_stages().await.unwrap()
            )
        );
        let context: crate::ContextOutput =
            serde_json::from_str(&checkpoints[2].output_json).unwrap();
        let analyst_input: analyst::AnalystInput =
            serde_json::from_str(checkpoints[3].input_json.as_deref().unwrap()).unwrap();
        assert_eq!(analyst_input.issue, "preserve the declared plan path");
        assert!(analyst_input.brief.is_empty());
        assert!(analyst_input.constraints.is_empty());
        assert_eq!(
            analyst_input.scout.papertrail_summary,
            context.scout.papertrail_summary
        );
        assert_eq!(
            analyst_input.memory.memories.len(),
            context.memory.memories.len()
        );

        let runs = turn.runs.lock().expect("sequenced runner mutex poisoned");
        assert_eq!(
            runs.iter().map(|run| run.node.as_str()).collect::<Vec<_>>(),
            ["context_distillation", "analyst"]
        );
        assert_eq!(runs[1].session, ratatoskr_core::SessionScope::Compacted);
        assert!(runs[0].ledger_id.is_some());
        assert_eq!(runs[0].ledger_id, runs[1].ledger_id);
        assert!(runs[0].question.starts_with(
            "Input contract: ContextDistillationInput\nOutput contract: Distillation\n\n"
        ));
        assert!(runs[0].question.contains("this repository keeps none"));
        assert!(
            runs[1]
                .question
                .starts_with("Input contract: AnalystInput\nOutput contract: AnalystOutput\n\n")
        );
        assert!(runs[1].question.contains("preserve the declared plan path"));
        drop(runs);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn bundled_standard_plan_cannot_succeed_without_required_checkpoints() {
        let runtime = WorkflowRuntime::bundled_with_includes(
            "incomplete-standard-plan",
            r#"defineWorkflow({ name: "incomplete-standard-plan" });
               export async function plan(input) { return input; }"#,
            &[],
            &[],
        )
        .await
        .unwrap();
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-incomplete-standard-plan-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let ctx = WorkflowContext::new(
            None,
            &RatatoskrConfig::default(),
            &store,
            "run-incomplete-standard-plan",
            "skip every stage",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();

        let error =
            match run_plan_scripted_with_turn(runtime, ctx, Arc::new(SequencedStageTurn::new([])))
                .await
            {
                Ok(_) => panic!("a plan without required checkpoints must fail"),
                Err(error) => error,
            };
        // Named, and actionable: a `plan` entry that composes freely still has to drive the two
        // calls the plan is reconstructed from, and nothing else documents that.
        let error = error.to_string();
        for expected in [
            "incomplete-standard-plan",
            "`scout`",
            "context()",
            "analyst()",
        ] {
            assert!(
                error.contains(expected),
                "the refusal never mentions {expected}: {error}"
            );
        }
        assert!(
            !error.contains("converged"),
            "`plan` has no converge loop to ask about: {error}"
        );
        assert_eq!(
            store
                .run_status("run-incomplete-standard-plan")
                .await
                .unwrap()
                .as_deref(),
            Some(RunStatus::Failed.as_str())
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_workflow_that_declares_no_layout_is_still_drawn_one_box_per_node() {
        // A layout is optional and `examples/workflow.ts` declares none, so this is a supported
        // configuration rather than a degenerate one. Such a run records no positions — nothing
        // knows where its nodes belong — but its registry composes them exactly as a laid-out run's
        // does, and the dashboard has to draw one box per node all the same.
        //
        // Read back through the real reader, because everything downstream of the record is name
        // matching: a case on either side of the boundary alone passes its own fixture names in.
        let runtime = WorkflowRuntime::bundled_with_includes(
            "unplaced",
            r#"defineWorkflow({ name: "unplaced" });
               export async function plan(input) {
                 const gathered = await context(input.issue);
                 const analysis = await analyst({
                   issue: input.issue,
                   scout: gathered.scout,
                   memory: gathered.memory,
                 });
                 return { context: gathered, analyst: analysis };
               }"#,
            &[],
            &[],
        )
        .await
        .unwrap();
        assert!(
            runtime.meta().layout.is_empty(),
            "this case is about a workflow that lays out nothing"
        );

        let dir = std::env::temp_dir().join(format!("ratatoskr-unplaced-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let run_id = "run-unplaced";
        let mut config = RatatoskrConfig::default();
        config.models.insert("context".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "draw a run nobody laid out",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        run_plan_scripted_with_turn(
            runtime,
            Arc::clone(&ctx),
            Arc::new(SequencedStageTurn::new([
                json!({
                    "brief": "no layout, same registry",
                    "constraints": [],
                    "prior_art": [],
                    "papertrail_summary": "unplaced"
                }),
                json!({ "impact_summary": "draw the boxes", "changes_code": false }),
            ])) as Arc<dyn StageTurn>,
        )
        .await
        .unwrap();

        let checkpoints = store.checkpoints_for_run(run_id).await.unwrap();
        // The distillation is the context node's model turn and records under its own name; the
        // box's aggregate is `context`. Two rows, one box — which is the whole problem here.
        assert_eq!(
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.node_name.as_str())
                .collect::<Vec<_>>(),
            ["issue", "context_distillation", "context", "analyst"]
        );
        let run = store.run(run_id).await.unwrap().unwrap();
        let recorded = ratatoskr_core::shape::recorded(run.shape_json.as_deref());
        assert!(
            recorded.nodes.is_empty(),
            "a workflow that laid nothing out places nothing"
        );
        assert_eq!(
            recorded.index().members("context"),
            ["context_distillation"]
        );

        let drawn = ratatoskr_serve::pipeline::derive_with(
            Some(RunStatus::Planned.as_str()),
            &checkpoints,
            None,
            run.shape_json.as_deref(),
        );
        // One box per node, not one per stage. `context_distillation` is the context node's work
        // and is drawn inside it.
        assert_eq!(
            drawn
                .iter()
                .map(|node| node.name.as_str())
                .collect::<Vec<_>>(),
            ["context", "analyst"]
        );
        // The record answers where the node list cannot. Before anything has checkpointed there
        // are no nodes at all — the window in which a stage IS executing and an operator reaches
        // for Stop — and the registry still says which box its events belong in. That is why it is
        // shipped beside `nodes` rather than on them.
        assert!(
            ratatoskr_serve::pipeline::derive_with(
                Some(RunStatus::Running.as_str()),
                &[],
                None,
                run.shape_json.as_deref(),
            )
            .is_empty(),
            "nothing has checkpointed, so nothing is placed"
        );
        assert_eq!(recorded.index().node_of("context_distillation"), "context");

        // And the address a control is aimed at is the box the run answers under. `serve` polls
        // for a stop by the name it draws, so a box drawn under a member's name reaches nothing.
        for node in &drawn {
            assert!(
                crate::stage::for_node(&standard_stages().await.unwrap(), &node.name)
                    .is_some_and(|stage| stage.node_id() == node.name)
                    || checkpoints
                        .iter()
                        .any(|checkpoint| checkpoint.node_name == node.name),
                "`{}` is drawn as a box nothing runs or records under",
                node.name
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn the_bundled_definitions_declare_the_box_their_adapter_writes() {
        // The other half of `required_node`'s bolt. Validation refuses an override that names a box
        // its adapter does not write, and this is what keeps the table honest against the
        // definitions that ship: a stage whose declaration and whose policy entry disagree would
        // make the standard workflow itself unloadable, and nothing else would say so.
        let stages = standard_stages().await.unwrap();
        let mut checked = 0;
        for stage in &stages {
            let Some(required) = crate::policy::required_node(&stage.id) else {
                continue;
            };
            assert_eq!(
                stage.node_id(),
                required,
                "`{}` is declared in the box `{}` and its adapter writes `{required}`",
                stage.id,
                stage.node_id()
            );
            checked += 1;
        }
        assert_eq!(
            checked, 4,
            "every stage a Rust caller folds into a box is checked here; found {checked}"
        );
    }

    #[tokio::test]
    async fn bundled_standard_full_sequences_revision_review_and_rust_terminal_actions() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-full-entry-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("worktree")).unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("analyst".to_string(), model_route());
        config.implementer.max_iterations = 3;
        let run_id = "run-standard-full";
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "migrate the standard full flow",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        *ctx.worktree.lock().unwrap() = Some(WorktreePath(dir.join("worktree")));
        store
            .upsert_run(run_id, None, RunStatus::Running.as_str())
            .await
            .unwrap();
        checkpoint(&store, run_id, "issue", &json!({ "issue": ctx.issue }))
            .await
            .unwrap();

        let initial = json!({
            "impact_summary": "use the bundled full workflow",
            "changes_code": true,
            "requirements": ["keep terminal actions in Rust"]
        });
        let revised = json!({
            "impact_summary": "use the bundled full workflow with the corrected plan",
            "changes_code": true,
            "requirements": ["keep terminal actions private to Rust"]
        });
        let turn = Arc::new(SequencedStageTurn::new([initial, revised]));
        let runtime = standard_runtime().await.unwrap();
        let stages = install_execution_stages(&ctx, &runtime).await.unwrap();
        let mut hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        assert!(!hosts.contains_key("publisher"));
        assert!(!hosts.contains_key("bookkeeper"));

        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let context_out = crate::ContextOutput {
            brief: "the workflow runtime owns composition".to_string(),
            constraints: vec![crate::context::Constraint {
                says: "Rust owns terminal side effects".to_string(),
                from_memory_id: "memory-terminal-boundary".to_string(),
            }],
            scout: ScoutOutput {
                related_items: Vec::new(),
                papertrail_summary: "standard-v1 is the built-in runtime".to_string(),
            },
            memory: MemoryOutput::default(),
        };
        let context_calls = Arc::clone(&calls);
        let context_value = context_out.clone();
        hosts.insert(
            "context".to_string(),
            binding(Arc::clone(&ctx), move |ctx, arg| {
                let calls = Arc::clone(&context_calls);
                let output = context_value.clone();
                async move {
                    calls.lock().unwrap().push("context".to_string());
                    note(&ctx, "context", &output, Some(arg)).await?;
                    serde_json::to_string(&output).map_err(|error| error.to_string())
                }
            }),
        );

        let baseline = red(&["pre_existing"], &["baseline_pass"], 1);
        let red_calls = Arc::clone(&calls);
        hosts.insert(
            "redTeam".to_string(),
            binding(Arc::clone(&ctx), move |ctx, _arg| {
                let calls = Arc::clone(&red_calls);
                let output = baseline.clone();
                async move {
                    calls.lock().unwrap().push("redTeam".to_string());
                    note(&ctx, "redteam", &output, None).await?;
                    serde_json::to_string(&output).map_err(|error| error.to_string())
                }
            }),
        );

        let first = ImplementerOutput {
            worktree_path: dir.join("worktree").display().to_string(),
            branch: "ratatoskr/standard-full".to_string(),
            failing_tests: Vec::new(),
            passed_tests: 1,
            exit_code: 0,
            ..imp(&[], &[], 0)
        };
        let implement_calls = Arc::clone(&calls);
        let first_output = first.clone();
        hosts.insert(
            "implement".to_string(),
            binding(Arc::clone(&ctx), move |ctx, arg| {
                let calls = Arc::clone(&implement_calls);
                let output = first_output.clone();
                async move {
                    calls.lock().unwrap().push("implement".to_string());
                    note(&ctx, "implementer", &output, Some(arg)).await?;
                    serde_json::to_string(&output).map_err(|error| error.to_string())
                }
            }),
        );

        let plan_finding = verifier::Finding {
            severity: verifier::Severity::P1,
            kind: verifier::FindingKind::Plan,
            summary: "the terminal boundary is underspecified".to_string(),
            failure_scenario: "a script can invoke delivery directly".to_string(),
            file: "crates/ratatoskr-nodes/src/workflow.rs".to_string(),
            line: Some(1),
        };
        let reviews = Arc::new(Mutex::new(VecDeque::from([
            verifier::VerifierOutput {
                findings: vec![plan_finding],
                assessment: "revise the plan".to_string(),
            },
            verifier::VerifierOutput {
                findings: Vec::new(),
                assessment: "the corrected change is clean".to_string(),
            },
        ])));
        let verify_calls = Arc::clone(&calls);
        let verify_outputs = Arc::clone(&reviews);
        hosts.insert(
            "verify".to_string(),
            binding(Arc::clone(&ctx), move |ctx, arg| {
                let calls = Arc::clone(&verify_calls);
                let outputs = Arc::clone(&verify_outputs);
                async move {
                    calls.lock().unwrap().push("verify".to_string());
                    let supplied: serde_json::Value =
                        serde_json::from_str(&arg).map_err(|error| error.to_string())?;
                    let analyst: AnalystOutput = serde_json::from_value(
                        supplied
                            .get("analyst")
                            .cloned()
                            .ok_or_else(|| "verify input has no analyst".to_string())?,
                    )
                    .map_err(|error| error.to_string())?;
                    let output = outputs
                        .lock()
                        .unwrap()
                        .pop_front()
                        .expect("one verifier result per call");
                    let verifier_input = verifier::VerifierInput {
                        issue: ctx.issue.clone(),
                        analyst,
                        diff: "diff --git a/old b/new".to_string(),
                        touched_files: vec!["src/lib.rs".to_string()],
                        previous_findings: Vec::new(),
                    };
                    note(
                        &ctx,
                        "verifier",
                        &output,
                        Some(serde_json::to_string(&verifier_input).unwrap()),
                    )
                    .await?;
                    serde_json::to_string(&verification_result(output, verifier::Severity::P2))
                        .map_err(|error| error.to_string())
                }
            }),
        );

        let iterate_calls = Arc::clone(&calls);
        let iterated = first.clone();
        hosts.insert(
            "iterate".to_string(),
            binding(Arc::clone(&ctx), move |ctx, arg| {
                let calls = Arc::clone(&iterate_calls);
                let output = iterated.clone();
                async move {
                    calls.lock().unwrap().push("iterate".to_string());
                    note(&ctx, "implementer", &output, Some(arg)).await?;
                    serde_json::to_string(&output).map_err(|error| error.to_string())
                }
            }),
        );

        runtime
            .run_with_question_renderers(
                "run",
                json!({
                    "issue": ctx.issue,
                    "maxIterations": config.implementer.max_iterations,
                    "alwaysFork": false,
                })
                .to_string(),
                hosts,
                stage_question_renderers(&stages),
            )
            .await
            .unwrap();

        assert_eq!(
            calls.lock().unwrap().as_slice(),
            [
                "context",
                "redTeam",
                "implement",
                "verify",
                "iterate",
                "verify"
            ]
        );
        let checkpoints = store.checkpoints_for_run(run_id).await.unwrap();
        assert_eq!(
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.node_name.as_str())
                .collect::<Vec<_>>(),
            [
                "issue",
                "context",
                "analyst",
                "redteam",
                "implementer",
                "verifier",
                "analyst",
                "implementer",
                "verifier",
            ]
        );
        // The bolt for `policy::AGGREGATE_IDENTITIES`, which cannot be derived: it is what says a
        // membership or a layout column may name a box no stage carries the name of, so deriving it
        // from membership would let a workflow authorize its own box. It is held to what a full run
        // records instead, in both directions.
        //
        // Every name this run checkpointed under that no stage of its registry carries — the issue
        // pseudo-node aside, which is the run's input and not a box — is a box a Rust operation
        // host wrote the aggregate of, and so must be listed. A host writing under a new name fails
        // here rather than silently becoming a box nothing will accept.
        let registry = standard_stages().await.unwrap();
        let aggregates = checkpoints
            .iter()
            .map(|checkpoint| checkpoint.node_name.clone())
            .filter(|name| name != "issue" && !registry.iter().any(|stage| stage.id == *name))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            aggregates,
            crate::policy::AGGREGATE_IDENTITIES
                .iter()
                .map(|name| (*name).to_string())
                .collect::<std::collections::BTreeSet<_>>(),
            "a full run records under exactly the boxes policy calls aggregate identities"
        );
        // And every box the standard stages join is one of them, so a membership the bundled
        // definitions declare is one this gate accepts.
        for stage in registry.iter().filter(|stage| !stage.is_own_node()) {
            assert!(
                crate::policy::is_aggregate_identity(stage.node_id()),
                "`{}` joins `{}`, which nothing writes the aggregate of",
                stage.id,
                stage.node_id()
            );
        }
        let revision: analyst::AnalystInput = serde_json::from_str(
            checkpoints[6]
                .input_json
                .as_deref()
                .expect("revision keeps its typed input"),
        )
        .unwrap();
        assert_eq!(revision.brief, context_out.brief);
        assert_eq!(revision.constraints.len(), 1);
        assert!(revision.previous.is_some());
        assert_eq!(revision.findings.len(), 1);

        {
            let runs = turn.runs.lock().unwrap();
            assert_eq!(
                runs.iter().map(|run| run.node.as_str()).collect::<Vec<_>>(),
                ["analyst", "analyst"]
            );
            assert!(
                runs.iter()
                    .all(|run| run.session == ratatoskr_core::SessionScope::Compacted)
            );
            assert_eq!(runs[0].ledger_id, runs[1].ledger_id);
        }

        let actions = RecordingTerminalActions::new(true, true);
        let outcome = finish_full(&ctx, &actions).await.unwrap();
        assert_eq!(outcome.status, RunStatus::Converged);
        assert_eq!(outcome.iterations, 2);
        assert_eq!(outcome.state.artifacts.len(), 2);
        assert!(matches!(
            actions.calls().first(),
            Some(TerminalCall::Commit { .. })
        ));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn ceiling_replan_is_checkpoint_derived_and_can_add_exactly_one_attempt() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-ceiling-replan-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("worktree")).unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.implementer.max_iterations = 1;
        let run_id = "run-standard-ceiling-replan";
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "recover from a sequence of execution findings",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        store
            .upsert_run(run_id, None, RunStatus::Running.as_str())
            .await
            .unwrap();
        *ctx.worktree.lock().unwrap() = Some(WorktreePath(dir.join("worktree")));

        let gathered = crate::ContextOutput {
            brief: "three fixes exposed three different faults".to_string(),
            constraints: Vec::new(),
            scout: ScoutOutput {
                related_items: Vec::new(),
                papertrail_summary: String::new(),
            },
            memory: MemoryOutput::default(),
        };
        note(
            &ctx,
            "context",
            &gathered,
            Some(json!(ctx.issue).to_string()),
        )
        .await
        .unwrap();
        let initial = AnalystOutput {
            impact_summary: "implement the original plan".to_string(),
            touched: Vec::new(),
            risks: Vec::new(),
            requirements: vec!["keep the original behavior".to_string()],
            residual_risk: String::new(),
            changes_code: true,
            acceptance: Vec::new(),
            interface: Vec::new(),
        };
        let initial_input =
            crate::analyst::AnalystInput::from_context(ctx.issue.clone(), gathered.clone());
        note(
            &ctx,
            "analyst",
            &initial,
            Some(serde_json::to_string(&initial_input).unwrap()),
        )
        .await
        .unwrap();
        note(&ctx, "redteam", &red(&[], &["baseline"], 0), None)
            .await
            .unwrap();
        let first = ImplementerOutput {
            worktree_path: dir.join("worktree").display().to_string(),
            ..imp(&[], &["post"], 0)
        };
        note(
            &ctx,
            "implementer",
            &first,
            Some("first attempt".to_string()),
        )
        .await
        .unwrap();
        let finding = verifier::Finding {
            severity: verifier::Severity::P1,
            kind: verifier::FindingKind::Execution,
            summary: "the correction exposed another fault".to_string(),
            failure_scenario: "the edge case is still mishandled".to_string(),
            file: "src/lib.rs".to_string(),
            line: Some(7),
        };
        note(
            &ctx,
            "verifier",
            &verifier::VerifierOutput {
                findings: vec![finding.clone()],
                assessment: "the plan may be the common cause".to_string(),
            },
            None,
        )
        .await
        .unwrap();

        let revised = AnalystOutput {
            impact_summary: "amend the common faulty assumption".to_string(),
            requirements: vec!["handle the edge case explicitly".to_string()],
            ..initial.clone()
        };
        let final_implementation = ImplementerOutput {
            worktree_path: dir.join("worktree").display().to_string(),
            diff_summary: "one bounded recovery".to_string(),
            ..first.clone()
        };
        let recovery = RecordingCeilingRecovery {
            revised: revised.clone(),
            implementation: final_implementation.clone(),
            revisions: Mutex::new(Vec::new()),
            diagnostics: Mutex::new(Vec::new()),
        };
        let stages = Arc::new(standard_stages().await.unwrap());
        let executor = StageExecutor::new(
            Arc::clone(&ctx),
            stages,
            Arc::new(RecordingStageTurn::default()),
        );

        let first_recovery =
            replan_at_ceiling_with(Arc::clone(&ctx), Arc::clone(&executor), &recovery)
                .await
                .unwrap();
        let output: CeilingReplanResult = serde_json::from_str(&first_recovery).unwrap();
        assert_eq!(output.analyst.requirements, revised.requirements);
        assert_eq!(output.implementation.diff_summary, "one bounded recovery");
        assert_eq!(ctx.iterations.load(Ordering::SeqCst), 1);

        let second_recovery = replan_at_ceiling_with(ctx.clone(), executor, &recovery)
            .await
            .unwrap();
        assert_eq!(second_recovery, "null");
        assert_eq!(
            count_checkpoints(&store, run_id, "implementer")
                .await
                .unwrap(),
            2
        );
        let revisions = recovery.revisions.lock().unwrap();
        assert_eq!(revisions.len(), 1);
        assert_eq!(revisions[0].findings.len(), 1);
        assert_eq!(revisions[0].findings[0].summary, finding.summary);
        assert_eq!(
            revisions[0].previous.as_deref().unwrap().requirements,
            initial.requirements
        );
        drop(revisions);
        let diagnostics = recovery.diagnostics.lock().unwrap();
        assert_eq!(diagnostics.len(), 1);
        let requirement = diagnostics[0]
            .find("handle the edge case explicitly")
            .unwrap();
        let evidence = diagnostics[0]
            .find("the correction exposed another fault")
            .unwrap();
        assert!(requirement < evidence);
        drop(diagnostics);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn bundled_standard_full_stops_before_fork_on_an_explicit_no_code_plan() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-full-no-code-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("analyst".to_string(), model_route());
        let run_id = "run-standard-full-no-code";
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "explain the architecture",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        store
            .upsert_run(run_id, None, RunStatus::Running.as_str())
            .await
            .unwrap();
        checkpoint(&store, run_id, "issue", &json!({ "issue": ctx.issue }))
            .await
            .unwrap();
        let turn = Arc::new(SequencedStageTurn::new([json!({
            "impact_summary": "answer from the plan",
            "changes_code": false
        })]));
        let runtime = standard_runtime().await.unwrap();
        let stages = install_execution_stages(&ctx, &runtime).await.unwrap();
        let mut hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        let context_out = crate::ContextOutput {
            brief: String::new(),
            constraints: Vec::new(),
            scout: ScoutOutput {
                related_items: Vec::new(),
                papertrail_summary: String::new(),
            },
            memory: MemoryOutput::default(),
        };
        hosts.insert(
            "context".to_string(),
            binding(Arc::clone(&ctx), move |ctx, arg| {
                let output = context_out.clone();
                async move {
                    note(&ctx, "context", &output, Some(arg)).await?;
                    serde_json::to_string(&output).map_err(|error| error.to_string())
                }
            }),
        );
        for name in ["redTeam", "implement", "verify", "iterate"] {
            hosts.insert(
                name.to_string(),
                Arc::new(move |_| {
                    Box::pin(async move { Err(format!("{name} must not run for no-code work")) })
                }),
            );
        }
        runtime
            .run_with_question_renderers(
                "run",
                json!({
                    "issue": ctx.issue,
                    "maxIterations": config.implementer.max_iterations,
                    "alwaysFork": false,
                })
                .to_string(),
                hosts,
                stage_question_renderers(&stages),
            )
            .await
            .unwrap();

        let actions = RecordingTerminalActions::new(true, true);
        let outcome = finish_full(&ctx, &actions).await.unwrap();
        assert_eq!(outcome.status, RunStatus::NoCodeChange);
        assert_eq!(
            store
                .checkpoints_for_run(run_id)
                .await
                .unwrap()
                .iter()
                .map(|checkpoint| checkpoint.node_name.as_str())
                .collect::<Vec<_>>(),
            ["issue", "context", "analyst"]
        );
        assert_eq!(actions.calls().len(), 1);
        assert!(matches!(actions.calls()[0], TerminalCall::Publish { .. }));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn implementer_host_shares_and_reconstructs_run_clarifications() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-clarifier-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config
            .models
            .insert("implementer".to_string(), model_route());
        let run_id = "run-standard-clarifier";
        let analyst = terminal_plan(&store, run_id, false).await;
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "preserve implementer clarification",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let implementer = build_implementer(&ctx, analyst).await.unwrap();
        let clarifier = implementer
            .clarifier
            .expect("the native implementer receives the run clarifier");
        let _ = clarifier
            .answer(
                "implementer",
                "analyst",
                "Which invariant controls this change?",
                None,
            )
            .await;

        let outcome = finish_full(&ctx, &RecordingTerminalActions::new(false, false))
            .await
            .unwrap();
        assert_eq!(outcome.state.clarifications.len(), 1);
        assert_eq!(outcome.state.clarifications[0]["from"], "implementer");
        assert_eq!(outcome.state.clarifications[0]["to"], "analyst");
        assert!(
            store
                .checkpoints_for_run(run_id)
                .await
                .unwrap()
                .iter()
                .any(|checkpoint| checkpoint.node_name == "clarification")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    struct StaticClarifier;

    impl ratatoskr_agent::Clarifier for StaticClarifier {
        fn answer<'a>(
            &'a self,
            _from: &'a str,
            _to: &'a str,
            _question: &'a str,
            _control: Option<ratatoskr_agent::RuntimeControl>,
        ) -> Pin<Box<dyn Future<Output = ratatoskr_agent::ClarificationAnswer> + Send + 'a>>
        {
            Box::pin(async {
                ratatoskr_agent::ClarificationAnswer::Text("static answer".to_string())
            })
        }
    }

    #[tokio::test]
    async fn stage_executor_applies_the_declared_session_override() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-declared-session-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "declared_review".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-session",
            "review this",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let mut stage = crate::stage::stage_fixture("publisher", "publish");
        stage.id = "declared_review".to_string();
        stage.agent = "reason".to_string();
        stage.output_contract = "ReviewOutput".to_string();
        stage.output_schema = Some(json!({ "type": "object" }));
        stage.session = Some(ratatoskr_core::SessionScope::Compacted);
        let stages = Arc::new(vec![stage.clone()]);
        let turn = Arc::new(RecordingStageTurn::default());
        let executor = StageExecutor::new(ctx, stages, Arc::clone(&turn) as Arc<dyn StageTurn>);

        let output = executor
            .execute(StageInvocation {
                stage,
                input_json: "{}".to_string(),
                rendered_question: None,
                resource_root: None,
                capability_ceiling: ratatoskr_core::Capability::Read,
                rag_rat_worktree: None,
                shell: None,
                publish: None,
                clarifier: None,
                invocation_guidance: None,
                output: StageOutput::Evidence,
            })
            .await
            .unwrap();

        assert_eq!(output, "{}");
        assert_eq!(
            *turn
                .sessions
                .lock()
                .expect("recording runner mutex poisoned"),
            vec![ratatoskr_core::SessionScope::Compacted],
            "the declared stage must override its route before NodeRun reaches the agent"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Two stages sharing one `governedBy` are two pieces of work, and every record a run makes has
    /// to say which of them ran.
    ///
    /// `governedBy` selects the route, the ruleset and the plugin bindings — that is what it is
    /// documented to do and both halves still resolve the same ones. What it must NOT decide is who
    /// ran: the span, the `node_start` event, the ledger claim and the conversation key all name the
    /// stage. The conversation matters beyond bookkeeping — a shared key hands a read-only half the
    /// write-capable half's continued session the moment a route stops being `fresh`.
    ///
    /// The control address is the exception, and deliberately: an operator stops the box they can
    /// see, so both halves poll under the identity the graph draws.
    #[tokio::test]
    async fn stages_sharing_a_governance_identity_still_run_under_their_own() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-split-identity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "shared_box".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "one-route".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-split",
            "share a route without sharing a name",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();

        let half = |id: &str| {
            let mut stage = crate::stage::stage_fixture(id, "reason");
            stage.governed_by = Some("shared_box".to_string());
            stage.output_schema = Some(json!({ "type": "object" }));
            stage
        };
        let first = half("first_half");
        let second = half("second_half");
        let stages = Arc::new(vec![first.clone(), second.clone()]);
        let turn = Arc::new(RecordingStageTurn::default());
        let executor = StageExecutor::new(ctx, stages, Arc::clone(&turn) as Arc<dyn StageTurn>);

        for stage in [first, second] {
            executor
                .execute(StageInvocation {
                    stage,
                    input_json: "{}".to_string(),
                    rendered_question: None,
                    resource_root: None,
                    capability_ceiling: ratatoskr_core::Capability::Read,
                    rag_rat_worktree: None,
                    shell: None,
                    publish: None,
                    clarifier: None,
                    invocation_guidance: None,
                    output: StageOutput::Evidence,
                })
                .await
                .unwrap();
        }

        let recorded = |field: &Mutex<Vec<String>>| {
            field
                .lock()
                .expect("recording runner mutex poisoned")
                .clone()
        };
        assert_eq!(
            recorded(&turn.nodes),
            ["first_half", "second_half"],
            "the span, the node_start event and the ledger claim all name the stage that ran"
        );
        assert_eq!(
            *turn
                .conversations
                .lock()
                .expect("recording runner mutex poisoned"),
            [
                Some("run-split-first_half".to_string()),
                Some("run-split-second_half".to_string())
            ],
            "a shared conversation key would hand one stage the other's session"
        );
        assert_eq!(
            recorded(&turn.models),
            ["one-route", "one-route"],
            "`governedBy` still selects the route both halves run on"
        );
        // Neither declares a membership, so each is its own box and answers at its own address.
        assert_eq!(
            *turn
                .controls
                .lock()
                .expect("recording runner mutex poisoned"),
            [
                Some("first_half".to_string()),
                Some("second_half".to_string())
            ]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The other half of the split: a stage that DOES belong to a node still answers the operator
    /// at that node's address, because the box is what the graph draws and what they can click.
    #[tokio::test]
    async fn the_by_design_unclaimed_names_are_the_ones_a_run_actually_leaves() {
        // The other direction of the guard, and the one that makes the list shrink visibly when a
        // fix lands: a name is listed only if a real invocation really does leave it unclaimed.
        //
        // Behavioural, because there is nothing static to predict from. Whether a turn is claimed
        // depends on the DISPOSITION its caller chose — `execute_after_guard` writes a checkpoint
        // when the invocation checkpoints or the stage belongs to another node — and disposition is
        // a property of the call site, not of the registry. `characterizer` is the proof: it is an
        // ordinary workflow host that checkpoints when a workflow calls it, and `testrun.rs`
        // invokes the same stage as evidence, where nothing claims it.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-unclaimed-listed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config
            .models
            .insert("characterizer".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-unclaimed-listed",
            "characterize",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let turn = Arc::new(SequencedStageTurn::new([json!({
            "checks": [],
            "total": 0
        })]));
        let _ = evaluate_standard_stage_with_turn(
            Arc::clone(&ctx),
            "characterizer",
            json!({ "outcomes": [] }).to_string(),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .await;

        assert_eq!(
            ctx.ledger.unclaimed(),
            ["characterizer"],
            "the evidence invocation leaves its turn for nobody, which is why it is listed"
        );
        for name in ctx.ledger.unclaimed() {
            assert!(
                UNCLAIMED_BY_DESIGN
                    .iter()
                    .any(|(known, reason)| *known == name && reason.len() > 20),
                "`{name}` goes unclaimed and is not listed with a reason"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_real_run_leaves_no_turn_unclaimed() {
        // The same invariant against a run rather than the registry, on the path that drives real
        // operation hosts. `SequencedStageTurn` records into the ledger exactly as the live turn
        // does, so what the executor claims — and what it leaves behind — is the real arithmetic.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-unclaimed-guard-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("context".to_string(), model_route());
        config.models.insert("analyst".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-unclaimed-guard",
            "leave nothing in the bin",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let turn = Arc::new(SequencedStageTurn::new([
            json!({
                "brief": "b",
                "constraints": [],
                "prior_art": [],
                "papertrail_summary": "p"
            }),
            json!({ "impact_summary": "i", "changes_code": false }),
        ]));
        run_plan_scripted_with_turn(
            standard_runtime().await.unwrap(),
            Arc::clone(&ctx),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .await
        .unwrap();

        assert_eq!(
            turn.runs
                .lock()
                .expect("sequenced runner mutex poisoned")
                .len(),
            2,
            "both turns ran, so there was something to claim"
        );
        assert!(
            ctx.ledger.unclaimed().is_empty(),
            "a run left turns nobody claimed: {:?}",
            ctx.ledger.unclaimed()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn every_standard_stage_is_controlled_at_the_address_its_run_draws() {
        // The guard, not a case. A Stop is written under the name the dashboard offers — the box a
        // run RECORDS for that stage — and the stage's turn has to be polled under that same name
        // or the button reaches nothing. Both halves are read from the real thing here: the address
        // comes from the recorded shape (`Registry::node_of`), which is what `serve` ships and what
        // the pause ledger keys, and the identity the turn is given comes from running the stage
        // through the executor. A member stage added tomorrow is covered without anyone
        // remembering, because the registry is what this iterates.
        //
        // The other half of the chain — that a turn given an identity actually polls under it — is
        // `ratatoskr-agent`'s `a_member_stage_is_polled_for_control_under_the_box_an_operator_addresses`,
        // which drives the real hook. Two independent statements: this one cannot restate the
        // executor's own expression, and that one cannot be satisfied by a field being set.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-control-guard-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-control-guard", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let runtime = standard_runtime().await.unwrap();
        let stages = standard_stages().await.unwrap();
        let registry = crate::stage::shape_from_workflow(runtime.meta(), &stages);
        let registry = registry.index();

        // A route per governance identity, so no stage is turned away before its turn is built.
        let mut config = RatatoskrConfig::default();
        for stage in &stages {
            config
                .models
                .insert(stage.governance_id().to_string(), model_route());
        }
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-control-guard",
            "stop anything",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let turn = Arc::new(RecordingStageTurn::default());
        let executor = StageExecutor::new(
            ctx,
            Arc::new(stages.clone()),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        );

        for stage in &stages {
            // The result is not the subject: a stage's output gate may reject this generic answer,
            // and the identity it was to be controlled under was decided before the turn ran.
            let _ = executor
                .execute(StageInvocation {
                    stage: stage.clone(),
                    input_json: "{}".to_string(),
                    rendered_question: Some("anything".to_string()),
                    resource_root: None,
                    capability_ceiling: ratatoskr_core::Capability::Read,
                    rag_rat_worktree: None,
                    shell: None,
                    publish: None,
                    clarifier: None,
                    invocation_guidance: None,
                    output: StageOutput::Evidence,
                })
                .await;
            let control = turn
                .controls
                .lock()
                .expect("recording runner mutex poisoned")
                .last()
                .cloned()
                .unwrap_or_else(|| panic!("`{}` never reached its turn", stage.id));
            assert_eq!(
                control.as_deref(),
                Some(registry.node_of(&stage.id)),
                "`{}` is controlled under a name the run does not draw a box for",
                stage.id
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_member_stage_is_controlled_at_its_nodes_address() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-member-control-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-member-control", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "redteam".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-member-control",
            "stop the red team",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let turn = Arc::new(RecordingStageTurn::default());
        let executor = StageExecutor::new(
            ctx,
            Arc::new(stages.clone()),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        );
        let stage = stages
            .iter()
            .find(|stage| stage.id == "redteam_classifier")
            .unwrap()
            .clone();
        executor
            .execute(StageInvocation {
                stage,
                input_json: serde_json::to_string(&crate::redteam::ClassifierInput {
                    failing: Vec::new(),
                    raw_output: String::new(),
                })
                .unwrap(),
                rendered_question: Some("classify these".to_string()),
                resource_root: None,
                capability_ceiling: ratatoskr_core::Capability::Read,
                rag_rat_worktree: None,
                shell: None,
                publish: None,
                clarifier: None,
                invocation_guidance: None,
                output: StageOutput::Evidence,
            })
            .await
            .unwrap();

        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["redteam_classifier"],
            "the record says which half ran"
        );
        assert_eq!(
            *turn
                .controls
                .lock()
                .expect("recording runner mutex poisoned"),
            [Some("redteam".to_string())],
            "and an operator stops the box the graph drew"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_declared_stage_gets_file_mutation_tools_only_against_a_supplied_root() {
        // A declared stage host owns no worktree lifecycle. Without a root from Rust its file tools
        // would resolve against the process's working directory — the operator's own checkout — so
        // the declaration alone must not put `Write` or `Edit` on the table.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-declared-write-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "leak".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-declared-write",
            "write something",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let mut stage = crate::stage::stage_fixture("publisher", "publish");
        stage.id = "leak".to_string();
        stage.agent = "build".to_string();
        stage.governed_by = None;
        stage.output_contract = "LeakOutput".to_string();
        stage.output_schema = Some(json!({ "type": "object" }));
        stage.capabilities = vec![ratatoskr_core::Capability::Write];
        stage.tools = [
            "Read",
            "Write",
            "Edit",
            "Bash",
            ratatoskr_agent::ASK_TOOL_NAME,
        ]
        .map(str::to_string)
        .to_vec();
        stage.question_renderer = None;
        let stages = Arc::new(vec![stage.clone()]);
        let turn = Arc::new(RecordingStageTurn::default());
        let executor = StageExecutor::new(ctx, stages, Arc::clone(&turn) as Arc<dyn StageTurn>);

        // Root alone, ceiling alone, and both: only the invocation that was granted BOTH a root and
        // a `write` ceiling may mutate. The read-only grant is `verify_host`'s: a review turn is
        // handed the implementer's worktree to read, and a `capabilities: ["write"]` override of
        // the verifier must not turn that root into Edit/Write in the tree it judges.
        let clarifier: Arc<dyn ratatoskr_agent::Clarifier> = Arc::new(StaticClarifier);
        for (resource_root, ceiling, clarifier) in [
            (None, ratatoskr_core::Capability::Write, None),
            (Some(dir.clone()), ratatoskr_core::Capability::Read, None),
            (
                Some(dir.clone()),
                ratatoskr_core::Capability::Write,
                Some(Arc::clone(&clarifier)),
            ),
        ] {
            executor
                .execute(StageInvocation {
                    stage: stage.clone(),
                    input_json: "{}".to_string(),
                    rendered_question: None,
                    resource_root,
                    capability_ceiling: ceiling,
                    rag_rat_worktree: None,
                    shell: None,
                    publish: None,
                    clarifier,
                    invocation_guidance: None,
                    output: StageOutput::Evidence,
                })
                .await
                .unwrap();
        }

        let offered = turn.tools.lock().expect("recording runner mutex poisoned");
        for tool in ["Write", "Edit"] {
            assert!(
                !offered[0].iter().any(|offered| offered == tool),
                "{tool} was offered to a stage Rust gave no root"
            );
            assert!(
                !offered[1].iter().any(|offered| offered == tool),
                "{tool} was offered to a stage Rust granted only `read` in the supplied root"
            );
            assert!(
                offered[2].iter().any(|offered| offered == tool),
                "{tool} must still be offered against a supplied root and a `write` grant"
            );
        }
        // The same rule for the shell, which no invocation was granted.
        assert!(
            !offered.iter().any(|run| run.iter().any(|t| t == "Bash")),
            "Bash was offered without a shell grant"
        );
        // And for `ask`: without a clarifier the call reaches a stub that errors every time, so the
        // only thing an offer buys is turns spent discovering that.
        for run in &offered[..2] {
            assert!(
                !run.iter()
                    .any(|tool| tool == ratatoskr_agent::ASK_TOOL_NAME),
                "`ask` was offered without a clarifier to answer it"
            );
        }
        assert!(
            offered[2]
                .iter()
                .any(|tool| tool == ratatoskr_agent::ASK_TOOL_NAME),
            "`ask` must still be offered where Rust wired a clarifier"
        );
        // Reading is not gated on the root, and never was.
        assert!(offered[0].iter().any(|tool| tool == "Read"));
        drop(offered);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn arbitrary_declared_stage_id_uses_the_generic_executor() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-arbitrary-declared-stage-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-arbitrary-stage", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "arbitrary_probe".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-arbitrary-stage",
            "probe this",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let mut stage = crate::stage::stage_fixture("publisher", "publish");
        stage.id = "arbitrary_probe".to_string();
        stage.agent = "reason".to_string();
        stage.input_contract = "ProbeInput".to_string();
        stage.output_contract = "ProbeOutput".to_string();
        stage.output_schema = Some(json!({
            "type": "object",
            "required": ["ok"],
            "properties": { "ok": { "type": "boolean" } }
        }));
        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "ok": true }).to_string(),
            ..Default::default()
        });
        let mut hosts =
            build_hosts_with_turn(&ctx, &[stage], Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();

        let output = hosts.remove("arbitrary_probe").unwrap()("{}".to_string())
            .await
            .unwrap();

        assert_eq!(output, json!({ "ok": true }).to_string());
        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["arbitrary_probe"]
        );
        let checkpoints = store
            .checkpoints_for_run("run-arbitrary-stage")
            .await
            .unwrap();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].node_name, "arbitrary_probe");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A workflow may call one declared stage twice at once — `Promise.all([probe(a), probe(b)])`
    /// — and nothing stops it: only `iterate`, `verify` and `replanAtCeiling` hold the iterate
    /// lock, and `implement`/`redTeam` have order guards. Two such invocations are two turns and
    /// two records, and each record has to report what its own invocation spent.
    #[tokio::test]
    async fn concurrent_invocations_of_one_stage_each_report_their_own_cost() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-concurrent-declared-stage-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-concurrent-stage", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "concurrent_probe".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-concurrent-stage",
            "probe this twice",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let mut stage = crate::stage::stage_fixture("concurrent_probe", "reason");
        stage.input_contract = "ProbeInput".to_string();
        stage.output_contract = "ProbeOutput".to_string();
        stage.output_schema = Some(json!({
            "type": "object",
            "required": ["ok"],
            "properties": { "ok": { "type": "boolean" } }
        }));
        let turn = ChargingStageTurn {
            output: json!({ "ok": true }).to_string(),
            telemetry: ratatoskr_core::NodeTelemetry {
                model: Some("anthropic/test-model".to_string()),
                duration_ms: Some(50),
                usage: ratatoskr_core::TokenUsage {
                    input_tokens: 10,
                    output_tokens: 100,
                    ..Default::default()
                },
                turns: Some(1),
                ..Default::default()
            },
            // Both turns record before either checkpoint claims — the ordering the ledger has to
            // survive, and the one a real pair of overlapping model turns produces.
            barrier: Some(Arc::new(tokio::sync::Barrier::new(2))),
        };
        let hosts =
            build_hosts_with_turn(&ctx, &[stage], Arc::new(turn) as Arc<dyn StageTurn>).unwrap();
        let host = hosts.get("concurrent_probe").unwrap().clone();

        let (first, second) = tokio::join!(host("{}".to_string()), host("{}".to_string()));
        first.unwrap();
        second.unwrap();

        let checkpoints = store
            .checkpoints_for_run("run-concurrent-stage")
            .await
            .unwrap();
        assert_eq!(
            checkpoints.len(),
            2,
            "each invocation writes its own record"
        );
        for checkpoint in &checkpoints {
            assert_eq!(
                checkpoint.telemetry.usage.input_tokens, 10,
                "each record reports its own invocation's turn, not both and not neither"
            );
            assert_eq!(checkpoint.telemetry.usage.output_tokens, 100);
            assert_eq!(checkpoint.telemetry.turns, Some(1));
            assert_eq!(
                checkpoint.telemetry.model.as_deref(),
                Some("anthropic/test-model")
            );
        }
        assert!(
            ctx.ledger.unclaimed().is_empty(),
            "both turns were claimed by a checkpoint"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn generic_executor_consumes_a_rendered_question_and_checkpoints_original_input() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-rendered-stage-question-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-rendered-question", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "rendered_probe".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-rendered-question",
            "render this",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let mut stage = crate::stage::stage_fixture("publisher", "publish");
        stage.id = "rendered_probe".to_string();
        stage.agent = "reason".to_string();
        stage.input_contract = "ProbeInput".to_string();
        stage.output_contract = "ProbeOutput".to_string();
        stage.output_schema = Some(json!({ "type": "object" }));
        stage.instructions = "stage guidance".to_string();
        stage.question_renderer = Some("function (input) { return input.issue; }".to_string());
        let turn = Arc::new(RecordingStageTurn::default());
        let mut hosts =
            build_hosts_with_turn(&ctx, &[stage], Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        let envelope = json!({
            "__ratatoskrRenderedQuestion": {
                "input": { "issue": "original checkpoint input" },
                "question": "CONDITIONALLY RENDERED QUESTION"
            }
        })
        .to_string();

        hosts.remove("rendered_probe").unwrap()(envelope)
            .await
            .unwrap();

        {
            let preambles = turn
                .preambles
                .lock()
                .expect("recording runner mutex poisoned");
            let questions = turn
                .questions
                .lock()
                .expect("recording runner mutex poisoned");
            let full_prompt = format!("{}\n\n{}", preambles[0], questions[0]);
            assert!(
                full_prompt.find("stage guidance") < full_prompt.find("CONDITIONALLY RENDERED")
            );
            assert!(questions[0].contains("Input contract: ProbeInput"));
            assert!(questions[0].ends_with("CONDITIONALLY RENDERED QUESTION"));
            assert!(!questions[0].contains("original checkpoint input"));
        }
        let checkpoints = store
            .checkpoints_for_run("run-rendered-question")
            .await
            .unwrap();
        assert_eq!(
            checkpoints[0].input_json.as_deref(),
            Some(r#"{"issue":"original checkpoint input"}"#)
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn operation_host_adapters_remain_registered_and_guarded() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-temporary-operations-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let ctx = WorkflowContext::new(
            None,
            &RatatoskrConfig::default(),
            &store,
            "run-temporary-operations",
            "test adapters",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let hosts =
            build_hosts_with_turn(&ctx, &[], Arc::new(RecordingStageTurn::default())).unwrap();

        for (name, _) in OPERATION_HOSTS {
            assert!(
                hosts.contains_key(*name),
                "missing operation adapter `{name}`"
            );
        }
        // `memory` was superseded by the composite `context` operation, which is what guarantees
        // the ranked search happened. Nothing calls it, so it is not a host.
        for removed in [
            "memory",
            "redteam",
            "implementer",
            "newlyIntroducedFailures",
        ] {
            assert!(
                !hosts.contains_key(removed),
                "obsolete operation alias `{removed}` was re-registered"
            );
        }
        assert!(
            !hosts.contains_key("analyst"),
            "the canonical analyst id belongs to the declarative stage"
        );
        assert!(
            !hosts.contains_key("verifier"),
            "the canonical verifier id belongs to the declarative stage"
        );
        assert!(
            !hosts.contains_key("analyze"),
            "the direct analyst compatibility path must not remain registered"
        );
        let error = hosts["verify"]("{}".to_string()).await.unwrap_err();
        assert!(
            error.contains("verify arg"),
            "operation host argument check changed: {error}"
        );

        ctx.invocations.store(INVOCATION_CEILING, Ordering::Relaxed);
        let error = hosts["iterate"]("{}".to_string()).await.unwrap_err();
        assert!(error.contains("runaway loop"));

        let context = crate::stage::stage_fixture("context", "explore");
        let error = match build_hosts_with_turn(
            &ctx,
            &[context],
            Arc::new(RecordingStageTurn::default()),
        ) {
            Ok(_) => panic!("a declared stage replaced an operation host"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("workflow operation host"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn repository_workflows_cannot_invoke_standard_terminal_stages() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-terminal-stage-boundary-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"defineWorkflow({ name: "terminal-probe" });
               export async function plan(input) {
                 return input.target === "publisher"
                   ? await publisher(input)
                   : await bookkeeper(input);
               }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let ctx = WorkflowContext::new(
            None,
            &RatatoskrConfig::default(),
            &store,
            "run-terminal-stage-boundary",
            "try a terminal stage",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = install_execution_stages(&ctx, &runtime).await.unwrap();
        let hosts = build_hosts(&ctx, &stages).unwrap();
        assert!(!hosts.contains_key("publisher"));
        assert!(!hosts.contains_key("bookkeeper"));

        for target in ["publisher", "bookkeeper"] {
            let error = runtime
                .run_with_question_renderers(
                    "plan",
                    json!({ "target": target }).to_string(),
                    hosts.clone(),
                    stage_question_renderers(&stages),
                )
                .await
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(target),
                "terminal call error changed: {error}"
            );
            assert!(
                error.contains("not defined"),
                "terminal stage unexpectedly had a host: {error}"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn repository_workflows_cannot_invoke_internal_write_stages() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-internal-write-stage-boundary-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"defineWorkflow({ name: "write-stage-probe" });
               export async function plan(input) {
                 return input.target === "redteam_author"
                   ? await redteam_author(input)
                   : await implementer_attempt(input);
               }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let ctx = WorkflowContext::new(
            None,
            &RatatoskrConfig::default(),
            &store,
            "run-internal-write-stage-boundary",
            "try an internal write stage",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = install_execution_stages(&ctx, &runtime).await.unwrap();
        for stage_id in ["redteam_author", "implementer_attempt"] {
            let stage = stages
                .iter()
                .find(|stage| stage.id == stage_id)
                .expect("internal stage remains available to Rust adapters");
            assert_eq!(stage.capabilities, [ratatoskr_core::Capability::Write]);
        }
        let turn = Arc::new(RecordingStageTurn::default());
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        assert!(hosts.contains_key("redTeam"));
        assert!(hosts.contains_key("implement"));
        for stage_id in ["redteam_author", "implementer_attempt"] {
            assert!(!hosts.contains_key(stage_id));
            let error = runtime
                .run_with_question_renderers(
                    "plan",
                    json!({ "target": stage_id }).to_string(),
                    hosts.clone(),
                    stage_question_renderers(&stages),
                )
                .await
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(stage_id),
                "write-stage error changed: {error}"
            );
            assert!(
                error.contains("not defined"),
                "internal write stage unexpectedly had a JS host: {error}"
            );
        }
        assert!(
            turn.nodes
                .lock()
                .expect("recording runner mutex poisoned")
                .is_empty(),
            "rejected repository calls must not reach a model turn"
        );
        assert!(
            store
                .checkpoints_for_run("run-internal-write-stage-boundary")
                .await
                .unwrap()
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_scout_uses_generic_dispatch_and_checkpoints_normalized_output() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-standard-scout-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-standard-scout", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.get_mut("scout").unwrap().session = ratatoskr_core::SessionScope::Compacted;
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-standard-scout",
            "find prior work",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let scout = stages
            .iter()
            .find(|stage| stage.id == "scout")
            .unwrap()
            .clone();
        assert_eq!(scout.agent, "explore");
        assert_eq!(scout.capabilities, [ratatoskr_core::Capability::Read]);
        assert_eq!(scout.tools, ["papertrail_issue_search", "semantic_search"]);
        let mut declared = scout.output_schema.clone().unwrap();
        let mut generated =
            serde_json::to_value(schemars::schema_for!(crate::scout::ScoutOutput)).unwrap();
        without_schema_annotations(&mut declared);
        without_schema_annotations(&mut generated);
        assert_eq!(declared, generated, "schema drift for scout");
        assert_eq!(scout.session, None, "TOML retains the session decision");

        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "related_items": [
                    {
                        "item_key": "  ",
                        "title": "",
                        "url": "",
                        "relation": "",
                        "summary": ""
                    },
                    {
                        "item_key": "214",
                        "title": "Reusable stages",
                        "url": "https://example.test/214",
                        "relation": "same execution seam",
                        "summary": "introduced declared stages"
                    }
                ],
                "papertrail_summary": "One related change."
            })
            .to_string(),
            ..Default::default()
        });
        let mut hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        let host = hosts.remove(&scout.id).unwrap();

        let raw = host(serde_json::to_string("find prior work").unwrap())
            .await
            .unwrap();
        let output: ScoutOutput = serde_json::from_str(&raw).unwrap();
        assert_eq!(output.related_items.len(), 1);
        assert_eq!(output.related_items[0].item_key, "214");

        let checkpoints = store
            .checkpoints_for_run("run-standard-scout")
            .await
            .unwrap();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].node_name, "scout");
        let checkpoint: ScoutOutput = serde_json::from_str(&checkpoints[0].output_json).unwrap();
        assert_eq!(checkpoint.related_items.len(), 1);

        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["scout"]
        );
        assert_eq!(
            *turn
                .sessions
                .lock()
                .expect("recording runner mutex poisoned"),
            [ratatoskr_core::SessionScope::Compacted]
        );
        assert!(
            turn.has_ledger
                .lock()
                .expect("recording runner mutex poisoned")[0],
            "generic dispatch must report telemetry to the run ledger"
        );
        let offered = &turn.tools.lock().expect("recording runner mutex poisoned")[0];
        assert!(offered.contains(&"Read".to_string()));
        assert!(offered.contains(&"Grep".to_string()));
        assert!(!offered.iter().any(|tool| tool == "Write" || tool == "Edit"));
        let preamble = &turn
            .preambles
            .lock()
            .expect("recording runner mutex poisoned")[0];
        assert!(
            preamble.find("Return JSON") < preamble.find("You are the scout"),
            "platform guidance must precede the bundled scout prompt"
        );
        let question = &turn
            .questions
            .lock()
            .expect("recording runner mutex poisoned")[0];
        assert!(question.ends_with(r#""find prior work""#));
        assert!(!preamble.contains("find prior work"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_analyst_renders_revisions_and_reuses_its_compacted_run_session() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-standard-analyst-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"export async function plan(input) {
                 await analyst(input.fresh);
                 return await analyst(input.revision);
               }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let rules_dir = dir.join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();
        let engine = ScriptEngine::load(&rules_dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-standard-analyst", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.get_mut("analyst").unwrap().session = ratatoskr_core::SessionScope::Fresh;
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-standard-analyst",
            "preserve the contract",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let analyst = stages.iter().find(|stage| stage.id == "analyst").unwrap();
        assert_eq!(
            analyst.session,
            Some(ratatoskr_core::SessionScope::Compacted)
        );
        assert_eq!(analyst.capabilities, [ratatoskr_core::Capability::Read]);
        assert_eq!(
            analyst.tools,
            ["impact_surface", "symbol_lookup", "semantic_search"]
        );

        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "impact_summary": "preserve the interface" }).to_string(),
            ..Default::default()
        });
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        let fresh = json!({
            "issue": "preserve the contract",
            "brief": "The writer is single-owner.",
            "constraints": [{
                "says": "Keep one writer",
                "from_memory_id": "mem_writer"
            }],
            "scout": {
                "related_items": [],
                "papertrail_summary": "The existing plan introduced Store::claim."
            },
            "memory": { "memories": [] }
        });
        let revision = json!({
            "issue": "preserve the contract",
            "brief": "The writer is single-owner.",
            "constraints": [],
            "scout": {
                "related_items": [],
                "papertrail_summary": "The existing plan introduced Store::claim."
            },
            "memory": { "memories": [] },
            "previous": {
                "impact_summary": "Add a claim operation.",
                "requirements": ["Keep one writer"],
                "interface": [{
                    "name": "Store::claim",
                    "shape": "fn claim(&self, run: &str) -> Result<Claim, StoreError>",
                    "happy": ["an unclaimed run yields a Claim"],
                    "sad": ["a claimed run errors rather than blocking"]
                }]
            },
            "findings": [{
                "severity": "P1",
                "kind": "plan",
                "file": "crates/store.rs",
                "summary": "The retry contract was omitted",
                "failure_scenario": "a second claim waits forever"
            }]
        });
        runtime
            .run_with_question_renderers(
                "plan",
                json!({ "fresh": fresh, "revision": revision }).to_string(),
                hosts,
                stage_question_renderers(&stages),
            )
            .await
            .unwrap();

        {
            let questions = turn
                .questions
                .lock()
                .expect("recording runner mutex poisoned");
            assert_eq!(questions.len(), 2);
            assert!(questions[0].contains("ISSUE:\npreserve the contract"));
            assert!(questions[0].contains("WHAT BEARS ON THIS:\nThe writer is single-owner."));
            assert!(questions[0].contains("Keep one writer [mem_writer]"));
            assert!(!questions[0].contains("THIS IS A REVISION"));
            assert!(questions[1].contains("THIS IS A REVISION"));
            assert!(questions[1].contains("Tests are already written against it"));
            assert!(questions[1].contains("Store::claim"));
            assert!(questions[1].contains("fn claim(&self, run: &str)"));
            assert!(questions[1].contains("happy: an unclaimed run yields a Claim"));
            assert!(questions[1].contains("sad: a claimed run errors rather than blocking"));
            assert!(questions[1].contains("[P1] (crates/store.rs) The retry contract was omitted"));
        }
        assert_eq!(
            *turn
                .sessions
                .lock()
                .expect("recording runner mutex poisoned"),
            [
                ratatoskr_core::SessionScope::Compacted,
                ratatoskr_core::SessionScope::Compacted
            ],
            "the declared stage session must override the fresh TOML route"
        );
        assert_eq!(
            *turn
                .conversations
                .lock()
                .expect("recording runner mutex poisoned"),
            [
                Some("run-standard-analyst-analyst".to_string()),
                Some("run-standard-analyst-analyst".to_string())
            ]
        );
        {
            let ledger_ids = turn
                .ledger_ids
                .lock()
                .expect("recording runner mutex poisoned");
            assert_eq!(ledger_ids.len(), 2);
            assert_eq!(ledger_ids[0], ledger_ids[1]);
            assert!(ledger_ids[0].is_some());
        }

        let checkpoints = store
            .checkpoints_for_run("run-standard-analyst")
            .await
            .unwrap();
        assert_eq!(checkpoints.len(), 2);
        assert!(checkpoints.iter().all(|row| row.node_name == "analyst"));
        let first: analyst::AnalystInput =
            serde_json::from_str(checkpoints[0].input_json.as_deref().unwrap()).unwrap();
        assert_eq!(first.issue, "preserve the contract");
        assert!(first.previous.is_none());
        let second: analyst::AnalystInput =
            serde_json::from_str(checkpoints[1].input_json.as_deref().unwrap()).unwrap();
        assert_eq!(second.previous.unwrap().interface[0].name, "Store::claim");
        assert_eq!(second.findings[0].summary, "The retry contract was omitted");
        let _ = std::fs::remove_dir_all(dir);
    }

    fn without_schema_annotations(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(object) => {
                object.remove("$schema");
                object.remove("title");
                object.remove("description");
                object.remove("default");
                object.remove("format");
                for value in object.values_mut() {
                    without_schema_annotations(value);
                }
            }
            serde_json::Value::Array(array) => {
                for value in array {
                    without_schema_annotations(value);
                }
            }
            _ => {}
        }
    }

    #[tokio::test]
    async fn standard_stage_prompts_are_exact_and_schemas_do_not_default_output() {
        let stages = standard_stages().await.unwrap();
        let expected_prompts = [
            ("analyst", include_str!("../prompts/analyst.md")),
            ("bookkeeper", include_str!("../prompts/bookkeeper.md")),
            ("characterizer", include_str!("../prompts/characterizer.md")),
            (
                "context_distillation",
                include_str!("../prompts/context.md"),
            ),
            (
                "implementer_attempt",
                include_str!("../prompts/implementer.md"),
            ),
            ("overseer", include_str!("../prompts/overseer.md")),
            ("publisher", include_str!("../prompts/publisher.md")),
            (
                "redteam_author",
                include_str!("../prompts/redteam-author.md"),
            ),
            (
                "redteam_classifier",
                include_str!("../prompts/redteam-classifier.md"),
            ),
            ("scout", include_str!("../prompts/scout.md")),
            ("verifier", include_str!("../prompts/verifier.md")),
        ];
        assert_eq!(stages.len(), expected_prompts.len());
        for (stage_id, expected_prompt) in expected_prompts {
            let stage = stages.iter().find(|stage| stage.id == stage_id).unwrap();
            assert_eq!(
                stage.instructions,
                expected_prompt.trim(),
                "prompt drift for {stage_id}"
            );
            let declared = stage.output_schema.as_ref().unwrap();
            let mut without_defaults = declared.clone();
            without_schema_defaults(&mut without_defaults);
            assert_eq!(
                declared, &without_defaults,
                "workflow schema materializes output defaults for {stage_id}"
            );
            assert!(
                !declared.to_string().contains("\"format\""),
                "workflow schema uses a non-portable format annotation for {stage_id}"
            );
        }
    }

    #[test]
    fn publisher_prompt_treats_the_committed_result_as_immutable() {
        let prompt = include_str!("../prompts/publisher.md");

        assert!(prompt.contains("Never change the repository."));
        assert!(prompt.contains("having a tool is not permission to use it here"));
        assert!(prompt.contains("through\n`git_push` and `gh`"));
        assert!(prompt.contains("do not repair it"));
    }

    struct RendererParityCase {
        stage: &'static str,
        input: serde_json::Value,
        expected_question: &'static str,
    }

    #[tokio::test]
    async fn standard_typescript_renderers_produce_the_exact_prompt_text() {
        let cases = vec![
            RendererParityCase {
                stage: "overseer",
                input: json!({ "issue": "x", "choices": [] }),
                expected_question: "AVAILABLE WORKFLOWS:\n\nTHE TASK:\nx\n",
            },
            RendererParityCase {
                stage: "characterizer",
                input: json!({ "outcomes": [] }),
                expected_question: "",
            },
            RendererParityCase {
                stage: "redteam_classifier",
                input: json!({ "failing": ["a"], "raw_output": "boom" }),
                expected_question: "These tests fail in the current baseline (before any change):\na\n\nTest output:\nboom\n\nClassify each as \"flaky\" or \"real\" with a one-line reason.",
            },
            RendererParityCase {
                stage: "redteam_author",
                input: json!({ "issue": "x", "interface": [] }),
                expected_question: "THE TASK, for context only:\nx\n\nTHE INTERFACE. This is the contract, and it is all you get — the code does not exist yet, and the person writing it is working from this same description:\n\n\nWrite tests for these. Follow the repository's own layout and conventions, cover the sad cases as carefully as the happy ones, and change nothing that already exists.",
            },
            RendererParityCase {
                stage: "implementer_attempt",
                input: json!({ "diagnostic": "fix it" }),
                expected_question: "fix it",
            },
            RendererParityCase {
                stage: "context_distillation",
                input: json!({ "issue": "x", "memory": { "memories": [] }, "searchable": false }),
                expected_question: "TASK:\nx\n\nRECORDED MEMORIES: this repository keeps none — there is no memory index here. Work from what you can read.\n",
            },
            RendererParityCase {
                stage: "analyst",
                input: json!({
                    "issue": "x",
                    "scout": { "papertrail_summary": "", "related_items": [] },
                    "memory": { "memories": [] }
                }),
                expected_question: "ISSUE:\nx\n\nSCOUT SUMMARY:\n\n\n",
            },
            RendererParityCase {
                stage: "bookkeeper",
                input: json!({
                    "converged": true,
                    "iterations": 1,
                    "issue": "x",
                    "analyst": { "impact_summary": "", "risks": [] },
                    "implementer": { "diff_summary": "", "narrative": null, "touched_files": [], "failing_tests": [] },
                    "friction": { "diagnostics": [], "errors": [], "effort": [] }
                }),
                expected_question: "OUTCOME: the run CONVERGED — the change landed and the tests pass.\n\nTASK:\nx\n\n",
            },
            RendererParityCase {
                stage: "publisher",
                input: json!({
                    "issue": "x",
                    "status": "converged",
                    "iterations": 0,
                    "unresolved": [],
                    "analyst": { "impact_summary": "", "requirements": [] },
                    "implementer": null
                }),
                expected_question: "THE TASK:\nx\n\nOUTCOME: converged after 0 implementer iteration(s).\n\nNO CODE WAS CHANGED. This run produced an answer, not a change — there is nothing to open a pull request for.\n",
            },
            RendererParityCase {
                stage: "verifier",
                input: json!({
                    "issue": "x",
                    "analyst": { "requirements": [], "impact_summary": "", "risks": [] },
                    "touched_files": [],
                    "previous_findings": [],
                    "diff": "diff\n"
                }),
                expected_question: "TASK:\nx\n\nTHE CHANGE:\ndiff\n\n",
            },
        ];
        let calls = cases
            .iter()
            .enumerate()
            .map(|(index, case)| format!("await {}(input.c{index});", case.stage))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!(
            "defineWorkflow({{ name: \"renderer-parity\" }}); export async function run(input) {{ {calls} return true; }}"
        );
        let runtime = WorkflowRuntime::bundled_with_includes("renderer-parity", &source, &[], &[])
            .await
            .unwrap();
        let captured = Arc::new(Mutex::new(HashMap::<String, Vec<serde_json::Value>>::new()));
        let mut hosts: HashMap<String, HostFn> = HashMap::new();
        for case in &cases {
            let stage = case.stage.to_string();
            let captured = Arc::clone(&captured);
            let capture_key = stage.clone();
            hosts.entry(stage).or_insert_with(|| {
                Arc::new(move |arg: String| {
                    let captured = Arc::clone(&captured);
                    let capture_key = capture_key.clone();
                    Box::pin(async move {
                        captured
                            .lock()
                            .expect("renderer capture mutex poisoned")
                            .entry(capture_key)
                            .or_default()
                            .push(serde_json::from_str(&arg).unwrap());
                        Ok("{}".to_string())
                    })
                })
            });
        }
        let input = serde_json::Value::Object(
            cases
                .iter()
                .enumerate()
                .map(|(index, case)| (format!("c{index}"), case.input.clone()))
                .collect(),
        );
        let stages = standard_stages().await.unwrap();
        runtime
            .run_with_question_renderers(
                "run",
                input.to_string(),
                hosts,
                stage_question_renderers(&stages),
            )
            .await
            .unwrap();

        let captured = captured.lock().expect("renderer capture mutex poisoned");
        let mut offsets = HashMap::<&str, usize>::new();
        for case in &cases {
            let offset = offsets.entry(case.stage).or_default();
            let envelope = &captured[case.stage][*offset];
            *offset += 1;
            assert_eq!(
                envelope["__ratatoskrRenderedQuestion"]["question"], case.expected_question,
                "rendered question drift for {}",
                case.stage
            );
        }
    }

    #[tokio::test]
    async fn bundled_default_renderer_ownership_is_explicit_and_complete() {
        // Both direct declared-stage calls and Rust-owned lifecycle adapters enter the bundled
        // renderer before StageExecutor. Scout accepts its issue string directly and has no
        // renderQuestion declaration.
        let typescript_rendered = [
            "analyst",
            "overseer",
            "characterizer",
            "redteam_classifier",
            "redteam_author",
            "implementer_attempt",
            "context_distillation",
            "bookkeeper",
            "publisher",
            "verifier",
        ];
        let stages = standard_stages().await.unwrap();
        let mut declared = stage_question_renderers(&stages)
            .into_keys()
            .collect::<Vec<_>>();
        declared.sort();
        let mut owned = typescript_rendered
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        owned.sort();
        assert_eq!(owned, declared, "every bundled renderer needs one owner");
        assert!(
            stages
                .iter()
                .find(|stage| stage.id == "scout")
                .is_some_and(|stage| stage.question_renderer.is_none())
        );
    }

    #[tokio::test]
    async fn an_overridden_verifier_agent_is_what_decides_whether_review_runs() {
        // Enablement asked a fixed table which agent the verifier used, so moving it to a
        // configured profile made `verifier_enabled` resolve through the *imported* stage, find no
        // model, and report `{configured:false}` — a run that converged with no review at all and
        // said nothing about it.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-verifier-agent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"import * as nodes from "ratatoskr/nodes";
               defineWorkflow({
                 name: "ours",
                 stages: [stage("verifier", { ...nodes.verifier, agent: "reason" })],
               });
               export async function plan(input) { return input; }"#,
        )
        .unwrap();
        let definitions = standard_definitions().unwrap();
        let runtime = WorkflowRuntime::load(
            &workflow_path,
            &[(STANDARD_DEFINITIONS_MODULE, definitions.as_str())],
        )
        .await
        .unwrap()
        .unwrap();

        let rules_dir = dir.join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();
        let engine = ScriptEngine::load(&rules_dir).await.unwrap();
        // Only `reason` has a model: no `[models.verifier]`, and `explore` — the agent the
        // imported verifier uses — has none.
        let mut config = RatatoskrConfig::default();
        config.agents.insert(
            "reason".to_string(),
            ratatoskr_core::AgentProfileConfig {
                model: Some(ratatoskr_core::ModelRoute {
                    provider: "anthropic".to_string(),
                    model: "test-model".to_string(),
                    max_tokens: None,
                    context_window: None,
                    temperature: None,
                    params: None,
                    session: Default::default(),
                }),
                ..Default::default()
            },
        );

        let standard = standard_stages().await.unwrap();
        assert!(
            !crate::verifier_enabled(&engine, &config, &standard),
            "without the override there is nowhere for the verifier to run"
        );
        let stages = overlaid_stages(&runtime).await.unwrap();
        assert_eq!(
            stages
                .iter()
                .find(|stage| stage.id == "verifier")
                .map(|stage| stage.agent.as_str()),
            Some("reason")
        );
        assert!(
            crate::verifier_enabled(&engine, &config, &stages),
            "the registry the run executes decides, so the override's profile enables review"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn selection_and_review_are_not_callable_from_a_workflow() {
        // `scripted_review` reads the *last* `verifier` checkpoint, so a workflow that could call
        // `verifier({..})` after `verify()` would answer the gate that judges it. `overseer` is
        // refused as a declaration; installing it as a host anyway would let a workflow burn the
        // `[models.overseer]` route and overwrite the recorded routing decision.
        let store = Store::open_in_memory().unwrap();
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-no-gate-host-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let ctx = WorkflowContext::new(
            None,
            &RatatoskrConfig::default(),
            &store,
            "run-no-gate-host",
            "try to answer the gate",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let hosts = build_hosts_with_turn(
            &ctx,
            &stages,
            Arc::new(RecordingStageTurn::default()) as Arc<dyn StageTurn>,
        )
        .unwrap();
        for gate in ["verifier", "overseer"] {
            assert!(
                stages.iter().any(|stage| stage.id == gate),
                "`{gate}` must stay in the registry for the Rust adapter that runs it"
            );
            assert!(
                !hosts.contains_key(gate),
                "`{gate}` must not be reachable from a workflow"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_overridden_standard_stage_is_the_one_a_run_executes() {
        // The registry a run actually executes from. The override keeps the id `analyst`, so the
        // standard definition must be gone rather than sitting ahead of it — the by-id scan that
        // picks a stage to execute and resolves a delegation target takes the first match.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-exec-override-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"import * as nodes from "ratatoskr/nodes";
               defineWorkflow({
                 name: "ours",
                 stages: [
                   stage("analyst", { ...nodes.analyst, instructions: "our own analysis" }),
                   stage("reviewer", {
                     agent: "reason",
                     instructions: "review",
                     delegation: { target: "analyst", evidenceContract: "AnalystOutput" },
                   }),
                 ],
               });
               export async function plan(input) { return input; }"#,
        )
        .unwrap();
        let definitions = standard_definitions().unwrap();
        let runtime = WorkflowRuntime::load(
            &workflow_path,
            &[(STANDARD_DEFINITIONS_MODULE, definitions.as_str())],
        )
        .await
        .unwrap()
        .unwrap();

        let stages = overlaid_stages(&runtime).await.unwrap();
        assert_eq!(
            stages.iter().filter(|stage| stage.id == "analyst").count(),
            1,
            "the override replaces the imported stage instead of joining it"
        );
        let resolved = stages.iter().find(|stage| stage.id == "analyst").unwrap();
        assert_eq!(resolved.instructions, "our own analysis");
        // Everything the override did not restate is still the standard definition's.
        let standard = standard_stages().await.unwrap();
        let imported = standard.iter().find(|stage| stage.id == "analyst").unwrap();
        assert_ne!(imported.instructions, resolved.instructions);
        assert_eq!(resolved.output_schema, imported.output_schema);
        // And the delegation target resolves through the same scan.
        let reviewer = stages.iter().find(|stage| stage.id == "reviewer").unwrap();
        let target = reviewer.delegation.as_ref().unwrap().target.clone();
        assert_eq!(
            stages
                .iter()
                .find(|stage| stage.id == target)
                .unwrap()
                .instructions,
            "our own analysis"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_override_of_an_adapter_invoked_stage_is_what_the_model_turn_runs() {
        // `implementer_attempt` and `redteam_author` never appear in the JavaScript host table:
        // they run from Rust lifecycle adapters. Those adapters used to build a registry of their
        // own, so an override of either validated at startup and was then ignored by the turn that
        // actually ran. Asserted through the adapter entry point, not the registry.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-adapter-override-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"defineWorkflow({
                 name: "ours",
                 stages: [
                   stage("implementer_attempt", {
                     agent: "build",
                     governedBy: "implementer",
                     instructions: "OVERRIDDEN IMPLEMENTER",
                     outputSchema: { type: "object" },
                     renderQuestion(input: any) { return "OVERRIDDEN IMPLEMENTER QUESTION"; },
                   }),
                   stage("redteam_author", {
                     agent: "build",
                     governedBy: "redteam",
                     instructions: "OVERRIDDEN AUTHOR",
                     outputSchema: { type: "object" },
                     renderQuestion(input: any) { return "OVERRIDDEN AUTHOR QUESTION"; },
                   }),
                 ],
               });
               export async function run(input) { return input; }"#,
        )
        .unwrap();
        let definitions = standard_definitions().unwrap();
        let runtime = WorkflowRuntime::load(
            &workflow_path,
            &[(STANDARD_DEFINITIONS_MODULE, definitions.as_str())],
        )
        .await
        .unwrap()
        .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("redteam".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-adapter-override",
            "override an adapter-invoked stage",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        install_execution_stages(&ctx, &runtime).await.unwrap();

        for (stage_id, sentinel) in [
            ("implementer_attempt", "OVERRIDDEN IMPLEMENTER"),
            ("redteam_author", "OVERRIDDEN AUTHOR"),
        ] {
            // Both are identifiers Rust reads back as a type, so the stub output has to be one.
            let turn = Arc::new(RecordingStageTurn {
                output: json!({ "summary": "the override ran" }).to_string(),
                ..Default::default()
            });
            evaluate_standard_stage_with_turn(
                Arc::clone(&ctx),
                stage_id,
                "{}".to_string(),
                Arc::clone(&turn) as Arc<dyn StageTurn>,
            )
            .await
            .unwrap();
            let preambles = turn
                .preambles
                .lock()
                .expect("recording runner mutex poisoned");
            let questions = turn
                .questions
                .lock()
                .expect("recording runner mutex poisoned");
            assert_eq!(preambles.len(), 1, "`{stage_id}` ran once");
            assert!(
                preambles[0].contains(sentinel),
                "the adapter ran the standard `{stage_id}` instead of the override: {}",
                preambles[0]
            );
            assert!(
                questions[0].ends_with(&format!("{sentinel} QUESTION")),
                "the override's renderQuestion is not the one that composed the turn: {}",
                questions[0]
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_delegation_invoked_as_evidence_is_refused_rather_than_dropped() {
        // `context_distillation` is both a global a workflow may call directly (checkpointed —
        // delegation runs) and the stage `context()` invokes as evidence. Load-time validation
        // cannot tell one invocation from the other, so the evidence path has to refuse the
        // declaration rather than run the parent turn with the child silently omitted.
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-evidence-delegation-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"import * as nodes from "ratatoskr/nodes";
               defineWorkflow({
                 name: "ours",
                 stages: [
                   stage("context_distillation", {
                     ...nodes.context_distillation,
                     delegation: { target: "helper", inputLimit: 65536 },
                   }),
                   stage("helper", { agent: "explore", instructions: "help" }),
                 ],
               });
               export async function plan(input) { return input; }"#,
        )
        .unwrap();
        let definitions = standard_definitions().unwrap();
        let runtime = WorkflowRuntime::load(
            &workflow_path,
            &[(STANDARD_DEFINITIONS_MODULE, definitions.as_str())],
        )
        .await
        .unwrap()
        .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("context".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-evidence-delegation",
            "delegate from an evidence stage",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        install_execution_stages(&ctx, &runtime).await.unwrap();
        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "brief": "a brief",
                "constraints": [],
                "prior_art": [],
                "papertrail_summary": ""
            })
            .to_string(),
            ..Default::default()
        });
        let error = evaluate_standard_stage_with_turn(
            Arc::clone(&ctx),
            "context_distillation",
            json!({ "issue": "x", "memory": { "memories": [] }, "searchable": false }).to_string(),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .await
        .expect_err("a delegation this caller cannot honour must be refused");
        assert!(
            error.contains("context_distillation") && error.contains("helper"),
            "the refusal names neither the stage nor its delegation target: {error}"
        );
        assert!(
            turn.nodes
                .lock()
                .expect("recording runner mutex poisoned")
                .is_empty(),
            "the parent turn ran with its delegation dropped"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_overridden_implementer_attempt_runs_on_its_own_profiles_model() {
        // Startup validation never consults `config.models` for a stage, so an override whose
        // profile carries its own model is legal with no `[models.implementer]` at all. The
        // lifecycle adapter used to resolve a second route of its own, against the *built-in*
        // stage table rather than the run's registry, and refused the run with
        // `MissingRoute("implementer")` before the executor — which resolves the override
        // correctly — ever ran.
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-implementer-profile-route-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"defineWorkflow({
                 name: "ours",
                 stages: [
                   stage("implementer_attempt", {
                     agent: "shipwright",
                     governedBy: "implementer",
                     inputContract: "ImplementerAttemptInput",
                     outputContract: "Report",
                     outputSchema: { type: "object" },
                     instructions: "OURS",
                     capabilities: ["write"],
                   }),
                 ],
               });
               export async function run(input) { return input; }"#,
        )
        .unwrap();
        let definitions = standard_definitions().unwrap();
        let runtime = WorkflowRuntime::load(
            &workflow_path,
            &[(STANDARD_DEFINITIONS_MODULE, definitions.as_str())],
        )
        .await
        .unwrap()
        .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        // The headline case: the override brings its own model, and the run has none.
        config.models.remove("implementer");
        config.agents.insert(
            "shipwright".to_string(),
            ratatoskr_core::AgentProfileConfig {
                model: Some(ratatoskr_core::ModelRoute {
                    model: "shipwright-model".to_string(),
                    ..model_route()
                }),
                capabilities: vec![ratatoskr_core::Capability::Write],
                ..Default::default()
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-implementer-profile-route",
            "override the implementer's stage",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        install_execution_stages(&ctx, &runtime).await.unwrap();

        let analyst = AnalystOutput {
            impact_summary: "The override supplies the route.".to_string(),
            touched: Vec::new(),
            risks: Vec::new(),
            requirements: Vec::new(),
            residual_risk: String::new(),
            changes_code: true,
            acceptance: Vec::new(),
            interface: Vec::new(),
        };
        let node = build_implementer(&ctx, analyst).await;
        assert!(
            node.is_ok(),
            "an override carrying its own model is a complete route: {:?}",
            node.err()
        );

        // And the turn that actually runs is on that model.
        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "summary": "the override ran" }).to_string(),
            ..Default::default()
        });
        evaluate_standard_stage_with_turn(
            Arc::clone(&ctx),
            "implementer_attempt",
            "{}".to_string(),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .await
        .unwrap();
        assert_eq!(
            turn.models
                .lock()
                .expect("recording runner mutex poisoned")
                .as_slice(),
            ["shipwright-model"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_override_that_omits_render_question_gets_no_renderer_at_all() {
        // An override restates only what it changes, so omitting `renderQuestion` is how a stage
        // says it wants its structured input. The bundled workflow is evaluated on every adapter
        // call and registers the standard renderers as it goes; installing the run's renderers over
        // that left the bundled one answering for a stage that had dropped it, and a replacement
        // that also changed its input contract got a prompt written for the shape it replaced.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-omitted-renderer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"defineWorkflow({
                 name: "ours",
                 stages: [
                   stage("implementer_attempt", {
                     agent: "build",
                     governedBy: "implementer",
                     instructions: "OVERRIDDEN IMPLEMENTER",
                     inputContract: "OurImplementerInput",
                     outputContract: "OurReport",
                     outputSchema: { type: "object" },
                   }),
                   stage("redteam_author", {
                     agent: "build",
                     governedBy: "redteam",
                     instructions: "OVERRIDDEN AUTHOR",
                     inputContract: "OurAuthorInput",
                     outputContract: "OurTests",
                     outputSchema: { type: "object" },
                     renderQuestion(input: any) { return `AUTHOR: ${input.note}`; },
                   }),
                 ],
               });
               export async function run(input) { return input; }"#,
        )
        .unwrap();
        let definitions = standard_definitions().unwrap();
        let runtime = WorkflowRuntime::load(
            &workflow_path,
            &[(STANDARD_DEFINITIONS_MODULE, definitions.as_str())],
        )
        .await
        .unwrap()
        .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("redteam".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-omitted-renderer",
            "drop a bundled renderer",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        install_execution_stages(&ctx, &runtime).await.unwrap();

        // The bundled `implementer_attempt` renderer reads `input.analyst.impact_summary`; this
        // input is the override's own shape and has no such field.
        let input = json!({ "note": "the shape this override declares" }).to_string();
        // Both stages are read back as a type, so the stub output has to be one.
        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "summary": "the override ran" }).to_string(),
            ..Default::default()
        });
        evaluate_standard_stage_with_turn(
            Arc::clone(&ctx),
            "implementer_attempt",
            input.clone(),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .await
        .unwrap();
        // And a stage that did restate `renderQuestion` still gets it: installing the run's table
        // exactly must not swing into installing nothing.
        evaluate_standard_stage_with_turn(
            Arc::clone(&ctx),
            "redteam_author",
            input.clone(),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .await
        .unwrap();

        let questions = turn
            .questions
            .lock()
            .expect("recording runner mutex poisoned");
        assert_eq!(
            questions[0],
            format!("Input contract: OurImplementerInput\nOutput contract: OurReport\n\n{input}"),
            "the stage that dropped renderQuestion must receive its structured input"
        );
        assert_eq!(
            questions[1],
            "Input contract: OurAuthorInput\nOutput contract: OurTests\n\n\
             AUTHOR: the shape this override declares"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_repository_workflow_may_take_the_bundled_name() {
        // Provenance is a property of the runtime, not of its declared name. Comparing against
        // `ratatoskr-standard-v1` made a repository workflow that chose that name lose every
        // declaration it made — its own stages and its overrides both.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-bundled-name-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"import * as nodes from "ratatoskr/nodes";
               defineWorkflow({
                 name: "ratatoskr-standard-v1",
                 stages: [
                   stage("analyst", { ...nodes.analyst, instructions: "our own analysis" }),
                   stage("reviewer", { agent: "reason", instructions: "review" }),
                 ],
               });
               export async function plan(input) { return input; }"#,
        )
        .unwrap();
        let definitions = standard_definitions().unwrap();
        let runtime = WorkflowRuntime::load(
            &workflow_path,
            &[(STANDARD_DEFINITIONS_MODULE, definitions.as_str())],
        )
        .await
        .unwrap()
        .unwrap();
        assert!(!runtime.is_bundled());
        assert!(standard_runtime().await.unwrap().is_bundled());

        let stages = overlaid_stages(&runtime).await.unwrap();
        assert_eq!(
            stages
                .iter()
                .find(|stage| stage.id == "analyst")
                .unwrap()
                .instructions,
            "our own analysis"
        );
        assert!(
            stages.iter().any(|stage| stage.id == "reviewer"),
            "a workflow's own stages must survive its choice of name"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn bookkeeper_declaration_matches_its_typed_schema_and_rust_question() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-bookkeeper-renderer-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            "export async function run(input) { return await bookkeeper(input); }",
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let stages = standard_stages().await.unwrap();
        // The declaration stays in the run's registry — the Rust terminal adapter resolves it
        // there. What a repository workflow never gets is a host to call it with, which
        // `repository_workflows_cannot_invoke_terminal_stages` holds.
        assert!(
            overlaid_stages(&runtime)
                .await
                .unwrap()
                .iter()
                .any(|stage| stage.id == "bookkeeper"),
            "the terminal adapter must still be able to resolve its declaration"
        );
        let bookkeeper = stages
            .iter()
            .find(|stage| stage.id == "bookkeeper")
            .unwrap();
        assert_eq!(bookkeeper.agent, "reason");
        assert_eq!(bookkeeper.capabilities, [ratatoskr_core::Capability::Read]);
        assert_eq!(
            bookkeeper.tools,
            ["semantic_search", "symbol_lookup", "memory_search", "ask"]
        );
        assert_eq!(bookkeeper.session, None);

        let mut declared = bookkeeper.output_schema.clone().unwrap();
        let mut generated =
            serde_json::to_value(schemars::schema_for!(crate::bookkeeper::MemoryDecisions))
                .unwrap();
        without_schema_annotations(&mut declared);
        without_schema_annotations(&mut generated);
        assert_eq!(declared, generated, "schema drift for bookkeeper");

        let input = crate::bookkeeper::BookkeeperInput {
            issue: "Preserve the run's expensive lesson".to_string(),
            analyst: AnalystOutput {
                impact_summary: "Memory application remains Rust-owned".to_string(),
                touched: vec!["crates/ratatoskr-nodes".to_string()],
                risks: vec!["a model could attempt an unvalidated write".to_string()],
                requirements: Vec::new(),
                residual_risk: String::new(),
                changes_code: true,
                acceptance: Vec::new(),
                interface: Vec::new(),
            },
            implementer: ImplementerOutput {
                worktree_path: "/tmp/ratatoskr/worktree".to_string(),
                branch: "ratatoskr/bookkeeper".to_string(),
                diff_summary: " bookkeeper.rs | 4 ++++".to_string(),
                touched_files: vec!["bookkeeper.rs".to_string()],
                rewritten_files: Vec::new(),
                commit_kind: "refactor".to_string(),
                commit_scope: "nodes".to_string(),
                commit_subject: "declare the bookkeeper".to_string(),
                failing_tests: vec!["bookkeeper::schema".to_string()],
                passed_tests: 3,
                exit_code: 101,
                narrative: Some("The writer must remain deterministic.".to_string()),
            },
            iterations: 2,
            converged: false,
            friction: crate::bookkeeper::RunFriction {
                diagnostics: vec!["The decision omitted its action.".to_string()],
                errors: vec![crate::bookkeeper::NodeFailure {
                    node: "bookkeeper".to_string(),
                    error: "schema validation failed".to_string(),
                }],
                effort: vec![crate::bookkeeper::NodeEffort {
                    node: "bookkeeper".to_string(),
                    turns: 7,
                    seconds: 12,
                }],
            },
        };
        let captured = Arc::new(Mutex::new(None));
        let capture = Arc::clone(&captured);
        let host: HostFn = Arc::new(move |arg| {
            let capture = Arc::clone(&capture);
            Box::pin(async move {
                *capture.lock().expect("bookkeeper capture mutex poisoned") =
                    Some(serde_json::from_str::<serde_json::Value>(&arg).unwrap());
                Ok(json!({ "decisions": [] }).to_string())
            })
        });
        runtime
            .run_with_question_renderers(
                "run",
                serde_json::to_string(&input).unwrap(),
                HashMap::from([("bookkeeper".to_string(), host)]),
                stage_question_renderers(&stages),
            )
            .await
            .unwrap();
        let envelope = captured
            .lock()
            .expect("bookkeeper capture mutex poisoned")
            .take()
            .unwrap();
        let question = envelope["__ratatoskrRenderedQuestion"]["question"]
            .as_str()
            .unwrap();
        assert!(question.contains("OUTCOME: the run HIT A WALL"));
        assert!(question.contains("The decision omitted its action."));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn invalid_bookkeeper_decisions_cannot_write_memory_or_checkpoint() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-bookkeeper-invalid-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-bookkeeper-invalid", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "bookkeeper".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-bookkeeper-invalid",
            "remember this",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = Arc::new(standard_stages().await.unwrap());
        let bookkeeper = stages
            .iter()
            .find(|stage| stage.id == "bookkeeper")
            .unwrap()
            .clone();
        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "decisions": [{ "reason": "missing action" }] }).to_string(),
            ..Default::default()
        });
        let error = StageExecutor::new(
            Arc::clone(&ctx),
            Arc::clone(&stages),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .execute(StageInvocation {
            stage: bookkeeper,
            input_json: "{}".to_string(),
            rendered_question: Some("compose a memory".to_string()),
            resource_root: Some(dir.clone()),
            capability_ceiling: ratatoskr_core::Capability::Read,
            rag_rat_worktree: None,
            shell: None,
            publish: None,
            clarifier: None,
            invocation_guidance: None,
            output: StageOutput::Checkpoint,
        })
        .await
        .unwrap_err();
        assert!(
            error.contains("invalid `MemoryDecisions` output"),
            "{error}"
        );
        let offered_tools = turn.tools.lock().expect("recording runner mutex poisoned")[0].clone();
        assert!(!offered_tools.iter().any(|tool| {
            matches!(
                tool.as_str(),
                "memory_create" | "memory_update" | "memory_mark_obsolete"
            )
        }));
        assert!(
            store
                .checkpoints_for_run("run-bookkeeper-invalid")
                .await
                .unwrap()
                .is_empty(),
            "invalid decisions must not become a bookkeeper checkpoint"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn publisher_declaration_matches_its_typed_schema_and_rust_question() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-publisher-renderer-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            "export async function run(input) { return await publisher(input); }",
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let stages = standard_stages().await.unwrap();
        assert!(
            overlaid_stages(&runtime)
                .await
                .unwrap()
                .iter()
                .any(|stage| stage.id == "publisher"),
            "the terminal adapter must still be able to resolve its declaration"
        );
        let publisher = stages.iter().find(|stage| stage.id == "publisher").unwrap();
        assert_eq!(publisher.agent, "publish");
        assert_eq!(
            publisher.capabilities,
            [ratatoskr_core::Capability::Publish]
        );
        assert_eq!(
            publisher.tools,
            [ratatoskr_agent::publish::GH, ratatoskr_agent::publish::PUSH]
        );
        assert_eq!(publisher.session, None);

        let mut declared = publisher.output_schema.clone().unwrap();
        let mut generated =
            serde_json::to_value(schemars::schema_for!(crate::publisher::PublisherOutput)).unwrap();
        without_schema_annotations(&mut declared);
        without_schema_annotations(&mut generated);
        assert_eq!(declared, generated, "schema drift for publisher");

        let input = crate::publisher::PublisherInput {
            issue: "GitHub issue #210: publish the result".to_string(),
            analyst: AnalystOutput {
                impact_summary: "keeps tracker links distinct".to_string(),
                touched: vec!["crates/ratatoskr-nodes".to_string()],
                risks: Vec::new(),
                requirements: vec!["report the unresolved review".to_string()],
                residual_risk: String::new(),
                changes_code: true,
                acceptance: Vec::new(),
                interface: Vec::new(),
            },
            implementer: Some(ImplementerOutput {
                worktree_path: "/tmp/ratatoskr/worktree".to_string(),
                branch: "ratatoskr/publisher".to_string(),
                diff_summary: " publisher.rs | 2 ++".to_string(),
                touched_files: vec!["publisher.rs".to_string()],
                rewritten_files: Vec::new(),
                commit_kind: "fix".to_string(),
                commit_scope: "publisher".to_string(),
                commit_subject: "keep links distinct".to_string(),
                failing_tests: vec!["publisher::links".to_string()],
                passed_tests: 4,
                exit_code: 101,
                narrative: None,
            }),
            status: "max_iterations_reached".to_string(),
            iterations: 2,
            unresolved: vec![verifier::Finding {
                severity: verifier::Severity::P2,
                kind: verifier::FindingKind::Execution,
                file: "publisher.rs".to_string(),
                line: Some(12),
                summary: "the URLs can still run together".to_string(),
                failure_scenario: "both links are returned in one field".to_string(),
            }],
        };
        let input_json = serde_json::to_string(&input).unwrap();
        let captured = Arc::new(Mutex::new(None));
        let capture = Arc::clone(&captured);
        let host: HostFn = Arc::new(move |arg| {
            let capture = Arc::clone(&capture);
            Box::pin(async move {
                *capture.lock().expect("publisher capture mutex poisoned") =
                    Some(serde_json::from_str::<serde_json::Value>(&arg).unwrap());
                Ok(json!({ "action": "none", "reasoning": "captured" }).to_string())
            })
        });
        runtime
            .run_with_question_renderers(
                "run",
                input_json,
                HashMap::from([("publisher".to_string(), host)]),
                stage_question_renderers(&stages),
            )
            .await
            .unwrap();
        let envelope = captured
            .lock()
            .expect("publisher capture mutex poisoned")
            .take()
            .unwrap();
        let question = envelope["__ratatoskrRenderedQuestion"]["question"]
            .as_str()
            .unwrap();
        assert!(question.contains("THIS RUN DID NOT FINISH CLEAN"));
        assert!(question.contains("the URLs can still run together"));
        assert!(!question.contains("WORKTREE:"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn publisher_tools_require_a_rust_grant_and_invalid_output_is_not_checkpointed() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-publisher-authority-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-publisher-authority", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "publisher".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-publisher-authority",
            "publish this",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = Arc::new(standard_stages().await.unwrap());
        let publisher = stages
            .iter()
            .find(|stage| stage.id == "publisher")
            .unwrap()
            .clone();

        let ungranted_turn = Arc::new(RecordingStageTurn {
            output: json!({ "action": "none", "reasoning": "not granted" }).to_string(),
            ..Default::default()
        });
        StageExecutor::new(
            Arc::clone(&ctx),
            Arc::clone(&stages),
            Arc::clone(&ungranted_turn) as Arc<dyn StageTurn>,
        )
        .execute(StageInvocation {
            stage: publisher.clone(),
            input_json: "{}".to_string(),
            rendered_question: Some("do not publish".to_string()),
            resource_root: Some(dir.clone()),
            capability_ceiling: ratatoskr_core::Capability::Publish,
            rag_rat_worktree: None,
            shell: None,
            publish: None,
            clarifier: None,
            invocation_guidance: None,
            output: StageOutput::Evidence,
        })
        .await
        .unwrap();
        let ungranted_tools = ungranted_turn
            .tools
            .lock()
            .expect("recording runner mutex poisoned")[0]
            .clone();
        assert!(!ungranted_tools.iter().any(|tool| {
            tool == ratatoskr_agent::publish::GH || tool == ratatoskr_agent::publish::PUSH
        }));
        assert!(
            !ungranted_turn
                .has_push
                .lock()
                .expect("recording runner mutex poisoned")[0]
        );

        let invalid_turn = Arc::new(RecordingStageTurn {
            output: json!({ "action": "none" }).to_string(),
            ..Default::default()
        });
        let error = StageExecutor::new(
            Arc::clone(&ctx),
            Arc::clone(&stages),
            Arc::clone(&invalid_turn) as Arc<dyn StageTurn>,
        )
        .execute(StageInvocation {
            stage: publisher,
            input_json: "{}".to_string(),
            rendered_question: Some("publish".to_string()),
            resource_root: Some(dir.clone()),
            capability_ceiling: ratatoskr_core::Capability::Publish,
            rag_rat_worktree: None,
            shell: None,
            publish: Some(StandardStagePublishResources {
                push: Some(ratatoskr_agent::publish::PushAccess {
                    repo_root: dir.clone(),
                    branch: "ratatoskr/publisher-authority".to_string(),
                    issue: Some("GitHub issue #210".to_string()),
                }),
            }),
            clarifier: None,
            invocation_guidance: None,
            output: StageOutput::Checkpoint,
        })
        .await
        .unwrap_err();
        assert!(
            error.contains("invalid `PublisherOutput` output"),
            "{error}"
        );
        let granted_tools = invalid_turn
            .tools
            .lock()
            .expect("recording runner mutex poisoned")[0]
            .clone();
        assert!(granted_tools.contains(&ratatoskr_agent::publish::GH.to_string()));
        assert!(granted_tools.contains(&ratatoskr_agent::publish::PUSH.to_string()));
        assert!(
            invalid_turn
                .has_push
                .lock()
                .expect("recording runner mutex poisoned")[0]
        );
        assert!(
            store
                .checkpoints_for_run("run-publisher-authority")
                .await
                .unwrap()
                .is_empty(),
            "invalid output must not become a publisher result"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn publisher_tool_assembly_grants_publication_and_read_tools_but_never_mutation() {
        // Issue #227, tool-assembly regression: the publisher declares exactly `gh` and
        // `git_push` under a `Publish` ceiling. `Publish` is authority for the external
        // publication actions Rust granted — it is not `Write`, so the ordered capability must
        // not add `Write`, `Edit` or `Bash` to the offer.
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-publisher-tool-assembly-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-publisher-tools", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("publisher".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-publisher-tools",
            "publish this",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = Arc::new(standard_stages().await.unwrap());
        let publisher = stages
            .iter()
            .find(|stage| stage.id == "publisher")
            .unwrap()
            .clone();

        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "action": "none", "reasoning": "nothing to deliver" }).to_string(),
            ..Default::default()
        });
        StageExecutor::new(
            Arc::clone(&ctx),
            Arc::clone(&stages),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .execute(StageInvocation {
            stage: publisher,
            input_json: "{}".to_string(),
            rendered_question: Some("deliver the run".to_string()),
            resource_root: Some(dir.clone()),
            capability_ceiling: ratatoskr_core::Capability::Publish,
            rag_rat_worktree: None,
            shell: None,
            publish: Some(StandardStagePublishResources {
                push: Some(ratatoskr_agent::publish::PushAccess {
                    repo_root: dir.clone(),
                    branch: "ratatoskr/run-publisher-tools".to_string(),
                    issue: Some("GitHub issue #227".to_string()),
                }),
            }),
            clarifier: None,
            invocation_guidance: None,
            output: StageOutput::Evidence,
        })
        .await
        .unwrap();

        let tools = turn.tools.lock().expect("recording runner mutex poisoned")[0].clone();
        // The publication tools the stage declared, both granted by Rust here...
        assert!(tools.contains(&ratatoskr_agent::publish::GH.to_string()));
        assert!(tools.contains(&ratatoskr_agent::publish::PUSH.to_string()));
        assert!(
            turn.has_push
                .lock()
                .expect("recording runner mutex poisoned")[0]
        );
        // ...plus the ordinary read reach a publisher reasons with...
        for read in [
            ratatoskr_agent::files::READ,
            ratatoskr_agent::files::GREP,
            ratatoskr_agent::files::GLOB,
        ] {
            assert!(
                tools.iter().any(|tool| tool == read),
                "publisher lost {read}: {tools:?}"
            );
        }
        // ...and no repository mutation. A Publish ceiling is not Write authority.
        for mutation in [
            ratatoskr_agent::files::WRITE,
            ratatoskr_agent::files::EDIT,
            ratatoskr_agent::shell::BASH,
        ] {
            assert!(
                !tools.iter().any(|tool| tool == mutation),
                "publisher was offered {mutation} from a Publish ceiling: {tools:?}"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn publisher_without_a_worktree_keeps_gh_but_loses_push_and_all_mutation_tools() {
        // A run that changed no code still publishes — a comment is the sensible form — so `gh`
        // is granted against the run context's captured repository root. There is no run branch,
        // so no `git_push`, and still no generic file mutation.
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-publisher-no-worktree-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run(
                "run-publisher-no-worktree",
                None,
                RunStatus::Running.as_str(),
            )
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("publisher".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-publisher-no-worktree",
            "publish this",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = Arc::new(standard_stages().await.unwrap());
        let publisher = stages
            .iter()
            .find(|stage| stage.id == "publisher")
            .unwrap()
            .clone();

        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "action": "none", "reasoning": "nothing to deliver" }).to_string(),
            ..Default::default()
        });
        StageExecutor::new(
            Arc::clone(&ctx),
            Arc::clone(&stages),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .execute(StageInvocation {
            stage: publisher,
            input_json: "{}".to_string(),
            rendered_question: Some("deliver the run".to_string()),
            resource_root: Some(dir.clone()),
            capability_ceiling: ratatoskr_core::Capability::Publish,
            rag_rat_worktree: None,
            shell: None,
            publish: Some(StandardStagePublishResources { push: None }),
            clarifier: None,
            invocation_guidance: None,
            output: StageOutput::Evidence,
        })
        .await
        .unwrap();

        let tools = turn.tools.lock().expect("recording runner mutex poisoned")[0].clone();
        assert!(tools.contains(&ratatoskr_agent::publish::GH.to_string()));
        assert!(
            !tools.contains(&ratatoskr_agent::publish::PUSH.to_string()),
            "no run branch means no push tool: {tools:?}"
        );
        assert!(
            !turn
                .has_push
                .lock()
                .expect("recording runner mutex poisoned")[0]
        );
        for mutation in [
            ratatoskr_agent::files::WRITE,
            ratatoskr_agent::files::EDIT,
            ratatoskr_agent::shell::BASH,
        ] {
            assert!(
                !tools.iter().any(|tool| tool == mutation),
                "a comment-only publisher was offered {mutation}: {tools:?}"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn publisher_without_a_rust_grant_gets_no_publication_and_no_mutation_tools() {
        // No publish resources at all: a `Publish` ceiling and the publish profile alone must
        // assemble nothing beyond reads — no `gh`, no `git_push`, and no `Write`/`Edit`/`Bash`.
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-publisher-ungranted-tools-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run(
                "run-publisher-ungranted-tools",
                None,
                RunStatus::Running.as_str(),
            )
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("publisher".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-publisher-ungranted-tools",
            "publish this",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = Arc::new(standard_stages().await.unwrap());
        let publisher = stages
            .iter()
            .find(|stage| stage.id == "publisher")
            .unwrap()
            .clone();

        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "action": "none", "reasoning": "nothing to deliver" }).to_string(),
            ..Default::default()
        });
        StageExecutor::new(
            Arc::clone(&ctx),
            Arc::clone(&stages),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .execute(StageInvocation {
            stage: publisher,
            input_json: "{}".to_string(),
            rendered_question: Some("deliver the run".to_string()),
            resource_root: Some(dir.clone()),
            capability_ceiling: ratatoskr_core::Capability::Publish,
            rag_rat_worktree: None,
            shell: None,
            publish: None,
            clarifier: None,
            invocation_guidance: None,
            output: StageOutput::Evidence,
        })
        .await
        .unwrap();

        let tools = turn.tools.lock().expect("recording runner mutex poisoned")[0].clone();
        for publication in [
            ratatoskr_agent::publish::GH,
            ratatoskr_agent::publish::PUSH,
            ratatoskr_agent::files::WRITE,
            ratatoskr_agent::files::EDIT,
            ratatoskr_agent::shell::BASH,
        ] {
            assert!(
                !tools.iter().any(|tool| tool == publication),
                "an ungranted publisher was offered {publication}: {tools:?}"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The publisher turn from issue #227: handed any mutation authority, it uses it against
    /// whatever file root it was given — which the live run proved was the operator's checkout.
    #[derive(Default)]
    struct OperatorCheckoutDirtyingTurn {
        /// Every tool name offered, per invocation.
        tools: Mutex<Vec<Vec<String>>>,
        /// The mutation-capable tools offered, per invocation.
        mutation_tools: Mutex<Vec<Vec<String>>>,
        /// The file root handed to the turn, per invocation.
        roots: Mutex<Vec<Option<PathBuf>>>,
    }

    impl StageTurn for OperatorCheckoutDirtyingTurn {
        fn run<'a>(
            &'a self,
            run: ratatoskr_agent::NodeRun<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<String, ratatoskr_agent::AgentError>> + Send + 'a>>
        {
            let tools = run.tools.names();
            let mutation: Vec<String> = [
                ratatoskr_agent::files::WRITE,
                ratatoskr_agent::files::EDIT,
                ratatoskr_agent::shell::BASH,
            ]
            .into_iter()
            .filter(|name| tools.iter().any(|tool| tool == name))
            .map(str::to_string)
            .collect();
            // Play the call the live publisher made: a write under the root it was handed. The
            // offered Write/Edit are rooted writes, so a direct rooted write here is exactly
            // their effect — the authority being probed is whether they are offered at all.
            let can_write = mutation.iter().any(|name| {
                name == ratatoskr_agent::files::WRITE || name == ratatoskr_agent::files::EDIT
            });
            if can_write && let Some(root) = run.files.as_ref() {
                let target = root.join("crates/ratatoskr-core/src/config.rs");
                std::fs::create_dir_all(target.parent().unwrap()).unwrap();
                std::fs::write(&target, "dirtied by the publisher\n").unwrap();
            }
            self.tools
                .lock()
                .expect("dirtying turn mutex poisoned")
                .push(tools);
            self.mutation_tools
                .lock()
                .expect("dirtying turn mutex poisoned")
                .push(mutation);
            self.roots
                .lock()
                .expect("dirtying turn mutex poisoned")
                .push(run.files.clone());
            Box::pin(async move {
                Ok(json!({ "action": "none", "reasoning": "nothing to deliver" }).to_string())
            })
        }
    }

    #[tokio::test]
    async fn a_publisher_with_a_writable_looking_request_leaves_the_operator_checkout_untouched() {
        // Issue #227: a publisher asks for `gh`/`git_push` under a Publish ceiling and behaves as
        // the reported incident did if mutation is available. Its Rust-granted root is a committed
        // run worktree, while the separate operator checkout must remain byte-for-byte clean.
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-publisher-containment-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let operator = init_test_repo(&dir).await;
        let run_worktree = dir.join("committed-run-worktree");
        let created = tokio::process::Command::new("git")
            .arg("-C")
            .arg(&operator)
            .args([
                "worktree",
                "add",
                "-q",
                "-b",
                "ratatoskr/run-publisher-containment",
            ])
            .arg(&run_worktree)
            .output()
            .await
            .unwrap();
        assert!(created.status.success(), "git worktree add: {created:?}");
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let run_id = "run-publisher-containment";
        store
            .upsert_run(run_id, None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("publisher".to_string(), model_route());
        config.publish.enabled = true;
        let mut ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "publish the run",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        Arc::get_mut(&mut ctx)
            .expect("publisher context has no other owners")
            .repo_path = operator.clone();

        // The model-facing input claims the operator checkout; it is data for the write-up, never
        // the root. Rust grants the separate committed run worktree below.
        let implementer = ImplementerOutput {
            worktree_path: operator.display().to_string(),
            branch: format!("ratatoskr/{run_id}"),
            ..imp(&[], &["post"], 0)
        };
        let input = crate::publisher::PublisherInput {
            issue: "GitHub issue #227: keep the publisher out of the live checkout".to_string(),
            analyst: AnalystOutput {
                impact_summary: "publisher containment".to_string(),
                touched: vec!["crates/ratatoskr-nodes".to_string()],
                risks: Vec::new(),
                requirements: vec!["the operator checkout stays clean".to_string()],
                residual_risk: String::new(),
                changes_code: true,
                acceptance: Vec::new(),
                interface: Vec::new(),
            },
            implementer: Some(implementer),
            status: "converged".to_string(),
            iterations: 1,
            unresolved: Vec::new(),
        };
        let turn = Arc::new(OperatorCheckoutDirtyingTurn::default());
        let actions =
            LiveTerminalActions::with_publisher_turn(Arc::clone(&turn) as Arc<dyn StageTurn>);
        let published = actions
            .publish(&ctx, input, true, Some(&WorktreePath(run_worktree.clone())))
            .await;
        assert!(published.is_some(), "terminal publication did not run");

        let offered = turn.tools.lock().expect("dirtying turn mutex poisoned")[0].clone();
        assert!(
            offered.contains(&ratatoskr_agent::publish::GH.to_string()),
            "the publisher can still do its job: {offered:?}"
        );
        let mutation = turn
            .mutation_tools
            .lock()
            .expect("dirtying turn mutex poisoned")[0]
            .clone();
        assert!(
            mutation.is_empty(),
            "the publisher was offered mutation tools: {mutation:?}"
        );
        // The root the turn ran with is the one Rust supplied — the model-claimed path in the
        // input never reaches the tool layer.
        let root = turn.roots.lock().expect("dirtying turn mutex poisoned")[0].clone();
        assert_eq!(root.as_deref(), Some(run_worktree.as_path()));

        let status = tokio::process::Command::new("git")
            .arg("-C")
            .arg(&operator)
            .args(["status", "--porcelain"])
            .output()
            .await
            .unwrap();
        assert!(status.status.success());
        let stdout = String::from_utf8_lossy(&status.stdout);
        assert!(
            stdout.trim().is_empty(),
            "the operator checkout was dirtied: {stdout}"
        );
        assert_eq!(
            std::fs::read_to_string(operator.join("tracked.txt")).unwrap(),
            "baseline\n"
        );
        assert!(
            !operator
                .join("crates/ratatoskr-core/src/config.rs")
                .exists()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn terminal_publication_uses_the_rust_held_worktree_not_the_model_claimed_path() {
        // Issue #227, root selection: the implementer's output is model-produced, so its
        // `worktree_path` is a claim, not authority. The terminal phase — the commit and the
        // publication that follows it — must resolve its file/Git root from the worktree Rust
        // created and held on the run context, even when the model-reported path disagrees.
        //
        // Contract reading: `publisher(PublisherInput)` carries no root; Rust supplies the
        // trusted `&WorktreePath`, and `finish_full` is where the terminal phase's root is
        // chosen. Asserted on the commit call and the outcome, the two places that root leaves
        // this function.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-publisher-root-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rust_worktree = dir.join("run-worktree");
        std::fs::create_dir_all(&rust_worktree).unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let run_id = "run-publisher-root";
        terminal_plan(&store, run_id, true).await;
        let config = RatatoskrConfig::default();
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "publish from the run worktree",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        *ctx.worktree.lock().unwrap() = Some(WorktreePath(rust_worktree.clone()));
        checkpoint(&store, run_id, "redteam", &red(&[], &["baseline"], 0))
            .await
            .unwrap();
        // The model claims a different directory — standing in for the operator checkout — was
        // its worktree.
        let claimed = dir.join("operator-checkout");
        std::fs::create_dir_all(&claimed).unwrap();
        let implementer = ImplementerOutput {
            worktree_path: claimed.display().to_string(),
            branch: format!("ratatoskr/{run_id}"),
            ..imp(&[], &["post"], 0)
        };
        checkpoint(&store, run_id, "implementer", &implementer)
            .await
            .unwrap();

        let actions = RecordingTerminalActions::new(true, false);
        let outcome = finish_full(&ctx, &actions).await.unwrap();
        assert_eq!(
            outcome.worktree.as_ref().map(WorktreePath::as_path),
            Some(rust_worktree.as_path()),
            "the terminal phase took the model-claimed path as its root"
        );
        assert_eq!(
            actions.publisher_worktrees(),
            [Some(rust_worktree.clone())],
            "publisher ran outside the Rust-held worktree"
        );
        assert!(
            matches!(
                actions.calls().first(),
                Some(TerminalCall::Commit { worktree, .. }) if worktree == &rust_worktree
            ),
            "commit ran against the model-claimed path: {:?}",
            actions.calls()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    fn without_schema_defaults(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(object) => {
                object.remove("default");
                for value in object.values_mut() {
                    without_schema_defaults(value);
                }
            }
            serde_json::Value::Array(array) => {
                for value in array {
                    without_schema_defaults(value);
                }
            }
            _ => {}
        }
    }

    #[tokio::test]
    async fn standard_analyst_contract_matches_the_typed_gate_and_rejects_missing_impact() {
        let stages = standard_stages().await.unwrap();
        let analyst_stage = stages.iter().find(|stage| stage.id == "analyst").unwrap();
        assert_eq!(
            analyst_stage.instructions,
            include_str!("../prompts/analyst.md").trim()
        );
        assert!(
            analyst_stage
                .instructions
                .contains("Treat the issue's proposed implementation as evidence")
        );
        assert!(analyst_stage.instructions.contains("run an extension test"));
        assert!(
            analyst_stage
                .instructions
                .contains("why the generic alternative loses")
        );
        let mut generated =
            serde_json::to_value(schemars::schema_for!(analyst::AnalystOutput)).unwrap();
        without_schema_annotations(&mut generated);
        assert_eq!(analyst_stage.output_schema.as_ref().unwrap(), &generated);

        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-analyst-schema-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-analyst-schema", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let config = RatatoskrConfig::default();
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-analyst-schema",
            "validate this",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let turn = Arc::new(RecordingStageTurn::default());
        let mut hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        let envelope = json!({
            "__ratatoskrRenderedQuestion": {
                "input": {
                    "issue": "validate this",
                    "scout": { "related_items": [], "papertrail_summary": "none" },
                    "memory": { "memories": [] }
                },
                "question": "ISSUE:\nvalidate this"
            }
        })
        .to_string();
        let error = hosts.remove("analyst").unwrap()(envelope)
            .await
            .unwrap_err();
        assert!(error.contains("invalid `AnalystOutput` output"), "{error}");
        assert!(error.contains("impact_summary"), "{error}");
        assert!(
            store
                .checkpoints_for_run("run-analyst-schema")
                .await
                .unwrap()
                .is_empty(),
            "invalid output must not be checkpointed"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_context_distillation_uses_generic_dispatch_and_preserves_its_input() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-context-distillation-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"export async function plan(input) { return await context_distillation(input); }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run(
                "run-standard-context-distillation",
                None,
                RunStatus::Running.as_str(),
            )
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "context".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Compacted,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-standard-context-distillation",
            "explain the checkpoint contract",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let stage = stages
            .iter()
            .find(|stage| stage.id == "context_distillation")
            .unwrap();
        assert_eq!(stage.governance_id(), "context");
        assert_eq!(
            stage.node_id(),
            "context",
            "the distillation is the context node's turn"
        );
        assert_eq!(stage.agent, "explore");
        assert_eq!(stage.session, None);
        assert_eq!(
            stage.tools,
            [
                "papertrail_issue_search",
                "semantic_search",
                "symbol_lookup",
                "memory_search"
            ]
        );

        let input = crate::context::ContextDistillationInput {
            issue: "explain the checkpoint contract".to_string(),
            memory: crate::MemoryOutput {
                memories: vec![crate::memory::MemoryRecord {
                    memory_id: "mem_exact".to_string(),
                    kind: "Invariant".to_string(),
                    title: "Checkpoints keep original input".to_string(),
                    confidence: "high".to_string(),
                    status: "active".to_string(),
                    body: "The full source evidence.".to_string(),
                    summary: Some("The compact source evidence.".to_string()),
                }],
            },
            searchable: true,
        };
        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "brief": "The checkpoint retains its source evidence.",
                "constraints": [{
                    "says": "keep the original input",
                    "from_memory_id": "mem_exact"
                }],
                "prior_art": [
                    { "item_key": "", "title": "" },
                    { "item_key": "#118", "title": "Context evidence" }
                ],
                "papertrail_summary": "Issue 118 introduced the split."
            })
            .to_string(),
            ..Default::default()
        });
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        runtime
            .run_with_question_renderers(
                "plan",
                serde_json::to_string(&input).unwrap(),
                hosts,
                stage_question_renderers(&stages),
            )
            .await
            .unwrap();

        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["context_distillation"],
            "the turn is the distillation's, whatever box it is drawn in"
        );
        assert_eq!(
            *turn
                .sessions
                .lock()
                .expect("recording runner mutex poisoned"),
            [ratatoskr_core::SessionScope::Compacted]
        );
        let offered = turn.tools.lock().expect("recording runner mutex poisoned")[0].clone();
        for tool in ["Read", "Grep", "Glob"] {
            assert!(
                offered.iter().any(|offered| offered == tool),
                "missing {tool}"
            );
        }
        assert!(
            turn.questions
                .lock()
                .expect("recording runner mutex poisoned")[0]
                .contains("mem_exact")
        );

        let checkpoints = store
            .checkpoints_for_run("run-standard-context-distillation")
            .await
            .unwrap();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].node_name, "context_distillation");
        let checkpoint_input: crate::context::ContextDistillationInput =
            serde_json::from_str(checkpoints[0].input_json.as_deref().unwrap()).unwrap();
        assert_eq!(
            checkpoint_input.memory.memories[0].body,
            "The full source evidence."
        );
        let checkpoint_output: serde_json::Value =
            serde_json::from_str(&checkpoints[0].output_json).unwrap();
        assert_eq!(checkpoint_output["prior_art"].as_array().unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_context_contract_rejects_missing_brief_without_a_checkpoint() {
        let stages = standard_stages().await.unwrap();
        let stage = stages
            .iter()
            .find(|stage| stage.id == "context_distillation")
            .unwrap();
        assert_eq!(
            stage.instructions,
            include_str!("../prompts/context.md").trim()
        );
        let mut generated =
            serde_json::to_value(schemars::schema_for!(crate::context::Distillation)).unwrap();
        without_schema_annotations(&mut generated);
        let mut declared = stage.output_schema.clone().unwrap();
        without_schema_annotations(&mut declared);
        assert_eq!(declared, generated);
        assert!(
            stage.output_schema.as_ref().unwrap()["$defs"]["RelatedItem"]["properties"]
                .get("title")
                .is_some()
        );

        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-context-schema-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-context-schema", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "context".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-context-schema",
            "context",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "constraints": [], "prior_art": [] }).to_string(),
            ..Default::default()
        });
        let mut hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        let envelope = json!({
            "__ratatoskrRenderedQuestion": {
                "input": {
                    "issue": "context",
                    "memory": { "memories": [] },
                    "searchable": false
                },
                "question": "TASK:\ncontext\n\nRECORDED MEMORIES: none"
            }
        })
        .to_string();
        let error = hosts.remove("context_distillation").unwrap()(envelope)
            .await
            .unwrap_err();
        assert!(error.contains("invalid `Distillation` output"), "{error}");
        assert!(error.contains("brief"), "{error}");
        assert!(
            store
                .checkpoints_for_run("run-context-schema")
                .await
                .unwrap()
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn context_operation_keeps_the_no_rag_rat_baseline_rust_owned() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-context-no-rag-rat-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"export async function plan(input) { return await context(input.issue); }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-context-no-rag", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "context".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-context-no-rag",
            "explain context",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "brief": "No indexed memories are available.",
                "constraints": [],
                "prior_art": [],
                "papertrail_summary": ""
            })
            .to_string(),
            ..Default::default()
        });
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        runtime
            .run_with_question_renderers(
                "plan",
                json!({ "issue": "explain context" }).to_string(),
                hosts,
                stage_question_renderers(&stages),
            )
            .await
            .unwrap();

        let question = turn
            .questions
            .lock()
            .expect("recording runner mutex poisoned")[0]
            .clone();
        assert!(
            question.contains("this repository keeps none"),
            "{question}"
        );
        assert!(!question.contains("Search again"), "{question}");
        let checkpoints = store
            .checkpoints_for_run("run-context-no-rag")
            .await
            .unwrap();
        // The distillation's own row, then the box's aggregate.
        assert_eq!(
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.node_name.as_str())
                .collect::<Vec<_>>(),
            ["context_distillation", "context"]
        );
        let checkpoints = &checkpoints[1..];
        assert_eq!(
            checkpoints[0].input_json.as_deref(),
            Some(r#""explain context""#)
        );
        let output: crate::ContextOutput =
            serde_json::from_str(&checkpoints[0].output_json).unwrap();
        assert!(output.memory.memories.is_empty());
        assert!(output.scout.related_items.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn scripted_redteam_matches_built_in_evidence_and_worktree_lifecycle() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-scripted-redteam-parity-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let repo = init_test_repo(&dir).await;
        let rules = dir.join("rules");
        let engine = ScriptEngine::load(&rules).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let run_id = "redteam-parity";
        store
            .upsert_run(run_id, None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let analyst = AnalystOutput {
            impact_summary: "Keep scripted red-team evidence equivalent.".to_string(),
            touched: Vec::new(),
            risks: Vec::new(),
            requirements: Vec::new(),
            residual_risk: String::new(),
            changes_code: true,
            acceptance: Vec::new(),
            // An empty interface proves the host lifecycle without spending a model turn. The
            // declared author invocation and its rooted Write ceiling are covered below by the
            // generic-stage parity test.
            interface: Vec::new(),
        };
        checkpoint(&store, run_id, "analyst", &analyst)
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.worktree.root = dir.join("worktrees");
        config
            .models
            .insert("implementer".to_string(), model_route());
        config.models.insert("redteam".to_string(), model_route());
        let mut ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "exercise red-team parity",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        Arc::get_mut(&mut ctx).unwrap().repo_path = repo.clone();
        *ctx.acceptance.lock().unwrap() = Some(Vec::new());

        // A `redteam` route enables both halves. What each may reach is not decided here: both
        // drive their turn through the stage executor, and the write ceiling that separates them
        // is asserted against the turn it actually composes by the generic-stage parity test.
        let configured = build_red_team(&ctx, Vec::new()).await.unwrap();
        assert!(configured.author.is_some());
        assert!(configured.classifier.is_some());

        let returned: RedTeamOutput = serde_json::from_str(
            &red_team_host(Arc::clone(&ctx), "null".to_string())
                .await
                .unwrap(),
        )
        .unwrap();
        let worktree = ctx
            .worktree
            .lock()
            .unwrap()
            .clone()
            .expect("redTeam prepares and retains the implementer worktree");
        assert!(worktree.as_path().exists());
        let scripted = latest_checkpoint::<RedTeamOutput>(&store, run_id, "redteam")
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(&returned).unwrap(),
            serde_json::to_value(&scripted).unwrap()
        );

        // Both flows use `run_and_author`: the scripted checkpoint must carry the same
        // deterministic evidence as a built-in invocation on the retained pre-change tree.
        let built_in = build_red_team(&ctx, Vec::new())
            .await
            .unwrap()
            .run_and_author(worktree.as_path(), &ctx.issue, &analyst.interface)
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(&scripted).unwrap(),
            serde_json::to_value(&built_in).unwrap()
        );
        assert!(scripted.authored.is_none());
        assert!(
            !ratatoskr_exec::managed_worktree_branches(&repo)
                .await
                .unwrap()
                .contains(&format!(
                    "ratatoskr/{}-baseline",
                    run_id.chars().take(8).collect::<String>()
                )),
            "the fresh baseline worktree and branch are removed after measurement"
        );

        remove_worktree(&repo, &worktree).await.unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn scripted_redteam_and_implementer_cannot_edit_one_tree_concurrently() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-redteam-order-gate-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let ctx = WorkflowContext::new(
            None,
            &RatatoskrConfig::default(),
            &store,
            "redteam-order-gate",
            "keep model writes ordered",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        ctx.red_team_started.store(true, Ordering::SeqCst);
        let implement_error = implement_host(Arc::clone(&ctx), "{}".to_string())
            .await
            .unwrap_err();
        assert!(implement_error.contains("redTeam() call has finished"));

        let other = WorkflowContext::new(
            None,
            &RatatoskrConfig::default(),
            &store,
            "redteam-order-gate-other",
            "keep model writes ordered",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        other.implement_started.store(true, Ordering::SeqCst);
        let redteam_error = red_team_host(other, "null".to_string()).await.unwrap_err();
        assert!(redteam_error.contains("test authoring cannot race implementation"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_redteam_stages_use_generic_dispatch_with_faithful_prompts_and_ceilings() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-redteam-stages-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"export async function plan(input) {
                return await redteam_classifier(input.classifier);
            }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-standard-redteam", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "redteam".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Compacted,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-standard-redteam",
            "add Store::prune",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let author_stage = stages
            .iter()
            .find(|stage| stage.id == "redteam_author")
            .unwrap();
        let classifier_stage = stages
            .iter()
            .find(|stage| stage.id == "redteam_classifier")
            .unwrap();
        assert_eq!(author_stage.governance_id(), "redteam");
        assert_eq!(classifier_stage.governance_id(), "redteam");
        assert_eq!(author_stage.node_id(), "redteam");
        assert_eq!(
            classifier_stage.node_id(),
            "redteam",
            "both halves are the red team's work, and each keeps its own identity doing it"
        );
        assert_eq!(
            author_stage.instructions,
            include_str!("../prompts/redteam-author.md").trim()
        );
        assert_eq!(
            classifier_stage.instructions,
            include_str!("../prompts/redteam-classifier.md").trim()
        );

        let author = crate::redteam::TestAuthorInput {
            issue: "add Store::prune".to_string(),
            interface: vec![crate::analyst::InterfaceItem {
                name: "Store::prune".to_string(),
                shape: "async fn prune(Duration) -> Result<u64>".to_string(),
                happy: vec!["returns the deleted row count".to_string()],
                sad: vec!["a zero duration deletes nothing".to_string()],
            }],
        };
        let classifier = crate::redteam::ClassifierInput {
            failing: vec!["store::tests::prune_zero".to_string()],
            raw_output: "assertion failed: deleted > 0".to_string(),
        };
        let turn = Arc::new(RecordingStageTurn::default());
        let author_root = dir.join("author-root");
        std::fs::create_dir_all(&author_root).unwrap();
        evaluate_standard_stage_with_resources_and_turn(
            Arc::clone(&ctx),
            "redteam_author",
            serde_json::to_string(&author).unwrap(),
            StandardStageResources {
                resource_root: author_root.clone(),
                capability_ceiling: ratatoskr_core::Capability::Write,
                rag_rat_worktree: Some(author_root.clone()),
                shell: None,
                publish: None,
                clarifier: None,
                guidance: None,
            },
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .await
        .unwrap();
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        assert!(!hosts.contains_key("redteam_author"));
        assert!(hosts.contains_key("redteam_classifier"));
        runtime
            .run_with_question_renderers(
                "plan",
                json!({ "classifier": classifier }).to_string(),
                hosts,
                stage_question_renderers(&stages),
            )
            .await
            .unwrap();

        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["redteam_author", "redteam_classifier"]
        );
        assert_eq!(
            *turn
                .sessions
                .lock()
                .expect("recording runner mutex poisoned"),
            [
                ratatoskr_core::SessionScope::Fresh,
                ratatoskr_core::SessionScope::Fresh
            ]
        );
        let tools = turn
            .tools
            .lock()
            .expect("recording runner mutex poisoned")
            .clone();
        for tool in ["Read", "Grep", "Glob", "Write", "Edit"] {
            assert!(
                tools[0].iter().any(|offered| offered == tool),
                "missing {tool}"
            );
        }
        assert_eq!(
            turn.files.lock().expect("recording runner mutex poisoned")[0],
            Some(author_root.clone())
        );
        assert_eq!(
            *turn
                .rag_rat_worktrees
                .lock()
                .expect("recording runner mutex poisoned"),
            [Some(author_root), None],
            "the author uses its host-selected worktree; the base classifier uses the index"
        );
        assert!(
            !tools[1]
                .iter()
                .any(|tool| tool == "Write" || tool == "Edit")
        );
        for tool in ["Read", "Grep", "Glob"] {
            assert!(
                tools[1].iter().any(|offered| offered == tool),
                "missing {tool}"
            );
        }
        let questions = turn
            .questions
            .lock()
            .expect("recording runner mutex poisoned")
            .clone();
        assert!(questions[0].contains("Store::prune"));
        assert!(questions[0].contains("the code does not exist"));
        assert!(questions[1].contains("store::tests::prune_zero"));
        assert!(questions[1].contains("assertion failed: deleted > 0"));
        let checkpoints = store
            .checkpoints_for_run("run-standard-redteam")
            .await
            .unwrap();
        // Both halves ran and both recorded their own turn; no aggregate, because the operation
        // host that writes `redteam` is not what this case drives.
        assert_eq!(
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.node_name.as_str())
                .collect::<Vec<_>>(),
            ["redteam_author", "redteam_classifier"]
        );
        let checkpoint_classifier: crate::redteam::ClassifierInput =
            serde_json::from_str(checkpoints[1].input_json.as_deref().unwrap()).unwrap();
        assert_eq!(
            checkpoint_classifier.raw_output,
            "assertion failed: deleted > 0"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_run_with_both_red_team_halves_reports_what_both_of_them_cost() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-redteam-cost-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let author_root = dir.join("implementer-tree");
        std::fs::create_dir_all(&author_root).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-redteam-cost", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "redteam".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-redteam-cost",
            "add Store::prune",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();

        // A run that enables the red team enables both halves, and both spend a turn before the
        // one `redteam` record is written. They resolve their route through their own agent
        // profile — `build` for the author, `reason` for the classifier — so the models they run
        // on can genuinely differ.
        let author = ChargingStageTurn {
            barrier: None,
            output: json!({ "files": [], "tests": [], "covers": "no interface" }).to_string(),
            telemetry: ratatoskr_core::NodeTelemetry {
                model: Some("anthropic/author-model".to_string()),
                duration_ms: Some(200),
                usage: ratatoskr_core::TokenUsage {
                    input_tokens: 20,
                    output_tokens: 200,
                    ..Default::default()
                },
                turns: Some(2),
                tools: vec!["Write".to_string()],
                ..Default::default()
            },
        };
        evaluate_standard_stage_with_resources_and_turn(
            Arc::clone(&ctx),
            "redteam_author",
            serde_json::to_string(&crate::redteam::TestAuthorInput {
                issue: "add Store::prune".to_string(),
                interface: Vec::new(),
            })
            .unwrap(),
            StandardStageResources {
                resource_root: author_root,
                capability_ceiling: ratatoskr_core::Capability::Write,
                rag_rat_worktree: None,
                shell: None,
                publish: None,
                clarifier: None,
                guidance: None,
            },
            Arc::new(author),
        )
        .await
        .unwrap();
        let classifier = ChargingStageTurn {
            barrier: None,
            output: json!({ "classifications": [] }).to_string(),
            telemetry: ratatoskr_core::NodeTelemetry {
                model: Some("anthropic/classifier-model".to_string()),
                duration_ms: Some(100),
                usage: ratatoskr_core::TokenUsage {
                    input_tokens: 10,
                    output_tokens: 100,
                    ..Default::default()
                },
                turns: Some(1),
                tools: vec!["semantic_search".to_string()],
                ..Default::default()
            },
        };
        evaluate_standard_stage_with_turn(
            Arc::clone(&ctx),
            "redteam_classifier",
            serde_json::to_string(&crate::redteam::ClassifierInput {
                failing: vec!["store::tests::prune_zero".to_string()],
                raw_output: "assertion failed: deleted > 0".to_string(),
            })
            .unwrap(),
            Arc::new(classifier),
        )
        .await
        .unwrap();

        note(
            &ctx,
            "redteam",
            &red(&["store::tests::prune_zero"], &["store::tests::prune"], 1),
            None,
        )
        .await
        .unwrap();

        let checkpoints = store.checkpoints_for_run("run-redteam-cost").await.unwrap();
        let row = |node: &str| {
            checkpoints
                .iter()
                .find(|checkpoint| checkpoint.node_name == node)
                .unwrap_or_else(|| panic!("no `{node}` checkpoint"))
                .telemetry
                .clone()
        };

        // One row per turn, each describing the turn it covers. This is what a folded row could
        // not do: the halves run on different profiles with different tool sets, so a row naming
        // both routes is true of the box and useless about either.
        let author = row("redteam_author");
        assert_eq!(author.model.as_deref(), Some("anthropic/author-model"));
        assert_eq!(author.tools, ["Write"]);
        assert_eq!(author.usage.input_tokens, 20);
        assert_eq!(author.turns, Some(2));

        let classifier = row("redteam_classifier");
        assert_eq!(
            classifier.model.as_deref(),
            Some("anthropic/classifier-model")
        );
        assert_eq!(classifier.tools, ["semantic_search"]);
        assert_eq!(classifier.usage.input_tokens, 10);
        assert_eq!(classifier.turns, Some(1));

        // The box's own record is the parent of those two. It ran no turn itself, so it reports
        // none; what the red team cost is the sum of its members, which the shape API totals.
        assert_eq!(row("redteam").model, None);
        assert_eq!(
            author.usage.input_tokens + classifier.usage.input_tokens,
            30
        );
        assert_eq!(
            author.usage.output_tokens + classifier.usage.output_tokens,
            300
        );
        assert!(
            ctx.ledger.unclaimed().is_empty(),
            "nothing the red team spent is left for the run to discard"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn redteam_evidence_keeps_the_author_root_and_rejects_invalid_classification() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-redteam-evidence-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let author_root = dir.join("implementer-tree");
        std::fs::create_dir_all(&author_root).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-redteam-evidence", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "redteam".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-redteam-evidence",
            "add Store::prune",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let author = crate::redteam::TestAuthorInput {
            issue: "add Store::prune".to_string(),
            interface: Vec::new(),
        };
        let author_turn = Arc::new(RecordingStageTurn {
            output: json!({ "files": [], "tests": [], "covers": "no interface" }).to_string(),
            ..Default::default()
        });
        evaluate_standard_stage_with_resources_and_turn(
            Arc::clone(&ctx),
            "redteam_author",
            serde_json::to_string(&author).unwrap(),
            StandardStageResources {
                resource_root: author_root.clone(),
                capability_ceiling: ratatoskr_core::Capability::Write,
                rag_rat_worktree: None,
                shell: None,
                publish: None,
                clarifier: None,
                guidance: None,
            },
            Arc::clone(&author_turn) as Arc<dyn StageTurn>,
        )
        .await
        .unwrap();
        assert_eq!(
            author_turn
                .files
                .lock()
                .expect("recording runner mutex poisoned")[0],
            Some(author_root)
        );

        let classifier = crate::redteam::ClassifierInput {
            failing: vec!["store::tests::prune_zero".to_string()],
            raw_output: "failed".to_string(),
        };
        let invalid_turn = Arc::new(RecordingStageTurn {
            output: json!({ "classifications": [{ "category": "real" }] }).to_string(),
            ..Default::default()
        });
        let error = evaluate_standard_stage_with_turn(
            ctx,
            "redteam_classifier",
            serde_json::to_string(&classifier).unwrap(),
            invalid_turn,
        )
        .await
        .unwrap_err();
        assert!(error.contains("invalid `Classification` output"), "{error}");
        assert!(error.contains("test"), "{error}");
        // The author's turn happened and is recorded under its own name; the classifier's failed
        // its output gate before reaching a record. Neither writes the `redteam` aggregate — that
        // is the operation host's, and it never ran here.
        assert_eq!(
            store
                .checkpoints_for_run("run-redteam-evidence")
                .await
                .unwrap()
                .iter()
                .map(|checkpoint| checkpoint.node_name.as_str())
                .collect::<Vec<_>>(),
            ["redteam_author"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_redteam_contracts_match_the_typed_output_gates() {
        let stages = standard_stages().await.unwrap();
        for (stage_id, mut generated) in [
            (
                "redteam_author",
                serde_json::to_value(schemars::schema_for!(crate::redteam::AuthoredTests)).unwrap(),
            ),
            (
                "redteam_classifier",
                serde_json::to_value(schemars::schema_for!(crate::redteam::Classification))
                    .unwrap(),
            ),
        ] {
            let stage = stages.iter().find(|stage| stage.id == stage_id).unwrap();
            let mut declared = stage.output_schema.clone().unwrap();
            without_schema_annotations(&mut generated);
            without_schema_annotations(&mut declared);
            assert_eq!(declared, generated, "schema drift for {stage_id}");
        }
    }

    fn implementer_attempt_input(
        diagnostic: Option<&str>,
    ) -> crate::implementer::ImplementerAttemptInput {
        crate::implementer::ImplementerAttemptInput {
            issue: "add Store::claim".to_string(),
            analyst: AnalystOutput {
                impact_summary: "The store owns run claims.".to_string(),
                touched: vec!["crates/ratatoskr-store".to_string()],
                risks: vec!["Do not permit two owners".to_string()],
                requirements: vec!["Keep claim acquisition atomic".to_string()],
                residual_risk: String::new(),
                changes_code: true,
                acceptance: Vec::new(),
                interface: vec![crate::analyst::InterfaceItem {
                    name: "Store::claim".to_string(),
                    shape: "fn claim(&self, run: &str) -> Result<Claim, StoreError>".to_string(),
                    happy: vec!["an unclaimed run yields a Claim".to_string()],
                    sad: vec!["an owned run returns an error".to_string()],
                }],
            },
            acceptance: vec![ratatoskr_core::AcceptanceStep {
                name: "store tests".to_string(),
                command: vec![
                    "cargo".to_string(),
                    "test".to_string(),
                    "-p".to_string(),
                    "ratatoskr-store".to_string(),
                ],
            }],
            diagnostic: diagnostic.map(str::to_string),
        }
    }

    #[tokio::test]
    async fn standard_implementer_attempt_is_generic_and_reuses_compacted_session() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-implementer-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run(
                "run-standard-implementer",
                None,
                RunStatus::Running.as_str(),
            )
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.get_mut("implementer").unwrap().session = ratatoskr_core::SessionScope::Fresh;
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-standard-implementer",
            "add Store::claim",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let stage = stages
            .iter()
            .find(|stage| stage.id == "implementer_attempt")
            .unwrap();
        assert_eq!(stage.governance_id(), "implementer");
        assert_eq!(
            stage.node_id(),
            "implementer",
            "an attempt is the implementer's work"
        );
        assert_eq!(stage.session, Some(ratatoskr_core::SessionScope::Compacted));
        assert_eq!(stage.capabilities, [ratatoskr_core::Capability::Write]);
        assert_eq!(
            stage.instructions,
            crate::implementer::NATIVE_PREAMBLE.trim()
        );

        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "summary": "made claim acquisition atomic",
                "kind": "fix",
                "scope": "store",
                "subject": "make claim acquisition atomic"
            })
            .to_string(),
            ..Default::default()
        });
        let initial = implementer_attempt_input(None);
        let iteration = implementer_attempt_input(Some("Fix the failing concurrent claim test."));
        for input in [&initial, &iteration] {
            evaluate_standard_stage_with_resources_and_turn(
                Arc::clone(&ctx),
                "implementer_attempt",
                serde_json::to_string(input).unwrap(),
                StandardStageResources {
                    resource_root: dir.clone(),
                    capability_ceiling: ratatoskr_core::Capability::Write,
                    rag_rat_worktree: None,
                    shell: None,
                    publish: None,
                    // As the implementer adapter wires it — `ask` is offered on this grant, not on
                    // the declaration.
                    clarifier: Some(Arc::new(StaticClarifier)),
                    guidance: None,
                },
                Arc::clone(&turn) as Arc<dyn StageTurn>,
            )
            .await
            .unwrap();
        }

        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["implementer_attempt", "implementer_attempt"]
        );
        assert_eq!(
            *turn
                .sessions
                .lock()
                .expect("recording runner mutex poisoned"),
            [
                ratatoskr_core::SessionScope::Compacted,
                ratatoskr_core::SessionScope::Compacted
            ],
            "the declared stage overrides the fresh route"
        );
        assert_eq!(
            *turn
                .conversations
                .lock()
                .expect("recording runner mutex poisoned"),
            [
                Some("run-standard-implementer-implementer_attempt".to_string()),
                Some("run-standard-implementer-implementer_attempt".to_string())
            ],
            "the conversation is the stage's, so a peer under the same governance cannot inherit it"
        );
        {
            let ledger_ids = turn
                .ledger_ids
                .lock()
                .expect("recording runner mutex poisoned");
            assert_eq!(ledger_ids.len(), 2);
            assert_eq!(ledger_ids[0], ledger_ids[1]);
            assert!(ledger_ids[0].is_some());
        }
        {
            let questions = turn
                .questions
                .lock()
                .expect("recording runner mutex poisoned");
            assert!(
                questions[0]
                    .contains("Implement this task in the current repository:\n\nadd Store::claim")
            );
            assert!(questions[0].contains(
                "Apply the change directly with your editing tools — do NOT ask for confirmation"
            ));
            assert_eq!(
                questions[1],
                "Input contract: ImplementerAttemptInput\nOutput contract: Report\n\nFix the failing concurrent claim test."
            );
        }
        for offered in turn
            .tools
            .lock()
            .expect("recording runner mutex poisoned")
            .iter()
        {
            // The root Rust supplied is what puts the file-mutation tools on the table.
            for expected in ["Read", "Grep", "Glob", "Write", "Edit", "ask"] {
                assert!(
                    offered.iter().any(|tool| tool == expected),
                    "missing {expected}"
                );
            }
            // These invocations were granted no shell, so `Bash` has nothing behind it and is not
            // offered — a tool whose every call is refused only spends turns.
            assert!(
                !offered.iter().any(|tool| tool == "Bash"),
                "Bash was offered without a shell grant"
            );
        }
        let checkpoints = store
            .checkpoints_for_run("run-standard-implementer")
            .await
            .unwrap();
        assert_eq!(
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.node_name.as_str())
                .collect::<Vec<_>>(),
            ["implementer_attempt", "implementer_attempt"],
            "each attempt records its own turn; the `implementer` aggregate stays the host's"
        );
        assert_eq!(
            *turn.files.lock().expect("recording runner mutex poisoned"),
            [Some(dir.clone()), Some(dir.clone())]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn implementer_stage_uses_host_resources_and_rejects_invalid_reports() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-implementer-resources-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let worktree = dir.join("worktree");
        std::fs::create_dir_all(&worktree).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run(
                "run-implementer-resources",
                None,
                RunStatus::Running.as_str(),
            )
            .await
            .unwrap();
        let config = RatatoskrConfig::default();
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-implementer-resources",
            "add Store::claim",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let input = implementer_attempt_input(None);
        let shell = ratatoskr_agent::shell::ShellAccess {
            spec: ratatoskr_exec::SandboxSpec {
                backend: "landlock".to_string(),
                name: "implementer-test-shell".to_string(),
                image: String::new(),
                workdir: "/workspace".to_string(),
                mounts: Vec::new(),
                command: Vec::new(),
                cpus: 1,
                memory_mib: 64,
                network: false,
            },
        };
        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "summary": "implemented the claim" }).to_string(),
            ..Default::default()
        });
        evaluate_standard_stage_with_resources_and_turn(
            Arc::clone(&ctx),
            "implementer_attempt",
            serde_json::to_string(&input).unwrap(),
            StandardStageResources {
                resource_root: worktree.clone(),
                capability_ceiling: ratatoskr_core::Capability::Write,
                rag_rat_worktree: Some(worktree.clone()),
                shell: Some(shell.clone()),
                publish: None,
                clarifier: Some(Arc::new(StaticClarifier)),
                guidance: Some("# WHERE YOU ARE\nThis is the owned worktree.".to_string()),
            },
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .await
        .unwrap();
        assert_eq!(
            turn.files.lock().expect("recording runner mutex poisoned")[0],
            Some(worktree.clone())
        );
        assert_eq!(
            turn.rag_rat_worktrees
                .lock()
                .expect("recording runner mutex poisoned")[0],
            Some(worktree)
        );
        assert!(
            turn.has_shell
                .lock()
                .expect("recording runner mutex poisoned")[0]
        );
        assert!(
            turn.has_clarifier
                .lock()
                .expect("recording runner mutex poisoned")[0]
        );
        assert!(
            turn.preambles
                .lock()
                .expect("recording runner mutex poisoned")[0]
                .contains("This is the owned worktree")
        );

        let invalid = Arc::new(RecordingStageTurn {
            output: json!({ "kind": "fix" }).to_string(),
            ..Default::default()
        });
        let error = evaluate_standard_stage_with_resources_and_turn(
            ctx,
            "implementer_attempt",
            serde_json::to_string(&input).unwrap(),
            StandardStageResources {
                resource_root: dir.clone(),
                capability_ceiling: ratatoskr_core::Capability::Write,
                rag_rat_worktree: None,
                shell: Some(shell),
                publish: None,
                clarifier: None,
                guidance: None,
            },
            invalid,
        )
        .await
        .unwrap_err();
        assert!(error.contains("invalid `Report` output"), "{error}");
        assert!(error.contains("summary"), "{error}");
        assert_eq!(
            store
                .checkpoints_for_run("run-implementer-resources")
                .await
                .unwrap()
                .iter()
                .map(|checkpoint| checkpoint.node_name.as_str())
                .collect::<Vec<_>>(),
            ["implementer_attempt"],
            "an attempt records its own turn; the `implementer` aggregate stays the host's"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_implementer_contract_matches_the_native_report_gate() {
        let stages = standard_stages().await.unwrap();
        let stage = stages
            .iter()
            .find(|stage| stage.id == "implementer_attempt")
            .unwrap();
        let mut declared = stage.output_schema.clone().unwrap();
        let mut generated =
            serde_json::to_value(schemars::schema_for!(crate::implementer::Report)).unwrap();
        without_schema_annotations(&mut declared);
        without_schema_annotations(&mut generated);
        assert_eq!(declared, generated);
    }

    #[tokio::test]
    async fn implementer_stage_migration_keeps_scripted_operation_guards() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-implementer-guards-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-implementer-guards", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let config = RatatoskrConfig::default();
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-implementer-guards",
            "guard implementer lifecycle",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::new(RecordingStageTurn::default())).unwrap();
        assert!(!hosts.contains_key("implementer_attempt"));
        assert!(hosts.contains_key("implement"));
        assert!(hosts.contains_key("iterate"));

        ctx.implement_started.store(true, Ordering::SeqCst);
        let duplicate = implement_host(Arc::clone(&ctx), "{}".to_string())
            .await
            .unwrap_err();
        assert_eq!(duplicate, "implement() called more than once in a workflow");

        let _held = ctx.iterate_lock.lock().await;
        let overlap = iterate_host(Arc::clone(&ctx), "{}".to_string())
            .await
            .unwrap_err();
        assert_eq!(overlap, "iterate() is already in progress");
        // Review excludes editing for the same reason: a verifier that runs while an implementer
        // is mid-edit reviews a torn tree, and its checkpoint is what decides terminal status.
        let executor = StageExecutor::new(
            Arc::clone(&ctx),
            Arc::new(stages),
            Arc::new(RecordingStageTurn::default()),
        );
        let reviewing = verify_host(
            Arc::clone(&ctx),
            executor,
            json!({ "analyst": review_plan() }).to_string(),
        )
        .await
        .unwrap_err();
        assert_eq!(reviewing, "verify() cannot overlap iterate()");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_characterizer_uses_generic_dispatch_and_preserves_its_input_and_prompt() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-characterizer-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"export async function plan(input) { return await characterizer(input); }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run(
                "run-standard-characterizer",
                None,
                RunStatus::Running.as_str(),
            )
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "characterizer".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Compacted,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-standard-characterizer",
            "characterize checks",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let stage = stages
            .iter()
            .find(|stage| stage.id == "characterizer")
            .unwrap();
        assert_eq!(stage.agent, "transcribe");
        assert_eq!(stage.session, Some(ratatoskr_core::SessionScope::Fresh));
        assert!(stage.tools.is_empty());

        let input = crate::testrun::CharacterizerInput {
            outcomes: vec![crate::testrun::StepOutcome {
                name: "workspace tests".to_string(),
                command: vec!["cargo".to_string(), "test".to_string()],
                exit_code: 101,
                output: "test suite::fails ... FAILED\n1 failed; 8 passed".to_string(),
            }],
        };
        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "failing": ["suite::fails"],
                "passed": 8
            })
            .to_string(),
            ..Default::default()
        });
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        runtime
            .run_with_question_renderers(
                "plan",
                serde_json::to_string(&input).unwrap(),
                hosts,
                stage_question_renderers(&stages),
            )
            .await
            .unwrap();

        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["characterizer"]
        );
        assert_eq!(
            *turn
                .sessions
                .lock()
                .expect("recording runner mutex poisoned"),
            [ratatoskr_core::SessionScope::Fresh]
        );
        assert!(
            turn.tools.lock().expect("recording runner mutex poisoned")[0].is_empty(),
            "a transcription stage must receive no tools"
        );
        let question = turn
            .questions
            .lock()
            .expect("recording runner mutex poisoned")[0]
            .clone();
        assert!(question.contains("workspace tests"));
        assert!(question.contains("test suite::fails ... FAILED"));

        let checkpoints = store
            .checkpoints_for_run("run-standard-characterizer")
            .await
            .unwrap();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].node_name, "characterizer");
        let checkpoint_input: crate::testrun::CharacterizerInput =
            serde_json::from_str(checkpoints[0].input_json.as_deref().unwrap()).unwrap();
        assert_eq!(checkpoint_input.outcomes[0].exit_code, 101);
        assert_eq!(checkpoint_input.outcomes[0].command, ["cargo", "test"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_characterizer_contract_rejects_bad_counts_without_a_checkpoint() {
        let stages = standard_stages().await.unwrap();
        let stage = stages
            .iter()
            .find(|stage| stage.id == "characterizer")
            .unwrap();
        assert_eq!(
            stage.instructions,
            include_str!("../prompts/characterizer.md").trim()
        );
        let mut generated =
            serde_json::to_value(schemars::schema_for!(crate::testrun::CharacterizerOutput))
                .unwrap();
        without_schema_annotations(&mut generated);
        assert_eq!(stage.output_schema.as_ref().unwrap(), &generated);

        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-characterizer-schema-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run(
                "run-characterizer-schema",
                None,
                RunStatus::Running.as_str(),
            )
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "characterizer".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-characterizer-schema",
            "characterize",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "failing": [], "passed": "many" }).to_string(),
            ..Default::default()
        });
        let mut hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        let envelope = json!({
            "__ratatoskrRenderedQuestion": {
                "input": {
                    "outcomes": [{
                        "name": "tests",
                        "command": ["cargo", "test"],
                        "exit_code": 0,
                        "output": "8 passed"
                    }]
                },
                "question": "rendered acceptance output"
            }
        })
        .to_string();
        let error = hosts.remove("characterizer").unwrap()(envelope)
            .await
            .unwrap_err();
        assert!(
            error.contains("invalid `CharacterizerOutput` output"),
            "{error}"
        );
        assert!(error.contains("passed"), "{error}");
        assert!(
            store
                .checkpoints_for_run("run-characterizer-schema")
                .await
                .unwrap()
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn embedded_characterizer_evidence_waits_for_rust_reconciliation_before_checkpointing() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-embedded-characterizer-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run(
                "run-embedded-characterizer",
                None,
                RunStatus::Running.as_str(),
            )
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "characterizer".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Compacted,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-embedded-characterizer",
            "characterize",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let input = crate::testrun::CharacterizerInput {
            outcomes: vec![crate::testrun::StepOutcome {
                name: "tests".to_string(),
                command: vec!["cargo".to_string(), "test".to_string()],
                exit_code: 101,
                output: "one failed".to_string(),
            }],
        };
        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "failing": ["suite::one"], "passed": 3 }).to_string(),
            ..Default::default()
        });
        let output = evaluate_standard_stage_with_turn(
            ctx,
            "characterizer",
            serde_json::to_string(&input).unwrap(),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        )
        .await
        .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&output).unwrap(),
            json!({ "failing": ["suite::one"], "passed": 3 })
        );
        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["characterizer"]
        );
        assert!(
            store
                .checkpoints_for_run("run-embedded-characterizer")
                .await
                .unwrap()
                .is_empty(),
            "the composite node checkpoints only after Rust accepts and reconciles this evidence"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_overseer_uses_generic_dispatch_and_preserves_its_input_and_prompt() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-overseer-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"export async function plan(input) { return await overseer(input); }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let rules_dir = dir.join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();
        let engine = ScriptEngine::load(&rules_dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-standard-overseer", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "overseer".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Compacted,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-standard-overseer",
            "explain the session registry",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let overseer_stage = stages.iter().find(|stage| stage.id == "overseer").unwrap();
        assert_eq!(overseer_stage.agent, "reason");
        assert_eq!(
            overseer_stage.session,
            Some(ratatoskr_core::SessionScope::Fresh)
        );
        assert_eq!(
            overseer_stage.tools,
            ["papertrail_issue_search", "semantic_search"]
        );

        let input = crate::overseer::OverseerInput {
            issue: "explain the session registry".to_string(),
            choices: vec![
                crate::overseer::Choice {
                    name: "built-in".to_string(),
                    purpose: "implement a repository change".to_string(),
                    when_to_use: Vec::new(),
                },
                crate::overseer::Choice {
                    name: "research".to_string(),
                    purpose: "answer a repository question".to_string(),
                    when_to_use: vec!["the task asks what or why".to_string()],
                },
            ],
        };
        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "workflow": "research",
                "reasoning": "The task asks to explain the registry."
            })
            .to_string(),
            ..Default::default()
        });
        // Selection is Rust-invoked. `overseer` is never a workflow global, so the dispatch under
        // test is the adapter's — a script asking for it finds nothing there.
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        assert!(!hosts.contains_key("overseer"));
        assert!(
            runtime
                .run_with_question_renderers(
                    "plan",
                    serde_json::to_string(&input).unwrap(),
                    hosts,
                    stage_question_renderers(&stages),
                )
                .await
                .is_err()
        );
        let executor = StageExecutor::new(
            Arc::clone(&ctx),
            Arc::new(stages.clone()),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        );
        execute_standard_stage(
            &executor,
            stages
                .iter()
                .find(|stage| stage.id == "overseer")
                .cloned()
                .unwrap(),
            serde_json::to_string(&input).unwrap(),
            StandardStageInvocation {
                resource_root: None,
                capability_ceiling: ratatoskr_core::Capability::Read,
                rag_rat_worktree: None,
                shell: None,
                publish: None,
                clarifier: None,
                invocation_guidance: None,
                output: StageOutput::Checkpoint,
                after_guard: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["overseer"]
        );
        assert_eq!(
            *turn
                .sessions
                .lock()
                .expect("recording runner mutex poisoned"),
            [ratatoskr_core::SessionScope::Fresh]
        );
        let question = turn
            .questions
            .lock()
            .expect("recording runner mutex poisoned")[0]
            .clone();
        assert!(question.contains("TASK:\nexplain the session registry"));
        assert!(question.contains("name: research\npurpose: answer a repository question"));

        let checkpoints = store
            .checkpoints_for_run("run-standard-overseer")
            .await
            .unwrap();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].node_name, "overseer");
        let checkpoint_input: crate::overseer::OverseerInput =
            serde_json::from_str(checkpoints[0].input_json.as_deref().unwrap()).unwrap();
        assert_eq!(checkpoint_input.choices[1].name, "research");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_overseer_contract_matches_the_typed_gate_and_rejects_bare_choices() {
        let stages = standard_stages().await.unwrap();
        let overseer_stage = stages.iter().find(|stage| stage.id == "overseer").unwrap();
        assert_eq!(
            overseer_stage.instructions,
            include_str!("../prompts/overseer.md").trim()
        );
        let mut generated =
            serde_json::to_value(schemars::schema_for!(crate::overseer::OverseerOutput)).unwrap();
        without_schema_annotations(&mut generated);
        assert_eq!(overseer_stage.output_schema.as_ref().unwrap(), &generated);

        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-overseer-schema-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-overseer-schema", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "overseer".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-overseer-schema",
            "choose",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "workflow": "research" }).to_string(),
            ..Default::default()
        });
        let executor = StageExecutor::new(
            Arc::clone(&ctx),
            Arc::new(stages.clone()),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        );
        let envelope = json!({
            "__ratatoskrRenderedQuestion": {
                "input": {
                    "issue": "choose",
                    "choices": [{
                        "name": "research",
                        "purpose": "answer",
                        "when_to_use": ["a question"]
                    }]
                },
                "question": "AVAILABLE WORKFLOWS:\n\nname: research\n\nTHE TASK:\nchoose\n"
            }
        })
        .to_string();
        // Reached through the executor, not a global: overseer is Rust-invoked.
        let error = executor.host(
            stages
                .iter()
                .find(|stage| stage.id == "overseer")
                .cloned()
                .unwrap(),
        )(envelope)
        .await
        .unwrap_err();
        assert!(error.contains("invalid `OverseerOutput` output"), "{error}");
        assert!(error.contains("reasoning"), "{error}");
        assert!(
            store
                .checkpoints_for_run("run-overseer-schema")
                .await
                .unwrap()
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_verifier_uses_generic_dispatch_and_preserves_its_input_and_prompt() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-verifier-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            r#"export async function plan(input) { return await verifier(input); }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let rules_dir = dir.join("rules");
        std::fs::create_dir_all(&rules_dir).unwrap();
        let engine = ScriptEngine::load(&rules_dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-standard-verifier", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "verifier".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Compacted,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-standard-verifier",
            "review the change",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let stages = standard_stages().await.unwrap();
        let verifier_stage = stages.iter().find(|stage| stage.id == "verifier").unwrap();
        assert_eq!(verifier_stage.agent, "explore");
        assert_eq!(
            verifier_stage.session,
            Some(ratatoskr_core::SessionScope::Fresh)
        );
        assert_eq!(
            verifier_stage.tools,
            [
                "semantic_search",
                "symbol_lookup",
                "impact_surface",
                "memory_search"
            ]
        );

        let input = verifier::VerifierInput {
            issue: "review the change".to_string(),
            analyst: AnalystOutput {
                impact_summary: "changes the session registry".to_string(),
                touched: Vec::new(),
                risks: vec!["P1: a reused key could cross runs".to_string()],
                requirements: vec!["isolate run-local sessions".to_string()],
                residual_risk: String::new(),
                changes_code: true,
                acceptance: Vec::new(),
                interface: Vec::new(),
            },
            diff: "diff --git a/session.rs b/session.rs\n+scope by run id".to_string(),
            touched_files: vec!["session.rs".to_string(), "workflow.rs".to_string()],
            previous_findings: vec![verifier::Finding {
                severity: verifier::Severity::P2,
                kind: verifier::FindingKind::Execution,
                file: "session.rs".to_string(),
                line: Some(8),
                summary: "the key omitted the run".to_string(),
                failure_scenario: "two runs use the same stage id".to_string(),
            }],
        };
        let input_value = serde_json::to_value(&input).unwrap();
        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "findings": [],
                "assessment": "checked the session key and its callers"
            })
            .to_string(),
            ..Default::default()
        });
        // Review is Rust-invoked. `verifier` is never a workflow global — `scripted_review` reads
        // the *last* `verifier` checkpoint, so a script able to call it could answer its own gate.
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        assert!(!hosts.contains_key("verifier"));
        assert!(
            runtime
                .run_with_question_renderers(
                    "plan",
                    input_value.to_string(),
                    hosts,
                    stage_question_renderers(&stages),
                )
                .await
                .is_err()
        );
        let executor = StageExecutor::new(
            Arc::clone(&ctx),
            Arc::new(stages.clone()),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        );
        execute_standard_stage(
            &executor,
            stages
                .iter()
                .find(|stage| stage.id == "verifier")
                .cloned()
                .unwrap(),
            input_value.to_string(),
            StandardStageInvocation {
                resource_root: None,
                capability_ceiling: ratatoskr_core::Capability::Read,
                rag_rat_worktree: None,
                shell: None,
                publish: None,
                clarifier: None,
                invocation_guidance: None,
                output: StageOutput::Checkpoint,
                after_guard: true,
            },
        )
        .await
        .unwrap();

        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["verifier"]
        );
        assert_eq!(
            *turn
                .sessions
                .lock()
                .expect("recording runner mutex poisoned"),
            [ratatoskr_core::SessionScope::Fresh],
            "each review receives its explicit prior findings instead of hidden session history"
        );
        let question = turn
            .questions
            .lock()
            .expect("recording runner mutex poisoned")[0]
            .clone();
        assert!(question.starts_with(
            "Input contract: VerifierInput\nOutput contract: VerifierOutput\n\nTASK:\n"
        ));
        assert!(question.contains("[P2/Execution] session.rs: the key omitted the run"));
        assert!(question.contains("THE CHANGE:\ndiff --git a/session.rs b/session.rs"));

        let checkpoints = store
            .checkpoints_for_run("run-standard-verifier")
            .await
            .unwrap();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].node_name, "verifier");
        let checkpoint_input: verifier::VerifierInput =
            serde_json::from_str(checkpoints[0].input_json.as_deref().unwrap()).unwrap();
        assert_eq!(checkpoint_input.diff, input.diff);
        assert_eq!(
            checkpoint_input.previous_findings[0].summary,
            "the key omitted the run"
        );
        let checkpoint_output: verifier::VerifierOutput =
            serde_json::from_str(&checkpoints[0].output_json).unwrap();
        assert_eq!(
            checkpoint_output.assessment,
            "checked the session key and its callers"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn standard_verifier_contract_matches_the_typed_gate_and_rejects_bad_findings() {
        let stages = standard_stages().await.unwrap();
        let verifier_stage = stages.iter().find(|stage| stage.id == "verifier").unwrap();
        assert_eq!(
            verifier_stage.instructions,
            include_str!("../prompts/verifier.md").trim()
        );
        let mut generated =
            serde_json::to_value(schemars::schema_for!(verifier::VerifierOutput)).unwrap();
        without_schema_annotations(&mut generated);
        assert_eq!(verifier_stage.output_schema.as_ref().unwrap(), &generated);

        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-standard-verifier-schema-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("run-verifier-schema", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "verifier".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-verifier-schema",
            "validate this",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "findings": [{
                    "severity": "P1",
                    "kind": "plan",
                    "summary": "missing the required failure scenario"
                }],
                "assessment": "reviewed"
            })
            .to_string(),
            ..Default::default()
        });
        let executor = StageExecutor::new(
            Arc::clone(&ctx),
            Arc::new(stages.clone()),
            Arc::clone(&turn) as Arc<dyn StageTurn>,
        );
        let envelope = json!({
            "__ratatoskrRenderedQuestion": {
                "input": {
                    "issue": "validate this",
                    "analyst": { "impact_summary": "check it" },
                    "diff": "+change",
                    "touched_files": [],
                    "previous_findings": []
                },
                "question": "TASK:\nvalidate this\n\nTHE CHANGE:\n+change\n"
            }
        })
        .to_string();
        // Reached through the executor, not a global: verifier is Rust-invoked.
        let error = executor.host(
            stages
                .iter()
                .find(|stage| stage.id == "verifier")
                .cloned()
                .unwrap(),
        )(envelope)
        .await
        .unwrap_err();
        assert!(error.contains("invalid `VerifierOutput` output"), "{error}");
        assert!(error.contains("failure_scenario"), "{error}");
        assert!(
            store
                .checkpoints_for_run("run-verifier-schema")
                .await
                .unwrap()
                .is_empty(),
            "invalid output must not be checkpointed"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn verifier_threshold_and_plan_routing_remain_rust_owned() {
        let output = verifier::VerifierOutput {
            findings: vec![
                verifier::Finding {
                    severity: verifier::Severity::P3,
                    kind: verifier::FindingKind::Execution,
                    file: String::new(),
                    line: None,
                    summary: "non-blocking nit".to_string(),
                    failure_scenario: "a cosmetic label is awkward".to_string(),
                },
                verifier::Finding {
                    severity: verifier::Severity::P1,
                    kind: verifier::FindingKind::Plan,
                    file: "session.rs".to_string(),
                    line: None,
                    summary: "the required key cannot isolate runs".to_string(),
                    failure_scenario: "two runs collide".to_string(),
                },
            ],
            assessment: "checked both findings".to_string(),
        };

        let result = verification_result(output, verifier::Severity::P2);

        assert_eq!(result.findings.len(), 2, "all findings remain recorded");
        assert_eq!(result.blocking.len(), 1, "P3 stays below the P2 gate");
        assert_eq!(result.blocking[0].severity, verifier::Severity::P1);
        assert!(
            result.needs_replan,
            "a blocking plan fault routes to the analyst"
        );
    }

    fn review_checkpoint<T: serde::Serialize>(
        node_name: &str,
        output: &T,
        input: Option<&impl serde::Serialize>,
    ) -> ratatoskr_store::Checkpoint {
        ratatoskr_store::Checkpoint {
            node_name: node_name.to_string(),
            output_json: serde_json::to_string(output).unwrap(),
            input_json: input.map(|input| serde_json::to_string(input).unwrap()),
            ..Default::default()
        }
    }

    fn review_plan() -> AnalystOutput {
        AnalystOutput {
            impact_summary: "keep the correction path Rust-owned".to_string(),
            touched: vec!["workflow.rs".to_string()],
            risks: Vec::new(),
            requirements: vec!["validate review causality".to_string()],
            residual_risk: String::new(),
            changes_code: true,
            acceptance: Vec::new(),
            interface: Vec::new(),
        }
    }

    fn review_finding(kind: verifier::FindingKind, summary: &str) -> verifier::Finding {
        verifier::Finding {
            severity: verifier::Severity::P1,
            kind,
            file: "workflow.rs".to_string(),
            line: Some(1),
            summary: summary.to_string(),
            failure_scenario: "a workflow retries the wrong thing".to_string(),
        }
    }

    #[test]
    fn iterate_uses_only_the_checkpointed_review_for_execution_corrections() {
        let plan = review_plan();
        let verifier_input = verifier::VerifierInput {
            issue: "preserve convergence".to_string(),
            analyst: plan,
            diff: "+change".to_string(),
            touched_files: vec!["workflow.rs".to_string()],
            previous_findings: Vec::new(),
        };
        let output = verifier::VerifierOutput {
            findings: vec![review_finding(
                verifier::FindingKind::Execution,
                "the retry omits the review",
            )],
            assessment: "the code needs correction".to_string(),
        };
        let checkpoints = vec![
            review_checkpoint("implementer", &imp(&[], &["a"], 0), None::<&&str>),
            review_checkpoint("verifier", &output, Some(&verifier_input)),
        ];
        let supplied = verification_result(output.clone(), verifier::Severity::P2);

        assert_eq!(
            review_correction(&checkpoints, &supplied, verifier::Severity::P2).unwrap(),
            verifier::correction(&output.blocking(verifier::Severity::P2))
        );

        let fabricated = VerifyResult {
            configured: true,
            unavailable: false,
            findings: Vec::new(),
            blocking: Vec::new(),
            needs_replan: false,
        };
        let error =
            review_correction(&checkpoints, &fabricated, verifier::Severity::P2).unwrap_err();
        assert!(error.contains("does not match"), "{error}");
    }

    #[test]
    fn plan_review_requires_a_causal_analyst_revision() {
        let plan = review_plan();
        let finding = review_finding(
            verifier::FindingKind::Plan,
            "the requirement routes the wrong correction",
        );
        let verifier_input = verifier::VerifierInput {
            issue: "preserve convergence".to_string(),
            analyst: plan.clone(),
            diff: "+change".to_string(),
            touched_files: vec!["workflow.rs".to_string()],
            previous_findings: Vec::new(),
        };
        let output = verifier::VerifierOutput {
            findings: vec![finding.clone()],
            assessment: "the plan needs revision".to_string(),
        };
        let supplied = verification_result(output.clone(), verifier::Severity::P2);
        let mut checkpoints = vec![
            review_checkpoint("analyst", &plan, None::<&&str>),
            review_checkpoint("implementer", &imp(&[], &["a"], 0), None::<&&str>),
            review_checkpoint("verifier", &output, Some(&verifier_input)),
        ];

        let error = review_correction(&checkpoints, &supplied, verifier::Severity::P2).unwrap_err();
        assert!(error.contains("must run analyst() after"), "{error}");

        let mut revised = plan.clone();
        revised.requirements = vec!["route review evidence through the host".to_string()];
        let revision_input = crate::analyst::AnalystInput {
            issue: verifier_input.issue.clone(),
            scout: crate::ScoutOutput {
                related_items: Vec::new(),
                papertrail_summary: String::new(),
            },
            memory: Default::default(),
            brief: String::new(),
            constraints: Vec::new(),
            previous: Some(Box::new(plan)),
            findings: vec![finding],
        };
        checkpoints.push(review_checkpoint(
            "analyst",
            &revised,
            Some(&revision_input),
        ));

        assert_eq!(
            review_correction(&checkpoints, &supplied, verifier::Severity::P2).unwrap(),
            crate::replan(&revised, &output.blocking(verifier::Severity::P2))
        );
    }

    #[test]
    fn verifier_history_contains_only_prior_blocking_review_chains() {
        let blocking = verifier::VerifierOutput {
            findings: vec![review_finding(
                verifier::FindingKind::Execution,
                "first blocking defect",
            )],
            assessment: String::new(),
        };
        let below_threshold = verifier::VerifierOutput {
            findings: vec![verifier::Finding {
                severity: verifier::Severity::P3,
                ..review_finding(verifier::FindingKind::Execution, "cosmetic nit")
            }],
            assessment: String::new(),
        };
        let checkpoints = vec![
            review_checkpoint("verifier", &blocking, None::<&&str>),
            review_checkpoint("verifier", &below_threshold, None::<&&str>),
            review_checkpoint(
                "verifier",
                &serde_json::json!({ "error": "offline" }),
                None::<&&str>,
            ),
        ];

        let findings = previous_verifier_findings(&checkpoints, verifier::Severity::P2);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].summary, "first blocking defect");
    }

    #[tokio::test]
    async fn scripted_verify_returns_rust_routing_and_carries_findings_to_the_next_pass() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-scripted-verify-parity-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let repo = init_test_repo(&dir).await;
        std::fs::write(repo.join("tracked.txt"), "changed\n").unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let run_id = "scripted-verify-parity";
        store
            .upsert_run(run_id, None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let plan = review_plan();
        checkpoint(&store, run_id, "analyst", &plan).await.unwrap();
        checkpoint(&store, run_id, "implementer", &imp(&[], &["a"], 0))
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert("verifier".to_string(), model_route());
        let mut ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "preserve scripted review parity",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        Arc::get_mut(&mut ctx).unwrap().repo_path = repo.clone();
        *ctx.worktree.lock().unwrap() = Some(WorktreePath(repo));
        let finding = review_finding(
            verifier::FindingKind::Execution,
            "the current attempt misses a correction",
        );
        let turn = Arc::new(RecordingStageTurn {
            output: serde_json::to_string(&verifier::VerifierOutput {
                findings: vec![finding.clone()],
                assessment: "the attempt needs correction".to_string(),
            })
            .unwrap(),
            ..Default::default()
        });
        let stages = standard_stages().await.unwrap();
        let mut hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        let verify = hosts.remove("verify").unwrap();
        let arg = json!({ "analyst": plan }).to_string();

        let first: VerifyResult =
            serde_json::from_str(&verify(arg.clone()).await.unwrap()).unwrap();
        assert!(first.configured);
        assert!(!first.unavailable);
        assert_eq!(first.blocking.len(), 1);
        assert!(!first.needs_replan);
        let second: VerifyResult = serde_json::from_str(&verify(arg).await.unwrap()).unwrap();
        assert_eq!(second.blocking.len(), 1);

        let checkpoints = store.checkpoints_for_run(run_id).await.unwrap();
        let input: verifier::VerifierInput =
            serde_json::from_str(checkpoints.last().unwrap().input_json.as_deref().unwrap())
                .unwrap();
        assert_eq!(input.previous_findings.len(), 1);
        assert_eq!(input.previous_findings[0].summary, finding.summary);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn status_is_converged_only_when_post_ran_with_no_new_failures() {
        let baseline = red(&["a"], &["b"], 1);
        // Post ran and introduced nothing the baseline lacked → converged.
        assert_eq!(
            infer_status(
                &baseline,
                &imp(&["a"], &["b", "c"], 0),
                &[],
                None,
                verifier::Severity::P2
            ),
            RunStatus::Converged
        );
        // Post introduced a new failure → wall, not converged.
        assert_eq!(
            infer_status(
                &baseline,
                &imp(&["a", "c"], &["b"], 1),
                &[],
                None,
                verifier::Severity::P2
            ),
            RunStatus::MaxIterationsReached
        );
        // Post didn't run to completion (no tests) → wall, even with an empty failing list — this is
        // the P1a check the hardcoded loop also applies to the implementer output.
        assert_eq!(
            infer_status(
                &baseline,
                &imp(&[], &[], 101),
                &[],
                None,
                verifier::Severity::P2
            ),
            RunStatus::MaxIterationsReached
        );
    }

    #[test]
    fn unavailable_scripted_review_is_unreviewed_without_hiding_other_failed_gates() {
        let unavailable = [ratatoskr_store::Checkpoint {
            node_name: "verifier".to_string(),
            output_json: json!({ "error": "provider offline" }).to_string(),
            ..Default::default()
        }];
        let review = scripted_review(&unavailable);
        assert!(matches!(review, ScriptedReview::Unavailable));
        assert_eq!(
            status_with_review_availability(RunStatus::Converged, &review),
            RunStatus::Unreviewed
        );
        assert_eq!(
            status_with_review_availability(RunStatus::MaxIterationsReached, &review),
            RunStatus::MaxIterationsReached,
            "review availability must not hide a failed deterministic gate"
        );

        let not_run = scripted_review(&[]);
        assert!(matches!(not_run, ScriptedReview::NotRun));
        assert_eq!(
            status_with_review_availability(RunStatus::Converged, &not_run),
            RunStatus::Converged,
            "a workflow that never requested review keeps the documented test-only result"
        );
    }

    #[test]
    fn a_review_the_implementer_has_since_overwritten_is_not_this_runs_review() {
        let clean = verifier::VerifierOutput {
            findings: Vec::new(),
            assessment: "nothing blocking".to_string(),
        };
        let implementer = json!({ "summary": "edited" });
        let none: Option<&()> = None;

        // The bundled order — implement, review — is the reviewed one, and stays reviewed.
        let reviewed = [
            review_checkpoint("implementer", &implementer, none),
            review_checkpoint("verifier", &clean, none),
        ];
        assert!(matches!(
            scripted_review(&reviewed),
            ScriptedReview::Available(_)
        ));

        // Reviewed, then edited again: the review describes a tree that no longer exists, so
        // terminal status must not rest on it.
        let superseded = [
            review_checkpoint("implementer", &implementer, none),
            review_checkpoint("verifier", &clean, none),
            review_checkpoint("implementer", &implementer, none),
        ];
        assert!(matches!(
            scripted_review(&superseded),
            ScriptedReview::NotRun
        ));
    }

    #[tokio::test]
    async fn reconstruct_plan_rebuilds_from_checkpoints_and_missing_is_the_gate() {
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("r1", None, "running").await.unwrap();

        // No scout checkpoint yet → the script can't claim a plan without having run it.
        let missing = reconstruct_plan(&store, "r1").await;
        assert!(matches!(
            missing,
            Err(PlanError::MissingCheckpoint(_, "scout"))
        ));

        store
            .insert_checkpoint(ratatoskr_store::CheckpointWrite {
                run_id: "r1",
                node_name: "scout",
                output_json: r#"{"related_items":[],"papertrail_summary":"s"}"#,
                ..Default::default()
            })
            .await
            .unwrap();
        store
            .insert_checkpoint(ratatoskr_store::CheckpointWrite {
                run_id: "r1",
                node_name: "memory",
                output_json: r#"{"memories":[]}"#,
                ..Default::default()
            })
            .await
            .unwrap();
        store
            .insert_checkpoint(ratatoskr_store::CheckpointWrite {
                run_id: "r1",
                node_name: "analyst",
                output_json: r#"{"impact_summary":"i"}"#,
                ..Default::default()
            })
            .await
            .unwrap();

        let plan = reconstruct_plan(&store, "r1").await.unwrap();
        assert_eq!(plan.scout.papertrail_summary, "s");
        assert_eq!(plan.analyst.impact_summary, "i");
        assert_eq!(plan.state.status, RunStatus::Planned);
    }

    #[test]
    fn reference_example_transpiles() {
        // The shipped example must stay valid TS the runtime can load.
        let src = include_str!("../../../examples/workflow.ts");
        assert!(ratatoskr_script::transpile::transpile_ts(src).is_ok());
    }

    #[tokio::test]
    async fn iterations_count_from_implementer_checkpoints() {
        let store = Store::open_in_memory().unwrap();
        store.upsert_run("r1", None, "running").await.unwrap();
        let cp = r#"{"worktree_path":"/w","failing_tests":[],"passed_tests":1,"exit_code":0}"#;
        for _ in 0..3 {
            store
                .insert_checkpoint(ratatoskr_store::CheckpointWrite {
                    run_id: "r1",
                    node_name: "implementer",
                    output_json: cp,
                    ..Default::default()
                })
                .await
                .unwrap();
        }
        assert_eq!(
            count_checkpoints(&store, "r1", "implementer")
                .await
                .unwrap(),
            3
        );
    }

    #[tokio::test]
    async fn scripted_implementer_checkpoints_preserve_attempt_ordinals_for_friction() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-scripted-implementer-iteration-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let run_id = "scripted-implementer-iteration";
        store
            .upsert_run(run_id, None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let ctx = WorkflowContext::new(
            None,
            &RatatoskrConfig::default(),
            &store,
            run_id,
            "preserve corrective diagnostics",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();

        note(&ctx, "implementer", &imp(&[], &["first"], 0), None)
            .await
            .unwrap();
        note(
            &ctx,
            "implementer",
            &imp(&[], &["second"], 0),
            Some("Fix the checkpointed correction.".to_string()),
        )
        .await
        .unwrap();

        let checkpoints = store.checkpoints_for_run(run_id).await.unwrap();
        assert_eq!(
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.iteration)
                .collect::<Vec<_>>(),
            [Some(1), Some(2)]
        );
        assert_eq!(
            crate::bookkeeper::RunFriction::from_checkpoints(&checkpoints).diagnostics,
            ["Fix the checkpointed correction."]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn scripted_no_code_terminal_publishes_the_plan_without_fork_or_bookkeeping() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-scripted-no-code-terminal-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let run_id = "scripted-no-code-terminal";
        terminal_plan(&store, run_id, false).await;
        let config = RatatoskrConfig::default();
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "summarize the architecture",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let actions = RecordingTerminalActions::new(true, true);

        let outcome = finish_full(&ctx, &actions).await.unwrap();

        assert_eq!(outcome.status, RunStatus::NoCodeChange);
        assert!(outcome.red_team.is_none());
        assert!(outcome.implementer.is_none());
        assert!(outcome.worktree.is_none());
        assert!(outcome.bookkeeper.is_none());
        assert_eq!(outcome.iterations, 0);
        assert_eq!(outcome.state.artifacts.len(), 1);
        assert_eq!(
            store.run_status(run_id).await.unwrap().as_deref(),
            Some(RunStatus::NoCodeChange.as_str())
        );
        assert_eq!(
            actions.delivery_statuses(),
            [Some(RunStatus::Running.as_str().to_string())],
            "publisher remains resumable until it has finished"
        );
        assert_eq!(
            actions.calls(),
            [TerminalCall::Publish {
                status: RunStatus::NoCodeChange.as_str().to_string(),
                has_implementer: false,
                terminal: true,
                iterations: 0,
                unresolved: 0,
            }]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn scripted_terminal_delivery_keeps_the_run_resumable_until_every_stage_finishes() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-scripted-terminal-parity-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let run_id = "scripted-terminal-parity";
        terminal_plan(&store, run_id, true).await;
        checkpoint(&store, run_id, "redteam", &red(&["old"], &[], 1))
            .await
            .unwrap();
        let worktree = dir.join("worktree");
        std::fs::create_dir_all(&worktree).unwrap();
        let mut implementer = imp(&["old"], &["new"], 0);
        implementer.worktree_path = worktree.display().to_string();
        checkpoint(&store, run_id, "implementer", &implementer)
            .await
            .unwrap();
        checkpoint(
            &store,
            run_id,
            "verifier",
            &json!({ "error": "provider unavailable" }),
        )
        .await
        .unwrap();
        let config = RatatoskrConfig::default();
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "implement terminal parity",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        *ctx.worktree.lock().unwrap() = Some(WorktreePath(worktree.clone()));
        let actions = RecordingTerminalActions::new(true, true);

        let outcome = finish_full(&ctx, &actions).await.unwrap();

        assert_eq!(outcome.status, RunStatus::Unreviewed);
        assert!(outcome.bookkeeper.is_some());
        assert_eq!(outcome.state.artifacts.len(), 2);
        assert_eq!(
            store.run_status(run_id).await.unwrap().as_deref(),
            Some(RunStatus::Unreviewed.as_str())
        );
        assert_eq!(
            actions.delivery_statuses(),
            [
                Some(RunStatus::Running.as_str().to_string()),
                Some(RunStatus::Running.as_str().to_string()),
            ],
            "publisher and bookkeeper both run while the dashboard can resume a provider pause"
        );
        let calls = actions.calls();
        assert_eq!(
            calls.first(),
            Some(&TerminalCall::Commit {
                branch: implementer.branch.clone(),
                worktree,
            }),
            "commit must finish before either external delivery begins"
        );
        assert!(calls.contains(&TerminalCall::Bookkeep {
            converged: false,
            iterations: 1,
        }));
        assert!(calls.contains(&TerminalCall::Publish {
            status: RunStatus::Unreviewed.as_str().to_string(),
            has_implementer: true,
            terminal: true,
            iterations: 1,
            unresolved: 0,
        }));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn scripted_full_failure_removes_its_owned_worktree_and_records_failed() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-scripted-full-cleanup-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let repo = init_test_repo(&dir).await;
        let workflow_path = dir.join("workflow.ts");
        std::fs::write(
            &workflow_path,
            "export async function run() { throw new Error('terminal failure'); }",
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path, &[])
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir.join("rules")).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let config = RatatoskrConfig::default();
        let run_id = "scripted-full-cleanup";
        let mut ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            run_id,
            "fail after creating a worktree",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        Arc::get_mut(&mut ctx).unwrap().repo_path = repo.clone();
        let worktree = ratatoskr_exec::create_worktree(
            &repo,
            &dir.join("worktrees"),
            "ratatoskr/scripted-full-cleanup",
        )
        .await
        .unwrap();
        *ctx.worktree.lock().unwrap() = Some(worktree.clone());

        let error = match run_full_scripted_with_actions(
            runtime,
            Arc::clone(&ctx),
            &RecordingTerminalActions::new(false, false),
        )
        .await
        {
            Ok(_) => panic!("the workflow failure must fail the run"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("terminal failure"), "{error}");
        assert!(!worktree.as_path().exists());
        assert!(ctx.worktree.lock().unwrap().is_none());
        assert_eq!(
            store.run_status(run_id).await.unwrap().as_deref(),
            Some(RunStatus::Failed.as_str())
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A failing first attempt must send the standard loop back to `iterate()`, not on to review.
    ///
    /// Every host is an `async function`, so a bare `testCommandRan(x) && isConverged(y)` is a
    /// Promise — always truthy — and the loop would take the clean branch regardless of the test
    /// results. This drives the bundled composition and reads the decision off the host calls it
    /// actually made, which is the only place the difference shows.
    #[tokio::test]
    async fn a_failing_first_attempt_iterates_before_it_reviews() {
        let runtime = standard_runtime().await.unwrap();
        let calls = Arc::new(Mutex::new(Vec::<String>::new()));

        // Every host records its own name; the ones the loop branches on answer from the
        // implementation shape it is handed, exactly as their Rust owners do.
        let mut hosts: HashMap<String, HostFn> = HashMap::new();
        let mut bind = |name: &str, reply: fn(serde_json::Value) -> serde_json::Value| {
            let calls = Arc::clone(&calls);
            let recorded = name.to_string();
            let host: HostFn = Arc::new(move |arg: String| {
                let calls = Arc::clone(&calls);
                let recorded = recorded.clone();
                Box::pin(async move {
                    calls.lock().expect("call log poisoned").push(recorded);
                    let arg: serde_json::Value = serde_json::from_str(&arg).unwrap();
                    Ok(reply(arg).to_string())
                })
            });
            hosts.insert(name.to_string(), host);
        };

        bind("context", |_| json!({ "scout": {}, "memory": {} }));
        bind("analyst", |_| json!({ "changes_code": true }));
        bind("redTeam", |_| json!({ "failing_tests": ["acceptance"] }));
        // The first attempt leaves the acceptance test failing; `iterate` is what fixes it.
        bind(
            "implement",
            |_| json!({ "failing_tests": ["acceptance"], "passed_tests": 0, "exit_code": 1 }),
        );
        bind(
            "iterate",
            |_| json!({ "failing_tests": [], "passed_tests": 1, "exit_code": 0 }),
        );
        bind("testCommandRan", |arg| json!(arg["exit_code"] == 0));
        bind("isConverged", |arg| {
            json!(arg["post"]["failing_tests"].as_array().unwrap().is_empty())
        });
        bind("verify", |_| {
            json!({
                "configured": false,
                "unavailable": false,
                "findings": [],
                "blocking": [],
                "needsReplan": false,
            })
        });
        bind("replanAtCeiling", |_| serde_json::Value::Null);

        runtime
            .run(
                "run",
                json!({ "issue": "x", "maxIterations": 3, "alwaysFork": false }).to_string(),
                hosts,
            )
            .await
            .unwrap();

        let calls = calls.lock().expect("call log poisoned").clone();
        let first = |name: &str| calls.iter().position(|call| call == name);
        assert!(
            first("iterate").is_some(),
            "a failing attempt must be sent back to the implementer; calls were {calls:?}"
        );
        assert!(
            first("iterate") < first("verify"),
            "review must not run until the tests are clean; calls were {calls:?}"
        );
    }

    /// Guards the whole class the behavioural test above catches one instance of.
    ///
    /// A host call is an `async function` call, so its value is a Promise until awaited — and a
    /// Promise is truthy, is not `===` anything, and stringifies to `[object Promise]`. None of
    /// that is a type error, so nothing but reading the source catches it. Every host call in a
    /// source we ship therefore has to be awaited on the spot. A workflow that deliberately forks
    /// with `Promise.all` is the one shape this forbids; award it its own awaited binding first.
    #[test]
    fn every_host_call_in_a_shipped_workflow_is_awaited() {
        let sources = [
            ("workflows/standard-v1.ts", STANDARD_WORKFLOW_V1),
            (
                "examples/workflow.ts",
                include_str!("../../../examples/workflow.ts"),
            ),
        ];
        let mut unawaited = Vec::new();
        for (path, source) in sources {
            // A source's hosts are the Rust-owned operations plus every stage it declares:
            // `build_declared_stage_hosts` installs declared stages through the same async
            // wrapper, so they carry the same Promise, and they are most of what a workflow calls.
            let mut hosts: Vec<String> = OPERATION_HOSTS
                .iter()
                .map(|(name, _)| name.to_string())
                .collect();
            hosts.extend(source.match_indices("stage(\"").filter_map(|(at, opener)| {
                let rest = &source[at + opener.len()..];
                rest.split_once('"').map(|(id, _)| id.to_string())
            }));
            assert!(
                hosts.len() > OPERATION_HOSTS.len(),
                "{path} declares no stages, so this guard would only cover the operations"
            );
            for (number, line) in source.lines().enumerate() {
                if line.trim_start().starts_with("//") {
                    continue;
                }
                for host in &hosts {
                    let host = host.as_str();
                    for (at, _) in line.match_indices(host) {
                        let before = &line[..at];
                        let after = line[at + host.len()..].trim_start();
                        // A call, not a property, a key, or a longer identifier that contains it.
                        if !after.starts_with('(')
                            || before
                                .chars()
                                .next_back()
                                .is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.')
                        {
                            continue;
                        }
                        if !before.trim_end().ends_with("await") {
                            unawaited.push(format!("{path}:{}: {}", number + 1, line.trim()));
                        }
                    }
                }
            }
        }
        assert!(
            unawaited.is_empty(),
            "these host calls are missing an `await`, so each yields a truthy Promise \
             rather than its value:\n{}",
            unawaited.join("\n")
        );
    }

    #[test]
    fn scripted_review_warning_describes_the_optional_binding() {
        assert!(crate::SCRIPTED_REVIEW_WARNING.contains("controls whether to run the verifier"));
        assert!(crate::SCRIPTED_REVIEW_WARNING.contains("if it omits verify()"));
        assert!(!crate::SCRIPTED_REVIEW_WARNING.contains("has no verifier binding"));
    }

    fn finding(severity: verifier::Severity) -> verifier::Finding {
        verifier::Finding {
            severity,
            kind: verifier::FindingKind::Execution,
            file: "a.rs".into(),
            line: None,
            summary: "s".into(),
            failure_scenario: "f".into(),
        }
    }

    #[test]
    fn a_script_cannot_converge_over_a_review_it_ignored() {
        // Calling verify(), leaving blocking findings standing and returning is not convergence.
        // The status is inferred from the checkpoint, so ignoring the result is not an option the
        // script has.
        let baseline = red(&["a"], &["b"], 1);
        let clean_tests = imp(&["a"], &["b", "c"], 0);
        let blocked = verifier::VerifierOutput {
            findings: vec![finding(verifier::Severity::P1)],
            assessment: String::new(),
        };
        assert_eq!(
            infer_status(
                &baseline,
                &clean_tests,
                &[],
                Some(&blocked),
                verifier::Severity::P2
            ),
            RunStatus::MaxIterationsReached
        );

        // A review that found only what falls below the threshold does not block — recorded is not
        // the same as blocking.
        let nits = verifier::VerifierOutput {
            findings: vec![finding(verifier::Severity::P3)],
            assessment: String::new(),
        };
        assert_eq!(
            infer_status(
                &baseline,
                &clean_tests,
                &[],
                Some(&nits),
                verifier::Severity::P2
            ),
            RunStatus::Converged
        );

        // And a workflow that never reviewed converges on its tests, which is the behaviour the
        // run-start warning describes rather than a silent one.
        assert_eq!(
            infer_status(&baseline, &clean_tests, &[], None, verifier::Severity::P2),
            RunStatus::Converged
        );
    }

    #[test]
    fn acceptance_is_decided_once_and_reused() {
        // A script can re-analyse between iterations. If each binding resolved its own acceptance,
        // the plan could move the bar it is judged against mid-run.
        let step = |name: &str| ratatoskr_core::AcceptanceStep {
            name: name.to_string(),
            command: vec![name.to_string()],
        };
        let config = RatatoskrConfig::default();
        let slot: Mutex<Option<Vec<ratatoskr_core::AcceptanceStep>>> = Mutex::new(None);
        let resolve = |proposed: &[ratatoskr_core::AcceptanceStep]| {
            let mut guard = slot.lock().unwrap();
            guard
                .get_or_insert_with(|| config.sandbox.acceptance(proposed))
                .clone()
        };

        let first = resolve(&[step("wasm")]);
        assert_eq!(first[0].name, "wasm");
        // A later call proposing something else gets what the run already froze.
        let later = resolve(&[step("something-else")]);
        assert_eq!(later[0].name, "wasm");
    }

    /// Contract reading (#206): the judgement's violations reach `infer_status` as a
    /// `&[referee::Violation]` where `may_modify_tests` used to sit. The exemption is applied by
    /// `converge::referee_candidates` before the judgement runs, so by the time the status is
    /// inferred there is nothing left to exempt — what arrives here is what the judgement found.
    fn violation(file: &str, reason: &str) -> crate::referee::Violation {
        crate::referee::Violation {
            file: file.into(),
            reason: reason.into(),
        }
    }

    #[test]
    fn referee_violations_block_convergence_even_when_the_tests_are_clean() {
        // The #205 shape with the gate working: the inline characterisation module was deleted,
        // every remaining test passes, and the run still must not converge.
        let baseline = red(&["a"], &["b"], 1);
        let mut cheated = imp(&[], &["a", "b"], 0);
        cheated.rewritten_files = vec!["crates/ratatoskr-nodes/src/lib.rs".to_string()];
        let violations = vec![violation(
            "crates/ratatoskr-nodes/src/lib.rs",
            "deleted the #[cfg(test)] module that characterised the move",
        )];
        assert_eq!(
            infer_status(
                &baseline,
                &cheated,
                &violations,
                None,
                verifier::Severity::P2
            ),
            RunStatus::MaxIterationsReached
        );
    }

    #[test]
    fn an_empty_judgement_converges_on_the_tests_alone() {
        let baseline = red(&["a"], &["b"], 1);
        // A mayModifyTests-exempt rewrite never reaches the judgement, so the violations are
        // empty and the rewrite cannot flip the status — infer_status trusts the judgement it is
        // handed rather than re-deriving anything from the rewritten-file list.
        let mut exempt = imp(&["a"], &["b", "c"], 0);
        exempt.rewritten_files = vec!["tests/api.rs".to_string()];
        assert_eq!(
            infer_status(&baseline, &exempt, &[], None, verifier::Severity::P2),
            RunStatus::Converged
        );
        // Which is also the no-route case: no referee and no verifier configured, the scripted
        // path infers Converged from the tests alone (the #112 shape).
        let plain = imp(&["a"], &["b", "c"], 0);
        assert_eq!(
            infer_status(&baseline, &plain, &[], None, verifier::Severity::P2),
            RunStatus::Converged
        );
    }

    // --- the run's one resolved container image (#149) -------------------------
    //
    // Contract reading: WorkflowContext holds the run's one resolved container image and hands
    // it out through `resolved_container_image` — `Ok(Some(digest))` for a container-backed run,
    // `Ok(None)` for a backend with no image. The contract names no accessor; what is pinned
    // here is the behaviour it states: resolution happens once, every sandbox the run builds
    // uses that identifier, a retag after resolution cannot move it, and selecting `container`
    // without a runtime is an error rather than a downgrade to landlock.

    /// Restores `PATH` on drop. Prepending is safe for tests running alongside (every lookup
    /// that resolved before still resolves); replacing is not, so the replaced window is kept
    /// to a single resolution call.
    struct PathGuard(Option<std::ffi::OsString>);

    impl PathGuard {
        fn prepended(dir: &std::path::Path) -> Self {
            let old = std::env::var_os("PATH");
            let mut paths = vec![dir.to_path_buf()];
            if let Some(old) = &old {
                paths.extend(std::env::split_paths(old));
            }
            // SAFETY: process-environment mutation races with other tests; this only adds a
            // directory, and drop restores.
            unsafe { std::env::set_var("PATH", std::env::join_paths(paths).unwrap()) };
            Self(old)
        }

        fn replaced_with(dir: &std::path::Path) -> Self {
            let old = std::env::var_os("PATH");
            // SAFETY: races with concurrent tests' subprocess spawns for the guard's lifetime;
            // kept to the one call that must see a runtime-less PATH, and drop restores.
            unsafe { std::env::set_var("PATH", dir) };
            Self(old)
        }
    }

    impl Drop for PathGuard {
        fn drop(&mut self) {
            // SAFETY: restores exactly what the guard found.
            unsafe {
                match &self.0 {
                    Some(value) => std::env::set_var("PATH", value),
                    None => std::env::remove_var("PATH"),
                }
            }
        }
    }

    /// A directory posing as a container-runtime installation: an executable `docker` whose
    /// `inspect` answers with `digest` — raw when asked with `--format`, as a JSON array
    /// otherwise, because which of the two a correct implementation uses is its own business —
    /// and which appends to `<dir>/inspections` every time it is asked, so a test can count.
    fn fake_container_runtime(digest: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-fake-runtime-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let docker = dir.join("docker");
        write_fake_runtime(&docker, &dir.join("inspections"), digest);
        dir
    }

    fn write_fake_runtime(docker: &std::path::Path, inspections: &std::path::Path, digest: &str) {
        // One long format string rather than line continuations: the script's own indentation
        // is significant enough that eating it with Rust's `\`-newline would be a quiet bug.
        let script = format!(
            "#!/bin/sh\ncase \" $* \" in\n  *\" inspect \"*)\n    echo ask >> '{}'\n    case \" $* \" in\n      *\" --format \"*) echo '{digest}' ;;\n      *) printf '[{{\"Id\":\"{digest}\"}}]\\n' ;;\n    esac ;;\n  *) exit 1 ;;\nesac\n",
            inspections.display()
        );
        std::fs::write(docker, script).unwrap();
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(docker, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    #[tokio::test]
    async fn the_runs_container_image_is_resolved_once_and_a_retag_cannot_move_it() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-image-freeze-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.sandbox.backend = "container".to_string();
        config.sandbox.image = "ratatoskr-checks".to_string();
        config
            .models
            .insert("implementer".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-image-freeze",
            "pin the execution environment",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();

        let first_digest = format!("sha256:{}", "ab".repeat(32));
        let runtime_dir = fake_container_runtime(&first_digest);
        let _path = PathGuard::prepended(&runtime_dir);

        let first = ctx.resolved_container_image().await.unwrap();
        assert_eq!(first.as_deref(), Some(first_digest.as_str()));

        // The tag moves underneath the run — a new build pushed under the same name.
        let later_digest = format!("sha256:{}", "cd".repeat(32));
        write_fake_runtime(
            &runtime_dir.join("docker"),
            &runtime_dir.join("inspections"),
            &later_digest,
        );

        // A later sandbox construction in the same run gets the first identifier, not the new
        // one — a run is one immutable execution environment.
        let second = ctx.resolved_container_image().await.unwrap();
        assert_eq!(
            second.as_deref(),
            Some(first_digest.as_str()),
            "a retag after resolution changed the image a later step would run"
        );
        let inspections =
            std::fs::read_to_string(runtime_dir.join("inspections")).unwrap_or_default();
        assert_eq!(
            inspections.lines().count(),
            1,
            "the image was inspected more than once in one run: {inspections}"
        );

        // And what was resolved is what the implementer's sandbox is built from — the digest,
        // not the mutable tag the config named.
        let implementer = build_implementer(&ctx, review_plan()).await.unwrap();
        let red_team = build_red_team(&ctx, Vec::new()).await.unwrap();
        assert_eq!(implementer.sandbox.image, first_digest);
        assert_eq!(red_team.sandbox.image, first_digest);

        let _ = std::fs::remove_dir_all(runtime_dir);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn the_landlock_backend_attempts_no_image_resolution() {
        // The configured fallback keeps its existing behaviour: no OCI inspection is attempted,
        // and there is no image digest provenance to record.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-image-landlock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.sandbox.backend = "landlock".to_string();
        config
            .models
            .insert("implementer".to_string(), model_route());
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-image-landlock",
            "keep the fallback unchanged",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();

        let runtime_dir = fake_container_runtime(&format!("sha256:{}", "ab".repeat(32)));
        let _path = PathGuard::prepended(&runtime_dir);

        let resolved = ctx.resolved_container_image().await.unwrap();
        assert!(
            resolved.is_none(),
            "a backend with no image has no digest to resolve"
        );
        assert!(
            !runtime_dir.join("inspections").exists(),
            "the landlock backend attempted an OCI inspection"
        );

        // The sandbox the run builds is untouched: the host root and the host's toolchain,
        // exactly as configured.
        let implementer = build_implementer(&ctx, review_plan()).await.unwrap();
        assert_eq!(implementer.sandbox.backend, "landlock");
        assert_eq!(implementer.sandbox.image, config.sandbox.image);

        let _ = std::fs::remove_dir_all(runtime_dir);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_container_run_without_a_runtime_fails_resolution_instead_of_downgrading() {
        // Selecting `container` on a host with neither Docker nor Podman fails before any
        // sandboxed step starts, naming that a container runtime is required. The one thing it
        // must not do is run landlock instead: the config asked for no host root, and the
        // fallback's exposure is exactly that.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-image-no-runtime-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        let mut config = RatatoskrConfig::default();
        config.sandbox.backend = "container".to_string();
        config.sandbox.image = "ratatoskr-checks".to_string();
        let ctx = WorkflowContext::new(
            None,
            &config,
            &store,
            "run-image-no-runtime",
            "fail rather than downgrade",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();

        let empty = dir.join("no-runtime-here");
        std::fs::create_dir_all(&empty).unwrap();
        let err = {
            let _path = PathGuard::replaced_with(&empty);
            ctx.resolved_container_image()
                .await
                .expect_err("resolution without a runtime must fail")
        };
        assert!(
            err.to_string().contains("container runtime"),
            "the failure must name what is required: {err}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }
}
