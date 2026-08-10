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
use ratatoskr_graph::{Node, NodeError};
use ratatoskr_mcp::{RagRatClient, ServerTools};
use ratatoskr_script::{HostFn, ScriptEngine, WorkflowRuntime};
use ratatoskr_store::Store;
use rmcp::service::ServerSink;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::json;

use crate::{
    AnalystOutput, BookkeeperInput, BookkeeperOutput, ChildTask, ImplementerNode,
    ImplementerOutput, MemoryNode, MemoryOutput, PlanError, PlanOutcome, RedTeamNode,
    RedTeamOutput, RunOutcome, ScoutOutput, Stage, bookkeeper, checkpoint, converge, memory,
    publisher::{PublisherInput, PublisherOutput},
    redteam, referee, stage_agent_config, verifier,
};

/// Backstop on total node-running binding calls per run — a runaway-loop guard, far above any real
/// workflow. `max_iterations` and the false-convergence guard are the precise limits; this only
/// catches a script that ignores them and loops.
const INVOCATION_CEILING: usize = 500;

const STANDARD_WORKFLOW_NAME: &str = "ratatoskr-standard-v1";
const STANDARD_WORKFLOW_V1: &str = include_str!("../workflows/standard-v1.ts");
const STANDARD_WORKFLOW_INCLUDES: &[(&str, &str)] = &[
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
    /// rag-rat's whole offer, the base of every node's tool pool.
    rag_rat: Option<ServerTools>,
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
}

pub(crate) struct WorkflowContextParams<'a> {
    pub client: Option<&'a RagRatClient>,
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
        let clarifier = crate::clarify::NodeClarifier::new(config, store, engine, run_id, issue);
        Ok(Arc::new(Self {
            ledger,
            clarifier,
            acceptance: Mutex::new(None),
            plugin_context,
            config: config.clone(),
            store: store.clone(),
            engine: Arc::clone(engine),
            run_id: run_id.to_string(),
            issue: issue.to_string(),
            sink: client.map(|c| c.sink()),
            rag_rat: client.map(|c| c.offer()),
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
    let Some(checkpoint) = checkpoints
        .iter()
        .rev()
        .find(|checkpoint| checkpoint.node_name == "verifier")
    else {
        return ScriptedReview::NotRun;
    };
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

async fn memory_host(ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    ctx.guard()?;
    let input: memory::MemoryInput =
        serde_json::from_str(&arg).map_err(|e| format!("memory arg: {e}"))?;
    let node = MemoryNode {
        sink: ctx.sink.clone(),
    };
    let out = node
        .run(input, &RunState::new(&ctx.run_id, None))
        .await
        .map_err(|e| e.to_string())?;
    note(&ctx, "memory", &out, Some(arg)).await?;
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

/// `acceptance` is passed in rather than read here because the baseline and the post-change run
/// must execute the same steps — a red team that resolved its own would drift from the implementer
/// the moment a plan proposed anything.
fn build_red_team(
    ctx: &Arc<WorkflowContext>,
    acceptance: Vec<ratatoskr_core::AcceptanceStep>,
) -> Result<RedTeamNode, PlanError> {
    let short: String = ctx.run_id.chars().take(8).collect();
    let enabled = crate::classifier_enabled(&ctx.engine, &ctx.config);
    let classifier = match enabled {
        true => {
            let mut plugins = ctx.plugin_context.for_node("redteam");
            let cfg = stage_agent_config(
                &ctx.engine,
                &ctx.config,
                ctx.plugin_context.pool_for("redteam", ctx.rag_rat.clone()),
                "redteam",
                redteam::CLASSIFIER_TOOLS,
                &mut plugins,
            )?;
            Some(redteam::RedTeamClassifier {
                route: cfg.route,
                tools: cfg.tools,
                files: cfg.files,
                ledger: Some(Arc::clone(&ctx.ledger)),
                policy: cfg.policy,
                max_turns: cfg.max_turns,
                clarifier: None,
                system_prompt: cfg.system_prompt,
                plugins,
                declared_context: Arc::clone(ctx),
            })
        }
        false => None,
    };
    let author = match enabled {
        true => {
            let mut plugins = ctx.plugin_context.for_node("redteam");
            let mut tools = ctx.plugin_context.pool_for("redteam", ctx.rag_rat.clone());
            tools
                .local()
                .tools
                .extend(ratatoskr_agent::files::edit_declarations());
            let cfg = crate::plugins::redteam_author_agent_config(
                &ctx.engine,
                &ctx.config,
                tools,
                redteam::AUTHOR_TOOLS,
                &mut plugins,
            )?;
            Some(redteam::TestAuthor {
                route: cfg.route,
                tools: cfg.tools,
                policy: cfg.policy,
                max_turns: cfg.max_turns,
                system_prompt: cfg.system_prompt,
                conventions: crate::repo_conventions(&ctx.repo_path),
                plugins,
                ledger: Some(Arc::clone(&ctx.ledger)),
                declared_context: Arc::clone(ctx),
            })
        }
        false => None,
    };
    Ok(RedTeamNode {
        author,
        acceptance,
        characterizer: crate::build_characterizer(
            &ctx.engine,
            &ctx.config,
            &ctx.plugin_context,
            ctx.rag_rat.clone(),
            Some(Arc::clone(&ctx.ledger)),
            Some(Arc::clone(ctx)),
        )?,
        repo_path: ctx.repo_path.clone(),
        worktree_root: ctx.config.worktree.root.clone(),
        baseline_branch: format!("ratatoskr/{short}-baseline"),
        sandbox: ctx.config.sandbox.clone(),
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
    crate::record(crate::Record {
        store: &ctx.store,
        run_id: &ctx.run_id,
        node,
        output: out,
        input,
        // A script chooses its own order, so a checkpoint's position in the loop is whatever the
        // script made it; counting them here would invent an iteration the script never declared.
        iteration: None,
        ledger: Some(&ctx.ledger),
    })
    .await
    .map_err(|e| e.to_string())
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
    let acceptance = ctx.acceptance(&analyst.acceptance);
    let implementer = build_implementer(&ctx, analyst.clone()).map_err(|e| e.to_string())?;
    let worktree = implementer.prepare().await.map_err(|e| e.to_string())?;
    *ctx.worktree.lock().unwrap() = Some(worktree.clone());
    let node = build_red_team(&ctx, acceptance).map_err(|e| e.to_string())?;
    let out = node
        .run_and_author(worktree.as_path(), &ctx.issue, &analyst.interface)
        .await
        .map_err(|e| e.to_string())?;
    // Checkpoint before the guard so a failed baseline stays inspectable.
    note(&ctx, "red_team", &out, None).await?;
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

fn build_implementer(
    ctx: &Arc<WorkflowContext>,
    analyst: AnalystOutput,
) -> Result<ImplementerNode, PlanError> {
    let (cfg, plugins) = crate::build_implementer_agent(
        &ctx.engine,
        &ctx.config,
        &ctx.plugin_context,
        ctx.rag_rat.clone(),
    )?;
    Ok(ImplementerNode {
        clarifier: Some(ctx.clarifier.as_dyn()),
        acceptance: ctx.acceptance(&analyst.acceptance),
        characterizer: crate::build_characterizer(
            &ctx.engine,
            &ctx.config,
            &ctx.plugin_context,
            ctx.rag_rat.clone(),
            Some(Arc::clone(&ctx.ledger)),
            Some(Arc::clone(ctx)),
        )
        .ok()
        .flatten(),
        repo_path: ctx.repo_path.clone(),
        worktree_root: ctx.config.worktree.root.clone(),
        sandbox: ctx.config.sandbox.clone(),
        route: cfg.route,
        tools: cfg.tools,
        policy: cfg.policy,
        max_turns: cfg.max_turns,
        system_prompt: cfg.system_prompt,
        conventions: crate::repo_conventions(&ctx.repo_path),
        plugins,
        ledger: Some(Arc::clone(&ctx.ledger)),
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
    let node = build_implementer(&ctx, input.analyst).map_err(|e| e.to_string())?;
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
    note(&ctx, "implementer", &out, Some(arg)).await?;
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

async fn referee_judgement(
    ctx: &Arc<WorkflowContext>,
    worktree: &WorktreePath,
    analyst: &AnalystOutput,
    implementer: &ImplementerOutput,
) -> Vec<referee::Violation> {
    let violations = match referee::judge(
        &ctx.engine,
        &ctx.config,
        &ctx.ledger,
        &ctx.issue,
        &analyst.requirements,
        implementer,
        worktree,
    )
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
    let red_team: RedTeamOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "red_team")
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

    let node = build_implementer(&ctx, analyst).map_err(|e| e.to_string())?;
    let out = node
        .iterate(&worktree, &diagnostic)
        .await
        .map_err(|e| e.to_string())?;
    // The diagnostic, not the binding's argument: the script does not author it, so it is the one
    // thing that explains what this iteration was actually asked to fix.
    note(&ctx, "implementer", &out, Some(diagnostic)).await?;
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
    if !crate::verifier_enabled(&ctx.engine, &ctx.config) {
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
        let rendered_question = verifier::render_prompt(&verifier_input);
        let stage = executor
            .stages
            .iter()
            .find(|stage| stage.id == "verifier")
            .cloned()
            .ok_or_else(|| "standard verifier stage is not registered".to_string())?;
        match executor
            .execute_after_guard(StageInvocation {
                stage,
                input_json: input_json.clone(),
                rendered_question: Some(rendered_question),
                resource_root: Some(worktree.0.clone()),
                shell: None,
                publish: None,
                clarifier: None,
                invocation_guidance: None,
                output: StageOutput::Checkpoint,
            })
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
        let raw = match executor
            .execute_after_guard(StageInvocation {
                stage,
                input_json,
                rendered_question: Some(crate::analyst::render_prompt(input)),
                resource_root: None,
                shell: None,
                publish: None,
                clarifier: None,
                invocation_guidance: None,
                output: StageOutput::Checkpoint,
            })
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
        build_implementer(ctx, revised.clone())
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

    let red_team: RedTeamOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "red_team")
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
    // the wall just like the legacy loop; it never earns a retry of the extra budget.
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
/// `scout()` and `memory()` remain for a script that composes them itself. This is the one that
/// guarantees the ranked memory search happened.
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
    let rendered_question =
        crate::context::render_prompt(&input.issue, &input.memory, input.searchable);
    let stage = executor
        .stages
        .iter()
        .find(|stage| stage.id == "context_distillation")
        .cloned()
        .ok_or_else(|| "standard stage `context_distillation` is not registered".to_string())?;
    let raw = executor
        .execute_after_guard(StageInvocation {
            stage,
            input_json,
            rendered_question: Some(rendered_question),
            resource_root: None,
            shell: None,
            publish: None,
            clarifier: None,
            invocation_guidance: None,
            output: StageOutput::Evidence,
        })
        .await?;
    let distilled: crate::context::Distillation =
        serde_json::from_str(&raw).map_err(|error| error.to_string())?;
    let out = crate::context::attach_evidence(distilled, memory);
    note(&ctx, "context", &out, Some(arg)).await?;
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
    })
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
            .pool_for(&governance_id, self.ctx.rag_rat.clone());
        if ratatoskr_core::Capability::ceiling(&stage.capabilities)
            .is_some_and(|ceiling| ceiling.permits(ratatoskr_core::Capability::Write))
        {
            offered
                .local()
                .tools
                .extend(ratatoskr_agent::files::edit_declarations());
        }
        if stage
            .tools
            .iter()
            .any(|tool| tool == ratatoskr_agent::shell::BASH)
        {
            offered
                .local()
                .tools
                .push(ratatoskr_agent::shell::declaration());
        }
        if stage
            .tools
            .iter()
            .any(|tool| tool == ratatoskr_agent::ASK_TOOL_NAME)
        {
            offered.local().tools.push(crate::clarify::ask_tool());
        }
        if publish.is_some()
            && stage
                .tools
                .iter()
                .any(|tool| tool == ratatoskr_agent::publish::GH)
        {
            offered
                .local()
                .tools
                .push(ratatoskr_agent::publish::declaration());
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
            offered
                .local()
                .tools
                .push(ratatoskr_agent::publish::push_declaration());
        }
        let (mut cfg, profile) = crate::plugins::declared_stage_agent_config(
            &self.ctx.engine,
            &self.ctx.config,
            offered,
            &stage,
            &default_tools,
            &plugins,
        )
        .map_err(|e| e.to_string())?;
        cfg.route.session = stage.session_scope(cfg.route.session);

        // A child is evidence within its parent's call, never a second checkpointed graph stage.
        let runtime_input = if let Some(delegation) = stage
            .delegation
            .as_ref()
            .filter(|_| disposition == StageOutput::Checkpoint)
        {
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
        let conversation = format!("{}-{governance_id}", self.ctx.run_id);
        let raw = self
            .turn
            .run(ratatoskr_agent::NodeRun {
                node: &governance_id,
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
        if disposition == StageOutput::Checkpoint {
            note(&self.ctx, &stage.id, &output, Some(input_json)).await?;
        }
        serde_json::to_string(&output).map_err(|e| e.to_string())
    }
}

/// Compatibility adapter for scripted operations that still own Rust-only workflow state or
/// gates. Migrating one means replacing this entry with a standard declarative stage.
#[derive(Clone, Copy)]
enum TemporaryOperation {
    Context,
    Memory,
    RedTeam,
    Implement,
    Iterate,
    ReplanAtCeiling,
    Verify,
    IsConverged,
    TestCommandRan,
}

impl TemporaryOperation {
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
            Self::Memory => binding(Arc::clone(ctx), memory_host),
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

const TEMPORARY_OPERATIONS: &[(&str, TemporaryOperation)] = &[
    ("context", TemporaryOperation::Context),
    ("memory", TemporaryOperation::Memory),
    ("redTeam", TemporaryOperation::RedTeam),
    ("implement", TemporaryOperation::Implement),
    ("iterate", TemporaryOperation::Iterate),
    ("replanAtCeiling", TemporaryOperation::ReplanAtCeiling),
    ("verify", TemporaryOperation::Verify),
    ("isConverged", TemporaryOperation::IsConverged),
    ("testCommandRan", TemporaryOperation::TestCommandRan),
];

fn build_legacy_operation_hosts(
    ctx: &Arc<WorkflowContext>,
    executor: &Arc<StageExecutor>,
) -> HashMap<String, HostFn> {
    TEMPORARY_OPERATIONS
        .iter()
        .map(|(name, operation)| ((*name).to_string(), operation.host(ctx, executor)))
        .collect()
}

// These model turns need write authority only inside Rust-owned lifecycle adapters that provide
// the prepared worktree and, for implementation, the sandbox and clarification rendezvous. The
// declarations stay in `StageExecutor` for those adapters, but must never become repository-JS
// globals: a generic host has no worktree lifecycle from which to derive a safe resource root.
const INTERNAL_WRITE_STAGE_IDS: &[&str] = &["redteam_author", "implementer_attempt"];

fn build_declared_stage_hosts(executor: &Arc<StageExecutor>) -> HashMap<String, HostFn> {
    executor
        .stages
        .iter()
        .filter(|stage| !INTERNAL_WRITE_STAGE_IDS.contains(&stage.id.as_str()))
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
    let mut hosts = build_legacy_operation_hosts(ctx, &executor);
    if let Some(stage) = stages.iter().find(|stage| hosts.contains_key(&stage.id)) {
        return Err(PlanError::Configuration(format!(
            "stage `{}` conflicts with a legacy workflow operation",
            stage.id
        )));
    }
    hosts.extend(declared);
    Ok(hosts)
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
    WorkflowRuntime::bundled_with_includes(
        STANDARD_WORKFLOW_NAME,
        STANDARD_WORKFLOW_V1,
        STANDARD_WORKFLOW_INCLUDES,
    )
    .await
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
    rendered_question: String,
) -> Result<String, String> {
    evaluate_standard_stage_with_turn(
        ctx,
        stage_id,
        input_json,
        rendered_question,
        Arc::new(LiveStageTurn),
    )
    .await
}

/// Evaluate a bundled standard stage with its file tools rooted at a Rust-owned resource.
///
/// The caller retains ownership of that resource's lifecycle. In particular, the red-team author
/// may write into the implementer's pre-change worktree without giving a declared stage authority
/// to create, select, retain, or remove worktrees.
pub(crate) async fn evaluate_standard_stage_at(
    ctx: Arc<WorkflowContext>,
    stage_id: &str,
    input_json: String,
    rendered_question: String,
    resource_root: std::path::PathBuf,
) -> Result<String, String> {
    evaluate_standard_stage_with_resources(
        ctx,
        stage_id,
        input_json,
        rendered_question,
        StandardStageResources {
            resource_root,
            shell: None,
            publish: None,
            clarifier: None,
            guidance: None,
        },
    )
    .await
}

/// Rust-owned resources granted to one bundled evidence turn.
///
/// A stage may use these resources but cannot create, replace, retain, or clean them up. That
/// keeps worktree and sandbox lifecycle in the operation adapter while the model call remains a
/// generic declared-stage execution.
pub(crate) struct StandardStageResources {
    pub resource_root: PathBuf,
    pub shell: Option<ratatoskr_agent::shell::ShellAccess>,
    pub publish: Option<StandardStagePublishResources>,
    pub clarifier: Option<Arc<dyn ratatoskr_agent::Clarifier>>,
    pub guidance: Option<String>,
}

/// External publish authority granted by Rust to one terminal stage invocation.
///
/// Merely declaring `gh` or `git_push` never installs their host implementations. This grant is
/// what keeps a repository workflow from publishing before Rust has accepted a terminal outcome.
pub(crate) struct StandardStagePublishResources {
    pub push: Option<ratatoskr_agent::publish::PushAccess>,
}

pub(crate) async fn evaluate_standard_stage_with_resources(
    ctx: Arc<WorkflowContext>,
    stage_id: &str,
    input_json: String,
    rendered_question: String,
    resources: StandardStageResources,
) -> Result<String, String> {
    evaluate_standard_stage_with_resources_and_turn(
        ctx,
        stage_id,
        input_json,
        rendered_question,
        resources,
        Arc::new(LiveStageTurn),
    )
    .await
}

pub(crate) async fn evaluate_standard_stage_with_turn(
    ctx: Arc<WorkflowContext>,
    stage_id: &str,
    input_json: String,
    rendered_question: String,
    turn: Arc<dyn StageTurn>,
) -> Result<String, String> {
    evaluate_standard_stage_with_turn_and_resources(
        ctx,
        stage_id,
        input_json,
        rendered_question,
        None,
        turn,
    )
    .await
}

pub(crate) async fn evaluate_standard_stage_with_resources_and_turn(
    ctx: Arc<WorkflowContext>,
    stage_id: &str,
    input_json: String,
    rendered_question: String,
    resources: StandardStageResources,
    turn: Arc<dyn StageTurn>,
) -> Result<String, String> {
    evaluate_standard_stage_with_turn_and_resources(
        ctx,
        stage_id,
        input_json,
        rendered_question,
        Some(resources),
        turn,
    )
    .await
}

async fn evaluate_standard_stage_with_turn_and_resources(
    ctx: Arc<WorkflowContext>,
    stage_id: &str,
    input_json: String,
    rendered_question: String,
    resources: Option<StandardStageResources>,
    turn: Arc<dyn StageTurn>,
) -> Result<String, String> {
    let (resource_root, shell, publish, clarifier, invocation_guidance) = match resources {
        Some(resources) => (
            Some(resources.resource_root),
            resources.shell,
            resources.publish,
            resources.clarifier,
            resources.guidance,
        ),
        None => (None, None, None, None, None),
    };
    let stages = Arc::new(standard_stages().await.map_err(|error| error.to_string())?);
    let stage = stages
        .iter()
        .find(|stage| stage.id == stage_id)
        .cloned()
        .ok_or_else(|| format!("standard stage `{stage_id}` is not registered"))?;
    StageExecutor::new(ctx, stages, turn)
        .execute(StageInvocation {
            stage,
            input_json,
            rendered_question: Some(rendered_question),
            resource_root,
            shell,
            publish,
            clarifier,
            invocation_guidance,
            output: StageOutput::Evidence,
        })
        .await
}

async fn execution_stages(runtime: &WorkflowRuntime) -> Result<Vec<Stage>, PlanError> {
    let mut stages = standard_stages().await?;
    // Delivery is terminal external I/O, not a workflow operation. These declarations are
    // executed only by their Rust terminal adapters, after Rust has accepted the run outcome.
    stages.retain(|stage| !matches!(stage.id.as_str(), "bookkeeper" | "publisher"));
    // The bundled runtime declares the base stages themselves. Repository workflows add only
    // their own declarations; appending standard-v1 to itself would duplicate every model host and
    // reintroduce the terminal declarations filtered above.
    if runtime.meta().name != STANDARD_WORKFLOW_NAME {
        stages.extend(crate::stage::stages_from_workflow(runtime.meta()));
    }
    Ok(stages)
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
    // A scripted run is measured the same way a built-in one is; the script picks the order, not
    // whether the run is comparable to another afterwards.
    crate::record_provenance(&ctx.store, &ctx.run_id, &ctx.config).await;
    checkpoint(
        &ctx.store,
        &ctx.run_id,
        "issue",
        &json!({ "issue": ctx.issue }),
    )
    .await?;

    let stages = execution_stages(&runtime).await?;
    let hosts = build_hosts_with_turn(&ctx, &stages, turn)?;
    let input = json!({ "issue": ctx.issue }).to_string();
    let result = runtime
        .run_with_question_renderers("plan", input, hosts, stage_question_renderers(&stages))
        .await;

    let outcome = match result {
        Ok(_) => reconstruct_plan(&ctx.store, &ctx.run_id).await,
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
    ctx.plugin_context.session_end(status.as_str()).await;
    outcome
}

/// Scripted `run`: the full flow via the script's `run(input)` entry. Rust infers the terminal
/// status from checkpoints and does the bookkeeping — the script only sequences.
pub async fn run_full_scripted(
    runtime: WorkflowRuntime,
    ctx: Arc<WorkflowContext>,
) -> Result<RunOutcome, PlanError> {
    run_full_scripted_with_actions(runtime, ctx, &LiveTerminalActions).await
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
    // A scripted run is measured the same way a built-in one is; the script picks the order, not
    // whether the run is comparable to another afterwards.
    crate::record_provenance(&ctx.store, &ctx.run_id, &ctx.config).await;
    checkpoint(
        &ctx.store,
        &ctx.run_id,
        "issue",
        &json!({ "issue": ctx.issue }),
    )
    .await?;

    let stages = execution_stages(&runtime).await?;
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
    ) -> Option<PublisherOutput>;

    async fn bookkeep(
        &self,
        ctx: &Arc<WorkflowContext>,
        input: BookkeeperInput,
    ) -> Option<BookkeeperOutput>;
}

struct LiveTerminalActions;

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
    ) -> Option<PublisherOutput> {
        let run = crate::Run {
            client: None,
            config: &ctx.config,
            store: &ctx.store,
            run_id: &ctx.run_id,
            issue: &ctx.issue,
            engine: &ctx.engine,
            clarifier: &ctx.clarifier,
            context: &ctx.plugin_context,
            ledger: &ctx.ledger,
        };
        crate::publish_if_enabled(&run, input, terminal).await
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
        if let Err(error) = ctx
            .store
            .upsert_run(&ctx.run_id, None, status.as_str())
            .await
        {
            tracing::warn!("failed to record the run's final status: {error}");
        }
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
            )
            .await;
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

    let red_team: RedTeamOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "red_team").await?;
    let implementer: ImplementerOutput =
        latest_checkpoint(&ctx.store, &ctx.run_id, "implementer").await?;
    let iterations = count_checkpoints(&ctx.store, &ctx.run_id, "implementer").await?;
    let worktree = WorktreePath(PathBuf::from(&implementer.worktree_path));

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
    if let Err(error) = ctx
        .store
        .upsert_run(&ctx.run_id, None, status.as_str())
        .await
    {
        tracing::warn!("failed to record final run status: {error}");
    }

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
        )
    );

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
            bookkeeper::render_prompt(&input),
            StandardStageResources {
                resource_root: ctx.repo_path.clone(),
                shell: None,
                publish: None,
                clarifier: None,
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
        published: Option<PublisherOutput>,
        bookkeeper: Option<BookkeeperOutput>,
    }

    impl RecordingTerminalActions {
        fn new(publish: bool, bookkeep: bool) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                published: publish.then(|| PublisherOutput {
                    action: "comment".to_string(),
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

        fn calls(&self) -> Vec<TerminalCall> {
            self.calls
                .lock()
                .expect("terminal calls mutex poisoned")
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
            _ctx: &Arc<WorkflowContext>,
            input: PublisherInput,
            terminal: bool,
        ) -> Option<PublisherOutput> {
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
            _ctx: &Arc<WorkflowContext>,
            input: BookkeeperInput,
        ) -> Option<BookkeeperOutput> {
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
        let mut stage = crate::built_in_stages().pop().unwrap();
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
        let mut stage = crate::built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();
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
        nodes: Mutex<Vec<String>>,
        conversations: Mutex<Vec<Option<String>>>,
        ledger_ids: Mutex<Vec<Option<usize>>>,
        tools: Mutex<Vec<Vec<String>>>,
        files: Mutex<Vec<Option<std::path::PathBuf>>>,
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
                nodes: Mutex::new(Vec::new()),
                conversations: Mutex::new(Vec::new()),
                ledger_ids: Mutex::new(Vec::new()),
                tools: Mutex::new(Vec::new()),
                files: Mutex::new(Vec::new()),
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
            self.nodes
                .lock()
                .expect("recording runner mutex poisoned")
                .push(run.node.to_string());
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
            ["issue", "context", "analyst"]
        );
        let context: crate::ContextOutput =
            serde_json::from_str(&checkpoints[1].output_json).unwrap();
        let analyst_input: analyst::AnalystInput =
            serde_json::from_str(checkpoints[2].input_json.as_deref().unwrap()).unwrap();
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
            ["context", "analyst"]
        );
        assert_eq!(runs[1].session, ratatoskr_core::SessionScope::Compacted);
        assert!(runs[0].ledger_id.is_some());
        assert_eq!(runs[0].ledger_id, runs[1].ledger_id);
        let context_input = crate::context::distillation_input(
            "preserve the declared plan path",
            crate::MemoryOutput::default(),
            false,
        );
        let expected_context = crate::context::render_prompt(
            &context_input.issue,
            &context_input.memory,
            context_input.searchable,
        );
        assert!(runs[0].question.starts_with(
            "Input contract: ContextDistillationInput\nOutput contract: Distillation\n\n"
        ));
        assert!(runs[0].question.ends_with(&expected_context));
        assert!(
            runs[1]
                .question
                .starts_with("Input contract: AnalystInput\nOutput contract: AnalystOutput\n\n")
        );
        assert!(
            runs[1]
                .question
                .ends_with(&analyst::render_prompt(&analyst_input))
        );
        drop(runs);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn bundled_standard_plan_cannot_succeed_without_required_checkpoints() {
        let runtime = WorkflowRuntime::bundled_with_includes(
            "incomplete-standard-plan",
            r#"defineWorkflow({ name: "incomplete-standard-plan" });
               async function plan(input) { return input; }"#,
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
        assert!(
            matches!(error, PlanError::MissingCheckpoint(_, "scout")),
            "missing context evidence must fail reconstruction: {error}"
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
        let stages = execution_stages(&runtime).await.unwrap();
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
                    note(&ctx, "red_team", &output, None).await?;
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
                "red_team",
                "implementer",
                "verifier",
                "analyst",
                "implementer",
                "verifier",
            ]
        );
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
        note(&ctx, "red_team", &red(&[], &["baseline"], 0), None)
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
        let stages = execution_stages(&runtime).await.unwrap();
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
        let implementer = build_implementer(&ctx, analyst).unwrap();
        let clarifier = implementer
            .clarifier
            .expect("the native implementer receives the run clarifier");
        let _ = clarifier
            .answer(
                "implementer",
                "analyst",
                "Which invariant controls this change?",
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
        ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
            Box::pin(async { "static answer".to_string() })
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
        let mut stage = crate::built_in_stages().pop().unwrap();
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
        let mut stage = crate::built_in_stages().pop().unwrap();
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
        let mut stage = crate::built_in_stages().pop().unwrap();
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
    async fn temporary_operation_adapters_remain_registered_and_guarded() {
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

        for (name, _) in TEMPORARY_OPERATIONS {
            assert!(
                hosts.contains_key(*name),
                "missing operation adapter `{name}`"
            );
        }
        assert!(
            hosts.contains_key("memory"),
            "repository workflows still use the deterministic memory adapter"
        );
        for removed in ["red_team", "implementer", "newlyIntroducedFailures"] {
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
            "legacy alias changed: {error}"
        );

        ctx.invocations.store(INVOCATION_CEILING, Ordering::Relaxed);
        let error = hosts["memory"]("{}".to_string()).await.unwrap_err();
        assert!(error.contains("runaway loop"));

        let context = crate::built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "context")
            .unwrap();
        let error = match build_hosts_with_turn(
            &ctx,
            &[context],
            Arc::new(RecordingStageTurn::default()),
        ) {
            Ok(_) => panic!("a declared stage replaced a temporary operation adapter"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("legacy workflow operation"));
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
               async function plan(input) {
                 return input.target === "publisher"
                   ? await publisher(input)
                   : await bookkeeper(input);
               }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
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
        let stages = execution_stages(&runtime).await.unwrap();
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
               async function plan(input) {
                 return input.target === "redteam_author"
                   ? await redteam_author(input)
                   : await implementer_attempt(input);
               }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
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
        let stages = execution_stages(&runtime).await.unwrap();
        for stage_id in INTERNAL_WRITE_STAGE_IDS {
            let stage = stages
                .iter()
                .find(|stage| stage.id == *stage_id)
                .expect("internal stage remains available to Rust adapters");
            assert_eq!(stage.capabilities, [ratatoskr_core::Capability::Write]);
        }
        let turn = Arc::new(RecordingStageTurn::default());
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        assert!(hosts.contains_key("redTeam"));
        assert!(hosts.contains_key("implement"));
        for stage_id in INTERNAL_WRITE_STAGE_IDS {
            assert!(!hosts.contains_key(*stage_id));
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
            r#"async function plan(input) {
                 await analyst(input.fresh);
                 return await analyst(input.revision);
               }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
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
        }
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
            "async function run(input) { return await bookkeeper(input); }",
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
            .await
            .unwrap()
            .unwrap();
        let stages = standard_stages().await.unwrap();
        assert!(
            execution_stages(&runtime)
                .await
                .unwrap()
                .iter()
                .all(|stage| stage.id != "bookkeeper"),
            "repository workflows must not manufacture terminal bookkeeping"
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
        let expected_question = crate::bookkeeper::render_prompt(&input);
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
        assert_eq!(
            envelope["__ratatoskrRenderedQuestion"]["question"],
            expected_question
        );
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
            "async function run(input) { return await publisher(input); }",
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
            .await
            .unwrap()
            .unwrap();
        let stages = standard_stages().await.unwrap();
        assert!(
            execution_stages(&runtime)
                .await
                .unwrap()
                .iter()
                .all(|stage| stage.id != "publisher"),
            "repository workflows must not acquire terminal publish authority"
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
        let expected_question = crate::publisher::render_prompt(&input);
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
        assert_eq!(
            envelope["__ratatoskrRenderedQuestion"]["question"],
            expected_question
        );
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
            r#"async function plan(input) { return await context_distillation(input); }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
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
        let expected_prompt =
            crate::context::render_prompt(&input.issue, &input.memory, input.searchable);
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
            ["context"]
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
        assert_eq!(
            turn.questions
                .lock()
                .expect("recording runner mutex poisoned")[0],
            format!(
                "Input contract: ContextDistillationInput\nOutput contract: Distillation\n\n{expected_prompt}"
            )
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
            r#"async function plan(input) { return await context(input.issue); }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
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
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].node_name, "context");
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

        let configured = build_red_team(&ctx, Vec::new()).unwrap();
        let author_tools = configured.author.as_ref().unwrap().tools.names();
        assert!(author_tools.iter().any(|tool| tool == "Write"));
        assert!(author_tools.iter().any(|tool| tool == "Edit"));
        let classifier_tools = configured.classifier.as_ref().unwrap().tools.names();
        assert!(
            !classifier_tools
                .iter()
                .any(|tool| tool == "Write" || tool == "Edit")
        );

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
        let scripted = latest_checkpoint::<RedTeamOutput>(&store, run_id, "red_team")
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(&returned).unwrap(),
            serde_json::to_value(&scripted).unwrap()
        );

        // Both flows use `run_and_author`: the scripted checkpoint must carry the same
        // deterministic evidence as a built-in invocation on the retained pre-change tree.
        let built_in = build_red_team(&ctx, Vec::new())
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
            r#"async function plan(input) {
                return await redteam_classifier(input.classifier);
            }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
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
            crate::redteam::author_prompt(&author.issue, &author.interface),
            StandardStageResources {
                resource_root: author_root.clone(),
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
            ["redteam", "redteam"]
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
            Some(author_root)
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
        assert_eq!(
            questions[0],
            format!(
                "Input contract: TestAuthorInput\nOutput contract: AuthoredTests\n\n{}",
                crate::redteam::author_prompt(&author.issue, &author.interface)
            )
        );
        assert_eq!(
            questions[1],
            format!(
                "Input contract: ClassifierInput\nOutput contract: Classification\n\n{}",
                crate::redteam::classifier_prompt(&classifier.failing, &classifier.raw_output)
            )
        );
        let checkpoints = store
            .checkpoints_for_run("run-standard-redteam")
            .await
            .unwrap();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].node_name, "redteam_classifier");
        let checkpoint_classifier: crate::redteam::ClassifierInput =
            serde_json::from_str(checkpoints[0].input_json.as_deref().unwrap()).unwrap();
        assert_eq!(
            checkpoint_classifier.raw_output,
            "assertion failed: deleted > 0"
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
            crate::redteam::author_prompt(&author.issue, &author.interface),
            StandardStageResources {
                resource_root: author_root.clone(),
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
            crate::redteam::classifier_prompt(&classifier.failing, &classifier.raw_output),
            invalid_turn,
        )
        .await
        .unwrap_err();
        assert!(error.contains("invalid `Classification` output"), "{error}");
        assert!(error.contains("test"), "{error}");
        assert!(
            store
                .checkpoints_for_run("run-redteam-evidence")
                .await
                .unwrap()
                .is_empty()
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
                crate::implementer::render_attempt_prompt(input),
                StandardStageResources {
                    resource_root: dir.clone(),
                    shell: None,
                    publish: None,
                    clarifier: None,
                    guidance: None,
                },
                Arc::clone(&turn) as Arc<dyn StageTurn>,
            )
            .await
            .unwrap();
        }

        assert_eq!(
            *turn.nodes.lock().expect("recording runner mutex poisoned"),
            ["implementer", "implementer"]
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
                Some("run-standard-implementer-implementer".to_string()),
                Some("run-standard-implementer-implementer".to_string())
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
        {
            let questions = turn
                .questions
                .lock()
                .expect("recording runner mutex poisoned");
            let initial = implementer_attempt_input(None);
            assert_eq!(
                questions[0],
                format!(
                    "Input contract: ImplementerAttemptInput\nOutput contract: Report\n\n{}",
                    crate::implementer::render_attempt_prompt(&initial)
                )
            );
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
            for expected in ["Read", "Grep", "Glob", "Write", "Edit", "Bash", "ask"] {
                assert!(
                    offered.iter().any(|tool| tool == expected),
                    "missing {expected}"
                );
            }
        }
        let checkpoints = store
            .checkpoints_for_run("run-standard-implementer")
            .await
            .unwrap();
        assert!(
            checkpoints.is_empty(),
            "internal evidence turns do not own workflow checkpoints"
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
            crate::implementer::render_attempt_prompt(&input),
            StandardStageResources {
                resource_root: worktree.clone(),
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
            crate::implementer::render_attempt_prompt(&input),
            StandardStageResources {
                resource_root: dir.clone(),
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
        assert!(
            store
                .checkpoints_for_run("run-implementer-resources")
                .await
                .unwrap()
                .is_empty(),
            "evidence invocations never own workflow checkpoints"
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
            r#"async function plan(input) { return await characterizer(input); }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
            .await
            .unwrap()
            .unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
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
        let expected_prompt = crate::testrun::render_prompt(&input.outcomes);
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
        assert_eq!(
            turn.questions
                .lock()
                .expect("recording runner mutex poisoned")[0],
            format!(
                "Input contract: CharacterizerInput\nOutput contract: CharacterizerOutput\n\n{expected_prompt}"
            )
        );

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
        let question = crate::testrun::render_prompt(&input.outcomes);
        let turn = Arc::new(RecordingStageTurn {
            output: json!({ "failing": ["suite::one"], "passed": 3 }).to_string(),
            ..Default::default()
        });
        let output = evaluate_standard_stage_with_turn(
            ctx,
            "characterizer",
            serde_json::to_string(&input).unwrap(),
            question,
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
            r#"async function plan(input) { return await overseer(input); }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
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
        let expected_prompt = crate::overseer::render_prompt(&input.issue, &input.choices);
        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "workflow": "research",
                "reasoning": "The task asks to explain the registry."
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
        assert_eq!(
            question,
            format!(
                "Input contract: OverseerInput\nOutput contract: OverseerOutput\n\n{expected_prompt}"
            )
        );

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
        let mut hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
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
        let error = hosts.remove("overseer").unwrap()(envelope)
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
            r#"async function plan(input) { return await verifier(input); }"#,
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
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
        let expected_prompt = verifier::render_prompt(&input);
        let turn = Arc::new(RecordingStageTurn {
            output: json!({
                "findings": [],
                "assessment": "checked the session key and its callers"
            })
            .to_string(),
            ..Default::default()
        });
        let hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
        runtime
            .run_with_question_renderers(
                "plan",
                input_value.to_string(),
                hosts,
                stage_question_renderers(&stages),
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
        assert_eq!(
            question,
            format!(
                "Input contract: VerifierInput\nOutput contract: VerifierOutput\n\n{expected_prompt}"
            )
        );
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
        let mut hosts =
            build_hosts_with_turn(&ctx, &stages, Arc::clone(&turn) as Arc<dyn StageTurn>).unwrap();
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
        let error = hosts.remove("verifier").unwrap()(envelope)
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
    async fn scripted_terminal_commits_before_delivery_and_bookkeeps_unreviewed_work() {
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
        checkpoint(&store, run_id, "red_team", &red(&["old"], &[], 1))
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
        let actions = RecordingTerminalActions::new(true, true);

        let outcome = finish_full(&ctx, &actions).await.unwrap();

        assert_eq!(outcome.status, RunStatus::Unreviewed);
        assert!(outcome.bookkeeper.is_some());
        assert_eq!(outcome.state.artifacts.len(), 2);
        assert_eq!(
            store.run_status(run_id).await.unwrap().as_deref(),
            Some(RunStatus::Unreviewed.as_str())
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
            "async function run() { throw new Error('terminal failure'); }",
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&workflow_path)
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
}
