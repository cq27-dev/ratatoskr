//! Scriptable orchestration: run `.ratatoskr/workflow.ts` (issue #18) instead of the hardcoded
//! `run_plan`/`run_full` flow. The script composes host bindings — one per node call site — but
//! every gate stays Rust-enforced: schema validation and checkpointing happen inside each binding,
//! the false-convergence guard lives in `redTeam`, `max_iterations` is capped in `iterate`, and the
//! terminal status is inferred from checkpoints after the script returns (never trusted from the
//! script). A missing `workflow.ts` runs the built-in Rust flow unchanged (see [`super::run_full`]).
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
    AnalystNode, AnalystOutput, BookkeeperInput, BookkeeperNode, BookkeeperOutput, ChildTask,
    ImplementerNode, ImplementerOutput, MemoryNode, MemoryOutput, PlanError, PlanOutcome,
    RedTeamNode, RedTeamOutput, RunOutcome, ScoutOutput, Stage, analyst, bookkeeper, checkpoint,
    converge, memory, redteam, referee, stage_agent_config, verifier,
};

/// Backstop on total node-running binding calls per run — a runaway-loop guard, far above any real
/// workflow. `max_iterations` and the false-convergence guard are the precise limits; this only
/// catches a script that ignores them and loops.
const INVOCATION_CEILING: usize = 500;

const STANDARD_WORKFLOW_V1: &str = include_str!("../workflows/standard-v1.ts");

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
    /// Set by `implement`, read by `iterate` and cleanup. The script never sees a raw path.
    worktree: Mutex<Option<WorktreePath>>,
    implement_started: AtomicBool,
    /// Serializes `iterate` calls — two implementers editing one worktree would corrupt it.
    iterate_lock: tokio::sync::Mutex<()>,
    invocations: AtomicUsize,
    iterations: AtomicU32,
    /// What plugins contributed for this run, prefixed to each node's preamble.
    plugin_context: crate::PluginContext,
    /// Where this run's nodes report what their turns cost. A scripted run records the same
    /// telemetry as a built-in one — the script chooses the order, not what gets measured.
    ledger: Arc<ratatoskr_agent::RunLedger>,
    /// The acceptance this run is judged by, resolved once and reused.
    ///
    /// The built-in flow resolves it before the fork and freezes it for the same reason it matters
    /// more here: a script can re-analyse between iterations, and if each binding resolved its own
    /// the plan could move the bar it is judged against mid-run. Whichever binding runs acceptance
    /// first decides it; everything after gets that.
    acceptance: Mutex<Option<Vec<ratatoskr_core::AcceptanceStep>>>,
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
        let repo_path = std::env::current_dir()
            .map_err(|e| PlanError::node("workflow", NodeError::Failed(format!("cwd: {e}"))))?;
        Ok(Arc::new(Self {
            ledger: Arc::new(ratatoskr_agent::RunLedger::default()),
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
            implement_started: AtomicBool::new(false),
            iterate_lock: tokio::sync::Mutex::new(()),
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

async fn analyze_host(ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    ctx.guard()?;
    let input: analyst::AnalystInput =
        serde_json::from_str(&arg).map_err(|e| format!("analyze arg: {e}"))?;
    let mut plugins = ctx.plugin_context.for_node("analyst");
    let cfg = stage_agent_config(
        &ctx.engine,
        &ctx.config,
        ctx.plugin_context.pool_for("analyst", ctx.rag_rat.clone()),
        "analyst",
        analyst::ANALYST_TOOLS,
        &mut plugins,
    )
    .map_err(|e| e.to_string())?;
    let node = AnalystNode {
        // A revision continues the plan it revises, when the route asks for that.
        conversation: Some(format!("{}-analyst", ctx.run_id)),
        route: cfg.route,
        tools: cfg.tools,
        files: cfg.files,
        ledger: Some(Arc::clone(&ctx.ledger)),
        policy: cfg.policy,
        max_turns: cfg.max_turns,
        system_prompt: cfg.system_prompt,
        plugins,
    };
    let out = node
        .run(input, &RunState::new(&ctx.run_id, None))
        .await
        .map_err(|e| e.to_string())?;
    note(&ctx, "analyst", &out, Some(arg)).await?;
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

/// `acceptance` is passed in rather than read here because the baseline and the post-change run
/// must execute the same steps — a red team that resolved its own would drift from the implementer
/// the moment a plan proposed anything.
fn build_red_team(
    ctx: &WorkflowContext,
    acceptance: Vec<ratatoskr_core::AcceptanceStep>,
) -> Result<RedTeamNode, PlanError> {
    let short: String = ctx.run_id.chars().take(8).collect();
    let classifier = match crate::classifier_enabled(&ctx.engine, &ctx.config) {
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
            })
        }
        false => None,
    };
    Ok(RedTeamNode {
        // The scripted path forks with its own sequencing and does not create the worktree before
        // red-team runs, so there is nothing to write tests into. Authoring belongs to the built-in
        // flow until a script can say where the tree is.
        author: None,
        acceptance,
        characterizer: crate::build_characterizer(
            &ctx.engine,
            &ctx.config,
            &ctx.plugin_context,
            ctx.rag_rat.clone(),
            Some(Arc::clone(&ctx.ledger)),
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
    // A script may call `redTeam()` before or after `analyst()`. When a plan exists, its acceptance
    // is what the baseline is measured with; otherwise the configured command stands in.
    let planned = latest_checkpoint::<AnalystOutput>(&ctx.store, &ctx.run_id, "analyst")
        .await
        .map(|a| a.acceptance)
        .unwrap_or_default();
    let acceptance = ctx.acceptance(&planned);
    let node = build_red_team(&ctx, acceptance).map_err(|e| e.to_string())?;
    let out = node.run().await.map_err(|e| e.to_string())?;
    // Checkpoint before the guard so a failed baseline stays inspectable.
    note(&ctx, "red_team", &out, None).await?;
    // The false-convergence guard is enforced here — the script cannot skip it.
    if !converge::test_command_ran(&out.failing_tests, out.passed_tests, out.exit_code) {
        return Err(format!(
            "the baseline acceptance run produced no checks (exit {}); check the analyst's acceptance, [sandbox] test_command and the sandbox backend",
            out.exit_code
        ));
    }
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

fn build_implementer(
    ctx: &WorkflowContext,
    analyst: AnalystOutput,
) -> Result<ImplementerNode, PlanError> {
    let (cfg, plugins) = crate::build_implementer_agent(
        &ctx.engine,
        &ctx.config,
        &ctx.plugin_context,
        ctx.rag_rat.clone(),
    )?;
    Ok(ImplementerNode {
        // As every node on the scripted path: clarification is wired by the built-in flow, which
        // owns the run's `NodeClarifier`.
        clarifier: None,
        acceptance: ctx.acceptance(&analyst.acceptance),
        characterizer: crate::build_characterizer(
            &ctx.engine,
            &ctx.config,
            &ctx.plugin_context,
            ctx.rag_rat.clone(),
            Some(Arc::clone(&ctx.ledger)),
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
    let input: ImplementArg =
        serde_json::from_str(&arg).map_err(|e| format!("implement arg: {e}"))?;
    let node = build_implementer(&ctx, input.analyst).map_err(|e| e.to_string())?;
    let (worktree, out) = node.run().await.map_err(|e| e.to_string())?;
    *ctx.worktree.lock().unwrap() = Some(worktree);
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

async fn iterate_host(ctx: Arc<WorkflowContext>, _arg: String) -> Result<String, String> {
    ctx.guard()?;
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
    let diagnostic = if !referee.is_empty() {
        referee::correction(&referee)
    } else if !post_ran {
        format!(
            "The test command did not run to completion (exit {}) — your change likely does not \
             compile. Fix it so the tests run and pass.",
            prev.exit_code
        )
    } else {
        let new_failures =
            converge::newly_introduced_failures(&red_team.failing_tests, &prev.failing_tests);
        format!(
            "Your change introduced NEW failing tests not present in the baseline: {}. Fix them \
             without breaking other tests.",
            new_failures.join(", ")
        )
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

async fn newly_introduced_host(_ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    let bp: BaselinePost =
        serde_json::from_str(&arg).map_err(|e| format!("newlyIntroducedFailures arg: {e}"))?;
    let v = converge::newly_introduced_failures(&bp.baseline.failing_tests, &bp.post.failing_tests);
    serde_json::to_string(&v).map_err(|e| e.to_string())
}

/// What `verify()` hands the script.
///
/// `blocking` is the part that matters and the part the script does not get to compute: Rust reads
/// `[implementer] verify_threshold` and decides what clears it. A workflow chooses *whether* to
/// review and what to do about findings; it cannot decide that a P1 is not a P1.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
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
    let verifier_input = verifier::VerifierInput {
        // The scripted path reviews once per call and keeps no history of its own.
        previous_findings: Vec::new(),
        issue: ctx.issue.clone(),
        analyst: input.analyst,
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

    let threshold = crate::parse_threshold(&ctx.config.implementer.verify_threshold);
    serde_json::to_string(&verification_result(out, threshold)).map_err(|e| e.to_string())
}

/// `context(issue)` — the merged gather step: distilled findings plus the memories unmodified.
///
/// `scout()` and `memory()` remain for a script that composes them itself. This is the one that
/// guarantees the ranked memory search happened.
async fn context_host(ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    ctx.guard()?;
    let issue: String = serde_json::from_str(&arg).map_err(|e| format!("context arg: {e}"))?;
    let mut plugins = ctx.plugin_context.for_node("context");
    let cfg = stage_agent_config(
        &ctx.engine,
        &ctx.config,
        ctx.plugin_context.pool_for("context", ctx.rag_rat.clone()),
        "context",
        crate::context::CONTEXT_TOOLS,
        &mut plugins,
    )
    .map_err(|e| e.to_string())?;
    let node = crate::ContextNode {
        route: cfg.route,
        tools: cfg.tools,
        sink: ctx.sink.clone(),
        files: cfg.files,
        ledger: Some(Arc::clone(&ctx.ledger)),
        policy: cfg.policy,
        max_turns: cfg.max_turns,
        clarifier: None,
        system_prompt: cfg.system_prompt,
        plugins,
    };
    let out = node.run(&issue).await.map_err(|e| e.to_string())?;
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
    crate::effective_preamble(&stage.id, &base, None, None, skills)
}

fn declared_stage_question(stage: &Stage, runtime_question: &str) -> String {
    format!(
        "Input contract: {}\nOutput contract: {}\n\n{runtime_question}",
        stage.input_contract, stage.output_contract
    )
}

trait StageTurn: Send + Sync {
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
            output: disposition,
        } = invocation;
        let input: serde_json::Value =
            serde_json::from_str(&input_json).map_err(|e| format!("{} arg: {e}", stage.id))?;
        let plugins = self.ctx.plugin_context.for_node(&stage.id);
        let default_tools = stage.tools.iter().map(String::as_str).collect::<Vec<_>>();
        let (mut cfg, profile) = crate::plugins::declared_stage_agent_config(
            &self.ctx.engine,
            &self.ctx.config,
            self.ctx
                .plugin_context
                .pool_for(&stage.id, self.ctx.rag_rat.clone()),
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
        let preamble = declared_stage_preamble(
            &stage,
            &profile,
            cfg.system_prompt.as_deref(),
            &repository_guidance,
            plugins.context.as_deref(),
            &plugins.skills,
        );
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
        let conversation = format!("{}-{}", self.ctx.run_id, stage.id);
        let raw = self
            .turn
            .run(ratatoskr_agent::NodeRun {
                node: &stage.id,
                route: &cfg.route,
                preamble: &preamble,
                question: &question,
                tools: cfg.tools,
                output_schema: output_schema
                    .unwrap_or_else(|| schemars::schema_for!(serde_json::Value)),
                policy: cfg.policy,
                max_turns: cfg.max_turns,
                clarifier: None,
                observer: plugins.observer.clone(),
                skills: crate::skills::loaded(&plugins.skills, &stage.id),
                files: resource_root.or(cfg.files),
                shell: None,
                push: None,
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
    Analyze,
    RedTeam,
    Implement,
    Iterate,
    Verify,
    IsConverged,
    TestCommandRan,
    NewlyIntroducedFailures,
}

impl TemporaryOperation {
    fn host(self, ctx: &Arc<WorkflowContext>, executor: &Arc<StageExecutor>) -> HostFn {
        match self {
            Self::Context => binding(Arc::clone(ctx), context_host),
            Self::Memory => binding(Arc::clone(ctx), memory_host),
            Self::Analyze => binding(Arc::clone(ctx), analyze_host),
            Self::RedTeam => binding(Arc::clone(ctx), red_team_host),
            Self::Implement => binding(Arc::clone(ctx), implement_host),
            Self::Iterate => binding(Arc::clone(ctx), iterate_host),
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
            Self::NewlyIntroducedFailures => binding(Arc::clone(ctx), newly_introduced_host),
        }
    }
}

const TEMPORARY_OPERATIONS: &[(&str, TemporaryOperation)] = &[
    ("context", TemporaryOperation::Context),
    ("memory", TemporaryOperation::Memory),
    ("analyze", TemporaryOperation::Analyze),
    ("red_team", TemporaryOperation::RedTeam),
    ("redTeam", TemporaryOperation::RedTeam),
    ("implementer", TemporaryOperation::Implement),
    ("implement", TemporaryOperation::Implement),
    ("iterate", TemporaryOperation::Iterate),
    ("verify", TemporaryOperation::Verify),
    ("isConverged", TemporaryOperation::IsConverged),
    ("testCommandRan", TemporaryOperation::TestCommandRan),
    (
        "newlyIntroducedFailures",
        TemporaryOperation::NewlyIntroducedFailures,
    ),
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

fn build_declared_stage_hosts(executor: &Arc<StageExecutor>) -> HashMap<String, HostFn> {
    executor
        .stages
        .iter()
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
    let meta = WorkflowRuntime::bundled_meta("ratatoskr-standard-v1", STANDARD_WORKFLOW_V1)
        .await
        .map_err(|error| PlanError::node("workflow", NodeError::Failed(error.to_string())))?;
    let stages = crate::stage::stages_from_workflow(&meta);
    crate::validate::validate_declared_contracts(&stages)?;
    Ok(stages)
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
    let stages = Arc::new(standard_stages().await.map_err(|error| error.to_string())?);
    let stage = stages
        .iter()
        .find(|stage| stage.id == stage_id)
        .cloned()
        .ok_or_else(|| format!("standard stage `{stage_id}` is not registered"))?;
    StageExecutor::new(ctx, stages, Arc::new(LiveStageTurn))
        .execute(StageInvocation {
            stage,
            input_json,
            rendered_question: Some(rendered_question),
            resource_root: None,
            output: StageOutput::Evidence,
        })
        .await
}

async fn execution_stages(runtime: &WorkflowRuntime) -> Result<Vec<Stage>, PlanError> {
    let mut stages = standard_stages().await?;
    stages.extend(crate::stage::stages_from_workflow(runtime.meta()));
    Ok(stages)
}

// --- wrappers (own every status write, gate, and cleanup) -------------------

/// Scripted `plan`: scout → memory → analyst, composed by the script's `plan(input)` entry.
pub async fn run_plan_scripted(
    runtime: WorkflowRuntime,
    ctx: Arc<WorkflowContext>,
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
    let hosts = build_hosts(&ctx, &stages)?;
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
    let input =
        json!({ "issue": ctx.issue, "maxIterations": ctx.config.implementer.max_iterations })
            .to_string();

    // Run the script, then reconstruct the outcome. EITHER failing is a run failure: on any error
    // (a script/binding error, or a reconstruction error like a missing checkpoint) the worktree is
    // cleaned up and the run is marked `Failed` — never left orphaned or stuck at `Running`.
    let result = match runtime
        .run_with_question_renderers("run", input, hosts, stage_question_renderers(&stages))
        .await
    {
        Ok(_) => finish_full(&ctx).await,
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

/// Reconstruct the `RunOutcome` from the store after a successful script run, write the Rust-inferred
/// terminal status, and do the run-back bookkeeping. Any error here is handled by the caller's
/// cleanup path.
async fn finish_full(ctx: &Arc<WorkflowContext>) -> Result<RunOutcome, PlanError> {
    // The store is the source of truth the script can't fake; a missing checkpoint is a hard error.
    let plan = reconstruct_plan(&ctx.store, &ctx.run_id).await?;
    let red_team: RedTeamOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "red_team").await?;
    let implementer: ImplementerOutput =
        latest_checkpoint(&ctx.store, &ctx.run_id, "implementer").await?;
    let iterations = count_checkpoints(&ctx.store, &ctx.run_id, "implementer").await?;
    let worktree = WorktreePath(PathBuf::from(&implementer.worktree_path));

    // Terminal status is Rust-inferred, never trusted from the script.
    // The last review, if the script ran one. Absent is not the same as clean: a workflow that
    // never verified simply has no verifier checkpoint, and the warning at run start already said
    // the change would be accepted on its tests alone.
    let review: Option<verifier::VerifierOutput> =
        latest_checkpoint(&ctx.store, &ctx.run_id, "verifier")
            .await
            .ok();
    let referee = referee_judgement(ctx, &worktree, &plan.analyst, &implementer).await;
    let status = infer_status(
        &red_team,
        &implementer,
        &referee,
        review.as_ref(),
        crate::parse_threshold(&ctx.config.implementer.verify_threshold),
    );
    ctx.store
        .upsert_run(&ctx.run_id, None, status.as_str())
        .await?;

    let bookkeeper = if matches!(
        status,
        RunStatus::Converged | RunStatus::MaxIterationsReached
    ) {
        let input = BookkeeperInput {
            issue: ctx.issue.clone(),
            analyst: plan.analyst.clone(),
            implementer: implementer.clone(),
            iterations,
            converged: status == RunStatus::Converged,
            friction: crate::friction_of(&ctx.store, &ctx.run_id).await,
        };
        match bookkeep_scripted(ctx, input).await {
            Ok(bk) => Some(bk),
            Err(e) => {
                tracing::warn!("bookkeeping failed: {e}");
                None
            }
        }
    } else {
        None
    };

    let mut state = plan.state.clone();
    state.red_team = Some(serde_json::to_value(&red_team)?);
    state.implementer = Some(serde_json::to_value(&implementer)?);
    state.status = status;
    if let Some(bk) = &bookkeeper {
        state.artifacts = vec![serde_json::to_value(bk)?];
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
    ctx: &WorkflowContext,
    input: BookkeeperInput,
) -> Result<BookkeeperOutput, PlanError> {
    let mut plugins = ctx.plugin_context.for_node("bookkeeper");
    let cfg = stage_agent_config(
        &ctx.engine,
        &ctx.config,
        ctx.plugin_context
            .pool_for("bookkeeper", ctx.rag_rat.clone()),
        "bookkeeper",
        bookkeeper::BOOKKEEPER_TOOLS,
        &mut plugins,
    )?;
    let node = BookkeeperNode {
        route: cfg.route,
        tools: cfg.tools,
        files: cfg.files,
        ledger: Some(Arc::clone(&ctx.ledger)),
        sink: ctx.sink.clone(),
        policy: cfg.policy,
        max_turns: cfg.max_turns,
        clarifier: None,
        system_prompt: cfg.system_prompt,
        plugins,
    };
    let out = node
        .run(input)
        .await
        .map_err(|e| PlanError::node("bookkeeper", e))?;
    note(ctx, "bookkeeper", &out, None)
        .await
        .map_err(|e| PlanError::node("bookkeeper", NodeError::Failed(e)))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            !hosts.contains_key("analyst"),
            "the canonical analyst id belongs to the declarative stage"
        );
        assert!(
            !hosts.contains_key("verifier"),
            "the canonical verifier id belongs to the declarative stage"
        );
        let error = hosts["analyze"]("{}".to_string()).await.unwrap_err();
        assert!(
            error.contains("analyze arg"),
            "legacy alias changed: {error}"
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
        assert!(scout.output_schema.is_some());
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
