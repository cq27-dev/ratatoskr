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
use std::path::PathBuf;
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
    AnalystNode, AnalystOutput, BookkeeperInput, BookkeeperNode, BookkeeperOutput, ImplementerNode,
    ImplementerOutput, MemoryNode, MemoryOutput, PlanError, PlanOutcome, RedTeamNode,
    RedTeamOutput, RunOutcome, ScoutNode, ScoutOutput, analyst, bookkeeper, checkpoint, converge,
    memory, node_agent_config, redteam, scout, verifier,
};

/// Backstop on total node-running binding calls per run — a runaway-loop guard, far above any real
/// workflow. `max_iterations` and the false-convergence guard are the precise limits; this only
/// catches a script that ignores them and loops.
const INVOCATION_CEILING: usize = 500;

/// Everything the bindings need, cloned as an `Arc` into every host closure. Holds the run's shared
/// mutable state (the worktree handle and the invocation/iteration counters) behind atomics/a mutex.
pub struct WorkflowContext {
    config: RatatoskrConfig,
    store: Store,
    engine: Arc<ScriptEngine>,
    run_id: String,
    issue: String,
    sink: ServerSink,
    /// rag-rat's whole offer, the base of every node's tool pool.
    rag_rat: ServerTools,
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
        client: &RagRatClient,
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
            sink: client.sink(),
            rag_rat: client.offer(),
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

/// Terminal status, inferred from the baseline and the final implementer output — never trusted from
/// the script. `Converged` only if the change left the referee alone AND the post run completed AND
/// it introduced no failures the baseline lacked; anything else is a wall hit. The referee check
/// comes first: once the tests or their runner have been edited, the test comparison is describing
/// a bar the change wrote for itself.
fn infer_status(
    red_team: &RedTeamOutput,
    implementer: &ImplementerOutput,
    may_modify_tests: &[String],
    review: Option<&verifier::VerifierOutput>,
    threshold: verifier::Severity,
) -> RunStatus {
    let referee = converge::referee_touches(&implementer.rewritten_files, may_modify_tests);
    if !referee.is_empty() {
        tracing::warn!(files = ?referee, "run touched the referee; not converged");
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
        &implementer.passing_tests,
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
    passing_tests: Vec<String>,
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

async fn scout_host(ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    ctx.guard()?;
    let issue: String = serde_json::from_str(&arg).map_err(|e| format!("scout arg: {e}"))?;
    let plugins = ctx.plugin_context.for_node("scout");
    let cfg = node_agent_config(
        &ctx.engine,
        &ctx.config,
        ctx.plugin_context.pool_for("scout", ctx.rag_rat.clone()),
        "scout",
        scout::SCOUT_TOOLS,
        &plugins,
    )
    .map_err(|e| e.to_string())?;
    let node = ScoutNode {
        route: cfg.route,
        tools: cfg.tools,
        files: cfg.files,
        ledger: Some(Arc::clone(&ctx.ledger)),
        policy: cfg.policy,
        max_turns: cfg.max_turns,
        // Node-to-node clarification is built-in-flow only for now; the scripted path opts out.
        clarifier: None,
        system_prompt: cfg.system_prompt,
        plugins,
    };
    let out = node
        .run(issue, &RunState::new(&ctx.run_id, None))
        .await
        .map_err(|e| e.to_string())?;
    note(&ctx, "scout", &out, Some(arg)).await?;
    serde_json::to_string(&out).map_err(|e| e.to_string())
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
    let plugins = ctx.plugin_context.for_node("analyst");
    let cfg = node_agent_config(
        &ctx.engine,
        &ctx.config,
        ctx.plugin_context.pool_for("analyst", ctx.rag_rat.clone()),
        "analyst",
        analyst::ANALYST_TOOLS,
        &plugins,
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
            let plugins = ctx.plugin_context.for_node("redteam");
            let cfg = node_agent_config(
                &ctx.engine,
                &ctx.config,
                ctx.plugin_context.pool_for("redteam", ctx.rag_rat.clone()),
                "redteam",
                redteam::CLASSIFIER_TOOLS,
                &plugins,
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
    // A script may call `redTeam()` before or after `analyze()`. When a plan exists, its acceptance
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
    if !converge::test_command_ran(&out.failing_tests, &out.passing_tests, out.exit_code) {
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
        converge::test_command_ran(&prev.failing_tests, &prev.passing_tests, prev.exit_code);
    let referee = converge::referee_touches(&prev.rewritten_files, ctx.engine.may_modify_tests());
    // Referee first, same as the built-in loop: a moved referee makes the test sets meaningless,
    // so reverting it is what this iteration has to be told to do.
    let diagnostic = if !referee.is_empty() {
        converge::referee_correction(&referee)
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

    let analyst: AnalystOutput = latest_checkpoint(&ctx.store, &ctx.run_id, "analyst")
        .await
        .map_err(|e| e.to_string())?;
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
    let v = converge::test_command_ran(&s.failing_tests, &s.passing_tests, s.exit_code);
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

/// `verify({ analyst })` — read the worktree's diff against the plan.
///
/// Mirrors the built-in flow's second gate. The script decides when to call it; every judgement
/// inside stays here.
async fn verify_host(ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
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

    let plugins = ctx.plugin_context.for_node("verifier");
    let cfg = node_agent_config(
        &ctx.engine,
        &ctx.config,
        ctx.plugin_context.pool_for("verifier", ctx.rag_rat.clone()),
        "verifier",
        verifier::VERIFIER_TOOLS,
        &plugins,
    )
    .map_err(|e| e.to_string())?;
    let node = verifier::VerifierNode {
        route: cfg.route,
        tools: cfg.tools,
        files: cfg.files,
        ledger: Some(Arc::clone(&ctx.ledger)),
        policy: cfg.policy,
        max_turns: cfg.max_turns,
        system_prompt: cfg.system_prompt,
        plugins,
    };

    // The patch, not the `--stat` the implementer records: a summary cannot show a weakened
    // assertion, which is one of the things this gate exists to catch.
    let diff = ratatoskr_exec::diff_text(&worktree)
        .await
        .unwrap_or_default();
    let out = match node
        .run(verifier::VerifierInput {
            issue: ctx.issue.clone(),
            analyst: input.analyst,
            diff,
            touched_files: implementer.touched_files.clone(),
        })
        .await
    {
        Ok(out) => out,
        Err(e) => {
            // A verifier that cannot run must not fail a change that was made and passed. Recorded
            // and reported as unavailable, exactly as the built-in flow treats it.
            tracing::warn!("the verifier could not review this change: {e}");
            note(
                &ctx,
                "verifier",
                &serde_json::json!({ "error": e.to_string() }),
                None,
            )
            .await?;
            return none(true, true);
        }
    };
    note(&ctx, "verifier", &out, None).await?;

    let threshold = crate::parse_threshold(&ctx.config.implementer.verify_threshold);
    let blocking: Vec<verifier::Finding> = out.blocking(threshold).into_iter().cloned().collect();
    let needs_replan = blocking
        .iter()
        .any(|f| f.kind == verifier::FindingKind::Plan);
    serde_json::to_string(&VerifyResult {
        configured: true,
        unavailable: false,
        findings: out.findings,
        blocking,
        needs_replan,
    })
    .map_err(|e| e.to_string())
}

/// `context(issue)` — the merged gather step: distilled findings plus the memories unmodified.
///
/// `scout()` and `memory()` remain for a script that composes them itself. This is the one that
/// guarantees the ranked memory search happened.
async fn context_host(ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    ctx.guard()?;
    let issue: String = serde_json::from_str(&arg).map_err(|e| format!("context arg: {e}"))?;
    let plugins = ctx.plugin_context.for_node("context");
    let cfg = node_agent_config(
        &ctx.engine,
        &ctx.config,
        ctx.plugin_context.pool_for("context", ctx.rag_rat.clone()),
        "context",
        crate::context::CONTEXT_TOOLS,
        &plugins,
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

fn build_hosts(ctx: &Arc<WorkflowContext>) -> HashMap<String, HostFn> {
    let mut h = HashMap::new();
    h.insert("context".into(), binding(Arc::clone(ctx), context_host));
    h.insert("scout".into(), binding(Arc::clone(ctx), scout_host));
    h.insert("memory".into(), binding(Arc::clone(ctx), memory_host));
    h.insert("analyze".into(), binding(Arc::clone(ctx), analyze_host));
    h.insert("redTeam".into(), binding(Arc::clone(ctx), red_team_host));
    h.insert("implement".into(), binding(Arc::clone(ctx), implement_host));
    h.insert("iterate".into(), binding(Arc::clone(ctx), iterate_host));
    h.insert("verify".into(), binding(Arc::clone(ctx), verify_host));
    h.insert(
        "isConverged".into(),
        binding(Arc::clone(ctx), is_converged_host),
    );
    h.insert(
        "testCommandRan".into(),
        binding(Arc::clone(ctx), test_command_ran_host),
    );
    h.insert(
        "newlyIntroducedFailures".into(),
        binding(Arc::clone(ctx), newly_introduced_host),
    );
    h
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

    let hosts = build_hosts(&ctx);
    let input = json!({ "issue": ctx.issue }).to_string();
    let result = runtime.run("plan", input, hosts).await;

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

    let hosts = build_hosts(&ctx);
    let input =
        json!({ "issue": ctx.issue, "maxIterations": ctx.config.implementer.max_iterations })
            .to_string();

    // Run the script, then reconstruct the outcome. EITHER failing is a run failure: on any error
    // (a script/binding error, or a reconstruction error like a missing checkpoint) the worktree is
    // cleaned up and the run is marked `Failed` — never left orphaned or stuck at `Running`.
    let result = match runtime.run("run", input, hosts).await {
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
    let status = infer_status(
        &red_team,
        &implementer,
        ctx.engine.may_modify_tests(),
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
    let plugins = ctx.plugin_context.for_node("bookkeeper");
    let cfg = node_agent_config(
        &ctx.engine,
        &ctx.config,
        ctx.plugin_context
            .pool_for("bookkeeper", ctx.rag_rat.clone()),
        "bookkeeper",
        bookkeeper::BOOKKEEPER_TOOLS,
        &plugins,
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
            passing_tests: passing.iter().map(|s| s.to_string()).collect(),
            exit_code: exit,
            classifications: vec![],
        }
    }

    fn imp(failing: &[&str], passing: &[&str], exit: i32) -> ImplementerOutput {
        ImplementerOutput {
            worktree_path: "/wt".to_string(),
            diff_summary: String::new(),
            touched_files: vec![],
            rewritten_files: Vec::new(),
            failing_tests: failing.iter().map(|s| s.to_string()).collect(),
            passing_tests: passing.iter().map(|s| s.to_string()).collect(),
            exit_code: exit,
            narrative: None,
        }
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
    fn a_clean_test_run_does_not_convert_a_rewritten_referee_into_success() {
        // The BenchJack shape: every test passes, because conftest.py was rewritten to say so.
        let baseline = red(&["a"], &["b"], 1);
        let mut cheated = imp(&[], &["a", "b"], 0);
        cheated.touched_files = vec!["conftest.py".to_string()];
        cheated.rewritten_files = vec!["conftest.py".to_string()];
        cheated.rewritten_files = vec!["conftest.py".to_string()];
        assert_eq!(
            infer_status(&baseline, &cheated, &[], None, verifier::Severity::P2),
            RunStatus::MaxIterationsReached
        );
        // Unless the task declared it up front.
        assert_eq!(
            infer_status(
                &baseline,
                &cheated,
                &["conftest.py".to_string()],
                None,
                verifier::Severity::P2
            ),
            RunStatus::Converged
        );
    }

    #[test]
    fn adding_a_test_is_not_a_referee_touch() {
        // The gate refuses rewriting a test to make failures stop; it must not refuse writing one.
        // An implementer that believes otherwise ships untested code and contorts its design to
        // avoid a fixture it was never forbidden to extend.
        let baseline = red(&["a"], &["b"], 1);
        let mut honest = imp(&[], &["a", "b"], 0);
        honest.touched_files = vec!["src/lib.rs".to_string(), "tests/api.rs".to_string()];
        // Only added lines, so nothing was rewritten.
        honest.rewritten_files = vec!["src/lib.rs".to_string()];
        assert_eq!(
            infer_status(&baseline, &honest, &[], None, verifier::Severity::P2),
            RunStatus::Converged
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
        let cp = r#"{"worktree_path":"/w","failing_tests":[],"passing_tests":["t"],"exit_code":0}"#;
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
}
