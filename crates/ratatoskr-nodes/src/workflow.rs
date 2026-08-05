//! Scriptable orchestration: run `.ratatoskr/workflow.ts` (issue #18) instead of the hardcoded
//! `run_plan`/`run_full` flow. The script composes host bindings — one per node call site — but
//! every gate stays Rust-enforced: schema validation and checkpointing happen inside each binding,
//! the false-convergence guard lives in `redTeam`, `max_iterations` is capped in `iterate`, and the
//! terminal status is inferred from checkpoints after the script returns (never trusted from the
//! script). A missing `workflow.ts` runs the built-in Rust flow unchanged (see [`super::run_full`]).
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
use ratatoskr_mcp::RagRatClient;
use ratatoskr_script::{HostFn, ScriptEngine, WorkflowRuntime};
use ratatoskr_store::Store;
use rmcp::model::Tool;
use rmcp::service::ServerSink;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::json;

use crate::{
    AnalystNode, AnalystOutput, BookkeeperInput, BookkeeperNode, BookkeeperOutput, ImplementerNode,
    ImplementerOutput, MemoryNode, MemoryOutput, PlanError, PlanOutcome, RedTeamNode,
    RedTeamOutput, RunOutcome, ScoutNode, ScoutOutput, analyst, bookkeeper, checkpoint, converge,
    memory, node_agent_config, redteam, scout,
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
    tools: Vec<Tool>,
    repo_path: PathBuf,
    /// Set by `implement`, read by `iterate` and cleanup. The script never sees a raw path.
    worktree: Mutex<Option<WorktreePath>>,
    implement_started: AtomicBool,
    /// Serializes `iterate` calls — two concurrent ACP sessions on one worktree would corrupt it.
    iterate_lock: tokio::sync::Mutex<()>,
    invocations: AtomicUsize,
    iterations: AtomicU32,
}

impl WorkflowContext {
    pub fn new(
        client: &RagRatClient,
        config: &RatatoskrConfig,
        store: &Store,
        run_id: &str,
        issue: &str,
        engine: &Arc<ScriptEngine>,
    ) -> Result<Arc<Self>, PlanError> {
        let repo_path = std::env::current_dir()
            .map_err(|e| PlanError::node("workflow", NodeError::Failed(format!("cwd: {e}"))))?;
        Ok(Arc::new(Self {
            config: config.clone(),
            store: store.clone(),
            engine: Arc::clone(engine),
            run_id: run_id.to_string(),
            issue: issue.to_string(),
            sink: client.sink(),
            tools: client.tools(),
            repo_path,
            worktree: Mutex::new(None),
            implement_started: AtomicBool::new(false),
            iterate_lock: tokio::sync::Mutex::new(()),
            invocations: AtomicUsize::new(0),
            iterations: AtomicU32::new(0),
        }))
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
/// the script. `Converged` only if the post run completed AND introduced no failures the baseline
/// lacked; anything else is a wall hit.
fn infer_status(red_team: &RedTeamOutput, implementer: &ImplementerOutput) -> RunStatus {
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
    let scout: ScoutOutput = latest_checkpoint(store, run_id, "scout").await?;
    let memory: MemoryOutput = latest_checkpoint(store, run_id, "memory").await?;
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
    let cfg = node_agent_config(
        &ctx.engine,
        &ctx.config,
        &ctx.tools,
        "scout",
        scout::SCOUT_TOOLS,
    )
    .map_err(|e| e.to_string())?;
    let node = ScoutNode {
        route: cfg.route,
        tools: cfg.tools,
        sink: ctx.sink.clone(),
        policy: cfg.policy,
        max_turns: cfg.max_turns,
    };
    let out = node
        .run(issue, &RunState::new(&ctx.run_id, None))
        .await
        .map_err(|e| e.to_string())?;
    checkpoint(&ctx.store, &ctx.run_id, "scout", &out)
        .await
        .map_err(|e| e.to_string())?;
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
    checkpoint(&ctx.store, &ctx.run_id, "memory", &out)
        .await
        .map_err(|e| e.to_string())?;
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

async fn analyze_host(ctx: Arc<WorkflowContext>, arg: String) -> Result<String, String> {
    ctx.guard()?;
    let input: analyst::AnalystInput =
        serde_json::from_str(&arg).map_err(|e| format!("analyze arg: {e}"))?;
    let cfg = node_agent_config(
        &ctx.engine,
        &ctx.config,
        &ctx.tools,
        "analyst",
        analyst::ANALYST_TOOLS,
    )
    .map_err(|e| e.to_string())?;
    let node = AnalystNode {
        route: cfg.route,
        tools: cfg.tools,
        sink: ctx.sink.clone(),
        policy: cfg.policy,
        max_turns: cfg.max_turns,
    };
    let out = node
        .run(input, &RunState::new(&ctx.run_id, None))
        .await
        .map_err(|e| e.to_string())?;
    checkpoint(&ctx.store, &ctx.run_id, "analyst", &out)
        .await
        .map_err(|e| e.to_string())?;
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

fn build_red_team(ctx: &WorkflowContext) -> Result<RedTeamNode, PlanError> {
    let short: String = ctx.run_id.chars().take(8).collect();
    let classifier = match ctx.config.models.get("redteam") {
        Some(_) => {
            let cfg = node_agent_config(
                &ctx.engine,
                &ctx.config,
                &ctx.tools,
                "redteam",
                redteam::CLASSIFIER_TOOLS,
            )?;
            Some(redteam::RedTeamClassifier {
                route: cfg.route,
                tools: cfg.tools,
                sink: ctx.sink.clone(),
                policy: cfg.policy,
                max_turns: cfg.max_turns,
            })
        }
        None => None,
    };
    Ok(RedTeamNode {
        repo_path: ctx.repo_path.clone(),
        sandbox: ctx.config.sandbox.clone(),
        name: format!("ratatoskr-redteam-{short}"),
        classifier,
    })
}

async fn red_team_host(ctx: Arc<WorkflowContext>, _arg: String) -> Result<String, String> {
    ctx.guard()?;
    let node = build_red_team(&ctx).map_err(|e| e.to_string())?;
    let out = node.run().await.map_err(|e| e.to_string())?;
    // Checkpoint before the guard so a failed baseline stays inspectable.
    checkpoint(&ctx.store, &ctx.run_id, "red_team", &out)
        .await
        .map_err(|e| e.to_string())?;
    // The false-convergence guard is enforced here — the script cannot skip it.
    if !converge::test_command_ran(&out.failing_tests, &out.passing_tests, out.exit_code) {
        return Err(format!(
            "baseline test command produced no tests (exit {}); check [sandbox] test_command and backend",
            out.exit_code
        ));
    }
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

fn build_implementer(ctx: &WorkflowContext, analyst: AnalystOutput) -> ImplementerNode {
    ImplementerNode {
        repo_path: ctx.repo_path.clone(),
        worktree_root: ctx.config.worktree.root.clone(),
        sandbox: ctx.config.sandbox.clone(),
        implementer: ctx.config.implementer.clone(),
        run_id: ctx.run_id.clone(),
        issue: ctx.issue.clone(),
        analyst,
    }
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
    let node = build_implementer(&ctx, input.analyst);
    let (worktree, out) = node.run().await.map_err(|e| e.to_string())?;
    *ctx.worktree.lock().unwrap() = Some(worktree);
    checkpoint(&ctx.store, &ctx.run_id, "implementer", &out)
        .await
        .map_err(|e| e.to_string())?;
    serde_json::to_string(&out).map_err(|e| e.to_string())
}

async fn iterate_host(ctx: Arc<WorkflowContext>, _arg: String) -> Result<String, String> {
    ctx.guard()?;
    // One iterate at a time — reject overlapping calls (e.g. `Promise.all([iterate(), iterate()])`)
    // that would drive two ACP sessions against the same worktree. Held for the whole call.
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
    let diagnostic = if !post_ran {
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
    let node = build_implementer(&ctx, analyst);
    let out = node
        .iterate(&worktree, &diagnostic)
        .await
        .map_err(|e| e.to_string())?;
    checkpoint(&ctx.store, &ctx.run_id, "implementer", &out)
        .await
        .map_err(|e| e.to_string())?;
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

fn build_hosts(ctx: &Arc<WorkflowContext>) -> HashMap<String, HostFn> {
    let mut h = HashMap::new();
    h.insert("scout".into(), binding(Arc::clone(ctx), scout_host));
    h.insert("memory".into(), binding(Arc::clone(ctx), memory_host));
    h.insert("analyze".into(), binding(Arc::clone(ctx), analyze_host));
    h.insert("redTeam".into(), binding(Arc::clone(ctx), red_team_host));
    h.insert("implement".into(), binding(Arc::clone(ctx), implement_host));
    h.insert("iterate".into(), binding(Arc::clone(ctx), iterate_host));
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
    checkpoint(
        &ctx.store,
        &ctx.run_id,
        "issue",
        &json!({ "issue": ctx.issue }),
    )
    .await?;
    ctx.store
        .upsert_run(&ctx.run_id, None, RunStatus::Running.as_str())
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
    outcome
}

/// Scripted `run`: the full flow via the script's `run(input)` entry. Rust infers the terminal
/// status from checkpoints and does the bookkeeping — the script only sequences.
pub async fn run_full_scripted(
    runtime: WorkflowRuntime,
    ctx: Arc<WorkflowContext>,
) -> Result<RunOutcome, PlanError> {
    checkpoint(
        &ctx.store,
        &ctx.run_id,
        "issue",
        &json!({ "issue": ctx.issue }),
    )
    .await?;
    ctx.store
        .upsert_run(&ctx.run_id, None, RunStatus::Running.as_str())
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
    let status = infer_status(&red_team, &implementer);
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
        red_team,
        implementer,
        worktree,
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
    let cfg = node_agent_config(
        &ctx.engine,
        &ctx.config,
        &ctx.tools,
        "bookkeeper",
        bookkeeper::BOOKKEEPER_TOOLS,
    )?;
    let node = BookkeeperNode {
        route: cfg.route,
        tools: cfg.tools,
        sink: ctx.sink.clone(),
        policy: cfg.policy,
        max_turns: cfg.max_turns,
    };
    let out = node
        .run(input)
        .await
        .map_err(|e| PlanError::node("bookkeeper", e))?;
    checkpoint(&ctx.store, &ctx.run_id, "bookkeeper", &out).await?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red(failing: &[&str], passing: &[&str], exit: i32) -> RedTeamOutput {
        RedTeamOutput {
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
            infer_status(&baseline, &imp(&["a"], &["b", "c"], 0)),
            RunStatus::Converged
        );
        // Post introduced a new failure → wall, not converged.
        assert_eq!(
            infer_status(&baseline, &imp(&["a", "c"], &["b"], 1)),
            RunStatus::MaxIterationsReached
        );
        // Post didn't run to completion (no tests) → wall, even with an empty failing list — this is
        // the P1a check the hardcoded loop also applies to the implementer output.
        assert_eq!(
            infer_status(&baseline, &imp(&[], &[], 101)),
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
            .insert_checkpoint(
                "r1",
                "scout",
                r#"{"related_items":[],"papertrail_summary":"s"}"#,
            )
            .await
            .unwrap();
        store
            .insert_checkpoint("r1", "memory", r#"{"memories":[]}"#)
            .await
            .unwrap();
        store
            .insert_checkpoint("r1", "analyst", r#"{"impact_summary":"i"}"#)
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
                .insert_checkpoint("r1", "implementer", cp)
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
}
