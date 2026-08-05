//! Concrete Phase 2 nodes (scout, memory, analyst) and the straight-line `plan` executor.
//!
//! Per the Phase 2 decision, this is a plain sequential `async fn`, not a generic edge-walking
//! interpreter: with exactly three fixed nodes and no branching, `run_plan` delivers the same
//! policy guarantee (schema-validated handoffs, a checkpoint after every node, nothing skipped)
//! with nothing to get wrong. The real executor arrives in Phase 3 when fork/join needs one.

pub mod analyst;
pub mod converge;
pub mod implementer;
pub mod memory;
pub mod redteam;
pub mod scout;
pub mod testrun;

pub use analyst::{AnalystNode, AnalystOutput, Risk};
pub use implementer::{ImplementerNode, ImplementerOutput};
pub use memory::{MemoryNode, MemoryOutput, MemoryRecord};
pub use redteam::{RedTeamNode, RedTeamOutput};
pub use scout::{RelatedItem, ScoutNode, ScoutOutput};

use std::path::PathBuf;

use ratatoskr_core::{RatatoskrConfig, RunState, RunStatus};
use ratatoskr_exec::{WorktreePath, remove_worktree};
use ratatoskr_graph::{Node, NodeError};
use ratatoskr_mcp::RagRatClient;
use ratatoskr_store::{Store, StoreError};
use rmcp::model::Tool;
use serde::Serialize;

/// Everything a completed plan run produced. `state` carries the run's status and the node slots;
/// the typed fields are the nodes' validated outputs.
pub struct PlanOutcome {
    pub state: RunState,
    pub scout: ScoutOutput,
    pub memory: MemoryOutput,
    pub analyst: AnalystOutput,
}

/// Errors from a plan run. A `Node` error means that node failed or produced invalid output; the
/// run is marked `Failed` before this is returned.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("node `{node}` failed: {source}")]
    Node {
        node: &'static str,
        #[source]
        source: NodeError,
    },
    #[error("store error: {0}")]
    Store(#[from] StoreError),
    #[error("serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("no model route `{0}` in config — add a [models.{0}] entry")]
    MissingRoute(String),
}

impl PlanError {
    fn node(node: &'static str, source: NodeError) -> Self {
        PlanError::Node { node, source }
    }
}

/// Run scout → memory → analyst in sequence, checkpointing after each, and record the run's final
/// status. On any node failure the run is marked `Failed` and the error names the node.
pub async fn run_plan(
    client: &RagRatClient,
    config: &RatatoskrConfig,
    store: &Store,
    run_id: &str,
    issue: &str,
) -> Result<PlanOutcome, PlanError> {
    store
        .upsert_run(run_id, None, RunStatus::Running.as_str())
        .await?;

    let outcome = run_nodes(client, config, store, run_id, issue).await;

    let final_status = if outcome.is_ok() {
        RunStatus::Planned
    } else {
        RunStatus::Failed
    };
    // Best-effort status write; don't mask a node error with a store error on the failure path.
    if let Err(e) = store.upsert_run(run_id, None, final_status.as_str()).await {
        tracing::warn!("failed to record final run status: {e}");
    }

    outcome
}

async fn run_nodes(
    client: &RagRatClient,
    config: &RatatoskrConfig,
    store: &Store,
    run_id: &str,
    issue: &str,
) -> Result<PlanOutcome, PlanError> {
    let sink = client.sink();
    let all_tools = client.tools();
    let mut state = RunState::new(run_id, None);
    state.status = RunStatus::Running;

    // --- scout ---
    let scout = ScoutNode {
        route: route(config, "scout")?,
        tools: filter_tools(&all_tools, scout::SCOUT_TOOLS),
        sink: sink.clone(),
    };
    let scout_out = scout
        .run(issue.to_string(), &state)
        .await
        .map_err(|e| PlanError::node("scout", e))?;
    checkpoint(store, run_id, "scout", &scout_out).await?;
    state.scout_report = Some(serde_json::to_value(&scout_out)?);

    // --- memory ---
    let memory = MemoryNode { sink: sink.clone() };
    let memory_out = memory
        .run(
            memory::MemoryInput {
                issue: issue.to_string(),
                context: scout_out.papertrail_summary.clone(),
            },
            &state,
        )
        .await
        .map_err(|e| PlanError::node("memory", e))?;
    checkpoint(store, run_id, "memory", &memory_out).await?;
    state.memories = memory_out
        .memories
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()?;

    // --- analyst ---
    let analyst = AnalystNode {
        route: route(config, "analyst")?,
        tools: filter_tools(&all_tools, analyst::ANALYST_TOOLS),
        sink: sink.clone(),
    };
    let analyst_out = analyst
        .run(
            analyst::AnalystInput {
                issue: issue.to_string(),
                scout: scout_out.clone(),
                memory: memory_out.clone(),
            },
            &state,
        )
        .await
        .map_err(|e| PlanError::node("analyst", e))?;
    checkpoint(store, run_id, "analyst", &analyst_out).await?;
    state.analysis = Some(serde_json::to_value(&analyst_out)?);

    state.status = RunStatus::Planned;
    Ok(PlanOutcome {
        state,
        scout: scout_out,
        memory: memory_out,
        analyst: analyst_out,
    })
}

async fn checkpoint<T: Serialize>(
    store: &Store,
    run_id: &str,
    node: &str,
    output: &T,
) -> Result<(), PlanError> {
    let json = serde_json::to_string(output)?;
    store.insert_checkpoint(run_id, node, &json).await?;
    Ok(())
}

fn route(config: &RatatoskrConfig, name: &str) -> Result<ratatoskr_core::ModelRoute, PlanError> {
    config
        .models
        .get(name)
        .cloned()
        .ok_or_else(|| PlanError::MissingRoute(name.to_string()))
}

/// Keep only the tools named in `names` — a focused subset per node. Names not present in the
/// server's tool list are silently absent (logged); the node runs with whatever it got.
pub fn filter_tools(all: &[Tool], names: &[&str]) -> Vec<Tool> {
    let kept: Vec<Tool> = all
        .iter()
        .filter(|t| names.contains(&t.name.as_ref()))
        .cloned()
        .collect();
    if kept.len() < names.len() {
        tracing::warn!(
            requested = ?names,
            found = kept.len(),
            "some requested rag-rat tools were not offered by the server"
        );
    }
    kept
}

/// Everything a full fork+converge run produced. The worktree is the reviewable deliverable and is
/// left in place on a terminal status (converged or max-iterations); it's removed on a hard error.
pub struct RunOutcome {
    pub state: RunState,
    pub plan: PlanOutcome,
    pub red_team: RedTeamOutput,
    pub implementer: ImplementerOutput,
    pub worktree: WorktreePath,
    pub iterations: u32,
    pub status: RunStatus,
}

/// The full Phase 3 run: plan (scout → memory → analyst), then fork red-team ∥ implementer, then
/// converge. Reuses [`run_plan`] for the planning half.
pub async fn run_full(
    client: &RagRatClient,
    config: &RatatoskrConfig,
    store: &Store,
    run_id: &str,
    issue: &str,
) -> Result<RunOutcome, PlanError> {
    let plan = run_plan(client, config, store, run_id, issue).await?;
    let mut state = plan.state.clone();

    let result = fork_and_converge(config, store, run_id, issue, &plan).await;

    let status = match &result {
        Ok((_, _, _, status, _)) => *status,
        Err(_) => RunStatus::Failed,
    };
    if let Err(e) = store.upsert_run(run_id, None, status.as_str()).await {
        tracing::warn!("failed to record final run status: {e}");
    }

    let (red_team, implementer, worktree, status, iterations) = result?;
    state.red_team = Some(serde_json::to_value(&red_team)?);
    state.implementer = Some(serde_json::to_value(&implementer)?);
    state.status = status;

    Ok(RunOutcome {
        state,
        plan,
        red_team,
        implementer,
        worktree,
        iterations,
        status,
    })
}

/// The fork + converge half. Returns the terminal status; leaves the worktree in place on a
/// terminal outcome and removes it on a hard error.
async fn fork_and_converge(
    config: &RatatoskrConfig,
    store: &Store,
    run_id: &str,
    issue: &str,
    plan: &PlanOutcome,
) -> Result<
    (
        RedTeamOutput,
        ImplementerOutput,
        WorktreePath,
        RunStatus,
        u32,
    ),
    PlanError,
> {
    let repo_path: PathBuf = std::env::current_dir()
        .map_err(|e| PlanError::node("fork", NodeError::Failed(format!("cwd: {e}"))))?;
    let short: String = run_id.chars().take(8).collect();

    let red_team = RedTeamNode {
        repo_path: repo_path.clone(),
        sandbox: config.sandbox.clone(),
        name: format!("ratatoskr-redteam-{short}"),
    };
    let implementer = ImplementerNode {
        repo_path: repo_path.clone(),
        worktree_root: config.worktree.root.clone(),
        sandbox: config.sandbox.clone(),
        implementer: config.implementer.clone(),
        run_id: run_id.to_string(),
        issue: issue.to_string(),
        analyst: plan.analyst.clone(),
    };

    // Fork: both branches run concurrently off the same frozen post-analyst state. join! (not
    // spawn) because both are I/O-bound (subprocess/sandbox) and borrow their nodes.
    let (rt_res, impl_res) = tokio::join!(red_team.run(), implementer.run());

    let red_team_out = rt_res.map_err(|e| PlanError::node("red_team", e))?;
    let (worktree, mut impl_out) = impl_res.map_err(|e| PlanError::node("implementer", e))?;

    checkpoint(store, run_id, "red_team", &red_team_out).await?;
    checkpoint(store, run_id, "implementer", &impl_out).await?;

    // Converge: iterate the implementer (not red-team — the baseline doesn't change) until it
    // introduces no new failures, or the budget runs out.
    let mut iterations = 1u32;
    let status = loop {
        if converge::is_converged(&red_team_out.failing_tests, &impl_out.failing_tests) {
            break RunStatus::Converged;
        }
        if iterations >= config.implementer.max_iterations {
            break RunStatus::MaxIterationsReached;
        }
        let new_failures = converge::newly_introduced_failures(
            &red_team_out.failing_tests,
            &impl_out.failing_tests,
        );
        let diagnostic = format!(
            "Your change introduced NEW failing tests not present in the baseline: {}. \
             Fix them without breaking other tests.",
            new_failures.join(", ")
        );
        impl_out = match implementer.iterate(&worktree, &diagnostic).await {
            Ok(out) => out,
            Err(e) => {
                // Hard error mid-converge: don't leave the worktree behind.
                if let Err(rm) = remove_worktree(&repo_path, &worktree).await {
                    tracing::warn!("failed to clean up worktree after converge error: {rm}");
                }
                return Err(PlanError::node("implementer", e));
            }
        };
        checkpoint(store, run_id, "implementer", &impl_out).await?;
        iterations += 1;
    };

    Ok((red_team_out, impl_out, worktree, status, iterations))
}
