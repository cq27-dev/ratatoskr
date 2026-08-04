//! Concrete Phase 2 nodes (scout, memory, analyst) and the straight-line `plan` executor.
//!
//! Per the Phase 2 decision, this is a plain sequential `async fn`, not a generic edge-walking
//! interpreter: with exactly three fixed nodes and no branching, `run_plan` delivers the same
//! policy guarantee (schema-validated handoffs, a checkpoint after every node, nothing skipped)
//! with nothing to get wrong. The real executor arrives in Phase 3 when fork/join needs one.

pub mod analyst;
pub mod memory;
pub mod scout;

pub use analyst::{AnalystNode, AnalystOutput, Risk};
pub use memory::{MemoryNode, MemoryOutput, MemoryRecord};
pub use scout::{RelatedItem, ScoutNode, ScoutOutput};

use ratatoskr_core::{RatatoskrConfig, RunState, RunStatus};
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
