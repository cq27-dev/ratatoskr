//! Concrete Phase 2 nodes (scout, memory, analyst) and the straight-line `plan` executor.
//!
//! Per the Phase 2 decision, this is a plain sequential `async fn`, not a generic edge-walking
//! interpreter: with exactly three fixed nodes and no branching, `run_plan` delivers the same
//! policy guarantee (schema-validated handoffs, a checkpoint after every node, nothing skipped)
//! with nothing to get wrong. The real executor arrives in Phase 3 when fork/join needs one.

pub mod analyst;
pub mod bookkeeper;
pub mod clarify;
pub mod converge;
pub mod implementer;
pub mod memory;
pub mod redteam;
pub mod scout;
pub mod testrun;
pub mod workflow;

pub use analyst::{AnalystNode, AnalystOutput, Risk};
pub use bookkeeper::{BookkeeperInput, BookkeeperNode, BookkeeperOutput, MemoryWritten};
pub use implementer::{ImplementerNode, ImplementerOutput};
pub use memory::{MemoryNode, MemoryOutput, MemoryRecord};
pub use redteam::{RedTeamNode, RedTeamOutput};
pub use scout::{RelatedItem, ScoutNode, ScoutOutput};

use std::path::PathBuf;
use std::sync::Arc;

use ratatoskr_core::{RatatoskrConfig, RunState, RunStatus, ToolPolicy};
use ratatoskr_exec::{WorktreePath, remove_worktree};
use ratatoskr_graph::{Node, NodeError};
use ratatoskr_mcp::RagRatClient;
use ratatoskr_script::{ScriptEngine, WorkflowRuntime};
use ratatoskr_store::{Store, StoreError};

use crate::clarify::NodeClarifier;
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
    #[error("run {0} has no `{1}` checkpoint — not a converged run?")]
    MissingCheckpoint(String, &'static str),
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
    engine: &Arc<ScriptEngine>,
) -> Result<PlanOutcome, PlanError> {
    // A `.ratatoskr/workflow.ts` overrides the built-in scout → memory → analyst sequencing.
    if let Some(runtime) = load_workflow().await? {
        let ctx = workflow::WorkflowContext::new(client, config, store, run_id, issue, engine)?;
        return workflow::run_plan_scripted(runtime, ctx).await;
    }

    store
        .upsert_run(run_id, None, RunStatus::Running.as_str())
        .await?;

    let clarifier = NodeClarifier::new(config, store, engine, run_id, issue, client.sink());
    let outcome = run_nodes(client, config, store, run_id, issue, engine, &clarifier)
        .await
        // Drain the plan-half clarifications into the outcome's state (the clarifier can't reach the
        // borrowed RunState during the run).
        .map(|mut o| {
            o.state.clarifications = clarifier.drain();
            o
        });

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
    engine: &Arc<ScriptEngine>,
    clarifier: &Arc<NodeClarifier>,
) -> Result<PlanOutcome, PlanError> {
    let sink = client.sink();
    let all_tools = client.tools();
    let mut state = RunState::new(run_id, None);
    state.status = RunStatus::Running;

    // Persist the issue so `ratatoskr bookkeep <run-id>` can replay against stored checkpoints.
    checkpoint(
        store,
        run_id,
        "issue",
        &serde_json::json!({ "issue": issue }),
    )
    .await?;

    // --- scout ---
    let scout_cfg = node_agent_config(engine, config, &all_tools, "scout", scout::SCOUT_TOOLS)?;
    let mut scout_tools = scout_cfg.tools;
    scout_tools.push(clarify::ask_tool());
    let scout = ScoutNode {
        route: scout_cfg.route,
        tools: scout_tools,
        sink: sink.clone(),
        policy: scout_cfg.policy,
        max_turns: scout_cfg.max_turns,
        clarifier: Some(clarifier.as_dyn()),
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
    let analyst_cfg = node_agent_config(
        engine,
        config,
        &all_tools,
        "analyst",
        analyst::ANALYST_TOOLS,
    )?;
    let analyst = AnalystNode {
        route: analyst_cfg.route,
        tools: analyst_cfg.tools,
        sink: sink.clone(),
        policy: analyst_cfg.policy,
        max_turns: analyst_cfg.max_turns,
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

/// Load `.ratatoskr/workflow.ts` if present — the optional scriptable-orchestration override.
async fn load_workflow() -> Result<Option<WorkflowRuntime>, PlanError> {
    WorkflowRuntime::load(std::path::Path::new(".ratatoskr/workflow.ts"))
        .await
        .map_err(|e| PlanError::node("workflow", NodeError::Failed(e.to_string())))
}

fn route(config: &RatatoskrConfig, name: &str) -> Result<ratatoskr_core::ModelRoute, PlanError> {
    config
        .models
        .get(name)
        .cloned()
        .ok_or_else(|| PlanError::MissingRoute(name.to_string()))
}

/// The resolved agent settings for one node: base config plus any `.ratatoskr/rules/<node>.ts`
/// overrides (model, tool set, per-call policy, max turns).
struct NodeAgentConfig {
    route: ratatoskr_core::ModelRoute,
    tools: Vec<Tool>,
    policy: Option<Arc<dyn ToolPolicy>>,
    max_turns: Option<usize>,
}

/// Resolve a node's agent settings from `[models.<node>]` + `default_tools`, then layer the ruleset:
/// `allow` (if given) REPLACES the default tool set, `deny` is always removed, a `model` rule
/// overrides provider/model, and `onToolCall` (if defined) becomes the per-call [`ToolPolicy`].
fn node_agent_config(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    all_tools: &[Tool],
    node: &str,
    default_tools: &[&str],
) -> Result<NodeAgentConfig, PlanError> {
    let mut route = route(config, node)?;
    let ruleset = engine.ruleset(node);
    let rc = ruleset.as_ref().map(|r| r.config());

    if let Some(m) = rc.and_then(|c| c.model.as_ref()) {
        route = ratatoskr_core::ModelRoute {
            provider: m.provider.clone(),
            model: m.model.clone(),
        };
    }

    let allow: Vec<&str> = match rc
        .and_then(|c| c.tools.as_ref())
        .and_then(|t| t.allow.as_deref())
    {
        Some(a) => a.iter().map(String::as_str).collect(),
        None => default_tools.to_vec(),
    };
    let deny: Vec<&str> = rc
        .and_then(|c| c.tools.as_ref())
        .map(|t| t.deny.iter().map(String::as_str).collect())
        .unwrap_or_default();
    let tools: Vec<Tool> = filter_tools(all_tools, &allow)
        .into_iter()
        .filter(|t| !deny.contains(&t.name.as_ref()))
        .collect();

    let max_turns = rc.and_then(|c| c.max_turns);
    let policy: Option<Arc<dyn ToolPolicy>> = match ruleset {
        Some(r) if r.config().has_on_tool_call => Some(Arc::new(r) as Arc<dyn ToolPolicy>),
        _ => None,
    };

    Ok(NodeAgentConfig {
        route,
        tools,
        policy,
        max_turns,
    })
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
    /// Bookkeeper result — `Some` on a terminal fork outcome (converged, or max-iterations with an
    /// `unresolved`-tagged memory); `None` otherwise.
    pub bookkeeper: Option<BookkeeperOutput>,
}

/// The full Phase 3 run: plan (scout → memory → analyst), then fork red-team ∥ implementer, then
/// converge. Reuses [`run_plan`] for the planning half.
pub async fn run_full(
    client: &RagRatClient,
    config: &RatatoskrConfig,
    store: &Store,
    run_id: &str,
    issue: &str,
    engine: &Arc<ScriptEngine>,
) -> Result<RunOutcome, PlanError> {
    // A `.ratatoskr/workflow.ts` overrides the whole run flow (plan + fork + converge).
    if let Some(runtime) = load_workflow().await? {
        let ctx = workflow::WorkflowContext::new(client, config, store, run_id, issue, engine)?;
        return workflow::run_full_scripted(runtime, ctx).await;
    }

    let plan = run_plan(client, config, store, run_id, issue, engine).await?;
    // `plan.state.clarifications` already holds the plan-half asks; the fork/bookkeep half gets its
    // own clarifier, drained and appended at the end.
    let mut state = plan.state.clone();
    let clarifier = NodeClarifier::new(config, store, engine, run_id, issue, client.sink());

    let result = fork_and_converge(
        client, config, store, run_id, issue, &plan, engine, &clarifier,
    )
    .await;

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

    // Bookkeeping fires on a terminal fork outcome: `Converged` (record the learning) or
    // `MaxIterationsReached` (record the wall, tagged `unresolved`). A bookkeeping failure is
    // logged but doesn't discard the run's work.
    let bookkeeper = if matches!(
        status,
        RunStatus::Converged | RunStatus::MaxIterationsReached
    ) {
        let input = BookkeeperInput {
            issue: issue.to_string(),
            analyst: plan.analyst.clone(),
            implementer: implementer.clone(),
            iterations,
            converged: status == RunStatus::Converged,
        };
        match bookkeep_and_checkpoint(client, config, store, run_id, input, engine, &clarifier)
            .await
        {
            Ok(bk) => {
                state.artifacts = vec![serde_json::to_value(&bk)?];
                Some(bk)
            }
            Err(e) => {
                tracing::warn!("bookkeeping failed: {e}");
                None
            }
        }
    } else {
        None
    };

    // Append the fork/bookkeep-half clarifications to the plan-half ones.
    state.clarifications.extend(clarifier.drain());

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

/// Build the bookkeeper node, run it, and checkpoint its output. Shared by `run_full` (auto path)
/// and `run_bookkeeper` (replay).
async fn bookkeep_and_checkpoint(
    client: &RagRatClient,
    config: &RatatoskrConfig,
    store: &Store,
    run_id: &str,
    input: BookkeeperInput,
    engine: &Arc<ScriptEngine>,
    clarifier: &Arc<NodeClarifier>,
) -> Result<BookkeeperOutput, PlanError> {
    let cfg = node_agent_config(
        engine,
        config,
        &client.tools(),
        "bookkeeper",
        bookkeeper::BOOKKEEPER_TOOLS,
    )?;
    let mut tools = cfg.tools;
    tools.push(clarify::ask_tool());
    let node = BookkeeperNode {
        route: cfg.route,
        tools,
        sink: client.sink(),
        policy: cfg.policy,
        max_turns: cfg.max_turns,
        clarifier: Some(clarifier.as_dyn()),
    };
    let out = node
        .run(input)
        .await
        .map_err(|e| PlanError::node("bookkeeper", e))?;
    checkpoint(store, run_id, "bookkeeper", &out).await?;
    Ok(out)
}

/// Replay the bookkeeper alone against a previously-run run's stored checkpoints — no Phase 3
/// re-run. Reads the issue/analyst/implementer checkpoints and composes a fresh memory.
pub async fn run_bookkeeper(
    client: &RagRatClient,
    config: &RatatoskrConfig,
    store: &Store,
    run_id: &str,
    engine: &Arc<ScriptEngine>,
) -> Result<BookkeeperOutput, PlanError> {
    let checkpoints = store.checkpoints_for_run(run_id).await?;
    let latest = |name: &'static str| {
        checkpoints
            .iter()
            .rev()
            .find(|c| c.node_name == name)
            .ok_or(PlanError::MissingCheckpoint(run_id.to_string(), name))
    };

    let issue = checkpoints
        .iter()
        .rev()
        .find(|c| c.node_name == "issue")
        .and_then(|c| serde_json::from_str::<serde_json::Value>(&c.output_json).ok())
        .and_then(|v| v.get("issue").and_then(|i| i.as_str()).map(str::to_string))
        .unwrap_or_default();

    let analyst: AnalystOutput = serde_json::from_str(&latest("analyst")?.output_json)?;
    let implementer: ImplementerOutput = serde_json::from_str(&latest("implementer")?.output_json)?;
    let iterations = checkpoints
        .iter()
        .filter(|c| c.node_name == "implementer")
        .count() as u32;
    // A replay treats anything not recorded as `converged` as a wall hit.
    let converged =
        store.run_status(run_id).await?.as_deref() == Some(RunStatus::Converged.as_str());

    // Build the clarifier before `issue` is moved into the input (it clones the issue internally).
    let clarifier = NodeClarifier::new(config, store, engine, run_id, &issue, client.sink());
    let input = BookkeeperInput {
        issue,
        analyst,
        implementer,
        iterations,
        converged,
    };
    bookkeep_and_checkpoint(client, config, store, run_id, input, engine, &clarifier).await
}

/// The fork + converge half. Returns the terminal status; leaves the worktree in place on a
/// terminal outcome and removes it on a hard error.
#[allow(clippy::too_many_arguments)] // run context (client/config/store/run_id/issue/plan/engine) + the clarifier
async fn fork_and_converge(
    client: &RagRatClient,
    config: &RatatoskrConfig,
    store: &Store,
    run_id: &str,
    issue: &str,
    plan: &PlanOutcome,
    engine: &Arc<ScriptEngine>,
    clarifier: &Arc<NodeClarifier>,
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
        // Opt-in: classify baseline failures only when a [models.redteam] route is configured.
        // When it is, its `.ratatoskr/rules/redteam.ts` ruleset (if any) applies on top.
        classifier: match config.models.get("redteam") {
            Some(_) => {
                let cfg = node_agent_config(
                    engine,
                    config,
                    &client.tools(),
                    "redteam",
                    redteam::CLASSIFIER_TOOLS,
                )?;
                let mut tools = cfg.tools;
                tools.push(clarify::ask_tool());
                Some(redteam::RedTeamClassifier {
                    route: cfg.route,
                    tools,
                    sink: client.sink(),
                    policy: cfg.policy,
                    max_turns: cfg.max_turns,
                    clarifier: Some(clarifier.as_dyn()),
                })
            }
            None => None,
        },
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

    // Hard guard: red-team must have actually characterized the baseline. If the test command
    // produced no tests, converge would compare against empty data and falsely "converge".
    if !converge::test_command_ran(
        &red_team_out.failing_tests,
        &red_team_out.passing_tests,
        red_team_out.exit_code,
    ) {
        return Err(PlanError::node(
            "red_team",
            NodeError::Failed(format!(
                "baseline test command produced no tests (exit {}); \
                 check [sandbox] test_command and backend",
                red_team_out.exit_code
            )),
        ));
    }

    // Converge: iterate the implementer (not red-team — the baseline doesn't change) until it
    // introduces no new failures, or the budget runs out.
    let mut iterations = 1u32;
    let status = loop {
        let post_ran = converge::test_command_ran(
            &impl_out.failing_tests,
            &impl_out.passing_tests,
            impl_out.exit_code,
        );
        if post_ran && converge::is_converged(&red_team_out.failing_tests, &impl_out.failing_tests)
        {
            break RunStatus::Converged;
        }
        if iterations >= config.implementer.max_iterations {
            break RunStatus::MaxIterationsReached;
        }
        // A post-change run that didn't complete usually means the edit broke the build — say that
        // specifically instead of reporting "no new failures".
        let diagnostic = if !post_ran {
            format!(
                "The test command did not run to completion (exit {}) — your change likely does \
                 not compile. Fix it so the tests run and pass.",
                impl_out.exit_code
            )
        } else {
            let new_failures = converge::newly_introduced_failures(
                &red_team_out.failing_tests,
                &impl_out.failing_tests,
            );
            format!(
                "Your change introduced NEW failing tests not present in the baseline: {}. \
                 Fix them without breaking other tests.",
                new_failures.join(", ")
            )
        };
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
