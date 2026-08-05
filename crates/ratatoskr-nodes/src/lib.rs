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

pub use analyst::{AnalystNode, AnalystOutput};
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
        let plugin_context =
            PluginContext::resolve(config, &std::env::current_dir().unwrap_or_default()).await;
        let ctx = workflow::WorkflowContext::new(
            client,
            config,
            store,
            run_id,
            issue,
            engine,
            plugin_context,
        )?;
        return workflow::run_plan_scripted(runtime, ctx).await;
    }

    // Once per run: `SessionStart` describes the repository, not the node.
    let context =
        PluginContext::resolve(config, &std::env::current_dir().unwrap_or_default()).await;
    plan_half(client, config, store, run_id, issue, engine, &context).await
}

/// The planning half, against an already-resolved plugin context. `run_full` resolves it once and
/// reuses it for both halves, so a full run doesn't pay for every plugin's `SessionStart` twice.
async fn plan_half(
    client: &RagRatClient,
    config: &RatatoskrConfig,
    store: &Store,
    run_id: &str,
    issue: &str,
    engine: &Arc<ScriptEngine>,
    context: &PluginContext,
) -> Result<PlanOutcome, PlanError> {
    // First, and here rather than in a caller: every checkpoint references this row, and the
    // schema enforces it. `run_full` reaches this function directly, so a caller-side write would
    // leave that path creating checkpoints for a run that doesn't exist yet.
    store
        .upsert_run(run_id, None, RunStatus::Running.as_str())
        .await?;

    let clarifier = NodeClarifier::new(config, store, engine, run_id, issue, client.sink());
    let run = Run {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        clarifier: &clarifier,
        context,
    };
    let outcome = run_nodes(&run)
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

async fn run_nodes(run: &Run<'_>) -> Result<PlanOutcome, PlanError> {
    // Every field is a shared reference, so this just names them locally.
    let &Run {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        clarifier,
        context,
    } = run;
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
        system_prompt: scout_cfg.system_prompt,
        context: context.0.clone(),
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
        system_prompt: analyst_cfg.system_prompt,
        context: context.0.clone(),
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
    // The third structured event: a node produced output. Tool calls and model text come from
    // the agent's observability hook; this is what says a node actually finished.
    tracing::info!(kind = "checkpoint", node, bytes = json.len(), "checkpoint");
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

/// The red-team classifier is opt-in: it runs only when redteam has a model route to run on,
/// whether that comes from `[models.redteam]` or from its ruleset.
fn classifier_enabled(engine: &Arc<ScriptEngine>, config: &RatatoskrConfig) -> bool {
    config.models.contains_key("redteam")
        || engine
            .ruleset("redteam")
            .is_some_and(|r| r.config().model.is_some())
}

/// The resolved agent settings for one node: base config plus any `.ratatoskr/rules/<node>.ts`
/// overrides (model, tool set, per-call policy, max turns).
struct NodeAgentConfig {
    route: ratatoskr_core::ModelRoute,
    tools: Vec<Tool>,
    policy: Option<Arc<dyn ToolPolicy>>,
    max_turns: Option<usize>,
    /// Replaces the node's built-in preamble when the ruleset declares one.
    system_prompt: Option<String>,
}

/// Everything a run's helpers need in common: the rag-rat connection, the run's identity and
/// configuration, and the two things resolved once per run — the clarifier and the plugin context.
///
/// A parameter struct because these travel together through every stage; passing them
/// individually made each helper's signature grow with the run rather than with its job.
struct Run<'a> {
    client: &'a RagRatClient,
    config: &'a RatatoskrConfig,
    store: &'a Store,
    run_id: &'a str,
    issue: &'a str,
    engine: &'a Arc<ScriptEngine>,
    clarifier: &'a Arc<NodeClarifier>,
    context: &'a PluginContext,
}

/// What plugins contributed for this run, prefixed to every node's preamble.
///
/// Resolved once because `SessionStart` describes the repository, not the node. Once plugins are
/// bound per node this becomes per node too.
#[derive(Clone, Default)]
pub struct PluginContext(pub Option<String>);

impl PluginContext {
    /// Run every discovered plugin's `SessionStart` hook. Never fails: a plugin that is missing,
    /// broken, or slow simply contributes nothing.
    pub async fn resolve(config: &RatatoskrConfig, cwd: &std::path::Path) -> Self {
        let plugins = ratatoskr_plugin::discover(&config.plugins.search_paths(cwd));
        if plugins.is_empty() {
            return PluginContext(None);
        }
        for plugin in &plugins {
            tracing::info!(plugin = plugin.name, "loaded plugin");
        }
        let context = ratatoskr_plugin::session_start_context(&plugins, cwd).await;
        match &context {
            // Worth a line: this text is prefixed to every prompt these nodes make.
            Some(text) => tracing::info!(chars = text.len(), "plugin session context"),
            None => tracing::debug!("plugins contributed no session context"),
        }
        PluginContext(context)
    }

    pub fn as_deref(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

/// The preamble a node actually runs with: its built-in text, or a ruleset's replacement for it,
/// prefixed by whatever context plugins contributed for this run.
pub(crate) fn effective_preamble(
    built_in: &str,
    system_prompt: Option<&str>,
    context: Option<&str>,
) -> String {
    let base = system_prompt.unwrap_or(built_in);
    match context {
        Some(context) => format!("{context}\n\n{base}"),
        None => base.to_string(),
    }
}

/// Resolve a node's agent settings. The ruleset is authoritative where it speaks: its `model` is
/// the route (so a fully-declared node needs no `[models.<node>]` entry — that's only the
/// fallback), `allow` (if given) REPLACES `default_tools`, `deny` is always removed,
/// `systemPrompt` replaces the node's built-in preamble, and `onToolCall` (if defined) becomes the
/// per-call [`ToolPolicy`].
fn node_agent_config(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    all_tools: &[Tool],
    node: &str,
    default_tools: &[&str],
) -> Result<NodeAgentConfig, PlanError> {
    let ruleset = engine.ruleset(node);
    let rc = ruleset.as_ref().map(|r| r.config());

    // Ruleset model FIRST: a node whose ruleset declares one needs no `[models.<node>]` entry.
    // `route()` — and its "add a [models.<node>]" error — is only the fallback.
    let route = match rc.and_then(|c| c.model.as_ref()) {
        Some(m) => ratatoskr_core::ModelRoute {
            provider: m.provider.clone(),
            model: m.model.clone(),
        },
        None => route(config, node)?,
    };

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
    let system_prompt = rc.and_then(|c| c.system_prompt.clone());
    let policy: Option<Arc<dyn ToolPolicy>> = match ruleset {
        Some(r) if r.config().has_on_tool_call => Some(Arc::new(r) as Arc<dyn ToolPolicy>),
        _ => None,
    };

    Ok(NodeAgentConfig {
        route,
        tools,
        policy,
        max_turns,
        system_prompt,
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
        let plugin_context =
            PluginContext::resolve(config, &std::env::current_dir().unwrap_or_default()).await;
        let ctx = workflow::WorkflowContext::new(
            client,
            config,
            store,
            run_id,
            issue,
            engine,
            plugin_context,
        )?;
        return workflow::run_full_scripted(runtime, ctx).await;
    }

    // Resolved here and shared with both halves.
    let plan_context =
        PluginContext::resolve(config, &std::env::current_dir().unwrap_or_default()).await;
    let plan = plan_half(client, config, store, run_id, issue, engine, &plan_context).await?;
    // `plan.state.clarifications` already holds the plan-half asks; the fork/bookkeep half gets its
    // own clarifier, drained and appended at the end.
    let mut state = plan.state.clone();
    let clarifier = NodeClarifier::new(config, store, engine, run_id, issue, client.sink());

    // `run_plan` signs off with `Planned`, but a full run is only half done — the fork+converge
    // phase that follows is the longest one. Without this write the store would report `Planned`
    // for its entire duration, making an in-flight full run indistinguishable from a finished
    // `plan` (and a run that died mid-fork look like it planned successfully).
    //
    // Best-effort like the other mid-run status writes: this is observability bookkeeping, and
    // failing the run over it would discard completed planning work for a cosmetic reason.
    if let Err(e) = store
        .upsert_run(run_id, None, RunStatus::Running.as_str())
        .await
    {
        tracing::warn!("failed to record run status before the fork: {e}");
    }

    let run = Run {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        clarifier: &clarifier,
        // The plan half's context, reused: `SessionStart` runs once per run, not once per stage.
        context: &plan_context,
    };
    let result = fork_and_converge(&run, &plan).await;

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
        match bookkeep_and_checkpoint(&run, input).await {
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
    run: &Run<'_>,
    input: BookkeeperInput,
) -> Result<BookkeeperOutput, PlanError> {
    // Every field is a shared reference, so this just names them locally. The issue comes from
    // `input` here, which on a replay is reconstructed from the store rather than passed in.
    let &Run {
        client,
        config,
        store,
        run_id,
        engine,
        clarifier,
        context,
        ..
    } = run;
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
        system_prompt: cfg.system_prompt,
        context: context.0.clone(),
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
    let context =
        PluginContext::resolve(config, &std::env::current_dir().unwrap_or_default()).await;
    let run = Run {
        client,
        config,
        store,
        run_id,
        issue: &input.issue.clone(),
        engine,
        clarifier: &clarifier,
        context: &context,
    };
    bookkeep_and_checkpoint(&run, input).await
}

/// The fork + converge half. Returns the terminal status; leaves the worktree in place on a
/// terminal outcome and removes it on a hard error.
async fn fork_and_converge(
    run: &Run<'_>,
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
    let &Run {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        clarifier,
        context,
    } = run;
    let repo_path: PathBuf = std::env::current_dir()
        .map_err(|e| PlanError::node("fork", NodeError::Failed(format!("cwd: {e}"))))?;
    let short: String = run_id.chars().take(8).collect();

    let red_team = RedTeamNode {
        repo_path: repo_path.clone(),
        sandbox: config.sandbox.clone(),
        name: format!("ratatoskr-redteam-{short}"),
        // Opt-in: classify baseline failures only when redteam has a route — from
        // `[models.redteam]` or from its `.ratatoskr/rules/redteam.ts` ruleset.
        classifier: match classifier_enabled(engine, config) {
            true => {
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
                    system_prompt: cfg.system_prompt,
                    context: context.0.clone(),
                })
            }
            false => None,
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

#[cfg(test)]
mod agent_config_tests {
    use super::*;

    /// scout: full ruleset (model + prompt), no `[models.scout]`. bookkeeper: partial ruleset
    /// (no model) → TOML route + built-in preamble. memory: no ruleset at all.
    ///
    /// The fixture directory is unique per test *and* per process: these tests run concurrently,
    /// and `fs::write` truncates before writing, so a shared path lets one test's engine load
    /// another's half-written file and see a ruleset that is missing agents.
    async fn engine(case: &str) -> Arc<ScriptEngine> {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-nodes-agent-config-{}-{case}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("agents.ts"),
            r#"
            defineAgent("scout", {
                model: { provider: "openai", model: "gpt-5" },
                systemPrompt: "Be brief.",
            });
            defineAgent("bookkeeper", { maxTurns: 3 });
            "#,
        )
        .unwrap();
        ScriptEngine::load(&dir).await.unwrap()
    }

    #[test]
    fn plugin_context_prefixes_whichever_preamble_applies() {
        // A ruleset replaces the node's own text; plugin context is prepended to whatever wins,
        // so a repository digest never costs a node its instructions.
        assert_eq!(effective_preamble("built-in", None, None), "built-in");
        assert_eq!(
            effective_preamble("built-in", Some("override"), None),
            "override"
        );
        assert_eq!(
            effective_preamble("built-in", None, Some("digest")),
            "digest\n\nbuilt-in"
        );
        assert_eq!(
            effective_preamble("built-in", Some("override"), Some("digest")),
            "digest\n\noverride"
        );
    }

    #[tokio::test]
    async fn ruleset_model_replaces_the_toml_route() {
        let engine = engine("model-override").await;
        let mut config = RatatoskrConfig::default();
        // The whole point: no `[models.scout]` entry at all.
        config.models.remove("scout");

        let cfg = node_agent_config(&engine, &config, &[], "scout", &[]).unwrap();
        assert_eq!(cfg.route.provider, "openai");
        assert_eq!(cfg.route.model, "gpt-5");
        assert_eq!(cfg.system_prompt.as_deref(), Some("Be brief."));
    }

    #[tokio::test]
    async fn a_ruleset_without_a_model_still_falls_back_to_toml() {
        let engine = engine("toml-fallback").await;
        let config = RatatoskrConfig::default();

        let cfg = node_agent_config(&engine, &config, &[], "bookkeeper", &[]).unwrap();
        assert_eq!(cfg.route.provider, config.models["bookkeeper"].provider);
        assert_eq!(cfg.route.model, config.models["bookkeeper"].model);
        assert!(cfg.system_prompt.is_none());
        assert_eq!(cfg.max_turns, Some(3));
    }

    #[tokio::test]
    async fn no_ruleset_and_no_toml_route_is_still_an_error() {
        let engine = engine("no-route").await;
        let mut config = RatatoskrConfig::default();
        config.models.remove("analyst");

        assert!(matches!(
            node_agent_config(&engine, &config, &[], "analyst", &[]),
            Err(PlanError::MissingRoute(n)) if n == "analyst"
        ));
    }

    #[tokio::test]
    async fn redteam_classifier_opts_in_on_either_route_source() {
        let engine = engine("redteam-optin").await;
        let mut config = RatatoskrConfig::default();
        assert!(!classifier_enabled(&engine, &config));
        config.models.insert(
            "redteam".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "openai".to_string(),
                model: "gpt-5".to_string(),
            },
        );
        assert!(classifier_enabled(&engine, &config));
    }
}
