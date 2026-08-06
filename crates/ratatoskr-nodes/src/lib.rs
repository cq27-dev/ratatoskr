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
pub mod skills;
pub mod testrun;
pub mod verifier;
pub mod workflow;

pub use analyst::{AnalystNode, AnalystOutput};
pub use bookkeeper::{BookkeeperInput, BookkeeperNode, BookkeeperOutput, MemoryWritten};
pub use implementer::{ImplementerNode, ImplementerOutput};
pub use memory::{MemoryNode, MemoryOutput, MemoryRecord};
pub use redteam::{RedTeamNode, RedTeamOutput};
pub use scout::{RelatedItem, ScoutNode, ScoutOutput};
pub use verifier::{Finding, FindingKind, Severity, VerifierNode, VerifierOutput};

use std::path::PathBuf;
use std::sync::Arc;

use ratatoskr_core::{RatatoskrConfig, RunState, RunStatus, ToolPolicy};
use ratatoskr_exec::{WorktreePath, remove_worktree};
use ratatoskr_graph::{Node, NodeError};
use ratatoskr_mcp::{Connection, RagRatClient, ServerTools, ToolSet};
use ratatoskr_script::{ScriptEngine, WorkflowRuntime};
use ratatoskr_store::{Store, StoreError};

use ratatoskr_agent::RunLedger;

use crate::clarify::NodeClarifier;
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
pub async fn run_plan(request: RunRequest<'_>) -> Result<PlanOutcome, PlanError> {
    let RunRequest {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        workflow,
    } = request;
    // A workflow, when this repo defines one, overrides the built-in sequencing.
    if let Workflow::Scripted(runtime) = select(registry().await?, workflow)? {
        let plugin_context =
            PluginContext::resolve(config, engine, &std::env::current_dir().unwrap_or_default())
                .await?;
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
        PluginContext::resolve(config, engine, &std::env::current_dir().unwrap_or_default())
            .await?;
    let outcome = plan_half(client, config, store, run_id, issue, engine, &context).await;
    // Closed by whoever owns the run's lifetime: a `plan` ends here, a `run` carries on.
    context.session_end(status_of(&outcome)).await;
    outcome
}

/// The status a finished plan reports, which is also the reason its session ended.
fn status_of(outcome: &Result<PlanOutcome, PlanError>) -> &'static str {
    match outcome.is_ok() {
        true => RunStatus::Planned.as_str(),
        false => RunStatus::Failed.as_str(),
    }
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

    record_provenance(store, run_id, config).await;

    let clarifier = NodeClarifier::new(config, store, engine, run_id, issue);
    let ledger = Arc::new(RunLedger::default());
    let run = Run {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        clarifier: &clarifier,
        ledger: &ledger,
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
        ledger,
        context,
    } = run;
    let sink = client.sink();
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
    let plugins_scout = context.for_node("scout");
    let scout_cfg = node_agent_config(
        engine,
        config,
        context.pool_for("scout", client.offer()),
        "scout",
        scout::SCOUT_TOOLS,
        &plugins_scout,
    )?;
    let mut scout_tools = scout_cfg.tools;
    scout_tools.add_local(clarify::ask_tool());
    let scout = ScoutNode {
        route: scout_cfg.route,
        tools: scout_tools,
        policy: scout_cfg.policy,
        max_turns: scout_cfg.max_turns,
        clarifier: Some(clarifier.as_dyn()),
        system_prompt: scout_cfg.system_prompt,
        plugins: plugins_scout,
        files: scout_cfg.files,
        ledger: Some(Arc::clone(ledger)),
    };
    let scout_out = scout
        .run(issue.to_string(), &state)
        .await
        .map_err(|e| PlanError::node("scout", e))?;
    record(Record {
        store,
        run_id,
        node: "scout",
        output: &scout_out,
        input: Some(serde_json::to_string(issue)?),
        iteration: None,
        ledger: Some(ledger),
    })
    .await?;
    state.scout_report = Some(serde_json::to_value(&scout_out)?);

    // --- memory ---
    let memory = MemoryNode { sink: sink.clone() };
    let memory_in = memory::MemoryInput {
        issue: issue.to_string(),
        context: scout_out.papertrail_summary.clone(),
    };
    let memory_input_json = serde_json::to_string(&memory_in)?;
    let memory_out = memory
        .run(memory_in, &state)
        .await
        .map_err(|e| PlanError::node("memory", e))?;
    record(Record {
        store,
        run_id,
        node: "memory",
        output: &memory_out,
        input: Some(memory_input_json),
        iteration: None,
        // The memory node calls rag-rat directly rather than a model, so it reports no usage.
        ledger: Some(ledger),
    })
    .await?;
    state.memories = memory_out
        .memories
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()?;

    // --- analyst ---
    let plugins_analyst = context.for_node("analyst");
    let analyst_cfg = node_agent_config(
        engine,
        config,
        context.pool_for("analyst", client.offer()),
        "analyst",
        analyst::ANALYST_TOOLS,
        &plugins_analyst,
    )?;
    let analyst = AnalystNode {
        route: analyst_cfg.route,
        tools: analyst_cfg.tools,
        policy: analyst_cfg.policy,
        max_turns: analyst_cfg.max_turns,
        system_prompt: analyst_cfg.system_prompt,
        plugins: plugins_analyst,
        files: analyst_cfg.files,
        ledger: Some(Arc::clone(ledger)),
    };
    let analyst_in =
        analyst::AnalystInput::fresh(issue.to_string(), scout_out.clone(), memory_out.clone());
    let analyst_input_json = serde_json::to_string(&analyst_in)?;
    let analyst_out = analyst
        .run(analyst_in, &state)
        .await
        .map_err(|e| PlanError::node("analyst", e))?;
    record(Record {
        store,
        run_id,
        node: "analyst",
        output: &analyst_out,
        input: Some(analyst_input_json),
        iteration: None,
        ledger: Some(ledger),
    })
    .await?;
    state.analysis = Some(serde_json::to_value(&analyst_out)?);

    state.status = RunStatus::Planned;
    Ok(PlanOutcome {
        state,
        scout: scout_out,
        memory: memory_out,
        analyst: analyst_out,
    })
}

/// One checkpoint to write: which node, what it produced, and — for a node that ran a model — what
/// it was given and what the turn cost.
///
/// `input` and `ledger` are optional because not every checkpoint has them: the `issue` row is the
/// run's own input rather than a node's, and the implementer drives a coding CLI that reports no
/// token usage. A missing value here means "there was none", never "we forgot to look".
struct Record<'a, T> {
    store: &'a Store,
    run_id: &'a str,
    node: &'a str,
    output: &'a T,
    /// What the node was given, already serialized. Without it a checkpoint shows what came out
    /// with no way to ask why, which is the difference between a log and something a run can be
    /// replayed from. Serialized by the caller because each node's input is a different type and
    /// erasing it behind a trait object buys nothing but a `Send + Sync` bound to satisfy.
    input: Option<String>,
    /// Which pass of the converge loop this is; `None` for a node that runs once.
    iteration: Option<u32>,
    ledger: Option<&'a Arc<RunLedger>>,
}

async fn record<T: Serialize>(r: Record<'_, T>) -> Result<(), PlanError> {
    let json = serde_json::to_string(r.output)?;
    let input_json = r.input;
    // Claimed rather than borrowed: the ledger holds one entry per model turn, and taking it here
    // is what keeps the converge loop's repeated implementer turns matched to their own rows.
    let telemetry = r.ledger.and_then(|l| l.take(r.node)).unwrap_or_default();
    r.store
        .insert_checkpoint(ratatoskr_store::CheckpointWrite {
            run_id: r.run_id,
            node_name: r.node,
            output_json: &json,
            input_json: input_json.as_deref(),
            iteration: r.iteration,
            telemetry,
        })
        .await?;
    // The third structured event: a node produced output. Tool calls and model text come from
    // the agent's observability hook; this is what says a node actually finished.
    tracing::info!(
        kind = "checkpoint",
        node = r.node,
        bytes = json.len(),
        "checkpoint"
    );
    Ok(())
}

/// Record what it would take to say two runs were the same experiment: the resolved config, a
/// fingerprint of the graph that ran, and the commit it ran against.
///
/// Best-effort throughout. This is what makes runs comparable afterwards, which is never worth
/// failing a run over — a run with no provenance is still a run, and one refused because `git` was
/// slow is not.
async fn record_provenance(store: &Store, run_id: &str, config: &RatatoskrConfig) {
    let config_json = serde_json::to_string(config)
        .inspect_err(|e| tracing::warn!("could not record the run's config: {e}"))
        .ok();
    let repo = std::env::current_dir().unwrap_or_default();
    let repo_sha = ratatoskr_exec::head_sha(&repo).await.ok();
    if let Err(e) = store
        .record_run_provenance(
            run_id,
            config_json.as_deref(),
            Some(&graph_fingerprint(&repo)),
            repo_sha.as_deref(),
        )
        .await
    {
        tracing::warn!("could not record run provenance: {e}");
    }
}

/// A fingerprint of the orchestration that ran: every workflow and every ruleset, in a fixed order.
///
/// Deliberately not a cryptographic digest and deliberately not `DefaultHasher`. Nothing here
/// defends against a forged match — it answers "did the graph change between these two runs", and
/// for that it only has to be stable across processes and releases. `DefaultHasher` guarantees
/// neither, so a stored value would silently stop matching on a toolchain bump; FNV-1a is fixed
/// because it is written here.
fn graph_fingerprint(repo: &std::path::Path) -> String {
    let scripts_in = |dir: PathBuf| -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x == "ts"))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut sources = scripts_in(repo.join(".ratatoskr/rules"));
    // Every workflow, not just the one this run used: which workflows exist is part of what the
    // graph *is* now, and two runs cannot be compared across a registry that changed under them.
    sources.extend(scripts_in(repo.join(WORKFLOW_DIR)));
    // Sorted, because `read_dir` order is the filesystem's business and a fingerprint that depends
    // on it would differ between two checkouts of identical files.
    sources.sort();
    sources.insert(0, repo.join(LEGACY_WORKFLOW));

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for path in sources {
        // A missing file still contributes its name, so adding a `workflow.ts` changes the
        // fingerprint even if it is empty.
        for byte in path
            .file_name()
            .map(|n| n.as_encoded_bytes().to_vec())
            .unwrap_or_default()
            .iter()
            .chain(std::fs::read(&path).unwrap_or_default().iter())
        {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

/// A checkpoint with nothing to measure: no recorded input, no model turn behind it.
async fn checkpoint<T: Serialize>(
    store: &Store,
    run_id: &str,
    node: &str,
    output: &T,
) -> Result<(), PlanError> {
    record(Record {
        store,
        run_id,
        node,
        output,
        input: None,
        iteration: None,
        ledger: None,
    })
    .await
}

/// Where a repo keeps the workflows it defines.
pub const WORKFLOW_DIR: &str = ".ratatoskr/workflows";

/// The name of the flow this binary implements in Rust.
pub const BUILT_IN: &str = "built-in";

/// One workflow a run can use.
///
/// The built-in is not a script and deliberately is not going to become one. Its gates — the
/// referee check, the verifier and the analyst re-entry it routes findings to, the frozen
/// acceptance — live in `fork_and_converge`, and the scripted path does not have them. Rewriting it
/// as a script would register it in the same list at the cost of the review gate, which is the
/// opposite trade to the one worth making.
pub enum Workflow {
    /// scout → memory → analyst → (red-team ∥ implementer) → verify → converge → bookkeeper.
    BuiltIn,
    Scripted(WorkflowRuntime),
}

impl Workflow {
    pub fn name(&self) -> &str {
        match self {
            Workflow::BuiltIn => BUILT_IN,
            Workflow::Scripted(w) => &w.meta().name,
        }
    }

    /// What it is for, for whatever is choosing.
    pub fn purpose(&self) -> &str {
        match self {
            Workflow::BuiltIn => {
                "Plan a change, implement it in an isolated worktree, and iterate until it passes \
                 the acceptance check and survives review."
            }
            Workflow::Scripted(w) => &w.meta().purpose,
        }
    }
}

/// The single-script path, still honoured so a repo that has one keeps working untouched.
const LEGACY_WORKFLOW: &str = ".ratatoskr/workflow.ts";

/// Every workflow a run could use: the built-in, then whatever this repo defines.
pub async fn registry() -> Result<Vec<Workflow>, PlanError> {
    let mut all = vec![Workflow::BuiltIn];
    all.extend(defined().await?.into_iter().map(Workflow::Scripted));
    Ok(all)
}

/// The workflows this repo defines, in name order.
///
/// `.ratatoskr/workflows/*.ts` first, then a bare `.ratatoskr/workflow.ts` if one is there. Both,
/// rather than one superseding the other: a repo that has the old file gets it as an ordinary
/// entry in the registry rather than a migration to perform before anything runs.
pub async fn defined() -> Result<Vec<WorkflowRuntime>, PlanError> {
    let fail = |e: ratatoskr_script::ScriptError| {
        PlanError::node("workflow", NodeError::Failed(e.to_string()))
    };
    let mut found = WorkflowRuntime::discover(std::path::Path::new(WORKFLOW_DIR))
        .await
        .map_err(fail)?;
    if let Some(legacy) = WorkflowRuntime::load(std::path::Path::new(LEGACY_WORKFLOW))
        .await
        .map_err(fail)?
    {
        // Only when the directory has not already claimed that name, so a repo mid-move does not
        // get a duplicate-name error for a file it has already copied across.
        if !found.iter().any(|w| w.meta().name == legacy.meta().name) {
            found.push(legacy);
        }
    }
    Ok(found)
}

/// Pick the workflow a run should use.
///
/// `wanted` names one and fails if it is not there — a run that asked for a specific shape must not
/// quietly get a different one, and the error lists what it could have asked for.
///
/// With nothing named, the *defined* workflows decide: none means the built-in, exactly one means
/// that one, and several decline to guess. The built-in is deliberately not counted in that tally.
/// It is always available, so counting it would make a repo with a single script ambiguous, and
/// asking a repo to name a workflow it has only one of is a question with one answer.
pub fn select(found: Vec<Workflow>, wanted: Option<&str>) -> Result<Workflow, PlanError> {
    let listed = found
        .iter()
        .map(|w| w.name().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if let Some(name) = wanted {
        return found.into_iter().find(|w| w.name() == name).ok_or_else(|| {
            PlanError::node(
                "workflow",
                NodeError::Failed(format!("no workflow named `{name}`; available: {listed}")),
            )
        });
    }

    let mut defined: Vec<Workflow> = found
        .into_iter()
        .filter(|w| !matches!(w, Workflow::BuiltIn))
        .collect();
    match defined.len() {
        0 => Ok(Workflow::BuiltIn),
        1 => Ok(defined.remove(0)),
        _ => Err(PlanError::node(
            "workflow",
            NodeError::Failed(format!(
                "this repo defines several workflows; name one with --workflow. Available: {listed}"
            )),
        )),
    }
}

/// What one run needs to start.
///
/// A struct rather than a seventh positional argument: these travel together through both entry
/// points, and every one of them is a borrow of something the caller already holds, so a positional
/// list grows with the run rather than with the job.
pub struct RunRequest<'a> {
    pub client: &'a RagRatClient,
    pub config: &'a RatatoskrConfig,
    pub store: &'a Store,
    pub run_id: &'a str,
    pub issue: &'a str,
    pub engine: &'a Arc<ScriptEngine>,
    /// Which workflow to run, when the caller knows. `None` uses the repo's only one, or the
    /// built-in flow when it defines none — and refuses to guess when it defines several.
    pub workflow: Option<&'a str>,
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
    tools: ToolSet,
    /// The repository the node's built-in file tools read within.
    files: Option<PathBuf>,
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
    /// Where this run's nodes report what their turns cost, drained as each checkpoint is written.
    ledger: &'a Arc<RunLedger>,
}

/// What each plugin contributed for this run.
///
/// Hooks run once per run — `SessionStart` describes the repository, not the node — and each node
/// then composes its own context from the plugins its ruleset binds. Per-node binding therefore
/// costs nothing extra in hook executions.
#[derive(Clone, Default)]
pub struct PluginContext {
    contexts: std::collections::BTreeMap<String, String>,
    /// Every plugin found, which is what a node inherits when no ruleset narrows it.
    discovered: Vec<String>,
    engine: Option<Arc<ScriptEngine>>,
    /// The MCP servers the discovered plugins declare, connected once and shared by every node
    /// that binds the plugin. Held for the run: dropping a connection kills its subprocess.
    servers: Arc<Vec<PluginServer>>,
    /// The loaded plugins themselves, for the hooks that run around a node's tool calls.
    plugins: Arc<Vec<ratatoskr_plugin::Plugin>>,
    /// Wall-clock the run has spent inside tool hooks, shared by every node so the budget is the
    /// run's rather than each node's.
    hook_time: Arc<std::sync::atomic::AtomicU64>,
    /// What this repo lets its plugins' hooks spend.
    limits: ratatoskr_core::HookLimits,
}

/// What the plugins a node binds contribute to that node.
///
/// One value rather than a field per contribution: a node's plugins are resolved in one place and
/// travel together, and every new thing plugins can give a node would otherwise mean another
/// parameter threaded through every node struct and every construction site.
#[derive(Clone, Default)]
pub struct NodePlugins {
    /// Session context, prefixed to whichever preamble the node runs with.
    pub context: Option<String>,
    /// Runs the node's tool calls past its plugins' `PreToolUse`/`PostToolUse` hooks. `None` when
    /// nothing it binds registers one, so a node that gains nothing pays nothing.
    pub observer: Option<Arc<dyn ratatoskr_agent::PluginHooks>>,
    /// Skills the plugins it binds ship, in binding order.
    pub skills: Vec<ratatoskr_plugin::Skill>,
}

/// Runs one node's bound plugins around each of its tool calls.
///
/// Holds only the plugins that node binds, so the per-node binding that decides its context and
/// its tools decides its hooks too.
struct NodeObserver {
    plugins: Vec<ratatoskr_plugin::Plugin>,
    cwd: PathBuf,
    hook_time: Arc<std::sync::atomic::AtomicU64>,
    limits: ratatoskr_core::HookLimits,
}

impl NodeObserver {
    /// Run one event past this node's plugins.
    fn run<'a>(&'a self, event: ratatoskr_plugin::HookEvent<'a>) -> ratatoskr_agent::Answer<'a> {
        Box::pin(ratatoskr_plugin::run_event(
            &self.plugins,
            event,
            &self.cwd,
            &self.limits,
            &self.hook_time,
        ))
    }
}

impl ratatoskr_agent::PluginHooks for NodeObserver {
    fn starting<'a>(&'a self, node: &'a str) -> ratatoskr_agent::Answer<'a> {
        self.run(ratatoskr_plugin::HookEvent::subagent_start(node))
    }

    fn prompting<'a>(&'a self, prompt: &'a str) -> ratatoskr_agent::Answer<'a> {
        self.run(ratatoskr_plugin::HookEvent::user_prompt_submit(prompt))
    }

    fn before<'a>(&'a self, tool: &'a str, args: &'a str) -> ratatoskr_agent::Answer<'a> {
        self.run(ratatoskr_plugin::HookEvent::pre_tool_use(tool, args))
    }

    fn after<'a>(
        &'a self,
        tool: &'a str,
        args: &'a str,
        result: &'a str,
    ) -> ratatoskr_agent::Answer<'a> {
        self.run(ratatoskr_plugin::HookEvent::post_tool_use(
            tool, args, result,
        ))
    }

    fn finished<'a>(
        &'a self,
        node: &'a str,
        outcome: Result<&'a str, &'a str>,
    ) -> ratatoskr_agent::Answer<'a> {
        Box::pin(async move {
            // Both, because the format has both and a plugin may register either: a node is the
            // subagent that stopped, and the turn that ended. A turn that failed ended as
            // `StopFailure`, which is what that event is for — `Stop` keeps meaning a turn that
            // produced an answer. `SubagentStop` fires either way: the subagent is over.
            let ended = match outcome {
                Ok(last) => ratatoskr_plugin::HookEvent::stop(node, last),
                Err(error) => ratatoskr_plugin::HookEvent::stop_failure(node, error),
            };
            let last = outcome.unwrap_or_else(|error| error);
            let stop = self.run(ended).await;
            let subagent = self
                .run(ratatoskr_plugin::HookEvent::subagent_stop(node, last))
                .await;
            match [stop, subagent].into_iter().flatten().collect::<Vec<_>>() {
                parts if parts.is_empty() => None,
                parts => Some(parts.join("\n\n")),
            }
        })
    }
}

/// The events a run actually has. Every other event a plugin registers describes a session with a
/// person in it, or a lifecycle this host does not have, and is not ours to fire.
const NODE_EVENTS: [&str; 7] = [
    "SubagentStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
    "StopFailure",
    "SubagentStop",
];

/// One connected server, and the plugin that declared it — which is what a node binds, not the
/// server's own name, and what the format names its tools after.
struct PluginServer {
    plugin: String,
    connection: Connection,
}

impl PluginServer {
    /// This server's tools, named the way the format names a plugin's.
    fn offer(&self) -> ServerTools {
        ServerTools {
            prefix: Some(ratatoskr_mcp::qualified_prefix(
                &self.plugin,
                self.connection.origin(),
            )),
            ..self.connection.offer()
        }
    }
}

impl PluginContext {
    /// Discover plugins, run their `SessionStart` hooks, and check that every plugin a ruleset
    /// names actually exists.
    ///
    /// A plugin that is missing, broken, or slow contributes nothing and is logged. A ruleset that
    /// *names* a plugin nobody installed is a different thing — a typo in configuration — and
    /// fails the run rather than silently binding less than its author asked for.
    pub async fn resolve(
        config: &RatatoskrConfig,
        engine: &Arc<ScriptEngine>,
        cwd: &std::path::Path,
    ) -> Result<Self, PlanError> {
        let plugins = ratatoskr_plugin::discover(&config.plugins.search_paths(cwd));
        for plugin in &plugins {
            tracing::info!(plugin = plugin.name, "loaded plugin");
        }

        let missing: Vec<String> = engine
            .declared_plugins()
            .into_iter()
            .filter(|name| !plugins.iter().any(|p| &p.name == name))
            .collect();
        if !missing.is_empty() {
            let known: Vec<&str> = plugins.iter().map(|p| p.name.as_str()).collect();
            return Err(PlanError::node(
                "plugins",
                NodeError::Failed(format!(
                    "ruleset names plugin(s) that were not found: {}; discovered: {}",
                    missing.join(", "),
                    if known.is_empty() {
                        "none".to_string()
                    } else {
                        known.join(", ")
                    }
                )),
            ));
        }

        let discovered: Vec<String> = plugins.iter().map(|p| p.name.clone()).collect();
        let contexts = ratatoskr_plugin::session_start(&plugins, cwd, &config.plugins.hooks).await;
        for (name, text) in &contexts {
            tracing::info!(plugin = name, chars = text.len(), "plugin session context");
        }
        Ok(PluginContext {
            contexts,
            discovered,
            engine: Some(Arc::clone(engine)),
            servers: Arc::new(connect_plugin_servers(&plugins, cwd).await),
            plugins: Arc::new(plugins),
            hook_time: Arc::default(),
            limits: config.plugins.hooks.clone(),
        })
    }

    /// Tell every plugin the run is over, and why.
    ///
    /// Run-level rather than per node, like `SessionStart`: a plugin that keeps state across a
    /// session closes it once, not once per node. Nothing is injected — there is no conversation
    /// left to inject into — so this runs for what a hook *does*.
    pub async fn session_end(&self, reason: &str) {
        let cwd = std::env::current_dir().unwrap_or_default();
        if let Some(unused) = ratatoskr_plugin::run_event(
            &self.plugins,
            ratatoskr_plugin::HookEvent::session_end(reason),
            &cwd,
            &self.limits,
            &self.hook_time,
        )
        .await
        {
            tracing::info!(
                chars = unused.len(),
                "a hook answered at the end of the run; its context has nowhere to go"
            );
        }
    }

    /// What the plugins `node` binds give it: their session context, and a hook runner when any of
    /// them registers one for a tool call.
    pub fn for_node(&self, node: &str) -> NodePlugins {
        let bound = self.bound(node);
        let hooked: Vec<ratatoskr_plugin::Plugin> = self
            .plugins
            .iter()
            .filter(|p| bound.contains(&p.name))
            .filter(|p| {
                p.hooks
                    .iter()
                    .any(|h| NODE_EVENTS.contains(&h.event.as_str()))
            })
            .cloned()
            .collect();
        NodePlugins {
            skills: self
                .plugins
                .iter()
                .filter(|p| bound.contains(&p.name))
                .flat_map(|p| p.skills.iter().cloned())
                .collect(),
            context: ratatoskr_plugin::compose(&self.contexts, &bound, &self.limits),
            // `None` rather than an empty runner: it is what keeps the hook off the agent
            // entirely for a node whose plugins have nothing to say about its tool calls.
            observer: (!hooked.is_empty()).then(|| {
                Arc::new(NodeObserver {
                    plugins: hooked,
                    cwd: std::env::current_dir().unwrap_or_default(),
                    hook_time: Arc::clone(&self.hook_time),
                    limits: self.limits.clone(),
                }) as Arc<dyn ratatoskr_agent::PluginHooks>
            }),
        }
    }

    /// Which plugins `node` binds — its ruleset's declaration, or every plugin found.
    fn bound(&self, node: &str) -> Vec<String> {
        match &self.engine {
            Some(engine) => engine.plugins_for(node, &self.discovered),
            None => Vec::new(),
        }
    }

    /// Every tool `node` may call: rag-rat's catalogue, then the servers its plugins declare.
    ///
    /// rag-rat comes first so it wins any name collision — see [`ToolSet::from_servers`].
    fn pool_for(&self, node: &str, rag_rat: ServerTools) -> ToolSet {
        let bound = self.bound(node);
        let mut servers = vec![rag_rat];
        servers.extend(
            self.servers
                .iter()
                .filter(|s| bound.contains(&s.plugin))
                .map(PluginServer::offer),
        );
        ToolSet::from_servers(servers)
    }
}

/// Connect the MCP servers the discovered plugins declare, once per run.
///
/// A server that will not start costs its plugin's tools and nothing else: a broken plugin must
/// not fail a run.
async fn connect_plugin_servers(
    plugins: &[ratatoskr_plugin::Plugin],
    cwd: &std::path::Path,
) -> Vec<PluginServer> {
    let mut connected = Vec::new();
    for (plugin, spec) in servers_to_start(plugins) {
        match Connection::spawn(&spec.name, &spec.command, &spec.env, Some(cwd)).await {
            Ok(connection) => connected.push(PluginServer {
                plugin: plugin.to_string(),
                connection,
            }),
            Err(e) => tracing::warn!(
                plugin,
                server = spec.name,
                "plugin MCP server unavailable, its tools are not offered: {e}"
            ),
        }
    }
    connected
}

/// Which declared servers actually get started, paired with the plugin that declared each.
///
/// One per server name, and rag-rat's name counts as already taken: the rag-rat plugin declares
/// the very server ratatoskr launched from `[rag_rat]`, and a second copy would pay for another
/// index load to offer the identical tools.
fn servers_to_start(
    plugins: &[ratatoskr_plugin::Plugin],
) -> Vec<(&str, &ratatoskr_plugin::McpServerSpec)> {
    let mut claimed: Vec<&str> = vec![ratatoskr_mcp::RAG_RAT];
    let mut start = Vec::new();
    for plugin in plugins {
        for spec in &plugin.mcp_servers {
            if claimed.contains(&spec.name.as_str()) {
                tracing::info!(
                    plugin = plugin.name,
                    server = spec.name,
                    "MCP server already connected; not starting a second copy"
                );
                continue;
            }
            claimed.push(&spec.name);
            start.push((plugin.name.as_str(), spec));
        }
    }
    start
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
    mut tools: ToolSet,
    node: &str,
    default_tools: &[&str],
    plugins: &NodePlugins,
) -> Result<NodeAgentConfig, PlanError> {
    // Offered before narrowing, so `deny` can take them away: a node reasons about a repository,
    // and reading a file it found is the ordinary case rather than the dangerous one. These are
    // also the names a plugin's hooks are written against — `Read`, `Grep`, `Glob` — which is what
    // makes an unmodified plugin's `PreToolUse` fire for a planning node at all.
    let files = std::env::current_dir().ok();
    if files.is_some() {
        tools
            .local()
            .tools
            .extend(ratatoskr_agent::files::declarations());
    }
    let ruleset = engine.ruleset(node);
    let rc = ruleset.as_ref().map(|r| r.config());

    // Ruleset model FIRST: a node whose ruleset declares one needs no `[models.<node>]` entry.
    // `route()` — and its "add a [models.<node>]" error — is only the fallback.
    let route = match rc.and_then(|c| c.model.as_ref()) {
        Some(m) => ratatoskr_core::ModelRoute {
            provider: m.provider.clone(),
            model: m.model.clone(),
            // A ruleset declares which model, not how much of it. The cap comes from the default,
            // which is always sent — so a ruleset naming a brand-new model still works.
            max_tokens: None,
        },
        None => route(config, node)?,
    };

    // A ruleset's `allow` is exhaustive. The default is not just the node's built-in list: those
    // name rag-rat tools, written before any plugin was in the picture, so a plugin the node binds
    // would otherwise contribute a server whose every tool is filtered straight back out.
    let from_plugins = tools.names_beyond(ratatoskr_mcp::RAG_RAT);
    let spelled_out = rc
        .and_then(|c| c.tools.as_ref())
        .and_then(|t| t.allow.as_deref());
    let allow: Vec<String> = match spelled_out {
        Some(a) => a.to_vec(),
        None => default_allow(default_tools, from_plugins.clone()),
    };
    let deny: Vec<String> = rc
        .and_then(|c| c.tools.as_ref())
        .map(|t| t.deny.clone())
        .unwrap_or_default();

    // Named but nowhere on offer: a typo, or a tool the server stopped exposing. Reported by name
    // — a count can't be acted on, and a `deny` elsewhere in the ruleset must not explain it away.
    let offered = tools.names();
    let missing: Vec<&String> = allow
        .iter()
        .filter(|n| !offered.contains(n) && !deny.contains(n))
        .collect();
    if !missing.is_empty() {
        tracing::warn!(node, ?missing, "no connected MCP server offers these tools");
    }
    // An `allow` written before the plugin was bound is exhaustive too, so it silently excludes
    // every tool the plugin brought — the node gets that plugin's context and none of its reach.
    if spelled_out.is_some() && !from_plugins.is_empty() {
        let excluded: Vec<&String> = from_plugins.iter().filter(|n| !allow.contains(n)).collect();
        if !excluded.is_empty() {
            tracing::warn!(
                node,
                ?excluded,
                "this node's plugins offer tools its ruleset's `allow` does not name; add them, \
                 or unbind the plugin"
            );
        }
    }
    tools.narrow(&allow, &deny);

    let max_turns = rc.and_then(|c| c.max_turns);
    let system_prompt = rc.and_then(|c| c.system_prompt.clone());
    let policy: Option<Arc<dyn ToolPolicy>> = match ruleset {
        Some(r) if r.config().has_on_tool_call => Some(Arc::new(r) as Arc<dyn ToolPolicy>),
        _ => None,
    };

    // Every node reaches this function, which is why the skill tool is added here rather than at
    // each construction site: a node that binds a skill and is never offered it is the failure
    // this seam exists to prevent.
    if let Some(tool) = skills::skill_tool(&plugins.skills) {
        tools.add_local(tool);
    }

    Ok(NodeAgentConfig {
        route,
        tools,
        files,
        policy,
        max_turns,
        system_prompt,
    })
}

/// What a node may call when its ruleset names no tools: its built-in list, plus everything the
/// plugins it binds offer.
///
/// The built-in lists name rag-rat tools and were written before any plugin was in the picture, so
/// on their own they would filter a bound plugin's every tool straight back out — binding a plugin
/// would deliver its session context and none of its capability.
fn default_allow(built_in: &[&str], from_plugins: Vec<String>) -> Vec<String> {
    built_in
        .iter()
        .map(|t| t.to_string())
        .chain(from_plugins)
        .collect()
}

/// Everything a full fork+converge run produced. The worktree is the reviewable deliverable and is
/// left in place on a terminal status (converged or max-iterations); it's removed on a hard error.
pub struct RunOutcome {
    pub state: RunState,
    pub plan: PlanOutcome,
    /// The fork's three products, `None` when the fork never ran because the analyst judged the
    /// task to call for no code change (`RunStatus::NoCodeChange`).
    pub red_team: Option<RedTeamOutput>,
    pub implementer: Option<ImplementerOutput>,
    pub worktree: Option<WorktreePath>,
    pub iterations: u32,
    pub status: RunStatus,
    /// Bookkeeper result — `Some` on a terminal fork outcome (converged, or max-iterations with an
    /// `unresolved`-tagged memory); `None` otherwise.
    pub bookkeeper: Option<BookkeeperOutput>,
}

/// The run's friction, read back from its checkpoints.
///
/// Best-effort: a store read that fails costs the bookkeeper its richest input, and failing the
/// run over it would discard completed work to record less about it.
async fn friction_of(store: &Store, run_id: &str) -> bookkeeper::RunFriction {
    match store.checkpoints_for_run(run_id).await {
        Ok(checkpoints) => bookkeeper::RunFriction::from_checkpoints(&checkpoints),
        Err(e) => {
            tracing::warn!("could not read the run's checkpoints for bookkeeping: {e}");
            bookkeeper::RunFriction::default()
        }
    }
}

/// Whether to run the fork at all.
///
/// The analyst owns this call — it is the node that turns a task into a plan, so it is the one that
/// knows whether carrying the plan out means editing code. `always_fork` is the override for a
/// human who disagrees, and it only ever adds work: nothing can configure the fork *away* when the
/// analyst says a change is needed.
fn fork_is_needed(analyst: &AnalystOutput, config: &RatatoskrConfig) -> bool {
    analyst.changes_code || config.implementer.always_fork
}

/// Finish a run whose analyst judged that carrying out the plan changes no code.
///
/// The plan itself is the artifact, and it is already checkpointed. The bookkeeper is not run:
/// it composes memories from the implementer's diff, and there is none — a research run's learning
/// is worth recording, but doing it means teaching that node to compose from the analyst alone,
/// which is a change to what it produces rather than to when it fires.
async fn no_code_change(
    store: &Store,
    run_id: &str,
    context: &PluginContext,
    plan: PlanOutcome,
) -> Result<RunOutcome, PlanError> {
    tracing::info!(
        kind = "fork_skipped",
        impact = %plan.analyst.impact_summary,
        "the analyst judged this task to need no code change; skipping the fork"
    );
    let status = RunStatus::NoCodeChange;
    if let Err(e) = store.upsert_run(run_id, None, status.as_str()).await {
        tracing::warn!("failed to record the run's final status: {e}");
    }
    context.session_end(status.as_str()).await;

    let mut state = plan.state.clone();
    state.status = status;
    Ok(RunOutcome {
        state,
        plan,
        red_team: None,
        implementer: None,
        worktree: None,
        iterations: 0,
        status,
        bookkeeper: None,
    })
}

/// The full Phase 3 run: plan (scout → memory → analyst), then fork red-team ∥ implementer, then
/// converge. Reuses [`run_plan`] for the planning half.
pub async fn run_full(request: RunRequest<'_>) -> Result<RunOutcome, PlanError> {
    let RunRequest {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        workflow,
    } = request;
    // A workflow, when this repo defines one, overrides the whole run flow.
    if let Workflow::Scripted(runtime) = select(registry().await?, workflow)? {
        // Said out loud because it is a gate the run will not have. The scripted path checkpoints,
        // validates and enforces the referee and iteration limits, but it has no verifier binding
        // — so a change that passes its tests is accepted without anything reading the diff.
        tracing::warn!(
            workflow = runtime.meta().name,
            "this workflow does not run the verifier; the change will be accepted on its tests \
             alone. `--workflow {BUILT_IN}` runs the flow that reviews the diff."
        );
        let plugin_context =
            PluginContext::resolve(config, engine, &std::env::current_dir().unwrap_or_default())
                .await?;
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
        PluginContext::resolve(config, engine, &std::env::current_dir().unwrap_or_default())
            .await?;
    let plan = match plan_half(client, config, store, run_id, issue, engine, &plan_context).await {
        Ok(plan) => plan,
        Err(e) => {
            plan_context.session_end(RunStatus::Failed.as_str()).await;
            return Err(e);
        }
    };

    // Some tasks call for no code change: research, a review, an architecture answer. Running the
    // fork for one costs a sandboxed baseline test run and an ACP session to produce an empty diff,
    // and then reports `Converged` — a success claim about a change that was never made.
    if !fork_is_needed(&plan.analyst, config) {
        return no_code_change(store, run_id, &plan_context, plan).await;
    }
    // `plan.state.clarifications` already holds the plan-half asks; the fork/bookkeep half gets its
    // own clarifier, drained and appended at the end.
    let mut state = plan.state.clone();
    let clarifier = NodeClarifier::new(config, store, engine, run_id, issue);

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

    let ledger = Arc::new(RunLedger::default());
    let run = Run {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        clarifier: &clarifier,
        ledger: &ledger,
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
    plan_context.session_end(status.as_str()).await;

    let (red_team, implementer, worktree, status, iterations) = result?;
    state.red_team = Some(serde_json::to_value(&red_team)?);
    state.implementer = Some(serde_json::to_value(&implementer)?);
    state.status = status;

    // Bookkeeping fires on a terminal fork outcome: `Converged` (record the learning),
    // `MaxIterationsReached` (record the wall, tagged `unresolved`), or `Unreviewed` — a change was
    // still made and its friction is still worth recording, whether or not anyone reviewed it. A
    // bookkeeping failure is logged but doesn't discard the run's work.
    let bookkeeper = if matches!(
        status,
        RunStatus::Converged | RunStatus::MaxIterationsReached | RunStatus::Unreviewed
    ) {
        // Read back what the run's own checkpoints recorded about its path. The same source the
        // `bookkeep` replay reads, so a replay composes from exactly what the live run did.
        let input = BookkeeperInput {
            issue: issue.to_string(),
            analyst: plan.analyst.clone(),
            implementer: implementer.clone(),
            iterations,
            converged: status == RunStatus::Converged,
            friction: friction_of(store, run_id).await,
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
        red_team: Some(red_team),
        implementer: Some(implementer),
        worktree: Some(worktree),
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
        ledger,
        context,
        ..
    } = run;
    let plugins_bookkeeper = context.for_node("bookkeeper");
    let cfg = node_agent_config(
        engine,
        config,
        context.pool_for("bookkeeper", client.offer()),
        "bookkeeper",
        bookkeeper::BOOKKEEPER_TOOLS,
        &plugins_bookkeeper,
    )?;
    let mut tools = cfg.tools;
    tools.add_local(clarify::ask_tool());
    let node = BookkeeperNode {
        route: cfg.route,
        tools,
        sink: client.sink(),
        policy: cfg.policy,
        max_turns: cfg.max_turns,
        clarifier: Some(clarifier.as_dyn()),
        system_prompt: cfg.system_prompt,
        plugins: plugins_bookkeeper,
        files: cfg.files,
        ledger: Some(Arc::clone(ledger)),
    };
    let input_json = serde_json::to_string(&input)?;
    let out = node
        .run(input)
        .await
        .map_err(|e| PlanError::node("bookkeeper", e))?;
    record(Record {
        store,
        run_id,
        node: "bookkeeper",
        output: &out,
        input: Some(input_json),
        iteration: None,
        ledger: Some(run.ledger),
    })
    .await?;
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
    let clarifier = NodeClarifier::new(config, store, engine, run_id, &issue);
    let input = BookkeeperInput {
        issue,
        analyst,
        implementer,
        iterations,
        converged,
        friction: bookkeeper::RunFriction::from_checkpoints(&checkpoints),
    };
    let context =
        PluginContext::resolve(config, engine, &std::env::current_dir().unwrap_or_default())
            .await?;
    let ledger = Arc::new(RunLedger::default());
    let run = Run {
        client,
        config,
        store,
        run_id,
        issue: &input.issue.clone(),
        engine,
        clarifier: &clarifier,
        ledger: &ledger,
        context: &context,
    };
    bookkeep_and_checkpoint(&run, input).await
}

/// The fork + converge half. Returns the terminal status; leaves the worktree in place on a
/// terminal outcome and removes it on a hard error.
/// The review stage: the verifier, plus the analyst re-entry it routes plan-level findings to.
///
/// Built once per run and reused across converge iterations, so a second review costs a model call
/// rather than a rebuild. `None` when the verifier has no route — like the red team's classifier,
/// it is opt-in by being given a model rather than by a separate switch.
struct Review {
    verifier: verifier::VerifierNode,
    threshold: verifier::Severity,
    /// The analyst, kept alive for revisions. It is the principal: it owns the plan, so it is the
    /// only node that can tell "the plan was wrong" from "the code did not follow the plan".
    analyst: AnalystNode,
    scout: ScoutOutput,
    memory: MemoryOutput,
}

/// What a review concluded the run should do next.
///
/// `Unavailable` is the case that is easy to get wrong: a verifier that could not run has not
/// approved anything, but it has not found anything either. Its error is evidence about our
/// infrastructure, not about the change, so it must neither block the run nor pass as a clean
/// review.
enum Reviewed {
    /// Nothing above the threshold. The change is accepted.
    Clean,
    /// Send this back.
    Fix(Correction),
    /// The verifier could not be asked. The reason is on its checkpoint.
    Unavailable,
}

/// What a review concluded the run should do next.
struct Correction {
    /// What to hand the implementer.
    prompt: String,
    /// The amended plan, when the analyst revised one. Kept so later reviews judge the change
    /// against what was actually asked for by the end, not against the plan that was wrong.
    revised: Option<AnalystOutput>,
}

/// Build the acceptance characterizer, when `[models.characterizer]` gives it somewhere to run.
///
/// Optional on purpose: without it a run still converges on exit codes, comparing at step
/// granularity. Coarser than named checks, and never wrong about them — so a repo that has not
/// configured one loses detail, not correctness.
pub(crate) fn build_characterizer(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    context: &PluginContext,
    offer: ServerTools,
) -> Result<Option<testrun::Characterizer>, PlanError> {
    if !config.models.contains_key("characterizer")
        && !engine
            .ruleset("characterizer")
            .is_some_and(|r| r.config().model.is_some())
    {
        return Ok(None);
    }
    let plugins = context.for_node("characterizer");
    // No default tools: it transcribes output it was handed. Reading the repo would invite it to
    // decide whether a failure matters, which is the one thing it must not do.
    let cfg = node_agent_config(
        engine,
        config,
        context.pool_for("characterizer", offer),
        "characterizer",
        &[],
        &plugins,
    )?;
    Ok(Some(testrun::Characterizer {
        route: cfg.route,
        tools: cfg.tools,
        max_turns: cfg.max_turns,
    }))
}

/// The verifier is opt-in on having somewhere to run, the same way the red-team classifier is.
fn verifier_enabled(engine: &Arc<ScriptEngine>, config: &RatatoskrConfig) -> bool {
    config.models.contains_key("verifier")
        || engine
            .ruleset("verifier")
            .is_some_and(|r| r.config().model.is_some())
}

/// Read `[implementer] verify_threshold`. An unrecognised value is a typo, and a typo must not
/// quietly relax a gate — it falls back to the default and says so.
fn parse_threshold(raw: &str) -> verifier::Severity {
    match raw.trim().to_ascii_uppercase().as_str() {
        "P1" => verifier::Severity::P1,
        "P2" => verifier::Severity::P2,
        "P3" => verifier::Severity::P3,
        other => {
            tracing::warn!("unknown verify_threshold {other:?}; using P2");
            verifier::Severity::P2
        }
    }
}

impl Review {
    fn build(run: &Run<'_>, plan: &PlanOutcome) -> Result<Option<Self>, PlanError> {
        let &Run {
            client,
            config,
            engine,
            context,
            ledger,
            ..
        } = run;
        if !verifier_enabled(engine, config) {
            return Ok(None);
        }

        let plugins_verifier = context.for_node("verifier");
        let cfg = node_agent_config(
            engine,
            config,
            context.pool_for("verifier", client.offer()),
            "verifier",
            verifier::VERIFIER_TOOLS,
            &plugins_verifier,
        )?;
        let verifier = verifier::VerifierNode {
            route: cfg.route,
            tools: cfg.tools,
            policy: cfg.policy,
            max_turns: cfg.max_turns,
            system_prompt: cfg.system_prompt,
            plugins: plugins_verifier,
            files: cfg.files,
            ledger: Some(Arc::clone(ledger)),
        };

        let plugins_analyst = context.for_node("analyst");
        let acfg = node_agent_config(
            engine,
            config,
            context.pool_for("analyst", client.offer()),
            "analyst",
            analyst::ANALYST_TOOLS,
            &plugins_analyst,
        )?;
        Ok(Some(Review {
            verifier,
            threshold: parse_threshold(&config.implementer.verify_threshold),
            analyst: AnalystNode {
                route: acfg.route,
                tools: acfg.tools,
                policy: acfg.policy,
                max_turns: acfg.max_turns,
                system_prompt: acfg.system_prompt,
                plugins: plugins_analyst,
                files: acfg.files,
                ledger: Some(Arc::clone(ledger)),
            },
            scout: plan.scout.clone(),
            memory: plan.memory.clone(),
        }))
    }

    /// Review the change.
    async fn review(
        &self,
        run: &Run<'_>,
        plan: &AnalystOutput,
        impl_out: &ImplementerOutput,
        worktree: &WorktreePath,
        iteration: u32,
    ) -> Result<Reviewed, PlanError> {
        let &Run {
            store,
            run_id,
            issue,
            ledger,
            ..
        } = run;

        // The patch, not the `--stat` the implementer records: a summary cannot show a weakened
        // assertion, and that is one of the things this stage exists to catch.
        let diff = ratatoskr_exec::diff_text(worktree)
            .await
            .unwrap_or_default();
        let input = verifier::VerifierInput {
            issue: issue.to_string(),
            analyst: plan.clone(),
            diff,
            touched_files: impl_out.touched_files.clone(),
        };
        let input_json = serde_json::to_string(&serde_json::json!({
            "requirements": plan.requirements,
            "touched_files": input.touched_files,
            "diff_bytes": input.diff.len(),
        }))?;
        // A verifier that cannot run must not discard a change that was made and passed. Every
        // other fallible node here is best-effort for the same reason; this one propagating its
        // error was an oversight, and the run it cost had already implemented the task correctly.
        let out = match self.verifier.run(input).await {
            Ok(out) => out,
            Err(e) => {
                tracing::warn!("the verifier could not review this change: {e}");
                record(Record {
                    store,
                    run_id,
                    node: "verifier",
                    output: &serde_json::json!({ "error": e.to_string() }),
                    input: Some(input_json),
                    iteration: Some(iteration),
                    ledger: Some(ledger),
                })
                .await?;
                return Ok(Reviewed::Unavailable);
            }
        };
        record(Record {
            store,
            run_id,
            node: "verifier",
            output: &out,
            // The diff itself is not recorded: it is reproducible from the worktree, and a copy of
            // it in every checkpoint would dwarf everything else in the store.
            input: Some(input_json),
            iteration: Some(iteration),
            ledger: Some(ledger),
        })
        .await?;

        let blocking = out.blocking(self.threshold);
        if blocking.is_empty() {
            return Ok(Reviewed::Clean);
        }
        // Findings below the threshold were still recorded above; say what was set aside so a
        // reader of the logs does not read "2 findings" as "2 problems being fixed".
        tracing::info!(
            blocking = blocking.len(),
            total = out.findings.len(),
            "the review found problems the tests did not catch"
        );

        // Anything the verifier judged a fault in the PLAN goes to the analyst first. Sending it
        // to the implementer instead would re-drive it against a requirement already shown to be
        // wrong, which is the loop this stage exists to break.
        let plan_faults: Vec<verifier::Finding> = blocking
            .iter()
            .filter(|f| f.kind == verifier::FindingKind::Plan)
            .map(|f| (*f).clone())
            .collect();
        if plan_faults.is_empty() {
            return Ok(Reviewed::Fix(Correction {
                prompt: verifier::correction(&blocking),
                revised: None,
            }));
        }

        let revision = analyst::AnalystInput {
            issue: issue.to_string(),
            scout: self.scout.clone(),
            memory: self.memory.clone(),
            previous: Some(Box::new(plan.clone())),
            findings: plan_faults,
        };
        let revision_json = serde_json::to_string(&revision)?;
        let revised = self
            .analyst
            .run(revision, &RunState::new(run_id, None))
            .await
            .map_err(|e| PlanError::node("analyst", e))?;
        record(Record {
            store,
            run_id,
            node: "analyst",
            output: &revised,
            input: Some(revision_json),
            iteration: Some(iteration),
            ledger: Some(ledger),
        })
        .await?;

        Ok(Reviewed::Fix(Correction {
            prompt: replan(&revised, &blocking),
            revised: Some(revised),
        }))
    }
}

/// What the implementer is told after the plan itself was amended.
///
/// The revised requirements come first and the findings after, because the implementer's job is
/// now to satisfy the new plan — the findings are why it changed, not a separate list of fixes.
fn replan(revised: &AnalystOutput, findings: &[&verifier::Finding]) -> String {
    use std::fmt::Write as _;
    let mut s = String::from(
        "A review of your change found that the PLAN was wrong, not just the code. The \
         requirements have been amended. Bring your change in line with them:\n\n",
    );
    for r in &revised.requirements {
        let _ = writeln!(s, "- {r}");
    }
    if !revised.impact_summary.is_empty() {
        let _ = write!(
            s,
            "\nWhat this is meant to achieve:\n{}\n",
            revised.impact_summary
        );
    }
    let _ = write!(s, "\nWhat the review found, for context:\n");
    for f in findings {
        let _ = writeln!(s, "- [{:?}] {}", f.severity, f.summary);
        let _ = writeln!(s, "  Fails when: {}", f.failure_scenario);
    }
    s
}

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
        ledger,
        context,
    } = run;
    let repo_path: PathBuf = std::env::current_dir()
        .map_err(|e| PlanError::node("fork", NodeError::Failed(format!("cwd: {e}"))))?;
    let short: String = run_id.chars().take(8).collect();
    // Resolved once and shared by both branches: the baseline and the post-change run must execute
    // the same steps, or the two sets converge compares are not comparable. Frozen here for the
    // whole run — a later analyst revision amends requirements, never the bar.
    let acceptance = config.sandbox.acceptance(&plan.analyst.acceptance);
    tracing::info!(
        steps = ?acceptance.iter().map(|s| &s.name).collect::<Vec<_>>(),
        "acceptance for this run"
    );

    let red_team = RedTeamNode {
        repo_path: repo_path.clone(),
        sandbox: config.sandbox.clone(),
        name: format!("ratatoskr-redteam-{short}"),
        // Opt-in: classify baseline failures only when redteam has a route — from
        // `[models.redteam]` or from its `.ratatoskr/rules/redteam.ts` ruleset.
        acceptance: acceptance.clone(),
        characterizer: build_characterizer(engine, config, context, client.offer())?,
        classifier: match classifier_enabled(engine, config) {
            true => {
                let plugins_redteam = context.for_node("redteam");
                let cfg = node_agent_config(
                    engine,
                    config,
                    context.pool_for("redteam", client.offer()),
                    "redteam",
                    redteam::CLASSIFIER_TOOLS,
                    &plugins_redteam,
                )?;
                let mut tools = cfg.tools;
                tools.add_local(clarify::ask_tool());
                Some(redteam::RedTeamClassifier {
                    route: cfg.route,
                    tools,
                    policy: cfg.policy,
                    max_turns: cfg.max_turns,
                    clarifier: Some(clarifier.as_dyn()),
                    system_prompt: cfg.system_prompt,
                    plugins: plugins_redteam,
                    files: cfg.files,
                    ledger: Some(Arc::clone(ledger)),
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
        acceptance,
        characterizer: build_characterizer(engine, config, context, client.offer())?,
    };

    // Built before the fork so a misconfigured verifier fails the run here rather than after an
    // ACP session and a sandboxed test run have already been spent on it.
    let review = Review::build(run, plan)?;

    // Fork: both branches run concurrently off the same frozen post-analyst state. join! (not
    // spawn) because both are I/O-bound (subprocess/sandbox) and borrow their nodes.
    let (rt_res, impl_res) = tokio::join!(red_team.run(), implementer.run());

    let red_team_out = rt_res.map_err(|e| PlanError::node("red_team", e))?;
    let (worktree, mut impl_out) = impl_res.map_err(|e| PlanError::node("implementer", e))?;

    record(Record {
        store,
        run_id,
        node: "red_team",
        output: &red_team_out,
        input: None,
        iteration: Some(1),
        ledger: Some(ledger),
    })
    .await?;
    record(Record {
        store,
        run_id,
        node: "implementer",
        output: &impl_out,
        // The implementer drives a coding CLI over ACP rather than a model turn here, so the
        // ledger has nothing for it; its row carries the iteration and the outcome.
        input: Some(serde_json::to_string(&plan.analyst)?),
        iteration: Some(1),
        ledger: Some(ledger),
    })
    .await?;

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
                "the baseline acceptance run produced no checks (exit {}); \
                 check the analyst's acceptance, [sandbox] test_command and the sandbox backend",
                red_team_out.exit_code
            )),
        ));
    }

    // Converge: iterate the implementer (not red-team — the baseline doesn't change) until the
    // change both passes the tests and survives review, or the budget runs out.
    // The plan in force. A revision replaces it, so a later review judges the change against what
    // was actually asked for by the end rather than against the requirement that was wrong.
    let mut in_force = plan.analyst.clone();
    let mut iterations = 1u32;
    let status = loop {
        let post_ran = converge::test_command_ran(
            &impl_out.failing_tests,
            &impl_out.passing_tests,
            impl_out.exit_code,
        );
        let tests_clean = post_ran
            && converge::is_converged(&red_team_out.failing_tests, &impl_out.failing_tests);

        // Did the change edit the referee? Checked BEFORE `tests_clean` is trusted: a conftest.py
        // that rewrites every outcome, or an edited test, makes the passing/failing sets describe a
        // bar the change wrote for itself.
        let referee = converge::referee_touches(&impl_out.touched_files, engine.may_modify_tests());

        // What to do next. The referee check comes first, then the test gate, then the review: a
        // moved referee makes the test result meaningless, and a test result is stronger evidence
        // than a model's judgement, so reviewing a change that does not build wastes the call.
        let correction: Reviewed = if !referee.is_empty() {
            tracing::warn!(files = ?referee, "iteration touched the referee; not accepting it");
            Reviewed::Fix(Correction {
                prompt: converge::referee_correction(&referee),
                revised: None,
            })
        } else if !tests_clean {
            // A post-change run that didn't complete usually means the edit broke the build — say
            // that specifically instead of reporting "no new failures".
            let prompt = if !post_ran {
                format!(
                    "The test command did not run to completion (exit {}) — your change likely \
                     does not compile. Fix it so the tests run and pass.",
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
            Reviewed::Fix(Correction {
                prompt,
                revised: None,
            })
        } else if let Some(review) = &review {
            review
                .review(run, &in_force, &impl_out, &worktree, iterations)
                .await?
        } else {
            Reviewed::Clean
        };

        let correction = match correction {
            Reviewed::Clean => break RunStatus::Converged,
            // The change passed its tests and nobody was able to review it. Saying `Converged`
            // would claim a review that did not happen; failing would discard work that did.
            Reviewed::Unavailable => break RunStatus::Unreviewed,
            Reviewed::Fix(correction) => correction,
        };
        if let Some(revised) = correction.revised {
            in_force = revised;
        }
        if iterations >= config.implementer.max_iterations {
            break RunStatus::MaxIterationsReached;
        }
        impl_out = match implementer.iterate(&worktree, &correction.prompt).await {
            Ok(out) => out,
            Err(e) => {
                // Hard error mid-converge: don't leave the worktree behind.
                if let Err(rm) = remove_worktree(&repo_path, &worktree).await {
                    tracing::warn!("failed to clean up worktree after converge error: {rm}");
                }
                return Err(PlanError::node("implementer", e));
            }
        };
        record(Record {
            store,
            run_id,
            node: "implementer",
            output: &impl_out,
            // The correction is what this iteration was actually given — the thing that explains
            // why it did what it did, and the one input a replay would need.
            input: Some(serde_json::to_string(&correction.prompt)?),
            iteration: Some(iterations + 1),
            ledger: Some(ledger),
        })
        .await?;
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

    /// A ruleset directory built for one test.
    async fn binding_engine(case: &str, source: &str) -> Arc<ScriptEngine> {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-nodes-binding-{}-{case}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("agents.ts"), source).unwrap();
        ScriptEngine::load(&dir).await.unwrap()
    }

    /// A plugin directory whose `PreToolUse` hook answers with an envelope for `matcher`.
    fn hooking_plugin(name: &str, matcher: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("ratatoskr-node-hook-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".claude-plugin")).unwrap();
        std::fs::create_dir_all(root.join("hooks")).unwrap();
        std::fs::write(
            root.join(".claude-plugin/plugin.json"),
            format!(r#"{{"name": "{name}"}}"#),
        )
        .unwrap();
        std::fs::write(
            root.join("hooks/hooks.json"),
            format!(
                r#"{{"hooks": {{"PreToolUse": [{{"matcher": "{matcher}", "hooks": [
                    {{"type": "command", "command": "cat ${{CLAUDE_PLUGIN_ROOT}}/answer"}}
                ]}}]}}}}"#
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("answer"),
            format!(r#"{{"hookSpecificOutput": {{"additionalContext": "from {name}"}}}}"#),
        )
        .unwrap();
        root
    }

    #[tokio::test]
    async fn a_node_runs_the_tool_hooks_of_the_plugins_it_binds_and_no_others() {
        // The same binding that decides a node's context and its tools decides its hooks.
        let engine = binding_engine(
            "node-hooks",
            r#"
            defineDefaults({ plugins: ["hookful"] });
            defineAgent("analyst", { plugins: { inherit: false } });
            "#,
        )
        .await;
        let hookful = hooking_plugin("hookful", "^semantic_search$");
        let quiet = hooking_plugin("quiet", "^nothing_calls_this$");
        let plugins = ratatoskr_plugin::discover(&[hookful.clone(), quiet.clone()]);

        let context = PluginContext {
            discovered: plugins.iter().map(|p| p.name.clone()).collect(),
            plugins: Arc::new(plugins),
            engine: Some(engine),
            ..Default::default()
        };

        let scout = context.for_node("scout").observer.expect("scout binds it");
        assert_eq!(
            scout.before("semantic_search", "{}").await.as_deref(),
            Some("from hookful")
        );
        // The matcher still decides which calls it sees, and a PreToolUse hook says nothing after.
        assert_eq!(scout.before("impact_surface", "{}").await, None);
        assert_eq!(scout.after("semantic_search", "{}", "result").await, None);

        // A node that binds nothing carries no runner at all, so the hook never reaches its agent.
        assert!(context.for_node("analyst").observer.is_none());

        let _ = std::fs::remove_dir_all(&hookful);
        let _ = std::fs::remove_dir_all(&quiet);
    }

    #[tokio::test]
    async fn a_node_only_carries_the_context_of_the_plugins_it_binds() {
        // The hooks run once per run; what differs per node is which of their outputs it carries.
        let engine = binding_engine(
            "per-node",
            r#"
            defineDefaults({ plugins: ["everywhere"] });
            defineAgent("analyst", { plugins: { add: ["analyst-only"] } });
            defineAgent("scout", { plugins: { inherit: false } });
            "#,
        )
        .await;

        let context = PluginContext {
            contexts: [
                ("everywhere".to_string(), "SHARED".to_string()),
                ("analyst-only".to_string(), "DEEP".to_string()),
            ]
            .into_iter()
            .collect(),
            discovered: vec!["everywhere".to_string(), "analyst-only".to_string()],
            engine: Some(engine),
            ..Default::default()
        };

        // Defaults first, then what the node added.
        assert_eq!(
            context.for_node("analyst").context.as_deref(),
            Some("SHARED\n\nDEEP")
        );
        // A node that inherits nothing and adds nothing carries nothing.
        assert_eq!(context.for_node("scout").context, None);
        // A node with no ruleset still gets the defaults.
        assert_eq!(
            context.for_node("bookkeeper").context.as_deref(),
            Some("SHARED")
        );
    }

    /// A plugin declaring one MCP server, for the start-decision tests.
    fn plugin_with_server(name: &str, server: &str) -> ratatoskr_plugin::Plugin {
        ratatoskr_plugin::Plugin {
            name: name.to_string(),
            root: PathBuf::from("/nonexistent"),
            hooks: Vec::new(),
            skills: Vec::new(),
            mcp_servers: vec![ratatoskr_plugin::McpServerSpec {
                name: server.to_string(),
                command: vec!["true".to_string()],
                env: Default::default(),
            }],
        }
    }

    #[test]
    fn a_server_is_started_once_and_never_a_second_rag_rat() {
        // The rag-rat plugin declares the very server `[rag_rat]` already launched. Two plugins
        // naming one server is the same situation a level out: connect it once.
        let plugins = [
            plugin_with_server("rag-rat", "rag-rat"),
            plugin_with_server("linty", "lint"),
            plugin_with_server("also-linty", "lint"),
            plugin_with_server("fresh", "fresh"),
        ];

        assert_eq!(
            servers_to_start(&plugins)
                .into_iter()
                .map(|(plugin, spec)| (plugin, spec.name.as_str()))
                .collect::<Vec<_>>(),
            [("linty", "lint"), ("fresh", "fresh")]
        );
    }

    #[test]
    fn a_bound_plugins_tools_are_offered_without_being_named() {
        // The built-in lists name rag-rat tools; a plugin's are only known once its server has
        // answered, so they join the default rather than having to be listed.
        assert_eq!(
            default_allow(&["semantic_search"], vec!["lint".to_string()]),
            ["semantic_search", "lint"]
        );
        // Nothing bound, nothing added.
        assert_eq!(
            default_allow(&["semantic_search"], vec![]),
            ["semantic_search"]
        );
    }

    #[tokio::test]
    async fn a_context_with_no_engine_composes_nothing() {
        // The default value is what a run gets before plugins are resolved; it must not panic.
        assert_eq!(PluginContext::default().for_node("scout").context, None);
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

        let cfg = node_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            "scout",
            &[],
            &NodePlugins::default(),
        )
        .unwrap();
        assert_eq!(cfg.route.provider, "openai");
        assert_eq!(cfg.route.model, "gpt-5");
        assert_eq!(cfg.system_prompt.as_deref(), Some("Be brief."));
    }

    #[tokio::test]
    async fn a_ruleset_without_a_model_still_falls_back_to_toml() {
        let engine = engine("toml-fallback").await;
        let config = RatatoskrConfig::default();

        let cfg = node_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            "bookkeeper",
            &[],
            &NodePlugins::default(),
        )
        .unwrap();
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
            node_agent_config(
                &engine,
                &config,
                ToolSet::default(),
                "analyst",
                &[],
                &NodePlugins::default(),
            ),
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
                max_tokens: None,
            },
        );
        assert!(classifier_enabled(&engine, &config));
    }

    #[test]
    fn the_graph_fingerprint_tracks_the_scripts_and_nothing_else() {
        let root = std::env::temp_dir().join(format!("ratatoskr-fp-{}", std::process::id()));
        let rules = root.join(".ratatoskr/rules");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&rules).unwrap();
        std::fs::write(rules.join("scout.ts"), "a").unwrap();
        std::fs::write(rules.join("analyst.ts"), "b").unwrap();

        let base = graph_fingerprint(&root);
        assert_eq!(
            base,
            graph_fingerprint(&root),
            "same inputs, same fingerprint"
        );

        // A file the fingerprint doesn't cover must not move it, or every run of an unchanged
        // graph would look like a different experiment.
        std::fs::write(root.join(".ratatoskr/notes.md"), "irrelevant").unwrap();
        assert_eq!(base, graph_fingerprint(&root));

        // Editing a ruleset changes the graph.
        std::fs::write(rules.join("scout.ts"), "a2").unwrap();
        let edited = graph_fingerprint(&root);
        assert_ne!(base, edited);

        // So does adding one — even one whose contents match an existing file, because the name
        // is folded in alongside the bytes.
        std::fs::write(rules.join("redteam.ts"), "a2").unwrap();
        assert_ne!(edited, graph_fingerprint(&root));

        // And so does introducing a workflow script where there was none.
        let with_rules = graph_fingerprint(&root);
        std::fs::write(root.join(".ratatoskr/workflow.ts"), "x").unwrap();
        assert_ne!(with_rules, graph_fingerprint(&root));

        let _ = std::fs::remove_dir_all(&root);
    }

    fn analyst_saying(changes_code: bool) -> AnalystOutput {
        AnalystOutput {
            impact_summary: "impact".into(),
            touched: vec!["a.rs".into()],
            risks: Vec::new(),
            requirements: Vec::new(),
            residual_risk: String::new(),
            changes_code,
            acceptance: Vec::new(),
        }
    }

    #[test]
    fn the_fork_runs_when_the_plan_changes_code_and_when_a_human_insists() {
        let mut config = RatatoskrConfig::default();
        assert!(fork_is_needed(&analyst_saying(true), &config));
        assert!(
            !fork_is_needed(&analyst_saying(false), &config),
            "a task that changes no code does not pay for a baseline test run and an ACP session"
        );

        // The override only ever adds work. There is no configuration that skips the fork when the
        // analyst says a change is needed — that would drop the work silently.
        config.implementer.always_fork = true;
        assert!(fork_is_needed(&analyst_saying(false), &config));
        assert!(fork_is_needed(&analyst_saying(true), &config));
    }

    #[test]
    fn a_misspelled_threshold_does_not_quietly_relax_the_gate() {
        use verifier::Severity;
        assert_eq!(parse_threshold("P1"), Severity::P1);
        assert_eq!(parse_threshold("p2"), Severity::P2);
        assert_eq!(parse_threshold(" P3 "), Severity::P3);
        // The dangerous direction: a typo must not read as "block on nothing". It falls back to
        // the default, which is stricter than P1, and warns.
        assert_eq!(parse_threshold("critical"), Severity::P2);
        assert_eq!(parse_threshold(""), Severity::P2);
    }

    #[test]
    fn the_replan_prompt_leads_with_the_amended_requirements() {
        let revised = AnalystOutput {
            impact_summary: "narrower than before".into(),
            touched: Vec::new(),
            risks: Vec::new(),
            requirements: vec!["only handle the utf-8 case".into()],
            residual_risk: String::new(),
            changes_code: true,
            acceptance: Vec::new(),
        };
        let finding = verifier::Finding {
            severity: verifier::Severity::P1,
            kind: verifier::FindingKind::Plan,
            file: "a.rs".into(),
            line: None,
            summary: "the requirement asked for something impossible".into(),
            failure_scenario: "any non-utf-8 path".into(),
        };
        let prompt = replan(&revised, &[&finding]);

        // The implementer's job is now to satisfy the new plan; the findings explain why it
        // changed, so the requirements must come first.
        let req = prompt.find("only handle the utf-8 case").unwrap();
        let why = prompt
            .find("the requirement asked for something impossible")
            .unwrap();
        assert!(req < why, "requirements lead, findings follow:\n{prompt}");
        assert!(prompt.contains("any non-utf-8 path"));
    }

    /// The registry a repo defining `named` would have: the built-in plus those.
    async fn registry_of(case: &str, named: &[&str]) -> Vec<Workflow> {
        let dir = std::env::temp_dir().join(format!("ratatoskr-sel-{}-{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in named {
            std::fs::write(
                dir.join(format!("{name}.ts")),
                format!(r#"defineWorkflow({{ name: "{name}" }});"#),
            )
            .unwrap();
        }
        let found = WorkflowRuntime::discover(&dir).await.unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let mut all = vec![Workflow::BuiltIn];
        all.extend(found.into_iter().map(Workflow::Scripted));
        all
    }

    #[tokio::test]
    async fn naming_a_workflow_that_is_not_there_fails_and_says_what_is() {
        // A run that asked for a specific shape must not quietly get a different one.
        let err = match select(
            registry_of("missing", &["research", "fix"]).await,
            Some("migrate"),
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an unknown workflow name must fail"),
        };
        assert!(err.contains("no workflow named `migrate`"), "{err}");
        assert!(err.contains("fix") && err.contains("research"), "{err}");
        // The built-in is listed too — it is always something a run could have asked for.
        assert!(err.contains(BUILT_IN), "{err}");
    }

    #[tokio::test]
    async fn choosing_between_several_is_not_done_by_accident() {
        // Picking the first would look like a decision while being one nobody made.
        let err = match select(registry_of("several", &["research", "fix"]).await, None) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("several defined workflows with no name must not silently pick one"),
        };
        assert!(err.contains("--workflow"), "{err}");

        let picked = select(
            registry_of("several-named", &["research", "fix"]).await,
            Some("fix"),
        )
        .unwrap();
        assert_eq!(picked.name(), "fix");
    }

    #[tokio::test]
    async fn the_built_in_is_the_fallback_and_can_also_be_asked_for_by_name() {
        // A repo defining nothing gets the built-in.
        assert_eq!(
            select(vec![Workflow::BuiltIn], None).unwrap().name(),
            BUILT_IN
        );

        // One defined workflow is used without being named — the built-in is always present, so
        // counting it would make a repo with a single script ambiguous for no reason.
        let picked = select(registry_of("one", &["only"]).await, None).unwrap();
        assert_eq!(picked.name(), "only");

        // And the built-in can be demanded even when scripts exist, which is the way back to the
        // Rust flow's gates without deleting a file.
        let picked = select(registry_of("override", &["only"]).await, Some(BUILT_IN)).unwrap();
        assert!(matches!(picked, Workflow::BuiltIn));
    }
}
