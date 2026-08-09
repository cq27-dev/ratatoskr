//! Concrete Phase 2 nodes (scout, memory, analyst) and the straight-line `plan` executor.
//!
//! Per the Phase 2 decision, this is a plain sequential `async fn`, not a generic edge-walking
//! interpreter: with exactly three fixed nodes and no branching, `run_plan` delivers the same
//! policy guarantee (schema-validated handoffs, a checkpoint after every node, nothing skipped)
//! with nothing to get wrong. The real executor arrives in Phase 3 when fork/join needs one.

pub mod analyst;
pub mod bookkeeper;
pub mod child;
pub mod clarify;
pub mod context;
pub mod control;
pub mod converge;
pub mod implementer;
pub mod issue;
pub mod memory;
pub mod overseer;
pub mod plugins;
pub mod publisher;
pub mod redteam;
pub mod referee;
pub mod scout;
pub mod skills;
pub mod stage;
pub mod testrun;
pub mod validate;
pub mod verifier;
pub mod workflow;

pub(crate) use plugins::stage_agent_config;
pub use plugins::{NodePlugins, PluginContext};
#[cfg(test)]
use plugins::{default_allow, servers_to_start};

pub use analyst::AnalystOutput;
pub use bookkeeper::{BookkeeperInput, BookkeeperOutput, MemoryWritten};
pub use child::ChildTask;
pub use context::{Constraint, ContextNode, ContextOutput};
pub use implementer::{ImplementerNode, ImplementerOutput};
pub use memory::{MemoryNode, MemoryOutput, MemoryRecord};
pub use overseer::OverseerOutput;
pub use publisher::PublisherOutput;
pub use redteam::{RedTeamNode, RedTeamOutput};
pub use referee::{RefereeNode, RefereeOutput, Violation};
pub use scout::{RelatedItem, ScoutOutput};
pub use stage::{
    AgentProfile, Delegation, Stage, agent_profiles, built_in_agents, built_in_stages,
};
pub use validate::validate;
pub use verifier::{Finding, FindingKind, Severity, VerifierOutput};

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ratatoskr_core::{RatatoskrConfig, RunState, RunStatus, ToolPolicy};
use ratatoskr_exec::{WorktreePath, remove_worktree};
use ratatoskr_graph::{NodeError, parse_validated};
use ratatoskr_mcp::{RagRatClient, ServerTools, ToolSet};
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
    /// What the context node distilled. Carried so a later analyst revision keeps it: what bears
    /// on the task did not stop being true because the plan turned out wrong.
    pub brief: String,
    pub constraints: Vec<Constraint>,
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
    #[error("configuration error: {0}")]
    Configuration(String),
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
    validate_configured_stages(request.config).await?;
    // Before any node runs, so the first one can already be paused.
    control::install(request.run_id);
    // Before the issue checkpoint is written, so what is recorded is what every node was given.
    let filled = issue::enriched(request.issue, &std::env::current_dir().unwrap_or_default()).await;
    let request = RunRequest {
        issue: &filled,
        ..request
    };
    // Decided before the request is taken apart: choosing needs the whole of it.
    let chosen = choose(&request).await?;
    let RunRequest {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        ..
    } = request;
    // A workflow, when this repo defines one, overrides the built-in sequencing.
    if let Workflow::Scripted(runtime) = chosen {
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
    let ledger = Arc::new(RunLedger::default());
    let outcome = plan_half(PlanHalfRequest {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        context: &context,
        ledger: &ledger,
    })
    .await;
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

struct PlanHalfRequest<'a> {
    client: Option<&'a RagRatClient>,
    config: &'a RatatoskrConfig,
    store: &'a Store,
    run_id: &'a str,
    issue: &'a str,
    engine: &'a Arc<ScriptEngine>,
    context: &'a PluginContext,
    ledger: &'a Arc<RunLedger>,
}

/// The planning half, against an already-resolved plugin context. `run_full` resolves it once and
/// reuses it for both halves, so a full run doesn't pay for every plugin's `SessionStart` twice.
async fn plan_half(request: PlanHalfRequest<'_>) -> Result<PlanOutcome, PlanError> {
    let PlanHalfRequest {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        context,
        ledger,
    } = request;
    // First, and here rather than in a caller: every checkpoint references this row, and the
    // schema enforces it. `run_full` reaches this function directly, so a caller-side write would
    // leave that path creating checkpoints for a run that doesn't exist yet.
    store
        .upsert_run(run_id, None, RunStatus::Running.as_str())
        .await?;

    record_provenance(store, run_id, config).await;

    let clarifier = NodeClarifier::new(config, store, engine, run_id, issue);
    let run = Run {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        clarifier: &clarifier,
        ledger,
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
    let sink = client.map(|c| c.sink());
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

    // --- context ---
    let mut plugins_context = context.for_node("context");
    let ctx_cfg = stage_agent_config(
        engine,
        config,
        context.pool_for("context", client.map(|c| c.offer())),
        "context",
        context::CONTEXT_TOOLS,
        &mut plugins_context,
    )?;
    let mut ctx_tools = ctx_cfg.tools;
    ctx_tools.add_local(clarify::ask_tool());
    let declared_context =
        workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
            client,
            config,
            store,
            run_id,
            issue,
            engine,
            plugin_context: context.clone(),
            ledger: Arc::clone(ledger),
        })?;
    let context_node = ContextNode {
        route: ctx_cfg.route,
        tools: ctx_tools,
        sink: sink.clone(),
        policy: ctx_cfg.policy,
        max_turns: ctx_cfg.max_turns,
        clarifier: Some(clarifier.as_dyn()),
        system_prompt: ctx_cfg.system_prompt,
        plugins: plugins_context,
        files: ctx_cfg.files,
        ledger: Some(Arc::clone(ledger)),
        declared_context: Arc::clone(&declared_context),
    };
    let context_out = context_node
        .run(issue)
        .await
        .map_err(|e| PlanError::node("context", e))?;
    record(Record {
        store,
        run_id,
        node: "context",
        output: &context_out,
        input: Some(serde_json::to_string(issue)?),
        iteration: None,
        ledger: Some(ledger),
    })
    .await?;
    let scout_out = context_out.scout.clone();
    let memory_out = context_out.memory.clone();
    state.scout_report = Some(serde_json::to_value(&scout_out)?);

    // --- analyst ---
    let analyst_in =
        analyst::AnalystInput::fresh(issue.to_string(), scout_out.clone(), memory_out.clone());
    let analyst_input_json = serde_json::to_string(&analyst_in)?;
    let analyst_out = run_declared_analyst(&declared_context, &analyst_in).await?;
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
        brief: context_out.brief,
        constraints: context_out.constraints,
    })
}

async fn run_declared_analyst(
    context: &Arc<workflow::WorkflowContext>,
    input: &analyst::AnalystInput,
) -> Result<AnalystOutput, PlanError> {
    run_declared_analyst_with_turn(context, input, None).await
}

async fn run_declared_analyst_with_turn(
    context: &Arc<workflow::WorkflowContext>,
    input: &analyst::AnalystInput,
    turn: Option<Arc<dyn workflow::StageTurn>>,
) -> Result<AnalystOutput, PlanError> {
    let input_json = serde_json::to_string(input)?;
    let question = analyst::render_prompt(input);
    let result = match turn {
        Some(turn) => {
            workflow::evaluate_standard_stage_with_turn(
                Arc::clone(context),
                "analyst",
                input_json,
                question,
                turn,
            )
            .await
        }
        None => {
            workflow::evaluate_standard_stage(Arc::clone(context), "analyst", input_json, question)
                .await
        }
    };
    let raw = result.map_err(|error| {
        PlanError::node(
            "analyst",
            NodeError::Failed(format!("analyst agent failed: {error}")),
        )
    })?;
    parse_validated::<AnalystOutput>(&raw).map_err(|error| PlanError::node("analyst", error))
}

async fn run_declared_verifier(
    context: &Arc<workflow::WorkflowContext>,
    input: &verifier::VerifierInput,
    worktree: &WorktreePath,
) -> Result<VerifierOutput, PlanError> {
    run_declared_verifier_with_turn(context, input, worktree, None).await
}

async fn run_declared_verifier_with_turn(
    context: &Arc<workflow::WorkflowContext>,
    input: &verifier::VerifierInput,
    worktree: &WorktreePath,
    turn: Option<Arc<dyn workflow::StageTurn>>,
) -> Result<VerifierOutput, PlanError> {
    // Nothing to review is not a clean review. Preserve the no-turn result rather than asking a
    // declared stage to approve an empty patch.
    if input.diff.trim().is_empty() {
        return Ok(VerifierOutput {
            findings: Vec::new(),
            assessment: "there was no diff to review".to_string(),
        });
    }
    let input_json = serde_json::to_string(input)?;
    let question = verifier::render_prompt(input);
    let resources = workflow::StandardStageResources {
        resource_root: worktree.as_path().to_path_buf(),
        shell: None,
        publish: None,
        clarifier: None,
        guidance: None,
    };
    let result = match turn {
        Some(turn) => {
            workflow::evaluate_standard_stage_with_resources_and_turn(
                Arc::clone(context),
                "verifier",
                input_json,
                question,
                resources,
                turn,
            )
            .await
        }
        None => {
            workflow::evaluate_standard_stage_with_resources(
                Arc::clone(context),
                "verifier",
                input_json,
                question,
                resources,
            )
            .await
        }
    };
    let raw = result.map_err(|error| {
        PlanError::node(
            "verifier",
            NodeError::Failed(format!("verifier agent failed: {error}")),
        )
    })?;
    parse_validated::<VerifierOutput>(&raw).map_err(|error| PlanError::node("verifier", error))
}

/// One checkpoint to write: which node, what it produced, and — for a node that ran a model — what
/// it was given and what the turn cost.
///
/// `input` and `ledger` are optional because not every checkpoint has them: the `issue` row is the
/// run's own input rather than a node's. A missing value here means "there was none", never "we
/// forgot to look".
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

/// Fewest output tokens that could plausibly have produced `bytes` of output.
///
/// Eight bytes per token is far below any real tokenizer — three to four is typical — so this
/// cannot fire on a merely surprising number. It exists because the provider-reported count was
/// silently near-zero for weeks and nothing noticed: a figure recorded with the same confidence as
/// a correct one is the failure mode the whole telemetry column was added to prevent.
fn plausible_output_tokens(bytes: usize) -> u64 {
    (bytes / 8) as u64
}

async fn record<T: Serialize>(r: Record<'_, T>) -> Result<(), PlanError> {
    let json = serde_json::to_string(r.output)?;
    let input_json = r.input;
    // Claimed rather than borrowed: the ledger holds one entry per model turn, and taking it here
    // is what keeps the converge loop's repeated implementer turns matched to their own rows.
    let telemetry = r.ledger.and_then(|l| l.take(r.node)).unwrap_or_default();
    // Only for a node that actually ran a model — one with no `model` reported no usage because it
    // had none, which is ordinary.
    if telemetry.model.is_some() {
        let floor = plausible_output_tokens(json.len());
        if telemetry.usage.output_tokens < floor {
            tracing::warn!(
                node = r.node,
                reported = telemetry.usage.output_tokens,
                floor,
                bytes = json.len(),
                "fewer output tokens reported than this node's output could contain. The count \
                 comes back short from the endpoint, not from anything this side computes: a \
                 direct, non-streamed request returns `output_tokens: 4` for a response carrying \
                 a whole reasoning block and a tool call. Input, cache and reasoning figures are \
                 unaffected, so cost per turn is still readable from those. Run with \
                 `RUST_LOG=ratatoskr_agent=debug` for per-turn usage, and treat this warning on a \
                 tool-calling node as expected until the endpoint reports the real count"
            );
        }
    }
    // The event carries the same measurements as the row, because a viewer reconstructing where a
    // run WAS reads the log, not the store: the store holds only the latest state of each node, so
    // deriving a past moment from it shows final numbers against a historical position. Everything
    // the row records, the event has to be able to prove.
    let logged = telemetry.clone();
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
        iteration = r.iteration,
        model = logged.model.as_deref().unwrap_or_default(),
        tools = logged.tools.join(","),
        tools_used = logged.tools_used.join(","),
        thinking = logged.thinking,
        reuses_session = logged.reuses_session,
        turns = logged.turns.unwrap_or_default(),
        error = logged.error.as_deref().unwrap_or_default(),
        duration_ms = logged.duration_ms.unwrap_or_default(),
        "gen_ai.usage.input_tokens" = logged.usage.input_tokens,
        "gen_ai.usage.output_tokens" = logged.usage.output_tokens,
        "gen_ai.usage.cached_input_tokens" = logged.usage.cached_input_tokens,
        "gen_ai.usage.cache_creation_input_tokens" = logged.usage.cache_creation_input_tokens,
        "gen_ai.usage.reasoning_tokens" = logged.usage.reasoning_tokens,
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
            // The graph itself, not just a hash of it. A hash says two runs differed; the shape is
            // what lets a run be drawn by something that never had this pipeline.
            serde_json::to_string(&ratatoskr_core::shape::built_in())
                .ok()
                .as_deref(),
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
    let mut workflows = scripts_in(repo.join(WORKFLOW_DIR));
    workflows.push(repo.join(LEGACY_WORKFLOW));
    for workflow in &workflows {
        if let Ok(dependencies) = ratatoskr_script::workflow::dependencies(workflow) {
            sources.extend(dependencies);
        }
    }
    sources.extend(workflows);
    // Sorted, because `read_dir` order is the filesystem's business and a fingerprint that depends
    // on it would differ between two checkouts of identical files.
    sources.sort();
    sources.dedup();

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for path in sources {
        // A missing file still contributes its name, so adding a `workflow.ts` changes the
        // fingerprint even if it is empty.
        for byte in path
            .strip_prefix(repo)
            .unwrap_or(&path)
            .as_os_str()
            .as_encoded_bytes()
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

/// The nodes a ruleset may govern out of the box — the LLM agents that go through
/// `run_structured`. `memory` is absent because it is a direct rag-rat call with no model or tool
/// set to override, so targeting it is a config error rather than a no-op.
///
/// The implementer belongs here now that it drives a model directly rather than a coding CLI: it
/// resolves through `node_agent_config` like every other node, so a ruleset already shapes its
/// route, prompt, tools and plugins — this list was the only thing saying otherwise.
///
/// Lives here rather than in the CLI because this crate is what decides which nodes exist. A
/// workflow may add to this set; see [`Workflow::nodes`].
pub const BUILT_IN_NODES: &[&str] = &[
    "overseer",
    "publisher",
    "context",
    "scout",
    "analyst",
    "implementer",
    "bookkeeper",
    "redteam",
    "verifier",
    "characterizer",
];

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

    /// Node names this workflow governs beyond [`BUILT_IN_NODES`].
    ///
    /// A workflow that only re-sequences the existing nodes declares none. One that introduces a
    /// node has to say so, or its `.ratatoskr/rules/<node>.ts` is rejected at load as targeting
    /// something that does not exist — which is the right error for a typo and the wrong one for a
    /// node the workflow genuinely has.
    pub fn nodes(&self) -> &[String] {
        match self {
            Workflow::BuiltIn => &[],
            Workflow::Scripted(w) => &w.meta().nodes,
        }
    }

    /// The cases in which this workflow is the right one. Empty for the built-in: it is the
    /// fallback, so it is chosen by nothing else matching rather than by matching.
    pub fn when_to_use(&self) -> &[String] {
        match self {
            Workflow::BuiltIn => &[],
            Workflow::Scripted(w) => &w.meta().when_to_use,
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

/// Checkpointed internal gates that never become configurable workflow stages.
const INTERNAL_GATES: &[&str] = &["referee"];

/// Every node any workflow in this repo may govern: the built-in set plus what each declares.
///
/// The union across all of them, not just the one a run selects, because rulesets are loaded
/// before a workflow is chosen — and a ruleset targeting a node that some workflow declares is
/// legitimate whether or not this particular run uses that workflow. The internal referee is
/// never governable, even when a workflow names it.
fn governable_from(workflows: impl IntoIterator<Item = WorkflowRuntime>) -> Vec<String> {
    let mut names: Vec<String> = BUILT_IN_NODES.iter().map(|s| s.to_string()).collect();
    for workflow in workflows {
        names.extend(workflow.meta().nodes.iter().cloned());
        names.extend(workflow.meta().stages.iter().map(|stage| stage.id.clone()));
    }
    names.retain(|name| !INTERNAL_GATES.contains(&name.as_str()));
    names.sort();
    names.dedup();
    names
}

pub async fn governable_nodes() -> Result<Vec<String>, PlanError> {
    Ok(governable_from(defined().await?))
}

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

/// Validate the stage registry every configured workflow contributes before any run starts.
pub async fn validate_configured_stages(config: &RatatoskrConfig) -> Result<(), PlanError> {
    let profiles = agent_profiles(config);
    let mut stages = built_in_stages();
    stages.retain(|stage| {
        !matches!(
            stage.id.as_str(),
            "overseer" | "scout" | "analyst" | "verifier" | "characterizer"
        )
    });
    stages.extend(workflow::standard_stages().await?);
    for workflow in defined().await? {
        let workflow_stages = stage::stages_from_workflow(workflow.meta());
        validate::validate_declared_contracts(&workflow_stages)?;
        stages.extend(workflow_stages);
    }
    validate::validate(&stages, &profiles)
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

/// The overseer is opt-in on having somewhere to run, like the verifier and the characterizer.
fn overseer_enabled(engine: &Arc<ScriptEngine>, config: &RatatoskrConfig) -> bool {
    node_route(engine, config, "overseer").is_some()
}

fn should_consult_overseer(
    defined_workflows: usize,
    workflow_named: bool,
    configured: bool,
) -> bool {
    !workflow_named && defined_workflows > 1 && configured
}

struct OverseerDecision<'a> {
    store: &'a Store,
    run_id: &'a str,
    found: Vec<Workflow>,
    decided: OverseerOutput,
    input_json: String,
    ledger: &'a Arc<RunLedger>,
}

async fn select_and_record_overseer(decision: OverseerDecision<'_>) -> Result<Workflow, PlanError> {
    let OverseerDecision {
        store,
        run_id,
        found,
        decided,
        input_json,
        ledger,
    } = decision;
    let selected = select(found, Some(&decided.workflow))?;
    store
        .upsert_run(run_id, None, RunStatus::Running.as_str())
        .await?;
    record(Record {
        store,
        run_id,
        node: "overseer",
        output: &decided,
        input: Some(input_json),
        iteration: None,
        ledger: Some(ledger),
    })
    .await?;
    Ok(selected)
}

/// Pick the workflow for this run, asking the overseer when there is a real choice to make.
///
/// The order is deliberate. A named workflow wins outright — a caller that said which shape it
/// wanted is not asking to be second-guessed. Nothing to choose between resolves without a model
/// call, because paying for a decision with one answer is waste. Only a genuine choice reaches the
/// overseer, and without one configured the run still refuses to guess rather than picking.
pub async fn choose(request: &RunRequest<'_>) -> Result<Workflow, PlanError> {
    let found = registry().await?;
    let defined_workflows = found
        .iter()
        .filter(|workflow| !matches!(workflow, Workflow::BuiltIn))
        .count();
    if !should_consult_overseer(
        defined_workflows,
        request.workflow.is_some(),
        overseer_enabled(request.engine, request.config),
    ) {
        return select(found, request.workflow);
    }

    let choices: Vec<overseer::Choice> = found
        .iter()
        .map(|w| overseer::Choice {
            name: w.name().to_string(),
            purpose: w.purpose().to_string(),
            when_to_use: w.when_to_use().to_vec(),
        })
        .collect();

    let input = overseer::OverseerInput {
        issue: request.issue.to_string(),
        choices,
    };
    let input_json = serde_json::to_string(&input)?;
    let rendered_question = overseer::render_prompt(&input.issue, &input.choices);
    let cwd = std::env::current_dir().unwrap_or_default();
    let plugin_context = PluginContext::resolve(request.config, request.engine, &cwd).await?;
    let ctx = workflow::WorkflowContext::new(
        request.client,
        request.config,
        request.store,
        request.run_id,
        request.issue,
        request.engine,
        plugin_context,
    )?;
    let raw = workflow::evaluate_standard_stage(
        Arc::clone(&ctx),
        "overseer",
        input_json.clone(),
        rendered_question,
    )
    .await
    .map_err(|error| PlanError::node("overseer", NodeError::Failed(error)))?;
    let decided: OverseerOutput = serde_json::from_str(&raw)?;

    // Its own ledger: the overseer runs before the run's, and its cost is still a cost. Drained
    // straight onto the checkpoint below rather than carried, because nothing after this point
    // would claim it.
    // A model naming something absent is rejected before this writes anything, so the store never
    // presents a rejected routing decision as a completed stage.
    select_and_record_overseer(OverseerDecision {
        store: request.store,
        run_id: request.run_id,
        found,
        decided,
        input_json,
        ledger: ctx.ledger(),
    })
    .await
}

/// What one run needs to start.
///
/// A struct rather than a seventh positional argument: these travel together through both entry
/// points, and every one of them is a borrow of something the caller already holds, so a positional
/// list grows with the run rather than with the job.
pub struct RunRequest<'a> {
    /// `None` when this repository runs without rag-rat: the nodes keep their file tools and the
    /// memory baseline is simply empty. See `RagRatConfig::configured`.
    pub client: Option<&'a RagRatClient>,
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
    node_route(engine, config, "redteam").is_some()
}

/// The resolved agent settings for one stage: profile defaults plus stage ruleset overrides.
pub struct NodeAgentConfig {
    pub route: ratatoskr_core::ModelRoute,
    pub tools: ToolSet,
    pub capability_ceiling: Option<ratatoskr_core::Capability>,
    /// The repository the stage's built-in file tools read within.
    pub files: Option<PathBuf>,
    pub policy: Option<Arc<dyn ToolPolicy>>,
    pub max_turns: Option<usize>,
    /// Replaces the stage's built-in preamble when its ruleset declares one.
    pub system_prompt: Option<String>,
}

/// Everything a run's helpers need in common: the rag-rat connection, the run's identity and
/// configuration, and the two things resolved once per run — the clarifier and the plugin context.
///
/// A parameter struct because these travel together through every stage; passing them
/// individually made each helper's signature grow with the run rather than with its job.
pub(crate) struct Run<'a> {
    client: Option<&'a RagRatClient>,
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

/// The preamble a node actually runs with: reusable agent guidance, then its built-in text or a
/// ruleset replacement, prefixed by whatever context plugins contributed for this run and suffixed
/// by the listing of the skills its plugins bind.
///
/// The skill listing lives here rather than in the `Skill` tool's schema so it does not grow the
/// schema the node carries on every model call: it is prose in the cached prefix, and the tool
/// resolves the chosen name against the same set at call time. It is appended to whichever base
/// applies — a ruleset `systemPrompt` override replaces the node's text, not the skills its plugins
/// ship — so overriding the preamble cannot disenfranchise a bound skill. No skills bound leaves the
/// preamble byte-identical to the context-and-base composition.
pub(crate) fn effective_preamble(
    node: &str,
    built_in: &str,
    system_prompt: Option<&str>,
    context: Option<&str>,
    skills: &[ratatoskr_plugin::Skill],
) -> String {
    effective_preamble_with_profile(node, built_in, "", system_prompt, context, skills)
}

/// Compose reusable agent guidance before a stage's own preamble. A ruleset replaces only the
/// stage portion, so a shared profile cannot erase the stage contract or platform instructions.
pub(crate) fn effective_preamble_with_profile(
    node: &str,
    built_in: &str,
    profile_prompt: &str,
    system_prompt: Option<&str>,
    context: Option<&str>,
    skills: &[ratatoskr_plugin::Skill],
) -> String {
    let stage = system_prompt.unwrap_or(built_in);
    let base = [profile_prompt, stage]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let composed = match context {
        Some(context) => format!("{context}\n\n{base}"),
        None => base.to_string(),
    };
    // `node` attributes a duplicate-name warning to the node whose plugins bound the skills.
    match skills::listing(skills, node) {
        Some(listing) => format!(
            "{composed}\n\nAvailable skills: call the Skill tool with one of these names, and only \
             these, when its description matches what you are doing.\n{listing}"
        ),
        None => composed,
    }
}

/// The most of `AGENTS.md` a writing node's preamble will carry. It is paid on every model call
/// the node makes, so a file large enough to matter is reported and clipped rather than silently
/// taxing every turn. Generous enough for the ~130-line file this repo ships, a ceiling for a
/// runaway one.
pub(crate) const CONVENTIONS_BUDGET: usize = 16 * 1024;

/// The repository's own coding conventions, discovered by convention rather than configuration:
/// `AGENTS.md` at the repo root, falling back to `CLAUDE.md`. `None` when neither exists (or the
/// path is unreadable), so a caller leaves its preamble exactly as it is today.
///
/// Read from the repo root at plan-build time, not from an implementer's worktree — the worktree
/// is a copy, and loading once from the checkout keeps every converge iteration and the test
/// author on the same text. `AGENTS.md` is read first and `CLAUDE.md` only when it is absent, so
/// this repo's `CLAUDE.md`-symlink-to-`AGENTS.md` layout is read once, never doubled.
///
/// Bounded to [`CONVENTIONS_BUDGET`]: a file over the bound is clipped and logged (naming the file
/// and its full vs. injected size) rather than truncated in silence.
pub(crate) fn repo_conventions(repo_root: &std::path::Path) -> Option<String> {
    for name in ["AGENTS.md", "CLAUDE.md"] {
        let path = repo_root.join(name);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if text.len() > CONVENTIONS_BUDGET {
            let mut end = CONVENTIONS_BUDGET;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            let clipped = &text[..end];
            tracing::warn!(
                file = %path.display(),
                full_bytes = text.len(),
                injected_bytes = clipped.len(),
                "repository conventions exceed the preamble budget; injecting a prefix, not the whole file"
            );
            return Some(clipped.to_string());
        }
        return Some(text);
    }
    None
}

/// Prefix a writing node's preamble with the repository conventions, recording how much of the
/// composed preamble came from them. `None` conventions (no `AGENTS.md`) leaves `base` byte-for-byte
/// unchanged — no header, no separator — so a repo with no conventions file runs exactly as before.
#[cfg(test)]
pub(crate) fn with_conventions(node: &str, conventions: Option<&str>, base: String) -> String {
    let Some(conventions) = conventions else {
        return base;
    };
    tracing::info!(
        node,
        conventions_chars = conventions.len(),
        preamble_chars = conventions.len() + base.len(),
        "repository conventions injected into node preamble"
    );
    format!("{conventions}\n\n{base}")
}

#[cfg(test)]
mod migrated_stage_path_tests {
    #[test]
    fn migrated_native_components_have_one_stage_executor_model_path() {
        for (component, source) in [
            ("context", include_str!("context.rs")),
            ("implementer", include_str!("implementer.rs")),
            ("redteam", include_str!("redteam.rs")),
            ("characterizer", include_str!("testrun.rs")),
        ] {
            assert!(
                source.contains("evaluate_standard_stage"),
                "{component} must invoke the generic stage executor"
            );
            assert!(
                !source.contains("NodeRun"),
                "{component} must not retain a direct model-loop fallback"
            );
            assert!(
                !source.contains("declared_context: Option"),
                "{component} must require a declared workflow context"
            );
        }

        let built_in = include_str!("lib.rs");
        assert!(
            !built_in.contains(concat!("Analyst", "Node {")),
            "the built-in plan and review must not construct the compatibility analyst wrapper"
        );
        assert!(
            !built_in.contains(concat!("verifier::Verifier", "Node {")),
            "the built-in review must not construct the compatibility verifier wrapper"
        );

        for (component, source, wrapper) in [
            (
                "analyst",
                include_str!("analyst.rs"),
                concat!("Analyst", "Node"),
            ),
            (
                "bookkeeper",
                include_str!("bookkeeper.rs"),
                concat!("Bookkeeper", "Node"),
            ),
            (
                "overseer",
                include_str!("overseer.rs"),
                concat!("Overseer", "Node"),
            ),
            (
                "publisher",
                include_str!("publisher.rs"),
                concat!("Publisher", "Node"),
            ),
            ("scout", include_str!("scout.rs"), concat!("Scout", "Node")),
            (
                "verifier",
                include_str!("verifier.rs"),
                concat!("Verifier", "Node"),
            ),
        ] {
            assert!(
                !source.contains(wrapper),
                "{component} must not retain an obsolete direct model wrapper"
            );
        }

        assert!(
            !include_str!("workflow.rs").contains(concat!("(\"analy", "ze\",")),
            "the direct analyst compatibility operation must remain removed"
        );
    }
}

#[cfg(test)]
mod built_in_declared_review_tests {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct SeenTurn {
        node: String,
        session: ratatoskr_core::SessionScope,
        conversation: Option<String>,
        ledger: Option<usize>,
        files: Option<PathBuf>,
        question: String,
    }

    struct SequenceTurn {
        seen: Mutex<Vec<SeenTurn>>,
        outputs: Mutex<VecDeque<Result<String, String>>>,
    }

    impl SequenceTurn {
        fn new(outputs: impl IntoIterator<Item = Result<serde_json::Value, &'static str>>) -> Self {
            Self {
                seen: Mutex::new(Vec::new()),
                outputs: Mutex::new(
                    outputs
                        .into_iter()
                        .map(|output| {
                            output
                                .map(|value| value.to_string())
                                .map_err(str::to_string)
                        })
                        .collect(),
                ),
            }
        }
    }

    impl workflow::StageTurn for SequenceTurn {
        fn run<'a>(
            &'a self,
            run: ratatoskr_agent::NodeRun<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<String, ratatoskr_agent::AgentError>> + Send + 'a>>
        {
            self.seen
                .lock()
                .expect("seen-turn mutex poisoned")
                .push(SeenTurn {
                    node: run.node.to_string(),
                    session: run.route.session,
                    conversation: run.conversation.map(str::to_string),
                    ledger: run
                        .ledger
                        .as_ref()
                        .map(|ledger| Arc::as_ptr(ledger) as usize),
                    files: run.files.clone(),
                    question: run.question.to_string(),
                });
            let output = self
                .outputs
                .lock()
                .expect("sequence-output mutex poisoned")
                .pop_front()
                .expect("a canned output exists for every turn");
            Box::pin(async move { output.map_err(ratatoskr_agent::AgentError::Prompt) })
        }
    }

    async fn declared_contexts(
        case: &str,
    ) -> (
        Store,
        Arc<RunLedger>,
        Arc<workflow::WorkflowContext>,
        Arc<workflow::WorkflowContext>,
        std::path::PathBuf,
    ) {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-built-in-declared-{}-{case}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();
        store
            .upsert_run("built-in-declared", None, RunStatus::Running.as_str())
            .await
            .unwrap();
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "verifier".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "openai".to_string(),
                model: "test-model".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Compacted,
            },
        );
        let ledger = Arc::new(RunLedger::default());
        let context = |plugin_context| {
            workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
                client: None,
                config: &config,
                store: &store,
                run_id: "built-in-declared",
                issue: "keep the hand-off",
                engine: &engine,
                plugin_context,
                ledger: Arc::clone(&ledger),
            })
            .unwrap()
        };
        let planning = context(PluginContext::default());
        let review = context(PluginContext::default());
        (store, ledger, planning, review, dir)
    }

    fn initial_analyst_input() -> analyst::AnalystInput {
        analyst::AnalystInput::fresh(
            "keep the hand-off".to_string(),
            ScoutOutput {
                related_items: Vec::new(),
                papertrail_summary: "the original plan chose SessionKey".to_string(),
            },
            MemoryOutput {
                memories: Vec::new(),
            },
        )
    }

    #[tokio::test]
    async fn built_in_analyst_reentry_keeps_one_ledger_and_typed_checkpoint_handoff() {
        let (store, ledger, planning, review, dir) = declared_contexts("analyst").await;
        let turn = Arc::new(SequenceTurn::new([
            Ok(serde_json::json!({
                "impact_summary": "introduce SessionKey",
                "requirements": ["scope keys by run"],
                "interface": [{
                    "name": "SessionKey",
                    "shape": "struct SessionKey { run: String, stage: String }"
                }]
            })),
            Ok(serde_json::json!({
                "impact_summary": "retain SessionKey and reject empty run ids",
                "requirements": ["scope keys by run", "reject empty run ids"]
            })),
        ]));
        let fresh = initial_analyst_input();
        let initial = run_declared_analyst_with_turn(
            &planning,
            &fresh,
            Some(Arc::clone(&turn) as Arc<dyn workflow::StageTurn>),
        )
        .await
        .unwrap();
        record(Record {
            store: &store,
            run_id: "built-in-declared",
            node: "analyst",
            output: &initial,
            input: Some(serde_json::to_string(&fresh).unwrap()),
            iteration: None,
            ledger: Some(&ledger),
        })
        .await
        .unwrap();

        let revision = analyst::AnalystInput {
            issue: fresh.issue.clone(),
            scout: fresh.scout.clone(),
            memory: fresh.memory.clone(),
            brief: fresh.brief.clone(),
            constraints: fresh.constraints.clone(),
            previous: Some(Box::new(initial)),
            findings: vec![verifier::Finding {
                severity: verifier::Severity::P1,
                kind: verifier::FindingKind::Plan,
                file: "session.rs".to_string(),
                line: Some(17),
                summary: "empty run ids collide".to_string(),
                failure_scenario: "two empty run ids share a stage key".to_string(),
            }],
        };
        let revised = run_declared_analyst_with_turn(
            &review,
            &revision,
            Some(Arc::clone(&turn) as Arc<dyn workflow::StageTurn>),
        )
        .await
        .unwrap();
        record(Record {
            store: &store,
            run_id: "built-in-declared",
            node: "analyst",
            output: &revised,
            input: Some(serde_json::to_string(&revision).unwrap()),
            iteration: Some(2),
            ledger: Some(&ledger),
        })
        .await
        .unwrap();

        {
            let seen = turn.seen.lock().expect("seen-turn mutex poisoned");
            assert_eq!(seen.len(), 2);
            assert!(seen.iter().all(|turn| turn.node == "analyst"));
            assert!(seen.iter().all(|turn| {
                turn.session == ratatoskr_core::SessionScope::Compacted
                    && turn.conversation.as_deref() == Some("built-in-declared-analyst")
            }));
            assert_eq!(seen[0].ledger, seen[1].ledger);
            assert_eq!(seen[0].ledger, Some(Arc::as_ptr(&ledger) as usize));
            assert!(!seen[0].question.contains("THIS IS A REVISION"));
            assert!(seen[1].question.contains("THIS IS A REVISION"));
            assert!(seen[1].question.contains("SessionKey"));
            assert!(seen[1].question.contains("empty run ids collide"));
        }

        let checkpoints = store
            .checkpoints_for_run("built-in-declared")
            .await
            .unwrap();
        assert_eq!(checkpoints.len(), 2);
        let first: analyst::AnalystInput =
            serde_json::from_str(checkpoints[0].input_json.as_deref().unwrap()).unwrap();
        let second: analyst::AnalystInput =
            serde_json::from_str(checkpoints[1].input_json.as_deref().unwrap()).unwrap();
        assert!(first.previous.is_none());
        assert_eq!(second.previous.unwrap().interface[0].name, "SessionKey");
        assert_eq!(second.findings[0].summary, "empty run ids collide");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn built_in_verifier_keeps_empty_diff_local_and_roots_the_declared_turn() {
        let (_store, _ledger, _planning, review, dir) = declared_contexts("verifier").await;
        let turn = Arc::new(SequenceTurn::new([
            Ok(serde_json::json!({
                "findings": [{
                    "severity": "P1",
                    "kind": "plan",
                    "summary": "the plan cannot isolate runs",
                    "failure_scenario": "two runs collide"
                }],
                "assessment": "the diff follows an unsafe plan"
            })),
            Err("provider unavailable"),
        ]));
        let worktree = WorktreePath(dir.join("worktree"));
        std::fs::create_dir_all(worktree.as_path()).unwrap();
        let analyst = AnalystOutput {
            impact_summary: "scope the session key".to_string(),
            touched: Vec::new(),
            risks: Vec::new(),
            requirements: vec!["isolate runs".to_string()],
            residual_risk: String::new(),
            changes_code: true,
            acceptance: Vec::new(),
            interface: Vec::new(),
        };
        let mut input = verifier::VerifierInput {
            issue: "keep the hand-off".to_string(),
            analyst,
            diff: String::new(),
            touched_files: vec!["session.rs".to_string()],
            previous_findings: Vec::new(),
        };
        let empty = run_declared_verifier_with_turn(
            &review,
            &input,
            &worktree,
            Some(Arc::clone(&turn) as Arc<dyn workflow::StageTurn>),
        )
        .await
        .unwrap();
        assert!(empty.findings.is_empty());
        assert_eq!(empty.assessment, "there was no diff to review");
        assert!(
            turn.seen
                .lock()
                .expect("seen-turn mutex poisoned")
                .is_empty()
        );

        input.diff = "+pub struct SessionKey;".to_string();
        let reviewed = run_declared_verifier_with_turn(
            &review,
            &input,
            &worktree,
            Some(Arc::clone(&turn) as Arc<dyn workflow::StageTurn>),
        )
        .await
        .unwrap();
        assert_eq!(reviewed.blocking(verifier::Severity::P2).len(), 1);
        assert_eq!(reviewed.findings[0].kind, verifier::FindingKind::Plan);
        {
            let seen = turn.seen.lock().expect("seen-turn mutex poisoned");
            assert_eq!(seen.len(), 1);
            assert_eq!(seen[0].session, ratatoskr_core::SessionScope::Fresh);
            assert_eq!(seen[0].files.as_deref(), Some(worktree.as_path()));
            assert!(
                seen[0]
                    .question
                    .contains("THE CHANGE:\n+pub struct SessionKey;")
            );
        }

        let error = run_declared_verifier_with_turn(
            &review,
            &input,
            &worktree,
            Some(Arc::clone(&turn) as Arc<dyn workflow::StageTurn>),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("provider unavailable"));
        let _ = std::fs::remove_dir_all(dir);
    }
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

/// What the run's last review still objected to, read back from its checkpoints.
///
/// Best-effort, and empty is the ordinary answer: a repository with no verifier route never
/// reviews at all. The publisher is told to say what is unresolved, so this is where it finds out;
/// a store read that fails costs a sentence in a pull request and must not cost the run.
async fn unresolved_of(store: &Store, run_id: &str) -> Vec<verifier::Finding> {
    let checkpoints = match store.checkpoints_for_run(run_id).await {
        Ok(checkpoints) => checkpoints,
        Err(e) => {
            tracing::warn!("could not read the run's checkpoints for publishing: {e}");
            return Vec::new();
        }
    };
    // The last review is the one that still stands: an earlier pass's findings were either fixed
    // or raised again, and reporting a fixed one would be as misleading as reporting none.
    checkpoints
        .iter()
        .rev()
        .find(|c| c.node_name == "verifier")
        .and_then(|c| serde_json::from_str::<verifier::VerifierOutput>(&c.output_json).ok())
        .map(|v| v.findings)
        .unwrap_or_default()
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
    run: &Run<'_>,
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
    // This is the run the publisher exists for. Its deliverable is the plan, and before there was
    // anywhere to send it the whole thing finished as a checkpoint in SQLite that somebody had to
    // go and find.
    let published = publish_if_enabled(
        run,
        publisher::PublisherInput {
            issue: run.issue.to_string(),
            analyst: plan.analyst.clone(),
            implementer: None,
            status: status.as_str().to_string(),
            iterations: 0,
            // No fork ran, so there was nothing to review.
            unresolved: Vec::new(),
        },
        true,
    )
    .await;
    context.session_end(status.as_str()).await;

    let mut state = plan.state.clone();
    state.status = status;
    if let Some(p) = &published {
        state.artifacts.push(serde_json::to_value(p)?);
    }
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
    validate_configured_stages(request.config).await?;
    // Before any node runs, so the first one can already be paused.
    control::install(request.run_id);
    // Before the issue checkpoint is written, so what is recorded is what every node was given.
    let filled = issue::enriched(request.issue, &std::env::current_dir().unwrap_or_default()).await;
    let request = RunRequest {
        issue: &filled,
        ..request
    };
    // Decided before the request is taken apart: choosing needs the whole of it.
    let chosen = choose(&request).await?;
    let RunRequest {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        ..
    } = request;
    // A workflow, when this repo defines one, overrides the whole run flow.
    if let Workflow::Scripted(runtime) = chosen {
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
    let ledger = Arc::new(RunLedger::default());
    let plan = match plan_half(PlanHalfRequest {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        context: &plan_context,
        ledger: &ledger,
    })
    .await
    {
        Ok(plan) => plan,
        Err(e) => {
            plan_context.session_end(RunStatus::Failed.as_str()).await;
            return Err(e);
        }
    };

    // Built before the fork decision, because the no-code-change path publishes too and a
    // publisher needs the same run handle every other node gets.
    let clarifier = NodeClarifier::new(config, store, engine, run_id, issue);
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

    // Some tasks call for no code change: research, a review, an architecture answer. Running the
    // fork for one costs a sandboxed baseline test run and an implementer session to produce an
    // empty diff,
    // and then reports `Converged` — a success claim about a change that was never made.
    if !fork_is_needed(&plan.analyst, config) {
        return no_code_change(&run, store, run_id, &plan_context, plan).await;
    }
    // `plan.state.clarifications` already holds the plan-half asks; the fork/bookkeep half gets its
    // own clarifier, drained and appended at the end.
    let mut state = plan.state.clone();

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
    let terminal = matches!(
        status,
        RunStatus::Converged | RunStatus::MaxIterationsReached | RunStatus::Unreviewed
    );
    // Concurrently: one writes to the memory graph, the other to the tracker, and neither needs
    // the other's result. `join!` rather than spawn — both are I/O-bound and borrow their inputs.
    let (bookkeeper, published) = tokio::join!(
        async {
            if !terminal {
                return None;
            }
            // Read back what the run's own checkpoints recorded about its path. The same source
            // the `bookkeep` replay reads, so a replay composes from what the live run did.
            let input = BookkeeperInput {
                issue: issue.to_string(),
                analyst: plan.analyst.clone(),
                implementer: implementer.clone(),
                iterations,
                converged: status == RunStatus::Converged,
                friction: friction_of(store, run_id).await,
            };
            match bookkeep_and_checkpoint(&run, input).await {
                Ok(bk) => Some(bk),
                Err(e) => {
                    tracing::warn!("bookkeeping failed: {e}");
                    None
                }
            }
        },
        publish_if_enabled(
            &run,
            publisher::PublisherInput {
                issue: issue.to_string(),
                analyst: plan.analyst.clone(),
                implementer: Some(implementer.clone()),
                status: status.as_str().to_string(),
                iterations,
                unresolved: unresolved_of(store, run_id).await,
            },
            terminal,
        )
    );
    if let Some(bk) = &bookkeeper {
        state.artifacts = vec![serde_json::to_value(bk)?];
    }
    if let Some(p) = &published {
        state.artifacts.push(serde_json::to_value(p)?);
    }

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

/// Run the publisher, when this repo has turned it on and there is an outcome worth delivering.
///
/// Best-effort, like bookkeeping: the change is made, the tests are recorded, and failing the run
/// because a tracker was unreachable would discard completed work over a delivery problem.
async fn publish_if_enabled(
    run: &Run<'_>,
    input: publisher::PublisherInput,
    terminal: bool,
) -> Option<PublisherOutput> {
    if !terminal || !run.config.publish.enabled {
        return None;
    }
    match publish_and_checkpoint(run, input).await {
        Ok(out) => Some(out),
        Err(e) => {
            tracing::warn!("publishing failed: {e}");
            None
        }
    }
}

/// Build the publisher, run it, and checkpoint what it did.
async fn publish_and_checkpoint(
    run: &Run<'_>,
    input: publisher::PublisherInput,
) -> Result<PublisherOutput, PlanError> {
    let &Run {
        client,
        config,
        store,
        run_id,
        issue,
        engine,
        context,
        ledger,
        ..
    } = run;
    let repo_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    // Push is offered only when there is a branch to push, and only ever THAT branch: the access
    // carries it, and what the tool takes is a name's parts, never a ref. A run with no fork has
    // nothing to publish and is not given the tool at all.
    let push = input
        .implementer
        .as_ref()
        .map(|implementer| implementer.branch.clone())
        .filter(|branch| ratatoskr_agent::publish::pushable(branch))
        .map(|branch| ratatoskr_agent::publish::PushAccess {
            repo_root: repo_root.clone(),
            branch,
            // From the run, not from the publisher: the number is what the branch is *for*, and it
            // is not the naming step's to choose.
            issue: Some(input.issue.clone()),
        });
    let input_json = serde_json::to_string(&input)?;
    let rendered_question = publisher::render_prompt(&input);
    let declared_context =
        workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
            client,
            config,
            store,
            run_id,
            issue,
            engine,
            plugin_context: context.clone(),
            ledger: Arc::clone(ledger),
        })?;
    let raw = workflow::evaluate_standard_stage_with_resources(
        declared_context,
        "publisher",
        input_json,
        rendered_question,
        workflow::StandardStageResources {
            resource_root: repo_root,
            shell: None,
            publish: Some(workflow::StandardStagePublishResources { push }),
            clarifier: None,
            guidance: None,
        },
    )
    .await
    .map_err(|error| PlanError::node("publisher", NodeError::Failed(error)))?;
    let out: PublisherOutput = serde_json::from_str(&raw).map_err(|error| {
        PlanError::node(
            "publisher",
            NodeError::Failed(format!(
                "publisher output could not be reconstructed: {error}"
            )),
        )
    })?;
    record(Record {
        store,
        run_id,
        node: "publisher",
        output: &out,
        input: None,
        iteration: None,
        ledger: Some(ledger),
    })
    .await?;
    Ok(out)
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
        issue,
        engine,
        clarifier,
        ledger,
        context,
        ..
    } = run;
    let sink = client.map(RagRatClient::sink);
    let input_json = serde_json::to_string(&input)?;
    let out = if let Some(output) = bookkeeper::skipped_before_compose(&input, sink.is_some()) {
        output
    } else {
        let repo_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let rendered_question = bookkeeper::render_prompt(&input);
        let declared_context =
            workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
                client,
                config,
                store,
                run_id,
                issue,
                engine,
                plugin_context: context.clone(),
                ledger: Arc::clone(ledger),
            })?;
        let raw = workflow::evaluate_standard_stage_with_resources(
            declared_context,
            "bookkeeper",
            input_json.clone(),
            rendered_question,
            workflow::StandardStageResources {
                resource_root: repo_root,
                shell: None,
                publish: None,
                clarifier: Some(clarifier.as_dyn()),
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
            sink.as_ref()
                .expect("bookkeeper preflight requires a memory sink"),
            decisions.decisions,
            &input,
        )
        .await
        .map_err(|error| PlanError::node("bookkeeper", error))?
    };
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
    client: Option<&RagRatClient>,
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

/// The review stage: the verifier, plus the analyst re-entry it routes plan-level findings to.
///
/// Built once per run and reused across converge iterations, so a second review costs a model call
/// rather than a rebuild. `None` when the verifier has no route — like the red team's classifier,
/// it is opt-in by being given a model rather than by a separate switch.
pub(crate) struct Review {
    declared_context: Arc<workflow::WorkflowContext>,
    threshold: verifier::Severity,
    scout: ScoutOutput,
    memory: MemoryOutput,
    brief: String,
    constraints: Vec<Constraint>,
}

/// What a review concluded the run should do next.
///
/// `Unavailable` is the case that is easy to get wrong: a verifier that could not run has not
/// approved anything, but it has not found anything either. Its error is evidence about our
/// infrastructure, not about the change, so it must neither block the run nor pass as a clean
/// review.
pub(crate) enum Reviewed {
    /// Nothing above the threshold. The change is accepted.
    Clean,
    /// Send this back.
    Fix(Box<Correction>),
    /// The verifier could not be asked. The reason is on its checkpoint.
    Unavailable,
}

/// What a review concluded the run should do next.
pub(crate) struct Correction {
    /// What the review found, carried so the next pass can see what the last one said — and notice
    /// when a new finding exists only because of the fix for an old one.
    found: Vec<verifier::Finding>,
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
    offer: Option<ServerTools>,
    ledger: Option<Arc<RunLedger>>,
    declared_context: Option<Arc<workflow::WorkflowContext>>,
) -> Result<Option<testrun::Characterizer>, PlanError> {
    if node_route(engine, config, "characterizer").is_none() {
        return Ok(None);
    }
    // No skills either, and this is the seam that enforces it: `node_agent_config` grants the
    // `Skill` tool whenever a node has skills bound, while the hook that answers it is installed
    // from what the caller passes to `run_structured`. `Characterizer` passes none — so leaving
    // skills bound here would offer it a tool whose result nothing can produce, and a node that
    // reads a tool error as an instruction is a failure this repo has already paid for once.
    let mut plugins = NodePlugins {
        skills: Vec::new(),
        ..context.for_node("characterizer")
    };
    // No default tools: it transcribes output it was handed. Reading the repo would invite it to
    // decide whether a failure matters, which is the one thing it must not do.
    let cfg = stage_agent_config(
        engine,
        config,
        context.pool_for("characterizer", offer),
        "characterizer",
        &[],
        &mut plugins,
    )?;
    Ok(Some(testrun::Characterizer {
        route: cfg.route,
        tools: cfg.tools,
        max_turns: cfg.max_turns,
        ledger,
        declared_context: declared_context.expect("characterizer route requires its stage context"),
    }))
}

/// The implementer's agent configuration.
///
/// Distinct from every other node's in what it may reach: the editing tools and a shell, on top of
/// the read tools each node gets. Built in one place because those two powers belong together and
/// to exactly one node — a second construction site is how one of them comes to be granted
/// somewhere it was not meant to be.
fn build_implementer_agent(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    context: &PluginContext,
    offer: Option<ServerTools>,
) -> Result<(NodeAgentConfig, NodePlugins), PlanError> {
    let mut plugins = context.for_node("implementer");
    let mut tools = context.pool_for("implementer", offer);
    tools
        .local()
        .tools
        .extend(ratatoskr_agent::files::edit_declarations());
    tools
        .local()
        .tools
        .push(ratatoskr_agent::shell::declaration());
    // The implementer can ask. It has the most turns to spend and is the only node that changes
    // code, so it is the one most likely to meet a question worth asking — and the run-wide
    // `ASK_BUDGET` is what keeps that a relief valve rather than a way to spend a run.
    tools.local().tools.push(clarify::ask_tool());
    let mut cfg = stage_agent_config(
        engine,
        config,
        tools,
        "implementer",
        &implementer_default_tools(),
        &mut plugins,
    )?;
    // TOML governs the built-in implementer. A ruleset can still make a deliberate, more-specific
    // per-node override with `maxTurns`.
    if cfg.max_turns.is_none() {
        cfg.max_turns = Some(config.implementer.max_turns);
    }
    Ok((cfg, plugins))
}

/// The implementer's default `allow`: its rag-rat tools plus the ones that let it work — reading,
/// editing, and running commands. A ruleset naming its own `allow` replaces this wholesale, and
/// one that forgets `Write` or `Bash` leaves the node unable to do the job it exists for.
fn implementer_default_tools() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = implementer::IMPLEMENTER_TOOLS.to_vec();
    names.extend([
        ratatoskr_agent::files::READ,
        ratatoskr_agent::files::GREP,
        ratatoskr_agent::files::GLOB,
        ratatoskr_agent::files::WRITE,
        ratatoskr_agent::files::EDIT,
        ratatoskr_agent::shell::BASH,
    ]);
    names
}

/// Resolve a node's model from its ruleset first, then its TOML route.
fn node_route(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    node: &str,
) -> Option<ratatoskr_core::ModelRoute> {
    engine
        .ruleset(node)
        .and_then(|ruleset| ruleset.config().model.clone())
        .map(|model| ratatoskr_core::ModelRoute {
            provider: model.provider,
            model: model.model,
            max_tokens: None,
            context_window: None,
            temperature: None,
            params: None,
            session: Default::default(),
        })
        .or_else(|| stage::stage_profile(config, node).and_then(|profile| profile.model))
        .or_else(|| config.models.get(node).cloned())
}

/// The referee accepts only its TOML route, then falls back to the verifier's route. Its fixed
/// capability boundary is internal, so a `referee` ruleset is never consulted.
pub fn referee_route(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
) -> Option<ratatoskr_core::ModelRoute> {
    config
        .models
        .get("referee")
        .cloned()
        .or_else(|| node_route(engine, config, "verifier"))
}

/// The verifier is opt-in on having somewhere to run, the same way the red-team classifier is.
fn verifier_enabled(engine: &Arc<ScriptEngine>, config: &RatatoskrConfig) -> bool {
    node_route(engine, config, "verifier").is_some()
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
            store,
            run_id,
            issue,
            engine,
            context,
            ledger,
            ..
        } = run;
        if !verifier_enabled(engine, config) {
            return Ok(None);
        }

        let declared_context =
            workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
                client,
                config,
                store,
                run_id,
                issue,
                engine,
                plugin_context: context.clone(),
                ledger: Arc::clone(ledger),
            })?;
        Ok(Some(Review {
            declared_context,
            threshold: parse_threshold(&config.implementer.verify_threshold),
            scout: plan.scout.clone(),
            memory: plan.memory.clone(),
            brief: plan.brief.clone(),
            constraints: plan.constraints.clone(),
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
        previous_findings: &[verifier::Finding],
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
            previous_findings: previous_findings.to_vec(),
        };
        let input_json = serde_json::to_string(&serde_json::json!({
            "requirements": plan.requirements,
            "touched_files": input.touched_files,
            "diff_bytes": input.diff.len(),
        }))?;
        // A verifier that cannot run must not discard a change that was made and passed. Every
        // other fallible node here is best-effort for the same reason; this one propagating its
        // error was an oversight, and the run it cost had already implemented the task correctly.
        let out = match run_declared_verifier(&self.declared_context, &input, worktree).await {
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
        let found: Vec<verifier::Finding> = out.findings.clone();
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
            return Ok(Reviewed::Fix(Box::new(Correction {
                prompt: verifier::correction(&blocking),
                revised: None,
                found,
            })));
        }

        let revision = analyst::AnalystInput {
            issue: issue.to_string(),
            scout: self.scout.clone(),
            memory: self.memory.clone(),
            // Carried unchanged into the revision: what bears on the task and what constrains it
            // did not stop being true because the plan was wrong.
            brief: self.brief.clone(),
            constraints: self.constraints.clone(),
            previous: Some(Box::new(plan.clone())),
            findings: plan_faults,
        };
        let revision_json = serde_json::to_string(&revision)?;
        let revised = run_declared_analyst(&self.declared_context, &revision).await?;
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

        Ok(Reviewed::Fix(Box::new(Correction {
            prompt: replan(&revised, &blocking),
            revised: Some(revised),
            found,
        })))
    }
}

impl Review {
    /// Ask the analyst to look at the plan when the iteration budget is spent.
    ///
    /// The evidence for doing this rather than recording another failed attempt: on the run that
    /// prompted it, three passes found three *different* defects, each one existing because of the
    /// fix for the one before, with severity climbing P2 → P2 → P1. A fourth attempt at the same
    /// plan had nothing left to find. What was wrong was a decision made before any of it.
    ///
    /// Every finding goes over, not just the plan-tagged ones — the whole point is that the
    /// verifier called them execution faults one at a time and the pattern only shows in the set.
    async fn replan_at_ceiling(
        &self,
        run: &Run<'_>,
        plan: &AnalystOutput,
        findings: &[verifier::Finding],
        iteration: u32,
    ) -> Result<Option<(AnalystOutput, String)>, PlanError> {
        let &Run {
            store,
            run_id,
            issue,
            ledger,
            ..
        } = run;
        let revision = analyst::AnalystInput {
            issue: issue.to_string(),
            scout: self.scout.clone(),
            memory: self.memory.clone(),
            brief: self.brief.clone(),
            constraints: self.constraints.clone(),
            previous: Some(Box::new(plan.clone())),
            findings: findings.to_vec(),
        };
        let revision_json = serde_json::to_string(&revision)?;
        let revised = match run_declared_analyst(&self.declared_context, &revision).await {
            Ok(revised) => revised,
            // Best-effort, like every other recovery here: a failed re-plan leaves the run
            // recording what it already knew rather than losing the work as well.
            Err(e) => {
                tracing::warn!("the analyst could not re-plan at the ceiling: {e}");
                return Ok(None);
            }
        };
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
        let borrowed: Vec<&verifier::Finding> = findings.iter().collect();
        Ok(Some((revised.clone(), replan(&revised, &borrowed))))
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
    let repo_path: PathBuf = std::env::current_dir()
        .map_err(|e| PlanError::node("fork", NodeError::Failed(format!("cwd: {e}"))))?;
    let acceptance = run.config.sandbox.acceptance(&plan.analyst.acceptance);
    tracing::info!(
        steps = ?acceptance.iter().map(|s| &s.name).collect::<Vec<_>>(),
        "acceptance for this run"
    );

    // --- build agents ---
    let (red_team, implementer) = build_converge_agents(run, plan, &repo_path, acceptance)?;
    let review = build_reviewers(run, plan)?;

    // --- fork worktree ---
    let worktree = fork_worktree(&implementer).await?;

    // --- red-team baseline ---
    let red_team_out = red_team_baseline(&red_team, &worktree, run.issue, plan).await?;

    // --- first implementer attempt ---
    let impl_out = first_implementer_attempt(&implementer, &worktree).await?;
    record_initial_attempt(run, plan, &red_team_out, &impl_out).await?;
    validate_baseline(&red_team_out)?;

    // --- converge ---
    let (impl_out, status, iterations) = converge(ConvergeInput {
        run,
        plan,
        repo_path: &repo_path,
        implementer: &implementer,
        worktree: &worktree,
        red_team_out: &red_team_out,
        impl_out,
        review: review.as_ref(),
    })
    .await?;

    // --- final commit ---
    commit_run(run, &implementer, &worktree, &impl_out).await;
    Ok((red_team_out, impl_out, worktree, status, iterations))
}

/// Build the red-team and implementer agents used by the fork.
pub(crate) fn build_converge_agents(
    run: &Run<'_>,
    plan: &PlanOutcome,
    repo_path: &Path,
    acceptance: Vec<ratatoskr_core::AcceptanceStep>,
) -> Result<(RedTeamNode, ImplementerNode), PlanError> {
    let &Run {
        client,
        config,
        run_id,
        issue,
        engine,
        clarifier,
        ledger,
        context,
        ..
    } = run;
    let short: String = run_id.chars().take(8).collect();
    let conventions = repo_conventions(repo_path);
    let redteam_enabled = classifier_enabled(engine, config);
    let redteam_context = if redteam_enabled {
        Some(workflow::WorkflowContext::new_with_ledger(
            workflow::WorkflowContextParams {
                client,
                config,
                store: run.store,
                run_id,
                issue,
                engine,
                plugin_context: context.clone(),
                ledger: Arc::clone(ledger),
            },
        )?)
    } else {
        None
    };
    let characterizer_context = if node_route(engine, config, "characterizer").is_some() {
        Some(workflow::WorkflowContext::new_with_ledger(
            workflow::WorkflowContextParams {
                client,
                config,
                store: run.store,
                run_id,
                issue,
                engine,
                plugin_context: context.clone(),
                ledger: Arc::clone(ledger),
            },
        )?)
    } else {
        None
    };
    let red_team = RedTeamNode {
        repo_path: repo_path.to_path_buf(),
        worktree_root: config.worktree.root.clone(),
        baseline_branch: format!("ratatoskr/{short}-baseline"),
        sandbox: config.sandbox.clone(),
        name: format!("ratatoskr-redteam-{short}"),
        acceptance: acceptance.clone(),
        characterizer: build_characterizer(
            engine,
            config,
            context,
            client.map(|c| c.offer()),
            Some(Arc::clone(ledger)),
            characterizer_context.as_ref().map(Arc::clone),
        )?,
        classifier: match redteam_enabled {
            true => {
                let mut plugins_redteam = context.for_node("redteam");
                let cfg = stage_agent_config(
                    engine,
                    config,
                    context.pool_for("redteam", client.map(|c| c.offer())),
                    "redteam",
                    redteam::CLASSIFIER_TOOLS,
                    &mut plugins_redteam,
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
                    declared_context: Arc::clone(
                        redteam_context
                            .as_ref()
                            .expect("red-team route requires its stage context"),
                    ),
                })
            }
            false => None,
        },
        author: match redteam_enabled {
            true => {
                let mut plugins = context.for_node("redteam");
                let mut tools = context.pool_for("redteam", client.map(|c| c.offer()));
                tools
                    .local()
                    .tools
                    .extend(ratatoskr_agent::files::edit_declarations());
                let cfg = plugins::redteam_author_agent_config(
                    engine,
                    config,
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
                    conventions: conventions.clone(),
                    plugins,
                    ledger: Some(Arc::clone(ledger)),
                    declared_context: redteam_context
                        .expect("red-team route requires its stage context"),
                })
            }
            false => None,
        },
    };
    let (impl_cfg, impl_plugins) =
        build_implementer_agent(engine, config, context, client.map(|c| c.offer()))?;
    let implementer_context =
        workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
            client,
            config,
            store: run.store,
            run_id,
            issue,
            engine,
            plugin_context: context.clone(),
            ledger: Arc::clone(ledger),
        })?;
    let implementer = ImplementerNode {
        clarifier: Some(clarifier.as_dyn()),
        repo_path: repo_path.to_path_buf(),
        worktree_root: config.worktree.root.clone(),
        sandbox: config.sandbox.clone(),
        route: impl_cfg.route,
        tools: impl_cfg.tools,
        policy: impl_cfg.policy,
        max_turns: impl_cfg.max_turns,
        system_prompt: impl_cfg.system_prompt,
        conventions,
        plugins: impl_plugins,
        ledger: Some(Arc::clone(ledger)),
        run_id: run_id.to_string(),
        issue: issue.to_string(),
        analyst: plan.analyst.clone(),
        acceptance,
        characterizer: build_characterizer(
            engine,
            config,
            context,
            client.map(|c| c.offer()),
            Some(Arc::clone(ledger)),
            characterizer_context,
        )?,
        declared_context: implementer_context,
    };
    Ok((red_team, implementer))
}

/// Build the optional reviewer before work is spent in the fork.
pub(crate) fn build_reviewers(
    run: &Run<'_>,
    plan: &PlanOutcome,
) -> Result<Option<Review>, PlanError> {
    Review::build(run, plan)
}

/// Create the worktree that the red team and implementer share.
pub(crate) async fn fork_worktree(
    implementer: &ImplementerNode,
) -> Result<WorktreePath, PlanError> {
    implementer
        .prepare()
        .await
        .map_err(|e| PlanError::node("implementer", e))
}

/// Characterize the baseline and author any tests before the implementer opens the tree.
pub(crate) async fn red_team_baseline(
    red_team: &RedTeamNode,
    worktree: &WorktreePath,
    issue: &str,
    plan: &PlanOutcome,
) -> Result<RedTeamOutput, PlanError> {
    red_team
        .run_and_author(worktree.as_path(), issue, &plan.analyst.interface)
        .await
        .map_err(|e| PlanError::node("red_team", e))
}

/// Run the first implementation attempt, discarding its worktree on failure.
pub(crate) async fn first_implementer_attempt(
    implementer: &ImplementerNode,
    worktree: &WorktreePath,
) -> Result<ImplementerOutput, PlanError> {
    match implementer.work(worktree).await {
        Ok(out) => Ok(out),
        Err(e) => {
            implementer.discard(worktree).await;
            Err(PlanError::node("implementer", e))
        }
    }
}

/// Record the baseline and first implementation at their original ledger iteration.
pub(crate) async fn record_initial_attempt(
    run: &Run<'_>,
    plan: &PlanOutcome,
    red_team_out: &RedTeamOutput,
    impl_out: &ImplementerOutput,
) -> Result<(), PlanError> {
    record(Record {
        store: run.store,
        run_id: run.run_id,
        node: "red_team",
        output: red_team_out,
        input: None,
        iteration: Some(1),
        ledger: Some(run.ledger),
    })
    .await?;
    record(Record {
        store: run.store,
        run_id: run.run_id,
        node: "implementer",
        output: impl_out,
        input: Some(serde_json::to_string(&plan.analyst)?),
        iteration: Some(1),
        ledger: Some(run.ledger),
    })
    .await
}

/// Reject a baseline that did not produce any acceptance result.
pub(crate) fn validate_baseline(red_team_out: &RedTeamOutput) -> Result<(), PlanError> {
    if converge::test_command_ran(
        &red_team_out.failing_tests,
        red_team_out.passed_tests,
        red_team_out.exit_code,
    ) {
        return Ok(());
    }
    Err(PlanError::node(
        "red_team",
        NodeError::Failed(format!(
            "the baseline acceptance run produced no checks (exit {}); \
             check the analyst's acceptance, [sandbox] test_command and the sandbox backend",
            red_team_out.exit_code
        )),
    ))
}

/// The live context needed only when a clean test result reaches review.
pub(crate) struct ReviewRequest<'a, 'run> {
    review: &'a Review,
    run: &'a Run<'run>,
    plan: &'a AnalystOutput,
    worktree: &'a WorktreePath,
    iteration: u32,
    findings: &'a [verifier::Finding],
}

/// Apply the referee, test, and verifier gates in that order.
pub(crate) async fn decide_correction(
    referee_violations: &[referee::Violation],
    red_team_out: &RedTeamOutput,
    impl_out: &ImplementerOutput,
    authored: &[String],
    review: Option<ReviewRequest<'_, '_>>,
) -> Result<Reviewed, PlanError> {
    if !referee_violations.is_empty() {
        tracing::warn!(violations = ?referee_violations, "iteration weakened the referee; not accepting it");
        return Ok(Reviewed::Fix(Box::new(Correction {
            prompt: referee::correction(referee_violations),
            revised: None,
            found: Vec::new(),
        })));
    }

    let post_ran = converge::test_command_ran(
        &impl_out.failing_tests,
        impl_out.passed_tests,
        impl_out.exit_code,
    );
    let unsatisfied = converge::unsatisfied(authored, &impl_out.failing_tests);
    let tests_clean = post_ran
        && unsatisfied.is_empty()
        && converge::is_converged(&red_team_out.failing_tests, &impl_out.failing_tests);
    if !tests_clean {
        let prompt = if !post_ran {
            format!(
                "The test command did not run to completion (exit {}) — your change likely \
                 does not compile. Fix it so the tests run and pass.",
                impl_out.exit_code
            )
        } else if !unsatisfied.is_empty() {
            format!(
                "These tests were written for this change, from the interface, before any code \
                 existed to satisfy them — making them pass is what the change is for, and \
                 they are still failing: {}. They are not yours to edit; implement what they \
                 describe. If one of them is wrong about the contract rather than about your \
                 code, say so in your summary and implement the rest.",
                unsatisfied.join(", ")
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
        return Ok(Reviewed::Fix(Box::new(Correction {
            prompt,
            revised: None,
            found: Vec::new(),
        })));
    }

    match review {
        Some(review) => {
            review
                .review
                .review(
                    review.run,
                    review.plan,
                    impl_out,
                    review.worktree,
                    review.iteration,
                    review.findings,
                )
                .await
        }
        None => Ok(Reviewed::Clean),
    }
}

/// The decision reached when the iteration budget has been spent.
pub(crate) enum CeilingDecision {
    Stop(RunStatus),
    Replan(Box<Correction>),
}

/// Optionally escalate the accumulated findings once, after the budget is spent.
pub(crate) async fn at_ceiling(
    iterations: u32,
    max_iterations: u32,
    replanned: bool,
    findings: &[verifier::Finding],
    review: Option<ReviewRequest<'_, '_>>,
) -> Result<CeilingDecision, PlanError> {
    debug_assert!(iterations >= max_iterations);
    if replanned || findings.is_empty() {
        return Ok(CeilingDecision::Stop(RunStatus::MaxIterationsReached));
    }
    let Some(review) = review else {
        return Ok(CeilingDecision::Stop(RunStatus::MaxIterationsReached));
    };
    tracing::warn!(
        iterations,
        findings = findings.len(),
        "the iteration budget is spent; asking the analyst to look at the plan rather than recording another failed attempt"
    );
    match review
        .review
        .replan_at_ceiling(review.run, review.plan, findings, iterations)
        .await?
    {
        Some((revised, prompt)) => Ok(CeilingDecision::Replan(Box::new(Correction {
            prompt,
            revised: Some(revised),
            found: Vec::new(),
        }))),
        None => Ok(CeilingDecision::Stop(RunStatus::MaxIterationsReached)),
    }
}

/// Record one corrective implementer attempt.
pub(crate) async fn record_iteration(
    run: &Run<'_>,
    impl_out: &ImplementerOutput,
    input: String,
    iteration: u32,
) -> Result<(), PlanError> {
    record(Record {
        store: run.store,
        run_id: run.run_id,
        node: "implementer",
        output: impl_out,
        input: Some(input),
        iteration: Some(iteration),
        ledger: Some(run.ledger),
    })
    .await
}

/// Everything the convergence loop carries between its named decision stages.
pub(crate) struct ConvergeInput<'a, 'run> {
    run: &'a Run<'run>,
    plan: &'a PlanOutcome,
    repo_path: &'a Path,
    implementer: &'a ImplementerNode,
    worktree: &'a WorktreePath,
    red_team_out: &'a RedTeamOutput,
    impl_out: ImplementerOutput,
    review: Option<&'a Review>,
}

/// Iterate until the referee, tests, and review accept the current work.
pub(crate) async fn converge(
    input: ConvergeInput<'_, '_>,
) -> Result<(ImplementerOutput, RunStatus, u32), PlanError> {
    let ConvergeInput {
        run,
        plan,
        repo_path,
        implementer,
        worktree,
        red_team_out,
        mut impl_out,
        review,
    } = input;
    let mut in_force = plan.analyst.clone();
    let mut iterations = 1u32;
    let mut found_so_far: Vec<verifier::Finding> = Vec::new();
    let mut replanned = false;

    let status = loop {
        let referee_violations = match referee::judge(
            run.engine,
            run.config,
            run.ledger,
            run.issue,
            &in_force.requirements,
            &impl_out,
            worktree,
        )
        .await
        {
            Ok(Some(violations)) => {
                if let Err(error) = record(Record {
                    store: run.store,
                    run_id: run.run_id,
                    node: "referee",
                    output: &referee::RefereeOutput {
                        violations: violations.clone(),
                    },
                    input: None,
                    iteration: Some(iterations),
                    ledger: Some(run.ledger),
                })
                .await
                {
                    tracing::warn!("failed to record referee judgement: {error}");
                }
                violations
            }
            Ok(None) => Vec::new(),
            Err(error) => {
                tracing::warn!(
                    "the referee could not judge this change; trusting test results: {error}"
                );
                if let Err(record_error) = record(Record {
                    store: run.store,
                    run_id: run.run_id,
                    node: "referee",
                    output: &serde_json::json!({ "error": error.to_string() }),
                    input: None,
                    iteration: Some(iterations),
                    ledger: Some(run.ledger),
                })
                .await
                {
                    tracing::warn!("failed to record referee failure: {record_error}");
                }
                Vec::new()
            }
        };
        let authored = red_team_out
            .authored
            .as_ref()
            .map(|a| a.tests.as_slice())
            .unwrap_or_default();
        let review_request = review.map(|review| ReviewRequest {
            review,
            run,
            plan: &in_force,
            worktree,
            iteration: iterations,
            findings: &found_so_far,
        });
        let correction = match decide_correction(
            &referee_violations,
            red_team_out,
            &impl_out,
            authored,
            review_request,
        )
        .await?
        {
            Reviewed::Clean => break RunStatus::Converged,
            Reviewed::Unavailable => break RunStatus::Unreviewed,
            Reviewed::Fix(correction) => *correction,
        };
        found_so_far.extend(correction.found.iter().cloned());
        if let Some(revised) = correction.revised.as_ref() {
            in_force = revised.clone();
            replanned = true;
        }
        if iterations >= run.config.implementer.max_iterations {
            let ceiling_review = review.map(|review| ReviewRequest {
                review,
                run,
                plan: &in_force,
                worktree,
                iteration: iterations,
                findings: &found_so_far,
            });
            match at_ceiling(
                iterations,
                run.config.implementer.max_iterations,
                replanned,
                &found_so_far,
                ceiling_review,
            )
            .await?
            {
                CeilingDecision::Stop(status) => break status,
                CeilingDecision::Replan(correction) => {
                    let revised = correction
                        .revised
                        .expect("ceiling replans carry a revision");
                    in_force = revised;
                    replanned = true;
                    iterations += 1;
                    impl_out = match implementer.iterate(worktree, &correction.prompt).await {
                        Ok(out) => out,
                        Err(e) => {
                            if let Err(rm) = remove_worktree(repo_path, worktree).await {
                                tracing::warn!("failed to clean up worktree: {rm}");
                            }
                            return Err(PlanError::node("implementer", e));
                        }
                    };
                    record_iteration(
                        run,
                        &impl_out,
                        serde_json::to_string(&in_force)?,
                        iterations,
                    )
                    .await?;
                    continue;
                }
            }
        }
        impl_out = match implementer.iterate(worktree, &correction.prompt).await {
            Ok(out) => out,
            Err(e) => {
                if let Err(rm) = remove_worktree(repo_path, worktree).await {
                    tracing::warn!("failed to clean up worktree after converge error: {rm}");
                }
                return Err(PlanError::node("implementer", e));
            }
        };
        record_iteration(
            run,
            &impl_out,
            serde_json::to_string(&correction.prompt)?,
            iterations + 1,
        )
        .await?;
        iterations += 1;
    };
    Ok((impl_out, status, iterations))
}

/// Commit the run branch regardless of the settled outcome.
pub(crate) async fn commit_run(
    run: &Run<'_>,
    implementer: &ImplementerNode,
    worktree: &WorktreePath,
    impl_out: &ImplementerOutput,
) {
    let branch = implementer.branch();
    match ratatoskr_exec::commit_all(
        worktree,
        &branch,
        &commit_message(&run.config.publish, run.issue, impl_out),
        ratatoskr_exec::Committer {
            name: &run.config.publish.committer_name,
            email: &run.config.publish.committer_email,
        },
    )
    .await
    {
        Ok(Some(sha)) => {
            tracing::info!(kind = "committed", branch = %branch, sha = %sha, "committed")
        }
        Ok(None) => tracing::info!(branch = %branch, "nothing to commit"),
        Err(e) => tracing::warn!("could not commit the run's work to {branch}: {e}"),
    }
}

/// The message a run's commit carries.
///
/// The subject is composed from what the implementer said about its own change — type, scope and a
/// one-line subject — through `[publish] commit_subject`, so a repository whose history is not
/// conventional-commit shaped can say so rather than have this one imposed on it.
///
/// It is not the issue's first line. The issue says what was wanted and the commit says what was
/// done, and taking the former let a title longer than the limit be cut mid-word — a subject
/// ending "a fabricated tool res" reads as a truncated change, not a truncated string.
///
/// The body is the implementer's own account of what it changed and why — the only description
/// written by the thing that made the change. Not the diffstat, which `git log --stat` produces on
/// demand and which answers "which files" when the question a reader has is "why".
fn commit_message(
    publish: &ratatoskr_core::PublishConfig,
    issue: &str,
    out: &ImplementerOutput,
) -> String {
    // A model that reported nothing usable still has to produce a commit, and the issue's first
    // line is the only other thing that describes the work. Trimmed to a word boundary by the same
    // renderer, so the fallback cannot reintroduce the truncation it replaced.
    let subject = match out.commit_subject.trim().is_empty() {
        false => publish.commit_subject(&out.commit_kind, &out.commit_scope, &out.commit_subject),
        true => publish.commit_subject(
            "chore",
            "",
            issue
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("a change"),
        ),
    };
    let body = match out
        .narrative
        .as_deref()
        .map(str::trim)
        .filter(|n| !n.is_empty())
    {
        Some(narrative) => format!("\n\n{}", wrapped(narrative)),
        None => String::new(),
    };
    format!("{subject}{body}")
}

/// Most of one line of a commit body. The git convention, and what a terminal shows without
/// wrapping it somewhere the author did not choose.
const BODY_WIDTH: usize = 72;

/// Rewrap prose to [`BODY_WIDTH`], preserving paragraph and list structure.
///
/// A model writes one long line per paragraph, and `git log` does not wrap, so unwrapped prose
/// reads as a single line running off the terminal. Wrapping is done here rather than asked of the
/// model: a model told to wrap at 72 counts characters unreliably and spends attention doing it.
///
/// A line that is already short, or that begins a list item, is left alone — reflowing a bullet
/// list into a paragraph loses the structure that made it readable.
fn wrapped(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_end();
        let is_list = trimmed.trim_start().starts_with(['-', '*', '•'])
            || trimmed
                .trim_start()
                .split_once('.')
                .is_some_and(|(n, _)| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
        if trimmed.chars().count() <= BODY_WIDTH || is_list {
            out.push(trimmed.to_string());
            continue;
        }
        let mut current = String::new();
        for word in trimmed.split_whitespace() {
            // `+ 1` for the space this word would need. A word longer than the width goes on its
            // own line whole rather than being broken — it is a path or an identifier.
            if !current.is_empty()
                && current.chars().count() + 1 + word.chars().count() > BODY_WIDTH
            {
                out.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod agent_config_tests {
    use super::*;

    #[test]
    fn the_implementer_can_correct_a_memory_its_change_falsifies() {
        // The defect this closes: a review that found a memory contradicted by the diff routed the
        // finding here, and this node could read memories and write none — so converge asked for a
        // fix nobody in the run could make, every iteration, until the budget ran out.
        let tools = implementer_default_tools();
        assert!(tools.contains(&"memory_update"), "{tools:?}");
        assert!(tools.contains(&"memory_mark_obsolete"), "{tools:?}");
        // And composing new ones stays the bookkeeper's, done once with the whole run in view.
        assert!(!tools.contains(&"memory_create"), "{tools:?}");
    }

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

    /// A config and cwd that discover exactly `names` — bare plugins, manifest only, so `resolve`
    /// runs no hooks and connects no servers. The cwd holds no `.ratatoskr/plugins` of its own:
    /// what the test discovers must not depend on the real checkout it happens to run in.
    fn resolve_fixture(case: &str, names: &[&str]) -> (RatatoskrConfig, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "ratatoskr-nodes-resolve-{}-{case}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let plugins = root.join("plugins");
        for name in names {
            let manifest = plugins.join(name).join(".claude-plugin");
            std::fs::create_dir_all(&manifest).unwrap();
            std::fs::write(
                manifest.join("plugin.json"),
                format!(r#"{{"name": "{name}"}}"#),
            )
            .unwrap();
        }
        let cwd = root.join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let mut config = RatatoskrConfig::default();
        config.plugins.paths = vec![plugins];
        (config, cwd)
    }

    #[tokio::test]
    async fn an_agent_plugin_nobody_installed_still_fails_the_run() {
        // The gate stays for an explicit binding: a defineAgent name that matches nothing
        // installed is a typo in all three spellings, and fails on the `plugins` node with the
        // message naming both the missing plugin(s) and the discovered list.
        for (case, rule) in [
            ("only", r#"defineAgent("analyst", { plugins: ["ghost"] });"#),
            (
                "add",
                r#"defineAgent("scout", { plugins: { add: ["ghost"] } });"#,
            ),
            (
                "remove",
                r#"defineAgent("scout", { plugins: { remove: ["ghost"] } });"#,
            ),
        ] {
            let engine = binding_engine(&format!("agent-missing-{case}"), rule).await;
            let (config, cwd) = resolve_fixture(&format!("agent-missing-{case}"), &["ponytail"]);

            let error = match PluginContext::resolve(&config, &engine, &cwd).await {
                Err(error) => error,
                Ok(_) => {
                    panic!("an explicit binding naming an uninstalled plugin must fail ({case})")
                }
            };
            let message = match error {
                PlanError::Node {
                    node: "plugins",
                    source,
                } => source.to_string(),
                other => panic!("the failure belongs to the `plugins` node, got {other}"),
            };
            assert!(
                message.contains("ruleset names plugin(s) that were not found"),
                "{case}: the existing message shape: {message}"
            );
            assert!(
                message.contains("ghost"),
                "{case}: the missing name is reported: {message}"
            );
            assert!(
                message.contains("ponytail"),
                "{case}: the discovered list is reported: {message}"
            );
        }
    }

    #[tokio::test]
    async fn a_plugin_named_by_both_defaults_and_an_agent_still_fails_when_undiscovered() {
        // The explicit binding wins over the default preference: once a defineAgent names the
        // plugin, a miss is a requirement again — warn-and-narrow is for defaults-only names.
        let engine = binding_engine(
            "both-missing",
            r#"
            defineDefaults({ plugins: ["rag-rat"] });
            defineAgent("analyst", { plugins: { add: ["rag-rat"] } });
            "#,
        )
        .await;
        let (config, cwd) = resolve_fixture("both-missing", &["ponytail"]);

        let error = match PluginContext::resolve(&config, &engine, &cwd).await {
            Err(error) => error,
            Ok(_) => panic!("an agent-level binding makes the missing plugin required"),
        };
        let message = error.to_string();
        assert!(message.contains("rag-rat"), "{message}");
        assert!(message.contains("ponytail"), "{message}");
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
    fn agent_profile_guidance_precedes_the_stage_preamble() {
        assert_eq!(
            effective_preamble_with_profile("n", "stage", "profile", None, None, &[]),
            "profile\n\nstage"
        );
        assert_eq!(
            effective_preamble_with_profile("n", "stage", "profile", Some("override"), None, &[]),
            "profile\n\noverride"
        );
    }

    #[test]
    fn plugin_context_prefixes_whichever_preamble_applies() {
        // A ruleset replaces the node's own text; plugin context is prepended to whatever wins,
        // so a repository digest never costs a node its instructions.
        assert_eq!(
            effective_preamble("n", "built-in", None, None, &[]),
            "built-in"
        );
        assert_eq!(
            effective_preamble("n", "built-in", Some("override"), None, &[]),
            "override"
        );
        assert_eq!(
            effective_preamble("n", "built-in", None, Some("digest"), &[]),
            "digest\n\nbuilt-in"
        );
        assert_eq!(
            effective_preamble("n", "built-in", Some("override"), Some("digest"), &[]),
            "digest\n\noverride"
        );
    }

    // --- the skill listing lives in the preamble (issue #143) ---
    //
    // `effective_preamble` gained a `skills` parameter: it appends an 'Available skills:'
    // listing of every deduped bound skill's name and description to whichever base preamble
    // applies, so the tool schema can stay constant. These exercise that contracted signature.

    /// A skill ready to bind, as the plugin loader hands them out.
    fn bound_skill(name: &str, description: &str) -> ratatoskr_plugin::Skill {
        ratatoskr_plugin::Skill {
            name: name.to_string(),
            description: description.to_string(),
            body: format!("do {name} in ${{CLAUDE_SKILL_DIR}}"),
            dir: std::path::PathBuf::from(format!("/plugins/{name}")),
        }
    }

    #[test]
    fn the_preamble_lists_every_bound_skill_whichever_base_applies() {
        let skills = [
            bound_skill("dream-review", "when triaging findings"),
            bound_skill("using-rag-rat", "when navigating the repository"),
        ];
        let preamble = effective_preamble("n", "base", None, None, &skills);
        assert!(preamble.contains("base"), "{preamble}");
        assert!(
            preamble.contains("Available skills:"),
            "the listing lives in the preamble now: {preamble}"
        );
        for s in &skills {
            assert!(preamble.contains(&s.name), "missing {}: {preamble}", s.name);
            assert!(
                preamble.contains(&s.description),
                "missing the description of {}: {preamble}",
                s.name
            );
        }
    }

    #[test]
    fn a_ruleset_preamble_override_cannot_disenfranchise_a_bound_skill() {
        // `systemPrompt` replaces the node's built-in text, not the skills its plugins ship: the
        // listing is appended to whichever preamble applies.
        let skills = [
            bound_skill("dream-review", "when triaging findings"),
            bound_skill("using-rag-rat", "when navigating the repository"),
        ];
        let preamble =
            effective_preamble("n", "built-in", Some("override"), Some("digest"), &skills);
        assert!(preamble.contains("digest"), "{preamble}");
        assert!(preamble.contains("override"), "{preamble}");
        for s in &skills {
            assert!(
                preamble.contains(&s.name),
                "overriding the preamble dropped {}: {preamble}",
                s.name
            );
            assert!(
                preamble.contains(&s.description),
                "overriding the preamble dropped the description of {}: {preamble}",
                s.name
            );
        }
    }

    #[test]
    fn no_skills_bound_leaves_the_preamble_byte_identical() {
        // The empty slice is the ordinary case — most nodes bind no plugin — and it must not
        // grow an empty 'Available skills' section.
        assert_eq!(
            effective_preamble("n", "built-in", None, None, &[]),
            "built-in"
        );
        assert_eq!(
            effective_preamble("n", "built-in", Some("override"), None, &[]),
            "override"
        );
        assert_eq!(
            effective_preamble("n", "built-in", None, Some("digest"), &[]),
            "digest\n\nbuilt-in"
        );
        assert_eq!(
            effective_preamble("n", "built-in", Some("override"), Some("digest"), &[]),
            "digest\n\noverride"
        );
    }

    #[test]
    fn a_long_skill_description_is_listed_in_full_in_the_preamble() {
        // The budget that used to drop this skill existed because the listing was a tool schema;
        // as preamble prose there is nothing to be full of.
        let description = format!("when the {} case applies", "very long ".repeat(500));
        let skills = [bound_skill("verbose", &description)];
        let preamble = effective_preamble("n", "base", None, None, &skills);
        assert!(
            preamble.contains(&description),
            "the whole description is listed, not dropped to fit a budget"
        );
    }

    #[tokio::test]
    async fn ruleset_model_replaces_the_toml_route() {
        let engine = engine("model-override").await;
        let mut config = RatatoskrConfig::default();
        // The whole point: no `[models.scout]` entry at all.
        config.models.remove("scout");

        let cfg = stage_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            "scout",
            &[],
            &mut NodePlugins::default(),
        )
        .unwrap();
        assert_eq!(cfg.route.provider, "openai");
        assert_eq!(cfg.route.model, "gpt-5");
        assert_eq!(cfg.system_prompt.as_deref(), Some("Be brief."));
    }

    #[tokio::test]
    async fn a_node_that_declares_no_tools_is_not_handed_the_file_tools() {
        // "No tools" has to mean none. The characterizer transcribes output it was handed, and
        // when this leaked it spent its turn reading directories on the host and inventing a
        // diagnosis of the run instead of naming the checks.
        let engine = engine("no-tools").await;
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "characterizer".to_string(),
            ratatoskr_core::ModelRoute {
                context_window: None,
                provider: "anthropic".into(),
                model: "claude-haiku-4-5-20251001".into(),
                max_tokens: None,
                temperature: None,
                params: None,
                session: Default::default(),
            },
        );

        let none = stage_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            "characterizer",
            &[],
            &mut NodePlugins::default(),
        )
        .unwrap();
        assert!(none.tools.names().is_empty(), "{:?}", none.tools.names());
        // The root is still set, and that is not a capability: with no file tools offered there is
        // nothing to resolve against it. Gating the root on this list instead is what left the
        // publisher holding a `gh` stand-in that errors — it declares no default tools on purpose,
        // and the root is what lets the tool it *is* given resolve.
        assert!(none.files.is_some(), "the root is not the capability");

        // A node that does declare reach still gets them — this is the reading half of the
        // pipeline, not an exception for one node.
        let some = stage_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            "analyst",
            &["impact_surface", "symbol_lookup", "semantic_search"],
            &mut NodePlugins::default(),
        )
        .unwrap();
        assert!(some.files.is_some());
        assert!(some.tools.names().iter().any(|n| n == "Read"));
    }

    #[test]
    fn the_publishers_gh_resolves_to_something_that_can_actually_run() {
        // The failure this guards, seen on a live run: `gh` fell through to the stand-in whose
        // message says the tool "is answered inside the run and should never have been
        // dispatched". The publisher read that as an instruction and reported publishing nothing,
        // with reasoning that sounded entirely deliberate.
        let root = std::env::current_dir().expect("a working directory");
        assert!(
            ratatoskr_agent::publish::implementation(ratatoskr_agent::publish::GH, &root).is_some(),
            "with a root, `gh` is a real tool"
        );
        // Without one there is nothing to run it in, which is exactly the state the publisher was
        // left in.
        let mut tools = ToolSet::default();
        tools
            .local()
            .tools
            .push(ratatoskr_agent::publish::declaration());
        assert!(
            tools
                .names()
                .iter()
                .any(|n| n == ratatoskr_agent::publish::GH)
        );
    }

    #[tokio::test]
    async fn a_ruleset_without_a_model_still_falls_back_to_toml() {
        let engine = engine("toml-fallback").await;
        let config = RatatoskrConfig::default();

        let cfg = stage_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            "bookkeeper",
            &[],
            &mut NodePlugins::default(),
        )
        .unwrap();
        assert_eq!(cfg.route.provider, config.models["bookkeeper"].provider);
        assert_eq!(cfg.route.model, config.models["bookkeeper"].model);
        assert!(cfg.system_prompt.is_none());
        assert_eq!(cfg.max_turns, Some(3));
    }

    #[tokio::test]
    async fn toml_configures_the_implementer_turn_cap_unless_its_ruleset_overrides_it() {
        let mut config = RatatoskrConfig::default();
        config.implementer.max_turns = 250;
        let context = PluginContext::default();

        let engine = engine("implementer-toml-turn-cap").await;
        let (cfg, _) = build_implementer_agent(&engine, &config, &context, None).unwrap();
        assert_eq!(cfg.max_turns, Some(250));

        let engine = binding_engine(
            "implementer-ruleset-turn-cap",
            r#"defineAgent("implementer", { maxTurns: 7 });"#,
        )
        .await;
        let (cfg, _) = build_implementer_agent(&engine, &config, &context, None).unwrap();
        assert_eq!(cfg.max_turns, Some(7));
    }

    #[tokio::test]
    async fn no_ruleset_and_no_toml_route_is_still_an_error() {
        let engine = engine("no-route").await;
        let mut config = RatatoskrConfig::default();
        config.models.remove("analyst");

        assert!(matches!(
            stage_agent_config(
                &engine,
                &config,
                ToolSet::default(),
                "analyst",
                &[],
                &mut NodePlugins::default(),
            ),
            Err(PlanError::MissingRoute(n)) if n == "analyst"
        ));
    }

    #[tokio::test]
    async fn a_built_in_stage_uses_its_selected_profile() {
        let engine = binding_engine("build-profile", "").await;
        let mut config = RatatoskrConfig::default();
        config.agents.insert(
            "build".to_string(),
            ratatoskr_core::AgentProfileConfig {
                model: Some(ratatoskr_core::ModelRoute {
                    provider: "test".to_string(),
                    model: "profile-model".to_string(),
                    max_tokens: None,
                    context_window: None,
                    temperature: None,
                    params: None,
                    session: Default::default(),
                }),
                base_prompt: "profile prompt".to_string(),
                capabilities: vec![ratatoskr_core::Capability::Read],
                tool_policy: None,
                max_turns: Some(7),
            },
        );

        let mut plugins = NodePlugins::default();
        let cfg = stage_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            "implementer",
            &implementer_default_tools(),
            &mut plugins,
        )
        .unwrap();
        assert_eq!(cfg.route.model, "profile-model");
        assert_eq!(cfg.system_prompt, None);
        assert_eq!(plugins.profile_prompt, "profile prompt");
        assert_eq!(cfg.max_turns, Some(7));
        assert_eq!(cfg.tools.names(), ["Read", "Grep", "Glob"]);
    }

    #[tokio::test]
    async fn a_read_stage_does_not_widen_an_explicit_write_allow_list() {
        let engine = binding_engine(
            "read-ceiling-allow-list",
            r#"defineAgent("analyst", { tools: { allow: ["Write"] } });"#,
        )
        .await;
        let cfg = stage_agent_config(
            &engine,
            &RatatoskrConfig::default(),
            ToolSet::default(),
            "analyst",
            &["impact_surface", "symbol_lookup", "semantic_search"],
            &mut NodePlugins::default(),
        )
        .unwrap();
        assert!(cfg.tools.is_empty());
    }

    #[tokio::test]
    async fn redteam_classifier_opts_in_on_either_route_source() {
        let engine = engine("redteam-optin").await;
        let mut config = RatatoskrConfig::default();
        assert!(!classifier_enabled(&engine, &config));
        config.models.insert(
            "redteam".to_string(),
            ratatoskr_core::ModelRoute {
                context_window: None,
                provider: "openai".to_string(),
                model: "gpt-5".to_string(),
                max_tokens: None,
                temperature: None,
                params: None,
                session: Default::default(),
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

        // A prompt compiled into a workflow is part of that graph too. Otherwise two runs with
        // different model instructions would claim the same provenance merely because the small
        // TypeScript wrapper was unchanged.
        let workflows = root.join(WORKFLOW_DIR);
        std::fs::create_dir_all(&workflows).unwrap();
        std::fs::write(workflows.join("prompt.md"), "first prompt").unwrap();
        std::fs::write(
            workflows.join("review.ts"),
            "defineWorkflow({ name: 'review', stages: [stage('reviewer', { agent: 'reason', instructions: LOAD('prompt.md') })] });",
        )
        .unwrap();
        let with_loaded_prompt = graph_fingerprint(&root);
        std::fs::write(workflows.join("prompt.md"), "second prompt").unwrap();
        assert_ne!(with_loaded_prompt, graph_fingerprint(&root));

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
            interface: Vec::new(),
        }
    }

    #[test]
    fn the_fork_runs_when_the_plan_changes_code_and_when_a_human_insists() {
        let mut config = RatatoskrConfig::default();
        assert!(fork_is_needed(&analyst_saying(true), &config));
        assert!(
            !fork_is_needed(&analyst_saying(false), &config),
            "a task that changes no code does not pay for a baseline test run and an implementer \
             session"
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
    fn the_characterizer_is_offered_no_tool_it_cannot_answer() {
        // It passes no skills to `run_structured`, so the `SkillHook` is never installed. Granting
        // it the `Skill` tool anyway leaves a tool whose call can only return an error — and a node
        // reading a tool error as an instruction is exactly how the publisher once published
        // nothing.
        let plugins = NodePlugins {
            skills: vec![ratatoskr_plugin::Skill {
                name: "review".into(),
                description: "how to review".into(),
                body: "...".into(),
                dir: std::path::PathBuf::new(),
            }],
            ..Default::default()
        };
        assert!(
            crate::skills::skill_tool(&plugins.skills, "context").is_some(),
            "a node WITH skills bound is offered the tool — the grant itself is not the bug"
        );
        let stripped = NodePlugins {
            skills: Vec::new(),
            ..plugins
        };
        assert!(crate::skills::skill_tool(&stripped.skills, "context").is_none());
    }

    #[test]
    fn a_commit_body_says_why_rather_than_repeating_the_diffstat() {
        // What a run produced before this: a subject, then the numstat as the body. `git log`
        // already shows that on request, and it answers "which files" when the question a reader
        // of a commit has is "why". The implementer's own account is the only description written
        // by the thing that made the change.
        let out = ImplementerOutput {
            worktree_path: "/w".into(),
            branch: "ratatoskr/abc12345".into(),
            diff_summary: " crates/a.rs | 72 ++++++\n 1 file changed, 72 insertions(+)".into(),
            touched_files: vec!["crates/a.rs".into()],
            rewritten_files: Vec::new(),
            failing_tests: Vec::new(),
            passed_tests: 3,
            exit_code: 0,
            narrative: Some(
                "Fenced the acceptance output and bounded it across steps rather than per step, \
                 so one pathological step cannot fill the prompt on its own."
                    .into(),
            ),
            commit_kind: "fix".into(),
            commit_scope: "nodes".into(),
            commit_subject: "fence and bound acceptance output".into(),
        };
        let msg = commit_message(&ratatoskr_core::PublishConfig::default(), "an issue", &out);

        let (subject, body) = msg.split_once("\n\n").expect("a subject and a body");
        assert_eq!(subject, "fix(nodes): fence and bound acceptance output");
        assert!(body.contains("bounded it across steps"), "{body}");
        assert!(
            !body.contains("insertions(+)"),
            "the diffstat is gone: {body}"
        );
        // Wrapped here rather than asked of the model, which counts characters unreliably.
        for line in body.lines() {
            assert!(line.chars().count() <= 72, "{line:?}");
        }

        // An implementer that reported nothing still commits, and gets a subject with no body
        // rather than a body saying nothing.
        let silent = ImplementerOutput {
            narrative: None,
            ..out
        };
        let msg = commit_message(
            &ratatoskr_core::PublishConfig::default(),
            "an issue",
            &silent,
        );
        assert!(!msg.contains("\n\n"), "{msg}");
    }

    #[test]
    fn wrapping_keeps_a_list_a_list_and_never_splits_an_identifier() {
        // Reflowing a bullet list into a paragraph loses the structure that made it readable, and
        // a path broken across lines is a path nobody can copy.
        let text = "- the first item, which runs on well past the seventy-two column mark and then \
                    some more\n- second";
        assert_eq!(wrapped(text), text, "a list is left alone");

        let long = "see crates/ratatoskr-nodes/src/a-very-long-path-that-is-longer-than-the-whole-\
                    permitted-width.rs now";
        let out = wrapped(long);
        assert!(
            out.lines()
                .any(|l| l
                    .contains("a-very-long-path-that-is-longer-than-the-whole-permitted-width.rs")),
            "the path survives whole: {out}"
        );
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
            interface: Vec::new(),
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

    #[tokio::test]
    async fn a_workflow_can_add_to_the_nodes_a_ruleset_may_govern() {
        let built_in = Workflow::BuiltIn;
        // The built-in adds none: it governs exactly the standard set.
        assert!(built_in.nodes().is_empty());
        assert!(!BUILT_IN_NODES.contains(&"referee"));

        let dir = std::env::temp_dir().join(format!("ratatoskr-nodes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("deep.ts"),
            r#"defineWorkflow({ name: "deep", nodes: ["reviewer2", "triager"] });"#,
        )
        .unwrap();
        let found = WorkflowRuntime::discover(&dir).await.unwrap();
        let declared = Workflow::Scripted(found.into_iter().next().unwrap());
        assert_eq!(declared.nodes(), ["reviewer2", "triager"]);
        let _ = std::fs::remove_dir_all(&dir);

        // The standard set is what a repo defining nothing may govern. `memory` is deliberately
        // absent — a direct rag-rat call with no model or tool set to override, so targeting it is
        // a config error rather than a no-op.
        assert!(BUILT_IN_NODES.contains(&"verifier"));
        assert!(!BUILT_IN_NODES.contains(&"memory"));
        // The implementer resolves through `node_agent_config` like every other node now that it
        // drives a model rather than a coding CLI, so a ruleset shapes it like any other.
        assert!(BUILT_IN_NODES.contains(&"implementer"));
    }

    #[test]
    fn the_overseer_is_consulted_only_when_there_is_a_real_choice() {
        // A caller that named a workflow said which shape it wanted and is not asking to be
        // second-guessed.
        assert!(!should_consult_overseer(3, true, true));
        // One or none resolves without a model call: paying for a decision with one answer is
        // waste, and the built-in is what a repo defining nothing gets.
        assert!(!should_consult_overseer(1, false, true));
        assert!(!should_consult_overseer(0, false, true));
        // Unconfigured, the run refuses to guess rather than picking for itself.
        assert!(!should_consult_overseer(3, false, false));
        // The only case worth a call.
        assert!(should_consult_overseer(2, false, true));
    }

    #[tokio::test]
    async fn a_choice_naming_something_absent_is_refused_rather_than_run() {
        // The overseer returns a name; it does not get to select one that is not there. Routing on
        // an invented name would run a shape nobody defined.
        let dir = std::env::temp_dir().join(format!("ratatoskr-ovr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.ts"), r#"defineWorkflow({ name: "research" });"#).unwrap();
        let found = WorkflowRuntime::discover(&dir).await.unwrap();
        let mut registry = vec![Workflow::BuiltIn];
        registry.extend(found.into_iter().map(Workflow::Scripted));

        let store = Store::open_in_memory().unwrap();
        let ledger = Arc::new(RunLedger::default());
        let err = match select_and_record_overseer(OverseerDecision {
            store: &store,
            run_id: "run-invalid-overseer-choice",
            found: registry,
            decided: OverseerOutput {
                workflow: "invented".to_string(),
                reasoning: "the model invented a route".to_string(),
            },
            input_json: r#"{"issue":"choose","choices":[]}"#.to_string(),
            ledger: &ledger,
        })
        .await
        {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a name that is not in the registry must not select anything"),
        };
        assert!(err.contains("no workflow named `invented`"), "{err}");
        assert!(err.contains("research"), "{err}");
        assert!(
            store
                .checkpoints_for_run("run-invalid-overseer-choice")
                .await
                .unwrap()
                .is_empty(),
            "a rejected workflow name must not look like a valid overseer checkpoint"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_built_in_is_chosen_by_nothing_else_matching() {
        // It declares no cases on purpose — it is the fallback, so a model matching cases will
        // never match it, and the prompt says to land there when nothing else fits.
        assert!(Workflow::BuiltIn.when_to_use().is_empty());
        assert!(!Workflow::BuiltIn.purpose().is_empty());
    }

    #[test]
    fn an_output_token_count_below_what_the_output_could_contain_is_not_believable() {
        // The live figures that exposed this: 63 output tokens against a 5,286-byte answer.
        assert!(63 < plausible_output_tokens(5_286));
        // Eight bytes per token is far under any real tokenizer, so a plausible run clears it
        // easily and the warning stays rare enough to mean something.
        assert!(1_700 > plausible_output_tokens(5_286));
        // A node that produced almost nothing has almost no floor to clear.
        assert_eq!(plausible_output_tokens(0), 0);
        assert_eq!(plausible_output_tokens(7), 0);
    }

    fn toml_route(provider: &str, model: &str) -> ratatoskr_core::ModelRoute {
        ratatoskr_core::ModelRoute {
            context_window: None,
            provider: provider.into(),
            model: model.into(),
            max_tokens: None,
            temperature: None,
            params: None,
            session: Default::default(),
        }
    }

    // Contract reading (#206): the "referee_enabled/referee-route resolution next to
    // verifier_enabled" is pinned here as `referee_route(engine, config) -> Option<ModelRoute>`,
    // because the three outcomes the issue requires — the referee's own route, the verifier's
    // route as fallback, or no judgement at all — are exactly an Option of the route to judge on.

    #[tokio::test]
    async fn the_referee_falls_back_to_the_verifier_route() {
        let engine = engine("referee-via-verifier").await;
        let mut config = RatatoskrConfig::default();
        // Only [models.verifier] configured: a repo that already routes a model to judge the diff
        // has said it wants that kind of check.
        let verifier = toml_route("anthropic", "claude-sonnet-4-6");
        config.models.insert("verifier".to_string(), verifier);
        let route = referee_route(&engine, &config).expect("the verifier route is the fallback");
        assert_eq!(route.provider, "anthropic");
        assert_eq!(route.model, "claude-sonnet-4-6");

        // The same fallback through the verifier route's other spelling: ruleset("verifier").model.
        let ruleset = binding_engine(
            "referee-via-verifier-ruleset",
            r#"defineAgent("verifier", { model: { provider: "openai", model: "gpt-5" } });"#,
        )
        .await;
        let config = RatatoskrConfig::default();
        let route = referee_route(&ruleset, &config).expect("ruleset verifier is the fallback");
        assert_eq!(route.provider, "openai");
        assert_eq!(route.model, "gpt-5");
    }

    #[tokio::test]
    async fn the_referees_own_route_wins_over_the_verifiers() {
        let engine = engine("referee-own-route").await;
        let mut config = RatatoskrConfig::default();
        let verifier = toml_route("anthropic", "claude-sonnet-4-6");
        config.models.insert("verifier".to_string(), verifier);
        let own = toml_route("openai", "gpt-5");
        config.models.insert("referee".to_string(), own);
        let route = referee_route(&engine, &config).expect("a referee route is configured");
        assert_eq!(
            route.model, "gpt-5",
            "[models.referee] beats [models.verifier]"
        );

        // A ruleset named referee cannot override this internal route.
        let ruleset = binding_engine(
            "referee-ruleset-route",
            r#"defineAgent("referee", { model: { provider: "moonshot", model: "kimi-k2.5" } });"#,
        )
        .await;
        let route = referee_route(&ruleset, &config).expect("TOML referee route still wins");
        assert_eq!(route.provider, "openai");
        assert_eq!(route.model, "gpt-5");
    }

    #[tokio::test]
    async fn no_referee_and_no_verifier_route_means_no_judgement() {
        let engine = engine("referee-none").await;
        // The default config has neither [models.referee] nor [models.verifier], and the fixture
        // rulesets bind neither either.
        let config = RatatoskrConfig::default();
        assert!(
            referee_route(&engine, &config).is_none(),
            "with no route there is no judgement: converge trusts the test result alone, and says so"
        );
    }
}

#[cfg(test)]
mod repo_conventions_tests {
    use super::*;

    /// A fresh, empty directory unique to this test and process, so concurrent tests never share
    /// a repo root — the same pid-keying the agent-config fixtures use for the same reason.
    fn repo_root(case: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-conventions-{}-{case}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn agents_md_at_the_root_is_read() {
        // Happy: AGENTS.md at the root → Some(its content), non-empty, within the bound.
        let root = repo_root("agents");
        let conventions = "# Conventions\nParameter structs over long argument trains.\n";
        std::fs::write(root.join("AGENTS.md"), conventions).unwrap();

        let got = repo_conventions(&root).expect("AGENTS.md is present");
        assert_eq!(got, conventions);
        assert!(!got.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn claude_md_is_the_honoured_alternative_name() {
        // Happy: no AGENTS.md, but a CLAUDE.md regular file → its content. The ecosystem's
        // alternative name is honoured.
        let root = repo_root("claude");
        let conventions = "# Conventions\nInjected time and ids.\n";
        std::fs::write(root.join("CLAUDE.md"), conventions).unwrap();

        assert_eq!(repo_conventions(&root).as_deref(), Some(conventions));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_claude_symlink_to_agents_is_read_once_not_duplicated() {
        // Happy: this repo's own layout — CLAUDE.md is a symlink to AGENTS.md. AGENTS.md is
        // preferred and its content comes back once, not concatenated with itself.
        // (Unix-only symlink, matching where a run actually runs.)
        let root = repo_root("symlink");
        let conventions = "# Conventions\nClosed enums behind stable string tokens.\n";
        std::fs::write(root.join("AGENTS.md"), conventions).unwrap();
        std::os::unix::fs::symlink("AGENTS.md", root.join("CLAUDE.md")).unwrap();

        assert_eq!(repo_conventions(&root).as_deref(), Some(conventions));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn neither_file_present_yields_none() {
        // Sad: a repo with neither file → None, so the caller leaves the preamble exactly as it is
        // today and a repo with no conventions file runs unchanged.
        let root = repo_root("none");
        assert_eq!(repo_conventions(&root), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_file_larger_than_the_bound_is_truncated_not_dropped() {
        // Sad: an AGENTS.md far larger than the bound → bounded text, never a silent drop and
        // never injected whole. The exact preamble budget is not part of the contract, so this
        // asserts the observable guarantee: the result is non-empty and strictly shorter than an
        // input placed well past any plausible bound. (The contract also requires a log record
        // naming the file and its full vs. injected size; that side effect needs a tracing
        // subscriber this crate has no test harness for, so it is not asserted here.)
        let root = repo_root("huge");
        let huge = "x".repeat(1_000_000);
        std::fs::write(root.join("AGENTS.md"), &huge).unwrap();

        let got = repo_conventions(&root).expect("a huge file is still injected, bounded");
        assert!(!got.is_empty());
        assert!(
            got.len() < huge.len(),
            "a file past the bound is truncated, not injected whole: {} vs {}",
            got.len(),
            huge.len()
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_root_that_is_not_a_directory_yields_none_rather_than_erroring() {
        // Sad: an unreadable / non-directory root → None rather than an error that fails the run.
        let missing = std::env::temp_dir().join(format!(
            "ratatoskr-conventions-absent-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&missing);
        assert_eq!(repo_conventions(&missing), None);

        // A path that exists but is a regular file, not a repo directory.
        let root = repo_root("not-a-dir");
        let file = root.join("AGENTS.md");
        std::fs::write(&file, "conventions").unwrap();
        assert_eq!(repo_conventions(&file), None);
        let _ = std::fs::remove_dir_all(&root);
    }
}

#[cfg(test)]
mod converge_stage_tests {
    use super::*;

    // Contract reading (#198): `fork_and_converge` is split into the named stages its comments
    // already describe, two of which are the decisions the loop used to keep inline. The issue
    // names them "per the existing comments, e.g." — pinned here at the crate root, next to
    // `fork_and_converge`, as:
    //
    //   pub(crate) async fn decide_correction(
    //       violations: &[referee::Violation],
    //       red_team_out: &RedTeamOutput,
    //       impl_out: &ImplementerOutput,
    //       authored: &[String],
    //       review: Option<&Review>,
    //   ) -> Result<Reviewed, PlanError>
    //
    // — the contract's input list in its order (referee violations, the two outputs, the
    // authored test list, the review handle). Async and fallible because the tests-clean +
    // review-configured branch must await the review; with `review: None` it is pure data, which
    // is the half these tests exercise.
    //
    //   pub(crate) enum CeilingDecision { Stop(RunStatus), Replan(Correction) }
    //
    //   pub(crate) async fn at_ceiling(
    //       iterations: u32,
    //       max_iterations: u32,
    //       replanned: bool,
    //       findings: &[verifier::Finding],
    //       review: Option<&Review>,
    //   ) -> Result<CeilingDecision, PlanError>
    //
    // — the contract's "enum of the form { Stop(RunStatus) | Replan(Correction) }", with the
    // once-per-run escalation decidable without a loop. Same async/Result reading: the Replan
    // branch awaits the analyst, the Stop branches never do.
    //
    // What these tests deliberately do not pin: the loop's translation of a stage output into a
    // terminal status (Clean → Converged, Unavailable → Unreviewed), the replanned/in-force
    // bookkeeping, and the worktree cleanup on an implementer error. Those live in the loop body
    // `fork_and_converge` still owns, are not separately callable under the contract, and are
    // covered by the requirement that the existing converge tests pass unedited.

    fn red(failing: &[&str], passed: usize, exit: i32) -> RedTeamOutput {
        RedTeamOutput {
            authored: None,
            failing_tests: failing.iter().map(|s| s.to_string()).collect(),
            passed_tests: passed,
            exit_code: exit,
            classifications: vec![],
        }
    }

    fn imp(failing: &[&str], passed: usize, exit: i32) -> ImplementerOutput {
        ImplementerOutput {
            worktree_path: "/wt".to_string(),
            branch: "ratatoskr/test".into(),
            diff_summary: String::new(),
            touched_files: vec![],
            rewritten_files: Vec::new(),
            failing_tests: failing.iter().map(|s| s.to_string()).collect(),
            passed_tests: passed,
            exit_code: exit,
            narrative: None,
            commit_kind: String::new(),
            commit_scope: String::new(),
            commit_subject: String::new(),
        }
    }

    fn violation(file: &str, reason: &str) -> referee::Violation {
        referee::Violation {
            file: file.into(),
            reason: reason.into(),
        }
    }

    fn finding() -> verifier::Finding {
        verifier::Finding {
            severity: verifier::Severity::P2,
            kind: verifier::FindingKind::Execution,
            file: "a.rs".into(),
            line: None,
            summary: "s".into(),
            failure_scenario: "f".into(),
        }
    }

    /// Unwrap a `Reviewed::Fix` into its correction, failing loudly on any other verdict.
    fn correction_of(reviewed: Reviewed) -> Correction {
        match reviewed {
            Reviewed::Fix(correction) => *correction,
            Reviewed::Clean => panic!("expected a correction, got Reviewed::Clean"),
            Reviewed::Unavailable => panic!("expected a correction, got Reviewed::Unavailable"),
        }
    }

    #[tokio::test]
    async fn the_referee_correction_wins_and_the_tests_are_never_consulted() {
        // The two gates in order, referee first: violations are non-empty AND the test outcome
        // is as bad as it gets — the post-change run never completed (exit 101, nothing parsed)
        // and an authored test still fails. The prompt must come back byte-equal to
        // `referee::correction(&violations)`: no test-derived prompt can be that string, so
        // equality is what proves the test result was never consulted.
        let violations = vec![violation(
            "crates/foo/src/lib.rs",
            "deleted the module's #[cfg(test)] characterisation",
        )];
        let baseline = red(&["crate::authored_test"], 3, 1);
        let post = imp(&["crate::authored_test"], 0, 101);
        let authored = vec!["crate::authored_test".to_string()];

        let reviewed = decide_correction(&violations, &baseline, &post, &authored, None)
            .await
            .expect("the referee branch spends no model call");
        let correction = correction_of(reviewed);
        assert_eq!(correction.prompt, referee::correction(&violations));
        // A deterministic correction carries no review state: nothing found, no revised plan.
        assert!(correction.found.is_empty());
        assert!(correction.revised.is_none());
    }

    #[tokio::test]
    async fn a_test_run_that_did_not_complete_says_so_and_names_the_exit_code() {
        // Zero tests parsed and a non-zero exit: the change likely does not compile. Reporting
        // "no new failures" here would be the false-convergence reading this branch exists to
        // refuse — the prompt states the command did not run to completion, with the exit code.
        let baseline = red(&["a::pre_existing"], 10, 1);
        let post = imp(&[], 0, 101);

        let reviewed = decide_correction(&[], &baseline, &post, &[], None)
            .await
            .expect("pure data with no review configured");
        let prompt = correction_of(reviewed).prompt;
        assert!(
            prompt.contains("did not run to completion"),
            "the run's failure to complete is said, not hidden: {prompt}"
        );
        assert!(
            prompt.contains("101"),
            "the exit code is named so the implementer can see how it died: {prompt}"
        );
    }

    #[tokio::test]
    async fn authored_tests_still_failing_are_named_in_the_correction() {
        // Written for this change before any code existed, they fail in the baseline as a matter
        // of course — so *nothing is newly failing* here, and `is_converged` alone would wave
        // the change through. The unsatisfied gate is what refuses, and the prompt names exactly
        // the authored tests that still fail.
        let baseline = red(
            &["tests::writes_a_row", "tests::rejects_an_empty_name"],
            4,
            1,
        );
        // The run completed (tests parsed, so exit 1 is a real test result), one authored test
        // now passes, the other still fails.
        let post = imp(&["tests::rejects_an_empty_name"], 8, 1);
        let authored = vec![
            "tests::writes_a_row".to_string(),
            "tests::rejects_an_empty_name".to_string(),
        ];

        let reviewed = decide_correction(&[], &baseline, &post, &authored, None)
            .await
            .expect("pure data with no review configured");
        let prompt = correction_of(reviewed).prompt;
        assert!(
            prompt.contains("tests::rejects_an_empty_name"),
            "the unsatisfied test is named: {prompt}"
        );
        assert!(
            !prompt.contains("tests::writes_a_row"),
            "the authored test that now passes is not named: {prompt}"
        );
    }

    #[tokio::test]
    async fn newly_introduced_failures_are_named_and_pre_existing_ones_are_not() {
        // The regression branch: the run completed, no authored tests are outstanding, but the
        // change broke tests the baseline had green. The prompt names those and only those —
        // naming a pre-existing failure would send the implementer chasing a failure it did not
        // cause.
        let baseline = red(&["a::pre_existing"], 10, 1);
        let post = imp(&["a::pre_existing", "b::broke", "c::also_broke"], 9, 1);

        let reviewed = decide_correction(&[], &baseline, &post, &[], None)
            .await
            .expect("pure data with no review configured");
        let prompt = correction_of(reviewed).prompt;
        assert!(prompt.contains("b::broke"), "{prompt}");
        assert!(prompt.contains("c::also_broke"), "{prompt}");
        assert!(
            !prompt.contains("a::pre_existing"),
            "the pre-existing failure is not the implementer's problem: {prompt}"
        );
    }

    #[tokio::test]
    async fn clean_tests_and_no_review_is_clean() {
        // Every deterministic gate passes — the run completed, no authored test is outstanding,
        // nothing newly failing — and there is no verifier configured: the stage's verdict is
        // Reviewed::Clean, which the loop then translates to RunStatus::Converged. (The
        // translation is the loop's, not the stage's; the stage's half is returning Clean.)
        let baseline = red(&["a::pre_existing"], 10, 1);
        let post = imp(&["a::pre_existing"], 12, 0);
        let reviewed = decide_correction(&[], &baseline, &post, &[], None)
            .await
            .expect("pure data with no review configured");
        assert!(
            matches!(reviewed, Reviewed::Clean),
            "clean tests with nobody to ask converge as Clean"
        );

        // The all-green spelling of the same thing: empty baseline, everything passing.
        let reviewed = decide_correction(&[], &red(&[], 285, 0), &imp(&[], 300, 0), &[], None)
            .await
            .expect("pure data with no review configured");
        assert!(matches!(reviewed, Reviewed::Clean));
    }

    #[tokio::test]
    async fn a_spent_budget_with_nothing_found_stops_without_asking_the_analyst() {
        // Budget spent, no prior replan, but `found_so_far` is empty: there is no evidence to
        // hand the analyst, so the run stops at the wall rather than spending a replan on
        // nothing. No review handle is passed — with no findings the analyst must not be
        // reached even when there is one.
        let decision = at_ceiling(3, 3, false, &[], None)
            .await
            .expect("stopping spends no model call");
        assert!(
            matches!(
                decision,
                CeilingDecision::Stop(RunStatus::MaxIterationsReached)
            ),
            "an empty evidence base records the wall, not a replan"
        );
    }

    #[tokio::test]
    async fn a_spent_budget_after_a_replan_stops_for_real() {
        // The escalation is once per run: `replanned` stops the run even with findings standing,
        // because a second replan would be the same escalation on the same evidence.
        let findings = vec![finding()];
        let decision = at_ceiling(4, 3, true, &findings, None)
            .await
            .expect("stopping spends no model call");
        assert!(
            matches!(
                decision,
                CeilingDecision::Stop(RunStatus::MaxIterationsReached)
            ),
            "the second time at the ceiling is the wall, not another escalation"
        );
    }

    #[tokio::test]
    async fn a_spent_budget_without_a_review_stops_at_the_wall() {
        // Findings stand and the budget is spent, but with no review configured there is no
        // analyst re-entry either: the escalation goes through the review handle. Its absence is
        // MaxIterationsReached — never an error, and never a silently extended budget.
        let findings = vec![finding()];
        let decision = at_ceiling(3, 3, false, &findings, None)
            .await
            .expect("no review configured stops cleanly");
        assert!(
            matches!(
                decision,
                CeilingDecision::Stop(RunStatus::MaxIterationsReached)
            ),
            "with nobody to escalate to, the ceiling is the wall"
        );
    }
}

#[cfg(test)]
mod referee_governance_tests {
    use super::*;

    // Contract reading (#209): "referee" leaves `BUILT_IN_NODES` and with it
    // `governable_nodes()`, so a `.ratatoskr/rules/referee.ts` is rejected at startup exactly
    // the way a typo'd node name is today — the CLI's load_rules predicate, composed below.
    // `referee_route` keeps its signature but stops consulting a "referee" ruleset:
    // [models.referee] becomes a TOML-only override with the verifier's route as the fallback.
    // The fixed-capability construction and the single internal judgement path are pinned in
    // referee.rs's tests.

    /// A ruleset directory containing exactly `source`, loaded minus the CLI's governable-name
    /// gate — the tests compose that gate's predicate themselves, per test and per process
    /// unique so concurrent tests never share a half-written file.
    async fn rules_engine(case: &str, source: &str) -> Arc<ScriptEngine> {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-referee-governance-{}-{case}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("agents.ts"), source).unwrap();
        ScriptEngine::load(&dir).await.unwrap()
    }

    fn route(provider: &str, model: &str) -> ratatoskr_core::ModelRoute {
        ratatoskr_core::ModelRoute {
            context_window: None,
            provider: provider.into(),
            model: model.into(),
            max_tokens: None,
            temperature: None,
            params: None,
            session: Default::default(),
        }
    }

    #[tokio::test]
    async fn referee_is_no_longer_a_governable_node() {
        // The judgement keeps its checkpoint name but loses its governance socket: its
        // capability boundary is part of the gate's correctness contract, not something a
        // ruleset or plugin may shape.
        assert!(
            !BUILT_IN_NODES.contains(&"referee"),
            "the internal diff-judgement is not a governable node"
        );
        // Everything else stays exactly as governable as it was.
        for name in [
            "overseer",
            "publisher",
            "context",
            "scout",
            "analyst",
            "implementer",
            "bookkeeper",
            "redteam",
            "verifier",
            "characterizer",
        ] {
            assert!(
                BUILT_IN_NODES.contains(&name),
                "{name} must stay governable"
            );
        }

        // governable_nodes() reports the built-ins plus whatever this checkout's workflows
        // declare — none in the crate directory these tests run in — so the same two claims
        // hold of the set load_rules will consult.
        let governable = governable_nodes()
            .await
            .expect("reading the workflow registry");
        assert!(
            !governable.iter().any(|n| n == "referee"),
            "governable_nodes() must never report the internal judge: {governable:?}"
        );
        for name in [
            "overseer",
            "publisher",
            "context",
            "scout",
            "analyst",
            "implementer",
            "bookkeeper",
            "redteam",
            "verifier",
            "characterizer",
        ] {
            assert!(
                governable.iter().any(|n| n == name),
                "{name} must stay governable: {governable:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_referee_ruleset_fails_loading_the_way_a_typo_does() {
        // The CLI's startup gate (load_rules) rejects any defineAgent name outside
        // governable_nodes(), with an error naming the offender and listing the governable
        // set. What #209 changes is the set — compose the gate's exact predicate here and
        // "referee" lands on the rejected side of it.
        let engine = rules_engine(
            "referee-ruleset",
            r#"defineAgent("referee", { model: { provider: "openai", model: "gpt-5" } });"#,
        )
        .await;
        let governable = governable_nodes()
            .await
            .expect("reading the workflow registry");
        let rejected: Vec<String> = engine
            .declared_agents()
            .filter(|name| !governable.iter().any(|n| n.as_str() == *name))
            .map(str::to_string)
            .collect();
        assert_eq!(
            rejected,
            ["referee"],
            "defineAgent(\"referee\") is rejected at startup like any unknown node name"
        );

        // A ruleset for a node that remains built-in still passes the same predicate, so the
        // rejection above is about "referee" specifically and not a broken gate.
        let engine = rules_engine(
            "verifier-ruleset",
            r#"defineAgent("verifier", { maxTurns: 2 });"#,
        )
        .await;
        let rejected: Vec<String> = engine
            .declared_agents()
            .filter(|name| !governable.iter().any(|n| n.as_str() == *name))
            .map(str::to_string)
            .collect();
        assert!(
            rejected.is_empty(),
            "verifier stays governable: {rejected:?}"
        );
    }

    #[tokio::test]
    async fn workflow_declaration_cannot_re_enter_internal_referee() {
        // A workflow may declare arbitrary governable nodes, except the internal referee: that
        // name is a fixed capability boundary rather than a scriptable agent.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-referee-workflow-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("deep.ts"),
            r#"defineWorkflow({ name: "deep", nodes: ["referee"] });"#,
        )
        .unwrap();
        let found = WorkflowRuntime::discover(&dir).await.unwrap();
        assert_eq!(found[0].meta().nodes, ["referee"]);
        assert!(
            !governable_from(found).iter().any(|name| name == "referee"),
            "a workflow declaration cannot make the internal judge governable"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_referee_ruleset_contributes_no_route() {
        // The route override is TOML-only now: a defineAgent("referee", { model }) with no
        // [models.referee], no [models.verifier] and no verifier ruleset anywhere is no route
        // at all — the judgement is skipped, exactly as if the ruleset were absent.
        let engine = rules_engine(
            "referee-model-ignored",
            r#"defineAgent("referee", { model: { provider: "moonshot", model: "kimi-k2.5" } });"#,
        )
        .await;
        let config = RatatoskrConfig::default();
        assert!(
            referee_route(&engine, &config).is_none(),
            "a referee ruleset is never consulted for a route"
        );
    }

    #[tokio::test]
    async fn models_referee_is_toml_only_and_still_wins() {
        // [models.referee] set alongside every other spelling that could claim the route: it
        // wins, and the referee ruleset's model is not it.
        let engine = rules_engine(
            "referee-toml-wins",
            r#"
            defineAgent("referee", { model: { provider: "moonshot", model: "kimi-k2.5" } });
            defineAgent("verifier", { model: { provider: "openai", model: "gpt-5" } });
            "#,
        )
        .await;
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "verifier".to_string(),
            route("anthropic", "claude-sonnet-4-6"),
        );
        config.models.insert(
            "referee".to_string(),
            route("anthropic", "claude-haiku-4-5-20251001"),
        );

        let resolved = referee_route(&engine, &config).expect("[models.referee] is configured");
        assert_eq!(resolved.provider, "anthropic");
        assert_eq!(resolved.model, "claude-haiku-4-5-20251001");
        assert_ne!(
            resolved.model, "kimi-k2.5",
            "a referee ruleset is never consulted for a route"
        );
        assert_ne!(
            resolved.model, "gpt-5",
            "the verifier's route is only the fallback"
        );
    }

    #[tokio::test]
    async fn the_verifier_route_is_still_the_fallback() {
        // Only the verifier routed — here by TOML, with a referee ruleset present that must be
        // ignored — and the judgement still happens, on the verifier's model.
        let engine = rules_engine(
            "referee-fallback",
            r#"defineAgent("referee", { model: { provider: "moonshot", model: "kimi-k2.5" } });"#,
        )
        .await;
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "verifier".to_string(),
            route("anthropic", "claude-sonnet-4-6"),
        );
        let resolved = referee_route(&engine, &config).expect("the verifier route is the fallback");
        assert_eq!(resolved.provider, "anthropic");
        assert_eq!(resolved.model, "claude-sonnet-4-6");
    }
}
