//! Concrete Phase 2 nodes (scout, memory, analyst) and the straight-line `plan` executor.
//!
//! Per the Phase 2 decision, this is a plain sequential `async fn`, not a generic edge-walking
//! interpreter: with exactly three fixed nodes and no branching, `run_plan` delivers the same
//! policy guarantee (schema-validated handoffs, a checkpoint after every node, nothing skipped)
//! with nothing to get wrong. The real executor arrives in Phase 3 when fork/join needs one.

pub mod analyst;
pub mod bookkeeper;
pub mod clarify;
pub mod context;
pub mod control;
pub mod converge;
pub mod implementer;
pub mod issue;
pub mod memory;
pub mod overseer;
pub mod publisher;
pub mod redteam;
pub mod scout;
pub mod skills;
pub mod testrun;
pub mod verifier;
pub mod workflow;

pub use analyst::{AnalystNode, AnalystOutput};
pub use bookkeeper::{BookkeeperInput, BookkeeperNode, BookkeeperOutput, MemoryWritten};
pub use context::{Constraint, ContextNode, ContextOutput};
pub use implementer::{ImplementerNode, ImplementerOutput};
pub use memory::{MemoryNode, MemoryOutput, MemoryRecord};
pub use overseer::{OverseerNode, OverseerOutput};
pub use publisher::{PublisherNode, PublisherOutput};
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
    client: Option<&RagRatClient>,
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
    let plugins_context = context.for_node("context");
    let ctx_cfg = node_agent_config(
        engine,
        config,
        context.pool_for("context", client.map(|c| c.offer())),
        "context",
        context::CONTEXT_TOOLS,
        &plugins_context,
    )?;
    let mut ctx_tools = ctx_cfg.tools;
    ctx_tools.add_local(clarify::ask_tool());
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
    let plugins_analyst = context.for_node("analyst");
    let analyst_cfg = node_agent_config(
        engine,
        config,
        context.pool_for("analyst", client.map(|c| c.offer())),
        "analyst",
        analyst::ANALYST_TOOLS,
        &plugins_analyst,
    )?;
    let analyst = AnalystNode {
        conversation: Some(format!("{run_id}-analyst")),
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
        brief: context_out.brief,
        constraints: context_out.constraints,
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

/// Every node any workflow in this repo may govern: the built-in set plus what each declares.
///
/// The union across all of them, not just the one a run selects, because rulesets are loaded
/// before a workflow is chosen — and a ruleset targeting a node that some workflow declares is
/// legitimate whether or not this particular run uses that workflow.
pub async fn governable_nodes() -> Result<Vec<String>, PlanError> {
    let mut names: Vec<String> = BUILT_IN_NODES.iter().map(|s| s.to_string()).collect();
    for workflow in defined().await? {
        names.extend(workflow.meta().nodes.iter().cloned());
    }
    names.sort();
    names.dedup();
    Ok(names)
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
    config.models.contains_key("overseer")
        || engine
            .ruleset("overseer")
            .is_some_and(|r| r.config().model.is_some())
}

/// Pick the workflow for this run, asking the overseer when there is a real choice to make.
///
/// The order is deliberate. A named workflow wins outright — a caller that said which shape it
/// wanted is not asking to be second-guessed. Nothing to choose between resolves without a model
/// call, because paying for a decision with one answer is waste. Only a genuine choice reaches the
/// overseer, and without one configured the run still refuses to guess rather than picking.
pub async fn choose(request: &RunRequest<'_>) -> Result<Workflow, PlanError> {
    let found = registry().await?;
    let real_choice = request.workflow.is_none()
        && found
            .iter()
            .filter(|w| !matches!(w, Workflow::BuiltIn))
            .count()
            > 1;
    if !real_choice || !overseer_enabled(request.engine, request.config) {
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

    let cwd = std::env::current_dir().unwrap_or_default();
    let context = PluginContext::resolve(request.config, request.engine, &cwd).await?;
    let plugins = context.for_node("overseer");
    let cfg = node_agent_config(
        request.engine,
        request.config,
        context.pool_for("overseer", request.client.map(|c| c.offer())),
        "overseer",
        overseer::OVERSEER_TOOLS,
        &plugins,
    )?;
    // Its own ledger: the overseer runs before the run's, and its cost is still a cost. Drained
    // straight onto the checkpoint below rather than carried, because nothing after this point
    // would claim it.
    let ledger = Arc::new(RunLedger::default());
    let decided = OverseerNode {
        route: cfg.route,
        tools: cfg.tools,
        policy: cfg.policy,
        max_turns: cfg.max_turns,
        system_prompt: cfg.system_prompt,
        plugins,
        ledger: Some(Arc::clone(&ledger)),
        files: cfg.files,
    }
    .run(request.issue, &choices)
    .await
    .map_err(|e| PlanError::node("overseer", e))?;

    // Recorded before it is acted on, and recorded even when the name turns out to be wrong: the
    // reasoning is what a reader needs when a run went somewhere unexpected, and a rejected choice
    // is exactly such a case.
    request
        .store
        .upsert_run(request.run_id, None, RunStatus::Running.as_str())
        .await?;
    record(Record {
        store: request.store,
        run_id: request.run_id,
        node: "overseer",
        output: &decided,
        input: Some(serde_json::to_string(request.issue)?),
        iteration: None,
        ledger: Some(&ledger),
    })
    .await?;

    // A model naming something that is not there does not get to select it. Falling through to the
    // named lookup gives the error that lists what was available.
    select(found, Some(&decided.workflow))
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
    /// *requires* actually exists.
    ///
    /// A plugin that is missing, broken, or slow contributes nothing and is logged. A plugin a
    /// ruleset *names* splits by how it was named. An explicit `defineAgent` binding (an `Only`
    /// list, an `add`, or a `remove`) is a requirement: naming one nobody installed is a typo, and
    /// the run fails rather than binding less than its author asked for. A `defineDefaults` name is
    /// a preference — it applies to every node, so a missing one warns and narrows the tool pool to
    /// what was discovered rather than refusing the run (a rag-rat-less checkout falls back to file
    /// tools). A name in both categories is a requirement: the explicit binding wins.
    pub async fn resolve(
        config: &RatatoskrConfig,
        engine: &Arc<ScriptEngine>,
        cwd: &std::path::Path,
    ) -> Result<Self, PlanError> {
        let plugins = ratatoskr_plugin::discover(&config.plugins.search_paths(cwd));
        for plugin in &plugins {
            tracing::info!(plugin = plugin.name, "loaded plugin");
        }

        let installed = |name: &String| plugins.iter().any(|p| &p.name == name);
        let known = || {
            let names: Vec<&str> = plugins.iter().map(|p| p.name.as_str()).collect();
            if names.is_empty() {
                "none".to_string()
            } else {
                names.join(", ")
            }
        };

        // An explicit `defineAgent` binding is a requirement: a missing one fails the run.
        let required = engine.required_plugins();
        let missing_required: Vec<String> =
            required.iter().filter(|n| !installed(n)).cloned().collect();
        if !missing_required.is_empty() {
            return Err(PlanError::node(
                "plugins",
                NodeError::Failed(format!(
                    "ruleset names plugin(s) that were not found: {}; discovered: {}",
                    missing_required.join(", "),
                    known()
                )),
            ));
        }

        // A `defineDefaults` name is a preference: a missing one warns and narrows the pool. Names
        // promoted to required by an agent rule are handled above, so exclude them here.
        let missing_preferred: Vec<String> = engine
            .declared_plugins()
            .into_iter()
            .filter(|n| !required.contains(n) && !installed(n))
            .collect();
        if !missing_preferred.is_empty() {
            tracing::warn!(
                missing = missing_preferred.join(", "),
                discovered = known(),
                "ruleset default names plugin(s) that were not found; running without them"
            );
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
    ///
    /// Restricted to what was discovered: a `defineDefaults` name nobody installed is a preference
    /// (#185), so it drops out of the bound set rather than binding a plugin that does not exist.
    /// An undiscovered *explicit* binding never reaches here — `resolve` fails first.
    fn bound(&self, node: &str) -> Vec<String> {
        match &self.engine {
            Some(engine) => engine
                .plugins_for(node, &self.discovered)
                .into_iter()
                .filter(|name| self.discovered.contains(name))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Every tool `node` may call: rag-rat's catalogue, then the servers its plugins declare.
    ///
    /// rag-rat comes first so it wins any name collision — see [`ToolSet::from_servers`].
    /// The tools one node may call: rag-rat's, when there is a rag-rat, plus the plugin servers
    /// bound to that node.
    ///
    /// `None` omits the group rather than passing an empty one, so a pool without rag-rat is the
    /// same shape as a pool that never had it — nothing downstream has to special-case a server
    /// that offers nothing.
    fn pool_for(&self, node: &str, rag_rat: Option<ServerTools>) -> ToolSet {
        let bound = self.bound(node);
        let mut servers: Vec<ServerTools> = rag_rat.into_iter().collect();
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
/// prefixed by whatever context plugins contributed for this run and suffixed by the listing of the
/// skills its plugins bind.
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
    let base = system_prompt.unwrap_or(built_in);
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
    // Two different things, and conflating them cost the publisher its `gh`.
    //
    // Whether a node is *offered* the file tools follows from whether it declares any reach: an
    // empty list has to mean none, or "no tools" quietly means "Read, Grep and Glob" and a node
    // meant to transcribe output it was handed goes reading directories on the host.
    //
    // The root is separate, and always set. It is not a capability — it is where a tool resolves
    // paths, and a node with no file tools can do nothing with one. The publisher declares no
    // default tools on purpose, so `gh` cannot be handed to anyone by widening a shared constant,
    // and it is the root that lets `gh` resolve at all. Gating the root on the list left it
    // holding a stand-in that errors, which it dutifully reported as a reason not to publish.
    let files = std::env::current_dir().ok();
    if !default_tools.is_empty() {
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
            temperature: None,
            params: None,
            session: Default::default(),
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
        // A built-in default list names rag-rat's tools, so in a repository configured without
        // rag-rat every node would warn about every one of them — turning a supported setup into a
        // wall of noise that hides the warning this exists for. An explicit ruleset `allow` is
        // different: it named something by hand, so a name nothing offers is a typo either way.
        if spelled_out.is_none() && !tools.has_server(ratatoskr_mcp::RAG_RAT) {
            tracing::debug!(
                node,
                ?missing,
                "no rag-rat in this repository; these tools are absent by configuration"
            );
        } else {
            tracing::warn!(node, ?missing, "no connected MCP server offers these tools");
        }
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
    if let Some(tool) = skills::skill_tool(&plugins.skills, node) {
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
    let plan = match plan_half(client, config, store, run_id, issue, engine, &plan_context).await {
        Ok(plan) => plan,
        Err(e) => {
            plan_context.session_end(RunStatus::Failed.as_str()).await;
            return Err(e);
        }
    };

    // Built before the fork decision, because the no-code-change path publishes too and a
    // publisher needs the same run handle every other node gets.
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
        engine,
        context,
        ledger,
        ..
    } = run;
    let plugins = context.for_node("publisher");
    let cfg = node_agent_config(
        engine,
        config,
        context.pool_for("publisher", client.map(|c| c.offer())),
        "publisher",
        &[],
        &plugins,
    )?;
    let mut tools = cfg.tools;
    // The tools that write outside this machine. Added here rather than in the default list so no
    // other node can be handed one by widening a shared constant.
    tools
        .local()
        .tools
        .push(ratatoskr_agent::publish::declaration());

    // Push is offered only when there is a branch to push, and only ever THAT branch: the access
    // carries it, and what the tool takes is a name's parts, never a ref. A run with no fork has
    // nothing to publish and is not given the tool at all.
    let push = input
        .implementer
        .as_ref()
        .map(|im| im.branch.clone())
        .filter(|b| ratatoskr_agent::publish::pushable(b))
        .map(|branch| ratatoskr_agent::publish::PushAccess {
            repo_root: cfg.files.clone().unwrap_or_else(|| ".".into()),
            branch,
            // From the run, not from the publisher: the number is what the branch is *for*, and
            // it is not the naming step's to choose.
            issue: Some(input.issue.clone()),
        });
    if push.is_some() {
        tools
            .local()
            .tools
            .push(ratatoskr_agent::publish::push_declaration());
    }

    let node = PublisherNode {
        push,
        route: cfg.route,
        tools,
        policy: cfg.policy,
        max_turns: cfg.max_turns,
        system_prompt: cfg.system_prompt,
        plugins,
        ledger: Some(Arc::clone(ledger)),
        files: cfg.files,
    };
    let out = node
        .run(input)
        .await
        .map_err(|e| PlanError::node("publisher", e))?;
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
        context.pool_for("bookkeeper", client.map(|c| c.offer())),
        "bookkeeper",
        bookkeeper::BOOKKEEPER_TOOLS,
        &plugins_bookkeeper,
    )?;
    let mut tools = cfg.tools;
    tools.add_local(clarify::ask_tool());
    let node = BookkeeperNode {
        route: cfg.route,
        tools,
        sink: client.map(|c| c.sink()),
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
    brief: String,
    constraints: Vec<Constraint>,
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
    Fix(Box<Correction>),
    /// The verifier could not be asked. The reason is on its checkpoint.
    Unavailable,
}

/// What a review concluded the run should do next.
struct Correction {
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
) -> Result<Option<testrun::Characterizer>, PlanError> {
    if !config.models.contains_key("characterizer")
        && !engine
            .ruleset("characterizer")
            .is_some_and(|r| r.config().model.is_some())
    {
        return Ok(None);
    }
    // No skills either, and this is the seam that enforces it: `node_agent_config` grants the
    // `Skill` tool whenever a node has skills bound, while the hook that answers it is installed
    // from what the caller passes to `run_structured`. `Characterizer` passes none — so leaving
    // skills bound here would offer it a tool whose result nothing can produce, and a node that
    // reads a tool error as an instruction is a failure this repo has already paid for once.
    let plugins = NodePlugins {
        skills: Vec::new(),
        ..context.for_node("characterizer")
    };
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
        ledger,
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
    let plugins = context.for_node("implementer");
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
    let cfg = node_agent_config(
        engine,
        config,
        tools,
        "implementer",
        &implementer_default_tools(),
        &plugins,
    )?;
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
    /// Point the verifier's file tools at the tree the change is actually in.
    ///
    /// The verifier is handed the diff as text and then reads the repository to check it against.
    /// Without this it reads the process's working directory — the main checkout — which does not
    /// contain the change, so every `Read` and `Grep` it makes describes the code as it was before.
    /// On a live run that produced three consecutive blocking findings saying the diff's changes
    /// "are not actually present in the repository": true of the tree it could see, and the wrong
    /// conclusion, which sent the implementer back to fix work that was never broken.
    ///
    /// Set here rather than at construction because [`Review::build`] runs before the worktree
    /// exists — deliberately, so a misconfigured verifier fails the run before an implementer
    /// session has been spent on it. This is the earliest moment the path is known.
    ///
    /// The analyst kept alive for revisions is left rooted at the checkout: it owns the plan and
    /// reasons about the repository, not about the diff.
    fn rooted_at(&mut self, worktree: &std::path::Path) {
        self.verifier.files = Some(worktree.to_path_buf());
    }

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
            context.pool_for("verifier", client.map(|c| c.offer())),
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
            // Left unset here, and set by `rooted_at` once the worktree exists. `cfg.files` is the
            // process's working directory — the main checkout — which does not contain the change
            // this node reviews. Leaving it None means a review that was never rooted loses its file
            // tools and says so, rather than quietly reading the wrong tree and reporting the change
            // as missing.
            files: None,
            ledger: Some(Arc::clone(ledger)),
        };

        let plugins_analyst = context.for_node("analyst");
        let acfg = node_agent_config(
            engine,
            config,
            context.pool_for("analyst", client.map(|c| c.offer())),
            "analyst",
            analyst::ANALYST_TOOLS,
            &plugins_analyst,
        )?;
        Ok(Some(Review {
            verifier,
            threshold: parse_threshold(&config.implementer.verify_threshold),
            analyst: AnalystNode {
                conversation: Some(format!("{}-analyst", run.run_id)),
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
        let revised = match self
            .analyst
            .run(revision, &RunState::new(run_id, None))
            .await
        {
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
    // The repository's own conventions (`AGENTS.md`), loaded once from the checkout — not from an
    // implementer worktree, which is a copy — so every converge iteration and the test author see
    // the same text. Only the two nodes that write code carry it.
    let conventions = repo_conventions(&repo_path);
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
        worktree_root: config.worktree.root.clone(),
        baseline_branch: format!("ratatoskr/{short}-baseline"),
        sandbox: config.sandbox.clone(),
        name: format!("ratatoskr-redteam-{short}"),
        // Opt-in: classify baseline failures only when redteam has a route — from
        // `[models.redteam]` or from its `.ratatoskr/rules/redteam.ts` ruleset.
        acceptance: acceptance.clone(),
        characterizer: build_characterizer(
            engine,
            config,
            context,
            client.map(|c| c.offer()),
            Some(Arc::clone(ledger)),
        )?,
        classifier: match classifier_enabled(engine, config) {
            true => {
                let plugins_redteam = context.for_node("redteam");
                let cfg = node_agent_config(
                    engine,
                    config,
                    context.pool_for("redteam", client.map(|c| c.offer())),
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
        // Writing the change's tests needs the write tools the classifier has no use for, so the
        // set is built separately even though both hang off the same route.
        author: match classifier_enabled(engine, config) {
            true => {
                let plugins = context.for_node("redteam");
                let mut tools = context.pool_for("redteam", client.map(|c| c.offer()));
                tools
                    .local()
                    .tools
                    .extend(ratatoskr_agent::files::edit_declarations());
                let mut names: Vec<&str> = redteam::CLASSIFIER_TOOLS.to_vec();
                names.extend([
                    ratatoskr_agent::files::READ,
                    ratatoskr_agent::files::GREP,
                    ratatoskr_agent::files::GLOB,
                    ratatoskr_agent::files::WRITE,
                    ratatoskr_agent::files::EDIT,
                ]);
                let cfg = node_agent_config(engine, config, tools, "redteam", &names, &plugins)?;
                Some(redteam::TestAuthor {
                    route: cfg.route,
                    tools: cfg.tools,
                    policy: cfg.policy,
                    max_turns: cfg.max_turns,
                    system_prompt: cfg.system_prompt,
                    conventions: conventions.clone(),
                    plugins,
                    ledger: Some(Arc::clone(ledger)),
                })
            }
            false => None,
        },
    };
    let (impl_cfg, impl_plugins) =
        build_implementer_agent(engine, config, context, client.map(|c| c.offer()))?;
    let implementer = ImplementerNode {
        clarifier: Some(clarifier.as_dyn()),
        repo_path: repo_path.clone(),
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
        )?,
    };

    // Built before the fork so a misconfigured verifier fails the run here rather than after an
    // implementer session and a sandboxed test run have already been spent on it.
    let mut review = Review::build(run, plan)?;

    // The worktree first, because the red team writes the change's tests into it and cannot do
    // that until it exists.
    let worktree = implementer
        .prepare()
        .await
        .map_err(|e| PlanError::node("implementer", e))?;

    // And now the verifier can be told where the change lives. It is built above, before there is
    // a worktree to name, so this is the earliest point it can be rooted — see `rooted_at`.
    if let Some(review) = review.as_mut() {
        review.rooted_at(worktree.as_path());
    }

    // Red team next, not alongside: it characterises the baseline and writes the tests the change
    // will be judged against, and both have to be done before the implementer opens the tree. The
    // concurrency this gives up is small — the baseline is a minute against the implementer's ten
    // — and what it buys is that the tests are not written by the author of the code.
    let red_team_out = red_team
        .run_and_author(worktree.as_path(), issue, &plan.analyst.interface)
        .await
        .map_err(|e| PlanError::node("red_team", e))?;

    let mut impl_out = match implementer.work(&worktree).await {
        Ok(out) => out,
        Err(e) => {
            // A failed first attempt leaves nothing behind, as before.
            implementer.discard(&worktree).await;
            return Err(PlanError::node("implementer", e));
        }
    };

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
        // The implementer's own model turn is on the ledger under this name, and the row also
        // carries the iteration and the outcome.
        input: Some(serde_json::to_string(&plan.analyst)?),
        iteration: Some(1),
        ledger: Some(ledger),
    })
    .await?;

    // Hard guard: red-team must have actually characterized the baseline. If the test command
    // produced no tests, converge would compare against empty data and falsely "converge".
    if !converge::test_command_ran(
        &red_team_out.failing_tests,
        red_team_out.passed_tests,
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
    // Everything the review has said this run. Carried so a later pass can see the earlier ones,
    // and so the ceiling has the evidence to hand the analyst.
    let mut found_so_far: Vec<verifier::Finding> = Vec::new();
    // At most one re-plan per run: a second would be the same escalation on the same evidence.
    let mut replanned = false;
    let status = loop {
        let post_ran = converge::test_command_ran(
            &impl_out.failing_tests,
            impl_out.passed_tests,
            impl_out.exit_code,
        );
        // Tests written for this change before it existed. They fail in the baseline as a matter
        // of course, so "nothing newly failing" is not enough to call them satisfied.
        let authored = red_team_out
            .authored
            .as_ref()
            .map(|a| a.tests.as_slice())
            .unwrap_or_default();
        let unsatisfied = converge::unsatisfied(authored, &impl_out.failing_tests);
        let tests_clean = post_ran
            && unsatisfied.is_empty()
            && converge::is_converged(&red_team_out.failing_tests, &impl_out.failing_tests);

        // Did the change edit the referee? Checked BEFORE `tests_clean` is trusted: a conftest.py
        // that rewrites every outcome, or an edited test, makes the passing/failing sets describe a
        // bar the change wrote for itself.
        let referee =
            converge::referee_touches(&impl_out.rewritten_files, engine.may_modify_tests());

        // What to do next. The referee check comes first, then the test gate, then the review: a
        // moved referee makes the test result meaningless, and a test result is stronger evidence
        // than a model's judgement, so reviewing a change that does not build wastes the call.
        let correction: Reviewed = if !referee.is_empty() {
            tracing::warn!(files = ?referee, "iteration touched the referee; not accepting it");
            Reviewed::Fix(Box::new(Correction {
                prompt: converge::referee_correction(&referee),
                revised: None,
                found: Vec::new(),
            }))
        } else if !tests_clean {
            // A post-change run that didn't complete usually means the edit broke the build — say
            // that specifically instead of reporting "no new failures".
            let prompt = if !post_ran {
                format!(
                    "The test command did not run to completion (exit {}) — your change likely \
                     does not compile. Fix it so the tests run and pass.",
                    impl_out.exit_code
                )
            } else if !unsatisfied.is_empty() {
                // Said apart from the regression case, because it is the opposite situation: these
                // were failing before you started, and making them pass is the task.
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
            Reviewed::Fix(Box::new(Correction {
                prompt,
                revised: None,
                found: Vec::new(),
            }))
        } else if let Some(review) = &review {
            review
                .review(
                    run,
                    &in_force,
                    &impl_out,
                    &worktree,
                    iterations,
                    &found_so_far,
                )
                .await?
        } else {
            Reviewed::Clean
        };

        let correction = match correction {
            Reviewed::Clean => break RunStatus::Converged,
            // The change passed its tests and nobody was able to review it. Saying `Converged`
            // would claim a review that did not happen; failing would discard work that did.
            Reviewed::Unavailable => break RunStatus::Unreviewed,
            Reviewed::Fix(correction) => *correction,
        };
        // Everything the review has said this run, so the next pass can recognise a finding that
        // exists only because of the fix for an earlier one.
        found_so_far.extend(correction.found.iter().cloned());
        if let Some(revised) = correction.revised {
            in_force = revised;
            replanned = true;
        }
        if iterations >= config.implementer.max_iterations {
            // The budget is spent. Stopping here records "ran out of attempts", which is the one
            // reading the evidence usually does not support: a run that spends three iterations
            // trading each defect for its successor did not need a fourth attempt at the same
            // plan, it needed the plan looked at. So escalate once, then stop for real.
            if replanned || found_so_far.is_empty() {
                break RunStatus::MaxIterationsReached;
            }
            match review.as_ref() {
                None => break RunStatus::MaxIterationsReached,
                Some(review) => {
                    tracing::warn!(
                        iterations,
                        findings = found_so_far.len(),
                        "the iteration budget is spent; asking the analyst to look at the plan \
                         rather than recording another failed attempt"
                    );
                    let revised = review
                        .replan_at_ceiling(run, &in_force, &found_so_far, iterations)
                        .await?;
                    match revised {
                        None => break RunStatus::MaxIterationsReached,
                        Some((revised, prompt)) => {
                            in_force = revised;
                            replanned = true;
                            iterations += 1;
                            impl_out = match implementer.iterate(&worktree, &prompt).await {
                                Ok(out) => out,
                                Err(e) => {
                                    if let Err(rm) = remove_worktree(&repo_path, &worktree).await {
                                        tracing::warn!("failed to clean up worktree: {rm}");
                                    }
                                    return Err(PlanError::node("implementer", e));
                                }
                            };
                            record(Record {
                                store,
                                run_id,
                                node: "implementer",
                                output: &impl_out,
                                input: Some(serde_json::to_string(&in_force)?),
                                iteration: Some(iterations),
                                ledger: Some(ledger),
                            })
                            .await?;
                            continue;
                        }
                    }
                }
            }
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

    // The run's work becomes a commit on the run's own branch, whatever the outcome. A worktree
    // left with uncommitted changes is work that exists nowhere a reviewer can reach: the branch
    // still points where it forked from, so pushing it delivers nothing and a pull request against
    // it is refused for having no commits — which is what happened before this existed.
    //
    // Whatever the outcome, because the commit is a record rather than an endorsement. A run that
    // hit its iteration ceiling still produced something worth looking at, and publishing decides
    // separately whether anyone should be asked to.
    let branch = implementer.branch();
    match ratatoskr_exec::commit_all(
        &worktree,
        &branch,
        &commit_message(&config.publish, issue, &impl_out),
        ratatoskr_exec::Committer {
            name: &config.publish.committer_name,
            email: &config.publish.committer_email,
        },
    )
    .await
    {
        Ok(Some(sha)) => {
            tracing::info!(kind = "committed", branch = %branch, sha = %sha, "committed")
        }
        Ok(None) => tracing::info!(branch = %branch, "nothing to commit"),
        // Best-effort: the work and its checkpoints are already recorded, and failing the run here
        // would discard them over a step that only makes them reachable.
        Err(e) => tracing::warn!("could not commit the run's work to {branch}: {e}"),
    }

    Ok((red_team_out, impl_out, worktree, status, iterations))
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
    async fn a_default_plugin_nobody_installed_narrows_instead_of_failing() {
        // #185: `[rag_rat]` is optional, but the shipped ruleset still declares
        // `defineDefaults({ plugins: ["rag-rat"] })` — and a default applies to every node, so it
        // is a preference. Treating it like an explicit binding failed a rag-rat-less run on the
        // `plugins` node before any work started. (The tracing::warn! the contract asks for is
        // not asserted: capturing the global subscriber races the rest of the test suite.)
        let engine = binding_engine(
            "default-missing",
            r#"defineDefaults({ plugins: ["rag-rat"] });"#,
        )
        .await;
        let (config, cwd) = resolve_fixture("default-missing", &["ponytail"]);

        let context = PluginContext::resolve(&config, &engine, &cwd)
            .await
            .expect("a missing default narrows the tool pool; it does not fail the run");

        // The missing plugin is absent from every node's bound set, and nothing a node binds was
        // not actually discovered. Whether the narrowed set keeps `ponytail` or is empty is the
        // implementer's choice — the contract is only "only discovered plugins".
        for node in ["scout", "memory", "analyst", "implementer", "bookkeeper"] {
            let bound = context.bound(node);
            assert!(
                !bound.iter().any(|name| name == "rag-rat"),
                "{node} still binds a plugin nobody installed: {bound:?}"
            );
            assert!(
                bound.iter().all(|name| name == "ponytail"),
                "{node} binds something that was not discovered: {bound:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_missing_default_beside_a_discovered_agent_add_resolves_as_before() {
        // The narrowing must not disturb the satisfied path: both names discovered, both bound,
        // defaults first then the add.
        let engine = binding_engine(
            "default-and-add",
            r#"
            defineDefaults({ plugins: ["rag-rat"] });
            defineAgent("analyst", { plugins: { add: ["ponytail"] } });
            "#,
        )
        .await;
        let (config, cwd) = resolve_fixture("default-and-add", &["rag-rat", "ponytail"]);

        let context = PluginContext::resolve(&config, &engine, &cwd)
            .await
            .expect("everything named was discovered");
        assert_eq!(context.bound("analyst"), ["rag-rat", "ponytail"]);
        // A node that says nothing still gets exactly the defaults — discovering `ponytail` does
        // not bind it, the ruleset named the pool.
        assert_eq!(context.bound("scout"), ["rag-rat"]);
    }

    #[tokio::test]
    async fn when_everything_declared_is_discovered_resolve_is_unchanged() {
        // The strict path must not regress for a fully satisfied ruleset: no warning, no
        // narrowing, the defaults reach a node that says nothing.
        let engine =
            binding_engine("all-found", r#"defineDefaults({ plugins: ["rag-rat"] });"#).await;
        let (config, cwd) = resolve_fixture("all-found", &["rag-rat", "ponytail"]);

        let context = PluginContext::resolve(&config, &engine, &cwd)
            .await
            .expect("everything named was discovered");
        assert_eq!(context.bound("scout"), ["rag-rat"]);
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

    #[tokio::test]
    async fn no_plugins_discovered_at_all_still_runs_when_only_defaults_name_one() {
        // The rag-rat-less setup taken to its end: nothing installed, a default naming rag-rat,
        // and the run proceeds on file tools. A default is a preference.
        let engine = binding_engine(
            "none-discovered",
            r#"defineDefaults({ plugins: ["rag-rat"] });"#,
        )
        .await;
        let (config, cwd) = resolve_fixture("none-discovered", &[]);

        let context = PluginContext::resolve(&config, &engine, &cwd)
            .await
            .expect("nothing discovered and only a default declared: there is nothing to require");
        assert!(context.discovered.is_empty());
        for node in ["scout", "analyst", "implementer"] {
            assert!(
                !context.bound(node).iter().any(|name| name == "rag-rat"),
                "nothing was discovered, so nothing undiscovered may be bound"
            );
        }
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
    async fn a_node_that_declares_no_tools_is_not_handed_the_file_tools() {
        // "No tools" has to mean none. The characterizer transcribes output it was handed, and
        // when this leaked it spent its turn reading directories on the host and inventing a
        // diagnosis of the run instead of naming the checks.
        let engine = engine("no-tools").await;
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "characterizer".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".into(),
                model: "claude-haiku-4-5-20251001".into(),
                max_tokens: None,
                temperature: None,
                params: None,
                session: Default::default(),
            },
        );

        let none = node_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            "characterizer",
            &[],
            &NodePlugins::default(),
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
        let some = node_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            "analyst",
            analyst::ANALYST_TOOLS,
            &NodePlugins::default(),
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

    /// Whether `choose` would spend a model call, given what the repo defines and what was asked.
    fn would_consult(defined: usize, named: bool, configured: bool) -> bool {
        !named && defined > 1 && configured
    }

    #[test]
    fn the_overseer_is_consulted_only_when_there_is_a_real_choice() {
        // A caller that named a workflow said which shape it wanted and is not asking to be
        // second-guessed.
        assert!(!would_consult(3, true, true));
        // One or none resolves without a model call: paying for a decision with one answer is
        // waste, and the built-in is what a repo defining nothing gets.
        assert!(!would_consult(1, false, true));
        assert!(!would_consult(0, false, true));
        // Unconfigured, the run refuses to guess rather than picking for itself.
        assert!(!would_consult(3, false, false));
        // The only case worth a call.
        assert!(would_consult(2, false, true));
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

        let err = match select(registry, Some("invented")) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a name that is not in the registry must not select anything"),
        };
        assert!(err.contains("no workflow named `invented`"), "{err}");
        assert!(err.contains("research"), "{err}");
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
