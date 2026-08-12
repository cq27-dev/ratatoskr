//! Concrete Phase 2 nodes (scout, memory, analyst) and the straight-line `plan` executor.
//!
//! Per the Phase 2 decision, this is a plain sequential `async fn`, not a generic edge-walking
//! interpreter: with exactly three fixed nodes and no branching, `run_plan` delivers the same
//! policy guarantee (schema-validated handoffs, a checkpoint after every node, nothing skipped)
//! with nothing to get wrong. The real executor arrives in Phase 3 when fork/join needs one.

pub mod bookkeeper;
pub mod child;
pub mod clarify;
pub mod contracts;
pub mod control;
pub mod converge;
pub mod implementer;
pub mod issue;
pub mod memory;
pub mod plugins;
mod policy;
pub mod publisher;
pub mod redteam;
pub mod referee;
pub mod skills;
pub mod stage;
pub mod testrun;
pub mod validate;
pub mod verifier;
pub mod workflow;

pub use contracts::{analyst, context, overseer, scout};

pub(crate) use plugins::stage_agent_config;
pub use plugins::{NodePlugins, PluginContext};
#[cfg(test)]
use plugins::{default_allow, servers_to_start};

pub use analyst::AnalystOutput;
pub use bookkeeper::{BookkeeperInput, BookkeeperOutput, MemoryWritten};
pub use child::ChildTask;
pub use context::{Constraint, ContextOutput};
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
use ratatoskr_exec::WorktreePath;
use ratatoskr_graph::NodeError;
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
        configured,
        config,
        store,
        run_id,
        issue,
        engine,
        ..
    } = request;
    let runtime = match chosen {
        Workflow::BuiltIn => workflow::standard_runtime().await?,
        Workflow::Scripted(runtime) => runtime,
    };
    // Once per run: `SessionStart` describes the repository, not the stage or entrypoint.
    let plugin_context =
        PluginContext::resolve(config, engine, &std::env::current_dir().unwrap_or_default())
            .await?;
    let ctx = workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
        client,
        configured,
        config,
        store,
        run_id,
        issue,
        engine,
        plugin_context,
        ledger: Arc::new(RunLedger::default()),
    })?;
    workflow::run_plan_scripted(runtime, ctx).await
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
            None,
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
    graph_fingerprint_of(repo, &workflow::standard_definitions().unwrap_or_default())
}

/// `graph_fingerprint` against explicit standard definitions, so the contribution of the embedded
/// module is testable without rebuilding the binary.
fn graph_fingerprint_of(repo: &std::path::Path, definitions: &str) -> String {
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
    // The same module map a run gives a workflow, so one that legitimately imports the standard
    // definitions still reports its `LOAD` dependencies instead of failing to transpile.
    let modules = [(workflow::STANDARD_DEFINITIONS_MODULE, definitions)];
    for workflow in &workflows {
        if let Ok(dependencies) = ratatoskr_script::workflow::dependencies(workflow, &modules) {
            sources.extend(dependencies);
        }
    }
    sources.extend(workflows);
    // Sorted, because `read_dir` order is the filesystem's business and a fingerprint that depends
    // on it would differ between two checkouts of identical files.
    sources.sort();
    sources.dedup();

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut fold = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    // The standard definitions are compiled into the binary rather than checked into the repo, so
    // no dependency walk can reach them — yet they carry the stages every workflow imports, with
    // their `LOAD`ed prompts already inlined by the transpile. Without this, two builds that run
    // demonstrably different graphs would report the same hash.
    fold(definitions.as_bytes());
    for path in sources {
        // A missing file still contributes its name, so adding a `workflow.ts` changes the
        // fingerprint even if it is empty.
        fold(
            path.strip_prefix(repo)
                .unwrap_or(&path)
                .as_os_str()
                .as_encoded_bytes(),
        );
        fold(&std::fs::read(&path).unwrap_or_default());
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

const SCRIPTED_REVIEW_WARNING: &str = "this workflow controls whether to run the verifier; if it \
    omits verify(), the change will be accepted on its Rust-owned test and referee gates alone";

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
/// The built-in uses the bundled standard TypeScript runtime for composition. Its host operations
/// still own the referee, review routing, frozen acceptance, iteration limits, checkpoints, and
/// terminal effects, so repository-authored sequencing cannot weaken those gates.
pub enum Workflow {
    /// context → analyst → red-team → implementer → verify/converge → terminal delivery.
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
/// legitimate whether or not this particular run uses that workflow. The internal referee is
/// never governable, even when a workflow names it.
fn governable_from<'a>(workflows: impl IntoIterator<Item = &'a WorkflowRuntime>) -> Vec<String> {
    let mut names: Vec<String> = BUILT_IN_NODES.iter().map(|s| s.to_string()).collect();
    for workflow in workflows {
        names.extend(workflow.meta().nodes.iter().cloned());
        names.extend(workflow.meta().stages.iter().map(|stage| stage.id.clone()));
    }
    names.retain(|name| policy::reserved(name) != Some(policy::Reserved::InternalGate));
    names.sort();
    names.dedup();
    names
}

pub async fn governable_nodes() -> Result<Vec<String>, PlanError> {
    Ok(governable_from(&defined().await?))
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
    defined_in(
        std::path::Path::new(WORKFLOW_DIR),
        std::path::Path::new(LEGACY_WORKFLOW),
    )
    .await
}

/// `defined()` against explicit paths, so the import wiring is testable without a fixed cwd.
async fn defined_in(
    dir: &std::path::Path,
    legacy_path: &std::path::Path,
) -> Result<Vec<WorkflowRuntime>, PlanError> {
    let fail = |e: ratatoskr_script::ScriptError| {
        PlanError::node("workflow", NodeError::Failed(e.to_string()))
    };
    // A repository's own workflow imports the standard definitions exactly as the bundled workflow
    // does — otherwise a repo could only restate the stages it wanted to reuse.
    let definitions = workflow::standard_definitions()?;
    let modules = [(workflow::STANDARD_DEFINITIONS_MODULE, definitions.as_str())];
    let mut found = WorkflowRuntime::discover(dir, &modules)
        .await
        .map_err(fail)?;
    if let Some(legacy) = WorkflowRuntime::load(legacy_path, &modules)
        .await
        .map_err(fail)?
    {
        // Only when the directory has not already claimed that name, so a repo mid-move does not
        // get a duplicate-name error for a file it has already copied across.
        if !found.iter().any(|w| w.meta().name == legacy.meta().name) {
            found.push(legacy);
        }
    }
    // The bundled workflow is always in the registry, so a repository workflow answering to its
    // name puts two rows with one name in `ratatoskr workflows` and makes `--workflow built-in`
    // resolve to whichever the registry listed first. Two *scripted* workflows sharing a name are
    // already refused; the bundled one is a name just as taken.
    if found.iter().any(|w| w.meta().name == BUILT_IN) {
        return Err(PlanError::node(
            "workflow",
            NodeError::Failed(format!(
                "a workflow in this repository is named `{BUILT_IN}`, which is the bundled \
                 workflow's name; the bundled workflow is always in the registry, so that name is \
                 taken — rename it in its `defineWorkflow` call"
            )),
        ));
    }
    Ok(found)
}

/// Validate the stage registry every configured workflow contributes before any run starts.
pub async fn validate_configured_stages(config: &RatatoskrConfig) -> Result<(), PlanError> {
    let workflows = defined().await?;
    let standard_stages = workflow::standard_stages().await?;
    validate_configured_stage_registry(config, &workflows, standard_stages)
}

fn validate_configured_stage_registry(
    config: &RatatoskrConfig,
    workflows: &[WorkflowRuntime],
    standard_stages: Vec<Stage>,
) -> Result<(), PlanError> {
    let profiles = agent_profiles(config);
    // The registry the run will execute, and nothing else. `overlaid_stages` builds its base from
    // exactly this expression, so what validates here is what runs. A base carrying extra built-in
    // stages validated ghosts — `governedBy: "red_team"` passed startup against a stage the run
    // never registers, and was then refused by `governable_nodes()` when a ruleset appeared.
    let base = standard_stages;

    // Each workflow is judged against its *own* registry — the base with its declarations laid over
    // it — not against one pool of everything configured. Pooling rejects the documented case of a
    // workflow overriding `analyst`, and rejects it twice over when two workflows each override the
    // same standard id: a run only ever executes one workflow, so they never meet.
    let judge = |declared: Vec<Stage>, mut permitted: Vec<String>| -> Result<(), PlanError> {
        let mut stages = base.clone();
        stage::overlay(&mut stages, declared);
        permitted.extend(stages.iter().map(|stage| stage.id.clone()));
        permitted.sort();
        permitted.dedup();
        validate::validate(&stages, &profiles, &permitted)
    };

    if workflows.is_empty() {
        return judge(Vec::new(), governable_from(std::iter::empty()));
    }
    for workflow in workflows {
        let declared = stage::stages_from_workflow(workflow.meta());
        validate::validate_declared_contracts(&declared)?;
        validate::validate_declarations(&declared, &workflow.meta().name)?;
        judge(declared, governable_from([workflow]))?;
    }
    Ok(())
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
fn overseer_enabled(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    stages: &[Stage],
) -> bool {
    node_route(engine, config, stages, "overseer").is_some()
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
    // The choice comes first and the configuration second, so a run with nothing to choose between
    // does not evaluate the bundled registry to find that out.
    let consult = should_consult_overseer(defined_workflows, request.workflow.is_some(), true)
        // Selection runs before a workflow exists, so the bundled registry is the only one there
        // is — and `overseer` is reserved against declaration, so no override could change it.
        && overseer_enabled(
            request.engine,
            request.config,
            &workflow::standard_stages().await?,
        );
    if !consult {
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
    let cwd = std::env::current_dir().unwrap_or_default();
    let plugin_context = PluginContext::resolve(request.config, request.engine, &cwd).await?;
    let ctx = workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
        client: request.client,
        configured: request.configured,
        config: request.config,
        store: request.store,
        run_id: request.run_id,
        issue: request.issue,
        engine: request.engine,
        plugin_context,
        ledger: Arc::new(RunLedger::default()),
    })?;
    let raw = workflow::evaluate_standard_stage(
        Arc::clone(&ctx),
        workflow::SELECTION_STAGE_ID,
        input_json.clone(),
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
    /// Repository-configured MCP server offers, in configured precedence order.
    pub configured: &'a [ServerTools],
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
fn classifier_enabled(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    stages: &[Stage],
) -> bool {
    node_route(engine, config, stages, "redteam").is_some()
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
    configured: &'a [ServerTools],
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

        let context = include_str!("contracts/context.rs");
        assert!(
            !context.contains(concat!("Context", "Node")),
            "context must not retain an obsolete direct wrapper"
        );
        assert!(
            include_str!("workflow.rs").contains("context_distillation"),
            "the context operation must retain its declared StageExecutor turn"
        );

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
                include_str!("contracts/analyst.rs"),
                concat!("Analyst", "Node"),
            ),
            (
                "bookkeeper",
                include_str!("bookkeeper.rs"),
                concat!("Bookkeeper", "Node"),
            ),
            (
                "overseer",
                include_str!("contracts/overseer.rs"),
                concat!("Overseer", "Node"),
            ),
            (
                "publisher",
                include_str!("publisher.rs"),
                concat!("Publisher", "Node"),
            ),
            (
                "scout",
                include_str!("contracts/scout.rs"),
                concat!("Scout", "Node"),
            ),
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

/// Run the selected full workflow through the TypeScript composition runtime.
///
/// The bundled standard workflow uses the same host registry as a repository workflow. Rust still
/// owns every operation, gate, checkpoint, terminal action, and cleanup decision; the script owns
/// only their order.
pub async fn run_full(request: RunRequest<'_>) -> Result<RunOutcome, PlanError> {
    validate_configured_stages(request.config).await?;
    control::install(request.run_id);
    let filled = issue::enriched(request.issue, &std::env::current_dir().unwrap_or_default()).await;
    let request = RunRequest {
        issue: &filled,
        ..request
    };
    let chosen = choose(&request).await?;
    let RunRequest {
        client,
        configured,
        config,
        store,
        run_id,
        issue,
        engine,
        ..
    } = request;
    let (runtime, repository_workflow) = match chosen {
        Workflow::BuiltIn => (workflow::standard_runtime().await?, false),
        Workflow::Scripted(runtime) => (runtime, true),
    };
    if repository_workflow {
        tracing::warn!(workflow = runtime.meta().name, "{SCRIPTED_REVIEW_WARNING}");
    }
    let plugin_context =
        PluginContext::resolve(config, engine, &std::env::current_dir().unwrap_or_default())
            .await?;
    let ctx = workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
        client,
        configured,
        config,
        store,
        run_id,
        issue,
        engine,
        plugin_context,
        ledger: Arc::new(RunLedger::default()),
    })?;
    workflow::run_full_scripted(runtime, ctx).await
}

/// Run the publisher, when this repo has turned it on and there is an outcome worth delivering.
///
/// Best-effort, like bookkeeping: the change is made, the tests are recorded, and failing the run
/// because a tracker was unreachable would discard completed work over a delivery problem.
async fn publish_if_enabled(
    run: &Run<'_>,
    input: publisher::PublisherInput,
    terminal: bool,
    repository_root: &Path,
    worktree: Option<&WorktreePath>,
    turn: Arc<dyn workflow::StageTurn>,
) -> Option<PublisherOutput> {
    if !terminal || !run.config.publish.enabled {
        return None;
    }
    match publish_and_checkpoint(run, input, repository_root, worktree, turn).await {
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
    repository_root: &Path,
    worktree: Option<&WorktreePath>,
    turn: Arc<dyn workflow::StageTurn>,
) -> Result<PublisherOutput, PlanError> {
    let &Run {
        client,
        configured,
        config,
        store,
        run_id,
        issue,
        engine,
        context,
        ledger,
        ..
    } = run;
    // The terminal flow owns the worktree handle. A publisher sees the committed run worktree,
    // never its model-visible path or the terminal process's ambient directory. No-code delivery
    // keeps the run context's captured repository root only for the constrained `gh` action.
    let repo_root = worktree
        .map(WorktreePath::as_path)
        .unwrap_or(repository_root)
        .to_path_buf();
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
    let declared_context =
        workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
            client,
            configured,
            config,
            store,
            run_id,
            issue,
            engine,
            plugin_context: context.clone(),
            ledger: Arc::clone(ledger),
        })?;
    let raw = workflow::evaluate_standard_stage_with_resources_and_turn(
        declared_context,
        "publisher",
        input_json,
        workflow::StandardStageResources {
            resource_root: repo_root,
            rag_rat_worktree: worktree.map(|worktree| worktree.as_path().to_path_buf()),
            shell: None,
            publish: Some(workflow::StandardStagePublishResources { push }),
            clarifier: None,
            guidance: None,
        },
        turn,
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
        configured,
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
        let declared_context =
            workflow::WorkflowContext::new_with_ledger(workflow::WorkflowContextParams {
                client,
                configured,
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
            workflow::StandardStageResources {
                resource_root: repo_root,
                rag_rat_worktree: None,
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
    configured: &[ServerTools],
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
        configured,
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

/// Build the acceptance characterizer, when `[models.characterizer]` gives it somewhere to run.
///
/// Optional on purpose: without it a run still converges on exit codes, comparing at step
/// granularity. Coarser than named checks, and never wrong about them — so a repo that has not
/// configured one loses detail, not correctness.
pub(crate) fn build_characterizer(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    stages: &[Stage],
    declared_context: Option<Arc<workflow::WorkflowContext>>,
) -> Result<Option<testrun::Characterizer>, PlanError> {
    // Enablement only. What the turn runs on — route, tools, ceiling, prompt — is resolved by the
    // stage executor from the registry this gate reads, so resolving it a second time here could
    // only disagree with it.
    if node_route(engine, config, stages, "characterizer").is_none() {
        return Ok(None);
    }
    Ok(Some(testrun::Characterizer {
        declared_context: declared_context.expect("characterizer route requires its stage context"),
    }))
}

/// Resolve a node's model from its ruleset first, then its TOML route.
///
/// `stages` is the registry the run executes, so an override's profile — not the imported stage's —
/// decides where the node runs and therefore whether it runs at all.
///
/// Every lookup is keyed by the resolved stage's [`Stage::governance_id`], because that is the
/// identity the turn itself runs under: `declared_stage_agent_config` reads the ruleset and the
/// `[models.*]` route under it. Asking for `verifier` while the run's verifier declares
/// `governedBy: "review"` must therefore find `[models.review]` — resolving the profile through the
/// registry but the route under the caller's name made one decision out of two disagreeing answers,
/// which disabled a configured verifier or enabled one whose turn then had nowhere to run.
///
/// An un-overridden stage governs itself, so its governance id is its own id and this resolves
/// exactly as a lookup under the caller's name did.
fn node_route(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    stages: &[Stage],
    node: &str,
) -> Option<ratatoskr_core::ModelRoute> {
    // The profile keeps the caller's name: it is resolved from the stage this finds, and re-entering
    // the registry under a governance id could land on an unrelated stage that happens to bear it.
    let identity = stage::for_node(stages, node).map_or(node, Stage::governance_id);
    engine
        .ruleset(identity)
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
        .or_else(|| stage::profile_for(config, stages, node).and_then(|profile| profile.model))
        .or_else(|| config.models.get(identity).cloned())
}

/// The referee accepts only its TOML route, then falls back to the verifier's route. Its fixed
/// capability boundary is internal, so a `referee` ruleset is never consulted.
pub fn referee_route(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    stages: &[Stage],
) -> Option<ratatoskr_core::ModelRoute> {
    config
        .models
        .get("referee")
        .cloned()
        .or_else(|| node_route(engine, config, stages, "verifier"))
}

/// The verifier is opt-in on having somewhere to run, the same way the red-team classifier is.
fn verifier_enabled(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    stages: &[Stage],
) -> bool {
    node_route(engine, config, stages, "verifier").is_some()
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

pub(crate) async fn commit_worktree(
    config: &RatatoskrConfig,
    issue: &str,
    worktree: &WorktreePath,
    branch: &str,
    impl_out: &ImplementerOutput,
) {
    match ratatoskr_exec::commit_all(
        worktree,
        branch,
        &commit_message(&config.publish, issue, impl_out),
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

    #[tokio::test]
    async fn the_implementer_can_correct_a_memory_its_change_falsifies() {
        // The defect this closes: a review that found a memory contradicted by the diff routed the
        // finding here, and this node could read memories and write none — so converge asked for a
        // fix nobody in the run could make, every iteration, until the budget ran out.
        let stages = workflow::standard_stages().await.unwrap();
        let tools = &stages
            .iter()
            .find(|stage| stage.id == "implementer_attempt")
            .expect("the standard registry declares the implementer's attempt")
            .tools;
        assert!(
            tools.iter().any(|tool| tool == "memory_update"),
            "{tools:?}"
        );
        assert!(
            tools.iter().any(|tool| tool == "memory_mark_obsolete"),
            "{tools:?}"
        );
        // And composing new ones stays the bookkeeper's, done once with the whole run in view.
        assert!(
            !tools.iter().any(|tool| tool == "memory_create"),
            "{tools:?}"
        );
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
        tools.add_local(ratatoskr_agent::publish::declaration());
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
    async fn the_build_profile_caps_the_implementer_turn_unless_its_ruleset_overrides_it() {
        // Asserted through the path the attempt takes: `implementer_attempt` resolves its turn cap
        // from the profile it selects, and its ruleset — keyed by the governance identity,
        // `implementer` — is the more specific answer when it names one.
        let mut config = RatatoskrConfig::default();
        config.agents.insert(
            "build".to_string(),
            ratatoskr_core::AgentProfileConfig {
                capabilities: vec![ratatoskr_core::Capability::Write],
                max_turns: Some(250),
                ..Default::default()
            },
        );
        let stages = workflow::standard_stages().await.unwrap();
        let attempt = stages
            .iter()
            .find(|stage| stage.id == "implementer_attempt")
            .expect("the standard registry declares the implementer's attempt");

        let engine = engine("implementer-profile-turn-cap").await;
        let (cfg, _) = plugins::declared_stage_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            attempt,
            &[],
            &NodePlugins::default(),
        )
        .unwrap();
        assert_eq!(cfg.max_turns, Some(250));

        let engine = binding_engine(
            "implementer-ruleset-turn-cap",
            r#"defineAgent("implementer", { maxTurns: 7 });"#,
        )
        .await;
        let (cfg, _) = plugins::declared_stage_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            attempt,
            &[],
            &NodePlugins::default(),
        )
        .unwrap();
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
    async fn a_standard_stage_uses_its_selected_profile() {
        // Through the executor's resolver: a standard stage's model, turn cap, base prompt and
        // capability ceiling all come from the profile its declaration names, and the ceiling is
        // the narrower of the two — a `read` profile keeps the implementer's editing tools away
        // even though its stage declares `write`.
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

        let stages = workflow::standard_stages().await.unwrap();
        let attempt = stages
            .iter()
            .find(|stage| stage.id == "implementer_attempt")
            .expect("the standard registry declares the implementer's attempt");
        let default_tools = attempt.tools.iter().map(String::as_str).collect::<Vec<_>>();
        let (cfg, profile) = plugins::declared_stage_agent_config(
            &engine,
            &config,
            ToolSet::default(),
            attempt,
            &default_tools,
            &NodePlugins::default(),
        )
        .unwrap();
        assert_eq!(cfg.route.model, "profile-model");
        assert_eq!(cfg.system_prompt, None);
        assert_eq!(profile.base_prompt, "profile prompt");
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
        let stages = workflow::standard_stages().await.unwrap();
        let mut config = RatatoskrConfig::default();
        assert!(!classifier_enabled(&engine, &config, &stages));
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
        assert!(classifier_enabled(&engine, &config, &stages));
    }

    #[tokio::test]
    async fn a_stage_governed_by_another_identity_is_enabled_by_that_identitys_route() {
        // `stage("verifier", { ...nodes.verifier, governedBy: "review" })` — startup accepts it,
        // because `review` is a repository-owned identity with no reservation of its own.
        let engine = engine("governed-route").await;
        let mut stages = workflow::standard_stages().await.unwrap();
        stages
            .iter_mut()
            .find(|stage| stage.id == "verifier")
            .expect("the standard registry declares the verifier")
            .governed_by = Some("review".to_string());
        let route = || ratatoskr_core::ModelRoute {
            context_window: None,
            provider: "openai".to_string(),
            model: "gpt-5".to_string(),
            max_tokens: None,
            temperature: None,
            params: None,
            session: Default::default(),
        };

        // `review` is what the turn resolves its route under, so a route there is somewhere to run.
        let mut config = RatatoskrConfig::default();
        config.models.insert("review".to_string(), route());
        assert!(verifier_enabled(&engine, &config, &stages));

        // A route left behind under the stage id is not: the turn would ask for `review`, find
        // nothing, and the review the gate promised would be reported as unavailable instead.
        let mut config = RatatoskrConfig::default();
        config.models.insert("verifier".to_string(), route());
        assert!(!verifier_enabled(&engine, &config, &stages));

        // A stage that governs itself — every stage in an unmodified registry — is unaffected.
        let stages = workflow::standard_stages().await.unwrap();
        assert!(verifier_enabled(&engine, &config, &stages));
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

        // The standard definitions are the graph every workflow imports, and they ship inside the
        // binary — no file in the repo to walk. Two builds whose `nodes.ts` or whose `LOAD`ed
        // prompts differ must not report the same provenance. (The transpiled source already has
        // the prompts inlined, so hashing it covers both.)
        assert_ne!(
            graph_fingerprint_of(&root, "export const analyst = { instructions: 'first' };"),
            graph_fingerprint_of(&root, "export const analyst = { instructions: 'second' };"),
        );
        assert_eq!(
            graph_fingerprint(&root),
            graph_fingerprint_of(&root, &workflow::standard_definitions().unwrap()),
            "the fingerprint is the one taken over this build's definitions"
        );

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
        let found = WorkflowRuntime::discover(&dir, &[]).await.unwrap();
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
    async fn the_default_standard_stage_registry_has_unique_identifiers() {
        validate_configured_stages(&RatatoskrConfig::default())
            .await
            .expect("the bundled standard declarations replace legacy terminal placeholders");
    }

    #[test]
    fn configured_registry_allows_governance_by_another_declared_stage() {
        let template = built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();
        let mut policy = template.clone();
        policy.id = "shared_policy".to_string();
        let mut consumer = template;
        consumer.id = "custom_plan".to_string();
        consumer.governed_by = Some(policy.id.clone());

        validate_configured_stage_registry(
            &RatatoskrConfig::default(),
            &[],
            vec![policy, consumer],
        )
        .expect("resolved declared stage IDs are permitted governance identities");
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
        let found = WorkflowRuntime::discover(&dir, &[]).await.unwrap();
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
        let found = WorkflowRuntime::discover(&dir, &[]).await.unwrap();
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
        let route = referee_route(&engine, &config, &built_in_stages())
            .expect("the verifier route is the fallback");
        assert_eq!(route.provider, "anthropic");
        assert_eq!(route.model, "claude-sonnet-4-6");

        // The same fallback through the verifier route's other spelling: ruleset("verifier").model.
        let ruleset = binding_engine(
            "referee-via-verifier-ruleset",
            r#"defineAgent("verifier", { model: { provider: "openai", model: "gpt-5" } });"#,
        )
        .await;
        let config = RatatoskrConfig::default();
        let route = referee_route(&ruleset, &config, &built_in_stages())
            .expect("ruleset verifier is the fallback");
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
        let route = referee_route(&engine, &config, &built_in_stages())
            .expect("a referee route is configured");
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
        let route = referee_route(&ruleset, &config, &built_in_stages())
            .expect("TOML referee route still wins");
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
            referee_route(&engine, &config, &built_in_stages()).is_none(),
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
    async fn a_repository_workflow_imports_a_standard_definition_and_changes_one_part() {
        // The acceptance criterion: a repo takes a standard stage and changes part of it without
        // restating the rest, through the same `defined()` path a real checkout uses.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-import-nodes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("ours.ts"),
            r#"import * as nodes from "ratatoskr/nodes";
               defineWorkflow({
                 name: "ours",
                 stages: [stage("analyst", { ...nodes.analyst, agent: "explore" })],
               });
               export async function plan(input) { return input; }"#,
        )
        .unwrap();

        let found = defined_in(&dir, &dir.join("absent.ts")).await.unwrap();
        let ours = &found[0].meta().stages[0];
        let standard = workflow::standard_runtime().await.unwrap();
        let theirs = standard
            .meta()
            .stages
            .iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();

        assert_eq!(ours.agent, "explore");
        assert_ne!(theirs.agent, "explore");
        // Everything not overridden is the standard definition's, including the instructions its
        // `LOAD` resolved and the question renderer it declares.
        assert!(!ours.instructions.is_empty());
        assert_eq!(ours.instructions, theirs.instructions);
        assert_eq!(ours.tools, theirs.tools);
        assert_eq!(ours.output_schema, theirs.output_schema);
        assert_eq!(ours.question_renderer, theirs.question_renderer);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Write one workflow per source and discover them the way a checkout does.
    async fn workflows_in(tag: &str, sources: &[(&str, &str)]) -> (PathBuf, Vec<WorkflowRuntime>) {
        let dir = std::env::temp_dir().join(format!("ratatoskr-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, source) in sources {
            std::fs::write(dir.join(format!("{name}.ts")), source).unwrap();
        }
        let found = defined_in(&dir, &dir.join("absent.ts")).await.unwrap();
        (dir, found)
    }

    #[tokio::test]
    async fn a_repository_workflow_cannot_take_the_bundled_workflows_name() {
        // The bundled workflow is always in the registry, so a second row under its name makes
        // `ratatoskr workflows` list two `built-in`s and `--workflow built-in` resolve to whichever
        // came first. Two scripted workflows sharing a name are already refused; so is this.
        let dir = std::env::temp_dir().join(format!("ratatoskr-name-clash-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("ours.ts"),
            format!(
                r#"defineWorkflow({{ name: "{BUILT_IN}" }});
                   export async function plan(input) {{ return input; }}"#
            ),
        )
        .unwrap();

        let error = match defined_in(&dir, &dir.join("absent.ts")).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("the bundled workflow's name is taken"),
        };
        assert!(error.contains(BUILT_IN), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_overridden_standard_stage_passes_startup_validation() {
        // The headline case, through the registry validation `validate_configured_stages` runs
        // before any node starts: the override keeps the id `analyst`, so pooling it alongside the
        // standard definition would reject the repo at startup as a duplicate identifier.
        let (dir, found) = workflows_in(
            "override-validate",
            &[(
                "ours",
                r#"import * as nodes from "ratatoskr/nodes";
                   defineWorkflow({
                     name: "ours",
                     stages: [stage("analyst", { ...nodes.analyst, instructions: "ours" })],
                   });
                   export async function plan(input) { return input; }"#,
            )],
        )
        .await;
        let standard = workflow::standard_stages().await.unwrap();
        validate_configured_stage_registry(&RatatoskrConfig::default(), &found, standard)
            .expect("a workflow overriding `analyst` is a valid configuration");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn two_workflows_may_each_override_the_same_standard_stage() {
        // A run executes one workflow, so two overrides of `analyst` never meet. Validating them
        // against one pooled registry would make each repo's second workflow illegal.
        let ours = |name: &str, instructions: &str| {
            format!(
                r#"import * as nodes from "ratatoskr/nodes";
                   defineWorkflow({{
                     name: "{name}",
                     stages: [stage("analyst", {{ ...nodes.analyst, instructions: "{instructions}" }})],
                   }});
                   export async function plan(input) {{ return input; }}"#
            )
        };
        let (dir, found) = workflows_in(
            "override-twice",
            &[
                ("first", ours("first", "one").as_str()),
                ("second", ours("second", "two").as_str()),
            ],
        )
        .await;
        assert_eq!(found.len(), 2, "both workflows were discovered");
        let standard = workflow::standard_stages().await.unwrap();
        validate_configured_stage_registry(&RatatoskrConfig::default(), &found, standard)
            .expect("two workflows may each override `analyst`");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_new_stage_id_validates_and_a_repeated_one_within_a_workflow_does_not() {
        // Overlay semantics must not swallow the case it looks like: an id declared twice by the
        // same workflow overrides nothing, it is a workflow that cannot say which one it meant.
        let (dir, found) = workflows_in(
            "new-stage-id",
            &[(
                "ours",
                r#"defineWorkflow({
                     name: "ours",
                     stages: [stage("reviewer", { agent: "reason", instructions: "review" })],
                   });
                   export async function plan(input) { return input; }"#,
            )],
        )
        .await;
        let standard = workflow::standard_stages().await.unwrap();
        validate_configured_stage_registry(&RatatoskrConfig::default(), &found, standard)
            .expect("a genuinely new stage id is still added");
        let _ = std::fs::remove_dir_all(&dir);

        let (dir, found) = workflows_in(
            "repeated-stage-id",
            &[(
                "ours",
                r#"defineWorkflow({
                     name: "ours",
                     stages: [
                       stage("reviewer", { agent: "reason", instructions: "one" }),
                       stage("reviewer", { agent: "reason", instructions: "two" }),
                     ],
                   });
                   export async function plan(input) { return input; }"#,
            )],
        )
        .await;
        let standard = workflow::standard_stages().await.unwrap();
        let error =
            validate_configured_stage_registry(&RatatoskrConfig::default(), &found, standard)
                .expect_err("a workflow may not declare one id twice");
        assert!(
            error
                .to_string()
                .contains("declares stage `reviewer` more than once"),
            "unexpected error: {error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_reserved_name_is_refused_by_the_startup_gate() {
        // `validate_configured_stages` is the first statement of `run_plan` and `run_full`, before
        // the run row and the `issue` checkpoint. A name conflict that only surfaced when the host
        // table was built killed the run mid-flight, with checkpoints already written.
        let cases = [
            ("context", "Rust-owned workflow operation"),
            ("bookkeeper", "terminal adapter"),
            ("publisher", "terminal adapter"),
            // Selection runs in its own pre-selection context, before any workflow runtime exists,
            // so a declaration of it could never reach the routing turn it appears to configure.
            ("overseer", "selects between workflows"),
        ];
        for (declared, expected) in cases {
            let (dir, found) = workflows_in(
                &format!("reserved-{declared}"),
                &[(
                    "ours",
                    &format!(
                        r#"defineWorkflow({{
                             name: "ours",
                             stages: [stage("{declared}", {{ agent: "reason", instructions: "x" }})],
                           }});
                           export async function plan(input) {{ return input; }}"#
                    ),
                )],
            )
            .await;
            let standard = workflow::standard_stages().await.unwrap();
            let error =
                validate_configured_stage_registry(&RatatoskrConfig::default(), &found, standard)
                    .expect_err("a reserved name must be refused at load");
            let error = error.to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
            assert!(
                error.contains(declared),
                "the error must name the stage: {error}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
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
        let found = WorkflowRuntime::discover(&dir, &[]).await.unwrap();
        assert_eq!(found[0].meta().nodes, ["referee"]);
        assert!(
            !governable_from(&found).iter().any(|name| name == "referee"),
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
            referee_route(&engine, &config, &built_in_stages()).is_none(),
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

        let resolved = referee_route(&engine, &config, &built_in_stages())
            .expect("[models.referee] is configured");
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
        let resolved = referee_route(&engine, &config, &built_in_stages())
            .expect("the verifier route is the fallback");
        assert_eq!(resolved.provider, "anthropic");
        assert_eq!(resolved.model, "claude-sonnet-4-6");
    }
}
