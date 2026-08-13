//! Deriving per-node activity from what the store actually records.
//!
//! There is no per-node status column: `runs.status` is one value for the whole run and
//! `checkpoints` is an append-only log keyed by `node_name`. Everything here is inference over
//! those two facts, and the rules below are deliberately explicit about the places the pipeline
//! is *not* uniform rather than hiding them behind a clever general rule.

use ratatoskr_store::Checkpoint;
use serde::Serialize;

/// Whether this node's own failure can be what made a run `failed`.
///
/// A run fails when its workflow entry returns an error — any host call that returns `Err` and is
/// not caught. `bookkeeper` and `publisher` never can: they run after the terminal status is
/// written and their failure is only logged.
///
/// `verifier` is the one that depends on the run. `verify_host` turns a review that could not *run*
/// into an `unavailable` result rather than an error — a change that was made and passed its tests
/// is not failed by the reviewer being unreachable — but everything around that review still
/// errors: `verify()` called before `implement()`, an analyst argument that no longer matches its
/// checkpoint, an overlap with `iterate()`, a store read that failed. Those fail the run, and the
/// call reaches none of them without a route to review with: without one it answers "not
/// configured" before it looks at the run at all. So the question is not the name but this run's
/// config, read exactly as [`PlannedNode::of`] reads it. A verifier routed only from a ruleset
/// therefore reads as unrouted here, and such a run is attributed as it was before — the
/// approximation only ever withholds the doubt, never invents it.
///
/// Every other name is fallible, including a stage a workflow declared itself: its host error
/// propagates out of the script and fails the run.
fn can_fail_the_run(name: &str, config: Option<&ratatoskr_core::RatatoskrConfig>) -> bool {
    match name {
        "bookkeeper" | "publisher" => false,
        "verifier" => PlannedNode::of(config, "verifier").is_some(),
        _ => true,
    }
}

/// The issue text is checkpointed under this name so `bookkeep` can replay a stored run. It is
/// not a node — it's the run's input, and it's the only record of the run's subject.
pub const ISSUE_NODE: &str = "issue";

/// The one node whose caller [`caller_of`] resolves, and the node it resolves to.
///
/// `referee` qualifies because what the resolution needs is guaranteed rather than inferred: it is a
/// fixed internal gate that `validate.rs` refuses as a declared stage identifier, so the name cannot
/// belong to anything else; it is not routed under a governance alias, so the name in the record is
/// its own; and every instance of it in a run resolves to the same caller, so collapsing a run's
/// referee checkpoints into one node loses nothing.
const REFEREE_NODE: &str = "referee";
const IMPLEMENTER_NODE: &str = "implementer";

/// What a node is doing, as far as the store can honestly say.
///
/// The qualifier is load-bearing. This is inferred from checkpoints, which are durable and prove
/// what *completed* — and cannot answer what is happening now. Two cases it gets backwards, both
/// mid-converge: the implementer holds a checkpoint while still being re-run, so it reads as
/// `Working`; the verifier is an optional stage that has not checkpointed, so it reads as `Idle`.
/// A client with the event stream knows better and is expected to prefer it; one without gets the
/// best answer the store can give.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeState {
    /// No checkpoint and not currently reachable: either it hasn't started or it never runs for
    /// this kind of run (a `plan` run never reaches the fork).
    Idle,
    Working,
    Done,
    /// The run failed and this is where it stopped.
    Failed,
}

/// One node's row in a run's pipeline view.
#[derive(Debug, Clone, Serialize)]
pub struct NodeView {
    pub name: String,
    pub state: NodeState,
    /// Which stage this node belongs to, and which lane within it. The pipeline's shape is the
    /// server's to know — a workflow that declares its own nodes changes it — so the graph is
    /// positioned from these rather than from a table the frontend maintains in parallel.
    pub stage: usize,
    pub lane: usize,
    /// Whether the run's recorded shape is what put it there. False means [`append_unknown`]
    /// placed it, in first-checkpoint order — an order a client holding the event stream can
    /// better, since completion order is not start order once a workflow runs hosts concurrently.
    /// The distinction has to be on the wire: the two are otherwise indistinguishable, and a client
    /// that reordered a declared layout would be redrawing the graph the workflow asked for.
    pub shaped: bool,
    /// How many checkpoints this node wrote. Only the implementer (per converge iteration) and
    /// the bookkeeper (via `ratatoskr bookkeep` replay) can exceed one.
    pub checkpoints: usize,
    pub first_at: Option<String>,
    pub last_at: Option<String>,
    /// What the node ran on and cost, from its most recent checkpoint. Absent for a node that has
    /// not checkpointed, and for one that ran no model at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<NodeTelemetryView>,
    /// What this node *would* run on, read from config. Present before the node has run, so the
    /// pipeline says what it is going to do rather than staying blank until it does it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub planned: Option<PlannedNode>,
    /// The node that ran this one, for a node the shape does not place — see [`caller_of`]. A node
    /// the shape does place needs no attribution: its position already says what preceded it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caller: Option<String>,
}

/// A node's configured route: what it will run on, and the two choices that change how it behaves.
///
/// Tools are absent on purpose. They come from the node's built-in list, its ruleset, and whatever
/// the connected MCP servers actually offer — none of which this process can know without starting
/// the script engine and the servers. They arrive when the node announces itself.
#[derive(Debug, Clone, Serialize)]
pub struct PlannedNode {
    pub model: String,
    pub thinking: bool,
    pub reuses_session: bool,
    pub session: ratatoskr_core::SessionScope,
}

impl PlannedNode {
    /// Read a node's route out of the config, if it has one. A node with no route never runs.
    fn of(config: Option<&ratatoskr_core::RatatoskrConfig>, node: &str) -> Option<Self> {
        // `red_team` checkpoints under that name but is routed as `redteam`.
        let key = if node == "red_team" { "redteam" } else { node };
        let route = config?.models.get(key)?;
        Some(PlannedNode {
            model: format!("{}/{}", route.provider, route.model),
            thinking: route
                .params
                .as_ref()
                .and_then(|p| p.get("thinking"))
                .and_then(|t| t.get("type"))
                .and_then(|t| t.as_str())
                != Some("disabled"),
            reuses_session: matches!(route.session, ratatoskr_core::SessionScope::Reuse),
            session: route.session,
        })
    }
}

/// A node's model, cost, and the two facts a reader cannot infer from either: which tools it could
/// call, and whether it kept its session across attempts.
#[derive(Debug, Clone, Serialize)]
pub struct NodeTelemetryView {
    pub model: Option<String>,
    /// Model calls in the node's latest attempt.
    pub turns: Option<u64>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    /// Tokens written to cache rather than read from it. Billed at a premium and the number that
    /// separates a run that reused its context from one that rebuilt it, so it is reported rather
    /// than folded into the input total.
    pub cache_creation_input_tokens: u64,
    /// Non-zero when the model reasoned before answering. Zero from endpoints that do not report
    /// it, which is why `thinking` exists alongside.
    pub reasoning_tokens: u64,
    /// Whether the node was left free to reason. Configured, not observed — see `reasoning_tokens`.
    pub thinking: bool,
    pub duration_ms: Option<u64>,
    pub tools: Vec<String>,
    /// Of those, the ones it actually called.
    pub tools_used: Vec<String>,
    /// The node's memory carried over from an earlier attempt in this run.
    pub reuses_session: bool,
}

impl NodeTelemetryView {
    /// The node's latest checkpoint, when it recorded a model turn. `None` for a node that ran no
    /// model — the issue pseudo-node, or one whose turn was never claimed.
    fn latest(checkpoints: &[Checkpoint], node: &str) -> Option<Self> {
        let t = &checkpoints.iter().rfind(|c| c.node_name == node)?.telemetry;
        t.model.as_ref()?;
        Some(NodeTelemetryView {
            model: t.model.clone(),
            turns: t.turns,
            input_tokens: t.usage.input_tokens,
            output_tokens: t.usage.output_tokens,
            cached_input_tokens: t.usage.cached_input_tokens,
            cache_creation_input_tokens: t.usage.cache_creation_input_tokens,
            reasoning_tokens: t.usage.reasoning_tokens,
            thinking: t.thinking,
            duration_ms: t.duration_ms,
            tools: t.tools.clone(),
            tools_used: t.tools_used.clone(),
            reuses_session: t.reuses_session,
        })
    }
}

/// Whether the run is no longer executing, from the status string the store holds.
///
/// The classification itself is `RunStatus::is_terminal`, where an exhaustive match makes the
/// compiler flag a new variant nobody classified. This only turns the stored string back into the
/// enum: a status that is absent, or one written by a newer build than this one, reads as still
/// executing — the safe direction, since it shows a stale run rather than declaring a live one
/// finished.
pub(crate) fn is_terminal(status: Option<&str>) -> bool {
    status
        .and_then(|s| s.parse::<ratatoskr_core::RunStatus>().ok())
        .is_some_and(|s| s.is_terminal())
}

/// Derive each node's state from the run status and its checkpoints.
///
/// Three non-uniformities are handled explicitly:
/// - **The implementer re-runs.** Converge checkpoints it once per iteration, so "has a
///   checkpoint" does not mean finished — while the run is live it is still converging, and the
///   fork stage is not complete no matter how many checkpoints it has.
/// - **The bookkeeper runs before the terminal status is written**, so it remains working while
///   its provider request can be paused and resumed. A bookkeeping failure is only logged, so it
///   can never cause a `failed` run or be reported `Failed`. A terminal run with no bookkeeper
///   checkpoint is either silently failed or never applicable, and is reported `Idle` rather than
///   guessed at. Pair it with the run's `last_activity` to judge.
/// - **A `failed` run whose fork ran, with nothing fallible after it, died in converge.** The
///   implementer is then where it stopped, even though it has checkpoints from earlier iterations.
///   The inference is only sound while nothing after the fork [`can_fail_the_run`], because a
///   workflow may declare a fallible stage after the implementer — and a run that routed its
///   verifier has a fallible one there already. Where one does, *no* node is reported `Failed`:
///   the implementer's re-entry and the stage after it leave the same record, which is the
///   implementer's last checkpoint and nothing following it, so naming either is a guess. An
///   unattributed failure is worse than a correct attribution and much better than a wrong one.
///
/// `config` is what the run was started under, so a node that has not run yet can still say what it
/// will run on.
pub fn derive_with(
    status: Option<&str>,
    checkpoints: &[Checkpoint],
    config: Option<&ratatoskr_core::RatatoskrConfig>,
    shape_json: Option<&str>,
) -> Vec<NodeView> {
    // The graph the run recorded, and only that. A run from another machine — or from this one
    // before the pipeline changed — is drawn against its own shape; one that recorded none is
    // placed entirely by `append_unknown`, from the records it has.
    let shape = ratatoskr_core::shape::recorded(shape_json);
    let stages = stages_of(&shape);
    let terminal = is_terminal(status);
    let failed = status == Some("failed");
    // The fork is wherever the implementer is in THIS run's shape, not a fixed index.
    let fork = shape
        .iter()
        .find(|n| n.name == "implementer")
        .map(|n| n.stage);
    let fork_started = fork.is_some_and(|f| {
        stages
            .get(f)
            .is_some_and(|nodes| nodes.iter().any(|n| count(checkpoints, &n.name) > 0))
    });
    // Blaming the fork for a failure needs everything after it to be unable to have caused one.
    // Delivery qualifies, running after the terminal write — but a declared layout may put a
    // fallible stage after the fork, and a routed verifier is fallible too, and then the failure is
    // as likely theirs as the implementer's. Nothing in the record separates the two, so the run is
    // left unattributed rather than pinned on the implementer.
    let nothing_fallible_after_the_fork = fork.is_some_and(|f| {
        stages
            .iter()
            .skip(f + 1)
            .flatten()
            .all(|n| !can_fail_the_run(&n.name, config))
    });
    let converge_died = failed && fork_started && nothing_fallible_after_the_fork;
    // And where that inference does not hold, nothing else may take its place. The implementer
    // re-enters once per converge iteration and checkpoints once per attempt, so an attempt that
    // died leaves exactly the record a later host dying leaves: the implementer's last checkpoint
    // and nothing after it. The cursor is then on a stage that never started, and reporting the
    // failure there is the same guess as reporting it on the implementer, made about whichever
    // node happens to be next. Both stay unattributed; the run's own status still says it failed.
    let unattributable = failed && fork_started && !converge_died;

    // A node has finished only if it checkpointed *and* isn't the implementer mid-converge —
    // otherwise the fork would look complete on iteration 1 and the run's activity would be
    // attributed to the bookkeeper, which by the invariant above hasn't started.
    let finished =
        |name: &str| count(checkpoints, name) > 0 && !(name == "implementer" && !terminal);
    // Where the run has got to: the first stage that has not finished and has nothing after it
    // checkpointed. Without that second half a skipped verifier would hold the position forever
    // and report every later node as not yet reached, on a run that has finished.
    //
    // Optional stages are never it. Whether one runs is decided by config the store does not
    // record, so an empty optional stage is as likely skipped as pending — and claiming the
    // overseer is working while the context node is what's actually running is the visible
    // version of that guess.
    let last_seen = stages
        .iter()
        .rposition(|nodes| nodes.iter().any(|n| count(checkpoints, &n.name) > 0));
    let current = stages.iter().enumerate().position(|(idx, nodes)| {
        !nodes.iter().all(|n| n.optional)
            && !(nodes.iter().all(|n| finished(&n.name))
                || last_seen.is_some_and(|seen| seen > idx))
    });

    let mut out = Vec::new();
    for (idx, nodes) in stages.iter().enumerate() {
        for node in nodes {
            let (lane, name) = (node.lane, &node.name);
            let times: Vec<&str> = checkpoints
                .iter()
                .filter(|c| c.node_name == *name)
                .map(|c| c.created_at.as_str())
                .collect();

            let state = if converge_died && Some(idx) == fork {
                // Converge died. Whatever the implementer checkpointed came from earlier
                // iterations; red-team ran to completion if it recorded anything.
                match name.as_str() {
                    "implementer" => NodeState::Failed,
                    _ if times.is_empty() => NodeState::Failed,
                    _ => NodeState::Done,
                }
            } else if times.is_empty() {
                match () {
                    // Later than where the run is: nothing to say about it yet.
                    _ if current != Some(idx) => NodeState::Idle,
                    // A failure here belongs upstream: delivery runs past the terminal status, and
                    // an unrouted verifier reviews nothing.
                    _ if !can_fail_the_run(name, config) => NodeState::Idle,
                    // Past a fork that ran, with something fallible after it: which of the two
                    // died is not in the record, so neither is named.
                    _ if unattributable => NodeState::Idle,
                    _ if failed => NodeState::Failed,
                    _ if !terminal => NodeState::Working,
                    _ => NodeState::Idle,
                }
            } else {
                checkpointed_state(name, terminal)
            };

            out.push(NodeView {
                telemetry: NodeTelemetryView::latest(checkpoints, name),
                planned: PlannedNode::of(config, name),
                name: name.clone(),
                state,
                stage: idx,
                lane,
                shaped: true,
                checkpoints: times.len(),
                first_at: times.first().map(|s| s.to_string()),
                last_at: times.last().map(|s| s.to_string()),
                // A shaped node's caller is its position: the stage before it ran it.
                caller: None,
            });
        }
    }
    append_unknown(&mut out, checkpoints, config, terminal);
    out
}

/// What a node that has checkpointed is doing, wherever it sits.
///
/// A checkpoint proves the node completed something — but the implementer is checkpointed once per
/// converge iteration, so while the run is live one of its checkpoints says the opposite of
/// finished. That is the whole rule, and both placements share it: a node the shape places and one
/// [`append_unknown`] appends are the same evidence read the same way.
fn checkpointed_state(name: &str, terminal: bool) -> NodeState {
    if name == IMPLEMENTER_NODE && !terminal {
        NodeState::Working
    } else {
        NodeState::Done
    }
}

/// Add nodes the run has data for that its recorded shape does not place.
///
/// Two cases reach here, and neither is exotic. A run from a DIFFERENT graph — an imported one, or
/// one from before the pipeline changed — carries a shape whose columns do not name the nodes in
/// its records; without this its checkpoints would be silently dropped and the run would appear to
/// have done nothing. And a workflow that declares no layout records an empty shape, so *every*
/// node of such a run arrives here, including its own. Nothing about this path is a fallback for
/// foreign data only.
///
/// They go in trailing stages, in the order they first CHECKPOINTED, which is the only order the
/// records carry. Concurrent hosts do not finish in the order they started, so they are marked
/// `shaped: false` and a client holding the event stream places them by first mention instead.
/// That is not the shape they executed
/// in — it cannot be recovered from checkpoints alone — but it shows every node with its output and
/// its cost, which is what someone analysing an unplaced run came for. One stage each, because
/// adjacent columns are drawn joined: a chain in first-checkpoint order is the least wrong claim
/// available, where a shared column would assert nodes ran side by side that merely lack a layout. What each node is *doing*
/// comes from [`checkpointed_state`], the same rule a placed node gets: a live implementer holds a
/// checkpoint from an earlier converge iteration and is still working, wherever it was drawn.
fn append_unknown(
    out: &mut Vec<NodeView>,
    checkpoints: &[Checkpoint],
    config: Option<&ratatoskr_core::RatatoskrConfig>,
    terminal: bool,
) {
    let known: std::collections::HashSet<&str> = out.iter().map(|n| n.name.as_str()).collect();
    let mut seen = std::collections::HashSet::new();
    // Each out-of-shape name with the position of its FIRST checkpoint, which is what its caller is
    // resolved from. One row aggregates every checkpoint of that name, so a run whose
    // `clarification` rows were asked for by different nodes cannot express all of them in one
    // `caller`. Splitting a row per caller belongs to the placement work (#248), which owns layout.
    let mut extra: Vec<(&str, usize)> = Vec::new();
    for (idx, c) in checkpoints.iter().enumerate() {
        let name = c.node_name.as_str();
        // The issue pseudo-node writes a checkpoint and is deliberately not a pipeline node: it
        // records what the run was asked to do, which is not a stage of doing it.
        if name != ISSUE_NODE && !known.contains(name) && seen.insert(name) {
            extra.push((name, idx));
        }
    }
    let base = out.iter().map(|n| n.stage).max().map_or(0, |s| s + 1);
    for (i, (name, first)) in extra.into_iter().enumerate() {
        let times = node_times(checkpoints, name);
        out.push(NodeView {
            telemetry: NodeTelemetryView::latest(checkpoints, name),
            planned: PlannedNode::of(config, name),
            caller: caller_of(checkpoints, first),
            name: name.to_string(),
            state: checkpointed_state(name, terminal),
            stage: base + i,
            lane: 0,
            shaped: false,
            checkpoints: times.len(),
            first_at: times.first().map(|s| s.to_string()),
            last_at: times.last().map(|s| s.to_string()),
        });
    }
}

/// Which node ran the checkpoint at `index`, when the log can say so without guessing.
///
/// Only the referee. Every `referee_judgement` call site in `workflow.rs` judges
/// `latest_checkpoint(store, run_id, "implementer")`, fetched immediately before, so the nearest
/// preceding implementer checkpoint is literally the output being judged — the resolution mirrors the
/// call rather than inferring from adjacency. `"implementer"` is hardcoded because only the
/// orchestrator knows what the referee judges; a run's `stage` and `lane` are declarative layout, not
/// evidence of invocation.
///
/// Everything else resolves to `None`, and that is a statement about the record rather than a gap for
/// a cleverer reader to fill. A checkpoint does not record what invoked it, and the two substitutes
/// available here do not survive the general case:
///
/// * **A name in a record is not necessarily the node that wrote it.** A declared stage runs under its
///   `governedBy` identity — `StageExecutor` passes the governance id as `NodeRun.node` — so a
///   clarification's `from`, taken from that same value, names the governance identity. Two stages
///   sharing one `governedBy` are already indistinguishable by the time a record is written.
/// * **Position is not provenance.** "Followed an implementer" is true of most work a run does late.
///
/// A caller for anything beyond the referee needs the producer to record it — an explicit caller
/// identity per invocation (#244) — and somewhere to put more than one, since `append_unknown`
/// collapses a name's checkpoints into a single node. Until then this stays narrow on purpose: a wrong
/// caller is worse than an absent one, because the graph draws it.
fn caller_of(checkpoints: &[Checkpoint], index: usize) -> Option<String> {
    let c = checkpoints.get(index)?;
    if c.node_name != REFEREE_NODE {
        return None;
    }
    checkpoints[..index]
        .iter()
        .rev()
        .find(|c| c.node_name == IMPLEMENTER_NODE)
        .map(|c| c.node_name.clone())
}

/// Group a shape's nodes into stages, indexed by column.
fn stages_of(
    shape: &[ratatoskr_core::shape::ShapeNode],
) -> Vec<Vec<&ratatoskr_core::shape::ShapeNode>> {
    let width = shape.iter().map(|n| n.stage).max().map_or(0, |s| s + 1);
    let mut stages: Vec<Vec<_>> = vec![Vec::new(); width];
    for node in shape {
        stages[node.stage].push(node);
    }
    for nodes in &mut stages {
        nodes.sort_by_key(|n| n.lane);
    }
    stages
}

/// When each of `node`'s checkpoints was written, in order.
fn node_times(checkpoints: &[Checkpoint], node: &str) -> Vec<String> {
    checkpoints
        .iter()
        .filter(|c| c.node_name == node)
        .map(|c| c.created_at.clone())
        .collect()
}

fn count(checkpoints: &[Checkpoint], node: &str) -> usize {
    checkpoints.iter().filter(|c| c.node_name == node).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout `standard-v1.ts` declares, as these cases' fixture.
    ///
    /// Written out here because these cases are *about* the standard pipeline — a skipped verifier
    /// holding the current position, a `failed` run that reached the fork, the two deliveries
    /// sharing a column. Nothing in this crate knows that layout: a run brings its own, and the
    /// workflow that ran is what declared it. Keep this in step with `standard-v1.ts` if the
    /// standard columns change; a stale fixture only makes these cases describe a pipeline that no
    /// longer exists, never a run drawn wrongly.
    fn standard_shape() -> String {
        shape_of(&[
            (&["overseer"], true),
            (&["context"], false),
            (&["analyst"], false),
            (&["red_team", "implementer"], false),
            (&["verifier"], true),
            (&["bookkeeper", "publisher"], false),
        ])
    }

    /// A recorded shape from its columns, each a list of lane names and whether it may be skipped.
    fn shape_of(columns: &[(&[&str], bool)]) -> String {
        let nodes: Vec<ratatoskr_core::shape::ShapeNode> = columns
            .iter()
            .enumerate()
            .flat_map(|(stage, (names, optional))| {
                names
                    .iter()
                    .enumerate()
                    .map(move |(lane, name)| ratatoskr_core::shape::ShapeNode {
                        name: (*name).to_string(),
                        stage,
                        lane,
                        optional: *optional,
                    })
            })
            .collect();
        serde_json::to_string(&nodes).unwrap()
    }

    /// A run of the standard pipeline, which is what every case below is about unless it says
    /// otherwise.
    fn derive(status: Option<&str>, checkpoints: &[Checkpoint]) -> Vec<NodeView> {
        derive_with(status, checkpoints, None, Some(&standard_shape()))
    }

    fn cp(node: &str, at: &str) -> Checkpoint {
        Checkpoint {
            node_name: node.to_string(),
            output_json: "{}".to_string(),
            created_at: at.to_string(),
            ..Default::default()
        }
    }

    /// A config that gives one node somewhere to run. A node with no route here never runs, which
    /// is what makes the verifier's presence in a run a question the config answers.
    fn routed(node: &str) -> ratatoskr_core::RatatoskrConfig {
        let mut config = ratatoskr_core::RatatoskrConfig::default();
        config.models.insert(
            node.to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".into(),
                model: "claude-sonnet-5".into(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: ratatoskr_core::SessionScope::Reuse,
            },
        );
        config
    }

    /// A clarification exchange as `clarify.rs` writes one: all four fields, every value a string.
    const EXCHANGE: &str =
        r#"{"from":"analyst","to":"scout","question":"which one?","answer":"the first"}"#;

    fn state_of(views: &[NodeView], name: &str) -> NodeState {
        views.iter().find(|v| v.name == name).unwrap().state
    }

    fn caller_of_view(views: &[NodeView], name: &str) -> Option<String> {
        views
            .iter()
            .find(|v| v.name == name)
            .unwrap()
            .caller
            .clone()
    }

    /// A checkpoint whose output records who asked for it, as `clarify.rs` writes it.
    fn cp_from(node: &str, at: &str, output_json: &str) -> Checkpoint {
        Checkpoint {
            output_json: output_json.to_string(),
            ..cp(node, at)
        }
    }

    #[test]
    fn a_run_that_recorded_no_shape_is_still_drawn_from_its_records() {
        // A workflow that declares no layout records none, and there is no compiled-in pipeline to
        // stand in for it. Every node the run has evidence for is still placed, in the order it
        // first ran — the most that can be said when nothing declared where they sat.
        let views = derive_with(
            Some("planned"),
            &[cp("gather", "t1"), cp("decide", "t2"), cp("gather", "t3")],
            None,
            None,
        );
        assert_eq!(
            views.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
            ["gather", "decide"]
        );
        assert_eq!(views[0].stage, 0);
        assert_eq!(views[1].stage, 1);
        assert_eq!(
            views[0].checkpoints, 2,
            "one row aggregates a node's records"
        );
        assert_eq!(state_of(&views, "decide"), NodeState::Done);
    }

    #[test]
    fn a_live_run_marks_the_next_uncheckpointed_node_working() {
        let views = derive(Some("running"), &[cp("context", "t1")]);
        assert_eq!(state_of(&views, "context"), NodeState::Done);
        assert_eq!(state_of(&views, "analyst"), NodeState::Working);
        // Nothing downstream of where the run sits is claimed to be doing anything.
        assert_eq!(state_of(&views, "implementer"), NodeState::Idle);
        assert_eq!(state_of(&views, "bookkeeper"), NodeState::Idle);
        // And the overseer, which this run never ran, reads as skipped rather than pending —
        // something after it checkpointed, so the run is past it either way.
        assert_eq!(state_of(&views, "overseer"), NodeState::Idle);
    }

    #[test]
    fn a_failed_run_marks_the_node_it_died_on() {
        let views = derive(Some("failed"), &[cp("context", "t1")]);
        assert_eq!(state_of(&views, "context"), NodeState::Done);
        assert_eq!(state_of(&views, "analyst"), NodeState::Failed);
        // A failure doesn't retroactively implicate nodes that never got their turn.
        assert_eq!(state_of(&views, "implementer"), NodeState::Idle);
    }

    #[test]
    fn the_implementer_is_still_working_while_converge_iterates() {
        // Converge writes one checkpoint per iteration, so "has a checkpoint" is not "finished".
        let live = derive(
            Some("running"),
            &[
                cp("context", "t1"),
                cp("analyst", "t3"),
                cp("red_team", "t4"),
                cp("implementer", "t5"),
                cp("implementer", "t6"),
            ],
        );
        assert_eq!(state_of(&live, "red_team"), NodeState::Done);
        assert_eq!(state_of(&live, "implementer"), NodeState::Working);
        // The converge loop must not advance the pipeline past the fork: the bookkeeper only
        // runs after a terminal status, so claiming it's working would be structurally impossible.
        assert_eq!(state_of(&live, "bookkeeper"), NodeState::Idle);
        assert_eq!(
            live.iter()
                .find(|v| v.name == "implementer")
                .unwrap()
                .checkpoints,
            2
        );

        // Once the run reaches a terminal status the same checkpoints mean it's finished.
        let done = derive(
            Some("converged"),
            &[cp("red_team", "t4"), cp("implementer", "t5")],
        );
        assert_eq!(state_of(&done, "implementer"), NodeState::Done);
    }

    #[test]
    fn a_finished_plan_run_leaves_the_fork_idle_not_pending() {
        // `planned` is terminal, so the fork nodes a `plan` run never reaches aren't shown as
        // work about to happen. This is only unambiguous because a full run records `running`
        // for its fork phase rather than staying `planned`.
        let views = derive(Some("planned"), &[cp("context", "t1"), cp("analyst", "t3")]);
        assert_eq!(state_of(&views, "analyst"), NodeState::Done);
        assert_eq!(state_of(&views, "red_team"), NodeState::Idle);
        assert_eq!(state_of(&views, "implementer"), NodeState::Idle);
    }

    #[test]
    fn a_failure_during_converge_lands_on_the_implementer_not_the_bookkeeper() {
        // Converge died on a later iteration, so the implementer has checkpoints from earlier
        // ones. The only step after the fork is bookkeeping, which can't fail a run — so this
        // failure is the implementer's.
        let views = derive(
            Some("failed"),
            &[
                cp("context", "t1"),
                cp("analyst", "t3"),
                cp("red_team", "t4"),
                cp("implementer", "t5"),
            ],
        );
        assert_eq!(state_of(&views, "implementer"), NodeState::Failed);
        assert_eq!(state_of(&views, "red_team"), NodeState::Done);
        assert_eq!(state_of(&views, "bookkeeper"), NodeState::Idle);
    }

    #[test]
    fn a_failed_run_whose_verifier_had_a_route_is_not_pinned_on_the_implementer() {
        // The same records as the converge-death case above, from a run that routed its verifier.
        // `verify()` degrades a review it could not run, but it still returns an error for an
        // analyst argument that no longer matches its checkpoint, an overlap with `iterate()`, or a
        // store read that failed — and those fail the run. Which of the two died is not in the
        // record: the last checkpoint is the implementer's either way, because `verify()` runs
        // directly after the attempt it reviews. So nothing is attributed.
        let config = routed("verifier");
        let views = derive_with(
            Some("failed"),
            &[
                cp("context", "t1"),
                cp("analyst", "t3"),
                cp("red_team", "t4"),
                cp("implementer", "t5"),
            ],
            Some(&config),
            Some(&standard_shape()),
        );
        assert_eq!(state_of(&views, "implementer"), NodeState::Done);
        assert!(
            !views.iter().any(|v| v.state == NodeState::Failed),
            "an unattributed failure, not one pinned on whichever node is convenient"
        );
    }

    #[test]
    fn a_routed_verifier_after_a_fork_that_ran_is_not_where_a_failed_run_stopped_either() {
        // A column the layout does not mark optional says every run reaches it, and a routed
        // verifier can fail a run. It still may not be reported as the failure: the implementer
        // re-enters on every converge iteration, so an attempt that died leaves this same record —
        // the implementer's last checkpoint and no verifier one. Reading `failed` off the box that
        // happens to be next is the mirror of pinning it on the implementer, and no more true.
        let shape = shape_of(&[
            (&["red_team", "implementer"], false),
            (&["verifier"], false),
            (&["bookkeeper"], false),
        ]);
        let config = routed("verifier");
        let views = derive_with(
            Some("failed"),
            &[cp("red_team", "t1"), cp("implementer", "t2")],
            Some(&config),
            Some(&shape),
        );
        assert_eq!(state_of(&views, "verifier"), NodeState::Idle);
        assert_eq!(state_of(&views, "implementer"), NodeState::Done);
        assert!(
            !views.iter().any(|v| v.state == NodeState::Failed),
            "neither half of an ambiguous failure is named"
        );

        // Unrouted, the same run reaches a `verify()` that answers "not configured" before it does
        // anything of its own, so the fork is still the only thing that can have died — evidence
        // that does point somewhere is still reported.
        let unrouted = derive_with(
            Some("failed"),
            &[cp("red_team", "t1"), cp("implementer", "t2")],
            None,
            Some(&shape),
        );
        assert_eq!(state_of(&unrouted, "verifier"), NodeState::Idle);
        assert_eq!(state_of(&unrouted, "implementer"), NodeState::Failed);
    }

    #[test]
    fn a_failed_run_with_a_fallible_stage_after_the_fork_blames_neither_of_them() {
        // Blaming the implementer for any failed run that reached the fork is only sound where
        // nothing after it can fail a run. A declared layout may put an ordinary stage there — its
        // host error propagates out of the workflow and fails the run — but so does a later
        // `iterate()` attempt, and that writes no checkpoint either. A reader of this graph must
        // not be shown a stage that never started as the run's failure.
        let shape = shape_of(&[
            (&["red_team", "implementer"], false),
            (&["deploy"], false),
            (&["bookkeeper"], false),
        ]);
        let views = derive_with(
            Some("failed"),
            &[cp("red_team", "t1"), cp("implementer", "t2")],
            None,
            Some(&shape),
        );
        assert_eq!(state_of(&views, "implementer"), NodeState::Done);
        assert_eq!(state_of(&views, "red_team"), NodeState::Done);
        assert_eq!(state_of(&views, "deploy"), NodeState::Idle);
        assert!(
            !views.iter().any(|v| v.state == NodeState::Failed),
            "an unattributed failure, not one pinned on the stage that happens to be next"
        );
    }

    #[test]
    fn a_failed_run_that_never_reached_the_fork_still_names_the_node_it_died_on() {
        // The ambiguity above is the fork's: it comes from the implementer re-entering without a
        // record. Before the fork there is nothing to re-enter, so the stage the run stopped at is
        // the only candidate and is still reported — withholding that would lose the attribution
        // the graph exists to show.
        let shape = shape_of(&[
            (&["analyst"], false),
            (&["implementer"], false),
            (&["deploy"], false),
        ]);
        let views = derive_with(Some("failed"), &[cp("analyst", "t1")], None, Some(&shape));
        assert_eq!(state_of(&views, "analyst"), NodeState::Done);
        assert_eq!(state_of(&views, "implementer"), NodeState::Failed);
        assert_eq!(state_of(&views, "deploy"), NodeState::Idle);
    }

    #[test]
    fn the_bookkeeper_is_never_blamed_for_a_failed_run() {
        // Even with the whole fork complete, a `failed` status can't have come from bookkeeping.
        let views = derive(
            Some("failed"),
            &[
                cp("context", "t1"),
                cp("analyst", "t3"),
                cp("red_team", "t4"),
                cp("implementer", "t5"),
            ],
        );
        assert_ne!(state_of(&views, "bookkeeper"), NodeState::Failed);
    }

    #[test]
    fn a_converged_run_that_has_not_bookkept_yet_is_idle_not_working() {
        // Bookkeeping runs after the terminal write and its failure is only logged, so this is
        // ambiguous (in flight / silently failed / never applicable) — don't guess.
        let views = derive(
            Some("converged"),
            &[cp("red_team", "t4"), cp("implementer", "t5")],
        );
        assert_eq!(state_of(&views, "implementer"), NodeState::Done);
        assert_eq!(state_of(&views, "bookkeeper"), NodeState::Idle);
    }

    #[test]
    fn awaiting_clarification_counts_as_live() {
        // A blocked node is still that run's active node, not an idle one.
        let views = derive(Some("awaiting_clarification"), &[cp("context", "t1")]);
        assert_eq!(state_of(&views, "analyst"), NodeState::Working);
    }

    #[test]
    fn a_run_with_no_status_row_is_still_derivable() {
        // The scripted path writes the issue checkpoint before the runs row exists.
        let views = derive(None, &[cp("context", "t1")]);
        assert_eq!(state_of(&views, "context"), NodeState::Done);
        assert_eq!(state_of(&views, "analyst"), NodeState::Working);
    }

    #[test]
    fn a_node_says_what_it_will_run_on_before_it_runs() {
        // Otherwise the pipeline is blank until a node finishes, which is the wrong way round: a
        // reader wants to know what is about to happen, not only what already did.
        let config = routed("redteam");

        let views = derive_with(None, &[], Some(&config), Some(&standard_shape()));
        // Routed as `redteam`, checkpointed as `red_team` — the view is keyed by the latter.
        let planned = views
            .iter()
            .find(|v| v.name == "red_team")
            .and_then(|v| v.planned.as_ref())
            .expect("a routed node says what it will run on");
        assert_eq!(planned.model, "anthropic/claude-sonnet-5");
        assert!(planned.reuses_session);
        assert_eq!(planned.session, ratatoskr_core::SessionScope::Reuse);
        assert!(planned.thinking, "nothing disabled it");

        // A node with no route never runs, and claims nothing.
        assert!(
            views
                .iter()
                .find(|v| v.name == "publisher")
                .unwrap()
                .planned
                .is_none()
        );
    }

    #[test]
    fn a_node_reports_what_it_ran_on_and_what_it_could_reach() {
        // The two facts a reader cannot get from anywhere else: the tools it could call and
        // whether it kept its memory across attempts. Both are properties of the run, and the
        // config that produced a past run may no longer exist.
        let mut ran = cp("analyst", "t1");
        ran.telemetry = ratatoskr_core::NodeTelemetry {
            model: Some("anthropic/claude-opus-4-8".into()),
            turns: Some(12),
            usage: ratatoskr_core::TokenUsage {
                input_tokens: 30,
                output_tokens: 106,
                cached_input_tokens: 132_771,
                reasoning_tokens: 4_000,
                ..Default::default()
            },
            tools: vec!["Read".into(), "semantic_search".into()],
            tools_used: vec!["Read".into()],
            reuses_session: true,
            thinking: true,
            ..Default::default()
        };
        let views = derive(Some("running"), &[ran]);
        let t = views
            .iter()
            .find(|v| v.name == "analyst")
            .and_then(|v| v.telemetry.as_ref())
            .expect("the analyst ran a model");
        assert_eq!(t.turns, Some(12));
        assert_eq!(t.cached_input_tokens, 132_771);
        assert!(t.reuses_session, "its memory carried over");
        assert_eq!(t.reasoning_tokens, 4_000, "and it thought before answering");
        assert!(t.thinking, "which the route left it free to do");
        assert_eq!(t.tools, ["Read", "semantic_search"]);
        assert_eq!(t.tools_used, ["Read"], "given two, reached for one");

        // A node that ran no model reports none rather than a row of zeroes.
        assert!(
            derive(Some("running"), &[cp("analyst", "t1")])
                .iter()
                .find(|v| v.name == "analyst")
                .unwrap()
                .telemetry
                .is_none()
        );
    }

    #[test]
    fn a_skipped_optional_stage_does_not_hold_the_pipeline_behind_it() {
        // The overseer and the verifier only run where they are configured. A run that skipped
        // both still reached the bookkeeper, and reading the stage list literally would report
        // every node after the first gap as never reached — on a run that has finished.
        let views = derive(
            Some("converged"),
            &[
                cp("context", "t1"),
                cp("analyst", "t2"),
                cp("red_team", "t3"),
                cp("implementer", "t4"),
                cp("bookkeeper", "t5"),
            ],
        );
        assert_eq!(state_of(&views, "overseer"), NodeState::Idle);
        assert_eq!(state_of(&views, "verifier"), NodeState::Idle);
        assert_eq!(state_of(&views, "implementer"), NodeState::Done);
        assert_eq!(state_of(&views, "bookkeeper"), NodeState::Done);
        // Nothing is claimed to have failed, and nothing is left looking live.
        assert!(!views.iter().any(|v| v.state == NodeState::Failed));
        assert!(!views.iter().any(|v| v.state == NodeState::Working));
    }

    #[test]
    fn the_issue_pseudo_node_is_not_part_of_the_pipeline() {
        let views = derive(Some("running"), &[cp(ISSUE_NODE, "t0")]);
        assert!(!views.iter().any(|v| v.name == ISSUE_NODE));
        // ...and it doesn't advance the pipeline: gathering context is still the working node.
        // Not the overseer, even though it comes first — whether a run has one is config the
        // store never sees, so it is not something to report a run as busy doing.
        assert_eq!(state_of(&views, "context"), NodeState::Working);
        assert_eq!(state_of(&views, "overseer"), NodeState::Idle);
    }

    #[test]
    fn every_run_status_is_classified() {
        use strum::IntoEnumIterator as _;

        // `is_terminal` matches on the persisted strings, so the compiler cannot flag a new
        // `RunStatus` variant that belongs in the list. A status missing from both sets leaves a
        // finished run showing as executing forever, which is why every variant must be named here
        // deliberately rather than falling through to a default.
        let in_flight = ["pending", "running", "awaiting_clarification"];
        for status in ratatoskr_core::RunStatus::iter() {
            let s = status.as_str();
            assert_ne!(
                is_terminal(Some(s)),
                in_flight.contains(&s),
                "`{s}` is either terminal or in flight — classify it in one and only one"
            );
        }
    }

    #[test]
    fn a_run_that_needed_no_code_change_is_finished() {
        // The fork never ran, so there is no implementer checkpoint. That must read as "done",
        // not as "still working on the fork".
        assert!(is_terminal(Some("no_code_change")));
        let views = derive(Some("no_code_change"), &[cp("analyst", "t")]);
        assert_ne!(state_of(&views, "analyst"), NodeState::Working);
        assert_ne!(state_of(&views, "implementer"), NodeState::Working);
    }

    #[test]
    fn a_referee_checkpoint_is_attributed_to_the_implementer_it_judged() {
        // The referee is excluded from the shape, so it arrives through `append_unknown` with no
        // position to explain what ran it. `referee_judgement` judges the latest implementer
        // checkpoint, so the nearest preceding one is the output it looked at.
        let views = derive(
            Some("converged"),
            &[
                cp("context", "t1"),
                cp("analyst", "t2"),
                cp("red_team", "t3"),
                cp("implementer", "t4"),
                cp("referee", "t5"),
            ],
        );
        assert_eq!(
            caller_of_view(&views, "referee").as_deref(),
            Some("implementer")
        );
    }

    #[test]
    fn only_implementer_checkpoints_before_the_referee_can_have_been_judged() {
        // The scan runs backward from the referee's own checkpoint. An implementer row written
        // afterwards is a later iteration the referee never saw, so it must not be claimed as the
        // caller — the resolved name is the same either way, which is exactly why the direction has
        // to be pinned by a case where a wrong direction changes the answer.
        let after = derive(
            Some("running"),
            &[cp("referee", "t1"), cp("implementer", "t2")],
        );
        assert_eq!(caller_of_view(&after, "referee"), None);

        // With implementer checkpoints on both sides, the ones before it resolve it.
        let both = derive(
            Some("running"),
            &[
                cp("implementer", "t1"),
                cp("implementer", "t2"),
                cp("referee", "t3"),
                cp("implementer", "t4"),
            ],
        );
        assert_eq!(
            caller_of_view(&both, "referee").as_deref(),
            Some("implementer")
        );
    }

    #[test]
    fn a_node_the_rules_were_not_written_for_claims_no_caller() {
        // `append_unknown` appends whatever a custom workflow checkpoints, so both rules are
        // dispatched on the node name rather than tried on everything. Neither signal generalises:
        // being preceded by an implementer is true of most late work, and another node's `from` may
        // mean a branch or a revision. Guessing here would put false parentage in the API, which
        // #248 draws.
        let after_implementer = derive(
            Some("converged"),
            &[
                cp("implementer", "t1"),
                cp("deploy", "t2"),
                cp("smoke_test", "t3"),
            ],
        );
        assert_eq!(caller_of_view(&after_implementer, "deploy"), None);
        assert_eq!(caller_of_view(&after_implementer, "smoke_test"), None);

        // A `from` on a foreign node means whatever that node meant by it.
        let own_from = derive(
            Some("converged"),
            &[
                cp("implementer", "t1"),
                Checkpoint {
                    node_name: "backport".to_string(),
                    output_json: r#"{"from":"release-1.2"}"#.to_string(),
                    created_at: "t2".to_string(),
                    ..Default::default()
                },
            ],
        );
        assert_eq!(caller_of_view(&own_from, "backport"), None);
    }

    #[test]
    fn a_referee_with_no_implementer_before_it_has_an_unknown_caller() {
        // Legacy or pathological data. Unknown is reported as unknown, and nothing panics.
        let views = derive(
            Some("converged"),
            &[cp("context", "t1"), cp("referee", "t2")],
        );
        assert_eq!(caller_of_view(&views, "referee"), None);
    }

    #[test]
    fn a_clarification_claims_no_caller_even_though_its_record_names_one() {
        // A clarification records `from`, and it is tempting to read it as the asking node. It is
        // not: a declared stage runs under its `governedBy` identity, and that is the value this
        // field carries — so a stage governed by `implementer` would be reported as the implementer
        // having asked itself, and two stages sharing one `governedBy` are indistinguishable here.
        // Naming the wrong asker is worse than naming none, because #248 anchors a branch on it, so
        // this stays silent until the producer records the caller per invocation (#244).
        let views = derive(
            Some("awaiting_clarification"),
            &[
                cp("implementer", "t1"),
                cp_from("clarification", "t2", EXCHANGE),
            ],
        );
        assert_eq!(caller_of_view(&views, "clarification"), None);
    }

    #[test]
    fn a_node_the_shape_places_claims_no_caller() {
        // Its position is the answer; a name there would be a second, driftable source for it.
        let views = derive(
            Some("converged"),
            &[
                cp("context", "t1"),
                cp("implementer", "t2"),
                cp("referee", "t3"),
            ],
        );
        for v in views.iter().filter(|v| v.name != "referee") {
            assert_eq!(v.caller, None, "`{}` is placed by the shape", v.name);
        }
    }

    #[test]
    fn a_run_nobody_could_review_is_finished_not_failed() {
        // The change was made and passed its tests; only the reviewer was unavailable. Reporting
        // that as still-executing would leave it spinning in the dashboard forever.
        assert!(is_terminal(Some("unreviewed")));
        let views = derive(
            Some("unreviewed"),
            &[
                cp("red_team", "t"),
                cp("implementer", "t"),
                cp("verifier", "t"),
            ],
        );
        assert_eq!(state_of(&views, "implementer"), NodeState::Done);
        assert_ne!(state_of(&views, "verifier"), NodeState::Working);
    }

    #[test]
    fn a_live_implementer_is_working_in_a_run_that_declared_no_layout() {
        // A workflow with no `layout` records an empty shape, so every node of the run — its own
        // included — is placed by `append_unknown`. `implement()` checkpoints on its first pass
        // while `iterate()` carries on, so a checkpoint there does not mean the implementer is
        // finished, exactly as it does not in a run that declared a layout.
        let checkpoints = [
            cp("issue", "t0"),
            cp("context", "t1"),
            cp("implementer", "t2"),
        ];
        let views = derive_with(Some("running"), &checkpoints, None, Some("[]"));
        assert_eq!(state_of(&views, "implementer"), NodeState::Working);
        assert_eq!(state_of(&views, "context"), NodeState::Done);

        // A finished run's nodes still read Done, including names this build knows nothing about.
        let done = derive_with(
            Some("converged"),
            &[cp("implementer", "t2"), cp("gather", "t3")],
            None,
            Some("[]"),
        );
        assert_eq!(state_of(&done, "implementer"), NodeState::Done);
        assert_eq!(state_of(&done, "gather"), NodeState::Done);
    }
}
