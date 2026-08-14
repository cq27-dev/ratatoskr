//! Deriving per-node activity from what the store actually records.
//!
//! There is no per-node status column: `runs.status` is one value for the whole run and
//! `checkpoints` is an append-only log keyed by `node_name`. Everything here is inference over
//! those two facts, and the rules below are deliberately explicit about the places the pipeline
//! is *not* uniform rather than hiding them behind a clever general rule.

use ratatoskr_store::Checkpoint;
use serde::Serialize;

/// A run's checkpoints grouped by the name each was written under, in order.
///
/// Built once per derivation, for the same reason the registry is indexed: every question this file
/// asks of the log is "the rows under this name", it asks one per node and per stage, and the log is
/// a run author's document that `Store::import` writes verbatim. A scan per question is quadratic
/// work an imported run can dictate.
type Rows<'a> = std::collections::HashMap<&'a str, Vec<&'a Checkpoint>>;

fn rows_by_node(checkpoints: &[Checkpoint]) -> Rows<'_> {
    let mut rows: Rows<'_> = Rows::with_capacity(checkpoints.len());
    for c in checkpoints {
        rows.entry(c.node_name.as_str()).or_default().push(c);
    }
    rows
}

/// Whether this node's own failure can be what made a run `failed`.
///
/// A run fails when its workflow entry returns an error — any host call that returns `Err` and is
/// not caught. `bookkeeper` and `publisher` never can: they run after the terminal status is
/// written and their failure is only logged, so a run that failed did not fail in either of them.
///
/// Every other name is fallible, including a stage a workflow declared itself: its host error
/// propagates out of the script and fails the run. Nothing here reads config. A node's route decides
/// what it will run on, not whether its stage can error — a verifier reached through a ruleset, an
/// agent profile, or an overridden `governedBy` has no entry under `models` and is fallible all the
/// same.
fn can_fail_the_run(name: &str) -> bool {
    !matches!(name, "bookkeeper" | "publisher")
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
    /// What the node ran on and cost, totalled over the stages composing it — the latest record of
    /// each, folded. Absent for a node that has not checkpointed, and for one that ran no model at
    /// all.
    ///
    /// Which stages those are is NOT repeated here. It is the run's recorded registry, shipped once
    /// per run rather than once per box, because a box that has not checkpointed yet has no row in
    /// this list at all and its membership is needed exactly then.
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
    /// Every distinct route the node's stages resolve, comma-joined — the same rule
    /// `NodeTelemetry::fold` reports a folded row's models by, so what a box plans to run on and
    /// what it reports having run on read alike.
    pub model: String,
    pub thinking: bool,
    /// Every distinct scope the node's stages will run under, in registry order.
    ///
    /// A set rather than one value, and rather than nothing when they differ. A route is one field
    /// and two values of it are genuinely unsayable; a session scope is not the same question.
    /// Compacted continuation is a property a MEMBER has — it receives a summary of its own last
    /// attempt — and a box with a compacted member has it whatever its siblings do. Collapsing left
    /// a reader with only a boolean, which a compacted re-entry sets too, so the box showed the
    /// endpoint-reuse mark for a half that never touches an endpoint session.
    ///
    /// The scope each stage will RUN under, not its route's. A stage may declare its own, and
    /// execution honours the declaration — so two stages on one route can still differ.
    ///
    /// No `reuses_session` beside it: that is `sessions.contains(Reuse)`, and a second copy of one
    /// fact is what a reader falls back to when the first stops answering.
    pub sessions: Vec<ratatoskr_core::SessionScope>,
}

impl PlannedNode {
    /// What a node would run on, read from config across the stages that do its work.
    ///
    /// A route is keyed by a stage's GOVERNANCE identity, which need not be the box's name and
    /// often is not: the implementer's box runs `[models.implementer]` through
    /// `implementer_attempt`, and a stage drawn under its own id may declare `governedBy` freely.
    /// Reading the config under the box's own name reports the wrong route for the first and none
    /// at all for the second, on a node execution routes perfectly well.
    ///
    /// A box whose stages resolve DIFFERENT routes names each of them, rather than picking one.
    /// That can genuinely happen — a composed node's halves resolve through their own profiles
    /// (#277) — and it is the same choice folded telemetry already makes for the same reason:
    /// dropping the disagreement would report a route the box does not entirely have, and emptying
    /// it would read as "this node has nowhere to run".
    ///
    /// `None` when no stage of the node has a route. A node with no route never runs.
    fn of(
        config: Option<&ratatoskr_core::RatatoskrConfig>,
        stages: &[&str],
        registry: &ratatoskr_core::shape::Registry<'_>,
    ) -> Option<Self> {
        let config = config?;
        // Each stage's route, and the scope it will actually run under. `Stage::session_scope` is
        // what execution applies — the stage's own declaration wins, an absent one preserves the
        // route — so a box whose stages declared differently against ONE route still has no single
        // scope, and reading `route.session` here would report one, confidently and wrongly.
        let planned: Vec<(&ratatoskr_core::ModelRoute, ratatoskr_core::SessionScope)> = stages
            .iter()
            .filter_map(|stage| {
                let route = config.models.get(registry.governance_of(stage))?;
                Some((route, registry.session_of(stage).unwrap_or(route.session)))
            })
            .collect();
        // A vector for the output and a set to decide what goes in it. The order is meaningful — a
        // box lists every distinct route it runs on, in registry order — but asking the vector
        // whether it already holds one rescans everything collected so far, once per member, and a
        // workflow may compose a box out of as many stages as it likes.
        let mut models: Vec<String> = Vec::new();
        let mut named = std::collections::HashSet::new();
        // `sessions` needs no set. `SessionScope` has three variants, so this vector is three long
        // at worst and the scan is bounded by the enum rather than by the box.
        let mut sessions: Vec<ratatoskr_core::SessionScope> = Vec::new();
        for (route, session) in &planned {
            let model = format!("{}/{}", route.provider, route.model);
            if named.insert(model.clone()) {
                models.push(model);
            }
            if !sessions.contains(session) {
                sessions.push(*session);
            }
        }
        if models.is_empty() {
            return None;
        }
        Some(PlannedNode {
            model: models.join(", "),
            thinking: planned.iter().any(|(route, _)| thinking(route)),
            sessions,
        })
    }
}

/// Whether a route leaves the model free to reason. Configured, not observed.
fn thinking(route: &ratatoskr_core::ModelRoute) -> bool {
    route
        .params
        .as_ref()
        .and_then(|p| p.get("thinking"))
        .and_then(|t| t.get("type"))
        .and_then(|t| t.as_str())
        != Some("disabled")
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
    /// What a node cost, totalled over the stages that do its work.
    ///
    /// A node's cost is not one row. Each stage composing it records the turn it ran under its own
    /// name, and the box's own record — the aggregate its operation host writes — carries no turn
    /// at all, so reading a single canonical name reports nothing for a composed node and a
    /// perfectly ordinary number for every other. The latest row of each member, folded.
    ///
    /// Identical to reading the one row for a node that is a single stage, which is nearly all of
    /// them: the fold of one thing is that thing.
    ///
    /// `None` when no member recorded a model turn — the issue pseudo-node, one whose turn was
    /// never claimed, or a node that has not run.
    fn totalled(rows: &Rows<'_>, stages: &[&str]) -> Option<Self> {
        // `NodeTelemetry::fold` is the same arithmetic a multi-turn checkpoint is written with:
        // figures add, a figure nobody reported stays unreported, and the models and tool sets are
        // named distinctly rather than one overwriting the other. A box that ran two profiles
        // therefore names both — true of the box, while each member's own row stays true of its
        // turn.
        let t = stages
            .iter()
            .filter_map(|stage| rows.get(stage)?.last())
            .map(|c| c.telemetry.clone())
            // The same question the event contract asks, asked through the same predicate: a row an
            // operation host wrote covers no turn, and its zeros are defaults rather than figures.
            .filter(ratatoskr_core::NodeTelemetry::ran_a_model)
            .reduce(|mut folded, next| {
                folded.fold(next);
                folded
            })?;
        Some(NodeTelemetryView {
            model: t.model,
            turns: t.turns,
            input_tokens: t.usage.input_tokens,
            output_tokens: t.usage.output_tokens,
            cached_input_tokens: t.usage.cached_input_tokens,
            cache_creation_input_tokens: t.usage.cache_creation_input_tokens,
            reasoning_tokens: t.usage.reasoning_tokens,
            thinking: t.thinking,
            duration_ms: t.duration_ms,
            tools: t.tools,
            tools_used: t.tools_used,
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

/// Whether the run reached its end under its own power — see `RunStatus::ran_to_completion`.
///
/// Read the same way as [`is_terminal`], and defaulting the same direction: a status this build
/// cannot parse reads as interrupted, so nothing is claimed finished on the strength of a name
/// nobody here classified.
fn ran_to_completion(status: Option<&str>) -> bool {
    status
        .and_then(|s| s.parse::<ratatoskr_core::RunStatus>().ok())
        .is_some_and(|s| s.ran_to_completion())
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
/// - **Once the implementer has checkpointed, a `failed` run names nobody.** The implementer
///   re-enters once per converge iteration and checkpoints once per attempt, so an attempt that died
///   leaves exactly the record a later host dying leaves: the implementer's last checkpoint and
///   nothing after it. Two candidates fit — the attempt that never checkpointed, and whatever stage
///   the cursor sits at — and nothing here separates them, so neither is named. Before the
///   implementer runs there is nothing to re-enter, the stage the run stopped at is the only
///   candidate, and it is still reported.
///
/// Which node a run died in is answered far better by the event stream, where the node the host
/// killed is the one left working with no checkpoint to follow it — see `web/src/derive.ts`. That is
/// the answer the dashboard draws. This derivation is what a run whose log has rotated away has
/// left, and it says only what checkpoints can prove.
///
/// `config` is what the run was started under, so a node that has not run yet can still say what it
/// will run on.
/// [`derive_from`], for a caller holding the recording still serialized.
///
/// The parse is the largest single pass over a document an imported run brought with it, so a
/// caller that needs the recording for anything else parses once and calls `derive_from`.
pub fn derive_with(
    status: Option<&str>,
    checkpoints: &[Checkpoint],
    config: Option<&ratatoskr_core::RatatoskrConfig>,
    shape_json: Option<&str>,
) -> Vec<NodeView> {
    // The graph the run recorded, and only that. A run from another machine — or from this one
    // before the pipeline changed — is drawn against its own shape; one that recorded no layout is
    // placed entirely by `append_unknown`, from the records it has. Its membership still applies
    // there: which stages compose a node is the registry's, not the layout's.
    derive_from(
        status,
        checkpoints,
        config,
        &ratatoskr_core::shape::recorded(shape_json),
    )
}

pub fn derive_from(
    status: Option<&str>,
    checkpoints: &[Checkpoint],
    config: Option<&ratatoskr_core::RatatoskrConfig>,
    recorded: &ratatoskr_core::shape::Recorded,
) -> Vec<NodeView> {
    // Both lookups this derivation makes, indexed once: the registry it asks about every node, and
    // the log it asks about every name. Either scanned per question is quadratic over a document an
    // imported run brought with it.
    let registry = recorded.index();
    let rows = rows_by_node(checkpoints);
    let stages = stages_of(&recorded.nodes);
    let terminal = is_terminal(status);
    let completed = ran_to_completion(status);
    let failed = status == Some("failed");
    // Once the implementer has checkpointed, a failure has two candidates and the record separates
    // neither. The implementer re-enters once per converge iteration without announcing it here, so
    // an attempt that died leaves the same trace a later host dying leaves — the implementer's last
    // checkpoint and nothing after it. Reporting the stage the cursor happens to sit at is a guess
    // about whichever node is drawn next, and reporting the implementer is the same guess mirrored.
    // Neither is named; the run's own status still says it failed.
    //
    // The implementer's OWN checkpoints, not its column's. A declared fork column may hold any
    // lanes a workflow likes, and a peer that checkpointed says nothing about whether the
    // implementer ever ran.
    let unattributable = failed && count(&rows, IMPLEMENTER_NODE) > 0;

    // A node has finished only if it checkpointed *and* isn't the implementer mid-converge —
    // otherwise the fork would look complete on iteration 1 and the run's activity would be
    // attributed to the bookkeeper, which by the invariant above hasn't started.
    let finished = |name: &str| count(&rows, name) > 0 && !(name == IMPLEMENTER_NODE && !terminal);
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
        .rposition(|nodes| nodes.iter().any(|n| count(&rows, &n.name) > 0));
    let current = stages.iter().enumerate().position(|(idx, nodes)| {
        !nodes.iter().all(|n| n.optional)
            && !(nodes.iter().all(|n| finished(&n.name))
                || last_seen.is_some_and(|seen| seen > idx))
    });
    // How many nodes of the stage the run stopped at could be the one it died in. A declared column
    // may hold any lanes a workflow likes, and a failure that fits several of them is evidence about
    // none: naming them all paints boxes red that may never have run. Only a lone candidate is named.
    let candidates = current.map_or(0, |idx| {
        stages[idx]
            .iter()
            .filter(|n| count(&rows, &n.name) == 0 && can_fail_the_run(&n.name))
            .count()
    });

    let mut out = Vec::new();
    for (idx, nodes) in stages.iter().enumerate() {
        for node in nodes {
            let (lane, name) = (node.lane, &node.name);
            let times = node_times(&rows, name);
            // What its members' records say, when it has none of its own — the same question
            // `append_unknown` asks, asked through the same function.
            let by_members = match times.is_empty() {
                false => None,
                true => from_members(
                    registry
                        .members(name)
                        .iter()
                        .any(|member| *member != name && count(&rows, member) > 0),
                    terminal,
                    completed,
                ),
            };

            let state = if let Some(state) = by_members {
                state
            } else if times.is_empty() {
                match () {
                    // Later than where the run is: nothing to say about it yet.
                    _ if current != Some(idx) => NodeState::Idle,
                    // A failure here belongs upstream: delivery runs past the terminal status.
                    _ if !can_fail_the_run(name) => NodeState::Idle,
                    // Once the implementer has run, this stage and its next attempt fit the record
                    // equally well, so neither is named.
                    _ if unattributable => NodeState::Idle,
                    // Several lanes of this column fit the failure equally.
                    _ if candidates > 1 => NodeState::Idle,
                    _ if failed => NodeState::Failed,
                    _ if !terminal => NodeState::Working,
                    _ => NodeState::Idle,
                }
            } else {
                checkpointed_state(name, terminal)
            };

            let stages = registry.members(name);
            out.push(NodeView {
                telemetry: NodeTelemetryView::totalled(&rows, &stages),
                planned: PlannedNode::of(config, &stages, &registry),
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
    append_unknown(
        &mut out,
        checkpoints,
        config,
        terminal,
        completed,
        &registry,
        &rows,
    );
    out
}

/// What a box with no record of its own is doing, from the fact that its MEMBERS have records.
///
/// `None` when the records say nothing and the caller's own rules apply — either because no member
/// has recorded (the box has not started) or because the run stopped and what is missing proves
/// nothing about what ran.
///
/// One function, called from both placements, because expressing this rule twice is exactly what
/// produced the defect it now prevents: `append_unknown` was taught that a run which reached its
/// end under its own power completes a member-only box, and the placed branch kept the old answer,
/// so a laid-out pipeline drew finished work as never-started.
///
/// The rule itself: a member always writes its own row, so its presence cannot separate a workflow
/// that called the member DIRECTLY — no aggregate is ever written for that box — from an operation
/// host that died before writing one. The run's outcome separates them. Every operation host writes
/// its aggregate before returning, so on a run that completed, a missing aggregate means no host
/// ran and the member's work is the box's work, done. On a run still going the box is mid-flight.
/// On one that failed or was abandoned, nothing is claimed here.
fn from_members(members_recorded: bool, terminal: bool, completed: bool) -> Option<NodeState> {
    match (members_recorded, terminal, completed) {
        (false, ..) => None,
        (true, false, _) => Some(NodeState::Working),
        (true, true, true) => Some(NodeState::Done),
        (true, true, false) => None,
    }
}

/// What a node whose OWN record exists is doing, wherever it sits.
///
/// Its own, not a member's. A composed box's aggregate is written after its stages have run, so a
/// member's row proves the box started and nothing more — see [`append_unknown`], which is the only
/// place a box can be reached through a member and reads that case for itself.
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
///
/// A record is placed under the NODE it belongs to, not the stage that wrote it. A composed node's
/// members each write their own row, so a layout-less run of the standard workflow has rows under
/// `context_distillation`, `redteam_classifier`, `redteam_author` and `implementer_attempt` beside
/// the three aggregates their operation hosts write — one box each here would draw four strays and,
/// worse, offer each of them controls under a name the runtime never polls. The member's row folds
/// into its box, exactly as it does for a node the layout placed.
fn append_unknown(
    out: &mut Vec<NodeView>,
    checkpoints: &[Checkpoint],
    config: Option<&ratatoskr_core::RatatoskrConfig>,
    terminal: bool,
    completed: bool,
    registry: &ratatoskr_core::shape::Registry<'_>,
    rows: &Rows<'_>,
) {
    // A box the layout already placed. A member's row resolves to its box through
    // `Recorded::node_of` below, so only the box names have to be listed here.
    let known: std::collections::HashSet<&str> = out.iter().map(|n| n.name.as_str()).collect();
    let mut seen = std::collections::HashSet::new();
    // Each out-of-shape name with the position of its FIRST checkpoint, which is what its caller is
    // resolved from. One row aggregates every checkpoint of that name, so a run whose
    // `clarification` rows were asked for by different nodes cannot express all of them in one
    // `caller`. Splitting a row per caller belongs to the placement work (#248), which owns layout.
    let mut extra: Vec<(&str, usize)> = Vec::new();
    for (idx, c) in checkpoints.iter().enumerate() {
        // The box the record belongs to. A member's row is its node's, so the several rows a
        // composed node's stages write claim one position between them — the first of them.
        let name = registry.node_of(c.node_name.as_str());
        // The issue pseudo-node writes a checkpoint and is deliberately not a pipeline node: it
        // records what the run was asked to do, which is not a stage of doing it.
        if name != ISSUE_NODE && !known.contains(name) && seen.insert(name) {
            extra.push((name, idx));
        }
    }
    let base = out.iter().map(|n| n.stage).max().map_or(0, |s| s + 1);
    for (i, (name, first)) in extra.into_iter().enumerate() {
        let times = node_times(rows, name);
        let stages = registry.members(name);
        // `Done` means the box's OWN record exists. A box arrives here because something it
        // composes checkpointed, and that may be a member rather than the box: the red team's
        // classifier finishes while its author is still writing tests, and the aggregate its host
        // writes lands after both. Reading a member's row as the box's completion reports it done
        // with no checkpoints of its own, which hides its controls while it is still working.
        //
        // With only members recorded, a live run has the box mid-flight — that is what a member
        // having finished and the aggregate not having landed means.
        //
        // On a stopped run the same rows have two histories and the record does not separate them.
        // A member ALWAYS writes its own row, so their presence proves nothing on its own: a
        // workflow may call a member stage directly, whose generic host checkpoints under the stage
        // id and never writes an aggregate at all, and an operation host that died partway leaves
        // exactly the same trace. What separates them is the RUN's outcome. Every operation host
        // writes its aggregate before returning, so on a run that reached its end under its own
        // power a missing aggregate means no host ran — the members were invoked directly, and the
        // box's work is done. On one that failed or was abandoned, nothing is claimed: `Idle` is
        // this derivation's answer wherever the evidence names nobody, as it is for an
        // unattributable failure above, and a client holding the event stream answers it properly.
        // A box reaches here because something it composes recorded, so if it has no row of its
        // own its members must have one. `Idle` is what is left when the records name nobody, as it
        // is everywhere else in this derivation.
        let state = match times.is_empty() {
            false => checkpointed_state(name, terminal),
            true => from_members(true, terminal, completed).unwrap_or(NodeState::Idle),
        };
        out.push(NodeView {
            telemetry: NodeTelemetryView::totalled(rows, &stages),
            planned: PlannedNode::of(config, &stages, registry),
            caller: caller_of(checkpoints, first),
            name: name.to_string(),
            state,
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
/// * **A name in a record is not necessarily a node the graph draws.** A clarification's `from` is
///   the STAGE that asked, and a stage may compose another node rather than being one — an
///   `implementer_attempt` asking is drawn inside the implementer's box, under a name no column
///   carries.
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
fn node_times<'a>(rows: &Rows<'a>, node: &str) -> Vec<&'a str> {
    rows.get(node).map_or_else(Vec::new, |rows| {
        rows.iter().map(|c| c.created_at.as_str()).collect()
    })
}

fn count(rows: &Rows<'_>, node: &str) -> usize {
    rows.get(node).map_or(0, Vec::len)
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
        shape_with(STANDARD_COLUMNS, STANDARD_COMPOSED)
    }

    const STANDARD_COLUMNS: &[(&[&str], bool)] = &[
        (&["overseer"], true),
        (&["context"], false),
        (&["analyst"], false),
        (&["redteam", "implementer"], false),
        (&["verifier"], true),
        (&["bookkeeper", "publisher"], false),
    ];

    /// The three boxes the standard workflow composes out of stages that are not boxes of their
    /// own. Each member records its own turn; the box records the aggregate.
    const STANDARD_COMPOSED: &[(&str, &[&str])] = &[
        ("context", &["context_distillation"]),
        ("redteam", &["redteam_classifier", "redteam_author"]),
        ("implementer", &["implementer_attempt"]),
    ];

    /// A recording from its columns, each a list of lane names and whether it may be skipped. Every
    /// box is a single stage of its own name.
    fn shape_of(columns: &[(&[&str], bool)]) -> String {
        shape_with(columns, &[])
    }

    /// As [`shape_of`], with the stages composing the boxes that are made of more than themselves.
    fn shape_with(columns: &[(&[&str], bool)], composed: &[(&str, &[&str])]) -> String {
        serde_json::to_string(&ratatoskr_core::shape::Recorded {
            nodes: columns
                .iter()
                .enumerate()
                .flat_map(|(stage, (names, optional))| {
                    names.iter().enumerate().map(move |(lane, name)| {
                        ratatoskr_core::shape::ShapeNode {
                            name: (*name).to_string(),
                            stage,
                            lane,
                            optional: *optional,
                        }
                    })
                })
                .collect(),
            stages: registry_of(columns, composed),
        })
        .unwrap()
    }

    /// The registry such a run would have: every box that composes nothing is one stage of its own
    /// name governing as itself, and a composed one is its members, each governing as the box —
    /// which is what the three bundled composed nodes do, and why one `[models.redteam]` serves
    /// both red-team halves.
    fn registry_of(
        columns: &[(&[&str], bool)],
        composed: &[(&str, &[&str])],
    ) -> Vec<ratatoskr_core::shape::RunStage> {
        columns
            .iter()
            .flat_map(|(names, _)| names.iter())
            .flat_map(|name| {
                let members: Vec<String> = composed
                    .iter()
                    .find(|(box_name, _)| box_name == name)
                    .map_or_else(
                        || vec![(*name).to_string()],
                        |(_, members)| members.iter().map(|m| (*m).to_string()).collect(),
                    );
                members
                    .into_iter()
                    .map(|id| ratatoskr_core::shape::RunStage {
                        governed_by: (id != *name).then(|| (*name).to_string()),
                        id,
                        node: (*name).to_string(),
                        session: None,
                    })
            })
            .collect()
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

    /// The membership `run_detail` ships beside the nodes — the run's recorded registry, which is
    /// where the client reads a box's stages from. Not `NodeView`: a box that has not checkpointed
    /// has no row there, and that is the window a control is used in.
    fn membership(shape_json: &str, node: &str) -> Vec<String> {
        ratatoskr_core::shape::recorded(Some(shape_json))
            .index()
            .members(node)
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    fn state_of(views: &[NodeView], name: &str) -> NodeState {
        views.iter().find(|v| v.name == name).unwrap().state
    }

    fn view<'a>(views: &'a [NodeView], name: &str) -> &'a NodeView {
        views
            .iter()
            .find(|v| v.name == name)
            .unwrap_or_else(|| panic!("no `{name}` node"))
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
    fn a_run_that_laid_nothing_out_still_draws_one_box_per_node() {
        // A workflow need not declare a layout, so its run records no positions — but it composes
        // its nodes out of the same stages a laid-out run does, and membership is recorded whether
        // or not anything was placed. Without it every member of the standard workflow's three
        // composed nodes draws as a box of its own, and the dashboard offers each of those boxes a
        // Stop under a name the runtime never polls for.
        let unplaced = serde_json::to_string(&ratatoskr_core::shape::Recorded {
            nodes: Vec::new(),
            stages: registry_of(STANDARD_COLUMNS, STANDARD_COMPOSED),
        })
        .unwrap();
        let views = derive_with(
            Some("succeeded"),
            &[
                cp("context_distillation", "t1"),
                cp("context", "t2"),
                cp("analyst", "t3"),
                cp("redteam_classifier", "t4"),
                cp("redteam_author", "t5"),
                cp("redteam", "t6"),
                cp("implementer_attempt", "t7"),
                cp("implementer", "t8"),
                cp("verifier", "t9"),
            ],
            None,
            Some(&unplaced),
        );
        assert_eq!(
            views.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
            ["context", "analyst", "redteam", "implementer", "verifier"],
            "one box per node, in the order the run first recorded under each"
        );
        // And the membership the client folds the stream by comes from the same record, so it
        // holds for a box drawn here as for one the layout placed.
        assert_eq!(
            membership(&unplaced, "redteam"),
            ["redteam_classifier", "redteam_author"]
        );
        assert_eq!(
            membership(&unplaced, "implementer"),
            ["implementer_attempt"]
        );
        assert_eq!(membership(&unplaced, "analyst"), ["analyst"]);
        // The box's own record is what says how many times it ran, exactly as for a placed one:
        // the members' rows are turns inside it, not repeats of it.
        assert_eq!(view(&views, "redteam").checkpoints, 1);
    }

    #[test]
    fn a_placed_box_reads_its_members_the_way_an_unplaced_one_does() {
        // The same records, the same run, the same answer — whether or not a layout placed the box.
        // A workflow may call a member stage directly, and then no aggregate is ever written: the
        // box's work is its member's, and on a run that reached its end under its own power that
        // work is done. Deciding it only where `append_unknown` runs left a laid-out pipeline
        // drawing finished work as never-started.
        let midway = [cp("redteam_classifier", "t1")];
        let placed = derive_with(Some("converged"), &midway, None, Some(&standard_shape()));
        let unplaced = serde_json::to_string(&ratatoskr_core::shape::Recorded {
            nodes: Vec::new(),
            stages: registry_of(STANDARD_COLUMNS, STANDARD_COMPOSED),
        })
        .unwrap();
        let appended = derive_with(Some("converged"), &midway, None, Some(&unplaced));
        assert_eq!(
            state_of(&placed, "redteam"),
            NodeState::Done,
            "a placed box completed by its member reads as completed"
        );
        assert_eq!(state_of(&placed, "redteam"), state_of(&appended, "redteam"));

        // Live, both say working; stopped, neither claims anything. The rule is one rule.
        for (status, expected) in [
            (Some("running"), NodeState::Working),
            (Some("abandoned"), NodeState::Idle),
        ] {
            let placed = derive_with(status, &midway, None, Some(&standard_shape()));
            assert_eq!(
                state_of(&placed, "redteam"),
                expected,
                "a placed box on a `{status:?}` run"
            );
        }
    }

    #[test]
    fn a_box_is_done_only_once_its_own_record_exists() {
        // A box reaches `append_unknown` because SOMETHING it composes checkpointed, and that may
        // be a member: the red team's classifier finishes while its author is still writing tests,
        // and the aggregate its host writes lands after both. Reading a member's row as the box's
        // completion reports `done` with `checkpoints: 0` — a client without usable event history
        // then hides the box's controls and calls it finished while it is still working.
        let unplaced = serde_json::to_string(&ratatoskr_core::shape::Recorded {
            nodes: Vec::new(),
            stages: registry_of(STANDARD_COLUMNS, STANDARD_COMPOSED),
        })
        .unwrap();
        let midway = [cp("redteam_classifier", "t1")];
        let live = derive_with(Some("running"), &midway, None, Some(&unplaced));
        assert_eq!(
            view(&live, "redteam").state,
            NodeState::Working,
            "a member has recorded and the box has not, on a run that is still going"
        );
        assert_eq!(view(&live, "redteam").checkpoints, 0);

        // The aggregate lands and the box is done, by its own record.
        let whole = [cp("redteam_classifier", "t1"), cp("redteam", "t2")];
        assert_eq!(
            state_of(
                &derive_with(Some("running"), &whole, None, Some(&unplaced)),
                "redteam"
            ),
            NodeState::Done
        );

        // A run that RAN TO COMPLETION and left a member's row and no aggregate did finish that
        // box: a workflow may call a member stage directly, whose generic host checkpoints under
        // the stage id, and every operation host writes its aggregate before returning — so on a
        // run that completed, a missing aggregate means no host ran, not that one died.
        for status in [
            "converged",
            "planned",
            "max_iterations_reached",
            "unreviewed",
        ] {
            assert_eq!(
                state_of(
                    &derive_with(Some(status), &midway, None, Some(&unplaced)),
                    "redteam"
                ),
                NodeState::Done,
                "a `{status}` run reached the member's completion with nothing left to write"
            );
        }

        // A run that STOPPED did not. Both leave the same rows — a member's, no aggregate — and
        // only the run's own outcome separates them, so nothing is claimed for the ones that died.
        for status in ["failed", "abandoned"] {
            assert_eq!(
                state_of(
                    &derive_with(Some(status), &midway, None, Some(&unplaced)),
                    "redteam"
                ),
                NodeState::Idle,
                "a `{status}` run cannot say whether the box completed before it stopped"
            );
        }
    }

    #[test]
    fn a_large_recording_is_derived_in_work_proportional_to_its_size() {
        // `Store::import` writes `shape_json` and a run's checkpoints verbatim, so both are the run
        // author's documents and both can be large. Every question this file asks — a box's members,
        // a record's box, the rows under a name — used to be a scan of one of them, asked once per
        // node and once per record, which is quadratic work an imported recording can dictate. The
        // position bound in `shape::recorded` caps the indices in such a document; it says nothing
        // about how many rows it has.
        //
        // Sized from measurement rather than guessed, and asserted as a ceiling rather than a
        // window, which is what keeps it off the flaky list. At this size a debug build derives
        // this in about 0.2s indexed and about 370s scanning, so the bound below sits ~25x above
        // the first and ~74x below the second: neither a slow machine nor a loaded one moves the
        // answer. Re-derive if it ever looks tight — the scan curve is 4x per doubling of N and the
        // index curve is 2x, so the gap only widens.
        const N: usize = 30_000;
        let recorded = ratatoskr_core::shape::Recorded {
            nodes: (0..N)
                .map(|i| ratatoskr_core::shape::ShapeNode {
                    name: format!("n{i}"),
                    stage: i,
                    lane: 0,
                    optional: false,
                })
                .collect(),
            stages: (0..N)
                .map(|i| ratatoskr_core::shape::RunStage {
                    id: format!("s{i}"),
                    node: format!("n{i}"),
                    governed_by: None,
                    session: None,
                })
                .collect(),
        };
        let shape = serde_json::to_string(&recorded).unwrap();
        let checkpoints: Vec<Checkpoint> = (0..N).map(|i| cp(&format!("s{i}"), "t")).collect();

        let started = std::time::Instant::now();
        let views = derive_with(Some("converged"), &checkpoints, None, Some(&shape));
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "deriving {N} nodes over {N} records took {elapsed:?}, which is scan-shaped work"
        );

        // And it is derived correctly at that size, so an index that dropped or reordered entries
        // is caught here too rather than only being fast.
        assert_eq!(views.len(), N);
        let registry = recorded.index();
        assert_eq!(registry.members("n7"), ["s7"]);
        assert_eq!(registry.node_of("s7"), "n7");
        assert_eq!(recorded.stages.len(), N);
    }

    #[test]
    fn a_box_of_many_stages_is_derived_in_work_proportional_to_its_members() {
        // A DIFFERENT exposure from the wide registry above, and the one that case cannot reach: it
        // gives every box one stage and supplies no config, so `PlannedNode::of` returns before its
        // loop runs. A box's own metadata — every distinct route it will run on, and every distinct
        // session scope — is collected per member, and testing membership by rescanning what has
        // been collected so far is quadratic in the member count.
        //
        // A workflow may compose a box out of as many stages as it likes, and a recording may be
        // imported. A wide REGISTRY and a wide BOX are different exposures and both are cheap, so
        // both cases are kept.
        //
        // Sized from measurement and asserted as a ceiling, like its sibling: at this size a debug
        // build plans this in about 0.24s deduplicating with a set and about 14.5s rescanning, so
        // the bound sits ~8x above the first and ~7x below the second. Re-derive if it looks tight
        // — the rescan curve is 4x per doubling of N and the set curve is 2x, so the gap widens.
        const N: usize = 50_000;
        let recorded = ratatoskr_core::shape::Recorded {
            nodes: vec![ratatoskr_core::shape::ShapeNode {
                name: "wide".to_string(),
                stage: 0,
                lane: 0,
                optional: false,
            }],
            stages: (0..N)
                .map(|i| ratatoskr_core::shape::RunStage {
                    id: format!("s{i}"),
                    node: "wide".to_string(),
                    // Each member governs as itself, so each resolves its own route and none of
                    // them dedupes away — the worst case, and the one a composed box can reach.
                    governed_by: None,
                    session: None,
                })
                .collect(),
        };
        let shape = serde_json::to_string(&recorded).unwrap();
        let mut config = ratatoskr_core::RatatoskrConfig::default();
        for i in 0..N {
            let mut route = routed("x").models.remove("x").expect("a route");
            route.model = format!("model-{i}");
            config.models.insert(format!("s{i}"), route);
        }

        let started = std::time::Instant::now();
        let views = derive_with(
            Some("converged"),
            &[cp("wide", "t")],
            Some(&config),
            Some(&shape),
        );
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "planning a box of {N} stages took {elapsed:?}, which is rescan-shaped work"
        );

        // And it still says what it is for: every distinct route, in registry order.
        let planned = view(&views, "wide")
            .planned
            .as_ref()
            .expect("every member is routed");
        assert_eq!(planned.model.split(", ").count(), N);
        assert!(
            planned
                .model
                .starts_with("anthropic/model-0, anthropic/model-1, ")
        );
        assert_eq!(planned.sessions, [ratatoskr_core::SessionScope::Reuse]);
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
                cp("redteam", "t4"),
                cp("implementer", "t5"),
                cp("implementer", "t6"),
            ],
        );
        assert_eq!(state_of(&live, "redteam"), NodeState::Done);
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
            &[cp("redteam", "t4"), cp("implementer", "t5")],
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
        assert_eq!(state_of(&views, "redteam"), NodeState::Idle);
        assert_eq!(state_of(&views, "implementer"), NodeState::Idle);
    }

    #[test]
    fn a_failure_during_converge_names_nobody_from_checkpoints_alone() {
        // Converge died on a later iteration, so the implementer has checkpoints from earlier ones.
        // This used to be read as proof the implementer died — nothing after the fork could fail the
        // run, so it was the only thing left. It is not proof: the implementer re-enters without
        // announcing it here, so a dead attempt and a dead later host leave the same record. Which
        // node the host died under is answered from the event stream, where the dying node is the
        // one left working; these checkpoints cannot answer it and no longer pretend to.
        let views = derive(
            Some("failed"),
            &[
                cp("context", "t1"),
                cp("analyst", "t3"),
                cp("redteam", "t4"),
                cp("implementer", "t5"),
            ],
        );
        assert_eq!(state_of(&views, "redteam"), NodeState::Done);
        assert_eq!(state_of(&views, "bookkeeper"), NodeState::Idle);
        assert!(
            !views.iter().any(|v| v.state == NodeState::Failed),
            "an unattributed failure, not one pinned on whichever node is convenient"
        );
    }

    #[test]
    fn an_optional_stage_after_the_fork_is_no_more_blamable_than_a_required_one() {
        // The routing case above with the standard shape, whose verifier column is OPTIONAL. That
        // changes where the cursor sits — an empty optional stage never holds it — and must not
        // change the attribution: an optional stage that did run and died leaves the same record as
        // one that was skipped while a later host died, which is the implementer's last checkpoint
        // and nothing after it. Neither is named.
        let config = routed("verifier");
        let views = derive_with(
            Some("failed"),
            &[
                cp("context", "t1"),
                cp("analyst", "t3"),
                cp("redteam", "t4"),
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
    fn how_a_verifier_is_routed_does_not_change_who_a_failed_run_blames() {
        // Attribution used to turn on whether `config.models` held a route for the verifier: routed
        // meant the column could have failed the run and nobody was named, unrouted meant it could
        // not and the implementer was. That read the wrong fact. A verifier reached through a
        // ruleset, an agent profile, or an overridden `governedBy` has no `models` entry and would
        // have been treated as unable to fail — which fired the inference and blamed the implementer
        // for a verifier-side error. A node's route says what it runs on, never whether its stage
        // can error, so nothing here reads config and both runs answer the same.
        let shape = shape_of(&[
            (&["redteam", "implementer"], false),
            (&["verifier"], false),
            (&["bookkeeper"], false),
        ]);
        let checkpoints = [cp("redteam", "t1"), cp("implementer", "t2")];
        let config = routed("verifier");
        for config in [Some(&config), None] {
            let views = derive_with(Some("failed"), &checkpoints, config, Some(&shape));
            assert_eq!(state_of(&views, "verifier"), NodeState::Idle);
            assert_eq!(state_of(&views, "implementer"), NodeState::Done);
            assert!(
                !views.iter().any(|v| v.state == NodeState::Failed),
                "neither half of an ambiguous failure is named, however the verifier got its model"
            );
        }
    }

    #[test]
    fn a_failed_run_does_not_redden_every_lane_of_the_column_it_stopped_at() {
        // A declared column may hold any lanes a workflow likes. Two of these never checkpointed and
        // either could be the one that died — so neither is named, exactly as two candidates either
        // side of the fork are not. Marking them both used to be how the fork's own column was
        // drawn, which reddened lanes that may never have run.
        let shape = shape_of(&[(&["scribe", "auditor", "publisher"], false)]);
        let views = derive_with(Some("failed"), &[], None, Some(&shape));
        assert_eq!(state_of(&views, "scribe"), NodeState::Idle);
        assert_eq!(state_of(&views, "auditor"), NodeState::Idle);
        // Delivery is still never blamed, and is what leaves the other two as the only candidates.
        assert_eq!(state_of(&views, "publisher"), NodeState::Idle);
        assert!(!views.iter().any(|v| v.state == NodeState::Failed));

        // With one lane the run's stopping place is unambiguous again, and is still reported.
        let alone = shape_of(&[(&["scribe", "publisher"], false)]);
        let named = derive_with(Some("failed"), &[], None, Some(&alone));
        assert_eq!(state_of(&named, "scribe"), NodeState::Failed);
        assert_eq!(state_of(&named, "publisher"), NodeState::Idle);
    }

    #[test]
    fn a_failed_run_with_a_fallible_stage_after_the_fork_blames_neither_of_them() {
        // Blaming the implementer for any failed run that reached the fork is only sound where
        // nothing after it can fail a run. A declared layout may put an ordinary stage there — its
        // host error propagates out of the workflow and fails the run — but so does a later
        // `iterate()` attempt, and that writes no checkpoint either. A reader of this graph must
        // not be shown a stage that never started as the run's failure.
        let shape = shape_of(&[
            (&["redteam", "implementer"], false),
            (&["deploy"], false),
            (&["bookkeeper"], false),
        ]);
        let views = derive_with(
            Some("failed"),
            &[cp("redteam", "t1"), cp("implementer", "t2")],
            None,
            Some(&shape),
        );
        assert_eq!(state_of(&views, "implementer"), NodeState::Done);
        assert_eq!(state_of(&views, "redteam"), NodeState::Done);
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
                cp("redteam", "t4"),
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
            &[cp("redteam", "t4"), cp("implementer", "t5")],
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
        let planned = views
            .iter()
            .find(|v| v.name == "redteam")
            .and_then(|v| v.planned.as_ref())
            .expect("a routed node says what it will run on");
        // Under `redteam`, though neither stage doing the work is called that: both halves govern
        // as the box, which is what a `[models.redteam]` entry routes.
        assert_eq!(planned.model, "anthropic/claude-sonnet-5");
        assert_eq!(
            planned.sessions,
            [ratatoskr_core::SessionScope::Reuse],
            "one route and no declaration, so the box's one scope is that route's"
        );
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
    fn a_node_plans_on_the_route_its_stages_govern_under_not_the_one_its_name_would_read() {
        // A stage is drawn under its own id and routed under its governance identity, and the two
        // are independent. Reading the config under the box's name reports nothing for a stage that
        // governs as something else, on a node execution routes perfectly well.
        let recorded = ratatoskr_core::shape::Recorded {
            nodes: vec![ratatoskr_core::shape::ShapeNode {
                name: "strategist".to_string(),
                stage: 0,
                lane: 0,
                optional: false,
            }],
            stages: vec![ratatoskr_core::shape::RunStage {
                id: "strategist".to_string(),
                node: "strategist".to_string(),
                governed_by: Some("analyst".to_string()),
                session: None,
            }],
        };
        let views = derive_with(
            None,
            &[],
            Some(&routed("analyst")),
            Some(&serde_json::to_string(&recorded).unwrap()),
        );
        assert_eq!(
            view(&views, "strategist")
                .planned
                .as_ref()
                .map(|planned| planned.model.as_str()),
            Some("anthropic/claude-sonnet-5"),
            "the box runs `models.analyst`, because that is what its stage governs as"
        );
    }

    #[test]
    fn a_stage_that_declared_its_own_session_plans_on_that_and_not_on_its_routes() {
        // Execution applies `Stage::session_scope`: a stage's own declaration wins over the route's,
        // and an absent one preserves the route. Reading `route.session` alone reports the box on a
        // scope its stages will not run, and — because it is one route — reports it confidently.
        let recorded = ratatoskr_core::shape::Recorded {
            nodes: vec![ratatoskr_core::shape::ShapeNode {
                name: "redteam".to_string(),
                stage: 0,
                lane: 0,
                optional: false,
            }],
            stages: vec![
                // Both halves reach one `[models.redteam]`, and one of them declares `fresh`.
                ratatoskr_core::shape::RunStage {
                    id: "redteam_classifier".to_string(),
                    node: "redteam".to_string(),
                    governed_by: Some("redteam".to_string()),
                    session: None,
                },
                ratatoskr_core::shape::RunStage {
                    id: "redteam_author".to_string(),
                    node: "redteam".to_string(),
                    governed_by: Some("redteam".to_string()),
                    session: Some(ratatoskr_core::SessionScope::Compacted),
                },
            ],
        };
        let views = derive_with(
            None,
            &[],
            Some(&routed("redteam")),
            Some(&serde_json::to_string(&recorded).unwrap()),
        );
        let planned = view(&views, "redteam")
            .planned
            .as_ref()
            .expect("one route serves both halves");
        // One route, so one model — the disagreement is in what the stages declared, not in where
        // they run.
        assert_eq!(planned.model, "anthropic/claude-sonnet-5");
        // Both scopes, not neither. A route is one field and two values of it collapse to nothing
        // sayable; a session scope is not that question — compacted is a property a MEMBER has, and
        // a box with a compacted member has it whatever its siblings do. Collapsing left the client
        // reading `reuses_session`, which a compacted re-entry also sets, so the box drew the
        // endpoint-reuse mark for a half that never reuses an endpoint.
        assert_eq!(
            planned.sessions,
            [
                ratatoskr_core::SessionScope::Reuse,
                ratatoskr_core::SessionScope::Compacted
            ],
            "each half's own scope, in registry order"
        );

        // And a lone stage's declaration is the box's, rather than being overwritten by the route.
        let fresh = ratatoskr_core::shape::Recorded {
            stages: vec![ratatoskr_core::shape::RunStage {
                id: "redteam_classifier".to_string(),
                node: "redteam".to_string(),
                governed_by: Some("redteam".to_string()),
                session: Some(ratatoskr_core::SessionScope::Fresh),
            }],
            ..recorded
        };
        let views = derive_with(
            None,
            &[],
            Some(&routed("redteam")),
            Some(&serde_json::to_string(&fresh).unwrap()),
        );
        let planned = view(&views, "redteam").planned.as_ref().unwrap();
        assert_eq!(
            planned.sessions,
            [ratatoskr_core::SessionScope::Fresh],
            "it declared itself out of the route's reuse"
        );
    }

    #[test]
    fn a_box_whose_stages_route_differently_names_every_route_it_would_run_on() {
        // A composed node's halves resolve through their own profiles, so they genuinely can differ
        // (#277). Naming one of them reports a route the box does not entirely have and naming none
        // reads as "this node has nowhere to run" — the same reason folded telemetry names every
        // model it covers.
        let mut config = routed("redteam_classifier");
        config.models.insert(
            "redteam_author".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "anthropic".into(),
                model: "claude-haiku-5".into(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: Some(toml::Value::Table(
                    "thinking = { type = \"disabled\" }"
                        .parse::<toml::Table>()
                        .unwrap(),
                )),
                session: ratatoskr_core::SessionScope::Fresh,
            },
        );
        // Both halves governing as themselves, which is what a workflow gets by declaring `node`
        // without `governedBy` — the split #277 established can carry two routes.
        let recorded = ratatoskr_core::shape::Recorded {
            nodes: vec![ratatoskr_core::shape::ShapeNode {
                name: "redteam".to_string(),
                stage: 0,
                lane: 0,
                optional: false,
            }],
            stages: ["redteam_classifier", "redteam_author"]
                .map(|id| ratatoskr_core::shape::RunStage {
                    id: id.to_string(),
                    node: "redteam".to_string(),
                    governed_by: None,
                    session: None,
                })
                .to_vec(),
        };
        let views = derive_with(
            None,
            &[],
            Some(&config),
            Some(&serde_json::to_string(&recorded).unwrap()),
        );
        let planned = view(&views, "redteam")
            .planned
            .as_ref()
            .expect("both halves are routed");
        assert_eq!(
            planned.model,
            "anthropic/claude-sonnet-5, anthropic/claude-haiku-5"
        );
        // The two facts a reader needs stay answerable across the disagreement: one half reasons,
        // one half carries its context. The session scope does not, so it is absent rather than
        // asserted, and a reader falls back to `reuses_session`.
        assert!(planned.thinking);
        assert_eq!(
            planned.sessions,
            [
                ratatoskr_core::SessionScope::Reuse,
                ratatoskr_core::SessionScope::Fresh
            ]
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
    fn a_box_costs_what_its_stages_cost_between_them() {
        // A composed node's own record carries no turn — the operation host writes it after the
        // stages have run — so reading one canonical name reports nothing at all for the red team.
        // The cost is on its members' rows, one per turn, and the box is their total.
        let spent = |node: &str, at: &str, model: &str, input: u64, tool: &str| {
            let mut c = cp(node, at);
            c.telemetry = ratatoskr_core::NodeTelemetry {
                model: Some(model.into()),
                turns: Some(1),
                usage: ratatoskr_core::TokenUsage {
                    input_tokens: input,
                    ..Default::default()
                },
                tools: vec![tool.into()],
                ..Default::default()
            };
            c
        };
        let views = derive(
            Some("converged"),
            &[
                spent("redteam_classifier", "t1", "anthropic/reasoner", 10, "Read"),
                spent("redteam_author", "t2", "anthropic/builder", 20, "Write"),
                cp("redteam", "t3"),
            ],
        );
        let t = view(&views, "redteam")
            .telemetry
            .as_ref()
            .expect("the red team ran two model turns");
        assert_eq!(
            t.input_tokens, 30,
            "what the box cost is what its halves did"
        );
        assert_eq!(t.turns, Some(2));
        // Both routes are named, because both ran. Which turn ran on which is answered by the
        // members' own rows, and that is why they exist: a box cannot say it honestly.
        assert_eq!(
            t.model.as_deref(),
            Some("anthropic/reasoner, anthropic/builder")
        );
        assert_eq!(t.tools, ["Read", "Write"]);
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
                cp("redteam", "t3"),
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
            // Completing is a strictly narrower thing than stopping, and a status that claimed to
            // have run to completion while still in flight would report finished boxes mid-run.
            assert!(
                !ran_to_completion(Some(s)) || is_terminal(Some(s)),
                "`{s}` claims to have run to completion without being terminal"
            );
        }
        // A status this build cannot parse is neither, so nothing is claimed finished on the
        // strength of a name nobody here classified.
        assert!(!is_terminal(Some("from_a_newer_build")));
        assert!(!ran_to_completion(Some("from_a_newer_build")));
        assert!(!ran_to_completion(None));
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
    fn a_boxs_members_are_drawn_inside_it_rather_than_beside_it() {
        // The regression this membership exists to prevent, and it is name-matched, so it fails
        // quietly: the red team's halves keep their own identities now, so each writes a per-turn
        // row under a name no column carries. Without membership `append_unknown` would tack
        // `redteam_classifier` and `redteam_author` on as two floating boxes next to the red team.
        let views = derive(
            Some("converged"),
            &[
                cp("context_distillation", "t1"),
                cp("context", "t2"),
                cp("analyst", "t3"),
                cp("redteam_classifier", "t4"),
                cp("redteam_author", "t5"),
                cp("redteam", "t6"),
                cp("implementer_attempt", "t7"),
                cp("implementer", "t8"),
            ],
        );
        for member in [
            "redteam_classifier",
            "redteam_author",
            "implementer_attempt",
            "context_distillation",
        ] {
            assert!(
                !views.iter().any(|view| view.name == member),
                "`{member}` is the red team's or the implementer's work, not a box of its own"
            );
        }
        assert!(views.iter().all(|view| view.shaped), "{views:?}");
        assert_eq!(
            membership(&standard_shape(), "redteam"),
            ["redteam_classifier", "redteam_author"]
        );
        // A node that is one stage says so the same way, so a reader never special-cases.
        assert_eq!(membership(&standard_shape(), "analyst"), ["analyst"]);
    }

    #[test]
    fn a_recording_this_build_cannot_read_places_nothing_and_infers_nothing() {
        // Not a fallback: there is one recorded format and anything else is unreadable. The run is
        // then drawn from its own records, as any unplaced run is, and every node is exactly its
        // own name — which is all a reader with no registry can say.
        let views = derive_with(
            Some("converged"),
            &[cp("analyst", "t")],
            None,
            Some(r#"[{"name":"analyst","stage":0,"lane":0,"optional":false}]"#),
        );
        assert_eq!(view(&views, "analyst").state, NodeState::Done);
        assert!(!view(&views, "analyst").shaped, "nothing placed it");
        assert_eq!(membership("[]", "analyst"), ["analyst"]);
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
                cp("redteam", "t3"),
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
        // A shape the producer does not emit, but the reader still has to survive: unknown is
        // reported as unknown, and nothing panics.
        let views = derive(
            Some("converged"),
            &[cp("context", "t1"), cp("referee", "t2")],
        );
        assert_eq!(caller_of_view(&views, "referee"), None);
    }

    #[test]
    fn a_clarification_claims_no_caller_even_though_its_record_names_one() {
        // A clarification records `from`, and it is tempting to read it as the asking node. It
        // names the STAGE that asked, which is not the same thing: a stage that composes another
        // node is drawn inside that node's box, so `implementer_attempt` asking would be reported
        // as a node no column names. Naming the wrong asker is worse than naming none, because #248
        // anchors a branch on it, so this stays silent until the producer records the caller per
        // invocation (#244).
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
                cp("redteam", "t"),
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
