//! Deriving per-node activity from what the store actually records.
//!
//! There is no per-node status column: `runs.status` is one value for the whole run and
//! `checkpoints` is an append-only log keyed by `node_name`. Everything here is inference over
//! those two facts, and the rules below are deliberately explicit about the places the pipeline
//! is *not* uniform rather than hiding them behind a clever general rule.

use ratatoskr_store::Checkpoint;
use serde::Serialize;

/// Nodes that run after the terminal status is written and whose failure is only logged. They can
/// never be the reason a run failed, so they are never reported `Failed`.
const CANNOT_FAIL_THE_RUN: &[&str] = &["bookkeeper", "publisher"];

/// The issue text is checkpointed under this name so `bookkeep` can replay a stored run. It is
/// not a node — it's the run's input, and it's the only record of the run's subject.
pub const ISSUE_NODE: &str = "issue";

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
/// - **A `failed` run that reached the fork died in converge.** Since the only step after the
///   fork is bookkeeping and that cannot fail the run, the implementer is where it stopped, even
///   though it has checkpoints from earlier iterations.
pub fn derive(status: Option<&str>, checkpoints: &[Checkpoint]) -> Vec<NodeView> {
    derive_with(status, checkpoints, None, None)
}

/// [`derive`], plus the config the run was started under — so a node that has not run yet can still
/// say what it will run on.
pub fn derive_with(
    status: Option<&str>,
    checkpoints: &[Checkpoint],
    config: Option<&ratatoskr_core::RatatoskrConfig>,
    shape_json: Option<&str>,
) -> Vec<NodeView> {
    // The graph the run recorded, not the one this build happens to have. A run from another
    // machine — or from this one before the pipeline changed — is drawn against its own shape.
    let shape = ratatoskr_core::shape::recorded_or_built_in(shape_json);
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

            let state = if failed && Some(idx) == fork && fork_started {
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
                    // A failure here belongs upstream: these run past the terminal status.
                    _ if CANNOT_FAIL_THE_RUN.contains(&name.as_str()) => NodeState::Idle,
                    _ if failed => NodeState::Failed,
                    _ if !terminal => NodeState::Working,
                    _ => NodeState::Idle,
                }
            } else if name == "implementer" && !terminal {
                // Checkpointed at least once, but converge may still be iterating on it.
                NodeState::Working
            } else {
                NodeState::Done
            };

            out.push(NodeView {
                telemetry: NodeTelemetryView::latest(checkpoints, name),
                planned: PlannedNode::of(config, name),
                name: name.clone(),
                state,
                stage: idx,
                lane,
                checkpoints: times.len(),
                first_at: times.first().map(|s| s.to_string()),
                last_at: times.last().map(|s| s.to_string()),
            });
        }
    }
    append_unknown(&mut out, checkpoints, config);
    out
}

/// Add nodes the run has data for that this build's pipeline does not contain.
///
/// The shape above is compiled in, so a run of the standard pipeline renders anywhere — including
/// on an installation with no config at all, which is what an imported run has to survive. A run
/// from a DIFFERENT graph is the case this covers: a custom workflow's nodes are not in the list,
/// and without this its checkpoints would be silently dropped and the run would appear to have
/// done nothing.
///
/// They go in trailing stages, in the order they first ran. That is not the shape they executed
/// in — it cannot be recovered from checkpoints alone — but it shows every node with its output
/// and its cost, which is what someone analysing a foreign run came for.
fn append_unknown(
    out: &mut Vec<NodeView>,
    checkpoints: &[Checkpoint],
    config: Option<&ratatoskr_core::RatatoskrConfig>,
) {
    let known: std::collections::HashSet<&str> = out.iter().map(|n| n.name.as_str()).collect();
    let mut seen = std::collections::HashSet::new();
    let mut extra: Vec<&str> = Vec::new();
    for c in checkpoints {
        let name = c.node_name.as_str();
        // The issue pseudo-node writes a checkpoint and is deliberately not a pipeline node: it
        // records what the run was asked to do, which is not a stage of doing it.
        if name != ISSUE_NODE && !known.contains(name) && seen.insert(name) {
            extra.push(name);
        }
    }
    let base = out.iter().map(|n| n.stage).max().map_or(0, |s| s + 1);
    for (i, name) in extra.into_iter().enumerate() {
        let times = node_times(checkpoints, name);
        out.push(NodeView {
            telemetry: NodeTelemetryView::latest(checkpoints, name),
            planned: PlannedNode::of(config, name),
            name: name.to_string(),
            // A foreign node that wrote a checkpoint has run; nothing here can say more than that.
            state: NodeState::Done,
            stage: base + i,
            lane: 0,
            checkpoints: times.len(),
            first_at: times.first().map(|s| s.to_string()),
            last_at: times.last().map(|s| s.to_string()),
        });
    }
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

    fn cp(node: &str, at: &str) -> Checkpoint {
        Checkpoint {
            node_name: node.to_string(),
            output_json: "{}".to_string(),
            created_at: at.to_string(),
            ..Default::default()
        }
    }

    fn state_of(views: &[NodeView], name: &str) -> NodeState {
        views.iter().find(|v| v.name == name).unwrap().state
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
        let mut config = ratatoskr_core::RatatoskrConfig::default();
        config.models.insert(
            "redteam".to_string(),
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

        let views = derive_with(None, &[], Some(&config), None);
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
}
