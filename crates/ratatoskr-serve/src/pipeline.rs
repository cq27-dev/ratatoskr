//! Deriving per-node activity from what the store actually records.
//!
//! There is no per-node status column: `runs.status` is one value for the whole run and
//! `checkpoints` is an append-only log keyed by `node_name`. Everything here is inference over
//! those two facts, and the rules below are deliberately explicit about the places the pipeline
//! is *not* uniform rather than hiding them behind a clever general rule.

use ratatoskr_store::Checkpoint;
use serde::Serialize;

/// The pipeline in execution order, one entry per stage. The fork is a single stage with two
/// nodes: `run_full` joins them, so in a built-in run both checkpoints land at the same moment.
const PIPELINE: &[&[&str]] = &[
    &["scout"],
    &["memory"],
    &["analyst"],
    &["red_team", "implementer"],
    // Optional, and `Idle` in a repo that has not given it a route — the same reading as any node
    // that never runs for this kind of run.
    &["verifier"],
    &["bookkeeper"],
];

/// The issue text is checkpointed under this name so `bookkeep` can replay a stored run. It is
/// not a node — it's the run's input, and it's the only record of the run's subject.
pub const ISSUE_NODE: &str = "issue";

/// What a node is doing, as far as the store can honestly say.
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
    /// How many checkpoints this node wrote. Only the implementer (per converge iteration) and
    /// the bookkeeper (via `ratatoskr bookkeep` replay) can exceed one.
    pub checkpoints: usize,
    pub first_at: Option<String>,
    pub last_at: Option<String>,
}

/// Statuses that mean the run is no longer executing. Note `planned` belongs here only because
/// `run_full` now records `running` for its fork+converge phase — otherwise a full run in flight
/// would be indistinguishable from a finished `plan`.
///
/// This matches `RunStatus`'s persisted strings, so the compiler cannot flag a new variant that
/// belongs here — and one missing from the list leaves a finished run showing as still executing,
/// forever. `every_run_status_is_classified` is what actually catches that.
fn is_terminal(status: Option<&str>) -> bool {
    matches!(
        status,
        Some(
            "planned"
                | "converged"
                | "max_iterations_reached"
                | "no_code_change"
                | "unreviewed"
                | "failed"
                | "abandoned"
        )
    )
}

/// Index of the fork stage (`red_team` ∥ `implementer`) in [`PIPELINE`].
const FORK: usize = 3;

/// Derive each node's state from the run status and its checkpoints.
///
/// Three non-uniformities are handled explicitly:
/// - **The implementer re-runs.** Converge checkpoints it once per iteration, so "has a
///   checkpoint" does not mean finished — while the run is live it is still converging, and the
///   fork stage is not complete no matter how many checkpoints it has.
/// - **The bookkeeper runs after the terminal status is written**, and a bookkeeping failure is
///   only logged. So it can never be the cause of a `failed` run, and it is never reported
///   `Failed`. A terminal run with no bookkeeper checkpoint is genuinely ambiguous — in flight,
///   silently failed, or never applicable — and is reported `Idle` rather than guessed at. Pair
///   it with the run's `last_activity` to judge.
/// - **A `failed` run that reached the fork died in converge.** Since the only step after the
///   fork is bookkeeping and that cannot fail the run, the implementer is where it stopped, even
///   though it has checkpoints from earlier iterations.
pub fn derive(status: Option<&str>, checkpoints: &[Checkpoint]) -> Vec<NodeView> {
    let terminal = is_terminal(status);
    let failed = status == Some("failed");
    let fork_started = PIPELINE[FORK].iter().any(|n| count(checkpoints, n) > 0);

    // A node has finished only if it checkpointed *and* isn't the implementer mid-converge —
    // otherwise the fork would look complete on iteration 1 and the run's activity would be
    // attributed to the bookkeeper, which by the invariant above hasn't started.
    let finished =
        |name: &str| count(checkpoints, name) > 0 && !(name == "implementer" && !terminal);
    let current = PIPELINE
        .iter()
        .position(|stage| stage.iter().any(|n| !finished(n)));

    let mut out = Vec::new();
    for (idx, stage) in PIPELINE.iter().enumerate() {
        for name in *stage {
            let times: Vec<&str> = checkpoints
                .iter()
                .filter(|c| c.node_name == *name)
                .map(|c| c.created_at.as_str())
                .collect();

            let state = if failed && idx == FORK && fork_started {
                // Converge died. Whatever the implementer checkpointed came from earlier
                // iterations; red-team ran to completion if it recorded anything.
                match *name {
                    "implementer" => NodeState::Failed,
                    _ if times.is_empty() => NodeState::Failed,
                    _ => NodeState::Done,
                }
            } else if times.is_empty() {
                match () {
                    // Later than where the run is: nothing to say about it yet.
                    _ if current != Some(idx) => NodeState::Idle,
                    // The bookkeeper can't fail a run — a failure here belongs upstream.
                    _ if *name == "bookkeeper" => NodeState::Idle,
                    _ if failed => NodeState::Failed,
                    _ if !terminal => NodeState::Working,
                    _ => NodeState::Idle,
                }
            } else if *name == "implementer" && !terminal {
                // Checkpointed at least once, but converge may still be iterating on it.
                NodeState::Working
            } else {
                NodeState::Done
            };

            out.push(NodeView {
                name: (*name).to_string(),
                state,
                checkpoints: times.len(),
                first_at: times.first().map(|s| s.to_string()),
                last_at: times.last().map(|s| s.to_string()),
            });
        }
    }
    out
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
        let views = derive(Some("running"), &[cp("scout", "t1")]);
        assert_eq!(state_of(&views, "scout"), NodeState::Done);
        assert_eq!(state_of(&views, "memory"), NodeState::Working);
        // Nothing downstream of where the run sits is claimed to be doing anything.
        assert_eq!(state_of(&views, "analyst"), NodeState::Idle);
        assert_eq!(state_of(&views, "bookkeeper"), NodeState::Idle);
    }

    #[test]
    fn a_failed_run_marks_the_node_it_died_on() {
        let views = derive(Some("failed"), &[cp("scout", "t1"), cp("memory", "t2")]);
        assert_eq!(state_of(&views, "memory"), NodeState::Done);
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
                cp("scout", "t1"),
                cp("memory", "t2"),
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
        let views = derive(
            Some("planned"),
            &[cp("scout", "t1"), cp("memory", "t2"), cp("analyst", "t3")],
        );
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
                cp("scout", "t1"),
                cp("memory", "t2"),
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
                cp("scout", "t1"),
                cp("memory", "t2"),
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
        let views = derive(Some("awaiting_clarification"), &[cp("scout", "t1")]);
        assert_eq!(state_of(&views, "memory"), NodeState::Working);
    }

    #[test]
    fn a_run_with_no_status_row_is_still_derivable() {
        // The scripted path writes the issue checkpoint before the runs row exists.
        let views = derive(None, &[cp("scout", "t1")]);
        assert_eq!(state_of(&views, "scout"), NodeState::Done);
        assert_eq!(state_of(&views, "memory"), NodeState::Working);
    }

    #[test]
    fn the_issue_pseudo_node_is_not_part_of_the_pipeline() {
        let views = derive(Some("running"), &[cp(ISSUE_NODE, "t0")]);
        assert!(!views.iter().any(|v| v.name == ISSUE_NODE));
        // ...and it doesn't advance the pipeline: scout is still the working node.
        assert_eq!(state_of(&views, "scout"), NodeState::Working);
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
