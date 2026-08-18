//! The run state that flows through the graph and round-trips through the checkpoint store.
//!
//! Per-node payloads are kept as untyped [`serde_json::Value`] in Phase 0: the nodes that
//! produce them don't exist yet, so each node crate will define its own `JsonSchema`-deriving
//! struct and validate at the handoff boundary once it's actually built (Phase 2 onward).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Why a checkpoint exists, when the ordinary path is not the whole answer.
///
/// A run's records say WHAT each node produced; a few of them are produced by a path Rust drove for
/// a reason the node names cannot carry. The ceiling recovery is the case this exists for: it
/// revises the plan and runs one more attempt, writing an `analyst` row and an `implementer` row
/// that are indistinguishable from a mid-loop replan and the retry that follows it — while meaning
/// the opposite thing. One is the loop working, the other is the run giving up gracefully.
///
/// Only paths RUST owns get a cause. A workflow calling the analyst again with a previous plan is
/// the script's decision, and stating a reason for it would be a claim the record cannot support —
/// such a row carries no cause, and its input says what it was given.
///
/// Persists (strum) and serializes (serde) as the same snake_case token, so the stored column, the
/// live event and any later analysis read one vocabulary.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    JsonSchema,
    strum::IntoStaticStr,
    strum::Display,
    strum::EnumString,
    strum::EnumIter,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum RecordCause {
    /// The one bounded recovery a run gets after its iteration budget is spent: an analyst revision
    /// and a final implementer attempt, both driven by Rust rather than asked for by the workflow.
    CeilingRecovery,
}

impl RecordCause {
    /// The stable token this persists as.
    pub fn as_str(&self) -> &'static str {
        self.into()
    }
}

/// Lifecycle status of a run. Serializes (serde) and persists (strum, in the store's `status`
/// column) as the same snake_case string — `strum::AsRefStr`/`EnumString` give the string ⇄ enum
/// mapping, kept in one place with the variants.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    JsonSchema,
    strum::IntoStaticStr,
    strum::Display,
    strum::EnumString,
    strum::EnumIter,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum RunStatus {
    Pending,
    Running,
    AwaitingClarification,
    /// The linear scout → memory → analyst planning flow finished successfully (Phase 2).
    /// Distinct from `Converged`, which is reserved for real done-criteria.
    Planned,
    /// The fork+converge loop reached real success: no newly-introduced test failures (Phase 3).
    Converged,
    /// Converge ran out of its `max_iterations` budget with a legible residual failure set —
    /// distinct from `Failed` (an error): the loop worked, it just didn't finish in budget.
    MaxIterationsReached,
    /// The change passed its acceptance run, and the verifier could not be asked whether it was
    /// the right change.
    ///
    /// Not `Converged`: that means the change held up, and since the review gate exists a reader is
    /// entitled to read it as including review. Not `Failed` either — the work was done and it
    /// passed. A verifier *error* is evidence about our infrastructure and says nothing about the
    /// change, so it must neither block the run nor be quietly reported as a clean review.
    Unreviewed,
    /// The analyst judged that carrying out the plan means changing no code in this repository —
    /// research, a review, an architecture answer — so the fork never ran.
    ///
    /// Its own status rather than `Converged`: that one means "the implementer's change held up
    /// against the baseline", and reporting it for a run that produced no change describes a
    /// success nobody had. Terminal and not a failure; the run's artifact is its plan.
    NoCodeChange,
    Failed,
    Abandoned,
}

impl RunStatus {
    /// The persisted string form (delegates to `strum::IntoStaticStr`).
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }

    /// Whether the run is no longer executing.
    ///
    /// An exhaustive match rather than a list of strings, so a new variant cannot be added without
    /// classifying it. The failure it prevents is silent and permanent in both directions: an
    /// unclassified terminal status leaves a finished run reading as still executing forever, and
    /// a live one classified as terminal invites a second process onto the same worktree.
    ///
    /// `Planned` is terminal because a full run records `Running` for its fork and converge phase;
    /// without that, a finished `plan` would be indistinguishable from a run still in flight.
    pub fn is_terminal(&self) -> bool {
        match self {
            RunStatus::Pending | RunStatus::Running | RunStatus::AwaitingClarification => false,
            RunStatus::Planned
            | RunStatus::Converged
            | RunStatus::MaxIterationsReached
            | RunStatus::Unreviewed
            | RunStatus::NoCodeChange
            | RunStatus::Failed
            | RunStatus::Abandoned => true,
        }
    }

    /// Whether the run reached its end under its own power, rather than stopping partway.
    ///
    /// Not "succeeded". `MaxIterationsReached` spent its budget and `Unreviewed` could not reach a
    /// verifier, and both are outcomes the orchestration produced deliberately: every host it
    /// entered finished and wrote what it writes. `Failed` and `Abandoned` are the two that stop
    /// mid-flight, and after either, what is missing from the record proves nothing about what ran.
    ///
    /// A reader uses it for exactly that: to tell an absent record that was never going to be
    /// written from one whose writer died. An exhaustive match, so a new variant has to be
    /// classified rather than silently reading as interrupted.
    pub fn ran_to_completion(&self) -> bool {
        match self {
            RunStatus::Planned
            | RunStatus::Converged
            | RunStatus::MaxIterationsReached
            | RunStatus::Unreviewed
            | RunStatus::NoCodeChange => true,
            RunStatus::Pending
            | RunStatus::Running
            | RunStatus::AwaitingClarification
            | RunStatus::Failed
            | RunStatus::Abandoned => false,
        }
    }
}

/// The minimal state shape a run carries between nodes. Node-produced slots stay untyped
/// (`Value`) in Phase 0 — see the module docs.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RunState {
    pub run_id: String,
    pub issue_id: Option<String>,
    pub status: RunStatus,
    #[serde(default)]
    pub scout_report: Option<Value>,
    #[serde(default)]
    pub memories: Vec<Value>,
    #[serde(default)]
    pub clarifications: Vec<Value>,
    #[serde(default)]
    pub analysis: Option<Value>,
    #[serde(default)]
    pub red_team: Option<Value>,
    #[serde(default)]
    pub implementer: Option<Value>,
    #[serde(default)]
    pub artifacts: Vec<Value>,
}

impl RunState {
    /// A fresh, `Pending` run with all node slots empty.
    pub fn new(run_id: impl Into<String>, issue_id: Option<String>) -> Self {
        RunState {
            run_id: run_id.into(),
            issue_id,
            status: RunStatus::Pending,
            scout_report: None,
            memories: Vec::new(),
            clarifications: Vec::new(),
            analysis: None,
            red_team: None,
            implementer: None,
            artifacts: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    #[test]
    fn run_state_round_trips_through_json() {
        let mut state = RunState::new("run-1", Some("issue-42".to_string()));
        state.status = RunStatus::AwaitingClarification;
        state.scout_report = Some(serde_json::json!({ "files": ["a.rs"] }));
        state.memories.push(serde_json::json!({ "id": "m1" }));

        let json = serde_json::to_string(&state).unwrap();
        let back: RunState = serde_json::from_str(&json).unwrap();

        assert_eq!(back.run_id, "run-1");
        assert_eq!(back.issue_id.as_deref(), Some("issue-42"));
        assert_eq!(back.status, RunStatus::AwaitingClarification);
        assert_eq!(back.scout_report, state.scout_report);
        assert_eq!(back.memories.len(), 1);
    }

    #[test]
    fn a_run_is_either_executing_or_finished() {
        use strum::IntoEnumIterator as _;

        // The exhaustive match means a new variant cannot be added without classifying it, but it
        // cannot say which side is right. This names the executing three, so a variant quietly
        // moved into that arm — where a killed run would never be reported finished, and a live
        // one could be started twice — fails here.
        let executing = ["pending", "running", "awaiting_clarification"];
        for status in RunStatus::iter() {
            assert_ne!(
                status.is_terminal(),
                executing.contains(&status.as_str()),
                "`{status}` must be terminal or executing, and only one"
            );
        }
    }

    #[test]
    fn run_status_str_round_trips() {
        for status in [
            RunStatus::Pending,
            RunStatus::Running,
            RunStatus::AwaitingClarification,
            RunStatus::Planned,
            RunStatus::Converged,
            RunStatus::MaxIterationsReached,
            RunStatus::Failed,
            RunStatus::Abandoned,
        ] {
            assert_eq!(RunStatus::from_str(status.as_str()).unwrap(), status);
        }
        assert!(RunStatus::from_str("nonsense").is_err());
    }
}
