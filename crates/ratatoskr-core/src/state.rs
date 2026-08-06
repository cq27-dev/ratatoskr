//! The run state that flows through the graph and round-trips through the checkpoint store.
//!
//! Per-node payloads are kept as untyped [`serde_json::Value`] in Phase 0: the nodes that
//! produce them don't exist yet, so each node crate will define its own `JsonSchema`-deriving
//! struct and validate at the handoff boundary once it's actually built (Phase 2 onward).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
