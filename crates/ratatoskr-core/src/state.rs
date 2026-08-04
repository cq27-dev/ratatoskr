//! The run state that flows through the graph and round-trips through the checkpoint store.
//!
//! Per-node payloads are kept as untyped [`serde_json::Value`] in Phase 0: the nodes that
//! produce them don't exist yet, so each node crate will define its own `JsonSchema`-deriving
//! struct and validate at the handoff boundary once it's actually built (Phase 2 onward).

use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Lifecycle status of a run. Serializes as a lowercase string (e.g. `"awaiting_clarification"`),
/// which is also the form persisted in the checkpoint store's `status` column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Pending,
    Running,
    AwaitingClarification,
    /// The linear scout → memory → analyst planning flow finished successfully (Phase 2).
    /// Distinct from `Converged`, which is reserved for Phase 4's real done-criteria.
    Planned,
    Converged,
    Failed,
    Abandoned,
}

impl RunStatus {
    /// The persisted string form. Kept in sync with the `#[serde(rename_all)]` above.
    pub fn as_str(&self) -> &'static str {
        match self {
            RunStatus::Pending => "pending",
            RunStatus::Running => "running",
            RunStatus::AwaitingClarification => "awaiting_clarification",
            RunStatus::Planned => "planned",
            RunStatus::Converged => "converged",
            RunStatus::Failed => "failed",
            RunStatus::Abandoned => "abandoned",
        }
    }
}

/// Error returned when a status string from the store (or config) doesn't name a known variant.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("unknown run status: {0:?}")]
pub struct ParseRunStatusError(pub String);

impl FromStr for RunStatus {
    type Err = ParseRunStatusError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "pending" => RunStatus::Pending,
            "running" => RunStatus::Running,
            "awaiting_clarification" => RunStatus::AwaitingClarification,
            "planned" => RunStatus::Planned,
            "converged" => RunStatus::Converged,
            "failed" => RunStatus::Failed,
            "abandoned" => RunStatus::Abandoned,
            other => return Err(ParseRunStatusError(other.to_string())),
        })
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
            RunStatus::Failed,
            RunStatus::Abandoned,
        ] {
            assert_eq!(RunStatus::from_str(status.as_str()), Ok(status));
        }
        assert!(RunStatus::from_str("nonsense").is_err());
    }
}
