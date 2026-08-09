//! Ephemeral, bounded delegation support.

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{AgentProfile, PlanError, Stage};

/// A child invocation carries projected input and evidence only. It deliberately has no run or
/// checkpoint identifier, so callers cannot persist it as an independent graph stage.
#[derive(Clone, Debug)]
pub struct ChildTask {
    pub parent: String,
    pub target: String,
    pub input: Value,
    pub evidence_contract: String,
    pub capabilities: Option<ratatoskr_core::Capability>,
}

impl ChildTask {
    pub fn spawn(
        parent: &Stage,
        parent_profile: &AgentProfile,
        target: &Stage,
        target_profile: &AgentProfile,
        input: Value,
    ) -> Result<Self, PlanError> {
        let delegation = parent.delegation.as_ref().ok_or_else(|| {
            PlanError::Configuration(format!("stage `{}` is not eligible to delegate", parent.id))
        })?;
        if delegation.target != target.id {
            return Err(PlanError::Configuration(format!(
                "stage `{}` may not delegate to `{}`",
                parent.id, target.id
            )));
        }
        let bytes = serde_json::to_vec(&input)?;
        if bytes.len() > delegation.input_limit {
            return Err(PlanError::Configuration(format!(
                "stage `{}` projected {} bytes for child `{}`, above its {} byte limit",
                parent.id,
                bytes.len(),
                target.id,
                delegation.input_limit
            )));
        }
        let parent_ceiling = parent.effective_ceiling(parent_profile);
        let target_ceiling = target.effective_ceiling(target_profile);
        if target_ceiling > parent_ceiling {
            return Err(PlanError::Configuration(format!(
                "stage `{}` delegates to more privileged target `{}`",
                parent.id, target.id
            )));
        }
        Ok(Self {
            parent: parent.id.clone(),
            target: target.id.clone(),
            input,
            evidence_contract: delegation.evidence_contract.clone(),
            capabilities: parent_ceiling.min(target_ceiling),
        })
    }

    /// Deserialize the child's schema-validated output at its parent boundary without narrowing
    /// its JSON shape. A declared contract can validly produce an array or scalar as well as an
    /// object, and the parent must preserve that evidence exactly.
    pub fn evidence<T: DeserializeOwned>(&self, output: Value) -> Result<T, PlanError> {
        serde_json::from_value(output).map_err(|error| {
            PlanError::Configuration(format!(
                "child task `{}` returned invalid `{}` evidence for `{}`: {error}",
                self.target, self.evidence_contract, self.parent
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn task() -> ChildTask {
        ChildTask {
            parent: "parent".to_string(),
            target: "child".to_string(),
            input: json!(null),
            evidence_contract: "Evidence".to_string(),
            capabilities: None,
        }
    }

    #[test]
    fn evidence_preserves_contract_valid_non_object_json() {
        for evidence in [json!(["one", "two"]), json!("a scalar"), json!(42)] {
            let preserved: Value = task().evidence(evidence.clone()).unwrap();
            assert_eq!(preserved, evidence);
        }
    }
}
