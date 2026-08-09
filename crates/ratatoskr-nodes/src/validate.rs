//! Validation for the resolved stage/profile registry.

use std::collections::BTreeSet;

use crate::{AgentProfile, PlanError, Stage};

const INTERNAL_GATES: &[&str] = &["referee"];
// These names are kept for existing workflow.ts scripts. They are hosts, not stages: accepting
// one as a declared stage would replace its declared binding with the legacy host below it.
const LEGACY_HOST_ALIASES: &[&str] = &["memory", "analyze", "implement", "iterate", "verify"];

/// Reject invalid stage references before a workflow can start a model call.
pub fn validate(stages: &[Stage], profiles: &[AgentProfile]) -> Result<(), PlanError> {
    let agents: BTreeSet<&str> = profiles.iter().map(|profile| profile.id.as_str()).collect();
    let stage_names: BTreeSet<&str> = stages.iter().map(|stage| stage.id.as_str()).collect();
    if stage_names.len() != stages.len() {
        return Err(PlanError::Configuration(
            "stage identifiers must be unique across configured workflows".to_string(),
        ));
    }

    for profile in profiles {
        if INTERNAL_GATES.contains(&profile.id.as_str()) {
            return Err(PlanError::Configuration(format!(
                "agent `{}` claims internal fixed capability; internal gates are not configurable agents",
                profile.id
            )));
        }
    }

    for stage in stages {
        if INTERNAL_GATES.contains(&stage.id.as_str()) {
            return Err(PlanError::Configuration(format!(
                "stage `{}` claims internal fixed capability; valid stages: {}",
                stage.id,
                stage_names
                    .iter()
                    .filter(|name| !INTERNAL_GATES.contains(name))
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        if LEGACY_HOST_ALIASES.contains(&stage.id.as_str()) {
            return Err(PlanError::Configuration(format!(
                "stage `{}` conflicts with a legacy workflow host alias; choose a different stage identifier",
                stage.id
            )));
        }
        if !agents.contains(stage.agent.as_str()) {
            return Err(PlanError::Configuration(format!(
                "stage `{}` references unknown agent `{}`; valid agents: {}",
                stage.id,
                stage.agent,
                agents.iter().copied().collect::<Vec<_>>().join(", ")
            )));
        }
        if !machine_name(&stage.id) {
            return Err(PlanError::Configuration(format!(
                "stage `{}` must use an underscore-separated identifier",
                stage.id
            )));
        }
    }

    for parent in stages {
        let Some(delegation) = &parent.delegation else {
            continue;
        };
        let Some(target) = stages.iter().find(|stage| stage.id == delegation.target) else {
            return Err(PlanError::Configuration(format!(
                "stage `{}` delegates to unknown target `{}`; valid stages: {}",
                parent.id,
                delegation.target,
                stage_names.iter().copied().collect::<Vec<_>>().join(", ")
            )));
        };
        if !target.output_contract.is_empty() && target.output_schema.is_none() {
            return Err(PlanError::Configuration(format!(
                "stage `{}` delegates to `{}`, whose output contract `{}` has no outputSchema",
                parent.id, target.id, target.output_contract
            )));
        }
        let parent_profile = profiles
            .iter()
            .find(|profile| profile.id == parent.agent)
            .expect("validated above");
        let target_profile = profiles
            .iter()
            .find(|profile| profile.id == target.agent)
            .expect("validated above");
        if target.effective_ceiling(target_profile) > parent.effective_ceiling(parent_profile) {
            return Err(PlanError::Configuration(format!(
                "stage `{}` delegates to more privileged target `{}`",
                parent.id, target.id
            )));
        }
        if !delegation.evidence_contract.is_empty()
            && delegation.evidence_contract != target.output_contract
        {
            return Err(PlanError::Configuration(format!(
                "stage `{}` expects `{}` evidence from `{}`, whose output contract is `{}`",
                parent.id, delegation.evidence_contract, target.id, target.output_contract
            )));
        }
    }
    Ok(())
}

/// A declared output contract needs the JSON Schema that makes its name enforceable. Compile the
/// schema before a run begins, rather than letting a malformed declaration reach a model call.
pub fn validate_declared_contracts(stages: &[Stage]) -> Result<(), PlanError> {
    for stage in stages {
        let Some(schema) = stage.output_schema.as_ref() else {
            if !stage.output_contract.is_empty() {
                return Err(PlanError::Configuration(format!(
                    "stage `{}` declares output contract `{}` without outputSchema",
                    stage.id, stage.output_contract
                )));
            }
            continue;
        };
        if let Err(error) = ratatoskr_graph::validate_raw("{}", schema)
            && error
                .to_string()
                .starts_with("output failed schema validation: could not compile")
        {
            return Err(PlanError::Configuration(format!(
                "stage `{}` has invalid outputSchema: {error}",
                stage.id
            )));
        }
    }
    Ok(())
}

/// Adjacent workflow stages exchange their declared contracts. Empty contracts are unspecified,
/// preserving existing workflows that only declare names.
pub fn validate_sequence(stages: &[Stage]) -> Result<(), PlanError> {
    for pair in stages.windows(2) {
        let (source, target) = (&pair[0], &pair[1]);
        if !source.output_contract.is_empty()
            && !target.input_contract.is_empty()
            && source.output_contract != target.input_contract
        {
            return Err(PlanError::Configuration(format!(
                "stage `{}` outputs `{}`, incompatible with successor `{}` input `{}`",
                source.id, source.output_contract, target.id, target.input_contract
            )));
        }
    }
    Ok(())
}

fn machine_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_contracts_require_a_valid_schema() {
        let mut stage = crate::built_in_stages().pop().unwrap();
        stage.id = "security_evidence".to_string();
        stage.output_contract = "SecurityEvidence".to_string();
        assert!(validate_declared_contracts(&[stage.clone()]).is_err());

        stage.output_schema = Some(serde_json::json!({
            "type": "object",
            "required": ["finding"],
            "properties": { "finding": { "type": "string" } }
        }));
        assert!(validate_declared_contracts(&[stage]).is_ok());
    }

    #[test]
    fn incompatible_declared_successors_are_rejected() {
        let mut stages = crate::built_in_stages();
        stages.extend([
            Stage {
                id: "first".to_string(),
                agent: "reason".to_string(),
                input_contract: String::new(),
                output_contract: "plan".to_string(),
                output_schema: None,
                instructions: String::new(),
                context: String::new(),
                capabilities: Vec::new(),
                delegation: None,
                append_repository_guidance: true,
            },
            Stage {
                id: "second".to_string(),
                agent: "reason".to_string(),
                input_contract: "review".to_string(),
                output_contract: String::new(),
                output_schema: None,
                instructions: String::new(),
                context: String::new(),
                capabilities: Vec::new(),
                delegation: None,
                append_repository_guidance: true,
            },
        ]);
        assert!(validate_sequence(&stages[10..]).is_err());
    }

    #[test]
    fn declared_stages_cannot_shadow_legacy_workflow_hosts() {
        let template = crate::built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();
        let profiles = crate::built_in_agents();

        for alias in LEGACY_HOST_ALIASES {
            let mut stage = template.clone();
            stage.id = (*alias).to_string();
            let error = validate(&[stage], &profiles).unwrap_err().to_string();
            assert!(
                error.contains("legacy workflow host alias"),
                "{alias} must be reserved: {error}"
            );
        }
    }
}
