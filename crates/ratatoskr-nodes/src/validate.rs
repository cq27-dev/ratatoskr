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
        // A delegated child is invoked directly by the Rust executor, not through the JavaScript
        // host wrapper where `renderQuestion` runs. Refuse the unsupported shape up front instead
        // of silently handing the child raw JSON under a declaration that promised another prompt.
        if target.question_renderer.is_some() {
            return Err(PlanError::Configuration(format!(
                "stage `{}` delegates to `{}`, whose renderQuestion requires an explicit workflow host call",
                parent.id, target.id
            )));
        }
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
    fn declared_stages_are_a_registry_not_an_execution_sequence() {
        let template = crate::built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();
        let mut plan = template.clone();
        plan.id = "plan".to_string();
        plan.input_contract = "Issue".to_string();
        plan.output_contract = "Plan".to_string();
        plan.output_schema = Some(serde_json::json!({ "type": "object" }));

        let mut review = template;
        review.id = "review".to_string();
        review.input_contract = "ReviewInput".to_string();
        review.output_contract = "Review".to_string();
        review.output_schema = Some(serde_json::json!({ "type": "array" }));

        let stages = [plan, review];
        // The workflow script calls hosts explicitly, so metadata order cannot create a dataflow
        // edge or make two independently useful stages incompatible.
        assert!(validate_declared_contracts(&stages).is_ok());
        assert!(validate(&stages, &crate::built_in_agents()).is_ok());
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

    #[test]
    fn a_delegated_renderer_is_refused_before_execution() {
        let template = crate::built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();
        let mut parent = template.clone();
        parent.id = "parent".to_string();
        parent.delegation = Some(crate::Delegation {
            target: "child".to_string(),
            evidence_contract: "Evidence".to_string(),
            input_limit: 1_000,
        });
        let mut child = template;
        child.id = "child".to_string();
        child.output_contract = "Evidence".to_string();
        child.output_schema = Some(serde_json::json!({ "type": "object" }));
        child.question_renderer = Some("input => JSON.stringify(input)".to_string());

        let error = validate(&[parent, child], &crate::built_in_agents())
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("requires an explicit workflow host call"),
            "{error}"
        );
    }
}
