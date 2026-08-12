//! Validation for the resolved stage/profile registry.

use std::collections::BTreeSet;

use crate::{AgentProfile, PlanError, Stage};

const INTERNAL_GATES: &[&str] = &["referee"];
// These names are kept for existing workflow.ts scripts. They are hosts, not stages: accepting
// one as a declared stage would replace its declared binding with the legacy host below it.
const LEGACY_HOST_ALIASES: &[&str] = &["memory", "implement", "iterate", "verify"];
// The run writes checkpoints under these names itself — `issue` for the task it was given,
// `clarification` for a completed question exchange — and readers identify those records by name
// alone: `issue_text` in ratatoskr-serve, the clarification-history check in this crate's workflow
// module, and the caller resolution the shape API does for a node it cannot place. A declared stage
// sharing a name would land its own output in the same column and be read as one of those records.
const RESERVED_RECORD_NAMES: &[&str] = &["issue", "clarification"];

/// Reject a workflow that declares the same stage identifier twice.
///
/// A workflow may reuse a *standard* identifier — that is an override, and the overlay replaces the
/// imported stage with it. Declaring the same id twice in one workflow is not an override of
/// anything: the overlay would silently keep only the last, so it is refused here instead.
pub fn validate_unique_declarations(stages: &[Stage], workflow: &str) -> Result<(), PlanError> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for stage in stages {
        if !seen.insert(stage.id.as_str()) {
            return Err(PlanError::Configuration(format!(
                "workflow `{workflow}` declares stage `{}` more than once",
                stage.id
            )));
        }
    }
    Ok(())
}

/// Reject invalid stage references before a workflow can start a model call.
pub fn validate(
    stages: &[Stage],
    profiles: &[AgentProfile],
    permitted_governance: &[String],
) -> Result<(), PlanError> {
    let agents: BTreeSet<&str> = profiles.iter().map(|profile| profile.id.as_str()).collect();
    let stage_names: BTreeSet<&str> = stages.iter().map(|stage| stage.id.as_str()).collect();
    let governance_names: BTreeSet<&str> =
        permitted_governance.iter().map(String::as_str).collect();
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
        if RESERVED_RECORD_NAMES.contains(&stage.id.as_str()) {
            return Err(PlanError::Configuration(format!(
                "stage `{}` conflicts with a checkpoint the run writes itself; choose a different stage identifier",
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
        if let Some(governed_by) = stage.governed_by.as_deref() {
            if !machine_name(governed_by) {
                return Err(PlanError::Configuration(format!(
                    "stage `{}` governedBy `{governed_by}` must use an underscore-separated identifier",
                    stage.id
                )));
            }
            if !governance_names.contains(governed_by) {
                return Err(PlanError::Configuration(format!(
                    "stage `{}` references unknown governedBy `{governed_by}`; valid governance identities: {}",
                    stage.id,
                    governance_names
                        .iter()
                        .copied()
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
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

    fn permitted_for(stages: &[Stage]) -> Vec<String> {
        crate::BUILT_IN_NODES
            .iter()
            .map(|name| (*name).to_string())
            .chain(stages.iter().map(|stage| stage.id.clone()))
            .collect()
    }

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
        assert!(validate(&stages, &crate::built_in_agents(), &permitted_for(&stages)).is_ok());
    }

    #[test]
    fn a_stage_may_not_take_the_name_of_a_record_the_run_writes_itself() {
        // `issue` and `clarification` are written by the run, and readers identify them by name
        // alone — so a declared stage under either name would put its own output where a reader
        // expects the run's, and the shape API would report its `from` field as a node caller.
        for reserved in ["issue", "clarification"] {
            let mut stage = crate::built_in_stages()
                .into_iter()
                .find(|stage| stage.id == "analyst")
                .unwrap();
            stage.id = reserved.to_string();
            stage.governed_by = None;

            let err = validate(&[stage], &crate::built_in_agents(), &[])
                .expect_err("a reserved record name must not be accepted as a stage");
            assert!(
                format!("{err}").contains("a checkpoint the run writes itself"),
                "`{reserved}` was rejected for the wrong reason: {err}"
            );
        }
    }

    #[test]
    fn omitted_governance_keeps_the_stage_identifier_fallback() {
        let mut stage = crate::built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();
        stage.id = "custom_plan".to_string();
        stage.governed_by = None;

        assert!(validate(&[stage], &crate::built_in_agents(), &[]).is_ok());
    }

    #[test]
    fn explicit_builtin_governance_is_permitted() {
        let mut stage = crate::built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();
        stage.id = "test_author".to_string();
        stage.governed_by = Some("redteam".to_string());
        let permitted = crate::BUILT_IN_NODES
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();

        assert!(validate(&[stage], &crate::built_in_agents(), &permitted).is_ok());
    }

    #[test]
    fn explicit_workflow_governance_is_permitted() {
        let mut stage = crate::built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();
        stage.id = "custom_plan".to_string();
        stage.governed_by = Some("shared_policy".to_string());

        assert!(
            validate(
                &[stage],
                &crate::built_in_agents(),
                &["shared_policy".to_string()],
            )
            .is_ok()
        );
    }

    #[test]
    fn explicit_governance_may_name_another_declared_stage() {
        let template = crate::built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();
        let mut policy = template.clone();
        policy.id = "shared_policy".to_string();
        let mut plan = template;
        plan.id = "custom_plan".to_string();
        plan.governed_by = Some(policy.id.clone());
        let stages = [policy, plan];

        assert!(validate(&stages, &crate::built_in_agents(), &permitted_for(&stages),).is_ok());
    }

    #[test]
    fn explicit_unknown_governance_is_rejected_before_execution() {
        let mut stage = crate::built_in_stages()
            .into_iter()
            .find(|stage| stage.id == "analyst")
            .unwrap();
        stage.id = "custom_plan".to_string();
        stage.governed_by = Some("verifer".to_string());
        let permitted = crate::BUILT_IN_NODES
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();

        let error = validate(&[stage], &crate::built_in_agents(), &permitted)
            .unwrap_err()
            .to_string();

        assert!(error.contains("unknown governedBy `verifer`"), "{error}");
        assert!(error.contains("verifier"), "{error}");
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
            let stages = [stage];
            let error = validate(&stages, &profiles, &permitted_for(&stages))
                .unwrap_err()
                .to_string();
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

        let stages = [parent, child];
        let error = validate(&stages, &crate::built_in_agents(), &permitted_for(&stages))
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("requires an explicit workflow host call"),
            "{error}"
        );
    }
}
