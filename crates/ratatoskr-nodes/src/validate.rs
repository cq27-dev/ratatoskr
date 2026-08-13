//! Validation for the resolved stage/profile registry.

use std::collections::BTreeSet;

use crate::{AgentProfile, PlanError, Stage, policy};

/// Judge what one workflow declares, before its declarations are laid over the standard registry.
///
/// A workflow may reuse a *standard* identifier — that is an override, and the overlay replaces the
/// imported stage with it. What it may not do is take a name the run itself owns, or change a
/// contract the Rust side deserializes. Both come from [`crate::policy`], the one table that
/// classifies a standard identifier. Every case is refused here, at load, rather than filtered
/// afterwards: silently dropping a declaration is how an override comes to validate at startup and
/// then not run.
pub fn validate_declarations(stages: &[Stage], workflow: &str) -> Result<(), PlanError> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for stage in stages {
        // Declaring the same id twice is not an override of anything: the overlay would silently
        // keep only the last.
        if !seen.insert(stage.id.as_str()) {
            return Err(PlanError::Configuration(format!(
                "workflow `{workflow}` declares stage `{}` more than once",
                stage.id
            )));
        }
        if let Some(reason) = policy::reserved(&stage.id) {
            return Err(PlanError::Configuration(format!(
                "workflow `{workflow}` declares stage `{}`, which is {}",
                stage.id,
                reason.because()
            )));
        }
        // An override replaces the stage a Rust adapter runs, and that adapter deserializes the
        // output into a concrete type. Changing the contract is accepted by every other gate and
        // then fails — or, worse, deserializes into the wrong shape — in the middle of a run.
        if let Some(required) = policy::required_contract(&stage.id)
            && stage.output_contract != required
        {
            return Err(PlanError::Configuration(format!(
                "workflow `{workflow}` overrides stage `{}` with output contract `{}`; the run deserializes that stage's output as `{required}`, so an override must keep it",
                stage.id,
                if stage.output_contract.is_empty() {
                    "<none>"
                } else {
                    &stage.output_contract
                },
            )));
        }
        // `governedBy` is the identity the model turn is recorded under: its ruleset, its
        // `[models.*]` route, its plugin bindings, its telemetry attribution and its conversation
        // key. A name the run owns is no more available there than it is as a stage id.
        if let Some(governed_by) = stage.governed_by.as_deref()
            && let Some(reason) = policy::reserved_for_governance(governed_by)
        {
            return Err(PlanError::Configuration(format!(
                "workflow `{workflow}` stage `{}` is governedBy `{governed_by}`, which is {}",
                stage.id,
                reason.because()
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
        if policy::reserved(&profile.id) == Some(policy::Reserved::InternalGate) {
            return Err(PlanError::Configuration(format!(
                "agent `{}` claims internal fixed capability; internal gates are not configurable agents",
                profile.id
            )));
        }
    }

    for stage in stages {
        if policy::reserved(&stage.id) == Some(policy::Reserved::InternalGate) {
            return Err(PlanError::Configuration(format!(
                "stage `{}` claims internal fixed capability; valid stages: {}",
                stage.id,
                stage_names
                    .iter()
                    .filter(|name| policy::reserved(name) != Some(policy::Reserved::InternalGate))
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        if policy::reserved(&stage.id) == Some(policy::Reserved::Record) {
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
        // Delegation folds a child's evidence into the parent's runtime input on the way to the
        // parent's checkpoint. A stage the run reads back as evidence never reaches that, so the
        // declaration would validate here and then be dropped, unmentioned, at every invocation.
        if policy::folded_as_evidence(&parent.id) {
            return Err(PlanError::Configuration(format!(
                "stage `{}` declares delegation but is {}",
                parent.id,
                policy::FOLDED_AS_EVIDENCE_BECAUSE
            )));
        }
        let Some(target) = stages.iter().find(|stage| stage.id == delegation.target) else {
            return Err(PlanError::Configuration(format!(
                "stage `{}` delegates to unknown target `{}`; valid stages: {}",
                parent.id,
                delegation.target,
                stage_names.iter().copied().collect::<Vec<_>>().join(", ")
            )));
        };
        // The executor invokes a delegated child at the evidence disposition, and refuses a stage
        // that carries a delegation there — so a chain accepted here is a registry guaranteed to
        // fail the moment its parent runs. Refuse it while the error can still name the
        // declaration. Self-delegation is this same case pointing at itself, and would be an
        // infinite regress if it were honoured.
        //
        // Folding evidence recursively is the alternative. It would buy a depth of model turns
        // nothing bounds, for a shape no workflow here asks for: one stage gathering evidence from
        // one other is what delegation is for.
        if let Some(onwards) = &target.delegation {
            return Err(PlanError::Configuration(format!(
                "stage `{}` delegates to `{}`, which delegates onwards to `{}`; a delegation \
                 target must not delegate, and `{}` is invoked as evidence where its own \
                 delegation cannot run",
                parent.id, target.id, onwards.target, target.id
            )));
        }
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

/// Refuse a layout column naming a node the run has no way to produce.
///
/// The layout is what a viewer places a run's records against, so a name nothing records under is a
/// box that stays empty forever — a typo and a genuinely missing stage look identical once the run
/// is drawn.
///
/// A run records one node's work under *two* names, and a box only draws it whole when they are the
/// same name. `StageExecutor` passes the stage's [`Stage::governance_id`] as the model turn's node,
/// so that is what `node_start` and every model event carry; it checkpoints under the stage's `id`.
/// A column may therefore only name something both halves arrive under:
///
/// - a stage's **own id**, when the stage governs as itself — the ordinary case, and what the
///   standard layout's `analyst`, `verifier`, `bookkeeper` and `publisher` columns are. A stage the
///   run folds into another's record as evidence never checkpoints under its own name at all, so
///   `implementer_attempt` and `redteam_author` are boxes nothing could fill.
/// - an identity **the run itself checkpoints under** while the model turn for that work also runs
///   under it: `context`, `implementer`, `red_team`, `memory`. This is why the standard layout's
///   `context` column is legal though no stage is called `context`.
///
/// What is refused is a stage whose governance identity differs from its id and is not one of those:
/// its events land in one box and its checkpoint in another, and no single name draws it. Separating
/// the two identities (#259) is what makes such a stage drawable; until then it stays out of the
/// layout rather than being drawn half-empty under either spelling.
///
/// A name may appear once. Two columns naming it would draw two boxes with one identity, and the
/// viewer keys nodes by name — the second would overwrite the first's edges and state rather than
/// appearing beside it.
///
/// What is deliberately NOT checked is whether the column ORDER matches what the workflow does.
/// Order is meaningful — it is what draws the graph's hand-off edges, see
/// [`ratatoskr_script::workflow::WorkflowMeta::layout`] — so a layout can claim a hand-off its entry
/// function never performs, and that stays legal. Which hosts a run reaches, and in what order, is
/// decided by ordinary TypeScript control flow while the run executes; nothing in the declaration
/// states a data flow to compare a drawing against, and one layout legitimately serves every path a
/// run can take, which is what an `optional` column is for. Refusing a legal drawing on a guess
/// about an imperative script would be exactly the inference this design avoids. The checks here
/// are the ones the declaration can actually answer: that a named box can be filled at all, and
/// that each name is drawn once.
pub fn validate_layout(
    layout: &[ratatoskr_script::workflow::WorkflowLayoutColumn],
    stages: &[Stage],
    workflow: &str,
) -> Result<(), PlanError> {
    let mut known: BTreeSet<&str> = policy::checkpoint_identities().collect();
    for stage in stages {
        if stage.governance_id() == stage.id && !policy::folded_as_evidence(&stage.id) {
            known.insert(stage.id.as_str());
        }
    }
    let mut placed: BTreeSet<&str> = BTreeSet::new();
    for (index, column) in layout.iter().enumerate() {
        // A column is a position, and an empty one still takes its place in the order: the nodes
        // after it are drawn a column further along with nothing to their left. Declaring the whole
        // layout empty is worse — it records exactly what declaring none records, so a workflow
        // that meant to say where its nodes go is read as having said nothing.
        if column.nodes.is_empty() {
            return Err(PlanError::Configuration(format!(
                "workflow `{workflow}` lays out an empty column at position {index}; a column is \
                 the nodes drawn side by side in it, and one with none leaves a gap rather than \
                 placing anything"
            )));
        }
        for node in &column.nodes {
            if !known.contains(node.as_str()) {
                let drawable = known.iter().copied().collect::<Vec<_>>().join(", ");
                // Name the split where there is one: "nothing records under it" is true but
                // misleading for a stage whose records exist and simply arrive under two names.
                let split = stages.iter().find(|stage| {
                    stage.governance_id() != stage.id
                        && !policy::folded_as_evidence(&stage.id)
                        && (stage.id == *node || stage.governance_id() == node)
                });
                return Err(PlanError::Configuration(match split {
                    Some(stage) => format!(
                        "workflow `{workflow}` lays out node `{node}`, but stage `{id}` is \
                         governedBy `{governed}`: its model events are recorded under `{governed}` \
                         and its checkpoint under `{id}`, so neither name draws it whole. Drop \
                         `governedBy` from `{id}` so the two agree, or lay out one of: {drawable}",
                        id = stage.id,
                        governed = stage.governance_id(),
                    ),
                    None => format!(
                        "workflow `{workflow}` lays out node `{node}`, which nothing it runs \
                         records under; nodes that can be drawn: {drawable}"
                    ),
                }));
            }
            if !placed.insert(node.as_str()) {
                return Err(PlanError::Configuration(format!(
                    "workflow `{workflow}` lays out node `{node}` more than once; a run records \
                     one box under that name, so the later column would replace the earlier"
                )));
            }
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

    /// The governance identities a registry offers: every stage's id and the identity it runs
    /// under. What `governable_from` derives from the real standard stages, over the stages a case
    /// declares.
    fn permitted_for(stages: &[Stage]) -> Vec<String> {
        stages
            .iter()
            .flat_map(|stage| [stage.id.clone(), stage.governance_id().to_string()])
            .collect()
    }

    #[test]
    fn declared_contracts_require_a_valid_schema() {
        let mut stage = crate::stage::stage_fixture("publisher", "publish");
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
        let template = crate::stage::stage_fixture("analyst", "reason");
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
            let mut stage = crate::stage::stage_fixture("analyst", "reason");
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
        let mut stage = crate::stage::stage_fixture("analyst", "reason");
        stage.id = "custom_plan".to_string();
        stage.governed_by = None;

        assert!(validate(&[stage], &crate::built_in_agents(), &[]).is_ok());
    }

    #[test]
    fn explicit_builtin_governance_is_permitted() {
        let mut stage = crate::stage::stage_fixture("analyst", "reason");
        stage.id = "test_author".to_string();
        stage.governed_by = Some("redteam".to_string());

        // `redteam` is what the standard red-team stages are governed by, so it is in the set
        // `governable_from` derives from them.
        assert!(
            validate(
                &[stage],
                &crate::built_in_agents(),
                &["redteam".to_string()],
            )
            .is_ok()
        );
    }

    #[test]
    fn explicit_workflow_governance_is_permitted() {
        let mut stage = crate::stage::stage_fixture("analyst", "reason");
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
        let template = crate::stage::stage_fixture("analyst", "reason");
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
        let mut stage = crate::stage::stage_fixture("analyst", "reason");
        stage.id = "custom_plan".to_string();
        stage.governed_by = Some("verifer".to_string());
        let permitted = ["verifier".to_string(), "redteam".to_string()];

        let error = validate(&[stage], &crate::built_in_agents(), &permitted)
            .unwrap_err()
            .to_string();

        assert!(error.contains("unknown governedBy `verifer`"), "{error}");
        assert!(error.contains("verifier"), "{error}");
    }

    #[test]
    fn declared_stages_cannot_shadow_a_workflow_operation_host() {
        // Reserved from the host table itself, so a host added there cannot pick up a validation
        // gap. `context` is the case that used to slip through: it is an operation the bundled
        // `plan` and `full` both call, and a declaration of it failed only once the run was already
        // writing checkpoints.
        let template = crate::stage::stage_fixture("analyst", "reason");

        for (name, _) in crate::workflow::OPERATION_HOSTS {
            let mut stage = template.clone();
            stage.id = (*name).to_string();
            let error = validate_declarations(&[stage], "repo-workflow")
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("Rust-owned workflow operation"),
                "`{name}` must be reserved: {error}"
            );
            assert!(error.contains(name), "{error}");
        }
    }

    #[test]
    fn declared_stages_cannot_take_a_terminal_adapter_name() {
        let template = crate::stage::stage_fixture("analyst", "reason");

        for terminal in ["bookkeeper", "publisher"] {
            let mut stage = template.clone();
            stage.id = terminal.to_string();
            let error = validate_declarations(&[stage], "repo-workflow")
                .unwrap_err()
                .to_string();
            assert!(error.contains("terminal adapter"), "{error}");
            assert!(error.contains(terminal), "{error}");
        }
    }

    #[test]
    fn a_standard_stage_may_still_be_overridden() {
        let mut stage = crate::stage::stage_fixture("analyst", "reason");
        stage.id = "implementer_attempt".to_string();
        stage.output_contract = "Report".to_string();

        assert!(validate_declarations(&[stage], "repo-workflow").is_ok());
    }

    #[test]
    fn declared_stages_cannot_take_a_lifecycle_checkpoint_identity() {
        // The run counts `implementer` checkpoints for the iteration ordinal and the ceiling gate,
        // and deserializes `implementer`, `red_team` and `memory` into concrete types. A stage
        // checkpointing its own output under one of those names inflates the count, spends the
        // ceiling recovery early, or fails deserialization in the middle of a run.
        let template = crate::stage::stage_fixture("analyst", "reason");

        for reserved in ["implementer", "red_team", "memory"] {
            let mut stage = template.clone();
            stage.id = reserved.to_string();
            let error = validate_declarations(&[stage], "repo-workflow")
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("lifecycle checkpoint identity"),
                "`{reserved}` must be reserved: {error}"
            );
            assert!(error.contains(reserved), "{error}");
        }
    }

    #[test]
    fn a_declared_stage_cannot_shadow_the_red_teams_governance_identity() {
        // Stage resolution matches an exact id before falling back to `governedBy`, so a stage
        // named `redteam` answers the lookups made for `redteam_classifier` and `redteam_author`:
        // it decides whether the red team is enabled, and on whose profile. Enabling it that way
        // then leaves route resolution with a `redteam` that has no stage to resolve, and the run
        // dies on the name the workflow just introduced. `implementer` and `context`, the sibling
        // identities, are covered by reservations of their own.
        let template = crate::stage::stage_fixture("analyst", "reason");

        let mut stage = template.clone();
        stage.id = "redteam".to_string();
        let error = validate_declarations(&[stage], "repo-workflow")
            .unwrap_err()
            .to_string();
        assert!(error.contains("stage `redteam`"), "{error}");
        assert!(error.contains("governance identity"), "{error}");

        // The stages it governs keep their own declarable ids.
        for declarable in ["redteam_classifier", "redteam_author"] {
            let mut stage = template.clone();
            stage.id = declarable.to_string();
            stage.output_contract = policy::required_contract(declarable).unwrap().to_string();
            stage.governed_by = Some("redteam".to_string());
            assert!(
                validate_declarations(&[stage], "repo-workflow").is_ok(),
                "`{declarable}` must stay declarable"
            );
        }
    }

    #[test]
    fn an_override_may_not_change_a_contract_the_run_deserializes() {
        // `redTeam()` reads the `analyst` checkpoint back as `AnalystOutput` and `implement()`
        // takes one as its argument. An override that checkpoints another shape passes every other
        // gate and fails when the adapter reads it — after the stage has already run.
        let mut template = crate::stage::stage_fixture("analyst", "reason");
        template.output_contract = "AnalystOutput".to_string();

        let mut changed = template.clone();
        changed.output_contract = "SecurityPlan".to_string();
        let error = validate_declarations(&[changed], "repo-workflow")
            .unwrap_err()
            .to_string();
        assert!(error.contains("stage `analyst`"), "{error}");
        assert!(error.contains("`AnalystOutput`"), "{error}");
        assert!(error.contains("`SecurityPlan`"), "{error}");

        // Everything else about the stage is still the workflow's to change.
        let mut kept = template;
        kept.agent = "explore".to_string();
        kept.instructions = "read the diff first".to_string();
        assert!(validate_declarations(&[kept], "repo-workflow").is_ok());
    }

    #[test]
    fn governance_may_not_name_a_turn_the_workflow_never_makes() {
        // A stage's model turn runs under its `governedBy` identity: that ruleset, that
        // `[models.*]` route, those plugin bindings, that telemetry attribution and that
        // conversation key. Selection runs before a workflow is chosen and delivery after its
        // outcome is accepted, so neither is an identity a workflow stage can run as.
        let template = crate::stage::stage_fixture("analyst", "reason");

        for reserved in ["publisher", "bookkeeper", "overseer"] {
            let mut stage = template.clone();
            stage.id = "custom_plan".to_string();
            stage.governed_by = Some(reserved.to_string());
            let error = validate_declarations(&[stage], "repo-workflow")
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!("governedBy `{reserved}`")),
                "{error}"
            );
        }

        // The operations a workflow does drive keep their governance identity: the bundled
        // definitions declare `implementer_attempt` as governedBy `implementer` and
        // `context_distillation` as governedBy `context`.
        for permitted in ["implementer", "context", "redteam"] {
            let mut stage = template.clone();
            stage.id = "custom_plan".to_string();
            stage.governed_by = Some(permitted.to_string());
            assert!(
                validate_declarations(&[stage], "repo-workflow").is_ok(),
                "`{permitted}` must remain a governance identity"
            );
        }
    }

    #[test]
    fn a_delegated_renderer_is_refused_before_execution() {
        let template = crate::stage::stage_fixture("analyst", "reason");
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

    #[test]
    fn a_delegation_target_that_itself_delegates_is_refused_at_load() {
        // The executor invokes a delegated child at the evidence disposition, and a stage with a
        // delegation of its own is refused there — so a chain that loads is a registry guaranteed
        // to fail the moment its parent runs. Folding evidence recursively is the alternative, and
        // it buys a depth of model turns nothing bounds for a shape no workflow here wants.
        //
        // Self-delegation is the same declaration pointing at itself: an infinite regress if it
        // were honoured, and the same refusal covers it.
        let template = crate::stage::stage_fixture("analyst", "reason");
        let delegating = |id: &str, target: &str| {
            let mut stage = template.clone();
            stage.id = id.to_string();
            stage.output_contract = "Evidence".to_string();
            stage.output_schema = Some(serde_json::json!({ "type": "object" }));
            stage.delegation = Some(crate::Delegation {
                target: target.to_string(),
                evidence_contract: "Evidence".to_string(),
                input_limit: 1_000,
            });
            stage
        };
        let mut leaf = template.clone();
        leaf.id = "leaf".to_string();
        leaf.output_contract = "Evidence".to_string();
        leaf.output_schema = Some(serde_json::json!({ "type": "object" }));

        let chain = [
            delegating("first", "second"),
            delegating("second", "leaf"),
            leaf,
        ];
        let error = validate(&chain, &crate::built_in_agents(), &permitted_for(&chain))
            .expect_err("a delegation target that delegates must be refused")
            .to_string();
        assert!(error.contains("first"), "{error}");
        assert!(error.contains("second"), "{error}");
        assert!(error.contains("delegates onwards"), "{error}");

        let itself = [delegating("loop", "loop")];
        let error = validate(&itself, &crate::built_in_agents(), &permitted_for(&itself))
            .expect_err("a stage that delegates to itself must be refused")
            .to_string();
        assert!(error.contains("loop"), "{error}");
    }

    #[test]
    fn a_stage_the_run_folds_as_evidence_may_not_delegate() {
        // The bolt between the policy table and this gate. Delegation only runs on the way to a
        // checkpoint, so a standard stage whose output an adapter folds into another record takes
        // the declaration and drops it — silently, on every invocation. Adding one at that
        // disposition without classifying it here is what this refuses to let happen quietly.
        let template = crate::stage::stage_fixture("analyst", "reason");
        let mut child = template.clone();
        child.id = "child".to_string();
        child.output_contract = "Evidence".to_string();
        child.output_schema = Some(serde_json::json!({ "type": "object" }));
        let delegating = |id: &str| {
            let mut parent = template.clone();
            parent.id = id.to_string();
            parent.delegation = Some(crate::Delegation {
                target: "child".to_string(),
                evidence_contract: "Evidence".to_string(),
                input_limit: 1_000,
            });
            parent
        };

        let folded: Vec<&str> = policy::STANDARD_IDENTIFIERS
            .iter()
            .map(|(id, _)| *id)
            .filter(|id| policy::folded_as_evidence(id))
            .collect();
        assert!(
            !folded.is_empty(),
            "the table classifies no stage as evidence"
        );
        for id in folded {
            let stages = [delegating(id), child.clone()];
            let error = validate(&stages, &crate::built_in_agents(), &permitted_for(&stages))
                .expect_err("a stage the run folds as evidence must not take a delegation")
                .to_string();
            assert!(error.contains(id), "{error}");
            assert!(error.contains("rather than checkpointed itself"), "{error}");
        }

        // The verifier's adapter checkpoints what it runs, so delegation from it reaches execution.
        let stages = [delegating("verifier"), child];
        assert!(validate(&stages, &crate::built_in_agents(), &permitted_for(&stages)).is_ok());
    }

    fn column(node: &str) -> [ratatoskr_script::workflow::WorkflowLayoutColumn; 1] {
        [ratatoskr_script::workflow::WorkflowLayoutColumn {
            nodes: vec![node.to_string()],
            optional: false,
        }]
    }

    #[test]
    fn a_stage_governed_under_another_identity_is_not_drawable_under_either_name() {
        let mut reviewer = crate::stage::stage_fixture("reviewer", "reason");
        reviewer.governed_by = Some("analyst".to_string());
        let analyst = crate::stage::stage_fixture("analyst", "reason");
        let stages = [reviewer, analyst];

        // A run records the reviewer's model events under `analyst` and its checkpoint under
        // `reviewer`, so a column under either name draws half of it. The one it does not already
        // own is refused outright...
        let error = validate_layout(&column("reviewer"), &stages, "ours")
            .expect_err("a split identity cannot be drawn as one box")
            .to_string();
        assert!(error.contains("governedBy `analyst`"), "{error}");
        assert!(error.contains("neither name draws it whole"), "{error}");

        // ...and `analyst` stays drawable, because the analyst stage governs as itself: that box is
        // the analyst's own record. The reviewer's turns joining it is what separating execution
        // identity from governance identity fixes, not something the layout can express.
        assert!(validate_layout(&column("analyst"), &stages, "ours").is_ok());
    }

    #[test]
    fn the_identities_a_run_checkpoints_under_are_drawable_without_a_stage_of_that_name() {
        // No stage is called any of these; the run writes them itself and the model turn behind
        // each one is recorded under the same name, so a column names the whole node.
        let stages = [crate::stage::stage_fixture("analyst", "reason")];
        for identity in ["context", "implementer", "red_team", "memory"] {
            assert!(
                validate_layout(&column(identity), &stages, "ours").is_ok(),
                "`{identity}` is a name a run records under"
            );
        }
    }
}
