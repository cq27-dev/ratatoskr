//! Reusable execution profiles and graph-specific stages.

use std::sync::Arc;

use ratatoskr_core::{
    AgentProfileConfig, Capability, ModelRoute, SessionScope, ToolPolicy, shape::ShapeNode,
};

/// Reusable model and authority defaults. A profile is not a checkpoint identity; [`Stage`] is.
#[derive(Clone)]
pub struct AgentProfile {
    pub id: String,
    pub model: Option<ModelRoute>,
    pub base_prompt: String,
    pub capabilities: Vec<Capability>,
    pub tool_policy: Option<Arc<dyn ToolPolicy>>,
    pub max_turns: Option<usize>,
}

impl AgentProfile {
    pub fn from_config(id: impl Into<String>, config: AgentProfileConfig) -> Self {
        Self {
            id: id.into(),
            model: config.model,
            base_prompt: config.base_prompt,
            capabilities: config.capabilities,
            tool_policy: config.tool_policy,
            max_turns: config.max_turns,
        }
    }

    pub fn ceiling(&self) -> Option<Capability> {
        Capability::ceiling(&self.capabilities)
    }
}

/// A stable graph stage. Its identifier is also its persisted checkpoint name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage {
    pub id: String,
    pub agent: String,
    /// Stable route/rules/plugin identity. Defaults to [`Self::id`].
    pub governed_by: Option<String>,
    pub input_contract: String,
    pub output_contract: String,
    /// JSON Schema for [`Self::output_contract`] on user-declared stages.
    pub output_schema: Option<serde_json::Value>,
    pub instructions: String,
    pub context: String,
    /// A stage may only narrow its profile's ceiling.
    pub capabilities: Vec<Capability>,
    /// Default tools offered before repository rules narrow or replace the list.
    pub tools: Vec<String>,
    /// Overrides the selected route's attempt-continuation policy when present.
    pub session: Option<SessionScope>,
    /// Pure TypeScript source that renders structured runtime input into this stage's question.
    pub question_renderer: Option<String>,
    /// Generic output cleanup performed after schema validation.
    pub array_normalization: Vec<ArrayNormalization>,
    pub delegation: Option<Delegation>,
    /// Built-ins append repository guidance; user-defined stages may replace their prompt.
    pub append_repository_guidance: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayNormalization {
    pub field: String,
    pub default_empty: bool,
    pub retain_when_any_non_blank: Vec<String>,
}

impl Stage {
    pub fn governance_id(&self) -> &str {
        self.governed_by.as_deref().unwrap_or(&self.id)
    }

    pub fn effective_ceiling(&self, profile: &AgentProfile) -> Option<Capability> {
        match (profile.ceiling(), Capability::ceiling(&self.capabilities)) {
            (Some(profile), Some(stage)) => Some(profile.min(stage)),
            (profile, None) => profile,
            (None, Some(_)) => None,
        }
    }

    /// The attempt-continuation policy for this stage. An omitted declaration preserves the
    /// selected route, which keeps legacy `[models.<stage>]` configuration authoritative.
    pub fn session_scope(&self, route_default: SessionScope) -> SessionScope {
        self.session.unwrap_or(route_default)
    }

    /// Compose a stage prompt in authority-independent, stable order.
    pub fn prompt(
        &self,
        platform_invariants: &str,
        profile: &AgentProfile,
        repository_guidance: &str,
    ) -> String {
        [
            platform_invariants,
            profile.base_prompt.as_str(),
            self.instructions.as_str(),
            self.context.as_str(),
            if self.append_repository_guidance {
                repository_guidance
            } else {
                ""
            },
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
    }
}

/// A bounded child invocation declaration. It is evidence for its parent, never a checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Delegation {
    pub target: String,
    pub evidence_contract: String,
    /// Maximum serialized bytes projected from parent input into the child task.
    pub input_limit: usize,
}

/// The built-in reusable profiles. Routes remain stage-keyed until a repository opts into profiles,
/// which keeps `[models.<stage>]` and rulesets backwards compatible.
pub fn built_in_agents() -> Vec<AgentProfile> {
    vec![
        AgentProfile {
            id: "explore".into(),
            model: None,
            base_prompt: String::new(),
            capabilities: vec![Capability::Read],
            tool_policy: None,
            max_turns: None,
        },
        AgentProfile {
            id: "reason".into(),
            model: None,
            base_prompt: String::new(),
            capabilities: vec![Capability::Read],
            tool_policy: None,
            max_turns: None,
        },
        AgentProfile {
            id: "transcribe".into(),
            model: None,
            base_prompt: String::new(),
            capabilities: Vec::new(),
            tool_policy: None,
            max_turns: None,
        },
        AgentProfile {
            id: "build".into(),
            model: None,
            base_prompt: String::new(),
            capabilities: vec![Capability::Write],
            tool_policy: None,
            max_turns: None,
        },
        AgentProfile {
            id: "publish".into(),
            model: None,
            base_prompt: String::new(),
            capabilities: vec![Capability::Publish],
            tool_policy: None,
            max_turns: None,
        },
    ]
}

/// Built-in stage identities are intentionally the historic checkpoint names.
pub fn agent_profiles(config: &ratatoskr_core::RatatoskrConfig) -> Vec<AgentProfile> {
    let mut profiles = built_in_agents();
    for (id, profile) in &config.agents {
        if let Some(existing) = profiles.iter_mut().find(|existing| existing.id == *id) {
            *existing = AgentProfile::from_config(id, profile.clone());
        } else {
            profiles.push(AgentProfile::from_config(id, profile.clone()));
        }
    }
    profiles
}

/// Resolve the profile of the stage that runs under `node` in `stages`.
///
/// The registry is a parameter because a route or an enablement decision is about the stage that
/// will actually run, and a workflow may have overridden it. Resolving against a fixed table
/// instead is how `stage("verifier", { ...nodes.verifier, agent: "reason" })` came to report the
/// *built-in* verifier's agent, find no model for it, and disable review with no mention of it.
///
/// By stage id first — the standard stages are named after their identities — then by `governedBy`,
/// which is how the Rust-owned operations name the stage that runs for them: `implementer` resolves
/// to `implementer_attempt`, `redteam` to `redteam_classifier`, `context` to
/// `context_distillation`.
pub fn profile_for(
    config: &ratatoskr_core::RatatoskrConfig,
    stages: &[Stage],
    node: &str,
) -> Option<AgentProfile> {
    let stage = for_node(stages, node)?;
    agent_profiles(config)
        .into_iter()
        .find(|profile| profile.id == stage.agent)
}

/// The stage that runs when `node` is asked for, by the resolution [`profile_for`] documents.
///
/// Separate from [`profile_for`] because a route decision needs the stage itself, not its profile:
/// a stage's ruleset and `[models.*]` route are keyed by [`Stage::governance_id`], and looking those
/// up under the caller's name while the profile came from here is how the two halves of one decision
/// came to disagree.
pub fn for_node<'a>(stages: &'a [Stage], node: &str) -> Option<&'a Stage> {
    stages
        .iter()
        .find(|stage| stage.id == node)
        .or_else(|| stages.iter().find(|stage| stage.governance_id() == node))
}

/// Resolve a workflow's script metadata into the registry type validated by the execution layer.
pub fn stages_from_workflow(meta: &ratatoskr_script::workflow::WorkflowMeta) -> Vec<Stage> {
    meta.stages
        .iter()
        .map(|stage| Stage {
            id: stage.id.clone(),
            agent: stage.agent.clone(),
            governed_by: stage.governed_by.clone(),
            input_contract: stage.input_contract.clone(),
            output_contract: stage.output_contract.clone(),
            output_schema: stage.output_schema.clone(),
            instructions: stage.instructions.clone(),
            context: stage.context.clone(),
            capabilities: stage.capabilities.clone(),
            tools: stage.tools.clone(),
            session: stage.session,
            question_renderer: stage.question_renderer.clone(),
            array_normalization: stage
                .array_normalization
                .iter()
                .map(|normalization| ArrayNormalization {
                    field: normalization.field.clone(),
                    default_empty: normalization.default_empty,
                    retain_when_any_non_blank: normalization.retain_when_any_non_blank.clone(),
                })
                .collect(),
            delegation: stage.delegation.as_ref().map(|delegation| Delegation {
                target: delegation.target.clone(),
                evidence_contract: delegation.evidence_contract.clone(),
                input_limit: delegation.input_limit,
            }),
            append_repository_guidance: stage.append_repository_guidance,
        })
        .collect()
}

/// The layout a workflow declared, as the shape a run records.
///
/// A column is a `stage` and its `nodes` are its `lane`s, which is the whole vocabulary — the
/// declaration maps onto [`ShapeNode`] one for one and adds nothing to it. A workflow that declares
/// no layout gets an empty shape rather than a guessed one: nothing knows where its nodes belong,
/// and a viewer places what a run actually recorded instead of a position no one declared.
pub fn shape_from_workflow(meta: &ratatoskr_script::workflow::WorkflowMeta) -> Vec<ShapeNode> {
    meta.layout
        .iter()
        .enumerate()
        .flat_map(|(stage, column)| {
            column
                .nodes
                .iter()
                .enumerate()
                .map(move |(lane, name)| ShapeNode {
                    name: name.clone(),
                    stage,
                    lane,
                    optional: column.optional,
                })
        })
        .collect()
}

/// Lay a workflow's own declarations over a base registry: a declaration whose id is already there
/// *replaces* that stage in place, a new id is appended.
///
/// Importing `ratatoskr/nodes` and changing one field of a standard stage is the point of the
/// import, so the result has to be that one stage rather than two competing definitions of it.
/// Replacing in place also keeps the override where the standard stage sat, so the lookups that
/// resolve a stage by scanning the vec — delegation targets among them — find the override.
pub fn overlay(base: &mut Vec<Stage>, declared: Vec<Stage>) {
    for stage in declared {
        match base.iter_mut().find(|existing| existing.id == stage.id) {
            Some(existing) => *existing = stage,
            None => base.push(stage),
        }
    }
}

/// One bare stage, for a case that needs a [`Stage`] to mutate rather than a pipeline.
///
/// Deliberately not a registry: the stages a run has are `nodes.ts`'s, resolved by
/// [`crate::workflow::standard_stages`], and a second list of them here would be a second answer to
/// what the pipeline is. A case that needs the real registry awaits that; a case that only needs
/// *a* stage builds one here.
#[cfg(test)]
pub(crate) fn stage_fixture(id: &str, agent: &str) -> Stage {
    Stage {
        id: id.into(),
        agent: agent.into(),
        governed_by: None,
        input_contract: String::new(),
        output_contract: String::new(),
        output_schema: None,
        instructions: String::new(),
        context: String::new(),
        capabilities: Vec::new(),
        tools: Vec::new(),
        session: None,
        question_renderer: None,
        array_normalization: Vec::new(),
        delegation: None,
        append_repository_guidance: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declared_session_overrides_a_route_and_an_undeclared_one_preserves_it() {
        let workflow = ratatoskr_script::workflow::WorkflowMeta {
            name: "custom".to_string(),
            purpose: String::new(),
            when_to_use: Vec::new(),
            nodes: Vec::new(),
            stages: vec![ratatoskr_script::workflow::WorkflowStage {
                id: "review".to_string(),
                agent: "reason".to_string(),
                governed_by: None,
                input_contract: "ReviewInput".to_string(),
                output_contract: "ReviewOutput".to_string(),
                output_schema: Some(serde_json::json!({ "type": "object" })),
                instructions: String::new(),
                context: String::new(),
                capabilities: vec![Capability::Read],
                tools: Vec::new(),
                session: Some(SessionScope::Compacted),
                question_renderer: None,
                array_normalization: Vec::new(),
                delegation: None,
                append_repository_guidance: true,
            }],
            layout: Vec::new(),
        };

        let stages = stages_from_workflow(&workflow);
        assert_eq!(
            stages[0].session_scope(SessionScope::Fresh),
            SessionScope::Compacted
        );
        assert_eq!(stages[0].governance_id(), "review");
        // A stage that declares no session keeps whatever its `[models.*]` route chose, which is
        // what makes an existing TOML route authoritative until a workflow says otherwise.
        assert_eq!(
            stage_fixture("review", "reason").session_scope(SessionScope::Reuse),
            SessionScope::Reuse
        );
    }
}
