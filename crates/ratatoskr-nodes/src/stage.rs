//! Reusable execution profiles and graph-specific stages.

use std::sync::Arc;

use ratatoskr_core::{AgentProfileConfig, Capability, ModelRoute, ToolPolicy};

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
    pub input_contract: String,
    pub output_contract: String,
    /// JSON Schema for [`Self::output_contract`] on user-declared stages.
    pub output_schema: Option<serde_json::Value>,
    pub instructions: String,
    pub context: String,
    /// A stage may only narrow its profile's ceiling.
    pub capabilities: Vec<Capability>,
    pub delegation: Option<Delegation>,
    /// Built-ins append repository guidance; user-defined stages may replace their prompt.
    pub append_repository_guidance: bool,
}

impl Stage {
    pub fn effective_ceiling(&self, profile: &AgentProfile) -> Option<Capability> {
        match (profile.ceiling(), Capability::ceiling(&self.capabilities)) {
            (Some(profile), Some(stage)) => Some(profile.min(stage)),
            (profile, None) => profile,
            (None, Some(_)) => None,
        }
    }

    /// Compose a stage prompt in authority-independent, stable order.
    pub fn prompt(
        &self,
        platform_invariants: &str,
        profile: &AgentProfile,
        repository_guidance: &str,
        runtime_input: &str,
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
            runtime_input,
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

/// Resolve the profile selected by a built-in stage.
pub fn stage_profile(
    config: &ratatoskr_core::RatatoskrConfig,
    stage_id: &str,
) -> Option<AgentProfile> {
    let stage_id = if stage_id == "redteam" {
        "red_team"
    } else {
        stage_id
    };
    let stage = built_in_stages()
        .into_iter()
        .find(|stage| stage.id == stage_id)?;
    agent_profiles(config)
        .into_iter()
        .find(|profile| profile.id == stage.agent)
}

/// Resolve a workflow's script metadata into the registry type validated by the execution layer.
pub fn stages_from_workflow(meta: &ratatoskr_script::workflow::WorkflowMeta) -> Vec<Stage> {
    meta.stages
        .iter()
        .map(|stage| Stage {
            id: stage.id.clone(),
            agent: stage.agent.clone(),
            input_contract: stage.input_contract.clone(),
            output_contract: stage.output_contract.clone(),
            output_schema: stage.output_schema.clone(),
            instructions: stage.instructions.clone(),
            context: stage.context.clone(),
            capabilities: stage.capabilities.clone(),
            delegation: stage.delegation.as_ref().map(|delegation| Delegation {
                target: delegation.target.clone(),
                evidence_contract: delegation.evidence_contract.clone(),
                input_limit: delegation.input_limit,
            }),
            append_repository_guidance: stage.append_repository_guidance,
        })
        .collect()
}

/// Built-in stage identities are intentionally the historic checkpoint names.
pub fn built_in_stages() -> Vec<Stage> {
    [
        ("overseer", "reason", "Vec<Workflow>", "OverseerOutput"),
        ("scout", "explore", "String", "ScoutOutput"),
        ("analyst", "reason", "AnalystInput", "AnalystOutput"),
        ("implementer", "build", "ImplementArg", "ImplementerOutput"),
        ("verifier", "explore", "VerifierInput", "VerifierOutput"),
        (
            "characterizer",
            "reason",
            "CharacterizerInput",
            "CharacterizerOutput",
        ),
        ("red_team", "reason", "()", "RedTeamOutput"),
        ("context", "explore", "String", "ContextOutput"),
        (
            "bookkeeper",
            "reason",
            "BookkeeperInput",
            "BookkeeperOutput",
        ),
        ("publisher", "publish", "PublisherInput", "PublisherOutput"),
    ]
    .into_iter()
    .map(|(id, agent, input_contract, output_contract)| Stage {
        id: id.into(),
        agent: agent.into(),
        input_contract: input_contract.into(),
        output_contract: output_contract.into(),
        output_schema: None,
        instructions: String::new(),
        context: String::new(),
        capabilities: Vec::new(),
        delegation: None,
        append_repository_guidance: true,
    })
    .collect()
}
