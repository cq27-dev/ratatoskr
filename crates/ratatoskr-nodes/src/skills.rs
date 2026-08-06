//! The synthetic `Skill` tool: how a node is offered the skills its plugins ship.
//!
//! Both halves of the format's progressive disclosure live here. Every bound skill's *description*
//! goes into this one tool's schema, which the node carries for the whole run; the *body* of the
//! one it picks arrives as the tool's result, and only then. A node that binds no skill gets no
//! tool at all.

use rmcp::model::Tool;
use serde_json::json;

/// How much of the skill listing a node will carry.
///
/// The listing is in a tool schema, so it is paid for on every model call that node makes — the
/// same tax a long preamble would be, which is what a skill exists to avoid. A plugin that ships
/// fifteen skills should not be able to spend a node's context describing all of them.
const LISTING_BUDGET: usize = 4000;

/// The `Skill` tool for `skills`, or `None` when there is nothing to offer.
///
/// Named as the format names it, so a skill written for a coding CLI is invoked the same way here.
/// Like `ask`, it is a system capability rather than a rag-rat tool: the hook that answers it runs
/// before the ruleset gate, and a repo that does not want it unbinds the plugin.
pub(crate) fn skill_tool(skills: &[ratatoskr_plugin::Skill]) -> Option<Tool> {
    let offered = within_budget(skills);
    if offered.is_empty() {
        return None;
    }

    let listing = offered
        .iter()
        .map(|s| format!("- {}: {}", s.name, s.description))
        .collect::<Vec<_>>()
        .join("\n");
    let names: Vec<&str> = offered.iter().map(|s| s.name.as_str()).collect();

    let schema = json!({
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "enum": names,
                "description": "Which skill to load."
            }
        },
        "required": ["skill"]
    });

    let mut tool = Tool::default();
    tool.name = ratatoskr_agent::SKILL_TOOL_NAME.into();
    tool.description = Some(
        format!(
            "Load a skill's full instructions as this tool's result, then follow them. A skill is \
             a procedure this repository has written down; the descriptions below say when each \
             applies. Load one when its description matches what you are doing — not otherwise, \
             and not more than you need.\n\nAvailable skills:\n{listing}"
        )
        .into(),
    );
    tool.input_schema = std::sync::Arc::new(schema.as_object().cloned().unwrap_or_default());
    Some(tool)
}

/// The skills that fit the listing budget, whole descriptions in or out.
///
/// In binding order, so a node's own plugins are offered before ones it merely inherits, and the
/// drop is logged — a skill silently missing from the listing can never be chosen.
fn within_budget(skills: &[ratatoskr_plugin::Skill]) -> Vec<&ratatoskr_plugin::Skill> {
    let mut used = 0usize;
    let mut kept: Vec<&ratatoskr_plugin::Skill> = Vec::new();
    for skill in skills {
        // Two bound plugins can ship a skill of the same name. Offering it twice would put a
        // duplicate in the enum and leave which one loads to discovery order.
        if kept.iter().any(|k| k.name == skill.name) {
            tracing::warn!(
                skill = skill.name,
                "not offering skill: another bound plugin already offers that name"
            );
            continue;
        }
        let cost = skill.name.len() + skill.description.len() + 4;
        if used + cost > LISTING_BUDGET {
            tracing::warn!(
                skill = skill.name,
                "not offering skill: the listing is full"
            );
            continue;
        }
        used += cost;
        kept.push(skill);
    }
    kept
}

/// What the agent needs to answer a `Skill` call: the name, and the instructions.
///
/// `${CLAUDE_SKILL_DIR}` is resolved here rather than in the loader, because it is only meaningful
/// once the body is about to be read by a model that might act on the paths in it.
pub(crate) fn loaded(skills: &[ratatoskr_plugin::Skill]) -> Vec<ratatoskr_agent::Skill> {
    within_budget(skills)
        .into_iter()
        .map(|skill| ratatoskr_agent::Skill {
            name: skill.name.clone(),
            body: skill
                .body
                .replace("${CLAUDE_SKILL_DIR}", &skill.dir.display().to_string()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn skill(name: &str, description: &str) -> ratatoskr_plugin::Skill {
        ratatoskr_plugin::Skill {
            name: name.to_string(),
            description: description.to_string(),
            body: format!("do {name} in ${{CLAUDE_SKILL_DIR}}"),
            dir: PathBuf::from(format!("/plugins/{name}")),
        }
    }

    #[test]
    fn the_listing_describes_every_offered_skill_and_the_enum_bounds_the_choice() {
        let skills = [skill("dream-review", "when triaging findings")];
        let tool = skill_tool(&skills).expect("a tool");

        assert_eq!(tool.name, ratatoskr_agent::SKILL_TOOL_NAME);
        let described = tool.description.unwrap_or_default();
        assert!(described.contains("dream-review: when triaging findings"));
        // An enum rather than free text: a node cannot ask for a skill nobody has.
        let schema = serde_json::to_value(&*tool.input_schema).unwrap();
        assert_eq!(schema["properties"]["skill"]["enum"][0], "dream-review");
    }

    #[test]
    fn nothing_bound_means_no_tool_at_all() {
        assert!(skill_tool(&[]).is_none());
    }

    #[test]
    fn a_skill_that_does_not_fit_the_listing_is_not_offered_either() {
        // Offered and loadable must agree: a skill missing from the listing can never be chosen,
        // and one that could be chosen but was never described is worse.
        let skills = [
            skill("small", "brief"),
            skill("huge", &"x".repeat(LISTING_BUDGET)),
            skill("also-small", "brief"),
        ];
        let offered: Vec<&str> = within_budget(&skills)
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(offered, ["small", "also-small"]);
        assert_eq!(
            loaded(&skills)
                .iter()
                .map(|s| s.name.clone())
                .collect::<Vec<_>>(),
            ["small", "also-small"]
        );
    }

    #[test]
    fn one_name_is_offered_once_however_many_plugins_ship_it() {
        // Otherwise the enum holds a duplicate and which body loads is discovery order.
        let skills = [
            skill("shared", "from the first"),
            skill("shared", "from the second"),
        ];
        let offered = within_budget(&skills);
        assert_eq!(offered.len(), 1);
        assert_eq!(offered[0].description, "from the first");
    }

    #[test]
    fn a_skill_body_can_address_its_own_directory() {
        let loaded = loaded(&[skill("demo", "d")]);
        assert_eq!(loaded[0].body, "do demo in /plugins/demo");
    }
}
