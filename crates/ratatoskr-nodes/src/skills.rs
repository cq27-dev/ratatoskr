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
pub(crate) fn skill_tool(skills: &[ratatoskr_plugin::Skill], node: &str) -> Option<Tool> {
    let offered = within_budget(skills, node);
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
fn within_budget<'a>(
    skills: &'a [ratatoskr_plugin::Skill],
    node: &str,
) -> Vec<&'a ratatoskr_plugin::Skill> {
    let mut used = 0usize;
    let mut kept: Vec<&ratatoskr_plugin::Skill> = Vec::new();
    for skill in skills {
        // Two bound plugins can ship a skill of the same name. Offering it twice would put a
        // duplicate in the enum and leave which one loads to discovery order.
        if kept.iter().any(|k| k.name == skill.name) {
            tracing::warn!(
                skill = skill.name,
                node,
                "not offering skill: another bound plugin already offers that name"
            );
            continue;
        }
        let cost = skill.name.len() + skill.description.len() + 4;
        if used + cost > LISTING_BUDGET {
            tracing::warn!(
                skill = skill.name,
                node,
                cost,
                remaining = LISTING_BUDGET - used,
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
pub(crate) fn loaded(
    skills: &[ratatoskr_plugin::Skill],
    node: &str,
) -> Vec<ratatoskr_agent::Skill> {
    within_budget(skills, node)
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
        let tool = skill_tool(&skills, "scout").expect("a tool");

        assert_eq!(tool.name, ratatoskr_agent::SKILL_TOOL_NAME);
        let described = tool.description.unwrap_or_default();
        assert!(described.contains("dream-review: when triaging findings"));
        // An enum rather than free text: a node cannot ask for a skill nobody has.
        let schema = serde_json::to_value(&*tool.input_schema).unwrap();
        assert_eq!(schema["properties"]["skill"]["enum"][0], "dream-review");
    }

    #[test]
    fn nothing_bound_means_no_tool_at_all() {
        assert!(skill_tool(&[], "scout").is_none());
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
        let offered: Vec<&str> = within_budget(&skills, "scout")
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(offered, ["small", "also-small"]);
        assert_eq!(
            loaded(&skills, "scout")
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
        let offered = within_budget(&skills, "scout");
        assert_eq!(offered.len(), 1);
        assert_eq!(offered[0].description, "from the first");
    }

    #[test]
    fn a_skill_body_can_address_its_own_directory() {
        let loaded = loaded(&[skill("demo", "d")], "scout");
        assert_eq!(loaded[0].body, "do demo in /plugins/demo");
    }

    // --- node attribution on the dropped-skill warnings (issue #165) ---
    //
    // These tests exercise the contracted new signatures: `within_budget(skills, node)`,
    // `skill_tool(skills, node)` and `loaded(skills, node)`. They capture the `tracing` warn
    // records with a minimal in-crate subscriber (the `nodes` crate does not depend on
    // `tracing-subscriber`) and assert the node the skill was dropped for rides on the record.

    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// One captured event: its fields, keyed by name, rendered to strings.
    type Record = HashMap<String, String>;

    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<Record>>>);

    struct FieldVisitor<'a>(&'a mut Record);

    impl tracing::field::Visit for FieldVisitor<'_> {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
        fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
        fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
    }

    impl tracing::Subscriber for Captured {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            let mut fields = Record::new();
            let mut visitor = FieldVisitor(&mut fields);
            event.record(&mut visitor);
            self.0.lock().expect("capture mutex").push(fields);
        }
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    /// Run `f` with warnings captured, and return the records it emitted.
    fn capture<F: FnOnce()>(f: F) -> Vec<Record> {
        let sink = Captured::default();
        tracing::subscriber::with_default(sink.clone(), f);
        sink.0.lock().expect("capture mutex").clone()
    }

    /// True when any field of the record renders to exactly `needle`. Used for the cost/remaining
    /// numbers because the contract fixes the values but not the field names they are logged under.
    fn any_value_is(record: &Record, needle: &str) -> bool {
        record.values().any(|v| v == needle)
    }

    #[test]
    fn within_budget_keeps_binding_order_and_warns_nothing_when_all_fit() {
        let skills = [skill("a", "brief"), skill("b", "brief")];
        let records = capture(|| {
            let offered: Vec<&str> = within_budget(&skills, "scout")
                .iter()
                .map(|s| s.name.as_str())
                .collect();
            assert_eq!(offered, ["a", "b"]);
        });
        assert!(records.is_empty(), "no drop, no warning: {records:?}");
    }

    #[test]
    fn within_budget_full_listing_warning_names_the_node_cost_and_remaining() {
        // Nothing spent yet, so the remaining budget is the whole LISTING_BUDGET and the one
        // oversized skill overflows it. cost = name.len() + description.len() + 4.
        let name = "scout-huge";
        let description = "x".repeat(5000);
        let cost = name.len() + description.len() + 4;
        let skills = [skill(name, &description)];

        let records = capture(|| {
            let offered = within_budget(&skills, "scout");
            assert!(offered.is_empty(), "the oversized skill is excluded");
        });

        let dropped = records
            .iter()
            .find(|r| {
                r.get("message")
                    .is_some_and(|m| m.contains("the listing is full"))
            })
            .expect("a full-listing warning");
        assert_eq!(dropped.get("node").map(String::as_str), Some("scout"));
        // The skill's cost and the room that was left, whatever field names carry them.
        assert!(
            any_value_is(dropped, &cost.to_string()),
            "the skill's cost must be reported: {dropped:?}"
        );
        assert!(
            any_value_is(dropped, &LISTING_BUDGET.to_string()),
            "the remaining budget must be reported: {dropped:?}"
        );
    }

    #[test]
    fn within_budget_duplicate_name_warning_names_the_node() {
        let skills = [
            skill("shared", "from the first"),
            skill("shared", "from the second"),
        ];
        let records = capture(|| {
            let offered = within_budget(&skills, "scout");
            assert_eq!(offered.len(), 1);
        });
        let dropped = records
            .iter()
            .find(|r| {
                r.get("message")
                    .is_some_and(|m| m.contains("another bound plugin already offers that name"))
            })
            .expect("a duplicate-name warning");
        assert_eq!(dropped.get("node").map(String::as_str), Some("scout"));
    }

    #[test]
    fn within_budget_empty_slice_returns_empty_and_warns_nothing() {
        let records = capture(|| {
            assert!(within_budget(&[], "scout").is_empty());
        });
        assert!(records.is_empty(), "{records:?}");
    }

    #[test]
    fn within_budget_empty_node_still_warns_with_empty_node_field() {
        let skills = [skill("huge", &"x".repeat(LISTING_BUDGET))];
        // Must not panic on an empty node; the record still carries the (empty) node field.
        let records = capture(|| {
            let _ = within_budget(&skills, "");
        });
        let dropped = records.first().expect("a warning");
        assert_eq!(dropped.get("node").map(String::as_str), Some(""));
    }

    #[test]
    fn skill_tool_offers_when_something_fits_and_warns_drops_for_the_node() {
        let skills = [skill("fits", "brief"), skill("huge", &"x".repeat(5000))];
        let records = capture(|| {
            let tool = skill_tool(&skills, "analyst").expect("a tool, since one skill fits");
            let described = tool.description.unwrap_or_default();
            assert!(described.contains("fits: brief"));
        });
        let dropped = records
            .iter()
            .find(|r| {
                r.get("message")
                    .is_some_and(|m| m.contains("the listing is full"))
            })
            .expect("the oversized skill is warned");
        assert_eq!(dropped.get("node").map(String::as_str), Some("analyst"));
    }

    #[test]
    fn skill_tool_returns_none_when_none_fit_and_still_warns_for_the_node() {
        let skills = [skill("huge", &"x".repeat(5000))];
        let records = capture(|| {
            assert!(skill_tool(&skills, "analyst").is_none());
        });
        let dropped = records.first().expect("a warning");
        assert_eq!(dropped.get("node").map(String::as_str), Some("analyst"));
    }

    #[test]
    fn loaded_matches_what_skill_tool_offers_for_the_node() {
        let skills = [skill("a", "brief"), skill("b", "brief")];
        let loaded_names: Vec<String> = loaded(&skills, "context")
            .into_iter()
            .map(|s| s.name)
            .collect();
        let offered_names: Vec<String> = within_budget(&skills, "context")
            .into_iter()
            .map(|s| s.name.clone())
            .collect();
        assert_eq!(loaded_names, offered_names);
        // Directory placeholder resolved for the model that will read the body.
        let bodies: Vec<String> = loaded(&skills, "context")
            .into_iter()
            .map(|s| s.body)
            .collect();
        assert_eq!(bodies[0], "do a in /plugins/a");
    }

    #[test]
    fn loaded_excludes_a_budget_dropped_skill_and_warns_for_the_node() {
        let skills = [skill("fits", "brief"), skill("huge", &"x".repeat(5000))];
        let records = capture(|| {
            let names: Vec<String> = loaded(&skills, "context")
                .into_iter()
                .map(|s| s.name)
                .collect();
            assert_eq!(names, ["fits"]);
        });
        let dropped = records
            .iter()
            .find(|r| {
                r.get("message")
                    .is_some_and(|m| m.contains("the listing is full"))
            })
            .expect("the oversized skill is warned");
        assert_eq!(dropped.get("node").map(String::as_str), Some("context"));
    }
}
