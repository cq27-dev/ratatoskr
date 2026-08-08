//! The synthetic `Skill` tool: how a node is offered the skills its plugins ship.
//!
//! The format's two halves of progressive disclosure are split by where each lives. Every bound
//! skill's *name and description* is listed in the node's preamble (see `effective_preamble`),
//! prose in the cached prefix rather than a schema that grows with the number of skills; the *body*
//! of the one it picks arrives as this tool's result, and only then. The tool's schema is a single
//! free-form `skill: string` — the name is resolved at call time, so an unknown one is an
//! answerable error rather than unrepresentable. A node that binds no skill gets no tool at all.

use rmcp::model::Tool;
use serde_json::json;

/// The `Skill` tool for `skills`, or `None` when there is nothing to offer.
///
/// Named as the format names it, so a skill written for a coding CLI is invoked the same way here.
/// Like `ask`, it is a system capability rather than a rag-rat tool: the hook that answers it runs
/// before the ruleset gate, and a repo that does not want it unbinds the plugin.
///
/// The schema and description are constant — they do not name any skill and do not grow with the
/// number bound. The listing of what is available lives in the node's preamble instead.
pub(crate) fn skill_tool(skills: &[ratatoskr_plugin::Skill], node: &str) -> Option<Tool> {
    if deduped(skills, node).is_empty() {
        return None;
    }

    let schema = json!({
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "The name of the skill to load, exactly as listed in your instructions."
            }
        },
        "required": ["skill"]
    });

    let mut tool = Tool::default();
    tool.name = ratatoskr_agent::SKILL_TOOL_NAME.into();
    tool.description = Some(
        "Load a skill's full instructions as this tool's result, then follow them. A skill is a \
         procedure this repository has written down; your instructions list the available skills \
         and say when each applies. Load one when its description matches what you are doing — not \
         otherwise, and not more than you need. Call this only with a skill name listed in your \
         instructions."
            .into(),
    );
    tool.input_schema = std::sync::Arc::new(schema.as_object().cloned().unwrap_or_default());
    Some(tool)
}

/// The bound skills, deduped by name in binding order — a node's own plugins before ones it merely
/// inherits, and the first binding of a name wins.
///
/// The drop of a same-named later binding is logged: two bound plugins can ship a skill of the same
/// name, and offering it twice would leave which body loads to discovery order.
fn deduped<'a>(
    skills: &'a [ratatoskr_plugin::Skill],
    node: &str,
) -> Vec<&'a ratatoskr_plugin::Skill> {
    let mut kept: Vec<&ratatoskr_plugin::Skill> = Vec::new();
    for skill in skills {
        if kept.iter().any(|k| k.name == skill.name) {
            tracing::warn!(
                skill = skill.name,
                node,
                "not offering skill: another bound plugin already offers that name"
            );
            continue;
        }
        kept.push(skill);
    }
    kept
}

/// The 'Available skills:' listing for a node's preamble: every deduped bound skill, one per line.
///
/// Empty (and no header) when nothing is bound, so a node that binds no plugin runs with a preamble
/// byte-identical to before. This is the same set [`loaded`] answers, so offered and loadable agree.
pub(crate) fn listing(skills: &[ratatoskr_plugin::Skill], node: &str) -> Option<String> {
    let kept = deduped(skills, node);
    if kept.is_empty() {
        return None;
    }
    Some(
        kept.iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// What the agent needs to answer a `Skill` call: the name, and the instructions.
///
/// `${CLAUDE_SKILL_DIR}` is resolved here rather than in the loader, because it is only meaningful
/// once the body is about to be read by a model that might act on the paths in it.
pub(crate) fn loaded(
    skills: &[ratatoskr_plugin::Skill],
    node: &str,
) -> Vec<ratatoskr_agent::Skill> {
    deduped(skills, node)
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
    fn nothing_bound_means_no_tool_at_all() {
        assert!(skill_tool(&[], "scout").is_none());
    }

    #[test]
    fn a_skill_body_can_address_its_own_directory() {
        let loaded = loaded(&[skill("demo", "d")], "scout");
        assert_eq!(loaded[0].body, "do demo in /plugins/demo");
    }

    // --- node attribution on the duplicate-name warning (issue #165) ---
    //
    // These capture the `tracing` warn records with a minimal in-crate subscriber (the `nodes`
    // crate does not depend on `tracing-subscriber`) and assert the node the skill was dropped for
    // rides on the record.

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

    #[test]
    fn a_duplicate_name_warning_names_the_node() {
        let skills = [
            skill("shared", "from the first"),
            skill("shared", "from the second"),
        ];
        let records = capture(|| {
            let offered = deduped(&skills, "scout");
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
    fn a_duplicate_name_warning_survives_an_empty_node_field() {
        let skills = [
            skill("shared", "from the first"),
            skill("shared", "from the second"),
        ];
        // Must not panic on an empty node; the record still carries the (empty) node field.
        let records = capture(|| {
            let _ = deduped(&skills, "");
        });
        let dropped = records.first().expect("a warning");
        assert_eq!(dropped.get("node").map(String::as_str), Some(""));
    }

    // --- the listing moves out of the tool schema (issue #143) ---
    //
    // The schema stops growing with the number of bound skills (a single free-form
    // `skill: string`); the listing lives in the node's preamble instead, and the name is
    // resolved at call time. These exercise the contracted shapes: `skill_tool`'s schema and
    // description are constant, `loaded` returns every deduped bound skill, and the preamble
    // listing (`effective_preamble` in lib.rs) is always the same set `loaded` can answer.

    /// Six skills whose descriptions together would bust the old listing budget (~1 000 chars
    /// each). Descriptions deliberately do not contain the skill's own name, so tests can count
    /// name occurrences in a listing without matching a description.
    fn six_skills() -> Vec<ratatoskr_plugin::Skill> {
        ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
            .into_iter()
            .map(|name| {
                skill(
                    name,
                    &format!("when this one applies: {}", "x".repeat(1000)),
                )
            })
            .collect()
    }

    #[test]
    fn the_schema_is_constant_however_many_skills_are_bound() {
        let one = [skill("alpha", "when alpha is wanted")];
        let many = six_skills();
        let one_tool = skill_tool(&one, "scout").expect("a tool for one skill");
        let many_tool = skill_tool(&many, "scout").expect("a tool for six skills");

        let one_schema = serde_json::to_value(&*one_tool.input_schema).unwrap();
        let many_schema = serde_json::to_value(&*many_tool.input_schema).unwrap();
        assert_eq!(
            one_schema, many_schema,
            "the schema must not grow with the skills bound"
        );
        // A free-form name, not an enum: the choice is resolved at call time, so asking for an
        // unknown skill is an answerable error rather than unrepresentable.
        assert_eq!(many_schema["type"], "object");
        assert_eq!(many_schema["properties"]["skill"]["type"], "string");
        assert!(
            many_schema["properties"]["skill"].get("enum").is_none(),
            "no per-skill enum: {many_schema}"
        );
        assert_eq!(many_schema["required"], serde_json::json!(["skill"]));
        // And nothing per-skill outside the schema either: the description is the same text for
        // one skill or six.
        assert_eq!(one_tool.description, many_tool.description);
    }

    #[test]
    fn the_tool_description_names_no_individual_skill() {
        let many = six_skills();
        let tool = skill_tool(&many, "scout").expect("a tool");
        let described = tool.description.unwrap_or_default();
        for s in &many {
            assert!(
                !described.contains(&s.name),
                "the description must not list {}: {described}",
                s.name
            );
            assert!(
                !described.contains(&s.description),
                "the description must not describe {}",
                s.name
            );
        }
        let schema = serde_json::to_value(&*tool.input_schema)
            .unwrap()
            .to_string();
        for s in &many {
            assert!(
                !schema.contains(&s.name),
                "the schema must not name {}: {schema}",
                s.name
            );
        }
    }

    #[test]
    fn a_duplicate_name_still_offers_the_tool_and_warns_for_the_node() {
        // Two plugins shipping one name narrows the listing; it must not cost the node the tool.
        let skills = [
            skill("shared", "from the first"),
            skill("shared", "from the second"),
        ];
        let records = capture(|| {
            assert!(
                skill_tool(&skills, "analyst").is_some(),
                "the tool is still offered"
            );
        });
        let dropped = records
            .iter()
            .find(|r| r.values().any(|v| v.contains("shared")))
            .expect("the duplicate is dropped with a warning");
        assert_eq!(dropped.get("node").map(String::as_str), Some("analyst"));
    }

    #[test]
    fn a_plugin_with_six_skills_has_all_six_loadable_in_binding_order() {
        // The issue's acceptance case: nothing caps what a node can load any more, so a
        // six-skill plugin is carried whole — binding order, directory placeholder resolved.
        let skills = six_skills();
        let loaded = loaded(&skills, "scout");
        let names: Vec<&str> = loaded.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
        );
        assert_eq!(loaded[0].body, "do alpha in /plugins/alpha");
    }

    #[test]
    fn a_duplicate_name_loads_the_first_binding_once_and_warns_for_the_node() {
        // Distinct directories, so which binding wins is observable in the resolved body.
        let mut first = skill("shared", "from the first");
        first.dir = PathBuf::from("/plugins/first");
        let mut second = skill("shared", "from the second");
        second.dir = PathBuf::from("/plugins/second");
        let skills = [first, second];
        let records = capture(|| {
            let loaded = loaded(&skills, "context");
            assert_eq!(loaded.len(), 1, "one entry, not one per binding");
            assert_eq!(
                loaded[0].body, "do shared in /plugins/first",
                "the first binding wins"
            );
        });
        let dropped = records
            .iter()
            .find(|r| r.values().any(|v| v.contains("shared")))
            .expect("the duplicate is dropped with a warning");
        assert_eq!(dropped.get("node").map(String::as_str), Some("context"));
    }

    #[test]
    fn the_preamble_listing_and_the_loadable_set_are_always_the_same() {
        // Offered (the preamble listing) and loadable (what `loaded` hands the Skill hook) must
        // agree: a skill the model can name but nothing can answer — or one that loads but was
        // never listed — is exactly the failure mode this change exists to remove.
        let mut skills = six_skills();
        skills.push(skill("alpha", "a second binding of alpha"));
        let loaded_names: Vec<String> = loaded(&skills, "scout")
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(
            loaded_names
                .iter()
                .filter(|n| n.as_str() == "alpha")
                .count(),
            1,
            "deduped, like the listing"
        );

        let preamble = crate::effective_preamble("n", "base", None, None, &skills);
        let listing = preamble
            .split("Available skills:")
            .nth(1)
            .expect("a skills listing in the preamble");
        for name in &loaded_names {
            assert!(
                listing.contains(name.as_str()),
                "loadable but not listed: {name}"
            );
        }
        // And nothing listed twice: the duplicate binding appears once, as `loaded` kept it once.
        assert_eq!(
            listing.matches("alpha").count(),
            1,
            "listed once, as it loads once: {listing}"
        );
    }

    #[test]
    fn listing_a_long_description_in_the_preamble_warns_nothing_about_a_budget() {
        // The listing is prose in the preamble now, so there is nothing to be full of: the whole
        // description is listed and the old "the listing is full" warning is gone with the budget.
        let description = format!("when the {} case applies", "very long ".repeat(500));
        let skills = [skill("verbose", &description)];
        let records = capture(|| {
            let preamble = crate::effective_preamble("n", "base", None, None, &skills);
            assert!(
                preamble.contains(&description),
                "listed in full, not dropped to fit a budget"
            );
        });
        assert!(
            !records.iter().any(|r| r
                .get("message")
                .is_some_and(|m| m.contains("the listing is full"))),
            "the budget warning no longer exists: {records:?}"
        );
    }
}
