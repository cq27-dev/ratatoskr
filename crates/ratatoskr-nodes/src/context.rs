//! Context: what this repository already knows that bears on the task.
//!
//! One node where there were two. `scout` investigated the tracker and the code, `memory` ran a
//! ranked `memory_search`, and both handed raw material to the analyst — which then had to
//! synthesise it. Nothing produced "here is what constrains this task", and no node could notice
//! that a recorded memory contradicts a tracker decision, because that finding needs both in one
//! head.
//!
//! The output is in two parts and the split is the design. The distillation is what the analyst
//! reads; the evidence is what it can check the distillation against. A model writes the first and
//! never touches the second.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::memory::MemoryOutput;
use crate::scout::{RelatedItem, ScoutOutput};

/// rag-rat tools this node may use. `memory_search` is here so it can look for more than the
/// guaranteed baseline once it knows what the task is about — not so it can decide whether the
/// baseline happens.
pub const CONTEXT_TOOLS: &[&str] = &[
    "papertrail_issue_search",
    "semantic_search",
    "symbol_lookup",
    "memory_search",
];

/// One thing this task has to respect, and where that came from.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Constraint {
    /// The constraint, in the terms of this task.
    pub says: String,
    /// The `memory_id` it was read from, when it came from a recorded memory. Empty when it came
    /// from the tracker or the code instead.
    ///
    /// Present so a reader can check the wording against the source — an interpretation that
    /// cannot be traced to what it interprets is one nobody can verify.
    #[serde(default)]
    pub from_memory_id: String,
}

/// What the model produced. Deliberately not the node's whole output: there is no field here for
/// the memories, so a model cannot write them, reword them, or leave them out.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Distillation {
    /// What a reader needs to know before planning this task. The analyst reads this first.
    pub brief: String,
    /// What the task must respect, each traced to where it came from.
    #[serde(default)]
    pub constraints: Vec<Constraint>,
    /// Tracker issues and PRs that bear on this task.
    #[serde(default)]
    pub prior_art: Vec<RelatedItem>,
    /// Free-text summary of the papertrail, carried for the analyst's existing prompt.
    #[serde(default)]
    pub papertrail_summary: String,
}

/// Deterministic evidence prepared before the model distils repository context.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct ContextDistillationInput {
    pub issue: String,
    pub memory: MemoryOutput,
    pub searchable: bool,
}

/// The node's output: the distillation, plus the evidence it was drawn from, unmodified.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ContextOutput {
    pub brief: String,
    #[serde(default)]
    pub constraints: Vec<Constraint>,
    /// What the scout half found, in the shape the analyst and the replay path already read.
    pub scout: ScoutOutput,
    /// The ranked memories, byte-for-byte as rag-rat returned them.
    ///
    /// Never routed through the model. A recorded invariant that reaches the analyst reworded is
    /// worse than one that never arrives: it looks like a constraint while no longer being the
    /// constraint.
    pub memory: MemoryOutput,
}

pub(crate) fn distillation_input(
    issue: &str,
    memory: MemoryOutput,
    searchable: bool,
) -> ContextDistillationInput {
    ContextDistillationInput {
        issue: issue.to_string(),
        memory,
        searchable,
    }
}

pub(crate) fn attach_evidence(mut distilled: Distillation, memory: MemoryOutput) -> ContextOutput {
    // Empty placeholder items reach the analyst as noise otherwise — the scout node carried this
    // same filter for the same reason. The evidence is attached only after the model output gate.
    distilled
        .prior_art
        .retain(|item| !item.item_key.trim().is_empty() || !item.title.trim().is_empty());
    ContextOutput {
        brief: distilled.brief,
        constraints: distilled.constraints,
        scout: ScoutOutput {
            related_items: distilled.prior_art,
            papertrail_summary: distilled.papertrail_summary,
        },
        memory,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryRecord;

    fn record(id: &str, title: &str, body: &str) -> MemoryRecord {
        MemoryRecord {
            memory_id: id.to_string(),
            kind: "Invariant".to_string(),
            title: title.to_string(),
            confidence: "high".to_string(),
            status: "active".to_string(),
            body: body.to_string(),
            summary: None,
        }
    }

    #[test]
    fn no_rag_rat_prepares_an_empty_non_searchable_baseline() {
        let input = distillation_input("explain the store", MemoryOutput::default(), false);
        assert!(input.memory.memories.is_empty());
        assert!(!input.searchable);
    }

    #[test]
    fn rust_attaches_the_baseline_verbatim_after_distillation() {
        let memory = MemoryOutput {
            memories: vec![record(
                "mem_exact",
                "Keep this wording",
                "This body must arrive byte-for-byte.",
            )],
        };
        let expected = serde_json::to_value(&memory).unwrap();
        let output = attach_evidence(
            Distillation {
                brief: "what bears on the task".to_string(),
                constraints: vec![Constraint {
                    says: "respect the recorded invariant".to_string(),
                    from_memory_id: "mem_exact".to_string(),
                }],
                prior_art: vec![
                    RelatedItem::default(),
                    RelatedItem {
                        item_key: "#118".to_string(),
                        title: "context evidence".to_string(),
                        ..Default::default()
                    },
                ],
                papertrail_summary: "one relevant issue".to_string(),
            },
            memory,
        );
        assert_eq!(serde_json::to_value(&output.memory).unwrap(), expected);
        assert_eq!(output.scout.related_items.len(), 1);
        assert_eq!(output.scout.related_items[0].item_key, "#118");
    }

    #[test]
    fn the_distillation_schema_has_nowhere_to_put_a_memory() {
        // The guarantee is structural, not a prompt instruction: the model fills `Distillation`,
        // which has no memories field, and Rust attaches the retrieved ones afterwards. A model
        // cannot reword, drop, or invent a recorded constraint because it is never asked for one.
        let schema = serde_json::to_value(schemars::schema_for!(Distillation)).unwrap();
        let properties = schema["properties"].as_object().unwrap();
        assert!(properties.contains_key("brief"));
        assert!(properties.contains_key("constraints"));
        assert!(!properties.contains_key("memory"));
        assert!(!properties.contains_key("memories"));
    }

    #[test]
    fn a_constraint_carries_the_source_it_was_read_from() {
        let raw = r#"{"brief":"b","constraints":[
            {"says":"both places or neither migrates","from_memory_id":"mem_1"},
            {"says":"from the tracker, not a memory"}
        ]}"#;
        let out = ratatoskr_graph::parse_validated::<Distillation>(raw).unwrap();
        assert_eq!(out.constraints[0].from_memory_id, "mem_1");
        // A constraint from the code or the tracker has no memory to cite, and that is not an
        // error — it is the absence of one.
        assert!(out.constraints[1].from_memory_id.is_empty());
    }
}
