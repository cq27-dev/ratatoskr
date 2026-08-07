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

use std::fmt::Write as _;

use ratatoskr_core::ModelRoute;
use ratatoskr_graph::{NodeError, parse_validated};
use ratatoskr_mcp::ToolSet;
use rmcp::service::ServerSink;
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

const PREAMBLE: &str = include_str!("../prompts/context.md");

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

/// The context node: one agent turn over a guaranteed memory baseline.
pub struct ContextNode {
    pub route: ModelRoute,
    pub tools: ToolSet,
    /// For the deterministic `memory_search` that runs whatever the model does.
    pub sink: ServerSink,
    pub policy: Option<std::sync::Arc<dyn ratatoskr_core::ToolPolicy>>,
    pub max_turns: Option<usize>,
    pub clarifier: Option<std::sync::Arc<dyn ratatoskr_agent::Clarifier>>,
    /// Ruleset `systemPrompt`; replaces [`PREAMBLE`] when set.
    pub system_prompt: Option<String>,
    pub plugins: crate::NodePlugins,
    pub ledger: Option<std::sync::Arc<ratatoskr_agent::RunLedger>>,
    pub files: Option<std::path::PathBuf>,
}

impl ContextNode {
    pub async fn run(&self, issue: &str) -> Result<ContextOutput, NodeError> {
        // The baseline retrieval happens before the model is asked anything, and it happens
        // whatever the model does. Making it a tool the model may call would make "were the repo's
        // recorded constraints consulted" a thing that varies per run.
        let memory = crate::memory::search(&self.sink, issue, "").await?;

        let raw = ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "context",
            route: &self.route,
            preamble: &crate::effective_preamble(
                PREAMBLE,
                self.system_prompt.as_deref(),
                self.plugins.context.as_deref(),
            ),
            question: &render_prompt(issue, &memory),
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(Distillation),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            clarifier: self.clarifier.clone(),
            observer: self.plugins.observer.clone(),
            skills: crate::skills::loaded(&self.plugins.skills),
            files: self.files.clone(),
            // Reads and edits, but runs nothing.
            shell: None,
            conversation: None,
            ledger: self.ledger.clone(),
            produces: Some(
                "a brief on what bears on this task, the constraints it must respect with their \
                 sources, and the prior art found",
            ),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("context agent failed: {e}")))?;

        let mut distilled = parse_validated::<Distillation>(&raw)?;
        // Empty placeholder items reach the analyst as noise otherwise — the scout node carried
        // this same filter for the same reason.
        distilled
            .prior_art
            .retain(|i| !i.item_key.trim().is_empty() || !i.title.trim().is_empty());

        Ok(ContextOutput {
            brief: distilled.brief,
            constraints: distilled.constraints,
            scout: ScoutOutput {
                related_items: distilled.prior_art,
                papertrail_summary: distilled.papertrail_summary,
            },
            memory,
        })
    }
}

fn render_prompt(issue: &str, memory: &MemoryOutput) -> String {
    let mut s = String::new();
    let _ = write!(s, "TASK:\n{issue}\n\n");
    if memory.memories.is_empty() {
        s.push_str(
            "RECORDED MEMORIES: none matched this task. Search again yourself with different \
             terms before concluding this repository records nothing about it.\n",
        );
        return s;
    }
    s.push_str(
        "RECORDED MEMORIES — already retrieved for you, ranked. Quote from these when you write a \
         constraint, and cite the id:\n\n",
    );
    for m in &memory.memories {
        let _ = writeln!(s, "id: {}", m.memory_id);
        let _ = writeln!(s, "[{} | {}] {}", m.kind, m.confidence, m.title);
        let body = m.summary.as_deref().unwrap_or(&m.body);
        let _ = writeln!(s, "{body}\n");
    }
    s
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
    fn the_model_is_shown_the_memories_it_is_expected_to_cite() {
        let memory = MemoryOutput {
            memories: vec![record(
                "mem_1",
                "Adding a column needs both places",
                "schema.sql and ADDED_COLUMNS; neither alone migrates an existing store.",
            )],
        };
        let prompt = render_prompt("Add a repo_sha column.", &memory);
        assert!(prompt.contains("id: mem_1"));
        assert!(prompt.contains("ADDED_COLUMNS"), "the body arrives whole");
        assert!(prompt.contains("cite the id"));
    }

    #[test]
    fn no_matches_is_told_apart_from_nothing_recorded() {
        // Otherwise an empty baseline reads as "this repo records nothing", and the model stops
        // looking — when the far more likely truth is that the ranked query missed.
        let prompt = render_prompt("x", &MemoryOutput { memories: vec![] });
        assert!(prompt.contains("none matched"));
        assert!(prompt.contains("Search again"));
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
        let out = parse_validated::<Distillation>(raw).unwrap();
        assert_eq!(out.constraints[0].from_memory_id, "mem_1");
        // A constraint from the code or the tracker has no memory to cite, and that is not an
        // error — it is the absence of one.
        assert!(out.constraints[1].from_memory_id.is_empty());
    }
}
