//! Analyst: given the issue, scout's findings, and repo memories, assess impact and risk.

use std::fmt::Write as _;

use ratatoskr_core::{ModelRoute, RunState};
use ratatoskr_graph::{Node, NodeError, parse_validated};
use ratatoskr_mcp::ToolSet;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::memory::MemoryOutput;
use crate::scout::ScoutOutput;

/// rag-rat tools the analyst may use to resolve what the change actually touches.
pub const ANALYST_TOOLS: &[&str] = &["impact_surface", "symbol_lookup", "semantic_search"];

const PREAMBLE: &str = include_str!("../prompts/analyst.md");

/// Input to the analyst: the issue plus the two upstream node outputs.
///
/// The last two fields are what makes this node re-enterable. The analyst used to produce
/// requirements exactly once, so a run that discovered on iteration three that the plan was wrong
/// could only re-drive the implementer against a plan already shown to be poor.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AnalystInput {
    pub issue: String,
    pub scout: ScoutOutput,
    pub memory: MemoryOutput,
    /// What the context node distilled: what a planner needs to know before starting.
    ///
    /// Defaulted, so a script still composing `scout` and `memory` by hand keeps working — it just
    /// hands over the evidence without the synthesis.
    #[serde(default)]
    pub brief: String,
    /// What this task must respect, each traced to the memory it was read from.
    #[serde(default)]
    pub constraints: Vec<crate::context::Constraint>,
    /// The plan being revised, when this is a revision. The analyst amends rather than re-derives:
    /// a blank sheet would discard the reasoning that was right along with the part that was not.
    #[serde(default)]
    pub previous: Option<Box<AnalystOutput>>,
    /// Why it is being revised — review findings the verifier judged to be faults in the plan
    /// rather than in the code.
    #[serde(default)]
    pub findings: Vec<crate::verifier::Finding>,
}

impl AnalystInput {
    /// A first plan, with no revision history.
    pub fn fresh(issue: String, scout: ScoutOutput, memory: MemoryOutput) -> Self {
        AnalystInput {
            issue,
            scout,
            memory,
            previous: None,
            findings: Vec::new(),
            brief: String::new(),
            constraints: Vec::new(),
        }
    }

    /// A first plan from a context node's output.
    pub fn from_context(issue: String, context: crate::context::ContextOutput) -> Self {
        AnalystInput {
            issue,
            scout: context.scout,
            memory: context.memory,
            brief: context.brief,
            constraints: context.constraints,
            previous: None,
            findings: Vec::new(),
        }
    }

    fn is_revision(&self) -> bool {
        self.previous.is_some() && !self.findings.is_empty()
    }
}

/// Analyst's structured output — the plan's substance.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AnalystOutput {
    pub impact_summary: String,
    /// Specific symbols/paths the change touches.
    #[serde(default)]
    pub touched: Vec<String>,
    /// Risks, one short line each (severity is just part of the text). Free text on purpose: no
    /// consumer branches on a structured severity, and a plain list can't fail schema validation the
    /// way a `{description, severity}` object did when the model wrote a stringy risk.
    #[serde(default)]
    pub risks: Vec<String>,
    #[serde(default)]
    pub requirements: Vec<String>,
    /// What remains uncertain after analysis — drives Phase 5's clarification edge later.
    #[serde(default)]
    pub residual_risk: String,
    /// Whether carrying out this plan means editing code in this repository.
    ///
    /// The one signal that decides whether the fork runs at all. A plain bool on purpose: the
    /// structured `{description, severity}` risk on this same type had to be reverted because the
    /// model wrote values that failed schema validation, and a flag has no such failure mode.
    ///
    /// Defaults to `true` when the model omits it, so a missing field costs a fork rather than
    /// silently skipping the work. Note that `touched` is NOT this signal — it lists what the
    /// eventual change would touch, which a research task has plenty of.
    #[serde(default = "changes_code_by_default")]
    pub changes_code: bool,
    /// What must run and pass for this change to be believed done, as ordered named steps.
    ///
    /// The analyst decides because "done" varies by change, not just by repository: a refactor is
    /// accepted by the existing suite, a new endpoint is not accepted until something exercises
    /// the endpoint. Empty falls back to `[sandbox] test_command`, and `[sandbox] pin_acceptance`
    /// ignores this entirely.
    ///
    /// Frozen once the fork starts. A revision (see `previous`) amends requirements and must never
    /// touch this: a change that can move the bar it is judged against is not judged.
    #[serde(default)]
    pub acceptance: Vec<ratatoskr_core::AcceptanceStep>,
}

/// A plan is assumed to involve a code change unless the analyst says otherwise. The failure this
/// guards is asymmetric: wrongly running the fork wastes a sandboxed test run, wrongly skipping it
/// drops the work the run was asked to do.
fn changes_code_by_default() -> bool {
    true
}

/// The analyst node: a stronger agent restricted to impact/lookup tools.
pub struct AnalystNode {
    pub route: ModelRoute,
    pub tools: ToolSet,
    pub policy: Option<std::sync::Arc<dyn ratatoskr_core::ToolPolicy>>,
    pub max_turns: Option<usize>,
    /// Ruleset `systemPrompt`; replaces [`PREAMBLE`] when set.
    pub system_prompt: Option<String>,
    /// What the plugins this node binds contribute to it.
    pub plugins: crate::NodePlugins,
    /// Where this node reports what its turn cost, for the checkpoint the executor writes.
    pub ledger: Option<std::sync::Arc<ratatoskr_agent::RunLedger>>,
    /// The repository its built-in file tools read within.
    pub files: Option<std::path::PathBuf>,
}

impl Node for AnalystNode {
    type Input = AnalystInput;
    type Output = AnalystOutput;

    fn name(&self) -> &'static str {
        "analyst"
    }

    async fn run(
        &self,
        input: AnalystInput,
        _run_state: &RunState,
    ) -> Result<AnalystOutput, NodeError> {
        let prompt = render_prompt(&input);
        let raw = ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "analyst",
            route: &self.route,
            preamble: &crate::effective_preamble(
                PREAMBLE,
                self.system_prompt.as_deref(),
                self.plugins.context.as_deref(),
            ),
            question: &prompt,
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(AnalystOutput),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            // Analyst is the clarification terminus — it answers other nodes but never asks.
            clarifier: None,
            observer: self.plugins.observer.clone(),
            skills: crate::skills::loaded(&self.plugins.skills),
            files: self.files.clone(),
            // Reads and edits, but runs nothing.
            shell: None,
            ledger: self.ledger.clone(),
            produces: Some(
                "an impact summary, the symbols and paths touched, risks, the concrete requirements the implementation must satisfy, and the acceptance steps that prove it done",
            ),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("analyst agent failed: {e}")))?;

        parse_validated::<AnalystOutput>(&raw)
    }
}

/// Fold the issue + upstream outputs into the analyst's prompt.
fn render_prompt(input: &AnalystInput) -> String {
    let mut s = String::new();
    if input.is_revision() {
        s.push_str(
            "THIS IS A REVISION. A change was implemented against your previous plan and reviewed. \
             The review found faults it judged to be in the PLAN rather than in the code — the \
             requirement was wrong, missing, or impossible as written, so re-implementing it will \
             not help.\n\n\
             Decide, for each finding: does the plan need to change, or was the plan right and the \
             implementation simply did not follow it? Amend the requirements where they were \
             wrong. Where they were right, restate them unchanged — repeating a correct \
             requirement is how you say the fault was in the execution.\n\n\
             Keep everything that still holds. You are amending a plan, not writing a new one.\n\n",
        );
    }
    let _ = write!(s, "ISSUE:\n{}\n\n", input.issue);
    // First, because it is the one section written for a reader about to plan rather than a record
    // of what was found.
    if !input.brief.is_empty() {
        let _ = write!(s, "WHAT BEARS ON THIS:\n{}\n\n", input.brief);
    }
    if !input.constraints.is_empty() {
        s.push_str("CONSTRAINTS THIS MUST RESPECT:\n");
        for c in &input.constraints {
            let from = match c.from_memory_id.as_str() {
                "" => String::new(),
                id => format!(" [{id}]"),
            };
            let _ = writeln!(s, "- {}{from}", c.says);
        }
        s.push('\n');
    }
    if let Some(previous) = &input.previous {
        let _ = write!(s, "YOUR PREVIOUS PLAN:\n{}\n", previous.impact_summary);
        if !previous.requirements.is_empty() {
            s.push_str("Requirements you set:\n");
            for r in &previous.requirements {
                let _ = writeln!(s, "- {r}");
            }
        }
        s.push('\n');
    }
    if !input.findings.is_empty() {
        s.push_str("WHAT THE REVIEW FOUND:\n");
        for f in &input.findings {
            let where_ = match f.file.as_str() {
                "" => String::new(),
                file => format!(" ({file})"),
            };
            let _ = writeln!(s, "- [{:?}]{} {}", f.severity, where_, f.summary);
            let _ = writeln!(s, "  Fails when: {}", f.failure_scenario);
        }
        s.push('\n');
    }
    let _ = write!(s, "SCOUT SUMMARY:\n{}\n\n", input.scout.papertrail_summary);

    if !input.scout.related_items.is_empty() {
        s.push_str("RELATED ITEMS:\n");
        for item in &input.scout.related_items {
            let _ = writeln!(
                s,
                "- [{}] {} — {}",
                item.item_key, item.title, item.relation
            );
        }
        s.push('\n');
    }

    if !input.memory.memories.is_empty() {
        s.push_str("REPO MEMORIES:\n");
        for m in &input.memory.memories {
            let detail = m.summary.as_deref().unwrap_or(&m.body);
            let _ = writeln!(s, "- ({}) {}: {}", m.kind, m.title, detail);
        }
        s.push('\n');
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_canned_analyst_response() {
        let raw = r#"{
            "impact_summary": "touches the store",
            "touched": ["ratatoskr-store::Store"],
            "risks": ["medium: lock contention"],
            "requirements": ["keep single-writer"],
            "residual_risk": "throughput under load unknown"
        }"#;
        let out = parse_validated::<AnalystOutput>(raw).unwrap();
        assert_eq!(out.touched, ["ratatoskr-store::Store"]);
        assert_eq!(out.risks[0], "medium: lock contention");
    }

    #[test]
    fn rejects_a_malformed_analyst_response() {
        // Missing the essential `impact_summary` → rejected.
        let raw = r#"{"touched":[],"risks":[],"requirements":[],"residual_risk":"none"}"#;
        assert!(matches!(
            parse_validated::<AnalystOutput>(raw),
            Err(NodeError::InvalidOutput(_))
        ));
        // Wrong type for risks (object, not array) → also rejected.
        let raw = r#"{"impact_summary":"x","risks":{"description":"d"}}"#;
        assert!(matches!(
            parse_validated::<AnalystOutput>(raw),
            Err(NodeError::InvalidOutput(_))
        ));
    }

    #[test]
    fn an_omitted_changes_code_costs_a_fork_rather_than_skipping_the_work() {
        // The failure is asymmetric: wrongly forking wastes a sandboxed test run, wrongly skipping
        // drops the work the run was asked to do. A model that never learns the field must land on
        // the wasteful side.
        let raw = r#"{"impact_summary":"x"}"#;
        let out = parse_validated::<AnalystOutput>(raw).unwrap();
        assert!(out.changes_code);
    }

    #[test]
    fn a_research_task_can_say_it_changes_no_code() {
        let raw = r#"{"impact_summary":"answer the question","changes_code":false,
                      "touched":["a.rs","b.rs"]}"#;
        let out = parse_validated::<AnalystOutput>(raw).unwrap();
        assert!(!out.changes_code);
        // `touched` is a relevance list, not a work order — a question about two files is still a
        // question, and reading it as a signal that code changes is how the fork ran on a run that
        // produced an empty diff.
        assert_eq!(out.touched.len(), 2);
    }

    #[test]
    fn the_brief_and_its_constraints_lead_the_analyst_prompt() {
        use crate::context::{Constraint, ContextOutput};
        let context = ContextOutput {
            brief: "The store migrates by ALTER, not by rewriting schema.sql.".into(),
            constraints: vec![
                Constraint {
                    says: "a new column needs both schema.sql and ADDED_COLUMNS".into(),
                    from_memory_id: "mem_1".into(),
                },
                Constraint {
                    says: "read from the code, not a memory".into(),
                    from_memory_id: String::new(),
                },
            ],
            scout: ScoutOutput {
                related_items: Vec::new(),
                papertrail_summary: "nothing in the tracker".into(),
            },
            memory: MemoryOutput {
                memories: Vec::new(),
            },
        };
        let input = AnalystInput::from_context("Add repo_sha.".into(), context);
        let prompt = render_prompt(&input);

        // The distillation is written for a reader about to plan; the record of what was found is
        // not, so it comes first.
        let brief_at = prompt.find("WHAT BEARS ON THIS").unwrap();
        assert!(brief_at < prompt.find("SCOUT SUMMARY").unwrap());
        assert!(prompt.contains("ALTER, not by rewriting"));

        // A constraint carries its source so the analyst can check the wording against it.
        assert!(prompt.contains("[mem_1]"), "{prompt}");
        // One drawn from the code has no id to cite, and does not get an empty bracket.
        assert!(!prompt.contains("[]"), "{prompt}");
    }

    #[test]
    fn a_hand_composed_analyst_input_still_works_without_a_brief() {
        // A script that composes `scout()` and `memory()` itself hands over the evidence with no
        // synthesis. That has to keep working, not fail schema validation.
        let raw = r#"{"issue":"x","scout":{"papertrail_summary":"s"},"memory":{"memories":[]}}"#;
        let input: AnalystInput = serde_json::from_str(raw).unwrap();
        assert!(input.brief.is_empty());
        assert!(input.constraints.is_empty());
        let prompt = render_prompt(&input);
        assert!(!prompt.contains("WHAT BEARS ON THIS"));
    }
}
