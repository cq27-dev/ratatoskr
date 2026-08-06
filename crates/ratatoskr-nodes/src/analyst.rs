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

const PREAMBLE: &str = "You are the analyst in a code-planning pipeline. You are given an issue, \
    the scout's findings, and relevant repo memories. Use `impact_surface` and `symbol_lookup` to \
    determine what this change actually touches and its blast radius — call the tools, don't guess. \
    Produce: an impact summary, the specific symbols/paths touched, a list of risks (each a short \
    line — lead with the severity if it's clear-cut), a list of concrete requirements the \
    implementation must satisfy, and a residual-risk note capturing what remains uncertain or \
    unknown after your analysis. You are also the pipeline's fallback answerer: when another node \
    cannot resolve something on its own, its question routes to you, so hold clear, present-tense \
    judgments about the change that you can share when asked.";

/// Input to the analyst: the issue plus the two upstream node outputs.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AnalystInput {
    pub issue: String,
    pub scout: ScoutOutput,
    pub memory: MemoryOutput,
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
}

/// The analyst node: a stronger agent restricted to impact/lookup tools.
pub struct AnalystNode {
    pub route: ModelRoute,
    pub tools: ToolSet,
    pub policy: Option<std::sync::Arc<dyn ratatoskr_core::ToolPolicy>>,
    pub max_turns: Option<usize>,
    /// Ruleset `systemPrompt`; replaces [`PREAMBLE`] when set.
    pub system_prompt: Option<String>,
    /// Session context contributed by plugins, prefixed to whichever preamble applies.
    pub context: Option<String>,
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
                self.context.as_deref(),
            ),
            question: &prompt,
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(AnalystOutput),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            // Analyst is the clarification terminus — it answers other nodes but never asks.
            clarifier: None,
        })
        .await
        .map_err(|e| NodeError::Failed(format!("analyst agent failed: {e}")))?;

        parse_validated::<AnalystOutput>(&raw)
    }
}

/// Fold the issue + upstream outputs into the analyst's prompt.
fn render_prompt(input: &AnalystInput) -> String {
    let mut s = String::new();
    let _ = write!(s, "ISSUE:\n{}\n\n", input.issue);
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
}
