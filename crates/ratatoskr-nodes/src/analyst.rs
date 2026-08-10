//! Analyst: given the issue, scout's findings, and repo memories, assess impact and risk.

#[cfg(test)]
use ratatoskr_graph::{NodeError, parse_validated};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::memory::MemoryOutput;
use crate::scout::ScoutOutput;

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
    /// The surface the change is contracted to have, and what it should do when used.
    ///
    /// This is what lets the tests be written by someone other than the author. The red team turns
    /// it into tests and the implementer builds against it, from the same description — so the
    /// tests are not shaped around the implementation that happens to appear, which is the failure
    /// an author writing their own tests cannot see in themselves.
    ///
    /// Empty when the change has no callable surface — an internal refactor, a doc fix. That is an
    /// ordinary answer, and better than a contract invented to fill the field.
    #[serde(default)]
    pub interface: Vec<InterfaceItem>,
}

/// One piece of surface the change adds or alters, with what it owes its caller.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct InterfaceItem {
    /// What it is, as a caller names it: `path::function`, a CLI flag, an HTTP route, a config key.
    pub name: String,
    /// Its shape after the change — the signature, the parameters and their types, the fields.
    /// Enough that someone could call it without reading the implementation, because they cannot:
    /// it does not exist yet.
    pub shape: String,
    /// What it does when used correctly. One expectation per entry, each concrete enough to be
    /// checked: the input, and the result it must produce.
    #[serde(default)]
    pub happy: Vec<String>,
    /// What it does when misused, or when the world does not cooperate — a bad argument, a missing
    /// file, a value at its limit. Same standard: an input, and the result it must produce.
    #[serde(default)]
    pub sad: Vec<String>,
}

/// A plan is assumed to involve a code change unless the analyst says otherwise. The failure this
/// guards is asymmetric: wrongly running the fork wastes a sandboxed test run, wrongly skipping it
/// drops the work the run was asked to do.
fn changes_code_by_default() -> bool {
    true
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
}
