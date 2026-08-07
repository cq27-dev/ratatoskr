//! Verifier: read the change and say whether it is the right one.
//!
//! Converge asks one question — did the failing-test set grow. A change can be green and still be
//! wrong: a test weakened until it passes, a requirement misread, a resource leaked that nothing
//! exercises, a bug reintroduced that a repo memory already warns about. This node asks the
//! question the tests cannot.
//!
//! It runs **after** the test gate is clean, never instead of it. A model's judgement is weaker
//! evidence than a test result, and treating it as a substitute is how a confident wrong patch gets
//! approved.

use std::fmt::Write as _;

use ratatoskr_core::ModelRoute;
use ratatoskr_graph::{NodeError, parse_validated};
use ratatoskr_mcp::ToolSet;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::analyst::AnalystOutput;

/// rag-rat tools the verifier may use. Read-only and grounding: it needs to check the change
/// against current code and against what the repo already knows, not to explore freely.
pub const VERIFIER_TOOLS: &[&str] = &[
    "semantic_search",
    "symbol_lookup",
    "impact_surface",
    "memory_search",
];

const PREAMBLE: &str = include_str!("../prompts/verifier.md");

/// How bad a finding is. Ordered: `P1 < P2 < P3` by variant order, so "at least as severe as" is a
/// `<=` on the enum.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "UPPERCASE")]
pub enum Severity {
    /// Must fix: a correctness bug, a security issue, a silent regression, a defeated test.
    P1,
    /// Should fix: a missed case, a misleading name or doc, a poor error.
    P2,
    /// A nit.
    P3,
}

impl Severity {
    /// Whether this finding is at least as severe as `threshold`, and so blocks.
    pub fn blocks_at(self, threshold: Severity) -> bool {
        self <= threshold
    }
}

/// Who a finding belongs to.
///
/// The routing decision, and the reason the verifier makes it rather than the executor: it holds
/// the diff and the requirements together, so telling "the code does not match the plan" from "the
/// plan was wrong" costs nothing extra here and would cost a whole node's call anywhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum FindingKind {
    /// The plan was right; the code does not match it. The implementer can fix this alone.
    Execution,
    /// The requirement was wrong, missing, or impossible as written. Re-implementing it will not
    /// help, so it goes back to the analyst.
    Plan,
}

/// One thing wrong with the change.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Finding {
    pub severity: Severity,
    pub kind: FindingKind,
    #[serde(default)]
    pub file: String,
    #[serde(default)]
    pub line: Option<u32>,
    pub summary: String,
    /// Concrete input or state, and the wrong result it produces. Required: a finding without one
    /// is a preference, and preferences are what turn a review gate into a style argument.
    pub failure_scenario: String,
}

/// What the verifier concluded.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VerifierOutput {
    #[serde(default)]
    pub findings: Vec<Finding>,
    /// What was checked, in a line or two. Present even when nothing was found — "no findings"
    /// with no account of what was looked at is indistinguishable from a verifier that did nothing.
    #[serde(default)]
    pub assessment: String,
}

impl VerifierOutput {
    /// The findings that block, most severe first.
    pub fn blocking(&self, threshold: Severity) -> Vec<&Finding> {
        let mut found: Vec<&Finding> = self
            .findings
            .iter()
            .filter(|f| f.severity.blocks_at(threshold))
            .collect();
        found.sort_by_key(|f| f.severity);
        found
    }
}

/// What the verifier is given: the change, and what it was for.
pub struct VerifierInput {
    pub issue: String,
    pub analyst: AnalystOutput,
    /// The change as a patch. Not a `--stat`: a summary cannot show a weakened assertion.
    pub diff: String,
    pub touched_files: Vec<String>,
    /// What earlier passes in this run already found, oldest first.
    ///
    /// Without it every pass reviews as if it were the first, and the one pattern that matters
    /// most is invisible: a finding that exists *because* of the fix for the last one. That is not
    /// a fresh defect to patch, it is the plan being wrong, and saying so is the only way the run
    /// stops trading one symptom for the next.
    pub previous_findings: Vec<Finding>,
}

/// The verifier node: a reviewer restricted to read-only grounding tools.
pub struct VerifierNode {
    pub route: ModelRoute,
    pub tools: ToolSet,
    pub policy: Option<std::sync::Arc<dyn ratatoskr_core::ToolPolicy>>,
    pub max_turns: Option<usize>,
    /// Ruleset `systemPrompt`; replaces [`PREAMBLE`] when set.
    pub system_prompt: Option<String>,
    pub plugins: crate::NodePlugins,
    pub ledger: Option<std::sync::Arc<ratatoskr_agent::RunLedger>>,
    pub files: Option<std::path::PathBuf>,
}

impl VerifierNode {
    pub async fn run(&self, input: VerifierInput) -> Result<VerifierOutput, NodeError> {
        // Nothing to review is not a clean review. Saying "no findings" about an empty diff would
        // report the change as verified when there was no change.
        if input.diff.trim().is_empty() {
            return Ok(VerifierOutput {
                findings: Vec::new(),
                assessment: "there was no diff to review".to_string(),
            });
        }

        let raw = ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "verifier",
            route: &self.route,
            preamble: &crate::effective_preamble(
                PREAMBLE,
                self.system_prompt.as_deref(),
                self.plugins.context.as_deref(),
            ),
            question: &render_prompt(&input),
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(VerifierOutput),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            // The verifier judges; it does not negotiate. Asking the analyst what it meant would
            // let the node being reviewed-for shape the review.
            clarifier: None,
            observer: self.plugins.observer.clone(),
            skills: crate::skills::loaded(&self.plugins.skills),
            files: self.files.clone(),
            // Reads the diff; runs nothing — the acceptance run already happened.
            shell: None,
            push: None,
            conversation: None,
            ledger: self.ledger.clone(),
            produces: Some(
                "findings on the diff — each with a severity, a plan/execution kind, and a concrete failure scenario — or none",
            ),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("verifier agent failed: {e}")))?;

        parse_validated::<VerifierOutput>(&raw)
    }
}

fn render_prompt(input: &VerifierInput) -> String {
    let mut s = String::new();
    let _ = write!(s, "TASK:\n{}\n\n", input.issue);

    let a = &input.analyst;
    if !a.requirements.is_empty() {
        s.push_str("REQUIREMENTS THE CHANGE MUST SATISFY:\n");
        for r in &a.requirements {
            let _ = writeln!(s, "- {r}");
        }
        s.push('\n');
    }
    if !a.impact_summary.is_empty() {
        let _ = write!(s, "EXPECTED IMPACT:\n{}\n\n", a.impact_summary);
    }
    if !a.risks.is_empty() {
        s.push_str("RISKS THE PLAN FLAGGED — check whether the change hit any:\n");
        for r in &a.risks {
            let _ = writeln!(s, "- {r}");
        }
        s.push('\n');
    }
    if !input.touched_files.is_empty() {
        let _ = write!(s, "FILES CHANGED: {}\n\n", input.touched_files.join(", "));
    }
    if !input.previous_findings.is_empty() {
        s.push_str(
            "WHAT YOU ALREADY FOUND IN THIS RUN, and the implementer has since tried to fix. Read \
             these before the diff. If what you are about to report exists because of the fix for \
             one of them, the plan is wrong and you must say so with kind `plan` — reporting it as \
             another `execution` finding buys one more patch and the next finding after it:\n",
        );
        for f in &input.previous_findings {
            let _ = writeln!(
                s,
                "- [{:?}/{:?}] {}: {}",
                f.severity, f.kind, f.file, f.summary
            );
        }
        s.push('\n');
    }
    let _ = write!(s, "THE CHANGE:\n{}\n", input.diff);
    s
}

/// Turn blocking findings into the correction the implementer is re-driven with.
///
/// The findings go over verbatim rather than summarised: the failure scenario is the part that
/// makes a finding actionable, and a paraphrase of it is what turns a specific defect into vague
/// pressure to change something.
pub fn correction(findings: &[&Finding]) -> String {
    let mut s = String::from(
        "A review of your change found problems that the tests did not catch. Fix each of these \
         without breaking anything that currently passes:\n\n",
    );
    for f in findings {
        let where_ = match (f.file.as_str(), f.line) {
            ("", _) => String::new(),
            (file, Some(line)) => format!(" ({file}:{line})"),
            (file, None) => format!(" ({file})"),
        };
        let _ = writeln!(s, "- [{:?}]{} {}", f.severity, where_, f.summary);
        let _ = writeln!(s, "  Fails when: {}\n", f.failure_scenario);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(severity: Severity, kind: FindingKind) -> Finding {
        Finding {
            severity,
            kind,
            file: "a.rs".into(),
            line: Some(4),
            summary: "summary".into(),
            failure_scenario: "scenario".into(),
        }
    }

    #[test]
    fn a_threshold_blocks_on_that_severity_and_everything_worse() {
        assert!(Severity::P1.blocks_at(Severity::P2));
        assert!(Severity::P2.blocks_at(Severity::P2));
        assert!(
            !Severity::P3.blocks_at(Severity::P2),
            "a nit does not re-drive an implementer session at the default threshold"
        );
        // The stricter setting a repo can opt into.
        assert!(Severity::P3.blocks_at(Severity::P3));
        // And the loosest: only must-fix findings block.
        assert!(!Severity::P2.blocks_at(Severity::P1));
        assert!(Severity::P1.blocks_at(Severity::P1));
    }

    #[test]
    fn blocking_findings_come_back_worst_first() {
        let out = VerifierOutput {
            findings: vec![
                finding(Severity::P3, FindingKind::Execution),
                finding(Severity::P2, FindingKind::Execution),
                finding(Severity::P1, FindingKind::Plan),
            ],
            assessment: String::new(),
        };
        let blocking = out.blocking(Severity::P2);
        assert_eq!(blocking.len(), 2, "the nit is recorded but does not block");
        assert_eq!(blocking[0].severity, Severity::P1);
        assert_eq!(blocking[1].severity, Severity::P2);
        assert!(out.blocking(Severity::P3).len() == 3);
    }

    #[test]
    fn the_correction_carries_the_failure_scenario_verbatim() {
        // What makes a finding actionable is the scenario. Summarising it is how a specific defect
        // becomes vague pressure to change something.
        let f = finding(Severity::P1, FindingKind::Execution);
        let text = correction(&[&f]);
        assert!(text.contains("a.rs:4"));
        assert!(text.contains("P1"));
        assert!(text.contains("scenario"));
    }

    #[test]
    fn a_pass_is_shown_what_the_last_one_found() {
        // The pattern this exists to surface: three passes on one live run found three different
        // defects, each caused by the fix for the one before, severity climbing P2 -> P2 -> P1.
        // A pass that cannot see the earlier findings reviews as if it were the first, reports
        // another execution fault, and buys exactly one more patch.
        let earlier = Finding {
            severity: Severity::P2,
            kind: FindingKind::Execution,
            file: "lib.rs".into(),
            line: None,
            summary: "terminal gate not updated for the new status".into(),
            failure_scenario: "a no-change run skips publishing".into(),
        };
        let input = VerifierInput {
            issue: "i".into(),
            analyst: AnalystOutput {
                impact_summary: String::new(),
                touched: Vec::new(),
                risks: Vec::new(),
                requirements: Vec::new(),
                residual_risk: String::new(),
                changes_code: true,
                acceptance: Vec::new(),
                interface: Vec::new(),
            },
            diff: "--- a
+++ b
"
            .into(),
            touched_files: vec!["lib.rs".into()],
            previous_findings: vec![earlier],
        };
        let p = render_prompt(&input);
        assert!(p.contains("WHAT YOU ALREADY FOUND"), "the history is shown");
        assert!(p.contains("terminal gate not updated"), "and its substance");
        assert!(
            p.contains("the plan is wrong"),
            "with what it means when the new finding follows from the old fix"
        );

        // A first pass has none, and is not told to look for a pattern that cannot exist yet.
        let first = VerifierInput {
            previous_findings: Vec::new(),
            ..input
        };
        assert!(!render_prompt(&first).contains("WHAT YOU ALREADY FOUND"));
    }

    #[test]
    fn severity_and_kind_parse_from_the_shapes_a_model_writes() {
        let raw = r#"{"findings":[{"severity":"P1","kind":"plan","summary":"s",
                      "failure_scenario":"f","file":"a.rs"}],"assessment":"looked at it"}"#;
        let out = parse_validated::<VerifierOutput>(raw).unwrap();
        assert_eq!(out.findings[0].severity, Severity::P1);
        assert_eq!(out.findings[0].kind, FindingKind::Plan);
        assert_eq!(out.findings[0].line, None);

        // A finding with no failure scenario is a preference; the schema refuses it.
        let bad = r#"{"findings":[{"severity":"P1","kind":"plan","summary":"s"}]}"#;
        assert!(parse_validated::<VerifierOutput>(bad).is_err());
    }
}
