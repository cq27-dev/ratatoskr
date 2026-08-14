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

use ratatoskr_graph::{NodeError, parse_validated};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::analyst::AnalystOutput;

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
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct VerifierOutput {
    #[serde(default)]
    pub findings: Vec<Finding>,
    /// What was checked, in a line or two. Present even when nothing was found — "no findings"
    /// with no account of what was looked at is indistinguishable from a verifier that did nothing.
    #[serde(default)]
    pub assessment: String,
    /// What this pass could not reach, named so the next one can continue over it. Empty for a
    /// review that finished.
    ///
    /// A review cut short — by a change larger than the context it was given, a tool that failed,
    /// a sweep it ran out of room for — otherwise returns exactly what a clean review returns: no
    /// findings. The loop then reads it as "nothing wrong" and the run converges on a review that
    /// never happened. That is the one failure mode that gets WORSE as reviews get more thorough,
    /// because the more a verifier is asked to check, the more often "I ran out of room" is the
    /// honest answer.
    ///
    /// Naming areas rather than setting a flag, because the continuation has to be given something
    /// to continue over — and it makes honesty concrete: an incomplete answer that says what it
    /// missed costs a further pass, while claiming completeness it does not have is what this is
    /// meant to be cheaper than.
    #[serde(default)]
    pub unchecked: Vec<String>,
}

impl VerifierOutput {
    /// The findings that block, most severe first.
    /// Whether this review reached the end of what it set out to check.
    ///
    /// The one answer to it. A review that did not is not a clean one however few findings it
    /// carries, and nothing may read an empty `findings` as a verdict without asking this first.
    pub fn complete(&self) -> bool {
        self.unchecked.is_empty()
    }

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
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
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
    /// What an earlier pass in this run said it could not reach, when this call is continuing that
    /// review. Empty for a first pass, and for a continuation of a review that finished.
    ///
    /// Carried as input rather than left to the prompt, because a continuation that re-reviewed the
    /// whole change would spend a pass re-deriving what the last one already established — and
    /// would be as likely to run out of room in the same place.
    #[serde(default)]
    pub unchecked: Vec<String>,
}

/// Run and schema-validate a judgement node.
pub(crate) async fn run_judgement<T: DeserializeOwned + JsonSchema>(
    run: ratatoskr_agent::NodeRun<'_>,
    name: &str,
) -> Result<T, NodeError> {
    let raw = ratatoskr_agent::run_structured(run)
        .await
        .map_err(|e| NodeError::Failed(format!("{name} agent failed: {e}")))?;
    parse_validated::<T>(&raw)
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
            ..Default::default()
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
