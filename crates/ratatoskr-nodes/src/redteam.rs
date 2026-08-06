//! Red-team: characterize the baseline test run. No LLM in the core path — it mounts the existing
//! checkout into a sandbox, runs the repo's tests, and parses pass/fail deterministically. That
//! deterministic characterization is what converge compares the implementer's run against.
//!
//! An OPTIONAL classifier (enabled only when redteam has a model route — from `[models.redteam]` or
//! its ruleset) adds one LLM
//! pass labeling each baseline failure flaky vs real — a separate, additive field. The strict
//! pass/fail is never touched by it.

use std::path::PathBuf;

use ratatoskr_core::ModelRoute;
use ratatoskr_graph::{NodeError, parse_validated};
use ratatoskr_mcp::ToolSet;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::testrun::run_tests;

/// rag-rat tools the classifier may use to inspect the failing tests' code.
pub const CLASSIFIER_TOOLS: &[&str] = &["symbol_lookup", "semantic_search"];

const CLASSIFY_PREAMBLE: &str = "You classify failing tests. For each test, decide whether it is \
    \"flaky\" (fails non-deterministically — timing, ordering, environment, network — and would \
    likely pass on a retry) or \"real\" (a genuine, reproducible failure in the code under test). \
    Base the call on the test output and, if useful, the test's code. Be conservative: only call \
    something flaky when the evidence points to non-determinism.";

/// One baseline failure's classification. Additive context, not part of the strict pass/fail.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FailureClassification {
    pub test: String,
    /// "flaky" or "real" (anything else is treated as unknown by consumers).
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub reason: String,
}

/// The compose model's output.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Classification {
    #[serde(default)]
    classifications: Vec<FailureClassification>,
}

/// Deterministic baseline characterization (strict schema — built from a real test run, not an LLM).
/// `classifications` is optional/additive: empty unless a redteam classifier ran (it runs when
/// redteam has a route from `[models.redteam]` or its ruleset).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RedTeamOutput {
    pub failing_tests: Vec<String>,
    pub passing_tests: Vec<String>,
    pub exit_code: i32,
    #[serde(default)]
    pub classifications: Vec<FailureClassification>,
}

/// Optional LLM classifier for baseline failures.
pub struct RedTeamClassifier {
    pub route: ModelRoute,
    pub tools: ToolSet,
    pub policy: Option<std::sync::Arc<dyn ratatoskr_core::ToolPolicy>>,
    pub max_turns: Option<usize>,
    pub clarifier: Option<std::sync::Arc<dyn ratatoskr_agent::Clarifier>>,
    /// Ruleset `systemPrompt`; replaces [`CLASSIFY_PREAMBLE`] when set.
    pub system_prompt: Option<String>,
    /// What the plugins this node binds contribute to it.
    pub plugins: crate::NodePlugins,
}

impl RedTeamClassifier {
    async fn classify(
        &self,
        failing: &[String],
        raw_output: &str,
    ) -> Result<Vec<FailureClassification>, NodeError> {
        let prompt = format!(
            "These tests fail in the current baseline (before any change):\n{}\n\nTest output:\n{}\n\n\
             Classify each as \"flaky\" or \"real\" with a one-line reason.",
            failing.join("\n"),
            truncate(raw_output, 6000)
        );
        let raw = ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "redteam",
            route: &self.route,
            preamble: &crate::effective_preamble(
                CLASSIFY_PREAMBLE,
                self.system_prompt.as_deref(),
                self.plugins.context.as_deref(),
            ),
            question: &prompt,
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(Classification),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            clarifier: self.clarifier.clone(),
            observer: self.plugins.observer.clone(),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("red-team classifier failed: {e}")))?;
        Ok(parse_validated::<Classification>(&raw)?.classifications)
    }
}

/// The red-team node: run the baseline checkout's tests in a sandbox, optionally classify failures.
pub struct RedTeamNode {
    pub repo_path: PathBuf,
    pub sandbox: ratatoskr_core::SandboxConfig,
    /// Unique sandbox name for this run.
    pub name: String,
    /// Enabled only when redteam has a route — from `[models.redteam]` or its ruleset.
    pub classifier: Option<RedTeamClassifier>,
}

impl RedTeamNode {
    pub async fn run(&self) -> Result<RedTeamOutput, NodeError> {
        let results = run_tests(&self.sandbox, &self.name, &self.repo_path)
            .await
            .map_err(NodeError::Failed)?;

        // Classification is best-effort: never let it fail the deterministic characterization.
        let classifications = match (&self.classifier, results.failing.is_empty()) {
            (Some(c), false) => c
                .classify(&results.failing, &results.raw_output)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!("red-team classification failed: {e}");
                    Vec::new()
                }),
            _ => Vec::new(),
        };

        Ok(RedTeamOutput {
            failing_tests: results.failing,
            passing_tests: results.passing,
            exit_code: results.exit_code,
            classifications,
        })
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut i = max;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    format!("{}…", &s[..i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_classifier_response() {
        let raw = r#"{"classifications":[
            {"test":"a::flaps","category":"flaky","reason":"depends on wall-clock timing"},
            {"test":"b::broken","category":"real"}
        ]}"#;
        let c = parse_validated::<Classification>(raw).unwrap();
        assert_eq!(c.classifications.len(), 2);
        assert_eq!(c.classifications[0].category, "flaky");
        // `reason` defaults when omitted.
        assert_eq!(c.classifications[1].reason, "");
    }
}
