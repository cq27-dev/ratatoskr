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

use crate::testrun::{Characterizer, by_exit_code, run_acceptance};

/// rag-rat tools the classifier may use to inspect the failing tests' code.
pub const CLASSIFIER_TOOLS: &[&str] = &["symbol_lookup", "semantic_search"];

const CLASSIFY_PREAMBLE: &str = include_str!("../prompts/redteam-classifier.md");

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
    /// Tests written for this change before the code was, when the red team had a route to write
    /// them with.
    #[serde(default)]
    pub authored: Option<AuthoredTests>,
    pub failing_tests: Vec<String>,
    /// How many checks passed. Only the count is carried: nothing reads a passing check's name,
    /// and a suite of several hundred costs the characterizer more output than the rest of the
    /// pipeline combined to write out.
    #[serde(default)]
    pub passed_tests: usize,
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
    /// Where this node reports what its turn cost, for the checkpoint the executor writes.
    pub ledger: Option<std::sync::Arc<ratatoskr_agent::RunLedger>>,
    /// The repository its built-in file tools read within.
    pub files: Option<std::path::PathBuf>,
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
            skills: crate::skills::loaded(&self.plugins.skills),
            files: self.files.clone(),
            // Reads and edits, but runs nothing.
            shell: None,
            conversation: None,
            ledger: self.ledger.clone(),
            produces: Some(
                "a classification of each baseline test failure as flaky or real, with the reason",
            ),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("red-team classifier failed: {e}")))?;
        Ok(parse_validated::<Classification>(&raw)?.classifications)
    }
}

/// The red-team node: run the baseline checkout's tests in a sandbox, optionally classify failures.
/// What the red team wrote, and what it says those tests cover.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AuthoredTests {
    /// Files written or extended, as repository-relative paths.
    #[serde(default)]
    pub files: Vec<String>,
    /// The tests written, named as the test runner will report them.
    ///
    /// Not decoration: a test written before the code fails in the baseline by construction, so
    /// convergence — which asks only whether anything *newly* fails — would pass a change that
    /// ignored every one of them. Naming them is what lets the run require them to pass.
    #[serde(default)]
    pub tests: Vec<String>,
    /// One line per file on what it covers, or why nothing was written.
    #[serde(default)]
    pub covers: String,
}

const AUTHOR_PREAMBLE: &str = include_str!("../prompts/redteam-author.md");

/// The red team's other half: writing the tests the change will be judged against.
///
/// Separate from the classifier because they are different jobs on different sides of the run —
/// one reads a baseline that already happened, the other writes what has to become true. Both are
/// opt-in on having a route.
pub struct TestAuthor {
    pub route: ModelRoute,
    pub tools: ToolSet,
    pub policy: Option<std::sync::Arc<dyn ratatoskr_core::ToolPolicy>>,
    pub max_turns: Option<usize>,
    pub system_prompt: Option<String>,
    pub plugins: crate::NodePlugins,
    pub ledger: Option<std::sync::Arc<ratatoskr_agent::RunLedger>>,
}

impl TestAuthor {
    /// Write tests for `interface` into `worktree`, before any code exists to satisfy them.
    pub async fn write(
        &self,
        worktree: &std::path::Path,
        issue: &str,
        interface: &[crate::analyst::InterfaceItem],
    ) -> Result<AuthoredTests, NodeError> {
        let raw = ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "redteam",
            route: &self.route,
            preamble: &crate::effective_preamble(
                AUTHOR_PREAMBLE,
                self.system_prompt.as_deref(),
                self.plugins.context.as_deref(),
            ),
            question: &author_prompt(issue, interface),
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(AuthoredTests),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            clarifier: None,
            observer: self.plugins.observer.clone(),
            skills: crate::skills::loaded(&self.plugins.skills),
            // Rooted at the worktree: the tests have to land where the implementer will meet them.
            files: Some(worktree.to_path_buf()),
            // No shell. Writing a test is not running one, and the baseline run is what says
            // whether these fail — which at this point they should.
            shell: None,
            conversation: None,
            ledger: self.ledger.clone(),
            produces: Some("tests covering the contracted interface, written before the code"),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("test author failed: {e}")))?;
        parse_validated::<AuthoredTests>(&raw)
    }
}

/// What the author is given: the task for context, and the contract it writes against.
fn author_prompt(issue: &str, interface: &[crate::analyst::InterfaceItem]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = write!(s, "THE TASK, for context only:\n{issue}\n\n");
    s.push_str(
        "THE INTERFACE. This is the contract, and it is all you get — the code does not exist \
         yet, and the person writing it is working from this same description:\n\n",
    );
    crate::analyst::render_interface(&mut s, interface, "happy", "sad");
    s.push_str(
        "\nWrite tests for these. Follow the repository's own layout and conventions, cover the \
         sad cases as carefully as the happy ones, and change nothing that already exists.",
    );
    s
}

pub struct RedTeamNode {
    pub repo_path: PathBuf,
    pub sandbox: ratatoskr_core::SandboxConfig,
    /// Unique sandbox name for this run.
    pub name: String,
    /// Enabled only when redteam has a route — from `[models.redteam]` or its ruleset.
    pub classifier: Option<RedTeamClassifier>,
    /// What "done" means for this task, from the analyst. The baseline runs exactly what the
    /// post-change run will, or the two sets converge compares are not comparable.
    pub acceptance: Vec<ratatoskr_core::AcceptanceStep>,
    /// Names the checks inside each step. `None` compares at step granularity instead.
    pub characterizer: Option<Characterizer>,
    /// Writes the tests the change is judged against. Opt-in on a `[models.redteam]` route, like
    /// the classifier — without one, the run falls back to whatever tests already exist.
    pub author: Option<TestAuthor>,
}

impl RedTeamNode {
    /// Characterise the baseline and write the change's tests, in one step.
    ///
    /// The two are independent — one reads the repository as it is, the other writes into a
    /// worktree — so they run together. Both finish before the implementer starts, which is the
    /// point: it meets the tests it has to satisfy rather than writing them itself.
    pub async fn run_and_author(
        &self,
        worktree: &std::path::Path,
        issue: &str,
        interface: &[crate::analyst::InterfaceItem],
    ) -> Result<RedTeamOutput, NodeError> {
        let (baseline, authored) =
            tokio::join!(self.run(), self.author_tests(worktree, issue, interface));
        let mut out = baseline?;
        out.authored = authored;
        Ok(out)
    }

    /// Write the tests, when there is an author and something to write against.
    ///
    /// Best-effort, like the classifier: a failed authoring turn leaves the change to be judged by
    /// the tests that already exist, which is where every run stood before this node could write
    /// any. Failing the run instead would trade a weaker judgement for no judgement.
    async fn author_tests(
        &self,
        worktree: &std::path::Path,
        issue: &str,
        interface: &[crate::analyst::InterfaceItem],
    ) -> Option<AuthoredTests> {
        let author = self.author.as_ref()?;
        if interface.is_empty() {
            return None;
        }
        match author.write(worktree, issue, interface).await {
            Ok(written) => {
                tracing::info!(
                    kind = "authored_tests",
                    files = %written.files.join(", "),
                    "the red team wrote the change's tests"
                );
                Some(written)
            }
            Err(e) => {
                tracing::warn!(
                    "writing the change's tests failed; it will be judged by the tests that \
                     already exist: {e}"
                );
                None
            }
        }
    }

    pub async fn run(&self) -> Result<RedTeamOutput, NodeError> {
        let outcomes = run_acceptance(&self.sandbox, &self.name, &self.repo_path, &self.acceptance)
            .await
            .map_err(NodeError::Failed)?;
        let results = match &self.characterizer {
            Some(c) => c.read(&outcomes).await,
            None => by_exit_code(&outcomes),
        };

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
            authored: None,
            failing_tests: results.failing,
            passed_tests: results.passed,
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

    fn item() -> crate::analyst::InterfaceItem {
        crate::analyst::InterfaceItem {
            name: "store::prune".into(),
            shape: "pub async fn prune(&self, older_than: Duration) -> Result<u64, StoreError>"
                .into(),
            happy: vec!["removes rows older than the cutoff and returns how many".into()],
            sad: vec!["a zero duration removes nothing and returns 0".into()],
        }
    }

    #[test]
    fn the_author_is_given_the_contract_and_told_the_code_does_not_exist() {
        // The whole reason this node writes the tests: it works from the contract, so its tests
        // can be wrong about the implementation and still right about the requirement.
        let p = author_prompt("Prune old rows", &[item()]);
        assert!(p.contains("store::prune"), "the surface");
        assert!(p.contains("older_than: Duration"), "and its exact shape");
        assert!(p.contains("happy: removes rows older than the cutoff"));
        assert!(p.contains("sad: a zero duration removes nothing"));
        assert!(
            p.contains("the code does not exist"),
            "why it cannot read it"
        );
    }

    #[test]
    fn the_author_preamble_says_what_makes_these_tests_different() {
        let p = AUTHOR_PREAMBLE.to_ascii_lowercase();
        // Written before the code, so failing now is expected and must not be papered over.
        assert!(
            p.contains("fail now"),
            "what a good test does at this point"
        );
        assert!(p.contains("asserting nothing"), "and what it must not do");
        // It adds; it does not adjust the judge.
        assert!(p.contains("do not modify tests that are already there"));
    }

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
