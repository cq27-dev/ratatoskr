//! Publisher: deliver what a run made to where a person will find it.
//!
//! The bookkeeper records what a run *learned*; nothing delivered what it *made*. A run whose whole
//! deliverable was an expanded issue description finished with that description sitting in SQLite,
//! and the person who started it had to go and find it.
//!
//! Runs alongside the bookkeeper rather than after it — one writes to the memory graph, the other
//! to the tracker, and neither needs the other's result.
//!
//! Off unless configured. This is the only node that acts outside this machine, and a run that
//! opens a pull request nobody expected is worse than one that publishes nothing.

use std::fmt::Write as _;

use ratatoskr_core::ModelRoute;
use ratatoskr_graph::{NodeError, parse_validated};
use ratatoskr_mcp::ToolSet;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::analyst::AnalystOutput;
use crate::implementer::ImplementerOutput;

const PREAMBLE: &str = include_str!("../prompts/publisher.md");

/// What the publisher did.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PublisherOutput {
    /// `pull_request`, `comment`, `both`, or `none`. Free text rather than an enum: nothing
    /// branches on it, and a model writing an unexpected word should not fail a run whose work is
    /// already done and whose `reasoning` says what happened.
    pub action: String,
    /// Where it landed, when it landed anywhere.
    #[serde(default)]
    pub url: String,
    /// Why this was the right form — and, when nothing was published, the whole result.
    pub reasoning: String,
}

/// What the publisher is given: the run, and what it produced.
pub struct PublisherInput {
    pub issue: String,
    pub analyst: AnalystOutput,
    /// `None` for a run that changed no code — the case where a comment is the only sensible form.
    pub implementer: Option<ImplementerOutput>,
    /// The run's terminal status, so the write does not overclaim what the acceptance run showed.
    pub status: String,
    pub iterations: u32,
    /// What the last review still objected to, if there was a review.
    ///
    /// The publisher is told to say what is unresolved, and until this existed it had no way to
    /// know. A run can end with its tests green and its review unsatisfied — that is exactly what
    /// `max_iterations_reached` means — and a pull request written from the test result alone
    /// reads as a clean landing.
    pub unresolved: Vec<crate::verifier::Finding>,
}

/// The publisher node.
pub struct PublisherNode {
    pub route: ModelRoute,
    pub tools: ToolSet,
    pub policy: Option<std::sync::Arc<dyn ratatoskr_core::ToolPolicy>>,
    pub max_turns: Option<usize>,
    /// Ruleset `systemPrompt`; replaces [`PREAMBLE`] when set.
    pub system_prompt: Option<String>,
    pub plugins: crate::NodePlugins,
    pub ledger: Option<std::sync::Arc<ratatoskr_agent::RunLedger>>,
    /// The repository `gh` runs in. Also what roots the file tools it reads the diff with.
    pub files: Option<std::path::PathBuf>,
    /// The one branch this run may push, when it has one. `None` leaves the node without the tool.
    pub push: Option<ratatoskr_agent::publish::PushAccess>,
}

impl PublisherNode {
    pub async fn run(&self, input: PublisherInput) -> Result<PublisherOutput, NodeError> {
        let raw = ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "publisher",
            route: &self.route,
            preamble: &crate::effective_preamble(
                PREAMBLE,
                self.system_prompt.as_deref(),
                self.plugins.context.as_deref(),
            ),
            question: &render_prompt(&input),
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(PublisherOutput),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            // It runs after everything that could answer it has finished.
            clarifier: None,
            observer: self.plugins.observer.clone(),
            skills: crate::skills::loaded(&self.plugins.skills, "publisher"),
            files: self.files.clone(),
            // Reads and edits, but runs nothing.
            shell: None,
            push: self.push.clone(),
            conversation: None,
            ledger: self.ledger.clone(),
            produces: Some("what was published, where, and why"),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("publisher agent failed: {e}")))?;

        parse_validated::<PublisherOutput>(&raw)
    }
}

fn render_prompt(input: &PublisherInput) -> String {
    let mut s = String::new();
    let _ = write!(s, "THE TASK:\n{}\n\n", input.issue);
    let _ = writeln!(
        s,
        "OUTCOME: {} after {} implementer iteration(s).",
        input.status, input.iterations
    );
    // Spelled out rather than left as a status word among many. The line below this one reports
    // the acceptance run, and on a run that ends unclean it says every test passes — which is the
    // more concrete claim and the one a reader believes. A live run published exactly that: a
    // status of `max_iterations_reached`, a pull request describing a clean landing, and no
    // mention of the review it had failed four times.
    if input.status != ratatoskr_core::RunStatus::Converged.as_str() {
        s.push_str(
            "THIS RUN DID NOT FINISH CLEAN. Whatever you write must say so in its own words, \
             near the top, before anything it did well. The tests passing is not the same as the \
             work being done.\n",
        );
    }
    s.push('\n');

    if !input.unresolved.is_empty() {
        s.push_str(
            "THE REVIEW STILL OBJECTED TO THIS. Report each one — a reviewer who finds it \
             themselves has been misled by what you wrote:\n",
        );
        for f in &input.unresolved {
            let _ = writeln!(s, "- [{:?}] {}", f.severity, f.summary);
            if !f.failure_scenario.is_empty() {
                let _ = writeln!(s, "    {}", f.failure_scenario);
            }
        }
        s.push('\n');
    }

    let a = &input.analyst;
    if !a.impact_summary.is_empty() {
        let _ = write!(s, "WHAT THE PLAN SAID:\n{}\n\n", a.impact_summary);
    }
    if !a.requirements.is_empty() {
        s.push_str("REQUIREMENTS IT WAS MEANT TO SATISFY:\n");
        for r in &a.requirements {
            let _ = writeln!(s, "- {r}");
        }
        s.push('\n');
    }

    match &input.implementer {
        None => {
            s.push_str(
                "NO CODE WAS CHANGED. This run produced an answer, not a change — there is \
                 nothing to open a pull request for.\n",
            );
        }
        Some(im) => {
            // The branch by name, and the tree separately. These are different strings and the
            // publisher acts on both: it pushes and opens a pull request against the branch.
            let _ = writeln!(s, "BRANCH: {}", im.branch);
            let _ = write!(s, "WORKTREE: {}\n\n", im.worktree_path);
            if !im.touched_files.is_empty() {
                let _ = writeln!(s, "FILES CHANGED: {}", im.touched_files.join(", "));
            }
            if !im.diff_summary.is_empty() {
                let _ = write!(s, "\nDIFF:\n{}\n", im.diff_summary);
            }
            let _ = write!(
                s,
                "\nACCEPTANCE: {} failing, {} passing (exit {}).\n",
                im.failing_tests.len(),
                im.passed_tests,
                im.exit_code
            );
            if !im.failing_tests.is_empty() {
                let _ = writeln!(s, "Still failing: {}", im.failing_tests.join(", "));
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyst() -> AnalystOutput {
        AnalystOutput {
            impact_summary: "widens the store".into(),
            touched: Vec::new(),
            risks: Vec::new(),
            requirements: vec!["keep existing rows readable".into()],
            residual_risk: String::new(),
            changes_code: true,
            acceptance: Vec::new(),
            interface: Vec::new(),
        }
    }

    fn finding(summary: &str) -> crate::verifier::Finding {
        crate::verifier::Finding {
            severity: crate::verifier::Severity::P2,
            kind: crate::verifier::FindingKind::Execution,
            file: String::new(),
            line: None,
            summary: summary.to_string(),
            failure_scenario: "a reader is told the opposite of what the code does".to_string(),
        }
    }

    fn implementer(failing: &[&str]) -> ImplementerOutput {
        ImplementerOutput {
            worktree_path: "/w/ratatoskr/abc".into(),
            branch: "ratatoskr/abc".into(),
            diff_summary: " store.rs | 12 ++".into(),
            touched_files: vec!["store.rs".into()],
            rewritten_files: Vec::new(),
            commit_kind: String::new(),
            commit_scope: String::new(),
            commit_subject: String::new(),
            failing_tests: failing.iter().map(|f| (*f).to_string()).collect(),
            passed_tests: 2,
            exit_code: if failing.is_empty() { 0 } else { 101 },
            narrative: None,
        }
    }

    #[test]
    fn a_run_that_changed_nothing_is_told_there_is_nothing_to_open() {
        // The case the whole node exists for: a research run's deliverable is an answer, and a
        // pull request is the wrong shape for it.
        let prompt = render_prompt(&PublisherInput {
            issue: "Why does the migration fail?".into(),
            analyst: analyst(),
            implementer: None,
            status: "no_code_change".into(),
            iterations: 0,
            unresolved: Vec::new(),
        });
        assert!(prompt.contains("NO CODE WAS CHANGED"));
        assert!(prompt.contains("nothing to open a pull request for"));
    }

    #[test]
    fn an_unresolved_run_hands_over_what_is_still_failing() {
        // So the write can say so. A pull request that oversells itself wastes the review it is
        // asking for, and the publisher can only be honest about what it was told.
        let prompt = render_prompt(&PublisherInput {
            issue: "Fix it".into(),
            analyst: analyst(),
            implementer: Some(implementer(&["store::migrates"])),
            status: "max_iterations_reached".into(),
            iterations: 3,
            unresolved: Vec::new(),
        });
        assert!(prompt.contains("max_iterations_reached"));
        assert!(prompt.contains("3 implementer iteration"));
        assert!(prompt.contains("Still failing: store::migrates"));
    }

    #[test]
    fn an_unclean_run_is_told_so_in_words_it_cannot_read_past() {
        // The failure this closes, from run `aab02a3d`: the status word said
        // `max_iterations_reached` and the line under it said every test passed. The pull request
        // was written from the second one and read as a clean landing.
        let prompt = render_prompt(&PublisherInput {
            issue: "Fix it".into(),
            analyst: analyst(),
            implementer: Some(implementer(&[])),
            status: "max_iterations_reached".into(),
            iterations: 4,
            unresolved: vec![finding(
                "the memory still describes the design this replaced",
            )],
        });
        assert!(prompt.contains("THIS RUN DID NOT FINISH CLEAN"), "{prompt}");
        assert!(prompt.contains("THE REVIEW STILL OBJECTED"), "{prompt}");
        assert!(
            prompt.contains("the memory still describes the design this replaced"),
            "{prompt}"
        );
        // And the scenario, because a summary without one is a claim the writer cannot check.
        assert!(prompt.contains("a reader is told the opposite"), "{prompt}");
    }

    #[test]
    fn a_clean_run_is_not_warned_about_a_review_that_passed() {
        // The warning has to mean something. A run that converged with nothing outstanding must
        // not carry text telling it to confess, or the next one that does will read the same.
        let prompt = render_prompt(&PublisherInput {
            issue: "Add a column".into(),
            analyst: analyst(),
            implementer: Some(implementer(&[])),
            status: "converged".into(),
            iterations: 1,
            unresolved: Vec::new(),
        });
        assert!(!prompt.contains("DID NOT FINISH CLEAN"), "{prompt}");
        assert!(!prompt.contains("THE REVIEW STILL OBJECTED"), "{prompt}");
    }

    #[test]
    fn a_converged_run_hands_over_the_branch_and_the_diff() {
        let prompt = render_prompt(&PublisherInput {
            issue: "Add a column".into(),
            analyst: analyst(),
            implementer: Some(implementer(&[])),
            status: "converged".into(),
            iterations: 1,
            unresolved: Vec::new(),
        });
        assert!(prompt.contains("BRANCH: ratatoskr/abc"), "{prompt}");
        assert!(prompt.contains("WORKTREE: /w/ratatoskr/abc"), "{prompt}");
        assert!(prompt.contains("store.rs"));
        assert!(prompt.contains("0 failing, 2 passing"));
        // The requirements travel too: they are what a description has to be checked against.
        assert!(prompt.contains("keep existing rows readable"));
    }

    #[test]
    fn nothing_published_still_has_to_say_why() {
        // When there is no URL, the reasoning is the entire result — someone reads it to decide
        // whether publishing nothing was right.
        let raw = r#"{"action":"none","reasoning":"the run changed nothing and answered nothing"}"#;
        let out = parse_validated::<PublisherOutput>(raw).unwrap();
        assert_eq!(out.action, "none");
        assert!(out.url.is_empty());

        assert!(parse_validated::<PublisherOutput>(r#"{"action":"none"}"#).is_err());
    }
}
