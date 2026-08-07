//! Overseer: pick which workflow runs a task.
//!
//! It runs once, before anything else, and it executes nothing. That makes it the cheapest node in
//! the pipeline and the most expensive one to get wrong: a bad choice commits the whole run to the
//! wrong shape, and every node downstream then does competent work on the wrong question. The
//! failure does not look like a failure — it looks like a good run that answered something nobody
//! asked — which is why the choice and its reasoning land on a checkpoint like any node's output.
//!
//! Opt-in on having a route, like the verifier and the characterizer. Without one, a repo with
//! several workflows is asked to name one rather than having a choice made for it silently.

use std::fmt::Write as _;

use ratatoskr_core::ModelRoute;
use ratatoskr_graph::{NodeError, parse_validated};
use ratatoskr_mcp::ToolSet;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// rag-rat tools the overseer may use. It reads the task, not the repository — but a task that
/// names an issue is worth resolving before deciding what kind of work it is.
pub const OVERSEER_TOOLS: &[&str] = &["papertrail_issue_search", "semantic_search"];

const PREAMBLE: &str = include_str!("../prompts/overseer.md");

/// One workflow, as the overseer is shown it.
#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub name: String,
    pub purpose: String,
    pub when_to_use: Vec<String>,
}

/// What the overseer decided.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OverseerOutput {
    /// The chosen workflow's name. Validated against the registry by the caller — a model naming
    /// something that is not there must not select it.
    pub workflow: String,
    /// Why this task fits that workflow. Recorded and read by a human when a run went somewhere
    /// unexpected, so it has to name what in the task drove the choice.
    pub reasoning: String,
}

/// The overseer node.
pub struct OverseerNode {
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

impl OverseerNode {
    pub async fn run(&self, issue: &str, choices: &[Choice]) -> Result<OverseerOutput, NodeError> {
        let raw = ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "overseer",
            route: &self.route,
            preamble: &crate::effective_preamble(
                PREAMBLE,
                self.system_prompt.as_deref(),
                self.plugins.context.as_deref(),
            ),
            question: &render_prompt(issue, choices),
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(OverseerOutput),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            // It chooses before any node exists to ask. There is nobody to clarify with.
            clarifier: None,
            observer: self.plugins.observer.clone(),
            skills: crate::skills::loaded(&self.plugins.skills),
            files: self.files.clone(),
            // Reads and edits, but runs nothing.
            shell: None,
            push: None,
            conversation: None,
            ledger: self.ledger.clone(),
            produces: Some("the name of the workflow to run, and why this task fits it"),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("overseer agent failed: {e}")))?;

        parse_validated::<OverseerOutput>(&raw)
    }
}

fn render_prompt(issue: &str, choices: &[Choice]) -> String {
    let mut s = String::new();
    s.push_str("AVAILABLE WORKFLOWS:\n\n");
    for c in choices {
        let _ = writeln!(s, "name: {}", c.name);
        if !c.purpose.is_empty() {
            let _ = writeln!(s, "purpose: {}", c.purpose);
        }
        for case in &c.when_to_use {
            let _ = writeln!(s, "  use when: {case}");
        }
        s.push('\n');
    }
    let _ = write!(s, "THE TASK:\n{issue}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(name: &str, cases: &[&str]) -> Choice {
        Choice {
            name: name.to_string(),
            purpose: format!("what {name} does"),
            when_to_use: cases.iter().map(|c| (*c).to_string()).collect(),
        }
    }

    #[test]
    fn the_prompt_shows_every_workflow_with_the_cases_that_select_it() {
        // `whenToUse` is the part selection actually matches against — a purpose alone describes a
        // workflow, the cases say when it is the right one.
        let prompt = render_prompt(
            "Rename the store's migrate helper.",
            &[
                choice("built-in", &["the task asks for a code change"]),
                choice("research", &["the task asks what or why"]),
            ],
        );
        assert!(prompt.contains("name: built-in"));
        assert!(prompt.contains("use when: the task asks what or why"));
        // The task comes last, so it is what the model reads immediately before deciding.
        let task_at = prompt.find("THE TASK:").unwrap();
        assert!(task_at > prompt.find("name: research").unwrap());
        assert!(prompt.contains("Rename the store's migrate helper."));
    }

    #[test]
    fn a_workflow_that_declared_no_cases_still_appears() {
        // Otherwise a workflow becomes unselectable by omitting an optional field, which is a
        // silent way to disable something.
        let prompt = render_prompt("x", &[choice("bare", &[])]);
        assert!(prompt.contains("name: bare"));
    }

    #[test]
    fn the_decision_parses_from_the_shape_a_model_writes() {
        let raw = r#"{"workflow":"research","reasoning":"The task asks whether X is possible."}"#;
        let out = parse_validated::<OverseerOutput>(raw).unwrap();
        assert_eq!(out.workflow, "research");

        // Reasoning is not optional: an unexplained choice is the one nobody can review after a
        // run went somewhere unexpected.
        let bare = r#"{"workflow":"research"}"#;
        assert!(parse_validated::<OverseerOutput>(bare).is_err());
    }
}
