//! Scout: search the tracker papertrail and code for context related to the issue.

use ratatoskr_core::{ModelRoute, RunState};
use ratatoskr_graph::{Node, NodeError, parse_validated};
use rmcp::model::Tool;
use rmcp::service::ServerSink;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// rag-rat tools the scout is allowed to use — a focused subset keeps a fast model reliable.
pub const SCOUT_TOOLS: &[&str] = &["papertrail_issue_search", "semantic_search"];

const PREAMBLE: &str = "You are the scout in a code-planning pipeline. Given an issue description, \
    find prior art and context in THIS repository: use `papertrail_issue_search` to find related \
    tracker issues/PRs and `semantic_search` to find related code. Call the tools — do not guess. \
    Then produce a structured summary: a list of the most relevant related items (with your \
    one-line take on how each relates), and a short free-text papertrail summary the downstream \
    analyst can build on. Be concrete and grounded in what the tools returned.";

/// One tracker item (or code area) the scout judged relevant. Fields are optional (the agent's
/// output is best-effort) — the gate enforces the object shape and types, and the essential
/// narrative lives in [`ScoutOutput::papertrail_summary`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct RelatedItem {
    /// Tracker item key (e.g. issue/PR number), or a short code locator if it's a code hit.
    pub item_key: String,
    pub title: String,
    /// URL if the tool provided one; empty string otherwise.
    pub url: String,
    /// The scout's one-line take on how this relates to the issue.
    pub relation: String,
    pub summary: String,
}

impl RelatedItem {
    /// Whether this item carries any identity — the model sometimes pads `related_items` with empty
    /// placeholders, which should be dropped before they reach the analyst or the CLI summary.
    pub fn is_meaningful(&self) -> bool {
        !(self.item_key.trim().is_empty() && self.title.trim().is_empty())
    }
}

/// Scout's structured output.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ScoutOutput {
    #[serde(default)]
    pub related_items: Vec<RelatedItem>,
    /// Free-text context for the analyst node — the essential, always-required output.
    pub papertrail_summary: String,
}

/// The scout node: a fast agent restricted to search tools.
pub struct ScoutNode {
    pub route: ModelRoute,
    pub tools: Vec<Tool>,
    pub sink: ServerSink,
    pub policy: Option<std::sync::Arc<dyn ratatoskr_core::ToolPolicy>>,
    pub max_turns: Option<usize>,
    pub clarifier: Option<std::sync::Arc<dyn ratatoskr_agent::Clarifier>>,
    /// Ruleset `systemPrompt`; replaces [`PREAMBLE`] when set.
    pub system_prompt: Option<String>,
}

impl Node for ScoutNode {
    type Input = String;
    type Output = ScoutOutput;

    fn name(&self) -> &'static str {
        "scout"
    }

    async fn run(&self, issue: String, _run_state: &RunState) -> Result<ScoutOutput, NodeError> {
        let raw = ratatoskr_agent::run_structured(
            "scout",
            &self.route,
            self.system_prompt.as_deref().unwrap_or(PREAMBLE),
            &issue,
            self.tools.clone(),
            self.sink.clone(),
            schemars::schema_for!(ScoutOutput),
            self.policy.clone(),
            self.max_turns,
            self.clarifier.clone(),
        )
        .await
        .map_err(|e| NodeError::Failed(format!("scout agent failed: {e}")))?;

        let mut out = parse_validated::<ScoutOutput>(&raw)?;
        // Drop empty placeholder items so the checkpoint, the analyst's input, and the CLI summary
        // stay clean.
        out.related_items.retain(RelatedItem::is_meaningful);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_canned_scout_response() {
        let raw = r#"{
            "related_items": [
                {"item_key": "42", "title": "Fix store lock", "url": "http://x/42",
                 "relation": "same subsystem", "summary": "prior work on the mutex"}
            ],
            "papertrail_summary": "One prior issue touched the store."
        }"#;
        let out = parse_validated::<ScoutOutput>(raw).unwrap();
        assert_eq!(out.related_items.len(), 1);
        assert_eq!(out.related_items[0].item_key, "42");
    }

    #[test]
    fn empty_related_items_are_dropped() {
        let item = |k: &str, t: &str| RelatedItem {
            item_key: k.to_string(),
            title: t.to_string(),
            url: String::new(),
            relation: String::new(),
            summary: String::new(),
        };
        // Empty and whitespace-only placeholders are not meaningful; anything with a key or title is.
        assert!(!item("", "").is_meaningful());
        assert!(!item("  ", "\t").is_meaningful());
        assert!(item("42", "").is_meaningful());
        assert!(item("", "Fix the lock").is_meaningful());

        let mut items = vec![item("", ""), item("42", "Fix"), item("  ", "")];
        items.retain(RelatedItem::is_meaningful);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].item_key, "42");
    }

    #[test]
    fn rejects_a_malformed_scout_response() {
        // Missing the essential `papertrail_summary` → rejected, not silently defaulted.
        let raw = r#"{"related_items": []}"#;
        assert!(matches!(
            parse_validated::<ScoutOutput>(raw),
            Err(NodeError::InvalidOutput(_))
        ));
        // Wrong type for related_items (string, not array) → also rejected.
        let raw = r#"{"related_items": "nope", "papertrail_summary": "x"}"#;
        assert!(matches!(
            parse_validated::<ScoutOutput>(raw),
            Err(NodeError::InvalidOutput(_))
        ));
    }
}
