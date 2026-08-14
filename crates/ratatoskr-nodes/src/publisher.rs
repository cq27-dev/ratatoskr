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

#[cfg(test)]
use ratatoskr_graph::parse_validated;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::analyst::AnalystOutput;
use crate::implementer::ImplementerOutput;

/// What the publisher did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PublisherAction {
    PullRequest,
    Comment,
    Both,
    None,
}

/// What the publisher did.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PublisherOutput {
    /// The dashboard uses this closed classification to decide which published URLs are present.
    pub action: PublisherAction,
    /// The one pull request URL, if a pull request was opened.
    ///
    /// Never combine this with an issue-comment URL or a label such as `PR:`. It defaults to the
    /// empty string rather than being required, because a run that opened no pull request has
    /// none to report.
    #[serde(default)]
    pub pull_request_url: String,
    /// The one issue-comment URL, if a comment was posted.
    ///
    /// This stays separate from [`Self::pull_request_url`] when `action` is `both`.
    #[serde(default)]
    pub comment_url: String,
    /// Why this was the right form — and, when nothing was published, the whole result.
    pub reasoning: String,
}

/// What the publisher is given: the run, and what it produced.
#[derive(Serialize)]
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
    /// What the last review could not reach, if it could not finish.
    ///
    /// Empty for a review that completed, and for a run that never reviewed. A run ends
    /// `unreviewed` when its review ran out of room, and the areas it named are the one actionable
    /// thing about that outcome — without them the pull request can say the run did not finish
    /// clean and nothing about what a human still has to look at. The verifier is told naming a gap
    /// is cheap because someone will act on it; this is where that promise is kept.
    pub unchecked: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_published_still_has_to_say_why() {
        // When there is no URL, the reasoning is the entire result — someone reads it to decide
        // whether publishing nothing was right.
        let raw = r#"{"action":"none","reasoning":"the run changed nothing and answered nothing"}"#;
        let out = parse_validated::<PublisherOutput>(raw).unwrap();
        assert_eq!(out.action, PublisherAction::None);
        assert!(out.pull_request_url.is_empty());
        assert!(out.comment_url.is_empty());

        assert!(parse_validated::<PublisherOutput>(r#"{"action":"none"}"#).is_err());
    }

    #[test]
    fn publishing_both_keeps_the_two_urls_in_their_own_fields() {
        let raw = r#"{
            "action":"both",
            "pull_request_url":"https://github.com/o/r/pull/214",
            "comment_url":"https://github.com/o/r/issues/210#issuecomment-1",
            "reasoning":"the change and its issue update are both published"
        }"#;
        let out = parse_validated::<PublisherOutput>(raw).unwrap();

        assert_eq!(out.pull_request_url, "https://github.com/o/r/pull/214");
        assert_eq!(
            out.comment_url,
            "https://github.com/o/r/issues/210#issuecomment-1"
        );
    }

    #[test]
    fn action_is_enforced_by_the_json_schema() {
        let raw = r#"{
            "action":"pr",
            "pull_request_url":"https://github.com/o/r/pull/221",
            "reasoning":"opened the pull request"
        }"#;

        assert!(parse_validated::<PublisherOutput>(raw).is_err());
    }
}
