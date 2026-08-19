//! Projections from stored checkpoints to what the dashboard shows.
//!
//! A checkpoint is a node's recorded output as JSON text. These read the fields the dashboard needs
//! out of it and answer `None` rather than failing when a record is absent or malformed — a run
//! whose publisher wrote something odd is still a run worth looking at.

use std::path::Path;

use ratatoskr_store::Checkpoint;
use serde::Serialize;

use crate::pipeline::ISSUE_NODE;

/// The implementer's worktree — the reviewable deliverable, kept on `converged` and
/// `max_iterations_reached` and removed by a hard error or `ratatoskr clean`. Reported separately
/// from node state on purpose: a converged run's worktree is usually still on disk.
#[derive(Debug, Serialize)]
pub(crate) struct WorktreeView {
    path: String,
    exists: bool,
}

/// A pull request a run opened. The publisher's `url` is only a PR for `action` `pull_request`
/// or `both`; `#number` is the URL's last path segment (`/pull/139` → `139`).
#[derive(Debug, Serialize)]
pub(crate) struct PullRequestView {
    number: u64,
    url: String,
}

/// Pull the run's issue text out of the `issue` pseudo-checkpoint.
pub(crate) fn issue_text(checkpoints: &[Checkpoint]) -> Option<String> {
    let raw = checkpoints
        .iter()
        .find(|c| c.node_name == ISSUE_NODE)?
        .output_json
        .as_str();
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    value
        .get("issue")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// The implementer records an absolute `worktree_path`; iterations reuse it, so the latest
/// checkpoint is authoritative. Whether it's still on disk is a filesystem question, not a
/// store one — `ratatoskr clean` removes worktrees without touching checkpoints.
pub(crate) fn worktree_view(checkpoints: &[Checkpoint]) -> Option<WorktreeView> {
    let raw = checkpoints
        .iter()
        .rev()
        .find(|c| c.node_name == "implementer")?
        .output_json
        .as_str();
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    let path = value.get("worktree_path")?.as_str()?.to_string();
    let exists = Path::new(&path).exists();
    Some(WorktreeView { path, exists })
}

/// The pull request the latest `publisher` checkpoint opened, if any.
///
/// A checkpoint records `pull_request_url` separately from `comment_url`, and the action keeps
/// comment-only checkpoints out of the dashboard. The parser accepts a URL only when it has
/// GitHub's pull-request path shape, so a malformed field is never used as an anchor verbatim.
pub(crate) fn pull_request_view(checkpoints: &[Checkpoint]) -> Option<PullRequestView> {
    let raw = checkpoints
        .iter()
        .rev()
        .find(|c| c.node_name == "publisher")?
        .output_json
        .as_str();
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    match value.get("action").and_then(|v| v.as_str()) {
        Some("pull_request") | Some("both") => {}
        _ => return None,
    }
    value
        .get("pull_request_url")
        .and_then(serde_json::Value::as_str)
        .and_then(pull_request_url)
}

/// Extract one GitHub pull-request URL from a checkpoint field.
///
/// The field is meant to hold the URL alone, but a model writes it. Splitting on whitespace
/// recovers the URL from a labelled or otherwise chatty value without letting the dashboard use
/// the surrounding text as its `href`.
fn pull_request_url(value: &str) -> Option<PullRequestView> {
    value
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|c| matches!(c, '"' | '\'' | '(' | ')' | '[' | ']' | ',' | '.'))
        })
        .find_map(parse_github_pull_request_url)
}

fn parse_github_pull_request_url(url: &str) -> Option<PullRequestView> {
    let mut segments = url.strip_prefix("https://github.com/")?.split('/');
    let owner = segments.next()?;
    let repository = segments.next()?;
    if owner.is_empty() || repository.is_empty() || segments.next()? != "pull" {
        return None;
    }
    let number_and_suffix = segments.next()?;
    if segments.next().is_some() {
        return None;
    }

    let number_end = number_and_suffix
        .find(['?', '#'])
        .unwrap_or(number_and_suffix.len());
    let (digits, suffix) = number_and_suffix.split_at(number_end);
    if digits.is_empty() || (!suffix.is_empty() && !suffix.starts_with(['?', '#'])) {
        return None;
    }
    let number = digits.parse().ok()?;
    Some(PullRequestView {
        number,
        url: url.to_string(),
    })
}

/// Parse stored JSON, falling back to the raw text so a malformed checkpoint is still visible
/// rather than swallowing the whole response.
pub(crate) fn parse_or_raw(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cp(node: &str, json: &str) -> Checkpoint {
        Checkpoint {
            node_name: node.to_string(),
            output_json: json.to_string(),
            created_at: "t".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn reads_the_issue_text_out_of_the_pseudo_node() {
        let cps = vec![cp(ISSUE_NODE, r#"{"issue":"fix the flaky retry"}"#)];
        assert_eq!(issue_text(&cps).as_deref(), Some("fix the flaky retry"));
        assert!(issue_text(&[]).is_none());
        // A malformed record is absent, not a panic.
        assert!(issue_text(&[cp(ISSUE_NODE, "not json")]).is_none());
    }

    #[test]
    fn takes_the_worktree_from_the_latest_implementer_checkpoint() {
        let cps = vec![
            cp("implementer", r#"{"worktree_path":"/tmp/old"}"#),
            cp(
                "implementer",
                r#"{"worktree_path":"/tmp/ratatoskr-definitely-absent"}"#,
            ),
        ];
        let wt = worktree_view(&cps).unwrap();
        assert_eq!(wt.path, "/tmp/ratatoskr-definitely-absent");
        assert!(!wt.exists);
        assert!(worktree_view(&[]).is_none());
    }

    #[test]
    fn a_malformed_checkpoint_still_renders() {
        assert_eq!(parse_or_raw(r#"{"a":1}"#), serde_json::json!({"a": 1}));
        assert_eq!(parse_or_raw("garbage"), serde_json::json!("garbage"));
    }

    // --- pull_request_view -------------------------------------------------
    // The contract leaves `PullRequestView::number` as u64/i64/String; these tests read it as an
    // integer (the `139` literal compiles for u64 or i64), which is the reading that makes the
    // "last URL segment is numeric" requirement checkable. `url` is asserted as a `&str`.

    #[test]
    fn reads_the_pull_request_from_a_publisher_checkpoint() {
        let cps = vec![cp(
            "publisher",
            r#"{"action":"pull_request","pull_request_url":"https://github.com/o/r/pull/139","reasoning":"..."}"#,
        )];
        let pr = pull_request_view(&cps).unwrap();
        assert_eq!(pr.number, 139);
        assert_eq!(pr.url, "https://github.com/o/r/pull/139");
    }

    #[test]
    fn action_both_still_yields_the_pull_request() {
        let cps = vec![cp(
            "publisher",
            r#"{"action":"both","pull_request_url":"https://github.com/o/r/pull/42","comment_url":"https://github.com/o/r/issues/1#issuecomment-2","reasoning":"x"}"#,
        )];
        let pr = pull_request_view(&cps).unwrap();
        assert_eq!(pr.number, 42);
        assert_eq!(pr.url, "https://github.com/o/r/pull/42");
    }

    #[test]
    fn takes_the_pull_request_from_the_latest_publisher_checkpoint() {
        // Latest-wins, like worktree_view: the later publisher checkpoint is authoritative.
        let cps = vec![
            cp(
                "publisher",
                r#"{"action":"pull_request","pull_request_url":"https://github.com/o/r/pull/1"}"#,
            ),
            cp(
                "publisher",
                r#"{"action":"pull_request","pull_request_url":"https://github.com/o/r/pull/2"}"#,
            ),
        ];
        let pr = pull_request_view(&cps).unwrap();
        assert_eq!(pr.number, 2);
        assert_eq!(pr.url, "https://github.com/o/r/pull/2");
    }

    #[test]
    fn a_comment_is_never_presented_as_a_pull_request() {
        let cps = vec![cp(
            "publisher",
            r#"{"action":"comment","comment_url":"https://github.com/o/r/issues/12#issuecomment-999"}"#,
        )];
        assert!(pull_request_view(&cps).is_none());
    }

    #[test]
    fn action_none_with_no_url_is_absent() {
        let cps = vec![cp("publisher", r#"{"action":"none"}"#)];
        assert!(pull_request_view(&cps).is_none());
    }

    #[test]
    fn no_publisher_checkpoint_is_absent() {
        assert!(pull_request_view(&[]).is_none());
        let cps = vec![cp("implementer", r#"{"worktree_path":"/tmp/x"}"#)];
        assert!(pull_request_view(&cps).is_none());
    }

    #[test]
    fn a_malformed_publisher_checkpoint_is_absent_not_a_panic() {
        assert!(pull_request_view(&[cp("publisher", "not json")]).is_none());
    }

    #[test]
    fn a_pull_request_with_a_non_numeric_last_segment_is_absent() {
        let cps = vec![cp(
            "publisher",
            r#"{"action":"pull_request","pull_request_url":"https://github.com/o/r/pull/not-a-number"}"#,
        )];
        assert!(pull_request_view(&cps).is_none());
        let empty_url = vec![cp(
            "publisher",
            r#"{"action":"pull_request","pull_request_url":""}"#,
        )];
        assert!(pull_request_view(&empty_url).is_none());
    }

    #[test]
    fn a_labelled_pull_request_field_yields_only_the_url() {
        // The publisher is told to write the URL alone, and a model writes it. A value that came
        // back labelled still anchors the dashboard at the pull request, not at the label.
        let cps = vec![cp(
            "publisher",
            r#"{"action":"both","pull_request_url":"PR: https://github.com/cq27-dev/ratatoskr/pull/214","comment_url":"https://github.com/cq27-dev/ratatoskr/issues/210#issuecomment-5231512849"}"#,
        )];

        let pr = pull_request_view(&cps).unwrap();
        assert_eq!(pr.number, 214);
        assert_eq!(pr.url, "https://github.com/cq27-dev/ratatoskr/pull/214");
    }

    #[test]
    fn a_non_github_pull_request_url_is_absent() {
        let cps = vec![cp(
            "publisher",
            r#"{"action":"pull_request","pull_request_url":"https://example.com/o/r/pull/214"}"#,
        )];
        assert!(pull_request_view(&cps).is_none());
    }

    #[test]
    fn a_pull_request_missing_the_url_field_is_absent() {
        let cps = vec![cp("publisher", r#"{"action":"pull_request"}"#)];
        assert!(pull_request_view(&cps).is_none());
    }

    #[test]
    fn a_pull_request_view_serializes_number_and_url() {
        // Mirrors `RunDetail.pull_request`: the JSON the API/api.ts consumer sees.
        let cps = vec![cp(
            "publisher",
            r#"{"action":"pull_request","pull_request_url":"https://github.com/o/r/pull/139"}"#,
        )];
        let pr = pull_request_view(&cps).unwrap();
        let json = serde_json::to_value(&pr).unwrap();
        assert_eq!(json["number"], serde_json::json!(139));
        assert_eq!(
            json["url"],
            serde_json::json!("https://github.com/o/r/pull/139")
        );
    }
}
