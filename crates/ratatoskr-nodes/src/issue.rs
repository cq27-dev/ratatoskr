//! Fetching the issue a run was only given a reference to.
//!
//! A run's task often arrives as a title and a number — `Issue #143: move the skill listing out of
//! the tool schema` — because that is what a person types, what a run list offers, and what the
//! GitHub trigger builds from a comment. The body is where the issue does its work: the design it
//! argues for, the alternative it rejects, and the acceptance it will be judged against.
//!
//! Without it a run plans from a title and whatever the code index happens to have cached. That is
//! not a hypothetical: a live run planned #143 from its title plus an 815-character search snippet,
//! and reached the opposite conclusion from the one the issue spends two paragraphs arguing for —
//! its analyst said so itself, flagging the destination as "inferred, not confirmed". The plan
//! satisfied one of the issue's three acceptance criteria.
//!
//! Best-effort, like every other thing this pipeline reaches outside itself for. No tracker, no
//! `gh`, no network, an issue that does not exist: the run proceeds with exactly the text it was
//! given, which is what it would have had anyway.

use std::path::Path;
use std::time::Duration;

/// How long the fetch may take. A run should not be held at the starting line by a slow tracker,
/// and what is being waited for is a nicety — the run has a task either way.
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);

/// The run's task, with the referenced issue's title and body appended when they are missing.
///
/// Returns `issue` unchanged when there is no reference to resolve, when the text already carries
/// the body, or when the tracker cannot be reached.
pub async fn enriched(issue: &str, root: &Path) -> String {
    let Some(number) = ratatoskr_agent::publish::issue_number(issue) else {
        return issue.to_string();
    };
    let Some(fetched) = fetch(&number, root).await else {
        return issue.to_string();
    };
    // Already there: a caller that pasted the whole issue, or a re-run from a stored task. Appending
    // would hand the model the same prose twice and invite it to treat the copies as two sources.
    if fetched.body.trim().is_empty() || issue.contains(fetched.body.trim()) {
        return issue.to_string();
    }
    tracing::info!(
        issue = %number,
        chars = fetched.body.len(),
        "read the issue's body from the tracker"
    );
    format!(
        "{issue}\n\n--- issue #{number} as the tracker holds it ---\n\n{}\n\n{}",
        fetched.title, fetched.body
    )
}

/// What the tracker holds for one issue.
struct Fetched {
    title: String,
    body: String,
}

/// Read one issue through `gh`, in `root` so it resolves the repository the way every other
/// tracker call in this pipeline does.
///
/// Deliberately narrow: one subcommand, one issue, JSON out. This is not the publisher's `gh`
/// tool — no model chooses these arguments, so there is nothing here to constrain.
async fn fetch(number: &str, root: &Path) -> Option<Fetched> {
    let output = tokio::time::timeout(
        FETCH_TIMEOUT,
        tokio::process::Command::new("gh")
            .current_dir(root)
            .args(["issue", "view", number, "--json", "title,body"])
            .output(),
    )
    .await;
    let output = match output {
        Ok(Ok(output)) if output.status.success() => output,
        // Every failure is the same answer: a run with the text it was given. Logged at debug
        // because the ordinary case for a repository with no tracker is failing here every run.
        other => {
            tracing::debug!(issue = %number, "could not read the issue from the tracker: {other:?}");
            return None;
        }
    };
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    Some(Fetched {
        title: parsed.get("title")?.as_str()?.to_string(),
        body: parsed.get("body")?.as_str()?.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_task_naming_no_issue_is_left_alone() {
        // No reference, so nothing to fetch and no reason to spend a subprocess finding out.
        let task = "make the scrubber sticky";
        assert_eq!(enriched(task, Path::new(".")).await, task);
    }

    #[tokio::test]
    async fn an_unreachable_tracker_leaves_the_run_its_task() {
        // A directory that is not a repository is the case every non-GitHub project is in. The run
        // must start regardless, with exactly what it was handed.
        let task = "Issue #143: move the skill listing out of the tool schema";
        let nowhere = std::env::temp_dir();
        assert_eq!(enriched(task, &nowhere).await, task);
    }
}
