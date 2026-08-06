//! Implementer: drive the coding CLI (Claude Code via ACP) to make edits in an isolated worktree,
//! then run the repo's tests in a sandbox against that worktree. Converge iterates this node on the
//! same worktree with a diagnostic prompt when the change introduces new failures.

use std::fmt::Write as _;
use std::path::PathBuf;

use ratatoskr_core::{ImplementerConfig, SandboxConfig};
use ratatoskr_exec::worktree::{self, WorktreePath};
use ratatoskr_exec::{acp, create_worktree, remove_worktree};
use ratatoskr_graph::NodeError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::analyst::AnalystOutput;
use crate::testrun::{Characterizer, by_exit_code, run_acceptance};

/// Implementer output. Test fields are deterministic; `diff_summary`/`touched_files`/`narrative`
/// are best-effort context (relaxed) for the bookkeeper in Phase 4.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ImplementerOutput {
    pub worktree_path: String,
    #[serde(default)]
    pub diff_summary: String,
    #[serde(default)]
    pub touched_files: Vec<String>,
    pub failing_tests: Vec<String>,
    pub passing_tests: Vec<String>,
    pub exit_code: i32,
    /// The CLI's own narrative of what it did (optional).
    #[serde(default)]
    pub narrative: Option<String>,
}

/// What a natively-driven implementer is told.
///
/// A file rather than a string constant because it is 76 lines of prose that changes for prompt
/// reasons, not code reasons — keeping it out of the source means a wording change reads as a
/// wording change in review, and the file can be diffed against a run's behaviour directly.
///
/// Written for this pipeline rather than adapted from a coding CLI's: those address an assistant
/// with a human watching, and most of what they spend words on — tone, when to explain yourself,
/// how to format a reply — is inapplicable here and costs attention. What replaces it is the part
/// no interactive agent needs: what "done" means when nobody will confirm it, that the referee is
/// off limits and why, and what to do when the session opens with a diagnostic rather than a plan.
pub const NATIVE_PREAMBLE: &str = include_str!("../prompts/implementer.md");

/// The implementer node. Holds everything needed to create the worktree and drive the CLI.
pub struct ImplementerNode {
    pub repo_path: PathBuf,
    pub worktree_root: PathBuf,
    pub sandbox: SandboxConfig,
    pub implementer: ImplementerConfig,
    pub run_id: String,
    pub issue: String,
    pub analyst: AnalystOutput,
    /// What "done" means for this task. Frozen at plan time: a change must not be able to move the
    /// bar it is judged against, so an analyst revision amends requirements and never this.
    pub acceptance: Vec<ratatoskr_core::AcceptanceStep>,
    /// Names the checks inside each step. `None` compares at step granularity instead.
    pub characterizer: Option<Characterizer>,
}

impl ImplementerNode {
    /// Create the worktree and make the first attempt. Returns the worktree (for iteration and
    /// cleanup) alongside the output.
    pub async fn run(&self) -> Result<(WorktreePath, ImplementerOutput), NodeError> {
        let branch = format!("ratatoskr/{}", self.short_id());
        let worktree = create_worktree(&self.repo_path, &self.worktree_root, &branch)
            .await
            .map_err(|e| NodeError::Failed(format!("worktree create failed: {e}")))?;
        // If the first attempt fails, remove the worktree so a failed run leaves nothing behind.
        match self.attempt(&worktree, &self.initial_prompt()).await {
            Ok(out) => Ok((worktree, out)),
            Err(e) => {
                if let Err(rm) = remove_worktree(&self.repo_path, &worktree).await {
                    tracing::warn!("failed to clean up worktree after implementer error: {rm}");
                }
                Err(e)
            }
        }
    }

    /// Re-drive the CLI on the same worktree with a diagnostic prompt (converge iteration).
    pub async fn iterate(
        &self,
        worktree: &WorktreePath,
        diagnostic: &str,
    ) -> Result<ImplementerOutput, NodeError> {
        self.attempt(worktree, diagnostic).await
    }

    /// One attempt: drive the CLI, then run the worktree's tests and read its diff.
    async fn attempt(
        &self,
        worktree: &WorktreePath,
        prompt: &str,
    ) -> Result<ImplementerOutput, NodeError> {
        let command = acp_command(&self.implementer.cli)?;
        let turn = acp::drive(&command, worktree.as_path(), prompt)
            .await
            .map_err(|e| NodeError::Failed(format!("ACP session failed: {e}")))?;

        let outcomes = run_acceptance(
            &self.sandbox,
            &format!("ratatoskr-impl-{}", self.short_id()),
            worktree.as_path(),
            &self.acceptance,
        )
        .await
        .map_err(NodeError::Failed)?;
        let tests = match &self.characterizer {
            Some(c) => c.read(&outcomes).await,
            None => by_exit_code(&outcomes),
        };

        // Diff/touched reads are best-effort — don't fail the attempt if they hiccup.
        let diff_summary = worktree::diff_stat(worktree).await.unwrap_or_default();
        let touched_files = worktree::touched_files(worktree).await.unwrap_or_default();

        Ok(ImplementerOutput {
            worktree_path: worktree.as_path().display().to_string(),
            diff_summary,
            touched_files,
            failing_tests: tests.failing,
            passing_tests: tests.passing,
            exit_code: tests.exit_code,
            narrative: Some(turn.output),
        })
    }

    fn short_id(&self) -> String {
        self.run_id.chars().take(8).collect()
    }

    /// The initial prompt: the issue plus the analyst's requirements and risks.
    fn initial_prompt(&self) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "Implement this task in the current repository:\n\n{}\n\n",
            self.issue
        );
        let a = &self.analyst;
        if !a.impact_summary.is_empty() {
            let _ = write!(s, "Impact analysis:\n{}\n\n", a.impact_summary);
        }
        if !a.requirements.is_empty() {
            s.push_str("Requirements the implementation must satisfy:\n");
            for req in &a.requirements {
                let _ = writeln!(s, "- {req}");
            }
            s.push('\n');
        }
        if !a.risks.is_empty() {
            s.push_str("Known risks to avoid:\n");
            for risk in &a.risks {
                let _ = writeln!(s, "- {risk}");
            }
            s.push('\n');
        }
        s.push_str(
            "Apply the change directly with your editing tools — do NOT ask for confirmation or \
             present options to choose between; just make the fix. Then run the repo's tests and \
             ensure they pass.",
        );
        s
    }
}

/// Map a config `implementer.cli` to the ACP agent command. Phase 3 supports Claude Code only.
fn acp_command(cli: &str) -> Result<String, NodeError> {
    match cli {
        // The renamed Zed adapter (was @zed-industries/claude-code-acp) speaks ACP over stdio.
        "claude" => Ok("npx -y @agentclientprotocol/claude-agent-acp".to_string()),
        other => Err(NodeError::Failed(format!(
            "unsupported implementer.cli {other:?}; only \"claude\" is wired in Phase 3"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_native_preamble_states_the_rules_a_run_actually_enforces() {
        // The prompt lives in a separate file, so nothing in the type system ties it to the gates
        // it describes. These are the four an iteration is actually rejected or re-driven on: a
        // prompt that stopped mentioning one would leave the model to discover it by failing.
        let p = NATIVE_PREAMBLE;
        assert!(p.contains("REFEREE GATE"), "the hard-rejection rule");
        assert!(
            p.contains("conftest.py") && p.contains("Cargo.toml"),
            "the referee files"
        );
        assert!(p.contains("DEFINITION OF DONE"), "when to stop");
        assert!(p.contains("exactly once"), "Edit's uniqueness contract");
        assert!(p.contains("DIAGNOSTIC"), "the re-driven path");
        // It is told there is nobody to ask, which is the fact every interactive prompt assumes
        // the opposite of.
        assert!(p.contains("no human") || p.contains("There is no human"));
    }
}
