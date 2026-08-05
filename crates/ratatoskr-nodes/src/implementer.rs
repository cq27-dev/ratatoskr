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
use crate::testrun::run_tests;

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

/// The implementer node. Holds everything needed to create the worktree and drive the CLI.
pub struct ImplementerNode {
    pub repo_path: PathBuf,
    pub worktree_root: PathBuf,
    pub sandbox: SandboxConfig,
    pub implementer: ImplementerConfig,
    pub run_id: String,
    pub issue: String,
    pub analyst: AnalystOutput,
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

        let tests = run_tests(
            &self.sandbox,
            &format!("ratatoskr-impl-{}", self.short_id()),
            worktree.as_path(),
        )
        .await
        .map_err(NodeError::Failed)?;

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
                let _ = writeln!(s, "- [{}] {}", risk.severity, risk.description);
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
