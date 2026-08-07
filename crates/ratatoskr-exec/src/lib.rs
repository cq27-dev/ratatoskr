//! Execution primitives for the fork: isolated git worktrees and sandboxed command runs. Two
//! tightly-coupled concerns with exactly one caller set (the implementer node + red-team's test
//! run), bundled into one crate rather than two.

pub mod sandbox;
pub mod worktree;

pub use sandbox::{ExecOutput, Mount, SandboxError, SandboxSpec, run as sandbox_run};
pub use worktree::{
    ManagedWorktree, WorktreePath, WorktreeSurvey, commit_all, create as create_worktree,
    delete_branch as delete_worktree_branch, diff_text, head_sha,
    managed_branches as managed_worktree_branches, prune as prune_worktrees,
    remove as remove_worktree, rewritten_files, survey as survey_worktrees,
};

/// Errors from the worktree module (git subprocess failures).
#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error("git {op} failed (exit {code}): {stderr}")]
    Git {
        op: String,
        code: i32,
        stderr: String,
    },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("path is not valid UTF-8: {0:?}")]
    NonUtf8Path(std::path::PathBuf),
}
