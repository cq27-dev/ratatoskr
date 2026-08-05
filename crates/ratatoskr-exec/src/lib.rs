//! Execution primitives for Phase 3's fork: isolated git worktrees, sandboxed command runs, and
//! an ACP client for driving a coding CLI. Three tightly-coupled concerns with exactly one caller
//! set (the implementer node + red-team's test run), bundled into one crate rather than three.

pub mod acp;
pub mod sandbox;
pub mod worktree;

pub use acp::{AcpError, AcpTurnResult};
pub use sandbox::{ExecOutput, Mount, SandboxError, SandboxSpec, run as sandbox_run};
pub use worktree::{WorktreePath, create as create_worktree, remove as remove_worktree};

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
