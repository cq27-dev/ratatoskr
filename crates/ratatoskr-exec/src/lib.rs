//! Execution primitives for the fork: isolated git worktrees and sandboxed command runs. Two
//! tightly-coupled concerns with exactly one caller set (the implementer node + red-team's test
//! run), bundled into one crate rather than two.

pub mod sandbox;
pub mod worktree;

pub use sandbox::{
    ExecOutput, Mount, SandboxError, SandboxSpec, resolve_container_image, run as sandbox_run,
};
pub use worktree::{
    Committer, ManagedWorktree, WorktreePath, WorktreeSurvey, commit_all,
    create as create_worktree, delete_branch as delete_worktree_branch, diff_text, full_diff_text,
    head_sha, managed_branches as managed_worktree_branches, prune as prune_worktrees,
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

#[cfg(test)]
mod tests {
    #[test]
    fn execution_has_no_external_agent_protocol_surface() {
        let workspace_manifest =
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml"));
        let lockfile = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"));
        let crate_manifest = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));

        for dependency_file in [workspace_manifest, lockfile, crate_manifest] {
            assert!(!dependency_file.contains("agent-client-protocol"));
        }
        assert!(
            !std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/acp.rs")
                .exists()
        );
        let forbidden_module = ["pub mod ", "acp"].concat();
        assert!(!include_str!("lib.rs").contains(&forbidden_module));
    }
}
