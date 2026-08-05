//! Isolated git worktrees for the implementer.
//!
//! `gix` (as of 0.86) can only *enumerate* worktrees, not create them — its `worktree` API is all
//! inspection. So creation/removal shells out to real `git worktree add`/`remove`, matching what
//! `majiayu000/harness` found is the only option. Read-only summaries (diff stat, touched files)
//! also go through `git` here for simplicity; `gix` reads could replace them later.

use std::path::{Path, PathBuf};

use tokio::process::Command;

use crate::ExecError;

/// The path to a created worktree.
#[derive(Debug, Clone)]
pub struct WorktreePath(pub PathBuf);

impl WorktreePath {
    pub fn as_path(&self) -> &Path {
        &self.0
    }
}

/// Run `git -C <cwd> <args...>`, returning stdout on success or an `ExecError::Git` on failure.
async fn git(cwd: &Path, op: &str, args: &[&str]) -> Result<String, ExecError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .await?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(ExecError::Git {
            op: op.to_string(),
            code: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        })
    }
}

fn path_str(p: &Path) -> Result<&str, ExecError> {
    p.to_str()
        .ok_or_else(|| ExecError::NonUtf8Path(p.to_path_buf()))
}

/// Create a worktree of `repo_root` at `<worktree_root>/<branch>` on a new branch `branch`.
pub async fn create(
    repo_root: &Path,
    worktree_root: &Path,
    branch: &str,
) -> Result<WorktreePath, ExecError> {
    // The path must be absolute: git resolves a relative worktree path against `repo_root`, but
    // downstream consumers (the ACP session's `cwd`) require an absolute path.
    let abs_root = if worktree_root.is_absolute() {
        worktree_root.to_path_buf()
    } else {
        repo_root.join(worktree_root)
    };
    std::fs::create_dir_all(&abs_root)?;
    let path = abs_root.join(branch);
    let path_s = path_str(&path)?;
    git(
        repo_root,
        "worktree add",
        &["worktree", "add", path_s, "-b", branch],
    )
    .await?;
    Ok(WorktreePath(path))
}

/// Remove a worktree (force, so uncommitted changes don't block cleanup).
pub async fn remove(repo_root: &Path, worktree: &WorktreePath) -> Result<(), ExecError> {
    let path_s = path_str(worktree.as_path())?;
    git(
        repo_root,
        "worktree remove",
        &["worktree", "remove", path_s, "--force"],
    )
    .await?;
    Ok(())
}

/// A `git diff --stat` summary of the worktree's changes (tracked + newly-added, via intent-to-add).
pub async fn diff_stat(worktree: &WorktreePath) -> Result<String, ExecError> {
    let cwd = worktree.as_path();
    // `-N` (intent-to-add) makes untracked files appear in the diff without staging their content.
    git(cwd, "add -N", &["add", "-N", "."]).await?;
    git(
        cwd,
        "diff --stat",
        &["--no-pager", "diff", "--stat", "HEAD"],
    )
    .await
}

/// The paths the worktree touched, from `git status --porcelain`.
pub async fn touched_files(worktree: &WorktreePath) -> Result<Vec<String>, ExecError> {
    let out = git(worktree.as_path(), "status", &["status", "--porcelain"]).await?;
    Ok(out
        .lines()
        .filter_map(|l| l.get(3..).map(str::to_string))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn init_repo(dir: &Path) {
        // A throwaway repo with one commit, so `worktree add -b` has a HEAD to branch from.
        for args in [
            vec!["init", "-q"],
            vec![
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "init",
            ],
        ] {
            let ok = Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(&args)
                .status()
                .await
                .unwrap()
                .success();
            assert!(ok, "git {args:?} failed");
        }
    }

    #[tokio::test]
    async fn worktree_create_and_remove_lifecycle() {
        let tmp = std::env::temp_dir().join(format!("ratatoskr-wt-test-{}", std::process::id()));
        let repo = tmp.join("repo");
        let wt_root = tmp.join("worktrees");
        std::fs::create_dir_all(&repo).unwrap();
        init_repo(&repo).await;

        let wt = create(&repo, &wt_root, "run-abc").await.unwrap();
        assert!(wt.as_path().exists(), "worktree dir should exist");
        assert!(
            wt.as_path().join(".git").exists(),
            "worktree should be a git checkout"
        );

        remove(&repo, &wt).await.unwrap();
        assert!(
            !wt.as_path().exists(),
            "worktree dir should be gone after remove"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }
}
