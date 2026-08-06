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
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Ratatoskr's run branches (and their worktrees) live under this prefix — these are ours to
/// reclaim; anything else is the user's own or a foreign worktree and is never touched.
const MANAGED_BRANCH_PREFIX: &str = "ratatoskr/";

/// A ratatoskr-created worktree and the branch it's on (short name, e.g. `ratatoskr/abc12345`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedWorktree {
    pub path: WorktreePath,
    pub branch: String,
}

/// What `clean` operates on: the main worktree root (a stable git anchor — it's never a target, so
/// operations still work when invoked from inside a worktree being removed) plus the managed set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorktreeSurvey {
    pub main_root: PathBuf,
    pub managed: Vec<ManagedWorktree>,
}

/// One `git worktree list --porcelain` entry: path and short branch name (`None` if detached/bare).
struct Entry {
    path: PathBuf,
    branch: Option<String>,
}

fn parse_worktrees(porcelain: &str) -> Vec<Entry> {
    let mut entries = Vec::new();
    let mut path: Option<PathBuf> = None;
    let mut branch: Option<String> = None;
    for line in porcelain.lines() {
        if let Some(p) = line.strip_prefix("worktree ") {
            if let Some(prev) = path.take() {
                entries.push(Entry {
                    path: prev,
                    branch: branch.take(),
                });
            }
            path = Some(PathBuf::from(p));
        } else if let Some(b) = line
            .strip_prefix("branch ")
            .and_then(|r| r.strip_prefix("refs/heads/"))
        {
            branch = Some(b.to_string());
        }
    }
    if let Some(p) = path.take() {
        entries.push(Entry { path: p, branch });
    }
    entries
}

/// Survey `repo_root`'s worktrees: the main root (always the first entry git lists) and every
/// worktree on a `ratatoskr/*` branch. Foreign and user worktrees are excluded by the branch prefix;
/// an empty/failed listing yields no managed entries — never a wildcard removal.
pub async fn survey(repo_root: &Path) -> Result<WorktreeSurvey, ExecError> {
    let out = git(
        repo_root,
        "worktree list",
        &["worktree", "list", "--porcelain"],
    )
    .await?;
    let entries = parse_worktrees(&out);
    let main_root = entries
        .first()
        .map(|e| e.path.clone())
        .unwrap_or_else(|| repo_root.to_path_buf());
    let managed = entries
        .into_iter()
        .filter_map(|e| {
            let branch = e.branch?;
            branch
                .starts_with(MANAGED_BRANCH_PREFIX)
                .then_some(ManagedWorktree {
                    path: WorktreePath(e.path),
                    branch,
                })
        })
        .collect();
    Ok(WorktreeSurvey { main_root, managed })
}

/// Every local `ratatoskr/*` branch — including orphans whose worktree was already removed (the old
/// hard-error path, or a partial `clean`), which the worktree listing alone can't surface.
pub async fn managed_branches(repo_root: &Path) -> Result<Vec<String>, ExecError> {
    let out = git(
        repo_root,
        "branch list",
        &[
            "branch",
            "--list",
            "ratatoskr/*",
            "--format=%(refname:short)",
        ],
    )
    .await?;
    Ok(out
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

/// Force-delete a local branch. Refuses (errors) if the branch is still checked out in a worktree.
pub async fn delete_branch(repo_root: &Path, branch: &str) -> Result<(), ExecError> {
    git(repo_root, "branch -D", &["branch", "-D", branch]).await?;
    Ok(())
}

/// Prune stale worktree registrations (dirs deleted out-of-band).
pub async fn prune(repo_root: &Path) -> Result<(), ExecError> {
    git(repo_root, "worktree prune", &["worktree", "prune"]).await?;
    Ok(())
}

/// The commit `repo_root` is currently on.
///
/// What a run was measured against: two runs of the same graph on different commits are not
/// comparable, and nothing else in a checkpoint says which tree the work started from.
pub async fn head_sha(repo_root: &Path) -> Result<String, ExecError> {
    let out = git(repo_root, "rev-parse HEAD", &["rev-parse", "HEAD"]).await?;
    Ok(out.trim().to_string())
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

/// How much patch text [`diff_text`] will hand back before it starts cutting.
///
/// A review reads the change; it does not need every line of a vendored lockfile to do that. The
/// cap exists because the diff goes into a model's context, where an unbounded one costs the whole
/// budget and buries the hunks that matter — and because a truncated diff that says so is more
/// useful than a request that fails for being too large.
const MAX_DIFF_BYTES: usize = 200_000;

/// The worktree's changes as an actual patch — what a reviewer reads.
///
/// Distinct from [`diff_stat`], which is filenames and line counts. A `--stat` cannot show a test
/// weakened to pass, an error swallowed, or a condition inverted, so anything judging the *content*
/// of a change needs this instead.
pub async fn diff_text(worktree: &WorktreePath) -> Result<String, ExecError> {
    let cwd = worktree.as_path();
    // `-N` (intent-to-add) so a newly-created file's content appears without staging it. A change
    // that only adds files would otherwise diff as empty and read as "nothing was done".
    git(cwd, "add -N", &["add", "-N", "."]).await?;
    let out = git(cwd, "diff", &["--no-pager", "diff", "HEAD"]).await?;
    Ok(truncate_diff(out, MAX_DIFF_BYTES))
}

/// Cut `diff` to `max` bytes at a line boundary, saying so where it cut.
///
/// The marker is not decoration: a reader given a silently truncated patch concludes the change
/// ends there, and a reviewer that thinks it has seen the whole diff will approve what it cannot
/// see.
fn truncate_diff(diff: String, max: usize) -> String {
    if diff.len() <= max {
        return diff;
    }
    // Back up to a line boundary so the last hunk shown is readable rather than cut mid-token.
    let cut = diff[..max].rfind('\n').map_or(max, |i| i + 1);
    let dropped = diff.len() - cut;
    format!(
        "{}\n[diff truncated: {dropped} more bytes not shown]\n",
        &diff[..cut]
    )
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

    #[test]
    fn parse_worktrees_pairs_paths_and_isolates_ratatoskr_branches() {
        let porcelain = "\
worktree /repo
HEAD aaaa
branch refs/heads/main

worktree /repo/.ratatoskr/worktrees/ratatoskr/abc12345
HEAD bbbb
branch refs/heads/ratatoskr/abc12345

worktree /repo/detached
HEAD cccc
detached

worktree /elsewhere/feature
HEAD dddd
branch refs/heads/feature/x
";
        let entries = parse_worktrees(porcelain);
        assert_eq!(entries.len(), 4);
        // The main worktree is the first entry — the stable anchor for git operations.
        assert_eq!(entries[0].path, PathBuf::from("/repo"));
        assert_eq!(entries[0].branch.as_deref(), Some("main"));
        assert_eq!(entries[2].branch, None, "detached entry has no branch");

        let managed: Vec<(PathBuf, String)> = entries
            .into_iter()
            .filter_map(|e| {
                let b = e.branch?;
                b.starts_with(MANAGED_BRANCH_PREFIX).then_some((e.path, b))
            })
            .collect();
        assert_eq!(managed.len(), 1, "only the ratatoskr/* worktree is managed");
        assert_eq!(managed[0].1, "ratatoskr/abc12345");
        assert_eq!(
            managed[0].0,
            PathBuf::from("/repo/.ratatoskr/worktrees/ratatoskr/abc12345")
        );
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

    #[test]
    fn a_truncated_diff_says_that_it_was_truncated() {
        let diff = "line one\nline two\nline three\n".to_string();
        assert_eq!(
            truncate_diff(diff.clone(), 1000),
            diff,
            "a small diff is untouched"
        );

        let cut = truncate_diff(diff, 12);
        // A reader given a silently truncated patch concludes the change ends there, and a
        // reviewer that thinks it saw the whole diff will approve what it could not see.
        assert!(cut.contains("[diff truncated"), "{cut}");
        assert!(cut.starts_with("line one\n"));
        // Cut at a line boundary, so the last hunk shown is readable rather than split mid-token.
        assert!(!cut.contains("line t\n"));
    }
}
