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
    // downstream consumers (the sandbox's workdir, the implementer's file tools) require an
    // absolute path.
    let abs_root = if worktree_root.is_absolute() {
        worktree_root.to_path_buf()
    } else {
        repo_root.join(worktree_root)
    };
    std::fs::create_dir_all(&abs_root)?;
    warn_if_nested(repo_root, &abs_root);
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

/// Warn when worktrees are kept inside the repository they are worktrees of.
///
/// Build tools find their project root by walking *up* from the working directory, and a worktree
/// nested in the checkout has the original's project files above it. Cargo resolves such a worktree
/// to the outer workspace and builds into the outer `target/` — so the sandbox, which binds only
/// the worktree writable, fails on a read-only filesystem, and a build that did succeed would be
/// building the wrong tree. The same applies to any tool that resolves a root by walking up.
///
/// A warning rather than an error: the run still works for repositories whose acceptance command
/// does not care, and where the worktrees live is the operator's decision.
fn warn_if_nested(repo_root: &Path, worktree_root: &Path) {
    let (Ok(repo), Ok(root)) = (repo_root.canonicalize(), worktree_root.canonicalize()) else {
        return;
    };
    if is_nested(&repo, &root) {
        tracing::warn!(
            worktree_root = %root.display(),
            repo_root = %repo.display(),
            "worktrees are kept inside the repository; a build tool that finds its project root by \
             walking up will resolve the outer checkout instead of the worktree. Point `[worktree] \
             root` at a directory outside the repository."
        );
    }
}

/// Whether `worktree_root` is inside `repo_root`. Both are expected canonical, so a symlinked or
/// `..`-relative root is judged by where it actually lands rather than by how it was spelled.
fn is_nested(repo_root: &Path, worktree_root: &Path) -> bool {
    worktree_root.starts_with(repo_root)
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

/// Files whose diff removed or replaced a line that was already there.
///
/// The distinction the referee gate needs. Adding a test is work anyone would want from an
/// implementer; rewriting one is the shortcut the gate exists to refuse, and a path on its own
/// cannot tell the two apart. `--numstat` reports added and deleted counts per file, and a purely
/// additive change has a deleted count of zero.
///
/// A rename reports as a delete plus an add, so a renamed test file counts as rewritten. That is
/// the right answer: the test that used to run under that name no longer does.
/// Who a run's commits are authored by.
///
/// A pair rather than two arguments because git will not take one without the other: a name with
/// no address, or an address with no name, is a half-configured identity and git's own fallback
/// fills the gap from the environment — which is the person running this, and the one answer that
/// must not appear.
#[derive(Debug, Clone, Copy)]
pub struct Committer<'a> {
    pub name: &'a str,
    pub email: &'a str,
}

/// Commit everything in `worktree`, on the branch the run was given and no other.
///
/// Returns the new commit's sha, or `None` when there was nothing to commit.
///
/// The branch is checked against `expected` before anything is staged. A worktree is a checkout
/// like any other and `git commit` writes to whatever HEAD points at, so a run that has somehow
/// ended up on another branch must not commit there — the point of a per-run branch is that a run's
/// work lands on it and nowhere else.
///
/// Identity is set per invocation rather than read from the environment: a run is not the person
/// whose `user.name` happens to be configured, and a commit that claims otherwise is a lie in the
/// history.
pub async fn commit_all(
    worktree: &WorktreePath,
    expected: &str,
    message: &str,
    who: Committer<'_>,
) -> Result<Option<String>, ExecError> {
    let path = worktree.as_path();
    let head = git(
        path,
        "rev-parse --abbrev-ref HEAD",
        &["rev-parse", "--abbrev-ref", "HEAD"],
    )
    .await?
    .trim()
    .to_string();
    if head != expected {
        return Err(ExecError::Git {
            op: "commit".to_string(),
            code: -1,
            stderr: format!(
                "refusing to commit: this worktree is on `{head}`, and the run's branch is \
                 `{expected}`"
            ),
        });
    }

    git(path, "add -A", &["add", "-A"]).await?;
    // Nothing staged is an ordinary outcome — a run that changed nothing has nothing to record.
    let staged = git(
        path,
        "diff --cached --name-only",
        &["diff", "--cached", "--name-only"],
    )
    .await?;
    if staged.trim().is_empty() {
        return Ok(None);
    }
    git(
        path,
        "commit",
        &[
            "-c",
            &format!("user.name={}", who.name),
            "-c",
            &format!("user.email={}", who.email),
            "commit",
            "-q",
            "-m",
            message,
        ],
    )
    .await?;
    let sha = git(path, "rev-parse HEAD", &["rev-parse", "HEAD"])
        .await?
        .trim()
        .to_string();
    Ok(Some(sha))
}

pub async fn rewritten_files(worktree: &WorktreePath) -> Result<Vec<String>, ExecError> {
    let cwd = worktree.as_path();
    // As in `diff_stat`: `-N` makes a new file visible to `diff` without staging its content.
    git(cwd, "add -N", &["add", "-N", "."]).await?;
    let out = git(
        cwd,
        "diff --numstat",
        &["--no-pager", "diff", "--numstat", "HEAD"],
    )
    .await?;
    Ok(out.lines().filter_map(rewritten_in).collect())
}

/// One `--numstat` line, when it reports a deletion. Format is `added\tdeleted\tpath`, with `-`
/// for a binary file — which is counted as rewritten, since nothing about it can be called additive.
fn rewritten_in(line: &str) -> Option<String> {
    let mut fields = line.split('\t');
    let added = fields.next()?;
    let deleted = fields.next()?;
    let path = fields.next()?.trim();
    if path.is_empty() {
        return None;
    }
    let binary = added == "-" || deleted == "-";
    let removed_lines = deleted.parse::<u64>().unwrap_or(0) > 0;
    (binary || removed_lines).then(|| path.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_change_that_removes_an_existing_line_counts_as_rewritten() {
        // Adding a test is work worth having; rewriting one is the shortcut the referee refuses,
        // and the path alone cannot tell them apart.
        assert_eq!(rewritten_in("12\t0\tcrates/foo/tests/api.rs"), None);
        assert_eq!(
            rewritten_in("3\t4\tcrates/foo/tests/api.rs").as_deref(),
            Some("crates/foo/tests/api.rs")
        );
        assert_eq!(
            rewritten_in("0\t9\ttests/gone.rs").as_deref(),
            Some("tests/gone.rs")
        );
        // A binary file has no additive reading.
        assert_eq!(
            rewritten_in("-\t-\ttests/fixture.bin").as_deref(),
            Some("tests/fixture.bin")
        );
        assert_eq!(rewritten_in("garbage"), None);
    }

    #[test]
    fn a_worktree_root_inside_the_repository_is_nested() {
        // Why it matters: a build tool walking up from the worktree finds the outer project's
        // files first, builds the wrong tree, and writes where the sandbox mounts read-only.
        let repo = Path::new("/src/app");
        assert!(is_nested(repo, Path::new("/src/app/.ratatoskr/worktrees")));
        assert!(is_nested(repo, repo));

        assert!(!is_nested(repo, Path::new("/src/.ratatoskr-worktrees")));
        assert!(!is_nested(repo, Path::new("/var/tmp/worktrees")));
        // A sibling whose name merely starts with the repo's path is not inside it.
        assert!(!is_nested(repo, Path::new("/src/app-worktrees")));
    }

    fn who() -> Committer<'static> {
        Committer {
            name: "ratatoskr",
            email: "ratatoskr@localhost",
        }
    }

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
    async fn a_run_commits_to_its_own_branch_and_refuses_any_other() {
        let tmp = std::env::temp_dir().join(format!("ratatoskr-commit-{}", std::process::id()));
        let repo = tmp.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        init_repo(&repo).await;
        let wt = create(&repo, &tmp.join("wts"), "ratatoskr/abc12345")
            .await
            .unwrap();

        // Nothing changed is not a failure — a run that touched nothing has nothing to record.
        assert_eq!(
            commit_all(&wt, "ratatoskr/abc12345", "no-op", who())
                .await
                .unwrap(),
            None
        );

        std::fs::write(wt.as_path().join("new.rs"), "fn main() {}\n").unwrap();
        let sha = commit_all(&wt, "ratatoskr/abc12345", "feat: a thing", who())
            .await
            .unwrap()
            .expect("a change is committed");
        assert_eq!(sha.len(), 40, "{sha}");
        // And the branch actually moved, which is the whole point: a pushed branch with no commits
        // is what a pull request cannot be opened against.
        let log = git(wt.as_path(), "log", &["log", "--oneline", "-1"])
            .await
            .unwrap();
        assert!(log.contains("feat: a thing"), "{log}");

        // Authored by who the caller said, and never by whoever is configured on this machine. A
        // run is not a person, and a forge attributes a commit to whichever account owns the
        // address — so the wrong one here credits somebody with work they did not do.
        let author = git(
            wt.as_path(),
            "log author",
            &["log", "-1", "--format=%an <%ae>"],
        )
        .await
        .unwrap();
        assert_eq!(author.trim(), "ratatoskr <ratatoskr@localhost>");

        // Whatever the deployment configures is what lands, so this is genuinely a setting and not
        // a constant with a parameter in front of it.
        std::fs::write(wt.as_path().join("third.rs"), "fn third() {}\n").unwrap();
        commit_all(
            &wt,
            "ratatoskr/abc12345",
            "feat: another",
            Committer {
                name: "somebody else",
                email: "runs@example.invalid",
            },
        )
        .await
        .unwrap()
        .expect("a change is committed");
        let author = git(
            wt.as_path(),
            "log author",
            &["log", "-1", "--format=%an <%ae>"],
        )
        .await
        .unwrap();
        assert_eq!(author.trim(), "somebody else <runs@example.invalid>");

        // The guard: this worktree is on the run's branch, so committing to another name is
        // refused rather than silently landing work somewhere nobody will look for it.
        std::fs::write(wt.as_path().join("second.rs"), "fn other() {}\n").unwrap();
        let err = commit_all(&wt, "ratatoskr/deadbeef", "wrong branch", who())
            .await
            .expect_err("must refuse");
        assert!(format!("{err}").contains("the run's branch"), "{err}");

        let _ = remove(&repo, &wt).await;
        let _ = std::fs::remove_dir_all(&tmp);
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
