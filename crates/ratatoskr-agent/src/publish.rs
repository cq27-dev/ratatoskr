//! The two tools that write somewhere other than this machine.
//!
//! A run's output has to leave the store to be worth anything, and the tracker already knows how to
//! receive it. Rather than reimplement a maintained CLI — its auth handling, its API versioning,
//! its fork resolution, its template discovery — the publisher drives `gh`.
//!
//! "Shell out" here does not mean a shell. `argv[0]` is fixed, the process is exec'd directly with
//! an argument list, and there is no interpreter to read a `;` or a `$(…)`. What a model supplies
//! is the argument list, and the allowlist below decides whether it runs.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_agent::tool::{DynamicTool, ToolExecutionError};
use rig_core::tool::ToolOutput;
use rmcp::model::Tool;
use serde_json::json;

/// The name this is offered under.
pub const GH: &str = "gh";

/// The name the push tool is offered under.
pub const PUSH: &str = "git_push";

/// How long one `gh` call may take. A tracker that is slow or unreachable must not hold a run open
/// indefinitely — the work is already done and checkpointed by the time this runs.
const CALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Most output one call reports back. `gh pr view` on a busy pull request is unbounded.
const MAX_OUTPUT: usize = 16 * 1024;

/// The subcommands that may run, as `(first, second)` argument pairs.
///
/// Exhaustive on purpose. An allowlist that enumerates what is permitted refuses a subcommand
/// nobody considered; a denylist permits it. `api` is the reason this distinction matters —
/// `gh api` is a general-purpose authenticated HTTP client wearing a subcommand's clothing, and
/// allowing it would make every other entry here decorative.
const ALLOWED: &[(&str, &str)] = &[
    ("pr", "create"),
    ("pr", "comment"),
    ("pr", "view"),
    ("pr", "list"),
    ("issue", "comment"),
    ("issue", "view"),
];

/// The tool declaration.
pub fn declaration() -> Tool {
    let schema = json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Arguments to `gh`, without the program name. Permitted: \
                    `pr create`, `pr comment`, `pr view`, `pr list`, `issue comment`, \
                    `issue view`. Anything else is refused."
            },
            "body": {
                "type": "string",
                "description": "Prose body for a pull request or comment. Pass it here rather \
                    than as a `--body` argument: it is written to a file and passed as \
                    `--body-file`, so newlines and backticks survive intact."
            }
        },
        "required": ["command"]
    });
    let mut tool = Tool::default();
    tool.name = GH.into();
    tool.description = Some(
        "Run a `gh` command against this repository's tracker. Use `pr view` or `issue view` to \
         check what already exists before creating anything — opening a second pull request for a \
         branch that has one is the failure worth avoiding. Bodies go in `body`, never in the \
         argument list."
            .to_string()
            .into(),
    );
    tool.input_schema = Arc::new(schema.as_object().cloned().expect("schema literal"));
    tool
}

/// The implementation, rooted at the repository `gh` runs against.
pub fn implementation(name: &str, root: &Path) -> Option<DynamicTool> {
    if name != GH {
        return None;
    }
    let root = root.to_path_buf();
    Some(crate::answered_by(declaration(), move |_ctx, args| {
        let root = root.clone();
        Box::pin(async move { run(&root, &args).await.map(ToolOutput::text) })
    }))
}

/// Whether `args` name a permitted subcommand.
///
/// The subcommand must be the first two arguments, exactly. No search, no skipping of leading
/// flags: skipping them means deciding which tokens are flag *values*, and a rule that has to guess
/// that is a rule something can be slipped past — `--repo o/r pr view` reads as the subcommand
/// `o/r pr` to a filter that drops only the tokens beginning with a dash.
///
/// `gh pr view --repo o/r` is the ordinary form and works. Leading flags are refused, which costs
/// a caller one reordering and costs an attacker the ambiguity.
fn permitted(args: &[String]) -> bool {
    match (args.first(), args.get(1)) {
        (Some(a), Some(b)) => ALLOWED.iter().any(|(x, y)| a == x && b == y),
        _ => false,
    }
}

async fn run(root: &Path, args: &serde_json::Value) -> Result<String, ToolExecutionError> {
    let command: Vec<String> = args
        .get("command")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str())
                .map(str::to_string)
                .collect()
        })
        .ok_or_else(|| ToolExecutionError::invalid_args("`command` must be an array of strings"))?;

    if !permitted(&command) {
        let allowed = ALLOWED
            .iter()
            .map(|(a, b)| format!("`{a} {b}`"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ToolExecutionError::invalid_args(format!(
            "refused: only {allowed} may be run"
        )));
    }
    // The body belongs in a file, so a `--body` in the argument list means the caller is about to
    // put multi-line prose through argv. Refused rather than fixed up, because the fix is one word.
    if command.iter().any(|a| a == "--body" || a == "-b") {
        return Err(ToolExecutionError::invalid_args(
            "pass the text in `body`, not as a `--body` argument",
        ));
    }

    let mut argv = command;
    // Written by this code rather than by the caller: a body is prose with newlines and backticks
    // in it, and argv is where quoting bugs and injected arguments both live.
    let body_file = match args.get("body").and_then(|v| v.as_str()) {
        Some(body) if !body.trim().is_empty() => {
            let path = body_path(root);
            std::fs::write(&path, body)
                .map_err(|e| ToolExecutionError::other(format!("could not stage the body: {e}")))?;
            argv.push("--body-file".to_string());
            argv.push(path.display().to_string());
            Some(path)
        }
        _ => None,
    };

    let output = tokio::time::timeout(
        CALL_TIMEOUT,
        tokio::process::Command::new(GH)
            .args(&argv)
            .current_dir(root)
            .output(),
    )
    .await;
    if let Some(path) = body_file {
        let _ = std::fs::remove_file(path);
    }

    let output = match output {
        Err(_) => {
            return Err(ToolExecutionError::timeout(format!(
                "`gh` did not finish within {}s",
                CALL_TIMEOUT.as_secs()
            )));
        }
        Ok(Err(e)) => {
            return Err(ToolExecutionError::other(format!(
                "could not run `gh`: {e} (is it installed and authenticated?)"
            )));
        }
        Ok(Ok(output)) => output,
    };

    let stdout = truncate(&String::from_utf8_lossy(&output.stdout));
    let stderr = truncate(&String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        // Returned as an error so the model sees a failure as a failure. `gh` writes its
        // diagnostics to stderr, and they are usually actionable — "no pull requests found",
        // "already exists".
        return Err(ToolExecutionError::other(format!(
            "`gh {}` failed (exit {}): {stderr}",
            argv.join(" "),
            output.status.code().unwrap_or(-1)
        )));
    }
    Ok(match stdout.trim().is_empty() {
        true => format!("done. {stderr}").trim().to_string(),
        false => stdout,
    })
}

/// Where a body is staged. Inside the repository so it is on the same filesystem, and named so a
/// leftover is identifiable rather than mysterious.
fn body_path(root: &Path) -> PathBuf {
    root.join(format!(".ratatoskr-gh-body-{}", std::process::id()))
}

fn truncate(s: &str) -> String {
    match s.char_indices().nth(MAX_OUTPUT) {
        None => s.to_string(),
        Some((at, _)) => format!("{}\n[output truncated]", &s[..at]),
    }
}

/// The branch a run may push, and the repository it lives in.
///
/// Bound when the publisher is built, from the run that created the branch. This is the whole
/// safety argument: the tool takes NO arguments, so there is no branch name, remote, refspec or
/// flag for a model to supply. An allowlist would have to decide whether a supplied string is
/// acceptable; there is no supplied string to decide about.
#[derive(Debug, Clone)]
pub struct PushAccess {
    /// Where `git` runs. Worktrees share the main checkout's refs, so the repository root can push
    /// a branch that is checked out in a linked worktree.
    pub repo_root: PathBuf,
    /// The branch this run authored. Must be one this repository manages.
    pub branch: String,
}

/// Whether `branch` is one a run authored and may therefore push.
///
/// The prefix is how this repository marks the branches it creates and reclaims, so it is also the
/// right boundary for what a run may publish. Everything else is someone's work: `main`, a
/// colleague's feature branch, a release branch.
///
/// The rest of the checks are about the name being a name. A refspec is `src:dst`, a leading `-`
/// is a flag, and `..`/`~`/`^`/`?`/`*`/`[`/whitespace are revision syntax — none of which can
/// appear in a branch this repository created, so a name carrying one is not one of ours however
/// it came to be constructed.
pub fn pushable(branch: &str) -> bool {
    branch.starts_with("ratatoskr/")
        && !branch.starts_with('-')
        && !branch.contains("..")
        && !branch.contains([':', '~', '^', '?', '*', '[', '\\', ' ', '\t', '\n', '\r'])
}

/// Exactly what gets exec'd. Split out so the argument list is a thing a test can assert on.
///
/// Fully-qualified on both sides of the refspec: `push origin <branch>` consults `push.default`
/// and the remote's configured refspecs to decide what it means, and this must mean one thing.
/// No `--force`, no `--tags`, no `--delete` — a run publishes its own work and never rewrites
/// anyone's history.
fn push_argv(branch: &str) -> Vec<String> {
    vec![
        "push".to_string(),
        "--set-upstream".to_string(),
        "origin".to_string(),
        format!("refs/heads/{branch}:refs/heads/{branch}"),
    ]
}

/// The push tool's declaration: no parameters, deliberately.
pub fn push_declaration() -> Tool {
    let mut tool = Tool::default();
    tool.name = PUSH.into();
    tool.description = Some(
        "Push this run's own branch to `origin`, so a pull request can be opened against it. \
         Takes no arguments: the branch is the one this run created, and no other branch can be \
         pushed. Call it before `gh pr create` — a pull request cannot be opened for a branch the \
         remote has never seen."
            .to_string()
            .into(),
    );
    tool.input_schema = Arc::new(
        json!({ "type": "object", "properties": {} })
            .as_object()
            .cloned()
            .expect("schema literal"),
    );
    tool
}

/// The implementation, bound to one run's branch.
pub fn push_implementation(name: &str, access: &PushAccess) -> Option<DynamicTool> {
    if name != PUSH {
        return None;
    }
    let access = access.clone();
    Some(crate::answered_by(
        push_declaration(),
        move |_ctx, _args| {
            let access = access.clone();
            Box::pin(async move { push(&access).await.map(ToolOutput::text) })
        },
    ))
}

async fn push(access: &PushAccess) -> Result<String, ToolExecutionError> {
    // Checked here rather than only at construction: this is the last point before the process
    // runs, and it is the check that has to hold no matter how the access came to be built.
    if !pushable(&access.branch) {
        return Err(ToolExecutionError::other(format!(
            "refusing to push `{}`: a run may push only the branch it authored",
            access.branch
        )));
    }
    let out = tokio::time::timeout(
        CALL_TIMEOUT,
        tokio::process::Command::new("git")
            .arg("-C")
            .arg(&access.repo_root)
            .args(push_argv(&access.branch))
            .output(),
    )
    .await
    .map_err(|_| ToolExecutionError::other("git push timed out"))?
    .map_err(|e| ToolExecutionError::other(format!("git push failed to start: {e}")))?;

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if out.status.success() {
        // git reports a push on stderr, so both streams carry the answer.
        Ok(truncate(&format!(
            "pushed {}\n{stdout}{stderr}",
            access.branch
        )))
    } else {
        // Returned as output, not as an error: the publisher can still comment on the issue, and
        // an error would read to it as the tool being broken rather than the push being refused.
        Ok(truncate(&format!(
            "push failed (exit {}). The branch is not on the remote, so a pull request cannot be \
             opened for it.\n{stdout}{stderr}",
            out.status.code().unwrap_or(-1)
        )))
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn only_a_branch_this_run_authored_may_be_pushed() {
        assert!(pushable("ratatoskr/abc12345"));

        // Someone else's work, which is the whole point of the guard.
        assert!(!pushable("main"));
        assert!(!pushable("master"));
        assert!(!pushable("release/2.0"));
        assert!(!pushable("feature/colleagues-work"));

        // Prefix games: the marker has to start the name, not appear in it.
        assert!(!pushable("not-ratatoskr/abc"));
        assert!(!pushable("../ratatoskr/abc"));

        // A branch name that is trying to be something else. None of these can occur in a name
        // this repository created, so any of them means the name did not come from us.
        assert!(!pushable("ratatoskr/a:refs/heads/main"));
        assert!(!pushable("ratatoskr/a b"));
        assert!(!pushable("ratatoskr/a~1"));
        assert!(!pushable("ratatoskr/a^"));
        assert!(!pushable("ratatoskr/a..b"));
        assert!(!pushable("ratatoskr/*"));
    }

    #[tokio::test]
    async fn a_push_of_someone_elses_branch_is_refused_at_the_last_moment() {
        // The construction site filters too, but this is the check that has to hold however the
        // access was built — it is the last point before a process runs.
        let access = PushAccess {
            repo_root: ".".into(),
            branch: "main".to_string(),
        };
        let err = push(&access).await.expect_err("must refuse");
        assert!(
            format!("{err}").contains("only the branch it authored"),
            "{err}"
        );
    }

    #[test]
    fn the_push_command_is_fully_qualified_and_never_forced() {
        let argv = push_argv("ratatoskr/abc12345");
        assert_eq!(
            argv,
            [
                "push",
                "--set-upstream",
                "origin",
                "refs/heads/ratatoskr/abc12345:refs/heads/ratatoskr/abc12345",
            ]
        );
        // The properties that matter, stated so a future edit has to break them deliberately.
        assert!(!argv.iter().any(|a| a.contains("force")), "{argv:?}");
        assert!(
            !argv.iter().any(|a| a == "--tags" || a == "--delete"),
            "{argv:?}"
        );
    }

    #[test]
    fn the_push_tool_takes_no_arguments_at_all() {
        // The safety argument in one assertion: there is no branch, remote, refspec or flag for a
        // model to supply, so there is no supplied string to validate.
        let d = push_declaration();
        assert_eq!(d.name, PUSH);
        let props = d
            .input_schema
            .get("properties")
            .expect("schema has properties");
        assert!(props.as_object().is_some_and(|p| p.is_empty()), "{props:?}");
        assert!(d.input_schema.get("required").is_none());
    }
    use super::*;

    fn argv(args: &[&str]) -> Vec<String> {
        args.iter().map(|a| (*a).to_string()).collect()
    }

    #[test]
    fn only_the_enumerated_subcommands_run() {
        assert!(permitted(&argv(&["pr", "create", "--title", "x"])));
        assert!(permitted(&argv(&["issue", "comment", "79"])));
        // Flags go after the subcommand, which is the ordinary form.
        assert!(permitted(&argv(&["pr", "view", "--repo", "o/r"])));
        // And a leading flag is refused rather than searched past: deciding which token is a flag's
        // *value* is exactly the ambiguity something gets slipped through.
        assert!(!permitted(&argv(&["--repo", "o/r", "pr", "view"])));

        // The one that makes the rest of the list meaningful: an authenticated HTTP client with a
        // subcommand's name on it.
        assert!(!permitted(&argv(&["api", "-X", "DELETE", "/repos/o/r"])));
        assert!(!permitted(&argv(&["auth", "token"])));
        assert!(!permitted(&argv(&["pr", "merge", "1"])));
        assert!(!permitted(&argv(&["repo", "delete", "o/r"])));
        assert!(!permitted(&argv(&["release", "create", "v1"])));
        // A subcommand nobody thought about is refused by the list being exhaustive rather than
        // permitted by a denylist that never heard of it.
        assert!(!permitted(&argv(&["gist", "create"])));
        assert!(!permitted(&argv(&["pr"])));
        assert!(!permitted(&[]));
    }

    #[tokio::test]
    async fn a_refused_command_never_reaches_the_process() {
        // `gh` may not even be installed here; the point is that the refusal happens before any
        // attempt to run something.
        let root = std::env::temp_dir();
        let err = run(&root, &json!({ "command": ["api", "/user"] }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("refused"), "{err}");
        assert!(
            err.contains("`pr create`"),
            "the message names what is allowed: {err}"
        );
    }

    #[tokio::test]
    async fn a_body_in_the_argument_list_is_refused_with_the_fix() {
        let root = std::env::temp_dir();
        let err = run(
            &root,
            &json!({ "command": ["pr", "create", "--body", "line one\nline two"] }),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("pass the text in `body`"), "{err}");
    }

    #[test]
    fn the_declaration_tells_the_model_to_look_before_it_creates() {
        // Opening a second pull request for a branch that already has one is the failure this
        // tool's shape is meant to make avoidable.
        let d = declaration();
        assert_eq!(d.name, GH);
        let description = d.description.clone().unwrap_or_default().to_string();
        assert!(description.contains("view"));
        assert!(description.contains("second pull request"));
    }
}
