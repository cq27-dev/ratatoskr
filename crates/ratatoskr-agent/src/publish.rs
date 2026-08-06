//! The one tool that writes somewhere other than this machine.
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
    let declaration = declaration();
    let schema = serde_json::Value::Object((*declaration.input_schema).clone());
    let description = declaration
        .description
        .clone()
        .unwrap_or_default()
        .to_string();
    let root = root.to_path_buf();

    Some(DynamicTool::new(
        GH.to_string(),
        description,
        schema,
        move |_ctx, args| {
            let root = root.clone();
            Box::pin(async move { run(&root, &args).await.map(ToolOutput::text) })
        },
    ))
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

#[cfg(test)]
mod tests {
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
