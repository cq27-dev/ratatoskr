//! `Bash`: the tool a node runs commands with.
//!
//! Every call goes through the same sandbox the acceptance run uses. That is the point of running
//! commands here rather than handing the job to a coding CLI: the CLI decides for itself what it
//! may execute and asks a human when it is unsure, and there is no human. The boundary has to be
//! ours, and it has to be the one the run already trusts.
//!
//! The node supplies the sandbox — backend, mounts, network — because that is policy and it knows
//! it. This supplies only the command.

use std::sync::Arc;

use ratatoskr_exec::{SandboxSpec, sandbox_run};
use rig_agent::tool::{DynamicTool, ToolExecutionError};
use rig_core::tool::ToolOutput;
use rmcp::model::Tool;
use serde_json::json;

/// The name this is offered under. Named as the coding CLIs name it, so a prompt written for one
/// of them asks for the tool that exists.
pub const BASH: &str = "Bash";

/// How long one command may run.
///
/// Generous, because a cold `cargo test` legitimately takes minutes and a timeout that fires on
/// real work is worse than none. It exists for the command that never returns at all — a watcher,
/// a REPL, a `sleep` — which would otherwise hold the run open forever with nothing to report.
const DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// Most output one call reports back, per stream. A failing suite can emit megabytes; the tail is
/// the part that matters, because runners print their summary last.
const MAX_OUTPUT: usize = 24 * 1024;

/// What a node needs to offer `Bash`: the sandbox each command runs in.
///
/// `command` on the spec is ignored — this fills it per call — and `name` is suffixed so two
/// concurrent calls are two sandboxes rather than a collision.
#[derive(Debug, Clone)]
pub struct ShellAccess {
    pub spec: SandboxSpec,
}

/// The tool declaration.
pub fn declaration() -> Tool {
    let schema = json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command line to run, as you would type it in a shell. \
                    Interpreted by `sh -c`, so pipes, redirection and `&&` work. It runs in the \
                    worktree, inside a sandbox, with no network."
            },
            "description": {
                "type": "string",
                "description": "A few words on what this call is for, in the imperative — \
                    \"run the test suite\". Read by a person following the run."
            }
        },
        "required": ["command"]
    });
    let mut tool = Tool::default();
    tool.name = BASH.into();
    tool.description = Some(
        "Run a shell command in the worktree, inside a sandbox with no network access. Returns \
         the exit code with stdout and stderr. Use it to run builds, tests and linters, and to \
         inspect the tree with commands that have no dedicated tool — prefer Read, Grep and Glob \
         where they fit, since they answer without a shell."
            .to_string()
            .into(),
    );
    tool.input_schema = Arc::new(schema.as_object().cloned().expect("schema literal"));
    tool
}

/// The implementation, for a node that was given shell access.
pub fn implementation(name: &str, shell: &ShellAccess) -> Option<DynamicTool> {
    if name != BASH {
        return None;
    }
    let shell = shell.clone();
    Some(crate::answered_by(declaration(), move |_ctx, args| {
        let shell = shell.clone();
        Box::pin(async move { run(&shell, &args).await.map(ToolOutput::text) })
    }))
}

/// One call counter per process, so concurrent calls get distinct sandbox names.
static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

async fn run(shell: &ShellAccess, args: &serde_json::Value) -> Result<String, ToolExecutionError> {
    let command = args
        .get("command")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .ok_or_else(|| ToolExecutionError::invalid_args("`command` must be a non-empty string"))?;

    let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let spec = SandboxSpec {
        name: format!("{}-sh-{n}", shell.spec.name),
        // Through `sh -c` rather than split into argv here: the model writes a command line, and
        // splitting one correctly means implementing a shell's quoting rules. The sandbox is what
        // makes that safe to hand over — the shell runs inside it, not on the host.
        command: vec!["sh".to_string(), "-c".to_string(), command.to_string()],
        ..shell.spec.clone()
    };

    let output = match tokio::time::timeout(DEFAULT_TIMEOUT, sandbox_run(spec)).await {
        Err(_) => {
            return Err(ToolExecutionError::timeout(format!(
                "`{command}` did not finish within {}s and was abandoned",
                DEFAULT_TIMEOUT.as_secs()
            )));
        }
        Ok(Err(e)) => return Err(ToolExecutionError::other(format!("sandbox failed: {e}"))),
        Ok(Ok(output)) => output,
    };

    // A non-zero exit is reported as output, not as a tool error: it is usually the answer the
    // model asked for — the test failures it is about to fix — and an error would drop the body
    // that says what failed.
    let mut report = format!("exit {}", output.exit_code);
    for (stream, text) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
        let text = text.trim();
        if !text.is_empty() {
            report.push_str(&format!(
                "\n\n--- {stream} ---\n{}",
                crate::tail(text, MAX_OUTPUT)
            ));
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn access() -> ShellAccess {
        ShellAccess {
            spec: SandboxSpec {
                backend: "landlock".into(),
                name: "test".into(),
                image: "unused".into(),
                workdir: std::env::temp_dir().display().to_string(),
                mounts: Vec::new(),
                command: Vec::new(),
                cpus: 1,
                memory_mib: 512,
                network: false,
            },
        }
    }

    #[tokio::test]
    async fn an_empty_command_is_refused_before_any_sandbox_starts() {
        let err = run(&access(), &json!({ "command": "   " }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-empty"), "{err}");
    }

    #[test]
    fn the_declaration_says_where_the_command_runs() {
        // The model has to know it is sandboxed and offline, or it will read a network failure as
        // a broken repository and start working around it.
        let d = declaration();
        assert_eq!(d.name, BASH);
        let description = d.description.clone().unwrap_or_default().to_string();
        assert!(description.contains("sandbox"));
        assert!(description.contains("no network"));
    }

    #[tokio::test]
    #[ignore = "requires bwrap on the host; run with --ignored"]
    async fn a_command_runs_and_reports_its_exit_code_and_output() {
        let out = run(&access(), &json!({ "command": "echo hi && exit 3" }))
            .await
            .unwrap();
        assert!(out.starts_with("exit 3"), "{out}");
        assert!(out.contains("hi"), "{out}");
    }
}
