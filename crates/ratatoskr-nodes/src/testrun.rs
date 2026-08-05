//! Shared helper: run the target repo's test command in a sandbox and parse per-test pass/fail.
//!
//! Both red-team (baseline checkout) and implementer (worktree) run tests this way. The parser is
//! `cargo test`-shaped (`test <name> ... ok|FAILED`) — the one CLI target for this phase; other
//! frameworks are additive later.

use std::path::Path;

use ratatoskr_core::SandboxConfig;
use ratatoskr_exec::{Mount, SandboxSpec, sandbox_run};

/// Where the repo/worktree is mounted inside the sandbox.
pub const GUEST_WORKSPACE: &str = "/workspace";

/// Deterministic per-test characterization of a test run.
#[derive(Debug, Clone)]
pub struct TestResults {
    pub failing: Vec<String>,
    pub passing: Vec<String>,
    pub exit_code: i32,
}

/// Run `cfg.test_command` against `host_path` mounted into a sandbox named `name`.
pub async fn run_tests(
    cfg: &SandboxConfig,
    name: &str,
    host_path: &Path,
) -> Result<TestResults, String> {
    let spec = SandboxSpec {
        backend: cfg.backend.clone(),
        name: name.to_string(),
        image: cfg.image.clone(),
        workdir: GUEST_WORKSPACE.to_string(),
        mounts: vec![Mount {
            host: host_path.to_path_buf(),
            guest: GUEST_WORKSPACE.to_string(),
        }],
        command: cfg.test_command.clone(),
        cpus: 2,
        memory_mib: 2048,
        network: false,
    };
    let out = sandbox_run(spec)
        .await
        .map_err(|e| format!("sandbox test run failed: {e}"))?;
    let (failing, passing) = parse_cargo_test_output(&out.stdout, &out.stderr);
    Ok(TestResults {
        failing,
        passing,
        exit_code: out.exit_code,
    })
}

/// Parse `cargo test` output into (failing, passing) test names.
pub fn parse_cargo_test_output(stdout: &str, stderr: &str) -> (Vec<String>, Vec<String>) {
    let mut failing = Vec::new();
    let mut passing = Vec::new();
    for line in stdout.lines().chain(stderr.lines()) {
        let Some(rest) = line.strip_prefix("test ") else {
            continue;
        };
        if let Some(name) = rest.strip_suffix(" ... ok") {
            passing.push(name.trim().to_string());
        } else if let Some(name) = rest.strip_suffix(" ... FAILED") {
            failing.push(name.trim().to_string());
        }
    }
    (failing, passing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cargo_test_lines() {
        let stdout = "\
running 3 tests
test store::tests::opens ... ok
test store::tests::writes ... FAILED
test config::tests::parses ... ok
";
        let (failing, passing) = parse_cargo_test_output(stdout, "");
        assert_eq!(failing, ["store::tests::writes"]);
        assert_eq!(passing, ["store::tests::opens", "config::tests::parses"]);
    }
}
