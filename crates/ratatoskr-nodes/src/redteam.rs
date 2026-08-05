//! Red-team: characterize the baseline test run. No LLM, no worktree — it changes nothing, so it
//! mounts the existing checkout into a sandbox and runs the repo's tests, parsing pass/fail
//! deterministically. Its output is the baseline converge compares the implementer's run against.

use std::path::PathBuf;

use ratatoskr_core::SandboxConfig;
use ratatoskr_graph::NodeError;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::testrun::run_tests;

/// Deterministic baseline characterization (strict schema — built from a real test run, not an LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RedTeamOutput {
    pub failing_tests: Vec<String>,
    pub passing_tests: Vec<String>,
    pub exit_code: i32,
}

/// The red-team node: run the baseline checkout's tests in a sandbox.
pub struct RedTeamNode {
    pub repo_path: PathBuf,
    pub sandbox: SandboxConfig,
    /// Unique sandbox name for this run.
    pub name: String,
}

impl RedTeamNode {
    pub async fn run(&self) -> Result<RedTeamOutput, NodeError> {
        let results = run_tests(&self.sandbox, &self.name, &self.repo_path)
            .await
            .map_err(NodeError::Failed)?;
        Ok(RedTeamOutput {
            failing_tests: results.failing,
            passing_tests: results.passing,
            exit_code: results.exit_code,
        })
    }
}
