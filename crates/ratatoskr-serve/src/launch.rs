//! Launching runs on behalf of the dashboard.
//!
//! A run is a **child process** (`ratatoskr run`), never an in-process call. Three reasons:
//!
//! - It keeps this server read-only against the store. The store's single-writer discipline holds
//!   because `serve` never writes; running in-process would make it a writer.
//! - A run's repo comes from the process's working directory, which is process-global. Spawning
//!   with an explicit `cwd` is what makes more than one project possible at all.
//! - Crash isolation, and it drives the same code path the CLI already exercises rather than a
//!   second implementation of it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::process::Command;
use tokio::sync::Semaphore;

/// Why a run couldn't be started.
#[derive(Debug, thiserror::Error)]
pub enum LaunchError {
    #[error("issue text is empty")]
    EmptyIssue,
    #[error(
        "at capacity: {0} run(s) already in flight — wait, or restart `serve` with a higher \
         --max-runs"
    )]
    AtCapacity(usize),
    #[error("could not locate the ratatoskr binary: {0}")]
    NoBinary(std::io::Error),
    #[error("failed to spawn a run: {0}")]
    Spawn(std::io::Error),
}

/// Spawns runs, bounded by a fixed number of concurrent children.
pub struct Launcher {
    /// Working directory for spawned runs — the project's repo root.
    dir: PathBuf,
    config: PathBuf,
    /// Where a run should reach this server to ask a human a question.
    dashboard: String,
    /// Which project the run belongs to. A run id is only unique *within* a project — an
    /// operator-chosen `--run-id` can repeat across them — so questions are scoped by both.
    project: String,
    permits: Arc<Semaphore>,
    max: usize,
}

impl Launcher {
    pub fn new(dir: &Path, config: &Path, max: usize, dashboard: &str, project: &str) -> Self {
        let max = max.max(1);
        Launcher {
            dir: dir.to_path_buf(),
            config: config.to_path_buf(),
            dashboard: dashboard.to_string(),
            project: project.to_string(),
            permits: Arc::new(Semaphore::new(max)),
            max,
        }
    }

    /// Start a run for `issue` and return its id immediately, without waiting for it to finish.
    ///
    /// Refuses rather than queues when every permit is taken. A queued run would exist only in
    /// this server's memory — invisible in the store, lost on restart, and impossible to show
    /// honestly in a UI whose entire model is "what the store recorded". Refusing keeps every
    /// accepted run real.
    pub fn spawn(&self, issue: &str) -> Result<String, LaunchError> {
        if issue.trim().is_empty() {
            return Err(LaunchError::EmptyIssue);
        }
        let permit = Arc::clone(&self.permits)
            .try_acquire_owned()
            .map_err(|_| LaunchError::AtCapacity(self.max))?;

        // The currently running binary, not a PATH lookup: `serve` and the run it starts should
        // always be the same build.
        let exe = std::env::current_exe().map_err(LaunchError::NoBinary)?;
        let run_id = uuid::Uuid::new_v4().to_string();

        let mut cmd = Command::new(exe);
        // A run outlives the dashboard by design: it owns a worktree and real API spend, and is
        // followed through the store rather than this process. Children inherit the server's
        // process group by default, so a Ctrl-C on a foreground `serve` would signal the whole
        // group and take running work down with it.
        #[cfg(unix)]
        cmd.process_group(0);
        // How the run reaches back to ask a human. A run started any other way doesn't get this,
        // and answers its own questions exactly as it does today.
        cmd.env("RATATOSKR_DASHBOARD", &self.dashboard);
        cmd.env("RATATOSKR_PROJECT", &self.project);
        let mut child = cmd
            .arg("run")
            .arg("--run-id")
            .arg(&run_id)
            .arg("--config")
            .arg(&self.config)
            // Everything after `--` is positional. Issue text is untrusted input and may start
            // with a dash; without this it would be parsed as a flag.
            .arg("--")
            .arg(issue)
            .current_dir(&self.dir)
            .kill_on_drop(false)
            .spawn()
            .map_err(LaunchError::Spawn)?;

        // Hold the permit until the child exits, and reap it so it doesn't linger as a zombie.
        let id = run_id.clone();
        tokio::spawn(async move {
            // `run_id` as a field, not just in the message: these are emitted in the server
            // process, outside the child's `run` span, so the field is all a consumer has to
            // attribute them by.
            match child.wait().await {
                Ok(status) if status.success() => {
                    tracing::info!(kind = "run_finished", run_id = %id, "run finished")
                }
                Ok(status) => {
                    tracing::warn!(kind = "run_failed", run_id = %id, %status, "run exited")
                }
                Err(e) => {
                    tracing::warn!(kind = "run_failed", run_id = %id, error = %e, "run not reaped")
                }
            }
            drop(permit);
        });

        Ok(run_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launcher(max: usize) -> Launcher {
        Launcher::new(
            Path::new("."),
            Path::new("ratatoskr.toml"),
            max,
            "http://127.0.0.1:7878",
            "test",
        )
    }

    #[tokio::test]
    async fn an_empty_issue_is_refused_before_anything_is_spawned() {
        assert!(matches!(
            launcher(1).spawn("   \n "),
            Err(LaunchError::EmptyIssue)
        ));
    }

    #[tokio::test]
    async fn capacity_is_a_refusal_not_a_queue() {
        // Take the only permit directly rather than spawning a real run: this asserts the
        // refusal itself, and does so identically whether or not the environment can spawn.
        let l = launcher(1);
        let _in_flight = Arc::clone(&l.permits)
            .try_acquire_owned()
            .expect("the first permit is free");

        assert!(matches!(
            l.spawn("an issue while busy"),
            Err(LaunchError::AtCapacity(1))
        ));
    }

    #[test]
    fn a_zero_cap_is_clamped_rather_than_deadlocking() {
        assert_eq!(launcher(0).max, 1);
    }
}
