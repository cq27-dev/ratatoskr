//! Run a task's acceptance check in a sandbox, and turn its output into pass/fail.
//!
//! Both red-team (baseline checkout) and implementer (worktree) go through here, so the two runs
//! converge compares are produced the same way.
//!
//! Two halves, deliberately separated. [`run_acceptance`] is deterministic: it executes each step
//! and reports its exit code. [`Characterizer`] is a model that reads the raw output and names the
//! individual checks inside a step. There is no parser: a regex only ever understands the
//! frameworks someone taught it, and "compile to wasm, then drive it in a browser" is not one of
//! them.

use std::fmt::Write as _;
use std::path::Path;

use ratatoskr_core::{AcceptanceStep, ModelRoute, SandboxConfig};
use ratatoskr_exec::{Mount, SandboxSpec, sandbox_run};
use ratatoskr_mcp::ToolSet;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Where the repo/worktree is mounted inside the sandbox.
pub const GUEST_WORKSPACE: &str = "/workspace";

/// How much acceptance output the characterizer is shown in total, across every step.
///
/// A failing suite can emit megabytes, and a run has an unbounded number of steps — 40k per step
/// over N steps is a cost and denial-of-service surface, not a bound. This is a single total: the
/// tail is the part that matters (runners print their summary last), so the budget is spent from
/// the last step backwards and each cut is stated.
/// What one acceptance step did. Entirely deterministic; no model involved.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StepOutcome {
    pub name: String,
    pub command: Vec<String>,
    pub exit_code: i32,
    pub output: String,
}

impl StepOutcome {
    fn ok(&self) -> bool {
        self.exit_code == 0
    }
}

/// Per-check characterization of an acceptance run — the shape converge compares. A check failing
/// before and after is pre-existing; one that only fails after is a regression.
#[derive(Debug, Clone)]
pub struct TestResults {
    pub failing: Vec<String>,
    /// How many checks passed. At the exit-code floor this counts whole steps; when a
    /// characterizer read the output it counts individual checks.
    pub passed: usize,
    /// The first non-zero exit across the steps; zero only if every step succeeded.
    pub exit_code: i32,
    /// Combined output — context for the optional failure classifier.
    pub raw_output: String,
}

/// What an acceptance run needs: the policy, who is running it, and the two paths it spans.
pub struct Acceptance<'a> {
    pub cfg: &'a SandboxConfig,
    /// The node running these steps, so a failure in the log is attributable.
    pub node: &'a str,
    /// Sandbox name prefix; each step gets its own suffix.
    pub name: &'a str,
    /// The project. Its prepared dependency caches are shared by every run of it.
    pub repo_root: &'a Path,
    /// The tree these steps run in — this run's worktree, never the checkout.
    pub worktree: &'a Path,
    pub steps: &'a [AcceptanceStep],
}

/// The worktree, writable, plus whatever `prepare` left in the project's caches, read-only.
///
/// The caches are how a check runs offline in a tree that was just forked and has no dependencies
/// in it. Read-only for two reasons: several runs across several projects read them at once, and a
/// check that could write to one would be changing what every later run sees.
pub fn mounts_for(cfg: &SandboxConfig, repo_root: &Path, worktree: &Path) -> Vec<Mount> {
    let mut mounts = vec![Mount {
        host: worktree.to_path_buf(),
        guest: GUEST_WORKSPACE.to_string(),
        // Writable: a check builds, and a build writes — `target/`, `.pytest_cache`, a bundler's
        // output. The tree is the run's own worktree, never the checkout.
        read_only: false,
    }];
    mounts.extend(
        cfg.cache_mounts(repo_root, worktree)
            .into_iter()
            .map(|(host, guest)| Mount {
                host,
                guest: guest.display().to_string(),
                read_only: true,
            }),
    );
    mounts
}

/// Run each acceptance step in a sandbox, in order.
///
/// Steps run even after one fails: a later step's output frequently explains an earlier failure,
/// and stopping early would report a build error as "the tests did not run" with nothing to say
/// why. The exit code carries the failure regardless of where it happened.
pub async fn run_acceptance(a: Acceptance<'_>) -> Result<Vec<StepOutcome>, String> {
    let Acceptance {
        cfg,
        node,
        name,
        repo_root,
        worktree,
        steps,
    } = a;
    let mut outcomes = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        let spec = SandboxSpec {
            backend: cfg.backend.clone(),
            // Distinct per step: two sandboxes sharing a name is a collision, not a reuse.
            name: format!("{name}-{i}"),
            image: cfg.image.clone(),
            workdir: GUEST_WORKSPACE.to_string(),
            mounts: mounts_for(cfg, repo_root, worktree),
            command: step.command.clone(),
            cpus: 2,
            memory_mib: 2048,
            // Offline unless this step's program was named in `[sandbox] network_allow`. A test
            // that reaches the network fails for reasons the repository does not control; an
            // install step has to, and a repository whose deps are not vendored cannot check
            // anything until it has run.
            network: cfg.may_use_network(&step.command),
        };
        let out = sandbox_run(spec)
            .await
            .map_err(|e| format!("sandbox run of acceptance step `{}` failed: {e}", step.name))?;
        // Logged here because this is the run's most consequential deterministic result and the
        // only account of it otherwise is a model's paraphrase of it. A characterizer that
        // misreads a read-only-filesystem error as "cargo is not installed" sends whoever reads
        // the run after a problem that does not exist.
        let combined = format!("{}\n{}", out.stdout, out.stderr);
        // Attributed to the node running it. A suite takes minutes, and unattributed the node
        // that is plainly working looks idle for the whole of it to anything reading the stream.
        tracing::info!(
            kind = "acceptance_step",
            node,
            step = %step.name,
            command = %step.command.join(" "),
            exit_code = out.exit_code,
            output = %ratatoskr_agent::tail(combined.trim(), 2_000),
            "acceptance step finished"
        );
        outcomes.push(StepOutcome {
            name: step.name.clone(),
            command: step.command.clone(),
            exit_code: out.exit_code,
            output: combined,
        });
    }
    Ok(outcomes)
}

/// Pass/fail from exit codes alone, one result per step.
///
/// What a run without a characterizer gets, and the floor everything else is checked against: a
/// step that exited non-zero failed, and no model opinion is involved in that. Coarser than named
/// checks, never wrong about them.
pub fn by_exit_code(outcomes: &[StepOutcome]) -> TestResults {
    let (failing, passing): (Vec<_>, Vec<_>) = outcomes.iter().partition(|o| !o.ok());
    TestResults {
        failing: failing.iter().map(|o| o.name.clone()).collect(),
        passed: passing.len(),
        exit_code: outcomes
            .iter()
            .map(|o| o.exit_code)
            .find(|c| *c != 0)
            .unwrap_or(0),
        raw_output: joined_output(outcomes),
    }
}

fn joined_output(outcomes: &[StepOutcome]) -> String {
    let mut s = String::new();
    for o in outcomes {
        let _ = write!(
            s,
            "=== {} (exit {}) ===\n{}\n",
            o.name, o.exit_code, o.output
        );
    }
    s
}

#[cfg(test)]
const PREAMBLE: &str = include_str!("../prompts/characterizer.md");

/// What the model extracted from an acceptance run.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct CharacterizerOutput {
    #[serde(default)]
    failing: Vec<String>,
    /// How many checks passed — a count, never the names.
    ///
    /// Nothing downstream reads a passing check's name: converge compares failures, and the only
    /// other readers ask "did anything run" and "how many". Transcribing a few hundred identifiers
    /// to answer that is the single largest output in the pipeline, and it grows with the suite.
    #[serde(default)]
    passed: usize,
}

/// The deterministic acceptance evidence presented to one characterizer turn.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct CharacterizerInput {
    pub outcomes: Vec<StepOutcome>,
}

/// Reads an acceptance run's raw output and names the checks inside it.
///
/// Optional by design: with no `[models.characterizer]` route a run still converges on
/// [`by_exit_code`], comparing at step granularity. Coarser, never wrong.
pub struct Characterizer {
    pub route: ModelRoute,
    pub tools: ToolSet,
    pub max_turns: Option<usize>,
    /// Where its cost is charged. It runs on every acceptance run — twice per converge iteration —
    /// so leaving it unreported understated a run by one of its most frequent calls.
    pub ledger: Option<std::sync::Arc<ratatoskr_agent::RunLedger>>,
    /// The generic stage executor context used for characterization.
    pub(crate) declared_context: std::sync::Arc<crate::workflow::WorkflowContext>,
}

impl Characterizer {
    /// Name the checks in `outcomes`, falling back to exit codes whenever the answer cannot be
    /// trusted. A characterizer that cannot answer must not fail the run: the deterministic result
    /// is still there, and it is the one converge actually needs.
    pub async fn read(&self, outcomes: &[StepOutcome]) -> TestResults {
        let floor = by_exit_code(outcomes);
        let input = CharacterizerInput {
            outcomes: outcomes.to_vec(),
        };
        let input_json = match serde_json::to_string(&input) {
            Ok(input) => input,
            Err(error) => {
                tracing::warn!("serializing the acceptance run failed: {error}; using exit codes");
                return floor;
            }
        };
        let turn = crate::workflow::evaluate_standard_stage(
            std::sync::Arc::clone(&self.declared_context),
            "characterizer",
            input_json,
        )
        .await;
        let raw = match turn {
            Ok(raw) => raw,
            Err(e) => {
                tracing::warn!("characterizing the acceptance run failed: {e}; using exit codes");
                return floor;
            }
        };
        let Ok(read) = ratatoskr_graph::parse_validated::<CharacterizerOutput>(&raw) else {
            tracing::warn!("the characterization did not validate; using exit codes");
            return floor;
        };
        reconcile(read, floor)
    }
}

/// Hold the characterization to what the exit codes already prove.
///
/// The one invariant: a run where something failed must never characterize as nothing failing.
/// That is the direction that loses a real regression — converge would compare an empty failing set
/// against the baseline and call it converged — so it falls back rather than trusting the names.
/// The opposite direction needs no guard: extra named failures are at worst noise the loop fixes.
fn reconcile(read: CharacterizerOutput, floor: TestResults) -> TestResults {
    if floor.exit_code != 0 && read.failing.is_empty() {
        tracing::warn!(
            "a step failed but the characterization named no failing check; using exit codes"
        );
        return floor;
    }
    TestResults {
        failing: read.failing,
        // Never below what the exit codes already prove ran. A miscounted zero would read as "the
        // command never ran" downstream and strand a green suite.
        passed: read.passed.max(floor.passed),
        exit_code: floor.exit_code,
        raw_output: floor.raw_output,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_check_may_write_its_tree_and_may_not_write_the_cache() {
        // The distinction the whole prepared-cache design rests on. The worktree is this run's own
        // and a build writes to it. The cache is shared by every run of the project — several of
        // them at once — so a check that could write to one would be changing what the next run
        // sees, and the baseline and post-change runs would stop being comparable.
        let repo = std::env::temp_dir().join(format!("ratatoskr-mounts-{}", std::process::id()));
        let worktree = repo.join("wt");
        std::fs::create_dir_all(repo.join(ratatoskr_core::CACHE_ROOT).join("node")).unwrap();
        let cfg = SandboxConfig {
            cache: vec![ratatoskr_core::CacheMount {
                from: "node".into(),
                at: "web/node_modules".into(),
            }],
            ..Default::default()
        };

        let mounts = mounts_for(&cfg, &repo, &worktree);
        assert_eq!(mounts.len(), 2, "{mounts:?}");
        assert_eq!(mounts[0].host, worktree);
        assert!(!mounts[0].read_only, "the tree a build writes to");
        assert_eq!(
            mounts[1].host,
            repo.join(ratatoskr_core::CACHE_ROOT).join("node")
        );
        assert!(mounts[1].read_only, "the cache every run shares");
        // And it lands where the resolver looks, not where it was stored.
        assert_eq!(
            mounts[1].guest,
            worktree.join("web/node_modules").display().to_string()
        );

        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    fn the_characterizer_is_told_there_is_nobody_to_ask() {
        // On a live run it answered a failed acceptance step with "What would you like me to help
        // with?" and a list of options, having invented a diagnosis of the sandbox. Its turn
        // produced nothing, and the diagnosis was wrong and was believed.
        assert!(PREAMBLE.contains("no human"), "the fact it lacked");
        assert!(PREAMBLE.contains("exit code"), "why a guess is worse");
    }

    fn outcome(name: &str, exit_code: i32, output: &str) -> StepOutcome {
        StepOutcome {
            name: name.to_string(),
            command: vec!["run".to_string()],
            exit_code,
            output: output.to_string(),
        }
    }

    #[test]
    fn exit_codes_alone_characterize_a_run() {
        let outcomes = [
            outcome("wasm build", 0, "built"),
            outcome("browser tests", 1, "1 failed"),
        ];
        let results = by_exit_code(&outcomes);
        assert_eq!(results.passed, 1);
        assert_eq!(results.failing, ["browser tests"]);
        assert_eq!(results.exit_code, 1);
        // Every step's output is kept: a later step frequently explains an earlier failure.
        assert!(results.raw_output.contains("built"));
        assert!(results.raw_output.contains("1 failed"));
    }

    #[test]
    fn a_run_where_everything_passed_reports_checks_and_a_zero_exit() {
        let results = by_exit_code(&[outcome("a", 0, ""), outcome("b", 0, "")]);
        assert!(results.failing.is_empty());
        assert_eq!(results.exit_code, 0);
        // A non-zero count matters: `converge::test_command_ran` reads nothing-and-nonzero as "the
        // command never ran", so a run that checked something must say so.
        assert_eq!(results.passed, 2);
    }

    #[test]
    fn the_first_failure_sets_the_exit_code_wherever_it_happened() {
        assert_eq!(
            by_exit_code(&[outcome("a", 0, ""), outcome("b", 101, "")]).exit_code,
            101
        );
        assert_eq!(
            by_exit_code(&[outcome("a", 2, ""), outcome("b", 0, "")]).exit_code,
            2
        );
    }

    #[test]
    fn a_characterization_that_loses_a_failure_is_refused() {
        let floor = by_exit_code(&[outcome("browser tests", 1, "1 failed")]);
        // The dangerous direction: converge would compare an empty failing set against the
        // baseline and call a broken change converged.
        let blind = CharacterizerOutput {
            failing: Vec::new(),
            passed: 12,
        };
        let out = reconcile(blind, floor.clone());
        assert_eq!(
            out.failing,
            ["browser tests"],
            "falls back to the exit code"
        );

        // Named failures are taken as given — finer than the step, and the exit code still rules
        // whether the run passed.
        let named = CharacterizerOutput {
            failing: vec!["spec/login.spec.ts:12".into()],
            passed: 3,
        };
        let out = reconcile(named, floor);
        assert_eq!(out.failing, ["spec/login.spec.ts:12"]);
        assert_eq!(out.exit_code, 1);
    }

    #[test]
    fn exit_codes_and_the_deterministic_pass_floor_override_model_claims() {
        let floor = by_exit_code(&[
            outcome("build", 0, "built"),
            outcome("tests", 101, "one failed"),
        ]);
        let read = CharacterizerOutput {
            failing: vec!["suite::one_case".into()],
            passed: 0,
        };
        let out = reconcile(read, floor);
        assert_eq!(out.failing, ["suite::one_case"]);
        assert_eq!(out.passed, 1, "the model cannot erase a passing step");
        assert_eq!(out.exit_code, 101, "the model cannot rewrite the exit code");
    }

    #[test]
    fn a_clean_run_may_legitimately_name_no_failures() {
        let floor = by_exit_code(&[outcome("tests", 0, "ok")]);
        let read = CharacterizerOutput {
            failing: Vec::new(),
            passed: 41,
        };
        let out = reconcile(read, floor);
        assert!(out.failing.is_empty());
        assert_eq!(
            out.passed, 41,
            "the finer count is kept over the one-step floor"
        );
    }
}
