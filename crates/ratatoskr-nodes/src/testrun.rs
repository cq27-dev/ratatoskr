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

/// How much of a step's output the characterizer is shown, per step.
///
/// A failing suite can emit megabytes. The tail is the part that matters — runners print their
/// summary last — so this keeps the end and says where it cut.
const MAX_OUTPUT_CHARS: usize = 40_000;

/// What one acceptance step did. Entirely deterministic; no model involved.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Run each acceptance step in a sandbox, in order.
///
/// Steps run even after one fails: a later step's output frequently explains an earlier failure,
/// and stopping early would report a build error as "the tests did not run" with nothing to say
/// why. The exit code carries the failure regardless of where it happened.
pub async fn run_acceptance(
    cfg: &SandboxConfig,
    node: &str,
    name: &str,
    host_path: &Path,
    steps: &[AcceptanceStep],
) -> Result<Vec<StepOutcome>, String> {
    let mut outcomes = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        let spec = SandboxSpec {
            backend: cfg.backend.clone(),
            // Distinct per step: two sandboxes sharing a name is a collision, not a reuse.
            name: format!("{name}-{i}"),
            image: cfg.image.clone(),
            workdir: GUEST_WORKSPACE.to_string(),
            mounts: vec![Mount {
                host: host_path.to_path_buf(),
                guest: GUEST_WORKSPACE.to_string(),
            }],
            command: step.command.clone(),
            cpus: 2,
            memory_mib: 2048,
            network: false,
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

/// Keep the last `max` chars, saying where it cut. Runners print their summary last, so the tail
/// is the part that names what failed.
fn tail(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let kept: String = s.chars().skip(count - max).collect();
    format!("[earlier output omitted]\n{kept}")
}

const PREAMBLE: &str = include_str!("../prompts/characterizer.md");

/// What the model extracted from an acceptance run.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Characterization {
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
}

impl Characterizer {
    /// Name the checks in `outcomes`, falling back to exit codes whenever the answer cannot be
    /// trusted. A characterizer that cannot answer must not fail the run: the deterministic result
    /// is still there, and it is the one converge actually needs.
    pub async fn read(&self, outcomes: &[StepOutcome]) -> TestResults {
        let floor = by_exit_code(outcomes);
        let raw = match ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "characterizer",
            route: &self.route,
            preamble: PREAMBLE,
            question: &render_prompt(outcomes),
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(Characterization),
            policy: None,
            max_turns: self.max_turns,
            // It transcribes output. It has nothing to ask and nothing to be told.
            clarifier: None,
            observer: None,
            skills: Vec::new(),
            files: None,
            // Reads output it was handed, and touches neither the tree nor a shell.
            shell: None,
            push: None,
            conversation: None,
            ledger: self.ledger.clone(),
            // One turn over output it was handed: there is no history to outgrow, so a compaction
            // policy would only cost a summariser it never calls.
            produces: None,
        })
        .await
        {
            Ok(raw) => raw,
            Err(e) => {
                tracing::warn!("characterizing the acceptance run failed: {e}; using exit codes");
                return floor;
            }
        };
        let Ok(read) = ratatoskr_graph::parse_validated::<Characterization>(&raw) else {
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
fn reconcile(read: Characterization, floor: TestResults) -> TestResults {
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

fn render_prompt(outcomes: &[StepOutcome]) -> String {
    let mut s = String::new();
    for o in outcomes {
        let _ = write!(
            s,
            "=== STEP `{}` — `{}` — exit {} ===\n{}\n\n",
            o.name,
            o.command.join(" "),
            o.exit_code,
            tail(&o.output, MAX_OUTPUT_CHARS)
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let blind = Characterization {
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
        let named = Characterization {
            failing: vec!["spec/login.spec.ts:12".into()],
            passed: 3,
        };
        let out = reconcile(named, floor);
        assert_eq!(out.failing, ["spec/login.spec.ts:12"]);
        assert_eq!(out.exit_code, 1);
    }

    #[test]
    fn a_clean_run_may_legitimately_name_no_failures() {
        let floor = by_exit_code(&[outcome("tests", 0, "ok")]);
        let read = Characterization {
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

    #[test]
    fn the_tail_is_kept_because_runners_summarise_last() {
        let long: String = std::iter::repeat_n('x', 100)
            .chain("SUMMARY: 1 failed".chars())
            .collect();
        let cut = tail(&long, 30);
        assert!(cut.contains("SUMMARY: 1 failed"), "{cut}");
        assert!(cut.starts_with("[earlier output omitted]"));
        // Short output is handed over untouched.
        assert_eq!(tail("brief", 30), "brief");
    }
}
