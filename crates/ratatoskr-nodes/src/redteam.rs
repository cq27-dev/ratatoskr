//! Red-team: characterize the baseline test run. No LLM in the core path — it runs the repo's
//! tests in a worktree of its own and parses pass/fail deterministically. That deterministic
//! characterization is what converge compares the implementer's run against.
//!
//! An OPTIONAL classifier (enabled only when redteam has a model route — from `[models.redteam]` or
//! its ruleset) adds one LLM
//! pass labeling each baseline failure flaky vs real — a separate, additive field. The strict
//! pass/fail is never touched by it.

use std::path::PathBuf;

use ratatoskr_exec::{WorktreePath, create_worktree, delete_worktree_branch, remove_worktree};
use ratatoskr_graph::{NodeError, parse_validated};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::testrun::{Acceptance, Characterizer, by_exit_code, run_acceptance};

/// One baseline failure's classification. Additive context, not part of the strict pass/fail.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FailureClassification {
    pub test: String,
    /// "flaky" or "real" (anything else is treated as unknown by consumers).
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub reason: String,
}

/// The compose model's output.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct Classification {
    #[serde(default)]
    pub(crate) classifications: Vec<FailureClassification>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct ClassifierInput {
    pub failing: Vec<String>,
    pub raw_output: String,
}

/// Deterministic baseline characterization (strict schema — built from a real test run, not an LLM).
/// `classifications` is optional/additive: empty unless a redteam classifier ran (it runs when
/// redteam has a route from `[models.redteam]` or its ruleset).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RedTeamOutput {
    /// Tests written for this change before the code was, when the red team had a route to write
    /// them with.
    #[serde(default)]
    pub authored: Option<AuthoredTests>,
    pub failing_tests: Vec<String>,
    /// How many checks passed. Only the count is carried: nothing reads a passing check's name,
    /// and a suite of several hundred costs the characterizer more output than the rest of the
    /// pipeline combined to write out.
    #[serde(default)]
    pub passed_tests: usize,
    pub exit_code: i32,
    #[serde(default)]
    pub classifications: Vec<FailureClassification>,
}

/// Optional LLM classifier for baseline failures.
///
/// Carries nothing but the stage context: the classification turn runs through the stage executor,
/// which resolves this stage's route, tools, capability ceiling, prompt and ledger from the run's
/// registry. Holding a second copy of any of that here could only disagree with what runs.
pub struct RedTeamClassifier {
    /// The generic stage executor context used for classification.
    pub(crate) declared_context: std::sync::Arc<crate::workflow::WorkflowContext>,
}

impl RedTeamClassifier {
    async fn classify(
        &self,
        failing: &[String],
        raw_output: &str,
    ) -> Result<Vec<FailureClassification>, NodeError> {
        let input = ClassifierInput {
            failing: failing.to_vec(),
            raw_output: raw_output.to_string(),
        };
        let input_json = serde_json::to_string(&input)
            .map_err(|error| NodeError::Failed(format!("red-team classifier input: {error}")))?;
        let raw = crate::workflow::evaluate_standard_stage(
            std::sync::Arc::clone(&self.declared_context),
            "redteam_classifier",
            input_json,
        )
        .await
        .map_err(|error| NodeError::Failed(format!("red-team classifier failed: {error}")))?;
        Ok(parse_validated::<Classification>(&raw)?.classifications)
    }
}

/// The red-team node: run the baseline checkout's tests in a sandbox, optionally classify failures.
/// What the red team wrote, and what it says those tests cover.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AuthoredTests {
    /// Files written or extended, as repository-relative paths.
    #[serde(default)]
    pub files: Vec<String>,
    /// The tests written, named as the test runner will report them.
    ///
    /// Not decoration: a test written before the code fails in the baseline by construction, so
    /// convergence — which asks only whether anything *newly* fails — would pass a change that
    /// ignored every one of them. Naming them is what lets the run require them to pass.
    #[serde(default)]
    pub tests: Vec<String>,
    /// One line per file on what it covers, or why nothing was written.
    #[serde(default)]
    pub covers: String,
    /// Authored tests that did **not** fail on the baseline, and so gate nothing.
    ///
    /// A test written before the code is supposed to fail without it. One that passes anyway is
    /// either asserting behaviour that already existed or not running at all, and in both cases it
    /// proves the change did nothing — so it is moved out of [`tests`](Self::tests) rather than
    /// left there to be satisfied for free.
    ///
    /// Kept rather than discarded because the fallback has to be *said*. A run whose task had no
    /// testable outcome should fall back to the ordinary gate openly; one that quietly dropped an
    /// unprovable test would look identical to one that never wrote a test at all.
    ///
    /// Absent from the generated schema — `schemars(skip)` — because this struct is two things:
    /// the contract the authoring model is validated against, and the record the run carries. The
    /// model does not write this field; the baseline decides it afterwards. Putting it in the
    /// contract would ask the model for an answer only the run can give, and
    /// `standard_redteam_contracts_match_the_typed_output_gates` is what noticed.
    #[serde(default)]
    #[schemars(skip)]
    pub unproven: Vec<String>,
}

impl AuthoredTests {
    /// Keep only the tests the proving run showed fail without the change.
    ///
    /// `seeded` is the pass that had these files copied in; `clean` is the pass that never saw
    /// them and is the only one converge compares against. Matching is by the name the runner
    /// reports, which is what [`tests`](Self::tests) is documented to hold — see
    /// [`RedTeamNode::proven`] for why the two runs are separate.
    fn proven_by(self, seeded: &RedTeamOutput, clean: &RedTeamOutput) -> Self {
        // The suite ran clean and then ran nothing at all with these files in it: they broke the
        // build. That is proof, not absence — a test that cannot compile certainly does not pass
        // without the change, which is the only question being asked. Reading it as "unproven"
        // would throw the gate away in the case the authoring prompt calls expected.
        if clean.passed_tests > 0 && seeded.passed_tests == 0 {
            tracing::info!(
                kind = "authored_tests_proven",
                tests = %self.tests.join(", "),
                "the authored tests stop the suite building without the change"
            );
            return self;
        }
        let (tests, unproven): (Vec<String>, Vec<String>) = self
            .tests
            .into_iter()
            .partition(|t| seeded.failing_tests.contains(t));
        if !unproven.is_empty() {
            tracing::warn!(
                kind = "unproven_tests",
                tests = %unproven.join(", "),
                "these authored tests did not fail without the change, so they gate nothing"
            );
        }
        Self {
            tests,
            unproven: [self.unproven, unproven].concat(),
            ..self
        }
    }

    /// Nothing could be proven, so nothing gates — and the run says which way it failed.
    fn none_proven(self, why: &str) -> Self {
        if !self.tests.is_empty() {
            tracing::warn!(
                kind = "unproven_tests",
                tests = %self.tests.join(", "),
                "{why}; the change is judged by the tests that already exist"
            );
        }
        Self {
            unproven: [self.unproven, self.tests].concat(),
            tests: Vec::new(),
            ..self
        }
    }
}

/// `root` joined with `rel`, but only when the result is genuinely under `root`.
///
/// Resolved component by component rather than by canonicalising: the destination does not exist
/// yet, so there is nothing to canonicalise, and a `..` has to be cancelled against what precedes
/// it rather than against the filesystem. An absolute path, a `..` that escapes, or a root prefix
/// yields `None` — the same rule the file tools apply, for the same reason.
fn contained(root: &std::path::Path, rel: &str) -> Option<std::path::PathBuf> {
    use std::path::Component;
    let mut out = std::path::PathBuf::new();
    for part in std::path::Path::new(rel).components() {
        match part {
            Component::Normal(p) => out.push(p),
            // A `..` may only cancel a component this path itself contributed.
            Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (!out.as_os_str().is_empty()).then(|| root.join(out))
}

/// Authored test files, and the worktree to copy them out of.
struct Seed<'a> {
    from: &'a std::path::Path,
    files: &'a [String],
}

impl Seed<'_> {
    /// Copy the authored files into `into`, creating the directories they need.
    ///
    /// Every path here comes from a model, and this runs on the HOST, in the daemon process,
    /// outside every sandbox, on a value ultimately derived from issue text that `serve` accepts
    /// from GitHub. So each one is contained before it is touched, the way every other path-taking
    /// operation in this workspace is.
    ///
    /// `Path::join` is the trap: joining an absolute path discards the base entirely, so
    /// `from.join("/etc/passwd")` and `into.join("/etc/passwd")` are the same file — and
    /// `fs::copy(p, p)` does not fail, it opens the destination with `truncate(true)`, returns
    /// `Ok(0)`, and leaves the file EMPTY. An unguarded copy therefore destroys whatever the model
    /// names, reports success, and takes the authored tests with it. `..` reaches the same state:
    /// the two worktrees sit at equal depth, so any `../` prefix cancels identically.
    ///
    /// Errors are collected per file rather than returned at the first: one mistyped path must not
    /// discard the seeding of every other.
    fn plant(&self, into: &std::path::Path) -> Result<(), String> {
        let mut refused = Vec::new();
        for file in self.files {
            let (Some(src), Some(dst)) = (contained(self.from, file), contained(into, file)) else {
                refused.push(format!("{file} is not inside the worktree"));
                continue;
            };
            if src == dst {
                refused.push(format!("{file} resolves to one file in both trees"));
                continue;
            }
            if let Some(parent) = dst.parent()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                refused.push(format!("creating {}: {e}", parent.display()));
                continue;
            }
            if let Err(e) = std::fs::copy(&src, &dst) {
                refused.push(format!("copying {file}: {e}"));
            }
        }
        match refused.is_empty() {
            true => Ok(()),
            false => Err(refused.join("; ")),
        }
    }
}

#[cfg(test)]
const AUTHOR_PREAMBLE: &str = include_str!("../prompts/redteam-author.md");

/// The red team's other half: writing the tests the change will be judged against.
///
/// Separate from the classifier because they are different jobs on different sides of the run —
/// one reads a baseline that already happened, the other writes what has to become true. Both are
/// opt-in on having a route.
///
/// Like the classifier, it carries only the stage context. The `redteam_author` stage declares its
/// own `write` ceiling and its own tools, and the executor resolves them per turn — which is what
/// keeps the classifier's read ceiling from taking `Write` and `Edit` away from the author that
/// shares its governance identity.
pub struct TestAuthor {
    /// The generic stage executor context used for test authoring.
    pub(crate) declared_context: std::sync::Arc<crate::workflow::WorkflowContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct TestAuthorInput {
    pub issue: String,
    pub interface: Vec<crate::analyst::InterfaceItem>,
}

impl TestAuthor {
    /// Write tests for `interface` into `worktree`, before any code exists to satisfy them.
    pub async fn write(
        &self,
        worktree: &std::path::Path,
        issue: &str,
        interface: &[crate::analyst::InterfaceItem],
    ) -> Result<AuthoredTests, NodeError> {
        let input = TestAuthorInput {
            issue: issue.to_string(),
            interface: interface.to_vec(),
        };
        let input_json = serde_json::to_string(&input)
            .map_err(|error| NodeError::Failed(format!("test author input: {error}")))?;
        // The author is the one red-team half that mutates: it writes failing tests into the
        // implementer's pre-change worktree. The grant is stated here rather than inferred from the
        // root, so the classifier's read-only half — and any override of either — cannot borrow it.
        let raw = crate::workflow::evaluate_standard_stage_with_resources(
            std::sync::Arc::clone(&self.declared_context),
            "redteam_author",
            input_json,
            crate::workflow::StandardStageResources {
                resource_root: worktree.to_path_buf(),
                capability_ceiling: ratatoskr_core::Capability::Write,
                rag_rat_worktree: Some(worktree.to_path_buf()),
                shell: None,
                publish: None,
                clarifier: None,
                guidance: None,
            },
        )
        .await
        .map_err(|error| NodeError::Failed(format!("test author failed: {error}")))?;
        parse_validated::<AuthoredTests>(&raw)
    }
}

pub struct RedTeamNode {
    pub repo_path: PathBuf,
    /// Where the baseline's own worktree is created — the same root the implementer forks into.
    pub worktree_root: PathBuf,
    /// The branch the baseline's worktree is created on. Under `ratatoskr/`, so a run that dies
    /// before cleanup leaves something `clean` reclaims rather than an orphan nobody owns.
    pub baseline_branch: String,
    pub sandbox: ratatoskr_core::SandboxConfig,
    /// Unique sandbox name for this run.
    pub name: String,
    /// Enabled only when redteam has a route — from `[models.redteam]` or its ruleset.
    pub classifier: Option<RedTeamClassifier>,
    /// What "done" means for this task, from the analyst. The baseline runs exactly what the
    /// post-change run will, or the two sets converge compares are not comparable.
    pub acceptance: Vec<ratatoskr_core::AcceptanceStep>,
    /// Names the checks inside each step. `None` compares at step granularity instead.
    pub characterizer: Option<Characterizer>,
    /// Writes the tests the change is judged against. Opt-in on a `[models.redteam]` route, like
    /// the classifier — without one, the run falls back to whatever tests already exist.
    pub author: Option<TestAuthor>,
}

impl RedTeamNode {
    /// Write the change's tests, then characterise the baseline *with those tests in it*.
    ///
    /// Sequential, and that ordering is the point. A test written before the code is only worth
    /// gating on if it fails without the code — one that already passes proves nothing and would
    /// satisfy [`converge::unsatisfied`](crate::converge::unsatisfied) no matter what the
    /// implementer did. The baseline is the only place that question can be answered, because the
    /// baseline is the tree without the change.
    ///
    /// So the authored files are copied into the baseline worktree before it runs, and an authored
    /// test earns its place in [`AuthoredTests::tests`] only by appearing in that run's failures.
    /// This costs the authoring turn on the critical path — the two used to run concurrently — and
    /// buys the difference between a gate and a formality. It costs no extra suite run: the
    /// baseline had to run anyway, and it now answers two questions instead of one.
    ///
    /// Authored tests landing in `failing_tests` is correct and not a leak: `is_converged` asks
    /// only what *newly* fails, so a test that failed in the baseline cannot be read as damage the
    /// implementer did, while `unsatisfied` requires that same test to pass once the change is in.
    pub async fn run_and_author(
        &self,
        worktree: &std::path::Path,
        issue: &str,
        interface: &[crate::analyst::InterfaceItem],
    ) -> Result<RedTeamOutput, NodeError> {
        let authored = self.author_tests(worktree, issue, interface).await;
        // The clean pass first, and it is the ONLY one converge ever compares against.
        let mut out = self.run_seeded(None).await?;
        if let Some(authored) = authored {
            out.authored = Some(self.proven(authored, worktree, &out).await);
        }
        Ok(out)
    }

    /// Decide which authored tests fail without the change, in a pass of their own.
    ///
    /// A separate run, not the baseline with the files dropped in. Seeding the baseline looks
    /// cheaper and is wrong: a test written before its code routinely does not compile — the
    /// authoring prompt says so, because the symbol does not exist yet — and in most ecosystems one
    /// file that fails to build means the suite runs nothing. The characterizer then reports the
    /// acceptance STEP as the single failing check, which is the same name the implementer's run
    /// reports when it fails to build for the same reason. `newly_introduced_failures` compares the
    /// two, finds no difference, and a change that did nothing converges. That is the very failure
    /// this gate exists to catch, so the numbers converge reads must come from a tree that has
    /// never seen these files.
    ///
    /// The cost is one extra suite run, and only when tests were actually written.
    async fn proven(
        &self,
        authored: AuthoredTests,
        from: &std::path::Path,
        clean: &RedTeamOutput,
    ) -> AuthoredTests {
        if authored.files.is_empty() || authored.tests.is_empty() {
            return authored.none_proven("the red team named no files to prove them with");
        }
        let seed = Seed {
            from,
            files: &authored.files,
        };
        match self.run_seeded(Some(seed)).await {
            Ok(seeded) => authored.proven_by(&seeded, clean),
            Err(e) => authored.none_proven(&format!("the proving run failed: {e}")),
        }
    }

    /// Write the tests, when there is an author and something to write against.
    ///
    /// Best-effort, like the classifier: a failed authoring turn leaves the change to be judged by
    /// the tests that already exist, which is where every run stood before this node could write
    /// any. Failing the run instead would trade a weaker judgement for no judgement.
    async fn author_tests(
        &self,
        worktree: &std::path::Path,
        issue: &str,
        interface: &[crate::analyst::InterfaceItem],
    ) -> Option<AuthoredTests> {
        let author = self.author.as_ref()?;
        if interface.is_empty() {
            return None;
        }
        match author.write(worktree, issue, interface).await {
            Ok(written) => {
                tracing::info!(
                    kind = "authored_tests",
                    files = %written.files.join(", "),
                    "the red team wrote the change's tests"
                );
                Some(written)
            }
            Err(e) => {
                tracing::warn!(
                    "writing the change's tests failed; it will be judged by the tests that \
                     already exist: {e}"
                );
                None
            }
        }
    }

    /// A fresh worktree at the commit the implementer forks from, for the baseline to run in.
    ///
    /// Not the live checkout. A sandbox mount is writable — that is what makes it a mount rather
    /// than part of the read-only host root — and the checkout holds `.git/hooks`, which the host
    /// executes on the next `git worktree add`, alongside `.ratatoskr/rules/`, `ratatoskr.toml`
    /// and `.env`. It is also the wrong tree to measure: the baseline says what already failed
    /// before the change, and it can only mean that if it ran where the change will run. A live
    /// checkout carries installed dependencies and build output a fresh fork does not, so the two
    /// runs disagree about things that have nothing to do with the change.
    ///
    /// Not the implementer's worktree either — that one is where the change happens, and a
    /// baseline measured in it would be measuring the change. The authored tests are *copied* into
    /// this tree instead (see [`RedTeamNode::run_and_author`]), which is deliberate: they have to
    /// fail here to be worth gating on, and filing them as pre-existing failures is exactly right.
    /// `is_converged` ignores that set, so they cannot read as damage; `unsatisfied` requires them
    /// to pass once the change is in.
    async fn baseline_worktree(&self) -> Result<WorktreePath, NodeError> {
        create_worktree(&self.repo_path, &self.worktree_root, &self.baseline_branch)
            .await
            .map_err(|e| NodeError::Failed(format!("baseline worktree create failed: {e}")))
    }

    /// Remove the baseline's worktree and its branch. Best-effort: the measurement is already
    /// taken, and failing the run over a leftover directory would trade a nuisance for a loss.
    /// `clean` reclaims what this misses, which is why the branch is under `ratatoskr/`.
    async fn discard(&self, worktree: WorktreePath) {
        if let Err(e) = remove_worktree(&self.repo_path, &worktree).await {
            tracing::warn!("failed to remove the baseline worktree: {e}");
            return;
        }
        // Only once the worktree is gone: git refuses to delete a branch that is still checked out.
        if let Err(e) = delete_worktree_branch(&self.repo_path, &self.baseline_branch).await {
            tracing::warn!("failed to delete the baseline branch: {e}");
        }
    }

    pub async fn run(&self) -> Result<RedTeamOutput, NodeError> {
        self.run_seeded(None).await
    }

    /// The baseline run, optionally with the authored tests copied in first.
    ///
    /// Seeding is best-effort in the same sense the rest of this node is: a file that cannot be
    /// copied leaves its tests unproven rather than failing the run, because a weaker judgement
    /// beats no judgement. It is never silent — [`AuthoredTests::proven_by`] records every test
    /// that did not earn its place.
    async fn run_seeded(&self, seed: Option<Seed<'_>>) -> Result<RedTeamOutput, NodeError> {
        let worktree = self.baseline_worktree().await?;
        if let Some(seed) = seed
            && let Err(e) = seed.plant(worktree.as_path())
        {
            tracing::warn!(
                "could not put the authored tests in the baseline ({e}); they cannot be proven to \
                 fail without the change, so none of them will gate it"
            );
        }
        let outcomes = run_acceptance(Acceptance {
            cfg: &self.sandbox,
            node: "redteam",
            name: &self.name,
            repo_root: &self.repo_path,
            worktree: worktree.as_path(),
            steps: &self.acceptance,
        })
        .await;
        self.discard(worktree).await;
        let outcomes = outcomes.map_err(NodeError::Failed)?;
        let results = match &self.characterizer {
            Some(c) => c.read(&outcomes).await,
            None => by_exit_code(&outcomes),
        };

        // Classification is best-effort: never let it fail the deterministic characterization.
        let classifications = match (&self.classifier, results.failing.is_empty()) {
            (Some(c), false) => c
                .classify(&results.failing, &results.raw_output)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!("red-team classification failed: {e}");
                    Vec::new()
                }),
            _ => Vec::new(),
        };

        Ok(RedTeamOutput {
            authored: None,
            failing_tests: results.failing,
            passed_tests: results.passed,
            exit_code: results.exit_code,
            classifications,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authored(tests: &[&str], files: &[&str]) -> AuthoredTests {
        AuthoredTests {
            files: files.iter().map(|s| s.to_string()).collect(),
            tests: tests.iter().map(|s| s.to_string()).collect(),
            covers: String::new(),
            unproven: Vec::new(),
        }
    }

    fn ran(failing: &[&str], passed: usize) -> RedTeamOutput {
        RedTeamOutput {
            authored: None,
            failing_tests: failing.iter().map(|s| s.to_string()).collect(),
            passed_tests: passed,
            exit_code: if failing.is_empty() { 0 } else { 1 },
            classifications: Vec::new(),
        }
    }

    /// A test that already passes without the change proves nothing, and must not gate.
    ///
    /// This is the hole the reproduction gate had: `unsatisfied` asks which authored tests are
    /// still failing afterwards, so a test that never failed is satisfied for free and waves
    /// through a change that did nothing.
    #[test]
    fn an_authored_test_that_passes_without_the_change_gates_nothing() {
        let clean = ran(&[], 10);
        let seeded = ran(&["repro::the_bug_is_fixed"], 9);
        let kept = authored(
            &["repro::the_bug_is_fixed", "repro::two_plus_two_is_four"],
            &["tests/repro.rs"],
        )
        .proven_by(&seeded, &clean);

        assert_eq!(kept.tests, ["repro::the_bug_is_fixed"]);
        assert_eq!(kept.unproven, ["repro::two_plus_two_is_four"]);
        assert_eq!(
            crate::converge::unsatisfied(&kept.tests, &["repro::the_bug_is_fixed".to_string()]),
            ["repro::the_bug_is_fixed"]
        );
        // The rejected one cannot be satisfied for free, because it is no longer asked about.
        assert!(crate::converge::unsatisfied(&kept.tests, &[]).is_empty());
    }

    /// A test that stops the suite building has demonstrably not passed without the change.
    ///
    /// The modal case: the symbol does not exist yet, so the file does not compile, so the runner
    /// reports the acceptance STEP rather than any test name. Reading that as "unproven" would
    /// throw the gate away exactly when the authored test is doing its job.
    #[test]
    fn a_test_that_breaks_the_build_without_the_change_is_proven_by_that() {
        let clean = ran(&[], 10);
        let seeded = ran(&["cargo test"], 0);
        let kept =
            authored(&["repro::the_bug_is_fixed"], &["tests/repro.rs"]).proven_by(&seeded, &clean);
        assert_eq!(kept.tests, ["repro::the_bug_is_fixed"], "still gates");
        assert!(kept.unproven.is_empty());
    }

    /// The clean run is what converge compares against, and it never sees the authored files.
    ///
    /// Seeding the baseline instead would make a no-op change converge: both runs fail to build,
    /// both report the same acceptance step, and `newly_introduced_failures` finds no difference.
    #[test]
    fn a_change_that_does_nothing_does_not_converge_when_the_tests_break_the_build() {
        let clean = ran(&[], 10);
        // What the implementer's run reports when it changed nothing and the authored file is there.
        let after_no_op = vec!["cargo test".to_string()];
        assert!(
            !crate::converge::is_converged(
                &clean.failing_tests,
                Some(&crate::testrun::AcceptanceResult {
                    failing_tests: after_no_op.clone(),
                    passed_tests: 0,
                    exit_code: 101,
                })
            ),
            "a build the change did not fix is a NEW failure against a clean baseline"
        );
        // Had the baseline been seeded, it would report the same step and the difference vanishes.
        let seeded_baseline = ran(&["cargo test"], 0);
        assert!(
            crate::converge::is_converged(
                &seeded_baseline.failing_tests,
                Some(&crate::testrun::AcceptanceResult {
                    failing_tests: after_no_op.clone(),
                    passed_tests: 0,
                    exit_code: 101,
                })
            ),
            "which is precisely why the baseline must not be seeded"
        );
    }

    /// Nothing proven means the ordinary gate, said out loud rather than silently.
    #[test]
    fn tests_that_all_pass_on_the_baseline_leave_nothing_to_gate_on() {
        let kept = authored(&["a", "b"], &["tests/repro.rs"]).proven_by(&ran(&[], 5), &ran(&[], 5));
        assert!(kept.tests.is_empty(), "none of them earned a place");
        assert_eq!(kept.unproven, ["a", "b"], "and the run can say why");
    }

    /// Named tests with no file to prove them by gate nothing, and say so.
    #[test]
    fn tests_with_no_files_cannot_be_proven() {
        let kept = authored(&["a"], &[]).none_proven("no files");
        assert!(kept.tests.is_empty());
        assert_eq!(kept.unproven, ["a"]);
    }

    /// A model-supplied path must not reach `fs::copy` uncontained.
    ///
    /// `Path::join` discards the base for an absolute path, so `from.join(p)` and `into.join(p)`
    /// become the SAME file — and `fs::copy(p, p)` does not fail, it truncates to zero and returns
    /// `Ok`. Unguarded, one absolute path in the model's output deletes the file it names.
    #[test]
    fn a_path_that_escapes_the_worktree_is_refused_rather_than_copied() {
        let root = std::path::Path::new("/srv/wt/impl");
        assert_eq!(
            contained(root, "tests/repro.rs").unwrap(),
            root.join("tests/repro.rs")
        );
        assert_eq!(contained(root, "./a/../b.rs").unwrap(), root.join("b.rs"));
        for escape in [
            "/etc/passwd",
            "../../../.env",
            "../sibling/x.rs",
            "..",
            "",
            "./",
        ] {
            assert!(
                contained(root, escape).is_none(),
                "`{escape}` must be refused"
            );
        }
    }

    /// And the refusal is reported per file, so one bad path does not discard the rest.
    #[test]
    fn one_refused_path_does_not_discard_the_other_files() {
        let tmp = std::env::temp_dir().join(format!("ratatoskr-refuse-{}", std::process::id()));
        let (from, into) = (tmp.join("impl"), tmp.join("base"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(from.join("tests")).unwrap();
        std::fs::create_dir_all(&into).unwrap();
        std::fs::write(from.join("tests/good.rs"), "ok").unwrap();

        let err = Seed {
            from: &from,
            files: &["/etc/passwd".to_string(), "tests/good.rs".to_string()],
        }
        .plant(&into)
        .expect_err("the absolute path is refused");

        assert!(err.contains("/etc/passwd"), "the refusal names it: {err}");
        assert_eq!(
            std::fs::read_to_string(into.join("tests/good.rs")).unwrap(),
            "ok",
            "the good file is still planted"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Seeding is what puts the authored tests where the proving run can judge them.
    #[test]
    fn planting_copies_the_authored_files_into_the_baseline_tree() {
        let tmp = std::env::temp_dir().join(format!("ratatoskr-seed-{}", std::process::id()));
        let (from, into) = (tmp.join("impl"), tmp.join("base"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(from.join("tests/deep")).unwrap();
        std::fs::create_dir_all(&into).unwrap();
        std::fs::write(from.join("tests/deep/repro.rs"), "fn t() {}").unwrap();

        Seed {
            from: &from,
            files: &["tests/deep/repro.rs".to_string()],
        }
        .plant(&into)
        .expect("the file is copied, directories and all");

        assert_eq!(
            std::fs::read_to_string(into.join("tests/deep/repro.rs")).unwrap(),
            "fn t() {}"
        );
        // A file that is not there is reported rather than silently skipped: unreported, its tests
        // would look proven-absent and quietly stop gating.
        assert!(
            Seed {
                from: &from,
                files: &["tests/missing.rs".to_string()],
            }
            .plant(&into)
            .is_err()
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A repository with one commit, plus an untracked `node_modules/installed` — what a live
    /// checkout carries and a fresh fork does not.
    async fn checkout_with_dependencies_installed(tmp: &std::path::Path) -> PathBuf {
        let repo = tmp.join("repo");
        std::fs::create_dir_all(repo.join("node_modules")).unwrap();
        std::fs::write(repo.join("src.rs"), "fn main() {}\n").unwrap();
        std::fs::write(repo.join("node_modules/installed"), "").unwrap();
        std::fs::write(repo.join(".gitignore"), "node_modules/\n").unwrap();
        for args in [
            vec!["init", "-q", "-b", "main"],
            vec!["config", "user.email", "t@example.com"],
            vec!["config", "user.name", "T"],
            vec!["add", "."],
            vec!["commit", "-qm", "initial"],
        ] {
            let out = tokio::process::Command::new("git")
                .arg("-C")
                .arg(&repo)
                .args(&args)
                .output()
                .await
                .unwrap();
            assert!(out.status.success(), "git {args:?}: {out:?}");
        }
        repo
    }

    fn baseline_node(repo: &std::path::Path, tmp: &std::path::Path, steps: &[&str]) -> RedTeamNode {
        RedTeamNode {
            repo_path: repo.to_path_buf(),
            worktree_root: tmp.join("worktrees"),
            baseline_branch: "ratatoskr/test-baseline".into(),
            sandbox: ratatoskr_core::SandboxConfig {
                backend: "landlock".into(),
                ..Default::default()
            },
            name: "ratatoskr-redteam-test".into(),
            classifier: None,
            acceptance: match steps.is_empty() {
                true => Vec::new(),
                false => vec![ratatoskr_core::AcceptanceStep {
                    name: "deps".into(),
                    command: steps.iter().map(|s| (*s).to_string()).collect(),
                }],
            },
            characterizer: None,
            author: None,
        }
    }

    #[tokio::test]
    async fn the_baseline_gets_a_fresh_tree_and_never_the_live_checkout() {
        // Two failures with one cause. The live checkout is bind-mounted *writable* by the
        // sandbox, and it holds `.git/hooks` — which the host runs on the next `git worktree add`
        // — along with `.ratatoskr/rules/`, `ratatoskr.toml` and `.env`. It is also the wrong tree
        // to measure in: it carries installed dependencies and build output that the tree the
        // change is written in does not, so the same command gives opposite answers for reasons
        // that have nothing to do with the change.
        let tmp = std::env::temp_dir().join(format!("ratatoskr-rt-{}", std::process::id()));
        let repo = checkout_with_dependencies_installed(&tmp).await;
        let node = baseline_node(&repo, &tmp, &[]);

        let wt = node.baseline_worktree().await.unwrap();
        assert_ne!(wt.as_path(), repo, "the baseline is not given the checkout");
        assert!(wt.as_path().join("src.rs").exists(), "same commit");
        assert!(
            !wt.as_path().join("node_modules/installed").exists(),
            "a fresh fork does not carry what was installed into the checkout"
        );

        // And it takes its own worktree and branch away with it, so a run does not leave one
        // behind per baseline.
        node.discard(wt.clone()).await;
        assert!(!wt.as_path().exists());
        let branches = ratatoskr_exec::managed_worktree_branches(&repo)
            .await
            .unwrap();
        assert!(!branches.contains(&node.baseline_branch), "{branches:?}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[tokio::test]
    #[ignore = "requires bwrap on the host; run with --ignored"]
    async fn a_dependency_only_the_checkout_has_does_not_make_the_baseline_pass() {
        // The observed failure, end to end: the baseline reported every step green because it ran
        // where the dependencies were already installed, while the implementer's identical
        // commands failed in a fresh worktree. Converge then compares two runs that differ by more
        // than the change, which is the one thing it is not allowed to do.
        let tmp = std::env::temp_dir().join(format!("ratatoskr-rt-e2e-{}", std::process::id()));
        let repo = checkout_with_dependencies_installed(&tmp).await;
        let node = baseline_node(&repo, &tmp, &["test", "-e", "node_modules/installed"]);

        let out = node.run().await.unwrap();
        assert_ne!(
            out.exit_code, 0,
            "the baseline saw a dependency that only the live checkout has"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn the_author_preamble_says_what_makes_these_tests_different() {
        let p = AUTHOR_PREAMBLE.to_ascii_lowercase();
        // Written before the code, so failing now is expected and must not be papered over.
        assert!(
            p.contains("fail now"),
            "what a good test does at this point"
        );
        assert!(p.contains("asserting nothing"), "and what it must not do");
        // It adds; it does not adjust the judge.
        assert!(p.contains("do not modify tests that are already there"));
    }

    #[test]
    fn parses_a_classifier_response() {
        let raw = r#"{"classifications":[
            {"test":"a::flaps","category":"flaky","reason":"depends on wall-clock timing"},
            {"test":"b::broken","category":"real"}
        ]}"#;
        let c = parse_validated::<Classification>(raw).unwrap();
        assert_eq!(c.classifications.len(), 2);
        assert_eq!(c.classifications[0].category, "flaky");
        // `reason` defaults when omitted.
        assert_eq!(c.classifications[1].reason, "");
    }
}
