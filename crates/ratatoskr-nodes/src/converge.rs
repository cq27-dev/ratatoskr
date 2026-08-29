//! Converge: decide whether the implementer's change is done. Pure data comparison — no LLM.
//!
//! The signal is *newly-introduced* failures: tests failing after the change that were not failing
//! in the baseline (red-team). A pre-existing failure is not the implementer's problem; a new one
//! is. Empty new-failure set → converged.

use crate::testrun::AcceptanceResult;

/// Tests failing after the change that were not failing in the baseline.
pub fn newly_introduced_failures(
    baseline_failing: &[String],
    post_failing: &[String],
) -> Vec<String> {
    post_failing
        .iter()
        .filter(|t| !baseline_failing.contains(t))
        .cloned()
        .collect()
}

/// Authored tests that are still failing after the change.
///
/// These need their own gate. They were written before the code, so they fail in the baseline as a
/// matter of course — and [`is_converged`] asks only whether anything *newly* fails, which would
/// wave through a change that satisfied none of them. A test written to specify the change is the
/// one test the change is not allowed to leave failing.
pub fn unsatisfied(authored: &[String], post_failing: &[String]) -> Vec<String> {
    authored
        .iter()
        .filter(|t| post_failing.contains(t))
        .cloned()
        .collect()
}

/// Whether the change converged: it introduced no new failures.
///
/// Takes the post-change result as an `Option` for the same reason [`test_command_ran`] does: an
/// implementer that wrote nothing ran no suite, and an empty failing set introduces no new
/// failures by construction. A change that does not exist has not converged; it has not happened,
/// and no comparison of failing sets can say otherwise. The baseline stays flat — the red team
/// always runs its suite, so there is no absence to represent on that side.
pub fn is_converged(baseline_failing: &[String], post: Option<&AcceptanceResult>) -> bool {
    post.is_some_and(|post| {
        newly_introduced_failures(baseline_failing, &post.failing_tests).is_empty()
    })
}

/// Whether a test run actually produced results. Zero tests parsed *and* a non-zero exit means the
/// command didn't run to completion (a broken build, a missing runner, a sandbox mis-mount) — NOT
/// "no failures". Without this guard, both branches report zero tests and converge falsely reports
/// success on empty data (the failure mode the first live run hit).
///
/// Takes the whole result rather than its parts, so absence has to be dealt with: an implementer
/// that wrote nothing has no acceptance run at all, and `None` is the only honest way to say so.
/// Zeros could not — `passed: 0, exit_code: 0` is also a clean run of an empty suite.
pub fn test_command_ran(acceptance: Option<&AcceptanceResult>) -> bool {
    acceptance
        .is_some_and(|a| !a.failing_tests.is_empty() || a.passed_tests > 0 || a.exit_code == 0)
}

/// Files whose diff removed or replaced existing lines and are not exempted by the task's
/// up-front `defineDefaults({ mayModifyTests: [...] })` declaration. The model referee judges
/// these candidates; this deterministic half intentionally has no language-specific path rules.
pub fn referee_candidates(rewritten: &[String], exempt: &[String]) -> Vec<String> {
    rewritten
        .iter()
        .filter(|path| !exempt.iter().any(|prefix| under(path, prefix)))
        .cloned()
        .collect()
}

/// Whether `path` is at or under `prefix`, by segment — so exempting `tests` does not also exempt
/// `tests_helper.rs`.
fn under(path: &str, prefix: &str) -> bool {
    let prefix = prefix.trim_end_matches('/');
    path == prefix || path.starts_with(&format!("{prefix}/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    /// An acceptance run that reported `failing`. The counts are only ever read through
    /// [`test_command_ran`], so they are the shape of a run that got as far as reporting.
    fn shape(failing: &[&str], passed: usize, exit_code: i32) -> AcceptanceResult {
        AcceptanceResult {
            failing_tests: v(failing),
            passed_tests: passed,
            exit_code,
        }
    }

    fn result(failing: &[String]) -> AcceptanceResult {
        AcceptanceResult {
            failing_tests: failing.to_vec(),
            passed_tests: 1,
            exit_code: 1,
        }
    }

    fn ran(failing: &[&str]) -> AcceptanceResult {
        AcceptanceResult {
            failing_tests: v(failing),
            passed_tests: 1,
            exit_code: if failing.is_empty() { 0 } else { 1 },
        }
    }

    /// The other half of the skipped-suite pair. A repository-authored workflow may call
    /// `isConverged` without calling `testCommandRan` beside it, so this cannot rely on the other
    /// guard having been asked first.
    #[test]
    fn a_change_that_was_never_made_has_not_converged() {
        assert!(
            !is_converged(&v(&["a"]), None),
            "an empty failing set from a suite that never ran is not a clean one"
        );
        assert!(is_converged(&v(&["a"]), Some(&ran(&[]))));
    }

    #[test]
    fn converged_when_no_new_failures() {
        // Pre-existing failure stays; nothing new introduced.
        assert!(is_converged(
            &v(&["a::pre_existing"]),
            Some(&ran(&["a::pre_existing"]))
        ));
        assert!(is_converged(&v(&["a::pre_existing"]), Some(&ran(&[]))));
        assert!(is_converged(&v(&[]), Some(&ran(&[]))));
    }

    #[test]
    fn not_converged_when_change_introduces_a_failure() {
        let new = newly_introduced_failures(
            &v(&["a::pre_existing"]),
            &v(&["a::pre_existing", "b::broke"]),
        );
        assert_eq!(new, ["b::broke"]);
        assert!(!is_converged(
            &v(&["a::pre_existing"]),
            Some(&ran(&["a::pre_existing", "b::broke"]))
        ));
    }

    #[test]
    fn a_test_written_for_the_change_must_pass_even_though_it_failed_at_baseline() {
        // The hole this closes: tests written before the code fail in the baseline as a matter of
        // course, so `is_converged` — which asks only what is *newly* failing — would wave through
        // a change that satisfied none of them.
        let authored = v(&[
            "store::prunes_old_rows",
            "store::zero_duration_removes_nothing",
        ]);
        let baseline_failing = authored.clone();
        let still_failing = v(&["store::zero_duration_removes_nothing"]);

        assert!(
            is_converged(&baseline_failing, Some(&result(&still_failing))),
            "nothing is newly failing, which is exactly why this is not enough on its own"
        );
        assert_eq!(
            unsatisfied(&authored, &still_failing),
            ["store::zero_duration_removes_nothing"],
            "and the sad-path test nobody implemented is named"
        );

        // Satisfied: the change made them pass.
        assert!(unsatisfied(&authored, &[]).is_empty());
        // A run with no authored tests is unaffected.
        assert!(unsatisfied(&[], &still_failing).is_empty());
    }

    #[test]
    fn the_declared_exemption_covers_its_subtree_and_nothing_else() {
        let rewritten = v(&["crates/foo/tests/api.rs", "crates/bar/tests/api.rs"]);
        assert_eq!(
            referee_candidates(&rewritten, &v(&["crates/foo/tests"])),
            ["crates/bar/tests/api.rs"]
        );
        // A prefix that isn't a path boundary doesn't exempt: `tests` ≠ `tests_helper`.
        assert_eq!(
            referee_candidates(&v(&["tests_helper/conftest.py"]), &v(&["tests"])),
            ["tests_helper/conftest.py"]
        );
        assert!(referee_candidates(&rewritten, &v(&["crates"])).is_empty());
    }

    #[test]
    fn zero_tests_with_nonzero_exit_did_not_run() {
        // The false-convergence case: no tests parsed and the command failed.
        assert!(!test_command_ran(Some(&shape(&[], 0, 101))));
        // A genuinely empty suite that exited 0 counts as "ran" (nothing to break).
        assert!(test_command_ran(Some(&shape(&[], 0, 0))));
        // Any parsed test means it ran, regardless of exit code.
        assert!(test_command_ran(Some(&shape(&["a"], 0, 101))));
        // A passing count alone proves it ran, which is the whole reason the count is carried.
        assert!(test_command_ran(Some(&shape(&[], 285, 101))));
    }

    /// An implementer that wrote nothing has no acceptance run to report — and the absence is the
    /// only honest way to say so, since a zeroed result is also what a clean run of an empty suite
    /// reports. That shape is what the workflow reads before deciding to review rather than
    /// iterate, so it must not be constructible for a run that never started one.
    #[test]
    fn a_suite_that_never_ran_is_absent_not_zeroed() {
        assert!(
            !test_command_ran(None),
            "there is no result, which is not the same as a clean one"
        );
        assert!(
            test_command_ran(Some(&shape(&[], 0, 0))),
            "the zeros a real clean run of an empty suite reports still read as a run"
        );
        // And there is no way to say it with numbers: `None` is the whole vocabulary, which is
        // the point — a skipped run has no shape a reader could mistake for a real one.
    }

    #[test]
    fn every_rewritten_file_is_a_candidate_whatever_the_language() {
        // The gate no longer holds opinions about where a language keeps its tests: any file with
        // removed or replaced lines is shown to the judgement — including an ordinary source
        // file. Rust unit tests live inside the file they test, which is exactly the blindness
        // #205 exploited: the deleted `#[cfg(test)] mod` matched no test dir, runner file or
        // infix, so the path list waved it through.
        assert_eq!(
            referee_candidates(&v(&["src/lib.rs", "tests/a.rs"]), &[]),
            ["src/lib.rs", "tests/a.rs"],
            "both come back, in the order the diff reported them"
        );
        assert_eq!(
            referee_candidates(&v(&["crates/ratatoskr-nodes/src/lib.rs"]), &[]),
            ["crates/ratatoskr-nodes/src/lib.rs"]
        );
    }

    #[test]
    fn nothing_rewritten_means_nothing_to_judge() {
        // No candidates, no judgement: additions alone never bring you here.
        assert!(referee_candidates(&[], &[]).is_empty());
        assert!(referee_candidates(&[], &v(&["tests"])).is_empty());
    }

    #[test]
    fn the_candidate_exemption_is_segment_boundaried() {
        // mayModifyTests is matched by path segment: exempting `tests` must not also exempt
        // `tests_helper`.
        assert_eq!(
            referee_candidates(&v(&["tests_helper/conftest.py"]), &v(&["tests"])),
            ["tests_helper/conftest.py"]
        );
        // It covers its own subtree, and only its own.
        assert_eq!(
            referee_candidates(
                &v(&["crates/foo/tests/api.rs", "crates/bar/tests/api.rs"]),
                &v(&["crates/foo/tests"])
            ),
            ["crates/bar/tests/api.rs"]
        );
        let under_crates = referee_candidates(
            &v(&["crates/foo/tests/api.rs", "crates/bar/tests/api.rs"]),
            &v(&["crates"]),
        );
        assert!(
            under_crates.is_empty(),
            "exempting `crates` exempts everything under it"
        );
    }
}
