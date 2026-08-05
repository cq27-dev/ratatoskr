//! Converge: decide whether the implementer's change is done. Pure data comparison — no LLM.
//!
//! The signal is *newly-introduced* failures: tests failing after the change that were not failing
//! in the baseline (red-team). A pre-existing failure is not the implementer's problem; a new one
//! is. Empty new-failure set → converged.

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

/// Whether the change converged: it introduced no new failures.
pub fn is_converged(baseline_failing: &[String], post_failing: &[String]) -> bool {
    newly_introduced_failures(baseline_failing, post_failing).is_empty()
}

/// Whether a test run actually produced results. Zero tests parsed *and* a non-zero exit means the
/// command didn't run to completion (a broken build, a missing runner, a sandbox mis-mount) — NOT
/// "no failures". Without this guard, both branches report zero tests and converge falsely reports
/// success on empty data (the failure mode the first live run hit).
pub fn test_command_ran(failing: &[String], passing: &[String], exit_code: i32) -> bool {
    !failing.is_empty() || !passing.is_empty() || exit_code == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn converged_when_no_new_failures() {
        // Pre-existing failure stays; nothing new introduced.
        assert!(is_converged(
            &v(&["a::pre_existing"]),
            &v(&["a::pre_existing"])
        ));
        assert!(is_converged(&v(&["a::pre_existing"]), &v(&[])));
        assert!(is_converged(&v(&[]), &v(&[])));
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
            &v(&["a::pre_existing", "b::broke"])
        ));
    }

    #[test]
    fn zero_tests_with_nonzero_exit_did_not_run() {
        // The false-convergence case: no tests parsed and the command failed.
        assert!(!test_command_ran(&v(&[]), &v(&[]), 101));
        // A genuinely empty suite that exited 0 counts as "ran" (nothing to break).
        assert!(test_command_ran(&v(&[]), &v(&[]), 0));
        // Any parsed test means it ran, regardless of exit code.
        assert!(test_command_ran(&v(&["a"]), &v(&[]), 101));
    }
}
