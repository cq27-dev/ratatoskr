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
}
