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
pub fn is_converged(baseline_failing: &[String], post_failing: &[String]) -> bool {
    newly_introduced_failures(baseline_failing, post_failing).is_empty()
}

/// Whether a test run actually produced results. Zero tests parsed *and* a non-zero exit means the
/// command didn't run to completion (a broken build, a missing runner, a sandbox mis-mount) — NOT
/// "no failures". Without this guard, both branches report zero tests and converge falsely reports
/// success on empty data (the failure mode the first live run hit).
pub fn test_command_ran(failing: &[String], passed: usize, exit_code: i32) -> bool {
    !failing.is_empty() || passed > 0 || exit_code == 0
}

/// Runner configuration, by exact filename. `Cargo.toml` and `package.json` are referee files only
/// in part (`[dev-dependencies]`, `[[test]]`, `scripts`), but the touched-file list is path-granular
/// — any touch counts, so a run that legitimately bumps a dependency needs the exemption too.
const RUNNER_FILES: &[&str] = &[
    // pytest imports conftest.py without being asked: the BenchJack exploit is a new one that
    // rewrites every test's reported outcome.
    "conftest.py",
    "pytest.ini",
    "tox.ini",
    "noxfile.py",
    "setup.cfg",
    "pyproject.toml",
    "Cargo.toml",
    "package.json",
    "Makefile",
    "justfile",
];

/// Runner config whose extension varies (`jest.config.ts`, `.mocharc.yml`, ...).
const RUNNER_PREFIXES: &[&str] = &[
    "jest.config.",
    "vitest.config.",
    "karma.conf.",
    "playwright.config.",
    ".mocharc.",
];

/// Filename fragments that name a test across the ecosystems we run: `foo_test.go`,
/// `foo.test.ts`, `foo.spec.js`, `test_foo.py`.
const TEST_INFIXES: &[&str] = &[".test.", ".spec.", "_test.", "_spec."];

/// Directory names whose contents are tests.
const TEST_DIRS: &[&str] = &["test", "tests", "spec", "specs", "__tests__"];

/// The files the implementer touched that are part of the *referee* — the tests themselves and the
/// machinery that decides what they report. Empty means the iteration left the referee alone.
///
/// This must be checked before the test comparison is believed: once the referee moves, the
/// passing/failing sets describe a bar the change wrote for itself. "No new failing tests" is
/// cheapest to satisfy by making the tests stop failing some other way.
///
/// Two known ceilings, both structural:
/// - *Auto-loaded* files can't be enumerated. `conftest.py` is the known one; every runner has its
///   own auto-load surface (plugin entry points, `sitecustomize.py`, a shell rc the sandbox reads).
///   This narrows the class, it does not close it.
/// - Detection is path-granular, so a section-level distinction (`[dependencies]` vs
///   `[dev-dependencies]`) can't be made. Manifests count as referee touches outright.
///
/// `exempt` is the task's up-front declaration (`defineDefaults({ mayModifyTests: [...] })`),
/// matched as a path prefix at segment boundaries.
pub fn referee_touches(touched: &[String], exempt: &[String]) -> Vec<String> {
    touched
        .iter()
        .filter(|p| is_referee(p) && !exempt.iter().any(|e| under(p, e)))
        .cloned()
        .collect()
}

/// What to send back when the iteration rewrote the referee. Names the offending files *and* the
/// exemption, so a task that legitimately rewrites tests but never declared it learns how to
/// instead of churning silently to the iteration wall.
///
/// It also says what is *not* refused. The gate reads rewritten files, not touched ones, so a new
/// test is always allowed — and an implementer that believes otherwise writes untested code and
/// contorts its design to avoid a fixture it was never forbidden to extend.
pub fn referee_correction(referee: &[String]) -> String {
    format!(
        "You rewrote files that decide whether this task is done: {}. Revert those edits and make \
         the change satisfy the tests as they stand — rewriting a test, its runner config, or \
         anything the runner auto-loads is not a way to pass. Note what this is not: *adding* a \
         test is allowed and expected, and does not bring you here. Only removing or replacing \
         lines that were already there does. If this task really is supposed to change existing \
         tests, that has to be declared up front, before the work, in .ratatoskr/rules/*.ts with \
         `defineDefaults({{ mayModifyTests: [\"<path>\"] }})`.",
        referee.join(", ")
    )
}

/// Whether `path` is at or under `prefix`, by segment — so exempting `tests` does not also exempt
/// `tests_helper.rs`.
fn under(path: &str, prefix: &str) -> bool {
    let prefix = prefix.trim_end_matches('/');
    path == prefix || path.starts_with(&format!("{prefix}/"))
}

fn is_referee(path: &str) -> bool {
    let mut segments = path.split('/').collect::<Vec<_>>();
    let file = segments.pop().unwrap_or(path);
    segments.iter().any(|d| TEST_DIRS.contains(d))
        || RUNNER_FILES.contains(&file)
        || RUNNER_PREFIXES.iter().any(|p| file.starts_with(p))
        || file.starts_with("test_")
        || TEST_INFIXES.iter().any(|i| file.contains(i))
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
    fn a_touched_referee_is_named() {
        let touched = v(&[
            "src/lib.rs",
            "crates/foo/tests/api.rs",
            "conftest.py",
            "pytest.ini",
            "Cargo.toml",
            "package.json",
            "jest.config.ts",
            "app/foo.test.ts",
            "app/test_foo.py",
        ]);
        let mut found = referee_touches(&touched, &[]);
        found.sort();
        assert_eq!(
            found,
            [
                "Cargo.toml",
                "app/foo.test.ts",
                "app/test_foo.py",
                "conftest.py",
                "crates/foo/tests/api.rs",
                "jest.config.ts",
                "package.json",
                "pytest.ini",
            ]
        );
        // Ordinary source is not the referee.
        assert!(referee_touches(&v(&["src/lib.rs"]), &[]).is_empty());
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
            is_converged(&baseline_failing, &still_failing),
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
        let touched = v(&["crates/foo/tests/api.rs", "crates/bar/tests/api.rs"]);
        assert_eq!(
            referee_touches(&touched, &v(&["crates/foo/tests"])),
            ["crates/bar/tests/api.rs"]
        );
        // A prefix that isn't a path boundary doesn't exempt: `tests` ≠ `tests_helper`.
        assert_eq!(
            referee_touches(&v(&["tests_helper/conftest.py"]), &v(&["tests"])),
            ["tests_helper/conftest.py"]
        );
        assert!(referee_touches(&touched, &v(&["crates"])).is_empty());
    }

    #[test]
    fn zero_tests_with_nonzero_exit_did_not_run() {
        // The false-convergence case: no tests parsed and the command failed.
        assert!(!test_command_ran(&v(&[]), 0, 101));
        // A genuinely empty suite that exited 0 counts as "ran" (nothing to break).
        assert!(test_command_ran(&v(&[]), 0, 0));
        // Any parsed test means it ran, regardless of exit code.
        assert!(test_command_ran(&v(&["a"]), 0, 101));
        // A passing count alone proves it ran, which is the whole reason the count is carried.
        assert!(test_command_ran(&v(&[]), 285, 101));
    }
}
