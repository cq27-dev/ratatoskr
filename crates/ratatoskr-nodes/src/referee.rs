use std::fmt::Write as _;

use ratatoskr_core::ModelRoute;
use ratatoskr_graph::{NodeError, parse_validated};
use ratatoskr_mcp::ToolSet;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};

/// The only tools the referee receives: enough to confirm a removed check still exists after a
/// relocation, without any way to change the worktree or run commands.
pub const REFEREE_TOOLS: &[&str] = &[
    ratatoskr_agent::files::READ,
    ratatoskr_agent::files::GREP,
    ratatoskr_agent::files::GLOB,
];

const PREAMBLE: &str = include_str!("../prompts/referee.md");

fn non_empty<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if value.trim().is_empty() {
        return Err(serde::de::Error::custom("must not be empty"));
    }
    Ok(value)
}

/// A change that weakens the check deciding whether the task is done.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Violation {
    #[schemars(length(min = 1))]
    #[serde(deserialize_with = "non_empty")]
    pub file: String,
    #[schemars(length(min = 1))]
    #[serde(deserialize_with = "non_empty")]
    pub reason: String,
}

/// The referee's judgement. An empty list means the diff did not weaken the bar, including when a
/// removed check was confirmed to have moved elsewhere in the worktree.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RefereeOutput {
    #[serde(default)]
    pub violations: Vec<Violation>,
}

/// What the referee reads. `hunks` contains only files where existing lines were removed or
/// replaced; additions alone never need a judgement.
pub struct RefereeInput {
    pub issue: String,
    pub requirements: Vec<String>,
    pub hunks: String,
    pub candidates: Vec<String>,
}

/// Extract the complete unified-diff sections for exactly `paths`.
pub fn hunks_for(diff: &str, paths: &[String]) -> String {
    if diff.is_empty() || paths.is_empty() {
        return String::new();
    }

    let mut starts = Vec::new();
    if diff.starts_with("diff --git ") {
        starts.push(0);
    }
    starts.extend(diff.match_indices("\ndiff --git ").map(|(i, _)| i + 1));

    starts
        .iter()
        .enumerate()
        .filter_map(|(i, start)| {
            let end = starts.get(i + 1).copied().unwrap_or(diff.len());
            let section = &diff[*start..end];
            let header = section.lines().next()?;
            let (old, new) = header.strip_prefix("diff --git a/")?.split_once(" b/")?;
            paths
                .iter()
                .any(|path| path == old || path == new)
                .then_some(section)
        })
        .collect()
}

/// Turn violations into the correction the implementer receives.
pub fn correction(violations: &[Violation]) -> String {
    let mut text = String::from(
        "Your change weakened files that decide whether this task is done. Revert or repair each \
         of these changes; rewriting checks, their runner configuration, or anything the runner \
         auto-loads is not a way to pass:\n\n",
    );
    for violation in violations {
        let _ = writeln!(text, "- {}: {}", violation.file, violation.reason);
    }
    text.push_str(
        "\nAdding a test is allowed and never brings you here. If this task really must change \
         existing tests, declare `mayModifyTests` in `defineDefaults` up front in \
         .ratatoskr/rules/*.ts.",
    );
    text
}

/// A read-only model judgement over the hunks that removed or replaced existing lines.
pub struct RefereeNode {
    pub route: ModelRoute,
    pub tools: ToolSet,
    pub policy: Option<std::sync::Arc<dyn ratatoskr_core::ToolPolicy>>,
    pub max_turns: Option<usize>,
    pub system_prompt: Option<String>,
    pub plugins: crate::NodePlugins,
    pub ledger: Option<std::sync::Arc<ratatoskr_agent::RunLedger>>,
    pub files: Option<std::path::PathBuf>,
}

impl RefereeNode {
    pub async fn run(&self, input: RefereeInput) -> Result<RefereeOutput, NodeError> {
        if input.candidates.is_empty() {
            return Ok(RefereeOutput {
                violations: Vec::new(),
            });
        }

        let raw = ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "referee",
            route: &self.route,
            preamble: &crate::effective_preamble(
                "referee",
                PREAMBLE,
                self.system_prompt.as_deref(),
                self.plugins.context.as_deref(),
                &self.plugins.skills,
            ),
            question: &render_prompt(&input),
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(RefereeOutput),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            clarifier: None,
            observer: self.plugins.observer.clone(),
            skills: crate::skills::loaded(&self.plugins.skills, "referee"),
            files: self.files.clone(),
            shell: None,
            push: None,
            conversation: None,
            ledger: self.ledger.clone(),
            produces: Some("files whose diff hunks weakened task-completion checks, each with a reason, or none"),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("referee agent failed: {e}")))?;

        parse_validated::<RefereeOutput>(&raw)
    }
}

fn render_prompt(input: &RefereeInput) -> String {
    let mut text = String::new();
    let _ = write!(text, "TASK:\n{}\n\n", input.issue);
    if !input.requirements.is_empty() {
        text.push_str("REQUIREMENTS THE CHANGE MUST SATISFY:\n");
        for requirement in &input.requirements {
            let _ = writeln!(text, "- {requirement}");
        }
        text.push('\n');
    }
    let _ = write!(
        text,
        "FILES WITH REMOVED OR REPLACED LINES: {}\n\nDIFF HUNKS TO JUDGE:\n{}\n",
        input.candidates.join(", "),
        input.hunks
    );
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    use ratatoskr_graph::parse_validated;

    fn paths(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    /// A three-file unified diff. `src/foo.rs.bak` is in it so the exact-match case has something
    /// a prefix match would wrongly pick up.
    const DIFF: &str = "\
diff --git a/src/a.rs b/src/a.rs
index 1111111..2222222 100644
--- a/src/a.rs
+++ b/src/a.rs
@@ -1,2 +1,2 @@
 fn a() {
-    old();
+    new();
 }
diff --git a/src/b.rs b/src/b.rs
index 3333333..4444444 100644
--- a/src/b.rs
+++ b/src/b.rs
@@ -1,3 +1,2 @@
 keep();
-gone();
 keep_too();
diff --git a/src/foo.rs.bak b/src/foo.rs.bak
index 5555555..6666666 100644
--- a/src/foo.rs.bak
+++ b/src/foo.rs.bak
@@ -1 +1 @@
-backup
+backup2
";

    #[test]
    fn hunks_for_extracts_exactly_the_named_files_section() {
        let only_b = hunks_for(DIFF, &paths(&["src/b.rs"]));
        // Header and hunks both: the judgement needs the diff/---/+++ header to know what it is
        // looking at, and the hunks to see what was removed.
        assert!(
            only_b.contains("diff --git a/src/b.rs b/src/b.rs"),
            "{only_b}"
        );
        assert!(only_b.contains("--- a/src/b.rs"), "{only_b}");
        assert!(only_b.contains("+++ b/src/b.rs"), "{only_b}");
        assert!(only_b.contains("-gone();"), "{only_b}");
        // And nothing from any other file — the judgement sees only files with removed lines.
        assert!(!only_b.contains("src/a.rs"), "{only_b}");
        assert!(!only_b.contains("foo.rs.bak"), "{only_b}");
    }

    #[test]
    fn hunks_for_naming_every_file_returns_the_whole_diff() {
        let all = paths(&["src/a.rs", "src/b.rs", "src/foo.rs.bak"]);
        assert_eq!(hunks_for(DIFF, &all), DIFF);
    }

    #[test]
    fn hunks_for_matches_paths_exactly_not_by_prefix() {
        // `src/foo.rs` is not in the diff; `src/foo.rs.bak` is. A prefix match would hand the
        // judgement a section it was never asked about.
        assert!(hunks_for(DIFF, &paths(&["src/foo.rs"])).is_empty());
        // A path absent from the diff contributes nothing.
        assert!(hunks_for(DIFF, &paths(&["src/missing.rs"])).is_empty());
        // And there is nothing to extract from nothing.
        assert!(hunks_for("", &paths(&["src/b.rs"])).is_empty());
    }

    #[test]
    fn hunks_for_matches_either_side_of_a_rename() {
        let renamed = "\
diff --git a/src/old.rs b/src/new.rs
similarity index 80%
rename from src/old.rs
rename to src/new.rs
@@ -1 +1 @@
-old_check();
+new_check();
";
        for path in ["src/old.rs", "src/new.rs"] {
            let hunks = hunks_for(renamed, &paths(&[path]));
            assert!(hunks.contains("-old_check();"), "{path}: {hunks}");
        }
    }

    #[test]
    fn no_violations_parses_as_a_clean_judgement() {
        // The move-confirmed, pure-addition and clean-rewrite outcomes all look like this.
        let out = parse_validated::<RefereeOutput>(r#"{"violations":[]}"#).unwrap();
        assert!(out.violations.is_empty());
        // `violations` defaults: a model that answers `{}` has flagged nothing.
        let out = parse_validated::<RefereeOutput>(r"{}").unwrap();
        assert!(out.violations.is_empty());
    }

    #[test]
    fn a_violation_carries_its_file_and_reason_intact() {
        let raw = r#"{"violations":[{"file":"crates/ratatoskr-nodes/src/lib.rs","reason":"deleted the #[cfg(test)] module that characterised the move"}]}"#;
        let out = parse_validated::<RefereeOutput>(raw).unwrap();
        assert_eq!(out.violations.len(), 1);
        assert_eq!(out.violations[0].file, "crates/ratatoskr-nodes/src/lib.rs");
        assert_eq!(
            out.violations[0].reason,
            "deleted the #[cfg(test)] module that characterised the move"
        );
    }

    #[test]
    fn a_flag_without_a_reason_is_not_actionable() {
        // A violation the implementer cannot act on is worse than none: it burns an iteration.
        assert!(parse_validated::<RefereeOutput>(r#"{"violations":[{"file":"a.rs"}]}"#).is_err());
        assert!(
            parse_validated::<RefereeOutput>(r#"{"violations":[{"file":"a.rs","reason":""}]}"#)
                .is_err()
        );
        assert!(
            parse_validated::<RefereeOutput>(r#"{"violations":[{"file":"","reason":"r"}]}"#)
                .is_err()
        );
        assert!(
            parse_validated::<RefereeOutput>(r#"{"violations":[{"file":"a.rs","reason":" "}]}"#)
                .is_err()
        );
        // And a reason naming no file points at nothing to fix.
        assert!(parse_validated::<RefereeOutput>(r#"{"violations":[{"reason":"r"}]}"#).is_err());
    }

    #[test]
    fn the_correction_names_every_file_and_what_it_weakened() {
        let violations = vec![
            Violation {
                file: "crates/ratatoskr-nodes/src/lib.rs".into(),
                reason: "deleted the #[cfg(test)] module that characterised the move".into(),
            },
            Violation {
                file: "tests/api.rs".into(),
                reason: "relaxed the tolerance from exact to within 0.5".into(),
            },
        ];
        let text = correction(&violations);
        // Verbatim, all of them: the implementer is told what it weakened, not just where.
        for v in &violations {
            assert!(text.contains(&v.file), "file missing from: {text}");
            assert!(text.contains(&v.reason), "reason missing from: {text}");
        }
        // What this is not: adding a test is allowed and never brings you here — an implementer
        // that believes otherwise ships untested code.
        let lower = text.to_lowercase();
        assert!(lower.contains("adding"), "{text}");
        assert!(lower.contains("allowed"), "{text}");
        // And the escape hatch is named, so a task that legitimately rewrites tests learns how to
        // declare it up front instead of churning to the iteration wall.
        assert!(text.contains("mayModifyTests"), "{text}");
        assert!(text.contains("defineDefaults"), "{text}");
    }

    #[tokio::test]
    async fn nothing_to_judge_spends_no_model_call() {
        // No candidates means no judgement: the route below points nowhere, so any attempted
        // model call would have to error — an Ok here can only come from short-circuiting first.
        //
        // Field reading of the contract: `RefereeInput { issue, requirements, hunks, candidates }`
        // as String / Vec<String> / String / Vec<String>, and the node built with VerifierNode's
        // shape (all fields public, tools/ledger/files optional).
        let node = RefereeNode {
            route: ratatoskr_core::ModelRoute {
                context_window: None,
                provider: "no-such-provider".into(),
                model: "no-such-model".into(),
                max_tokens: None,
                temperature: None,
                params: None,
                session: Default::default(),
            },
            tools: ratatoskr_mcp::ToolSet::default(),
            policy: None,
            max_turns: None,
            system_prompt: None,
            plugins: crate::NodePlugins::default(),
            ledger: None,
            files: None,
        };
        let out = node
            .run(RefereeInput {
                issue: "the task".into(),
                requirements: vec!["keep the characterisation tests intact".into()],
                hunks: String::new(),
                candidates: Vec::new(),
            })
            .await
            .expect("an empty candidate set is a clean judgement, not an error");
        assert!(out.violations.is_empty());
    }
}
