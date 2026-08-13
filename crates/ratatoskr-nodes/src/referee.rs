use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use ratatoskr_agent::RunLedger;
use ratatoskr_core::{ModelRoute, RatatoskrConfig};
use ratatoskr_exec::WorktreePath;
use ratatoskr_graph::NodeError;
use ratatoskr_mcp::ToolSet;
use ratatoskr_script::ScriptEngine;
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
    route: ModelRoute,
    tools: ToolSet,
    ledger: Option<Arc<RunLedger>>,
    /// Set after the fork when the worktree becomes known.
    pub files: Option<PathBuf>,
}

impl RefereeNode {
    /// Construct the internal judge with its fixed, read-only capability boundary.
    pub fn fixed(
        route: ModelRoute,
        ledger: Option<Arc<RunLedger>>,
        files: Option<PathBuf>,
    ) -> Self {
        let mut tools = ToolSet::default();
        tools.add_local_tools(ratatoskr_agent::files::declarations());
        tools.narrow(
            &REFEREE_TOOLS
                .iter()
                .map(|name| (*name).to_string())
                .collect::<Vec<_>>(),
            &[],
        );
        Self {
            route,
            tools,
            ledger,
            files,
        }
    }

    pub async fn run(&self, input: RefereeInput) -> Result<RefereeOutput, NodeError> {
        if input.candidates.is_empty() {
            return Ok(RefereeOutput {
                violations: Vec::new(),
            });
        }

        crate::verifier::run_judgement(
            ratatoskr_agent::NodeRun {
                node: "referee",
                controlled_as: None,
                route: &self.route,
                preamble: PREAMBLE,
                question: &render_prompt(&input),
                tools: self.tools.clone(),
                output_schema: schemars::schema_for!(RefereeOutput),
                policy: None,
                max_turns: None,
                clarifier: None,
                observer: None,
                skills: Vec::new(),
                files: self.files.clone(),
                rag_rat_worktree: self.files.clone(),
                shell: None,
                push: None,
                conversation: None,
                ledger: self.ledger.clone(),
                produces: Some("files whose diff hunks weakened task-completion checks, each with a reason, or none"),
            },
            "referee",
        )
        .await
    }
}

/// Judge the current change before its acceptance result is trusted.
///
/// This is deliberately separate from the callers' checkpoint mechanics: both convergence paths
/// use the same fixed route, candidates, diff extraction and model invocation, while retaining
/// their own iteration metadata when they write the observable `referee` record. `Ok(None)` means
/// no route was configured, so callers must not write a referee checkpoint.
pub(crate) struct Judgement<'a> {
    pub engine: &'a Arc<ScriptEngine>,
    pub config: &'a RatatoskrConfig,
    /// The registry this run executes, so the verifier fallback resolves against the stage that
    /// will actually review rather than a fixed table.
    pub stages: &'a [crate::Stage],
    pub ledger: &'a Arc<RunLedger>,
    pub issue: &'a str,
    pub requirements: &'a [String],
    pub implementer: &'a crate::ImplementerOutput,
    pub worktree: &'a WorktreePath,
}

pub(crate) async fn judge(
    judgement: Judgement<'_>,
) -> Result<Option<Vec<Violation>>, crate::PlanError> {
    let Judgement {
        engine,
        config,
        stages,
        ledger,
        issue,
        requirements,
        implementer,
        worktree,
    } = judgement;
    let Some(route) = crate::referee_route(engine, config, stages) else {
        tracing::info!("no referee or verifier route configured; trusting test results alone");
        return Ok(None);
    };
    let candidates = crate::converge::referee_candidates(
        &implementer.rewritten_files,
        engine.may_modify_tests(),
    );
    if candidates.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let diff = ratatoskr_exec::full_diff_text(worktree)
        .await
        .unwrap_or_default();
    let input = RefereeInput {
        issue: issue.to_string(),
        requirements: requirements.to_vec(),
        hunks: hunks_for(&diff, &candidates),
        candidates,
    };
    let node = RefereeNode::fixed(
        route,
        Some(Arc::clone(ledger)),
        Some(worktree.as_path().to_path_buf()),
    );
    node.run(input)
        .await
        .map(|out| Some(out.violations))
        .map_err(|error| crate::PlanError::node("referee", error))
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

    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use ratatoskr_agent::RunLedger;
    use ratatoskr_core::RatatoskrConfig;
    use ratatoskr_exec::WorktreePath;
    use ratatoskr_graph::parse_validated;
    use ratatoskr_script::ScriptEngine;

    use crate::implementer::ImplementerOutput;

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
        // as String / Vec<String> / String / Vec<String>, with the fixed referee retaining its own
        // private route, tool, and ledger fields.
        let node = RefereeNode::fixed(
            ratatoskr_core::ModelRoute {
                context_window: None,
                provider: "no-such-provider".into(),
                model: "no-such-model".into(),
                max_tokens: None,
                temperature: None,
                params: None,
                session: Default::default(),
            },
            None,
            None,
        );
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

    // Contract reading (#209): the referee stops being a governable node and becomes an
    // internal, fixed-capability judgement. Two symbols pin the change here:
    //
    //   RefereeNode::fixed(route: ModelRoute, ledger: Option<Arc<RunLedger>>, files: Option<PathBuf>) -> RefereeNode
    //
    // — the only construction path, building its ToolSet directly from REFEREE_TOOLS. The
    // contract's sad case (a ruleset with tools.deny = ["Read", "Grep", "Glob"], a bound plugin
    // offering Write/Bash) is enforced structurally: no engine, plugin pool, skill,
    // system-prompt or policy parameter exists for that influence to arrive through, so the
    // tests below assert the observable set and the absence of every influence socket rather
    // than feeding a ruleset in.
    //
    // — the one entry point both convergence paths (the built-in loop and workflow.rs's
    // iterate_host / finish_full) call. It returns `None` only when no route is configured, so
    // callers skip the referee record; checkpointing itself stays with the callers so they can
    // retain their own iteration metadata.

    fn route(provider: &str, model: &str) -> ModelRoute {
        ModelRoute {
            context_window: None,
            provider: provider.into(),
            model: model.into(),
            max_tokens: None,
            temperature: None,
            params: None,
            session: Default::default(),
        }
    }

    /// A ruleset directory containing exactly `source`, loaded minus the CLI's governable-name
    /// gate: rejecting "referee" at startup is the gate's job (lib.rs's governance tests
    /// compose its predicate); the engine itself still parses the file, which is what lets the
    /// route tests below prove a referee ruleset is never consulted.
    async fn rules_engine(case: &str, source: &str) -> Arc<ScriptEngine> {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-referee-fixed-{}-{case}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("agents.ts"), source).unwrap();
        ScriptEngine::load(&dir).await.unwrap()
    }

    fn implementer(rewritten: &[&str]) -> ImplementerOutput {
        ImplementerOutput {
            worktree_path: "/wt".into(),
            branch: "ratatoskr/test".into(),
            diff_summary: String::new(),
            touched_files: Vec::new(),
            rewritten_files: rewritten.iter().map(|s| s.to_string()).collect(),
            failing_tests: Vec::new(),
            passed_tests: 0,
            exit_code: 0,
            narrative: None,
            commit_kind: String::new(),
            commit_scope: String::new(),
            commit_subject: String::new(),
        }
    }

    /// A worktree path that is a plain temp directory, not a git checkout: diff extraction from
    /// it fails and yields the empty diff the callers already treat as "nothing to judge".
    fn worktree(case: &str) -> WorktreePath {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-referee-wt-{}-{case}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        WorktreePath(dir)
    }

    #[test]
    fn fixed_construction_hands_the_judge_exactly_the_read_tools() {
        let node = RefereeNode::fixed(route("anthropic", "claude-sonnet-4-6"), None, None);
        // Exactly these three: enough to confirm a relocated check still exists, nothing that
        // can change the worktree or run a command.
        assert_eq!(node.tools.names(), REFEREE_TOOLS);
        assert_eq!(
            node.tools.names(),
            ["Read", "Grep", "Glob"],
            "the fixed set is the read tools and only the read tools"
        );
    }

    #[test]
    fn fixed_construction_admits_no_ruleset_or_plugin_influence() {
        // The contract's sad case is not expressible against this constructor, and that is the
        // point: the enforcement is structural. `fixed` has no engine, pool, skill, prompt or
        // policy parameter, so there is no socket a tools.deny ruleset or a Write/Bash-offering
        // plugin could plug into. What is observable is the constructed node carrying none of
        // those influences and the full read set regardless.
        let node = RefereeNode::fixed(
            route("openai", "gpt-5"),
            Some(Arc::new(RunLedger::default())),
            Some(PathBuf::from("/wt")),
        );
        let names = node.tools.names();
        for &read in REFEREE_TOOLS {
            assert!(
                names.iter().any(|n| n.as_str() == read),
                "the judge lost {read}: {names:?}"
            );
        }
        assert!(
            !names.iter().any(|n| n == "Write" || n == "Bash"),
            "the judge gained a write capability: {names:?}"
        );
        // The inputs the constructor takes are kept: the route to judge on, the run's ledger, and
        // the worktree the file tools are rooted at.
        assert_eq!(node.route.provider, "openai");
        assert_eq!(node.route.model, "gpt-5");
        assert!(node.ledger.is_some());
        assert_eq!(node.files.as_deref(), Some(Path::new("/wt")));
    }

    #[tokio::test]
    async fn a_fixed_node_with_nothing_to_judge_spends_no_model_call() {
        // The route points nowhere, so any attempted model call would have to error — an Ok
        // here can only come from short-circuiting on the empty candidate set first.
        let node = RefereeNode::fixed(route("no-such-provider", "no-such-model"), None, None);
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

    #[tokio::test]
    async fn a_fixed_node_with_candidates_really_calls_the_model() {
        // The complement of the short-circuit: with candidates to judge, the node does attempt
        // the call, so an unreachable route surfaces as an error at this level (fail-open lives
        // in `judge`, one level up). This is what makes the empty-candidate Ok above — and the
        // skip cases below — meaningful rather than a swallowed failure.
        let node = RefereeNode::fixed(route("no-such-provider", "no-such-model"), None, None);
        let result = node
            .run(RefereeInput {
                issue: "the task".into(),
                requirements: Vec::new(),
                hunks: "diff --git a/src/lib.rs b/src/lib.rs\n@@ -1 +1 @@\n-old();\n".into(),
                candidates: vec!["src/lib.rs".to_string()],
            })
            .await;
        assert!(
            result.is_err(),
            "candidates to judge and no reachable model must surface an error"
        );
    }

    #[tokio::test]
    async fn no_configured_route_skips_the_judgement_without_an_error() {
        // Neither [models.referee] nor any verifier route anywhere: the acceptance result is
        // trusted, the log says why, and the skipped judgement is not a run failure — even with
        // rewritten files that would otherwise be judged.
        let engine = rules_engine("judge-no-route", "").await;
        let config = RatatoskrConfig::default();
        let ledger = Arc::new(RunLedger::default());
        let violations = judge(Judgement {
            engine: &engine,
            config: &config,
            stages: &[crate::stage::stage_fixture("verifier", "explore")],
            ledger: &ledger,
            issue: "the issue",
            requirements: &["keep the tests intact".to_string()],
            implementer: &implementer(&["crates/foo/src/lib.rs"]),
            worktree: &worktree("judge-no-route"),
        })
        .await
        .expect("no route is a skipped judgement, not an error");
        assert!(
            violations.is_none(),
            "no route has no judgement to checkpoint"
        );
    }

    #[tokio::test]
    async fn exempt_rewrites_and_empty_candidate_sets_skip_the_judgement() {
        // A route IS configured but points nowhere: were the judgement to run, the model call
        // would fail (see `a_fixed_node_with_candidates_really_calls_the_model`). The empty
        // results here are therefore meaningful only as skips — candidates must be computed and
        // found empty, with the mayModifyTests exemption applied, before any diff extraction or
        // model call is paid for.
        let engine = rules_engine(
            "judge-exempt",
            r#"defineDefaults({ mayModifyTests: ["crates/foo/tests"] });"#,
        )
        .await;
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "referee".to_string(),
            route("no-such-provider", "no-such-model"),
        );
        let ledger = Arc::new(RunLedger::default());

        // Every rewrite sits under the declared exemption: no candidates, no judgement.
        let violations = judge(Judgement {
            engine: &engine,
            config: &config,
            stages: &[crate::stage::stage_fixture("verifier", "explore")],
            ledger: &ledger,
            issue: "the issue",
            requirements: &[],
            implementer: &implementer(&["crates/foo/tests/api.rs"]),
            worktree: &worktree("judge-exempt"),
        })
        .await
        .expect("nothing to judge is not an error");
        assert!(matches!(violations, Some(ref violations) if violations.is_empty()));

        // And the trivial spelling: the implementer rewrote nothing at all.
        let violations = judge(Judgement {
            engine: &engine,
            config: &config,
            stages: &[crate::stage::stage_fixture("verifier", "explore")],
            ledger: &ledger,
            issue: "the issue",
            requirements: &[],
            implementer: &implementer(&[]),
            worktree: &worktree("judge-exempt"),
        })
        .await
        .expect("nothing rewritten is nothing to judge");
        assert!(matches!(violations, Some(ref violations) if violations.is_empty()));
    }

    #[tokio::test]
    async fn a_failing_judgement_surfaces_an_error_for_checkpointing() {
        // Candidates to judge, a route that resolves, and a model that cannot answer: `judge`
        // returns the error so each convergence path can checkpoint it under "referee" before
        // trusting the acceptance result.
        let engine = rules_engine("judge-fails-open", "").await;
        let mut config = RatatoskrConfig::default();
        config.models.insert(
            "referee".to_string(),
            route("no-such-provider", "no-such-model"),
        );
        let ledger = Arc::new(RunLedger::default());
        let result = judge(Judgement {
            engine: &engine,
            config: &config,
            stages: &[crate::stage::stage_fixture("verifier", "explore")],
            ledger: &ledger,
            issue: "the issue",
            requirements: &[],
            implementer: &implementer(&["crates/foo/src/lib.rs"]),
            worktree: &worktree("judge-fails-open"),
        })
        .await;
        assert!(
            result.is_err(),
            "a failed judgement must reach the caller for failure checkpointing"
        );
    }
}
