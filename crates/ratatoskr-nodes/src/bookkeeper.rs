//! Bookkeeper: on a converged run, write a durable memory back into rag-rat — the phase that makes
//! a run change what the *next* run knows.
//!
//! It composes the memory content with a cheap LLM (real prose is the point — a templated dump
//! wouldn't rank on a later `MemoryNode` retrieval), then calls rag-rat's own `memory_create`
//! directly (deterministic), adopting rag-rat's `kind` taxonomy rather than inventing one.

use std::fmt::Write as _;

use ratatoskr_graph::{NodeError, parse_validated};
use ratatoskr_mcp::ToolSet;
use rmcp::model::CallToolRequestParams;
use rmcp::service::ServerSink;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::analyst::AnalystOutput;
use crate::implementer::ImplementerOutput;

/// What a run struggled with — the part of a run that its diff does not show.
///
/// The diff says what the change was. This says what nobody knew when it started: the constraint
/// that only surfaced when a test broke, the assumption in the plan that turned out wrong, the
/// node that spent forty turns finding something. That is what a future run would pay to be told,
/// and without it every run ends by discarding its most expensive lesson.
///
/// Derived entirely from checkpoints so the live path and `ratatoskr bookkeep`'s replay compose
/// from the same source and reach the same memories.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RunFriction {
    /// The converge diagnostic each implementer iteration past the first was given — literally
    /// "your change broke these, fix them". Each one cost a full implementer session.
    pub diagnostics: Vec<String>,
    /// Nodes that failed, and why. A node that had to be retried hit something.
    pub errors: Vec<NodeFailure>,
    /// How much work each node's turn took. No threshold is applied: what counts as an unusual
    /// number of turns is a judgement about this repo, so the numbers are handed over as they are.
    pub effort: Vec<NodeEffort>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeFailure {
    pub node: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeEffort {
    pub node: String,
    pub turns: u64,
    pub seconds: u64,
}

impl RunFriction {
    /// Read a run's path out of its checkpoints.
    pub fn from_checkpoints(checkpoints: &[ratatoskr_store::Checkpoint]) -> Self {
        let mut friction = RunFriction::default();
        for cp in checkpoints {
            // Iteration 1 was given the plan; everything after it was given a diagnostic saying
            // what the previous attempt broke. Only the latter is friction.
            if cp.node_name == "implementer"
                && cp.iteration.is_some_and(|i| i > 1)
                && let Some(input) = &cp.input_json
            {
                friction.diagnostics.push(unquote(input));
            }
            if let Some(error) = &cp.telemetry.error {
                friction.errors.push(NodeFailure {
                    node: cp.node_name.clone(),
                    error: error.clone(),
                });
            }
            if let (Some(turns), Some(ms)) = (cp.telemetry.turns, cp.telemetry.duration_ms) {
                friction.effort.push(NodeEffort {
                    node: cp.node_name.clone(),
                    turns,
                    seconds: ms / 1000,
                });
            }
        }
        friction
    }

    /// Whether the run's path holds anything at all. A run that changed nothing AND hit nothing
    /// has genuinely nothing to teach; one that changed nothing after a struggle does.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty() && self.errors.is_empty()
    }
}

/// Why nothing was recorded, when nothing was.
///
/// `None` as soon as anything was written: a run that recorded one memory and declined another
/// recorded something, and a skip reason reported alongside it reads as a failure it was not.
fn skip_reason(recorded: usize, declined: &[String]) -> Option<String> {
    if recorded > 0 {
        return None;
    }
    Some(match declined.is_empty() {
        true => "the bookkeeper decided nothing".to_string(),
        false => declined.join("; "),
    })
}

/// A checkpoint's `input_json` is serialized, so a plain string input arrives JSON-quoted. Show the
/// model the diagnostic it was given, not a quoted rendering of it.
fn unquote(raw: &str) -> String {
    serde_json::from_str::<String>(raw).unwrap_or_else(|_| raw.to_string())
}

/// rag-rat tools the compose agent may use to ground the memory (and to engage the tool-composing
/// output mode — see the OutputMode note in ratatoskr-agent).
pub const BOOKKEEPER_TOOLS: &[&str] = &["semantic_search", "symbol_lookup", "memory_search"];

/// rag-rat's memory taxonomy (`McpMemoryKind`). The compose model must pick one of these; anything
/// else is normalized to `Decision`.
const VALID_KINDS: &[&str] = &[
    "Invariant",
    "Decision",
    "RejectedAlternative",
    "Risk",
    "BugPattern",
    "TestExpectation",
    "PerformanceNote",
    "SecurityNote",
    "FFIBoundary",
    "PlatformQuirk",
    "FollowUp",
    "OpenQuestion",
    "Concept",
];

/// What the bookkeeper is told to look for, and in what order.
///
/// The ordering is not arbitrary and should not be shortened without a reason as good as the ones
/// that put each line here:
///
/// - **Rejected alternatives lead** because they are simultaneously the most-asked question about
///   unfamiliar code and the least-supplied answer. The alternative not taken leaves no artifact
///   anywhere — not in the diff, not in the types, not in the history. This repo had thirty-odd
///   memories and none of that kind, while its commit messages argue about little else.
/// - **"Could this be recovered by reading the repo?" is the gate**, stated first, because a
///   generated record that restates the repo measurably makes agents *worse* rather than merely
///   failing to help: it costs attention and returns nothing. Recording nothing is the correct
///   answer for most runs, which is why `none` is described as common rather than as a fallback.
/// - **Translate, never store the trajectory.** Retrieved raw traces score worse than having no
///   memory at all; distilled situation-and-action rules are what carries the benefit.
/// - **Trigger-and-action shape, with the evidence quoted.** Situated questions are not answered
///   by general advice, and quoting what the rule was generalised from is what stops the next
///   reader having to trust a summary of a summary.
/// - **`revise` is preferred to `create`** because a memory that has drifted actively misleads
///   while a missing one merely fails to help — removing wrong records is the single
///   best-evidenced improvement to a knowledge base.
/// - **Terseness is a staleness strategy, not a style preference**: every extra detail is another
///   thing a later change can falsify, and inconsistent records carry a measurably higher chance
///   of a bug in the code that trusts them.
const PREAMBLE: &str = include_str!("../prompts/bookkeeper.md");

/// What the bookkeeper decided the repository's memory should say.
///
/// Flat rather than a tagged union because models fill a flat shape far more reliably, and
/// because the fields a decision needs overlap: Rust validates the combination.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryDecision {
    /// `none`, `create`, or `revise`.
    pub action: String,
    /// Why this was the right call. The whole result when nothing is recorded, so it is never
    /// optional — "nothing to record" without a reason is indistinguishable from a failure.
    #[serde(default)]
    pub reason: String,
    /// Which memory to rewrite, for `revise`.
    #[serde(default)]
    pub memory_id: Option<String>,
    /// One of rag-rat's kinds; normalized to `Decision` if unrecognized.
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub body: String,
    /// The file this lesson is about. Often not a file the diff touched: the constraint that bit
    /// is frequently in the code that was left alone. Falls back to the first touched file.
    #[serde(default)]
    pub anchor: Option<String>,
}

/// What the bookkeeper decided, in full.
///
/// A list because one run can teach more than one thing, and a single-decision shape forces it to
/// pick — which is how a run that hit three separate footguns records one of them. Wrapped in a
/// struct rather than returned as a bare array: the schema gate and the output tool both take an
/// object, and a named field is what the model fills most reliably.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryDecisions {
    #[serde(default)]
    pub decisions: Vec<MemoryDecision>,
}

/// One memory rag-rat wrote back (id/anchor/kind strict — they're rag-rat's response).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryWritten {
    pub kind: String,
    pub anchor: String,
    pub memory_id: String,
    #[serde(default)]
    pub summary: Option<String>,
}

/// Bookkeeper output: the memories written plus the run's artifact fields.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BookkeeperOutput {
    /// Memories created. Empty is an ordinary outcome, not a failure.
    pub memories_written: Vec<MemoryWritten>,
    /// Memories this run rewrote because it made them wrong.
    #[serde(default)]
    pub memories_revised: Vec<MemoryWritten>,
    /// Why nothing was recorded, when nothing was. Present exactly when both lists are empty.
    #[serde(default)]
    pub skipped: Option<String>,
    pub iterations: u32,
    pub residual_risk_accepted: bool,
}

/// Everything the bookkeeper composes from.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BookkeeperInput {
    pub issue: String,
    pub analyst: AnalystOutput,
    pub implementer: ImplementerOutput,
    pub iterations: u32,
    /// Whether the run converged. `false` means it exhausted its iteration budget with unresolved
    /// failures — the memory is framed as a wall hit and tagged `unresolved`.
    pub converged: bool,
    /// What the run struggled with. The diff says what changed; this says what nobody knew, and
    /// it is usually where the memory worth writing comes from.
    #[serde(default)]
    pub friction: RunFriction,
}

impl BookkeeperInput {
    /// A result that records nothing, and says why.
    fn nothing_recorded(&self, reason: &str) -> BookkeeperOutput {
        BookkeeperOutput {
            memories_written: Vec::new(),
            memories_revised: Vec::new(),
            skipped: Some(reason.to_string()),
            iterations: self.iterations,
            residual_risk_accepted: false,
        }
    }
}

/// The bookkeeper node. Holds a cheap model route, a small tool subset (for the compose agent), and
/// rag-rat's sink (to call `memory_create` itself, outside the agent).
pub struct BookkeeperNode {
    pub route: ratatoskr_core::ModelRoute,
    pub tools: ToolSet,
    pub sink: ServerSink,
    pub policy: Option<std::sync::Arc<dyn ratatoskr_core::ToolPolicy>>,
    pub max_turns: Option<usize>,
    pub clarifier: Option<std::sync::Arc<dyn ratatoskr_agent::Clarifier>>,
    /// Ruleset `systemPrompt`; replaces [`PREAMBLE`] when set.
    pub system_prompt: Option<String>,
    /// What the plugins this node binds contribute to it.
    pub plugins: crate::NodePlugins,
    /// Where this node reports what its turn cost, for the checkpoint the executor writes.
    pub ledger: Option<std::sync::Arc<ratatoskr_agent::RunLedger>>,
    /// The repository its built-in file tools read within.
    pub files: Option<std::path::PathBuf>,
}

impl BookkeeperNode {
    pub async fn run(&self, input: BookkeeperInput) -> Result<BookkeeperOutput, NodeError> {
        // A run that changed nothing AND hit nothing has nothing to teach. The friction check is
        // what keeps this from throwing away the interesting case: a run that fought its way to an
        // empty diff learned something expensive about why the change was not needed.
        if input.converged
            && input.implementer.touched_files.is_empty()
            && input.implementer.diff_summary.trim().is_empty()
            && input.friction.is_empty()
        {
            tracing::info!("nothing was changed and nothing went wrong; recording no memory");
            return Ok(input.nothing_recorded("the run changed nothing and hit nothing"));
        }

        let prompt = render_prompt(&input);
        let raw = ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "bookkeeper",
            route: &self.route,
            preamble: &crate::effective_preamble(
                PREAMBLE,
                self.system_prompt.as_deref(),
                self.plugins.context.as_deref(),
            ),
            question: &prompt,
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(MemoryDecisions),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            clarifier: self.clarifier.clone(),
            observer: self.plugins.observer.clone(),
            skills: crate::skills::loaded(&self.plugins.skills),
            files: self.files.clone(),
            // Reads and edits, but runs nothing.
            shell: None,
            ledger: self.ledger.clone(),
            produces: Some(
                "a decision per durable learning: create, revise or none, with the memory body and its anchor",
            ),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("bookkeeper compose failed: {e}")))?;

        let decided = parse_validated::<MemoryDecisions>(&raw)?;
        self.act_on(decided.decisions, &input).await
    }

    /// Carry out what the model decided. The model chooses; this performs the writes, so the
    /// memory layer only ever changes through a call this code made deliberately.
    ///
    /// Each entry is independent: one that names no memory to revise, or composes an empty body,
    /// is dropped without taking the others with it. A run that learned three things and botched
    /// the wording of one should still record the other two.
    async fn act_on(
        &self,
        decisions: Vec<MemoryDecision>,
        input: &BookkeeperInput,
    ) -> Result<BookkeeperOutput, NodeError> {
        let mut written = Vec::new();
        let mut revised = Vec::new();
        let mut declined = Vec::new();

        for decision in decisions {
            match decision.action.trim().to_ascii_lowercase().as_str() {
                "revise" => match self.revise(&decision).await? {
                    Some(entry) => revised.push(entry),
                    None => declined.push("a revision named no memory to rewrite".to_string()),
                },
                "create" => match self.create(&decision, input).await? {
                    Some(entry) => written.push(entry),
                    None => declined.push("a composed memory was empty".to_string()),
                },
                // Including anything unrecognised: the safe reading of a decision we can't parse
                // is that nothing should be written.
                other => declined.push(match decision.reason.trim() {
                    "" => format!("nothing recorded ({other})"),
                    reason => reason.to_string(),
                }),
            }
        }

        let skipped = skip_reason(written.len() + revised.len(), &declined);
        if let Some(reason) = &skipped {
            tracing::info!(reason = %reason, "recording no memory");
        }
        Ok(BookkeeperOutput {
            memories_written: written,
            memories_revised: revised,
            skipped,
            iterations: input.iterations,
            residual_risk_accepted: false,
        })
    }

    /// Rewrite a memory this run made wrong. `None` when the decision named no target or no body:
    /// a revision without an id is a create that lost its id, and guessing which memory to
    /// overwrite is worse than recording nothing.
    async fn revise(&self, decision: &MemoryDecision) -> Result<Option<MemoryWritten>, NodeError> {
        let (Some(id), false) = (
            decision.memory_id.as_deref().filter(|id| !id.is_empty()),
            decision.body.trim().is_empty(),
        ) else {
            tracing::warn!("bookkeeper asked to revise without a memory id or body");
            return Ok(None);
        };
        let title = (!decision.title.trim().is_empty()).then(|| decision.title.clone());
        self.update_memory(id, title.as_deref(), &decision.body)
            .await?;
        tracing::info!(memory_id = id, "revised a memory this run made wrong");
        Ok(Some(MemoryWritten {
            kind: normalize_kind(&decision.kind),
            anchor: String::new(),
            memory_id: id.to_string(),
            summary: title.or_else(|| Some(decision.reason.clone())),
        }))
    }

    /// Write a new memory. `None` when the model composed an empty one.
    async fn create(
        &self,
        decision: &MemoryDecision,
        input: &BookkeeperInput,
    ) -> Result<Option<MemoryWritten>, NodeError> {
        if decision.title.trim().is_empty() || decision.body.trim().is_empty() {
            tracing::warn!("bookkeeper produced an empty memory; recording nothing");
            return Ok(None);
        }
        let kind = normalize_kind(&decision.kind);
        // The lesson's own file first: what a run learned is often about code it did not change,
        // and anchoring that to the diff would file it where nobody looking for it would look.
        let anchor = decision
            .anchor
            .as_deref()
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .map(str::to_string)
            .or_else(|| input.implementer.touched_files.first().cloned());
        // Tag unresolved (max-iterations) runs so they're distinguishable from success write-backs.
        let tags: &[&str] = if input.converged {
            &["ratatoskr", "bookkeeper"]
        } else {
            &["ratatoskr", "bookkeeper", "unresolved"]
        };

        let memory_id = self
            .create_memory(
                &kind,
                &decision.title,
                &decision.body,
                anchor.as_deref(),
                tags,
            )
            .await?;
        Ok(Some(MemoryWritten {
            kind,
            anchor: anchor.unwrap_or_default(),
            memory_id,
            summary: Some(decision.title.clone()),
        }))
    }

    /// Rewrite an existing memory through rag-rat's `memory_update`.
    ///
    /// The body replaces the old one outright rather than being appended to: a memory that ends
    /// in a list of what changed is one nobody finishes reading, and the point of revising is that
    /// it states what is true now.
    async fn update_memory(
        &self,
        memory_id: &str,
        title: Option<&str>,
        body: &str,
    ) -> Result<(), NodeError> {
        let mut args = serde_json::json!({ "memory_id": memory_id, "body": body });
        if let Some(title) = title {
            args["title"] = serde_json::json!(title);
        }
        let arguments = args.as_object().cloned().expect("json object literal");
        let param = CallToolRequestParams::new("memory_update").with_arguments(arguments);

        let result = self
            .sink
            .call_tool(param)
            .await
            .map_err(|e| NodeError::Failed(format!("memory_update call failed: {e}")))?;

        if result.is_error.unwrap_or(false) {
            let text = result
                .content
                .iter()
                .filter_map(|c| c.as_text())
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join("");
            return Err(NodeError::Failed(format!(
                "memory_update returned an error: {text}"
            )));
        }
        Ok(())
    }

    /// Call rag-rat's `memory_create` directly and return the new memory id.
    async fn create_memory(
        &self,
        kind: &str,
        title: &str,
        body: &str,
        anchor: Option<&str>,
        tags: &[&str],
    ) -> Result<String, NodeError> {
        let mut args = serde_json::json!({
            "kind": kind,
            "title": title,
            "body": body,
            "confidence": "medium",
            "source": "agent",
            "tags": tags,
        });
        // Always anchor: rag-rat rejects an unanchored memory unless it's a Task/Concept. Use the
        // touched file if we have one, else bind to the repo root directory.
        args["bind"] = match anchor {
            Some(path) => serde_json::json!({ "path": path }),
            None => serde_json::json!({ "dir": "" }),
        };
        let arguments = args.as_object().cloned().expect("json object literal");
        let param = CallToolRequestParams::new("memory_create").with_arguments(arguments);

        let result = self
            .sink
            .call_tool(param)
            .await
            .map_err(|e| NodeError::Failed(format!("memory_create call failed: {e}")))?;

        let text = result
            .content
            .iter()
            .filter_map(|c| c.as_text())
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("");

        if result.is_error.unwrap_or(false) {
            return Err(NodeError::Failed(format!(
                "memory_create returned an error: {text}"
            )));
        }
        parse_memory_id(&text)
    }
}

/// Pull the created memory id out of `memory_create`'s JSON response.
fn parse_memory_id(text: &str) -> Result<String, NodeError> {
    let value: serde_json::Value = serde_json::from_str(text.trim()).map_err(|e| {
        NodeError::Failed(format!(
            "memory_create response not JSON ({e}); is rag-rat on --json?"
        ))
    })?;
    value
        .get("memory")
        .and_then(|m| m.get("memory_id"))
        .or_else(|| value.get("memory_id"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| NodeError::Failed(format!("no memory_id in memory_create response: {text}")))
}

/// Map the model's `kind` to a valid rag-rat kind, defaulting to `Decision`.
fn normalize_kind(kind: &str) -> String {
    VALID_KINDS
        .iter()
        .find(|k| k.eq_ignore_ascii_case(kind.trim()))
        .map(|k| k.to_string())
        .unwrap_or_else(|| "Decision".to_string())
}

fn render_prompt(input: &BookkeeperInput) -> String {
    let mut s = String::new();
    if input.converged {
        s.push_str("OUTCOME: the run CONVERGED — the change landed and the tests pass.\n\n");
    } else {
        let _ = write!(
            s,
            "OUTCOME: the run HIT A WALL — after {} implementer iterations it could not resolve \
             these failing tests: {}. Record what a future run should know about this wall / this \
             class of change.\n\n",
            input.iterations,
            input.implementer.failing_tests.join(", ")
        );
    }
    let _ = write!(s, "TASK:\n{}\n\n", input.issue);
    let a = &input.analyst;
    if !a.impact_summary.is_empty() {
        let _ = write!(s, "IMPACT:\n{}\n\n", a.impact_summary);
    }
    if !a.risks.is_empty() {
        s.push_str("RISKS FLAGGED:\n");
        for r in &a.risks {
            let _ = writeln!(s, "- {r}");
        }
        s.push('\n');
    }
    let im = &input.implementer;
    if !im.diff_summary.is_empty() {
        let _ = write!(s, "DIFF:\n{}\n\n", im.diff_summary);
    }
    if let Some(narrative) = &im.narrative
        && !narrative.is_empty()
    {
        let _ = write!(s, "IMPLEMENTER NOTES:\n{narrative}\n\n");
    }
    if !im.touched_files.is_empty() {
        let _ = writeln!(s, "TOUCHED FILES: {}", im.touched_files.join(", "));
    }

    // Last, and deliberately: it is the section the preamble tells the model to reason from, and
    // what a model reads last is what it reasons from first.
    let f = &input.friction;
    if !f.diagnostics.is_empty() || !f.errors.is_empty() || !f.effort.is_empty() {
        s.push_str("\nFRICTION — what this run struggled with:\n");
    }
    if !f.diagnostics.is_empty() {
        let _ = writeln!(
            s,
            "\nEach of these was handed to a fresh implementer session after the previous attempt \
             broke something. Whatever a diagnostic keeps pointing at is a constraint nobody had \
             written down:"
        );
        for (i, d) in f.diagnostics.iter().enumerate() {
            let _ = writeln!(s, "- attempt {}: {}", i + 2, d);
        }
    }
    if !f.errors.is_empty() {
        s.push_str("\nNodes that failed:\n");
        for e in &f.errors {
            let _ = writeln!(s, "- {}: {}", e.node, e.error);
        }
    }
    if !f.effort.is_empty() {
        s.push_str(
            "\nWhat each node's turn took. A node that spent many turns was hunting for \
                    something — that is a fact about how hard this repo is to navigate, not about \
                    the node:\n",
        );
        for e in &f.effort {
            let _ = writeln!(s, "- {}: {} turns, {}s", e.node, e.turns, e.seconds);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_kinds() {
        assert_eq!(normalize_kind("risk"), "Risk");
        assert_eq!(normalize_kind("Invariant"), "Invariant");
        assert_eq!(normalize_kind("nonsense"), "Decision");
        assert_eq!(normalize_kind(""), "Decision");
    }

    #[test]
    fn parses_memory_id_from_json() {
        let ok =
            parse_memory_id(r#"{"memory":{"memory_id":"mem_abc","kind":"Decision"}}"#).unwrap();
        assert_eq!(ok, "mem_abc");
        let flat = parse_memory_id(r#"{"memory_id":"mem_xyz"}"#).unwrap();
        assert_eq!(flat, "mem_xyz");
        assert!(parse_memory_id("not json").is_err());
        assert!(parse_memory_id(r#"{"nope":1}"#).is_err());
    }

    #[test]
    fn a_decision_parses_with_only_what_that_action_needs() {
        // Declining carries a reason and nothing else.
        let none =
            parse_validated::<MemoryDecision>(r#"{"action":"none","reason":"already recorded"}"#)
                .unwrap();
        assert_eq!(none.action, "none");
        assert!(none.title.is_empty());

        // Creating without a kind still normalizes to a valid one.
        let create =
            parse_validated::<MemoryDecision>(r#"{"action":"create","title":"t","body":"b"}"#)
                .unwrap();
        assert_eq!(normalize_kind(&create.kind), "Decision");

        let revise = parse_validated::<MemoryDecision>(
            r#"{"action":"revise","memory_id":"mem_1","body":"now true"}"#,
        )
        .unwrap();
        assert_eq!(revise.memory_id.as_deref(), Some("mem_1"));

        // An action is the one thing a decision must state.
        assert!(parse_validated::<MemoryDecision>(r#"{"reason":"x"}"#).is_err());
    }

    fn input(converged: bool, touched: &[&str], diff: &str) -> BookkeeperInput {
        BookkeeperInput {
            issue: "an issue".into(),
            analyst: AnalystOutput {
                impact_summary: "impact".into(),
                touched: Vec::new(),
                risks: Vec::new(),
                requirements: Vec::new(),
                residual_risk: String::new(),
                changes_code: true,
                acceptance: Vec::new(),
            },
            implementer: ImplementerOutput {
                worktree_path: "/tmp/wt".into(),
                diff_summary: diff.into(),
                touched_files: touched.iter().map(|s| (*s).to_string()).collect(),
                failing_tests: Vec::new(),
                passing_tests: Vec::new(),
                exit_code: 0,
                narrative: None,
            },
            iterations: 1,
            converged,
            friction: RunFriction::default(),
        }
    }

    #[test]
    fn a_run_that_changed_nothing_records_nothing() {
        // The case that used to store a memory whose content was that there was nothing to store.
        let out = input(true, &[], "").nothing_recorded("the run changed nothing");
        assert!(out.memories_written.is_empty());
        assert!(out.memories_revised.is_empty());
        assert_eq!(out.skipped.as_deref(), Some("the run changed nothing"));
        assert_eq!(out.iterations, 1);
    }

    #[test]
    fn a_run_that_hit_a_wall_without_touching_files_still_has_something_to_say() {
        // Converged-and-untouched is a no-op; walled-and-untouched is a wall worth recording.
        let walled = input(false, &[], "");
        assert!(
            !(walled.converged
                && walled.implementer.touched_files.is_empty()
                && walled.implementer.diff_summary.trim().is_empty()),
            "the skip must not swallow a run that failed to get started"
        );
    }

    fn checkpoint(
        node: &str,
        iteration: Option<u32>,
        input: Option<&str>,
    ) -> ratatoskr_store::Checkpoint {
        ratatoskr_store::Checkpoint {
            node_name: node.into(),
            iteration,
            input_json: input.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn friction_is_the_diagnostics_a_rerun_was_given_not_the_original_plan() {
        let cps = [
            checkpoint(
                "implementer",
                Some(1),
                Some(r#"{"requirements":["do the thing"]}"#),
            ),
            checkpoint(
                "implementer",
                Some(2),
                Some(r#""You broke store::migrate. Fix it.""#),
            ),
            checkpoint(
                "implementer",
                Some(3),
                Some(r#""You broke store::migrate again.""#),
            ),
        ];
        let f = RunFriction::from_checkpoints(&cps);

        // Iteration 1 was handed the plan; only what came after it is friction.
        assert_eq!(f.diagnostics.len(), 2);
        // Unquoted, so the model reads the diagnostic rather than a JSON rendering of it.
        assert_eq!(f.diagnostics[0], "You broke store::migrate. Fix it.");
        assert!(!f.is_empty());
    }

    #[test]
    fn a_clean_run_has_no_friction_but_a_failed_node_does() {
        let clean = [checkpoint("implementer", Some(1), Some(r#"{"plan":1}"#))];
        assert!(RunFriction::from_checkpoints(&clean).is_empty());

        let mut failed = checkpoint("analyst", None, None);
        failed.telemetry.error = Some("output failed schema validation".into());
        failed.telemetry.turns = Some(41);
        failed.telemetry.duration_ms = Some(90_000);
        let f = RunFriction::from_checkpoints(&[failed]);

        assert!(!f.is_empty(), "a node that failed is something the run hit");
        assert_eq!(f.errors[0].node, "analyst");
        assert_eq!(f.effort[0].turns, 41);
        assert_eq!(f.effort[0].seconds, 90);
    }

    #[test]
    fn effort_alone_is_not_friction() {
        // Every node reports turns and duration. If those counted, no run would ever be quiet and
        // the "nothing happened" short-circuit would never fire.
        let mut cp = checkpoint("scout", None, None);
        cp.telemetry.turns = Some(4);
        cp.telemetry.duration_ms = Some(1_000);
        let f = RunFriction::from_checkpoints(&[cp]);
        assert!(!f.effort.is_empty());
        assert!(f.is_empty());
    }

    #[test]
    fn a_skip_reason_is_only_reported_when_nothing_was_recorded() {
        let declined = vec!["a composed memory was empty".to_string()];
        assert_eq!(
            skip_reason(0, &declined).as_deref(),
            Some("a composed memory was empty")
        );
        assert_eq!(
            skip_reason(1, &declined),
            None,
            "one memory written and one declined is a run that recorded something"
        );
        // Nothing decided at all still owes an explanation: an empty result with no reason is
        // indistinguishable from a failure.
        assert!(skip_reason(0, &[]).is_some());
    }

    #[test]
    fn the_prompt_shows_the_model_what_the_run_struggled_with() {
        let mut input = input(true, &["a.rs"], "diff");
        input.friction.diagnostics = vec!["You broke store::migrate.".into()];
        let prompt = render_prompt(&input);
        assert!(prompt.contains("FRICTION"));
        assert!(prompt.contains("You broke store::migrate."));
        // Numbered from 2: the first attempt was given the plan, so the first diagnostic belongs
        // to the second attempt.
        assert!(prompt.contains("attempt 2"));
    }
}
