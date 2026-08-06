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

const PREAMBLE: &str = "You are the bookkeeper. A coding run just finished (the prompt says \
    whether it succeeded or hit a wall). Decide what, if anything, the repository's memory should \
    now say — and act on the one that fits.\n\n\
    FIRST search the existing memories with `memory_search` for whatever this change touched. What \
    you find decides between three outcomes:\n\
    - `revise` — this run made an existing memory WRONG or incomplete. Rewrite its body to state \
      what is true NOW; do not append a status section or a changelog. This is the right answer \
      more often than it looks, because a change that alters behaviour usually contradicts \
      something already recorded.\n\
    - `create` — there is a durable, non-obvious learning here that nothing already covers: an \
      invariant, a decision and its rationale, a gotcha, a risk, or (if the run hit a wall) what \
      that wall was and what to watch for.\n\
    - `none` — nothing worth recording. This is a perfectly good and COMMON answer. A vague, \
      obvious, or duplicate memory is worse than no memory at all, because every future run pays \
      to read it. If the change was routine, or what you would write is already recorded, choose \
      this and say why.\n\n\
    Write in the present tense: what is true now and how to apply it, NOT a narrative of what this \
    run did. Be specific and grounded. Choose a `kind` from rag-rat's taxonomy: Invariant, \
    Decision, RejectedAlternative, Risk, BugPattern, TestExpectation, PerformanceNote, \
    SecurityNote, FFIBoundary, PlatformQuirk, FollowUp, OpenQuestion, Concept.";

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
        // A run that changed nothing has nothing to teach. Exact, and it costs no model call —
        // this is the case that used to store a memory saying there was nothing to store.
        if input.converged
            && input.implementer.touched_files.is_empty()
            && input.implementer.diff_summary.trim().is_empty()
        {
            tracing::info!("nothing was changed; recording no memory");
            return Ok(input.nothing_recorded("the run changed nothing"));
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
            output_schema: schemars::schema_for!(MemoryDecision),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            clarifier: self.clarifier.clone(),
            observer: self.plugins.observer.clone(),
            skills: crate::skills::loaded(&self.plugins.skills),
            files: self.files.clone(),
            ledger: self.ledger.clone(),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("bookkeeper compose failed: {e}")))?;

        let decision = parse_validated::<MemoryDecision>(&raw)?;
        self.act_on(decision, &input).await
    }

    /// Carry out what the model decided. The model chooses; this performs the write, so the
    /// memory layer only ever changes through a call this code made deliberately.
    async fn act_on(
        &self,
        decision: MemoryDecision,
        input: &BookkeeperInput,
    ) -> Result<BookkeeperOutput, NodeError> {
        match decision.action.trim().to_ascii_lowercase().as_str() {
            "revise" => {
                // A revision without a target is a create that lost its id; treat the ambiguity as
                // "record nothing" rather than guessing which memory to overwrite.
                let (Some(id), false) = (
                    decision.memory_id.as_deref().filter(|id| !id.is_empty()),
                    decision.body.trim().is_empty(),
                ) else {
                    tracing::warn!("bookkeeper asked to revise without a memory id or body");
                    return Ok(input.nothing_recorded("the revision named no memory to rewrite"));
                };
                let title = (!decision.title.trim().is_empty()).then(|| decision.title.clone());
                self.update_memory(id, title.as_deref(), &decision.body)
                    .await?;
                tracing::info!(memory_id = id, "revised a memory this run made wrong");

                Ok(BookkeeperOutput {
                    memories_written: Vec::new(),
                    memories_revised: vec![MemoryWritten {
                        kind: normalize_kind(&decision.kind),
                        anchor: String::new(),
                        memory_id: id.to_string(),
                        summary: title.or(Some(decision.reason)),
                    }],
                    skipped: None,
                    iterations: input.iterations,
                    residual_risk_accepted: false,
                })
            }
            "create" => {
                if decision.title.trim().is_empty() || decision.body.trim().is_empty() {
                    tracing::warn!("bookkeeper produced an empty memory; recording nothing");
                    return Ok(input.nothing_recorded("the composed memory was empty"));
                }
                let kind = normalize_kind(&decision.kind);
                let anchor = input.implementer.touched_files.first().cloned();
                // Tag unresolved (max-iterations) runs so they're distinguishable from success
                // write-backs.
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

                Ok(BookkeeperOutput {
                    memories_written: vec![MemoryWritten {
                        kind,
                        anchor: anchor.unwrap_or_default(),
                        memory_id,
                        summary: Some(decision.title),
                    }],
                    memories_revised: Vec::new(),
                    skipped: None,
                    iterations: input.iterations,
                    residual_risk_accepted: false,
                })
            }
            // Including anything unrecognised: the safe reading of a decision we can't parse is
            // that nothing should be written.
            other => {
                let reason = if decision.reason.trim().is_empty() {
                    format!("nothing recorded ({other})")
                } else {
                    decision.reason.clone()
                };
                tracing::info!(reason = %reason, "recording no memory");
                Ok(input.nothing_recorded(&reason))
            }
        }
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
}
