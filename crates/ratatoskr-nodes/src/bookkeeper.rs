//! Bookkeeper: on a converged run, write a durable memory back into rag-rat — the phase that makes
//! a run change what the *next* run knows.
//!
//! It composes the memory content with a cheap LLM (real prose is the point — a templated dump
//! wouldn't rank on a later `MemoryNode` retrieval), then calls rag-rat's own `memory_create`
//! directly (deterministic), adopting rag-rat's `kind` taxonomy rather than inventing one.

use std::fmt::Write as _;

use ratatoskr_graph::{NodeError, parse_validated};
use rmcp::model::CallToolRequestParams;
use rmcp::service::ServerSink;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::analyst::AnalystOutput;
use crate::implementer::ImplementerOutput;

/// rag-rat tools the compose agent may use to ground the memory (and to engage the tool-composing
/// output mode — see the OutputMode note in ratatoskr-agent).
pub const BOOKKEEPER_TOOLS: &[&str] = &["semantic_search", "symbol_lookup"];

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

const PREAMBLE: &str = "You are the bookkeeper. A coding run just finished (the prompt says whether \
    it succeeded or hit a wall). Distill ONE durable, non-obvious learning a FUTURE run on a \
    related change would want — an invariant, a decision + its rationale, a gotcha, a risk, or (if \
    the run hit a wall) what that wall was and what to watch for. Write it in the present tense: \
    what is true now and how to apply it, NOT a changelog of what this run did. Be specific and \
    grounded; a vague or obvious memory is worse than none. Choose a `kind` from rag-rat's \
    taxonomy: Invariant, Decision, RejectedAlternative, Risk, BugPattern, TestExpectation, \
    PerformanceNote, SecurityNote, FFIBoundary, PlatformQuirk, FollowUp, OpenQuestion, Concept.";

/// What the compose model produces.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryDraft {
    /// One of rag-rat's kinds; normalized to `Decision` if unrecognized.
    #[serde(default)]
    pub kind: String,
    pub title: String,
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
    pub memories_written: Vec<MemoryWritten>,
    pub iterations: u32,
    pub residual_risk_accepted: bool,
}

/// Everything the bookkeeper composes from.
#[derive(Debug, Clone)]
pub struct BookkeeperInput {
    pub issue: String,
    pub analyst: AnalystOutput,
    pub implementer: ImplementerOutput,
    pub iterations: u32,
    /// Whether the run converged. `false` means it exhausted its iteration budget with unresolved
    /// failures — the memory is framed as a wall hit and tagged `unresolved`.
    pub converged: bool,
}

/// The bookkeeper node. Holds a cheap model route, a small tool subset (for the compose agent), and
/// the sink (to call `memory_create`).
pub struct BookkeeperNode {
    pub route: ratatoskr_core::ModelRoute,
    pub tools: Vec<rmcp::model::Tool>,
    pub sink: ServerSink,
}

impl BookkeeperNode {
    pub async fn run(&self, input: BookkeeperInput) -> Result<BookkeeperOutput, NodeError> {
        let prompt = render_prompt(&input);
        let raw = ratatoskr_agent::run_structured(
            &self.route,
            PREAMBLE,
            &prompt,
            self.tools.clone(),
            self.sink.clone(),
            schemars::schema_for!(MemoryDraft),
        )
        .await
        .map_err(|e| NodeError::Failed(format!("bookkeeper compose failed: {e}")))?;

        let draft = parse_validated::<MemoryDraft>(&raw)?;
        let kind = normalize_kind(&draft.kind);
        let anchor = input.implementer.touched_files.first().cloned();

        // Tag unresolved (max-iterations) runs so they're distinguishable from success write-backs.
        let tags: &[&str] = if input.converged {
            &["ratatoskr", "bookkeeper"]
        } else {
            &["ratatoskr", "bookkeeper", "unresolved"]
        };

        let memory_id = self
            .create_memory(&kind, &draft.title, &draft.body, anchor.as_deref(), tags)
            .await?;

        Ok(BookkeeperOutput {
            memories_written: vec![MemoryWritten {
                kind,
                anchor: anchor.unwrap_or_default(),
                memory_id,
                summary: Some(draft.title),
            }],
            iterations: input.iterations,
            residual_risk_accepted: false,
        })
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
            let _ = writeln!(s, "- [{}] {}", r.severity, r.description);
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
    fn draft_parses_and_defaults_kind() {
        // Missing kind → default field, later normalized.
        let d = parse_validated::<MemoryDraft>(r#"{"title":"t","body":"b"}"#).unwrap();
        assert_eq!(d.title, "t");
        assert_eq!(normalize_kind(&d.kind), "Decision");
    }
}
