//! Memory corroboration: a direct `memory_search` call (no LLM), letting rag-rat rank the
//! repo memories relevant to the issue. This is the "graph enforces policy, LLM fills content
//! only where it's needed" principle made concrete — retrieval + ranking live in rag-rat.

use ratatoskr_core::RunState;
use ratatoskr_graph::{Node, NodeError, parse_validated};
use rmcp::model::CallToolRequestParams;
use rmcp::service::ServerSink;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Cap on the query text sent to `memory_search` (issue + scout context can be long).
const MAX_QUERY_CHARS: usize = 2000;
const MEMORY_LIMIT: u32 = 10;

/// Input to the memory node: the issue plus scout's papertrail summary, used as the search query.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryInput {
    pub issue: String,
    pub context: String,
}

/// One repo memory rag-rat's ranking returned. A lenient subset of rag-rat's `RepoMemory` — extra
/// fields (bindings, tags, ...) are allowed and ignored.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryRecord {
    pub memory_id: String,
    pub kind: String,
    pub title: String,
    pub confidence: String,
    pub status: String,
    /// Full body (present unless rag-rat is configured for summary surface).
    #[serde(default)]
    pub body: String,
    /// Compacted summary (present under summary surface).
    #[serde(default)]
    pub summary: Option<String>,
}

/// Memory node output: the ranked memories.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MemoryOutput {
    pub memories: Vec<MemoryRecord>,
}

/// The memory node: a direct rag-rat `memory_search`, no agent.
pub struct MemoryNode {
    pub sink: ServerSink,
}

impl Node for MemoryNode {
    type Input = MemoryInput;
    type Output = MemoryOutput;

    fn name(&self) -> &'static str {
        "memory"
    }

    async fn run(
        &self,
        input: MemoryInput,
        _run_state: &RunState,
    ) -> Result<MemoryOutput, NodeError> {
        search(&self.sink, &input.issue, &input.context).await
    }
}

/// Run rag-rat's ranked `memory_search` for `issue` (plus any `context` narrowing it).
///
/// Extracted from the node so the context node runs the identical retrieval: the guarantee that
/// matters is that the same deterministic search happens, not which node called it.
pub async fn search(
    sink: &ServerSink,
    issue: &str,
    context: &str,
) -> Result<MemoryOutput, NodeError> {
    let mut query = format!("{issue}\n{context}");
    query.truncate(floor_char_boundary(&query, MAX_QUERY_CHARS));

    let args = serde_json::json!({ "query": query, "limit": MEMORY_LIMIT })
        .as_object()
        .cloned()
        .expect("json object literal");
    let param = CallToolRequestParams::new("memory_search").with_arguments(args);

    let result = sink
        .call_tool(param)
        .await
        .map_err(|e| NodeError::Failed(format!("memory_search call failed: {e}")))?;

    let text = result
        .content
        .iter()
        .filter_map(|c| c.as_text())
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join("");

    if result.is_error.unwrap_or(false) {
        return Err(NodeError::Failed(format!(
            "memory_search returned an error: {text}"
        )));
    }

    parse_memory_result(&text)
}

/// Wrap `memory_search`'s bare JSON array as `{"memories": [...]}` and run it through the schema
/// gate. Kept separate so it's unit-testable without a live rag-rat.
fn parse_memory_result(array_json: &str) -> Result<MemoryOutput, NodeError> {
    let trimmed = array_json.trim();
    if trimmed.is_empty() {
        return Err(NodeError::Failed(
            "memory_search returned no content (is rag-rat running with `--json`?)".to_string(),
        ));
    }
    let wrapped = format!(r#"{{"memories":{trimmed}}}"#);
    parse_validated::<MemoryOutput>(&wrapped)
}

/// Largest `<= max` byte index that lands on a char boundary of `s`.
fn floor_char_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut i = max;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_canned_memory_array() {
        let raw = r#"[
            {"memory_id":"m1","kind":"Decision","title":"pin rmcp","confidence":"high",
             "status":"active","body":"because...","bindings":[],"tags":["deps"]}
        ]"#;
        let out = parse_memory_result(raw).unwrap();
        assert_eq!(out.memories.len(), 1);
        assert_eq!(out.memories[0].memory_id, "m1");
        assert_eq!(out.memories[0].kind, "Decision");
    }

    #[test]
    fn empty_array_is_ok() {
        assert!(parse_memory_result("[]").unwrap().memories.is_empty());
    }

    #[test]
    fn malformed_memory_trips_the_gate() {
        // Missing required `confidence`/`status`.
        let raw = r#"[{"memory_id":"m1","kind":"Decision","title":"t"}]"#;
        assert!(matches!(
            parse_memory_result(raw),
            Err(NodeError::InvalidOutput(_))
        ));
    }

    #[test]
    fn non_json_output_is_a_clear_error() {
        // e.g. TOON (rag-rat launched without --json) — not valid JSON.
        let err = parse_memory_result("memories[1]{id}: m1").unwrap_err();
        assert!(matches!(err, NodeError::InvalidOutput(_)));
    }
}
