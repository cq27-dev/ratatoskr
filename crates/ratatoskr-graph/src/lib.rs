//! The graph vocabulary: the [`Node`] trait and the [`Edge`] type.
//!
//! Phase 0 fixes the *shape* only — there is no executor that sequences nodes (that's Phase 2).
//! The trait deliberately uses native `async fn` and is **not** made object-safe yet: a
//! `dyn`-compatible, heterogeneous node registry only makes sense once a real executor exists to
//! validate the shape against, and guessing it now risks getting it wrong.

use ratatoskr_core::RunState;
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Error produced by a node.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    /// The node's own work failed (a tool call, an agent turn, a client error).
    #[error("{0}")]
    Failed(String),
    /// The node produced output that didn't parse or didn't match its schema — the
    /// [`parse_validated`] gate rejected it, so the run must stop rather than accept it.
    #[error("output failed schema validation: {0}")]
    InvalidOutput(String),
}

/// The schema-validation gate: parse best-effort model/tool output into a typed `T`, rejecting
/// anything that doesn't match `T`'s JSON Schema. This is the enforcement behind "structured JSON
/// handoffs, not chat transcripts" — call it on every node's raw output before accepting it into
/// `RunState`.
///
/// `raw` may be wrapped in prose or ```json fences (agents in `OutputMode::Tool` are *instructed*
/// but not *forced* to emit clean JSON), so the object between the outermost braces is extracted
/// first.
pub fn parse_validated<T>(raw: &str) -> Result<T, NodeError>
where
    T: DeserializeOwned + JsonSchema,
{
    let json = extract_json_object(raw).ok_or_else(|| {
        NodeError::InvalidOutput(format!("no JSON object found in output: {}", elide(raw)))
    })?;
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| NodeError::InvalidOutput(format!("output is not valid JSON: {e}")))?;

    let schema = serde_json::to_value(schemars::schema_for!(T))
        .map_err(|e| NodeError::InvalidOutput(format!("could not build schema: {e}")))?;
    let validator = jsonschema::validator_for(&schema)
        .map_err(|e| NodeError::InvalidOutput(format!("could not compile schema: {e}")))?;
    if let Err(err) = validator.validate(&value) {
        return Err(NodeError::InvalidOutput(format!(
            "{} does not match its schema: {err}",
            std::any::type_name::<T>()
        )));
    }

    serde_json::from_value(value)
        .map_err(|e| NodeError::InvalidOutput(format!("could not deserialize: {e}")))
}

/// Return the substring from the first `{` to the last `}`, or `None` if there isn't a brace pair.
fn extract_json_object(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    (end > start).then(|| &raw[start..=end])
}

fn elide(s: &str) -> String {
    let s = s.trim();
    if s.chars().count() <= 200 {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(200).collect::<String>())
    }
}

/// A unit of work in the graph. Its `Input`/`Output` are schema-bearing so the executor can, in
/// Phase 2, check that one node's output lines up with the next node's input before wiring them.
///
/// `run_state` is read-only context: the executor (not the node) owns writing a node's output
/// back into state and persisting the checkpoint.
pub trait Node {
    type Input: Serialize + DeserializeOwned + JsonSchema;
    type Output: Serialize + DeserializeOwned + JsonSchema;

    fn name(&self) -> &'static str;

    // Native `async fn` in a trait (see module docs). The `async_fn_in_trait` lint warns that
    // callers can't add their own `Send` bound on the returned future; that's fine until Phase 2's
    // executor needs one, at which point this becomes an explicit `-> impl Future + Send`.
    #[allow(async_fn_in_trait)]
    async fn run(
        &self,
        input: Self::Input,
        run_state: &RunState,
    ) -> Result<Self::Output, NodeError>;
}

/// A directed edge between two nodes, keyed by [`Node::name`]. Just enough for Phase 2's executor
/// to validate a wiring; no weight, no condition yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub from: &'static str,
    pub to: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct Message {
        text: String,
    }

    /// A node that echoes its input — exists only to prove the trait shape compiles and runs.
    struct EchoNode;

    impl Node for EchoNode {
        type Input = Message;
        type Output = Message;

        fn name(&self) -> &'static str {
            "echo"
        }

        async fn run(
            &self,
            input: Self::Input,
            _run_state: &RunState,
        ) -> Result<Self::Output, NodeError> {
            Ok(input)
        }
    }

    #[tokio::test]
    async fn echo_node_returns_its_input() {
        let node = EchoNode;
        let state = RunState::new("run-1", None);
        let out = node
            .run(
                Message {
                    text: "hello".to_string(),
                },
                &state,
            )
            .await
            .unwrap();
        assert_eq!(node.name(), "echo");
        assert_eq!(
            out,
            Message {
                text: "hello".to_string()
            }
        );
    }

    #[test]
    fn edge_is_a_plain_pair() {
        let e = Edge {
            from: "scout",
            to: "analyst",
        };
        assert_eq!(e.from, "scout");
        assert_eq!(e.to, "analyst");
    }

    #[test]
    fn parse_validated_accepts_good_output_even_wrapped_in_prose() {
        let ok: Message = parse_validated(r#"{"text":"hi"}"#).unwrap();
        assert_eq!(ok.text, "hi");

        // OutputMode::Tool is best-effort; tolerate fences/prose around the object.
        let wrapped: Message =
            parse_validated("Here you go:\n```json\n{\"text\":\"hi\"}\n```").unwrap();
        assert_eq!(wrapped.text, "hi");
    }

    #[test]
    fn parse_validated_rejects_schema_violations() {
        // Missing the required `text` field → InvalidOutput, not a silent default.
        let err = parse_validated::<Message>(r#"{"nope":1}"#).unwrap_err();
        assert!(matches!(err, NodeError::InvalidOutput(_)));

        // Not JSON at all.
        let err = parse_validated::<Message>("total garbage").unwrap_err();
        assert!(matches!(err, NodeError::InvalidOutput(_)));
    }
}
