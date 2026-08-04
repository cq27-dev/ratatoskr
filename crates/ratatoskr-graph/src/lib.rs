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

/// Error produced by a node. Kept minimal in Phase 0; nodes refine it as they're built.
#[derive(Debug, thiserror::Error)]
#[error("node error: {0}")]
pub struct NodeError(pub String);

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
}
