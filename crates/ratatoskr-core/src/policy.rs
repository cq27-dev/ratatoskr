//! The tool-call policy seam: a decision a ruleset makes about a proposed tool call.
//!
//! Lives in `ratatoskr-core` (not `ratatoskr-agent`) deliberately: it references no `rig`/`rmcp`
//! type, so `ratatoskr-script` can implement it while depending only on this light crate — keeping
//! the whole agent/rig/rmcp stack out of the script crate's build graph. The trait is
//! dyn-compatible (a boxed future, not a native `async fn`) so it can be held as `Arc<dyn ToolPolicy>`.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

/// What a policy decides about a proposed tool call.
#[derive(Debug, Clone)]
pub enum ToolDecision {
    /// Run the tool with the current arguments.
    Allow,
    /// Do not run it; return this feedback to the model instead.
    Deny(String),
    /// Run it, but with these replacement arguments.
    Rewrite(Value),
}

/// A per-tool-call policy — the Rust side of a ruleset's `onToolCall` hook.
pub trait ToolPolicy: Send + Sync {
    /// Decide what to do with a proposed call to `tool_name` with `args_json` (the raw JSON args).
    fn decide<'a>(
        &'a self,
        tool_name: &'a str,
        args_json: &'a str,
    ) -> Pin<Box<dyn Future<Output = ToolDecision> + Send + 'a>>;
}
