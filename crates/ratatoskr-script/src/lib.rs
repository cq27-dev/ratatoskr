//! TypeScript scripting for Ratatoskr: transpile (swc) + evaluate (rquickjs) `.ratatoskr/rules/*.ts`
//! agent rulesets, exposing a per-tool-call [`ratatoskr_core::ToolPolicy`] backed by each ruleset's
//! `onToolCall` hook, plus its static config (model / tools / maxTurns).
//!
//! Depends only on `ratatoskr-core` — no `rig`/`rmcp` — so embedding a JS engine doesn't pull the
//! whole agent stack into this crate's build graph.

pub mod ruleset;
pub mod transpile;

pub use ruleset::{AgentRuleset, ModelRule, NodeRuleset, ScriptEngine, ToolRule};

/// Errors loading or evaluating ruleset scripts.
#[derive(Debug, thiserror::Error)]
pub enum ScriptError {
    #[error("io error at {0}: {1}")]
    Io(String, std::io::Error),
    #[error("transpile error: {0}")]
    Transpile(String),
    #[error("script eval error: {0}")]
    Eval(String),
}
