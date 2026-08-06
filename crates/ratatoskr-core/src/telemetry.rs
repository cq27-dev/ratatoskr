//! What a node's turn cost and which model produced it.
//!
//! Lives here rather than in `ratatoskr-agent` because both ends need it and neither should depend
//! on the other: the agent crate produces it from provider-reported usage, and the store crate
//! persists it. Keeping it in the shared base also keeps `rig`'s types out of the store's public
//! API, so a `rig` upgrade cannot ripple into the schema.

use serde::{Deserialize, Serialize};

/// Provider-reported token counts for one node's turn, summed across every model call it made.
///
/// Always what the provider reported, never anything inferred from text length — a count we
/// estimated would be indistinguishable from a real one downstream, and cost caps built on it would
/// be wrong in the direction that costs money.
///
/// The cache figures are worth their columns: a cached prefix costs 1.25x to write and 0.1x to read,
/// so it only pays for itself on the second hit. A tool list that changes between converge
/// iterations invalidates the prefix silently, and `cache_creation_input_tokens` climbing on every
/// iteration is what that looks like from the outside.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

impl TokenUsage {
    /// Fold another turn's counts in. Every model call a node makes is billed, including one whose
    /// response is later retried, so accumulation is a plain sum with nothing filtered out.
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
    }
}

/// One node turn's measurements, recorded alongside the output it produced.
///
/// Every field is what actually happened rather than what was configured: `model` is the resolved
/// route, not the config alias that selected it, so a checkpoint stays readable after the alias is
/// repointed at a different model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeTelemetry {
    /// The resolved route as `provider/model`.
    pub model: Option<String>,
    /// Wall-clock time the node's turn took. The turn's start is `created_at - duration_ms`; it is
    /// not stored separately, because two columns that can disagree about the same instant
    /// eventually will.
    pub duration_ms: Option<u64>,
    pub usage: TokenUsage,
    /// How many model calls the turn took. A node that hit its turn cap looks identical to one that
    /// finished early in every other column.
    pub turns: Option<u64>,
    /// Why the node failed, when it did. A checkpoint is written for a failed node too — the reason
    /// it failed is the most useful thing about that row.
    pub error: Option<String>,
}
