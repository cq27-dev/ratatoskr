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
    /// Tokens spent on the model's own reasoning before it answered.
    ///
    /// Billed as output and reported apart from it, so a node that thinks before every tool call
    /// looks nearly free when only `output_tokens` is read — and thinking is the reason such a
    /// node's turns are slow, which makes this the number that explains the wall-clock.
    #[serde(default)]
    pub reasoning_tokens: u64,
}

impl TokenUsage {
    /// Fold another turn's counts in. Every model call a node makes is billed, including one whose
    /// response is later retried, so accumulation is a plain sum with nothing filtered out.
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
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
    /// The tools this node could call, by the names the model saw.
    ///
    /// Recorded rather than looked up later because it is a property of the run: rulesets, plugins
    /// and config all shape it, and the config that produced a past run may no longer exist.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Of those, the ones it actually called.
    ///
    /// Kept apart from `tools` rather than replacing it: what a node was *given* is a decision
    /// someone made, and what it *reached for* is what it did with that. A node handed a shell it
    /// never used is worth seeing.
    #[serde(default)]
    pub tools_used: Vec<String>,
    /// Whether this node's endpoint session carried over from an earlier attempt in the run.
    #[serde(default)]
    pub reuses_session: bool,
    /// Whether the node was left free to reason before answering.
    ///
    /// Recorded as configured rather than observed, because `usage.reasoning_tokens` comes back
    /// zero from endpoints that do not report it — and a node that plainly thought would then look
    /// like one that did not. `false` means the route disabled it explicitly; `true` means it was
    /// not disabled, which is not quite the same as "it happened": with thinking left alone, the
    /// endpoint decides, and several turn it on as soon as a request carries tools.
    #[serde(default)]
    pub thinking: bool,
}
