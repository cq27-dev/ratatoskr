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
    /// Tokens spent on the model's own reasoning before it answered, when the endpoint reports
    /// them apart from the rest.
    ///
    /// `None` where the number was not measured, which is not the same statement as `Some(0)`.
    /// Anthropic bills thinking inside `output_tokens` and reports no separate figure at all, so a
    /// zero there was this endpoint's default rather than the model's answer — and a node that
    /// plainly thought reported `thinking_requested = true` beside `reasoning_tokens = 0` on every
    /// record that set both, two fields on one row asserting opposite things.
    ///
    /// Where it IS reported, it explains the wall-clock: billed as output and reported apart from
    /// it, so a node that thinks before every tool call looks nearly free when only `output_tokens`
    /// is read.
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

impl TokenUsage {
    /// Fold another turn's counts in. Every model call a node makes is billed, including one whose
    /// response is later retried, so accumulation is a plain sum with nothing filtered out.
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
        // Absence stays absent unless something real is folded in: summing an unmeasured turn as
        // zero would make a folded row claim a measurement none of its turns made.
        self.reasoning_tokens = match (self.reasoning_tokens, other.reasoning_tokens) {
            (Some(mine), Some(theirs)) => Some(mine + theirs),
            (mine, theirs) => mine.or(theirs),
        };
    }
}

/// One node turn's measurements, recorded alongside the output it produced.
///
/// Every field is what actually happened rather than what was configured: `model` is the resolved
/// route, not the config alias that selected it, so a checkpoint stays readable after the alias is
/// repointed at a different model.
///
/// A checkpoint may cover more than one turn — see [`NodeTelemetry::fold`] — in which case each
/// field is what that doc says it is across all of them.
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
    /// Why the node failed, when it did and the run carried on from it: the referee and the
    /// verifier both record a failure this way, and the reason is the most useful thing about that
    /// row.
    ///
    /// A node that fails OUTRIGHT writes no row at all — the run ends at it — so this is not where
    /// a failed run's cause is looked up. That lives in the event stream, which carries the turn's
    /// start and its cost record with the error on it whether or not a checkpoint follows.
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
    /// Whether the route left the node free to reason before answering — CONFIGURATION, which is
    /// why the name says so.
    ///
    /// It sat beside the measurements as `thinking`, and a reader had no way to tell that this one
    /// field describes what was asked for rather than what happened. `false` means the route
    /// disabled reasoning explicitly; `true` means it did not, which is not the same as "it
    /// happened": with thinking left alone the endpoint decides, and several turn it on as soon as
    /// a request carries tools. What the turn actually spent, where the endpoint says, is
    /// [`TokenUsage::reasoning_tokens`].
    #[serde(default)]
    pub thinking_requested: bool,
}

impl NodeTelemetry {
    /// Whether this record covers a model turn at all.
    ///
    /// The one answer to it, because the question is asked in several places and two spellings of
    /// it drift. A record written by an operation host — the aggregate under `redteam`,
    /// `implementer` or `context` — covers no turn of its own, and every cost field on it is a
    /// default rather than a measurement.
    ///
    /// Read off the route rather than the turn count: a turn that failed before completing a call
    /// still resolved a route and may still have been billed, and reporting nothing for it would
    /// lose real cost. [`Self::usage`] is only a claim when this is true.
    pub fn ran_a_model(&self) -> bool {
        self.model.is_some()
    }

    /// Fold another turn recorded under the same node into this one.
    ///
    /// One checkpoint can cover several model turns: the red team's classifier and its test author
    /// both run under `redteam`, and any node whose history is compacted charges the summarising
    /// turn to its own name. Keeping one of them and dropping the rest reported less than the run
    /// paid, and which one survived depended on the order they happened to finish in.
    ///
    /// The measurements sum. `duration_ms` sums with them, so on a folded row it is time spent on
    /// this node's model calls rather than a wall-clock span — two turns that ran concurrently
    /// spent both, and folded turns keep no start instant to reconstruct a span from.
    ///
    /// `model` names every distinct route it folded, comma-joined, rather than asserting one of
    /// them: the halves of a node resolve their route through their own agent profile, so they
    /// genuinely can differ. Emptying it on disagreement would be worse than the lie it avoids — a
    /// null model reads as "this node ran no model", and readers drop the whole row's cost with it.
    ///
    /// `tools` is the union, which is the node's reach across the turns and not what any single one
    /// was offered; per-turn fidelity needs per-stage records (#259/#260), not a different summary
    /// here. `error` keeps every distinct failure, because a half whose failure is swallowed as
    /// best-effort is exactly the one nothing else in the run records.
    pub fn fold(&mut self, other: NodeTelemetry) {
        self.usage.add(other.usage);
        self.turns = sum(self.turns, other.turns);
        self.duration_ms = sum(self.duration_ms, other.duration_ms);
        self.model = join(self.model.take(), other.model, ", ");
        self.error = join(self.error.take(), other.error, "; ");
        extend_distinct(&mut self.tools, other.tools);
        extend_distinct(&mut self.tools_used, other.tools_used);
        self.reuses_session |= other.reuses_session;
        self.thinking_requested |= other.thinking_requested;
    }
}

/// `None` only when neither turn reported the figure at all: a turn that reported nothing is not a
/// turn that reported zero, and one that did must not be erased by one that did not.
fn sum(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (None, None) => None,
        _ => Some(left.unwrap_or_default() + right.unwrap_or_default()),
    }
}

/// Append `incoming` unless the same value is already named, so folding two turns on one model
/// leaves one name rather than the same one twice.
fn join(existing: Option<String>, incoming: Option<String>, separator: &str) -> Option<String> {
    let Some(incoming) = incoming else {
        return existing;
    };
    let Some(existing) = existing else {
        return Some(incoming);
    };
    match existing.split(separator).any(|part| part == incoming) {
        true => Some(existing),
        false => Some(format!("{existing}{separator}{incoming}")),
    }
}

fn extend_distinct(into: &mut Vec<String>, from: Vec<String>) {
    for name in from {
        if !into.contains(&name) {
            into.push(name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(model: &str, input_tokens: u64) -> NodeTelemetry {
        NodeTelemetry {
            model: Some(model.to_string()),
            duration_ms: Some(100),
            usage: TokenUsage {
                input_tokens,
                ..Default::default()
            },
            turns: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn a_folded_row_accounts_for_every_turn_it_covers() {
        let mut folded = NodeTelemetry {
            tools: vec!["Read".to_string(), "Write".to_string()],
            error: Some("the author gave up".to_string()),
            thinking_requested: true,
            ..turn("anthropic/opus", 10)
        };
        folded.fold(NodeTelemetry {
            tools: vec!["Read".to_string(), "semantic_search".to_string()],
            ..turn("openai/gpt", 20)
        });

        assert_eq!(folded.usage.input_tokens, 30);
        assert_eq!(folded.turns, Some(2));
        assert_eq!(folded.duration_ms, Some(200));
        // Both routes named, because the two turns genuinely resolved different ones. Picking one
        // would depend on which finished first, and naming none takes the cost off the dashboard.
        assert_eq!(folded.model.as_deref(), Some("anthropic/opus, openai/gpt"));
        assert_eq!(folded.tools, ["Read", "Write", "semantic_search"]);
        // A best-effort half's failure survives its checkpoint; nothing else records it.
        assert_eq!(folded.error.as_deref(), Some("the author gave up"));
        assert!(folded.thinking_requested);
    }

    #[test]
    fn folding_two_turns_on_one_model_names_it_once() {
        let mut folded = turn("anthropic/opus", 10);
        folded.fold(turn("anthropic/opus", 5));
        assert_eq!(folded.model.as_deref(), Some("anthropic/opus"));
        assert_eq!(folded.usage.input_tokens, 15);
    }

    #[test]
    fn a_figure_nothing_reported_stays_unreported() {
        let mut folded = NodeTelemetry::default();
        folded.fold(NodeTelemetry::default());
        assert_eq!(folded.turns, None, "no turn reported a count");
        assert_eq!(folded.duration_ms, None);
        assert_eq!(folded.model, None);
    }
}
