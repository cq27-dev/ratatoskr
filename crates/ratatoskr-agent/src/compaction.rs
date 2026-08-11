//! Keeping a long node conversation inside its context window.
//!
//! The machinery is `rig`'s: [`CompactingMemory`] runs a [`TokenWindowMemory`] policy and hands
//! whatever it evicts to a [`Compactor`], splicing the returned summary back at the front of the
//! history and carrying it forward into the next compaction. What is written here is the part rig
//! deliberately leaves open — what a good summary of a coding session actually contains.
//!
//! That shape matters more than it looks. A summary that reads like prose about what happened is
//! worse than useless to the turn that inherits it: the next tool call needs the exact path, the
//! exact identifier, the exact command. So the template asks for those verbatim and says why,
//! rather than asking for "a summary".

use std::sync::{Arc, Mutex};

use rig_agent::completion::Prompt;
use rig_core::completion::message::{ToolResult, ToolResultContent};
use rig_core::completion::{CompletionModel, Message};
use rig_core::memory::{Compactor, ConversationMemory, MemoryError};
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_memory::{InMemoryConversationMemory, TokenWindowMemory};

/// The history budget for a route that does not say how large its window is.
///
/// Conservative on purpose: it has to be safe on the smallest model anyone routes here, because
/// the cost of being wrong in the other direction is a context-length error mid-run rather than a
/// compaction. A route that says its window gets a budget scaled to it instead — see
/// [`budget_for`].
const HISTORY_TOKEN_BUDGET: usize = 120_000;

/// What share of a model's window a node's *history* may occupy.
///
/// The rest is not slack. A compaction happens by making a model call, so the window has to hold,
/// at that moment: everything kept, the summary prompt, every tool declaration, and the turn that
/// tripped the budget — which is routinely the largest of them, being a whole file read back or a
/// failing suite's output. Two thirds leaves that room at every window size, where a fixed reserve
/// would be most of a small window and a rounding error in a large one.
const HISTORY_SHARE: (usize, usize) = (2, 3);

/// Tokens of history a node on this route keeps before its oldest turns are summarised.
pub fn budget_for(window: Option<u64>) -> usize {
    let Some(window) = window else {
        return HISTORY_TOKEN_BUDGET;
    };
    let (num, den) = HISTORY_SHARE;
    // Saturating rather than wrapping: a window declared absurdly large is a config mistake, and
    // the budget it produces should stay a number rather than becoming a small one.
    usize::try_from(window)
        .unwrap_or(usize::MAX)
        .saturating_mul(num)
        / den
}

/// Characters per token, for the estimate that decides when to compact.
///
/// An approximation on purpose. The provider reports real usage only after a call, and a policy
/// that could only act on the previous turn's count would always be one turn late. Erring low means
/// compacting slightly early, which costs a summary; erring high means not compacting at all, which
/// costs the run.
const CHARS_PER_TOKEN: usize = 3;

/// The compaction instruction, minus the part that depends on which node is being compacted.
///
/// Written for this pipeline rather than for a chat assistant, and the difference is load-bearing:
/// every node here ends by filling a JSON schema, so the test for "did this summary keep enough" is
/// whether the node can still fill its schema from it — not whether it reads well.
const PREAMBLE: &str = include_str!("../prompts/compaction.md");

/// Most of one tool-result content block that the compactor sees.
///
/// A result can be a whole file or a noisy test suite, and carrying it without a limit turns the
/// summary request into the same context-length failure compaction is meant to avoid. Keep both
/// ends: reads identify their path and range at the front, while commands put their diagnostics and
/// exit summary at the end.
const MAX_TOOL_RESULT_CHARS: usize = 16 * 1024;

/// A run-local conversation that is reduced to one summary before its next attempt.
///
/// `rig` retains every tool call and result in its conversation memory. The normal compaction
/// budget is deliberately generous for a single long-running attempt, but a re-driven node needs
/// a small hand-off rather than the entire previous attempt. A zero-token window demotes the whole
/// completed attempt when the next one loads this memory, and the compactor replaces it with one
/// summary. The run ledger owns this state by stage/session key, so rebuilt node values continue
/// within one run while a later run cannot inherit it merely by using the same route.
#[derive(Clone, Default)]
pub struct CompactedSession {
    memory: Arc<Mutex<Option<Arc<dyn ConversationMemory>>>>,
}

impl CompactedSession {
    /// Return the one memory backend shared by every attempt of this node.
    pub(crate) fn memory<M>(
        &self,
        model: M,
        node: &str,
        produces: &str,
        ledger: Option<Arc<crate::RunLedger>>,
        provider_calls: crate::ProviderCallQueue,
    ) -> Arc<dyn ConversationMemory>
    where
        M: CompletionModel + 'static,
    {
        let mut slot = self
            .memory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(memory) = &*slot {
            return Arc::clone(memory);
        }

        // Each stored message costs at least one token, so this window carries no raw history into
        // a re-entry. CompactingMemory retains its rolling summary separately and includes it in
        // the following compaction, preserving facts from every earlier attempt.
        let memory: Arc<dyn ConversationMemory> = Arc::new(compacting_memory(
            model,
            node,
            produces,
            0,
            ledger,
            provider_calls,
        ));
        *slot = Some(Arc::clone(&memory));
        memory
    }

    #[cfg(test)]
    pub(crate) fn same_state_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.memory, &other.memory)
    }
}

/// Summarises evicted turns with a model call.
pub struct SummaryCompactor<M> {
    model: Arc<M>,
    /// Where a summarisation's cost is charged, and the node it is charged to.
    ///
    /// To the node that caused it, not to a line of its own: a compaction happens because *this*
    /// node's conversation outgrew its window, so its tokens are that node's. Reported separately,
    /// a run's per-node totals would sum to less than the invoice — and these are the calls that
    /// fire precisely when a session got long and expensive.
    ledger: Option<Arc<crate::RunLedger>>,
    provider_calls: crate::ProviderCallQueue,
    /// The node being compacted, and what it has to end up producing.
    ///
    /// Included in the instruction because "keep what matters" is unanswerable in the abstract: what
    /// a scout must retain to write a papertrail summary is not what an implementer must retain to
    /// finish an edit. Telling the summariser which one it is serving is the whole difference
    /// between a summary that reads well and one the node can still work from.
    node: String,
    produces: String,
}

impl<M> SummaryCompactor<M> {
    fn new(
        model: M,
        node: &str,
        produces: &str,
        ledger: Option<Arc<crate::RunLedger>>,
        provider_calls: crate::ProviderCallQueue,
    ) -> Self {
        SummaryCompactor {
            model: Arc::new(model),
            ledger,
            provider_calls,
            node: node.to_string(),
            produces: produces.to_string(),
        }
    }
}

impl<M> Compactor for SummaryCompactor<M>
where
    M: CompletionModel + 'static,
{
    type Artifact = Message;

    fn compact<'a>(
        &'a self,
        _conversation_id: &'a str,
        evicted: &'a [Message],
        carry_over: Option<&'a Message>,
    ) -> WasmBoxedFuture<'a, Result<Message, MemoryError>> {
        Box::pin(async move {
            let mut prompt = String::new();
            // The previous summary first: compaction is rolling, and a summary that dropped what an
            // earlier one established would lose it permanently — the turns it came from are gone.
            if let Some(previous) = carry_over {
                prompt.push_str(
                    "THE SUMMARY SO FAR — carry everything still true into your new one:\n",
                );
                prompt.push_str(&render(previous));
                prompt.push_str("\n\n");
            }
            // Fenced, and named as a transcript: without a hard boundary the rendered turns read
            // as an ongoing conversation, and a model answers them instead of summarising — the
            // observed failure was a tool call for a path it invented from the last line.
            prompt.push_str("=== BEGIN TRANSCRIPT TO SUMMARISE ===\n");
            for message in evicted {
                prompt.push_str(&render(message));
                prompt.push('\n');
            }
            prompt.push_str(
                "=== END TRANSCRIPT ===\n\nWrite the summary of the transcript above now.",
            );

            let (builder, meter) = crate::metered(
                (*self.model).clone(),
                &format!(
                    "{PREAMBLE}\n\nTHIS SESSION: the `{}` node. It must finish by producing: {}",
                    self.node, self.produces
                ),
                None,
                crate::Request::plain(),
                Arc::clone(&self.provider_calls),
            );
            let agent = builder.build();
            // The enclosing agent prompt owns the one no-verdict retry. Retrying here as well
            // would let two empty summaries trigger two compactions on each outer attempt.
            let answer = agent.prompt(prompt.as_str()).await;
            // Charged whether or not the summary came back: a compaction that failed still spent
            // what it spent, and dropping that would make the failure look free.
            if let Some(ledger) = &self.ledger {
                let (usage, calls) = meter.read();
                ledger.record(
                    &self.node,
                    ratatoskr_core::NodeTelemetry {
                        usage,
                        turns: Some(calls),
                        error: answer.as_ref().err().map(ToString::to_string),
                        ..Default::default()
                    },
                );
            }
            let summary = answer
                // `Backend` is the variant rig documents for a remote-LLM fault, and the adapter
                // propagates it unchanged.
                .map_err(|e| {
                    MemoryError::Backend(Box::new(std::io::Error::other(e.to_string())))
                })?;

            Ok(Message::assistant(summary))
        })
    }
}

/// A message as flat text, for handing to the summariser.
///
/// Tool calls and their results are included rather than dropped: what a node *did* and what it got
/// back is most of what the next turn needs, and a history of assistant prose with the tool calls
/// removed reads as a session that decided things for no reason.
fn render(message: &Message) -> String {
    use rig_core::completion::message::{AssistantContent, UserContent};
    match message {
        Message::System { content } => format!("system: {content}"),
        Message::User { content, .. } => {
            let parts: Vec<String> = content
                .iter()
                .map(|c| match c {
                    UserContent::Text(t) => t.text.clone(),
                    UserContent::ToolResult(result) => render_tool_result(result),
                    _ => "[attachment]".to_string(),
                })
                .collect();
            format!("user: {}", parts.join("\n"))
        }
        Message::Assistant { content, .. } => {
            let parts: Vec<String> = content
                .iter()
                .map(|c| match c {
                    AssistantContent::Text(t) => t.text.clone(),
                    AssistantContent::ToolCall(call) => format!(
                        "[called {} with {}]",
                        call.function.name, call.function.arguments
                    ),
                    _ => "[reasoning]".to_string(),
                })
                .collect();
            format!("assistant: {}", parts.join("\n"))
        }
    }
}

/// Render the usable parts of a tool result for the compaction transcript.
///
/// The agent's conversation memory holds typed tool output, not the MCP response, so both text and
/// JSON must be made explicit here. Images have no textual facts to preserve and are named rather
/// than inlining their base64 payload.
fn render_tool_result(result: &ToolResult) -> String {
    let content = result
        .content
        .iter()
        .map(|content| match content {
            ToolResultContent::Text(text) => bounded_tool_result(&text.text),
            ToolResultContent::Json { value } => bounded_tool_result(&value.to_string()),
            ToolResultContent::Image(_) => "[image result omitted]".to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("[result of {}]\n{content}", result.id)
}

/// Bound a tool result without dropping either the file/range prefix or a command's final error.
fn bounded_tool_result(content: &str) -> String {
    let count = content.chars().count();
    if count <= MAX_TOOL_RESULT_CHARS {
        return content.to_string();
    }

    let head_len = MAX_TOOL_RESULT_CHARS / 2;
    let tail_len = MAX_TOOL_RESULT_CHARS - head_len;
    let head: String = content.chars().take(head_len).collect();
    let tail: String = content.chars().skip(count - tail_len).collect();
    format!(
        "{head}\n[{} middle characters omitted]\n{tail}",
        count - MAX_TOOL_RESULT_CHARS
    )
}

/// Estimate a message's token cost from its rendered length.
fn tokens_in(message: &Message) -> usize {
    (render(message).len() / CHARS_PER_TOKEN).max(1)
}

/// History that summarises its oldest turns rather than dropping them.
///
/// Dropping is what a plain window does, and for a coding session it is the wrong trade: the turn
/// that discovered a constraint is exactly the one far enough back to be evicted, and losing it
/// means rediscovering it — or, worse, retrying the approach it ruled out.
pub(crate) fn compacting_memory<M>(
    model: M,
    node: &str,
    produces: &str,
    budget: usize,
    ledger: Option<Arc<crate::RunLedger>>,
    provider_calls: crate::ProviderCallQueue,
) -> rig_memory::CompactingMemory<InMemoryConversationMemory, TokenWindowMemory, SummaryCompactor<M>>
where
    M: CompletionModel + 'static,
{
    rig_memory::CompactingMemory::new(
        InMemoryConversationMemory::new(),
        TokenWindowMemory::new(budget, tokens_in),
        SummaryCompactor::new(model, node, produces, ledger, provider_calls),
    )
}

/// The default budget, for a caller with no route to ask.
pub fn default_budget() -> usize {
    HISTORY_TOKEN_BUDGET
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::OneOrMany;
    use rig_core::completion::message::{ToolResult, ToolResultContent, UserContent};

    #[test]
    fn compaction_leaves_retrying_to_the_enclosing_prompt() {
        let source = include_str!("compaction.rs");
        let retry_helper = ["retry_prompt", "_once"].concat();
        assert!(!source.contains(&retry_helper));
    }

    #[test]
    fn a_rendered_turn_keeps_the_tool_calls_and_their_results() {
        // A history rendered as prose with the tool calls stripped reads as a session that decided
        // things for no reason — and the summariser has nothing to preserve verbatim.
        let assistant = Message::assistant("I will look at the store.");
        assert!(render(&assistant).starts_with("assistant: "));
        assert!(render(&assistant).contains("look at the store"));

        let user = Message::user("read crates/ratatoskr-store/src/lib.rs");
        assert!(render(&user).contains("crates/ratatoskr-store/src/lib.rs"));
    }

    #[test]
    fn a_compacted_tool_result_keeps_bounded_text_and_json_content() {
        let result = Message::User {
            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                id: "call-read".into(),
                call_id: None,
                content: OneOrMany::many(vec![
                    ToolResultContent::text("crates/ratatoskr-store/src/lib.rs:42\nmissing column"),
                    ToolResultContent::json(serde_json::json!({
                        "test": "store::tests::migrates_existing_database",
                        "status": "failed",
                    })),
                ])
                .unwrap(),
            })),
        };

        let rendered = render(&result);
        assert!(rendered.contains("call-read"));
        assert!(rendered.contains("missing column"));
        assert!(rendered.contains("migrates_existing_database"));

        let long = format!(
            "crates/ratatoskr-store/src/lib.rs:1\n{}\nerror: migration test failed",
            "x".repeat(MAX_TOOL_RESULT_CHARS)
        );
        let bounded = bounded_tool_result(&long);
        assert!(bounded.contains("lib.rs:1"));
        assert!(bounded.contains("error: migration test failed"));
        assert!(bounded.contains("middle characters omitted"));
        assert!(bounded.chars().count() < long.chars().count());
    }

    #[test]
    fn the_token_estimate_never_reports_a_free_message() {
        // A zero would let an unbounded number of messages sit under any budget.
        assert!(tokens_in(&Message::user("")) >= 1);
        let long = Message::user("x".repeat(3_000));
        assert!(tokens_in(&long) >= 900, "{}", tokens_in(&long));
    }

    #[test]
    fn a_route_that_states_its_window_gets_a_budget_scaled_to_it() {
        // The defect this closes: every node compacted at a fixed 120k whatever it was running on,
        // so a large-window model summarised its history away long before it had to — paying for a
        // summary, and losing detail, to solve a problem it did not have.
        assert!(budget_for(Some(200_000)) > budget_for(None));
        assert!(budget_for(Some(1_000_000)) > budget_for(Some(200_000)));
        // And a window smaller than the fallback is respected rather than overrun.
        assert!(budget_for(Some(32_000)) < budget_for(None));
    }

    #[test]
    fn a_budget_always_leaves_the_window_room_to_compact_in() {
        // Compaction is itself a model call: what is kept, the summary prompt, every tool
        // declaration and the turn that tripped the budget all have to fit at that moment. A
        // budget at the window's edge produces a context-length error instead of a compaction,
        // which is the failure this whole module exists to avoid.
        for window in [8_000u64, 32_000, 128_000, 200_000, 1_000_000] {
            let budget = budget_for(Some(window));
            let spare = window as usize - budget;
            assert!(budget < window as usize, "{window}: {budget}");
            // Room for a large tool result on top of everything else, at every size.
            assert!(spare >= window as usize / 4, "{window}: only {spare} spare");
        }
    }

    #[test]
    fn an_unstated_window_keeps_the_budget_that_was_there_before() {
        // Most routes will never state one, and their behaviour must not move because this exists.
        assert_eq!(budget_for(None), 120_000);
        assert_eq!(budget_for(None), default_budget());
    }

    #[test]
    fn the_budget_leaves_room_for_the_call_that_triggers_compaction() {
        // Set at a model's window this would produce a context-length error instead of a
        // compaction: the summary call, the tool declarations and the triggering turn all have to
        // fit alongside what is kept.
        assert!(default_budget() < 200_000);
        assert!(default_budget() > 10_000);
    }

    /// The compactor against the real provider: it summarises, and it keeps the exact identifiers.
    ///
    /// Ignored by default — it spends money and needs `ANTHROPIC_API_KEY`. Worth its cost because
    /// the unit tests above only check the plumbing around the call. Whether a summary actually
    /// preserves a path verbatim is a property of the instruction, and an instruction is only
    /// verifiable by running it.
    #[tokio::test]
    #[ignore = "calls the Anthropic API; run with --ignored"]
    async fn a_summary_keeps_the_identifiers_it_was_told_to_keep() {
        use rig_core::client::{ProviderClient, completion::CompletionClient};

        let client = rig_core::providers::anthropic::Client::from_env().unwrap();
        let compactor = SummaryCompactor::new(
            client.completion_model("claude-haiku-4-5-20251001"),
            "analyst",
            "the requirements an implementation must satisfy",
            None,
            crate::ProviderCallQueue::default(),
        );

        let evicted = vec![
            Message::user("Plan the fix for the migration bug."),
            Message::assistant(
                "I read crates/ratatoskr-store/src/lib.rs. A repo memory says: adding a column \
                 needs an entry in both schema.sql and ADDED_COLUMNS; neither alone migrates an \
                 existing store. I tried putting it only in schema.sql and the existing-database \
                 test failed with `no such column: repo_sha`.",
            ),
        ];

        let summary = compactor.compact("c1", &evicted, None).await.unwrap();
        let text = render(&summary);

        // The three things the instruction singles out: the exact path, the constraint that was
        // looked up, and the approach that was tried and failed.
        assert!(text.contains("schema.sql"), "lost the path:\n{text}");
        assert!(
            text.contains("ADDED_COLUMNS"),
            "lost the identifier:\n{text}"
        );
        assert!(
            text.to_lowercase().contains("both") || text.contains("neither"),
            "lost the constraint:\n{text}"
        );

        // And a rolling compaction carries the earlier summary forward rather than replacing it.
        let second = compactor
            .compact(
                "c1",
                &[Message::assistant(
                    "Then I read crates/ratatoskr-core/src/config.rs.",
                )],
                Some(&summary),
            )
            .await
            .unwrap();
        let text = render(&second);
        assert!(text.contains("config.rs"), "lost the new turn:\n{text}");
        assert!(
            text.contains("ADDED_COLUMNS"),
            "dropped the carried summary:\n{text}"
        );
    }
}
