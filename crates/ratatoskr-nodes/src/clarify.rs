//! Node-to-node clarification (issue #5). A planning node's LLM can call the synthetic `ask` tool
//! (`ratatoskr_agent::ASK_TOOL_NAME`); `ratatoskr_agent`'s clarification hook routes the call here.
//! [`NodeClarifier::answer`] runs the target node ONCE against its checkpointed context and returns a
//! text answer, which the hook hands back as the tool's result — so the asking node's conversation
//! (and its prompt cache) continue in place, no re-run.
//!
//! Design notes: the answerer gets no `ask` tool, so recursion is impossible (nesting depth is always
//! 1). Answers are always text (a failure becomes best-effort guidance, never an error that breaks
//! the asker). A per-run [`ASK_BUDGET`] backstops a runaway asker. Every exchange is recorded for
//! `RunState.clarifications` and written as a `clarification` checkpoint (inert to replay, which is
//! name-keyed on the node checkpoints).

use std::fmt::Write as _;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ratatoskr_agent::Clarifier;
use ratatoskr_core::RatatoskrConfig;
use ratatoskr_mcp::ToolSet;
use ratatoskr_script::ScriptEngine;
use ratatoskr_store::Store;
use rmcp::model::Tool;
use serde_json::{Value, json};
use tracing::Instrument;

use crate::{checkpoint, stage_agent_config};

/// Total `ask` calls allowed per run — a runaway backstop. Answerers get no `ask` tool, so this is a
/// flat per-run budget, not a recursion depth.
const ASK_BUDGET: usize = 4;

/// Turn budget for an answer-mode invocation. Answer mode runs with NO tools (the context is already
/// in the prompt), so the model answers in one turn; the small ceiling only guards a pathology.
const ANSWER_MAX_TURNS: usize = 3;

/// Cap on how much of a target's prior checkpoint is fed back as answer context.
const CONTEXT_LIMIT: usize = 4000;

/// Hard ceiling on the request that waits for a human. The dashboard gives up first (its own
/// timeout is shorter); this only stops a wedged connection from blocking a node indefinitely,
/// and stays well under the prompt-cache TTL that the block-the-node design depends on.
const USER_ANSWER_CEILING: Duration = Duration::from_secs(150);

/// The synthetic `ask` tool declaration, injected into an asker node's tool list. Like the
/// structured-output tool, it's a system capability — not a rag-rat tool subject to a ruleset's
/// allow/deny (the clarification hook handles it before the ruleset hook sees it). A per-node opt-out
/// would be a config flag, not a tool deny.
pub(crate) fn ask_tool() -> Tool {
    let schema = json!({
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "enum": ["scout", "analyst", "bookkeeper", "redteam", "user"],
                "description": "Which node to ask. `user` reaches the human operator when one is \
                    watching the dashboard, and is answered by the `analyst` when nobody is — so \
                    prefer a peer node when one actually holds the answer."
            },
            "question": { "type": "string", "description": "A self-contained question." }
        },
        "required": ["to", "question"]
    });
    let mut tool = Tool::default();
    tool.name = ratatoskr_agent::ASK_TOOL_NAME.into();
    tool.description = Some(
        "Ask another planning node a question and receive its answer as this tool's result, without \
         ending your turn. Use only when you genuinely cannot proceed without information another \
         node holds."
            .into(),
    );
    tool.input_schema = Arc::new(
        schema
            .as_object()
            .cloned()
            .expect("schema literal is an object"),
    );
    tool
}

/// Runs the target node against its stored context to answer another node's `ask`. Built once per
/// run and `Arc`-shared into every asker node (via `run_structured`'s `clarifier` arg).
pub struct NodeClarifier {
    config: RatatoskrConfig,
    store: Store,
    engine: Arc<ScriptEngine>,
    run_id: String,
    issue: String,
    budget: AtomicUsize,
    recorded: Mutex<Vec<Value>>,
}

impl NodeClarifier {
    pub fn new(
        config: &RatatoskrConfig,
        store: &Store,
        engine: &Arc<ScriptEngine>,
        run_id: &str,
        issue: &str,
    ) -> Arc<Self> {
        Arc::new(Self {
            config: config.clone(),
            store: store.clone(),
            engine: Arc::clone(engine),
            run_id: run_id.to_string(),
            issue: issue.to_string(),
            budget: AtomicUsize::new(0),
            recorded: Mutex::new(Vec::new()),
        })
    }

    /// This clarifier as a trait object for a node's `clarifier` field (coerces via the return type;
    /// `Arc` unsizing can't be done with an `as` cast).
    pub fn as_dyn(self: &Arc<Self>) -> Arc<dyn Clarifier> {
        let clarifier: Arc<dyn Clarifier> = self.clone();
        clarifier
    }

    /// Take the recorded exchanges for `RunState.clarifications` (the caller drains once at the end;
    /// the clarifier can't reach the borrowed `RunState` itself).
    pub fn drain(&self) -> Vec<Value> {
        std::mem::take(&mut self.recorded.lock().unwrap())
    }

    async fn latest_output(&self, node: &str) -> Option<String> {
        let checkpoints = self.store.checkpoints_for_run(&self.run_id).await.ok()?;
        checkpoints
            .iter()
            .rev()
            .find(|c| c.node_name == node)
            .map(|c| c.output_json.clone())
    }

    async fn record(&self, from: &str, to: &str, question: &str, answer: &str) {
        let entry = json!({ "from": from, "to": to, "question": question, "answer": answer });
        self.recorded.lock().unwrap().push(entry.clone());
        // Durable trail; a failure to record must not break the asking node.
        if let Err(e) = checkpoint(&self.store, &self.run_id, "clarification", &entry).await {
            tracing::warn!("failed to checkpoint clarification: {e}");
        }
    }

    /// Offer the question to a human, if this run was started by a dashboard and somebody is
    /// watching it. `None` means nobody answered — for any reason — and the caller falls through
    /// to the node path, which is exactly what an unattended run does today.
    async fn ask_the_user(&self, from: &str, question: &str) -> Option<String> {
        let dashboard = std::env::var("RATATOSKR_DASHBOARD").ok()?;
        let question_id = uuid::Uuid::new_v4().to_string();

        // Emit before waiting: the dashboard learns about the question by tailing this, so it has
        // to be on disk before the request that blocks on an answer.
        tracing::info!(
            kind = "question",
            question_id,
            from,
            question,
            "waiting on the user"
        );

        let answer = self.await_user_answer(&dashboard, &question_id).await;

        // Always announce the outcome, including the ordinary one where nobody answered. The
        // dashboard clears its prompt on this event, so without it a viewer is left staring at a
        // question the run has long since moved past.
        tracing::info!(
            kind = "question_answered",
            question_id,
            answered = answer.is_some(),
            "question resolved"
        );
        answer
    }

    /// The blocking half. Any failure — unreachable dashboard, malformed reply, nobody watching,
    /// nobody typing — is the same `None`, because the caller's response to all of them is the
    /// same: fall through to the node path.
    async fn await_user_answer(&self, dashboard: &str, question_id: &str) -> Option<String> {
        let reply = reqwest::Client::new()
            .post(format!("{dashboard}/internal/clarifications"))
            .json(&serde_json::json!({
                "run_id": self.run_id,
                "question_id": question_id,
                // Empty unless the dashboard spawned this run, which is also the only case in
                // which it can be answered.
                "project": std::env::var("RATATOSKR_PROJECT").unwrap_or_default(),
            }))
            .timeout(USER_ANSWER_CEILING)
            .send()
            .await
            .ok()?
            .json::<serde_json::Value>()
            .await
            .ok()?;
        reply.get("answer")?.as_str().map(str::to_string)
    }

    async fn answer_inner(&self, from: &str, to: &str, question: &str) -> String {
        // A question addressed to the user goes to a human first; everything else keeps its
        // existing routing, because a peer node holds answers a person doesn't.
        if to.trim() == "user"
            && let Some(answer) = self.ask_the_user(from, question).await
        {
            return format!("Answer from the user:\n{answer}");
        }

        let (answerer, checkpoint_name) = resolve_target(to);

        let mut context = format!("ISSUE:\n{}\n", self.issue);
        if let Some(prior) = self.latest_output(checkpoint_name).await {
            let _ = write!(
                context,
                "\nYOUR PRIOR OUTPUT:\n{}\n",
                elide(&prior, CONTEXT_LIMIT)
            );
        }

        // Only the route matters here; answer mode runs with no tools. Label with the RESOLVED
        // answerer (not the raw `to`), so a fallback to analyst isn't misattributed.
        let mut plugins = crate::NodePlugins::default();
        let (route, system_prompt) = match stage_agent_config(
            &self.engine,
            &self.config,
            ToolSet::default(),
            answerer,
            &[],
            // Answer mode runs with no tools at all, so no skills either.
            &mut plugins,
        ) {
            Ok(cfg) => (cfg.route, cfg.system_prompt),
            Err(_) => {
                return format!(
                    "Could not reach `{answerer}`: no model route is configured for it. Proceed with \
                     your best assumption and flag it as a residual risk."
                );
            }
        };

        // A ruleset `systemPrompt` replaces the node's *persona*, but the answer-mode contract is
        // this call site's own and always applies — otherwise a scout-shaped prompt would make the
        // answerer go scout instead of answering.
        let persona = match (plugins.profile_prompt.as_str(), system_prompt) {
            ("", Some(prompt)) => prompt,
            ("", None) => format!("You are the {answerer} in a code-planning pipeline."),
            (profile, Some(prompt)) => format!("{profile}\n\n{prompt}"),
            (profile, None) => {
                format!("{profile}\n\nYou are the {answerer} in a code-planning pipeline.")
            }
        };
        let preamble = format!(
            "{persona}\n\nA peer node is asking you a question mid-run. Answer it concisely and \
             concretely from the context you are given; if you cannot answer from what you have, \
             say so plainly rather than guessing."
        );
        let prompt = format!("A peer node (`{from}`) asks:\n{question}\n\nContext:\n{context}");

        let span = tracing::info_span!("clarify", from, answerer);
        let body = match ratatoskr_agent::ask(
            &route,
            &preamble,
            &prompt,
            ToolSet::default(),
            Some(ANSWER_MAX_TURNS),
        )
        .instrument(span)
        .await
        {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("clarify: `{answerer}` could not answer: {e}");
                format!("could not answer ({e}); proceed with your best assumption")
            }
        };
        format!("Answer from `{answerer}`:\n{body}")
    }
}

impl Clarifier for NodeClarifier {
    fn answer<'a>(
        &'a self,
        from: &'a str,
        to: &'a str,
        question: &'a str,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
        Box::pin(async move {
            // Charge the budget; the exhausted case is still recorded (every exchange is).
            let answer = if self.budget.fetch_add(1, Ordering::SeqCst) >= ASK_BUDGET {
                "The clarification budget for this run is exhausted. Proceed with your best \
                 assumption and note it as a residual risk."
                    .to_string()
            } else {
                self.answer_inner(from, to, question).await
            };
            self.record(from, to, question, &answer).await;
            answer
        })
    }
}

/// Map an `ask` target to (answerer node, its checkpoint name). `analyst` is the fallback for the
/// user, unknown targets, and empty — it can answer from the issue alone. Note the `redteam` →
/// `red_team` checkpoint-name mismatch.
fn resolve_target(to: &str) -> (&'static str, &'static str) {
    match to.trim() {
        "scout" => ("scout", "scout"),
        "bookkeeper" => ("bookkeeper", "bookkeeper"),
        "redteam" | "red_team" => ("redteam", "red_team"),
        _ => ("analyst", "analyst"),
    }
}

/// Trim `s` to `max` chars for prompt context, with an ellipsis when cut.
fn elide(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_target_maps_names_and_falls_back_to_analyst() {
        assert_eq!(resolve_target("scout"), ("scout", "scout"));
        assert_eq!(resolve_target("redteam"), ("redteam", "red_team"));
        assert_eq!(resolve_target("bookkeeper"), ("bookkeeper", "bookkeeper"));
        // user / unknown / empty → analyst fallback.
        assert_eq!(resolve_target("user"), ("analyst", "analyst"));
        assert_eq!(resolve_target("implementer"), ("analyst", "analyst"));
        assert_eq!(resolve_target(""), ("analyst", "analyst"));
    }

    #[test]
    fn ask_tool_is_named_and_schema_shaped() {
        let t = ask_tool();
        assert_eq!(t.name, ratatoskr_agent::ASK_TOOL_NAME);
        assert!(t.input_schema.contains_key("properties"));
    }
}
