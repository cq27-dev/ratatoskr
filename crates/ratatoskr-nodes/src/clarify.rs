//! Node-to-node clarification (issue #5). A planning node's LLM can call the synthetic `ask` tool
//! (`ratatoskr_agent::ASK_TOOL_NAME`); `ratatoskr_agent`'s clarification hook routes the call here.
//! [`NodeClarifier::answer`] runs the target node ONCE against its checkpointed context and returns a
//! text answer, which the hook hands back as the tool's result — so the asking node's conversation
//! (and its prompt cache) continue in place, no re-run. An operator Stop is the sole exception: it
//! terminates the asking turn instead of becoming a synthetic answer.
//!
//! Design notes: the answerer gets no `ask` tool, so recursion is impossible (nesting depth is always
//! 1). Ordinary failures become best-effort guidance, never an error that breaks the asker. A
//! per-run [`ASK_BUDGET`] backstops a runaway asker. Completed exchanges are recorded for
//! `RunState.clarifications` and written as a `clarification` checkpoint (inert to replay, which is
//! name-keyed on the node checkpoints).

use std::fmt::Write as _;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ratatoskr_agent::{ClarificationAnswer, Clarifier, RuntimeControl};
use ratatoskr_core::RatatoskrConfig;
use ratatoskr_mcp::ToolSet;
use ratatoskr_script::ScriptEngine;
use ratatoskr_store::Store;
use rmcp::model::Tool;
use serde_json::{Value, json};
use tracing::Instrument;

use crate::checkpoint;

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

enum UserAnswer {
    Text(String),
    Unavailable,
    Stopped,
}

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
                "enum": ["context", "analyst", "bookkeeper", "redteam", "user"],
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
    /// The registry this run executes, shared with its `WorkflowContext`. An answerer is the stage
    /// the run would run, not the compiled-in stage that happens to share its name.
    stages: crate::workflow::ExecutionStages,
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
        stages: crate::workflow::ExecutionStages,
    ) -> Arc<Self> {
        Arc::new(Self {
            config: config.clone(),
            store: store.clone(),
            engine: Arc::clone(engine),
            run_id: run_id.to_string(),
            issue: issue.to_string(),
            stages,
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
    async fn ask_the_user(
        &self,
        from: &str,
        question: &str,
        control: Option<&RuntimeControl>,
    ) -> UserAnswer {
        let Ok(dashboard) = std::env::var("RATATOSKR_DASHBOARD") else {
            return UserAnswer::Unavailable;
        };
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

        let controlled_answer = match control {
            Some(control) => {
                control
                    .wait_for_stop_or(self.await_user_answer(&dashboard, &question_id))
                    .await
            }
            None => Some(self.await_user_answer(&dashboard, &question_id).await),
        };
        let stopped = controlled_answer.is_none();
        let answer = controlled_answer.flatten();

        // Always announce the outcome, including the ordinary one where nobody answered. The
        // dashboard clears its prompt on this event, so without it a viewer is left staring at a
        // question the run has long since moved past.
        tracing::info!(
            kind = "question_answered",
            question_id,
            answered = answer.is_some(),
            "question resolved"
        );
        if stopped {
            UserAnswer::Stopped
        } else if let Some(answer) = answer {
            UserAnswer::Text(answer)
        } else {
            UserAnswer::Unavailable
        }
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

    /// The turn settings the run itself would give `answerer` — resolved through the run's registry
    /// by the same chain `node_route` uses, so an answer speaks on the route its stage executes on.
    ///
    /// `None` when the run has no stage for that role, or none of them has a model route.
    async fn answerer_agent(&self, answerer: &str) -> Option<(crate::NodeAgentConfig, String)> {
        // `resolve_target` maps the asker's word to a ROLE; the registry maps that role to the
        // stage that plays it — by id, then by `governedBy`, the precedence `node_route` uses. So
        // an overridden `analyst` answers under its own governance identity rather than under the
        // name it was asked by.
        //
        // Several stages may govern as one name, and they route separately: `redteam` is the
        // classifier on the `reason` profile and the author on `build`, and a run may route either
        // half alone. The one that can answer is the one with somewhere to run, so this takes the
        // first candidate that *resolves* rather than the first that matches — otherwise a run with
        // a red team is told its red team does not exist because the other half is declared first.
        let stages = crate::workflow::execution_stages(&self.stages).await.ok()?;
        let by_id = stages.iter().filter(|stage| stage.id == answerer);
        let by_governance = stages
            .iter()
            .filter(|stage| stage.id != answerer && stage.governance_id() == answerer);
        by_id.chain(by_governance).find_map(|stage| {
            let (cfg, profile) = crate::plugins::declared_stage_agent_config(
                &self.engine,
                &self.config,
                ToolSet::default(),
                stage,
                // Answer mode runs with no tools at all, so no skills either.
                &[],
                &crate::NodePlugins::default(),
                ratatoskr_core::Capability::Read,
            )
            .ok()?;
            Some((cfg, profile.base_prompt))
        })
    }

    async fn answer_inner(
        &self,
        from: &str,
        to: &str,
        question: &str,
        control: Option<RuntimeControl>,
    ) -> ClarificationAnswer {
        // A question addressed to the user goes to a human first; everything else keeps its
        // existing routing, because a peer node holds answers a person doesn't.
        if to.trim() == "user" {
            match self.ask_the_user(from, question, control.as_ref()).await {
                UserAnswer::Text(answer) => {
                    return ClarificationAnswer::Text(format!("Answer from the user:\n{answer}"));
                }
                UserAnswer::Stopped => return ClarificationAnswer::Stopped,
                UserAnswer::Unavailable => {}
            }
        }

        let answerer = resolve_target(to);

        let mut context = format!("ISSUE:\n{}\n", self.issue);
        // "That answerer's prior output" is the record under the answerer's own name — the BOX's
        // record, when the box is composed of stages. `ask("redteam", ..)` gets the red team's
        // aggregate: its classification and its authored tests together, which is the red team's
        // answer to what it was asked to do. The halves' own rows exist beside it and are
        // deliberately not read here — one of them is half an answer, and which half depends on
        // which ran last, so an asker would get the classifier's verdict or the author's file list
        // depending on timing. A question addressed to a name is answered by what that name
        // produced.
        if let Some(prior) = self.latest_output(answerer).await {
            let _ = write!(
                context,
                "\nYOUR PRIOR OUTPUT:\n{}\n",
                elide(&prior, CONTEXT_LIMIT)
            );
        }

        // Only the route matters here; answer mode runs with no tools. Label with the RESOLVED
        // answerer (not the raw `to`), so a fallback to analyst isn't misattributed.
        let mut plugins = crate::NodePlugins::default();
        let (route, system_prompt) = match self.answerer_agent(answerer).await {
            Some((cfg, profile_prompt)) => {
                plugins.profile_prompt = profile_prompt;
                (cfg.route, cfg.system_prompt)
            }
            None => {
                return ClarificationAnswer::Text(format!(
                    "Could not reach `{answerer}`: no model route is configured for it. Proceed with \
                     your best assumption and flag it as a residual risk."
                ));
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
        let answer = ratatoskr_agent::ask(
            &route,
            &preamble,
            &prompt,
            ToolSet::default(),
            Some(ANSWER_MAX_TURNS),
            control.clone(),
        )
        .instrument(span);
        let response = match control.as_ref() {
            Some(control) => match control.wait_for_stop_or(answer).await {
                Some(response) => response,
                None => return ClarificationAnswer::Stopped,
            },
            None => answer.await,
        };
        let stopped = control.as_ref().is_some_and(RuntimeControl::is_stopped);
        answer_after_model(answerer, stopped, response)
    }
}

/// Keep ordinary clarification failures as guidance, but let an operator Stop terminate the
/// asker's turn before it sends a further provider request.
fn answer_after_model(
    answerer: &str,
    stopped: bool,
    response: Result<String, ratatoskr_agent::AgentError>,
) -> ClarificationAnswer {
    if stopped {
        return ClarificationAnswer::Stopped;
    }
    let body = match response {
        Ok(text) => text,
        Err(e) => {
            tracing::warn!("clarify: `{answerer}` could not answer: {e}");
            format!("could not answer ({e}); proceed with your best assumption")
        }
    };
    ClarificationAnswer::Text(format!("Answer from `{answerer}`:\n{body}"))
}

impl Clarifier for NodeClarifier {
    fn answer<'a>(
        &'a self,
        from: &'a str,
        to: &'a str,
        question: &'a str,
        control: Option<RuntimeControl>,
    ) -> Pin<Box<dyn Future<Output = ClarificationAnswer> + Send + 'a>> {
        Box::pin(async move {
            // Charge the budget; each completed exchange, including an exhausted one, is recorded.
            let answer = if self.budget.fetch_add(1, Ordering::SeqCst) >= ASK_BUDGET {
                ClarificationAnswer::Text(
                    "The clarification budget for this run is exhausted. Proceed with your best \
                     assumption and note it as a residual risk."
                        .to_string(),
                )
            } else {
                self.answer_inner(from, to, question, control).await
            };
            if let ClarificationAnswer::Text(answer) = &answer {
                self.record(from, to, question, answer).await;
            }
            answer
        })
    }
}

/// Map an `ask` target to the node that answers it, which is also the name its prior output is
/// checkpointed under. `analyst` is the fallback for the user, unknown targets, and empty — it can
/// answer from the issue alone.
fn resolve_target(to: &str) -> &'static str {
    match to.trim() {
        // `context`, not `scout`: what the scout used to produce is a field of the context box's
        // record now, and the box is what answers. Offering a target the run has no stage for left
        // the model addressing something that resolves to nothing.
        "context" => "context",
        "bookkeeper" => "bookkeeper",
        "redteam" => "redteam",
        _ => "analyst",
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
        assert_eq!(resolve_target("context"), "context");
        // The name it replaced falls through to the analyst like any other unknown.
        assert_eq!(resolve_target("scout"), "analyst");
        assert_eq!(resolve_target("redteam"), "redteam");
        assert_eq!(resolve_target("bookkeeper"), "bookkeeper");
        // user / unknown / empty → analyst fallback.
        assert_eq!(resolve_target("user"), "analyst");
        assert_eq!(resolve_target("implementer"), "analyst");
        assert_eq!(resolve_target(""), "analyst");
    }

    /// An `ask` must be answered by the stage the run would run, on the route that run gave it.
    /// Resolving the answerer out of the compiled-in stage table instead means one run routes the
    /// analyst two ways: the executor through the overlaid registry and its governance id, the
    /// clarifier through a map the workflow never touched.
    #[tokio::test]
    async fn the_clarifier_reaches_the_analyst_the_run_routes() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-clarifier-registry-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();

        let mut config = ratatoskr_core::RatatoskrConfig::default();
        config.models.insert(
            "planning".to_string(),
            ratatoskr_core::ModelRoute {
                provider: "openai".to_string(),
                model: "gpt-5".to_string(),
                max_tokens: None,
                context_window: None,
                temperature: None,
                params: None,
                session: Default::default(),
            },
        );

        // What the run executes: the analyst, overridden to govern as `planning`.
        let mut stages = crate::workflow::standard_stages().await.unwrap();
        let analyst = stages
            .iter_mut()
            .find(|stage| stage.id == "analyst")
            .expect("the standard registry declares the analyst");
        analyst.governed_by = Some("planning".to_string());
        let registry: crate::workflow::ExecutionStages = Arc::default();
        registry.set(Arc::new(stages)).unwrap();

        let clarifier = NodeClarifier::new(
            &config,
            &store,
            &engine,
            "run-clarify",
            "an issue",
            registry,
        );
        let (cfg, _) = clarifier
            .answerer_agent("analyst")
            .await
            .expect("the run's analyst has a route");
        assert_eq!(cfg.route.model, "gpt-5");

        let _ = std::fs::remove_dir_all(dir);
    }

    /// `redteam` is one advertised answerer over two stages that route separately: the classifier
    /// on the `reason` profile, the author on `build`. A run that routes only the author still has
    /// a red team, and an implementer asking it must reach the half that can answer rather than
    /// the half that happens to be declared first.
    #[tokio::test]
    async fn the_clarifier_reaches_the_red_team_half_the_run_can_route() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-clarifier-redteam-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = ScriptEngine::load(&dir).await.unwrap();
        let store = Store::open_in_memory().unwrap();

        // Only the author's profile is routed: no `[models.redteam]`, nothing on `reason`.
        let mut config = ratatoskr_core::RatatoskrConfig::default();
        config.agents.insert(
            "build".to_string(),
            ratatoskr_core::AgentProfileConfig {
                model: Some(ratatoskr_core::ModelRoute {
                    provider: "openai".to_string(),
                    model: "gpt-5".to_string(),
                    max_tokens: None,
                    context_window: None,
                    temperature: None,
                    params: None,
                    session: Default::default(),
                }),
                base_prompt: String::new(),
                capabilities: vec![ratatoskr_core::Capability::Write],
                tool_policy: None,
                max_turns: None,
            },
        );

        let stages = crate::workflow::standard_stages().await.unwrap();
        let registry: crate::workflow::ExecutionStages = Arc::default();
        registry.set(Arc::new(stages)).unwrap();

        let clarifier = NodeClarifier::new(
            &config,
            &store,
            &engine,
            "run-clarify-redteam",
            "an issue",
            registry,
        );
        let (cfg, _) = clarifier
            .answerer_agent("redteam")
            .await
            .expect("a routed red-team half answers for `redteam`");
        assert_eq!(cfg.route.model, "gpt-5");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ask_tool_is_named_and_schema_shaped() {
        let t = ask_tool();
        assert_eq!(t.name, ratatoskr_agent::ASK_TOOL_NAME);
        assert!(t.input_schema.contains_key("properties"));
    }

    #[test]
    fn a_stopped_nested_answer_terminates_the_asker() {
        let answer = answer_after_model(
            "analyst",
            true,
            Err(ratatoskr_agent::AgentError::Prompt(
                "the operator stopped this node".to_string(),
            )),
        );
        assert_eq!(answer, ClarificationAnswer::Stopped);
    }
}
