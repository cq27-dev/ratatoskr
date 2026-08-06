//! Builds a `rig` agent bound to a model and rag-rat's MCP tools, and runs one prompt.
//!
//! Phase 1 has exactly one caller (`ratatoskr ask`), so provider resolution is a small `match`
//! rather than a registry. The agent's own multi-turn loop (from `rig-agent`) does the tool
//! calling — we hand it the tools and a client handle via `.rmcp_tools()`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use ratatoskr_core::{ModelRoute, ToolDecision, ToolPolicy};
use ratatoskr_mcp::ToolSet;
use rig_agent::AgentBuilder;
use rig_agent::agent::{
    Agent, AgentHook, HookContext, ModelTurnAction, ModelTurnFinished, NoToolConfig, OutputMode,
    ToolCall, ToolCallAction, ToolResultAction, ToolResultEvent, WithBuilderTools,
};
use rig_agent::completion::Prompt;
use rig_agent::tool::{DynamicTool, ToolExecutionError};
use rig_core::OneOrMany;
use rig_core::client::completion::CompletionClient;
use rig_core::client::{ProviderClient, ProviderClientError};
use rig_core::completion::CompletionModel;
use rig_core::message::{AssistantContent, ImageMediaType, MimeType, ToolResultContent};
use rig_core::providers::{anthropic, moonshot};
use rig_core::tool::ToolOutput;
use rmcp::model::{CallToolRequestParams, ContentBlock, Tool};
use rmcp::service::ServerSink;
use tracing::Instrument;

/// How many tool-calling turns the agent may take before it must produce a final answer. A node
/// that does real work (the analyst's impact analysis walks callers/callees/tests across the graph)
/// needs a generous budget; 10 was a toy limit that tripped `MaxTurns` mid-analysis. Overridable
/// per node via a ruleset's `maxTurns`.
const DEFAULT_MAX_TURNS: usize = 100;

/// The bound rig's own MCP adapter applies to a call, matched here so a renamed tool is not the
/// one that can hang forever on a response the transport lost.
const MCP_CALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// rig-agent's default name for the synthetic structured-output tool (`OutputMode::Tool`). Kept in
/// sync with `rig_agent`'s `DEFAULT_OUTPUT_TOOL_NAME`; a ruleset must not be able to deny it.
const OUTPUT_TOOL_NAME: &str = "final_result";

/// The synthetic tool a node calls to ask another node a question. Intercepted by
/// [`ClarificationHook`] and answered in-conversation; it is never dispatched to the rag-rat sink.
pub const ASK_TOOL_NAME: &str = "ask";

/// The synthetic tool a node calls to load a skill's instructions. Named as the plugin format
/// names it, so a skill written for that host is invoked here the same way.
pub const SKILL_TOOL_NAME: &str = "Skill";

/// A skill a node may load, as the agent needs it: the name it is asked for by, and the
/// instructions handed back.
///
/// The *description* is deliberately not here. It belongs to the tool's schema, because it is what
/// the model reads to choose; this is what it reads once it has chosen.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub body: String,
}

/// Answers a node's `ask` call by running the target node against its stored context (implemented in
/// `ratatoskr-nodes`). Lives here so [`ClarificationHook`] can hold it without a dependency cycle.
/// Always yields text — a failure to answer becomes best-effort guidance, never an error that breaks
/// the asking node's turn loop.
pub trait Clarifier: Send + Sync {
    fn answer<'a>(
        &'a self,
        from: &'a str,
        to: &'a str,
        question: &'a str,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>>;
}

/// Runs a node's tool calls past the hooks its plugins register (implemented in
/// `ratatoskr-nodes`, which is where a node's plugin bindings are known).
///
/// Both sides answer with context for the model and nothing else: whether a call proceeds is a
/// ruleset's `onToolCall` decision, not a plugin's. Neither may fail — a hook that breaks
/// contributes no text and the call is unaffected.
pub trait ToolObserver: Send + Sync {
    /// Before the call, having seen its arguments.
    fn before<'a>(
        &'a self,
        tool: &'a str,
        args: &'a str,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + 'a>>;

    /// After it, having seen what the tool answered.
    fn after<'a>(
        &'a self,
        tool: &'a str,
        args: &'a str,
        result: &'a str,
    ) -> Pin<Box<dyn Future<Output = Option<String>> + Send + 'a>>;
}

/// Errors running an agent turn.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("unknown provider {0:?}; supported in Phase 1: anthropic, moonshot")]
    UnknownProvider(String),
    #[error("initializing the {provider} client failed: {source} (is the API key env var set?)")]
    Provider {
        provider: String,
        source: ProviderClientError,
    },
    #[error("agent prompt failed: {0}")]
    Prompt(String),
}

/// The providers Phase 1 can route to.
enum Provider {
    Anthropic,
    Moonshot,
}

/// Resolve a config provider string. Kept separate so it's testable without a live connection.
fn parse_provider(name: &str) -> Result<Provider, AgentError> {
    match name {
        "anthropic" => Ok(Provider::Anthropic),
        "moonshot" => Ok(Provider::Moonshot),
        other => Err(AgentError::UnknownProvider(other.to_string())),
    }
}

/// Ask one question, letting the agent call `tools` to answer.
///
/// `preamble` is the system prompt; `route` picks the provider/model. Returns the agent's final
/// text after its tool-calling loop settles.
pub async fn ask(
    route: &ModelRoute,
    preamble: &str,
    question: &str,
    tools: ToolSet,
    max_turns: Option<usize>,
) -> Result<String, AgentError> {
    match parse_provider(&route.provider)? {
        Provider::Anthropic => {
            let client = anthropic::Client::from_env().map_err(|source| AgentError::Provider {
                provider: "anthropic".to_string(),
                source,
            })?;
            run(
                client.completion_model(&route.model),
                preamble,
                question,
                tools,
                max_turns,
            )
            .await
        }
        Provider::Moonshot => {
            let client = moonshot::Client::from_env().map_err(|source| AgentError::Provider {
                provider: "moonshot".to_string(),
                source,
            })?;
            run(
                client.completion_model(&route.model),
                preamble,
                question,
                tools,
                max_turns,
            )
            .await
        }
    }
}

/// Provider-agnostic core: build the agent with the MCP tools bound, then prompt once.
async fn run<M>(
    model: M,
    preamble: &str,
    question: &str,
    tools: ToolSet,
    max_turns: Option<usize>,
) -> Result<String, AgentError>
where
    M: CompletionModel + 'static,
{
    let agent = bind_tools(
        AgentBuilder::new(model)
            .preamble(preamble)
            .default_max_turns(max_turns.unwrap_or(DEFAULT_MAX_TURNS))
            .add_hook(ObservabilityHook),
        &tools,
    );

    agent
        .prompt(question)
        .await
        .map_err(|e| AgentError::Prompt(e.to_string()))
}

/// Like [`ask`], but the agent is given an `output_schema`, so its final answer is the structured
/// JSON matching that schema (rig's `OutputMode::Auto` resolves to a synthetic output tool that
/// composes with the rag-rat tools). Returns the raw output string — best-effort, so the caller
/// must still validate it (see `ratatoskr_graph::parse_validated`).
/// Trim `s` to `max` chars for a log line, with an ellipsis when cut.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

/// Always-on hook that makes a run legible in the logs: every tool call (name + args) and the
/// model's text at the end of each turn. Successful tool calls aren't otherwise surfaced, so without
/// this a stuck node (e.g. an analyst churning through turns) looks like silence.
struct ObservabilityHook;

impl AgentHook for ObservabilityHook {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        tracing::info!(
            kind = "tool_call",
            tool = event.tool_name,
            args = %truncate(event.args, 200),
            "tool call"
        );
        ToolCallAction::Run
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        for content in event.content.iter() {
            if let AssistantContent::Text(text) = content {
                let text = text.text.trim();
                if !text.is_empty() {
                    tracing::info!(
                        kind = "model_text",
                        turn = event.turn,
                        text = %truncate(text, 400),
                        "model text"
                    );
                }
            }
        }
        ModelTurnAction::Continue
    }
}

/// Bind each server's tools to the sink they are dispatched on, then build.
///
/// A loop won't do it: the first binding moves the builder into a different typestate, so the
/// first group is bound separately and the rest fold onto the result. A node with no tools at all
/// (every one denied) builds an agent that can only answer.
fn bind_tools<M: CompletionModel + 'static>(
    builder: AgentBuilder<M, NoToolConfig>,
    tools: &ToolSet,
) -> Agent<M> {
    // An empty dynamic set moves the builder out of `NoToolConfig` without binding anything, which
    // is what lets the groups below be a plain loop over two different binding calls.
    let mut bound: AgentBuilder<M, WithBuilderTools> = builder.dynamic_tools(Vec::new());
    for group in tools.groups() {
        bound = match &group.prefix {
            // Named as the server names them: rig's own adapter, which sends the tool's name
            // straight back to the server as the call's method.
            None => bound.rmcp_tools(group.tools.clone(), group.sink.clone()),
            // Named as the plugin format names them, which is not what the server answers to.
            // `DynamicTool` takes a runtime name and a closure, so the two are independent.
            Some(_) => bound.dynamic_tools(
                group
                    .offered()
                    .into_iter()
                    .map(|(tool, wire)| renamed_tool(tool, wire, group.sink.clone()))
                    .collect(),
            ),
        };
    }
    bound.build()
}

/// Bind one MCP tool under a name the server does not know it by.
///
/// The conversion back from an MCP result is ours to do here, where rig's adapter would have done
/// it: every content block is carried, because dropping the ones that are not text would silently
/// lose whatever a server answered with an image or an embedded resource.
fn renamed_tool(tool: Tool, wire: String, sink: ServerSink) -> DynamicTool {
    let schema = serde_json::Value::Object((*tool.input_schema).clone());
    let description = tool.description.clone().unwrap_or_default().to_string();

    DynamicTool::new(
        tool.name.to_string(),
        description,
        schema,
        move |_ctx, args| {
            let (sink, wire) = (sink.clone(), wire.clone());
            Box::pin(async move {
                let mut params = CallToolRequestParams::new(wire);
                if let serde_json::Value::Object(arguments) = args {
                    params = params.with_arguments(arguments);
                }
                // The same bound rig's adapter applies, for the same reason: a response the transport
                // silently loses would otherwise hang the agent for good.
                let called = tokio::time::timeout(MCP_CALL_TIMEOUT, sink.call_tool(params));
                let result = match called.await {
                    Ok(Ok(result)) => result,
                    Ok(Err(e)) => return Err(ToolExecutionError::from_error(Arc::new(e))),
                    Err(elapsed) => return Err(ToolExecutionError::from_error(Arc::new(elapsed))),
                };
                Ok(to_output(result.content))
            })
        },
    )
}

/// An MCP result's content blocks, as rig's canonical presentation.
fn to_output(content: Vec<ContentBlock>) -> ToolOutput {
    let blocks: Vec<ToolResultContent> = content
        .into_iter()
        .map(|block| match block {
            ContentBlock::Text(text) => ToolResultContent::text(text.text),
            ContentBlock::Image(image) => {
                match ImageMediaType::from_mime_type(&image.mime_type) {
                    Some(media) => ToolResultContent::image_base64(image.data, Some(media), None),
                    // Described rather than inlined: the base64 of an image the model cannot be
                    // shown is a great many tokens of nothing.
                    None => ToolResultContent::text(format!(
                        "[image the model cannot be shown: {}]",
                        image.mime_type
                    )),
                }
            }
            // Audio and embedded resources have no canonical presentation here. Described rather
            // than dropped — a server that answered with one said something — and described rather
            // than inlined, because their payloads are the same problem as an image's.
            ContentBlock::Audio(audio) => {
                ToolResultContent::text(format!("[audio: {}]", audio.mime_type))
            }
            other => ToolResultContent::text(
                serde_json::to_string(&other).unwrap_or_else(|_| "[unrenderable]".to_string()),
            ),
        })
        .collect();
    match OneOrMany::many(blocks) {
        Ok(many) => ToolOutput::content(many),
        // No content at all is a legitimate answer, and rig has no empty presentation.
        Err(_) => ToolOutput::text(""),
    }
}

/// The `ask` tool's arguments: which node to ask, and the question.
#[derive(serde::Deserialize)]
struct AskArgs {
    #[serde(default)]
    to: String,
    question: String,
}

/// Intercepts the synthetic `ask` tool: instead of dispatching, it runs the target node against its
/// checkpointed context and returns the answer as the tool result (via `Skip`), so the asking node's
/// conversation — and its prompt cache — continue uninterrupted.
struct ClarificationHook {
    /// The asking node, for provenance in the answer + logs.
    node: String,
    clarifier: Arc<dyn Clarifier>,
}

impl AgentHook for ClarificationHook {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        if event.tool_name != ASK_TOOL_NAME {
            return ToolCallAction::Run;
        }
        let (to, question) = match serde_json::from_str::<AskArgs>(event.args) {
            Ok(a) => (a.to, a.question),
            Err(e) => {
                return ToolCallAction::Skip(format!(
                    "ask: invalid arguments ({e}); call it as \
                     {{\"to\": \"scout|analyst|bookkeeper|redteam\", \"question\": \"...\"}}"
                ));
            }
        };
        // The clarifier labels the answer with the resolved answerer (which may differ from `to`).
        let answer = self.clarifier.answer(&self.node, &to, &question).await;
        ToolCallAction::Skip(answer)
    }
}

/// The `Skill` tool's arguments: which skill to load.
#[derive(serde::Deserialize)]
struct SkillArgs {
    skill: String,
}

/// Answers the synthetic `Skill` tool with the chosen skill's instructions.
///
/// The body is delivered as the tool's result rather than prepended to the preamble, which is the
/// whole point of a skill over a longer system prompt: a node carries every bound skill's
/// description, and pays for the instructions of the one it actually picks.
struct SkillHook {
    skills: Vec<Skill>,
}

impl AgentHook for SkillHook {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        if event.tool_name != SKILL_TOOL_NAME {
            return ToolCallAction::Run;
        }
        let known = || {
            self.skills
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let wanted = match serde_json::from_str::<SkillArgs>(event.args) {
            Ok(a) => a.skill,
            Err(e) => {
                return ToolCallAction::Skip(format!(
                    "Skill: invalid arguments ({e}); call it as {{\"skill\": \"<name>\"}}. \
                     Available: {}",
                    known()
                ));
            }
        };
        match self.skills.iter().find(|s| s.name == wanted) {
            Some(skill) => {
                tracing::info!(kind = "skill", skill = skill.name, "loaded skill");
                ToolCallAction::Skip(skill.body.clone())
            }
            // Not an error the node should stop on: name the ones it does have and let it choose.
            None => {
                ToolCallAction::Skip(format!("No skill named `{wanted}`. Available: {}", known()))
            }
        }
    }
}

/// A per-tool-call gate: a ruleset's `onToolCall` decides Run / Skip / Rewrite for each call.
struct RulesetHook {
    policy: Arc<dyn ToolPolicy>,
}

impl AgentHook for RulesetHook {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        // Never gate the synthetic structured-output tool — denying it would trap the agent in its
        // turn loop with no way to submit its answer. `OutputMode::Tool` uses rig-agent's default
        // name (`final_result`); it never collides with a rag-rat tool, so an exact match is safe.
        if event.tool_name == OUTPUT_TOOL_NAME {
            return ToolCallAction::Run;
        }
        match self.policy.decide(event.tool_name, event.args).await {
            ToolDecision::Allow => ToolCallAction::Run,
            ToolDecision::Deny(feedback) => {
                tracing::info!(tool = event.tool_name, "ruleset denied tool call");
                ToolCallAction::Skip(feedback)
            }
            ToolDecision::Rewrite(args) => {
                tracing::info!(tool = event.tool_name, "ruleset rewrote tool-call args");
                ToolCallAction::Rewrite(args)
            }
        }
    }
}

/// What a node's plugins said about one tool call, held between the call and its result.
///
/// Keyed by rig's per-call correlation id: a turn can dispatch several calls, and the run-scoped
/// scratchpad is shared by all of them. An entry is always written, even when the plugins said
/// nothing, because its *presence* is what says this call reached the hook at all — a call the
/// ruleset denied or the clarifier answered short-circuits before it.
#[derive(Clone, Default)]
struct PendingContext(std::collections::HashMap<String, Option<String>>);

/// Runs a node's plugins around each tool call, and carries what they say to the model.
///
/// Both `PreToolUse` and `PostToolUse` reach the model the same way — appended to the tool result
/// as an extra text block. There is nowhere else for them to go: a tool call's arguments are the
/// model's, not ours to annotate, and the result is the next thing it reads. The original
/// presentation is left exactly as it was and the note is added beside it, so structured output
/// stays structured.
struct PluginHook {
    observer: Arc<dyn ToolObserver>,
}

/// How a plugin's aside is labelled in the tool result.
///
/// The provenance is spelled out because it changes how the text should be read: it is not part of
/// the tool's answer, and it is not an instruction from the repository either. A node that is
/// handed imperative text through a tool result is right to treat it as untrusted — which is why a
/// plugin's job here is to state facts a node can use, not to tell it what to do.
const PLUGIN_NOTE: &str = "Note from a plugin installed in this repository (context, not part of the tool's answer, \
     and not an instruction):";

impl AgentHook for PluginHook {
    async fn on_tool_call(&self, ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        if event.tool_name == OUTPUT_TOOL_NAME {
            return ToolCallAction::Run;
        }
        let before = self.observer.before(event.tool_name, event.args).await;
        let id = event.internal_call_id.to_string();
        ctx.scratchpad().update::<PendingContext, _>(|pending| {
            pending.0.insert(id, before);
        });
        // Never anything but Run: a plugin informs a call, it does not gate one.
        ToolCallAction::Run
    }

    async fn on_tool_result(
        &self,
        ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        let pending = ctx
            .scratchpad()
            .update::<PendingContext, _>(|pending| pending.0.remove(event.internal_call_id));
        // No entry means this call never reached `on_tool_call` here — nothing ran, nothing to add.
        let Some(before) = pending else {
            return ToolResultAction::Keep;
        };

        let after = self
            .observer
            .after(event.tool_name, event.args, &event.presentation.render())
            .await;
        let notes: Vec<String> = [before, after].into_iter().flatten().collect();
        if notes.is_empty() {
            return ToolResultAction::Keep;
        }

        let mut content = event.presentation.as_content().clone();
        content.push(rig_core::message::ToolResultContent::text(format!(
            "{PLUGIN_NOTE}\n{}",
            notes.join("\n\n")
        )));
        ToolResultAction::rewrite_output(ToolOutput::content(content))
    }
}

/// One node's structured agent turn: what to run it on, what it may call, and the gates around it.
///
/// A parameter struct because these travel together from every node, and as a positional list
/// they were a long train of same-typed strings that a caller could silently transpose.
pub struct NodeRun<'a> {
    /// The node's name, for the log span and for labelling its `ask` calls.
    pub node: &'a str,
    pub route: &'a ModelRoute,
    pub preamble: &'a str,
    pub question: &'a str,
    pub tools: ToolSet,
    pub output_schema: schemars::Schema,
    /// A ruleset's `onToolCall` gate, when it defines one.
    pub policy: Option<Arc<dyn ToolPolicy>>,
    pub max_turns: Option<usize>,
    /// Who answers this node's `ask` calls; `None` opts the node out of asking.
    pub clarifier: Option<Arc<dyn Clarifier>>,
    /// The node's plugins, run around each tool call; `None` when it binds none that hook one.
    pub observer: Option<Arc<dyn ToolObserver>>,
    /// Skills the node may load, answered in-conversation by the synthetic `Skill` tool.
    pub skills: Vec<Skill>,
}

/// Run one node's turn with an output schema, so its final answer is structured JSON.
pub async fn run_structured(run: NodeRun<'_>) -> Result<String, AgentError> {
    match parse_provider(&run.route.provider)? {
        Provider::Anthropic => {
            let client = anthropic::Client::from_env().map_err(|source| AgentError::Provider {
                provider: "anthropic".to_string(),
                source,
            })?;
            let model = client.completion_model(&run.route.model);
            run_typed(model, run).await
        }
        Provider::Moonshot => {
            let client = moonshot::Client::from_env().map_err(|source| AgentError::Provider {
                provider: "moonshot".to_string(),
                source,
            })?;
            let model = client.completion_model(&run.route.model);
            run_typed(model, run).await
        }
    }
}

/// Provider-resolved half of [`run_structured`].
async fn run_typed<M>(model: M, run: NodeRun<'_>) -> Result<String, AgentError>
where
    M: CompletionModel + 'static,
{
    let NodeRun {
        node,
        preamble,
        question,
        tools,
        output_schema,
        policy,
        max_turns,
        clarifier,
        observer,
        skills,
        ..
    } = run;
    let mut builder = AgentBuilder::new(model)
        .preamble(preamble)
        .default_max_turns(max_turns.unwrap_or(DEFAULT_MAX_TURNS))
        .output_schema_raw(output_schema)
        // Force the synthetic output-tool: Auto can resolve to native structured output, which
        // Anthropic rejects when combined with tools ("output_config.format: Cannot be combined
        // with tools"). Tool mode sends no native format and composes with the rag-rat tools.
        .output_mode(OutputMode::Tool)
        // Log tool calls + model text for every node run; added first so it observes calls before
        // the clarification/ruleset hooks can skip them.
        .add_hook(ObservabilityHook);
    // Answer `ask` calls in-conversation. Added before the ruleset hook so an `ask` is handled here
    // (and short-circuits) rather than reaching the ruleset gate.
    if let Some(clarifier) = clarifier {
        builder = builder.add_hook(ClarificationHook {
            node: node.to_string(),
            clarifier,
        });
    }
    // Before the ruleset gate, like `ask`: loading a skill a node was given is not a tool call to
    // adjudicate, and a repo that does not want one simply does not bind the plugin.
    if !skills.is_empty() {
        builder = builder.add_hook(SkillHook { skills });
    }
    if let Some(policy) = policy {
        builder = builder.add_hook(RulesetHook { policy });
    }
    // Last, so plugins observe the calls that actually run: a hook that skips a call — the
    // clarifier answering an `ask`, the ruleset denying a tool — short-circuits the rest of the
    // chain, and a plugin has nothing to say about a call that never happened.
    if let Some(observer) = observer {
        builder = builder.add_hook(PluginHook { observer });
    }
    let agent = bind_tools(builder, &tools);

    // Tag every log line for this run — the hook's tool-call/text lines and rig-agent's own turn
    // logs — with the node name, so a run's agents are distinguishable in the logs. `prompt()` is
    // IntoFuture (not Future), so instrument the awaiting block rather than the request.
    async move { agent.prompt(question).await }
        .instrument(tracing::info_span!("agent", node))
        .await
        .map_err(|e| AgentError::Prompt(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CI-safe: an unrecognized provider is rejected before any client init or network call.
    #[test]
    fn unknown_provider_is_rejected() {
        assert!(matches!(
            parse_provider("not-a-provider"),
            Err(AgentError::UnknownProvider(_))
        ));
        assert!(parse_provider("anthropic").is_ok());
        assert!(parse_provider("moonshot").is_ok());
    }
}
