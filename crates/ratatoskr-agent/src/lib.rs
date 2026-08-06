//! Builds a `rig` agent bound to a model and rag-rat's MCP tools, and runs one prompt.
//!
//! Phase 1 has exactly one caller (`ratatoskr ask`), so provider resolution is a small `match`
//! rather than a registry. The agent's own multi-turn loop (from `rig-agent`) does the tool
//! calling — we hand it the tools and a client handle via `.rmcp_tools()`.

pub mod compaction;
pub mod files;

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ratatoskr_core::{ModelRoute, NodeTelemetry, TokenUsage, ToolDecision, ToolPolicy};
use ratatoskr_mcp::ToolSet;
use rig_agent::AgentBuilder;
use rig_agent::agent::{
    Agent, AgentHook, CompletionResponseEvent, HookContext, ModelTurnAction, ModelTurnFinished,
    NoToolConfig, ObservationAction, OutputMode, ToolCall, ToolCallAction, ToolResultAction,
    ToolResultEvent, WithBuilderTools,
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
use rmcp::ServiceError;
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CallToolResult, ClientRequest, ContentBlock,
    ServerResult, Tool,
};
use rmcp::service::{PeerRequestOptions, ServerSink};
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

/// What a node's run reports to the plugins bound to it (implemented in `ratatoskr-nodes`, which
/// is where a node's plugin bindings are known).
///
/// Every one of these answers with context for the model and nothing else: whether a call proceeds
/// is a ruleset's `onToolCall` decision, not a plugin's. None may fail — a hook that breaks
/// contributes no text and the node is unaffected.
pub trait PluginHooks: Send + Sync {
    /// The node is starting. Its context opens the node's conversation.
    fn starting<'a>(&'a self, node: &'a str) -> Answer<'a>;

    /// The node is about to be prompted. Its context rides alongside the prompt.
    fn prompting<'a>(&'a self, prompt: &'a str) -> Answer<'a>;

    /// Before a tool call, having seen its arguments.
    fn before<'a>(&'a self, tool: &'a str, args: &'a str) -> Answer<'a>;

    /// After it, having seen what the tool answered.
    fn after<'a>(&'a self, tool: &'a str, args: &'a str, result: &'a str) -> Answer<'a>;

    /// The node has finished — with the last thing it said, or with why it could not.
    ///
    /// Fired on both paths, because a plugin that opened something at `starting` has to be told
    /// the node is over however it ended. Nothing is injected: the node's answer is already made,
    /// and its next reader is a schema. A hook here runs for what it *does* — recording,
    /// notifying, syncing — and any context it returns is reported as unused rather than dropped.
    fn finished<'a>(&'a self, node: &'a str, outcome: Result<&'a str, &'a str>) -> Answer<'a>;
}

/// What a plugin hook contributes, once it has run.
pub type Answer<'a> = Pin<Box<dyn Future<Output = Option<String>> + Send + 'a>>;

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
                route.max_tokens(),
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
                route.max_tokens(),
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
    max_tokens: u64,
) -> Result<String, AgentError>
where
    M: CompletionModel + 'static,
{
    let (builder, meter) = metered(model, preamble, max_turns, max_tokens);
    let agent = bind_tools(builder, &tools, None);

    let answer = agent.prompt(question).await;
    // No store to checkpoint to here, so it goes to the log. A one-shot question whose cost is
    // unknowable is the same defect as an uncounted node, in a smaller place.
    let (usage, calls) = meter.read();
    tracing::info!(
        kind = "usage",
        calls,
        "gen_ai.usage.input_tokens" = usage.input_tokens,
        "gen_ai.usage.output_tokens" = usage.output_tokens,
        "gen_ai.usage.cached_input_tokens" = usage.cached_input_tokens,
        "ask usage"
    );
    answer.map_err(|e| AgentError::Prompt(e.to_string()))
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

/// Start an agent, metered.
///
/// The only place in this crate that constructs one. Every model call therefore carries the usage
/// hook and a token cap by construction rather than by whoever wrote the call site remembering —
/// which is how the compactor and the `ask` path came to spend tokens nobody counted.
///
/// Returns the builder alongside the handles its usage accumulates into; the caller reads them
/// after the prompt settles, including when it failed.
fn metered<M: CompletionModel + 'static>(
    model: M,
    preamble: &str,
    max_turns: Option<usize>,
    max_tokens: u64,
) -> (AgentBuilder<M, NoToolConfig>, Meter) {
    let usage = UsageHook::default();
    let meter = Meter {
        total: Arc::clone(&usage.total),
        calls: Arc::clone(&usage.calls),
    };
    let builder = AgentBuilder::new(model)
        .preamble(preamble)
        .default_max_turns(max_turns.unwrap_or(DEFAULT_MAX_TURNS))
        // Always set, never left to the provider client to infer from the model name: its table of
        // known prefixes does not include models released after it was compiled, and a model that
        // falls through it goes out with no cap at all — which Anthropic rejects outright, losing
        // the run at that node's first call.
        .max_tokens(max_tokens)
        // Log tool calls + model text; added before the gates so it observes calls the
        // clarification and ruleset hooks may skip.
        .add_hook(ObservabilityHook)
        .add_hook(usage);
    (builder, meter)
}

/// What one metered agent spent, readable once its prompt has settled.
pub struct Meter {
    total: Arc<Mutex<TokenUsage>>,
    calls: Arc<AtomicU64>,
}

impl Meter {
    /// The usage so far. Read after the prompt returns — including on the error path, where the
    /// calls made before the failure cost exactly what they cost.
    pub fn read(&self) -> (TokenUsage, u64) {
        (
            *self.total.lock().expect("usage mutex poisoned"),
            self.calls.load(Ordering::Relaxed),
        )
    }
}

/// Accumulates what the provider said each model call cost.
///
/// It counts on `on_completion_response`, not `on_model_turn_finished`, because a turn a hook later
/// rejects and retries was still billed. Counting accepted turns only would report a number smaller
/// than the invoice, which is the wrong direction for anything that decides whether to keep going.
///
/// The total is read back after the run, including when the run failed: the calls made before the
/// failure cost the same as the ones before a success.
#[derive(Default)]
struct UsageHook {
    total: Arc<Mutex<TokenUsage>>,
    calls: Arc<AtomicU64>,
}

impl AgentHook for UsageHook {
    /// Counts model calls, including ones a later hook rejects and retries — all of them were
    /// billed, so the call count is what says how much work a turn actually took.
    async fn on_completion_response(
        &self,
        _ctx: &HookContext,
        _event: CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        ObservationAction::Continue
    }

    /// Accumulates what the turn cost.
    ///
    /// Not `on_completion_response`, which is where this started and where it was wrong. On the
    /// streaming path a response is assembled from chunks and its usage is drained at the end, so
    /// the per-response event carries the input and cache counts and almost none of the output —
    /// live runs recorded 63 output tokens for sixteen turns that produced 5 KB of structured
    /// answer. This event fires after the stream is drained, which is where the output count
    /// becomes known.
    ///
    /// The cost is that a turn rejected and retried contributes nothing here. That is the better
    /// error: an undercount bounded by the retry rate beats an undercount of everything.
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.total
            .lock()
            .expect("usage mutex poisoned")
            .add(TokenUsage {
                input_tokens: event.usage.input_tokens,
                output_tokens: event.usage.output_tokens,
                cached_input_tokens: event.usage.cached_input_tokens,
                cache_creation_input_tokens: event.usage.cache_creation_input_tokens,
            });
        ModelTurnAction::Continue
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
    files: Option<&std::path::Path>,
) -> Agent<M> {
    // An empty dynamic set moves the builder out of `NoToolConfig` without binding anything, which
    // is what lets the groups below be a plain loop over two different binding calls.
    let mut bound: AgentBuilder<M, WithBuilderTools> = builder.dynamic_tools(Vec::new());
    for group in tools.groups() {
        bound = match (&group.sink, &group.prefix) {
            // Named as the server names them: rig's own adapter, which sends the tool's name
            // straight back to the server as the call's method.
            (Some(sink), None) => bound.rmcp_tools(group.tools.clone(), sink.clone()),
            // Named as the plugin format names them, which is not what the server answers to.
            // `DynamicTool` takes a runtime name and a closure, so the two are independent.
            (Some(sink), Some(_)) => bound.dynamic_tools(
                group
                    .offered()
                    .into_iter()
                    .map(|(tool, wire)| renamed_tool(tool, wire, sink.clone()))
                    .collect(),
            ),
            // Answered by this host: a built-in it implements, or a synthetic one a hook
            // intercepts before dispatch — for which the implementation is never reached.
            (None, _) => bound.dynamic_tools(
                group
                    .tools
                    .iter()
                    .map(|tool| local_tool(tool, files))
                    .collect(),
            ),
        };
    }
    bound.build()
}

/// Text with a plugin's contribution ahead of it, labelled for what it is.
fn prefixed(text: &str, context: Option<String>) -> String {
    match context {
        Some(context) => format!("{PLUGIN_NOTE}\n{context}\n\n{text}"),
        None => text.to_string(),
    }
}

/// Bind a tool this host answers itself: a built-in file tool, or — for the synthetic ones a hook
/// answers in-conversation — a stand-in that says so if it is ever actually dispatched.
fn local_tool(tool: &Tool, files: Option<&std::path::Path>) -> DynamicTool {
    if let Some(root) = files
        && let Some(implemented) = files::implementation(&tool.name, root)
    {
        return implemented;
    }
    let name = tool.name.to_string();
    let schema = serde_json::Value::Object((*tool.input_schema).clone());
    let description = tool.description.clone().unwrap_or_default().to_string();
    DynamicTool::new(name.clone(), description, schema, move |_ctx, _args| {
        let name = name.clone();
        Box::pin(async move {
            Err(ToolExecutionError::other(format!(
                "{name} is answered inside the run and should never have been dispatched"
            )))
        })
    })
}

/// Bind one MCP tool under a name the server does not know it by.
///
/// Everything rig's own adapter does for a tool bound the ordinary way has to be done here too,
/// because binding by hand is what buys the rename: a per-call timeout that actually cancels, a
/// reported error that stays an error, and a result converted whole.
fn renamed_tool(tool: Tool, wire: String, sink: ServerSink) -> DynamicTool {
    let schema = serde_json::Value::Object((*tool.input_schema).clone());
    let description = tool.description.clone().unwrap_or_default().to_string();
    let shown = tool.name.to_string();

    DynamicTool::new(shown.clone(), description, schema, move |_ctx, args| {
        let (sink, wire, shown) = (sink.clone(), wire.clone(), shown.clone());
        Box::pin(async move {
            // Checked before the server is contacted. Quietly turning an array or a scalar into a
            // no-argument call can run a different operation than the model asked for.
            let mut params = CallToolRequestParams::new(wire);
            if let Some(arguments) = arguments(args, &shown)? {
                params = params.with_arguments(arguments);
            }
            interpret(&call_tool(&sink, params, &shown).await?, &shown)
        })
    })
}

/// The arguments to send, or a refusal the model can act on.
///
/// Quietly turning an array or a scalar into a no-argument call can run a different operation than
/// the model asked for, which is why this is checked before the server is contacted.
fn arguments(
    args: serde_json::Value,
    shown: &str,
) -> Result<Option<rmcp::model::JsonObject>, ToolExecutionError> {
    match args {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Object(arguments) => Ok(Some(arguments)),
        other => Err(ToolExecutionError::invalid_args(format!(
            "{shown} takes a JSON object of arguments, not {}",
            json_kind(&other)
        ))),
    }
}

/// What a finished call means: the presentation, or a failure carrying it.
///
/// A tool that says it failed has failed. Presenting that as a success would hide it from the
/// agent's own turn loop and from anything watching results.
fn interpret(result: &CallToolResult, shown: &str) -> Result<ToolOutput, ToolExecutionError> {
    let output = to_output(result);
    match result.is_error {
        Some(true) => Err(ToolExecutionError::other(format!(
            "{shown} reported an execution error"
        ))
        .with_model_output(output)),
        _ => Ok(output),
    }
}

/// One call, bounded — and cancelled at the server when the bound is reached, so a tool we have
/// stopped waiting for stops working too.
async fn call_tool(
    sink: &ServerSink,
    params: CallToolRequestParams,
    shown: &str,
) -> Result<CallToolResult, ToolExecutionError> {
    let mut options = PeerRequestOptions::no_options();
    options.timeout = Some(MCP_CALL_TIMEOUT);
    let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
    let response = match sink.send_cancellable_request(request, options).await {
        Ok(handle) => handle.await_response().await,
        Err(e) => Err(e),
    };
    match response {
        Ok(ServerResult::CallToolResult(result)) => Ok(result),
        Ok(_) => Err(ToolExecutionError::provider(format!(
            "{shown} answered something that was not a tool result"
        ))),
        Err(e @ ServiceError::Timeout { timeout }) => Err(ToolExecutionError::timeout(format!(
            "{shown} timed out after {timeout:?}"
        ))
        .with_source(e)),
        Err(e) => {
            Err(ToolExecutionError::provider(format!("{shown} request failed: {e}")).with_source(e))
        }
    }
}

/// What kind of JSON something is, for an argument error the model can act on.
fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// An MCP result as rig's canonical presentation: every content block, plus the structured value
/// when the server sent one.
fn to_output(result: &CallToolResult) -> ToolOutput {
    // rmcp's constructors repeat a structured value as a text block for older clients. Replace
    // that block with the typed value rather than showing the model both.
    let structured = result.structured_content.as_ref();
    let repeated = structured.map(serde_json::Value::to_string);
    let mut replaced = false;

    let mut blocks: Vec<ToolResultContent> = Vec::with_capacity(result.content.len());
    for block in &result.content {
        match (&block, repeated.as_deref(), structured) {
            (ContentBlock::Text(text), Some(repeated), Some(structured))
                if !replaced && text.text == repeated =>
            {
                blocks.push(ToolResultContent::json(structured.clone()));
                replaced = true;
            }
            _ => blocks.push(content_block(block)),
        }
    }
    if let Some(structured) = structured
        && !replaced
    {
        // Genuine content alongside a structured result: keep every block, typed value first.
        blocks.insert(0, ToolResultContent::json(structured.clone()));
    }

    match OneOrMany::many(blocks) {
        Ok(many) => ToolOutput::content(many),
        // No content at all is a legitimate answer, and rig has no empty presentation.
        Err(_) => ToolOutput::text(""),
    }
}

/// One MCP content block, as rig content.
fn content_block(block: &ContentBlock) -> ToolResultContent {
    match block {
        ContentBlock::Text(text) => ToolResultContent::text(text.text.clone()),
        ContentBlock::Image(image) => match ImageMediaType::from_mime_type(&image.mime_type) {
            Some(media) => ToolResultContent::image_base64(image.data.clone(), Some(media), None),
            // Described rather than inlined: the base64 of an image the model cannot be shown is
            // a great many tokens of nothing.
            None => ToolResultContent::text(format!(
                "[image the model cannot be shown: {}]",
                image.mime_type
            )),
        },
        // Audio has no presentation here, and its payload is the same problem as an image's.
        ContentBlock::Audio(audio) => {
            ToolResultContent::text(format!("[audio: {}]", audio.mime_type))
        }
        // A resource is usually small and usually text; carried as its JSON rather than dropped,
        // because a server that answered with one said something.
        other => ToolResultContent::text(
            serde_json::to_string(other).unwrap_or_else(|_| "[unrenderable]".to_string()),
        ),
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
    observer: Arc<dyn PluginHooks>,
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

/// Where a run's node turns report what they cost.
///
/// A node's `run` returns its typed output and nothing else — the graph vocabulary is about what a
/// node produces, and threading measurements through every node's return type would put bookkeeping
/// in the trait that models the work. Instead the executor creates one ledger per run, hands it to
/// each node, and drains it when it writes that node's checkpoint.
///
/// Entries are claimed oldest-first per node name, which is what makes the converge loop work: the
/// implementer runs once per iteration, and each checkpoint takes the turn that preceded it. The
/// fork's concurrent nodes have different names, so they never contend for the same entry.
#[derive(Default)]
pub struct RunLedger {
    entries: Mutex<Vec<(String, NodeTelemetry)>>,
}

impl RunLedger {
    /// Record what one node turn cost.
    pub fn record(&self, node: &str, telemetry: NodeTelemetry) {
        self.entries
            .lock()
            .expect("ledger mutex poisoned")
            .push((node.to_string(), telemetry));
    }

    /// Claim the oldest unclaimed entry for `node`, if it made a model turn at all. A node that
    /// drives a coding CLI rather than a model (the implementer) never records one, so `None` here
    /// is ordinary and means "nothing to report", not "something went missing".
    pub fn take(&self, node: &str) -> Option<NodeTelemetry> {
        let mut entries = self.entries.lock().expect("ledger mutex poisoned");
        let at = entries.iter().position(|(name, _)| name == node)?;
        Some(entries.remove(at).1)
    }

    /// The names of turns nobody claimed.
    ///
    /// Always empty on a finished run. Anything left means a node ran a model under one name and
    /// was checkpointed under another, and its cost went in the bin — the exact failure this whole
    /// table exists to stop, and one that is otherwise invisible because a dropped number reads
    /// identically to a node that never called a model.
    pub fn unclaimed(&self) -> Vec<String> {
        self.entries
            .lock()
            .expect("ledger mutex poisoned")
            .iter()
            .map(|(name, _)| name.clone())
            .collect()
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
    /// The node's plugins; `None` when it binds none that hook anything it does.
    pub observer: Option<Arc<dyn PluginHooks>>,
    /// Skills the node may load, answered in-conversation by the synthetic `Skill` tool.
    pub skills: Vec<Skill>,
    /// The repository the node's built-in file tools read within; `None` when it has none.
    pub files: Option<std::path::PathBuf>,
    /// Where this turn reports what it cost; `None` outside a run that records checkpoints.
    pub ledger: Option<Arc<RunLedger>>,
    /// What this node has to end up producing, in one line, for the compactor.
    ///
    /// A summary can only be judged against what the node still needs to do — what a scout must
    /// keep to write a papertrail summary is not what an implementer must keep to finish an edit.
    /// `None` leaves the node without compaction, which is right for one whose history cannot grow:
    /// a single-turn transcription has nothing to summarise and would only pay for the policy.
    pub produces: Option<&'a str>,
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
    let max_tokens = run.route.max_tokens();
    // The compactor summarises with the same model the node runs on. Cheaper would be tempting, but
    // a summary is the only record of the turns it replaces: the reader that has to reconstruct a
    // session from it is this model, and a weaker one deciding what that reader needs is a false
    // economy paid for in rediscovery.
    let for_compaction = model.clone();
    let NodeRun {
        node,
        route,
        preamble,
        question,
        tools,
        output_schema,
        policy,
        max_turns,
        clarifier,
        observer,
        skills,
        files,
        ledger,
        produces,
    } = run;
    let model_name = format!("{}/{}", route.provider, route.model);
    // `SubagentStart` opens the node's conversation, `UserPromptSubmit` rides with the prompt —
    // where each lands in the format, and a cleaner place for a plugin to speak than a tool result.
    let (preamble, question) = match &observer {
        Some(hooks) => (
            prefixed(preamble, hooks.starting(node).await),
            prefixed(question, hooks.prompting(question).await),
        ),
        None => (preamble.to_string(), question.to_string()),
    };

    let (builder, meter) = metered(model, &preamble, max_turns, max_tokens);
    let mut builder = builder
        .output_schema_raw(output_schema)
        // Force the synthetic output-tool: Auto can resolve to native structured output, which
        // Anthropic rejects when combined with tools ("output_config.format: Cannot be combined
        // with tools"). Tool mode sends no native format and composes with the rag-rat tools.
        .output_mode(OutputMode::Tool);
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
    if let Some(observer) = observer.clone() {
        builder = builder.add_hook(PluginHook { observer });
    }
    // Summarise the oldest turns rather than dropping them, once history outgrows the budget. A
    // plain window would evict exactly the turn that discovered a constraint — it is the oldest —
    // and the node would rediscover it, or retry the approach it ruled out.
    if let Some(produces) = produces {
        builder = builder.memory(compaction::compacting_memory(
            for_compaction,
            node,
            produces,
            compaction::default_budget(),
            ledger.clone(),
        ));
    }
    let agent = bind_tools(builder, &tools, files.as_deref());

    // Tag every log line for this run — the hook's tool-call/text lines and rig-agent's own turn
    // logs — with the node name, so a run's agents are distinguishable in the logs. `prompt()` is
    // IntoFuture (not Future), so instrument the awaiting block rather than the request.
    // Field names follow the OpenTelemetry GenAI semantic conventions. Those are still unstable and
    // not worth an SDK dependency yet, but naming to match now makes adopting one a layer swap
    // rather than a rename of every field a dashboard reads.
    let started = std::time::Instant::now();
    let answer = async move { agent.prompt(&question).await }
        .instrument(tracing::info_span!(
            "agent",
            node,
            "gen_ai.operation.name" = "invoke_agent",
            "gen_ai.agent.name" = node,
            "gen_ai.request.model" = %model_name,
        ))
        .await
        .map_err(|e| AgentError::Prompt(e.to_string()));

    let (usage, calls) = meter.read();
    if let Some(ledger) = &ledger {
        let telemetry = NodeTelemetry {
            model: Some(model_name),
            duration_ms: Some(started.elapsed().as_millis() as u64),
            usage,
            turns: Some(calls),
            // A node that failed still spent what it spent, and why it failed is the most useful
            // thing about its row.
            error: answer.as_ref().err().map(ToString::to_string),
        };
        tracing::info!(
            kind = "usage",
            node,
            "gen_ai.usage.input_tokens" = telemetry.usage.input_tokens,
            "gen_ai.usage.output_tokens" = telemetry.usage.output_tokens,
            "gen_ai.usage.cached_input_tokens" = telemetry.usage.cached_input_tokens,
            duration_ms = telemetry.duration_ms,
            "node usage"
        );
        ledger.record(node, telemetry);
    }

    // The node is over either way. A plugin told it was starting has to be told it stopped, or a
    // pairing it opened there is never closed.
    if let Some(hooks) = &observer {
        let failure = answer.as_ref().err().map(ToString::to_string);
        let outcome = match (&answer, &failure) {
            (Ok(answer), _) => Ok(answer.as_str()),
            (_, Some(failure)) => Err(failure.as_str()),
            (Err(_), None) => Err(""),
        };
        if let Some(unused) = hooks.finished(node, outcome).await {
            tracing::info!(
                node,
                chars = unused.len(),
                "a hook answered after the node finished; its context has nowhere to go"
            );
        }
    }
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A result carrying one text block.
    fn said(text: &str) -> CallToolResult {
        CallToolResult::success(vec![ContentBlock::text(text)])
    }

    #[test]
    fn a_tool_that_reports_failure_is_not_presented_as_success() {
        // Bound by hand, this is ours to notice: rig's own adapter checks it, and a `DynamicTool`
        // that answers `Ok` is a success however the tool described itself.
        let mut failed = said("lint config missing");
        failed.is_error = Some(true);

        let err = interpret(&failed, "mcp__plugin_linty_lint__check").expect_err("a failure");
        assert!(err.to_string().contains("reported an execution error"));
        // And the tool's own words still reach the model.
        assert!(interpret(&said("fine"), "x").is_ok());
    }

    #[test]
    fn a_structured_answer_is_carried_and_never_shown_twice() {
        // A server answering only with `structuredContent` used to reach the model as nothing.
        let mut structured = CallToolResult::success(Vec::new());
        structured.structured_content = Some(serde_json::json!({"findings": 2}));
        let output = to_output(&structured);
        assert_eq!(output.as_json(), Some(&serde_json::json!({"findings": 2})));

        // rmcp repeats it as text for older clients; that block becomes the typed value rather
        // than a second copy of it.
        let value = serde_json::json!({"findings": 2});
        let mut repeated = said(&value.to_string());
        repeated.structured_content = Some(value.clone());
        let output = to_output(&repeated);
        assert_eq!(output.as_content().len(), 1, "not the text and the value");
        assert_eq!(output.as_json(), Some(&value));
    }

    #[test]
    fn arguments_that_are_not_an_object_are_refused_rather_than_dropped() {
        // Dropping them would call the tool with no arguments at all, which for many tools is a
        // different operation rather than a failure.
        let err = arguments(serde_json::json!(["file.txt"]), "read").expect_err("refused");
        assert!(err.to_string().contains("not an array"), "{err}");
        assert!(
            arguments(serde_json::json!({"path": "f"}), "read")
                .unwrap()
                .is_some()
        );
        // No arguments at all is ordinary.
        assert!(
            arguments(serde_json::Value::Null, "read")
                .unwrap()
                .is_none()
        );
    }

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

    #[test]
    fn the_ledger_hands_each_checkpoint_its_own_turn() {
        let ledger = RunLedger::default();
        let cost = |n: u64| NodeTelemetry {
            usage: TokenUsage {
                input_tokens: n,
                ..Default::default()
            },
            ..Default::default()
        };
        // The converge loop runs the implementer repeatedly; each checkpoint must claim the turn
        // that preceded it, not the newest one.
        ledger.record("implementer", cost(1));
        ledger.record("red_team", cost(9));
        ledger.record("implementer", cost(2));

        assert_eq!(ledger.take("implementer").unwrap().usage.input_tokens, 1);
        assert_eq!(ledger.take("implementer").unwrap().usage.input_tokens, 2);
        assert!(
            ledger.take("implementer").is_none(),
            "a claimed entry is not handed out twice"
        );
        // A different node's entry is untouched by the drain of another's.
        assert_eq!(ledger.take("red_team").unwrap().usage.input_tokens, 9);
        // A node that never ran a model turn reports nothing rather than someone else's numbers.
        assert!(ledger.take("bookkeeper").is_none());
    }

    #[test]
    fn usage_accumulates_across_a_turn() {
        let mut total = TokenUsage::default();
        total.add(TokenUsage {
            input_tokens: 10,
            output_tokens: 1,
            cached_input_tokens: 8,
            cache_creation_input_tokens: 2,
        });
        total.add(TokenUsage {
            input_tokens: 5,
            output_tokens: 3,
            ..Default::default()
        });
        assert_eq!(total.input_tokens, 15);
        assert_eq!(total.output_tokens, 4);
        assert_eq!(total.cached_input_tokens, 8);
        assert_eq!(total.cache_creation_input_tokens, 2);
    }

    /// The whole write path against the real provider: declarations reach the model, it calls
    /// `Write` and `Edit`, and the file on disk is what it asked for.
    ///
    /// Ignored by default because it spends money and needs `ANTHROPIC_API_KEY`. It exists because
    /// the unit tests call `write`/`edit` directly — they prove the functions behave, and say
    /// nothing about whether a model can actually drive them through the schema they are declared
    /// with. That gap is where a tool that works in isolation turns out to be unusable.
    #[tokio::test]
    #[ignore = "calls the Anthropic API; run with --ignored"]
    async fn a_model_can_drive_the_write_tools_end_to_end() {
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        struct Done {
            /// What was changed.
            summary: String,
        }

        let root = std::env::temp_dir().join(format!("ratatoskr-live-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let mut tools = ratatoskr_mcp::ToolSet::default();
        tools.local().tools.extend(files::declarations());
        tools.local().tools.extend(files::edit_declarations());

        let route = ModelRoute {
            provider: "anthropic".into(),
            model: "claude-haiku-4-5-20251001".into(),
            max_tokens: None,
        };
        let answer = run_structured(NodeRun {
            node: "livetest",
            route: &route,
            preamble: "You edit files with the tools you are given. Do exactly what is asked.",
            question: "Create `greet.py` containing exactly these two lines:\n\
                       def greet(name):\n\
                       \x20   return \"hello \" + name\n\
                       Then use Edit to change the word hello to goodbye. Then report what you did.",
            tools,
            output_schema: schemars::schema_for!(Done),
            policy: None,
            max_turns: Some(12),
            clarifier: None,
            observer: None,
            skills: Vec::new(),
            files: Some(root.clone()),
            ledger: None,
            produces: Some("a summary of the change"),
        })
        .await
        .expect("the live run should complete");

        let written = std::fs::read_to_string(root.join("greet.py"))
            .expect("the model should have created greet.py");
        assert!(written.contains("def greet(name)"), "{written}");
        assert!(
            written.contains("goodbye") && !written.contains("hello"),
            "the Edit should have replaced hello with goodbye:\n{written}"
        );
        // The structured output still has to parse — a run that edited the file correctly but
        // could not fill its schema is a node failure, not a success.
        let done: Done = serde_json::from_str(
            answer
                .find('{')
                .zip(answer.rfind('}'))
                .map(|(a, b)| &answer[a..=b])
                .expect("structured output"),
        )
        .expect("the answer should match the schema");
        assert!(!done.summary.trim().is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The usage figures against the real provider.
    ///
    /// Ignored by default — it spends money. It exists because this is not unit-testable in any way
    /// that would have caught the bug: a hand-made `Usage` handed to the hook passes whichever
    /// event the hook listens on, and the whole defect was that the streaming path populates one of
    /// them and not the other. Only a real call distinguishes them.
    #[tokio::test]
    #[ignore = "calls the Anthropic API; run with --ignored"]
    async fn a_real_turn_reports_the_output_tokens_it_actually_spent() {
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        struct Answer {
            /// Several sentences, so the answer is unambiguously more than a handful of tokens.
            text: String,
        }

        let ledger = Arc::new(RunLedger::default());
        let route = ModelRoute {
            provider: "anthropic".into(),
            model: "claude-haiku-4-5-20251001".into(),
            max_tokens: None,
        };
        let answer = run_structured(NodeRun {
            node: "usagetest",
            route: &route,
            preamble: "You answer briefly but completely.",
            question: "In four or five full sentences, describe what a git worktree is.",
            tools: ratatoskr_mcp::ToolSet::default(),
            output_schema: schemars::schema_for!(Answer),
            policy: None,
            max_turns: Some(4),
            clarifier: None,
            observer: None,
            skills: Vec::new(),
            files: None,
            ledger: Some(Arc::clone(&ledger)),
            produces: None,
        })
        .await
        .expect("the live run should complete");

        let telemetry = ledger.take("usagetest").expect("the turn was recorded");
        let parsed: Answer = serde_json::from_str(
            answer
                .find('{')
                .zip(answer.rfind('}'))
                .map(|(a, b)| &answer[a..=b])
                .expect("structured output"),
        )
        .expect("the answer should match the schema");

        // The bug this pins: output tokens read as near-zero regardless of how much was produced.
        // The floor is bytes/8, far below any real tokenizer, so a correct count clears it easily.
        let floor = (parsed.text.len() / 8) as u64;
        assert!(
            telemetry.usage.output_tokens > floor,
            "reported {} output tokens for {} bytes of answer",
            telemetry.usage.output_tokens,
            parsed.text.len()
        );
        // Input is reported too, and the turn count is separate from the usage.
        assert!(telemetry.usage.input_tokens > 0);
        assert!(telemetry.turns.unwrap_or(0) > 0);
    }

    #[test]
    fn every_model_call_in_this_crate_is_built_through_the_metered_constructor() {
        // The defect this guards is not a wrong number, it is an absent one: the compactor and the
        // `ask` path each spent tokens nobody counted, because attaching the hook was a per-call
        // -site decision and forgetting it compiles. A fourth `AgentBuilder::new` would restore
        // exactly that, so there is exactly one, inside `metered`.
        let sources = [
            include_str!("lib.rs"),
            include_str!("compaction.rs"),
            include_str!("files.rs"),
        ];
        let built: usize = sources
            .iter()
            .map(|src| src.matches("AgentBuilder::new(").count())
            .sum();
        // One construction, plus this test's own mention of the name.
        assert_eq!(
            built, 2,
            "an agent is constructed somewhere other than `metered`, so its calls are unmetered"
        );
    }
}
