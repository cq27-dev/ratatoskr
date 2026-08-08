//! Builds a `rig` agent bound to a model and rag-rat's MCP tools, and runs one prompt.
//!
//! Phase 1 has exactly one caller (`ratatoskr ask`), so provider resolution is a small `match`
//! rather than a registry. The agent's own multi-turn loop (from `rig-agent`) does the tool
//! calling — we hand it the tools and a client handle via `.rmcp_tools()`.

pub mod compaction;
pub mod files;
pub mod publish;
pub mod shell;

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ratatoskr_core::{
    Control, Directive, ModelRoute, NodeTelemetry, TokenUsage, ToolDecision, ToolPolicy,
};
use ratatoskr_mcp::ToolSet;
use rig_agent::AgentBuilder;
use rig_agent::agent::{
    Agent, AgentHook, CompletionResponseEvent, HookContext, ModelTurnAction, ModelTurnFinished,
    NoToolConfig, ObservationAction, OutputMode, RetryRequest, ToolCall, ToolCallAction,
    ToolResultAction, ToolResultEvent, WithBuilderTools,
};
use rig_agent::completion::Prompt;
use rig_agent::tool::{DynamicTool, ToolExecutionError};
use rig_core::OneOrMany;
use rig_core::client::completion::CompletionClient;
use rig_core::client::{ProviderClient, ProviderClientError};
use rig_core::completion::CompletionModel;
use rig_core::message::{AssistantContent, ImageMediaType, MimeType, ToolResultContent};
use rig_core::providers::{anthropic, moonshot, openai};
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

/// Where a node asks what the operator watching it wants (implemented in `ratatoskr-nodes`, which
/// knows how to reach the dashboard).
///
/// Asked at turn boundaries only. A pause that landed mid-tool-call would leave a command half
/// run and a conversation holding an unanswered call, so the question is only ever put where the
/// answer can be acted on.
pub trait Controller: Send + Sync {
    fn poll<'a>(&'a self, node: &'a str) -> Pin<Box<dyn Future<Output = Control> + Send + 'a>>;
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
    #[error("unknown provider {0:?}; supported: anthropic, openai, moonshot")]
    UnknownProvider(String),
    #[error("initializing the {provider} client failed: {source} (is the API key env var set?)")]
    Provider {
        provider: String,
        source: ProviderClientError,
    },
    #[error("agent prompt failed: {0}")]
    Prompt(String),
}

/// The providers a route can name.
enum Provider {
    Anthropic,
    OpenAi,
    Moonshot,
}

/// Resolve a config provider string. Kept separate so it's testable without a live connection.
fn parse_provider(name: &str) -> Result<Provider, AgentError> {
    match name {
        "anthropic" => Ok(Provider::Anthropic),
        "openai" => Ok(Provider::OpenAi),
        "moonshot" => Ok(Provider::Moonshot),
        other => Err(AgentError::UnknownProvider(other.to_string())),
    }
}

/// An OpenAI client, on the Responses API rather than chat completions.
///
/// `from_env` reads `OPENAI_API_KEY` and `OPENAI_BASE_URL`, which is all this needs — the endpoint
/// headers are Anthropic's meridian arrangement and mean nothing here.
///
/// Responses is rig's default surface for this provider and the right one for a reasoning model: a
/// reasoning item round-trips across the agent's tool-calling turns, where chat completions drops
/// it and the model re-derives its thinking on every turn.
fn openai_client() -> Result<openai::Client, AgentError> {
    openai::Client::from_env().map_err(|source| AgentError::Provider {
        provider: "openai".to_string(),
        source,
    })
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
            let client = anthropic_client(&uuid::Uuid::new_v4().to_string())?;
            run(
                caching(client.completion_model(&route.model)),
                preamble,
                question,
                tools,
                max_turns,
                route,
            )
            .await
        }
        Provider::OpenAi => {
            run(
                openai_client()?.completion_model(&route.model),
                preamble,
                question,
                tools,
                max_turns,
                route,
            )
            .await
        }
        Provider::Moonshot => {
            let client = moonshot::Client::from_env().map_err(|source| AgentError::Provider {
                provider: "moonshot".to_string(),
                source,
            })?;
            run(
                // Not `caching`: the field exists on this provider's model too, but whether the
                // endpoint honours an Anthropic `cache_control` is its business, and sending one
                // it rejects would cost the call rather than the cache.
                client.completion_model(&route.model),
                preamble,
                question,
                tools,
                max_turns,
                route,
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
    route: &ModelRoute,
) -> Result<String, AgentError>
where
    M: CompletionModel + 'static,
{
    let (builder, meter) = metered(model, preamble, max_turns, Request::of(route));
    let agent = bind_tools(builder, &tools, None, None, None);

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
        // The write half, and the expensive one: a cache write is billed above the ordinary input
        // rate, so a cost read from hits alone reads as cheaper than it was.
        "gen_ai.usage.cache_creation_input_tokens" = usage.cache_creation_input_tokens,
        "gen_ai.usage.reasoning_tokens" = usage.reasoning_tokens,
        "ask usage"
    );
    answer.map_err(|e| AgentError::Prompt(e.to_string()))
}

/// Bind a tool declaration to the closure that answers it.
///
/// The name, description and schema a tool is offered under are the ones in its declaration.
/// Unpacking them again at each implementation is how the two come to disagree — a tool described
/// one way and answering to another.
pub(crate) fn answered_by<F>(declaration: Tool, callback: F) -> DynamicTool
where
    F: for<'a> Fn(
            &'a mut rig_agent::tool::ToolContext,
            serde_json::Value,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<ToolOutput, ToolExecutionError>> + Send + 'a>,
        > + Send
        + Sync
        + 'static,
{
    DynamicTool::new(
        declaration.name.to_string(),
        declaration
            .description
            .clone()
            .unwrap_or_default()
            .to_string(),
        serde_json::Value::Object((*declaration.input_schema).clone()),
        callback,
    )
}

/// Keep the last `max` chars of `s`, saying how much was dropped.
///
/// The end is the half worth keeping for anything a command printed: runners put their summary
/// last, so a head-truncated failure is the part that says a suite ran and not the part that says
/// what failed. Stating the loss matters as much — a reader not told it was cut reads a partial
/// suite as a whole one.
pub fn tail(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let kept: String = s.chars().skip(count - max).collect();
    format!("[{} earlier characters omitted]\n{kept}", count - max)
}

/// Strip invisible characters that carry no legitimate meaning in text entering a prompt.
///
/// Two ranges, both delivery mechanisms for instructions no renderer shows: Unicode Tag
/// characters (U+E0000–U+E007F), and the zero-width set (ZWSP, ZWNJ, ZWJ, and the BOM/ZWNBSP).
/// Ordinary text — including any legitimate non-ASCII — is left byte-identical. Deliberately not a
/// filter for all format characters: the caller's content is machine output, so the named ranges
/// are enough and a wider net would drop characters with real meaning elsewhere.
pub fn sanitize(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(*c, '\u{E0000}'..='\u{E007F}' | '\u{200B}'..='\u{200D}' | '\u{FEFF}'))
        .collect()
}

/// Trim `s` to `max` chars for a log line, with an ellipsis when cut.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

/// True when the text is the model writing the *other side* of the conversation — a fabricated
/// tool result — rather than its own output. The endpoint reconstructs each request as a prompt
/// into a coding-CLI session, so the model sees a transcript and sometimes continues it past its
/// own turn, inventing what it expects a tool to return. Such text is not model output and must
/// not be recorded as `model_text` (issue #126).
///
/// Anchored tightly on the bracketed header the harness uses: `[your <tool> …]:` at the start, or
/// `user[your ` anywhere. Callers strip the harness's `No response requested.` filler prefix
/// first, so this sees the text that follows it. The bare word `your` is not enough — a genuine
/// turn may mention it — so the bracket is the anchor.
fn is_transcript_continuation(text: &str) -> bool {
    let text = text.trim_start();
    // The other party's turn injected mid-stream: `user[your <tool> …]:`.
    if text.contains("user[your ") {
        return true;
    }
    // A reproduced tool result opening the block: `[your <tool> …]:`.
    if let Some(rest) = text.strip_prefix("[your ")
        && let Some(close) = rest.find(']')
    {
        // A tool name must sit between the bracket and its close, and the header ends in `]:`.
        return !rest[..close].trim().is_empty() && rest[close..].starts_with("]:");
    }
    false
}

/// A tool call's arguments, bounded but still parseable.
///
/// Truncating the serialized JSON is what a reader wants and what a parser cannot use: cutting
/// `{"file_path":"x","old_string":"..."}` mid-string leaves text that no longer parses, so every
/// consumer downstream loses ALL the arguments rather than the long one. `Read` survived it only by
/// being short enough to fit; `Edit` and `Write` never did. Bounding each value instead keeps the
/// shape intact, so the field a reader identifies the call by is always there.
/// Fields whose value is recoverable from somewhere better than a log line.
///
/// File contents live in the worktree and its branch, so a diff is reproducible from git long
/// after the log has rotated; a prose body ends up on the pull request or the issue. Keeping them
/// whole here would multiply the log by the size of the files a run touches to duplicate a record
/// that already exists.
const RECOVERABLE: &[&str] = &["content", "old_string", "new_string", "body"];

/// Nothing else is recoverable, so nothing else is abridged — up to a ceiling no real argument
/// reaches, which is there so one pathological call cannot swamp a day's log.
const ARGUMENT_CEILING: usize = 4_000;

fn abridged_args(raw: &str, bulk: usize) -> String {
    let Ok(mut parsed) = serde_json::from_str::<serde_json::Value>(raw) else {
        // Not JSON to begin with — nothing to preserve, so bound the whole thing.
        return truncate(raw, 200);
    };
    fn walk(v: &mut serde_json::Value, key: Option<&str>, bulk: usize) {
        match v {
            serde_json::Value::String(s) => {
                // The command that ran, the pattern searched, the path read: short, and written
                // down nowhere else. A run's account of what it did is only as good as these.
                let max = match key {
                    Some(k) if RECOVERABLE.contains(&k) => bulk,
                    _ => ARGUMENT_CEILING,
                };
                if s.chars().count() > max {
                    *s = truncate(s, max);
                }
            }
            serde_json::Value::Array(items) => {
                items.iter_mut().for_each(|i| walk(i, key, bulk));
            }
            serde_json::Value::Object(map) => {
                for (k, value) in map.iter_mut() {
                    walk(value, Some(k.as_str()), bulk);
                }
            }
            _ => {}
        }
    }
    walk(&mut parsed, None, bulk);
    parsed.to_string()
}

/// Start an agent, metered.
///
/// The only place in this crate that constructs one. Every model call therefore carries the usage
/// hook and a token cap by construction rather than by whoever wrote the call site remembering —
/// which is how the compactor and the `ask` path came to spend tokens nobody counted.
///
/// Returns the builder alongside the handles its usage accumulates into; the caller reads them
/// after the prompt settles, including when it failed.
/// Appended to every node's preamble.
///
/// A turn costs one round-trip to the model — measured at roughly six seconds against a large
/// cached context, and growing with it — while the tools themselves answer in about a tenth of a
/// second. A node that reads twelve files one per turn therefore spends over a minute waiting and
/// barely a second working. Left unasked, that is exactly what happens: an unprompted node batches
/// almost nothing, and the run's wall-clock is its turn count.
///
/// Nothing enforces this — it is the model's choice per turn — so it is worded as the default to
/// depart from rather than a rule, since a call that genuinely depends on the previous result must
/// still wait for it.
const TOOL_USE_GUIDANCE: &str = "\n\n## Calling tools\n\nWhen you need several things and none of \
    them depends on another's result, ask for them all in the same turn rather than one at a time. \
    Reading four files is one turn with four calls, not four turns. Only wait for a result when \
    what you do next actually depends on it.";

/// Whether this route leaves the model free to reason before answering.
///
/// Read from the route's provider params, since that is the only place it can be turned off from
/// here. `params.thinking.type = "disabled"` is the one shape that means no; anything else — a
/// budget, another type, or no `thinking` key at all — leaves the decision to the endpoint.
fn thinking_left_on(route: &ModelRoute) -> bool {
    let Some(params) = route.params.as_ref() else {
        return true;
    };
    params
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(|t| t.as_str())
        != Some("disabled")
}

/// The endpoint session this node attempt belongs to.
///
/// Fresh per attempt by default. `SessionScope::Reuse` keeps one session for the node across the
/// whole run instead, so a re-driven attempt — the implementer on a converge iteration, the
/// analyst on a revision — continues where the last one stopped rather than meeting the repository
/// for the first time again. Reuse needs a `conversation` key to be stable across those attempts;
/// without one there is nothing to be stable about, and it falls back to fresh.
fn session_id(run: &NodeRun<'_>) -> String {
    match (run.route.session, run.conversation) {
        (ratatoskr_core::SessionScope::Reuse, Some(key)) => key.to_string(),
        _ => uuid::Uuid::new_v4().to_string(),
    }
}

/// The headers one request carries: this deployment's static set, plus the session id when the
/// endpoint keys a session off one.
///
/// Separate from building the client so it can be checked without a provider or a network. The
/// failure worth catching is a header that never reaches the wire, and that is decided here.
fn endpoint_headers(
    endpoint: Option<&ratatoskr_core::EndpointConfig>,
    session: &str,
) -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    for (name, value) in endpoint.map(|e| &e.headers).into_iter().flatten() {
        match (
            http::HeaderName::try_from(name.as_str()),
            http::HeaderValue::try_from(value.as_str()),
        ) {
            (Ok(n), Ok(v)) => {
                headers.insert(n, v);
            }
            // Warned rather than failed: a malformed header is a config typo, and taking the run
            // down for it is a worse answer than sending the request without it.
            _ => tracing::warn!(
                header = %name,
                "ignoring an `[endpoint] headers` entry that is not a valid HTTP header"
            ),
        }
    }
    if let Some(name) = endpoint.and_then(|e| e.session_header.as_deref())
        && let (Ok(n), Ok(v)) = (
            http::HeaderName::try_from(name),
            http::HeaderValue::try_from(session),
        )
    {
        headers.insert(n, v);
    }
    headers
}

/// How this client addresses the endpoint, set once from config at startup.
///
/// Process-wide rather than threaded through `NodeRun` because it describes the deployment, not
/// the node: every node in a run talks to the same endpoint, and passing it down twenty call sites
/// would put the same value in twenty places for nobody's benefit.
static ENDPOINT: std::sync::OnceLock<ratatoskr_core::EndpointConfig> = std::sync::OnceLock::new();

/// Tell the agent layer how to address the endpoint. Called once, before any node runs.
pub fn configure_endpoint(endpoint: ratatoskr_core::EndpointConfig) {
    let _ = ENDPOINT.set(endpoint);
}

/// An Anthropic client carrying this deployment's headers.
///
/// `from_env` reads `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL` and nothing else, so the builder
/// is spelled out here to add them. `session` is fresh per node attempt and constant across that
/// attempt's turns — an endpoint that keys a session off it then continues one conversation rather
/// than rebuilding it every turn.
fn anthropic_client(session: &str) -> Result<anthropic::Client, AgentError> {
    let key = std::env::var("ANTHROPIC_API_KEY").map_err(|source| AgentError::Provider {
        provider: "anthropic".to_string(),
        source: ProviderClientError::EnvironmentVariable {
            name: "ANTHROPIC_API_KEY",
            source,
        },
    })?;
    let mut builder = anthropic::Client::builder().api_key(key);
    if let Ok(base) = std::env::var("ANTHROPIC_BASE_URL") {
        builder = builder.base_url(&base);
    }

    let headers = endpoint_headers(ENDPOINT.get(), session);
    if !headers.is_empty() {
        builder = builder.http_headers(headers);
    }
    builder.build().map_err(|source| AgentError::Provider {
        provider: "anthropic".to_string(),
        source: source.into(),
    })
}

/// Whether a failure was the call not completing, rather than the model answering unfavourably.
///
/// Matched on the message because that is all the provider client hands back — its error type
/// erases the distinction, and the alternative is retrying everything or nothing. Kept narrow: a
/// pattern that catches too much turns a permanent failure into two permanent failures.
fn is_transport_error(message: &str) -> bool {
    const TRANSPORT: [&str; 5] = [
        "HttpError",
        "error sending request",
        "connection closed",
        "connection reset",
        "timed out",
    ];
    TRANSPORT.iter().any(|m| message.contains(m))
}

/// Anthropic prompt caching, which rig leaves off entirely.
///
/// Without it no `cache_control` is sent and an agent loop pays for its whole transcript on every
/// turn: a live run showed the cache write growing 12k → 22k tokens across nine calls while the
/// read stayed flat at 7k, the hit rate falling 36% → 24% as the conversation grew. The history
/// was re-sent and re-written each turn — at the write premium — rather than read back.
///
/// Per-block markers only. Adding rig's top-level automatic breakpoint as well makes this *worse*,
/// which is the opposite of what the two names suggest: with both set, rig hands the moving message
/// point to the top-level breakpoint and stops marking messages itself, and this endpoint does
/// nothing with that field — so the growing half ends up cached by nobody.
///
/// Captured from real requests. With both on, markers sit on the system prompt and the last tool
/// and nowhere in `messages`. With only this one, the marker sits on the last message block and
/// advances with the conversation — message 0, then message 2 — which is what lets each turn read
/// the prefix instead of rewriting it.
fn caching(
    model: anthropic::completion::CompletionModel,
) -> anthropic::completion::CompletionModel {
    model.with_prompt_caching()
}

/// What one call asks of the provider, beyond the prompt and the tools.
///
/// A struct rather than three arguments because they arrive together from a route, and because the
/// compactor has settings but no route — it summarises with the node's model and asks nothing
/// extra of it.
pub struct Request {
    max_tokens: u64,
    temperature: Option<f64>,
    params: Option<serde_json::Value>,
}

impl Request {
    /// What a route asks for.
    pub fn of(route: &ModelRoute) -> Self {
        Request {
            max_tokens: route.max_tokens(),
            temperature: route.temperature,
            params: route.params.as_ref().and_then(|p| {
                serde_json::to_value(p)
                    .map_err(|e| {
                        // Warned rather than failed: the run is otherwise fine, and a route that
                        // cannot encode its own extras should not take the run down with it.
                        tracing::warn!(
                            model = %route.model,
                            "could not encode this route's `params`, sending the call without \
                             them: {e}"
                        );
                    })
                    .ok()
            }),
        }
    }

    /// The defaults: a cap and nothing else. What the compactor asks for — a summary wants neither
    /// a temperature of the node's choosing nor its extended thinking budget.
    pub fn plain() -> Self {
        Request {
            max_tokens: ratatoskr_core::DEFAULT_MAX_TOKENS,
            temperature: None,
            params: None,
        }
    }
}

fn metered<M: CompletionModel + 'static>(
    model: M,
    preamble: &str,
    max_turns: Option<usize>,
    request: Request,
) -> (AgentBuilder<M, NoToolConfig>, Meter) {
    let preamble = &format!("{preamble}{TOOL_USE_GUIDANCE}");
    let usage = UsageHook::default();
    let observability = ObservabilityHook::default();
    let meter = Meter {
        total: Arc::clone(&usage.total),
        calls: Arc::clone(&usage.calls),
        used: Arc::clone(&observability.used),
    };
    let builder = AgentBuilder::new(model)
        .preamble(preamble)
        .default_max_turns(max_turns.unwrap_or(DEFAULT_MAX_TURNS))
        // Always set, never left to the provider client to infer from the model name: its table of
        // known prefixes does not include models released after it was compiled, and a model that
        // falls through it goes out with no cap at all — which Anthropic rejects outright, losing
        // the run at that node's first call.
        .max_tokens(request.max_tokens)
        // Log tool calls + model text; added before the gates so it observes calls the
        // clarification and ruleset hooks may skip.
        .add_hook(observability)
        .add_hook(usage);
    // Left to the provider's default when the route says nothing, rather than given a default
    // here: "unset" is a position, and picking one for every node from one place would be picking
    // it for nodes nobody thought about.
    let builder = match request.temperature {
        Some(t) => builder.temperature(t),
        None => builder,
    };
    // Provider-specific fields, verbatim. Anthropic's extended thinking is the reason this exists.
    let builder = match request.params {
        Some(params) => builder.additional_params(params),
        None => builder,
    };
    (builder, meter)
}

/// What one metered agent spent, readable once its prompt has settled.
pub struct Meter {
    total: Arc<Mutex<TokenUsage>>,
    calls: Arc<AtomicU64>,
    /// Which tools the node actually called, as distinct from which it could have. Ordered, so a
    /// reader sees the same list twice for the same run.
    used: Arc<Mutex<std::collections::BTreeSet<String>>>,
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

    /// The tools the node called, in name order.
    pub fn used(&self) -> Vec<String> {
        self.used
            .lock()
            .expect("used-tools mutex poisoned")
            .iter()
            .cloned()
            .collect()
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
        tracing::debug!(
            kind = "response_usage",
            input = _event.usage.input_tokens,
            output = _event.usage.output_tokens,
            cached = _event.usage.cached_input_tokens,
            "response usage"
        );
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
        // Per turn, at debug: a total that looks wrong is otherwise impossible to attribute — the
        // question is always whether one turn was mismeasured or most turns reported nothing.
        tracing::debug!(
            kind = "turn_usage",
            turn = event.turn,
            input = event.usage.input_tokens,
            output = event.usage.output_tokens,
            reasoning = event.usage.reasoning_tokens,
            cached = event.usage.cached_input_tokens,
            "turn usage"
        );
        self.total
            .lock()
            .expect("usage mutex poisoned")
            .add(TokenUsage {
                input_tokens: event.usage.input_tokens,
                output_tokens: event.usage.output_tokens,
                cached_input_tokens: event.usage.cached_input_tokens,
                cache_creation_input_tokens: event.usage.cache_creation_input_tokens,
                reasoning_tokens: event.usage.reasoning_tokens,
            });
        ModelTurnAction::Continue
    }
}

/// Always-on hook that makes a run legible in the logs: every tool call (name + args), how long it
/// took, and the model's text at the end of each turn. Successful tool calls aren't otherwise
/// surfaced, so without this a stuck node (e.g. an analyst churning through turns) looks like
/// silence.
///
/// The duration is what makes a slow node diagnosable. Without it the only measurable interval is
/// call-to-next-call, which is a tool's own time and the model's next response added together —
/// and the two have completely different fixes.
#[derive(Default)]
struct ObservabilityHook {
    /// Names of the tools this node called. Shared with the [`Meter`], which reports them
    /// alongside the cost.
    used: Arc<Mutex<std::collections::BTreeSet<String>>>,
    /// Start times by rig's correlation id. Entries are removed when the result arrives; a tool
    /// whose result never comes leaves one behind, which is bounded by the turn ceiling.
    started: Mutex<std::collections::HashMap<String, std::time::Instant>>,
}

impl AgentHook for ObservabilityHook {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        if let Ok(mut used) = self.used.lock() {
            used.insert(event.tool_name.to_string());
        }
        if let Ok(mut started) = self.started.lock() {
            started.insert(
                event.internal_call_id.to_string(),
                std::time::Instant::now(),
            );
        }
        tracing::info!(
            kind = "tool_call",
            tool = event.tool_name,
            args = %abridged_args(event.args, 120),
            "tool call"
        );
        ToolCallAction::Run
    }

    async fn on_tool_result(
        &self,
        _ctx: &HookContext,
        event: rig_agent::agent::ToolResultEvent<'_>,
    ) -> ToolResultAction {
        let elapsed = self
            .started
            .lock()
            .ok()
            .and_then(|mut started| started.remove(event.internal_call_id))
            .map(|at| at.elapsed().as_millis() as u64);
        tracing::info!(
            kind = "tool_result",
            tool = event.tool_name,
            duration_ms = elapsed,
            "tool result"
        );
        ToolResultAction::Keep
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        for content in event.content.iter() {
            if let AssistantContent::Text(text) = content {
                let text = text.text.trim();
                // The harness's filler when a turn ends having only called tools; four of five real
                // fabrications carry it, so strip it before recognising the continuation.
                let body = text
                    .strip_prefix("No response requested.")
                    .map(str::trim_start)
                    .unwrap_or(text);
                if !text.is_empty() && !is_transcript_continuation(body) {
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
    shell: Option<&shell::ShellAccess>,
    push: Option<&publish::PushAccess>,
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
                    .map(|tool| local_tool(tool, files, shell, push))
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
fn local_tool(
    tool: &Tool,
    files: Option<&std::path::Path>,
    shell: Option<&shell::ShellAccess>,
    push: Option<&publish::PushAccess>,
) -> DynamicTool {
    if let Some(implemented) = shell.and_then(|s| shell::implementation(&tool.name, s)) {
        return implemented;
    }
    if let Some(implemented) = push.and_then(|p| publish::push_implementation(&tool.name, p)) {
        return implemented;
    }
    if let Some(root) = files {
        if let Some(implemented) = files::implementation(&tool.name, root) {
            return implemented;
        }
        if let Some(implemented) = publish::implementation(&tool.name, root) {
            return implemented;
        }
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

/// Where every node in this process asks what the operator wants.
///
/// Process-wide for the same reason as [`ENDPOINT`], and more strongly: a process runs exactly one
/// run, so "this process" and "this run" are the same thing, and every node in it answers to the
/// same dashboard. Unset is the ordinary case — a run started from the command line has nobody
/// watching it, and nothing to ask.
static CONTROL: std::sync::OnceLock<Arc<dyn Controller>> = std::sync::OnceLock::new();

/// Give this run's nodes somewhere to ask. Called once, before any node runs.
pub fn configure_control(controller: Arc<dyn Controller>) {
    let _ = CONTROL.set(controller);
}

/// How long a held node waits before asking again.
///
/// The dashboard is on loopback and a turn takes seconds, so this is cheap. It is also the worst
/// case for how long "resume" takes to be noticed, which is why it is a second rather than ten.
const CONTROL_POLL: std::time::Duration = std::time::Duration::from_secs(1);

/// What a node is told when the operator stops it. Ends the turn loop; the run then parks.
const STOPPED_BY_OPERATOR: &str = "the operator stopped this node";

/// How the operator's text is labelled where the model reads it.
///
/// Named as a person, and deliberately not as a tool's answer or a repository fact: this is the
/// one channel where imperative text is legitimate, because a human watching the run wrote it on
/// purpose. Everything arriving through a tool is the opposite case — see `PLUGIN_NOTE`.
const OPERATOR_NOTE: &str = "Message from the operator watching this run:";

/// What one node's control state carries between the hook and the run that owns it.
#[derive(Default)]
struct Pending {
    /// Operator text taken from the dashboard but not yet put in front of the model.
    steer: Mutex<Vec<String>>,
    /// Set when the operator stopped this node, so the caller can tell a stop from a failure.
    stopped: std::sync::atomic::AtomicBool,
}

impl Pending {
    /// Take everything waiting, as one labelled block, or `None` if there is nothing to say.
    fn take(&self) -> Option<String> {
        let taken: Vec<String> = std::mem::take(&mut *self.steer.lock().expect("steer poisoned"));
        if taken.is_empty() {
            return None;
        }
        Some(format!("{OPERATOR_NOTE}\n{}", taken.join("\n\n")))
    }
}

/// Applies the operator's pause, stop and steer to one node's turn loop.
///
/// All three act at a turn boundary. Pause holds the loop there, which keeps the conversation and
/// its prompt cache intact so resuming costs only the wait. Stop ends the loop. Steering rides to
/// the model on the next tool result — the same channel a plugin's context uses — because rig
/// refuses to retry a turn that made tool calls, and nearly every turn here makes one.
struct ControlHook {
    node: String,
    controller: Arc<dyn Controller>,
    pending: Arc<Pending>,
}

impl AgentHook for ControlHook {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        loop {
            let control = self.controller.poll(&self.node).await;
            if !control.steer.is_empty() {
                self.pending
                    .steer
                    .lock()
                    .expect("steer poisoned")
                    .extend(control.steer);
            }
            match control.directive {
                Directive::Stop => {
                    self.pending
                        .stopped
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    tracing::info!(kind = "control", node = %self.node, "stopped by the operator");
                    return ModelTurnAction::Stop(STOPPED_BY_OPERATOR.to_string());
                }
                Directive::Hold => tokio::time::sleep(CONTROL_POLL).await,
                Directive::Continue => break,
            }
        }

        // A turn that called no tool has no tool result to ride, and it is also the only kind rig
        // will let us retry. Feeding it back here is what stops a message sitting unread while the
        // node writes its answer.
        let called_tools = event
            .content
            .iter()
            .any(|c| matches!(c, AssistantContent::ToolCall(_)));
        if !called_tools && let Some(text) = self.pending.take() {
            return ModelTurnAction::Retry(RetryRequest::Feedback(text));
        }
        ModelTurnAction::Continue
    }

    async fn on_tool_result(
        &self,
        _ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        // Never on the output tool: that result is the node's own answer being accepted, and
        // there is no turn after it in which the model could act on what it was told.
        if event.tool_name == OUTPUT_TOOL_NAME {
            return ToolResultAction::Keep;
        }
        let Some(text) = self.pending.take() else {
            return ToolResultAction::Keep;
        };
        tracing::info!(kind = "control", node = %self.node, "steering the node");
        let mut content = event.presentation.as_content().clone();
        content.push(rig_core::message::ToolResultContent::text(text));
        ToolResultAction::rewrite_output(ToolOutput::content(content))
    }
}

/// Wait for the operator to start `node` again, keeping anything they say meanwhile.
///
/// Returns the text that arrived while parked, so a message sent to a stopped node reaches the
/// attempt that replaces it rather than being answered by nobody.
async fn park(controller: &Arc<dyn Controller>, node: &str) -> Vec<String> {
    let mut said = Vec::new();
    loop {
        let control = controller.poll(node).await;
        said.extend(control.steer);
        if control.directive != Directive::Stop {
            return said;
        }
        tokio::time::sleep(CONTROL_POLL).await;
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

    /// Claim the oldest unclaimed entry for `node`, if it made a model turn at all. `None` is
    /// ordinary for a node that ran no model — it means "nothing to report", not "something went
    /// missing", which is what [`RunLedger::unclaimed`] is for.
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
    /// A key naming this node's conversation within the run, when the route asks for the endpoint
    /// session to be reused across attempts.
    ///
    /// Only consulted for `SessionScope::Reuse`; ignored otherwise, so a node that supplies one is
    /// not thereby opting into reuse — the route decides that.
    pub conversation: Option<&'a str>,
    /// The sandbox the node's `Bash` calls run in; `None` for a node that runs no commands.
    ///
    /// Separate from `files` because they are different powers: reading and editing a tree is not
    /// running code in it, and only the node that has to build and test its own work is given the
    /// second.
    pub shell: Option<shell::ShellAccess>,
    /// The branch this node may push, if any. Only the publisher is given one.
    pub push: Option<publish::PushAccess>,
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
/// Like [`ask`], but the agent is given an `output_schema`, so its final answer is the structured
/// JSON matching that schema. Returns the raw output string — best-effort, so the caller must
/// still validate it (see `ratatoskr_graph::parse_validated`), though a first answer that misses
/// the schema is handed back for correction before it reaches one.
pub async fn run_structured(run: NodeRun<'_>) -> Result<String, AgentError> {
    match parse_provider(&run.route.provider)? {
        Provider::Anthropic => {
            let client = anthropic_client(&session_id(&run))?;
            let model = caching(client.completion_model(&run.route.model));
            run_typed(model, run).await
        }
        Provider::OpenAi => {
            let model = openai_client()?.completion_model(&run.route.model);
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
        shell,
        push,
        conversation,
        ledger,
        produces,
    } = run;
    let control = CONTROL.get();
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

    let (builder, meter) = metered(model, &preamble, max_turns, Request::of(route));
    // Kept for the validation below: the builder consumes the schema, and a node's answer has to
    // be checked against it here, where the agent that wrote it can still be asked to fix it.
    let schema_value = serde_json::to_value(&output_schema).unwrap_or(serde_json::Value::Null);
    let mut builder = builder
        .output_schema_raw(output_schema)
        // Force the synthetic output-tool: Auto can resolve to native structured output, which
        // Anthropic rejects when combined with tools ("output_config.format: Cannot be combined
        // with tools"). Tool mode sends no native format and composes with the rag-rat tools.
        .output_mode(OutputMode::Tool);
    // First of the gates: what the operator asked for outranks anything the conversation is in the
    // middle of, and a node being stopped should not spend a turn on the hooks below it.
    let pending = Arc::new(Pending::default());
    if let Some(controller) = control {
        builder = builder.add_hook(ControlHook {
            node: node.to_string(),
            controller: Arc::clone(controller),
            pending: Arc::clone(&pending),
        });
    }
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
    // Before the set is handed to the agent: what the model could call is part of what this turn
    // was, and a reader of the run cannot reconstruct it from a config that has since changed.
    let tool_names = tools.names();
    let agent = bind_tools(
        builder,
        &tools,
        files.as_deref(),
        shell.as_ref(),
        push.as_ref(),
    );

    // Tag every log line for this run — the hook's tool-call/text lines and rig-agent's own turn
    // logs — with the node name, so a run's agents are distinguishable in the logs. `prompt()` is
    // IntoFuture (not Future), so instrument the awaiting block rather than the request.
    // Field names follow the OpenTelemetry GenAI semantic conventions. Those are still unstable and
    // not worth an SDK dependency yet, but naming to match now makes adopting one a layer swap
    // rather than a rename of every field a dashboard reads.
    let span = tracing::info_span!(
        "agent",
        node,
        "gen_ai.operation.name" = "invoke_agent",
        "gen_ai.agent.name" = node,
        "gen_ai.request.model" = %model_name,
    );
    // Announced at the start, because a checkpoint only exists once the node has finished — and
    // the moment a reader most wants to know what a node is running on is while it is still
    // running. The facts here are the configured ones; cost arrives with the checkpoint.
    tracing::info!(
        kind = "node_start",
        node,
        model = %model_name,
        tools = %tool_names.join(","),
        thinking = thinking_left_on(route),
        reuses_session = matches!(route.session, ratatoskr_core::SessionScope::Reuse)
            && conversation.is_some(),
        "node started"
    );
    let started = std::time::Instant::now();
    // A node the operator stopped has not failed, and its work is not thrown away: the run parks
    // here until they start it again, then runs the node afresh on the same question — which is
    // the one its checkpoints hold, so "start from checkpoint" is exactly what a new attempt is.
    // The wait is inside the node's own duration, because from the run's side that is what
    // happened: the node was still the thing in progress.
    let mut answer = loop {
        let attempt = async { agent.prompt(&question).await }
            .instrument(span.clone())
            .await
            .map_err(|e| AgentError::Prompt(e.to_string()));
        let stopped = pending
            .stopped
            .swap(false, std::sync::atomic::Ordering::SeqCst);
        match (control, stopped) {
            (Some(controller), true) => {
                tracing::info!(
                    kind = "control",
                    node,
                    "parked; waiting to be started again"
                );
                let said = park(controller, node).await;
                // Anything the operator said to the stopped node belongs to the attempt that
                // replaces it — they were talking about this work, not the abandoned transcript.
                pending.steer.lock().expect("steer poisoned").extend(
                    said.into_iter().chain([
                        "This node was stopped and started again. Its previous conversation is \
                         gone; you are running from the beginning."
                            .to_string(),
                    ]),
                );
            }
            _ => break attempt,
        }
    };

    // One retry when the call never reached a verdict. A dropped connection is not an answer, and
    // it cost a live run twenty minutes of implementer work at the last node before the diff: the
    // request went out, the proxy in front of the API closed it, and the run failed holding a
    // worktree full of finished edits. Retrying costs another attempt; not retrying costs all of
    // the attempt already made.
    //
    // Transport only. A refusal, a bad request or an exhausted turn budget will answer the same
    // way twice, and retrying those spends a node's budget to arrive back where it started.
    if let Err(e) = &answer
        && is_transport_error(&e.to_string())
    {
        tracing::warn!(
            node,
            "the model call failed in transport, retrying once: {e}"
        );
        answer = async { agent.prompt(&question).await }
            .instrument(span.clone())
            .await
            .map_err(|e| AgentError::Prompt(e.to_string()));
    }
    let answer = answer;

    // Give a malformed answer back to the agent that wrote it, once. The alternative is what a
    // schema failure used to cost: the node's whole run discarded — every tool call, every file
    // read, minutes of it — over a key in the wrong shape, which is the one kind of mistake a
    // model corrects immediately when told. The correction is a fresh short prompt rather than a
    // continuation, so the preamble and tools stay cached and the transcript does not grow.
    let answer = match &answer {
        Ok(raw) => match ratatoskr_graph::validate_raw(raw, &schema_value) {
            Ok(_) => answer,
            Err(invalid) => {
                tracing::warn!(node, %invalid, "output failed its schema; asking for a correction");
                let correction = format!(
                    "Your answer did not match the schema you were given: {invalid}\n\n\
                     Here is what you returned:\n{raw}\n\n\
                     Return the same content corrected to match the schema. Change only what the \
                     error names — keep every finding, do not shorten anything, and do not go and \
                     look anything up again. Answer by calling the output tool.",
                );
                async { agent.prompt(&correction).await }
                    .instrument(span)
                    .await
                    .map_err(|e| AgentError::Prompt(e.to_string()))
            }
        },
        Err(_) => answer,
    };

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
            tools: tool_names,
            tools_used: meter.used(),
            reuses_session: matches!(route.session, ratatoskr_core::SessionScope::Reuse)
                && conversation.is_some(),
            thinking: thinking_left_on(route),
        };
        tracing::info!(
            kind = "usage",
            node,
            "gen_ai.usage.input_tokens" = telemetry.usage.input_tokens,
            "gen_ai.usage.output_tokens" = telemetry.usage.output_tokens,
            "gen_ai.usage.cached_input_tokens" = telemetry.usage.cached_input_tokens,
            "gen_ai.usage.cache_creation_input_tokens" =
                telemetry.usage.cache_creation_input_tokens,
            "gen_ai.usage.reasoning_tokens" = telemetry.usage.reasoning_tokens,
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
    /// The operator controls, at the two points where this crate makes a decision of its own:
    /// waiting for a stopped node to be started, and holding what was said until the model can be
    /// told. The rules about what a command *means* are `ratatoskr_core::control`'s, and tested
    /// there.
    mod control {
        use std::sync::Mutex;

        use ratatoskr_core::{Control, Directive};

        use super::super::*;

        /// Answers with a scripted sequence, then `Continue` forever.
        struct Scripted(Mutex<std::vec::IntoIter<Control>>);

        impl Scripted {
            fn answering(answers: Vec<Control>) -> Arc<dyn Controller> {
                Arc::new(Scripted(Mutex::new(answers.into_iter())))
            }
        }

        impl Controller for Scripted {
            fn poll<'a>(
                &'a self,
                _node: &'a str,
            ) -> Pin<Box<dyn Future<Output = Control> + Send + 'a>> {
                let next = self.0.lock().expect("script poisoned").next();
                Box::pin(async move { next.unwrap_or_default() })
            }
        }

        fn stop() -> Control {
            Control {
                directive: Directive::Stop,
                steer: Vec::new(),
            }
        }

        #[tokio::test(start_paused = true)]
        async fn parking_ends_when_the_node_is_started_again() {
            // Two more polls saying "still stopped" before the operator presses play. Time is
            // paused, so the waits between them cost the test nothing.
            let controller = Scripted::answering(vec![stop(), stop(), Control::carry_on()]);
            assert!(park(&controller, "implementer").await.is_empty());
        }

        #[tokio::test(start_paused = true)]
        async fn what_was_said_to_a_stopped_node_survives_the_wait() {
            // The operator stops a node, says something to it, then starts it. The message is
            // about the work, not about the abandoned conversation, so the attempt that replaces
            // it must be the one that hears it.
            let controller = Scripted::answering(vec![
                stop(),
                Control {
                    directive: Directive::Stop,
                    steer: vec!["use the existing helper".to_string()],
                },
                Control::carry_on(),
            ]);
            assert_eq!(
                park(&controller, "implementer").await,
                ["use the existing helper"]
            );
        }

        #[test]
        fn text_is_labelled_as_a_person_and_handed_over_once() {
            let pending = Pending::default();
            pending.steer.lock().expect("steer poisoned").extend([
                "look at the ruleset".to_string(),
                "and the gate".to_string(),
            ]);

            let taken = pending.take().expect("something to say");
            assert!(taken.starts_with(OPERATOR_NOTE));
            assert!(taken.contains("look at the ruleset"));
            assert!(taken.contains("and the gate"));
            // Twice would put the operator's words in front of the model on every later tool
            // result, reading as them saying it again and again.
            assert!(pending.take().is_none());
        }
    }

    #[test]
    fn sanitize_strips_tag_and_zero_width_but_keeps_ordinary_text() {
        // Unchanged when there is nothing to strip.
        assert_eq!(super::sanitize("tests passed"), "tests passed");
        // A tag char between visible ones is removed, the visible ones intact.
        assert_eq!(super::sanitize(&format!("a{}b", '\u{E0041}')), "ab");
        // Zero-width space and BOM/ZWNBSP removed, the rest intact.
        assert_eq!(
            super::sanitize(&format!("x{}y{}z", '\u{200B}', '\u{FEFF}')),
            "xyz"
        );
        // Legitimate non-ASCII is byte-identical.
        let text = "café — 日本語";
        assert_eq!(super::sanitize(text), text);
        // Edge cases: empty, and all-invisible.
        assert_eq!(super::sanitize(""), "");
        assert_eq!(
            super::sanitize(&format!("{}{}", '\u{E0041}', '\u{200B}')),
            ""
        );
    }

    #[test]
    fn a_fabricated_tool_result_is_recognised_as_transcript_continuation() {
        // The hardest real shape (issue #126, run 6fbb7f25): a reproduced `Read` result anchored
        // on one true line, then diverging into invented file contents — same gutter, same
        // truncation format, indistinguishable from a real result by eye.
        let fabricated = "[your Read crates/ratatoskr-cli/src/main.rs]:\n\
   169\t    /// Without `--force` it only lists what would go. Deletion takes the run's checkpoints and its\n\
   170\t    /// events; the run row and its provenance are kept, so a re-import can restore it.\n\
   171\t    Prune {\n\
   172\t        /// Runs to delete. Defaults to nothing — you must name at least one.\n\
   173\t        run_ids: Vec<String>,";
        assert!(
            super::is_transcript_continuation(fabricated),
            "the anchor-then-diverge Read block must be caught"
        );

        // The other-side turn injected mid-string.
        assert!(super::is_transcript_continuation(
            "some preamble user[your Grep model_text]: matches"
        ));

        // Ordinary analyst prose is genuine output and must be kept.
        assert!(
            !super::is_transcript_continuation(
                "The Rm variant deletes checkpoints; I'll plan from the real file."
            ),
            "genuine prose must be classified as genuine"
        );

        // Edge cases the anchor must not trip on.
        assert!(!super::is_transcript_continuation(""));
        assert!(!super::is_transcript_continuation(
            "your change looks correct"
        ));
        assert!(
            !super::is_transcript_continuation("   169\t"),
            "a bare gutter fragment without the bracket header is not enough"
        );
    }

    #[test]
    fn an_edits_arguments_survive_being_bounded() {
        // The dashboard identifies a call by `file_path`. Truncating the serialized JSON cut
        // `Edit` mid-`old_string`, leaving text that would not parse — so the feed showed no
        // argument at all for the one tool whose subject a reader most wants to see.
        let long = "x".repeat(4000);
        let raw = format!(
            r#"{{"file_path":"crates/foo/src/lib.rs","old_string":"{long}","new_string":"{long}"}}"#
        );
        let out = super::abridged_args(&raw, 120);

        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("stays valid JSON, which is the whole point");
        assert_eq!(parsed["file_path"], "crates/foo/src/lib.rs");
        assert!(
            parsed["old_string"].as_str().unwrap().chars().count() <= 121,
            "the long value is still bounded"
        );
        assert!(
            out.len() < 500,
            "the log line stays readable: {}",
            out.len()
        );
    }

    #[test]
    fn what_a_run_did_is_kept_whole_and_what_it_wrote_is_not() {
        // The distinction the log has to make: a command or a pattern exists nowhere else, so the
        // account of what the run did is only as good as what is written here. File contents are
        // reproducible from the branch the run worked on, so keeping them would duplicate a better
        // record at the cost of multiplying the log by the size of the files touched.
        let long = "x".repeat(3000);
        let raw = format!(
            r#"{{"command":"cargo test --workspace -- --nocapture {long}",
                 "file_path":"crates/foo.rs","old_string":"{long}"}}"#
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&super::abridged_args(&raw, 120)).unwrap();

        assert!(
            parsed["command"].as_str().unwrap().len() > 3000,
            "the command that ran is kept whole"
        );
        assert_eq!(parsed["file_path"], "crates/foo.rs");
        assert!(
            parsed["old_string"].as_str().unwrap().chars().count() <= 121,
            "file contents are abridged: git has them"
        );
    }

    #[test]
    fn arguments_that_are_not_json_are_still_bounded() {
        let out = super::abridged_args(&"y".repeat(9000), 120);
        assert!(out.chars().count() <= 201, "{}", out.chars().count());
    }

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
        assert!(parse_provider("openai").is_ok());
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
            reasoning_tokens: 700,
        });
        total.add(TokenUsage {
            input_tokens: 5,
            output_tokens: 3,
            reasoning_tokens: 300,
            ..Default::default()
        });
        assert_eq!(total.input_tokens, 15);
        assert_eq!(total.output_tokens, 4);
        assert_eq!(total.cached_input_tokens, 8);
        assert_eq!(total.cache_creation_input_tokens, 2);
        // Thinking accumulates like the rest: a turn that thought and called one tool spent most
        // of what it spent here, and a sum that drops it says the node was nearly free.
        assert_eq!(total.reasoning_tokens, 1_000);
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
            temperature: None,
            params: None,
            session: Default::default(),
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
            shell: None,
            push: None,
            conversation: None,
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
            temperature: None,
            params: None,
            session: Default::default(),
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
            shell: None,
            push: None,
            conversation: None,
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
    fn a_reused_session_is_stable_across_attempts_and_a_fresh_one_is_not() {
        // What re-driving a node costs when it is wrong: a converge iteration that starts a new
        // session meets the worktree it just edited for the first time again.
        let route = |session| ModelRoute {
            provider: "anthropic".into(),
            model: "claude-opus-4-8".into(),
            max_tokens: None,
            temperature: None,
            params: None,
            session,
        };
        fn run<'a>(route: &'a ModelRoute, conversation: Option<&'a str>) -> NodeRun<'a> {
            NodeRun {
                node: "implementer",
                route,
                preamble: "",
                question: "",
                tools: ratatoskr_mcp::ToolSet::default(),
                output_schema: schemars::schema_for!(String),
                policy: None,
                max_turns: None,
                clarifier: None,
                observer: None,
                skills: Vec::new(),
                files: None,
                shell: None,
                push: None,
                conversation,
                ledger: None,
                produces: None,
            }
        }

        let reuse = route(ratatoskr_core::SessionScope::Reuse);
        let key = Some("run-7-implementer");
        assert_eq!(session_id(&run(&reuse, key)), session_id(&run(&reuse, key)));

        // Fresh is the default, and two attempts never share one.
        let fresh = route(ratatoskr_core::SessionScope::Fresh);
        assert_ne!(session_id(&run(&fresh, key)), session_id(&run(&fresh, key)));

        // Reuse without a key has nothing to be stable about, so it does not pretend otherwise.
        assert_ne!(
            session_id(&run(&reuse, None)),
            session_id(&run(&reuse, None))
        );
    }

    #[test]
    fn the_endpoint_is_told_who_is_calling_and_which_conversation_this_is() {
        // What reaches the wire decides how the far side treats us. An endpoint that adapts per
        // client defaults an unrecognised one to somebody else's adapter, and one that tracks
        // sessions rebuilds the conversation every turn without an id to match it to.
        let mut headers = std::collections::HashMap::new();
        headers.insert("x-meridian-agent".to_string(), "passthrough".to_string());
        headers.insert("not a header".to_string(), "dropped".to_string());
        let cfg = ratatoskr_core::EndpointConfig {
            headers,
            session_header: Some("x-litellm-session-id".to_string()),
        };

        let sent = endpoint_headers(Some(&cfg), "session-abc");
        assert_eq!(sent.get("x-meridian-agent").unwrap(), "passthrough");
        assert_eq!(sent.get("x-litellm-session-id").unwrap(), "session-abc");
        // A typo costs its own header and nothing else — not the run.
        assert_eq!(sent.len(), 2);

        // Unconfigured is the default, and sends nothing of its own.
        assert!(endpoint_headers(None, "session-abc").is_empty());
    }

    #[test]
    fn only_a_call_that_never_landed_is_worth_retrying() {
        // The one that cost a live run: the request went out, the proxy in front of the API closed
        // it, and the node failed holding a worktree full of finished edits.
        assert!(is_transport_error(
            "CompletionError: HttpError: Http client error: error sending request for url \
             (http://127.0.0.1:3456/v1/messages)"
        ));
        assert!(is_transport_error("connection reset by peer"));

        // And the ones that will answer the same way twice, where a retry spends a node's budget
        // to arrive back where it started.
        assert!(!is_transport_error("MaxTurnsError: reached 100 turns"));
        assert!(!is_transport_error(
            "ProviderError: invalid_request_error: max_tokens is too large"
        ));
        assert!(!is_transport_error("output failed schema validation"));
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

    // Recognising a `model_text` that is the model writing the *other side* of the conversation
    // (issue #126). The interface leaves the predicate's exact name to the implementer but fixes
    // its shape: a free `fn(&str) -> bool`. These tests are written against the name given in the
    // contract, `is_transcript_continuation`; if the implementer chooses a different name, only the
    // reference here needs to follow it — the cases are the contract.

    /// The real fabricated block from run `6fbb7f25`, analyst turn 15: a `[your Read …]:` header
    /// anchored on one true line and diverging into invented file contents, reproduced with the
    /// `Read` tool's own gutter (`   169\t`). This is the shape the acceptance says a future change
    /// must not quietly stop catching.
    #[test]
    fn a_fabricated_read_block_is_recognised_as_transcript_continuation() {
        let fabricated = "[your Read crates/ratatoskr-cli/src/main.rs]:\n\
             \x20  169\t    /// Without `--force` it only lists what would go. Deletion takes the run's checkpoints and its\n\
             \x20  170\t    /// events; the run row and its provenance are kept, so a re-import can restore it.\n\
             \x20  171\t    Prune {\n\
             \x20  172\t        /// Runs to delete. Defaults to nothing — you must name at least one.\n\
             \x20  173\t        run_ids: Vec<String>,";
        assert!(is_transcript_continuation(fabricated));
    }

    /// The marker can appear mid-string rather than at the very start.
    #[test]
    fn an_embedded_user_your_tool_marker_is_recognised() {
        assert!(is_transcript_continuation(
            "Let me check the logs. user[your Grep model_text]: 5 matches found"
        ));
    }

    /// The caller strips the `No response requested.` filler before calling the predicate, so what
    /// the predicate sees begins with the `[your <tool> …]:` header. It must still recognise it.
    #[test]
    fn after_the_filler_is_stripped_a_tool_header_is_still_recognised() {
        // The filler has already been removed by the logging site; this is the residue.
        let after_strip = "[your Grep model_text]:\ncrates/ratatoskr-agent/src/lib.rs";
        assert!(is_transcript_continuation(after_strip));
    }

    /// Genuine model prose — even prose that talks about the same subject matter — is the node's
    /// own output and must be kept.
    #[test]
    fn genuine_analyst_prose_is_not_a_continuation() {
        assert!(!is_transcript_continuation(
            "The Rm variant deletes checkpoints; I'll plan from the real file."
        ));
    }

    /// The empty string never produces an event, and the predicate must handle it without panic.
    #[test]
    fn an_empty_string_is_not_a_continuation() {
        assert!(!is_transcript_continuation(""));
    }

    /// The anchor is the bracketed `[your <tool> …]:` form, not the bare word `your`. Ordinary
    /// prose that mentions the word must not be mistaken for a transcript.
    #[test]
    fn the_bare_word_your_is_not_the_bracket_anchor() {
        assert!(!is_transcript_continuation("your change looks correct"));
    }

    /// A line that is only the tab-gutter fragment, with no `[your …]:` header above it, is code
    /// that happens to contain gutter-shaped text — anchoring on the bracket header avoids that
    /// false positive.
    #[test]
    fn a_lone_gutter_fragment_without_a_header_is_not_a_continuation() {
        assert!(!is_transcript_continuation("   169\t"));
    }
}
