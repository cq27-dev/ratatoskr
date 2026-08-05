//! Builds a `rig` agent bound to a model and rag-rat's MCP tools, and runs one prompt.
//!
//! Phase 1 has exactly one caller (`ratatoskr ask`), so provider resolution is a small `match`
//! rather than a registry. The agent's own multi-turn loop (from `rig-agent`) does the tool
//! calling — we hand it the tools and a client handle via `.rmcp_tools()`.

use std::sync::Arc;

use ratatoskr_core::{ModelRoute, ToolDecision, ToolPolicy};
use rig_agent::AgentBuilder;
use rig_agent::agent::{AgentHook, HookContext, OutputMode, ToolCall, ToolCallAction};
use rig_agent::completion::Prompt;
use rig_core::client::completion::CompletionClient;
use rig_core::client::{ProviderClient, ProviderClientError};
use rig_core::completion::CompletionModel;
use rig_core::providers::{anthropic, moonshot};
use rmcp::model::Tool;
use rmcp::service::ServerSink;

/// How many tool-calling turns the agent may take before it must produce a final answer.
const DEFAULT_MAX_TURNS: usize = 10;

/// rig-agent's default name for the synthetic structured-output tool (`OutputMode::Tool`). Kept in
/// sync with `rig_agent`'s `DEFAULT_OUTPUT_TOOL_NAME`; a ruleset must not be able to deny it.
const OUTPUT_TOOL_NAME: &str = "final_result";

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

/// Ask one question, letting the agent call rag-rat's `tools` (via `sink`) to answer.
///
/// `preamble` is the system prompt; `route` picks the provider/model. Returns the agent's final
/// text after its tool-calling loop settles.
pub async fn ask(
    route: &ModelRoute,
    preamble: &str,
    question: &str,
    tools: Vec<Tool>,
    sink: ServerSink,
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
                sink,
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
                sink,
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
    tools: Vec<Tool>,
    sink: ServerSink,
) -> Result<String, AgentError>
where
    M: CompletionModel + 'static,
{
    let agent = AgentBuilder::new(model)
        .preamble(preamble)
        .default_max_turns(DEFAULT_MAX_TURNS)
        .rmcp_tools(tools, sink)
        .build();

    agent
        .prompt(question)
        .await
        .map_err(|e| AgentError::Prompt(e.to_string()))
}

/// Like [`ask`], but the agent is given an `output_schema`, so its final answer is the structured
/// JSON matching that schema (rig's `OutputMode::Auto` resolves to a synthetic output tool that
/// composes with the rag-rat tools). Returns the raw output string — best-effort, so the caller
/// must still validate it (see `ratatoskr_graph::parse_validated`).
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

#[allow(clippy::too_many_arguments)] // route + preamble + question + tools + sink + schema + policy + max_turns are all inherent
pub async fn run_structured(
    route: &ModelRoute,
    preamble: &str,
    question: &str,
    tools: Vec<Tool>,
    sink: ServerSink,
    output_schema: schemars::Schema,
    policy: Option<Arc<dyn ToolPolicy>>,
    max_turns: Option<usize>,
) -> Result<String, AgentError> {
    match parse_provider(&route.provider)? {
        Provider::Anthropic => {
            let client = anthropic::Client::from_env().map_err(|source| AgentError::Provider {
                provider: "anthropic".to_string(),
                source,
            })?;
            run_typed(
                client.completion_model(&route.model),
                preamble,
                question,
                tools,
                sink,
                output_schema,
                policy,
                max_turns,
            )
            .await
        }
        Provider::Moonshot => {
            let client = moonshot::Client::from_env().map_err(|source| AgentError::Provider {
                provider: "moonshot".to_string(),
                source,
            })?;
            run_typed(
                client.completion_model(&route.model),
                preamble,
                question,
                tools,
                sink,
                output_schema,
                policy,
                max_turns,
            )
            .await
        }
    }
}

/// Structured variant of [`run`]: sets an output schema so the final answer is structured JSON.
#[allow(clippy::too_many_arguments)] // mirrors run_structured's inherent parameter list
async fn run_typed<M>(
    model: M,
    preamble: &str,
    question: &str,
    tools: Vec<Tool>,
    sink: ServerSink,
    output_schema: schemars::Schema,
    policy: Option<Arc<dyn ToolPolicy>>,
    max_turns: Option<usize>,
) -> Result<String, AgentError>
where
    M: CompletionModel + 'static,
{
    let mut builder = AgentBuilder::new(model)
        .preamble(preamble)
        .default_max_turns(max_turns.unwrap_or(DEFAULT_MAX_TURNS))
        .output_schema_raw(output_schema)
        // Force the synthetic output-tool: Auto can resolve to native structured output, which
        // Anthropic rejects when combined with tools ("output_config.format: Cannot be combined
        // with tools"). Tool mode sends no native format and composes with the rag-rat tools.
        .output_mode(OutputMode::Tool)
        .rmcp_tools(tools, sink);
    if let Some(policy) = policy {
        builder = builder.add_hook(RulesetHook { policy });
    }
    let agent = builder.build();

    agent
        .prompt(question)
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
