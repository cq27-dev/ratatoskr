//! Client to rag-rat's MCP server, over a stdio subprocess.
//!
//! [`RagRatClient::connect`] spawns rag-rat (per [`RagRatConfig::command`]), performs the MCP
//! handshake, and lists the server's tools. It hands back the tool list plus a [`ServerSink`]
//! (a cloneable client handle) — everything `ratatoskr-agent` needs to bind those tools to a
//! `rig` agent. The running service is held so the subprocess stays alive; drop or
//! [`shutdown`](RagRatClient::shutdown) tears it down.

use ratatoskr_core::RagRatConfig;
use rmcp::ServiceExt;
use rmcp::model::Tool;
use rmcp::service::{RoleClient, RunningService, ServerSink};
use rmcp::transport::TokioChildProcess;

/// Errors connecting to or talking with rag-rat's MCP server.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("rag_rat.command is empty; set it in ratatoskr.toml")]
    EmptyCommand,
    #[error(
        "failed to launch rag-rat (`{program}`): {source}. \
         Is rag-rat installed? Check `rag_rat.command` in ratatoskr.toml."
    )]
    Spawn {
        program: String,
        source: std::io::Error,
    },
    #[error("MCP handshake with rag-rat failed: {0}")]
    Handshake(String),
    #[error("listing rag-rat's tools failed: {0}")]
    ListTools(String),
    #[error("shutting down rag-rat failed: {0}")]
    Shutdown(String),
}

/// A live connection to rag-rat's MCP server. Holds the running service so the subprocess stays up.
pub struct RagRatClient {
    service: RunningService<RoleClient, ()>,
    tools: Vec<Tool>,
}

impl RagRatClient {
    /// Spawn rag-rat, complete the MCP handshake, and list its tools.
    pub async fn connect(config: RagRatConfig) -> Result<Self, McpError> {
        let (program, args) = config.command.split_first().ok_or(McpError::EmptyCommand)?;

        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args);
        if let Some(dir) = &config.working_dir {
            cmd.current_dir(dir);
        }

        // `new` spawns the child; a missing program surfaces here as the most common failure mode.
        let transport = TokioChildProcess::new(cmd).map_err(|source| McpError::Spawn {
            program: program.clone(),
            source,
        })?;

        // `()` is rmcp's no-op client handler — we drive the server, it makes no callbacks to us.
        let service = ().serve(transport).await.map_err(|e| McpError::Handshake(e.to_string()))?;

        let tools = service
            .peer()
            .list_all_tools()
            .await
            .map_err(|e| McpError::ListTools(e.to_string()))?;

        tracing::info!(
            tool_count = tools.len(),
            tools = ?tools.iter().map(|t| t.name.as_ref()).collect::<Vec<_>>(),
            "connected to rag-rat MCP server"
        );

        Ok(RagRatClient { service, tools })
    }

    /// The tools rag-rat exposes, for handing to `ratatoskr-agent`.
    pub fn tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    /// A cloneable client handle the agent uses to call rag-rat's tools.
    pub fn sink(&self) -> ServerSink {
        self.service.peer().clone()
    }

    /// Cleanly cancel the connection and tear down the subprocess.
    pub async fn shutdown(self) -> Result<(), McpError> {
        self.service
            .cancel()
            .await
            .map(|_| ())
            .map_err(|e| McpError::Shutdown(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CI-safe: connecting with a command that doesn't exist fails fast as a clear spawn error,
    /// no real rag-rat binary required. Exercises the connect path's error mapping.
    #[tokio::test]
    async fn connect_to_missing_binary_is_a_spawn_error() {
        let config = RagRatConfig {
            command: vec![
                "definitely-not-a-real-binary-xyz".to_string(),
                "mcp".to_string(),
            ],
            working_dir: None,
        };
        assert!(matches!(
            RagRatClient::connect(config).await,
            Err(McpError::Spawn { .. })
        ));
    }

    #[tokio::test]
    async fn empty_command_is_rejected() {
        let config = RagRatConfig {
            command: vec![],
            working_dir: None,
        };
        assert!(matches!(
            RagRatClient::connect(config).await,
            Err(McpError::EmptyCommand)
        ));
    }
}
