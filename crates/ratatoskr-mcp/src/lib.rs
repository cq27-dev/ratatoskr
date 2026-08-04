//! Client to rag-rat's MCP server.
//!
//! Phase 0 is a deliberate stub: [`RagRatClient::connect`] returns [`McpError::NotImplemented`]
//! rather than spawning a subprocess. The `rmcp` dependency is pinned so the tree resolves in CI;
//! the real `rmcp::TokioChildProcess`-backed client lands in Phase 1, built against rag-rat's
//! actual tool surface rather than a guess at it.

use ratatoskr_core::RagRatConfig;

/// Errors from the rag-rat MCP client.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("rag-rat MCP client is not implemented until Phase 1")]
    NotImplemented,
}

/// A handle to the rag-rat MCP server. Empty in Phase 0; wraps the `rmcp` client from Phase 1 on.
pub struct RagRatClient {
    _private: (),
}

impl RagRatClient {
    /// Connect to rag-rat's MCP server over stdio. Not implemented until Phase 1.
    pub async fn connect(_config: RagRatConfig) -> Result<Self, McpError> {
        Err(McpError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn connect_is_not_implemented_yet() {
        let config = RagRatConfig {
            command: vec![
                "rag-rat".to_string(),
                "mcp".to_string(),
                "serve".to_string(),
            ],
            working_dir: None,
        };
        assert!(matches!(
            RagRatClient::connect(config).await,
            Err(McpError::NotImplemented)
        ));
    }
}
