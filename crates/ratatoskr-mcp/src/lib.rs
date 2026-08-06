//! Clients to the MCP servers a run talks to, each over a stdio subprocess.
//!
//! [`Connection::spawn`] launches a server, performs the MCP handshake, and lists its tools. It
//! hands back the tool list plus a [`ServerSink`] (a cloneable client handle) — everything
//! `ratatoskr-agent` needs to bind those tools to a `rig` agent. The running service is held so the
//! subprocess stays alive; drop or [`shutdown`](Connection::shutdown) tears it down.
//!
//! rag-rat is the server ratatoskr is built around and is connected from config
//! ([`RagRatClient`]); a plugin's servers are connected alongside it. A node's tools can therefore
//! come from several servers at once, which is what [`ToolSet`] carries: tools grouped by the
//! server each is dispatched to.

use std::collections::BTreeMap;
use std::path::Path;

use ratatoskr_core::RagRatConfig;
use rmcp::ServiceExt;
use rmcp::model::Tool;
use rmcp::service::{RoleClient, RunningService, ServerSink};
use rmcp::transport::TokioChildProcess;

/// The origin name of the connection made from `[rag_rat]` in the config.
///
/// Also what a plugin-declared server is checked against: the rag-rat plugin declares the same
/// server ratatoskr already launches, and it must not be spawned twice.
pub const RAG_RAT: &str = "rag-rat";

/// Errors connecting to or talking with an MCP server.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("rag_rat.command is empty; set it in ratatoskr.toml")]
    EmptyCommand,
    #[error(
        "failed to launch MCP server `{origin}` (`{program}`): {source}. Is it installed? \
         rag-rat's launch command is `rag_rat.command` in ratatoskr.toml; a plugin's is in its \
         manifest."
    )]
    Spawn {
        origin: String,
        program: String,
        source: std::io::Error,
    },
    #[error("MCP handshake with `{origin}` failed: {detail}")]
    Handshake { origin: String, detail: String },
    #[error("listing `{origin}`'s tools failed: {detail}")]
    ListTools { origin: String, detail: String },
    #[error("shutting down `{origin}` failed: {detail}")]
    Shutdown { origin: String, detail: String },
}

/// A live connection to one MCP server. Holds the running service so the subprocess stays up.
pub struct Connection {
    origin: String,
    service: RunningService<RoleClient, ()>,
    tools: Vec<Tool>,
}

impl Connection {
    /// Spawn a server, complete the MCP handshake, and list its tools.
    ///
    /// `origin` names the connection in logs and in a [`ToolSet`] — for rag-rat, [`RAG_RAT`]; for
    /// a plugin's server, the name it was declared under.
    pub async fn spawn(
        origin: &str,
        command: &[String],
        env: &BTreeMap<String, String>,
        working_dir: Option<&Path>,
    ) -> Result<Self, McpError> {
        let (program, args) = command.split_first().ok_or(McpError::EmptyCommand)?;

        let mut cmd = tokio::process::Command::new(program);
        cmd.args(args);
        cmd.envs(env);
        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        // `new` spawns the child; a missing program surfaces here as the most common failure mode.
        let transport = TokioChildProcess::new(cmd).map_err(|source| McpError::Spawn {
            origin: origin.to_string(),
            program: program.clone(),
            source,
        })?;

        // `()` is rmcp's no-op client handler — we drive the server, it makes no callbacks to us.
        let service = ().serve(transport).await.map_err(|e| McpError::Handshake {
            origin: origin.to_string(),
            detail: e.to_string(),
        })?;

        let tools = service
            .peer()
            .list_all_tools()
            .await
            .map_err(|e| McpError::ListTools {
                origin: origin.to_string(),
                detail: e.to_string(),
            })?;

        tracing::info!(
            server = origin,
            tool_count = tools.len(),
            tools = ?tools.iter().map(|t| t.name.as_ref()).collect::<Vec<_>>(),
            "connected to MCP server"
        );

        Ok(Connection {
            origin: origin.to_string(),
            service,
            tools,
        })
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// The tools this server exposes, for handing to `ratatoskr-agent`.
    pub fn tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    /// A cloneable client handle for calling this server's tools.
    pub fn sink(&self) -> ServerSink {
        self.service.peer().clone()
    }

    /// This server's whole offer, as one group of a [`ToolSet`].
    pub fn offer(&self) -> ServerTools {
        ServerTools {
            origin: self.origin.clone(),
            sink: self.sink(),
            tools: self.tools.clone(),
        }
    }

    /// Cleanly cancel the connection and tear down the subprocess.
    pub async fn shutdown(self) -> Result<(), McpError> {
        let origin = self.origin;
        self.service
            .cancel()
            .await
            .map(|_| ())
            .map_err(|e| McpError::Shutdown {
                origin,
                detail: e.to_string(),
            })
    }
}

/// The connection to rag-rat, which every run has and every node can call.
pub struct RagRatClient(Connection);

impl RagRatClient {
    /// Spawn rag-rat per `[rag_rat]`, complete the handshake, and list its tools.
    pub async fn connect(config: RagRatConfig) -> Result<Self, McpError> {
        Connection::spawn(
            RAG_RAT,
            &config.command,
            &BTreeMap::new(),
            config.working_dir.as_deref(),
        )
        .await
        .map(RagRatClient)
    }

    pub fn tools(&self) -> Vec<Tool> {
        self.0.tools()
    }

    pub fn sink(&self) -> ServerSink {
        self.0.sink()
    }

    /// rag-rat's tools as a [`ToolSet`] group — the base every node's pool is built on.
    pub fn offer(&self) -> ServerTools {
        self.0.offer()
    }

    pub async fn shutdown(self) -> Result<(), McpError> {
        self.0.shutdown().await
    }
}

/// One server's contribution to a node's tools: the tools, and the handle they are called on.
#[derive(Clone)]
pub struct ServerTools {
    pub origin: String,
    pub sink: ServerSink,
    pub tools: Vec<Tool>,
}

/// The tools one node may call, grouped by the server that serves each.
///
/// Grouped rather than flat because a tool is dispatched on the sink of the server that offered
/// it: two servers' tools are not interchangeable even when their names are.
#[derive(Clone, Default)]
pub struct ToolSet {
    groups: Vec<ServerTools>,
}

impl ToolSet {
    /// Assemble a node's tools from the servers it binds, in precedence order.
    ///
    /// **Name collisions: the first server to offer a name keeps it**, and the loser is dropped
    /// with a warning naming both. rag-rat is passed first, so a plugin can never shadow a tool a
    /// node's prompt was written against. The alternative — qualifying a plugin's tools with its
    /// name — would change the string a ruleset's `allow`/`deny` and an `onToolCall` gate match on,
    /// and rig's MCP adapter sends the tool's name straight back to the server as the call's
    /// method, so a qualified name would have to be unqualified again on the way out.
    pub fn from_servers(mut servers: Vec<ServerTools>) -> Self {
        let kept = claim_names(
            &servers
                .iter()
                .map(|s| (s.origin.as_str(), s.tools.as_slice()))
                .collect::<Vec<_>>(),
        );
        for (server, tools) in servers.iter_mut().zip(kept) {
            server.tools = tools;
        }
        servers.retain(|s| !s.tools.is_empty());
        ToolSet { groups: servers }
    }

    /// Keep only the tools named in `allow`, then drop those named in `deny`.
    pub fn narrow(&mut self, allow: &[String], deny: &[String]) {
        for group in &mut self.groups {
            group.tools = narrowed(&group.tools, allow, deny);
        }
        self.groups.retain(|g| !g.tools.is_empty());
    }

    /// Add a tool that is answered locally rather than dispatched — the synthetic `ask`, which a
    /// hook intercepts. It joins the first group so it reaches the agent with everything else; the
    /// sink it nominally belongs to is never used for it.
    pub fn add_local(&mut self, tool: Tool, sink: ServerSink) {
        match self.groups.first_mut() {
            Some(group) => group.tools.push(tool),
            None => self.groups.push(ServerTools {
                origin: RAG_RAT.to_string(),
                sink,
                tools: vec![tool],
            }),
        }
    }

    /// Every tool name offered by a server other than `origin`, in precedence order.
    ///
    /// What a node's default tool list can't name: the built-in lists were written against
    /// rag-rat's catalogue, and a plugin's tools are only known once its server has answered.
    pub fn names_beyond(&self, origin: &str) -> Vec<String> {
        self.groups
            .iter()
            .filter(|g| g.origin != origin)
            .flat_map(|g| g.tools.iter().map(|t| t.name.to_string()))
            .collect()
    }

    /// How many tools the set holds, across every server.
    pub fn len(&self) -> usize {
        self.groups.iter().map(|g| g.tools.len()).sum()
    }

    /// The groups, for binding to an agent one server at a time.
    pub fn groups(&self) -> &[ServerTools] {
        &self.groups
    }

    pub fn is_empty(&self) -> bool {
        self.groups.iter().all(|g| g.tools.is_empty())
    }
}

/// Settle name collisions across servers: each group keeps the names no earlier group claimed.
///
/// Separate from [`ToolSet::from_servers`] because the rule is the interesting part and a
/// [`ServerSink`] only exists behind a live subprocess.
fn claim_names(groups: &[(&str, &[Tool])]) -> Vec<Vec<Tool>> {
    let mut claimed: BTreeMap<&str, &str> = BTreeMap::new();
    groups
        .iter()
        .map(|(origin, tools)| {
            tools
                .iter()
                .filter(|tool| match claimed.get(tool.name.as_ref()) {
                    Some(owner) => {
                        tracing::warn!(
                            tool = %tool.name,
                            kept = owner,
                            dropped = origin,
                            "two MCP servers offer the same tool; the first one connected keeps it"
                        );
                        false
                    }
                    None => {
                        claimed.insert(tool.name.as_ref(), origin);
                        true
                    }
                })
                .cloned()
                .collect()
        })
        .collect()
}

/// The tools that survive an `allow`/`deny` pair: named in `allow`, and not in `deny`.
fn narrowed(tools: &[Tool], allow: &[String], deny: &[String]) -> Vec<Tool> {
    let named = |names: &[String], tool: &Tool| names.iter().any(|n| n == tool.name.as_ref());
    tools
        .iter()
        .filter(|t| named(allow, t) && !named(deny, t))
        .cloned()
        .collect()
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

    fn tool(name: &'static str) -> Tool {
        Tool::new(
            name,
            "",
            std::sync::Arc::new(rmcp::model::JsonObject::new()),
        )
    }

    fn names(groups: &[Vec<Tool>]) -> Vec<Vec<&str>> {
        groups
            .iter()
            .map(|g| g.iter().map(|t| t.name.as_ref()).collect())
            .collect()
    }

    #[test]
    fn the_first_server_to_offer_a_name_keeps_it() {
        // rag-rat is claimed first, so a plugin can never shadow a tool a node's prompt was
        // written against — and the plugin keeps everything that doesn't collide.
        let rag_rat = [tool("semantic_search"), tool("memory_create")];
        let plugin = [tool("semantic_search"), tool("lint")];
        let other = [tool("lint")];

        let kept = claim_names(&[
            (RAG_RAT, &rag_rat[..]),
            ("linty", &plugin[..]),
            ("also-linty", &other[..]),
        ]);
        assert_eq!(
            names(&kept),
            vec![
                vec!["semantic_search", "memory_create"],
                vec!["lint"],
                Vec::<&str>::new(),
            ]
        );
    }

    #[test]
    fn allow_selects_and_deny_removes() {
        let tools = [
            tool("semantic_search"),
            tool("impact_surface"),
            tool("lint"),
        ];
        let say = |names: &[&str]| names.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        assert_eq!(
            names(&[narrowed(&tools, &say(&["semantic_search", "lint"]), &[])]),
            vec![vec!["semantic_search", "lint"]]
        );
        // deny wins over allow, and a name nothing offers is simply absent.
        assert_eq!(
            names(&[narrowed(
                &tools,
                &say(&["semantic_search", "nope"]),
                &say(&["semantic_search"])
            )]),
            vec![Vec::<&str>::new()]
        );
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
