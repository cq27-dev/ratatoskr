//! Clients to the MCP servers a run talks to, over stdio subprocesses or streamable HTTP.
//!
//! [`Connection::spawn`] and [`Connection::connect_http`] perform the MCP handshake and list tools.
//! They hand back the tool list plus a [`ServerSink`] (a cloneable client handle) — everything
//! `ratatoskr-agent` needs to bind those tools to a `rig` agent. The running service is held so the
//! connection stays alive; drop or [`shutdown`](Connection::shutdown) tears it down.
//!
//! rag-rat is the server ratatoskr is built around and is connected from config
//! ([`RagRatClient`]); a plugin's servers are connected alongside it. A node's tools can therefore
//! come from several servers at once, which is what [`ToolSet`] carries: tools grouped by the
//! server each is dispatched to.

use std::collections::BTreeMap;
use std::path::Path;

use ratatoskr_core::{Capability, McpServerConfig, McpTransport, RagRatConfig};
use rmcp::ServiceExt;
use rmcp::model::Tool;
use rmcp::service::{RoleClient, RunningService, ServerSink};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};

/// The origin name of the connection made from `[rag_rat]` in the config.
///
/// Also what a plugin-declared server is checked against: the rag-rat plugin declares the same
/// server ratatoskr already launches, and it must not be spawned twice.
pub const RAG_RAT: &str = "rag-rat";

/// The origin of tools this host answers itself rather than dispatching.
pub const LOCAL: &str = "builtin";

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

/// A live connection to one MCP server. Holds the running service so the session stays alive.
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

    /// Connect to a streamable-HTTP MCP server, complete its handshake, and list its tools.
    ///
    /// rmcp owns request-scoped and persistent SSE streams, including session/protocol negotiation
    /// and recovery. HTTP transport errors deliberately cross this boundary without server response
    /// text because a proxy may reflect the bearer credential in that text.
    pub async fn connect_http(
        origin: &str,
        url: &str,
        bearer_token: Option<String>,
    ) -> Result<Self, McpError> {
        let config = bearer_token.map_or_else(
            || StreamableHttpClientTransportConfig::with_uri(url),
            |token| StreamableHttpClientTransportConfig::with_uri(url).auth_header(token),
        );
        let transport = StreamableHttpClientTransport::from_config(config);
        let service = ().serve(transport).await.map_err(|_| McpError::Handshake {
            origin: origin.to_string(),
            detail: "streamable HTTP transport failed".to_string(),
        })?;
        let tools = service
            .peer()
            .list_all_tools()
            .await
            .map_err(|_| McpError::ListTools {
                origin: origin.to_string(),
                detail: "streamable HTTP transport failed".to_string(),
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
            sink: Some(self.sink()),
            tools: self.tools.clone(),
            prefix: None,
            renames: BTreeMap::new(),
            capabilities: BTreeMap::new(),
            provenance: ServerProvenance::Configured,
        }
    }

    /// Cleanly cancel the connection.
    pub async fn shutdown(self) -> Result<(), McpError> {
        let origin = self.origin;
        self.service
            .cancel()
            .await
            .map(|_| ())
            // An rmcp HTTP error can retain reflected request credentials in its response text.
            .map_err(|_| McpError::Shutdown {
                origin,
                detail: "MCP service shutdown failed".to_string(),
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

/// A connection declared in `[mcp.servers]`, with host-owned naming and authority metadata.
pub struct ConfiguredMcpClient {
    connection: Connection,
    renames: BTreeMap<String, String>,
    capabilities: BTreeMap<String, Capability>,
}

impl ConfiguredMcpClient {
    pub async fn connect(origin: &str, config: &McpServerConfig) -> Result<Self, McpError> {
        let bearer_token = config
            .bearer_token_env
            .as_deref()
            .and_then(|name| std::env::var(name).ok())
            .filter(|token| !token.is_empty());
        let connection = match config.transport {
            McpTransport::StreamableHttp => {
                Connection::connect_http(origin, &config.url, bearer_token).await?
            }
        };
        let renames = config
            .tools
            .iter()
            .filter_map(|(wire, tool)| tool.name.clone().map(|name| (wire.clone(), name)))
            .collect();
        let capabilities = config
            .tools
            .iter()
            .map(|(wire, tool)| {
                (
                    tool.name.clone().unwrap_or_else(|| wire.clone()),
                    tool.capability,
                )
            })
            .collect();
        Ok(Self {
            connection,
            renames,
            capabilities,
        })
    }

    pub fn offer(&self) -> ServerTools {
        ServerTools {
            renames: self.renames.clone(),
            capabilities: self.capabilities.clone(),
            ..self.connection.offer()
        }
    }

    pub fn origin(&self) -> &str {
        self.connection.origin()
    }

    pub async fn shutdown(self) -> Result<(), McpError> {
        self.connection.shutdown().await
    }
}

/// Where a tool offer came from. Authority defaults depend on provenance, never origin spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerProvenance {
    Configured,
    Plugin,
    Builtin,
}

/// One server's contribution to a node's tools: the tools, and the handle they are called on.
#[derive(Clone)]
pub struct ServerTools {
    pub origin: String,
    /// Where a call goes. `None` for tools this host answers itself — the synthetic ones a hook
    /// intercepts, and the built-ins it implements.
    pub sink: Option<ServerSink>,
    pub tools: Vec<Tool>,
    /// Prefixed onto every tool name this server offers, for the model and for everything that
    /// matches on a name. `None` for a server this host launched itself, whose tools keep the
    /// names they were always called by.
    pub prefix: Option<String>,
    /// Per-tool wire-name to display-name rewrites. Unlike a prefix, these can follow a host's
    /// existing vocabulary exactly while dispatch still uses the wire name.
    pub renames: BTreeMap<String, String>,
    /// Minimum authority keyed by the display name seen by stages and rulesets.
    pub capabilities: BTreeMap<String, Capability>,
    pub provenance: ServerProvenance,
}

/// How the plugin format names the tools of a server a plugin declared:
/// `mcp__plugin_<plugin>_<server>__<tool>`.
///
/// The bare `mcp__<server>__<tool>` form belongs to a server the *user* configured. rag-rat is
/// that case here, and keeps its plain names — every node's built-in tool list, every ruleset and
/// every recorded memory names `semantic_search`, not a qualified spelling of it.
pub fn qualified_prefix(plugin: &str, server: &str) -> String {
    format!("mcp__plugin_{}_{}__", segment(plugin), segment(server))
}

/// One name segment, with everything outside the format's allowed set replaced.
fn segment(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl ServerTools {
    /// Whether this server was declared by a plugin rather than host configuration.
    pub fn is_plugin(&self) -> bool {
        self.provenance == ServerProvenance::Plugin
    }

    /// The name the model sees for `tool`.
    pub fn display_name(&self, tool: &Tool) -> String {
        if let Some(name) = self.renames.get(tool.name.as_ref()) {
            return name.clone();
        }
        match &self.prefix {
            Some(prefix) => format!("{prefix}{}", tool.name),
            None => tool.name.to_string(),
        }
    }

    /// Every name this server contributes, in order.
    pub fn display_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| self.display_name(t)).collect()
    }

    /// This server's tools as the model is offered them: named as it will call them, paired with
    /// the name the server itself answers to.
    pub fn offered(&self) -> Vec<(Tool, String)> {
        self.tools
            .iter()
            .map(|tool| {
                let wire = tool.name.to_string();
                let mut renamed = tool.clone();
                renamed.name = self.display_name(tool).into();
                (renamed, wire)
            })
            .collect()
    }
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
    /// node's prompt was written against — which matters much less now that a plugin server's
    /// tools are qualified and so rarely collide at all, but still settles the case where two
    /// plugins declare one server name.
    ///
    /// Every rule here is applied to the *display* name, because that is the string a ruleset's
    /// `allow`/`deny`, an `onToolCall` gate, a hook matcher and the model all see.
    pub fn from_servers(mut servers: Vec<ServerTools>) -> Self {
        let named: Vec<(&str, Vec<String>)> = servers
            .iter()
            .map(|s| (s.origin.as_str(), s.display_names()))
            .collect();
        let kept = claim_names(&named);
        for (server, keep) in servers.iter_mut().zip(kept) {
            let mut mask = keep.into_iter();
            server.tools.retain(|_| mask.next().unwrap_or(false));
        }
        servers.retain(|s| !s.tools.is_empty());
        ToolSet { groups: servers }
    }

    /// Keep only the tools named in `allow`, then drop those named in `deny`.
    pub fn narrow(&mut self, allow: &[String], deny: &[String]) {
        for group in &mut self.groups {
            let mut mask = narrowed(&group.display_names(), allow, deny).into_iter();
            group.tools.retain(|_| mask.next().unwrap_or(false));
        }
        self.groups.retain(|g| !g.tools.is_empty());
    }

    /// Add a tool that is answered locally rather than dispatched — the synthetic `ask`, which a
    /// hook intercepts. It joins the first group so it reaches the agent with everything else; the
    /// sink it nominally belongs to is never used for it.
    ///
    /// The name is *taken*, not merely added: the hook that answers it matches on the name alone,
    /// so a server offering the same one would be shadowed anyway — silently, and with the wrong
    /// argument schema shown to the model. Any such tool is dropped here instead.
    pub fn add_local(&mut self, tool: Tool) {
        for group in &mut self.groups {
            let prefix = group.prefix.clone();
            let renames = group.renames.clone();
            group.tools.retain(|t| {
                let shown = renames
                    .get(t.name.as_ref())
                    .cloned()
                    .or_else(|| prefix.as_ref().map(|p| format!("{p}{}", t.name)))
                    .unwrap_or_else(|| t.name.to_string());
                let clash = shown == tool.name;
                if clash {
                    tracing::warn!(
                        tool = %shown,
                        server = %group.origin,
                        "dropping a server's tool: the name is answered inside the run"
                    );
                }
                !clash
            });
        }
        self.local().tools.push(tool);
    }

    /// The group of tools this host answers itself, created on first use.
    pub fn local(&mut self) -> &mut ServerTools {
        if !self.groups.iter().any(|g| g.sink.is_none()) {
            self.groups.push(ServerTools {
                origin: LOCAL.to_string(),
                sink: None,
                tools: Vec::new(),
                prefix: None,
                renames: BTreeMap::new(),
                capabilities: BTreeMap::new(),
                provenance: ServerProvenance::Builtin,
            });
        }
        self.groups
            .iter_mut()
            .find(|g| g.sink.is_none())
            .expect("just ensured")
    }

    /// Every tool name in the set, as the model sees it, in precedence order.
    pub fn names(&self) -> Vec<String> {
        self.groups.iter().flat_map(|g| g.display_names()).collect()
    }

    /// The minimum authority each offered tool requires. A server-owned tool without an explicit
    /// declaration is `Publish`: an unknown remote operation must never slip through a narrower
    /// stage merely because its name sounds harmless.
    pub fn capability(&self, name: &str) -> Capability {
        self.groups
            .iter()
            .find_map(|group| {
                group
                    .display_names()
                    .into_iter()
                    .any(|offered| offered == name)
                    .then(|| {
                        group.capabilities.get(name).copied().unwrap_or_else(|| {
                            match group.provenance {
                                ServerProvenance::Builtin => declared_capability(LOCAL, name),
                                ServerProvenance::Configured if group.origin == RAG_RAT => {
                                    declared_capability(RAG_RAT, name)
                                }
                                ServerProvenance::Configured | ServerProvenance::Plugin => {
                                    Capability::Publish
                                }
                            }
                        })
                    })
            })
            .unwrap_or(Capability::Publish)
    }

    /// Every tool name offered by a server other than `origin`, in precedence order.
    ///
    /// What a node's default tool list can't name: the built-in lists were written against
    /// rag-rat's catalogue, and a plugin's tools are only known once its server has answered.
    pub fn names_beyond(&self, origin: &str) -> Vec<String> {
        self.groups
            .iter()
            .filter(|g| g.origin != origin)
            .flat_map(|g| g.display_names())
            .collect()
    }

    /// Every tool contributed by a non-host, non-rag-rat server. Plugin MCP servers have no
    /// capability declaration, so callers must treat these as untrusted authority.
    pub fn external_names(&self) -> Vec<String> {
        self.groups
            .iter()
            .filter(|g| g.origin != RAG_RAT && g.origin != LOCAL)
            .flat_map(|g| g.display_names())
            .collect()
    }

    /// Whether a server of this origin contributed to the set.
    ///
    /// Distinguishes "that server offers no such tool" from "that server is not here at all",
    /// which read the same from a name lookup and mean opposite things: one is a typo, the other
    /// is how this repository is configured.
    pub fn has_server(&self, origin: &str) -> bool {
        self.groups.iter().any(|g| g.origin == origin)
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

fn declared_capability(origin: &str, name: &str) -> Capability {
    match origin {
        LOCAL => match name {
            "Read" | "Grep" | "Glob" | "ask" | "Skill" => Capability::Read,
            "Write" | "Edit" | "Bash" => Capability::Write,
            _ => Capability::Publish,
        },
        RAG_RAT => match name {
            "semantic_search"
            | "symbol_lookup"
            | "find_callers"
            | "trace_callees"
            | "impact_surface"
            | "read_chunk"
            | "repo_brief"
            | "repo_clusters"
            | "important_symbols"
            | "memory_search"
            | "memory_show"
            | "memory_for_symbol"
            | "memory_for_path"
            | "memory_for_call_path"
            | "papertrail_issue_search" => Capability::Read,
            "memory_create" | "memory_update" | "memory_mark_obsolete" => Capability::Write,
            _ => Capability::Publish,
        },
        // Plugin and user-provided MCP servers do not currently declare authority in their
        // manifests. Publish is the safe declaration until they do.
        _ => Capability::Publish,
    }
}

/// Settle name collisions across servers: each group keeps the names no earlier group claimed.
///
/// Returns a keep-mask per group rather than the tools themselves, so the rule stays a pure
/// function of the names — a [`ServerSink`] only exists behind a live subprocess.
fn claim_names(groups: &[(&str, Vec<String>)]) -> Vec<Vec<bool>> {
    let mut claimed: BTreeMap<&str, &str> = BTreeMap::new();
    groups
        .iter()
        .map(|(origin, names)| {
            names
                .iter()
                .map(|name| match claimed.get(name.as_str()) {
                    Some(owner) => {
                        tracing::warn!(
                            tool = name,
                            kept = owner,
                            dropped = origin,
                            "two MCP servers offer the same tool; the first one connected keeps it"
                        );
                        false
                    }
                    None => {
                        claimed.insert(name, origin);
                        true
                    }
                })
                .collect()
        })
        .collect()
}

/// Which names survive an `allow`/`deny` pair: in `allow`, and not in `deny`.
fn narrowed(names: &[String], allow: &[String], deny: &[String]) -> Vec<bool> {
    names
        .iter()
        .map(|name| allow.contains(name) && !deny.contains(name))
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

    #[tokio::test]
    async fn unreachable_http_server_is_a_handshake_error() {
        assert!(matches!(
            Connection::connect_http("remote", "http://127.0.0.1:9/mcp", None).await,
            Err(McpError::Handshake { origin, .. }) if origin == "remote"
        ));
    }

    fn tool(name: &str) -> Tool {
        let mut tool = Tool::default();
        tool.name = name.to_string().into();
        tool
    }

    #[test]
    fn renamed_tools_are_offered_and_narrowed_by_display_name() {
        let server = ServerTools {
            origin: "remote".to_string(),
            sink: None,
            tools: vec![
                tool("web_fetch_exa"),
                tool("web_search_exa"),
                tool("other_exa_tool"),
            ],
            prefix: None,
            renames: BTreeMap::from([
                ("web_fetch_exa".to_string(), "WebFetch".to_string()),
                ("web_search_exa".to_string(), "WebSearch".to_string()),
            ]),
            capabilities: BTreeMap::from([
                ("WebFetch".to_string(), Capability::Read),
                ("WebSearch".to_string(), Capability::Read),
            ]),
            provenance: ServerProvenance::Configured,
        };
        assert_eq!(
            server.display_names(),
            say(&["WebFetch", "WebSearch", "other_exa_tool"])
        );
        assert_eq!(
            server
                .offered()
                .into_iter()
                .map(|(shown, wire)| (shown.name.to_string(), wire))
                .collect::<Vec<_>>(),
            vec![
                ("WebFetch".to_string(), "web_fetch_exa".to_string()),
                ("WebSearch".to_string(), "web_search_exa".to_string()),
                ("other_exa_tool".to_string(), "other_exa_tool".to_string()),
            ]
        );
        let mut wire_name = ToolSet::from_servers(vec![server.clone()]);
        wire_name.narrow(&say(&["web_fetch_exa"]), &[]);
        assert!(wire_name.is_empty());

        let mut set = ToolSet::from_servers(vec![server]);
        assert_eq!(set.capability("WebFetch"), Capability::Read);
        assert_eq!(set.capability("WebSearch"), Capability::Read);
        assert_eq!(set.capability("other_exa_tool"), Capability::Publish);
        set.narrow(&say(&["WebFetch"]), &[]);
        assert_eq!(set.names(), say(&["WebFetch"]));
    }

    #[test]
    fn absent_webfetch_name_is_a_no_op_when_narrowing() {
        let mut set = ToolSet::default();
        set.narrow(&say(&["WebFetch"]), &[]);
        assert!(set.is_empty());
    }

    #[test]
    fn origin_spelling_cannot_grant_authority_to_an_external_server() {
        for provenance in [ServerProvenance::Configured, ServerProvenance::Plugin] {
            let set = ToolSet::from_servers(vec![ServerTools {
                origin: LOCAL.to_string(),
                sink: None,
                tools: vec![tool("Read")],
                prefix: None,
                renames: BTreeMap::new(),
                capabilities: BTreeMap::new(),
                provenance,
            }]);
            assert_eq!(set.capability("Read"), Capability::Publish);
        }
    }

    /// The names a keep-mask leaves, per group.
    fn kept(names: &[Vec<String>], masks: &[Vec<bool>]) -> Vec<Vec<String>> {
        names
            .iter()
            .zip(masks)
            .map(|(group, mask)| {
                group
                    .iter()
                    .zip(mask)
                    .filter(|(_, keep)| **keep)
                    .map(|(n, _)| n.clone())
                    .collect()
            })
            .collect()
    }

    fn say(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_first_server_to_offer_a_name_keeps_it() {
        // rag-rat is claimed first, so a plugin can never shadow a tool a node's prompt was
        // written against — and the plugin keeps everything that doesn't collide.
        let groups = [
            (RAG_RAT, say(&["semantic_search", "memory_create"])),
            ("linty", say(&["semantic_search", "lint"])),
            ("also-linty", say(&["lint"])),
        ];
        let masks = claim_names(&groups);
        let names: Vec<Vec<String>> = groups.iter().map(|(_, n)| n.clone()).collect();
        assert_eq!(
            kept(&names, &masks),
            vec![
                say(&["semantic_search", "memory_create"]),
                say(&["lint"]),
                Vec::<String>::new(),
            ]
        );
    }

    #[test]
    fn a_plugin_servers_tools_are_named_the_way_the_format_names_them() {
        assert_eq!(
            qualified_prefix("my-plugin", "database-tools"),
            "mcp__plugin_my-plugin_database-tools__"
        );
        // Anything outside the allowed set becomes `_`, per segment.
        assert_eq!(
            qualified_prefix("acme.tools", "db/main"),
            "mcp__plugin_acme_tools_db_main__"
        );
    }

    #[test]
    fn allow_selects_and_deny_removes() {
        let offered = [say(&["semantic_search", "impact_surface", "lint"])];

        assert_eq!(
            kept(
                &offered,
                &[narrowed(
                    &offered[0],
                    &say(&["semantic_search", "lint"]),
                    &[]
                )]
            ),
            vec![say(&["semantic_search", "lint"])]
        );
        // deny wins over allow, and a name nothing offers is simply absent.
        assert_eq!(
            kept(
                &offered,
                &[narrowed(
                    &offered[0],
                    &say(&["semantic_search", "nope"]),
                    &say(&["semantic_search"])
                )]
            ),
            vec![Vec::<String>::new()]
        );
    }

    #[test]
    fn a_ruleset_and_a_hook_matcher_see_the_qualified_name() {
        // The whole point: the string every gate matches on is the one the model calls, so a
        // plugin's tool is denied and matched under the name that plugin's author would write.
        let prefix = qualified_prefix("linty", "lint");
        let offered = [vec![format!("{prefix}check")]];

        assert_eq!(
            kept(&offered, &[narrowed(&offered[0], &offered[0], &[])]),
            vec![vec!["mcp__plugin_linty_lint__check".to_string()]]
        );
        // Its bare name is not a name anything sees, so denying it does nothing.
        assert_eq!(
            kept(
                &offered,
                &[narrowed(&offered[0], &offered[0], &say(&["check"]))]
            ),
            vec![vec!["mcp__plugin_linty_lint__check".to_string()]]
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
