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
            sink: Some(self.sink()),
            tools: self.tools.clone(),
            prefix: None,
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
    /// Where a call goes. `None` for tools this host answers itself — the synthetic ones a hook
    /// intercepts, and the built-ins it implements.
    pub sink: Option<ServerSink>,
    pub tools: Vec<Tool>,
    /// Prefixed onto every tool name this server offers, for the model and for everything that
    /// matches on a name. `None` for a server this host launched itself, whose tools keep the
    /// names they were always called by.
    pub prefix: Option<String>,
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
    /// The name the model sees for `tool`.
    pub fn display_name(&self, tool: &Tool) -> String {
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
            group.tools.retain(|t| {
                let shown = match &prefix {
                    Some(prefix) => format!("{prefix}{}", t.name),
                    None => t.name.to_string(),
                };
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
