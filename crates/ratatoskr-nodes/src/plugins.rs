//! The plugin plumbing: what a plugin gives a node, and how it gets there.
//!
//! Discovery, per-node binding, the MCP servers plugins declare, the hooks that run around a
//! node's tool calls, and the resolution of one node's agent settings against its ruleset. This is
//! not about running a pipeline — the orchestration in `lib.rs` reaches in here for [`PluginContext`]
//! and [`NodePlugins`], re-exported at the crate root so `crate::PluginContext` still resolves.

use std::path::PathBuf;
use std::sync::Arc;

use ratatoskr_core::{Capability, RatatoskrConfig, ToolDecision, ToolPolicy};
use ratatoskr_graph::NodeError;
use ratatoskr_mcp::{Connection, ServerTools, ToolSet};
use ratatoskr_script::ScriptEngine;

use crate::stage::stage_profile;
use crate::{
    AgentProfile, NodeAgentConfig, PlanError, Stage, agent_profiles, built_in_stages, route, skills,
};

/// What each plugin contributed for this run.
///
/// Hooks run once per run — `SessionStart` describes the repository, not the node — and each node
/// then composes its own context from the plugins its ruleset binds. Per-node binding therefore
/// costs nothing extra in hook executions.
#[derive(Clone, Default)]
pub struct PluginContext {
    contexts: std::collections::BTreeMap<String, String>,
    /// Every plugin found, which is what a node inherits when no ruleset narrows it.
    discovered: Vec<String>,
    engine: Option<Arc<ScriptEngine>>,
    /// The MCP servers the discovered plugins declare, connected once and shared by every node
    /// that binds the plugin. Held for the run: dropping a connection kills its subprocess.
    servers: Arc<Vec<PluginServer>>,
    /// The loaded plugins themselves, for the hooks that run around a node's tool calls.
    plugins: Arc<Vec<ratatoskr_plugin::Plugin>>,
    /// Wall-clock the run has spent inside tool hooks, shared by every node so the budget is the
    /// run's rather than each node's.
    hook_time: Arc<std::sync::atomic::AtomicU64>,
    /// What this repo lets its plugins' hooks spend.
    limits: ratatoskr_core::HookLimits,
    /// Skills this repository does not want offered, by name.
    skills_deny: Vec<String>,
}

/// Shared node execution context: plugin contributions plus reusable profile guidance.
///
/// One value rather than a field per contribution: settings are resolved in one place and travel
/// together, and every new one would otherwise mean another parameter threaded through every node
/// struct and every construction site.
#[derive(Clone, Default)]
pub struct NodePlugins {
    /// Session context, prefixed to whichever preamble the node runs with.
    pub context: Option<String>,
    /// Reusable agent guidance, composed ahead of the stage's own instructions.
    pub profile_prompt: String,
    /// Runs the node's tool calls past its plugins' `PreToolUse`/`PostToolUse` hooks. `None` when
    /// nothing it binds registers one, so a node that gains nothing pays nothing.
    pub observer: Option<Arc<dyn ratatoskr_agent::PluginHooks>>,
    /// Skills the plugins it binds ship, in binding order.
    pub skills: Vec<ratatoskr_plugin::Skill>,
}

/// Runs one node's bound plugins around each of its tool calls.
///
/// Holds only the plugins that node binds, so the per-node binding that decides its context and
/// its tools decides its hooks too.
struct NodeObserver {
    plugins: Vec<ratatoskr_plugin::Plugin>,
    cwd: PathBuf,
    hook_time: Arc<std::sync::atomic::AtomicU64>,
    limits: ratatoskr_core::HookLimits,
}

impl NodeObserver {
    /// Run one event past this node's plugins.
    fn run<'a>(&'a self, event: ratatoskr_plugin::HookEvent<'a>) -> ratatoskr_agent::Answer<'a> {
        Box::pin(ratatoskr_plugin::run_event(
            &self.plugins,
            event,
            &self.cwd,
            &self.limits,
            &self.hook_time,
        ))
    }
}

impl ratatoskr_agent::PluginHooks for NodeObserver {
    fn starting<'a>(&'a self, node: &'a str) -> ratatoskr_agent::Answer<'a> {
        self.run(ratatoskr_plugin::HookEvent::subagent_start(node))
    }

    fn prompting<'a>(&'a self, prompt: &'a str) -> ratatoskr_agent::Answer<'a> {
        self.run(ratatoskr_plugin::HookEvent::user_prompt_submit(prompt))
    }

    fn before<'a>(&'a self, tool: &'a str, args: &'a str) -> ratatoskr_agent::Answer<'a> {
        self.run(ratatoskr_plugin::HookEvent::pre_tool_use(tool, args))
    }

    fn after<'a>(
        &'a self,
        tool: &'a str,
        args: &'a str,
        result: &'a str,
    ) -> ratatoskr_agent::Answer<'a> {
        self.run(ratatoskr_plugin::HookEvent::post_tool_use(
            tool, args, result,
        ))
    }

    fn finished<'a>(
        &'a self,
        node: &'a str,
        outcome: Result<&'a str, &'a str>,
    ) -> ratatoskr_agent::Answer<'a> {
        Box::pin(async move {
            // Both, because the format has both and a plugin may register either: a node is the
            // subagent that stopped, and the turn that ended. A turn that failed ended as
            // `StopFailure`, which is what that event is for — `Stop` keeps meaning a turn that
            // produced an answer. `SubagentStop` fires either way: the subagent is over.
            let ended = match outcome {
                Ok(last) => ratatoskr_plugin::HookEvent::stop(node, last),
                Err(error) => ratatoskr_plugin::HookEvent::stop_failure(node, error),
            };
            let last = outcome.unwrap_or_else(|error| error);
            let stop = self.run(ended).await;
            let subagent = self
                .run(ratatoskr_plugin::HookEvent::subagent_stop(node, last))
                .await;
            match [stop, subagent].into_iter().flatten().collect::<Vec<_>>() {
                parts if parts.is_empty() => None,
                parts => Some(parts.join("\n\n")),
            }
        })
    }
}

/// The events a run actually has. Every other event a plugin registers describes a session with a
/// person in it, or a lifecycle this host does not have, and is not ours to fire.
const NODE_EVENTS: [&str; 7] = [
    "SubagentStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
    "StopFailure",
    "SubagentStop",
];

/// One connected server, and the plugin that declared it — which is what a node binds, not the
/// server's own name, and what the format names its tools after.
struct PluginServer {
    plugin: String,
    connection: Connection,
}

impl PluginServer {
    /// This server's tools, named the way the format names a plugin's.
    fn offer(&self) -> ServerTools {
        ServerTools {
            prefix: Some(ratatoskr_mcp::qualified_prefix(
                &self.plugin,
                self.connection.origin(),
            )),
            ..self.connection.offer()
        }
    }
}

impl PluginContext {
    /// Discover plugins, run their `SessionStart` hooks, and check that every plugin a ruleset
    /// *requires* actually exists.
    ///
    /// A plugin that is missing, broken, or slow contributes nothing and is logged. A plugin a
    /// ruleset *names* splits by how it was named. An explicit `defineAgent` binding (an `Only`
    /// list, an `add`, or a `remove`) is a requirement: naming one nobody installed is a typo, and
    /// the run fails rather than binding less than its author asked for. A `defineDefaults` name is
    /// a preference — it applies to every node, so a missing one warns and narrows the tool pool to
    /// what was discovered rather than refusing the run (a rag-rat-less checkout falls back to file
    /// tools). A name in both categories is a requirement: the explicit binding wins.
    pub async fn resolve(
        config: &RatatoskrConfig,
        engine: &Arc<ScriptEngine>,
        cwd: &std::path::Path,
    ) -> Result<Self, PlanError> {
        let plugins = ratatoskr_plugin::discover(&config.plugins.search_paths(cwd));
        for plugin in &plugins {
            tracing::info!(plugin = plugin.name, "loaded plugin");
        }

        let installed = |name: &String| plugins.iter().any(|p| &p.name == name);
        let known = || {
            let names: Vec<&str> = plugins.iter().map(|p| p.name.as_str()).collect();
            if names.is_empty() {
                "none".to_string()
            } else {
                names.join(", ")
            }
        };

        // An explicit `defineAgent` binding is a requirement: a missing one fails the run.
        let required = engine.required_plugins();
        let missing_required: Vec<String> =
            required.iter().filter(|n| !installed(n)).cloned().collect();
        if !missing_required.is_empty() {
            return Err(PlanError::node(
                "plugins",
                NodeError::Failed(format!(
                    "ruleset names plugin(s) that were not found: {}; discovered: {}",
                    missing_required.join(", "),
                    known()
                )),
            ));
        }

        // A `defineDefaults` name is a preference: a missing one warns and narrows the pool. Names
        // promoted to required by an agent rule are handled above, so exclude them here.
        let missing_preferred: Vec<String> = engine
            .declared_plugins()
            .into_iter()
            .filter(|n| !required.contains(n) && !installed(n))
            .collect();
        if !missing_preferred.is_empty() {
            tracing::warn!(
                missing = missing_preferred.join(", "),
                discovered = known(),
                "ruleset default names plugin(s) that were not found; running without them"
            );
        }

        let discovered: Vec<String> = plugins.iter().map(|p| p.name.clone()).collect();
        let contexts = ratatoskr_plugin::session_start(&plugins, cwd, &config.plugins.hooks).await;
        for (name, text) in &contexts {
            tracing::info!(plugin = name, chars = text.len(), "plugin session context");
        }
        Ok(PluginContext {
            contexts,
            discovered,
            engine: Some(Arc::clone(engine)),
            servers: Arc::new(connect_plugin_servers(&plugins, cwd).await),
            plugins: Arc::new(plugins),
            hook_time: Arc::default(),
            limits: config.plugins.hooks.clone(),
            skills_deny: config.plugins.skills_deny.clone(),
        })
    }

    /// Tell every plugin the run is over, and why.
    ///
    /// Run-level rather than per node, like `SessionStart`: a plugin that keeps state across a
    /// session closes it once, not once per node. Nothing is injected — there is no conversation
    /// left to inject into — so this runs for what a hook *does*.
    pub async fn session_end(&self, reason: &str) {
        let cwd = std::env::current_dir().unwrap_or_default();
        if let Some(unused) = ratatoskr_plugin::run_event(
            &self.plugins,
            ratatoskr_plugin::HookEvent::session_end(reason),
            &cwd,
            &self.limits,
            &self.hook_time,
        )
        .await
        {
            tracing::info!(
                chars = unused.len(),
                "a hook answered at the end of the run; its context has nowhere to go"
            );
        }
    }

    /// Whether this repository has asked that `skill` never be offered.
    ///
    /// Matched on the name as the plugin spells it, ignoring case and surrounding space, so a
    /// config entry does not have to reproduce a name's exact typography to take effect. Names are
    /// kebab-case by the format's convention, and a repository denying `init-rag-rat` means the
    /// one it can see in its own logs.
    fn denied(&self, skill: &str) -> bool {
        self.skills_deny
            .iter()
            .any(|d| d.trim().eq_ignore_ascii_case(skill.trim()))
    }

    /// What the plugins `node` binds give it: their session context, and a hook runner when any of
    /// them registers one for a tool call.
    pub fn for_node(&self, node: &str) -> NodePlugins {
        let bound = self.bound(node);
        let hooked: Vec<ratatoskr_plugin::Plugin> = self
            .plugins
            .iter()
            .filter(|p| bound.contains(&p.name))
            .filter(|p| {
                p.hooks
                    .iter()
                    .any(|h| NODE_EVENTS.contains(&h.event.as_str()))
            })
            .cloned()
            .collect();
        NodePlugins {
            skills: self
                .plugins
                .iter()
                .filter(|p| bound.contains(&p.name))
                .flat_map(|p| p.skills.iter().cloned())
                .filter(|s| !self.denied(&s.name))
                .collect(),
            context: ratatoskr_plugin::compose(&self.contexts, &bound, &self.limits),
            profile_prompt: String::new(),
            // `None` rather than an empty runner: it is what keeps the hook off the agent
            // entirely for a node whose plugins have nothing to say about its tool calls.
            observer: (!hooked.is_empty()).then(|| {
                Arc::new(NodeObserver {
                    plugins: hooked,
                    cwd: std::env::current_dir().unwrap_or_default(),
                    hook_time: Arc::clone(&self.hook_time),
                    limits: self.limits.clone(),
                }) as Arc<dyn ratatoskr_agent::PluginHooks>
            }),
        }
    }

    /// Which plugins `node` binds — its ruleset's declaration, or every plugin found.
    ///
    /// Restricted to what was discovered: a `defineDefaults` name nobody installed is a preference
    /// (#185), so it drops out of the bound set rather than binding a plugin that does not exist.
    /// An undiscovered *explicit* binding never reaches here — `resolve` fails first.
    fn bound(&self, node: &str) -> Vec<String> {
        match &self.engine {
            Some(engine) => engine
                .plugins_for(node, &self.discovered)
                .into_iter()
                .filter(|name| self.discovered.contains(name))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Every tool `node` may call: rag-rat's catalogue, then the servers its plugins declare.
    ///
    /// rag-rat comes first so it wins any name collision — see [`ToolSet::from_servers`].
    /// The tools one node may call: rag-rat's, when there is a rag-rat, plus the plugin servers
    /// bound to that node.
    ///
    /// `None` omits the group rather than passing an empty one, so a pool without rag-rat is the
    /// same shape as a pool that never had it — nothing downstream has to special-case a server
    /// that offers nothing.
    pub(crate) fn pool_for(&self, node: &str, rag_rat: Option<ServerTools>) -> ToolSet {
        let bound = self.bound(node);
        let mut servers: Vec<ServerTools> = rag_rat.into_iter().collect();
        servers.extend(
            self.servers
                .iter()
                .filter(|s| bound.contains(&s.plugin))
                .map(PluginServer::offer),
        );
        ToolSet::from_servers(servers)
    }
}

/// Connect the MCP servers the discovered plugins declare, once per run.
///
/// A server that will not start costs its plugin's tools and nothing else: a broken plugin must
/// not fail a run.
async fn connect_plugin_servers(
    plugins: &[ratatoskr_plugin::Plugin],
    cwd: &std::path::Path,
) -> Vec<PluginServer> {
    let mut connected = Vec::new();
    for (plugin, spec) in servers_to_start(plugins) {
        match Connection::spawn(&spec.name, &spec.command, &spec.env, Some(cwd)).await {
            Ok(connection) => connected.push(PluginServer {
                plugin: plugin.to_string(),
                connection,
            }),
            Err(e) => tracing::warn!(
                plugin,
                server = spec.name,
                "plugin MCP server unavailable, its tools are not offered: {e}"
            ),
        }
    }
    connected
}

/// Which declared servers actually get started, paired with the plugin that declared each.
///
/// One per server name, and rag-rat's name counts as already taken: the rag-rat plugin declares
/// the very server ratatoskr launched from `[rag_rat]`, and a second copy would pay for another
/// index load to offer the identical tools.
pub(crate) fn servers_to_start(
    plugins: &[ratatoskr_plugin::Plugin],
) -> Vec<(&str, &ratatoskr_plugin::McpServerSpec)> {
    let mut claimed: Vec<&str> = vec![ratatoskr_mcp::RAG_RAT];
    let mut start = Vec::new();
    for plugin in plugins {
        for spec in &plugin.mcp_servers {
            if claimed.contains(&spec.name.as_str()) {
                tracing::info!(
                    plugin = plugin.name,
                    server = spec.name,
                    "MCP server already connected; not starting a second copy"
                );
                continue;
            }
            claimed.push(&spec.name);
            start.push((plugin.name.as_str(), spec));
        }
    }
    start
}

struct AgentSettings<'a> {
    capabilities: &'a [Capability],
    profile: Option<AgentProfile>,
}

/// Apply a stage ruleset first, then the reusable profile policy to the resulting arguments. The
/// profile is the authority boundary, so its decision is always last.
struct ProfilePolicy {
    stage: Arc<dyn ToolPolicy>,
    profile: Arc<dyn ToolPolicy>,
}

impl ToolPolicy for ProfilePolicy {
    fn decide<'a>(
        &'a self,
        tool_name: &'a str,
        args_json: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ToolDecision> + Send + 'a>> {
        Box::pin(async move {
            match self.stage.decide(tool_name, args_json).await {
                ToolDecision::Allow => self.profile.decide(tool_name, args_json).await,
                ToolDecision::Deny(reason) => ToolDecision::Deny(reason),
                ToolDecision::Rewrite(args) => {
                    self.profile.decide(tool_name, &args.to_string()).await
                }
            }
        })
    }
}

/// Resolve a node's agent settings. Ruleset values override the selected profile, which overrides
/// the historic stage-keyed TOML route; `allow` (if given) REPLACES `default_tools`, `deny` is
/// always removed, and `onToolCall` (if defined) becomes the per-call [`ToolPolicy`].
/// Offered tools are narrowed to the stage's effective capability ceiling.
fn node_agent_config(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    mut tools: ToolSet,
    node: &str,
    default_tools: &[&str],
    plugins: &NodePlugins,
    settings: AgentSettings<'_>,
) -> Result<NodeAgentConfig, PlanError> {
    // Offered before narrowing, so `deny` can take them away: a node reasons about a repository,
    // and reading a file it found is the ordinary case rather than the dangerous one. These are
    // also the names a plugin's hooks are written against — `Read`, `Grep`, `Glob` — which is what
    // makes an unmodified plugin's `PreToolUse` fire for a planning node at all.
    // Two different things, and conflating them cost the publisher its `gh`.
    //
    // Whether a node is *offered* the file tools follows from whether it declares any reach: an
    // empty list has to mean none, or "no tools" quietly means "Read, Grep and Glob" and a node
    // meant to transcribe output it was handed goes reading directories on the host.
    //
    // The root is separate, and always set. It is not a capability — it is where a tool resolves
    // paths, and a node with no file tools can do nothing with one. The publisher declares no
    // default tools on purpose, so `gh` cannot be handed to anyone by widening a shared constant,
    // and it is the root that lets `gh` resolve at all. Gating the root on the list left it
    // holding a stand-in that errors, which it dutifully reported as a reason not to publish.
    let files = std::env::current_dir().ok();
    let ceiling = Capability::ceiling(settings.capabilities);
    let profile = settings.profile.as_ref();
    if !default_tools.is_empty()
        && ceiling.is_some_and(|capability| capability.permits(Capability::Read))
    {
        tools
            .local()
            .tools
            .extend(ratatoskr_agent::files::declarations());
    }
    let ruleset = engine.ruleset(node);
    let rc = ruleset.as_ref().map(|r| r.config());

    // Ruleset model FIRST; the selected profile is the reusable fallback before the historic
    // stage-keyed route.
    let route = match rc.and_then(|c| c.model.as_ref()) {
        Some(m) => ratatoskr_core::ModelRoute {
            provider: m.provider.clone(),
            model: m.model.clone(),
            // A ruleset declares which model, not how much of it. The cap comes from the default,
            // which is always sent — so a ruleset naming a brand-new model still works. The window
            // is unstated for the same reason, and falls back to the conservative history budget.
            max_tokens: None,
            context_window: None,
            temperature: None,
            params: None,
            session: Default::default(),
        },
        None => match profile.and_then(|profile| profile.model.clone()) {
            Some(model) => model,
            None => route(config, node)?,
        },
    };

    // A ruleset's `allow` is exhaustive. The default is not just the node's built-in list: those
    // name rag-rat tools, written before any plugin was in the picture, so a plugin the node binds
    // would otherwise contribute a server whose every tool is filtered straight back out.
    let from_plugins = tools.names_beyond(ratatoskr_mcp::RAG_RAT);
    let spelled_out = rc
        .and_then(|c| c.tools.as_ref())
        .and_then(|t| t.allow.as_deref());
    let allow: Vec<String> = match spelled_out {
        Some(a) => a.to_vec(),
        None => default_allow(default_tools, from_plugins.clone()),
    };
    let allowed: Vec<String> = allow
        .iter()
        .filter(|tool| ceiling.is_some_and(|ceiling| ceiling.permits(tools.capability(tool))))
        .cloned()
        .collect();
    let excluded_by_ceiling: Vec<&String> = allow
        .iter()
        .filter(|tool| !ceiling.is_some_and(|ceiling| ceiling.permits(tools.capability(tool))))
        .collect();
    if !excluded_by_ceiling.is_empty() {
        tracing::warn!(
            node,
            excluded = ?excluded_by_ceiling,
            "stage capability ceiling excluded declared tools"
        );
    }
    let allow = allowed;
    let deny: Vec<String> = rc
        .and_then(|c| c.tools.as_ref())
        .map(|t| t.deny.clone())
        .unwrap_or_default();

    // Named but nowhere on offer: a typo, or a tool the server stopped exposing. Reported by name
    // — a count can't be acted on, and a `deny` elsewhere in the ruleset must not explain it away.
    let offered = tools.names();
    let missing: Vec<&String> = allow
        .iter()
        .filter(|n| !offered.contains(n) && !deny.contains(n))
        .collect();
    if !missing.is_empty() {
        // A built-in default list names rag-rat's tools, so in a repository configured without
        // rag-rat every node would warn about every one of them — turning a supported setup into a
        // wall of noise that hides the warning this exists for. An explicit ruleset `allow` is
        // different: it named something by hand, so a name nothing offers is a typo either way.
        if spelled_out.is_none() && !tools.has_server(ratatoskr_mcp::RAG_RAT) {
            tracing::debug!(
                node,
                ?missing,
                "no rag-rat in this repository; these tools are absent by configuration"
            );
        } else {
            tracing::warn!(node, ?missing, "no connected MCP server offers these tools");
        }
    }
    // An `allow` written before the plugin was bound is exhaustive too, so it silently excludes
    // every tool the plugin brought — the node gets that plugin's context and none of its reach.
    if spelled_out.is_some() && !from_plugins.is_empty() {
        let excluded: Vec<&String> = from_plugins.iter().filter(|n| !allow.contains(n)).collect();
        if !excluded.is_empty() {
            tracing::warn!(
                node,
                ?excluded,
                "this node's plugins offer tools its ruleset's `allow` does not name; add them, \
                 or unbind the plugin"
            );
        }
    }
    tools.narrow(&allow, &deny);

    let max_turns = rc
        .and_then(|c| c.max_turns)
        .or_else(|| profile.and_then(|profile| profile.max_turns));
    let system_prompt = rc.and_then(|c| c.system_prompt.clone());
    let stage_policy = match ruleset {
        Some(r) if r.config().has_on_tool_call => Some(Arc::new(r) as Arc<dyn ToolPolicy>),
        _ => None,
    };
    let policy = match (
        stage_policy,
        profile.and_then(|profile| profile.tool_policy.clone()),
    ) {
        (Some(stage), Some(profile)) => {
            Some(Arc::new(ProfilePolicy { stage, profile }) as Arc<dyn ToolPolicy>)
        }
        (Some(stage), None) => Some(stage),
        (None, Some(profile)) => Some(profile),
        (None, None) => None,
    };

    // Every node reaches this function, which is why the skill tool is added here rather than at
    // each construction site: a node that binds a skill and is never offered it is the failure
    // this seam exists to prevent.
    if ceiling.is_some_and(|capability| capability.permits(Capability::Read))
        && let Some(tool) = skills::skill_tool(&plugins.skills, node)
    {
        tools.add_local(tool);
    }

    Ok(NodeAgentConfig {
        route,
        tools,
        capability_ceiling: ceiling,
        files,
        policy,
        max_turns,
        system_prompt,
    })
}

/// Resolve a built-in stage through its selected reusable profile.
pub(crate) fn stage_agent_config(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    tools: ToolSet,
    node: &str,
    default_tools: &[&str],
    plugins: &mut NodePlugins,
) -> Result<NodeAgentConfig, PlanError> {
    let stages = built_in_stages();
    let stage_id = if node == "redteam" { "red_team" } else { node };
    let stage = stages.iter().find(|stage| stage.id == stage_id);
    let profile = stage_profile(config, node);
    plugins.profile_prompt = profile
        .as_ref()
        .map_or_else(String::new, |profile| profile.base_prompt.clone());
    let ceiling = match (stage, profile.as_ref()) {
        (Some(stage), Some(profile)) => stage.effective_ceiling(profile),
        _ => Some(Capability::Read),
    };
    let capabilities = ceiling.into_iter().collect::<Vec<_>>();
    node_agent_config(
        engine,
        config,
        tools,
        node,
        default_tools,
        plugins,
        AgentSettings {
            capabilities: &capabilities,
            profile,
        },
    )
}

/// Resolve the red-team test author. It shares the `redteam` route and ruleset with the optional
/// classifier, but its fixed job is to write tests into the pre-implementation worktree. The
/// classifier stage's read ceiling must therefore not remove `Write` or `Edit` from the author.
pub(crate) fn redteam_author_agent_config(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    tools: ToolSet,
    default_tools: &[&str],
    plugins: &mut NodePlugins,
) -> Result<NodeAgentConfig, PlanError> {
    let profile = stage_profile(config, "redteam");
    plugins.profile_prompt = profile
        .as_ref()
        .map_or_else(String::new, |profile| profile.base_prompt.clone());
    let capabilities = [Capability::Write];
    node_agent_config(
        engine,
        config,
        tools,
        "redteam",
        default_tools,
        plugins,
        AgentSettings {
            capabilities: &capabilities,
            profile,
        },
    )
}

/// Resolve a declared workflow stage through the profile it names.
pub(crate) fn declared_stage_agent_config(
    engine: &Arc<ScriptEngine>,
    config: &RatatoskrConfig,
    tools: ToolSet,
    stage: &Stage,
    default_tools: &[&str],
    plugins: &NodePlugins,
) -> Result<(NodeAgentConfig, AgentProfile), PlanError> {
    let profile = agent_profiles(config)
        .into_iter()
        .find(|profile| profile.id == stage.agent)
        .ok_or_else(|| {
            PlanError::Configuration(format!(
                "stage `{}` references unknown agent `{}`",
                stage.id, stage.agent
            ))
        })?;
    let ceiling = stage.effective_ceiling(&profile);
    let capabilities = ceiling.into_iter().collect::<Vec<_>>();
    let cfg = node_agent_config(
        engine,
        config,
        tools,
        &stage.id,
        default_tools,
        plugins,
        AgentSettings {
            capabilities: &capabilities,
            profile: Some(profile.clone()),
        },
    )?;
    Ok((cfg, profile))
}

/// What a node may call when its ruleset names no tools: its built-in list, plus everything the
/// plugins it binds offer.
///
/// The built-in lists name rag-rat tools and were written before any plugin was in the picture, so
/// on their own they would filter a bound plugin's every tool straight back out — binding a plugin
/// would deliver its session context and none of its capability.
pub(crate) fn default_allow(built_in: &[&str], from_plugins: Vec<String>) -> Vec<String> {
    built_in
        .iter()
        .map(|t| t.to_string())
        .chain(from_plugins)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ruleset directory built for one test.
    async fn binding_engine(case: &str, source: &str) -> Arc<ScriptEngine> {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-nodes-binding-{}-{case}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("agents.ts"), source).unwrap();
        ScriptEngine::load(&dir).await.unwrap()
    }

    struct DenyTool(&'static str);

    impl ToolPolicy for DenyTool {
        fn decide<'a>(
            &'a self,
            tool_name: &'a str,
            _args_json: &'a str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ToolDecision> + Send + 'a>>
        {
            Box::pin(async move {
                if tool_name == self.0 {
                    ToolDecision::Deny("profile policy".to_string())
                } else {
                    ToolDecision::Allow
                }
            })
        }
    }

    #[tokio::test]
    async fn a_profile_policy_applies_to_every_stage_using_the_profile() {
        let engine = binding_engine("profile-policy", "").await;
        let mut config = RatatoskrConfig::default();
        config.agents.insert(
            "reason".to_string(),
            ratatoskr_core::AgentProfileConfig {
                tool_policy: Some(Arc::new(DenyTool("Write"))),
                ..Default::default()
            },
        );

        for node in ["analyst", "bookkeeper"] {
            let cfg = stage_agent_config(
                &engine,
                &config,
                ToolSet::default(),
                node,
                &[],
                &mut NodePlugins::default(),
            )
            .unwrap();
            assert!(matches!(
                cfg.policy.unwrap().decide("Write", "{}").await,
                ToolDecision::Deny(reason) if reason == "profile policy"
            ));
        }
    }

    #[tokio::test]
    async fn redteam_test_author_retains_its_write_tools() {
        let engine = binding_engine("redteam-author-writes", "").await;
        let mut config = RatatoskrConfig::default();
        let route = config.models["analyst"].clone();
        config.models.insert("redteam".to_string(), route);
        let mut tools = ToolSet::default();
        tools
            .local()
            .tools
            .extend(ratatoskr_agent::files::edit_declarations());

        let cfg = redteam_author_agent_config(
            &engine,
            &config,
            tools,
            crate::redteam::AUTHOR_TOOLS,
            &mut NodePlugins::default(),
        )
        .unwrap();

        for required in [ratatoskr_agent::files::WRITE, ratatoskr_agent::files::EDIT] {
            assert!(
                cfg.tools.names().iter().any(|name| name == required),
                "the red-team test author needs {required}: {:?}",
                cfg.tools.names()
            );
        }
    }

    /// A config and cwd that discover exactly `names` — bare plugins, manifest only, so `resolve`
    /// runs no hooks and connects no servers. The cwd holds no `.ratatoskr/plugins` of its own:
    /// what the test discovers must not depend on the real checkout it happens to run in.
    fn resolve_fixture(case: &str, names: &[&str]) -> (RatatoskrConfig, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "ratatoskr-nodes-resolve-{}-{case}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let plugins = root.join("plugins");
        for name in names {
            let manifest = plugins.join(name).join(".claude-plugin");
            std::fs::create_dir_all(&manifest).unwrap();
            std::fs::write(
                manifest.join("plugin.json"),
                format!(r#"{{"name": "{name}"}}"#),
            )
            .unwrap();
        }
        let cwd = root.join("cwd");
        std::fs::create_dir_all(&cwd).unwrap();
        let mut config = RatatoskrConfig::default();
        config.plugins.paths = vec![plugins];
        (config, cwd)
    }

    /// A plugin directory whose `PreToolUse` hook answers with an envelope for `matcher`.
    fn hooking_plugin(name: &str, matcher: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("ratatoskr-node-hook-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".claude-plugin")).unwrap();
        std::fs::create_dir_all(root.join("hooks")).unwrap();
        std::fs::write(
            root.join(".claude-plugin/plugin.json"),
            format!(r#"{{"name": "{name}"}}"#),
        )
        .unwrap();
        std::fs::write(
            root.join("hooks/hooks.json"),
            format!(
                r#"{{"hooks": {{"PreToolUse": [{{"matcher": "{matcher}", "hooks": [
                    {{"type": "command", "command": "cat ${{CLAUDE_PLUGIN_ROOT}}/answer"}}
                ]}}]}}}}"#
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("answer"),
            format!(r#"{{"hookSpecificOutput": {{"additionalContext": "from {name}"}}}}"#),
        )
        .unwrap();
        root
    }

    #[tokio::test]
    async fn a_node_runs_the_tool_hooks_of_the_plugins_it_binds_and_no_others() {
        // The same binding that decides a node's context and its tools decides its hooks.
        let engine = binding_engine(
            "node-hooks",
            r#"
            defineDefaults({ plugins: ["hookful"] });
            defineAgent("analyst", { plugins: { inherit: false } });
            "#,
        )
        .await;
        let hookful = hooking_plugin("hookful", "^semantic_search$");
        let quiet = hooking_plugin("quiet", "^nothing_calls_this$");
        let plugins = ratatoskr_plugin::discover(&[hookful.clone(), quiet.clone()]);

        let context = PluginContext {
            discovered: plugins.iter().map(|p| p.name.clone()).collect(),
            plugins: Arc::new(plugins),
            engine: Some(engine),
            ..Default::default()
        };

        let scout = context.for_node("scout").observer.expect("scout binds it");
        assert_eq!(
            scout.before("semantic_search", "{}").await.as_deref(),
            Some("from hookful")
        );
        // The matcher still decides which calls it sees, and a PreToolUse hook says nothing after.
        assert_eq!(scout.before("impact_surface", "{}").await, None);
        assert_eq!(scout.after("semantic_search", "{}", "result").await, None);

        // A node that binds nothing carries no runner at all, so the hook never reaches its agent.
        assert!(context.for_node("analyst").observer.is_none());

        let _ = std::fs::remove_dir_all(&hookful);
        let _ = std::fs::remove_dir_all(&quiet);
    }

    #[tokio::test]
    async fn a_default_plugin_nobody_installed_narrows_instead_of_failing() {
        // #185: `[rag_rat]` is optional, but the shipped ruleset still declares
        // `defineDefaults({ plugins: ["rag-rat"] })` — and a default applies to every node, so it
        // is a preference. Treating it like an explicit binding failed a rag-rat-less run on the
        // `plugins` node before any work started. (The tracing::warn! the contract asks for is
        // not asserted: capturing the global subscriber races the rest of the test suite.)
        let engine = binding_engine(
            "default-missing",
            r#"defineDefaults({ plugins: ["rag-rat"] });"#,
        )
        .await;
        let (config, cwd) = resolve_fixture("default-missing", &["ponytail"]);

        let context = PluginContext::resolve(&config, &engine, &cwd)
            .await
            .expect("a missing default narrows the tool pool; it does not fail the run");

        // The missing plugin is absent from every node's bound set, and nothing a node binds was
        // not actually discovered. Whether the narrowed set keeps `ponytail` or is empty is the
        // implementer's choice — the contract is only "only discovered plugins".
        for node in ["scout", "memory", "analyst", "implementer", "bookkeeper"] {
            let bound = context.bound(node);
            assert!(
                !bound.iter().any(|name| name == "rag-rat"),
                "{node} still binds a plugin nobody installed: {bound:?}"
            );
            assert!(
                bound.iter().all(|name| name == "ponytail"),
                "{node} binds something that was not discovered: {bound:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_missing_default_beside_a_discovered_agent_add_resolves_as_before() {
        // The narrowing must not disturb the satisfied path: both names discovered, both bound,
        // defaults first then the add.
        let engine = binding_engine(
            "default-and-add",
            r#"
            defineDefaults({ plugins: ["rag-rat"] });
            defineAgent("analyst", { plugins: { add: ["ponytail"] } });
            "#,
        )
        .await;
        let (config, cwd) = resolve_fixture("default-and-add", &["rag-rat", "ponytail"]);

        let context = PluginContext::resolve(&config, &engine, &cwd)
            .await
            .expect("everything named was discovered");
        assert_eq!(context.bound("analyst"), ["rag-rat", "ponytail"]);
        // A node that says nothing still gets exactly the defaults — discovering `ponytail` does
        // not bind it, the ruleset named the pool.
        assert_eq!(context.bound("scout"), ["rag-rat"]);
    }

    #[tokio::test]
    async fn when_everything_declared_is_discovered_resolve_is_unchanged() {
        // The strict path must not regress for a fully satisfied ruleset: no warning, no
        // narrowing, the defaults reach a node that says nothing.
        let engine =
            binding_engine("all-found", r#"defineDefaults({ plugins: ["rag-rat"] });"#).await;
        let (config, cwd) = resolve_fixture("all-found", &["rag-rat", "ponytail"]);

        let context = PluginContext::resolve(&config, &engine, &cwd)
            .await
            .expect("everything named was discovered");
        assert_eq!(context.bound("scout"), ["rag-rat"]);
    }

    #[tokio::test]
    async fn no_plugins_discovered_at_all_still_runs_when_only_defaults_name_one() {
        // The rag-rat-less setup taken to its end: nothing installed, a default naming rag-rat,
        // and the run proceeds on file tools. A default is a preference.
        let engine = binding_engine(
            "none-discovered",
            r#"defineDefaults({ plugins: ["rag-rat"] });"#,
        )
        .await;
        let (config, cwd) = resolve_fixture("none-discovered", &[]);

        let context = PluginContext::resolve(&config, &engine, &cwd)
            .await
            .expect("nothing discovered and only a default declared: there is nothing to require");
        assert!(context.discovered.is_empty());
        for node in ["scout", "analyst", "implementer"] {
            assert!(
                !context.bound(node).iter().any(|name| name == "rag-rat"),
                "nothing was discovered, so nothing undiscovered may be bound"
            );
        }
    }

    #[tokio::test]
    async fn a_node_only_carries_the_context_of_the_plugins_it_binds() {
        // The hooks run once per run; what differs per node is which of their outputs it carries.
        let engine = binding_engine(
            "per-node",
            r#"
            defineDefaults({ plugins: ["everywhere"] });
            defineAgent("analyst", { plugins: { add: ["analyst-only"] } });
            defineAgent("scout", { plugins: { inherit: false } });
            "#,
        )
        .await;

        let context = PluginContext {
            contexts: [
                ("everywhere".to_string(), "SHARED".to_string()),
                ("analyst-only".to_string(), "DEEP".to_string()),
            ]
            .into_iter()
            .collect(),
            discovered: vec!["everywhere".to_string(), "analyst-only".to_string()],
            engine: Some(engine),
            ..Default::default()
        };

        // Defaults first, then what the node added.
        assert_eq!(
            context.for_node("analyst").context.as_deref(),
            Some("SHARED\n\nDEEP")
        );
        // A node that inherits nothing and adds nothing carries nothing.
        assert_eq!(context.for_node("scout").context, None);
        // A node with no ruleset still gets the defaults.
        assert_eq!(
            context.for_node("bookkeeper").context.as_deref(),
            Some("SHARED")
        );
    }

    #[test]
    fn a_denied_skill_is_offered_to_no_node() {
        // A plugin is installed whole, and some of what it ships is written for a person at a
        // keyboard. `init-rag-rat` sets up an unindexed repository by asking questions and being
        // answered — a procedure no node can carry out, costing every node that binds the plugin
        // the space its description takes on every call.
        let context = PluginContext {
            skills_deny: vec!["init-rag-rat".to_string()],
            ..Default::default()
        };
        assert!(context.denied("init-rag-rat"));
        // Spelling is forgiven; the rest of the plugin is untouched.
        assert!(context.denied(" INIT-RAG-RAT "));
        assert!(!context.denied("using-rag-rat"));
        assert!(!context.denied("dream-review"));

        // And a repository that denied nothing keeps everything.
        assert!(!PluginContext::default().denied("init-rag-rat"));
    }
}
