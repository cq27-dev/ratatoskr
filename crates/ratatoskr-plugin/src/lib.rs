//! Plugins in the format coding CLIs already use, so a plugin written once works here too.
//!
//! A plugin is a directory holding a manifest (`.claude-plugin/plugin.json`) and, optionally,
//! `hooks/hooks.json` mapping an event to a command to run. Adopting that layout rather than
//! inventing one keeps the declarative surface tiny: the schema does matching and nothing else,
//! and every plugin's actual intelligence lives in the command it names.
//!
//! Nothing here may fail a run. A plugin that is missing, malformed, slow, or broken is logged and
//! skipped — a node that would have got some extra context simply doesn't.

pub mod registry;
pub mod skill;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use ratatoskr_core::HookLimits;

use regex::Regex;
use serde::Deserialize;
pub use skill::Skill;

/// What a run reports as its `SessionStart` source, and what a `SessionStart` matcher is tested
/// against. A run always begins; it is never resumed, cleared, or compacted.
const SESSION_SOURCE: &str = "startup";

/// Most output read from a single hook before it is cut off and the process killed.
///
/// A hard limit rather than a budget: the *read* has to be bounded, not just the text kept, or a
/// hook that writes without end would be buffered in full before anything could reject it. Set
/// well above any configured [`HookLimits::output_budget`], so it only ever catches a runaway.
const MAX_HOOK_OUTPUT: u64 = 1024 * 1024;

/// A loaded plugin.
#[derive(Debug, Clone)]
pub struct Plugin {
    pub name: String,
    /// The plugin's directory, substituted for `${CLAUDE_PLUGIN_ROOT}` in its commands.
    pub root: PathBuf,
    pub hooks: Vec<Hook>,
    /// MCP servers this plugin brings, in manifest order.
    pub mcp_servers: Vec<McpServerSpec>,
    /// Skills this plugin ships, in name order.
    pub skills: Vec<Skill>,
}

/// One MCP server a plugin declares: how to launch it, over stdio.
///
/// `name` is the key it was declared under, which is the identity two plugins can collide on —
/// the caller decides what to do about that, since only it knows what is already connected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerSpec {
    pub name: String,
    /// Program plus arguments, with `${CLAUDE_PLUGIN_ROOT}` already resolved.
    pub command: Vec<String>,
    pub env: BTreeMap<String, String>,
}

/// What a hook's `matcher` means. Three forms, decided once at load.
///
/// Not "always a regex": the format reads a matcher of only letters, digits, `_`, `-`, spaces,
/// `,` and `|` as an exact name or a `|`/`,`-separated list of them. Treating those as a regex
/// over-fires — `Write|Edit` would match `NotebookEdit` and `MultiEdit` too.
#[derive(Debug, Clone)]
enum Matcher {
    /// No matcher, the empty string, or `*`.
    Everything,
    /// One or more exact names.
    Exactly(Vec<String>),
    /// An unanchored regular expression.
    Pattern(Regex),
    /// A pattern that would not compile. Matches nothing, so a broken one costs its own hook.
    Nothing,
}

impl Matcher {
    fn parse(matcher: Option<&str>) -> Self {
        let Some(raw) = matcher.map(str::trim) else {
            return Matcher::Everything;
        };
        if raw.is_empty() || raw == "*" {
            return Matcher::Everything;
        }
        // The "safe" set. Anything outside it makes the whole value a regex.
        if raw
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ' ' | ',' | '|'))
        {
            let names: Vec<String> = raw
                .split(['|', ','])
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .map(str::to_string)
                .collect();
            if names.is_empty() {
                tracing::warn!("ignoring hook matcher `{raw}`: it names nothing");
                return Matcher::Nothing;
            }
            return Matcher::Exactly(names);
        }
        match Regex::new(raw) {
            Ok(re) => Matcher::Pattern(re),
            Err(e) => {
                tracing::warn!("ignoring hook matcher `{raw}`: {e}");
                Matcher::Nothing
            }
        }
    }

    fn matches(&self, subject: &str) -> bool {
        match self {
            Matcher::Everything => true,
            Matcher::Exactly(names) => names.iter().any(|n| n == subject),
            Matcher::Pattern(re) => re.is_match(subject),
            Matcher::Nothing => false,
        }
    }
}

/// One hook a plugin registers.
#[derive(Debug, Clone)]
pub struct Hook {
    pub event: String,
    /// The matcher as written, for logs.
    pub matcher: Option<String>,
    /// What it means, decided at load because it is evaluated on every tool call.
    rule: Matcher,
    pub command: String,
    /// Argument list. When present, `command` is an executable spawned directly, with no shell.
    pub args: Option<Vec<String>>,
    /// `bash` or `powershell`; the format defaults to `bash`. Ignored when `args` is set.
    pub shell: Option<String>,
    /// The hook's own `timeout`, in seconds, when it declares one. Resolved against
    /// [`HookLimits`] at run time rather than at load, so a config change needs no reload.
    pub timeout: Option<u64>,
}

impl Hook {
    /// How long this hook may run: what it asked for, defaulted and capped by `limits`.
    pub fn timeout(&self, limits: &HookLimits) -> Duration {
        let asked = self.timeout.unwrap_or(limits.timeout_secs);
        Duration::from_secs(asked.min(limits.max_timeout_secs))
    }

    /// Whether this hook fires for `subject` — a tool name for the tool events, and the session
    /// `source` for `SessionStart`, which is what its matcher is written against.
    pub fn matches(&self, subject: &str) -> bool {
        self.rule.matches(subject)
    }
}

#[derive(Debug, Deserialize)]
struct Manifest {
    name: Option<String>,
    /// `mcpServers` and `hooks` are both `string | array | object`. Typing either as only the
    /// object form is not a missing feature — serde fails the whole manifest, `load` returns
    /// `None`, and the plugin vanishes along with everything else it ships.
    #[serde(default, rename = "mcpServers")]
    mcp_servers: Option<Source<McpServers>>,
    #[serde(default)]
    hooks: Option<Source<HookEvents>>,
}

/// A component the manifest either points at or spells out — or several of both.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Source<T> {
    /// A path relative to the plugin root, or a bundle we cannot open.
    Path(String),
    Inline(T),
    Many(Vec<Source<T>>),
}

impl<T: for<'de> Deserialize<'de>> Source<T> {
    /// Every inline value this source resolves to, in declaration order — later wins on a
    /// conflict, which is how the manifest overrides a file the plugin also ships.
    fn resolve(self, root: &Path, what: &str) -> Vec<T> {
        match self {
            Source::Inline(value) => vec![value],
            Source::Many(sources) => sources
                .into_iter()
                .flat_map(|s| s.resolve(root, what))
                .collect(),
            Source::Path(path) => {
                // A bundle is a packaging format, not JSON we can read.
                if path.ends_with(".mcpb") || path.ends_with(".dxt") {
                    tracing::warn!("ignoring {what} bundle `{path}`: bundles are not supported");
                    return Vec::new();
                }
                // The format requires a `./` path relative to the plugin root. Absolute paths
                // discard the root entirely when joined, and `..` walks out of it; neither is a
                // component of *this* plugin, which is all this key can name.
                let relative = path.trim_start_matches("./");
                if Path::new(relative)
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
                {
                    tracing::warn!("ignoring {what} path `{path}`: it leaves the plugin directory");
                    return Vec::new();
                }
                let full = root.join(relative);
                let Ok(raw) = std::fs::read_to_string(&full) else {
                    tracing::warn!("ignoring {what} path `{}`: cannot read it", full.display());
                    return Vec::new();
                };
                match serde_json::from_str::<T>(&raw) {
                    Ok(value) => vec![value],
                    Err(e) => {
                        tracing::warn!("ignoring {what} at {}: {e}", full.display());
                        Vec::new()
                    }
                }
            }
        }
    }
}

/// A server declaration in either spelling: the file shape a `.mcp.json` uses, and the bare map
/// the manifest's inline form uses.
///
/// `Wrapped` requires its key rather than defaulting it — with a default, an inline block would
/// match it and resolve to nothing, because every field of a `ServerEntry` is optional and so
/// almost any object parses as a bare map.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum McpServers {
    Wrapped {
        #[serde(rename = "mcpServers")]
        mcp_servers: BTreeMap<String, ServerEntry>,
    },
    Bare(BTreeMap<String, ServerEntry>),
}

impl McpServers {
    fn into_servers(self) -> BTreeMap<String, ServerEntry> {
        match self {
            McpServers::Wrapped { mcp_servers } => mcp_servers,
            McpServers::Bare(servers) => servers,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ServerEntry {
    /// Absent for the transports we don't speak (`http`, `sse`); such an entry is skipped.
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

/// A hooks declaration in either spelling: the file shape a `hooks.json` uses, and the bare event
/// map the manifest's inline form uses. Both occur, so both are read.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum HookEvents {
    /// `{"hooks": {"PreToolUse": [...]}}` — and any sibling keys, which are ignored. The key is
    /// required, not defaulted: a default would make a bare event map match this variant and
    /// resolve to nothing.
    Wrapped {
        hooks: BTreeMap<String, Vec<HookGroup>>,
    },
    /// `{"PreToolUse": [...]}`
    Bare(BTreeMap<String, Vec<HookGroup>>),
}

impl HookEvents {
    fn into_events(self) -> BTreeMap<String, Vec<HookGroup>> {
        match self {
            HookEvents::Wrapped { hooks } => hooks,
            HookEvents::Bare(events) => events,
        }
    }
}

#[derive(Debug, Deserialize)]
struct HookGroup {
    #[serde(default)]
    matcher: Option<String>,
    #[serde(default)]
    hooks: Vec<HookEntry>,
}

#[derive(Debug, Deserialize)]
struct HookEntry {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    command: String,
    #[serde(default)]
    args: Option<Vec<String>>,
    #[serde(default)]
    shell: Option<String>,
    #[serde(default)]
    timeout: Option<u64>,
}

/// Load every plugin under `dirs`.
///
/// Each entry is either a plugin directory itself or a directory *of* plugin directories, because
/// both conventions exist in the wild: a repo-local `.ratatoskr/plugins/` holds several, while a
/// path to an installed plugin points at one.
pub fn discover(dirs: &[PathBuf]) -> Vec<Plugin> {
    let mut found = Vec::new();
    for dir in dirs {
        if let Some(plugin) = load(dir) {
            found.push(plugin);
            continue;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if let Some(plugin) = load(&entry.path()) {
                found.push(plugin);
            }
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found.dedup_by(|a, b| a.root == b.root);
    one_per_name(found, &home())
}

/// The user's home directory, where a coding CLI records which plugins it has installed.
fn home() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default()
}

/// One plugin per name.
///
/// A path naming a plugin in a coding CLI's cache holds *every version ever installed*, because
/// that cache is laid out `<marketplace>/<plugin>/<version>/` and older versions are kept. Loading
/// them all runs one plugin's `SessionStart` hooks once per version and leaves which copy answers
/// to directory order.
///
/// The host records the current one, so that is used when it is known; otherwise the first found
/// is kept, which is at least deterministic. Either way the run is told what it is not using.
fn one_per_name(found: Vec<Plugin>, home: &Path) -> Vec<Plugin> {
    let installed = registry::installed(home);
    let mut kept: Vec<Plugin> = Vec::new();

    for plugin in found {
        let current = registry::is_current(&installed, &plugin.name, &plugin.root);
        match kept.iter().position(|k| k.name == plugin.name) {
            None => kept.push(plugin),
            Some(at) => {
                // Prefer the copy the host says is installed; failing that, keep the first.
                let replace = current == Some(true)
                    && registry::is_current(&installed, &kept[at].name, &kept[at].root)
                        != Some(true);
                let (dropped, reason) = match replace {
                    true => (kept[at].root.clone(), "superseded by the installed copy"),
                    false => (plugin.root.clone(), "another copy is already loaded"),
                };
                tracing::info!(
                    plugin = plugin.name,
                    path = %dropped.display(),
                    "not loading this copy of the plugin: {reason}"
                );
                if replace {
                    kept[at] = plugin;
                }
            }
        }
    }
    kept
}

/// Read one plugin directory, or `None` if it isn't one.
fn load(root: &Path) -> Option<Plugin> {
    let manifest_path = root.join(".claude-plugin/plugin.json");
    let raw = std::fs::read_to_string(&manifest_path).ok()?;
    let manifest: Manifest = match serde_json::from_str(&raw) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("ignoring plugin at {}: {e}", root.display());
            return None;
        }
    };
    // Fall back to the directory name: a manifest without a name is still a usable plugin, and
    // refusing it would be pedantry.
    let name = manifest.name.unwrap_or_else(|| {
        root.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("plugin")
            .to_string()
    });

    Some(Plugin {
        name,
        hooks: read_hooks(root, manifest.hooks),
        mcp_servers: read_mcp_servers(root, manifest.mcp_servers),
        skills: skill::read_skills(root),
        root: root.to_path_buf(),
    })
}

/// The MCP servers a plugin declares, from its manifest and from a sibling `.mcp.json`.
///
/// Both spellings are in use, and a plugin commonly carries the same block twice so it works in
/// hosts that read one or the other. The manifest wins on a shared name so the file cannot
/// silently redirect a server the manifest already described.
fn read_mcp_servers(root: &Path, manifest: Option<Source<McpServers>>) -> Vec<McpServerSpec> {
    // `.mcp.json` first, then everything the manifest names, so the manifest wins a shared key.
    let mut declared: BTreeMap<String, ServerEntry> = BTreeMap::new();
    if let Ok(raw) = std::fs::read_to_string(root.join(".mcp.json")) {
        match serde_json::from_str::<McpServers>(&raw) {
            Ok(file) => declared.extend(file.into_servers()),
            Err(e) => tracing::warn!("ignoring .mcp.json for plugin at {}: {e}", root.display()),
        }
    }
    for block in manifest
        .into_iter()
        .flat_map(|s| s.resolve(root, "mcpServers"))
    {
        declared.extend(block.into_servers());
    }

    declared
        .into_iter()
        .filter_map(|(name, entry)| {
            // A remote-transport entry names no command. Nothing here can launch it, and pretending
            // otherwise would spawn the wrong thing.
            let program = entry.command?;
            let command = std::iter::once(program)
                .chain(entry.args)
                .map(|s| substitute_root(&s, root))
                .collect();
            let env = entry
                .env
                .into_iter()
                .map(|(k, v)| (k, substitute_root(&v, root)))
                .collect();
            Some(McpServerSpec { name, command, env })
        })
        .collect()
}

/// Resolve `${CLAUDE_PLUGIN_ROOT}` — how a plugin addresses its own files.
///
/// Hooks get this through the environment (a shell expands it), but a server is launched directly,
/// with no shell to do the expansion.
fn substitute_root(text: &str, root: &Path) -> String {
    text.replace("${CLAUDE_PLUGIN_ROOT}", &root.display().to_string())
}

/// The hooks a plugin registers: the conventional `hooks/hooks.json`, plus anything the manifest's
/// `hooks` key names.
///
/// The manifest key *adds* rather than replaces, so a plugin can carry both — and one that carries
/// only the key (naming a differently-named file) is not silently left with no hooks at all.
fn read_hooks(root: &Path, manifest: Option<Source<HookEvents>>) -> Vec<Hook> {
    let mut events: BTreeMap<String, Vec<HookGroup>> = BTreeMap::new();

    if let Ok(raw) = std::fs::read_to_string(root.join("hooks/hooks.json")) {
        match serde_json::from_str::<HookEvents>(&raw) {
            Ok(file) => events.extend(file.into_events()),
            Err(e) => tracing::warn!("ignoring hooks for plugin at {}: {e}", root.display()),
        }
    }
    for block in manifest.into_iter().flat_map(|s| s.resolve(root, "hooks")) {
        for (event, groups) in block.into_events() {
            events.entry(event).or_default().extend(groups);
        }
    }

    events
        .into_iter()
        .flat_map(|(event, groups)| {
            groups.into_iter().flat_map(move |group| {
                let (event, matcher) = (event.clone(), group.matcher.clone());
                group.hooks.into_iter().filter_map(move |entry| {
                    // `command` is the only type in use; anything else is a format we don't speak.
                    if entry.kind.as_deref().unwrap_or("command") != "command" {
                        return None;
                    }
                    Some(Hook {
                        event: event.clone(),
                        // Decided once at load: matched on every tool call.
                        rule: Matcher::parse(matcher.as_deref()),
                        matcher: matcher.clone(),
                        command: entry.command,
                        args: entry.args,
                        shell: entry.shell,
                        timeout: entry.timeout,
                    })
                })
            })
        })
        .collect()
}

/// Run every plugin's `SessionStart` hooks and keep each plugin's output under its own name.
///
/// Per plugin rather than one joined string because nodes bind different sets: the hooks run once
/// per run, and each node composes from this map. `SessionStart` answers with plain text on stdout
/// — no envelope — and silence is the normal "nothing to say".
pub async fn session_start(
    plugins: &[Plugin],
    cwd: &Path,
    limits: &HookLimits,
) -> BTreeMap<String, String> {
    let mut contexts = BTreeMap::new();

    for plugin in plugins {
        let mut parts: Vec<String> = Vec::new();
        for hook in plugin
            .hooks
            .iter()
            // `SessionStart`'s matcher is written against the *source*, not a tool name — a hook
            // scoped to `resume` or `compact` describes a session this host never has.
            .filter(|h| h.event == "SessionStart" && h.matches(SESSION_SOURCE))
        {
            let payload = serde_json::json!({
                "session_id": "",
                "cwd": cwd.display().to_string(),
                "hook_event_name": "SessionStart",
                "source": SESSION_SOURCE,
            });
            let Some(text) = run_hook(plugin, hook, hook.timeout(limits), &payload, cwd).await
            else {
                continue;
            };
            let text = text.trim();
            if !text.is_empty() {
                parts.push(text.to_string());
            }
        }
        if parts.is_empty() {
            continue;
        }
        // Whole plugins in or out, decided here rather than at composition: half a digest is
        // worse than none of one, and refusing it now means nothing over the budget is held
        // resident for the run.
        let text = parts.join("\n\n");
        if text.len() > limits.output_budget {
            tracing::warn!(
                plugin = plugin.name,
                chars = text.len(),
                "dropping session context: over budget"
            );
            continue;
        }
        contexts.insert(plugin.name.clone(), text);
    }
    contexts
}

/// One tool call, as the plugins around it see it.
pub struct ToolEvent<'a> {
    /// `PreToolUse` before the call, `PostToolUse` after it.
    pub event: &'a str,
    pub tool: &'a str,
    /// The call's JSON arguments.
    pub input: &'a str,
    /// What the tool answered — `PostToolUse` only.
    pub response: Option<&'a str>,
}

/// A tool hook's answer. Only `additionalContext` is read.
///
/// The other fields of this envelope decide whether a call proceeds, and that is not a plugin's
/// call to make here: gating already has an owner in a ruleset's `onToolCall`, which is the
/// repository's decision about its own agents rather than a third party's about them.
#[derive(Debug, Deserialize)]
struct HookEnvelope {
    #[serde(rename = "hookSpecificOutput")]
    hook_specific_output: Option<SpecificOutput>,
}

#[derive(Debug, Deserialize)]
struct SpecificOutput {
    #[serde(rename = "additionalContext")]
    additional_context: Option<String>,
}

/// Run the hooks `plugins` register for one tool call, and return what they want the model to see.
///
/// Every matching hook runs; their answers are joined in plugin order and capped at
/// [`TOOL_CONTEXT_BUDGET`], whole hooks in or out. A hook that fails, times out, or answers with
/// something that isn't the envelope contributes nothing — this text is an aside on a tool result,
/// and nothing about it may disturb the call it rode in on.
///
/// `spent` accumulates the wall-clock these hooks cost; once it passes [`TOOL_HOOK_BUDGET`] this
/// returns immediately without running anything, for the rest of the run.
pub async fn tool_hooks(
    plugins: &[Plugin],
    event: ToolEvent<'_>,
    cwd: &Path,
    limits: &HookLimits,
    spent: &AtomicU64,
) -> Option<String> {
    let budget = tool_time_budget(limits);
    if budget.is_some_and(|b| Duration::from_millis(spent.load(Ordering::Relaxed)) >= b) {
        return None;
    }
    let started = std::time::Instant::now();
    let mut parts: Vec<String> = Vec::new();
    let mut used = 0usize;

    for plugin in plugins {
        for hook in plugin
            .hooks
            .iter()
            .filter(|h| h.event == event.event && h.matches(event.tool))
        {
            let mut payload = serde_json::json!({
                "session_id": "",
                "cwd": cwd.display().to_string(),
                "hook_event_name": event.event,
                "tool_name": event.tool,
                "tool_input": serde_json::from_str::<serde_json::Value>(event.input)
                    .unwrap_or(serde_json::Value::Null),
            });
            if let Some(response) = event.response {
                payload["tool_response"] = serde_json::json!(response);
            }

            let Some(raw) = run_hook(plugin, hook, hook.timeout(limits), &payload, cwd).await
            else {
                continue;
            };
            let Some(text) = additional_context(&raw, &plugin.name) else {
                continue;
            };
            // Whole hooks in or out: half an aside is worse than none of one.
            if used + text.len() > limits.output_budget {
                tracing::debug!(
                    plugin = plugin.name,
                    tool = event.tool,
                    "dropping a tool hook's context: over budget"
                );
                continue;
            }
            used += text.len();
            parts.push(text);
        }
    }

    // Warn exactly on the call that exhausts it, so a run says once that its plugins went quiet.
    let cost = started.elapsed();
    let before = Duration::from_millis(spent.fetch_add(cost.as_millis() as u64, Ordering::Relaxed));
    if let Some(budget) = budget
        && before < budget
        && before + cost >= budget
    {
        tracing::warn!(
            "plugins have spent their {budget:?} of tool-hook time; \
             no more tool hooks will run this run"
        );
    }
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// The run's tool-hook time budget, or `None` when it is unlimited.
fn tool_time_budget(limits: &HookLimits) -> Option<Duration> {
    (limits.tool_time_budget_secs > 0).then(|| Duration::from_secs(limits.tool_time_budget_secs))
}

/// Read `additionalContext` out of a hook's envelope, or nothing.
///
/// Silence and an empty envelope are both ordinary — "nothing to say" is the common answer. Output
/// that is not the envelope at all is a plugin written against a different contract, and is logged
/// once rather than pasted into the model's context as-is.
fn additional_context(raw: &str, plugin: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let envelope: HookEnvelope = serde_json::from_str(raw)
        .inspect_err(|e| tracing::warn!(plugin, "ignoring a tool hook's output: {e}"))
        .ok()?;
    let text = envelope.hook_specific_output?.additional_context?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// Compose one node's session context from the plugins it is bound to, in binding order.
///
/// Capped at [`CONTEXT_BUDGET`] and truncated between plugins rather than mid-sentence: this text
/// is prepended to the node's preamble, so it is paid for on every model call that node makes.
pub fn compose(
    contexts: &BTreeMap<String, String>,
    names: &[String],
    limits: &HookLimits,
) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    let mut used = 0usize;

    for name in names {
        let Some(text) = contexts.get(name) else {
            continue;
        };
        // Whole plugins in or out: half a digest is worse than none of one.
        if used + text.len() > limits.context_budget {
            tracing::debug!("dropping {name}'s session context: over budget");
            continue;
        }
        used += text.len();
        parts.push(text);
    }

    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// Run one hook and return its stdout, or `None` if it had nothing to say or could not be run.
async fn run_hook(
    plugin: &Plugin,
    hook: &Hook,
    timeout: Duration,
    payload: &serde_json::Value,
    cwd: &Path,
) -> Option<String> {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    let mut command = match &hook.args {
        // Exec form: `command` is an executable, spawned directly with no shell and no
        // tokenization. A shell here would re-split arguments the plugin already separated.
        Some(args) => {
            let mut command = tokio::process::Command::new(&hook.command);
            command.args(args);
            command
        }
        // Shell form: the command carries its own quoting. The format's default shell is `bash`,
        // and plugins write for it — `sh` is a different language for anything beyond the basics.
        None => {
            let shell = hook.shell.as_deref().unwrap_or("bash");
            let mut command = tokio::process::Command::new(shell);
            command.arg("-c").arg(&hook.command);
            command
        }
    };

    // A hook inherits this process's environment, which in a run started from a coding CLI already
    // holds that host's `CLAUDE_*` values — a plugin would then read another host's data directory
    // as its own. Clear them all and set the three this host actually defines.
    for key in inherited_host_vars(std::env::vars_os().map(|(k, _)| k)) {
        command.env_remove(&key);
    }
    let mut child = command
        // Plugins address their own files through these; the shell expands them from the
        // environment, so no path is ever spliced into the command text.
        .env("CLAUDE_PLUGIN_ROOT", &plugin.root)
        .env("CLAUDE_PROJECT_DIR", cwd)
        .env("CLAUDE_PLUGIN_DATA", plugin_data_dir(&plugin.name, cwd))
        .current_dir(cwd)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        // A hook that outlives its timeout is killed rather than orphaned to init.
        .kill_on_drop(true)
        .spawn()
        .inspect_err(|e| tracing::warn!("plugin {} hook failed to start: {e}", plugin.name))
        .ok()?;

    let mut stdin = child.stdin.take();
    let mut stdout = child.stdout.take();
    let payload = payload.to_string();

    // Write and read concurrently. A `PostToolUse` payload carries what the tool answered and can
    // exceed the pipe buffer, so a hook that does not read its stdin leaves the write parked until
    // the hook exits and the pipe breaks. Nothing hangs either way — the timeout covers both — but
    // reading alongside means a hook's answer is collected as it arrives rather than after a write
    // nobody was listening to. Dropping `stdin` at the end of the write is what lets a hook that
    // reads to EOF proceed.
    let collect = async {
        let write = async {
            if let Some(mut stdin) = stdin.take() {
                let _ = stdin.write_all(payload.as_bytes()).await;
            }
        };
        let read = async {
            let mut buf = Vec::new();
            if let Some(stdout) = stdout.as_mut() {
                let _ = stdout.take(MAX_HOOK_OUTPUT).read_to_end(&mut buf).await;
            }
            let capped = buf.len() as u64 >= MAX_HOOK_OUTPUT;
            (buf, capped)
        };
        let (_, (buf, capped)) = tokio::join!(write, read);
        if capped {
            // Its output is already past any budget it could have fit, and a hook that writes
            // without end will not exit — waiting on it would spend the whole timeout to reach a
            // result certain to be dropped.
            return (buf, capped, None);
        }
        // Inside the timeout, not after it. Reaching EOF on stdout does not mean the hook has
        // finished: one that closes or redirects its own output and keeps working would otherwise
        // be awaited here with no bound at all, and its declared timeout would mean nothing.
        (buf, capped, child.wait().await.ok())
    };

    let Ok((buf, capped, status)) = tokio::time::timeout(timeout, collect).await else {
        tracing::warn!("plugin {} hook timed out after {timeout:?}", plugin.name);
        return None;
    };
    if capped {
        let _ = child.start_kill();
        tracing::warn!("plugin {} hook wrote without stopping", plugin.name);
        return None;
    }

    // Only exit 0 means the output is an answer. The format is explicit that stdout is read only
    // on success; a hook that failed is telling us so on stderr, and treating what it managed to
    // print as context is how a broken hook quietly steers a node.
    match status {
        Some(status) if status.success() => Some(String::from_utf8_lossy(&buf).into_owned()),
        Some(status) => {
            tracing::warn!("plugin {} hook exited with {status}", plugin.name);
            None
        }
        None => {
            tracing::warn!("plugin {} hook could not be waited on", plugin.name);
            None
        }
    }
}

/// The variables of *another* host that a hook must not inherit from this process.
///
/// Separate from the spawn so the rule is testable without mutating the process environment, which
/// is global to every test running beside it.
fn inherited_host_vars(keys: impl Iterator<Item = std::ffi::OsString>) -> Vec<std::ffi::OsString> {
    keys.filter(|k| k.to_string_lossy().starts_with("CLAUDE_"))
        .collect()
}

/// Where a plugin keeps state that outlives one run.
///
/// Deliberately under the project rather than in the coding CLI's own plugin directory: this is a
/// different host, and a plugin's data here is not that host's to read or ours to overwrite.
fn plugin_data_dir(plugin: &str, cwd: &Path) -> PathBuf {
    // Escaped rather than replaced: mapping every unsafe character to one `-` would give
    // `acme.tools` and `acme-tools` the same directory, and each would then read and overwrite the
    // other's state. `_` escapes itself, which is what keeps the encoding reversible — and so
    // collision-free — while leaving an ordinary kebab-case name unchanged.
    let mut safe = String::with_capacity(plugin.len());
    for c in plugin.chars() {
        match c {
            '_' => safe.push_str("_5f"),
            c if c.is_ascii_alphanumeric() || c == '-' => safe.push(c),
            c => safe.push_str(&format!("_{:x}", c as u32)),
        }
    }
    let dir = cwd.join(".ratatoskr/plugin-data").join(safe);
    // The format's own host creates it on first reference; a hook that appends to a file in it
    // should not have to make the directory first.
    let _ = std::fs::create_dir_all(&dir);
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The format's own defaults, which is what a repo gets unless `ratatoskr.toml` says otherwise.
    fn limits() -> HookLimits {
        HookLimits::default()
    }

    #[test]
    fn mcp_servers_are_read_from_either_spelling_and_rooted() {
        // Plugins commonly carry the same block twice so it works in hosts that read one or the
        // other; `${CLAUDE_PLUGIN_ROOT}` is how a plugin addresses its own files, and a server is
        // launched with no shell to expand it.
        let root = plugin_dir(
            "servers",
            r#"{
                "name": "served",
                "mcpServers": {
                    "local": {
                        "command": "${CLAUDE_PLUGIN_ROOT}/bin/serve",
                        "args": ["--root", "${CLAUDE_PLUGIN_ROOT}"],
                        "env": { "DATA": "${CLAUDE_PLUGIN_ROOT}/data" }
                    },
                    "remote": { "type": "http", "url": "https://example.invalid/mcp" }
                }
            }"#,
            None,
        );
        std::fs::write(
            root.join(".mcp.json"),
            r#"{"mcpServers": {
                "local": {"command": "should-not-win"},
                "extra": {"command": "npx", "args": ["-y", "extra"]}
            }}"#,
        )
        .unwrap();

        let found = discover(std::slice::from_ref(&root));
        let here = root.display().to_string();
        assert_eq!(
            found[0]
                .mcp_servers
                .iter()
                .map(|s| (s.name.as_str(), s.command.clone()))
                .collect::<Vec<_>>(),
            [
                (
                    "extra",
                    vec!["npx".to_string(), "-y".to_string(), "extra".to_string()]
                ),
                (
                    "local",
                    vec![
                        format!("{here}/bin/serve"),
                        "--root".to_string(),
                        here.clone()
                    ]
                ),
            ],
            "the manifest wins on a shared name, and a transport we can't launch is skipped"
        );
        assert_eq!(
            found[0].mcp_servers[1].env.get("DATA"),
            Some(&format!("{here}/data"))
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A plugin whose PreToolUse hook echoes an envelope, matching `matcher`.
    fn tool_plugin(case: &str, matcher: &str, event: &str, answer: &str) -> PathBuf {
        let root = plugin_dir(
            case,
            &format!(r#"{{"name": "{case}"}}"#),
            Some(&format!(
                r#"{{"hooks": {{"{event}": [{{"matcher": "{matcher}", "hooks": [
                    {{"type": "command", "command": "cat ${{CLAUDE_PLUGIN_ROOT}}/answer"}}
                ]}}]}}}}"#
            )),
        );
        std::fs::write(root.join("answer"), answer).unwrap();
        root
    }

    const ENVELOPE: &str = r#"{"hookSpecificOutput":
        {"hookEventName": "PreToolUse", "additionalContext": "  mind the clones  "}}"#;

    #[test]
    fn a_component_the_manifest_points_at_is_read_rather_than_losing_the_plugin() {
        // `mcpServers` and `hooks` are both `string | array | object`. Typed as only the object
        // form, the path spelling failed the whole manifest and the plugin disappeared with it.
        let root = plugin_dir(
            "pointed",
            r#"{ "name": "pointed",
                 "mcpServers": ["./one.json", { "inline": { "command": "true" } }],
                 "hooks": "./extra-hooks.json" }"#,
            // Also a conventional hooks file: the manifest key adds to it rather than replacing.
            Some(
                r#"{"hooks": {"PreToolUse": [{"matcher": "Read",
                "hooks": [{"type": "command", "command": "conventional"}]}]}}"#,
            ),
        );
        std::fs::write(
            root.join("one.json"),
            r#"{"mcpServers": {"from-path": {"command": "npx", "args": ["-y", "x"]}}}"#,
        )
        .unwrap();
        std::fs::write(
            root.join("extra-hooks.json"),
            r#"{"hooks": {"PostToolUse": [{"hooks": [{"type": "command", "command": "extra"}]}]}}"#,
        )
        .unwrap();

        let found = discover(std::slice::from_ref(&root));
        assert_eq!(found.len(), 1, "the plugin loads at all");
        assert_eq!(
            found[0]
                .mcp_servers
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["from-path", "inline"],
            "an array mixing a path and an inline block resolves to both"
        );
        let commands: Vec<&str> = found[0].hooks.iter().map(|h| h.command.as_str()).collect();
        assert!(commands.contains(&"conventional") && commands.contains(&"extra"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_safe_matcher_is_an_exact_list_and_not_a_regex() {
        // The defect this replaces: as an unanchored regex, `Write|Edit` also caught
        // `NotebookEdit` and `MultiEdit`, so a hook fired on tools it never asked for.
        let list = Matcher::parse(Some("Write|Edit"));
        assert!(list.matches("Write") && list.matches("Edit"));
        assert!(!list.matches("NotebookEdit") && !list.matches("MultiEdit"));

        // Commas separate too, with whitespace around them, and hyphens stay exact.
        let spaced = Matcher::parse(Some("Edit, Write"));
        assert!(spaced.matches("Edit") && spaced.matches("Write"));
        let hyphen = Matcher::parse(Some("code-reviewer"));
        assert!(hyphen.matches("code-reviewer") && !hyphen.matches("senior-code-reviewer"));

        // One character outside the safe set makes the whole value a regex, still unanchored.
        let pattern = Matcher::parse(Some("^Notebook"));
        assert!(pattern.matches("NotebookEdit") && !pattern.matches("Edit"));
        let unanchored = Matcher::parse(Some("mcp__memory__.*"));
        assert!(unanchored.matches("mcp__memory__get"));

        // Absent, empty, and `*` all mean everything.
        for all in [None, Some(""), Some("*")] {
            assert!(Matcher::parse(all).matches("anything at all"));
        }
    }

    #[tokio::test]
    async fn a_hook_gets_this_hosts_variables() {
        // Which are set, and point where this host says. That the *other* host's are cleared is
        // `another_hosts_variables_are_the_ones_cleared`.
        // Reported through the exec form, so no shell quoting stands between us and the values.
        let root = tool_plugin("environment", ".*", "SessionStart", "");
        std::fs::write(
            root.join("hooks/hooks.json"),
            r#"{"hooks": {"SessionStart": [{"hooks": [{"type": "command",
                "command": "sh", "args": ["-c",
                "echo [$CLAUDE_PROJECT_DIR][$CLAUDE_PLUGIN_DATA]"]}]}]}}"#,
        )
        .unwrap();

        let plugins = discover(std::slice::from_ref(&root));
        let contexts = session_start(&plugins, Path::new("."), &limits()).await;
        let seen = contexts.get("environment").expect("the hook ran").clone();

        assert!(
            !seen.contains("/somewhere/else") && !seen.contains("/not/this/project"),
            "the surrounding host's values did not leak in: {seen}"
        );
        assert!(
            seen.contains("[.]"),
            "the project directory is this run's: {seen}"
        );
        assert!(
            seen.contains(".ratatoskr/plugin-data/environment"),
            "the data directory belongs to this host and this plugin: {seen}"
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(".ratatoskr/plugin-data/environment");
    }

    #[tokio::test]
    async fn a_hook_that_closes_its_output_and_keeps_going_is_still_bounded() {
        // Reaching EOF on stdout does not mean the hook has finished. Waiting for it to exit
        // after that, outside the timeout, is how a one-second hook takes as long as it likes.
        let root = tool_plugin("lingering", ".*", "SessionStart", "");
        std::fs::write(
            root.join("hooks/hooks.json"),
            r#"{"hooks": {"SessionStart": [{"hooks": [{"type": "command",
                "command": "sh", "args": ["-c", "echo done; exec 1>&-; sleep 30"],
                "timeout": 1}]}]}}"#,
        )
        .unwrap();

        let plugins = discover(std::slice::from_ref(&root));
        let started = std::time::Instant::now();
        let _ = session_start(&plugins, Path::new("."), &limits()).await;
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the hook's own timeout still bounds it: took {:?}",
            started.elapsed()
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn two_plugin_names_never_share_a_data_directory() {
        // Replacing every unsafe character with one `-` gave these the same directory, and each
        // would then read and overwrite the other's state.
        let cwd = Path::new(".");
        let distinct: Vec<PathBuf> = ["acme.tools", "acme-tools", "acme/tools", "acme_tools"]
            .iter()
            .map(|n| plugin_data_dir(n, cwd))
            .collect();
        let mut unique = distinct.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), distinct.len(), "{distinct:?}");
        // And none of them escapes the directory they belong in.
        for dir in &distinct {
            assert!(dir.starts_with("./.ratatoskr/plugin-data"), "{dir:?}");
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn a_component_path_cannot_leave_the_plugin() {
        // `..` walks out of the plugin root, and an absolute path discards it entirely.
        let root = plugin_dir(
            "escapee",
            r#"{"name": "escapee", "mcpServers": ["../../../etc/passwd", "/etc/shadow"]}"#,
            None,
        );
        let found = discover(std::slice::from_ref(&root));
        assert_eq!(found.len(), 1, "the plugin still loads");
        assert!(found[0].mcp_servers.is_empty(), "and reads neither path");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn another_hosts_variables_are_the_ones_cleared() {
        // A pure decision, so it needs no process-wide environment mutation to test — that is
        // global to every test running beside it.
        let seen = |names: [&str; 4]| {
            inherited_host_vars(names.iter().map(std::ffi::OsString::from))
                .iter()
                .map(|k| k.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            seen(["CLAUDE_PLUGIN_DATA", "PATH", "CLAUDE_PROJECT_DIR", "HOME"]),
            ["CLAUDE_PLUGIN_DATA", "CLAUDE_PROJECT_DIR"]
        );
    }

    #[tokio::test]
    async fn only_a_hook_that_succeeded_is_believed() {
        // The format reads stdout only on exit 0. A hook that printed and then failed is not
        // offering context; treating what it managed to print as context is how a broken hook
        // quietly steers a node.
        let failed = tool_plugin("exit-nonzero", ".*", "SessionStart", "");
        std::fs::write(
            failed.join("hooks/hooks.json"),
            r#"{"hooks": {"SessionStart": [{"hooks": [{"type": "command",
                "command": "echo half an answer; exit 1"}]}]}}"#,
        )
        .unwrap();

        let plugins = discover(std::slice::from_ref(&failed));
        let contexts = session_start(&plugins, Path::new("."), &limits()).await;
        assert!(
            contexts.is_empty(),
            "what a failing hook printed is not context: {contexts:?}"
        );
        let _ = std::fs::remove_dir_all(&failed);
    }

    #[tokio::test]
    async fn the_exec_form_does_not_go_through_a_shell() {
        // With `args`, the command is an executable spawned directly. A shell would re-split
        // arguments the plugin already separated — here, on the spaces inside one of them.
        let root = tool_plugin("exec-form", ".*", "SessionStart", "");
        std::fs::write(
            root.join("hooks/hooks.json"),
            r#"{"hooks": {"SessionStart": [{"hooks": [{"type": "command",
                "command": "echo", "args": ["one argument; not two"]}]}]}}"#,
        )
        .unwrap();

        let plugins = discover(std::slice::from_ref(&root));
        let contexts = session_start(&plugins, Path::new("."), &limits()).await;
        assert_eq!(
            contexts.get("exec-form").map(String::as_str),
            Some("one argument; not two")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn a_session_start_matcher_is_read_against_the_source() {
        // It matches the session source, not a tool name. A run always starts; it is never
        // resumed or compacted, so a hook scoped to those describes a session we never have.
        let ours = tool_plugin("src-startup", "startup|clear", "SessionStart", "");
        std::fs::write(ours.join("answer"), "for a fresh session").unwrap();
        let theirs = tool_plugin("src-resume", "resume", "SessionStart", "");
        std::fs::write(theirs.join("answer"), "should not run").unwrap();

        let plugins = discover(&[ours.clone(), theirs.clone()]);
        let contexts = session_start(&plugins, Path::new("."), &limits()).await;
        assert_eq!(
            contexts.get("src-startup").map(String::as_str),
            Some("for a fresh session")
        );
        assert!(!contexts.contains_key("src-resume"));

        let _ = std::fs::remove_dir_all(&ours);
        let _ = std::fs::remove_dir_all(&theirs);
    }

    #[test]
    fn a_matcher_selects_tools_and_a_broken_one_selects_none() {
        let root = tool_plugin("matching", "^(Grep|Read)$", "PreToolUse", "");
        let hook = &discover(std::slice::from_ref(&root))[0].hooks[0];
        assert!(hook.matches("Grep") && hook.matches("Read"));
        // Declaring no timeout of its own, it takes the configured default.
        assert_eq!(
            hook.timeout(&limits()),
            Duration::from_secs(limits().timeout_secs)
        );
        assert!(!hook.matches("semantic_search"));
        let _ = std::fs::remove_dir_all(&root);

        // A pattern that will not compile costs its own hook and no other.
        let broken = tool_plugin("bad-matcher", "^(unclosed", "PreToolUse", "");
        let hook = &discover(std::slice::from_ref(&broken))[0].hooks[0];
        assert!(!hook.matches("unclosed"));
        let _ = std::fs::remove_dir_all(&broken);

        // A group with no matcher fires for every tool.
        let all = plugin_dir(
            "no-matcher",
            r#"{"name": "no-matcher"}"#,
            Some(
                r#"{"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "true"}]}]}}"#,
            ),
        );
        assert!(discover(std::slice::from_ref(&all))[0].hooks[0].matches("anything"));
        let _ = std::fs::remove_dir_all(&all);
    }

    #[tokio::test]
    async fn a_tool_hook_contributes_only_its_envelope_s_additional_context() {
        let root = tool_plugin("envelope", "^semantic_search$", "PreToolUse", ENVELOPE);
        let plugins = discover(std::slice::from_ref(&root));
        let spent = AtomicU64::new(0);
        let event = |tool| ToolEvent {
            event: "PreToolUse",
            tool,
            input: r#"{"query":"x"}"#,
            response: None,
        };

        assert_eq!(
            tool_hooks(
                &plugins,
                event("semantic_search"),
                Path::new("."),
                &limits(),
                &spent
            )
            .await,
            Some("mind the clones".to_string())
        );
        // The matcher decides; an unmatched tool runs nothing at all.
        assert_eq!(
            tool_hooks(
                &plugins,
                event("impact_surface"),
                Path::new("."),
                &limits(),
                &spent
            )
            .await,
            None
        );
        // So does the event: a PreToolUse hook has nothing to say after the call.
        assert_eq!(
            tool_hooks(
                &plugins,
                ToolEvent {
                    event: "PostToolUse",
                    ..event("semantic_search")
                },
                Path::new("."),
                &limits(),
                &spent
            )
            .await,
            None
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn output_that_is_not_the_envelope_reaches_the_model_as_nothing() {
        // A hook written against a host that reads bare stdout, and one that gates rather than
        // informs: neither is this contract, and neither may paste itself into the model's context.
        let plain = tool_plugin("plain", ".*", "PreToolUse", "just some text");
        let deny = tool_plugin(
            "deny",
            ".*",
            "PreToolUse",
            r#"{"permissionDecision": "deny"}"#,
        );
        let quiet = tool_plugin("quiet", ".*", "PreToolUse", "");

        let spent = AtomicU64::new(0);
        for root in [&plain, &deny, &quiet] {
            let plugins = discover(std::slice::from_ref(root));
            let got = tool_hooks(
                &plugins,
                ToolEvent {
                    event: "PreToolUse",
                    tool: "Read",
                    input: "{}",
                    response: None,
                },
                Path::new("."),
                &limits(),
                &spent,
            )
            .await;
            assert_eq!(got, None, "{} contributed something", root.display());
            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[tokio::test]
    async fn a_hook_that_never_reads_its_stdin_still_answers_a_large_payload() {
        // A `PostToolUse` payload carries what the tool answered, which can dwarf the pipe buffer.
        // The fixture's `cat <file>` never reads stdin, so this pins the case where the hook
        // answers and exits with most of its input still unsent.
        let root = tool_plugin("big-payload", ".*", "PostToolUse", ENVELOPE);
        let plugins = discover(std::slice::from_ref(&root));
        let huge = "x".repeat(4 * 1024 * 1024);
        let spent = AtomicU64::new(0);

        let started = std::time::Instant::now();
        let got = tool_hooks(
            &plugins,
            ToolEvent {
                event: "PostToolUse",
                tool: "Read",
                input: "{}",
                response: Some(&huge),
            },
            Path::new("."),
            &limits(),
            &spent,
        )
        .await;

        assert_eq!(got.as_deref(), Some("mind the clones"));
        assert!(
            started.elapsed() < Duration::from_secs(limits().timeout_secs),
            "the hook answered rather than waiting out its timeout on a blocked write"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn a_spent_run_stops_running_tool_hooks() {
        // A per-hook timeout bounds one call; this is what bounds a hundred of them.
        let root = tool_plugin("budget", ".*", "PreToolUse", ENVELOPE);
        let plugins = discover(std::slice::from_ref(&root));
        let event = || ToolEvent {
            event: "PreToolUse",
            tool: "Read",
            input: "{}",
            response: None,
        };

        let spent = AtomicU64::new(0);
        assert!(
            tool_hooks(&plugins, event(), Path::new("."), &limits(), &spent)
                .await
                .is_some()
        );
        assert!(
            spent.load(Ordering::Relaxed) > 0,
            "a hook that ran is charged for the time it took"
        );

        let budgeted = HookLimits {
            tool_time_budget_secs: 60,
            ..limits()
        };
        let exhausted =
            AtomicU64::new(Duration::from_secs(budgeted.tool_time_budget_secs).as_millis() as u64);
        assert_eq!(
            tool_hooks(&plugins, event(), Path::new("."), &budgeted, &exhausted).await,
            None,
            "past its budget a run runs no more tool hooks"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A plugin directory, built for one test.
    fn plugin_dir(case: &str, manifest: &str, hooks: Option<&str>) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("ratatoskr-plugin-{}-{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".claude-plugin")).unwrap();
        std::fs::write(root.join(".claude-plugin/plugin.json"), manifest).unwrap();
        if let Some(hooks) = hooks {
            std::fs::create_dir_all(root.join("hooks")).unwrap();
            std::fs::write(root.join("hooks/hooks.json"), hooks).unwrap();
        }
        root
    }

    const HOOKS: &str = r#"{
        "hooks": {
            "SessionStart": [
                { "hooks": [{ "type": "command", "command": "echo hello", "timeout": 5 }] }
            ],
            "PreToolUse": [
                { "matcher": "^(Grep|Read)$",
                  "hooks": [{ "type": "command", "command": "true", "timeout": 99 }] }
            ]
        }
    }"#;

    #[test]
    fn a_plugin_is_read_from_its_manifest_and_hooks() {
        let root = plugin_dir("read", r#"{"name": "demo"}"#, Some(HOOKS));
        let found = discover(std::slice::from_ref(&root));

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "demo");

        let session: Vec<_> = found[0]
            .hooks
            .iter()
            .filter(|h| h.event == "SessionStart")
            .collect();
        assert_eq!(session.len(), 1);
        assert_eq!(session[0].command, "echo hello");
        assert!(session[0].matcher.is_none());

        let pre: Vec<_> = found[0]
            .hooks
            .iter()
            .filter(|h| h.event == "PreToolUse")
            .collect();
        assert_eq!(pre[0].matcher.as_deref(), Some("^(Grep|Read)$"));
        // A plugin cannot hold a node for as long as it likes.
        // A hook gets what it asked for, up to the configured ceiling.
        assert_eq!(pre[0].timeout, Some(99));
        assert_eq!(pre[0].timeout(&limits()), Duration::from_secs(99));
        let strict = HookLimits {
            max_timeout_secs: 10,
            ..limits()
        };
        assert_eq!(pre[0].timeout(&strict), Duration::from_secs(10));
        assert_eq!(session[0].timeout(&limits()), Duration::from_secs(5));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_plugin_is_loaded_once_however_many_copies_are_on_disk() {
        // A path naming a plugin in a coding CLI's cache holds every version ever installed.
        // Loading them all ran one plugin's SessionStart hooks once per version.
        let cache = std::env::temp_dir().join(format!("ratatoskr-versions-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache);
        for version in ["0.20.0", "0.21.0", "0.22.0"] {
            let root = cache.join("demo").join(version);
            std::fs::create_dir_all(root.join(".claude-plugin")).unwrap();
            std::fs::write(
                root.join(".claude-plugin/plugin.json"),
                format!(r#"{{"name": "demo", "version": "{version}"}}"#),
            )
            .unwrap();
        }

        // With no registry to consult, one copy is kept and the run is told about the others.
        let found = discover(&[cache.join("demo")]);
        assert_eq!(
            found.len(),
            1,
            "{:?}",
            found.iter().map(|p| &p.root).collect::<Vec<_>>()
        );

        // With one, the copy the host says is installed is the copy that is used.
        let home = std::env::temp_dir().join(format!("ratatoskr-vhome-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(home.join(".claude/plugins")).unwrap();
        std::fs::write(
            home.join(".claude/plugins/installed_plugins.json"),
            format!(
                r#"{{"plugins": {{"demo@somewhere": [{{"installPath": "{}"}}]}}}}"#,
                cache.join("demo/0.21.0").display()
            ),
        )
        .unwrap();

        let all: Vec<Plugin> = ["0.20.0", "0.21.0", "0.22.0"]
            .iter()
            .filter_map(|v| load(&cache.join("demo").join(v)))
            .collect();
        let kept = one_per_name(all, &home);
        assert_eq!(kept.len(), 1);
        assert!(kept[0].root.ends_with("0.21.0"), "{:?}", kept[0].root);

        let _ = std::fs::remove_dir_all(&cache);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_directory_of_plugins_is_walked_too() {
        // `.ratatoskr/plugins/` holds several; a path to an installed one points at just it.
        let parent = std::env::temp_dir().join(format!("ratatoskr-plugins-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&parent);
        for name in ["alpha", "beta"] {
            let root = parent.join(name);
            std::fs::create_dir_all(root.join(".claude-plugin")).unwrap();
            std::fs::write(
                root.join(".claude-plugin/plugin.json"),
                format!(r#"{{"name": "{name}"}}"#),
            )
            .unwrap();
        }

        let found = discover(std::slice::from_ref(&parent));
        assert_eq!(
            found.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            ["alpha", "beta"]
        );
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn nothing_that_is_not_a_plugin_is_an_error() {
        // A missing directory, a directory of unrelated files, and a malformed manifest are all
        // ordinary — none of them may fail a run.
        let stray = std::env::temp_dir().join(format!("ratatoskr-stray-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&stray);
        std::fs::create_dir_all(stray.join(".claude-plugin")).unwrap();
        std::fs::write(stray.join(".claude-plugin/plugin.json"), "{ not json").unwrap();

        assert!(discover(&[PathBuf::from("/nonexistent/plugins")]).is_empty());
        assert!(discover(std::slice::from_ref(&stray)).is_empty());

        // A plugin with no hooks file at all is still a plugin.
        let bare = plugin_dir("bare", r#"{"name": "bare"}"#, None);
        let found = discover(std::slice::from_ref(&bare));
        assert_eq!(found.len(), 1);
        assert!(found[0].hooks.is_empty());

        let _ = std::fs::remove_dir_all(&stray);
        let _ = std::fs::remove_dir_all(&bare);
    }

    #[tokio::test]
    async fn session_start_collects_stdout_and_ignores_silence() {
        let talkative = plugin_dir(
            "talk",
            r#"{"name": "talkative"}"#,
            Some(
                r#"{"hooks": {"SessionStart": [{"hooks": [
                    {"type": "command", "command": "echo ' repo digest '"}
                ]}]}}"#,
            ),
        );
        let quiet = plugin_dir(
            "quiet",
            r#"{"name": "quiet"}"#,
            Some(
                r#"{"hooks": {"SessionStart": [{"hooks": [
                    {"type": "command", "command": "exit 0"}
                ]}]}}"#,
            ),
        );

        let plugins = discover(&[talkative.clone(), quiet.clone()]);
        let context = compose(
            &session_start(&plugins, Path::new("."), &limits()).await,
            &plugins.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            &limits(),
        );
        assert_eq!(context.as_deref(), Some("repo digest"));

        let _ = std::fs::remove_dir_all(&talkative);
        let _ = std::fs::remove_dir_all(&quiet);
    }

    #[tokio::test]
    async fn a_failing_or_hanging_hook_costs_only_its_own_context() {
        let broken = plugin_dir(
            "broken",
            r#"{"name": "broken"}"#,
            Some(
                r#"{"hooks": {"SessionStart": [{"hooks": [
                    {"type": "command", "command": "definitely-not-a-real-command-xyz"},
                    {"type": "command", "command": "sleep 30", "timeout": 1},
                    {"type": "command", "command": "echo survived"}
                ]}]}}"#,
            ),
        );

        let plugins = discover(std::slice::from_ref(&broken));
        let started = std::time::Instant::now();
        let context = compose(
            &session_start(&plugins, Path::new("."), &limits()).await,
            &plugins.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            &limits(),
        );

        assert_eq!(context.as_deref(), Some("survived"));
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the timeout is enforced rather than waited out"
        );
        let _ = std::fs::remove_dir_all(&broken);
    }

    #[tokio::test]
    async fn a_hook_that_floods_stdout_is_cut_off_rather_than_buffered() {
        // A plugin must not be able to exhaust the run's memory. `yes` writes without end, so if
        // the read were unbounded this would never return.
        let flood = plugin_dir(
            "flood",
            r#"{"name": "flood"}"#,
            Some(
                r#"{"hooks": {"SessionStart": [{"hooks": [
                    {"type": "command", "command": "yes ratatoskr", "timeout": 10}
                ]}]}}"#,
            ),
        );
        let plugins = discover(std::slice::from_ref(&flood));
        let started = std::time::Instant::now();
        let context = compose(
            &session_start(&plugins, Path::new("."), &limits()).await,
            &plugins.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            &limits(),
        );

        assert!(
            started.elapsed() < Duration::from_secs(9),
            "the read is bounded, so it returns without waiting out the timeout"
        );
        // Way over the context budget, so none of it is carried — but the run survived it.
        assert!(context.is_none(), "an over-budget hook contributes nothing");
        let _ = std::fs::remove_dir_all(&flood);
    }

    #[tokio::test]
    async fn the_plugin_root_is_available_to_a_command() {
        // Plugins address their own files through `${CLAUDE_PLUGIN_ROOT}`.
        let root = plugin_dir(
            "root",
            r#"{"name": "rooted"}"#,
            Some(
                r#"{"hooks": {"SessionStart": [{"hooks": [
                    {"type": "command", "command": "echo \"${CLAUDE_PLUGIN_ROOT}\""}
                ]}]}}"#,
            ),
        );
        let plugins = discover(std::slice::from_ref(&root));
        let names: Vec<String> = plugins.iter().map(|p| p.name.clone()).collect();
        let context = compose(
            &session_start(&plugins, Path::new("."), &limits()).await,
            &names,
            &limits(),
        )
        .expect("the hook echoed its root");
        assert_eq!(context, root.display().to_string());
        let _ = std::fs::remove_dir_all(&root);
    }
}
