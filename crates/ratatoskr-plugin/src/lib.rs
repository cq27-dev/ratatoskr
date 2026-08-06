//! Plugins in the format coding CLIs already use, so a plugin written once works here too.
//!
//! A plugin is a directory holding a manifest (`.claude-plugin/plugin.json`) and, optionally,
//! `hooks/hooks.json` mapping an event to a command to run. Adopting that layout rather than
//! inventing one keeps the declarative surface tiny: the schema does matching and nothing else,
//! and every plugin's actual intelligence lives in the command it names.
//!
//! Nothing here may fail a run. A plugin that is missing, malformed, slow, or broken is logged and
//! skipped — a node that would have got some extra context simply doesn't.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

/// Longest a single hook may take before it is abandoned. Plugins declare their own timeout; this
/// caps it, because the caller is a node waiting to start.
const MAX_TIMEOUT: Duration = Duration::from_secs(10);

/// Timeout for a hook that declares none.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Most output read from a single hook before it is cut off and the process killed.
///
/// The read itself has to be bounded, not just the text kept: a hook that writes without end
/// would otherwise be buffered in full before anything got a chance to reject it, and a plugin
/// must not be able to exhaust the run's memory.
const MAX_HOOK_OUTPUT: u64 = 256 * 1024;

/// How much hook output a node will carry.
///
/// This text is prepended to the node's preamble, so it is paid for on *every* model call that
/// node makes — the budget is what keeps an orientation digest from becoming a tax.
pub const CONTEXT_BUDGET: usize = 4000;

/// A loaded plugin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plugin {
    pub name: String,
    /// The plugin's directory, substituted for `${CLAUDE_PLUGIN_ROOT}` in its commands.
    pub root: PathBuf,
    pub hooks: Vec<Hook>,
    /// MCP servers this plugin brings, in manifest order.
    pub mcp_servers: Vec<McpServerSpec>,
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

/// One hook a plugin registers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hook {
    pub event: String,
    /// Regex over the tool name, as written. Unused for `SessionStart`.
    pub matcher: Option<String>,
    pub command: String,
    pub timeout: Duration,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    name: Option<String>,
    #[serde(default, rename = "mcpServers")]
    mcp_servers: BTreeMap<String, ServerEntry>,
}

/// The `.mcp.json` a plugin may carry instead of (or as well as) a manifest `mcpServers` block.
#[derive(Debug, Deserialize)]
struct McpFile {
    #[serde(default, rename = "mcpServers")]
    mcp_servers: BTreeMap<String, ServerEntry>,
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

#[derive(Debug, Deserialize)]
struct HooksFile {
    #[serde(default)]
    hooks: BTreeMap<String, Vec<HookGroup>>,
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
    found
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
        hooks: read_hooks(root),
        mcp_servers: read_mcp_servers(root, manifest.mcp_servers),
        root: root.to_path_buf(),
    })
}

/// The MCP servers a plugin declares, from its manifest and from a sibling `.mcp.json`.
///
/// Both spellings are in use, and a plugin commonly carries the same block twice so it works in
/// hosts that read one or the other. The manifest wins on a shared name so the file cannot
/// silently redirect a server the manifest already described.
fn read_mcp_servers(root: &Path, manifest: BTreeMap<String, ServerEntry>) -> Vec<McpServerSpec> {
    let mut declared = manifest;
    if let Ok(raw) = std::fs::read_to_string(root.join(".mcp.json")) {
        match serde_json::from_str::<McpFile>(&raw) {
            Ok(file) => {
                for (name, entry) in file.mcp_servers {
                    declared.entry(name).or_insert(entry);
                }
            }
            Err(e) => tracing::warn!("ignoring .mcp.json for plugin at {}: {e}", root.display()),
        }
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

/// Hooks are conventional, not declared in the manifest, and a plugin without them is normal.
fn read_hooks(root: &Path) -> Vec<Hook> {
    let Ok(raw) = std::fs::read_to_string(root.join("hooks/hooks.json")) else {
        return Vec::new();
    };
    let parsed: HooksFile = match serde_json::from_str(&raw) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!("ignoring hooks for plugin at {}: {e}", root.display());
            return Vec::new();
        }
    };

    parsed
        .hooks
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
                        matcher: matcher.clone(),
                        command: entry.command,
                        timeout: entry
                            .timeout
                            .map_or(DEFAULT_TIMEOUT, |s| Duration::from_secs(s).min(MAX_TIMEOUT)),
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
pub async fn session_start(plugins: &[Plugin], cwd: &Path) -> BTreeMap<String, String> {
    let mut contexts = BTreeMap::new();

    for plugin in plugins {
        let mut parts: Vec<String> = Vec::new();
        for hook in plugin.hooks.iter().filter(|h| h.event == "SessionStart") {
            let payload = serde_json::json!({
                "session_id": "",
                "cwd": cwd.display().to_string(),
                "hook_event_name": "SessionStart",
                // Plugins commonly gate on this; a run beginning is a startup.
                "source": "startup",
            });
            let Some(text) = run_hook(plugin, hook, &payload, cwd).await else {
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
        if text.len() > CONTEXT_BUDGET {
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

/// Compose one node's session context from the plugins it is bound to, in binding order.
///
/// Capped at [`CONTEXT_BUDGET`] and truncated between plugins rather than mid-sentence: this text
/// is prepended to the node's preamble, so it is paid for on every model call that node makes.
pub fn compose(contexts: &BTreeMap<String, String>, names: &[String]) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    let mut used = 0usize;

    for name in names {
        let Some(text) = contexts.get(name) else {
            continue;
        };
        // Whole plugins in or out: half a digest is worse than none of one.
        if used + text.len() > CONTEXT_BUDGET {
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
    payload: &serde_json::Value,
    cwd: &Path,
) -> Option<String> {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    // Commands are written for a shell — they carry their own quoting.
    let mut child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&hook.command)
        // Plugins address their own files through this; `sh` expands it from the environment, so
        // the path is never spliced into the command text.
        .env("CLAUDE_PLUGIN_ROOT", &plugin.root)
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

    let collect = async {
        if let Some(mut stdin) = stdin.take() {
            // A hook that never reads stdin would block this write; it is inside the timeout, and
            // the close is what lets a hook that reads to EOF proceed.
            let _ = stdin.write_all(payload.as_bytes()).await;
            drop(stdin);
        }
        let mut buf = Vec::new();
        if let Some(stdout) = stdout.as_mut() {
            let _ = stdout.take(MAX_HOOK_OUTPUT).read_to_end(&mut buf).await;
        }
        buf
    };

    match tokio::time::timeout(hook.timeout, collect).await {
        Ok(buf) => {
            // Reap it, but never wait on it: the output is already in hand.
            let _ = child.start_kill();
            Some(String::from_utf8_lossy(&buf).into_owned())
        }
        Err(_) => {
            tracing::warn!(
                "plugin {} hook timed out after {:?}",
                plugin.name,
                hook.timeout
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(pre[0].timeout, MAX_TIMEOUT);

        let _ = std::fs::remove_dir_all(&root);
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
            &session_start(&plugins, Path::new(".")).await,
            &plugins.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
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
            &session_start(&plugins, Path::new(".")).await,
            &plugins.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
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
            &session_start(&plugins, Path::new(".")).await,
            &plugins.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
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
        let context = compose(&session_start(&plugins, Path::new(".")).await, &names)
            .expect("the hook echoed its root");
        assert_eq!(context, root.display().to_string());
        let _ = std::fs::remove_dir_all(&root);
    }
}
