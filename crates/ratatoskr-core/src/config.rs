//! `ratatoskr.toml` configuration.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Top-level config loaded from `ratatoskr.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RatatoskrConfig {
    pub rag_rat: RagRatConfig,
    pub store: StoreConfig,
    pub worktree: WorktreeConfig,
    /// Per-node model routing, keyed by node name (`"scout"`, `"analyst"`, ...).
    #[serde(default)]
    pub models: HashMap<String, ModelRoute>,
    #[serde(default)]
    pub implementer: ImplementerConfig,
    #[serde(default)]
    pub sandbox: SandboxConfig,
    #[serde(default)]
    pub plugins: PluginConfig,
    #[serde(default)]
    pub publish: PublishConfig,
    #[serde(default)]
    pub endpoint: EndpointConfig,
}

/// How to address the model endpoint, beyond its URL and key.
///
/// Exists because a local endpoint in front of a provider is not always a proxy. One that
/// reconstructs the request — replaying it as a prompt into its own agent session — decides for
/// itself what is cached and what is discarded, and it decides that from what the client looks
/// like. A client it does not recognise gets whatever default the author picked for someone else.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointConfig {
    /// Headers sent with every model request.
    ///
    /// What identifies this client to the thing in front of the provider. Empty by default: this
    /// is the address of a specific deployment, not a property of ratatoskr, and a header invented
    /// here would be a guess about somebody else's software.
    #[serde(default)]
    pub headers: std::collections::HashMap<String, String>,
    /// Header carrying a per-conversation id, when the endpoint keys a session off one.
    ///
    /// Each node attempt gets a fresh id, held for every turn of that attempt. An endpoint that
    /// tracks sessions can then continue one conversation rather than rebuilding it per turn —
    /// which is the difference between reading the history back and paying to write it again.
    #[serde(default)]
    pub session_header: Option<String>,
}

/// Where to look for agent plugins. `.ratatoskr/plugins/` is always searched; `paths` adds
/// plugins installed elsewhere, and may name either a plugin or a directory of them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginConfig {
    #[serde(default)]
    pub paths: Vec<PathBuf>,
    #[serde(default)]
    pub hooks: HookLimits,
}

/// What a plugin's hooks may spend of a run.
///
/// The defaults are the Claude Code plugin format's own, so a plugin written against that host
/// behaves here the way its author tested it. They are generous — a hook may take ten minutes and
/// answer with ten thousand characters — because that host has a person watching it. Ratatoskr
/// runs unattended, so every one of them is overridable in `ratatoskr.toml`, and a repo that
/// treats plugins as a latency budget rather than a convenience should lower them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HookLimits {
    /// Seconds a hook gets when it declares no `timeout` of its own.
    #[serde(default = "default_hook_timeout")]
    pub timeout_secs: u64,
    /// Ceiling on a hook-declared `timeout`. The format sets none; this defaults to the same value
    /// as `timeout_secs`, so a hook asking for less gets less and one asking for more is capped.
    #[serde(default = "default_hook_timeout")]
    pub max_timeout_secs: u64,
    /// Characters one event's hooks may contribute, across all of them.
    #[serde(default = "default_output_budget")]
    pub output_budget: usize,
    /// Characters of plugin context a node will carry into its preamble, across every plugin it
    /// binds. Paid on every model call that node makes, which is why it has its own limit.
    #[serde(default = "default_output_budget")]
    pub context_budget: usize,
    /// Total seconds a run will spend in hooks that run around tool calls, after which it stops
    /// running them. `0` means no limit.
    ///
    /// Not part of the plugin format — it has no equivalent because a person can interrupt an
    /// interactive session. A hook on a tool call fires on every one a node makes, so this is the
    /// only bound on what plugins cost a run as a whole.
    #[serde(default)]
    pub tool_time_budget_secs: u64,
}

/// The plugin format's default hook timeout.
fn default_hook_timeout() -> u64 {
    600
}

/// The plugin format's cap on a hook's output.
fn default_output_budget() -> usize {
    10_000
}

impl Default for HookLimits {
    fn default() -> Self {
        HookLimits {
            timeout_secs: default_hook_timeout(),
            max_timeout_secs: default_hook_timeout(),
            output_budget: default_output_budget(),
            context_budget: default_output_budget(),
            tool_time_budget_secs: 0,
        }
    }
}

impl PluginConfig {
    /// Every directory to search, convention first, resolved against the project root.
    ///
    /// Relative paths are joined to `root` rather than left to the process's working directory:
    /// a config's paths belong to the project it configures, and one process can serve several.
    pub fn search_paths(&self, root: &Path) -> Vec<PathBuf> {
        let mut dirs = vec![root.join(".ratatoskr/plugins")];
        dirs.extend(self.paths.iter().map(|p| {
            let p = expand_home(p);
            if p.is_absolute() { p } else { root.join(p) }
        }));
        dirs
    }
}

/// Expand a leading `~` against `HOME`.
///
/// A plugin installed by a coding CLI lives under the home directory, so `~/.claude/plugins/...` is
/// the natural way to write it — and without this it is not absolute, so it is joined onto the
/// repository root and searched for at `<repo>/~/.claude/...`. That directory never exists, so the
/// plugin is silently absent: no error, no warning, just a node that never gets its tools.
///
/// Only a leading `~/` (or a bare `~`). A `~` elsewhere in a path is a literal character, and some
/// editors really do create files with one.
fn expand_home(path: &Path) -> PathBuf {
    let Some(rest) = path.to_str().and_then(|p| p.strip_prefix('~')) else {
        return path.to_path_buf();
    };
    if !rest.is_empty() && !rest.starts_with('/') {
        // `~user` is somebody else's home, which resolving would guess at.
        return path.to_path_buf();
    }
    let Some(home) = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|h| !h.is_empty())
    else {
        return path.to_path_buf();
    };
    PathBuf::from(home).join(rest.trim_start_matches('/'))
}

/// Phase 3 implementer settings.
///
/// Unknown keys are refused. Every field here changes whether the run does something or skips it —
/// how many times it retries, what sends a change back, whether the fork runs at all — so a
/// misspelled key that silently kept the default is the worst kind of typo: the run looks
/// configured and is not, and nothing about the output says otherwise.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImplementerConfig {
    /// How many times converge may re-run the implementer before giving up.
    pub max_iterations: u32,
    /// The least severe review finding that sends the change back to be fixed.
    ///
    /// `"P1"` blocks only on must-fix defects, `"P2"` (the default) also on should-fix ones, and
    /// `"P3"` on nits too. The default stops short of nits deliberately: the verifier's back edge
    /// shares `max_iterations` with the test-fixing loop, so a loop that re-drives on style can
    /// spend the whole budget there and leave none for a real failure found on the last pass.
    ///
    /// Findings below the threshold are still recorded on the checkpoint — not blocking is not the
    /// same as not worth knowing.
    #[serde(default = "default_verify_threshold")]
    pub verify_threshold: String,
    /// Run the fork even when the analyst says the task calls for no code change.
    ///
    /// The override for disagreeing with that judgement. It is a config key rather than a silent
    /// heuristic because the analyst's call is recorded in its checkpoint and named by the run's
    /// status: a human who thinks it got the task wrong should be able to say so, and have that be
    /// as visible as the decision it overrules.
    #[serde(default)]
    pub always_fork: bool,
}

impl Default for ImplementerConfig {
    fn default() -> Self {
        ImplementerConfig {
            max_iterations: 3,
            verify_threshold: default_verify_threshold(),
            always_fork: false,
        }
    }
}

fn default_verify_threshold() -> String {
    "P2".to_string()
}

/// Where a run's output goes when it is finished.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishConfig {
    /// Whether the publisher may write to the tracker.
    ///
    /// Off by default, and deliberately a switch rather than an inference from having a route. This
    /// is the only node that acts outside this machine: a run that opens a pull request nobody
    /// expected is worse than one that publishes nothing, so turning it on should be a thing
    /// somebody did.
    #[serde(default)]
    pub enabled: bool,
    /// The label every pull request a run opens carries, so a reviewer can tell what wrote the
    /// change in front of them.
    ///
    /// Set it to another word to use that instead, or to `""` to add none. A repository that
    /// already labels automation its own way should not be given a second scheme, and one that
    /// does not want the noise should not have to take it.
    #[serde(default = "default_publish_label")]
    pub label: String,
}

fn default_publish_label() -> String {
    "ratatoskr".to_string()
}

/// Written out rather than derived, so a config built in code and one parsed from a file with the
/// key absent agree about the label. `#[serde(default = …)]` covers only the parsing path, and a
/// derived `Default` would give the empty string — which here means "add no label", a different
/// answer to the same question depending on how the config was made.
impl Default for PublishConfig {
    fn default() -> Self {
        PublishConfig {
            enabled: false,
            label: default_publish_label(),
        }
    }
}

/// Phase 3 sandbox settings — where red-team and implementer run the acceptance check.
///
/// Unknown keys are refused, for the same reason `ImplementerConfig` refuses them and with a
/// sharper edge here: `pin_acceptance` is what stops a model choosing what proves the code works,
/// and a misspelling of it fails open — the repo believes acceptance is pinned and it is not.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SandboxConfig {
    /// `"microsandbox"` (MicroVM, needs KVM) or `"landlock"` (bwrap+Landlock fallback).
    pub backend: String,
    /// OCI image the sandbox boots (microsandbox backend).
    pub image: String,
    /// The default acceptance check: one command, for the repo that has one and nothing to add.
    /// Superseded per-task by the analyst's `acceptance` unless `pin_acceptance` is set.
    pub test_command: Vec<String>,
    /// Ignore whatever acceptance the analyst proposes and always run `test_command`.
    ///
    /// The escape hatch for a repo that does not want a model deciding what proves its code works.
    /// Off by default because what counts as done varies by change — a refactor is accepted by the
    /// existing suite, a new endpoint is not accepted until something exercises the endpoint.
    #[serde(default)]
    pub pin_acceptance: bool,
}

/// One step of an acceptance check: a name to attribute a failure to, and the command to run.
///
/// A list of these replaces a single test command because acceptance is frequently a pipeline
/// rather than an invocation — "build the wasm, then drive it in a browser" is two steps whose
/// first produces an artifact and whose second reports nothing shaped like a test runner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AcceptanceStep {
    /// Short label, used to attribute a failure when the step's output has no finer structure.
    pub name: String,
    /// argv, run inside the sandbox with the worktree mounted. Never run on the host.
    pub command: Vec<String>,
}

impl SandboxConfig {
    /// The acceptance to run, given what the analyst proposed.
    ///
    /// `pin_acceptance` wins, then a non-empty proposal, then the configured `test_command`. The
    /// order matters: a repo that pinned its acceptance must not have it replaced by a plan, and a
    /// plan that proposed nothing must still be checked against something.
    pub fn acceptance(&self, proposed: &[AcceptanceStep]) -> Vec<AcceptanceStep> {
        if !self.pin_acceptance && !proposed.is_empty() {
            return proposed.to_vec();
        }
        vec![AcceptanceStep {
            name: "tests".to_string(),
            command: self.test_command.clone(),
        }]
    }
}

impl Default for SandboxConfig {
    fn default() -> Self {
        SandboxConfig {
            // landlock builds with no network build script; microsandbox is opt-in behind
            // ratatoskr-exec's `microsandbox` feature (see its Cargo.toml).
            backend: "landlock".to_string(),
            image: "docker.io/library/rust:1-slim".to_string(),
            test_command: vec!["cargo".to_string(), "test".to_string()],
            pin_acceptance: false,
        }
    }
}

/// How to launch rag-rat's MCP server over stdio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagRatConfig {
    pub command: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<PathBuf>,
}

/// Ratatoskr's own checkpoint database — deliberately a separate file from rag-rat's index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreConfig {
    pub path: PathBuf,
}

/// Root for the per-run git worktrees `run`'s implementer fork creates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorktreeConfig {
    pub root: PathBuf,
}

/// A `provider`/`model` pair for one node.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRoute {
    pub provider: String,
    pub model: String,
    /// Cap on the tokens one model call may produce. `None` uses [`DEFAULT_MAX_TOKENS`].
    ///
    /// Set explicitly rather than left to the provider client, which fills it from a table of
    /// model-name prefixes it was compiled with. Any model released after that table falls through
    /// it, the field goes unset, and the request is rejected — a whole run lost at the first call
    /// of whichever node happened to be routed to the new model.
    ///
    /// Raise it for a route that reasons at length: on Anthropic, thinking tokens count against
    /// this. Lower it for a model whose own ceiling is below the default.
    #[serde(default)]
    pub max_tokens: Option<u64>,
    /// Sampling temperature. `None` leaves the provider's default, which on Anthropic is 1.0.
    ///
    /// Worth setting to 0 for a node whose job is transcription or extraction rather than
    /// judgement — the characterizer reads test output and names the checks in it, and a node
    /// sampling creatively over that invents detail the output does not contain.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Provider-specific request fields, merged into the call verbatim.
    ///
    /// The escape hatch for what a provider offers and this config has no word for — Anthropic's
    /// extended thinking is the reason it exists:
    ///
    /// ```toml
    /// [models.analyst.params.thinking]
    /// type = "enabled"
    /// budget_tokens = 4000
    /// ```
    ///
    /// Passed through unvalidated, so a misspelling here is a provider error at the first call
    /// rather than a config error at load. Thinking tokens count against `max_tokens`; raise it
    /// alongside the budget or the call is rejected for a cap it cannot meet.
    #[serde(default)]
    pub params: Option<toml::Value>,
    /// Whether this node's conversation continues across attempts, or starts over each time.
    ///
    /// A node that is re-driven — the implementer on a converge iteration, the analyst on a
    /// revision — is given a diagnostic rather than the original task, and its message history
    /// starts empty either way. What `Reuse` keeps is the *endpoint's* session: a gateway that
    /// tracks one can carry what the previous attempt established, so the second attempt does not
    /// re-read the tree it just edited.
    ///
    /// Default is `Fresh`, because reuse is only sound where a later attempt genuinely continues
    /// the earlier one. A node re-run on an unrelated task under a reused session would inherit
    /// context that has nothing to do with it.
    #[serde(default)]
    pub session: SessionScope,
}

/// Whether a node's endpoint session continues across attempts within a run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionScope {
    /// A new session per attempt. The safe default.
    #[default]
    Fresh,
    /// One session for this node across the whole run, so a re-driven attempt continues where the
    /// last one stopped.
    Reuse,
}

/// The per-call output cap when a route does not set one.
///
/// Chosen to clear every node's real output — the largest structured plan this repo has produced is
/// about 1,700 tokens — while staying at or under the ceiling of every Claude model from 3.5
/// onwards, so the default never turns into the very error it exists to prevent.
pub const DEFAULT_MAX_TOKENS: u64 = 8_192;

impl ModelRoute {
    /// The cap to send with each call.
    pub fn max_tokens(&self) -> u64 {
        self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)
    }
}

/// Error parsing or validating `ratatoskr.toml`.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to parse ratatoskr.toml: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("invalid config: {0}")]
    Invalid(String),
}

impl RatatoskrConfig {
    /// Parse a config from TOML source.
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        Ok(toml::from_str(s)?)
    }

    /// Render this config as TOML — what `ratatoskr init` writes.
    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Reject configs that would otherwise fail deep in a run with a cryptic error. Structural
    /// checks only — this does not probe the environment (whether the sandbox backend's kernel
    /// features are present, whether the CLI is installed); those surface at run time.
    pub fn validate(&self) -> Result<(), ConfigError> {
        const BACKENDS: [&str; 2] = ["microsandbox", "landlock"];

        if self.rag_rat.command.is_empty() {
            return Err(ConfigError::Invalid(
                "rag_rat.command is empty — set the command that launches rag-rat's MCP server"
                    .to_string(),
            ));
        }
        if self.sandbox.test_command.is_empty() {
            return Err(ConfigError::Invalid(
                "sandbox.test_command is empty — set the repo's test command".to_string(),
            ));
        }
        if !BACKENDS.contains(&self.sandbox.backend.as_str()) {
            return Err(ConfigError::Invalid(format!(
                "sandbox.backend `{}` is not one of {BACKENDS:?}",
                self.sandbox.backend
            )));
        }
        if self.implementer.max_iterations == 0 {
            return Err(ConfigError::Invalid(
                "implementer.max_iterations must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for RatatoskrConfig {
    /// The starter config `ratatoskr init` writes. Model routes are illustrative — the real
    /// provider/model choices are a Phase 2/3 decision once there are nodes to route.
    fn default() -> Self {
        let route = |provider: &str, model: &str| ModelRoute {
            max_tokens: None,
            temperature: None,
            params: None,
            session: SessionScope::default(),
            provider: provider.to_string(),
            model: model.to_string(),
        };
        RatatoskrConfig {
            publish: PublishConfig::default(),
            endpoint: EndpointConfig::default(),
            rag_rat: RagRatConfig {
                // `--json` makes rag-rat emit JSON (not its default TOON), so nodes that parse
                // tool results directly (MemoryNode) get a stable shape.
                command: ["npx", "-y", "@rag-rat/bin", "mcp", "--json"]
                    .map(str::to_string)
                    .to_vec(),
                working_dir: None,
            },
            store: StoreConfig {
                path: PathBuf::from(".ratatoskr/state.sqlite3"),
            },
            worktree: WorktreeConfig {
                root: PathBuf::from(".ratatoskr/worktrees"),
            },
            models: HashMap::from([
                // `ask` is the only route consumed in Phase 1; the rest are illustrative,
                // forward-looking node routes (Phase 2+).
                ("ask".to_string(), route("anthropic", "claude-sonnet-4-6")),
                ("scout".to_string(), route("moonshot", "kimi-k2.5")),
                (
                    "analyst".to_string(),
                    route("anthropic", "claude-sonnet-4-6"),
                ),
                (
                    "implementer".to_string(),
                    route("anthropic", "claude-opus-4-8"),
                ),
                // Bookkeeper composes memory prose — a cheap/fast tier is fine.
                ("bookkeeper".to_string(), route("moonshot", "kimi-k2.5")),
            ]),
            implementer: ImplementerConfig::default(),
            sandbox: SandboxConfig::default(),
            plugins: PluginConfig::default(),
        }
    }
}

#[cfg(test)]
mod plugin_path_tests {
    use super::*;

    #[test]
    fn a_home_relative_plugin_path_resolves_to_the_home_directory() {
        // The failure this prevents is silent: `~/…` is not absolute, so it was joined onto the
        // repository root and searched for at `<repo>/~/.claude/…`. Nothing is there, nothing
        // errors, and the plugin simply never loads.
        let home = std::env::var("HOME").expect("HOME is set in the test environment");
        let config = PluginConfig {
            paths: vec![PathBuf::from("~/.claude/plugins/cache/ponytail/ponytail")],
            ..Default::default()
        };
        let found = config.search_paths(Path::new("/repo"));
        assert_eq!(
            found[1],
            PathBuf::from(&home).join(".claude/plugins/cache/ponytail/ponytail")
        );
        assert!(!found[1].to_string_lossy().contains('~'), "{:?}", found[1]);
    }

    #[test]
    fn only_a_leading_tilde_is_a_home_directory() {
        let home = std::env::var("HOME").unwrap();
        // A bare `~` is the home directory itself.
        assert_eq!(expand_home(Path::new("~")), PathBuf::from(&home));
        // `~user` is somebody else's home, and resolving it would be a guess.
        assert_eq!(
            expand_home(Path::new("~someone/plugins")),
            PathBuf::from("~someone/plugins")
        );
        // A tilde that is not leading is a character in a name, and some editors write those.
        assert_eq!(
            expand_home(Path::new("plugins/backup~/thing")),
            PathBuf::from("plugins/backup~/thing")
        );
        // An absolute path is untouched, and a relative one stays relative for the caller to root.
        assert_eq!(
            expand_home(Path::new("/abs/path")),
            PathBuf::from("/abs/path")
        );
        assert_eq!(
            expand_home(Path::new("rel/path")),
            PathBuf::from("rel/path")
        );
    }
}

#[cfg(test)]
mod publish_label_tests {
    use super::*;

    #[test]
    fn the_label_defaults_the_same_way_however_the_config_was_made() {
        // Both paths answer the same question, and a config built in code must not quietly mean
        // "add no label" while one parsed from a file with the key absent means "ratatoskr".
        assert_eq!(PublishConfig::default().label, "ratatoskr");
        let parsed: PublishConfig = toml::from_str("enabled = true").unwrap();
        assert_eq!(parsed.label, "ratatoskr");
    }

    #[test]
    fn a_repository_can_rename_the_label_or_ask_for_none() {
        let renamed: PublishConfig = toml::from_str(r#"label = "automation""#).unwrap();
        assert_eq!(renamed.label, "automation");
        // The opt-out: empty means no label at all, not a label named "".
        let none: PublishConfig = toml::from_str(r#"label = """#).unwrap();
        assert!(none.label.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_limits_default_to_the_plugin_formats_own_and_are_overridable() {
        // A config that says nothing about plugins gets the format's defaults, so a plugin
        // behaves here the way its author tested it.
        let bare = RatatoskrConfig::from_toml_str(
            r#"
            [rag_rat]
            command = ["rag-rat", "mcp"]
            [store]
            path = ".ratatoskr/state.sqlite3"
            [worktree]
            root = ".ratatoskr/worktrees"
            "#,
        )
        .unwrap();
        assert_eq!(bare.plugins.hooks.timeout_secs, 600);
        assert_eq!(bare.plugins.hooks.max_timeout_secs, 600);
        assert_eq!(bare.plugins.hooks.output_budget, 10_000);
        assert_eq!(bare.plugins.hooks.context_budget, 10_000);
        // Not the format's: a run is unattended, so this one is opt-in.
        assert_eq!(bare.plugins.hooks.tool_time_budget_secs, 0);

        // Each is overridable on its own; the rest stay at their defaults.
        let tight = RatatoskrConfig::from_toml_str(
            r#"
            [rag_rat]
            command = ["rag-rat", "mcp"]
            [store]
            path = ".ratatoskr/state.sqlite3"
            [worktree]
            root = ".ratatoskr/worktrees"
            [plugins.hooks]
            max_timeout_secs = 10
            tool_time_budget_secs = 60
            "#,
        )
        .unwrap();
        assert_eq!(tight.plugins.hooks.max_timeout_secs, 10);
        assert_eq!(tight.plugins.hooks.tool_time_budget_secs, 60);
        assert_eq!(tight.plugins.hooks.timeout_secs, 600);
        assert_eq!(tight.plugins.hooks.output_budget, 10_000);
    }

    #[test]
    fn a_misspelled_hook_limit_is_refused_rather_than_ignored() {
        // A limit that silently stayed at its default would be the worst kind of typo: the run
        // looks configured and isn't.
        let err = RatatoskrConfig::from_toml_str(
            r#"
            [rag_rat]
            command = ["rag-rat", "mcp"]
            [store]
            path = ".ratatoskr/state.sqlite3"
            [worktree]
            root = ".ratatoskr/worktrees"
            [plugins.hooks]
            timeout_seconds = 30
            "#,
        );
        assert!(err.is_err(), "an unknown key is a typo, not a preference");
    }

    #[test]
    fn parses_a_minimal_config() {
        let cfg = RatatoskrConfig::from_toml_str(
            r#"
            [rag_rat]
            command = ["rag-rat", "mcp", "serve"]

            [store]
            path = ".ratatoskr/state.sqlite3"

            [worktree]
            root = ".ratatoskr/worktrees"

            [models.scout]
            provider = "kimi"
            model = "k2"
            "#,
        )
        .unwrap();

        assert_eq!(cfg.rag_rat.command, ["rag-rat", "mcp", "serve"]);
        assert_eq!(cfg.store.path, PathBuf::from(".ratatoskr/state.sqlite3"));
        assert_eq!(cfg.models["scout"].provider, "kimi");
    }

    #[test]
    fn default_config_is_valid() {
        RatatoskrConfig::default().validate().unwrap();
    }

    #[test]
    fn validate_rejects_unusable_configs() {
        let invalid = |mutate: fn(&mut RatatoskrConfig)| {
            let mut cfg = RatatoskrConfig::default();
            mutate(&mut cfg);
            match cfg.validate() {
                Err(ConfigError::Invalid(_)) => {}
                other => panic!("expected Invalid, got {other:?}"),
            }
        };

        invalid(|c| c.rag_rat.command.clear());
        invalid(|c| c.sandbox.test_command.clear());
        invalid(|c| c.sandbox.backend = "docker".to_string());
        invalid(|c| c.implementer.max_iterations = 0);
    }

    #[test]
    fn landlock_backend_is_valid() {
        let mut cfg = RatatoskrConfig::default();
        cfg.sandbox.backend = "landlock".to_string();
        cfg.validate().unwrap();
    }

    #[test]
    fn default_config_serializes_and_reparses() {
        let toml_str = toml::to_string(&RatatoskrConfig::default()).unwrap();
        let reparsed = RatatoskrConfig::from_toml_str(&toml_str).unwrap();
        assert_eq!(reparsed.rag_rat.command.len(), 5);
        assert_eq!(reparsed.models.len(), 5);
        assert_eq!(reparsed.models["ask"].provider, "anthropic");
    }

    #[test]
    fn the_implementer_reads_its_keys_and_refuses_ones_it_does_not_know() {
        let config = RatatoskrConfig::from_toml_str(
            r#"
            [rag_rat]
            command = ["rag-rat", "mcp"]
            [store]
            path = ".ratatoskr/state.sqlite3"
            [worktree]
            root = ".ratatoskr/worktrees"
            [implementer]
            max_iterations = 5
            verify_threshold = "P1"
            "#,
        )
        .unwrap();
        assert_eq!(config.implementer.max_iterations, 5);
        assert_eq!(config.implementer.verify_threshold, "P1");
        // Omitted keys keep their defaults rather than failing.
        assert!(!config.implementer.always_fork);

        // A misspelled key must not read as "left at the default". Every field here decides
        // whether the run does something or skips it, so a silent default is a run that looks
        // configured and is not.
        let typo = r#"
            [rag_rat]
            command = ["rag-rat", "mcp"]
            [implementer]
            max_iterations = 3
            verify_treshold = "P1"
        "#;
        let err = toml::from_str::<RatatoskrConfig>(typo)
            .unwrap_err()
            .to_string();
        assert!(err.contains("verify_treshold"), "{err}");
    }

    #[test]
    fn acceptance_prefers_the_plan_but_a_pin_always_wins() {
        let mut sandbox = SandboxConfig {
            test_command: vec!["cargo".into(), "test".into()],
            ..Default::default()
        };
        let planned = vec![
            AcceptanceStep {
                name: "wasm build".into(),
                command: vec!["wasm-pack".into(), "build".into()],
            },
            AcceptanceStep {
                name: "browser tests".into(),
                command: vec!["npx".into(), "playwright".into(), "test".into()],
            },
        ];

        // What a task's own plan asks for, in order.
        let resolved = sandbox.acceptance(&planned);
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].name, "wasm build");
        assert_eq!(resolved[1].command, ["npx", "playwright", "test"]);

        // A plan that proposes nothing is still checked against something.
        let fallback = sandbox.acceptance(&[]);
        assert_eq!(fallback.len(), 1);
        assert_eq!(fallback[0].command, ["cargo", "test"]);

        // The escape hatch: a repo that pinned its acceptance must not have a plan replace it.
        // Misspelling it fails open, so the key is spelled once here and refused everywhere else.
        assert!(
            RatatoskrConfig::from_toml_str(
                r#"
                [rag_rat]
                command = ["rag-rat", "mcp"]
                [store]
                path = "s"
                [worktree]
                root = "w"
                [sandbox]
                backend = "landlock"
                image = "x"
                test_command = ["cargo", "test"]
                pin_acceptence = true
                "#,
            )
            .is_err(),
            "a misspelled pin must not read as unpinned"
        );
        sandbox.pin_acceptance = true;
        let pinned = sandbox.acceptance(&planned);
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned[0].command, ["cargo", "test"]);
    }

    #[test]
    fn every_route_sends_a_cap_whether_or_not_it_names_one() {
        let config = RatatoskrConfig::from_toml_str(
            r#"
            [rag_rat]
            command = ["rag-rat", "mcp"]
            [store]
            path = "s"
            [worktree]
            root = "w"
            [models.scout]
            provider = "anthropic"
            model = "claude-brand-new-9"
            [models.analyst]
            provider = "anthropic"
            model = "claude-opus-4-8"
            max_tokens = 64000
            "#,
        )
        .unwrap();

        // The case that lost a whole run: a model the provider client's table has never heard of
        // still goes out with a cap, because we set it rather than letting the client infer it.
        assert_eq!(config.models["scout"].max_tokens, None);
        assert_eq!(config.models["scout"].max_tokens(), DEFAULT_MAX_TOKENS);
        // And a route that needs more room says so.
        assert_eq!(config.models["analyst"].max_tokens(), 64_000);

        // A misspelling must not read as "left at the default" — this key decides whether a long
        // answer is truncated.
        assert!(
            RatatoskrConfig::from_toml_str(
                r#"
                [rag_rat]
                command = ["rag-rat", "mcp"]
                [store]
                path = "s"
                [worktree]
                root = "w"
                [models.scout]
                provider = "anthropic"
                model = "m"
                max_token = 64000
                "#,
            )
            .is_err()
        );
    }

    #[test]
    fn publishing_is_off_until_somebody_turns_it_on() {
        let bare = RatatoskrConfig::from_toml_str(
            r#"
            [rag_rat]
            command = ["rag-rat", "mcp"]
            [store]
            path = "s"
            [worktree]
            root = "w"
            "#,
        )
        .unwrap();
        // The only node that acts outside this machine. A run that opens a pull request nobody
        // expected is worse than one that publishes nothing, so this is a switch somebody throws
        // rather than something inferred from having configured a model.
        assert!(!bare.publish.enabled);

        let on = RatatoskrConfig::from_toml_str(
            r#"
            [rag_rat]
            command = ["rag-rat", "mcp"]
            [store]
            path = "s"
            [worktree]
            root = "w"
            [publish]
            enabled = true
            "#,
        )
        .unwrap();
        assert!(on.publish.enabled);

        // And a misspelling does not read as "left off" — it fails, like every other gate key.
        assert!(
            RatatoskrConfig::from_toml_str(
                r#"
                [rag_rat]
                command = ["rag-rat", "mcp"]
                [store]
                path = "s"
                [worktree]
                root = "w"
                [publish]
                enable = true
                "#,
            )
            .is_err()
        );
    }
}
