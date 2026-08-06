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
            if p.is_absolute() {
                p.clone()
            } else {
                root.join(p)
            }
        }));
        dirs
    }
}

/// Phase 3 implementer settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementerConfig {
    /// Which coding CLI to drive via ACP (`"claude"`). One target per the Phase 3 non-goals.
    pub cli: String,
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
            cli: "claude".to_string(),
            max_iterations: 3,
            verify_threshold: default_verify_threshold(),
            always_fork: false,
        }
    }
}

fn default_verify_threshold() -> String {
    "P2".to_string()
}

/// Phase 3 sandbox settings — where red-team and implementer run the repo's test command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// `"microsandbox"` (MicroVM, needs KVM) or `"landlock"` (bwrap+Landlock fallback).
    pub backend: String,
    /// OCI image the sandbox boots (microsandbox backend).
    pub image: String,
    /// The target repo's test command, run inside the sandbox to characterize pass/fail.
    pub test_command: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        SandboxConfig {
            // landlock builds with no network build script; microsandbox is opt-in behind
            // ratatoskr-exec's `microsandbox` feature (see its Cargo.toml).
            backend: "landlock".to_string(),
            image: "docker.io/library/rust:1-slim".to_string(),
            test_command: vec!["cargo".to_string(), "test".to_string()],
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
pub struct ModelRoute {
    pub provider: String,
    pub model: String,
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
        const CLIS: [&str; 1] = ["claude"];

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
        if !CLIS.contains(&self.implementer.cli.as_str()) {
            return Err(ConfigError::Invalid(format!(
                "implementer.cli `{}` is not one of {CLIS:?}",
                self.implementer.cli
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
            provider: provider.to_string(),
            model: model.to_string(),
        };
        RatatoskrConfig {
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
        invalid(|c| c.implementer.cli = "aider".to_string());
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
}
