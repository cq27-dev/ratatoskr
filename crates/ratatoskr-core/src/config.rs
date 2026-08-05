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
}

impl Default for ImplementerConfig {
    fn default() -> Self {
        ImplementerConfig {
            cli: "claude".to_string(),
            max_iterations: 3,
        }
    }
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
