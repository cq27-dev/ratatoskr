//! `ratatoskr.toml` configuration.
//!
//! Several fields exist in Phase 0 as *shape* only — nothing reads them yet — so later phases
//! are written against a config that already has a slot for them rather than retrofitting one:
//! [`WorktreeConfig`] (Phase 3) and [`RatatoskrConfig::models`] (Phase 2 node routing).

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Top-level config loaded from `ratatoskr.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RatatoskrConfig {
    pub rag_rat: RagRatConfig,
    pub store: StoreConfig,
    pub worktree: WorktreeConfig,
    /// Per-node model routing, keyed by node name (`"scout"`, `"analyst"`, ...). Unused until
    /// Phase 2 wires nodes; present now so nodes route against an existing table.
    #[serde(default)]
    pub models: HashMap<String, ModelRoute>,
    #[serde(default)]
    pub implementer: ImplementerConfig,
    #[serde(default)]
    pub sandbox: SandboxConfig,
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
            backend: "microsandbox".to_string(),
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

/// Root for Phase 3's per-run git worktrees. Field exists now, unused until Phase 3.
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

/// Error parsing `ratatoskr.toml`.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to parse ratatoskr.toml: {0}")]
    Parse(#[from] toml::de::Error),
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
            ]),
            implementer: ImplementerConfig::default(),
            sandbox: SandboxConfig::default(),
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
    fn default_config_serializes_and_reparses() {
        let toml_str = toml::to_string(&RatatoskrConfig::default()).unwrap();
        let reparsed = RatatoskrConfig::from_toml_str(&toml_str).unwrap();
        assert_eq!(reparsed.rag_rat.command.len(), 5);
        assert_eq!(reparsed.models.len(), 4);
        assert_eq!(reparsed.models["ask"].provider, "anthropic");
    }
}
