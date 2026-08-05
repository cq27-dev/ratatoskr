//! Domain types shared across Ratatoskr: the run state and the config shape.
//!
//! This crate is the stable base every other crate depends on and deliberately has **no** async
//! runtime dependency — if a type here ever needs `tokio`, it belongs in a different crate.

pub mod config;
pub mod policy;
pub mod state;

pub use config::{
    ConfigError, ImplementerConfig, ModelRoute, RagRatConfig, RatatoskrConfig, SandboxConfig,
    StoreConfig, WorktreeConfig,
};
pub use policy::{ToolDecision, ToolPolicy};
pub use state::{RunState, RunStatus};
