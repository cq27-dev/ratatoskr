//! Domain types shared across Ratatoskr: the run state and the config shape.
//!
//! This crate is the stable base every other crate depends on and deliberately has **no** async
//! runtime dependency — if a type here ever needs `tokio`, it belongs in a different crate.

pub mod config;
pub mod policy;
pub mod shape;
pub mod state;
pub mod telemetry;

pub use config::{
    AcceptanceStep, CACHE_ROOT, CacheMount, ConfigError, DEFAULT_MAX_TOKENS, EndpointConfig,
    HookLimits, ImplementerConfig, MAX_SUBJECT_CHARS, ModelRoute, PluginConfig, PublishConfig,
    RagRatConfig, RatatoskrConfig, SandboxConfig, SessionScope, StoreConfig, WorktreeConfig,
};
pub use policy::{ToolDecision, ToolPolicy};
pub use state::{RunState, RunStatus};
pub use telemetry::{NodeTelemetry, TokenUsage};
