//! Domain types shared across Ratatoskr: the run state and the config shape.
//!
//! This crate is the stable base every other crate depends on and deliberately has **no** async
//! runtime dependency — if a type here ever needs `tokio`, it belongs in a different crate.

pub mod config;
pub mod state;

pub use config::{
    ConfigError, ModelRoute, RagRatConfig, RatatoskrConfig, StoreConfig, WorktreeConfig,
};
pub use state::{ParseRunStatusError, RunState, RunStatus};
