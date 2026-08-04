//! The `ratatoskr` binary. Phase 0 exposes only `--version` (free from clap) and `init`.
//! The `ask` / `plan` / `run` / `status` commands belong to later phases and are deliberately
//! absent — an empty stub command looks implemented when it isn't.

use std::path::Path;

use anyhow::Context;
use clap::{CommandFactory, Parser, Subcommand};
use ratatoskr_core::RatatoskrConfig;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "ratatoskr",
    version,
    about = "Orchestrator for rag-rat-driven coding runs"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Write a default `ratatoskr.toml` into the current directory.
    Init,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    match Cli::parse().command {
        Some(Command::Init) => init(),
        None => {
            Cli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}

/// Write a default config, leaving any existing `ratatoskr.toml` untouched.
fn init() -> anyhow::Result<()> {
    let path = Path::new("ratatoskr.toml");
    if path.exists() {
        println!("ratatoskr.toml already exists; leaving it untouched");
        return Ok(());
    }
    let toml = RatatoskrConfig::default()
        .to_toml_string()
        .context("serializing default config")?;
    std::fs::write(path, toml).context("writing ratatoskr.toml")?;
    println!("wrote ratatoskr.toml");
    Ok(())
}
