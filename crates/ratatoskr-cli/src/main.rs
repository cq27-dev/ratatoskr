//! The `ratatoskr` binary. Phase 1 exposes `--version`, `init`, and `ask`.
//! The `plan` / `run` / `status` commands belong to later phases and are deliberately absent —
//! an empty stub command looks implemented when it isn't.

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::{CommandFactory, Parser, Subcommand};
use ratatoskr_core::RatatoskrConfig;
use tracing_subscriber::EnvFilter;

/// System prompt for `ask`: ground answers in rag-rat's tools, don't guess.
const ASK_PREAMBLE: &str = "You are a coding assistant answering questions about a specific \
    repository. You have rag-rat tools (semantic_search, symbol_lookup, and others) to search and \
    understand the code. Always ground your answer in what those tools return — call them rather \
    than guessing. If the tools don't surface an answer, say so.";

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
    /// Ask a question about the target repo; a single agent answers using rag-rat's tools.
    Ask {
        /// The question to answer.
        question: String,
        /// Path to the config file.
        #[arg(long, default_value = "ratatoskr.toml")]
        config: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load a gitignored `.env` (API keys, ANTHROPIC_BASE_URL, RUST_LOG) before anything reads
    // the environment. Real env vars already set take precedence over the file.
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    match Cli::parse().command {
        Some(Command::Init) => init(),
        Some(Command::Ask { question, config }) => ask(&question, &config).await,
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

/// Launch rag-rat, bind one agent to its tools, and answer `question`.
async fn ask(question: &str, config_path: &Path) -> anyhow::Result<()> {
    let config = load_config(config_path)?;
    let route = config
        .models
        .get("ask")
        .with_context(|| "no [models.ask] entry in the config; add one to route `ask`")?
        .clone();

    // Keep the client alive for the whole turn; the subprocess dies when it's dropped/shut down.
    let client = ratatoskr_mcp::RagRatClient::connect(config.rag_rat)
        .await
        .context("connecting to rag-rat")?;

    let answer = ratatoskr_agent::ask(
        &route,
        ASK_PREAMBLE,
        question,
        client.tools(),
        client.sink(),
    )
    .await;

    // Tear down rag-rat regardless of how the agent turn went, so no subprocess is orphaned.
    if let Err(e) = client.shutdown().await {
        tracing::warn!("failed to shut down rag-rat cleanly: {e}");
    }

    println!("{}", answer.context("agent failed to answer")?);
    Ok(())
}

fn load_config(path: &Path) -> anyhow::Result<RatatoskrConfig> {
    if !path.exists() {
        bail!(
            "config {} not found; run `ratatoskr init` to create one",
            path.display()
        );
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config {}", path.display()))?;
    RatatoskrConfig::from_toml_str(&text)
        .with_context(|| format!("parsing config {}", path.display()))
}
