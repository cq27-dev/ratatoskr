//! The `ratatoskr` binary. Phase 2 exposes `--version`, `init`, `ask`, and `plan`.
//! The `run` / `status` commands belong to later phases and are deliberately absent —
//! an empty stub command looks implemented when it isn't.

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::{CommandFactory, Parser, Subcommand};
use ratatoskr_core::RatatoskrConfig;
use ratatoskr_nodes::PlanOutcome;
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
    /// Plan work for an issue: scout → memory → analyst, printing a grounded summary.
    Plan {
        /// The issue description (omit and use --file for long text).
        description: Option<String>,
        /// Read the issue description from a file instead of the argument.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Path to the config file.
        #[arg(long, default_value = "ratatoskr.toml")]
        config: PathBuf,
        /// Print the raw structured RunState as JSON instead of a formatted summary.
        #[arg(long)]
        json: bool,
    },
    /// Full run: plan, then fork red-team ∥ implementer in a worktree, then converge.
    Run {
        /// The issue description (omit and use --file for long text).
        description: Option<String>,
        /// Read the issue description from a file instead of the argument.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Path to the config file.
        #[arg(long, default_value = "ratatoskr.toml")]
        config: PathBuf,
        /// Print the raw structured RunState as JSON instead of a formatted summary.
        #[arg(long)]
        json: bool,
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
        Some(Command::Plan {
            description,
            file,
            config,
            json,
        }) => plan(description, file, &config, json).await,
        Some(Command::Run {
            description,
            file,
            config,
            json,
        }) => run_cmd(description, file, &config, json).await,
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

/// Run the scout → memory → analyst plan flow for an issue and print the result.
async fn plan(
    description: Option<String>,
    file: Option<PathBuf>,
    config_path: &Path,
    json: bool,
) -> anyhow::Result<()> {
    let issue = read_issue(description, file)?;

    let config = load_config(config_path)?;
    let store = ratatoskr_store::Store::open(&config.store.path)
        .with_context(|| format!("opening store at {}", config.store.path.display()))?;
    let client = ratatoskr_mcp::RagRatClient::connect(config.rag_rat.clone())
        .await
        .context("connecting to rag-rat")?;

    let run_id = uuid::Uuid::new_v4().to_string();
    let result = ratatoskr_nodes::run_plan(&client, &config, &store, &run_id, &issue).await;

    // Tear down rag-rat regardless of outcome.
    if let Err(e) = client.shutdown().await {
        tracing::warn!("failed to shut down rag-rat cleanly: {e}");
    }

    let outcome = result.context("plan run failed")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&outcome.state)?);
    } else {
        print_summary(&run_id, &outcome);
    }
    Ok(())
}

/// Render a plan outcome as a short, readable report — this is what makes `plan` useful to run.
fn print_summary(run_id: &str, outcome: &PlanOutcome) {
    println!("── plan {run_id} ──\n");

    println!("RELATED ITEMS ({}):", outcome.scout.related_items.len());
    for item in &outcome.scout.related_items {
        println!("  • [{}] {} — {}", item.item_key, item.title, item.relation);
    }
    if !outcome.scout.papertrail_summary.is_empty() {
        println!("  {}", outcome.scout.papertrail_summary);
    }

    println!("\nREPO MEMORIES ({}):", outcome.memory.memories.len());
    for m in &outcome.memory.memories {
        println!("  • ({}) {}", m.kind, m.title);
    }

    let a = &outcome.analyst;
    println!("\nIMPACT:\n  {}", a.impact_summary);
    if !a.touched.is_empty() {
        println!("  touches: {}", a.touched.join(", "));
    }

    println!("\nRISKS ({}):", a.risks.len());
    for r in &a.risks {
        println!("  • [{}] {}", r.severity, r.description);
    }

    println!("\nREQUIREMENTS ({}):", a.requirements.len());
    for req in &a.requirements {
        println!("  • {req}");
    }

    println!("\nRESIDUAL RISK:\n  {}", a.residual_risk);
}

/// Full fork+converge run for an issue.
async fn run_cmd(
    description: Option<String>,
    file: Option<PathBuf>,
    config_path: &Path,
    json: bool,
) -> anyhow::Result<()> {
    let issue = read_issue(description, file)?;

    let config = load_config(config_path)?;
    let store = ratatoskr_store::Store::open(&config.store.path)
        .with_context(|| format!("opening store at {}", config.store.path.display()))?;
    let client = ratatoskr_mcp::RagRatClient::connect(config.rag_rat.clone())
        .await
        .context("connecting to rag-rat")?;

    let run_id = uuid::Uuid::new_v4().to_string();
    let result = ratatoskr_nodes::run_full(&client, &config, &store, &run_id, &issue).await;

    if let Err(e) = client.shutdown().await {
        tracing::warn!("failed to shut down rag-rat cleanly: {e}");
    }

    let outcome = result.context("run failed")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&outcome.state)?);
    } else {
        print_run_summary(&run_id, &outcome);
    }
    Ok(())
}

/// Read the issue text from a positional argument or `--file` (exactly one).
fn read_issue(description: Option<String>, file: Option<PathBuf>) -> anyhow::Result<String> {
    match (description, file) {
        (Some(d), None) => Ok(d),
        (None, Some(f)) => {
            std::fs::read_to_string(&f).with_context(|| format!("reading {}", f.display()))
        }
        (Some(_), Some(_)) => bail!("pass either a description or --file, not both"),
        (None, None) => bail!("provide an issue description or --file"),
    }
}

/// Render a full-run outcome: the plan summary plus the fork+converge result.
fn print_run_summary(run_id: &str, outcome: &ratatoskr_nodes::RunOutcome) {
    println!("── run {run_id} ──\n");
    println!(
        "STATUS: {}  (implementer iterations: {})",
        outcome.status, outcome.iterations
    );

    let rt = &outcome.red_team;
    println!(
        "\nBASELINE (red-team): {} failing, {} passing",
        rt.failing_tests.len(),
        rt.passing_tests.len()
    );

    let im = &outcome.implementer;
    println!(
        "AFTER CHANGE: {} failing, {} passing",
        im.failing_tests.len(),
        im.passing_tests.len()
    );

    let new_failures =
        ratatoskr_nodes::converge::newly_introduced_failures(&rt.failing_tests, &im.failing_tests);
    if new_failures.is_empty() {
        println!("NEW FAILURES: none");
    } else {
        println!("NEW FAILURES ({}):", new_failures.len());
        for f in &new_failures {
            println!("  • {f}");
        }
    }

    println!("\nWORKTREE: {}", im.worktree_path);
    if !im.touched_files.is_empty() {
        println!("TOUCHED: {}", im.touched_files.join(", "));
    }
    if !im.diff_summary.is_empty() {
        println!("\nDIFF:\n{}", im.diff_summary);
    }
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
