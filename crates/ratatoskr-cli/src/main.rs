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
    /// Replay the bookkeeper against a stored run's checkpoints — write memories, no re-run.
    Bookkeep {
        /// The run id to bookkeep.
        run_id: String,
        /// Path to the config file.
        #[arg(long, default_value = "ratatoskr.toml")]
        config: PathBuf,
    },
    /// Show a run's status and its per-node checkpoints from the store (no rag-rat, no LLM).
    Status {
        /// The run id to inspect.
        run_id: String,
        /// Path to the config file.
        #[arg(long, default_value = "ratatoskr.toml")]
        config: PathBuf,
    },
    /// Reclaim ratatoskr's per-run worktrees and their `ratatoskr/*` branches.
    ///
    /// Without `--force` it only lists what would be removed. Removal is destructive: it discards
    /// each worktree's uncommitted changes and force-deletes its branch.
    Clean {
        /// Actually remove the worktrees and branches (default is a listing only).
        #[arg(long)]
        force: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load a gitignored `.env` (API keys, ANTHROPIC_BASE_URL, RUST_LOG) before anything reads
    // the environment. Real env vars already set take precedence over the file.
    dotenvy::dotenv().ok();

    // Keep the file-logger guard alive for the whole process, else buffered logs are dropped.
    let _log_guard = init_logging();

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
        Some(Command::Bookkeep { run_id, config }) => bookkeep(&run_id, &config).await,
        Some(Command::Status { run_id, config }) => status(&run_id, &config).await,
        Some(Command::Clean { force }) => clean(force).await,
        None => {
            Cli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}

/// Set up logging: the console at `info` (or `RUST_LOG`), plus a verbose, daily-rotating file
/// under `.ratatoskr/logs/` capturing everything at `debug` (or `RATATOSKR_LOG`) for later
/// analysis. Returns the file-writer guard, which must be held for the process's lifetime.
fn init_logging() -> Option<tracing_appender::non_blocking::WorkerGuard> {
    use tracing_subscriber::Layer as _;
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    let console_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let console = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(console_filter);

    // Best-effort file layer; if the log dir can't be created, fall back to console-only.
    let (file_layer, guard) = match std::fs::create_dir_all(".ratatoskr/logs") {
        Ok(()) => {
            let file_filter = EnvFilter::new(
                std::env::var("RATATOSKR_LOG").unwrap_or_else(|_| "debug".to_string()),
            );
            let appender = tracing_appender::rolling::daily(".ratatoskr/logs", "ratatoskr.log");
            let (writer, guard) = tracing_appender::non_blocking(appender);
            let layer = tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(writer)
                .with_filter(file_filter);
            (Some(layer), Some(guard))
        }
        Err(e) => {
            eprintln!("warning: could not create .ratatoskr/logs ({e}); logging to console only");
            (None, None)
        }
    };

    tracing_subscriber::registry()
        .with(console)
        .with(file_layer)
        .init();
    guard
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
        None,
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

    let engine = load_rules().await?;
    let run_id = uuid::Uuid::new_v4().to_string();
    let result =
        ratatoskr_nodes::run_plan(&client, &config, &store, &run_id, &issue, &engine).await;

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

    // Skip empty placeholder items (e.g. from an older checkpoint) so the count and lines are clean.
    let items: Vec<_> = outcome
        .scout
        .related_items
        .iter()
        .filter(|i| i.is_meaningful())
        .collect();
    println!("RELATED ITEMS ({}):", items.len());
    for item in items {
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

    let engine = load_rules().await?;
    let run_id = uuid::Uuid::new_v4().to_string();
    let result =
        ratatoskr_nodes::run_full(&client, &config, &store, &run_id, &issue, &engine).await;

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

/// Replay the bookkeeper against a stored run's checkpoints.
async fn bookkeep(run_id: &str, config_path: &Path) -> anyhow::Result<()> {
    let config = load_config(config_path)?;
    let store = ratatoskr_store::Store::open(&config.store.path)
        .with_context(|| format!("opening store at {}", config.store.path.display()))?;
    let client = ratatoskr_mcp::RagRatClient::connect(config.rag_rat.clone())
        .await
        .context("connecting to rag-rat")?;

    let engine = load_rules().await?;
    let result = ratatoskr_nodes::run_bookkeeper(&client, &config, &store, run_id, &engine).await;

    if let Err(e) = client.shutdown().await {
        tracing::warn!("failed to shut down rag-rat cleanly: {e}");
    }

    let out = result.context("bookkeeper failed")?;
    print_bookkeeper(&out);
    Ok(())
}

/// Print the memories a bookkeeper run wrote.
fn print_bookkeeper(out: &ratatoskr_nodes::BookkeeperOutput) {
    if out.memories_written.is_empty() {
        println!("no memories written");
        return;
    }
    println!("wrote {} memory(ies):", out.memories_written.len());
    for m in &out.memories_written {
        let anchor = if m.anchor.is_empty() {
            "<unanchored>"
        } else {
            &m.anchor
        };
        println!("  • {} [{}] @ {}", m.memory_id, m.kind, anchor);
        if let Some(s) = &m.summary {
            println!("    {s}");
        }
    }
}

/// Show a run's status and per-node checkpoints from the store — pure read, no rag-rat or LLM.
async fn status(run_id: &str, config_path: &Path) -> anyhow::Result<()> {
    let config = load_config(config_path)?;
    let store = ratatoskr_store::Store::open(&config.store.path)
        .with_context(|| format!("opening store at {}", config.store.path.display()))?;

    let run_status = store.run_status(run_id).await?;
    let checkpoints = store.checkpoints_for_run(run_id).await?;

    match run_status {
        Some(s) => println!("run {run_id}: {s}"),
        None if checkpoints.is_empty() => bail!("no run {run_id} in the store"),
        None => println!("run {run_id}: (no status row)"),
    }

    if checkpoints.is_empty() {
        println!("(no checkpoints)");
        return Ok(());
    }
    for c in &checkpoints {
        println!("\n── {} @ {} ──", c.node_name, c.created_at);
        // Pretty-print the stored JSON; fall back to the raw text if it doesn't parse.
        match serde_json::from_str::<serde_json::Value>(&c.output_json) {
            Ok(v) => println!(
                "{}",
                serde_json::to_string_pretty(&v).unwrap_or_else(|_| c.output_json.clone())
            ),
            Err(_) => println!("{}", c.output_json),
        }
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
    for c in &rt.classifications {
        let reason = if c.reason.is_empty() {
            String::new()
        } else {
            format!(" — {}", c.reason)
        };
        println!("  [{}] {}{}", c.category, c.test, reason);
    }

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

    if let Some(bk) = &outcome.bookkeeper {
        println!("\nBOOKKEEPER:");
        print_bookkeeper(bk);
    }
}

/// Reclaim ratatoskr's per-run worktrees and their `ratatoskr/*` branches. Only worktrees on a
/// `ratatoskr/*` branch are touched — never the user's own or a foreign worktree. Needs no config;
/// it works off the current repo's git worktree registry.
async fn clean(force: bool) -> anyhow::Result<()> {
    let cwd = std::env::current_dir().context("resolving the current directory")?;
    // `main_root` is the stable anchor for every git call: it's never a removal target, so cleanup
    // still works when invoked from inside a worktree it's about to delete.
    let survey = ratatoskr_exec::survey_worktrees(&cwd)
        .await
        .context("listing ratatoskr worktrees")?;
    let main_root = &survey.main_root;
    let branches = ratatoskr_exec::managed_worktree_branches(main_root)
        .await
        .context("listing ratatoskr branches")?;

    if survey.managed.is_empty() && branches.is_empty() {
        println!("No ratatoskr worktrees or branches to clean.");
        return Ok(());
    }

    if !force {
        println!(
            "Would remove {} worktree(s) and {} branch(es):",
            survey.managed.len(),
            branches.len()
        );
        let attached: std::collections::HashSet<&str> =
            survey.managed.iter().map(|w| w.branch.as_str()).collect();
        for w in &survey.managed {
            println!(
                "  worktree {} (branch {})",
                w.path.as_path().display(),
                w.branch
            );
        }
        for b in &branches {
            if !attached.contains(b.as_str()) {
                println!("  branch {b} (orphaned — no worktree)");
            }
        }
        println!(
            "\nRemoval discards each worktree's uncommitted changes. \
             Re-run `ratatoskr clean --force` to proceed."
        );
        return Ok(());
    }

    for w in &survey.managed {
        match ratatoskr_exec::remove_worktree(main_root, &w.path).await {
            Ok(()) => println!("removed worktree {}", w.path.as_path().display()),
            Err(e) => eprintln!(
                "warning: could not remove worktree {}: {e}",
                w.path.as_path().display()
            ),
        }
    }
    // Clear registrations left by dirs removed above (or deleted out-of-band).
    if let Err(e) = ratatoskr_exec::prune_worktrees(main_root).await {
        eprintln!("warning: `git worktree prune` failed: {e}");
    }
    // Sweep branches independently — catches orphans whose worktree was already gone.
    for b in &branches {
        match ratatoskr_exec::delete_worktree_branch(main_root, b).await {
            Ok(()) => println!("deleted branch {b}"),
            Err(e) => eprintln!("warning: could not delete branch {b}: {e}"),
        }
    }
    println!("Done.");
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
    let config = RatatoskrConfig::from_toml_str(&text)
        .with_context(|| format!("parsing config {}", path.display()))?;
    // Reject unusable configs here, before any command touches rag-rat, a sandbox, or a model.
    config
        .validate()
        .with_context(|| format!("in config {}", path.display()))?;
    Ok(config)
}

/// The nodes a ruleset may govern — the LLM agents that go through `run_structured`. `memory` and
/// `implementer` don't (no model/tool set to override), so targeting them is a config error.
const RULESET_NODES: &[&str] = &["scout", "analyst", "bookkeeper", "redteam"];

/// Load the `.ratatoskr/rules/*.ts` agent rulesets (empty engine if the dir is absent), rejecting
/// any `defineAgent(name)` that isn't a governable node.
async fn load_rules() -> anyhow::Result<std::sync::Arc<ratatoskr_script::ScriptEngine>> {
    let engine = ratatoskr_script::ScriptEngine::load(Path::new(".ratatoskr/rules"))
        .await
        .context("loading .ratatoskr/rules")?;
    for name in engine.declared_agents() {
        if !RULESET_NODES.contains(&name) {
            bail!(
                "defineAgent(\"{name}\") targets an unknown node; rulesets apply to: {}",
                RULESET_NODES.join(", ")
            );
        }
    }
    Ok(engine)
}
