//! The `ratatoskr` binary. Phase 2 exposes `--version`, `init`, `ask`, and `plan`.
//! The `run` / `status` commands belong to later phases and are deliberately absent —
//! an empty stub command looks implemented when it isn't.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::{CommandFactory, Parser, Subcommand};
use ratatoskr_core::RatatoskrConfig;
use ratatoskr_nodes::PlanOutcome;
use tracing::Instrument as _;
use tracing_subscriber::EnvFilter;

/// System prompt for `ask`: ground answers in rag-rat's tools, don't guess.
const ASK_PREAMBLE: &str = include_str!("../prompts/ask.md");

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
        /// Which workflow to run, when this repo defines more than one.
        #[arg(long)]
        workflow: Option<String>,
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
        /// Which workflow to run, when this repo defines more than one. Omitted, a repo with one
        /// workflow uses it and a repo with several is asked to name one rather than guessed at.
        #[arg(long)]
        workflow: Option<String>,
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
        /// Use this run id instead of generating one. Lets a caller that spawns this command
        /// (the dashboard) record and follow the run without waiting for it to finish.
        #[arg(long)]
        run_id: Option<String>,
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
    /// Serve the observability dashboard over the checkpoint store.
    Serve {
        /// Address to bind. Defaults to loopback, and should stay there: the dashboard can
        /// START RUNS, and there is no auth — anyone who can reach this port can drive a coding
        /// CLI against the repo and spend API credits.
        #[arg(long, default_value = "127.0.0.1:7878")]
        addr: SocketAddr,
        /// Path to the config file, for the current directory. Ignored when `--project` is used:
        /// each of those reads its own `ratatoskr.toml`.
        #[arg(long, default_value = "ratatoskr.toml")]
        config: PathBuf,
        /// Watch another project's repository root. Repeatable. Each keeps its own store,
        /// worktrees, and logs; nothing is merged. Defaults to the current directory.
        #[arg(long = "project")]
        projects: Vec<PathBuf>,
        /// How many dashboard-started runs may be in flight at once, per project. The default is
        /// 1 because red-team characterises the baseline against the main checkout, so concurrent
        /// runs contend on one build directory and serialise there regardless.
        #[arg(long, default_value_t = 1)]
        max_runs: usize,
    },
    /// List the workflows a run can be given, and what each is for.
    Workflows,
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
            workflow,
            file,
            config,
            json,
        }) => plan(description, file, &config, json, workflow).await,
        Some(Command::Run {
            workflow,
            description,
            file,
            config,
            json,
            run_id,
        }) => run_cmd(description, file, &config, json, run_id, workflow).await,
        Some(Command::Bookkeep { run_id, config }) => bookkeep(&run_id, &config).await,
        Some(Command::Status { run_id, config }) => status(&run_id, &config).await,
        Some(Command::Serve {
            addr,
            config,
            projects,
            max_runs,
        }) => serve(addr, &config, projects, max_runs).await,
        Some(Command::Workflows) => workflows().await,
        Some(Command::Clean { force }) => clean(force).await,
        None => {
            Cli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}

/// Set up logging. Three sinks, each for a different reader:
///
/// - the console at `info` (or `RUST_LOG`), for whoever is watching;
/// - `.ratatoskr/logs/ratatoskr.log`, verbose prose at `debug` (or `RATATOSKR_LOG`), for reading
///   after the fact;
/// - `.ratatoskr/logs/ratatoskr.jsonl.<YYYY-MM-DD>`, one JSON object per event, for machines — the
///   dashboard tails it to show what a node is doing between checkpoints. Both files rotate daily
///   and the date is a *suffix*, so there is no bare `ratatoskr.jsonl`: a consumer opens the
///   newest match and follows the rollover. Consumers depend on this shape:
///
///   ```json
///   {"timestamp":"…","level":"INFO","message":"tool call","kind":"tool_call",
///    "tool":"semantic_search","target":"ratatoskr_agent",
///    "spans":[{"name":"run","run_id":"…"},{"name":"agent","node":"scout"}]}
///   ```
///
///   `kind` is one of `tool_call`, `model_text`, or `checkpoint`. `run_id` and `node` come from
///   the enclosing spans — the log file is per process and day, so concurrent runs interleave and
///   `run_id` is what separates them. A `checkpoint` event carries `node` as a field directly.
///
///   Not every line has a `run` span: anything logged before a run starts, and `serve`'s own
///   lines about launching and reaping children, are emitted outside one. Those carry `run_id` as
///   a plain field where they know it, so match on the field first and fall back to the span.
///
/// Returns the file-writer guards, which must be held for the process's lifetime or buffered
/// output is dropped.
fn init_logging() -> Vec<tracing_appender::non_blocking::WorkerGuard> {
    use tracing_subscriber::Layer as _;
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    let console_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let console = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(console_filter);

    // Best-effort file layers; if the log dir can't be created, fall back to console-only.
    let log_dir = std::fs::create_dir_all(".ratatoskr/logs");
    if let Err(e) = &log_dir {
        eprintln!("warning: could not create .ratatoskr/logs ({e}); logging to console only");
    }

    let (file_layer, guard) = match &log_dir {
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
        Err(_) => (None, None),
    };

    // The structured sink. Deliberately narrow: only ratatoskr's own events, so the file stays a
    // stream of run activity rather than a transcript of every dependency's chatter. `spans` is
    // what carries `run_id` and `node`, so a consumer can attribute an event without parsing prose.
    //
    // `ratatoskr` prefix-matches every `ratatoskr_*` target, so this is "our crates at info" and
    // nothing else. It also has to cover this binary itself — the `[[bin]]` is named `ratatoskr`,
    // so that, not `ratatoskr_cli`, is main.rs's module path — because the filter gates *spans*
    // as well as events, and the `run` span carrying `run_id` is opened here.
    let (json_layer, json_guard) = match &log_dir {
        Ok(()) => {
            let filter = EnvFilter::new(
                std::env::var("RATATOSKR_JSON_LOG")
                    .unwrap_or_else(|_| "ratatoskr=info".to_string()),
            );
            let appender = tracing_appender::rolling::daily(".ratatoskr/logs", "ratatoskr.jsonl");
            let (writer, guard) = tracing_appender::non_blocking(appender);
            let layer = tracing_subscriber::fmt::layer()
                .json()
                .flatten_event(true)
                .with_current_span(false)
                .with_span_list(true)
                .with_writer(writer)
                .with_filter(filter);
            (Some(layer), Some(guard))
        }
        Err(_) => (None, None),
    };

    tracing_subscriber::registry()
        .with(console)
        .with(file_layer)
        .with(json_layer)
        .init();
    [guard, json_guard].into_iter().flatten().collect()
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
        ratatoskr_mcp::ToolSet::from_servers(vec![client.offer()]),
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
    workflow: Option<String>,
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
    let result = ratatoskr_nodes::run_plan(ratatoskr_nodes::RunRequest {
        client: &client,
        config: &config,
        store: &store,
        run_id: &run_id,
        issue: &issue,
        engine: &engine,
        workflow: workflow.as_deref(),
    })
    .instrument(tracing::info_span!("run", run_id = %run_id))
    .await;

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
        println!("  • {r}");
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
    run_id: Option<String>,
    workflow: Option<String>,
) -> anyhow::Result<()> {
    let issue = read_issue(description, file)?;

    let config = load_config(config_path)?;
    let store = ratatoskr_store::Store::open(&config.store.path)
        .with_context(|| format!("opening store at {}", config.store.path.display()))?;
    let client = ratatoskr_mcp::RagRatClient::connect(config.rag_rat.clone())
        .await
        .context("connecting to rag-rat")?;

    let engine = load_rules().await?;
    let run_id = match run_id {
        // Reusing an id would interleave this run's checkpoints with the existing run's, since
        // the run row is an upsert and checkpoints are an unconstrained append.
        Some(id) if store.run_status(&id).await?.is_some() => {
            bail!("run {id} already exists; omit --run-id to start a new run")
        }
        Some(id) => id,
        None => uuid::Uuid::new_v4().to_string(),
    };
    let result = ratatoskr_nodes::run_full(ratatoskr_nodes::RunRequest {
        client: &client,
        config: &config,
        store: &store,
        run_id: &run_id,
        issue: &issue,
        engine: &engine,
        workflow: workflow.as_deref(),
    })
    .instrument(tracing::info_span!("run", run_id = %run_id))
    .await;

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
    let result = ratatoskr_nodes::run_bookkeeper(&client, &config, &store, run_id, &engine)
        .instrument(tracing::info_span!("run", run_id = %run_id))
        .await;

    if let Err(e) = client.shutdown().await {
        tracing::warn!("failed to shut down rag-rat cleanly: {e}");
    }

    let out = result.context("bookkeeper failed")?;
    print_bookkeeper(&out);
    Ok(())
}

/// Print what the bookkeeper decided the repository's memory should now say.
fn print_bookkeeper(out: &ratatoskr_nodes::BookkeeperOutput) {
    // Recording nothing is an ordinary outcome, so say why rather than reporting a bare absence.
    if let Some(reason) = &out.skipped {
        println!("no memory recorded: {reason}");
        return;
    }

    let list = |label: &str, memories: &[ratatoskr_nodes::MemoryWritten]| {
        if memories.is_empty() {
            return;
        }
        println!("{label} {} memory(ies):", memories.len());
        for m in memories {
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
    };
    list("wrote", &out.memories_written);
    list("revised", &out.memories_revised);
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

/// Serve the dashboard over one or more projects' stores. Reads them directly; runs started from
/// the dashboard are spawned as child processes, so this one never writes to them.
async fn serve(
    addr: SocketAddr,
    config_path: &Path,
    projects: Vec<PathBuf>,
    max_runs: usize,
) -> anyhow::Result<()> {
    let specs = if projects.is_empty() {
        // No `--project`: watch the current directory, exactly as before.
        let dir = std::env::current_dir().context("resolving the project directory")?;
        vec![project_spec(&dir, config_path)?]
    } else {
        projects
            .iter()
            .map(|dir| project_spec(dir, &dir.join("ratatoskr.toml")))
            .collect::<anyhow::Result<Vec<_>>>()?
    };

    ratatoskr_serve::serve(ratatoskr_serve::ServeOptions {
        addr,
        projects: specs,
        max_runs,
    })
    .await?;
    Ok(())
}

/// Resolve one project for `serve`: its config, and where that config's store actually is.
///
/// `store.path` is relative to the project it belongs to, but this process has a single working
/// directory, so it has to be joined here rather than left for the server to guess.
fn project_spec(dir: &Path, config_path: &Path) -> anyhow::Result<ratatoskr_serve::ProjectSpec> {
    let config = load_config(config_path)?;
    let dir = dir
        .canonicalize()
        .with_context(|| format!("resolving project directory {}", dir.display()))?;
    let store_path = if config.store.path.is_absolute() {
        config.store.path.clone()
    } else {
        dir.join(&config.store.path)
    };
    Ok(ratatoskr_serve::ProjectSpec {
        dir,
        config_path: config_path.to_path_buf(),
        store_path,
    })
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

    // The fork does not run when the analyst judged the task to call for no code change. The plan
    // above is the whole result, and reporting an empty baseline against an empty diff would read
    // as a change that passed rather than as work that was never asked for.
    let (Some(rt), Some(im)) = (&outcome.red_team, &outcome.implementer) else {
        println!(
            "\nNO CODE CHANGE: the analyst judged this task to need none, so the fork did not run."
        );
        return;
    };

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
    // Here rather than at each command, because every command that reaches a model reaches it
    // through this function — and a command that loaded the config but not the endpoint's headers
    // would talk to it as an unidentified client, which is how a run gets whatever default the
    // endpoint keeps for somebody else.
    ratatoskr_agent::configure_endpoint(config.endpoint.clone());
    Ok(config)
}

/// Print the registry. What `--workflow` accepts, and what each entry claims to be for — the same
/// declaration whatever chooses automatically will read.
async fn workflows() -> anyhow::Result<()> {
    let found = ratatoskr_nodes::registry().await?;
    for workflow in &found {
        println!("{}", workflow.name());
        let purpose = workflow.purpose();
        if !purpose.is_empty() {
            println!("    {purpose}");
        }
    }
    // A repo that has defined none still has the built-in, so this says which one a bare `run`
    // would use rather than leaving it to be inferred from a list of one.
    if found.len() == 1 {
        println!("\nThis repo defines no workflows of its own; `run` uses the built-in.");
    }
    Ok(())
}

/// Load the `.ratatoskr/rules/*.ts` agent rulesets (empty engine if the dir is absent), rejecting
/// any `defineAgent(name)` that no workflow governs.
///
/// The allowed set is the built-in nodes plus what every defined workflow declares — the union,
/// because rulesets load before a workflow is selected. Validating here rather than at first use
/// keeps a typo an error at startup instead of a node that silently ignores its ruleset.
async fn load_rules() -> anyhow::Result<std::sync::Arc<ratatoskr_script::ScriptEngine>> {
    let engine = ratatoskr_script::ScriptEngine::load(Path::new(".ratatoskr/rules"))
        .await
        .context("loading .ratatoskr/rules")?;
    let governable = ratatoskr_nodes::governable_nodes().await?;
    for name in engine.declared_agents() {
        if !governable.iter().any(|n| n == name) {
            bail!(
                "defineAgent(\"{name}\") targets a node no workflow governs; rulesets apply to: {}. \
                 A workflow that introduces a node declares it with defineWorkflow({{ nodes: [...] }}).",
                governable.join(", ")
            );
        }
    }
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::sync::{Arc, Mutex};

    /// Captures a layer's output so the emitted JSON can be asserted on.
    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Buffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("buffer mutex").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Buffer {
        type Writer = Buffer;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Pins the record shape `init_logging`'s JSON sink produces, because it is a contract the
    /// dashboard parses: `kind` and the event's own fields at the top level, and `run_id` reachable
    /// through `spans`. The layer options here mirror `init_logging`; change them together.
    #[test]
    fn the_json_sink_emits_the_documented_shape() {
        use tracing_subscriber::layer::SubscriberExt as _;

        let buf = Buffer::default();
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_current_span(false)
            .with_span_list(true)
            .with_writer(buf.clone());

        tracing::subscriber::with_default(tracing_subscriber::registry().with(layer), || {
            let span = tracing::info_span!("run", run_id = "run-abc");
            let _entered = span.enter();
            tracing::info!(
                kind = "checkpoint",
                node = "scout",
                bytes = 12,
                "checkpoint"
            );
        });

        let raw = String::from_utf8(buf.0.lock().expect("buffer mutex").clone()).expect("utf-8");
        let record: serde_json::Value =
            serde_json::from_str(raw.trim()).expect("each line is one JSON object");

        // Event fields are flattened to the top level, not nested under `fields`.
        assert_eq!(record["kind"], "checkpoint");
        assert_eq!(record["node"], "scout");
        assert_eq!(record["bytes"], 12);

        // `run_id` rides the enclosing span — this is what lets a consumer separate concurrent
        // runs sharing one file.
        let spans = record["spans"].as_array().expect("a span list");
        assert!(
            spans.iter().any(|s| s["run_id"] == "run-abc"),
            "run_id must be reachable through spans, got {spans:?}"
        );
    }

    #[test]
    fn the_buffer_writer_captures_what_is_written() {
        let mut buf = Buffer::default();
        buf.write_all(b"x").unwrap();
        assert_eq!(buf.0.lock().unwrap().as_slice(), b"x");
    }
}
