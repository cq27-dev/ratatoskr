//! The `ratatoskr` binary. Phase 2 exposes `--version`, `init`, `ask`, and `plan`.
//! The `run` / `status` commands belong to later phases and are deliberately absent —
//! an empty stub command looks implemented when it isn't.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::{CommandFactory, Parser, Subcommand};
use ratatoskr_core::auth::{Role, Visibility};
use ratatoskr_core::{RatatoskrConfig, RunStatus};
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
        /// Address to bind.
        ///
        /// Safe to expose, unlike earlier versions: starting a run and answering a clarification
        /// both need an operator session, and a project is readable without one only if it was
        /// named with `--public`. Put TLS in front of it and pass `--secure-cookies`.
        #[arg(long, default_value = "127.0.0.1:7878")]
        addr: SocketAddr,
        /// Where the clarification rendezvous binds — the endpoint a run process calls to ask a
        /// human a question. Loopback only: reaching it is what stands in for a credential.
        #[arg(long, default_value = "127.0.0.1:7879")]
        internal_addr: SocketAddr,
        /// Serve this project's runs to anyone, with no sign-in. Repeatable, by directory name.
        ///
        /// Everything a run recorded becomes public: the issue text, the model's output, and the
        /// contents of every file its tools read. Starting runs still needs an operator.
        #[arg(long = "public")]
        public: Vec<String>,
        /// Where this instance keeps its accounts and sessions. Not a project's store — one
        /// instance can watch several projects, and identity belongs to none of them.
        #[arg(long, default_value = ".ratatoskr/auth.sqlite3")]
        auth_db: PathBuf,
        /// Enable the GitHub integration under this trigger word, without any sigil.
        ///
        /// A comment saying `/<word> …` or `@<word> …` starts a run, if whoever wrote it maps to
        /// an operator here (`ratatoskr users link-github`). Prefer the slash form unless the word
        /// is a GitHub account you own: an `@` is a real mention, so it notifies whoever does.
        ///
        /// The webhook secret comes from RATATOSKR_GITHUB_WEBHOOK_SECRET; which repository a
        /// delivery is about is read from each project's `origin`, so there is nothing to keep in
        /// step by hand.
        #[arg(long = "github-bot")]
        github_bot: Option<String>,
        /// The GitHub login the bot's own comments come from, if it has an account.
        ///
        /// Only needed when it differs from the trigger word — the account available on GitHub is
        /// rarely the word people want to type. It is what stops the bot treating its own comments
        /// as new instructions once it starts posting; a GitHub App's `[bot]` suffix is handled.
        #[arg(long = "github-account")]
        github_account: Option<String>,
        /// Mark the session cookie `Secure`. Set this whenever the instance is reached over
        /// https, and leave it off for loopback — a browser discards a `Secure` cookie sent over
        /// plain http, which looks exactly like sign-in silently failing.
        #[arg(long)]
        secure_cookies: bool,
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
    /// Fetch this project's dependencies into the caches a run mounts.
    ///
    /// The one place a project is allowed a network. Runs it in the configured image with the
    /// network on and the caches writable; every run afterwards mounts them read-only and offline,
    /// so an acceptance check never resolves a dependency and the baseline and post-change runs
    /// cannot disagree about what a registry served at two different moments.
    ///
    /// Re-run it when a lockfile changes. Nothing runs it for you: that is the point.
    Prepare {
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
    /// List, tag, delete, export and import runs.
    Runs {
        #[command(subcommand)]
        command: RunsCommand,
        /// Path to the config file.
        #[arg(long, default_value = "ratatoskr.toml")]
        config: PathBuf,
    },
    /// Manage who may use a hosted instance.
    ///
    /// Only needed for an instance other people can reach. A loopback dashboard needs no accounts:
    /// whoever can reach the port already owns the checkout.
    Users {
        #[command(subcommand)]
        command: UsersCommand,
        /// The identity database `serve` was pointed at.
        #[arg(long, default_value = ".ratatoskr/auth.sqlite3")]
        auth_db: PathBuf,
    },
}

/// Managing who may use a hosted instance.
#[derive(Subcommand)]
enum UsersCommand {
    /// Create an account.
    ///
    /// The password is read from the `RATATOSKR_PASSWORD` environment variable, never from an
    /// argument: an argument is visible to every other process on the machine through `ps` and is
    /// written to the shell's history file.
    Add {
        /// The username to sign in with.
        username: String,
        /// What to call them in the dashboard. Defaults to the username.
        #[arg(long)]
        name: Option<String>,
        /// `viewer` reads, `operator` also starts runs and answers clarifications, `admin` also
        /// manages accounts.
        #[arg(long, default_value = "viewer")]
        role: String,
    },
    /// List accounts, their roles, and which are disabled.
    List,
    /// Change an account's password, ending every session it had open.
    ///
    /// Read from `RATATOSKR_PASSWORD`, as for `add`.
    Passwd { username: String },
    /// Change what an account may do. Takes effect on that account's next request.
    Role {
        username: String,
        /// `viewer`, `operator`, or `admin`.
        role: String,
    },
    /// Stop an account signing in, and close the sessions it already has.
    Disable { username: String },
    /// Let a disabled account sign in again.
    Enable { username: String },
    /// Let an account act through GitHub, so mentioning the bot as them starts a run.
    ///
    /// Takes GitHub's numeric user id, not a login: a login can be changed and then handed to
    /// someone else, and an identity keyed on it would follow the name rather than the person.
    /// `curl -s https://api.github.com/users/<login> | jq .id` prints it.
    LinkGithub {
        username: String,
        /// GitHub's numeric user id.
        github_id: String,
    },
}

#[derive(Subcommand)]
enum RunsCommand {
    /// List runs, most recent first.
    List {
        /// Only runs carrying this tag. Repeatable; a run must carry all of them.
        #[arg(long = "tag")]
        tags: Vec<String>,
        /// Only runs with this status.
        #[arg(long)]
        status: Option<String>,
        /// Only runs imported from this origin, or `local` for ones produced here.
        #[arg(long)]
        origin: Option<String>,
        #[arg(long, default_value_t = 30)]
        limit: usize,
    },
    /// Add tags to a run. Tags are what group runs into the arms of an experiment.
    Tag {
        run_id: String,
        /// One or more tags.
        #[arg(required = true)]
        tags: Vec<String>,
    },
    /// Remove tags from a run.
    Untag {
        run_id: String,
        #[arg(required = true)]
        tags: Vec<String>,
    },
    /// Mark runs abandoned, for ones whose process is gone.
    ///
    /// A run's status is written by the process running it, so one that was killed, crashed, or
    /// outlived its machine stays `running` in the store and on the dashboard forever. This is how
    /// a human says it is not coming back. Nothing else is touched: the checkpoints, the events
    /// and the worktree are all still there to read.
    ///
    /// Refuses a run that already reached a terminal status, so a finished run cannot be relabelled
    /// by a mistyped id.
    Abandon {
        #[arg(required = true)]
        run_ids: Vec<String>,
    },
    /// Delete runs and everything recorded about them.
    ///
    /// Without `--force` it only lists what would go. Deletion takes the run's checkpoints and its
    /// event history with it, and cannot be undone — export first if the run is worth keeping.
    Rm {
        #[arg(required = true)]
        run_ids: Vec<String>,
        #[arg(long)]
        force: bool,
    },
    /// Store a run's event history from the log files, so it survives their rotation.
    ///
    /// Runs automatically when a run finishes. Use this to backfill runs that finished before the
    /// store kept histories, or whose ingest was interrupted. Idempotent.
    Ingest {
        /// Runs to ingest. Defaults to every run in the store.
        run_ids: Vec<String>,
    },
    /// Write runs to a BSON bundle for someone else to analyse.
    Export {
        #[arg(required = true)]
        run_ids: Vec<String>,
        /// Where to write it.
        #[arg(long, short)]
        out: PathBuf,
    },
    /// Read a bundle exported somewhere else.
    ///
    /// Imported runs land alongside your own, tagged with where they came from. A run whose id is
    /// already here is left alone rather than overwritten.
    Import { bundle: PathBuf },
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
            internal_addr,
            public,
            auth_db,
            secure_cookies,
            github_bot,
            github_account,
            config,
            projects,
            max_runs,
        }) => {
            serve(ServeArgs {
                addr,
                internal_addr,
                config_path: config,
                projects,
                public,
                max_runs,
                auth_db,
                secure_cookies,
                github_bot,
                github_account,
            })
            .await
        }
        Some(Command::Workflows) => workflows().await,
        Some(Command::Prepare { config }) => prepare(&config).await,
        Some(Command::Clean { force }) => clean(force).await,
        Some(Command::Runs { command, config }) => runs(command, &config).await,
        Some(Command::Users { command, auth_db }) => users(command, &auth_db).await,
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
    let client = connect_rag_rat(&config.rag_rat).await?;

    let answer = ratatoskr_agent::ask(
        &route,
        ASK_PREAMBLE,
        question,
        ratatoskr_mcp::ToolSet::from_servers(client.iter().map(|c| c.offer()).collect()),
        None,
    )
    .await;

    // Tear down rag-rat regardless of how the agent turn went, so no subprocess is orphaned.
    shutdown_rag_rat(client).await;

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
    let client = connect_rag_rat(&config.rag_rat).await?;
    let exa = connect_exa(config.exa.as_ref()).await;

    let engine = load_rules(&config).await?;
    let run_id = uuid::Uuid::new_v4().to_string();
    let result = ratatoskr_nodes::run_plan(ratatoskr_nodes::RunRequest {
        client: client.as_ref(),
        exa: exa.as_ref(),
        config: &config,
        store: &store,
        run_id: &run_id,
        issue: &issue,
        engine: &engine,
        workflow: workflow.as_deref(),
    })
    .instrument(tracing::info_span!("run", run_id = %run_id))
    .await;

    // Tear down configured MCP clients regardless of outcome.
    shutdown_exa(exa).await;
    shutdown_rag_rat(client).await;

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
    let client = connect_rag_rat(&config.rag_rat).await?;
    let exa = connect_exa(config.exa.as_ref()).await;

    let engine = load_rules(&config).await?;
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
        client: client.as_ref(),
        exa: exa.as_ref(),
        config: &config,
        store: &store,
        run_id: &run_id,
        issue: &issue,
        engine: &engine,
        workflow: workflow.as_deref(),
    })
    .instrument(tracing::info_span!("run", run_id = %run_id))
    .await;

    shutdown_exa(exa).await;
    shutdown_rag_rat(client).await;

    // Make the run's history durable now that it has finished, so it survives the log files
    // rotating away without anyone remembering to run `runs ingest`. This happens before the
    // error is propagated below, because a failed run's events are precisely the ones most worth
    // reviewing later. Best-effort, like provenance: a store hiccup here must not mask the run's
    // real outcome, and `runs ingest` remains the idempotent way to backfill if this does not land.
    let log_dir = PathBuf::from(LOG_DIR);
    if let Err(e) = ingest_run(&store, &log_dir, &run_id).await {
        tracing::warn!(run_id = %run_id, "failed to ingest run events: {e}");
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
    let client = connect_rag_rat(&config.rag_rat).await?;

    let engine = load_rules(&config).await?;
    let result = ratatoskr_nodes::run_bookkeeper(client.as_ref(), &config, &store, run_id, &engine)
        .instrument(tracing::info_span!("run", run_id = %run_id))
        .await;

    shutdown_rag_rat(client).await;

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
/// What `serve` was asked for, as one value rather than an argument train.
struct ServeArgs {
    addr: SocketAddr,
    internal_addr: SocketAddr,
    config_path: PathBuf,
    projects: Vec<PathBuf>,
    /// Slugs to make readable without logging in.
    public: Vec<String>,
    max_runs: usize,
    auth_db: PathBuf,
    secure_cookies: bool,
    github_bot: Option<String>,
    github_account: Option<String>,
}

async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    let ServeArgs {
        addr,
        internal_addr,
        config_path,
        projects,
        public,
        max_runs,
        auth_db,
        secure_cookies,
        github_bot,
        github_account,
    } = args;
    let config_path = config_path.as_path();
    let mut specs = if projects.is_empty() {
        // No `--project`: watch the current directory, exactly as before.
        let dir = std::env::current_dir().context("resolving the project directory")?;
        vec![project_spec(&dir, config_path)?]
    } else {
        projects
            .iter()
            .map(|dir| project_spec(dir, &dir.join("ratatoskr.toml")))
            .collect::<anyhow::Result<Vec<_>>>()?
    };

    // Visibility is decided here, from this instance's flags — never from the project's own
    // config, which lives in the repository and would let a checkout publish itself.
    let public: std::collections::BTreeSet<&str> = public.iter().map(String::as_str).collect();
    for spec in &mut specs {
        let slug = spec
            .dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if public.contains(slug) {
            spec.visibility = Visibility::Public;
        }
    }
    // A name that matches nothing is a typo, and the failure it produces — a project that stays
    // private — is invisible. Better to say so than to serve the opposite of what was asked.
    let known: std::collections::BTreeSet<&str> = specs
        .iter()
        .filter_map(|s| s.dir.file_name().and_then(|n| n.to_str()))
        .collect();
    let unknown: Vec<&&str> = public.difference(&known).collect();
    if !unknown.is_empty() {
        bail!(
            "--public names no project being served: {}",
            unknown
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // From the environment, never a flag: a webhook secret in an argument is readable by every
    // process on the machine and lands in shell history, and it is the only thing standing between
    // a public endpoint and someone else's runs.
    let github = match github_bot {
        Some(bot) => {
            let secret = std::env::var("RATATOSKR_GITHUB_WEBHOOK_SECRET").map_err(|_| {
                anyhow::anyhow!(
                    "--github-bot needs RATATOSKR_GITHUB_WEBHOOK_SECRET set to the same secret \
                     the webhook was created with"
                )
            })?;
            if secret.chars().count() < 16 {
                bail!("that webhook secret is shorter than 16 characters");
            }
            Some(ratatoskr_serve::github::GitHubConfig {
                trigger: bot.trim_start_matches(['@', '/']).to_string(),
                account: github_account.map(|a| a.trim_start_matches('@').to_string()),
                secret,
            })
        }
        None => None,
    };

    ratatoskr_serve::serve(ratatoskr_serve::ServeOptions {
        addr,
        internal_addr,
        projects: specs,
        max_runs,
        auth_db,
        secure_cookies,
        github,
    })
    .await?;
    Ok(())
}

/// Account management for a hosted instance.
async fn users(command: UsersCommand, auth_db: &Path) -> anyhow::Result<()> {
    let auth = ratatoskr_store::auth::AuthStore::open(auth_db)
        .with_context(|| format!("opening the identity database at {}", auth_db.display()))?;

    match command {
        UsersCommand::Add {
            username,
            name,
            role,
        } => {
            let role = parse_role(&role)?;
            let password = password_from_env()?;
            let display = name.unwrap_or_else(|| username.clone());
            auth.create_local(&username, &password, &display, role)
                .await?;
            println!("added {username} ({role})");
            if role >= Role::Operator {
                // Worth saying out loud: this account can now spend money and change a checkout.
                println!("  {username} can start runs and answer clarifications");
            }
        }
        UsersCommand::List => {
            let listed = auth.list_principals().await?;
            if listed.is_empty() {
                println!("no accounts — `ratatoskr users add <username> --role operator`");
            }
            for (principal, disabled) in listed {
                let state = if disabled { "  (disabled)" } else { "" };
                println!(
                    "{:<10} {:<20} {}{state}",
                    principal.role.as_str(),
                    principal.display_name,
                    principal.principal_id
                );
            }
        }
        UsersCommand::Passwd { username } => {
            let password = password_from_env()?;
            if !auth.set_password(&username, &password).await? {
                bail!("no account called `{username}`");
            }
            println!("changed the password for {username}; its open sessions are closed");
        }
        UsersCommand::Role { username, role } => {
            let role = parse_role(&role)?;
            let principal_id = principal_id_for(&auth, &username).await?;
            auth.set_role(&principal_id, role).await?;
            println!("{username} is now {role}");
        }
        UsersCommand::Disable { username } => {
            let principal_id = principal_id_for(&auth, &username).await?;
            auth.set_disabled(&principal_id, true).await?;
            println!("disabled {username}; its open sessions are closed");
        }
        UsersCommand::Enable { username } => {
            let principal_id = principal_id_for(&auth, &username).await?;
            auth.set_disabled(&principal_id, false).await?;
            println!("enabled {username}");
        }
        UsersCommand::LinkGithub {
            username,
            github_id,
        } => {
            if !github_id.chars().all(|c| c.is_ascii_digit()) {
                bail!(
                    "`{github_id}` is not a GitHub user id — it is numeric, and a login will not \
                     do: logins can be changed and reassigned"
                );
            }
            let principal = auth
                .principal_for_local(&username)
                .await?
                .ok_or_else(|| anyhow::anyhow!("no account called `{username}`"))?;
            auth.attach_identity(&principal.principal_id, "github", &github_id)
                .await?;
            println!("{username} can now act through GitHub user {github_id}");
            if principal.role < Role::Operator {
                // Linked but powerless, which is a confusing state to leave someone in silently.
                println!(
                    "  note: {username} is a {} — mentioning the bot will be ignored until they \
                     are an operator",
                    principal.role.as_str()
                );
            }
        }
    }
    Ok(())
}

/// The password, from the environment rather than an argument.
///
/// An argument is readable by every process on the machine for as long as the command runs, and
/// ends up in the shell's history file afterwards. Neither is acceptable for the credential that
/// guards starting runs.
fn password_from_env() -> anyhow::Result<String> {
    let password = std::env::var("RATATOSKR_PASSWORD").map_err(|_| {
        anyhow::anyhow!(
            "set RATATOSKR_PASSWORD to the password for this account \
             (an argument would be visible in `ps` and in shell history)"
        )
    })?;
    if password.chars().count() < 12 {
        // A length floor rather than a character-class rule: length is what actually resists
        // guessing, and composition rules mostly produce predictable substitutions.
        bail!("that password is shorter than 12 characters");
    }
    Ok(password)
}

fn parse_role(role: &str) -> anyhow::Result<Role> {
    role.parse::<Role>()
        .map_err(|_| anyhow::anyhow!("`{role}` is not a role — use viewer, operator, or admin"))
}

/// The principal behind a local username, for the commands that address one by name.
async fn principal_id_for(
    auth: &ratatoskr_store::auth::AuthStore,
    username: &str,
) -> anyhow::Result<String> {
    auth.principal_for_local(username)
        .await?
        .map(|p| p.principal_id)
        .ok_or_else(|| anyhow::anyhow!("no account called `{username}`"))
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
        // Overridden by `--public` in `serve`; private is the direction a mistake should fail in.
        visibility: Visibility::default(),
    })
}

/// Connect to rag-rat, unless this repository runs without it.
///
/// `None` is an ordinary answer, not a degraded one: a config with no `[rag_rat]` section is a
/// repository that wants the harness and not a code index. What is lost is real — semantic search,
/// the call graph, the papertrail, and memory — so it is said once, at `info`, rather than left for
/// someone to infer from a node that never searches anything.
async fn connect_rag_rat(
    config: &ratatoskr_core::RagRatConfig,
) -> anyhow::Result<Option<ratatoskr_mcp::RagRatClient>> {
    if !config.configured() {
        tracing::info!(
            "no [rag_rat] in the config: running without semantic search, graph traversal, the \
             papertrail, or memory"
        );
        return Ok(None);
    }
    Ok(Some(
        ratatoskr_mcp::RagRatClient::connect(config.clone())
            .await
            .context("connecting to rag-rat")?,
    ))
}

/// Tear down rag-rat if there was one. A failure here loses nothing but a subprocess.
async fn shutdown_rag_rat(client: Option<ratatoskr_mcp::RagRatClient>) {
    if let Some(client) = client
        && let Err(e) = client.shutdown().await
    {
        tracing::warn!("failed to shut down rag-rat cleanly: {e}");
    }
}

/// Connect to Exa only when the config explicitly opts into third-party web egress.
async fn connect_exa(
    config: Option<&ratatoskr_core::ExaConfig>,
) -> Option<ratatoskr_mcp::ExaClient> {
    let config = config.filter(|config| config.configured())?;
    match ratatoskr_mcp::ExaClient::connect(config).await {
        Ok(client) => Some(client),
        Err(error) => {
            tracing::warn!("Exa MCP server unavailable, web tools are not offered: {error}");
            None
        }
    }
}

async fn shutdown_exa(client: Option<ratatoskr_mcp::ExaClient>) {
    if let Some(client) = client
        && let Err(error) = client.shutdown().await
    {
        tracing::warn!("failed to shut down Exa cleanly: {error}");
    }
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
        rt.passed_tests
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
        im.passed_tests
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
/// Where this project's daily logs are written. Fixed, like the directory `run` writes them to.
const LOG_DIR: &str = ".ratatoskr/logs";

/// Shorten a run id to the prefix every listing shows.
fn short(id: Option<&str>) -> String {
    id.map(|s| s.chars().take(8).collect())
        .unwrap_or_else(|| "—".to_string())
}

/// Make a run's event history durable, so it survives the log files rotating away.
async fn ingest_run(
    store: &ratatoskr_store::Store,
    log_dir: &Path,
    run_id: &str,
) -> anyhow::Result<usize> {
    let rows = ratatoskr_serve::events::rows_for_run(log_dir, run_id).await;
    if rows.is_empty() {
        return Ok(0);
    }
    Ok(store.ingest_events(run_id, rows).await?)
}

/// Who an export says it came from. Not identity — a label for telling one machine's runs from
/// another's after they are side by side.
fn exported_by() -> String {
    let user = std::env::var("USER").unwrap_or_else(|_| "unknown".to_string());
    let host = std::fs::read_to_string("/etc/hostname")
        .map(|h| h.trim().to_string())
        .unwrap_or_else(|_| "host".to_string());
    format!("{user}@{host}")
}

async fn runs(command: RunsCommand, config_path: &Path) -> anyhow::Result<()> {
    let config = load_config(config_path)?;
    let store = ratatoskr_store::Store::open(&config.store.path)
        .with_context(|| format!("opening the store at {}", config.store.path.display()))?;
    let log_dir = PathBuf::from(LOG_DIR);
    match command {
        RunsCommand::List {
            tags,
            status,
            origin,
            limit,
        } => {
            let mut all = store.list_runs().await?;
            store.attach_tags(&mut all).await?;
            let matching: Vec<_> = all
                .into_iter()
                .filter(|r| status.as_ref().is_none_or(|s| &r.status == s))
                .filter(|r| {
                    origin.as_ref().is_none_or(|o| match o.as_str() {
                        // The one origin that is not a name: runs this machine produced.
                        "local" => r.origin.is_none(),
                        want => r.origin.as_deref() == Some(want),
                    })
                })
                .filter(|r| tags.iter().all(|t| r.tags.contains(t)))
                .take(limit)
                .collect();

            if matching.is_empty() {
                println!("no runs match");
                return Ok(());
            }
            println!(
                "{:<10} {:<8} {:<22} {:<24} ORIGIN",
                "RUN", "ISSUE", "STATUS", "TAGS"
            );
            for r in &matching {
                println!(
                    "{:<10} {:<8} {:<22} {:<24} {}",
                    short(Some(&r.run_id)),
                    r.issue_id.as_deref().unwrap_or("—"),
                    r.status,
                    if r.tags.is_empty() {
                        "—".to_string()
                    } else {
                        r.tags.join(",")
                    },
                    r.origin.as_deref().unwrap_or("local"),
                );
            }
            Ok(())
        }

        RunsCommand::Tag { run_id, tags } => {
            let run_id = resolve(&store, &run_id).await?;
            store.tag_run(&run_id, tags.clone()).await?;
            println!("tagged {} {}", short(Some(&run_id)), tags.join(","));
            Ok(())
        }

        RunsCommand::Untag { run_id, tags } => {
            let run_id = resolve(&store, &run_id).await?;
            store.untag_run(&run_id, tags.clone()).await?;
            println!("untagged {} {}", short(Some(&run_id)), tags.join(","));
            Ok(())
        }

        RunsCommand::Abandon { run_ids } => {
            for id in &run_ids {
                let run_id = resolve(&store, id).await?;
                let status = store.run_status(&run_id).await?;
                // An unparseable status is left alone rather than overwritten: it was written by a
                // build that knows something this one does not, and abandoning it would discard
                // that.
                match status.as_deref().map(str::parse::<RunStatus>) {
                    Some(Ok(s)) if s.is_terminal() => {
                        println!("{} already finished ({s})", short(Some(&run_id)));
                    }
                    Some(Err(_)) => {
                        println!(
                            "{} has an unrecognised status ({}); left alone",
                            short(Some(&run_id)),
                            status.as_deref().unwrap_or_default()
                        );
                    }
                    _ => {
                        store
                            .upsert_run(&run_id, None, RunStatus::Abandoned.as_str())
                            .await?;
                        println!("abandoned {}", short(Some(&run_id)));
                    }
                }
            }
            Ok(())
        }

        RunsCommand::Rm { run_ids, force } => {
            let mut resolved = Vec::new();
            for id in &run_ids {
                resolved.push(resolve(&store, id).await?);
            }
            if !force {
                println!("would delete (re-run with --force):");
                for id in &resolved {
                    let events = store.events_for_run(id).await?.len();
                    let checkpoints = store.checkpoints_for_run(id).await?.len();
                    println!(
                        "  {}  {checkpoints} checkpoints, {events} events",
                        short(Some(id))
                    );
                }
                return Ok(());
            }
            for id in &resolved {
                if store.delete_run(id).await? {
                    println!("deleted {}", short(Some(id)));
                }
            }
            Ok(())
        }

        RunsCommand::Ingest { run_ids } => {
            let ids = if run_ids.is_empty() {
                store
                    .list_runs()
                    .await?
                    .into_iter()
                    .map(|r| r.run_id)
                    .collect()
            } else {
                let mut out = Vec::new();
                for id in &run_ids {
                    out.push(resolve(&store, id).await?);
                }
                out
            };
            for id in ids {
                let added = ingest_run(&store, &log_dir, &id).await?;
                if added > 0 {
                    println!("{}  +{added} events", short(Some(&id)));
                }
            }
            Ok(())
        }

        RunsCommand::Export { run_ids, out } => {
            let mut resolved = Vec::new();
            for id in &run_ids {
                resolved.push(resolve(&store, id).await?);
            }
            // Ingest first: a bundle whose events are still only in the log files would import as
            // a run nobody can look through, which is most of the reason to send one.
            for id in &resolved {
                ingest_run(&store, &log_dir, id).await?;
            }
            let at = store.now().await?;
            let bundle = store.export(&resolved, &exported_by(), &at).await?;
            let bytes = ratatoskr_store::bundle::to_bytes(&bundle)?;
            std::fs::write(&out, &bytes).with_context(|| format!("writing {}", out.display()))?;
            let events: usize = bundle.runs.iter().map(|r| r.events.len()).sum();
            println!(
                "exported {} run(s), {events} events, {} KB to {}",
                bundle.runs.len(),
                bytes.len() / 1024,
                out.display()
            );
            Ok(())
        }

        RunsCommand::Import { bundle } => {
            let bytes =
                std::fs::read(&bundle).with_context(|| format!("reading {}", bundle.display()))?;
            let read = ratatoskr_store::bundle::from_bytes(&bytes)?;
            let report = store.import(&read).await?;
            println!("from {} ({})", read.exported_by, read.exported_at);
            for r in &report {
                if r.inserted {
                    println!(
                        "  imported {}  {} checkpoints, {} events",
                        short(Some(&r.run_id)),
                        r.checkpoints,
                        r.events
                    );
                } else {
                    println!("  skipped  {}  already here", short(Some(&r.run_id)));
                }
            }
            Ok(())
        }
    }
}

/// Accept a run id prefix, the way every other tool that shows short ids does.
async fn resolve(store: &ratatoskr_store::Store, prefix: &str) -> anyhow::Result<String> {
    let matching: Vec<String> = store
        .list_runs()
        .await?
        .into_iter()
        .map(|r| r.run_id)
        .filter(|id| id.starts_with(prefix))
        .collect();
    match matching.len() {
        1 => Ok(matching.into_iter().next().expect("checked")),
        0 => bail!("no run starts with `{prefix}`"),
        n => bail!("`{prefix}` matches {n} runs; give more of the id"),
    }
}

/// Populate the project's dependency caches, so runs can check things offline.
///
/// The project is mounted **read-only** and the caches writable, which is the shape that makes the
/// result reproducible: a prepare step may fetch and unpack, and may not rewrite the lockfile it is
/// supposed to be obeying. A command that wants to — `npm install` against a `^` range rather than
/// `npm ci` — fails here rather than quietly making every later run depend on the day it ran.
async fn prepare(config_path: &Path) -> anyhow::Result<()> {
    let config = load_config(config_path)?;
    let sandbox = &config.sandbox;
    if sandbox.prepare.is_empty() {
        println!(
            "Nothing to prepare: `[sandbox] prepare` is empty. A project whose checks need no \
             dependencies beyond the image needs nothing here."
        );
        return Ok(());
    }
    let repo_root = std::env::current_dir().context("resolving the current directory")?;

    // Created before the mounts are built, because a runtime handed a source path that does not
    // exist either invents an empty directory or refuses to start — and `cache_mounts` skips a
    // cache that is not there, which would silently prepare into nothing.
    let cache_root = repo_root.join(ratatoskr_core::CACHE_ROOT);
    for cache in &sandbox.cache {
        std::fs::create_dir_all(cache_root.join(&cache.from))
            .with_context(|| format!("creating the {} cache", cache.from))?;
        // And the mountpoint, when the cache lands inside the project. The project is mounted
        // read-only here, and a runtime cannot create a mountpoint inside a read-only mount — it
        // fails with a rootfs error naming an overlay path, which says nothing about the cause.
        // A run does not need this: its worktree is writable, so the runtime makes its own.
        let at = Path::new(&cache.at);
        if at.is_relative() {
            std::fs::create_dir_all(repo_root.join(at))
                .with_context(|| format!("creating the mountpoint {}", cache.at))?;
        }
    }

    // The project read-only at the workspace path, each cache writable where a run will see it.
    let workspace = Path::new(ratatoskr_nodes::testrun::GUEST_WORKSPACE);
    let mut mounts = vec![ratatoskr_exec::Mount {
        host: repo_root.clone(),
        guest: ratatoskr_nodes::testrun::GUEST_WORKSPACE.to_string(),
        read_only: true,
    }];
    mounts.extend(
        sandbox
            .cache_mounts(&repo_root, workspace)
            .into_iter()
            .map(|(host, guest)| ratatoskr_exec::Mount {
                host,
                guest: guest.display().to_string(),
                read_only: false,
            }),
    );

    for (i, command) in sandbox.prepare.iter().enumerate() {
        if command.is_empty() {
            anyhow::bail!("`[sandbox] prepare` entry {i} is an empty command");
        }
        println!("→ {}", command.join(" "));
        let out = ratatoskr_exec::sandbox_run(ratatoskr_exec::SandboxSpec {
            backend: sandbox.backend.clone(),
            name: format!("ratatoskr-prepare-{}-{i}", std::process::id()),
            image: sandbox.image.clone(),
            workdir: ratatoskr_nodes::testrun::GUEST_WORKSPACE.to_string(),
            mounts: mounts.clone(),
            command: command.clone(),
            cpus: 2,
            memory_mib: 4096,
            // The whole reason this is a separate command rather than part of a run.
            network: true,
        })
        .await
        .with_context(|| format!("running `{}`", command.join(" ")))?;

        if !out.stdout.trim().is_empty() {
            println!("{}", out.stdout.trim());
        }
        if !out.success() {
            // A prepare that tries to write to the project hits the read-only mount, and the
            // package manager reports it as a filesystem error — `EROFS`, `rofs`, "read-only file
            // system" — which says nothing about why the filesystem is read-only. It is read-only
            // because a prepare may fetch and may not rewrite the lockfile it is obeying, and the
            // fix is the frozen form of the same command.
            let stderr = out.stderr.trim();
            let hint = match stderr.to_ascii_lowercase().contains("read-only")
                || stderr.contains("EROFS")
                || stderr.contains("rofs")
            {
                true => {
                    "\n\nThe project is mounted read-only during prepare, so a command that \
                     rewrites a lockfile fails here rather than making every later run depend on \
                     the day it ran. Use the frozen form: `npm ci`, `bun install \
                     --frozen-lockfile`, `cargo fetch --locked`, `uv sync --frozen`."
                }
                false => "",
            };
            anyhow::bail!(
                "`{}` failed (exit {}): {stderr}{hint}",
                command.join(" "),
                out.exit_code,
            );
        }
    }

    println!(
        "\nPrepared {} cache(s) under {}. Runs mount them read-only, with no network.",
        sandbox.cache.len(),
        cache_root.display()
    );
    Ok(())
}

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
    ratatoskr_agent::publish::configure_label(config.publish.label.clone());
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
async fn load_rules(
    config: &RatatoskrConfig,
) -> anyhow::Result<std::sync::Arc<ratatoskr_script::ScriptEngine>> {
    ratatoskr_nodes::validate_configured_stages(config).await?;
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
