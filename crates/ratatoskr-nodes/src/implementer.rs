//! Implementer: make edits in an isolated worktree, then run the repo's acceptance checks in a
//! sandbox against that worktree. Converge iterates this node on the same worktree with a
//! diagnostic prompt when the change introduces new failures.
//!
//! The model is driven here, with this pipeline's own tools, rather than by handing the task to a
//! coding CLI. A CLI is built around a human who is watching: it decides for itself what it may
//! run, asks when it is unsure, and reports progress to a terminal. None of that is available in a
//! run — nobody answers, and a question is a stopped node. Driving the model directly is also what
//! puts every command inside the run's own sandbox and every model turn on the run's ledger.

use std::path::PathBuf;
use std::sync::Arc;

use ratatoskr_agent::shell::ShellAccess;
use ratatoskr_core::SandboxConfig;
use ratatoskr_exec::worktree::{self, WorktreePath};
use ratatoskr_exec::{SandboxSpec, create_worktree, remove_worktree};
use ratatoskr_graph::{NodeError, parse_validated};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::analyst::AnalystOutput;
use crate::testrun::{
    Acceptance, Characterizer, GUEST_WORKSPACE, by_exit_code, mounts_for, run_acceptance,
};

/// Implementer output. Test fields are deterministic; `diff_summary`/`touched_files`/`narrative`
/// are best-effort context (relaxed) for the bookkeeper in Phase 4.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ImplementerOutput {
    pub worktree_path: String,
    /// The branch this run authored, by name.
    ///
    /// Carried explicitly because the worktree path is not it: the path ends in the id, while the
    /// branch is `ratatoskr/<id>`. The publisher was once shown the path under a `BRANCH:` label
    /// and opened a pull request against the last path segment, which the remote had never heard
    /// of.
    #[serde(default)]
    pub branch: String,
    #[serde(default)]
    pub diff_summary: String,
    #[serde(default)]
    pub touched_files: Vec<String>,
    /// Of those, the ones where a line that was already there was removed or replaced.
    ///
    /// What the referee gate reads. Adding a test is work worth having from an implementer;
    /// rewriting one is the shortcut the gate refuses, and a path on its own cannot tell them
    /// apart.
    #[serde(default)]
    pub rewritten_files: Vec<String>,
    /// What the acceptance run reported, or `None` when there was no run to report.
    ///
    /// `None` is not "nothing failed". The implementer wrote nothing — measured against the tree
    /// it was handed, so the red team's authored tests do not count as its doing — and an
    /// unchanged tree has an unchanged test result, so the suite is not run again. A zeroed result
    /// would be indistinguishable from a clean one, which is how a change that does not exist came
    /// to report convergence; the absence is the fact, and every reader has to unwrap it.
    #[serde(default)]
    pub acceptance: Option<crate::testrun::AcceptanceResult>,
    /// The CLI's own narrative of what it did (optional).
    #[serde(default)]
    pub narrative: Option<String>,
    /// How the change describes itself for the commit: type, scope, and a one-line subject.
    ///
    /// From the implementer rather than from the issue, because they are different things. The
    /// issue says what was wanted; the commit subject says what was done, and a run that fixed
    /// half of a two-part issue should not claim the whole of it in its history.
    #[serde(default)]
    pub commit_kind: String,
    #[serde(default)]
    pub commit_scope: String,
    /// The subject alone, without the `type(scope): ` prefix — that is added for you.
    ///
    /// Aim for 55 characters or fewer, so the whole line stays within git's customary 72. Nothing
    /// shortens it for you: what you write is what the history keeps, so a subject that runs long
    /// is merely untidy, and one that trails off mid-thought is wrong.
    #[serde(default)]
    pub commit_subject: String,
}

/// What a natively-driven implementer is told.
///
/// A file rather than a string constant because it is 76 lines of prose that changes for prompt
/// reasons, not code reasons — keeping it out of the source means a wording change reads as a
/// wording change in review, and the file can be diffed against a run's behaviour directly.
///
/// Written for this pipeline rather than adapted from a coding CLI's: those address an assistant
/// with a human watching, and most of what they spend words on — tone, when to explain yourself,
/// how to format a reply — is inapplicable here and costs attention. What replaces it is the part
/// no interactive agent needs: what "done" means when nobody will confirm it, that the referee is
/// off limits and why, and what to do when the session opens with a diagnostic rather than a plan.
pub const NATIVE_PREAMBLE: &str = include_str!("../prompts/implementer.md");

/// What the model reports when it stops. Everything that decides the run — the diff, the checks —
/// is read from the worktree afterwards, so this carries only the part nothing else can see: what
/// it believes it did.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct Report {
    /// What was changed and why. This becomes the commit's body, so write it for whoever reads
    /// the change later with no memory of the run: what you did, and the reasoning that is not
    /// recoverable from the diff. Not a list of files — the diff is already the list of files.
    pub(crate) summary: String,
    /// The conventional-commit type of this change: `feat`, `fix`, `chore`, `docs`, `perf`,
    /// `refactor`, `style`, `test`, `ci`, `build`.
    #[serde(default)]
    pub(crate) kind: String,
    /// The part of the repository this touches, as the log already names it — a crate, a module,
    /// a subsystem. Empty when the change belongs to no particular part.
    #[serde(default)]
    pub(crate) scope: String,
    /// One line, imperative, no trailing period, under 60 characters: what the commit does. Not a
    /// restatement of the issue's title — what *this change* does.
    #[serde(default)]
    pub(crate) subject: String,
}

/// Deterministic input for one model attempt. Rust decides whether this is the initial plan or a
/// correction and TypeScript only renders that already-decided input into the user question.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(crate) struct ImplementerAttemptInput {
    pub issue: String,
    pub analyst: AnalystOutput,
    pub acceptance: Vec<ratatoskr_core::AcceptanceStep>,
    #[serde(default)]
    pub diagnostic: Option<String>,
}

/// The implementer node. Holds everything needed to create the worktree and run the acceptance
/// checks against it.
///
/// What the model turn runs on is deliberately absent: every attempt goes through the stage
/// executor, which resolves `implementer_attempt`'s route, tools, capability ceiling, turn cap and
/// prompt from the run's registry. A workflow may override that stage, so a copy resolved here
/// could only be the wrong answer held next to the right one.
pub struct ImplementerNode {
    pub repo_path: PathBuf,
    pub worktree_root: PathBuf,
    pub sandbox: SandboxConfig,
    /// Who answers when the implementer cannot resolve something itself.
    ///
    /// It is the node with the most turns to spend and the only one that changes code, so it is
    /// also the one most likely to hit a question worth asking — a plan that contradicts the tree,
    /// or its own tool results looking wrong. Without this its only options were to proceed on a
    /// belief nobody checked or to stop. Bounded by the run-wide `ASK_BUDGET`, which it shares with
    /// every other asker, so a stuck node cannot spend the run asking.
    pub clarifier: Option<Arc<dyn ratatoskr_agent::Clarifier>>,
    pub run_id: String,
    pub issue: String,
    pub analyst: AnalystOutput,
    /// What "done" means for this task. Frozen at plan time: a change must not be able to move the
    /// bar it is judged against, so an analyst revision amends requirements and never this.
    pub acceptance: Vec<ratatoskr_core::AcceptanceStep>,
    /// Names the checks inside each step. `None` compares at step granularity instead.
    pub characterizer: Option<Characterizer>,
    /// The generic stage executor context used for every implementation attempt.
    pub(crate) declared_context: Arc<crate::workflow::WorkflowContext>,
}

impl ImplementerNode {
    /// Create the worktree and make the first attempt. Returns the worktree (for iteration and
    /// cleanup) alongside the output.
    pub async fn run(&self) -> Result<(WorktreePath, ImplementerOutput), NodeError> {
        let worktree = self.prepare().await?;
        // If the first attempt fails, remove the worktree so a failed run leaves nothing behind.
        match self.work(&worktree).await {
            Ok(out) => Ok((worktree, out)),
            Err(e) => {
                self.discard(&worktree).await;
                Err(e)
            }
        }
    }

    /// Create the branch and worktree this run will work in.
    ///
    /// Separate from doing the work because the worktree is not only the implementer's: the red
    /// team writes its tests into the same tree before any code exists to satisfy them, and it
    /// cannot do that until the tree is there.
    pub async fn prepare(&self) -> Result<WorktreePath, NodeError> {
        create_worktree(&self.repo_path, &self.worktree_root, &self.branch())
            .await
            .map_err(|e| NodeError::Failed(format!("worktree create failed: {e}")))
    }

    /// The first attempt, in a worktree that already exists.
    pub async fn work(&self, worktree: &WorktreePath) -> Result<ImplementerOutput, NodeError> {
        self.attempt(worktree, self.initial_input()).await
    }

    /// Remove a worktree whose run is not going to finish. Best-effort: a leftover tree is a
    /// nuisance, and failing the run over one would trade a nuisance for a loss.
    pub async fn discard(&self, worktree: &WorktreePath) {
        if let Err(rm) = remove_worktree(&self.repo_path, worktree).await {
            tracing::warn!("failed to clean up worktree after implementer error: {rm}");
        }
    }

    /// Re-drive the CLI on the same worktree with a diagnostic prompt (converge iteration).
    pub async fn iterate(
        &self,
        worktree: &WorktreePath,
        diagnostic: &str,
    ) -> Result<ImplementerOutput, NodeError> {
        self.attempt(
            worktree,
            ImplementerAttemptInput {
                issue: self.issue.clone(),
                analyst: self.analyst.clone(),
                acceptance: self.acceptance.clone(),
                diagnostic: Some(diagnostic.to_string()),
            },
        )
        .await
    }

    /// One attempt: drive the model, then run the worktree's acceptance checks and read its diff.
    async fn attempt(
        &self,
        worktree: &WorktreePath,
        input: ImplementerAttemptInput,
    ) -> Result<ImplementerOutput, NodeError> {
        let input_json = serde_json::to_string(&input)
            .map_err(|error| NodeError::Failed(format!("implementer input: {error}")))?;
        // Captured before the turn, and only once: on the first attempt this is the tree with the
        // red team's authored tests already in it, which is what the implementer was handed. Later
        // attempts keep comparing against that same tree, so "did the implementer change anything"
        // stays a question about the run rather than about the last iteration.
        // Held on the run's context, not on this node: `iterate` and `replanAtCeiling` each build
        // a fresh `ImplementerNode`, and a snapshot retaken per node would be of a tree already
        // carrying an earlier attempt's work — so a last iteration that added nothing would report
        // the whole run as having produced none, and the change in the worktree would go
        // unpublished.
        let handed = self
            .declared_context
            .handed
            .get_or_init(|| async { worktree::full_diff_text(worktree).await.ok() })
            .await
            .clone();
        let raw = crate::workflow::evaluate_standard_stage_with_resources(
            Arc::clone(&self.declared_context),
            "implementer_attempt",
            input_json,
            crate::workflow::StandardStageResources {
                resource_root: worktree.as_path().to_path_buf(),
                capability_ceiling: ratatoskr_core::Capability::Write,
                rag_rat_worktree: Some(worktree.as_path().to_path_buf()),
                shell: Some(self.shell_access(worktree)),
                publish: None,
                clarifier: self.clarifier.clone(),
                guidance: Some(where_you_are()),
            },
        )
        .await
        .map_err(|error| NodeError::Failed(format!("implementer agent failed: {error}")))?;
        let report = parse_validated::<Report>(&raw)?;

        // An unchanged tree has an unchanged test result. The baseline already ran this suite, so
        // running it again to compare a set against itself is the most expensive way to learn
        // nothing — a full sandboxed run, twice, to find that 53 passing tests still pass.
        //
        // Compared against what the implementer was handed, never against an empty tree: the red
        // team's authored tests are already in this worktree, so a tree that merely has files in
        // it says nothing about whether the implementer wrote any of them.
        //
        // Either diff failing to read leaves this CHANGED. A git that cannot answer must not be
        // able to skip the suite and report a change nobody checked.
        let now = worktree::full_diff_text(worktree).await.ok();
        let produced_change = match (&handed, &now) {
            (Some(handed), Some(now)) => handed != now,
            _ => true,
        };
        let acceptance = if !produced_change {
            tracing::info!(
                kind = "acceptance_skipped_unchanged_tree",
                "the implementer wrote nothing, so the baseline's result still stands; not \
                 running the acceptance suite again"
            );
            // Absent, never zeroed: there is no result, and a reader must not be able to mistake
            // one for a clean run.
            None
        } else {
            let outcomes = run_acceptance(Acceptance {
                cfg: &self.sandbox,
                node: "implementer",
                name: &format!("ratatoskr-impl-{}", self.short_id()),
                repo_root: &self.repo_path,
                worktree: worktree.as_path(),
                steps: &self.acceptance,
            })
            .await
            .map_err(NodeError::Failed)?;
            Some(
                match &self.characterizer {
                    Some(c) => c.read(&outcomes).await,
                    None => by_exit_code(&outcomes),
                }
                .into(),
            )
        };

        // Diff/touched reads are best-effort — don't fail the attempt if they hiccup.
        let diff_summary = worktree::diff_stat(worktree).await.unwrap_or_default();
        let touched_files = worktree::touched_files(worktree).await.unwrap_or_default();
        let rewritten_files = worktree::rewritten_files(worktree)
            .await
            .unwrap_or_default();

        Ok(ImplementerOutput {
            acceptance,
            worktree_path: worktree.as_path().display().to_string(),
            branch: self.branch(),
            diff_summary,
            touched_files,
            rewritten_files,
            commit_kind: report.kind,
            commit_scope: report.scope,
            commit_subject: report.subject,
            narrative: Some(report.summary),
        })
    }

    /// The sandbox this attempt's `Bash` calls run in: the same one its acceptance checks run in,
    /// so a command that passes for the model passes for the run. Anything else and the model
    /// would be debugging a different machine from the one that judges it.
    fn shell_access(&self, worktree: &WorktreePath) -> ShellAccess {
        ShellAccess {
            spec: SandboxSpec {
                backend: self.sandbox.backend.clone(),
                name: format!("ratatoskr-impl-{}", self.short_id()),
                image: self.sandbox.image.clone(),
                workdir: GUEST_WORKSPACE.to_string(),
                // The worktree it changes, plus whatever `prepare` left in the project's caches.
                // The same set the acceptance check gets, so a command the implementer runs by
                // hand behaves as it will when the check runs it.
                mounts: mounts_for(&self.sandbox, &self.repo_path, worktree.as_path()),
                // Filled in per call.
                command: Vec::new(),
                cpus: self.sandbox.cpus,
                memory_mib: self.sandbox.memory_mib,
                network: false,
            },
        }
    }

    fn short_id(&self) -> String {
        self.run_id.chars().take(8).collect()
    }

    /// The branch this run works on. One definition, because the name is used to create the
    /// worktree, to push, and to open a pull request, and three spellings of it is two bugs.
    pub fn branch(&self) -> String {
        format!("ratatoskr/{}", self.short_id())
    }

    fn initial_input(&self) -> ImplementerAttemptInput {
        ImplementerAttemptInput {
            issue: self.issue.clone(),
            analyst: self.analyst.clone(),
            acceptance: self.acceptance.clone(),
            diagnostic: None,
        }
    }
}

/// Where the model is working, appended to its preamble.
///
/// On the preamble rather than in the task prompt because a re-driven attempt is a fresh
/// conversation that receives only a diagnostic — it would otherwise have to rediscover this every
/// iteration. Told rather than left to be found: a node that has to run `git rev-parse` and `ls`
/// before it can start has spent two turns learning something nobody had a reason to withhold.
fn where_you_are() -> String {
    "\n\n# WHERE YOU ARE\n\nYour tools already start in this run's worktree, which is already \
     checked out on your own branch. Use relative paths with Read, Edit, and Bash; relative paths \
     resolve against the worktree. Reading or writing outside the worktree is refused. It is a full \
     copy of the repository — the change you are asked for is made here, and nowhere else."
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_git(repo: &std::path::Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn every_attempt_is_told_which_worktree_it_is_in() {
        // Each attempt is a fresh conversation — a re-driven one receives only a diagnostic — so
        // this rides on the preamble rather than the task prompt. Left out, the first thing a run
        // does is spend turns on `git rev-parse` and `ls` to find out where it woke up.
        let told = where_you_are();
        assert!(!told.contains("/w/ratatoskr/abc12345"), "{told}");
        assert!(
            told.contains("relative paths with Read, Edit, and Bash"),
            "{told}"
        );
        assert!(told.contains("branch"), "{told}");
    }

    #[test]
    fn where_you_are_states_the_backend_neutral_worktree_contract() {
        // Issue #212: the guidance block takes no WorktreePath — the signature is the contract,
        // because a parameter is an invitation to render it. Asserted on the facts rather than the
        // wording, whitespace-collapsed so a reflowed paragraph is not a failing test.
        let told = where_you_are()
            .to_ascii_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        // The tools start in the run's worktree and relative paths resolve there.
        assert!(told.contains("worktree"), "{told}");
        assert!(told.contains("relative"), "{told}");
        // The implementer is told to use relative paths with each tool it would otherwise
        // hand an absolute path to.
        assert!(told.contains("read"), "{told}");
        assert!(told.contains("edit"), "{told}");
        assert!(told.contains("bash"), "{told}");
        // The invariants the run actually enforces are still stated: its own already-checked-out
        // branch, and access outside the worktree refused.
        assert!(told.contains("branch"), "{told}");
        assert!(told.contains("outside"), "{told}");
        assert!(told.contains("refused"), "{told}");
    }

    #[test]
    fn where_you_are_renders_no_host_absolute_path() {
        // Regression for #212: the preamble used to render the host-side WorktreePath, which is
        // not a valid path for every sandbox backend (the container backend mounts the worktree
        // at a guest path; bwrap binds it in place). The block takes no worktree argument now, so
        // the only way this fails is a host path being reintroduced by name.
        let told = where_you_are();
        assert!(
            !told.contains("/w/ratatoskr/abc12345"),
            "the implementer preamble must not contain a host absolute worktree path: {told}"
        );
        // No absolute path of any flavour: the guidance is relative-paths-only, so a `/`-rooted
        // token has no legitimate reason to appear.
        assert!(
            !told
                .split_whitespace()
                .any(|word| word.starts_with('/') && word.len() > 1),
            "the implementer preamble must not name an absolute path: {told}"
        );
    }

    #[test]
    fn where_you_are_is_identical_regardless_of_sandbox_backend() {
        // The block must hold for both the container and bwrap backends: it takes no backend
        // input, so the strongest checkable form of the contract is that the output names no
        // backend and carries no backend-conditional text.
        let told = where_you_are().to_ascii_lowercase();
        assert!(!told.contains("container"), "{told}");
        assert!(!told.contains("bwrap"), "{told}");
        assert!(!told.contains("microsandbox"), "{told}");
    }

    #[tokio::test]
    async fn a_failed_native_stage_attempt_removes_its_worktree() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-implementer-cleanup-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let repo = dir.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        run_git(&repo, &["init", "-q", "-b", "main"]);
        run_git(&repo, &["config", "user.email", "test@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Test"]);
        run_git(&repo, &["commit", "--allow-empty", "-qm", "initial"]);
        let engine = ratatoskr_script::ScriptEngine::load(&dir).await.unwrap();
        let store = ratatoskr_store::Store::open_in_memory().unwrap();
        let run_id = format!("{:08x}-cleanup", std::process::id());
        store.upsert_run(&run_id, None, "running").await.unwrap();
        let mut config = ratatoskr_core::RatatoskrConfig::default();
        config.models.get_mut("implementer").unwrap().provider = "not-a-provider".to_string();
        let context = crate::workflow::WorkflowContext::new(
            None,
            &config,
            &store,
            &run_id,
            "exercise cleanup",
            &engine,
            crate::PluginContext::default(),
        )
        .unwrap();
        let node = ImplementerNode {
            repo_path: repo.clone(),
            worktree_root: dir.join("worktrees"),
            sandbox: config.sandbox.clone(),
            clarifier: None,
            run_id,
            issue: "exercise cleanup".to_string(),
            analyst: AnalystOutput {
                impact_summary: "The model turn will fail before editing.".to_string(),
                touched: Vec::new(),
                risks: Vec::new(),
                requirements: Vec::new(),
                residual_risk: String::new(),
                changes_code: true,
                acceptance: Vec::new(),
                interface: Vec::new(),
            },
            acceptance: Vec::new(),
            characterizer: None,
            declared_context: context,
        };
        let branch = node.branch();

        let error = node.run().await.unwrap_err().to_string();
        assert!(error.contains("unknown provider"), "{error}");
        assert!(
            !node.worktree_root.join(&branch).exists(),
            "failed attempts must not leave a linked worktree"
        );
        run_git(&repo, &["branch", "-D", &branch]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_native_preamble_states_the_rules_a_run_actually_enforces() {
        // The prompt lives in a separate file, so nothing in the type system ties it to what the
        // run actually does. Asserted on the facts rather than the wording: the prompt is meant to
        // read like a briefing and gets rewritten, and a test keyed to its phrasing would either
        // fail on every edit or quietly stop checking anything.
        // Whitespace-collapsed, so a reflowed paragraph is not a failing test.
        let p = NATIVE_PREAMBLE
            .to_ascii_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        // Rewriting a test comes back; adding one does not. Both halves matter — a prompt that
        // says only the first leaves an implementer writing untested code to stay safe.
        assert!(p.contains("rewrote") || p.contains("rewrite"), "the gate");
        assert!(
            p.contains("adding one is never held against you"),
            "and its limit"
        );
        assert!(
            p.contains("maymodifytests"),
            "how a real exemption is declared"
        );
        assert!(p.contains("stopping is a claim"), "when to stop");
        assert!(p.contains("exactly once"), "Edit's uniqueness contract");
        assert!(p.contains("diagnostic"), "the re-driven path");
        assert!(p.contains("git status --short"), "recovering prior edits");
        assert!(p.contains("git diff --cached"), "recovering staged edits");
        assert!(
            p.contains("git mv") && p.contains("git rm"),
            "tracked file moves and removals"
        );
        // It can ask, and is told what is worth asking about. The failure this guards against is a
        // prompt that says a question produces nothing while the node holds an `ask` tool — a node
        // told not to ask will not ask, however wired it is, and the one case where asking clearly
        // beats guessing is when its own tool results look fabricated.
        assert!(p.contains("you do have `ask`"), "that it can");
        assert!(p.contains("budget"), "and that asking is not free");
        assert!(
            p.contains("fabricated tool result"),
            "the case where it plainly beats re-deriving the tree"
        );
        assert!(
            p.contains("not for permission"),
            "and what does not earn one"
        );
    }

    #[test]
    fn conventions_are_loaded_when_present_and_reach_only_the_writing_nodes() {
        let dir = std::env::temp_dir().join(format!("ratatoskr-conv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // No conventions file: the loader is None and a writing node's preamble is byte-identical
        // to the plain composition — no header, no separator.
        assert_eq!(crate::repo_conventions(&dir), None);
        let plain = crate::effective_preamble("implementer", NATIVE_PREAMBLE, None, None, &[]);
        assert_eq!(
            crate::with_conventions("implementer", None, plain.clone()),
            plain,
            "no conventions must leave the preamble untouched"
        );

        // Present: bounded Some, and a writing node carries it while a non-writing node does not.
        std::fs::write(
            dir.join("AGENTS.md"),
            "# House rules\nUse parameter structs.\n",
        )
        .unwrap();
        let conv = crate::repo_conventions(&dir).expect("AGENTS.md is loaded");
        assert!(conv.contains("Use parameter structs"));
        assert!(!conv.is_empty() && conv.len() <= crate::CONVENTIONS_BUDGET);

        let writer = crate::with_conventions("implementer", Some(&conv), plain.clone());
        assert!(writer.contains("Use parameter structs"), "writer gets them");
        // A non-writing node composes with effective_preamble and never sees with_conventions.
        assert!(
            !plain.contains("Use parameter structs"),
            "non-writing preamble stays free of conventions"
        );

        // CLAUDE.md is honoured when AGENTS.md is absent (the ecosystem's alternative name).
        std::fs::remove_file(dir.join("AGENTS.md")).unwrap();
        std::fs::write(dir.join("CLAUDE.md"), "from claude\n").unwrap();
        assert_eq!(
            crate::repo_conventions(&dir).as_deref(),
            Some("from claude\n")
        );

        // Over budget: clipped to the bound, not silently past it.
        let big = "x".repeat(crate::CONVENTIONS_BUDGET * 2);
        std::fs::write(dir.join("AGENTS.md"), &big).unwrap();
        let clipped = crate::repo_conventions(&dir).unwrap();
        assert_eq!(clipped.len(), crate::CONVENTIONS_BUDGET);

        std::fs::remove_dir_all(&dir).ok();
    }
}
