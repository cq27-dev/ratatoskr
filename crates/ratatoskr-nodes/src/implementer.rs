//! Implementer: make edits in an isolated worktree, then run the repo's acceptance checks in a
//! sandbox against that worktree. Converge iterates this node on the same worktree with a
//! diagnostic prompt when the change introduces new failures.
//!
//! The model is driven here, with this pipeline's own tools, rather than by handing the task to a
//! coding CLI. A CLI is built around a human who is watching: it decides for itself what it may
//! run, asks when it is unsure, and reports progress to a terminal. None of that is available in a
//! run — nobody answers, and a question is a stopped node. Driving the model directly is also what
//! puts every command inside the run's own sandbox and every model turn on the run's ledger.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use ratatoskr_agent::RunLedger;
use ratatoskr_agent::shell::ShellAccess;
use ratatoskr_core::{ModelRoute, SandboxConfig, ToolPolicy};
use ratatoskr_exec::worktree::{self, WorktreePath};
use ratatoskr_exec::{Mount, SandboxSpec, create_worktree, remove_worktree};
use ratatoskr_graph::{NodeError, parse_validated};
use ratatoskr_mcp::ToolSet;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::analyst::AnalystOutput;
use crate::testrun::{Characterizer, GUEST_WORKSPACE, by_exit_code, run_acceptance};

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
    pub failing_tests: Vec<String>,
    /// How many checks passed. Only the count is carried: nothing reads a passing check's name,
    /// and a suite of several hundred costs the characterizer more output than the rest of the
    /// pipeline combined to write out.
    #[serde(default)]
    pub passed_tests: usize,
    pub exit_code: i32,
    /// The CLI's own narrative of what it did (optional).
    #[serde(default)]
    pub narrative: Option<String>,
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

/// The rag-rat tools the implementer gets by default.
///
/// Wider than a planning node's: it is the only node that changes code, and the prompt tells it to
/// check the blast radius and the recorded memories of every symbol it touches — which it can only
/// do if it can reach them.
pub const IMPLEMENTER_TOOLS: &[&str] = &[
    "impact_surface",
    "symbol_lookup",
    "semantic_search",
    "find_callers",
    "memory_search",
    "read_chunk",
];

/// What the model reports when it stops. Everything that decides the run — the diff, the checks —
/// is read from the worktree afterwards, so this carries only the part nothing else can see: what
/// it believes it did.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Report {
    /// What was changed and why, for the reader of the run rather than for another node.
    summary: String,
}

/// The implementer node. Holds everything needed to create the worktree and drive the model.
pub struct ImplementerNode {
    pub repo_path: PathBuf,
    pub worktree_root: PathBuf,
    pub sandbox: SandboxConfig,
    pub route: ModelRoute,
    pub tools: ToolSet,
    pub policy: Option<Arc<dyn ToolPolicy>>,
    pub max_turns: Option<usize>,
    /// Ruleset `systemPrompt`; replaces [`NATIVE_PREAMBLE`] when set.
    pub system_prompt: Option<String>,
    pub plugins: crate::NodePlugins,
    pub ledger: Option<Arc<RunLedger>>,
    pub run_id: String,
    pub issue: String,
    pub analyst: AnalystOutput,
    /// What "done" means for this task. Frozen at plan time: a change must not be able to move the
    /// bar it is judged against, so an analyst revision amends requirements and never this.
    pub acceptance: Vec<ratatoskr_core::AcceptanceStep>,
    /// Names the checks inside each step. `None` compares at step granularity instead.
    pub characterizer: Option<Characterizer>,
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
        self.attempt(worktree, &self.initial_prompt()).await
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
        self.attempt(worktree, diagnostic).await
    }

    /// One attempt: drive the model, then run the worktree's acceptance checks and read its diff.
    async fn attempt(
        &self,
        worktree: &WorktreePath,
        prompt: &str,
    ) -> Result<ImplementerOutput, NodeError> {
        // One conversation for this node across the run, so a converge iteration continues the
        // attempt it is fixing instead of meeting the worktree for the first time again.
        let conversation = format!("{}-implementer", self.run_id);
        let raw = ratatoskr_agent::run_structured(ratatoskr_agent::NodeRun {
            node: "implementer",
            route: &self.route,
            preamble: &format!(
                "{}{}",
                crate::effective_preamble(
                    NATIVE_PREAMBLE,
                    self.system_prompt.as_deref(),
                    self.plugins.context.as_deref(),
                ),
                where_you_are(worktree),
            ),
            question: prompt,
            tools: self.tools.clone(),
            output_schema: schemars::schema_for!(Report),
            policy: self.policy.clone(),
            max_turns: self.max_turns,
            // Nobody to ask. The prompt says so, and offering the tool would contradict it.
            clarifier: None,
            observer: self.plugins.observer.clone(),
            skills: crate::skills::loaded(&self.plugins.skills),
            // Rooted at the worktree, not the checkout: every path it reads or writes has to be
            // the copy it owns, or one attempt edits the tree another node is reading.
            files: Some(worktree.as_path().to_path_buf()),
            shell: Some(self.shell_access(worktree)),
            push: None,
            conversation: Some(&conversation),
            ledger: self.ledger.clone(),
            produces: Some("a change that satisfies the plan and passes the acceptance checks"),
        })
        .await
        .map_err(|e| NodeError::Failed(format!("implementer agent failed: {e}")))?;
        let report = parse_validated::<Report>(&raw)?;

        let outcomes = run_acceptance(
            &self.sandbox,
            "implementer",
            &format!("ratatoskr-impl-{}", self.short_id()),
            worktree.as_path(),
            &self.acceptance,
        )
        .await
        .map_err(NodeError::Failed)?;
        let tests = match &self.characterizer {
            Some(c) => c.read(&outcomes).await,
            None => by_exit_code(&outcomes),
        };

        // Diff/touched reads are best-effort — don't fail the attempt if they hiccup.
        let diff_summary = worktree::diff_stat(worktree).await.unwrap_or_default();
        let touched_files = worktree::touched_files(worktree).await.unwrap_or_default();
        let rewritten_files = worktree::rewritten_files(worktree)
            .await
            .unwrap_or_default();

        Ok(ImplementerOutput {
            worktree_path: worktree.as_path().display().to_string(),
            branch: self.branch(),
            diff_summary,
            touched_files,
            rewritten_files,
            failing_tests: tests.failing,
            passed_tests: tests.passed,
            exit_code: tests.exit_code,
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
                mounts: vec![Mount {
                    host: worktree.as_path().to_path_buf(),
                    guest: GUEST_WORKSPACE.to_string(),
                }],
                // Filled in per call.
                command: Vec::new(),
                cpus: 2,
                memory_mib: 2048,
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

    /// The initial prompt: the issue plus the analyst's requirements and risks.
    fn initial_prompt(&self) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "Implement this task in the current repository:\n\n{}\n\n",
            self.issue
        );
        let a = &self.analyst;
        if !a.impact_summary.is_empty() {
            let _ = write!(s, "Impact analysis:\n{}\n\n", a.impact_summary);
        }
        if !a.requirements.is_empty() {
            s.push_str("Requirements the implementation must satisfy:\n");
            for req in &a.requirements {
                let _ = writeln!(s, "- {req}");
            }
            s.push('\n');
        }
        if !a.interface.is_empty() {
            s.push_str(
                "THE INTERFACE YOU ARE BUILDING TO. Someone else is writing tests against this \
                 same description, without seeing your code. Match the shape exactly — a \
                 signature that differs by a parameter name or an argument order will fail tests \
                 that are not wrong:\n\n",
            );
            crate::analyst::render_interface(&mut s, &a.interface, "must", "must also");
            s.push('\n');
        }
        if !a.risks.is_empty() {
            s.push_str("Known risks to avoid:\n");
            for risk in &a.risks {
                let _ = writeln!(s, "- {risk}");
            }
            s.push('\n');
        }
        if !self.acceptance.is_empty() {
            s.push_str(
                "The acceptance checks this change is judged by, which you can run yourself with \
                 Bash:\n",
            );
            for step in &self.acceptance {
                let _ = writeln!(s, "- {}: `{}`", step.name, step.command.join(" "));
            }
            s.push('\n');
        }
        s.push_str(
            "Apply the change directly with your editing tools — do NOT ask for confirmation or \
             present options to choose between; just make the fix. Then run the acceptance checks \
             and make them pass.",
        );
        s
    }
}

/// Where the model is working, appended to its preamble.
///
/// On the preamble rather than in the task prompt because a re-driven attempt is a fresh
/// conversation that receives only a diagnostic — it would otherwise have to rediscover this every
/// iteration. Told rather than left to be found: a node that has to run `git rev-parse` and `ls`
/// before it can start has spent two turns learning something nobody had a reason to withhold.
fn where_you_are(worktree: &WorktreePath) -> String {
    format!(
        "\n\n# WHERE YOU ARE\n\nYour worktree is `{}`, and it is already checked out on your own \
         branch. Your tools start there: a relative path is resolved against it, Bash starts there, \
         and reading or writing outside it is refused. It is a full copy of the repository — the \
         change you are asked for is made here, and nowhere else.",
        worktree.as_path().display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_attempt_is_told_which_worktree_it_is_in() {
        // Each attempt is a fresh conversation — a re-driven one receives only a diagnostic — so
        // this rides on the preamble rather than the task prompt. Left out, the first thing a run
        // does is spend turns on `git rev-parse` and `ls` to find out where it woke up.
        let told = where_you_are(&WorktreePath(PathBuf::from("/w/ratatoskr/abc12345")));
        assert!(told.contains("/w/ratatoskr/abc12345"), "{told}");
        assert!(told.contains("branch"), "{told}");
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
        // It is told there is nobody to ask, which is the fact every interactive prompt assumes
        // the opposite of.
        assert!(p.contains("nobody to ask"));
    }
}
