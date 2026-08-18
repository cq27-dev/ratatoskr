// Reference `.ratatoskr/workflow.ts` — follows Ratatoskr's bundled standard-v1 topology and adds
// a declared requirements stage. Copy it to `.ratatoskr/workflow.ts` and edit to customize how a
// run is sequenced. The `requirements` agent it uses is declared right here in `agents:`; the
// optional `examples/agent-profiles.toml` shows how deployment routes that profile to a model.
//
// A workflow is an ES module. Its entries are the functions it **exports** — `plan` and `run`
// below — so a declaration without `export` is module-scoped and the run fails saying so.
// `import` resolves only from what the host offers (`ratatoskr/nodes`), never from the filesystem,
// and only from a string literal; the prelude (`defineWorkflow`, `stage`, `str`/`obj`/…) and the
// node bindings are globals, which module scope reads normally. Top-level `this` is `undefined`.
//
// The module only *composes* the node bindings. Every gate stays Rust-enforced no matter what the
// script does: `context()` owns deterministic evidence collection; `redTeam()` owns its fresh
// baseline and test-author worktree; `implement()`/`iterate()` own the sandbox and acceptance;
// `verify()` owns diff collection and review thresholds; and terminal status and effects are
// reconstructed from Rust checkpoints, never from anything the script returns.
//
// Relevant bindings. EVERY one is async and must be awaited — including the two that answer a
// plain boolean. An unawaited call is a Promise, and a Promise is truthy however it resolves, so
// `testCommandRan(x) && isConverged(y)` is always true. Await each call before a boolean, a
// ternary, an `if`, a `!`, a `===`, or a template string reads it:
//   context(issue) -> ContextOutput
//   analyst({issue, scout, memory, brief?, constraints?, previous?, findings?}) -> AnalystOutput
//   redTeam() -> RedTeamOutput                     // prepares the tree; baseline + authored tests
//   implement({analyst}) -> ImplementerOutput      // edits the prepared worktree (once)
//   iterate({ review? }) -> ImplementerOutput       // applies Rust-derived test/review correction
//   verify({analyst}) -> VerifyResult              // reviews the diff against the plan
//   replanAtCeiling() -> null | {analyst, implementation}
//                                                    // one Rust-authorized final recovery at most
//   isConverged({baseline, post}) -> boolean
//   testCommandRan(output) -> boolean
//
// `verify()` returns { configured, unavailable, findings, blocking, needsReplan, unchecked,
// retryable }. Rust applies
// `[implementer] verify_threshold` — a script decides *whether* to review and what to do about
// findings, never what counts as blocking. `needsReplan` means a blocking finding faults the PLAN,
// so the useful response is `analyst({...., previous, findings})` before
// `iterate({ review })`, rather than
// re-driving the implementer at a requirement already shown to be wrong.
//
// A run that calls verify() and returns with blocking findings standing does NOT converge: the
// terminal status is inferred from the verifier checkpoint, not from what the script returns.
//
// Nor does one that returns on a review which could not finish. `unchecked` names what a pass could
// not reach, and `retryable` says calling verify() again would continue that review rather than
// repeat it — Rust carries the named areas into the next call and bounds how many continuations a
// tree may have. A script that breaks on `blocking.length === 0` alone accepts a review cut short,
// and the run ends `Unreviewed` after one pass instead of covering the gap.
//
// `replanAtCeiling()` takes no workflow-supplied plan or evidence. Rust reconstructs both from the
// checkpoint ledger and either performs one bounded analyst revision plus implementation attempt,
// or returns null. A script cannot replay it into an unbounded extra loop.
//
// A workflow that introduces a stage of its own declares it, so its `.ratatoskr/rules/<node>.ts` is
// accepted rather than read as a typo:
//
//   defineWorkflow({ name: "deep", nodes: ["reviewer2"] });

defineWorkflow({
  name: "standard",
  purpose: "Plan and implement a repository change with an explicit requirements digest.",
  whenToUse: ["the task requests a code change"],
  // The workflow owns its agents' structure: prompt, capability ceiling, turn cap. Deployment's
  // `ratatoskr.toml` may route a declared agent to a model, and nothing else.
  agents: {
    requirements: {
      basePrompt: "Extract the task's non-negotiable requirements before implementation planning.",
      capabilities: ["read"],
      maxTurns: 24,
    },
  },
  stages: [
    stage("requirements", {
      agent: "requirements",
      inputContract: "{ issue: string }",
      outputContract: "RequirementsDigest",
      outputSchema: obj(
        {
          summary: str(),
          risks: arr(str()),
        },
        ["summary", "risks"],
        { additionalProperties: false },
      ),
      // A longer prompt can live beside the workflow and be compiled in with
      // `LOAD("requirements.md").trim()`. LOAD accepts one literal relative path, never runs at
      // runtime, and cannot leave the workflow directory.
      instructions:
        "Extract the requirements and delivery risks that must shape the implementation plan. " +
        "Do not propose code. Return only the declared JSON object.",
      // Optional and synchronous: format structured input for the model without changing what the
      // Rust host checkpoints. The function must be self-contained and return a string.
      renderQuestion(input: { issue: string }) {
        return `ISSUE TO DISTIL:\n${input.issue}`;
      },
      capabilities: ["read"],
      appendRepositoryGuidance: true,
    }),
  ],
});

function analystInput(
  input: { issue: string },
  gathered: any,
  requirementsOut: { summary: string; risks: string[] },
  previous?: any,
  findings?: any[],
) {
  const requirementsBrief =
    `EXPLICIT REQUIREMENTS:\n${requirementsOut.summary}\n` +
    requirementsOut.risks.map((risk) => `Risk: ${risk}`).join("\n");
  return {
    issue: input.issue,
    scout: gathered.scout,
    memory: gathered.memory,
    brief: [gathered.brief, requirementsBrief].filter(Boolean).join("\n\n"),
    constraints: gathered.constraints,
    previous,
    findings,
  };
}

async function gatherPlan(input: { issue: string }) {
  const gathered = await context(input.issue);
  const requirementsOut = await requirements({ issue: input.issue });
  const analysis = await analyst(analystInput(input, gathered, requirementsOut));
  return { gathered, requirementsOut, analysis };
}

// `plan`: context -> requirements -> analyst. Rust reconstructs the PlanOutcome from checkpoints.
export async function plan(input: { issue: string }) {
  const { gathered, requirementsOut, analysis } = await gatherPlan(input);
  return { context: gathered, requirements: requirementsOut, analyst: analysis };
}

// `run`: plan, optional fork, sequential red-team -> implementation, then bounded convergence.
export async function run(input: { issue: string; maxIterations: number; alwaysFork: boolean }) {
  const planned = await gatherPlan(input);
  let analystOut = planned.analysis;

  // Only an explicit no-code plan may skip the fork; alwaysFork can only add work.
  if (analystOut.changes_code === false && !input.alwaysFork) {
    return { context: planned.gathered, analyst: analystOut, iterations: 0 };
  }

  // Both operations use the prepared worktree. Authoring completes against the frozen interface
  // before implementation starts, so the author cannot tailor tests to code that already exists.
  const redTeamOut = await redTeam();
  let impl = await implement({ analyst: analystOut });
  let iterations = 1;

  async function recoverAtCeiling() {
    const recovery = await replanAtCeiling();
    if (recovery === null) return false;
    analystOut = recovery.analyst;
    impl = recovery.implementation;
    iterations += 1;
    return true;
  }

  // The loop mirrors standard-v1, but Rust owns every decision that grants another model attempt.
  while (true) {
    // Both hosts are `async function`s, so each call must be awaited before `&&` reads it — an
    // unawaited call is a Promise, which is truthy no matter what it resolves to.
    const testsClean =
      (await testCommandRan(impl)) &&
      (await isConverged({ baseline: redTeamOut, post: impl }));
    if (testsClean) {
      const review = await verify({ analyst: analystOut });
      if (!review.configured || review.unavailable) break;
      // Continue a review that could not finish before deciding anything about it: it returns
      // exactly what a clean one returns, so accepting it here is accepting an unreviewed change.
      if (review.retryable) continue;
      if (review.blocking.length === 0) break;
      if (iterations >= input.maxIterations) {
        if (await recoverAtCeiling()) continue;
        break;
      }
      if (review.needsReplan) {
        analystOut = await analyst(
          analystInput(
            input,
            planned.gathered,
            planned.requirementsOut,
            analystOut,
            review.blocking,
          ),
        );
      }
      impl = await iterate({ review });
      iterations += 1;
      continue;
    }
    if (iterations >= input.maxIterations) {
      if (await recoverAtCeiling()) continue;
      break;
    }
    impl = await iterate({});
    iterations += 1;
  }
  return { context: planned.gathered, analyst: analystOut, iterations };
}
