// Reference `.ratatoskr/workflow.ts` — reproduces Ratatoskr's built-in run flow. Copy it to
// `.ratatoskr/workflow.ts` and edit to customize how a run is sequenced.
//
// The script only *composes* the node bindings. Every gate stays Rust-enforced no matter what the
// script does: each binding validates and checkpoints its own output; `redTeam()` runs the
// false-convergence guard itself; `iterate()` enforces `max_iterations`; and the run's terminal
// status (converged vs. max-iterations) is inferred from the checkpoints Rust wrote, never from
// anything the script returns. A script can reorder and choose, but it cannot weaken a gate.
//
// Bindings (all async except the pure converge helpers):
//   scout(issue) -> ScoutOutput
//   memory({issue, context}) -> MemoryOutput
//   analyze({issue, scout, memory}) -> AnalystOutput
//   redTeam() -> RedTeamOutput                     // baseline; throws if it ran no tests
//   implement({analyst}) -> ImplementerOutput      // creates the worktree (once)
//   iterate({}) -> ImplementerOutput               // re-drives the CLI on that worktree
//   verify({analyst}) -> VerifyResult              // reviews the diff against the plan
//
// `verify()` returns { configured, unavailable, findings, blocking, needsReplan }. Rust applies
// `[implementer] verify_threshold` — a script decides *whether* to review and what to do about
// findings, never what counts as blocking. `needsReplan` means a blocking finding faults the PLAN,
// so the useful response is `analyze({...., previous, findings})` before `iterate()`, rather than
// re-driving the implementer at a requirement already shown to be wrong.
//
// A run that calls verify() and returns with blocking findings standing does NOT converge: the
// terminal status is inferred from the verifier checkpoint, not from what the script returns.
//
// A workflow that introduces a node of its own declares it, so its `.ratatoskr/rules/<node>.ts` is
// accepted rather than read as a typo:
//
//   defineWorkflow({ name: "deep", nodes: ["reviewer2"] });
//   isConverged({baseline, post}) -> boolean
//   testCommandRan(output) -> boolean
//   newlyIntroducedFailures({baseline, post}) -> string[]

// `plan`: scout -> memory -> analyst. Backs `ratatoskr plan`.
async function plan(input: { issue: string }) {
  const scoutOut = await scout(input.issue);
  const memoryOut = await memory({ issue: input.issue, context: scoutOut.papertrail_summary });
  const analystOut = await analyze({ issue: input.issue, scout: scoutOut, memory: memoryOut });
  return { scout: scoutOut, memory: memoryOut, analyst: analystOut };
}

// `run`: plan, then fork red-team ∥ implementer, then converge. Backs `ratatoskr run`.
async function run(input: { issue: string; maxIterations: number }) {
  const scoutOut = await scout(input.issue);
  const memoryOut = await memory({ issue: input.issue, context: scoutOut.papertrail_summary });
  const analystOut = await analyze({ issue: input.issue, scout: scoutOut, memory: memoryOut });

  // Fork: both run concurrently off the frozen post-analyst state.
  const [redTeamOut, first] = await Promise.all([redTeam(), implement({ analyst: analystOut })]);

  // Converge: iterate the implementer until it introduces no new failures, or the budget runs out.
  // (`maxIterations` is also hard-enforced inside `iterate` — this loop just stops first.)
  let impl = first;
  let iterations = 1;
  while (true) {
    if (testCommandRan(impl) && isConverged({ baseline: redTeamOut, post: impl })) break;
    if (iterations >= input.maxIterations) break;
    impl = await iterate({});
    iterations += 1;
  }
  return { iterations };
}
