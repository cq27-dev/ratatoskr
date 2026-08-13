// Bundled standard workflow, version 1. The declarations own generic model-stage contracts; its
// entrypoints compose Rust-owned operation hosts, which retain checkpointing and workflow gates.
import * as nodes from "ratatoskr/nodes";

defineWorkflow({
  name: "ratatoskr-standard-v1",
  stages: [
    stage("overseer", nodes.overseer),
    stage("characterizer", nodes.characterizer),
    stage("redteam_classifier", nodes.redteam_classifier),
    stage("redteam_author", nodes.redteam_author),
    stage("implementer_attempt", nodes.implementer_attempt),
    stage("context_distillation", nodes.context_distillation),
    stage("scout", nodes.scout),
    stage("analyst", nodes.analyst),
    stage("bookkeeper", nodes.bookkeeper),
    stage("publisher", nodes.publisher),
    stage("verifier", nodes.verifier),
  ],
  // Where a run of this workflow is drawn: one entry per column, `nodes` its lanes in order. The
  // names are the ones the run checkpoints under, which is what a viewer matches a node's record
  // against. `optional` marks a column that may legitimately be skipped — the overseer only runs
  // where a workflow has to be chosen, the verifier only where the repo gave it a route — so an
  // empty one has not stalled.
  //
  // Column ORDER is meaningful: adjacent columns are joined, every node of one to every node of the
  // next, so placing a node in the column after another draws a hand-off saying the first fed the
  // second. Within a column nothing is ordered — the two deliveries share the last one because
  // neither needs the other's result, and the red team and the implementer share the fork column
  // though the implementer cannot start until the red team has finished. A lane is a position, not
  // evidence of concurrency; what ran after what is read from the run's own events.
  layout: [
    { nodes: ["overseer"], optional: true },
    { nodes: ["context"] },
    { nodes: ["analyst"] },
    { nodes: ["redteam", "implementer"] },
    { nodes: ["verifier"], optional: true },
    { nodes: ["bookkeeper", "publisher"] },
  ],
});

// Back `ratatoskr plan` with the same runtime used by repository workflows. `context` owns the
// deterministic evidence baseline and checkpoints the merged result; `analyst` is the declared
// model stage and checkpoints the exact structured input below. Rust reconstructs the returned
// PlanOutcome from those checkpoints, so this function's return value is informational only.
export async function plan(input: { issue: string }) {
  const gathered = await context(input.issue);
  // The initial built-in hand-off is AnalystInput::fresh. Rust retains the brief and constraints
  // in PlanOutcome and supplies them if review later asks the analyst to revise the plan.
  const analysis = await analyst({
    issue: input.issue,
    scout: gathered.scout,
    memory: gathered.memory,
  });
  return { context: gathered, analyst: analysis };
}

// The standard full flow composes only ordinary stage/operation hosts. Worktree creation,
// acceptance, the referee, review thresholds, iteration limits, checkpointing, terminal status,
// commits, publishing, bookkeeping, and cleanup remain inside their Rust owners.
async function full(input: {
  issue: string;
  maxIterations: number;
  alwaysFork: boolean;
}) {
  const gathered = await context(input.issue);
  let analysis = await analyst({
    issue: input.issue,
    scout: gathered.scout,
    memory: gathered.memory,
  });

  // AnalystOutput defaults an omitted changes_code to true at the Rust typed boundary. Mirror that
  // fail-safe here: only an explicit false may skip the fork, and the configured override only
  // ever adds work.
  if (analysis.changes_code === false && !input.alwaysFork) {
    return { context: gathered, analyst: analysis, iterations: 0 };
  }

  // Red-team authoring must finish before implementation: both use the same prepared worktree,
  // and tests are authored from the frozen interface before implementation can observe them.
  const baseline = await redTeam();
  let implementation = await implement({ analyst: analysis });
  let iterations = 1;

  // The ordinary budget is hard. Once it is spent, Rust may offer exactly one recovery when the
  // checkpointed review history shows a pattern worth taking back to the analyst. This operation
  // accepts no evidence or plan from the workflow and performs the revision + final attempt as one
  // bounded host action; `null` means the ceiling is final.
  async function recoverAtCeiling() {
    const recovery = await replanAtCeiling();
    if (recovery === null) return false;
    analysis = recovery.analyst;
    implementation = recovery.implementation;
    iterations += 1;
    return true;
  }

  while (true) {
    // Every host is an `async function`: an unawaited call is a Promise, and a Promise is truthy
    // whatever it resolves to. Both operands must be awaited before the `&&` sees them.
    const testsClean =
      (await testCommandRan(implementation)) &&
      (await isConverged({ baseline, post: implementation }));
    if (testsClean) {
      const review = await verify({ analyst: analysis });
      if (!review.configured || review.unavailable || review.blocking.length === 0) break;
      if (iterations >= input.maxIterations) {
        if (await recoverAtCeiling()) continue;
        break;
      }
      if (review.needsReplan) {
        analysis = await analyst({
          issue: input.issue,
          scout: gathered.scout,
          memory: gathered.memory,
          brief: gathered.brief,
          constraints: gathered.constraints,
          previous: analysis,
          findings: review.blocking,
        });
      }
      implementation = await iterate({ review });
      iterations += 1;
      continue;
    }
    if (iterations >= input.maxIterations) {
      if (await recoverAtCeiling()) continue;
      break;
    }
    implementation = await iterate({});
    iterations += 1;
  }
  return { context: gathered, analyst: analysis, iterations };
}

export async function run(input: {
  issue: string;
  maxIterations: number;
  alwaysFork: boolean;
}) {
  return full(input);
}

// Rust lifecycle adapters enter standard model stages here. The adapter supplies only the one
// stage host it is authorized to run, while this bundled runtime applies that stage's declared
// renderQuestion before the generic executor receives it.
export async function standardStageTurn(input: { stage: string; input: any }) {
  switch (input.stage) {
    case "overseer": return await overseer(input.input);
    case "characterizer": return await characterizer(input.input);
    case "redteam_classifier": return await redteam_classifier(input.input);
    case "redteam_author": return await redteam_author(input.input);
    case "implementer_attempt": return await implementer_attempt(input.input);
    case "context_distillation": return await context_distillation(input.input);
    case "analyst": return await analyst(input.input);
    case "bookkeeper": return await bookkeeper(input.input);
    case "publisher": return await publisher(input.input);
    case "verifier": return await verifier(input.input);
    default: throw new Error(`unknown standard stage ${input.stage}`);
  }
}
