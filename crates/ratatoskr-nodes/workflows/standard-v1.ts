// Bundled standard-stage declarations, version 1. Repository workflow scripts still own
// sequencing; these declarations own the generic host contract for migrated standard stages.
defineWorkflow({
  name: "ratatoskr-standard-v1",
  stages: [
    {
      id: "scout",
      agent: "explore",
      inputContract: "String",
      outputContract: "ScoutOutput",
      outputSchema: {
        type: "object",
        properties: {
          related_items: {
            type: "array",
            items: {
              type: "object",
              properties: {
                item_key: { type: "string" },
                title: { type: "string" },
                url: { type: "string" },
                relation: { type: "string" },
                summary: { type: "string" },
              },
            },
          },
          papertrail_summary: { type: "string" },
        },
        required: ["papertrail_summary"],
      },
      instructions: `You are the scout in a code-planning pipeline. Given an issue description, find prior art and
context in THIS repository: use \`papertrail_issue_search\` to find related tracker issues/PRs and
\`semantic_search\` to find related code. Call the tools — do not guess. Then produce a structured
summary: a list of the most relevant related items (with your one-line take on how each
relates), and a short free-text papertrail summary the downstream analyst can build on. Be
concrete and grounded in what the tools returned.

An issue's COMMENTS carry as much as its body, and often more: a correction to the original
description, a decision taken after it was filed, a measurement someone added, a note that half of
it is already done. The body is what somebody thought at the time of writing; the comments are what
was learnt since. Read them for every item you report on, and when a comment contradicts the body,
say so — the analyst plans from what you return, and a plan built on a superseded description is
wrong before it starts.

Distinguish "the tracker has nothing on this" from "the tracker is not readable from here". An
empty result means one of them and you cannot tell which from the result alone: check whether the
tracker search returns anything for a broad term you would expect to match, and if it does not,
report that the papertrail is unavailable rather than that there is no prior art. They are opposite
findings. One says the ground is clear; the other says you cannot see the ground.`,
      capabilities: ["read"],
      tools: ["papertrail_issue_search", "semantic_search"],
      // null means the selected TOML/profile/ruleset route owns the session policy.
      session: null,
      appendRepositoryGuidance: false,
      arrayNormalization: [
        {
          field: "related_items",
          defaultEmpty: true,
          retainWhenAnyNonBlank: ["item_key", "title"],
        },
      ],
    },
    {
      id: "analyst",
      agent: "reason",
      inputContract: "AnalystInput",
      outputContract: "AnalystOutput",
      outputSchema: {
        type: "object",
        properties: {
          acceptance: {
            type: "array",
            default: [],
            items: { "$ref": "#/$defs/AcceptanceStep" },
          },
          changes_code: { type: "boolean", default: true },
          impact_summary: { type: "string" },
          interface: {
            type: "array",
            default: [],
            items: { "$ref": "#/$defs/InterfaceItem" },
          },
          requirements: {
            type: "array",
            default: [],
            items: { type: "string" },
          },
          residual_risk: { type: "string", default: "" },
          risks: {
            type: "array",
            default: [],
            items: { type: "string" },
          },
          touched: {
            type: "array",
            default: [],
            items: { type: "string" },
          },
        },
        required: ["impact_summary"],
        "$defs": {
          AcceptanceStep: {
            type: "object",
            properties: {
              command: { type: "array", items: { type: "string" } },
              name: { type: "string" },
            },
            required: ["name", "command"],
          },
          InterfaceItem: {
            type: "object",
            properties: {
              happy: {
                type: "array",
                default: [],
                items: { type: "string" },
              },
              name: { type: "string" },
              sad: {
                type: "array",
                default: [],
                items: { type: "string" },
              },
              shape: { type: "string" },
            },
            required: ["name", "shape"],
          },
        },
      },
      instructions: `You are the analyst in a code-planning pipeline. You are given an issue, the scout's findings,
and relevant repo memories. Use \`impact_surface\` and \`symbol_lookup\` to determine what this
change actually touches and its blast radius — call the tools, don't guess. Produce: an impact
summary, the specific symbols/paths touched, a list of risks (each a short line — lead with the
severity if it's clear-cut), a list of concrete requirements the implementation must satisfy,
and a residual-risk note capturing what remains uncertain or unknown after your analysis. Also
set \`changes_code\`: true when carrying out this plan means editing code in this repository,
false when it does not — research, a review, an architecture answer, or expanding an issue's
description all produce no code change. Judge the task you were given, not the breadth of what
it touches: a question about eight files is still a question. When it does change code, also set
\`acceptance\`: the ordered steps that must run and pass for this change to be believed done. It is a
list of objects, each with a short \`name\` and a \`command\` given as an argv array — not a shell
string, because these run without a shell to split them:

    [{"name": "tests", "command": ["cargo", "test", "--workspace"]}]

Use the repo's own tooling, and include every step the check needs — building an artifact before
testing it is two steps, not one. The steps run in a fresh worktree with nothing installed, so a
repository whose dependencies are not committed needs the install as its own first step: a check
that assumes them fails on the framework rather than on the change, and says nothing about whether
the change is right. Leave the list empty to accept the repository's configured test command, which
is the right answer whenever the existing suite already covers the change.

**Read the repository's CI configuration and take the acceptance from it**, when there is one —
\`.github/workflows/*.yml\`, or whatever the repository uses. Those are the checks that decide
whether the change can be merged, so a change that passes something weaker is a change that
reddens CI and comes back. A run that tested only the suite while CI also ran a formatter has
delivered work that fails the moment it is opened.

Take the jobs that gate a change, and only those: the ones triggered by \`push\` or \`pull_request\`
that build, test, lint or format. Do NOT take deploy, release, publish, or scheduled jobs — those
act outside this machine, and running one from a sandbox is at best waste and at worst an
unintended release. Do not reproduce a matrix either: one representative configuration is the
check, and eight are the same check eight times at eight times the cost. Take the commands the
workflow runs, not the workflow file — there is no CI runner here, so \`actions/checkout\` and a
toolchain-install action have no equivalent and no purpose in a tree that is already checked out.

Where CI's checks and the repository's documented ones disagree, prefer CI: it is the one that
actually refuses the change.

Also set \`interface\`: the surface this change is contracted to have. Someone else writes the tests
from it — the red team, working only from what you say here — and the implementer builds against
the same description. That is the point: tests written by the author of the code are shaped around
the code that appeared, and neither of them can see it.

Each entry names one piece of surface (\`name\`), its shape after the change (\`shape\` — the
signature, the parameters and their types, enough to call it without reading an implementation
that does not exist yet), and two lists of expectations:

- \`happy\` — used correctly. Each entry an input and the result it must produce.
- \`sad\` — misused, or the world not cooperating: a bad argument, a missing file, a value at its
  limit, a concurrent caller. These are the ones an implementer writing its own tests quietly omits.

Write expectations that can be checked, not intentions. "Rejects a negative timeout with an error
naming the field" is one; "handles errors gracefully" is not. Leave \`interface\` empty when the
change genuinely has no callable surface — an internal refactor, a comment — rather than inventing
a contract to fill it.

You are also the pipeline's fallback answerer: when another node
cannot resolve something on its own, its question routes to you, so hold clear, present-tense
judgments about the change that you can share when asked.`,
      renderQuestion(input: any) {
        let question = "";
        const findings = input.findings || [];
        if (input.previous && findings.length > 0) {
          question +=
            "THIS IS A REVISION. A change was implemented against your previous plan and reviewed. " +
            "The review found faults it judged to be in the PLAN rather than in the code — the " +
            "requirement was wrong, missing, or impossible as written, so re-implementing it will " +
            "not help.\n\n" +
            "Decide, for each finding: does the plan need to change, or was the plan right and the " +
            "implementation simply did not follow it? Amend the requirements where they were " +
            "wrong. Where they were right, restate them unchanged — repeating a correct " +
            "requirement is how you say the fault was in the execution.\n\n" +
            "Keep everything that still holds. You are amending a plan, not writing a new one.\n\n";
        }
        question += `ISSUE:\n${input.issue}\n\n`;
        if (input.brief) question += `WHAT BEARS ON THIS:\n${input.brief}\n\n`;
        const constraints = input.constraints || [];
        if (constraints.length > 0) {
          question += "CONSTRAINTS THIS MUST RESPECT:\n";
          for (const constraint of constraints) {
            const source = constraint.from_memory_id ? ` [${constraint.from_memory_id}]` : "";
            question += `- ${constraint.says}${source}\n`;
          }
          question += "\n";
        }
        if (input.previous) {
          question += `YOUR PREVIOUS PLAN:\n${input.previous.impact_summary}\n`;
          const requirements = input.previous.requirements || [];
          if (requirements.length > 0) {
            question += "Requirements you set:\n";
            for (const requirement of requirements) question += `- ${requirement}\n`;
          }
          const priorInterface = input.previous.interface || [];
          if (priorInterface.length > 0) {
            question +=
              "\nThe interface you contracted. Tests are already written against it, by " +
              "someone who cannot see the code. Restate it unchanged unless a finding is about " +
              "the interface itself: changing a name or a signature here breaks tests that are " +
              "not wrong.\n";
            for (const item of priorInterface) {
              question += `- ${item.name}\n  ${item.shape}\n`;
              for (const happy of item.happy || []) question += `  happy: ${happy}\n`;
              for (const sad of item.sad || []) question += `  sad: ${sad}\n`;
            }
          }
          question += "\n";
        }
        if (findings.length > 0) {
          question += "WHAT THE REVIEW FOUND:\n";
          for (const finding of findings) {
            const file = finding.file ? ` (${finding.file})` : "";
            question += `- [${finding.severity}]${file} ${finding.summary}\n`;
            question += `  Fails when: ${finding.failure_scenario}\n`;
          }
          question += "\n";
        }
        question += `SCOUT SUMMARY:\n${input.scout.papertrail_summary}\n\n`;
        const related = input.scout.related_items || [];
        if (related.length > 0) {
          question += "RELATED ITEMS:\n";
          for (const item of related) {
            question += `- [${item.item_key}] ${item.title} — ${item.relation}\n`;
          }
          question += "\n";
        }
        const memories = input.memory.memories || [];
        if (memories.length > 0) {
          question += "REPO MEMORIES:\n";
          for (const memory of memories) {
            const detail = memory.summary === undefined || memory.summary === null
              ? memory.body
              : memory.summary;
            question += `- (${memory.kind}) ${memory.title}: ${detail}\n`;
          }
          question += "\n";
        }
        return question;
      },
      capabilities: ["read"],
      tools: ["impact_surface", "symbol_lookup", "semantic_search"],
      session: "compacted",
      appendRepositoryGuidance: false,
    },
  ],
});
