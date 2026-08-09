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
  ],
});
