// Ruleset for the scout node. `defineAgent` and the `onToolCall` hook are provided by the
// Ratatoskr script runtime; each tool call the scout agent makes is passed here to allow or deny.
// The TypeScript types below are stripped at load — they exist for editor ergonomics only.

type ToolCall = { tool: string; args: Record<string, unknown> };
type Decision = "allow" | "deny";

defineAgent("scout", {
  // A ruleset can also fully declare a node's route and persona — uncomment to override:
  //   model: { provider: "moonshot", model: "kimi-k2.5" },  // sufficient on its own; no [models.scout] needed
  //   systemPrompt: "You are the scout...",                 // replaces the node's built-in preamble
  //   tools: { allow: ["semantic_search"] },                // REPLACES default_tools; `deny` also supported
  //   maxTurns: 40,
  onToolCall(_call: ToolCall): Decision {
    // Nothing is denied here. This hook stays as the seam a repo reaches for when it does have a
    // rule to enforce — the gate is per tool call, so it sees the arguments as well as the name.
    //
    // It previously denied `papertrail_issue_search` to demonstrate the mechanism. That cost a run
    // whose task was "read GitHub issue #6" two round-trips to a human and got it nowhere, because
    // denying a scout its tracker is denying it the thing it was asked to read. A demonstration is
    // not worth a standing policy.
    return "allow";
  },
});
