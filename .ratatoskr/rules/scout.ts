// Demo ruleset for the scout node. `defineAgent` and the `onToolCall` hook are provided by the
// Ratatoskr script runtime; each tool call the scout agent makes is passed here to allow or deny.
// The TypeScript types below are stripped at load — they exist for editor ergonomics only.

type ToolCall = { tool: string; args: Record<string, unknown> };
type Decision = "allow" | "deny";

defineAgent("scout", {
  onToolCall({ tool }: ToolCall): Decision {
    // Block the tracker search; the scout should lean on semantic_search for this repo. The agent
    // receives the denial as tool feedback and continues with its remaining tools.
    if (tool === "papertrail_issue_search") return "deny";
    return "allow";
  },
});
