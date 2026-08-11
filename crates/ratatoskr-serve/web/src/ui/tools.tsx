/**
 * What a tool is, as an icon.
 *
 * One table, read by both surfaces that draw tools: the capability icons on a node's box, and the
 * per-call icon in the feed. They were the same question asked twice — "what kind of thing is
 * this?" — and answering it in two places is how the graph ends up saying a call was a search
 * while the feed says it was something else.
 *
 * Grouped rather than one icon per tool. A node carries up to a dozen, and a dozen distinct glyphs
 * is a texture rather than information; the grouping answers what a reader actually asks — can this
 * node read, can it write, can it run things — and the hover title carries the exact names when
 * that is not enough.
 */
import {
  BookOpen,
  ClipboardCheck,
  FileText,
  Flag,
  FlaskConical,
  GitCommitHorizontal,
  MessageCircleQuestion,
  Bookmark,
  History,
  Info,
  MessageSquare,
  Pencil,
  Pause,
  Play,
  Search,
  Send,
  Terminal,
  Wrench,
} from "lucide-react";
import type { JSX } from "react";

export type ToolGroup = {
  icon: typeof FileText;
  label: string;
  match: (tool: string) => boolean;
};

/**
 * Most specific first. `match` is tried in order and the first hit wins, so a rule that would also
 * catch a more specific tool must come after it.
 */
export const TOOL_GROUPS: ReadonlyArray<ToolGroup> = [
  { icon: Pencil, label: "edits files", match: (t) => t === "Write" || t === "Edit" },
  { icon: Terminal, label: "runs commands", match: (t) => t === "Bash" },
  { icon: FileText, label: "reads files", match: (t) => ["Read", "Grep", "Glob"].includes(t) },
  { icon: BookOpen, label: "loads a skill", match: (t) => t === "Skill" },
  { icon: MessageCircleQuestion, label: "asks another node", match: (t) => t === "ask" },
  { icon: Send, label: "publishes", match: (t) => t === "gh" || t === "git_push" },
  // Before the index rule: `memory_search` and `commit_search` both contain "search", and what
  // they reach for is not the code graph.
  { icon: Bookmark, label: "reads and writes memories", match: (t) => t.startsWith("memory") },
  {
    icon: History,
    label: "reads the papertrail",
    match: (t) =>
      t.startsWith("papertrail") ||
      t.startsWith("git_") ||
      t.startsWith("commit") ||
      t === "git_blame_chunk",
  },
  {
    icon: Search,
    label: "searches the index",
    // Named rather than pattern-matched where a name gives nothing away: `read_chunk` and
    // `find_callers` are index tools whose names contain none of the words a pattern looks for,
    // and both fell through to the generic wrench until they were listed.
    match: (t) =>
      t.includes("search") ||
      t.includes("symbol") ||
      t.includes("impact") ||
      [
        "read_chunk",
        "find_callers",
        "trace_callees",
        "repo_brief",
        "repo_clusters",
        "docs_for_symbol",
        "find_clones",
        "clones_for_symbol",
        "ffi_surface",
      ].includes(t),
  },
];

/** The group a single tool falls in, or nothing when none claims it. */
export function groupOf(tool: string): ToolGroup | undefined {
  return TOOL_GROUPS.find((g) => g.match(tool));
}

/**
 * One tool call's icon, for a feed row.
 *
 * Falls back to the same wrench the node box uses for tools no group claims, so an unrecognised
 * tool is still visibly a tool rather than a gap where the other rows have one.
 */
export function ToolIcon({ tool, size = 12 }: { tool: string; size?: number }): JSX.Element {
  const group = groupOf(tool);
  const Icon = group?.icon ?? Wrench;
  return (
    <span className="ev-i" data-tip={group ? `${tool} — ${group.label}` : tool}>
      <Icon size={size} aria-hidden="true" />
    </span>
  );
}

/* ── Row kinds ─────────────────────────────────────────────────────────── */

/**
 * What a non-tool row is: the word for it, and its glyph.
 *
 * Here rather than beside the feed for the same reason the tool groups are — the label and the
 * icon are one decision about what a row *is*, and splitting them across two files is how they
 * drift into disagreeing.
 *
 * The label is not always the kind's name. `model_text` is the commonest row in any run and
 * "MODEL TEXT" spends two words on the least surprising thing a model does; `msg` says it and
 * gives the column back to what was actually said.
 */
export const KINDS: Record<string, { label: string; icon: typeof FileText }> = {
  model_text: { label: "msg", icon: MessageSquare },
  // Information, not a warning. Most events are ordinary notices, and a row that looks
  // like an alert a hundred times over stops meaning anything when one genuinely is.
  event: { label: "event", icon: Info },
  node_start: { label: "start", icon: Play },
  run_paused: { label: "paused", icon: Pause },
  run_resumed: { label: "continued", icon: Play },
  checkpoint: { label: "checkpoint", icon: Flag },
  acceptance_step: { label: "check", icon: FlaskConical },
  authored_tests: { label: "tests written", icon: ClipboardCheck },
  committed: { label: "commit", icon: GitCommitHorizontal },
  question: { label: "asks", icon: MessageCircleQuestion },
};

/** The word for a row of this kind. Unknown kinds keep their own name, underscores and all. */
export function kindLabel(kind: string): string {
  return KINDS[kind]?.label ?? kind.replace(/_/g, " ");
}

/**
 * A row's icon: the tool's when it invoked one, the kind's otherwise.
 *
 * Every row gets one. A column where only some rows carry a glyph scans worse than one where none
 * do — the eye reads the gaps as meaningful when they only mean "no icon was defined".
 */
export function RowIcon({ kind, action }: { kind: string; action: string }): JSX.Element | null {
  if (kind === "tool_call") return <ToolIcon tool={action} />;
  const Icon = KINDS[kind]?.icon;
  if (!Icon) return null;
  return (
    <span className="ev-i" data-tip={kind.replace(/_/g, " ")}>
      <Icon size={12} aria-hidden="true" />
    </span>
  );
}
