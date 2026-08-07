/**
 * The shapes `ratatoskr-serve` returns, mirroring its Rust types: `RunSummary`, `RunDetail`,
 * `WorktreeView`, `CheckpointView` in `src/lib.rs` and `NodeView` / `NodeState` in
 * `src/pipeline.rs`. Anything `Option<T>` in Rust is `T | null` here — serde emits explicit
 * nulls, not absent keys.
 */

/** Mirrors `pipeline::NodeState` (serde `rename_all = "snake_case"`). */
export type NodeState = "idle" | "working" | "done" | "failed";

/** Statuses the store persists, from `ratatoskr_core::RunStatus`. */
export type RunStatus =
  | "pending"
  | "running"
  | "awaiting_clarification"
  | "planned"
  | "converged"
  | "max_iterations_reached"
  | "failed"
  | "abandoned";

export interface NodeView {
  name: string;
  state: NodeState;
  /** Only the implementer (per converge iteration) and bookkeeper (replay) exceed one. */
  checkpoints: number;
  /** Position in the pipeline: stage is the column, lane the row within it. */
  stage: number;
  lane: number;
  /** Absent for a node that has not run, or that ran no model. */
  telemetry?: NodeTelemetry;
  /** What the node *would* run on, from config. Present before it has run. */
  planned?: PlannedNode;
}

/** Mirrors `pipeline::NodeTelemetryView`. */
export interface NodeTelemetry {
  model: string | null;
  /** Model calls in the node's latest attempt. */
  turns: number | null;
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens: number;
  /** Non-zero when the model reasoned before answering. Zero from endpoints that never report it. */
  reasoning_tokens: number;
  /** Whether the node was left free to reason. Configured, not observed. */
  thinking: boolean;
  duration_ms: number | null;
  tools: string[];
  /** Of those, the ones it actually called. */
  tools_used: string[];
  /** The node's memory carried over from an earlier attempt in this run. */
  reuses_session: boolean;
  first_at: string | null;
  last_at: string | null;
}

/** A watched project. Each has its own store, worktrees and logs; nothing is shared. */
export interface ProjectView {
  slug: string;
  dir: string;
}

export interface RunSummary {
  run_id: string;
  issue_id: string | null;
  status: RunStatus;
  updated_at: string;
}

export interface WorktreeView {
  path: string;
  exists: boolean;
}

export interface RunDetail {
  run_id: string;
  /** Null for a run with checkpoints but no `runs` row. */
  status: RunStatus | null;
  issue_id: string | null;
  updated_at: string | null;
  issue: string | null;
  last_activity: string | null;
  nodes: NodeView[];
  worktree: WorktreeView | null;
}

export interface CheckpointView {
  node_name: string;
  created_at: string;
  output: unknown;
}

/** Statuses that mean the run is still executing. */
export const LIVE: ReadonlySet<string> = new Set([
  "running",
  "awaiting_clarification",
  "pending",
]);

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return (await res.json()) as T;
}

const scope = (project: string) =>
  `/api/projects/${encodeURIComponent(project)}`;

export const listProjects = () => getJSON<ProjectView[]>("/api/projects");

export const listRuns = (project: string) =>
  getJSON<RunSummary[]>(`${scope(project)}/runs`);

/**
 * Start a run. Resolves as soon as the run has been spawned — a run takes minutes, so the id
 * comes back immediately and the run is followed through the normal endpoints. A 409 means the
 * server is already at its concurrent-run cap.
 */
export async function startRun(
  project: string,
  issue: string,
): Promise<string> {
  const res = await fetch(`${scope(project)}/runs`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ issue }),
  });
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const msg =
      body && typeof body === "object" && "error" in body
        ? String((body as { error: unknown }).error)
        : `${res.status}`;
    throw new Error(msg);
  }
  return (body as StartedRun).run_id;
}

interface StartedRun {
  run_id: string;
}

export const getRun = (project: string, runId: string) =>
  getJSON<RunDetail>(`${scope(project)}/runs/${encodeURIComponent(runId)}`);

/** One thing a run did, from a run's event stream. Mirrors `events::LiveEvent`. */
export interface LiveEvent {
  at: string;
  /** `tool_call`, `model_text`, `checkpoint`, `question`, `question_answered`, `run_*`, … */
  kind: string;
  node: string | null;
  detail: string;
  /** Present on a `question` event: what an answer is posted against. */
  question_id?: string;
  /** The one argument that identifies a tool call: the path read, the pattern searched. */
  arg?: string;
  /** How long a tool took, on its `tool_result`. */
  duration_ms?: number;
  /** Present on a `node_start` event: what the node is about to run on. */
  facts?: NodeFacts;
}

/**
 * Answer a question a run is waiting on. Rejects if it is no longer pending — already answered,
 * timed out, or replayed from history — which the caller should show rather than retry.
 */
export async function answerQuestion(
  questionId: string,
  answer: string,
): Promise<void> {
  const res = await fetch(
    `/api/clarifications/${encodeURIComponent(questionId)}`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ answer }),
    },
  );
  if (!res.ok) {
    const body: unknown = await res.json().catch(() => null);
    throw new Error(
      body && typeof body === "object" && "error" in body
        ? String((body as { error: unknown }).error)
        : `${res.status}`,
    );
  }
}

/**
 * Follow a run's activity. Returns a teardown function.
 *
 * The server replays recent history on connect and then streams, so a dashboard opened mid-run
 * shows what already happened instead of an empty pane.
 */
export function followRun(
  project: string,
  runId: string,
  handlers: { onEvent: (event: LiveEvent) => void; onReset: () => void },
): () => void {
  const source = new EventSource(
    `${scope(project)}/runs/${encodeURIComponent(runId)}/events`,
  );
  // EventSource reconnects on its own after a drop, and the server replays history to every new
  // connection — so without clearing here, one blip duplicates the whole feed.
  source.onopen = () => handlers.onReset();
  source.onmessage = (message) => {
    try {
      handlers.onEvent(JSON.parse(message.data) as LiveEvent);
    } catch {
      // A malformed frame shouldn't tear down the stream.
    }
  };
  return () => source.close();
}

export const getNodeCheckpoints = (
  project: string,
  runId: string,
  node: string,
) =>
  getJSON<CheckpointView[]>(
    `${scope(project)}/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(node)}`,
  );

/** Mirrors `pipeline::PlannedNode` — a node's configured route, known before it runs. */
export interface PlannedNode {
  model: string;
  thinking: boolean;
  reuses_session: boolean;
}

/** Mirrors `events::LiveNodeFacts` — what a node announced when it started. */
export interface NodeFacts {
  model: string;
  tools: string[];
  thinking: boolean;
  reuses_session: boolean;
}
