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
  first_at: string | null;
  last_at: string | null;
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

export const listRuns = () => getJSON<RunSummary[]>("/api/runs");

/**
 * Start a run. Resolves as soon as the run has been spawned — a run takes minutes, so the id
 * comes back immediately and the run is followed through the normal endpoints. A 409 means the
 * server is already at its concurrent-run cap.
 */
export async function startRun(issue: string): Promise<string> {
  const res = await fetch("/api/runs", {
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

export const getRun = (runId: string) =>
  getJSON<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`);

/** One thing a run did, from `GET /api/runs/{id}/events`. Mirrors `events::LiveEvent`. */
export interface LiveEvent {
  at: string;
  /** `tool_call`, `model_text`, `checkpoint`, `run_started`, `run_finished`, `run_failed`, … */
  kind: string;
  node: string | null;
  detail: string;
}

/**
 * Follow a run's activity. Returns a teardown function.
 *
 * The server replays recent history on connect and then streams, so a dashboard opened mid-run
 * shows what already happened instead of an empty pane.
 */
export function followRun(
  runId: string,
  handlers: { onEvent: (event: LiveEvent) => void; onReset: () => void },
): () => void {
  const source = new EventSource(
    `/api/runs/${encodeURIComponent(runId)}/events`,
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

export const getNodeCheckpoints = (runId: string, node: string) =>
  getJSON<CheckpointView[]>(
    `/api/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(node)}`,
  );
