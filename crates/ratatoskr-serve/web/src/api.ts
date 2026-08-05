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

export const getRun = (runId: string) =>
  getJSON<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`);

export const getNodeCheckpoints = (runId: string, node: string) =>
  getJSON<CheckpointView[]>(
    `/api/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(node)}`,
  );
