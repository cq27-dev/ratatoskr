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
  /** The stages whose work this node is — its own name for a node that is one stage, several for
   *  one they compose. Members run under their own identities, so their events arrive under names
   *  no box carries; this is what folds them into the box instead of drawing each beside it.
   *  Absent only from a `NodeView` this client built itself. */
  stages?: string[];
  /** Whether the run's recorded shape is what put it there. False means the server placed it from
   *  its checkpoints, in completion order — which `applyDerived` replaces with the stream's. */
  shaped?: boolean;
  /** Absent for a node that has not run, or that ran no model. */
  telemetry?: NodeTelemetry;
  /** What the node *would* run on, from config. Present before it has run. */
  planned?: PlannedNode;
  /** The node that ran this one. Only present for a node the shape does not place — a placed node's
   *  position already says what preceded it. */
  caller?: string;
}

/** Mirrors `pipeline::NodeTelemetryView`. */
export interface NodeTelemetry {
  model: string | null;
  /** Model calls in the node's latest attempt. */
  turns: number | null;
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens: number;
  /** Written to cache rather than read from it. Billed at a premium, and what separates a run that
   *  reused its context from one that rebuilt it. */
  cache_creation_input_tokens: number;
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
  /** Absent for a caller who is not signed in — the server does not tell strangers its paths. */
  dir?: string;
}

/** What a role may do. Ordered weakest-first, mirroring the server's `Role`. */
export type Role = "viewer" | "operator" | "admin";

/**
 * Who the viewer is. Every field is absent when nobody is signed in, which is a valid state and
 * not an error: an anonymous caller can read a public project.
 */
export interface Me {
  principal_id?: string;
  display_name?: string;
  role?: Role;
}

/** Whether this role may start runs and answer clarifications. */
/**
 * Whether a run has stopped executing. Mirrors `RunStatus::is_terminal` on the server, and the
 * same two-sided rule applies: a status this build has never heard of reads as still executing,
 * which shows a stale run rather than declaring a live one finished.
 */
export function isTerminal(status: RunStatus | null): boolean {
  return (
    status !== null &&
    status !== "pending" &&
    status !== "running" &&
    status !== "awaiting_clarification"
  );
}

export function mayAct(role: Role | undefined): boolean {
  return role === "operator" || role === "admin";
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

export interface PullRequestView {
  number: number;
  url: string;
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
  /** The pull request the run opened, if any. Null for comment-only runs, runs that published
   * nothing, and runs that never reached the publisher. */
  pull_request: PullRequestView | null;
  control: ControlView;
}

/**
 * What the operator has asked of a run — what the controls should show.
 *
 * What was *asked for*, not what the run has done about it: a node acts at its next turn
 * boundary, so a button that sprang back until the run noticed would read as a lost click.
 */
export interface ControlView {
  paused: boolean;
  /** Nodes stopped and waiting to be started again. */
  stopped: string[];
  /** Nodes with text delivered to the server but not yet picked up by the run. */
  steering: string[];
}

/** One thing an operator does to a run in flight. */
export type Command =
  | { command: "pause" }
  | { command: "resume" }
  | { command: "stop"; node: string }
  | { command: "start"; node: string }
  | { command: "steer"; node: string; text: string };

/**
 * Issue a command to a run. Resolves with what the server now holds, which is what the buttons
 * render — so a click shows immediately rather than waiting for the next poll.
 */
export async function control(
  project: string,
  runId: string,
  command: Command,
): Promise<ControlView> {
  const res = await fetch(
    `${scope(project)}/runs/${encodeURIComponent(runId)}/control`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(command),
    },
  );
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(
      body && typeof body === "object" && "error" in body
        ? String((body as { error: unknown }).error)
        : `${res.status}`,
    );
  }
  return body as ControlView;
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

/**
 * Thrown when the server says who you are is the problem.
 *
 * Carried as a type rather than a message so the dashboard can tell the two apart: 401 means a
 * sign-in would help and the form is worth showing, 403 means it would not.
 */
export class AuthRequired extends Error {
  constructor(public readonly status: number) {
    super(status === 403 ? "not allowed" : "sign in");
  }
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (res.status === 401 || res.status === 403) throw new AuthRequired(res.status);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return (await res.json()) as T;
}

/** Who the viewer is. Never fails on "nobody" — that is an answer. */
export const whoami = () => getJSON<Me>("/api/auth/me");

/** Exchange a username and password for a session cookie. */
export async function login(username: string, password: string): Promise<Me> {
  const res = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const message =
      body && typeof body === "object" && "error" in body
        ? String((body as { error: unknown }).error)
        : `sign-in failed (${res.status})`;
    throw new Error(message);
  }
  return body as Me;
}

/** End this session. Succeeds even if there was not one, so a stale tab can always get clean. */
export async function logout(): Promise<void> {
  await fetch("/api/auth/logout", { method: "POST" });
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
  /** An optional producer-provided summary for a tool call. */
  subject?: string;
  /** The bounded JSON arguments supplied to a tool call. */
  args?: unknown;
  /** How long a tool took, on its `tool_result`. */
  duration_ms?: number;
  /** Present on `node_start` and `checkpoint`: what the node ran on. */
  facts?: NodeFacts;
  /** Present on `usage` and `checkpoint`: what the attempt cost. */
  usage?: EventUsage;
  /** Model calls the attempt took, on a `checkpoint`. */
  turns?: number;
  /** Why the node failed, on a `checkpoint`. Its presence is what makes a node read as failed. */
  error?: string;
  /** Which attempt this was, on a `checkpoint`. */
  iteration?: number;
}

/** What one attempt cost, off the event stream. */
export interface EventUsage {
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens: number;
  cache_creation_input_tokens: number;
  reasoning_tokens: number;
  duration_ms: number;
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
/** Every event a run produced, oldest first — the record a historical view is rebuilt from. */
export const getHistory = (project: string, runId: string) =>
  getJSON<LiveEvent[]>(`${scope(project)}/runs/${encodeURIComponent(runId)}/history`);

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
  session: "fresh" | "reuse" | "compacted";
}

/** Mirrors `events::LiveNodeFacts` — what a node announced when it started. */
export interface NodeFacts {
  model: string;
  tools: string[];
  thinking: boolean;
  reuses_session: boolean;
}
