import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import PipelineGraph from "./PipelineGraph";
import {
  LIVE,
  getNodeCheckpoints,
  getRun,
  answerQuestion,
  followRun,
  listProjects,
  listRuns,
  startRun,
  type CheckpointView,
  type LiveEvent,
  type NodeFacts,
  type ProjectView,
  type RunDetail,
  type RunSummary,
} from "./api";

/** A live run with nothing recorded for this long is almost certainly dead, not busy. */
const STALE_MS = 120_000;
/** How many live events to keep on screen. Old ones scroll away; the log file keeps everything. */
const FEED_LIMIT = 250;

const short = (id: string | null) => (id ? id.slice(0, 8) : "—");
const clock = (ts: string | null) => (ts ? ts.slice(11, 19) : "—");

function Projects({
  projects,
  selected,
  onSelect,
}: {
  projects: ProjectView[];
  selected: string | null;
  onSelect: (slug: string) => void;
}) {
  // With one project there is nothing to choose, so the switcher stays out of the way.
  if (projects.length < 2) return null;
  return (
    <div className="projects">
      <div className="sec">
        <span>[ PROJECTS ]</span>
        <output>{projects.length}</output>
      </div>
      {projects.map((p) => (
        <button
          key={p.slug}
          className="proj"
          aria-current={p.slug === selected}
          onClick={() => onSelect(p.slug)}
          title={p.dir}
        >
          {p.slug}
        </button>
      ))}
    </div>
  );
}

function NewRun({
  project,
  onStarted,
}: {
  project: string;
  onStarted: (runId: string) => void;
}) {
  const [issue, setIssue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!issue.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const runId = await startRun(project, issue);
      setIssue("");
      onStarted(runId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      box.current?.focus();
    }
  };

  return (
    <form className="newrun" onSubmit={(e) => void submit(e)}>
      <div className="sec">
        <span>[ NEW RUN ]</span>
      </div>
      <textarea
        ref={box}
        value={issue}
        onChange={(e) => setIssue(e.target.value)}
        placeholder="describe the task…"
        rows={3}
        spellCheck={false}
        // Enter submits; newlines still available for multi-line issues.
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) void submit(e);
        }}
      />
      <button type="submit" disabled={busy || !issue.trim()}>
        {busy ? "STARTING…" : ">>> START RUN"}
      </button>
      {error && <p className="newrun-error hazard">{error}</p>}
    </form>
  );
}

function Rail({
  projects,
  project,
  onProject,
  runs,
  selected,
  onSelect,
  onStarted,
}: {
  projects: ProjectView[];
  project: string;
  onProject: (slug: string) => void;
  runs: RunSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
  onStarted: (runId: string) => void;
}) {
  return (
    <nav className="rail">
      <Projects projects={projects} selected={project} onSelect={onProject} />
      <NewRun project={project} onStarted={onStarted} />
      <div className="sec">
        <span>[ RUNS ]</span>
        <output>{runs.length}</output>
      </div>
      {runs.length === 0 && <p className="empty">no runs recorded</p>}
      {runs.map((r) => (
        <button
          key={r.run_id}
          className="run"
          aria-current={r.run_id === selected}
          onClick={() => onSelect(r.run_id)}
        >
          <span className="run-id">
            <samp>{short(r.run_id)}</samp>
            <span className={`st st--${r.status}`}>{r.status}</span>
          </span>
          <span className="run-sub">
            {r.updated_at.replace("T", " ").slice(0, 19)}
          </span>
        </button>
      ))}
    </nav>
  );
}

function RunMeta({ detail, lastEventAt }: { detail: RunDetail; lastEventAt: number | null }) {
  // Checkpoints are minutes apart by design — the implementer runs for ten of them between two —
  // so a run judged on those alone reads as stale while it is plainly working. The event stream is
  // the finer signal: tool calls arrive every few seconds, and silence there is real silence.
  const lastSeen = Math.max(
    detail.last_activity ? Date.parse(detail.last_activity) : 0,
    lastEventAt ?? 0,
  );
  const stale =
    detail.status !== null &&
    LIVE.has(detail.status) &&
    lastSeen > 0 &&
    Date.now() - lastSeen > STALE_MS;

  return (
    <header className="runmeta">
      <h2>{detail.issue ? detail.issue.split("\n")[0] : "UNTITLED RUN"}</h2>
      <dl>
        <div>
          <dt>Run</dt>
          <dd>
            <samp>{detail.run_id}</samp>
          </dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd>
            <span className={`st st--${detail.status ?? "idle"}`}>
              {detail.status ?? "no row"}
            </span>
            {/* `updated_at` only moves on status transitions, so a killed run keeps
                claiming it's running. Staleness is the only tell the store can give. */}
            {stale && <span className="hazard"> / STALE</span>}
          </dd>
        </div>
        <div>
          <dt>Last activity</dt>
          <dd>
            <data value={detail.last_activity ?? ""}>
              {clock(detail.last_activity)}
            </data>
          </dd>
        </div>
        <div>
          <dt>Worktree</dt>
          <dd>
            {detail.worktree ? (
              <span className={detail.worktree.exists ? "" : "muted"}>
                {detail.worktree.exists ? "ON DISK" : "RECLAIMED"}
              </span>
            ) : (
              <span className="muted">—</span>
            )}
          </dd>
        </div>
      </dl>
    </header>
  );
}

/**
 * A run is blocked waiting for an answer. This has to be unmissable: until it is answered or
 * times out, a node is doing nothing, and the only thing that unblocks it is a person reading it.
 */
function Question({
  question,
  onAnswered,
}: {
  question: LiveEvent;
  onAnswered: (questionId: string) => void;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    box.current?.focus();
  }, [question.question_id]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || busy || !question.question_id) return;
    setBusy(true);
    setError(null);
    try {
      await answerQuestion(question.question_id, text);
      setText("");
      onAnswered(question.question_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="ask" onSubmit={(e) => void submit(e)}>
      <div className="sec ask-head">
        <span>
          /// {question.node ?? "a node"} is waiting on you
        </span>
        <span>{clock(question.at)}</span>
      </div>
      <p className="ask-q">{question.detail}</p>
      <textarea
        ref={box}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="your answer…"
        rows={2}
        spellCheck={false}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) void submit(e);
        }}
      />
      <button type="submit" disabled={busy || !text.trim()}>
        {busy ? "SENDING…" : ">>> ANSWER"}
      </button>
      {error && <p className="ask-error hazard">{error}</p>}
    </form>
  );
}

/**
 * What the run is doing right now. Scoped to the selected node when there is one, because during
 * the fork two nodes are genuinely concurrent and an interleaved feed is the hardest possible read.
 */
/** One line of the feed: an event, or a run of identical tool calls collapsed into one. */
interface Row {
  at: string;
  node: string | null;
  action: string;
  detail: string;
  /** The identifying argument, shown apart from the action so it can be styled as data. */
  arg?: string;
  /** How many identical calls this row stands for. 1 unless collapsed. */
  count: number;
  durationMs?: number;
  kind: string;
}

/**
 * Fold the raw stream into what a reader can scan.
 *
 * Three things happen here, all of them because a tool loop emits far more lines than it has
 * events worth reading: a call and its result become one row, a run of the same call collapses to
 * a count, and the argument that distinguishes one call from the next is kept.
 */
function rows(events: LiveEvent[]): Row[] {
  const out: Row[] = [];
  for (const e of events) {
    // A result is not its own line — it finishes the call above it. Tools run one at a time, so
    // the most recent matching call is the right one.
    if (e.kind === "tool_result") {
      const call = [...out].reverse().find((r) => r.kind === "tool_call" && r.node === e.node && r.action === e.detail);
      if (call && e.duration_ms !== undefined) call.durationMs = (call.durationMs ?? 0) + e.duration_ms;
      continue;
    }
    const last = out[out.length - 1];
    // Reading four files in a row is one thing a reader wants to know, not four.
    if (
      last &&
      e.kind === "tool_call" &&
      last.kind === "tool_call" &&
      last.node === e.node &&
      last.action === e.detail
    ) {
      last.count += 1;
      // The last argument, so the row still says where the run got to.
      if (e.arg) last.arg = e.arg;
      last.at = e.at;
      continue;
    }
    out.push({
      at: e.at,
      node: e.node,
      action: e.kind === "tool_call" ? e.detail : e.kind.replace("_", " "),
      detail: e.kind === "tool_call" ? "" : e.detail,
      ...(e.arg ? { arg: e.arg } : {}),
      count: 1,
      kind: e.kind,
    });
  }
  return out;
}

function Feed({ events, node }: { events: LiveEvent[]; node: string | null }) {
  const shown = useMemo(
    () => rows(node ? events.filter((e) => e.node === node) : events),
    [events, node],
  );
  const tail = useRef<HTMLDivElement>(null);

  useEffect(() => {
    tail.current?.scrollIntoView({ block: "end" });
  }, [shown.length]);

  return (
    <div className="feed">
      <div className="sec">
        <span>[ ACTIVITY {node ? `/ ${node.replace("_", " ")}` : ""} ]</span>
        <output>{shown.length}</output>
      </div>
      {shown.length === 0 && <p className="empty">no activity recorded yet</p>}
      {shown.map((r, i) => (
        <div className="ev" key={`${r.at}-${i}`}>
          <span className="ev-t">{clock(r.at)}</span>
          {/* Who, then what. A feed reads as a sentence about a node, not a list of verbs. */}
          {!node && <span className="ev-n">{r.node ?? "—"}</span>}
          <span className={`ev-k ev-k--${r.kind}`}>
            {r.action}
            {r.count > 1 && <span className="ev-x"> {r.count}×</span>}
          </span>
          {r.arg && <span className="ev-a">{r.arg}</span>}
          {r.detail && <span className="ev-d">{r.detail}</span>}
          {r.durationMs !== undefined && r.durationMs >= 1000 && (
            <span className="ev-ms">{(r.durationMs / 1000).toFixed(1)}s</span>
          )}
        </div>
      ))}
      <div ref={tail} />
    </div>
  );
}

function Detail({
  runId,
  node,
  checkpoints,
}: {
  runId: string | null;
  node: string | null;
  checkpoints: CheckpointView[] | null;
}) {
  if (!node) return <p className="empty">select a node to inspect its output</p>;
  if (checkpoints === null) return <p className="empty">loading {node}…</p>;
  if (checkpoints.length === 0)
    return <p className="empty">{node} has recorded no output</p>;

  return (
    <div>
      <div className="sec">
        <span>
          [ {node.replace("_", " ")} ] <samp>{short(runId)}</samp>
        </span>
        <output>
          {checkpoints.length}{" "}
          {checkpoints.length === 1 ? "checkpoint" : "checkpoints"}
        </output>
      </div>
      {checkpoints.map((c, i) => (
        <section className="iter" key={`${c.created_at}-${i}`}>
          {/* Every checkpoint, not just the last: for the implementer these are the converge
              iterations, and the progression between them is the interesting part. */}
          {checkpoints.length > 1 && (
            <div className="sec">
              <span>ITERATION {String(i + 1).padStart(2, "0")}</span>
              <span>{clock(c.created_at)}</span>
            </div>
          )}
          <pre>{JSON.stringify(c.output, null, 2)}</pre>
        </section>
      ))}
    </div>
  );
}

export default function App() {
  const [projects, setProjects] = useState<ProjectView[]>([]);
  const [project, setProject] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [node, setNode] = useState<string | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointView[] | null>(null);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  /** When the last live event arrived, as the liveness signal a checkpoint cannot give. */
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const [answered, setAnswered] = useState<ReadonlySet<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  // Which projects this dashboard watches. Fixed at startup, so fetched once.
  useEffect(() => {
    void (async () => {
      try {
        const found = await listProjects();
        setProjects(found);
        setProject((cur) => cur ?? found[0]?.slug ?? null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const refresh = useCallback(async () => {
    if (!project) return;
    try {
      const list = await listRuns(project);
      setRuns(list);
      setError(null);
      setRunId((cur) => cur ?? list[0]?.run_id ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [project]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Switching project means a different store entirely, so nothing selected carries over. This
  // has to happen in the same update as the project change: as a reactive effect it would run
  // *after* the fetch effects had already re-run, asking the new project for the old run's id.
  const onProject = useCallback((slug: string) => {
    setProject(slug);
    setRunId(null);
    setRuns([]);
    setDetail(null);
    setEvents([]);
    setError(null);
  }, []);

  // Follow a freshly started run immediately; its rows land in the store moments later.
  const onStarted = useCallback(
    (newRunId: string) => {
      setRunId(newRunId);
      setDetail(null);
      void refresh();
    },
    [refresh],
  );

  // What the view is actually on, so a slow response for something we've since left can be
  // dropped instead of overwriting the current one. Keyed by project as well as run: switching
  // project can leave a request for the same-named run in flight.
  const shown = useRef<string | null>(null);
  const showing = project && runId ? `${project}/${runId}` : null;
  useEffect(() => {
    shown.current = showing;
  }, [showing]);

  const load = useCallback(async () => {
    if (!runId || !project) return;
    try {
      const d = await getRun(project, runId);
      if (shown.current === `${project}/${runId}`) setDetail(d);
    } catch (e) {
      if (shown.current === `${project}/${runId}`) {
        setError(e instanceof Error ? e.message : String(e));
      }
    }
  }, [runId, project]);

  useEffect(() => {
    void load();
  }, [load]);

  // Follow the run's activity rather than polling for it. Checkpoints only say a node *finished*;
  // the stream is what shows a node working through a long turn. Node state still comes from the
  // store, so a checkpoint event is the cue to re-read it.
  useEffect(() => {
    if (!runId || !project) return;
    setEvents([]);
    setLastEventAt(null);
    setAnswered(new Set());
    const stop = followRun(project, runId, {
      onReset: () => setEvents([]),
      onEvent: (event) => {
        setEvents((prev) => [...prev.slice(-(FEED_LIMIT - 1)), event]);
        setLastEventAt(Date.now());
        if (event.kind === "checkpoint" || event.kind.startsWith("run_")) {
          void load();
          void refresh();
        }
      },
    });
    return stop;
  }, [runId, project, load, refresh]);

  // What a node announced when it started, plus its tool calls so far. A checkpoint carries the
  // same facts, but only once the node has stopped — this is what fills the box while it works.
  const live = useMemo(() => {
    const out = new Map<string, { facts?: NodeFacts; cycles: number; used: Set<string> }>();
    for (const e of events) {
      if (!e.node) continue;
      const at = out.get(e.node) ?? { cycles: 0, used: new Set<string>() };
      // A node_start means a fresh attempt: its counts start again.
      if (e.kind === "node_start" && e.facts) {
        out.set(e.node, { facts: e.facts, cycles: 0, used: new Set() });
        continue;
      }
      if (e.kind === "tool_call") {
        at.cycles += 1;
        // `detail` is the tool name for this kind.
        if (e.detail) at.used.add(e.detail);
      }
      out.set(e.node, at);
    }
    return out;
  }, [events]);

  /**
   * The node that last did something, from the stream.
   *
   * The store cannot answer this. It sees checkpoints, and mid-converge the implementer has one
   * while still being re-run — so it reads as working — while the verifier, which is an optional
   * stage and has not checkpointed, reads as not started. Both are the wrong way round exactly
   * when a viewer is watching the verifier work.
   */
  const active = useMemo(() => {
    const WORKING = new Set(["tool_call", "model_text", "node_start", "tool_result"]);
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      if (e?.node && WORKING.has(e.kind)) return e.node;
    }
    return null;
  }, [events]);

  useEffect(() => {
    setNode(null);
    setCheckpoints(null);
  }, [runId]);

  useEffect(() => {
    if (!runId || !node || !project) return;
    let cancelled = false;
    setCheckpoints(null);
    getNodeCheckpoints(project, runId, node)
      .then((cps) => {
        if (!cancelled) setCheckpoints(cps);
      })
      .catch(() => {
        if (!cancelled) setCheckpoints([]);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, node, project]);

  // Every question stands until its *own* resolution arrives — during the fork two nodes run
  // concurrently and can both be waiting, and resolving one must not hide the other. A viewer who
  // attaches mid-wait still sees them, because the stream replays history on connect.
  const open = new Map<string, LiveEvent>();
  for (const event of events) {
    if (!event.question_id) {
      // A run ending resolves anything still outstanding.
      if (event.kind.startsWith("run_")) open.clear();
      continue;
    }
    if (event.kind === "question") open.set(event.question_id, event);
    if (event.kind === "question_answered") open.delete(event.question_id);
  }
  // Answered in this tab: clear immediately rather than waiting for the run's event to come back
  // round through the log.
  for (const id of answered) open.delete(id);
  const pending = [...open.values()];

  return (
    <div className="shell">
      <div className="masthead">
        <h1>
          Ratatoskr Run Telemetry
        </h1>
        <span className="rev">
          {error ? (
            <span className="hazard">/// LINK DOWN — {error}</span>
          ) : (
            "REV 0.1 / LOCAL"
          )}
        </span>
      </div>

      <Rail
        projects={projects}
        project={project ?? ""}
        onProject={onProject}
        runs={runs}
        selected={runId}
        onSelect={setRunId}
        onStarted={onStarted}
      />

      <main className="stage stage--split">
        {detail ? (
          <>
            <RunMeta detail={detail} lastEventAt={lastEventAt} />
            <div className="graph">
              <PipelineGraph
                nodes={detail.nodes}
                live={live}
                active={active}
                selected={node}
                onSelect={setNode}
              />
            </div>
            <div className="lower">
              <div className="activity">
                {pending.map((question) => (
                  <Question
                    key={question.question_id}
                    question={question}
                    onAnswered={(id) =>
                      setAnswered((prev) => new Set(prev).add(id))
                    }
                  />
                ))}
                <Feed events={events} node={node} />
              </div>
              {node && (
                <div className="detail">
                  <Detail runId={runId} node={node} checkpoints={checkpoints} />
                </div>
              )}
            </div>
          </>
        ) : (
          <p className="empty">no run selected</p>
        )}
      </main>
    </div>
  );
}
