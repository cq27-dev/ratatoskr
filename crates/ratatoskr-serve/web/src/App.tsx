import { useCallback, useEffect, useRef, useState } from "react";
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

function RunMeta({ detail }: { detail: RunDetail }) {
  const stale =
    detail.status !== null &&
    LIVE.has(detail.status) &&
    detail.last_activity !== null &&
    Date.now() - Date.parse(detail.last_activity) > STALE_MS;

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
function Feed({ events, node }: { events: LiveEvent[]; node: string | null }) {
  const shown = node ? events.filter((e) => e.node === node) : events;
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
      {shown.map((e, i) => (
        <div className="ev" key={`${e.at}-${i}`}>
          <span className="ev-t">{clock(e.at)}</span>
          <span className={`ev-k ev-k--${e.kind}`}>{e.kind.replace("_", " ")}</span>
          {!node && <span className="ev-n">{e.node ?? "—"}</span>}
          <span className="ev-d">{e.detail}</span>
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
    setAnswered(new Set());
    const stop = followRun(project, runId, {
      onReset: () => setEvents([]),
      onEvent: (event) => {
        setEvents((prev) => [...prev.slice(-(FEED_LIMIT - 1)), event]);
        if (event.kind === "checkpoint" || event.kind.startsWith("run_")) {
          void load();
          void refresh();
        }
      },
    });
    return stop;
  }, [runId, project, load, refresh]);

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
          Ratatoskr<span className="hazard">®</span> Run Telemetry
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
            <RunMeta detail={detail} />
            <div className="graph">
              <PipelineGraph
                nodes={detail.nodes}
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
