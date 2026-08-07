import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { applyDerived, nodesFromEvents } from "./derive";
import PipelineGraph from "./PipelineGraph";
import {
  LIVE,
  getHistory,
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
  /**
   * The calls this row folded together, in order.
   *
   * Collapsing answers "what was it doing" — four reads in a row is one fact. It also hides which
   * four, which is the next question a reader has, so the parts are kept rather than counted.
   */
  parts: RowPart[];
}

interface RowPart {
  at: string;
  arg?: string;
  durationMs?: number;
}

/**
 * Fold the raw stream into what a reader can scan.
 *
 * Three things happen here, all of them because a tool loop emits far more lines than it has
 * events worth reading: a call and its result become one row, a run of the same call collapses to
 * a count, and the argument that distinguishes one call from the next is kept.
 */
/**
 * Move through a run's timeline.
 *
 * Scrubbing is a prefix of the event stream: everything the view shows — the boxes, the
 * highlighting, the feed — is a fold over the events, so cutting the list short IS the historical
 * view. There is no replay engine and no second code path to keep in step with the live one.
 */
function Scrubber({
  total,
  cursor,
  at,
  onScrub,
}: {
  total: number;
  cursor: number | null;
  at: string | null;
  onScrub: (cursor: number | null) => void;
}) {
  // Nothing to move through until the history has loaded.
  if (total < 2) return null;
  const position = cursor ?? total - 1;
  const following = cursor === null;
  return (
    <div className="scrub">
      <button
        type="button"
        className={following ? "scrub-live is-live" : "scrub-live"}
        onClick={() => onScrub(following ? position : null)}
        title={following ? "Following the end of the run" : "Return to the end of the run"}
      >
        {/* Both words are six characters, so the button is the same size in either state and the
            slider beside it does not change length when the mode flips. */}
        {following ? "FOLLOW" : "REPLAY"}
      </button>
      <input
        type="range"
        min={0}
        max={total - 1}
        value={position}
        onChange={(e) => {
          const next = Number(e.target.value);
          // Landing on the last event means following again, so the view resumes on its own
          // rather than freezing one event short of the present.
          onScrub(next >= total - 1 ? null : next);
        }}
        aria-label="Position in the run"
      />
      <span className="scrub-at" title={at ?? undefined}>
        {/* Padded to the total's width: unpadded, the label is narrower at 1/654 than at 654/654
            and the slider changes length as you drag it. */}
        {String(position + 1).padStart(String(total).length, "0")}/{total}
        {at ? ` · ${at.slice(11, 19)}` : ""}
      </span>
    </div>
  );
}

/** Kinds that are measurements rather than actions: they fill the node boxes, not the feed. */
const TELEMETRY = new Set(["usage"]);

/**
 * Filler the endpoint's harness emits, not the model.
 *
 * Requests go through a proxy that runs each one inside a Claude Code subprocess, and that CLI
 * writes this when a turn ends having only called tools — the string is in its binary, not in
 * anything here. It is still recorded in the log, which is the provenance record; the feed is a
 * reading aid, and a line that says nothing costs a row and breaks the run of identical calls
 * around it, so two reads of the same file stop collapsing into one.
 */
const HARNESS_FILLER = "No response requested.";

function rows(events: LiveEvent[]): Row[] {
  const out: Row[] = [];
  for (const e of events) {
    // A `usage` event is the node's cost, which the box reports. As a feed line it reads as
    // "bookkeeper / usage / node usage" — three words for a number that is already on screen.
    if (TELEMETRY.has(e.kind)) continue;
    // The filler is sometimes the whole message and sometimes a prefix on real output, so strip
    // it rather than dropping the line — discarding the row would lose what the model actually
    // said in the second case.
    let detail = e.detail;
    if (e.kind === "model_text" && detail.startsWith(HARNESS_FILLER)) {
      detail = detail.slice(HARNESS_FILLER.length).trimStart();
      if (!detail) continue;
    }
    // A result is not its own line — it finishes the call above it. Tools run one at a time, so
    // the most recent matching call is the right one.
    if (e.kind === "tool_result") {
      const call = [...out].reverse().find((r) => r.kind === "tool_call" && r.node === e.node && r.action === e.detail);
      if (call && e.duration_ms !== undefined) {
        call.durationMs = (call.durationMs ?? 0) + e.duration_ms;
        // The row's total is the sum; the part's is its own, so an expanded group shows which of
        // the calls was the slow one.
        const part = call.parts[call.parts.length - 1];
        if (part) part.durationMs = e.duration_ms;
      }
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
      last.parts.push({ at: e.at, ...(e.arg ? { arg: e.arg } : {}) });
      continue;
    }
    const action = e.kind === "tool_call" ? e.detail : e.kind.replace("_", " ");
    out.push({
      at: e.at,
      node: e.node,
      action,
      // A message that only restates the action is not a second column: `checkpoint checkpoint`.
      detail: e.kind === "tool_call" || detail === action ? "" : detail,
      ...(e.arg ? { arg: e.arg } : {}),
      count: 1,
      kind: e.kind,
      parts: [{ at: e.at, ...(e.arg ? { arg: e.arg } : {}) }],
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
  // Which collapsed rows are open, by their own key rather than index: the feed grows from the
  // end, and an index would move the open row out from under the reader.
  const [open, setOpen] = useState<ReadonlySet<string>>(new Set());

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
      {shown.map((r, i) => {
        const key = `${r.at}-${i}`;
        const grouped = r.count > 1;
        const expanded = open.has(key);
        const body = (
          <>
            <span className="ev-t">{clock(r.at)}</span>
            {/* Who, then what. A feed reads as a sentence about a node, not a list of verbs. */}
            {!node && <span className="ev-n">{r.node ?? "—"}</span>}
            <span className={`ev-k ev-k--${r.kind}`}>
              {r.action}
              {grouped && (
                <span className="ev-x">
                  {" "}
                  {r.count}× {expanded ? "▾" : "▸"}
                </span>
              )}
            </span>
            {r.arg && <span className="ev-a">{r.arg}</span>}
            {r.detail && <span className="ev-d">{r.detail}</span>}
            {r.durationMs !== undefined && r.durationMs >= 1000 && (
              <span className="ev-ms">{(r.durationMs / 1000).toFixed(1)}s</span>
            )}
          </>
        );
        return (
          <div key={key}>
            {grouped ? (
              // A real button, so it is reachable by keyboard and announces its state. A row that
              // stands for several calls is the one place the feed hides something.
              <button
                type="button"
                className="ev ev--group"
                aria-expanded={expanded}
                onClick={() =>
                  setOpen((prev) => {
                    const next = new Set(prev);
                    if (!next.delete(key)) next.add(key);
                    return next;
                  })
                }
              >
                {body}
              </button>
            ) : (
              <div className="ev">{body}</div>
            )}
            {grouped &&
              expanded &&
              r.parts.map((p, n) => (
                <div className="ev ev--sub" key={`${key}-${n}`}>
                  <span className="ev-t">{clock(p.at)}</span>
                  {!node && <span className="ev-n" />}
                  <span className="ev-k ev-k--sub">{r.action}</span>
                  {p.arg && <span className="ev-a">{p.arg}</span>}
                  {p.durationMs !== undefined && p.durationMs >= 1000 && (
                    <span className="ev-ms">{(p.durationMs / 1000).toFixed(1)}s</span>
                  )}
                </div>
              ))}
          </div>
        );
      })}
      <div ref={tail} />
    </div>
  );
}

function Detail({
  runId,
  node,
  checkpoints,
  until,
}: {
  runId: string | null;
  node: string | null;
  checkpoints: CheckpointView[] | null;
  /** While scrubbing, the moment being looked at: later checkpoints have not happened yet. */
  until: string | null;
}) {
  if (!node) return <p className="empty">select a node to inspect its output</p>;
  if (checkpoints === null) return <p className="empty">loading {node}…</p>;

  // The store returns every checkpoint a node ever wrote, which is its state at the END of the
  // run. Showing that against a scrubbed position would put output on screen that the run had not
  // produced yet — and for the implementer, whose iterations are the interesting part, it would
  // show the final answer while the map says it is still working.
  const shown = until
    ? checkpoints.filter((c) => Date.parse(c.created_at) <= Date.parse(until))
    : checkpoints;

  if (shown.length === 0) {
    return (
      <p className="empty">
        {node} {until ? "had recorded no output by this point" : "has recorded no output"}
      </p>
    );
  }

  return (
    <div>
      <div className="sec">
        <span>
          [ {node.replace("_", " ")} ] <samp>{short(runId)}</samp>
        </span>
        <output>
          {shown.length} {shown.length === 1 ? "checkpoint" : "checkpoints"}
          {until && shown.length < checkpoints.length ? ` of ${checkpoints.length}` : ""}
        </output>
      </div>
      {shown.map((c, i) => (
        <section className="iter" key={`${c.created_at}-${i}`}>
          {/* Every checkpoint, not just the last: for the implementer these are the converge
              iterations, and the progression between them is the interesting part. */}
          {shown.length > 1 && (
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
  /** The run's whole event history, loaded once so a finished run can be moved through. */
  const [history, setHistory] = useState<LiveEvent[] | null>(null);
  /** How far through `shown` the viewer has scrubbed; `null` means "follow the end". */
  const [cursor, setCursor] = useState<number | null>(null);
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

  /**
   * Re-read the run's whole history.
   *
   * Both sides of the timeline are windows: the client keeps the last `FEED_LIMIT` live events, and
   * the server replays a bounded tail to each new connection. History is the only complete account,
   * so anything that can leave a hole between those windows — a long run, a dropped stream, a
   * server restart — has to re-read it or the gap is permanent. A node whose checkpoint falls in
   * such a hole is shown working forever, because the fold never sees it finish.
   */
  const historyReadAt = useRef(0);
  const loadHistory = useCallback(async () => {
    if (!runId || !project) return;
    // Attaching replays a tail that contains every checkpoint the run has written, and each one
    // asks for the history again — a dozen reads of the same few hundred KB in one second. One
    // read covers all of them, so the rest are skipped rather than fetched and discarded.
    const now = Date.now();
    if (now - historyReadAt.current < 5_000) return;
    historyReadAt.current = now;
    try {
      const h = await getHistory(project, runId);
      if (shown.current === `${project}/${runId}`) setHistory(h);
    } catch {
      // A run whose log has rotated away and was never ingested has no timeline. The live feed
      // still works, and the boxes fall back to what the store knows.
      if (shown.current === `${project}/${runId}`) setHistory([]);
    }
  }, [runId, project]);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  // Follow the run's activity rather than polling for it. Checkpoints only say a node *finished*;
  // the stream is what shows a node working through a long turn. Node state still comes from the
  // store, so a checkpoint event is the cue to re-read it.
  useEffect(() => {
    if (!runId || !project) return;
    setEvents([]);
    setLastEventAt(null);
    setAnswered(new Set());
    const stop = followRun(project, runId, {
      onReset: () => {
        setEvents([]);
        // The stream restarted, so the live window starts over from wherever it replays. Whatever
        // it does not replay is only in history, which has to be re-read to cover it.
        void loadHistory();
      },
      onEvent: (event) => {
        setEvents((prev) => [...prev.slice(-(FEED_LIMIT - 1)), event]);
        setLastEventAt(Date.now());
        if (event.kind === "checkpoint" || event.kind.startsWith("run_")) {
          void load();
          void refresh();
          // A long run outgrows the live window while it runs; re-reading here keeps the timeline
          // whole without polling for it.
          void loadHistory();
        }
      },
    });
    return stop;
  }, [runId, project, load, refresh, loadHistory]);

  /**
   * The run's whole timeline: its history, plus anything the stream has delivered since.
   *
   * The history is read once and the stream keeps arriving, so the two overlap; timestamps are
   * ISO-8601 and sort lexicographically, which is enough to take only the genuinely newer ones.
   */
  const timeline = useMemo(() => {
    if (!history?.length) return events;
    const last = history[history.length - 1]!.at;
    return [...history, ...events.filter((e) => e.at > last)];
  }, [history, events]);

  /** The instant being looked at while scrubbing; `null` when following the run's end. */
  const shownEvents = useMemo(
    () => (cursor === null ? timeline : timeline.slice(0, cursor + 1)),
    [timeline, cursor],
  );
  const shownAt = cursor === null ? null : (shownEvents[shownEvents.length - 1]?.at ?? null);

  /**
   * Every node's box, rebuilt from the stream rather than read from the store.
   *
   * The store holds each node's LATEST row, so at any point but the end it answers a different
   * question than the one being asked. Only the pipeline's shape — which nodes exist and where —
   * still comes from the server, because that is a property of the graph, not of a moment.
   */
  const graphNodes = useMemo(() => {
    if (!detail) return [];
    // Having a timeline is what makes the stream authoritative — not whether this position has
    // reached a node yet. At the very start of a run the only event is the issue checkpoint, which
    // is not a pipeline node, so the derivation is legitimately empty; falling back to the store
    // there showed every node finished, with its final counts, at step one of the run.
    if (!shownEvents.length) return detail.nodes;
    return applyDerived(detail.nodes, nodesFromEvents(shownEvents));
  }, [detail, shownEvents]);

  // What a node announced when it started, plus its tool calls so far. A checkpoint carries the
  // same facts, but only once the node has stopped — this is what fills the box while it works.
  const live = useMemo(() => {
    const out = new Map<string, { facts?: NodeFacts; cycles: number; used: Set<string> }>();
    for (const e of shownEvents) {
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
  }, [shownEvents]);

  /**
   * Every node currently working, from the stream.
   *
   * The store cannot answer this. It sees checkpoints, and mid-converge the implementer has one
   * while still being re-run — so it reads as working — while the verifier, which is an optional
   * stage and has not checkpointed, reads as not started. Both are the wrong way round exactly
   * when a viewer is watching the verifier work.
   *
   * A SET, not the latest speaker: the bookkeeper and the publisher run concurrently at the end of
   * a run, and taking whoever spoke last made the highlight alternate between them as their events
   * interleaved. A node is working once it acts and stops when it checkpoints — which is the run's
   * own meaning of the word, and holds however many are in flight.
   */
  const active = useMemo(() => {
    const WORKING = new Set(["tool_call", "model_text", "node_start", "tool_result"]);
    const working = new Set<string>();
    for (const e of shownEvents) {
      if (!e?.node) continue;
      if (WORKING.has(e.kind)) working.add(e.node);
      // Its checkpoint is the node saying it is finished. The implementer checkpoints once per
      // converge iteration and is then re-driven, which re-adds it on its next event.
      else if (e.kind === "checkpoint") working.delete(e.node);
    }
    return working;
  }, [shownEvents]);

  useEffect(() => {
    setNode(null);
    setCheckpoints(null);
    setHistory(null);
    setCursor(null);
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
            <Scrubber
              total={timeline.length}
              cursor={cursor}
              at={shownEvents.length ? shownEvents[shownEvents.length - 1]!.at : null}
              onScrub={setCursor}
            />
            <div className="graph">
              <PipelineGraph
                nodes={graphNodes}
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
                <Feed events={shownEvents} node={node} />
              </div>
              {node && (
                <div className="detail">
                  <Detail
                    runId={runId}
                    node={node}
                    checkpoints={checkpoints}
                    until={shownAt}
                  />
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
