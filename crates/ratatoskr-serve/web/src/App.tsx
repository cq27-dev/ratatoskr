import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQueryState } from "nuqs";
import { applyDerived, convergeLoops, nodesFromEvents, workingNodeNames } from "./derive";
import { pendingQuestions } from "./questions";
import {
  elapsedAt,
  indexAtElapsed,
  parseAsElapsed,
  readPath,
  sameRun,
  writePath,
} from "./url";
import Tooltips from "./ui/Tooltip";
import PipelineGraph from "./panels/PipelineGraph";
import { Detail } from "./panels/Detail";
import { Feed } from "./panels/Feed";
import { Question } from "./panels/Question";
import { Rail } from "./panels/Rail";
import { SignIn } from "./panels/SignIn";
import { RunMeta } from "./panels/RunMeta";
import { Scrubber } from "./panels/Scrubber";
import { Controls } from "./panels/Controls";
import { Steer } from "./panels/Steer";
import {
  getHistory,
  getNodeCheckpoints,
  getRun,
  followRun,
  listProjects,
  isTerminal,
  listRuns,
  mayAct,
  whoami,
  type CheckpointView,
  type ControlView,
  type LiveEvent,
  type NodeFacts,
  type Me,
  type ProjectView,
  type RunDetail,
  type RunSummary,
} from "./api";

/** A run nobody has touched. A constant so the identity is stable — as a literal it would be a new
 * object on every render, and the effect that seeds it would never stop firing. */
const EMPTY_CONTROL: ControlView = { paused: false, stopped: [], steering: [] };

/** How many live events to keep on screen. Old ones scroll away; the log file keeps everything. */
const FEED_LIMIT = 250;
/**
 * How often to re-read the run list while the tab is visible.
 *
 * A run started or deleted outside the dashboard produces no event to subscribe to, so this is
 * the only way the list learns about it. Short enough that a run started in a terminal appears
 * while you are still looking for it, long enough that a handful of rows is not worth streaming.
 */
const RUN_LIST_POLL_MS = 10_000;
/**
 * How long a burst of checkpoint events is gathered before the run's rows are re-read.
 *
 * Long enough to absorb a stream's replayed tail, short enough that a checkpoint landing while
 * you watch shows up about as promptly as the feed row announcing it.
 */
const SYNC_COALESCE_MS = 250;
export default function App() {
  const [projects, setProjects] = useState<ProjectView[]>([]);
  /**
   * Who the viewer is, or `{}` for nobody.
   *
   * `null` until the first answer, which is distinct from "signed out": drawing the sign-in
   * control before the server has said would flash SIGN IN at someone who is already signed in,
   * on every load.
   */
  const [me, setMe] = useState<Me | null>(null);
  /*
   * Project, run, node and position live in the address bar rather than in React.
   *
   * They are the whole answer to "what am I looking at", so they are what a link has to carry: a
   * reload lands where it was, and the address pasted into a message shows the same thing to
   * someone else. Everything else on this page is derived from these four and is fetched again on
   * its own. See url.ts for why they replace the history entry instead of pushing.
   *
   * Split across the two halves of the URL by what each one is. The project and the run are the
   * thing being looked at, so they are the path; the node and the position are views into it, so
   * they are query parameters. Ordinary React state here, seeded from the path once and written
   * back by the effect below — the path has two segments and no parsing worth a library.
   */
  const opened = useRef(readPath());
  const [project, setProject] = useState<string | null>(opened.current.project);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState<string | null>(opened.current.run);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  /** The run's whole event history, loaded once so a finished run can be moved through. */
  const [history, setHistory] = useState<LiveEvent[] | null>(null);
  /**
   * How far through `shown` the viewer has scrubbed; `null` means "follow the end".
   *
   * The index stays here and the address bar gets the elapsed time, one way only: written when the
   * viewer scrubs, read once when a run loads. A second holds several events, so a round trip
   * through `m:ss` does not land back on the index it started from — mirroring the two would make
   * the handle jump under the pointer. Throttled because dragging produces a position per frame
   * and each one would otherwise be a history write.
   */
  const [cursor, setCursor] = useState<number | null>(null);
  const [at, setAt] = useQueryState("at", parseAsElapsed.withOptions({ throttleMs: 300 }));
  const [node, setNode] = useQueryState("node");
  const [checkpoints, setCheckpoints] = useState<CheckpointView[] | null>(null);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  /** When the last live event arrived, as the liveness signal a checkpoint cannot give. */
  const [answered, setAnswered] = useState<ReadonlySet<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    writePath({ project, run: runId });
  }, [project, runId]);

  // Who is looking. Read before anything else, because it decides what the rest of these calls
  // are even allowed to return.
  useEffect(() => {
    void whoami()
      .then(setMe)
      .catch(() => setMe({}));
  }, []);

  /** Re-read everything after signing in or out: what may be seen has just changed. */
  const onIdentity = useCallback((next: Me) => {
    setMe(next);
    void listProjects()
      .then(setProjects)
      .catch(() => setProjects([]));
  }, []);

  // Which projects this dashboard watches. Fixed at startup, so fetched once.
  useEffect(() => {
    void (async () => {
      try {
        const found = await listProjects();
        setProjects(found);
        // Keeps a project named in the address bar, unless it is not one of ours — a stale
        // link should land somewhere real rather than on an empty dashboard.
        setProject((cur) =>
          cur && found.some((p) => p.slug === cur) ? cur : (found[0]?.slug ?? null),
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, [me]);

  const refresh = useCallback(async () => {
    if (!project) return;
    try {
      const list = await listRuns(project);
      setRuns(list);
      setError(null);
      // Keep the current selection when it still exists, fall back to the newest when it does
      // not. A run deleted by `ratatoskr runs rm` would otherwise stay selected: its row is gone
      // from the list while the detail pane goes on showing it.
      // `startsWith`, not equality: a link carries an eight-character run id, so what is selected
      // on a cold open is a prefix. Matching it here replaces it with the full id, which is what
      // the rest of the page compares against — and the address bar keeps showing the short form
      // either way, since that is what `writePath` writes.
      setRunId((cur) => {
        const known = cur && list.find((r) => r.run_id.startsWith(cur));
        return known ? known.run_id : (list[0]?.run_id ?? null);
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [project]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // The run list changes outside this tab and nothing tells us: `ratatoskr run` in a terminal adds
  // one, `runs rm` removes one, and neither goes through the dashboard. Without this the list is
  // whatever it was when the project was selected, so a run in flight is invisible until a reload
  // and a deleted run lingers indefinitely.
  //
  // Polled rather than streamed because there is no event for "the set of runs changed" — the
  // per-run stream only exists once you know a run to subscribe to, which is the thing being
  // missed. Only while the tab is visible, and immediately on becoming visible again, so a
  // backgrounded dashboard costs nothing and a returning one is current at once.
  useEffect(() => {
    if (!project) return;
    const tick = () => {
      if (document.visibilityState === "visible") void refresh();
    };
    const id = window.setInterval(tick, RUN_LIST_POLL_MS);
    document.addEventListener("visibilitychange", tick);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [project, refresh]);

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

  /*
   * What the view is actually on, so a slow response for something we've since left can be dropped
   * instead of overwriting the current one. Keyed by project as well as run: switching project can
   * leave a request for the same-named run in flight.
   *
   * Compared with `sameRun` rather than by string. On a cold open the request goes out under the
   * eight-character id from the link and the run list expands it to the full one moments later —
   * so an exact comparison declared the reply stale and dropped the history the page was waiting
   * for, leaving the scrubber on the live tail with no way back.
   */
  const shown = useRef<{ project: string; run: string } | null>(null);
  useEffect(() => {
    shown.current = project && runId ? { project, run: runId } : null;
  }, [project, runId]);
  const stillShowing = (forProject: string, forRun: string) =>
    shown.current?.project === forProject && sameRun(shown.current.run, forRun);

  const load = useCallback(async () => {
    if (!runId || !project) return;
    try {
      const d = await getRun(project, runId);
      if (stillShowing(project, runId)) setDetail(d);
    } catch (e) {
      if (stillShowing(project, runId)) {
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
      if (stillShowing(project, runId)) setHistory(h);
    } catch {
      // A run whose log has rotated away and was never ingested has no timeline. The live feed
      // still works, and the boxes fall back to what the store knows.
      if (stillShowing(project, runId)) setHistory([]);
    }
  }, [runId, project]);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  // Follow the run's activity rather than polling for it. Checkpoints only say a node *finished*;
  // the stream is what shows a node working through a long turn. Node state still comes from the
  // store, so a checkpoint event is the cue to re-read it.
  /**
   * Re-read the run's own rows: its detail, the run list, and the history.
   *
   * Coalesced, because these are triggered by checkpoints arriving on the stream and checkpoints
   * do not arrive one at a time. On connect the stream replays a tail, so every checkpoint in it
   * lands at once — a run with fourteen of them asked for twenty-four fetches inside thirty
   * milliseconds, all for the same answer, every time a different run was clicked.
   *
   * A burst now schedules one read. A run that genuinely checkpoints twice in a quarter second
   * also gets one, which is the same answer a viewer wanted and a sixth of the requests.
   */
  const syncing = useRef<number | null>(null);
  const sync = useCallback(() => {
    if (syncing.current !== null) return;
    syncing.current = window.setTimeout(() => {
      syncing.current = null;
      void load();
      void refresh();
      // A long run outgrows the live window while it runs; re-reading here keeps the timeline
      // whole without polling for it.
      void loadHistory();
    }, SYNC_COALESCE_MS);
  }, [load, refresh, loadHistory]);
  useEffect(
    () => () => {
      if (syncing.current !== null) window.clearTimeout(syncing.current);
    },
    [],
  );

  useEffect(() => {
    if (!runId || !project) return;
    setEvents([]);
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
        // A checkpoint or a status change means the run's own rows moved, so they are re-read —
        // once per burst rather than once per event. See `sync`.
        if (event.kind === "checkpoint" || event.kind.startsWith("run_")) sync();
      },
    });
    return stop;
  }, [runId, project, sync]);

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

  /**
   * Which node each event belongs to, in timeline order — what colours the scrubber's track.
   *
   * Its own memo rather than derived inside `Scrubber`: the timeline is the longest list on the
   * page and this walks all of it, so recomputing on every cursor move would cost a pass per drag
   * frame for a value that does not change while dragging.
   */
  const timelineNodes = useMemo(() => timeline.map((e) => e.node ?? null), [timeline]);

  /**
   * A run has been selected and its rows have not arrived.
   *
   * Both halves matter: the detail carries the run's own row and the history carries its events,
   * and either arriving alone still leaves the page half-built.
   */
  const loading = runId !== null && (detail === null || history === null);

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
   * question than the one being asked. The pipeline's shape — which nodes exist and where — comes
   * from the server, because that is a property of the graph, not of a moment; a node the shape
   * does not place is positioned from the stream, which is the only thing that has seen it yet.
   *
   * The one thing the store answers better is where a stopped run stopped. A host error writes no
   * checkpoint and no node-scoped event, so the stream leaves the dying node working forever; at
   * the end of a run that has stopped, the store settles it. Passed only when the cursor is at the
   * live end — mid-run the stream is still the authority. See `applyDerived`.
   */
  const ended = detail && cursor === null && isTerminal(detail.status) ? detail.status : null;
  const graphNodes = useMemo(() => {
    if (!detail) return [];
    // Having a timeline is what makes the stream authoritative — not whether this position has
    // reached a node yet. At the very start of a run the only event is the issue checkpoint, which
    // is not a pipeline node, so the derivation is legitimately empty; falling back to the store
    // there showed every node finished, with its final counts, at step one of the run.
    if (!shownEvents.length) return detail.nodes;
    return applyDerived(detail.nodes, nodesFromEvents(shownEvents), ended);
  }, [detail, shownEvents, ended]);

  /**
   * Which nodes are working right now — what stop and steer can be aimed at.
   *
   * From the live pipeline rather than the scrubbed view: scrubbing back through a run does not
   * change what it is doing this second, and a control aimed at where the slider happens to sit
   * would stop a node that finished ten minutes ago.
   */
  const workingNodes = useMemo(
    () => workingNodeNames(detail?.nodes ?? [], timeline),
    [detail, timeline],
  );

  /**
   * What the operator has asked of this run.
   *
   * Seeded from the server and updated by each command's reply, so a click shows at once instead
   * of waiting for the next poll — and a reload, or a second tab, still sees the same pause.
   */
  const [controlState, setControlState] = useState<ControlView>(EMPTY_CONTROL);
  /** Whether the message box is open. Down with the questions, not up by the button that opens it. */
  const [composing, setComposing] = useState(false);
  useEffect(() => {
    setControlState(detail?.control ?? EMPTY_CONTROL);
  }, [detail]);
  useEffect(() => {
    if (!workingNodes.length) setComposing(false);
  }, [workingNodes]);

  /**
   * Characters the node column reserves, from the longest name this run can ever show.
   *
   * Both sources matter and neither alone is enough. The graph's nodes are fixed when the run
   * starts, so they are stable — but the feed also carries rows for things that are not graph
   * nodes, `issue` among them. The whole timeline covers those, and using the whole of it rather
   * than the scrubbed prefix is what stops the column resizing as you drag.
   */
  const nameWidth = useMemo(() => {
    let n = 0;
    for (const g of graphNodes) n = Math.max(n, g.name.length);
    for (const e of timeline) if (e.node) n = Math.max(n, e.node.length);
    // A floor, so a run whose events have not loaded yet does not start at zero and jump.
    return Math.max(n, 8);
  }, [graphNodes, timeline]);

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
   * How many times the implementer was re-entered, split by the route that brought it back.
   *
   * Folded from the same prefix the boxes are folded from, and for the same reason: edge state
   * taken from anywhere else drifts out of step with the nodes beside it, which is how the
   * converge loop came to glow while a different node was working. Never from checkpoint counts —
   * a traversal is a re-entry, so counting the implementer's rows overstates it by the initial
   * `implement()` call.
   */
  const loops = useMemo(() => convergeLoops(shownEvents), [shownEvents]);

  /*
   * Leaving a run drops everything read for it.
   *
   * `seenRun` is what tells a switch from the first render. On mount this effect fires with the
   * run the address bar named, and `node` and `at` already holding the rest of that link — the
   * unconditional reset this used to be threw both away, so every deep link resolved to the run
   * and nothing inside it.
   */
  const seenRun = useRef<string | null>(null);
  useEffect(() => {
    setCheckpoints(null);
    setHistory(null);
    // `sameRun`, not `!==`. A link names a run in eight characters and the run list names it in
    // thirty-six, so the id changes once on a cold open as the prefix is expanded — and reading
    // that as a switch cleared the node and position the link had just supplied.
    if (seenRun.current !== null && !sameRun(seenRun.current, runId)) {
      void setNode(null);
      setCursor(null);
      void setAt(null);
    }
    seenRun.current = runId;
  }, [runId, setNode, setAt]);

  /**
   * Put the scrubber where a link said, once the run it belongs to has arrived.
   *
   * Once per run, and only from the address bar's opening value: after that the position is the
   * viewer's and `at` follows it rather than the other way round.
   */
  const seededRun = useRef<string | null>(null);
  useEffect(() => {
    if (!runId || history === null || sameRun(seededRun.current, runId)) return;
    seededRun.current = runId;
    if (at !== null) setCursor(indexAtElapsed(timeline, at));
  }, [runId, history, timeline, at]);

  /** Scrubbing is the only thing that moves the position, so it is the only thing that writes it. */
  const onScrub = useCallback(
    (next: number | null) => {
      setCursor(next);
      void setAt(next === null ? null : elapsedAt(timeline, next));
    },
    [timeline, setAt],
  );


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

  const pending = pendingQuestions(events, answered);

  return (
    <div className="shell">
      {/* Once, at the root: every `data-tip` on the page is served by this one element. */}
      <Tooltips />
      <div className="masthead">
        <h1>
          Ratatoskr
        </h1>
        <SignIn me={me} onChange={onIdentity} />
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
        mayAct={mayAct(me?.role)}
      />

      <main className="stage stage--split">
        {detail ? (
          <>
            <RunMeta detail={detail} lastEvent={timeline[timeline.length - 1]?.at ?? null} />
            <Scrubber
              total={timeline.length}
              cursor={cursor}
              at={shownEvents.length ? shownEvents[shownEvents.length - 1]!.at : null}
              nodes={timelineNodes}
              startedAt={timeline[0]?.at ?? null}
              onScrub={onScrub}
              controls={
                // Only for a run still executing. A finished run has nothing to pause, and
                // controls that could never do anything are worse than none: they invite a click
                // and then explain themselves.
                project && runId && !isTerminal(detail.status) ? (
                  <Controls
                    project={project}
                    runId={runId}
                    state={controlState}
                    working={workingNodes}
                    mayAct={mayAct(me?.role)}
                    onChange={setControlState}
                    onCompose={() => setComposing((open) => !open)}
                  />
                ) : undefined
              }
            />
            <div className="graph">
              <PipelineGraph
                nodes={graphNodes}
                live={live}
                loops={loops}
                selected={node}
                onSelect={setNode}
              />
            </div>
            <div className="lower">
              <div className="activity">
                {composing && project && runId && (
                  <Steer
                    project={project}
                    runId={runId}
                    working={workingNodes}
                    onSent={() => setComposing(false)}
                    onDismiss={() => setComposing(false)}
                  />
                )}
                {pending.map((question) => (
                  <Question
                    key={question.question_id}
                    question={question}
                    onAnswered={(id) =>
                      setAnswered((prev) => new Set(prev).add(id))
                    }
                  />
                ))}
                <Feed events={shownEvents} node={node} nameWidth={nameWidth} loading={loading} />
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
