import { useCallback, useEffect, useState } from "react";
import PipelineGraph from "./PipelineGraph";
import {
  LIVE,
  getNodeCheckpoints,
  getRun,
  listRuns,
  type CheckpointView,
  type RunDetail,
  type RunSummary,
} from "./api";

/** A live run with nothing recorded for this long is almost certainly dead, not busy. */
const STALE_MS = 120_000;
const POLL_MS = 3000;

const short = (id: string | null) => (id ? id.slice(0, 8) : "—");
const clock = (ts: string | null) => (ts ? ts.slice(11, 19) : "—");

function Rail({
  runs,
  selected,
  onSelect,
}: {
  runs: RunSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <nav className="rail">
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
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [node, setNode] = useState<string | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointView[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await listRuns();
      setRuns(list);
      setError(null);
      setRunId((cur) => cur ?? list[0]?.run_id ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll while the selected run is live. #39 replaces this with a pushed log tail; until then
  // this is deliberately coarse — checkpoint arrival is the only signal the store can give.
  const live = detail?.status != null && LIVE.has(detail.status);
  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    const load = async () => {
      try {
        const d = await getRun(runId);
        if (!cancelled) setDetail(d);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    };
    void load();
    // The cleanup must be registered on every path, including the non-polling one: without it
    // `cancelled` never flips, and a slow response for the previously selected run lands after
    // you've already switched, overwriting the newer one.
    let timer: ReturnType<typeof setInterval> | undefined;
    if (live) {
      timer = setInterval(() => {
        void load();
        void refresh();
      }, POLL_MS);
    }
    return () => {
      cancelled = true;
      if (timer !== undefined) clearInterval(timer);
    };
  }, [runId, live, refresh]);

  useEffect(() => {
    setNode(null);
    setCheckpoints(null);
  }, [runId]);

  useEffect(() => {
    if (!runId || !node) return;
    let cancelled = false;
    setCheckpoints(null);
    getNodeCheckpoints(runId, node)
      .then((cps) => {
        if (!cancelled) setCheckpoints(cps);
      })
      .catch(() => {
        if (!cancelled) setCheckpoints([]);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, node]);

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

      <Rail runs={runs} selected={runId} onSelect={setRunId} />

      <main className={node ? "stage stage--split" : "stage"}>
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
            {node && (
              <div className="detail">
                <Detail runId={runId} node={node} checkpoints={checkpoints} />
              </div>
            )}
          </>
        ) : (
          <p className="empty">no run selected</p>
        )}
      </main>
    </div>
  );
}
