import { useEffect, useMemo, useRef, useState } from "react";
import { Json, Prose } from "../ui/format";
import { RowIcon, kindLabel } from "../ui/tools";
import { accent, prose, rowTint } from "../ui/tint";
import { clock } from "../ui/text";
import type { LiveEvent } from "../api";

/**
 * How long a load may take before it is worth drawing a placeholder for.
 *
 * Measured: switching runs against a local store settles between 40 and 300 milliseconds, so this
 * sits above nearly all of it. The placeholder is for the case that is actually slow — a cold
 * store, a long history, a project on another disk — and showing it for the ordinary case would
 * trade a quiet pause for a flicker.
 */
const SKELETON_AFTER_MS = 350;

/**
 * `flag`, but only after it has been true for `ms`.
 *
 * A skeleton that appears for fifty milliseconds is worse than no skeleton: it reads as a flicker,
 * and the eye reports it as the page breaking rather than the page working. Switching runs
 * normally settles well inside this, so nothing is drawn at all — the placeholder is for the cold
 * store, the large history, the project on a slow disk.
 */
function useSettled(flag: boolean, ms: number): boolean {
  const [on, setOn] = useState(false);
  useEffect(() => {
    if (!flag) {
      setOn(false);
      return;
    }
    const id = window.setTimeout(() => setOn(true), ms);
    return () => window.clearTimeout(id);
  }, [flag, ms]);
  return on;
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
  /** A producer-provided tool-call summary. */
  subject?: string;
  /** The tool call's bounded structured arguments. */
  args?: unknown;
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
  subject?: string;
  args?: unknown;
  durationMs?: number;
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

/**
 * Kinds whose text was *written* by something, rather than captured from somewhere.
 *
 * A model's prose and this pipeline's own notices are both composed for a reader, backticks and
 * all — the output-token warning quotes `output_tokens` and a `RUST_LOG=` invocation, and reading
 * them as literal text wastes the markers their author put there on purpose.
 *
 * Everything else is captured: an acceptance step's output, a tool result, a path. Those contain
 * asterisks and underscores because build output does, not because anyone meant emphasis, and
 * formatting them would invent structure the producer never asked for.
 */
const PROSE = new Set(["model_text", "event"]);

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
      // The summary retains the latest explicit subject; every call keeps its own arguments below.
      if (e.subject !== undefined) last.subject = e.subject;
      last.args = e.args;
      last.at = e.at;
      last.parts.push({
        at: e.at,
        ...(e.subject !== undefined ? { subject: e.subject } : {}),
        ...(e.args !== undefined ? { args: e.args } : {}),
      });
      continue;
    }
    const action = e.kind === "tool_call" ? e.detail : kindLabel(e.kind);
    out.push({
      at: e.at,
      node: e.node,
      action,
      // A message that only restates the action is not a second column: `checkpoint checkpoint`.
      detail: e.kind === "tool_call" || detail === action ? "" : detail,
      ...(e.subject !== undefined ? { subject: e.subject } : {}),
      ...(e.args !== undefined ? { args: e.args } : {}),
      count: 1,
      kind: e.kind,
      parts: [
        {
          at: e.at,
          ...(e.subject !== undefined ? { subject: e.subject } : {}),
          ...(e.args !== undefined ? { args: e.args } : {}),
        },
      ],
    });
  }
  return out;
}

function Arguments({ args }: { args: unknown }) {
  return (
    <details className="ev-args">
      <summary>arguments</summary>
      <Json value={args} />
    </details>
  );
}

/**
 * Placeholder rows, shaped like the feed they stand in for.
 *
 * Bars at the columns' own widths rather than a generic block: the point of a skeleton is that the
 * page does not jump when the content lands, and that only holds if the placeholder occupies the
 * same geometry. Widths vary per row because real messages do, and a stack of identical bars reads
 * as a broken table rather than as text arriving.
 */
function Skeleton() {
  const widths = ["62%", "48%", "77%", "35%", "68%", "54%", "83%", "41%"];
  return (
    <div aria-busy="true" aria-label="loading the run's activity">
      {widths.map((w, i) => (
        <div className="ev ev--skel" key={i}>
          <span className="ev-t skel" />
          <span className="ev-n skel" />
          <span className="ev-k skel" style={{ width: "7ch" }} />
          <span className="ev-d skel" style={{ width: w }} />
        </div>
      ))}
    </div>
  );
}

export function Feed({
  events,
  node,
  nameWidth,
  loading,
}: {
  events: LiveEvent[];
  node: string | null;
  /** Characters to reserve for the node column, so every row's icon starts at the same x. */
  nameWidth: number;
  /** The run's history has been asked for and has not arrived. */
  loading: boolean;
}) {
  const shown = useMemo(
    () => rows(node ? events.filter((e) => e.node === node) : events),
    [events, node],
  );
  // Delayed, so a switch that lands in fifty milliseconds shows no placeholder at all.
  const waiting = useSettled(loading, SKELETON_AFTER_MS);
  const tail = useRef<HTMLDivElement>(null);
  // Which collapsed rows are open, by their own key rather than index: the feed grows from the
  // end, and an index would move the open row out from under the reader.
  const [open, setOpen] = useState<ReadonlySet<string>>(new Set());

  useEffect(() => {
    tail.current?.scrollIntoView({ block: "end" });
  }, [shown.length]);

  return (
    // The node column is sized here rather than by the widest row on screen: sized by content it
    // would step sideways the moment a longer name streamed in, and a feed that moves while you
    // read it is worse than one that is a little wide.
    <div className="feed" style={{ "--node-col": `${nameWidth}ch` } as React.CSSProperties}>
      <div className="sec">
        <span>[ ACTIVITY {node ? `/ ${node.replace("_", " ")}` : ""} ]</span>
        <output>{shown.length}</output>
      </div>
      {/* Three states, not two. "Nothing happened" is a claim about the run and is only made once
          the run has been read — it used to be shown while loading, which asserted it of a run
          nobody had looked at yet. A load too short to be worth a placeholder shows neither: an
          empty pause reads as fast, and a placeholder that appears for a moment reads as a fault. */}
      {shown.length === 0 && !loading && <p className="empty">no activity recorded yet</p>}
      {shown.length === 0 && waiting && <Skeleton />}
      {shown.map((r, i) => {
        const key = `${r.at}-${i}`;
        const grouped = r.count > 1;
        const expanded = open.has(key);
        const body = (
          <>
            <span className="ev-t">{clock(r.at)}</span>
            {/* Who, then what. A feed reads as a sentence about a node, not a list of verbs. */}
            {!node && (
              <span className="ev-n" style={r.node ? { color: accent(r.node) } : undefined}>
                {r.node ?? "—"}
              </span>
            )}
            <span className={`ev-k ev-k--${r.kind}`}>
              {/* A tool call carries the icon that tool has on a node's box, so the call and the
                  capability it came from read as the same thing; every other row carries its
                  kind's. */}
              <RowIcon kind={r.kind} action={r.action} />
              {r.action}
              {grouped && (
                <span className="ev-x">
                  {" "}
                  {r.count}× {expanded ? "▾" : "▸"}
                </span>
              )}
            </span>
            {r.subject !== undefined && <span className="ev-d">{r.subject}</span>}
            {!grouped && r.args !== undefined && <Arguments args={r.args} />}
            {r.detail && (
              <span
                className="ev-d"
                // What a node said carries its hue, so a paragraph is attributable without
                // reading back to the name column. Only `model_text`: an event is this
                // pipeline speaking, not the node, and colouring it would say otherwise.
                style={
                  r.kind === "model_text" && r.node ? { color: prose(r.node) } : undefined
                }
              >
                {PROSE.has(r.kind) ? <Prose text={r.detail} /> : r.detail}
              </span>
            )}
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
              // A checkpoint is where a node stopped and produced something — the one row in
              // hundreds that marks a boundary — so it carries that node's hue as a wash.
              <div
                className={r.kind === "checkpoint" ? "ev ev--checkpoint" : "ev"}
                style={
                  r.kind === "checkpoint" && r.node
                    ? ({ "--row-tint": rowTint(r.node), "--tint": accent(r.node) } as React.CSSProperties)
                    : undefined
                }
              >
                {body}
              </div>
            )}
            {grouped &&
              expanded &&
              r.parts.map((p, n) => (
                <div className="ev ev--sub" key={`${key}-${n}`}>
                  <span className="ev-t">{clock(p.at)}</span>
                  {!node && <span className="ev-n" />}
                  <span className="ev-k ev-k--sub">{r.action}</span>
                  {p.subject !== undefined && <span className="ev-d">{p.subject}</span>}
                  {p.args !== undefined && <Arguments args={p.args} />}
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
