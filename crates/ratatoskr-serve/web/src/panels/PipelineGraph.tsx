import { useEffect, useMemo, useRef } from "react";
import { accent, wash } from "../ui/tint";
import {
  BaseEdge,
  Handle,
  Position,
  ReactFlow,
  useEdgesState,
  getNodesBounds,
  useNodesInitialized,
  useNodesState,
  useReactFlow,
  useStore,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { Brain, Infinity as InfinityIcon, Repeat, Wrench } from "lucide-react";
import { TOOL_GROUPS } from "../ui/tools";
import {
  LANE_GAP,
  LOOP_SHELF_STEP,
  NODE_SIZE,
  BRANCH_TAP,
  anchoredBranches,
  tapSide,
  wiredToSpine,
  branchPlace,
  crowdLimit,
  place,
  rowExtent,
  spineNodes,
  LOOP_BAND,
  SPAN_BAND,
  SPAN_RADIUS,
  fittedBounds,
  tallestNeighbours,
  spanRiser,
  spanShelf,
} from "./layout";
import type {
  NodeFacts,
  NodeTelemetry,
  NodeView,
  PlannedNode,
  RunStage,
  SessionScope,
} from "../api";
import {
  type Transition,
  forkHandoff,
  handoffDrawn,
  skippedSpans,
  type ConvergeLoops,
  type DerivedNode,
} from "../derive";

/*
 * Positions are computed from the `stage` and `lane` the server sends with each node, not from a
 * table here: the pipeline's shape is the server's to know, and a workflow that declares its own
 * nodes changes it. A copy of the shape on this side would be a copy that goes stale silently,
 * with the missing node logged to a console nobody has open.
 *
 * A stage is a column; its nodes are the lanes within it, centred against the tallest stage. This
 * is not a general graph layout — elkjs/dagre exist for graphs whose edges aren't known until
 * runtime, and here every edge is "the stage before it".
 */

function stageLabel(id: string): string {
  return id
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}


/**
 * Declared up front rather than measured. React Flow keeps a node `visibility: hidden` until a
 * ResizeObserver reports its size, and these are a fixed-size box anyway — giving the dimensions
 * removes the dependency on that callback ever firing, which is what decides whether the graph
 * appears at all. Keep in step with `.node` in style.css.
 */
/** Thousands separators are noise at this size; magnitude is the whole message. */
function short(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return `${n}`;
}

/**
 * One node's capabilities and cost, as icons with the detail on hover.
 *
 * Takes whichever source has spoken. A finished node has a checkpoint carrying both what it ran on
 * and what it cost; a working one has only what it announced at the start, so the cost line reads
 * as pending rather than as zero — a node mid-flight has not spent nothing.
 */
function NodeFacts({
  telemetry,
  live,
  planned,
}: {
  telemetry: NodeTelemetry | undefined;
  live: DerivedNode | undefined;
  planned: PlannedNode | undefined;
}) {
  // Three sources, most-actual first: what the node recorded, what it announced when it started,
  // and what config says it would use. The last is what fills a node that has not run yet.
  const tools = telemetry?.tools?.length ? telemetry.tools : (live?.telemetry?.tools ?? []);
  // What it reached for, as against what it was handed. A node given a shell it never used is
  // worth seeing, so the two are drawn differently rather than the unused ones being dropped.
  const used = telemetry?.tools_used?.length
    ? new Set(telemetry.tools_used)
    : (live?.used ?? new Set<string>());
  const modelFull = telemetry?.model ?? live?.telemetry?.model ?? planned?.model ?? null;
  const thinkingRequested =
    telemetry?.thinking_requested ??
    live?.telemetry?.thinking_requested ??
    planned?.thinking_requested ??
    false;
  // Every scope this box's stages run under, so a box whose halves continue differently shows each
  // of their marks rather than one of them winning. Config is the only source that can say WHICH: a
  // recorded `reuses_session` is set by a compacted re-entry too, so on its own it can say no more
  // than "something continued", and it is the last resort for a box config has no route for.
  const scopes = new Set<SessionScope>(planned?.sessions ?? []);
  if (!scopes.size && (telemetry?.reuses_session || live?.telemetry?.reuses_session)) {
    scopes.add("reuse");
  }
  const cycles = telemetry?.turns ?? live?.cycles ?? null;
  const groups = TOOL_GROUPS.filter((g) => tools.some(g.match));
  const ungrouped = tools.filter((t) => !TOOL_GROUPS.some((g) => g.match(t)));
  // A node whose record covers more than one turn names every route it ran on, comma-separated —
  // the red team's two halves resolve theirs separately. Shorten each, or the split swallows all
  // but the last and the box asserts one of them.
  const model = modelFull?.split(", ").map((m) => m.split("/").pop()).join(", ") ?? "—";
  const tokens = telemetry
    ? `${short(telemetry.input_tokens + telemetry.cached_input_tokens)} in / ${short(telemetry.output_tokens)} out`
    : "—";

  return (
    <>
      <div className="node-model" data-tip={modelFull ?? undefined}>
        {model}
      </div>
      <div className="node-meta">
        <span data-tip="model calls in this node's latest attempt">
          {cycles ?? "—"} {cycles === 1 ? "cycle" : "cycles"}
        </span>
        <span
          data-tip={
            telemetry
              ? `${telemetry.input_tokens} fresh + ${telemetry.cached_input_tokens} cached in, ${telemetry.output_tokens} out`
              : "counted when the node checkpoints"
          }
        >
          {tokens}
        </span>
      </div>
      <div className="node-icons">
        {/* Lucide takes no `title`, and a wrapper is the better hover target anyway. */}
        {scopes.has("reuse") && (
          <span
            className="node-icon"
            data-tip="Endpoint continuation: this node keeps its endpoint session when it is re-entered"
          >
            <InfinityIcon size={13} aria-label="compounding" />
          </span>
        )}
        {scopes.has("compacted") && (
          <span
            className="node-icon"
            data-tip="Compacted continuation: a re-entered node receives a local summary of its previous attempt"
          >
            <Repeat size={13} aria-label="compacted continuation" />
          </span>
        )}
        {thinkingRequested && (
          <span
            className="node-icon"
            data-tip={thinkingTip(telemetry, live)}
          >
            <Brain size={13} aria-label="thinking" />
          </span>
        )}
        {groups.map(({ icon: Icon, label, match }) => {
          const mine = tools.filter(match);
          const called = mine.filter((t) => used.has(t));
          return (
            <span
              key={label}
              className={`node-icon${called.length ? " node-icon--used" : ""}`}
              data-tip={
                called.length
                  ? `${label} — used: ${called.join(", ")}${
                      mine.length > called.length
                        ? ` (also available: ${mine.filter((t) => !used.has(t)).join(", ")})`
                        : ""
                    }`
                  : `${label} — available, not used: ${mine.join(", ")}`
              }
            >
              <Icon size={13} aria-label={label} />
            </span>
          );
        })}
        {ungrouped.length > 0 && (
          <span
            className={`node-icon${ungrouped.some((t) => used.has(t)) ? " node-icon--used" : ""}`}
            data-tip={ungrouped
              .map((t) => (used.has(t) ? `${t} (used)` : t))
              .join(", ")}
          >
            <Wrench size={13} aria-label="other tools" />
          </span>
        )}
      </div>
    </>
  );
}

/**
 * A node update, keeping what React Flow measured on the last one.
 *
 * The same box in the same place with new `data` keeps its measurement: React Flow owns measurement
 * and writes it back through `onNodesChange`, and replacing the object it is working on makes it
 * re-measure on every render and drop the edges it cannot route until both endpoints are measured.
 *
 * A box that changed SIZE is not the same box: its measurement describes something that no longer
 * exists, and React Flow would go on fitting, routing and hit-testing the size it used to be. Only
 * the measurement is replaced. Handing back the bare node instead takes its handle bounds away with
 * it, and a node without those reads as uninitialised — the condition the fit waits on, so the graph
 * would never refit around the box that grew.
 */
/**
 * What the thinking marker says, given what the turn reported about its reasoning.
 *
 * Four states, because absence means two different things and only one of them is about the
 * endpoint. A count of zero is a MEASUREMENT — the endpoint was asked and answered none — while an
 * absent count on a turn that HAS reported its cost means the endpoint reports no such figure at
 * all, which is Anthropic's answer for every turn because it bills thinking inside its output
 * count. A truthiness check collapses those two, and so does treating a node whose cost has not
 * arrived yet as either of them: a live node makes no claim about its endpoint until one of its
 * turns has answered.
 */
export function thinkingTip(
  telemetry: Pick<NodeTelemetry, "reasoning_tokens"> | undefined,
  live: Pick<DerivedNode, "telemetry" | "costed"> | undefined,
): string {
  // The same sources in the same order as the flag beside it, because a checkpoint is absent until
  // a node finishes: reading only that made a live turn's reported figure — zero or otherwise —
  // render as an endpoint that reports none.
  const reasoning = telemetry?.reasoning_tokens ?? live?.telemetry?.reasoning_tokens ?? null;
  // Whether any cost report has arrived at all, from either.
  const costReported = telemetry != null || live?.costed === true;
  if (reasoning == null && !costReported) {
    return "Thinking: this node is not stopped from reasoning before it answers, and has not reported what this turn spent yet";
  }
  if (reasoning == null) {
    return "Thinking: this node is not stopped from reasoning before it answers (whether it does is the endpoint's call, and this endpoint reports no reasoning figure at all)";
  }
  if (reasoning === 0) {
    return "Thinking: this node was free to reason, and the endpoint reported 0 reasoning tokens for this turn";
  }
  return `Thinking: ${short(reasoning)} reasoning tokens before answering`;
}

export function carryMeasurement(
  previous: PipelineNodeType | undefined,
  next: PipelineNodeType,
): PipelineNodeType {
  if (!previous) return next;
  const size = { width: next.width, height: next.height };
  if (previous.width === size.width && previous.height === size.height) {
    return { ...previous, ...next };
  }
  if (size.width === undefined || size.height === undefined) return next;
  return { ...previous, ...next, measured: { width: size.width, height: size.height } };
}

/** Nodes grouped into their stages, in pipeline order, from what the server sent. */
function stages(nodes: NodeView[]): NodeView[][] {
  const byStage = new Map<number, NodeView[]>();
  for (const n of nodes) {
    const lanes = byStage.get(n.stage) ?? [];
    lanes.push(n);
    byStage.set(n.stage, lanes);
  }
  return [...byStage.entries()]
    .sort(([a], [b]) => a - b)
    .map(([, lanes]) => lanes.sort((a, b) => a.lane - b.lane));
}

/**
 * The box to pulse when the latest transition has no drawn edge, or `null` when an edge carries
 * it.
 *
 * A pair can be undrawable on purpose: a node invoked by two different boxes anchors nowhere and
 * chains from nothing — the graph deliberately refuses every in-edge for it — yet the hand-off
 * still happened. Pulsing a nonexistent edge would assert a relation the graph refuses; pulsing
 * the box asserts only what the record proved, that it just became active.
 */
export function pulsedBox(
  transition: Transition | null,
  edges: readonly { source: string; target: string }[],
): string | null {
  if (!transition) return null;
  const drawn = edges.some((e) => e.source === transition.from && e.target === transition.to);
  return drawn ? null : transition.to;
}

/**
 * How many pips a composed box shows before the rest fold into a `+N` tile.
 *
 * The box must NOT grow with the stage count: `NODE_SIZE` is fixed and unmeasured, and the scrub
 * magnification budget assumes neighbouring boxes can grow without covering each other — a strip
 * that widened the box would spend room the layout never reserved.
 */
export const PIP_CAP = 6;

/**
 * A box's pips: its declared member stages with what each is doing.
 *
 * `declared` is ALL of the box's stage ids, the self-named row included, and the threshold counts
 * them all: a box of two stages shows its strip whether the second is a peer beside a self-named
 * stage or two composed members — a self-plus-one-peer box that lost its strip hid exactly the
 * peer state this exists to show. The SELF stage still draws no pip, because the box's own
 * chrome — its border, its dot, its state line — already says what the box itself is doing; only
 * a truly single-stage box is pip-free. None where the stream has not spoken for the box at all.
 */
export function pipsOf(
  declared: readonly string[] | undefined,
  box: string,
  states: ReadonlyMap<string, string> | undefined,
): { id: string; state: string }[] {
  if (!declared || declared.length < 2 || !states) return [];
  return declared
    .filter((id) => id !== box)
    .map((id) => ({ id, state: states.get(id) ?? "idle" }));
}

/** The pips a strip actually draws: everything, or `cap - 1` of them plus how many folded — the
 *  `+N` tile takes the slot the overflow starts at, so the strip never exceeds `cap` tiles. */
export function pipStrip<T>(
  pips: readonly T[],
  cap: number = PIP_CAP,
): { shown: readonly T[]; more: number } {
  if (pips.length <= cap) return { shown: pips, more: 0 };
  const shown = pips.slice(0, cap - 1);
  return { shown, more: pips.length - shown.length };
}

type PipelineNodeData = {
  node: NodeView;
  live: DerivedNode | undefined;
  isSelected: boolean;
  /** Whether this box just became active with no drawn edge to carry the pulse; see [`pulsedBox`]. */
  entered: boolean;
  /**
   * The box's declared member stages with what each is doing, in declaration order — empty for a
   * single-stage box, and for one the stream has not spoken for. Which stages belong here is the
   * run's REGISTRY's answer; a stage the shape never assigned to this box must not appear in it.
   */
  pips: readonly { id: string; state: string }[];
};
type PipelineNodeType = Node<PipelineNodeData, "pipeline">;

/** One pipeline node: name, live dot, state, and its checkpoint count. */
function PipelineNode({ data }: NodeProps<PipelineNodeType>) {
  const { node, isSelected } = data;
  const cls = [
    "node",
    `node--${node.state}`,
    isSelected ? "node--selected" : "",
    data.entered ? "node--entered" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    // `--tint` rather than a colour per property: the hue comes from the node's name, which CSS
    // cannot derive, but everything downstream of it is ordinary styling and belongs in the
    // stylesheet. The wash fades out before the top, so the state backgrounds below it still read.
    <div
      className={cls}
      style={
        {
          "--tint": accent(node.name),
          backgroundImage: wash(node.name),
        } as React.CSSProperties
      }
    >
      {/* Named, all five of them: with more than one handle per type an edge that names none is
          resolved by whichever registered without an id, and that ordering is not ours to rely on.
          Unnamed, the stage edges vanished intermittently on re-render. */}
      <Handle type="target" id="in" position={Position.Left} isConnectable={false} />
      {/* Where a hand-off from the lane above lands. Its own handle because `in` is on the left,
          where the stage edges arrive, and a step down a lane comes in from the top. */}
      <Handle type="target" id="lane-in" position={Position.Top} isConnectable={false} />
      <div className="node-name">
        <span>{stageLabel(node.name)}</span>
        <span className="dot" aria-hidden="true" />
      </div>
      <div className="node-meta">
        <span className={`st st--${node.state}`}>{node.state}</span>
        <span data-tip="checkpoints written">
          {node.checkpoints > 0 ? `${node.checkpoints} CP` : "—"}
        </span>
      </div>
      {data.pips.length > 0 &&
        (() => {
          const { shown, more } = pipStrip(data.pips);
          return (
            <div className="node-pips">
              {shown.map((p) => (
                <span key={p.id} className={`pip pip--${p.state}`} data-tip={`${p.id} — ${p.state}`} />
              ))}
              {more > 0 && (
                <span className="pip pip--more" data-tip={`${more} more stages — open the box for all of them`}>
                  +{more}
                </span>
              )}
            </div>
          );
        })()}
      {(node.telemetry || data.live?.telemetry || node.planned) && (
        <NodeFacts telemetry={node.telemetry} live={data.live} planned={node.planned} />
      )}
      <Handle type="source" id="out" position={Position.Right} isConnectable={false} />
      {/* The branch taps: where a caller edge drops out of a parent and into the child hung below
          it — a straight vertical, so the offsets on the two boxes must match. Off-centre and
          inside the column, because the gaps carry the loop shelves and the bottom centre carries
          the loop handles; outside the converge self-loop's reach on either side. Both sides
          exist statically — `tapSide` picks per parent, away from any loop shelf ending at its
          column, since a declared layout may order its stages either way. */}
      <Handle
        type="source"
        id="branch-out-left"
        position={Position.Bottom}
        style={{ left: `${BRANCH_TAP * 100}%` }}
        isConnectable={false}
      />
      <Handle
        type="target"
        id="branch-in-left"
        position={Position.Top}
        style={{ left: `${BRANCH_TAP * 100}%` }}
        isConnectable={false}
      />
      <Handle
        type="source"
        id="branch-out-right"
        position={Position.Bottom}
        style={{ left: `${(1 - BRANCH_TAP) * 100}%` }}
        isConnectable={false}
      />
      <Handle
        type="target"
        id="branch-in-right"
        position={Position.Top}
        style={{ left: `${(1 - BRANCH_TAP) * 100}%` }}
        isConnectable={false}
      />
      <Handle type="target" id="loop-in" position={Position.Bottom} isConnectable={false} />
      <Handle type="source" id="loop-out" position={Position.Bottom} isConnectable={false} />
    </div>
  );
}

/**
 * The converge loop, drawn below the implementer. React Flow imposes no acyclicity, so the loop is
 * a real edge rather than an annotation.
 */
function ConvergeEdge({ id, sourceX, sourceY, label, markerEnd, style }: EdgeProps) {
  // Sized off the node so the loop stays under its own box as the box changes, and symmetric about
  // it: both handles sit at the bottom centre, so `sourceX` IS the centre and the two sides have to
  // reach equally far or the loop hangs visibly off to one side of the node it belongs to.
  const drop = LANE_GAP / 2;
  const reach = NODE_SIZE.width / 4;
  // Quadratic corners, to match the rounded stage edges rather than being the one square turn left.
  const r = 14;
  const right = sourceX + reach;
  const left = sourceX - reach;
  const bottom = sourceY + drop;
  const path = [
    `M ${right},${sourceY}`,
    `L ${right},${bottom - r}`,
    `Q ${right},${bottom} ${right - r},${bottom}`,
    `L ${left + r},${bottom}`,
    `Q ${left},${bottom} ${left},${bottom - r}`,
    `L ${left},${sourceY}`,
  ].join(" ");

  return (
    <>
      {/* `interactionWidth` 0 removes the invisible 20px-wide hit path React Flow lays over every
          edge. The wiring is a diagram, not a control — it states a relation between two boxes and
          has nothing to show when picked — and with the hit path there a click stuck the edge in
          its selected stroke. Now the click is inert: the edge layer is not the pane, so this does
          not clear an open node either, which is right for a miss between two boxes. */}
      <BaseEdge
        id={id}
        path={path}
        interactionWidth={0}
        {...(style ? { style } : {})}
        {...(markerEnd ? { markerEnd } : {})}
      />
      <text
        className="react-flow__edge-text"
        x={sourceX}
        y={sourceY + drop + 12}
        textAnchor="middle"
      >
        {label}
      </text>
    </>
  );
}

/**
 * How far apart the loop shelves sit under the deepest row of boxes.
 *
 * Half a lane gap, which is what `ConvergeEdge` already drops its self-loop by — so the three
 * loops are evenly spaced whether or not the self-loop is drawn, and the spacing follows the
 * layout constants rather than being a number that has to be re-measured when they change.
 */

/**
 * How deep the band of span shelves reaches above the row.
 *
 * The depth the loop shelves occupy below it, so the graph is no taller above than below and
 * `fitView`'s padding covers both. Fixed: the spans divide this band between them however many
 * there are, rather than each taking a step and pushing the outermost out of the fitted view.
 */

/** What a back-edge needs beyond its endpoints: its caption, its shelf, and its riser's offset. */
type BackLoopData = { label: string; shelfY: number; takeoff: number };
type BackLoopEdgeType = Edge<BackLoopData, "backloop">;

/**
 * A loop back to an earlier stage: down out of the source, along a shelf, and up into the target.
 *
 * Its own component rather than a stock edge type because the shelf is the whole point — two of
 * these leave the same handle, and left to route themselves they would trace the same line. The
 * caller assigns each a fixed shelf and a riser offset, so the geometry is deterministic and the
 * two never double-stroke.
 */
function BackLoopEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  data,
  markerEnd,
  style,
}: EdgeProps<BackLoopEdgeType>) {
  const { label, shelfY, takeoff } = data ?? { label: "", shelfY: sourceY, takeoff: 0 };
  // Matches `ConvergeEdge`'s corners, which match the stage edges' rounding.
  const r = 14;
  const sx = sourceX + takeoff;
  // Which way the shelf runs. A loop goes backwards, so the target is normally to the left — but a
  // workflow is free to declare its stages in another order, and a hardcoded direction would turn
  // the corners inside out rather than simply looking odd.
  const dir = targetX < sx ? 1 : -1;
  const path = [
    `M ${sx},${sourceY}`,
    `L ${sx},${shelfY - r}`,
    `Q ${sx},${shelfY} ${sx - r * dir},${shelfY}`,
    `L ${targetX + r * dir},${shelfY}`,
    `Q ${targetX},${shelfY} ${targetX},${shelfY - r}`,
    `L ${targetX},${targetY}`,
  ].join(" ");

  return (
    <>
      {/* `style` carries `--tint` and nothing else does — dropping it here loses the colour with
          no type error to say so. See the note in `ConvergeEdge` on `interactionWidth`. */}
      <BaseEdge
        id={id}
        path={path}
        interactionWidth={0}
        {...(style ? { style } : {})}
        {...(markerEnd ? { markerEnd } : {})}
      />
      <text
        className="react-flow__edge-text"
        x={(sx + targetX) / 2}
        y={shelfY + 12}
        textAnchor="middle"
      >
        {label}
      </text>
    </>
  );
}

/**
 * Centre the map once its boxes are measured, and again if the pane itself changes size.
 *
 * Two things defeat the `fitView` prop here. It fits at init, and at init there is nothing to fit:
 * the nodes arrive from the server a moment later. And the pane is still settling — the scrubber
 * and the feed take their rows after the first paint — so a fit computed against the taller pane
 * leaves the map sitting low once the rows appear.
 *
 * So: fit when the boxes are measured, and refit while the pane is still resizing. `moved` stops
 * that the instant someone pans or zooms, because after that the view is theirs and a refit would
 * yank it back.
 *
 * Must be rendered INSIDE `<ReactFlow>`: that is what puts it in the provider's context.
 */
/**
 * Publishes the viewport's zoom as `--mag`: how much a node must grow to become readable.
 *
 * The graph is fitted to its pane, so in a two-thirds-width window every node draws at about 45%
 * and its 9px text lands near 4px. A fixed magnification cannot fix that — it scales by the same
 * factor whether the graph sits at 45% or 100%, so it is too little when zoomed out and absurd
 * when zoomed in. Dividing by the zoom targets a *size* rather than a factor: the node reaches
 * roughly what it was designed to be, whatever the graph is doing around it.
 *
 * Must be rendered INSIDE `<ReactFlow>` — that is what puts it in the provider's context.
 */
function Magnification({ crowd }: { crowd: number }) {
  const zoom = useStore((s) => s.transform[2]);
  useEffect(() => {
    // Clamped: under 1.2 it is not worth the movement, and past 3 the node covers its whole
    // neighbourhood and takes away the context that made it worth looking at.
    const mag = Math.min(3, Math.max(1.2, 1.05 / (zoom || 1)));
    const root = document.documentElement.style;
    root.setProperty("--mag", mag.toFixed(2));
    // Tracks `--mag` where there is room and stops where there is not, so a narrow pane — which is
    // exactly where `--mag` is largest — does not turn the working nodes into one merged block.
    root.setProperty("--mag-scrub", Math.min(1 + (mag - 1) * 0.7, crowd).toFixed(2));
  }, [zoom, crowd]);
  return null;
}

function FitToPane({
  count,
  reserveTop,
  reserveBottom,
  moved,
}: {
  count: number;
  reserveTop: number;
  reserveBottom: number;
  moved: boolean;
}) {
  const initialized = useNodesInitialized();
  const { fitView, fitBounds, getNodes } = useReactFlow();
  // React Flow measures its own pane; taking the size from its store rather than observing the
  // DOM means refitting on exactly the changes it has already noticed.
  const width = useStore((s) => s.width);
  const height = useStore((s) => s.height);
  /*
   * The rectangle the committed nodes occupy, read from what React Flow has actually laid out.
   *
   * A box growing or MOVING changes neither the node count nor the pane size, so without this
   * nothing refits and the graph stays fitted to the bounds it had before. Both axes: a node
   * leaving a trailing column for a caller branch — or switching branch columns as replay
   * completeness changes — moves horizontally with the count, the depth and the reserved bands
   * all unchanged, and tracking depth alone left the graph zoomed for a column that no longer
   * exists. Taken from the store rather than from the layout the component just computed, because
   * that changes a render EARLIER than the nodes do: fitting on it would fit the old boxes, and
   * by the time the new ones land nothing has changed to fit again. One string, so the selector
   * compares equal when nothing moved.
   */
  const bounds = useStore((s) => {
    let top = Infinity;
    let bottom = -Infinity;
    let left = Infinity;
    let right = -Infinity;
    for (const node of s.nodeLookup.values()) {
      const { x, y } = node.internals.positionAbsolute;
      top = Math.min(top, y);
      bottom = Math.max(bottom, y + (node.measured.height ?? 0));
      left = Math.min(left, x);
      right = Math.max(right, x + (node.measured.width ?? 0));
    }
    return bottom > top ? `${left}:${right}:${top}:${bottom}` : "";
  });
  useEffect(() => {
    if (moved || !initialized || count === 0 || width === 0 || height === 0) return;
    // `fitView` fits NODE bounds, and the span shelves hang above them. Padding is a fraction of
    // what is being fitted, so a short row leaves less room above it than a tall one — a
    // three-column graph fits with about 40px of headroom while the band wants 93, and the shelves
    // are clipped until someone pans. Worse, adding a span changes no node, so nothing refits.
    //
    // Fitting explicit bounds says what the graph actually occupies. Both ends of it: reserving
    // only the band above would take away the padding the loop shelves below had been riding on,
    // since fitting bounds fits them tighter than `fitView` does.
    if (reserveTop > 0 || reserveBottom > 0) {
      void fitBounds(fittedBounds(getNodesBounds(getNodes()), reserveTop, reserveBottom), {
        padding: 0.15,
      });
      return;
    }
    void fitView({ padding: 0.3 });
  }, [
    initialized,
    count,
    bounds,
    width,
    height,
    moved,
    reserveTop,
    reserveBottom,
    fitView,
    fitBounds,
    getNodes,
  ]);
  return null;
}

const nodeTypes = { pipeline: PipelineNode };
/** What a span needs beyond its endpoints: its shelf, and where each riser stands. */
type SpanData = { shelfY: number; takeoff: number; landing: number };
type SpanEdgeType = Edge<SpanData, "span">;

/**
 * A hand-off across stages the run never entered: out of the source, up a column gap, along a shelf
 * above the row, down the gap before the target, and in.
 *
 * Routed rather than left to `smoothstep`, for the reason `BackLoopEdge` is: the span crosses whole
 * columns of boxes, and a self-routing edge would trace through them.
 *
 * The risers stand in the COLUMN GAPS, not on the boxes' centre line. Every lane of a column shares
 * one centre x, so a riser leaving a box in a lower lane would pass behind every sibling above it —
 * the standard `implementer -> publisher` skip has exactly that shape, with `redteam` over one end
 * and `bookkeeper` over the other, and the edge would vanish and reappear through boxes it has
 * nothing to do with. A gap is empty by construction.
 *
 * Above the row rather than below because the band underneath is the loop shelves', and an edge
 * sharing their space reads as one of them.
 */
/**
 * The corner radius a span turns on, and the clearance its risers need at both ends of the gap.
 *
 * A riser closer to its box than this puts the corner's control point BEHIND the handle — the path
 * then enters the node it just left and doubles back out. Named here because the geometry and the
 * distribution have to agree about it: eight lanes across a 96px gap put risers 10.7px out, inside
 * a 14px corner.
 */

function SpanEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  data,
  markerEnd,
  style,
}: EdgeProps<SpanEdgeType>) {
  const { shelfY, takeoff, landing } = data ?? { shelfY: sourceY, takeoff: 0, landing: 0 };
  const r = SPAN_RADIUS;
  // A span runs forwards, so the target is normally to the right — but a workflow declares its own
  // column order, and a hardcoded direction would turn the corners inside out.
  const dir = targetX >= sourceX ? 1 : -1;
  const up = sourceX + takeoff * dir;
  const down = targetX - landing * dir;
  const path = [
    `M ${sourceX},${sourceY}`,
    `L ${up - r * dir},${sourceY}`,
    `Q ${up},${sourceY} ${up},${sourceY - r}`,
    `L ${up},${shelfY + r}`,
    `Q ${up},${shelfY} ${up + r * dir},${shelfY}`,
    `L ${down - r * dir},${shelfY}`,
    `Q ${down},${shelfY} ${down},${shelfY + r}`,
    `L ${down},${targetY - r}`,
    `Q ${down},${targetY} ${down + r * dir},${targetY}`,
    `L ${targetX},${targetY}`,
  ].join(" ");

  return (
    <BaseEdge
      id={id}
      path={path}
      interactionWidth={0}
      {...(style ? { style } : {})}
      {...(markerEnd ? { markerEnd } : {})}
    />
  );
}

const edgeTypes = { converge: ConvergeEdge, backloop: BackLoopEdge, span: SpanEdge };

interface Props {
  /**
   * The pipeline as it stands at the position being shown — the server's shape with every
   * per-moment fact already folded in from the event stream, and, at the end of a stopped run,
   * reconciled against the store. One list, computed once in `App`: a second correction applied
   * here is how the node a failed run died in went on reading "working" under a failed status,
   * with the server's answer computed, sent, and overwritten before it could be drawn.
   */
  nodes: NodeView[];
  /** Keyed by node name. Fills the box while a node is still working. */
  live: Map<string, DerivedNode>;
  /** Implementer re-entries so far, by route. Folded from the same event prefix as `nodes`. */
  loops: ConvergeLoops;
  /**
   * Whether the stream shows the red-team hand-off, or `null` where it cannot say. Evidence for
   * the edge that asserts it, instead of inference from box state that arbitrary composed stages
   * can also produce.
   */
  handoff: boolean | null;
  /**
   * The edge that was just traversed — the last hand-off at or before the shown position — or
   * `null` before anything has. Which edge lights is decided here by matching endpoints, so the
   * pulse lands on whatever the pair is drawn as: a stage edge, a caller drop, the converge
   * self-loop, or a back-loop.
   */
  transition: Transition | null;
  /** The run's recorded registry — which stages compose each box, in declaration order. */
  stages: readonly RunStage[];
  selected: string | null;
  /** `null` clears the selection, which returns the lower pane to the combined feed. */
  onSelect: (name: string | null) => void;
}

export default function PipelineGraph({
  nodes,
  live,
  loops,
  handoff,
  transition,
  stages: registry,
  selected,
  onSelect,
}: Props) {
  /*
   * Everything below — boxes, edges, and the converge loop — reads `nodes` and nothing else.
   * Deriving any of them from a second source is how the loop came to glow green while a different
   * node worked, and how a failed run's dead node went on reading "working".
   */
  const byName = useMemo(() => new Map(nodes.map((n) => [n.name, n])), [nodes]);
  /*
   * Nodes the stream proved were invoked from inside another hang off their caller's box instead
   * of holding trailing columns — the spine's stage/lane math never sees them, so a run with no
   * dynamic nodes lays out exactly as before. Which nodes qualify is `branchParent`'s caller rule
   * plus `anchoredBranches`' geometric gates; all that is decided here is geometry.
   */
  const anchoredSet = useMemo(
    () => anchoredBranches(nodes, wiredToSpine(loops, (name) => byName.has(name))),
    [nodes, loops, byName],
  );
  const branches = useMemo(
    () => nodes.filter((n) => anchoredSet.has(n.name)),
    [nodes, anchoredSet],
  );
  // The spine list itself, not only its columns: everything that reasons about drawn columns —
  // the stage fans, the skipped-span jumps — must read THIS list. Deriving a jump from the
  // original `nodes` sees an anchored child still holding the trailing column it vacated, and
  // draws a span from some earlier stage to a box that is no longer in any column.
  const spineList = useMemo(() => spineNodes(nodes, anchoredSet), [nodes, anchoredSet]);
  const columns = useMemo(() => stages(spineList), [spineList]);

  /*
   * Where the boxes go, computed once and read by everything that hangs off them — the boxes
   * themselves, the shelves above and below, and the rectangle the view fits. Two derivations of
   * one layout is how a shelf comes to hang off a row the boxes are no longer on.
   *
   * The extent is the SPINE's: the shelves hug the row, and branch boxes hang below them — an
   * extent that included the branches would push every shelf under the boxes they annotate.
   */
  const spine = useMemo(() => place(columns), [columns]);
  const extent = useMemo(() => rowExtent(spine.values()), [spine]);
  const placed = useMemo(() => {
    const all = new Map(spine);
    for (const [name, box] of branchPlace(branches, (name) => spine.get(name), extent.bottom)) {
      all.set(name, box);
    }
    return all;
  }, [spine, branches, extent]);
  // How far a scrubbed box may grow before it covers the one under it. Off the placement, since
  // that is what says which boxes are neighbours and how tall they are.
  const crowd = useMemo(() => crowdLimit(tallestNeighbours(placed.values())), [placed]);

  // Which stages compose each box, in declaration order, from the run's REGISTRY — never from the
  // records: which stages a box holds is a property of the graph, and a box mid-run whose second
  // member has not spoken yet still shows the pip waiting for it. ALL rows, the self-named one
  // included: whether a box is multi-stage is counted over everything it holds, and `pipsOf` is
  // what leaves the self stage pip-less.
  const membersOf = useMemo(() => {
    const out = new Map<string, string[]>();
    for (const s of registry) {
      const list = out.get(s.node);
      if (list) list.push(s.id);
      else out.set(s.node, [s.id]);
    }
    return out;
  }, [registry]);

  const desiredEdges = useMemo<Edge[]>(() => {
    // Every node in a stage feeds every node in the next one — which is what a fork joining back
    // together looks like, and the only edge relation the pipeline has. Except into a node the
    // shape does not place, whose column is the client's ordering rather than a declared hand-off:
    // see `handoffDrawn`.
    // Rounded rather than square: the boxes carry the substrate's right angles, and the wiring
    // reads better when it does not compete with them.
    const forward = (source: NodeView, target: NodeView) => ({
      id: `${source.name}-${target.name}`,
      source: source.name,
      target: target.name,
      sourceHandle: "out",
      targetHandle: "in",
      type: "smoothstep",
      pathOptions: { borderRadius: 24 },
    });
    const edges: Edge[] = columns.flatMap((lanes, i) =>
      (columns[i + 1] ?? []).flatMap((target) =>
        lanes.filter((source) => handoffDrawn(source, target)).map((source) => forward(source, target)),
      ),
    );

    /*
     * The one in-edge an appended node can prove: its resolved caller. An ANCHORED child hangs
     * directly below its parent, and its edge is a straight vertical between the two branch
     * taps — inside the column, left of the converge self-loop's reach, so it crosses none of
     * the column's own loop wiring and no stage edge, which all live in the gaps. A
     * trailing-column target with a caller keeps the ordinary forward edge: it sits in a column
     * of its own.
     */
    // Which side each parent's tap drops on: away from any loop shelf ending at its column. A
    // declared layout may order its stages either way, and a shelf arrives from its other
    // endpoint's side — a verifier LEFT of the implementer runs the fix shelf across the left
    // tap. The self-loop reaches both sides symmetrically and both taps clear it by construction.
    const approaches = new Map<string, number[]>();
    const approach = (name: string, other: string) => {
      const from = placed.get(other);
      if (!from) return;
      const xs = approaches.get(name);
      const x = from.x + from.width / 2;
      if (xs) xs.push(x);
      else approaches.set(name, [x]);
    };
    if (loops.fix > 0) {
      approach("implementer", "verifier");
      approach("verifier", "implementer");
    }
    if (loops.replan > 0) {
      approach("analyst", "verifier");
      approach("verifier", "analyst");
    }
    // Indexed once — scanning the whole list per caller edge is quadratic in a fan-out, and an
    // imported history does not bound how many dynamic nodes one run may call.
    const ids = new Set(edges.map((e) => e.id));
    for (const target of nodes) {
      const source = target.shaped === false && target.caller ? byName.get(target.caller) : undefined;
      if (!source) continue;
      let edge: Edge;
      if (anchoredSet.has(target.name)) {
        const box = placed.get(source.name);
        const side = tapSide(
          (box?.x ?? 0) + (box?.width ?? NODE_SIZE.width) / 2,
          approaches.get(source.name) ?? [],
        );
        edge = {
          id: `${source.name}-${target.name}`,
          source: source.name,
          target: target.name,
          sourceHandle: `branch-out-${side}`,
          targetHandle: `branch-in-${side}`,
          type: "straight",
        };
      } else {
        edge = forward(source, target);
      }
      if (!ids.has(edge.id)) {
        ids.add(edge.id);
        edges.push(edge);
      }
    }

    /*
     * The one sequenced pair inside a stage: the implementer receives a tree whose tests the red
     * team has already written, and cannot start before it has. Stage edges only ever join one
     * stage to the next, so without this the two read as a fork.
     *
     * Both names are hardcoded, and that is not an oversight to generalise from `lane`: only the
     * orchestrator knows which pair within a stage is sequenced, and the shape does not express it —
     * lane order stacks the boxes and proves nothing about ordering or concurrency. (A node's
     * *stage* does carry a claim, which is what the edges above are; a lane carries none.)
     * `forkHandoff` needs both boxes present in this list, so a workflow
     * without a red team draws nothing. A short vertical line down the lane gap, unlabelled and
     * untinted: it is a forward hand-off and should look like the other forward edges.
     */
    if (forkHandoff(nodes, handoff)) {
      edges.push({
        id: "redteam-implementer",
        source: "redteam",
        target: "implementer",
        sourceHandle: "loop-out",
        targetHandle: "lane-in",
        type: "straight",
      });
    }

    /*
     * The three ways the implementer is re-entered, each drawn only if it actually happened.
     *
     * Gated on the derived count, never on a checkpoint total: a traversal is a re-entry, so the
     * implementer's rows are one more than its loops and every straight-through run used to be
     * captioned `×1` for a loop it never made. And gated on the participating boxes existing, so a
     * workflow that declares no verifier draws nothing rather than an edge to nowhere.
     *
     * Tinted by the node each returns TO — the hue says *who*, matching the box the loop lands on,
     * its name in the feed, and its stretch of the scrubber. Left as a variable so the idle stroke
     * stays with the rest of the wiring; `.is-loop` is what makes it show at rest as well as live.
     */
    const impl = byName.get("implementer");
    const verifier = byName.get("verifier");
    const analyst = byName.get("analyst");
    // The shelves hang under the whole layout, not under one box, so two of them cannot cross a
    // node in a deeper lane — and over it for the same reason, since one clearing only its own box
    // would cross a box in a shallower lane. Read off the boxes actually being drawn, so a taller
    // one takes its shelves with it.
    const { top: rowTop, bottom: rowBottom } = extent;

    /*
     * Hand-offs across stages the run never entered. An ordinary edge joins adjacent columns, so
     * without these a run that skipped a stage draws a break exactly where it made its hand-off.
     * Which jumps happened is `skippedSpans`; all that is decided here is where the line goes.
     *
     * Offset per pair, so two spans leaving one box do not trace the same line — the same reason
     * the loop shelves are offset from each other.
     */
    /*
     * Every hand-off gets its own shelf, and they are DISTRIBUTED across a fixed band above the row
     * rather than stepped upward one by one. Both halves matter. Stepping had no ceiling — enough
     * spans and the outer shelves sat outside what `fitView` fits, which is node bounds plus a
     * fixed padding. Sharing one shelf per jump was bounded but illegible: the horizontal segments
     * coincide exactly, so four hand-offs drew as one line and sixty-four drew as one line.
     *
     * A band of the same depth the loop shelves occupy below the row, divided by the number of
     * spans, is bounded whatever a workflow declares and separates them whenever there is room.
     */
    const spans = skippedSpans(spineList);
    const band = new Map<number, NodeView[]>();
    for (const lanes of columns) if (lanes[0]) band.set(lanes[0].stage, lanes);
    /*
     * Where a box's riser stands in the gap beside its column: distributed across the gap by the
     * box's position among its lanes, never measured inward from one edge. Subtracting a fixed step
     * per lane runs out — a column of eight lanes put the eighth riser past the gap and inside the
     * box, crossing it and its siblings, and nothing caps how wide a column may be. A fraction of
     * the gap is strictly inside it for any number of lanes.
     */
    const riser = (node: NodeView) => {
      const lanes = band.get(node.stage) ?? [node];
      return spanRiser(
        lanes.findIndex((lane) => lane.name === node.name),
        lanes.length,
      );
    };
    spans.forEach(({ from, to }, i) => {
      const source = byName.get(from);
      const target = byName.get(to);
      if (!source || !target) return;
      edges.push({
        id: `span-${from}-${to}`,
        source: from,
        target: to,
        sourceHandle: "out",
        targetHandle: "in",
        type: "span",
        data: {
          shelfY: spanShelf(i, spans.length, rowTop),
          takeoff: riser(source),
          landing: riser(target),
        },
      });
    });

    const loop = (target: string): Partial<Edge> => ({
      sourceHandle: "loop-out",
      targetHandle: "loop-in",
      style: { "--tint": accent(target) } as React.CSSProperties,
    });

    // The suite never went clean, so the implementer ran again on its own. Its own box is both
    // ends of it, which is the self-loop the graph has always drawn.
    if (impl && loops.retry > 0) {
      edges.push({
        ...loop("implementer"),
        id: "loop-retry",
        source: "implementer",
        target: "implementer",
        type: "converge",
        label: `RETRY ×${loops.retry}`,
        // Composed, never assigned: `is-live` is what lights the loop while the implementer works,
        // and overwriting `className` with one of the two silently drops the other.
        className: ["is-loop", impl.state === "working" ? "is-live" : ""].filter(Boolean).join(" "),
      });
    }

    if (impl && verifier && loops.fix > 0) {
      edges.push({
        ...loop("implementer"),
        id: "loop-fix",
        source: "verifier",
        target: "implementer",
        type: "backloop",
        className: "is-loop",
        // Both back-edges leave the verifier's single bottom handle, so their risers are offset
        // apart; without that the two vertical segments sit on the same line and double-stroke.
        data: { label: `FIX ×${loops.fix}`, shelfY: rowBottom + 2 * LOOP_SHELF_STEP, takeoff: -10 },
      });
    }

    // The finding faulted the plan, so the analyst ran again first. The return leg
    // `analyst -> implementer` is already drawn as a forward stage edge.
    if (verifier && analyst && loops.replan > 0) {
      edges.push({
        ...loop("analyst"),
        id: "loop-replan",
        source: "verifier",
        target: "analyst",
        type: "backloop",
        className: "is-loop",
        data: {
          label: `REPLAN ×${loops.replan}`,
          shelfY: rowBottom + 3 * LOOP_SHELF_STEP,
          takeoff: 10,
        },
      });
    }
    // Not clickable, and not focusable by tab: an edge here states a relation between two nodes and
    // has nothing to show when you pick it. See the note in `ConvergeEdge` on the hit path.
    //
    // The traversed edge is tagged by its ENDPOINTS, not by kind: the last hand-off lights
    // whatever the pair is drawn as — a stage edge, a caller drop, the converge self-loop
    // (from === to), or a back-loop. Composed onto whatever class the edge already carries,
    // for the same reason the loops compose `is-live`.
    return edges.map((e) => {
      const traversed =
        transition !== null && e.source === transition.from && e.target === transition.to;
      const className = [e.className, traversed ? "is-traversed" : ""].filter(Boolean).join(" ");
      return {
        ...e,
        selectable: false,
        focusable: false,
        interactionWidth: 0,
        ...(className ? { className } : {}),
      };
    });
  }, [byName, columns, extent, loops, nodes, spineList, handoff, branches, placed, transition]);
  // Where the pulse lands when no edge can carry it. Decided from the edges actually being
  // drawn, so the fallback and the edge tag can never both fire — or neither.
  const pulsed = useMemo(() => pulsedBox(transition, desiredEdges), [transition, desiredEdges]);


  const desiredNodes = useMemo<PipelineNodeType[]>(() => {
    return [...columns.flatMap((lanes) => lanes), ...branches].map((n) => {
      const box = placed.get(n.name) ?? { x: 0, y: 0, ...NODE_SIZE };
      return {
        id: n.name,
        type: "pipeline" as const,
        position: { x: box.x, y: box.y },
        data: {
          node: n,
          live: live.get(n.name),
          isSelected: selected === n.name,
          entered: pulsed === n.name,
          // Only where the stream has spoken for the box: a run whose log rotated away has no
          // per-member evidence, and a strip of guessed pips would assert states nobody recorded.
          // A single declared member is the box's own work and shows nothing the box does not.
          pips: pipsOf(membersOf.get(n.name), n.name, live.get(n.name)?.memberStates),
        },
        draggable: false,
        width: box.width,
        height: box.height,
      };
    });
  }, [columns, branches, placed, live, selected, pulsed, membersOf]);


  /*
   * React Flow is a controlled component: it owns node measurement and writes the result back
   * through `onNodesChange`. Passing `nodes` with no change handler leaves it nowhere to put a
   * measurement, so it re-measures on every render and edges — which cannot be routed until both
   * endpoints are measured — were dropped whenever a re-read landed mid-measurement, permanently.
   *
   * These hooks are that channel. Server-derived data still drives the graph; it is applied to the
   * state React Flow owns rather than replacing the object it is working on.
   */
  // Once someone pans or zooms, the view belongs to them and nothing refits it.
  const moved = useRef(false);
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState<PipelineNodeType>([]);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    setRfNodes((prev) => {
      const previous = new Map(prev.map((n) => [n.id, n]));
      return desiredNodes.map((n) => carryMeasurement(previous.get(n.id), n));
    });
  }, [desiredNodes, setRfNodes]);

  useEffect(() => setRfEdges(desiredEdges), [desiredEdges, setRfEdges]);

  return (
    <ReactFlow
      nodes={rfNodes}
      edges={rfEdges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodeClick={(_, node) => onSelect(node.id)}
      /* Clicking the substrate clears the selection, which puts the lower pane back to the
         combined feed it starts on. Without it a node, once opened, could never be closed. */
      onPaneClick={() => onSelect(null)}
      /* Only a real gesture counts: `onMoveStart` fires with an event for a user pan or zoom and
         without one for a programmatic `fitView`, which must not count as taking the view over. */
      onMoveStart={(event) => {
        if (event) moved.current = true;
      }}
      fitView
      fitViewOptions={{ padding: 0.3 }}
      proOptions={{ hideAttribution: true }}
      nodesConnectable={false}
      nodesDraggable={false}
      /* Wheel and pinch zoom, drag to pan — what a map is expected to do. `panOnScroll` made the
         wheel pan and left zooming to a double-click, which is a gesture for opening things. */
      zoomOnScroll
      zoomOnPinch
      zoomOnDoubleClick={false}
      minZoom={0.4}
      maxZoom={1.6}
      colorMode="dark"
    >
      <FitToPane
        count={rfNodes.length}
        // What hangs off the boxes and has to be fitted with them. Both ends: the loop shelves
        // below were riding on `fitView`'s padding, and reserving only the span band above took
        // that padding away from them.
        reserveTop={rfEdges.some((e) => e.type === "span") ? SPAN_BAND : 0}
        // The union of the shelves and the boxes, not the band below the deepest node: the loop
        // shelves reach LOOP_BAND below the SPINE, and a branch box hangs deeper than that
        // already — reserving the whole band under it fitted a second, empty band beneath the
        // branch. Only the part of the band the boxes do not cover is reserved; with no branch
        // the two bottoms coincide and this is exactly LOOP_BAND.
        reserveBottom={
          rfEdges.some((e) => e.type === "backloop" || e.type === "converge")
            ? Math.max(0, extent.bottom + LOOP_BAND - rowExtent(placed.values()).bottom)
            : 0
        }
        moved={moved.current}
      />
      <Magnification crowd={crowd} />
    </ReactFlow>
  );
}
