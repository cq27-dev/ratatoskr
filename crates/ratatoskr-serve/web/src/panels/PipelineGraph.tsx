import { useEffect, useMemo, useRef } from "react";
import { accent, wash } from "../ui/tint";
import {
  BaseEdge,
  Handle,
  Position,
  ReactFlow,
  useEdgesState,
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
import type { NodeFacts, NodeTelemetry, NodeView, PlannedNode, SessionScope } from "../api";
import {
  forkHandoff,
  handoffDrawn,
  skippedSpans,
  type ConvergeLoops,
  type LiveNode,
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
/* Pitch is the box plus the room an edge needs to turn in. Derived from NODE_SIZE rather than
 * written as a literal: the two drifted apart once already, leaving 20px for a right-angled edge
 * to route through, and the edges rendered as smears. */
// Wide and tall enough that the meta line still fits on one row inside `.node`'s padding: the
// cycles and token counts wrapped when the padding grew, and a wrapped count reads as two facts.
const NODE_SIZE = { width: 202, height: 104 };
const COLUMN_GAP = 96;
const LANE_GAP = 62;
const COLUMN_PITCH = NODE_SIZE.width + COLUMN_GAP;
const LANE_PITCH = NODE_SIZE.height + LANE_GAP;

function stageLabel(id: string): string {
  return id
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function position(node: NodeView, lanesInStage: number, maxLanes: number) {
  const offset = (maxLanes - lanesInStage) / 2;
  return {
    x: node.stage * COLUMN_PITCH,
    y: (node.lane + offset) * LANE_PITCH,
  };
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
  live: LiveNode | undefined;
  planned: PlannedNode | undefined;
}) {
  // Three sources, most-actual first: what the node recorded, what it announced when it started,
  // and what config says it would use. The last is what fills a node that has not run yet.
  const tools = telemetry?.tools?.length ? telemetry.tools : (live?.facts?.tools ?? []);
  // What it reached for, as against what it was handed. A node given a shell it never used is
  // worth seeing, so the two are drawn differently rather than the unused ones being dropped.
  const used = telemetry?.tools_used?.length
    ? new Set(telemetry.tools_used)
    : (live?.used ?? new Set<string>());
  const modelFull = telemetry?.model ?? live?.facts?.model ?? planned?.model ?? null;
  const thinking = telemetry?.thinking ?? live?.facts?.thinking ?? planned?.thinking ?? false;
  // Every scope this box's stages run under, so a box whose halves continue differently shows each
  // of their marks rather than one of them winning. Config is the only source that can say WHICH: a
  // recorded `reuses_session` is set by a compacted re-entry too, so on its own it can say no more
  // than "something continued", and it is the last resort for a box config has no route for.
  const scopes = new Set<SessionScope>(planned?.sessions ?? []);
  if (!scopes.size && (telemetry?.reuses_session || live?.facts?.reuses_session)) {
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
        {thinking && (
          <span
            className="node-icon"
            data-tip={
              telemetry && telemetry.reasoning_tokens > 0
                ? `Thinking: ${short(telemetry.reasoning_tokens)} reasoning tokens before answering`
                : "Thinking: this node is not stopped from reasoning before it answers (whether it does is the endpoint's call, and this one reports no reasoning tokens)"
            }
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

type PipelineNodeData = { node: NodeView; live: LiveNode | undefined; isSelected: boolean };
type PipelineNodeType = Node<PipelineNodeData, "pipeline">;

/** One pipeline node: name, live dot, state, and its checkpoint count. */
function PipelineNode({ data }: NodeProps<PipelineNodeType>) {
  const { node, isSelected } = data;
  const cls = ["node", `node--${node.state}`, isSelected ? "node--selected" : ""]
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
      {(node.telemetry || data.live?.facts || node.planned) && (
        <NodeFacts telemetry={node.telemetry} live={data.live} planned={node.planned} />
      )}
      <Handle type="source" id="out" position={Position.Right} isConnectable={false} />
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
const LOOP_SHELF_STEP = LANE_GAP / 2;

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
/**
 * The most a node can grow while its neighbours grow too, before they touch.
 *
 * Hovering does not need this: it enlarges one box and lifts it above the others, so covering a
 * neighbour is the point. Scrubbing enlarges every node that was working at that moment, and two of
 * those can be adjacent — with no one on top, they have to fit. Growth is centred, so each of two
 * neighbours reaches half the gap; the pair therefore has the whole gap to share, and `0.7` of it
 * leaves a visible sliver rather than letting them meet edge to edge.
 *
 * Derived from the layout constants rather than written as a number, because it is a fact *about*
 * them: change the pitch and the safe magnification changes with it.
 */
const CROWD_LIMIT = Math.min(
  1 + (COLUMN_GAP * 0.7) / NODE_SIZE.width,
  1 + (LANE_GAP * 0.7) / NODE_SIZE.height,
);

function Magnification() {
  const zoom = useStore((s) => s.transform[2]);
  useEffect(() => {
    // Clamped: under 1.2 it is not worth the movement, and past 3 the node covers its whole
    // neighbourhood and takes away the context that made it worth looking at.
    const mag = Math.min(3, Math.max(1.2, 1.05 / (zoom || 1)));
    const root = document.documentElement.style;
    root.setProperty("--mag", mag.toFixed(2));
    // Tracks `--mag` where there is room and stops where there is not, so a narrow pane — which is
    // exactly where `--mag` is largest — does not turn the working nodes into one merged block.
    root.setProperty("--mag-scrub", Math.min(1 + (mag - 1) * 0.7, CROWD_LIMIT).toFixed(2));
  }, [zoom]);
  return null;
}

function FitToPane({ count, moved }: { count: number; moved: boolean }) {
  const initialized = useNodesInitialized();
  const { fitView } = useReactFlow();
  // React Flow measures its own pane; taking the size from its store rather than observing the
  // DOM means refitting on exactly the changes it has already noticed.
  const width = useStore((s) => s.width);
  const height = useStore((s) => s.height);
  useEffect(() => {
    if (moved || !initialized || count === 0 || width === 0 || height === 0) return;
    void fitView({ padding: 0.3 });
  }, [initialized, count, width, height, moved, fitView]);
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
  // Matches the loop shelves', which match the stage edges' rounding.
  const r = 14;
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
  live: Map<string, LiveNode>;
  /** Implementer re-entries so far, by route. Folded from the same event prefix as `nodes`. */
  loops: ConvergeLoops;
  selected: string | null;
  /** `null` clears the selection, which returns the lower pane to the combined feed. */
  onSelect: (name: string | null) => void;
}

export default function PipelineGraph({ nodes, live, loops, selected, onSelect }: Props) {
  /*
   * Everything below — boxes, edges, and the converge loop — reads `nodes` and nothing else.
   * Deriving any of them from a second source is how the loop came to glow green while a different
   * node worked, and how a failed run's dead node went on reading "working".
   */
  const columns = useMemo(() => stages(nodes), [nodes]);
  const byName = useMemo(() => new Map(nodes.map((n) => [n.name, n])), [nodes]);

  const desiredNodes = useMemo<PipelineNodeType[]>(() => {
    const maxLanes = Math.max(1, ...columns.map((c) => c.length));
    return columns.flatMap((lanes) =>
      lanes.map((n) => ({
        id: n.name,
        type: "pipeline" as const,
        position: position(n, lanes.length, maxLanes),
        data: { node: n, live: live.get(n.name), isSelected: selected === n.name },
        draggable: false,
        ...NODE_SIZE,
      })),
    );
  }, [columns, live, selected]);

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
     * The one in-edge an appended node can prove. The server resolves `caller` only where the
     * record names the invocation rather than adjacency suggesting it — today the referee, which
     * judges the implementer checkpoint fetched immediately before it — so where it is present it
     * is the hand-off the columns above deliberately do not draw. Drawn like any other forward
     * edge, and skipped when the caller has no box or the pair is already joined.
     */
    for (const target of nodes) {
      const source = target.shaped === false && target.caller ? byName.get(target.caller) : undefined;
      if (!source) continue;
      const edge = forward(source, target);
      if (!edges.some((e) => e.id === edge.id)) edges.push(edge);
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
    if (forkHandoff(nodes)) {
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
    // node in a deeper lane. Measured off the positions actually being drawn.
    const maxLanes = Math.max(1, ...columns.map((c) => c.length));
    const laneTops = columns.flatMap((lanes) =>
      lanes.map((n) => position(n, lanes.length, maxLanes).y),
    );
    const rowBottom = Math.max(0, ...laneTops) + NODE_SIZE.height;
    // The spans hang over the whole layout for the same reason the loops hang under it: one that
    // cleared only its own box would cross a box in a shallower lane.
    const rowTop = Math.min(0, ...laneTops);

    /*
     * Hand-offs across stages the run never entered. An ordinary edge joins adjacent columns, so
     * without these a run that skipped a stage draws a break exactly where it made its hand-off.
     * Which jumps happened is `skippedSpans`; all that is decided here is where the line goes.
     *
     * Offset per pair, so two spans leaving one box do not trace the same line — the same reason
     * the loop shelves are offset from each other.
     */
    /*
     * One shelf per JUMP, not per pair. A jump between two columns of two boxes each is four
     * hand-offs and one claim — the run went from that column to this one — so they share a shelf
     * and fan out from it. Per-pair shelves stacked one above the other, and `fitView` fits node
     * bounds with a fixed padding: enough shelves and the outer ones sit outside the pane until
     * someone pans. The jumps a run can make are bounded by its columns; its box pairs are not.
     */
    const jumps = new Map<string, number>();
    const byNameSpan = (name: string) => nodes.find((n) => n.name === name);
    for (const { from, to } of skippedSpans(nodes)) {
      const source = byNameSpan(from);
      const target = byNameSpan(to);
      if (!source || !target) continue;
      const jump = `${source.stage}-${target.stage}`;
      if (!jumps.has(jump)) jumps.set(jump, jumps.size);
      const shelf = jumps.get(jump)!;
      // Risers stand in the column gaps, offset per lane so two spans of one jump do not trace the
      // same vertical line.
      const gap = COLUMN_GAP / 2;
      edges.push({
        id: `span-${from}-${to}`,
        source: from,
        target: to,
        sourceHandle: "out",
        targetHandle: "in",
        type: "span",
        data: {
          shelfY: rowTop - (shelf + 1) * LOOP_SHELF_STEP,
          takeoff: gap - source.lane * 8,
          landing: gap - target.lane * 8,
        },
      });
    }

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
    return edges.map((e) => ({ ...e, selectable: false, focusable: false, interactionWidth: 0 }));
  }, [byName, columns, loops, nodes]);

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
      const measured = new Map(prev.map((n) => [n.id, n]));
      // Carry each node's existing measurement across a data update: it is the same box in the
      // same place, and only `data` has changed.
      return desiredNodes.map((n) => ({ ...measured.get(n.id), ...n }));
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
      <FitToPane count={rfNodes.length} moved={moved.current} />
      <Magnification />
    </ReactFlow>
  );
}
