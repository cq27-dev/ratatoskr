import { useMemo } from "react";
import {
  BaseEdge,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import {
  Brain,
  FileText,
  Infinity as InfinityIcon,
  Search,
  Terminal,
  Pencil,
  Wrench,
} from "lucide-react";
import type { NodeFacts, NodeTelemetry, NodeView, PlannedNode } from "./api";

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
const NODE_SIZE = { width: 190, height: 96 };
const COLUMN_GAP = 96;
const LANE_GAP = 62;
const COLUMN_PITCH = NODE_SIZE.width + COLUMN_GAP;
const LANE_PITCH = NODE_SIZE.height + LANE_GAP;

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

/*
 * Tools are grouped rather than listed: a node carries up to a dozen, and a dozen glyphs is a
 * texture, not information. The grouping answers what a reader actually asks — can this node read,
 * can it write, can it run things — and the title carries the exact names for when that is not
 * enough.
 */
const TOOL_GROUPS: ReadonlyArray<{
  icon: typeof FileText;
  label: string;
  match: (tool: string) => boolean;
}> = [
  { icon: Pencil, label: "edits files", match: (t) => t === "Write" || t === "Edit" },
  { icon: Terminal, label: "runs commands", match: (t) => t === "Bash" },
  { icon: FileText, label: "reads files", match: (t) => ["Read", "Grep", "Glob"].includes(t) },
  { icon: Search, label: "searches the index", match: (t) => t.includes("search") || t.includes("symbol") || t.includes("impact") },
];

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
  const reuses =
    telemetry?.reuses_session ?? live?.facts?.reuses_session ?? planned?.reuses_session ?? false;
  const cycles = telemetry?.turns ?? live?.cycles ?? null;
  const groups = TOOL_GROUPS.filter((g) => tools.some(g.match));
  const ungrouped = tools.filter((t) => !TOOL_GROUPS.some((g) => g.match(t)));
  const model = modelFull?.split("/").pop() ?? "—";
  const tokens = telemetry
    ? `${short(telemetry.input_tokens + telemetry.cached_input_tokens)} in / ${short(telemetry.output_tokens)} out`
    : "—";

  return (
    <>
      <div className="node-model" title={modelFull ?? undefined}>
        {model}
      </div>
      <div className="node-meta">
        <span title="model calls in this node's latest attempt">
          {cycles ?? "—"} {cycles === 1 ? "cycle" : "cycles"}
        </span>
        <span
          title={
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
        {reuses && (
          <span
            className="node-icon"
            /* Not a setting — this says the session was actually carried over. */
            title="Compounding: this node keeps its memory when it is re-entered, so a later attempt continues the earlier one"
          >
            <InfinityIcon size={13} aria-label="compounding" />
          </span>
        )}
        {thinking && (
          <span
            className="node-icon"
            title={
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
              title={
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
            title={ungrouped
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
    <div className={cls}>
      <Handle type="target" position={Position.Left} />
      <div className="node-name">
        <span>{node.name.replace("_", " ")}</span>
        <span className="dot" aria-hidden="true" />
      </div>
      <div className="node-meta">
        <span className={`st st--${node.state}`}>{node.state}</span>
        <span title="checkpoints written">
          {node.checkpoints > 0 ? `${node.checkpoints} CP` : "—"}
        </span>
      </div>
      {(node.telemetry || data.live?.facts || node.planned) && (
        <NodeFacts telemetry={node.telemetry} live={data.live} planned={node.planned} />
      )}
      <Handle type="source" position={Position.Right} />
      <Handle type="target" id="loop-in" position={Position.Bottom} />
      <Handle type="source" id="loop-out" position={Position.Bottom} />
    </div>
  );
}

/**
 * The converge loop, drawn below the implementer. React Flow imposes no acyclicity, so the loop is
 * a real edge rather than an annotation.
 */
function ConvergeEdge({ id, sourceX, sourceY, label, markerEnd }: EdgeProps) {
  // Sized off the node so the loop stays under its own box as the box changes.
  const drop = LANE_GAP / 2;
  const half = NODE_SIZE.width / 2 - 14;
  // Quadratic corners, to match the rounded stage edges rather than being the one square turn left.
  const r = 14;
  const right = sourceX + 14;
  const left = sourceX - half;
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
      <BaseEdge id={id} path={path} {...(markerEnd ? { markerEnd } : {})} />
      <text
        className="react-flow__edge-text"
        x={sourceX - 16}
        y={sourceY + drop + 12}
        textAnchor="middle"
      >
        {label}
      </text>
    </>
  );
}

const nodeTypes = { pipeline: PipelineNode };
const edgeTypes = { converge: ConvergeEdge };

/** What a node has said about itself so far, before it has checkpointed anything. */
export interface LiveNode {
  facts?: NodeFacts;
  cycles: number;
  /** Tools called so far in this attempt. */
  used: Set<string>;
}

interface Props {
  nodes: NodeView[];
  /** Keyed by node name. Fills the box while a node is still working. */
  live: Map<string, LiveNode>;
  /** The node that last did something, from the event stream. Overrides the derived state. */
  active: string | null;
  selected: string | null;
  onSelect: (name: string) => void;
}

export default function PipelineGraph({ nodes, live, active, selected, onSelect }: Props) {

  /*
   * The pipeline as it actually stands, from two sources each authoritative for a different half.
   *
   * The store proves what has *completed*: checkpoints are durable and survive a reload. The event
   * stream is the only thing that knows what is happening *now* — and the store's guess at that is
   * inverted precisely when it matters. Mid-converge the implementer holds a checkpoint and is
   * still being re-run, so it reads as working; the verifier is an optional stage that has not
   * checkpointed, so it reads as not started. Someone watching the verifier sees the implementer
   * lit up instead.
   *
   * Everything below reads from this one list — boxes, edges, and the converge loop. Deriving any
   * of them from the raw list is how the loop came to glow green while a different node worked.
   */
  const view = useMemo(() => {
    if (!active) return nodes;
    return nodes.map((n) => {
      if (n.name === active) return { ...n, state: "working" as const };
      // Something else is talking, so this is not working: fall back to what its checkpoints
      // support. A node with one is done; a node with none never started.
      if (n.state === "working") {
        return { ...n, state: (n.checkpoints > 0 ? "done" : "idle") as NodeView["state"] };
      }
      return n;
    });
  }, [nodes, active]);

  const columns = useMemo(() => stages(view), [view]);
  const byName = useMemo(() => new Map(view.map((n) => [n.name, n])), [view]);

  const rfNodes = useMemo<PipelineNodeType[]>(() => {
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

  const rfEdges = useMemo<Edge[]>(() => {
    // Every node in a stage feeds every node in the next one — which is what a fork joining back
    // together looks like, and the only edge relation the pipeline has.
    // Rounded rather than square: the boxes carry the substrate's right angles, and the wiring
    // reads better when it does not compete with them.
    const edges: Edge[] = columns.flatMap((lanes, i) =>
      (columns[i + 1] ?? []).flatMap((target) =>
        lanes.map((source) => ({
          id: `${source.name}-${target.name}`,
          source: source.name,
          target: target.name,
          type: "smoothstep",
          pathOptions: { borderRadius: 24 },
        })),
      ),
    );

    // The converge loop only exists once the implementer has actually run.
    const impl = byName.get("implementer");
    if (impl && impl.checkpoints > 0) {
      edges.push({
        id: "converge",
        source: "implementer",
        target: "implementer",
        type: "converge",
        sourceHandle: "loop-out",
        targetHandle: "loop-in",
        label: `CONVERGE ×${impl.checkpoints}`,
        ...(impl.state === "working" ? { className: "is-live" } : {}),
      });
    }
    return edges;
  }, [byName]);

  return (
    <ReactFlow
      nodes={rfNodes}
      edges={rfEdges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodeClick={(_, node) => onSelect(node.id)}
      fitView
      fitViewOptions={{ padding: 0.3 }}
      proOptions={{ hideAttribution: true }}
      nodesConnectable={false}
      nodesDraggable={false}
      panOnScroll
      minZoom={0.4}
      maxZoom={1.6}
      colorMode="dark"
    />
  );
}
