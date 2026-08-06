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
import type { NodeView } from "./api";

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
const COLUMN_PITCH = 210;
const LANE_PITCH = 140;

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
const NODE_SIZE = { width: 150, height: 52 };

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

type PipelineNodeData = { node: NodeView; isSelected: boolean };
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
        <span>{node.checkpoints > 0 ? `${node.checkpoints} CP` : "—"}</span>
      </div>
      <Handle type="source" position={Position.Right} />
      <Handle type="target" id="loop-in" position={Position.Bottom} />
      <Handle type="source" id="loop-out" position={Position.Bottom} />
    </div>
  );
}

/**
 * The converge loop, drawn as a right-angled path below the implementer. React Flow imposes no
 * acyclicity, so the loop is a real edge rather than an annotation — and square corners keep it
 * consistent with the rest of the substrate.
 */
function ConvergeEdge({ id, sourceX, sourceY, label, markerEnd }: EdgeProps) {
  const drop = 30;
  const half = 46;
  const path = [
    `M ${sourceX + 14},${sourceY}`,
    `L ${sourceX + 14},${sourceY + drop}`,
    `L ${sourceX - half},${sourceY + drop}`,
    `L ${sourceX - half},${sourceY}`,
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

interface Props {
  nodes: NodeView[];
  selected: string | null;
  onSelect: (name: string) => void;
}

export default function PipelineGraph({ nodes, selected, onSelect }: Props) {
  const byName = useMemo(
    () => new Map(nodes.map((n) => [n.name, n])),
    [nodes],
  );

  const columns = useMemo(() => stages(nodes), [nodes]);

  const rfNodes = useMemo<PipelineNodeType[]>(() => {
    const maxLanes = Math.max(1, ...columns.map((c) => c.length));
    return columns.flatMap((lanes) =>
      lanes.map((n) => ({
        id: n.name,
        type: "pipeline" as const,
        position: position(n, lanes.length, maxLanes),
        data: { node: n, isSelected: selected === n.name },
        draggable: false,
        ...NODE_SIZE,
      })),
    );
  }, [columns, selected]);

  const rfEdges = useMemo<Edge[]>(() => {
    // Every node in a stage feeds every node in the next one — which is what a fork joining back
    // together looks like, and the only edge relation the pipeline has.
    // `step`, not the default bezier: every corner on this substrate is 90 degrees.
    const edges: Edge[] = columns.flatMap((lanes, i) =>
      (columns[i + 1] ?? []).flatMap((target) =>
        lanes.map((source) => ({
          id: `${source.name}-${target.name}`,
          source: source.name,
          target: target.name,
          type: "step",
          pathOptions: { borderRadius: 0 },
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
