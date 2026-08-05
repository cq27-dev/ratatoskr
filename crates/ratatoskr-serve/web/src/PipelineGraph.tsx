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
 * The pipeline's shape is fixed and known ahead of time, so the layout is hand-authored rather
 * than run through a general graph-layout pass: elkjs/dagre exist for graphs whose shape isn't
 * known until runtime, which is not this one.
 *
 * Coordinates use a 190px column pitch so the fork's two lanes stay aligned.
 */
const LAYOUT: Record<string, { x: number; y: number }> = {
  scout: { x: 0, y: 70 },
  memory: { x: 190, y: 70 },
  analyst: { x: 380, y: 70 },
  red_team: { x: 590, y: 0 },
  implementer: { x: 590, y: 140 },
  bookkeeper: { x: 800, y: 70 },
};

const EDGES: ReadonlyArray<readonly [string, string]> = [
  ["scout", "memory"],
  ["memory", "analyst"],
  ["analyst", "red_team"],
  ["analyst", "implementer"],
  ["red_team", "bookkeeper"],
  ["implementer", "bookkeeper"],
];

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

  const rfNodes = useMemo<PipelineNodeType[]>(
    () =>
      nodes.flatMap((n) => {
        const position = LAYOUT[n.name];
        if (!position) {
          // Loud rather than invisible: a stage added server-side without a layout entry would
          // otherwise silently vanish from the graph.
          console.warn(`no layout for pipeline node "${n.name}" — omitted from the graph`);
          return [];
        }
        return [
          {
            id: n.name,
            type: "pipeline" as const,
            position,
            data: { node: n, isSelected: selected === n.name },
            draggable: false,
          },
        ];
      }),
    [nodes, selected],
  );

  const rfEdges = useMemo<Edge[]>(() => {
    // `step`, not the default bezier: every corner on this substrate is 90 degrees.
    const edges: Edge[] = EDGES.map(([source, target]) => ({
      id: `${source}-${target}`,
      source,
      target,
      type: "step",
      pathOptions: { borderRadius: 0 },
    }));

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
