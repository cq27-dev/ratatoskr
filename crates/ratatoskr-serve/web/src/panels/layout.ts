/**
 * The pipeline graph's fixed geometry, and the arithmetic that places what hangs off it.
 *
 * Apart from the component so it can be tested without a browser. Every span defect this file
 * exists because of was arithmetic — an offset that outgrew its gap, a shelf that outgrew the band,
 * a riser that landed inside the corner it turns on — and each was found by rendering one more
 * layout by hand. A number that has to hold for every lane count and every span count is checkable
 * for every lane count and every span count.
 */
import type { NodeView } from "../api";

/* Pitch is the box plus the room an edge needs to turn in. Derived from NODE_SIZE rather than
 * written as a literal: the two drifted apart once already, leaving 20px for a right-angled edge
 * to route through, and the edges rendered as smears. */
// Wide and tall enough that the meta line still fits on one row inside `.node`'s padding: the
// cycles and token counts wrapped when the padding grew, and a wrapped count reads as two facts.
export const NODE_SIZE = { width: 202, height: 104 };
export const COLUMN_GAP = 96;
export const LANE_GAP = 62;
export const COLUMN_PITCH = NODE_SIZE.width + COLUMN_GAP;
export const LANE_PITCH = NODE_SIZE.height + LANE_GAP;

/**
 * How far apart the loop shelves sit under the deepest row of boxes.
 *
 * Half a lane gap, which is what `ConvergeEdge` already drops its self-loop by — so the three
 * loops are evenly spaced whether or not the self-loop is drawn, and the spacing follows the
 * layout constants rather than being a number that has to be re-measured when they change.
 */
export const LOOP_SHELF_STEP = LANE_GAP / 2;

/**
 * How deep the loop shelves reach below the deepest row of boxes.
 *
 * Three steps: the two back-edges take the outer two, and `ConvergeEdge`'s self-loop drops within
 * the first. Their captions sit just under each shelf, so a little more is reserved for the text.
 */
export const LOOP_BAND = 3 * LOOP_SHELF_STEP + 16;

/**
 * How deep the band of span shelves reaches above the row.
 *
 * The depth the loop shelves occupy below it, so the graph is no taller above than below. Fixed:
 * the spans divide this band between them however many there are, rather than each taking a step
 * and pushing the outermost out of view.
 */
export const SPAN_BAND = 3 * LOOP_SHELF_STEP;

/**
 * The corner radius a span turns on, and the clearance its risers need at both ends of the gap.
 *
 * A riser closer to its box than this puts the corner's control point BEHIND the handle — the path
 * then enters the node it just left and doubles back out.
 */
export const SPAN_RADIUS = 14;

/**
 * How tall a box is drawn.
 *
 * The one place a height is decided, because every vertical number in the layout is stacked out of
 * it: where the lanes below it sit, how deep the row reaches, where the shelves hang, what the view
 * has to fit. A box that grows — a node showing the subagents it spawned — grows here, and the rest
 * follows without a second constant to keep in step.
 *
 * Declared rather than measured. React Flow keeps a node hidden until a ResizeObserver reports its
 * size, so measuring makes whether the graph appears at all depend on that callback firing; and a
 * box's contents are known before it is drawn, so there is nothing to learn by asking the DOM.
 */
export function nodeHeight(_node: NodeView): number {
  return NODE_SIZE.height;
}

/** A rectangle, in the flow's own coordinates. */
export type Bounds = { x: number; y: number; width: number; height: number };

/**
 * Where every box goes: columns left to right by stage, lanes stacked down by their own heights.
 *
 * Stacked rather than multiplied by a fixed pitch, so one box being taller than its siblings moves
 * the lanes under it instead of drawing over them. With every box the same height this is exactly a
 * constant pitch — which is what `layout.test.js` pins, since that is the layout in use today.
 *
 * A lane the column does not fill still takes a collapsed box's worth of room: `lane` is a declared
 * position, a workflow may leave a hole in one, and a hole that closed up would move every box under
 * it somewhere its layout did not ask for.
 *
 * Each column is centred against the tallest, so a fork sits either side of the row its neighbours
 * are on.
 */
export function place(
  columns: readonly (readonly NodeView[])[],
  heightOf: (node: NodeView) => number = nodeHeight,
): Map<string, Bounds> {
  /*
   * One column, top to bottom: where each of its boxes sits relative to the column's own top, and
   * how deep it reaches.
   *
   * The lanes it leaves empty are charged for arithmetically rather than materialised as slots.
   * `lane` is a declared position bounded only by the run's node count, so a recording may put one
   * box at lane N-1 in each of N columns, and a slot per lane would make drawing it quadratic in a
   * shape the read gate accepts. It also removes a disagreement that put a column off centre: a
   * sparse array's holes are skipped by `reduce` and visited by `for...of`, so the depth a column
   * was centred by and the depth it was drawn at were different numbers.
   */
  const stack = (lanes: readonly NodeView[]) => {
    const boxes: { node: NodeView; y: number; height: number }[] = [];
    let y = 0;
    let filled = 0; // the lowest lane not yet accounted for
    for (const node of [...lanes].sort((a, b) => a.lane - b.lane)) {
      const empty = Math.max(0, node.lane - filled);
      y += empty * (NODE_SIZE.height + LANE_GAP);
      const height = heightOf(node);
      boxes.push({ node, y, height });
      y += height + LANE_GAP;
      filled = node.lane + 1;
    }
    // `y` has a trailing gap on it — a column is as deep as its last box, not the room after it.
    return { boxes, depth: Math.max(0, y - LANE_GAP) };
  };

  const stacked = columns.map(stack);
  const tallest = Math.max(0, ...stacked.map((column) => column.depth));
  const placed = new Map<string, Bounds>();
  for (const column of stacked) {
    const top = (tallest - column.depth) / 2;
    for (const { node, y, height } of column.boxes) {
      placed.set(node.name, {
        x: node.stage * COLUMN_PITCH,
        y: top + y,
        width: NODE_SIZE.width,
        height,
      });
    }
  }
  return placed;
}

/**
 * How far the boxes reach up and down — what the shelves hang off and the view has to fit.
 *
 * Read off the placements rather than derived from the lane count and a fixed height: a row is only
 * as deep as the boxes actually in it, and one of them growing has to move what hangs under it.
 */
export function rowExtent(placed: Iterable<Bounds>): { top: number; bottom: number } {
  const boxes = [...placed];
  return {
    top: Math.min(0, ...boxes.map((b) => b.y)),
    bottom: Math.max(0, ...boxes.map((b) => b.y + b.height)),
  };
}

/**
 * How far out into the column gap the riser for lane `k` of `lanes` stands.
 *
 * Distributed across the gap MINUS a corner's clearance at each end, so it is inside the gap and
 * clear of its own turn for any number of lanes. Stepping a fixed amount inward from one edge ran
 * out instead: eight lanes put the last riser past the gap and inside the box, and nothing caps how
 * wide a column may be.
 */
export function spanRiser(k: number, lanes: number): number {
  const usable = Math.max(0, COLUMN_GAP - 2 * SPAN_RADIUS);
  return SPAN_RADIUS + (usable * (Math.max(0, k) + 1)) / (Math.max(1, lanes) + 1);
}

/**
 * Where the shelf for span `i` of `count` runs, above a row whose topmost box is at `rowTop`.
 *
 * Distributed across a fixed band rather than stepped upward one by one. Stepping had no ceiling,
 * and the fitted view reserves a fixed depth; sharing one shelf put every span on the same line,
 * so any number of hand-offs drew as one.
 */
export function spanShelf(i: number, count: number, rowTop: number): number {
  return rowTop - (SPAN_BAND * (i + 1)) / (Math.max(1, count) + 1);
}

/**
 * The rectangle a fit has to cover: the boxes, plus whatever hangs off them.
 *
 * `fitView` fits node bounds and leaves the rest to padding, which is a FRACTION of what it fits —
 * so a short row gets proportionally less room around it than a fixed band needs, and the outermost
 * shelf is clipped until someone pans. Adding a span or a loop changes no node, so nothing refits
 * to reveal it either.
 *
 * Both directions, because reserving only one is how the loops came to be clipped: the span band
 * went into the bounds and the padding that had been covering the loops was reduced to pay for it.
 */
export function fittedBounds(nodes: Bounds, above: number, below: number): Bounds {
  return {
    x: nodes.x,
    y: nodes.y - above,
    width: nodes.width,
    height: nodes.height + above + below,
  };
}

/**
 * The largest pair of boxes that sit next to each other in a column, added together.
 *
 * What the scrub magnification is bounded by: two neighbours both growing share the gap between
 * them, so what matters is the tallest adjacent pair rather than the tallest box. Boxes with an
 * empty lane between them are counted as neighbours too, which only over-reserves — they have that
 * lane's room as well.
 *
 * A box with no box under or over it is in no pair and contributes nothing, however tall it is:
 * there is no lane gap for it to grow into, and counting it against the lane bound holds a lone
 * grown node down to a magnification it never needed. Zero when no column holds two, which leaves
 * `crowdLimit` to the column gap.
 */
export function tallestNeighbours(placed: Iterable<Bounds>): number {
  const columns = new Map<number, Bounds[]>();
  for (const box of placed) {
    // Appended, not rebuilt: a spread here re-copies the column for every box in it, which is
    // quadratic in a column the read gate is happy to accept.
    const column = columns.get(box.x);
    if (column) column.push(box);
    else columns.set(box.x, [box]);
  }
  let tallest = 0;
  for (const column of columns.values()) {
    const heights = column.sort((a, b) => a.y - b.y).map((box) => box.height);
    for (let i = 1; i < heights.length; i += 1) {
      tallest = Math.max(tallest, (heights[i - 1] ?? 0) + (heights[i] ?? 0));
    }
  }
  return tallest;
}

/**
 * The most a node can grow while its neighbours grow too, before they touch.
 *
 * Hovering does not need this: it enlarges one box and lifts it above the others, so covering a
 * neighbour is the point. Scrubbing enlarges every node that was working at that moment, and two of
 * those can be adjacent — with no one on top, they have to fit. Growth is centred, so a pair reaches
 * half its own added height into the gap between them, and `0.7` of the gap leaves a visible sliver
 * rather than letting them meet edge to edge.
 *
 * Taken from the heights actually placed rather than from `NODE_SIZE`, because the pair is what the
 * bound is about: a 300px box beside a collapsed one eats 67px of a 62px lane gap at the magnifica-
 * tion two collapsed boxes are safe at. With every box the same height this is exactly the constant
 * it replaces.
 *
 * The horizontal half needs no such care — every box is `NODE_SIZE.width` wide, and columns are a
 * whole gap apart whatever is in them.
 */
export function crowdLimit(neighbours: number): number {
  return Math.min(
    1 + (COLUMN_GAP * 0.7) / NODE_SIZE.width,
    1 + (2 * LANE_GAP * 0.7) / Math.max(1, neighbours),
  );
}
