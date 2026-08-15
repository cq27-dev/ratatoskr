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

/** Where a box sits, in the column its stage names and the lane within it. */
export function position(node: NodeView, lanesInStage: number, maxLanes: number) {
  const offset = (maxLanes - lanesInStage) / 2;
  return {
    x: node.stage * COLUMN_PITCH,
    y: (node.lane + offset) * LANE_PITCH,
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

/** A rectangle, in the flow's own coordinates. */
export type Bounds = { x: number; y: number; width: number; height: number };

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
