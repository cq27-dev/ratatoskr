import { expect, test } from "bun:test";

import {
  COLUMN_GAP,
  COLUMN_PITCH,
  LANE_GAP,
  LANE_PITCH,
  NODE_SIZE,
  LOOP_BAND,
  SPAN_BAND,
  SPAN_RADIUS,
  fittedBounds,
  place,
  rowExtent,
  spanRiser,
  spanShelf,
} from "./panels/layout";

/**
 * The span geometry, over every shape a workflow may declare rather than the ones that happened to
 * be rendered.
 *
 * Every defect these pin was a constant that held for the layouts it had been looked at with: a
 * per-lane step that outgrew its gap, a per-pair shelf that outgrew the band, a distribution that
 * ignored the corner it turns on. Each was found by drawing one more layout by hand. `validate_layout`
 * caps neither how wide a column may be nor how many hand-offs a run may make, so the numbers have to
 * hold for any of them.
 */
const LANE_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 20, 64];

test("a riser stands inside its column gap for any number of lanes", () => {
  for (const lanes of LANE_COUNTS) {
    for (let k = 0; k < lanes; k += 1) {
      const at = spanRiser(k, lanes);
      expect(at).toBeGreaterThan(0);
      expect(at).toBeLessThan(COLUMN_GAP);
    }
  }
});

test("a riser clears the corner it turns on, at both ends of the gap", () => {
  // Nearer its box than the radius and the turn's control point lands behind the handle: the path
  // enters the node it just left before doubling back out. The far end is the target's box.
  for (const lanes of LANE_COUNTS) {
    for (let k = 0; k < lanes; k += 1) {
      const at = spanRiser(k, lanes);
      expect(at).toBeGreaterThanOrEqual(SPAN_RADIUS);
      expect(COLUMN_GAP - at).toBeGreaterThanOrEqual(SPAN_RADIUS);
    }
  }
});

test("no two lanes of a column raise their risers on the same line", () => {
  for (const lanes of LANE_COUNTS) {
    const seen = new Set();
    for (let k = 0; k < lanes; k += 1) seen.add(spanRiser(k, lanes));
    expect(seen.size).toBe(lanes);
  }
});

test("risers keep their lanes' order, so spans do not cross on their way up", () => {
  for (const lanes of LANE_COUNTS) {
    for (let k = 1; k < lanes; k += 1) {
      expect(spanRiser(k, lanes)).toBeGreaterThan(spanRiser(k - 1, lanes));
    }
  }
});

test("every shelf sits inside the band, for any number of spans", () => {
  // The fitted view reserves exactly this depth. A shelf outside it is clipped until someone pans,
  // and adding a span changes no node, so nothing refits to reveal it.
  for (const count of LANE_COUNTS) {
    for (let i = 0; i < count; i += 1) {
      const y = spanShelf(i, count, 0);
      expect(y).toBeLessThan(0);
      expect(y).toBeGreaterThanOrEqual(-SPAN_BAND);
    }
  }
});

test("no two spans share a shelf", () => {
  // Sharing one drew any number of hand-offs as a single line: the horizontal segments coincide.
  for (const count of LANE_COUNTS) {
    const seen = new Set();
    for (let i = 0; i < count; i += 1) seen.add(spanShelf(i, count, 0));
    expect(seen.size).toBe(count);
  }
});

test("the band's depth does not grow with the number of spans", () => {
  // The property the per-pair stacking broke: one span and sixty-four reach the same distance up.
  const one = spanShelf(0, 1, 0);
  const many = spanShelf(63, 64, 0);
  expect(Math.abs(one)).toBeLessThanOrEqual(SPAN_BAND);
  expect(Math.abs(many)).toBeLessThanOrEqual(SPAN_BAND);
});

test("a lane index out of range is placed rather than escaping the gap", () => {
  // Defensive: `findIndex` returns -1 for a name the column does not carry, and a negative offset
  // would put the riser inside the box on the far side.
  const at = spanRiser(-1, 3);
  expect(at).toBeGreaterThanOrEqual(SPAN_RADIUS);
  expect(COLUMN_GAP - at).toBeGreaterThanOrEqual(SPAN_RADIUS);
  expect(spanShelf(0, 0, 0)).toBeGreaterThanOrEqual(-SPAN_BAND);
});

test("a fit covers everything hanging off the boxes, at both ends", () => {
  // `fitView` fits node bounds and leaves the rest to padding — a fraction of what it fits, so a
  // short row gets less room than a fixed band needs. Reserving only one end is how the loops came
  // to be clipped: the span band went into the bounds and the padding covering the loops was
  // reduced to pay for it.
  const boxes = { x: 0, y: 0, width: 800, height: 270 };

  const both = fittedBounds(boxes, SPAN_BAND, LOOP_BAND);
  expect(both.y).toBe(-SPAN_BAND);
  expect(both.y + both.height).toBe(270 + LOOP_BAND);

  // Every shelf a span can take is inside it, whatever the span count.
  for (const count of [1, 4, 9, 64]) {
    for (let i = 0; i < count; i += 1) {
      expect(spanShelf(i, count, boxes.y)).toBeGreaterThanOrEqual(both.y);
    }
  }

  // One end alone still covers that end, and leaves the other where the boxes are.
  const spansOnly = fittedBounds(boxes, SPAN_BAND, 0);
  expect(spansOnly.y).toBe(-SPAN_BAND);
  expect(spansOnly.y + spansOnly.height).toBe(270);

  const loopsOnly = fittedBounds(boxes, 0, LOOP_BAND);
  expect(loopsOnly.y).toBe(0);
  expect(loopsOnly.y + loopsOnly.height).toBe(270 + LOOP_BAND);

  // The width is the boxes': nothing hangs off the sides.
  expect(both.x).toBe(boxes.x);
  expect(both.width).toBe(boxes.width);
});

/** A column of boxes, named for their lanes. */
const column = (stage, lanes) => lanes.map((lane) => ({ name: `s${stage}l${lane}`, stage, lane }));

const boxes = (placed) => [...placed.values()];

test("with every box the same height, the layout is the fixed pitch it has always been", () => {
  // The pin. Stacking heights and multiplying a pitch are the same layout while nothing has grown,
  // and this is the layout every rendered case on this branch was checked against.
  const columns = [column(0, [0]), column(1, [0, 1]), column(2, [0, 1, 2])];
  const placed = place(columns);
  const maxLanes = 3;
  for (const lanes of columns) {
    for (const n of lanes) {
      const offset = (maxLanes - lanes.length) / 2;
      expect(placed.get(n.name)).toEqual({
        x: n.stage * COLUMN_PITCH,
        y: (n.lane + offset) * LANE_PITCH,
        ...NODE_SIZE,
      });
    }
  }
});

test("a lane a column does not fill still takes its room", () => {
  // `lane` is a declared position, not an index into whoever turned up. A hole that closed up would
  // move every box under it somewhere the workflow's layout did not ask for.
  const placed = place([column(0, [0, 2])]);
  expect(placed.get("s0l2").y - placed.get("s0l0").y).toBe(2 * LANE_PITCH);
  expect(placed.size).toBe(2);
});

test("a taller box moves the lanes under it rather than drawing over them", () => {
  const columns = [column(0, [0, 1, 2])];
  const grown = (n) => (n.lane === 0 ? NODE_SIZE.height * 3 : NODE_SIZE.height);
  const placed = place(columns, grown);

  let previous = null;
  for (const lane of [0, 1, 2]) {
    const box = placed.get(`s0l${lane}`);
    expect(box.height).toBe(grown({ lane }));
    if (previous) expect(box.y - (previous.y + previous.height)).toBe(LANE_GAP);
    previous = box;
  }
});

test("a box growing moves nothing in another column, and the row re-centres", () => {
  const columns = [column(0, [0]), column(1, [0, 1])];
  const flat = place(columns);
  const grown = place(columns, (n) => (n.stage === 1 && n.lane === 0 ? 300 : NODE_SIZE.height));

  // The one that grew keeps its lane order and pushes only its own sibling.
  expect(grown.get("s1l1").y - grown.get("s1l0").y).toBe(300 + LANE_GAP);
  // Its column is now the tallest, so the single-box column centres against it — which is the
  // same rule that puts a lone analyst level with a fork today, applied to a bigger fork.
  const last = grown.get("s1l1");
  expect(grown.get("s0l0").y + NODE_SIZE.height / 2).toBeCloseTo(
    (grown.get("s1l0").y + last.y + last.height) / 2,
    6,
  );
  expect(flat.get("s0l0").x).toBe(grown.get("s0l0").x);
});

test("no two boxes in a column overlap, at any mix of heights", () => {
  const columns = [column(0, [0, 1, 2, 3, 4, 5, 6, 7])];
  for (const heights of [
    (n) => NODE_SIZE.height,
    (n) => NODE_SIZE.height * (1 + n.lane),
    (n) => (n.lane % 2 ? 400 : 40),
    (n) => 1,
  ]) {
    const placed = place(columns, heights);
    const sorted = boxes(placed).sort((a, b) => a.y - b.y);
    for (let i = 1; i < sorted.length; i += 1) {
      expect(sorted[i].y).toBeGreaterThanOrEqual(sorted[i - 1].y + sorted[i - 1].height);
    }
  }
});

test("the row's extent covers every box, whatever grew", () => {
  // What the shelves hang off and the view fits. A box outside it is a box the fit can clip.
  const columns = [column(0, [0]), column(1, [0, 1, 2])];
  for (const heights of [undefined, (n) => (n.lane === 2 ? 500 : NODE_SIZE.height)]) {
    const placed = place(columns, heights);
    const { top, bottom } = rowExtent(placed.values());
    for (const box of boxes(placed)) {
      expect(box.y).toBeGreaterThanOrEqual(top);
      expect(box.y + box.height).toBeLessThanOrEqual(bottom);
    }
    // The bands hang off these, so they have to be the boxes' own edges and not a floor.
    expect(bottom).toBe(Math.max(...boxes(placed).map((b) => b.y + b.height)));
  }
});
