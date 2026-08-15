import { expect, test } from "bun:test";

import {
  COLUMN_GAP,
  LOOP_BAND,
  SPAN_BAND,
  SPAN_RADIUS,
  fittedBounds,
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
