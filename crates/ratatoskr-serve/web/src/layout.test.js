import { expect, test } from "bun:test";

import {
  BRANCH_DROP,
  BRANCH_TAP,
  COLUMN_GAP,
  COLUMN_PITCH,
  LANE_GAP,
  LANE_PITCH,
  NODE_SIZE,
  LOOP_BAND,
  LOOP_SHELF_STEP,
  SPAN_BAND,
  SPAN_RADIUS,
  anchoredBranches,
  branchPlace,
  crowdLimit,
  fittedBounds,
  place,
  regionBands,
  rowExtent,
  spanRiser,
  spanShelf,
  spineNodes,
  tallestNeighbours,
  tapSide,
  wiredToSpine,
} from "./panels/layout";
import { carryMeasurement } from "./panels/PipelineGraph";

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

test("a lane a column does not fill still takes its room, and is centred as if it were filled", () => {
  // `lane` is a declared position, not an index into whoever turned up. A hole that closed up would
  // move every box under it somewhere the workflow's layout did not ask for.
  //
  // Both halves matter, and the second is the one that is easy to lose: a column reaching lane 2 is
  // as deep as any other column reaching lane 2, so the two sit level. Charging the hole when
  // drawing but not when measuring the column drops it half a lane below its neighbours — which is
  // what a sparse array does, since `reduce` skips its holes and `for...of` walks them.
  const placed = place([column(0, [0, 2]), column(1, [0, 1, 2])]);
  expect(placed.get("s0l2").y - placed.get("s0l0").y).toBe(2 * LANE_PITCH);
  expect(placed.get("s0l0").y).toBe(placed.get("s1l0").y);
  expect(placed.get("s0l2").y).toBe(placed.get("s1l2").y);
  expect(placed.size).toBe(5);
});

test("placing costs what the boxes cost, not what their lane numbers say", () => {
  // `lane` is bounded by the run's node count and nothing else, so a recording the read gate accepts
  // may put a single box at lane N-1 in each of N columns. Giving every lane a slot draws that in
  // time quadratic in the node count — measured at 172ms here against 4ms for the arithmetic, and
  // it is the node count that squares, so the gap widens with the shape.
  const deep = 3000;
  const columns = Array.from({ length: deep }, (_, stage) => column(stage, [deep - 1]));
  const started = performance.now();
  const placed = place(columns);
  expect(performance.now() - started).toBeLessThan(60);

  // And it places them where a filled column would: every box is at the same depth, so they are
  // level, and each is as far down as its lane says.
  expect(placed.size).toBe(deep);
  for (const box of placed.values()) expect(box.y).toBe((deep - 1) * LANE_PITCH);
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

test("a box that changed size does not keep the measurement of the one it replaced", () => {
  // React Flow owns measurement, so an update carries the last one across — otherwise it re-measures
  // every render and drops the edges it cannot route yet. But a measurement of a box that no longer
  // exists is worse than none: it is what the view is fitted to and what the edges are routed by.
  const box = (height, data) => ({
    id: "implementer",
    type: "pipeline",
    position: { x: 0, y: 0 },
    data,
    width: NODE_SIZE.width,
    height,
  });
  const mounted = {
    ...box(NODE_SIZE.height, "first"),
    measured: { width: NODE_SIZE.width, height: NODE_SIZE.height },
    internals: { handleBounds: "measured by react flow" },
  };

  // Same size, new data: everything React Flow worked out is still true.
  const updated = carryMeasurement(mounted, box(NODE_SIZE.height, "second"));
  expect(updated.data).toBe("second");
  expect(updated.measured).toEqual({ width: NODE_SIZE.width, height: NODE_SIZE.height });
  expect(updated.internals).toBe(mounted.internals);

  // Grown: the measurement follows the box, and the handle bounds survive — a node without those
  // reads as uninitialised, which is the condition the refit waits on.
  const grown = carryMeasurement(mounted, box(NODE_SIZE.height + 160, "third"));
  expect(grown.height).toBe(NODE_SIZE.height + 160);
  expect(grown.measured).toEqual({ width: NODE_SIZE.width, height: NODE_SIZE.height + 160 });
  expect(grown.internals).toBe(mounted.internals);

  // Nothing to carry from.
  expect(carryMeasurement(undefined, box(NODE_SIZE.height, "first")).measured).toBeUndefined();
});

test("the scrub magnification is bounded by the pair that has to share a gap", () => {
  // Scrubbing enlarges every box that was working at that moment, centred, so two adjacent ones
  // each reach half their own height into the gap between them. What bounds it is therefore the
  // tallest adjacent PAIR, not the tallest box — and not a constant, once a box can grow.
  const encroachment = (scale, a, b) => ((scale - 1) * (a + b)) / 2;

  // The pin: with every box the height it is today, this is the constant it replaces.
  const uniform = crowdLimit(2 * NODE_SIZE.height);
  expect(uniform).toBeCloseTo(
    Math.min(1 + (COLUMN_GAP * 0.7) / NODE_SIZE.width, 1 + (LANE_GAP * 0.7) / NODE_SIZE.height),
    12,
  );
  expect(encroachment(uniform, NODE_SIZE.height, NODE_SIZE.height)).toBeLessThan(LANE_GAP);

  // And the case that is not covered by a constant: a grown box beside a collapsed one.
  for (const grown of [140, 300, 900]) {
    const pair = grown + NODE_SIZE.height;
    // The pair never meets, and never eats more than the sliver rule allows. It may eat less: for
    // a modest growth the column gap is the binding half, exactly as it is today.
    expect(encroachment(crowdLimit(pair), grown, NODE_SIZE.height)).toBeLessThan(LANE_GAP);
    expect(encroachment(crowdLimit(pair), grown, NODE_SIZE.height)).toBeLessThan(
      LANE_GAP * 0.7 + 1e-9,
    );
    // Still a magnification worth having, and never more than the columns can take.
    expect(crowdLimit(pair)).toBeGreaterThan(1);
    expect(crowdLimit(pair)).toBeLessThanOrEqual(1 + (COLUMN_GAP * 0.7) / NODE_SIZE.width);
  }
});

test("the pair that bounds it is read off the boxes as placed", () => {
  const columns = [column(0, [0]), column(1, [0, 1, 2])];
  const heights = { s1l0: 300, s1l1: 60, s1l2: 200 };
  const placed = place(columns, (n) => heights[n.name] ?? NODE_SIZE.height);

  // 60 + 200 are adjacent and 300 + 60 are adjacent; 300 + 200 are not.
  expect(tallestNeighbours(placed.values())).toBe(360);

  // A box with nothing under or over it is in no pair, however tall: there is no lane gap for it to
  // grow into. Counting it would hold a lone grown node down to a magnification it never needed —
  // the column gap beside it is what bounds it, and `crowdLimit` applies that anyway.
  const lone = [{ x: 0, y: 0, width: NODE_SIZE.width, height: 900 }];
  expect(tallestNeighbours(lone)).toBe(0);
  expect(tallestNeighbours([])).toBe(0);
  const columnCap = 1 + (COLUMN_GAP * 0.7) / NODE_SIZE.width;
  expect(crowdLimit(tallestNeighbours(lone))).toBeCloseTo(columnCap, 12);
  expect(crowdLimit(0)).toBeCloseTo(columnCap, 12);
});

test("finding the pair costs what the boxes cost", () => {
  // The same mistake as materialising lanes, one function along: rebuilding each column as it is
  // accumulated copies it once per box, which is quadratic in a column the read gate accepts.
  const deep = 3000;
  const tall = Array.from({ length: deep }, (_, lane) => ({
    x: 0,
    y: lane * LANE_PITCH,
    width: NODE_SIZE.width,
    height: NODE_SIZE.height,
  }));
  const started = performance.now();
  expect(tallestNeighbours(tall)).toBe(2 * NODE_SIZE.height);
  expect(performance.now() - started).toBeLessThan(60);
});



test("an anchored node's vacated trailing column closes, and declared stages never move", () => {
  // `place` keys a column's x off `node.stage`, so a branch leaving its trailing column would
  // otherwise leave a column of empty viewport where it used to be. Only the synthetic stages are
  // renumbered — a declared layout keeps its columns exactly, including any hole it chose.
  const declared = (name, stage) => ({ name, state: "done", checkpoints: 1, stage, lane: 0 });
  const trailing = (name, stage, caller) => ({
    name, state: "done", checkpoints: 1, stage, lane: 0, shaped: false,
    ...(caller ? { caller } : {}),
  });
  const nodes = [
    declared("analyst", 0),
    declared("implementer", 2),
    trailing("referee", 3, "implementer"),
    trailing("stray", 4),
  ];

  const spine = spineNodes(nodes, new Set(["referee"]));
  expect(spine.map((n) => n.name)).toEqual(["analyst", "implementer", "stray"]);
  // The stray moves into the column the referee vacated.
  expect(spine.find((n) => n.name === "stray").stage).toBe(3);
  // Declared stages are untouched, hole at stage 1 included.
  expect(spine.find((n) => n.name === "analyst").stage).toBe(0);
  expect(spine.find((n) => n.name === "implementer").stage).toBe(2);
  // Nothing anchored: every node keeps exactly the stage it arrived with.
  const kept = spineNodes(nodes, new Set());
  expect(kept.map((n) => n.stage)).toEqual([0, 2, 3, 4]);
});




test("a branch child hangs directly below its parent, under the loop band", () => {
  // Directly below, because the caller edge is a straight vertical between the two branch taps.
  // Below the band, because the space beside a parent belongs to the spine and the shelves are
  // placed off the spine's extent. A child whose parent has no box is left to the caller.
  const parent = { x: 3 * COLUMN_PITCH, y: LANE_PITCH, width: NODE_SIZE.width, height: NODE_SIZE.height };
  const child = (name, caller) => ({ name, state: "done", checkpoints: 1, stage: 6, lane: 0, shaped: false, caller });
  const rowBottom = 2 * NODE_SIZE.height + LANE_GAP;

  const boxes = branchPlace(
    [child("referee", "implementer"), child("stray", "gone")],
    (name) => (name === "implementer" ? parent : undefined),
    rowBottom,
  );
  expect(boxes.get("referee")).toEqual({
    x: parent.x,
    y: rowBottom + LOOP_BAND + BRANCH_DROP,
    ...NODE_SIZE,
  });
  expect(boxes.has("stray")).toBe(false);
});

test("the branch tap clears every wire its own column draws", () => {
  // The drop runs at BRANCH_TAP of the box width, inside the column: the gaps carry the back-loop
  // shelves, so any vertical there necessarily crosses them. Within the column the wires are the
  // converge self-loop — reaching width/4 either side of the bottom centre — and the loop handles
  // at the centre itself. The tap must sit strictly between the box corner and the self-loop's
  // left edge, with a corner's clearance from each.
  const tap = BRANCH_TAP * NODE_SIZE.width;
  const selfLoopLeft = NODE_SIZE.width / 2 - NODE_SIZE.width / 4;
  expect(tap).toBeGreaterThanOrEqual(SPAN_RADIUS);
  expect(selfLoopLeft - tap).toBeGreaterThanOrEqual(SPAN_RADIUS);
});

test("only a column's deepest lane anchors, and only one child per parent", () => {
  // The caller edge is a straight drop from the parent's bottom: a box below the parent would be
  // crossed by it, and a second child would sit below the first with its drop running through it.
  // What these gates refuse keeps its trailing column, which always draws clean.
  const declared = (name, stage, lane) => ({ name, state: "done", checkpoints: 1, stage, lane });
  const trailing = (name, stage, caller) => ({
    name, state: "done", checkpoints: 1, stage, lane: 0, shaped: false,
    ...(caller ? { caller } : {}),
  });

  // The flagship shape: the implementer is its column's deepest lane, the referee anchors.
  const standard = [
    declared("redteam", 3, 0),
    declared("implementer", 3, 1),
    trailing("referee", 6, "implementer"),
  ];
  expect(anchoredBranches(standard)).toEqual(new Set(["referee"]));

  // A child of the UPPER lane is refused — the drop would cross the implementer's box.
  const upper = [
    declared("redteam", 3, 0),
    declared("implementer", 3, 1),
    trailing("aide", 6, "redteam"),
  ];
  expect(anchoredBranches(upper)).toEqual(new Set());

  // A second child of one parent is refused; the first (in list order) anchors.
  const two = [
    declared("implementer", 3, 0),
    trailing("referee", 6, "implementer"),
    trailing("second", 7, "implementer"),
  ];
  expect(anchoredBranches(two)).toEqual(new Set(["referee"]));

  // A trailing parent is its column's deepest trivially.
  const chainRoot = [
    declared("analyst", 0, 0),
    trailing("helper", 5),
    trailing("aide", 6, "helper"),
  ];
  expect(anchoredBranches(chainRoot)).toEqual(new Set(["aide"]));
});

test("the vertical regions are owned: ordered, disjoint, and every class stays home", () => {
  // The region model exists so a new wiring class collides in a failing test instead of a review
  // round: spans above the row, boxes and stage edges in it, loop shelves below, branch children
  // below that. Verticals may CROSS a band — the branch drop crosses the loop band — but
  // horizontal runs and boxes stay inside their own region.
  const rowTop = 0;
  const rowBottom = 3 * LANE_PITCH - LANE_GAP;
  const r = regionBands(rowTop, rowBottom);

  expect(r.spans.bottom).toBeLessThanOrEqual(r.row.top);
  expect(r.row.bottom).toBeLessThanOrEqual(r.loops.top);
  expect(r.loops.bottom).toBeLessThanOrEqual(r.branches.top);

  // Span shelves live in the span band, whatever the span count.
  for (const count of [1, 4, 64]) {
    for (let i = 0; i < count; i += 1) {
      const y = spanShelf(i, count, rowTop);
      expect(y).toBeGreaterThanOrEqual(r.spans.top);
      expect(y).toBeLessThan(r.spans.bottom);
    }
  }

  // The three loop shelves live in the loop band, and the converge self-loop's drop below a
  // bottom-lane box stays inside it too.
  for (const step of [1, 2, 3]) {
    const y = rowBottom + step * LOOP_SHELF_STEP;
    expect(y).toBeGreaterThan(r.loops.top);
    expect(y).toBeLessThanOrEqual(r.loops.bottom);
  }
  expect(LANE_GAP / 2).toBeLessThanOrEqual(LOOP_BAND);

  // Branch children start below the loop band.
  const parent = { x: 0, y: rowBottom - NODE_SIZE.height, width: NODE_SIZE.width, height: NODE_SIZE.height };
  const child = { name: "c", state: "done", checkpoints: 1, stage: 5, lane: 0, shaped: false, caller: "p" };
  const box = branchPlace([child], () => parent, rowBottom).get("c");
  expect(box.y).toBeGreaterThanOrEqual(r.branches.top);
});

test("branch eligibility costs what the candidates cost, not their square", () => {
  // Candidates are the imported-history shape nothing bounds. The deepest-lane check used to
  // sweep the whole node list per candidate, which froze the render for most of a second at a
  // ten-thousand-child fan-out before React Flow drew anything. Bounded generously — the
  // quadratic form measured ~800ms here against single-digit milliseconds for the indexed one.
  const nodes = [{ name: "parent", state: "done", checkpoints: 1, stage: 0, lane: 0 }];
  for (let i = 0; i < 10_000; i += 1) {
    nodes.push({
      name: `child-${i}`, state: "done", checkpoints: 1, stage: 1 + i, lane: 0,
      shaped: false, caller: "parent",
    });
  }
  const started = performance.now();
  const anchored = anchoredBranches(nodes);
  expect(performance.now() - started).toBeLessThan(100);
  // One child per parent: the fan-out anchors exactly one and the rest keep trailing columns.
  expect(anchored.size).toBe(1);
});

test("a loop participant stays on the spine, however good its caller looks", () => {
  // The loop edges and the fork hand-off route against spine handles and the spine's extent: a
  // back-loop lands on its target's BOTTOM handle from a shelf above it, so a target anchored
  // below the band would be entered straight through its own box. Anchoring moves only children;
  // the pin is on the candidate.
  const present = new Set(["implementer", "verifier", "analyst", "redteam"]);
  const has = (name) => present.has(name);

  expect(wiredToSpine({ retry: 1, fix: 0, replan: 0 }, has)).toEqual(
    new Set(["implementer", "redteam"]),
  );
  expect(wiredToSpine({ retry: 0, fix: 1, replan: 0 }, has)).toEqual(
    new Set(["implementer", "verifier", "redteam"]),
  );
  expect(wiredToSpine({ retry: 0, fix: 0, replan: 2 }, has)).toEqual(
    new Set(["verifier", "analyst", "implementer", "redteam"]),
  );
  // No loops and no fork pair: nothing pinned.
  expect(wiredToSpine({ retry: 0, fix: 0, replan: 0 }, () => false)).toEqual(new Set());

  // And the pin holds through eligibility: an unplaced analyst with a clean caller keeps its
  // trailing column while a replan loop addresses it.
  const nodes = [
    { name: "verifier", state: "done", checkpoints: 1, stage: 0, lane: 0 },
    { name: "analyst", state: "done", checkpoints: 1, stage: 5, lane: 0, shaped: false, caller: "verifier" },
  ];
  expect(anchoredBranches(nodes, new Set(["analyst"]))).toEqual(new Set());
  expect(anchoredBranches(nodes)).toEqual(new Set(["analyst"]));
});

test("the tap drops away from the side a loop shelf arrives on", () => {
  // A shelf ends at its target's bottom centre and arrives from its other endpoint's side — a
  // verifier LEFT of the implementer runs the fix shelf across the left tap. Approached from
  // both sides, or neither, the left tap stands: with both crossed there is nothing to duck.
  const center = 500;
  expect(tapSide(center, [])).toBe("left");
  expect(tapSide(center, [900])).toBe("left");
  expect(tapSide(center, [100])).toBe("right");
  expect(tapSide(center, [100, 900])).toBe("left");
});

test("both taps clear the self-loop's reach and their box corner", () => {
  // The right tap mirrors the left, and both must sit strictly between a corner and the converge
  // self-loop's reach with a corner's clearance from each — a declared layout may put the
  // approaching shelf on either side, so both sides have to be as clear as the one tap was.
  for (const tap of [BRANCH_TAP * NODE_SIZE.width, (1 - BRANCH_TAP) * NODE_SIZE.width]) {
    const fromCorner = Math.min(tap, NODE_SIZE.width - tap);
    const fromSelfLoop = Math.abs(tap - NODE_SIZE.width / 2) - NODE_SIZE.width / 4;
    expect(fromCorner).toBeGreaterThanOrEqual(SPAN_RADIUS);
    expect(fromSelfLoop).toBeGreaterThanOrEqual(SPAN_RADIUS);
  }
});
