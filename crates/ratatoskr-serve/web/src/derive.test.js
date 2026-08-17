import { expect, test } from "bun:test";

import {
  applyDerived,
  branchParent,
  convergeLoops,
  forkHandoff,
  contiguous,
  handoffEvidence,
  handoffDrawn,
  inNodeBoxes,
  nodesFromEvents,
  skippedSpans,
  stagesOf,
  transitions,
  workingNodeNames,
} from "./derive";

function node(name, state) {
  return { name, state, checkpoints: 0, stage: 0, lane: 0 };
}

/** A `node_start`, which is the only kind the loop classifier reads. */
function start(name) {
  return { at: "2026-08-12T10:00:00Z", kind: "node_start", node: name, detail: "node started" };
}

/** The checkpoint that finishes a node — and that makes the server start placing it. */
function checkpointed(name) {
  return { at: "2026-08-12T10:00:01Z", kind: "checkpoint", node: name, detail: "checkpoint" };
}

test("a live delivery stage replaces the inferred implementer control target", () => {
  const nodes = [
    node("implementer", "working"),
    node("bookkeeper", "idle"),
    node("publisher", "idle"),
  ];
  const events = [
    {
      at: "2026-08-12T10:00:00Z",
      kind: "checkpoint",
      node: "implementer",
      detail: "checkpoint",
    },
    {
      at: "2026-08-12T10:00:01Z",
      kind: "node_start",
      node: "publisher",
      detail: "node started",
    },
  ];

  expect(workingNodeNames(nodes, events)).toEqual(["publisher"]);
});

test("the stored pipeline remains the fallback before a node announces itself", () => {
  expect(workingNodeNames([node("analyst", "working")], [])).toEqual(["analyst"]);
});

test("a completed live attempt does not revive a stale stored target", () => {
  const events = [
    {
      at: "2026-08-12T10:00:00Z",
      kind: "node_start",
      node: "publisher",
      detail: "node started",
    },
    {
      at: "2026-08-12T10:00:01Z",
      kind: "checkpoint",
      node: "publisher",
      detail: "checkpoint",
    },
  ];

  expect(workingNodeNames([node("implementer", "working")], events)).toEqual([]);
});

test("the initial implement() call is not a loop, so nothing draws x1", () => {
  expect(convergeLoops([start("analyst"), start("implementer")], [])).toEqual({
    fix: 0,
    replan: 0,
    retry: 0,
  });
});

test("a verifier between two implementer starts is a direct fix", () => {
  expect(convergeLoops([start("implementer"), start("verifier"), start("implementer")], [])).toEqual({
    fix: 1,
    replan: 0,
    retry: 0,
  });
});

// `iterate_host` runs the referee unconditionally (workflow.rs:915), including on the
// tests-not-clean path that reaches `iterate({})` without `verify()` ever running. So a referee
// start on its own is a failed-test retry, and must never be read as the verifier's fix.
test("a referee with no verifier is a retry, because the referee runs on both paths", () => {
  expect(convergeLoops([start("implementer"), start("referee"), start("implementer")], [])).toEqual({
    fix: 0,
    replan: 0,
    retry: 1,
  });
});

test("the common real fix cycle through verifier then referee is a direct fix", () => {
  const events = [
    start("implementer"),
    start("verifier"),
    start("referee"),
    start("implementer"),
  ];
  expect(convergeLoops(events, [])).toEqual({ fix: 1, replan: 0, retry: 0 });
});

test("an analyst re-run makes it a replan even though the verifier also ran", () => {
  const events = [
    start("implementer"),
    start("verifier"),
    start("analyst"),
    start("implementer"),
  ];
  expect(convergeLoops(events, [])).toEqual({ fix: 0, replan: 1, retry: 0 });
});

test("re-entering with no verifier and no analyst is a retry", () => {
  expect(convergeLoops([start("implementer"), start("implementer")], [])).toEqual({
    fix: 0,
    replan: 0,
    retry: 1,
  });
});

test("nodes outside the loop shape do not change the classification", () => {
  const events = [
    start("implementer"),
    start("characterizer"),
    start("verifier"),
    // A name this classifier has never seen must fall through the same way.
    start("archivist"),
    start("implementer"),
    start("characterizer"),
    start("implementer"),
  ];
  expect(convergeLoops(events, [])).toEqual({ fix: 1, replan: 0, retry: 1 });
});

test("a run shaped like 414fb163 counts one of each, retry included", () => {
  // The real run's three segments, in order: the tests never went clean and the
  // implementer ran again with only the referee in between; then a verifier fix;
  // then a verifier finding that faulted the plan and went back through the analyst.
  const events = [
    start("scout"),
    start("analyst"),
    start("implementer"),
    start("characterizer"),
    start("referee"),
    start("implementer"),
    start("referee"),
    start("verifier"),
    start("implementer"),
    start("referee"),
    start("verifier"),
    start("analyst"),
    start("implementer"),
    start("publisher"),
  ];
  expect(convergeLoops(events, [])).toEqual({ fix: 1, replan: 1, retry: 1 });
});

test("counts are those of the prefix given, never the run's final totals", () => {
  const events = [
    start("implementer"),
    start("verifier"),
    start("implementer"),
    start("verifier"),
    start("analyst"),
    start("implementer"),
  ];
  const at = (n) => convergeLoops(events.slice(0, n), []);

  expect(at(1)).toEqual({ fix: 0, replan: 0, retry: 0 });
  expect(at(2)).toEqual({ fix: 0, replan: 0, retry: 0 });
  expect(at(3)).toEqual({ fix: 1, replan: 0, retry: 0 });
  expect(at(5)).toEqual({ fix: 1, replan: 0, retry: 0 });
  expect(at(6)).toEqual({ fix: 1, replan: 1, retry: 0 });
});

test("events belonging to no node are skipped", () => {
  const events = [
    start("implementer"),
    { at: "2026-08-12T10:00:00Z", kind: "run_started", node: null, detail: "run started" },
    start("verifier"),
    { at: "2026-08-12T10:00:01Z", kind: "node_start", node: null, detail: "node started" },
    start("implementer"),
  ];
  expect(convergeLoops(events, [])).toEqual({ fix: 1, replan: 0, retry: 0 });
});

// The implementer cannot start before the red team has finished (`implement_host` in
// ratatoskr-nodes/src/workflow.rs refuses to), so both boxes having started is the whole test.
test("both the red team and the implementer having started draws the hand-off", () => {
  expect(forkHandoff([node("redteam", "done"), node("implementer", "working")], null)).toBe(true);
});

test("a started red team alone draws no hand-off, because nothing has received the tree", () => {
  expect(forkHandoff([node("redteam", "working"), node("implementer", "idle")], null)).toBe(false);
});

test("neither node having started draws no hand-off", () => {
  expect(forkHandoff([node("redteam", "idle"), node("implementer", "idle")], null)).toBe(false);
});

test("a workflow with no red team at all draws no hand-off from nothing", () => {
  expect(forkHandoff([node("analyst", "done"), node("implementer", "working")], null)).toBe(false);
});

// The edge is a vertical step down the lane gap between two boxes in one column. A layout that
// puts them in different columns already joins them the ordinary way, and this one would render
// as a diagonal across the graph on top of it.
test("a layout that puts the two in different columns draws no lane hand-off", () => {
  const nodes = [
    { ...node("redteam", "done"), stage: 0 },
    { ...node("implementer", "working"), stage: 2 },
  ];
  expect(forkHandoff(nodes, null)).toBe(false);
});

test("sharing a column is what draws it", () => {
  const nodes = [
    { ...node("redteam", "done"), stage: 3, lane: 0 },
    { ...node("implementer", "working"), stage: 3, lane: 1 },
  ];
  expect(forkHandoff(nodes, null)).toBe(true);
});

test("a failed red team still handed the tree over, so the hand-off is drawn", () => {
  expect(forkHandoff([node("redteam", "failed"), node("implementer", "working")], null)).toBe(true);
});

/** A node the shape places, at a column of its own. */
function placed(name, stage) {
  return { name, state: "idle", checkpoints: 0, stage, lane: 0 };
}

const applied = (shape, events) => applyDerived(shape, nodesFromEvents(events));

/** What the server sends for a node that reported nothing, so a case can vary one field of it. */
const blankTelemetry = {
  model: null,
  turns: null,
  input_tokens: 0,
  output_tokens: 0,
  cached_input_tokens: 0,
  cache_creation_input_tokens: 0,
  reasoning_tokens: 0,
  thinking: false,
  duration_ms: null,
  tools: [],
  tools_used: [],
  reuses_session: false,
  first_at: null,
  last_at: null,
};

// A workflow that declares no layout records an empty shape, so the server can only place a node
// once it has checkpointed. Until then the stream is the only thing that knows it exists.
test("a node the shape does not place still gets a box while it works", () => {
  // Marked unshaped: the column is this side's ordering, and what is drawn into it turns on that.
  expect(applied([], [start("analyst")])).toEqual([
    {
      name: "analyst",
      state: "working",
      checkpoints: 0,
      stage: 0,
      lane: 0,
      shaped: false,
      // What the stream saw, kept apart from `state` so a settle cannot take it back.
      entered: true,
    },
  ]);
});

test("an appended node lands in a column of its own, after everything the shape placed", () => {
  const view = applied([placed("context", 0)], [start("context"), start("analyst")]);
  expect(view.map((n) => [n.name, n.stage])).toEqual([
    ["context", 0],
    ["analyst", 1],
  ]);
});

// `issue` records what the run was asked to do, which is not a stage of doing it — the server
// leaves it out of the shape for exactly that reason.
test("the issue record is not a node and gets no box", () => {
  const events = [{ at: "2026-08-12T10:00:00Z", kind: "checkpoint", node: "issue" }];
  expect(applied([], events)).toEqual([]);
});

// A run whose workflow declared a layout is drawn by that layout and nothing else.
test("a shaped run draws its shape, unchanged", () => {
  const shape = [placed("context", 0), placed("analyst", 1)];
  const view = applied(shape, [start("context"), start("analyst")]);
  expect(view.map((n) => [n.name, n.stage, n.lane])).toEqual([
    ["context", 0, 0],
    ["analyst", 1, 0],
  ]);
});

// The costed rule is the appended node's too: derived zeros are only shown once the stream has
// actually reported a cost.
test("an appended node reports what the stream says it cost", () => {
  const events = [
    start("analyst"),
    {
      at: "2026-08-12T10:00:01Z",
      kind: "usage",
      node: "analyst",
      usage: { input_tokens: 7, output_tokens: 3 },
    },
  ];
  const [analyst] = applied([], events);
  expect(analyst.telemetry.input_tokens).toBe(7);
});

/** A node the server appended from a checkpoint rather than from a declared layout. */
function appended(name, stage) {
  return { ...placed(name, stage), shaped: false };
}

/**
 * A run with no declared layout runs its hosts concurrently, so the order they finish in is not
 * the order they started in. The server can only place a node once it has checkpointed, and it
 * orders those by first checkpoint; the stream is the only record of what started when. Adopting
 * the server's numbers would move a box the moment its checkpoint arrived — and since column
 * adjacency is what the hand-off arrow is drawn from, the arrow would then point backwards.
 */
test("an unshaped box keeps its place when the server starts placing it", () => {
  const events = [start("slow"), start("fast"), checkpointed("fast")];
  expect(applied([], events.slice(0, 2)).map((n) => n.name)).toEqual(["slow", "fast"]);
  expect(applied([appended("fast", 0)], events).map((n) => [n.name, n.stage])).toEqual([
    ["slow", 0],
    ["fast", 1],
  ]);
});

// A workflow that declared a layout is drawn by it, whatever order the stream saw.
test("a declared layout is not reordered by the stream", () => {
  const shape = [placed("fast", 0), placed("slow", 1)];
  const view = applied(shape, [start("slow"), start("fast")]);
  expect(view.map((n) => [n.name, n.stage])).toEqual([
    ["fast", 0],
    ["slow", 1],
  ]);
});

// An appended node's column is the client's ordering, not a hand-off the shape declared. Drawing
// one into it turned the referee an iterating run appends — invoked by the implementer
// mid-converge — into a final stage judging what the publisher had just published.
test("no forward edge is drawn from a shaped column into an appended node", () => {
  expect(handoffDrawn(placed("publisher", 5), appended("referee", 6))).toBe(false);
});

// The order the stream first saw each node is all the ordering a layout-less run has, so its own
// chain stays.
test("an appended node still hands off to the next appended node", () => {
  expect(handoffDrawn(appended("transform", 0), appended("publish_docs", 1))).toBe(true);
});

test("an appended node that names its caller receives no adjacency edge", () => {
  // The chain is the least wrong claim available only where nothing better exists. In a dynamic
  // chain X -> A -> B where A anchored under X and B stayed trailing, the closed-up columns make
  // X and B neighbours — and the chain edge then asserts X -> B beside caller edges that say
  // otherwise. A target with a caller takes exactly one in-edge: the caller's.
  const called = { ...appended("aide", 1), caller: "helper" };
  expect(handoffDrawn(appended("helper", 0), called)).toBe(false);
  expect(handoffDrawn(placed("analyst", 0), called)).toBe(false);
  // The caller-bearing node still hands off DOWN the chain, where the next node has no better
  // claim of its own.
  expect(handoffDrawn(called, appended("scribe", 2))).toBe(true);
});

test("a declared layout's own hand-offs are drawn in full", () => {
  expect(handoffDrawn(placed("analyst", 2), placed("implementer", 3))).toBe(true);
});

/** A tool call: a node working, with no checkpoint to follow it. */
function working(name) {
  return { at: "2026-08-12T10:00:02Z", kind: "tool_call", node: name, detail: "read_file" };
}

// A host error writes no checkpoint for the node it kills and no node-scoped event, so the fold
// leaves it "working" and the run emits nothing more to move it. That node is therefore the
// evidence: at the end of a failed run it is the one that started and never finished.
//
// This used to take the SERVER's state for such a node, and the server derived it from position —
// which stage the implementer sat in, what followed it, whether the verifier had a route. The state
// the server sends is now ignored where the stream names a candidate, so the shape below says
// "done" and the box still reads failed.
test("a failed run marks the one node its stream left working", () => {
  const shape = [placed("analyst", 0), { ...placed("implementer", 1), state: "done" }];
  const view = applyDerived(
    shape,
    nodesFromEvents([start("analyst"), checkpointed("analyst"), start("implementer")]),
    "failed",
  );
  expect(view.map((n) => [n.name, n.state])).toEqual([
    ["analyst", "done"],
    ["implementer", "failed"],
  ]);
});

// The case the deleted machinery existed for: converge died on a later iteration, so the
// implementer holds a checkpoint from an earlier one and was then re-entered. The store can only
// see the checkpoint and says "done"; the stream saw the re-entry, and it is the whole difference.
test("a converge death still marks the implementer, which its checkpoints deny", () => {
  const shape = [
    { ...placed("redteam", 0), state: "done" },
    { ...placed("implementer", 0), lane: 1, state: "done" },
    { ...placed("bookkeeper", 1), state: "idle" },
  ];
  const events = [
    start("redteam"),
    checkpointed("redteam"),
    start("implementer"),
    checkpointed("implementer"),
    start("implementer"),
  ];
  const view = applyDerived(shape, nodesFromEvents(events), "failed");
  expect(view.map((n) => [n.name, n.state])).toEqual([
    ["redteam", "done"],
    ["implementer", "failed"],
    ["bookkeeper", "idle"],
  ]);
});

// A verifier that died after the fork. The store cannot name it — the implementer's re-entry fits
// the same checkpoints — so it sends "idle" and the box used to draw grey on a run that plainly
// failed in it. The stream saw the verifier start and never finish, which names it outright.
test("a verifier that died after the fork is marked, not left grey", () => {
  const shape = [
    { ...placed("implementer", 0), state: "done" },
    { ...placed("verifier", 1), state: "idle" },
  ];
  const events = [start("implementer"), checkpointed("implementer"), start("verifier")];
  const view = applyDerived(shape, nodesFromEvents(events), "failed");
  expect(view.map((n) => [n.name, n.state])).toEqual([
    ["implementer", "done"],
    ["verifier", "failed"],
  ]);
});

// Two shaped hosts in flight when one of them died. The run's status says the run died, never which
// of them died in it, so neither is named — the same answer the store gives past a fork that ran.
test("a failed run with two shaped nodes still working names neither", () => {
  const shape = [
    { ...placed("implementer", 0), state: "done" },
    { ...placed("deploy", 1), state: "idle" },
  ];
  const events = [start("implementer"), start("deploy")];
  const view = applyDerived(shape, nodesFromEvents(events), "failed");
  expect(view.map((n) => [n.name, n.state])).toEqual([
    ["implementer", "done"],
    ["deploy", "idle"],
  ]);
});

// The same run, scrubbed back into the middle of itself. A node that genuinely WAS working then
// still reads working — settling it against the store there is the same lie as showing a run's
// final state at step one.
test("scrubbed into the middle of a stopped run, a working node still reads working", () => {
  const shape = [placed("analyst", 0), { ...placed("implementer", 1), state: "failed" }];
  const view = applyDerived(
    shape,
    nodesFromEvents([start("analyst"), checkpointed("analyst"), start("implementer")]),
    null,
  );
  expect(view.find((n) => n.name === "implementer").state).toBe("working");
});

// A stopped run says nothing about a node that finished: the stream saw the checkpoint and is the
// finer record, above all for the implementer, whose rows come one per converge iteration.
test("only a working node is settled — a checkpointed one keeps what the stream saw", () => {
  const shape = [{ ...placed("analyst", 0), state: "idle" }];
  const view = applyDerived(shape, nodesFromEvents([start("analyst"), checkpointed("analyst")]), "converged");
  expect(view[0].state).toBe("done");
});

// With no declared layout the server places a node only once it has checkpointed, so the node the
// host died under has no server row at all. The run's status is then the only record of it, and
// without this a failed run draws every box green and nothing wrong.
test("a failed run marks the node it died in even when the store has no row for it", () => {
  const events = [start("ingest"), checkpointed("ingest"), start("publish"), working("publish")];
  const view = applyDerived([appended("ingest", 0)], nodesFromEvents(events), "failed");
  expect(view.map((n) => [n.name, n.state])).toEqual([
    ["ingest", "done"],
    ["publish", "failed"],
  ]);
});

// A layout-less workflow may run its hosts concurrently, and if one dies before any of them
// checkpoints, none of them has a server row to be settled against. The run's status says the run
// died — never which of them died in it — so a reader must not be shown three failed boxes for one
// failure. Only an unambiguous candidate is named.
test("a failed run does not mark every concurrent host it has no row for", () => {
  const events = [start("lint"), working("lint"), start("build"), working("build")];
  const view = applyDerived([], nodesFromEvents(events), "failed");
  expect(view.map((n) => [n.name, n.state])).toEqual([
    ["lint", "idle"],
    ["build", "idle"],
  ]);
});

// One of the two finished before the run died, so the other is the only candidate left and is
// still named — withholding attribution the record does support would lose the one thing this
// correction exists to show.
test("a failed run still marks the one host left working among several", () => {
  const events = [start("lint"), checkpointed("lint"), start("build"), working("build")];
  const view = applyDerived([], nodesFromEvents(events), "failed");
  expect(view.map((n) => [n.name, n.state])).toEqual([
    ["lint", "done"],
    ["build", "failed"],
  ]);
});

// A run from another graph: the shape names some of its nodes and not others. The named ones stay
// where the shape puts them; only the rest are the client's to order.
test("a partly shaped run keeps its shaped nodes and orders the rest by the stream", () => {
  const shape = [placed("context", 0), appended("fast", 1)];
  const events = [start("context"), start("slow"), start("fast"), checkpointed("fast")];
  expect(applied(shape, events).map((n) => [n.name, n.stage])).toEqual([
    ["context", 0],
    ["slow", 1],
    ["fast", 2],
  ]);
});

/** A box drawn from several stages, as the server places it once it has checkpointed. */
function composed(name, state) {
  return { name, state, checkpoints: 0, stage: 0, lane: 0 };
}

/**
 * The run's recorded registry, from `[box, ...members]` entries.
 *
 * A box that is one stage names itself. This is what the run writes down, and it is written before
 * the first stage runs — which is why membership is read from here and never from `nodes`.
 */
function registry(...boxes) {
  return boxes.flatMap(([node, ...members]) =>
    (members.length ? members : [node]).map((id) => ({ id, node })),
  );
}

test("a box's member stages are folded into it rather than drawn beside it", () => {
  // The regression the whole pairing exists to prevent, and it is name-matched, so it fails
  // quietly: a red-team half starts under its own id, and without membership `applyDerived` gives
  // that name a trailing column of its own next to the box it belongs to.
  const shape = [composed("redteam", "idle")];
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"]);
  const events = [start("redteam_classifier"), checkpointed("redteam_classifier")];

  const strays = applied(shape, events);
  expect(strays.map((n) => n.name)).toEqual(["redteam", "redteam_classifier"]);

  const drawn = applied(shape, inNodeBoxes(events, stages));
  expect(drawn.map((n) => n.name)).toEqual(["redteam"]);
  // WORKING, not done: the classifier finishing is the box started, and the author has not run.
  // This case asserted `done` while the server said otherwise for the same records — the box's own
  // aggregate is what completes it, on both sides.
  expect(drawn[0].state).toBe("working");
  expect(drawn[0].checkpoints).toBe(0);
});

test("a box is done when its own record lands, and counts only its own", () => {
  const shape = [composed("redteam", "idle")];
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"]);
  const events = [
    start("redteam_classifier"),
    checkpointed("redteam_classifier"),
    start("redteam_author"),
    checkpointed("redteam_author"),
    checkpointed("redteam"),
  ];
  const drawn = applied(shape, inNodeBoxes(events, stages));
  expect(drawn[0].state).toBe("done");
  // One, not three. The server counts the box's own rows, and a number that changes when a log
  // rotates away is a number nobody can read.
  expect(drawn[0].checkpoints).toBe(1);
});

test("a box that is itself a stage keeps working while a peer composed into it runs", () => {
  // A membership target may be "a stage of its own called `{node}`" (`ratatoskr-nodes/src/
  // validate.rs`), so a workflow may compose `security` into the peer stage `review` and invoke
  // both hosts at once. The box's own record is then ALSO one member's, and reading completion off
  // the name alone marks the box done on whichever host finishes first — taking the Stop aimed at
  // the one still running with it.
  const shape = [composed("review", "idle")];
  const stages = registry(["review", "review", "security"]);
  const events = [start("review"), start("security"), checkpointed("review")];

  const boxed = inNodeBoxes(events, stages);
  const drawn = applied(shape, boxed);
  expect(drawn.map((n) => n.name)).toEqual(["review"]);
  expect(drawn[0].state).toBe("working");
  // The control address, which is what a Stop is aimed at.
  expect(workingNodeNames(shape, boxed)).toEqual(["review"]);

  // And it completes when the peer lands: the box's own record exists and nothing is left running.
  const whole = inNodeBoxes([...events, checkpointed("security")], stages);
  expect(applied(shape, whole)[0].state).toBe("done");
  expect(workingNodeNames(shape, whole)).toEqual([]);
});

test("a stage invoked twice at once is working until both invocations have returned", () => {
  // `Promise.all([probe(a), probe(b)])` is two LIVE invocations of one stage, and both record
  // under that stage's one name. The first checkpoint ends an invocation, never the stage: reading
  // it as the stage finishing marks the box done and drops the Stop aimed at the one still going.
  const shape = [composed("probe", "idle")];
  const stages = registry(["probe"]);
  const events = [start("probe"), start("probe"), checkpointed("probe")];

  const boxed = inNodeBoxes(events, stages);
  expect(applied(shape, boxed)[0].state).toBe("working");
  // The control address: what Stop and Steer are offered against while the second one runs.
  expect(workingNodeNames(shape, boxed)).toEqual(["probe"]);

  // The negative control. One invocation and one checkpoint finish the box — without this the case
  // above would read the same against a box that simply never finishes.
  const once = inNodeBoxes([start("probe"), checkpointed("probe")], stages);
  expect(applied(shape, once)[0].state).toBe("done");
  expect(workingNodeNames(shape, once)).toEqual([]);

  // And the second invocation's own record is what finishes the concurrent pair.
  const both = inNodeBoxes([...events, checkpointed("probe")], stages);
  expect(applied(shape, both)[0].state).toBe("done");
  expect(workingNodeNames(shape, both)).toEqual([]);
});

test("a box whose aggregate has landed is still working where a member has re-entered", () => {
  // The stale completion, and it is the same arithmetic: the implementer is re-driven per converge
  // iteration, so its attempt can be running again by the time the aggregate for the previous one
  // lands. The box's own record exists and the member's latest record says `done`, and neither
  // fact is about the invocation currently executing.
  const shape = [composed("implementer", "idle")];
  const stages = registry(["implementer", "implementer_attempt"]);
  const events = [
    start("implementer_attempt"),
    start("implementer_attempt"),
    checkpointed("implementer_attempt"),
    checkpointed("implementer"),
  ];

  const boxed = inNodeBoxes(events, stages);
  const drawn = applied(shape, boxed);
  expect(drawn[0].state).toBe("working");
  expect(drawn[0].checkpoints).toBe(1);
  expect(workingNodeNames(shape, boxed)).toEqual(["implementer"]);
});

test("a start the run died under does not keep a stopped run's box working", () => {
  // The count is only as balanced as the stream, and a host that dies writes no checkpoint — so a
  // box with an invocation still counted live reads working and the stream will say nothing more.
  // That is the case the terminal settle already exists for, and it still reaches it: this is the
  // same position as a stage that never checkpointed at all.
  const shape = [composed("probe", "idle")];
  const stages = registry(["probe"]);
  const derived = nodesFromEvents(
    inNodeBoxes([start("probe"), start("probe"), checkpointed("probe")], stages),
  );
  expect(derived.get("probe").state).toBe("working");
  expect(applyDerived(shape, derived, "failed")[0].state).toBe("failed");
});

test("a stage executing with nothing checkpointed yet is already drawn in its box", () => {
  // The live window, and the one that matters: an operator reaches for Stop while a stage is
  // running. The server derives `nodes` from checkpoints, so the box it belongs to is NOT in that
  // list yet — a mapping read from there is empty for exactly the box being looked at, the half
  // draws as its own box, and the Stop offered goes to a name the runtime never polls. The
  // registry is the run's own record and says so before anything has finished.
  const stages = registry(["context", "context_distillation"], ["analyst"]);
  const events = [start("context_distillation")];
  // What the server has placed: nothing. No checkpoint has landed.
  const shape = [];

  const boxed = inNodeBoxes(events, stages);
  const drawn = applied(shape, boxed);
  expect(drawn.map((n) => n.name)).toEqual(["context"]);
  expect(drawn[0].state).toBe("working");
  // The control address, which is the half of this that reaches the run.
  expect(workingNodeNames(shape, boxed)).toEqual(["context"]);
  // And the feed filter for that box, asked while it has no row of its own: its member's name and
  // its own, because the box's aggregate and its acceptance output are logged under the box.
  expect(stagesOf(stages, "context")).toEqual(["context_distillation", "context"]);
});

test("reading events as boxes leaves the events themselves alone", () => {
  // The feed shows which half ran — that is what the split bought. Only the fold is renamed.
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"]);
  const events = [start("redteam_author")];
  expect(events[0].node).toBe("redteam_author");
  expect(inNodeBoxes(events, stages)[0].node).toBe("redteam");
  expect(events[0].node).toBe("redteam_author");

  // A stage that is its own box keeps its name, and so does one the registry never heard of. Both
  // still carry `member`, because a fold has to be able to tell a member's record from its box's
  // whether or not a rename happened.
  const plain = registry(["analyst"], ["scout"]);
  const untouched = [start("analyst"), start("scout"), start("characterizer")];
  const read = inNodeBoxes(untouched, plain);
  expect(read.map((e) => e.node)).toEqual(["analyst", "scout", "characterizer"]);
  expect(read.map((e) => e.member)).toEqual(["analyst", "scout", "characterizer"]);
  expect(untouched.some((e) => "member" in e)).toBe(false);
});

test("a re-entry is counted from the box, whichever stage announced it", () => {
  // `convergeLoops` reads `implementer` starts. The implementer's turn now announces itself as
  // `implementer_attempt`, so counting the raw stream would report a run that looped as one that
  // went straight through.
  const stages = registry(["implementer", "implementer_attempt"], ["verifier"]);
  const events = [
    start("implementer_attempt"),
    start("verifier"),
    start("implementer_attempt"),
  ];
  expect(convergeLoops(events, [])).toEqual({ fix: 0, replan: 0, retry: 0 });
  expect(convergeLoops(inNodeBoxes(events, stages), [])).toEqual({ fix: 1, replan: 0, retry: 0 });
});

test("the live map is keyed by the box, so the box draws with what its member announced", () => {
  // The pre-checkpoint window again, from the other side. `PipelineGraph` asks `live.get(box)`, so
  // a map folded from the raw stream is keyed by the member and the box draws with no model, no
  // tools and no cycle count for as long as the member is the one running — which is the whole of
  // that window.
  const stages = registry(["context", "context_distillation"]);
  const announced = {
    at: "2026-08-12T10:00:00Z",
    kind: "node_start",
    node: "context_distillation",
    detail: "",
    facts: { model: "anthropic/claude-sonnet-5", tools: ["Read"], thinking: true, reuses_session: false },
  };
  const called = {
    at: "2026-08-12T10:00:01Z",
    kind: "tool_call",
    node: "context_distillation",
    detail: "Read",
  };
  const events = [announced, called];

  const raw = nodesFromEvents(events);
  expect([...raw.keys()]).toEqual(["context_distillation"]);
  expect(raw.get("context")).toBeUndefined();

  const boxed = nodesFromEvents(inNodeBoxes(events, stages));
  expect([...boxed.keys()]).toEqual(["context"]);
  const box = boxed.get("context");
  expect(box.telemetry.model).toBe("anthropic/claude-sonnet-5");
  expect(box.cycles).toBe(1);
  expect([...box.used]).toEqual(["Read"]);
});

test("the live map keeps a member's activity when a sibling starts beside it", () => {
  // Two stages of one box in flight at once — `Promise.all([classify(x), author(y)])`, which the
  // workflow may compose and the fold beside this one already handles. Both announce under the
  // same box, so a map that keeps one entry per box lets the later start throw away everything the
  // earlier member has done. This is the window before any checkpoint, where the live map is all
  // the box has to draw with, so that work does not merely move — it disappears from the screen.
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"]);
  const facts = (model) => ({ model, tools: ["Read"], thinking: false, reuses_session: false });
  const announced = (name, model) => ({ ...start(name), facts: facts(model) });
  const called = (name, tool) => ({
    at: "2026-08-12T10:00:02Z",
    kind: "tool_call",
    node: name,
    detail: tool,
  });
  const events = [
    announced("redteam_classifier", "anthropic/claude-haiku-5"),
    called("redteam_classifier", "Grep"),
    announced("redteam_author", "anthropic/claude-sonnet-5"),
    called("redteam_author", "Write"),
  ];

  const box = nodesFromEvents(inNodeBoxes(events, stages)).get("redteam");
  expect(box.cycles).toBe(2);
  expect([...box.used].sort()).toEqual(["Grep", "Write"]);
  // And the same for what they announced: a box running two profiles names both, exactly as the
  // checkpointed fold beside this one does.
  expect(box.telemetry.model).toBe("anthropic/claude-haiku-5, anthropic/claude-sonnet-5");

  // Per MEMBER, not per invocation: a member re-entered starts its own counts again, and #285 is
  // where a second invocation of one stage gets numbers of its own.
  const again = [...events, announced("redteam_classifier", "anthropic/claude-haiku-5")];
  const reentered = nodesFromEvents(inNodeBoxes(again, stages)).get("redteam");
  expect(reentered.cycles).toBe(1);
  expect([...reentered.used]).toEqual(["Write"]);
});

test("a member re-entering restarts its counts whether or not it announces facts", () => {
  // `facts` is optional on a `node_start`. Restarting only when they are present made a re-entry
  // that announced nothing accumulate onto the previous attempt, so the box drew a cycle count that
  // was two attempts added together — while the checkpointed fold beside it, which restarts on
  // every `node_start`, drew the second attempt alone. Same stream, two answers.
  const stages = registry(["analyst"]);
  const called = (tool) => ({ at: "t", kind: "tool_call", node: "analyst", detail: tool });
  const events = [
    { ...start("analyst"), facts: { model: "m", tools: [], thinking: false, reuses_session: false } },
    called("Read"),
    called("Grep"),
    start("analyst"),
    called("Write"),
  ];

  const boxed = inNodeBoxes(events, stages);
  const live = nodesFromEvents(boxed).get("analyst");
  expect(live.cycles).toBe(1);
  expect([...live.used]).toEqual(["Write"]);
  // And what it announced first survives a restart that announced nothing — the model is a fact
  // about the member, not about the attempt.
  expect(live.telemetry.model).toBe("m");
});

test("a usage event costs the member whatever it reports, zero included", () => {
  // Scrub honesty, which is what this whole derivation exists for. `fromStream` keeps the server's
  // telemetry — the run's FINAL state — for a node the stream never costed, so a node left
  // uncosted at an earlier position displays a later attempt's model, tokens and tools.
  //
  // A `usage` event is the endpoint's own report of a turn: its presence is the authority, and a
  // zero is a measurement rather than an absence. Only a CHECKPOINT has to be doubted, because a
  // box's turn-less aggregate carries the keys as zeros whether or not anything ran.
  const stages = registry(["analyst"]);
  const quiet = {
    at: "t2",
    kind: "usage",
    node: "analyst",
    detail: "",
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      cached_input_tokens: 0,
      cache_creation_input_tokens: 0,
      reasoning_tokens: 0,
      duration_ms: 40,
    },
  };
  const derived = nodesFromEvents(inNodeBoxes([start("analyst"), quiet], stages)).get("analyst");
  expect(derived.costed).toBe(true);

  // And the fold is then what the box shows, rather than the run's end reaching back into an
  // earlier position.
  const server = [
    {
      name: "analyst",
      state: "done",
      checkpoints: 2,
      stage: 0,
      lane: 0,
      telemetry: { ...blankTelemetry, model: "the later attempt", input_tokens: 900 },
    },
  ];
  const drawn = applyDerived(server, nodesFromEvents(inNodeBoxes([start("analyst"), quiet], stages)));
  expect(drawn[0].telemetry.input_tokens).toBe(0);
  expect(drawn[0].telemetry.duration_ms).toBe(40);
});

test("a checkpoint is costed when a turn happened, not when it spent tokens", () => {
  // Some endpoints make a real call and omit token accounting: the checkpoint carries `turns`, a
  // model and a duration while all five counters read zero. Asking "did it spend" leaves that
  // member uncosted, `fromStream` substitutes the store's FINAL telemetry, and scrubbing back to an
  // earlier attempt of a repeated node shows a later attempt's model, tokens and tools.
  //
  // "Did a turn happen" is the actual question and the record answers it: the server omits `facts`
  // from a checkpoint whose node ran no model (`LiveNodeFacts::of`), so their presence IS the turn.
  // Token counters are the wrong basis anyway — `reasoning_tokens` is hardcoded to zero by the
  // provider and `output_tokens` is under-reported.
  const stages = registry(["analyst"]);
  const noCounts = {
    at: "t2",
    kind: "checkpoint",
    node: "analyst",
    detail: "",
    facts: { model: "an endpoint that counts nothing", tools: ["Read"], thinking: false, reuses_session: false },
    turns: 1,
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      cached_input_tokens: 0,
      cache_creation_input_tokens: 0,
      reasoning_tokens: 0,
      duration_ms: 90,
    },
  };
  const events = inNodeBoxes([start("analyst"), noCounts], stages);
  expect(nodesFromEvents(events).get("analyst").costed).toBe(true);

  const server = [
    {
      name: "analyst",
      state: "done",
      checkpoints: 2,
      stage: 0,
      lane: 0,
      telemetry: { ...blankTelemetry, model: "the later attempt", input_tokens: 900 },
    },
  ];
  const drawn = applyDerived(server, nodesFromEvents(events));
  expect(drawn[0].telemetry.model).toBe("an endpoint that counts nothing");
  expect(drawn[0].telemetry.input_tokens).toBe(0);

  // And the box's own aggregate still is not a turn — now because it reports no cost at all rather
  // than because a guard doubted the zeros it used to carry. A record covering no turn carries none
  // of the usage keys, so the server sends no `usage` for it, and there is nothing to mistake for a
  // measurement of nothing.
  const aggregate = {
    at: "t3",
    kind: "checkpoint",
    node: "redteam",
    detail: "",
  };
  const composed = registry(["redteam", "redteam_classifier"]);
  expect(nodesFromEvents(inNodeBoxes([aggregate], composed)).get("redteam").costed).toBe(false);
});

test("a box's cost is the fold of its members, not whichever record landed last", () => {
  // Two defects in one: the box's own aggregate carries a `usage` block of zeros because it covers
  // no turn, and two members each carry their own. Overwriting on every checkpoint reported the
  // zeros; replacing rather than folding reported only the last member.
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"]);
  const spent = (node, at, input, model) => ({
    at,
    kind: "checkpoint",
    node,
    detail: "",
    facts: { model, tools: ["Read"], thinking: false, reuses_session: false },
    usage: {
      input_tokens: input,
      output_tokens: 1,
      cached_input_tokens: 0,
      cache_creation_input_tokens: 0,
      reasoning_tokens: 0,
      duration_ms: 10,
    },
    turns: 1,
  });
  const aggregate = {
    at: "t3",
    kind: "checkpoint",
    node: "redteam",
    detail: "",
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      cached_input_tokens: 0,
      cache_creation_input_tokens: 0,
      reasoning_tokens: 0,
      duration_ms: 0,
    },
  };
  const events = [
    spent("redteam_classifier", "t1", 20, "sonnet"),
    spent("redteam_author", "t2", 10, "haiku"),
    aggregate,
  ];
  const derived = nodesFromEvents(inNodeBoxes(events, stages)).get("redteam");
  expect(derived.telemetry.input_tokens).toBe(30);
  expect(derived.telemetry.output_tokens).toBe(2);
  expect(derived.telemetry.turns).toBe(2);
  // Both routes named, as the server names them, rather than the later one silently winning.
  expect(derived.telemetry.model).toBe("sonnet, haiku");
  // The aggregate's zeros are not a report of cost, so they neither establish `costed` nor let a
  // zeroed row displace the server's folded figure.
  expect(derived.costed).toBe(true);
  expect(derived.state).toBe("done");
  expect(derived.checkpoints).toBe(1);
});

test("a working box shows every member's model and tools, not the latest one's", () => {
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"]);
  const announced = (node, at, model, tool) => ({
    at,
    kind: "node_start",
    node,
    detail: "",
    facts: { model, tools: [tool], thinking: false, reuses_session: false },
  });
  const events = [
    announced("redteam_classifier", "t1", "sonnet", "Read"),
    announced("redteam_author", "t2", "haiku", "Write"),
  ];
  const derived = nodesFromEvents(inNodeBoxes(events, stages)).get("redteam");
  expect(derived.telemetry.model).toBe("sonnet, haiku");
  expect(derived.telemetry.tools).toEqual(["Read", "Write"]);
  expect(derived.state).toBe("working");
});

test("a control is aimed at the box the stage runs inside", () => {
  // The run polls for a Stop under the node's name, so this has to name the same thing — a control
  // offered against `redteam_classifier` would be sent to an address nothing answers.
  const shape = [composed("redteam", "idle")];
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"]);
  const events = [start("redteam_classifier")];
  expect(workingNodeNames(shape, inNodeBoxes(events, stages))).toEqual(["redteam"]);
});

test("the stages of a box are what its feed is filtered by", () => {
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"], ["analyst"]);
  // Its members AND its own name: `redteam` is not a stage id, so a members-only answer drops the
  // aggregate rows and the acceptance suite, both logged under the box.
  expect(stagesOf(stages, "redteam")).toEqual([
    "redteam_classifier",
    "redteam_author",
    "redteam",
  ]);
  // A box that is one stage, and a name the registry does not carry at all, are both exactly
  // themselves.
  expect(stagesOf(stages, "analyst")).toEqual(["analyst"]);
  expect(stagesOf(stages, "characterizer")).toEqual(["characterizer"]);
});


/** A node at a column, with a state. */
function at(name, stage, state) {
  return { name, state, checkpoints: 0, stage, lane: 0 };
}

test("a run that skipped a stage spans the gap it actually jumped", () => {
  // The no-code-change shortcut: the analyst decides there is nothing to build, and the run goes
  // straight to delivery. An edge joins adjacent columns only, so the graph drew a break exactly
  // where the run made its hand-off.
  const nodes = [
    at("analyst", 0, "done"),
    at("redteam", 1, "idle"),
    at("implementer", 2, "idle"),
    at("verifier", 3, "idle"),
    at("publisher", 4, "done"),
  ];
  expect(skippedSpans(nodes)).toEqual([{ from: "analyst", to: "publisher" }]);
});

test("a stage that has not run yet is not a stage that was skipped", () => {
  // Mid-flight, and the same shape scrubbing back through a finished run: everything ahead is idle
  // for the ordinary reason. Nothing later has started, so there is no jump to assert.
  const nodes = [
    at("analyst", 0, "done"),
    at("redteam", 1, "working"),
    at("implementer", 2, "idle"),
    at("verifier", 3, "idle"),
    at("publisher", 4, "idle"),
  ];
  expect(skippedSpans(nodes)).toEqual([]);
});

test("a stage that looked skipped and then ran is spanned no longer", () => {
  const skipped = [
    at("analyst", 0, "done"),
    at("verifier", 1, "idle"),
    at("publisher", 2, "done"),
  ];
  expect(skippedSpans(skipped)).toEqual([{ from: "analyst", to: "publisher" }]);

  const ranAfterAll = skipped.map((n) => (n.name === "verifier" ? { ...n, state: "done" } : n));
  expect(skippedSpans(ranAfterAll)).toEqual([]);
});

test("a run that used every stage spans nothing", () => {
  const nodes = [
    at("analyst", 0, "done"),
    at("redteam", 1, "done"),
    at("implementer", 2, "done"),
    at("publisher", 3, "done"),
  ];
  expect(skippedSpans(nodes)).toEqual([]);
});

test("a span joins every box of one column to every box of the next that ran", () => {
  // The same relation an adjacent-column edge draws: a fork rejoining is many-to-many, and a
  // skipped stage does not change what a hand-off between two columns means.
  const nodes = [
    at("analyst", 0, "done"),
    at("redteam", 1, "idle"),
    { ...at("bookkeeper", 2, "done"), lane: 0 },
    { ...at("publisher", 2, "done"), lane: 1 },
  ];
  expect(skippedSpans(nodes)).toEqual([
    { from: "analyst", to: "bookkeeper" },
    { from: "analyst", to: "publisher" },
  ]);
});

test("a box the run never entered is not an endpoint of a span", () => {
  // Two boxes share the column the run jumped to, and only one of them ran.
  const nodes = [
    at("analyst", 0, "done"),
    at("redteam", 1, "idle"),
    { ...at("bookkeeper", 2, "idle"), lane: 0 },
    { ...at("publisher", 2, "done"), lane: 1 },
  ];
  expect(skippedSpans(nodes)).toEqual([{ from: "analyst", to: "publisher" }]);
});

test("a stage two nodes died in is not a stage the run skipped", () => {
  // A failed run with two uncheckpointed nodes in flight blames neither — attribution would be a
  // guess — so `applyDerived` settles both back to their stored state, which is `idle`. Their
  // `node_start` events still prove the stage ran, and reading rendered state alone would assert a
  // hand-off straight across it: the one thing that certainly did not happen, drawn over the boxes
  // where the run actually died.
  const shape = [
    { name: "analyst", state: "done", checkpoints: 1, stage: 0, lane: 0, shaped: true },
    { name: "redteam", state: "idle", checkpoints: 0, stage: 1, lane: 0, shaped: true },
    { name: "implementer", state: "idle", checkpoints: 0, stage: 1, lane: 1, shaped: true },
    { name: "publisher", state: "done", checkpoints: 1, stage: 2, lane: 0, shaped: true },
  ];
  const stages = registry(["analyst"], ["redteam"], ["implementer"], ["publisher"]);
  const events = [
    checkpointed("analyst"),
    start("redteam"),
    start("implementer"),
    checkpointed("publisher"),
  ];
  const view = applyDerived(shape, nodesFromEvents(inNodeBoxes(events, stages)), "failed");

  const at = (name) => view.find((n) => n.name === name);
  expect(at("redteam").state).toBe("idle");
  expect(at("implementer").state).toBe("idle");
  expect(at("redteam").entered).toBe(true);
  expect(at("implementer").entered).toBe(true);
  expect(skippedSpans(view)).toEqual([]);
});

/** A `node_start` for one execution, with the model and cost that attempt will report. */
/** A tool call, optionally from a named execution. */
function called(name, tool, span) {
  return {
    at: "2026-08-12T10:00:02Z",
    kind: "tool_call",
    node: name,
    detail: tool,
    ...(span ? { span_id: span } : {}),
  };
}

function attemptStart(name, span, model) {
  return {
    at: "2026-08-12T10:00:00Z",
    kind: "node_start",
    node: name,
    detail: "node started",
    span_id: span,
    facts: { model, tools: ["Read"], thinking: false, reuses_session: false },
  };
}

function attemptCheckpoint(name, span, tokens) {
  return {
    at: "2026-08-12T10:00:05Z",
    kind: "checkpoint",
    node: name,
    detail: "checkpoint",
    span_id: span,
    turns: 1,
    usage: {
      input_tokens: tokens,
      output_tokens: tokens,
      cached_input_tokens: 0,
      cache_creation_input_tokens: 0,
      reasoning_tokens: 0,
      duration_ms: tokens,
    },
  };
}

test("an attempt's figures are its own, not the ones the attempt before it reported", () => {
  // The implementer is re-driven once per converge pass, and every attempt records under the one
  // name. Keyed by name, a second attempt opened holding the first one's model, tokens and duration
  // — so a viewer scrubbed back to the moment the second one started saw the first one's figures
  // against it, and the box's cost was two attempts added together.
  const stages = registry(["implementer"]);
  const first = "00000000000000a1";
  const second = "00000000000000b2";
  const events = [
    attemptStart("implementer", first, "opus"),
    attemptCheckpoint("implementer", first, 100),
    attemptStart("implementer", second, "sonnet"),
    attemptCheckpoint("implementer", second, 7),
  ];

  const of = (prefix) =>
    nodesFromEvents(inNodeBoxes(prefix, stages)).get("implementer").telemetry;

  // Where the run WAS: the first attempt, and nothing of the second.
  const one = of(events.slice(0, 2));
  expect(one.model).toBe("opus");
  expect(one.input_tokens).toBe(100);

  // The moment the second starts: its own model, and none of the first's cost carried into it.
  const opened = of(events.slice(0, 3));
  expect(opened.model).toBe("sonnet");
  expect(opened.input_tokens).toBe(0);
  // Nothing measured yet reads as nothing measured: a fresh attempt has no duration, and `null`
  // is how this fold says that — a zero would be a claim that it took no time.
  expect(opened.duration_ms).toBe(null);

  // And when it finishes, its own figures — not the two attempts added together.
  const both = of(events);
  expect(both.model).toBe("sonnet");
  expect(both.input_tokens).toBe(7);
  expect(both.duration_ms).toBe(7);
});

test("two invocations of one stage keep their own figures while both are live", () => {
  // `Promise.all([probe(a), probe(b)])`. Both record under one name at once, so without an identity
  // on each record there is nothing to say which invocation a cost belongs to — the later record
  // simply overwrote the earlier one's.
  const stages = registry(["probe"]);
  const a = "00000000000000aa";
  const b = "00000000000000bb";
  const events = [
    attemptStart("probe", a, "opus"),
    attemptStart("probe", b, "sonnet"),
    attemptCheckpoint("probe", a, 50),
  ];
  const boxed = inNodeBoxes(events, stages);

  // The box is still working: one invocation returned, the other has not.
  expect(nodesFromEvents(boxed).get("probe").state).toBe("working");
  expect(workingNodeNames([composed("probe", "idle")], boxed)).toEqual(["probe"]);

  // The current invocation is the one still running, and it reports what IT was given — the
  // returning sibling's cost is not attributed to it.
  const live = nodesFromEvents(boxed).get("probe").telemetry;
  expect(live.model).toBe("sonnet");
  expect(live.input_tokens).toBe(0);
});

test("a record whose start this view never saw gets its own attempt", () => {
  // An ingested log may begin mid-run. A checkpoint for an execution with no `node_start` in view
  // must not be charged to whichever attempt happens to be open — that attributes a turn to an
  // invocation that did not run it.
  const stages = registry(["analyst"]);
  const events = [
    attemptStart("analyst", "00000000000000a1", "opus"),
    attemptCheckpoint("analyst", "00000000000000c3", 42),
  ];
  const box = nodesFromEvents(inNodeBoxes(events, stages)).get("analyst");
  // The invocation this view watched start is still running, so it is what the box shows — and the
  // stray record's cost is NOT charged to it, which is the whole point.
  expect(box.state).toBe("working");
  expect(box.telemetry.model).toBe("opus");
  expect(box.telemetry.input_tokens).toBe(0);

  // And a record about an invocation is not evidence that the invocation is running: the stray
  // opened no LIVE attempt, so once the one that started ends, the box ends with it. Nothing can
  // ever close an attempt this view never saw start, and one left live works forever.
  const after = nodesFromEvents(
    inNodeBoxes([...events, attemptCheckpoint("analyst", "00000000000000a1", 3)], stages),
  ).get("analyst");
  expect(after.state).toBe("done");
  // Whichever of the two it shows, it shows ONE of them. 45 would mean two invocations added
  // together, which is what keying by name did.
  expect([3, 42]).toContain(after.telemetry.input_tokens);
});

test("a tool call from the older of two live invocations is not shown against the newer", () => {
  // The graph draws live cycles and tools from the same fold the boxes come from — it used to
  // come from a second one, member-keyed, which showed the wrong thing. `Promise.all` overlaps two
  // invocations of one stage: A's tool call arrives after B has started, and keyed by name it lands
  // on B — which is drawn as the box's live activity.
  const stages = registry(["probe"]);
  const a = "00000000000000aa";
  const b = "00000000000000bb";
  const called = (span, tool) => ({
    at: "2026-08-12T10:00:02Z",
    kind: "tool_call",
    node: "probe",
    detail: tool,
    span_id: span,
  });
  const events = [
    attemptStart("probe", a, "opus"),
    called(a, "Read"),
    attemptStart("probe", b, "sonnet"),
    called(a, "Grep"),
    called(b, "Write"),
  ];
  const boxed = inNodeBoxes(events, stages);

  // The current invocation is B, and what it has done is its own — one call, not three.
  const live = nodesFromEvents(boxed).get("probe");
  expect(live.cycles).toBe(1);
  expect([...live.used]).toEqual(["Write"]);
  expect(live.telemetry.model).toBe("sonnet");

});

test("a running attempt shows its own pending figures, not the finished one's", () => {
  // The last hop, and the one that decided what a viewer actually sees. A fresh attempt has
  // reported no cost yet, and reading that as "the stream cannot say" hands the box back the
  // SERVER's telemetry — which is the run's final state. The graph then drew the first attempt's
  // model, tokens and duration against the second while it ran.
  const stages = registry(["implementer"]);
  const first = "00000000000000a1";
  const second = "00000000000000b2";
  const events = [
    attemptStart("implementer", first, "opus"),
    attemptCheckpoint("implementer", first, 100),
    attemptStart("implementer", second, "sonnet"),
  ];
  // What the server holds is the run's final state: the first attempt's record.
  const served = [
    {
      name: "implementer",
      state: "done",
      checkpoints: 1,
      stage: 0,
      lane: 0,
      shaped: true,
      telemetry: { ...blankTelemetry, model: "opus", input_tokens: 100, duration_ms: 100 },
    },
  ];

  const drawn = applyDerived(served, nodesFromEvents(inNodeBoxes(events, stages)))[0];
  expect(drawn.state).toBe("working");
  expect(drawn.telemetry.model).toBe("sonnet");
  expect(drawn.telemetry.input_tokens).toBe(0);

  // And where the stream genuinely cannot answer — an ingested tail with no start in view — the
  // server's record is still what fills the gap.
  const tail = applyDerived(
    served,
    nodesFromEvents(inNodeBoxes([called("implementer", "Read")], stages)),
  )[0];
  expect(tail.telemetry.model).toBe("opus");
});

test("an event about a running node does not open an invocation nothing can close", () => {
  // `acceptance_step` names a node and, before the host recorded one, no execution. It opened an
  // attempt that no checkpoint matched — the aggregate carries the host call's identity — and that
  // attempt stayed live, so the box read working for the rest of the run with its controls still
  // offered. A record ABOUT an invocation is not evidence that one is running.
  const stages = registry(["implementer"]);
  const host = "00000000000000e5";
  const events = [
    { at: "t", kind: "acceptance_step", node: "implementer", detail: "cargo test" },
    attemptCheckpoint("implementer", host, 10),
  ];
  const box = nodesFromEvents(inNodeBoxes(events, stages)).get("implementer");
  expect(box.checkpoints).toBe(1);
  expect(box.state).toBe("done");
});

test("an invocation that writes no checkpoint is ended by its own span_end", () => {
  // An evidence-only stage, and a turn whose failure the workflow recovered from: executions that
  // end without a record of their own, so nothing in the stream could close them and the box worked
  // for the rest of the run with its Stop still offered.
  //
  // The lifecycle event closes them. It names an execution and no node — a host call is an
  // execution the shape cannot place — so it is matched by identity wherever that execution is.
  const shape = [composed("characterizer", "idle")];
  const stages = registry(["characterizer"]);
  const span = "00000000000000f6";
  const host = "00000000000000e6";
  const start = { ...attemptStart("characterizer", span, "opus"), parent_span_id: host };

  const running = inNodeBoxes([start], stages);
  expect(applied(shape, running)[0].state).toBe("working");
  expect(workingNodeNames(shape, running)).toEqual(["characterizer"]);

  // The turn's own end alone settles nothing: the stage validates and checkpoints AFTER the model
  // turn returns, and any of that can still fail it.
  const turnOnly = inNodeBoxes(
    [
      start,
      { at: "t", kind: "span_end", outcome: "completed", span_id: span, execution: "node", execution_name: "characterizer" },
    ],
    stages,
  );
  expect(applied(shape, turnOnly)[0].state).toBe("working");

  // The host call closing clean IS the stage boundary, and is what finishes the box.
  const ended = inNodeBoxes(
    [
      start,
      { at: "t", kind: "span_end", outcome: "completed", span_id: span, execution: "node", execution_name: "characterizer" },
      { at: "t", kind: "span_end", outcome: "completed", span_id: host, execution: "host", execution_name: "characterize" },
    ],
    stages,
  );
  expect(applied(shape, ended)[0].state).toBe("done");
  expect(workingNodeNames(shape, ended)).toEqual([]);

  // And a host that ERRORED after the turn — validation, normalisation, the checkpoint write —
  // does not close clean, so the stage stays a candidate for the failure it caused.
  const failedAfter = inNodeBoxes(
    [
      start,
      { at: "t", kind: "span_end", outcome: "completed", span_id: span, execution: "node", execution_name: "characterizer" },
      { at: "t", kind: "span_end", outcome: "unvalidated", span_id: host, execution: "host", execution_name: "characterize" },
    ],
    stages,
  );
  expect(applied(shape, failedAfter)[0].state).toBe("working");
  expect(applyDerived(shape, nodesFromEvents(failedAfter), "failed")[0].state).toBe("failed");
});

test("a composed member ending does not finish the box its host is still driving", () => {
  // `implementer_attempt` announces its end BEFORE the host that drove it runs the suite and writes
  // the aggregate. Reading one member's end as the box's would drop the implementer's working state
  // for the window in between — and a graph that draws hand-offs from state would draw one.
  const shape = [composed("implementer", "idle")];
  const stages = registry(["implementer", "implementer_attempt"]);
  const span = "00000000000000d4";
  const mid = inNodeBoxes(
    [
      attemptStart("implementer_attempt", span, "opus"),
      { at: "t", kind: "span_end", outcome: "completed", span_id: span, execution: "node", execution_name: "implementer_attempt" },
    ],
    stages,
  );
  expect(applied(shape, mid)[0].state).toBe("working");

  // The host's own record is what finishes it, as it always was.
  const whole = inNodeBoxes(
    [
      { ...attemptStart("implementer_attempt", span, "opus"), parent_span_id: "00000000000000d5" },
      { at: "t", kind: "span_end", outcome: "completed", span_id: span, execution: "node", execution_name: "implementer_attempt" },
      checkpointed("implementer"),
      // The host closes after the aggregate it wrote, and its clean end is what settles the
      // members it drove.
      { at: "t", kind: "span_end", outcome: "completed", span_id: "00000000000000d5", execution: "host", execution_name: "implement" },
    ],
    stages,
  );
  expect(applied(shape, whole)[0].state).toBe("done");
});

test("an end that arrived first keeps the outcome it carried", () => {
  // Arriving first does not change what a record said. The early-end path used to settle "done"
  // regardless, so a cancelled or unvalidated end read as success purely because the tail began at
  // the end record — and the node left the candidates its failed run is attributed among.
  const shape = [composed("characterizer", "idle")];
  const stages = registry(["characterizer"]);
  const span = "00000000000000f9";
  const tail = inNodeBoxes(
    [
      {
        at: "t",
        kind: "span_end",
        outcome: "cancelled",
        span_id: span,
        execution: "node",
        execution_name: "characterizer",
      },
      called("characterizer", "Read", span),
    ],
    stages,
  );

  expect(nodesFromEvents(tail).get("characterizer").state).toBe("working");
  expect(applyDerived(shape, nodesFromEvents(tail), "failed")[0].state).toBe("failed");
});

test("an end seen before anything else of its execution still ends it", () => {
  // The guard emitting `span_end` drops as the execution leaves, which is BEFORE its caller writes
  // anything about it — and an imported tail can begin anywhere, so the end may be the first record
  // of that execution in view. Dropped, whatever follows opens an attempt that never ended, and a
  // box with no aggregate of its own reads working for the rest of the run with its Stop offered.
  const shape = [composed("characterizer", "idle")];
  const stages = registry(["characterizer"]);
  const span = "00000000000000e5";
  const host = "00000000000000e7";
  const tail = inNodeBoxes(
    [
      { at: "t", kind: "span_end", outcome: "completed", span_id: span, execution: "node", execution_name: "characterizer" },
      { ...called("characterizer", "Read", span), parent_span_id: host },
      { at: "t", kind: "span_end", outcome: "completed", span_id: host, execution: "host", execution_name: "characterize" },
    ],
    stages,
  );

  expect(nodesFromEvents(tail).get("characterizer").state).toBe("done");
  expect(workingNodeNames(shape, tail)).toEqual([]);
});

test("a finished invocation does not stand in for one still running", () => {
  // Two invocations overlap and the SECOND finishes first. Taking the newest attempt outright drew
  // its model and its tools while the one still going went unseen — and its later tool calls landed
  // on an attempt nobody was looking at.
  const stages = registry(["probe"]);
  const a = "00000000000000aa";
  const b = "00000000000000bb";
  const events = [
    attemptStart("probe", a, "opus"),
    attemptStart("probe", b, "sonnet"),
    attemptCheckpoint("probe", b, 5),
    called("probe", "Read", a),
  ];
  const boxed = inNodeBoxes(events, stages);

  const box = nodesFromEvents(boxed).get("probe");
  expect(box.state).toBe("working");
  expect(box.telemetry.model).toBe("opus");
  const live = nodesFromEvents(boxed).get("probe");
  expect([...live.used]).toEqual(["Read"]);
  expect(live.telemetry.model).toBe("opus");
});

test("folding a long history costs what the history costs", () => {
  // An imported bundle has no bound on how many times a stage was invoked. Searching a member's
  // attempts per record is quadratic in that number, and the fold runs on every render and every
  // scrub of the timeline.
  const stages = registry(["implementer"]);
  const many = 20_000;
  const events = [];
  for (let n = 1; n <= many; n += 1) {
    const span = `${n}`.padStart(16, "0");
    events.push(attemptStart("implementer", span, "opus"), attemptCheckpoint("implementer", span, 1));
  }
  const boxed = inNodeBoxes(events, stages);

  const started = performance.now();
  const box = nodesFromEvents(boxed).get("implementer");
  nodesFromEvents(boxed);
  expect(performance.now() - started).toBeLessThan(500);
  expect(box.checkpoints).toBe(many);
  // The last invocation's own figures, not the sum of twenty thousand.
  expect(box.telemetry.input_tokens).toBe(1);
});

test("a wide fan-out under one host costs what the children cost", () => {
  // The shapes above vary the ATTEMPT count; this one varies the children registered under one
  // parent, which is its own accumulation — and rebuilding the sibling list per child is quadratic
  // in a host's fan-out. A host may drive any number of invocations, an imported history is
  // unbounded, and the fold runs on every render and every scrub.
  const stages = registry(["probe"]);
  const host = "00000000000000d9";
  const many = 20_000;
  const events = [];
  for (let n = 1; n <= many; n += 1) {
    const span = `${n}`.padStart(16, "0");
    events.push(
      { ...attemptStart("probe", span, "opus"), parent_span_id: host },
      {
        at: "t",
        kind: "span_end",
        outcome: "completed",
        span_id: span,
        execution: "node",
        execution_name: "probe",
      },
    );
  }
  events.push({
    at: "t",
    kind: "span_end",
    outcome: "completed",
    span_id: host,
    execution: "host",
    execution_name: "probe",
  });
  const boxed = inNodeBoxes(events, stages);

  const started = performance.now();
  const box = nodesFromEvents(boxed).get("probe");
  expect(performance.now() - started).toBeLessThan(500);
  // And the host's clean close settled every one of them.
  expect(box.state).toBe("done");
});

test("overlapping invocations cost what they cost to end", () => {
  // The existing history test alternates start and checkpoint, so only one invocation is ever live
  // and removing it is free. A history may instead hold many at once — and removing from the middle
  // of the live set costs a scan and a shift, which is quadratic across N starts and N ends.
  const stages = registry(["probe"]);
  const many = 50_000;
  const spans = Array.from({ length: many }, (_, n) => `${n + 1}`.padStart(16, "0"));
  const events = [
    ...spans.map((span) => attemptStart("probe", span, "opus")),
    ...spans.map((span) => attemptCheckpoint("probe", span, 1)),
  ];
  const boxed = inNodeBoxes(events, stages);

  const started = performance.now();
  const box = nodesFromEvents(boxed).get("probe");
  nodesFromEvents(boxed);
  expect(performance.now() - started).toBeLessThan(500);
  expect(box.checkpoints).toBe(many);
  expect(box.state).toBe("done");
});

test("a start that announces nothing shows nothing, not the run's final figures", () => {
  // A `node_start` carries facts only sometimes. Where the stream watched an attempt begin and has
  // nothing to say about it yet, that IS the answer — the server's record is the run's final state,
  // and leaving it in place draws the previous attempt's model and tokens against a fresh one.
  const stages = registry(["implementer"]);
  const bare = { at: "t", kind: "node_start", node: "implementer", detail: "node started" };
  const served = [
    {
      name: "implementer",
      state: "done",
      checkpoints: 1,
      stage: 0,
      lane: 0,
      shaped: true,
      telemetry: { ...blankTelemetry, model: "opus", input_tokens: 100 },
    },
  ];

  const drawn = applyDerived(served, nodesFromEvents(inNodeBoxes([bare], stages)))[0];
  expect(drawn.state).toBe("working");
  expect(drawn.telemetry).toBeUndefined();
});

test("a clarification answerer is not offered as a control target", () => {
  // The answerer's turn runs on the ASKING node's control — a Stop during a clarification ends the
  // asking turn, which is the point of blocking it. So nothing addressed to the answerer's own box
  // is ever polled: offering it hands an operator a button that does nothing, which reads as an
  // option that was ignored.
  const stages = registry(["analyst"], ["implementer"]);
  const answering = {
    at: "t",
    kind: "node_start",
    node: "analyst",
    detail: "node started",
    span_id: "00000000000000b2",
    parent_span_id: "00000000000000a1",
    controlled_as: "implementer",
    facts: { model: "opus", tools: [], thinking: false, reuses_session: false },
  };
  const boxed = inNodeBoxes([answering], stages);

  // It IS working, and drawn as working — a viewer should see the run is doing something.
  expect(nodesFromEvents(boxed).get("analyst").state).toBe("working");
  // It is not something a control can reach.
  expect(workingNodeNames([composed("analyst", "idle")], boxed)).toEqual([]);

  // An ordinary turn, whose control is its own box, still is.
  const ordinary = inNodeBoxes([attemptStart("analyst", "00000000000000c3", "opus")], stages);
  expect(workingNodeNames([composed("analyst", "idle")], ordinary)).toEqual(["analyst"]);

  // An answerer abandoned by a Stop resolves QUIETLY. Its turn was cancelled because the ASKING
  // node was stopped — the failure story is the asker's — and a box left "working" by it reads as
  // live forever, and stands as a stale second candidate that blocks a later failure's attribution.
  const abandoned = inNodeBoxes(
    [
      answering,
      {
        at: "t",
        kind: "span_end",
        outcome: "cancelled",
        span_id: "00000000000000b2",
        execution: "node",
        execution_name: "analyst",
      },
    ],
    stages,
  );
  expect(nodesFromEvents(abandoned).get("analyst").state).toBe("idle");
  expect(workingNodeNames([composed("analyst", "idle")], abandoned)).toEqual([]);
  // Not a candidate: a later failure elsewhere still has one story.
  expect(applyDerived([composed("analyst", "idle")], nodesFromEvents(abandoned), "failed")[0].state).toBe(
    "idle",
  );
});

test("a stage answering a clarification can still be stopped for its own turn", () => {
  // One stage running two invocations: its own turn, controlled here, and a clarification it is
  // answering for another node, controlled there. Reading only the newest live attempt took the
  // Stop away from the ordinary turn for as long as the answerer ran — a control that existed,
  // was reachable, and was not offered.
  const stages = registry(["analyst"], ["implementer"]);
  const answering = {
    at: "t",
    kind: "node_start",
    node: "analyst",
    detail: "node started",
    span_id: "00000000000000b2",
    parent_span_id: "00000000000000a1",
    controlled_as: "implementer",
    facts: { model: "opus", tools: [], thinking: false, reuses_session: false },
  };
  // The answerer starts SECOND, so it is the one a display would choose.
  const both = inNodeBoxes(
    [attemptStart("analyst", "00000000000000c3", "opus"), answering],
    stages,
  );

  expect(nodesFromEvents(both).get("analyst").state).toBe("working");
  expect(workingNodeNames([composed("analyst", "idle")], both)).toEqual(["analyst"]);

  // With only the answerer live, there is nothing here to reach.
  const alone = inNodeBoxes([answering], stages);
  expect(workingNodeNames([composed("analyst", "idle")], alone)).toEqual([]);
});

test("a turn that writes no checkpoint reports all of what it cost", () => {
  // An answerer and an evidence-only stage never checkpoint, so their `usage` record is the only
  // account of them there is. A stream carrying a cost is authoritative — whatever this leaves out
  // is displayed as measured absence, not filled in from the store — so a three-turn answer with a
  // tool call must not read as one cycle on no model.
  const stages = registry(["analyst"]);
  const span = "00000000000000b2";
  const spent = {
    at: "t",
    kind: "usage",
    node: "analyst",
    detail: "node usage",
    span_id: span,
    turns: 3,
    facts: { model: "opus", tools: ["Read", "Grep"], thinking: true, reuses_session: false },
    usage: {
      input_tokens: 90,
      output_tokens: 12,
      cached_input_tokens: 0,
      cache_creation_input_tokens: 0,
      reasoning_tokens: 0,
      duration_ms: 4200,
    },
  };
  const box = nodesFromEvents(inNodeBoxes([spent], stages)).get("analyst");

  expect(box.telemetry.turns).toBe(3);
  expect(box.telemetry.model).toBe("opus");
  expect(box.telemetry.tools).toEqual(["Read", "Grep"]);
  expect(box.telemetry.thinking).toBe(true);
  expect(box.telemetry.input_tokens).toBe(90);
  expect(box.telemetry.duration_ms).toBe(4200);
});

test("a box with a finished member and a live answerer offers no control", () => {
  // A composed box: one member done, the other answering someone else's clarification. Reading
  // reachability per member let the FINISHED one answer for the box, so a Stop was offered against
  // work controlled by the node that asked — a button nothing polls.
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"], ["implementer"]);
  const done = [
    attemptStart("redteam_classifier", "00000000000000c1", "opus"),
    attemptCheckpoint("redteam_classifier", "00000000000000c1", 5),
  ];
  const answering = {
    at: "t",
    kind: "node_start",
    node: "redteam_author",
    detail: "node started",
    span_id: "00000000000000d2",
    parent_span_id: "00000000000000a1",
    controlled_as: "implementer",
    facts: { model: "opus", tools: [], thinking: false, reuses_session: false },
  };
  const boxed = inNodeBoxes([...done, answering], stages);

  expect(nodesFromEvents(boxed).get("redteam").state).toBe("working");
  expect(workingNodeNames([composed("redteam", "idle")], boxed)).toEqual([]);
});

test("a turn whose start has scrolled out of view keeps its controls and its address", () => {
  // A viewer attaching to a long tool-heavy turn gets a tail with no `node_start` in it. The fold
  // reconstructs the turn from the records that remain — and requiring a start anywhere in view
  // threw that away for the store's answer, which still says idle until the node checkpoints. The
  // controls disappeared for exactly as long as the turn was interesting.
  const stages = registry(["analyst"], ["implementer"]);
  const server = [{ ...composed("analyst", "idle"), shaped: true }];
  const tail = inNodeBoxes(
    [
      {
        at: "t",
        kind: "tool_call",
        node: "analyst",
        detail: "Read",
        span_id: "00000000000000c3",
      },
    ],
    stages,
  );
  expect(nodesFromEvents(tail).get("analyst").state).toBe("working");
  expect(workingNodeNames(server, tail)).toEqual(["analyst"]);

  // And the address survives the same trimming, because the turn's span carries it: an answerer
  // whose start is out of view must not read as controllable.
  const answering = inNodeBoxes(
    [
      {
        at: "t",
        kind: "tool_call",
        node: "analyst",
        detail: "Read",
        span_id: "00000000000000d4",
        controlled_as: "implementer",
      },
    ],
    stages,
  );
  expect(nodesFromEvents(answering).get("analyst").state).toBe("working");
  expect(workingNodeNames(server, answering)).toEqual([]);
});

test("a usage record carries what a turn reached for and whether it failed", () => {
  // The only terminal record a turn without a checkpoint leaves. Its tools were dropped on the way
  // in, and its error was never emitted at all — so a recovered failure read as done the moment its
  // execution ended.
  const stages = registry(["analyst"]);
  const span = "00000000000000e5";
  const failed = {
    at: "t",
    kind: "usage",
    node: "analyst",
    detail: "node usage",
    span_id: span,
    turns: 2,
    error: "could not answer",
    tools_used: ["Read", "Grep"],
    usage: {
      input_tokens: 10,
      output_tokens: 2,
      cached_input_tokens: 0,
      cache_creation_input_tokens: 0,
      reasoning_tokens: 0,
      duration_ms: 30,
    },
  };
  const ended = inNodeBoxes(
    [
      attemptStart("analyst", span, "opus"),
      failed,
      { at: "t", kind: "span_end", outcome: "completed", span_id: span, execution: "node", execution_name: "analyst" },
    ],
    stages,
  );
  const box = nodesFromEvents(ended).get("analyst");

  expect([...box.telemetry.tools_used]).toEqual(["Read", "Grep"]);
  expect(box.state).toBe("failed");
});

for (const outcome of ["cancelled", "unvalidated"]) {
test(`an execution that ended ${outcome} is not reported as having finished`, () => {
  // `span_end` is emitted however an execution leaves — returned, errored, or dropped when the run
  // was stopped. Reading any end as success rendered a cancelled node as done, and a failed run is
  // attributed among the nodes still reading as working: the one that died was excluded from its
  // own reconciliation.
  const shape = [composed("characterizer", "idle")];
  const stages = registry(["characterizer"]);
  const span = "00000000000000f7";
  const cancelled = inNodeBoxes(
    [
      attemptStart("characterizer", span, "opus"),
      {
        at: "t",
        kind: "span_end",
        outcome,
        span_id: span,
        execution: "node",
        execution_name: "characterizer",
      },
    ],
    stages,
  );

  // Not done: nothing recorded how it went, and a viewer told "done" is told something nobody knows.
  expect(nodesFromEvents(cancelled).get("characterizer").state).toBe("working");
  // Which is what keeps it a candidate when the run is reconciled as failed.
  expect(applyDerived(shape, nodesFromEvents(cancelled), "failed")[0].state).toBe("failed");
});
}

test("an execution that ended keeps its failure state but is no longer a control target", () => {
  // A cancelled end deliberately leaves the state working — that is what keeps the node among a
  // failed run's candidates — but the execution is known to be OVER, and a Stop offered against it
  // reaches nothing. The two questions come apart here: still working as far as blame goes, not
  // reachable as far as controls go.
  const shape = [composed("characterizer", "idle")];
  const stages = registry(["characterizer"]);
  const span = "00000000000000fa";
  const cancelled = inNodeBoxes(
    [
      attemptStart("characterizer", span, "opus"),
      {
        at: "t",
        kind: "span_end",
        outcome: "cancelled",
        span_id: span,
        execution: "node",
        execution_name: "characterizer",
      },
    ],
    stages,
  );

  expect(nodesFromEvents(cancelled).get("characterizer").state).toBe("working");
  expect(workingNodeNames(shape, cancelled)).toEqual([]);
});

test("the hand-off is drawn from redTeam completing and implement then starting", () => {
  // Two non-idle boxes prove the hand-off only for the Rust operations, and only in one lifecycle:
  // `implement_host` waits for a red team that was CALLED first, rejects one still in flight, and
  // permits implementation where none was called. Starts alone cannot tell those apart — a host
  // announces itself before its body runs, so a failing `redTeam()` a workflow catches, and the
  // `implement()` the guard then rejects, both leave starts behind. The evidence is redTeam's
  // completed END before implement's start.
  const shape = [node("redteam", "done"), node("implementer", "working")];
  const stages = registry(["redteam"], ["implementer"]);
  const lifecycle = (...records) =>
    handoffEvidence(
      inNodeBoxes(
        records.map(([kind, name, span, outcome], i) => ({
          at: `t${i}`,
          kind,
          span_id: span,
          execution: "host",
          execution_name: name,
          ...(outcome ? { outcome } : {}),
        })),
        stages,
      ),
    );
  const H1 = "00000000000000a1";
  const H2 = "00000000000000b2";
  const driven = {
    at: "t9",
    kind: "node_start",
    node: "implementer",
    detail: "node started",
    span_id: "00000000000000c9",
    parent_span_id: H2,
  };

  // The standard flow: redTeam completes, implement starts, and DRIVES the implementer's work.
  const standard = handoffEvidence(
    inNodeBoxes(
      [
        { at: "t0", kind: "span_start", span_id: H1, execution: "host", execution_name: "redTeam" },
        { at: "t1", kind: "span_end", outcome: "completed", span_id: H1, execution: "host", execution_name: "redTeam" },
        { at: "t2", kind: "span_start", span_id: H2, execution: "host", execution_name: "implement" },
        driven,
      ],
      stages,
    ),
  );
  expect(standard).toBe(true);
  expect(forkHandoff(shape, standard)).toBe(true);

  // The implement call alone is not the hand-off: it can fail in its own preparation — the guard,
  // argument parsing, the worktree — after announcing itself and before driving anything. Until
  // work starts under it, nothing is in the implementer's hands.
  const stillborn = lifecycle(
    ["span_start", "redTeam", H1],
    ["span_end", "redTeam", H1, "completed"],
    ["span_start", "implement", H2],
    ["span_end", "implement", H2, "unvalidated"],
  );
  expect(stillborn).toBe(false);
  expect(forkHandoff(shape, stillborn)).toBe(false);

  // A failing redTeam the workflow caught, then an implement the guard rejects: both STARTED, and
  // nothing was handed to anyone. Custom stages may then populate both boxes.
  const caught = lifecycle(
    ["span_start", "redTeam", H1],
    ["span_end", "redTeam", H1, "unvalidated"],
    ["span_start", "implement", H2],
  );
  expect(caught).toBe(false);
  expect(forkHandoff(shape, caught)).toBe(false);

  // implement before redTeam ever completed, and implement without any redTeam.
  expect(
    lifecycle(["span_start", "implement", H2], ["span_start", "redTeam", H1]),
  ).toBe(false);
  expect(lifecycle(["span_start", "my_probe", H1], ["span_start", "implement", H2])).toBe(false);

  // A stream that announces no hosts is from before executions did: it cannot say, and the box
  // fallback — what every stream got before there was evidence — draws the edge.
  expect(handoffEvidence(inNodeBoxes([], stages))).toBe(null);
  expect(forkHandoff(shape, null)).toBe(true);
});
test("a custom stage in the implementer box is not a converge loop", () => {
  // The loop being counted is the standard operation's. A workflow may compose any stage into the
  // implementer box, and its starts arrive under that box's name — counting them displayed a retry
  // no operation ever ran. What drove a start is its parent execution, and a declared stage's host
  // can never bear an operation's name.
  const stages = registry(["implementer", "implementer_attempt", "my_builder"], ["analyst"]);
  const host = (span, name) => ({
    at: "t",
    kind: "span_start",
    span_id: span,
    execution: "host",
    execution_name: name,
  });
  const startUnder = (name, span, parent) => ({
    at: "t",
    kind: "node_start",
    node: name,
    detail: "node started",
    span_id: span,
    parent_span_id: parent,
  });

  // Driven twice by a declared host, once by the operation: one entry, no re-entries counted.
  const custom = inNodeBoxes(
    [
      host("00000000000000a1", "implement"),
      startUnder("implementer_attempt", "00000000000000b1", "00000000000000a1"),
      host("00000000000000a2", "my_builder"),
      startUnder("my_builder", "00000000000000b2", "00000000000000a2"),
      startUnder("my_builder", "00000000000000b3", "00000000000000a2"),
    ],
    stages,
  );
  expect(convergeLoops(custom, stages)).toEqual({ fix: 0, replan: 0, retry: 0 });

  // The same shape driven by the operation counts, and classifies as it always has.
  const standard = inNodeBoxes(
    [
      host("00000000000000a1", "implement"),
      startUnder("implementer_attempt", "00000000000000b1", "00000000000000a1"),
      host("00000000000000a3", "iterate"),
      startUnder("implementer_attempt", "00000000000000b4", "00000000000000a3"),
    ],
    stages,
  );
  expect(convergeLoops(standard, stages)).toEqual({ fix: 0, replan: 0, retry: 1 });

  // An operation this client has never heard of still counts, because operation-ness is derived
  // from the run's own registry — a host is a declared stage or a Rust operation, nothing else —
  // rather than copied as a list of names. A copy was a second authority: a recovery operation
  // added in Rust would have read as a known non-operation, and its re-entries were discarded.
  const future = inNodeBoxes(
    [
      host("00000000000000a1", "implement"),
      startUnder("implementer_attempt", "00000000000000b1", "00000000000000a1"),
      host("00000000000000a4", "recoverBuild"),
      startUnder("implementer_attempt", "00000000000000b5", "00000000000000a4"),
    ],
    stages,
  );
  expect(convergeLoops(future, stages)).toEqual({ fix: 0, replan: 0, retry: 1 });
});

test("a reconnect gap is not a complete account", () => {
  // `onReset` clears the live buffer while the history re-read is throttled, so stale history gets
  // joined to a fresh bounded tail with the slice between them missing. An absence in that slice
  // proves nothing — a hand-off whose redTeam records fell in the gap was being suppressed as
  // though the stream had denied it.
  const ev = (at) => ({ at, kind: "model_text", node: "analyst", detail: "…" });
  const history = [ev("t1"), ev("t2"), ev("t3")];

  // The replay overlaps history: nothing fell between.
  expect(contiguous(history, [ev("t3"), ev("t4")], null)).toBe(true);
  // An empty buffer has nothing after history to miss.
  expect(contiguous(history, [], null)).toBe(true);
  // The replay begins after history ends: the slice between is missing, and the account is not
  // complete — evidence read from it would prove absences it cannot.
  expect(contiguous(history, [ev("t5")], null)).toBe(false);
  // No history at all is the bounded tail.
  expect(contiguous(null, [ev("t1")], null)).toBe(false);
  expect(contiguous([], [], null)).toBe(false);

  // The replay preserves `question*` events AHEAD of its bounded tail, however old — a run blocked
  // on a human must show its question — so a preserved question's timestamp says nothing about
  // where the tail begins. Reading it as the buffer's start claimed continuity across a missing
  // slice, and an absence in that slice suppressed a hand-off the run made.
  const question = { at: "t2", kind: "question", node: null, detail: "which way?" };
  expect(contiguous(history, [question, ev("t5")], null)).toBe(false);
  // While a tail that genuinely overlaps still proves itself past the preserved front.
  expect(contiguous(history, [question, ev("t3"), ev("t5")], null)).toBe(true);
  // A buffer of preserved questions alone has no tail to miss anything.
  expect(contiguous(history, [question], null)).toBe(true);
});

test("a view scrubbed to inside the history is complete whatever the buffer holds", () => {
  // A reconnect gap sits AFTER the history's end, so a scrub position at or before it shows
  // nothing the gap could have swallowed — the history is one read from the run's beginning.
  // Judging the whole timeline instead of the shown prefix turned that definite view into a
  // fallback, and the fallback drew a hand-off the composed stages never made.
  const ev = (at) => ({ at, kind: "model_text", node: "analyst", detail: "…" });
  const history = [ev("t1"), ev("t2"), ev("t3")];
  const gapped = [ev("t5")];

  // Scrubbed to inside (or exactly to the end of) the history: complete despite the gap after it.
  expect(contiguous(history, gapped, "t2")).toBe(true);
  expect(contiguous(history, gapped, "t3")).toBe(true);
  // Scrubbed past the history's end, the view includes the join — and the join is broken.
  expect(contiguous(history, gapped, "t5")).toBe(false);
  // The live end is the same question as a position past the history.
  expect(contiguous(history, gapped, null)).toBe(false);
  // No history means even an early position sits in a bounded tail, not a complete account.
  expect(contiguous(null, gapped, "t5")).toBe(false);
});

test("a nested node resolves its caller through the host call that drove it", () => {
  // The chain to a caller runs THROUGH spans no box owns: an answerer's turn hangs off the ask
  // host call, whose parent is the asking node's own turn. The walk steps over the host span and
  // anchors the box to the nearest span another box's execution opened.
  const stages = registry(["analyst"], ["helper"]);
  const asking = "00000000000000a1";
  const ask = "00000000000000b1";
  const events = [
    attemptStart("analyst", asking, "opus"),
    { at: "t1", kind: "span_start", span_id: ask, parent_span_id: asking, execution: "host", execution_name: "ask" },
    { ...attemptStart("helper", "00000000000000c1", "haiku"), parent_span_id: ask },
  ];
  const boxes = nodesFromEvents(inNodeBoxes(events, stages));
  expect(boxes.get("helper").caller).toBe("analyst");
  // The asking node itself is driven by nothing the stream shows.
  expect(boxes.get("analyst").caller).toBeUndefined();
});

test("a node driven by the run itself resolves no caller", () => {
  // A run-driven invocation that STATES nothing: its parent is an operation host the run drove,
  // and above that host there is no box. The stream honestly has no anchor, and nothing here
  // invents one.
  const stages = registry(["referee"]);
  const host = "00000000000000b2";
  const events = [
    { at: "t0", kind: "span_start", span_id: host, parent_span_id: "00000000000000ff", execution: "host", execution_name: "iterate" },
    { ...attemptStart("referee", "00000000000000c2", "opus"), parent_span_id: host },
  ];
  expect(nodesFromEvents(inNodeBoxes(events, stages)).get("referee").caller).toBeUndefined();
});

test("a stated caller anchors a run-driven invocation from its first record", () => {
  // The referee: invoked by the converge host, which no box owns, yet judging exactly the
  // implementer's latest checkpoint — so the producer STATES the caller on the node_start, and
  // the box anchors from the first moment of the turn. Waiting for the checkpoint's mirrored
  // caller left the flagship case in a trailing column for its entire model turn — the one
  // stretch of time the placement exists to show.
  const stages = registry(["implementer"], ["referee"]);
  const host = "00000000000000b9";
  const live = [
    attemptStart("implementer", "00000000000000a9", "opus"),
    { at: "t0", kind: "span_start", span_id: host, parent_span_id: "00000000000000ff", execution: "host", execution_name: "iterate" },
    { ...attemptStart("referee", "00000000000000c9", "opus"), parent_span_id: host, caller: "implementer" },
  ];
  const derived = nodesFromEvents(inNodeBoxes(live, stages));
  expect(derived.get("referee").caller).toBe("implementer");

  // And it reaches the row before any server row exists: mid-turn the shape has no referee, and
  // the anchor must not wait for the checkpoint that ends the turn.
  const view = applyDerived([composed("implementer", "working")], derived, null, true);
  expect(view.find((n) => n.name === "referee").caller).toBe("implementer");

  // A statement is a resolution like any other: two invocations stating different callers fit
  // two histories, and anchor the box to neither.
  const conflicted = [
    ...live,
    { ...attemptStart("referee", "00000000000000d9", "opus"), parent_span_id: host, caller: "analyst" },
  ];
  expect(nodesFromEvents(inNodeBoxes(conflicted, stages)).get("referee").caller).toBeNull();
});

test("two invocations that resolve different callers anchor the box to neither", () => {
  // One box holds one place in the graph. An anchor that fits two histories asserts neither, so
  // a node invoked from two different boxes keeps its trailing column. The refusal is `null`,
  // not absence: it is evidence, and a durable record must not re-anchor what it contradicts.
  const stages = registry(["analyst"], ["scout"], ["helper"]);
  const events = [
    attemptStart("analyst", "00000000000000a3", "opus"),
    attemptStart("scout", "00000000000000b3", "opus"),
    { ...attemptStart("helper", "00000000000000c3", "haiku"), parent_span_id: "00000000000000a3" },
    { ...attemptStart("helper", "00000000000000d3", "haiku"), parent_span_id: "00000000000000b3" },
  ];
  expect(nodesFromEvents(inNodeBoxes(events, stages)).get("helper").caller).toBeNull();

  // And the refusal holds through `applyDerived`: a persisted caller — an imported run's referee
  // row, say — must not re-anchor a box whose complete history refused the anchor. Silence lets
  // the server's answer stand; a refusal does not.
  const shape = [
    composed("analyst", "done"),
    { name: "helper", state: "done", checkpoints: 2, stage: 5, lane: 0, shaped: false, caller: "scout" },
  ];
  const view = applyDerived(shape, nodesFromEvents(inNodeBoxes(events, stages)), null, true);
  expect(view.find((n) => n.name === "helper").caller).toBeUndefined();
});

test("an answerer resolves the asker through the exchange, not the exchange's row label", () => {
  // `NodeClarifier::answer` opens an exchange execution inside the asking node's turn, runs the
  // answerer as its child, and writes the exchange's row — node: "clarification" — on the
  // EXCHANGE span. A row's label names what was produced, not the execution that opened the span
  // it rides on: reading it as ownership resolved the answerer's caller to a "clarification"
  // box — a two-level dynamic chain the anchor rules rightly refuse — leaving the advertised
  // answerer case in the trailing column this resolution exists to remove.
  const stages = registry(["analyst"], ["helper"], ["clarification"]);
  const asker = attemptStart("analyst", "00000000000000a8", "opus");
  const exchange = {
    at: "t1",
    kind: "span_start",
    span_id: "00000000000000b8",
    parent_span_id: "00000000000000a8",
    execution: "clarification",
    execution_name: "clarify",
  };
  const answerer = { ...attemptStart("helper", "00000000000000c8", "haiku"), parent_span_id: "00000000000000b8" };
  const row = attemptCheckpoint("clarification", "00000000000000b8", 5);

  // The row may land before or after the answerer's start; neither order may claim the span.
  for (const events of [[asker, exchange, answerer, row], [asker, exchange, row, answerer]]) {
    expect(nodesFromEvents(inNodeBoxes(events, stages)).get("helper").caller).toBe("analyst");
  }
});

test("a workflow-driven invocation beside a nested one anchors the box to neither", () => {
  // One invocation nested under `analyst`, one driven by the run itself — its parent chain
  // reaches no box. The box's history fits two placements, under the caller and in its own
  // trailing column, and a root resolution is a vote exactly as a concrete caller is: silently
  // dropping it moved the box under a caller that only sometimes called it.
  const stages = registry(["analyst"], ["helper"]);
  const nested = { ...attemptStart("helper", "00000000000000b7", "haiku"), parent_span_id: "00000000000000a7" };
  const op = { at: "t0", kind: "span_start", span_id: "00000000000000c7", parent_span_id: "00000000000000ee", execution: "host", execution_name: "helperOp" };
  const driven = { ...attemptStart("helper", "00000000000000d7", "haiku"), parent_span_id: "00000000000000c7" };
  const asker = attemptStart("analyst", "00000000000000a7", "opus");

  // Either order: the conflict is about the history, not about which arrived last. And it is a
  // REFUSAL (`null`), not silence — the durable fallback must not undo it.
  for (const events of [[asker, nested, op, driven], [asker, op, driven, nested]]) {
    expect(nodesFromEvents(inNodeBoxes(events, stages)).get("helper").caller).toBeNull();
  }
});

test("a cycle in producer parentage costs a bounded walk and anchors nothing", () => {
  // Parentage is producer-supplied data. Two spans naming each other as parent must cost a capped
  // walk rather than hang the render, and prove no caller.
  const stages = registry(["helper"]);
  const events = [
    { at: "t0", kind: "span_start", span_id: "00000000000000a4", parent_span_id: "00000000000000b4", execution: "host", execution_name: "x" },
    { at: "t0", kind: "span_start", span_id: "00000000000000b4", parent_span_id: "00000000000000a4", execution: "host", execution_name: "y" },
    { ...attemptStart("helper", "00000000000000c4", "haiku"), parent_span_id: "00000000000000a4" },
  ];
  expect(nodesFromEvents(inNodeBoxes(events, stages)).get("helper").caller).toBeUndefined();
});

test("an unplaced node carries the stream's caller, and the server's stands where the stream is silent", () => {
  const stages = registry(["analyst"], ["helper"]);
  const shape = [
    composed("analyst", "done"),
    // The server placed `helper` from its checkpoint in a trailing column, with the caller its
    // own resolution produced.
    { name: "helper", state: "done", checkpoints: 1, stage: 5, lane: 0, shaped: false, caller: "scout" },
  ];
  // The stream watched THIS run's parentage, so its answer outranks the mirrored call site —
  // where the account is complete.
  const seen = [
    attemptStart("analyst", "00000000000000a5", "opus"),
    { ...attemptStart("helper", "00000000000000b5", "haiku"), parent_span_id: "00000000000000a5" },
  ];
  const view = applyDerived(shape, nodesFromEvents(inNodeBoxes(seen, stages)), null, true);
  expect(view.find((n) => n.name === "helper").caller).toBe("analyst");

  // An INCOMPLETE account resolves no stream caller: the resolution rests on every invocation of
  // the box agreeing, and a bounded tail may have dropped an earlier one from a different caller —
  // its agreement proves nothing. The server's answer is from the durable record and stands.
  const bounded = applyDerived(shape, nodesFromEvents(inNodeBoxes(seen, stages)), null, false);
  expect(bounded.find((n) => n.name === "helper").caller).toBe("scout");
  // And incomplete is what an unstated window is.
  const unstated = applyDerived(shape, nodesFromEvents(inNodeBoxes(seen, stages)));
  expect(unstated.find((n) => n.name === "helper").caller).toBe("scout");

  // No parentage in the stream: the server's answer is the only one and stands.
  const silent = [
    attemptStart("analyst", "00000000000000a6", "opus"),
    attemptStart("helper", "00000000000000b6", "haiku"),
  ];
  const kept = applyDerived(shape, nodesFromEvents(inNodeBoxes(silent, stages)), null, true);
  expect(kept.find((n) => n.name === "helper").caller).toBe("scout");
});

test("a branch hangs only off a parent that holds a column of its own", () => {
  const byName = new Map([
    ["implementer", { name: "implementer", state: "done", checkpoints: 1, stage: 3, lane: 1 }],
    ["referee", { name: "referee", state: "done", checkpoints: 1, stage: 6, lane: 0, shaped: false, caller: "implementer" }],
    ["meta", { name: "meta", state: "done", checkpoints: 1, stage: 7, lane: 0, shaped: false, caller: "referee" }],
    ["stray", { name: "stray", state: "done", checkpoints: 0, stage: 8, lane: 0, shaped: false, caller: "gone" }],
  ]);
  expect(branchParent(byName.get("referee"), byName)).toBe("implementer");
  // One level deep: a dynamic node called by another dynamic node keeps its trailing column,
  // because hanging it off a box that itself moved is an assertion the walk cannot stand behind.
  expect(branchParent(byName.get("meta"), byName)).toBeNull();
  // A caller with no box anchors nothing.
  expect(branchParent(byName.get("stray"), byName)).toBeNull();
  // A placed node never branches, whatever it carries.
  expect(branchParent(byName.get("implementer"), byName)).toBeNull();
});

test("the last transition at any prefix is the edge that just lit", () => {
  // A fold of the shown prefix, so scrubbing is free: back before a hand-off, it has not happened.
  const stages = registry(["analyst"], ["implementer"]);
  const events = inNodeBoxes(
    [start("analyst"), checkpointed("analyst"), start("implementer")],
    stages,
  );

  expect(transitions(events.slice(0, 0))).toEqual([]);
  expect(transitions(events.slice(0, 1)).at(-1)).toEqual({
    from: null,
    to: "analyst",
    at: events[0].at,
  });
  // The checkpoint alone changes nothing — a box ending is not a traversal.
  expect(transitions(events.slice(0, 2)).at(-1).to).toBe("analyst");
  expect(transitions(events).at(-1)).toEqual({
    from: "analyst",
    to: "implementer",
    at: events[2].at,
  });
});

test("the converge self-loop reads as implementer to implementer", () => {
  const stages = registry(["implementer"]);
  const events = inNodeBoxes(
    [start("implementer"), checkpointed("implementer"), start("implementer")],
    stages,
  );
  expect(transitions(events).at(-1)).toEqual({
    from: "implementer",
    to: "implementer",
    at: events[2].at,
  });
});

test("member churn inside a box is not a transition", () => {
  // The classifier finishing while the author starts is the box working, not the graph moving —
  // a member's checkpoint must not end the box, or the author's start invents a self-loop.
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"]);
  const events = inNodeBoxes(
    [
      start("redteam_classifier"),
      checkpointed("redteam_classifier"),
      start("redteam_author"),
    ],
    stages,
  );
  const seen = transitions(events);
  expect(seen).toHaveLength(1);
  expect(seen[0].to).toBe("redteam");
});

test("a transition's from is provenance before adjacency", () => {
  const stages = registry(["implementer"], ["verifier"], ["referee"], ["analyst"], ["helper"]);

  // Stated: the referee starts while the verifier was the last box settled, and the statement
  // names the implementer — the edge that lights is the caller drop, not verifier adjacency.
  const host = "00000000000000e1";
  const stated = inNodeBoxes(
    [
      start("verifier"),
      checkpointed("verifier"),
      { at: "t0", kind: "span_start", span_id: host, parent_span_id: "00000000000000ff", execution: "host", execution_name: "iterate" },
      { ...attemptStart("referee", "00000000000000e2", "opus"), parent_span_id: host, caller: "implementer" },
    ],
    stages,
  );
  expect(transitions(stated).at(-1)).toMatchObject({ from: "implementer", to: "referee" });

  // Walked: an answerer resolves the asker through the exchange, whatever settled before it.
  const asking = "00000000000000e3";
  const exchange = "00000000000000e4";
  const walked = inNodeBoxes(
    [
      start("verifier"),
      checkpointed("verifier"),
      attemptStart("analyst", asking, "opus"),
      { at: "t1", kind: "span_start", span_id: exchange, parent_span_id: asking, execution: "clarification", execution_name: "clarify" },
      { ...attemptStart("helper", "00000000000000e5", "haiku"), parent_span_id: exchange },
    ],
    stages,
  );
  expect(transitions(walked).at(-1)).toMatchObject({ from: "analyst", to: "helper" });
});

test("a box's own record does not end it while a peer is still live", () => {
  // A box that is itself a stage can write its own row while a composed peer still runs — the
  // exact case nodesFromEvents keeps working. Clearing the active box there let the peer's next
  // start invent a review -> review self-loop the run never made.
  const stages = registry(["review", "review", "security"]);
  const events = inNodeBoxes(
    [
      attemptStart("review", "00000000000000f1", "opus"),
      attemptStart("security", "00000000000000f2", "haiku"),
      attemptCheckpoint("review", "00000000000000f1", 5),
      attemptStart("security", "00000000000000f3", "haiku"),
    ],
    stages,
  );
  const seen = transitions(events);
  expect(seen).toHaveLength(1);
  expect(seen[0].to).toBe("review");

  // Once the peer drains too, the box is over, and a re-entry is a real self-loop again.
  const drained = inNodeBoxes(
    [
      attemptStart("review", "00000000000000f4", "opus"),
      attemptStart("security", "00000000000000f5", "haiku"),
      attemptCheckpoint("review", "00000000000000f4", 5),
      attemptCheckpoint("security", "00000000000000f5", 5),
      attemptStart("review", "00000000000000f6", "opus"),
    ],
    stages,
  );
  expect(transitions(drained).at(-1)).toMatchObject({ from: "review", to: "review" });
});

test("a concurrent second invocation is not a new activation", () => {
  // Workflows may run hosts concurrently: A starts, B starts, and A's SECOND invocation lands
  // while its first is still live. Comparing against a single last-started box read that as a
  // fresh activation and pulsed a B -> A hand-off that never happened — execution never left A,
  // so the pulse belongs on the transition into B.
  const stages = registry(["alpha"], ["beta"]);
  const events = inNodeBoxes(
    [
      attemptStart("alpha", "00000000000000f7", "opus"),
      attemptStart("beta", "00000000000000f8", "opus"),
      attemptStart("alpha", "00000000000000f9", "opus"),
    ],
    stages,
  );
  const seen = transitions(events);
  expect(seen).toHaveLength(2);
  expect(seen.at(-1)).toMatchObject({ from: "alpha", to: "beta" });
});

test("a peer starting mid-cycle does not erase the box's completion", () => {
  // The box's own checkpoint lands while one peer is live, and ANOTHER peer starts before the
  // first drains. Resetting the latch on every start erased that completion, the cycle could
  // then never close, and the next genuine re-entry was swallowed instead of drawing its
  // self-loop.
  const stages = registry(["review", "review", "security"]);
  const events = inNodeBoxes(
    [
      attemptStart("review", "0000000000000101", "opus"),
      attemptStart("security", "0000000000000102", "haiku"),
      attemptCheckpoint("review", "0000000000000101", 5),
      attemptStart("security", "0000000000000103", "haiku"),
      attemptCheckpoint("security", "0000000000000102", 5),
      attemptCheckpoint("security", "0000000000000103", 5),
      attemptStart("review", "0000000000000104", "opus"),
    ],
    stages,
  );
  expect(transitions(events).at(-1)).toMatchObject({ from: "review", to: "review" });
});

test("a lifecycle end closes a checkpoint-free cycle", () => {
  // An evidence-only stage or an answerer legitimately writes no checkpoint; its own execution's
  // end is its completion. Without reading it, the cycle never closed and every later hand-off
  // into that box was swallowed as though the first invocation still ran.
  const stages = registry(["characterizer"]);
  const events = inNodeBoxes(
    [
      attemptStart("characterizer", "0000000000000105", "opus"),
      { at: "t1", kind: "span_end", outcome: "completed", span_id: "0000000000000105", execution: "node", execution_name: "characterizer" },
      attemptStart("characterizer", "0000000000000106", "opus"),
    ],
    stages,
  );
  const seen = transitions(events);
  expect(seen).toHaveLength(2);
  expect(seen.at(-1)).toMatchObject({ from: "characterizer", to: "characterizer" });
});

test("a turn ending is not the box finishing — the boundary is", () => {
  // An ordinary stage's execution ends BEFORE its host validates and writes the checkpoint —
  // exactly where the node fold refuses to settle the turn. Completing the box on the raw end
  // closed its cycle in that window, and a composed peer starting there read as a fresh
  // transition the run never made.
  const stages = registry(["review", "review", "security"]);
  const host = "0000000000000110";
  const events = inNodeBoxes(
    [
      { at: "t0", kind: "span_start", span_id: host, parent_span_id: "00000000000000ff", execution: "host", execution_name: "review_host" },
      { ...attemptStart("review", "0000000000000111", "opus"), parent_span_id: host },
      // The turn returns; validation and the checkpoint are still to come.
      { at: "t1", kind: "span_end", outcome: "completed", span_id: "0000000000000111", execution: "node", execution_name: "review" },
      { ...attemptStart("security", "0000000000000112", "haiku"), parent_span_id: host },
    ],
    stages,
  );
  const seen = transitions(events);
  expect(seen).toHaveLength(1);
  expect(seen[0].to).toBe("review");
});

test("a checkpoint-free box completes at its boundary, in either order", () => {
  // The answerer/evidence-only shape: no checkpoint ever, so the own turn's completed end plus
  // the completed end of the execution that invoked it are what finish the box — whichever
  // arrives second. Only then does a re-entry read as the self-loop.
  const stages = registry(["characterizer"]);
  const hostOf = (span) => `00000000000001${span}`;
  const turn = (span) => [
    { at: "t0", kind: "span_start", span_id: hostOf(span), parent_span_id: "00000000000000ff", execution: "host", execution_name: "characterize" },
    { ...attemptStart("characterizer", `00000000000000${span}`, "opus"), parent_span_id: hostOf(span) },
  ];
  const childEnd = (span, at) => ({ at, kind: "span_end", outcome: "completed", span_id: `00000000000000${span}`, execution: "node", execution_name: "characterizer" });
  const hostEnd = (span, at) => ({ at, kind: "span_end", outcome: "completed", span_id: hostOf(span), execution: "host", execution_name: "characterize" });

  for (const first of [
    // Turn end first, boundary second — the usual order.
    [...turn("21"), childEnd("21", "t1"), hostEnd("21", "t2")],
    // Boundary's end seen first — an imported tail may interleave them.
    [...turn("22"), hostEnd("22", "t1"), childEnd("22", "t2")],
  ]) {
    const events = inNodeBoxes(
      [...first, attemptStart("characterizer", "0000000000000123", "opus")],
      stages,
    );
    const seen = transitions(events);
    expect(seen).toHaveLength(2);
    expect(seen.at(-1).to).toBe("characterizer");
  }
});

test("a box reports what each member stage is doing, correct at any scrub position", () => {
  // The red team mid-run: the classifier finished, the author is writing tests. The box shows one
  // state; which member is where is this map, from the same fold, so it tracks the scrubber.
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"]);
  const events = inNodeBoxes(
    [
      start("redteam_classifier"),
      checkpointed("redteam_classifier"),
      start("redteam_author"),
    ],
    stages,
  );

  const mid = nodesFromEvents(events.slice(0, 2)).get("redteam").memberStates;
  expect(mid.get("redteam_classifier")).toBe("done");
  expect(mid.has("redteam_author")).toBe(false);

  const later = nodesFromEvents(events).get("redteam").memberStates;
  expect(later.get("redteam_classifier")).toBe("done");
  expect(later.get("redteam_author")).toBe("working");
});

test("the box's own aggregate is not a member state", () => {
  // The aggregate row is the box, not a pip of itself — and a single-stage node has no member
  // states at all, which is what keeps it looking as it does today.
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"], ["analyst"]);
  const events = inNodeBoxes(
    [
      start("redteam_classifier"),
      checkpointed("redteam_classifier"),
      checkpointed("redteam"),
      start("analyst"),
      checkpointed("analyst"),
    ],
    stages,
  );
  const boxes = nodesFromEvents(events);
  expect(boxes.get("redteam").memberStates.has("redteam")).toBe(false);
  // The self-staged analyst's one member IS itself: its map is present — the stream spoke for
  // the box — and empty, which is what tells a pip-free box from a stream-silent one.
  expect(boxes.get("analyst").memberStates.size).toBe(0);
});

test("a peer waits as idle while only the self stage has run", () => {
  // review composed of its self-named stage plus security, scrubbed to where only review has
  // spoken. The box's derivation IS the stream having reached it — omitting the empty map there
  // read as silence, and the strip vanished with security's waiting pip in it.
  const stages = registry(["review", "review", "security"]);
  const events = inNodeBoxes([start("review")], stages);
  const box = nodesFromEvents(events).get("review");
  expect(box.memberStates).toBeDefined();
  expect(box.memberStates.size).toBe(0);
});

test("a member that failed stays failed in the strip, without failing the box", () => {
  // A checkpoint's error is the event contract's failure signal, and the strip reads member
  // states directly — flattening a member's error to done drew a failed substage as completed.
  // The BOX still fails only on its own record, exactly as before: the classifier erring is a
  // fact about the classifier, and the box works on while the author runs.
  const stages = registry(["redteam", "redteam_classifier", "redteam_author"]);
  const events = inNodeBoxes(
    [
      start("redteam_classifier"),
      { ...checkpointed("redteam_classifier"), error: "no baseline" },
      start("redteam_author"),
    ],
    stages,
  );
  const box = nodesFromEvents(events).get("redteam");
  expect(box.memberStates.get("redteam_classifier")).toBe("failed");
  expect(box.memberStates.get("redteam_author")).toBe("working");
  expect(box.state).toBe("working");
});
