import { expect, test } from "bun:test";

import {
  applyDerived,
  convergeLoops,
  forkHandoff,
  handoffDrawn,
  inNodeBoxes,
  liveNodes,
  nodesFromEvents,
  stagesOf,
  workingNodeNames,
} from "./derive";
import { isTerminal } from "./api";

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
  expect(convergeLoops([start("analyst"), start("implementer")])).toEqual({
    fix: 0,
    replan: 0,
    retry: 0,
  });
});

test("a verifier between two implementer starts is a direct fix", () => {
  expect(convergeLoops([start("implementer"), start("verifier"), start("implementer")])).toEqual({
    fix: 1,
    replan: 0,
    retry: 0,
  });
});

// `iterate_host` runs the referee unconditionally (workflow.rs:915), including on the
// tests-not-clean path that reaches `iterate({})` without `verify()` ever running. So a referee
// start on its own is a failed-test retry, and must never be read as the verifier's fix.
test("a referee with no verifier is a retry, because the referee runs on both paths", () => {
  expect(convergeLoops([start("implementer"), start("referee"), start("implementer")])).toEqual({
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
  expect(convergeLoops(events)).toEqual({ fix: 1, replan: 0, retry: 0 });
});

test("an analyst re-run makes it a replan even though the verifier also ran", () => {
  const events = [
    start("implementer"),
    start("verifier"),
    start("analyst"),
    start("implementer"),
  ];
  expect(convergeLoops(events)).toEqual({ fix: 0, replan: 1, retry: 0 });
});

test("re-entering with no verifier and no analyst is a retry", () => {
  expect(convergeLoops([start("implementer"), start("implementer")])).toEqual({
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
  expect(convergeLoops(events)).toEqual({ fix: 1, replan: 0, retry: 1 });
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
  expect(convergeLoops(events)).toEqual({ fix: 1, replan: 1, retry: 1 });
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
  const at = (n) => convergeLoops(events.slice(0, n));

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
  expect(convergeLoops(events)).toEqual({ fix: 1, replan: 0, retry: 0 });
});

// The implementer cannot start before the red team has finished (`implement_host` in
// ratatoskr-nodes/src/workflow.rs refuses to), so both boxes having started is the whole test.
test("both the red team and the implementer having started draws the hand-off", () => {
  expect(forkHandoff([node("redteam", "done"), node("implementer", "working")])).toBe(true);
});

test("a started red team alone draws no hand-off, because nothing has received the tree", () => {
  expect(forkHandoff([node("redteam", "working"), node("implementer", "idle")])).toBe(false);
});

test("neither node having started draws no hand-off", () => {
  expect(forkHandoff([node("redteam", "idle"), node("implementer", "idle")])).toBe(false);
});

test("a workflow with no red team at all draws no hand-off from nothing", () => {
  expect(forkHandoff([node("analyst", "done"), node("implementer", "working")])).toBe(false);
});

// The edge is a vertical step down the lane gap between two boxes in one column. A layout that
// puts them in different columns already joins them the ordinary way, and this one would render
// as a diagonal across the graph on top of it.
test("a layout that puts the two in different columns draws no lane hand-off", () => {
  const nodes = [
    { ...node("redteam", "done"), stage: 0 },
    { ...node("implementer", "working"), stage: 2 },
  ];
  expect(forkHandoff(nodes)).toBe(false);
});

test("sharing a column is what draws it", () => {
  const nodes = [
    { ...node("redteam", "done"), stage: 3, lane: 0 },
    { ...node("implementer", "working"), stage: 3, lane: 1 },
  ];
  expect(forkHandoff(nodes)).toBe(true);
});

test("a failed red team still handed the tree over, so the hand-off is drawn", () => {
  expect(forkHandoff([node("redteam", "failed"), node("implementer", "working")])).toBe(true);
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
    { name: "analyst", state: "working", checkpoints: 0, stage: 0, lane: 0, shaped: false },
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
  expect(convergeLoops(events)).toEqual({ fix: 0, replan: 0, retry: 0 });
  expect(convergeLoops(inNodeBoxes(events, stages))).toEqual({ fix: 1, replan: 0, retry: 0 });
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

  const raw = liveNodes(events);
  expect([...raw.keys()]).toEqual(["context_distillation"]);
  expect(raw.get("context")).toBeUndefined();

  const boxed = liveNodes(inNodeBoxes(events, stages));
  expect([...boxed.keys()]).toEqual(["context"]);
  const box = boxed.get("context");
  expect(box.facts.model).toBe("anthropic/claude-sonnet-5");
  expect(box.cycles).toBe(1);
  expect([...box.used]).toEqual(["Read"]);
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

  // And the box's own aggregate still is not a turn: it carries the usage keys as zeros because it
  // covers nothing, and the server gives it no facts and no turns.
  const aggregate = {
    at: "t3",
    kind: "checkpoint",
    node: "redteam",
    detail: "",
    usage: { ...noCounts.usage, duration_ms: 0 },
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

test("a status this build has never heard of reads as still executing", () => {
  // The two-sided rule the Rust side keeps: an unclassified status must show a stale run rather
  // than declare a live one finished. Read the other way, an old bundle against a server that has
  // since added a non-terminal status hides a live run's controls — `applyDerived` settles every
  // working node, and `ended` is what turns that on.
  expect(isTerminal("from_a_newer_build")).toBe(false);
  expect(isTerminal(null)).toBe(false);
  for (const live of ["pending", "running", "awaiting_clarification"]) {
    expect(isTerminal(live)).toBe(false);
  }
  // And every status the server can actually persist is classified, including the two the union
  // had never been told about.
  for (const done of [
    "planned",
    "converged",
    "max_iterations_reached",
    "unreviewed",
    "no_code_change",
    "failed",
    "abandoned",
  ]) {
    expect(isTerminal(done)).toBe(true);
  }
});

test("a run's header reads a newer server's status as live, like everything else", () => {
  // `isTerminal` is deliberately open-world: anything this build cannot classify is still
  // executing. A closed set of LIVE names is the same question asked the other way round, and it
  // answers the opposite — so a status a newer server sent gave the run live controls and
  // unsettled nodes while its header said no pull request would ever appear and never marked a
  // silent run stale.
  for (const live of ["running", "pending", "awaiting_clarification", "from_a_newer_build"]) {
    expect(isTerminal(live)).toBe(false);
  }
  // And a status that IS classified terminal stays terminal on both.
  for (const done of ["converged", "failed", "no_code_change"]) {
    expect(isTerminal(done)).toBe(true);
  }
});
