import { expect, test } from "bun:test";

import { convergeLoops, workingNodeNames } from "./derive";

function node(name, state) {
  return { name, state, checkpoints: 0, stage: 0, lane: 0 };
}

/** A `node_start`, which is the only kind the loop classifier reads. */
function start(name) {
  return { at: "2026-08-12T10:00:00Z", kind: "node_start", node: name, detail: "node started" };
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
