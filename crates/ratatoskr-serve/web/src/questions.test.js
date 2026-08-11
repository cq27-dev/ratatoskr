import { expect, test } from "bun:test";

import { pendingQuestions } from "./questions";

test("a provider pause does not dismiss another node's clarification", () => {
  const pending = pendingQuestions(
    [
      { kind: "question", question_id: "q-1", node: "analyst", detail: "Which API?" },
      { kind: "run_paused", node: "implementer", detail: "provider overloaded" },
    ],
    new Set(),
  );

  expect(pending.map((event) => event.question_id)).toEqual(["q-1"]);
});

test("a confirmed child exit dismisses outstanding clarifications", () => {
  const pending = pendingQuestions(
    [
      { kind: "question", question_id: "q-1", node: "analyst", detail: "Which API?" },
      { kind: "run_finished", node: null, detail: "run finished" },
    ],
    new Set(),
  );

  expect(pending).toEqual([]);
});
