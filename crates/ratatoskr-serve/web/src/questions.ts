import type { LiveEvent } from "./api";

/** Events emitted by the launcher's confirmed child-exit path. */
const RUN_COMPLETION_KINDS = new Set(["run_finished", "run_failed"]);

/**
 * Questions remain open until their own answer arrives. A provider pause is a run-level event,
 * but it does not resolve a clarification that another concurrent node is still waiting on.
 */
export function pendingQuestions(
  events: readonly LiveEvent[],
  answered: ReadonlySet<string>,
): LiveEvent[] {
  const open = new Map<string, LiveEvent>();
  for (const event of events) {
    if (!event.question_id) {
      if (RUN_COMPLETION_KINDS.has(event.kind)) open.clear();
      continue;
    }
    if (event.kind === "question") open.set(event.question_id, event);
    if (event.kind === "question_answered") open.delete(event.question_id);
  }
  // Answered in this tab: clear immediately rather than waiting for the run's event to come back
  // round through the log.
  for (const id of answered) open.delete(id);
  return [...open.values()];
}
