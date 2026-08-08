import { useState } from "react";
import { control } from "../api";
import { Compose } from "./Compose";

/**
 * Say something to a node that did not ask.
 *
 * The other half of the clarification box, and deliberately the same box: a node asking a question
 * and an operator volunteering a correction are one conversation, and splitting them into two
 * differently-shaped inputs would make the page harder to read than the feature is worth.
 *
 * It reaches the model on that node's next tool result, so it is a nudge rather than an interrupt —
 * a node deep in a long turn hears it when the turn ends, not now.
 */
export function Steer({
  project,
  runId,
  working,
  onSent,
  onDismiss,
}: {
  project: string;
  runId: string;
  /** Nodes working right now. More than one means the fork is running and it has to be said which. */
  working: string[];
  onSent: () => void;
  /** Close without sending — a click elsewhere, while nothing has been typed. */
  onDismiss: () => void;
}) {
  const [node, setNode] = useState(working[0] ?? "");
  // A node that finishes while the box is open leaves the choice stale; falling back to whatever is
  // working now beats sending to a node that has stopped listening.
  const to = working.includes(node) ? node : (working[0] ?? "");

  return (
    <Compose
      heading={
        <>
          ///{" "}
          {working.length > 1 ? (
            <select
              className="ask-to"
              value={to}
              onChange={(e) => setNode(e.target.value)}
              aria-label="Which node to message"
            >
              {working.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          ) : (
            to
          )}{" "}
          is listening
        </>
      }
      onDismiss={onDismiss}
      placeholder="what should it know…"
      submit=">>> SEND"
      onSubmit={async (text) => {
        await control(project, runId, { command: "steer", node: to, text });
        onSent();
      }}
    />
  );
}
