import { useEffect, useRef, useState } from "react";
import { control, type Command, type ControlView } from "../api";

/**
 * Pause, stop and steer a run in flight.
 *
 * All three act at a node's next turn boundary, which can be a minute away on a node that thinks
 * before every tool call. So the buttons show **what was asked for**, not what the run has done
 * about it: a control that sprang back until the run noticed would read as a lost click. The state
 * comes from the server for the same reason — another tab, or a reload, must see the same pause.
 *
 * Stop and steer name a node, because a run's fork has two working at once and "stop the run"
 * would be ambiguous exactly when it matters. With one node working there is nothing to choose and
 * the click acts directly; with several, it asks which.
 */
export function Controls({
  project,
  runId,
  state,
  working,
  mayAct,
  onChange,
}: {
  project: string;
  runId: string;
  state: ControlView;
  /** Nodes working right now — what stop and steer can be aimed at. */
  working: string[];
  /** Viewers see the controls disabled rather than hidden: the run *is* controllable, just not
   * by them, and a control that vanishes reads as a feature that does not exist. */
  mayAct: boolean;
  onChange: (view: ControlView) => void;
}) {
  const [menu, setMenu] = useState<"stop" | "steer" | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);

  // Close on anything that means "I am doing something else now". Escape and an outside click are
  // the two ways out of a popover people already know.
  useEffect(() => {
    if (!menu) return;
    const away = (e: PointerEvent) => {
      if (!box.current?.contains(e.target as Node)) setMenu(null);
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenu(null);
    };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", key);
    };
  }, [menu]);

  async function send(command: Command) {
    setBusy(true);
    setError(null);
    try {
      onChange(await control(project, runId, command));
      setMenu(null);
      setText("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const stopped = state.stopped;
  // One target and no choice to make: the common case, and the one where a picker would be a
  // second click for a decision with one option.
  const only = working.length === 1 ? working[0]! : null;

  function aim(kind: "stop" | "steer") {
    if (kind === "stop" && only) return void send({ command: "stop", node: only });
    setMenu(kind);
  }

  return (
    <div className="controls" ref={box}>
      <button
        type="button"
        className={state.paused ? "control is-on" : "control"}
        disabled={!mayAct || busy}
        onClick={() => send({ command: state.paused ? "resume" : "pause" })}
        data-tip={
          state.paused
            ? "Resume: every node carries on from where it held"
            : "Pause the run after the turn each node is on"
        }
        aria-label={state.paused ? "Resume the run" : "Pause the run"}
      >
        {state.paused ? PLAY : PAUSE}
      </button>

      {/* Stop and start are one button in two states, because they are one decision: this node is
          either running or waiting to be run again. */}
      <button
        type="button"
        className={stopped.length ? "control is-on" : "control"}
        disabled={!mayAct || busy || (!stopped.length && !working.length)}
        onClick={() =>
          stopped.length === 1
            ? send({ command: "start", node: stopped[0]! })
            : stopped.length
              ? setMenu("stop")
              : aim("stop")
        }
        data-tip={
          stopped.length
            ? `Start ${stopped.join(", ")} again, from its checkpoint`
            : "Stop a node. Its run waits until you start it again"
        }
        aria-label={stopped.length ? "Start the stopped node" : "Stop a node"}
      >
        {stopped.length ? PLAY : STOP}
      </button>

      <button
        type="button"
        className={state.steering.length ? "control is-on" : "control"}
        disabled={!mayAct || busy || !working.length}
        onClick={() => aim("steer")}
        data-tip={
          working.length
            ? "Say something to a working node. It arrives on its next turn"
            : "Nothing is working, so there is nobody to talk to"
        }
        aria-label="Send a message to a node"
      >
        {BUBBLE}
      </button>

      {menu && (
        <div className="control-menu">
          {menu === "stop" ? (
            <>
              <p className="control-ask">
                {stopped.length ? "Start which node?" : "Stop which node?"}
              </p>
              {(stopped.length ? stopped : working).map((node) => (
                <button
                  key={node}
                  type="button"
                  className="control-pick"
                  disabled={busy}
                  onClick={() =>
                    send({
                      command: stopped.length ? "start" : "stop",
                      node,
                    })
                  }
                >
                  {node}
                </button>
              ))}
            </>
          ) : (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                const node = only ?? pickedRef(box);
                if (!node || !text.trim()) return;
                void send({ command: "steer", node, text: text.trim() });
              }}
            >
              {/* Only when there is a choice: a select with one option is a question with one
                  answer, which is not a question. */}
              {!only && (
                <select className="control-node" name="node" defaultValue={working[0]}>
                  {working.map((node) => (
                    <option key={node} value={node}>
                      {node}
                    </option>
                  ))}
                </select>
              )}
              <textarea
                className="control-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={`Message ${only ?? "the node"}…`}
                rows={3}
                autoFocus
                // Enter sends, because this is a message rather than a document — the same as the
                // clarification box next to it. Shift+Enter still breaks a line.
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    e.currentTarget.form?.requestSubmit();
                  }
                }}
              />
              <button type="submit" className="control-send" disabled={busy || !text.trim()}>
                SEND
              </button>
            </form>
          )}
          {error && <p className="control-error">{error}</p>}
        </div>
      )}
    </div>
  );
}

/** The node chosen in the steer form's select, when there was a choice to make. */
function pickedRef(box: React.RefObject<HTMLDivElement | null>): string | null {
  const select = box.current?.querySelector<HTMLSelectElement>("select.control-node");
  return select?.value ?? null;
}

// Glyphs rather than words: three controls in the space the scrubber can spare, and every one of
// them is a shape people already read on a transport control. The speech bubble is drawn rather
// than typed — the emoji renders as a colour picture on most systems, which is the one thing this
// interface never does.
const PAUSE = "❚❚";
const PLAY = "▶";
const STOP = "■";
const BUBBLE = (
  <svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true">
    <path
      d="M1 1h14v10H5.5L2 14.5V11H1z"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinejoin="miter"
    />
  </svg>
);
