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
 * Stop names a node, because a run's fork has two working at once and "stop the run" would be
 * ambiguous exactly when it matters. With one node working there is nothing to choose and the
 * click acts directly; with several, it asks which. Steering opens the same box a node's question
 * opens, down in the activity area — see `Steer`.
 */
export function Controls({
  project,
  runId,
  state,
  working,
  mayAct,
  onChange,
  onCompose,
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
  /** Open the message box. Not a popover of its own — it is the clarification box. */
  onCompose: () => void;
}) {
  const [picking, setPicking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);

  // Close on anything that means "I am doing something else now". Escape and an outside click are
  // the two ways out of a popover people already know.
  useEffect(() => {
    if (!picking) return;
    const away = (e: PointerEvent) => {
      if (!box.current?.contains(e.target as Node)) setPicking(false);
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPicking(false);
    };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", key);
    };
  }, [picking]);

  async function send(command: Command) {
    setBusy(true);
    setError(null);
    try {
      onChange(await control(project, runId, command));
      setPicking(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const stopped = state.stopped;
  /** What the stop button acts on, when there is no choice to make. */
  const target = stopped.length ? stopped : working;
  const only = target.length === 1 ? target[0]! : null;

  return (
    <div className="controls" ref={box}>
      <button
        type="button"
        className={state.paused ? "control is-on" : "control"}
        disabled={!mayAct || busy}
        onClick={() => void send({ command: state.paused ? "resume" : "pause" })}
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
        disabled={!mayAct || busy || !target.length}
        onClick={() =>
          only
            ? void send({ command: stopped.length ? "start" : "stop", node: only })
            : setPicking(true)
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
        onClick={onCompose}
        data-compose-toggle=""
        data-tip={
          working.length
            ? "Say something to a working node. It arrives on its next turn"
            : "Nothing is working, so there is nobody to talk to"
        }
        aria-label="Send a message to a node"
      >
        {BUBBLE}
      </button>

      {picking && (
        <div className="control-menu">
          <p className="control-ask">{stopped.length ? "Start which node?" : "Stop which node?"}</p>
          {target.map((node) => (
            <button
              key={node}
              type="button"
              className="control-pick"
              disabled={busy}
              onClick={() => void send({ command: stopped.length ? "start" : "stop", node })}
            >
              {node}
            </button>
          ))}
          {error && <p className="control-error">{error}</p>}
        </div>
      )}
    </div>
  );
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
