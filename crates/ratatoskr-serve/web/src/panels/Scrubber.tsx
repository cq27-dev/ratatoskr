import type { ReactNode } from "react";
import { band } from "../ui/tint";
import { since } from "../ui/text";

/**
 * Fold the raw stream into what a reader can scan.
 *
 * Three things happen here, all of them because a tool loop emits far more lines than it has
 * events worth reading: a call and its result become one row, a run of the same call collapses to
 * a count, and the argument that distinguishes one call from the next is kept.
 */
/**
 * Move through a run's timeline.
 *
 * Scrubbing is a prefix of the event stream: everything the view shows — the boxes, the
 * highlighting, the feed — is a fold over the events, so cutting the list short IS the historical
 * view. There is no replay engine and no second code path to keep in step with the live one.
 */
export function Scrubber({
  total,
  cursor,
  at,
  startedAt,
  nodes,
  onScrub,
  controls,
}: {
  total: number;
  cursor: number | null;
  at: string | null;
  /** When the run's first event landed, so a position can be shown as time into the run. */
  startedAt: string | null;
  /** The node each event belongs to, in timeline order. Colours the track. */
  nodes: (string | null)[];
  onScrub: (cursor: number | null) => void;
  /** The run's controls, if this run can be controlled at all. A slot rather than props of its
   * own: moving through a finished run and steering a live one are different jobs that happen to
   * share a row, and this one has no business knowing how the other works. */
  controls?: ReactNode;
}) {
  // Rendered even with nothing to scrub through, disabled. Returning null instead removed the
  // control's 38 pixels from the layout until the history arrived, and everything below — the
  // graph, the feed — moved down when it landed.
  const ready = total >= 2;
  const position = ready ? (cursor ?? total - 1) : 0;
  const following = cursor === null;
  return (
    <div className="scrub">
      <button
        type="button"
        className={following ? "scrub-live is-live" : "scrub-live"}
        disabled={!ready}
        onClick={() => onScrub(following ? position : null)}
        data-tip={following ? "Following the end of the run" : "Return to the end of the run"}
      >
        {/* Both words are six characters, so the button is the same size in either state and the
            slider beside it does not change length when the mode flips. */}
        {following ? "FOLLOW" : "REPLAY"}
      </button>
      {controls}
      <input
        type="range"
        min={0}
        max={Math.max(1, total - 1)}
        value={position}
        disabled={!ready}
        // The track is which node each stretch of the run belongs to, in the same hue as that
        // node's box and its name in the feed. It replaces the flat accent colour: a bar that was
        // one colour end to end said only "this is a slider", which the slider already said.
        style={{ backgroundImage: band(nodes) }}
        onChange={(e) => {
          const next = Number(e.target.value);
          // Landing on the last event means following again, so the view resumes on its own
          // rather than freezing one event short of the present.
          onScrub(next >= total - 1 ? null : next);
        }}
        aria-label="Position in the run"
      />
      <span className="scrub-at">
        {/* Padded to the total's width: unpadded, the label is narrower at 1/654 than at 654/654
            and the slider changes length as you drag it. */}
        {/* Placeholders with the same glyph count as the real values, so the readout is the same
            width before and after loading and the track beside it does not change length. A
            shorter placeholder let the slider stretch while a run loaded and snap back when the
            numbers arrived. */}
        <span className="scrub-count">
          {ready ? `${String(position + 1).padStart(String(total).length, "0")}/${total}` : "···/···"}
        </span>
        <span className="scrub-when" data-tip={at ?? undefined}>
          <span>{ready && at ? at.slice(11, 19) : "··:··:··"}</span>
          <span className="scrub-since">{ready ? since(startedAt, at) : "+··:··"}</span>
        </span>
      </span>
    </div>
  );
}
