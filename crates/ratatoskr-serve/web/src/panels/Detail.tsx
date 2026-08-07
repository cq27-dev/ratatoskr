import { Json } from "../ui/format";
import { clock, short } from "../ui/text";
import type { CheckpointView } from "../api";

export function Detail({
  runId,
  node,
  checkpoints,
  until,
}: {
  runId: string | null;
  node: string | null;
  checkpoints: CheckpointView[] | null;
  /** While scrubbing, the moment being looked at: later checkpoints have not happened yet. */
  until: string | null;
}) {
  if (!node) return <p className="empty">select a node to inspect its output</p>;
  if (checkpoints === null) return <p className="empty">loading {node}…</p>;

  // The store returns every checkpoint a node ever wrote, which is its state at the END of the
  // run. Showing that against a scrubbed position would put output on screen that the run had not
  // produced yet — and for the implementer, whose iterations are the interesting part, it would
  // show the final answer while the map says it is still working.
  const shown = until
    ? checkpoints.filter((c) => Date.parse(c.created_at) <= Date.parse(until))
    : checkpoints;

  if (shown.length === 0) {
    return (
      <p className="empty">
        {node} {until ? "had recorded no output by this point" : "has recorded no output"}
      </p>
    );
  }

  return (
    <div>
      <div className="sec">
        <span>
          [ {node.replace("_", " ")} ] <samp>{short(runId)}</samp>
        </span>
        <output>
          {shown.length} {shown.length === 1 ? "checkpoint" : "checkpoints"}
          {until && shown.length < checkpoints.length ? ` of ${checkpoints.length}` : ""}
        </output>
      </div>
      {shown.map((c, i) => (
        <section className="iter" key={`${c.created_at}-${i}`}>
          {/* Every checkpoint, not just the last: for the implementer these are the converge
              iterations, and the progression between them is the interesting part. */}
          {shown.length > 1 && (
            <div className="sec">
              <span>ITERATION {String(i + 1).padStart(2, "0")}</span>
              <span>{clock(c.created_at)}</span>
            </div>
          )}
          <Json value={c.output} />
        </section>
      ))}
    </div>
  );
}
