import { useEffect, useRef, useState } from "react";
import { clock } from "../ui/text";
import { isTerminal, type RunDetail } from "../api";

/** A live run with nothing recorded for this long is almost certainly dead, not busy. */
const STALE_MS = 120_000;

export function RunMeta({ detail, lastEvent }: { detail: RunDetail; lastEvent: string | null }) {
  /*
   * When the run was last seen doing anything.
   *
   * `detail.last_activity` is the server's answer and it is a coarse one: it is the newest
   * checkpoint, or a status transition. Checkpoints are minutes apart by design — the implementer
   * runs for ten of them between two — so on its own it reports a working run as having done
   * nothing since before lunch. The event stream is the real activity: tool calls land every few
   * seconds, and silence there is silence.
   *
   * The event's OWN timestamp, not when it reached this tab. Arrival time is wrong twice: the
   * stream replays a tail on connect, so opening a run that died hours ago stamped it as active a
   * second ago — which is exactly the case the staleness flag below exists to catch.
   *
   * ISO-8601 sorts lexicographically, so the later string is the later moment.
   */
  const seen = [detail.last_activity, lastEvent].filter((t): t is string => !!t).sort();
  const lastSeen = seen.length ? seen[seen.length - 1]! : null;
  const stale =
    detail.status !== null &&
    // Still executing means NOT terminal, asked the same open-world way `isTerminal` answers it: a
    // status this build has never heard of is a run that may still be going, so it is one this can
    // still call stale. A closed list of live names answered the opposite for exactly those.
    !isTerminal(detail.status) &&
    lastSeen !== null &&
    Date.now() - Date.parse(lastSeen) > STALE_MS;

  // Collapsed by default: the header's job is to say which run this is, and the first line of an
  // issue does that. Whoever wants the rest asks for it.
  const [titleOpen, setTitleOpen] = useState(false);
  // Whether there is anything hidden to reveal. A title that fits gets no control at all — a
  // pointer cursor and a focus stop on something that cannot change is a promise the interface
  // does not keep.
  const titleBox = useRef<HTMLHeadingElement>(null);
  const [clipped, setClipped] = useState(false);
  useEffect(() => {
    const box = titleBox.current;
    if (!box) return;
    // Observing the heading, not the text inside it. The text element is swapped between a span
    // and a button as this very state changes, so an observer attached to it ends up watching a
    // node that is no longer in the document and never fires again — which left a title that
    // plainly overflowed with no way to expand it.
    const measure = () => {
      const el = box.firstElementChild as HTMLElement | null;
      // Only meaningful while collapsed: expanded, the text wraps and never overflows, which
      // would read as "nothing to reveal" and remove the control mid-interaction.
      if (!el || titleOpen) return;
      setClipped(el.scrollWidth > el.clientWidth + 1);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(box);
    // Web fonts land after first paint and change how wide the text is; without this the first
    // measurement is of a fallback face and can disagree with what is on screen.
    void document.fonts?.ready.then(measure);
    return () => ro.disconnect();
  }, [titleOpen, detail.issue]);

  const title = detail.issue ? detail.issue.split("\n")[0] : "UNTITLED RUN";

  return (
    <header className="runmeta">
      {/* One line, elided, expanding on click. A title is as long as whoever filed the issue made
          it, so left to wrap it is one line for one run and three for the next — and every row
          below moves when you switch between them. Clipping makes the header a fixed height and
          costs nothing a click does not return. */}
      <h2 ref={titleBox}>
        {clipped || titleOpen ? (
          <button
            type="button"
            className={titleOpen ? "title" : "title title--clipped"}
            aria-expanded={titleOpen}
            onClick={() => setTitleOpen((v) => !v)}
            data-tip={titleOpen ? undefined : (detail.issue?.split("\n")[0] ?? undefined)}
          >
            {title}
          </button>
        ) : (
          <span className="title title--clipped">
            {title}
          </span>
        )}
      </h2>
      <dl>
        <div>
          <dt>Run</dt>
          <dd>
            <samp>{detail.run_id}</samp>
          </dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd>
            <span className={`st st--${detail.status ?? "idle"}`}>
              {detail.status ?? "no row"}
            </span>
            {/* `updated_at` only moves on status transitions, so a killed run keeps
                claiming it's running. Staleness is the only tell the store can give. */}
            {stale && <span className="hazard"> / STALE</span>}
          </dd>
        </div>
        <div>
          <dt>Last activity</dt>
          <dd>
            <data value={lastSeen ?? ""}>{clock(lastSeen)}</data>
          </dd>
        </div>
        <div>
          <dt>Worktree</dt>
          <dd>
            {detail.worktree ? (
              <span className={detail.worktree.exists ? "" : "muted"}>
                {detail.worktree.exists ? "ON DISK" : "RECLAIMED"}
              </span>
            ) : (
              <span className="muted">—</span>
            )}
          </dd>
        </div>
        {/* Always present, so the header is one row for every run. Dropping the cell when there is
            no pull request made the meta a line taller for runs that had one, and every row below
            moved when you switched between them.

            "Not yet" and "none" are different answers and the run's status is what tells them
            apart: a run still going may open one, and a finished run that has not is a run that
            never will. */}
        <div>
          <dt>Pull request</dt>
          <dd>
            {detail.pull_request ? (
              <a href={detail.pull_request.url} target="_blank" rel="noopener noreferrer">
                #{detail.pull_request.number}
              </a>
            ) : (
              <span className="muted">{detail.status && !isTerminal(detail.status) ? "not yet" : "none"}</span>
            )}
          </dd>
        </div>
      </dl>
    </header>
  );
}
