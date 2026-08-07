/**
 * The view, as an address.
 *
 * Which project, which run, which node, and how far into it — the four answers that decide what is
 * on screen. Keeping them in the URL is what makes a moment linkable: a reload lands where it was,
 * and pasting the address into a message shows someone else the same thing rather than whatever
 * their dashboard happened to be on.
 *
 * **`replaceState`, never `push`** — nuqs's default, and the reason it is left alone. Clicking
 * through eight runs is one act of looking, not eight places to go back to, and pushing would bury
 * whatever the viewer was doing before they opened the dashboard under a stack of its own state.
 * The consequence is that Back leaves the app: the address is a bookmark, not a trail.
 *
 * Three of the four are strings and need no parser. The scrub position is the one that does, which
 * is what nuqs is here for.
 */
import { createParser } from "nuqs";

/** An event, as far as the URL cares: something with a time. */
interface Moment {
  at: string;
}

/**
 * A position in a run, as `m:ss` or `h:mm:ss` into it.
 *
 * Elapsed time rather than the index the scrubber actually holds, for two reasons. It survives:
 * a live run's timeline grows, and a link to "event 298" means a different moment once fifty more
 * have landed, while "2:34 in" keeps meaning what it said. And it can be read — a link in a
 * message says where it points, and lines up with the elapsed readout beside the scrubber and with
 * the log file's own timestamps.
 *
 * The cost is precision: several events can share a second, and this cannot tell them apart. That
 * is why the URL is written from the position and never read back into it while scrubbing — see
 * `indexAtElapsed`.
 */
export const parseAsElapsed = createParser({
  parse(value: string): number | null {
    const m = /^(?:(\d+):)?(\d{1,2}):(\d{2})$/.exec(value);
    if (!m) return null;
    const [, h, mm, ss] = m;
    const minutes = Number(mm);
    const seconds = Number(ss);
    // `9:99` and `1:75:00` parse as digits and mean nothing. Rejected rather than normalised: a
    // link that cannot be honoured should fall back to the live end, not to a moment nobody meant.
    if (seconds > 59 || (h !== undefined && minutes > 59)) return null;
    return (h ? Number(h) * 3600 : 0) + minutes * 60 + seconds;
  },
  serialize(seconds: number): string {
    // Floored, not rounded, to agree with the elapsed readout beside the scrubber — `since()`
    // floors, so rounding here put `5:03` in the address bar next to a `+5:02` on screen, and
    // seeking to that rounded-up second landed a link one event past where it was made.
    const total = Math.max(0, Math.floor(seconds));
    const pad = (n: number) => String(n).padStart(2, "0");
    const h = Math.floor(total / 3600);
    return h > 0
      ? `${h}:${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`
      : `${Math.floor(total / 60)}:${pad(total % 60)}`;
  },
});

/** How far into the run its `index`th event is, in seconds. */
export function elapsedAt(timeline: readonly Moment[], index: number): number | null {
  const start = timeline[0]?.at;
  const moment = timeline[index]?.at;
  if (!start || !moment) return null;
  const ms = Date.parse(moment) - Date.parse(start);
  return Number.isFinite(ms) ? Math.max(0, ms / 1000) : null;
}

/**
 * Where a link lands: the last event at or before `seconds`.
 *
 * The address names a second, not an instant, and a busy second holds several events. This lands
 * on the last of them: `5:02` means "the run as of 5:02", so everything that happened during that
 * second has happened. Binary search — this runs over the whole timeline, which is the longest
 * list on the page.
 */
export function indexAtElapsed(timeline: readonly Moment[], seconds: number): number | null {
  const start = timeline[0]?.at;
  if (!start) return null;
  const target = Date.parse(start) + (seconds + 1) * 1000 - 1;
  let lo = 0;
  let hi = timeline.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (Date.parse(timeline[mid]!.at) <= target) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found >= 0 ? found : null;
}
