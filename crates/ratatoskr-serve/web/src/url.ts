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
import { short } from "./ui/text";

/**
 * What the path names: which project, and which run.
 *
 * In the path rather than the query string because these two are *what you are looking at*, and a
 * path is how a location is written down — `/ratatoskr/358e8441…` reads as a place, where
 * `?project=…&run=…` reads as a form submission.
 *
 * The selected node and the scrub position stay query parameters, because they are views into that
 * run rather than a different thing to look at. The run is written short — the same eight
 * characters the interface shows everywhere else, the way a git short hash works; the server
 * resolves the prefix and refuses an ambiguous one rather than picking — and because either can be absent while the run
 * still is what you are on. Nesting them would make `//implementer` a URL that has to mean
 * something.
 */
export interface Where {
  project: string | null;
  run: string | null;
}

/**
 * First path segments the server has already claimed.
 *
 * `/api` is the JSON API and `/assets` is the built bundle; both are matched before the fallback
 * that serves the dashboard, so a project of either name would be unreachable here. `open_all`
 * refuses those names when opening projects, which is what keeps this from having to be enforced
 * at read time too.
 */
const RESERVED = new Set(["api", "assets", "internal"]);

/**
 * Whether two run ids name the same run, allowing for one being a short form of the other.
 *
 * A link carries eight characters and the run list carries thirty-six, so the same run is written
 * two ways within one page load — and code that compares them with `===` sees a *different* run at
 * the moment the short one is expanded.
 */
export function sameRun(a: string | null, b: string | null): boolean {
  if (!a || !b) return a === b;
  return a.startsWith(b) || b.startsWith(a);
}

/** What the address bar currently names. */
export function readPath(): Where {
  const [project, run] = window.location.pathname
    .split("/")
    .filter(Boolean)
    .map(decodeURIComponent);
  if (!project || RESERVED.has(project)) return { project: null, run: null };
  return { project, run: run ?? null };
}

/**
 * Write the path, keeping whatever query string is there.
 *
 * The two halves of the URL have different owners — this one and nuqs — and each preserves the
 * other's, so neither has to know when the other runs. Silent when nothing would change, so this
 * can sit in an effect that fires on every render.
 */
export function writePath({ project, run }: Where): void {
  // A run without a project is not a place: the run id alone would parse back as a project name.
  const named = project ? (run ? [project, short(run)] : [project]) : [];
  const path = named.length ? `/${named.map(encodeURIComponent).join("/")}` : "/";
  if (path !== window.location.pathname) {
    window.history.replaceState(null, "", `${path}${window.location.search}`);
  }
}

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
