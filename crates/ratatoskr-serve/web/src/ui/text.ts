/**
 * The three ways a value is shortened for display.
 *
 * Shared because they decide how the page reads, not just how it computes: a run id is eight
 * characters everywhere or the rail and the detail pane disagree about what the same run is called.
 */
export const short = (id: string | null) => (id ? id.slice(0, 8) : "—");

export const clock = (ts: string | null) => (ts ? ts.slice(11, 19) : "—");

/**
 * How far into the run a moment is, as `m:ss` or `h:mm:ss`.
 *
 * The wall clock says when something happened in the world, which is rarely the question — a run
 * is read as a duration, and "eleven minutes in" is what places an event against the rest of it.
 * Both are shown because only the clock lines up with a log file or a colleague's screenshot.
 */
export function since(startIso: string | null, atIso: string | null): string {
  if (!startIso || !atIso) return "—";
  const ms = Date.parse(atIso) - Date.parse(startIso);
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `+${h}:${pad(m)}:${pad(sec)}` : `+${m}:${pad(sec)}`;
}
