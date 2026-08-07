/**
 * A colour per node, derived from its name and nothing else.
 *
 * The point is recognition across surfaces: the same hue marks a node's box in the graph, its name
 * in the feed, and its stretch of the scrubber, so a glance connects them without reading. Derived
 * rather than assigned because the node set is not fixed — a workflow can name nodes this build has
 * never heard of, and a lookup table would give them all the same colour or none.
 *
 * Deterministic, so a node is the same colour in this tab, the next one, and a run someone else
 * exported. Nothing here reads state.
 */

/** FNV-1a. Small, stable, and well spread over short lowercase identifiers, which is all it sees. */
function hash(name: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    // The FNV prime, as a shift/add so the intermediate stays inside a 32-bit int.
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h;
}

/**
 * How many hues are on offer, and the lightnesses each can take.
 *
 * Quantised rather than continuous. Taken straight off the hash, two of this pipeline's nodes
 * landed seven degrees apart — far enough to be different values and not nearly far enough to look
 * it, which reads as one node drawn inconsistently. Bucketing guarantees any two distinct hues are
 * a whole step apart, and the lightness axis keeps the two that share a bucket apart anyway.
 */
const HUES = 18;
const LIGHTNESS = [66, 74, 82];

/**
 * The arc of the wheel these hues are drawn from — red excluded.
 *
 * `--hazard` is red and means a failure, so a node washed red reads as a broken one whatever its
 * actual state. The first version of this gave the overseer hue 0 and its idle box looked like an
 * error. Starting at 40 keeps the palette clear of that and of the near-reds that suggest it.
 */
const HUE_MIN = 40;
const HUE_SPAN = 280;

/** The node's hue in degrees and its lightness, both fixed by its name. */
function tone(node: string): { h: number; l: number } {
  const n = hash(node);
  return {
    h: HUE_MIN + (n % HUES) * (HUE_SPAN / HUES),
    // A different slice of the hash, so hue and lightness do not move together.
    l: LIGHTNESS[(n >>> 9) % LIGHTNESS.length]!,
  };
}

/** The node's hue, in degrees. */
export function hue(node: string): number {
  return tone(node).h;
}

/**
 * Text colour for a node's name.
 *
 * Light and only moderately saturated: the panel is near-black, so a saturated mid-tone reads as
 * a warning rather than a label, and this must not compete with the hazard red or the live green
 * that already mean something specific.
 */
export function accent(node: string): string {
  const { h, l } = tone(node);
  return `hsl(${h} 52% ${l}%)`;
}

/**
 * The node box's wash: the tint along the bottom edge, fading to nothing before the top.
 *
 * Transparent at the far end rather than a colour, so whatever the box's own background is — panel,
 * or the selected/failed variants — shows through unchanged and this stays an accent on top of the
 * existing states instead of replacing them.
 */
export function wash(node: string): string {
  const { h } = tone(node);
  return `linear-gradient(to top, hsl(${h} 60% 58% / 0.20), hsl(${h} 60% 58% / 0.05) 45%, transparent 78%)`;
}

/** One slice of the scrubber: a node and the share of the slot it takes. */
type Slice = { node: string; colour: string };

/**
 * A gradient banding the scrubber by which node each stretch of the run belongs to.
 *
 * Bucketed rather than one stop per event: a long run has hundreds of events and two nodes running
 * in parallel alternate between them, which at one stop each renders as shimmer. A bucket that
 * caught several nodes is split evenly between them, which is what "both were running here" looks
 * like at this width.
 *
 * `nodes[i]` is the node the i-th event belongs to, or null for an event that belongs to none.
 */
export function band(nodes: (string | null)[], buckets = 96): string {
  if (nodes.length === 0) return "transparent";
  const stops: string[] = [];
  const width = 100 / buckets;

  for (let b = 0; b < buckets; b++) {
    const from = Math.floor((b * nodes.length) / buckets);
    const to = Math.max(from + 1, Math.floor(((b + 1) * nodes.length) / buckets));
    // In first-appearance order, so a split reads left-to-right as the nodes were seen.
    const seen: Slice[] = [];
    for (let i = from; i < to && i < nodes.length; i++) {
      const n = nodes[i];
      if (n && !seen.some((s) => s.node === n)) seen.push({ node: n, colour: accent(n) });
    }
    const start = b * width;
    if (seen.length === 0) {
      stops.push(`transparent ${start}%`, `transparent ${start + width}%`);
      continue;
    }
    const share = width / seen.length;
    seen.forEach((s, i) => {
      const a = start + i * share;
      // Hard stops: a band is a legend, not a gradient. Interpolating between two nodes' colours
      // would invent a third that means nothing.
      stops.push(`${s.colour} ${a}%`, `${s.colour} ${a + share}%`);
    });
  }
  return `linear-gradient(to right, ${stops.join(", ")})`;
}
