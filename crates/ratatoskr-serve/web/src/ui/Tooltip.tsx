/**
 * One tooltip for the whole page, driven by `data-tip`.
 *
 * The native `title` was doing this already, badly: it waits about a second, renders in the
 * platform's own chrome — pale, rounded, a system font — beside a monospace telemetry board, and
 * cannot be styled at all. Several of the things worth explaining here are icons, where the tooltip
 * *is* the label.
 *
 * A single element in a portal rather than a `::after` on each anchor, because of where the anchors
 * live. The feed scrolls under `overflow: auto` and would clip a pseudo-element to the row; the
 * node boxes sit inside React Flow's transformed pane and would scale their tooltip with the
 * viewport zoom and again with the magnifier, giving the least readable node the least readable
 * tooltip. Fixed-position, parented to `<body>`, escapes both.
 *
 * No dependency: this is a listener, a timer, and a rectangle.
 */
import { useEffect, useLayoutEffect, useRef, useState, type JSX } from "react";
import { createPortal } from "react-dom";

/** Long enough that crossing a row of icons does not flash six of them. */
const DELAY_MS = 220;
/** Between the anchor and the tooltip, and the closest it comes to the viewport edge. */
const GAP = 8;

export default function Tooltips(): JSX.Element | null {
  const [tip, setTip] = useState<{ text: string; rect: DOMRect } | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const timer = useRef<number | undefined>(undefined);
  // What the pointer is currently over, so moving within one anchor — over its icon, over its text
  // — does not keep restarting the delay and never showing anything.
  const anchor = useRef<Element | null>(null);

  useEffect(() => {
    const hide = () => {
      window.clearTimeout(timer.current);
      anchor.current = null;
      setTip(null);
    };

    const enter = (event: Event) => {
      const target = event.target;
      const el = target instanceof Element ? target.closest("[data-tip]") : null;
      if (el === anchor.current) return;
      window.clearTimeout(timer.current);
      anchor.current = el;
      setTip(null);
      const text = el?.getAttribute("data-tip");
      if (!el || !text) return;
      timer.current = window.setTimeout(
        () => setTip({ text, rect: el.getBoundingClientRect() }),
        DELAY_MS,
      );
    };

    document.addEventListener("pointerover", enter);
    // Keyboard reaches the same explanations: tabbing to a control shows what it does.
    document.addEventListener("focusin", enter);
    document.addEventListener("focusout", hide);
    // A tooltip is an explanation, not a layer to interact through — acting dismisses it.
    document.addEventListener("pointerdown", hide);
    // The rectangle was measured against a position the scroll has since invalidated. Capturing,
    // because the panes that scroll are inner ones and their scroll events do not bubble.
    window.addEventListener("scroll", hide, true);
    // `pointerover` cannot fire for a pointer that has left the window altogether.
    document.documentElement.addEventListener("pointerleave", hide);

    return () => {
      window.clearTimeout(timer.current);
      document.removeEventListener("pointerover", enter);
      document.removeEventListener("focusin", enter);
      document.removeEventListener("focusout", hide);
      document.removeEventListener("pointerdown", hide);
      window.removeEventListener("scroll", hide, true);
      document.documentElement.removeEventListener("pointerleave", hide);
    };
  }, []);

  // Placed after it is measured, not before: the width depends on the text, and the text is what
  // decides whether it fits above the anchor or has to go below it. `useLayoutEffect` runs before
  // paint, so the unpositioned first pass is never on screen.
  useLayoutEffect(() => {
    const el = box.current;
    if (!el || !tip) return;
    const own = el.getBoundingClientRect();
    const x = Math.max(
      GAP,
      Math.min(
        tip.rect.left + tip.rect.width / 2 - own.width / 2,
        window.innerWidth - own.width - GAP,
      ),
    );
    const above = tip.rect.top - own.height - GAP;
    el.style.left = `${Math.round(x)}px`;
    el.style.top = `${Math.round(above >= GAP ? above : tip.rect.bottom + GAP)}px`;
  }, [tip]);

  if (!tip) return null;
  return createPortal(
    <div className="tip" ref={box} role="tooltip">
      {tip.text}
    </div>,
    document.body,
  );
}
