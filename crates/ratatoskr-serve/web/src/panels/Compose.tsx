import { useEffect, useRef, useState, type ReactNode } from "react";
import { Prose } from "../ui/format";

/**
 * Typing something to a node.
 *
 * One component for both directions of that conversation: a node asking a question and waiting for
 * an answer, and an operator saying something to a node that did not ask. They are the same act
 * from the typist's side — a line of prose that reaches a model mid-run — so they get the same box,
 * the same Enter-to-send, and the same place on the page. Two boxes that behaved almost alike would
 * be two things to learn and two things to keep in step.
 */
export function Compose({
  heading,
  aside,
  prompt,
  placeholder,
  submit,
  onSubmit,
  focusKey,
  onDismiss,
}: {
  /** The `/// …` line above the box, saying who this is with. */
  heading: ReactNode;
  /** The right-hand side of that line — a timestamp, or nothing. */
  aside?: ReactNode | undefined;
  /** What was asked, when something was. Absent when the operator speaks first. */
  prompt?: string | undefined;
  placeholder: string;
  /** The button's label, without its busy state. */
  submit: string;
  /** Resolves when the text has been delivered; rejects with what to show if it has not. */
  onSubmit: (text: string) => Promise<void>;
  /** Changing this refocuses the box — a new question deserves the cursor. */
  focusKey?: string | undefined;
  /** Close on a click elsewhere, for a box the operator opened themselves. Absent for one a node
   * opened by asking: that box closes when the question is answered, not when attention wanders.
   *
   * Only ever while empty. Typed text is work, and a stray click is not an instruction to throw
   * work away. */
  onDismiss?: (() => void) | undefined;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const form = useRef<HTMLFormElement>(null);

  useEffect(() => {
    box.current?.focus();
  }, [focusKey]);

  useEffect(() => {
    if (!onDismiss) return;
    const away = (e: PointerEvent) => {
      const target = e.target as Element | null;
      // The control that opens this box owns its own state: a click there must reach the button
      // rather than being eaten here, or the box would close and reopen on one click.
      if (form.current?.contains(target ?? null) || target?.closest?.("[data-compose-toggle]")) {
        return;
      }
      if (!text.trim()) onDismiss();
    };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [onDismiss, text]);

  const send = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      await onSubmit(text.trim());
      setText("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="ask" ref={form} onSubmit={(e) => void send(e)}>
      <div className="sec ask-head">
        <span>{heading}</span>
        <span>{aside}</span>
      </div>
      {/* The same renderer the feed uses: a node asking a question writes the way it writes
          everywhere else — backticks around a symbol, a fenced block of the code it is asking
          about. Shown raw, those are noise exactly where the reader is being asked to decide
          something. */}
      {prompt && (
        <div className="ask-q">
          <Prose text={prompt} />
        </div>
      )}
      <textarea
        ref={box}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={placeholder}
        rows={2}
        spellCheck={false}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) void send(e);
        }}
      />
      <button type="submit" disabled={busy || !text.trim()}>
        {busy ? "SENDING…" : submit}
      </button>
      {error && <p className="ask-error hazard">{error}</p>}
    </form>
  );
}
