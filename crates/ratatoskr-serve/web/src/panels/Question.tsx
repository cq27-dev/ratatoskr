import { useEffect, useRef, useState } from "react";
import { clock } from "../ui/text";
import { answerQuestion, type LiveEvent } from "../api";

/**
 * A run is blocked waiting for an answer. This has to be unmissable: until it is answered or
 * times out, a node is doing nothing, and the only thing that unblocks it is a person reading it.
 */
export function Question({
  question,
  onAnswered,
}: {
  question: LiveEvent;
  onAnswered: (questionId: string) => void;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    box.current?.focus();
  }, [question.question_id]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || busy || !question.question_id) return;
    setBusy(true);
    setError(null);
    try {
      await answerQuestion(question.question_id, text);
      setText("");
      onAnswered(question.question_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="ask" onSubmit={(e) => void submit(e)}>
      <div className="sec ask-head">
        <span>
          /// {question.node ?? "a node"} is waiting on you
        </span>
        <span>{clock(question.at)}</span>
      </div>
      <p className="ask-q">{question.detail}</p>
      <textarea
        ref={box}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="your answer…"
        rows={2}
        spellCheck={false}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) void submit(e);
        }}
      />
      <button type="submit" disabled={busy || !text.trim()}>
        {busy ? "SENDING…" : ">>> ANSWER"}
      </button>
      {error && <p className="ask-error hazard">{error}</p>}
    </form>
  );
}
