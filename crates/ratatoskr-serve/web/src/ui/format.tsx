/**
 * Rendering for the two kinds of text a run produces: structured output, and prose a model wrote.
 *
 * Everything here builds React nodes. Nothing is handed to `dangerouslySetInnerHTML`, and that is
 * not a style preference — a run's feed carries model output and tool results verbatim, including
 * whatever a package's install banner or a test's failure message put there. React escapes what it
 * renders; string-to-HTML would make this page the one place that untrusted text becomes markup.
 */
import type { JSX, ReactNode } from "react";

/* ── JSON ──────────────────────────────────────────────────────────────── */

/**
 * A JSON value, coloured by type.
 *
 * Walks the value rather than tokenising `JSON.stringify` output: the parse already happened, so
 * re-deriving structure from text would be inventing a second, worse parser to answer a question
 * the object already answers.
 */
function Value({ v, indent }: { v: unknown; indent: number }): JSX.Element {
  const pad = "  ".repeat(indent);
  const inner = "  ".repeat(indent + 1);

  if (v === null) return <span className="j-null">null</span>;
  if (typeof v === "boolean") return <span className="j-bool">{String(v)}</span>;
  if (typeof v === "number") return <span className="j-num">{String(v)}</span>;
  if (typeof v === "string") return <span className="j-str">{JSON.stringify(v)}</span>;

  if (Array.isArray(v)) {
    if (v.length === 0) return <span className="j-punc">[]</span>;
    return (
      <>
        <span className="j-punc">[</span>
        {v.map((item, i) => (
          <span key={i}>
            {"\n"}
            {inner}
            <Value v={item} indent={indent + 1} />
            {i < v.length - 1 && <span className="j-punc">,</span>}
          </span>
        ))}
        {"\n"}
        {pad}
        <span className="j-punc">]</span>
      </>
    );
  }

  const entries = Object.entries(v as Record<string, unknown>);
  if (entries.length === 0) return <span className="j-punc">{"{}"}</span>;
  return (
    <>
      <span className="j-punc">{"{"}</span>
      {entries.map(([k, item], i) => (
        <span key={k}>
          {"\n"}
          {inner}
          <span className="j-key">{JSON.stringify(k)}</span>
          <span className="j-punc">: </span>
          <Value v={item} indent={indent + 1} />
          {i < entries.length - 1 && <span className="j-punc">,</span>}
        </span>
      ))}
      {"\n"}
      {pad}
      <span className="j-punc">{"}"}</span>
    </>
  );
}

/** A checkpoint's output, or any other structured value, pretty-printed and coloured. */
export function Json({ value }: { value: unknown }): JSX.Element {
  return (
    <pre className="json">
      <Value v={value} indent={0} />
    </pre>
  );
}

/* ── Prose ─────────────────────────────────────────────────────────────── */

/**
 * The inline markers a model reaches for without being asked, and nothing else.
 *
 * Deliberately not a markdown parser. Headings and lists are left as literal text: a feed row is
 * one line in a stream of them, and a model that opens with `## Summary` should not be able to
 * restructure the page it is being displayed on. What is here is the formatting that carries
 * meaning inside a sentence — a file path, an identifier, an emphasised word.
 */
const FENCE = /```(?:[a-zA-Z0-9_-]*)\n?([\s\S]*?)```/g;
const INLINE = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(__[^_\n]+__)|(\*[^*\n]+\*)|(_[^_\n]+_)/g;

/**
 * Inline markers within one run of text that is known to contain no fence.
 *
 * Emphasis recurses, code does not. `**a `b` c**` is bold containing a code span, and a first
 * version that emitted the matched text as a plain string dropped the inner marker — every
 * backtick in a real run turned out to be inside bold, so nothing was ever rendered as code.
 * Inside a code span the opposite rule holds: an asterisk is an asterisk, and a shell snippet full
 * of them must not come apart. Recursion terminates because the inner text is strictly shorter
 * than the match that produced it.
 */
function inline(text: string, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(INLINE)) {
    const at = m.index;
    if (at > last) out.push(text.slice(last, at));
    const [whole, code, bold1, bold2, em1, em2] = m;
    const key = `${keyBase}-${at}`;
    if (code) {
      out.push(<code key={key}>{code.slice(1, -1)}</code>);
    } else if (bold1 || bold2) {
      out.push(<strong key={key}>{inline((bold1 ?? bold2)!.slice(2, -2), `${key}b`)}</strong>);
    } else if (em1 || em2) {
      out.push(<em key={key}>{inline((em1 ?? em2)!.slice(1, -1), `${key}e`)}</em>);
    }
    last = at + whole.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/**
 * Model prose with its inline formatting rendered: fenced blocks, code spans, bold and emphasis.
 *
 * Fences are taken out first so their contents are never scanned for inline markers — an asterisk
 * inside a code block is an asterisk, and a shell snippet full of them would otherwise come apart.
 */
export function Prose({ text }: { text: string }): JSX.Element {
  const out: ReactNode[] = [];
  let last = 0;
  let n = 0;
  for (const m of text.matchAll(FENCE)) {
    const at = m.index;
    if (at > last) out.push(...inline(text.slice(last, at), `p${n}`));
    out.push(
      <pre className="fence" key={`f${n}`}>
        {m[1]}
      </pre>,
    );
    last = at + m[0].length;
    n++;
  }
  if (last < text.length) out.push(...inline(text.slice(last), `p${n}`));
  return <>{out}</>;
}
