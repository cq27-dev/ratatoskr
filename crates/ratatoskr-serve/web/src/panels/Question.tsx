import { clock } from "../ui/text";
import { answerQuestion, type LiveEvent } from "../api";
import { Compose } from "./Compose";

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
  return (
    <Compose
      // The asker, carried apart from `node`: the exchange is its own execution, so naming the
      // asker as this record's node would open an invocation of the asker that never happened.
      heading={<>/// {question.asked_by ?? question.node ?? "a node"} is waiting on you</>}
      aside={clock(question.at)}
      prompt={question.detail ?? undefined}
      placeholder="your answer…"
      submit=">>> ANSWER"
      focusKey={question.question_id ?? undefined}
      onSubmit={async (text) => {
        if (!question.question_id) return;
        await answerQuestion(question.question_id, text);
        onAnswered(question.question_id);
      }}
    />
  );
}
