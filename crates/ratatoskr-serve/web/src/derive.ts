import type { LiveEvent, NodeState, NodeTelemetry, NodeView } from "./api";

/**
 * A run's node state, rebuilt from its event stream.
 *
 * The store answers "where is this run now" and nothing else: it keeps each node's latest row, so
 * reading a past moment out of it gives final numbers against a historical position. The stream is
 * the only record that carries when each thing happened, so anything shown against a point in time
 * has to be derived from it.
 *
 * Only the pipeline's SHAPE still comes from the server — which nodes exist, and their stage and
 * lane. That is a property of the graph that ran, not of any moment inside it.
 */
export interface DerivedNode {
  state: NodeState;
  checkpoints: number;
  telemetry?: NodeTelemetry;
  /** Model calls so far in the current attempt, counted live before a checkpoint reports them. */
  cycles: number;
  /** Tools the node has reached for in this attempt. */
  used: Set<string>;
  /**
   * Whether the stream ever reported what this node cost.
   *
   * False means the log cannot answer the question — an older run whose checkpoints predate
   * carrying telemetry, or a node that ran no model. Reporting the derived zeros would be worse
   * than saying nothing: a node that plainly worked would read as having cost nothing.
   */
  costed: boolean;
}

/** The event kinds that mean a node is doing something. */
const WORKING = new Set([
  "tool_call",
  "tool_result",
  "model_text",
  "node_start",
  // Running the suite is the node working, and for the red team's baseline it is nearly all of it.
  "acceptance_step",
]);

/**
 * Fold `events` into per-node state as of the last event given.
 *
 * Pass a prefix of the stream to see where the run was at that point — that is the whole mechanism
 * behind scrubbing, and why it needs no separate replay engine.
 */
export function nodesFromEvents(events: readonly LiveEvent[]): Map<string, DerivedNode> {
  const out = new Map<string, DerivedNode>();
  const at = (name: string): DerivedNode => {
    const found = out.get(name);
    if (found) return found;
    const made: DerivedNode = {
      state: "idle",
      checkpoints: 0,
      cycles: 0,
      used: new Set(),
      costed: false,
    };
    out.set(name, made);
    return made;
  };

  for (const e of events) {
    if (!e.node) continue;
    const node = at(e.node);

    if (e.kind === "node_start") {
      // A fresh attempt. Counts start again, but the checkpoints already recorded stand: the
      // implementer is re-driven per converge iteration and each attempt is its own row.
      node.state = "working";
      node.cycles = 0;
      node.used = new Set();
      if (e.facts) {
        node.telemetry = {
          ...(node.telemetry ?? blank()),
          model: e.facts.model,
          tools: e.facts.tools,
          thinking: e.facts.thinking,
          reuses_session: e.facts.reuses_session,
        };
      }
      continue;
    }

    if (e.kind === "checkpoint") {
      node.checkpoints += 1;
      // An error is the only thing that tells a failed node from a finished one: both write a
      // checkpoint, and the fact of one proves only that the node stopped.
      node.state = e.error ? "failed" : "done";
      node.telemetry = {
        ...(node.telemetry ?? blank()),
        first_at: node.telemetry?.first_at ?? e.at,
        last_at: e.at,
        ...(e.facts
          ? {
              model: e.facts.model,
              tools: e.facts.tools,
              thinking: e.facts.thinking,
              reuses_session: e.facts.reuses_session,
            }
          : {}),
        ...(e.usage
          ? {
              input_tokens: e.usage.input_tokens,
              output_tokens: e.usage.output_tokens,
              cached_input_tokens: e.usage.cached_input_tokens,
              cache_creation_input_tokens: e.usage.cache_creation_input_tokens,
              reasoning_tokens: e.usage.reasoning_tokens,
              duration_ms: e.usage.duration_ms,
            }
          : {}),
        turns: e.turns ?? node.telemetry?.turns ?? null,
        tools_used: [...node.used],
      };
      if (e.usage) node.costed = true;
      continue;
    }

    if (e.kind === "usage" && e.usage) {
      node.costed = true;
      node.telemetry = {
        ...(node.telemetry ?? blank()),
        input_tokens: e.usage.input_tokens,
        output_tokens: e.usage.output_tokens,
        cached_input_tokens: e.usage.cached_input_tokens,
        cache_creation_input_tokens: e.usage.cache_creation_input_tokens,
        reasoning_tokens: e.usage.reasoning_tokens,
        duration_ms: e.usage.duration_ms,
      };
      continue;
    }

    if (e.kind === "tool_call") {
      node.cycles += 1;
      // `detail` carries the tool name for this kind.
      if (e.detail) node.used.add(e.detail);
    }
    if (WORKING.has(e.kind) && node.state !== "working") node.state = "working";
  }

  return out;
}

/**
 * The pipeline's shape from the server, with every per-moment fact taken from the stream.
 *
 * Call this only when the stream has something to say (`derived.size > 0`); then it is the
 * authority, and a node it does not mention has NOT RUN YET at this point in the run. Leaving such
 * a node at the state the server sent is what makes a rewind lie: the store holds the run's final
 * state, so the nodes that had not started would show as finished at every position.
 *
 * With no derived state at all — a run old enough that its log has rotated away — do not call
 * this. The store's final state is then the only answer there is, and it is an honest one.
 */
export function applyDerived(
  shape: readonly NodeView[],
  derived: Map<string, DerivedNode>,
): NodeView[] {
  return shape.map((n) => {
    const d = derived.get(n.name);
    // Not started at this point. It keeps what it is CONFIGURED to run on (`planned`), because
    // that is true before it runs, and loses everything it has not yet done.
    if (!d) {
      const { telemetry: _dropped, ...rest } = n;
      return { ...rest, state: "idle" as NodeState, checkpoints: 0 };
    }
    // State and checkpoint counts always come from the stream — those it can always prove. Cost
    // only when it actually carries it; otherwise the store's figures stand, which are true of
    // where the run ENDED and are marked as such by being all a rotated-away log can offer.
    const telemetry = d.costed ? d.telemetry : (n.telemetry ?? d.telemetry);
    return {
      ...n,
      state: d.state,
      checkpoints: d.checkpoints,
      ...(telemetry ? { telemetry } : {}),
    };
  });
}

function blank(): NodeTelemetry {
  return {
    model: null,
    turns: null,
    input_tokens: 0,
    output_tokens: 0,
    cached_input_tokens: 0,
    cache_creation_input_tokens: 0,
    reasoning_tokens: 0,
    thinking: false,
    duration_ms: null,
    tools: [],
    tools_used: [],
    reuses_session: false,
    first_at: null,
    last_at: null,
  };
}
