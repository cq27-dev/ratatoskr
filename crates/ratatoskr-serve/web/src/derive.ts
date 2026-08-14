import type {
  LiveEvent,
  NodeFacts,
  NodeState,
  NodeTelemetry,
  NodeView,
  RunStage,
  RunStatus,
} from "./api";

/**
 * A run's node state, rebuilt from its event stream.
 *
 * The store answers "where is this run now" and nothing else: it keeps each node's latest row, so
 * reading a past moment out of it gives final numbers against a historical position. The stream is
 * the only record that carries when each thing happened, so anything shown against a point in time
 * has to be derived from it.
 *
 * The pipeline's SHAPE — which nodes exist, and their stage and lane — comes from the server, since
 * that is a property of the graph that ran rather than of any moment inside it. The one exception is
 * a node the shape does not place: the server derives placement from checkpoints, so a node that has
 * started and not yet finished is placed here, from the stream, or it would have no box at all.
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
 * A stream event read as the box it is drawn in.
 *
 * `node` is the box; `member` is the stage that actually announced it, which for an ordinary
 * one-stage node is the same name. The distinction is the point: renaming a member's event to its
 * box and stopping there tells every fold below that the box did what the member did, and a fold
 * that then treats each node as one stage marks a box finished on a member's checkpoint, counts a
 * member's row as the box's, and lets the last member's model overwrite its sibling's.
 *
 * Branded, and `inNodeBoxes` is the only thing that can make one. That is what makes a fold's
 * requirement checkable: `nodesFromEvents(rawEvents)` does not compile, and the seven rounds of
 * defects this file has had were all a consumer reading a list nobody had boxed.
 */
declare const boxed: unique symbol;
export type BoxedEvent = LiveEvent & {
  /** The stage that announced this. Equal to `node` unless a box was folded around it. */
  readonly member: string | null;
  readonly [boxed]: true;
};

/**
 * Fold `events` into per-node state as of the last event given.
 *
 * Pass a prefix of the stream to see where the run was at that point — that is the whole mechanism
 * behind scrubbing, and why it needs no separate replay engine.
 *
 * A box's members are folded the way the server folds them, and the two must keep answering the
 * same question the same way — a run's numbers should not change when its log rotates away:
 *
 * - **cost** is the latest record of each member, folded arithmetically. Not the latest record
 *   overall: a box's own aggregate carries no turn and reports zeros, so overwriting with it wipes
 *   what its members spent, and with two members the last one to finish would speak for both.
 * - **completion** is the box's own record AND no member left working. A member's checkpoint alone
 *   proves the box STARTED — the classifier finishes while the author is still writing tests —
 *   which is `checkpointed_state`'s rule on the other side. The second half is what an aggregate
 *   cannot tell you: a box may BE a stage with peers composed into it (`ratatoskr-nodes`'s
 *   `validate.rs` accepts "a stage of its own called `{node}`" as a membership target), and that
 *   stage's ordinary checkpoint is the box's own record and one member's at once. Deciding on the
 *   record alone finishes such a box on whichever host returns first and drops the control aimed at
 *   its peer. An operation host writes its aggregate only after its stages return, so for a box
 *   that has one the two halves land together and nothing changes.
 * - **working** is a COUNT of live invocations, not a flag. A workflow may invoke one stage several
 *   times at once — `Promise.all([probe(a), probe(b)])` — and both invocations record under that
 *   stage's one name, so a flag is cleared by whichever finishes first and the box drops the Stop
 *   aimed at the other. Each `node_start` counts one more invocation live and each checkpoint one
 *   fewer; the member is working while the count is above zero. This says nothing about WHICH
 *   invocation a record belongs to — everything per-invocation (its cost, its cycles, its tools)
 *   is still one state per member, and #285 is where that is resolved.
 * - **the count** is the box's own rows, so a converge iteration is one, not one per member.
 */
export function nodesFromEvents(events: readonly BoxedEvent[]): Map<string, DerivedNode> {
  /** One member's latest turn, kept apart so the box can be the fold of them rather than the last. */
  interface Member {
    /** Where this member's OWN records leave it. The box is the fold of these, never the last one. */
    state: NodeState;
    /**
     * Invocations of this member that have started and not yet checkpointed.
     *
     * Arithmetic, because one stage may be invoked more than once at a time and every invocation
     * records under the same name: a checkpoint ends AN invocation, not the member. Only zero here
     * means the member has stopped.
     */
    live: number;
    telemetry?: NodeTelemetry;
    cycles: number;
    used: Set<string>;
    costed: boolean;
  }
  const boxes = new Map<string, { checkpoints: number; members: Map<string, Member> }>();
  const at = (name: string) => {
    const found = boxes.get(name);
    if (found) return found;
    const made = { checkpoints: 0, members: new Map<string, Member>() };
    boxes.set(name, made);
    return made;
  };
  const memberOf = (box: { members: Map<string, Member> }, name: string): Member => {
    const found = box.members.get(name);
    if (found) return found;
    const made: Member = { state: "idle", live: 0, cycles: 0, used: new Set(), costed: false };
    box.members.set(name, made);
    return made;
  };

  for (const e of events) {
    if (!e.node) continue;
    const box = at(e.node);
    const name = e.member ?? e.node;
    const own = name === e.node;
    const member = memberOf(box, name);

    if (e.kind === "node_start") {
      // A fresh attempt of THIS member. Its counts start again and its siblings' stand, which is
      // what keeps a box that ran two halves from reporting only the second. The checkpoints
      // already recorded stand too: the implementer is re-driven per converge iteration.
      member.live += 1;
      member.state = "working";
      member.cycles = 0;
      member.used = new Set();
      if (e.facts) {
        member.telemetry = {
          ...(member.telemetry ?? blank()),
          model: e.facts.model,
          tools: e.facts.tools,
          thinking: e.facts.thinking,
          reuses_session: e.facts.reuses_session,
        };
      }
      continue;
    }

    if (e.kind === "checkpoint") {
      // This record ends the MEMBER's turn, whoever it belongs to. What it means for the BOX is
      // decided below, from every member at once — a member finishing is never on its own the box
      // finished, and for a box that is itself a stage its own record is one member's too.
      //
      // An error is the only thing that tells a failed member from a finished one: both write a
      // checkpoint, and the fact of one proves only that the member stopped. Only the box's own
      // record can fail the box, as before.
      //
      // One invocation ended, not necessarily the member: a stage invoked twice at once writes two
      // checkpoints and is live until the second. Floored at zero because a stream is not
      // guaranteed balanced — an ingested log may begin mid-run, and a checkpoint whose start this
      // view never saw must not push the count negative and make the member unstoppable.
      member.live = Math.max(0, member.live - 1);
      member.state = own && e.error ? "failed" : "done";
      if (own) box.checkpoints += 1;
      member.telemetry = {
        ...(member.telemetry ?? blank()),
        first_at: member.telemetry?.first_at ?? e.at,
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
        turns: e.turns ?? member.telemetry?.turns ?? null,
        tools_used: [...member.used],
      };
      // Presence is the signal, the same as for a `usage` event below. A checkpoint covering no
      // turn — a box's own aggregate — carries no cost keys at all now, so the server reports no
      // usage for it rather than a zeroed one. The counters themselves are never the basis: an
      // endpoint may make a real call and report no accounting, and `reasoning_tokens` is
      // hardcoded to zero by one provider while `output_tokens` is under-reported. A zero that
      // arrives is a measurement; what a turn-less record has is nothing to read.
      if (e.usage) member.costed = true;
      continue;
    }

    if (e.kind === "usage" && e.usage) {
      // Unconditional, unlike the checkpoint above. A `usage` event is the endpoint's own report of
      // a turn: its presence is the authority and a zero is a measurement, not an absence. Doubting
      // it leaves the member uncosted, and `fromStream` then keeps the server's telemetry — the
      // run's FINAL state — so scrubbing back to an earlier attempt shows a later one's numbers.
      member.costed = true;
      member.telemetry = {
        ...(member.telemetry ?? blank()),
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
      member.cycles += 1;
      // `detail` carries the tool name for this kind.
      if (e.detail) member.used.add(e.detail);
    }
    if (WORKING.has(e.kind)) member.state = "working";
  }

  const out = new Map<string, DerivedNode>();
  for (const [name, box] of boxes) {
    const members = [...box.members.values()];
    // The box, from all of its members and its own rows. A member still working outranks
    // everything, including a failure: that is the live half a Stop has to keep reaching, and it is
    // what the box did on the way to the record it already has. Below that, the box's own record
    // settles it — failed if it carried an error, done otherwise — and a member's record with no
    // record of the box's own says only that the box started.
    //
    // A member is working while any invocation of it is live, which is what keeps a box whose
    // aggregate already landed from reading finished when a member has been re-entered: the count
    // is above zero again, and clause 1 outranks the checkpoints. `state` still carries the working
    // case for a stream whose only evidence is a tool call or model text — one that reports no
    // `node_start` has no invocation to count.
    const working = (m: Member) => m.live > 0 || m.state === "working";
    const state: NodeState = members.some(working)
      ? "working"
      : box.members.get(name)?.state === "failed"
        ? "failed"
        : box.checkpoints > 0
          ? "done"
          : members.some((m) => m.state !== "idle")
            ? "working"
            : "idle";
    const folded = members
      .map((m) => m.telemetry)
      .filter((t): t is NodeTelemetry => !!t)
      .reduce<NodeTelemetry | undefined>(
        (into, next) => (into ? fold(into, next) : next),
        undefined,
      );
    out.set(name, {
      state,
      checkpoints: box.checkpoints,
      ...(folded ? { telemetry: folded } : {}),
      cycles: members.reduce((n, m) => n + m.cycles, 0),
      used: new Set(members.flatMap((m) => [...m.used])),
      costed: members.some((m) => m.costed),
    });
  }
  return out;
}

/**
 * Fold one member's turn into another, as `NodeTelemetry::fold` does on the server.
 *
 * The same arithmetic on both sides, deliberately: figures add, a figure nobody reported stays
 * unreported, models are named distinctly rather than one overwriting the other, and tool lists are
 * the union. A box that ran two profiles names both — true of the box, while each member's own row
 * stays true of its turn.
 */
function fold(into: NodeTelemetry, next: NodeTelemetry): NodeTelemetry {
  const distinct = (a: string[], b: string[]) => [...new Set([...a, ...b])];
  const names = (a: string | null, b: string | null) =>
    a && b ? distinct(a.split(", "), b.split(", ")).join(", ") : (a ?? b);
  const sum = (a: number | null, b: number | null) => (a === null && b === null ? null : (a ?? 0) + (b ?? 0));
  const earliest = (a: string | null, b: string | null) => (a && b ? (a < b ? a : b) : (a ?? b));
  const latest = (a: string | null, b: string | null) => (a && b ? (a > b ? a : b) : (a ?? b));
  return {
    model: names(into.model, next.model),
    turns: sum(into.turns, next.turns),
    input_tokens: into.input_tokens + next.input_tokens,
    output_tokens: into.output_tokens + next.output_tokens,
    cached_input_tokens: into.cached_input_tokens + next.cached_input_tokens,
    cache_creation_input_tokens: into.cache_creation_input_tokens + next.cache_creation_input_tokens,
    reasoning_tokens: into.reasoning_tokens + next.reasoning_tokens,
    thinking: into.thinking || next.thinking,
    duration_ms: sum(into.duration_ms, next.duration_ms),
    tools: distinct(into.tools, next.tools),
    tools_used: distinct(into.tools_used, next.tools_used),
    reuses_session: into.reuses_session || next.reuses_session,
    first_at: earliest(into.first_at, next.first_at),
    last_at: latest(into.last_at, next.last_at),
  };
}

/**
 * How many times the implementer was re-entered, by the route that brought it back.
 *
 * `full()` returns to the implementer three different ways and the graph draws each as its own
 * edge, so the counts have to be separated before they can be drawn.
 */
export interface ConvergeLoops {
  /** The verifier faulted the code, and `iterate()` ran again on the same plan. */
  fix: number;
  /** The finding faulted the *plan*: `analyst()` re-ran first, so the loop reaches further back. */
  replan: number;
  /** The suite never went clean, so the implementer ran again without the verifier ever seeing it. */
  retry: number;
}

/**
 * Classify each re-entry of the implementer from the order its neighbours started.
 *
 * A traversal is a RE-ENTRY, so the first `implementer` `node_start` is the initial `implement()`
 * call and not a loop — counting the implementer's checkpoints instead overstates every run by
 * one, and drew `×1` on runs that went straight through.
 *
 * Classified by what started in between, because nothing in the record names the route directly:
 * an `analyst` start means the plan was re-made, a `verifier` start without one means the code was
 * faulted, and neither means the suite never went clean and the verifier sat the cycle out. Every
 * other node in the segment decides nothing and is ignored.
 *
 * The `verifier` start is the whole signal, and the `referee` must NOT be added to it. The referee
 * precedes every correction regardless of route: `iterate_host` calls `referee_judgement`
 * unconditionally at the top of the stage (`ratatoskr-nodes/src/workflow.rs:915`), before it
 * inspects anything, so a `referee` start appears on the tests-not-clean path too — which reaches
 * `iterate({})` without `full()` ever calling `verify()`. Counting it turns a real failed-test
 * retry (`implementer -> referee -> implementer`) into a fix and draws an edge out of a verifier
 * that never ran. `verify()` is reachable only inside `full()`'s `if (testsClean)` branch, so the
 * `verifier` start alone separates the two exactly.
 *
 * Array order only — never the `at` strings, whose precision is not ours to rely on. Pass a prefix
 * and the counts are the counts as of that point, which is what keeps the edges honest while
 * someone scrubs.
 */
export function convergeLoops(events: readonly BoxedEvent[]): ConvergeLoops {
  const out: ConvergeLoops = { fix: 0, replan: 0, retry: 0 };
  let entered = false;
  let since = new Set<string>();

  for (const e of events) {
    if (!e.node || e.kind !== "node_start") continue;
    if (e.node !== "implementer") {
      since.add(e.node);
      continue;
    }
    if (!entered) entered = true;
    else if (since.has("analyst")) out.replan += 1;
    else if (since.has("verifier")) out.fix += 1;
    else out.retry += 1;
    since = new Set();
  }

  return out;
}

/**
 * Whether the red team has handed the tree over to the implementer.
 *
 * The two share a stage, so no forward stage edge connects them and the graph reads as a fork —
 * which it is not. `implement()` cannot start until `redTeam()` has finished: `implement_host`
 * (`ratatoskr-nodes/src/workflow.rs`) errors with "implement() cannot start until the awaited
 * redTeam() call has finished" unless `red_team_completed` is set, and `red_team_host` sets that
 * flag only after its checkpoint write succeeds. Both boxes having left `idle` is therefore
 * complete proof the hand-off happened, and no ordering check is needed — nor wanted: `at` is
 * wall-clock text of unconfirmed precision, and historical runs batch-write their checkpoints, so
 * a missing upstream checkpoint says nothing about whether the stage completed.
 *
 * A failed red team still counts. The hand-off is what this answers; whether the stage went well
 * is the box's business.
 *
 * Takes the same event-corrected list the boxes are drawn from, so it cannot drift out of step
 * with them — an edge reading a different source than its endpoints is exactly the bug in c9b5e13.
 *
 * Only where the two SHARE a column, which is the case this exists for. The edge is a short
 * vertical step down the lane gap, geometry that assumes exactly that; a layout is free to put the
 * two in different columns — or to declare none, and have the client place them — and there the
 * same edge renders as a diagonal across the graph, duplicating the forward edge the columns
 * already draw.
 */
export function forkHandoff(nodes: readonly NodeView[]): boolean {
  const started = (name: string) => nodes.find((n) => n.name === name && n.state !== "idle");
  const redTeam = started("redteam");
  const implementer = started("implementer");
  return !!redTeam && !!implementer && redTeam.stage === implementer.stage;
}

/**
 * Whether a forward hand-off from `source` to `target` is a claim the record supports.
 *
 * Adjacent columns are drawn joined, so every stage edge asserts that the earlier column ran the
 * later one. A DECLARED layout is exactly that assertion, and is drawn in full. An appended node is
 * not: the shape never named it, and it is given a trailing column in first-mention order purely so
 * it has somewhere to be. Joining the last shaped column to it invents a hand-off — the referee an
 * iterating run appends is invoked by the implementer mid-converge, and reads instead as the
 * pipeline's final stage, judging what the publisher published.
 *
 * Chained edges *within* the appended run are kept. For a run whose workflow declared no layout,
 * the order the stream first saw each node is the only ordering it has.
 */
export function handoffDrawn(source: NodeView, target: NodeView): boolean {
  return target.shaped !== false || source.shaped === false;
}

/**
 * The stages whose records are `name`'s work — what to look for when the box is asked about.
 *
 * Its members and its own name, from the run's recorded REGISTRY and never from its `nodes`. A box is a box whether or not anything
 * has checkpointed under it, and `nodes` only knows the ones that have; reading membership from
 * there empties it for exactly the box currently executing, which is the one being looked at.
 *
 * A name the registry does not carry — one typed into the address bar, or a run recorded before the
 * registry travelled with it — is exactly itself, which is what it was.
 */
export function stagesOf(stages: readonly RunStage[], name: string): string[] {
  const members = stages.filter((s) => s.node === name).map((s) => s.id);
  if (!members.length) return [name];
  // The box's own name as well as its members'. `redteam`, `implementer` and `context` are not
  // stage ids, so a members-only answer drops everything logged under the box itself — its
  // operation host's aggregate, and the acceptance suite, which is nearly all of the red team's
  // visible work. Selecting a box then hides most of what it did.
  return members.includes(name) ? members : [...members, name];
}

/**
 * Rename each event to the box it is drawn in.
 *
 * A stage keeps its own identity, so a `node_start` from the red team says `redteam_author` while
 * the graph draws one `redteam` box. Everything that folds the stream into node state — the boxes,
 * the converge-loop edges, what a control can be aimed at — is about the box, and would otherwise
 * see a name it has never heard of and invent a node for it.
 *
 * FROM THE REGISTRY, NOT FROM `nodes`. The mapping has to hold before anything has checkpointed:
 * `node_start` for `context_distillation` with no checkpoint yet is the live window an operator
 * reaches for Stop in, and the server has not placed the `context` box then — so a mapping built
 * from `nodes` is empty at exactly that moment, the half draws as its own box, and the Stop it
 * offers goes to a name the runtime never polls. It is dead until a checkpoint makes the real box
 * appear, which is after the operator needed it.
 *
 * The events themselves are left alone: the feed shows which half actually ran, which is the whole
 * point of the split. Only this reading of them is renamed.
 */
export function inNodeBoxes(
  events: readonly LiveEvent[],
  stages: readonly RunStage[],
): readonly BoxedEvent[] {
  const box = new Map<string, string>();
  for (const stage of stages) {
    if (stage.id !== stage.node) box.set(stage.id, stage.node);
  }
  // The one place a `BoxedEvent` is made. `member` keeps what `node` used to say, because a fold
  // that cannot tell a member's record from its box's answers the box for the member.
  return events.map(
    (e) => ({ ...e, node: (e.node && box.get(e.node)) ?? e.node, member: e.node }) as BoxedEvent,
  );
}

/**
 * What each node announced when it started, plus the tools it has reached for since.
 *
 * A checkpoint carries the same facts and carries them better, but only once the node has stopped.
 * This is what fills a box while it works, and it is the only source there is for the whole of that
 * window.
 *
 * Feed it the stream READ AS BOXES, exactly as `nodesFromEvents` is fed. A member announces itself
 * under its own id, so a map built from the raw stream is keyed by `context_distillation` while the
 * graph asks it for `context` — and the box then draws with no model, no tools and no cycle count
 * for as long as the member is the one running.
 *
 * PER MEMBER, folded into the box, for the reason `nodesFromEvents` keeps its members apart: a
 * workflow may have two stages of one box in flight at once, both announce under the box, and a
 * single entry per box would have the later start throw away the cycles, tools and model of the
 * member already working. Before any checkpoint this map is all the box has to draw with, so that
 * activity does not move — it vanishes.
 *
 * A `node_start` is a fresh attempt of THAT member and restarts only its own counts. Which
 * INVOCATION of a member the numbers belong to is not answered here — a stage invoked twice at once
 * still folds into one state per member, and #285 is where that is resolved.
 */
export function liveNodes(events: readonly BoxedEvent[]): Map<string, LiveNode> {
  const boxes = new Map<string, Map<string, LiveNode>>();
  for (const e of events) {
    if (!e.node) continue;
    let members = boxes.get(e.node);
    if (!members) boxes.set(e.node, (members = new Map()));
    const name = e.member ?? e.node;
    let at = members.get(name);
    if (!at) members.set(name, (at = { cycles: 0, used: new Set() }));
    if (e.kind === "node_start") {
      // Every `node_start`, not only one carrying facts — `facts` is optional, and restarting on
      // its presence made a re-entry that announced nothing accumulate onto the previous attempt.
      // What it announced survives the restart: the model is a fact about the member, not about
      // the attempt, and a later start that says nothing has not changed it.
      at.cycles = 0;
      at.used = new Set();
      if (e.facts) at.facts = e.facts;
      continue;
    }
    if (e.kind === "tool_call") {
      at.cycles += 1;
      // `detail` is the tool name for this kind.
      if (e.detail) at.used.add(e.detail);
    }
  }

  const out = new Map<string, LiveNode>();
  for (const [box, members] of boxes) {
    const all = [...members.values()];
    // The same arithmetic the checkpointed fold uses on telemetry: counts add, tools are the union,
    // and two profiles are both named rather than one overwriting the other.
    const facts = all
      .map((m) => m.facts)
      .filter((f): f is NodeFacts => !!f)
      .reduce<NodeFacts | undefined>(
        (into, next) =>
          into
            ? {
                model: [...new Set([...into.model.split(", "), ...next.model.split(", ")])].join(
                  ", ",
                ),
                tools: [...new Set([...into.tools, ...next.tools])],
                thinking: into.thinking || next.thinking,
                reuses_session: into.reuses_session || next.reuses_session,
              }
            : next,
        undefined,
      );
    out.set(box, {
      ...(facts ? { facts } : {}),
      cycles: all.reduce((n, m) => n + m.cycles, 0),
      used: new Set(all.flatMap((m) => [...m.used])),
    });
  }
  return out;
}

/** What a node has said about itself so far, before it has checkpointed anything. */
export interface LiveNode {
  facts?: NodeFacts;
  cycles: number;
  /** Tools called so far in this attempt. */
  used: Set<string>;
}

/**
 * Nodes a current control can reach.
 *
 * The event stream records the process's active attempt, including the concurrent delivery
 * stages after convergence. The stored pipeline is only a fallback before the first node event:
 * it cannot distinguish a completed implementer from a publisher or bookkeeper now making a
 * provider request.
 */
export function workingNodeNames(
  nodes: readonly NodeView[],
  events: readonly BoxedEvent[],
): string[] {
  const active = [...nodesFromEvents(events)]
    .filter(([, node]) => node.state === "working")
    .map(([name]) => name);
  if (events.some((event) => event.kind === "node_start")) return active;
  return nodes.filter((node) => node.state === "working").map((node) => node.name);
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
 *
 * A node the shape does not place still gets a box, and the CLIENT places it. The server can only
 * place such a node once it has checkpointed, and orders those by first checkpoint — but a workflow
 * that declared no layout may run its hosts concurrently, and completion order is then not start
 * order. Inheriting the server's numbers would move a box the moment its checkpoint landed, and
 * since column adjacency is what the graph draws its hand-offs from, the arrow between two such
 * boxes would reverse. The stream is the one record that knows when each node started, so every
 * unplaced node is ordered by first mention here and keeps that place across refreshes.
 *
 * Only the ones the shape does not place. A declared layout is the graph the workflow asked for and
 * is reproduced exactly; a run whose shape names some of its nodes and not others keeps the named
 * ones where the shape puts them.
 *
 * `ended` is the run's status when it has stopped AND the view is at the live end — `null` at every
 * other position, and while the run is still executing. It is what lets a node the stream cannot
 * finish be settled; see `fromStream`.
 */
export function applyDerived(
  shape: readonly NodeView[],
  derived: Map<string, DerivedNode>,
  ended: RunStatus | null = null,
): NodeView[] {
  // Which node a failed run died in, when the record names one.
  //
  // A host error writes no checkpoint and no node-scoped event, so the node it killed is left
  // "working" by the fold and the run emits nothing more to move it. At the live end of a failed
  // run those nodes are therefore the candidates: the ones that started and never finished. This is
  // evidence about which node actually died — no other record has it, and it holds wherever the node
  // sits and whatever the shape declared.
  //
  // Spend it only where it names someone: exactly one candidate and no other. The run's status is a
  // fact about the RUN — it says the run died, never which node died in it — and a workflow may run
  // several hosts at once, so with two in flight both keep the state the stream gave them. An
  // unattributed failure is worse than a correct attribution and much better than a wrong one.
  const candidates =
    ended === "failed"
      ? [...derived].filter(([, d]) => d.state === "working").map(([name]) => name)
      : [];
  const died = candidates.length === 1 ? candidates[0] : null;

  const shaped = shape.filter((n) => n.shaped !== false);
  const placed = shaped.map((n) => fromStream(n, derived.get(n.name), ended, n.name === died));

  // Trailing columns: everything the shape does not place, in the order the stream first mentioned
  // it. A node the server did place from a checkpoint keeps everything else it said about it — its
  // caller above all — and only its column is taken back. One the stream has never mentioned has no
  // start to be ordered by and holds the last columns, which is where the server had it.
  const known = new Set(shaped.map((n) => n.name));
  const unplaced = new Map(
    shape.filter((n) => n.shaped === false).map((n) => [n.name, n] as const),
  );
  const order = [...derived.keys()].filter((name) => name !== ISSUE_NODE && !known.has(name));
  order.push(...[...unplaced.keys()].filter((name) => !derived.has(name)));

  const base = placed.reduce((max, n) => Math.max(max, n.stage + 1), 0);
  const extra = order.map((name, i) => ({
    ...fromStream(unplaced.get(name) ?? unrun(name), derived.get(name), ended, name === died),
    stage: base + i,
    lane: 0,
    // These columns are this function's ordering, not a hand-off any shape declared — including
    // for a node only the stream has seen, which arrives here with nothing said about it. What
    // is drawn between them depends on knowing that: see `handoffDrawn`.
    shaped: false,
  }));
  return [...placed, ...extra];
}

/**
 * One node's row: what the server said about it, with every per-moment fact taken from the stream.
 *
 * State and checkpoint counts always come from the stream — those it can always prove. Cost only
 * when it actually carries it; otherwise the store's figures stand, which are true of where the run
 * ENDED and are marked as such by being all a rotated-away log can offer. A node the stream does not
 * mention has NOT STARTED at this point: it keeps what it is CONFIGURED to run on (`planned`),
 * because that is true before it runs, and loses everything it has not yet done.
 *
 * One thing the stream cannot prove on its own, and it is the one that matters most: a node STOPPING
 * when the host dies under it. That writes no checkpoint and no node-scoped event — the only records
 * are the run's status and a `run_failed` carrying `node: null` — so the fold leaves the dying node
 * "working" and the run will emit nothing more to move it. `ended` is the run having stopped with
 * the view at its live end, and there a node the stream still calls working is finished by two
 * facts it cannot see itself: `died`, which `applyDerived` sets on the one node a failed run's
 * record names, and otherwise the server's state, which is the best the store can say.
 *
 * At every other position the stream stays the authority, `ended` being null. Scrubbed into the
 * middle of a run, a node that genuinely WAS working then must still read working — showing how it
 * ended is the same lie as showing a run's final state at step one.
 */
function fromStream(
  n: NodeView,
  d: DerivedNode | undefined,
  ended: RunStatus | null,
  died: boolean,
): NodeView {
  if (!d) {
    const { telemetry: _dropped, ...rest } = n;
    return { ...rest, state: "idle" as NodeState, checkpoints: 0 };
  }
  const telemetry = d.costed ? d.telemetry : (n.telemetry ?? d.telemetry);
  const settled: NodeState = died ? "failed" : n.state;
  return {
    ...n,
    state: ended && d.state === "working" ? settled : d.state,
    checkpoints: d.checkpoints,
    ...(telemetry ? { telemetry } : {}),
  };
}

/** A box for a name only the stream knows: the server has nothing to say about it yet. */
function unrun(name: string): NodeView {
  return { name, state: "idle", checkpoints: 0, stage: 0, lane: 0 };
}

/**
 * The one name in the stream that is not a node.
 *
 * `issue` records what the run was asked to do, not a stage of doing it. The server leaves it out
 * of the shape (`pipeline::ISSUE_NODE`) and this list has to agree, or an unshaped run draws a box
 * for its own brief.
 */
const ISSUE_NODE = "issue";

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
