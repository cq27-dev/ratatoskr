import type {
  LiveEvent,
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
  /**
   * Whether this view saw the invocation being shown START.
   *
   * Distinct from [`costed`], and the distinction matters at exactly one moment: a fresh attempt
   * has reported no cost yet, and treating that as "the stream cannot say" hands the box back the
   * server's telemetry — which is the run's FINAL state, so a second attempt renders holding the
   * first one's model, tokens and duration.
   */
  started: boolean;
  /**
   * Whether a Stop or a Steer aimed at this box would reach what is running in it.
   *
   * False for a box whose live work is a clarification answerer: that turn runs on the ASKING
   * node's control, so nothing addressed here is polled.
   */
  controllable: boolean;
  /**
   * What each of this box's MEMBER stages is doing, keyed by stage id — the box's own aggregate
   * excluded, since the box is not a pip of itself.
   *
   * From the same per-invocation bookkeeping the box state is folded from, which is what keeps a
   * substage lighting at the point in the run where it actually ran and staying correct while
   * scrubbing. Present only when a member has recorded; which stages a box is DECLARED to hold is
   * the registry's answer, and a stage the shape never assigned to this box must not appear in it.
   */
  memberStates?: Map<string, NodeState>;
  /**
   * The box whose execution this one's invocations ran INSIDE, when the stream shows one.
   *
   * Resolved by walking an invocation's span parentage outward to the nearest span some other
   * box's execution opened — through host-call spans, which belong to no box, without stopping.
   *
   * Three states, and the difference between the empty two is load-bearing. ABSENT: the stream
   * names no box — every chain reaches the run itself, the referee's shape — and a caller from a
   * durable record may stand in. NULL: the stream REFUSES one — two invocations resolved
   * different callers, or one resolved a box while another ran at the root — and that refusal is
   * evidence, so nothing else may re-anchor what the stream contradicted. Collapsing the two let
   * a persisted caller re-anchor a box whose complete history had explicitly refused it.
   */
  caller?: string | null;
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
 * What to file a record under.
 *
 * The producer's `span_id` when it recorded one. Otherwise a key standing for "the one attempt in
 * flight": a stream from before executions had identities, or a record written outside every
 * execution, still has to fold — and folding it into one attempt is what the fold did for all of
 * them before.
 */
function keyOf(e: BoxedEvent, member: string): string {
  return e.span_id ?? `unidentified:${member}`;
}

/** What every fold needs of an attempt, whatever else it keeps on one. */
interface Tracked {
  span: string;
  /** Started and not yet ended. Only a `node_start` opens a live one. */
  live: boolean;
  /** Opened by a `node_start` in this view, rather than inferred from a record about it. */
  started: boolean;
  /**
   * Whether a Stop or a Steer aimed at the box this is drawn in would reach it.
   *
   * A clarification answerer runs on the ASKING node's control — a Stop during one ends the asking
   * turn — so nothing addressed to the answerer's own box is ever polled. Offering it hands an
   * operator a button that does nothing, which is worse than offering none.
   */
  controllable: boolean;
  /**
   * The execution said it was over — its own checkpoint, or its `span_end`.
   *
   * Distinct from `!live`, which an attempt inferred from a record about an invocation is born as.
   * This is positive evidence that the work finished, and it is what tells a box whose members have
   * all ENDED from one that has merely been entered.
   */
  ended: boolean;
}

/**
 * One member's invocations: indexed by identity, in order, with the live ones to hand.
 *
 * Indexed because a stage may be invoked any number of times — an imported history has no bound on
 * it — and searching a growing list per record is quadratic in that number. The live list is short
 * by construction: it holds only what is running at once.
 */
function attempts<A extends Tracked>() {
  const list: A[] = [];
  const byId = new Map<string, A>();
  /**
   * The live ones, newest last — a stack that may hold ended entries, discarded when looked at.
   *
   * Removing from the middle of an array costs a scan and a shift, which is quadratic where a
   * history holds many overlapping invocations: N starts followed by N ends walk the whole live set
   * N times. Here an entry is pushed once and popped once, so a lookup pays only for what it
   * discards and never for what is still running.
   */
  const stack: A[] = [];
  const add = (made: A) => {
    list.push(made);
    byId.set(made.span, made);
    if (made.live) stack.push(made);
    return made;
  };
  const newestLive = () => {
    while (stack.length > 0 && !stack[stack.length - 1]!.live) stack.pop();
    return stack.at(-1);
  };
  const end = (attempt: A) => {
    attempt.live = false;
  };
  return {
    list,
    add,
    end,
    of: (span: string) => byId.get(span),
    /**
     * The attempt a record belongs to.
     *
     * With an identity, the attempt that IS it — and a fresh one if this view never saw it start,
     * since an ingested log may begin mid-run and filing the record under some other attempt would
     * charge that attempt for a turn it did not run. That fresh one is NOT live: a record about an
     * invocation is not evidence that it is still running, and an attempt nothing can ever close
     * leaves its box working for the rest of the run.
     *
     * Without one, the newest invocation still live, else the newest seen. Not the oldest: a record
     * that cannot say which invocation it belongs to belongs to the one in flight, and a re-entry
     * that announced no identity has to restart the counts rather than add to the attempt before
     * it.
     */
    for: (e: BoxedEvent, span: string, make: (live: boolean) => A): A => {
      if (e.span_id) return byId.get(span) ?? add(make(false));
      return newestLive() ?? list.at(-1) ?? add(make(false));
    },
    /**
     * The invocation a viewer is looking at: the newest still live, else the newest seen.
     *
     * Newest LIVE first, because two invocations may overlap and the one that finished first is not
     * what the box is doing. Taking the newest outright drew a finished sibling's model and tools
     * while the one still running went unseen.
     */
    current: (): A | undefined => newestLive() ?? list.at(-1),
  };
}
type Attempts<A extends Tracked> = ReturnType<typeof attempts<A>>;

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
/**
 * Span parentage and ownership, shared by every fold that resolves callers.
 *
 * One implementation, because two folds keeping their own copies of these rules is how they would
 * come to disagree about one stream. Parentage from EVERY record that names both ids — including
 * host and exchange lifecycle records, which node loops never reach — because a nested node's
 * chain to its caller runs THROUGH the host call that drove it. Ownership ONLY from a
 * `node_start`: any other record's `node` labels the row it produced, not the execution that
 * opened the span it rides on — a box's aggregate lands on its host call's span, a
 * clarification's row on the exchange span. The walk is hop-capped because parentage is
 * producer-supplied data, and a cycle in it must cost a bounded walk rather than hang the render.
 */
function spanIndex() {
  const parent = new Map<string, string>();
  const box = new Map<string, string>();
  return {
    /** Capture what this record proves. First mention wins for both maps: a producer states an
     *  execution's parentage when it opens it, and a later record contradicting it is not
     *  evidence the tree changed. */
    note(e: BoxedEvent | LiveEvent): void {
      if (e.span_id && e.parent_span_id && !parent.has(e.span_id)) {
        parent.set(e.span_id, e.parent_span_id);
      }
      if (e.kind === "node_start" && e.span_id && e.node && !box.has(e.span_id)) {
        box.set(e.span_id, e.node);
      }
    },
    /** The box that opened exactly this span, with no walk — `undefined` for a span no
     *  node execution announced. */
    boxOf(span: string): string | undefined {
      return box.get(span);
    },
    /** The recorded parent of exactly this span, with no walk. */
    parentOf(span: string): string | undefined {
      return parent.get(span);
    },
    /** The box owning the nearest node-execution span at or above `from`; `undefined` when the
     *  chain reaches the run itself, or data this view never saw. */
    owner(from: string | undefined): string | undefined {
      let span = from;
      for (let hop = 0; span !== undefined && hop < 64; hop += 1) {
        const found = box.get(span);
        if (found !== undefined) return found;
        span = parent.get(span);
      }
      return undefined;
    },
  };
}

/** One traversal of the graph: the box that just became active, and what handed off to it. */
export interface Transition {
  /** `null` for the run's very first start, which nothing handed off to. */
  from: string | null;
  to: string;
  at: string;
}

/**
 * Every hand-off the stream shows, in order — the last one at or before the cursor is the edge
 * that just lit.
 *
 * A transition is a box BECOMING active: a `node_start` for a box that is not already the active
 * one. Member churn inside a box is not a transition — the box was active throughout — and a box
 * is over when its OWN record lands, which is what lets the converge self-loop read as
 * implementer → implementer rather than nothing.
 *
 * `from` is provenance first and adjacency last: the invocation's STATED caller, else the box its
 * parentage resolves to, else the box most recently settled — the least wrong claim available,
 * exactly as the trailing columns' ordering is — else whatever was active. A resolution to the
 * box itself is internal structure and falls through to the fallbacks, which is also what draws
 * the self-loop.
 */
export function transitions(events: readonly BoxedEvent[]): Transition[] {
  const index = spanIndex();
  const out: Transition[] = [];
  /** The box most recently transitioned into — the `from` fallback of last resort. */
  let active: string | null = null;
  let settled: string | null = null;
  /**
   * Each box's CYCLE: opened by its first start, closed when its own completion has landed and
   * nothing of it is still live. The cycle is what a transition is about — a start inside an
   * open cycle is more work in a box execution never left, whether that is member churn, a
   * concurrent second invocation, or a peer beside a box that already wrote its own row.
   * Comparing against a single last-started box instead treated A's second concurrent start as
   * a new activation and pulsed a B → A hand-off that never happened.
   */
  const open = new Set<string>();
  /** Live invocations per box, keyed as the attempts are keyed. */
  const live = new Map<string, Set<string>>();
  /**
   * Whether the box's own completion has landed this cycle: its OWN checkpoint, or the lifecycle
   * end of an invocation that IS the box — an evidence-only stage or an answerer writes no
   * checkpoint, and a cycle nothing can close would swallow every later hand-off into that box.
   */
  const ownDone = new Set<string>();
  /** Spans opened by a box's OWN invocation (`member === node`), whose lifecycle end completes
   *  the cycle the way the box's own checkpoint does. */
  const ownSpan = new Set<string>();
  const closeIfOver = (name: string) => {
    if (open.has(name) && ownDone.has(name) && (live.get(name)?.size ?? 0) === 0) {
      open.delete(name);
      if (active === name) active = null;
    }
  };
  /**
   * The boundary rule, exactly as the node fold settles a turn: a turn ENDING is not the box
   * finishing. An ordinary stage's execution ends before its host validates and writes the
   * checkpoint, and completing the box on the raw end let a peer starting in that window read as
   * a fresh transition. A checkpoint-free box completes when its own turn ended COMPLETED and
   * the execution that INVOKED it has completed too — either order of arrival — and a turn with
   * no recorded parent answers to nobody, so its own completed end is its boundary.
   */
  const completedSpans = new Set<string>();
  const awaiting = new Map<string, { span: string; box: string }[]>();
  const finish = (box: string) => {
    ownDone.add(box);
    settled = box;
    closeIfOver(box);
  };
  for (const e of events) {
    index.note(e);
    // An execution ending closes its invocation wherever it is — the only closer for one that
    // writes no checkpoint. `boxOf` is a direct lookup: only a span this view saw a node open.
    if (e.kind === "span_end" && e.span_id) {
      if (e.outcome === "completed") {
        completedSpans.add(e.span_id);
        for (const waited of awaiting.get(e.span_id) ?? []) finish(waited.box);
        awaiting.delete(e.span_id);
      }
      const owner = index.boxOf(e.span_id);
      if (owner !== undefined) {
        live.get(owner)?.delete(e.span_id);
        if (ownSpan.has(e.span_id) && e.outcome === "completed") {
          const boundary = index.parentOf(e.span_id);
          if (boundary === undefined || completedSpans.has(boundary)) {
            finish(owner);
          } else {
            const queue = awaiting.get(boundary);
            if (queue) queue.push({ span: e.span_id, box: owner });
            else awaiting.set(boundary, [{ span: e.span_id, box: owner }]);
          }
        }
        closeIfOver(owner);
      }
    }
    if (!e.node) continue;
    const key = e.span_id ?? `unidentified:${e.member ?? e.node}`;
    if (e.kind === "node_start") {
      if ((e.member ?? e.node) === e.node && e.span_id) ownSpan.add(e.span_id);
      const running = live.get(e.node);
      if (running) running.add(key);
      else live.set(e.node, new Set([key]));
      if (!open.has(e.node)) {
        open.add(e.node);
        // The latch is a NEW cycle's to earn — reset here and only here. Resetting it on every
        // start let a peer's mid-cycle start erase a completion already recorded, and a cycle
        // whose latch kept vanishing could never close: the next genuine re-entry was then
        // swallowed instead of drawing its self-loop.
        ownDone.delete(e.node);
        const resolved = e.caller ?? index.owner(e.parent_span_id ?? undefined);
        const from = (resolved !== e.node ? resolved : undefined) ?? settled ?? active;
        out.push({ from, to: e.node, at: e.at });
        active = e.node;
      }
      continue;
    }
    // Each checkpoint ends ONE invocation of the box. The box's OWN record is a completion —
    // a member finishing mid-box is not — and even a completed box stays open until nothing of
    // it is live.
    if (e.kind === "checkpoint") {
      live.get(e.node)?.delete(key);
      if ((e.member ?? e.node) === e.node) {
        settled = e.node;
        ownDone.add(e.node);
      }
      closeIfOver(e.node);
    }
  }
  return out;
}

export function nodesFromEvents(events: readonly BoxedEvent[]): Map<string, DerivedNode> {
  /**
   * One invocation of one member: what it ran on, what it spent, and whether it is still going.
   *
   * Per invocation rather than per member, because a name never identified one. A stage is invoked
   * once per converge pass and may be invoked concurrently, so a second attempt used to open on the
   * first one's model, tokens and duration — and scrubbing back showed a later attempt's figures
   * against an earlier moment. Each invocation now carries its own, and the member is whichever of
   * them is current.
   */
  interface Attempt extends Tracked {
    state: NodeState;
    /** How this execution's own end said it went, once it has. */
    outcome?: string | undefined;
    /** The execution that invoked this one, where a record has said. */
    parent?: string;
    /**
     * The box this invocation's work is FOR, stated by its caller on the `node_start`.
     *
     * Producer provenance, above the parentage walk: a run-driven judgement's chain honestly
     * reaches no box, yet its caller is known exactly at the call site and stated there.
     */
    stated?: string;
    telemetry?: NodeTelemetry;
    cycles: number;
    used: Set<string>;
    costed: boolean;
  }
  type Member = Attempts<Attempt>;
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
    const made: Member = attempts<Attempt>();
    box.members.set(name, made);
    return made;
  };
  /**
   * Every attempt in the run by identity, so an execution's own end can close it.
   *
   * A `span_end` names an execution and no node — deliberately, since a host call is an execution
   * the shape cannot place — so it is matched here rather than through a member. It is what closes
   * an invocation that writes no checkpoint of its own: an answerer's turn, a stage whose evidence
   * is its only output, a turn whose failure the workflow recovered from.
   */
  const everywhere = new Map<string, { member: Member; attempt: Attempt }>();
  /**
   * Executions whose COMPLETED end has been seen, and the attempts each one invoked.
   *
   * A turn ending is not the stage finishing: a declared stage validates, normalises and
   * checkpoints AFTER its model turn returns, and any of those can still fail it. The host call is
   * the boundary that closes only once all of that has happened — so a turn settles as done when
   * its own end said "completed" AND the execution that invoked it says the boundary was reached.
   * Either may be seen first; the settle runs at whichever arrives second.
   */
  const completedEnds = new Set<string>();
  const childrenOf = new Map<string, Attempt[]>();
  const settled = (attempt: Attempt) => {
    if (
      attempt.outcome === "completed" &&
      attempt.state === "working" &&
      attempt.parent !== undefined &&
      completedEnds.has(attempt.parent)
    ) {
      attempt.state = "done";
    }
  };
  /**
   * Apply an execution's own end to its attempt, however the two met.
   *
   * An abandoned answerer resolves QUIETLY: its turn was cancelled because the asking node was
   * stopped, the failure story is the asker's, and a box left "working" by it reads as live
   * forever — and stands as a stale second candidate that blocks a later failure's attribution.
   * Idle rather than done, because nothing finished; the box simply has nothing of its own to
   * show any more.
   */
  const applyEnd = (member: Member, attempt: Attempt, outcome: string | undefined) => {
    member.end(attempt);
    attempt.ended = true;
    attempt.outcome = outcome;
    if ((outcome === "cancelled" || outcome === "unvalidated") && !attempt.controllable) {
      attempt.state = "idle";
    }
    settled(attempt);
  };
  /**
   * Executions whose end arrived before anything that names them.
   *
   * The guard that emits a `span_end` drops as the execution leaves, which is BEFORE its caller
   * writes the checkpoint — so in an imported tail whose `node_start` was rotated away, the end is
   * the first record of that execution this view sees. Dropped, the checkpoint that follows creates
   * an attempt that never ended, and a box with no aggregate of its own works forever.
   */
  const endedEarly = new Map<string, string | undefined>();
  const index = spanIndex();

  const make = (span: string, live: boolean, started: boolean): Attempt => ({
    span,
    live,
    started,
    ended: false,
    controllable: true,
    state: "working",
    cycles: 0,
    used: new Set(),
    costed: false,
  });
  const open = (member: Member, span: string): Attempt => {
    const made = member.add(make(span, true, true));
    everywhere.set(span, { member, attempt: made });
    return made;
  };
  const filed = (member: Member, e: BoxedEvent, attempt: Attempt): Attempt => {
    // Wherever the address arrives, not only on a start. Every record of a turn carries it, off the
    // span the turn opened — so an attempt reconstructed from a tail whose start has scrolled away
    // still knows a Stop aimed here would not reach it.
    if (e.controlled_as !== undefined) attempt.controllable = e.controlled_as === e.node;
    if (!everywhere.has(attempt.span)) everywhere.set(attempt.span, { member, attempt });
    if (attempt.parent === undefined && e.parent_span_id) {
      attempt.parent = e.parent_span_id;
      // Appended, never rebuilt: a spread here copies every sibling already registered for each
      // new one, which is quadratic in a host's fan-out — the third time this exact shape has
      // appeared in this file, and the fold runs on every render and every scrub.
      const siblings = childrenOf.get(e.parent_span_id);
      if (siblings) siblings.push(attempt);
      else childrenOf.set(e.parent_span_id, [attempt]);
      settled(attempt);
    }
    // An end this view saw before it saw anything else of that execution — applied with the
    // outcome it carried, because arriving first does not change what it said.
    if (endedEarly.has(attempt.span) && !attempt.ended) {
      applyEnd(member, attempt, endedEarly.get(attempt.span));
    }
    return attempt;
  };
  const attemptFor = (member: Member, e: BoxedEvent, span: string): Attempt =>
    filed(member, e, member.for(e, span, (live) => make(span, live, false)));

  for (const e of events) {
    index.note(e);
    // An execution's own end, which names no node. It closes the invocation it names wherever that
    // is — the only thing that can, for an invocation that writes no checkpoint.
    if (e.kind === "span_end" && e.span_id) {
      if (e.outcome === "completed") {
        completedEnds.add(e.span_id);
        for (const child of childrenOf.get(e.span_id) ?? []) settled(child);
      }
      const found = everywhere.get(e.span_id);
      if (!found) {
        endedEarly.set(e.span_id, e.outcome);
        continue;
      }
      applyEnd(found.member, found.attempt, e.outcome);
      continue;
    }
    if (!e.node) continue;
    const box = at(e.node);
    const name = e.member ?? e.node;
    const own = name === e.node;
    const member = memberOf(box, name);
    const span = keyOf(e, name);

    if (e.kind === "node_start") {
      // A fresh invocation of THIS member. Its own counts, its own cost, its own model: nothing of
      // the attempt before it carries over, which is what a name-keyed fold could not express. Its
      // siblings stand, and so do the checkpoints already recorded — the implementer is re-driven
      // once per converge iteration.
      const before = member.current()?.telemetry;
      const attempt = open(member, span);
      // Where a control for this turn is addressed. Stated only when it is not the node itself, and
      // the box a member is drawn in is its own address — so a member whose control goes somewhere
      // else entirely is one nothing here can reach.
      attempt.controllable = (e.controlled_as ?? e.node) === e.node;
      if (e.caller !== undefined) attempt.stated = e.caller;
      filed(member, e, attempt);
      // What it RAN ON carries across a re-entry; what it SPENT does not. The model, its tools and
      // its session are configuration — a start that announces nothing has not changed them — while
      // tokens, turns and duration belong to the attempt that spent them, which is the whole reason
      // this fold is keyed per invocation.
      const ran_on = e.facts ?? (before ? { ...before } : undefined);
      // Only when there is something to say. A start that announces nothing, with nothing announced
      // before it, leaves the attempt with no telemetry at all — a blank one would report zero
      // tokens and no duration, which is a measurement rather than the absence of one.
      if (ran_on) {
        attempt.telemetry = {
          ...blank(),
          model: ran_on.model,
          tools: ran_on.tools,
          thinking: ran_on.thinking,
          reuses_session: ran_on.reuses_session,
        };
      }
      continue;
    }

    if (e.kind === "checkpoint") {
      // This record ends ONE INVOCATION of the member, whoever it belongs to. What it means for the
      // BOX is decided below, from every member at once — a member finishing is never on its own
      // the box finished, and for a box that is itself a stage its own record is one member's too.
      //
      // An error is the only thing that tells a failed member from a finished one: both write a
      // checkpoint, and the fact of one proves only that the invocation stopped. The MEMBER keeps
      // its failure — an error on a checkpoint is the event contract's failure signal, and the
      // stage strip reads member states directly, so flattening it to done drew a failed substage
      // as completed — while the BOX still fails only on its own record: the box-state clause
      // below consults the box's own member alone, exactly as before.
      const attempt = attemptFor(member, e, span);
      member.end(attempt);
      attempt.state = e.error ? "failed" : "done";
      if (own) box.checkpoints += 1;
      attempt.telemetry = {
        ...(attempt.telemetry ?? blank()),
        first_at: attempt.telemetry?.first_at ?? e.at,
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
        turns: e.turns ?? attempt.telemetry?.turns ?? null,
        tools_used: [...attempt.used],
      };
      // Presence is the signal, the same as for a `usage` event below. A checkpoint covering no
      // turn — a box's own aggregate — carries no cost keys at all now, so the server reports no
      // usage for it rather than a zeroed one. The counters themselves are never the basis: an
      // endpoint may make a real call and report no accounting, and `reasoning_tokens` is
      // hardcoded to zero by one provider while `output_tokens` is under-reported. A zero that
      // arrives is a measurement; what a turn-less record has is nothing to read.
      if (e.usage) attempt.costed = true;
      continue;
    }

    if (e.kind === "usage" && e.usage) {
      // Unconditional, unlike the checkpoint above. A `usage` event is the endpoint's own report of
      // a turn: its presence is the authority and a zero is a measurement, not an absence. Doubting
      // it leaves the attempt uncosted, and `fromStream` then keeps the server's telemetry — the
      // run's FINAL state — so scrubbing back to an earlier attempt shows a later one's numbers.
      const attempt = attemptFor(member, e, span);
      attempt.costed = true;
      // All of it, not the counters alone. For a turn that writes no checkpoint — an answerer, an
      // evidence-only stage — this is the ONLY record of what it ran on and how many turns it took,
      // and a stream that carries a cost is authoritative: whatever this leaves out is displayed as
      // measured absence rather than filled in from anywhere else.
      attempt.telemetry = {
        ...(attempt.telemetry ?? blank()),
        input_tokens: e.usage.input_tokens,
        output_tokens: e.usage.output_tokens,
        cached_input_tokens: e.usage.cached_input_tokens,
        cache_creation_input_tokens: e.usage.cache_creation_input_tokens,
        reasoning_tokens: e.usage.reasoning_tokens,
        duration_ms: e.usage.duration_ms,
        turns: e.turns ?? attempt.telemetry?.turns ?? null,
        ...(e.facts
          ? {
              model: e.facts.model,
              tools: e.facts.tools,
              thinking: e.facts.thinking,
              reuses_session: e.facts.reuses_session,
            }
          : {}),
        tools_used: [
          ...new Set([
            ...(attempt.telemetry?.tools_used ?? []),
            ...(e.tools_used ?? []),
            ...attempt.used,
          ]),
        ],
      };
      // A turn that writes no checkpoint has only this record to say it FAILED, and its execution
      // ending says nothing about how. Without this a recovered failure — an answerer that could
      // not answer, an evidence-only turn that errored — reads as done the moment its span ends.
      if (e.error) attempt.state = "failed";
      continue;
    }

    if (e.kind === "tool_call") {
      const attempt = attemptFor(member, e, span);
      attempt.cycles += 1;
      // `detail` carries the tool name for this kind.
      if (e.detail) attempt.used.add(e.detail);
    }
    // Not over an execution that has announced its end: a record can arrive after it, and one that
    // means "this is running" does not make it run again.
    const seen = attemptFor(member, e, span);
    if (WORKING.has(e.kind) && !seen.ended) seen.state = "working";
  }

  /**
   * The box this one's invocations ran inside, when every chain that resolves agrees.
   *
   * From the invocation's PARENT outward — its own span proves only itself — through spans no box
   * owns, which is where host calls sit. A chain that reaches this box's own span stops without an
   * answer: a member nested inside its box's aggregate turn is the box's own structure, not a
   * caller. Hops are capped because parentage is producer-supplied data, and a cycle in it must
   * cost a bounded walk rather than hang the render.
   */
  // `undefined` when the stream names no box, `null` when it REFUSES one — a refusal is evidence
  // and must not read as silence, or a durable caller re-anchors what the stream contradicted.
  const callerOf = (name: string, box: { members: Map<string, Member> }): string | null | undefined => {
    let found: string | undefined;
    // An invocation whose chain reaches no box at all: driven by the workflow itself, or
    // unresolvable. That is a VOTE, not an abstention — a box invoked once at the root and once
    // inside another fits two placements, which is the same conflict as two different callers.
    // Alone it is not a refusal: every chain reaching the run is the stream having no box to
    // name, which is where a durable record's answer legitimately stands in.
    let rooted = false;
    for (const m of box.members.values()) {
      for (const a of m.list) {
        // Stated provenance supersedes this invocation's walk: a run-driven judgement's chain
        // honestly reaches no box, and reading that as a root VOTE against the statement would
        // refuse the one anchor the producer went out of its way to assert. Conflicts between
        // statements — or between a statement and another invocation's walk — still refuse.
        const owner = a.stated ?? index.owner(a.parent);
        // Nested under this box's own turn: internal structure, saying nothing about who called
        // the box — the only outcome that abstains.
        if (owner === name) continue;
        if (owner === undefined) {
          rooted = true;
          continue;
        }
        // Two invocations resolving different callers is an anchor that fits two histories,
        // which is an assertion about neither.
        if (found !== undefined && found !== owner) return null;
        found = owner;
      }
    }
    if (rooted && found !== undefined) return null;
    return found;
  };

  const out = new Map<string, DerivedNode>();
  for (const [name, box] of boxes) {
    const members = [...box.members.values()];
    const caller = callerOf(name, box);
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
    // A member is whichever of its invocations is current: the last one to start, which is the one
    // a viewer is looking at. Its earlier attempts are what they were and are not merged into it —
    // that merge is how a second attempt came to report the first one's model and tokens.
    const current = (m: Member) => m.current();
    const working = (m: Member) => m.list.some((a) => a.live) || current(m)?.state === "working";
    const state: NodeState = members.some(working)
      ? "working"
      : current(box.members.get(name) ?? attempts<Attempt>())?.state === "failed"
        ? "failed"
        : box.checkpoints > 0
          ? "done"
          : // No record of the box's own, and the box's OWN member has announced its end. That is
            // the stage that is its own node — an evidence-only one, or a turn whose failure the
            // workflow recovered from — which writes no aggregate ever, so a box waiting for one
            // works for the rest of the run with its Stop still offered.
            //
            // The box's own, not any member's, and two different things would go wrong otherwise.
            // A member's CHECKPOINT says the box STARTED, since a peer may still be to run. And a
            // composed member ENDING proves nothing about the box either: `implementer_attempt`
            // announces its end before the host that drove it writes the aggregate, so finishing
            // the box there drops its working state for the window in between.
            // The box's own member reached DONE without a record of the box's own: the stage that
            // is its own node, settled by the boundary that invoked it. Merely ENDED is not this —
            // a turn ends before its stage validates and checkpoints, and an end whose outcome the
            // boundary never confirmed is nobody's record, which keeps the node among a failed
            // run's candidates.
            box.members.get(name)?.current()?.state === "done"
            ? "done"
            : members.some((m) => current(m) && current(m)?.state !== "idle")
              ? "working"
              : "idle";
    const folded = members
      .map((m) => current(m)?.telemetry)
      .filter((t): t is NodeTelemetry => !!t)
      .reduce<NodeTelemetry | undefined>(
        (into, next) => (into ? fold(into, next) : next),
        undefined,
      );
    // What each MEMBER stage is doing, by the same rules its slice of the box state uses: working
    // while any invocation of it is live, else whatever its current attempt reached. The box's
    // own aggregate is the box, not a pip of itself.
    const memberStates = new Map<string, NodeState>();
    for (const [id, m] of box.members) {
      if (id === name) continue;
      memberStates.set(id, working(m) ? "working" : (current(m)?.state ?? "idle"));
    }
    out.set(name, {
      state,
      checkpoints: box.checkpoints,
      ...(memberStates.size ? { memberStates } : {}),
      ...(caller !== undefined ? { caller } : {}),
      ...(folded ? { telemetry: folded } : {}),
      cycles: members.reduce((n, m) => n + (current(m)?.cycles ?? 0), 0),
      used: new Set(members.flatMap((m) => [...(current(m)?.used ?? [])])),
      costed: members.some((m) => current(m)?.costed ?? false),
      // Whether the stream OPENED the invocation being shown, which is not the same question as
      // whether it has reported a cost yet. A started attempt that has spent nothing so far is this
      // view's answer and stands; one this view never saw start is a gap the server's record fills.
      started: members.some((m) => current(m)?.started ?? false),
      // Whether a control aimed here would reach ANY of what is running, not only the invocation
      // being displayed. A stage can be running an ordinary turn and answering a clarification at
      // once — the answerer controlled by the node that asked, the ordinary one by this box — and
      // reading the newest alone takes the Stop away from a turn that is still reachable.
      // Reachable if anything RUNNING here can be reached. Read per box rather than per member:
      // a box with a finished member and one live answerer would otherwise answer from the
      // finished one, and offer a Stop against a turn controlled by the node that asked.
      controllable: members.some((m) => m.list.some((a) => a.live))
        ? members.some((m) => m.list.some((a) => a.live && a.controllable))
        : // Nothing live in view. An attempt reconstructed from a trimmed tail may still be
          // running — its start is what scrolled away — so it stays reachable; one whose execution
          // has ENDED is known to be over, whatever state it kept for the failure ledger, and a
          // Stop offered against it reaches nothing.
          members.some((m) => {
            const shown = current(m);
            return shown !== undefined && !shown.ended && shown.controllable;
          }),
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
 * Whether this stream shows the red team handing the tree to the implementer — or `null` where it
 * cannot say.
 *
 * The hand-off happened exactly where `redTeam()` COMPLETED and an `implement()` called after it
 * DROVE work: a `node_start` under the implement call's span. Anything less proves less.
 * `implement_host` waits for a red team that was called first, rejects one still in flight, and
 * permits implementation where none was called at all — and a host announces itself before its
 * body runs, so a failing `redTeam()` a workflow catches, the `implement()` the guard then
 * rejects, and an `implement()` that fails in its own preparation before reaching the implementer
 * all leave starts behind with nothing handed to anyone.
 *
 * `null` for a stream that announces no hosts: one from before executions announced themselves,
 * which can prove nothing either way. A host-announcing stream that cannot show the completed
 * hand-off is a workflow that did not make it, and that is `false` — a custom workflow may
 * populate the same boxes through declared stages, and box state is exactly what this exists to
 * stop inferring from.
 */
/**
 * Whether the view ending at `shownAt` is a complete account from the run's beginning.
 *
 * The history is one read from the run's start, so a view that ends inside it — `shownAt` at or
 * before its last event — is complete no matter what the live buffer holds: a reconnect gap sits
 * AFTER everything such a view shows, and scrubbing back past the gap must not turn a definite
 * answer into a fallback. Only a view that extends into the buffer (`shownAt` past the history, or
 * `null` for the live end) needs the join checked: the buffer's first event at or before the
 * history's last means the bounded replay overlapped what history covers and no slice fell
 * between. Not a given — a reconnect clears the buffer while the history re-read is throttled, and
 * joining stale history to a fresh tail leaves a gap that reads as a complete account. Both states
 * self-heal, since the next history read advances its end past the replay's start.
 */
export function contiguous(
  history: readonly LiveEvent[] | null,
  buffer: readonly LiveEvent[],
  shownAt: string | null,
): boolean {
  if (!history?.length) return false;
  if (shownAt !== null && shownAt <= history[history.length - 1]!.at) return true;
  // The overlap is the TAIL's, and the replay's front is not the tail's front: `trim_replay`
  // preserves `question*` events ahead of the bounded tail — a run blocked on a human must show
  // its question however old — so a preserved question's timestamp proves nothing about where the
  // tail begins, and reading it as the buffer's start claimed continuity across a missing slice.
  // Skipping the leading questions errs the safe way only: a genuine tail that happens to open
  // with a question reads as a gap and falls back, never the reverse.
  const tail = buffer.find((e) => !e.kind.startsWith("question"));
  return tail === undefined || tail.at <= history[history.length - 1]!.at;
}

export function handoffEvidence(events: readonly BoxedEvent[]): boolean | null {
  let sawHosts = false;
  let redTeamCompleted = false;
  const hosts = new Map<string, string>();
  const implementCalls = new Set<string>();
  for (const e of events) {
    if (!e.span_id) continue;
    if (e.execution === "host") {
      if (e.kind === "span_start" && e.execution_name !== undefined) {
        sawHosts = true;
        hosts.set(e.span_id, e.execution_name);
        if (e.execution_name === "implement" && redTeamCompleted) implementCalls.add(e.span_id);
      }
      if (
        e.kind === "span_end" &&
        e.outcome === "completed" &&
        hosts.get(e.span_id) === "redTeam"
      ) {
        redTeamCompleted = true;
      }
      continue;
    }
    // The moment a tree is actually in the implementer's hands: work started UNDER the implement
    // call. The host's own start is not it — `implement()` can fail after announcing itself and
    // before driving anything, in its guard, its argument parsing or its worktree setup, handing
    // nothing to anyone. Whatever happens to the implementation afterwards, the hand-off this
    // start received is history.
    if (e.kind === "node_start" && e.parent_span_id && implementCalls.has(e.parent_span_id)) {
      return true;
    }
  }
  return sawHosts ? false : null;
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
export function convergeLoops(
  events: readonly BoxedEvent[],
  stages: readonly RunStage[],
): ConvergeLoops {
  const out: ConvergeLoops = { fix: 0, replan: 0, retry: 0 };
  let entered = false;
  let since = new Set<string>();
  // A host is either a Rust operation or a declared stage — `build_hosts_with_turn` refuses a
  // stage that takes an operation's name, so the two tables are disjoint by construction — and the
  // run's own registry says which. Derived per run rather than copied as a list of names, because
  // a copy is a second authority: a recovery operation added in Rust would have read here as a
  // known non-operation, and its real re-entries would have been silently discarded.
  const declared = new Set(stages.map((stage) => stage.id));
  // Every execution that has announced itself, so a start can say what DROVE it. The loop being
  // counted is the standard converge operation's, and the box name cannot say that: a custom stage
  // composed into the implementer box starts under the same box, and counting it displayed a retry
  // no operation ever ran.
  const drivers = new Map<string, { operation: boolean }>();

  for (const e of events) {
    if (e.kind === "span_start" && e.span_id) {
      drivers.set(e.span_id, {
        operation:
          e.execution === "host" &&
          e.execution_name !== undefined &&
          !declared.has(e.execution_name),
      });
      continue;
    }
    if (!e.node || e.kind !== "node_start") continue;
    if (e.node !== "implementer") {
      since.add(e.node);
      continue;
    }
    // Counted only where what drove it is the standard operation — or where the stream cannot say:
    // a start with no parentage, or a parent this view never saw announce itself, is a stream from
    // before executions had identities, and reads as it always did. A KNOWN non-operation driver —
    // a declared stage's host, a clarification exchange — is positive evidence this was not the
    // loop.
    const driver = e.parent_span_id ? drivers.get(e.parent_span_id) : undefined;
    if (driver !== undefined && !driver.operation) continue;
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
export function forkHandoff(
  nodes: readonly NodeView[],
  handoff: boolean | null,
): boolean {
  const started = (name: string) => nodes.find((n) => n.name === name && n.state !== "idle");
  const redTeam = started("redteam");
  const implementer = started("implementer");
  if (!redTeam || !implementer || redTeam.stage !== implementer.stage) return false;
  // Drawn from [`handoffEvidence`] where the stream can give it. `null` is a stream that cannot
  // say — one from before executions announced themselves, or a window that does not reach back to
  // the run's beginning, where an absent record proves nothing because it may simply have scrolled
  // out. Both fall back to the box inference, which is what every stream got before there was
  // evidence to read.
  return handoff ?? true;
}

/**
 * The hand-offs a run made across stages it never entered.
 *
 * An ordinary edge joins adjacent columns, so a run that skipped a stage draws a break exactly
 * where it made a hand-off: the no-code-change shortcut goes `analyst` straight to `publisher`, and
 * a run that never forks or never reviews does the same. Each pair here is one such jump — from the
 * last column that ran to the next column that ran, with nothing but unentered stages between.
 *
 * "Ran" is `state !== "idle"`, the same thing every other edge asks. A stage that has not run YET is
 * not a stage that was skipped, which is why a span needs a LATER column to have started before it
 * may be drawn: mid-flight, and while scrubbing back through a finished run, the columns ahead are
 * idle for the ordinary reason and nothing spans them. A stage that looked skipped and then runs
 * simply stops being spanned, because the fold is over the stream as it stands.
 *
 * Pairs, not a boolean, because where the edge goes is the renderer's business and which hand-offs
 * happened is this one's — the same split `forkHandoff` keeps.
 */
export function skippedSpans(nodes: readonly NodeView[]): { from: string; to: string }[] {
  const columns = new Map<number, NodeView[]>();
  for (const node of nodes) {
    const at = columns.get(node.stage);
    if (at) at.push(node);
    else columns.set(node.stage, [node]);
  }
  const ordered = [...columns.keys()].sort((a, b) => a - b);
  // `entered`, never `state`. They agree except where it matters most: at the terminal end of a
  // failed run, two uncheckpointed nodes in flight are blamed on neither, so both settle back to
  // `idle` while their `node_start` events prove the stage ran. Reading state there would invent a
  // span straight across it — the very hand-off that did not happen.
  const ran = (node: NodeView) => node.entered ?? node.state !== "idle";
  const columnRan = (stage: number) => (columns.get(stage) ?? []).some(ran);

  const spans: { from: string; to: string }[] = [];
  for (let i = 0; i < ordered.length; i += 1) {
    if (!columnRan(ordered[i]!)) continue;
    // The next column that ran. Everything between is unentered, or this is not a span.
    let next = i + 1;
    while (next < ordered.length && !columnRan(ordered[next]!)) next += 1;
    if (next >= ordered.length || next === i + 1) continue;
    // One edge per pair of boxes, the same relation an adjacent-column edge draws.
    for (const source of columns.get(ordered[i]!) ?? []) {
      if (!ran(source)) continue;
      for (const target of columns.get(ordered[next]!) ?? []) {
        if (!ran(target)) continue;
        if (!handoffDrawn(source, target)) continue;
        spans.push({ from: source.name, to: target.name });
      }
    }
  }
  return spans;
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
 * the order the stream first saw each node is the only ordering it has — the least wrong claim
 * available. But only where nothing better exists: an appended target that NAMES its caller has a
 * better claim, and the chain edge beside it asserts a hand-off the record contradicts. That
 * includes an adjacency the layout itself manufactured — an anchored node leaving the spine makes
 * the trailing columns either side of it neighbours, and in a chain X → A → B where A anchored
 * under X, the closed-up columns read X → B.
 */
export function handoffDrawn(source: NodeView, target: NodeView): boolean {
  if (target.shaped === false && target.caller) return false;
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
  // Working AND reachable. A clarification answerer is a live turn drawn in its own box whose
  // control belongs to the node that asked, so a Stop offered against it is a button that does
  // nothing — worse than offering none, because it reads as an option that was ignored.
  const active = [...nodesFromEvents(events)]
    .filter(([, node]) => node.state === "working" && node.controllable)
    .map(([name]) => name);
  // Anything the stream shows working is reachable, whether or not this view caught its start. A
  // viewer attaching to a long tool-heavy turn gets a tail whose `node_start` has already scrolled
  // out of it, and requiring one anywhere in view threw away a node the stream had reconstructed —
  // falling back to the store, which still says idle until that node checkpoints. The controls
  // disappeared for exactly as long as the turn was interesting.
  if (active.length > 0) return active;
  // Nothing running in view: the store's answer, which is where a viewer with no stream at all
  // starts from. A stream that shows a start and no live node is a run that has stopped, and that
  // has nothing for the store to add.
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
  // Whether the events behind `derived` are a complete account from the run's beginning — the
  // same fact `contiguous` answers for the hand-off. The stream's caller resolution rests on
  // having seen EVERY invocation of a box: a bounded tail may hide an earlier invocation from a
  // different caller, so its agreement proves nothing and no stream caller is read from it. The
  // server's `caller` is resolved from the durable record and stands either way. Defaulting to
  // incomplete errs toward a trailing column, which is honest; an anchor is an assertion.
  complete: boolean = false,
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
  const extra = order.map((name, i) => {
    // Who invoked it, with the stream's answer first: the stream watched THIS run's parentage,
    // while the server's `caller` mirrors a known call site — the referee — and covers nothing
    // else. The stream's answer only from a complete account: resolution rests on every
    // invocation agreeing, and a window that may have dropped one proves nothing by the
    // agreement of what remains. Its REFUSAL (`null`) is an answer too — the invocations
    // contradict each other — and the durable record must not re-anchor what the stream
    // contradicted; only genuine silence (`undefined`) lets the server's answer stand.
    const stream = complete ? derived.get(name)?.caller : undefined;
    const caller = stream === null ? undefined : (stream ?? unplaced.get(name)?.caller);
    // The persisted caller is stripped before the resolved one is re-added: it rides in on the
    // server row's spread, and leaving it there is exactly how a refusal would be undone.
    const { caller: _persisted, ...row } = fromStream(
      unplaced.get(name) ?? unrun(name),
      derived.get(name),
      ended,
      name === died,
    );
    return {
      ...row,
      stage: base + i,
      lane: 0,
      ...(caller !== undefined ? { caller } : {}),
      // These columns are this function's ordering, not a hand-off any shape declared — including
      // for a node only the stream has seen, which arrives here with nothing said about it. What
      // is drawn between them depends on knowing that: see `handoffDrawn`.
      shaped: false,
    };
  });
  return [...placed, ...extra];
}

/**
 * The box a dynamically-invoked node hangs off, when the drawn graph can honour the anchor.
 *
 * Only a node the shape does not place — a declared layout is reproduced exactly — and only off a
 * parent that holds a column of its own: one the shape placed, or one in a trailing column with no
 * caller of its own. One level deep, deliberately. A chain of dynamic calls anchors its first link
 * and leaves the rest in trailing columns, because the alternative — walking callers of callers —
 * has to answer for cycles and for a parent that moved, and a trailing column is honest where a
 * wrongly-hung box is an assertion.
 */
export function branchParent(
  n: NodeView,
  byName: ReadonlyMap<string, NodeView>,
): string | null {
  if (n.shaped !== false || !n.caller) return null;
  const parent = byName.get(n.caller);
  return parent && (parent.shaped !== false || !parent.caller) ? n.caller : null;
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
    return { ...rest, state: "idle" as NodeState, checkpoints: 0, entered: false };
  }
  // Whose numbers to show. The stream's, once it has either reported a cost or watched the current
  // invocation start — a fresh attempt that has spent nothing yet IS this view's answer, and
  // handing the box back the server's record there shows the run's FINAL state against a moment it
  // had not reached: a second attempt rendering the first one's model, tokens and duration.
  //
  // The server's only where the stream cannot answer at all: an ingested tail whose starts are in a
  // rotated file, or a run recorded before checkpoints carried telemetry.
  const shown = d.costed || d.started ? d.telemetry : (n.telemetry ?? d.telemetry);
  const settled: NodeState = died ? "failed" : n.state;
  // Dropped, not merely not-set. `n` is spread below, so leaving the key out keeps the SERVER's
  // telemetry — the run's final state — against an attempt this view watched start and knows
  // nothing about yet. A historical start that announced no facts is exactly that case: the stream
  // is authoritative and has nothing to say, and nothing is what it must show.
  const { telemetry: _stale, ...without } = n;
  return {
    ...without,
    state: ended && d.state === "working" ? settled : d.state,
    checkpoints: d.checkpoints,
    // What the STREAM saw, before the settling above can take it back. A failed run with two
    // uncheckpointed nodes in flight blames neither, so both render `idle` although their
    // `node_start` proves they ran — and a reader inferring that their stage was never entered
    // would assert a hand-off across a stage that did.
    entered: d.state !== "idle",
    ...(shown ? { telemetry: shown } : {}),
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
