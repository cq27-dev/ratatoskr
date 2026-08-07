# AGENTS.md

Guidance for coding agents working in the `ratatoskr` repository.
(`CLAUDE.md` is a symlink to this file.)

## Prefer the rag-rat MCP for code browsing

This repo is indexed by `rag-rat` — a local repo-intelligence index and MCP server. **Heavily prefer
the `rag-rat` MCP tools over raw `grep`/`cat`/file reads when browsing or understanding code.** One
MCP call returns more context, faster, than a shell sweep, and it surfaces *drive-by repo memories*
(source-anchored invariants, decisions, risks) attached to the code you're touching — context you
would otherwise never see.

Reach for these first:

- **`semantic_search`** — "where is this concept implemented?" Returns current source chunks with
  inline graph (callers/callees), git, and GitHub papertrail, all validated against current source.
- **`symbol_lookup`** — exact/fuzzy symbol resolution, with any bound memories attached.
- **`impact_surface`** — the coding preflight before editing a symbol: graph callers/callees, tests,
  git history, papertrail, and **repo memories** crossing the call path. Run it before changing
  anything non-trivial.
- **`find_callers` / `trace_callees`** — reverse/forward graph traversal instead of grepping for
  call sites.
- **`read_chunk`** — current text for a chunk with anchor validation + graph + memories.
- **`repo_brief` / `repo_clusters`** — orientation (spine, churn, god-modules, ownership clusters).
- **`important_symbols`** — load-bearing symbols by (SCIP-aware) PageRank; pass `personalize` to
  bias toward what you're editing.

**Symbol handle:** symbol-returning tools emit `id`, an opaque `sym_<hex>` token — the stable handle
to cache and pass back into graph/impact/memory tools as the `id` param (copy verbatim; never parse
it as a number). Use `ref` (the `path::name` qualified name) for the human-readable identity.

Use the MCP to *find and understand*; use the file tools to *change* (and to confirm exact text
before an `Edit`). The MCP server is read-only on source — it never edits files. If it returns empty
results, the index may be stale: `rag-rat index --discover` then `rag-rat reconcile`.

## Record durable learnings as rag-rat memories

**This is required, not optional.** When you discover something durable and non-obvious — a
load-bearing invariant, a decision + its rationale, a risk/footgun that cost you time, a perf
characteristic, a "do not do X because Y" — record it with `memory_create` **before you finish the
task**. If you had to read three files and reason for ten minutes to learn it, the next agent should
get it in one MCP call.

**Why rag-rat and not your own notes:** rag-rat memories live in this repo's shared index, so they
surface for **every** agent that uses the rag-rat MCP — not just the one that wrote them. An agent's
private/session memory is invisible to the others. rag-rat is the **cross-agent memory layer**.

How to do it well:
- **Anchor to the tightest stable target.** Prefer an `id` binding (the `sym_<hex>` logical-symbol
  handle — self-heals across cross-file moves); fall back to a `path` binding for file/area notes, or
  a commit/GitHub ref for historical rationale.
- **Pick the right `kind`:** `Invariant` (must stay true), `Decision`/`RejectedAlternative` (why it's
  this way / why not the other), `Risk`/`BugPattern` (footguns), `PerformanceNote`, `PlatformQuirk`,
  `FFIBoundary`. Concrete title, body with the *why* and *how to apply* — not just *what*.
- **Write the present tense, not a changelog.** A memory says what is true NOW and what to do about
  it. "Fixed in #123", "used to fail open" are unactionable: they go stale on the next change. When
  updating a memory whose warning no longer applies, rewrite the body to state the rule that now
  holds — don't append a status section — and `memory_mark_obsolete` it if nothing actionable
  survives.
- **`memory_search` first** to avoid duplicates; **`memory_update` / `memory_mark_obsolete`** when a
  memory is wrong or superseded.

## Public artifacts describe the change, not the process that produced it

PR descriptions, commit messages, and issue text are read by humans reviewing the **change**. Keep
them about the code — the problem, the fix, the rationale, how it was verified. Do **not** narrate
the process that produced them:

- No agent/subagent counts or fan-out language, no multi-agent / workflow / orchestration framing,
  and no internal phase or round codes.
- No naming of AI assistants or AI review tooling, and no review play-by-play ("caught in review and
  fixed before merge"). State the final behavior; if a concern shaped the design, explain the concern
  on its own terms.
- No trailing session links or assistant-attribution trailers (`Claude-Session:`, `Co-Authored-By` an
  assistant, etc.).

A reviewer should not be able to tell from a PR whether one agent or twenty produced it — only what
changed and why. Durable, checkable references are encouraged: issue/PR numbers, commit SHAs, test
names, file paths. (This is about *public* artifacts; rag-rat memories are the internal cross-agent
layer and may record provenance freely.)

The same applies to **files**: plans, design notes, and working scratch do not get committed. They
describe the process, they go stale the moment the feature lands, and the repo then carries a
document that contradicts the code. Put the durable part where it will be read — a comment at the
constraint, a rag-rat memory, or the issue/PR — and let the plan itself be disposable. Documentation
that a *user* needs (the README, a crate's README) is a different thing and belongs in the repo.

Commit messages use conventional-commit form: `type(scope): summary`.

## Repo orientation

Rust workspace (2024 edition), crates in a layered DAG under `crates/`:

- `ratatoskr-core` — domain types shared everywhere: config shape, run state, and the `ToolPolicy`
  seam. Deliberately has **no** async-runtime dependency.
- `ratatoskr-graph` — the graph vocabulary: the `Node` trait, `NodeError`, and `parse_validated`
  (the schema gate for node output).
- `ratatoskr-mcp` — client to rag-rat's MCP server over a stdio subprocess; hands back the tool list
  plus a cloneable `ServerSink`.
- `ratatoskr-agent` — builds a `rig` agent bound to a model + rag-rat's MCP tools; `run` /
  `run_structured`; the `RulesetHook` per-tool-call gate.
- `ratatoskr-script` — TypeScript rulesets: transpile (`swc`) + evaluate (`rquickjs`)
  `.ratatoskr/rules/*.ts` into a per-node `ToolPolicy` plus static config. See the invariants memory
  on `ScriptEngine`.
- `ratatoskr-nodes` — the concrete nodes (scout, memory, analyst, red-team, implementer, bookkeeper)
  and the `run_plan` / `run_full` orchestration.
- `ratatoskr-exec` — execution primitives for the fork: isolated git worktrees, sandboxed command
  runs (microsandbox / bwrap+Landlock), and the ACP client that drives a coding CLI.
- `ratatoskr-store` — the checkpoint store: a single SQLite file, one writer by construction. Also
  the instance's identity database (`auth.rs`), a *separate* file: `serve` writes sessions to that
  one and still never writes to a project's store.
- `ratatoskr-cli` — the `ratatoskr` binary (`init`, `ask`, `plan`, `run`, `bookkeep`, `status`).

`ratatoskr.toml` (repo root, gitignored) configures models, rag-rat launch, sandbox, and store;
`ratatoskr.toml.example` is the committed template. `.ratatoskr/` holds runtime state (logs, store,
worktrees) and is gitignored — except `.ratatoskr/rules/`, which is version-controlled.

## Worktree correctness

Changes to `ratatoskr-exec` worktree/sandbox behavior must keep the main checkout and linked
worktrees working against the same store, and should carry a regression test for the behavior they
touch. Live-run integration gotchas (ACP absolute cwd, permission-option selection, bwrap
mount-in-place) are recorded as rag-rat memories — check them before editing that path.

## Style

Parameter structs over long argument trains; `{self, ..}` imports for mixed lists; injected time and
IDs rather than ambient calls; closed/persisted enums behind stable string tokens (this repo uses
`strum` for `RunStatus`). Keep SQL in helpers named for the domain question, with invariant comments
and tests. The workspace sets `warnings = "deny"`, so a warning fails the build.

## Build / test

```bash
cargo build
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check   # CI uses stable rustfmt
```

CI (`.github/workflows/ci.yml`) runs fmt, clippy (`-D warnings`), and the test suite; all three must
be green. Run them locally before pushing — clippy and fmt failures redden CI just as test failures
do.
