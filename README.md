# Ratatoskr

An orchestrator for [rag-rat]-driven coding runs. A run is a graph of agent nodes that scout the
repository, analyse the impact, red-team the baseline, implement the change in an isolated worktree,
and iterate until the tests agree — checkpointing every step to a local SQLite store.

The point is the last node. On a finished run the **bookkeeper** writes what was learned back into
rag-rat's memory, so the *next* run's memory node surfaces it while planning. Runs compound instead
of starting from zero.

```
             ┌─────────── informs the next run ───────────┐
             ▼                                            │
scout ──→ memory ──→ analyst ─┬─→ red-team ─────┬─→ bookkeeper
                              └─→ implementer ──┘
                                    ↑        │
                                    └────────┘
                                     converge
```

## How a run works

1. **scout** searches the tracker papertrail and the code for what already exists.
2. **memory** corroborates from rag-rat's own repo memories — prior invariants, decisions, footguns.
3. **analyst** determines the blast radius and produces the requirements the change must satisfy.
4. **red-team ∥ implementer** run concurrently: red-team characterises the *baseline* test run in a
   sandbox, while the implementer drives a coding CLI (Claude Code over ACP) in a fresh git worktree.
5. **converge** re-runs the implementer with a diagnostic prompt until the change introduces no new
   failures (`converged`) or the iteration budget runs out (`max_iterations_reached`).
6. **bookkeeper** distils one durable learning and writes it to rag-rat via `memory_create`.

Every node's output is validated against its JSON Schema and checkpointed before the next node
runs, so a failure stops the run with `status = failed` attributed to the node that failed, and the
work up to that point is inspectable.

A node that gets stuck can **ask a peer** mid-run: the `ask` tool routes a question to another node
(the analyst answers anything unrouteable) and hands the answer back as the tool result, without
unwinding the asking node.

## Quickstart

Requires a repository already indexed by rag-rat, and `bwrap` for the default sandbox.

```sh
cargo build --workspace
cargo run -p ratatoskr-cli -- init          # writes ratatoskr.toml
export ANTHROPIC_API_KEY=...                # or MOONSHOT_API_KEY, per [models.*]
cargo run -p ratatoskr-cli -- run "Fix the flaky retry in the store"
```

`ratatoskr.toml` is gitignored; `ratatoskr.toml.example` is the committed template.

## Commands

| Command | What it does |
|---|---|
| `init` | Write a default `ratatoskr.toml`. |
| `ask <question>` | One agent answers a question about the repo, grounded in rag-rat's tools. |
| `plan <issue>` | scout → memory → analyst, printing a grounded plan. No code is changed. |
| `run <issue>` | Everything `plan` does, then fork, converge, and bookkeep. |
| `bookkeep <run-id>` | Replay just the bookkeeper against a stored run — no re-run. |
| `status <run-id>` | A run's status and every per-node checkpoint. Pure read: no rag-rat, no LLM. |
| `serve` | The observability dashboard (see below). |
| `clean` | Reclaim per-run worktrees and their `ratatoskr/*` branches. |

`plan` and `run` take the issue as an argument or `--file <path>` for long text, and `--json` for
the raw `RunState` instead of a summary. `clean` lists what it would remove unless given `--force`,
because removal discards each worktree's uncommitted changes.

## Dashboard

```sh
cargo run -p ratatoskr-cli -- serve          # http://127.0.0.1:7878
```

Watch runs as they happen: the pipeline graph with each node's state, the converge loop and its
iteration count, and every checkpoint's output. Runs can also be started from the browser.

It reads the same SQLite file a run writes to and never writes to it, so it is safe to leave open
against a live run. Runs started from the dashboard are spawned as child processes with an explicit
working directory, and are capped at one at a time by default (`--max-runs`).

A node that gets stuck can ask *you*: when you are watching a run, a question addressed to the user
appears in the dashboard and your answer goes back as that tool's result. Nobody watching, or nobody
answering in time, and the analyst answers instead — exactly as an unattended run behaves.

One dashboard can watch several projects. Each keeps its own store, worktrees, and logs; nothing is
merged, and a project switcher appears once there is more than one:

```sh
cargo run -p ratatoskr-cli -- serve --project ~/src/one --project ~/src/two
```

Bind it to loopback — the default — and keep it there. There is no auth, and it can start runs.

The UI is a separate build artifact and is optional: without it you still get the JSON API
(`/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/nodes/{node}`).

```sh
cd crates/ratatoskr-serve/web && bun install && bun run build
```

See the [web README](crates/ratatoskr-serve/web/README.md) for the dev-server workflow.

## Configuration

`ratatoskr.toml` covers model routing per node, how rag-rat's MCP server is launched, the sandbox
backend and test command, and the implementer's CLI and iteration budget. It is validated on load,
so a bad backend or an empty test command fails immediately rather than deep inside a run.

The sandbox backend is `landlock` (bubblewrap + Landlock — the default; no image, builds offline)
or `microsandbox` (a MicroVM, needs KVM). microsandbox sits behind a Cargo feature because its
build script downloads a helper binary, which fails in the network-less test sandbox; enable it
with `cargo build --features ratatoskr-exec/microsandbox`.

### Agent rulesets

`.ratatoskr/rules/<node>.ts` governs one node. A ruleset is authoritative where it speaks — a node
that declares a `model` needs no `[models.<node>]` entry at all, and that TOML entry becomes the
fallback:

```ts
defineAgent("scout", {
  model: { provider: "moonshot", model: "kimi-k2.5" },
  systemPrompt: "You are the scout...",     // replaces the node's built-in preamble
  tools: { allow: ["semantic_search"] },    // REPLACES the default tool set; `deny` also supported
  maxTurns: 40,
  onToolCall({ tool }) {                    // per-call gate, consulted for every tool call
    return tool === "papertrail_issue_search" ? "deny" : "allow";
  },
});
```

Rulesets apply to `scout`, `analyst`, `bookkeeper`, and `redteam` — the nodes that are LLM agents.
TypeScript is transpiled and evaluated in-process; the types are for editor ergonomics and are
stripped at load.

### Agent plugins

`.ratatoskr/plugins/` (and any path in `[plugins] paths`) holds plugins in the format coding CLIs
already use — a `.claude-plugin/plugin.json` manifest and an optional `hooks/hooks.json`. A
plugin's `SessionStart` hook runs once per run and its output is prefixed to each node's preamble,
so a node can open with the repository's shape instead of discovering it one tool call at a time.

Nothing a plugin *does* can fail a run: one that is missing, malformed, slow, or broken is logged
and skipped, and the node simply gets less context. Naming a plugin that isn't installed is
different — that is a typo, and it fails the run rather than silently binding less than you asked
for.

Installing a plugin is enough to use it: with no declaration anywhere, every discovered plugin
applies to every node. Rulesets *narrow* that, in the same place they govern a node's model and
tools:

```ts
// .ratatoskr/rules/_defaults.ts — every node inherits this
defineDefaults({ plugins: ["rag-rat"] });

// .ratatoskr/rules/analyst.ts — adjust what this one inherits
defineAgent("analyst", { plugins: { add: ["impact-lens"], remove: ["noisy"] } });

// or start from nothing, said out loud
defineAgent("scout", { plugins: { inherit: false, add: ["scout-only"] } });

// or name the set exactly
defineAgent("bookkeeper", { plugins: ["rag-rat"] });
```

Hooks still run once per run whichever way you bind them — each node composes its context from the
plugins it holds, so per-node binding costs nothing extra.

A plugin can also bring **tools**. Any stdio MCP server it declares (`mcpServers` in the manifest,
or a `.mcp.json` beside it) is connected once per run and offered to every node that binds the
plugin, alongside rag-rat's own catalogue. A node that names no tools of its own gets them for
free — binding the plugin is the statement. A node whose ruleset spells out `tools.allow` is a
different case: that list is exhaustive, so it must name the plugin's tools too (the run warns
when it doesn't). `deny` removes any of them, and an `onToolCall` gate sees them under the same
names.

Where two servers offer one name, the first one connected keeps it, so a plugin can never shadow a
rag-rat tool a node's prompt was written against; the collision is logged with both server names. A
server that will not start costs its plugin's tools and nothing else.

A plugin's `PreToolUse` and `PostToolUse` hooks run around a node's tool calls, matched on the tool
name by the group's `matcher` regex. Each answers with the usual envelope, and only
`additionalContext` is read:

```json
{"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "…"}}
```

That text is appended to the tool result as a labelled note, which is the only place it can reach
the model. **Write facts, not instructions.** A node treats imperative text arriving through a tool
result as untrusted — correctly, since that is where prompt injection would come from — so a hook
that says "always do X" is likely to be quoted and refused, while one that says what is true about
the repository gets used. Whether a call proceeds at all is not a plugin's decision: gating belongs
to a ruleset's `onToolCall`, which is the repository speaking about its own agents.

These fire on *every* matching call, so they are budgeted: 5s per hook, 2000 characters per call,
and 60s of total hook time per run, after which the run stops calling them and says so. Note that
existing plugins match a coding CLI's tool vocabulary (`^(Grep|Read|Bash|Write|Edit)$`) and a
planning node calls none of those — this is for hooks written against the tools nodes actually
call, like `semantic_search` and `impact_surface`.

### Scripted orchestration

`.ratatoskr/workflow.ts` replaces the built-in run flow outright when present, letting a repo
express its own ordering, fan-out, and gating over the same nodes. See
[`examples/workflow.ts`](examples/workflow.ts). Every safety gate stays enforced in Rust on the
bindings rather than delegated to the script.

`.ratatoskr/` otherwise holds runtime state — logs, the store, worktrees — and is gitignored, except
for `rules/` and `workflow.ts`, which are version-controlled.

## Workspace

| Crate | Role |
|---|---|
| `ratatoskr-core` | Domain types: `RunState`, `RunStatus`, `RatatoskrConfig`, the `ToolPolicy` seam. No async runtime. |
| `ratatoskr-graph` | The `Node` trait, `NodeError`, and the `parse_validated` schema gate. |
| `ratatoskr-mcp` | rag-rat MCP client — spawns rag-rat, lists tools, hands back a cloneable sink. |
| `ratatoskr-agent` | Builds a `rig` agent bound to a model + rag-rat's tools; the per-call ruleset gate. |
| `ratatoskr-script` | TypeScript rulesets and `workflow.ts`: transpile (swc) + evaluate (rquickjs). |
| `ratatoskr-nodes` | The nodes, plus the `run_plan` / `run_full` executors and the converge loop. |
| `ratatoskr-exec` | Git worktrees, sandboxed execution, and the ACP client that drives a coding CLI. |
| `ratatoskr-store` | SQLite checkpoint store, single-writer by construction. |
| `ratatoskr-serve` | Read-only HTTP API over the store, the run launcher, and the dashboard UI. |
| `ratatoskr-cli` | The `ratatoskr` binary. |

## Development

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

CI runs all three and they must be green; the workspace sets `warnings = "deny"`, so a warning
fails the build. `AGENTS.md` covers the conventions this repo expects from contributors.

> **rig note:** the agent runtime lives in `rig-agent` (split out of `rig-core` in 0.41), and its
> `.rmcp_tools()` bridge pins **`rmcp` 2.x** — so this workspace's `rmcp` is held at 2.x to match.
> A 3.x `ServerSink` would not type-check against it.

## License

MIT — see [LICENSE](LICENSE).

[rag-rat]: https://github.com/cq27-dev/rag-rat
