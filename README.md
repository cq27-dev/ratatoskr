# Ratatoskr

An orchestrator for rag-rat-driven coding runs: a graph of agent nodes that scout, analyze,
red-team, and implement against a repository, checkpointing state as it goes.

> **Status: Phase 2 (linear graph).** `ratatoskr plan` runs a three-node flow —
> scout → memory → analyst — over rag-rat's tools, schema-validating and checkpointing each
> node's output, and prints a grounded plan. No fork/worktrees/red-team yet — that's Phase 3.

## Workspace

| Crate | Role |
|---|---|
| `ratatoskr-core` | Domain types: `RunState`, `RatatoskrConfig`. No async runtime dependency. |
| `ratatoskr-graph` | The `Node` trait, `Edge`, and the `parse_validated` schema gate. No executor yet. |
| `ratatoskr-mcp` | rag-rat MCP client — spawns rag-rat, lists tools, hands back a client handle. |
| `ratatoskr-agent` | Builds a `rig` agent bound to a model + rag-rat's tools; one prompt, plain or structured. |
| `ratatoskr-nodes` | The scout/memory/analyst nodes and the straight-line `run_plan` executor. |
| `ratatoskr-store` | SQLite checkpoint store (single-writer): runs + per-node checkpoints. |
| `ratatoskr-cli` | The `ratatoskr` binary — `--version`, `init`, `ask`, `plan`. |

## Build

```sh
cargo build --workspace
cargo test --workspace
cargo run -p ratatoskr-cli -- init      # writes ratatoskr.toml
```

## `ratatoskr ask`

Answer a question about the repo rag-rat is indexing, grounded in rag-rat's tools:

```sh
export ANTHROPIC_API_KEY=...            # or MOONSHOT_API_KEY, per [models.ask] in ratatoskr.toml
RUST_LOG=info,rig_agent=debug,rmcp=debug \
  cargo run -p ratatoskr-cli -- ask "what does the store crate do?"
```

`RUST_LOG=…rig_agent=debug,rmcp=debug` surfaces each tool call the agent makes, so you can see the
answer is grounded in rag-rat rather than the model's guesswork.

> **rig note:** the agent runtime lives in `rig-agent` (split out of `rig-core` in 0.41), and its
> `.rmcp_tools()` bridge pins **`rmcp` 2.x** — so this workspace's `rmcp` is held at 2.x to match.
> A 3.x `ServerSink` would not type-check against it.

## `ratatoskr plan`

Plan work for an issue — scout searches the tracker papertrail + code, memory corroborates from
rag-rat's own memories, and the analyst assesses impact and risk:

```sh
cargo run -p ratatoskr-cli -- plan "Add a status subcommand that prints a run's checkpoints"
cargo run -p ratatoskr-cli -- plan --file issue.md          # long descriptions
cargo run -p ratatoskr-cli -- plan "..." --json             # raw structured RunState
```

Each node's output is validated against its JSON Schema and checkpointed to the store before the
next node runs; a node failure stops the run with `status = failed` and a node-attributed error.
`plan` routes scout via `[models.scout]` and analyst via `[models.analyst]`.

## License

MIT — see [LICENSE](LICENSE).
