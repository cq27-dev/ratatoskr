# Ratatoskr

An orchestrator for rag-rat-driven coding runs: a graph of agent nodes that scout, analyze,
red-team, and implement against a repository, checkpointing state as it goes.

> **Status: Phase 1 (loop only).** A single agent can launch rag-rat over MCP and answer a
> question about the target repo using rag-rat's real tools. No graph executor, no worktrees, no
> checkpoint writes for the turn yet — those are Phases 2–3.

## Workspace

| Crate | Role |
|---|---|
| `ratatoskr-core` | Domain types: `RunState`, `RatatoskrConfig`. No async runtime dependency. |
| `ratatoskr-graph` | The `Node` trait and `Edge` type. No executor yet. |
| `ratatoskr-mcp` | rag-rat MCP client — spawns rag-rat, lists tools, hands back a client handle. |
| `ratatoskr-agent` | Builds a `rig` agent bound to a model + rag-rat's tools, runs one prompt. |
| `ratatoskr-store` | SQLite checkpoint store (single-writer). |
| `ratatoskr-cli` | The `ratatoskr` binary — `--version`, `init`, `ask`. |

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

## License

MIT — see [LICENSE](LICENSE).
