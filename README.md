# Ratatoskr

An orchestrator for rag-rat-driven coding runs: a graph of agent nodes that scout, analyze,
red-team, and implement against a repository, checkpointing state as it goes.

> **Status: Phase 0 (skeleton).** The Cargo workspace, crate boundaries, config shape, and
> checkpoint schema are fixed and CI is green — but there is no agent logic, no graph executor,
> and no live rag-rat connection yet. Those are Phases 1–3.

## Workspace

| Crate | Role |
|---|---|
| `ratatoskr-core` | Domain types: `RunState`, `RatatoskrConfig`. No async runtime dependency. |
| `ratatoskr-graph` | The `Node` trait and `Edge` type. No executor yet. |
| `ratatoskr-mcp` | rag-rat MCP client. `connect()` is stubbed until Phase 1. |
| `ratatoskr-store` | SQLite checkpoint store (single-writer). Real in Phase 0. |
| `ratatoskr-cli` | The `ratatoskr` binary — `--version` and `init`. |

## Build

```sh
cargo build --workspace
cargo test --workspace
cargo run -p ratatoskr-cli -- --version
cargo run -p ratatoskr-cli -- init      # writes ratatoskr.toml
```

## License

MIT — see [LICENSE](LICENSE).
