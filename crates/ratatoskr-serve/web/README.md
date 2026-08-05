# Dashboard UI

The browser front end for `ratatoskr serve` — React + [`@xyflow/react`], TypeScript, built by Vite.

```sh
bun install
bun run build        # emits dist/, which `ratatoskr serve` picks up automatically
bun run dev          # hot-reloading UI on :5173, proxying /api to a running `ratatoskr serve`
bun run typecheck    # tsc --noEmit; also runs as part of `build`
```

`npm` works just as well; `bun` is only what this was developed against.

## How it's wired

`dist/` is **not** committed and the Rust build does not depend on it: `ratatoskr serve` looks for
`crates/ratatoskr-serve/web/dist/index.html` (override with `RATATOSKR_WEB_DIR`) and serves the API
alone if it isn't there. So a Rust-only checkout still works, it just has no UI until you run the
build above.

That default path is resolved at compile time from this crate's source directory, which only
exists for a build from a checkout. A binary installed some other way — `cargo install`, a package
built from the registry — will never find it and will run API-only no matter what you build. Point
`RATATOSKR_WEB_DIR` at the built `dist/` in that case.

## Shape

- `api.ts` — the server's response types, mirroring the Rust structs in `../src/lib.rs` and
  `../src/pipeline.rs`. Keep them in step; nothing generates one from the other.
- `PipelineGraph.tsx` — the run graph. The pipeline's shape is fixed and known ahead of time, so
  the layout is hand-authored rather than run through `elkjs`/`dagre`: those exist for graphs whose
  shape isn't known until runtime. The converge loop is a real self-edge, not an annotation.
- `App.tsx` — run rail, run header, and the per-node output panes. Polls while the selected run is
  live; that is replaced by a pushed log tail once the structured log lands.
- `style.css` — the tactical-telemetry substrate: dark, monospace, hairline rules, 90° corners
  everywhere. Terminal green is reserved for exactly one thing, the `working` state, so a live run
  is the only thing on screen that glows.

[`@xyflow/react`]: https://reactflow.dev
