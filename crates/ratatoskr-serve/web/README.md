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

```
src/
  App.tsx        run state, and the layout the panels sit in
  api.ts         the server's response types + the fetch/stream calls
  derive.ts      node state derived from the event stream
  url.ts         the view as an address: project, run, node, position
  panels/        the regions of the dashboard
  ui/            presentation used by more than one panel
  style.css      the substrate
```

- `api.ts` — mirrors the Rust structs in `../src/lib.rs` and `../src/pipeline.rs`. Keep them in
  step; nothing generates one from the other.
- `App.tsx` — holds the state (selected project, run and node; history; the live event stream) and
  nothing else. Panels take props and are otherwise unaware of each other. A run's rows are read
  once and kept current by a subscription, not by polling; the run *list* is polled, because a run
  started outside the dashboard produces no event to subscribe to.
- `url.ts` — project, run, node and scrub position live in the query string, so a moment in a run
  is linkable and a reload lands where it was. Written with `replaceState` throughout ([`nuqs`]'s
  default): clicking through eight runs is one act of looking, not eight places to go back to. The
  position is carried as elapsed `m:ss` rather than an event index, which survives a live run
  growing and can be read off the link; the cost is that a second holding several events resolves
  to the last of them.
- `panels/PipelineGraph.tsx` — the run graph. The pipeline's shape is fixed and known ahead of
  time, so the layout is hand-authored rather than run through `elkjs`/`dagre`: those exist for
  graphs whose shape isn't known until runtime. The converge loop is a real self-edge, not an
  annotation.
- `panels/` — `Rail` (projects, new run, run list), `RunMeta` (the run header), `Scrubber`,
  `Feed` (the activity log and the `rows()` that builds it), `Detail` (a node's checkpoints),
  `Question` (a blocked node's clarification).
- `ui/` — `tint` (a colour per node, from its name), `format` (JSON colouring and the small subset
  of markdown a model actually emits), `tools` (what a tool is, as an icon), `text` (the display
  shortenings), `Tooltip` (one tooltip for the whole page, driven by `data-tip`).
- `style.css` — the tactical-telemetry substrate: dark, monospace, hairline rules, 90° corners
  everywhere. Terminal green is reserved for exactly one thing, the `working` state, so a live run
  is the only thing on screen that glows. One file on purpose: the cascade is order-dependent and
  the variables are shared, so splitting it buys tidiness and costs a class of bug that is hard to
  see.

[`@xyflow/react`]: https://reactflow.dev
[`nuqs`]: https://nuqs.dev
