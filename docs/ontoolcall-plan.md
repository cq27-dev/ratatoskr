# Plan: onToolCall dynamic policy hook (the first executable-code ruleset need)

Chosen over static-config-in-TS because static config is just data (Fable: use TOML) — the TS engine
earns its place only for *executable* per-tool-call decisions. All three technical risks are retired:
rquickjs async (spiked, concurrent), swc `strip` transpile (API confirmed), and the interception seam
(`AgentHook::on_tool_call -> ToolCallAction`, `Skip` = deny, `Rewrite` = modify).

## New crate: `ratatoskr-script`

- `transpile`: TS→JS via `swc_core::ecma::transforms::typescript::strip` (parse TS → resolver →
  strip → codegen). Minimal feature set on `swc_core`.
- `ruleset`: load `.ratatoskr/rules/<node>.ts` → transpile → eval in a warm `rquickjs` async context
  (`parallel` + `full-async`, proven). The script calls `defineAgent(name, { model?, tools?,
  maxTurns?, onToolCall? })`; the host captures the config object incl. the `onToolCall` JS function,
  kept resident for the run (hot path — loaded once, called per tool-call).
- Public surface:
  ```rust
  pub struct AgentRuleset { model: Option<ModelRoute>, tools: Option<ToolRule>, max_turns: Option<usize>, has_on_tool_call: bool }
  pub enum ToolDecision { Allow, Deny(String), Rewrite(serde_json::Value) }
  #[async_trait-ish] pub trait ToolPolicy: Send + Sync {
      async fn decide(&self, tool_name: &str, args_json: &str) -> ToolDecision;
  }
  ```
  `ToolPolicy` lives in `ratatoskr-agent` (see deps); `ratatoskr-script::Ruleset` implements it by
  calling the resident JS `onToolCall({tool, args})` and mapping the string/object result.
- Eval safety: an interrupt handler / eval budget so an accidental infinite loop can't hang the run;
  transpile/eval errors are a hard, named failure at load — never silently "empty rulesets". Repo
  `.ts` is trusted-repo code (same tier as `ratatoskr.toml`).

## `ratatoskr-agent`

- Define the `ToolPolicy` trait here (so the crate that owns `run_structured` owns the seam;
  `ratatoskr-script` depends on `ratatoskr-agent` to implement it — no cycle).
- `RulesetHook { policy: Arc<dyn ToolPolicy> }` implements `rig_agent::agent::AgentHook`:
  `on_tool_call(call) -> ToolCallAction` = `policy.decide(call.tool_name, call.args).await` mapped
  `Allow→Run`, `Deny(s)→Skip(s)`, `Rewrite(v)→Rewrite(v)`. Other hook methods default.
- `run_structured` gains `policy: Option<Arc<dyn ToolPolicy>>`; when `Some`, push `RulesetHook` onto
  the builder (`.hook(...)` / `HookStack::push`). Also honor `max_turns` (Fable's Q2: add
  `max_turns: Option<usize>` defaulting to `DEFAULT_MAX_TURNS`).

## `ratatoskr-nodes` integration

- Load `Rulesets` once (in the CLI, thread through — not cwd IO inside nodes) and pass to the three
  pub entry points → `run_nodes`, `fork_and_converge`, AND `bookkeep_and_checkpoint` (Fable must-fix:
  the `run_bookkeeper` replay path must get the same rulesets).
- Per node with a ruleset: model override + tool set (allow **replaces** the `*_TOOLS` default,
  validated against the server list; deny removes) + max_turns + the `ToolPolicy` passed to
  `run_structured`. Applies to the 4 LLM agents: **scout, analyst, bookkeeper, redteam classifier**.
  `defineAgent("memory"/"implementer")` is a hard error (no model/tools there). Overrides are applied
  at construction sites; `config.models` is never mutated (Fable must-fix: would wrongly enable the
  opt-in classifier).
- Built-in default: no `.ratatoskr/rules/` → empty `Rulesets` → every existing expression runs
  unchanged, byte-for-byte.

## Deps / order

`ratatoskr-script` → `swc_core` (minimal), `rquickjs` (parallel, full-async), `ratatoskr-agent`
(ToolPolicy), `ratatoskr-core`, serde/serde_json/thiserror. Build order: transpile+eval spike-grade
core → ToolPolicy/RulesetHook → run_structured wiring → node threading → a `.ratatoskr/rules/*.ts`
demo denying a specific tool call, verified live on the fixture.

## Open questions for the gate

1. `run_structured` currently forces `OutputMode::Tool` and builds one `AgentBuilder`; does pushing a
   hook compose with that, and is `.hook()` the real builder method (confirm name/signature)?
2. `AgentHook::on_tool_call` is `async` + `Send+Sync`; calling into a resident rquickjs context from
   there — is the context `Send` under `parallel`, and does holding it across the `.await` in the hook
   cause `'js`-lifetime trouble (the spike's friction)?
3. Does denying via `Skip(feedback)` leave the agent's tool loop well-formed (no orphaned tool_use),
   given rig-agent's invalid-tool-call handling?
