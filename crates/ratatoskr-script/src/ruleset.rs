//! Load `.ratatoskr/rules/*.ts`, evaluate them in a resident rquickjs context, and expose each
//! agent's static config plus a [`ToolPolicy`] driven by its `onToolCall` hook.
//!
//! Per the design review: the `onToolCall` function stays JS-side (registered in `globalThis`) —
//! `rquickjs::Persistent` is `!Send` and can't sit in the `Send + Sync` policy. The resident Rust
//! state is only the `AsyncContext` (Send + Sync under `parallel`); `decide` looks the function up
//! *inside* `async_with` and returns only an owned [`ToolDecision`], so no `'js` value crosses the
//! hook boundary.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use ratatoskr_core::{ToolDecision, ToolPolicy};
use rquickjs::{AsyncContext, AsyncRuntime, CatchResultExt, Function};
use serde::Deserialize;

use crate::ScriptError;
use crate::transpile::transpile_ts;

/// JS prelude: the `defineAgent` / `defineDefaults` registries, the tool-call dispatcher, and a
/// static-config extractor.
const BOOTSTRAP: &str = r#"
globalThis.__agents = {};
globalThis.__defaults = {};
function defineAgent(name, config) { globalThis.__agents[name] = config || {}; }
function defineDefaults(config) { globalThis.__defaults = config || {}; }
globalThis.__onToolCall = function(name, tool, argsJson) {
    var a = globalThis.__agents[name];
    if (!a || typeof a.onToolCall !== 'function') return "allow";
    var args;
    try { args = JSON.parse(argsJson); } catch (e) { args = {}; }
    var r = a.onToolCall({ tool: tool, args: args });
    if (r === "deny" || (r && r.deny)) return "deny";
    return "allow";
};
globalThis.__staticConfig = function() {
    var agents = {};
    for (var k in globalThis.__agents) {
        var a = globalThis.__agents[k];
        agents[k] = {
            model: a.model || null,
            tools: a.tools || null,
            maxTurns: (typeof a.maxTurns === 'number') ? a.maxTurns : null,
            systemPrompt: (typeof a.systemPrompt === 'string') ? a.systemPrompt : null,
            systemPromptFile: (typeof a.systemPromptFile === 'string') ? a.systemPromptFile : null,
            plugins: (a.plugins === undefined) ? null : a.plugins,
            hasOnToolCall: typeof a.onToolCall === 'function'
        };
    }
    var d = globalThis.__defaults;
    // Same reasoning as the plugins rule: a misspelled key would otherwise declare nothing and say
    // nothing. `mayModifyTests` in particular is a safety exemption — silently binding nothing
    // there means every legitimate test change fails to converge with no explanation.
    for (var dk in d) {
        if (dk !== 'plugins' && dk !== 'mayModifyTests') {
            throw new Error("defineDefaults: unknown key '" + dk + "'");
        }
    }
    return JSON.stringify({
        agents: agents,
        defaults: {
            plugins: (d.plugins === undefined) ? null : d.plugins,
            mayModifyTests: d.mayModifyTests || []
        }
    });
};
"#;

/// A `{ provider, model }` override.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelRule {
    pub provider: String,
    pub model: String,
}

/// Tool allow/deny. `allow` (if present) REPLACES the node's default set; `deny` is always removed.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ToolRule {
    #[serde(default)]
    pub allow: Option<Vec<String>>,
    #[serde(default)]
    pub deny: Vec<String>,
}

/// Which plugins a node gets.
///
/// Two spellings, because both readings are natural: a bare list means exactly these, and an
/// object adjusts what it inherits. Inheriting is the default in the object form because the
/// common case is one repository-wide set, occasionally tweaked — opting out of it has to be
/// said out loud rather than implied by an absent key.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PluginRule {
    /// `plugins: ["a", "b"]` — exactly these, ignoring the defaults.
    Only(Vec<String>),
    Adjust(AdjustRule),
}

/// `plugins: { inherit?, add?, remove? }`.
///
/// Unknown keys are refused: a misspelled `adds` would otherwise match with every field defaulted,
/// binding nothing and reporting nothing — the quiet failure this whole feature exists to avoid.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdjustRule {
    /// `false` to start from nothing rather than the defaults.
    #[serde(default = "yes")]
    pub inherit: bool,
    #[serde(default)]
    pub add: Vec<String>,
    #[serde(default)]
    pub remove: Vec<String>,
}

fn yes() -> bool {
    true
}

/// What `defineDefaults` declared, inherited by every node.
///
/// Run-scoped, not per-agent: "which plugins this repo uses" and "which paths this task may change"
/// are properties of the run, not of one node.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Defaults {
    #[serde(default)]
    pub plugins: Option<Vec<String>>,
    /// Paths this task is allowed to change even though they are part of the referee (the tests and
    /// the machinery that runs them). Declared up front by whoever wrote the task — the implementer
    /// does not get to decide after the fact that the tests were the problem.
    #[serde(default, rename = "mayModifyTests")]
    pub may_modify_tests: Vec<String>,
}

/// The static (non-executable) config a ruleset declares for one agent.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AgentRuleset {
    #[serde(default)]
    pub model: Option<ModelRule>,
    #[serde(default)]
    pub tools: Option<ToolRule>,
    #[serde(default, rename = "maxTurns")]
    pub max_turns: Option<usize>,
    /// Replaces the node's built-in preamble when set.
    #[serde(default, rename = "systemPrompt")]
    pub system_prompt: Option<String>,
    /// A file to read the preamble from instead, resolved against the rules directory.
    ///
    /// The same override as `systemPrompt`, for a prompt long enough that inlining it as a TS
    /// string literal is how it stops being editable — which is the state the built-in prompts were
    /// in before they moved to files. Read at load, so a missing file is a startup error rather
    /// than a node that runs with its built-in preamble and no indication why.
    #[serde(default, rename = "systemPromptFile")]
    pub system_prompt_file: Option<String>,
    /// Which plugins this node gets; `None` means "whatever the defaults say".
    #[serde(default)]
    pub plugins: Option<PluginRule>,
    #[serde(default, rename = "hasOnToolCall")]
    pub has_on_tool_call: bool,
}

/// The whole static config a ruleset directory declares.
#[derive(Debug, Clone, Default, Deserialize)]
struct StaticConfig {
    #[serde(default)]
    agents: HashMap<String, AgentRuleset>,
    #[serde(default)]
    defaults: Defaults,
}

/// Keep first occurrences, drop repeats — a plugin named twice is bound once.
fn dedup(names: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    names
        .into_iter()
        .filter(|n| seen.insert(n.clone()))
        .collect()
}

/// A loaded ruleset engine: the resident JS context plus each agent's extracted static config.
pub struct ScriptEngine {
    // Keep the runtime alive alongside the context.
    _runtime: AsyncRuntime,
    context: AsyncContext,
    agents: HashMap<String, AgentRuleset>,
    defaults: Defaults,
}

impl ScriptEngine {
    /// Load and evaluate every `*.ts` in `rules_dir` (empty engine if the dir is absent).
    pub async fn load(rules_dir: &Path) -> Result<Arc<Self>, ScriptError> {
        // Bootstrap + all transpiled scripts, concatenated into one eval.
        let mut program = String::from(BOOTSTRAP);
        if rules_dir.is_dir() {
            let mut paths: Vec<_> = std::fs::read_dir(rules_dir)
                .map_err(|e| ScriptError::Io(rules_dir.display().to_string(), e))?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("ts"))
                .collect();
            paths.sort();
            for path in paths {
                let src = std::fs::read_to_string(&path)
                    .map_err(|e| ScriptError::Io(path.display().to_string(), e))?;
                program.push('\n');
                program.push_str(&transpile_ts(&src)?);
            }
        }

        let runtime = AsyncRuntime::new().map_err(|e| ScriptError::Eval(e.to_string()))?;
        let context = AsyncContext::full(&runtime)
            .await
            .map_err(|e| ScriptError::Eval(e.to_string()))?;

        let agents_json: String = context
            .async_with(async move |ctx| {
                ctx.eval::<(), _>(program)
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                let f: Function = ctx
                    .globals()
                    .get("__staticConfig")
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                f.call::<_, String>(())
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))
            })
            .await?;

        let mut static_config: StaticConfig = serde_json::from_str(&agents_json)
            .map_err(|e| ScriptError::Eval(format!("static config parse: {e}")))?;
        for (node, agent) in &mut static_config.agents {
            let Some(file) = agent.system_prompt_file.clone() else {
                continue;
            };
            if agent.system_prompt.is_some() {
                return Err(ScriptError::Eval(format!(
                    "defineAgent(\"{node}\") sets both systemPrompt and systemPromptFile; \
                     one of them would be silently ignored"
                )));
            }
            // Beside the ruleset that names it, so a workflow's prompts live with the rules that
            // select them rather than somewhere a reader has to be told about.
            let path = rules_dir.join(&file);
            let text = std::fs::read_to_string(&path)
                .map_err(|e| ScriptError::Io(path.display().to_string(), e))?;
            agent.system_prompt = Some(text);
        }

        Ok(Arc::new(ScriptEngine {
            _runtime: runtime,
            context,
            agents: static_config.agents,
            defaults: static_config.defaults,
        }))
        // ponytail: no JS eval budget yet (repo rules are trusted code). Add
        // AsyncRuntime::set_interrupt_handler if untrusted scripts ever run.
    }

    /// Which of the `discovered` plugins `node` gets, after inheritance.
    ///
    /// A node with no ruleset, or one that says nothing about plugins, gets the defaults — and the
    /// defaults are every discovered plugin unless `defineDefaults` says otherwise, so installing
    /// a plugin is enough to use it. Order follows the declaration: defaults first, then whatever
    /// the node adds, because that is the order their session context is read in.
    pub fn plugins_for(&self, node: &str, discovered: &[String]) -> Vec<String> {
        // Installing a plugin is itself the statement that you want it: with no `defineDefaults`,
        // every discovered plugin applies. A ruleset narrows that; it isn't a prerequisite for
        // plugins working at all.
        let defaults = || {
            self.defaults
                .plugins
                .clone()
                .unwrap_or_else(|| discovered.to_vec())
        };
        let Some(rule) = self.agents.get(node).and_then(|a| a.plugins.as_ref()) else {
            return defaults();
        };

        match rule {
            PluginRule::Only(only) => dedup(only.clone()),
            PluginRule::Adjust(rule) => {
                let mut names = if rule.inherit { defaults() } else { Vec::new() };
                names.extend(rule.add.iter().cloned());
                names.retain(|n| !rule.remove.contains(n));
                dedup(names)
            }
        }
    }

    /// Every plugin name any ruleset mentions, for validating them against what was discovered.
    pub fn declared_plugins(&self) -> Vec<String> {
        let mut names = self.defaults.plugins.clone().unwrap_or_default();
        for agent in self.agents.values() {
            match &agent.plugins {
                Some(PluginRule::Only(only)) => names.extend(only.iter().cloned()),
                Some(PluginRule::Adjust(rule)) => {
                    names.extend(rule.add.iter().cloned());
                    // A `remove` that names nothing real is just as much a typo as an `add`.
                    names.extend(rule.remove.iter().cloned());
                }
                None => {}
            }
        }
        dedup(names)
    }

    /// The ruleset governing `node`, if one was declared.
    pub fn ruleset(self: &Arc<Self>, node: &str) -> Option<NodeRuleset> {
        self.agents.get(node).map(|config| NodeRuleset {
            engine: Arc::clone(self),
            node: node.to_string(),
            config: config.clone(),
        })
    }

    /// Paths this task declared it may change even though they are part of the referee — the
    /// converge-time exemption. Empty means "the tests are off limits", which is the default.
    pub fn may_modify_tests(&self) -> &[String] {
        &self.defaults.may_modify_tests
    }

    /// Names of every agent a ruleset declared (for validating `defineAgent` targets).
    pub fn declared_agents(&self) -> impl Iterator<Item = &str> {
        self.agents.keys().map(String::as_str)
    }
}

/// A per-node handle: the static config plus a [`ToolPolicy`] over the node's `onToolCall`.
pub struct NodeRuleset {
    engine: Arc<ScriptEngine>,
    node: String,
    config: AgentRuleset,
}

impl NodeRuleset {
    pub fn config(&self) -> &AgentRuleset {
        &self.config
    }
}

impl ToolPolicy for NodeRuleset {
    fn decide<'a>(
        &'a self,
        tool_name: &'a str,
        args_json: &'a str,
    ) -> Pin<Box<dyn Future<Output = ToolDecision> + Send + 'a>> {
        // No onToolCall hook → allow without entering JS.
        if !self.config.has_on_tool_call {
            return Box::pin(async { ToolDecision::Allow });
        }
        let (node, tool, args) = (
            self.node.clone(),
            tool_name.to_string(),
            args_json.to_string(),
        );
        Box::pin(async move {
            let decision: String = self
                .engine
                .context
                .async_with(async move |ctx| {
                    let f: Function = match ctx.globals().get("__onToolCall") {
                        Ok(f) => f,
                        // Bootstrap always defines this, so this is effectively unreachable.
                        Err(_) => return "allow".to_string(),
                    };
                    // A throwing/ill-typed onToolCall fails OPEN (allow) so a buggy rule can't brick
                    // every run — but never silently: the JS error is logged loudly so the author
                    // sees that their policy isn't taking effect.
                    match f.call::<_, String>((node, tool.clone(), args)).catch(&ctx) {
                        Ok(d) => d,
                        Err(e) => {
                            tracing::error!(tool = %tool, error = %e, "onToolCall failed; failing open to allow");
                            "allow".to_string()
                        }
                    }
                })
                .await;
            if decision == "deny" {
                ToolDecision::Deny(format!("denied by ruleset policy for tool `{tool_name}`"))
            } else {
                ToolDecision::Allow
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ontoolcall_allows_and_denies_per_rule() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-script-ruleset-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("scout.ts"),
            r#"
            type ToolCall = { tool: string };
            defineAgent("scout", {
                onToolCall({ tool }: ToolCall): "allow" | "deny" {
                    return tool === "papertrail_issue_search" ? "deny" : "allow";
                },
            });
            "#,
        )
        .unwrap();

        let engine = ScriptEngine::load(&dir).await.unwrap();
        let rs = engine.ruleset("scout").expect("scout ruleset");
        assert!(rs.config().has_on_tool_call);

        assert!(matches!(
            rs.decide("papertrail_issue_search", "{}").await,
            ToolDecision::Deny(_)
        ));
        assert!(matches!(
            rs.decide("semantic_search", "{}").await,
            ToolDecision::Allow
        ));
        // A node with no ruleset gets nothing to gate.
        assert!(engine.ruleset("analyst").is_none());
    }

    /// Load a ruleset directory from source, for the plugin-binding tests.
    async fn engine_with(case: &str, source: &str) -> Arc<ScriptEngine> {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-plugin-binding-{}-{case}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("agents.ts"), source).unwrap();
        ScriptEngine::load(&dir).await.unwrap()
    }

    /// Stand-in for what discovery found.
    fn discovered() -> Vec<String> {
        [
            "rag-rat",
            "noisy",
            "impact-lens",
            "scout-only",
            "exactly-this",
            "extra",
            "a",
            "b",
            "c",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    #[tokio::test]
    async fn with_no_declarations_every_discovered_plugin_applies() {
        // Installing a plugin is the statement that you want it; a ruleset narrows, it is not a
        // prerequisite. Without this, adding the binding feature would silently unbind every
        // existing repo.
        let engine = engine_with("implicit", r#"defineAgent("scout", { maxTurns: 2 });"#).await;
        let discovered = vec!["rag-rat".to_string(), "other".to_string()];

        assert_eq!(
            engine.plugins_for("scout", &discovered),
            ["rag-rat", "other"]
        );
        assert_eq!(
            engine.plugins_for("analyst", &discovered),
            ["rag-rat", "other"]
        );
        assert!(engine.declared_plugins().is_empty());
    }

    #[tokio::test]
    async fn a_misspelled_key_is_refused_rather_than_defaulted() {
        // `adds` would otherwise match with every field defaulted: nothing bound, nothing said.
        let dir = std::env::temp_dir().join(format!("ratatoskr-typo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("agents.ts"),
            r#"defineAgent("analyst", { plugins: { adds: ["impact-lens"] } });"#,
        )
        .unwrap();

        assert!(
            ScriptEngine::load(&dir).await.is_err(),
            "an unknown key in a plugins rule fails the load"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn defaults_reach_a_node_that_says_nothing() {
        // Adding a plugin repository-wide must not require touching every node.
        let engine = engine_with(
            "inherit",
            r#"
            defineDefaults({ plugins: ["rag-rat"] });
            defineAgent("analyst", { maxTurns: 3 });
            "#,
        )
        .await;
        let discovered = discovered();

        assert_eq!(engine.plugins_for("analyst", &discovered), ["rag-rat"]);
        // Including a node with no ruleset at all.
        assert_eq!(engine.plugins_for("scout", &discovered), ["rag-rat"]);
    }

    #[tokio::test]
    async fn a_node_adds_and_removes_against_the_defaults() {
        let engine = engine_with(
            "adjust",
            r#"
            defineDefaults({ plugins: ["rag-rat", "noisy"] });
            defineAgent("analyst", { plugins: { add: ["impact-lens"], remove: ["noisy"] } });
            "#,
        )
        .await;
        let discovered = discovered();

        // Defaults first, then what the node adds — the order their context is read in.
        assert_eq!(
            engine.plugins_for("analyst", &discovered),
            ["rag-rat", "impact-lens"]
        );
        assert_eq!(
            engine.plugins_for("scout", &discovered),
            ["rag-rat", "noisy"]
        );
    }

    #[tokio::test]
    async fn opting_out_of_the_defaults_must_be_said_out_loud() {
        let engine = engine_with(
            "optout",
            r#"
            defineDefaults({ plugins: ["rag-rat"] });
            defineAgent("scout", { plugins: { inherit: false, add: ["scout-only"] } });
            defineAgent("bookkeeper", { plugins: ["exactly-this"] });
            defineAgent("analyst", { plugins: { add: ["extra"] } });
            "#,
        )
        .await;
        let discovered = discovered();

        assert_eq!(engine.plugins_for("scout", &discovered), ["scout-only"]);
        // A bare list is "exactly these" — the other natural reading of the same key.
        assert_eq!(
            engine.plugins_for("bookkeeper", &discovered),
            ["exactly-this"]
        );
        // Omitting `inherit` keeps the defaults, which is the common case.
        assert_eq!(
            engine.plugins_for("analyst", &discovered),
            ["rag-rat", "extra"]
        );
    }

    #[tokio::test]
    async fn every_mentioned_plugin_is_reported_for_validation() {
        // A name that matches no discovered plugin is a typo, whether it was added or removed.
        let engine = engine_with(
            "declared",
            r#"
            defineDefaults({ plugins: ["rag-rat"] });
            defineAgent("scout", { plugins: { add: ["a"], remove: ["b"] } });
            defineAgent("analyst", { plugins: ["c", "rag-rat"] });
            "#,
        )
        .await;

        let mut declared = engine.declared_plugins();
        declared.sort();
        assert_eq!(declared, ["a", "b", "c", "rag-rat"]);
    }

    #[tokio::test]
    async fn the_referee_exemption_is_declared_up_front_in_the_defaults() {
        let engine = engine_with(
            "may-modify",
            r#"defineDefaults({ mayModifyTests: ["crates/foo/tests", "conftest.py"] });"#,
        )
        .await;
        assert_eq!(
            engine.may_modify_tests(),
            ["crates/foo/tests", "conftest.py"]
        );

        // Undeclared is the default, and the strict reading: no exemption.
        let none = engine_with("may-modify-absent", r#"defineDefaults({ plugins: [] });"#).await;
        assert!(none.may_modify_tests().is_empty());

        // A typo would otherwise exempt nothing and report nothing — the failure mode that turns a
        // legitimate test-writing task into an unexplained MaxIterationsReached.
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-defaults-typo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("agents.ts"),
            r#"defineDefaults({ mayModifyTest: ["tests"] });"#,
        )
        .unwrap();
        assert!(ScriptEngine::load(&dir).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn model_and_system_prompt_deserialize() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-script-ruleset-prompt-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("agents.ts"),
            r#"
            defineAgent("analyst", {
                model: { provider: "anthropic", model: "claude-opus-4-8" },
                systemPrompt: "You are a terse analyst.",
                maxTurns: 7,
            });
            defineAgent("scout", { model: { provider: "moonshot", model: "kimi-k2.5" } });
            defineAgent("bookkeeper", { maxTurns: 3 });
            "#,
        )
        .unwrap();

        let engine = ScriptEngine::load(&dir).await.unwrap();

        let analyst = engine.ruleset("analyst").unwrap();
        let c = analyst.config();
        assert_eq!(c.model.as_ref().unwrap().provider, "anthropic");
        assert_eq!(c.system_prompt.as_deref(), Some("You are a terse analyst."));
        assert_eq!(c.max_turns, Some(7));

        // model-only: everything else stays absent.
        let scout = engine.ruleset("scout").unwrap();
        assert_eq!(scout.config().model.as_ref().unwrap().model, "kimi-k2.5");
        assert!(scout.config().system_prompt.is_none());

        // no model, no prompt — the fallback path.
        let bk = engine.ruleset("bookkeeper").unwrap();
        assert!(bk.config().model.is_none());
        assert!(bk.config().system_prompt.is_none());
    }

    #[tokio::test]
    async fn a_ruleset_can_read_its_preamble_from_a_file_beside_it() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-prompt-file-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("research-analyst.md"),
            "You are analysing a question, not planning a change.\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("analyst.ts"),
            r#"defineAgent("analyst", { systemPromptFile: "research-analyst.md" });"#,
        )
        .unwrap();

        let engine = ScriptEngine::load(&dir).await.unwrap();
        let prompt = engine
            .ruleset("analyst")
            .unwrap()
            .config()
            .system_prompt
            .clone();
        assert_eq!(
            prompt.as_deref(),
            Some("You are analysing a question, not planning a change.\n"),
            "the file's contents become the preamble, same as an inline systemPrompt"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_missing_prompt_file_and_a_double_declaration_are_both_startup_errors() {
        let dir = std::env::temp_dir().join(format!("ratatoskr-prompt-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // A missing file must not leave the node running its built-in preamble with no indication.
        std::fs::write(
            dir.join("a.ts"),
            r#"defineAgent("analyst", { systemPromptFile: "gone.md" });"#,
        )
        .unwrap();
        let err = match ScriptEngine::load(&dir).await {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a missing prompt file must fail at load"),
        };
        assert!(err.contains("gone.md"), "{err}");

        // Both set is ambiguous, and silently preferring one is how a prompt edit does nothing.
        std::fs::write(
            dir.join("a.ts"),
            r#"defineAgent("analyst", { systemPrompt: "inline", systemPromptFile: "x.md" });"#,
        )
        .unwrap();
        let err = match ScriptEngine::load(&dir).await {
            Err(e) => e.to_string(),
            Ok(_) => panic!("declaring both must fail"),
        };
        assert!(
            err.contains("both systemPrompt and systemPromptFile"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
