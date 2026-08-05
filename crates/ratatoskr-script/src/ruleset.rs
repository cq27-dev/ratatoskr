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

/// JS prelude: `defineAgent` registry, the tool-call dispatcher, and a static-config extractor.
const BOOTSTRAP: &str = r#"
globalThis.__agents = {};
function defineAgent(name, config) { globalThis.__agents[name] = config || {}; }
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
    var out = {};
    for (var k in globalThis.__agents) {
        var a = globalThis.__agents[k];
        out[k] = {
            model: a.model || null,
            tools: a.tools || null,
            maxTurns: (typeof a.maxTurns === 'number') ? a.maxTurns : null,
            hasOnToolCall: typeof a.onToolCall === 'function'
        };
    }
    return JSON.stringify(out);
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

/// The static (non-executable) config a ruleset declares for one agent.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AgentRuleset {
    #[serde(default)]
    pub model: Option<ModelRule>,
    #[serde(default)]
    pub tools: Option<ToolRule>,
    #[serde(default, rename = "maxTurns")]
    pub max_turns: Option<usize>,
    #[serde(default, rename = "hasOnToolCall")]
    pub has_on_tool_call: bool,
}

/// A loaded ruleset engine: the resident JS context plus each agent's extracted static config.
pub struct ScriptEngine {
    // Keep the runtime alive alongside the context.
    _runtime: AsyncRuntime,
    context: AsyncContext,
    agents: HashMap<String, AgentRuleset>,
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

        let agents: HashMap<String, AgentRuleset> = serde_json::from_str(&agents_json)
            .map_err(|e| ScriptError::Eval(format!("static config parse: {e}")))?;

        Ok(Arc::new(ScriptEngine {
            _runtime: runtime,
            context,
            agents,
        }))
        // ponytail: no JS eval budget yet (repo rules are trusted code). Add
        // AsyncRuntime::set_interrupt_handler if untrusted scripts ever run.
    }

    /// The ruleset governing `node`, if one was declared.
    pub fn ruleset(self: &Arc<Self>, node: &str) -> Option<NodeRuleset> {
        self.agents.get(node).map(|config| NodeRuleset {
            engine: Arc::clone(self),
            node: node.to_string(),
            config: config.clone(),
        })
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
        let dir = std::env::temp_dir().join("ratatoskr-script-ruleset-test");
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
}
