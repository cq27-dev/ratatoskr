//! Load and run `.ratatoskr/workflow.ts` — the optional script that composes the run flow.
//!
//! This is the node-agnostic seam. `ratatoskr-script` depends only on `ratatoskr-core`, so it can't
//! call the concrete nodes; instead the caller (`ratatoskr-nodes`) registers named async **host
//! functions**, each wrapping one node call site, and the script composes them. The runtime handles
//! only the JS↔Rust plumbing: transpile, register hosts, invoke the script's entry function, and
//! hand back JSON.
//!
//! Concurrency: a host function is exposed to JS as a promise-returning function backed by a spawned
//! Rust future, so `await Promise.all([a(), b()])` in a script genuinely forks (proven ~concurrent
//! under one `AsyncContext` — see the fork Decision memory), matching `tokio::join!`.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use rquickjs::promise::{Promise, Promised};
use rquickjs::{AsyncContext, AsyncRuntime, CatchResultExt, Function};

use crate::ScriptError;
use crate::transpile::transpile_ts;

/// A host function's result: `Ok(json)` — a JSON-encoded return value — or `Err(message)`, which the
/// script sees as a thrown `Error`.
pub type HostResult = Result<String, String>;

/// One host binding: takes the JSON-encoded JS argument, returns [`HostResult`]. `Send + Sync +
/// 'static` so it can be spawned as a JS-visible promise under the resident context.
pub type HostFn =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = HostResult> + Send>> + Send + Sync>;

/// JS prelude: wrap each raw host (`__name`, taking/returning JSON strings) as an ergonomic
/// `name(x)` that passes real JS values, and provide the entry invoker.
const BOOTSTRAP: &str = r#"
globalThis.__wrap = function (name) {
    return async function (x) {
        var raw = await globalThis["__" + name](JSON.stringify(x === undefined ? null : x));
        var r = JSON.parse(raw);
        if (r && Object.prototype.hasOwnProperty.call(r, "__error")) throw new Error(r.__error);
        return r.value;
    };
};
globalThis.__runEntry = async function (entry, inputJson) {
    var fn = globalThis[entry];
    if (typeof fn !== "function") {
        throw new Error("workflow.ts does not define a `" + entry + "` function");
    }
    var out = await fn(JSON.parse(inputJson));
    return JSON.stringify(out === undefined ? null : out);
};
"#;

/// A loaded workflow script: the resident JS context plus the transpiled source.
pub struct WorkflowRuntime {
    _runtime: AsyncRuntime,
    context: AsyncContext,
    source: String,
}

impl WorkflowRuntime {
    /// Load and transpile `path` (`.ratatoskr/workflow.ts`). `Ok(None)` if the file is absent —
    /// the caller then runs the built-in Rust flow, exactly as the ruleset engine treats a missing
    /// rules dir.
    pub async fn load(path: &Path) -> Result<Option<Self>, ScriptError> {
        if !path.is_file() {
            return Ok(None);
        }
        let src = std::fs::read_to_string(path)
            .map_err(|e| ScriptError::Io(path.display().to_string(), e))?;
        let source = transpile_ts(&src)?;

        let runtime = AsyncRuntime::new().map_err(|e| ScriptError::Eval(e.to_string()))?;
        let context = AsyncContext::full(&runtime)
            .await
            .map_err(|e| ScriptError::Eval(e.to_string()))?;

        Ok(Some(WorkflowRuntime {
            _runtime: runtime,
            context,
            source,
        }))
    }

    /// Register `hosts` and invoke the script's `entry` function with `input_json` (a JSON string),
    /// returning the entry's result as JSON. Any thrown error (host `Err`, or a bug in the script)
    /// surfaces as [`ScriptError::Eval`].
    pub async fn run(
        &self,
        entry: &str,
        input_json: String,
        hosts: HashMap<String, HostFn>,
    ) -> Result<String, ScriptError> {
        let program = format!("{BOOTSTRAP}\n{}", self.source);
        let entry = entry.to_string();

        self.context
            .async_with(async move |ctx| {
                ctx.eval::<(), _>(program)
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;

                for (name, hostfn) in hosts {
                    let hf = hostfn.clone();
                    let f = Function::new(ctx.clone(), move |arg: String| {
                        let hf = hf.clone();
                        Promised(async move {
                            match hf(arg).await {
                                Ok(json) => format!("{{\"value\":{json}}}"),
                                Err(e) => serde_json::json!({ "__error": e }).to_string(),
                            }
                        })
                    })
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;

                    ctx.globals()
                        .set(format!("__{name}"), f)
                        .catch(&ctx)
                        .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                    // Expose the ergonomic wrapper the script actually calls.
                    ctx.eval::<(), _>(format!(
                        "globalThis[{name:?}] = globalThis.__wrap({name:?});"
                    ))
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                }

                let run_entry: Function = ctx
                    .globals()
                    .get("__runEntry")
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                let promise: Promise = run_entry
                    .call((entry, input_json))
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                promise
                    .into_future::<String>()
                    .await
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))
            })
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host<F, Fut>(f: F) -> HostFn
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = HostResult> + Send + 'static,
    {
        Arc::new(move |arg| Box::pin(f(arg)))
    }

    async fn load(dir: &std::path::Path, ts: &str) -> WorkflowRuntime {
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join("workflow.ts");
        std::fs::write(&path, ts).unwrap();
        WorkflowRuntime::load(&path).await.unwrap().unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn composes_bindings_and_forks_concurrently() {
        let dir = std::env::temp_dir().join("ratatoskr-workflow-test-ok");
        let rt = load(
            &dir,
            r#"
            async function run(input: { x: number }) {
                const [a, b] = await Promise.all([slow(input.x), fast(input.x)]);
                const c = await combine({ a, b });
                return { a, b, c };
            }
            "#,
        )
        .await;

        let mut hosts: HashMap<String, HostFn> = HashMap::new();
        hosts.insert(
            "slow".into(),
            host(|arg| async move {
                let x: i64 = serde_json::from_str(&arg).unwrap();
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                Ok((x * 2).to_string())
            }),
        );
        hosts.insert(
            "fast".into(),
            host(|arg| async move {
                let x: i64 = serde_json::from_str(&arg).unwrap();
                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                Ok((x + 1).to_string())
            }),
        );
        hosts.insert(
            "combine".into(),
            host(|arg| async move {
                let v: serde_json::Value = serde_json::from_str(&arg).unwrap();
                let sum = v["a"].as_i64().unwrap() + v["b"].as_i64().unwrap();
                Ok(sum.to_string())
            }),
        );

        let start = std::time::Instant::now();
        let out = rt
            .run("run", serde_json::json!({ "x": 10 }).to_string(), hosts)
            .await
            .unwrap();
        let elapsed = start.elapsed();

        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["a"], 20);
        assert_eq!(v["b"], 11);
        assert_eq!(v["c"], 31);
        // slow(300) ∥ fast(150) via Promise.all → ~300ms, not ~450ms serial.
        assert!(
            elapsed < std::time::Duration::from_millis(430),
            "got {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn host_error_propagates_as_script_error() {
        let dir = std::env::temp_dir().join("ratatoskr-workflow-test-err");
        let rt = load(
            &dir,
            "async function run(input) { return await boom(input); }",
        )
        .await;

        let mut hosts: HashMap<String, HostFn> = HashMap::new();
        hosts.insert(
            "boom".into(),
            host(|_| async move { Err("sandbox unavailable".to_string()) }),
        );

        let err = rt.run("run", "null".to_string(), hosts).await.unwrap_err();
        assert!(
            format!("{err}").contains("sandbox unavailable"),
            "got: {err}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn missing_file_is_none() {
        let path = std::env::temp_dir().join("ratatoskr-workflow-absent/workflow.ts");
        assert!(WorkflowRuntime::load(&path).await.unwrap().is_none());
    }
}
