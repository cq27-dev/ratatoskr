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

use ratatoskr_core::Capability;
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
globalThis.__workflow = null;
globalThis.defineWorkflow = function (meta) {
    if (!meta || typeof meta.name !== "string" || meta.name === "") {
        throw new Error("defineWorkflow: `name` is required");
    }
    for (var k in meta) {
        if (k !== "name" && k !== "purpose" && k !== "whenToUse" && k !== "nodes" && k !== "stages") {
            throw new Error("defineWorkflow: unknown key '" + k + "'");
        }
    }
    globalThis.__workflow = {
        name: meta.name,
        purpose: meta.purpose || "",
        whenToUse: meta.whenToUse || [],
        nodes: meta.nodes || [],
        stages: meta.stages || []
    };
};
globalThis.__workflowMeta = function () {
    return JSON.stringify(globalThis.__workflow);
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

/// What a workflow says about itself, so something can choose between workflows without running
/// one to find out whether it fits.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkflowMeta {
    /// How it is named on the command line and in a checkpoint.
    pub name: String,
    /// What it does, in a sentence.
    #[serde(default)]
    pub purpose: String,
    /// What a task looks like when this is the right workflow for it. Read by whatever selects;
    /// concrete cases beat an abstract description, because selection is a matching problem.
    #[serde(default, rename = "whenToUse")]
    pub when_to_use: Vec<String>,
    /// Nodes this workflow governs beyond the built-in set, so a ruleset targeting one is accepted
    /// rather than read as a typo.
    #[serde(default)]
    pub nodes: Vec<String>,
    /// Ordered user-defined stages. Their contracts are checked at startup before a run starts.
    #[serde(default)]
    pub stages: Vec<WorkflowStage>,
}

/// A serializable stage declaration kept in the script crate so workflow metadata remains independent
/// of concrete node implementations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkflowStage {
    #[serde(alias = "name")]
    pub id: String,
    pub agent: String,
    #[serde(default, alias = "input")]
    pub input_contract: String,
    #[serde(default, alias = "output")]
    pub output_contract: String,
    /// JSON Schema that enforces the declared output contract at the model and checkpoint gates.
    #[serde(default)]
    pub output_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub instructions: String,
    #[serde(default)]
    pub context: String,
    #[serde(default)]
    pub capabilities: Vec<Capability>,
    #[serde(default)]
    pub delegation: Option<WorkflowDelegation>,
    #[serde(default = "default_append_repository_guidance")]
    pub append_repository_guidance: bool,
}

fn default_append_repository_guidance() -> bool {
    true
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkflowDelegation {
    pub target: String,
    #[serde(default)]
    pub evidence_contract: String,
    #[serde(default)]
    pub input_limit: usize,
}

/// A loaded workflow script: the resident JS context plus the transpiled source.
pub struct WorkflowRuntime {
    _runtime: AsyncRuntime,
    context: AsyncContext,
    source: String,
    meta: WorkflowMeta,
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

        // Evaluated once here to read what the script declares about itself. `run` evaluates it
        // again in the same context, which re-runs `defineWorkflow` — an idempotent assignment, and
        // the price of keeping the two paths independent.
        let declared = Self::declared(&context, &source).await?;
        let meta = declared.unwrap_or_else(|| WorkflowMeta {
            // A script that declares nothing is still usable and is named after its file. This is
            // what keeps a repo's existing `workflow.ts` working with no edit.
            name: path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "workflow".to_string()),
            purpose: String::new(),
            when_to_use: Vec::new(),
            nodes: Vec::new(),
            stages: Vec::new(),
        });

        Ok(Some(WorkflowRuntime {
            _runtime: runtime,
            context,
            source,
            meta,
        }))
    }

    /// What this workflow says about itself.
    pub fn meta(&self) -> &WorkflowMeta {
        &self.meta
    }

    /// Read the script's `defineWorkflow` call, if it makes one.
    async fn declared(
        context: &AsyncContext,
        source: &str,
    ) -> Result<Option<WorkflowMeta>, ScriptError> {
        let program = format!("{BOOTSTRAP}\n{source}");
        context
            .async_with(async move |ctx| {
                ctx.eval::<(), _>(program)
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                let get: Function = ctx
                    .globals()
                    .get("__workflowMeta")
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                let raw: String = get
                    .call(())
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                serde_json::from_str(&raw).map_err(|e| ScriptError::Eval(e.to_string()))
            })
            .await
    }

    /// Every workflow in `dir`, by name.
    ///
    /// A directory rather than one file because a task is not one shape: research, a review, a
    /// mechanical migration and a bug fix are different jobs, and bending each into a single graph
    /// is what produced the flag that skips half of it.
    ///
    /// Sorted by name, and a duplicate name is refused rather than resolved — two workflows
    /// answering to one name means whichever the filesystem listed first wins, silently.
    pub async fn discover(dir: &Path) -> Result<Vec<Self>, ScriptError> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(Vec::new());
        };
        let mut paths: Vec<std::path::PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "ts"))
            .collect();
        paths.sort();

        let mut found: Vec<Self> = Vec::new();
        for path in paths {
            let Some(workflow) = Self::load(&path).await? else {
                continue;
            };
            if let Some(clash) = found.iter().find(|w| w.meta.name == workflow.meta.name) {
                return Err(ScriptError::Eval(format!(
                    "two workflows are both named `{}`: {} and {}",
                    workflow.meta.name,
                    clash.meta.name,
                    path.display()
                )));
            }
            found.push(workflow);
        }
        Ok(found)
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
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-workflow-test-ok-{}", std::process::id()));
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
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-workflow-test-err-{}",
            std::process::id()
        ));
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

    fn scratch(case: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ratatoskr-wf-{}-{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn a_workflow_declares_what_it_is_for() {
        let dir = scratch("declared");
        std::fs::write(
            dir.join("research.ts"),
            r#"defineWorkflow({
                 name: "research",
                 purpose: "Answer a question about the repository without changing it.",
                 whenToUse: ["the task asks what or why", "no code change is expected"],
               });
               async function plan(issue) { return issue; }"#,
        )
        .unwrap();

        let found = WorkflowRuntime::discover(&dir).await.unwrap();
        assert_eq!(found.len(), 1);
        let meta = found[0].meta();
        assert_eq!(meta.name, "research");
        assert!(meta.purpose.starts_with("Answer a question"));
        // Selection is a matching problem, so the concrete cases are the part that carries.
        assert_eq!(meta.when_to_use.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn reference_workflow_declares_a_schema_checked_stage() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/workflow.ts");
        let runtime = WorkflowRuntime::load(&path).await.unwrap().unwrap();
        let meta = runtime.meta();

        assert_eq!(meta.name, "standard");
        assert_eq!(meta.stages.len(), 1);
        let requirements = &meta.stages[0];
        assert_eq!(requirements.id, "requirements");
        assert_eq!(requirements.agent, "requirements");
        assert_eq!(requirements.capabilities, [Capability::Read]);
        assert!(requirements.output_schema.is_some());
    }

    #[tokio::test]
    async fn a_script_that_declares_nothing_is_named_after_its_file() {
        // What keeps a repo's existing `workflow.ts` working with no edit.
        let dir = scratch("undeclared");
        std::fs::write(
            dir.join("legacy.ts"),
            "async function plan(i) { return i; }",
        )
        .unwrap();
        let found = WorkflowRuntime::discover(&dir).await.unwrap();
        assert_eq!(found[0].meta().name, "legacy");
        assert!(found[0].meta().purpose.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn two_workflows_with_one_name_is_refused() {
        // Otherwise whichever the filesystem listed first wins, silently, and `--workflow` picks a
        // shape nobody chose.
        let dir = scratch("clash");
        for file in ["a.ts", "b.ts"] {
            std::fs::write(
                dir.join(file),
                r#"defineWorkflow({ name: "same" }); async function plan(i) { return i; }"#,
            )
            .unwrap();
        }
        let err = match WorkflowRuntime::discover(&dir).await {
            Err(e) => e.to_string(),
            Ok(_) => panic!("this must be refused"),
        };
        assert!(err.contains("both named `same`"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_misspelled_declaration_key_is_refused() {
        // Same reason the config structs refuse unknown fields: a typo that silently declared
        // nothing would leave the workflow unselectable with no indication why.
        let dir = scratch("typo");
        std::fs::write(
            dir.join("w.ts"),
            r#"defineWorkflow({ name: "w", whenToUser: ["x"] });"#,
        )
        .unwrap();
        let err = match WorkflowRuntime::discover(&dir).await {
            Err(e) => e.to_string(),
            Ok(_) => panic!("this must be refused"),
        };
        assert!(err.contains("whenToUser"), "{err}");

        // And a declaration with no name at all.
        std::fs::write(dir.join("w.ts"), r#"defineWorkflow({ purpose: "x" });"#).unwrap();
        assert!(WorkflowRuntime::discover(&dir).await.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn discovery_is_ordered_and_an_absent_directory_is_ordinary() {
        let dir = scratch("order");
        for (file, name) in [("z.ts", "zeta"), ("a.ts", "alpha")] {
            std::fs::write(
                dir.join(file),
                format!(r#"defineWorkflow({{ name: "{name}" }});"#),
            )
            .unwrap();
        }
        let found = WorkflowRuntime::discover(&dir).await.unwrap();
        // By path, so two checkouts of the same files agree regardless of `read_dir` order.
        assert_eq!(found[0].meta().name, "alpha");
        assert_eq!(found[1].meta().name, "zeta");

        // A repo that defines no workflows is the common case, not an error.
        let missing = WorkflowRuntime::discover(&dir.join("nope")).await.unwrap();
        assert!(missing.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
