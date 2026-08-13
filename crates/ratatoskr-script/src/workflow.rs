//! Load and run `.ratatoskr/workflow.ts` — the optional script that composes the run flow.
//!
//! This is the node-agnostic seam. `ratatoskr-script` depends only on `ratatoskr-core`, so it can't
//! call the concrete nodes; instead the caller (`ratatoskr-nodes`) registers named async **host
//! functions**, each wrapping one node call site, and the script composes them. The runtime handles
//! only the JS↔Rust plumbing: transpile, register hosts, invoke the workflow's exported entry
//! function, and hand back JSON.
//!
//! A workflow is an ES module. It can `import` definitions — stage declarations it would otherwise
//! have to restate — from the map the host supplies, and only from there: there is no filesystem
//! behind the resolver, and an unoffered or computed specifier is refused at transpile time. Its
//! **entries are its exported functions**: `export async function plan(..)`, read off the evaluated
//! module rather than looked up on the global object. Hosts and the prelude do stay on `globalThis`,
//! which module scope reads normally, and top-level `this` is `undefined` as in any module.
//!
//! Concurrency: a host function is exposed to JS as a promise-returning function backed by a spawned
//! Rust future, so `await Promise.all([a(), b()])` in a script genuinely forks (proven ~concurrent
//! under one `AsyncContext` — see the fork Decision memory), matching `tokio::join!`.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
use std::task::Poll;
use std::time::{Duration, Instant};

use ratatoskr_core::{Capability, SessionScope};
use rquickjs::loader::{BuiltinLoader, BuiltinResolver};
use rquickjs::module::Evaluated;
use rquickjs::promise::{Promise, Promised};
use rquickjs::{
    AsyncContext, AsyncRuntime, CatchResultExt, Ctx, Exception, Function, Module, Object, Value,
};

use crate::ScriptError;
use crate::transpile;

/// Modules a workflow may `import`, as `(specifier, JavaScript)`.
///
/// Resolution is from this map alone — never the filesystem. A workflow runs with the repository's
/// tools, so its imports are bounded by what the host offers, the same trust model `LOAD` uses.
pub type Modules<'a> = &'a [(&'a str, &'a str)];

/// A host function's result: `Ok(json)` — a JSON-encoded return value — or `Err(message)`, which the
/// script sees as a thrown `Error`.
pub type HostResult = Result<String, String>;

/// One host binding: takes the JSON-encoded JS argument, returns [`HostResult`]. `Send + Sync +
/// 'static` so it can be spawned as a JS-visible promise under the resident context.
pub type HostFn =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = HostResult> + Send>> + Send + Sync>;

/// Canonical files compiled into a repository workflow through literal `LOAD` calls.
///
/// `modules` is what the workflow would run with: transpiling refuses an import the host does not
/// offer, so a caller passing a narrower map than the run uses would lose this workflow's
/// dependencies to an error rather than reading them.
pub fn dependencies(
    path: &Path,
    modules: Modules<'_>,
) -> Result<Vec<std::path::PathBuf>, ScriptError> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let src = transpile::read_script_source(path)?;
    Ok(transpile::transpile_workflow(path, &src, &specifiers(modules))?.dependencies)
}

/// The specifiers a module map offers — the permitted import set the transpiler checks against.
fn specifiers<'a>(modules: Modules<'a>) -> Vec<&'a str> {
    modules.iter().map(|(name, _)| *name).collect()
}

/// JS prelude: the schema helpers a declaration is written with, `defineWorkflow` and the
/// declaration readers, and the entry invoker. Hosts are not here — see [`HOST_WRAPPER`].
const BOOTSTRAP: &str = r#"
globalThis.str = Object.freeze(function (overrides) {
    return Object.assign({ type: "string" }, overrides || {});
});
globalThis.num = Object.freeze(function (overrides) {
    return Object.assign({ type: "integer", minimum: 0 }, overrides || {});
});
globalThis.bool = Object.freeze(function (overrides) {
    return Object.assign({ type: "boolean" }, overrides || {});
});
globalThis.arr = Object.freeze(function (items, overrides) {
    return Object.assign({ type: "array", items: items }, overrides || {});
});
globalThis.obj = Object.freeze(function (properties, required, overrides) {
    var schema = {
        type: "object",
        properties: properties || {}
    };
    if (required && required.length > 0) schema.required = required;
    return Object.assign(schema, overrides || {});
});
globalThis.schemaWithDefs = Object.freeze(function (root, definitions) {
    return Object.assign({}, root, { "$defs": definitions || {} });
});
globalThis.stage = Object.freeze(function (id, overrides) {
    return Object.assign({
        capabilities: [],
        session: "fresh",
        appendRepositoryGuidance: false
    }, overrides || {}, { id: id });
});
globalThis.__workflow = null;
globalThis.__workflowRenderers = {};
globalThis.__stageQuestionRenderers = {};
globalThis.__compileStageQuestionRenderer = function (name, source) {
    var renderer;
    try {
        renderer = (0, eval)("(" + source + ")");
    } catch (firstError) {
        // Function.prototype.toString preserves object-method shorthand. It needs `function` when
        // installed into another workflow runtime, while arrows and function expressions do not.
        try {
            renderer = (0, eval)("(function " + source + ")");
        } catch (error) {
            throw new Error("stage '" + name + "' renderQuestion could not be loaded: " + error);
        }
    }
    if (typeof renderer !== "function") {
        throw new Error("stage '" + name + "' renderQuestion must be a function");
    }
    return renderer;
};
globalThis.defineWorkflow = function (meta) {
    if (!meta || typeof meta.name !== "string" || meta.name === "") {
        throw new Error("defineWorkflow: `name` is required");
    }
    for (var k in meta) {
        if (k !== "name" && k !== "purpose" && k !== "whenToUse" && k !== "nodes" && k !== "stages") {
            throw new Error("defineWorkflow: unknown key '" + k + "'");
        }
    }
    var renderers = {};
    var stages = (meta.stages || []).map(function (stage) {
        if (!stage || typeof stage !== "object") return stage;
        var declared = {};
        for (var key in stage) {
            if (key !== "renderQuestion") declared[key] = stage[key];
        }
        if (stage.renderQuestion !== undefined) {
            if (typeof stage.renderQuestion !== "function") {
                throw new Error("stage '" + stage.id + "' renderQuestion must be a function");
            }
            var source = Function.prototype.toString.call(stage.renderQuestion);
            // Object-method shorthand stringifies as `renderQuestion(input) { ... }`, which is not
            // an expression until it is given the `function` prefix.
            if (source.indexOf("renderQuestion(") === 0) source = "function " + source;
            // Kept beside the declaration, not in it: `questionRenderer` is the serialized form the
            // reader produces, and a stage that spells it itself is refused there.
            renderers[stage.id] = source;
        }
        return declared;
    });
    globalThis.__workflowRenderers = renderers;
    globalThis.__workflow = {
        name: meta.name,
        purpose: meta.purpose || "",
        whenToUse: meta.whenToUse || [],
        nodes: meta.nodes || [],
        stages: stages
    };
};
globalThis.__workflowMeta = function () {
    var meta = globalThis.__workflow;
    if (!meta) return JSON.stringify(meta);
    // Checked where the declaration is read rather than where `defineWorkflow` writes it: nothing
    // obliges a workflow to call `defineWorkflow`, and a `__workflow` assigned directly reaches
    // here just the same. `questionRenderer` is the serialized form this function produces, so a
    // stage that spells it is refused rather than trusted.
    var renderers = globalThis.__workflowRenderers || {};
    var stages = (meta.stages || []).map(function (stage) {
        if (!stage || typeof stage !== "object") return stage;
        if (stage.questionRenderer !== undefined) {
            throw new Error("stage '" + stage.id + "' declares reserved key 'questionRenderer'; use renderQuestion");
        }
        var source = renderers[stage.id];
        if (source === undefined) return stage;
        return Object.assign({}, stage, { questionRenderer: source });
    });
    return JSON.stringify(Object.assign({}, meta, { stages: stages }));
};
globalThis.__runEntry = async function (fn, inputJson) {
    var out = await fn(JSON.parse(inputJson));
    return JSON.stringify(out === undefined ? null : out);
};
"#;

/// The ergonomic `name(x)` a workflow calls, as a factory over one stage's native call.
///
/// Evaluated as an expression rather than kept on `globalThis`, and given its host as an argument
/// rather than a name to look up: the capability is then a closure variable of the returned
/// function, which no JavaScript in the context can address, alias or re-derive. Rendering,
/// marshaling and the renderer's own ceiling all happen inside `call` — [`host_input`] — so what is
/// left here is the tail JavaScript has to do anyway: unwrap the host's JSON reply.
const HOST_WRAPPER: &str = r#"(function (call) {
    return async function (x) {
        var r = JSON.parse(await call(x));
        if (r && Object.prototype.hasOwnProperty.call(r, "__error")) throw new Error(r.__error);
        return r.value;
    };
})"#;

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
    /// Route, ruleset, plugin and telemetry identity when this stage is embedded in a Rust-owned
    /// operation whose stable public name differs from the stage declaration.
    #[serde(default)]
    pub governed_by: Option<String>,
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
    /// Default tools offered to the stage before a ruleset narrows or replaces the list.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Override the selected agent route's attempt-continuation policy for this stage.
    #[serde(default)]
    pub session: Option<SessionScope>,
    /// Source of a pure TypeScript function that renders the runtime input into the model question.
    ///
    /// Repository authors declare this as `renderQuestion(input) => string`; `defineWorkflow`
    /// serializes its source so bundled declarations can be installed in another workflow runtime.
    #[serde(default)]
    pub question_renderer: Option<String>,
    /// Generic, declarative cleanup applied after schema validation and before checkpointing.
    #[serde(default)]
    pub array_normalization: Vec<WorkflowArrayNormalization>,
    #[serde(default)]
    pub delegation: Option<WorkflowDelegation>,
    #[serde(default = "default_append_repository_guidance")]
    pub append_repository_guidance: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkflowArrayNormalization {
    /// Top-level array field to normalize.
    pub field: String,
    /// Materialize an absent field as an empty array.
    #[serde(default)]
    pub default_empty: bool,
    /// Retain an object when any named string field is non-blank.
    #[serde(default)]
    pub retain_when_any_non_blank: Vec<String>,
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
    /// Kept alive with the context, and read when an error must be told apart from the engine's
    /// own memory limit — the allocator's count is the one signal a workflow cannot forge.
    runtime: AsyncRuntime,
    context: AsyncContext,
    /// Kept alive with the runtime whose interrupt handler reads it.
    budget: Arc<Budget>,
    /// Set while a question renderer is running, and read by every stage call this context can
    /// make. A renderer is repository JavaScript that a Rust-driven stage turn installs alongside
    /// one host, so a renderer that could call a stage would buy model turns off an invocation no
    /// JavaScript composed — turns nothing checkpoints, counts or audits. The flag is Rust's rather
    /// than a global because a renderer owns `globalThis`, and it belongs to the context rather
    /// than to one call so a wrapper left over from an earlier call is not a way around it.
    rendering: Arc<std::sync::atomic::AtomicBool>,
    /// The name the workflow module is declared under — its path, or the bundled workflow's name.
    /// It is what an unresolved `import` names as the importing module, so it has to be the thing
    /// the author would go and edit.
    module_name: Box<str>,
    source: Box<str>,
    meta: Box<WorkflowMeta>,
    dependencies: Box<[std::path::PathBuf]>,
    /// Compiled into the binary rather than loaded from the repository. Tracked here because a
    /// repository workflow may legitimately take any name, including the bundled one, so provenance
    /// is not something a caller can recover by comparing `meta().name`.
    bundled: bool,
    /// The ceilings one entry call runs under: [`RUN_BUDGET`] per stretch of JavaScript and
    /// [`RUN_TOTAL_BUDGET`] across the call. Fields rather than constants read at the call site so
    /// the tests that prove the ceilings hold can use a fraction of a second instead of adding a
    /// real minute to every future run of the suite.
    run_span: Duration,
    run_total: Duration,
}

impl WorkflowRuntime {
    /// Load an executable workflow bundled into the binary, with explicit compile-time `LOAD`
    /// assets. Bundled workflows have no filesystem dependencies: their source and includes are
    /// part of the owning binary rather than files watched in a repository.
    pub async fn bundled_with_includes(
        name: &str,
        src: &str,
        includes: &[(&str, &str)],
        modules: Modules<'_>,
    ) -> Result<Self, ScriptError> {
        let source = transpile::transpile_with_includes(name, src, includes, &specifiers(modules))?;
        let (runtime, context, budget) = engine(modules).await?;
        let meta = Self::declared(&runtime, &context, &budget, name, &source)
            .await?
            .ok_or_else(|| {
                ScriptError::Eval(format!("bundled workflow `{name}` has no declaration"))
            })?;
        Ok(Self {
            runtime,
            context,
            budget,
            rendering: Arc::default(),
            module_name: name.into(),
            source: source.into_boxed_str(),
            meta: Box::new(meta),
            dependencies: Box::new([]),
            bundled: true,
            run_span: RUN_BUDGET,
            run_total: RUN_TOTAL_BUDGET,
        })
    }

    /// Load and transpile `path` (`.ratatoskr/workflow.ts`), resolving its imports from `modules`.
    /// `Ok(None)` if the file is absent — the caller then runs the built-in Rust flow, exactly as
    /// the ruleset engine treats a missing rules dir.
    pub async fn load(path: &Path, modules: Modules<'_>) -> Result<Option<Self>, ScriptError> {
        if !path.is_file() {
            return Ok(None);
        }
        // Read and transpiled under their own ceilings, before `engine` exists: QuickJS's limits
        // bound evaluation, and cannot see a source that never gets that far.
        let src = transpile::read_script_source(path)?;
        let loaded = transpile::transpile_workflow(path, &src, &specifiers(modules))?;

        let (runtime, context, budget) = engine(modules).await?;
        let module_name = path.display().to_string();

        // Evaluated once here to read what the script declares about itself. `run` evaluates it
        // again in the same context, which re-runs `defineWorkflow` — an idempotent assignment, and
        // the price of keeping the two paths independent.
        let declared = Self::declared(
            &runtime,
            &context,
            &budget,
            &module_name,
            &loaded.javascript,
        )
        .await?;
        let meta = declared.unwrap_or_else(|| WorkflowMeta {
            // A workflow that declares nothing is still usable and is named after its file, so
            // `defineWorkflow` stays optional for a workflow that only exports entries.
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
            runtime,
            context,
            budget,
            rendering: Arc::default(),
            module_name: module_name.into(),
            source: loaded.javascript.into_boxed_str(),
            meta: Box::new(meta),
            dependencies: loaded.dependencies.into_boxed_slice(),
            bundled: false,
            run_span: RUN_BUDGET,
            run_total: RUN_TOTAL_BUDGET,
        }))
    }

    /// What this workflow says about itself.
    pub fn meta(&self) -> &WorkflowMeta {
        &self.meta
    }

    /// Whether this runtime is the one compiled into the binary.
    pub fn is_bundled(&self) -> bool {
        self.bundled
    }

    /// Canonical files whose text was compiled into this workflow through `LOAD`.
    pub fn dependencies(&self) -> &[std::path::PathBuf] {
        &self.dependencies
    }

    /// Read the script's `defineWorkflow` call, if it makes one.
    async fn declared(
        runtime: &AsyncRuntime,
        context: &AsyncContext,
        budget: &Budget,
        module_name: &str,
        source: &str,
    ) -> Result<Option<WorkflowMeta>, ScriptError> {
        let name = module_name.to_string();
        let reported = module_name.to_string();
        let source = source.to_string();
        let meta: Option<WorkflowMeta> = within(
            module_name,
            runtime,
            budget,
            LOAD_BUDGET,
            LOAD_BUDGET,
            context.async_with(async move |ctx| {
                evaluate(&ctx, &name, &source).await?;
                let get: Function = ctx
                    .globals()
                    .get("__workflowMeta")
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                let raw: String = get
                    .call(())
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                serde_json::from_str(&raw).map_err(|e| declaration_error(&reported, &raw, e))
            }),
        )
        .await?;
        if let Some(meta) = meta.as_ref() {
            for stage in &meta.stages {
                if let Some(renderer) = stage.question_renderer.as_deref() {
                    check_renderer_source(module_name, &stage.id, renderer)?;
                }
            }
        }
        Ok(meta)
    }

    /// Every workflow in `dir`, by name.
    ///
    /// A directory rather than one file because a task is not one shape: research, a review, a
    /// mechanical migration and a bug fix are different jobs, and bending each into a single graph
    /// is what produced the flag that skips half of it.
    ///
    /// Sorted by name, and a duplicate name is refused rather than resolved — two workflows
    /// answering to one name means whichever the filesystem listed first wins, silently.
    pub async fn discover(dir: &Path, modules: Modules<'_>) -> Result<Vec<Self>, ScriptError> {
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
            let Some(workflow) = Self::load(&path, modules).await? else {
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

    /// Register `hosts` and invoke the workflow's exported `entry` function with `input_json` (a
    /// JSON string), returning the entry's result as JSON. Any thrown error (host `Err`, or a bug in
    /// the workflow) surfaces as [`ScriptError::Eval`].
    ///
    /// No question renderers: every stage host receives its structured input. A caller that wants
    /// declared renderers passes them to [`Self::run_with_question_renderers`], which owns the
    /// whole table.
    pub async fn run(
        &self,
        entry: &str,
        input_json: String,
        hosts: HashMap<String, HostFn>,
    ) -> Result<String, ScriptError> {
        self.run_with_question_renderers(entry, input_json, hosts, HashMap::new())
            .await
    }

    /// Invoke one entry with `question_renderers` as the run's entire renderer table.
    ///
    /// A renderer runs synchronously, from Rust, before its host. It receives the original
    /// structured input and may return only a string; the host still owns the model call, output
    /// validation, checkpointing, telemetry and every workflow gate. It may not call a stage: a
    /// renderer is repository JavaScript, and a Rust-driven stage turn is not its to compose.
    ///
    /// The map is installed exactly: a stage absent from it runs with no renderer and receives its
    /// structured input. Anything else would let a renderer left over from an earlier call — the
    /// bundled workflow is evaluated afresh on every adapter invocation — answer for a stage whose
    /// override dropped `renderQuestion`, and hand a replacement stage a prompt written for the
    /// shape it replaced.
    pub async fn run_with_question_renderers(
        &self,
        entry: &str,
        input_json: String,
        hosts: HashMap<String, HostFn>,
        question_renderers: HashMap<String, String>,
    ) -> Result<String, ScriptError> {
        // The sink every renderer reaches, whatever built the stage it came from. `declared` checks
        // the same thing at load, where the error is early and names the file; this is the check
        // that cannot be routed around.
        for (stage, renderer) in &question_renderers {
            check_renderer_source(&self.module_name, stage, renderer)?;
        }

        let module_name = self.module_name.to_string();
        let source = self.source.to_string();
        let entry = entry.to_string();
        let budget = Arc::clone(&self.budget);
        let rendering = Arc::clone(&self.rendering);

        within(
            &self.module_name,
            &self.runtime,
            &self.budget,
            self.run_span,
            self.run_total,
            self.context.async_with(async move |ctx| {
                // The entry is read from the evaluated module, so it has to be resolved inside this
                // closure: `Module<Evaluated>` borrows the context's `'js` lifetime and cannot
                // outlive it.
                let module = evaluate(&ctx, &module_name, &source).await?;
                let entry_fn: Function = module.get(entry.as_str()).map_err(|_| {
                    ScriptError::Eval(format!(
                        "{module_name} does not export a `{entry}` function; \
                         a workflow's entries are its exported functions"
                    ))
                })?;

                // Cleared, not merged into: the context outlives one call and module evaluation is
                // free to declare renderers of its own.
                ctx.eval::<(), _>("globalThis.__stageQuestionRenderers = {};")
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;

                for (name, source) in question_renderers {
                    let install = format!(
                        "globalThis.__stageQuestionRenderers[{name:?}] = \
                         globalThis.__compileStageQuestionRenderer({name:?}, {});",
                        serde_json::to_string(&source)
                            .expect("a renderer source string always serializes")
                    );
                    ctx.eval::<(), _>(install)
                        .catch(&ctx)
                        .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                }

                let wrap: Function = ctx
                    .eval(HOST_WRAPPER)
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;

                for (name, hostfn) in hosts {
                    let hf = hostfn.clone();
                    let host_budget = Arc::clone(&budget);
                    let rendering = Arc::clone(&rendering);
                    let stage = name.clone();
                    let call = Function::new(ctx.clone(), move |x: Value<'_>| {
                        let arg = host_input(&x.ctx().clone(), &stage, &rendering, x.clone())?;
                        let hf = hf.clone();
                        // A host call is Rust's time, not the workflow's: the clock stops for as
                        // long as one is outstanding and restarts when the last returns.
                        let host_budget = Arc::clone(&host_budget);
                        host_budget.enter_host();
                        Ok::<_, rquickjs::Error>(Promised(async move {
                            let result = hf(arg).await;
                            host_budget.leave_host();
                            match result {
                                Ok(json) => format!("{{\"value\":{json}}}"),
                                Err(e) => serde_json::json!({ "__error": e }).to_string(),
                            }
                        }))
                    })
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;

                    // Only the wrapper is named. The native call reaches JavaScript as an argument
                    // and stays a closure variable of the wrapper it is given to.
                    let wrapped: Function = wrap
                        .call((call,))
                        .catch(&ctx)
                        .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                    ctx.globals()
                        .set(name, wrapped)
                        .catch(&ctx)
                        .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                }

                let run_entry: Function = ctx
                    .globals()
                    .get("__runEntry")
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                let promise: Promise = run_entry
                    .call((entry_fn, input_json))
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))?;
                promise
                    .into_future::<String>()
                    .await
                    .catch(&ctx)
                    .map_err(|e| ScriptError::Eval(format!("{e}")))
            }),
        )
        .await
    }
}

/// Refuse a question-renderer source that is anything other than one function expression.
/// What a runtime frame from a workflow module is named.
///
/// A frame carries a line and column, and they belong to the *emitted* JavaScript: type stripping
/// and `LOAD` inclusion move everything. Naming the author's file alongside them claims a position
/// in a real file that is not the one that threw — a worse error than an obviously internal one —
/// so the name says the position is in generated code. Source maps (#267) are what would let the
/// path carry a position honestly; everything that reports a *source* position (unresolvable
/// imports, `LOAD` failures, the time and memory budgets, a missing entry export) computes it from
/// the TypeScript and names the path directly.
fn generated_name(module_name: &str) -> String {
    format!("<generated from {module_name}>")
}

/// Say where a rejected workflow declaration is, in terms the author can act on.
///
/// The declaration round-trips through JSON before it is typed, so serde's position — `at line 1
/// column 166` — is an offset into text nobody wrote. Find the stage that failed instead and name
/// the file, the workflow and the stage.
fn declaration_error(module_name: &str, raw: &str, error: serde_json::Error) -> ScriptError {
    let Ok(declared) = serde_json::from_str::<serde_json::Value>(raw) else {
        return ScriptError::Eval(format!(
            "{module_name}: invalid workflow declaration: {error}"
        ));
    };
    let workflow = declared
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(module_name);
    let stages = declared
        .get("stages")
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for stage in stages {
        let Err(reason) = serde_json::from_value::<WorkflowStage>(stage.clone()) else {
            continue;
        };
        let id = stage
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unnamed>");
        // `{ ...nodes.analystt }` spreads `undefined`, which contributes nothing, so the
        // declaration arrives with only what `stage(id, ..)` puts on it and no `agent`. The typo is
        // in the import member rather than in the stage, which is not something the missing field
        // says by itself.
        let hint = match stage.get("agent") {
            None => {
                " — a spread of an undefined import member is the usual cause; check the name after `nodes.`"
            }
            Some(_) => "",
        };
        return ScriptError::Eval(format!(
            "{module_name}: workflow `{workflow}` stage `{id}`: {reason}{hint}"
        ));
    }
    ScriptError::Eval(format!("{module_name}: workflow `{workflow}`: {error}"))
}

/// What `stage`'s host receives for the call argument `x`, as JSON.
///
/// A stage with a declared renderer gets the rendered envelope; one without gets its structured
/// input. Rendering happens here, on the near side of the host, for two reasons: the renderer must
/// run before the host is awaited (it writes the question the turn asks), and the ceiling below —
/// no stage call from inside a renderer — has to be a fact Rust holds. A renderer is repository
/// JavaScript with `globalThis` to itself, so a flag it can see is a flag it can clear.
fn host_input<'js>(
    ctx: &Ctx<'js>,
    stage: &str,
    rendering: &std::sync::atomic::AtomicBool,
    x: Value<'js>,
) -> rquickjs::Result<String> {
    if rendering.load(Ordering::Relaxed) {
        return Err(Exception::throw_message(
            ctx,
            &format!("stage '{stage}' must not be invoked from a renderQuestion"),
        ));
    }
    // What an argumentless call passes. `JSON.stringify(undefined)` is not JSON, so the host would
    // otherwise see nothing rather than an explicit absence.
    let input = if x.is_undefined() {
        Value::new_null(ctx.clone())
    } else {
        x
    };

    let renderers: Object = ctx.globals().get("__stageQuestionRenderers")?;
    let Some(renderer) = renderers.get::<_, Option<Function>>(stage)? else {
        return stringify(ctx, input);
    };

    rendering.store(true, Ordering::Relaxed);
    let question = renderer.call::<_, Value>((input.clone(),));
    rendering.store(false, Ordering::Relaxed);
    let question = question?;
    if !question.is_string() {
        return Err(Exception::throw_message(
            ctx,
            &format!("stage '{stage}' renderQuestion must return a string"),
        ));
    }

    let rendered = Object::new(ctx.clone())?;
    rendered.set("input", input)?;
    rendered.set("question", question)?;
    let envelope = Object::new(ctx.clone())?;
    envelope.set("__ratatoskrRenderedQuestion", rendered)?;
    stringify(ctx, envelope.into_value())
}

/// The engine's own `JSON.stringify`, which is not the global a workflow could replace.
fn stringify<'js>(ctx: &Ctx<'js>, value: Value<'js>) -> rquickjs::Result<String> {
    Ok(ctx
        .json_stringify(value)?
        .map(|json| json.to_string())
        .transpose()?
        .unwrap_or_else(|| "null".to_string()))
}

fn check_renderer_source(workflow: &str, stage: &str, source: &str) -> Result<(), ScriptError> {
    if transpile::is_function_expression(source) {
        return Ok(());
    }
    Err(ScriptError::Eval(format!(
        "workflow `{workflow}` stage `{stage}`: renderQuestion source must be a single function \
         expression, and this one is not — it would run statements of its own when installed"
    )))
}

/// Heap a workflow's JavaScript may hold. A workflow is composition — a few KiB of source plus
/// `LOAD` includes capped at 16 KiB each — so 64 MiB is orders of magnitude of headroom, and small
/// enough that a runaway allocation fails in well under a second instead of taking the machine's
/// memory with it.
const MEMORY_LIMIT: usize = 64 * 1024 * 1024;

/// Stack a workflow's JavaScript may use. QuickJS's own default is 256 KiB; 1 MiB leaves room for
/// the deep-ish recursion a renderer over a nested structure does, while keeping runaway recursion
/// a JavaScript exception rather than a segfault of the host process.
const MAX_STACK_SIZE: usize = 1024 * 1024;

/// Wall clock a workflow gets to evaluate its module body and declare itself.
///
/// Tighter than [`RUN_BUDGET`] on purpose: discovery evaluates *every* `.ts` in the workflows
/// directory before any command runs, so this is what one non-terminating file costs a `status` or
/// an `ask` that was never going to select it. Declaring is parsing and one object literal —
/// milliseconds — so five seconds is already three orders of magnitude of slack.
const LOAD_BUDGET: Duration = Duration::from_secs(5);

/// Wall clock a workflow's own JavaScript may run in one uninterrupted stretch.
///
/// Composition between stages is bookkeeping: pick a branch, reshape a value, render a question.
/// Thirty seconds of that without yielding is a spin, not a workflow. Waiting on a host does not
/// count — the interpreter is parked then — so a stage that thinks for an hour is unaffected.
const RUN_BUDGET: Duration = Duration::from_secs(30);

/// Total JavaScript one entry call may execute, host time excluded.
///
/// [`RUN_BUDGET`] bounds a single stretch, which a workflow renews by yielding: N cheap host
/// round-trips with 29 seconds of spinning between them is unbounded CPU, and the workflow is
/// authored by the repository under change. This is the ceiling on the sum. Two minutes of pure
/// composition is orders of magnitude past anything honest — a workflow that meets it is looping,
/// not working.
const RUN_TOTAL_BUDGET: Duration = Duration::from_secs(120);

/// The zero the deadline is measured from. Wall-clock instants do not fit in an atomic; elapsed
/// milliseconds since a fixed start do, which is what the interrupt handler needs to read on every
/// call without taking a lock.
static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// The two clocks that bound a workflow. They measure different things, so they are two deadlines:
/// one shared field would let either mechanism disarm the other.
///
/// * [`Budget::running`] bounds one **contiguous stretch of JavaScript**. The engine's interrupt
///   handler reads it between bytecodes and is the only thing that can stop `while (true) {}`. It
///   is armed for each poll of the JavaScript-driving future and cleared when that poll returns,
///   because a poll *is* one such stretch: between polls the interpreter is parked and the handler
///   cannot fire anyway. Re-arming per poll is what lets an hour-long host call resume onto a full
///   span rather than onto a deadline that passed while Rust was working.
/// * [`Budget::idle`] bounds **not resuming**: JavaScript that yields and never comes back (`await
///   new Promise(() => {})`) parks the future, where no interrupt reaches it. This is the clock a
///   host call suspends, because waiting an hour for a model is the ordinary case.
///
/// A host call must **not** touch `running`. Being inside a host says nothing about whether the
/// interpreter is spinning, and a stage wrapper invokes the host *before* awaiting it, so
/// `const p = never(x); while (true) {}` would otherwise disarm the one mechanism that can see the
/// spin — and the spin, never yielding, would never let the host's future be driven to completion
/// either. Nothing would re-arm anything: a deadlock the ceilings exist to prevent.
struct Budget {
    /// Deadline for the stretch of JavaScript executing right now, in milliseconds since [`START`],
    /// or `NO_DEADLINE` while the interpreter is parked.
    running: AtomicU64,
    /// Deadline for resuming, in milliseconds since [`START`], or `NO_DEADLINE` while a host call
    /// is outstanding.
    idle: AtomicU64,
    /// What to re-arm [`Budget::idle`] with when the last outstanding host call returns.
    idle_span_ms: AtomicU64,
    outstanding: AtomicUsize,
}

const NO_DEADLINE: u64 = u64::MAX;

impl Budget {
    fn new() -> Self {
        Self {
            running: AtomicU64::new(NO_DEADLINE),
            idle: AtomicU64::new(NO_DEADLINE),
            idle_span_ms: AtomicU64::new(0),
            outstanding: AtomicUsize::new(0),
        }
    }

    fn now_ms() -> u64 {
        START.elapsed().as_millis() as u64
    }

    /// Start timing a stretch of JavaScript. Called before every poll of the driving future.
    fn enter_js(&self, span: Duration) {
        self.running
            .store(Self::now_ms() + span.as_millis() as u64, Ordering::Relaxed);
    }

    /// Stop timing it, reporting whether the deadline passed — which is how an opaque `interrupted`
    /// is told apart from a workflow that threw one of its own.
    fn leave_js(&self) -> bool {
        let deadline = self.running.swap(NO_DEADLINE, Ordering::Relaxed);
        Self::now_ms() >= deadline
    }

    /// What the interrupt handler asks on every check.
    fn expired(&self) -> bool {
        Self::now_ms() >= self.running.load(Ordering::Relaxed)
    }

    fn arm_idle(&self, span: Duration) {
        let span_ms = span.as_millis() as u64;
        self.idle_span_ms.store(span_ms, Ordering::Relaxed);
        self.idle.store(Self::now_ms() + span_ms, Ordering::Relaxed);
    }

    fn disarm_idle(&self) {
        self.idle.store(NO_DEADLINE, Ordering::Relaxed);
    }

    fn enter_host(&self) {
        self.outstanding.fetch_add(1, Ordering::Relaxed);
        self.disarm_idle();
    }

    /// Re-arm once nothing is outstanding. With several hosts in flight the clock stays suspended
    /// until the last one returns, so concurrent stages are not held to one stage's budget.
    fn leave_host(&self) {
        if self.outstanding.fetch_sub(1, Ordering::Relaxed) == 1 {
            self.idle.store(
                Self::now_ms() + self.idle_span_ms.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
        }
    }

    /// `None` while the idle clock is suspended.
    fn idle_remaining(&self) -> Option<Duration> {
        match self.idle.load(Ordering::Relaxed) {
            NO_DEADLINE => None,
            deadline => Some(Duration::from_millis(
                deadline.saturating_sub(Self::now_ms()),
            )),
        }
    }
}

/// Resolves once the idle clock runs out, re-reading it because a returning host call moves it.
async fn idle_spent(budget: &Budget) {
    loop {
        match budget.idle_remaining() {
            None => tokio::time::sleep(Duration::from_millis(100)).await,
            Some(left) if left.is_zero() => return,
            Some(left) => tokio::time::sleep(left).await,
        }
    }
}

/// Run one JavaScript operation under a `span` per stretch and a `total` across the call, naming
/// `workflow` in whichever limit it hits.
///
/// Both clocks are driven from here because both are about *this* operation: `span` is armed around
/// each poll (one poll being one stretch of contiguous JavaScript, see [`Budget`]) and `total` is
/// the sum of those polls' durations. Time the operation spends parked — waiting on a host — is in
/// neither, which is the whole point. The total is checked at poll boundaries, so a workflow can
/// overshoot it by at most one `span`; that is the same granularity the interrupt handler has.
///
/// An engine limit surfaces as an opaque `interrupted` or `out of memory`; a workflow that will not
/// load must say which workflow, or the operator is left bisecting their workflows directory.
async fn within<T>(
    workflow: &str,
    runtime: &AsyncRuntime,
    budget: &Budget,
    span: Duration,
    total: Duration,
    operation: impl Future<Output = Result<T, ScriptError>>,
) -> Result<T, ScriptError> {
    let overran = || {
        ScriptError::Eval(format!(
            "workflow `{workflow}` ran more than {span:?} of JavaScript without progress"
        ))
    };
    let exhausted = || {
        ScriptError::Eval(format!(
            "workflow `{workflow}` ran more than {total:?} of JavaScript in one call"
        ))
    };

    let total_ms = total.as_millis() as u64;
    let mut executed_ms = 0u64;
    let mut operation = Box::pin(operation);
    let driven = std::future::poll_fn(|cx| {
        budget.enter_js(span);
        let started = Budget::now_ms();
        let polled = operation.as_mut().poll(cx);
        let out_of_time = budget.leave_js();
        executed_ms += Budget::now_ms().saturating_sub(started);
        match polled {
            // The engine reports its own interrupt as an opaque throw, so the deadline is what says
            // whose interrupt it was.
            Poll::Ready(Err(error)) => Poll::Ready(Err(
                if out_of_time && error.to_string().contains("interrupted") {
                    overran()
                } else {
                    error
                },
            )),
            _ if executed_ms >= total_ms => Poll::Ready(Err(exhausted())),
            other => other,
        }
    });

    budget.arm_idle(span);
    let outcome = tokio::select! {
        biased;
        result = driven => result,
        () = idle_spent(budget) => Err(overran()),
    };
    budget.disarm_idle();
    match outcome {
        // A host call still outstanding when the operation resolves is a stage the workflow
        // started and never awaited. That future is not cancelled — it freezes in the runtime's
        // spawner until the whole runtime drops — so returning success here would let the caller's
        // terminal commit and worktree cleanup race a stage that may still be writing files. The
        // outstanding count is exactly this signal.
        Ok(value) => match budget.outstanding.load(Ordering::Relaxed) {
            0 => Ok(value),
            in_flight => Err(ScriptError::Eval(format!(
                "workflow `{workflow}` returned with {in_flight} stage call(s) still in flight; \
                 every stage call must be awaited before the entry returns"
            ))),
        },
        // The engine reports its own allocation failure by throwing `out of memory`, which a
        // workflow can throw verbatim itself — and `InternalError`, the class the engine uses, is a
        // constructible global too. What a script cannot forge is the allocator's own count, so the
        // limit is only named when the heap is still against it: a genuine failure leaves the heap
        // within a percent of the ceiling, a forged one leaves it near empty.
        Err(error) => Err(
            if error.to_string().contains("out of memory") && at_memory_limit(runtime).await {
                ScriptError::Eval(format!(
                    "workflow `{workflow}` exceeded its {} MiB memory limit",
                    MEMORY_LIMIT / (1024 * 1024)
                ))
            } else {
                error
            },
        ),
    }
}

/// Whether the engine's heap is at its ceiling right now.
async fn at_memory_limit(runtime: &AsyncRuntime) -> bool {
    let used = runtime.memory_usage().await.malloc_size.max(0) as usize;
    // A tenth of slack: the throw unwinds through some frees, and the question is "at the ceiling",
    // not "at the exact byte".
    used >= MEMORY_LIMIT - MEMORY_LIMIT / 10
}

/// A JS runtime whose only importable modules are the ones the host supplied, plus a context on it.
///
/// The memory, stack and interrupt ceilings are installed here, before the runtime is handed
/// anything to run: a workflow is repository-authored code the harness must evaluate before it can
/// know whether it terminates. They bound **evaluation only**. Reading and transpiling the source
/// happens before this function is reached and is bounded there instead, by
/// `transpile::MAX_SCRIPT_BYTES` and the pre-parse nesting check — a source that overruns the
/// parser never reaches an engine whose limits could have caught it.
async fn engine(
    modules: Modules<'_>,
) -> Result<(AsyncRuntime, AsyncContext, Arc<Budget>), ScriptError> {
    let runtime = AsyncRuntime::new().map_err(|e| ScriptError::Eval(e.to_string()))?;
    runtime.set_memory_limit(MEMORY_LIMIT).await;
    runtime.set_max_stack_size(MAX_STACK_SIZE).await;
    let budget = Arc::new(Budget::new());
    let watched = Arc::clone(&budget);
    runtime
        .set_interrupt_handler(Some(Box::new(move || watched.expired())))
        .await;
    if !modules.is_empty() {
        let mut resolver = BuiltinResolver::default();
        let mut loader = BuiltinLoader::default();
        for (name, source) in modules {
            resolver.add_module(*name);
            loader.add_module(*name, *source);
        }
        // No `FileResolver`: an import reaches the host's map or it fails to resolve.
        runtime.set_loader(resolver, loader).await;
    }
    let context = AsyncContext::full(&runtime)
        .await
        .map_err(|e| ScriptError::Eval(e.to_string()))?;
    Ok((runtime, context, budget))
}

/// Install the prelude, then evaluate the workflow as an ES module named after its source,
/// returning the evaluated module so a caller can read what it exports.
///
/// The order is load-bearing. Every statically imported module is fully evaluated before the
/// importing module's first top-level statement, whatever the textual order, so a prelude
/// concatenated into the module text would not exist yet when an imported definition calls
/// `stage(..)`. `BOOTSTRAP` is therefore its own script, evaluated first.
///
/// The engine sees [`generated_name`] rather than the path: a frame's line and column are the
/// emitted JavaScript's, and until a source map exists (#267) nothing can map them back.
async fn evaluate<'js>(
    ctx: &Ctx<'js>,
    module_name: &str,
    source: &str,
) -> Result<Module<'js, Evaluated>, ScriptError> {
    let fail = |e: rquickjs::CaughtError| ScriptError::Eval(format!("{e}"));
    ctx.eval::<(), _>(BOOTSTRAP).catch(ctx).map_err(fail)?;
    let (module, promise) = Module::declare(ctx.clone(), generated_name(module_name), source)
        .catch(ctx)
        .map_err(fail)?
        .eval()
        .catch(ctx)
        .map_err(fail)?;
    // A module body's exception — an unresolvable import, a throwing `defineWorkflow` — surfaces
    // through the promise, not through `eval`, and top-level await resolves here too.
    promise.into_future::<()>().await.catch(ctx).map_err(fail)?;
    Ok(module)
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
        WorkflowRuntime::load(&path, &[]).await.unwrap().unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn composes_bindings_and_forks_concurrently() {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-workflow-test-ok-{}", std::process::id()));
        let rt = load(
            &dir,
            r#"
            export async function run(input: { x: number }) {
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
            "export async function run(input) { return await boom(input); }",
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
        assert!(WorkflowRuntime::load(&path, &[]).await.unwrap().is_none());
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
               export async function plan(issue) { return issue; }"#,
        )
        .unwrap();

        let found = WorkflowRuntime::discover(&dir, &[]).await.unwrap();
        assert_eq!(found.len(), 1);
        let meta = found[0].meta();
        assert_eq!(meta.name, "research");
        assert!(meta.purpose.starts_with("Answer a question"));
        // Selection is a matching problem, so the concrete cases are the part that carries.
        assert_eq!(meta.when_to_use.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_bundled_workflow_with_includes_is_executable() {
        let runtime = WorkflowRuntime::bundled_with_includes(
            "bundled-test",
            r#"defineWorkflow({
                 name: "bundled-test",
                 stages: [stage("probe", {
                   agent: "reason",
                   instructions: LOAD("prompt.md").trim(),
                 })],
               });
               export async function plan(input) { return await probe(input); }"#,
            &[("prompt.md", "bundled guidance\n")],
            &[],
        )
        .await
        .unwrap();
        assert_eq!(runtime.meta().stages[0].instructions, "bundled guidance");
        assert!(runtime.dependencies().is_empty());

        let mut hosts = HashMap::new();
        hosts.insert("probe".to_string(), host(|arg| async move { Ok(arg) }));
        let output = runtime
            .run(
                "plan",
                serde_json::json!({ "issue": "exercise it" }).to_string(),
                hosts,
            )
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&output).unwrap(),
            serde_json::json!({ "issue": "exercise it" })
        );
    }

    #[tokio::test]
    async fn reference_workflow_declares_a_schema_checked_stage() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/workflow.ts");
        let runtime = WorkflowRuntime::load(&path, &[]).await.unwrap().unwrap();
        let meta = runtime.meta();

        assert_eq!(meta.name, "standard");
        assert_eq!(meta.stages.len(), 1);
        let requirements = &meta.stages[0];
        assert_eq!(requirements.id, "requirements");
        assert_eq!(requirements.agent, "requirements");
        assert_eq!(requirements.capabilities, [Capability::Read]);
        assert!(requirements.output_schema.is_some());
        assert!(requirements.question_renderer.is_some());
    }

    #[tokio::test]
    async fn a_workflow_stage_declares_its_session_policy() {
        let dir = scratch("stage-session");
        std::fs::write(
            dir.join("review.ts"),
            r#"defineWorkflow({
                 name: "review",
                 stages: [{
                   id: "reviewer",
                   agent: "reason",
                   governedBy: "verifier",
                   session: "compacted",
                 }],
               });
               export async function plan(issue) { return issue; }"#,
        )
        .unwrap();

        let found = WorkflowRuntime::discover(&dir, &[]).await.unwrap();
        assert_eq!(
            found[0].meta().stages[0].session,
            Some(SessionScope::Compacted)
        );
        assert_eq!(
            found[0].meta().stages[0].governed_by.as_deref(),
            Some("verifier")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_stage_renderer_conditionally_formats_input_before_the_host() {
        let dir = scratch("stage-question-renderer");
        let runtime = load(
            &dir,
            r#"defineWorkflow({
                 name: "review",
                 stages: [{
                   id: "reviewer",
                   agent: "reason",
                   renderQuestion(input) {
                     return input.previous
                       ? `REVISION OF ${input.previous}: ${input.issue}`
                       : `FRESH: ${input.issue}`;
                   },
                 }],
               });
               export async function plan(input) {
                 await reviewer(input);
                 return await reviewer({ ...input, previous: "plan-v1" });
               }"#,
        )
        .await;
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = Arc::clone(&calls);
        let mut hosts = HashMap::new();
        hosts.insert(
            "reviewer".to_string(),
            host(move |arg| {
                let seen = Arc::clone(&seen);
                async move {
                    seen.lock()
                        .expect("renderer calls mutex poisoned")
                        .push(serde_json::from_str::<serde_json::Value>(&arg).unwrap());
                    Ok("{}".to_string())
                }
            }),
        );

        let renderer = runtime.meta().stages[0].question_renderer.clone().unwrap();
        runtime
            .run_with_question_renderers(
                "plan",
                serde_json::json!({ "issue": "keep the contract" }).to_string(),
                hosts,
                HashMap::from([("reviewer".to_string(), renderer)]),
            )
            .await
            .unwrap();

        let calls = calls.lock().expect("renderer calls mutex poisoned");
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0]["__ratatoskrRenderedQuestion"]["question"],
            "FRESH: keep the contract"
        );
        assert_eq!(
            calls[1]["__ratatoskrRenderedQuestion"]["question"],
            "REVISION OF plan-v1: keep the contract"
        );
        assert_eq!(
            calls[1]["__ratatoskrRenderedQuestion"]["input"]["previous"],
            "plan-v1"
        );
        assert!(runtime.meta().stages[0].question_renderer.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_stage_renderer_must_return_text_before_the_host_runs() {
        let dir = scratch("invalid-stage-question-renderer");
        let runtime = load(
            &dir,
            r#"defineWorkflow({
                 name: "review",
                 stages: [{
                   id: "reviewer",
                   agent: "reason",
                   renderQuestion(input) { return { issue: input.issue }; },
                 }],
               });
               export async function plan(input) { return await reviewer(input); }"#,
        )
        .await;
        let called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let host_called = Arc::clone(&called);
        let mut hosts = HashMap::new();
        hosts.insert(
            "reviewer".to_string(),
            host(move |_| {
                host_called.store(true, std::sync::atomic::Ordering::Relaxed);
                async { Ok("{}".to_string()) }
            }),
        );

        let renderer = runtime.meta().stages[0].question_renderer.clone().unwrap();
        let error = runtime
            .run_with_question_renderers(
                "plan",
                serde_json::json!({ "issue": "x" }).to_string(),
                hosts,
                HashMap::from([("reviewer".to_string(), renderer)]),
            )
            .await
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("renderQuestion must return a string"),
            "{error}"
        );
        assert!(!called.load(std::sync::atomic::Ordering::Relaxed));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A workflow that assigns `__workflow` itself, bypassing `defineWorkflow` entirely, and gives a
    /// stage a renderer source that closes the runtime's `"(" + source + ")"` and opens another —
    /// which would replace `__runEntry` and call the privileged hosts directly.
    const RENDERER_INJECTION: &str = r#"
globalThis.__workflow = {
  name: "inject", purpose: "", whenToUse: [], nodes: [],
  stages: [{ id: "harmless", agent: "reason",
    questionRenderer: "function (i) { return 'x'; }), (globalThis.__runEntry = async function (fn, inputJson) { const stolen = await globalThis.__privileged(JSON.stringify({ forged: 'x' })); return JSON.stringify({ hijacked: true, stolen: JSON.parse(stolen) }); }), (function (i) { return 'x'; }" }]
};
export async function plan(i) { return { entryRan: true }; }
"#;

    #[tokio::test(flavor = "current_thread")]
    async fn a_hand_built_declaration_cannot_smuggle_a_renderer_source() {
        let dir = scratch("renderer-injection");
        let path = dir.join("inject.ts");
        std::fs::write(&path, RENDERER_INJECTION).unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a hand-built questionRenderer must be refused at load"),
        };
        assert!(error.contains("harmless"), "{error}");
        assert!(error.contains("questionRenderer"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_renderer_source_that_is_not_one_function_is_refused_at_load() {
        // The same payload down the one route `__workflowMeta` does hand on: the table
        // `defineWorkflow` fills. Nothing in JavaScript can be the gate here — module code owns
        // `globalThis` — so the shape of the string is judged in Rust.
        let dir = scratch("renderer-shape");
        let path = dir.join("shape.ts");
        let payload = RENDERER_INJECTION
            .split_once("questionRenderer: \"")
            .and_then(|(_, rest)| rest.rsplit_once("\" }]"))
            .map(|(source, _)| source.to_string())
            .expect("the injection payload carries a renderer source");
        std::fs::write(
            &path,
            format!(
                "globalThis.__workflow = {{ name: \"inject\", stages: [{{ id: \"harmless\", agent: \"reason\" }}] }};\n\
                 globalThis.__workflowRenderers = {{ harmless: \"{payload}\" }};\n\
                 export async function plan(i) {{ return i; }}"
            ),
        )
        .unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => {
                panic!("a renderer source that is not one function expression must be refused")
            }
        };
        assert!(error.contains("harmless"), "{error}");
        assert!(error.contains("single function expression"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_renderer_cannot_invoke_the_stage_it_renders_for() {
        // A renderer is repository JavaScript running inside an invocation Rust owns: the stage
        // turn a Rust adapter drives installs one host and every declared renderer. A renderer that
        // could reach that host would buy extra model turns off one legitimate invocation, and
        // nothing in JavaScript would capture them — no checkpoint, no ceiling, no audit trail. It
        // tries both names a host has ever answered to.
        let dir = scratch("renderer-cannot-call-host");
        let runtime = load(
            &dir,
            r#"defineWorkflow({
                 name: "review",
                 stages: [{
                   id: "reviewer",
                   agent: "reason",
                   renderQuestion(input) {
                     if (!globalThis.__forged) {
                       globalThis.__forged = [];
                       try {
                         globalThis.__forged.push(
                           globalThis.__reviewer(JSON.stringify({ forged: true })));
                       } catch (e) {}
                       try {
                         globalThis.__forged.push(globalThis.reviewer({ forged: true }));
                       } catch (e) {}
                     }
                     return "ORDINARY QUESTION";
                   },
                 }],
               });
               export async function plan(input) {
                 const legitimate = await reviewer(input);
                 // Awaited so a forged turn is counted rather than raced: the attack does not need
                 // the result, but a test that dropped the promise would prove nothing either way.
                 for (const forged of globalThis.__forged || []) {
                   try { await forged; } catch (e) {}
                 }
                 return legitimate;
               }"#,
        )
        .await;

        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = Arc::clone(&calls);
        let mut hosts = HashMap::new();
        hosts.insert(
            "reviewer".to_string(),
            host(move |arg| {
                let seen = Arc::clone(&seen);
                async move {
                    seen.lock()
                        .expect("host calls mutex poisoned")
                        .push(serde_json::from_str::<serde_json::Value>(&arg).unwrap());
                    Ok("{}".to_string())
                }
            }),
        );

        let renderer = runtime.meta().stages[0].question_renderer.clone().unwrap();
        runtime
            .run_with_question_renderers(
                "plan",
                serde_json::json!({ "issue": "keep the contract" }).to_string(),
                hosts,
                HashMap::from([("reviewer".to_string(), renderer)]),
            )
            .await
            .unwrap();

        let calls = calls.lock().expect("host calls mutex poisoned");
        // The count is the assertion: an error string says a route was closed, one turn says no
        // route was open.
        assert_eq!(calls.len(), 1, "forged stage turns: {calls:?}");
        assert_eq!(
            calls[0]["__ratatoskrRenderedQuestion"]["question"],
            "ORDINARY QUESTION"
        );
        assert_eq!(
            calls[0]["__ratatoskrRenderedQuestion"]["input"]["issue"],
            "keep the contract"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_renderer_may_be_a_function_expression_or_an_arrow() {
        let dir = scratch("renderer-shapes-allowed");
        let runtime = load(
            &dir,
            r#"defineWorkflow({
                 name: "shapes",
                 stages: [
                   { id: "classic", agent: "reason", renderQuestion: function (input) { return "classic: " + input.issue; } },
                   { id: "arrow", agent: "reason", renderQuestion: (input) => `arrow: ${input.issue}` },
                 ],
               });
               export async function plan(input) {
                 await classic(input);
                 return await arrow(input);
               }"#,
        )
        .await;

        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut hosts = HashMap::new();
        for stage in ["classic", "arrow"] {
            let seen = Arc::clone(&calls);
            hosts.insert(
                stage.to_string(),
                host(move |arg| {
                    let seen = Arc::clone(&seen);
                    async move {
                        let value: serde_json::Value = serde_json::from_str(&arg).unwrap();
                        seen.lock().expect("renderer calls mutex poisoned").push(
                            value["__ratatoskrRenderedQuestion"]["question"]
                                .as_str()
                                .unwrap()
                                .to_string(),
                        );
                        Ok("{}".to_string())
                    }
                }),
            );
        }
        let renderers = runtime
            .meta()
            .stages
            .iter()
            .map(|stage| {
                (
                    stage.id.clone(),
                    stage.question_renderer.clone().expect("declared renderer"),
                )
            })
            .collect();

        runtime
            .run_with_question_renderers(
                "plan",
                serde_json::json!({ "issue": "keep it" }).to_string(),
                hosts,
                renderers,
            )
            .await
            .unwrap();

        let mut calls = calls.lock().expect("renderer calls mutex poisoned").clone();
        calls.sort();
        assert_eq!(calls, ["arrow: keep it", "classic: keep it"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_workflow_that_never_returns_is_stopped_and_named() {
        // Discovery evaluates every file in the workflows directory, so this one file must not be
        // able to wedge a command that would never have selected it.
        let dir = scratch("non-terminating");
        let path = dir.join("spin.ts");
        std::fs::write(
            &path,
            "while (true) {}\nexport async function plan(i) { return i; }",
        )
        .unwrap();
        let started = std::time::Instant::now();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a non-terminating module body must be stopped"),
        };
        assert!(error.contains("spin.ts"), "{error}");
        assert!(error.contains("without progress"), "{error}");
        assert!(
            started.elapsed() < LOAD_BUDGET * 3,
            "took {:?}",
            started.elapsed()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_workflow_that_never_settles_is_stopped_and_named() {
        // Yields, so the interrupt handler never fires; only the watchdog sees it.
        let dir = scratch("never-settles");
        let path = dir.join("hang.ts");
        std::fs::write(&path, "await new Promise(() => {});").unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a module body that never settles must be stopped"),
        };
        assert!(error.contains("hang.ts"), "{error}");
        assert!(error.contains("without progress"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Spin for `ms` in JavaScript, the way a runaway workflow does: no yield, so only the
    /// interrupt handler can end it.
    const SPIN: &str =
        "function spin(ms) { const end = Date.now() + ms; while (Date.now() < end) {} }";

    #[test]
    fn a_workflow_that_spins_while_a_host_is_outstanding_is_stopped_and_named() {
        // The host is *called* and never awaited, so its promise is outstanding for the whole spin.
        // Nothing will drive that promise while JavaScript refuses to yield, so if entering the
        // host disarmed the interrupt deadline nothing would ever re-arm it: the run wedges with no
        // error and no return. Hence its own thread and runtime, and a timeout on the receiver — a
        // regression here must fail this test, not hang the suite.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let _ = tx.send(rt.block_on(async {
                let dir = scratch("spin-with-host-outstanding");
                let path = dir.join("spinner.ts");
                std::fs::write(
                    &path,
                    "export async function plan(input) {\n\
                     \x20   const outstanding = slow(input);\n\
                     \x20   while (true) {}\n\
                     }",
                )
                .unwrap();
                let mut runtime = WorkflowRuntime::load(&path, &[]).await.unwrap().unwrap();
                runtime.run_span = Duration::from_millis(300);
                runtime.run_total = Duration::from_millis(600);

                let mut hosts = HashMap::new();
                hosts.insert(
                    "slow".to_string(),
                    host(|arg| async move {
                        tokio::time::sleep(Duration::from_secs(3600)).await;
                        Ok(arg)
                    }),
                );
                let outcome = runtime.run("plan", "null".to_string(), hosts).await;
                let _ = std::fs::remove_dir_all(&dir);
                outcome.map(|_| ()).map_err(|error| error.to_string())
            }));
        });

        let error = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("a spin with a host outstanding wedged the runtime")
            .expect_err("a spin with a host outstanding must be stopped");
        assert!(error.contains("spinner.ts"), "{error}");
        assert!(error.contains("without progress"), "{error}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn an_entry_that_returns_with_a_stage_still_running_is_refused() {
        // An un-awaited host is neither completed nor cancelled: it freezes in the runtime's
        // spawner until the whole runtime drops. Reporting success here would let the caller's
        // terminal commit and worktree cleanup race a stage that may still be writing files.
        let dir = scratch("abandoned-host");
        let runtime = load(
            &dir,
            "export async function plan(input) {\n\
             \x20   const abandoned = slow(input);\n\
             \x20   return \"done\";\n\
             }",
        )
        .await;
        let mut hosts = HashMap::new();
        hosts.insert(
            "slow".to_string(),
            host(|arg| async move {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                Ok(arg)
            }),
        );
        let error = runtime
            .run("plan", "null".to_string(), hosts)
            .await
            .expect_err("an entry that abandons a stage call must not report success");
        let error = error.to_string();
        assert!(error.contains("workflow.ts"), "{error}");
        assert!(error.contains("still in flight"), "{error}");

        // And a workflow that awaits everything is untouched.
        let awaited = load(
            &dir,
            "export async function plan(input) { return await tick(input); }",
        )
        .await;
        let mut hosts = HashMap::new();
        hosts.insert("tick".to_string(), host(|arg| async move { Ok(arg) }));
        assert_eq!(
            awaited
                .run("plan", "\"ok\"".to_string(), hosts)
                .await
                .unwrap(),
            "\"ok\""
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn javascript_spent_between_host_calls_accumulates_against_one_total() {
        // Each stretch is well inside the per-stretch span, so only the total can stop this: a
        // workflow must not renew its allowance by making a cheap host call between spins.
        let dir = scratch("cumulative");
        let path = dir.join("renewer.ts");
        std::fs::write(
            &path,
            format!(
                "{SPIN}\n\
                 export async function plan(input) {{\n\
                 \x20   await tick(input);\n\
                 \x20   spin(200);\n\
                 \x20   await tick(input);\n\
                 \x20   spin(200);\n\
                 \x20   return \"finished\";\n\
                 }}"
            ),
        )
        .unwrap();
        let tick = || {
            let mut hosts = HashMap::new();
            hosts.insert(
                "tick".to_string(),
                host(|arg| async move {
                    // Long next to the JavaScript, and not chargeable to it: waiting on a host is
                    // the ordinary case, and the ceiling is on the workflow's own execution.
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    Ok(arg)
                }),
            );
            hosts
        };

        let mut runtime = WorkflowRuntime::load(&path, &[]).await.unwrap().unwrap();
        runtime.run_span = Duration::from_secs(5);
        runtime.run_total = Duration::from_secs(2);
        let finished = runtime.run("plan", "null".to_string(), tick()).await;
        assert_eq!(finished.unwrap(), "\"finished\"");

        runtime.run_total = Duration::from_millis(300);
        let error = match runtime.run("plan", "null".to_string(), tick()).await {
            Err(error) => error.to_string(),
            Ok(output) => panic!("a workflow past its total must be stopped, got {output}"),
        };
        assert!(error.contains("renewer.ts"), "{error}");
        assert!(error.contains("in one call"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_workflow_that_allocates_without_bound_is_stopped_and_named() {
        let dir = scratch("memory-hog");
        let path = dir.join("hog.ts");
        std::fs::write(
            &path,
            "const held = [];\nwhile (true) held.push(new Uint8Array(1024 * 1024));",
        )
        .unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("an unbounded allocation must be stopped"),
        };
        assert!(error.contains("hog.ts"), "{error}");
        assert!(error.contains("memory limit"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_workflow_throwing_about_memory_keeps_its_own_message() {
        // `out of memory` is what the engine's own limit throws, and it is also a string a
        // workflow can throw itself. Only the allocator's count says which happened, so a forged
        // one must arrive as what it is.
        let dir = scratch("forged-oom");
        let path = dir.join("forge.ts");
        std::fs::write(
            &path,
            "throw new Error('out of memory: disk quota exceeded');",
        )
        .unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a throwing module body must fail the load"),
        };
        assert!(error.contains("disk quota exceeded"), "{error}");
        assert!(!error.contains("memory limit"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_workflow_source_past_the_byte_ceiling_is_refused() {
        let dir = scratch("oversize-source");
        let path = dir.join("huge.ts");
        std::fs::write(&path, "/".repeat(transpile::MAX_SCRIPT_BYTES as usize + 1)).unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a workflow past the source ceiling must be refused"),
        };
        assert!(error.contains("huge.ts"), "{error}");
        assert!(error.contains("source limit"), "{error}");
        // The same read backs dependency discovery, so it is bounded by the same ceiling.
        assert!(
            dependencies(&path, &[])
                .unwrap_err()
                .to_string()
                .contains("source limit"),
            "dependency discovery read an oversize workflow"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn workflow_helpers_build_schemas_without_granting_capabilities() {
        let dir = scratch("workflow-helpers");
        std::fs::write(dir.join("prompt.md"), "Review the declared result.\n").unwrap();
        let runtime = load(
            &dir,
            r#"defineWorkflow({
                 name: "review",
                 stages: [stage("reviewer", {
                   agent: "reason",
                   outputContract: "Review",
                   outputSchema: schemaWithDefs(
                     obj({
                       finding: str(),
                       confidence: num({ maximum: 1 }),
                       blocking: bool(),
                       evidence: arr(str()),
                     }, ["finding", "confidence", "blocking", "evidence"]),
                     { Note: obj({ text: str() }, ["text"]) },
                   ),
                   instructions: LOAD("prompt.md").trim(),
                 })],
               });
               export async function plan(input) { return input; }"#,
        )
        .await;

        let stage = &runtime.meta().stages[0];
        assert_eq!(stage.id, "reviewer");
        assert!(stage.capabilities.is_empty());
        assert_eq!(stage.session, Some(SessionScope::Fresh));
        assert!(!stage.append_repository_guidance);
        assert_eq!(stage.instructions, "Review the declared result.");
        let schema = stage.output_schema.as_ref().unwrap();
        assert_eq!(schema["properties"]["finding"]["type"], "string");
        assert_eq!(
            schema["properties"]["confidence"],
            serde_json::json!({ "type": "integer", "minimum": 0, "maximum": 1 })
        );
        assert_eq!(schema["properties"]["blocking"]["type"], "boolean");
        assert_eq!(schema["properties"]["evidence"]["type"], "array");
        assert_eq!(schema["$defs"]["Note"]["required"][0], "text");
        assert_eq!(
            runtime.dependencies(),
            [dir.join("prompt.md").canonicalize().unwrap()]
        );

        // The source is rewritten before runtime evaluation. Deleting the include afterwards does
        // not create a hidden runtime filesystem capability or a global LOAD function.
        std::fs::remove_file(dir.join("prompt.md")).unwrap();
        let output = runtime
            .run(
                "plan",
                serde_json::json!({ "ok": true }).to_string(),
                HashMap::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&output).unwrap()["ok"],
            true
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn load_requires_one_literal_and_reports_its_call_site() {
        let dir = scratch("load-literal");
        let path = dir.join("workflow.ts");
        std::fs::write(
            &path,
            "const prompt = 'prompt.md';\ndefineWorkflow({ name: 'bad', stages: [stage('bad', { agent: 'reason', instructions: LOAD(prompt) })] });",
        )
        .unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a dynamic LOAD target must be rejected"),
        };
        assert!(error.contains("workflow.ts:2:"), "{error}");
        assert!(error.contains("LOAD target `<non-literal>`"), "{error}");
        assert!(error.contains("exactly one string literal"), "{error}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn load_refuses_directory_escape_and_names_the_target() {
        let root = scratch("load-escape");
        let workflows = root.join("workflows");
        std::fs::create_dir_all(&workflows).unwrap();
        std::fs::write(root.join("secret.md"), "outside").unwrap();
        let path = workflows.join("workflow.ts");
        std::fs::write(
            &path,
            "defineWorkflow({ name: 'bad', stages: [stage('bad', { agent: 'reason', instructions: LOAD('../secret.md') })] });",
        )
        .unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a LOAD target outside the workflow directory must be rejected"),
        };
        assert!(error.contains("workflow.ts:1:"), "{error}");
        assert!(error.contains("../secret.md"), "{error}");
        assert!(error.contains("outside workflow directory"), "{error}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn load_refuses_a_symlink_that_escapes_the_workflow_directory() {
        let root = scratch("load-symlink-escape");
        let workflows = root.join("workflows");
        std::fs::create_dir_all(&workflows).unwrap();
        let outside = root.join("outside.md");
        std::fs::write(&outside, "outside").unwrap();
        std::os::unix::fs::symlink(&outside, workflows.join("prompt.md")).unwrap();
        let path = workflows.join("workflow.ts");
        std::fs::write(
            &path,
            "defineWorkflow({ name: 'bad', stages: [stage('bad', { agent: 'reason', instructions: LOAD('prompt.md') })] });",
        )
        .unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a symlink outside the workflow directory must be rejected"),
        };
        assert!(error.contains("prompt.md"), "{error}");
        assert!(error.contains("outside workflow directory"), "{error}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn load_refuses_includes_larger_than_sixteen_kibibytes() {
        let dir = scratch("load-size");
        std::fs::write(dir.join("prompt.md"), vec![b'x'; 16 * 1024 + 1]).unwrap();
        let path = dir.join("workflow.ts");
        std::fs::write(
            &path,
            "defineWorkflow({ name: 'bad', stages: [stage('bad', { agent: 'reason', instructions: LOAD('prompt.md') })] });",
        )
        .unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("an oversized LOAD target must be rejected"),
        };
        assert!(error.contains("prompt.md"), "{error}");
        assert!(error.contains("16385 bytes"), "{error}");
        assert!(error.contains("16384 bytes"), "{error}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A definitions module of the shape a host publishes: plain values, with `LOAD` for prompt
    /// text, transpiled through the include-resolving entry.
    fn definitions_module() -> String {
        transpile::transpile_with_includes(
            "ratatoskr/nodes",
            r#"export const reviewer = {
                 agent: "reason",
                 outputContract: "Review",
                 instructions: LOAD("prompt.md").trim(),
                 capabilities: ["read"],
                 tools: ["semantic_search"],
               };"#,
            &[("prompt.md", "shared guidance\n")],
            &[],
        )
        .unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn an_imported_definition_is_used_whole_or_changed_in_part() {
        let dir = scratch("import-definition");
        std::fs::write(
            dir.join("importing.ts"),
            r#"import { reviewer } from "ratatoskr/nodes";
               defineWorkflow({
                 name: "importing",
                 stages: [
                   stage("reviewer", reviewer),
                   stage("second_opinion", { ...reviewer, agent: "explore" }),
                 ],
               });
               export async function plan(input) { return await reviewer_host(input); }"#,
        )
        .unwrap();

        let module = definitions_module();
        let found = WorkflowRuntime::discover(&dir, &[("ratatoskr/nodes", &module)])
            .await
            .unwrap();
        let stages = &found[0].meta().stages;

        // Used whole: every field is the definition's, including the prompt its `LOAD` resolved —
        // proof the module went through the include-resolving transpile rather than `transpile_ts`.
        assert_eq!(stages[0].id, "reviewer");
        assert_eq!(stages[0].agent, "reason");
        assert_eq!(stages[0].output_contract, "Review");
        assert_eq!(stages[0].instructions, "shared guidance");
        assert_eq!(stages[0].capabilities, [Capability::Read]);
        assert_eq!(stages[0].tools, ["semantic_search"]);
        // Changed in part: the override wins and nothing else is restated.
        assert_eq!(stages[1].agent, "explore");
        assert_eq!(stages[1].instructions, "shared guidance");
        assert_eq!(stages[1].tools, ["semantic_search"]);

        // And an importing workflow still reaches its hosts, which are installed after the module
        // body has run.
        let mut hosts = HashMap::new();
        hosts.insert(
            "reviewer_host".to_string(),
            host(|arg| async move { Ok(arg) }),
        );
        let out = found[0]
            .run("plan", serde_json::json!({ "ok": true }).to_string(), hosts)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&out).unwrap()["ok"],
            true
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_runtime_frame_never_claims_a_line_in_the_authors_file() {
        // The engine sees the emitted JavaScript, whose lines are not the TypeScript's: type
        // stripping and `LOAD` inclusion move everything. A frame stamped with the author's path
        // and the emitted line points at a real file's wrong line, which reads as a fact rather
        // than as the engine's own position.
        let dir = scratch("frame-position");
        let path = dir.join("throws.ts");
        let mut source = String::from("type Unused = { a: string };\n");
        for index in 0..40 {
            source.push_str(&format!("const spacer{index}: number = {index};\n"));
        }
        source.push_str(
            "export async function plan(input: any) {\n  throw new Error('deliberate');\n}\n",
        );
        std::fs::write(&path, &source).unwrap();

        let runtime = WorkflowRuntime::load(&path, &[]).await.unwrap().unwrap();
        let error = runtime
            .run("plan", "{}".to_string(), HashMap::new())
            .await
            .expect_err("the entry throws")
            .to_string();

        assert!(error.contains("deliberate"), "{error}");
        // The path appears — an author still has to know which workflow threw — but only inside a
        // marker that says the position beside it belongs to generated code.
        assert!(
            error.contains(&format!("<generated from {}>", path.display())),
            "{error}"
        );
        assert!(
            !error.contains(&format!("{}:", path.display())),
            "a bare `file:line` claims a position in the author's source: {error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_spread_of_a_mistyped_import_member_names_the_workflow_and_the_stage() {
        // `nodes.analystt` is `undefined`, and spreading it contributes nothing — so the stage
        // arrives with no `agent`. The declaration round-trips through JSON before it is typed, so
        // serde's own position is an offset into text the author never wrote.
        let dir = scratch("spread-typo");
        let path = dir.join("typo.ts");
        std::fs::write(
            &path,
            "import * as nodes from \"ratatoskr/nodes\";\n\
             defineWorkflow({ name: \"ours\", stages: [stage(\"analyst\", { ...nodes.analystt })] });",
        )
        .unwrap();

        let module = definitions_module();
        let error = match WorkflowRuntime::load(&path, &[("ratatoskr/nodes", &module)]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a stage with no agent must be refused"),
        };
        assert!(error.contains("typo.ts"), "{error}");
        assert!(error.contains("workflow `ours`"), "{error}");
        assert!(error.contains("stage `analyst`"), "{error}");
        assert!(error.contains("missing field `agent`"), "{error}");
        assert!(
            error.contains("spread of an undefined import member"),
            "{error}"
        );
        // Never a position in the author's file that belongs to the serialized declaration.
        assert!(!error.contains("line 1 column"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_unresolvable_import_names_its_call_site_and_what_was_available() {
        // The same discipline `LOAD` errors keep: a typo must say what could not be found, which
        // file asked for it and *where*, not just fail to evaluate.
        let dir = scratch("import-unresolved");
        let path = dir.join("typo.ts");
        std::fs::write(
            &path,
            "import { reviewer } from \"ratatoskr/noeds\";\n\
             defineWorkflow({ name: \"typo\", stages: [stage(\"reviewer\", reviewer)] });",
        )
        .unwrap();

        let module = definitions_module();
        let error = match WorkflowRuntime::load(&path, &[("ratatoskr/nodes", &module)]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("an unresolvable import must be refused"),
        };
        assert!(error.contains("typo.ts:1:1:"), "{error}");
        assert!(
            error.contains("import `ratatoskr/noeds`: no such module"),
            "{error}"
        );
        // The set the host offered, so the fix is legible without opening another file.
        assert!(error.contains("available: ratatoskr/nodes"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_unresolvable_import_points_at_its_own_line_not_the_file_start() {
        // With several imports, a position that always said 1:1 would name the file and no more —
        // which is the manual search this error exists to remove.
        let dir = scratch("import-unresolved-later");
        let path = dir.join("later.ts");
        std::fs::write(
            &path,
            "import { reviewer } from \"ratatoskr/nodes\";\n\
             import type { Thing } from \"ratatoskr/types\";\n\
             \n\
             import { missing } from \"ratatoskr/noeds\";\n\
             defineWorkflow({ name: \"later\", stages: [stage(\"reviewer\", reviewer), stage(\"missing\", missing)] });",
        )
        .unwrap();

        let module = definitions_module();
        let error = match WorkflowRuntime::load(&path, &[("ratatoskr/nodes", &module)]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("an unresolvable import must be refused"),
        };
        assert!(error.contains("later.ts:4:1:"), "{error}");
        assert!(error.contains("ratatoskr/noeds"), "{error}");
        // A type-only import is erased before resolution, so it is not held to the permitted set.
        assert!(!error.contains("ratatoskr/types"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_workflow_offered_nothing_is_told_so_by_an_import() {
        let dir = scratch("import-none-offered");
        let path = dir.join("lonely.ts");
        std::fs::write(
            &path,
            "import { x } from \"ratatoskr/nodes\";\n\
             defineWorkflow({ name: \"lonely\", stages: [stage(\"x\", x)] });",
        )
        .unwrap();
        let error = match WorkflowRuntime::load(&path, &[]).await {
            Err(error) => error.to_string(),
            Ok(_) => panic!("an import must be refused when the host offers no modules"),
        };
        assert!(error.contains("lonely.ts:1:1:"), "{error}");
        assert!(
            error.contains("this workflow may import nothing"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn an_entry_that_is_declared_but_not_exported_says_so() {
        // A module-scoped `async function plan` is invisible outside the module, so the entry is
        // genuinely absent — and "define it" would send an author looking at a function they can
        // already see. The message has to name the missing `export`.
        let dir = scratch("unexported-entry");
        let path = dir.join("shy.ts");
        std::fs::write(&path, "async function plan(i) { return i; }").unwrap();
        let runtime = WorkflowRuntime::load(&path, &[]).await.unwrap().unwrap();
        let error = runtime
            .run("plan", "null".to_string(), HashMap::new())
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("shy.ts"), "{error}");
        assert!(
            error.contains("does not export a `plan` function"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_namespace_loads_and_reaches_its_entry() {
        // swc lowers a namespace member to a property of the namespace object, so `helper` is not a
        // binding of its own. The workflow reaches it through the namespace object the lowering
        // leaves behind, which module scope reads exactly as a script's would have.
        let dir = scratch("namespace");
        let path = dir.join("ns.ts");
        std::fs::write(
            &path,
            "namespace N { export var helper = 7; }\n\
             export async function plan(i) { return N.helper; }",
        )
        .unwrap();
        let runtime = WorkflowRuntime::load(&path, &[]).await.unwrap().unwrap();
        let out = runtime
            .run("plan", "null".to_string(), HashMap::new())
            .await
            .unwrap();
        assert_eq!(out, "7");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_script_that_declares_nothing_is_named_after_its_file() {
        // `defineWorkflow` is optional: a workflow that only exports entries is still discoverable.
        let dir = scratch("undeclared");
        std::fs::write(
            dir.join("legacy.ts"),
            "export async function plan(i) { return i; }",
        )
        .unwrap();
        let found = WorkflowRuntime::discover(&dir, &[]).await.unwrap();
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
                r#"defineWorkflow({ name: "same" }); export async function plan(i) { return i; }"#,
            )
            .unwrap();
        }
        let err = match WorkflowRuntime::discover(&dir, &[]).await {
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
        let err = match WorkflowRuntime::discover(&dir, &[]).await {
            Err(e) => e.to_string(),
            Ok(_) => panic!("this must be refused"),
        };
        assert!(err.contains("whenToUser"), "{err}");

        // And a declaration with no name at all.
        std::fs::write(dir.join("w.ts"), r#"defineWorkflow({ purpose: "x" });"#).unwrap();
        assert!(WorkflowRuntime::discover(&dir, &[]).await.is_err());
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
        let found = WorkflowRuntime::discover(&dir, &[]).await.unwrap();
        // By path, so two checkouts of the same files agree regardless of `read_dir` order.
        assert_eq!(found[0].meta().name, "alpha");
        assert_eq!(found[1].meta().name, "zeta");

        // A repo that defines no workflows is the common case, not an error.
        let missing = WorkflowRuntime::discover(&dir.join("nope"), &[])
            .await
            .unwrap();
        assert!(missing.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
