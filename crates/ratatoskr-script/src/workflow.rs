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
use std::time::{Duration, Instant};

use ratatoskr_core::{Capability, SessionScope};
use rquickjs::loader::{BuiltinLoader, BuiltinResolver};
use rquickjs::module::Evaluated;
use rquickjs::promise::{Promise, Promised};
use rquickjs::{AsyncContext, AsyncRuntime, CatchResultExt, Ctx, Function, Module};

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
    let src = std::fs::read_to_string(path)
        .map_err(|error| ScriptError::Io(path.display().to_string(), error))?;
    Ok(transpile::transpile_workflow(path, &src, &specifiers(modules))?.dependencies)
}

/// The specifiers a module map offers — the permitted import set the transpiler checks against.
fn specifiers<'a>(modules: Modules<'a>) -> Vec<&'a str> {
    modules.iter().map(|(name, _)| *name).collect()
}

/// JS prelude: wrap each raw host (`__name`, taking/returning JSON strings) as an ergonomic
/// `name(x)` that passes real JS values, and provide the entry invoker.
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
globalThis.__wrap = function (name) {
    return async function (x) {
        var input = x === undefined ? null : x;
        var renderer = globalThis.__stageQuestionRenderers[name];
        var hostInput = input;
        if (renderer !== undefined) {
            var question = renderer(input);
            if (typeof question !== "string") {
                throw new Error("stage '" + name + "' renderQuestion must return a string");
            }
            hostInput = {
                __ratatoskrRenderedQuestion: {
                    input: input,
                    question: question
                }
            };
        }
        var raw = await globalThis["__" + name](JSON.stringify(hostInput));
        var r = JSON.parse(raw);
        if (r && Object.prototype.hasOwnProperty.call(r, "__error")) throw new Error(r.__error);
        return r.value;
    };
};
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
    _runtime: AsyncRuntime,
    context: AsyncContext,
    /// Kept alive with the runtime whose interrupt handler reads it.
    budget: Arc<Budget>,
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
        let meta = Self::declared(&context, &budget, name, &source)
            .await?
            .ok_or_else(|| {
                ScriptError::Eval(format!("bundled workflow `{name}` has no declaration"))
            })?;
        Ok(Self {
            _runtime: runtime,
            context,
            budget,
            module_name: name.into(),
            source: source.into_boxed_str(),
            meta: Box::new(meta),
            dependencies: Box::new([]),
            bundled: true,
        })
    }

    /// Load and transpile `path` (`.ratatoskr/workflow.ts`), resolving its imports from `modules`.
    /// `Ok(None)` if the file is absent — the caller then runs the built-in Rust flow, exactly as
    /// the ruleset engine treats a missing rules dir.
    pub async fn load(path: &Path, modules: Modules<'_>) -> Result<Option<Self>, ScriptError> {
        if !path.is_file() {
            return Ok(None);
        }
        let src = std::fs::read_to_string(path)
            .map_err(|e| ScriptError::Io(path.display().to_string(), e))?;
        let loaded = transpile::transpile_workflow(path, &src, &specifiers(modules))?;

        let (runtime, context, budget) = engine(modules).await?;
        let module_name = path.display().to_string();

        // Evaluated once here to read what the script declares about itself. `run` evaluates it
        // again in the same context, which re-runs `defineWorkflow` — an idempotent assignment, and
        // the price of keeping the two paths independent.
        let declared = Self::declared(&context, &budget, &module_name, &loaded.javascript).await?;
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
            _runtime: runtime,
            context,
            budget,
            module_name: module_name.into(),
            source: loaded.javascript.into_boxed_str(),
            meta: Box::new(meta),
            dependencies: loaded.dependencies.into_boxed_slice(),
            bundled: false,
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
            budget,
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
    /// A renderer runs synchronously in JavaScript before its Rust host. It receives the original
    /// structured input and may return only a string; the host still owns the model call, output
    /// validation, checkpointing, telemetry and every workflow gate.
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

        within(
            &self.module_name,
            &self.budget,
            RUN_BUDGET,
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

                for (name, hostfn) in hosts {
                    let hf = hostfn.clone();
                    let host_budget = Arc::clone(&budget);
                    let f = Function::new(ctx.clone(), move |arg: String| {
                        let hf = hf.clone();
                        // A host call is Rust's time, not the workflow's: the clock stops for as
                        // long as one is outstanding and restarts when the last returns.
                        let host_budget = Arc::clone(&host_budget);
                        host_budget.enter_host();
                        Promised(async move {
                            let result = hf(arg).await;
                            host_budget.leave_host();
                            match result {
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

/// Wall clock a workflow's own JavaScript may run while no host call is outstanding.
///
/// Composition between stages is bookkeeping: pick a branch, reshape a value, render a question.
/// Thirty seconds of that without calling a host is a spin, not a workflow. A host call suspends
/// the clock entirely, so a stage that thinks for an hour is unaffected.
const RUN_BUDGET: Duration = Duration::from_secs(30);

/// The zero the deadline is measured from. Wall-clock instants do not fit in an atomic; elapsed
/// milliseconds since a fixed start do, which is what the interrupt handler needs to read on every
/// call without taking a lock.
static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Deadline shared by the engine's interrupt handler and the async watchdog.
///
/// The two together are what bounds a workflow: the handler stops JavaScript that never yields
/// (`while (true) {}`), the watchdog stops JavaScript that yields and never resumes (`await new
/// Promise(() => {})`). Neither alone catches both.
///
/// The clock is suspended while a host call is outstanding, because the ceiling is on *JavaScript*
/// and waiting an hour for a model is the ordinary case.
struct Budget {
    /// Milliseconds since [`START`], or `NO_DEADLINE` when nothing is being timed.
    deadline: AtomicU64,
    /// What to re-arm with when the last outstanding host call returns.
    span_ms: AtomicU64,
    outstanding: AtomicUsize,
}

const NO_DEADLINE: u64 = u64::MAX;

impl Budget {
    fn new() -> Self {
        Self {
            deadline: AtomicU64::new(NO_DEADLINE),
            span_ms: AtomicU64::new(0),
            outstanding: AtomicUsize::new(0),
        }
    }

    fn now_ms() -> u64 {
        START.elapsed().as_millis() as u64
    }

    fn arm(&self, span: Duration) {
        let span_ms = span.as_millis() as u64;
        self.span_ms.store(span_ms, Ordering::Relaxed);
        self.deadline
            .store(Self::now_ms() + span_ms, Ordering::Relaxed);
    }

    fn disarm(&self) {
        self.deadline.store(NO_DEADLINE, Ordering::Relaxed);
    }

    fn enter_host(&self) {
        self.outstanding.fetch_add(1, Ordering::Relaxed);
        self.disarm();
    }

    /// Re-arm once nothing is outstanding. With several hosts in flight the clock stays suspended
    /// until the last one returns, so concurrent stages are not held to one stage's budget.
    fn leave_host(&self) {
        if self.outstanding.fetch_sub(1, Ordering::Relaxed) == 1 {
            self.deadline.store(
                Self::now_ms() + self.span_ms.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
        }
    }

    fn expired(&self) -> bool {
        Self::now_ms() >= self.deadline.load(Ordering::Relaxed)
    }

    /// `None` while nothing is being timed.
    fn remaining(&self) -> Option<Duration> {
        match self.deadline.load(Ordering::Relaxed) {
            NO_DEADLINE => None,
            deadline => Some(Duration::from_millis(
                deadline.saturating_sub(Self::now_ms()),
            )),
        }
    }
}

/// Resolves once the budget is spent, re-reading it because a returning host call moves it.
async fn spent(budget: &Budget) {
    loop {
        match budget.remaining() {
            None => tokio::time::sleep(Duration::from_millis(100)).await,
            Some(left) if left.is_zero() => return,
            Some(left) => tokio::time::sleep(left).await,
        }
    }
}

/// Run one JavaScript operation under `span`, naming `workflow` in whichever limit it hits.
///
/// An engine limit surfaces as an opaque `interrupted` or `out of memory`; a workflow that will not
/// load must say which workflow, or the operator is left bisecting their workflows directory.
async fn within<T>(
    workflow: &str,
    budget: &Budget,
    span: Duration,
    operation: impl Future<Output = Result<T, ScriptError>>,
) -> Result<T, ScriptError> {
    let overran = || {
        ScriptError::Eval(format!(
            "workflow `{workflow}` ran more than {}s of JavaScript without progress",
            span.as_secs()
        ))
    };
    budget.arm(span);
    let outcome = tokio::select! {
        biased;
        result = operation => result,
        () = spent(budget) => Err(overran()),
    };
    let out_of_time = budget.expired();
    budget.disarm();
    outcome.map_err(|error| {
        let text = error.to_string();
        if out_of_time && text.contains("interrupted") {
            overran()
        } else if text.contains("out of memory") {
            ScriptError::Eval(format!(
                "workflow `{workflow}` exceeded its {} MiB memory limit",
                MEMORY_LIMIT / (1024 * 1024)
            ))
        } else {
            error
        }
    })
}

/// A JS runtime whose only importable modules are the ones the host supplied, plus a context on it.
///
/// Bounded before it is handed anything to run: a workflow is repository-authored code the harness
/// must evaluate before it can know whether it terminates.
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
/// The module name is the workflow's own path, so an unresolved specifier reports the file an
/// author would go and edit rather than a placeholder.
async fn evaluate<'js>(
    ctx: &Ctx<'js>,
    module_name: &str,
    source: &str,
) -> Result<Module<'js, Evaluated>, ScriptError> {
    let fail = |e: rquickjs::CaughtError| ScriptError::Eval(format!("{e}"));
    ctx.eval::<(), _>(BOOTSTRAP).catch(ctx).map_err(fail)?;
    let (module, promise) = Module::declare(ctx.clone(), module_name, source)
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
