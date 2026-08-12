//! TypeScript → JavaScript via swc (type stripping): parse → optional workflow rewrites → resolver
//! → strip → fixer → optional inspection of the emitted program → codegen.

use std::io::Read as _;
use std::path::{Component, Path, PathBuf};

use swc_core::common::sync::Lrc;
use swc_core::common::{FileName, GLOBALS, Mark, SourceMap, Span, Spanned};
use swc_core::ecma::ast::{
    ArrowExpr, CallExpr, Callee, Class, Decl, ExportAll, Expr, Function, ImportDecl, Lit,
    ModuleDecl, ModuleItem, NamedExport, ObjectPatProp, Pat, Program, Stmt, Str, VarDecl,
    VarDeclKind,
};
use swc_core::ecma::codegen::text_writer::JsWriter;
use swc_core::ecma::codegen::{Config as CodegenConfig, Emitter};
use swc_core::ecma::parser::{Parser, StringInput, Syntax, TsSyntax, lexer::Lexer};
use swc_core::ecma::transforms::base::{fixer::fixer, resolver};
use swc_core::ecma::transforms::typescript::strip;
use swc_core::ecma::visit::{Visit, VisitMut, VisitMutWith, VisitWith};

use crate::ScriptError;

/// Strip TypeScript types from `src`, returning JavaScript.
pub fn transpile_ts(src: &str) -> Result<String, ScriptError> {
    transpile(src, FileName::Anon, |_| Ok(()), |_| Ok(()))
}

const MAX_LOAD_BYTES: usize = 16 * 1024;

pub(crate) struct WorkflowSource {
    pub javascript: String,
    pub dependencies: Vec<PathBuf>,
}

struct LoadedText {
    content: String,
}

/// Transpile a repository workflow, replacing literal `LOAD` calls before JavaScript exists and
/// refusing any `import` specifier outside `permitted`.
pub(crate) fn transpile_workflow(
    path: &Path,
    src: &str,
    permitted: &[&str],
) -> Result<WorkflowSource, ScriptError> {
    let workflow_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let canonical_dir = workflow_dir
        .canonicalize()
        .map_err(|error| ScriptError::Io(workflow_dir.display().to_string(), error))?;
    let mut dependencies = Vec::new();
    let mut load = |requested: &str| {
        let requested_path = Path::new(requested);
        if requested_path.is_absolute() {
            return Err(format!("target `{requested}` must be relative"));
        }
        let joined = canonical_dir.join(requested_path);
        let canonical_target = joined
            .canonicalize()
            .map_err(|error| format!("target `{}`: {error}", joined.display()))?;
        if !canonical_target.starts_with(&canonical_dir) || canonical_target == canonical_dir {
            return Err(format!(
                "target `{}` resolves outside workflow directory `{}`",
                canonical_target.display(),
                canonical_dir.display()
            ));
        }
        let content = read_load_text(&canonical_target)?;
        dependencies.push(canonical_target.clone());
        Ok(LoadedText { content })
    };
    let javascript = transpile_with_loads(
        src,
        FileName::Real(path.to_path_buf()),
        path,
        permitted,
        &mut load,
    )?;
    dependencies.sort();
    dependencies.dedup();
    Ok(WorkflowSource {
        javascript,
        dependencies,
    })
}

/// Transpile a named source embedded in the binary by its owning crate, resolving its `LOAD` calls
/// against `includes`.
///
/// Public because a definitions module a workflow imports carries `LOAD` calls of its own, and only
/// this path resolves them: `transpile_ts` leaves `LOAD(..)` in the emitted JavaScript, where no
/// such function exists.
pub fn transpile_with_includes(
    name: &str,
    src: &str,
    includes: &[(&str, &str)],
    permitted: &[&str],
) -> Result<String, ScriptError> {
    let mut load = |requested: &str| {
        let normalized = normalize_bundled_target(requested)?;
        let (_, content) = includes
            .iter()
            .find(|(target, _)| *target == normalized)
            .ok_or_else(|| format!("bundled target `{normalized}` is not embedded"))?;
        let content = checked_load_text(normalized.as_str(), content.as_bytes().to_vec())?;
        Ok(LoadedText { content })
    };
    transpile_with_loads(
        src,
        FileName::Custom(name.to_string()),
        Path::new(name),
        permitted,
        &mut load,
    )
}

fn normalize_bundled_target(requested: &str) -> Result<String, String> {
    let path = Path::new(requested);
    if path.is_absolute() {
        return Err(format!("target `{requested}` must be relative"));
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "target `{requested}` escapes the bundled workflow source root"
                ));
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err("LOAD target is empty".to_string());
    }
    Ok(normalized.to_string_lossy().into_owned())
}

fn checked_load_text(target: &str, bytes: Vec<u8>) -> Result<String, String> {
    if bytes.len() > MAX_LOAD_BYTES {
        return Err(format!(
            "target `{target}` is {} bytes; LOAD limit is {MAX_LOAD_BYTES} bytes",
            bytes.len()
        ));
    }
    String::from_utf8(bytes).map_err(|_| format!("target `{target}` is not valid UTF-8"))
}

fn read_load_text(target: &Path) -> Result<String, String> {
    let display = target.display().to_string();
    let metadata = target
        .metadata()
        .map_err(|error| format!("target `{display}`: {error}"))?;
    if metadata.len() > MAX_LOAD_BYTES as u64 {
        return Err(format!(
            "target `{display}` is {} bytes; LOAD limit is {MAX_LOAD_BYTES} bytes",
            metadata.len()
        ));
    }
    let file =
        std::fs::File::open(target).map_err(|error| format!("target `{display}`: {error}"))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take((MAX_LOAD_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("target `{display}`: {error}"))?;
    checked_load_text(&display, bytes)
}

fn transpile_with_loads(
    src: &str,
    file_name: FileName,
    display_path: &Path,
    permitted: &[&str],
    load: &mut dyn FnMut(&str) -> Result<LoadedText, String>,
) -> Result<String, ScriptError> {
    let mut entries = Vec::new();
    let mut javascript = transpile(
        src,
        file_name,
        |parsed| {
            let mut rewriter = LoadRewriter {
                source_map: Lrc::clone(&parsed.source_map),
                display_path,
                load,
                error: None,
            };
            parsed.program.visit_mut_with(&mut rewriter);
            match rewriter.error {
                Some(error) => Err(error),
                None => Ok(()),
            }
        },
        |parsed| {
            entries = published_global_names(&parsed.program);
            let mut refs = ModuleRefs {
                source_map: &parsed.source_map,
                display_path,
                permitted,
                error: None,
            };
            parsed.program.visit_with(&mut refs);
            match refs.error {
                Some(error) => Err(error),
                None => Ok(()),
            }
        },
    )?;
    // A workflow is evaluated as an ES module, so `async function plan(..)` is module-scoped and
    // invisible to the entry invoker, which looks entries up on `globalThis`. Publishing them keeps
    // a workflow an ordinary file of functions instead of demanding `export` on every entry.
    for entry in entries {
        javascript.push_str(&format!("\nglobalThis.{entry} = {entry};"));
    }
    Ok(javascript)
}

/// The names a source would have put on `globalThis` when a workflow was evaluated as a script.
///
/// Runs on the emitted program, after type stripping, so it reads the declarations that actually
/// exist at runtime. A TypeScript construct that lowers to something else is therefore seen as what
/// it lowers to: `namespace N { export var helper = 1 }` publishes the `var N` the lowering creates
/// and not `helper`, which the lowering turned into a property of `N` with no binding of its own.
///
/// Two kinds reached the global object and must keep reaching it, or module scoping silently narrows
/// what a workflow may be written as — and an entry the runtime cannot find is a confusing way to
/// learn that:
///
/// * a top-level function declaration (ambient `declare function` emits nothing to assign, so it is
///   skipped);
/// * every binding a `var` introduces, wherever it sits. `var` hoists out of blocks, branches and
///   loops to its enclosing function, which at a script's top level is the global object, and it
///   binds through destructuring — `var { plan } = ..` published `plan`.
///
/// Not `let` or `const`: those are lexical even at a script's top level and never became properties
/// of the global object, so publishing them would invent reach a workflow never had.
fn published_global_names(program: &Program) -> Vec<String> {
    let mut names = Vec::new();
    for declaration in top_level_declarations(program) {
        if let Decl::Fn(function) = declaration
            && !function.declare
        {
            names.push(function.ident.sym.to_string());
        }
    }
    let mut hoisted = HoistedVars { names: Vec::new() };
    program.visit_with(&mut hoisted);
    names.extend(hoisted.names);
    names.dedup();
    names
}

fn top_level_declarations(program: &Program) -> Vec<&Decl> {
    match program {
        Program::Module(module) => module
            .body
            .iter()
            .filter_map(|item| match item {
                ModuleItem::Stmt(Stmt::Decl(declaration)) => Some(declaration),
                ModuleItem::ModuleDecl(ModuleDecl::ExportDecl(export)) => Some(&export.decl),
                _ => None,
            })
            .collect(),
        Program::Script(script) => script
            .body
            .iter()
            .filter_map(|statement| match statement {
                Stmt::Decl(declaration) => Some(declaration),
                _ => None,
            })
            .collect(),
    }
}

/// Collects the bindings of every `var` in the global statement scope.
struct HoistedVars {
    names: Vec<String>,
}

impl Visit for HoistedVars {
    fn visit_var_decl(&mut self, declaration: &VarDecl) {
        if declaration.kind == VarDeclKind::Var && !declaration.declare {
            for declarator in &declaration.decls {
                binding_names(&declarator.name, &mut self.names);
            }
        }
        declaration.visit_children_with(self);
    }

    // A `var` inside a function or a class body belongs to that scope, not to the global object, so
    // the walk stops at those boundaries rather than hoisting their locals into the global set.
    fn visit_function(&mut self, _: &Function) {}
    fn visit_arrow_expr(&mut self, _: &ArrowExpr) {}
    fn visit_class(&mut self, _: &Class) {}
}

/// Every identifier a binding pattern introduces, including through destructuring.
fn binding_names(pattern: &Pat, out: &mut Vec<String>) {
    match pattern {
        Pat::Ident(identifier) => out.push(identifier.id.sym.to_string()),
        Pat::Array(array) => array
            .elems
            .iter()
            .flatten()
            .for_each(|element| binding_names(element, out)),
        Pat::Object(object) => {
            for property in &object.props {
                match property {
                    ObjectPatProp::KeyValue(entry) => binding_names(&entry.value, out),
                    ObjectPatProp::Assign(entry) => out.push(entry.key.id.sym.to_string()),
                    ObjectPatProp::Rest(rest) => binding_names(&rest.arg, out),
                }
            }
        }
        Pat::Assign(assign) => binding_names(&assign.left, out),
        Pat::Rest(rest) => binding_names(&rest.arg, out),
        Pat::Expr(_) | Pat::Invalid(_) => {}
    }
}

struct ParsedProgram {
    source_map: Lrc<SourceMap>,
    program: Program,
}

/// Parse `src`, let `rewrite` edit the TypeScript AST, strip types, let `emitted` inspect the
/// result, then generate JavaScript.
///
/// The two hooks exist because they need different programs. `rewrite` runs on the source as
/// written, which is where `LOAD(..)` still is. `emitted` runs on what codegen is about to print:
/// namespaces and enums lowered, type-only imports gone — the only program whose declarations and
/// module references match what the engine will see.
fn transpile(
    src: &str,
    file_name: FileName,
    rewrite: impl FnOnce(&mut ParsedProgram) -> Result<(), ScriptError>,
    emitted: impl FnOnce(&ParsedProgram) -> Result<(), ScriptError>,
) -> Result<String, ScriptError> {
    GLOBALS.set(&Default::default(), || {
        let cm: Lrc<SourceMap> = Default::default();
        let fm = cm.new_source_file(Lrc::new(file_name), src.to_string());

        let lexer = Lexer::new(
            Syntax::Typescript(TsSyntax::default()),
            Default::default(),
            StringInput::from(&*fm),
            None,
        );
        let mut parser = Parser::new_from(lexer);
        let program: Program = parser
            .parse_program()
            .map_err(|e| ScriptError::Transpile(format!("{e:?}")))?;

        // swc recovers from many syntax errors instead of returning `Err`, accumulating them here.
        // Treat any as a hard load failure — a malformed ruleset must not silently half-parse.
        let recovered = parser.take_errors();
        if !recovered.is_empty() {
            let msgs: Vec<String> = recovered
                .iter()
                .map(|e| format!("{:?}", e.kind()))
                .collect();
            return Err(ScriptError::Transpile(format!(
                "{} syntax error(s): {}",
                msgs.len(),
                msgs.join("; ")
            )));
        }

        let mut parsed = ParsedProgram {
            source_map: Lrc::clone(&cm),
            program,
        };
        rewrite(&mut parsed)?;

        let unresolved = Mark::new();
        let top_level = Mark::new();
        parsed.program.mutate(resolver(unresolved, top_level, true));
        parsed.program.mutate(strip(unresolved, top_level));
        // Stripping rewrites a namespace or an enum into an IIFE over `(N || (N = {}))`, an AST that
        // only prints as valid JavaScript once the parentheses its precedence needs are inserted.
        // Codegen does not add them; this is the pass that does, and swc's own pipeline ends the
        // same way.
        parsed.program.mutate(fixer(None));
        emitted(&parsed)?;

        let mut buf = Vec::new();
        {
            let writer = JsWriter::new(cm.clone(), "\n", &mut buf, None);
            let mut emitter = Emitter {
                cfg: CodegenConfig::default(),
                cm: cm.clone(),
                comments: None,
                wr: writer,
            };
            emitter
                .emit_program(&parsed.program)
                .map_err(|e| ScriptError::Transpile(format!("codegen: {e}")))?;
        }
        String::from_utf8(buf).map_err(|e| ScriptError::Transpile(e.to_string()))
    })
}

/// `{file}:{line}:{col}: {message}` — a transpile failure at the construct that caused it, so the
/// fix is a jump rather than a search.
fn error_at(
    source_map: &SourceMap,
    display_path: &Path,
    span: Span,
    message: impl std::fmt::Display,
) -> ScriptError {
    let location = source_map.lookup_char_pos(span.lo());
    ScriptError::Transpile(format!(
        "{}:{}:{}: {message}",
        display_path.display(),
        location.line,
        location.col_display + 1,
    ))
}

/// Refuses every module reference the host does not offer.
///
/// Runs on the emitted program rather than the source AST, so what it judges is exactly the module
/// graph the engine will resolve. Every flavour of erased type-only import — `import type`,
/// `import { type Foo }`, and a plain import the source only ever used in type position — is already
/// gone by then, so none of them needs a specifier flag second-guessed on its behalf, and nothing
/// that survives can be exempted.
///
/// Resolution is from `permitted` alone, with no filesystem behind it, so an unknown specifier can
/// only fail. Failing here buys the reference's own line and column instead of the engine's
/// locationless resolver error — which stays as the backstop this check aims to make unreachable.
struct ModuleRefs<'a> {
    source_map: &'a SourceMap,
    display_path: &'a Path,
    permitted: &'a [&'a str],
    error: Option<ScriptError>,
}

impl ModuleRefs<'_> {
    fn check(&mut self, span: Span, requested: &str) {
        if self.error.is_some() || self.permitted.contains(&requested) {
            return;
        }
        let available = match self.permitted {
            [] => "this workflow may import nothing".to_string(),
            names => format!("available: {}", names.join(", ")),
        };
        self.error = Some(error_at(
            self.source_map,
            self.display_path,
            span,
            format_args!("import `{requested}`: no such module; {available}"),
        ));
    }
}

impl Visit for ModuleRefs<'_> {
    fn visit_import_decl(&mut self, import: &ImportDecl) {
        self.check(import.span, &import.src.value.to_string_lossy());
    }

    // `export { x } from ".."` and `export * from ".."` resolve the same map an `import` does.
    fn visit_named_export(&mut self, export: &NamedExport) {
        if let Some(source) = &export.src {
            self.check(export.span, &source.value.to_string_lossy());
        }
    }

    fn visit_export_all(&mut self, export: &ExportAll) {
        self.check(export.span, &export.src.value.to_string_lossy());
    }

    // A dynamic `import(..)` goes through the same resolver. Only a literal specifier can be judged
    // before evaluation; a computed one is left to the engine.
    fn visit_call_expr(&mut self, call: &CallExpr) {
        call.visit_children_with(self);
        if matches!(call.callee, Callee::Import(_))
            && let Some(argument) = call.args.first()
            && argument.spread.is_none()
            && let Expr::Lit(Lit::Str(requested)) = argument.expr.as_ref()
        {
            self.check(call.span, &requested.value.to_string_lossy());
        }
    }
}

struct LoadRewriter<'a> {
    source_map: Lrc<SourceMap>,
    display_path: &'a Path,
    load: &'a mut dyn FnMut(&str) -> Result<LoadedText, String>,
    error: Option<ScriptError>,
}

impl LoadRewriter<'_> {
    fn fail(&mut self, span: Span, target: &str, message: impl std::fmt::Display) {
        if self.error.is_some() {
            return;
        }
        self.error = Some(error_at(
            &self.source_map,
            self.display_path,
            span,
            format_args!("LOAD target `{target}`: {message}"),
        ));
    }
}

impl VisitMut for LoadRewriter<'_> {
    fn visit_mut_expr(&mut self, expression: &mut Expr) {
        expression.visit_mut_children_with(self);
        if self.error.is_some() {
            return;
        }
        let Expr::Call(call) = expression else {
            return;
        };
        let Callee::Expr(callee) = &call.callee else {
            return;
        };
        let Expr::Ident(identifier) = callee.as_ref() else {
            return;
        };
        if identifier.sym != *"LOAD" {
            return;
        }
        let span = call.span();
        let [argument] = call.args.as_slice() else {
            self.fail(span, "<non-literal>", "expected exactly one string literal");
            return;
        };
        if argument.spread.is_some() {
            self.fail(
                span,
                "<spread>",
                "expected one string literal, not a spread",
            );
            return;
        }
        let Expr::Lit(Lit::Str(requested)) = argument.expr.as_ref() else {
            self.fail(span, "<non-literal>", "expected exactly one string literal");
            return;
        };
        let requested = requested.value.to_string_lossy().into_owned();
        match (self.load)(&requested) {
            Ok(loaded) => {
                *expression = Expr::Lit(Lit::Str(Str {
                    span,
                    value: loaded.content.into(),
                    raw: None,
                }));
            }
            Err(error) => self.fail(span, &requested, error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_types_to_js() {
        let js = transpile_ts("const x: number = 1; function f(a: string): string { return a; }")
            .unwrap();
        assert!(js.contains("const x = 1"), "got: {js}");
        assert!(!js.contains(": number"), "types not stripped: {js}");
    }

    #[test]
    fn rejects_malformed_source() {
        // Whether swc recovers or hard-fails, malformed input must surface as a Transpile error,
        // not a silently half-parsed program.
        let err = transpile_ts("const = ;").unwrap_err();
        assert!(matches!(err, ScriptError::Transpile(_)), "got: {err:?}");
    }

    #[test]
    fn a_workflow_publishes_its_top_level_functions_as_globals() {
        // The entry invoker looks entries up on `globalThis`, and a workflow is evaluated as an ES
        // module, where a top-level `function` is module-scoped.
        let js = transpile_with_includes(
            "w",
            "async function plan(i: string) { return i; }\nexport function helper() {}",
            &[],
            &[],
        )
        .unwrap();
        assert!(js.contains("globalThis.plan = plan;"), "got: {js}");
        assert!(js.contains("globalThis.helper = helper;"), "got: {js}");
        // Rulesets are plain scripts and get no such rewriting.
        assert!(
            !transpile_ts("async function plan(i) { return i; }")
                .unwrap()
                .contains("globalThis"),
        );
    }

    #[test]
    fn the_published_set_is_the_one_script_evaluation_gave() {
        // `var` hoisted onto the global object when a workflow was a script, so an entry written
        // that way has to keep resolving; `let` and `const` are lexical and never did, so publishing
        // them now would invent reach a workflow never had.
        let js = transpile_with_includes(
            "w",
            "var plan = async function (i: string) { return i; };\n\
             let helper = () => {};\n\
             const other = 1;",
            &[],
            &[],
        )
        .unwrap();
        assert!(js.contains("globalThis.plan = plan;"), "got: {js}");
        assert!(!js.contains("globalThis.helper"), "got: {js}");
        assert!(!js.contains("globalThis.other"), "got: {js}");
    }

    #[test]
    fn a_var_publishes_wherever_it_sits_and_however_it_binds() {
        // `var` hoists out of blocks to the enclosing function — at a script's top level, the global
        // object — and binds through destructuring. Both reached `globalThis` before this engine
        // evaluated modules, so an entry written either way has to keep resolving.
        let js = transpile_with_includes(
            "w",
            "if (ready) { var branched = async (i) => i; }\n\
             for (var counted = 0; counted < 1; counted++) {}\n\
             var { plan, missing = 1 } = parts;\n\
             var [first, ...rest] = list;\n\
             function outer() { var inside = 1; return inside; }\n\
             const shape = { method() { var buried = 2; return buried; } };",
            &[],
            &[],
        )
        .unwrap();
        for published in ["branched", "counted", "plan", "missing", "first", "rest"] {
            assert!(
                js.contains(&format!("globalThis.{published} = {published};")),
                "`{published}` should be published, got: {js}"
            );
        }
        // A `var` inside a function or a method belongs to that scope, not the global object.
        assert!(!js.contains("globalThis.inside"), "got: {js}");
        assert!(!js.contains("globalThis.buried"), "got: {js}");
    }

    #[test]
    fn a_namespace_publishes_the_binding_its_lowering_leaves_behind() {
        // A namespace member is not a binding of its own: swc lowers `helper` to a property of `N`.
        // Publishing the name the source wrote would emit `globalThis.helper = helper;` against
        // nothing, and the workflow would throw at load. Collecting after the lowering sees only
        // `var N`, which is a real global and the one an entry could be reached through.
        let js = transpile_with_includes("w", "namespace N { export var helper = 1; }", &[], &[])
            .unwrap();
        assert!(!js.contains("globalThis.helper"), "got: {js}");
        assert!(js.contains("globalThis.N = N;"), "got: {js}");
    }

    #[test]
    fn an_import_erased_before_codegen_is_not_held_to_the_permitted_set() {
        // Each of these leaves no module reference in the emitted JavaScript, so there is nothing
        // for the engine to resolve and nothing for the permitted set to judge — including the ones
        // carrying no `type_only` flag on the declaration or on any specifier, which no inspection
        // of the source AST could have told apart from a value import.
        for source in [
            "import type { Foo } from \"unoffered\";\nexport const x: Foo = 1 as never;",
            "import { type Foo } from \"unoffered\";\nexport const x: Foo = 1 as never;",
            "import { Foo } from \"unoffered\";\ntype Alias = Foo;\nexport const x: Alias = 1 as never;",
            // Stripping also drops a specifier the source never used at all.
            "import { Foo } from \"unoffered\";\nexport const x = 1;",
        ] {
            let js = transpile_with_includes("w", source, &[], &[]).unwrap();
            assert!(!js.contains("unoffered"), "got: {js}");
        }
    }

    #[test]
    fn an_import_that_survives_stripping_is_still_refused() {
        // The exemption above is "erased", not "type-shaped": the moment a specifier is used as a
        // value the import is in the emitted module graph, and the permitted set applies.
        let error = transpile_with_includes(
            "w",
            "import { Foo } from \"unoffered\";\nconst x: Foo = Foo.make();",
            &[],
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("w:1:1:"), "{error}");
        assert!(
            error.contains("import `unoffered`: no such module; this workflow may import nothing"),
            "{error}"
        );
    }

    #[test]
    fn every_module_reference_is_checked_at_its_own_position() {
        // A re-export and a dynamic import reach the same resolver an `import` does, so leaving
        // them out would let them fall through to the engine's locationless failure.
        for (source, position) in [
            ("const ok = 1;\nexport { ok } from \"nope\";", "w:2:1:"),
            ("const ok = 1;\n\nexport * from \"nope\";", "w:3:1:"),
            (
                "export async function plan() {\n  const m = await import(\"nope\");\n  return m;\n}",
                "w:2:19:",
            ),
        ] {
            let error = transpile_with_includes("w", source, &[], &["ratatoskr/nodes"])
                .unwrap_err()
                .to_string();
            assert!(error.contains(position), "{error}");
            assert!(
                error.contains("import `nope`: no such module; available: ratatoskr/nodes"),
                "{error}"
            );
        }
    }

    #[test]
    fn ordinary_typescript_transpilation_does_not_expand_workflow_loads() {
        let js = transpile_ts("const prompt: string = LOAD('prompt.md');").unwrap();
        assert!(
            js.contains("LOAD(") && js.contains("prompt.md"),
            "got: {js}"
        );
    }
}
