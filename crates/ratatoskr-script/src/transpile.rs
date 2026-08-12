//! TypeScript → JavaScript via swc (type stripping): parse → optional workflow rewrites → resolver
//! → strip → codegen.

use std::io::Read as _;
use std::path::{Component, Path, PathBuf};

use swc_core::common::sync::Lrc;
use swc_core::common::{FileName, GLOBALS, Mark, SourceMap, Span, Spanned};
use swc_core::ecma::ast::{
    ArrowExpr, Callee, Class, Decl, Expr, Function, ImportDecl, Lit, ModuleDecl, ModuleItem,
    ObjectPatProp, Pat, Program, Stmt, Str, VarDecl, VarDeclKind,
};
use swc_core::ecma::codegen::text_writer::JsWriter;
use swc_core::ecma::codegen::{Config as CodegenConfig, Emitter};
use swc_core::ecma::parser::{Parser, StringInput, Syntax, TsSyntax, lexer::Lexer};
use swc_core::ecma::transforms::base::resolver;
use swc_core::ecma::transforms::typescript::strip;
use swc_core::ecma::visit::{Visit, VisitMut, VisitMutWith, VisitWith};

use crate::ScriptError;

/// Strip TypeScript types from `src`, returning JavaScript.
pub fn transpile_ts(src: &str) -> Result<String, ScriptError> {
    transpile(src, FileName::Anon, |_| Ok(()))
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
    let mut javascript = transpile(src, file_name, |parsed| {
        let mut rewriter = LoadRewriter {
            source_map: Lrc::clone(&parsed.source_map),
            display_path,
            permitted,
            load,
            error: None,
        };
        parsed.program.visit_mut_with(&mut rewriter);
        entries = published_global_names(&parsed.program);
        match rewriter.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })?;
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

fn transpile(
    src: &str,
    file_name: FileName,
    rewrite: impl FnOnce(&mut ParsedProgram) -> Result<(), ScriptError>,
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

struct LoadRewriter<'a> {
    source_map: Lrc<SourceMap>,
    display_path: &'a Path,
    /// Specifiers the host offers this workflow. Resolution is from this set alone, so anything
    /// else is a typo that can be caught here — with the import's span — instead of at evaluation,
    /// where the engine's own resolution error names no line.
    permitted: &'a [&'a str],
    load: &'a mut dyn FnMut(&str) -> Result<LoadedText, String>,
    error: Option<ScriptError>,
}

impl LoadRewriter<'_> {
    fn fail(&mut self, span: Span, target: &str, message: impl std::fmt::Display) {
        self.fail_at(span, format_args!("LOAD target `{target}`: {message}"));
    }

    fn fail_at(&mut self, span: Span, message: impl std::fmt::Display) {
        if self.error.is_some() {
            return;
        }
        let location = self.source_map.lookup_char_pos(span.lo());
        self.error = Some(ScriptError::Transpile(format!(
            "{}:{}:{}: {message}",
            self.display_path.display(),
            location.line,
            location.col_display + 1,
        )));
    }
}

impl VisitMut for LoadRewriter<'_> {
    // A type-only import is erased before any module is resolved, so it names nothing the host has
    // to offer. Every other specifier must be in the permitted set: there is no filesystem
    // resolver behind it, so an unknown one can only fail, and failing here buys the call site.
    fn visit_mut_import_decl(&mut self, import: &mut ImportDecl) {
        if import.type_only {
            return;
        }
        let requested = import.src.value.to_string_lossy();
        if self.permitted.contains(&requested.as_ref()) {
            return;
        }
        let available = match self.permitted {
            [] => "this workflow may import nothing".to_string(),
            names => format!("available: {}", names.join(", ")),
        };
        self.fail_at(
            import.span,
            format_args!("import `{requested}`: no such module; {available}"),
        );
    }

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
    fn ordinary_typescript_transpilation_does_not_expand_workflow_loads() {
        let js = transpile_ts("const prompt: string = LOAD('prompt.md');").unwrap();
        assert!(
            js.contains("LOAD(") && js.contains("prompt.md"),
            "got: {js}"
        );
    }
}
