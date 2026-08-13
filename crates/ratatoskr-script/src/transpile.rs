//! TypeScript → JavaScript via swc (type stripping): parse → optional workflow rewrites → resolver
//! → strip → fixer → optional inspection of the emitted program → codegen.
//!
//! A ruleset is a script and a workflow is an ES module, but only their surrounding checks differ:
//! this pass emits the same JavaScript for both and never rewrites a workflow's shape. A workflow's
//! entries are the functions it exports, which the runtime reads off the evaluated module, so
//! nothing here has to make module-scoped declarations reachable from somewhere else.

use std::io::Read as _;
use std::path::{Component, Path, PathBuf};

use swc_core::common::sync::Lrc;
use swc_core::common::{FileName, GLOBALS, Mark, SourceMap, Span, Spanned};
use swc_core::ecma::ast::{
    CallExpr, Callee, ExportAll, Expr, Ident, ImportDecl, Lit, NamedExport, Program, Stmt, Str,
};
use swc_core::ecma::codegen::text_writer::JsWriter;
use swc_core::ecma::codegen::{Config as CodegenConfig, Emitter};
use swc_core::ecma::parser::{EsSyntax, Parser, StringInput, Syntax, TsSyntax, lexer::Lexer};
use swc_core::ecma::transforms::base::{fixer::fixer, resolver};
use swc_core::ecma::transforms::typescript::strip;
use swc_core::ecma::visit::{Visit, VisitMut, VisitMutWith, VisitWith};

use crate::ScriptError;

/// Strip TypeScript types from `src`, returning JavaScript.
pub fn transpile_ts(src: &str) -> Result<String, ScriptError> {
    transpile(src, FileName::Anon, |_| Ok(()), |_| Ok(()))
}

const MAX_LOAD_BYTES: usize = 16 * 1024;

/// Bytes of TypeScript this module will read from the repository.
///
/// A workflow or a ruleset is composition — declarations and glue, with prompt text arriving
/// through `LOAD` at 16 KiB a target — so 256 KiB is thousands of lines past anything honest. The
/// ceiling exists because parsing is where the cost is: swc builds an AST several times the
/// source's size, and 100 MB of small statements takes ~3 GB of resident memory and aborts the
/// process before any engine limit has been installed. Checked against the file's metadata, so an
/// oversize source is refused without being read.
pub(crate) const MAX_SCRIPT_BYTES: u64 = 256 * 1024;

/// Bracket nesting a source may reach, counted on the raw text before it is parsed.
///
/// A byte ceiling does not cover this: nesting is a *stack* cost, not a heap one, and 150 levels of
/// parentheses is a 300-byte file. `swc_ecma_parser` grows the stack for its own recursion, but the
/// passes that run after it here — the `LOAD` rewrite, `resolver`, `strip`, `fixer`, codegen — are
/// plain recursive walks with no such guard, so past a depth this module cannot ask the parser to
/// bound, they overflow the thread's stack and abort the process. Measured: depth 150 aborts a
/// debug build on a 2 MiB thread stack; depth 100 does not.
///
/// 64 sits under that and far above real code — the deepest file in this repository, prose
/// included, nests 19.
const MAX_NESTING_DEPTH: usize = 64;

/// Read repository TypeScript, refusing a file past [`MAX_SCRIPT_BYTES`] before reading it.
pub(crate) fn read_script_source(path: &Path) -> Result<String, ScriptError> {
    let display = path.display().to_string();
    let metadata = path
        .metadata()
        .map_err(|error| ScriptError::Io(display.clone(), error))?;
    if metadata.len() > MAX_SCRIPT_BYTES {
        return Err(ScriptError::Transpile(format!(
            "{display} is {} bytes; source limit is {MAX_SCRIPT_BYTES} bytes",
            metadata.len()
        )));
    }
    std::fs::read_to_string(path).map_err(|error| ScriptError::Io(display, error))
}

/// Whether the scan is reading code or the text of a template literal.
#[derive(Clone, Copy)]
enum Lexical {
    Code,
    TemplateText,
}

/// Words after which a `/` opens a regular expression rather than dividing.
const REGEX_MAY_FOLLOW: &[&str] = &[
    "return",
    "typeof",
    "instanceof",
    "in",
    "of",
    "new",
    "delete",
    "void",
    "throw",
    "case",
    "do",
    "else",
    "yield",
    "await",
];

/// Refuse a source whose brackets nest past [`MAX_NESTING_DEPTH`], naming where it happened.
///
/// Counts the brackets that are *code*, skipping strings, template literal text, comments and
/// regular expressions — prose is where brackets go unbalanced, and a comment weighing sixty-five
/// options in parentheses is a file this must load, not one it must refuse. A template's `${…}` is
/// code again and counts, because the parser recurses into it like any other brace.
///
/// Hand-rolled rather than driven by swc's `Lexer`, which cannot resolve `/` on its own:
/// `Tokens::set_expr_allowed` and `set_next_regexp` are how the *parser* tells that lexer whether a
/// regex or a division follows, and there is no parser here — parsing is what this runs before, and
/// what it exists to keep off a source too deep to survive it. The rule used instead is the
/// previous significant token, and where it is unsure it reads `/` as division: counting a regex's
/// brackets costs nothing against a limit this far above honest code, while skipping a division
/// would run the scan to the next `/` in the file and swallow whatever nests after it.
fn check_nesting(file_name: &FileName, src: &str) -> Result<(), ScriptError> {
    let bytes = src.as_bytes();
    let (mut depth, mut line, mut line_start) = (0usize, 1usize, 0usize);
    // What to return to when the construct being read ends: a template's text when an
    // interpolation closes, code when a template literal does.
    let mut enclosing: Vec<Lexical> = Vec::new();
    // The bracket depth each open `${` reached, so its `}` is told from a block's.
    let mut interpolations: Vec<usize> = Vec::new();
    let mut reading = Lexical::Code;
    // Whether a `/` here would divide. False at the start of a source, where a regex may open.
    let mut divides = false;
    let mut at = 0usize;

    while at < bytes.len() {
        let from = at;
        let byte = bytes[at];
        match reading {
            Lexical::TemplateText => match byte {
                b'\\' => at += 2,
                b'`' => {
                    reading = enclosing.pop().unwrap_or(Lexical::Code);
                    divides = true;
                    at += 1;
                }
                b'$' if bytes.get(at + 1) == Some(&b'{') => {
                    depth += 1;
                    too_deep(file_name, depth, line, at - line_start + 1)?;
                    interpolations.push(depth);
                    enclosing.push(Lexical::TemplateText);
                    reading = Lexical::Code;
                    divides = false;
                    at += 2;
                }
                _ => at += 1,
            },
            Lexical::Code => match byte {
                b'/' if bytes.get(at + 1) == Some(&b'/') => at = to_line_end(bytes, at + 2),
                b'/' if bytes.get(at + 1) == Some(&b'*') => at = past_block_comment(bytes, at + 2),
                b'/' if !divides => {
                    at = past_regex(bytes, at + 1);
                    divides = true;
                }
                b'\'' | b'"' => {
                    at = past_string(bytes, at + 1, byte);
                    divides = true;
                }
                b'`' => {
                    enclosing.push(Lexical::Code);
                    reading = Lexical::TemplateText;
                    at += 1;
                }
                b'(' | b'[' | b'{' => {
                    depth += 1;
                    too_deep(file_name, depth, line, at - line_start + 1)?;
                    divides = false;
                    at += 1;
                }
                b')' | b']' => {
                    depth = depth.saturating_sub(1);
                    divides = true;
                    at += 1;
                }
                b'}' => {
                    if interpolations.last() == Some(&depth) {
                        interpolations.pop();
                        reading = enclosing.pop().unwrap_or(Lexical::Code);
                    }
                    depth = depth.saturating_sub(1);
                    divides = true;
                    at += 1;
                }
                // `a++ / b` divides; every other operator leaves a regex able to open.
                b'+' | b'-' if bytes.get(at + 1) == Some(&byte) => {
                    divides = true;
                    at += 2;
                }
                _ if is_word_byte(byte) => {
                    let end = past_word(bytes, at);
                    divides = !REGEX_MAY_FOLLOW.contains(&&src[at..end]);
                    at = end;
                }
                _ if byte.is_ascii_whitespace() => at += 1,
                _ => {
                    divides = false;
                    at += 1;
                }
            },
        }
        at = at.min(bytes.len());
        for (offset, byte) in bytes[from..at].iter().enumerate() {
            if *byte == b'\n' {
                line += 1;
                line_start = from + offset + 1;
            }
        }
    }
    Ok(())
}

fn too_deep(
    file_name: &FileName,
    depth: usize,
    line: usize,
    column: usize,
) -> Result<(), ScriptError> {
    if depth <= MAX_NESTING_DEPTH {
        return Ok(());
    }
    Err(ScriptError::Transpile(format!(
        "{file_name}:{line}:{column}: nesting limit is {MAX_NESTING_DEPTH} brackets, and parsing \
         deeper than that overflows the stack"
    )))
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'$' || byte >= 0x80
}

fn past_word(bytes: &[u8], from: usize) -> usize {
    let mut at = from;
    while at < bytes.len() && is_word_byte(bytes[at]) {
        at += 1;
    }
    at
}

fn to_line_end(bytes: &[u8], from: usize) -> usize {
    let mut at = from;
    while at < bytes.len() && bytes[at] != b'\n' {
        at += 1;
    }
    at
}

fn past_block_comment(bytes: &[u8], from: usize) -> usize {
    let mut at = from;
    while at + 1 < bytes.len() {
        if bytes[at] == b'*' && bytes[at + 1] == b'/' {
            return at + 2;
        }
        at += 1;
    }
    bytes.len()
}

/// Past the closing quote, or back to `from` if the literal does not close on its line — an
/// unterminated quote is a source that will not parse, and reading the rest of the file as its
/// contents would hide whatever nests there.
fn past_string(bytes: &[u8], from: usize, quote: u8) -> usize {
    let mut at = from;
    while at < bytes.len() {
        match bytes[at] {
            b'\\' => at += 2,
            b'\n' => return from,
            byte if byte == quote => return at + 1,
            _ => at += 1,
        }
    }
    from
}

/// Past the closing `/`, or back to `from` if it does not close on its line. `[…]` is a character
/// class, where a `/` is literal.
fn past_regex(bytes: &[u8], from: usize) -> usize {
    let mut at = from;
    let mut in_class = false;
    while at < bytes.len() {
        match bytes[at] {
            b'\\' => at += 2,
            b'\n' => return from,
            b'[' => {
                in_class = true;
                at += 1;
            }
            b']' => {
                in_class = false;
                at += 1;
            }
            b'/' if !in_class => return at + 1,
            _ => at += 1,
        }
    }
    from
}

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
    transpile(
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
    )
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
    // Before the parser, because the AST this builds is what cannot survive deep nesting.
    check_nesting(&file_name, src)?;
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

/// Whether `source` is exactly one function expression, and therefore usable as a stage's question
/// renderer.
///
/// The workflow runtime installs a renderer by evaluating `"(" + source + ")"`. A source that
/// closes that parenthesis and opens another runs statements of its own, in the runtime where the
/// privileged hosts live — so the shape of the string is the whole gate, and it is checked here,
/// in Rust, before any engine sees it. The two candidates are the two spellings the runtime tries:
/// object-method shorthand only becomes an expression once it is given the `function` prefix.
/// What a source is, judged from the program the parser builds rather than from its characters.
pub(crate) struct Shape {
    /// It has ES module syntax. `parse_program` returns a module exactly when `import`/`export` is
    /// present, which is the same question asked of the same grammar the engine will use.
    pub is_module: bool,
    /// It names `defineWorkflow` as code — not in a comment, a string, or its own prose.
    pub names_define_workflow: bool,
}

/// Parse `src` far enough to say what shape it is.
///
/// `None` when it does not parse or nests too deep to parse safely: the caller is diagnosing what a
/// file is, and a source that will not load at all is better reported by the transpile behind this,
/// which says what is wrong with it rather than what it might have been.
pub(crate) fn shape(src: &str) -> Option<Shape> {
    check_nesting(&FileName::Anon, src).ok()?;
    let cm: Lrc<SourceMap> = Default::default();
    let fm = cm.new_source_file(Lrc::new(FileName::Anon), src.to_string());
    let mut parser = Parser::new_from(Lexer::new(
        Syntax::Typescript(TsSyntax::default()),
        Default::default(),
        StringInput::from(&*fm),
        None,
    ));
    let program = parser.parse_program().ok()?;
    if !parser.take_errors().is_empty() {
        return None;
    }
    let mut named = NamesDefineWorkflow(false);
    program.visit_with(&mut named);
    Some(Shape {
        is_module: matches!(program, Program::Module(_)),
        names_define_workflow: named.0,
    })
}

struct NamesDefineWorkflow(bool);

impl Visit for NamesDefineWorkflow {
    fn visit_ident(&mut self, ident: &Ident) {
        self.0 |= &*ident.sym == "defineWorkflow";
    }
}

pub(crate) fn is_function_expression(source: &str) -> bool {
    lone_function_expression(&format!("({source})"))
        || lone_function_expression(&format!("(function {source})"))
}

/// One expression statement whose expression, past any parentheses, is a function or an arrow.
fn lone_function_expression(candidate: &str) -> bool {
    let cm: Lrc<SourceMap> = Default::default();
    let fm = cm.new_source_file(Lrc::new(FileName::Anon), candidate.to_string());
    let lexer = Lexer::new(
        // The engine parses JavaScript, so this must too: a TypeScript parse would accept an
        // annotation that the runtime then chokes on, and the point is to judge what runs.
        Syntax::Es(EsSyntax::default()),
        Default::default(),
        StringInput::from(&*fm),
        None,
    );
    let mut parser = Parser::new_from(lexer);
    let Ok(Program::Script(script)) = parser.parse_program() else {
        return false;
    };
    if !parser.take_errors().is_empty() {
        return false;
    }
    let [Stmt::Expr(statement)] = script.body.as_slice() else {
        return false;
    };
    let mut expr = statement.expr.as_ref();
    while let Expr::Paren(paren) = expr {
        expr = paren.expr.as_ref();
    }
    matches!(expr, Expr::Fn(_) | Expr::Arrow(_))
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

    fn reject_computed(&mut self, span: Span) {
        if self.error.is_some() {
            return;
        }
        self.error = Some(error_at(
            self.source_map,
            self.display_path,
            span,
            "import(..) with a computed specifier: a workflow's imports must be statically known",
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

    // A dynamic `import(..)` goes through the same resolver, so it is held to the same permitted
    // set. A computed specifier cannot be judged before evaluation, and letting it through would
    // leave one module reference reported by the engine with no line or column — so it is refused
    // here instead, at its own position.
    fn visit_call_expr(&mut self, call: &CallExpr) {
        call.visit_children_with(self);
        if !matches!(call.callee, Callee::Import(_)) {
            return;
        }
        let literal = call
            .args
            .first()
            .filter(|a| a.spread.is_none())
            .and_then(|argument| match argument.expr.as_ref() {
                Expr::Lit(Lit::Str(requested)) => Some(requested),
                _ => None,
            });
        match literal {
            Some(requested) => self.check(call.span, &requested.value.to_string_lossy()),
            None => self.reject_computed(call.span),
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
    fn a_computed_dynamic_import_is_refused_where_it_is_written() {
        // Nothing can judge it before evaluation, and leaving it to the engine is the one module
        // reference that would fail with no line or column — the thing every other check here
        // exists to avoid.
        let error = transpile_with_includes(
            "w",
            "const which = \"ratatoskr/nodes\";\n\
             export async function plan() {\n  return await import(which);\n}",
            &[],
            &["ratatoskr/nodes"],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("w:3:16:"), "{error}");
        assert!(
            error.contains("import(..) with a computed specifier"),
            "{error}"
        );
        assert!(error.contains("statically known"), "{error}");
    }

    #[test]
    fn nesting_past_the_limit_is_refused_before_anything_parses_it() {
        // 150 levels of parentheses is a 300-byte source, and every pass this module runs after
        // parsing — the LOAD rewrite, resolver, strip, fixer, codegen — is unguarded recursion:
        // reaching the parser with this aborts the process with a stack overflow. So the refusal
        // has to come from the pre-parse scan, and nothing in this test may reach swc.
        let source = format!("const x = {}1{};", "(".repeat(150), ")".repeat(150));
        let error = transpile_with_includes("w", &source, &[], &[])
            .unwrap_err()
            .to_string();
        // Positioned at the bracket that crossed the limit, like every other refusal here.
        assert!(error.starts_with("transpile error: w:1:"), "{error}");
        assert!(
            error.contains(&format!("nesting limit is {MAX_NESTING_DEPTH}")),
            "{error}"
        );
    }

    #[test]
    fn nesting_counts_brackets_that_are_code_and_not_brackets_in_prose() {
        // Prose is where brackets go unbalanced. Each of these opens far past the limit and closes
        // nothing, and none of it is a bracket the parser will ever recurse into.
        let opens = "(".repeat(MAX_NESTING_DEPTH + 1);
        let source = format!(
            "/* {opens} a block comment weighing the options */\n\
             // {opens} a line comment doing the same\n\
             const prose = \"{opens} in a string\";\n\
             const more = 'and {opens} in the other quote';\n\
             const templated = `a template {opens} holding text`;\n\
             const pattern = /{opens}/;\n\
             export const x = 1;\n"
        );
        transpile_with_includes("w", &source, &[], &[]).unwrap();
    }

    #[test]
    fn a_division_before_deep_nesting_does_not_hide_it() {
        // `/` is a regex only where one can start. Reading the division here as a regex would run
        // the scan to the next `/` in the file and swallow the nesting after it — the failure mode
        // that makes skipping the wrong thing worse than counting too much.
        let source = format!(
            "const half = (1 + 2) / 3;\nconst deep = {}1{};\n",
            "(".repeat(MAX_NESTING_DEPTH + 1),
            ")".repeat(MAX_NESTING_DEPTH + 1)
        );
        let error = transpile_with_includes("w", &source, &[], &[])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!("nesting limit is {MAX_NESTING_DEPTH}")),
            "{error}"
        );
    }

    #[test]
    fn a_template_interpolation_is_code_and_its_brackets_count() {
        // `${` opens a brace like any other: the expression inside it is code the parser recurses
        // into, so leaving template text must not mean leaving the count.
        let source = format!(
            "const x = `text ${{{}1{}}} more`;\n",
            "(".repeat(MAX_NESTING_DEPTH),
            ")".repeat(MAX_NESTING_DEPTH)
        );
        let error = transpile_with_includes("w", &source, &[], &[])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!("nesting limit is {MAX_NESTING_DEPTH}")),
            "{error}"
        );
    }

    #[test]
    fn nesting_a_real_source_reaches_still_transpiles() {
        // The deepest file in this repository, prose included, nests 19 brackets. A limit that
        // refused that would be a worse bug than the one it fixes.
        let source = format!("export const x = {}1{};", "[".repeat(19), "]".repeat(19));
        transpile_with_includes("w", &source, &[], &[]).unwrap();
    }

    #[test]
    fn a_source_past_the_byte_ceiling_is_refused_without_being_read() {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-transpile-oversize-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("workflow.ts");
        std::fs::write(&path, "/".repeat(MAX_SCRIPT_BYTES as usize + 1)).unwrap();
        let error = match read_script_source(&path) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a source past the ceiling must be refused, not read"),
        };
        assert!(error.contains("workflow.ts"), "{error}");
        assert!(
            error.contains(&format!("source limit is {MAX_SCRIPT_BYTES} bytes")),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
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
