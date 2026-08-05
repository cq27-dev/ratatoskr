//! TypeScript → JavaScript via swc (type stripping): parse → resolver → strip → codegen.

use swc_core::common::sync::Lrc;
use swc_core::common::{FileName, GLOBALS, Mark, SourceMap};
use swc_core::ecma::ast::Program;
use swc_core::ecma::codegen::text_writer::JsWriter;
use swc_core::ecma::codegen::{Config as CodegenConfig, Emitter};
use swc_core::ecma::parser::{Parser, StringInput, Syntax, TsSyntax, lexer::Lexer};
use swc_core::ecma::transforms::base::resolver;
use swc_core::ecma::transforms::typescript::strip;

use crate::ScriptError;

/// Strip TypeScript types from `src`, returning JavaScript.
pub fn transpile_ts(src: &str) -> Result<String, ScriptError> {
    GLOBALS.set(&Default::default(), || {
        let cm: Lrc<SourceMap> = Default::default();
        let fm = cm.new_source_file(Lrc::new(FileName::Anon), src.to_string());

        let lexer = Lexer::new(
            Syntax::Typescript(TsSyntax::default()),
            Default::default(),
            StringInput::from(&*fm),
            None,
        );
        let mut parser = Parser::new_from(lexer);
        let mut program: Program = parser
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

        let unresolved = Mark::new();
        let top_level = Mark::new();
        program.mutate(resolver(unresolved, top_level, true));
        program.mutate(strip(unresolved, top_level));

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
                .emit_program(&program)
                .map_err(|e| ScriptError::Transpile(format!("codegen: {e}")))?;
        }
        String::from_utf8(buf).map_err(|e| ScriptError::Transpile(e.to_string()))
    })
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
}
