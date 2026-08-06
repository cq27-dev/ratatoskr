//! The file-reading tools a plugin's hooks are written against.
//!
//! `Read`, `Grep` and `Glob`, under those names and with those argument shapes, because that is
//! what a plugin matches on and inspects: a `PreToolUse` hook keyed to `^(Read|Grep)$` reading
//! `tool_input.file_path` fires here exactly as it does in the host the plugin was written for.
//!
//! Read-only, deliberately. `Write`, `Edit` and `Bash` belong to the implementer, which delegates
//! them to a coding CLI inside a sandboxed worktree; a planning node that could edit the checkout
//! it is reasoning about would undo that separation for nothing. A node that wants a change
//! proposes one.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_agent::tool::{DynamicTool, ToolExecutionError};
use rig_core::tool::ToolOutput;
use rmcp::model::Tool;
use serde_json::json;

/// Most of one file that `Read` will return. Past this a node is reading the wrong thing — that is
/// what `Grep` is for.
const MAX_READ_BYTES: usize = 256 * 1024;

/// Most matches or paths any one call reports.
const MAX_RESULTS: usize = 200;

/// Directories never walked. Without a `.gitignore` reader this is the approximation that keeps a
/// `Grep` over a Rust repository from walking a multi-gigabyte build directory.
// ponytail: fixed skip list; read .gitignore if a repo needs more than this.
const SKIP: [&str; 5] = [".git", "target", "node_modules", ".venv", "dist"];

/// The names these tools are offered under. A ruleset denies them by these names, and a plugin's
/// matcher matches them by these names.
pub const READ: &str = "Read";
pub const GREP: &str = "Grep";
pub const GLOB: &str = "Glob";

/// Declarations for a node's tool set, so `allow`/`deny` and the collision rule see them like any
/// other tool.
pub fn declarations() -> Vec<Tool> {
    vec![
        declare(
            READ,
            "Read a file from the repository. Returns its lines, numbered.",
            json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file. A path relative to the \
                            repository root is also accepted."
                    },
                    "offset": { "type": "integer", "description": "First line to read (1-based)." },
                    "limit": { "type": "integer", "description": "How many lines to read." }
                },
                "required": ["file_path"]
            }),
        ),
        declare(
            GREP,
            "Search the repository's files for a regular expression.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "A regular expression." },
                    "path": { "type": "string", "description": "File or directory to search." },
                    "glob": { "type": "string", "description": "Only files matching this glob." },
                    "-i": { "type": "boolean", "description": "Case-insensitive." },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "Matching lines, matching files, or a count per file."
                    },
                    "head_limit": { "type": "integer", "description": "Keep only the first N." }
                },
                "required": ["pattern"]
            }),
        ),
        declare(
            GLOB,
            "Find files in the repository by glob pattern.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "A glob, e.g. `src/**/*.rs`." },
                    "path": { "type": "string", "description": "Directory to search within." }
                },
                "required": ["pattern"]
            }),
        ),
    ]
}

fn declare(name: &'static str, description: &str, schema: serde_json::Value) -> Tool {
    let mut tool = Tool::default();
    tool.name = name.into();
    tool.description = Some(description.to_string().into());
    tool.input_schema = Arc::new(schema.as_object().cloned().unwrap_or_default());
    tool
}

/// The implementation of `name`, rooted at `root`, or `None` if it is not one of these.
pub fn implementation(name: &str, root: &Path) -> Option<DynamicTool> {
    let declaration = declarations().into_iter().find(|t| t.name == name)?;
    let root = root.to_path_buf();
    let schema = serde_json::Value::Object((*declaration.input_schema).clone());
    let description = declaration
        .description
        .clone()
        .unwrap_or_default()
        .to_string();
    let name = name.to_string();

    Some(DynamicTool::new(
        name.clone(),
        description,
        schema,
        move |_ctx, args| {
            let (root, name) = (root.clone(), name.clone());
            Box::pin(async move {
                let answer = match name.as_str() {
                    READ => read(&root, &args),
                    GREP => grep(&root, &args),
                    _ => glob_files(&root, &args),
                };
                answer.map(ToolOutput::text)
            })
        },
    ))
}

/// A required string argument.
fn arg<'a>(args: &'a serde_json::Value, key: &str) -> Result<&'a str, ToolExecutionError> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolExecutionError::invalid_args(format!("`{key}` is required")))
}

/// Resolve a caller-supplied path against the repository, refusing anything outside it.
///
/// A planning node reasons about *this* repository; a path that leaves it is either a mistake or
/// an attempt to read something the run was never given.
fn within(root: &Path, path: Option<&str>) -> Result<PathBuf, ToolExecutionError> {
    // Resolved against the canonical root, so containment is judged between two absolute paths
    // even when the run was started from a relative one.
    let base = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let joined = match path {
        Some(path) if Path::new(path).is_absolute() => PathBuf::from(path),
        Some(path) => base.join(path),
        None => base.clone(),
    };
    let real = joined.canonicalize().unwrap_or(joined);
    if !real.starts_with(&base) {
        return Err(ToolExecutionError::invalid_args(format!(
            "{} is outside this repository",
            real.display()
        )));
    }
    Ok(real)
}

/// `Read`: one file's lines, numbered, from `offset` for `limit` lines.
fn read(root: &Path, args: &serde_json::Value) -> Result<String, ToolExecutionError> {
    let path = within(root, Some(arg(args, "file_path")?))?;
    let text = std::fs::read_to_string(&path)
        .map_err(|e| ToolExecutionError::other(format!("cannot read {}: {e}", path.display())))?;

    let offset = args
        .get("offset")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .max(1) as usize;
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(u64::MAX) as usize;

    let mut out = String::new();
    for (number, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
        // Numbered because that is how a node cites what it read, and how it asks for more.
        out.push_str(&format!("{:>6}\t{line}\n", number + 1));
        if out.len() > MAX_READ_BYTES {
            out.push_str("… truncated; read a narrower range or use Grep\n");
            break;
        }
    }
    Ok(out)
}

/// `Grep`: a regular expression over the repository's files.
fn grep(root: &Path, args: &serde_json::Value) -> Result<String, ToolExecutionError> {
    let pattern = arg(args, "pattern")?;
    let insensitive = args.get("-i").and_then(|v| v.as_bool()).unwrap_or(false);
    let re = regex::RegexBuilder::new(pattern)
        .case_insensitive(insensitive)
        .build()
        .map_err(|e| ToolExecutionError::invalid_args(format!("invalid pattern: {e}")))?;

    let filter = args
        .get("glob")
        .and_then(|v| v.as_str())
        .map(glob::Pattern::new)
        .transpose()
        .map_err(|e| ToolExecutionError::invalid_args(format!("invalid glob: {e}")))?;

    let mode = args
        .get("output_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("content");
    let head = args
        .get("head_limit")
        .and_then(|v| v.as_u64())
        .map_or(MAX_RESULTS, |n| (n as usize).min(MAX_RESULTS));

    let mut lines: Vec<String> = Vec::new();
    let mut counts: Vec<String> = Vec::new();
    for file in walk(&within(root, args.get("path").and_then(|v| v.as_str()))?) {
        if let Some(filter) = &filter
            && !filter.matches_path(file.strip_prefix(root).unwrap_or(&file))
        {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue; // Binary, or unreadable. Not an error; just not a match.
        };
        let shown = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .display()
            .to_string();
        let mut hits = 0usize;
        for (number, line) in text.lines().enumerate() {
            if !re.is_match(line) {
                continue;
            }
            hits += 1;
            if mode == "content" && lines.len() < head {
                lines.push(format!("{shown}:{}:{}", number + 1, line.trim_end()));
            }
        }
        if hits > 0 {
            match mode {
                "files_with_matches" => lines.push(shown),
                "count" => counts.push(format!("{shown}:{hits}")),
                _ => {}
            }
        }
        if lines.len() >= head && mode != "count" {
            break;
        }
    }

    let found = if mode == "count" { counts } else { lines };
    Ok(match found.is_empty() {
        true => "no matches".to_string(),
        false => found.join("\n"),
    })
}

/// `Glob`: paths matching a pattern.
fn glob_files(root: &Path, args: &serde_json::Value) -> Result<String, ToolExecutionError> {
    let pattern = glob::Pattern::new(arg(args, "pattern")?)
        .map_err(|e| ToolExecutionError::invalid_args(format!("invalid glob: {e}")))?;
    let base = within(root, args.get("path").and_then(|v| v.as_str()))?;

    let found: Vec<String> = walk(&base)
        .filter(|p| pattern.matches_path(p.strip_prefix(root).unwrap_or(p)))
        .take(MAX_RESULTS)
        .map(|p| p.strip_prefix(root).unwrap_or(&p).display().to_string())
        .collect();

    Ok(match found.is_empty() {
        true => "no files matched".to_string(),
        false => found.join("\n"),
    })
}

/// Every file under `base`, skipping the directories nothing wants searched.
fn walk(base: &Path) -> impl Iterator<Item = PathBuf> + use<> {
    walkdir::WalkDir::new(base)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !SKIP.contains(&name.as_ref()) && !name.starts_with('.') || e.depth() == 0
        })
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A repository with a couple of files in it.
    fn repo(case: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("ratatoskr-files-{}-{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {\n    todo!()\n}\n").unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn helper() {}\n").unwrap();
        std::fs::write(root.join("target/huge.rs"), "fn main() {}\n").unwrap();
        root
    }

    #[test]
    fn read_numbers_its_lines_and_honours_a_range() {
        let root = repo("read");
        let all = read(&root, &json!({ "file_path": "src/main.rs" })).unwrap();
        assert!(all.starts_with("     1\tfn main() {"), "{all}");

        let one = read(
            &root,
            &json!({ "file_path": "src/main.rs", "offset": 2, "limit": 1 }),
        )
        .unwrap();
        assert_eq!(one, "     2\t    todo!()\n");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn grep_reports_the_modes_the_format_defines() {
        let root = repo("grep");
        let content = grep(&root, &json!({ "pattern": "fn main" })).unwrap();
        assert!(content.contains("src/main.rs:1:fn main() {"), "{content}");
        // Never the build directory.
        assert!(!content.contains("target/"), "{content}");

        let files = grep(
            &root,
            &json!({ "pattern": "fn", "output_mode": "files_with_matches" }),
        )
        .unwrap();
        assert!(files.contains("src/main.rs") && files.contains("src/lib.rs"));

        let counts = grep(&root, &json!({ "pattern": "fn", "output_mode": "count" })).unwrap();
        assert!(counts.contains("src/lib.rs:1"), "{counts}");

        // A glob narrows it, and a pattern that matches nothing says so.
        let narrowed = grep(&root, &json!({ "pattern": "fn", "glob": "**/lib.rs" })).unwrap();
        assert!(narrowed.contains("lib.rs") && !narrowed.contains("main.rs"));
        assert_eq!(
            grep(&root, &json!({ "pattern": "zzz" })).unwrap(),
            "no matches"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn glob_finds_files_and_skips_what_nothing_wants_searched() {
        let root = repo("glob");
        let found = glob_files(&root, &json!({ "pattern": "**/*.rs" })).unwrap();
        assert!(found.contains("src/main.rs") && found.contains("src/lib.rs"));
        assert!(!found.contains("target/"), "{found}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn nothing_outside_the_repository_can_be_read() {
        // A planning node reasons about this repository. A path that leaves it is a mistake or an
        // attempt to read something the run was never given.
        let root = repo("escape");
        for path in ["../../../etc/passwd", "/etc/passwd"] {
            let err = read(&root, &json!({ "file_path": path })).expect_err("refused");
            assert!(err.to_string().contains("outside this repository"), "{err}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_bad_pattern_is_a_refusal_the_model_can_act_on() {
        let root = repo("bad-pattern");
        let err = grep(&root, &json!({ "pattern": "^(unclosed" })).expect_err("refused");
        assert!(err.to_string().contains("invalid pattern"), "{err}");
        let err = grep(&root, &json!({})).expect_err("refused");
        assert!(err.to_string().contains("`pattern` is required"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
