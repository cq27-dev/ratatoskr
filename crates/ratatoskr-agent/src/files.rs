//! The file-reading tools a plugin's hooks are written against.
//!
//! `Read`, `Grep` and `Glob`, under those names and with those argument shapes, because that is
//! what a plugin matches on and inspects: a `PreToolUse` hook keyed to `^(Read|Grep)$` reading
//! `tool_input.file_path` fires here exactly as it does in the host the plugin was written for.
//!
//! [`declarations`] is read-only, deliberately: a planning node that could edit the checkout it is
//! reasoning about would undo that separation for nothing. A node that wants a change proposes one.
//!
//! [`edit_declarations`] is the write-capable set — `Write` and `Edit`, under the names and argument
//! shapes a plugin matches on — and is offered only to the implementer. The two are separate
//! functions rather than a flag because the read-only guarantee should be visible at the call site
//! that grants it, not buried in a boolean.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_agent::tool::{DynamicTool, ToolExecutionError};
use rig_core::tool::ToolOutput;
use rmcp::model::Tool;
use serde_json::json;

/// Most of one file that `Read` will return. Past this a node is reading the wrong thing — that is
/// what `Grep` is for.
const MAX_READ_BYTES: usize = 256 * 1024;

/// Most of one line reported back. A minified bundle or an embedded data blob is a single line of
/// hundreds of kilobytes, and no node ever needed the whole of it.
const MAX_LINE_CHARS: usize = 2_000;

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
pub const WRITE: &str = "Write";
pub const EDIT: &str = "Edit";

/// Most one `Write` may put on disk. A model that means to write more than this has lost the plot,
/// and the worktree is on the host's filesystem.
const MAX_WRITE_BYTES: usize = 4 * 1024 * 1024;

/// Declarations for a node's tool set, so `allow`/`deny` and the collision rule see them like any
/// other tool.
pub fn declarations() -> Vec<Tool> {
    vec![
        declare(
            READ,
            "Return a file's contents with 1-based line numbers. offset selects the first line to return, \
            limit caps the line count — use them for large files, since output is capped at 256 KB. Read \
            before any Edit so old_string can be copied exactly (strip the line-number prefix). To \
            locate which files mention something, use Grep instead of reading files one by one.",
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
            "Search file contents across the repository with a regex. Prefer this over Reading many files \
            when hunting for a symbol, string, or pattern. path narrows to a file or directory, glob \
            filters filenames, -i makes it case-insensitive. output_mode: files_with_matches lists \
            matching files (best first pass), content shows matching lines, count gives per-file totals; \
            head_limit truncates output. Skips .git, target, node_modules, .venv, dist; results cap at \
            200 — if you hit the cap, narrow the pattern or path. Zero matches often means the pattern \
            is too strict: loosen it or drop anchors before concluding the code is absent.",
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
            "Find files by name pattern, e.g. src/**/*.rs, with optional path to scope the search. Use \
            this to discover file layout or locate a file whose name you partly know; use Grep when you \
            are matching file contents rather than names.",
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

/// The write-capable tools. Offered to the implementer and to nothing else — see the module docs.
///
/// Names and argument shapes follow the coding CLIs a plugin is written against, so a `PreToolUse`
/// hook keyed to `^(Write|Edit)$` inspecting `tool_input.file_path` fires here as it does there.
pub fn edit_declarations() -> Vec<Tool> {
    vec![
        declare(
            WRITE,
            "Write content to file_path, creating parent directories as needed. If the file exists it is \
            replaced entirely, so any lines you do not restate are lost — prefer Edit for modifying an \
            existing file and reserve Write for new files or intentional full rewrites. Paths outside \
            the repository are refused. Content over 4 MB is refused. To overwrite safely, Read the file \
            first so you know what you are discarding.",
            json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file. A path relative to the \
                            repository root is also accepted. Parent directories are created."
                    },
                    "content": { "type": "string", "description": "The file's complete contents." }
                },
                "required": ["file_path", "content"]
            }),
        ),
        declare(
            EDIT,
            "Replace one exact string in a file. old_string must match the file byte-for-byte, including \
            every space, tab, and newline of indentation — copy it from Read output (dropping the \
            line-number prefix), never retype it from memory. If old_string occurs more than once, the \
            call fails and nothing is written; fix by extending old_string with surrounding lines until \
            it is unique, or pass replace_all: true to change every occurrence. If old_string occurs \
            zero times, the tool retries with whitespace-normalized matching (lines trimmed, blank lines \
            ignored): exactly one region matching that way is edited and the result says so; several \
            such regions still fail, and the error quotes the nearest file lines with numbers — copy \
            your next old_string directly from that quote or re-Read the region. old_string identical to \
            new_string fails. Non-UTF-8 files are refused. Existing LF/CRLF line endings are preserved. \
            Prefer this over Write for any change to an existing file: it touches only the lines you \
            name.",
            json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "The file to edit." },
                    "old_string": {
                        "type": "string",
                        "description": "Text to replace, matched exactly including indentation."
                    },
                    "new_string": { "type": "string", "description": "What to replace it with." },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of requiring exactly one."
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
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
///
/// Covers both sets. Which tools a node may actually call is decided by what was declared for it —
/// this only says how each one behaves once called.
pub fn implementation(name: &str, root: &Path) -> Option<DynamicTool> {
    let declaration = declarations()
        .into_iter()
        .chain(edit_declarations())
        .find(|t| t.name == name)?;
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
                    WRITE => write(&root, &args),
                    EDIT => edit(&root, &args),
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
    let real = resolve(&joined);
    if !real.starts_with(&base) {
        return Err(ToolExecutionError::invalid_args(format!(
            "{} is outside this repository",
            real.display()
        )));
    }
    Ok(real)
}

/// Resolve a path as the OS will, as far as the filesystem can answer.
///
/// `canonicalize` on the whole path is not enough, because it fails outright for a path whose last
/// component does not exist — which is every `Write` that creates a file. Falling back to a purely
/// lexical normalisation is not enough either: it treats a symlinked directory as a directory
/// name, so `foo/new.txt` with `foo -> /tmp/elsewhere` passes a containment check and then lands
/// wherever `foo` actually points.
///
/// So resolve one component at a time, canonicalising each prefix that exists. A symlink is
/// followed where the filesystem knows about it, a name that does not exist yet is carried
/// forward as written, and `..` pops from the resolved prefix rather than from the text.
///
/// This is a check, and the write that follows it is a second syscall: a symlink planted in
/// between still wins. Closing that means performing the write inside the sandbox rather than
/// deciding about it out here, which is the same boundary `Bash` already runs behind.
fn resolve(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            part => {
                out.push(part);
                if let Ok(real) = out.canonicalize() {
                    out = real;
                }
            }
        }
    }
    out
}

/// `Write`: put `content` at `file_path`, creating parents.
fn write(root: &Path, args: &serde_json::Value) -> Result<String, ToolExecutionError> {
    let path = within(root, Some(arg(args, "file_path")?))?;
    let content = arg(args, "content")?;
    if content.len() > MAX_WRITE_BYTES {
        return Err(ToolExecutionError::invalid_args(format!(
            "refusing to write {} bytes to {}; the cap is {MAX_WRITE_BYTES}",
            content.len(),
            path.display()
        )));
    }
    if path.is_dir() {
        return Err(ToolExecutionError::invalid_args(format!(
            "{} is a directory",
            path.display()
        )));
    }
    // Created here rather than failing: a change that adds a module adds its directory too, and
    // making the model call a separate tool for that is a turn spent on nothing.
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            ToolExecutionError::other(format!("could not create {}: {e}", parent.display()))
        })?;
    }
    let existed = path.exists();
    std::fs::write(&path, content).map_err(|e| {
        ToolExecutionError::other(format!("could not write {}: {e}", path.display()))
    })?;
    let what = if existed { "Replaced" } else { "Created" };
    Ok(format!(
        "{what} {} ({} bytes, {} lines).",
        display(root, &path),
        content.len(),
        content.lines().count()
    ))
}

/// `Edit`: replace an exact string.
///
/// Exactness is the contract, and the failures are deliberately loud. A `no match` that silently
/// did nothing, or an ambiguous match that changed the first occurrence, both look like a
/// successful edit to the model — which then builds its next step on a change that never happened.
fn edit(root: &Path, args: &serde_json::Value) -> Result<String, ToolExecutionError> {
    let path = within(root, Some(arg(args, "file_path")?))?;
    let old = arg(args, "old_string")?;
    let new = args
        .get("new_string")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolExecutionError::invalid_args("`new_string` is required"))?;
    let all = args
        .get("replace_all")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if old == new {
        return Err(ToolExecutionError::invalid_args(
            "`old_string` and `new_string` are identical; this edit would change nothing",
        ));
    }
    if old.is_empty() {
        return Err(ToolExecutionError::invalid_args(
            "`old_string` is empty; use Write to create a file",
        ));
    }

    let before = std::fs::read_to_string(&path).map_err(|e| {
        ToolExecutionError::other(format!("could not read {}: {e}", path.display()))
    })?;

    let count = before.matches(old).count();
    if count == 0 {
        // Exact matching failed. Before giving up, look for the region whose *normalized* form
        // matches — same tokens, different indentation or blank lines. That is the difference a
        // model gets wrong most often and the one that matters least, and requiring the match to be
        // unique keeps this from editing an arbitrary candidate.
        if let Some(at) = locate(&before, old) {
            let lines: Vec<&str> = before.lines().collect();
            let needle_indent = old
                .lines()
                .find(|l| !l.trim().is_empty())
                .map(|l| &l[..l.len() - l.trim_start().len()])
                .unwrap_or("");
            let replacement = reindent(new, needle_indent, &at.indent);
            let mut out: Vec<String> = lines[..at.start].iter().map(|l| l.to_string()).collect();
            out.extend(replacement.lines().map(str::to_string));
            out.extend(lines[at.end..].iter().map(|l| l.to_string()));
            // `str::lines` strips a trailing `\r`, so rejoining with `\n` would quietly convert a
            // CRLF file to mixed endings — a whole-file diff dressed up as a one-line edit.
            let eol = if before.contains("\r\n") {
                "\r\n"
            } else {
                "\n"
            };
            let mut after = out.join(eol);
            if before.ends_with('\n') {
                after.push_str(eol);
            }
            std::fs::write(&path, &after).map_err(|e| {
                ToolExecutionError::other(format!("could not write {}: {e}", path.display()))
            })?;
            // Said plainly: the edit landed somewhere the arguments did not name exactly, and the
            // caller should know that rather than assume a literal match.
            return Ok(format!(
                "Edited {} at lines {}-{}, matched ignoring indentation and blank lines.",
                display(root, &path),
                at.start + 1,
                at.end
            ));
        }
        return Err(ToolExecutionError::invalid_args(format!(
            "`old_string` does not appear in {}{}",
            display(root, &path),
            not_found_help(&before, old)
        )));
    }
    if count > 1 && !all {
        return Err(ToolExecutionError::invalid_args(format!(
            "`old_string` appears {count} times in {}; include enough surrounding lines to make it \
             unique, or set `replace_all`",
            display(root, &path)
        )));
    }

    let after = if all {
        before.replace(old, new)
    } else {
        before.replacen(old, new, 1)
    };
    std::fs::write(&path, &after).map_err(|e| {
        ToolExecutionError::other(format!("could not write {}: {e}", path.display()))
    })?;
    Ok(format!(
        "Edited {} ({} replacement{}).",
        display(root, &path),
        if all { count } else { 1 },
        if all && count != 1 { "s" } else { "" }
    ))
}

/// A region of the file whose normalized form matches what the edit was aiming at.
struct Located {
    start: usize,
    end: usize,
    /// Leading whitespace the file uses on the first matched line, for re-indenting the
    /// replacement.
    indent: String,
}

/// Normalize as rag-rat anchors chunks: trim each line, drop the blank ones.
///
/// Indentation and blank lines are the two things a model reproduces least reliably and that matter
/// least to identity. Everything else — every token, in order — still has to match exactly.
fn normalized(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect()
}

/// The single line range whose normalized form equals the needle's, or `None` when there is none or
/// more than one.
///
/// The uniqueness requirement is the whole safety argument: relocating to *a* match when several
/// exist would edit an arbitrary one, which is the failure the exact-match rule exists to prevent.
fn locate(haystack: &str, needle: &str) -> Option<Located> {
    let wanted = normalized(needle);
    if wanted.is_empty() {
        return None;
    }
    let lines: Vec<&str> = haystack.lines().collect();
    let mut found: Option<Located> = None;
    for start in 0..lines.len() {
        // Walk forward until as many non-blank lines as the needle has are covered.
        let mut seen = 0usize;
        let mut end = start;
        while end < lines.len() && seen < wanted.len() {
            if !lines[end].trim().is_empty() {
                seen += 1;
            }
            end += 1;
        }
        if seen != wanted.len() {
            break;
        }
        if normalized(&lines[start..end].join("\n")) != wanted {
            continue;
        }
        if found.is_some() {
            return None;
        }
        let first = lines[start..end]
            .iter()
            .find(|l| !l.trim().is_empty())
            .unwrap_or(&"");
        found = Some(Located {
            start,
            end,
            indent: first[..first.len() - first.trim_start().len()].to_string(),
        });
    }
    found
}

/// Re-indent `text` by the difference between the file's indentation and the needle's.
///
/// Without this, relocating an edit whose `old_string` was under-indented would write the
/// replacement back at the model's indentation and silently reformat the block.
fn reindent(text: &str, from: &str, to: &str) -> String {
    if from == to {
        return text.to_string();
    }
    text.lines()
        .map(|l| match l.strip_prefix(from) {
            Some(rest) if !l.trim().is_empty() => format!("{to}{rest}"),
            _ if l.trim().is_empty() => l.to_string(),
            _ => format!("{to}{}", l.trim_start()),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Why an exact match failed, and — when nothing more specific applies — what the file actually
/// says where the edit was aimed.
///
/// The content matters more than the diagnosis. A bare "could not find old_string" is a message a
/// model will answer by sending the same hallucinated string again, and again: opencode has this
/// filed as an infinite loop, ten-plus identical calls after the same error. Putting the real lines
/// in the error gives the next attempt the text to copy, instead of telling it to go and look.
fn not_found_help(haystack: &str, needle: &str) -> String {
    // Specific causes first: these say exactly what to change, which beats showing content.
    if haystack.contains('\r') && haystack.replace("\r\n", "\n").contains(needle) {
        return ". The file has CRLF line endings and `old_string` has LF — match the file's."
            .to_string();
    }
    let flattened = squash(needle);
    if !flattened.is_empty() && squash(haystack).contains(&flattened) {
        return ". The text is present but its whitespace differs — copy the indentation exactly."
            .to_string();
    }
    match nearest(haystack, needle) {
        Some(window) => format!(
            ". The closest the file comes is:\n{window}\nCopy from there verbatim; do not retype \
             `old_string` from memory."
        ),
        None => String::new(),
    }
}

/// Collapse runs of whitespace so two texts can be compared ignoring indentation.
fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The lines around the file's best match for the needle's first line, numbered.
///
/// Scored by shared tokens rather than edit distance: the failure this serves is a model that got
/// the identifiers right and the punctuation or spacing wrong, and token overlap finds that.
fn nearest(haystack: &str, needle: &str) -> Option<String> {
    let target = needle.lines().find(|l| !l.trim().is_empty())?;
    let wanted: Vec<&str> = target.split_whitespace().collect();
    if wanted.is_empty() {
        return None;
    }
    let lines: Vec<&str> = haystack.lines().collect();
    let (best, score) = lines
        .iter()
        .enumerate()
        .fold((0usize, 0usize), |acc, (i, l)| {
            let hits = wanted.iter().filter(|w| l.contains(**w)).count();
            if hits > acc.1 { (i, hits) } else { acc }
        });
    // No token in common means the guess is unrelated to this file; quoting an arbitrary window
    // would be inventing a suggestion rather than making one.
    if score == 0 {
        return None;
    }
    let from = best.saturating_sub(2);
    let to = (best + 3).min(lines.len());
    Some(
        lines[from..to]
            .iter()
            .enumerate()
            .map(|(n, l)| format!("{:>6}\t{l}", from + n + 1))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// A path as the caller is likely to recognise it: relative to the repository when it is inside.
fn display(root: &Path, path: &Path) -> String {
    let base = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    path.strip_prefix(&base)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// `Read`: one file's lines, numbered, from `offset` for `limit` lines.
fn read(root: &Path, args: &serde_json::Value) -> Result<String, ToolExecutionError> {
    let path = within(root, Some(arg(args, "file_path")?))?;
    if path.is_dir() {
        return Err(ToolExecutionError::invalid_args(format!(
            "{} is a directory, not a file; list it with Glob",
            display(root, &path)
        )));
    }
    if is_binary(&path)? {
        // Said plainly rather than left to fail as a UTF-8 error deeper in, which reads as a
        // broken tool rather than as a file nobody should be asking for by line.
        return Err(ToolExecutionError::invalid_args(format!(
            "{} is a binary file and has no lines to read",
            display(root, &path)
        )));
    }
    let offset = args
        .get("offset")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .max(1) as usize;
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(u64::MAX) as usize;

    // Streamed, not read whole: `offset` and `limit` bound what is *returned*, and a file large
    // enough to matter would otherwise be held in memory in full before either applied.
    let mut out = String::new();
    let mut last = 0;
    let mut cut = false;
    for (number, line) in lines_of(&path)?.enumerate().skip(offset - 1).take(limit) {
        let line = line.map_err(|e| {
            ToolExecutionError::other(format!("cannot read {}: {e}", path.display()))
        })?;
        last = number + 1;
        // Numbered because that is how a node cites what it read, and how it asks for more.
        out.push_str(&format!("{:>6}\t{}\n", last, clip(&line)));
        // Checked after the line is clipped, so the cap is a cap: a single minified line is not a
        // way to hand back a megabyte from a tool that promised a quarter of one.
        if out.len() > MAX_READ_BYTES {
            cut = true;
            break;
        }
    }
    if out.is_empty() {
        return Ok(match last {
            0 if offset > 1 => format!("(no lines from offset {offset}; the file is shorter)"),
            _ => "(empty file)".to_string(),
        });
    }
    if cut {
        // With the offset to continue from: "read less" leaves the model to work out where it got
        // to, and it is the one thing this function knows for certain.
        out.push_str(&format!(
            "… stopped at the size cap; continue with offset={}",
            last + 1
        ));
    }
    Ok(out)
}

/// One line, bounded. A generated or minified file is one line long, and it is never the line a
/// node needed in full.
fn clip(line: &str) -> String {
    match line.char_indices().nth(MAX_LINE_CHARS) {
        None => line.to_string(),
        Some((at, _)) => format!("{}… (line clipped at {MAX_LINE_CHARS} chars)", &line[..at]),
    }
}

/// Whether the file looks binary, by the same rule git uses: a NUL byte near the start.
fn is_binary(path: &Path) -> Result<bool, ToolExecutionError> {
    use std::io::Read as _;
    let mut file = std::fs::File::open(path)
        .map_err(|e| ToolExecutionError::other(format!("cannot read {}: {e}", path.display())))?;
    let mut head = [0u8; 8192];
    let n = file
        .read(&mut head)
        .map_err(|e| ToolExecutionError::other(format!("cannot read {}: {e}", path.display())))?;
    Ok(head[..n].contains(&0))
}

/// One file's lines, read as they are needed.
fn lines_of(
    path: &Path,
) -> Result<std::io::Lines<std::io::BufReader<std::fs::File>>, ToolExecutionError> {
    use std::io::BufRead as _;
    let file = std::fs::File::open(path)
        .map_err(|e| ToolExecutionError::other(format!("cannot read {}: {e}", path.display())))?;
    Ok(std::io::BufReader::new(file).lines())
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
    for file in walk(
        root,
        &within(root, args.get("path").and_then(|v| v.as_str()))?,
    ) {
        if let Some(filter) = &filter
            && !filter.matches_path(file.strip_prefix(root).unwrap_or(&file))
        {
            continue;
        }
        // Streamed, so a file large enough to matter is never held whole in memory.
        let Ok(reading) = lines_of(&file) else {
            continue; // Unreadable. Not an error; just not a match.
        };
        let shown = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .display()
            .to_string();
        let mut hits = 0usize;
        for (number, line) in reading.enumerate() {
            let Ok(line) = line else {
                break; // Not UTF-8 past here; whatever already matched still counts.
            };
            if !re.is_match(&line) {
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
        // Every mode stops at the same bound. `count` walking the whole tree regardless was a
        // long synchronous stall over a large repository, and an answer nothing had asked to cap.
        if lines.len().max(counts.len()) >= head {
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

    let found: Vec<String> = walk(root, &base)
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
///
/// `filter_entry` never rejects the walk root (depth 0), so a `base` that already sits inside a
/// SKIP directory would otherwise be searched in full — the skip is only escaped by rooting the
/// walk there. Judged against `root`, not by absolute component, so a repo checked out under a
/// directory literally named `target` stays searchable while a `path` pointing into the repo's
/// own `target` returns nothing.
fn walk(root: &Path, base: &Path) -> impl Iterator<Item = PathBuf> + use<> {
    // Judge the base's path relative to the canonical root — the same canonicalisation `within`
    // used to produce `base`, so the prefix actually strips.
    let canon = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let escaped = base
        .strip_prefix(&canon)
        .unwrap_or(base)
        .components()
        .any(|c| {
            let name = c.as_os_str().to_string_lossy();
            SKIP.contains(&name.as_ref()) || name.starts_with('.')
        });
    walkdir::WalkDir::new(base)
        .into_iter()
        .filter_entry(move |e| {
            if escaped {
                return false;
            }
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
    fn a_path_that_does_not_exist_cannot_escape_either() {
        // `canonicalize` fails for a path that is not there, and `starts_with` compares `..` as
        // if it were a directory name — so the fallback has to normalise before it compares.
        let root = repo("dangling");
        for path in [
            "../../../etc/nonexistent-xyz",
            "src/../../../../etc/nonexistent-xyz",
        ] {
            let err = read(&root, &json!({ "file_path": path })).expect_err("refused");
            assert!(
                err.to_string().contains("outside this repository"),
                "{path}: {err}"
            );
        }
        // And a symlink inside the repository pointing out of it is resolved, then refused.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc/hosts", root.join("escape")).unwrap();
            let err = read(&root, &json!({ "file_path": "escape" })).expect_err("refused");
            assert!(err.to_string().contains("outside this repository"), "{err}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_directory_does_not_carry_a_write_out_of_the_repository() {
        // The case a whole-path `canonicalize` cannot see: the file being written does not exist
        // yet, so the call fails and the check falls back to something that has to decide about
        // `link/new.txt` on its own. Treating `link` as a directory *name* passes it, and then the
        // write follows the symlink at the OS level and lands wherever it points.
        let root = repo("symlinked-parent");
        let outside =
            std::env::temp_dir().join(format!("ratatoskr-outside-the-repo-{}", std::process::id()));
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();

        for tool in ["write", "read"] {
            let args = json!({ "file_path": "link/new.txt", "content": "landed" });
            let err = match tool {
                "write" => write(&root, &args).expect_err("refused"),
                _ => read(&root, &args).expect_err("refused"),
            };
            assert!(err.to_string().contains("outside this repository"), "{err}");
        }
        assert!(
            !outside.join("new.txt").exists(),
            "nothing was written outside the repository"
        );

        // The same directory reached the ordinary way is still fine: this refuses an escape, not
        // every path with a symlink in it.
        std::os::unix::fs::symlink(root.join("src"), root.join("inside")).unwrap();
        write(
            &root,
            &json!({ "file_path": "inside/new.rs", "content": "fn x() {}" }),
        )
        .unwrap();
        assert!(root.join("src/new.rs").exists());

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn every_grep_mode_stops_at_the_same_bound() {
        // `count` used to walk the whole tree however large, ignoring `head_limit` entirely.
        let root = repo("bounded");
        for name in 0..8 {
            std::fs::write(root.join("src").join(format!("f{name}.rs")), "fn x() {}\n").unwrap();
        }
        for mode in ["content", "files_with_matches", "count"] {
            let out = grep(
                &root,
                &json!({ "pattern": "fn", "output_mode": mode, "head_limit": 3 }),
            )
            .unwrap();
            assert!(out.lines().count() <= 3, "{mode}: {out}");
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

    #[test]
    fn read_bounds_what_one_pathological_file_can_return() {
        let root = scratch("read-bounds");

        // One enormous line. Without a per-line clip the byte cap is decorative: the check happens
        // after a line is appended, so a single minified bundle returns in full.
        let minified = root.join("bundle.min.js");
        std::fs::write(&minified, "x".repeat(MAX_READ_BYTES * 2)).unwrap();
        let out = read(&root, &json!({ "file_path": "bundle.min.js" })).unwrap();
        assert!(out.len() < MAX_READ_BYTES, "returned {} bytes", out.len());
        assert!(out.contains("line clipped"), "{out}");

        // A binary file is refused by name rather than failing as a UTF-8 error, which reads as a
        // broken tool instead of a file nobody should be reading by line.
        let binary = root.join("logo.png");
        std::fs::write(&binary, [0x89, 0x50, 0x4e, 0x47, 0x00, 0x01, 0x02]).unwrap();
        let err = read(&root, &json!({ "file_path": "logo.png" }))
            .unwrap_err()
            .to_string();
        assert!(err.contains("binary"), "{err}");

        // A directory says what to use instead. A node that reads one is looking for a listing.
        let err = read(&root, &json!({ "file_path": "." }))
            .unwrap_err()
            .to_string();
        assert!(err.contains("directory") && err.contains("Glob"), "{err}");
    }

    #[test]
    fn read_says_when_there_is_nothing_rather_than_returning_nothing() {
        // An empty answer is ambiguous — unreadable, empty, or past the end all look identical —
        // and a model that cannot tell which will read it again.
        let root = scratch("read-empty");
        std::fs::write(root.join("empty.rs"), "").unwrap();
        assert!(
            read(&root, &json!({ "file_path": "empty.rs" }))
                .unwrap()
                .contains("empty file")
        );

        std::fs::write(root.join("short.rs"), "one\ntwo\n").unwrap();
        let past = read(&root, &json!({ "file_path": "short.rs", "offset": 99 })).unwrap();
        assert!(past.contains("shorter"), "{past}");
    }

    #[test]
    fn a_truncated_read_says_where_to_continue_from() {
        // "read less" leaves the model to work out where it got to; the offset is the one thing
        // this function knows for certain.
        let root = scratch("read-continue");
        let line = format!("{}\n", "y".repeat(500));
        std::fs::write(root.join("big.rs"), line.repeat(2000)).unwrap();
        let out = read(&root, &json!({ "file_path": "big.rs" })).unwrap();
        assert!(out.contains("continue with offset="), "{out}");
    }

    fn scratch(case: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ratatoskr-edit-{}-{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_creates_parents_and_edit_replaces_exactly_once() {
        let root = scratch("basic");
        let created = write(
            &root,
            &json!({ "file_path": "src/deep/new.rs", "content": "fn a() {}\nfn b() {}\n" }),
        )
        .unwrap();
        assert!(created.starts_with("Created"), "{created}");
        assert!(root.join("src/deep/new.rs").is_file());

        let edited = edit(
            &root,
            &json!({ "file_path": "src/deep/new.rs", "old_string": "fn a() {}",
                     "new_string": "fn a() -> u8 { 1 }" }),
        )
        .unwrap();
        assert!(edited.starts_with("Edited"), "{edited}");
        let now = std::fs::read_to_string(root.join("src/deep/new.rs")).unwrap();
        assert_eq!(now, "fn a() -> u8 { 1 }\nfn b() {}\n");

        // Writing an existing file says so, rather than reporting a create.
        let again = write(
            &root,
            &json!({ "file_path": "src/deep/new.rs", "content": "x" }),
        )
        .unwrap();
        assert!(again.starts_with("Replaced"), "{again}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_ambiguous_edit_fails_instead_of_changing_the_first_match() {
        let root = scratch("ambiguous");
        std::fs::write(root.join("f.rs"), "let x = 1;\nlet x = 1;\n").unwrap();

        let err = edit(
            &root,
            &json!({ "file_path": "f.rs", "old_string": "let x = 1;", "new_string": "let x = 2;" }),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("appears 2 times"), "{err}");
        // Nothing was written: an ambiguous edit that silently took the first match would look
        // identical to a successful one from the model's side.
        assert_eq!(
            std::fs::read_to_string(root.join("f.rs")).unwrap(),
            "let x = 1;\nlet x = 1;\n"
        );

        let ok = edit(
            &root,
            &json!({ "file_path": "f.rs", "old_string": "let x = 1;", "new_string": "let x = 2;",
                     "replace_all": true }),
        )
        .unwrap();
        assert!(ok.contains("2 replacements"), "{ok}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_failed_match_hands_back_the_text_to_copy() {
        // The loop this prevents is documented upstream: told only "could not find old_string", a
        // model re-sends the same hallucinated string ten times over. The error carries the real
        // lines so the next attempt has something to copy.
        let root = scratch("nearest");
        std::fs::write(
            root.join("f.rs"),
            "fn main() {\n    if guess < secret_number {\n        println!(\"low\");\n    }\n}\n",
        )
        .unwrap();

        let err = edit(
            &root,
            &json!({ "file_path": "f.rs", "old_string": "if guess <<< secret secret_number {",
                     "new_string": "if guess > secret_number {" }),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("closest the file comes"), "{err}");
        assert!(err.contains("if guess < secret_number {"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_specific_causes_are_named_before_content_is_quoted() {
        let root = scratch("causes");
        // A CRLF file against an LF needle relocates rather than failing — and keeps its endings.
        // Rejoining with `\n` would turn a one-line edit into a whole-file diff.
        std::fs::write(root.join("crlf.rs"), "fn a() {}\r\nfn b() {}\r\n").unwrap();
        let ok = edit(
            &root,
            &json!({ "file_path": "crlf.rs", "old_string": "fn a() {}\nfn b() {}",
                     "new_string": "fn c() {}" }),
        )
        .unwrap();
        assert!(ok.contains("matched ignoring"), "{ok}");
        assert_eq!(
            std::fs::read_to_string(root.join("crlf.rs")).unwrap(),
            "fn c() {}\r\n"
        );

        // Tabs in the file against spaces in the needle: no substring match, and no unique
        // normalized region either, because both lines normalize the same.
        std::fs::write(root.join("ws.rs"), "\tdeep();\n\tdeep();\n").unwrap();
        let err = edit(
            &root,
            &json!({ "file_path": "ws.rs", "old_string": "    deep();", "new_string": "x();" }),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("whitespace differs"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_non_utf8_file_is_refused_rather_than_mangled() {
        // opencode carries this as a corruption bug: read-as-UTF-8, write-as-UTF-8 destroys a
        // Windows-1252 file's accented bytes. Rust's `read_to_string` refuses instead, which is the
        // behaviour to keep — never "fix" this with `from_utf8_lossy`, which would corrupt exactly
        // as reported there.
        let root = scratch("latin1");
        std::fs::write(root.join("l.txt"), [0x66, 0x6f, 0x6f, 0xE7, 0x0a]).unwrap();
        let err = edit(
            &root,
            &json!({ "file_path": "l.txt", "old_string": "foo", "new_string": "bar" }),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("could not read"), "{err}");
        // The bytes are untouched.
        assert_eq!(
            std::fs::read(root.join("l.txt")).unwrap(),
            [0x66, 0x6f, 0x6f, 0xE7, 0x0a]
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_no_op_edit_and_an_escape_from_the_repository_are_both_refused() {
        let root = scratch("guards");
        std::fs::write(root.join("f.rs"), "x").unwrap();
        // Identical strings waste a turn and read as success.
        assert!(
            edit(
                &root,
                &json!({ "file_path": "f.rs", "old_string": "x", "new_string": "x" })
            )
            .is_err()
        );
        // Containment holds for writes to paths that do not exist yet, which is where
        // `canonicalize` alone fails.
        assert!(
            write(
                &root,
                &json!({ "file_path": "../escaped.rs", "content": "no" })
            )
            .is_err()
        );
        assert!(!root.parent().unwrap().join("escaped.rs").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_under_indented_edit_relocates_and_keeps_the_files_indentation() {
        // The common real failure: the model retypes the body without its surrounding indentation.
        // Exact matching fails; the normalized region is unique, so the edit lands — and the
        // replacement is written at the file's indentation, not the model's.
        let root = scratch("reindent");
        std::fs::write(
            root.join("f.rs"),
            "impl T {\n    fn go(&self) {\n        do_it();\n    }\n}\n",
        )
        .unwrap();

        let ok = edit(
            &root,
            &json!({ "file_path": "f.rs", "old_string": "fn go(&self) {\n    do_it();\n}",
                     "new_string": "fn go(&self) {\n    do_it_twice();\n}" }),
        )
        .unwrap();
        assert!(ok.contains("matched ignoring indentation"), "{ok}");
        assert_eq!(
            std::fs::read_to_string(root.join("f.rs")).unwrap(),
            "impl T {\n    fn go(&self) {\n        do_it_twice();\n    }\n}\n",
            "the block keeps the file's indentation, not the argument's"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn relocation_refuses_an_ambiguous_region() {
        // Two blocks that normalize identically. Relocating to one of them would edit an arbitrary
        // choice, which is exactly what the exact-match rule exists to prevent.
        let root = scratch("ambiguous-relocate");
        std::fs::write(root.join("f.rs"), "  a();\n  b();\n\n\ta();\n\tb();\n").unwrap();
        let err = edit(
            &root,
            &json!({ "file_path": "f.rs", "old_string": "a();\nb();", "new_string": "c();" }),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("does not appear"), "{err}");
        // Untouched.
        assert_eq!(
            std::fs::read_to_string(root.join("f.rs")).unwrap(),
            "  a();\n  b();\n\n\ta();\n\tb();\n"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn normalization_ignores_indentation_and_blanks_but_nothing_else() {
        assert_eq!(normalized("  a\n\n\tb  \n"), ["a", "b"]);
        // Token order and content still have to match exactly — this is not fuzzy matching.
        assert_ne!(normalized("a b"), normalized("b a"));
        assert_ne!(normalized("a();"), normalized("a ();"));
    }

    #[test]
    fn grep_with_no_path_still_skips_node_modules() {
        // The existing behaviour to preserve: walking from the repo root, a SKIP directory is not
        // searched at all.
        let root = repo("grep-skips-nm");
        std::fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        std::fs::write(root.join("node_modules/pkg/index.js"), "fn main() {}\n").unwrap();
        let out = grep(&root, &json!({ "pattern": "fn main" })).unwrap();
        assert!(out.contains("src/main.rs"), "{out}");
        assert!(!out.contains("node_modules"), "{out}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn grep_rooted_in_a_skipped_dir_searches_nothing() {
        // The skip is escapable: the walk root always passes filter_entry at depth 0, so a `path`
        // pointing inside a SKIP directory searches it fully. Rooting a grep at node_modules must
        // still find nothing there.
        let root = repo("grep-in-nm");
        std::fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        std::fs::write(root.join("node_modules/evil.js"), "fn main() {}\n").unwrap();
        std::fs::write(root.join("node_modules/pkg/deep.js"), "fn main() {}\n").unwrap();
        let out = grep(
            &root,
            &json!({ "pattern": "fn main", "path": "node_modules" }),
        )
        .unwrap();
        assert_eq!(out, "no matches", "{out}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn glob_rooted_in_a_skipped_dir_matches_nothing() {
        // Same escape through Glob: `repo` already placed target/huge.rs, and rooting the glob at
        // the skipped `target` must not surface it.
        let root = repo("glob-in-target");
        let out = glob_files(&root, &json!({ "pattern": "**/*.rs", "path": "target" })).unwrap();
        assert_eq!(out, "no files matched", "{out}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
