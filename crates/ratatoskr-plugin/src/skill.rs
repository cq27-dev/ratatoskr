//! Skills a plugin ships: a `SKILL.md` whose frontmatter says when it applies and whose body says
//! what to do once it does.
//!
//! The point of a skill over a longer preamble is that the body stays out of context until the
//! model decides it is wanted — so only `name` and `description` are read eagerly. The rest of the
//! frontmatter describes capabilities of a coding CLI (`allowed-tools`, `model`, `hooks`, `shell`)
//! that a node has no way to honour, and is ignored rather than half-applied.

use std::path::{Path, PathBuf};

/// Largest `SKILL.md` that will be read. Matches the limit the format's own host applies, and
/// bounds what one tool result can put into a node's conversation.
const MAX_SKILL_BYTES: u64 = 128 * 1024;

/// One skill, ready to offer to a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    /// The skill's own name — its frontmatter `name`, else its directory name.
    pub name: String,
    /// When to use it. This is the only part a node carries before choosing the skill, so a skill
    /// without one can never be chosen deliberately.
    pub description: String,
    /// The instructions, loaded only once a node asks for them.
    pub body: String,
    /// The skill's own directory, which its body refers to as `${CLAUDE_SKILL_DIR}`.
    pub dir: PathBuf,
}

/// Read the skills a plugin ships, in name order.
///
/// Two layouts, both conventional: a `skills/` directory of skill directories, and a bare
/// `SKILL.md` at the plugin root for a plugin that is one skill. Paths named by the manifest's
/// `skills` key are not read yet.
pub fn read_skills(root: &Path) -> Vec<Skill> {
    let mut found: Vec<Skill> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(root.join("skills")) {
        for entry in entries.flatten() {
            if let Some(skill) = read_skill(&entry.path()) {
                found.push(skill);
            }
        }
    }
    // A plugin that is a single skill. Only when it has no `skills/` directory, so a plugin with
    // both does not offer the same skill twice under two names.
    if found.is_empty()
        && let Some(skill) = read_skill(root)
    {
        found.push(skill);
    }

    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

/// Read one skill directory, or `None` if it does not hold a usable `SKILL.md`.
fn read_skill(dir: &Path) -> Option<Skill> {
    let path = dir.join("SKILL.md");
    // Checked before reading: the point of the limit is not to load the file.
    match std::fs::metadata(&path) {
        Ok(meta) if meta.len() > MAX_SKILL_BYTES => {
            tracing::warn!(
                skill = %path.display(),
                bytes = meta.len(),
                "ignoring skill: over {MAX_SKILL_BYTES} bytes"
            );
            return None;
        }
        Ok(_) => {}
        Err(_) => return None,
    }
    let raw = std::fs::read_to_string(&path).ok()?;
    let (front, body) = split_frontmatter(&raw);

    // The directory name is the skill's identity in the format; frontmatter `name` overrides it.
    let fallback = dir.file_name().and_then(|n| n.to_str()).unwrap_or("skill");
    let name = field(front, "name").unwrap_or_else(|| fallback.to_string());
    // A skill nothing can describe cannot be chosen on purpose, so the first paragraph stands in.
    let description = field(front, "description").unwrap_or_else(|| first_paragraph(body));

    if description.is_empty() {
        tracing::warn!(
            skill = name,
            "ignoring skill: it says nothing about when it applies"
        );
        return None;
    }

    Some(Skill {
        name,
        description,
        body: body.trim().to_string(),
        dir: dir.to_path_buf(),
    })
}

/// Split a `---`-delimited frontmatter block from the body. No frontmatter is ordinary: the whole
/// file is then the body, and the skill is named and described from what is there.
fn split_frontmatter(raw: &str) -> (&str, &str) {
    let text = raw.trim_start_matches('\u{feff}');
    let mut lines = text.split_inclusive('\n');

    // Both delimiters are `---` alone on a line, judged the same way. Holding the opening one to a
    // stricter rule than the closing one is how a file with a trailing space after the first `---`
    // becomes all body — and then its frontmatter is read as prose and described from.
    let opening = lines.next().unwrap_or_default();
    if opening.trim_end() != "---" {
        return ("", text);
    }
    let rest = &text[opening.len()..];

    let mut end = 0usize;
    for line in rest.split_inclusive('\n') {
        if line.trim_end() == "---" {
            return (&rest[..end], &rest[end + line.len()..]);
        }
        end += line.len();
    }
    // Opened and never closed: not frontmatter, so the file is all body.
    ("", text)
}

/// One scalar field out of a frontmatter block.
///
/// Deliberately not a YAML parser: the schema here is a flat map of strings, and the shapes that
/// actually occur are a plain scalar, a quoted one, and a folded or literal block. A value of any
/// other shape (a list, a nested map) is skipped rather than guessed at, because the only fields
/// read here are strings.
fn field(front: &str, key: &str) -> Option<String> {
    let mut lines = front.lines();
    while let Some(line) = lines.next() {
        let Some(rest) = line.strip_prefix(key) else {
            continue;
        };
        // Requiring the colon immediately after is what keeps `name` from matching `names:`.
        let Some(value) = rest.strip_prefix(':') else {
            continue;
        };
        let value = value.trim();
        return match value.chars().next() {
            // A folded or literal block: the value is the indented lines that follow.
            Some('>') => Some(block(&mut lines, " ")),
            Some('|') => Some(block(&mut lines, "\n")),
            // A list or nested map — not a string, so not ours to read.
            Some('[') | Some('{') | None => None,
            _ => Some(unquote(value)),
        };
    }
    None
}

/// The indented lines following a block scalar, joined by `join` — a space for folded (`>`), a
/// newline for literal (`|`).
fn block<'a>(lines: &mut impl Iterator<Item = &'a str>, join: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for line in lines.by_ref() {
        // The block ends at the first line that is not indented. A blank line is part of it.
        if !line.trim().is_empty() && !line.starts_with(char::is_whitespace) {
            break;
        }
        parts.push(line.trim());
    }
    parts.join(join).trim().to_string()
}

/// Strip one layer of matching quotes, or a trailing comment from a plain scalar.
fn unquote(value: &str) -> String {
    for quote in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|v| v.strip_suffix(quote))
        {
            return inner.to_string();
        }
    }
    // YAML's rule: ` #` opens a comment in a plain scalar, and means nothing inside a quoted one.
    // Worth honouring, because otherwise a description ends with the author's note to themselves,
    // shown to the model as part of the instruction.
    match value.find(" #") {
        Some(at) => value[..at].trim_end().to_string(),
        None => value.to_string(),
    }
}

/// The body's first prose paragraph, for a skill whose frontmatter describes nothing.
fn first_paragraph(body: &str) -> String {
    body.split("\n\n")
        .map(str::trim)
        .find(|p| !p.is_empty() && !p.starts_with('#'))
        .unwrap_or_default()
        .replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill_dir(case: &str, contents: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("ratatoskr-skill-{}-{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("SKILL.md"), contents).unwrap();
        root
    }

    #[test]
    fn a_folded_description_is_unfolded_into_one_line() {
        // The shape rag-rat's own skills use, and the one a naive `key: value` split gets wrong.
        let root = skill_dir(
            "folded",
            "---\nname: dream-review\ndescription: >\n  Use when asked to review rag-rat\n  \
             \"dream\" findings — the memory-maintenance worklist.\n---\n\n# dream-review\n\nBody.",
        );
        let skill = read_skill(&root).expect("a skill");
        assert_eq!(skill.name, "dream-review");
        assert_eq!(
            skill.description,
            "Use when asked to review rag-rat \"dream\" findings — the memory-maintenance worklist."
        );
        assert_eq!(skill.body, "# dream-review\n\nBody.");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_other_scalar_shapes_that_occur_in_the_wild() {
        // A literal block keeps its line breaks; a quoted scalar loses its quotes.
        let root = skill_dir(
            "literal",
            "---\nname: \"quoted\"\ndescription: |\n  one\n  two\n---\nbody",
        );
        let skill = read_skill(&root).unwrap();
        assert_eq!(skill.name, "quoted");
        assert_eq!(skill.description, "one\ntwo");
        let _ = std::fs::remove_dir_all(&root);

        // A plain scalar, and a name that falls back to the directory.
        let root = skill_dir("plain", "---\ndescription: does a thing\n---\nbody");
        let skill = read_skill(&root).unwrap();
        assert!(skill.name.ends_with("-plain"), "named after its directory");
        assert_eq!(skill.description, "does a thing");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_key_is_matched_whole_and_a_non_string_value_is_left_alone() {
        // `name` must not match `names:`, and a list is not a description.
        let front = "names: not-this\nname: this\nallowed-tools: [Read, Grep]\n";
        assert_eq!(field(front, "name").as_deref(), Some("this"));
        assert_eq!(field(front, "allowed-tools"), None);
        assert_eq!(field(front, "missing"), None);
    }

    #[test]
    fn a_file_without_usable_frontmatter_still_yields_a_skill_or_none() {
        // No frontmatter: the first prose paragraph describes it.
        let root = skill_dir("bare", "# Title\n\nWhat this does.\n\nMore.");
        let skill = read_skill(&root).expect("described by its body");
        assert_eq!(skill.description, "What this does.");
        let _ = std::fs::remove_dir_all(&root);

        // Nothing to describe it at all: not offerable, so not offered.
        let root = skill_dir("silent", "# Title\n");
        assert!(read_skill(&root).is_none());
        let _ = std::fs::remove_dir_all(&root);

        // An unterminated block is not frontmatter; the text is the body.
        let root = skill_dir("unclosed", "---\nname: x\ndescription: y\n\nstill going");
        let skill = read_skill(&root).unwrap();
        assert!(skill.name.ends_with("-unclosed"));
        assert!(skill.body.starts_with("---"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_two_delimiters_are_judged_the_same_way() {
        // A trailing space after the opening `---` used to make the whole file body, and the
        // frontmatter was then read as prose — a wrong description rather than none.
        let root = skill_dir("loose", "--- \nname: x\ndescription: real\n---\nbody");
        let skill = read_skill(&root).unwrap();
        assert_eq!(skill.name, "x");
        assert_eq!(skill.description, "real");
        assert_eq!(skill.body, "body");
        let _ = std::fs::remove_dir_all(&root);

        // CRLF throughout, which is the same rule seen from the other side.
        let root = skill_dir("crlf", "---\r\nname: y\r\ndescription: real\r\n---\r\nbody");
        let skill = read_skill(&root).unwrap();
        assert_eq!(skill.name, "y");
        assert_eq!(skill.description, "real");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_trailing_comment_is_not_part_of_the_description() {
        // YAML's rule, and worth honouring: the alternative shows the model the author's aside.
        assert_eq!(unquote("does a thing # revisit later"), "does a thing");
        // Inside quotes it is text, not a comment.
        assert_eq!(unquote("\"issue #123\""), "issue #123");
        assert_eq!(unquote("no comment here"), "no comment here");
    }

    #[test]
    fn a_skills_directory_wins_over_a_root_skill_file() {
        let root = std::env::temp_dir().join(format!("ratatoskr-skills-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for name in ["beta", "alpha"] {
            std::fs::create_dir_all(root.join("skills").join(name)).unwrap();
            std::fs::write(
                root.join("skills").join(name).join("SKILL.md"),
                format!("---\ndescription: the {name} one\n---\nbody"),
            )
            .unwrap();
        }
        // Also a root SKILL.md, which must not be read as a third skill.
        std::fs::write(root.join("SKILL.md"), "---\ndescription: root\n---\nbody").unwrap();

        let skills = read_skills(&root);
        assert_eq!(
            skills.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            ["alpha", "beta"],
            "named after their directories, in name order"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
