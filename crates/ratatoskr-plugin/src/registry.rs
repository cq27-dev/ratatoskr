//! Which copy of a plugin is the installed one.
//!
//! A coding CLI copies a marketplace plugin into a cache laid out
//! `<marketplace>/<plugin>/<version>/`, and keeps older versions around. A path naming the plugin
//! — the obvious thing to configure — therefore holds every version ever installed, and walking it
//! finds them all.
//!
//! The host records which one is current in `installed_plugins.json`. Nothing documents that file,
//! so it is read as a hint and never depended on: when it is absent, unreadable, or says nothing
//! about a plugin, discovery falls back to refusing the same name twice.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Where a coding CLI keeps its plugin state, relative to the user's home.
const REGISTRY: &str = ".claude/plugins/installed_plugins.json";

#[derive(Debug, Deserialize)]
struct Registry {
    #[serde(default)]
    plugins: BTreeMap<String, Vec<Installed>>,
}

#[derive(Debug, Deserialize)]
struct Installed {
    #[serde(rename = "installPath")]
    install_path: PathBuf,
}

/// The directory each installed plugin currently lives in, keyed by plugin name.
///
/// The registry keys plugins `<plugin>@<marketplace>`; the part before the `@` is the name a
/// manifest carries and the name a ruleset binds. One plugin can be installed at more than one
/// scope, and every entry is kept — discovery only asks whether a directory it found is *an*
/// installed one, not which scope it came from.
pub fn installed(home: &Path) -> BTreeMap<String, Vec<PathBuf>> {
    let Ok(raw) = std::fs::read_to_string(home.join(REGISTRY)) else {
        return BTreeMap::new();
    };
    let registry: Registry = match serde_json::from_str(&raw) {
        Ok(registry) => registry,
        Err(e) => {
            tracing::debug!("ignoring the plugin registry: {e}");
            return BTreeMap::new();
        }
    };

    let mut found: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for (key, entries) in registry.plugins {
        let name = key.split('@').next().unwrap_or(&key).to_string();
        found
            .entry(name)
            .or_default()
            .extend(entries.into_iter().map(|e| e.install_path));
    }
    found
}

/// Whether `root` is the copy of `name` that is currently installed.
///
/// `None` means the registry says nothing about this plugin — every copy is then as good as any
/// other, and discovery falls back to keeping the first it found.
pub fn is_current(
    installed: &BTreeMap<String, Vec<PathBuf>>,
    name: &str,
    root: &Path,
) -> Option<bool> {
    let paths = installed.get(name)?;
    Some(paths.iter().any(|p| same_dir(p, root)))
}

/// Two paths naming the same directory, whether or not either is canonical.
fn same_dir(a: &Path, b: &Path) -> bool {
    a == b || matches!((a.canonicalize(), b.canonicalize()), (Ok(a), Ok(b)) if a == b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home(case: &str, contents: Option<&str>) -> PathBuf {
        let home =
            std::env::temp_dir().join(format!("ratatoskr-registry-{}-{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(home.join(".claude/plugins")).unwrap();
        if let Some(contents) = contents {
            std::fs::write(home.join(REGISTRY), contents).unwrap();
        }
        home
    }

    #[test]
    fn the_registry_names_the_installed_copy_of_each_plugin() {
        let dir = home(
            "read",
            Some(
                r#"{ "version": 2, "plugins": {
                    "rag-rat@rag-rat": [
                        { "installPath": "/cache/rag-rat/rag-rat/0.22.0", "version": "0.22.0" }
                    ],
                    "superpowers@claude-plugins-official": [
                        { "installPath": "/cache/official/superpowers/6.2.0" }
                    ]
                } }"#,
            ),
        );
        let installed = installed(&dir);

        // Keyed by the plugin's own name, not `<plugin>@<marketplace>`.
        assert_eq!(
            installed.get("rag-rat").map(Vec::as_slice),
            Some(&[PathBuf::from("/cache/rag-rat/rag-rat/0.22.0")][..])
        );
        assert_eq!(
            is_current(
                &installed,
                "rag-rat",
                Path::new("/cache/rag-rat/rag-rat/0.22.0")
            ),
            Some(true)
        );
        assert_eq!(
            is_current(
                &installed,
                "rag-rat",
                Path::new("/cache/rag-rat/rag-rat/0.21.0")
            ),
            Some(false),
            "an older version is present on disk but is not the installed one"
        );
        // A plugin the registry says nothing about is nobody's business to judge.
        assert_eq!(is_current(&installed, "local-thing", Path::new("/x")), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_registry_and_a_broken_one_are_both_ordinary() {
        // It is undocumented, so it is a hint. Neither absence nor rubbish may change a run.
        assert!(installed(&home("missing", None)).is_empty());
        assert!(installed(&home("broken", Some("{ not json"))).is_empty());
        assert!(installed(Path::new("/nonexistent-home")).is_empty());
    }
}
