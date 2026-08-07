//! The projects one dashboard watches.
//!
//! Every piece of ratatoskr state is per-repository — its own store, worktrees, logs, and rag-rat
//! index — so watching several is a matter of holding several read handles, not of merging
//! anything. Nothing is shared between projects except the port and the question desk (run ids are
//! unique, so questions need no further scoping).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ratatoskr_store::Store;
use serde::Serialize;

use crate::launch::Launcher;

/// What the caller must resolve before a project can be served: where it lives, which config its
/// runs should use, and where its store actually is.
///
/// The store path is resolved by the caller because a config's `store.path` is relative to *its*
/// project, and this process has one working directory for all of them.
pub struct ProjectSpec {
    pub dir: PathBuf,
    pub config_path: PathBuf,
    pub store_path: PathBuf,
}

/// One watched project, opened and ready to serve.
pub struct Project {
    pub slug: String,
    pub dir: PathBuf,
    /// Where this project's config lives, so per-node routes can be read when a client asks. Read
    /// per request rather than cached: a dashboard left open across a config edit should show the
    /// routes a run would use now, not the ones it started with.
    pub config_path: PathBuf,
    pub store: Store,
    pub log_dir: PathBuf,
    pub launcher: Arc<Launcher>,
}

/// How a project appears in the API.
#[derive(Debug, Serialize)]
pub struct ProjectView {
    pub slug: String,
    pub dir: String,
}

/// Errors opening the set of projects.
#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error(
        "no checkpoint store at {0} — run `ratatoskr plan` or `ratatoskr run` in that project \
         first, or point --project at the right directory"
    )]
    NoStore(PathBuf),
    #[error(
        "two projects would both be called `{0}` ({1} and {2}); rename one directory or serve \
         them separately"
    )]
    DuplicateSlug(String, PathBuf, PathBuf),
    #[error(
        "`{0}` and `{1}` both read the store at {2}; they would show the same runs under two \
         names, which is not two projects"
    )]
    SharedStore(String, String, PathBuf),
    #[error("no projects to serve")]
    Empty,
    #[error("store error: {0}")]
    Store(#[from] ratatoskr_store::StoreError),
}

/// How a project is named in URLs and in the UI: its directory name, which is what a person
/// actually calls the repository.
fn slug_for(dir: &Path) -> String {
    let raw = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("project");
    let slug: String = raw
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let trimmed = slug.trim_matches('-').to_ascii_lowercase();
    if trimmed.is_empty() {
        "project".to_string()
    } else {
        trimmed
    }
}

/// Open every project, keyed by slug.
///
/// A missing store is an error rather than an empty dashboard: `Store::open` would happily create
/// one, and a typo'd path would then look like "no runs yet" instead of a mistake.
pub fn open_all(
    specs: Vec<ProjectSpec>,
    max_runs: usize,
    dashboard_url: &str,
) -> Result<BTreeMap<String, Project>, ProjectError> {
    if specs.is_empty() {
        return Err(ProjectError::Empty);
    }
    let mut projects: BTreeMap<String, Project> = BTreeMap::new();
    let mut stores: BTreeMap<PathBuf, String> = BTreeMap::new();

    for spec in specs {
        if !spec.store_path.exists() {
            return Err(ProjectError::NoStore(spec.store_path));
        }
        let slug = slug_for(&spec.dir);
        if let Some(existing) = projects.get(&slug) {
            return Err(ProjectError::DuplicateSlug(
                slug,
                existing.dir.clone(),
                spec.dir,
            ));
        }
        // Distinct directories can still point at one store — an absolute `store.path`, a symlink,
        // a copied config. That is the same project twice wearing two names, and the isolation
        // everything here assumes would be a fiction.
        let store_key = spec
            .store_path
            .canonicalize()
            .unwrap_or_else(|_| spec.store_path.clone());
        if let Some(owner) = stores.get(&store_key) {
            return Err(ProjectError::SharedStore(owner.clone(), slug, store_key));
        }
        stores.insert(store_key, slug.clone());

        let launcher = Arc::new(Launcher::new(
            &spec.dir,
            &spec.config_path,
            max_runs,
            dashboard_url,
            &slug,
        ));
        projects.insert(
            slug.clone(),
            Project {
                slug,
                log_dir: spec.dir.join(".ratatoskr/logs"),
                config_path: spec.config_path.clone(),
                dir: spec.dir,
                store: Store::open(&spec.store_path)?,
                launcher,
            },
        );
    }
    Ok(projects)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slug_is_the_directory_name_made_url_safe() {
        assert_eq!(slug_for(Path::new("/home/kk/src/ratatoskr")), "ratatoskr");
        assert_eq!(slug_for(Path::new("/srv/My Project")), "my-project");
        assert_eq!(slug_for(Path::new("/srv/rag-rat")), "rag-rat");
        // Nothing usable to name it after.
        assert_eq!(slug_for(Path::new("/")), "project");
        assert_eq!(slug_for(Path::new("/srv/___")), "project");
    }

    #[test]
    fn projects_that_would_collide_are_refused_up_front() {
        // Two checkouts of the same repo would otherwise silently shadow each other in the URLs.
        let dir = std::env::temp_dir().join(format!("ratatoskr-proj-{}", std::process::id()));
        let (a, b) = (dir.join("a/thing"), dir.join("b/thing"));
        for p in [&a, &b] {
            std::fs::create_dir_all(p).unwrap();
            std::fs::write(p.join("state.sqlite3"), "").unwrap();
        }
        let spec = |d: &Path| ProjectSpec {
            dir: d.to_path_buf(),
            config_path: d.join("ratatoskr.toml"),
            store_path: d.join("state.sqlite3"),
        };

        let opened = open_all(vec![spec(&a), spec(&b)], 1, "http://127.0.0.1:1");
        assert!(
            matches!(&opened, Err(ProjectError::DuplicateSlug(s, ..)) if s == "thing"),
            "colliding names are refused up front"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_projects_pointed_at_one_store_are_refused() {
        // Distinct names, distinct directories, same database: the same runs would appear twice
        // under two identities.
        let dir = std::env::temp_dir().join(format!("ratatoskr-shared-{}", std::process::id()));
        let (a, b) = (dir.join("alpha"), dir.join("beta"));
        for p in [&a, &b] {
            std::fs::create_dir_all(p).unwrap();
        }
        let shared = dir.join("state.sqlite3");
        std::fs::write(&shared, "").unwrap();
        let spec = |d: &Path| ProjectSpec {
            dir: d.to_path_buf(),
            config_path: d.join("ratatoskr.toml"),
            store_path: shared.clone(),
        };

        let opened = open_all(vec![spec(&a), spec(&b)], 1, "http://127.0.0.1:1");
        assert!(
            matches!(&opened, Err(ProjectError::SharedStore(first, second, _))
                if first == "alpha" && second == "beta"),
            "both names are reported so the operator knows which two collided"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_store_names_the_path_rather_than_serving_nothing() {
        let missing = PathBuf::from("/nonexistent/state.sqlite3");
        let opened = open_all(
            vec![ProjectSpec {
                dir: PathBuf::from("/nonexistent"),
                config_path: PathBuf::from("/nonexistent/ratatoskr.toml"),
                store_path: missing.clone(),
            }],
            1,
            "http://127.0.0.1:1",
        );
        assert!(matches!(&opened, Err(ProjectError::NoStore(p)) if *p == missing));

        assert!(matches!(
            open_all(vec![], 1, "http://127.0.0.1:1"),
            Err(ProjectError::Empty)
        ));
    }
}
