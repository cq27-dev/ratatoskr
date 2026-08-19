//! The projects one dashboard watches.
//!
//! Every piece of ratatoskr state is per-repository — its own store, worktrees, logs, and rag-rat
//! index — so watching several is a matter of holding several read handles, not of merging
//! anything. Nothing is shared between projects except the port and the question desk (run ids are
//! unique, so questions need no further scoping).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ratatoskr_core::auth::Visibility;
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
    /// Whether anyone may read this project without logging in.
    ///
    /// Resolved by the caller, from the instance's own configuration, and deliberately never from
    /// the project's `ratatoskr.toml`: a repository must not be able to declare itself public.
    /// That is the host operator's decision, so it is made where the projects are listed.
    pub visibility: Visibility,
}

/// One watched project, opened and ready to serve.
pub struct Project {
    pub slug: String,
    pub dir: PathBuf,
    /// The GitHub repository this checkout pushes to, as `owner/name`, if it has one.
    ///
    /// Read from `origin` rather than configured, because a mapping written by hand is one that
    /// can be wrong — and pointing an integration at the wrong checkout means a run against the
    /// wrong repository. `None` for a project with no origin, no GitHub origin, or no git at all,
    /// which simply means the integration cannot address it.
    pub repository: Option<String>,
    /// See [`ProjectSpec::visibility`].
    pub visibility: Visibility,
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
    /// Where it lives on the host — absent for a caller who is not logged in.
    ///
    /// An absolute filesystem path says more about the machine than a stranger reading a public
    /// run needs to know: the operating system, the layout, often a username. It is useful to
    /// whoever runs the instance, so it is shown to them and to nobody else.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dir: Option<String>,
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
    #[error(
        "`{0}` is not a usable project name: the dashboard puts the project in the URL path, and \
         /{0} is already the server's own. Rename the directory or serve it from elsewhere"
    )]
    ReservedSlug(String),
    #[error("no projects to serve")]
    Empty,
    #[error("store error: {0}")]
    Store(#[from] ratatoskr_store::StoreError),
}

/// How a project is named in URLs and in the UI: its directory name, which is what a person
/// actually calls the repository.
///
/// Public because it is the project's identity, and every caller that names a project has to agree
/// with it. An operator spelling a project the way the dashboard shows it — `my-repo`, not
/// `My_Repo` — is naming the slug, so the flag that reads that name resolves it through here.
pub fn slug_for(dir: &Path) -> String {
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

/// The `owner/name` a checkout's `origin` points at on GitHub, if it does.
///
/// Shelled out rather than read with a git library: it runs once per project at startup, and `git`
/// is already a hard requirement of every run. Both URL forms are handled — `git@host:owner/name`
/// and `https://host/owner/name` — and anything that is not GitHub yields `None` rather than a
/// guess, because the only use of this is deciding which checkout a GitHub webhook is about.
fn github_repository(dir: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["config", "--get", "remote.origin.url"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_github_remote(String::from_utf8_lossy(&out.stdout).trim())
}

/// `owner/name` out of a remote URL, or `None` if it does not name a GitHub repository.
fn parse_github_remote(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("git@github.com:")
        .or_else(|| url.strip_prefix("ssh://git@github.com/"))
        .or_else(|| url.strip_prefix("https://github.com/"))
        .or_else(|| url.strip_prefix("http://github.com/"))?;
    let rest = rest.strip_suffix(".git").unwrap_or(rest);
    let rest = rest.trim_end_matches('/');
    // Exactly two segments. A URL with more is not a repository root, and one with fewer names no
    // repository at all.
    let mut parts = rest.split('/');
    let (owner, name) = (parts.next()?, parts.next()?);
    if parts.next().is_some() || owner.is_empty() || name.is_empty() {
        return None;
    }
    Some(format!("{owner}/{name}"))
}

/// Path segments the server answers itself, so no project may be named one.
///
/// Kept in step with `router` and with `RESERVED` in the dashboard's url.ts. Short enough to be
/// obvious, and a mismatch shows up as the ordinary duplicate-name error rather than silently.
const RESERVED_SLUGS: &[&str] = &["api", "assets", "internal"];

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
        // The dashboard addresses a project by the first path segment, and these are matched
        // ahead of the fallback that serves it — so a project called `api` would be a page nobody
        // could open. Caught here, where the name is chosen, rather than as a mystery 404 later.
        if RESERVED_SLUGS.contains(&slug.as_str()) {
            return Err(ProjectError::ReservedSlug(slug));
        }
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
                repository: github_repository(&spec.dir),
                slug,
                log_dir: spec.dir.join(".ratatoskr/logs"),
                config_path: spec.config_path.clone(),
                dir: spec.dir,
                visibility: spec.visibility,
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
            visibility: Visibility::default(),
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
            visibility: Visibility::default(),
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
    fn a_github_remote_is_read_in_either_form() {
        for url in [
            "git@github.com:cq27-dev/ratatoskr.git",
            "git@github.com:cq27-dev/ratatoskr",
            "https://github.com/cq27-dev/ratatoskr.git",
            "https://github.com/cq27-dev/ratatoskr/",
            "ssh://git@github.com/cq27-dev/ratatoskr.git",
        ] {
            assert_eq!(
                parse_github_remote(url).as_deref(),
                Some("cq27-dev/ratatoskr"),
                "{url}"
            );
        }
    }

    #[test]
    fn a_remote_that_is_not_a_github_repository_is_none_rather_than_a_guess() {
        // Deciding which checkout a webhook is about off a wrong guess means running against the
        // wrong repository, so anything unrecognised has to be no answer at all.
        for url in [
            "git@gitlab.com:cq27-dev/ratatoskr.git",
            "https://example.com/cq27-dev/ratatoskr",
            "https://github.com/cq27-dev",
            "https://github.com/cq27-dev/ratatoskr/tree/main",
            "https://github.com//ratatoskr",
            "",
            "not a url",
        ] {
            assert_eq!(parse_github_remote(url), None, "{url}");
        }
    }

    #[test]
    fn a_project_named_after_a_server_route_is_refused() {
        // `/api/...` is matched before the fallback that serves the dashboard, so a project called
        // `api` would be addressable by the API and by nothing else.
        let dir = std::env::temp_dir().join(format!("ratatoskr-reserved-{}", std::process::id()));
        let api = dir.join("api");
        std::fs::create_dir_all(&api).unwrap();
        std::fs::write(api.join("state.sqlite3"), "").unwrap();

        let opened = open_all(
            vec![ProjectSpec {
                dir: api.clone(),
                config_path: api.join("ratatoskr.toml"),
                store_path: api.join("state.sqlite3"),
                visibility: Visibility::default(),
            }],
            1,
            "http://127.0.0.1:1",
        );
        assert!(
            matches!(&opened, Err(ProjectError::ReservedSlug(s)) if s == "api"),
            "a reserved name has to be refused where it is chosen, not 404 later"
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
                visibility: Visibility::default(),
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
