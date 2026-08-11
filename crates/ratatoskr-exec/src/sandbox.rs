//! Sandboxed command execution behind a backend-agnostic interface.
//!
//! `run(SandboxSpec)` is Ratatoskr's own contract — no backend's types leak into it, so swapping
//! one for another is a genuine swap rather than a rewrite.
//!
//! Three backends implement it, and they are three rungs of the same ladder rather than three
//! alternatives of equal standing:
//!
//! - `landlock` — bubblewrap over the **host root**, read-only, with the worktree bind-mounted
//!   writable. Needs no image and no daemon, so it works anywhere `bwrap` does. Weakest: the host
//!   filesystem is readable, which is why `~/.ssh` and `~/.npmrc` are in scope and why the
//!   environment has to be cleared by hand.
//! - `container` — an **OCI container**, so the toolchain comes from an image and the mounts are
//!   the only host filesystem there is. Needs an ordinary container runtime and nothing else. This
//!   is the rung that removes the host root without requiring KVM; its toolchain and dependencies
//!   come from the image and prepared read-only caches, never an acceptance-time install.
//! - `microsandbox` — a **MicroVM**, adding a VM boundary on top of the image. Needs KVM, and is
//!   behind `--features microsandbox` because its build script needs the network.

use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use tokio::process::Command;

/// A host→guest bind mount (the worktree, or the baseline checkout for red-team).
#[derive(Debug, Clone)]
pub struct Mount {
    pub host: PathBuf,
    pub guest: String,
    /// Whether the command may write here.
    ///
    /// Stated per mount rather than assumed, because a mount is the *only* writable host
    /// filesystem a sandbox has, and giving one away by accident is how a run reaches something it
    /// was never meant to touch. A dependency cache is read-only; the worktree is not.
    pub read_only: bool,
}

/// What to run, where, and under what limits. Ratatoskr's own type — backend-neutral.
#[derive(Debug, Clone)]
pub struct SandboxSpec {
    /// `"container"`, `"landlock"` or `"microsandbox"` — see the module docs for what each buys.
    pub backend: String,
    /// Unique sandbox name. The container and microsandbox backends need one; the bwrap backend
    /// ignores it. A runtime rejects a name under two characters, so it is never a bare index.
    pub name: String,
    /// OCI image to run in. Used by the container and microsandbox backends; the bwrap backend has
    /// no image and runs the host's toolchain instead.
    pub image: String,
    /// Working directory inside the sandbox — usually a mount's guest path.
    pub workdir: String,
    pub mounts: Vec<Mount>,
    /// Program + args to run.
    pub command: Vec<String>,
    pub cpus: u8,
    pub memory_mib: u64,
    /// Whether the sandbox has network access (off by default for test runs).
    pub network: bool,
}

/// The result of a sandboxed run.
#[derive(Debug, Clone)]
pub struct ExecOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl ExecOutput {
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
}

/// Errors from a sandboxed run.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error(
        "unknown sandbox backend {0:?} (expected \"container\", \"landlock\" or \"microsandbox\")"
    )]
    UnknownBackend(String),
    #[error("empty sandbox command")]
    EmptyCommand,
    #[error("microsandbox error: {0}")]
    Microsandbox(String),
    #[error("bwrap not found or failed to launch: {0}")]
    Bwrap(#[from] std::io::Error),
    #[error("could not prepare linked-worktree Git metadata for the sandbox")]
    GitMetadata,
    #[error(
        "no container runtime found; the `container` backend needs one of {} on PATH",
        RUNTIMES.join(" or ")
    )]
    NoContainerRuntime,
    #[error("{runtime} failed to launch: {source}")]
    Container {
        runtime: &'static str,
        source: std::io::Error,
    },
    #[error("container image {0:?} is not an immutable sha256 identifier")]
    InvalidContainerImage(String),
    #[error("{runtime} could not inspect container image {image:?}: {detail}")]
    ContainerInspect {
        runtime: &'static str,
        image: String,
        detail: String,
    },
}

/// Run `spec.command` in a sandbox, returning its captured output. Dispatches on `spec.backend`.
pub async fn run(spec: SandboxSpec) -> Result<ExecOutput, SandboxError> {
    if spec.command.is_empty() {
        return Err(SandboxError::EmptyCommand);
    }
    match spec.backend.as_str() {
        #[cfg(feature = "microsandbox")]
        "microsandbox" => run_microsandbox(spec).await,
        #[cfg(not(feature = "microsandbox"))]
        "microsandbox" => Err(SandboxError::Microsandbox(
            "microsandbox backend not compiled in; rebuild ratatoskr-exec with \
             --features microsandbox"
                .to_string(),
        )),
        "container" | "docker" | "podman" => run_container(spec).await,
        "landlock" | "bwrap" => run_bwrap(spec).await,
        other => Err(SandboxError::UnknownBackend(other.to_string())),
    }
}

/// The container runtimes this knows how to drive, in preference order. Both take the subset of
/// arguments used here identically, so the choice is availability rather than configuration.
const RUNTIMES: &[&str] = &["docker", "podman"];
static BWRAP_LIMITS_WARNED: AtomicBool = AtomicBool::new(false);

/// The first runtime on `PATH`, or nothing.
fn container_runtime() -> Option<&'static str> {
    RUNTIMES.iter().copied().find(|runtime| {
        std::env::var_os("PATH")
            .map(|path| std::env::split_paths(&path).any(|dir| dir.join(runtime).is_file()))
            .unwrap_or(false)
    })
}

fn is_image_digest(image: &str) -> bool {
    image.strip_prefix("sha256:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    })
}

/// Resolve an OCI image selector to the immutable local image ID the chosen runtime executes.
///
/// Tags are deliberately resolved here, at execution time, rather than when TOML is loaded: a
/// local image may be built after configuration is read. The returned `sha256:` ID is suitable for
/// the same Docker/Podman `run` argv as the configured selector.
pub async fn resolve_container_image(image: &str) -> Result<String, SandboxError> {
    if image.trim().is_empty() || image.starts_with("sha256:") && !is_image_digest(image) {
        return Err(SandboxError::InvalidContainerImage(image.to_string()));
    }

    let runtime = container_runtime().ok_or(SandboxError::NoContainerRuntime)?;
    if is_image_digest(image) {
        return Ok(image.to_string());
    }
    let output = Command::new(runtime)
        .args(["image", "inspect", "--format", "{{.Id}}", image])
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|source| SandboxError::Container { runtime, source })?;
    let identity = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !output.status.success() || !is_image_digest(&identity) {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(SandboxError::ContainerInspect {
            runtime,
            image: image.to_string(),
            detail: if detail.is_empty() {
                format!("returned {identity:?}")
            } else {
                detail
            },
        });
    }
    Ok(identity)
}

/// The uid:gid a container should run its command as.
///
/// Not a detail. A rootful runtime runs as root by default, so everything written into the mounted
/// worktree comes out owned by root — the host then cannot read the diff it is supposed to review,
/// and `git worktree remove` cannot delete the tree. Read from `/proc/self`, which is owned by this
/// process's real user, so it needs no libc and no assumption about the environment.
fn host_user() -> Option<String> {
    use std::os::unix::fs::MetadataExt as _;
    let me = std::fs::metadata("/proc/self").ok()?;
    Some(format!("{}:{}", me.uid(), me.gid()))
}

/// Exactly what the runtime is exec'd with. Split out so the argument list is a thing a test can
/// assert on, as with [`bwrap_argv`].
///
/// What is *absent* is the point of this backend. There is no host root, so `$HOME` — `~/.ssh`,
/// `~/.aws`, `~/.npmrc` — is not readable; the mounts are the only host filesystem in scope. And
/// nothing carries this process's environment across: a container starts from the image's, which is
/// the same guarantee `--clearenv` buys the bwrap backend, here by construction.
///
/// A mount lands at its `guest` path here, unlike the bwrap backend, which can only mount in place
/// — the host root is `--ro-bind`ed there, so there is nowhere to create a fresh mount point. That
/// difference is why a prepared dependency cache is a container-backend feature: putting
/// `node_modules` where a resolver will find it means mounting one host path at a completely
/// different guest path, which is exactly what mounting in place cannot do. (Under bwrap the host
/// root is visible anyway, so the cache has nothing to add there.)
fn container_argv(spec: &SandboxSpec, user: Option<&str>) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "run".into(),
        // Nothing to reap afterwards: the container is the unit of isolation and it does not
        // outlive the call it was created for.
        "--rm".into(),
        "--name".into(),
        spec.name.clone(),
    ];
    if let Some(user) = user {
        args.push("--user".into());
        args.push(user.to_string());
    }
    if !spec.network {
        args.push("--network".into());
        args.push("none".into());
    }
    args.push("--cpus".into());
    args.push(spec.cpus.to_string());
    args.push("--memory".into());
    args.push(format!("{}m", spec.memory_mib));

    // Each mount at the guest path it asked for, so `workdir` and the paths in a command's output
    // are the guest's own and need no translation.
    for m in &spec.mounts {
        let host = m.host.to_string_lossy().into_owned();
        args.push("--volume".into());
        let mode = if m.read_only { ":ro" } else { "" };
        args.push(format!("{host}:{}{mode}", m.guest));
    }
    args.push("--workdir".into());
    args.push(spec.workdir.clone());
    args.push(spec.image.clone());
    args.extend(spec.command.iter().cloned());
    args
}

/// OCI container backend: the toolchain comes from an image, the worktree is the only host
/// filesystem in scope, and the network is off unless the step asked for it.
///
/// Between the bwrap backend (host root, no image, no special support) and microsandbox (a MicroVM,
/// needs KVM), this is the rung that removes the host filesystem without requiring anything beyond
/// an ordinary container runtime.
async fn run_container(spec: SandboxSpec) -> Result<ExecOutput, SandboxError> {
    let (spec, _git_view) = with_linked_git_metadata(spec)?;
    let runtime = container_runtime().ok_or(SandboxError::NoContainerRuntime)?;
    let args = container_argv(&spec, host_user().as_deref());

    // `kill_on_drop` reaches the client, and `--rm` plus the client's own signal handling is what
    // stops the container: dropping the future kills the `run` process, and the runtime tears the
    // container down with it.
    let output = Command::new(runtime)
        .args(&args)
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|source| SandboxError::Container { runtime, source })?;
    let exit_code = output.status.code().unwrap_or(-1);
    let mut stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if let Some(hint) = oom_hint(exit_code, spec.memory_mib) {
        if !stderr.is_empty() {
            stderr.push('\n');
        }
        stderr.push_str(&hint);
    }
    Ok(ExecOutput {
        exit_code,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr,
    })
}

/// Explain the exit code OCI runtimes report for a likely memory-limit kill.
fn oom_hint(exit_code: i32, memory_mib: u64) -> Option<String> {
    (exit_code == 137).then(|| {
        format!(
            "process was likely killed because it exceeded the configured {memory_mib} MiB memory limit"
        )
    })
}

/// microsandbox (MicroVM) backend: boot the image with the mounts, run, tear down.
#[cfg(feature = "microsandbox")]
async fn run_microsandbox(spec: SandboxSpec) -> Result<ExecOutput, SandboxError> {
    use microsandbox::Sandbox;

    let mut builder = Sandbox::builder(spec.name.clone())
        .image(spec.image.clone())
        .cpus(spec.cpus)
        .memory(spec.memory_mib as u32)
        .workdir(spec.workdir.clone());
    for m in &spec.mounts {
        let host = m.host.clone();
        builder = builder.volume(m.guest.clone(), move |mb| mb.bind(host));
    }
    if !spec.network {
        builder = builder.disable_network();
    }

    let sandbox = builder
        .create()
        .await
        .map_err(|e| SandboxError::Microsandbox(e.to_string()))?;

    let (cmd, args) = spec.command.split_first().expect("non-empty checked above");
    let result = sandbox.exec(cmd.clone(), args.to_vec()).await;

    // Best-effort teardown so no MicroVM is orphaned, regardless of exec outcome.
    if let Err(e) = sandbox.stop().await {
        tracing::warn!("failed to stop microsandbox {}: {e}", spec.name);
    }

    let output = result.map_err(|e| SandboxError::Microsandbox(e.to_string()))?;
    Ok(ExecOutput {
        exit_code: output.status().code,
        stdout: output.stdout().unwrap_or_default(),
        stderr: output.stderr().unwrap_or_default(),
    })
}

/// A sandbox-local view of a linked worktree's Git metadata.
///
/// Git writes its per-worktree index below the common Git directory, while the worktree's `.git`
/// file contains a host-absolute pointer to that directory.  A container only sees `/workspace`,
/// so mounting the worktree alone makes ordinary `git status` fail and exposes the host path in
/// its error.  This view rewrites both indirections to stable guest paths and keeps the real
/// metadata mounted with the least authority Git needs.
struct LinkedGitView {
    temp_dir: PathBuf,
    mounts: Vec<Mount>,
}

impl Drop for LinkedGitView {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.temp_dir)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(path = %self.temp_dir.display(), "failed to remove sandbox Git view: {error}");
        }
    }
}

const GIT_VIEW_ROOT: &str = "/run/ratatoskr/git";
static GIT_VIEW_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Extend a container spec with a private guest-path representation of linked-worktree metadata.
/// Ordinary repositories use a `.git` directory and therefore need no transformation.
fn with_linked_git_metadata(
    mut spec: SandboxSpec,
) -> Result<(SandboxSpec, Option<LinkedGitView>), SandboxError> {
    let Some(worktree_mount) = spec
        .mounts
        .iter()
        .find(|mount| mount.guest == spec.workdir && !mount.read_only)
    else {
        return Ok((spec, None));
    };
    let Some(view) = linked_git_view(worktree_mount)? else {
        return Ok((spec, None));
    };
    spec.mounts.extend(view.mounts.iter().cloned());
    Ok((spec, Some(view)))
}

fn linked_git_view(worktree_mount: &Mount) -> Result<Option<LinkedGitView>, SandboxError> {
    let dot_git = worktree_mount.host.join(".git");
    if dot_git.is_dir() || !dot_git.exists() {
        return Ok(None);
    }

    let pointer = std::fs::read_to_string(&dot_git).map_err(|_| SandboxError::GitMetadata)?;
    let Some(git_dir) = pointer
        .strip_prefix("gitdir: ")
        .map(str::trim)
        .filter(|path| !path.is_empty())
    else {
        return Err(SandboxError::GitMetadata);
    };
    let git_dir = dot_git
        .parent()
        .expect("`.git` has a parent")
        .join(git_dir)
        .canonicalize()
        .map_err(|_| SandboxError::GitMetadata)?;
    let common_relative = std::fs::read_to_string(git_dir.join("commondir"))
        .map_err(|_| SandboxError::GitMetadata)?;
    let common_dir = git_dir
        .join(common_relative.trim())
        .canonicalize()
        .map_err(|_| SandboxError::GitMetadata)?;
    let worktree_name = git_dir
        .strip_prefix(common_dir.join("worktrees"))
        .ok()
        .filter(|relative| relative.components().count() == 1)
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && !name.contains('/'))
        .ok_or(SandboxError::GitMetadata)?;

    let temp_dir = create_git_view_dir()?;
    let guest_git_dir = format!("{GIT_VIEW_ROOT}/common/worktrees/{worktree_name}");
    let guest_dot_git = format!("{worktree}/.git", worktree = worktree_mount.guest);
    let guest_gitdir_file = format!("{guest_git_dir}/gitdir");
    if std::fs::write(
        temp_dir.join("dot-git"),
        format!("gitdir: {guest_git_dir}\n"),
    )
    .is_err()
        || std::fs::write(temp_dir.join("gitdir"), format!("{guest_dot_git}\n")).is_err()
    {
        std::fs::remove_dir_all(&temp_dir).ok();
        return Err(SandboxError::GitMetadata);
    }

    Ok(Some(LinkedGitView {
        mounts: vec![
            Mount {
                host: common_dir,
                guest: format!("{GIT_VIEW_ROOT}/common"),
                read_only: true,
            },
            // This is deliberately after the common directory mount: `git status` refreshes the
            // worktree index, but it must not be able to rewrite shared refs or configuration.
            Mount {
                host: git_dir,
                guest: guest_git_dir.clone(),
                read_only: false,
            },
            Mount {
                host: temp_dir.join("dot-git"),
                guest: guest_dot_git,
                read_only: true,
            },
            // Git's metadata records the worktree `.git` location too. Replace that second
            // host-path pointer so inspection never leaks the checkout's location.
            Mount {
                host: temp_dir.join("gitdir"),
                guest: guest_gitdir_file,
                read_only: true,
            },
        ],
        temp_dir,
    }))
}

fn create_git_view_dir() -> Result<PathBuf, SandboxError> {
    for _ in 0..32 {
        let sequence = GIT_VIEW_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ratatoskr-git-view-{}-{sequence}",
            std::process::id()
        ));
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(SandboxError::GitMetadata),
        }
    }
    Err(SandboxError::GitMetadata)
}

/// The only variables a sandboxed command sees. Everything else `--clearenv` drops.
///
/// A passlist, not a denylist. This process loads `.env` before anything else runs, so its
/// environment carries `ANTHROPIC_API_KEY`; it also carries whatever the developer's shell exports
/// — `SSH_AUTH_SOCK`, `DBUS_SESSION_BUS_ADDRESS` (the keyring `gh` keeps its token in), cloud
/// credentials. Removing those one name at a time guarantees that the next secret to arrive is
/// readable by default, which is the wrong way round.
///
/// Each entry is here because a command legitimately needs it:
const PASSED_THROUGH: &[&str] = &[
    // The toolchain reaches the sandbox through `PATH` and nowhere else. `node` and `npm` come
    // from nvm, `cargo` from rustup — per-user directories, not fixed system locations — so this
    // cannot be replaced by a list of absolute paths. Drop it and every acceptance step fails with
    // "command not found", which looks nothing like its cause.
    "PATH",
    // cargo, npm, rustup, pip and git all resolve their caches, registries and config under
    // `$HOME`. Without it they fall back to the passwd entry or to no cache at all: a cold
    // download on every step, and for git, no identity to commit with.
    "HOME",
    // Who a tool reports itself running as. git and a fair number of build scripts read them.
    "USER",
    "LOGNAME",
    // Locale. A toolchain that believes the encoding is ASCII mangles non-ASCII output, and a
    // run's diff and test names are compared as text.
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    // Some tools assume a dumb terminal without it and some assume none at all.
    "TERM",
    // Toolchain roots, for a machine that puts them somewhere other than under `$HOME`.
    "CARGO_HOME",
    "RUSTUP_HOME",
    // Where a network step finds the CA bundle. Distribution-dependent; without it an allowed
    // `npm install` fails TLS verification on any host that does not use the default location.
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
];

/// Exactly what `bwrap` is exec'd with. Split out so the argument list is a thing a test can
/// assert on, and so the environment the sandbox gets is decided in one readable place.
fn bwrap_argv(spec: &SandboxSpec) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "--ro-bind",
        "/",
        "/", //
        "--dev",
        "/dev", //
        "--proc",
        "/proc", //
        "--tmpfs",
        "/tmp", //
        // Before any `--setenv`: bwrap applies these in order, and this drops everything not
        // subsequently named.
        "--clearenv",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    for name in PASSED_THROUGH {
        // Absent, or not text — either way there is nothing to forward.
        if let Ok(value) = std::env::var(name) {
            args.push("--setenv".into());
            args.push((*name).to_string());
            args.push(value);
        }
    }

    // Mount each bind in place (guest path = host path): the host root is ro-bound, so bwrap
    // can't create fresh mount points like /workspace under it. Translate a guest workdir that
    // matches a mount to the corresponding host path.
    let mut chdir = spec.workdir.clone();
    for m in &spec.mounts {
        let host = m.host.to_string_lossy().into_owned();
        if spec.workdir == m.guest {
            chdir = host.clone();
        }
        args.push(if m.read_only { "--ro-bind" } else { "--bind" }.into());
        args.push(host.clone());
        args.push(host);
    }
    if !spec.network {
        args.push("--unshare-net".into());
    }
    // A PID namespace, so the command cannot outlive the call. Without it a build that leaves a
    // daemon behind keeps that daemon holding the inherited stdout pipe, and reading output waits
    // for a process that has no reason to exit — the command finished, the call never returns.
    // Here bwrap is the namespace's init: when it goes, everything it started goes with it.
    // (`--proc /proc` above is what makes that namespace's /proc correct, and is already passed.)
    args.push("--unshare-pid".into());
    // And if this process dies, the sandbox does not outlive it.
    args.push("--die-with-parent".into());
    args.push("--chdir".into());
    args.push(chdir);
    args.push("--".into());
    args.extend(spec.command.iter().cloned());
    args
}

/// bwrap/Landlock backend: the host root read-only, mounts bind-mounted writable, network
/// optionally unshared. No image needed — runs the host's toolchain against the mounted worktree.
async fn run_bwrap(spec: SandboxSpec) -> Result<ExecOutput, SandboxError> {
    if !BWRAP_LIMITS_WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            cpus = spec.cpus,
            memory_mib = spec.memory_mib,
            "sandbox cpus and memory_mib limits are not enforced by the landlock backend"
        );
    }
    let args = bwrap_argv(&spec);

    // Killed if this future is dropped — which is what a caller's timeout does. Without it a
    // timeout abandons the command rather than stopping it: the caller reports a timeout and the
    // work carries on unwatched, holding the worktree it was told to stop touching.
    let output = Command::new("bwrap")
        .args(&args)
        .kill_on_drop(true)
        .output()
        .await?;
    Ok(ExecOutput {
        exit_code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_command_rejected() {
        // Pure validation, no sandbox needed.
        let spec = SandboxSpec {
            backend: "landlock".into(),
            name: "t".into(),
            image: String::new(),
            workdir: "/".into(),
            mounts: vec![],
            command: vec![],
            cpus: 1,
            memory_mib: 256,
            network: false,
        };
        let err = futures_block(run(spec));
        assert!(matches!(err, Err(SandboxError::EmptyCommand)));
    }

    fn spec(command: &[&str]) -> SandboxSpec {
        SandboxSpec {
            backend: "landlock".into(),
            name: "t".into(),
            image: String::new(),
            workdir: "/".into(),
            mounts: vec![],
            command: command.iter().map(|c| (*c).to_string()).collect(),
            cpus: 1,
            memory_mib: 256,
            network: false,
        }
    }

    #[test]
    fn the_sandbox_starts_from_an_empty_environment() {
        // Stated here so removing it has to be deliberate. Without `--clearenv` every command the
        // sandbox runs — every acceptance step, every `Bash` call a model makes — can read this
        // process's `ANTHROPIC_API_KEY`, its `SSH_AUTH_SOCK` and its keyring address.
        let argv = bwrap_argv(&spec(&["true"]));
        let clear = argv
            .iter()
            .position(|a| a == "--clearenv")
            .expect("the environment is cleared");

        // And it comes before what re-adds: bwrap applies these in order, so a `--setenv` ahead of
        // `--clearenv` is a variable that gets set and then thrown away.
        for (i, a) in argv.iter().enumerate() {
            assert!(
                a != "--setenv" || i > clear,
                "--setenv at {i} precedes {clear}"
            );
        }
    }

    #[test]
    fn only_named_variables_survive_and_path_is_one_of_them() {
        // `PATH` is the one that cannot be replaced by anything else: node comes from nvm and
        // cargo from rustup, both per-user directories rather than fixed system locations. Drop it
        // and the toolchain vanishes with a message that looks nothing like its cause.
        let argv = bwrap_argv(&spec(&["true"]));
        let set: Vec<&String> = argv
            .iter()
            .enumerate()
            .filter(|(i, _)| i > &0 && argv[i - 1] == "--setenv")
            .map(|(_, name)| name)
            .collect();
        assert!(set.iter().any(|n| *n == "PATH"), "{set:?}");
        for name in &set {
            assert!(
                PASSED_THROUGH.contains(&name.as_str()),
                "{name} is set but not on the passlist"
            );
        }
    }

    #[test]
    fn bwrap_argv_has_no_limit_flags() {
        let argv = bwrap_argv(&SandboxSpec {
            cpus: 4,
            memory_mib: 8192,
            ..spec(&["true"])
        });
        assert!(!argv.iter().any(|arg| arg == "--cpus"), "{argv:?}");
        assert!(!argv.iter().any(|arg| arg == "--memory"), "{argv:?}");
    }

    #[test]
    fn oom_hint_only_marks_sigkill_as_a_likely_memory_limit() {
        let hint = oom_hint(137, 2048).expect("SIGKILL gets an OOM hint");
        assert!(hint.contains("likely"), "{hint}");
        assert!(hint.contains("2048 MiB memory limit"), "{hint}");
        for exit_code in [0, 1, 139] {
            assert_eq!(oom_hint(exit_code, 2048), None, "exit {exit_code}");
        }
    }

    fn container_spec(command: &[&str]) -> SandboxSpec {
        SandboxSpec {
            backend: "container".into(),
            // As a run names them. A runtime rejects a name shorter than two characters, and every
            // real one is `ratatoskr-<node>-<run>-<step>`, so the fixture uses that shape too.
            name: format!("ratatoskr-test-{}", std::process::id()),
            image: "docker.io/library/alpine:3".into(),
            mounts: vec![Mount {
                host: std::env::temp_dir(),
                guest: "/workspace".into(),
                read_only: false,
            }],
            workdir: "/workspace".into(),
            ..spec(command)
        }
    }

    fn linked_worktree() -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "ratatoskr-linked-git-{}-{}",
            std::process::id(),
            GIT_VIEW_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let checkout = root.join("checkout");
        let worktree = root.join("worktree");
        std::fs::create_dir_all(&checkout).unwrap();
        for args in [
            ["init"].as_slice(),
            ["config", "user.email", "test@example.invalid"].as_slice(),
            ["config", "user.name", "Sandbox test"].as_slice(),
        ] {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(&checkout)
                .status()
                .unwrap();
            assert!(status.success(), "git {args:?}");
        }
        std::fs::write(checkout.join("tracked"), "contents\n").unwrap();
        for args in [
            ["add", "tracked"].as_slice(),
            ["commit", "-m", "initial"].as_slice(),
            ["worktree", "add", "-b", "run", worktree.to_str().unwrap()].as_slice(),
        ] {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(&checkout)
                .status()
                .unwrap();
            assert!(status.success(), "git {args:?}");
        }
        (root, worktree)
    }

    #[test]
    fn linked_worktree_git_metadata_is_mounted_at_guest_only_paths() {
        let (root, worktree) = linked_worktree();
        let spec = SandboxSpec {
            mounts: vec![Mount {
                host: worktree.clone(),
                guest: "/workspace".into(),
                read_only: false,
            }],
            workdir: "/workspace".into(),
            ..container_spec(&["true"])
        };

        let (mapped, view) = with_linked_git_metadata(spec).unwrap();
        let view = view.expect("linked worktree gets Git metadata mounts");
        let dot_git = std::fs::read_to_string(view.temp_dir.join("dot-git")).unwrap();
        let gitdir = std::fs::read_to_string(view.temp_dir.join("gitdir")).unwrap();
        assert!(dot_git.starts_with("gitdir: /run/ratatoskr/git/common/worktrees/"));
        assert_eq!(gitdir, "/workspace/.git\n");
        assert!(!dot_git.contains(&root.display().to_string()));
        assert!(!gitdir.contains(&root.display().to_string()));

        let mounts = &mapped.mounts[1..];
        assert_eq!(mounts[0].guest, "/run/ratatoskr/git/common");
        assert!(mounts[0].read_only);
        assert!(
            mounts[1]
                .guest
                .starts_with("/run/ratatoskr/git/common/worktrees/")
        );
        assert!(!mounts[1].read_only, "Git refreshes the worktree index");
        assert_eq!(mounts[2].guest, "/workspace/.git");
        assert!(mounts[2].read_only);
        assert!(mounts[3].guest.ends_with("/gitdir"));
        assert!(mounts[3].read_only);

        drop(view);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn linked_worktree_git_metadata_accepts_a_relative_pointer() {
        let (root, worktree) = linked_worktree();
        // Git permits this form and resolves it from the directory holding the `.git` file.
        let metadata_name = std::fs::read_to_string(worktree.join(".git"))
            .unwrap()
            .strip_prefix("gitdir: ")
            .unwrap()
            .trim()
            .rsplit('/')
            .next()
            .unwrap()
            .to_string();
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: ../checkout/.git/worktrees/{metadata_name}\n"),
        )
        .unwrap();
        let spec = SandboxSpec {
            mounts: vec![Mount {
                host: worktree,
                guest: "/workspace".into(),
                read_only: false,
            }],
            workdir: "/workspace".into(),
            ..container_spec(&["true"])
        };

        let (_, view) = with_linked_git_metadata(spec).unwrap();
        assert!(view.is_some());
        drop(view);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ordinary_repository_git_directory_needs_no_metadata_view() {
        let root = std::env::temp_dir().join(format!(
            "ratatoskr-ordinary-git-{}-{}",
            std::process::id(),
            GIT_VIEW_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let spec = SandboxSpec {
            mounts: vec![Mount {
                host: root.clone(),
                guest: "/workspace".into(),
                read_only: false,
            }],
            workdir: "/workspace".into(),
            ..container_spec(&["true"])
        };
        let (mapped, view) = with_linked_git_metadata(spec).unwrap();
        assert!(view.is_none());
        assert_eq!(mapped.mounts.len(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_container_gets_the_mounts_and_no_host_root() {
        // The whole argument for this backend: what a command can read is the mounts, and nothing
        // else. There is no equivalent of `--ro-bind / /` here, so `$HOME` is not in scope — and a
        // test that asserted its absence would be asserting the absence of a string, so what is
        // checked is that every host path named is one the caller asked for.
        let argv = container_argv(
            &SandboxSpec {
                cpus: 4,
                memory_mib: 8192,
                ..container_spec(&["true"])
            },
            Some("1000:1000"),
        );
        assert!(argv.windows(2).any(|w| w == ["--cpus", "4"]), "{argv:?}");
        assert!(
            argv.windows(2).any(|w| w == ["--memory", "8192m"]),
            "{argv:?}"
        );
        let tmp = std::env::temp_dir().display().to_string();
        let volumes: Vec<&String> = argv
            .iter()
            .enumerate()
            .filter(|(i, _)| i > &0 && argv[i - 1] == "--volume")
            .map(|(_, v)| v)
            .collect();
        // At its guest path, not in place. This is what the bwrap backend cannot do — the host
        // root is `--ro-bind`ed there, so there is nowhere to create a fresh mount point — and it
        // is why a prepared cache is a container-backend feature: `node_modules` has to land where
        // a resolver looks, which is not where the cache is stored.
        assert_eq!(volumes, [&format!("{tmp}:/workspace")], "{argv:?}");

        let at = argv
            .iter()
            .position(|a| a == "--workdir")
            .expect("a workdir");
        assert_eq!(argv[at + 1], "/workspace", "{argv:?}");

        // Offline unless the step asked otherwise.
        assert!(
            argv.windows(2).any(|w| w == ["--network", "none"]),
            "{argv:?}"
        );
        let online = container_argv(
            &SandboxSpec {
                network: true,
                ..container_spec(&["true"])
            },
            Some("1000:1000"),
        );
        assert!(!online.iter().any(|a| a == "--network"), "{online:?}");

        // The image and the command come last, in that order, so nothing the caller supplies is
        // read as an option to the runtime.
        let image = argv
            .iter()
            .position(|a| a == "docker.io/library/alpine:3")
            .expect("the image");
        assert_eq!(argv[image + 1..], ["true"], "{argv:?}");
    }

    #[test]
    fn a_container_writes_as_the_host_user_not_as_root() {
        // A rootful runtime runs as root by default, and everything it writes into the mounted
        // worktree comes out owned by root: the host cannot then read its own diff, and
        // `git worktree remove` cannot delete the tree. Reported as a run that produced nothing.
        let argv = container_argv(&container_spec(&["true"]), Some("1000:1000"));
        assert!(
            argv.windows(2).any(|w| w == ["--user", "1000:1000"]),
            "{argv:?}"
        );

        // A read-only mount says so where the runtime reads it, which is what stops a check
        // rewriting the prepared cache that several runs are reading at once.
        let cached = container_argv(
            &SandboxSpec {
                mounts: vec![Mount {
                    host: "/repo/.ratatoskr/deps/node".into(),
                    guest: "/wt/web/node_modules".into(),
                    read_only: true,
                }],
                ..container_spec(&["true"])
            },
            None,
        );
        assert!(
            cached
                .iter()
                .any(|a| a == "/repo/.ratatoskr/deps/node:/wt/web/node_modules:ro"),
            "{cached:?}"
        );

        // And on a host where the id cannot be read, the argument is omitted rather than guessed
        // at — a wrong uid is worse than the runtime's default, which at least fails visibly.
        let argv = container_argv(&container_spec(&["true"]), None);
        assert!(!argv.iter().any(|a| a == "--user"), "{argv:?}");
    }

    #[test]
    fn the_host_user_is_this_process_own() {
        // The value the argv test uses a literal for. `/proc/self` is owned by the real user, so
        // this is the same pair `id -u`/`id -g` reports, without a libc dependency.
        let user = host_user().expect("a uid:gid on linux");
        let (uid, gid) = user.split_once(':').expect("uid:gid");
        assert!(uid.parse::<u32>().is_ok(), "{user}");
        assert!(gid.parse::<u32>().is_ok(), "{user}");
    }

    #[tokio::test]
    #[ignore = "requires a container runtime and the alpine image; run with --ignored"]
    async fn the_container_backend_runs_a_command_and_cannot_see_the_host_root() {
        let dir = std::env::temp_dir().join(format!("ratatoskr-ctr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("mounted"), "visible").unwrap();
        let spec = SandboxSpec {
            mounts: vec![Mount {
                host: dir.clone(),
                guest: "/workspace".into(),
                read_only: false,
            }],
            ..container_spec(&["sh", "-c", "cat mounted; ls /root 2>&1 | head -1"])
        };

        let out = run(spec).await.unwrap();
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
        // The mount is there…
        assert!(out.stdout.contains("visible"), "{}", out.stdout);
        // …and the host's home is not: what `/root` holds is the image's, which is empty.
        assert!(!out.stdout.contains(".ssh"), "{}", out.stdout);

        // And what it writes into the mount belongs to the host user. Owned by root instead, the
        // host cannot read the diff it is meant to review and cannot remove the worktree — a run
        // that did the work and delivered nothing.
        let out = run(SandboxSpec {
            mounts: vec![Mount {
                host: dir.clone(),
                guest: "/workspace".into(),
                read_only: false,
            }],
            command: vec![
                "sh".into(),
                "-c".into(),
                "echo made > from-container".into(),
            ],
            ..container_spec(&[])
        })
        .await
        .unwrap();
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);

        use std::os::unix::fs::MetadataExt as _;
        let written = std::fs::metadata(dir.join("from-container")).unwrap();
        let me = std::fs::metadata("/proc/self").unwrap();
        assert_eq!(
            (written.uid(), written.gid()),
            (me.uid(), me.gid()),
            "the container wrote as somebody else"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    #[ignore = "requires Docker and the locally built ratatoskr-checks image"]
    async fn a_container_can_use_git_in_a_linked_worktree_without_host_paths() {
        let (root, worktree) = linked_worktree();
        let spec = SandboxSpec {
            image: "ratatoskr-checks".into(),
            mounts: vec![Mount {
                host: worktree,
                guest: "/workspace".into(),
                read_only: false,
            }],
            workdir: "/workspace".into(),
            command: vec![
                "sh".into(),
                "-c".into(),
                "git status --short && git diff --check && \
                 test \"$(git rev-parse --show-toplevel)\" = /workspace && \
                 test \"$(cat .git)\" != *'/home/'* && \
                 test \"$(cat \"$(git rev-parse --git-dir)/gitdir\")\" = /workspace/.git"
                    .into(),
            ],
            ..container_spec(&[])
        };

        let output = run(spec).await.unwrap();
        assert_eq!(
            output.exit_code, 0,
            "stdout {} stderr {}",
            output.stdout, output.stderr
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    #[ignore = "requires bwrap on the host; run with --ignored"]
    async fn a_sandboxed_command_cannot_read_this_processs_secrets() {
        // The whole point, run for real rather than asserted about an argument list.
        // SAFETY: single-threaded test process; nothing else reads the environment concurrently.
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key") };
        let out = run(spec(&["sh", "-c", "echo [$ANTHROPIC_API_KEY]"]))
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
        assert!(
            !out.stdout.contains("sk-ant"),
            "the key reached the sandbox: {}",
            out.stdout
        );
    }

    #[tokio::test]
    #[ignore = "requires bwrap on the host; run with --ignored"]
    async fn the_toolchain_still_resolves_inside_the_sandbox() {
        // The other half of the same change: an empty environment that also loses `PATH` breaks
        // every acceptance step, and asserting only that the argv says `--clearenv` would not
        // notice. `cargo` is the toolchain this repository's own acceptance runs.
        let out = run(spec(&["sh", "-c", "command -v cargo"])).await.unwrap();
        assert_eq!(
            out.exit_code, 0,
            "cargo did not resolve; stdout {} stderr {}",
            out.stdout, out.stderr
        );
    }

    #[tokio::test]
    #[ignore = "requires bwrap on the host; run with --ignored"]
    async fn bwrap_backend_runs_a_command() {
        let spec = SandboxSpec {
            backend: "landlock".into(),
            name: "smoke".into(),
            image: String::new(),
            workdir: "/".into(),
            mounts: vec![],
            command: vec!["echo".into(), "sandbox-ok".into()],
            cpus: 1,
            memory_mib: 256,
            network: false,
        };
        let out = run(spec).await.unwrap();
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
        assert!(out.stdout.contains("sandbox-ok"));
    }

    #[tokio::test]
    #[ignore = "requires bwrap on the host; run with --ignored"]
    async fn a_command_that_leaves_a_process_behind_still_returns() {
        // The failure this guards: a background child inherits stdout, so reading output waits on
        // a process that will never exit even though the command finished seconds ago. The PID
        // namespace is what ends it — the child cannot outlive the sandbox it was started in.
        let spec = SandboxSpec {
            backend: "landlock".into(),
            name: "orphan".into(),
            image: String::new(),
            workdir: "/".into(),
            mounts: vec![],
            command: vec!["sh".into(), "-c".into(), "sleep 300 & echo started".into()],
            cpus: 1,
            memory_mib: 256,
            network: false,
        };
        let out = tokio::time::timeout(std::time::Duration::from_secs(20), run(spec))
            .await
            .expect("the call must not wait on the process it left behind")
            .unwrap();
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
        assert!(out.stdout.contains("started"));
    }

    #[cfg(feature = "microsandbox")]
    #[tokio::test]
    #[ignore = "provisions the microsandbox runtime + boots a MicroVM; needs KVM + network"]
    async fn microsandbox_backend_boots_and_runs() {
        // Download libkrunfw + runtime binaries (idempotent once installed).
        microsandbox::setup::install()
            .await
            .expect("microsandbox runtime install");

        let spec = SandboxSpec {
            backend: "microsandbox".into(),
            name: "ratatoskr-smoke".into(),
            image: "docker.io/library/alpine".into(),
            workdir: "/".into(),
            mounts: vec![],
            command: vec!["echo".into(), "microsandbox-ok".into()],
            cpus: 1,
            memory_mib: 512,
            network: true,
        };
        let out = run(spec).await.unwrap();
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
        assert!(out.stdout.contains("microsandbox-ok"));
    }

    fn futures_block<F: std::future::Future>(f: F) -> F::Output {
        // Minimal single-threaded block_on for a test that never actually awaits I/O.
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(f)
    }

    // --- container image resolution ------------------------------------------

    /// Restores `PATH` on drop. Prepending is safe for tests running alongside; replacing is not
    /// (a concurrent test spawning `git` would not find it), so the replaced window is kept to a
    /// single resolution call.
    struct PathGuard(Option<std::ffi::OsString>);

    impl PathGuard {
        fn prepended(dir: &Path) -> Self {
            let old = std::env::var_os("PATH");
            let mut paths = vec![dir.to_path_buf()];
            if let Some(old) = &old {
                paths.extend(std::env::split_paths(old));
            }
            // SAFETY: process-environment mutation races with other tests, but this only adds a
            // directory — every lookup that resolved before still resolves — and drop restores.
            unsafe { std::env::set_var("PATH", std::env::join_paths(paths).unwrap()) };
            Self(old)
        }

        fn replaced_with(dir: &Path) -> Self {
            let old = std::env::var_os("PATH");
            // SAFETY: races with concurrent tests' subprocess spawns for the guard's lifetime;
            // kept to the one call that must see a runtime-less PATH, and drop restores.
            unsafe { std::env::set_var("PATH", dir) };
            Self(old)
        }
    }

    impl Drop for PathGuard {
        fn drop(&mut self) {
            // SAFETY: restores exactly what the guard found.
            unsafe {
                match &self.0 {
                    Some(value) => std::env::set_var("PATH", value),
                    None => std::env::remove_var("PATH"),
                }
            }
        }
    }

    static FAKE_RUNTIME_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A directory posing as a container-runtime installation: an executable `docker` whose
    /// `inspect` answers with `digest` — raw when asked with `--format`, as a JSON array
    /// otherwise, because which of the two a correct implementation uses is its own business —
    /// and which logs every inspection to `<dir>/inspections` so a test can count them.
    fn fake_runtime_reporting(digest: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-fake-runtime-{}-{}",
            std::process::id(),
            FAKE_RUNTIME_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let inspections = dir.join("inspections");
        // One long format string rather than line continuations: the script's own indentation
        // is significant enough that eating it with Rust's `\`-newline would be a quiet bug.
        let script = format!(
            "#!/bin/sh\ncase \" $* \" in\n  *\" inspect \"*)\n    echo ask >> '{}'\n    case \" $* \" in\n      *\" --format \"*) echo '{digest}' ;;\n      *) printf '[{{\"Id\":\"{digest}\"}}]\\n' ;;\n    esac ;;\n  *) exit 1 ;;\nesac\n",
            inspections.display()
        );
        let docker = dir.join("docker");
        std::fs::write(&docker, script).unwrap();
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&docker, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        dir
    }

    /// A fake runtime whose `inspect` fails the way it does for an image that is not there.
    fn fake_runtime_without_the_image() -> PathBuf {
        let dir = fake_runtime_reporting("sha256:unused");
        let script = "#!/bin/sh\ncase \" $* \" in\n  *\" inspect \"*) echo 'Error: no such image' >&2; exit 1 ;;\n  *) exit 1 ;;\nesac\n";
        std::fs::write(dir.join("docker"), script).unwrap();
        dir
    }

    fn digest(seed: &str) -> String {
        // A realistic immutable id: `sha256:` plus 64 hex chars.
        format!("sha256:{}", seed.repeat(64 / seed.len()))
    }

    #[test]
    fn image_digests_must_be_canonical_sha256_hex() {
        assert!(is_image_digest(&digest("ab")));
        assert!(!is_image_digest("sha256:abc"));
        assert!(!is_image_digest(&format!("sha256:{}", "A".repeat(64))));
    }

    #[tokio::test]
    async fn an_image_tag_resolves_to_the_immutable_digest_the_runtime_reports() {
        // The contract's happy path: config names a mutable tag, the run executes and records
        // the immutable identifier the runtime reports for it. Exercised through the crate-root
        // path, which is where the contract puts it.
        let want = digest("ab");
        let dir = fake_runtime_reporting(&want);
        let _path = PathGuard::prepended(&dir);

        let resolved = crate::resolve_container_image("ratatoskr-checks")
            .await
            .unwrap();
        assert_eq!(resolved, want);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn image_resolution_fails_when_no_container_runtime_is_available() {
        // Selecting `container` on a host without Docker or Podman is an error that says so —
        // never an implicit downgrade to landlock, which would silently trade the isolation
        // the config asked for (no host root) for the weakest rung while reporting success.
        let dir = std::env::temp_dir().join(format!(
            "ratatoskr-no-runtime-{}-{}",
            std::process::id(),
            FAKE_RUNTIME_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let err = {
            let _path = PathGuard::replaced_with(&dir);
            crate::resolve_container_image("ratatoskr-checks")
                .await
                .unwrap_err()
        };
        // "Names that a container runtime is required" — the variant this crate already has for
        // exactly this, or a message carrying the same words.
        assert!(
            matches!(err, SandboxError::NoContainerRuntime)
                || err.to_string().contains("container runtime"),
            "{err}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn image_resolution_fails_when_the_runtime_cannot_inspect_the_image() {
        // An image that is not there must fail the sandboxed phase — running the mutable tag
        // instead would pin nothing and record provenance for an image that did not run.
        let dir = fake_runtime_without_the_image();
        let _path = PathGuard::prepended(&dir);

        let err = crate::resolve_container_image("ratatoskr-checks")
            .await
            .unwrap_err();
        assert!(!err.to_string().is_empty(), "{err}");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn image_resolution_refuses_an_inspect_result_that_is_not_an_immutable_digest() {
        // `image inspect` answered but what it returned is another mutable reference, not a
        // `sha256:` identifier. Recording that as provenance would claim a pin that does not
        // exist; the phase fails instead.
        let dir = fake_runtime_reporting("ratatoskr-checks:latest");
        let _path = PathGuard::prepended(&dir);

        let err = crate::resolve_container_image("ratatoskr-checks")
            .await
            .unwrap_err();
        assert!(!err.to_string().is_empty(), "{err}");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    #[ignore = "requires a container runtime and the alpine image; run with --ignored"]
    async fn a_resolved_digest_runs_in_place_of_the_tag() {
        // End to end against a real runtime: the tag resolves to a digest, and the digest — not
        // the tag — is what the sandbox boots.
        let resolved = crate::resolve_container_image("docker.io/library/alpine:3")
            .await
            .unwrap();
        assert!(resolved.starts_with("sha256:"), "{resolved}");

        let out = run(SandboxSpec {
            image: resolved,
            ..container_spec(&["true"])
        })
        .await
        .unwrap();
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    }

    #[test]
    fn the_readme_states_what_each_sandbox_rung_exposes() {
        // The fallback's exposure is stated, not implied: a reader choosing `landlock` is
        // choosing the host root read-only and the host's toolchain, and "sandboxed" reads as
        // "isolated" unless the documentation says otherwise. Likewise the container rung is
        // documented as mounts-only with a pinned image — not asserted here as exact prose,
        // which would make a wording edit a test failure, but as the facts being present.
        let readme = include_str!("../../../README.md");
        let paragraph = readme
            .split("\n\n")
            .find(|p| p.contains("sandbox") && p.contains("landlock"))
            .expect("the README documents the sandbox backends");
        assert!(
            paragraph.contains("container"),
            "the README names the container backend: {paragraph}"
        );
        assert!(
            paragraph.contains("host root")
                || paragraph.contains("host's toolchain")
                || paragraph.contains("host toolchain"),
            "the README states that landlock exposes the host root and toolchain: {paragraph}"
        );
    }
}
