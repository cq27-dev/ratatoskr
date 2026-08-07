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
//!   is the rung that removes the host root without requiring KVM, and it is where a run that
//!   installs anything belongs.
//! - `microsandbox` — a **MicroVM**, adding a VM boundary on top of the image. Needs KVM, and is
//!   behind `--features microsandbox` because its build script needs the network.

use std::path::PathBuf;

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

/// The first runtime on `PATH`, or nothing.
fn container_runtime() -> Option<&'static str> {
    RUNTIMES.iter().copied().find(|runtime| {
        std::env::var_os("PATH")
            .map(|path| std::env::split_paths(&path).any(|dir| dir.join(runtime).is_file()))
            .unwrap_or(false)
    })
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
    Ok(ExecOutput {
        exit_code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
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

    #[test]
    fn a_container_gets_the_mounts_and_no_host_root() {
        // The whole argument for this backend: what a command can read is the mounts, and nothing
        // else. There is no equivalent of `--ro-bind / /` here, so `$HOME` is not in scope — and a
        // test that asserted its absence would be asserting the absence of a string, so what is
        // checked is that every host path named is one the caller asked for.
        let argv = container_argv(&container_spec(&["true"]), Some("1000:1000"));
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
}
