//! Sandboxed command execution behind a backend-agnostic interface.
//!
//! `run(SandboxSpec)` is Ratatoskr's own contract — no microsandbox type leaks into it, so the
//! bwrap/Landlock fallback is a genuine swap rather than a rewrite. Two backends implement it:
//! `microsandbox` (a MicroVM, needs KVM + an OCI image) and `landlock` (bubblewrap + the host
//! filesystem, no image, works wherever `bwrap` + Landlock are present).

use std::path::PathBuf;

use tokio::process::Command;

/// A host→guest bind mount (the worktree, or the baseline checkout for red-team).
#[derive(Debug, Clone)]
pub struct Mount {
    pub host: PathBuf,
    pub guest: String,
}

/// What to run, where, and under what limits. Ratatoskr's own type — backend-neutral.
#[derive(Debug, Clone)]
pub struct SandboxSpec {
    /// `"microsandbox"` or `"landlock"`.
    pub backend: String,
    /// Unique sandbox name (microsandbox requires one; ignored by the bwrap backend).
    pub name: String,
    /// OCI image to boot (microsandbox only; the bwrap backend uses the host root).
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
    #[error("unknown sandbox backend {0:?} (expected \"microsandbox\" or \"landlock\")")]
    UnknownBackend(String),
    #[error("empty sandbox command")]
    EmptyCommand,
    #[error("microsandbox error: {0}")]
    Microsandbox(String),
    #[error("bwrap not found or failed to launch: {0}")]
    Bwrap(#[from] std::io::Error),
}

/// Run `spec.command` in a sandbox, returning its captured output. Dispatches on `spec.backend`.
pub async fn run(spec: SandboxSpec) -> Result<ExecOutput, SandboxError> {
    if spec.command.is_empty() {
        return Err(SandboxError::EmptyCommand);
    }
    match spec.backend.as_str() {
        "microsandbox" => run_microsandbox(spec).await,
        "landlock" | "bwrap" => run_bwrap(spec).await,
        other => Err(SandboxError::UnknownBackend(other.to_string())),
    }
}

/// microsandbox (MicroVM) backend: boot the image with the mounts, run, tear down.
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

/// bwrap/Landlock backend: the host root read-only, mounts bind-mounted writable, network
/// optionally unshared. No image needed — runs the host's toolchain against the mounted worktree.
async fn run_bwrap(spec: SandboxSpec) -> Result<ExecOutput, SandboxError> {
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
    ]
    .into_iter()
    .map(String::from)
    .collect();

    for m in &spec.mounts {
        args.push("--bind".into());
        args.push(m.host.to_string_lossy().into_owned());
        args.push(m.guest.clone());
    }
    if !spec.network {
        args.push("--unshare-net".into());
    }
    args.push("--chdir".into());
    args.push(spec.workdir.clone());
    args.push("--".into());
    args.extend(spec.command.iter().cloned());

    let output = Command::new("bwrap").args(&args).output().await?;
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
