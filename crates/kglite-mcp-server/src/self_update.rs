//! Stale-build detection: tell the agent when the code this server runs was
//! replaced on disk (`pip install -U`, `cargo install`, a rebuilt dev binary).
//!
//! The framework (`mcp_methods::server::SelfCheck`) compares a stat snapshot of
//! a watched file on every tool call. This module decides *which* file that is,
//! what version is installed there, and which name and version this process
//! reports.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use mcp_methods::server::SelfCheck;

use crate::skills::SkillRefresher;
use crate::tools::GraphState;

/// Name and version a server reports in its stale notice, `ping` and
/// `serverInfo`. The default is this crate; an embedder that ships its own
/// package overrides it with [`crate::ServerExtensions::with_identity`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Identity {
    pub(crate) name: String,
    pub(crate) version: String,
}

impl Default for Identity {
    fn default() -> Self {
        Self {
            name: "kglite-mcp-server".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// How the watched file ships, which decides how its installed version is read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Artifact {
    /// The running executable itself (cargo binary, `sonagram-mcp-server`).
    Binary(PathBuf),
    /// A shared object loaded into another host process: the wheel's extension.
    Extension(PathBuf),
}

impl Artifact {
    pub(crate) fn path(&self) -> &Path {
        match self {
            Artifact::Binary(path) | Artifact::Extension(path) => path,
        }
    }
}

/// `current_exe()` with the ` (deleted)` suffix Linux appends once the file
/// was unlinked, so the path still names what a replacement will recreate.
pub(crate) fn current_exe_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let text = exe.to_string_lossy();
    Some(match text.strip_suffix(" (deleted)") {
        Some(stripped) => PathBuf::from(stripped),
        None => exe,
    })
}

/// The object file that contains this very function.
#[cfg(unix)]
fn loaded_object_path() -> Option<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    // SAFETY: `dladdr` only reads the address and fills `info`; `info` is
    // zero-initialised and `dli_fname` is read only when `dladdr` reported
    // success, as a NUL-terminated string owned by the loader.
    unsafe {
        let mut info: libc::Dl_info = std::mem::zeroed();
        let anchor = loaded_object_path as fn() -> Option<PathBuf>;
        if libc::dladdr(anchor as *const libc::c_void, &mut info) == 0 || info.dli_fname.is_null() {
            return None;
        }
        let name = std::ffi::CStr::from_ptr(info.dli_fname);
        Some(PathBuf::from(std::ffi::OsStr::from_bytes(name.to_bytes())))
    }
}

#[cfg(not(unix))]
fn loaded_object_path() -> Option<PathBuf> {
    None
}

fn canonical(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

/// Pure classification of the two candidate paths: the object containing this
/// code (`dladdr`) and the process executable.
pub(crate) fn classify(object: Option<PathBuf>, exe: Option<PathBuf>) -> Option<Artifact> {
    match (object.map(canonical), exe.map(canonical)) {
        (Some(object), Some(exe)) if object == exe => Some(Artifact::Binary(exe)),
        (Some(object), _) => Some(Artifact::Extension(object)),
        (None, Some(exe)) => Some(Artifact::Binary(exe)),
        (None, None) => None,
    }
}

pub(crate) fn watched_artifact() -> Option<Artifact> {
    classify(loaded_object_path(), current_exe_path())
}

/// `<site>/<pkg>-<version>.dist-info` next to the package directory that holds
/// the extension; the version of the newest one when pip left several.
pub(crate) fn dist_info_version(extension: &Path) -> Option<String> {
    let package_dir = extension.parent()?;
    let package = package_dir.file_name()?.to_str()?;
    let site = package_dir.parent()?;
    let wanted = normalise_dist_name(package);
    let mut found: Vec<(Vec<u64>, String)> = Vec::new();
    for entry in std::fs::read_dir(site).ok()?.flatten() {
        let name = entry.file_name();
        let Some(stem) = name.to_str().and_then(|n| n.strip_suffix(".dist-info")) else {
            continue;
        };
        // `<name>-<version>`: a normalised name holds no `-`, so the first one splits.
        let Some((dist, version)) = stem.split_once('-') else {
            continue;
        };
        if normalise_dist_name(dist) == wanted {
            found.push((version_key(version), version.to_string()));
        }
    }
    found.into_iter().max().map(|(_, version)| version)
}

fn normalise_dist_name(name: &str) -> String {
    name.to_ascii_lowercase().replace(['-', '.'], "_")
}

fn version_key(version: &str) -> Vec<u64> {
    version
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().unwrap_or(0))
        .collect()
}

/// Version printed by `<program> --version` (`name X.Y.Z`), within `timeout`.
pub(crate) fn probe_binary_version(program: &Path, timeout: Duration) -> Option<String> {
    let mut child = Command::new(program)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait().ok()? {
            Some(status) if status.success() => break,
            Some(_) => return None,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    let mut out = String::new();
    std::io::Read::read_to_string(&mut child.stdout.take()?, &mut out).ok()?;
    parse_version_line(&out)
}

pub(crate) fn parse_version_line(output: &str) -> Option<String> {
    let token = output.lines().next()?.split_whitespace().last()?;
    token
        .starts_with(|c: char| c.is_ascii_digit())
        .then(|| token.to_string())
}

/// The framework check for this process, or `None` when no file can be named.
/// `reload` additionally enables `ping(reload=true)` when a launch command
/// for the new build can be derived.
pub(crate) fn build_self_check(
    identity: &Identity,
    reload: Option<ReloadParts>,
) -> Option<SelfCheck> {
    let artifact = watched_artifact()?;
    let probe = artifact.clone();
    let check = SelfCheck::new(
        identity.name.clone(),
        artifact.path().to_path_buf(),
        identity.version.clone(),
    )
    .with_installed_version(move || match &probe {
        Artifact::Extension(path) => dist_info_version(path),
        Artifact::Binary(path) => probe_binary_version(path, Duration::from_secs(5)),
    });
    Some(match reload {
        Some(parts) => attach_reload(check, &artifact, parts),
        None => check,
    })
}

/// What reload needs from the running server.
pub(crate) struct ReloadParts {
    /// The process's own arguments after `argv[0]`.
    pub(crate) args: Vec<OsString>,
    pub(crate) graph_state: GraphState,
    pub(crate) refresher: SkillRefresher,
    /// `--graph` mode only: the served file can be re-bound by `load_graph`,
    /// `create_graph` and `save_graph_as`, so it is carried across.
    pub(crate) carry_graph: bool,
}

/// How to launch a fresh server: the executable itself for a binary, the
/// wheel's `KGLITE_MCP_RESPAWN` vector for an extension hosted by Python, and
/// nothing for an extension launched some other way (reload then stays off and
/// the notice says to restart the client).
fn launch_vector(artifact: &Artifact) -> Option<(OsString, Vec<OsString>)> {
    match artifact {
        Artifact::Binary(path) => Some((path.clone().into_os_string(), Vec::new())),
        Artifact::Extension(_) => {
            let raw = std::env::var("KGLITE_MCP_RESPAWN").ok()?;
            crate::selftest::parse_respawn_vector(&raw).ok()
        }
    }
}

const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

fn attach_reload(check: SelfCheck, artifact: &Artifact, parts: ReloadParts) -> SelfCheck {
    let Some((program, lead)) = launch_vector(artifact) else {
        return check;
    };
    let mut argv = vec![program.clone()];
    argv.extend(lead.iter().cloned());
    argv.extend(parts.args.iter().cloned());
    let version_args: Vec<OsString> = lead
        .iter()
        .cloned()
        .chain([OsString::from("--version")])
        .collect();
    let preflight_args: Vec<OsString> = lead
        .iter()
        .cloned()
        .chain(parts.args.iter().cloned())
        .chain([OsString::from("--reload-preflight")])
        .collect();

    let dirty_check = parts.graph_state.clone();
    let release = parts.graph_state.clone();
    let save = parts.graph_state.clone();
    let restore_state = parts.graph_state;
    let refresher = parts.refresher;
    let carry_graph = parts.carry_graph;
    check
        .with_reload(argv)
        .with_can_reload(move || {
            if dirty_check.is_dirty() {
                Err(
                    "the active graph has unsaved changes that a reload would lose; \
                     call save_graph to keep them or reload_graph(discard_unsaved=true) \
                     to drop them, then reload again"
                        .to_string(),
                )
            } else {
                Ok(())
            }
        })
        .with_selftest(move |_| {
            run_probe(&program, &version_args, "--version")?;
            run_probe(&program, &preflight_args, "--reload-preflight")
        })
        // Exec skips every destructor, so the active graph is dropped here:
        // that removes a writable disk graph's `.working-<pid>-*` directory
        // (the new image has the same pid and restarts the nonce at 0), the
        // spill directories, and stamps the writer lease as released.
        .with_before_exec(move || {
            *crate::tools::write_lock(&release.inner) = None;
        })
        .with_carry(
            move || {
                if !carry_graph {
                    return None;
                }
                let path = save.source_path()?;
                Some(serde_json::json!({ "graph_source_path": path }))
            },
            move |value| {
                let Some(path) = value.get("graph_source_path").and_then(|v| v.as_str()) else {
                    return Ok(());
                };
                let path = Path::new(path);
                if restore_state.source_path().as_deref() == Some(path) {
                    return Ok(());
                }
                restore_state
                    .open_or_create(path, None)
                    .map_err(|e| format!("could not reopen the graph {}: {e}", path.display()))?;
                refresher.refresh();
                Ok(())
            },
        )
}

/// Run the new build with `args`; `Err` names the check and the child's last
/// stderr lines. The child is killed and reaped on timeout.
fn run_probe(program: &std::ffi::OsStr, args: &[OsString], what: &str) -> Result<(), String> {
    let mut child = Command::new(program)
        .args(args)
        .env_remove("MCP_METHODS_RELOAD_HANDOVER")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("the new build could not be started for {what}: {e}"))?;
    let stderr = child.stderr.take();
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut stderr) = stderr {
            let _ = std::io::Read::read_to_string(&mut stderr, &mut text);
        }
        text
    });
    let deadline = Instant::now() + PROBE_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(format!(
                    "the new build did not finish {what} within {} s",
                    PROBE_TIMEOUT.as_secs()
                ));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => break Err(format!("waiting for {what} failed: {e}")),
        }
    };
    let stderr = reader.join().unwrap_or_default();
    let status = status?;
    if status.success() {
        return Ok(());
    }
    let tail: Vec<&str> = stderr.lines().rev().take(5).collect();
    let tail = tail.into_iter().rev().collect::<Vec<_>>().join(" | ");
    Err(format!("the new build failed {what} ({status}): {tail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wheel_dist_info_is_found_beside_the_package_directory() {
        let site = tempfile::tempdir().unwrap();
        let package = site.path().join("kglite");
        std::fs::create_dir_all(&package).unwrap();
        let so = package.join("kglite.abi3.so");
        std::fs::write(&so, b"x").unwrap();
        for dir in [
            "kglite-0.20.0.dist-info",
            "kglite-0.20.1.dist-info",
            "kglite-0.9.9.dist-info",
            "kglite_extra-9.0.0.dist-info",
            "other-5.0.dist-info",
        ] {
            std::fs::create_dir(site.path().join(dir)).unwrap();
        }
        assert_eq!(dist_info_version(&so).as_deref(), Some("0.20.1"));
    }

    #[test]
    fn dist_names_normalise_dashes_and_case() {
        let site = tempfile::tempdir().unwrap();
        let package = site.path().join("my_pkg");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::create_dir(site.path().join("My_Pkg-1.2.3.dist-info")).unwrap();
        assert_eq!(
            dist_info_version(&package.join("ext.so")).as_deref(),
            Some("1.2.3")
        );
    }

    #[test]
    fn missing_dist_info_is_unknown() {
        let site = tempfile::tempdir().unwrap();
        let package = site.path().join("kglite");
        std::fs::create_dir_all(&package).unwrap();
        assert_eq!(dist_info_version(&package.join("kglite.abi3.so")), None);
    }

    #[test]
    fn version_line_parsing() {
        assert_eq!(
            parse_version_line("kglite-mcp-server 0.20.1\n").as_deref(),
            Some("0.20.1")
        );
        assert_eq!(parse_version_line("error: nope\n"), None);
        assert_eq!(parse_version_line(""), None);
    }

    #[test]
    fn object_equal_to_the_executable_is_a_binary_else_an_extension() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("server");
        let so = dir.path().join("pkg").join("ext.so");
        std::fs::create_dir_all(so.parent().unwrap()).unwrap();
        std::fs::write(&exe, b"x").unwrap();
        std::fs::write(&so, b"x").unwrap();
        let same = classify(Some(exe.clone()), Some(exe.clone())).unwrap();
        assert!(matches!(same, Artifact::Binary(_)));
        let ext = classify(Some(so), Some(exe.clone())).unwrap();
        assert!(matches!(ext, Artifact::Extension(_)));
        assert!(matches!(
            classify(None, Some(exe)).unwrap(),
            Artifact::Binary(_)
        ));
        assert_eq!(classify(None, None), None);
    }

    #[test]
    fn this_test_binary_resolves_to_itself() {
        let artifact = watched_artifact().expect("resolves");
        assert!(matches!(artifact, Artifact::Binary(_)), "{artifact:?}");
        assert_eq!(
            artifact.path(),
            std::env::current_exe().unwrap().canonicalize().unwrap()
        );
    }

    #[test]
    fn default_identity_is_this_crate() {
        let identity = Identity::default();
        assert_eq!(identity.name, "kglite-mcp-server");
        assert_eq!(identity.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn an_embedder_identity_is_what_the_server_reports() {
        use rmcp::ServerHandler;
        let extensions = crate::ServerExtensions::new().with_identity("codingest", "9.8.7");
        assert_eq!(extensions.identity.name, "codingest");
        let check = build_self_check(&extensions.identity, None).expect("watched file");
        let mode = crate::cli::Mode::Graph {
            path: std::path::PathBuf::from("g.kgl"),
        };
        let options = mcp_methods::server::ServerOptions::from_manifest(
            None,
            &crate::cli::fallback_name(&mode, &extensions.identity),
        )
        .with_self_check(check);
        let info = mcp_methods::server::McpServer::new(options).get_info();
        assert_eq!(info.server_info.version, "9.8.7");
        assert_eq!(info.server_info.name, "codingest (single-graph)");
    }

    #[test]
    fn the_default_identity_keeps_the_kglite_server_name() {
        let identity = Identity::default();
        let graph = crate::cli::Mode::Graph {
            path: std::path::PathBuf::from("g.kgl"),
        };
        assert_eq!(
            crate::cli::fallback_name(&graph, &identity),
            "KGLite (single-graph)"
        );
        assert_eq!(
            crate::cli::fallback_name(&crate::cli::Mode::Bare, &identity),
            "KGLite"
        );
    }
}
