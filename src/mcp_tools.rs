//! pyo3 wrappers exposing `mcp-methods` framework tools to Python.
//!
//! 0.9.21+: instead of reimplementing read_source / grep / list_source /
//! github_issues / git_api / workspace tools in pure Python, kglite
//! depends on the pure-Rust `mcp-methods` crate (0.3.26+, three-crate
//! split with zero pyo3 in the library half) and exposes its functions
//! to the Python entry point via this module.
//!
//! 0.9.24+ extends this surface with `Manifest`, `Workspace`,
//! `start_watch` / `WatchHandle`, and `load_env_walk` — replacing the
//! pure-Python re-implementations that previously lived in
//! `kglite/mcp_server/{manifest,workspace,watch}.py`. The Python
//! wrappers under those paths become thin shims around these classes,
//! which inherit mcp-methods' validated behaviour (e.g. atomic-swap
//! sandbox checks in `Workspace::set_root_dir` that the prior Python
//! version got wrong by mutating its own `self.root`).
//!
//! The kglite cdylib already uses pyo3's `extension-module` feature
//! (no libpython link, abi3 portable across Python 3.10+), so wrapping
//! more Rust code here costs nothing in the wheel matrix. From the
//! agent's perspective the tool surface and output format are exactly
//! what the 0.9.18 Rust binary produced — same crate, same code paths.
//!
//! Registered as a submodule `kglite._mcp_internal` so it stays clearly
//! internal. The Python `kglite.mcp_server.server` entry point is the
//! intended consumer.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use mcp_methods::cache::ElementCache as RustElementCache;
use mcp_methods::github::{git_api_internal, github_issues_rust, has_git_token};
use mcp_methods::server::source::{self, GrepOpts, ListOpts, ReadOpts};
use mcp_methods::server::{
    find_sibling_manifest, find_workspace_manifest, load_manifest as load_manifest_rust,
    Manifest as RustManifest, PostActivateHook, ResolvedRegistry as RustResolvedRegistry,
    Skill as RustSkill, SkillRegistry as RustSkillRegistry, Workspace as RustWorkspace,
};

// ---------------------------------------------------------------------------
// source tools
// ---------------------------------------------------------------------------

/// File slice read against the configured source roots. Mirrors
/// `mcp_methods::server::source::read_source`. Path traversal attempts
/// return an `Error: ...` string rather than raising — same convention
/// as the 0.9.18 binary.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    file_path, source_roots, *,
    start_line=None, end_line=None,
    grep=None, grep_context=None,
    max_matches=None, max_chars=None,
))]
fn read_source(
    file_path: &str,
    source_roots: Vec<String>,
    start_line: Option<usize>,
    end_line: Option<usize>,
    grep: Option<String>,
    grep_context: Option<usize>,
    max_matches: Option<usize>,
    max_chars: Option<usize>,
) -> String {
    let opts = ReadOpts {
        start_line,
        end_line,
        grep,
        grep_context,
        max_matches,
        max_chars,
    };
    source::read_source(file_path, &source_roots, &opts)
}

/// Recursive regex search across source roots (ripgrep walker, .gitignore
/// aware). Output format: `path:lineno:content` for matches,
/// `path-lineno-content` for context.
#[pyfunction]
#[pyo3(signature = (
    pattern, source_roots, *,
    glob=None, context=0,
    max_results=None, case_insensitive=false,
))]
fn grep(
    pattern: &str,
    source_roots: Vec<String>,
    glob: Option<String>,
    context: usize,
    max_results: Option<usize>,
    case_insensitive: bool,
) -> String {
    let opts = GrepOpts {
        glob,
        context,
        max_results,
        case_insensitive,
    };
    source::grep(&source_roots, pattern, &opts)
}

/// Tree-style directory listing relative to the primary source root.
#[pyfunction]
#[pyo3(signature = (
    source_roots, *,
    path=String::from("."), depth=1,
    glob=None, dirs_only=false,
))]
fn list_source(
    source_roots: Vec<String>,
    path: String,
    depth: usize,
    glob: Option<String>,
    dirs_only: bool,
) -> String {
    // resolve_dir_under_roots picks the first root as primary.
    let target = match source::resolve_dir_under_roots(&path, &source_roots) {
        Some(p) => p,
        None => return format!("Error: path '{}' does not exist or access denied.", path),
    };
    let primary_root = match source_roots
        .first()
        .and_then(|d| PathBuf::from(d).canonicalize().ok())
    {
        Some(p) => p,
        None => return "Error: no source roots configured.".to_string(),
    };
    let opts = ListOpts {
        depth,
        glob,
        dirs_only,
    };
    source::list_source(&target, &primary_root, &opts)
}

/// Liveness probe — kept on the pyo3 side since it's trivial and there's
/// no equivalent in `mcp-methods` (the Rust framework exposes ping via
/// its `Server` boilerplate, which we don't use).
#[pyfunction]
#[pyo3(signature = (message=None))]
fn ping(message: Option<String>) -> String {
    message.unwrap_or_else(|| "pong".to_string())
}

// ---------------------------------------------------------------------------
// github tools
// ---------------------------------------------------------------------------

/// Generic GitHub REST GET. Relative paths (`pulls?state=open`) auto-
/// prefix with `/repos/<repo>/`; absolute paths (`search/issues?q=...`)
/// pass through. Returns pretty-printed JSON, truncated at `truncate_at`
/// chars.
#[pyfunction]
#[pyo3(signature = (repo, path, *, truncate_at=80_000))]
fn git_api(repo: &str, path: &str, truncate_at: usize) -> String {
    git_api_internal(repo, path, truncate_at)
}

/// Module-level check — used by the Python entry point to decide
/// whether to register the github_* tools at boot.
#[pyfunction]
fn has_github_token() -> bool {
    has_git_token()
}

/// `github_issues` tool. Stateful via [`RustElementCache`] for
/// drill-down (cb_N, comment_N, patch_N). Holds the cache across calls
/// so a follow-up `fetch_issue(number=N, element_id="cb_3")` doesn't
/// re-fetch from GitHub.
#[pyclass]
struct GithubIssues {
    cache: Mutex<RustElementCache>,
}

#[pymethods]
impl GithubIssues {
    #[new]
    fn new() -> Self {
        Self {
            cache: Mutex::new(RustElementCache::new()),
        }
    }

    /// FETCH mode: number given. Returns full issue/PR/discussion body
    /// (with element collapsing when over budget) — drill-down via
    /// element_id + optional lines/grep/context. `refresh=true` bypasses
    /// the cache.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        repo, number, *,
        element_id=None, lines=None,
        grep=None, context=3, refresh=false,
    ))]
    fn fetch(
        &self,
        py: Python<'_>,
        repo: &str,
        number: u64,
        element_id: Option<String>,
        lines: Option<String>,
        grep: Option<String>,
        context: usize,
        refresh: bool,
    ) -> String {
        py.detach(|| {
            let mut guard = self.cache.lock().unwrap();
            guard.fetch_issue(
                repo,
                number,
                element_id.as_deref(),
                lines.as_deref(),
                grep.as_deref(),
                context,
                refresh,
            )
        })
    }

    /// SEARCH or LIST mode — neither number nor element_id; query or
    /// no-args. Mirrors `github_issues_rust` dispatch.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        repo=None, query=None, kind=String::from("all"),
        state=String::from("open"), sort=None, limit=20, labels=None,
    ))]
    fn search_or_list(
        &self,
        py: Python<'_>,
        repo: Option<String>,
        query: Option<String>,
        kind: String,
        state: String,
        sort: Option<String>,
        limit: usize,
        labels: Option<String>,
    ) -> String {
        py.detach(|| {
            github_issues_rust(
                repo.as_deref(),
                None,
                query.as_deref(),
                &kind,
                &state,
                sort.as_deref(),
                limit,
                labels.as_deref(),
            )
        })
    }
}

// ---------------------------------------------------------------------------
// JSON → Python conversion (used by Manifest::as_dict)
// ---------------------------------------------------------------------------

/// Recursively convert a `serde_json::Value` into the closest Python
/// equivalent (None, bool, int, float, str, list, dict). Integer
/// representations are preferred over float when the JSON Number is
/// exactly integral — matches what Python's `json.loads` would produce.
///
/// Used by [`Manifest::as_dict`] and the `extensions` passthrough path.
/// Kept generic on the Rust side because mcp-methods 0.3.27's
/// `Manifest::to_json()` collapses every per-field getter into one
/// `serde_json::Value` round-trip; the converter below is the only
/// piece of schema-aware code we need on the kglite side.
fn json_value_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use serde_json::Value;
    match v {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => b.into_py_any(py),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py_any(py)
            } else if let Some(u) = n.as_u64() {
                u.into_py_any(py)
            } else {
                // serde_json rejects non-finite floats at parse time,
                // so `as_f64` is total in practice.
                n.as_f64()
                    .expect("serde_json::Number is finite")
                    .into_py_any(py)
            }
        }
        Value::String(s) => s.clone().into_py_any(py),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_value_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, val) in map {
                dict.set_item(k, json_value_to_py(py, val)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// Validated MCP manifest. Wraps `mcp_methods::server::Manifest`.
///
/// The Rust side owns YAML parsing, strict-unknown-key validation, and
/// all field invariants (e.g. `workspace.kind: local` requires
/// `workspace.root`); the Python wrapper is a single `as_dict()` view
/// over `Manifest::to_json()`. Field drift between mcp-methods and
/// downstream consumers is now a non-issue: the JSON shape is the
/// contract.
#[pyclass(name = "Manifest")]
pub struct PyManifest {
    inner: RustManifest,
}

#[pymethods]
impl PyManifest {
    /// Parse a manifest YAML file from disk. Raises `ValueError` on
    /// parse, unknown-key, or schema validation failures — same error
    /// shape that `mcp_methods::server::load` produces, surfaced
    /// directly to the operator.
    #[staticmethod]
    fn load(path: &str) -> PyResult<PyManifest> {
        match load_manifest_rust(Path::new(path)) {
            Ok(inner) => Ok(PyManifest { inner }),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    /// Return the validated manifest as a Python dict. Shape is
    /// guaranteed stable across mcp-methods patch releases (see
    /// `mcp_methods::server::Manifest::to_json` docstring); patch
    /// bumps may add fields non-breaking, renames or removals would
    /// constitute a minor/major bump.
    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_value_to_py(py, &self.inner.to_json())
    }

    /// Resolved manifest YAML path — useful for the Python side when
    /// resolving manifest-relative paths (source_root, env_file).
    #[getter]
    fn yaml_path(&self) -> String {
        self.inner.yaml_path.display().to_string()
    }

    /// Auto-detect a `<basename>_mcp.yaml` sibling next to a graph file.
    /// Returns the absolute path as a string, or None if no sibling
    /// manifest exists.
    #[staticmethod]
    fn find_sibling(graph_path: &str) -> Option<String> {
        find_sibling_manifest(Path::new(graph_path)).map(|p| p.display().to_string())
    }

    /// Auto-detect a `workspace_mcp.yaml` inside a workspace directory.
    #[staticmethod]
    fn find_workspace(workspace_dir: &str) -> Option<String> {
        find_workspace_manifest(Path::new(workspace_dir)).map(|p| p.display().to_string())
    }
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

/// Deferred slot for a Python `post_activate` callable.
///
/// `mcp_methods::server::Workspace` takes the post-activate hook at
/// construction (inside an `Arc<WorkspaceInner>` that is immutable
/// afterwards), but the Python boot sequence often needs to register
/// the hook AFTER the workspace exists (the hook typically closes
/// over a graph-state slot that hasn't been initialised yet). We
/// resolve that by always passing a Rust hook that dispatches through
/// this `OnceLock`. Python sets the callable via
/// `Workspace.set_post_activate` after construction; if no callable is
/// ever set, the hook is a no-op.
#[derive(Default)]
struct DeferredPyHook {
    callable: OnceLock<Py<PyAny>>,
}

impl DeferredPyHook {
    fn invoke(&self, path: &Path, name: &str) -> anyhow::Result<()> {
        let Some(callable) = self.callable.get() else {
            return Ok(());
        };
        Python::attach(|py| -> PyResult<()> {
            callable.call1(py, (path.display().to_string(), name.to_string()))?;
            Ok(())
        })
        .map_err(|e| anyhow::anyhow!("post_activate callback raised: {e}"))
    }
}

/// MCP workspace handle (github clone-tracker OR local-directory bind).
/// Wraps `mcp_methods::server::Workspace` — atomic root swaps via
/// RwLock, sandbox enforced against the canonicalised configured
/// roots (NOT the mutable active root, which is the bug the
/// pre-0.9.24 Python wrapper had).
#[pyclass(name = "Workspace")]
pub struct PyWorkspace {
    inner: RustWorkspace,
    deferred: Arc<DeferredPyHook>,
}

#[pymethods]
impl PyWorkspace {
    /// Github-flavoured workspace (clone-and-track). Pass the
    /// workspace directory + how many idle days before a clone is
    /// considered stale. The post-activate hook is registered later
    /// via `set_post_activate`.
    #[staticmethod]
    fn open(workspace_dir: &str, stale_after_days: u32) -> PyResult<PyWorkspace> {
        let deferred = Arc::new(DeferredPyHook::default());
        let hook = make_deferred_hook(Arc::clone(&deferred));
        let inner = RustWorkspace::open(PathBuf::from(workspace_dir), stale_after_days, Some(hook))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyWorkspace { inner, deferred })
    }

    /// Local-directory workspace. Binds `root` as the active source
    /// root immediately; `repo_management(update=True)` re-fingerprints
    /// + rebuilds, `set_root_dir(path)` swaps to a sibling under the
    /// configured root.
    #[staticmethod]
    fn open_local(root: &str) -> PyResult<PyWorkspace> {
        let deferred = Arc::new(DeferredPyHook::default());
        let hook = make_deferred_hook(Arc::clone(&deferred));
        let inner = RustWorkspace::open_local(PathBuf::from(root), Some(hook))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyWorkspace { inner, deferred })
    }

    /// Register the post-activate hook. Called after each successful
    /// `repo_management` activate / `set_root_dir`. Signature:
    /// `callback(repo_path: str, repo_name: str) -> None`. Raises if
    /// already set — the hook is one-shot per Workspace instance.
    fn set_post_activate(&self, callable: Py<PyAny>) -> PyResult<()> {
        self.deferred
            .callable
            .set(callable)
            .map_err(|_| PyRuntimeError::new_err("post_activate already registered"))
    }

    /// "github" or "local". Matches the manifest `workspace.kind`
    /// string verbatim so the Python side can mode-gate tool
    /// registration with a single comparison.
    fn kind(&self) -> &'static str {
        self.inner.kind().as_str()
    }

    fn workspace_dir(&self) -> String {
        self.inner.workspace_dir().display().to_string()
    }

    fn active_repo_name(&self) -> Option<String> {
        self.inner.active_repo_name()
    }

    fn active_repo_path(&self) -> Option<String> {
        self.inner
            .active_repo_path()
            .map(|p| p.display().to_string())
    }

    /// The `repo_management` MCP tool dispatcher. All four args are
    /// optional kwargs from the agent — Rust validates the combination
    /// and returns the user-facing string.
    #[pyo3(signature = (name=None, delete=false, update=false, force_rebuild=false))]
    fn repo_management(
        &self,
        py: Python<'_>,
        name: Option<&str>,
        delete: bool,
        update: bool,
        force_rebuild: bool,
    ) -> String {
        py.detach(|| {
            self.inner
                .repo_management(name, delete, update, force_rebuild)
        })
    }

    /// Local-mode lateral root swap. Sandbox check is performed
    /// against the canonicalised configured root by Rust — this is
    /// the call that fixes the 0.9.23 narrows-with-each-swap bug.
    fn set_root_dir(&self, py: Python<'_>, path: &str) -> String {
        py.detach(|| self.inner.set_root_dir(Path::new(path)))
    }

    fn last_built_sha(&self, name: &str) -> Option<String> {
        self.inner.last_built_sha(name)
    }
}

fn make_deferred_hook(deferred: Arc<DeferredPyHook>) -> PostActivateHook {
    Arc::new(move |path, name| deferred.invoke(path, name))
}

// ---------------------------------------------------------------------------
// Watch
// ---------------------------------------------------------------------------

/// Active filesystem watcher. Drop / `stop()` to tear it down.
///
/// The inner Rust handle owns the `notify_debouncer_mini` debouncer;
/// dropping it stops the watcher cleanly. Holding this handle alive
/// is the user's responsibility — Python's GC drives Drop, so simply
/// not retaining the returned object stops the watch.
#[pyclass(name = "WatchHandle")]
pub struct PyWatchHandle {
    // Option so `.stop()` can take ownership and drop the inner handle
    // before the pyclass itself is collected.
    inner: Mutex<Option<mcp_methods::server::WatchHandle>>,
}

#[pymethods]
impl PyWatchHandle {
    /// Explicitly stop watching. Idempotent — second call is a no-op.
    fn stop(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.take();
        }
    }
}

/// Start a recursive debounced filesystem watcher.
///
/// `callback` is invoked on a background thread with a single argument:
/// a list of changed path strings. The callback runs with the GIL
/// acquired automatically — it can call any Python API. Keep work
/// short or push it onto a queue; long-running callbacks block
/// subsequent debounce-window flushes.
#[pyfunction]
#[pyo3(signature = (dir, callback=None, debounce_ms=500))]
fn start_watch(
    dir: &str,
    callback: Option<Py<PyAny>>,
    debounce_ms: u64,
) -> PyResult<PyWatchHandle> {
    let on_change: Option<mcp_methods::server::ChangeHandler> = callback.map(|cb| {
        let cb = Arc::new(cb);
        let handler: mcp_methods::server::ChangeHandler = Arc::new(move |paths: &[PathBuf]| {
            let cb = Arc::clone(&cb);
            let path_strs: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
            // Acquire the GIL on the watcher's background thread to
            // dispatch into Python. Errors are logged to stderr and
            // swallowed — the watcher should keep running even if a
            // single callback invocation raises.
            let _ = Python::attach(|py| -> PyResult<()> {
                cb.call1(py, (path_strs,))?;
                Ok(())
            });
        });
        handler
    });
    let inner = mcp_methods::server::watch_dir(
        Path::new(dir),
        on_change,
        Some(Duration::from_millis(debounce_ms)),
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyWatchHandle {
        inner: Mutex::new(Some(inner)),
    })
}

// ---------------------------------------------------------------------------
// Env file walk-up
// ---------------------------------------------------------------------------

/// Walk upward from `start_path` looking for a `.env` file and load
/// the first one found into the process environment. Returns the
/// loaded file's path (for boot summaries) or None if nothing was
/// found before reaching the filesystem root.
///
/// Existing env vars are NOT overwritten — the file is a default,
/// not an override.
#[pyfunction]
fn load_env_walk(start_path: &str) -> Option<String> {
    mcp_methods::server::env::load_env_walk(Path::new(start_path)).map(|p| p.display().to_string())
}

// ---------------------------------------------------------------------------
// Skills (0.9.40+): wraps mcp-methods 0.3.38's Registry + ResolvedRegistry +
// Skill so kglite can load + iterate skills without pulling the mcp-methods
// PyPI wheel. Mirrors the surface that
// `kglite.mcp_server.skills_loader::_framework_skill_to_local` consumes —
// flat getters on PySkill match what the old `mcp_methods.Skill` exposed.
// ---------------------------------------------------------------------------

/// A loaded skill from the framework registry. Read-only; constructed via
/// `SkillRegistry::skills()`. Mirrors the flat shape the prior
/// `mcp_methods.Skill` pyo3 binding exposed to Python: the `frontmatter`
/// sub-struct is flattened up to direct getters so the consumer
/// (`skills_loader._framework_skill_to_local`) doesn't need to change.
#[pyclass(name = "Skill")]
pub struct PySkill {
    inner: RustSkill,
}

#[pymethods]
impl PySkill {
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    #[getter]
    fn description(&self) -> &str {
        self.inner.description()
    }

    #[getter]
    fn body(&self) -> &str {
        &self.inner.body
    }

    /// `"project"` | `"bundled"` | `"domain_pack:<path>"`. Mirrors the
    /// exact rendering the prior `mcp_methods.Skill.provenance` getter
    /// emitted (verified against `mcp-methods-py/src/lib.rs::from_core`),
    /// so the Python merge logic in `skills_loader` sees byte-identical
    /// strings before vs. after the wheel-dep drop.
    #[getter]
    fn provenance(&self) -> String {
        use mcp_methods::server::SkillProvenance;
        match &self.inner.provenance {
            SkillProvenance::Project => "project".to_string(),
            SkillProvenance::DomainPack(path) => format!("domain_pack:{}", path.display()),
            SkillProvenance::Bundled => "bundled".to_string(),
        }
    }

    #[getter]
    fn auto_inject_hint(&self) -> bool {
        self.inner.frontmatter.auto_inject_hint
    }

    #[getter]
    fn references_tools(&self) -> Vec<String> {
        self.inner.frontmatter.references_tools.clone()
    }

    /// Returns the parsed `applies_when:` block as a dict, or `None` if
    /// the skill's frontmatter has no `applies_when:`. Dict shape matches
    /// what `_framework_skill_to_local` reads:
    /// `{graph_has_node_type: list[str], graph_has_property: {node_type, prop_name},
    ///   tool_registered: str, extension_enabled: str}` — only populated
    /// keys are included.
    #[getter]
    fn applies_when<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(aw) = self.inner.frontmatter.applies_when.as_ref() else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        if let Some(types) = aw.graph_has_node_type.as_ref() {
            d.set_item("graph_has_node_type", types.clone())?;
        }
        if let Some(prop) = aw.graph_has_property.as_ref() {
            let inner = PyDict::new(py);
            inner.set_item("node_type", &prop.node_type)?;
            inner.set_item("prop_name", &prop.prop_name)?;
            d.set_item("graph_has_property", inner)?;
        }
        if let Some(tool) = aw.tool_registered.as_ref() {
            d.set_item("tool_registered", tool)?;
        }
        if let Some(ext) = aw.extension_enabled.as_ref() {
            d.set_item("extension_enabled", ext)?;
        }
        Ok(Some(d))
    }
}

/// Resolved skill registry — output of layering bundled defaults +
/// project layer + operator paths declared in a manifest. Wraps
/// `mcp_methods::server::ResolvedRegistry`. Constructed via the static
/// `from_manifest(...)`, then iterated via `.skills()`.
///
/// The `from_manifest` body delegates to `mcp_methods::server::SkillRegistry::from_manifest`
/// (added in 0.3.38) so kglite carries no orchestration logic —
/// upstream owns layer-step ordering and stays the single source of
/// truth.
#[pyclass(name = "SkillRegistry")]
pub struct PySkillRegistry {
    inner: RustResolvedRegistry,
}

#[pymethods]
impl PySkillRegistry {
    /// Load skills from a manifest path. `include_bundled=True` (the
    /// default) merges the framework's built-in skills first; the
    /// manifest's `skills:` layer wins on name collision per
    /// upstream's documented priority order.
    #[staticmethod]
    #[pyo3(signature = (manifest_path, *, include_bundled = true))]
    fn from_manifest(manifest_path: PathBuf, include_bundled: bool) -> PyResult<Self> {
        let resolved = RustSkillRegistry::from_manifest(&manifest_path, include_bundled)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySkillRegistry { inner: resolved })
    }

    /// List of loaded skills in registry-iteration order. The Python
    /// consumer materialises this eagerly (single boot-time call), so
    /// returning a Vec is fine — no need for an iterator protocol.
    fn skills(&self) -> Vec<PySkill> {
        self.inner
            .skill_names()
            .into_iter()
            .filter_map(|name| self.inner.get(&name).cloned())
            .map(|skill| PySkill { inner: skill })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "_mcp_internal")?;
    m.add_function(wrap_pyfunction!(read_source, &m)?)?;
    m.add_function(wrap_pyfunction!(grep, &m)?)?;
    m.add_function(wrap_pyfunction!(list_source, &m)?)?;
    m.add_function(wrap_pyfunction!(ping, &m)?)?;
    m.add_function(wrap_pyfunction!(git_api, &m)?)?;
    m.add_function(wrap_pyfunction!(has_github_token, &m)?)?;
    m.add_function(wrap_pyfunction!(start_watch, &m)?)?;
    m.add_function(wrap_pyfunction!(load_env_walk, &m)?)?;
    m.add_class::<GithubIssues>()?;
    m.add_class::<PyManifest>()?;
    m.add_class::<PyWorkspace>()?;
    m.add_class::<PyWatchHandle>()?;
    m.add_class::<PySkill>()?;
    m.add_class::<PySkillRegistry>()?;
    parent.add_submodule(&m)?;
    // Register in sys.modules so `import kglite._mcp_internal` works.
    // pyo3's `add_submodule` only sets the attribute; the import system
    // looks up qualified names in sys.modules separately.
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("kglite._mcp_internal", &m)?;
    Ok(())
}
