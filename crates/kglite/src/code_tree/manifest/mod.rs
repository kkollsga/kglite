//! Project manifest readers — pyproject.toml, Cargo.toml.
//!
//! Port of parsers/manifest.py. The Python module only supports these
//! two formats; package.json and pom.xml are listed as future work but
//! not yet implemented upstream.

use crate::code_tree::models::{DependencyInfo, ProjectInfo, SourceRoot};
use std::path::{Path, PathBuf};
use toml::Value;

pub fn read_manifest(project_root: &Path) -> Option<ProjectInfo> {
    if (project_root.join("pyproject.toml")).is_file() {
        return read_pyproject(&project_root.join("pyproject.toml"), project_root).ok();
    }
    if (project_root.join("Cargo.toml")).is_file() {
        return read_cargo(&project_root.join("Cargo.toml"), project_root).ok();
    }
    None
}

/// Read a specific manifest file by path; delegates to the right reader.
pub fn read_manifest_file(manifest_path: &Path, project_root: &Path) -> Option<ProjectInfo> {
    let name = manifest_path.file_name()?.to_str()?;
    match name {
        "pyproject.toml" => read_pyproject(manifest_path, project_root).ok(),
        "Cargo.toml" => read_cargo(manifest_path, project_root).ok(),
        _ => None,
    }
}

fn load_toml(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    toml::from_str::<Value>(&text).map_err(|e| e.to_string())
}

fn read_pyproject(manifest_path: &Path, project_root: &Path) -> Result<ProjectInfo, String> {
    let data = load_toml(manifest_path)?;
    let project = data
        .get("project")
        .cloned()
        .unwrap_or(Value::Table(toml::map::Map::new()));
    let build_sys = data
        .get("build-system")
        .cloned()
        .unwrap_or(Value::Table(toml::map::Map::new()));
    let tool = data
        .get("tool")
        .cloned()
        .unwrap_or(Value::Table(toml::map::Map::new()));

    let name = extract_pyproject_name(&project, &tool, project_root);
    let build_backend = build_sys
        .get("build-backend")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let mut info = ProjectInfo {
        name: name.clone(),
        version: extract_pyproject_version(&project, &tool),
        description: extract_pyproject_description(&project, &tool),
        languages: vec!["python".to_string()],
        authors: extract_authors(project.get("authors")),
        license: extract_license(project.get("license")),
        repository_url: extract_repo_url(project.get("urls")),
        manifest_path: manifest_path
            .strip_prefix(project_root)
            .unwrap_or(manifest_path)
            .to_string_lossy()
            .to_string(),
        source_roots: Vec::new(),
        test_roots: Vec::new(),
        dependencies: extract_pyproject_dependencies(&project),
        build_system: Some(if build_backend.is_empty() {
            "unknown".into()
        } else {
            build_backend.to_string()
        }),
        metadata: Default::default(),
    };

    info.source_roots = discover_python_source_roots(&data, project_root, &name);

    // Maturin → also read Cargo.toml for Rust roots.
    if build_backend.contains("maturin") {
        let cargo_path = project_root.join("Cargo.toml");
        if cargo_path.is_file() {
            if let Ok(rust_info) = read_cargo(&cargo_path, project_root) {
                info.source_roots.extend(rust_info.source_roots);
                info.languages.push("rust".to_string());
                info.dependencies.extend(rust_info.dependencies);
                if info.version.is_none() {
                    info.version = rust_info.version;
                }
            }
        }
    }

    info.test_roots = discover_test_roots(&tool, project_root);

    // Mixed-language safety net: a pyproject that finds a small Python
    // package next to a primary non-Python codebase (e.g. tooling pyproject
    // in a C/C++ repo with a script package alongside `src/`) used to
    // silently skip the rest of the repo because source_roots was non-empty
    // and the whole-repo fallback didn't fire. Supplement with first-level
    // dirs that contain code and aren't already covered.
    if !info.source_roots.is_empty() {
        let supplemental = discover_supplemental_roots(project_root, &info);
        info.source_roots.extend(supplemental);
    }

    Ok(info)
}

/// Project name with PEP 621 `[project].name` taking priority, then
/// `[tool.poetry].name` (pure-poetry projects), then the parent directory
/// as a last resort. Without this, a `[tool.poetry]`-only manifest leaves
/// `name` as the random tmpdir / cwd name and every name-keyed strategy
/// (find_python_package) misfires.
fn extract_pyproject_name(project: &Value, tool: &Value, project_root: &Path) -> String {
    if let Some(s) = project.get("name").and_then(|v| v.as_str()) {
        return s.to_string();
    }
    if let Some(s) = tool
        .get("poetry")
        .and_then(|v| v.get("name"))
        .and_then(|v| v.as_str())
    {
        return s.to_string();
    }
    project_root
        .file_name()
        .and_then(|o| o.to_str())
        .unwrap_or("")
        .to_string()
}

fn extract_pyproject_version(project: &Value, tool: &Value) -> Option<String> {
    if let Some(v) = project.get("version").and_then(|v| v.as_str()) {
        return Some(v.to_string());
    }
    if let Some(dyn_arr) = project.get("dynamic").and_then(|v| v.as_array()) {
        if dyn_arr.iter().any(|v| v.as_str() == Some("version")) {
            return Some("dynamic".to_string());
        }
    }
    tool.get("poetry")
        .and_then(|v| v.get("version"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn extract_pyproject_description(project: &Value, tool: &Value) -> Option<String> {
    if let Some(d) = project.get("description").and_then(|v| v.as_str()) {
        return Some(d.to_string());
    }
    tool.get("poetry")
        .and_then(|v| v.get("description"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn extract_pyproject_dependencies(project: &Value) -> Vec<DependencyInfo> {
    let mut deps: Vec<DependencyInfo> = Vec::new();
    if let Some(arr) = project.get("dependencies").and_then(|v| v.as_array()) {
        for d in arr {
            if let Some(s) = d.as_str() {
                deps.push(parse_dep_string(s));
            }
        }
    }
    if let Some(extras) = project
        .get("optional-dependencies")
        .and_then(|v| v.as_table())
    {
        for (group, group_deps) in extras {
            if let Some(arr) = group_deps.as_array() {
                for d in arr {
                    if let Some(s) = d.as_str() {
                        let mut dep = parse_dep_string(s);
                        dep.is_optional = true;
                        dep.group = Some(group.clone());
                        deps.push(dep);
                    }
                }
            }
        }
    }
    deps
}

/// Run each Python-package discovery strategy and merge into a single
/// dedup-by-canonical-path list. Strategies are independent and each is
/// responsible for one declaration shape (poetry packages, setuptools
/// packages, hatch packages, project-name convention) — adding a new
/// build backend is one new fn, not a new branch in a 150-line god fn.
fn discover_python_source_roots(
    data: &Value,
    project_root: &Path,
    project_name: &str,
) -> Vec<SourceRoot> {
    let mut out: Vec<SourceRoot> = Vec::new();
    out.extend(strategy_poetry_packages(data, project_root));
    out.extend(strategy_setuptools_explicit(data, project_root));
    out.extend(strategy_setuptools_find(data, project_root));
    out.extend(strategy_hatch_packages(data, project_root));
    out.extend(strategy_project_name_convention(project_root, project_name));
    dedup_source_roots(out)
}

/// `[tool.poetry].packages = [{include = "...", from = "..."}, ...]`.
/// `from` defaults to ".". `include` may name a package (resolved as a
/// directory) or a glob like "*.py" (which we ignore — the parser doesn't
/// support file-glob roots, and the whole-repo fallback covers that case).
fn strategy_poetry_packages(data: &Value, project_root: &Path) -> Vec<SourceRoot> {
    let Some(arr) = data
        .get("tool")
        .and_then(|t| t.get("poetry"))
        .and_then(|p| p.get("packages"))
        .and_then(|v| v.as_array())
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in arr {
        let Some(table) = entry.as_table() else {
            continue;
        };
        let Some(include) = table.get("include").and_then(|v| v.as_str()) else {
            continue;
        };
        if include.contains('*') {
            continue;
        }
        let from = table.get("from").and_then(|v| v.as_str()).unwrap_or(".");
        let path = project_root.join(from).join(include);
        if path.is_dir() {
            out.push(SourceRoot {
                path,
                language: Some("python".to_string()),
                is_test: false,
                label: Some("poetry-packages".to_string()),
            });
        }
    }
    out
}

/// `[tool.setuptools].packages = ["pkg_a", "pkg_b.sub"]` — explicit list.
/// Resolves each dotted name via `[tool.setuptools].package_dir` if
/// present, otherwise relative to `project_root`. Skipped if `packages`
/// is a table (handled by strategy_setuptools_find).
fn strategy_setuptools_explicit(data: &Value, project_root: &Path) -> Vec<SourceRoot> {
    let Some(setuptools) = data.get("tool").and_then(|t| t.get("setuptools")) else {
        return Vec::new();
    };
    let Some(arr) = setuptools.get("packages").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let pkg_dir = setuptools
        .get("package_dir")
        .and_then(|v| v.as_table())
        .and_then(|t| t.get(""))
        .and_then(|v| v.as_str())
        .unwrap_or(".");
    let mut out = Vec::new();
    for v in arr {
        let Some(name) = v.as_str() else { continue };
        let rel = name.replace('.', "/");
        let path = project_root.join(pkg_dir).join(&rel);
        if path.is_dir() {
            out.push(SourceRoot {
                path,
                language: Some("python".to_string()),
                is_test: false,
                label: Some("setuptools-packages".to_string()),
            });
        }
    }
    out
}

/// `[tool.setuptools.packages.find].where = ["dir1", "dir2"]` — auto-find
/// packages under each given directory. We add each `where` dir as a
/// source root and let the parser walk it; we don't try to enumerate
/// matching packages ourselves (that's the build backend's job).
fn strategy_setuptools_find(data: &Value, project_root: &Path) -> Vec<SourceRoot> {
    let Some(find_table) = data
        .get("tool")
        .and_then(|t| t.get("setuptools"))
        .and_then(|s| s.get("packages"))
        .and_then(|p| p.get("find"))
    else {
        return Vec::new();
    };
    let where_dirs: Vec<String> = find_table
        .get("where")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_else(|| vec![".".to_string()]);
    let mut out = Vec::new();
    for w in where_dirs {
        let path = project_root.join(&w);
        if path.is_dir() {
            out.push(SourceRoot {
                path,
                language: Some("python".to_string()),
                is_test: false,
                label: Some("setuptools-find".to_string()),
            });
        }
    }
    out
}

/// `[tool.hatch.build.targets.wheel].packages = ["src/x", ...]`. Each
/// entry is a path relative to the project root.
fn strategy_hatch_packages(data: &Value, project_root: &Path) -> Vec<SourceRoot> {
    let Some(arr) = data
        .get("tool")
        .and_then(|t| t.get("hatch"))
        .and_then(|h| h.get("build"))
        .and_then(|b| b.get("targets"))
        .and_then(|t| t.get("wheel"))
        .and_then(|w| w.get("packages"))
        .and_then(|v| v.as_array())
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for v in arr {
        let Some(s) = v.as_str() else { continue };
        let path = project_root.join(s);
        if path.is_dir() {
            out.push(SourceRoot {
                path,
                language: Some("python".to_string()),
                is_test: false,
                label: Some("hatch-packages".to_string()),
            });
        }
    }
    out
}

/// Convention: project name maps to `<name>/` flat layout or
/// `src/<name>/` src-layout. Lowest-priority strategy — used as a fallback
/// when no explicit declaration matched.
fn strategy_project_name_convention(project_root: &Path, project_name: &str) -> Vec<SourceRoot> {
    find_python_package(project_root, project_name)
        .into_iter()
        .map(|path| SourceRoot {
            path,
            language: Some("python".to_string()),
            is_test: false,
            label: Some("python-package".to_string()),
        })
        .collect()
}

/// Drop SourceRoots that resolve to the same canonical path as an earlier
/// entry, preserving first-occurrence order. Strategies are ordered
/// priority-first, so this means the highest-priority label wins on tie.
fn dedup_source_roots(roots: Vec<SourceRoot>) -> Vec<SourceRoot> {
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(roots.len());
    for r in roots {
        let key = r.path.canonicalize().unwrap_or_else(|_| r.path.clone());
        if seen.insert(key) {
            out.push(r);
        }
    }
    out
}

fn discover_test_roots(tool: &Value, project_root: &Path) -> Vec<SourceRoot> {
    let mut out: Vec<SourceRoot> = Vec::new();
    if let Some(testpaths) = tool
        .get("pytest")
        .and_then(|v| v.get("ini_options"))
        .and_then(|v| v.get("testpaths"))
        .and_then(|v| v.as_array())
    {
        for tp in testpaths {
            if let Some(s) = tp.as_str() {
                let test_dir = project_root.join(s);
                if test_dir.is_dir() {
                    out.push(SourceRoot {
                        path: test_dir,
                        language: None,
                        is_test: true,
                        label: Some("pytest".to_string()),
                    });
                }
            }
        }
    }
    if out.is_empty() {
        for candidate in ["tests", "test"] {
            let test_dir = project_root.join(candidate);
            if test_dir.is_dir() {
                out.push(SourceRoot {
                    path: test_dir,
                    language: None,
                    is_test: true,
                    label: Some("test-dir".to_string()),
                });
                break;
            }
        }
    }
    out
}

/// Mixed-language safety net (Case B): when a pyproject finds a small
/// Python package next to a primary non-Python codebase (e.g. tooling
/// pyproject + huge `src/*.c`), supplement source_roots with first-level
/// dirs that contain code in a language NOT already declared by the
/// manifest. The "undeclared language" gate is what keeps this surgical:
/// for a pure-Python repo, a sibling `scratch/` of unrelated `.py` scripts
/// stays out (the manifest is authoritative for declared languages); for a
/// mixed C/Python repo, the C dirs get pulled in.
///
/// Coverage rule: a first-level dir is *covered* if it equals or is an
/// ancestor of any declared (source or test) root. We skip covered dirs
/// to avoid double-walking files. This means a dir like `src/` with an
/// already-declared `src/<name>/` is skipped — files under `src/<other>/`
/// won't be picked up by this safety net, but that's an edge case beyond
/// the bug class this is meant to address.
fn discover_supplemental_roots(project_root: &Path, info: &ProjectInfo) -> Vec<SourceRoot> {
    let declared_paths: Vec<PathBuf> = info
        .source_roots
        .iter()
        .chain(info.test_roots.iter())
        .map(|r| r.path.clone())
        .collect();
    let declared_langs: std::collections::HashSet<String> = info
        .source_roots
        .iter()
        .filter_map(|r| r.language.clone())
        .chain(info.languages.iter().cloned())
        .collect();
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(project_root) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        if is_ignored_dir_name(&name) {
            continue;
        }
        let covered = declared_paths
            .iter()
            .any(|d| d == &path || d.starts_with(&path));
        if covered {
            continue;
        }
        if !directory_contains_undeclared_language(&path, &declared_langs) {
            continue;
        }
        out.push(SourceRoot {
            path,
            language: None,
            is_test: false,
            label: Some(format!("auto:{}", name)),
        });
    }
    out
}

/// First-level directories that should never be auto-supplemented:
/// Names of directories the source-tree walker should never descend
/// into at *any* depth: VCS dirs, virtualenvs, dependency caches,
/// language-bytecode caches, and the universal Rust build output.
/// Hidden-by-prefix (`.git`, `.venv`, `.tox`, `.pytest_cache`,
/// `.mypy_cache`, `.ruff_cache`, `.cargo`, `.idea`, `.vscode`,
/// `.next`, `.nuxt`, `.benchmarks`, `.gradle`) are caught by the
/// leading-dot check.
///
/// Used by both the supplemental-root detector and the parser walk —
/// 0.8.36 had this filter only at the top level, so a project with a
/// `subprojects/.venv/` containing a C extension would attract
/// `subprojects/` as a supplemental root and then index every Python
/// file in the .venv. The filter now applies at every depth.
///
/// `_build` is on the list because the leading-underscore convention
/// is unambiguously "tool-generated build output" (Sphinx, mkdocs,
/// mdBook, etc.) — e.g. `docs/_build/_static/*.js` is Sphinx tooling,
/// not source the user wrote.
///
/// **Not on the list**: `build`, `dist`, `out`. Those names are more
/// ambiguous — `dist/bundle.js` may be the user's webpack output they
/// want flagged as too-large rather than excluded outright (see
/// `tests/test_code_tree_skip.py::test_single_huge_line_skipped`).
/// `max_loc_per_file` handles oversized build artifacts without
/// name-based filtering.
pub(crate) fn is_ignored_dir_name(name: &str) -> bool {
    if name.starts_with('.') {
        return true;
    }
    matches!(
        name,
        "node_modules" | "target" | "_build" | "__pycache__" | "venv" | "env" | "site-packages"
    )
}

/// `WalkDir` filter predicate: skip directories whose name appears in
/// `is_ignored_dir_name`. Files always pass through (filtering files
/// here would prune them entirely; we want to walk them and let the
/// parser layer decide).
pub(crate) fn walk_filter(entry: &walkdir::DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return true;
    }
    let Some(name) = entry.file_name().to_str() else {
        return true;
    };
    !is_ignored_dir_name(name)
}

/// Walk `dir` looking for the first source file whose language is NOT in
/// `declared`. Returns as soon as one is found. Used to keep the safety
/// net cheap on dirs with no parseable code (docs/, assets/) or with only
/// already-covered languages. Skips ignored subdirs (`.venv`, `target`,
/// `node_modules`, …) at any depth — without this filter, a single C
/// extension source inside a venv attracts the parent dir as a
/// supplemental root and floods the graph with site-packages content.
fn directory_contains_undeclared_language(
    dir: &Path,
    declared: &std::collections::HashSet<String>,
) -> bool {
    use walkdir::WalkDir;
    for entry in WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_entry(walk_filter)
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        if let Some(lang) = crate::code_tree::parsers::language_for_path(entry.path()) {
            if !declared.contains(lang) {
                return true;
            }
        }
    }
    false
}

fn read_cargo(manifest_path: &Path, project_root: &Path) -> Result<ProjectInfo, String> {
    let data = load_toml(manifest_path)?;
    let package = data
        .get("package")
        .cloned()
        .unwrap_or(Value::Table(toml::map::Map::new()));

    let fallback_name = project_root
        .file_name()
        .and_then(|o| o.to_str())
        .unwrap_or("")
        .to_string();
    let name = package
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(&fallback_name)
        .to_string();

    let authors: Vec<String> = package
        .get("authors")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let mut info = ProjectInfo {
        name,
        version: package
            .get("version")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        description: package
            .get("description")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        languages: vec!["rust".to_string()],
        authors,
        license: package
            .get("license")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        repository_url: package
            .get("repository")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        manifest_path: manifest_path
            .strip_prefix(project_root)
            .unwrap_or(manifest_path)
            .to_string_lossy()
            .to_string(),
        source_roots: Vec::new(),
        test_roots: Vec::new(),
        dependencies: Vec::new(),
        build_system: Some("cargo".into()),
        metadata: Default::default(),
    };

    let src_dir = project_root.join("src");
    if src_dir.is_dir() {
        info.source_roots.push(SourceRoot {
            path: src_dir,
            language: Some("rust".to_string()),
            is_test: false,
            label: Some("rust-crate".to_string()),
        });
    }

    // Capture `[lib] crate-type` so downstream queries can tell apart a
    // regular `lib` crate (where `pub fn` is a real Rust API export) from a
    // `cdylib` PyO3 crate (where only `#[pyfunction]` / `#[pymethods]`
    // exposure matters and `pub fn` is meaningless to the outside world).
    if let Some(crate_types) = data
        .get("lib")
        .and_then(|v| v.as_table())
        .and_then(|lib| lib.get("crate-type"))
        .and_then(|v| v.as_array())
    {
        let types: Vec<serde_json::Value> = crate_types
            .iter()
            .filter_map(|v| v.as_str().map(|s| serde_json::Value::String(s.to_string())))
            .collect();
        if !types.is_empty() {
            info.metadata
                .insert("crate_type".to_string(), serde_json::Value::Array(types));
        }
    }

    // Workspace members — glob manually.
    if let Some(members) = data
        .get("workspace")
        .and_then(|v| v.get("members"))
        .and_then(|v| v.as_array())
    {
        for member_val in members {
            let Some(member_glob) = member_val.as_str() else {
                continue;
            };
            // Resolve simple globs relative to project_root. Full glob would
            // require the `glob` crate; Python used Path.glob here. Keep it
            // simple: support either a literal path or "dir/*".
            let matches = match member_glob.strip_suffix("/*") {
                Some(parent) => std::fs::read_dir(project_root.join(parent))
                    .ok()
                    .map(|iter| {
                        iter.filter_map(Result::ok)
                            .map(|e| e.path())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                None => vec![project_root.join(member_glob)],
            };
            for member_dir in matches {
                let member_src = member_dir.join("src");
                if member_src.is_dir() {
                    let label = format!(
                        "workspace:{}",
                        member_dir
                            .file_name()
                            .and_then(|o| o.to_str())
                            .unwrap_or(""),
                    );
                    info.source_roots.push(SourceRoot {
                        path: member_src,
                        language: Some("rust".to_string()),
                        is_test: false,
                        label: Some(label),
                    });
                }
            }
        }
    }

    let tests_dir = project_root.join("tests");
    if tests_dir.is_dir() {
        info.test_roots.push(SourceRoot {
            path: tests_dir,
            language: Some("rust".to_string()),
            is_test: true,
            label: Some("rust-integration-tests".to_string()),
        });
    }

    if let Some(deps) = data.get("dependencies").and_then(|v| v.as_table()) {
        for (name, spec) in deps {
            info.dependencies.push(parse_cargo_dep(name, spec, false));
        }
    }
    if let Some(deps) = data.get("dev-dependencies").and_then(|v| v.as_table()) {
        for (name, spec) in deps {
            info.dependencies.push(parse_cargo_dep(name, spec, true));
        }
    }

    Ok(info)
}

// ── Helpers ───────────────────────────────────────────────────────

fn find_python_package(project_root: &Path, project_name: &str) -> Option<PathBuf> {
    let pkg_name = project_name.replace('-', "_");
    let flat = project_root.join(&pkg_name);
    if flat.is_dir() && flat.join("__init__.py").is_file() {
        return Some(flat);
    }
    let src_layout = project_root.join("src").join(&pkg_name);
    if src_layout.is_dir() && src_layout.join("__init__.py").is_file() {
        return Some(src_layout);
    }
    None
}

fn extract_authors(value: Option<&Value>) -> Vec<String> {
    let Some(arr) = value.and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Table(t) => {
                let name = t.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let email = t.get("email").and_then(|v| v.as_str()).unwrap_or("");
                if email.is_empty() {
                    Some(name.to_string())
                } else {
                    Some(format!("{} <{}>", name, email))
                }
            }
            _ => None,
        })
        .collect()
}

fn extract_license(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.clone()),
        Value::Table(t) => t
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| t.get("file").and_then(|v| v.as_str()))
            .map(str::to_string),
        _ => None,
    }
}

fn extract_repo_url(urls: Option<&Value>) -> Option<String> {
    let urls = urls?.as_table()?;
    for key in ["Repository", "repository", "Source", "source", "Homepage"] {
        if let Some(v) = urls.get(key).and_then(|v| v.as_str()) {
            return Some(v.to_string());
        }
    }
    None
}

fn parse_dep_string(dep_str: &str) -> DependencyInfo {
    for (i, ch) in dep_str.char_indices() {
        if matches!(ch, '>' | '=' | '<' | '!' | '~') {
            return DependencyInfo {
                name: dep_str[..i].trim().to_string(),
                version_spec: Some(dep_str[i..].trim().to_string()),
                is_dev: false,
                is_optional: false,
                group: None,
            };
        }
    }
    DependencyInfo {
        name: dep_str.trim().to_string(),
        version_spec: None,
        is_dev: false,
        is_optional: false,
        group: None,
    }
}

fn parse_cargo_dep(name: &str, spec: &Value, is_dev: bool) -> DependencyInfo {
    let version = match spec {
        Value::String(s) => Some(s.clone()),
        Value::Table(t) => t
            .get("version")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        _ => None,
    };
    DependencyInfo {
        name: name.to_string(),
        version_spec: version,
        is_dev,
        is_optional: false,
        group: None,
    }
}
