//! Bundle directory walk: enumerate concept `.md` files and per-directory
//! `index.md` files.
//!
//! Reserved filenames (`index.md`, `log.md`) are not concepts. `index.md` is
//! captured per directory (it describes the directory — it enriches the `Folder`
//! node in the builder); `log.md` is skipped. Hidden directories (`.git`,
//! `.obsidian`, …) are pruned, mirroring codingest's `walk_filter`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// A discovered concept file, with its bundle-relative (forward-slashed) path.
#[derive(Debug, Clone)]
pub struct DiscoveredFile {
    /// Bundle-relative path, forward-slashed (e.g. `tables/users.md`).
    pub rel_path: String,
    /// Absolute path on disk (for reading the file).
    pub abs_path: PathBuf,
}

/// Result of a bundle walk: concept files + each directory's `index.md`.
#[derive(Debug, Clone, Default)]
pub struct WalkResult {
    pub concepts: Vec<DiscoveredFile>,
    /// Bundle-relative directory path (`""` = root) → that directory's
    /// `index.md` absolute path.
    pub index_files: HashMap<String, PathBuf>,
}

/// `log.md` carries no structure we use; `index.md` is handled specially.
const SKIPPED: &[&str] = &["log.md"];

fn is_ignored_dir(name: &str) -> bool {
    // Hidden dirs (.git, .obsidian, .venv, …) plus the usual build noise.
    name.starts_with('.')
        || matches!(
            name,
            "node_modules" | "target" | "__pycache__" | "venv" | "env" | "site-packages"
        )
}

/// True if a directory at bundle-relative path `rel` (basename `name`) matches a
/// caller `skip_dirs` entry: bare name → match at any depth; entry with `/` →
/// anchored relative-path prefix (the dir and its subtree).
fn matches_skip(rel: &str, name: &str, skip_dirs: &[String]) -> bool {
    skip_dirs.iter().any(|raw| {
        let entry = raw.trim_matches('/');
        if entry.is_empty() {
            false
        } else if entry.contains('/') {
            rel == entry || rel.starts_with(&format!("{entry}/"))
        } else {
            name == entry
        }
    })
}

/// Walk `root`, returning concept `.md` files plus per-directory `index.md`
/// files. `skip_dirs` prunes matching directories (and their subtrees). Errors
/// only on an unreadable root.
pub fn discover(root: &Path, skip_dirs: &[String]) -> Result<WalkResult, String> {
    if !root.exists() {
        return Err(format!(
            "OKF bundle path does not exist: {}",
            root.display()
        ));
    }
    if !root.is_dir() {
        return Err(format!(
            "OKF bundle path is not a directory: {}",
            root.display()
        ));
    }

    let mut out = Vec::new();
    let mut index_files: HashMap<String, PathBuf> = HashMap::new();
    let walker = WalkDir::new(root).into_iter().filter_entry(|e| {
        // Never prune the root itself (depth 0) — the bundle directory may
        // legitimately be hidden (e.g. a `.tmpXXXX` temp dir, or a path under
        // `.claude/`). Only prune *descendant* hidden / build / skip dirs.
        if e.depth() == 0 || !e.file_type().is_dir() {
            return true;
        }
        let Some(name) = e.file_name().to_str() else {
            return true;
        };
        if is_ignored_dir(name) {
            return false;
        }
        if !skip_dirs.is_empty() {
            let rel = e
                .path()
                .strip_prefix(root)
                .ok()
                .map(|r| {
                    r.components()
                        .filter_map(|c| c.as_os_str().to_str())
                        .collect::<Vec<_>>()
                        .join("/")
                })
                .unwrap_or_default();
            if matches_skip(&rel, name, skip_dirs) {
                return false;
            }
        }
        true
    });

    for entry in walker.filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let name = match entry.file_name().to_str() {
            Some(n) => n,
            None => continue,
        };
        if !name.ends_with(".md") || SKIPPED.contains(&name) {
            continue;
        }
        let rel = match entry.path().strip_prefix(root) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let rel_path = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect::<Vec<_>>()
            .join("/");
        if name == "index.md" {
            // Record per directory (bundle-relative dir path; "" = root).
            let dir = rel_path
                .rfind('/')
                .map(|i| rel_path[..i].to_string())
                .unwrap_or_default();
            index_files.insert(dir, entry.path().to_path_buf());
            continue;
        }
        out.push(DiscoveredFile {
            rel_path,
            abs_path: entry.path().to_path_buf(),
        });
    }
    // Deterministic order (parallelism happens at parse time, but a stable file
    // list keeps id-collision resolution and tests reproducible).
    out.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Ok(WalkResult {
        concepts: out,
        index_files,
    })
}
