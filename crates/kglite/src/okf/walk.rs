//! Bundle directory walk: enumerate concept `.md` files.
//!
//! Reserved filenames (`index.md`, `log.md`) are not concepts — they're skipped
//! here (the directory hierarchy is captured separately as structural `CONTAINS`
//! edges in the builder). Hidden directories (`.git`, `.obsidian`, …) are
//! pruned, mirroring `code_tree`'s `walk_filter`.

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

/// Filenames that carry structural meaning and are never concepts.
const RESERVED: &[&str] = &["index.md", "log.md"];

fn is_ignored_dir(name: &str) -> bool {
    // Hidden dirs (.git, .obsidian, .venv, …) plus the usual build noise.
    name.starts_with('.')
        || matches!(
            name,
            "node_modules" | "target" | "__pycache__" | "venv" | "env" | "site-packages"
        )
}

/// Walk `root`, returning every non-reserved `.md` concept file. Errors only on
/// an unreadable root.
pub fn discover(root: &Path) -> Result<Vec<DiscoveredFile>, String> {
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
    let walker = WalkDir::new(root).into_iter().filter_entry(|e| {
        // Never prune the root itself (depth 0) — the bundle directory may
        // legitimately be hidden (e.g. a `.tmpXXXX` temp dir, or a path under
        // `.claude/`). Only prune *descendant* hidden / build dirs.
        if e.depth() == 0 || !e.file_type().is_dir() {
            return true;
        }
        e.file_name()
            .to_str()
            .map(|n| !is_ignored_dir(n))
            .unwrap_or(true)
    });

    for entry in walker.filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let name = match entry.file_name().to_str() {
            Some(n) => n,
            None => continue,
        };
        if !name.ends_with(".md") || RESERVED.contains(&name) {
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
        out.push(DiscoveredFile {
            rel_path,
            abs_path: entry.path().to_path_buf(),
        });
    }
    // Deterministic order (parallelism happens at parse time, but a stable file
    // list keeps id-collision resolution and tests reproducible).
    out.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Ok(out)
}
