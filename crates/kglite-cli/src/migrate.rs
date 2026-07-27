//! `kglite migrate` — apply ordered Cypher migrations and advance the graph's
//! user-schema version stamp.
//!
//! **The whole mechanism.** A migration is a `.cypher` file named
//! `<version>_<name>.cypher` in one directory:
//!
//! ```text
//! migrations/
//!   001_add_email.cypher
//!   002_backfill_country.cypher
//!   003_split_display_name.cypher
//! ```
//!
//! The graph carries a single integer — `DirGraph::user_schema_version`,
//! persisted in `.kgl` metadata — recording how far it has been migrated.
//! `migrate` applies every migration numbered above that stamp, in ascending
//! order, advancing the stamp as it goes. There is no ledger table, no
//! checksum, no lock row: the ordered filenames are the history and the stamp
//! is the bookmark. That is deliberately the smallest thing that can be
//! correct, and it is the honest shape for an embedded single-file database.
//!
//! **Guarantees.**
//!
//! - *Idempotent.* A second run applies nothing and exits 0.
//! - *Ordered.* Migrations are applied in strictly ascending version order.
//! - *All-or-nothing at the file level.* Everything runs against an in-memory
//!   copy; the `.kgl` is written once, at the end, only if every statement
//!   succeeded. A failure part-way leaves the file on disk exactly as it was.
//! - *Refuses an out-of-sync stamp.* See [`plan`].
//!
//! **What it deliberately does not do.** It does not roll back an individual
//! migration (write your own inverse migration), and it does not detect a
//! migration edited after being applied. Both would need a ledger; neither is
//! worth the machinery here.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use kglite::api::io::{save_graph, GraphWriterLease};
use kglite::api::{make_dir_graph_mut, DirGraph};

use crate::exec::{self, QueryOptions};

/// The file extension a migration script must carry.
const MIGRATION_EXTENSION: &str = "cypher";

/// One discovered migration script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Migration {
    pub(crate) version: u32,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
}

/// Parse `007_add_email.cypher` into `(7, "add_email")`.
///
/// A `.cypher` file that does not lead with `<digits>_` is an error rather
/// than something to skip: silently ignoring a file that looks exactly like a
/// migration is how a migration gets lost.
fn parse_migration_name(file_name: &str) -> Result<(u32, String)> {
    let stem = file_name
        .strip_suffix(&format!(".{MIGRATION_EXTENSION}"))
        .with_context(|| format!("{file_name} is not a .{MIGRATION_EXTENSION} file"))?;
    let (digits, name) = stem.split_once('_').with_context(|| {
        format!("migration {file_name} must be named <version>_<name>.{MIGRATION_EXTENSION}")
    })?;
    let version = digits.parse::<u32>().with_context(|| {
        format!(
            "migration {file_name} must start with an integer version, got {digits:?} \
             (expected <version>_<name>.{MIGRATION_EXTENSION})"
        )
    })?;
    if version == 0 {
        anyhow::bail!(
            "migration {file_name} declares version 0, which is reserved for the \
             unversioned baseline — migrations start at 1"
        );
    }
    Ok((version, name.to_string()))
}

/// Discover the migrations in `dir`, sorted ascending by version.
///
/// Non-`.cypher` files are ignored outright so a `README.md` or a stray editor
/// backup can live alongside the scripts. Two migrations sharing a version is
/// an error: their relative order would be arbitrary.
pub(crate) fn discover(dir: &Path) -> Result<Vec<Migration>> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read migration directory {}", dir.display()))?;
    let mut by_version: BTreeMap<u32, Migration> = BTreeMap::new();
    for entry in entries {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !file_name.ends_with(&format!(".{MIGRATION_EXTENSION}")) {
            continue;
        }
        let (version, name) = parse_migration_name(file_name)?;
        if let Some(existing) = by_version.get(&version) {
            anyhow::bail!(
                "migrations {} and {} both declare version {version} — \
                 renumber one so the order is unambiguous",
                existing.name,
                name
            );
        }
        by_version.insert(
            version,
            Migration {
                version,
                name,
                path,
            },
        );
    }
    Ok(by_version.into_values().collect())
}

/// Decide which migrations to apply to a graph currently at `stamp`.
///
/// Returns the pending migrations in ascending order. Everything at or below
/// `stamp` is already applied and is skipped — that is what makes a re-run a
/// no-op.
///
/// **The out-of-sync refusal.** A non-zero `stamp` must correspond to a
/// migration that is actually present. If a graph says "I am at version 5" and
/// the directory has no version-5 migration, we are not looking at the history
/// that produced this graph — the likely causes are the wrong migration
/// directory, a deleted migration, or a hand-edited stamp. Blindly applying
/// "everything above 5" in that state either skips work or repeats it, so we
/// refuse and say so instead of guessing.
pub(crate) fn plan(stamp: u32, migrations: &[Migration]) -> Result<Vec<Migration>> {
    if stamp != 0 && !migrations.iter().any(|m| m.version == stamp) {
        let known: Vec<String> = migrations.iter().map(|m| m.version.to_string()).collect();
        anyhow::bail!(
            "the graph is stamped at user-schema version {stamp}, but no migration here \
             declares that version (found: {}). This migration set did not produce this \
             graph — check that you passed the right directory, and that no applied \
             migration has been deleted.",
            if known.is_empty() {
                "none".to_string()
            } else {
                known.join(", ")
            }
        );
    }
    Ok(migrations
        .iter()
        .filter(|m| m.version > stamp)
        .cloned()
        .collect())
}

/// Apply one migration's statements to the in-memory graph and advance the
/// stamp. The stamp moves only after every statement in the file succeeded, so
/// a failed migration never leaves a half-claimed version behind.
fn apply(graph: &mut Arc<DirGraph>, migration: &Migration) -> Result<usize> {
    let script = std::fs::read_to_string(&migration.path)
        .with_context(|| format!("cannot read migration {}", migration.path.display()))?;
    let statements = crate::repl::split_statements(&script);
    let params = std::collections::HashMap::new();
    let options = QueryOptions::default();
    for (index, statement) in statements.iter().enumerate() {
        exec::execute(graph, statement, &params, &options).with_context(|| {
            format!(
                "migration {:03}_{} failed at statement {} of {}: {statement}",
                migration.version,
                migration.name,
                index + 1,
                statements.len()
            )
        })?;
    }
    make_dir_graph_mut(graph).user_schema_version = migration.version;
    Ok(statements.len())
}

/// Run the pending migrations in `dir` against the graph at `graph_path`.
///
/// With `dry_run`, the plan is printed and nothing is executed or written.
pub(crate) fn run(graph_path: &Path, dir: &Path, dry_run: bool) -> Result<()> {
    let migrations = discover(dir)?;
    // Read the current stamp before taking a write lease, so a dry run never
    // blocks a concurrent writer.
    let stamp = crate::load_graph(graph_path)?.user_schema_version;
    let pending = plan(stamp, &migrations)?;

    let already = migrations.len() - pending.len();
    println!(
        "{} at user-schema version {stamp}: {} already applied, {} pending",
        graph_path.display(),
        already,
        pending.len()
    );
    if pending.is_empty() {
        println!("nothing to do");
        return Ok(());
    }
    for migration in &pending {
        println!("  {:03}_{}", migration.version, migration.name);
    }
    if dry_run {
        println!("dry run — nothing applied");
        return Ok(());
    }

    let _lease = GraphWriterLease::acquire(graph_path, crate::WRITE_LOCK_TIMEOUT)?;
    // Re-load under the lease: the stamp may have moved between the read above
    // and acquiring the lease, and applying a plan built against a stale stamp
    // would double-apply.
    let mut graph = crate::load_graph(graph_path)?;
    let locked_stamp = graph.user_schema_version;
    if locked_stamp != stamp {
        anyhow::bail!(
            "{} moved from user-schema version {stamp} to {locked_stamp} while this run was \
             starting — re-run to migrate from the new version",
            graph_path.display()
        );
    }

    for migration in &pending {
        let statements = apply(&mut graph, migration)?;
        println!(
            "applied {:03}_{} ({statements} statement(s)) — now at version {}",
            migration.version, migration.name, migration.version
        );
    }

    let path = graph_path.to_string_lossy().to_string();
    save_graph(&mut graph, &path).map_err(|e| anyhow::anyhow!("failed to save {path}: {e}"))?;
    println!(
        "saved {path} at user-schema version {}",
        graph.user_schema_version
    );
    Ok(())
}

/// Print the graph's current user-schema version.
pub(crate) fn print_version(graph_path: &Path) -> Result<()> {
    println!("{}", crate::load_graph(graph_path)?.user_schema_version);
    Ok(())
}

/// Stamp the graph at `version` without running anything.
///
/// This is the "adopt migrations on an existing graph" operation: a database
/// that already has the shape migrations 1..=N would have produced gets
/// stamped at N so those migrations are not replayed over live data. It is
/// deliberately a separate, explicit verb — it asserts a fact about the data
/// rather than changing it.
pub(crate) fn set_version(graph_path: &Path, version: u32) -> Result<()> {
    let _lease = GraphWriterLease::acquire(graph_path, crate::WRITE_LOCK_TIMEOUT)?;
    let mut graph = crate::load_graph(graph_path)?;
    let previous = graph.user_schema_version;
    make_dir_graph_mut(&mut graph).user_schema_version = version;
    let path = graph_path.to_string_lossy().to_string();
    save_graph(&mut graph, &path).map_err(|e| anyhow::anyhow!("failed to save {path}: {e}"))?;
    println!("{path}: user-schema version {previous} -> {version}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn migration(version: u32, name: &str) -> Migration {
        Migration {
            version,
            name: name.to_string(),
            path: PathBuf::from(format!("{version:03}_{name}.cypher")),
        }
    }

    #[test]
    fn names_parse_into_version_and_slug() {
        assert_eq!(
            parse_migration_name("001_add_email.cypher").unwrap(),
            (1, "add_email".to_string())
        );
        assert_eq!(
            parse_migration_name("42_split_name.cypher").unwrap(),
            (42, "split_name".to_string())
        );
    }

    #[test]
    fn malformed_migration_names_are_errors_not_silent_skips() {
        // No version prefix at all.
        assert!(parse_migration_name("add_email.cypher").is_err());
        // Non-numeric version.
        assert!(parse_migration_name("v1_add_email.cypher").is_err());
        // Version 0 is the unversioned-baseline sentinel.
        let error = parse_migration_name("000_baseline.cypher").unwrap_err();
        assert!(error.to_string().contains("reserved"), "{error}");
    }

    #[test]
    fn discover_sorts_by_version_and_ignores_non_migrations() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "002_second.cypher",
            "001_first.cypher",
            "010_tenth.cypher",
            "README.md",
        ] {
            std::fs::write(dir.path().join(name), "MATCH (n) RETURN n").unwrap();
        }
        let found = discover(dir.path()).unwrap();
        let versions: Vec<u32> = found.iter().map(|m| m.version).collect();
        assert_eq!(versions, vec![1, 2, 10], "numeric order, not lexicographic");
    }

    #[test]
    fn duplicate_versions_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("001_a.cypher"), "MATCH (n) RETURN n").unwrap();
        std::fs::write(dir.path().join("1_b.cypher"), "MATCH (n) RETURN n").unwrap();
        let error = discover(dir.path()).unwrap_err();
        assert!(
            error.to_string().contains("both declare version 1"),
            "{error}"
        );
    }

    #[test]
    fn plan_skips_applied_and_keeps_ascending_order() {
        let all = vec![migration(1, "a"), migration(2, "b"), migration(3, "c")];

        // Fresh graph: everything is pending, in order.
        let versions: Vec<u32> = plan(0, &all).unwrap().iter().map(|m| m.version).collect();
        assert_eq!(versions, vec![1, 2, 3]);

        // Part-way: only what is above the stamp.
        let versions: Vec<u32> = plan(2, &all).unwrap().iter().map(|m| m.version).collect();
        assert_eq!(versions, vec![3]);

        // Fully migrated: idempotent no-op.
        assert!(plan(3, &all).unwrap().is_empty());
    }

    #[test]
    fn plan_refuses_a_stamp_this_migration_set_cannot_explain() {
        let all = vec![migration(1, "a"), migration(2, "b")];
        // Stamped above anything present — a deleted migration, the wrong
        // directory, or a hand-edited stamp. Refuse rather than guess.
        let error = plan(5, &all).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("stamped at user-schema version 5"),
            "{error}"
        );
        assert!(error.to_string().contains("found: 1, 2"), "{error}");

        // A gap in the numbering is fine as long as the stamp is a real version.
        let sparse = vec![migration(1, "a"), migration(5, "e"), migration(9, "i")];
        let versions: Vec<u32> = plan(5, &sparse)
            .unwrap()
            .iter()
            .map(|m| m.version)
            .collect();
        assert_eq!(versions, vec![9]);
    }

    #[test]
    fn plan_on_an_empty_directory_accepts_only_an_unstamped_graph() {
        assert!(plan(0, &[]).unwrap().is_empty());
        let error = plan(1, &[]).unwrap_err();
        assert!(error.to_string().contains("found: none"), "{error}");
    }
}
