//! Demonstrates the snapshot/working CoW transaction model from
//! `kglite::api::session` — including OCC conflict handling.
//!
//! Two transactions race to mutate the same graph: A wins, B's
//! commit detects the conflict and is rejected. This is the
//! pattern bindings (Bolt server, etc.) use to surface
//! "Transaction conflict — retry the transaction" errors.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p kglite --example embedded_session
//! ```

use kglite::api::session::{CommitOutcome, ExecuteOptions, Session};
use kglite::api::DirGraph;
use std::collections::HashMap;
use std::sync::Arc;

fn opts<'a>(params: &'a HashMap<String, kglite::api::Value>) -> ExecuteOptions<'a> {
    ExecuteOptions {
        params,
        deadline: None,
        max_rows: None,
        lazy_eligible: false,
        disabled_passes: None,
        embedder: None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = Arc::new(Session::new(DirGraph::new()));
    let params: HashMap<String, kglite::api::Value> = HashMap::new();

    // ── Tx A: begin, create a node ────────────────────────────────
    let mut tx_a = session.begin();
    let working_a = tx_a.working_mut()?;
    kglite::api::session::execute_mut(
        working_a,
        "CREATE (:Person {id: 1, name: 'Alice'})",
        &opts(&params),
    )?;
    println!("Tx A: created Person(id=1, name='Alice') in working copy");

    // ── Tx B: begin (sees pre-A snapshot), create a different node ─
    let mut tx_b = session.begin();
    let working_b = tx_b.working_mut()?;
    kglite::api::session::execute_mut(
        working_b,
        "CREATE (:Person {id: 2, name: 'Bob'})",
        &opts(&params),
    )?;
    println!("Tx B: created Person(id=2, name='Bob') in working copy");

    // ── Commit A: succeeds, graph version bumps to 1 ──────────────
    let outcome_a = session.commit(tx_a, /* check_occ = */ true);
    match outcome_a {
        CommitOutcome::Committed { new_version } => {
            println!("\n✓ Tx A committed → version {}", new_version);
        }
        other => panic!("expected Committed, got {:?}", other),
    }

    // ── Commit B: OCC detects stale snapshot (base 0, current 1) ──
    let outcome_b = session.commit(tx_b, /* check_occ = */ true);
    match outcome_b {
        CommitOutcome::ConflictDetected {
            current_version,
            base_version,
        } => {
            println!(
                "✗ Tx B rejected (OCC): base_version={} but current_version={}",
                base_version, current_version
            );
            println!("  Client retry pattern: re-run the transaction against a fresh snapshot.");
        }
        other => panic!("expected ConflictDetected, got {:?}", other),
    }

    // ── Verify final state: only Alice landed ─────────────────────
    let snap = session.snapshot();
    let outcome = kglite::api::session::execute_read(
        &snap,
        "MATCH (p:Person) RETURN p.name AS name ORDER BY p.id",
        &opts(&params),
    )?;
    println!("\nFinal graph:");
    for row in &outcome.result.rows {
        if let Some(kglite::api::Value::String(s)) = row.first() {
            println!("  - {}", s);
        }
    }

    Ok(())
}
