//! Phase 5 native attribution driver. Run through session_attribution.py.
//! Uses public Session APIs, plain in-memory storage, no WAL/CDC or retries.

use kglite::api::session::{execute_mut, execute_read, CommitOutcome, ExecuteOptions, Session};
use kglite::api::{DirGraph, GraphRead, Value};
use serde_json::{json, Value as Json};
use std::collections::HashMap;
use std::time::Instant;

pub(crate) type Params = HashMap<String, Value>;
pub(crate) type Error = Box<dyn std::error::Error>;

pub(crate) fn number(params: &mut Params, name: &str, value: usize) {
    params.insert(name.into(), Value::Int64(value as i64));
}

fn integer(value: &Value) -> Result<usize, Error> {
    match value {
        Value::Int64(n) => Ok(usize::try_from(*n)?),
        Value::UniqueId(n) => Ok(*n as usize),
        other => Err(format!("expected integer, got {other:?}").into()),
    }
}

pub(crate) fn seed(nodes: usize, edges: usize) -> Result<Session, Error> {
    let mut graph = DirGraph::new();
    let mut params = Params::new();
    number(&mut params, "nodes", nodes);
    number(&mut params, "edges", edges);
    let opts = ExecuteOptions::eager(&params);
    execute_mut(
        &mut graph,
        "UNWIND range(0, $nodes - 1) AS i CREATE (:Person {id:i, title:toString(i), city:'seed'})",
        &opts,
    )?;
    if edges > 0 {
        execute_mut(
            &mut graph,
            "UNWIND range(0, $edges - 1) AS e WITH e % $nodes AS from_id, \
             (e + 1) % $nodes AS to_id MATCH (a:Person {id:from_id}), \
             (b:Person {id:to_id}) CREATE (a)-[:LINK]->(b)",
            &opts,
        )?;
    }
    if graph.graph.node_count() != nodes || graph.graph.edge_count() != edges {
        return Err("seed count mismatch".into());
    }
    Ok(Session::new(graph))
}

pub(crate) fn materialization_sanity() -> Result<Json, Error> {
    let session = seed(8, 0)?;
    let original_version = session.version();
    let mut tx = session.begin();
    let first = tx.working_mut()? as *mut DirGraph;
    let params = Params::new();
    execute_mut(
        tx.working_mut()?,
        "CREATE (:Person {id:8, title:'8', city:'uncommitted'})",
        &ExecuteOptions::eager(&params),
    )?;
    let second = tx.working_mut()?;
    if !std::ptr::eq(first, second) || second.graph.node_count() != 9 {
        return Err("repeated working_mut replaced or lost the working graph".into());
    }
    drop(tx);
    if session.version() != original_version {
        return Err("uncommitted sanity transaction advanced Session version".into());
    }
    oracle(&session, "create", 8, 0, 0, 1)?;
    Ok(json!({"passed":true, "same_working_graph":true, "uncommitted_state_isolated":true}))
}

pub(crate) fn backend(session: &Session) -> String {
    // The observation Arc is dropped before returning, hence before timing.
    let snapshot = session.snapshot();
    format!("{:?}", snapshot.graph)
}

pub(crate) fn oracle(
    session: &Session,
    kind: &str,
    nodes: usize,
    edges: usize,
    count: usize,
    fixed: usize,
) -> Result<Json, Error> {
    let snapshot = session.snapshot();
    let expected_count = nodes + if kind == "create" { count } else { 0 };
    if snapshot.graph.node_count() != expected_count || snapshot.graph.edge_count() != edges {
        return Err("final physical node/edge counts differ".into());
    }
    let params = Params::new();
    let rows = execute_read(
        &snapshot,
        "MATCH (p:Person) RETURN p.id, p.title, p.city ORDER BY p.id",
        &ExecuteOptions::eager(&params),
    )?;
    if rows.result.rows.len() != expected_count {
        return Err("final query count differs".into());
    }
    for (id, row) in rows.result.rows.iter().enumerate() {
        if row.len() != 3 || integer(&row[0])? != id {
            return Err(format!("missing/duplicate ID at sorted row {id}").into());
        }
        let city = if kind == "create" && id >= nodes {
            format!("city-{}", id - nodes)
        } else if kind == "set" && id < fixed && id < count {
            let last = id + (count - 1 - id) / fixed * fixed;
            format!("city-{last}")
        } else {
            "seed".into()
        };
        if row[1] != Value::String(id.to_string()) || row[2] != Value::String(city) {
            return Err(format!("incorrect title/city at ID {id}: {row:?}").into());
        }
    }
    Ok(
        json!({"passed":true, "nodes":expected_count, "edges":edges, "all_ids_titles_cities_checked":true}),
    )
}

pub(crate) fn summary(values: impl Iterator<Item = u64>) -> Json {
    let mut values: Vec<u64> = values.collect();
    values.sort_unstable();
    let sum: u128 = values.iter().map(|n| *n as u128).sum();
    let percentile =
        |p: usize| values[((values.len() * p).div_ceil(100) - 1).min(values.len() - 1)];
    json!({"sum_ns":sum.to_string(), "mean_ns":sum as f64 / values.len() as f64,
        "p95_ns":percentile(95), "p99_ns":percentile(99), "max_ns":values[values.len()-1]})
}

fn run(
    kind: &str,
    nodes: usize,
    edges: usize,
    count: usize,
    window: usize,
    fixed: usize,
) -> Result<Json, Error> {
    let session = seed(nodes, edges)?;
    let initial_version = session.version();
    let initial_backend = backend(&session);
    let query = if kind == "create" {
        "CREATE (:Person {id:$id, title:$title, city:$city})"
    } else {
        "MATCH (p:Person {id:$id}) SET p.city = $city"
    };
    let mut windows = Vec::new();
    for start in (0..count).step_by(window) {
        let end = (start + window).min(count);
        // Payload construction and backend observation are outside all clocks.
        let payloads: Vec<Params> = (start..end)
            .map(|i| {
                let id = if kind == "create" {
                    nodes + i
                } else {
                    i % fixed
                };
                let mut params = Params::new();
                number(&mut params, "id", id);
                params.insert("title".into(), Value::String(id.to_string()));
                params.insert("city".into(), Value::String(format!("city-{i}")));
                params
            })
            .collect();
        let before = backend(&session);
        let mut samples: Vec<[u64; 6]> = Vec::with_capacity(end - start);
        let wall = Instant::now();
        for params in &payloads {
            let opts = ExecuteOptions::eager(params);
            let t0 = Instant::now();
            let mut tx = session.begin();
            let t1 = Instant::now();
            let working = tx.working_mut()?;
            let t2 = Instant::now();
            let output = execute_mut(working, query, &opts)?;
            let t3 = Instant::now();
            drop(output);
            let t4 = Instant::now();
            let outcome = session.commit(tx, true);
            let t5 = Instant::now();
            match outcome {
                CommitOutcome::Committed { .. } => {}
                other => return Err(format!("unexpected serial commit outcome: {other:?}").into()),
            }
            let t6 = Instant::now();
            let ns = |a: Instant, b: Instant| b.duration_since(a).as_nanos() as u64;
            samples.push([
                ns(t0, t1),
                ns(t1, t2),
                ns(t2, t3),
                ns(t3, t4),
                ns(t4, t5),
                ns(t0, t6),
            ]);
        }
        let wall_ns = wall.elapsed().as_nanos().to_string();
        let after = backend(&session);
        let phases: serde_json::Map<String, Json> = [
            "begin",
            "first_working_mut",
            "execute",
            "result_drop",
            "commit",
            "full_operation",
        ]
        .iter()
        .enumerate()
        .map(|(i, name)| ((*name).into(), summary(samples.iter().map(|s| s[i]))))
        .collect();
        windows.push(
            json!({"start_commit":start, "end_commit":end, "wall_ns":wall_ns,
            "backend_before":before, "backend_after":after, "phases":phases}),
        );
    }
    if session.version() != initial_version + count as u64 {
        return Err("serial commit version accounting failed".into());
    }
    let verified = oracle(&session, kind, nodes, edges, count, fixed)?;
    Ok(
        json!({"kind":kind, "commits":count, "initial_backend":initial_backend,
        "initial_version":initial_version, "final_version":session.version(), "windows":windows, "oracle":verified}),
    )
}

fn main() -> Result<(), Error> {
    if cfg!(debug_assertions) {
        return Err("measurement driver refuses a debug artifact; build release".into());
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err(
            "usage: session_attribution NODES EDGES COMMITS WINDOW FIXED_NODES REPEATS".into(),
        );
    }
    let n: Vec<usize> = args.iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
    let (nodes, edges, count, window, fixed, repeats) = (n[0], n[1], n[2], n[3], n[4], n[5]);
    if nodes == 0
        || nodes > 100_000
        || edges > 300_000
        || count == 0
        || count > 100_000
        || window == 0
        || window > count
        || count.div_ceil(window) > 1000
        || fixed == 0
        || fixed > nodes
        || repeats == 0
        || repeats > 10
    {
        return Err("invalid or unbounded driver parameters".into());
    }
    let sanity = materialization_sanity()?;
    let mut runs = Vec::new();
    for repeat in 0..repeats {
        // Reverse arm order on alternating repeats to expose ordering effects.
        let kinds = if repeat % 2 == 0 {
            ["create", "set"]
        } else {
            ["set", "create"]
        };
        for kind in kinds {
            let mut result = run(kind, nodes, edges, count, window, fixed)?;
            result["repeat"] = json!(repeat);
            runs.push(result);
        }
    }
    println!(
        "{}",
        serde_json::to_string(&json!({"schema":1, "profile":"release", "nodes":nodes,
        "edges":edges, "window":window, "fixed_nodes":fixed, "repeats":repeats,
        "timing_scope":"parameter preparation and observations excluded; output destruction included; no warmup commits discarded",
        "materialization_sanity":sanity, "runs":runs}))?
    );
    Ok(())
}
