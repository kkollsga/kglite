//! Adaptive Phase 5 ownership/history/holder/durability controls.
// The shared baseline's standalone entry is unused in this control binary.
#[allow(dead_code)]
#[path = "session_attribution.rs"]
mod common;
use common::{backend, materialization_sanity, number, oracle, seed, summary, Error, Params};
use kglite::api::durable::{wal_path, DurabilityLevel};
use kglite::api::io::{load_file, save_graph_with, GraphWriterLease};
use kglite::api::session::{execute_mut, CommitOutcome, ExecuteOptions, Session};
use kglite::api::{DirGraph, Value};
use serde_json::{json, Value as Json};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn ns(start: Instant) -> u64 {
    start.elapsed().as_nanos() as u64
}

fn seed_owned(nodes: usize, edges: usize, bulk: usize) -> Result<DirGraph, Error> {
    let session = seed(nodes, edges)?;
    let graph = session.snapshot();
    drop(session);
    let mut graph = Arc::try_unwrap(graph).map_err(|_| "seed unexpectedly shared")?;
    if bulk > 0 {
        let mut params = Params::new();
        number(&mut params, "nodes", nodes);
        number(&mut params, "bulk", bulk);
        execute_mut(&mut graph,
            "UNWIND range(0, $bulk - 1) AS i CREATE (:Person {id:$nodes+i, title:toString($nodes+i), city:'city-'+toString(i)})",
            &ExecuteOptions::eager(&params))?;
    }
    Ok(graph)
}

fn verify_holder(
    holder: &Arc<DirGraph>,
    nodes: usize,
    edges: usize,
    count: usize,
) -> Result<(), Error> {
    let view = Session::from_arc(Arc::clone(holder));
    oracle(&view, "create", nodes, edges, count, 1)?;
    Ok(())
}

fn run(
    mode: &str,
    nodes: usize,
    edges: usize,
    count: usize,
    window: usize,
    bulk: usize,
    directory: &Path,
    repeat: usize,
) -> Result<Json, Error> {
    let graph = seed_owned(nodes, edges, bulk)?;
    let path = directory.join(format!("{mode}-{count}-{repeat}.kgl"));
    let path_str = path.to_str().ok_or("invalid scratch path")?;
    let mut lease = None;
    let mut owned = None;
    let session = if mode == "owned" {
        owned = Some(graph);
        None
    } else if mode == "normal" {
        std::fs::create_dir_all(directory)?;
        lease = Some(GraphWriterLease::acquire(&path, Duration::ZERO)?);
        let mut graph = Arc::new(graph);
        save_graph_with(&mut graph, path_str, false)?;
        Some(Session::open_durable(
            graph,
            path_str,
            DurabilityLevel::Normal,
        )?)
    } else {
        Some(Session::new(graph))
    };
    let initial_version = session.as_ref().map(Session::version);
    let initial_backend = session
        .as_ref()
        .map(backend)
        .unwrap_or_else(|| format!("{:?}", owned.as_ref().unwrap().graph));
    let mut holder = None;
    let mut events = Vec::new();
    if mode == "held" || mode == "drop-half" {
        let clock = Instant::now();
        holder = Some(session.as_ref().unwrap().snapshot());
        events.push(json!({"event":"initial_holder_acquire", "ns":ns(clock)}));
    }
    let mut windows = Vec::new();
    for start in (0..count).step_by(window) {
        let end = (start + window).min(count);
        let payloads: Vec<Params> = (start..end)
            .map(|i| {
                let mut params = Params::new();
                let id = nodes + bulk + i;
                number(&mut params, "id", id);
                params.insert("title".into(), Value::String(id.to_string()));
                params.insert("city".into(), Value::String(format!("city-{}", bulk + i)));
                params
            })
            .collect();
        let before = session
            .as_ref()
            .map(backend)
            .unwrap_or_else(|| format!("{:?}", owned.as_ref().unwrap().graph));
        let wal_before = if mode == "normal" {
            Some(std::fs::metadata(wal_path(&path))?.len())
        } else {
            None
        };
        let mut samples: Vec<[u64; 8]> = Vec::with_capacity(end - start);
        let mut wall_ns = 0u128;
        for (i, params) in (start..end).zip(&payloads) {
            if mode == "drop-half" && i == count / 2 {
                verify_holder(holder.as_ref().unwrap(), nodes, edges, bulk)?;
                let clock = Instant::now();
                drop(holder.take());
                events.push(
                    json!({"event":"midpoint_holder_drop", "before_commit":i,"ns":ns(clock)}),
                );
            }
            // Snapshot acquisition/drop are timed; exact snapshot verification
            // after publication is excluded from the event cost.
            let opts = ExecuteOptions::eager(params);
            let clock = Instant::now();
            let mut durations = [0u64; 8];
            if let Some(graph) = owned.as_mut() {
                let t = Instant::now();
                let output = execute_mut(
                    graph,
                    "CREATE (:Person {id:$id, title:$title, city:$city})",
                    &opts,
                )?;
                durations[2] = ns(t);
                let t = Instant::now();
                drop(output);
                durations[3] = ns(t);
            } else {
                let session = session.as_ref().unwrap();
                let t = Instant::now();
                let fresh = if mode == "fresh-holder" {
                    Some(session.snapshot())
                } else {
                    None
                };
                durations[6] = ns(t);
                let t = Instant::now();
                let mut tx = session.begin();
                durations[0] = ns(t);
                let t = Instant::now();
                let working = tx.working_mut()?;
                durations[1] = ns(t);
                let t = Instant::now();
                let output = execute_mut(
                    working,
                    "CREATE (:Person {id:$id, title:$title, city:$city})",
                    &opts,
                )?;
                durations[2] = ns(t);
                let t = Instant::now();
                drop(output);
                durations[3] = ns(t);
                let t = Instant::now();
                let outcome = session.commit(tx, true);
                durations[4] = ns(t);
                if !matches!(outcome, CommitOutcome::Committed { .. }) {
                    return Err(format!("unexpected outcome {outcome:?}").into());
                }
                // Verify this exact held snapshot after publication without charging
                // a whole read query to the transaction/holder lifetime clock.
                durations[5] = ns(clock);
                if let Some(fresh) = fresh {
                    verify_holder(&fresh, nodes, edges, bulk + i)?;
                    let t = Instant::now();
                    drop(fresh);
                    durations[7] = ns(t);
                    durations[5] += durations[7];
                }
            }
            if mode == "owned" {
                durations[5] = ns(clock);
            }
            wall_ns += durations[5] as u128;
            samples.push(durations);
        }
        let after = session
            .as_ref()
            .map(backend)
            .unwrap_or_else(|| format!("{:?}", owned.as_ref().unwrap().graph));
        let names = [
            "begin",
            "first_working_mut",
            "execute",
            "result_drop",
            "commit",
            "full_operation",
            "holder_acquire",
            "holder_drop",
        ];
        let phases: serde_json::Map<String, Json> = names
            .iter()
            .enumerate()
            .map(|(i, n)| ((*n).into(), summary(samples.iter().map(|s| s[i]))))
            .collect();
        let wal_after = if mode == "normal" {
            Some(std::fs::metadata(wal_path(&path))?.len())
        } else {
            None
        };
        windows.push(json!({"start_commit":start,"end_commit":end,"timed_events_ns":wall_ns.to_string(),"backend_before":before,"backend_after":after,"wal_before":wal_before,"wal_after":wal_after,"phases":phases}));
    }
    if let Some(holder) = holder.take() {
        verify_holder(&holder, nodes, edges, bulk)?;
        let clock = Instant::now();
        drop(holder);
        events.push(json!({"event":"final_holder_drop","ns":ns(clock)}));
    }
    let session = session.unwrap_or_else(|| Session::new(owned.take().unwrap()));
    if let Some(version) = initial_version {
        if session.version() != version + count as u64 {
            return Err("version mismatch".into());
        }
    }
    let verified = oracle(&session, "create", nodes, edges, bulk + count, 1)?;
    let final_version = session.version();
    if mode == "normal" {
        let clock = Instant::now();
        session.sync()?;
        events.push(json!({"event":"final_sync","ns":ns(clock)}));
    }
    let clock = Instant::now();
    drop(session);
    events.push(json!({"event":"session_drop","ns":ns(clock)}));
    let replay = if mode == "normal" {
        let clock = Instant::now();
        let recovered =
            Session::open_durable(load_file(path_str)?, path_str, DurabilityLevel::Normal)?;
        let replay_ns = ns(clock);
        let result = oracle(&recovered, "create", nodes, edges, bulk + count, 1)?;
        drop(recovered);
        json!({"cold_reopen_ns":replay_ns,"oracle":result})
    } else {
        Json::Null
    };
    drop(lease);
    Ok(
        json!({"kind":mode,"repeat":repeat,"commits":count,"bulk_prefill":bulk,"initial_backend":initial_backend,"initial_version":initial_version,"final_version":final_version,"windows":windows,"events":events,"oracle":verified,"replay":replay}),
    )
}

fn main() -> Result<(), Error> {
    if cfg!(debug_assertions) {
        return Err("measurement requires release".into());
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 9 {
        return Err("expected N E COUNT WINDOW FIXED REPEATS MODE BULK DIRECTORY".into());
    }
    let values: Vec<usize> = args[..6]
        .iter()
        .map(|v| v.parse())
        .collect::<Result<_, _>>()?;
    let (nodes, edges, count, window, repeats) =
        (values[0], values[1], values[2], values[3], values[5]);
    let bulk = args[7].parse::<usize>()?;
    if nodes == 0
        || nodes > 100000
        || count == 0
        || count > 100000
        || window == 0
        || window > count
        || repeats == 0
        || repeats > 10
        || bulk > 100000
    {
        return Err("unbounded controls".into());
    }
    let sanity = materialization_sanity()?;
    let mut runs = Vec::new();
    for repeat in 0..repeats {
        runs.push(run(
            &args[6],
            nodes,
            edges,
            count,
            window,
            bulk,
            Path::new(&args[8]),
            repeat,
        )?);
    }
    println!(
        "{}",
        serde_json::to_string(
            &json!({"schema":1,"profile":"release","materialization_sanity":sanity,"runs":runs,"timing_scope":"CREATE only; full event includes returned result destruction and holder acquire/drop; holder verification and seed excluded; final sync/drop and cold replay separate"})
        )?
    );
    Ok(())
}
