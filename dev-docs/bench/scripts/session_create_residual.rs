//! Bounded first/second-write attribution, run with session_create_residual.py.
use kglite::api::session::{execute_mut, execute_read, ExecuteOptions, Session};
use kglite::api::{DirGraph, GraphRead, Value};
use serde_json::{json, Value as Json};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
type Params = HashMap<String, Value>;
type Error = Box<dyn std::error::Error>;

fn payload(id: usize) -> Params {
    HashMap::from([
        ("id".into(), Value::Int64(id as i64)),
        ("title".into(), Value::String(id.to_string())),
        ("city".into(), Value::String("new".into())),
    ])
}

fn seed(nodes: usize, width: usize) -> Result<DirGraph, Error> {
    let mut graph = DirGraph::new();
    let params = HashMap::from([("nodes".into(), Value::Int64(nodes as i64))]);
    let extra = (1..width).map(|i| format!(", p{i}:i")).collect::<String>();
    for label in ["Person", "Other"] {
        execute_mut(&mut graph, &format!("UNWIND range(0, $nodes - 1) AS i CREATE (:{label} {{id:i, title:toString(i), city:'seed'{extra}}})"), &ExecuteOptions::eager(&params))?;
    }
    if graph.graph.is_forked() || graph.graph.node_count() != nodes * 2 {
        return Err("seed must be uniquely owned Memory with two populated types".into());
    }
    Ok(graph)
}

fn observation(graph: &DirGraph, label: &str) -> Json {
    graph
        .column_store(label)
        .map_or(json!({"present":false}), |s| {
            json!({"present":true,"store_owners":Arc::strong_count(s),
            "property_columns":s.column_count(),"heap_bytes":s.heap_bytes(),
            "rows":s.row_count(),"backend_forked":graph.graph.is_forked()})
        })
}

fn verify(
    graph: &DirGraph,
    nodes: usize,
    width: usize,
    appended: bool,
    empty: bool,
) -> Result<(), Error> {
    let params = Params::new();
    for label in ["Person", "Other", "Empty"] {
        if label == "Empty" && (!appended || !empty) {
            continue;
        }
        let expected = if label == "Person" {
            nodes + usize::from(appended) * 2
        } else if label == "Other" {
            nodes + usize::from(appended)
        } else {
            1
        };
        let extras = if label == "Empty" {
            String::new()
        } else {
            (1..width).map(|i| format!(", n.p{i}")).collect::<String>()
        };
        let rows = execute_read(
            graph,
            &format!("MATCH (n:{label}) RETURN n.id,n.title,n.city{extras} ORDER BY n.id"),
            &ExecuteOptions::eager(&params),
        )?
        .result
        .rows;
        if rows.len() != expected {
            return Err(format!("{label} row count mismatch").into());
        }
        for (i, row) in rows.iter().enumerate() {
            let id = if label == "Empty" { nodes } else { i };
            let new = label == "Empty" || id >= nodes;
            if row[0] != Value::Int64(id as i64)
                || row[1] != Value::String(id.to_string())
                || row[2] != Value::String(if new { "new" } else { "seed" }.into())
            {
                return Err(format!("{label} identity/body mismatch at {id}: {row:?}").into());
            }
            for v in &row[3..] {
                if *v
                    != if new {
                        Value::Null
                    } else {
                        Value::Int64(id as i64)
                    }
                {
                    return Err(format!("{label} omitted property mismatch at {id}").into());
                }
            }
        }
    }
    if graph.graph.node_count() != nodes * 2 + if appended { 3 + usize::from(empty) } else { 0 }
        || graph.graph.edge_count() != 0
    {
        return Err("physical counts mismatch".into());
    }
    Ok(())
}

fn sample(
    graph: &mut DirGraph,
    nodes: usize,
    empty: bool,
) -> Result<(Vec<[u64; 2]>, Vec<Json>), Error> {
    let payloads = [
        payload(nodes),
        payload(nodes + 1),
        payload(nodes),
        payload(nodes),
    ];
    let mut samples = Vec::new();
    let mut observations = Vec::new();
    for (label, params) in ["Person", "Person", "Other", "Empty"]
        .into_iter()
        .zip(&payloads)
    {
        if label == "Empty" && !empty {
            continue;
        }
        let query = format!("CREATE (:{label} {{id:$id, title:$title, city:$city}})");
        let opts = ExecuteOptions::eager(params);
        observations.push(observation(graph, label));
        let clock = Instant::now();
        let result = execute_mut(graph, &query, &opts)?;
        let execute_ns = clock.elapsed().as_nanos() as u64;
        let clock = Instant::now();
        drop(result);
        let drop_ns = clock.elapsed().as_nanos() as u64;
        samples.push([execute_ns, drop_ns]);
    }
    Ok((samples, observations))
}

fn summary(samples: &[u64]) -> Json {
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    let sum: u128 = ordered.iter().map(|n| *n as u128).sum();
    json!({"mean_ns":sum as f64/ordered.len() as f64,"sum_ns":sum.to_string(),"min_ns":ordered[0],
        "median_ns":ordered[ordered.len()/2],"p95_ns":ordered[(ordered.len()*95).div_ceil(100)-1],"max_ns":ordered[ordered.len()-1]})
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
    let nodes: usize = args[1].parse()?;
    let width: usize = args[2].parse()?;
    let events: usize = args[3].parse()?;
    let warmup: usize = args[4].parse()?;
    let reverse: usize = args[5].parse()?;
    let empty = args[6] == "width";
    let mut records = Vec::new();
    let modes = if reverse == 0 {
        ["session", "owned"]
    } else {
        ["owned", "session"]
    };
    for mode in modes {
        let mut samples = Vec::new();
        let mut observed = None;
        for event in 0..warmup + events {
            let graph = seed(nodes, width)?;
            let (values, obs) = if mode == "session" {
                let session = Session::new(graph);
                let mut tx = session.begin();
                let working = tx.working_mut()?;
                if !working.graph.is_forked() {
                    return Err("working graph did not fork".into());
                }
                let result = sample(working, nodes, empty)?;
                verify(working, nodes, width, true, empty)?;
                verify(&session.snapshot(), nodes, width, false, empty)?;
                drop(tx);
                result
            } else {
                let mut graph = graph;
                let result = sample(&mut graph, nodes, empty)?;
                verify(&graph, nodes, width, true, empty)?;
                result
            };
            if event >= warmup {
                samples.push(values);
                observed.get_or_insert(obs);
            }
        }
        let stages = [
            "first_person",
            "second_person",
            "first_other",
            "first_empty",
        ];
        let summaries:Vec<Json>=stages[..if empty {4} else {3}].iter().enumerate().map(|(i,name)| {
            json!({"stage":name,"execute":summary(&samples.iter().map(|s|s[i][0]).collect::<Vec<_>>()),
                "result_drop":summary(&samples.iter().map(|s|s[i][1]).collect::<Vec<_>>())})
        }).collect();
        records.push(json!({"mode":mode,"nodes_per_type":nodes,"width":width,"events":events,
            "warmup":warmup,"observations":observed,"stages":summaries,"samples_ns":samples,
            "oracle":{"passed":true,"all_seed_and_new_values_checked":true,"omitted_properties_null":true,"session_snapshot_unchanged":mode=="session"}}));
    }
    println!("{}", json!({"records":records}));
    Ok(())
}
