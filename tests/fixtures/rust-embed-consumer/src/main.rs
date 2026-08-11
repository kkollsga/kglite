use std::collections::HashMap;
use std::sync::Arc;

use kglite::api::io::{load_file, save_graph};
use kglite::api::session::{execute_mut, execute_read, ExecuteOptions};
use kglite::api::{DirGraph, Embedder, GraphRead, GraphWrite, KnowledgeGraph, Value};

struct DeterministicEmbedder;

impl Embedder for DeterministicEmbedder {
    fn dimension(&self) -> usize {
        2
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts
            .iter()
            .map(|text| vec![text.len() as f32, text.bytes().map(f32::from).sum()])
            .collect())
    }

    fn model_id(&self) -> Option<String> {
        Some("fixture/deterministic-v1".into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = DirGraph::new();

    execute_mut(&mut graph, "CREATE (:Person {id: 1, name: 'Alice'})", &opts)?;

    let result = execute_read(
        &graph,
        "MATCH (p:Person {id: 1}) RETURN p.name AS name",
        &opts,
    )?;
    assert_eq!(
        result.result.rows,
        vec![vec![Value::String("Alice".into())]]
    );

    // The 0.15.9 migration routes, exercised from *outside* the crate — the
    // only vantage point that can see an unexported symbol. The changelog
    // advertised `GraphWrite::set_node_property` as `NodeData::set_property`'s
    // replacement while `GraphWrite` was still `pub(crate)`, and every
    // internal build resolved it fine (that shipped in 0.15.9; no external
    // consumer could compile the migration until the trait was re-exported).
    let idx = graph
        .graph
        .node_indices()
        .next()
        .expect("the CREATE above made a node");
    assert!(
        graph.set_node_property(idx, "age", Value::Int64(30)),
        "the one-call string-keyed route must land on an existing node"
    );
    let city_key = graph.interner.try_get_or_intern("city")?;
    graph
        .graph
        .set_node_property(idx, city_key, Value::String("Oslo".into()));
    let view = graph.node_view(idx).expect("node has a view");
    assert_eq!(
        view.get_property("age").map(|v| v.into_owned()),
        Some(Value::Int64(30)),
        "a property written via DirGraph::set_node_property must read back via NodeView"
    );
    assert_eq!(
        view.get_property("city").map(|v| v.into_owned()),
        Some(Value::String("Oslo".into())),
        "a property written via GraphWrite with a registered key must read back via NodeView"
    );
    let keys = view.property_keys(&graph.interner);
    for wanted in ["age", "city"] {
        assert!(
            keys.contains(&wanted),
            "written key {wanted:?} must resolve during enumeration (an \
             unregistered key panics in property_keys and vanishes on save); got {keys:?}"
        );
    }
    drop(view);

    let path = std::env::temp_dir().join(format!(
        "kglite-rust-embed-consumer-{}.kgl",
        std::process::id()
    ));
    let path_string = path.to_string_lossy().into_owned();
    let mut graph = Arc::new(graph);
    save_graph(&mut graph, &path_string).map_err(std::io::Error::other)?;

    let loaded = load_file(&path_string)?;
    let mut handle = KnowledgeGraph::from_arc(loaded);
    handle.set_embedder_native(Arc::new(DeterministicEmbedder));
    let embedder = handle.embedder().expect("embedder should stay bound");
    assert_eq!(
        embedder.model_id().as_deref(),
        Some("fixture/deterministic-v1")
    );
    assert_eq!(embedder.embed(&["Alice".into()])?, vec![vec![5.0, 478.0]]);

    let reloaded = execute_read(
        handle.dir(),
        "MATCH (p:Person {id: 1}) RETURN p.name AS name",
        &ExecuteOptions::eager(&params),
    )?;
    assert_eq!(reloaded.result.rows, result.result.rows);

    std::fs::remove_file(path)?;
    Ok(())
}
