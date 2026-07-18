use std::collections::HashMap;
use std::sync::Arc;

use kglite::api::io::{load_file, save_graph};
use kglite::api::session::{execute_mut, execute_read, ExecuteOptions};
use kglite::api::{DirGraph, Embedder, KnowledgeGraph, Value};

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
