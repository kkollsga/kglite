//! Crash-shaped durable recovery for relationship embedding stores.
//!
//! Every case here starts from a **saved `.kgl` checkpoint**, reopens it with
//! [`Session::open_durable`], writes, and then abandons the session without a
//! checkpoint — so the only thing that can produce the reopened state is WAL
//! replay through `durability::finish_recovered_open`. That entry point is what
//! the in-process seams in `edge_embedding_wal_lifecycle_tests` cannot reach:
//! they wrap a live `DirGraph` through `recording::wrap_for_durability`, while
//! a recovered open wraps the backend directly and has to turn base capture on
//! for itself. A test that creates its store before the checkpoint, or never
//! takes one, exercises neither.
//!
//! Each case asserts the recovered **vectors, dimension and provenance**, not a
//! store count: a store that comes back with the right name and no vectors is
//! exactly the silent data loss under test.

use crate::datatypes::Value;
use crate::graph::edge_embeddings::edge_store_key;
use crate::graph::embedder::Embedder;
use crate::graph::io::file::{load_file, save_graph};
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};
use crate::graph::session::{CommitOutcome, Session};
use crate::graph::storage::GraphRead;
use crate::graph::wal::{recover, wal_path, DurabilityLevel, MutationOp, WalFrame};
use crate::graph::DirGraph;
use std::collections::HashMap;
use std::sync::Arc;

const SEED: &str = "CREATE (a:Doc {id: 1, summary: 'sa'}), (b:Doc {id: 2, summary: 'sb'}) \
     CREATE (a)-[:CLAIMS {text: 'alpha', k: 1}]->(b)";

/// Two parallel members of one group, so a member delete has something to
/// delete and the patch's prior-ordinal mapping is exercised.
const SEED_PARALLEL: &str =
    "CREATE (a:Doc {id: 1, summary: 'sa'}), (b:Doc {id: 2, summary: 'sb'}) \
     CREATE (a)-[:CLAIMS {text: 'alpha', k: 1}]->(b) \
     CREATE (a)-[:CLAIMS {text: 'beta', k: 2}]->(b)";

fn set_vector(k: i64, vector: &str) -> String {
    format!(
        "MATCH ()-[r:CLAIMS]->() WHERE r.k = {k} WITH collect(r) AS rs \
         CALL db.relationship_embeddings.set({{type: 'CLAIMS', text_column: 'text', \
         entries: [{{relationship: rs[0], vector: {vector}}}]}}) \
         YIELD stored RETURN stored"
    )
}

/// A deterministic two-dimensional embedder: the vector is the text's byte sum
/// and its length, so an assertion can name the exact expected vector.
struct StubEmbedder;

impl Embedder for StubEmbedder {
    fn dimension(&self) -> usize {
        2
    }
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts
            .iter()
            .map(|text| {
                vec![
                    text.bytes().map(|byte| byte as f32).sum::<f32>(),
                    text.len() as f32,
                ]
            })
            .collect())
    }
    fn model_id(&self) -> Option<String> {
        Some("stub/edge".into())
    }
}

fn open(path: &str) -> Session {
    try_open(path).unwrap_or_else(|error| panic!("open_durable({path}) failed: {error}"))
}

fn try_open(path: &str) -> Result<Session, String> {
    let graph = if std::path::Path::new(path).exists() {
        load_file(path).map_err(|error| error.to_string())?
    } else {
        Arc::new(DirGraph::new())
    };
    Session::open_durable(graph, path, DurabilityLevel::Full)
}

fn run(session: &Session, query: &str) {
    run_with(session, query, None);
}

fn run_embedding(session: &Session, query: &str) {
    run_with(session, query, Some(Arc::new(StubEmbedder)));
}

fn run_with(session: &Session, query: &str, embedder: Option<Arc<dyn Embedder>>) {
    let params = HashMap::new();
    let mut options = ExecuteOptions::eager(&params);
    options.embedder = embedder;
    let mut tx = session.begin();
    execute_mut(
        tx.working_mut()
            .expect("a durable session must be writable"),
        query,
        &options,
    )
    .unwrap_or_else(|error| panic!("query failed: {query}: {error}"));
    match session.commit(tx, true) {
        CommitOutcome::Committed { .. } => {}
        other => panic!("commit refused for {query}: {other:?}"),
    }
}

/// One store's recovered state, keyed by the stable `k` property rather than by
/// slot: replay rebuilds the graph, so an `EdgeIndex` is not comparable across
/// the crash.
#[derive(Debug, PartialEq)]
struct StoreState {
    dimension: usize,
    metric: Option<String>,
    model_id: Option<String>,
    /// `(k, vector, text_hash)` for every live member of the type, sorted by
    /// `k`. A member with no vector is `None`, which is a different outcome
    /// from a member that is missing entirely.
    cells: Vec<(i64, Option<Vec<f32>>, Option<u64>)>,
}

fn store_state(session: &Session, text_property: &str) -> Option<StoreState> {
    let snapshot = session.snapshot();
    let store = snapshot
        .edge_embeddings
        .get(&edge_store_key("CLAIMS", text_property))?;
    let mut cells: Vec<_> = snapshot
        .graph
        .edge_indices()
        .filter_map(|edge| {
            let weight = snapshot.graph.edge_weight(edge)?;
            if weight.connection_type_str(&snapshot.interner) != "CLAIMS" {
                return None;
            }
            let k = match weight.get_property("k") {
                Some(Value::Int64(k)) => *k,
                other => panic!("fixture: every CLAIMS edge carries an integer k, got {other:?}"),
            };
            Some((
                k,
                store.get(edge).map(<[f32]>::to_vec),
                store.text_hash(edge),
            ))
        })
        .collect();
    cells.sort_by_key(|(k, _, _)| *k);
    Some(StoreState {
        dimension: store.dimension(),
        metric: store.metric().map(str::to_string),
        model_id: store.model_id().map(str::to_string),
        cells,
    })
}

/// Relationship property values by `k`, so a property write's recovery is
/// asserted alongside the vectors it travels with.
fn edge_property(session: &Session, name: &str) -> Vec<(i64, Option<Value>)> {
    let snapshot = session.snapshot();
    let mut out: Vec<_> = snapshot
        .graph
        .edge_indices()
        .filter_map(|edge| {
            let weight = snapshot.graph.edge_weight(edge)?;
            if weight.connection_type_str(&snapshot.interner) != "CLAIMS" {
                return None;
            }
            let k = match weight.get_property("k") {
                Some(Value::Int64(k)) => *k,
                other => panic!("fixture: every CLAIMS edge carries an integer k, got {other:?}"),
            };
            Some((k, weight.get_property(name).cloned()))
        })
        .collect();
    out.sort_by_key(|(k, _)| *k);
    out
}

fn frames(path: &str) -> Vec<WalFrame> {
    recover(&wal_path(std::path::Path::new(path))).expect("the log must be readable after a crash")
}

fn ops(path: &str) -> Vec<MutationOp> {
    frames(path)
        .into_iter()
        .flat_map(|frame| frame.ops)
        .collect()
}

struct Fixture {
    _dir: tempfile::TempDir,
    path: String,
}

impl Fixture {
    /// A checkpoint holding `seed` and, when `store` is set, a `text`-store
    /// vector for member `k = 1`.
    fn new(seed: &str, store: bool) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("app.kgl").to_string_lossy().into_owned();
        {
            let mut graph = Arc::new(DirGraph::new());
            let params = HashMap::new();
            let options = ExecuteOptions::eager(&params);
            let dir_graph = crate::graph::handle::make_dir_graph_mut(&mut graph);
            execute_mut(dir_graph, seed, &options).expect("seed");
            save_graph(&mut graph, &path).expect("checkpoint");
        }
        if store {
            let session = open(&path);
            run(&session, &set_vector(1, "[1.0, 0.0]"));
            session.save(&path, true).expect("checkpoint with store");
        }
        Self { _dir: dir, path }
    }

    fn path(&self) -> &str {
        &self.path
    }
}

// ── the matrix ───────────────────────────────────────────────────────

/// D1/D2/D3 together: the checkpoint has no store at all, so replay's base is
/// an empty store set while capture's base was taken over the final one.
#[test]
fn a_first_store_created_by_set_survives_a_crash() {
    let fixture = Fixture::new(SEED, false);
    {
        let session = open(fixture.path());
        run(&session, &set_vector(1, "[1.0, 0.0]"));
    }
    let recovered = try_open(fixture.path()).expect("the log must replay");
    assert_eq!(
        store_state(&recovered, "text"),
        Some(StoreState {
            dimension: 2,
            // `db.relationship_embeddings.set` with no `metric:` leaves the store's
            // metric unset (scoring then defaults to cosine), so `None` here is
            // the written value, not a lost one.
            metric: None,
            model_id: None,
            cells: vec![(1, Some(vec![1.0, 0.0]), None)],
        }),
        "a store created inside the durable window must come back with its vectors"
    );
}

/// The generated route: `embed` stamps a model id, so this also proves
/// provenance survives — a store that returns with `model_id: None` would
/// silently defeat the later model-swap guard.
#[test]
fn a_first_store_created_by_embed_survives_a_crash() {
    let fixture = Fixture::new(SEED, false);
    {
        let session = open(fixture.path());
        run_embedding(
            &session,
            "MATCH ()-[r:CLAIMS]->() WITH collect(r) AS rs \
             CALL db.relationship_embeddings.embed({type: 'CLAIMS', text_column: 'text', \
             relationships: rs, mode: 'all'}) YIELD embedded RETURN embedded",
        );
    }
    let recovered = try_open(fixture.path()).expect("the log must replay");
    let state = store_state(&recovered, "text").expect("the generated store must be recovered");
    assert_eq!(state.dimension, 2);
    assert_eq!(state.model_id.as_deref(), Some("stub/edge"));
    let alpha: f32 = "alpha".bytes().map(|byte| byte as f32).sum();
    assert_eq!(state.cells.len(), 1);
    assert_eq!(state.cells[0].0, 1);
    assert_eq!(state.cells[0].1.as_deref(), Some(&[alpha, 5.0][..]));
    assert!(
        state.cells[0].2.is_some(),
        "a generated vector carries its source-text hash, which is what the \
         refresh path compares against"
    );
}

/// D2 alone: the checkpoint already has `text`, the commit adds `note`, and
/// capture digests its base over both.
#[test]
fn a_second_store_on_a_type_that_already_has_one_survives_a_crash() {
    let fixture = Fixture::new(SEED, true);
    {
        let session = open(fixture.path());
        run(
            &session,
            "MATCH ()-[r:CLAIMS]->() WITH collect(r) AS rs \
             CALL db.relationship_embeddings.set({type: 'CLAIMS', text_column: 'k', \
             entries: [{relationship: rs[0], vector: [0.5, 0.5]}]}) YIELD stored RETURN stored",
        );
    }
    let recovered = try_open(fixture.path()).expect("the log must replay");
    assert_eq!(
        store_state(&recovered, "text").map(|state| state.cells),
        Some(vec![(1, Some(vec![1.0, 0.0]), None)]),
        "the pre-existing store must be untouched"
    );
    assert_eq!(
        store_state(&recovered, "k").map(|state| state.cells),
        Some(vec![(1, Some(vec![0.5, 0.5]), None)]),
        "the store added inside the durable window must come back with its vectors"
    );
}

/// The delta scheme's reason to exist. With base capture off on a recovered
/// open, this shape passes anyway — by emitting a full-vector frame — so the
/// assertion is on the *frame*, not only on the reopen.
#[test]
fn a_property_only_write_emits_a_patch_frame_and_recovers() {
    let fixture = Fixture::new(SEED, true);
    {
        let session = open(fixture.path());
        run(&session, "MATCH ()-[r:CLAIMS]->() SET r.note = 'edited'");
    }
    let emitted = ops(fixture.path());
    assert!(
        emitted
            .iter()
            .any(|op| matches!(op, MutationOp::PatchEdgeGroupEmbeddings { .. })),
        "a property-only write over a checkpointed store must log a compact patch, \
         not a full re-send of every vector: {emitted:#?}"
    );
    assert!(
        !emitted
            .iter()
            .any(|op| matches!(op, MutationOp::ReplaceEdgeGroupEmbeddings { .. })),
        "{emitted:#?}"
    );
    let recovered = try_open(fixture.path()).expect("the log must replay");
    assert_eq!(
        store_state(&recovered, "text").map(|state| state.cells),
        Some(vec![(1, Some(vec![1.0, 0.0]), None)])
    );
    assert_eq!(
        edge_property(&recovered, "note"),
        vec![(1, Some(Value::String("edited".into())))]
    );
}

#[test]
fn a_vector_update_over_a_checkpointed_store_survives_a_crash() {
    let fixture = Fixture::new(SEED, true);
    {
        let session = open(fixture.path());
        run(&session, &set_vector(1, "[0.0, 1.0]"));
    }
    let recovered = try_open(fixture.path()).expect("the log must replay");
    assert_eq!(
        store_state(&recovered, "text").map(|state| state.cells),
        Some(vec![(1, Some(vec![0.0, 1.0]), None)]),
        "the newer vector must win; the checkpoint's [1,0] would mean the frame was lost"
    );
}

/// Two frames: a member delete (which forbids the patch path for that group)
/// followed by a property write (which takes it). The second frame's base must
/// be the *post-delete* group, not the checkpoint's.
#[test]
fn deleting_one_parallel_member_then_writing_a_property_survives_a_crash() {
    let fixture = Fixture::new(SEED_PARALLEL, false);
    {
        let session = open(fixture.path());
        run(&session, &set_vector(1, "[1.0, 0.0]"));
        run(&session, &set_vector(2, "[0.0, 1.0]"));
        session.save(fixture.path(), true).expect("checkpoint");
        run(&session, "MATCH ()-[r:CLAIMS]->() WHERE r.k = 2 DELETE r");
        run(&session, "MATCH ()-[r:CLAIMS]->() SET r.note = 'edited'");
    }
    let recovered = try_open(fixture.path()).expect("the log must replay");
    assert_eq!(
        store_state(&recovered, "text").map(|state| state.cells),
        Some(vec![(1, Some(vec![1.0, 0.0]), None)]),
        "the surviving member keeps its own vector; inheriting the deleted member's \
         would be the vector-identity confusion the replay refusal exists to prevent"
    );
    assert_eq!(
        edge_property(&recovered, "note"),
        vec![(1, Some(Value::String("edited".into())))]
    );
}

#[test]
fn dropping_a_store_inside_the_durable_window_survives_a_crash() {
    let fixture = Fixture::new(SEED, true);
    {
        let session = open(fixture.path());
        run(
            &session,
            "CALL db.relationship_embeddings.drop({type: 'CLAIMS', text_column: 'text'}) \
             YIELD dropped RETURN dropped",
        );
    }
    let recovered = try_open(fixture.path()).expect("the log must replay");
    assert_eq!(
        store_state(&recovered, "text"),
        None,
        "a dropped store must stay dropped; replay resurrecting it from the \
         checkpoint is data the user deleted coming back"
    );
}

/// Two frames again, in the other order: the group grows, then the new member
/// is embedded. The embedding frame's base is the grown group.
#[test]
fn creating_a_relationship_then_embedding_it_survives_a_crash() {
    let fixture = Fixture::new(SEED, true);
    {
        let session = open(fixture.path());
        run(
            &session,
            "MATCH (a:Doc {id: 1}), (b:Doc {id: 2}) \
             CREATE (a)-[:CLAIMS {text: 'gamma', k: 2}]->(b)",
        );
        run(&session, &set_vector(2, "[0.25, 0.75]"));
    }
    let recovered = try_open(fixture.path()).expect("the log must replay");
    assert_eq!(
        store_state(&recovered, "text").map(|state| state.cells),
        Some(vec![
            (1, Some(vec![1.0, 0.0]), None),
            (2, Some(vec![0.25, 0.75]), None),
        ])
    );
}

#[test]
fn building_a_vector_index_then_crashing_recovers_the_vectors() {
    let fixture = Fixture::new(SEED, true);
    {
        let session = open(fixture.path());
        run(&session, &set_vector(1, "[0.6, 0.8]"));
        run(
            &session,
            "CALL db.relationship_embeddings.build_index({type: 'CLAIMS', text_column: 'text'}) \
             YIELD indexed RETURN indexed",
        );
    }
    let recovered = try_open(fixture.path()).expect("the log must replay");
    assert_eq!(
        store_state(&recovered, "text").map(|state| state.cells),
        Some(vec![(1, Some(vec![0.6, 0.8]), None)])
    );
    let params = HashMap::new();
    let options = ExecuteOptions::eager(&params);
    let snapshot = recovered.snapshot();
    let rows = execute_read(
        &snapshot,
        "CALL db.relationship_embeddings.list({type: 'CLAIMS', text_column: 'text'}) \
         YIELD count RETURN count",
        &options,
    )
    .expect("list")
    .result
    .rows;
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0][0], Value::Int64(1), "{rows:?}");
}

/// A vacuum renumbers every slot. The embedding stores are keyed by slot, and
/// the checkpoint under replay predates the renumbering, so a frame written
/// after the vacuum must still land its vectors on the right logical members.
#[test]
fn a_vacuum_then_a_property_write_survives_a_crash() {
    let fixture = Fixture::new(SEED, true);
    {
        let session = open(fixture.path());
        run(
            &session,
            "UNWIND range(1, 200) AS i CREATE (:Doomed {id: i})",
        );
        session.save(fixture.path(), true).expect("checkpoint");
        // Every other one, so the survivors sit above holes: deleting the whole
        // contiguous top of the index space shrinks the bound by itself and
        // leaves the vacuum nothing to rebuild.
        run(
            &session,
            "MATCH (d:Doomed) WHERE d.id % 2 = 1 DETACH DELETE d",
        );
        {
            let mut tx = session.begin();
            let remap = tx.working_mut().expect("writable").vacuum();
            assert!(
                remap.describes_rebuild(),
                "fixture: the vacuum must actually renumber slots, or this case tests nothing"
            );
            assert!(matches!(
                session.commit(tx, true),
                CommitOutcome::Committed { .. }
            ));
        }
        run(
            &session,
            "MATCH ()-[r:CLAIMS]->() SET r.note = 'after vacuum'",
        );
    }
    let recovered = try_open(fixture.path()).expect("the log must replay");
    assert_eq!(
        store_state(&recovered, "text").map(|state| state.cells),
        Some(vec![(1, Some(vec![1.0, 0.0]), None)])
    );
    assert_eq!(
        edge_property(&recovered, "note"),
        vec![(1, Some(Value::String("after vacuum".into())))]
    );
}

/// The control. Node embedding stores go through a different capture path
/// (`MutationOp::SetEmbeddings`, a declaration), so a fix that repaired
/// relationships by breaking nodes would pass every case above and fail here.
#[test]
fn a_node_embedding_store_created_inside_the_durable_window_survives_a_crash() {
    let fixture = Fixture::new(SEED, false);
    {
        let session = open(fixture.path());
        let mut tx = session.begin();
        crate::graph::embeddings::set_embeddings(
            tx.working_mut().expect("writable"),
            "Doc",
            "summary",
            Some("cosine"),
            vec![(Value::Int64(1), vec![0.6_f32, 0.8])],
        )
        .expect("node store");
        assert!(matches!(
            session.commit(tx, true),
            CommitOutcome::Committed { .. }
        ));
    }
    let recovered = try_open(fixture.path()).expect("the log must replay");
    let snapshot = recovered.snapshot();
    let store = snapshot
        .embeddings
        .get(&crate::graph::embeddings::store_key("Doc", "summary"))
        .expect("the node store must be recovered");
    assert_eq!((store.dimension, store.len()), (2, 1));
    assert_eq!(store.metric.as_deref(), Some("cosine"));
    let params = HashMap::new();
    let options = ExecuteOptions::eager(&params);
    let score = execute_read(
        &snapshot,
        "MATCH (d:Doc {id: 1}) RETURN vector_score(d, 'summary_emb', [0.6, 0.8]) AS s",
        &options,
    )
    .expect("score")
    .result
    .rows;
    match score.first().map(|row| &row[0]) {
        Some(Value::Float64(value)) => assert!(
            (value - 1.0).abs() < 1e-6,
            "the recovered vector must be the one written, got {value}"
        ),
        other => panic!("expected a float score, got {other:?}"),
    }
}

// ── capture ↔ replay agreement ───────────────────────────────────────

/// The invariant behind every case above, checked directly over a pseudo-random
/// write sequence rather than through a hand-picked shape: for every frame the
/// capture side emits, the state replay reconstructs must equal the state the
/// writer had. Seeded and bounded so a failure is reproducible.
#[test]
fn capture_and_replay_agree_over_a_random_write_sequence() {
    // xorshift, so the corpus is reproducible without a dependency.
    let mut state = 0x5eed_1234_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let fixture = Fixture::new(SEED_PARALLEL, false);
    let (expected_stores, expected_note) = {
        let session = open(fixture.path());
        for step in 0..24u64 {
            match next() % 5 {
                0 => run(&session, &set_vector(1, "[1.0, 0.0]")),
                1 => run(&session, &set_vector(2, "[0.0, 1.0]")),
                2 => run(
                    &session,
                    &format!("MATCH ()-[r:CLAIMS]->() SET r.note = 'n{step}'"),
                ),
                3 => run(
                    &session,
                    "MATCH ()-[r:CLAIMS]->() WITH collect(r) AS rs \
                     CALL db.relationship_embeddings.set({type: 'CLAIMS', text_column: 'note', \
                     entries: [{relationship: rs[0], vector: [0.5, 0.5]}]}) \
                     YIELD stored RETURN stored",
                ),
                _ => run(
                    &session,
                    "CALL db.relationship_embeddings.drop({type: 'CLAIMS', text_column: 'note'}) \
                     YIELD dropped RETURN dropped",
                ),
            }
        }
        (
            vec![store_state(&session, "text"), store_state(&session, "note")],
            edge_property(&session, "note"),
        )
    };
    let recovered = try_open(fixture.path()).expect("the log must replay");
    assert_eq!(
        vec![
            store_state(&recovered, "text"),
            store_state(&recovered, "note"),
        ],
        expected_stores,
        "replay must reconstruct exactly the state the writer held at the crash"
    );
    assert_eq!(edge_property(&recovered, "note"), expected_note);
}
