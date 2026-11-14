//! The exactness oracle of the text column driver: after every step of a
//! seeded random history (create, set, null, remove, delete, slot reuse,
//! secondary labels, a rolled-back statement, a held fork, save and reload in
//! three storage placements, vacuum), the driver's answer for every
//! (type, property, matcher, mode, scope) must equal brute force over
//! `NodeView::resolved_field_str`, or be a decline the state explains.
//!
//! The oracle cannot pass by declining: served and declined calls are counted
//! per lane, every lane must serve, and a decline is accepted only for the
//! two states the test itself models (secondary-label carriers and a column
//! turned non-string).

use super::text_column_scan::{
    text_column_scan, TextAnswer, TextDecline, TextHit, TextMode, TextOutcome, TextScanRequest,
};
use crate::datatypes::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::dir_graph::DirGraph;
use crate::graph::io::file::{
    load_file_with, save_graph_with_layout, with_floors, LoadOptions, MapPolicy, SaveLayout,
};
use crate::graph::schema::InternedKey;
use crate::graph::session::{execute_mut, ExecuteOptions};
use crate::graph::storage::mode::StorageMode;
use crate::graph::storage::{GraphRead, StrField};
use crate::graph::text_scan::matcher::{Matcher, Pattern};
use petgraph::graph::NodeIndex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

// ── helpers shared with the named cases ──────────────────────────────

pub(super) fn run(
    graph: &mut DirGraph,
    query: &str,
    params: &HashMap<String, Value>,
) -> Result<(), String> {
    execute_mut(graph, query, &ExecuteOptions::eager(params))
        .map(|_| ())
        .map_err(|e| e.to_string())
}

pub(super) fn exec(graph: &mut DirGraph, query: &str) {
    run(graph, query, &HashMap::new()).unwrap_or_else(|e| panic!("{query}: {e}"));
}

pub(super) fn param(name: &str, value: &str) -> HashMap<String, Value> {
    HashMap::from([(name.to_string(), Value::String(value.to_string()))])
}

pub(super) fn matchers() -> &'static [(&'static str, Matcher)] {
    static MATCHERS: std::sync::OnceLock<Vec<(&'static str, Matcher)>> = std::sync::OnceLock::new();
    MATCHERS.get_or_init(build_matchers)
}

fn build_matchers() -> Vec<(&'static str, Matcher)> {
    let lit = |t: &str, ci| Pattern::Literal {
        text: t.into(),
        case_insensitive: ci,
    };
    let rx = |p: &str, ci| Pattern::Regex {
        pattern: p.into(),
        case_insensitive: ci,
    };
    let patterns = [
        ("literal", lit("alpha", false)),
        ("short literal", lit("ra", false)),
        ("ci literal", lit("ALPHA", true)),
        (
            "phrase",
            Pattern::Phrase {
                text: "alpha bravo".into(),
                case_insensitive: true,
            },
        ),
        ("anchored regex", rx("^al", false)),
        ("word regex", rx(r"\bbravo\b", false)),
        ("end regex", rx("ra$", true)),
    ];
    patterns
        .into_iter()
        .map(|(name, p)| (name, Matcher::compile(&p).unwrap()))
        .collect()
}

pub(super) fn drive(
    graph: &DirGraph,
    node_type: &str,
    property: &str,
    matcher: &Matcher,
    scope: Option<&[NodeIndex]>,
    mode: TextMode,
) -> TextOutcome {
    text_column_scan(
        graph,
        &TextScanRequest {
            node_type,
            property,
            matcher,
            scope,
            mode,
            interrupt: Interrupt::default(),
            serial: false,
        },
    )
}

/// Per node, its occurrences `(offset, length)` in the order the per-row path
/// visits nodes: the bucket, or the scope with foreign and dead nodes dropped
/// and repeats counted once.
pub(super) fn brute(
    graph: &DirGraph,
    node_type: &str,
    property: &str,
    matcher: &Matcher,
    scope: Option<&[NodeIndex]>,
) -> Vec<(NodeIndex, Vec<(u64, u64)>)> {
    let type_key = InternedKey::from_str(node_type);
    let field = graph.resolve_alias(node_type, property);
    let key = InternedKey::from_str(field);
    let candidates: Vec<NodeIndex> = match scope {
        None => graph
            .type_indices
            .get(node_type)
            .map(|n| n.to_vec())
            .unwrap_or_default(),
        Some(nodes) => {
            let mut seen = HashSet::new();
            nodes
                .iter()
                .copied()
                .filter(|n| seen.insert(*n))
                .filter(|n| {
                    graph
                        .graph
                        .node_view(*n)
                        .is_some_and(|v| v.node_type() == type_key)
                })
                .collect()
        }
    };
    let mut out = Vec::new();
    for idx in candidates {
        let view = graph.graph.node_view(idx).expect("candidate is live");
        let StrField::Str(text) = view.resolved_field_str(node_type, field, key) else {
            continue;
        };
        let mut hits = Vec::new();
        matcher.find_in_record(text.as_bytes(), &mut |s, e| {
            hits.push((s as u64, (e - s) as u64));
            true
        });
        if !hits.is_empty() {
            out.push((idx, hits));
        }
    }
    out
}

pub(super) fn expected(truth: &[(NodeIndex, Vec<(u64, u64)>)], mode: TextMode) -> TextAnswer {
    match mode {
        TextMode::Exists => TextAnswer::Nodes(truth.iter().map(|(n, _)| *n).collect()),
        TextMode::Count => TextAnswer::Count(truth.len() as u64),
        TextMode::Hits { max } => {
            let all: Vec<TextHit> = truth
                .iter()
                .flat_map(|(node, hits)| {
                    hits.iter().map(|&(offset, len)| TextHit {
                        node: *node,
                        offset,
                        len,
                    })
                })
                .collect();
            let truncated = max.is_some_and(|m| all.len() > m);
            let hits = all.into_iter().take(max.unwrap_or(usize::MAX)).collect();
            TextAnswer::Hits { hits, truncated }
        }
    }
}

pub(super) const MODES: [TextMode; 4] = [
    TextMode::Exists,
    TextMode::Count,
    TextMode::Hits { max: None },
    TextMode::Hits { max: Some(3) },
];

#[derive(Default, Debug, Clone, Copy)]
pub(super) struct Tally {
    pub served: usize,
    pub declined: usize,
}

/// Every matcher and mode against brute force on one (type, property, scope).
/// A decline must be one of `allowed`.
pub(super) fn assert_exact(
    graph: &DirGraph,
    node_type: &str,
    property: &str,
    scope: Option<&[NodeIndex]>,
    allowed: &[TextDecline],
    context: &str,
) -> Tally {
    assert_exact_with(graph, node_type, property, scope, allowed, context, &[])
}

/// [`assert_exact`] restricted to the named matchers (all when empty).
pub(super) fn assert_exact_with(
    graph: &DirGraph,
    node_type: &str,
    property: &str,
    scope: Option<&[NodeIndex]>,
    allowed: &[TextDecline],
    context: &str,
    only: &[&str],
) -> Tally {
    let mut tally = Tally::default();
    for (name, matcher) in matchers() {
        if !only.is_empty() && !only.contains(name) {
            continue;
        }
        let truth = brute(graph, node_type, property, matcher, scope);
        for mode in MODES {
            let got = drive(graph, node_type, property, matcher, scope, mode)
                .expect("no interrupt was raised");
            match got {
                Ok(answer) => {
                    assert_eq!(
                        answer,
                        expected(&truth, mode),
                        "{context}: {node_type}.{property} {name} {mode:?} scope={}",
                        scope.map_or("none".into(), |s| format!("{} nodes", s.len()))
                    );
                    tally.served += 1;
                }
                Err(decline) => {
                    assert!(
                        allowed.contains(&decline),
                        "{context}: {node_type}.{property} declined with {decline:?}, expected one of {allowed:?}"
                    );
                    tally.declined += 1;
                }
            }
        }
    }
    tally
}

// ── the history ──────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

const TYPES: [&str; 3] = ["A", "B", "C"];
const PROPS: [&str; 2] = ["a", "b"];
const WORDS: [&str; 17] = [
    "al",
    "pha",
    "alpha",
    "Alpha",
    "ALPHA",
    "bravo",
    "Bravo",
    "ravo",
    "br",
    "ra",
    "charlie\ncharlie",
    "delta",
    "é",
    "ß",
    "alpha bravo",
    "alpha  bravo",
    "alpha\nbravo",
];
const SEPARATORS: [&str; 4] = ["", " ", "\n", "-"];

fn random_text(rng: &mut Rng) -> String {
    let mut text = String::new();
    for i in 0..rng.below(4) {
        if i > 0 {
            text.push_str(rng.pick(&SEPARATORS));
        }
        text.push_str(rng.pick(&WORDS));
    }
    text
}

/// A different string of the same byte length, when the text has letters to flip.
fn same_length(text: &str, rng: &mut Rng) -> String {
    match rng.below(3) {
        0 => text.to_ascii_uppercase(),
        1 => text.to_ascii_lowercase(),
        _ => text.chars().rev().collect::<String>(),
    }
}

#[derive(Clone)]
struct ModelNode {
    /// A seeded filler row no later step touches (it keeps a column over the
    /// size at which a save writes it as a mappable section).
    fixed: bool,
    ty: usize,
    vals: [Option<String>; 2],
    label: Option<&'static str>,
}

#[derive(Clone, Default)]
struct Model {
    next_id: i64,
    nodes: BTreeMap<i64, ModelNode>,
    /// The big history keeps every edit on type A, the one with mapped columns.
    only_a: bool,
    /// (type, property) columns a non-string value was written to.
    non_string: HashSet<(usize, usize)>,
}

impl Model {
    /// The declines this state explains for (type, property).
    fn allowed(&self, ty: usize, prop: usize) -> Vec<TextDecline> {
        let mut allowed = Vec::new();
        if self
            .nodes
            .values()
            .any(|n| n.ty != ty && n.label == Some(TYPES[ty]))
        {
            allowed.push(TextDecline::SecondaryLabel);
        }
        if !self.nodes.values().any(|n| n.ty == ty) {
            allowed.push(TextDecline::NoStore);
        }
        if self.non_string.contains(&(ty, prop)) {
            allowed.push(TextDecline::NotStringColumn);
        }
        allowed
    }
}

#[derive(Debug, Clone)]
enum Op {
    Create,
    SetText,
    SetSameLength,
    SetNull,
    Remove,
    Delete,
    Label,
    Rollback,
    NonString,
    Fork,
    Reload,
    Vacuum,
}

struct Lane {
    name: &'static str,
    /// (type, property) states where the driver served a column whose bytes
    /// are a mapped file section and which also holds relocated rows.
    mapped_relocated_served: usize,
    /// Served beside a store that has a heap tail.
    tail_served: usize,
    graph: Arc<DirGraph>,
    /// Held snapshots, each with the model it was taken under.
    held: Vec<(Arc<DirGraph>, Model)>,
    tally: Tally,
}

type Effect = Box<dyn FnOnce(&mut Model)>;

fn mutate(lane: &mut Lane, query: &str, params: &HashMap<String, Value>) -> Result<(), String> {
    run(Arc::make_mut(&mut lane.graph), query, params)
}

fn pick_node(model: &Model, rng: &mut Rng) -> Option<i64> {
    let ids: Vec<i64> = model
        .nodes
        .iter()
        .filter(|(_, n)| !n.fixed && (!model.only_a || n.ty == 0))
        .map(|(id, _)| *id)
        .collect();
    (!ids.is_empty()).then(|| *rng.pick(&ids))
}

/// Apply `op` to the model and every lane. Returns false when the op had no
/// applicable target and was skipped.
fn apply(
    op: &Op,
    model: &mut Model,
    lanes: &mut [Lane],
    rng: &mut Rng,
    dir: &std::path::Path,
) -> bool {
    let mut params = HashMap::new();
    let (query, effect): (String, Effect) = match op {
        Op::Create => {
            let ty = if model.only_a { 0 } else { rng.below(3) };
            let id = model.next_id;
            let vals = [random_text(rng), random_text(rng)];
            let keep = [rng.below(5) != 0, rng.below(5) != 0];
            let mut props = vec![format!("id: {id}")];
            for p in 0..2 {
                // The planner refuses a property the type has not declared, so an
                // absent value is an explicit null.
                props.push(format!("{}: ${}", PROPS[p], PROPS[p]));
                params.insert(
                    PROPS[p].to_string(),
                    if keep[p] {
                        Value::String(vals[p].clone())
                    } else {
                        Value::Null
                    },
                );
            }
            let node = ModelNode {
                fixed: false,
                ty,
                vals: [
                    keep[0].then(|| vals[0].clone()),
                    keep[1].then(|| vals[1].clone()),
                ],
                label: None,
            };
            (
                format!("CREATE (:{} {{{}}})", TYPES[ty], props.join(", ")),
                Box::new(move |m: &mut Model| {
                    m.nodes.insert(id, node);
                    m.next_id += 1;
                }),
            )
        }
        Op::SetText | Op::SetSameLength | Op::SetNull | Op::Remove => {
            let Some(id) = pick_node(model, rng) else {
                return false;
            };
            let node = &model.nodes[&id];
            let p = rng.below(2);
            let (set, value): (String, Option<String>) = match op {
                Op::SetText => {
                    let v = random_text(rng);
                    params.insert("v".into(), Value::String(v.clone()));
                    (format!("n.{} = $v", PROPS[p]), Some(v))
                }
                Op::SetSameLength => {
                    let Some(current) = node.vals[p].clone() else {
                        return false;
                    };
                    let v = same_length(&current, rng);
                    params.insert("v".into(), Value::String(v.clone()));
                    (format!("n.{} = $v", PROPS[p]), Some(v))
                }
                Op::SetNull => (format!("n.{} = null", PROPS[p]), None),
                _ => (String::new(), None),
            };
            let clause = if matches!(op, Op::Remove) {
                format!("REMOVE n.{}", PROPS[p])
            } else {
                format!("SET {set}")
            };
            let ty = node.ty;
            (
                format!("MATCH (n:{} {{id: {id}}}) {clause}", TYPES[ty]),
                Box::new(move |m: &mut Model| m.nodes.get_mut(&id).unwrap().vals[p] = value),
            )
        }
        Op::Delete => {
            let Some(id) = pick_node(model, rng) else {
                return false;
            };
            let ty = model.nodes[&id].ty;
            (
                format!("MATCH (n:{} {{id: {id}}}) DELETE n", TYPES[ty]),
                Box::new(move |m: &mut Model| {
                    m.nodes.remove(&id);
                }),
            )
        }
        Op::Label => {
            let Some(id) = pick_node(model, rng) else {
                return false;
            };
            let node = &model.nodes[&id];
            if node.label.is_some() {
                return false;
            }
            let label = *rng.pick(&["A", "B", "C", "L"]);
            let ty = node.ty;
            (
                format!("MATCH (n:{} {{id: {id}}}) SET n:{label}", TYPES[ty]),
                Box::new(move |m: &mut Model| m.nodes.get_mut(&id).unwrap().label = Some(label)),
            )
        }
        Op::Rollback => {
            let Some(id) = pick_node(model, rng) else {
                return false;
            };
            let ty = model.nodes[&id].ty;
            let query = format!(
                "MATCH (n:{} {{id: {id}}}) SET n.a = 'rolled back to a longer value' SET n.id = 424242",
                TYPES[ty]
            );
            for lane in lanes.iter_mut() {
                assert!(
                    mutate(lane, &query, &params).is_err(),
                    "{}: the statement must fail after its write",
                    lane.name
                );
            }
            return true;
        }
        Op::NonString => {
            let Some(id) = pick_node(model, rng) else {
                return false;
            };
            let ty = model.nodes[&id].ty;
            (
                format!("MATCH (n:{} {{id: {id}}}) SET n.b = 7", TYPES[ty]),
                Box::new(move |m: &mut Model| {
                    m.nodes.get_mut(&id).unwrap().vals[1] = None;
                    m.non_string.insert((ty, 1));
                }),
            )
        }
        Op::Fork => {
            for lane in lanes.iter_mut() {
                lane.held.push((Arc::clone(&lane.graph), model.clone()));
                if lane.held.len() > 2 {
                    lane.held.remove(0);
                }
            }
            return true;
        }
        Op::Reload => {
            reload(lanes, dir);
            return true;
        }
        Op::Vacuum => {
            for lane in lanes.iter_mut() {
                Arc::make_mut(&mut lane.graph).vacuum();
            }
            return true;
        }
    };
    for lane in lanes.iter_mut() {
        mutate(lane, &query, &params).unwrap_or_else(|e| panic!("{}: {query}: {e}", lane.name));
    }
    effect(model);
    true
}

/// Replace every lane but the heap one with a load of its own saved state.
fn reload(lanes: &mut [Lane], dir: &std::path::Path) {
    let big = lanes.len() == 3;
    for (i, lane) in lanes.iter_mut().enumerate().skip(1) {
        let path = dir.join(format!("lane{i}.kgl"));
        let mut arc = Arc::clone(&lane.graph);
        with_floors(10, 10, || {
            save_graph_with_layout(&mut arc, path.to_str().unwrap(), true, SaveLayout::Auto)
        })
        .unwrap();
        let options = match i + usize::from(big) {
            1 => LoadOptions::new()
                .with_storage(StorageMode::Memory)
                .with_columns(MapPolicy::Memory),
            2 => LoadOptions::new()
                .with_storage(StorageMode::Memory)
                .with_columns(MapPolicy::Mapped),
            _ => LoadOptions::new()
                .with_storage(StorageMode::Mapped)
                .with_columns(MapPolicy::Mapped),
        };
        lane.graph = load_file_with(path.to_str().unwrap(), &options).unwrap();
    }
}

/// Whether the (type, property) `Str` column's bytes are a mapped section, and
/// how many rows its overlay holds.
pub(super) fn column_state(graph: &DirGraph, node_type: &str, property: &str) -> (bool, usize) {
    use crate::graph::storage::column_store::TypedColumn;
    let Some(store) = graph.column_store(node_type) else {
        return (false, 0);
    };
    match store
        .slot(InternedKey::from_str(property))
        .and_then(|s| store.column(s as usize))
    {
        Some(TypedColumn::Str {
            data, relocated, ..
        }) => (data.shared_section().is_some(), relocated.len()),
        _ => (false, 0),
    }
}

fn scope_for(graph: &DirGraph, rng: &mut Rng) -> Vec<NodeIndex> {
    use crate::graph::storage::GraphRead;
    let live: Vec<NodeIndex> = graph.graph.node_indices().collect();
    let bound = live.iter().map(|n| n.index()).max().unwrap_or(0) + 6;
    let mut scope = Vec::new();
    for _ in 0..rng.below(14) {
        if !live.is_empty() && rng.below(3) != 0 {
            scope.push(*rng.pick(&live));
        } else {
            // A dead slot, a hole, or a node of another type.
            scope.push(NodeIndex::new(rng.below(bound)));
        }
    }
    if let Some(&first) = scope.first() {
        scope.push(first);
    }
    scope
}

fn check_lane(lane: &mut Lane, model: &Model, rng: &mut Rng, step: usize, big: bool) {
    let mut graphs: Vec<(Arc<DirGraph>, Model, String)> =
        vec![(Arc::clone(&lane.graph), model.clone(), "live".into())];
    for (i, (g, m)) in lane.held.iter().enumerate() {
        graphs.push((Arc::clone(g), m.clone(), format!("held {i}")));
    }
    for (graph, model, which) in graphs {
        for (t, ty) in TYPES.iter().enumerate().take(if big { 1 } else { 3 }) {
            for (p, prop) in PROPS.iter().enumerate() {
                let allowed = model.allowed(t, p);
                let context = format!("{} step {step} {which}", lane.name);
                let scope = scope_for(&graph, rng);
                let scopes: &[Option<&[NodeIndex]>] = if big {
                    &[None, Some(scope.as_slice())]
                } else {
                    &[None, Some(scope.as_slice()), Some(&[])]
                };
                // Every scan of the big history covers 300 KB.
                let only: &[&str] = if big {
                    &["literal", "ci literal", "word regex"]
                } else {
                    &[]
                };
                for &scope in scopes {
                    let tally =
                        assert_exact_with(&graph, ty, prop, scope, &allowed, &context, only);
                    lane.tally.served += tally.served;
                    lane.tally.declined += tally.declined;
                    if tally.served > 0 && which == "live" {
                        let (mapped, relocated) = column_state(&graph, ty, prop);
                        lane.mapped_relocated_served += usize::from(mapped && relocated > 0);
                        lane.tail_served +=
                            usize::from(graph.column_store(ty).is_some_and(|s| s.has_heap_tail()));
                    }
                }
            }
        }
    }
}

/// 36 rows of 8 KB in each of A's two columns: over the 256 KB a save needs
/// before it writes a column as a section a load can map.
fn seed_big_type_a(model: &mut Model, lanes: &mut [Lane]) {
    for i in 0..36 {
        let mut value = "z".repeat(4000);
        if i % 3 == 0 {
            value.push_str(" alpha bravo ");
        }
        value.push_str(&"z".repeat(4000));
        let id = model.next_id;
        let params = HashMap::from([("v".to_string(), Value::String(value.clone()))]);
        for lane in lanes.iter_mut() {
            mutate(
                lane,
                &format!("CREATE (:A {{id: {id}, a: $v, b: $v}})"),
                &params,
            )
            .unwrap();
        }
        model.nodes.insert(
            id,
            ModelNode {
                fixed: true,
                ty: 0,
                vals: [Some(value.clone()), Some(value)],
                label: None,
            },
        );
        model.next_id += 1;
    }
}

fn history(seed: u64, steps: usize, big: bool) -> Vec<Lane> {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = Rng(seed);
    let mut model = Model::default();
    let new_lane = |name| Lane {
        name,
        mapped_relocated_served: 0,
        tail_served: 0,
        graph: Arc::new(DirGraph::new()),
        held: Vec::new(),
        tally: Tally::default(),
    };
    let mut lanes = if big {
        // Only the loads that map: the heap placements are exercised by the
        // small history, and every scan here covers 300 KB.
        vec![
            new_lane("heap"),
            new_lane("mapped columns"),
            new_lane("mapped mode"),
        ]
    } else {
        vec![
            new_lane("heap"),
            new_lane("loaded memory"),
            new_lane("mapped columns"),
            new_lane("mapped mode"),
        ]
    };
    if big {
        model.only_a = true;
        seed_big_type_a(&mut model, &mut lanes);
    }
    // Nulls and labels are the states that explain declines; the big history
    // is about served mapped columns, and the small one covers them.
    let (nulls, labels) = if big { (0, 0) } else { (2, 1) };
    // A create pushes onto a mapped column and copies it to the heap, and a
    // vacuum rebuilds it there; the big history favours edits so that
    // mapped columns with an overlay are what it sees.
    let (creates, edits, vacuums) = if big { (1, 6, 0) } else { (6, 4, 1) };
    let weighted = [
        (Op::Create, creates),
        (Op::SetText, edits),
        (Op::SetSameLength, edits),
        (Op::SetNull, nulls),
        (Op::Remove, nulls),
        (Op::Delete, 3),
        (Op::Label, labels),
        (Op::Rollback, 1),
        (Op::NonString, 0),
        (Op::Fork, 1),
        (Op::Reload, 2),
        (Op::Vacuum, vacuums),
    ];
    let bag: Vec<&Op> = weighted
        .iter()
        .flat_map(|(op, w)| std::iter::repeat_n(op, *w))
        .collect();
    // Seed rows so that every type has a column before the first reload.
    for _ in 0..9 {
        apply(&Op::Create, &mut model, &mut lanes, &mut rng, dir.path());
    }
    for step in 0..steps {
        let op = if step == steps / 2 && seed.is_multiple_of(2) && !big {
            Op::NonString
        } else {
            (*rng.pick(&bag)).clone()
        };
        if !apply(&op, &mut model, &mut lanes, &mut rng, dir.path()) {
            continue;
        }
        for lane in lanes.iter_mut() {
            check_lane(lane, &model, &mut rng, step, big);
        }
    }
    lanes
}

fn report(seed: u64, lanes: &[Lane]) {
    for lane in lanes {
        eprintln!(
            "oracle {seed:#x} {:<15} served {:>6} declined {:>5} mapped+relocated {:>4} tail {}",
            lane.name,
            lane.tally.served,
            lane.tally.declined,
            lane.mapped_relocated_served,
            lane.tail_served
        );
    }
}

#[test]
fn the_driver_equals_the_per_row_path_across_random_histories() {
    let (mut served, mut declined) = (0, 0);
    for seed in [
        0x9E37_79B9_7F4A_7C15u64,
        0xD1B5_4A32_D192_ED03,
        0x2545_F491_4F6C_DD1D,
    ] {
        let lanes = history(seed, 36, false);
        report(seed, &lanes);
        for lane in &lanes {
            assert!(lane.tally.served > 500, "{}: {:?}", lane.name, lane.tally);
        }
        served += lanes.iter().map(|l| l.tally.served).sum::<usize>();
        declined += lanes.iter().map(|l| l.tally.declined).sum::<usize>();
    }
    assert!(
        served > declined * 2,
        "served {served}, declined {declined}"
    );
}

/// The same oracle over columns big enough to be written as sections and
/// mapped on load, so that edits land in the overlay of a mapped buffer.
#[test]
fn mapped_columns_with_edited_rows_are_served_and_exact() {
    let mut mapped_relocated = [0usize; 3];
    for seed in [0xE703_7ED1_A0B4_28DBu64] {
        let lanes = history(seed, 18, true);
        report(seed, &lanes);
        for lane in &lanes {
            assert!(
                lane.tally.served > lane.tally.declined * 10,
                "{}: {:?}",
                lane.name,
                lane.tally
            );
        }
        for (i, lane) in lanes.iter().enumerate() {
            mapped_relocated[i] += lane.mapped_relocated_served;
        }
    }
    assert_eq!(mapped_relocated[0], 0, "the heap lane maps nothing");
    for (i, name) in [(1, "mapped columns"), (2, "mapped mode")] {
        assert!(
            mapped_relocated[i] > 0,
            "{name}: no served state had a mapped column with relocated rows"
        );
    }
}
