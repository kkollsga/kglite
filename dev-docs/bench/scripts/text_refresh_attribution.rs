//! Release-only attribution, using the current self-contained production core.
#[path = "../../../crates/kglite/src/graph/algorithms/text_index/mod.rs"]
mod text_index;

use std::{collections::BTreeMap, fs, hint::black_box, time::Instant};
use text_index::TextIndex;

fn main() {
    assert!(!cfg!(debug_assertions), "performance requires release");
    let args: Vec<_> = std::env::args().collect();
    let docs: Vec<String> = serde_json::from_slice(&fs::read(&args[1]).unwrap()).unwrap();
    let deltas: Vec<usize> = args[2].split(',').map(|s| s.parse().unwrap()).collect();
    let base = TextIndex::build(docs.iter().enumerate().map(|(slot, text)| (slot as u32, text)));
    let mut records = Vec::new();
    for delta in deltas {
        assert!(delta <= docs.len());
        // A coprime stride spreads replacements across the full slot range.
        let slots: Vec<_> = (0..delta).map(|i| ((i * 7919) % docs.len()) as u32).collect();
        let replacement = &docs[docs.len() - 1];
        let mut final_docs = docs.clone();
        for &slot in &slots { final_docs[slot as usize] = replacement.clone(); }
        let oracle = TextIndex::build(final_docs.iter().enumerate().map(|(slot, text)| (slot as u32, text)));
        let expected: BTreeMap<_, _> = oracle.iter_terms().collect();
        let query = oracle.prepare_query("w00002 w00249 w03999 w39999");
        let expected_top = oracle.top_k(&query, 10);
        for mode in ["direct", "batch", "rebuild"] {
            // Avoid the known quadratic direct path at very large deltas.
            if mode == "direct" && delta > 1500 { continue; }
            let mut times = Vec::new();
            let mut retained_bytes = 0;
            for _ in 0..5 {
                let mut index = base.clone();
                let start = Instant::now();
                match mode {
                    "direct" => for &slot in &slots { index.add_doc(slot, replacement); },
                    "batch" => { assert_eq!(index.replace_batch(slots.iter().map(|slot| (*slot, Some(replacement)))), delta); },
                    "rebuild" => index = TextIndex::build(final_docs.iter().enumerate().map(|(slot, text)| (slot as u32, text))),
                    _ => unreachable!(),
                }
                times.push(start.elapsed().as_secs_f64());
                black_box(&index);
                assert_eq!(index.total_docs(), oracle.total_docs());
                assert_eq!(index.avgdl(), oracle.avgdl());
                assert_eq!(index.iter_terms().collect::<BTreeMap<_, _>>(), expected);
                assert_eq!(index.top_k(&index.prepare_query("w00002 w00249 w03999 w39999"), 10), expected_top);
                retained_bytes = index.estimated_bytes();
            }
            records.push(serde_json::json!({"documents":docs.len(),"delta":delta,"mode":mode,"seconds":times,"retained_index_bytes":retained_bytes}));
        }
    }
    println!("{}", serde_json::to_string(&records).unwrap());
}
