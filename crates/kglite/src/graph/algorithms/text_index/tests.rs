//! Tests for the BM25 core. The centrepiece is a **naive reference** BM25
//! written from scratch below — its own tokenizer, its own corpus statistics,
//! its own scoring loop — asserted equal to the index *exactly*, not
//! approximately. Exact equality is available because both sides evaluate the
//! same formula in the same order over the same values, and IEEE-754 is
//! deterministic under those conditions; an approximate assertion would let a
//! genuine reordering bug (which shows up first as last-ulp disagreement on
//! near-ties) hide inside the tolerance.

use super::analyzer::analyze;
use super::bm25::{PreparedQuery, ScoredDoc, B, K1};
use super::TextIndex;
use std::borrow::Cow;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Naive reference implementation — the oracle.
// ---------------------------------------------------------------------------

/// Independent tokenizer. Deliberately *not* [`analyze`]: if the reference
/// borrowed the production tokenizer, tokenization would be the one part of
/// the pipeline the oracle could not see.
fn reference_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            current.extend(c.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// BM25 from scratch: tokenize the whole corpus, count everything, score every
/// document. O(corpus) per query and proud of it.
fn reference_scores(corpus: &[(u32, String)], query: &str) -> BTreeMap<u32, f64> {
    let docs: Vec<(u32, Vec<String>)> = corpus
        .iter()
        .map(|(slot, text)| (*slot, reference_tokens(text)))
        .collect();
    let n = docs.len();
    if n == 0 {
        return BTreeMap::new();
    }
    let total_len: usize = docs.iter().map(|(_, tokens)| tokens.len()).sum();
    let avgdl = total_len as f64 / n as f64;

    let mut terms: Vec<(String, f64)> = Vec::new();
    for token in reference_tokens(query) {
        if terms.iter().any(|(seen, _)| *seen == token) {
            continue;
        }
        let df = docs
            .iter()
            .filter(|(_, tokens)| tokens.contains(&token))
            .count();
        if df == 0 {
            continue;
        }
        let idf = (1.0 + (n as f64 - df as f64 + 0.5) / (df as f64 + 0.5)).ln();
        terms.push((token, idf));
    }

    let mut scores = BTreeMap::new();
    for (slot, tokens) in &docs {
        if avgdl <= 0.0 {
            scores.insert(*slot, 0.0);
            continue;
        }
        let norm = K1 * (1.0 - B + B * (tokens.len() as f64 / avgdl));
        let mut total = 0.0;
        for (token, idf) in &terms {
            let tf = tokens.iter().filter(|t| *t == token).count();
            if tf == 0 {
                continue;
            }
            let tf = tf as f64;
            total += idf * (tf * (K1 + 1.0)) / (tf + norm);
        }
        scores.insert(*slot, total);
    }
    scores
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn corpus(docs: &[(u32, &str)]) -> Vec<(u32, String)> {
    docs.iter()
        .map(|(slot, text)| (*slot, (*text).to_string()))
        .collect()
}

/// Everything about an index that is not an internal id assignment: the term →
/// postings mapping, per-document lengths, and the corpus counters. Two indexes
/// with equal snapshots score identically on every query.
type Snapshot = (BTreeMap<String, Vec<(u32, u32)>>, BTreeMap<u32, u32>, u64);

fn snapshot(index: &TextIndex) -> Snapshot {
    let terms = index
        .iter_terms()
        .map(|(name, postings)| {
            (
                name.to_string(),
                postings.iter().map(|p| (p.slot, p.tf)).collect(),
            )
        })
        .collect();
    let lengths = index
        .doc_slots()
        .map(|slot| (slot, index.doc_len(slot).unwrap()))
        .collect();
    (terms, lengths, index.total_len)
}

fn assert_matches_reference(docs: &[(u32, String)], queries: &[&str]) {
    let index = TextIndex::build(docs.iter().map(|(slot, text)| (*slot, text)));
    index.validate().unwrap();
    for query in queries {
        let prepared = index.prepare_query(query);
        let expected = reference_scores(docs, query);
        for (slot, want) in &expected {
            let got = index.score(*slot, &prepared);
            assert_eq!(
                got, *want,
                "query {query:?} slot {slot}: {got:?} != reference {want:?}"
            );
        }
    }
}

/// xorshift64*, so a "random" test is a fixed, replayable sequence rather than
/// a different test on every run.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

const WORDS: [&str; 8] = [
    "alpha", "beta", "gamma", "delta", "tromsø", "日本", "epsilon", "zeta",
];

fn random_text(rng: &mut Rng) -> String {
    let count = rng.below(7);
    (0..count)
        .map(|_| WORDS[rng.below(WORDS.len() as u64) as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

fn tokens(text: &str) -> Vec<String> {
    analyze(text).map(|token| token.into_owned()).collect()
}

#[test]
fn analyzer_keeps_unicode_letters_and_drops_punctuation() {
    assert_eq!(tokens("Tromsø, Norway!"), ["tromsø", "norway"]);
    assert_eq!(tokens("  --  "), Vec::<String>::new());
    assert_eq!(tokens(""), Vec::<String>::new());
    // Punctuation is a separator, never part of a term: no contraction and no
    // decimal survives whole.
    assert_eq!(tokens("it's 3.14"), ["it", "s", "3", "14"]);
}

#[test]
fn analyzer_lowercases_multi_char_expansions() {
    // U+0130 lowercases to two chars ("i" + combining dot above); a per-char
    // `to_lowercase().next()` would silently truncate it.
    let token = tokens("İstanbul").remove(0);
    assert_eq!(token.chars().count(), 9);
    assert!(token.starts_with('i'));
}

#[test]
fn analyzer_does_not_segment_cjk() {
    // Documented limitation, asserted so it cannot change unnoticed: Han and
    // Katakana are alphanumeric, so an unbroken run is one term.
    assert_eq!(tokens("日本語テキスト"), ["日本語テキスト"]);
    assert_eq!(tokens("日本語 テキスト"), ["日本語", "テキスト"]);
}

#[test]
fn analyzer_borrows_when_no_lowercasing_is_needed() {
    let borrowed: Vec<Cow<'_, str>> = analyze("already lower 42").collect();
    assert!(borrowed
        .iter()
        .all(|token| matches!(token, Cow::Borrowed(_))));
    let owned: Vec<Cow<'_, str>> = analyze("Mixed Case").collect();
    assert!(owned.iter().all(|token| matches!(token, Cow::Owned(_))));
}

#[test]
fn the_analyzer_is_the_only_tokenizer_the_index_uses() {
    // Build and query cannot drift because they share `analyze`; this asserts
    // the query side really does resolve a term the build side interned.
    let index = TextIndex::build([(0u32, "Tromsø Kommune")]);
    assert_eq!(index.df("tromsø"), 1);
    assert!(index.prepare_query("TROMSØ").terms().len() == 1);
}

// ---------------------------------------------------------------------------
// Reference-oracle equality
// ---------------------------------------------------------------------------

#[test]
fn fresh_build_scores_exactly_match_the_naive_reference() {
    let docs = corpus(&[
        (0, "the quick brown fox jumps over the lazy dog"),
        (1, "a quick brown dog outpaces a quick fox"),
        (2, "lorem ipsum dolor sit amet"),
        (3, "the dog barked; the DOG barked again"),
        (4, "Tromsø ligger i Nord-Norge, nord for polarsirkelen"),
    ]);
    assert_matches_reference(
        &docs,
        &[
            "quick fox",
            "dog",
            "the",
            "tromsø nord",
            "nothing matches here",
            "",
        ],
    );
}

#[test]
fn empty_documents_still_count_in_the_corpus() {
    let docs = corpus(&[(0, "alpha beta"), (1, ""), (2, "alpha")]);
    let index = TextIndex::build(docs.iter().map(|(slot, text)| (*slot, text)));
    index.validate().unwrap();
    assert_eq!(
        index.total_docs(),
        3,
        "an empty document is still a document"
    );
    assert_eq!(index.doc_len(1), Some(0));
    assert_eq!(index.avgdl(), 1.0);
    assert_matches_reference(&docs, &["alpha", "beta"]);
}

#[test]
fn single_document_corpus_scores_match_the_reference() {
    let docs = corpus(&[(7, "solitary document about rust")]);
    assert_matches_reference(&docs, &["rust", "document rust", "absent"]);
    let index = TextIndex::build(docs.iter().map(|(slot, text)| (*slot, text)));
    let prepared = index.prepare_query("rust");
    assert!(
        index.score(7, &prepared) > 0.0,
        "smoothed idf stays positive"
    );
}

#[test]
fn an_all_empty_corpus_scores_zero_rather_than_nan() {
    let index = TextIndex::build([(0u32, ""), (1u32, "")]);
    index.validate().unwrap();
    assert_eq!(index.avgdl(), 0.0);
    let prepared = index.prepare_query("anything");
    assert_eq!(index.score(0, &prepared), 0.0);
}

#[test]
fn unindexed_and_unmatched_slots_score_zero() {
    let index = TextIndex::build([(0u32, "alpha"), (1u32, "beta")]);
    let prepared = index.prepare_query("alpha");
    assert_eq!(index.score(1, &prepared), 0.0, "indexed but no shared term");
    assert_eq!(index.score(99, &prepared), 0.0, "never indexed");
    assert!(index.prepare_query("unknown-term").is_empty());
}

// ---------------------------------------------------------------------------
// Mutation surface
// ---------------------------------------------------------------------------

#[test]
fn incremental_adds_reach_the_same_index_as_a_bulk_build() {
    let docs = corpus(&[
        (3, "alpha beta gamma"),
        (0, "beta beta delta"),
        (9, "gamma delta epsilon"),
        (1, ""),
    ]);
    let bulk = TextIndex::build(docs.iter().map(|(slot, text)| (*slot, text)));
    let mut incremental = TextIndex::new();
    for (slot, text) in &docs {
        incremental.add_doc(*slot, text);
        incremental.validate().unwrap();
    }
    assert_eq!(snapshot(&bulk), snapshot(&incremental));
    for query in ["beta", "gamma delta", "alpha epsilon"] {
        let a = bulk.prepare_query(query);
        let b = incremental.prepare_query(query);
        for slot in bulk.doc_slots() {
            assert_eq!(bulk.score(slot, &a), incremental.score(slot, &b));
        }
    }
}

#[test]
fn remove_then_re_add_restores_the_index_exactly() {
    let docs = corpus(&[(0, "alpha beta"), (1, "beta gamma"), (2, "gamma delta")]);
    let mut index = TextIndex::build(docs.iter().map(|(slot, text)| (*slot, text)));
    let before = snapshot(&index);

    assert!(index.remove_doc(1));
    assert!(
        !index.remove_doc(1),
        "removing twice reports the second miss"
    );
    index.validate().unwrap();
    assert_eq!(index.total_docs(), 2);
    assert!(!index.postings_for("beta").iter().any(|p| p.slot == 1));

    index.add_doc(1, "beta gamma");
    index.validate().unwrap();
    assert_eq!(snapshot(&index), before);
}

#[test]
fn a_terms_last_posting_retires_it_and_recycles_the_id() {
    let mut index = TextIndex::build([(0u32, "alpha unique"), (1u32, "alpha")]);
    let retired = index.term_id("unique").unwrap();
    assert_eq!(index.vocabulary_len(), 2);

    index.remove_doc(0);
    index.validate().unwrap();
    assert_eq!(
        index.vocabulary_len(),
        1,
        "'unique' is gone from the dictionary"
    );
    assert_eq!(index.term_id("unique"), None);
    assert_eq!(index.df("unique"), 0);

    index.add_doc(2, "fresh");
    index.validate().unwrap();
    assert_eq!(
        index.term_id("fresh"),
        Some(retired),
        "the freed id is reused instead of growing the id space"
    );
}

#[test]
fn add_doc_replaces_the_slots_previous_text() {
    let mut index = TextIndex::build([(0u32, "alpha beta beta"), (1u32, "gamma")]);
    assert_eq!(index.doc_len(0), Some(3));
    index.add_doc(0, "delta");
    index.validate().unwrap();

    assert_eq!(index.doc_len(0), Some(1));
    assert_eq!(index.total_docs(), 2, "an upsert is not a second document");
    assert_eq!(index.df("beta"), 0, "the replaced text leaves no postings");
    assert_eq!(index.term_freq(0, index.term_id("delta").unwrap()), 1);
    assert_eq!(
        snapshot(&index),
        snapshot(&TextIndex::build([(0u32, "delta"), (1u32, "gamma")]))
    );
}

#[test]
fn a_repeated_slot_in_one_bulk_build_keeps_the_last_text() {
    let index = TextIndex::build([(0u32, "alpha"), (1u32, "beta"), (0u32, "gamma")]);
    index.validate().unwrap();
    assert_eq!(index.total_docs(), 2);
    assert_eq!(index.df("alpha"), 0);
    assert_eq!(index.df("gamma"), 1);
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

fn sorted_by_rank(index: &TextIndex, query: &PreparedQuery) -> Vec<ScoredDoc> {
    let mut all: Vec<ScoredDoc> = index
        .doc_slots()
        .map(|slot| ScoredDoc {
            slot,
            score: index.score(slot, query),
        })
        .filter(|scored| scored.score > 0.0)
        .collect();
    all.sort();
    all
}

#[test]
fn top_k_equals_sorting_every_document_and_taking_k() {
    let docs = corpus(&[
        (0, "rust systems programming language"),
        (1, "rust rust rust"),
        (2, "programming in a systems language"),
        (3, "unrelated prose about gardening"),
        (4, "rust programming"),
    ]);
    let index = TextIndex::build(docs.iter().map(|(slot, text)| (*slot, text)));
    for query in ["rust", "rust programming", "systems language", "gardening"] {
        let prepared = index.prepare_query(query);
        let expected = sorted_by_rank(&index, &prepared);
        for k in 0..=6 {
            let got = index.top_k(&prepared, k);
            let want: Vec<ScoredDoc> = expected.iter().copied().take(k).collect();
            assert_eq!(
                got.iter().map(|s| (s.slot, s.score)).collect::<Vec<_>>(),
                want.iter().map(|s| (s.slot, s.score)).collect::<Vec<_>>(),
                "query {query:?} k={k}"
            );
        }
    }
}

#[test]
fn ties_break_on_ascending_slot_deterministically() {
    // Four identical documents: BM25 cannot separate them, so only the
    // documented tiebreak decides the order — and it must decide it the same
    // way every run, whatever the hash map's iteration order happens to be.
    let index = TextIndex::build([
        (40u32, "same words here"),
        (10u32, "same words here"),
        (30u32, "same words here"),
        (20u32, "same words here"),
    ]);
    let prepared = index.prepare_query("same words");
    let ranked = index.top_k(&prepared, 4);
    assert_eq!(
        ranked.iter().map(|s| s.slot).collect::<Vec<_>>(),
        [10, 20, 30, 40]
    );
    for pair in ranked.windows(2) {
        assert_eq!(pair[0].score, pair[1].score);
    }
    assert_eq!(
        index
            .top_k(&prepared, 2)
            .iter()
            .map(|s| s.slot)
            .collect::<Vec<_>>(),
        [10, 20]
    );
}

#[test]
fn top_k_never_returns_a_document_that_shares_no_term() {
    let index = TextIndex::build([(0u32, "alpha"), (1u32, "beta"), (2u32, "gamma")]);
    let prepared = index.prepare_query("alpha");
    let ranked = index.top_k(&prepared, 10);
    assert_eq!(ranked.len(), 1, "k larger than the candidate set");
    assert_eq!(ranked[0].slot, 0);
    assert!(index.top_k(&index.prepare_query("absent"), 10).is_empty());
    assert!(index.top_k(&prepared, 0).is_empty());
}

#[test]
fn repeating_a_query_term_does_not_change_the_ranking() {
    let index = TextIndex::build([(0u32, "alpha beta"), (1u32, "beta beta")]);
    let once = index.top_k(&index.prepare_query("beta alpha"), 2);
    let thrice = index.top_k(&index.prepare_query("beta beta alpha beta"), 2);
    assert_eq!(
        once.iter().map(|s| (s.slot, s.score)).collect::<Vec<_>>(),
        thrice.iter().map(|s| (s.slot, s.score)).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Randomized CRUD
// ---------------------------------------------------------------------------

#[test]
fn randomized_crud_stays_identical_to_a_fresh_rebuild() {
    let mut rng = Rng(0x5EED_1234_ABCD_9876);
    let mut index = TextIndex::new();
    let mut model: BTreeMap<u32, String> = BTreeMap::new();
    let queries = ["alpha", "beta gamma", "tromsø 日本", "delta epsilon zeta"];

    for step in 0..500 {
        let slot = rng.below(24) as u32;
        if rng.below(10) < 6 {
            let text = random_text(&mut rng);
            index.add_doc(slot, &text);
            model.insert(slot, text);
        } else {
            assert_eq!(index.remove_doc(slot), model.remove(&slot).is_some());
        }
        index
            .validate()
            .unwrap_or_else(|why| panic!("step {step}: {why}"));

        if step % 25 != 0 {
            continue;
        }
        let fresh = TextIndex::build(model.iter().map(|(slot, text)| (*slot, text)));
        assert_eq!(snapshot(&index), snapshot(&fresh), "step {step}");
        for query in queries {
            let live = index.prepare_query(query);
            let rebuilt = fresh.prepare_query(query);
            for slot in 0..24u32 {
                assert_eq!(
                    index.score(slot, &live),
                    fresh.score(slot, &rebuilt),
                    "step {step}, query {query:?}, slot {slot}"
                );
            }
            for scored in index.top_k(&live, 5) {
                assert!(
                    model.contains_key(&scored.slot),
                    "step {step}: deleted slot {} scored {}",
                    scored.slot,
                    scored.score
                );
            }
        }
    }
    assert!(
        !model.is_empty(),
        "the sequence exercised more than deletes"
    );
}

#[test]
fn randomized_crud_matches_the_naive_reference() {
    let mut rng = Rng(0x0BAD_C0DE_1122_3344);
    let mut index = TextIndex::new();
    let mut model: BTreeMap<u32, String> = BTreeMap::new();
    for _ in 0..120 {
        let slot = rng.below(16) as u32;
        if rng.below(10) < 7 {
            let text = random_text(&mut rng);
            index.add_doc(slot, &text);
            model.insert(slot, text);
        } else {
            index.remove_doc(slot);
            model.remove(&slot);
        }
    }
    index.validate().unwrap();
    let docs: Vec<(u32, String)> = model.into_iter().collect();
    for query in ["alpha beta", "tromsø", "日本 zeta", "gamma"] {
        let prepared = index.prepare_query(query);
        let expected = reference_scores(&docs, query);
        for (slot, want) in &expected {
            assert_eq!(index.score(*slot, &prepared), *want, "query {query:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Memory shape
// ---------------------------------------------------------------------------

#[test]
fn memory_stays_far_under_a_hundred_bytes_per_token() {
    // A pathological structural blow-up (a per-document hash map, a term string
    // stored once per occurrence) is invisible in a five-document unit test and
    // fatal on a million-document build. 100 bytes/token is generous — the
    // measured figure is far below it — so this fails on a design regression,
    // not on allocator noise.
    let mut rng = Rng(0xFACE_FEED_0001_0002);
    let docs: Vec<(u32, String)> = (0..2_000u32)
        .map(|slot| {
            let text = (0..40)
                .map(|_| format!("w{}", rng.below(5_000)))
                .collect::<Vec<_>>()
                .join(" ");
            (slot, text)
        })
        .collect();
    let tokens: usize = docs.len() * 40;
    let index = TextIndex::build(docs.iter().map(|(slot, text)| (*slot, text)));
    index.validate().unwrap();

    let bytes = index.estimated_bytes();
    let per_token = bytes as f64 / tokens as f64;
    assert!(
        per_token < 100.0,
        "{bytes} bytes for {tokens} tokens = {per_token:.1} bytes/token"
    );
    assert!(bytes > tokens, "the estimate is not counting the postings");
}

// ---------------------------------------------------------------------------
// Absolute values
// ---------------------------------------------------------------------------

#[test]
fn kernel_constants_and_formula_match_hand_computed_bm25() {
    // The reference oracle imports K1 and B from the kernel, so it cannot see a
    // change to either — nor to the shape of the IDF. These numbers were worked
    // out by hand from the Okapi formula in the module docs (N=2, avgdl=1.5)
    // and are the only assertion in this file that pins the parameters.
    let index = TextIndex::build([(0u32, "a b"), (1u32, "a")]);
    assert_eq!(index.avgdl(), 1.5);

    let common = index.prepare_query("a"); // df = 2 of 2
    assert!((super::bm25::idf(2, 2) - 0.182_321_556_793_954_6).abs() < 1e-15);
    assert!((index.score(0, &common) - 0.160_442_969_978_680_07).abs() < 1e-15);
    assert!((index.score(1, &common) - 0.211_109_171_024_579_05).abs() < 1e-15);

    // "b" has df = 1 of 2, where the smoothed IDF is ln(1 + 1.5/1.5) = ln 2.
    let rare = index.prepare_query("b");
    assert!((super::bm25::idf(2, 1) - std::f64::consts::LN_2).abs() < 1e-15);
    assert!((index.score(0, &rare) - 0.609_969_518_892_751_9).abs() < 1e-15);
    assert_eq!(index.score(1, &rare), 0.0);

    // The rarer term outranks the common one on the same document: IDF is doing
    // the stopword work that v1 ships no stopword list for.
    assert!(index.score(0, &rare) > index.score(0, &common));
}

#[test]
fn top_k_huge_limit_bounds_capacity_by_actual_candidates() {
    let docs = corpus(&[(0, "needle"), (1, "needle"), (2, "other")]);
    let index = TextIndex::build(docs.iter().map(|(slot, text)| (*slot, text)));
    let query = index.prepare_query("needle");
    let expected = sorted_by_rank(&index, &query);
    assert_eq!(
        expected.iter().map(|hit| hit.slot).collect::<Vec<_>>(),
        vec![0, 1]
    );
    // usize::MAX first makes the pre-fix debug failure a checked k+1 overflow,
    // before any attempt to allocate an enormous buffer.
    for limit in [usize::MAX, i64::MAX as usize] {
        let hits = index.top_k(&query, limit);
        assert_eq!(
            hits.iter()
                .map(|hit| (hit.slot, hit.score))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|hit| (hit.slot, hit.score))
                .collect::<Vec<_>>()
        );
    }
}
