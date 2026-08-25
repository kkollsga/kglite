//! The Okapi BM25 kernel over a [`TextIndex`].
//!
//! For a query `Q` and document `D` in a corpus of `N` documents:
//!
//! ```text
//! score(D, Q) = Σ_{t ∈ Q}  idf(t) · ( f(t,D) · (k1 + 1) )
//!                          / ( f(t,D) + k1 · (1 − b + b · |D| / avgdl) )
//!
//! idf(t)      = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )
//! ```
//!
//! `f(t,D)` is the term's frequency in `D`, `|D|` its token count, `avgdl` the
//! corpus mean. [`K1`] and [`B`] carry the reasoning for their values.
//!
//! The smoothed IDF (the `1 +` inside the logarithm) is the form Lucene and
//! every modern BM25 implementation use; the textbook `ln((N − df + 0.5)/(df +
//! 0.5))` goes *negative* for a term present in more than half the corpus, so a
//! document could be penalised for containing a query word.

use super::{TermId, TextIndex};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Term-frequency saturation. 1.2 is the standard Okapi value: a term's
/// contribution grows quickly for the first few occurrences and then flattens,
/// so a document repeating a word 100 times does not out-rank one that uses it
/// meaningfully five times.
pub const K1: f64 = 1.2;

/// Length-normalization strength, in `[0, 1]`. 0.75 is the standard Okapi
/// value: mostly normalize by length (so long documents do not win by sheer
/// size) while still crediting a long document that genuinely repeats the term.
/// `b = 0` disables length normalization entirely; `b = 1` applies it fully.
pub const B: f64 = 0.75;

/// Inverse document frequency of a term appearing in `df` of `total_docs`
/// documents. Always positive for `df ≤ total_docs`, and `df = 0` is the
/// caller's job to exclude (an unknown term cannot occur in any document, so it
/// would contribute nothing anyway).
pub fn idf(total_docs: usize, df: usize) -> f64 {
    let n = total_docs as f64;
    let df = df as f64;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// One resolved query term: its interned id and the IDF it carries for the
/// corpus state the query was prepared against.
#[derive(Clone, Copy, Debug)]
pub struct QueryTerm {
    pub term: TermId,
    pub idf: f64,
}

/// A tokenized, resolved query — the reusable half of scoring.
///
/// Preparing costs one tokenization plus one dictionary lookup and one IDF per
/// distinct term; scoring a row then costs one binary search per term. A
/// per-row scalar prepares once and scores many rows.
///
/// The IDFs are a *snapshot*: a query prepared before documents are added keeps
/// the older corpus statistics. Prepare against the index you are about to
/// score with.
#[derive(Clone, Debug, Default)]
pub struct PreparedQuery {
    terms: Vec<QueryTerm>,
}

impl PreparedQuery {
    pub fn terms(&self) -> &[QueryTerm] {
        &self.terms
    }

    /// True when no query token exists in the corpus — every document scores 0
    /// and [`TextIndex::top_k`] returns nothing.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

/// A document and its score, ordered best-first by
/// **score descending, then slot ascending**.
///
/// The slot tiebreak is not cosmetic: without it, two documents with identical
/// scores would be ranked by hash-map iteration order, and a query would return
/// different rows on different runs over the same data.
#[derive(Clone, Copy, Debug)]
pub struct ScoredDoc {
    pub slot: u32,
    pub score: f64,
}

impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ScoredDoc {}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDoc {
    /// **Reverse ranking order**: `Greater` means *worse*. That makes a
    /// `BinaryHeap<ScoredDoc>` a min-heap on rank, whose root is the weakest
    /// survivor — exactly what a bounded top-k needs to evict — and makes
    /// `into_sorted_vec()` come out best-first.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.slot.cmp(&other.slot))
    }
}

impl TextIndex {
    /// Tokenize and resolve `query` against this index's dictionary.
    ///
    /// Unknown terms are dropped (they can occur in no document). Repeats are
    /// dropped too — v1 weights a query term once however often the query says
    /// it, so `"rust rust rust"` ranks exactly like `"rust"`. Surviving terms
    /// keep their first-occurrence order, which fixes the summation order and
    /// therefore makes the floating-point score reproducible.
    pub fn prepare_query(&self, query: &str) -> PreparedQuery {
        let total = self.total_docs();
        let mut terms: Vec<QueryTerm> = Vec::new();
        for token in super::analyze(query) {
            let Some(term) = self.term_id(&token) else {
                continue;
            };
            if terms.iter().any(|seen| seen.term == term) {
                continue;
            }
            terms.push(QueryTerm {
                term,
                idf: idf(total, self.postings_of(term).len()),
            });
        }
        PreparedQuery { terms }
    }

    /// BM25 score of one document. `0.0` for an unindexed slot or a document
    /// sharing no term with the query — "no evidence" and "no match" are the
    /// same answer to this function; the caller decides whether an unindexed
    /// row means null.
    pub fn score(&self, slot: u32, query: &PreparedQuery) -> f64 {
        let Some(doc) = self.docs.get(&slot) else {
            return 0.0;
        };
        let avgdl = self.avgdl();
        if avgdl <= 0.0 {
            // Every document is empty, so every term frequency is 0 anyway;
            // returning early keeps the 0/0 out of the normalization.
            return 0.0;
        }
        let norm = K1 * (1.0 - B + B * (f64::from(doc.len) / avgdl));
        let mut total = 0.0;
        for term in &query.terms {
            let tf = doc.term_freq(term.term);
            if tf == 0 {
                continue;
            }
            let tf = f64::from(tf);
            total += term.idf * (tf * (K1 + 1.0)) / (tf + norm);
        }
        total
    }

    /// The `k` best-scoring documents, best first.
    ///
    /// Postings drive *candidate generation* only — a document sharing no term
    /// with the query scores 0 and is never returned, so the whole corpus is
    /// never touched. Each candidate is then scored through [`TextIndex::score`]
    /// rather than by accumulating per-term contributions across the postings,
    /// which keeps the summation order identical to the per-row path: the same
    /// document gets bit-identical scores whether it arrives here or through
    /// the scalar. Accumulating instead would reorder the additions and produce
    /// scores that differ in the last ulp — enough to reorder near-ties.
    ///
    /// Returns fewer than `k` when fewer documents match.
    pub fn top_k(&self, query: &PreparedQuery, k: usize) -> Vec<ScoredDoc> {
        if k == 0 || query.is_empty() {
            return Vec::new();
        }
        let mut candidates: Vec<u32> = Vec::new();
        for term in &query.terms {
            candidates.extend(self.postings_of(term.term).iter().map(|p| p.slot));
        }
        candidates.sort_unstable();
        candidates.dedup();

        let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::with_capacity(k + 1);
        for slot in candidates {
            heap.push(ScoredDoc {
                slot,
                score: self.score(slot, query),
            });
            if heap.len() > k {
                heap.pop();
            }
        }
        heap.into_sorted_vec()
    }
}
