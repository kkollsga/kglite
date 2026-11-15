//! BM25 over a corpus that exists only for one query: the documents of a
//! graph-selected scope, with no index behind them.
//!
//! The statistics are the scope's own: `N` is the number of documents added,
//! the mean length is theirs, and a term's document frequency counts the
//! documents holding it as an analyzer token. Scoring goes through
//! [`score_document`] and [`idf`], the functions the heap index scores with,
//! so the answer equals building a [`TextIndex`](super::TextIndex) over just
//! these documents and asking for its top-k: same scores to the bit, same
//! order (score descending, slot ascending).
//!
//! One pass tokenizes each document once: it counts the document's length and
//! the query terms' frequencies together. Documents sharing no query term are
//! counted in `N` and the length total and then forgotten; only the others
//! keep their frequencies until the statistics are final.

use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;

use super::analyze;
use super::bm25::{idf, score_document, ScoredDoc};

/// One document's token count and the frequencies of the query's terms in it
/// (`None` when it holds none of them).
pub(crate) struct DocCount {
    slot: u32,
    len: u32,
    freq: Option<Vec<u32>>,
}

pub(crate) struct ScopedBm25 {
    /// The query's distinct terms in first-occurrence order, the order the
    /// heap index sums in.
    terms: Vec<String>,
    lookup: FxHashMap<String, usize>,
    documents: u64,
    total_len: u64,
    /// Documents holding each term.
    df: Vec<usize>,
    /// `(slot, token count)` of each document that holds a query term.
    candidates: Vec<(u32, u32)>,
    /// `terms.len()` frequencies per candidate, flat.
    tf: Vec<u32>,
}

impl ScopedBm25 {
    pub(crate) fn new(query: &str) -> Self {
        let mut terms: Vec<String> = Vec::new();
        let mut lookup = FxHashMap::default();
        for token in analyze(query) {
            if !lookup.contains_key(token.as_ref()) {
                lookup.insert(token.to_string(), terms.len());
                terms.push(token.into_owned());
            }
        }
        let df = vec![0; terms.len()];
        Self {
            terms,
            lookup,
            documents: 0,
            total_len: 0,
            df,
            candidates: Vec::new(),
            tf: Vec::new(),
        }
    }

    /// Whether the query has no term, so nothing can score.
    pub(crate) fn no_terms(&self) -> bool {
        self.terms.is_empty()
    }

    /// Add one document to the corpus. Each slot is added at most once.
    pub(crate) fn add_document(&mut self, slot: u32, text: &str) {
        let counted = self.count(slot, text);
        self.absorb(counted);
    }

    /// Tokenize one document against the query's terms. Reads only the query,
    /// so documents may be counted on several threads and then
    /// [`absorb`](Self::absorb)ed in order: the statistics and the scores do
    /// not depend on which thread counted what.
    pub(crate) fn count(&self, slot: u32, text: &str) -> DocCount {
        let width = self.terms.len();
        let mut len: u32 = 0;
        let mut freq: Option<Vec<u32>> = None;
        for token in analyze(text) {
            len += 1;
            if let Some(&at) = self.lookup.get(token.as_ref()) {
                freq.get_or_insert_with(|| vec![0u32; width])[at] += 1;
            }
        }
        DocCount { slot, len, freq }
    }

    /// Fold one counted document into the corpus. Call in document order.
    pub(crate) fn absorb(&mut self, doc: DocCount) {
        self.documents += 1;
        self.total_len += u64::from(doc.len);
        let Some(freq) = doc.freq else { return };
        for (df, &f) in self.df.iter_mut().zip(&freq) {
            *df += usize::from(f > 0);
        }
        self.candidates.push((doc.slot, doc.len));
        self.tf.extend_from_slice(&freq);
    }

    /// The best `limit` documents sharing a term with the query (all of them
    /// without a limit), best first.
    pub(crate) fn top(&self, limit: Option<usize>) -> Vec<ScoredDoc> {
        if limit == Some(0) || self.candidates.is_empty() {
            return Vec::new();
        }
        let n = self.documents as usize;
        let avgdl = self.total_len as f64 / self.documents as f64;
        let idfs: Vec<f64> = self.df.iter().map(|&df| idf(n, df)).collect();
        let width = self.terms.len();
        let mut heap: BinaryHeap<ScoredDoc> = BinaryHeap::new();
        for (i, &(slot, len)) in self.candidates.iter().enumerate() {
            let freqs = &self.tf[i * width..(i + 1) * width];
            let score = score_document(avgdl, len, idfs.iter().copied().zip(freqs.iter().copied()));
            heap.push(ScoredDoc { slot, score });
            if limit.is_some_and(|k| heap.len() > k) {
                heap.pop();
            }
        }
        heap.into_sorted_vec()
    }
}
