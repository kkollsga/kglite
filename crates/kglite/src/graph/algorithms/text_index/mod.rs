//! Lexical (BM25) text index — the self-contained core.
//!
//! This module is deliberately decoupled from the graph, the way
//! [`hnsw`](super::hnsw) is: a document is just a **slot** (`u32`) that the
//! caller assigns, so nothing here knows about `NodeIndex`, `DirGraph`, or
//! persistence. The caller owns the slot ↔ node mapping and the decision of
//! which property is indexed.
//!
//! # Two views, on purpose
//!
//! The index keeps both directions:
//!
//! * **Postings** — term → `[(slot, term_frequency)]`, sorted by slot. Drives
//!   candidate generation for [`TextIndex::top_k`].
//! * **Forward** — slot → `[(term, term_frequency)]` + document length. Drives
//!   the per-row score (`text_bm25(n.prop, 'query')` scores *one* row the
//!   planner already chose) and makes deletion **exact**: removing a document
//!   needs to know precisely which postings mention it, and scanning the whole
//!   postings map to find out would make deletes O(vocabulary). A stale posting
//!   left behind is not a slow query, it is a wrong answer — the slot can be
//!   handed to a different node later (petgraph reuses indices), and the ghost
//!   content would score.
//!
//! Both views cost roughly 2× a single-direction index; that is the accepted
//! trade (`release-train-0-16-10.md`, decision 2).
//!
//! # No stemming, no stopword list
//!
//! Configurable analyzer chains (Lucene, Tantivy, Neo4j's FULLTEXT) let you
//! bolt a stemmer and a stopword list onto the tokenizer. v1 ships neither,
//! which is a deliberate divergence rather than an omission: BM25's IDF term
//! already handles stopwords *statistically* — a term appearing in nearly
//! every document gets an IDF near zero and contributes nearly nothing, with
//! no language-specific list to maintain or to be wrong about. Stemming has
//! the opposite property: it is language-specific, it is lossy (`business` and
//! `busy` collapse under Porter), and a user who wanted the exact word cannot
//! undo it at query time. See [`analyzer`] for the tokenizer contract.
//!
//! # Term identity
//!
//! Terms are interned to a `u32` [`TermId`]. The forward view stores ids, not
//! strings, so a term appearing in 10 000 documents is stored once rather than
//! 10 000 times. Ids are an *internal* detail — two indexes over the same
//! corpus built in different orders assign different ids and are still
//! logically identical (and score identically); compare them through
//! [`TextIndex::iter_terms`], never by id.
//!
//! # Memory
//!
//! Roughly **16 bytes per (term, document) pair** across the two views (8
//! each), plus one dictionary entry per *distinct corpus term* (~60 bytes plus
//! the term's own bytes, counted once however many documents use it), plus ~70
//! bytes of per-document bookkeeping. Measured on the synthetic corpus in the
//! tests — 2 000 documents of 40 tokens drawn from a 5 000-term vocabulary,
//! which is close to worst-case because almost every token is a distinct term
//! — that is **26 bytes per token** (2.08 MB for 80 000 tokens). Real prose
//! repeats words within a document, which collapses two entries into one with a
//! higher frequency, so it lands lower.
//!
//! [`TextIndex::estimated_bytes`] reports a live estimate, and a test holds
//! that synthetic ratio under 100 bytes/token so a structural blow-up (a hash
//! map per document, a term string stored once per occurrence) fails here
//! rather than on someone's million-document build.

pub mod analyzer;
pub mod bm25;

pub use analyzer::analyze;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Interned term identity. Internal to one index — see the module docs.
pub type TermId = u32;

/// One document's appearance in a term's posting list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Posting {
    pub slot: u32,
    /// How many times the term occurs in that document. Never zero: a term
    /// that stops occurring loses its posting.
    pub tf: u32,
}

/// The forward view of one document.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Doc {
    /// `(term id, term frequency)`, sorted by term id so a query term is found
    /// by binary search without a per-document hash map.
    terms: Vec<(TermId, u32)>,
    /// Token count — the `|D|` of BM25's length normalization. Equals the sum
    /// of `terms`' frequencies; stored so scoring never has to fold.
    len: u32,
}

impl Doc {
    #[inline]
    fn term_freq(&self, term: TermId) -> u32 {
        match self.terms.binary_search_by_key(&term, |&(id, _)| id) {
            Ok(at) => self.terms[at].1,
            Err(_) => 0,
        }
    }
}

/// A BM25 lexical index over caller-assigned document slots.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TextIndex {
    /// Term string → id.
    ids: FxHashMap<Arc<str>, TermId>,
    /// Id → term string, `None` at a freed id. Shares its allocation with the
    /// `ids` key, so a term's bytes are stored once despite the two directions.
    names: Vec<Option<Arc<str>>>,
    /// Id → postings, sorted by slot. Empty at a freed id.
    postings: Vec<Vec<Posting>>,
    /// Ids whose term was dropped when its last posting went away.
    free_ids: Vec<TermId>,
    /// Slot → forward view.
    docs: FxHashMap<u32, Doc>,
    /// Σ document length, so `avgdl` needs no fold.
    total_len: u64,
}

impl TextIndex {
    /// Empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bulk build. Postings are appended per document and each list is sorted
    /// once at the end, rather than insertion-sorted per document as
    /// [`TextIndex::add_doc`] must do — a distinct code path from the
    /// incremental one on purpose, so the "incremental == rebuild" invariant
    /// the refresh contract rests on is a real assertion and not a tautology.
    ///
    /// A repeated slot keeps its last text (the same upsert rule `add_doc` has).
    pub fn build<I, S>(docs: I) -> Self
    where
        I: IntoIterator<Item = (u32, S)>,
        S: AsRef<str>,
    {
        let mut index = Self::new();
        for (slot, text) in docs {
            index.remove_doc(slot);
            let doc = index.intern_document(text.as_ref());
            for &(term, tf) in &doc.terms {
                index.postings[term as usize].push(Posting { slot, tf });
            }
            index.total_len += u64::from(doc.len);
            index.docs.insert(slot, doc);
        }
        for list in &mut index.postings {
            list.sort_unstable_by_key(|posting| posting.slot);
        }
        index
    }

    /// Rebuild an index from the id-independent view [`TextIndex::iter_terms`]
    /// produces, plus the slots whose document holds no terms.
    ///
    /// The inverse of that view, and the only constructor that does not
    /// tokenize — which is what makes it the right shape for persistence: term
    /// ids are internal, so a `.kgl` section records terms by *name* and this
    /// re-assigns ids in arrival order. The result is logically the index that
    /// was written and scores identically to it, and it is *compacted*: the
    /// free list a churned index accumulates does not survive the round-trip,
    /// because nothing outside the index can observe an id.
    ///
    /// Empty documents have to be named separately: they appear in no posting
    /// list, and losing them would change BM25's `N` and therefore every score
    /// in the corpus.
    ///
    /// **Trusts nothing.** A caller reconstructing from bytes runs
    /// [`TextIndex::validate`] on the result and discards a payload that does
    /// not describe a coherent index — duplicate terms, a posting list out of
    /// order, or a document length that overflowed all surface there.
    pub fn from_terms<S: AsRef<str>>(
        terms: impl IntoIterator<Item = (S, Vec<Posting>)>,
        empty_docs: &[u32],
    ) -> Self {
        let mut index = Self::new();
        for (name, postings) in terms {
            // A term with no postings is not live — `release_term` retires one
            // the moment its last posting goes — so interning it here would
            // break the id-space invariant `validate` checks.
            if postings.is_empty() {
                continue;
            }
            let name: Arc<str> = Arc::from(name.as_ref());
            let id = index.names.len() as TermId;
            index.names.push(Some(Arc::clone(&name)));
            index.ids.insert(name, id);
            index.postings.push(postings);
        }
        // The forward view is derived rather than carried: it is exactly the
        // transpose of the postings, so persisting it too would store every
        // (term, document) pair twice and let the two copies disagree.
        let mut docs: FxHashMap<u32, Doc> = FxHashMap::default();
        for (id, list) in index.postings.iter().enumerate() {
            for posting in list {
                let doc = docs.entry(posting.slot).or_default();
                doc.terms.push((id as TermId, posting.tf));
                // Saturating rather than wrapping: a corrupt length has to stay
                // *visibly* wrong so `validate` refuses it, and a wrapped one
                // could coincide with the true sum.
                doc.len = doc.len.saturating_add(posting.tf);
            }
        }
        for &slot in empty_docs {
            docs.entry(slot).or_default();
        }
        index.total_len = docs.values().map(|doc| u64::from(doc.len)).sum();
        index.docs = docs;
        index
    }

    /// Index `text` under `slot`, replacing whatever that slot held.
    ///
    /// This is the whole refresh surface: a node created after the build and a
    /// node whose indexed property was overwritten are both just `add_doc`.
    /// Corpus statistics (`N`, `avgdl`, every term's document frequency) shift
    /// as documents arrive, so a document's score legitimately changes when
    /// *other* documents are added — that is BM25 working, not drift. Scores
    /// are comparable only within one query against one corpus state.
    pub fn add_doc(&mut self, slot: u32, text: &str) {
        self.remove_doc(slot);
        let doc = self.intern_document(text);
        for &(term, tf) in &doc.terms {
            let list = &mut self.postings[term as usize];
            let at = list.partition_point(|posting| posting.slot < slot);
            list.insert(at, Posting { slot, tf });
        }
        self.total_len += u64::from(doc.len);
        self.docs.insert(slot, doc);
    }

    /// Drop `slot` and every posting that mentions it. Returns whether the slot
    /// was indexed.
    ///
    /// Exact, via the forward view: no tombstones, no lazy compaction, no
    /// window in which a removed document can still be scored or counted.
    pub fn remove_doc(&mut self, slot: u32) -> bool {
        let Some(doc) = self.docs.remove(&slot) else {
            return false;
        };
        self.total_len -= u64::from(doc.len);
        for &(term, _) in &doc.terms {
            let list = &mut self.postings[term as usize];
            // The linear fallback is not defensive padding: `build` appends
            // postings unsorted and sorts once at the end, so a repeated slot
            // is removed from a list the binary search cannot navigate. A
            // successful search is right either way — it only ever returns an
            // index whose slot matches.
            let at = list
                .binary_search_by_key(&slot, |posting| posting.slot)
                .ok()
                .or_else(|| list.iter().position(|posting| posting.slot == slot));
            if let Some(at) = at {
                list.remove(at);
            }
            if list.is_empty() {
                self.release_term(term);
            }
        }
        true
    }

    /// Tokenize `text` into a forward-view entry, interning new terms.
    fn intern_document(&mut self, text: &str) -> Doc {
        let mut counts: FxHashMap<TermId, u32> = FxHashMap::default();
        let mut len: u32 = 0;
        for token in analyze(text) {
            let term = self.intern_term(&token);
            *counts.entry(term).or_insert(0) += 1;
            len += 1;
        }
        let mut terms: Vec<(TermId, u32)> = counts.into_iter().collect();
        terms.sort_unstable_by_key(|&(id, _)| id);
        Doc { terms, len }
    }

    fn intern_term(&mut self, token: &str) -> TermId {
        if let Some(&id) = self.ids.get(token) {
            return id;
        }
        let name: Arc<str> = Arc::from(token);
        let id = match self.free_ids.pop() {
            Some(id) => {
                self.names[id as usize] = Some(Arc::clone(&name));
                id
            }
            None => {
                self.names.push(Some(Arc::clone(&name)));
                self.postings.push(Vec::new());
                (self.postings.len() - 1) as TermId
            }
        };
        self.ids.insert(name, id);
        id
    }

    /// Retire a term whose last posting is gone: its dictionary entry goes and
    /// its id returns to the free list, so the vocabulary does not grow forever
    /// under create/delete churn.
    fn release_term(&mut self, term: TermId) {
        if let Some(name) = self.names[term as usize].take() {
            self.ids.remove(&name);
            self.free_ids.push(term);
        }
    }

    /// Documents in the corpus — BM25's `N`. Counts empty documents, which do
    /// participate in the corpus statistics.
    pub fn total_docs(&self) -> usize {
        self.docs.len()
    }

    /// Whether the corpus is empty — the query-side short circuit.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Distinct terms currently indexed.
    pub fn vocabulary_len(&self) -> usize {
        self.ids.len()
    }

    pub fn contains_doc(&self, slot: u32) -> bool {
        self.docs.contains_key(&slot)
    }

    /// Token count of an indexed document, or `None` if the slot is unindexed.
    pub fn doc_len(&self, slot: u32) -> Option<u32> {
        self.docs.get(&slot).map(|doc| doc.len)
    }

    /// Mean document length. `0.0` for an empty corpus.
    pub fn avgdl(&self) -> f64 {
        if self.docs.is_empty() {
            return 0.0;
        }
        self.total_len as f64 / self.docs.len() as f64
    }

    /// How many documents contain `term`.
    ///
    /// Derived from the posting list's length rather than kept in a parallel
    /// counter: removal is exact, so a posting list never holds a dead entry
    /// and its length *is* the document frequency. A second copy of the number
    /// could only ever disagree with this one.
    // P12 reports term statistics through this; no production caller yet.
    #[allow(dead_code)]
    pub fn df(&self, term: &str) -> usize {
        self.postings_for(term).len()
    }

    /// Interned id of an already-known term, if any. Query preparation resolves
    /// terms once through this and then works in ids.
    pub fn term_id(&self, term: &str) -> Option<TermId> {
        self.ids.get(term).copied()
    }

    /// Postings for `term`, sorted by slot; empty when the term is unknown.
    // P12's by-name posting lookup; the id-keyed `postings_of` is what scoring uses today.
    #[allow(dead_code)]
    pub fn postings_for(&self, term: &str) -> &[Posting] {
        match self.ids.get(term) {
            Some(&id) => &self.postings[id as usize],
            None => &[],
        }
    }

    /// Postings by interned id, sorted by slot.
    pub fn postings_of(&self, term: TermId) -> &[Posting] {
        self.postings.get(term as usize).map_or(&[], Vec::as_slice)
    }

    /// Occurrences of `term` in `slot`; `0` if either is absent.
    // P11 serializes the forward view through this; no production caller yet.
    #[allow(dead_code)]
    pub fn term_freq(&self, slot: u32, term: TermId) -> u32 {
        self.docs.get(&slot).map_or(0, |doc| doc.term_freq(term))
    }

    /// Every indexed slot, in arbitrary order.
    pub fn doc_slots(&self) -> impl Iterator<Item = u32> + '_ {
        self.docs.keys().copied()
    }

    /// Every live term with its postings, in id order. This is the logical
    /// content of the index — id-independent, so it is the right basis for
    /// comparing two indexes and for serializing one.
    pub fn iter_terms(&self) -> impl Iterator<Item = (&str, &[Posting])> + '_ {
        self.names
            .iter()
            .enumerate()
            .filter_map(|(id, name)| Some((name.as_deref()?, self.postings[id].as_slice())))
    }

    /// Approximate heap footprint in bytes: allocation sizes plus a load-factor
    /// allowance for the two hash tables. Not exact — it cannot see allocator
    /// rounding — but tight enough to catch a structural blow-up.
    pub fn estimated_bytes(&self) -> usize {
        const TABLE_SLACK: usize = 8;
        let dictionary: usize = self
            .ids
            .keys()
            .map(|name| {
                // Arc<str> payload = 2 refcounts + the bytes; the id table and
                // the `names` vector each hold a fat pointer to it.
                name.len()
                    + 2 * std::mem::size_of::<usize>()
                    + 2 * std::mem::size_of::<Arc<str>>()
                    + std::mem::size_of::<TermId>()
                    + TABLE_SLACK
            })
            .sum();
        let postings: usize = self
            .postings
            .iter()
            .map(|list| {
                list.capacity() * std::mem::size_of::<Posting>() + std::mem::size_of_val(list)
            })
            .sum();
        let forward: usize = self
            .docs
            .values()
            .map(|doc| {
                doc.terms.capacity() * std::mem::size_of::<(TermId, u32)>()
                    + std::mem::size_of::<(u32, Doc)>()
                    + TABLE_SLACK
            })
            .sum();
        dictionary + postings + forward + self.free_ids.capacity() * std::mem::size_of::<TermId>()
    }

    /// Check every internal invariant and describe the first breach.
    ///
    /// The two views are redundant by construction, which is what makes them
    /// checkable: a mutation bug shows up as a disagreement between them long
    /// before it shows up as a wrong score. Tests call this after every
    /// mutation sequence.
    pub fn validate(&self) -> Result<(), String> {
        let mut expected_total: u64 = 0;
        let mut expected_postings: usize = 0;
        for (&slot, doc) in &self.docs {
            expected_total += u64::from(doc.len);
            expected_postings += doc.terms.len();
            self.validate_doc(slot, doc)?;
        }
        if expected_total != self.total_len {
            return Err(format!(
                "total_len {} != Σ doc lengths {expected_total}",
                self.total_len
            ));
        }
        self.validate_postings(expected_postings)
    }

    /// Forward-view half of [`TextIndex::validate`].
    fn validate_doc(&self, slot: u32, doc: &Doc) -> Result<(), String> {
        let mut sum: u64 = 0;
        let mut previous: Option<TermId> = None;
        for &(term, tf) in &doc.terms {
            if tf == 0 {
                return Err(format!("doc {slot} keeps a zero-frequency term {term}"));
            }
            if previous.is_some_and(|last| last >= term) {
                return Err(format!("doc {slot} terms are not strictly ascending"));
            }
            previous = Some(term);
            sum += u64::from(tf);
            if self.names[term as usize].is_none() {
                return Err(format!("doc {slot} references freed term id {term}"));
            }
            let list = self.postings_of(term);
            match list.binary_search_by_key(&slot, |posting| posting.slot) {
                Ok(at) if list[at].tf == tf => {}
                Ok(at) => {
                    return Err(format!(
                        "doc {slot} term {term}: forward tf {tf} != posting tf {}",
                        list[at].tf
                    ))
                }
                Err(_) => return Err(format!("doc {slot} term {term} has no posting")),
            }
        }
        if sum != u64::from(doc.len) {
            return Err(format!("doc {slot} length {} != Σtf {sum}", doc.len));
        }
        Ok(())
    }

    /// Postings-side half of [`TextIndex::validate`]: every posting is backed
    /// by a live forward entry, lists are sorted and duplicate-free, and the id
    /// space (dictionary + free list) has neither overlap nor leak.
    fn validate_postings(&self, expected_postings: usize) -> Result<(), String> {
        let mut seen: usize = 0;
        for (id, name) in self.names.iter().enumerate() {
            let Some(name) = name else { continue };
            let list = &self.postings[id];
            if list.is_empty() {
                return Err(format!("term '{name}' is interned with no postings"));
            }
            if self.ids.get(name.as_ref()) != Some(&(id as TermId)) {
                return Err(format!("term '{name}' does not resolve back to id {id}"));
            }
            seen += list.len();
            let mut previous: Option<u32> = None;
            for posting in list {
                if previous.is_some_and(|last| last >= posting.slot) {
                    return Err(format!("term '{name}' postings are not strictly ascending"));
                }
                previous = Some(posting.slot);
                match self.docs.get(&posting.slot) {
                    Some(doc) if doc.term_freq(id as TermId) == posting.tf => {}
                    Some(_) => {
                        return Err(format!(
                            "term '{name}' posting for slot {} disagrees with the forward view",
                            posting.slot
                        ))
                    }
                    None => {
                        return Err(format!(
                            "term '{name}' keeps a posting for unindexed slot {}",
                            posting.slot
                        ))
                    }
                }
            }
        }
        if seen != expected_postings {
            return Err(format!(
                "postings hold {seen} entries, the forward view {expected_postings}"
            ));
        }
        self.validate_id_space()
    }

    /// Dictionary, `names`, `postings` and the free list must describe one id
    /// space: no id both live and free, no id neither, no leaked slot.
    fn validate_id_space(&self) -> Result<(), String> {
        if self.names.len() != self.postings.len() {
            return Err(format!(
                "{} term names but {} posting lists",
                self.names.len(),
                self.postings.len()
            ));
        }
        for &id in &self.free_ids {
            if self.names[id as usize].is_some() {
                return Err(format!("term id {id} is both live and free"));
            }
            if !self.postings[id as usize].is_empty() {
                return Err(format!("freed term id {id} still has postings"));
            }
        }
        let live = self.names.iter().filter(|name| name.is_some()).count();
        if live != self.ids.len() {
            return Err(format!(
                "{live} named ids but {} dictionary entries",
                self.ids.len()
            ));
        }
        if live + self.free_ids.len() != self.names.len() {
            return Err(format!(
                "id space leak: {live} live + {} free != {} slots",
                self.free_ids.len(),
                self.names.len()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
