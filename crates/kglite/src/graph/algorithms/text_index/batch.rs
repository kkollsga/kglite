use super::{Posting, TermId, TextIndex};
use rustc_hash::{FxHashMap, FxHashSet};

impl TextIndex {
    /// Replace unique slots, merging each affected posting list once. `None`
    /// removes a document; an empty string remains an indexed document.
    ///
    /// Callers must supply each slot once (FreshnessDelta guarantees this).
    /// Text is consumed immediately, so graph-backed strings stay borrowed.
    pub(crate) fn replace_batch<I, S>(&mut self, changes: I) -> usize
    where
        I: IntoIterator<Item = (u32, Option<S>)>,
        S: AsRef<str>,
    {
        let mut changed = FxHashSet::default();
        let mut additions: FxHashMap<TermId, Vec<Posting>> = FxHashMap::default();
        for (slot, text) in changes {
            let unique = changed.insert(slot);
            debug_assert!(unique, "batch replacement requires unique document slots");
            if let Some(old) = self.docs.remove(&slot) {
                self.total_len -= u64::from(old.len);
                for (term, _) in old.terms {
                    additions.entry(term).or_default();
                }
            }
            if let Some(text) = text {
                let doc = self.intern_document(text.as_ref());
                for &(term, tf) in &doc.terms {
                    additions
                        .entry(term)
                        .or_default()
                        .push(Posting { slot, tf });
                }
                self.total_len += u64::from(doc.len);
                self.docs.insert(slot, doc);
            }
        }

        // No term is retired while collected edits still refer to its ID: a
        // temporarily empty term can reappear in another replacement, and ID
        // recycling before all interning completes would alias those edits.
        for (term, mut added) in additions {
            added.sort_unstable_by_key(|posting| posting.slot);
            let old = std::mem::take(&mut self.postings[term as usize]);
            let mut merged = Vec::with_capacity(old.len() + added.len());
            let mut added = added.into_iter().peekable();
            for posting in old {
                if changed.contains(&posting.slot) {
                    continue;
                }
                while added.peek().is_some_and(|next| next.slot < posting.slot) {
                    merged.push(added.next().unwrap());
                }
                merged.push(posting);
            }
            merged.extend(added);
            self.postings[term as usize] = merged;
            if self.postings[term as usize].is_empty() {
                self.release_term(term);
            }
        }
        changed.len()
    }
}
