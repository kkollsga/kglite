//! Dense IRI interner for the RDF fold.
//!
//! Maps each distinct subject/object IRI (or blank-node key) to a
//! sequential `u32` starting at 0, so the fold can use a `Vec`-indexed
//! accumulator and emit `Value::UniqueId(id)` node ids that double as
//! the dense materialisation order. The string table (`iris`) keeps the
//! original IRI so it can be stored as the node's `uri` property.

use std::collections::HashMap;

/// Interns IRI / blank-node keys to dense `u32` ids (0, 1, 2, …).
pub(super) struct IriInterner {
    map: HashMap<String, u32>,
    iris: Vec<String>,
}

impl IriInterner {
    pub(super) fn new() -> Self {
        IriInterner {
            map: HashMap::new(),
            iris: Vec::new(),
        }
    }

    /// Return the id for `iri`, interning it (and assigning the next
    /// sequential id) on first sight.
    pub(super) fn get_or_intern(&mut self, iri: &str) -> u32 {
        if let Some(&id) = self.map.get(iri) {
            return id;
        }
        let id = self.iris.len() as u32;
        self.iris.push(iri.to_string());
        self.map.insert(iri.to_string(), id);
        id
    }

    /// Resolve an interned id back to its IRI / blank-node key.
    pub(super) fn iri(&self, id: u32) -> &str {
        &self.iris[id as usize]
    }

    /// Number of distinct interned IRIs.
    pub(super) fn len(&self) -> usize {
        self.iris.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_sequential_ids() {
        let mut i = IriInterner::new();
        assert_eq!(i.get_or_intern("http://a"), 0);
        assert_eq!(i.get_or_intern("http://b"), 1);
        assert_eq!(i.get_or_intern("http://a"), 0);
        assert_eq!(i.get_or_intern("http://c"), 2);
        assert_eq!(i.len(), 3);
        assert_eq!(i.iri(1), "http://b");
    }
}
