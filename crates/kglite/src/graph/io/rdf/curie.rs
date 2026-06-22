//! CURIE compaction for predicate / type IRIs.
//!
//! Folds full IRIs like `http://xmlns.com/foaf/0.1/knows` into compact
//! `foaf:knows` labels so the property-graph schema reads naturally.
//! Seeded with the well-known RDF/RDFS/OWL/XSD/FOAF/schema.org/SKOS/DC
//! prefixes; document-declared `@prefix`es are added as parsing
//! progresses. When several namespaces match (one a prefix of another)
//! the LONGEST namespace wins, so `schema.org/` doesn't shadow a more
//! specific vocabulary sharing its stem.

/// The seed prefixes every RDF document is likely to use. Kept inline
/// rather than read from disk so the loader has zero config burden.
const WELL_KNOWN: &[(&str, &str)] = &[
    ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
    ("owl", "http://www.w3.org/2002/07/owl#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("foaf", "http://xmlns.com/foaf/0.1/"),
    ("schema", "http://schema.org/"),
    ("skos", "http://www.w3.org/2004/02/skos/core#"),
    ("dct", "http://purl.org/dc/terms/"),
    ("dcterms", "http://purl.org/dc/terms/"),
    ("dc", "http://purl.org/dc/elements/1.1/"),
];

/// Maps namespace IRIs to prefix names and compacts full IRIs.
pub(super) struct Curiefier {
    /// (prefix name, namespace IRI). Iterated linearly on compaction —
    /// a handful of entries, so a `Vec` beats a `HashMap`'s indirection.
    prefixes: Vec<(String, String)>,
    /// When set, `compact` is a no-op (returns the full IRI). Driven by
    /// `RdfConfig::keep_full_iris`.
    keep_full: bool,
}

impl Curiefier {
    pub(super) fn new(keep_full: bool) -> Self {
        let prefixes = WELL_KNOWN
            .iter()
            .map(|(n, i)| (n.to_string(), i.to_string()))
            .collect();
        Curiefier {
            prefixes,
            keep_full,
        }
    }

    /// Register a document-declared prefix. Dedups on namespace IRI —
    /// if the IRI is already known the first-seen name wins (well-known
    /// names are preferred over re-declarations).
    pub(super) fn add(&mut self, name: &str, iri: &str) {
        if name.is_empty() || iri.is_empty() {
            return;
        }
        if self.prefixes.iter().any(|(_, i)| i == iri) {
            return;
        }
        self.prefixes.push((name.to_string(), iri.to_string()));
    }

    /// Compact `iri` to `prefix:local` using the longest matching
    /// namespace. Falls back to the fragment after the last `#`/`/`, and
    /// finally to the full IRI. Never emits a CURIE with an empty local
    /// part (e.g. the namespace IRI itself).
    pub(super) fn compact(&self, iri: &str) -> String {
        if self.keep_full {
            return iri.to_string();
        }
        // Longest-namespace-first: pick the (name, ns) whose ns is a
        // prefix of `iri` and is the longest such ns.
        let mut best: Option<(&str, &str)> = None;
        for (name, ns) in &self.prefixes {
            if iri.starts_with(ns.as_str()) {
                let local = &iri[ns.len()..];
                if local.is_empty() {
                    continue;
                }
                match best {
                    Some((_, best_ns)) if best_ns.len() >= ns.len() => {}
                    _ => best = Some((name, ns)),
                }
            }
        }
        if let Some((name, ns)) = best {
            return format!("{}:{}", name, &iri[ns.len()..]);
        }

        // Fallback: the fragment after the last separator.
        if let Some(pos) = iri.rfind(['#', '/']) {
            let local = &iri[pos + 1..];
            if !local.is_empty() {
                return local.to_string();
            }
        }
        iri.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_known_compaction() {
        let c = Curiefier::new(false);
        assert_eq!(c.compact("http://xmlns.com/foaf/0.1/knows"), "foaf:knows");
        assert_eq!(
            c.compact("http://www.w3.org/2000/01/rdf-schema#label"),
            "rdfs:label"
        );
    }

    #[test]
    fn longest_namespace_wins() {
        let mut c = Curiefier::new(false);
        c.add("ex", "http://example.org/");
        c.add("exsub", "http://example.org/sub/");
        assert_eq!(c.compact("http://example.org/sub/thing"), "exsub:thing");
        assert_eq!(c.compact("http://example.org/thing"), "ex:thing");
    }

    #[test]
    fn fragment_fallback_and_keep_full() {
        let c = Curiefier::new(false);
        assert_eq!(c.compact("http://unknown.example/ns/Widget"), "Widget");
        let full = Curiefier::new(true);
        assert_eq!(
            full.compact("http://xmlns.com/foaf/0.1/knows"),
            "http://xmlns.com/foaf/0.1/knows"
        );
    }

    #[test]
    fn empty_local_not_emitted() {
        let c = Curiefier::new(false);
        // The namespace IRI itself has no local part → no CURIE, no
        // trailing-separator fragment → returns full IRI.
        assert_eq!(
            c.compact("http://xmlns.com/foaf/0.1/"),
            "http://xmlns.com/foaf/0.1/"
        );
    }
}
