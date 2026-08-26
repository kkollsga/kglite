//! The declared semantic layer: an `is_a` forest over type names plus
//! relationship semantics, persisted in `.kgl` metadata.
//!
//! **Annotations, not axioms.** The store never invents facts and never
//! changes what a query matches: it feeds `describe()`, provides defaults
//! for the rule-procedure validators, and (via the blueprint gate) turns
//! declarations into a load-time data-quality contract. In spirit this is
//! SKOS, not OWL — no entailment, no open-world semantics.
//!
//! Deliberately independent of `DirGraph::parent_types`: that map is a
//! *presentation* hierarchy (describe() tiering, `graph_scale`), this one is
//! *semantic* ("kind of"). `ProspectEstimate → Prospect` is ownership and
//! belongs there; `Licence is_a Licensable` belongs here. Neither is ever
//! derived from the other.
//!
//! Like `schema_from_value`, [`ontology_from_value`] is the one external
//! dialect chokepoint: Python dicts and C-ABI JSON both become a [`Value`]
//! before parsing, so every binding shares one grammar and one set of
//! messages. Canonical documents are JSON; YAML stays a Python-side
//! convenience.
//!
//! Grammar (all keys optional unless said otherwise):
//!
//! ```json
//! {
//!   "version": 1,
//!   "classes": {
//!     "Licensable": {"abstract": true, "description": "..."},
//!     "Licence":    {"is_a": "Licensable", "by": "kind", "description": "..."}
//!   },
//!   "relationships": {
//!     "HAS_OPERATOR": {
//!       "domain": "Licensable", "range": "Company",
//!       "required_properties": ["validFrom"],
//!       "property_types": {"validFrom": "date"},
//!       "inverse_name": "OPERATOR_OF",
//!       "cardinality": {"min": 0, "max": 1},
//!       "required": true, "transitive": false, "symmetric": false,
//!       "enforcement": "warn",
//!       "description": "Operatorship over time"
//!     }
//!   }
//! }
//! ```
//!
//! Semantics the fields promise (and no more):
//! - `is_a` is a forest — one parent, no cycles, parent must be declared.
//! - `cardinality` / `required` describe **outgoing** edges of the domain
//!   type (the validators they feed — `cardinality_violation`,
//!   `missing_required_edge` — count outgoing only).
//! - `symmetric` lowers to `inverse_violation(rel, rel)`, which reports each
//!   asymmetric pair once per direction encountered.
//! - `enforcement` is data for *callers* — the blueprint gate and
//!   `ontology_audit()` — never an engine write guarantee in this mode.
//! - `by` names a discriminator property and is documentation only.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::datatypes::values::Value;

/// Schema-vocabulary scale boundary, enforced rather than documented: the
/// layer is for tens-to-hundreds of classes. Thousand-class taxonomies
/// (Wikidata P279) are *data* and belong in edges.
pub const MAX_ONTOLOGY_CLASSES: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OntologyStore {
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub classes: BTreeMap<String, ClassDecl>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub relationships: BTreeMap<String, RelationshipDecl>,
}

fn default_version() -> u32 {
    1
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ClassDecl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_a: Option<String>,
    #[serde(
        rename = "abstract",
        default,
        skip_serializing_if = "std::ops::Not::not"
    )]
    pub is_abstract: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Documentation-only discriminator property name (unenforced).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RelationshipDecl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_properties: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub property_types: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inverse_name: Option<String>,
    /// Opt-in: audit `inverse_name` as a physical-pairing contract. Without
    /// it the name is a reading-direction alias only (describe()/agent
    /// metadata) and enrolls no check.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub inverse_enforced: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cardinality: Option<CardinalityDecl>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub required: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub transitive: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub symmetric: bool,
    #[serde(default, skip_serializing_if = "Enforcement::is_default")]
    pub enforcement: Enforcement,
    /// Per-check severities (keys from [`CHECK_NAMES`]); unlisted checks
    /// fall back to `enforcement`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub enforcement_overrides: BTreeMap<String, Enforcement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl RelationshipDecl {
    /// The severity governing one check of this declaration.
    pub fn enforcement_for(&self, check: &str) -> Enforcement {
        self.enforcement_overrides
            .get(check)
            .copied()
            .unwrap_or(self.enforcement)
    }

    /// The base severity, plus any per-check overrides as `check=severity`.
    /// Both reader surfaces (`SHOW ONTOLOGY`, `describe()`) render this, so a
    /// declaration whose overrides raise a check above its base severity can
    /// never read as the bare base in one of them.
    pub(crate) fn enforcement_summary(&self) -> String {
        let base = self.enforcement.as_str().to_string();
        if self.enforcement_overrides.is_empty() {
            return base;
        }
        let overrides: Vec<String> = self
            .enforcement_overrides
            .iter()
            .map(|(check, sev)| format!("{check}={}", sev.as_str()))
            .collect();
        format!("{base}; {}", overrides.join(", "))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Enforcement {
    #[default]
    Advisory,
    Warn,
    Error,
}

impl Enforcement {
    fn is_default(&self) -> bool {
        *self == Enforcement::Advisory
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Enforcement::Advisory => "advisory",
            Enforcement::Warn => "warn",
            Enforcement::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CardinalityDecl {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<u64>,
}

/// State of one materialized (managed) label — see
/// `dir_graph/ontology_apply.rs` for the invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ManagedLabelState {
    /// The engine is the bucket's only writer; it holds exactly the closure.
    Closed,
    /// Something outside the closure touched the bucket (manual SET, adopt,
    /// extend union). Correct, but closure-reliant optimizations stay off.
    Open,
}

impl ManagedLabelState {
    pub fn as_str(&self) -> &'static str {
        match self {
            ManagedLabelState::Closed => "closed",
            ManagedLabelState::Open => "open",
        }
    }
}

/// `skip_serializing_if` helper for the `Arc<OntologyStore>` field on
/// `DirGraph` (serde hands the function `&Arc<_>`, which method syntax
/// won't coerce in the derive).
pub fn arc_store_is_empty(store: &std::sync::Arc<OntologyStore>) -> bool {
    store.is_empty()
}

impl OntologyStore {
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty() && self.relationships.is_empty()
    }

    /// Ancestor chain of `class`, nearest first. Empty for roots and for
    /// names the store does not declare. Bounded by the forest invariant
    /// (`validate` rejects cycles), with a defensive cap for stores built
    /// without it.
    pub fn ancestors(&self, class: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut current = class;
        while let Some(parent) = self.classes.get(current).and_then(|c| c.is_a.as_deref()) {
            if out.len() > self.classes.len() {
                break;
            }
            out.push(parent.to_string());
            current = parent;
        }
        out
    }

    /// Structural validation: the forest invariant, dangling `is_a`
    /// targets, and the class cap. Graph-aware checks (abstract vs live
    /// primary types) live in `DirGraph::define_ontology`, which has the
    /// graph.
    pub fn validate(&self) -> Result<(), String> {
        if self.classes.len() > MAX_ONTOLOGY_CLASSES {
            return Err(format!(
                "ontology declares {} classes; the layer is for schema-level vocabularies \
                 (max {MAX_ONTOLOGY_CLASSES}). Large taxonomies are data — model them as edges \
                 (a `transitive` relationship declaration plus `*1..` paths).",
                self.classes.len()
            ));
        }
        for (name, decl) in &self.classes {
            if let Some(parent) = &decl.is_a {
                if !self.classes.contains_key(parent) {
                    return Err(format!(
                        "class '{name}': is_a target '{parent}' is not a declared class"
                    ));
                }
                if parent == name {
                    return Err(format!("class '{name}': is_a itself"));
                }
            }
        }
        // Cycle check: walk each chain; the forest has ≤ classes.len() edges,
        // so a longer walk is a cycle.
        for name in self.classes.keys() {
            let mut steps = 0usize;
            let mut current = name.as_str();
            while let Some(parent) = self.classes.get(current).and_then(|c| c.is_a.as_deref()) {
                steps += 1;
                if steps > self.classes.len() {
                    return Err(format!("class '{name}': is_a chain contains a cycle"));
                }
                current = parent;
            }
        }
        for (name, decl) in &self.relationships {
            for endpoint in [&decl.domain, &decl.range].into_iter().flatten() {
                if endpoint.is_empty() {
                    return Err(format!("relationship '{name}': empty endpoint name"));
                }
            }
            if let Some(card) = &decl.cardinality {
                if let (Some(min), Some(max)) = (card.min, card.max) {
                    if min > max {
                        return Err(format!(
                            "relationship '{name}': cardinality min {min} > max {max}"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Parse the external dialect out of the shared [`Value`] data model — the
/// single chokepoint every binding routes through (see module doc). Unknown
/// keys are refused with a did-you-mean, the `define_schema` posture: a
/// declaration key the parser cannot place is never a harmless extra.
pub fn ontology_from_value(doc: &Value) -> Result<OntologyStore, String> {
    let map = as_map(doc).ok_or("ontology document must be a map")?;
    reject_unknown(map, &["version", "classes", "relationships"], "ontology")?;

    let mut store = OntologyStore {
        version: 1,
        ..Default::default()
    };
    if let Some(v) = map.get("version") {
        store.version = match v {
            Value::Int64(n) if *n >= 1 => *n as u32,
            _ => return Err("ontology 'version' must be a positive integer".to_string()),
        };
    }
    if let Some(classes) = map.get("classes") {
        let classes = as_map(classes).ok_or("ontology 'classes' must be a map")?;
        for (name, decl) in classes {
            store
                .classes
                .insert(name.to_string(), class_from_value(name, decl)?);
        }
    }
    if let Some(rels) = map.get("relationships") {
        let rels = as_map(rels).ok_or("ontology 'relationships' must be a map")?;
        for (name, decl) in rels {
            store
                .relationships
                .insert(name.to_string(), relationship_from_value(name, decl)?);
        }
    }
    store.validate()?;
    Ok(store)
}

/// [`ontology_from_value`] over a JSON document — the C ABI / file entry
/// point, routed through the same JSON→[`Value`] conversion every other
/// JSON-carrying surface uses.
pub fn ontology_from_json(json: &str) -> Result<OntologyStore, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("ontology JSON parse: {e}"))?;
    let value = crate::param::json_value_to_kglite_value(&parsed);
    ontology_from_value(&value)
}

/// The closed accept-list for `property_types` values — every spelling
/// [`crate::graph::mutation::validation::value_matches_type`] resolves.
const PROPERTY_TYPE_NAMES: &[&str] = &[
    "string",
    "str",
    "integer",
    "int",
    "i64",
    "int64",
    "float",
    "double",
    "f64",
    "number",
    "float64",
    "boolean",
    "bool",
    "date",
    "datetime",
    "timestamp",
    "uniqueid",
    "point",
    "any",
];

/// Every declaration-driven check name (`DeclaredCheck::name` values) —
/// the accepted key set for the map form of `enforcement`.
pub const CHECK_NAMES: &[&str] = &[
    "domain",
    "range",
    "required",
    "required_properties",
    "property_types",
    "cardinality",
    "inverse",
    "symmetric",
    "transitive",
];

const CLASS_KEYS: &[&str] = &["is_a", "abstract", "description", "by"];
const REL_KEYS: &[&str] = &[
    "domain",
    "range",
    "required_properties",
    "property_types",
    "inverse_name",
    "inverse_enforced",
    "cardinality",
    "required",
    "transitive",
    "symmetric",
    "enforcement",
    "description",
];

fn class_from_value(name: &str, value: &Value) -> Result<ClassDecl, String> {
    let map = as_map(value).ok_or_else(|| format!("class '{name}' must be a map"))?;
    reject_unknown(map, CLASS_KEYS, &format!("class '{name}'"))?;
    Ok(ClassDecl {
        is_a: opt_string(map, "is_a", name)?,
        is_abstract: opt_bool(map, "abstract", name)?,
        description: opt_string(map, "description", name)?,
        by: opt_string(map, "by", name)?,
    })
}

fn relationship_from_value(name: &str, value: &Value) -> Result<RelationshipDecl, String> {
    let map = as_map(value).ok_or_else(|| format!("relationship '{name}' must be a map"))?;
    reject_unknown(map, REL_KEYS, &format!("relationship '{name}'"))?;
    let severity = |s: &str| -> Result<Enforcement, String> {
        match s {
            "advisory" => Ok(Enforcement::Advisory),
            "warn" => Ok(Enforcement::Warn),
            "error" => Ok(Enforcement::Error),
            other => Err(format!(
                "relationship '{name}': enforcement '{other}' is not one of \
                 'advisory', 'warn', 'error'"
            )),
        }
    };
    let (enforcement, enforcement_overrides) = match map.get("enforcement") {
        None => (Enforcement::Advisory, BTreeMap::new()),
        Some(Value::String(s)) => (severity(s)?, BTreeMap::new()),
        // Map form: per-check severities; unlisted checks keep the
        // advisory base.
        Some(other) => match as_map(other) {
            Some(per_check) => {
                let mut overrides = BTreeMap::new();
                for (check, sv) in per_check {
                    if !CHECK_NAMES.contains(&check) {
                        return Err(format!(
                            "relationship '{name}': enforcement key '{check}' \
                             is not a check — use one of {CHECK_NAMES:?}"
                        ));
                    }
                    let Value::String(sv) = sv else {
                        return Err(format!(
                            "relationship '{name}': enforcement['{check}'] \
                             must be a severity string"
                        ));
                    };
                    overrides.insert(check.to_string(), severity(sv)?);
                }
                (Enforcement::Advisory, overrides)
            }
            None => {
                return Err(format!(
                    "relationship '{name}': 'enforcement' must be a severity \
                     string or a {{check: severity}} map"
                ))
            }
        },
    };
    let cardinality = match map.get("cardinality") {
        None => None,
        Some(v) => {
            let card = as_map(v)
                .ok_or_else(|| format!("relationship '{name}': 'cardinality' must be a map"))?;
            reject_unknown(
                card,
                &["min", "max"],
                &format!("relationship '{name}' cardinality"),
            )?;
            Some(CardinalityDecl {
                min: opt_u64(card, "min", name)?,
                max: opt_u64(card, "max", name)?,
            })
        }
    };
    let required_properties = match map.get("required_properties") {
        None => Vec::new(),
        Some(Value::List(items)) => items
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => Err(format!(
                    "relationship '{name}': 'required_properties' entries must be strings"
                )),
            })
            .collect::<Result<_, _>>()?,
        Some(_) => {
            return Err(format!(
                "relationship '{name}': 'required_properties' must be a list"
            ))
        }
    };
    let property_types = match map.get("property_types") {
        None => BTreeMap::new(),
        Some(v) => {
            let types = as_map(v)
                .ok_or_else(|| format!("relationship '{name}': 'property_types' must be a map"))?;
            let mut out = BTreeMap::new();
            for (k, tv) in types {
                match tv {
                    Value::String(s) => {
                        // value_matches_type is permissive on unknown names,
                        // so a typo here would otherwise never fail anything.
                        if !PROPERTY_TYPE_NAMES.contains(&s.to_lowercase().as_str()) {
                            return Err(format!(
                                "relationship '{name}': 'property_types' entry \
                                 '{k}: {s}' names an unknown type — use one of \
                                 string, integer, float, boolean, date, \
                                 datetime, timestamp, point, any"
                            ));
                        }
                        out.insert(k.to_string(), s.clone());
                    }
                    _ => {
                        return Err(format!(
                            "relationship '{name}': 'property_types' values must be strings"
                        ))
                    }
                }
            }
            out
        }
    };
    let inverse_name = opt_string(map, "inverse_name", name)?;
    let inverse_enforced = opt_bool(map, "inverse_enforced", name)?;
    if inverse_enforced && inverse_name.is_none() {
        return Err(format!(
            "relationship '{name}': 'inverse_enforced' requires 'inverse_name'"
        ));
    }
    Ok(RelationshipDecl {
        domain: opt_string(map, "domain", name)?,
        range: opt_string(map, "range", name)?,
        required_properties,
        property_types,
        inverse_name,
        inverse_enforced,
        cardinality,
        required: opt_bool(map, "required", name)?,
        transitive: opt_bool(map, "transitive", name)?,
        symmetric: opt_bool(map, "symmetric", name)?,
        enforcement,
        enforcement_overrides,
        description: opt_string(map, "description", name)?,
    })
}

fn as_map(value: &Value) -> Option<&crate::datatypes::PropMap> {
    match value {
        Value::Map(map) => Some(map),
        _ => None,
    }
}

fn opt_string(
    map: &crate::datatypes::PropMap,
    key: &str,
    ctx: &str,
) -> Result<Option<String>, String> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(format!("'{ctx}': '{key}' must be a string")),
    }
}

fn opt_bool(map: &crate::datatypes::PropMap, key: &str, ctx: &str) -> Result<bool, String> {
    match map.get(key) {
        None => Ok(false),
        Some(Value::Boolean(b)) => Ok(*b),
        Some(_) => Err(format!("'{ctx}': '{key}' must be a boolean")),
    }
}

fn opt_u64(map: &crate::datatypes::PropMap, key: &str, ctx: &str) -> Result<Option<u64>, String> {
    match map.get(key) {
        None => Ok(None),
        Some(Value::Int64(n)) if *n >= 0 => Ok(Some(*n as u64)),
        Some(_) => Err(format!(
            "relationship '{ctx}': cardinality '{key}' must be a non-negative integer"
        )),
    }
}

fn reject_unknown(
    map: &crate::datatypes::PropMap,
    accepted: &[&str],
    ctx: &str,
) -> Result<(), String> {
    for (key, _) in map {
        if accepted.contains(&key) {
            continue;
        }
        let suggestion = crate::graph::mutation::validation::did_you_mean(key, accepted);
        if !suggestion.is_empty() {
            return Err(format!("{ctx}: unknown key '{key}'.{suggestion}"));
        }
        let list = accepted
            .iter()
            .map(|k| format!("'{k}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "{ctx}: unknown key '{key}'. Accepted keys: {list}."
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Result<OntologyStore, String> {
        ontology_from_json(json)
    }

    #[test]
    fn round_trips_the_full_grammar() {
        let store = parse(
            r#"{"version": 1,
                "classes": {
                  "Licensable": {"abstract": true, "description": "d"},
                  "Licence": {"is_a": "Licensable", "by": "kind"}
                },
                "relationships": {
                  "HAS_OPERATOR": {
                    "domain": "Licensable", "range": "Company",
                    "required_properties": ["validFrom"],
                    "property_types": {"validFrom": "date"},
                    "inverse_name": "OPERATOR_OF",
                    "cardinality": {"min": 0, "max": 1},
                    "required": true, "transitive": false, "symmetric": false,
                    "enforcement": "warn", "description": "op"
                  }
                }}"#,
        )
        .unwrap();
        assert!(store.classes["Licensable"].is_abstract);
        assert_eq!(store.classes["Licence"].is_a.as_deref(), Some("Licensable"));
        assert_eq!(store.ancestors("Licence"), vec!["Licensable"]);
        let rel = &store.relationships["HAS_OPERATOR"];
        assert_eq!(rel.enforcement, Enforcement::Warn);
        assert_eq!(rel.cardinality.unwrap().max, Some(1));
        // Serde round-trip (the FileMetadata path).
        let json = serde_json::to_string(&store).unwrap();
        let back: OntologyStore = serde_json::from_str(&json).unwrap();
        assert_eq!(back, store);
    }

    #[test]
    fn rejects_unknown_keys_with_suggestion() {
        let err = parse(r#"{"clases": {}}"#).unwrap_err();
        assert!(err.contains("classes"), "{err}");
        let err = parse(r#"{"classes": {"A": {"isa": "B"}}}"#).unwrap_err();
        assert!(err.contains("is_a"), "{err}");
        let err = parse(r#"{"relationships": {"R": {"enforcement": "fatal"}}}"#).unwrap_err();
        assert!(err.contains("advisory"), "{err}");
    }

    #[test]
    fn rejects_forest_violations() {
        let err = parse(r#"{"classes": {"A": {"is_a": "Missing"}}}"#).unwrap_err();
        assert!(err.contains("not a declared class"), "{err}");
        let err = parse(r#"{"classes": {"A": {"is_a": "B"}, "B": {"is_a": "A"}}}"#).unwrap_err();
        assert!(err.contains("cycle"), "{err}");
        let err = parse(r#"{"classes": {"A": {"is_a": "A"}}}"#).unwrap_err();
        assert!(err.contains("itself"), "{err}");
    }

    #[test]
    fn enforces_the_class_cap() {
        let classes: Vec<String> = (0..=MAX_ONTOLOGY_CLASSES)
            .map(|i| format!("\"C{i}\": {{}}"))
            .collect();
        let doc = format!("{{\"classes\": {{{}}}}}", classes.join(","));
        let err = parse(&doc).unwrap_err();
        assert!(err.contains("schema-level vocabularies"), "{err}");
    }

    #[test]
    fn cardinality_min_over_max_rejected() {
        let err = parse(r#"{"relationships": {"R": {"cardinality": {"min": 2, "max": 1}}}}"#)
            .unwrap_err();
        assert!(err.contains("min 2 > max 1"), "{err}");
    }

    #[test]
    fn empty_store_serializes_to_nothing_extra() {
        let store = OntologyStore::default();
        assert!(store.is_empty());
        assert_eq!(serde_json::to_string(&store).unwrap(), r#"{"version":0}"#);
    }
}
