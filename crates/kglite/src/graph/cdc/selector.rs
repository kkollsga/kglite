//! Read-time filters for [`db.cdc.query`](super) — the `selectors` argument.
//!
//! ## Why filtering happens at read time, not at capture
//!
//! One log serves every consumer on the graph. A capture-time filter would
//! make the ring's contents depend on whichever consumer configured it last,
//! so a second consumer with different interests would silently read a
//! stream with holes in it — and the holes would be invisible, because the
//! sequence numbers of the events that were never captured do not exist.
//!
//! Filtering at read time keeps **cursor meaning selector-independent**: a
//! cursor addresses a position in the one log, not a position in one
//! consumer's filtered view, so two consumers can hand each other cursors and
//! a consumer can change its selectors mid-stream without resyncing. The cost
//! is that a filtered poll may return zero rows while the log has advanced —
//! see the polling guidance in `CYPHER.md`.
//!
//! ## Vocabulary
//!
//! **A selector filters on exactly what the corresponding column shows.**
//! `operation` takes `create`/`update`/`delete`, not Neo4j's `c`/`u`/`d`;
//! `elementType` takes `node`/`relationship`. A selector that spelled a
//! concept differently from the column it filters would be a trap: the
//! consumer reads `operation: "update"` in the row and writes
//! `operation: 'u'` in the filter. One vocabulary, and it is the one already
//! on the wire.

use crate::datatypes::values::Value;

use super::event::{CdcChange, CdcEvent, CdcEventKind, EdgeState, NodeState};

/// One selector: a conjunction of constraints, all of which must hold.
///
/// A list of selectors is a **disjunction** — an event is returned if *any*
/// selector matches it, which is how a consumer asks for "creates of `Person`,
/// plus every delete".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CdcSelector {
    /// `node` / `relationship`, matched against the event's element kind.
    element_type: Option<ElementType>,
    operation: Option<CdcEventKind>,
    node_type: Option<String>,
    relationship_type: Option<String>,
    src_type: Option<String>,
    tgt_type: Option<String>,
    /// Identity equality. See [`ids_equal`] for the float caveat.
    node_id: Option<Value>,
    src_id: Option<Value>,
    tgt_id: Option<Value>,
    /// Every listed label must be present — see [`CdcSelector::labels_match`].
    labels: Vec<String>,
    /// Any listed property must differ across the commit.
    changes_to: Vec<String>,
}

/// The element discriminator a selector can pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElementType {
    Node,
    Relationship,
}

/// Every key a selector map accepts, for the refusal message.
const ACCEPTED_KEYS: &[&str] = &[
    "elementType",
    "operation",
    "nodeType",
    "relationshipType",
    "srcType",
    "tgtType",
    "nodeId",
    "srcId",
    "tgtId",
    "labels",
    "changesTo",
];

/// Parse the `selectors` argument: a list of maps.
///
/// An **empty list is no filter at all** (everything is returned), because an
/// absent filter is what an empty list of filters means to the caller who
/// built it. An empty *map* inside the list is refused — see
/// [`parse_selector`] — because that is a filter that constrains nothing,
/// which is a different thing and is almost always a construction bug.
pub fn parse_selectors(value: &Value) -> Result<Vec<CdcSelector>, String> {
    let Value::List(items) = value else {
        return Err(format!(
            "db.cdc.query: 'selectors' must be a list of maps, got {value:?}. Each map is one \
             filter and an event matches if *any* of them matches it, so \
             [{{operation: 'delete'}}, {{nodeType: 'Person'}}] reads every delete plus every \
             change to a Person."
        ));
    };
    items.iter().map(parse_selector).collect()
}

/// Parse one selector map, refusing anything it cannot honour.
fn parse_selector(value: &Value) -> Result<CdcSelector, String> {
    let Value::Map(map) = value else {
        return Err(format!(
            "db.cdc.query: every entry in 'selectors' must be a map of constraints, got \
             {value:?}."
        ));
    };
    if map.is_empty() {
        return Err(
            "db.cdc.query: an empty selector map constrains nothing, so it would match every \
             change while looking like a filter — which is what a selector built from an empty \
             set of conditions produces. Omit 'selectors' entirely to read everything."
                .to_string(),
        );
    }
    let mut selector = CdcSelector::default();
    for (key, value) in map {
        match key.as_str() {
            "elementType" => {
                selector.element_type = Some(match string_of(key, value)?.as_str() {
                    "node" => ElementType::Node,
                    "relationship" => ElementType::Relationship,
                    other => {
                        return Err(format!(
                            "db.cdc.query: 'elementType' must be 'node' or 'relationship', got \
                             '{other}'. These are the values the elementType column reports."
                        ))
                    }
                });
            }
            "operation" => {
                selector.operation = Some(match string_of(key, value)?.as_str() {
                    "create" => CdcEventKind::Create,
                    "update" => CdcEventKind::Update,
                    "delete" => CdcEventKind::Delete,
                    other => {
                        return Err(format!(
                            "db.cdc.query: 'operation' must be 'create', 'update' or 'delete', \
                             got '{other}'. These are the values the operation column reports — \
                             KGLite does not use Neo4j's single-letter 'c'/'u'/'d' spelling on \
                             either side."
                        ))
                    }
                });
            }
            "nodeType" => selector.node_type = Some(string_of(key, value)?),
            "relationshipType" => selector.relationship_type = Some(string_of(key, value)?),
            "srcType" => selector.src_type = Some(string_of(key, value)?),
            "tgtType" => selector.tgt_type = Some(string_of(key, value)?),
            "nodeId" => selector.node_id = Some(scalar_of(key, value)?),
            "srcId" => selector.src_id = Some(scalar_of(key, value)?),
            "tgtId" => selector.tgt_id = Some(scalar_of(key, value)?),
            "labels" => selector.labels = string_list_of(key, value)?,
            "changesTo" => selector.changes_to = string_list_of(key, value)?,
            other => {
                return Err(format!(
                    "db.cdc.query: unknown selector key '{other}'. Accepted: {}.",
                    ACCEPTED_KEYS.join(", ")
                ))
            }
        }
    }
    Ok(selector)
}

fn string_of(key: &str, value: &Value) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        other => Err(format!(
            "db.cdc.query: selector key '{key}' must be a string, got {other:?}."
        )),
    }
}

/// An identity value: any scalar the graph can carry as an id.
fn scalar_of(key: &str, value: &Value) -> Result<Value, String> {
    match value {
        Value::List(_) | Value::Map(_) | Value::Null => Err(format!(
            "db.cdc.query: selector key '{key}' must be a single id value, got {value:?}."
        )),
        other => Ok(other.clone()),
    }
}

fn string_list_of(key: &str, value: &Value) -> Result<Vec<String>, String> {
    let Value::List(items) = value else {
        return Err(format!(
            "db.cdc.query: selector key '{key}' must be a list of strings, got {value:?}."
        ));
    };
    if items.is_empty() {
        return Err(format!(
            "db.cdc.query: selector key '{key}' was given an empty list, which constrains \
             nothing. Drop the key instead."
        ));
    }
    items.iter().map(|item| string_of(key, item)).collect()
}

/// Whether `event` passes the filter.
///
/// An empty selector list is **no filter**: everything passes. Otherwise the
/// list is a disjunction.
pub fn selected(selectors: &[CdcSelector], event: &CdcEvent) -> bool {
    selectors.is_empty() || selectors.iter().any(|selector| selector.matches(event))
}

/// Whether any selector asks for a before/after comparison, which only a
/// before-image can answer.
pub fn needs_before_images(selectors: &[CdcSelector]) -> bool {
    selectors
        .iter()
        .any(|selector| !selector.changes_to.is_empty())
}

impl CdcSelector {
    /// Every constraint in this selector holds for `event`.
    fn matches(&self, event: &CdcEvent) -> bool {
        if let Some(operation) = self.operation {
            if event.kind != operation {
                return false;
            }
        }
        match &event.change {
            CdcChange::Node {
                node_type,
                id,
                before,
                after,
            } => {
                self.element_type != Some(ElementType::Relationship)
                    && self.relationship_type.is_none()
                    && self.src_type.is_none()
                    && self.tgt_type.is_none()
                    && self.src_id.is_none()
                    && self.tgt_id.is_none()
                    && matches_opt(&self.node_type, node_type)
                    && self.node_id.as_ref().is_none_or(|want| ids_equal(want, id))
                    && self.labels_match(before.as_ref(), after.as_ref())
                    && self.changes_match(
                        before.as_ref().map(node_properties),
                        after.as_ref().map(node_properties),
                    )
            }
            CdcChange::Edge {
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
                before,
                after,
            } => {
                self.element_type != Some(ElementType::Node)
                    && self.node_type.is_none()
                    && self.node_id.is_none()
                    // A relationship carries no labels, so a labels constraint
                    // cannot hold for one.
                    && self.labels.is_empty()
                    && matches_opt(&self.relationship_type, conn_type)
                    && matches_opt(&self.src_type, src_type)
                    && matches_opt(&self.tgt_type, tgt_type)
                    && self.src_id.as_ref().is_none_or(|want| ids_equal(want, src_id))
                    && self.tgt_id.as_ref().is_none_or(|want| ids_equal(want, tgt_id))
                    && self.changes_match(
                        before.as_ref().map(edge_properties),
                        after.as_ref().map(edge_properties),
                    )
            }
        }
    }

    /// **Every** listed label must be present — a conjunction, so
    /// `labels: ['Archived', 'Cold']` means "both", which is what a reader
    /// filtering a stream down to one cohort wants. Use two selectors for
    /// "either".
    ///
    /// Matched against the node's **secondary** labels, which is exactly what
    /// the `state.labels` column reports; the primary type is `nodeType`'s
    /// job. A delete is judged on its before-image, since it has no after —
    /// and a delete with no captured before-image (enrichment `off`) therefore
    /// cannot satisfy a labels constraint, which is the honest answer: nothing
    /// is known about its labels.
    fn labels_match(&self, before: Option<&NodeState>, after: Option<&NodeState>) -> bool {
        if self.labels.is_empty() {
            return true;
        }
        let Some(state) = after.or(before) else {
            return false;
        };
        self.labels
            .iter()
            .all(|wanted| state.labels.contains(wanted))
    }

    /// Any listed property differs across the commit.
    ///
    /// Both sides are read as `Option<&Value>` — an absent image and an absent
    /// key read the same, "not there" — so the rule is total: a create matches
    /// when it *set* one of the named properties, a delete when it *had* one,
    /// and an update when the value changed. No special case per operation.
    fn changes_match(
        &self,
        before: Option<Vec<(&String, &Value)>>,
        after: Option<Vec<(&String, &Value)>>,
    ) -> bool {
        if self.changes_to.is_empty() {
            return true;
        }
        let before = before.unwrap_or_default();
        let after = after.unwrap_or_default();
        self.changes_to.iter().any(|name| {
            let was = lookup(&before, name);
            let now = lookup(&after, name);
            match (was, now) {
                (None, None) => false,
                (Some(a), Some(b)) => !values_equal(a, b),
                _ => true,
            }
        })
    }
}

fn node_properties(state: &NodeState) -> Vec<(&String, &Value)> {
    state.properties.iter().map(|(k, v)| (k, v)).collect()
}

fn edge_properties(state: &EdgeState) -> Vec<(&String, &Value)> {
    state.properties.iter().map(|(k, v)| (k, v)).collect()
}

fn lookup<'a>(properties: &[(&'a String, &'a Value)], name: &str) -> Option<&'a Value> {
    properties
        .iter()
        .find(|(key, _)| key.as_str() == name)
        .map(|(_, value)| *value)
}

fn matches_opt(wanted: &Option<String>, actual: &str) -> bool {
    wanted.as_ref().is_none_or(|want| want == actual)
}

/// Identity comparison for the `*Id` selectors.
///
/// Plain `Value` equality, with the caveat that `Value` carries floats: a
/// float id never compares equal to an integer one of the same magnitude
/// (`1.0` is not `1`), and `NaN` matches nothing including itself. Graph ids
/// are integers or strings in practice, and a float id that cannot be
/// selected is a better outcome than one that matches approximately.
fn ids_equal(wanted: &Value, actual: &Value) -> bool {
    wanted == actual
}

fn values_equal(a: &Value, b: &Value) -> bool {
    a == b
}
