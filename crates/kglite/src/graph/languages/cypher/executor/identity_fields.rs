//! A node type's **declared identity-field spellings**, on the write path.
//!
//! `add_nodes(df, 'Person', 'person_id', 'person_name')` records the caller's
//! own column names in `DirGraph::id_field_aliases` / `title_field_aliases`, and
//! every *read* route honours them: `MATCH (p:Person {person_id: 1})` and
//! `p.person_id` both resolve through `DirGraph::resolve_alias` onto the node's
//! identity fields.
//!
//! The write path has to agree, and this module is where it does — for `CREATE`
//! (and `MERGE`'s create arm through it), `MERGE`'s match arm, `SET` (via
//! [`super::set_row`]) and `REMOVE`. When it did not agree, `CREATE (:Person
//! {person_id: 99})` stored 99 as an ordinary property next to an engine-minted
//! identity, and the dot read — which resolves the alias to the identity —
//! answered with the minted id while `properties(p)` still showed 99. One node,
//! two answers.
//!
//! The rule the whole module follows: an aliased value is **promoted** into its
//! identity field, never duplicated beside it, and only the *storage write* is
//! redirected — constraints, indexes and the type catalogue keep the spelling
//! their declaration used.

use std::collections::HashMap;

use super::CypherExecutor;
use crate::datatypes::values::Value;
use crate::graph::languages::cypher::ast::CreateNodePattern;
use crate::graph::languages::cypher::result::ResultRow;
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;

/// The property spellings a node type uses for its two identity fields.
///
/// `add_nodes(unique_id_field='person_id', node_title_field='term_name')`
/// records the caller's own column names in `DirGraph::id_field_aliases` /
/// `title_field_aliases`, and every *read* route honours them: both
/// `MATCH (p:Person {person_id: 1})` and `p.person_id` resolve through
/// [`DirGraph::resolve_alias`] onto the node's identity columns.
///
/// The write path has to agree. When it did not, `CREATE (:Person
/// {person_id: 99})` stored 99 as an ordinary property next to an
/// engine-minted identity — and because the dot read resolves the alias to the
/// identity, `p.person_id` answered with the minted id while `properties(p)`
/// still showed 99. Two routes, one node, two answers.
#[derive(Default)]
pub(super) struct IdentityAliases {
    id: Option<String>,
    title: Option<String>,
}

impl IdentityAliases {
    /// The aliases declared for `node_type`, or nothing on a graph that
    /// declares none — the common case, and the one that must allocate
    /// nothing.
    pub(super) fn for_type(graph: &DirGraph, node_type: &str) -> Self {
        if graph.id_field_aliases.is_empty() && graph.title_field_aliases.is_empty() {
            return Self::default();
        }
        Self {
            id: graph.id_field_aliases.get(node_type).cloned(),
            title: graph.title_field_aliases.get(node_type).cloned(),
        }
    }

    /// The type's declared id-field spelling, if it has one.
    pub(super) fn id_field(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// The type's declared title-field spelling, if it has one.
    pub(super) fn title_field(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// The canonical field `property` addresses on this type — the write-side
    /// twin of [`DirGraph::resolve_alias`], answered without re-reading the
    /// graph per row.
    pub(super) fn canonical<'a>(&self, property: &'a str) -> &'a str {
        if self.id.as_deref() == Some(property) {
            return "id";
        }
        if self.title.as_deref() == Some(property) {
            return "title";
        }
        property
    }
}

/// The identity fields a `CREATE` is about to write.
pub(super) struct CreatedIdentity {
    pub(super) id: Value,
    pub(super) title: Value,
    /// The title came from the pattern rather than from the `<Label>_<n>`
    /// fallback. Declared constraints read it: a `REQUIRE p.name IS NOT NULL`
    /// on a type whose title column *is* `name` asks the caller for a value,
    /// and an engine-minted fallback is not one.
    pub(super) title_supplied: bool,
}

/// The `id` and `title` a `CREATE` gives a node, honouring the type's declared
/// identity-field spellings.
///
/// Both are **promoted, not duplicated**: the key leaves the property map,
/// exactly as `add_nodes` keeps its `unique_id_field` / `node_title_field`
/// columns out of the property columns. Leaving a copy behind would put one
/// fact in two places, and the read routes would then disagree —
/// `insert_field_alias` yields the identity only for a key the property map
/// does not already carry, so a stored copy shadows the identity in
/// `properties(n)` while `n.<alias>` still resolves to the identity.
///
/// Absent → the title is fabricated from the label (unchanged), and the id is
/// *allocated*; an allocator must never hand out a live id — see
/// `DirGraph::next_auto_node_id` for the `node_bound()` reuse bug that replaced.
/// A caller-supplied id is taken as given: uniqueness is opt-in (see the gates
/// in `create_node`).
pub(super) fn create_identity(
    graph: &mut DirGraph,
    node_pat: &CreateNodePattern,
    label: &str,
    aliases: &IdentityAliases,
    properties: &mut HashMap<String, Value>,
) -> Result<CreatedIdentity, String> {
    // `id` stays accepted under its universal spelling on every type — the
    // documented `CREATE (n {id: 's1'})` round-trip — so a type carrying an
    // alias accepts two spellings for one field. Supplying both with different
    // values asks for an identity the node cannot have; refuse rather than pick.
    let aliased_id = aliases
        .id
        .as_deref()
        .and_then(|alias| properties.remove(alias));
    let literal_id = properties.remove("id");
    if let (Some(aliased), Some(literal)) = (&aliased_id, &literal_id) {
        if aliased != literal {
            let alias = aliases.id.as_deref().unwrap_or("id");
            return Err(format!(
                "CREATE gives node type '{label}' two different identities: '{alias}' is \
                 its declared id field (value {aliased}) and 'id' is the identity spelling \
                 every type accepts (value {literal}). Both name the same field, so supply \
                 one of them."
            ));
        }
    }
    let id = match aliased_id.or(literal_id) {
        Some(explicit) => {
            graph.observe_explicit_id(&explicit);
            explicit
        }
        None => graph.next_auto_node_id(),
    };

    // Title: the type's declared title field first, then the `name`/`title`
    // spellings every type accepts, then a fabricated `<Label>_<n>`. The
    // fabrication is the last resort it always was — what changed is that it no
    // longer overrides a value the caller actually supplied under the type's own
    // column name.
    let supplied_title = aliases
        .title
        .as_deref()
        .and_then(|alias| properties.remove(alias))
        .or_else(|| {
            properties
                .get("name")
                .or_else(|| properties.get("title"))
                .cloned()
        });
    let title_supplied = supplied_title.is_some();
    let title = supplied_title.unwrap_or_else(|| {
        let label = node_pat.label.as_deref().unwrap_or("Node");
        Value::String(format!("{}_{}", label, graph.graph.node_bound()))
    });

    Ok(CreatedIdentity {
        id,
        title,
        title_supplied,
    })
}

/// Refuse a `CREATE` whose identity is already taken, where identity uniqueness
/// is in force. Both gates are answered by the same id-index probe, so a type
/// subject to either pays one lookup, not two.
///
/// 1. **PRIMARY KEY** (opt-in, every storage mode). When this node type declares
///    a primary key via `define_schema`, a CREATE that would duplicate it is
///    rejected — MERGE is the explicit upsert path.
/// 2. **Durable capture** (any `Recording` backend). The write-ahead log names
///    every entity by its logical `(node_type, id)`, so a second node under a
///    live id is a write the log *cannot represent*: replay folds the two into
///    one and a node silently disappears across recovery. Refusing is the honest
///    answer — the same refuse-rather-than-degrade stance the durable open takes
///    towards a log it cannot replay.
///
/// `lookup_by_id_readonly` self-heals (builds + caches the id-index on a miss)
/// and is cross-mode, so the probe is O(1) amortised and behaves identically
/// across memory/mapped/disk. A non-durable type declaring nothing skips it
/// entirely, leaving the permissive default (and the dense-int hot path)
/// untouched.
///
/// The `id` case keeps this dedicated path because `id` is the node's identity,
/// not an entry in the property map. A primary key on any *other* property is
/// enforced by the unique-constraint index instead, installed when the schema
/// declares it (`DirGraph::set_schema`).
pub(super) fn check_identity_uniqueness(
    graph: &mut DirGraph,
    label: &str,
    id: &Value,
) -> Result<(), String> {
    let primary_key_on_id = graph.primary_key_for(label) == Some("id");
    let durable = graph.graph.is_recording();
    if !(primary_key_on_id || durable) || graph.lookup_by_id_readonly(label, id).is_none() {
        return Ok(());
    }
    Err(if primary_key_on_id {
        format!(
            "duplicate primary key: node type '{label}' declares a primary key and a \
             node with id {id} already exists. Use MERGE to upsert instead of CREATE, \
             or remove the duplicate."
        )
    } else {
        format!(
            "duplicate id in a durable graph: node type '{label}' already has a node \
             with id {id}, and the write-ahead log identifies every node by its \
             (type, id) — a second one could not be recovered, so reopening the graph \
             would merge the two and lose a node. Use MERGE to upsert, give the new \
             node a distinct id, or declare a primary key (define_schema) to enforce \
             this in every storage mode."
        )
    })
}

/// The field a `REMOVE n.<property>` clears, once the node type's declared
/// identity spellings are resolved — the `SET` rule, applied to a removal.
///
/// The type's id spelling names the immutable identity (refused, as the literal
/// `REMOVE n.id` is); its title spelling names the title, so the clear reaches
/// the field `n.<alias>` reads instead of dropping a property key that no longer
/// exists. Only the write is redirected: the caller keeps the declared name for
/// the constraint plan and index maintenance, both of which are keyed by it.
pub(super) fn remove_write_field<'a>(
    graph: &DirGraph,
    node_type: &str,
    property: &'a str,
) -> Result<&'a str, String> {
    match IdentityAliases::for_type(graph, node_type).canonical(property) {
        "id" => Err(format!(
            "Cannot REMOVE node id — it is immutable ('{property}' is the id field of \
             node type '{node_type}', so it names the identity)"
        )),
        "title" => Ok("title"),
        other => Ok(other),
    }
}

/// The property values a node-only MERGE pattern expects, keyed by the canonical
/// name of whichever field each one addresses.
///
/// A type's declared id / title spellings name the identity columns — the same
/// resolution `MATCH (p:Prospect {npdid: 1})` applies — so a MERGE key written
/// in the type's own spelling has to reach the O(1) id probe and the identity
/// read its caller performs. Without the resolution the scan looked for a
/// *stored property* under that name and found none (the create arm promotes the
/// value into the identity instead), so MERGE created a twin on every repeat.
pub(super) fn merge_expected_props<'p>(
    executor: &CypherExecutor<'_>,
    node_pat: &'p CreateNodePattern,
    row: &ResultRow,
    graph: &DirGraph,
) -> Result<Vec<(&'p str, Value)>, String> {
    let aliases = IdentityAliases::for_type(graph, node_pat.label.as_deref().unwrap_or("Node"));
    node_pat
        .properties
        .iter()
        .map(|(key, expr)| {
            executor
                .evaluate_expression(expr, row)
                .map(|val| (aliases.canonical(key.as_str()), val))
        })
        .collect()
}
