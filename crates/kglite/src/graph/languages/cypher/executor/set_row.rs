//! One row of a Cypher `SET n.prop = …`, and the statement-scoped memo that
//! keeps the statement-constant part of it out of the loop.
//!
//! Split out of `write.rs` (P6) because the two halves answer different
//! questions. `execute_set` decides *which* rows and values there are; this
//! module applies one of them and settles what the write owes the rest of the
//! engine — the type's `TypeSchema`, its index buckets, its unique claims, its
//! property catalogue, its freshness stamp.
//!
//! **The cost model.** Every one of those obligations is a fact about the
//! `(node type, property)` pair, not about the row, and the pre-P6 code
//! re-derived all of them per written row: the type name `to_string()`-ed off
//! the node, `auto_timestamp_for` and the schema-key probe hashed again, a
//! fresh `HashMap` plus two `String`s handed to `upsert_node_type_metadata`
//! only for it to find the fact already recorded, and a full incremental
//! index-maintenance pass on a type that might carry no index at all. A 100k-row
//! `SET` spent ~450 ns/row with the actual cell write at 10.4% of the profile.
//! [`SetMemos`] answers each question once per statement; what is left per row
//! is the write itself, the constraint plan, and a journal capture.

use std::collections::HashMap;
use std::sync::Arc;

use super::columnar_write::{set_via_column_master, ColumnMasterWrite};
use super::identity_fields::IdentityAliases;
use super::write::set_node_property_direct;
use super::write_scope::enforce_write_scope;
use crate::datatypes::values::Value;
use crate::graph::languages::cypher::result::MutationStats;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::borrow::Cow;

/// One row's node-property write, as `execute_set` resolved it.
pub(super) struct NodePropertySet<'a> {
    pub node_idx: NodeIndex,
    /// The property name **as the AST holds it** — the memo keys on it, so it
    /// must outlive the statement rather than be rebuilt per row.
    pub property: &'a str,
    pub value: Value,
}

/// The statement-scoped answers behind [`RowBookkeeping`].
///
/// Scope is exactly one `execute_set` call, which is what makes it safe to
/// treat these answers as constant: nothing inside a `SET` creates an index,
/// drops a `TypeSchema` key, or removes a metadata entry (`SET n = {…}`'s
/// nested `REMOVE` clears *values*, never the catalogue), and a failed
/// statement discards the memo along with the writes it describes.
#[derive(Default)]
pub(super) struct SetMemos<'a> {
    types: HashMap<InternedKey, TypeFacts>,
    properties: HashMap<(InternedKey, &'a str), PropertyFacts>,
}

/// What a `SET` needs to know about a node type, resolved on its first row.
struct TypeFacts {
    /// The type's name — resolved once instead of `to_string()`-ed per row.
    name: String,
    /// The type's declared identity-field spellings, so a write to one of them
    /// reaches the field the *reads* of that name resolve to. Statement-scoped
    /// like the rest of the memo: `SET` never registers an alias.
    aliases: IdentityAliases,
    maintains_indexes: bool,
    auto_timestamp: bool,
}

/// What a `SET` needs to know about a `(type, property)` pair.
struct PropertyFacts {
    /// The value-type name `node_type_metadata` is known to hold. A row whose
    /// value has a *different* type name (a heterogeneous `CASE`, say) still
    /// records, so last-write-wins is preserved exactly.
    recorded_type: &'static str,
    /// The property name, interned **and registered** in the graph's
    /// `StringInterner` — the key the write itself is addressed by. Interning
    /// registers the name, which is what lets `save()` resolve the key back to
    /// a string, and it is a fact about the name rather than the row: the write
    /// path used to hash and register it once per written row.
    key: InternedKey,
}

/// The per-row half of a `SET`'s bookkeeping — what is left once [`SetMemos`]
/// has answered the statement-constant questions.
#[derive(Clone, Copy)]
struct RowBookkeeping {
    /// The type carries at least one user index, so the bucket move is real.
    /// When false the old value was never read: nothing consumes it.
    maintains_indexes: bool,
    /// The type's shared `TypeSchema` may not carry the property key yet, so
    /// the (read-first) registration has to run. `TypeSchema`s only grow within
    /// a statement, so one successful pass makes this false for every later row.
    register_schema_key: bool,
    /// `Some(value type name)` when `node_type_metadata` does not already hold
    /// that name for this `(type, property)` — the catalogue write. `None` once
    /// an earlier row of the same statement recorded the same type name.
    record_metadata: Option<&'static str>,
    /// The type opted into `updated_at` stamping.
    auto_timestamp: bool,
    /// The written property, interned once per statement — see
    /// [`PropertyFacts::key`].
    key: InternedKey,
}

/// A single-property node `SET` that has already landed in storage, as the
/// post-write bookkeeping needs to see it.
struct LandedPropertyWrite<'a> {
    node_idx: NodeIndex,
    node_type: &'a str,
    /// The name the statement used — what a constraint, an index and the type
    /// catalogue are keyed by.
    property: &'a str,
    /// The field the value landed in, once the type's identity-field spellings
    /// were resolved. Differs from `property` only for a title alias.
    write_field: &'a str,
    old_value: Option<&'a Value>,
    value: &'a Value,
    constraint_plan: &'a crate::graph::dir_graph::constraints::PropertyWritePlan,
    /// What this row still owes, after the memos have answered everything that
    /// is constant across the statement's rows.
    owed: RowBookkeeping,
}

/// The schema lock's verdict on one property write.
///
/// The lock carries two independent verdicts, and a declared type constraint
/// exempts exactly one of them:
///
/// * The **typo guard** always applies. A type constraint may be declared on a
///   property no node holds — nothing violates it yet — and that leaves the
///   property absent from the observed metadata the lock validates against.
///   Exempting it here let a locked graph accept a property it does not know,
///   which is the one write `lock_schema()` exists to refuse. (The DDL guard
///   `validate_ddl_properties_declared` does not close this: it only runs when
///   the graph is *already* locked, so a constraint declared before the lock
///   never meets it.)
/// * The **observed-type verdict** yields to the declaration: both answer the
///   same question, one from observed metadata and one from what the user
///   wrote, and the user's wins. Without that, a schema-locked graph reported
///   the generic validation error for a value the user had explicitly
///   constrained — naming a type they never declared and losing the typed
///   `ConstraintViolationError` the constraint raises instead.
fn enforce_schema_lock(
    graph: &DirGraph,
    node_type: &str,
    property: &str,
    value: &Value,
) -> Result<(), String> {
    if !graph.schema_locked {
        return Ok(());
    }
    crate::graph::mutation::validation::validate_property_known(
        node_type,
        property,
        value,
        &graph.node_type_metadata,
        graph.schema_definition.as_ref(),
    )?;
    if graph.property_type_for(node_type, property).is_none() {
        crate::graph::mutation::validation::validate_property_type(
            node_type,
            property,
            value,
            &graph.node_type_metadata,
        )?;
    }
    Ok(())
}

/// Apply one row's `SET n.property = value`, with all of its bookkeeping.
///
/// Returns without writing when the binding no longer resolves to a live node —
/// the same no-op that row had before, reached one read earlier.
pub(super) fn apply_node_property_set<'a>(
    graph: &mut DirGraph,
    write: NodePropertySet<'a>,
    memos: &mut SetMemos<'a>,
    stats: &mut MutationStats,
    nodes_to_stamp: &mut HashMap<NodeIndex, String>,
) -> Result<(), String> {
    let NodePropertySet {
        node_idx,
        property,
        value,
    } = write;

    // The row's node type, and with it every statement-constant fact about
    // this write. Arena guard: the type read materializes on the disk backend
    // (protocol in disk/graph.rs); scoped so the borrow ends before the &mut
    // mutation below.
    let Some(type_key) = ({
        let _arena_guard = graph.graph.begin_query();
        GraphRead::node_type_of(&graph.graph, node_idx)
    }) else {
        return Ok(());
    };
    let facts = type_facts_for(&mut memos.types, graph, type_key);
    let node_type_str = facts.name.as_str();

    // The field this write actually lands in. A type's declared identity
    // spellings name the identity columns — the resolution every *read* of that
    // name applies — so `SET n.term_name = 'x'` on a type loaded with
    // `node_title_field='term_name'` has to write the title. It used to store an
    // ordinary property instead, which `n.term_name` could never return: the
    // write was invisible on the route that asked for it.
    //
    // The id half is refused rather than rewritten, for the reason the literal
    // `SET n.id` already is: the identity is immutable, and `add_nodes` treats
    // its `unique_id_field` column as the row's key rather than a value it
    // updates.
    //
    // Only the *storage* write is redirected. Constraints, index maintenance
    // and the type catalogue keep the name the statement (and the declaration)
    // used — `update_property_indices_for_set` resolves the alias itself, and a
    // constraint is declared and reported under the spelling its author wrote.
    let write_field = match facts.aliases.canonical(property) {
        "id" => {
            return Err(format!(
                "Cannot SET node id — it is immutable ('{property}' is the id field of \
                 node type '{node_type_str}', so it names the identity)"
            ))
        }
        "title" => "title",
        _ => property,
    };

    // The old value, read **only when an index will consume it**: its one
    // consumer is the bucket move in `update_property_indices_for_set`, and on
    // a type carrying no index that move is a provable no-op — so there the
    // read (a columnar cell fetch and a `Value` clone, per row) buys nothing.
    let old_value = if facts.maintains_indexes {
        let _arena_guard = graph.graph.begin_query();
        let Some(node) = graph.node_view(node_idx) else {
            return Ok(());
        };
        // For `name` (the canonical title-alias name in Cypher), the value is
        // stored on `node.title`, not in the property map.
        // `get_field_ref("name")` returns None for graphs where "name" isn't
        // also redundantly in properties — which is the case for `.kgl`-loaded
        // graphs and for indexes built from `get_node_title` (see
        // `dir_graph.rs::create_index`'s alias-resolution path). Falling back
        // to the title keeps index auto-maintenance consistent with how those
        // indexes were populated.
        match write_field {
            // `title()`, not the raw inline field: these indexes are built
            // from `get_node_title`.
            "name" => node
                .get_field_ref("name")
                .map(Cow::into_owned)
                .or_else(|| Some(node.title().into_owned())),
            "title" => Some(node.title().into_owned()),
            _ => node.get_field_ref(write_field).map(Cow::into_owned),
        }
    } else {
        None
    };

    // Role-scoped write guard: reject SET on a node type outside the active
    // write whitelist.
    enforce_write_scope(graph, node_type_str)?;

    enforce_schema_lock(graph, node_type_str, property, &value)?;

    // Declared structured shapes (tables.rs): checked pre-write with the
    // indexed error path (`line_items[3].qty: expected integer`).
    if let Some(shape) = graph.shape_for(node_type_str, property) {
        shape.check(property, &value)?;
    }

    // Declared UNIQUE / NOT NULL gates. Planned before the write so a violation
    // returns without mutating storage; the returned plan is redeemed after the
    // value lands.
    let constraint_plan = graph
        .plan_property_write(node_type_str, node_idx, property, Some(&value))
        .map_err(|violation| violation.to_string())?;

    // What this row still owes once the memos have answered everything
    // constant across the statement's rows.
    let owed = row_bookkeeping(
        &mut memos.properties,
        type_key,
        property,
        value.type_name(),
        facts,
        &mut graph.interner,
    );

    // Fast path for Columnar storage when the graph's master
    // `Arc<ColumnStore>` for this node-type is available: route the write
    // through the master once per batch instead of through each node's Arc
    // handle. The per-node Arcs all point at the same allocation, so
    // `Arc::make_mut` on a node Arc clones the entire store on every write —
    // O(N²) total for batch SETs. The master Arc has refcount=1 inside this
    // batch (after the initial clone, if any), so subsequent writes mutate in
    // place. Arena guard: node_weight materializes on the disk backend
    // (protocol in disk/graph.rs); scoped so the borrow ends before the &mut
    // writes below.
    let columnar_row_id = {
        let _arena_guard = graph.graph.begin_query();
        graph
            .graph
            .node_weight(node_idx)
            .and_then(|n| n.properties.columnar_row_id())
    };
    let wrote_via_master = set_via_column_master(
        graph,
        ColumnMasterWrite {
            node_idx,
            node_type: node_type_str,
            type_key,
            property: write_field,
            key: owed.key,
            value: &value,
            row_id: columnar_row_id,
        },
    );
    if wrote_via_master {
        stats.properties_set += 1;
    }
    if !wrote_via_master {
        // Row storage, or title/name, or a columnar node whose type the backend
        // has no store for (disk-mode graphs with their own staged-write path):
        // fall through to the backend's per-node setter, which routes by
        // storage variant. The clone lives here rather than before the write:
        // the master path borrows the value, so only the setter that *consumes*
        // one pays for it.
        if set_node_property_direct(graph, node_idx, write_field, value.clone()) {
            stats.properties_set += 1;
        }
    }

    finish_node_property_write(
        graph,
        LandedPropertyWrite {
            node_idx,
            node_type: node_type_str,
            property,
            write_field,
            old_value: old_value.as_ref(),
            value: &value,
            constraint_plan: &constraint_plan,
            owed,
        },
        nodes_to_stamp,
    );
    Ok(())
}

/// The bookkeeping a single-property node `SET` owes once the value has landed.
///
/// None of it is about *writing* the value: it registers the property key on
/// the type's `TypeSchema`, moves the node between index buckets, redeems the
/// constraint plan (handing the vacated unique tuples back and taking the new
/// ones), keeps `node_type_metadata` accurate so `schema()` reports the
/// property's type, and notes the node for the post-loop `updated_at` bump.
///
/// Order is load-bearing: the index move needs the old value, so it runs before
/// the constraint plan is redeemed. The `updated_at` bump is skipped when the
/// write *is* to `updated_at`, which would otherwise recurse. A `title` write
/// touches only the title field, not a property map, so it registers no schema
/// key — but it still moves the node between index buckets: an index on
/// `title`, or one registered under the type's title-alias spelling, is built
/// from exactly the value it changed.
fn finish_node_property_write(
    graph: &mut DirGraph,
    write: LandedPropertyWrite<'_>,
    nodes_to_stamp: &mut HashMap<NodeIndex, String>,
) {
    // A `title` write touches the node's title field, not the property map, so
    // it registers no schema key — under either spelling of that field.
    if write.owed.register_schema_key && write.write_field != "title" {
        // The memo's key — interned from `property` when the memo took it, so
        // it is the same value `InternedKey::from_str` would recompute here.
        let ik = write.owed.key;
        // Read-check before taking `&mut`: `type_schemas` is `Arc`-shared with
        // the rollback shell (`dir_graph::schema_cow`), so `type_schemas_mut()`
        // copies the whole O(types) map — and this runs per written row, almost
        // always for a key the schema already carries.
        let needs_key = graph
            .type_schemas
            .get(write.node_type)
            .is_some_and(|schema| schema.slot(ik).is_none());
        if needs_key {
            if let Some(schema_arc) = graph.type_schemas_mut().get_mut(write.node_type) {
                Arc::make_mut(schema_arc).add_key(ik);
            }
        }
    }
    if write.owed.maintains_indexes {
        graph.update_property_indices_for_set(
            write.node_type,
            write.node_idx,
            write.property,
            write.old_value,
            write.value,
        );
    }

    // `write_field` is the alias-resolved spelling the value actually landed
    // in, which is exactly what a text index records for itself — so a `SET`
    // of some other property of an indexed type discriminates to a no-op
    // rather than dirtying the row.
    crate::graph::index_freshness::write_hooks::note_property_written(
        graph,
        write.node_idx,
        write.node_type,
        Some(write.write_field),
    );

    graph.apply_property_write_plan(write.constraint_plan, write.node_idx);

    if let Some(value_type) = write.owed.record_metadata {
        let mut prop_type = HashMap::new();
        prop_type.insert(write.property.to_string(), value_type.to_string());
        graph.upsert_node_type_metadata(write.node_type, prop_type);
    }

    if write.owed.auto_timestamp && write.property != "updated_at" {
        nodes_to_stamp.insert(write.node_idx, write.node_type.to_string());
    }
}

/// The type facts for `type_key`, resolved on first sight and reused after.
fn type_facts_for<'m>(
    memo: &'m mut HashMap<InternedKey, TypeFacts>,
    graph: &DirGraph,
    type_key: InternedKey,
) -> &'m TypeFacts {
    memo.entry(type_key).or_insert_with(|| {
        let name = graph.interner.resolve(type_key).to_string();
        TypeFacts {
            maintains_indexes: graph.type_has_user_indexes(&name),
            auto_timestamp: graph.auto_timestamp_for(&name),
            aliases: IdentityAliases::for_type(graph, &name),
            name,
        }
    })
}

/// What this row still owes for `(type_key, property)` at `value_type` — and
/// marks the pair recorded, so the next row of the same statement owes less.
///
/// The property key is the `&str` **the AST holds**, so a lookup allocates
/// nothing.
fn row_bookkeeping<'a>(
    memo: &mut HashMap<(InternedKey, &'a str), PropertyFacts>,
    type_key: InternedKey,
    property: &'a str,
    value_type: &'static str,
    type_facts: &TypeFacts,
    interner: &mut crate::graph::storage::interner::StringInterner,
) -> RowBookkeeping {
    use std::collections::hash_map::Entry;

    let mut register_schema_key = true;
    let mut record_metadata = Some(value_type);
    let key = match memo.entry((type_key, property)) {
        Entry::Occupied(mut seen) => {
            register_schema_key = false;
            let key = seen.get().key;
            if seen.get().recorded_type == value_type {
                record_metadata = None;
            } else {
                seen.insert(PropertyFacts {
                    recorded_type: value_type,
                    key,
                });
            }
            key
        }
        Entry::Vacant(slot) => {
            let key = interner.get_or_intern(property);
            slot.insert(PropertyFacts {
                recorded_type: value_type,
                key,
            });
            key
        }
    };
    RowBookkeeping {
        maintains_indexes: type_facts.maintains_indexes,
        register_schema_key,
        record_metadata,
        auto_timestamp: type_facts.auto_timestamp,
        key,
    }
}
