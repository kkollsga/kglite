//! Edge-property column plumbing shared by the connection ingest paths in
//! `maintain.rs`: once-per-call column resolution, deferred interner
//! registration, and per-chain interning for `create_connections`.

use crate::datatypes::{DataFrame, Value};
use crate::graph::schema::InternedKey;
use crate::graph::storage::interner::StringInterner;
use std::collections::{HashMap, HashSet};

/// Resolve a connection frame's property columns once per call — name, FNV
/// key and positional index — excluding the endpoint id/title columns.
///
/// The key is computed with `InternedKey::from_str` (the same pure FNV that
/// `get_or_intern` uses) rather than by interning, because interning would
/// register an all-null column's *name* in the persisted interner table.
/// `add_connections` registers the names of the columns that actually carried
/// a value after its passes; no backend resolves an edge key during
/// `add_edge`, so registration only has to precede the first `resolve`
/// (schema metadata and serialization, both later).
pub(super) fn resolve_edge_property_columns(
    df_data: &DataFrame,
    source_id_field: &str,
    target_id_field: &str,
    source_title_field: Option<&str>,
    target_title_field: Option<&str>,
) -> Vec<(String, InternedKey, usize)> {
    df_data
        .get_column_names()
        .into_iter()
        .filter(|col_name| {
            let name = col_name.as_str();
            name != source_id_field
                && name != target_id_field
                && Some(name) != source_title_field
                && Some(name) != target_title_field
        })
        .filter_map(|col_name| {
            df_data.get_column_index(&col_name).map(|col_idx| {
                let key = InternedKey::from_str(&col_name);
                (col_name, key, col_idx)
            })
        })
        .collect()
}

/// Register the names of the property columns that actually carried a value,
/// so `resolve` (schema metadata, then serialization) can recover them from
/// the pure-hash keys the passes stored. A column that was null in every row
/// contributed no key, so it stays unregistered — the all-null column leaves
/// no trace, which is the contract
/// `all_null_edge_property_column_stores_nothing` pins.
pub(super) fn register_used_edge_property_names(
    interner: &mut StringInterner,
    property_columns: &[(String, InternedKey, usize)],
    used: &HashSet<InternedKey>,
) {
    for (col_name, key, _) in property_columns {
        if used.contains(key) {
            interner.get_or_intern(col_name);
        }
    }
}

/// Intern a per-chain edge property set. Unlike the frame paths, which
/// resolve keys once per call, `create_connections` copies node property
/// names discovered per chain, so the keys intern at use.
pub(super) fn intern_edge_props(
    props: HashMap<String, Value>,
    interner: &mut StringInterner,
) -> Vec<(InternedKey, Value)> {
    props
        .into_iter()
        .map(|(k, v)| (interner.get_or_intern(&k), v))
        .collect()
}
