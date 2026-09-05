//! Replay retains integer variants even in columns that ordinary ingest may
//! coerce to floats. Preparation is private to the unpublished replay batch.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{ColumnStore, TypedColumn};
use crate::datatypes::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::interner::StringInterner;

#[derive(Default)]
struct NumericShape {
    integer: bool,
    float: bool,
}
impl NumericShape {
    fn observe(&mut self, value: &Value) {
        self.integer |= matches!(value, Value::Int64(_));
        self.float |= matches!(value, Value::Float64(_));
    }
}

#[derive(Default)]
pub(crate) struct ExactValueColumns {
    ids: NumericShape,
    titles: NumericShape,
    integer_properties: HashSet<InternedKey>,
}

impl ExactValueColumns {
    pub(crate) fn note_identity(&mut self, id: &Value, title: &Value) {
        self.ids.observe(id);
        self.titles.observe(title);
    }
    pub(crate) fn note_property(&mut self, key: InternedKey, value: &Value) {
        if matches!(value, Value::Int64(_)) {
            self.integer_properties.insert(key);
        }
    }
}

impl ColumnStore {
    pub(crate) fn prepare_exact_values(
        &mut self,
        incoming: &ExactValueColumns,
        metadata: &HashMap<String, String>,
        interner: &StringInterner,
    ) {
        // A packed mmap base has no local property columns. Materialize once,
        // before updates as well as creates, so every exact setter addresses a
        // writable local column and snapshots retain their original mapping.
        self.materialize_for_append(metadata, interner);
        exact_identity_column(&mut self.id_column, &incoming.ids);
        exact_identity_column(&mut self.title_column, &incoming.titles);
        let mut keys: Vec<_> = incoming.integer_properties.iter().copied().collect();
        keys.sort_unstable_by_key(|key| key.as_u64());
        for key in keys {
            let slot = match self.schema.slot(key) {
                Some(slot) => slot,
                None => {
                    // The graph's schema can be wider than this existing
                    // store. Precreate from the complete batch's metadata:
                    // first-Float/later-Int must start Mixed, not infer Float
                    // from the first row and silently coerce the second.
                    let kind = metadata
                        .get(interner.resolve(key))
                        .map(String::as_str)
                        .unwrap_or("mixed");
                    self.append_column_typed(key, kind)
                }
            };
            if matches!(
                self.column(slot as usize),
                Some(TypedColumn::Float64 { .. })
            ) {
                self.demote_to_mixed(slot as usize);
            }
        }
    }
}

fn exact_identity_column(column: &mut Option<Arc<TypedColumn>>, incoming: &NumericShape) {
    if !incoming.integer {
        return;
    }
    let float_column = matches!(column.as_deref(), Some(TypedColumn::Float64 { .. }));
    let new_mixed_column = column.is_none() && incoming.float;
    if float_column || new_mixed_column {
        let data = column.as_ref().map_or_else(Vec::new, |column| {
            (0..column.len())
                .map(|row| column.get(row as u32).unwrap_or(Value::Null))
                .collect()
        });
        *column = Some(Arc::new(TypedColumn::Mixed { data }));
    }
}
