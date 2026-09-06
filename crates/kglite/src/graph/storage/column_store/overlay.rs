//! Effective values over immutable bases; explicit clears never mark padded cells.

use super::*;
use std::borrow::Cow;

type OverflowBytes<'a> = (Cow<'a, [u8]>, Cow<'a, [u8]>);

impl ColumnStore {
    pub(super) fn is_null_override(&self, row: u32, key: InternedKey) -> bool {
        self.null_overrides.as_ref().is_some_and(|clears| {
            self.schema
                .slot(key)
                .is_some_and(|slot| clears.contains(&(row, slot)))
        })
    }

    pub(super) fn overrides_base(&self, row: u32, key: InternedKey) -> bool {
        self.schema.slot(key).is_some_and(|slot| {
            self.columns
                .get(slot as usize)
                .is_some_and(|column| column.is_present(row))
                || self
                    .null_overrides
                    .as_ref()
                    .is_some_and(|clears| clears.contains(&(row, slot)))
        })
    }

    pub(super) fn record_null_override(&mut self, row: u32, slot: u16, value: &Value) {
        if matches!(value, Value::Null) && (self.mmap_store.is_some() || self.has_overflow()) {
            Arc::make_mut(
                self.null_overrides
                    .get_or_insert_with(|| Arc::new(HashSet::new())),
            )
            .insert((row, slot));
        } else if self
            .null_overrides
            .as_ref()
            .is_some_and(|clears| clears.contains(&(row, slot)))
        {
            let clears = self.null_overrides.as_mut().expect("present clear");
            Arc::make_mut(clears).remove(&(row, slot));
            if clears.is_empty() {
                self.null_overrides = None;
            }
        }
    }

    pub(super) fn prune_null_overrides(&mut self, rows: u32, columns: usize) {
        if let Some(clears) = self.null_overrides.as_mut() {
            if clears
                .iter()
                .any(|&(row, slot)| row >= rows || slot as usize >= columns)
            {
                Arc::make_mut(clears)
                    .retain(|&(row, slot)| row < rows && (slot as usize) < columns);
            }
            if clears.is_empty() {
                self.null_overrides = None;
            }
        }
    }

    pub(super) fn reconcile_replacement_nulls(&mut self, columns: &[TypedColumn]) {
        self.prune_null_overrides(self.row_count, columns.len());
        if self.mmap_store.is_none() && !self.has_overflow() {
            return;
        }
        for (slot, column) in columns.iter().enumerate() {
            for row in 0..column.len().min(self.row_count as usize) as u32 {
                let value = if column.is_present(row) {
                    Value::Boolean(true)
                } else {
                    Value::Null
                };
                self.record_null_override(row, slot as u16, &value);
            }
        }
    }

    pub(super) fn null_override_heap_bytes(&self) -> usize {
        self.null_overrides.as_ref().map_or(0, |clears| {
            // Hash-table cells plus control bytes; these cannot be reclaimed by spilling columns.
            clears.capacity() * (std::mem::size_of::<(u32, u16)>() + 1)
        })
    }

    fn base_overflow_bytes(&self) -> Option<OverflowBytes<'_>> {
        if let Some(ms) = self.mmap_store.as_ref() {
            if !ms.has_overflow || ms.overflow_offsets.len == 0 {
                return None;
            }
            let offsets = &ms.overflow_offsets;
            let data = &ms.overflow_data;
            return Some((
                Cow::Borrowed(&ms.mmap[offsets.offset..offsets.offset + offsets.len]),
                Cow::Borrowed(&ms.mmap[data.offset..data.offset + data.len]),
            ));
        }
        Some((
            Cow::Borrowed(self.overflow_offsets.as_ref()?.as_raw_bytes()),
            Cow::Borrowed(self.overflow_data.as_ref()?.as_raw_bytes()),
        ))
    }

    /// Coherent overflow output for writers of the current local columns.
    pub(crate) fn effective_overflow_bytes(&self) -> Option<OverflowBytes<'_>> {
        if self.columns.is_empty() && self.null_overrides.is_none() {
            return self.base_overflow_bytes();
        }
        self.filtered_overflow_bytes(|row, key| self.overrides_base(row, key))
    }

    fn filtered_overflow_bytes(
        &self,
        mut overridden: impl FnMut(u32, InternedKey) -> bool,
    ) -> Option<OverflowBytes<'_>> {
        let (offsets, data) = self.base_overflow_bytes()?;
        let mut output_offsets = Vec::with_capacity(offsets.len());
        let mut output_data = Vec::with_capacity(data.len());
        output_offsets.extend_from_slice(&0u64.to_le_bytes());
        for (row, pair) in offsets.windows(16).step_by(8).enumerate() {
            let start = u64::from_le_bytes(pair[..8].try_into().expect("offset width")) as usize;
            let end = u64::from_le_bytes(pair[8..].try_into().expect("offset width")) as usize;
            if let Some(blob) = data.get(start..end) {
                crate::graph::storage::overflow::append_filtered_blob(
                    blob,
                    &mut output_data,
                    |key| !overridden(row as u32, key),
                );
            }
            output_offsets.extend_from_slice(&(output_data.len() as u64).to_le_bytes());
        }
        Some((Cow::Owned(output_offsets), Cow::Owned(output_data)))
    }

    pub(super) fn write_overflow_columns(buf: &mut Vec<u8>, overflow: Option<&OverflowBytes<'_>>) {
        if let Some((offsets, data)) = overflow {
            for (name, bytes) in [
                ("__overflow_offsets__", offsets),
                ("__overflow_data__", data),
            ] {
                buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
                buf.extend_from_slice(name.as_bytes());
                buf.extend_from_slice(&3u16.to_le_bytes());
                buf.extend_from_slice(b"raw");
                buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                buf.extend_from_slice(bytes);
            }
        }
    }

    /// Flatten base and local columns without retaining superseded overflow entries.
    pub(super) fn write_packed_from_mmap(
        &self,
        base: &crate::graph::storage::mapped::column_store::MmapColumnStore,
        interner: &StringInterner,
        codec: crate::serde_codec::CodecVersion,
    ) -> io::Result<Vec<u8>> {
        let rows = self.row_count;
        let mut keys: Vec<_> = base.col_map.keys().copied().collect();
        let mut seen: HashSet<_> = keys.iter().copied().collect();
        for (_, key) in self.schema.iter() {
            if seen.insert(key) {
                keys.push(key);
            }
        }
        let columns: Vec<_> = keys
            .iter()
            .map(|&key| TypedColumn::Mixed {
                data: (0..rows)
                    .map(|row| self.get(row, key).unwrap_or(Value::Null))
                    .collect(),
            })
            .collect();
        let slots: HashMap<_, _> = keys
            .iter()
            .enumerate()
            .map(|(slot, &key)| (key, slot))
            .collect();
        // Materialization can promote an untouched overflow value into an output column.
        // Its encoded bag entry must then disappear too, or reloaded rows repeat the key.
        let overflow = self.filtered_overflow_bytes(|row, key| {
            slots
                .get(&key)
                .is_some_and(|&slot| columns[slot].is_present(row))
                || self.is_null_override(row, key)
        });
        let mut buf = Vec::new();
        let count = columns.len() as u32 + 2 + if overflow.is_some() { 2 } else { 0 };
        buf.extend_from_slice(&count.to_le_bytes());
        for (key, column) in keys.into_iter().zip(columns) {
            Self::write_packed_column(
                &mut buf,
                interner.resolve(key),
                &column,
                codec,
                IntColumnEncoding::Raw,
            )?;
        }
        let ids = TypedColumn::Mixed {
            data: (0..rows)
                .map(|row| self.get_id(row).unwrap_or(Value::Null))
                .collect(),
        };
        let titles = TypedColumn::Mixed {
            data: (0..rows)
                .map(|row| self.get_title(row).unwrap_or(Value::Null))
                .collect(),
        };
        Self::write_packed_column(&mut buf, "__id__", &ids, codec, IntColumnEncoding::Raw)?;
        Self::write_packed_column(
            &mut buf,
            "__title__",
            &titles,
            codec,
            IntColumnEncoding::Raw,
        )?;
        Self::write_overflow_columns(&mut buf, overflow.as_ref());
        Ok(buf)
    }
}
