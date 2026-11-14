//! Column buffers stored in a v9 container's data region instead of the
//! compressed column section.
//!
//! **Writer.** A save may *claim* a large column ([`ColumnSplit::claim`]): its
//! packed entry keeps the column's name and type tag with a [`MAP_TAG_SUFFIX`]
//! suffix and an empty payload, so slot order and `packed_column_names` stay
//! exactly as in v8, and the buffers become named sections (`data`/`nulls`,
//! or `offsets`/`bytes`/`nulls` for strings) that the container writer lays
//! out aligned and uncompressed.
//!
//! **Reader.** A `@map` entry is resolved through a [`ColumnSectionSource`]:
//! each part is either a read-only view of the mapped file
//! ([`PartData::Shared`]) or plain bytes to decode through the ordinary
//! heap/spill path ([`PartData::Bytes`]). String parts are validated here, at
//! load, because the row readers use `from_utf8_unchecked`.

use std::borrow::Cow;
use std::io;
use std::sync::Arc;

use super::*;
use crate::graph::io::file::mappable::{SectionDtype, SharedSection};

/// Type-tag suffix of a packed entry whose buffers live in the data region.
pub(crate) const MAP_TAG_SUFFIX: &str = "@map";

/// One buffer of a claimed column.
pub(crate) struct SplitPart<'a> {
    pub(crate) suffix: &'static str,
    pub(crate) dtype: SectionDtype,
    pub(crate) elem_count: u64,
    /// The bytes are UTF-8 the reader must re-validate at open.
    pub(crate) utf8: bool,
    pub(crate) bytes: Cow<'a, [u8]>,
}

/// A column a save moved out of the compressed section.
pub(crate) struct SplitColumn<'a> {
    pub(crate) name: String,
    pub(crate) parts: Vec<SplitPart<'a>>,
}

/// The columns of one store a save has claimed for the data region.
pub(crate) struct ColumnSplit<'a> {
    pub(crate) columns: Vec<SplitColumn<'a>>,
    claims: bool,
    /// Large columns the split could have taken and did not, with why: no
    /// section form, or a delta-compressed integer form that wins. Only a
    /// split that claims records them; `layout="compact"` asked for the
    /// compressed form.
    pub(crate) declined: Vec<(String, &'static str)>,
}

/// Rows from which a column is large enough that a decline is worth a report:
/// the row count at which a fixed-width column's buffer reaches
/// [`buffer_leaves_heap`].
const REPORT_MIN_ROWS: usize = super::typed_column::MMAP_THRESHOLD / 8;

impl<'a> ColumnSplit<'a> {
    /// A split that claims the columns a load would otherwise spill (see
    /// [`buffer_leaves_heap`]) when `claims` is set, and nothing otherwise.
    pub(crate) fn new(claims: bool) -> Self {
        ColumnSplit {
            columns: Vec::new(),
            claims,
            declined: Vec::new(),
        }
    }

    /// The same claims with their bytes copied out of the store they borrowed.
    pub(crate) fn into_owned(self) -> ColumnSplit<'static> {
        ColumnSplit {
            claims: self.claims,
            declined: self.declined,
            columns: self
                .columns
                .into_iter()
                .map(|column| SplitColumn {
                    name: column.name,
                    parts: column
                        .parts
                        .into_iter()
                        .map(|part| SplitPart {
                            suffix: part.suffix,
                            dtype: part.dtype,
                            elem_count: part.elem_count,
                            utf8: part.utf8,
                            bytes: Cow::Owned(part.bytes.into_owned()),
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    /// Record a column of `rows` rows that stays in the compressed section,
    /// when the split claims columns and the column is large.
    pub(crate) fn note_declined(&mut self, name: &str, rows: usize, reason: &'static str) {
        if self.claims && rows >= REPORT_MIN_ROWS {
            self.declined.push((name.to_string(), reason));
        }
    }

    /// Take `col` for the data region if it is a fixed-width or string column
    /// with a buffer that [`buffer_leaves_heap`]. Little-endian hosts only: the
    /// section bytes are the in-memory bytes.
    pub(crate) fn claim(&mut self, name: &str, col: &'a TypedColumn) -> bool {
        if !cfg!(target_endian = "little") {
            return false;
        }
        let Some(parts) = col.split_parts() else {
            self.note_declined(name, col.len(), no_section_form(col));
            return false;
        };
        if !self.claims
            || !parts
                .iter()
                .any(|part| buffer_leaves_heap(part.bytes.len()))
        {
            return false;
        }
        self.columns.push(SplitColumn {
            name: name.to_string(),
            parts,
        });
        true
    }
}

/// Why a column has no data-region form.
fn no_section_form(col: &TypedColumn) -> &'static str {
    match col {
        TypedColumn::Mixed { .. } => "values of mixed kinds have no raw form",
        TypedColumn::Str { relocated, .. } if !relocated.is_empty() => {
            "string cells were rewritten in place; a save of the reloaded graph writes the raw form"
        }
        _ => "buffers of inconsistent length have no raw form",
    }
}

fn part<'a, T: PackedElement>(
    suffix: &'static str,
    dtype: SectionDtype,
    values: &'a MmapOrVec<T>,
) -> SplitPart<'a> {
    SplitPart {
        suffix,
        dtype,
        elem_count: values.len() as u64,
        utf8: false,
        bytes: Cow::Borrowed(values.as_raw_bytes()),
    }
}

fn fixed_parts<'a, T: PackedElement>(
    dtype: SectionDtype,
    data: &'a MmapOrVec<T>,
    nulls: &'a MmapOrVec<u8>,
) -> Option<Vec<SplitPart<'a>>> {
    (data.len() == nulls.len()).then(|| {
        vec![
            part("data", dtype, data),
            part("nulls", SectionDtype::U8, nulls),
        ]
    })
}

impl TypedColumn {
    /// This column's buffers as data-region parts, or `None` when its shape
    /// has no section form (`Mixed`, a string column with relocated cells, a
    /// buffer set of inconsistent length).
    fn split_parts(&self) -> Option<Vec<SplitPart<'_>>> {
        match self {
            TypedColumn::Int64 { data, nulls } | TypedColumn::Timestamp { data, nulls } => {
                fixed_parts(SectionDtype::I64, data, nulls)
            }
            TypedColumn::Float64 { data, nulls } => fixed_parts(SectionDtype::F64, data, nulls),
            TypedColumn::UniqueId { data, nulls } => fixed_parts(SectionDtype::U32, data, nulls),
            TypedColumn::Bool { data, nulls } => fixed_parts(SectionDtype::U8, data, nulls),
            TypedColumn::Date { data, nulls } => fixed_parts(SectionDtype::I32, data, nulls),
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                relocated,
            } => {
                let consistent = relocated.is_empty()
                    && offsets.len() == nulls.len() + 1
                    && offsets.as_slice().last().copied() == Some(data.len() as u64);
                consistent.then(|| {
                    vec![
                        part("offsets", SectionDtype::U64, offsets),
                        SplitPart {
                            suffix: "bytes",
                            dtype: SectionDtype::U8,
                            elem_count: data.len() as u64,
                            utf8: true,
                            bytes: Cow::Borrowed(data.as_raw_bytes()),
                        },
                        part("nulls", SectionDtype::U8, nulls),
                    ]
                })
            }
            TypedColumn::Mixed { .. } => None,
        }
    }

    /// The mapped `.kgl` sections this column reads from.
    pub(crate) fn shared_sections(&self) -> Vec<&Arc<SharedSection>> {
        fn nulls_and<'c>(
            data: Option<&'c Arc<SharedSection>>,
            nulls: &'c MmapOrVec<u8>,
        ) -> Vec<&'c Arc<SharedSection>> {
            data.into_iter().chain(nulls.shared_section()).collect()
        }
        match self {
            TypedColumn::Int64 { data, nulls } | TypedColumn::Timestamp { data, nulls } => {
                nulls_and(data.shared_section(), nulls)
            }
            TypedColumn::Float64 { data, nulls } => nulls_and(data.shared_section(), nulls),
            TypedColumn::UniqueId { data, nulls } => nulls_and(data.shared_section(), nulls),
            TypedColumn::Bool { data, nulls } => nulls_and(data.shared_section(), nulls),
            TypedColumn::Date { data, nulls } => nulls_and(data.shared_section(), nulls),
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                ..
            } => offsets
                .shared_section()
                .into_iter()
                .chain(data.shared_section())
                .chain(nulls.shared_section())
                .collect(),
            TypedColumn::Mixed { .. } => Vec::new(),
        }
    }

    /// The first row whose null flag (or, for a bool column, value) is a byte
    /// other than 0 or 1, as `(what, row, byte)`. Readers treat any non-zero
    /// null byte as null, so such a byte is wrong without changing a read.
    pub(crate) fn first_bad_flag(&self) -> Option<(&'static str, usize, u8)> {
        let flags = |what, bytes: &[u8]| {
            bytes
                .iter()
                .position(|&b| b > 1)
                .map(|row| (what, row, bytes[row]))
        };
        match self {
            TypedColumn::Mixed { .. } => None,
            TypedColumn::Bool { data, nulls } => {
                flags("nulls", nulls.as_slice()).or_else(|| flags("data", data.as_slice()))
            }
            TypedColumn::Int64 { nulls, .. }
            | TypedColumn::Timestamp { nulls, .. }
            | TypedColumn::Float64 { nulls, .. }
            | TypedColumn::UniqueId { nulls, .. }
            | TypedColumn::Date { nulls, .. }
            | TypedColumn::Str { nulls, .. } => flags("nulls", nulls.as_slice()),
        }
    }

    /// Check the CRC32 of every mapped section this column reads from.
    pub(crate) fn verify_storage(&self) -> io::Result<()> {
        self.shared_sections()
            .into_iter()
            .try_for_each(|section| section.verify())
    }
}

impl ColumnStore {
    /// The first column of this store with a flag byte other than 0 or 1,
    /// described as `(column position, what, row, byte)`; the id and title
    /// columns come after the property columns.
    pub(crate) fn first_bad_flag(&self) -> Option<(usize, &'static str, usize, u8)> {
        self.columns
            .iter()
            .map(|c| &**c)
            .chain(self.id_column.as_deref())
            .chain(self.title_column.as_deref())
            .enumerate()
            .find_map(|(i, c)| c.first_bad_flag().map(|(what, row, b)| (i, what, row, b)))
    }

    /// Check the CRC32 of every mapped section this store's columns read from,
    /// so a corrupt section is refused before a save copies it into a new
    /// file.
    pub(crate) fn verify_storage(&self) -> io::Result<()> {
        self.columns
            .iter()
            .map(|c| &**c)
            .chain(self.id_column.as_deref())
            .chain(self.title_column.as_deref())
            .try_for_each(TypedColumn::verify_storage)
    }

    /// The mapped `.kgl` sections any column of this store reads from.
    pub(crate) fn shared_sections(&self) -> Vec<&Arc<SharedSection>> {
        self.columns
            .iter()
            .map(|c| &**c)
            .chain(self.id_column.as_deref())
            .chain(self.title_column.as_deref())
            .flat_map(TypedColumn::shared_sections)
            .collect()
    }

    /// Copy every column buffer that reads from a mapped `.kgl` onto the heap.
    pub(crate) fn promote_shared(&mut self) {
        let shared = |col: &Arc<TypedColumn>| !col.shared_sections().is_empty();
        for handle in self.columns.iter_mut().filter(|c| shared(c)) {
            Arc::make_mut(handle).materialize_to_heap();
        }
        for handle in [&mut self.id_column, &mut self.title_column]
            .into_iter()
            .flatten()
            .filter(|c| shared(c))
        {
            Arc::make_mut(handle).materialize_to_heap();
        }
    }
}

/// How [`ColumnStore::load_packed_sectioned`] decodes a payload: where large
/// decoded columns spill, the codec of `mixed` columns, and where the buffers
/// of `@map` columns come from (a payload with `@map` entries and no source is
/// refused).
#[derive(Clone, Copy)]
pub(crate) struct PackedDecode<'a> {
    pub(crate) temp_dir: Option<&'a Path>,
    pub(crate) codec: crate::serde_codec::CodecVersion,
    pub(crate) sections: Option<&'a dyn ColumnSectionSource>,
}

/// One buffer a `@map` entry names, as the loader receives it.
pub(crate) enum PartData<'a> {
    /// A read-only view of the mapped file.
    Shared(Arc<SharedSection>),
    /// Bytes to decode into an ordinary buffer.
    Bytes(&'a [u8]),
}

impl PartData<'_> {
    /// Drop the resident pages of a mapped part once it has been validated.
    fn release_resident(&self) {
        if let PartData::Shared(section) = self {
            section.release_resident();
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            PartData::Shared(section) => section.bytes(),
            PartData::Bytes(bytes) => bytes,
        }
    }
}

/// Where a loader finds the buffers of the `@map` columns of one store.
pub(crate) trait ColumnSectionSource {
    /// The `suffix` buffer of `column`: refused unless it exists, holds
    /// `dtype` elements and, when `expected` is given, that many of them.
    fn part(
        &self,
        column: &str,
        suffix: &str,
        dtype: SectionDtype,
        expected: Option<u64>,
    ) -> io::Result<PartData<'_>>;
}

fn mapped_vec<T: PackedElement>(
    data: PartData<'_>,
    len: usize,
    temp_dir: Option<&Path>,
    col_name: &str,
    ext: &str,
) -> io::Result<MmapOrVec<T>> {
    match data {
        PartData::Shared(section) => match MmapOrVec::<T>::shared(Arc::clone(&section)) {
            Some(view) if view.len() == len => Ok(view),
            _ => ColumnStore::load_typed_vec::<T>(section.bytes(), len, temp_dir, col_name, ext),
        },
        PartData::Bytes(bytes) => {
            ColumnStore::load_typed_vec::<T>(bytes, len, temp_dir, col_name, ext)
        }
    }
}

impl ColumnStore {
    /// Resolve a `@map` packed entry of base type `base_tag` into a column.
    pub(super) fn unpack_mapped_column(
        base_tag: &str,
        source: &dyn ColumnSectionSource,
        rows: usize,
        temp_dir: Option<&Path>,
        col_name: &str,
    ) -> io::Result<TypedColumn> {
        fn fixed<T: PackedElement>(
            source: &dyn ColumnSectionSource,
            rows: usize,
            temp_dir: Option<&Path>,
            col_name: &str,
            dtype: SectionDtype,
            ext: &str,
        ) -> io::Result<(MmapOrVec<T>, MmapOrVec<u8>)> {
            let want = Some(rows as u64);
            let data = mapped_vec::<T>(
                source.part(col_name, "data", dtype, want)?,
                rows,
                temp_dir,
                col_name,
                ext,
            )?;
            let nulls = mapped_vec::<u8>(
                source.part(col_name, "nulls", SectionDtype::U8, want)?,
                rows,
                temp_dir,
                col_name,
                "null",
            )?;
            Ok((data, nulls))
        }
        let args = (source, rows, temp_dir, col_name);
        match base_tag {
            "int64" => {
                let (data, nulls) =
                    fixed::<i64>(args.0, args.1, args.2, args.3, SectionDtype::I64, "i64")?;
                Ok(TypedColumn::Int64 { data, nulls })
            }
            "timestamp" => {
                let (data, nulls) =
                    fixed::<i64>(args.0, args.1, args.2, args.3, SectionDtype::I64, "ts")?;
                Ok(TypedColumn::Timestamp { data, nulls })
            }
            "float64" => {
                let (data, nulls) =
                    fixed::<f64>(args.0, args.1, args.2, args.3, SectionDtype::F64, "f64")?;
                Ok(TypedColumn::Float64 { data, nulls })
            }
            "uniqueid" => {
                let (data, nulls) =
                    fixed::<u32>(args.0, args.1, args.2, args.3, SectionDtype::U32, "u32")?;
                Ok(TypedColumn::UniqueId { data, nulls })
            }
            "bool" | "boolean" => {
                let (data, nulls) =
                    fixed::<u8>(args.0, args.1, args.2, args.3, SectionDtype::U8, "bool")?;
                Ok(TypedColumn::Bool { data, nulls })
            }
            "date" | "datetime" => {
                let (data, nulls) =
                    fixed::<i32>(args.0, args.1, args.2, args.3, SectionDtype::I32, "i32")?;
                Ok(TypedColumn::Date { data, nulls })
            }
            "string" => Self::unpack_mapped_string(source, rows, temp_dir, col_name),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("column '{col_name}' has a mapped type '{other}' this build cannot read"),
            )),
        }
    }

    /// A string column from its three parts. The offsets, the UTF-8 and the
    /// offset/char-boundary agreement are all checked here, whether the parts
    /// are mapped or copied: the row readers do not re-check.
    fn unpack_mapped_string(
        source: &dyn ColumnSectionSource,
        rows: usize,
        temp_dir: Option<&Path>,
        col_name: &str,
    ) -> io::Result<TypedColumn> {
        let offsets = source.part(
            col_name,
            "offsets",
            SectionDtype::U64,
            Some(rows as u64 + 1),
        )?;
        let offset_bytes = offsets.bytes();
        let last = offset_bytes
            .as_chunks::<8>()
            .0
            .last()
            .map(|chunk| u64::from_le_bytes(*chunk))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("column '{col_name}' (string) has no offsets"),
                )
            })?;
        let text = source.part(col_name, "bytes", SectionDtype::U8, Some(last))?;
        let validated = std::str::from_utf8(text.bytes()).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("column '{col_name}' contains invalid UTF-8: {e}"),
            )
        })?;
        Self::validate_string_offsets(offset_bytes, validated, last, col_name)?;
        // The checksum and the validation above read every page of both parts;
        // a query refaults only the rows it touches.
        text.release_resident();
        offsets.release_resident();
        let nulls = source.part(col_name, "nulls", SectionDtype::U8, Some(rows as u64))?;
        let data = match text {
            PartData::Shared(section) => MmapBytes::shared(section),
            PartData::Bytes(bytes) => Self::load_bytes(bytes, temp_dir, col_name, "str")?,
        };
        Ok(TypedColumn::Str {
            offsets: mapped_vec::<u64>(offsets, rows + 1, temp_dir, col_name, "off")?,
            data,
            nulls: mapped_vec::<u8>(nulls, rows, temp_dir, col_name, "null")?,
            relocated: rustc_hash::FxHashMap::default(),
        })
    }
}
