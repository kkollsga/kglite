//! Integrity reports: re-check every digest a `.kgl` carries, on demand.
//!
//! A load verifies compressed sections and string payloads eagerly and leaves
//! the CRC32 of mapped numeric and vector sections for first use (see
//! `Verify`). These functions run all of them now, for an operator who wants a
//! definite answer about a file ([`verify_file`]) or about the mapped data a
//! live graph is serving ([`verify_graph`]).

use std::io;
use std::path::Path;

use super::mappable::{MappableTable, MappedFile};
use super::*;

/// The outcome for one section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionStatus {
    /// Section name as the file records it (`topology`, `embeddings`,
    /// `emb:<Type>:<prop>:data`, ...).
    pub name: String,
    /// Payload bytes the check read.
    pub bytes: u64,
    /// Whether the data is served from a file mapping (graph reports only).
    pub mapped: bool,
    /// `None` when the section is intact.
    pub error: Option<String>,
}

/// Per-section results of an integrity check.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IntegrityReport {
    pub sections: Vec<SectionStatus>,
}

impl IntegrityReport {
    /// `true` when every section checked out.
    pub fn is_ok(&self) -> bool {
        self.sections.iter().all(|s| s.error.is_none())
    }
}

fn status(name: &str, bytes: u64, mapped: bool, result: io::Result<()>) -> SectionStatus {
    SectionStatus {
        name: name.to_string(),
        bytes,
        mapped,
        error: result.err().map(|e| e.to_string()),
    }
}

/// Check every section digest of the `.kgl` at `path` without loading it:
/// each compressed section's CRC32 and each mappable section's CRC32. A file
/// too damaged to read its metadata or section table is an `Err`; damage to a
/// section is reported in its [`SectionStatus`]. Disk-graph directories are
/// refused.
pub fn verify_file(path: &str) -> io::Result<IntegrityReport> {
    let p = Path::new(path);
    if p.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "verify_file checks a .kgl file; a disk-graph directory has no section digests",
        ));
    }
    let mapped = MappedFile::open(p)?;
    let buf = mapped.bytes();
    if buf.len() < 13 {
        return Err(invalid_data("File is too small to be a valid kglite file."));
    }
    let format_name = portable_container_format(buf, &format!("'{path}'"))?;
    let metadata_len = u32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]) as usize;
    if metadata_len > MAX_METADATA_BYTES {
        return Err(invalid_data(format!(
            "{format_name} metadata is {metadata_len} bytes; limit is {MAX_METADATA_BYTES}"
        )));
    }
    let (metadata, mut cursor) = parse_portable_metadata(buf, format_name, metadata_len, 13)?;
    let mut report = IntegrityReport::default();

    let mut compressed: Vec<(String, u64)> = vec![(
        TOPOLOGY_SECTION.to_string(),
        metadata.topology_compressed_size,
    )];
    for column in &metadata.column_sections {
        compressed.push((
            column_section_key(&column.type_name),
            column.compressed_size,
        ));
    }
    for (name, size) in [
        (EMBEDDINGS_SECTION, metadata.embeddings_compressed_size),
        (
            EDGE_EMBEDDINGS_SECTION,
            metadata.edge_embeddings_compressed_size,
        ),
        (TIMESERIES_SECTION, metadata.timeseries_compressed_size),
        (
            SECONDARY_LABELS_SECTION,
            metadata.secondary_labels_compressed_size,
        ),
        (VECTOR_INDEX_SECTION, metadata.vector_index_compressed_size),
        (TEXT_INDEX_SECTION, metadata.text_index_compressed_size),
        (
            EDGE_VECTOR_INDEX_SECTION,
            metadata.edge_vector_index_compressed_size,
        ),
        (
            EDGE_TEXT_INDEX_SECTION,
            metadata.edge_text_index_compressed_size,
        ),
    ] {
        compressed.push((name.to_string(), size));
    }
    for (name, size) in compressed.into_iter().filter(|(_, size)| *size > 0) {
        let result = cursor.take(size, &name).map(|_| ());
        let failed = result.is_err();
        report.sections.push(status(&name, size, false, result));
        if failed {
            // The cursor did not advance, so later sections cannot be located.
            return Ok(report);
        }
    }

    if !metadata.mappable.is_empty() {
        if format_name != "v9" {
            return Err(invalid_data(format!(
                "{format_name} container declares mappable sections, which only v9 containers carry"
            )));
        }
        let base = data_region_base(&metadata, 13 + metadata_len, buf.len() as u64)?;
        let table = MappableTable::open(
            &metadata.mappable,
            buf,
            base,
            Some(&mapped),
            Verify::Lazy,
            |_| true,
        )?;
        for section in table.iter() {
            report.sections.push(status(
                &section.name,
                section.as_bytes().len() as u64,
                true,
                section.verify(),
            ));
        }
    }
    Ok(report)
}

/// Check the mapped data `graph` is serving against the digests recorded when
/// its file was written, and that the file is still the one that was mapped
/// (same length, same modification time). Heap data has nothing to check, so
/// a graph with no mapped sections yields an empty (OK) report.
pub fn verify_graph(graph: &DirGraph) -> IntegrityReport {
    let mut report = IntegrityReport::default();
    let mut check = |label: String, store: &EmbeddingStore| {
        if !store.is_mapped() {
            return;
        }
        let result = store
            .verify_storage()
            .and_then(|()| match store.data.modified_since_open() {
                Some(Ok(true)) => Err(invalid_data(
                    "the source file's modification time changed since it was mapped",
                )),
                Some(Err(error)) => Err(error),
                _ => Ok(()),
            });
        report.sections.push(status(
            &label,
            (store.data.len() * 4
                + store.norms.len() * 4
                + store.quant_store().map_or(0, |q| q.byte_size())) as u64,
            true,
            result,
        ));
    };
    let mut node: Vec<_> = graph.embeddings.iter().collect();
    node.sort_by(|a, b| a.0.cmp(b.0));
    for (key, store) in node {
        check(format!("embeddings:{}.{}", key.0, key.1), store);
    }
    let mut edge: Vec<_> = graph.edge_embeddings.iter().collect();
    edge.sort_by(|a, b| a.0.cmp(b.0));
    for (key, store) in edge {
        check(
            format!("edge_embeddings:{}.{}", key.0, key.1),
            store.numeric(),
        );
    }
    if let Some(frozen) = graph.graph.frozen_base() {
        for section in frozen.shared_sections() {
            let result = section
                .verify()
                .and_then(|()| match section.modified_since_open() {
                    Some(Ok(true)) => Err(invalid_data(
                        "the source file's modification time changed since it was mapped",
                    )),
                    Some(Err(error)) => Err(error),
                    _ => Ok(()),
                });
            report.sections.push(status(
                section.name(),
                section.bytes().len() as u64,
                true,
                result,
            ));
        }
    }
    // Only a failed structural check adds a row, so a sound graph's report
    // lists the same sections as before.
    for failed in super::structure_checks::check_structure(graph) {
        report
            .sections
            .push(status(&failed.section, 0, true, Err(failed.error)));
    }
    let mut stores = graph.column_stores_by_name();
    stores.sort_by(|a, b| a.0.cmp(b.0));
    let mut seen = std::collections::HashSet::new();
    for (_, store) in stores {
        for section in store.shared_sections() {
            if !seen.insert(section.name().to_string()) {
                continue;
            }
            let result = section
                .verify()
                .and_then(|()| match section.modified_since_open() {
                    Some(Ok(true)) => Err(invalid_data(
                        "the source file's modification time changed since it was mapped",
                    )),
                    Some(Err(error)) => Err(error),
                    _ => Ok(()),
                });
            report.sections.push(status(
                section.name(),
                section.bytes().len() as u64,
                true,
                result,
            ));
        }
    }
    report
}
