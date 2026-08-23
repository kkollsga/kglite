//! `.kgl` container magic bytes, and the refusals an unreadable header earns.
//!
//! Recognising a container version and explaining a header this binary cannot
//! read are one concern, and it lives here rather than in `io/file.rs`.
//!
//! Every kglite single-file container since v3 begins `RGF` followed by a
//! one-byte version. That prefix is the only thing separating "an old kglite
//! file" from "not a kglite file at all" — the two cases that want opposite
//! advice, and that shared one message until this module existed.

use std::io;

/// Magic bytes for the v3 columnar format: "RGF\x03". Retained ONLY
/// so the loader can detect a v3 file and emit a specific
/// "rebuild your graph" error rather than a generic "unrecognized".
pub(crate) const V3_MAGIC: [u8; 4] = [0x52, 0x47, 0x46, 0x03];

/// Magic bytes for the v4 columnar format: "RGF\x04". v4 arrived
/// alongside the `Value::Node`/`Relationship`/`Path`/`List`/`Map` enum
/// extension. Hard break on v3 files (no read-compat path) per
/// docs/history/bolt-implementation.md.
pub(crate) const V4_MAGIC: [u8; 4] = [0x52, 0x47, 0x46, 0x04];

/// Magic bytes for the v5 columnar format. v5 retains the v4 section layout
/// but adds an explicit codec tag and writes Serde payloads with Postcard.
/// Still read (v5 files outlive the binary that wrote them); no longer
/// written.
pub(crate) const V5_MAGIC: [u8; 4] = [0x52, 0x47, 0x46, 0x05];

/// Magic bytes for the v6 columnar format — what this binary writes. v6 is v5
/// plus per-column integer encodings inside the packed column sections (see
/// `io::file`'s module header). Both are decoded by the same reader.
pub(crate) const V6_MAGIC: [u8; 4] = [0x52, 0x47, 0x46, 0x06];

/// Hard-break message for v3 files in a v4 binary. Per the
/// user decision in docs/history/bolt-implementation.md: no read-compat
/// path; rebuild the graph from source. Message gives the operator
/// enough breadcrumbs to know what changed and what to do.
pub(crate) const V3_HARD_BREAK_MSG: &str = "kglite .kgl file format v3 is not supported by this \
     binary. It predates the current RGF v5/Postcard container; the Value enum gained \
     structured Node / Relationship / Path / List / Map variants, which changes \
     the serialised property representation. Rebuild your graph from its \
     original source (CSV, DataFrame, dataset loader) and save again, \
     or downgrade kglite to the 0.9.x line if you need to read this \
     file. If you no longer have the original source but can still run \
     the old binary, open the file there and export a portable, \
     format-stable copy with g.export_csv('backup/'), then rebuild here \
     with kglite.from_blueprint('backup/blueprint.json').";

pub(crate) fn newer_portable_format_error(version: u8) -> io::Error {
    io::Error::other(format!(
        "File uses .kgl container version {version}, but this library only supports up to version {}. Please upgrade kglite.",
        V6_MAGIC[3]
    ))
}

/// Classify a header prefix that matched no supported container magic.
///
/// Two unrelated situations reach this point and want opposite advice, and
/// until 0.16.1 both got the second one:
///
/// - The bytes *are* a kglite container (`RGF` + a version this binary has no
///   reader for). Rebuilding from source and re-saving is exactly right.
/// - The bytes are something else entirely — a PNG, a CSV, a half-finished
///   download, the wrong path. Telling that user their file "was saved with an
///   older version of kglite" is a false statement about their data, and sends
///   them looking for an old binary that never existed.
///
/// The `RGF` prefix is the whole discriminator: every kglite single-file
/// container has carried it since v3, and disk-mode graphs are directories,
/// resolved before any header is read.
///
/// `origin` names what was refused (a quoted path, or "the byte buffer") so
/// the message is actionable when several files are being loaded at once.
pub(crate) fn unrecognized_magic_error(prefix: &[u8], origin: &str) -> io::Error {
    if prefix.len() >= 3 && prefix[..3] == V6_MAGIC[..3] {
        return io::Error::other(format!(
            "Unrecognized .kgl container version {} in {origin}. This file was saved with an \
             older version of kglite. Please rebuild the graph with the current version and \
             save again. If you no longer have the original source but can still run the old \
             binary, open the file there and export a portable copy with g.export_csv('backup/'), \
             then rebuild here with kglite.from_blueprint('backup/blueprint.json').",
            prefix.get(3).copied().unwrap_or(0)
        ));
    }
    io::Error::other(format!(
        "{origin} is not a kglite graph: it does not start with the .kgl container magic \
         \"RGF\" (first bytes: {}). Check the path — a disk-mode graph is a directory, not a \
         file inside it, and every .kgl kglite has ever written begins with RGF.",
        describe_header_bytes(prefix)
    ))
}

/// Render a header prefix for a human: hex, plus the ASCII spelling when the
/// bytes are printable (which is how a user recognises "oh, that's my CSV").
fn describe_header_bytes(prefix: &[u8]) -> String {
    let hex: Vec<String> = prefix.iter().map(|b| format!("{b:02x}")).collect();
    let hex = hex.join(" ");
    if prefix.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        format!("{hex} = \"{}\"", String::from_utf8_lossy(prefix))
    } else {
        hex
    }
}
