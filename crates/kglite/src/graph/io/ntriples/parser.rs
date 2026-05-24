//! N-Triples parser — line-level parsing and literal type inference.
//!
//! Triple AST (`Subject` / `Predicate` / `Object` / `EdgeBuffer`),
//! XSD literal → `Value` coercion, Wikidata Q-code helpers, and
//! language-filter utilities.

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::mapped::mmap_vec::MmapOrVec;
use std::collections::{HashMap, HashSet};

// ─── Parsed triple ──────────────────────────────────────────────────────────

/// The subject of a triple (only Wikidata Q-entities are interesting).
/// Borrows from the input line for zero-copy parsing.
pub(super) enum Subject<'a> {
    Entity(&'a str), // Q-code slice, e.g. "Q42"
    Other,
}

/// The predicate of a triple. Borrows P-code from input line.
pub(super) enum Predicate<'a> {
    WikidataDirect(&'a str), // P-code slice, e.g. "P31"
    Label,
    Description,
    AltLabel,
    Type,
    Other,
}

/// The object of a triple.
/// Entity and typed/lang metadata borrow from input line.
/// Literal values are owned (may contain escape sequences).
pub(super) enum Object<'a> {
    Entity(&'a str),               // Q-code slice
    Literal(String),               // owned (escape-processed)
    LangLiteral(String, &'a str),  // (owned value, borrowed lang)
    TypedLiteral(String, &'a str), // (owned value, borrowed type_uri)
    Other,
}

// ─── Line batch (cache-friendly transport from reader → loader) ────────────

/// A batch of N-triples lines packed into one contiguous buffer plus an
/// offset table. Replaces `Vec<String>` for the reader-thread → loader
/// channel: one allocation per batch instead of 200k separately-heap-
/// allocated `String` objects, and the loader iterates via byte-offset
/// math instead of pointer-chasing scattered heap addresses.
///
/// Each entry occupies `data[offsets[i]..offsets[i + 1]]` (or
/// `data[offsets.last()..data.len()]` for the final line). Lines retain
/// their trailing `\n`; `parse_line` already trims it.
pub(super) struct LineBuffer {
    pub(super) data: Vec<u8>,
    pub(super) offsets: Vec<u32>,
}

impl LineBuffer {
    pub(super) fn with_capacity(line_cap: usize, byte_cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(byte_cap),
            offsets: Vec::with_capacity(line_cap),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    pub(super) fn push_line(&mut self, line: &[u8]) {
        let start = self.data.len() as u32;
        self.data.extend_from_slice(line);
        self.offsets.push(start);
    }

    /// Byte slice of line `i`. Caller must ensure `i < self.offsets.len()`.
    #[inline]
    pub(super) fn line(&self, i: usize) -> &[u8] {
        let start = self.offsets[i] as usize;
        let end = if i + 1 < self.offsets.len() {
            self.offsets[i + 1] as usize
        } else {
            self.data.len()
        };
        &self.data[start..end]
    }
}

// ─── Edge buffer ────────────────────────────────────────────────────────────

/// Edge buffer: String-based (default mode) or compact u32-based (mapped mode).
pub(super) enum EdgeBuffer {
    /// Default mode: (source_qcode, target_qcode, predicate_label) — ~80 bytes each
    Strings(Vec<(String, String, String)>),
    /// Mapped mode: (source_qnum, target_qnum, interned_predicate) — 16 bytes each.
    /// File-backed (MmapOrVec) in disk mode to avoid holding ~14 GB in RAM.
    Compact(MmapOrVec<(u32, u32, InternedKey)>),
}

impl EdgeBuffer {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Strings(v) => v.len(),
            Self::Compact(v) => v.len(),
        }
    }
}

/// Parse "Q42" → 42. Returns None for non-Q-code or overflow.
pub(super) fn parse_qcode_number(qcode: &str) -> Option<u32> {
    qcode.strip_prefix('Q')?.parse::<u32>().ok()
}

// ─── Accumulated entity ─────────────────────────────────────────────────────

/// Collects all triples for one subject entity before flushing to graph.
pub(super) struct EntityAccumulator {
    pub(super) id: String, // Q-code
    pub(super) label: Option<String>,
    pub(super) description: Option<String>,
    pub(super) type_qcode: Option<String>, // P31 target, e.g. "Q5"
    pub(super) properties: HashMap<String, Value>,
    pub(super) outgoing_edges: Vec<(String, String)>, // (predicate, target Q-code)
}

impl EntityAccumulator {
    pub(super) fn new(id: String) -> Self {
        // Preallocate typical Wikidata entity sizes — most have 10-30
        // properties and a handful of outgoing edges. Avoids the
        // `RawVecInner::finish_grow` reallocs the loader profile showed
        // at ~2% of total CPU.
        Self {
            id,
            label: None,
            description: None,
            type_qcode: None,
            properties: HashMap::with_capacity(32),
            outgoing_edges: Vec::with_capacity(8),
        }
    }
}

// ─── Line parser ────────────────────────────────────────────────────────────

const WD_ENTITY: &str = "http://www.wikidata.org/entity/";
const WD_PROP_DIRECT: &str = "http://www.wikidata.org/prop/direct/";
const RDFS_LABEL: &str = "http://www.w3.org/2000/01/rdf-schema#label";
const SCHEMA_DESC: &str = "http://schema.org/description";
const SKOS_ALT: &str = "http://www.w3.org/2004/02/skos/core#altLabel";
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

/// Parse a single N-Triples line into (subject, predicate, object).
/// Returns borrowed slices into the input line — zero heap allocations.
///
/// Hot-path implementation: scans at the byte level using `memchr` to
/// locate URI boundaries (`>`), avoiding `str::find("> ")`'s
/// `TwoWaySearcher` setup which showed up at ~10% of loader CPU under
/// `samply`. N-triples URIs are guaranteed not to contain `>`, so a
/// single `memchr(b'>')` is sufficient — no two-byte needle search.
pub(super) fn parse_line(line: &str) -> Option<(Subject<'_>, Predicate<'_>, Object<'_>)> {
    // Manual byte-level trim — `str::trim()` has O(n) startup overhead.
    let bytes = line.as_bytes();
    let mut start = 0;
    while start < bytes.len() && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    let mut end = bytes.len();
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    if start >= end {
        return None;
    }
    let bytes = &bytes[start..end];

    // Skip comments + non-URI lines fast.
    if bytes[0] == b'#' || bytes[0] != b'<' {
        return None;
    }

    // Subject URI: find closing `>` at any offset >= 1. URIs in N-triples
    // cannot contain `>`, so a single memchr is correct without the
    // two-byte "> " search the previous version used.
    let subj_end = 1 + memchr::memchr(b'>', &bytes[1..])?;
    if bytes.get(subj_end + 1) != Some(&b' ') {
        return None;
    }
    // SAFETY: we sliced from a valid &str on byte boundaries; URI text
    // is ASCII so the boundary is preserved.
    let subj_uri = unsafe { std::str::from_utf8_unchecked(&bytes[1..subj_end]) };
    let subject = if let Some(qcode) = subj_uri.strip_prefix(WD_ENTITY) {
        if qcode.starts_with('Q') {
            Subject::Entity(qcode)
        } else {
            Subject::Other
        }
    } else {
        Subject::Other
    };

    // Predicate URI starts at subj_end + 2 ("> ").
    let pred_start = subj_end + 2;
    if bytes.get(pred_start) != Some(&b'<') {
        return None;
    }
    let pred_end_rel = memchr::memchr(b'>', &bytes[pred_start + 1..])?;
    let pred_end = pred_start + 1 + pred_end_rel;
    if bytes.get(pred_end + 1) != Some(&b' ') {
        return None;
    }
    // SAFETY: `bytes` originates from an N-Triples line read as UTF-8;
    // valid byte ranges within it remain valid UTF-8.
    let pred_uri = unsafe { std::str::from_utf8_unchecked(&bytes[pred_start + 1..pred_end]) };
    let predicate = if let Some(pcode) = pred_uri.strip_prefix(WD_PROP_DIRECT) {
        if pcode.starts_with('P') {
            Predicate::WikidataDirect(pcode)
        } else {
            Predicate::Other
        }
    } else if pred_uri == RDFS_LABEL {
        Predicate::Label
    } else if pred_uri == SCHEMA_DESC {
        Predicate::Description
    } else if pred_uri == SKOS_ALT {
        Predicate::AltLabel
    } else if pred_uri == RDF_TYPE {
        Predicate::Type
    } else {
        Predicate::Other
    };

    // Object section — strip trailing whitespace + final `.`.
    let mut obj_end = bytes.len();
    while obj_end > 0 && bytes[obj_end - 1].is_ascii_whitespace() {
        obj_end -= 1;
    }
    if obj_end == 0 || bytes[obj_end - 1] != b'.' {
        return None;
    }
    obj_end -= 1;
    while obj_end > 0 && bytes[obj_end - 1].is_ascii_whitespace() {
        obj_end -= 1;
    }
    let obj_start = pred_end + 2;
    if obj_start >= obj_end {
        return None;
    }
    // The object portion may include UTF-8 inside a quoted literal, so
    // we use checked from_utf8 here (parse_object expects &str).
    let obj_str = std::str::from_utf8(&bytes[obj_start..obj_end]).ok()?;
    let object = parse_object(obj_str);

    Some((subject, predicate, object))
}

/// Parse the object portion of an N-Triples line.
/// Entity Q-codes and lang/type tags borrow from the input. Literal values are owned.
pub(super) fn parse_object<'a>(s: &'a str) -> Object<'a> {
    if s.starts_with('<') {
        let uri = s.trim_start_matches('<').trim_end_matches('>');
        if let Some(qcode) = uri.strip_prefix(WD_ENTITY) {
            if qcode.starts_with('Q') {
                return Object::Entity(qcode); // borrow, no allocation
            }
        }
        Object::Other
    } else if s.starts_with('"') {
        if let Some((value, suffix)) = extract_quoted_string(s) {
            if suffix.is_empty() {
                Object::Literal(value)
            } else if let Some(lang) = suffix.strip_prefix('@') {
                Object::LangLiteral(value, lang) // lang borrows from input
            } else if let Some(type_part) = suffix.strip_prefix("^^<") {
                let type_uri = type_part.trim_end_matches('>');
                Object::TypedLiteral(value, type_uri) // type_uri borrows
            } else {
                Object::Literal(value)
            }
        } else {
            Object::Other
        }
    } else {
        Object::Other
    }
}

/// Extract the string content from a quoted N-Triples literal,
/// returning (unescaped_value, suffix_after_closing_quote).
pub(super) fn extract_quoted_string(s: &str) -> Option<(String, &str)> {
    let s = s.strip_prefix('"')?;
    let mut value = String::new();
    let mut chars = s.char_indices();
    let mut end_idx = 0;

    while let Some((idx, ch)) = chars.next() {
        if ch == '\\' {
            // Escape sequence
            if let Some((_, next_ch)) = chars.next() {
                match next_ch {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '"' => value.push('"'),
                    '\\' => value.push('\\'),
                    'u' => {
                        // \uXXXX
                        let hex: String = chars.by_ref().take(4).map(|(_, c)| c).collect();
                        if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(cp) {
                                value.push(c);
                            }
                        }
                    }
                    'U' => {
                        // \UXXXXXXXX
                        let hex: String = chars.by_ref().take(8).map(|(_, c)| c).collect();
                        if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(cp) {
                                value.push(c);
                            }
                        }
                    }
                    _ => {
                        value.push('\\');
                        value.push(next_ch);
                    }
                }
            }
        } else if ch == '"' {
            end_idx = idx + 1; // position after the closing quote in `s`
            break;
        } else {
            value.push(ch);
        }
    }

    Some((value, &s[end_idx..]))
}

// ─── Typed literal conversion ───────────────────────────────────────────────

pub(super) const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
pub(super) const XSD_DECIMAL: &str = "http://www.w3.org/2001/XMLSchema#decimal";
pub(super) const XSD_DOUBLE: &str = "http://www.w3.org/2001/XMLSchema#double";
pub(super) const XSD_FLOAT: &str = "http://www.w3.org/2001/XMLSchema#float";
pub(super) const XSD_DATE: &str = "http://www.w3.org/2001/XMLSchema#dateTime";
pub(super) const XSD_BOOLEAN: &str = "http://www.w3.org/2001/XMLSchema#boolean";

pub(super) fn typed_literal_to_value(text: &str, type_uri: &str) -> Value {
    match type_uri {
        XSD_INTEGER => text
            .parse::<i64>()
            .map(Value::Int64)
            .unwrap_or(Value::String(text.to_string())),
        XSD_DECIMAL => {
            // Wikidata decimals often have leading "+"
            let cleaned = text.trim_start_matches('+');
            if let Ok(i) = cleaned.parse::<i64>() {
                Value::Int64(i)
            } else if let Ok(f) = cleaned.parse::<f64>() {
                Value::Float64(f)
            } else {
                Value::String(text.to_string())
            }
        }
        XSD_DOUBLE | XSD_FLOAT => text
            .parse::<f64>()
            .map(Value::Float64)
            .unwrap_or(Value::String(text.to_string())),
        XSD_BOOLEAN => match text {
            "true" | "1" => Value::Boolean(true),
            "false" | "0" => Value::Boolean(false),
            _ => Value::String(text.to_string()),
        },
        XSD_DATE => {
            // Try to parse ISO date, keep as string if it fails
            Value::String(text.to_string())
        }
        _ => {
            // GeoSPARQL WKT literals, etc. — keep as string
            Value::String(text.to_string())
        }
    }
}

// ─── Main loader ────────────────────────────────────────────────────────────

pub(super) fn language_matches(lang: &str, filter: &Option<HashSet<String>>) -> bool {
    match filter {
        None => true,
        Some(langs) => langs.contains(lang),
    }
}

/// Extract text from a literal object, respecting language filter.
pub(super) fn extract_lang_text(
    object: &Object<'_>,
    languages: &Option<HashSet<String>>,
) -> Option<String> {
    match object {
        Object::LangLiteral(text, lang) => {
            if language_matches(lang, languages) {
                Some(text.clone())
            } else {
                None
            }
        }
        Object::Literal(text) => Some(text.clone()),
        _ => None,
    }
}
