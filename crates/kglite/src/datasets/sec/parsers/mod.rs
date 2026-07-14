//! Parsers for SEC EDGAR file formats.
//!
//! Each submodule is a pure byte-stream → typed-record transformer.
//! No I/O, no async, no network. This means every parser is trivially
//! testable against a fixture file and reusable from any orchestrator.

use quick_xml::events::{BytesRef, BytesText};

use crate::datasets::sec::error::{Result, SecError};

/// Decode and unescape one XML text event into an existing parser buffer.
///
/// quick-xml 0.41 split the former `BytesText::unescape` operation into
/// decoding and entity-unescaping. Keeping the pair here prevents SEC parsers
/// from drifting into decode-only behavior that would leak `&amp;` and numeric
/// entities into graph properties.
fn append_unescaped_text(target: &mut String, text: &BytesText<'_>, context: &str) -> Result<()> {
    let decoded = text
        .decode()
        .map_err(|error| SecError::Decode(format!("{context}: {error}")))?;
    let unescaped = quick_xml::escape::unescape(&decoded)
        .map_err(|error| SecError::Decode(format!("{context}: {error}")))?;
    target.push_str(&unescaped);
    Ok(())
}

/// Resolve a quick-xml 0.41 reference event with the same predefined/numeric
/// entity semantics the old `BytesText::unescape` path provided.
fn append_xml_reference(
    target: &mut String,
    reference: &BytesRef<'_>,
    context: &str,
) -> Result<()> {
    if let Some(character) = reference
        .resolve_char_ref()
        .map_err(|error| SecError::Decode(format!("{context}: {error}")))?
    {
        target.push(character);
        return Ok(());
    }
    let name = reference
        .decode()
        .map_err(|error| SecError::Decode(format!("{context}: {error}")))?;
    let value = quick_xml::escape::resolve_predefined_entity(&name)
        .ok_or_else(|| SecError::Decode(format!("{context}: unrecognized entity `{name}`")))?;
    target.push_str(value);
    Ok(())
}

pub mod earnings_release;
pub mod eightk;
pub mod exhibit21;
pub mod f13f;
pub mod form144;
pub mod form4;
pub mod formd;
pub mod html_text;
pub mod offering;
pub mod officer_change;
pub mod ownership_table;
pub mod proxy_governance;
pub mod related_party;
pub mod sc13d;
pub mod submissions;
pub mod summary_compensation;
pub mod xbrl_facts;
