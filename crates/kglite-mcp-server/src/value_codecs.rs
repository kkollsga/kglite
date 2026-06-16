//! `extensions.value_codecs` — parse the manifest block into the core
//! [`ValueCodec`] type that the engine applies (see `kglite::api::cypher`).
//!
//! Replaces the retired `extensions.cypher_preprocessor` (0.10.26). Where the
//! preprocessor rewrote raw query *text* (blindly, pre-parse), value codecs are
//! bound to a property and applied *after* parsing — see the core module's
//! safety invariants. This module is just the YAML → `Vec<ValueCodec>` builder;
//! all the actual decode/encode logic lives in the engine.
//!
//! No trust gate: a Tier-1 codec (prefix / map / regex) is pure declarative
//! data transformation — no subprocess, no code execution, no host access
//! (unlike the preprocessor's `command:` hook, which is why *that* needed
//! `trust.allow_query_preprocessor`). The presence of an operator-authored
//! `value_codecs:` block is the explicit opt-in, same as `tools:`.
//!
//! ```yaml
//! extensions:
//!   value_codecs:
//!     - property: id            # the stored column the codec governs
//!       kind: prefix
//!       prefix: "Q"             # 'Q42' <-> 42
//!       stored_type: int        # int (default) | float | str
//!     - property: status
//!       kind: map
//!       map: { active: 1, archived: 2 }   # bijective: encode reverses it
//!     - property: event_date
//!       kind: regex
//!       match: '^(\d{2})\.(\d{2})\.(\d{4})$'   # full-match on the literal
//!       decode: '$3-$2-$1'                      # 31.12.2020 -> 2020-12-31
//!       encode: { match: '^(\d{4})-(\d{2})-(\d{2})$', replace: '$3.$2.$1' }  # optional
//! ```

use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use kglite::api::cypher::{CodecKind, StoredType, ValueCodec};
use kglite::api::param::json_value_to_kglite_value;
use kglite::api::Value;
use regex::Regex;
use serde_json::Value as Json;

/// Parse `extensions.value_codecs` into the engine's codec list. `Ok(vec![])`
/// when absent. Errors (surfaced at boot, not per-query) on a malformed block,
/// an invalid regex, or a non-bijective `map`.
pub fn from_manifest(ext: Option<&Json>) -> Result<Vec<ValueCodec>> {
    let Some(value) = ext else {
        return Ok(Vec::new());
    };
    let arr = value
        .as_array()
        .ok_or_else(|| anyhow!("extensions.value_codecs must be a list of codec mappings"))?;
    arr.iter()
        .enumerate()
        .map(|(i, item)| parse_one(item, i))
        .collect()
}

fn parse_one(item: &Json, i: usize) -> Result<ValueCodec> {
    let obj = item
        .as_object()
        .ok_or_else(|| anyhow!("value_codecs[{i}] must be a mapping"))?;
    let property = obj
        .get("property")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("value_codecs[{i}].property (string) is required"))?
        .to_string();
    let kind = obj
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("value_codecs[{i}].kind is required (prefix | map | regex)"))?;

    let kind = match kind {
        "prefix" => parse_prefix(obj, i)?,
        "map" => parse_map(obj, i)?,
        "regex" => parse_regex(obj, i)?,
        other => {
            return Err(anyhow!(
                "value_codecs[{i}].kind = {other:?} is not supported (prefix | map | regex)"
            ))
        }
    };
    Ok(ValueCodec { property, kind })
}

fn parse_prefix(obj: &serde_json::Map<String, Json>, i: usize) -> Result<CodecKind> {
    let prefix = obj
        .get("prefix")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("value_codecs[{i}] (prefix) requires `prefix` (string)"))?
        .to_string();
    let stored_type = match obj.get("stored_type").and_then(|v| v.as_str()) {
        None | Some("int") => StoredType::Int, // int is the common prefixed-id case
        Some("float") => StoredType::Float,
        Some("str") => StoredType::Str,
        Some(other) => {
            return Err(anyhow!(
                "value_codecs[{i}].stored_type = {other:?} invalid (int | float | str)"
            ))
        }
    };
    Ok(CodecKind::Prefix {
        prefix,
        stored_type,
    })
}

fn parse_map(obj: &serde_json::Map<String, Json>, i: usize) -> Result<CodecKind> {
    let map = obj.get("map").and_then(|v| v.as_object()).ok_or_else(|| {
        anyhow!("value_codecs[{i}] (map) requires `map` (mapping of string → value)")
    })?;
    let mut decode = HashMap::new();
    let mut encode: HashMap<Value, String> = HashMap::new();
    for (key, raw) in map {
        let val = json_value_to_kglite_value(raw);
        decode.insert(key.clone(), val.clone());
        // Bijective check: two keys mapping to the same stored value make the
        // reverse (encode) ambiguous — reject at boot rather than guess.
        if encode.insert(val, key.clone()).is_some() {
            return Err(anyhow!(
                "value_codecs[{i}].map is not bijective — two keys map to the same value, \
                 so the result-side encode would be ambiguous"
            ));
        }
    }
    Ok(CodecKind::Map { decode, encode })
}

fn parse_regex(obj: &serde_json::Map<String, Json>, i: usize) -> Result<CodecKind> {
    let pat = obj
        .get("match")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("value_codecs[{i}] (regex) requires `match` (regex string)"))?;
    let matcher = Regex::new(pat)
        .with_context(|| format!("value_codecs[{i}].match {pat:?} is not a valid regex"))?;
    let decode_template = obj
        .get("decode")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            anyhow!("value_codecs[{i}] (regex) requires `decode` (replacement template)")
        })?
        .to_string();
    // Optional reverse: encode: { match, replace }.
    let encode = match obj.get("encode") {
        None | Some(Json::Null) => None,
        Some(enc) => {
            let eobj = enc.as_object().ok_or_else(|| {
                anyhow!("value_codecs[{i}].encode must be a mapping {{match, replace}}")
            })?;
            let em = eobj
                .get("match")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("value_codecs[{i}].encode.match (regex) is required"))?;
            let er = eobj
                .get("replace")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    anyhow!("value_codecs[{i}].encode.replace (template) is required")
                })?;
            let ematcher = Regex::new(em)
                .with_context(|| format!("value_codecs[{i}].encode.match {em:?} is not valid"))?;
            Some((ematcher, er.to_string()))
        }
    };
    Ok(CodecKind::Regex {
        matcher,
        decode_template,
        encode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: serde_json::Value) -> Result<Vec<ValueCodec>> {
        from_manifest(Some(&json))
    }

    #[test]
    fn absent_is_empty() {
        assert!(from_manifest(None).unwrap().is_empty());
    }

    #[test]
    fn parses_prefix_codec() {
        let codecs =
            parse(serde_json::json!([{"property":"id","kind":"prefix","prefix":"Q"}])).unwrap();
        assert_eq!(codecs.len(), 1);
        assert_eq!(codecs[0].property, "id");
        // round-trips through the engine's decode/encode
        assert_eq!(
            codecs[0].decode_value(&Value::String("Q42".into())),
            Some(Value::Int64(42))
        );
        assert_eq!(
            codecs[0].encode_value(&Value::Int64(42)),
            Some(Value::String("Q42".into()))
        );
    }

    #[test]
    fn map_rejects_non_bijective() {
        let err = parse(serde_json::json!([
            {"property":"s","kind":"map","map":{"a":1,"b":1}}
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("not bijective"), "{err}");
    }

    #[test]
    fn regex_invalid_pattern_errors() {
        let err = parse(serde_json::json!([
            {"property":"d","kind":"regex","match":"(","decode":"$1"}
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("not a valid regex"), "{err}");
    }

    #[test]
    fn unknown_kind_errors() {
        let err = parse(serde_json::json!([{"property":"x","kind":"frobnicate"}])).unwrap_err();
        assert!(err.to_string().contains("not supported"), "{err}");
    }
}
