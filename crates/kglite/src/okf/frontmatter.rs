//! YAML frontmatter: delimit, parse, and flatten into kglite [`Value`]s.
//!
//! Two layers, deliberately separated by complexity:
//!  - [`split`] — the trivial `---` … `---` delimiter (own implementation,
//!    mirrors Google's `OKFDocument.parse`).
//!  - [`parse`] — the genuinely hard part (the YAML body), delegated to the
//!    `yaml-rust2` parser. We never hand-roll YAML; the failure modes (folded
//!    scalars, quoting, the Norway problem, nested maps) silently corrupt data,
//!    which for a fidelity-focused read tool is the worst outcome.

use crate::datatypes::value_depth::{check_value, MAX_VALUE_DEPTH};
use crate::datatypes::values::Value;
use std::collections::BTreeMap;

const DELIM: &str = "---";

/// Split a document into its (optional) YAML frontmatter block and its body.
///
/// Returns `(Some(yaml), body)` when the file opens with a `---` line and has a
/// closing `---`; otherwise `(None, whole_text)` — a file with no frontmatter is
/// valid (it degrades to a body-only concept), mirroring the permissive OKF
/// consumption model.
pub fn split(text: &str) -> (Option<String>, String) {
    let mut lines = text.lines();
    match lines.next() {
        Some(first) if first.trim() == DELIM => {}
        _ => return (None, text.to_string()),
    }
    let mut yaml: Vec<&str> = Vec::new();
    let mut found_close = false;
    for line in lines.by_ref() {
        if line.trim() == DELIM {
            found_close = true;
            break;
        }
        yaml.push(line);
    }
    if !found_close {
        // Unterminated frontmatter: treat the whole file as body (soft-fail
        // rather than reject — the file may simply be malformed prose).
        return (None, text.to_string());
    }
    let body: Vec<&str> = lines.collect();
    (Some(yaml.join("\n")), body.join("\n"))
}

/// Parse a document's frontmatter into a flattened `key → Value` map.
///
/// - Scalars map directly (`String`/`Int64`/`Float64`/`Boolean`); ISO
///   timestamps stay `String` (lexicographically sortable, lossless).
/// - Sequences (`tags: [...]`) become [`Value::List`].
/// - Nested maps (`metadata: { type: x }`) flatten to dotted keys
///   (`metadata.type`) so they stay queryable in every storage backend.
///
/// A file with no frontmatter yields an empty map. A frontmatter block that is
/// not a YAML mapping is an error.
pub fn parse(text: &str) -> Result<BTreeMap<String, Value>, String> {
    let (yaml, _body) = split(text);
    let Some(yaml) = yaml else {
        return Ok(BTreeMap::new());
    };
    if yaml.trim().is_empty() {
        return Ok(BTreeMap::new());
    }
    check_nesting(&yaml).map_err(|e| format!("invalid YAML in frontmatter: {e}"))?;
    let docs = yaml_rust2::YamlLoader::load_from_str(&yaml)
        .map_err(|e| format!("invalid YAML in frontmatter: {e}"))?;
    let doc = match docs.into_iter().next() {
        Some(d) => d,
        None => return Ok(BTreeMap::new()),
    };
    let map = match doc {
        yaml_rust2::Yaml::Null | yaml_rust2::Yaml::BadValue => return Ok(BTreeMap::new()),
        yaml_rust2::Yaml::Hash(h) => h,
        _ => return Err("frontmatter must be a YAML mapping".to_string()),
    };
    let mut out = BTreeMap::new();
    flatten_into("", &map, &mut out)?;
    Ok(out)
}

/// Parse a whole YAML document into a **nested** [`Value`] — the counterpart
/// to [`parse`], which flattens top-level mappings into dotted keys.
///
/// `.kglite/vault.yaml` and a recipe's `parameters:` schema are both nested
/// documents whose shape *is* their meaning: flattening `types: {Article:
/// {toc_depth: int}}` into the key `types.Article.toc_depth` throws away the
/// boundary between a label and a property, and a JSON Schema's
/// `properties.id.type` is not a property named `properties.id.type`. An
/// empty or null document yields [`Value::Null`].
pub fn parse_yaml(yaml: &str) -> Result<Value, String> {
    if yaml.trim().is_empty() {
        return Ok(Value::Null);
    }
    check_nesting(yaml).map_err(|e| format!("invalid YAML: {e}"))?;
    let docs =
        yaml_rust2::YamlLoader::load_from_str(yaml).map_err(|e| format!("invalid YAML: {e}"))?;
    match docs.into_iter().next() {
        Some(doc) => yaml_to_value(&doc).map_err(|e| format!("invalid YAML: {e}")),
        None => Ok(Value::Null),
    }
}

/// Retype a frontmatter string that spells an ISO date or an RFC 3339
/// timestamp (VAULT.md §4.2). Everything else — including a list's elements —
/// passes through untouched.
///
/// Deliberately stricter than [`crate::graph::blueprint::typing::scalar`]'s
/// `parse_date`, which is the *declared*-type grammar: that one also accepts
/// epoch milliseconds, so `version: "1609459200000"` would silently become a
/// date. Inference only fires on the two unambiguous spellings, and the shape
/// check rejects chrono's lenient `2026-1-5`, which is not ISO `YYYY-MM-DD`.
pub fn infer_temporal(v: Value) -> Value {
    let Value::String(s) = &v else { return v };
    let t = s.trim();
    if is_iso_date(t) {
        if let Ok(d) = chrono::NaiveDate::parse_from_str(t, "%Y-%m-%d") {
            return Value::DateTime(d);
        }
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(t) {
        // Offsets normalise to UTC, matching the RDF literal loader — a
        // property's value must not depend on the writer's timezone.
        return Value::Timestamp(dt.naive_utc());
    }
    v
}

/// `YYYY-MM-DD` exactly: ten characters, dashes in positions 4 and 7, digits
/// everywhere else.
fn is_iso_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && b.iter()
            .enumerate()
            .all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit())
}

/// Flatten a YAML mapping into dotted keys. Nested mappings recurse with a
/// `prefix.` ; everything else converts via [`yaml_to_value`].
fn flatten_into(
    prefix: &str,
    map: &yaml_rust2::yaml::Hash,
    out: &mut BTreeMap<String, Value>,
) -> Result<(), String> {
    for (k, v) in map {
        let key = match k.as_str() {
            Some(s) => s.to_string(),
            None => continue, // non-string keys are not representable as properties
        };
        let full = if prefix.is_empty() {
            key
        } else {
            format!("{prefix}.{key}")
        };
        match v {
            yaml_rust2::Yaml::Hash(inner) => flatten_into(&full, inner, out)?,
            other => {
                out.insert(full, yaml_to_value(other)?);
            }
        }
    }
    Ok(())
}

/// Refuse YAML nested clearly deeper than [`MAX_VALUE_DEPTH`] before it is
/// parsed. `yaml-rust2` builds and drops its document tree recursively, so a
/// few thousand `- - - …` levels overflow the stack inside the parser; the
/// depth has to be bounded on the text.
///
/// The estimate is block levels (a stack of indentation columns, each leading
/// `- ` marker counting as a level) plus the flow brackets open on a line.
/// It undercounts rather than overcounts (a `- ` aligned with its parent key
/// adds no column), so only input that is clearly too deep is refused;
/// [`yaml_to_value`] bounds whatever the estimate lets through.
fn check_nesting(yaml: &str) -> Result<(), String> {
    let limit = MAX_VALUE_DEPTH + 1;
    let mut columns: Vec<usize> = Vec::new();
    for line in yaml.lines() {
        let trimmed = line.trim_start_matches(' ');
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut col = line.len() - trimmed.len();
        while columns.last().is_some_and(|&c| c >= col) {
            columns.pop();
        }
        let mut rest = trimmed;
        while rest == "-" || rest.starts_with("- ") {
            columns.push(col);
            let after = &rest[1..];
            let body = after.trim_start_matches(' ');
            col += 1 + (after.len() - body.len());
            rest = body;
        }
        columns.push(col);
        if columns.len() + flow_depth(rest) > limit {
            return Err(nesting_error());
        }
    }
    Ok(())
}

/// Deepest flow bracket nesting on one line, ignoring quoted text.
fn flow_depth(line: &str) -> usize {
    let (mut depth, mut deepest) = (0usize, 0usize);
    let mut quote: Option<char> = None;
    for c in line.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'') => quote = Some(c),
            (None, '[' | '{') => {
                depth += 1;
                deepest = deepest.max(depth);
            }
            (None, ']' | '}') => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    deepest
}

fn nesting_error() -> String {
    format!("YAML nested deeper than {MAX_VALUE_DEPTH} levels; flatten the structure")
}

/// Convert a single YAML value into a kglite [`Value`]. Mappings nested *inside*
/// a sequence are preserved as [`Value::Map`] (only top-level maps flatten).
fn yaml_to_value(v: &yaml_rust2::Yaml) -> Result<Value, String> {
    let value = convert_yaml(v)?;
    check_value(&value).map_err(|_| nesting_error())?;
    Ok(value)
}

fn convert_yaml(v: &yaml_rust2::Yaml) -> Result<Value, String> {
    use yaml_rust2::Yaml;
    Ok(match v {
        Yaml::Null | Yaml::BadValue => Value::Null,
        Yaml::Boolean(b) => Value::Boolean(*b),
        Yaml::Integer(i) => Value::Int64(*i),
        Yaml::Real(s) => s
            .parse::<f64>()
            .map(Value::Float64)
            .unwrap_or_else(|_| Value::String(s.clone())),
        Yaml::String(s) => Value::String(s.clone()),
        Yaml::Array(seq) => Value::List(seq.iter().map(convert_yaml).collect::<Result<_, _>>()?),
        Yaml::Hash(map) => {
            let mut pairs: Vec<(crate::datatypes::PropKey, Value)> = Vec::with_capacity(map.len());
            for (k, val) in map {
                if let Some(key) = k.as_str() {
                    pairs.push((crate::datatypes::PropKey::from(key), convert_yaml(val)?));
                }
            }
            Value::Map(crate::datatypes::PropMap::from_pairs(pairs))
        }
        Yaml::Alias(_) => Value::Null,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_frontmatter_is_empty_map() {
        let (fm, body) = split("# Just a note\n\nbody text");
        assert!(fm.is_none());
        assert!(body.starts_with("# Just a note"));
        assert!(parse("# Just a note").unwrap().is_empty());
    }

    #[test]
    fn splits_frontmatter_and_body() {
        let doc = "---\ntype: Table\n---\n\n# Body\ncontent";
        let (fm, body) = split(doc);
        assert_eq!(fm.as_deref(), Some("type: Table"));
        assert!(body.contains("# Body"));
    }

    #[test]
    fn unterminated_frontmatter_degrades_to_body() {
        let (fm, _) = split("---\ntype: Table\nno closing delim");
        assert!(fm.is_none());
    }

    #[test]
    fn scalars_and_lists() {
        let doc = "---\ntype: BigQuery Table\ntitle: Orders\ntags:\n- sales\n- orders\nrank: 3\nratio: 0.5\nactive: true\n---\nbody";
        let m = parse(doc).unwrap();
        assert_eq!(m.get("type"), Some(&Value::String("BigQuery Table".into())));
        assert_eq!(m.get("rank"), Some(&Value::Int64(3)));
        assert_eq!(m.get("ratio"), Some(&Value::Float64(0.5)));
        assert_eq!(m.get("active"), Some(&Value::Boolean(true)));
        assert_eq!(
            m.get("tags"),
            Some(&Value::List(vec![
                Value::String("sales".into()),
                Value::String("orders".into()),
            ]))
        );
    }

    #[test]
    fn nested_map_flattens_to_dotted_keys() {
        let doc = "---\nname: foo\nmetadata:\n  type: feedback\n  scope: project\n---\nbody";
        let m = parse(doc).unwrap();
        assert_eq!(m.get("name"), Some(&Value::String("foo".into())));
        assert_eq!(
            m.get("metadata.type"),
            Some(&Value::String("feedback".into()))
        );
        assert_eq!(
            m.get("metadata.scope"),
            Some(&Value::String("project".into()))
        );
        assert!(!m.contains_key("metadata"));
    }

    #[test]
    fn iso_timestamp_stays_string() {
        // Quoted and unquoted ISO timestamps must both remain strings — they
        // sort lexicographically, which is what staleness queries rely on.
        let doc = "---\nts1: '2026-05-28T23:31:54+00:00'\nts2: 2026-05-28\n---\nbody";
        let m = parse(doc).unwrap();
        assert_eq!(
            m.get("ts1"),
            Some(&Value::String("2026-05-28T23:31:54+00:00".into()))
        );
        assert!(matches!(m.get("ts2"), Some(Value::String(_))));
    }

    #[test]
    fn norway_problem_tags_stay_strings() {
        // serde_yml uses the YAML 1.2 core schema: only true/false are booleans.
        // `no`/`yes`/`on`/`off` must remain strings so a tag literally named
        // "no" survives.
        let doc = "---\ntags:\n- no\n- yes\n- on\n---\nbody";
        let m = parse(doc).unwrap();
        assert_eq!(
            m.get("tags"),
            Some(&Value::List(vec![
                Value::String("no".into()),
                Value::String("yes".into()),
                Value::String("on".into()),
            ]))
        );
    }

    #[test]
    fn folded_multiline_description() {
        let doc = "---\ndescription: This table contains\n  information about orders\n  across all channels.\n---\nbody";
        let m = parse(doc).unwrap();
        assert_eq!(
            m.get("description"),
            Some(&Value::String(
                "This table contains information about orders across all channels.".into()
            ))
        );
    }

    fn block_sequence_note(levels: usize) -> String {
        format!("---\na:\n  {}x\n---\nbody", "- ".repeat(levels))
    }

    #[test]
    fn a_block_sequence_nested_far_past_the_ceiling_is_refused_not_parsed() {
        // 20k levels overflowed the stack inside the YAML parser itself.
        let err = parse(&block_sequence_note(20_000)).unwrap_err();
        assert!(err.contains("deeper than 256 levels"), "{err}");
        let err = parse_yaml(&format!("a:\n  {}x", "- ".repeat(20_000))).unwrap_err();
        assert!(err.contains("deeper than 256 levels"), "{err}");
    }

    #[test]
    fn a_nested_map_and_a_flow_sequence_past_the_ceiling_are_refused() {
        let mut map = String::new();
        for i in 0..400 {
            map.push_str(&format!("{}k{i}:\n", " ".repeat(i)));
        }
        map.push_str(&format!("{}v: 1", " ".repeat(400)));
        assert!(parse(&format!("---\n{map}\n---")).is_err());
        let flow = format!("---\na: {}x{}\n---", "[".repeat(300), "]".repeat(300));
        assert!(parse(&flow).is_err());
    }

    #[test]
    fn nesting_at_the_ceiling_still_parses() {
        let fm = parse(&block_sequence_note(255)).unwrap();
        assert!(matches!(fm.get("a"), Some(Value::List(_))));
        let flow = format!("---\na: {}x{}\n---", "[".repeat(200), "]".repeat(200));
        assert!(parse(&flow).unwrap().contains_key("a"));
    }

    #[test]
    fn brackets_inside_quotes_do_not_count_as_nesting() {
        let text = format!("---\na: \"{}\"\n---", "[".repeat(1000));
        assert!(parse(&text).unwrap().contains_key("a"));
    }
}
