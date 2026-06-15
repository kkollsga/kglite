//! YAML frontmatter: delimit, parse, and flatten into kglite [`Value`]s.
//!
//! Two layers, deliberately separated by complexity:
//!  - [`split`] — the trivial `---` … `---` delimiter (own implementation,
//!    mirrors Google's `OKFDocument.parse`).
//!  - [`parse`] — the genuinely hard part (the YAML body), delegated to the
//!    `yaml-rust2` parser. We never hand-roll YAML; the failure modes (folded
//!    scalars, quoting, the Norway problem, nested maps) silently corrupt data,
//!    which for a fidelity-focused read tool is the worst outcome.

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
    flatten_into("", &map, &mut out);
    Ok(out)
}

/// Flatten a YAML mapping into dotted keys. Nested mappings recurse with a
/// `prefix.` ; everything else converts via [`yaml_to_value`].
fn flatten_into(prefix: &str, map: &yaml_rust2::yaml::Hash, out: &mut BTreeMap<String, Value>) {
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
            yaml_rust2::Yaml::Hash(inner) => flatten_into(&full, inner, out),
            other => {
                out.insert(full, yaml_to_value(other));
            }
        }
    }
}

/// Convert a single YAML value into a kglite [`Value`]. Mappings nested *inside*
/// a sequence are preserved as [`Value::Map`] (only top-level maps flatten).
fn yaml_to_value(v: &yaml_rust2::Yaml) -> Value {
    use yaml_rust2::Yaml;
    match v {
        Yaml::Null | Yaml::BadValue => Value::Null,
        Yaml::Boolean(b) => Value::Boolean(*b),
        Yaml::Integer(i) => Value::Int64(*i),
        Yaml::Real(s) => s
            .parse::<f64>()
            .map(Value::Float64)
            .unwrap_or_else(|_| Value::String(s.clone())),
        Yaml::String(s) => Value::String(s.clone()),
        Yaml::Array(seq) => Value::List(seq.iter().map(yaml_to_value).collect()),
        Yaml::Hash(map) => {
            let mut bt = BTreeMap::new();
            for (k, val) in map {
                if let Some(key) = k.as_str() {
                    bt.insert(key.to_string(), yaml_to_value(val));
                }
            }
            Value::Map(bt)
        }
        Yaml::Alias(_) => Value::Null,
    }
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
}
