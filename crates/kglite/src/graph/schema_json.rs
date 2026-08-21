//! The **one** external dialect for a declarative schema.
//!
//! [`SchemaDefinition`] derives `Serialize`/`Deserialize`, so it already had a
//! serde shape — `{"node_schemas": {"T": {"required_fields": [...], ...}}}`.
//! The dialect users actually write, through Python's `define_schema`, is a
//! different one: `{"nodes": {"T": {"required": [...], "types": {...},
//! "primary_key": "id"}}}`. Shipping the serde shape out through the C ABI
//! would have made those two dialects both *published*, and a published C
//! signature never changes within an ABI major — the fork would be permanent.
//!
//! So this module is the chokepoint: [`schema_from_value`] parses the
//! **Python dialect** out of the shared [`Value`] data model, and every binding
//! goes through it. Python converts its dict with `py_value_to_value` and calls
//! it; the C ABI parses JSON with [`schema_from_json`] and calls it. One
//! grammar, one set of messages, one place to change.
//!
//! ## The grammar
//!
//! ```json
//! {
//!   "nodes": {
//!     "Person": {
//!       "required":       ["id", "name"],
//!       "optional":       ["email"],
//!       "types":          {"id": "integer", "name": "string"},
//!       "primary_key":    "id",
//!       "unique":         [["email"]],
//!       "layer":          "managed",
//!       "auto_timestamp": true
//!     }
//!   },
//!   "connections": {
//!     "KNOWS": {
//!       "source":              "Person",
//!       "target":              "Person",
//!       "cardinality":         "many-to-many",
//!       "required_properties": ["since"],
//!       "property_types":      {"since": "integer"},
//!       "auto_timestamp":      false
//!     }
//!   }
//! }
//! ```
//!
//! `source` and `target` are the only required keys anywhere. An absent — or
//! non-map — `nodes` / `connections` is a no-op rather than an error, which is
//! the long-standing behaviour of the hand-written Python walk this was lifted
//! from and is what makes `{"connections": {...}}` alone legal.
//!
//! Every key that *is* present, at every level, must be one the grammar knows.
//! A schema is a promise about integrity, so a key the parser cannot place is
//! never a harmless extra: `{"uniqe": [["email"]]}` reads as a declared UNIQUE
//! constraint and enforces nothing, and `{"Task": {...}}` — the `"nodes"`
//! wrapper forgotten — parsed to a completely empty schema. Both were accepted
//! in silence until 0.16.6; both are refused now, with the near-miss named
//! where one is close enough to guess.

use super::schema::{ConnectionSchemaDefinition, NodeSchemaDefinition, SchemaDefinition};
use crate::datatypes::values::Value;

/// Which class of mistake a [`SchemaParseError`] describes.
///
/// Carried rather than folded into the message so a binding can raise its own
/// language's conventional exception: the Python wrapper maps these to
/// `TypeError` / `KeyError` / `ValueError`, exactly as its hand-written parser
/// did before delegating here. The C ABI collapses all three to
/// `KGLITE_ERR_INVALID_ARGUMENT` and passes the message through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaParseErrorKind {
    /// A value has the wrong shape (a list where a map was needed, …).
    Type,
    /// A required key is missing.
    Key,
    /// The value is the right shape but not an accepted one.
    Value,
}

/// A schema document the parser refused, with the class of refusal.
#[derive(Debug, Clone)]
pub struct SchemaParseError {
    pub kind: SchemaParseErrorKind,
    pub message: String,
}

impl std::fmt::Display for SchemaParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SchemaParseError {}

fn type_error(message: impl Into<String>) -> SchemaParseError {
    SchemaParseError {
        kind: SchemaParseErrorKind::Type,
        message: message.into(),
    }
}

fn key_error(message: impl Into<String>) -> SchemaParseError {
    SchemaParseError {
        kind: SchemaParseErrorKind::Key,
        message: message.into(),
    }
}

fn value_error(message: impl Into<String>) -> SchemaParseError {
    SchemaParseError {
        kind: SchemaParseErrorKind::Value,
        message: message.into(),
    }
}

type ParseResult<T> = Result<T, SchemaParseError>;

fn as_map(value: &Value) -> Option<&crate::datatypes::PropMap> {
    match value {
        Value::Map(map) => Some(map),
        _ => None,
    }
}

fn as_str(value: &Value) -> Option<&str> {
    match value {
        Value::String(s) => Some(s),
        _ => None,
    }
}

/// A list of strings. A *bare* string is deliberately not accepted — the
/// Python parser's `extract::<Vec<String>>()` rejects it too, and silently
/// reading `"id"` as `["i", "d"]` is the classic footgun.
fn as_string_list(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::List(items) => items
            .iter()
            .map(|item| as_str(item).map(str::to_string))
            .collect(),
        _ => None,
    }
}

/// A `{name: type-name}` mapping (`types` / `property_types`). Both key and
/// value must be strings.
fn parse_type_map(
    value: &Value,
    what: &str,
    into: &mut std::collections::HashMap<String, String>,
) -> ParseResult<()> {
    let map = as_map(value).ok_or_else(|| type_error(format!("{what} must be a dictionary")))?;
    for (field, type_val) in map {
        let type_name = as_str(type_val).ok_or_else(|| {
            type_error(format!(
                "{what} entry '{field}' must name a type as a string, got {}",
                type_val.type_name()
            ))
        })?;
        into.insert(field.to_string(), type_name.to_string());
    }
    Ok(())
}

/// Parse a whole schema document out of the shared [`Value`] data model.
///
/// This is the chokepoint every binding calls: Python after
/// `py_value_to_value`, the C ABI after [`schema_from_json`]. The document must
/// be a map; `nodes` and `connections` are both optional.
pub fn schema_from_value(value: &Value) -> ParseResult<SchemaDefinition> {
    let doc = as_map(value).ok_or_else(|| {
        type_error(format!(
            "schema must be a dictionary with 'nodes' and/or 'connections' keys, got {}",
            value.type_name()
        ))
    })?;
    reject_unknown_keys(doc, TOP_LEVEL_KEYS, &TopLevel)?;
    let mut schema = SchemaDefinition::new();
    parse_node_schemas(doc, &mut schema)?;
    parse_connection_schemas(doc, &mut schema)?;
    Ok(schema)
}

/// The only keys a schema document may carry at its top level.
const TOP_LEVEL_KEYS: &[&str] = &["nodes", "connections"];

/// The keys one node type's declaration may carry.
const NODE_DECLARATION_KEYS: &[&str] = &[
    "required",
    "optional",
    "types",
    "primary_key",
    "unique",
    "layer",
    "auto_timestamp",
];

/// The keys one connection type's declaration may carry.
const CONNECTION_DECLARATION_KEYS: &[&str] = &[
    "source",
    "target",
    "cardinality",
    "required_properties",
    "property_types",
    "auto_timestamp",
];

/// Where an unknown key was found, which decides how the refusal reads.
trait UnknownKeyContext {
    /// The message for a key with no close match among the accepted ones.
    fn message(&self, key: &str, value: &Value, accepted: &[&str]) -> String;
}

/// Unknown key at the document's top level.
///
/// The overwhelmingly common cause is a forgotten `"nodes"` wrapper — the key
/// is a *node type name* and its value is that type's declaration — so a map
/// value gets the wrapper hint rather than a bare "unknown key", which would
/// name the user's own type as the mistake.
struct TopLevel;

impl UnknownKeyContext for TopLevel {
    fn message(&self, key: &str, value: &Value, accepted: &[&str]) -> String {
        if as_map(value).is_some() {
            return format!(
                "Schema key '{key}' is not a top-level schema key. Node types go inside a \
                 'nodes' wrapper: define_schema({{'nodes': {{'{key}': {{...}}}}}}). Top-level \
                 keys are {}.",
                quoted_list(accepted)
            );
        }
        format!(
            "Unknown top-level schema key '{key}'. Expected {}.",
            quoted_list(accepted)
        )
    }
}

/// Unknown key inside one node or connection type's declaration.
struct Declaration<'a> {
    /// "node type" / "connection type".
    what: &'a str,
    /// The type whose declaration carries the key.
    owner: &'a str,
}

impl UnknownKeyContext for Declaration<'_> {
    fn message(&self, key: &str, _value: &Value, accepted: &[&str]) -> String {
        format!(
            "Unknown key '{key}' in the declaration for {} '{}'. Accepted keys are {}.",
            self.what,
            self.owner,
            quoted_list(accepted)
        )
    }
}

/// `'a', 'b' and 'c'`, for listing the accepted keys in a refusal.
fn quoted_list(keys: &[&str]) -> String {
    let quoted: Vec<String> = keys.iter().map(|k| format!("'{k}'")).collect();
    match quoted.split_last() {
        None => String::new(),
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
    }
}

/// Refuse the first key of `map` that `accepted` does not contain.
///
/// A near-miss is named — `'uniqe'` against `'unique'` is a declared constraint
/// the caller believes they have and does not — using the same
/// [`did_you_mean`](crate::graph::mutation::validation::did_you_mean) the rest
/// of the engine's typo guards use, so the suggestion bar ("genuinely close, or
/// silent") is one rule, not two.
fn reject_unknown_keys(
    map: &crate::datatypes::PropMap,
    accepted: &[&str],
    context: &dyn UnknownKeyContext,
) -> ParseResult<()> {
    for (key, value) in map {
        if accepted.contains(&key) {
            continue;
        }
        let suggestion = crate::graph::mutation::validation::did_you_mean(key, accepted);
        if !suggestion.is_empty() {
            return Err(key_error(format!(
                "Unknown schema key '{key}'.{suggestion}"
            )));
        }
        return Err(key_error(context.message(key, value, accepted)));
    }
    Ok(())
}

/// [`schema_from_value`] over a JSON document — the C ABI's entry point.
///
/// The JSON is routed through the same
/// [`json_value_to_kglite_value`](crate::param::json_value_to_kglite_value)
/// every other JSON-carrying symbol uses, so a JSON schema and a Python dict
/// schema become the identical [`Value`] before either is parsed.
pub fn schema_from_json(json: &str) -> ParseResult<SchemaDefinition> {
    let parsed: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| value_error(format!("schema JSON could not be parsed: {e}")))?;
    schema_from_value(&crate::param::json_value_to_kglite_value(&parsed))
}

/// Parse the `nodes` mapping into `schema`.
///
/// Absent or non-map `nodes` is a no-op rather than an error, matching the
/// long-standing behaviour of the Python walk this was lifted from.
fn parse_node_schemas(
    doc: &crate::datatypes::PropMap,
    schema: &mut SchemaDefinition,
) -> ParseResult<()> {
    let Some(nodes) = doc.get("nodes").and_then(as_map) else {
        return Ok(());
    };
    for (node_type, node_schema_val) in nodes {
        let node_schema_map = as_map(node_schema_val).ok_or_else(|| {
            type_error(format!(
                "Schema for node type '{node_type}' must be a dictionary"
            ))
        })?;

        reject_unknown_keys(
            node_schema_map,
            NODE_DECLARATION_KEYS,
            &Declaration {
                what: "node type",
                owner: node_type,
            },
        )?;

        let mut node_schema = NodeSchemaDefinition::default();
        if let Some(required) = node_schema_map.get("required") {
            node_schema.required_fields = as_string_list(required).ok_or_else(|| {
                type_error(format!(
                    "required for node type '{node_type}' must be a list of property names"
                ))
            })?;
        }
        if let Some(optional) = node_schema_map.get("optional") {
            node_schema.optional_fields = as_string_list(optional).ok_or_else(|| {
                type_error(format!(
                    "optional for node type '{node_type}' must be a list of property names"
                ))
            })?;
        }
        if let Some(types) = node_schema_map.get("types") {
            parse_type_map(types, "types", &mut node_schema.field_types)?;
        }

        parse_node_declarations(node_type, node_schema_map, &mut node_schema)?;
        schema.add_node_schema(node_type.to_string(), node_schema);
    }
    Ok(())
}

/// Parse the `connections` mapping into `schema`. The edge counterpart of
/// [`parse_node_schemas`]; `source`/`target` are the only required keys.
fn parse_connection_schemas(
    doc: &crate::datatypes::PropMap,
    schema: &mut SchemaDefinition,
) -> ParseResult<()> {
    let Some(connections) = doc.get("connections").and_then(as_map) else {
        return Ok(());
    };
    for (conn_type, conn_schema_val) in connections {
        let conn_map = as_map(conn_schema_val).ok_or_else(|| {
            type_error(format!(
                "Schema for connection type '{conn_type}' must be a dictionary"
            ))
        })?;

        reject_unknown_keys(
            conn_map,
            CONNECTION_DECLARATION_KEYS,
            &Declaration {
                what: "connection type",
                owner: conn_type,
            },
        )?;

        let mut conn_schema = ConnectionSchemaDefinition {
            source_type: required_endpoint(conn_map, conn_type, "source")?,
            target_type: required_endpoint(conn_map, conn_type, "target")?,
            cardinality: None,
            required_properties: Vec::new(),
            property_types: std::collections::HashMap::new(),
            auto_timestamp: None,
        };

        if let Some(cardinality) = conn_map.get("cardinality") {
            conn_schema.cardinality = Some(
                as_str(cardinality)
                    .ok_or_else(|| {
                        type_error(format!(
                            "cardinality for connection type '{conn_type}' must be a string"
                        ))
                    })?
                    .to_string(),
            );
        }
        // Opt-in freshness provenance for edges of this type.
        if let Some(ts_val) = conn_map.get("auto_timestamp") {
            conn_schema.auto_timestamp = Some(as_bool(ts_val, conn_type, "connection type")?);
        }
        if let Some(required_props) = conn_map.get("required_properties") {
            conn_schema.required_properties = as_string_list(required_props).ok_or_else(|| {
                type_error(format!(
                    "required_properties for connection type '{conn_type}' must be a list of \
                     property names"
                ))
            })?;
        }
        if let Some(prop_types) = conn_map.get("property_types") {
            parse_type_map(
                prop_types,
                "property_types",
                &mut conn_schema.property_types,
            )?;
        }

        schema.add_connection_schema(conn_type.to_string(), conn_schema);
    }
    Ok(())
}

/// One of a connection schema's two mandatory endpoint keys.
fn required_endpoint(
    conn_map: &crate::datatypes::PropMap,
    conn_type: &str,
    key: &str,
) -> ParseResult<String> {
    let value = conn_map.get(key).ok_or_else(|| {
        key_error(format!(
            "Connection '{conn_type}' missing required '{key}' field"
        ))
    })?;
    Ok(as_str(value)
        .ok_or_else(|| {
            type_error(format!(
                "Connection '{conn_type}' field '{key}' must name a node type as a string, got {}",
                value.type_name()
            ))
        })?
        .to_string())
}

/// A strict boolean — an integer `1` is not `true`, matching PyO3's `bool`
/// extraction, so a typo cannot silently enable timestamping.
fn as_bool(value: &Value, owner: &str, what: &str) -> ParseResult<bool> {
    match value {
        Value::Boolean(b) => Ok(*b),
        other => Err(type_error(format!(
            "auto_timestamp for {what} '{owner}' must be true or false, got {}",
            other.type_name()
        ))),
    }
}

/// Parse the opt-in per-node-type declarations accepted beyond the field list:
/// `primary_key`, `unique`, `layer`, and `auto_timestamp`.
fn parse_node_declarations(
    node_type: &str,
    node_schema_map: &crate::datatypes::PropMap,
    node_schema: &mut NodeSchemaDefinition,
) -> ParseResult<()> {
    // Any property may be the primary key: `id` is enforced through the O(1)
    // per-type id-index, anything else through a unique secondary index that
    // `set_schema` installs. Either way the key is unique *and* required
    // (NODE KEY semantics).
    if let Some(pk_val) = node_schema_map.get("primary_key") {
        let pk = as_str(pk_val).ok_or_else(|| {
            type_error(format!(
                "primary_key for node type '{node_type}' must be a property name as a string, \
                 got {}",
                pk_val.type_name()
            ))
        })?;
        if pk.is_empty() {
            return Err(value_error(format!(
                "primary_key for node type '{node_type}' must name a property."
            )));
        }
        node_schema.primary_key = Some(pk.to_string());
    }

    if let Some(unique_val) = node_schema_map.get("unique") {
        node_schema.unique = Some(parse_unique_declaration(node_type, unique_val)?);
    }

    // The ownership layer for the two-writer contract. An unknown value is
    // rejected as a typo-guard.
    if let Some(layer_val) = node_schema_map.get("layer") {
        let layer = as_str(layer_val).ok_or_else(|| {
            type_error(format!(
                "layer for node type '{node_type}' must be a string, got {}",
                layer_val.type_name()
            ))
        })?;
        if layer != "managed" && layer != "runtime" {
            return Err(value_error(format!(
                "layer for node type '{node_type}' must be 'managed' or 'runtime'. Got '{layer}'."
            )));
        }
        node_schema.layer = Some(layer.to_string());
    }

    // Freshness provenance: stamp `updated_at` (+ the caller's git_sha) on
    // every write to this type.
    if let Some(ts_val) = node_schema_map.get("auto_timestamp") {
        node_schema.auto_timestamp = Some(as_bool(ts_val, node_type, "node type")?);
    }

    Ok(())
}

/// Parse the optional `unique` declaration for one node type into a list of
/// property tuples, so `[["email"], ["first", "last"]]` declares one
/// single-property and one composite constraint.
///
/// A bare property name and a flat list of names are both natural mistakes that
/// would otherwise declare something surprising, so the two shorthands are
/// accepted explicitly rather than guessed at: a flat list becomes one
/// single-property constraint per entry, not one composite over all of them.
fn parse_unique_declaration(node_type: &str, unique_val: &Value) -> ParseResult<Vec<Vec<String>>> {
    let malformed = || {
        value_error(format!(
            "unique for node type '{node_type}' must be a property name, a list of property \
             names, or a list of property tuples — e.g. 'email', ['email'], or \
             [['first', 'last']]."
        ))
    };
    let tuples: Vec<Vec<String>> = if let Some(single) = as_str(unique_val) {
        vec![vec![single.to_string()]]
    } else if let Some(flat) = as_string_list(unique_val) {
        flat.into_iter().map(|property| vec![property]).collect()
    } else {
        match unique_val {
            Value::List(items) => items
                .iter()
                .map(|item| as_string_list(item).ok_or_else(malformed))
                .collect::<ParseResult<Vec<Vec<String>>>>()?,
            _ => return Err(malformed()),
        }
    };
    for tuple in &tuples {
        if tuple.is_empty() {
            return Err(value_error(format!(
                "unique for node type '{node_type}' contains an empty property tuple."
            )));
        }
    }
    Ok(tuples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> SchemaDefinition {
        schema_from_json(json).unwrap_or_else(|e| panic!("schema rejected: {e}"))
    }

    fn err(json: &str) -> SchemaParseError {
        schema_from_json(json).expect_err("schema should have been rejected")
    }

    #[test]
    fn the_python_dialect_round_trips_every_node_declaration() {
        let schema = parse(
            r#"{"nodes": {"Person": {
                 "required": ["id", "name"],
                 "optional": ["email"],
                 "types": {"id": "integer", "name": "string"},
                 "primary_key": "id",
                 "unique": [["email"], ["first", "last"]],
                 "layer": "managed",
                 "auto_timestamp": true
               }}}"#,
        );
        let person = &schema.node_schemas["Person"];
        assert_eq!(person.required_fields, ["id", "name"]);
        assert_eq!(person.optional_fields, ["email"]);
        assert_eq!(person.field_types["id"], "integer");
        assert_eq!(person.field_types["name"], "string");
        assert_eq!(person.primary_key.as_deref(), Some("id"));
        assert_eq!(
            person.unique.as_deref(),
            Some(
                [
                    vec!["email".to_string()],
                    vec!["first".to_string(), "last".to_string()]
                ]
                .as_slice()
            )
        );
        assert_eq!(person.layer.as_deref(), Some("managed"));
        assert_eq!(person.auto_timestamp, Some(true));
    }

    #[test]
    fn connections_round_trip_and_need_only_their_endpoints() {
        let schema = parse(
            r#"{"connections": {
                 "KNOWS": {"source": "Person", "target": "Person",
                           "cardinality": "many-to-many",
                           "required_properties": ["since"],
                           "property_types": {"since": "integer"},
                           "auto_timestamp": false},
                 "OWNS": {"source": "Person", "target": "Thing"}
               }}"#,
        );
        let knows = &schema.connection_schemas["KNOWS"];
        assert_eq!(knows.source_type, "Person");
        assert_eq!(knows.target_type, "Person");
        assert_eq!(knows.cardinality.as_deref(), Some("many-to-many"));
        assert_eq!(knows.required_properties, ["since"]);
        assert_eq!(knows.property_types["since"], "integer");
        assert_eq!(knows.auto_timestamp, Some(false));

        let owns = &schema.connection_schemas["OWNS"];
        assert_eq!(owns.target_type, "Thing");
        assert_eq!(owns.cardinality, None);
        assert!(owns.required_properties.is_empty());
        assert_eq!(owns.auto_timestamp, None);
    }

    #[test]
    fn the_three_unique_shorthands_all_parse() {
        // A bare name.
        assert_eq!(
            parse(r#"{"nodes": {"U": {"unique": "email"}}}"#).node_schemas["U"]
                .unique
                .clone()
                .unwrap(),
            [vec!["email".to_string()]]
        );
        // A flat list is one single-property constraint *per entry*, not one
        // composite over all of them.
        assert_eq!(
            parse(r#"{"nodes": {"U": {"unique": ["email", "handle"]}}}"#).node_schemas["U"]
                .unique
                .clone()
                .unwrap(),
            [vec!["email".to_string()], vec!["handle".to_string()]]
        );
        // Explicit tuples.
        assert_eq!(
            parse(r#"{"nodes": {"U": {"unique": [["first", "last"]]}}}"#).node_schemas["U"]
                .unique
                .clone()
                .unwrap(),
            [vec!["first".to_string(), "last".to_string()]]
        );
    }

    #[test]
    fn an_absent_or_non_map_section_is_a_no_op() {
        assert!(parse("{}").node_schemas.is_empty());
        // A non-map `nodes` is ignored rather than rejected — the behaviour the
        // Python walk has always had.
        let schema = parse(r#"{"nodes": ["Person"], "connections": 7}"#);
        assert!(schema.node_schemas.is_empty());
        assert!(schema.connection_schemas.is_empty());
        // Connections alone are legal.
        assert_eq!(
            parse(r#"{"connections": {"R": {"source": "A", "target": "B"}}}"#)
                .connection_schemas
                .len(),
            1
        );
    }

    #[test]
    fn every_refusal_carries_its_class_and_message() {
        use SchemaParseErrorKind::*;
        for (json, kind, needle) in [
            (r#"[]"#, Type, "must be a dictionary"),
            (
                r#"{"nodes": {"P": ["required"]}}"#,
                Type,
                "Schema for node type 'P' must be a dictionary",
            ),
            (
                r#"{"nodes": {"P": {"required": "id"}}}"#,
                Type,
                "must be a list of property names",
            ),
            (
                r#"{"nodes": {"P": {"types": ["id"]}}}"#,
                Type,
                "types must be a dictionary",
            ),
            (
                r#"{"nodes": {"P": {"primary_key": ""}}}"#,
                Value,
                "must name a property.",
            ),
            (
                r#"{"nodes": {"P": {"layer": "other"}}}"#,
                Value,
                "must be 'managed' or 'runtime'",
            ),
            (
                r#"{"nodes": {"P": {"unique": 7}}}"#,
                Value,
                "must be a property name, a list of property names",
            ),
            (
                r#"{"nodes": {"P": {"unique": [[]]}}}"#,
                Value,
                "contains an empty property tuple.",
            ),
            (
                r#"{"nodes": {"P": {"auto_timestamp": 1}}}"#,
                Type,
                "must be true or false",
            ),
            (
                r#"{"connections": {"R": {"target": "B"}}}"#,
                Key,
                "missing required 'source' field",
            ),
            (
                r#"{"connections": {"R": {"source": "A"}}}"#,
                Key,
                "missing required 'target' field",
            ),
            (
                r#"{"connections": {"R": {"source": 1, "target": "B"}}}"#,
                Type,
                "must name a node type as a string",
            ),
            (r#"{"nodes":"#, Value, "could not be parsed"),
        ] {
            let e = err(json);
            assert_eq!(e.kind, kind, "wrong class for {json}: {e}");
            assert!(
                e.message.contains(needle),
                "message for {json} must contain {needle:?}, got {:?}",
                e.message
            );
        }
    }

    #[test]
    fn json_and_value_entry_points_agree() {
        // The C ABI's route (JSON) and the Python route (a converted dict) must
        // land on the same parse — that equality is the whole point of the
        // chokepoint.
        let json = r#"{"nodes": {"P": {"required": ["id"], "primary_key": "id"}}}"#;
        let via_json = schema_from_json(json).unwrap();
        let as_value =
            crate::param::json_value_to_kglite_value(&serde_json::from_str(json).unwrap());
        let via_value = schema_from_value(&as_value).unwrap();
        assert_eq!(
            via_json.node_schemas["P"].required_fields,
            via_value.node_schemas["P"].required_fields
        );
        assert_eq!(
            via_json.node_schemas["P"].primary_key,
            via_value.node_schemas["P"].primary_key
        );
    }

    /// A key the grammar cannot place is a *silent* constraint loss: the schema
    /// installs, reports success, and enforces nothing the misspelled key named.
    #[test]
    fn a_misspelled_declaration_key_is_refused_with_the_near_miss_named() {
        let e = err(r#"{"nodes": {"Task": {"uniqe": [["name"]]}}}"#);
        assert_eq!(e.kind, SchemaParseErrorKind::Key);
        assert!(e.message.contains("'uniqe'"), "{}", e.message);
        assert!(
            e.message.contains("Did you mean 'unique'?"),
            "{}",
            e.message
        );

        // Connection declarations get the same treatment.
        let e =
            err(r#"{"connections": {"R": {"source": "A", "target": "B", "cardinaliy": "1-1"}}}"#);
        assert_eq!(e.kind, SchemaParseErrorKind::Key);
        assert!(
            e.message.contains("Did you mean 'cardinality'?"),
            "{}",
            e.message
        );
    }

    /// A declaration key with no near miss still names the accepted set rather
    /// than being dropped.
    #[test]
    fn an_unrecognisable_declaration_key_lists_what_is_accepted() {
        let e = err(r#"{"nodes": {"Task": {"indexes": ["name"]}}}"#);
        assert_eq!(e.kind, SchemaParseErrorKind::Key);
        assert!(
            e.message.contains("node type 'Task'") && e.message.contains("'primary_key'"),
            "{}",
            e.message
        );
    }

    /// The forgotten `"nodes"` wrapper: every declaration used to be dropped and
    /// an empty schema installed, so a caller who wrote real constraints got
    /// none and no complaint.
    #[test]
    fn a_missing_nodes_wrapper_is_refused_with_the_wrapper_named() {
        let e = err(r#"{"Task": {"required": ["id"], "unique": [["name"]]}}"#);
        assert_eq!(e.kind, SchemaParseErrorKind::Key);
        assert!(e.message.contains("'nodes' wrapper"), "{}", e.message);
        assert!(e.message.contains("'Task'"), "{}", e.message);
    }

    /// A near-miss on the wrapper itself is a typo, not a forgotten wrapper —
    /// the suggestion is more useful than the wrapper lecture.
    #[test]
    fn a_misspelled_top_level_key_suggests_the_real_one() {
        let e = err(r#"{"node": {"Task": {"required": ["id"]}}}"#);
        assert!(e.message.contains("Did you mean 'nodes'?"), "{}", e.message);
    }

    /// A top-level key whose value is not a declaration map cannot be a
    /// forgotten wrapper, so it reads as what it is.
    #[test]
    fn an_unknown_scalar_top_level_key_is_reported_plainly() {
        let e = err(r#"{"version": 2}"#);
        assert_eq!(e.kind, SchemaParseErrorKind::Key);
        assert!(
            e.message.contains("Unknown top-level schema key"),
            "{}",
            e.message
        );
        assert!(
            e.message.contains("'nodes' and 'connections'"),
            "{}",
            e.message
        );
    }
}
