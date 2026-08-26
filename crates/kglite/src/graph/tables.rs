//! Structured-data support over the existing list/map substrate: the
//! table-property fidelity metadata and the structured property shapes.
//!
//! Deliberately **no** `Value::Table`: a "table" property is stored as a
//! plain `list<map>` — queryable with today's Cypher (`UNWIND o.line_items`,
//! `o.line_items[2].qty`) and today's persistence. What a bare list of maps
//! loses is DataFrame fidelity (`PropMap` keys are sorted, so column order
//! is gone, and dtypes/nullability are per-cell) — [`TablePropertyMeta`]
//! restores exactly that at reconstruction time, and nothing else reads it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::datatypes::values::Value;

/// Column-order + dtype record for one table-valued property, keyed in
/// `DirGraph::table_property_meta` by `"NodeType\u{1f}property"` (unit
/// separator — neither side may contain it, both are validated identifiers).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TablePropertyMeta {
    /// Column names in original DataFrame order.
    pub columns: Vec<String>,
    /// Declared dtype string per column (pandas dtype spelling, e.g.
    /// `int64`, `float64`, `object`, `datetime64[ns]`, `boolean`).
    pub dtypes: BTreeMap<String, String>,
    /// Columns that contained at least one null when stored — reconstruction
    /// uses pandas nullable dtypes for these.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nullable: Vec<String>,
}

/// Registry key for `(node_type, property)`.
pub fn table_meta_key(node_type: &str, property: &str) -> String {
    format!("{node_type}\u{1f}{property}")
}

// ─── structured property shapes ───────────────────────────────────────────

/// A declared shape for a collection-valued property — the vocabulary the
/// `IS :: TYPE` constraint deliberately rejects (`property_types.rs`:
/// `LIST<…>` resolves to `None`). Declared through `define_schema`'s
/// `types` map with the grammar below, stored in a side table
/// (`DirGraph::property_shapes`) so the `.kgl`-format `DeclaredType` enum
/// stays untouched, and enforced at the same three write gates the DDL
/// property-type constraints use (Cypher SET, CREATE, the batch funnel).
/// **WAL replay never validates** — the log is authoritative.
///
/// Grammar (whitespace-insensitive):
///
/// ```text
/// list<int>                                — homogeneous scalar list
/// list<map{sku: string!, qty: int!, price: float}>
/// map{status: string!, note: string}
/// ```
///
/// `!` marks a required key; unmarked keys may be absent or null. Scalar
/// names accept the `define_schema` spellings (`string|str`,
/// `int|integer`, `float`, `bool|boolean`, `date|datetime`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyShape {
    /// `list<inner>`
    List(Box<PropertyShape>),
    /// `map{key: shape[!], ...}` — the bool is "required".
    Map(BTreeMap<String, (PropertyShape, bool)>),
    Scalar(ScalarShape),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarShape {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
}

impl ScalarShape {
    fn name(&self) -> &'static str {
        match self {
            ScalarShape::String => "string",
            ScalarShape::Integer => "integer",
            ScalarShape::Float => "float",
            ScalarShape::Boolean => "boolean",
            ScalarShape::DateTime => "datetime",
        }
    }

    fn accepts(&self, value: &Value) -> bool {
        matches!(
            (self, value),
            (ScalarShape::String, Value::String(_))
                | (ScalarShape::Integer, Value::Int64(_))
                | (ScalarShape::Integer, Value::UniqueId(_))
                | (ScalarShape::Float, Value::Float64(_))
                | (ScalarShape::Boolean, Value::Boolean(_))
                | (ScalarShape::DateTime, Value::DateTime(_))
                | (ScalarShape::DateTime, Value::Timestamp(_))
        )
    }
}

impl PropertyShape {
    /// Render back to the declaration grammar (stable — used by describe()
    /// and error messages).
    pub fn render(&self) -> String {
        match self {
            PropertyShape::Scalar(s) => s.name().to_string(),
            PropertyShape::List(inner) => format!("list<{}>", inner.render()),
            PropertyShape::Map(fields) => {
                let inner: Vec<String> = fields
                    .iter()
                    .map(|(k, (shape, required))| {
                        format!(
                            "{k}: {}{}",
                            shape.render(),
                            if *required { "!" } else { "" }
                        )
                    })
                    .collect();
                format!("map{{{}}}", inner.join(", "))
            }
        }
    }

    /// Validate `value` against the shape, returning the first violation as
    /// an indexed path (`line_items[37].qty: expected integer, got String`).
    /// Null/absent at the top level passes (constraint semantics: a type
    /// declaration is not an existence declaration).
    pub fn check(&self, property: &str, value: &Value) -> Result<(), String> {
        if matches!(value, Value::Null) {
            return Ok(());
        }
        self.check_at(property, value)
    }

    fn check_at(&self, path: &str, value: &Value) -> Result<(), String> {
        match self {
            PropertyShape::Scalar(scalar) => {
                if matches!(value, Value::Null) || scalar.accepts(value) {
                    Ok(())
                } else {
                    Err(format!(
                        "{path}: expected {}, got {}",
                        scalar.name(),
                        value.type_name()
                    ))
                }
            }
            PropertyShape::List(inner) => match value {
                Value::List(items) => {
                    for (i, item) in items.iter().enumerate() {
                        inner.check_at(&format!("{path}[{i}]"), item)?;
                    }
                    Ok(())
                }
                _ => Err(format!(
                    "{path}: expected a list, got {}",
                    value.type_name()
                )),
            },
            PropertyShape::Map(fields) => match value {
                Value::Map(map) => {
                    for (key, (shape, required)) in fields {
                        match map.get(key) {
                            Some(Value::Null) | None if *required => {
                                return Err(format!("{path}.{key}: required key is missing"));
                            }
                            Some(v) => shape.check_at(&format!("{path}.{key}"), v)?,
                            None => {}
                        }
                    }
                    for (key, _) in map {
                        if !fields.contains_key(key) {
                            return Err(format!(
                                "{path}.{key}: key is not in the declared shape ({})",
                                self.render()
                            ));
                        }
                    }
                    Ok(())
                }
                _ => Err(format!("{path}: expected a map, got {}", value.type_name())),
            },
        }
    }
}

/// Parse the declaration grammar. `None` when `text` does not LOOK like a
/// shape (no `list<` / `map{` head) — the caller then treats it as a plain
/// advisory type string exactly as before, so nothing existing changes
/// meaning. A malformed shape (looks structured, does not parse) is an
/// error, never silently advisory — the `property_types`-as-rename lesson.
pub fn parse_property_shape(text: &str) -> Option<Result<PropertyShape, String>> {
    let trimmed = text.trim();
    if !(trimmed.starts_with("list<") || trimmed.starts_with("map{")) {
        return None;
    }
    let mut p = ShapeParser {
        chars: trimmed.char_indices().peekable(),
        src: trimmed,
    };
    Some(p.parse_shape().and_then(|shape| {
        p.skip_ws();
        match p.chars.peek() {
            None => Ok(shape),
            Some((i, _)) => Err(format!(
                "property shape: unexpected trailing input at '{}'",
                &p.src[*i..]
            )),
        }
    }))
}

struct ShapeParser<'a> {
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    src: &'a str,
}

impl ShapeParser<'_> {
    fn skip_ws(&mut self) {
        while matches!(self.chars.peek(), Some((_, c)) if c.is_whitespace()) {
            self.chars.next();
        }
    }

    fn eat(&mut self, expected: char) -> Result<(), String> {
        self.skip_ws();
        match self.chars.next() {
            Some((_, c)) if c == expected => Ok(()),
            other => Err(format!(
                "property shape: expected '{expected}', found {:?}",
                other.map(|(_, c)| c)
            )),
        }
    }

    fn word(&mut self) -> String {
        self.skip_ws();
        let mut out = String::new();
        while matches!(self.chars.peek(), Some((_, c)) if c.is_alphanumeric() || *c == '_') {
            out.push(self.chars.next().unwrap().1);
        }
        out
    }

    fn parse_shape(&mut self) -> Result<PropertyShape, String> {
        let head = self.word();
        match head.as_str() {
            "list" => {
                self.eat('<')?;
                let inner = self.parse_shape()?;
                self.eat('>')?;
                Ok(PropertyShape::List(Box::new(inner)))
            }
            "map" => {
                self.eat('{')?;
                let mut fields = BTreeMap::new();
                loop {
                    self.skip_ws();
                    if matches!(self.chars.peek(), Some((_, '}'))) {
                        self.chars.next();
                        break;
                    }
                    let key = self.word();
                    if key.is_empty() {
                        return Err("property shape: expected a key name in map{...}".to_string());
                    }
                    self.eat(':')?;
                    let shape = self.parse_shape()?;
                    self.skip_ws();
                    let required = if matches!(self.chars.peek(), Some((_, '!'))) {
                        self.chars.next();
                        true
                    } else {
                        false
                    };
                    if fields.insert(key.clone(), (shape, required)).is_some() {
                        return Err(format!("property shape: duplicate key '{key}'"));
                    }
                    self.skip_ws();
                    if matches!(self.chars.peek(), Some((_, ','))) {
                        self.chars.next();
                    }
                }
                Ok(PropertyShape::Map(fields))
            }
            scalar => match scalar {
                "string" | "str" => Ok(PropertyShape::Scalar(ScalarShape::String)),
                "int" | "integer" => Ok(PropertyShape::Scalar(ScalarShape::Integer)),
                "float" => Ok(PropertyShape::Scalar(ScalarShape::Float)),
                "bool" | "boolean" => Ok(PropertyShape::Scalar(ScalarShape::Boolean)),
                "date" | "datetime" => Ok(PropertyShape::Scalar(ScalarShape::DateTime)),
                other => Err(format!(
                    "property shape: unknown type '{other}' (accepted: string, int, float, \
                     bool, date, list<...>, map{{...}})"
                )),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::PropMap;

    fn shape(text: &str) -> PropertyShape {
        parse_property_shape(text)
            .expect("looks structured")
            .expect("parses")
    }

    #[test]
    fn grammar_round_trips() {
        let s = shape("list<map{sku: string!, qty: int!, price: float}>");
        assert_eq!(
            s.render(),
            "list<map{price: float, qty: integer!, sku: string!}>"
        );
        assert_eq!(shape("list<int>").render(), "list<integer>");
        assert_eq!(shape("map{a: bool}").render(), "map{a: boolean}");
    }

    #[test]
    fn plain_types_are_not_shapes() {
        assert!(parse_property_shape("string").is_none());
        assert!(parse_property_shape("integer").is_none());
        // Structured-looking but malformed is an ERROR, never advisory.
        assert!(parse_property_shape("list<oops>").unwrap().is_err());
        assert!(parse_property_shape("map{a string}").unwrap().is_err());
        assert!(parse_property_shape("list<int> trailing").unwrap().is_err());
    }

    #[test]
    fn validation_reports_indexed_paths() {
        let s = shape("list<map{sku: string!, qty: int!, price: float}>");
        let good_row = |sku: &str, qty: i64| {
            Value::Map(PropMap::from_pairs(vec![
                ("sku".into(), Value::String(sku.to_string())),
                ("qty".into(), Value::Int64(qty)),
                ("price".into(), Value::Float64(9.5)),
            ]))
        };
        let ok = Value::List(vec![good_row("a", 1), good_row("b", 2)]);
        assert!(s.check("line_items", &ok).is_ok());

        let mut rows = vec![good_row("a", 1)];
        rows.push(Value::Map(PropMap::from_pairs(vec![
            ("sku".into(), Value::String("c".into())),
            ("qty".into(), Value::String("eight".into())),
        ])));
        let err = s.check("line_items", &Value::List(rows)).unwrap_err();
        assert_eq!(err, "line_items[1].qty: expected integer, got String");

        let missing = Value::List(vec![Value::Map(PropMap::from_pairs(vec![(
            "sku".into(),
            Value::String("x".into()),
        )]))]);
        let err = s.check("line_items", &missing).unwrap_err();
        assert!(
            err.contains("line_items[0].qty: required key is missing"),
            "{err}"
        );

        let extra = Value::List(vec![Value::Map(PropMap::from_pairs(vec![
            ("sku".into(), Value::String("x".into())),
            ("qty".into(), Value::Int64(1)),
            ("colour".into(), Value::String("red".into())),
        ]))]);
        let err = s.check("line_items", &extra).unwrap_err();
        assert!(err.contains("line_items[0].colour"), "{err}");

        // Null top-level value passes; null optional cell passes.
        assert!(s.check("line_items", &Value::Null).is_ok());
    }
}
