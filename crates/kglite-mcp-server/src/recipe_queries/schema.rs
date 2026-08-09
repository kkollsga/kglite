use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Context, Result};
use serde_json::{Map, Number, Value};

use super::validation::{
    json_equal, number_cmp, validate_exact_i64_recursive, VariablesValidationError,
};

const ROOT_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "description",
];

const ALLOWED_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "additionalProperties",
    "description",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ValueType {
    Null,
    Boolean,
    Object,
    Array,
    Number,
    Integer,
    String,
}

impl ValueType {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "null" => Ok(Self::Null),
            "boolean" => Ok(Self::Boolean),
            "object" => Ok(Self::Object),
            "array" => Ok(Self::Array),
            "number" => Ok(Self::Number),
            "integer" => Ok(Self::Integer),
            "string" => Ok(Self::String),
            other => bail!("unsupported JSON Schema type {other:?}"),
        }
    }

    pub(super) fn display(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Boolean => "boolean",
            Self::Object => "object",
            Self::Array => "array",
            Self::Number => "number",
            Self::Integer => "integer",
            Self::String => "string",
        }
    }
}

/// A boot-compiled schema retaining its source shape for MCP discovery.
#[derive(Debug, Clone)]
pub(crate) struct ParameterSchema {
    raw: Map<String, Value>,
    root: SchemaNode,
}

impl ParameterSchema {
    pub(crate) fn compile_root(raw: &Value, cypher_parameters: &[String]) -> Result<Self> {
        let raw_map = raw
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("must be a mapping"))?;
        reject_unknown_keys(raw_map, ROOT_KEYWORDS, "parameters")?;
        for required_keyword in ["properties", "required", "additionalProperties"] {
            if !raw_map.contains_key(required_keyword) {
                bail!("root {required_keyword} is required");
            }
        }
        let root = SchemaNode::compile(raw, "parameters")?;
        if root.types != BTreeSet::from([ValueType::Object]) {
            bail!("root type must be exactly \"object\"");
        }
        if root.additional_properties != Some(false) {
            bail!("root additionalProperties must be explicitly false");
        }

        let property_names: BTreeSet<_> = root.properties.keys().cloned().collect();
        let referenced: BTreeSet<_> = cypher_parameters.iter().cloned().collect();
        if property_names != referenced {
            let missing: Vec<_> = referenced.difference(&property_names).cloned().collect();
            let unused: Vec<_> = property_names.difference(&referenced).cloned().collect();
            bail!(
                "parameter properties must exactly match Cypher $parameters; missing={missing:?}, unused={unused:?}"
            );
        }
        if root.required != property_names {
            let optional: Vec<_> = property_names.difference(&root.required).cloned().collect();
            let unknown: Vec<_> = root.required.difference(&property_names).cloned().collect();
            bail!(
                "required must list every parameter property exactly; optional={optional:?}, unknown={unknown:?}"
            );
        }

        Ok(Self {
            raw: raw_map.clone(),
            root,
        })
    }

    pub(crate) fn as_json(&self) -> &Map<String, Value> {
        &self.raw
    }

    pub(crate) fn validate_variables(
        &self,
        variables: &Map<String, Value>,
    ) -> Result<(), VariablesValidationError> {
        let mut issues = Vec::new();
        validate_exact_i64_recursive(&Value::Object(variables.clone()), "$", &mut issues);
        self.root
            .validate(&Value::Object(variables.clone()), "$", true, &mut issues);
        if issues.is_empty() {
            Ok(())
        } else {
            Err(VariablesValidationError { issues })
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct SchemaNode {
    pub(super) types: BTreeSet<ValueType>,
    pub(super) properties: BTreeMap<String, SchemaNode>,
    pub(super) required: BTreeSet<String>,
    pub(super) items: Option<Box<SchemaNode>>,
    pub(super) enum_values: Option<Vec<Value>>,
    pub(super) minimum: Option<Number>,
    pub(super) maximum: Option<Number>,
    pub(super) min_items: Option<usize>,
    pub(super) max_items: Option<usize>,
    pub(super) additional_properties: Option<bool>,
}

impl SchemaNode {
    fn compile(raw: &Value, path: &str) -> Result<Self> {
        let map = raw
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("{path} must be a schema mapping"))?;
        reject_unknown_keywords(map, path)?;
        if let Some(description) = map.get("description") {
            if !description.is_string() {
                bail!("{path}.description must be a string");
            }
        }

        let types = parse_types(
            map.get("type")
                .ok_or_else(|| anyhow::anyhow!("{path}.type is required"))?,
            path,
        )?;

        let properties = parse_properties(map.get("properties"), path)?;
        let required = parse_required(map.get("required"), path)?;
        let additional_properties = parse_bool_keyword(map, "additionalProperties", path)?;
        let items = map
            .get("items")
            .map(|value| SchemaNode::compile(value, &format!("{path}.items")).map(Box::new))
            .transpose()?;
        let enum_values = parse_enum(map.get("enum"), path)?;
        let minimum = parse_number_keyword(map, "minimum", path)?;
        let maximum = parse_number_keyword(map, "maximum", path)?;
        let min_items = parse_usize_keyword(map, "minItems", path)?;
        let max_items = parse_usize_keyword(map, "maxItems", path)?;

        validate_keyword_applicability(
            &types,
            &KeywordPresence {
                object: !properties.is_empty()
                    || map.contains_key("properties")
                    || map.contains_key("required")
                    || map.contains_key("additionalProperties"),
                items: items.is_some(),
                numeric_bounds: minimum.is_some() || maximum.is_some(),
                item_bounds: min_items.is_some() || max_items.is_some(),
            },
            path,
        )?;

        let unknown_required: Vec<_> = required
            .difference(&properties.keys().cloned().collect())
            .cloned()
            .collect();
        if !unknown_required.is_empty() {
            bail!("{path}.required names unknown properties {unknown_required:?}");
        }
        if let (Some(minimum), Some(maximum)) = (&minimum, &maximum) {
            if number_cmp(minimum, maximum) == Some(Ordering::Greater) {
                bail!("{path}.minimum must not exceed maximum");
            }
        }
        if let (Some(min_items), Some(max_items)) = (min_items, max_items) {
            if min_items > max_items {
                bail!("{path}.minItems must not exceed maxItems");
            }
        }

        let node = Self {
            types,
            properties,
            required,
            items,
            enum_values,
            minimum,
            maximum,
            min_items,
            max_items,
            additional_properties,
        };
        node.validate_enum_values(path)?;
        Ok(node)
    }

    fn validate_enum_values(&self, path: &str) -> Result<()> {
        let Some(values) = self.enum_values.as_ref() else {
            return Ok(());
        };
        for (index, value) in values.iter().enumerate() {
            let mut issues = Vec::new();
            validate_exact_i64_recursive(value, &format!("{path}.enum[{index}]"), &mut issues);
            self.validate(value, &format!("{path}.enum[{index}]"), false, &mut issues);
            if let Some(issue) = issues.first() {
                bail!("{}", issue.message);
            }
        }
        Ok(())
    }
}

fn reject_unknown_keywords(map: &Map<String, Value>, path: &str) -> Result<()> {
    reject_unknown_keys(map, ALLOWED_KEYWORDS, path)
}

fn reject_unknown_keys(map: &Map<String, Value>, allowed: &[&str], path: &str) -> Result<()> {
    let allowed: BTreeSet<_> = allowed.iter().copied().collect();
    let unknown: Vec<_> = map
        .keys()
        .filter(|keyword| !allowed.contains(keyword.as_str()))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        bail!("{path} uses unsupported JSON Schema keywords {unknown:?}");
    }
    Ok(())
}

fn parse_types(raw: &Value, path: &str) -> Result<BTreeSet<ValueType>> {
    let names: Vec<&str> = match raw {
        Value::String(name) => vec![name],
        Value::Array(names) if !names.is_empty() => names
            .iter()
            .map(|name| {
                name.as_str()
                    .ok_or_else(|| anyhow::anyhow!("{path}.type array must contain strings"))
            })
            .collect::<Result<_>>()?,
        Value::Array(_) => bail!("{path}.type array must not be empty"),
        _ => bail!("{path}.type must be a string or non-empty string array"),
    };
    let mut types = BTreeSet::new();
    for name in names {
        let parsed = ValueType::parse(name).with_context(|| format!("{path}.type"))?;
        if !types.insert(parsed) {
            bail!("{path}.type contains duplicate type {name:?}");
        }
    }
    Ok(types)
}

fn parse_properties(raw: Option<&Value>, path: &str) -> Result<BTreeMap<String, SchemaNode>> {
    let Some(raw) = raw else {
        return Ok(BTreeMap::new());
    };
    let map = raw
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{path}.properties must be a mapping"))?;
    map.iter()
        .map(|(name, value)| {
            SchemaNode::compile(value, &format!("{path}.properties.{name}"))
                .map(|schema| (name.clone(), schema))
        })
        .collect()
}

fn parse_required(raw: Option<&Value>, path: &str) -> Result<BTreeSet<String>> {
    let Some(raw) = raw else {
        return Ok(BTreeSet::new());
    };
    let items = raw
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("{path}.required must be an array of strings"))?;
    let mut required = BTreeSet::new();
    for item in items {
        let name = item
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| anyhow::anyhow!("{path}.required must contain non-empty strings"))?;
        if !required.insert(name.to_string()) {
            bail!("{path}.required contains duplicate property {name:?}");
        }
    }
    Ok(required)
}

fn parse_enum(raw: Option<&Value>, path: &str) -> Result<Option<Vec<Value>>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let values = raw
        .as_array()
        .filter(|values| !values.is_empty())
        .ok_or_else(|| anyhow::anyhow!("{path}.enum must be a non-empty array"))?;
    for (index, value) in values.iter().enumerate() {
        if values[..index].iter().any(|other| json_equal(other, value)) {
            bail!("{path}.enum contains duplicate value {value}");
        }
    }
    Ok(Some(values.clone()))
}

fn parse_bool_keyword(map: &Map<String, Value>, keyword: &str, path: &str) -> Result<Option<bool>> {
    map.get(keyword)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| anyhow::anyhow!("{path}.{keyword} must be a boolean"))
        })
        .transpose()
}

fn parse_number_keyword(
    map: &Map<String, Value>,
    keyword: &str,
    path: &str,
) -> Result<Option<Number>> {
    map.get(keyword)
        .map(|value| {
            let number = value
                .as_number()
                .ok_or_else(|| anyhow::anyhow!("{path}.{keyword} must be a number"))?;
            reject_out_of_range_integer(number, &format!("{path}.{keyword}"))?;
            Ok(number.clone())
        })
        .transpose()
}

fn parse_usize_keyword(
    map: &Map<String, Value>,
    keyword: &str,
    path: &str,
) -> Result<Option<usize>> {
    map.get(keyword)
        .map(|value| {
            let number = value.as_u64().ok_or_else(|| {
                anyhow::anyhow!("{path}.{keyword} must be a non-negative integer")
            })?;
            usize::try_from(number)
                .map_err(|_| anyhow::anyhow!("{path}.{keyword} exceeds this platform's size range"))
        })
        .transpose()
}

/// Which keyword groups a compiled schema mapping actually declared.
struct KeywordPresence {
    object: bool,
    items: bool,
    numeric_bounds: bool,
    item_bounds: bool,
}

fn validate_keyword_applicability(
    types: &BTreeSet<ValueType>,
    present: &KeywordPresence,
    path: &str,
) -> Result<()> {
    if present.object && !types.contains(&ValueType::Object) {
        bail!("{path} uses object keywords without type object");
    }
    if present.items && !types.contains(&ValueType::Array) {
        bail!("{path}.items requires type array");
    }
    if present.item_bounds && !types.contains(&ValueType::Array) {
        bail!("{path} uses minItems/maxItems without type array");
    }
    if present.numeric_bounds
        && !types.contains(&ValueType::Number)
        && !types.contains(&ValueType::Integer)
    {
        bail!("{path} uses minimum/maximum without type number or integer");
    }
    Ok(())
}

fn reject_out_of_range_integer(number: &Number, path: &str) -> Result<()> {
    if number.as_i64().is_none() && number.as_u64().is_some() {
        bail!("{path} integer is outside KGLite's exact signed 64-bit range");
    }
    Ok(())
}
