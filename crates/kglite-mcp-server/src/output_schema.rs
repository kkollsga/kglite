//! Tool output schemas declared as object schemas at the root.
//!
//! MCP 2025-06-18 requires `Tool.outputSchema` to carry `"type": "object"` at
//! its root, and strict clients (the Claude desktop app) reject the whole
//! `tools/list` when one tool's schema does not. schemars renders an untagged
//! enum as a bare `anyOf`, so every route declares its schema through
//! [`ObjectOutputSchema::with_object_output_schema`].

use rmcp::model::Tool;
use schemars::JsonSchema;
use std::sync::Arc;

pub(crate) trait ObjectOutputSchema {
    /// `with_output_schema::<T>()`, then `"type": "object"` at the root when
    /// the generated schema has no root type. Valid for the union shapes used
    /// here because every branch is itself an object.
    ///
    /// Non-standard integer `format` values are not handled here:
    /// `McpServer` standardizes every tool schema it lists.
    fn with_object_output_schema<T: JsonSchema + 'static>(self) -> Self;
}

impl ObjectOutputSchema for Tool {
    fn with_object_output_schema<T: JsonSchema + 'static>(self) -> Self {
        let mut tool = self.with_output_schema::<T>();
        if let Some(schema) = tool.output_schema.as_mut() {
            Arc::make_mut(schema)
                .entry("type")
                .or_insert_with(|| serde_json::Value::from("object"));
        }
        tool
    }
}

#[cfg(test)]
mod tests {
    use mcp_methods::server::schema::standardize;
    use serde_json::json;

    /// The framework's cleanup is what keeps these schemas portable, so the
    /// cases that once needed a local copy are pinned against it.
    #[test]
    fn a_property_named_default_or_enum_is_still_a_schema() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "default": {"type": "integer", "format": "uint64"},
                "enum": {"type": "integer", "format": "int64"},
            },
        });
        standardize(&mut schema);
        assert!(schema["properties"]["default"].get("format").is_none());
        assert_eq!(schema["properties"]["default"]["minimum"], 0);
        assert!(schema["properties"]["enum"].get("format").is_none());
    }

    #[test]
    fn instance_data_keeps_a_format_key() {
        let mut schema = json!({
            "type": "object",
            "default": {"format": "csv"},
            "examples": [{"format": "csv"}],
            "properties": {"out": {"const": {"format": "csv"}, "enum": [{"format": "csv"}]}},
        });
        standardize(&mut schema);
        assert_eq!(schema["default"]["format"], "csv");
        assert_eq!(schema["examples"][0]["format"], "csv");
        assert_eq!(schema["properties"]["out"]["const"]["format"], "csv");
        assert_eq!(schema["properties"]["out"]["enum"][0]["format"], "csv");
    }
}
