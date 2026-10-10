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
