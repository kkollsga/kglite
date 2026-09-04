//! JSON request and response types for the two recipe tools.

use rmcp::model::CallToolResult;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::errors::RecipeErrorEnvelope;

#[derive(Debug, Default, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ListRecipeQueriesArgs {
    /// Optional recipe name. Omit for compact catalog summaries.
    pub(crate) recipe: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunRecipeQueryArgs {
    pub(crate) recipe: String,
    pub(crate) query: String,
    /// Required even for parameter-free operations (pass `{}`). This keeps a
    /// missing field distinguishable from an explicitly empty binding map.
    pub(crate) variables: Map<String, Value>,
    /// Include the stored parameterized Cypher and separate parameters map for
    /// audit. Both fields are omitted when this is false.
    #[serde(default)]
    pub(crate) include_cypher: bool,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub(crate) struct RecipeSummary {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) query_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) queries: Option<Vec<RecipeQuerySummary>>,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub(crate) struct RecipeQuerySummary {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) parameters: Value,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub(crate) struct ListRecipeQueriesSuccess {
    pub(super) recipes: Vec<RecipeSummary>,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
#[serde(untagged)]
pub(crate) enum ListRecipeQueriesOutput {
    Success(ListRecipeQueriesSuccess),
    Error(RecipeErrorEnvelope),
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub(crate) struct RecipeQueryResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Value>,
    pub(super) columns: Vec<String>,
    pub(super) rows: Vec<Vec<Value>>,
    pub(super) row_count: usize,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub(crate) struct RunRecipeQuerySuccess {
    pub(super) recipe: String,
    pub(super) query: String,
    pub(super) result: RecipeQueryResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cypher: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) parameters: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
#[serde(untagged)]
pub(crate) enum RunRecipeQueryOutput {
    Success(RunRecipeQuerySuccess),
    Error(RecipeErrorEnvelope),
}

impl ListRecipeQueriesOutput {
    pub(crate) fn into_call_tool_result(self) -> CallToolResult {
        match self {
            Self::Success(success) => structured_success(&success),
            Self::Error(error) => structured_error(&error),
        }
    }
}

impl RunRecipeQueryOutput {
    pub(crate) fn into_call_tool_result(self) -> CallToolResult {
        match self {
            Self::Success(success) => structured_success(&success),
            Self::Error(error) => structured_error(&error),
        }
    }
}

pub(crate) fn structured_error_result(error: RecipeErrorEnvelope) -> CallToolResult {
    structured_error(&error)
}

fn structured_success(value: &impl Serialize) -> CallToolResult {
    CallToolResult::structured(serde_json::to_value(value).expect("recipe success is serializable"))
}

fn structured_error(value: &impl Serialize) -> CallToolResult {
    CallToolResult::structured_error(
        serde_json::to_value(value).expect("recipe error is serializable"),
    )
}
