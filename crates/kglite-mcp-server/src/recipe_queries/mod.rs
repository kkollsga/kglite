//! Immutable, boot-validated Cypher recipe configuration.
//!
//! MCP route registration and result envelopes deliberately live in later
//! layers. This module owns only the manifest/query/schema contract so every
//! caller observes the same closed set of validated definitions.

mod config;
mod errors;
mod result;
mod schema;
mod validation;
mod wire;

#[cfg(test)]
mod result_tests;

pub(crate) use config::{CatalogSummary, RecipeCatalog, RecipeDefinition, RecipeQueryDefinition};
pub(crate) use errors::RecipeErrorEnvelope;
pub(crate) use result::{list_recipe_queries, run_recipe_query};
pub(crate) use schema::ParameterSchema;
pub(crate) use validation::{VariableIssue, VariableIssueKind, VariablesValidationError};
pub(crate) use wire::{
    ListRecipeQueriesArgs, ListRecipeQueriesOutput, RunRecipeQueryArgs, RunRecipeQueryOutput,
};

/// Maximum rows the structured recipe route may return in one MCP payload.
///
/// A stored literal `LIMIT` equal to this value is rejected at boot: it would
/// make an overflowing query look complete before the server can observe and
/// report its true cardinality.
pub(crate) const RECIPE_RESULT_ROW_LIMIT: usize = 200;
