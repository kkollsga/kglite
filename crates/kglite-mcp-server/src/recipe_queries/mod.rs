//! Immutable, boot-validated Cypher recipe configuration.
//!
//! MCP route registration and result envelopes deliberately live in later
//! layers. This module owns only the manifest/query/schema contract so every
//! caller observes the same closed set of validated definitions.

mod config;
mod schema;
mod validation;

pub(crate) use config::{CatalogSummary, RecipeCatalog, RecipeDefinition, RecipeQueryDefinition};
pub(crate) use schema::ParameterSchema;
pub(crate) use validation::{VariableIssue, VariableIssueKind, VariablesValidationError};

/// Maximum rows the structured recipe route may return in one MCP payload.
///
/// A stored literal `LIMIT` equal to this value is rejected at boot: it would
/// make an overflowing query look complete before the server can observe and
/// report its true cardinality.
pub(crate) const RECIPE_RESULT_ROW_LIMIT: usize = 200;
