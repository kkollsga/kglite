//! Construction of stable structured recipe errors.

use kglite::api::{KgError, KgErrorCode};
use schemars::JsonSchema;
use serde::Serialize;
use serde_json::{Map, Value};

use super::wire::RunRecipeQueryArgs;
use super::{
    RecipeQueryDefinition, VariableIssueKind, VariablesValidationError, RECIPE_RESULT_ROW_LIMIT,
};
use crate::tools::WorkspaceRebuildFailureSnapshot;

const MULTI_REVISION_REQUIRED_PREFIX: &str =
    "CALL rev_diff: this graph has no `revs` property — it is not a multi-rev graph.";
const UNKNOWN_REVISION_PREFIX: &str = "CALL rev_diff: revision ";
const UNKNOWN_REVISION_SUFFIX: &str = " is not present in this graph.";

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub(super) struct QueryFailureCause {
    /// Stable recipe-level category in snake_case.
    pub(super) category: String,
    /// Closest stable KGLite taxonomy code.
    pub(super) kglite_code: String,
    pub(super) message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) position: Option<QueryFailurePosition>,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub(super) struct QueryFailurePosition {
    line: usize,
    column: usize,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RecipeErrorCode {
    InvalidRequest,
    UnknownRecipe,
    UnknownQuery,
    InvalidVariables,
    NoActiveGraph,
    StaleGraph,
    QueryFailed,
    ResultLimitExceeded,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
pub(crate) struct RecipeErrorEnvelope {
    code: RecipeErrorCode,
    message: String,
    /// Boxed so the envelope stays small enough to travel by value through
    /// `Result` and the untagged output enums. `Box` is transparent to both
    /// `serde` and `schemars`, so the emitted JSON and schema are unchanged.
    details: Box<RecipeErrorDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cypher: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
#[serde(untagged)]
enum RecipeErrorDetails {
    InvalidRequest(InvalidRequestDetails),
    UnknownRecipe(UnknownRecipeDetails),
    UnknownQuery(UnknownQueryDetails),
    InvalidVariables(InvalidVariablesDetails),
    Query(QueryIdentityDetails),
    StaleGraph(StaleGraphDetails),
    QueryFailed(QueryFailedDetails),
    ResultLimitExceeded(ResultLimitExceededDetails),
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
struct InvalidRequestDetails {
    reason: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
struct UnknownRecipeDetails {
    recipe: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
struct UnknownQueryDetails {
    recipe: String,
    query: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
struct QueryIdentityDetails {
    recipe: String,
    query: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
struct InvalidVariablesDetails {
    recipe: String,
    query: String,
    missing: Vec<String>,
    unknown: Vec<String>,
    issues: Vec<VariableIssueDetail>,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
struct VariableIssueDetail {
    path: String,
    category: String,
    message: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
struct StaleGraphDetails {
    recipe: String,
    query: String,
    reason: String,
    failure_message: String,
    failed_at: String,
    consecutive_failures: u32,
    retry_limit: u32,
    recovery: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
struct QueryFailedDetails {
    recipe: String,
    query: String,
    cause: QueryFailureCause,
}

#[derive(Debug, Clone, Serialize, JsonSchema, PartialEq)]
struct ResultLimitExceededDetails {
    recipe: String,
    query: String,
    limit: usize,
    observed_count: usize,
}

impl RecipeErrorEnvelope {
    pub(crate) fn invalid_request(reason: impl Into<String>) -> Self {
        Self {
            code: RecipeErrorCode::InvalidRequest,
            message: "Recipe tool request is invalid.".to_string(),
            details: Box::new(RecipeErrorDetails::InvalidRequest(InvalidRequestDetails {
                reason: reason.into(),
            })),
            cypher: None,
            parameters: None,
        }
    }

    pub(super) fn unknown_recipe(recipe: String) -> Self {
        Self {
            code: RecipeErrorCode::UnknownRecipe,
            message: format!("Unknown recipe {recipe:?}."),
            details: Box::new(RecipeErrorDetails::UnknownRecipe(UnknownRecipeDetails {
                recipe,
            })),
            cypher: None,
            parameters: None,
        }
    }

    pub(super) fn unknown_query(recipe: String, query: String) -> Self {
        Self {
            code: RecipeErrorCode::UnknownQuery,
            message: format!("Unknown query {query:?} in recipe {recipe:?}."),
            details: Box::new(RecipeErrorDetails::UnknownQuery(UnknownQueryDetails {
                recipe,
                query,
            })),
            cypher: None,
            parameters: None,
        }
    }

    pub(super) fn invalid_variables(
        args: &RunRecipeQueryArgs,
        query: &RecipeQueryDefinition,
        error: VariablesValidationError,
    ) -> Self {
        let missing = top_level_issue_names(&error, VariableIssueKind::Missing);
        let unknown = top_level_issue_names(&error, VariableIssueKind::Unknown);
        let issues = error
            .issues
            .into_iter()
            .map(|issue| VariableIssueDetail {
                path: issue.path,
                category: issue_kind_name(issue.kind).to_string(),
                message: issue.message,
            })
            .collect();
        let (cypher, parameters) = audit_fields(args, query);
        Self {
            code: RecipeErrorCode::InvalidVariables,
            message: "Recipe query variables do not match the declared schema.".to_string(),
            details: Box::new(RecipeErrorDetails::InvalidVariables(
                InvalidVariablesDetails {
                    recipe: args.recipe.clone(),
                    query: args.query.clone(),
                    missing,
                    unknown,
                    issues,
                },
            )),
            cypher,
            parameters,
        }
    }

    pub(super) fn no_active_graph(
        args: &RunRecipeQueryArgs,
        query: &RecipeQueryDefinition,
    ) -> Self {
        let (cypher, parameters) = audit_fields(args, query);
        Self {
            code: RecipeErrorCode::NoActiveGraph,
            message: "No active graph is available for the recipe query.".to_string(),
            details: Box::new(RecipeErrorDetails::Query(QueryIdentityDetails {
                recipe: args.recipe.clone(),
                query: args.query.clone(),
            })),
            cypher,
            parameters,
        }
    }

    pub(super) fn stale_graph(
        args: &RunRecipeQueryArgs,
        query: &RecipeQueryDefinition,
        failure: WorkspaceRebuildFailureSnapshot,
    ) -> Self {
        let (cypher, parameters) = audit_fields(args, query);
        Self {
            code: RecipeErrorCode::StaleGraph,
            message: "Workspace graph rebuild failed; no stale recipe data was returned."
                .to_string(),
            details: Box::new(RecipeErrorDetails::StaleGraph(StaleGraphDetails {
                recipe: args.recipe.clone(),
                query: args.query.clone(),
                reason: failure.reason.code().to_string(),
                failure_message: failure.message,
                failed_at: chrono::DateTime::<chrono::Utc>::from(failure.failed_at)
                    .format("%Y-%m-%dT%H:%M:%SZ")
                    .to_string(),
                consecutive_failures: failure.consecutive_failures,
                retry_limit: failure.retry_limit,
                recovery: "Fix the workspace build failure or trigger a new relevant filesystem event, then retry."
                    .to_string(),
            })),
            cypher,
            parameters,
        }
    }

    pub(super) fn query_failed(
        args: &RunRecipeQueryArgs,
        query: &RecipeQueryDefinition,
        error: KgError,
    ) -> Self {
        Self::query_failure_cause(args, query, classify_query_failure(&error))
    }

    pub(super) fn query_failure_cause(
        args: &RunRecipeQueryArgs,
        query: &RecipeQueryDefinition,
        cause: QueryFailureCause,
    ) -> Self {
        let (cypher, parameters) = audit_fields(args, query);
        Self {
            code: RecipeErrorCode::QueryFailed,
            message: "Recipe query execution failed.".to_string(),
            details: Box::new(RecipeErrorDetails::QueryFailed(QueryFailedDetails {
                recipe: args.recipe.clone(),
                query: args.query.clone(),
                cause,
            })),
            cypher,
            parameters,
        }
    }

    pub(super) fn result_limit_exceeded(
        args: &RunRecipeQueryArgs,
        query: &RecipeQueryDefinition,
        observed_count: usize,
    ) -> Self {
        let (cypher, parameters) = audit_fields(args, query);
        Self {
            code: RecipeErrorCode::ResultLimitExceeded,
            message: format!(
                "Recipe query returned {observed_count} rows, exceeding the {RECIPE_RESULT_ROW_LIMIT}-row MCP payload limit."
            ),
            details: Box::new(RecipeErrorDetails::ResultLimitExceeded(ResultLimitExceededDetails {
                recipe: args.recipe.clone(),
                query: args.query.clone(),
                limit: RECIPE_RESULT_ROW_LIMIT,
                observed_count,
            })),
            cypher,
            parameters,
        }
    }
}

pub(super) fn audit_fields(
    args: &RunRecipeQueryArgs,
    query: &RecipeQueryDefinition,
) -> (Option<String>, Option<Map<String, Value>>) {
    if args.include_cypher {
        (Some(query.cypher.clone()), Some(args.variables.clone()))
    } else {
        (None, None)
    }
}

fn top_level_issue_names(error: &VariablesValidationError, kind: VariableIssueKind) -> Vec<String> {
    error
        .issues
        .iter()
        .filter(|issue| issue.kind == kind)
        .filter_map(|issue| issue.path.strip_prefix("$.").map(str::to_string))
        .filter(|path| !path.contains(['.', '[']))
        .collect()
}

const fn issue_kind_name(kind: VariableIssueKind) -> &'static str {
    match kind {
        VariableIssueKind::Missing => "missing",
        VariableIssueKind::Unknown => "unknown",
        VariableIssueKind::WrongType => "wrong_type",
        VariableIssueKind::IntegerRange => "integer_range",
        VariableIssueKind::Enum => "enum",
        VariableIssueKind::Minimum => "minimum",
        VariableIssueKind::Maximum => "maximum",
        VariableIssueKind::MinItems => "min_items",
        VariableIssueKind::MaxItems => "max_items",
    }
}

fn classify_query_failure(error: &KgError) -> QueryFailureCause {
    let category = match error {
        KgError::CypherExecution { message, .. }
            if message.starts_with(MULTI_REVISION_REQUIRED_PREFIX) =>
        {
            "multi_revision_graph_required"
        }
        KgError::CypherExecution { message, .. }
            if message.starts_with(UNKNOWN_REVISION_PREFIX)
                && message.contains(UNKNOWN_REVISION_SUFFIX) =>
        {
            "unknown_revision"
        }
        _ => kg_error_category(error.code()),
    };
    QueryFailureCause {
        category: category.to_string(),
        kglite_code: error.code().as_str().to_string(),
        message: error.to_string(),
        position: error
            .position()
            .map(|(line, column)| QueryFailurePosition { line, column }),
    }
}

const fn kg_error_category(code: KgErrorCode) -> &'static str {
    match code {
        KgErrorCode::CypherSyntax => "cypher_syntax",
        KgErrorCode::CypherTimeout => "cypher_timeout",
        KgErrorCode::CypherExecution => "cypher_execution",
        KgErrorCode::CypherTypeMismatch => "cypher_type_mismatch",
        KgErrorCode::Cancelled => "cancelled",
        KgErrorCode::Schema => "schema",
        KgErrorCode::Validation => "validation",
        KgErrorCode::Expr => "expr",
        KgErrorCode::ConstraintViolation => "constraint_violation",
        KgErrorCode::ConstraintCreationFailed => "constraint_creation_failed",
        KgErrorCode::TransactionConflict => "transaction_conflict",
        KgErrorCode::NodeNotFound => "node_not_found",
        KgErrorCode::ConnectionNotFound => "connection_not_found",
        KgErrorCode::PropertyNotFound => "property_not_found",
        KgErrorCode::FileNotFound => "file_not_found",
        KgErrorCode::FileFormat => "file_format",
        KgErrorCode::FileIo => "file_io",
        KgErrorCode::InvalidArgument => "invalid_argument",
        KgErrorCode::MissingArgument => "missing_argument",
        KgErrorCode::Internal => "internal",
    }
}
