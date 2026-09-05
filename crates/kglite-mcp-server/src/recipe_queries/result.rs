//! Recipe-listing and strict structured execution.
//!
//! Wire types, error construction, and KGLite failure classification live in
//! sibling modules so this file remains the small orchestration seam that the
//! MCP routes will call.

use std::collections::HashMap;

use kglite::api::cypher::CypherResult;
use kglite::api::Value as KgliteValue;
use serde_json::Value;

use super::errors::RecipeErrorEnvelope;
use super::wire::{
    ListRecipeQueriesArgs, ListRecipeQueriesOutput, ListRecipeQueriesSuccess, RecipeQueryResult,
    RecipeQuerySummary, RecipeSummary, RunRecipeQueryArgs, RunRecipeQueryOutput,
    RunRecipeQuerySuccess,
};
use super::{RecipeCatalog, RecipeQueryDefinition, RECIPE_RESULT_ROW_LIMIT};
use crate::tools::{CypherRunError, GraphState, StrictCypherReadError};

pub(crate) fn list_recipe_queries(
    catalog: &RecipeCatalog,
    args: ListRecipeQueriesArgs,
) -> ListRecipeQueriesOutput {
    let recipes = match args.recipe {
        Some(name) => {
            let Some(recipe) = catalog.get(&name) else {
                return ListRecipeQueriesOutput::Error(RecipeErrorEnvelope::unknown_recipe(name));
            };
            vec![RecipeSummary {
                name: recipe.name.clone(),
                description: recipe.description.clone(),
                query_count: recipe.queries().len(),
                queries: Some(
                    recipe
                        .queries()
                        .map(|query| RecipeQuerySummary {
                            name: query.name.clone(),
                            description: query.description.clone(),
                            parameters: Value::Object(query.parameters.as_json().clone()),
                        })
                        .collect(),
                ),
            }]
        }
        None => catalog
            .recipes()
            .map(|recipe| RecipeSummary {
                name: recipe.name.clone(),
                description: recipe.description.clone(),
                query_count: recipe.queries().len(),
                queries: None,
            })
            .collect(),
    };
    ListRecipeQueriesOutput::Success(ListRecipeQueriesSuccess { recipes })
}

pub(crate) fn run_recipe_query(
    state: &GraphState,
    catalog: &RecipeCatalog,
    args: RunRecipeQueryArgs,
) -> RunRecipeQueryOutput {
    let Some(recipe) = catalog.get(&args.recipe) else {
        return RunRecipeQueryOutput::Error(RecipeErrorEnvelope::unknown_recipe(args.recipe));
    };
    let Some(query) = recipe.get(&args.query) else {
        return RunRecipeQueryOutput::Error(RecipeErrorEnvelope::unknown_query(
            args.recipe,
            args.query,
        ));
    };
    if let Err(error) = query.validate_variables(&args.variables) {
        return RunRecipeQueryOutput::Error(RecipeErrorEnvelope::invalid_variables(
            &args, query, error,
        ));
    }

    let params: HashMap<String, KgliteValue> = args
        .variables
        .iter()
        .map(|(name, value)| {
            (
                name.clone(),
                kglite::api::param::json_value_to_kglite_value(value),
            )
        })
        .collect();
    match state.execute_cypher_read_strict(&query.cypher, params) {
        Ok(outcome) => serialize_success(&args, query, &outcome.result),
        Err(StrictCypherReadError::StaleGraph(failure)) => {
            RunRecipeQueryOutput::Error(RecipeErrorEnvelope::stale_graph(&args, query, failure))
        }
        Err(StrictCypherReadError::Cypher(CypherRunError::NoActiveGraph)) => {
            RunRecipeQueryOutput::Error(RecipeErrorEnvelope::no_active_graph(&args, query))
        }
        Err(StrictCypherReadError::Cypher(CypherRunError::Engine(error))) => {
            RunRecipeQueryOutput::Error(RecipeErrorEnvelope::query_failed(&args, query, *error))
        }
        Err(StrictCypherReadError::Cypher(CypherRunError::MutationNotAllowed)) => {
            // Boot validation forbids this, but retain a typed failure if a
            // future caller constructs definitions through another path.
            RunRecipeQueryOutput::Error(RecipeErrorEnvelope::query_failure_cause(
                &args,
                query,
                super::errors::QueryFailureCause {
                    category: "invalid_argument".to_string(),
                    kglite_code: "InvalidArgument".to_string(),
                    message: "stored recipe query is not read-only".to_string(),
                    position: None,
                },
            ))
        }
    }
}

pub(super) fn serialize_success(
    args: &RunRecipeQueryArgs,
    query: &RecipeQueryDefinition,
    result: &CypherResult,
) -> RunRecipeQueryOutput {
    if result.rows.len() > RECIPE_RESULT_ROW_LIMIT {
        return RunRecipeQueryOutput::Error(RecipeErrorEnvelope::result_limit_exceeded(
            args,
            query,
            result.rows.len(),
        ));
    }
    let rows = result
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(kglite::api::param::kglite_value_to_json)
                .collect()
        })
        .collect();
    let (cypher, parameters) = super::errors::audit_fields(args, query);
    RunRecipeQueryOutput::Success(RunRecipeQuerySuccess {
        recipe: args.recipe.clone(),
        query: args.query.clone(),
        result: RecipeQueryResult {
            diagnostics: result.diagnostics.as_ref().map(|d| serde_json::json!(d)),
            columns: result.columns.clone(),
            rows,
            row_count: result.rows.len(),
        },
        cypher,
        parameters,
    })
}
