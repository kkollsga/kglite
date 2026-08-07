use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Context, Result};
use kglite::api::cypher;
use serde_json::{Map, Value};

use super::{ParameterSchema, VariablesValidationError, RECIPE_RESULT_ROW_LIMIT};

const RECIPE_KEYS: &[&str] = &["description", "queries"];
const QUERY_KEYS: &[&str] = &["description", "parameters", "cypher"];

/// Cheap immutable catalog dimensions used by boot logging and discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CatalogSummary {
    pub(crate) recipe_count: usize,
    pub(crate) query_count: usize,
}

/// All validated recipes from one manifest, ordered deterministically by name.
#[derive(Debug, Clone, Default)]
pub(crate) struct RecipeCatalog {
    recipes: BTreeMap<String, RecipeDefinition>,
}

impl RecipeCatalog {
    /// Parse `extensions.cypher_recipes`. Absence and an empty mapping are the
    /// only disabled shapes; malformed or partially configured catalogs fail
    /// server boot.
    pub(crate) fn from_manifest_value(raw: Option<&Value>) -> Result<Self> {
        let Some(raw) = raw else {
            return Ok(Self::default());
        };
        let recipes = raw
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("must be a mapping of recipe names"))?;
        if recipes.is_empty() {
            return Ok(Self::default());
        }

        let mut parsed = BTreeMap::new();
        for (name, raw_recipe) in recipes {
            validate_identifier(name, "recipe")?;
            let recipe = RecipeDefinition::parse(name, raw_recipe)
                .with_context(|| format!("recipe {name:?}"))?;
            parsed.insert(name.clone(), recipe);
        }
        Ok(Self { recipes: parsed })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.recipes.is_empty()
    }

    pub(crate) fn summary(&self) -> CatalogSummary {
        CatalogSummary {
            recipe_count: self.recipes.len(),
            query_count: self
                .recipes
                .values()
                .map(|recipe| recipe.queries.len())
                .sum(),
        }
    }

    /// Discovery dimensions only when route registration is enabled.
    /// Absent and explicitly empty manifests both compile to an empty catalog
    /// and therefore contribute no overview hint.
    pub(crate) fn discovery_summary(&self) -> Option<CatalogSummary> {
        (!self.is_empty()).then(|| self.summary())
    }

    pub(crate) fn recipes(&self) -> impl ExactSizeIterator<Item = &RecipeDefinition> {
        self.recipes.values()
    }

    pub(crate) fn get(&self, name: &str) -> Option<&RecipeDefinition> {
        self.recipes.get(name)
    }
}

/// One named group of related query operations.
#[derive(Debug, Clone)]
pub(crate) struct RecipeDefinition {
    pub(crate) name: String,
    pub(crate) description: String,
    queries: BTreeMap<String, RecipeQueryDefinition>,
}

impl RecipeDefinition {
    fn parse(name: &str, raw: &Value) -> Result<Self> {
        let map = object_with_only(raw, RECIPE_KEYS, "recipe")?;
        let description = required_nonempty_string(map, "description")?;
        let raw_queries = map
            .get("queries")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("queries must be a non-empty mapping"))?;
        if raw_queries.is_empty() {
            bail!("queries must be a non-empty mapping");
        }

        let mut queries = BTreeMap::new();
        for (query_name, raw_query) in raw_queries {
            validate_identifier(query_name, "query")?;
            let query = RecipeQueryDefinition::parse(query_name, raw_query)
                .with_context(|| format!("query {query_name:?}"))?;
            queries.insert(query_name.clone(), query);
        }
        Ok(Self {
            name: name.to_string(),
            description,
            queries,
        })
    }

    pub(crate) fn queries(&self) -> impl ExactSizeIterator<Item = &RecipeQueryDefinition> {
        self.queries.values()
    }

    pub(crate) fn get(&self, name: &str) -> Option<&RecipeQueryDefinition> {
        self.queries.get(name)
    }
}

/// One stored, parameterized, read-only Cypher operation.
#[derive(Debug, Clone)]
pub(crate) struct RecipeQueryDefinition {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: ParameterSchema,
    pub(crate) cypher: String,
}

impl RecipeQueryDefinition {
    fn parse(name: &str, raw: &Value) -> Result<Self> {
        let map = object_with_only(raw, QUERY_KEYS, "query")?;
        let description = required_nonempty_string(map, "description")?;
        let cypher_source = required_nonempty_string(map, "cypher")?;
        let raw_parameters = map
            .get("parameters")
            .ok_or_else(|| anyhow::anyhow!("parameters is required"))?;

        let features =
            cypher::query_features(&cypher_source).context("cypher is not a valid KGLite query")?;
        validate_read_only_query(&features)?;

        let parameter_names = cypher::parameter_names(&cypher_source)
            .context("could not collect Cypher parameter names")?;
        let parameters = ParameterSchema::compile_root(raw_parameters, &parameter_names)
            .context("parameters schema is invalid")?;

        Ok(Self {
            name: name.to_string(),
            description,
            parameters,
            cypher: cypher_source,
        })
    }

    pub(crate) fn validate_variables(
        &self,
        variables: &Map<String, Value>,
    ) -> Result<(), VariablesValidationError> {
        self.parameters.validate_variables(variables)
    }
}

fn validate_read_only_query(features: &cypher::QueryFeatures) -> Result<()> {
    if features.explain {
        bail!("EXPLAIN is not allowed in recipe queries");
    }
    if features.profile {
        bail!("PROFILE is not allowed in recipe queries");
    }
    if features.format_csv {
        bail!("FORMAT CSV is not allowed in recipe queries");
    }
    if features.has_load_csv {
        bail!("LOAD CSV is not allowed in recipe queries");
    }
    if features
        .literal_limits
        .contains(&(RECIPE_RESULT_ROW_LIMIT as i64))
    {
        bail!(
            "literal LIMIT {RECIPE_RESULT_ROW_LIMIT} is reserved for the recipe result payload cap; stored queries must not hide overflow from the server"
        );
    }
    if features.is_mutation {
        bail!("cypher must be read-only; mutation clauses are not allowed");
    }
    Ok(())
}

fn object_with_only<'a>(
    raw: &'a Value,
    allowed: &[&str],
    label: &str,
) -> Result<&'a Map<String, Value>> {
    let map = raw
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{label} must be a mapping"))?;
    let allowed: BTreeSet<_> = allowed.iter().copied().collect();
    let unknown: Vec<_> = map
        .keys()
        .filter(|key| !allowed.contains(key.as_str()))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        bail!("unsupported {label} keys: {unknown:?}");
    }
    Ok(map)
}

fn required_nonempty_string(map: &Map<String, Value>, key: &str) -> Result<String> {
    let value = map
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("{key} must be a non-empty string"))?;
    Ok(value.to_string())
}

fn validate_identifier(identifier: &str, label: &str) -> Result<()> {
    let mut chars = identifier.chars();
    let valid_start = chars
        .next()
        .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_');
    if !valid_start || !chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_') {
        bail!("{label} identifier {identifier:?} must match ^[A-Za-z_][A-Za-z0-9_]*$");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use serde_json::json;

    fn query(parameters: Value, cypher: &str) -> Value {
        json!({
            "description": "A stored read operation.",
            "parameters": parameters,
            "cypher": cypher,
        })
    }

    fn catalog(query_value: Value) -> Value {
        json!({
            "code_review": {
                "description": "Code review operations.",
                "queries": {"direct_callers": query_value}
            }
        })
    }

    fn string_parameter() -> Value {
        json!({
            "type": "object",
            "properties": {"qualified_name": {"type": "string"}},
            "required": ["qualified_name"],
            "additionalProperties": false
        })
    }

    #[test]
    fn absent_and_empty_catalogs_are_disabled() {
        let absent = RecipeCatalog::from_manifest_value(None).unwrap();
        assert!(absent.is_empty());
        assert_eq!(absent.discovery_summary(), None);

        let empty = RecipeCatalog::from_manifest_value(Some(&json!({}))).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.discovery_summary(), None);
    }

    #[test]
    fn catalog_is_immutable_and_summarized_deterministically() {
        let raw = catalog(query(
            string_parameter(),
            "MATCH (n:Function) WHERE n.qualified_name = $qualified_name RETURN n.name",
        ));
        let parsed = RecipeCatalog::from_manifest_value(Some(&raw)).unwrap();
        assert_eq!(
            parsed.discovery_summary(),
            Some(CatalogSummary {
                recipe_count: 1,
                query_count: 1
            })
        );
        let recipe = parsed.get("code_review").unwrap();
        assert_eq!(recipe.name, "code_review");
        assert_eq!(recipe.queries().len(), 1);
        assert_eq!(recipe.get("direct_callers").unwrap().name, "direct_callers");
        assert_eq!(parsed.recipes().len(), 1);
    }

    #[test]
    fn rejects_invalid_identifiers_empty_fields_and_unknown_config() {
        let invalid_identifier = json!({
            "bad-name": {
                "description": "Recipe.",
                "queries": {"q": query(json!({
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": false
                }), "RETURN 1")}
            }
        });
        assert!(
            RecipeCatalog::from_manifest_value(Some(&invalid_identifier))
                .unwrap_err()
                .to_string()
                .contains("identifier")
        );

        let empty = json!({"r": {"description": " ", "queries": {}}});
        let error = RecipeCatalog::from_manifest_value(Some(&empty)).unwrap_err();
        assert!(format!("{error:#}").contains("description"));

        let unknown = json!({"r": {"description": "R", "queries": {}, "workflow": []}});
        let error = RecipeCatalog::from_manifest_value(Some(&unknown)).unwrap_err();
        assert!(format!("{error:#}").contains("unsupported recipe keys"));
    }

    #[test]
    fn requires_exact_parameter_property_and_required_sets() {
        let missing_property = catalog(query(
            json!({
                "type": "object", "properties": {}, "required": [],
                "additionalProperties": false
            }),
            "RETURN $qualified_name",
        ));
        let error = RecipeCatalog::from_manifest_value(Some(&missing_property)).unwrap_err();
        assert!(format!("{error:#}").contains("parameter properties"));

        let optional_property = catalog(query(
            json!({
                "type": "object",
                "properties": {"qualified_name": {"type": "string"}},
                "required": [],
                "additionalProperties": false
            }),
            "RETURN $qualified_name",
        ));
        let error = RecipeCatalog::from_manifest_value(Some(&optional_property)).unwrap_err();
        assert!(format!("{error:#}").contains("required must list every"));
    }

    #[test]
    fn tokenizer_ignores_parameter_lookalikes_in_strings_and_comments() {
        let raw = catalog(query(
            string_parameter(),
            "// $comment\nRETURN '$literal' AS text, $qualified_name AS name",
        ));
        RecipeCatalog::from_manifest_value(Some(&raw)).unwrap();
    }

    #[test]
    fn rejects_mutations_and_banned_read_modes() {
        let no_parameters = json!({
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": false
        });
        let cases = [
            ("CREATE (:Thing)", "read-only"),
            ("EXPLAIN RETURN 1", "EXPLAIN"),
            ("PROFILE RETURN 1", "PROFILE"),
            ("RETURN 1 FORMAT CSV", "FORMAT CSV"),
            ("LOAD CSV FROM 'rows.csv' AS row RETURN row", "LOAD CSV"),
        ];
        for (cypher, expected) in cases {
            let raw = catalog(query(no_parameters.clone(), cypher));
            let error = RecipeCatalog::from_manifest_value(Some(&raw)).unwrap_err();
            let chain = format!("{error:#}");
            assert!(chain.contains(expected), "{cypher:?}: {chain}");
        }
    }

    #[test]
    fn third_party_queries_without_order_by_remain_valid() {
        let raw = catalog(query(
            string_parameter(),
            "MATCH (n) WHERE n.name = $qualified_name RETURN n.name",
        ));
        RecipeCatalog::from_manifest_value(Some(&raw)).unwrap();
    }

    #[test]
    fn payload_cap_limit_is_rejected_but_other_semantic_limits_are_valid() {
        let no_parameters = json!({
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": false
        });
        let cap = catalog(query(no_parameters.clone(), "RETURN 1 LIMIT 200"));
        let error = RecipeCatalog::from_manifest_value(Some(&cap)).unwrap_err();
        assert!(format!("{error:#}").contains("reserved for the recipe result payload cap"));

        let semantic = catalog(query(no_parameters, "RETURN 1 LIMIT 20"));
        RecipeCatalog::from_manifest_value(Some(&semantic)).unwrap();
    }

    #[test]
    fn manifest_loader_rejects_duplicate_yaml_recipe_and_query_keys() {
        let cases = [
            r#"
extensions:
  cypher_recipes:
    review: {description: First, queries: {}}
    review: {description: Second, queries: {}}
"#,
            r#"
extensions:
  cypher_recipes:
    review:
      description: Review.
      queries:
        lookup: {description: First, parameters: {}, cypher: RETURN 1}
        lookup: {description: Second, parameters: {}, cypher: RETURN 2}
"#,
        ];
        for yaml in cases {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            file.write_all(yaml.as_bytes()).unwrap();
            let error = mcp_methods::server::load_manifest(file.path()).unwrap_err();
            assert!(error.to_string().contains("duplicate entry"), "{error}");
        }
    }
}
