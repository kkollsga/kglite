//! Cypher executor — where_clause methods.

use super::super::ast::*;
use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::algorithms::vector as vs;
use crate::graph::core::pattern_matching::{MatchBinding, PatternExecutor, PatternMatch};
use crate::graph::storage::GraphRead;
use std::collections::HashSet;
use std::sync::Arc;

impl<'a> CypherExecutor<'a> {
    pub(super) fn bindings_compatible(&self, row: &ResultRow, m: &PatternMatch) -> bool {
        for (var, binding) in &m.bindings {
            if let Some(&existing_idx) = row.node_bindings.get(var) {
                // Variable already bound - check it matches
                match binding {
                    MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                        if *index != existing_idx {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    // ========================================================================
    // WHERE
    // ========================================================================

    pub(super) fn execute_where(
        &self,
        clause: &WhereClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Try index-accelerated filtering for simple equality predicates
        let index_filters = self.extract_indexable_predicates(&clause.predicate);
        for (variable, property, value) in &index_filters {
            if let Some(node_type) = self.infer_node_type(variable, &result_set) {
                if let Some(matching_indices) =
                    self.graph.lookup_by_index(&node_type, property, value)
                {
                    let index_set: HashSet<petgraph::graph::NodeIndex> =
                        matching_indices.into_iter().collect();
                    result_set.rows.retain(|row| {
                        row.node_bindings
                            .get(variable.as_str())
                            .is_some_and(|idx| index_set.contains(idx))
                    });
                }
            }
        }

        // Try index-accelerated filtering for IN predicates
        let in_filters = Self::extract_in_indexable_predicates(&clause.predicate);
        for (variable, property, values) in &in_filters {
            if let Some(node_type) = self.infer_node_type(variable, &result_set) {
                // Collect matching node indices from all IN values
                let mut index_set: HashSet<petgraph::graph::NodeIndex> = HashSet::new();
                let mut any_indexed = false;
                for val in values {
                    if let Some(matching_indices) =
                        self.graph.lookup_by_index(&node_type, property, val)
                    {
                        any_indexed = true;
                        index_set.extend(matching_indices);
                    }
                }
                if any_indexed {
                    result_set.rows.retain(|row| {
                        row.node_bindings
                            .get(variable.as_str())
                            .is_some_and(|idx| index_set.contains(idx))
                    });
                }
            }
        }

        // Fold constant sub-expressions once before row iteration
        let folded_pred = self.fold_constants_pred(&clause.predicate);

        // Fast path: spatial contains() filter bypasses expression evaluator
        if let Some((spec, remainder)) = Self::try_extract_contains_filter(&folded_pred) {
            result_set.rows.retain(|row| {
                // Get container geometry from spatial cache
                let container_idx = match row.node_bindings.get(&spec.container_variable) {
                    Some(&idx) => idx,
                    None => return false,
                };
                self.ensure_node_spatial_cached(container_idx);
                // Scope read lock: clone Arc + bbox, then drop lock
                let container = {
                    let cache = self.spatial_node_cache.read().unwrap();
                    cache
                        .get(&container_idx.index())
                        .and_then(|opt| opt.as_ref())
                        .and_then(|data| data.geometry.as_ref())
                        .map(|(g, bb)| (Arc::clone(g), *bb))
                };
                let (geom, bbox) = match container {
                    Some((g, bb)) => (g, bb),
                    None => return false,
                };

                // Get contained side: a Point (location | constant) or
                // — 0.9.0 §4 — a Geometry (when the contained variable
                // is a polygon-bearing node with no Location). The
                // pre-§4 fast path silently returned false for the
                // polygon-vs-polygon case, masking outer-contains-inner
                // matches.
                #[derive(Clone)]
                enum ContainedSide {
                    Point(f64, f64),
                    Geom(Arc<geo::Geometry<f64>>, Option<geo::Rect<f64>>),
                }
                let contained = match &spec.contained {
                    ContainsTarget::ConstantPoint(lat, lon) => ContainedSide::Point(*lat, *lon),
                    ContainsTarget::Variable { name } => {
                        let contained_idx = match row.node_bindings.get(name) {
                            Some(&idx) => idx,
                            None => return false,
                        };
                        self.ensure_node_spatial_cached(contained_idx);
                        let cache = self.spatial_node_cache.read().unwrap();
                        let resolved = cache
                            .get(&contained_idx.index())
                            .and_then(|opt| opt.as_ref())
                            .and_then(|data| {
                                if let Some((lat, lon)) = data.location {
                                    Some(ContainedSide::Point(lat, lon))
                                } else {
                                    data.geometry
                                        .as_ref()
                                        .map(|(g, bb)| ContainedSide::Geom(Arc::clone(g), *bb))
                                }
                            });
                        match resolved {
                            Some(c) => c,
                            None => return false,
                        }
                    }
                };

                let result = match &contained {
                    ContainedSide::Point(lat, lon) => {
                        // Bbox pre-filter
                        if let Some(bb) = bbox {
                            if *lon < bb.min().x
                                || *lon > bb.max().x
                                || *lat < bb.min().y
                                || *lat > bb.max().y
                            {
                                return spec.negated;
                            }
                        }
                        let pt = geo::Point::new(*lon, *lat);
                        crate::graph::features::spatial::geometry_contains_point(&geom, &pt)
                    }
                    ContainedSide::Geom(g2, bbox2) => {
                        // Bbox pre-filter for geom-vs-geom
                        if let (Some(bb1), Some(bb2)) = (bbox, *bbox2) {
                            if bb1.max().x < bb2.min().x
                                || bb2.max().x < bb1.min().x
                                || bb1.max().y < bb2.min().y
                                || bb2.max().y < bb1.min().y
                            {
                                return spec.negated;
                            }
                        }
                        crate::graph::features::spatial::geometry_contains_geometry(&geom, g2)
                    }
                };
                if spec.negated {
                    !result
                } else {
                    result
                }
            });
            self.check_deadline()?;
            if let Some(rest) = remainder {
                let mut keep = Vec::with_capacity(result_set.rows.len());
                for row in result_set.rows {
                    match self.evaluate_predicate(rest, &row) {
                        Ok(true) => keep.push(row),
                        Ok(false) => {}
                        Err(e) => return Err(e),
                    }
                }
                result_set.rows = keep;
            }
            return Ok(result_set);
        }

        // Fast path: specialized distance filter bypasses expression evaluator
        if let Some((spec, remainder)) = Self::try_extract_distance_filter(&folded_pred) {
            let graph = self.graph;
            result_set.rows.retain(|row| {
                let idx = match row.node_bindings.get(&spec.variable) {
                    Some(&idx) => idx,
                    None => return false,
                };
                let node = match graph.graph.node_weight(idx) {
                    Some(n) => n,
                    None => return false,
                };
                let lat = match node
                    .get_property(&spec.lat_prop)
                    .as_deref()
                    .and_then(crate::graph::core::value_operations::value_to_f64)
                {
                    Some(v) => v,
                    None => return false,
                };
                let lon = match node
                    .get_property(&spec.lon_prop)
                    .as_deref()
                    .and_then(crate::graph::core::value_operations::value_to_f64)
                {
                    Some(v) => v,
                    None => return false,
                };
                let dist = crate::graph::features::spatial::geodesic_distance(
                    lat,
                    lon,
                    spec.center_lat,
                    spec.center_lon,
                );
                if spec.less_than {
                    if spec.inclusive {
                        dist <= spec.threshold
                    } else {
                        dist < spec.threshold
                    }
                } else if spec.inclusive {
                    dist >= spec.threshold
                } else {
                    dist > spec.threshold
                }
            });
            self.check_deadline()?;
            // Apply remainder predicate if there were additional AND conditions
            if let Some(rest) = remainder {
                let mut keep = Vec::with_capacity(result_set.rows.len());
                for row in result_set.rows {
                    match self.evaluate_predicate(rest, &row) {
                        Ok(true) => keep.push(row),
                        Ok(false) => {}
                        Err(e) => return Err(e),
                    }
                }
                result_set.rows = keep;
            }
            return Ok(result_set);
        }

        // Fast path: specialized vector_score filter bypasses expression evaluator
        if let Some((spec, remainder)) = self.try_extract_vector_score_filter(&folded_pred) {
            let graph = self.graph;
            result_set.rows.retain(|row| {
                let idx = match row.node_bindings.get(&spec.variable) {
                    Some(&idx) => idx,
                    None => return false,
                };
                let node_type = match graph.graph.node_weight(idx) {
                    Some(n) => n.node_type_str(&graph.interner),
                    None => return false,
                };
                let store = match graph.embedding_store(node_type, &spec.prop_name) {
                    Some(s) => s,
                    None => return false,
                };
                let embedding = match store.get_embedding(idx.index()) {
                    Some(e) => e,
                    None => return false,
                };
                let score = (spec.similarity_fn)(&spec.query_vec, embedding) as f64;
                if spec.greater_than {
                    if spec.inclusive {
                        score >= spec.threshold
                    } else {
                        score > spec.threshold
                    }
                } else if spec.inclusive {
                    score <= spec.threshold
                } else {
                    score < spec.threshold
                }
            });
            self.check_deadline()?;
            if let Some(rest) = remainder {
                let mut keep = Vec::with_capacity(result_set.rows.len());
                for row in result_set.rows {
                    match self.evaluate_predicate(rest, &row) {
                        Ok(true) => keep.push(row),
                        Ok(false) => {}
                        Err(e) => return Err(e),
                    }
                }
                result_set.rows = keep;
            }
            return Ok(result_set);
        }

        // Apply full predicate evaluation for remaining/non-indexable conditions.
        self.check_deadline()?;

        let mut filtered_rows = Vec::new();
        for row in result_set.rows {
            match self.evaluate_predicate(&folded_pred, &row) {
                Ok(true) => filtered_rows.push(row),
                Ok(false) => {}
                Err(e) => return Err(e),
            }
        }
        result_set.rows = filtered_rows;
        Ok(result_set)
    }

    /// Extract simple equality predicates (variable.property = literal) from AND-trees.
    pub(super) fn extract_indexable_predicates(
        &self,
        predicate: &Predicate,
    ) -> Vec<(String, String, Value)> {
        let mut results = Vec::new();
        Self::collect_indexable(predicate, &mut results);
        results
    }

    /// Extract IN predicates (variable.property IN [literals]) from AND-trees.
    pub(super) fn extract_in_indexable_predicates(
        predicate: &Predicate,
    ) -> Vec<(String, String, Vec<Value>)> {
        let mut results = Vec::new();
        Self::collect_in_indexable(predicate, &mut results);
        results
    }

    pub(super) fn collect_indexable(
        predicate: &Predicate,
        results: &mut Vec<(String, String, Value)>,
    ) {
        match predicate {
            Predicate::Comparison {
                left,
                operator,
                right,
            } if *operator == ComparisonOp::Equals => {
                if let (
                    Expression::PropertyAccess { variable, property },
                    Expression::Literal(value),
                ) = (left, right)
                {
                    results.push((variable.clone(), property.clone(), value.clone()));
                } else if let (
                    Expression::Literal(value),
                    Expression::PropertyAccess { variable, property },
                ) = (left, right)
                {
                    results.push((variable.clone(), property.clone(), value.clone()));
                }
            }
            Predicate::And(left, right) => {
                Self::collect_indexable(left, results);
                Self::collect_indexable(right, results);
            }
            _ => {}
        }
    }

    pub(super) fn collect_in_indexable(
        predicate: &Predicate,
        results: &mut Vec<(String, String, Vec<Value>)>,
    ) {
        match predicate {
            Predicate::In {
                expr: Expression::PropertyAccess { variable, property },
                list,
            } => {
                let all_literal: Option<Vec<Value>> = list
                    .iter()
                    .map(|item| {
                        if let Expression::Literal(v) = item {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                if let Some(values) = all_literal {
                    results.push((variable.clone(), property.clone(), values));
                }
            }
            Predicate::InLiteralSet {
                expr: Expression::PropertyAccess { variable, property },
                values,
            } => {
                results.push((
                    variable.clone(),
                    property.clone(),
                    values.iter().cloned().collect(),
                ));
            }
            Predicate::And(left, right) => {
                Self::collect_in_indexable(left, results);
                Self::collect_in_indexable(right, results);
            }
            _ => {}
        }
    }

    /// Infer the node type for a variable by checking the first row's binding.
    pub(super) fn infer_node_type(&self, variable: &str, result_set: &ResultSet) -> Option<String> {
        result_set.rows.iter().find_map(|row| {
            row.node_bindings
                .get(variable)
                .and_then(|&idx| self.graph.graph.node_weight(idx))
                .map(|node| node.node_type_str(&self.graph.interner).to_string())
        })
    }

    /// Evaluate a predicate in boolean (WHERE-row-keep) terms.
    ///
    /// External callers (HAVING, OPTIONAL MATCH filter, list comprehensions,
    /// spatial joins) only care whether to keep the row. This wrapper
    /// collapses three-valued NULL semantics: `Some(true)` → `true`,
    /// `Some(false)` and `None` → `false`. That preserves the historical
    /// "row drops on false" contract at every callsite while letting the
    /// internal tristate machinery enforce correct NULL propagation
    /// through `NOT` / `AND` / `OR` / `XOR` / comparison operators and
    /// string predicates (B1, B2). See `evaluate_predicate_tristate`.
    pub(super) fn evaluate_predicate(
        &self,
        pred: &Predicate,
        row: &ResultRow,
    ) -> Result<bool, String> {
        Ok(self.evaluate_predicate_tristate(pred, row)? == Some(true))
    }

    /// Three-valued predicate evaluator implementing openCypher NULL
    /// semantics:
    ///
    /// - Comparison operators (`=`, `<>`, `<`, `<=`, `>`, `>=`) with any
    ///   NULL operand → `None`. Fixes B1: `WHERE x <> 'lit'` no longer
    ///   keeps rows where `x` is missing.
    /// - String predicates (`STARTS WITH`, `ENDS WITH`, `CONTAINS`) with
    ///   NULL operand → `None`. Combined with the NULL-aware `Not` arm,
    ///   fixes B2: `WHERE NOT (x CONTAINS 'y')` no longer keeps
    ///   rows where `x` is missing.
    /// - `AND` / `OR` follow Kleene three-valued logic with short-circuit
    ///   on the absorbing element.
    /// - `XOR` is `None` if either side is `None`.
    /// - `NOT None` is `None`; `NOT Some(b)` is `Some(!b)`.
    ///
    /// `IN`/`InExpression`/`InLiteralSet` keep their current boolean
    /// behaviour for now. Strict openCypher says they should return
    /// `None` when the LHS is NULL or when the list contains NULL and no
    /// match is found — both are deferred (no concrete consumer broken
    /// today, and the fix is symmetric to this one but lives in three
    /// arms).
    fn evaluate_predicate_tristate(
        &self,
        pred: &Predicate,
        row: &ResultRow,
    ) -> Result<Option<bool>, String> {
        match pred {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => {
                let left_val = self.evaluate_expression(left, row)?;
                let right_val = self.evaluate_expression(right, row)?;
                if matches!(left_val, Value::Null) || matches!(right_val, Value::Null) {
                    return Ok(None);
                }
                evaluate_comparison(&left_val, operator, &right_val, Some(&self.regex_cache))
                    .map(Some)
            }
            Predicate::And(left, right) => {
                // Kleene AND: FALSE absorbs (short-circuits even past NULL);
                // NULL propagates only when no FALSE is present.
                let lv = self.evaluate_predicate_tristate(left, row)?;
                if lv == Some(false) {
                    return Ok(Some(false));
                }
                let rv = self.evaluate_predicate_tristate(right, row)?;
                if rv == Some(false) {
                    return Ok(Some(false));
                }
                if lv.is_none() || rv.is_none() {
                    return Ok(None);
                }
                Ok(Some(true))
            }
            Predicate::Or(left, right) => {
                // Kleene OR: TRUE absorbs; NULL propagates only when no
                // TRUE is present.
                let lv = self.evaluate_predicate_tristate(left, row)?;
                if lv == Some(true) {
                    return Ok(Some(true));
                }
                let rv = self.evaluate_predicate_tristate(right, row)?;
                if rv == Some(true) {
                    return Ok(Some(true));
                }
                if lv.is_none() || rv.is_none() {
                    return Ok(None);
                }
                Ok(Some(false))
            }
            Predicate::Xor(left, right) => {
                let lv = self.evaluate_predicate_tristate(left, row)?;
                let rv = self.evaluate_predicate_tristate(right, row)?;
                match (lv, rv) {
                    (Some(a), Some(b)) => Ok(Some(a ^ b)),
                    _ => Ok(None),
                }
            }
            Predicate::Not(inner) => {
                Ok(self.evaluate_predicate_tristate(inner, row)?.map(|b| !b))
            }
            Predicate::LabelCheck { variable, label } => {
                // True iff the variable is bound to a node whose type matches `label`.
                // Unbound (OPTIONAL MATCH) or non-node bindings are false.
                if let Some(&idx) = row.node_bindings.get(variable) {
                    if let Some(node) = self.graph.graph.node_weight(idx) {
                        return Ok(Some(node.node_type_str(&self.graph.interner) == label));
                    }
                }
                Ok(Some(false))
            }
            Predicate::IsNull(expr) => {
                let val = self.evaluate_expression(expr, row)?;
                Ok(Some(matches!(val, Value::Null)))
            }
            Predicate::IsNotNull(expr) => {
                let val = self.evaluate_expression(expr, row)?;
                Ok(Some(!matches!(val, Value::Null)))
            }
            Predicate::In { expr, list } => {
                // Deferred: strict openCypher returns NULL when LHS is NULL
                // or when no match is found and the list contains NULL.
                let val = self.evaluate_expression(expr, row)?;
                for item in list {
                    let item_val = self.evaluate_expression(item, row)?;
                    if crate::graph::core::filtering::values_equal(&val, &item_val) {
                        return Ok(Some(true));
                    }
                }
                Ok(Some(false))
            }
            Predicate::InLiteralSet { expr, values } => {
                let val = self.evaluate_expression(expr, row)?;
                // Try fast HashSet lookup first, fall back to cross-type comparison
                Ok(Some(
                    values.contains(&val)
                        || values
                            .iter()
                            .any(|v| crate::graph::core::filtering::values_equal(v, &val)),
                ))
            }
            Predicate::StartsWith { expr, pattern } => {
                let val = self.evaluate_expression(expr, row)?;
                let pat = self.evaluate_expression(pattern, row)?;
                if matches!(val, Value::Null) || matches!(pat, Value::Null) {
                    return Ok(None);
                }
                match (&val, &pat) {
                    (Value::String(s), Value::String(p)) => Ok(Some(s.starts_with(p.as_str()))),
                    _ => Ok(Some(false)),
                }
            }
            Predicate::EndsWith { expr, pattern } => {
                let val = self.evaluate_expression(expr, row)?;
                let pat = self.evaluate_expression(pattern, row)?;
                if matches!(val, Value::Null) || matches!(pat, Value::Null) {
                    return Ok(None);
                }
                match (&val, &pat) {
                    (Value::String(s), Value::String(p)) => Ok(Some(s.ends_with(p.as_str()))),
                    _ => Ok(Some(false)),
                }
            }
            Predicate::Contains { expr, pattern } => {
                let val = self.evaluate_expression(expr, row)?;
                let pat = self.evaluate_expression(pattern, row)?;
                if matches!(val, Value::Null) || matches!(pat, Value::Null) {
                    return Ok(None);
                }
                match (&val, &pat) {
                    (Value::String(s), Value::String(p)) => Ok(Some(s.contains(p.as_str()))),
                    _ => Ok(Some(false)),
                }
            }
            Predicate::Exists {
                patterns,
                where_clause,
            } => {
                // Fast path: single 3-element pattern with one bound node
                // — check edge existence directly without PatternExecutor
                if let Some(result) = self.try_fast_exists_check(patterns, where_clause, row) {
                    return result.map(Some);
                }

                // Slow path: full pattern execution for complex EXISTS.
                //
                // Multi-pattern subqueries (`EXISTS { MATCH ... MATCH ... [WHERE ...] }`)
                // share variables across patterns, so we accumulate bindings
                // progressively. Each pattern intersects with the running
                // `combined_rows` set; a pattern that produces zero compatible
                // matches short-circuits the whole subquery to false.
                //
                // The WHERE predicate is evaluated *once*, against the fully
                // merged bindings, after all patterns have matched. Evaluating
                // it per-pattern (the previous behaviour) breaks subqueries
                // where the predicate references a variable bound in a later
                // MATCH — `prod` in `MATCH ... MATCH (prod) WHERE prod.price > X`
                // wouldn't be in scope when the first MATCH's results were
                // checked.
                let mut combined_rows: Vec<ResultRow> = vec![row.clone()];
                for pattern in patterns {
                    if combined_rows.is_empty() {
                        return Ok(Some(false));
                    }
                    // Resolve EqualsVar references against current row
                    let resolved;
                    let pat = if Self::pattern_has_vars(pattern) {
                        resolved = self.resolve_pattern_vars(pattern, row);
                        &resolved
                    } else {
                        pattern
                    };
                    let executor = PatternExecutor::with_bindings_and_params(
                        self.graph,
                        None,
                        &row.node_bindings,
                        self.params,
                    )
                    .set_deadline(self.deadline);
                    let matches = executor.execute(pat)?;

                    let mut next_rows: Vec<ResultRow> = Vec::new();
                    for current in &combined_rows {
                        for m in &matches {
                            if !self.bindings_compatible(current, m) {
                                continue;
                            }
                            let mut merged = current.clone();
                            self.merge_match_into_row(&mut merged, m);
                            next_rows.push(merged);
                        }
                    }
                    combined_rows = next_rows;
                }

                if combined_rows.is_empty() {
                    return Ok(Some(false));
                }
                if let Some(ref where_pred) = where_clause {
                    Ok(Some(combined_rows.iter().any(|r| {
                        // EXISTS treats a NULL inner predicate as "no match"
                        // — same as `false` — to keep with Cypher's "exists
                        // a row that satisfies" semantics. Strict tristate
                        // is preserved at the outer boundary, not here.
                        matches!(self.evaluate_predicate_tristate(where_pred, r), Ok(Some(true)))
                    })))
                } else {
                    Ok(Some(true))
                }
            }
            Predicate::InExpression { expr, list_expr } => {
                // Deferred: NULL LHS / NULL-bearing list should propagate NULL.
                let val = self.evaluate_expression(expr, row)?;
                let list_val = self.evaluate_expression(list_expr, row)?;
                let items = parse_list_value(&list_val);
                for item in &items {
                    if crate::graph::core::filtering::values_equal(&val, item) {
                        return Ok(Some(true));
                    }
                }
                Ok(Some(false))
            }
        }
    }

    // ========================================================================
    // Specialized Distance Filter (Fast Path)
    // ========================================================================

    /// Try to extract a distance filter from a (folded) predicate.
    /// Returns (spec, optional remainder predicate for other AND conditions).
    /// Try to extract a `vector_score(n, prop, vec [, metric]) {>|>=|<|<=} threshold`
    /// pattern from a (folded) predicate. Returns the spec and optional remainder.
    pub(super) fn try_extract_vector_score_filter<'p>(
        &self,
        pred: &'p Predicate,
    ) -> Option<(VectorScoreFilterSpec, Option<&'p Predicate>)> {
        match pred {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => {
                // Determine which side has vector_score and which has the threshold
                let (vs_expr, threshold_expr, greater_than, inclusive) = match operator {
                    ComparisonOp::GreaterThan => (left, right, true, false),
                    ComparisonOp::GreaterThanEq => (left, right, true, true),
                    ComparisonOp::LessThan => (left, right, false, false),
                    ComparisonOp::LessThanEq => (left, right, false, true),
                    _ => return None,
                };

                // Try vs_expr as vector_score, threshold_expr as literal
                if let Some(spec) =
                    self.extract_vector_score_spec(vs_expr, threshold_expr, greater_than, inclusive)
                {
                    return Some((spec, None));
                }

                // Try flipped: threshold_expr as vector_score, vs_expr as literal
                // Flip comparison direction
                if let Some(spec) = self.extract_vector_score_spec(
                    threshold_expr,
                    vs_expr,
                    !greater_than,
                    inclusive,
                ) {
                    return Some((spec, None));
                }

                None
            }
            Predicate::And(left, right) => {
                if let Some((spec, None)) = self.try_extract_vector_score_filter(left) {
                    return Some((spec, Some(right)));
                }
                if let Some((spec, None)) = self.try_extract_vector_score_filter(right) {
                    return Some((spec, Some(left)));
                }
                None
            }
            _ => None,
        }
    }

    /// Extract a VectorScoreFilterSpec from a vector_score() function call + threshold.
    pub(super) fn extract_vector_score_spec(
        &self,
        func_expr: &Expression,
        threshold_expr: &Expression,
        greater_than: bool,
        inclusive: bool,
    ) -> Option<VectorScoreFilterSpec> {
        // func_expr must be vector_score(variable, prop, query_vec [, metric])
        let (name, args) = match func_expr {
            Expression::FunctionCall { name, args, .. } => (name, args),
            _ => return None,
        };
        if name != "vector_score" || args.len() < 3 || args.len() > 4 {
            return None;
        }

        // threshold must be a literal number
        let threshold = match threshold_expr {
            Expression::Literal(val) => crate::graph::core::value_operations::value_to_f64(val)?,
            _ => return None,
        };

        // Arg 0: must be a variable
        let variable = match &args[0] {
            Expression::Variable(v) => v.clone(),
            _ => return None,
        };

        // Arg 1: prop name (should be folded to literal string)
        let prop_name = match &args[1] {
            Expression::Literal(Value::String(s)) => s.clone(),
            _ => return None,
        };

        // Arg 2: query vector (should be folded to literal)
        let query_vec = match &args[2] {
            Expression::Literal(Value::String(s)) => parse_json_float_list(s).ok()?,
            Expression::ListLiteral(items) => {
                let mut vec = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Expression::Literal(Value::Float64(f)) => vec.push(*f as f32),
                        Expression::Literal(Value::Int64(i)) => vec.push(*i as f32),
                        _ => return None,
                    }
                }
                vec
            }
            _ => return None,
        };

        // Arg 3: optional metric (default cosine)
        let similarity_fn = if args.len() > 3 {
            match &args[3] {
                Expression::Literal(Value::String(s)) => match s.as_str() {
                    "cosine" => vs::cosine_similarity as fn(&[f32], &[f32]) -> f32,
                    "dot_product" => vs::dot_product,
                    "euclidean" => vs::neg_euclidean_distance,
                    _ => return None,
                },
                _ => vs::cosine_similarity,
            }
        } else {
            vs::cosine_similarity
        };

        Some(VectorScoreFilterSpec {
            variable,
            prop_name,
            query_vec,
            similarity_fn,
            threshold,
            greater_than,
            inclusive,
        })
    }

    pub(super) fn try_extract_distance_filter(
        pred: &Predicate,
    ) -> Option<(DistanceFilterSpec, Option<&Predicate>)> {
        match pred {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => {
                // distance(...) < threshold  or  threshold > distance(...)
                let (dist_expr, threshold_expr, less_than, inclusive) = match operator {
                    ComparisonOp::LessThan => (left, right, true, false),
                    ComparisonOp::LessThanEq => (left, right, true, true),
                    ComparisonOp::GreaterThan => (right, left, true, false),
                    ComparisonOp::GreaterThanEq => (right, left, true, true),
                    _ => return None,
                };

                // threshold must be a literal number
                let threshold = match threshold_expr {
                    Expression::Literal(val) => {
                        crate::graph::core::value_operations::value_to_f64(val)?
                    }
                    _ => return None,
                };

                // dist_expr must be distance(...)
                let spec = Self::extract_distance_call(dist_expr, threshold, less_than, inclusive)?;
                Some((spec, None))
            }
            Predicate::And(left, right) => {
                // Try extracting from left side
                if let Some((spec, None)) = Self::try_extract_distance_filter(left) {
                    return Some((spec, Some(right)));
                }
                // Try extracting from right side
                if let Some((spec, None)) = Self::try_extract_distance_filter(right) {
                    return Some((spec, Some(left)));
                }
                None
            }
            _ => None,
        }
    }

    /// Extract a DistanceFilterSpec from a `distance(...)` function call expression.
    pub(super) fn extract_distance_call(
        expr: &Expression,
        threshold: f64,
        less_than: bool,
        inclusive: bool,
    ) -> Option<DistanceFilterSpec> {
        if let Expression::FunctionCall { name, args, .. } = expr {
            if name != "distance" {
                return None;
            }
            match args.len() {
                // 2-arg: distance(point(n.lat, n.lon), point(C1, C2))
                2 => {
                    let (var, lat_prop, lon_prop) = Self::extract_point_var_props(&args[0])?;
                    let (center_lat, center_lon) = Self::extract_point_constants(&args[1])?;
                    Some(DistanceFilterSpec {
                        variable: var,
                        lat_prop,
                        lon_prop,
                        center_lat,
                        center_lon,
                        threshold,
                        less_than,
                        inclusive,
                    })
                }
                // 4-arg: distance(n.lat, n.lon, C1, C2)
                4 => {
                    let (var1, lat_prop) = Self::extract_prop_access(&args[0])?;
                    let (var2, lon_prop) = Self::extract_prop_access(&args[1])?;
                    if var1 != var2 {
                        return None;
                    }
                    let center_lat = Self::extract_literal_f64(&args[2])?;
                    let center_lon = Self::extract_literal_f64(&args[3])?;
                    Some(DistanceFilterSpec {
                        variable: var1,
                        lat_prop,
                        lon_prop,
                        center_lat,
                        center_lon,
                        threshold,
                        less_than,
                        inclusive,
                    })
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Extract (variable, lat_prop, lon_prop) from point(n.lat, n.lon)
    pub(super) fn extract_point_var_props(expr: &Expression) -> Option<(String, String, String)> {
        if let Expression::FunctionCall { name, args, .. } = expr {
            if name != "point" || args.len() != 2 {
                return None;
            }
            let (var1, lat_prop) = Self::extract_prop_access(&args[0])?;
            let (var2, lon_prop) = Self::extract_prop_access(&args[1])?;
            if var1 != var2 {
                return None;
            }
            Some((var1, lat_prop, lon_prop))
        } else {
            None
        }
    }

    /// Extract (center_lat, center_lon) from point(Literal, Literal)
    /// or from a folded Literal(Point{lat, lon}).
    pub(super) fn extract_point_constants(expr: &Expression) -> Option<(f64, f64)> {
        // After constant folding, point(59.91, 10.75) becomes Literal(Point{lat, lon})
        if let Expression::Literal(Value::Point { lat, lon }) = expr {
            return Some((*lat, *lon));
        }
        if let Expression::FunctionCall { name, args, .. } = expr {
            if name != "point" || args.len() != 2 {
                return None;
            }
            let lat = Self::extract_literal_f64(&args[0])?;
            let lon = Self::extract_literal_f64(&args[1])?;
            Some((lat, lon))
        } else {
            None
        }
    }

    /// Extract (variable, property) from PropertyAccess
    pub(super) fn extract_prop_access(expr: &Expression) -> Option<(String, String)> {
        if let Expression::PropertyAccess { variable, property } = expr {
            Some((variable.clone(), property.clone()))
        } else {
            None
        }
    }

    /// Extract f64 from a Literal expression
    pub(super) fn extract_literal_f64(expr: &Expression) -> Option<f64> {
        if let Expression::Literal(val) = expr {
            crate::graph::core::value_operations::value_to_f64(val)
        } else {
            None
        }
    }

    // ========================================================================
    // Contains Filter Extraction
    // ========================================================================

    /// Try to extract a contains() fast-path spec from a WHERE predicate.
    /// Matches patterns like: contains(a, point(C1, C2)) or contains(a, b)
    pub(super) fn try_extract_contains_filter(
        pred: &Predicate,
    ) -> Option<(ContainsFilterSpec, Option<&Predicate>)> {
        match pred {
            // contains(a, b) <> false  — the parser's truthy wrapper
            Predicate::Comparison {
                left,
                operator: ComparisonOp::NotEquals,
                right: Expression::Literal(Value::Boolean(false)),
            } => {
                let spec = Self::extract_contains_call(left, false)?;
                Some((spec, None))
            }
            // NOT contains(a, b) — negated
            Predicate::Not(inner) => {
                if let Some((mut spec, None)) = Self::try_extract_contains_filter(inner) {
                    spec.negated = !spec.negated;
                    Some((spec, None))
                } else {
                    None
                }
            }
            // AND extraction
            Predicate::And(left, right) => {
                if let Some((spec, None)) = Self::try_extract_contains_filter(left) {
                    return Some((spec, Some(right)));
                }
                if let Some((spec, None)) = Self::try_extract_contains_filter(right) {
                    return Some((spec, Some(left)));
                }
                None
            }
            _ => None,
        }
    }

    /// Extract a ContainsFilterSpec from a contains() function call expression.
    pub(super) fn extract_contains_call(
        expr: &Expression,
        negated: bool,
    ) -> Option<ContainsFilterSpec> {
        if let Expression::FunctionCall { name, args, .. } = expr {
            if name != "contains" || args.len() != 2 {
                return None;
            }
            // Arg 1: must be a bare Variable (node with geometry config)
            let container_variable = match &args[0] {
                Expression::Variable(name) => name.clone(),
                _ => return None,
            };
            // Arg 2: constant point or variable
            let contained = match &args[1] {
                // Folded point literal: point(59.91, 10.75) → Literal(Point{...})
                Expression::Literal(Value::Point { lat, lon }) => {
                    ContainsTarget::ConstantPoint(*lat, *lon)
                }
                // Unfolded point with constant args
                Expression::FunctionCall {
                    name: pname,
                    args: pargs,
                    ..
                } if pname == "point" && pargs.len() == 2 => {
                    let lat = Self::extract_literal_f64(&pargs[0])?;
                    let lon = Self::extract_literal_f64(&pargs[1])?;
                    ContainsTarget::ConstantPoint(lat, lon)
                }
                // Variable: contains(a, b)
                Expression::Variable(name) => ContainsTarget::Variable { name: name.clone() },
                _ => return None,
            };
            Some(ContainsFilterSpec {
                container_variable,
                contained,
                negated,
            })
        } else {
            None
        }
    }

    // ========================================================================
    // Constant Expression Folding
    // ========================================================================

    /// Check if an expression can be evaluated without any row bindings
    /// (i.e., it contains no PropertyAccess, Variable, Star, or aggregate references).
    pub(super) fn is_row_independent(expr: &Expression) -> bool {
        match expr {
            Expression::Literal(_) | Expression::Parameter(_) => true,
            Expression::PropertyAccess { .. } | Expression::Variable(_) | Expression::Star => false,
            Expression::FunctionCall { name, args, .. } => {
                // Aggregates depend on row groups, not individual rows
                if is_aggregate_expression(expr) {
                    return false;
                }
                // Non-deterministic functions must be evaluated per-row even
                // when all args are constants — otherwise constant folding
                // collapses them to a single value for the whole query.
                if matches!(name.as_str(), "rand" | "random" | "randomuuid") {
                    return false;
                }
                args.iter().all(Self::is_row_independent)
            }
            Expression::Add(l, r)
            | Expression::Subtract(l, r)
            | Expression::Multiply(l, r)
            | Expression::Divide(l, r)
            | Expression::Modulo(l, r)
            | Expression::Concat(l, r) => {
                Self::is_row_independent(l) && Self::is_row_independent(r)
            }
            Expression::Negate(inner) => Self::is_row_independent(inner),
            Expression::ListLiteral(items) => items.iter().all(Self::is_row_independent),
            // Conservative: skip complex expressions
            Expression::Case { .. }
            | Expression::ListComprehension { .. }
            | Expression::IndexAccess { .. }
            | Expression::ListSlice { .. }
            | Expression::MapProjection { .. }
            | Expression::MapLiteral(_)
            | Expression::IsNull(_)
            | Expression::IsNotNull(_)
            | Expression::QuantifiedList { .. }
            | Expression::WindowFunction { .. }
            | Expression::PredicateExpr(_)
            | Expression::ExprPropertyAccess { .. }
            | Expression::CountSubquery { .. }
            | Expression::Reduce { .. } => false,
        }
    }

    /// Fold constant sub-expressions in an expression tree into Literal values.
    /// Returns a new expression with all row-independent sub-trees pre-evaluated.
    pub(crate) fn fold_constants_expr(&self, expr: &Expression) -> Expression {
        // Already a literal — nothing to fold
        if matches!(expr, Expression::Literal(_)) {
            return expr.clone();
        }
        // If the whole expression is row-independent, evaluate it once
        if Self::is_row_independent(expr) {
            let dummy = ResultRow::new();
            if let Ok(val) = self.evaluate_expression(expr, &dummy) {
                return Expression::Literal(val);
            }
            // If evaluation fails (e.g., missing parameter), keep original
            return expr.clone();
        }
        // Recursively fold children
        match expr {
            Expression::FunctionCall {
                name,
                args,
                distinct,
            } => Expression::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|a| self.fold_constants_expr(a)).collect(),
                distinct: *distinct,
            },
            Expression::Add(l, r) => Expression::Add(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Subtract(l, r) => Expression::Subtract(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Multiply(l, r) => Expression::Multiply(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Divide(l, r) => Expression::Divide(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Modulo(l, r) => Expression::Modulo(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Concat(l, r) => Expression::Concat(
                Box::new(self.fold_constants_expr(l)),
                Box::new(self.fold_constants_expr(r)),
            ),
            Expression::Negate(inner) => {
                Expression::Negate(Box::new(self.fold_constants_expr(inner)))
            }
            Expression::ListLiteral(items) => {
                Expression::ListLiteral(items.iter().map(|i| self.fold_constants_expr(i)).collect())
            }
            Expression::IndexAccess { expr, index } => Expression::IndexAccess {
                expr: Box::new(self.fold_constants_expr(expr)),
                index: Box::new(self.fold_constants_expr(index)),
            },
            Expression::ListSlice { expr, start, end } => Expression::ListSlice {
                expr: Box::new(self.fold_constants_expr(expr)),
                start: start
                    .as_ref()
                    .map(|s| Box::new(self.fold_constants_expr(s))),
                end: end.as_ref().map(|e| Box::new(self.fold_constants_expr(e))),
            },
            Expression::IsNull(inner) => {
                Expression::IsNull(Box::new(self.fold_constants_expr(inner)))
            }
            Expression::IsNotNull(inner) => {
                Expression::IsNotNull(Box::new(self.fold_constants_expr(inner)))
            }
            Expression::PredicateExpr(pred) => {
                Expression::PredicateExpr(Box::new(self.fold_constants_pred(pred)))
            }
            Expression::ExprPropertyAccess { expr, property } => Expression::ExprPropertyAccess {
                expr: Box::new(self.fold_constants_expr(expr)),
                property: property.clone(),
            },
            _ => expr.clone(),
        }
    }

    /// Fold constant sub-expressions in a predicate tree.
    pub(super) fn fold_constants_pred(&self, pred: &Predicate) -> Predicate {
        match pred {
            Predicate::Comparison {
                left,
                operator,
                right,
            } => Predicate::Comparison {
                left: self.fold_constants_expr(left),
                operator: *operator,
                right: self.fold_constants_expr(right),
            },
            Predicate::And(l, r) => Predicate::And(
                Box::new(self.fold_constants_pred(l)),
                Box::new(self.fold_constants_pred(r)),
            ),
            Predicate::Or(l, r) => Predicate::Or(
                Box::new(self.fold_constants_pred(l)),
                Box::new(self.fold_constants_pred(r)),
            ),
            Predicate::Xor(l, r) => Predicate::Xor(
                Box::new(self.fold_constants_pred(l)),
                Box::new(self.fold_constants_pred(r)),
            ),
            Predicate::Not(inner) => Predicate::Not(Box::new(self.fold_constants_pred(inner))),
            Predicate::IsNull(e) => Predicate::IsNull(self.fold_constants_expr(e)),
            Predicate::IsNotNull(e) => Predicate::IsNotNull(self.fold_constants_expr(e)),
            Predicate::In { expr, list } => {
                let folded_expr = self.fold_constants_expr(expr);
                let folded_list: Vec<Expression> =
                    list.iter().map(|i| self.fold_constants_expr(i)).collect();
                // If all items are literals, convert to InLiteralSet for O(1) lookup
                let all_literal: Option<std::collections::HashSet<Value>> = folded_list
                    .iter()
                    .map(|item| {
                        if let Expression::Literal(v) = item {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                if let Some(values) = all_literal {
                    Predicate::InLiteralSet {
                        expr: folded_expr,
                        values,
                    }
                } else {
                    Predicate::In {
                        expr: folded_expr,
                        list: folded_list,
                    }
                }
            }
            Predicate::InLiteralSet { .. } => pred.clone(),
            Predicate::StartsWith { expr, pattern } => Predicate::StartsWith {
                expr: self.fold_constants_expr(expr),
                pattern: self.fold_constants_expr(pattern),
            },
            Predicate::EndsWith { expr, pattern } => Predicate::EndsWith {
                expr: self.fold_constants_expr(expr),
                pattern: self.fold_constants_expr(pattern),
            },
            Predicate::Contains { expr, pattern } => Predicate::Contains {
                expr: self.fold_constants_expr(expr),
                pattern: self.fold_constants_expr(pattern),
            },
            Predicate::Exists { .. } => pred.clone(),
            Predicate::InExpression { expr, list_expr } => Predicate::InExpression {
                expr: self.fold_constants_expr(expr),
                list_expr: self.fold_constants_expr(list_expr),
            },
            Predicate::LabelCheck { .. } => pred.clone(),
        }
    }
}
