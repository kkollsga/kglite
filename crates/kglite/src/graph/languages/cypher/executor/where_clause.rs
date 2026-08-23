use super::helpers::*;
use super::*;
use crate::datatypes::values::Value;
use crate::graph::algorithms::vector as vs;
use crate::graph::core::membership::{self, MembershipSet};
use crate::graph::core::pattern_matching::{MatchBinding, PatternMatch};
use crate::graph::storage::GraphRead;
use std::collections::HashSet;
use std::sync::Arc;

impl<'a> CypherExecutor<'a> {
    pub(super) fn bindings_compatible(&self, row: &ResultRow, m: &PatternMatch) -> bool {
        for (var, binding) in &m.bindings {
            if let Some(&existing_idx) = row.node_bindings.get(var) {
                match binding {
                    MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => {
                        if *index != existing_idx {
                            return false;
                        }
                    }
                    _ => return false,
                }
            } else if let MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) = binding
            {
                // A node variable carried only as a projected VALUE — a
                // `Value::Node` / `Value::NodeRef` from `WITH n` after the
                // fold pass rewrote bindings, or `UNWIND collect(n) AS n` —
                // constrains the pattern to exactly that node, the same
                // openCypher re-MATCH semantics the Edge arm below applies
                // to relationship variables. A null-valued binding matches
                // nothing; a non-node projected value can never satisfy a
                // node pattern. (`NodeValue.id` / `NodeRef` both carry the
                // petgraph NodeIndex — see `materialize_node_value`.)
                match row.projected.get(var) {
                    None => {}
                    Some(Value::Node(nv)) => {
                        if nv.id as usize != index.index() {
                            return false;
                        }
                    }
                    Some(Value::NodeRef(i)) => {
                        if *i as usize != index.index() {
                            return false;
                        }
                    }
                    Some(_) => return false,
                }
            }
            // The same re-MATCH semantics for a relationship variable already
            // bound on the row — as a carried edge binding, or as a projected
            // value from `WITH r` / `UNWIND collect(r)`.
            if let MatchBinding::Edge { edge_index, .. } = binding {
                if let Some(existing) = row.edge_bindings.get(var) {
                    if existing.edge_index != *edge_index {
                        return false;
                    }
                } else {
                    match row.projected.get(var) {
                        None => {}
                        Some(Value::Relationship(rel)) => {
                            if rel.id as usize != edge_index.index() {
                                return false;
                            }
                        }
                        Some(_) => return false,
                    }
                }
            }
        }
        true
    }

    /// `EXISTS { … }` — whether the subquery has at least one match, scoped to
    /// this row's bindings.
    ///
    /// The single evaluation site for every spelling EXISTS has: a bare
    /// pattern predicate, `NOT EXISTS` (the `Not` wrapper inverts what this
    /// returns), a projected `RETURN EXISTS {…}`, and `CASE WHEN EXISTS`.
    fn evaluate_exists_subquery(
        &self,
        patterns: &[crate::graph::core::pattern_matching::Pattern],
        pattern_groups: &[usize],
        where_clause: &Option<Box<Predicate>>,
        row: &ResultRow,
    ) -> Result<Option<bool>, String> {
        // Fast path: a 3-element pattern with one bound node is an edge-
        // existence check, done without a PatternExecutor.
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
        // Relationship uniqueness (the openCypher trail rule) applies
        // across the subquery's comma patterns exactly as across the
        // comma patterns of one MATCH: two different pattern edges may
        // not bind the same relationship. It does NOT apply across the
        // multi-clause subquery form's separate MATCHes
        // (`EXISTS { MATCH … MATCH … }`) — those are distinct clause
        // scopes (`pattern_groups`), and edges may repeat across them
        // exactly as across top-level MATCH clauses. Only enforced
        // when some group carries two or more edge patterns — the
        // common single-pattern EXISTS pays nothing.
        let enforce_rel_uniqueness =
            match_clause::grouped_patterns_need_rel_uniqueness(patterns, pattern_groups);
        // One witness decides EXISTS — and decides NOT EXISTS just as
        // well, since zero witnesses is the same answer capped or not.
        // The cap is exact, so `PatternExecutor::execute`'s uncapped
        // retry still fires when its advisory pre-caps bit and the
        // pass came back empty: a witness that only exists past the
        // candidate cap is found on the retry, not missed.
        let witness_cap = Self::exists_witness_cap(patterns, where_clause, row);
        let mut combined_rows: Vec<ResultRow> = vec![row.clone()];
        // Parallel to `combined_rows` when enforcing: the edge indices
        // each row consumed within the CURRENT clause group (clause-
        // local — outer MATCH edges may legitimately reappear here,
        // and the sets reset at every group boundary).
        let mut clause_edge_sets: Vec<Vec<petgraph::graph::EdgeIndex>> = if enforce_rel_uniqueness {
            vec![Vec::new()]
        } else {
            Vec::new()
        };
        let mut prev_group: Option<usize> = None;
        for (pi, pattern) in patterns.iter().enumerate() {
            if combined_rows.is_empty() {
                return Ok(Some(false));
            }
            // New clause group (a MATCH separator): edges bound by
            // earlier groups no longer constrain — reset each row's
            // clause-local edge set.
            let group = pattern_groups.get(pi).copied().unwrap_or(0);
            if enforce_rel_uniqueness && prev_group.is_some_and(|g| g != group) {
                for set in &mut clause_edge_sets {
                    set.clear();
                }
            }
            prev_group = Some(group);
            let resolved;
            let pat = if Self::pattern_has_vars(pattern) {
                resolved = self.resolve_pattern_vars(pattern, row);
                &resolved
            } else {
                pattern
            };
            // `witness_cap` is `None` for exactly the shapes whose expansion
            // can run away (see `exists_witness_cap`), so the operator's
            // match ceiling is what bounds those.
            let matches = self
                .materializing_executor(
                    witness_cap,
                    &row.node_bindings,
                    "EXISTS subquery expansion",
                )
                .execute(pat)?;

            let mut next_rows: Vec<ResultRow> = Vec::new();
            let mut next_sets: Vec<Vec<petgraph::graph::EdgeIndex>> = Vec::new();
            for (ci, current) in combined_rows.iter().enumerate() {
                for m in &matches {
                    if !self.bindings_compatible(current, m) {
                        continue;
                    }
                    if enforce_rel_uniqueness {
                        let mut m_edges = Vec::new();
                        match_clause::match_edge_indices(m, &mut m_edges);
                        if m_edges.iter().any(|e| clause_edge_sets[ci].contains(e)) {
                            continue; // trail rule: edge re-use across patterns
                        }
                        let mut next = clause_edge_sets[ci].clone();
                        next.extend(m_edges);
                        next_sets.push(next);
                    }
                    let mut merged = current.clone();
                    self.merge_match_into_row(&mut merged, m);
                    next_rows.push(merged);
                }
            }
            combined_rows = next_rows;
            if enforce_rel_uniqueness {
                clause_edge_sets = next_sets;
            }
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
                matches!(
                    self.evaluate_predicate_tristate(where_pred, r),
                    Ok(Some(true))
                )
            })))
        } else {
            Ok(Some(true))
        }
    }

    /// How many matches an `EXISTS { … }` subquery needs from one of its
    /// patterns: `Some(1)` when the first match the executor returns settles
    /// the predicate, `None` when it might not.
    ///
    /// Three things can make a returned match *not* an answer, and each one
    /// makes the cap unsound because the arm would read "no witness" from a
    /// truncated run:
    ///
    /// 1. **More than one pattern.** The arm joins the patterns; a match of
    ///    the first may be incompatible with every match of the second.
    /// 2. **An inner `WHERE`.** It is applied after the join, so the first
    ///    witness may fail it while a later one passes.
    /// 3. **A binding the executor does not enforce.** Node variables in
    ///    `row.node_bindings` are pushed down as pre-bindings, so every match
    ///    already agrees with them. Values carried only in `row.projected`
    ///    (`UNWIND collect(n) AS n`, a folded `WITH n`) and relationship
    ///    variables bound on the row are not: `bindings_compatible` rejects
    ///    those afterwards, which a cap of one has no second candidate to
    ///    survive. `try_fast_exists_check` applies guards 1 and 3 as well; it
    ///    does not need 2, evaluating an inner WHERE per candidate edge.
    pub(super) fn exists_witness_cap(
        patterns: &[crate::graph::core::pattern_matching::Pattern],
        where_clause: &Option<Box<Predicate>>,
        row: &ResultRow,
    ) -> Option<usize> {
        use crate::graph::core::pattern_matching::PatternElement;

        if patterns.len() != 1 || where_clause.is_some() {
            return None;
        }
        for element in &patterns[0].elements {
            match element {
                PatternElement::Node(np) => {
                    if let Some(var) = np.variable.as_deref() {
                        if !row.node_bindings.contains_key(var) && row.projected.contains_key(var) {
                            return None;
                        }
                    }
                }
                PatternElement::Edge(ep) => {
                    if let Some(var) = ep.variable.as_deref() {
                        if row.edge_bindings.contains_key(var)
                            || row.projected.contains_key(var)
                            || row.node_bindings.contains_key(var)
                        {
                            return None;
                        }
                    }
                }
            }
        }
        Some(1)
    }

    pub(super) fn execute_where(
        &self,
        clause: &WhereClause,
        mut result_set: ResultSet,
    ) -> Result<ResultSet, String> {
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

        let in_filters = Self::extract_in_indexable_predicates(&clause.predicate);
        for (variable, property, values) in &in_filters {
            if let Some(node_type) = self.infer_node_type(variable, &result_set) {
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

        let folded_pred = self.fold_constants_pred(&clause.predicate);

        // Fast path: spatial contains() filter bypasses expression evaluator
        if let Some((spec, remainder)) = Self::try_extract_contains_filter(&folded_pred) {
            result_set.rows.retain(|row| {
                let container_idx = match row.node_bindings.get(&spec.container_variable) {
                    Some(&idx) => idx,
                    None => return false,
                };
                self.ensure_node_spatial_cached(container_idx);
                // Scope read lock: clone Arc + bbox, then drop lock
                let container = {
                    let cache = self.spatial_shard(container_idx.index()).read().unwrap();
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

                // Contained side: a Point (a constant, or the node's
                // Location) or a Geometry (a polygon-bearing node with no
                // Location). Without the Geometry arm the fast path
                // silently returned false for polygon-vs-polygon, masking
                // outer-contains-inner matches.
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
                        let cache = self.spatial_shard(contained_idx.index()).read().unwrap();
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
                let node = match graph.graph.node_view(idx) {
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
                let node_type = match graph.graph.node_view(idx) {
                    Some(n) => n.node_type_str(&graph.interner),
                    None => return false,
                };
                let store = match graph.embedding_store(node_type, &spec.prop_name) {
                    Some(s) => s,
                    None => return false,
                };
                let (embedding, norm) = match store.get_embedding_with_norm(idx.index()) {
                    Some(e) => e,
                    None => return false,
                };
                let score = spec.scorer.score(&spec.query_vec, embedding, norm) as f64;
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
                .and_then(|&idx| self.graph.graph.node_view(idx))
                .map(|node| node.node_type_str(&self.graph.interner).to_string())
        })
    }

    /// Evaluate a predicate in boolean (WHERE-row-keep) terms.
    ///
    /// External callers (HAVING, OPTIONAL MATCH filter, list comprehensions,
    /// spatial joins) only care whether to keep the row, so NULL collapses
    /// with `false` here — the historical "row drops on false" contract at
    /// every callsite — while NULL propagation stays exact internally. See
    /// `evaluate_predicate_tristate`.
    pub(super) fn evaluate_predicate(
        &self,
        pred: &Predicate,
        row: &ResultRow,
    ) -> Result<bool, String> {
        Ok(self.evaluate_predicate_tristate(pred, row)? == Some(true))
    }

    /// Drop every row of `rows` that `pred` does not keep, with the fused
    /// paths' error contract applied.
    ///
    /// Post-aggregation filters (HAVING, `WITH … WHERE`) have always dropped a
    /// row whose predicate could not be evaluated — that is how an unbound
    /// `OPTIONAL MATCH` binding and the aggregate-reference quirks behave, and
    /// it stays. An uncompilable regex and an unbound `$parameter` are a
    /// different animal: they are wrong for every row, and the unfused `WHERE`
    /// path raises both, so swallowing them here answered an invalid query
    /// with a silent empty result. See
    /// [`super::helpers::is_user_input_error`].
    pub(super) fn retain_rows_matching(
        &self,
        rows: &mut Vec<ResultRow>,
        pred: &Predicate,
    ) -> Result<(), String> {
        // `retain` cannot return, so the first flagged error is carried out.
        let mut failure: Option<String> = None;
        rows.retain(|row| match self.evaluate_predicate(pred, row) {
            Ok(keep) => keep,
            Err(e) => {
                if failure.is_none() && super::helpers::is_user_input_error(&e) {
                    failure = Some(e);
                }
                false
            }
        });
        match failure {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// `WHERE n:Label` — true iff `variable` is bound to a node carrying
    /// `label` as its primary type OR a secondary label. An unbound binding
    /// (`OPTIONAL MATCH` that did not match) or a non-node binding is false,
    /// never an error.
    fn row_binding_has_label(&self, row: &ResultRow, variable: &str, label: &str) -> bool {
        let Some(&idx) = row.node_bindings.get(variable) else {
            return false;
        };
        if self.graph.graph.node_view(idx).is_none() {
            return false;
        }
        self.graph
            .node_has_label(idx, crate::graph::schema::InternedKey::from_str(label))
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
    pub(super) fn evaluate_predicate_tristate(
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
                evaluate_comparison(&left_val, operator, &right_val).map(Some)
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
            Predicate::Not(inner) => Ok(self.evaluate_predicate_tristate(inner, row)?.map(|b| !b)),
            Predicate::LabelCheck {
                variable, label, ..
            } => Ok(Some(self.row_binding_has_label(row, variable, label))),
            Predicate::IsNull(expr) => {
                let val = self.evaluate_expression(expr, row)?;
                Ok(Some(matches!(val, Value::Null)))
            }
            Predicate::IsNotNull(expr) => {
                let val = self.evaluate_expression(expr, row)?;
                Ok(Some(!matches!(val, Value::Null)))
            }
            Predicate::In { expr, list } => {
                // openCypher three-valued IN semantics:
                //   NULL IN anything                  → NULL
                //   x IN [..]  (match present)        → true (NULLs in the list are immaterial)
                //   x IN [..]  (no match, list has NULL) → NULL
                //   x IN [..]  (no match, no NULL)    → false
                let val = self.evaluate_expression(expr, row)?;
                if matches!(val, Value::Null) {
                    return Ok(None);
                }
                // Reaching here means the list is genuinely per-row: constant
                // folding rewrites an all-literal (or row-independent) list to
                // `InLiteralSet` with a prepared MembershipSet. There is no
                // list to index — every element has to be evaluated for this
                // row regardless — so the elements are probed one at a time,
                // through the same shared rule, keeping the early exit.
                let mut saw_null = false;
                for item in list {
                    let item_val = self.evaluate_expression(item, row)?;
                    match membership::probe_element(&val, &item_val) {
                        Some(true) => return Ok(Some(true)),
                        Some(false) => {}
                        None => saw_null = true,
                    }
                }
                if saw_null {
                    Ok(None)
                } else {
                    Ok(Some(false))
                }
            }
            Predicate::InLiteralSet { expr, values } => {
                // Same Kleene rules as Predicate::In; the difference is that
                // `values` is a pre-built MembershipSet, so both the match and
                // the no-match answer cost one coercion-normalized probe —
                // and the NULL element is a flag read rather than a scan.
                let val = self.evaluate_expression(expr, row)?;
                if matches!(val, Value::Null) {
                    return Ok(None);
                }
                if values.matches(&val) {
                    return Ok(Some(true));
                }
                if values.has_null() {
                    return Ok(None);
                }
                Ok(Some(false))
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
                pattern_groups,
                where_clause,
            } => self.evaluate_exists_subquery(patterns, pattern_groups, where_clause, row),
            Predicate::InExpression { expr, list_expr } => {
                // Same Kleene rules as Predicate::In; the LHS and the list are
                // both arbitrary expressions, so NULL can come from either.
                // `parse_list_value(&Value::Null)` returns an empty vec, so a
                // NULL list_val collapses to "empty list, no NULLs seen" —
                // we lift that check explicitly so it propagates NULL.
                let val = self.evaluate_expression(expr, row)?;
                if matches!(val, Value::Null) {
                    return Ok(None);
                }
                let list_val = self.evaluate_expression(list_expr, row)?;
                if matches!(list_val, Value::Null) {
                    return Ok(None);
                }
                // Borrow the list where it already is a list. The previous
                // `parse_list_value(&list_val)` cloned every element of the
                // whole list for every row; only the string-encoded form
                // needs parsing at all. A row-*independent* list never gets
                // here — constant folding turns it into `InLiteralSet`.
                let parsed;
                let items: &[Value] = match &list_val {
                    Value::List(items) => items,
                    other => {
                        parsed = parse_list_value(other);
                        &parsed
                    }
                };
                Ok(membership::kleene_contains_linear(&val, items))
            }
        }
    }

    /// Try to extract a `vector_score(n, prop, vec [, metric]) {>|>=|<|<=} threshold`
    /// pattern from a (folded) predicate. Returns the spec and optional
    /// remainder predicate for the other AND conditions.
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
                let (vs_expr, threshold_expr, greater_than, inclusive) = match operator {
                    ComparisonOp::GreaterThan => (left, right, true, false),
                    ComparisonOp::GreaterThanEq => (left, right, true, true),
                    ComparisonOp::LessThan => (left, right, false, false),
                    ComparisonOp::LessThanEq => (left, right, false, true),
                    _ => return None,
                };

                if let Some(spec) =
                    self.extract_vector_score_spec(vs_expr, threshold_expr, greater_than, inclusive)
                {
                    return Some((spec, None));
                }

                // Flipped operands, so the comparison direction flips with them.
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

    pub(super) fn extract_vector_score_spec(
        &self,
        func_expr: &Expression,
        threshold_expr: &Expression,
        greater_than: bool,
        inclusive: bool,
    ) -> Option<VectorScoreFilterSpec> {
        // func_expr must be vector_score(variable, prop, query_vec [, metric]),
        // with prop and query_vec already constant-folded to literals.
        let (name, args) = match func_expr {
            Expression::FunctionCall { name, args, .. } => (name, args),
            _ => return None,
        };
        if name != "vector_score" || args.len() < 3 || args.len() > 4 {
            return None;
        }

        let threshold = match threshold_expr {
            Expression::Literal(val) => crate::graph::core::value_operations::value_to_f64(val)?,
            _ => return None,
        };

        let variable = match &args[0] {
            Expression::Variable(v) => v.clone(),
            _ => return None,
        };

        let prop_name = match &args[1] {
            Expression::Literal(Value::String(s)) => s.clone(),
            _ => return None,
        };

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

        // Optional metric (default cosine). An unrecognized name bails the fast
        // path (None) so the general evaluator handles it.
        let metric = if args.len() > 3 {
            match &args[3] {
                Expression::Literal(Value::String(s)) => vs::DistanceMetric::from_name(s)?,
                _ => vs::DistanceMetric::Cosine,
            }
        } else {
            vs::DistanceMetric::Cosine
        };
        let scorer = vs::Scorer::new(metric, &query_vec);

        Some(VectorScoreFilterSpec {
            variable,
            prop_name,
            query_vec,
            scorer,
            threshold,
            greater_than,
            inclusive,
        })
    }

    /// Try to extract a distance filter from a (folded) predicate. Returns the
    /// spec and optional remainder predicate for the other AND conditions.
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

                let threshold = match threshold_expr {
                    Expression::Literal(val) => {
                        crate::graph::core::value_operations::value_to_f64(val)?
                    }
                    _ => return None,
                };

                let spec = Self::extract_distance_call(dist_expr, threshold, less_than, inclusive)?;
                Some((spec, None))
            }
            Predicate::And(left, right) => {
                if let Some((spec, None)) = Self::try_extract_distance_filter(left) {
                    return Some((spec, Some(right)));
                }
                if let Some((spec, None)) = Self::try_extract_distance_filter(right) {
                    return Some((spec, Some(left)));
                }
                None
            }
            _ => None,
        }
    }

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

    pub(super) fn extract_literal_f64(expr: &Expression) -> Option<f64> {
        if let Expression::Literal(val) = expr {
            crate::graph::core::value_operations::value_to_f64(val)
        } else {
            None
        }
    }

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
            Predicate::Not(inner) => {
                if let Some((mut spec, None)) = Self::try_extract_contains_filter(inner) {
                    spec.negated = !spec.negated;
                    Some((spec, None))
                } else {
                    None
                }
            }
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

    pub(super) fn extract_contains_call(
        expr: &Expression,
        negated: bool,
    ) -> Option<ContainsFilterSpec> {
        if let Expression::FunctionCall { name, args, .. } = expr {
            if name != "contains" || args.len() != 2 {
                return None;
            }
            // The container must be a bare Variable (a node with geometry
            // configured), not an expression.
            let container_variable = match &args[0] {
                Expression::Variable(name) => name.clone(),
                _ => return None,
            };
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

    /// Return a copy of `expr` with every row-independent sub-tree replaced by
    /// the `Literal` it evaluates to.
    pub(crate) fn fold_constants_expr(&self, expr: &Expression) -> Expression {
        if matches!(expr, Expression::Literal(_)) {
            return expr.clone();
        }
        if Self::is_row_independent(expr) {
            let dummy = ResultRow::new();
            if let Ok(val) = self.evaluate_expression(expr, &dummy) {
                return Expression::Literal(val);
            }
            // If evaluation fails (e.g., missing parameter), keep original
            return expr.clone();
        }
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
                // If all items are literals, convert to InLiteralSet so the
                // list is indexed once instead of walked per row.
                let all_literal: Option<Vec<Value>> = folded_list
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
                        values: MembershipSet::new(values),
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
            Predicate::InExpression { expr, list_expr } => {
                let folded_expr = self.fold_constants_expr(expr);
                let folded_list = self.fold_constants_expr(list_expr);
                // A row-independent RHS — a `$param`, a literal list, any
                // expression that folds to one — is resolved and indexed
                // here, once, instead of being re-evaluated and re-cloned for
                // every row. `Value::Null` is deliberately left alone:
                // `x IN null` is UNKNOWN, which an empty set would answer as
                // false. A string-encoded list also stays on the
                // `InExpression` path, which is what knows how to parse it.
                if let Expression::Literal(Value::List(items)) = &folded_list {
                    return Predicate::InLiteralSet {
                        expr: folded_expr,
                        values: MembershipSet::new(items.clone()),
                    };
                }
                Predicate::InExpression {
                    expr: folded_expr,
                    list_expr: folded_list,
                }
            }
            Predicate::LabelCheck { .. } => pred.clone(),
        }
    }
}
