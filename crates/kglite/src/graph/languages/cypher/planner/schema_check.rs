//! Schema validation pass — runs after parse, before optimize.
//!
//! Catches unknown property references inside **pattern-literal syntax**
//! (`{prop: value}`) before the executor commits to a scan. These are
//! unambiguously property names — a typo here can never be a virtual
//! column or computed alias, so rejecting produces zero false positives.
//!
//! Covers:
//! - Variable scope across projections, subqueries, expressions, and updates.
//! - `MATCH (n:T {prop: v})` and `OPTIONAL MATCH` (read patterns).
//! - `EXISTS { MATCH (n:T {prop: v}) }` inside WHERE / AND / OR / NOT
//!   subqueries.
//! - **`CREATE (n:T {prop: v})`** and `CREATE`-style multi-element paths.
//! - **`MERGE (n:T {prop: v})`** including the embedded `CREATE` shape.
//!
//! Deliberately *does not reject* (these are legal, so they are warned about
//! non-fatally instead — see below — never turned into errors):
//! - Unknown node types in MATCH (`MATCH (n:Nonexistent)` legitimately
//!   returns zero rows and is a common existence-check idiom).
//! - Unknown connection types (same rationale).
//!
//! Deliberately *ignores entirely*:
//! - Property references in WHERE / RETURN expressions (virtual columns,
//!   timeseries sub-nodes, aliases can be legitimate `n.prop` accesses
//!   not present in `node_type_metadata`).
//! - `SET n.prop = X` and `REMOVE n.prop` — SET may legitimately
//!   introduce new properties depending on kglite's mutation policy;
//!   REMOVE of a non-existent property is benign.
//!
//! ## Non-fatal "did you mean?" warnings
//!
//! [`collect_unknown_pattern_warnings`] / [`warn_unknown_pattern_refs`] flag
//! MATCH patterns that reference an unknown node label or relationship type —
//! the most common "why is my query empty?" typo — with an edit-distance hint,
//! *without* rejecting (the zero-row existence-check idiom stays valid). The
//! wrapper emits to stderr (kglite's existing `warning:` convention); routing
//! the same messages into `QueryDiagnostics` so MCP/agent callers see them
//! structurally is the natural next step.
//!
//! ## Pipeline placement
//!
//! Called by three downstream Cypher consumers after `parse_cypher` and
//! before the planner's `optimize_with_disabled`:
//! - `src/graph/pyapi/kg_core.rs::cypher` (Python boundary)
//! - `crates/kglite-mcp-server/src/tools.rs::cypher_query`
//! - `crates/kglite-bolt-server/src/backend.rs::execute`

use super::super::ast::*;
use crate::graph::core::pattern_matching::{Pattern, PatternElement};
use crate::graph::mutation::validation::did_you_mean;
use crate::graph::schema::{DirGraph, InternedKey};
use std::collections::{HashMap, HashSet};

/// Built-in fields valid on any node type — mirrors BUILTIN_FIELDS in
/// `mutation/validation.rs`. Listed explicitly so it's obvious what's
/// tolerated without a metadata entry.
const BUILTIN_FIELDS: &[&str] = &["id", "title", "name", "type"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaErrorKind {
    UnknownProperty,
    UndefinedVariable,
}

#[derive(Debug, Clone)]
pub struct SchemaError {
    #[allow(dead_code)] // Test-only.
    pub kind: SchemaErrorKind,
    pub message: String,
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SchemaError {}

/// Validate a parsed Cypher query against the graph schema.
///
/// Runs in O(clauses) with O(1) lookups against `node_type_metadata`.
/// Returns early on the first violation.
///
/// Skips validation entirely when the graph has neither declared node
/// types (`node_type_metadata`) nor live nodes (`type_indices`) — that
/// state appears in tests and during initial construction.
pub fn validate_schema(query: &CypherQuery, graph: &DirGraph) -> Result<(), SchemaError> {
    validate_scope(query, &HashSet::new())?;

    // If the graph has no declared node types at all, there is nothing to
    // validate against — skip. This covers fresh graphs and construction
    // before any nodes/edges exist.
    if graph.node_type_metadata.is_empty() && graph.type_indices.is_empty() {
        return Ok(());
    }

    let mut var_types: HashMap<String, String> = HashMap::new();
    validate_query(query, graph, &mut var_types)
}

fn undefined_variable(name: &str) -> SchemaError {
    SchemaError {
        kind: SchemaErrorKind::UndefinedVariable,
        message: format!("Undefined variable '{}'", name),
    }
}

fn require_variable(name: &str, scope: &HashSet<String>) -> Result<(), SchemaError> {
    if scope.contains(name) {
        Ok(())
    } else {
        Err(undefined_variable(name))
    }
}

fn bind_pattern(pattern: &Pattern, scope: &mut HashSet<String>) {
    for element in &pattern.elements {
        match element {
            PatternElement::Node(node) => scope.extend(node.variable.iter().cloned()),
            PatternElement::Edge(edge) => scope.extend(edge.variable.iter().cloned()),
        }
    }
}

fn bind_create_pattern(pattern: &CreatePattern, scope: &mut HashSet<String>) {
    for element in &pattern.elements {
        match element {
            CreateElement::Node(node) => scope.extend(node.variable.iter().cloned()),
            CreateElement::Edge(edge) => scope.extend(edge.variable.iter().cloned()),
        }
    }
}

fn validate_scope(query: &CypherQuery, initial: &HashSet<String>) -> Result<(), SchemaError> {
    let mut scope = initial.clone();
    for clause in &query.clauses {
        match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => {
                for pattern in &m.patterns {
                    bind_pattern(pattern, &mut scope);
                }
                scope.extend(m.path_assignments.iter().map(|path| path.variable.clone()));
            }
            Clause::Where(where_clause) => {
                validate_predicate_scope(&where_clause.predicate, &scope)?
            }
            Clause::Return(return_clause) => {
                for item in &return_clause.items {
                    validate_expression_scope(&item.expression, &scope)?;
                }
                let mut having_scope = scope.clone();
                having_scope.extend(
                    return_clause
                        .items
                        .iter()
                        .filter_map(|item| item.alias.clone()),
                );
                if let Some(having) = &return_clause.having {
                    validate_predicate_scope(having, &having_scope)?;
                }
                scope = having_scope;
            }
            Clause::With(with_clause) => {
                for item in &with_clause.items {
                    validate_expression_scope(&item.expression, &scope)?;
                }
                let preserves_all = with_clause
                    .items
                    .iter()
                    .any(|item| matches!(item.expression, Expression::Star));
                let mut projected = if preserves_all {
                    scope.clone()
                } else {
                    HashSet::new()
                };
                for item in &with_clause.items {
                    if let Some(alias) = &item.alias {
                        projected.insert(alias.clone());
                    } else if let Expression::Variable(name) = &item.expression {
                        projected.insert(name.clone());
                    }
                }
                if let Some(where_clause) = &with_clause.where_clause {
                    validate_predicate_scope(&where_clause.predicate, &projected)?;
                }
                scope = projected;
            }
            Clause::OrderBy(order) => {
                for item in &order.items {
                    validate_expression_scope(&item.expression, &scope)?;
                }
            }
            Clause::Skip(skip) => validate_expression_scope(&skip.count, &scope)?,
            Clause::Limit(limit) => validate_expression_scope(&limit.count, &scope)?,
            Clause::Unwind(unwind) => {
                validate_expression_scope(&unwind.expression, &scope)?;
                scope.insert(unwind.alias.clone());
            }
            Clause::Union(union) => validate_scope(&union.query, initial)?,
            Clause::Create(create) => {
                for pattern in &create.patterns {
                    bind_create_pattern(pattern, &mut scope);
                    for element in &pattern.elements {
                        let properties = match element {
                            CreateElement::Node(node) => &node.properties,
                            CreateElement::Edge(edge) => &edge.properties,
                        };
                        for (_, expression) in properties {
                            validate_expression_scope(expression, &scope)?;
                        }
                    }
                }
            }
            Clause::Set(set) => {
                for item in &set.items {
                    match item {
                        SetItem::Property {
                            variable,
                            expression,
                            ..
                        }
                        | SetItem::Map {
                            variable,
                            expression,
                            ..
                        } => {
                            require_variable(variable, &scope)?;
                            validate_expression_scope(expression, &scope)?;
                        }
                        SetItem::Label { variable, .. } => require_variable(variable, &scope)?,
                    }
                }
            }
            Clause::Delete(delete) => {
                for expression in &delete.expressions {
                    validate_expression_scope(expression, &scope)?;
                }
            }
            Clause::Remove(remove) => {
                for item in &remove.items {
                    let variable = match item {
                        RemoveItem::Property { variable, .. }
                        | RemoveItem::Label { variable, .. } => variable,
                    };
                    require_variable(variable, &scope)?;
                }
            }
            Clause::Merge(merge) => {
                bind_create_pattern(&merge.pattern, &mut scope);
                for element in &merge.pattern.elements {
                    let properties = match element {
                        CreateElement::Node(node) => &node.properties,
                        CreateElement::Edge(edge) => &edge.properties,
                    };
                    for (_, expression) in properties {
                        validate_expression_scope(expression, &scope)?;
                    }
                }
                for items in [&merge.on_create, &merge.on_match].into_iter().flatten() {
                    for item in items {
                        match item {
                            SetItem::Property {
                                variable,
                                expression,
                                ..
                            }
                            | SetItem::Map {
                                variable,
                                expression,
                                ..
                            } => {
                                require_variable(variable, &scope)?;
                                validate_expression_scope(expression, &scope)?;
                            }
                            SetItem::Label { variable, .. } => require_variable(variable, &scope)?,
                        }
                    }
                }
            }
            Clause::Foreach {
                variable,
                list,
                body,
            } => {
                validate_expression_scope(list, &scope)?;
                let mut inner = scope.clone();
                inner.insert(variable.clone());
                validate_scope(
                    &CypherQuery {
                        clauses: body.clone(),
                        explain: false,
                        profile: false,
                        output_format: OutputFormat::Default,
                        optimizer_tags: Vec::new(),
                    },
                    &inner,
                )?;
            }
            Clause::Call(call) => {
                for (_, expression) in &call.parameters {
                    validate_expression_scope(expression, &scope)?;
                }
                scope.extend(
                    call.yield_items
                        .iter()
                        .map(|item| item.alias.as_ref().unwrap_or(&item.name).clone()),
                );
            }
            Clause::CallSubquery { import, body } => {
                for name in import {
                    require_variable(name, &scope)?;
                }
                let imported: HashSet<String> = import.iter().cloned().collect();
                validate_scope(body, &imported)?;
                if let Some(Clause::Return(return_clause)) = body
                    .clauses
                    .iter()
                    .rev()
                    .find(|clause| matches!(clause, Clause::Return(_)))
                {
                    for item in &return_clause.items {
                        if let Some(alias) = &item.alias {
                            scope.insert(alias.clone());
                        } else if let Expression::Variable(name) = &item.expression {
                            scope.insert(name.clone());
                        }
                    }
                }
            }
            // Physical clauses exist only after validation.
            _ => {}
        }
    }
    Ok(())
}

fn validate_predicate_scope(
    predicate: &Predicate,
    scope: &HashSet<String>,
) -> Result<(), SchemaError> {
    match predicate {
        Predicate::And(left, right) | Predicate::Or(left, right) | Predicate::Xor(left, right) => {
            validate_predicate_scope(left, scope)?;
            validate_predicate_scope(right, scope)
        }
        Predicate::Not(inner) => validate_predicate_scope(inner, scope),
        Predicate::Comparison { left, right, .. } => {
            validate_expression_scope(left, scope)?;
            validate_expression_scope(right, scope)
        }
        Predicate::IsNull(expression)
        | Predicate::IsNotNull(expression)
        | Predicate::InLiteralSet {
            expr: expression, ..
        } => validate_expression_scope(expression, scope),
        Predicate::In { expr, list } => {
            validate_expression_scope(expr, scope)?;
            for item in list {
                validate_expression_scope(item, scope)?;
            }
            Ok(())
        }
        Predicate::InExpression { expr, list_expr } => {
            validate_expression_scope(expr, scope)?;
            validate_expression_scope(list_expr, scope)
        }
        Predicate::StartsWith { expr, pattern }
        | Predicate::EndsWith { expr, pattern }
        | Predicate::Contains { expr, pattern } => {
            validate_expression_scope(expr, scope)?;
            validate_expression_scope(pattern, scope)
        }
        Predicate::LabelCheck { variable, .. } => require_variable(variable, scope),
        Predicate::Exists {
            patterns,
            where_clause,
            ..
        } => {
            let mut inner = scope.clone();
            for pattern in patterns {
                bind_pattern(pattern, &mut inner);
            }
            if let Some(where_clause) = where_clause {
                validate_predicate_scope(where_clause, &inner)?;
            }
            Ok(())
        }
    }
}

fn validate_expression_scope(
    expression: &Expression,
    scope: &HashSet<String>,
) -> Result<(), SchemaError> {
    match expression {
        Expression::Variable(name) | Expression::PropertyAccess { variable: name, .. } => {
            require_variable(name, scope)
        }
        Expression::FunctionCall { args, .. } | Expression::ListLiteral(args) => {
            for argument in args {
                validate_expression_scope(argument, scope)?;
            }
            Ok(())
        }
        Expression::Add(left, right)
        | Expression::Subtract(left, right)
        | Expression::Multiply(left, right)
        | Expression::Divide(left, right)
        | Expression::Modulo(left, right)
        | Expression::Concat(left, right)
        | Expression::IndexAccess {
            expr: left,
            index: right,
        } => {
            validate_expression_scope(left, scope)?;
            validate_expression_scope(right, scope)
        }
        Expression::Negate(inner)
        | Expression::IsNull(inner)
        | Expression::IsNotNull(inner)
        | Expression::ExprPropertyAccess { expr: inner, .. } => {
            validate_expression_scope(inner, scope)
        }
        Expression::Case {
            operand,
            when_clauses,
            else_expr,
        } => {
            if let Some(operand) = operand {
                validate_expression_scope(operand, scope)?;
            }
            for (condition, result) in when_clauses {
                match condition {
                    CaseCondition::Predicate(predicate) => {
                        validate_predicate_scope(predicate, scope)?
                    }
                    CaseCondition::Expression(expression) => {
                        validate_expression_scope(expression, scope)?
                    }
                }
                validate_expression_scope(result, scope)?;
            }
            if let Some(else_expr) = else_expr {
                validate_expression_scope(else_expr, scope)?;
            }
            Ok(())
        }
        Expression::ListComprehension {
            variable,
            list_expr,
            filter,
            map_expr,
        } => {
            validate_expression_scope(list_expr, scope)?;
            let mut inner = scope.clone();
            inner.insert(variable.clone());
            if let Some(filter) = filter {
                validate_predicate_scope(filter, &inner)?;
            }
            if let Some(map_expr) = map_expr {
                validate_expression_scope(map_expr, &inner)?;
            }
            Ok(())
        }
        Expression::ListSlice { expr, start, end } => {
            validate_expression_scope(expr, scope)?;
            if let Some(start) = start {
                validate_expression_scope(start, scope)?;
            }
            if let Some(end) = end {
                validate_expression_scope(end, scope)?;
            }
            Ok(())
        }
        Expression::MapProjection { variable, items } => {
            require_variable(variable, scope)?;
            for item in items {
                if let MapProjectionItem::Alias { expr, .. } = item {
                    validate_expression_scope(expr, scope)?;
                }
            }
            Ok(())
        }
        Expression::MapLiteral(entries) => {
            for (_, expression) in entries {
                validate_expression_scope(expression, scope)?;
            }
            Ok(())
        }
        Expression::QuantifiedList {
            variable,
            list_expr,
            filter,
            ..
        } => {
            validate_expression_scope(list_expr, scope)?;
            let mut inner = scope.clone();
            inner.insert(variable.clone());
            validate_predicate_scope(filter, &inner)
        }
        Expression::Reduce {
            accumulator,
            init,
            variable,
            list_expr,
            body,
        } => {
            validate_expression_scope(init, scope)?;
            validate_expression_scope(list_expr, scope)?;
            let mut inner = scope.clone();
            inner.insert(accumulator.clone());
            inner.insert(variable.clone());
            validate_expression_scope(body, &inner)
        }
        Expression::PredicateExpr(predicate) => validate_predicate_scope(predicate, scope),
        Expression::WindowFunction {
            partition_by,
            order_by,
            ..
        } => {
            for expression in partition_by {
                validate_expression_scope(expression, scope)?;
            }
            for item in order_by {
                validate_expression_scope(&item.expression, scope)?;
            }
            Ok(())
        }
        Expression::CountSubquery {
            patterns,
            where_clause,
        } => {
            let mut inner = scope.clone();
            for pattern in patterns {
                bind_pattern(pattern, &mut inner);
            }
            if let Some(where_clause) = where_clause {
                validate_predicate_scope(where_clause, &inner)?;
            }
            Ok(())
        }
        Expression::Literal(_) | Expression::Parameter(_) | Expression::Star => Ok(()),
    }
}

/// Best-effort, NON-FATAL warnings for `WHERE var.prop …` where `prop` exists
/// on **no** node of `var`'s label — so `null <op> x` is false and the
/// predicate silently filters out every row (operator feedback A1b 2026-06-17).
/// A warning, not an error: a legitimately-sparse property is still in the
/// type's metadata (set on ≥1 node), so only a *genuinely-absent* property
/// trips this — no false positive on nullable columns.
fn absent_property_warnings(query: &CypherQuery, graph: &DirGraph) -> Vec<String> {
    if graph.node_type_metadata.is_empty() {
        return Vec::new();
    }
    // var → single known node label, from MATCH/OPTIONAL MATCH node patterns.
    // Multi-label / unknown-label vars are dropped (can't reason precisely).
    let mut var_label: HashMap<&str, &str> = HashMap::new();
    for clause in &query.clauses {
        if let Clause::Match(m) | Clause::OptionalMatch(m) = clause {
            for pattern in &m.patterns {
                for el in &pattern.elements {
                    if let PatternElement::Node(np) = el {
                        if let (Some(var), Some(label)) =
                            (np.variable.as_deref(), np.node_type.as_deref())
                        {
                            if np.extra_labels.is_empty()
                                && graph.node_type_metadata.contains_key(label)
                            {
                                var_label.insert(var, label);
                            } else {
                                var_label.remove(var);
                            }
                        }
                    }
                }
            }
        }
    }
    if var_label.is_empty() {
        return Vec::new();
    }

    let mut seen: HashSet<(&str, &str)> = HashSet::new();
    let mut out = Vec::new();
    for clause in &query.clauses {
        let pred = match clause {
            Clause::Where(w) => Some(&w.predicate),
            Clause::With(w) => w.where_clause.as_ref().map(|wc| &wc.predicate),
            _ => None,
        };
        if let Some(p) = pred {
            warn_absent_in_predicate(p, &var_label, graph, &mut seen, &mut out);
        }
    }
    out
}

/// True when `prop` is neither a built-in field nor in `node_type`'s declared
/// metadata (and the type *has* declared metadata — empty ⇒ skip, as
/// [`validate_property`] does, to avoid false positives on under-declared graphs).
fn property_absent(graph: &DirGraph, node_type: &str, prop: &str) -> bool {
    if BUILTIN_FIELDS.contains(&prop) {
        return false;
    }
    match graph.node_type_metadata.get(node_type) {
        Some(tp) => !tp.is_empty() && !tp.contains_key(prop),
        None => false,
    }
}

fn warn_absent_in_predicate<'q>(
    pred: &'q Predicate,
    var_label: &HashMap<&'q str, &'q str>,
    graph: &DirGraph,
    seen: &mut HashSet<(&'q str, &'q str)>,
    out: &mut Vec<String>,
) {
    match pred {
        Predicate::And(a, b) | Predicate::Or(a, b) | Predicate::Xor(a, b) => {
            warn_absent_in_predicate(a, var_label, graph, seen, out);
            warn_absent_in_predicate(b, var_label, graph, seen, out);
        }
        Predicate::Not(p) => warn_absent_in_predicate(p, var_label, graph, seen, out),
        Predicate::Comparison { left, right, .. } => {
            warn_absent_in_expr(left, var_label, graph, seen, out);
            warn_absent_in_expr(right, var_label, graph, seen, out);
        }
        Predicate::In { expr, .. }
        | Predicate::InLiteralSet { expr, .. }
        | Predicate::InExpression { expr, .. }
        | Predicate::StartsWith { expr, .. }
        | Predicate::EndsWith { expr, .. }
        | Predicate::Contains { expr, .. }
        | Predicate::IsNull(expr)
        | Predicate::IsNotNull(expr) => {
            warn_absent_in_expr(expr, var_label, graph, seen, out);
        }
        _ => {}
    }
}

fn warn_absent_in_expr<'q>(
    expr: &'q Expression,
    var_label: &HashMap<&'q str, &'q str>,
    graph: &DirGraph,
    seen: &mut HashSet<(&'q str, &'q str)>,
    out: &mut Vec<String>,
) {
    match expr {
        Expression::PropertyAccess { variable, property } => {
            if let Some(&label) = var_label.get(variable.as_str()) {
                if property_absent(graph, label, property)
                    && seen.insert((variable.as_str(), property.as_str()))
                {
                    let candidates: Vec<&str> = graph
                        .node_type_metadata
                        .get(label)
                        .map(|m| m.keys().map(|s| s.as_str()).collect())
                        .unwrap_or_default();
                    out.push(format!(
                        "WHERE references property '{property}' which no {label} node has — the \
                         comparison is null (always false), so this filters out every row.{}",
                        did_you_mean(property, &candidates)
                    ));
                }
            }
        }
        Expression::Add(a, b)
        | Expression::Subtract(a, b)
        | Expression::Multiply(a, b)
        | Expression::Divide(a, b)
        | Expression::Modulo(a, b)
        | Expression::Concat(a, b) => {
            warn_absent_in_expr(a, var_label, graph, seen, out);
            warn_absent_in_expr(b, var_label, graph, seen, out);
        }
        Expression::Negate(e) => warn_absent_in_expr(e, var_label, graph, seen, out),
        Expression::FunctionCall { args, .. } => {
            for a in args {
                warn_absent_in_expr(a, var_label, graph, seen, out);
            }
        }
        Expression::ListLiteral(items) => {
            for it in items {
                warn_absent_in_expr(it, var_label, graph, seen, out);
            }
        }
        _ => {}
    }
}

/// Non-fatal counterpart to [`validate_schema`]: collect "did you mean?"
/// warnings for MATCH patterns that reference a node label or relationship
/// type the graph has never seen (a zero-row existence check is legal Cypher,
/// so this is *not* an error), plus the absent-property warnings from
/// [`absent_property_warnings`]. Pure (no I/O), so directly testable;
/// [`warn_unknown_pattern_refs`] is the stderr-emitting wrapper.
pub fn collect_unknown_pattern_warnings(query: &CypherQuery, graph: &DirGraph) -> Vec<String> {
    let have_node_schema =
        !graph.node_type_metadata.is_empty() || graph.type_indices.keys().next().is_some();
    let have_edge_schema = !graph.connection_type_metadata.is_empty();
    if !have_node_schema && !have_edge_schema {
        return Vec::new();
    }

    // Walk MATCH / OPTIONAL MATCH patterns, checking each label/relationship
    // against the schema directly. The all-valid path (the overwhelming common
    // case) allocates nothing — only confirmed-unknown, not-yet-seen names are
    // recorded, and the candidate lists for "did you mean?" are built lazily
    // only if there's at least one unknown.
    let mut seen: HashSet<String> = HashSet::new();
    let mut unknown_labels: Vec<String> = Vec::new();
    let mut unknown_rels: Vec<String> = Vec::new();

    for clause in &query.clauses {
        if let Clause::Match(m) | Clause::OptionalMatch(m) = clause {
            for pattern in &m.patterns {
                for element in &pattern.elements {
                    match element {
                        PatternElement::Node(np) if have_node_schema => {
                            for label in np.node_type.iter().chain(np.extra_labels.iter()) {
                                // A label is known if it's a declared primary
                                // type OR a secondary label applied via
                                // add_label (`MATCH (n:Reviewer)` is valid even
                                // though `Reviewer` is no node's primary type).
                                let known = graph.node_type_metadata.contains_key(label)
                                    || graph.type_indices.contains_key(label)
                                    || graph
                                        .secondary_label_index
                                        .contains_key(&InternedKey::from_str(label));
                                if !known && seen.insert(format!("L:{label}")) {
                                    unknown_labels.push(label.clone());
                                }
                            }
                        }
                        PatternElement::Edge(ep) if have_edge_schema => {
                            let single = ep.connection_type.iter();
                            let multi = ep.connection_types.iter().flatten();
                            for rel in single.chain(multi) {
                                if !graph.connection_type_metadata.contains_key(rel)
                                    && seen.insert(format!("R:{rel}"))
                                {
                                    unknown_rels.push(rel.clone());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Seed with the absent-property warnings (A1b) so they're emitted alongside
    // the unknown-label/rel ones even when the labels/rels are all valid.
    let mut out: Vec<String> = absent_property_warnings(query, graph);
    if unknown_labels.is_empty() && unknown_rels.is_empty() {
        return out;
    }

    out.reserve(unknown_labels.len() + unknown_rels.len());
    if !unknown_labels.is_empty() {
        let candidates: Vec<&str> = graph
            .node_type_metadata
            .keys()
            .map(|s| s.as_str())
            .chain(graph.type_indices.keys())
            .collect();
        for label in &unknown_labels {
            out.push(format!(
                "MATCH references unknown node label '{label}' — the graph has no such type, \
                 so this pattern returns no rows.{}",
                did_you_mean(label, &candidates)
            ));
        }
    }
    if !unknown_rels.is_empty() {
        let candidates: Vec<&str> = graph
            .connection_type_metadata
            .keys()
            .map(|s| s.as_str())
            .collect();
        for rel in &unknown_rels {
            out.push(format!(
                "MATCH references unknown relationship type '{rel}' — the graph has no such \
                 edge type, so this pattern returns no rows.{}",
                did_you_mean(rel, &candidates)
            ));
        }
    }
    out
}

/// Emit [`collect_unknown_pattern_warnings`] to stderr (matching kglite's
/// existing `warning:`-prefixed convention for non-fatal query/load issues).
/// Called from the shared execute path so every binding gets the signal.
pub fn warn_unknown_pattern_refs(query: &CypherQuery, graph: &DirGraph) {
    for msg in collect_unknown_pattern_warnings(query, graph) {
        eprintln!("warning: {msg}");
    }
}

fn validate_query(
    query: &CypherQuery,
    graph: &DirGraph,
    var_types: &mut HashMap<String, String>,
) -> Result<(), SchemaError> {
    for clause in &query.clauses {
        validate_clause(clause, graph, var_types)?;
    }
    Ok(())
}

fn validate_clause(
    clause: &Clause,
    graph: &DirGraph,
    var_types: &mut HashMap<String, String>,
) -> Result<(), SchemaError> {
    match clause {
        Clause::Match(m) | Clause::OptionalMatch(m) => {
            for pattern in &m.patterns {
                validate_pattern(pattern, graph, var_types)?;
            }
        }
        Clause::Where(w) => {
            walk_predicate_for_nested_patterns(&w.predicate, graph, var_types)?;
        }
        Clause::With(w) => {
            for item in &w.items {
                if let Some(alias) = &item.alias {
                    var_types.remove(alias);
                }
            }
            if let Some(wc) = &w.where_clause {
                walk_predicate_for_nested_patterns(&wc.predicate, graph, var_types)?;
            }
        }
        Clause::Union(u) => {
            let mut inner_vars: HashMap<String, String> = HashMap::new();
            validate_query(&u.query, graph, &mut inner_vars)?;
        }
        Clause::CallSubquery { import, body } => {
            // Recurse into the body so pattern-literal property typos inside
            // `CALL { MATCH (n:T {prp: v}) ... }` get the same "did you
            // mean?" treatment as top-level patterns (mirrors the Union /
            // EXISTS recursion).
            //
            // Imported variables are EXTERNALLY bound: the leading importing
            // `WITH` was stripped at parse time, so the body re-binds them
            // from the outer scope. Seed the body's `var_types` with the
            // imported names' outer types so a body pattern that re-anchors
            // on an import (`CALL { WITH p MATCH (p {prp: v}) }`) validates
            // against the import's real type. Non-imported body variables
            // start fresh (§1.2 rule 1 — a bare body name is a new variable).
            let mut inner_vars: HashMap<String, String> = HashMap::new();
            for name in import {
                if let Some(ty) = var_types.get(name) {
                    inner_vars.insert(name.clone(), ty.clone());
                }
            }
            validate_query(body, graph, &mut inner_vars)?;
        }
        Clause::Return(_) | Clause::OrderBy(_) | Clause::Unwind(_) => {}
        Clause::Create(c) => {
            for pattern in &c.patterns {
                validate_create_pattern(pattern, graph, var_types)?;
            }
        }
        Clause::Merge(m) => {
            validate_create_pattern(&m.pattern, graph, var_types)?;
            // MERGE's `ON CREATE SET` / `ON MATCH SET` use `SetItem`,
            // which we intentionally skip — see the module doc-comment.
        }
        Clause::Set(_) | Clause::Delete(_) | Clause::Remove(_) | Clause::Call(_) => {}
        _ => {}
    }
    Ok(())
}

/// Validate a CREATE / MERGE pattern's node-pattern property names
/// against the schema. Edge-pattern properties aren't validated —
/// `connection_type_metadata` is keyed differently and edge schemas
/// in kglite are looser; revisit if a real divergence shows up.
fn validate_create_pattern(
    pattern: &CreatePattern,
    graph: &DirGraph,
    var_types: &mut HashMap<String, String>,
) -> Result<(), SchemaError> {
    for element in &pattern.elements {
        if let CreateElement::Node(np) = element {
            if let Some(ref node_type) = np.label {
                if let Some(ref var) = np.variable {
                    var_types.insert(var.clone(), node_type.clone());
                }
                for (prop_name, _expr) in &np.properties {
                    validate_property(node_type, prop_name, graph)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_pattern(
    pattern: &Pattern,
    graph: &DirGraph,
    var_types: &mut HashMap<String, String>,
) -> Result<(), SchemaError> {
    for element in &pattern.elements {
        if let PatternElement::Node(np) = element {
            if let Some(ref node_type) = np.node_type {
                if let Some(ref var) = np.variable {
                    var_types.insert(var.clone(), node_type.clone());
                }
                if let Some(ref props) = np.properties {
                    for prop_name in props.keys() {
                        validate_property(node_type, prop_name, graph)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Descend the predicate to find nested `EXISTS { MATCH ... }` patterns
/// so their pattern-literal properties get validated too. Does not check
/// expression-level property accesses or label checks — those are
/// delivered as Phase 3 diagnostics instead.
fn walk_predicate_for_nested_patterns(
    predicate: &Predicate,
    graph: &DirGraph,
    var_types: &HashMap<String, String>,
) -> Result<(), SchemaError> {
    match predicate {
        Predicate::And(a, b) | Predicate::Or(a, b) | Predicate::Xor(a, b) => {
            walk_predicate_for_nested_patterns(a, graph, var_types)?;
            walk_predicate_for_nested_patterns(b, graph, var_types)?;
        }
        Predicate::Not(p) => walk_predicate_for_nested_patterns(p, graph, var_types)?,
        Predicate::Exists {
            patterns,
            where_clause,
            ..
        } => {
            let mut inner_vars = var_types.clone();
            for p in patterns {
                validate_pattern(p, graph, &mut inner_vars)?;
            }
            if let Some(w) = where_clause {
                walk_predicate_for_nested_patterns(w, graph, &inner_vars)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_property(node_type: &str, property: &str, graph: &DirGraph) -> Result<(), SchemaError> {
    if BUILTIN_FIELDS.contains(&property) {
        return Ok(());
    }
    let Some(type_props) = graph.node_type_metadata.get(node_type) else {
        // No declared property metadata for this type — skip rather than
        // false-positive on dynamically-typed or under-declared graphs.
        return Ok(());
    };
    if type_props.is_empty() || type_props.contains_key(property) {
        return Ok(());
    }
    let candidates: Vec<&str> = type_props.keys().map(|s| s.as_str()).collect();
    let hint = did_you_mean(property, &candidates);
    let mut sorted = candidates;
    sorted.sort();
    Err(SchemaError {
        kind: SchemaErrorKind::UnknownProperty,
        message: format!(
            "Unknown property '{}' on {}.{}\n  Valid properties: {}",
            property,
            node_type,
            hint,
            sorted.join(", ")
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::languages::cypher::parser::parse_cypher;

    fn graph_with_schema() -> DirGraph {
        let mut g = DirGraph::new();
        let mut person_props = HashMap::new();
        person_props.insert("age".to_string(), "int".to_string());
        person_props.insert("email".to_string(), "string".to_string());
        g.upsert_node_type_metadata("Person", person_props);
        let mut paper_props = HashMap::new();
        paper_props.insert("year".to_string(), "int".to_string());
        g.upsert_node_type_metadata("Paper", paper_props);
        g.upsert_connection_type_metadata("KNOWS", "Person", "Person", HashMap::new());
        g.upsert_connection_type_metadata("AUTHORED", "Person", "Paper", HashMap::new());
        g
    }

    #[test]
    fn validates_known_node_type() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person) RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn tolerates_unknown_node_type() {
        // `MATCH (n:Nonexistent) RETURN n` is valid Cypher — returns 0
        // rows. Phase 3 will surface a "did you mean?" hint via
        // diagnostics rather than rejecting.
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:person) RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn tolerates_unknown_connection_type() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (a:Person)-[:nonexistent]->(b:Person) RETURN a").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn warns_unknown_node_label_with_hint() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Persn) RETURN n").unwrap();
        let warnings = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(warnings.len(), 1, "got: {warnings:?}");
        assert!(warnings[0].contains("unknown node label 'Persn'"));
        assert!(
            warnings[0].contains("Did you mean 'Person'?"),
            "got: {}",
            warnings[0]
        );
    }

    #[test]
    fn warns_unknown_relationship_type_with_hint() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (a:Person)-[:KNOWZ]->(b:Person) RETURN a").unwrap();
        let warnings = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(warnings.len(), 1, "got: {warnings:?}");
        assert!(warnings[0].contains("unknown relationship type 'KNOWZ'"));
        assert!(
            warnings[0].contains("Did you mean 'KNOWS'?"),
            "got: {}",
            warnings[0]
        );
    }

    #[test]
    fn no_warning_for_secondary_label() {
        // A label applied via add_label is valid in MATCH even though it is no
        // node's primary type — must NOT be flagged as unknown (regression
        // guard for a false positive: the warning once claimed `:Reviewer`
        // was unknown and "returns no rows" while it returned rows).
        let mut g = graph_with_schema();
        g.secondary_label_index
            .entry(InternedKey::from_str("Reviewer"))
            .or_default();
        let q = parse_cypher("MATCH (n:Reviewer) RETURN n").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
        // And it still warns on a genuine typo of the secondary label.
        let q2 = parse_cypher("MATCH (n:Reviewr) RETURN n").unwrap();
        assert_eq!(collect_unknown_pattern_warnings(&q2, &g).len(), 1);
    }

    #[test]
    fn no_warning_for_known_label_and_relationship() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
    }

    #[test]
    fn no_warning_on_schemaless_graph() {
        // A fresh graph has nothing to compare against → never warns.
        let g = DirGraph::new();
        let q = parse_cypher("MATCH (n:Anything) RETURN n").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
    }

    #[test]
    fn rejects_unknown_property_in_pattern_literal() {
        // The `{agee: 30}` form is unambiguous — validate.
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person {agee: 30}) RETURN n").unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
        assert!(err.message.contains("age"), "got: {}", err.message);
    }

    #[test]
    fn tolerates_unknown_property_in_where_expression() {
        // `n.prop` accesses in WHERE may reference virtual columns
        // (timeseries sub-nodes, computed aliases). Do not flag.
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person) WHERE n.birth_yr = 1900 RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn tolerates_unknown_property_in_return_expression() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person) RETURN n.agee").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn allows_builtin_fields_in_pattern_literal() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person {id: 1}) RETURN n.title").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn skips_validation_on_empty_schema() {
        let g = DirGraph::new();
        let q = parse_cypher("MATCH (n:Anything) RETURN n.whatever").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn validates_untyped_patterns_permissively() {
        // Untyped patterns (no :Label) are common and legal — we only
        // validate when the user has declared intent via a label.
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n) WHERE n.whatever = 1 RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn rejects_unknown_property_on_multi_hop_pattern_literal() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person {agee: 30}) RETURN a").unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
    }

    #[test]
    fn allows_order_by_and_return_of_known_properties() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n:Person) RETURN n.age ORDER BY n.email").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn tolerates_unknown_label_in_where_label_check() {
        let g = graph_with_schema();
        let q = parse_cypher("MATCH (n) WHERE n:person RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn rejects_unknown_property_in_create_pattern_literal() {
        // CREATE typos slip through silently today: `CREATE (:Person {ttle: 'x'})`
        // adds a node with a `ttle` property instead of `title`. Pre-flight
        // validation surfaces the typo with a "did you mean?" hint.
        let g = graph_with_schema();
        let q = parse_cypher("CREATE (:Person {agee: 30})").unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
        assert!(err.message.contains("age"), "got: {}", err.message);
    }

    #[test]
    fn allows_known_property_in_create() {
        let g = graph_with_schema();
        let q = parse_cypher("CREATE (:Person {age: 30, email: 'a@b'})").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn rejects_unknown_property_in_create_multi_element_path() {
        // CREATE (a:Person {age: 30})-[:KNOWS]->(b:Person {agee: 25})
        // — typo on the second node.
        let g = graph_with_schema();
        let q =
            parse_cypher("CREATE (a:Person {age: 30})-[:KNOWS]->(b:Person {agee: 25}) RETURN a, b")
                .unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
    }

    #[test]
    fn rejects_unknown_property_in_merge_pattern_literal() {
        let g = graph_with_schema();
        let q = parse_cypher("MERGE (n:Person {agee: 30}) RETURN n").unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
    }

    #[test]
    fn allows_known_property_in_merge() {
        let g = graph_with_schema();
        let q = parse_cypher("MERGE (n:Person {email: 'a@b'}) RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn create_with_untyped_node_is_permissive() {
        // CREATE (n {anything: 1}) — no label, no type metadata to
        // check against. Permissive (matches MATCH behavior).
        let g = graph_with_schema();
        let q = parse_cypher("CREATE (n {anything_at_all: 1}) RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn create_on_unknown_node_type_is_permissive() {
        // Symmetric with `tolerates_unknown_node_type` for MATCH —
        // unknown labels are not rejected here either (consistent
        // rule: only validate when metadata is declared).
        let g = graph_with_schema();
        let q = parse_cypher("CREATE (n:NewType {whatever: 1}) RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn create_builtin_field_is_allowed() {
        let g = graph_with_schema();
        let q = parse_cypher("CREATE (n:Person {id: 99, title: 'Eve'}) RETURN n").unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn validates_property_inside_call_subquery_body() {
        // A pattern-literal typo inside a CALL { } body must be caught with
        // the same "did you mean?" quality as a top-level pattern.
        let g = graph_with_schema();
        let q = parse_cypher("CALL { MATCH (n:Person {agee: 1}) RETURN n.name AS nm } RETURN nm")
            .unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
        assert!(err.message.contains("age"), "got: {}", err.message);
    }

    #[test]
    fn validates_labeled_pattern_literal_inside_correlated_call_body() {
        // A correlated body that introduces a NEW labeled node with a
        // property typo is caught (same rule as a top-level labeled
        // pattern literal). `f:Person {agee:...}` inside the body is
        // invalid.
        let g = graph_with_schema();
        let q = parse_cypher(
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f:Person {agee: 1}) RETURN count(f) AS c } RETURN p.name, c",
        )
        .unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
        assert!(err.message.contains("age"), "got: {}", err.message);
    }

    #[test]
    fn call_subquery_body_non_imported_var_is_fresh_scope() {
        // A bare body variable that shadows an outer name is a FRESH
        // variable (§1.2 rule 1) — without the import, the body's `n` has
        // no declared type, so a property reference is permissively allowed
        // (matches top-level untyped behaviour). No false positive.
        let g = graph_with_schema();
        let q = parse_cypher(
            "MATCH (p:Person) CALL { MATCH (n) RETURN n.anything AS a } RETURN p.name, a",
        )
        .unwrap();
        assert!(validate_schema(&q, &g).is_ok());
    }

    #[test]
    fn validates_property_inside_exists_nested_pattern() {
        // Pattern literals inside EXISTS { MATCH ... } should be
        // validated too.
        let g = graph_with_schema();
        let q = parse_cypher(
            "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->(b:Person {agee: 1}) } RETURN a",
        )
        .unwrap();
        let err = validate_schema(&q, &g).unwrap_err();
        assert_eq!(err.kind, SchemaErrorKind::UnknownProperty);
    }

    // ── A1b: WHERE-clause absent-property warnings (non-fatal) ──────────────

    #[test]
    fn warns_on_where_property_absent_from_label() {
        let g = graph_with_schema();
        // `is_external` is not a Person property → the comparison is always
        // null/false and filters everything; warn (non-fatal) + did-you-mean.
        let q = parse_cypher("MATCH (p:Person) WHERE p.is_external = false RETURN p").unwrap();
        let w = collect_unknown_pattern_warnings(&q, &g);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(
            w[0].contains("is_external") && w[0].contains("Person"),
            "{}",
            w[0]
        );
        // A near-miss still gets a suggestion.
        let q2 = parse_cypher("MATCH (p:Person) WHERE p.agee = 1 RETURN p").unwrap();
        let w2 = collect_unknown_pattern_warnings(&q2, &g);
        assert!(
            w2.iter().any(|m| m.contains("Did you mean 'age'")),
            "{w2:?}"
        );
    }

    #[test]
    fn no_warning_on_present_or_builtin_property() {
        let g = graph_with_schema();
        // Declared property → no warning.
        let q = parse_cypher("MATCH (p:Person) WHERE p.age = 30 RETURN p").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
        // Built-in field → no warning.
        let q2 = parse_cypher("MATCH (p:Person) WHERE p.id = 1 RETURN p").unwrap();
        assert!(collect_unknown_pattern_warnings(&q2, &g).is_empty());
    }

    #[test]
    fn no_warning_on_untyped_var() {
        let g = graph_with_schema();
        // No label on the var → can't reason about its properties → no warning
        // (avoids false positives on dynamically-typed graphs).
        let q = parse_cypher("MATCH (n) WHERE n.whatever = 1 RETURN n").unwrap();
        assert!(collect_unknown_pattern_warnings(&q, &g).is_empty());
    }
}
