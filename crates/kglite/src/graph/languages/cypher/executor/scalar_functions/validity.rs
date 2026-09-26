//! `valid_at()` and `valid_during()`: an element's validity interval in Cypher.
//!
//! Two forms each:
//!
//! - `valid_at(x, t)` / `valid_during(x, a, b)` read the bounds and the
//!   convention from the declaration of `x`'s type (`db.temporal.declare`, a
//!   loader's `validFrom`/`validTo`, `set_temporal`) — the declaration the
//!   fluent `select()`/`traverse()` filters follow. A type with no declaration
//!   is an error naming the fix.
//! - `valid_at(x, t, 'from', 'to')` / `valid_during(x, a, b, 'from', 'to')`
//!   name the bounds. When the type's declaration names the same two
//!   properties its convention applies, so a half-open declaration answers
//!   half-open here as it does fluently; an undeclared pair reads closed.
//!
//! A relationship's declaration is looked up the way the fluent filters look
//! it up: its source node type's keyed declaration first, then the unkeyed
//! ones. A node's is its primary type's, then a secondary label's.

use super::super::*;
use crate::datatypes::values::Value;
use crate::graph::core::value_operations::format_value_compact;
use crate::graph::features::temporal::eval::{
    self as temporal_eval, BoundSide, Instant, IntervalConvention, TemporalError,
};
use crate::graph::property_types::value_type_name;
use crate::graph::schema::{InternedKey, TemporalConfig};
use crate::graph::storage::GraphRead;

/// The element type one validity call reads, and the bound names it asked
/// for (`None` for the two-argument form): what its resolved bounds depend on.
#[derive(Clone, PartialEq, Eq)]
pub(in crate::graph::languages::cypher::executor) struct ValidityKey {
    /// A node's primary type, or a relationship's type and its source's type.
    element: (InternedKey, Option<InternedKey>),
    named: Option<(String, String)>,
}

/// The first validity call site's resolution, reused by every row of the same
/// element type — one declaration lookup per site, not per row. A row of
/// another type resolves uncached.
pub(in crate::graph::languages::cypher::executor) struct ValidityCache {
    key: ValidityKey,
    resolved: Option<ResolvedBounds>,
}

/// The two bound properties and the convention a call evaluates under.
#[derive(Clone)]
struct ResolvedBounds {
    from: String,
    to: String,
    convention: IntervalConvention,
}

/// The element a validity call's variable is bound to.
struct Element {
    key: (InternedKey, Option<InternedKey>),
    /// `node type 'M'` / `relationship type 'R'`, for messages.
    describe: String,
    /// The node label or relationship type a declaration would name.
    target: String,
    kind: ElementKind,
}

enum ElementKind {
    Node(petgraph::graph::NodeIndex),
    /// A relationship, by its endpoints (its errors name them).
    Edge {
        source: petgraph::graph::NodeIndex,
        target: petgraph::graph::NodeIndex,
    },
}

/// What a validity call's first argument holds on one row.
enum Subject {
    Element(Element),
    /// Null — an unmatched `OPTIONAL MATCH`, a null list item: the call is null.
    Null,
    /// Some other value (a map, a path): only the named form reads it.
    Other,
}

impl CypherExecutor<'_> {
    /// `valid_at(entity, date)` or `valid_at(entity, date, 'from', 'to')`.
    pub(super) fn eval_valid_at(
        &self,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        if args.len() != 2 && args.len() != 4 {
            return Err(
                "valid_at() takes (entity, date) on a type with a declared validity \
                        interval, or (entity, date, from_field, to_field)"
                    .into(),
            );
        }
        let var_name = validity_variable("valid_at", &args[0])?;
        let subject = self.validity_subject(var_name, row);
        if let Subject::Null = subject {
            return Ok(Value::Null);
        }
        let date_val = self.evaluate_expression(&args[1], row)?;
        let Some(resolved) =
            self.resolve_validity("valid_at", var_name, &subject, &args[2..], row)?
        else {
            return Ok(Value::Boolean(true));
        };
        let bounds = self.validity_bounds("valid_at", var_name, &subject, &resolved, row)?;
        let instant = bounds.instant(&date_val, "date")?;
        temporal_eval::interval_contains(
            &bounds.from_val,
            &bounds.to_val,
            instant,
            resolved.convention,
        )
        .map(Value::Boolean)
        .map_err(|e| self.validity_bound_error(&bounds, &subject, e))
    }

    /// `valid_during(entity, start, end)` or
    /// `valid_during(entity, start, end, 'from', 'to')`: whether the interval
    /// overlaps `[start, end]`.
    pub(super) fn eval_valid_during(
        &self,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        if args.len() != 3 && args.len() != 5 {
            return Err(
                "valid_during() takes (entity, start, end) on a type with a declared \
                        validity interval, or (entity, start, end, from_field, to_field)"
                    .into(),
            );
        }
        let var_name = validity_variable("valid_during", &args[0])?;
        let subject = self.validity_subject(var_name, row);
        if let Subject::Null = subject {
            return Ok(Value::Null);
        }
        let start_val = self.evaluate_expression(&args[1], row)?;
        let end_val = self.evaluate_expression(&args[2], row)?;
        let Some(resolved) =
            self.resolve_validity("valid_during", var_name, &subject, &args[3..], row)?
        else {
            return Ok(Value::Boolean(true));
        };
        let bounds = self.validity_bounds("valid_during", var_name, &subject, &resolved, row)?;
        let start = bounds.instant(&start_val, "start")?;
        let end = bounds.instant(&end_val, "end")?;
        temporal_eval::interval_overlaps(
            &bounds.from_val,
            &bounds.to_val,
            start,
            end,
            resolved.convention,
        )
        .map(Value::Boolean)
        .map_err(|e| self.validity_bound_error(&bounds, &subject, e))
    }

    /// The bounds and convention one row evaluates under. `named` is empty for
    /// the declared form and holds the two property-name arguments otherwise.
    ///
    /// `Ok(None)` — the row is valid — only for a relationship whose type has
    /// several unkeyed declarations of which it carries none, which the fluent
    /// `traverse()` filter also treats as not temporal.
    fn resolve_validity(
        &self,
        function: &'static str,
        var_name: &str,
        subject: &Subject,
        named: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<ResolvedBounds>, String> {
        let named = match named {
            [] => None,
            [from, to] => Some((
                self.bound_name(function, "from_field", from, row)?,
                self.bound_name(function, "to_field", to, row)?,
            )),
            _ => unreachable!("arity checked by the caller"),
        };
        let Subject::Element(element) = subject else {
            // A value that is not a node or relationship (a map): the named
            // form reads its two keys, closed; the declared form has no type.
            return match named {
                Some((from, to)) => Ok(Some(ResolvedBounds {
                    from,
                    to,
                    convention: IntervalConvention::Closed,
                })),
                None => Err(format!(
                    "{function}(): first argument must be a node or relationship; \
                     {var_name} is neither"
                )),
            };
        };
        let key = ValidityKey {
            element: element.key,
            named: named.clone(),
        };
        if let Some(cache) = self.validity_cache.get() {
            if cache.key == key {
                return match (&cache.resolved, named) {
                    (Some(resolved), _) => Ok(Some(resolved.clone())),
                    (None, Some((from, to))) => Ok(Some(ResolvedBounds {
                        from,
                        to,
                        convention: IntervalConvention::Closed,
                    })),
                    (None, None) => Err(undeclared_error(function, var_name, element)),
                };
            }
        }
        let candidates = self.declared_configs(element);
        let (resolved, cacheable) = match &named {
            Some((from, to)) => (
                candidates
                    .iter()
                    .find(|c| &c.valid_from == from && &c.valid_to == to)
                    .map(|c| resolved_from(c)),
                true,
            ),
            None => self.declared_choice(var_name, element, &candidates, row),
        };
        if cacheable {
            let _ = self.validity_cache.set(ValidityCache {
                key,
                resolved: resolved.clone(),
            });
        }
        match (resolved, named) {
            (Some(resolved), _) => Ok(Some(resolved)),
            (None, Some((from, to))) => Ok(Some(ResolvedBounds {
                from,
                to,
                convention: IntervalConvention::Closed,
            })),
            (None, None) if !cacheable => Ok(None),
            (None, None) => Err(undeclared_error(function, var_name, element)),
        }
    }

    /// The declaration a two-argument call follows, and whether that choice
    /// holds for every row of the element type. A relationship type with
    /// several unkeyed declarations and none keyed to this source picks per
    /// row, the first whose bounds the relationship carries — the fluent
    /// filter's rule — so its answer is not cached, and `None` there means
    /// "carries none".
    fn declared_choice(
        &self,
        var_name: &str,
        element: &Element,
        candidates: &[&TemporalConfig],
        row: &ResultRow,
    ) -> (Option<ResolvedBounds>, bool) {
        let ElementKind::Edge { .. } = element.kind else {
            return (candidates.first().map(|c| resolved_from(c)), true);
        };
        if let Some(keyed) = candidates.iter().find(|c| c.source_type.is_some()) {
            return (Some(resolved_from(keyed)), true);
        }
        if candidates.len() <= 1 {
            return (candidates.first().map(|c| resolved_from(c)), true);
        }
        let carried = candidates.iter().find(|c| {
            [&c.valid_from, &c.valid_to].into_iter().any(|property| {
                !matches!(
                    self.resolve_property(var_name, property, row),
                    Ok(Value::Null) | Err(_)
                )
            })
        });
        (carried.map(|c| resolved_from(c)), false)
    }

    /// Every declaration that could apply to `element`, in lookup order.
    fn declared_configs(&self, element: &Element) -> Vec<&TemporalConfig> {
        use crate::graph::features::temporal::{edge_configs, node_config};
        let graph = self.graph;
        match element.kind {
            ElementKind::Node(idx) => {
                let mut out: Vec<&TemporalConfig> =
                    node_config(graph, &element.target).into_iter().collect();
                let mut secondary: Vec<&str> = graph
                    .secondary_labels(idx)
                    .into_iter()
                    .map(|label| graph.interner.resolve(label))
                    .collect();
                secondary.sort_unstable();
                out.extend(
                    secondary
                        .into_iter()
                        .filter_map(|label| node_config(graph, label)),
                );
                out
            }
            ElementKind::Edge { .. } => {
                let configs = edge_configs(graph, &element.target);
                let source = element.key.1.map(|key| graph.interner.resolve(key));
                let keyed = configs
                    .iter()
                    .filter(|c| c.source_type.is_some() && c.source_type.as_deref() == source);
                let unkeyed = configs.iter().filter(|c| c.source_type.is_none());
                keyed.chain(unkeyed).collect()
            }
        }
    }

    /// What `var_name` holds on this row, resolved in `resolve_property`'s
    /// order so the element whose declaration is read is the one whose bounds
    /// are: a matched binding, else a projected node or relationship value — an
    /// item of `collect()`, `UNWIND`, `nodes(p)`, `relationships(p)` or a
    /// variable-length list carries its type as the bound variable does.
    fn validity_subject(&self, var_name: &str, row: &ResultRow) -> Subject {
        if let Some(&idx) = row.node_bindings.get(var_name) {
            return self.node_subject(idx, None);
        }
        if let Some(edge) = row.edge_bindings.get(var_name) {
            let Some(weight) = self.graph.graph.edge_weight(edge.edge_index) else {
                return Subject::Null;
            };
            let rel_type = weight.connection_type_str(&self.graph.interner).to_string();
            let (source, target) = self
                .graph
                .graph
                .edge_endpoints(edge.edge_index)
                .unwrap_or((edge.source, edge.target));
            return self.edge_subject(rel_type, source, target);
        }
        if row.path_bindings.contains_key(var_name) {
            return Subject::Other;
        }
        match row.projected.get(var_name) {
            None | Some(Value::Null) => Subject::Null,
            Some(Value::NodeRef(idx)) => {
                self.node_subject(petgraph::graph::NodeIndex::new(*idx as usize), None)
            }
            Some(Value::Node(node)) => self.node_subject(
                petgraph::graph::NodeIndex::new(node.id as usize),
                node.labels.first().map(String::as_str),
            ),
            Some(Value::Relationship(rel)) => self.edge_subject(
                rel.rel_type.clone(),
                petgraph::graph::NodeIndex::new(rel.start_id as usize),
                petgraph::graph::NodeIndex::new(rel.end_id as usize),
            ),
            Some(_) => Subject::Other,
        }
    }

    /// A node, typed from the graph — or from the value's own primary label
    /// when the graph no longer holds it.
    fn node_subject(&self, idx: petgraph::graph::NodeIndex, label: Option<&str>) -> Subject {
        let graph = self.graph;
        let node_type = match (graph.graph.node_type_of(idx), label) {
            (Some(key), _) => key,
            (None, Some(label)) => InternedKey::from_str(label),
            (None, None) => return Subject::Null,
        };
        let target = graph
            .interner
            .try_resolve(node_type)
            .or(label)
            .unwrap_or("?")
            .to_string();
        Subject::Element(Element {
            key: (node_type, None),
            describe: format!("node type '{target}'"),
            target,
            kind: ElementKind::Node(idx),
        })
    }

    fn edge_subject(
        &self,
        rel_type: String,
        source: petgraph::graph::NodeIndex,
        target: petgraph::graph::NodeIndex,
    ) -> Subject {
        Subject::Element(Element {
            key: (
                InternedKey::from_str(&rel_type),
                self.graph.graph.node_type_of(source),
            ),
            describe: format!("relationship type '{rel_type}'"),
            target: rel_type,
            kind: ElementKind::Edge { source, target },
        })
    }

    fn bound_name(
        &self,
        function: &str,
        argument: &str,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<String, String> {
        match self.evaluate_expression(expr, row)? {
            Value::String(s) => Ok(s),
            _ => Err(format!("{function}(): {argument} must be a string")),
        }
    }

    /// Read the row's two bounds under `resolved`, refusing a bound property
    /// the element's type does not have at all.
    fn validity_bounds<'n>(
        &self,
        function: &'static str,
        var_name: &'n str,
        subject: &Subject,
        resolved: &'n ResolvedBounds,
        row: &ResultRow,
    ) -> Result<ValidityBounds<'n>, String> {
        let bounds = ValidityBounds {
            function,
            var_name,
            from_field: &resolved.from,
            from_val: self.resolve_property(var_name, &resolved.from, row)?,
            to_field: &resolved.to,
            to_val: self.resolve_property(var_name, &resolved.to, row)?,
        };
        if let Subject::Element(element) = subject {
            self.require_known_bounds(&bounds, element)?;
        }
        Ok(bounds)
    }

    /// Refuse a null bound read from a property the element's type does not
    /// have at all. A null bound is open, so a misspelled name (`'validfrom'`)
    /// used to answer every row as unbounded on that side, silently. A property
    /// the type records but this row leaves null stays open — the same test
    /// `db.temporal.declare` applies to the two names.
    fn require_known_bounds(
        &self,
        bounds: &ValidityBounds<'_>,
        element: &Element,
    ) -> Result<(), String> {
        for (field, value) in [
            (bounds.from_field, &bounds.from_val),
            (bounds.to_field, &bounds.to_val),
        ] {
            if !matches!(value, Value::Null) || self.element_has_property(element, field) {
                continue;
            }
            let kind = match element.kind {
                ElementKind::Node(_) => "node type",
                ElementKind::Edge { .. } => "relationship type",
            };
            return Err(crate::graph::features::temporal::unknown_bound_message(
                bounds.function,
                kind,
                &element.target,
                field,
            ));
        }
        Ok(())
    }

    fn element_has_property(&self, element: &Element, field: &str) -> bool {
        use crate::graph::features::temporal::{
            node_type_has_property, relationship_type_has_property,
        };
        match element.kind {
            ElementKind::Node(_) => node_type_has_property(self.graph, &element.target, field),
            ElementKind::Edge { .. } => {
                relationship_type_has_property(self.graph, &element.target, field)
            }
        }
    }

    /// The error for a stored bound the evaluator cannot read, naming the
    /// element by its user id (a node's `id`, a relationship's endpoints).
    fn validity_bound_error(
        &self,
        bounds: &ValidityBounds<'_>,
        subject: &Subject,
        err: TemporalError,
    ) -> String {
        let field = match &err {
            TemporalError::Bound {
                side: BoundSide::To,
                ..
            } => bounds.to_field,
            _ => bounds.from_field,
        };
        let id = |idx| {
            self.graph
                .graph
                .get_node_id(idx)
                .map_or_else(|| "?".to_string(), |v| format_value_compact(&v))
        };
        let element = match subject {
            Subject::Element(Element {
                kind: ElementKind::Node(idx),
                ..
            }) => format!("node '{}'", id(*idx)),
            Subject::Element(Element {
                kind: ElementKind::Edge { source, target },
                ..
            }) => format!("relationship from '{}' to '{}'", id(*source), id(*target)),
            _ => format!("'{}'", bounds.var_name),
        };
        format!(
            "{}(): {}.{field} on {element}: {err}. Store bounds as date() or datetime() values, \
             or as ISO strings such as '2009-06-30'",
            bounds.function, bounds.var_name
        )
    }
}

fn resolved_from(config: &TemporalConfig) -> ResolvedBounds {
    ResolvedBounds {
        from: config.valid_from.clone(),
        to: config.valid_to.clone(),
        convention: config.convention,
    }
}

fn validity_variable<'e>(function: &str, arg: &'e Expression) -> Result<&'e str, String> {
    match arg {
        Expression::Variable(v) => Ok(v),
        _ => Err(format!(
            "{function}(): first argument must be a node or relationship variable"
        )),
    }
}

/// The two-argument form on a type nothing declared.
fn undeclared_error(function: &str, var_name: &str, element: &Element) -> String {
    let key = match element.kind {
        ElementKind::Node(_) => "node",
        ElementKind::Edge { .. } => "relationship",
    };
    let rest = if function == "valid_during" {
        "start, end"
    } else {
        "date"
    };
    format!(
        "{function}({var_name}, {rest}): {} has no declared validity interval, so there are no \
         bounds to read. Declare one — CALL db.temporal.declare({{{key}: '{}', from: \
         'valid_from', to: 'valid_to', convention: 'half_open'}}) — or name the bounds: \
         {function}({var_name}, {rest}, 'valid_from', 'valid_to')",
        element.describe, element.target
    )
}

/// One `valid_at` / `valid_during` call's element bounds, for its errors.
struct ValidityBounds<'a> {
    function: &'static str,
    var_name: &'a str,
    from_field: &'a str,
    from_val: Value,
    to_field: &'a str,
    to_val: Value,
}

impl ValidityBounds<'_> {
    /// Read a query instant, or explain which argument could not be compared
    /// with which bound — both type names — and how to write it instead.
    fn instant(&self, value: &Value, argument: &str) -> Result<Instant, String> {
        temporal_eval::parse_instant(value).map_err(|err| {
            let (field, bound) =
                if matches!(self.from_val, Value::Null) && !matches!(self.to_val, Value::Null) {
                    (self.to_field, &self.to_val)
                } else {
                    (self.from_field, &self.from_val)
                };
            format!(
                "{}(): the {argument} argument {err}, so it cannot be compared with {}.{field} ({}). \
                 Pass a date such as date('2009'), a datetime, or an ISO string such as '2009-06-30'",
                self.function,
                self.var_name,
                value_type_name(bound)
            )
        })
    }
}
