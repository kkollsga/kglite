//! Streaming hash aggregate.
//!
//! Builds per-group state inline as upstream rows arrive. Replaces the
//! materialize-then-bucket path in
//! [`super::super::CypherExecutor::execute_return_with_aggregation`]
//! for queries the planner-recognition function in [`super::pipeline`]
//! can absorb. Same I/O profile as the materialized path because we
//! preserve the NodeIndex-surrogate trick: group-key property reads are
//! deferred until the post-scan resolution pass, so the disk I/O is
//! O(distinct groups) rather than O(rows).
//!
//! # Supported aggregates
//! `count(*)`, `count(expr)`, `count(DISTINCT expr)`, and
//! `sum/avg/min/max[(DISTINCT) expr]`. The recognition function refuses
//! to build the streaming path for other aggregates (`collect`, `std`,
//! `variance`, `median`, `percentile_*`) or for items that are neither
//! pure variable / property access nor a single aggregate function call
//! — the materialized executor handles those as before.

use super::super::super::ast::{is_aggregate_expression, Expression, ReturnClause};
use super::super::super::result::{Bindings, ResultRow};
use super::super::helpers::return_item_column_name;
use super::super::CypherExecutor;
use super::RowStream;
use crate::datatypes::values::Value;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

/// Surrogate key for a single grouping expression. Mirrors the
/// `GroupKeyPart` enum in `return_clause` so the streaming path
/// preserves the same per-group node-property read deduplication.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum GroupKeyPart {
    /// Bound-node property access — resolve later, once per (NodeIndex, slot).
    NodeProp(NodeIndex),
    /// Pre-evaluated value (anything that isn't a node-binding property
    /// access, or where the variable wasn't a node binding for this row).
    Resolved(Value),
}

/// Per-grouping-expression strategy chosen once before iterating rows.
enum GroupExprStrategy {
    NodeProp { variable: String },
    Eval,
}

impl GroupExprStrategy {
    fn for_expr(expr: &Expression) -> Self {
        if let Expression::PropertyAccess { variable, .. } = expr {
            Self::NodeProp {
                variable: variable.clone(),
            }
        } else {
            Self::Eval
        }
    }
}

/// What kind of aggregate the streaming path knows how to compute
/// inline. Anything else causes the recognition function to bail.
#[derive(Clone, Copy)]
enum AggKind {
    CountStar,
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// Compiled aggregate spec — one per aggregate column in the RETURN/WITH.
pub(crate) struct AggSpec {
    kind: AggKind,
    /// Argument expression. `None` only for `count(*)`. Held as owned
    /// `Expression` so the caller can hand us folded forms without
    /// keeping its own arena alive.
    arg: Option<Expression>,
    /// True for `count(DISTINCT …)`, `sum(DISTINCT …)`, etc.
    distinct: bool,
    /// True when the argument is a node variable — enables the cheap
    /// NodeIndex-based DISTINCT path used by the materialized executor.
    arg_is_node_var: Option<String>,
    /// True when the argument is an edge variable — enables EdgeIndex DISTINCT.
    arg_is_edge_var: Option<String>,
}

/// State kept per group per aggregate. The variants stay narrow so the
/// hot loop branches predictably; complex aggregates are out of scope
/// for the streaming path.
struct AggState {
    count: i64,
    sum: f64,
    sum_was_int: bool,
    sum_seen_value: bool,
    min: Option<Value>,
    max: Option<Value>,
    /// Populated only when the corresponding `AggSpec` has `distinct`.
    distinct_nodes: Option<HashSet<usize>>,
    distinct_edges: Option<HashSet<usize>>,
    distinct_values: Option<HashSet<Value>>,
}

impl AggState {
    fn new(spec: &AggSpec) -> Self {
        let (distinct_nodes, distinct_edges, distinct_values) = if spec.distinct {
            (
                Some(HashSet::new()),
                Some(HashSet::new()),
                Some(HashSet::new()),
            )
        } else {
            (None, None, None)
        };
        AggState {
            count: 0,
            sum: 0.0,
            sum_was_int: true,
            sum_seen_value: false,
            min: None,
            max: None,
            distinct_nodes,
            distinct_edges,
            distinct_values,
        }
    }

    /// Apply `value` to this state for `spec`. Caller has already
    /// resolved any DISTINCT skip-decision via `should_count`.
    fn record(&mut self, value: Option<Value>, spec: &AggSpec) {
        // count(*) takes no value — caller passes `None`. Other aggregates
        // skip nulls, matching the materialized executor.
        let val = match value {
            Some(v) if !matches!(v, Value::Null) => v,
            None if matches!(spec.kind, AggKind::CountStar) => {
                self.count += 1;
                return;
            }
            _ => return,
        };
        match spec.kind {
            AggKind::CountStar | AggKind::Count => {
                self.count += 1;
            }
            AggKind::Sum | AggKind::Avg => {
                if let Some(f) = value_to_f64(&val) {
                    self.sum += f;
                    self.count += 1;
                    self.sum_seen_value = true;
                    // Materialized `evaluate_aggregate_with_rows` for
                    // `sum` calls `probe_source_type_is_int` which only
                    // returns true for `Value::Int64(_)` — UniqueId and
                    // other numeric variants force the result to
                    // Float64. Match that to keep the streaming result
                    // shape identical.
                    if !matches!(val, Value::Int64(_)) {
                        self.sum_was_int = false;
                    }
                }
            }
            AggKind::Min => {
                self.min = Some(match self.min.take() {
                    None => val,
                    Some(current) => {
                        if cmp_lt(&val, &current) {
                            val
                        } else {
                            current
                        }
                    }
                });
            }
            AggKind::Max => {
                self.max = Some(match self.max.take() {
                    None => val,
                    Some(current) => {
                        if cmp_gt(&val, &current) {
                            val
                        } else {
                            current
                        }
                    }
                });
            }
        }
    }

    /// Merge `other` into self when the post-scan re-bucket pass collapses
    /// two NodeIndex surrogate groups into one resolved-value group.
    fn merge(&mut self, other: AggState) {
        self.count += other.count;
        self.sum += other.sum;
        if other.sum_seen_value {
            self.sum_seen_value = true;
            if !other.sum_was_int {
                self.sum_was_int = false;
            }
        }
        self.min = combine(self.min.take(), other.min, false);
        self.max = combine(self.max.take(), other.max, true);
        if let (Some(a), Some(b)) = (self.distinct_nodes.as_mut(), other.distinct_nodes) {
            a.extend(b);
        }
        if let (Some(a), Some(b)) = (self.distinct_edges.as_mut(), other.distinct_edges) {
            a.extend(b);
        }
        if let (Some(a), Some(b)) = (self.distinct_values.as_mut(), other.distinct_values) {
            a.extend(b);
        }
    }

    /// Produce the final `Value` for this state given its `spec`.
    fn finalize(&self, spec: &AggSpec) -> Value {
        match spec.kind {
            AggKind::CountStar => Value::Int64(self.count),
            AggKind::Count => {
                if spec.distinct {
                    let n = self.distinct_nodes.as_ref().map(|s| s.len()).unwrap_or(0)
                        + self.distinct_edges.as_ref().map(|s| s.len()).unwrap_or(0)
                        + self.distinct_values.as_ref().map(|s| s.len()).unwrap_or(0);
                    Value::Int64(n as i64)
                } else {
                    Value::Int64(self.count)
                }
            }
            AggKind::Sum => {
                if !self.sum_seen_value {
                    Value::Int64(0)
                } else if self.sum_was_int && self.sum.fract() == 0.0 {
                    Value::Int64(self.sum as i64)
                } else {
                    Value::Float64(self.sum)
                }
            }
            AggKind::Avg => {
                if self.count == 0 {
                    Value::Null
                } else {
                    Value::Float64(self.sum / self.count as f64)
                }
            }
            AggKind::Min => self.min.clone().unwrap_or(Value::Null),
            AggKind::Max => self.max.clone().unwrap_or(Value::Null),
        }
    }
}

/// Per-surrogate-group entry: parallel `Vec`s sized by the number of
/// aggregate specs.
struct GroupAcc {
    states: Vec<AggState>,
    /// Captured node-binding for the first row in this group, by
    /// variable name. Used to preserve `node_bindings` on the output
    /// row so downstream MATCH/OPTIONAL MATCH clauses can constrain
    /// patterns to the same nodes (matches existing materialized
    /// behavior in `execute_return_with_aggregation`).
    first_node_bindings: Bindings<NodeIndex>,
}

impl GroupAcc {
    fn new(specs: &[AggSpec]) -> Self {
        GroupAcc {
            states: specs.iter().map(AggState::new).collect(),
            first_node_bindings: Bindings::new(),
        }
    }
}

/// Recognition / construction errors. The pipeline builder converts these
/// into a `None` (bail to materialized executor) rather than propagating.
#[derive(Debug)]
pub enum AggregateBail {
    /// Saw an aggregate expression the streaming path doesn't know how
    /// to compute inline (e.g. `collect`, `std`, arithmetic on aggregates).
    UnsupportedAggregate,
    /// Saw a non-aggregate item that isn't a pure variable / property
    /// access — the materialized path handles arbitrary scalar
    /// projection but the streaming path keeps things simple.
    UnsupportedItem,
}

/// `(group_indices, agg_indices, specs)` produced by
/// [`try_compile_specs`]. Parallel arrays — `group_indices[k]` /
/// `agg_indices[k]` index into `return_clause.items`; `specs[k]`
/// describes the aggregate at `agg_indices[k]`.
pub(crate) type CompiledSpecs = (Vec<usize>, Vec<usize>, Vec<AggSpec>);

/// Attempt to compile a return-clause shape into an [`AggSpec`] vector.
/// Returns `Err(AggregateBail::…)` if any item is outside the streaming
/// path's reach, in which case the caller must fall back to the
/// materialized executor.
pub fn try_compile_specs(return_clause: &ReturnClause) -> Result<CompiledSpecs, AggregateBail> {
    let mut group_indices = Vec::new();
    let mut agg_indices = Vec::new();
    let mut specs = Vec::new();

    for (i, item) in return_clause.items.iter().enumerate() {
        if is_aggregate_expression(&item.expression) {
            let spec = compile_agg(&item.expression)?;
            agg_indices.push(i);
            specs.push(spec);
        } else {
            // Non-aggregate items in a streaming aggregate must resolve
            // to a value the surrogate-key trick can recover from a
            // bound row. Variables and property accesses qualify;
            // arithmetic and function calls do not.
            match &item.expression {
                Expression::Variable(_) | Expression::PropertyAccess { .. } => {
                    group_indices.push(i);
                }
                _ => return Err(AggregateBail::UnsupportedItem),
            }
        }
    }

    Ok((group_indices, agg_indices, specs))
}

fn compile_agg(expr: &Expression) -> Result<AggSpec, AggregateBail> {
    let (name, args, distinct) = match expr {
        Expression::FunctionCall {
            name,
            args,
            distinct,
        } => (name.as_str(), args, *distinct),
        _ => return Err(AggregateBail::UnsupportedAggregate),
    };

    let kind = match name {
        "count" => {
            if args.len() == 1 && matches!(args[0], Expression::Star) {
                AggKind::CountStar
            } else {
                AggKind::Count
            }
        }
        "sum" => AggKind::Sum,
        "avg" | "mean" | "average" => AggKind::Avg,
        "min" => AggKind::Min,
        "max" => AggKind::Max,
        _ => return Err(AggregateBail::UnsupportedAggregate),
    };

    let arg = if matches!(kind, AggKind::CountStar) {
        None
    } else if args.len() == 1 {
        Some(args[0].clone())
    } else {
        return Err(AggregateBail::UnsupportedAggregate);
    };

    let arg_is_node_var = arg.as_ref().and_then(|a| match a {
        Expression::Variable(v) => Some(v.clone()),
        _ => None,
    });

    Ok(AggSpec {
        kind,
        arg,
        distinct,
        arg_is_node_var,
        arg_is_edge_var: None,
    })
}

/// Apply the streaming aggregate to `upstream`, returning a new
/// `RowStream` whose iterator yields one row per group (post-DISTINCT,
/// if any).
pub fn apply<'q>(
    executor: &'q CypherExecutor<'q>,
    upstream: RowStream<'q>,
    return_clause: &ReturnClause,
    group_indices: &[usize],
    agg_indices: &[usize],
    specs: &[AggSpec],
) -> Result<RowStream<'q>, String> {
    // Pre-fold group key expressions once; cuts per-row constant evaluation.
    let folded_group_exprs: Vec<Expression> = group_indices
        .iter()
        .map(|&i| executor.fold_constants_expr(&return_clause.items[i].expression))
        .collect();

    let strategies: Vec<GroupExprStrategy> = folded_group_exprs
        .iter()
        .map(GroupExprStrategy::for_expr)
        .collect();

    // Pre-fold aggregate argument expressions too.
    let folded_args: Vec<Option<Expression>> = specs
        .iter()
        .map(|s| s.arg.as_ref().map(|e| executor.fold_constants_expr(e)))
        .collect();

    // Single-pass over the upstream iterator. Surrogate groups: keyed by
    // `Vec<GroupKeyPart>` (NodeProp surrogates + resolved values).
    // Equivalent to the materialized path's first pass at
    // `return_clause::execute_return_with_aggregation` ~lines 242-273.
    let mut surrogate_groups: Vec<(Vec<GroupKeyPart>, GroupAcc)> = Vec::new();
    let mut surrogate_index: HashMap<Vec<GroupKeyPart>, usize> = HashMap::new();

    // Capture variable names of group-key expressions that are pure
    // variable references — used to copy node bindings forward on the
    // first row of each group.
    let group_var_names: Vec<Option<String>> = group_indices
        .iter()
        .map(|&i| match &return_clause.items[i].expression {
            Expression::Variable(v) => Some(v.clone()),
            Expression::PropertyAccess { variable, .. } => Some(variable.clone()),
            _ => None,
        })
        .collect();

    let mut row_count = 0u64;
    for row_result in upstream {
        let row = row_result?;
        row_count += 1;
        if row_count.is_multiple_of(4096) {
            executor.check_deadline()?;
        }

        let key_parts: Vec<GroupKeyPart> = strategies
            .iter()
            .zip(folded_group_exprs.iter())
            .map(|(strategy, expr)| match strategy {
                GroupExprStrategy::NodeProp { variable } => {
                    if let Some(&idx) = row.node_bindings.get(variable) {
                        GroupKeyPart::NodeProp(idx)
                    } else {
                        GroupKeyPart::Resolved(
                            executor
                                .evaluate_expression(expr, &row)
                                .unwrap_or(Value::Null),
                        )
                    }
                }
                GroupExprStrategy::Eval => GroupKeyPart::Resolved(
                    executor
                        .evaluate_expression(expr, &row)
                        .unwrap_or(Value::Null),
                ),
            })
            .collect();

        let group_idx = match surrogate_index.get(&key_parts) {
            Some(&idx) => idx,
            None => {
                let idx = surrogate_groups.len();
                surrogate_index.insert(key_parts.clone(), idx);
                let mut acc = GroupAcc::new(specs);
                // Capture node bindings for variables that are
                // group-key expressions (pure Variable or PropertyAccess).
                for var_opt in group_var_names.iter().flatten() {
                    if let Some(&node_idx) = row.node_bindings.get(var_opt) {
                        acc.first_node_bindings.insert(var_opt.clone(), node_idx);
                    }
                }
                surrogate_groups.push((key_parts, acc));
                idx
            }
        };

        // Update each aggregate's state for this row.
        let acc = &mut surrogate_groups[group_idx].1;
        for (ai, spec) in specs.iter().enumerate() {
            update_agg_state(
                &mut acc.states[ai],
                spec,
                folded_args[ai].as_ref(),
                &row,
                executor,
            );
        }
    }

    // Resolve NodeProp surrogates and re-bucket. Mirrors the second
    // half of `execute_return_with_aggregation` (~lines 279-316). One
    // disk read per (NodeIndex, slot) pair, deduplicated.
    let mut resolved_node_props: HashMap<(NodeIndex, usize), Value> = HashMap::new();
    for (key_parts, _) in &surrogate_groups {
        for (slot, part) in key_parts.iter().enumerate() {
            if let GroupKeyPart::NodeProp(idx) = part {
                resolved_node_props.entry((*idx, slot)).or_insert_with(|| {
                    executor.resolve_node_prop_for_group(*idx, &folded_group_exprs[slot])
                });
            }
        }
    }

    let mut groups: Vec<(Vec<Value>, GroupAcc)> = Vec::new();
    let mut group_index_map: HashMap<Vec<Value>, usize> = HashMap::new();

    for (key_parts, acc) in surrogate_groups {
        let resolved: Vec<Value> = key_parts
            .iter()
            .enumerate()
            .map(|(slot, part)| match part {
                GroupKeyPart::NodeProp(idx) => resolved_node_props
                    .get(&(*idx, slot))
                    .cloned()
                    .unwrap_or(Value::Null),
                GroupKeyPart::Resolved(v) => v.clone(),
            })
            .collect();

        match group_index_map.get(&resolved) {
            Some(&idx) => {
                // Merge accumulators (count two NodeIndexes-with-same-name into one group).
                let existing = std::mem::replace(&mut groups[idx].1, GroupAcc::new(specs));
                let merged = merge_group_accs(existing, acc);
                groups[idx].1 = merged;
            }
            None => {
                let idx = groups.len();
                group_index_map.insert(resolved.clone(), idx);
                groups.push((resolved, acc));
            }
        }
    }

    // Build output rows.
    let columns: Vec<String> = return_clause
        .items
        .iter()
        .map(return_item_column_name)
        .collect();

    let mut output_rows: Vec<ResultRow> = Vec::with_capacity(groups.len());
    for (resolved_keys, acc) in &groups {
        let mut projected = Bindings::with_capacity(return_clause.items.len());

        for (ki, &item_idx) in group_indices.iter().enumerate() {
            let key = return_item_column_name(&return_clause.items[item_idx]);
            projected.insert(key, resolved_keys[ki].clone());
        }
        for (ai, spec) in specs.iter().enumerate() {
            let key = return_item_column_name(&return_clause.items[agg_indices[ai]]);
            projected.insert(key, acc.states[ai].finalize(spec));
        }

        let mut row = ResultRow::from_projected(projected);
        for (k, v) in acc.first_node_bindings.iter() {
            row.node_bindings.insert(k.clone(), *v);
        }
        output_rows.push(row);
    }

    // Empty-set aggregation: no rows seen, no group keys -> emit the
    // identity row (count = 0, sum = 0, min/max/avg = Null). Matches
    // materialized behavior at execute_return_with_aggregation:209-220.
    if output_rows.is_empty() && group_indices.is_empty() {
        let mut projected = Bindings::with_capacity(return_clause.items.len());
        for (ai, spec) in specs.iter().enumerate() {
            let key = return_item_column_name(&return_clause.items[agg_indices[ai]]);
            let empty_state = AggState::new(spec);
            projected.insert(key, empty_state.finalize(spec));
        }
        output_rows.push(ResultRow::from_projected(projected));
    }

    // DISTINCT post-filter on projected columns. Mirrors materialized
    // path's DISTINCT step in `execute_return_with_aggregation`.
    if return_clause.distinct {
        let mut seen = HashSet::new();
        output_rows.retain(|row| {
            let key: Vec<Value> = columns
                .iter()
                .map(|c| row.projected.get(c).cloned().unwrap_or(Value::Null))
                .collect();
            seen.insert(key)
        });
    }

    Ok(RowStream::from_vec(output_rows, columns))
}

fn update_agg_state(
    state: &mut AggState,
    spec: &AggSpec,
    folded_arg: Option<&Expression>,
    row: &ResultRow,
    executor: &CypherExecutor<'_>,
) {
    if matches!(spec.kind, AggKind::CountStar) {
        state.record(None, spec);
        return;
    }
    let expr = match folded_arg {
        Some(e) => e,
        None => return,
    };

    if spec.distinct {
        // DISTINCT path. NodeIndex/EdgeIndex hashing avoids
        // materializing string forms of node bindings (matches
        // `evaluate_aggregate_with_rows` count(DISTINCT) at
        // return_clause.rs:441-466).
        if let Some(var_name) = &spec.arg_is_node_var {
            if let Some(&idx) = row.node_bindings.get(var_name) {
                let key = idx.index();
                let dn = state.distinct_nodes.get_or_insert_with(HashSet::new);
                if !dn.insert(key) {
                    return;
                }
                // Treat as non-null; record drives the aggregate kind logic.
                let val = match spec.kind {
                    AggKind::Count => Value::Boolean(true), // count() inc; record skips Null
                    _ => return, // other aggregates with DISTINCT-on-node are unusual; skip
                };
                state.record(Some(val), spec);
                return;
            }
        }
        if let Some(var_name) = &spec.arg_is_edge_var {
            if let Some(eb) = row.edge_bindings.get(var_name) {
                let key = eb.edge_index.index();
                let de = state.distinct_edges.get_or_insert_with(HashSet::new);
                if !de.insert(key) {
                    return;
                }
                state.record(Some(Value::Boolean(true)), spec);
                return;
            }
        }
        let val = executor
            .evaluate_expression(expr, row)
            .unwrap_or(Value::Null);
        if matches!(val, Value::Null) {
            return;
        }
        let dv = state.distinct_values.get_or_insert_with(HashSet::new);
        if !dv.insert(val.clone()) {
            return;
        }
        state.record(Some(val), spec);
    } else {
        let val = executor
            .evaluate_expression(expr, row)
            .unwrap_or(Value::Null);
        state.record(Some(val), spec);
    }
}

fn merge_group_accs(mut a: GroupAcc, b: GroupAcc) -> GroupAcc {
    debug_assert_eq!(a.states.len(), b.states.len());
    let mut merged_states = Vec::with_capacity(a.states.len());
    for (sa, sb) in a.states.drain(..).zip(b.states) {
        let mut sa = sa;
        sa.merge(sb);
        merged_states.push(sa);
    }
    a.states = merged_states;
    // Keep the first-seen node bindings — matches materialized behavior
    // (execute_return_with_aggregation uses the first row of the group).
    GroupAcc {
        states: a.states,
        first_node_bindings: a.first_node_bindings,
    }
}

// ---- Local helpers ---------------------------------------------------------

/// Permissive `Value -> f64` coercion mirroring `helpers::value_to_f64`.
/// Defined locally to keep this module independent of the helpers module's
/// fold/eval cycle, but kept narrow on purpose — only types the inline
/// aggregates accept.
fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        Value::UniqueId(u) => Some(*u as f64),
        Value::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn cmp_lt(a: &Value, b: &Value) -> bool {
    matches!(
        crate::graph::core::filtering::compare_values(a, b),
        Some(std::cmp::Ordering::Less)
    )
}

fn cmp_gt(a: &Value, b: &Value) -> bool {
    matches!(
        crate::graph::core::filtering::compare_values(a, b),
        Some(std::cmp::Ordering::Greater)
    )
}

fn combine(a: Option<Value>, b: Option<Value>, want_max: bool) -> Option<Value> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => Some(match (want_max, cmp_lt(&a, &b)) {
            (true, true) => b,
            (true, false) => a,
            (false, true) => a,
            (false, false) => b,
        }),
    }
}
