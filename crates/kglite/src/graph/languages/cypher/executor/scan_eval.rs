//! Scan-compiled expressions and predicates for the fused node-scan operators.
//!
//! The fused node-scan shapes (`FusedNodeScanAggregate`, `FusedNodeScanTopK`)
//! bind exactly one node variable and evaluate the same handful of expressions
//! once per candidate. Routed through [`CypherExecutor::evaluate_expression`],
//! every one of those evaluations re-does work that is constant for the whole
//! scan — or at worst for one node *type*:
//!
//! * the row's `node_bindings` lookup (a `String` compare per access),
//! * `GraphRead::node_view`, which re-resolves the type's column store from the
//!   backend's map on every call,
//! * `DirGraph::resolve_alias` plus the alias fast-reject, which hashes the
//!   property **name**, and
//! * `InternedKey::from_str`, which hashes it a second time.
//!
//! This module hoists all four. The caller resolves one [`NodeView`] per row
//! (memoising the store handle by node type, as
//! `execute_fused_match_return_aggregate` already does) and the compiled tree
//! addresses properties by *slot*: a [`PropRoute`] resolved once per node type
//! that says which of the five things a property name can mean — the `id`
//! virtual, the `title` virtual, a stored property, or one of the two
//! structural soft aliases — this scan is reading.
//!
//! It also gives the WHERE evaluator a **borrowed** string route. A predicate
//! the planner could not push (`<>`, an OR-combined comparison, the safety net
//! retained behind a text-index probe) otherwise materialises an owned
//! `Value::String` per row purely to compare it and drop it;
//! [`ScanPred::StrCmp`] compares out of the column.
//!
//! **Compilation is opt-in per node, never lossy.** Anything this module does
//! not model compiles to [`ScanExpr::Generic`] / [`ScanPred::Generic`], which
//! calls straight back into the interpreter with the same row — so the fused
//! operators keep exactly one semantics, and a shape that is not hoisted is
//! merely not *faster*.

use super::super::ast::{ComparisonOp, Expression, Predicate};
use super::helpers::evaluate_comparison;
use super::CypherExecutor;
use crate::datatypes::values::Value;
use crate::graph::core::filtering::str_values_equal;
use crate::graph::core::membership::MembershipSet;
use crate::graph::languages::cypher::result::ResultRow;
use crate::graph::schema::{soft_alias_fallback, DirGraph, InternedKey, SoftAliasFallback};
use crate::graph::storage::{ColumnStore, GraphRead, NodeView, StrField};
use petgraph::graph::NodeIndex;
use std::borrow::Cow;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Property routes — what a name means, resolved once per node type
// ---------------------------------------------------------------------------

/// What `variable.property` reads, for one node type.
///
/// This is [`super::helpers::resolve_node_property`]'s decision tree with the
/// decision lifted out of the row loop: alias resolution and the `id`/`title`/
/// soft-alias name tests are pure functions of `(node type, written name)`.
/// Spatial virtuals are the one branch not represented — see
/// [`ScanCompiler::new`], which refuses to compile anything on a graph that has
/// a spatial configuration.
#[derive(Clone, Copy, Debug)]
enum PropRoute {
    /// The type's id field (`n.id`, or the type's declared id alias).
    Id,
    /// The type's title field (`n.title`, or the type's declared title alias).
    Title,
    /// A plain stored property.
    Stored(InternedKey),
    /// `name` — a stored property wins (KG-1), else the node's title.
    SoftTitle(InternedKey),
    /// `type` / `node_type` / `label` — a stored property wins, else the type.
    SoftType(InternedKey),
}

impl PropRoute {
    fn resolve(graph: &DirGraph, type_str: &str, property: &str) -> PropRoute {
        let resolved = graph.resolve_alias(type_str, property);
        match resolved {
            "id" => PropRoute::Id,
            "title" => PropRoute::Title,
            _ => {
                let key = InternedKey::from_str(resolved);
                match soft_alias_fallback(resolved) {
                    None => PropRoute::Stored(key),
                    Some(SoftAliasFallback::Title) => PropRoute::SoftTitle(key),
                    Some(SoftAliasFallback::TypeString) => PropRoute::SoftType(key),
                }
            }
        }
    }

    /// The owned read — byte for byte what `resolve_node_property` returns for
    /// this name on a non-spatial graph.
    #[inline]
    fn read(self, node: NodeView<'_>, type_str: &str) -> Value {
        match self {
            PropRoute::Id => node.id().into_owned(),
            PropRoute::Title => node.title().into_owned(),
            PropRoute::Stored(key) => node.get_value(key).unwrap_or(Value::Null),
            PropRoute::SoftTitle(key) => node
                .get_value(key)
                .unwrap_or_else(|| node.title().into_owned()),
            PropRoute::SoftType(key) => node
                .get_value(key)
                .unwrap_or_else(|| Value::String(type_str.to_string())),
        }
    }

    /// The borrowed read — [`NodeView::resolved_field_str`]'s resolution order,
    /// with the name tests already decided.
    #[inline]
    fn read_str<'v>(self, node: NodeView<'v>, type_str: &'v str) -> StrField<'v> {
        match self {
            PropRoute::Id => node.id_field(),
            PropRoute::Title => node.title_field(),
            PropRoute::Stored(key) => node.str_field(key),
            PropRoute::SoftTitle(key) => match node.str_field(key) {
                StrField::Absent => node.title_field(),
                resolved => resolved,
            },
            PropRoute::SoftType(key) => match node.str_field(key) {
                StrField::Absent => StrField::Str(Cow::Borrowed(type_str)),
                resolved => resolved,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime — one node view and one route table per row
// ---------------------------------------------------------------------------

/// Per-row state for a compiled scan: the current node's view, and the property
/// routes resolved for its type.
///
/// [`Self::bind`] is the only way to advance a row, which is what keeps the
/// route table and the view in step — a compiled `Prop` slot is only ever read
/// against a view whose type the table was resolved for.
pub(super) struct ScanRuntime<'g> {
    /// Written property names, in slot order. Borrowed from the folded query.
    names: Vec<&'g str>,
    /// Parallel to `names`; valid for `current_type`.
    routes: Vec<PropRoute>,
    current_type: Option<InternedKey>,
    type_str: &'g str,
    /// One column-store handle per node **type**. [`GraphRead::node_view`]
    /// re-resolves the backend's store map on every call; a scan calls it once
    /// per candidate (and the interpreter, once per candidate *per property*).
    store: Option<(InternedKey, Option<&'g Arc<ColumnStore>>)>,
}

impl<'g> ScanRuntime<'g> {
    /// A second runtime over the same compiled slots, with an empty memo.
    ///
    /// The compiled trees ([`ScanExpr`]/[`ScanPred`]) are immutable and shared;
    /// the *runtime* is per-row mutable state (the route table and the store
    /// handle), so a scan that fans out needs one per partition. Forking is
    /// cheap — the slot names are `&str` into the folded query.
    pub(super) fn fork(&self) -> ScanRuntime<'g> {
        ScanRuntime {
            names: self.names.clone(),
            routes: Vec::with_capacity(self.names.len()),
            current_type: None,
            type_str: "",
            store: None,
        }
    }

    /// `true` when no compiled node in this scan reads a property, so the
    /// caller can skip [`Self::bind`] entirely. On the disk backend that is
    /// always true — nothing compiles there — and it matters, because
    /// `node_weight` materialises into the query arena.
    pub(super) fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Resolve one node: its view, and (on a type change) its route table.
    #[inline]
    pub(super) fn bind(&mut self, graph: &'g DirGraph, idx: NodeIndex) -> Option<NodeView<'g>> {
        let data = graph.graph.node_weight(idx)?;
        let type_key = data.node_type;
        let store = match self.store {
            Some((memo_key, store)) if memo_key == type_key => store,
            _ => {
                let store = graph.graph.column_store(type_key);
                self.store = Some((type_key, store));
                store
            }
        };
        if self.current_type != Some(type_key) {
            self.retarget(graph, type_key);
        }
        let resolved = data
            .properties
            .columnar_row_id()
            .and_then(|row_id| store.map(|store| (&**store, row_id)));
        Some(NodeView::new(data, resolved))
    }

    #[cold]
    fn retarget(&mut self, graph: &'g DirGraph, type_key: InternedKey) {
        self.type_str = graph.interner.try_resolve(type_key).unwrap_or("");
        self.routes.clear();
        for slot in 0..self.names.len() {
            self.routes
                .push(PropRoute::resolve(graph, self.type_str, self.names[slot]));
        }
        self.current_type = Some(type_key);
    }

    #[inline]
    fn read(&self, node: Option<NodeView<'_>>, slot: usize) -> Value {
        match node {
            Some(node) => self.routes[slot].read(node, self.type_str),
            None => Value::Null,
        }
    }
}

// ---------------------------------------------------------------------------
// Compiled trees
// ---------------------------------------------------------------------------

/// The arithmetic/`||` operators the compiler models. Each dispatches to the
/// same `value_operations` function the interpreter's `evaluate_binary` uses,
/// with the same left-then-right evaluation order.
#[derive(Clone, Copy, Debug)]
pub(super) enum ScanBinOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Concat,
}

/// A scan-compiled expression over one bound node variable.
pub(super) enum ScanExpr<'q> {
    /// A property of the scan variable, addressed by route slot.
    Prop(usize),
    /// A value folded before the row loop.
    Const(Value),
    Binary(ScanBinOp, Box<ScanExpr<'q>>, Box<ScanExpr<'q>>),
    Negate(Box<ScanExpr<'q>>),
    /// Anything not modelled here — evaluated by the interpreter, unchanged.
    Generic(&'q Expression),
}

/// A scan-compiled predicate. Three-valued exactly like
/// [`CypherExecutor::evaluate_predicate_tristate`], which the `Generic` arm
/// calls directly.
pub(super) enum ScanPred<'q> {
    And(Box<ScanPred<'q>>, Box<ScanPred<'q>>),
    Or(Box<ScanPred<'q>>, Box<ScanPred<'q>>),
    Xor(Box<ScanPred<'q>>, Box<ScanPred<'q>>),
    Not(Box<ScanPred<'q>>),
    Comparison {
        left: ScanExpr<'q>,
        operator: ComparisonOp,
        right: ScanExpr<'q>,
    },
    /// A property tested against a constant string, answered from a borrowed
    /// column read.
    StrCmp {
        slot: usize,
        op: StrOp,
        needle: String,
    },
    IsNull(ScanExpr<'q>),
    IsNotNull(ScanExpr<'q>),
    InLiteralSet {
        expr: ScanExpr<'q>,
        values: &'q MembershipSet,
    },
    Generic(&'q Predicate),
}

/// The string tests [`ScanPred::StrCmp`] answers without materialising a
/// `Value`. Every variant reproduces its interpreter counterpart exactly —
/// see [`StrOp::test`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StrOp {
    Equals,
    NotEquals,
    LessThan,
    LessThanEq,
    GreaterThan,
    GreaterThanEq,
    StartsWith,
    EndsWith,
    Contains,
}

impl StrOp {
    /// The comparison operator this string test stands in for, or `None` for
    /// the three text predicates (which have no `ComparisonOp` spelling).
    fn as_comparison(self) -> Option<ComparisonOp> {
        Some(match self {
            StrOp::Equals => ComparisonOp::Equals,
            StrOp::NotEquals => ComparisonOp::NotEquals,
            StrOp::LessThan => ComparisonOp::LessThan,
            StrOp::LessThanEq => ComparisonOp::LessThanEq,
            StrOp::GreaterThan => ComparisonOp::GreaterThan,
            StrOp::GreaterThanEq => ComparisonOp::GreaterThanEq,
            StrOp::StartsWith | StrOp::EndsWith | StrOp::Contains => return None,
        })
    }

    /// Mirror image, for a flipped comparison (`'lit' < n.prop`).
    fn flipped(self) -> Self {
        match self {
            StrOp::LessThan => StrOp::GreaterThan,
            StrOp::LessThanEq => StrOp::GreaterThanEq,
            StrOp::GreaterThan => StrOp::LessThan,
            StrOp::GreaterThanEq => StrOp::LessThanEq,
            other => other,
        }
    }

    /// The borrowed test.
    ///
    /// The ordering arms are `compare_values`' `(String, String)` arm — a plain
    /// `str` `Ord`. The equality arms are `values_equal`'s: byte equality
    /// **plus** its single-element-JSON-list equivalence (`["Oslo"] = 'Oslo'`),
    /// which is why this is not a bare `==`. The two are deliberately separate
    /// tests rather than one unified "string equality": unifying them was
    /// measured slower and reverted during the shape-convergence fix loop.
    #[inline]
    fn test(self, value: &str, needle: &str) -> bool {
        match self {
            StrOp::Equals => str_values_equal(value, needle),
            StrOp::NotEquals => !str_values_equal(value, needle),
            StrOp::LessThan => value < needle,
            StrOp::LessThanEq => value <= needle,
            StrOp::GreaterThan => value > needle,
            StrOp::GreaterThanEq => value >= needle,
            StrOp::StartsWith => value.starts_with(needle),
            StrOp::EndsWith => value.ends_with(needle),
            StrOp::Contains => value.contains(needle),
        }
    }
}

impl ScanExpr<'_> {
    /// `false` if any node of this tree falls back to the interpreter. The
    /// runtime parallel gate reads this as the expression's cost class: a
    /// fully compiled tree is a column read plus arithmetic (tens of ns a
    /// row), a `Generic` one re-enters `evaluate_expression`.
    pub(super) fn is_compiled(&self) -> bool {
        match self {
            ScanExpr::Prop(_) | ScanExpr::Const(_) => true,
            ScanExpr::Binary(_, lhs, rhs) => lhs.is_compiled() && rhs.is_compiled(),
            ScanExpr::Negate(inner) => inner.is_compiled(),
            ScanExpr::Generic(_) => false,
        }
    }
}

impl ScanPred<'_> {
    /// `false` if any node of this predicate falls back to the interpreter.
    /// See [`ScanExpr::is_compiled`].
    pub(super) fn is_compiled(&self) -> bool {
        match self {
            ScanPred::And(lhs, rhs) | ScanPred::Or(lhs, rhs) | ScanPred::Xor(lhs, rhs) => {
                lhs.is_compiled() && rhs.is_compiled()
            }
            ScanPred::Not(inner) => inner.is_compiled(),
            ScanPred::Comparison { left, right, .. } => left.is_compiled() && right.is_compiled(),
            ScanPred::StrCmp { .. } => true,
            ScanPred::IsNull(expr) | ScanPred::IsNotNull(expr) => expr.is_compiled(),
            ScanPred::InLiteralSet { expr, .. } => expr.is_compiled(),
            ScanPred::Generic(_) => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

/// Compiles a scan's expressions and predicate, accumulating the property slots
/// they address. One compiler per scan; [`Self::finish`] hands the caller the
/// runtime whose slots the compiled trees index.
pub(super) struct ScanCompiler<'q> {
    node_var: &'q str,
    names: Vec<&'q str>,
    /// `false` on backends and graph shapes this compiler does not model;
    /// everything then compiles to `Generic`.
    enabled: bool,
}

impl<'q> ScanCompiler<'q> {
    pub(super) fn new(executor: &CypherExecutor<'_>, node_var: &'q str) -> Self {
        // Two shapes stay entirely on the interpreter:
        //   * the disk backend, which resolves properties through its own
        //     arena-backed route in `CypherExecutor::resolve_property`, and
        //   * a graph with any spatial configuration, whose types answer
        //     `location` / `geometry` / configured point and shape names out of
        //     *other* columns — a branch `PropRoute` does not model.
        let enabled = !executor.graph.graph.is_disk() && executor.graph.spatial_configs.is_empty();
        ScanCompiler {
            node_var,
            names: Vec::new(),
            enabled,
        }
    }

    /// The runtime the compiled trees index into.
    pub(super) fn finish(self) -> ScanRuntime<'q> {
        ScanRuntime {
            routes: Vec::with_capacity(self.names.len()),
            names: self.names,
            current_type: None,
            type_str: "",
            store: None,
        }
    }

    /// Slot for `property`, deduplicated — `n.a + n.a` resolves one route.
    fn slot(&mut self, property: &'q str) -> usize {
        match self.names.iter().position(|name| *name == property) {
            Some(slot) => slot,
            None => {
                self.names.push(property);
                self.names.len() - 1
            }
        }
    }

    /// The slot for a `PropertyAccess` on the scan variable, or `None` when the
    /// expression is not one (or this compiler is disabled).
    fn prop_slot(&mut self, expr: &'q Expression) -> Option<usize> {
        if !self.enabled {
            return None;
        }
        let Expression::PropertyAccess { variable, property } = expr else {
            return None;
        };
        if variable != self.node_var {
            return None;
        }
        Some(self.slot(property))
    }

    pub(super) fn expr(&mut self, expr: &'q Expression) -> ScanExpr<'q> {
        if let Some(slot) = self.prop_slot(expr) {
            return ScanExpr::Prop(slot);
        }
        if !self.enabled {
            return ScanExpr::Generic(expr);
        }
        match expr {
            Expression::Literal(value) => ScanExpr::Const(value.clone()),
            Expression::Add(left, right) => self.binary(ScanBinOp::Add, left, right, expr),
            Expression::Subtract(left, right) => {
                self.binary(ScanBinOp::Subtract, left, right, expr)
            }
            Expression::Multiply(left, right) => {
                self.binary(ScanBinOp::Multiply, left, right, expr)
            }
            Expression::Divide(left, right) => self.binary(ScanBinOp::Divide, left, right, expr),
            Expression::Modulo(left, right) => self.binary(ScanBinOp::Modulo, left, right, expr),
            Expression::Concat(left, right) => self.binary(ScanBinOp::Concat, left, right, expr),
            Expression::Negate(inner) => {
                let inner = self.expr(inner);
                if inner.is_generic() {
                    ScanExpr::Generic(expr)
                } else {
                    ScanExpr::Negate(Box::new(inner))
                }
            }
            _ => ScanExpr::Generic(expr),
        }
    }

    /// Compile a binary node, collapsing back to `Generic` when neither operand
    /// gained anything — an all-generic subtree is cheaper evaluated in one
    /// interpreter call than in three.
    fn binary(
        &mut self,
        op: ScanBinOp,
        left: &'q Expression,
        right: &'q Expression,
        whole: &'q Expression,
    ) -> ScanExpr<'q> {
        let left = self.expr(left);
        let right = self.expr(right);
        if left.is_generic() && right.is_generic() {
            return ScanExpr::Generic(whole);
        }
        ScanExpr::Binary(op, Box::new(left), Box::new(right))
    }

    pub(super) fn pred(&mut self, pred: &'q Predicate) -> ScanPred<'q> {
        if !self.enabled {
            return ScanPred::Generic(pred);
        }
        match pred {
            Predicate::And(left, right) => {
                ScanPred::And(Box::new(self.pred(left)), Box::new(self.pred(right)))
            }
            Predicate::Or(left, right) => {
                ScanPred::Or(Box::new(self.pred(left)), Box::new(self.pred(right)))
            }
            Predicate::Xor(left, right) => {
                ScanPred::Xor(Box::new(self.pred(left)), Box::new(self.pred(right)))
            }
            Predicate::Not(inner) => ScanPred::Not(Box::new(self.pred(inner))),
            Predicate::Comparison {
                left,
                operator,
                right,
            } => self.comparison(pred, left, *operator, right),
            Predicate::StartsWith { expr, pattern } => {
                self.text(pred, expr, pattern, StrOp::StartsWith)
            }
            Predicate::EndsWith { expr, pattern } => {
                self.text(pred, expr, pattern, StrOp::EndsWith)
            }
            Predicate::Contains { expr, pattern } => {
                self.text(pred, expr, pattern, StrOp::Contains)
            }
            Predicate::IsNull(expr) => match self.expr(expr) {
                ScanExpr::Generic(_) => ScanPred::Generic(pred),
                compiled => ScanPred::IsNull(compiled),
            },
            Predicate::IsNotNull(expr) => match self.expr(expr) {
                ScanExpr::Generic(_) => ScanPred::Generic(pred),
                compiled => ScanPred::IsNotNull(compiled),
            },
            Predicate::InLiteralSet { expr, values } => match self.expr(expr) {
                ScanExpr::Generic(_) => ScanPred::Generic(pred),
                compiled => ScanPred::InLiteralSet {
                    expr: compiled,
                    values,
                },
            },
            _ => ScanPred::Generic(pred),
        }
    }

    fn comparison(
        &mut self,
        whole: &'q Predicate,
        left: &'q Expression,
        operator: ComparisonOp,
        right: &'q Expression,
    ) -> ScanPred<'q> {
        // Property-vs-constant-string in either spelling takes the borrowed
        // route; `=~` keeps the interpreter's regex path.
        let str_op = match operator {
            ComparisonOp::Equals => Some(StrOp::Equals),
            ComparisonOp::NotEquals => Some(StrOp::NotEquals),
            ComparisonOp::LessThan => Some(StrOp::LessThan),
            ComparisonOp::LessThanEq => Some(StrOp::LessThanEq),
            ComparisonOp::GreaterThan => Some(StrOp::GreaterThan),
            ComparisonOp::GreaterThanEq => Some(StrOp::GreaterThanEq),
            ComparisonOp::RegexMatch => None,
        };
        if let Some(op) = str_op {
            if let Some(pred) = self.str_cmp(left, op, right) {
                return pred;
            }
            if let Some(pred) = self.str_cmp(right, op.flipped(), left) {
                return pred;
            }
        }
        let left = self.expr(left);
        let right = self.expr(right);
        if left.is_generic() && right.is_generic() {
            return ScanPred::Generic(whole);
        }
        ScanPred::Comparison {
            left,
            operator,
            right,
        }
    }

    fn str_cmp(
        &mut self,
        prop_side: &'q Expression,
        op: StrOp,
        literal_side: &Expression,
    ) -> Option<ScanPred<'q>> {
        let Expression::Literal(Value::String(needle)) = literal_side else {
            return None;
        };
        let slot = self.prop_slot(prop_side)?;
        Some(ScanPred::StrCmp {
            slot,
            op,
            needle: needle.clone(),
        })
    }

    fn text(
        &mut self,
        whole: &'q Predicate,
        expr: &'q Expression,
        pattern: &'q Expression,
        op: StrOp,
    ) -> ScanPred<'q> {
        self.str_cmp(expr, op, pattern)
            .unwrap_or(ScanPred::Generic(whole))
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

impl ScanExpr<'_> {
    fn is_generic(&self) -> bool {
        matches!(self, ScanExpr::Generic(_))
    }

    pub(super) fn eval(
        &self,
        executor: &CypherExecutor<'_>,
        runtime: &ScanRuntime<'_>,
        node: Option<NodeView<'_>>,
        row: &ResultRow,
    ) -> Result<Value, String> {
        use crate::graph::core::value_operations as ops;
        match self {
            ScanExpr::Prop(slot) => Ok(runtime.read(node, *slot)),
            ScanExpr::Const(value) => Ok(value.clone()),
            ScanExpr::Generic(expr) => executor.evaluate_expression(expr, row),
            ScanExpr::Negate(inner) => {
                super::helpers::arithmetic_negate(&inner.eval(executor, runtime, node, row)?)
            }
            ScanExpr::Binary(op, left, right) => {
                // Evaluation order is observable through errors and budgets;
                // left first, exactly as `evaluate_binary` does.
                let left = left.eval(executor, runtime, node, row)?;
                let right = right.eval(executor, runtime, node, row)?;
                match op {
                    ScanBinOp::Add => ops::arithmetic_add_checked(&left, &right),
                    ScanBinOp::Subtract => ops::arithmetic_sub_checked(&left, &right),
                    ScanBinOp::Multiply => ops::arithmetic_mul_checked(&left, &right),
                    ScanBinOp::Divide => super::helpers::arithmetic_div(&left, &right),
                    ScanBinOp::Modulo => super::helpers::arithmetic_mod(&left, &right),
                    ScanBinOp::Concat => Ok(ops::string_concat(&left, &right)),
                }
            }
        }
    }
}

impl ScanPred<'_> {
    pub(super) fn eval(
        &self,
        executor: &CypherExecutor<'_>,
        runtime: &ScanRuntime<'_>,
        node: Option<NodeView<'_>>,
        row: &ResultRow,
    ) -> Result<Option<bool>, String> {
        match self {
            ScanPred::Generic(pred) => executor.evaluate_predicate_tristate(pred, row),
            ScanPred::And(left, right) => {
                // Kleene AND — FALSE absorbs past NULL, as in the interpreter.
                let lv = left.eval(executor, runtime, node, row)?;
                if lv == Some(false) {
                    return Ok(Some(false));
                }
                let rv = right.eval(executor, runtime, node, row)?;
                if rv == Some(false) {
                    return Ok(Some(false));
                }
                if lv.is_none() || rv.is_none() {
                    return Ok(None);
                }
                Ok(Some(true))
            }
            ScanPred::Or(left, right) => {
                let lv = left.eval(executor, runtime, node, row)?;
                if lv == Some(true) {
                    return Ok(Some(true));
                }
                let rv = right.eval(executor, runtime, node, row)?;
                if rv == Some(true) {
                    return Ok(Some(true));
                }
                if lv.is_none() || rv.is_none() {
                    return Ok(None);
                }
                Ok(Some(false))
            }
            ScanPred::Xor(left, right) => {
                let lv = left.eval(executor, runtime, node, row)?;
                let rv = right.eval(executor, runtime, node, row)?;
                match (lv, rv) {
                    (Some(a), Some(b)) => Ok(Some(a ^ b)),
                    _ => Ok(None),
                }
            }
            ScanPred::Not(inner) => Ok(inner.eval(executor, runtime, node, row)?.map(|b| !b)),
            ScanPred::Comparison {
                left,
                operator,
                right,
            } => {
                let left = left.eval(executor, runtime, node, row)?;
                let right = right.eval(executor, runtime, node, row)?;
                if matches!(left, Value::Null) || matches!(right, Value::Null) {
                    return Ok(None);
                }
                evaluate_comparison(&left, operator, &right).map(Some)
            }
            ScanPred::IsNull(expr) => Ok(Some(matches!(
                expr.eval(executor, runtime, node, row)?,
                Value::Null
            ))),
            ScanPred::IsNotNull(expr) => Ok(Some(!matches!(
                expr.eval(executor, runtime, node, row)?,
                Value::Null
            ))),
            ScanPred::InLiteralSet { expr, values } => {
                let value = expr.eval(executor, runtime, node, row)?;
                if matches!(value, Value::Null) {
                    return Ok(None);
                }
                if values.matches(&value) {
                    return Ok(Some(true));
                }
                if values.has_null() {
                    return Ok(None);
                }
                Ok(Some(false))
            }
            ScanPred::StrCmp { slot, op, needle } => {
                Self::eval_str_cmp(runtime, node, *slot, *op, needle)
            }
        }
    }

    /// The borrowed string test, with the two non-string outcomes routed back
    /// onto the owned values the interpreter would have compared.
    #[inline]
    fn eval_str_cmp(
        runtime: &ScanRuntime<'_>,
        node: Option<NodeView<'_>>,
        slot: usize,
        op: StrOp,
        needle: &str,
    ) -> Result<Option<bool>, String> {
        let Some(node) = node else {
            return Ok(None);
        };
        let route = runtime.routes[slot];
        match route.read_str(node, runtime.type_str) {
            StrField::Str(value) => Ok(Some(op.test(&value, needle))),
            // `Absent` is exactly what makes the owned read return `Null`, and
            // every interpreter arm reads a NULL operand as unknown.
            StrField::Absent => Ok(None),
            // Present but not a string. Rare, and not a case the borrowed route
            // can answer: `compare_values` parses dates out of the literal, and
            // a stored `Value::Null` still has to propagate NULL. Fall back to
            // the values the interpreter would have built.
            StrField::NotString => {
                let value = route.read(node, runtime.type_str);
                if matches!(value, Value::Null) {
                    return Ok(None);
                }
                let needle = Value::String(needle.to_string());
                match op.as_comparison() {
                    Some(operator) => evaluate_comparison(&value, &operator, &needle).map(Some),
                    // The text predicates answer `false` for a non-string
                    // left-hand side (their `_ => Ok(Some(false))` arm).
                    None => Ok(Some(false)),
                }
            }
        }
    }
}
