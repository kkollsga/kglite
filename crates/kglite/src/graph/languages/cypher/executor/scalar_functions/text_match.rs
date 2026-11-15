//! `text_match(value, query [, {mode, ignore_case}])`: whether the value's
//! text holds a non-empty match of the query.
//!
//! Null for a null value or query and for a value without text (see
//! [`document_text`]); an error for an unusable query or option. This is the
//! per-row meaning every access path must reproduce.
//!
//! It runs once per candidate row, so two costs are kept off the row. The
//! compiled matcher of a constant query and options is held in the executor
//! (`tm_cache`, one slot, keyed by the arguments themselves, as `text_bm25`'s
//! is); another call site in the same statement compiles through the shared
//! cache each row instead. A string property read off a node binding is
//! borrowed from its column rather than cloned into a `Value`.

use std::borrow::Cow;
use std::sync::Arc;

use super::super::execution_support::{ArgKey, TextMatchCache};
use super::super::*;
use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::StrField;
use crate::graph::text_scan::matcher::Matcher;
use crate::graph::text_scan::spec::{document_text, has_match, TextQuery};

const WHO: &str = "text_match()";

impl CypherExecutor<'_> {
    pub(super) fn eval_text_match(
        &self,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        if args.len() != 2 && args.len() != 3 {
            return Err("text_match() requires 2 or 3 args: (value, query[, options])".into());
        }
        let Some(matcher) = self.text_match_matcher(args, row)? else {
            return Ok(Value::Null);
        };
        if let Some(text) = self.borrowed_text(&args[0], row) {
            return Ok(Value::Boolean(has_match(&matcher, &text)));
        }
        let value = self.evaluate_expression(&args[0], row)?;
        Ok(match document_text(&value) {
            Some(document) => Value::Boolean(has_match(&matcher, &document)),
            None => Value::Null,
        })
    }

    /// The compiled matcher for this row's query, `None` for a null query.
    fn text_match_matcher(
        &self,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Arc<Matcher>>, String> {
        if let Some(cache) = self.tm_cache.get() {
            if cache.query.matches(&args[1])
                && match (&cache.options, args.get(2)) {
                    (None, None) => true,
                    (Some(key), Some(expr)) => key.matches(expr),
                    _ => false,
                }
            {
                return Ok(Some(Arc::clone(&cache.matcher)));
            }
        }
        let query = self.evaluate_expression(&args[1], row)?;
        let options = match args.get(2) {
            Some(expr) => Some(self.evaluate_expression(expr, row)?),
            None => None,
        };
        let text = match &query {
            Value::String(text) => text,
            Value::Null => return Ok(None),
            _ => return Err("text_match() expects a string query".into()),
        };
        let matcher = TextQuery::new(text, options.as_ref(), WHO)?.compile(WHO)?;
        let keys = ArgKey::of(&args[1]).map(|query| (query, args.get(2).map(ArgKey::of)));
        if let Some((query, options)) = keys {
            // `Some(None)` is an options argument with no key: row-dependent.
            if let Some(options) = options.map_or(Some(None), |key| key.map(Some)) {
                let _ = self.tm_cache.set(TextMatchCache {
                    query,
                    options,
                    matcher: Arc::clone(&matcher),
                });
            }
        }
        Ok(Some(matcher))
    }

    /// The string a `variable.property` read would produce, borrowed from the
    /// node's column. `None` for every other shape and for a value that is not
    /// a string (a list property, a number, null), which take the evaluated
    /// path and its rules.
    pub(super) fn borrowed_text<'r>(
        &'r self,
        expression: &Expression,
        row: &'r ResultRow,
    ) -> Option<Cow<'r, str>> {
        let Expression::PropertyAccess { variable, property } = expression else {
            return None;
        };
        if row.projected.contains_key(variable) {
            return None;
        }
        let &index = row.node_bindings.get(variable)?;
        let node = self.graph.graph.node_view(index)?;
        let type_str = node.node_type_str(&self.graph.interner);
        let field = self.graph.resolve_alias(type_str, property);
        match node.resolved_field_str(type_str, field, InternedKey::from_str(field)) {
            StrField::Str(text) => Some(text),
            StrField::NotString | StrField::Absent => None,
        }
    }
}
