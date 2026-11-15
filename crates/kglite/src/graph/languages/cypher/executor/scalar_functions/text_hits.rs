//! `text_hits(value, query [, options])`: where a query matches inside one
//! value, as a list of `{offset, length, char_offset, snippet}` maps.
//!
//! The scalar twin of `db.text.search`'s per-value rows: the same matcher, the
//! same non-empty leftmost matches, the same byte `offset`/`length`, the same
//! `char_offset` and `snippet`. It runs on whatever row the pattern produced,
//! so it needs no procedure and no `collect`. Options are `text_match`'s
//! (`mode`, `ignore_case`) plus `snippet_chars` (default 40) and `max` (keep
//! the first that many hits). A value without text (see
//! [`document_text`]) or a null query gives null; a value with text and no
//! match gives `[]`.

use super::super::*;
use crate::datatypes::prop_map::PropMap;
use crate::datatypes::values::Value;
use crate::graph::text_scan::spec::{document_text, snippet, CharCursor, TextQuery};

const WHO: &str = "text_hits()";
const DEFAULT_SNIPPET_CHARS: usize = 40;

/// `text_hits`' own options split off the ones `TextQuery` reads.
struct HitOptions {
    query_options: Value,
    snippet_chars: usize,
    max: Option<usize>,
}

fn count_option(value: &Value, key: &str) -> Result<usize, String> {
    match value {
        Value::Int64(n) if *n >= 0 => Ok(*n as usize),
        _ => Err(format!(
            "{WHO}: option '{key}' must be a non-negative integer"
        )),
    }
}

fn hit_options(options: Option<&Value>) -> Result<HitOptions, String> {
    let mut out = HitOptions {
        query_options: Value::Null,
        snippet_chars: DEFAULT_SNIPPET_CHARS,
        max: None,
    };
    let map = match options {
        None | Some(Value::Null) => return Ok(out),
        Some(Value::Map(map)) => map,
        Some(_) => return Err(format!("{WHO}: options must be a map")),
    };
    let mut rest = Vec::new();
    for (key, value) in map.iter() {
        match key {
            "snippet_chars" => out.snippet_chars = count_option(value, key)?,
            "max" => out.max = Some(count_option(value, key)?),
            "mode" | "ignore_case" => rest.push((key.to_string(), value.clone())),
            other => {
                return Err(format!(
                    "{WHO}: unknown option '{other}'; use 'mode', 'ignore_case', \
                     'snippet_chars' and 'max'"
                ))
            }
        }
    }
    out.query_options = Value::Map(PropMap::from_pairs(rest));
    Ok(out)
}

impl CypherExecutor<'_> {
    pub(super) fn eval_text_hits(
        &self,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Value, String> {
        if args.len() != 2 && args.len() != 3 {
            return Err(format!(
                "{WHO} requires 2 or 3 args: (value, query[, options])"
            ));
        }
        let value = self.evaluate_expression(&args[0], row)?;
        let query = self.evaluate_expression(&args[1], row)?;
        let options = match args.get(2) {
            Some(expr) => Some(self.evaluate_expression(expr, row)?),
            None => None,
        };
        let text = match &query {
            Value::String(text) => text,
            Value::Null => return Ok(Value::Null),
            _ => return Err(format!("{WHO} expects a string query")),
        };
        let options = hit_options(options.as_ref())?;
        let matcher = TextQuery::new(text, Some(&options.query_options), WHO)?.compile(WHO)?;
        let Some(document) = document_text(&value) else {
            return Ok(Value::Null);
        };
        let mut cursor = CharCursor::default();
        let mut hits = Vec::new();
        matcher.find_in_record(document.as_bytes(), &mut |start, end| {
            if options.max.is_some_and(|m| hits.len() >= m) {
                return false;
            }
            let map = PropMap::from_pairs(vec![
                ("offset".to_string(), Value::Int64(start as i64)),
                ("length".to_string(), Value::Int64((end - start) as i64)),
                (
                    "char_offset".to_string(),
                    Value::Int64(cursor.char_offset(&document, start) as i64),
                ),
                (
                    "snippet".to_string(),
                    Value::String(snippet(
                        &document,
                        start,
                        end - start,
                        options.snippet_chars,
                    )),
                ),
            ]);
            hits.push(Value::Map(map));
            options.max.is_none_or(|m| hits.len() < m)
        });
        Ok(Value::List(hits))
    }
}
