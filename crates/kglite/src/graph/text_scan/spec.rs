//! A text query as a user writes it (`text_match(value, query, options)`,
//! `db.text.search`), turned into a compiled [`Matcher`] once.
//!
//! The per-row meaning of a query is [`Matcher::find_in_record`] on the
//! value's text: a hit is a non-empty match. [`document_text`] says which
//! values have text at all.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use super::matcher::{Matcher, Pattern};
use crate::datatypes::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum QueryMode {
    Literal,
    Phrase,
    Regex,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TextQuery {
    pub(crate) mode: QueryMode,
    pub(crate) ignore_case: bool,
    pub(crate) text: String,
}

impl TextQuery {
    /// Read the `{mode, ignore_case}` options map. `who` names the caller in
    /// error messages (`text_match()`, `db.text.search`).
    pub(crate) fn new(text: &str, options: Option<&Value>, who: &str) -> Result<Self, String> {
        let mut query = TextQuery {
            mode: QueryMode::Literal,
            ignore_case: false,
            text: text.to_string(),
        };
        match options {
            None | Some(Value::Null) => {}
            Some(Value::Map(map)) => {
                for (key, value) in map.iter() {
                    match (key, value) {
                        ("mode", Value::String(mode)) => {
                            query.mode = match mode.as_str() {
                                "literal" => QueryMode::Literal,
                                "phrase" => QueryMode::Phrase,
                                "regex" => QueryMode::Regex,
                                other => {
                                    return Err(format!(
                                "{who}: unknown mode '{other}'; use 'literal', 'phrase' or 'regex'"
                            ))
                                }
                            }
                        }
                        ("mode", _) => {
                            return Err(format!("{who}: option 'mode' must be a string"))
                        }
                        ("ignore_case", Value::Boolean(flag)) => query.ignore_case = *flag,
                        ("ignore_case", _) => {
                            return Err(format!("{who}: option 'ignore_case' must be a boolean"))
                        }
                        (other, _) => {
                            return Err(format!(
                                "{who}: unknown option '{other}'; use 'mode' and 'ignore_case'"
                            ))
                        }
                    }
                }
            }
            Some(_) => return Err(format!("{who}: options must be a map")),
        }
        Ok(query)
    }

    pub(crate) fn pattern(&self) -> Pattern {
        let (text, case_insensitive) = (self.text.clone(), self.ignore_case);
        match self.mode {
            QueryMode::Literal => Pattern::Literal {
                text,
                case_insensitive,
            },
            QueryMode::Phrase => Pattern::Phrase {
                text,
                case_insensitive,
            },
            QueryMode::Regex => Pattern::Regex {
                pattern: text,
                case_insensitive,
            },
        }
    }

    /// The compiled matcher, from a shared cache keyed by the whole query.
    pub(crate) fn compile(&self, who: &str) -> Result<Arc<Matcher>, String> {
        static CACHE: LazyLock<Mutex<HashMap<TextQuery, Arc<Matcher>>>> =
            LazyLock::new(|| Mutex::new(HashMap::new()));
        const CAPACITY: usize = 256;
        if let Some(hit) = CACHE.lock().ok().and_then(|c| c.get(self).cloned()) {
            return Ok(hit);
        }
        let matcher =
            Arc::new(Matcher::compile(&self.pattern()).map_err(|e| format!("{who}: {e}"))?);
        if let Ok(mut cache) = CACHE.lock() {
            if cache.len() >= CAPACITY {
                cache.clear();
            }
            cache.insert(self.clone(), Arc::clone(&matcher));
        }
        Ok(matcher)
    }
}

/// The text a value offers a query: a string is itself; a list of strings
/// (nulls skipped) is its members joined by one space, the document the BM25
/// index builds; anything else, and a list holding a non-text member, has none.
pub(crate) fn document_text(value: &Value) -> Option<Cow<'_, str>> {
    match value {
        Value::String(text) => Some(Cow::Borrowed(text.as_str())),
        Value::List(items) => crate::graph::text_indexes::join_text_list(items).map(Cow::Owned),
        _ => None,
    }
}

/// Whether `text` holds a non-empty match.
pub(crate) fn has_match(matcher: &Matcher, text: &str) -> bool {
    matcher.is_match_in_record(text.as_bytes())
}

/// The matched bytes `offset..offset + len` of `text` with up to `context`
/// characters either side, on character boundaries.
pub(crate) fn snippet(text: &str, offset: usize, len: usize, context: usize) -> String {
    let end = (offset + len).min(text.len());
    let start = offset.min(end);
    let from = text[..start]
        .char_indices()
        .rev()
        .take(context)
        .last()
        .map_or(start, |(i, _)| i);
    let to = text[end..]
        .char_indices()
        .nth(context)
        .map_or(text.len(), |(i, _)| end + i);
    text[from..to].to_string()
}
