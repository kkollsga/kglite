//! Cypher scalar functions — string category. Split out of the monolithic
//! `evaluate_scalar_function` dispatcher; arms are verbatim. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not
//! one of this category's functions so the dispatcher tries the next.
use super::super::helpers::*;
use super::super::*;
use crate::datatypes::values::Value;

impl<'a> CypherExecutor<'a> {
    pub(super) fn eval_string_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        let result: Result<Value, String> = match name {
            "toupper" | "touppercase" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::String(s.to_uppercase())),
                    _ => Ok(Value::Null),
                }
            }
            "tolower" | "tolowercase" => {
                let val = self.evaluate_expression(&args[0], row)?;
                match val {
                    Value::String(s) => Ok(Value::String(s.to_lowercase())),
                    _ => Ok(Value::Null),
                }
            }
            "tostring" => {
                let val = self.evaluate_expression(&args[0], row)?;
                Ok(Value::String(format_value_compact(&val)))
            }
            "text_edit_distance" => {
                if args.len() != 2 {
                    return Err("text_edit_distance() requires 2 arguments".into());
                }
                let a = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let b = coerce_to_string(self.evaluate_expression(&args[1], row)?);
                match (&a, &b) {
                    (Value::String(s1), Value::String(s2)) => {
                        Ok(Value::Int64(levenshtein(s1, s2) as i64))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "text_normalize" => {
                if args.len() != 1 {
                    return Err("text_normalize() requires 1 argument".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => {
                        let mut out = String::with_capacity(s.len());
                        let mut last_space = true;
                        for c in s.chars() {
                            if c.is_alphanumeric() {
                                for lc in c.to_lowercase() {
                                    out.push(lc);
                                }
                                last_space = false;
                            } else if c.is_whitespace() && !last_space {
                                out.push(' ');
                                last_space = true;
                            }
                            // punctuation: drop
                        }
                        Ok(Value::String(out.trim().to_string()))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "text_jaccard" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(
                        "text_jaccard() requires 2-3 arguments: (a, b [, separator])".into(),
                    );
                }
                let a = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let b = coerce_to_string(self.evaluate_expression(&args[1], row)?);
                let sep = if args.len() == 3 {
                    match self.evaluate_expression(&args[2], row)? {
                        Value::String(s) => Some(s),
                        _ => return Err("text_jaccard(): separator must be a string".into()),
                    }
                } else {
                    None
                };
                match (&a, &b) {
                    (Value::String(s1), Value::String(s2)) => {
                        let tokenize = |s: &str| -> std::collections::HashSet<String> {
                            match &sep {
                                Some(d) => s.split(d.as_str()).map(|t| t.to_string()).collect(),
                                None => s.split_whitespace().map(|t| t.to_string()).collect(),
                            }
                        };
                        let set_a = tokenize(s1);
                        let set_b = tokenize(s2);
                        if set_a.is_empty() && set_b.is_empty() {
                            return Ok(Some(Value::Float64(1.0)));
                        }
                        let inter = set_a.intersection(&set_b).count() as f64;
                        let union = set_a.union(&set_b).count() as f64;
                        Ok(Value::Float64(inter / union))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "text_ngrams" => {
                // Phase A.1 / C4 — native Value::List of Value::String.
                if args.len() != 2 {
                    return Err("text_ngrams() requires 2 arguments: (string, n)".into());
                }
                let s_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let n_val = self.evaluate_expression(&args[1], row)?;
                match (&s_val, &n_val) {
                    (Value::String(s), Value::Int64(n)) => {
                        let n = *n as usize;
                        if n == 0 {
                            return Err("text_ngrams(): n must be ≥ 1".into());
                        }
                        let chars: Vec<char> = s.chars().collect();
                        let mut grams: Vec<Value> = Vec::new();
                        if chars.len() >= n {
                            for i in 0..=chars.len() - n {
                                let gram: String = chars[i..i + n].iter().collect();
                                grams.push(Value::String(gram));
                            }
                        }
                        Ok(Value::List(grams))
                    }
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    _ => Ok(Value::Null),
                }
            }
            "text_contains_any" => {
                if args.is_empty() {
                    return Err("text_contains_any() requires at least 1 argument".into());
                }
                let s_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let s = match &s_val {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Some(Value::Null)),
                };
                // Phase A.1 / C4 — accept native Value::List, legacy
                // JSON-string list, single-string second arg, or
                // variadic remaining args.
                if args.len() == 2 {
                    let list_val = self.evaluate_expression(&args[1], row)?;
                    if let Value::List(needles) = &list_val {
                        for needle in needles {
                            if let Value::String(n) = needle {
                                if s.contains(n.as_str()) {
                                    return Ok(Some(Value::Boolean(true)));
                                }
                            }
                        }
                        return Ok(Some(Value::Boolean(false)));
                    }
                    if let Value::String(ref ls) = list_val {
                        if ls.starts_with('[') && ls.ends_with(']') {
                            let needles = parse_list_value(&list_val);
                            for needle in needles {
                                if let Value::String(n) = needle {
                                    if s.contains(n.as_str()) {
                                        return Ok(Some(Value::Boolean(true)));
                                    }
                                }
                            }
                            return Ok(Some(Value::Boolean(false)));
                        }
                        if s.contains(ls.as_str()) {
                            return Ok(Some(Value::Boolean(true)));
                        }
                        return Ok(Some(Value::Boolean(false)));
                    }
                }
                for arg in &args[1..] {
                    let needle = self.evaluate_expression(arg, row)?;
                    if let Value::String(n) = needle {
                        if s.contains(n.as_str()) {
                            return Ok(Some(Value::Boolean(true)));
                        }
                    }
                }
                Ok(Value::Boolean(false))
            }
            "text_starts_with_any" => {
                if args.is_empty() {
                    return Err("text_starts_with_any() requires at least 1 argument".into());
                }
                let s_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let s = match &s_val {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Some(Value::Null)),
                };
                // Phase A.1 / C4 — same native-list handling as
                // text_contains_any.
                if args.len() == 2 {
                    let list_val = self.evaluate_expression(&args[1], row)?;
                    if let Value::List(prefixes) = &list_val {
                        for prefix in prefixes {
                            if let Value::String(p) = prefix {
                                if s.starts_with(p.as_str()) {
                                    return Ok(Some(Value::Boolean(true)));
                                }
                            }
                        }
                        return Ok(Some(Value::Boolean(false)));
                    }
                    if let Value::String(ref ls) = list_val {
                        if ls.starts_with('[') && ls.ends_with(']') {
                            let prefixes = parse_list_value(&list_val);
                            for prefix in prefixes {
                                if let Value::String(p) = prefix {
                                    if s.starts_with(p.as_str()) {
                                        return Ok(Some(Value::Boolean(true)));
                                    }
                                }
                            }
                            return Ok(Some(Value::Boolean(false)));
                        }
                        if s.starts_with(ls.as_str()) {
                            return Ok(Some(Value::Boolean(true)));
                        }
                        return Ok(Some(Value::Boolean(false)));
                    }
                }
                for arg in &args[1..] {
                    let prefix = self.evaluate_expression(arg, row)?;
                    if let Value::String(p) = prefix {
                        if s.starts_with(p.as_str()) {
                            return Ok(Some(Value::Boolean(true)));
                        }
                    }
                }
                Ok(Value::Boolean(false))
            }
            // Regex matching (2026-05-25 broad-scan lift, Batch 3).
            // Real use case: server-side pattern filtering on large
            // graphs — `MATCH (n) WHERE text_match_regex(n.name,
            // '^[A-Z]{2}\\d+$') RETURN n` filters in-graph instead of
            // shipping rows to the client. Pattern compilation cached
            // via `super::regex_cache::get_or_compile`.
            //
            // Flag syntax: third arg is a Rust-regex flag string
            // (`i` case-insensitive, `m` multiline, `s` dot-matches-
            // newline, `x` ignore-whitespace, `U` ungreedy). Internally
            // we prepend `(?<flags>)` to the pattern. Equivalent to
            // writing the flags inline in the pattern string.
            "text_match_regex" => {
                if args.len() != 2 && args.len() != 3 {
                    return Err(
                        "text_match_regex() requires 2 or 3 args: (text, pattern[, flags])".into(),
                    );
                }
                let text = self.evaluate_expression(&args[0], row)?;
                let pattern = self.evaluate_expression(&args[1], row)?;
                let flags: Option<String> = if args.len() == 3 {
                    match self.evaluate_expression(&args[2], row)? {
                        Value::String(s) => Some(s),
                        Value::Null => None,
                        _ => return Err("text_match_regex() flags must be a string".into()),
                    }
                } else {
                    None
                };
                let (text_str, pattern_str) = match (&text, &pattern) {
                    (Value::String(t), Value::String(p)) => (t.as_str(), p.as_str()),
                    (Value::Null, _) | (_, Value::Null) => return Ok(Some(Value::Null)),
                    _ => {
                        return Err("text_match_regex() expects (string, string[, string])".into());
                    }
                };
                let effective_pattern = if let Some(f) = &flags {
                    for c in f.chars() {
                        if !"imsxU".contains(c) {
                            return Err(format!(
                                "text_match_regex() unknown flag '{c}'; valid: i, m, s, x, U"
                            ));
                        }
                    }
                    format!("(?{f}){pattern_str}")
                } else {
                    pattern_str.to_string()
                };
                let re = super::regex_cache::get_or_compile(&effective_pattern)
                    .map_err(|e| format!("text_match_regex() invalid pattern: {e}"))?;
                Ok(Value::Boolean(re.is_match(text_str)))
            }
            // ── String functions ──────────────────────────────────
            "split" => {
                if args.len() != 2 {
                    return Err("split() requires 2 arguments: string, delimiter".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let delim_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &delim_val) {
                    (Value::String(s), Value::String(delim)) => {
                        // Return a native Value::List (Cypher semantics),
                        // consistent with range()/labels()/keys(). Downstream
                        // list ops (head/last/size/index/reverse) all accept it.
                        let parts: Vec<Value> = s
                            .split(delim.as_str())
                            .map(|p| Value::String(p.to_string()))
                            .collect();
                        Ok(Value::List(parts))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "replace" => {
                if args.len() != 3 {
                    return Err(
                        "replace() requires 3 arguments: string, search, replacement".into(),
                    );
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let search_val = self.evaluate_expression(&args[1], row)?;
                let replace_val = self.evaluate_expression(&args[2], row)?;
                match (&str_val, &search_val, &replace_val) {
                    (Value::String(s), Value::String(search), Value::String(replacement)) => Ok(
                        Value::String(s.replace(search.as_str(), replacement.as_str())),
                    ),
                    _ => Ok(Value::Null),
                }
            }
            "substring" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(
                        "substring() requires 2-3 arguments: string, start [, length]".into(),
                    );
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let start_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &start_val) {
                    (Value::String(s), Value::Int64(start)) => {
                        let start_idx = (*start).max(0) as usize;
                        let substr: String = if args.len() == 3 {
                            let len_val = self.evaluate_expression(&args[2], row)?;
                            match len_val {
                                Value::Int64(len) => {
                                    let take = (len).max(0) as usize;
                                    s.chars().skip(start_idx).take(take).collect()
                                }
                                _ => return Ok(Some(Value::Null)),
                            }
                        } else {
                            s.chars().skip(start_idx).collect()
                        };
                        Ok(Value::String(substr))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "left" => {
                if args.len() != 2 {
                    return Err("left() requires 2 arguments: string, length".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let len_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &len_val) {
                    (Value::String(s), Value::Int64(len)) => {
                        let result: String = s.chars().take(*len as usize).collect();
                        Ok(Value::String(result))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "right" => {
                if args.len() != 2 {
                    return Err("right() requires 2 arguments: string, length".into());
                }
                let str_val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                let len_val = self.evaluate_expression(&args[1], row)?;
                match (&str_val, &len_val) {
                    (Value::String(s), Value::Int64(len)) => {
                        let char_count = s.chars().count();
                        let skip = char_count.saturating_sub(*len as usize);
                        let result: String = s.chars().skip(skip).collect();
                        Ok(Value::String(result))
                    }
                    _ => Ok(Value::Null),
                }
            }
            "trim" | "btrim" => {
                if args.len() != 1 {
                    return Err("trim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "ltrim" => {
                if args.len() != 1 {
                    return Err("ltrim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim_start().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            "rtrim" => {
                if args.len() != 1 {
                    return Err("rtrim() requires 1 argument: string".into());
                }
                let val = coerce_to_string(self.evaluate_expression(&args[0], row)?);
                match val {
                    Value::String(s) => Ok(Value::String(s.trim_end().to_string())),
                    _ => Ok(Value::Null),
                }
            }
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
