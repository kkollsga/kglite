//! `text_column_scan`: let a text predicate over a string property find its
//! candidate nodes by scanning the property's column, instead of testing
//! every node of the label one at a time.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::super::ast::*;
use super::PassCtx;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher, TextScanSpec};
use crate::graph::text_scan::matcher::{required_literal, Pattern as ScanPattern};
use crate::graph::text_scan::spec::TextQuery;

/// A text predicate on `variable.property` the driver can find candidates for.
struct Recognised {
    variable: String,
    property: String,
    patterns: Vec<ScanPattern>,
}

/// **Pass:** `text_column_scan`.
///
/// **Precondition:** a plain `MATCH` clause (not `OPTIONAL`) whose WHERE has a
/// positive top-level conjunct that is one of
/// `text_match(v.p, 'q' [, {mode, ignore_case}])`, `v.p CONTAINS 'lit'`,
/// `toLower(v.p) CONTAINS 'lit'`, `v.p =~ 'regex'` or
/// `text_match_regex(v.p, 'regex')`, with every argument but `v.p` a literal
/// or a `$parameter` known at plan time, and `v` the *first* node of its
/// pattern (the node the matcher starts from after the start-node passes),
/// bound by this clause alone, with exactly one label, no inline property of
/// the same name and no other property filter (inline or pushed from the
/// WHERE).
///
/// **Rewrite:** a [`PropertyMatcher::TextScan`] joins `v`'s properties. When
/// no index serves the pattern, the matcher asks the column driver for the
/// candidates; the pattern's property filter and the WHERE predicate, both
/// untouched, then decide every candidate. A `CONTAINS` already pushed into
/// the pattern is wrapped (its own test kept) instead of added.
///
/// **Why the candidates are a superset:** the patterns scanned find every
/// value the predicate accepts and possibly more. `text_match` scans its own
/// pattern. `CONTAINS` scans its literal. `=~` and `text_match_regex` scan the
/// regex only when it has a literal every match must contain (so it cannot
/// match the empty string, and a full match implies a search match). `toLower`
/// scans the literal case-insensitively, plus `İ` (the one character whose
/// lowercase is two characters) when the literal could overlap its expansion,
/// plus each character the regex crate's older case tables do not fold to a
/// character the literal names (`fold_gaps`); the unit tests over every
/// Unicode scalar check these claims.
///
/// **Why-bail** (the plan is left exactly as written): `OPTIONAL MATCH`; a
/// predicate under `NOT`, `OR` or `XOR`; a non-constant argument; a label-less,
/// multi-label or alternation node; a node bound by an earlier clause or not
/// first in its pattern (reached by expansion); another property filter on the
/// node, which an index may serve and a fused count already scans; a
/// second text predicate on the same property; a literal that is empty, or
/// for `toLower` not already lowercase (the predicate can never hold); a regex
/// without a required literal; `text_match_regex` flags; `STARTS WITH` and
/// `ENDS WITH` (the prefix index and the per-row test already serve them);
/// `toUpper` (several characters uppercase to more than one); a query that
/// does not compile (the per-row evaluation reports the error).
///
/// **At execution** the driver declines on disk graphs, secondary-label
/// carriers, ontology closures, id/title and soft-alias fields, columns it
/// cannot prove (see `text_column_scan.rs`), and when more than a tenth of the
/// type matches (probed on the first sixteenth of the type, then capped on the
/// full scan), and the matcher then scans as it always did. When another candidate source (an index, a bound variable) is
/// available it is used instead.
pub(super) fn pass(query: &mut CypherQuery, ctx: &PassCtx) {
    let mut i = 0;
    while i + 1 < query.clauses.len() {
        if !matches!(query.clauses[i], Clause::Match(_)) {
            i += 1;
            continue;
        }
        let Clause::Where(w) = &query.clauses[i + 1] else {
            i += 1;
            continue;
        };
        let mut conjuncts = Vec::new();
        collect_conjuncts(&w.predicate, &mut conjuncts);
        let found: Vec<Recognised> = conjuncts
            .into_iter()
            .filter_map(|c| recognise(c, ctx.params))
            .collect();
        let bound_earlier = bound_before(&query.clauses[..i]);
        if let Clause::Match(m) = &mut query.clauses[i] {
            for hit in found {
                if !bound_earlier.contains(&hit.variable) {
                    attach(&mut m.patterns, hit);
                }
            }
        }
        i += 1;
    }
}

fn collect_conjuncts<'a>(pred: &'a Predicate, out: &mut Vec<&'a Predicate>) {
    match pred {
        Predicate::And(l, r) => {
            collect_conjuncts(l, out);
            collect_conjuncts(r, out);
        }
        other => out.push(other),
    }
}

fn bound_before(clauses: &[Clause]) -> HashSet<String> {
    let mut vars = HashSet::new();
    for clause in clauses {
        match clause {
            Clause::Match(m) => vars.extend(
                super::index_selection::collect_pattern_variables(&m.patterns)
                    .into_iter()
                    .map(|(v, _)| v),
            ),
            Clause::OptionalMatch(m) => vars.extend(
                super::index_selection::collect_pattern_variables(&m.patterns)
                    .into_iter()
                    .map(|(v, _)| v),
            ),
            _ => {}
        }
    }
    vars
}

/// Put the matcher on the variable's node if it is the first, single-label
/// node of its pattern and nothing else claims the property.
fn attach(patterns: &mut [crate::graph::core::pattern_matching::Pattern], hit: Recognised) {
    let mut seen = 0;
    let mut target = None;
    for (pi, pattern) in patterns.iter().enumerate() {
        for (ei, element) in pattern.elements.iter().enumerate() {
            if let PatternElement::Node(np) = element {
                if np.variable.as_deref() == Some(hit.variable.as_str()) {
                    seen += 1;
                    target = Some((pi, ei));
                }
            }
        }
    }
    // Bound twice in one clause: leave it.
    let (Some((pi, ei)), 1) = (target, seen) else {
        return;
    };
    if ei != 0 {
        return;
    }
    let PatternElement::Node(np) = &mut patterns[pi].elements[ei] else {
        return;
    };
    if np.node_type.is_none() || np.multi_label_constrained() || !np.label_params.is_empty() {
        return;
    }
    // Another property filter on the node (an index may serve it, and a
    // fused count scans the column with it): the existing path is already
    // cheap there, and measured no faster with a scan in front.
    if np.properties.iter().flatten().any(|(key, matcher)| {
        key != &hit.property
            && !matches!(
                matcher,
                PropertyMatcher::TextScan(_) | PropertyMatcher::Contains(_)
            )
    }) {
        return;
    }
    let props = np.properties.get_or_insert_with(HashMap::new);
    match props.get(&hit.property) {
        None => {
            props.insert(
                hit.property,
                PropertyMatcher::TextScan(Arc::new(TextScanSpec::new(hit.patterns, None))),
            );
        }
        Some(PropertyMatcher::Contains(needle)) if hit.patterns.len() == 1 => {
            // The pushed CONTAINS matcher keeps deciding per node.
            let inner = PropertyMatcher::Contains(needle.clone());
            props.insert(
                hit.property,
                PropertyMatcher::TextScan(Arc::new(TextScanSpec::new(hit.patterns, Some(inner)))),
            );
        }
        Some(_) => {}
    }
}

fn plan_value(expr: &Expression, params: &HashMap<String, Value>) -> Option<Value> {
    match expr {
        Expression::Literal(v) => Some(v.clone()),
        Expression::Parameter(name) => params.get(name.as_str()).cloned(),
        Expression::MapLiteral(entries) => {
            let mut pairs = Vec::with_capacity(entries.len());
            for (key, value) in entries {
                pairs.push((key.clone(), plan_value(value, params)?));
            }
            Some(Value::Map(crate::datatypes::prop_map::PropMap::from_pairs(
                pairs,
            )))
        }
        _ => None,
    }
}

fn plan_string(expr: &Expression, params: &HashMap<String, Value>) -> Option<String> {
    match plan_value(expr, params)? {
        Value::String(s) if !s.is_empty() => Some(s),
        _ => None,
    }
}

fn property_of(expr: &Expression) -> Option<(String, String)> {
    match expr {
        Expression::PropertyAccess { variable, property } => {
            Some((variable.clone(), property.clone()))
        }
        _ => None,
    }
}

fn recognise(pred: &Predicate, params: &HashMap<String, Value>) -> Option<Recognised> {
    match pred {
        // A bare boolean function parses as `f(..) <> false`.
        Predicate::Comparison {
            left: Expression::FunctionCall { name, args, .. },
            operator: ComparisonOp::NotEquals,
            right: Expression::Literal(Value::Boolean(false)),
        } => recognise_function(name, args, params),
        Predicate::Comparison {
            left,
            operator: ComparisonOp::RegexMatch,
            right,
        } => {
            let (variable, property) = property_of(left)?;
            let regex = plan_string(right, params)?;
            regex_patterns(variable, property, regex)
        }
        Predicate::Contains { expr, pattern } => {
            let literal = plan_string(pattern, params)?;
            if let Some((variable, property)) = property_of(expr) {
                return Some(Recognised {
                    variable,
                    property,
                    patterns: vec![ScanPattern::Literal {
                        text: literal,
                        case_insensitive: false,
                    }],
                });
            }
            let Expression::FunctionCall { name, args, .. } = expr else {
                return None;
            };
            if !matches!(
                name.to_ascii_lowercase().as_str(),
                "tolower" | "tolowercase"
            ) || args.len() != 1
            {
                return None;
            }
            let (variable, property) = property_of(&args[0])?;
            lower_patterns(variable, property, literal)
        }
        _ => None,
    }
}

fn recognise_function(
    name: &str,
    args: &[Expression],
    params: &HashMap<String, Value>,
) -> Option<Recognised> {
    match name.to_ascii_lowercase().as_str() {
        "text_match" if args.len() == 2 || args.len() == 3 => {
            let (variable, property) = property_of(&args[0])?;
            let text = plan_string(&args[1], params)?;
            let options = match args.get(2) {
                Some(expr) => Some(plan_value(expr, params)?),
                None => None,
            };
            let query = TextQuery::new(&text, options.as_ref(), "text_match()").ok()?;
            query.compile("text_match()").ok()?;
            Some(Recognised {
                variable,
                property,
                patterns: vec![query.pattern()],
            })
        }
        "text_match_regex" if args.len() == 2 => {
            let (variable, property) = property_of(&args[0])?;
            let regex = plan_string(&args[1], params)?;
            regex_patterns(variable, property, regex)
        }
        _ => None,
    }
}

/// A regex scan is a superset only when every match must contain a literal.
fn regex_patterns(variable: String, property: String, regex: String) -> Option<Recognised> {
    required_literal(&regex, false)?;
    crate::graph::text_scan::matcher::Matcher::compile(&ScanPattern::Regex {
        pattern: regex.clone(),
        case_insensitive: false,
    })
    .ok()?;
    Some(Recognised {
        variable,
        property,
        patterns: vec![ScanPattern::Regex {
            pattern: regex,
            case_insensitive: false,
        }],
    })
}

/// `(upper, lower)` pairs where `lower` is the single-character lowercase of
/// `upper` but `(?i)lower` does not match `upper`: the regex crate's case
/// tables trail the standard library's Unicode version, so its case-insensitive
/// scan would miss `upper`. Computed once, by comparing the two over every
/// scalar value.
fn fold_gaps() -> &'static [(char, char)] {
    static GAPS: std::sync::LazyLock<Vec<(char, char)>> = std::sync::LazyLock::new(|| {
        (0..=0x10FFFFu32)
            .filter_map(char::from_u32)
            .filter_map(|c| {
                let mut lower = c.to_lowercase();
                let y = lower.next()?;
                (lower.next().is_none() && y != c).then_some((c, y))
            })
            .filter(|(c, y)| {
                regex::Regex::new(&format!("(?i){}", regex::escape(&y.to_string())))
                    .map_or(true, |re| !re.is_match(&c.to_string()))
            })
            .collect()
    });
    &GAPS
}

/// `U+0130`, the one scalar whose lowercase is more than one character
/// (`i` + U+0307).
pub(crate) const DOTTED_CAPITAL_I: char = '\u{130}';

fn lower_patterns(variable: String, property: String, literal: String) -> Option<Recognised> {
    // `toLower(x)` is lowercase, so a literal with an uppercase letter never matches.
    if literal.to_lowercase() != literal {
        return None;
    }
    let mut patterns = vec![ScanPattern::Literal {
        text: literal.clone(),
        case_insensitive: true,
    }];
    // A match may overlap the expansion `i` + U+0307 of `İ`.
    if literal.contains('i') || literal.contains('\u{307}') {
        patterns.push(ScanPattern::Literal {
            text: DOTTED_CAPITAL_I.to_string(),
            case_insensitive: false,
        });
    }
    // Characters the regex crate does not case-fold to one the literal names.
    for (upper, lower) in fold_gaps() {
        if literal.contains(*lower) {
            patterns.push(ScanPattern::Literal {
                text: upper.to_string(),
                case_insensitive: false,
            });
        }
    }
    Some(Recognised {
        variable,
        property,
        patterns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::text_scan::matcher::Matcher;
    use crate::graph::text_scan::spec::has_match;

    /// `toLower` routing rests on two facts about Unicode, checked here over
    /// every scalar value so that a Unicode update that breaks either turns
    /// this red: only `İ` lowercases to more than one character, and every
    /// other character that changes case is matched by `(?i)` of its
    /// lowercase.
    #[test]
    fn to_lower_routing_rests_on_facts_checked_over_every_scalar() {
        let mut multi = Vec::new();
        let mut single_changed = Vec::new();
        for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            let mut lower = c.to_lowercase();
            let first = lower.next().unwrap();
            if lower.next().is_some() {
                multi.push(c);
            } else if first != c {
                single_changed.push((c, first));
            }
        }
        assert_eq!(
            multi,
            [DOTTED_CAPITAL_I],
            "characters with a multi-character lowercase"
        );
        // The characters `(?i)` does not fold are exactly `fold_gaps()`: the
        // regex crate's tables lag the standard library's Unicode version.
        let gaps: Vec<(char, char)> = single_changed
            .into_iter()
            .filter(|(c, y)| {
                let re =
                    regex::Regex::new(&format!("(?i){}", regex::escape(&y.to_string()))).unwrap();
                !re.is_match(&c.to_string())
            })
            .collect();
        assert_eq!(gaps, fold_gaps());
    }

    /// Every substring of `toLower(value)` finds `value` through the
    /// patterns the pass scans, over all strings of up to three characters
    /// from a pool of the awkward ones (sigmas, dotted and dotless i, the
    /// Kelvin and Angstrom signs, sharp s).
    #[test]
    fn every_substring_of_a_lowercased_value_finds_the_value() {
        // Includes `꟎` (a capital newer than the regex crate's tables) and `ɷ`,
        // the lowercase it maps to.
        let pool: Vec<char> = "aAbBiIıİkKKΣσςßẞåÅÅ\u{307}꟎ɷ-".chars().collect();
        let mut cache: HashMap<String, Vec<Matcher>> = HashMap::new();
        let mut values = vec![String::new()];
        for _ in 0..3 {
            let next: Vec<String> = values
                .iter()
                .flat_map(|v| pool.iter().map(move |c| format!("{v}{c}")))
                .collect();
            for value in &next {
                let lowered = value.to_lowercase();
                let bounds: Vec<usize> = lowered
                    .char_indices()
                    .map(|(i, _)| i)
                    .chain([lowered.len()])
                    .collect();
                for (a, &start) in bounds.iter().enumerate() {
                    for &end in &bounds[a + 1..] {
                        let literal = lowered[start..end].to_string();
                        let matchers = cache.entry(literal.clone()).or_insert_with(|| {
                            lower_patterns("v".into(), "p".into(), literal.clone())
                                .expect("a lowercase literal routes")
                                .patterns
                                .iter()
                                .map(|p| Matcher::compile(p).unwrap())
                                .collect()
                        });
                        assert!(
                            matchers.iter().any(|m| has_match(m, value)),
                            "toLower({value:?}) contains {literal:?} but the scan misses it"
                        );
                    }
                }
            }
            values = next;
        }
    }

    #[test]
    fn a_literal_with_an_uppercase_letter_is_not_routed() {
        assert!(lower_patterns("v".into(), "p".into(), "Alpha".into()).is_none());
    }
}
