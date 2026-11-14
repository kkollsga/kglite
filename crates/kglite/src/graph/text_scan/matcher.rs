//! Compiled queries and the per-record oracle.
//!
//! [`Matcher::find_in_record`] is the definition of a correct answer: the
//! leftmost, non-overlapping, non-empty matches of the query in one record's
//! bytes, as if that record were the whole haystack. It doubles as the overlay
//! matcher for text that lives outside the layout. The scan paths in
//! [`super::scan`] are measured against it.
//!
//! # Which query takes which path
//!
//! * **Case-sensitive literal** (`memmem`) and **case-insensitive literal**
//!   (`(?i)` over the escaped text): scanned as one haystack. A match found
//!   inside a record is a hit; one that crosses a record boundary is not, and
//!   the search resumes at the *start of the next record*. That is exact for
//!   these two because every match of one fixed-length needle (fixed in bytes,
//!   or in characters under simple case folding, with record boundaries always
//!   on character boundaries) that starts after a straddling match's start in
//!   the same record straddles too.
//! * **Phrase** and **regex**: whole-slice context is wrong for them (`^`,
//!   `$`, `\b`, `\A`, `\z`, and a preferred match that straddles hiding a
//!   shorter in-record one), so a hit is only ever produced by running the
//!   regex on the isolated record. A required literal, when one can be
//!   extracted ([`required_literal`]), selects the candidate records through
//!   `memmem`; with none, every record is tried.
//!
//! Case folding is whatever the `regex` crate's `(?i)` does on the record
//! (simple Unicode case folding): `ß` does not match `SS`, `ẞ` matches `ß`, and
//! `İ` does not match `i`. Tests pin these against the per-record oracle.

use memchr::memmem;

#[derive(Debug, Clone)]
pub(crate) enum Pattern {
    Literal {
        text: String,
        case_insensitive: bool,
    },
    /// Words separated by whitespace; the gap matches one or more whitespace
    /// characters of any kind (values may contain newlines).
    Phrase {
        text: String,
        case_insensitive: bool,
    },
    Regex {
        pattern: String,
        case_insensitive: bool,
    },
}

#[derive(Debug)]
pub(crate) enum Engine {
    Memmem(memmem::Finder<'static>),
    /// A case-insensitive literal, scanned as one haystack.
    CiLiteral(regex::bytes::Regex),
    /// Confirmed per record, optionally behind a literal prefilter.
    Regex {
        re: regex::bytes::Regex,
        prefilter: Option<memmem::Finder<'static>>,
    },
}

#[derive(Debug)]
pub(crate) struct Matcher {
    pub(crate) engine: Engine,
}

fn build_regex(pattern: &str, ci: bool) -> Result<regex::bytes::Regex, String> {
    let re = regex::bytes::RegexBuilder::new(pattern)
        .case_insensitive(ci)
        .unicode(true)
        .build()
        .map_err(|e| format!("invalid pattern: {e}"))?;
    if re.is_match(b"") {
        return Err("the pattern matches the empty string, so every position would hit".into());
    }
    Ok(re)
}

/// A literal every match of `pattern` must contain, extracted without parsing
/// the regex: the longest run of at least 3 bytes made of plain characters at
/// the top level (outside groups and classes).
///
/// Declines (returns `None`) whenever the text cannot prove the literal is
/// required: case-insensitive matching, any flag group (`(?i)`, `(?x)` ...), a
/// top-level `|`, or no qualifying run. A character followed by `*`, `?` or `{`
/// is optional and is dropped from its run; `+` keeps it. Plain means
/// alphanumeric, space, `_` or `-`; everything else (escapes, `.`, anchors,
/// classes) ends a run, so the literal is always a verbatim substring of every
/// match.
pub(crate) fn required_literal(pattern: &str, ci: bool) -> Option<String> {
    if ci
        || pattern
            .match_indices("(?")
            .any(|(i, _)| !pattern[i..].starts_with("(?:"))
    {
        return None;
    }
    let mut scan = LiteralScan::default();
    for c in pattern.chars() {
        scan.feed(c)?;
    }
    scan.end_run();
    (!scan.best.is_empty()).then_some(scan.best)
}

/// Character-at-a-time state for [`required_literal`].
#[derive(Default)]
struct LiteralScan {
    depth: usize,
    class: bool,
    escaped: bool,
    brace: bool,
    run: String,
    best: String,
}

impl LiteralScan {
    fn end_run(&mut self) {
        if self.run.len() >= 3 && self.run.len() > self.best.len() {
            self.best = self.run.clone();
        }
        self.run.clear();
    }

    /// Whether `c` is inside an escape, class or `{..}` and so carries no
    /// literal; updates that state.
    fn in_skipped_syntax(&mut self, c: char) -> bool {
        if self.escaped {
            self.escaped = false;
        } else if self.class {
            self.class = c != ']';
        } else if self.brace {
            self.brace = c != '}';
        } else {
            return false;
        }
        true
    }

    /// `None` when a top-level `|` makes every literal optional.
    fn feed(&mut self, c: char) -> Option<()> {
        if self.in_skipped_syntax(c) {
            return Some(());
        }
        match c {
            '\\' => self.escaped = true,
            '[' => self.class = true,
            '(' => self.depth += 1,
            ')' => self.depth = self.depth.saturating_sub(1),
            _ if self.depth > 0 => return Some(()),
            '|' => return None,
            '*' | '?' | '{' => {
                self.brace = c == '{';
                self.run.pop();
            }
            _ if c.is_alphanumeric() || matches!(c, ' ' | '_' | '-') => {
                self.run.push(c);
                return Some(());
            }
            _ => {}
        }
        self.end_run();
        Some(())
    }
}

impl Matcher {
    pub(crate) fn compile(pattern: &Pattern) -> Result<Self, String> {
        let engine = match pattern {
            Pattern::Literal {
                text,
                case_insensitive,
            } => Self::literal(text, *case_insensitive)?,
            Pattern::Phrase {
                text,
                case_insensitive,
            } => {
                let words: Vec<String> = text.split_whitespace().map(regex::escape).collect();
                match words.as_slice() {
                    [] => return Err("the phrase is empty".into()),
                    [_] => Self::literal(text.trim(), *case_insensitive)?,
                    _ => Self::regex(&words.join("\\s+"), *case_insensitive)?,
                }
            }
            Pattern::Regex {
                pattern,
                case_insensitive,
            } => Self::regex(pattern, *case_insensitive)?,
        };
        Ok(Self { engine })
    }

    fn literal(text: &str, ci: bool) -> Result<Engine, String> {
        if text.is_empty() {
            return Err("the query is empty".into());
        }
        Ok(if ci {
            Engine::CiLiteral(build_regex(&regex::escape(text), true)?)
        } else {
            Engine::Memmem(memmem::Finder::new(text.as_bytes()).into_owned())
        })
    }

    fn regex(pattern: &str, ci: bool) -> Result<Engine, String> {
        Ok(Engine::Regex {
            re: build_regex(pattern, ci)?,
            prefilter: required_literal(pattern, ci)
                .map(|lit| memmem::Finder::new(lit.as_bytes()).into_owned()),
        })
    }

    /// First match of the *whole-slice* engines at or after `from`. Only
    /// meaningful for [`Engine::Memmem`] and [`Engine::CiLiteral`].
    pub(crate) fn find_at(&self, hay: &[u8], from: usize) -> Option<(usize, usize)> {
        if from > hay.len() {
            return None;
        }
        match &self.engine {
            Engine::Memmem(f) => f
                .find(&hay[from..])
                .map(|s| (from + s, from + s + f.needle().len())),
            Engine::CiLiteral(re) | Engine::Regex { re, .. } => {
                re.find_at(hay, from).map(|m| (m.start(), m.end()))
            }
        }
    }

    /// **The oracle.** Non-empty, non-overlapping, leftmost matches in
    /// `record` taken as the whole haystack, handed to `sink` until it
    /// returns `false`.
    pub(crate) fn find_in_record(&self, record: &[u8], sink: &mut dyn FnMut(usize, usize) -> bool) {
        match &self.engine {
            Engine::Memmem(f) => {
                for s in f.find_iter(record) {
                    if !sink(s, s + f.needle().len()) {
                        return;
                    }
                }
            }
            Engine::CiLiteral(re) | Engine::Regex { re, .. } => {
                for m in re.find_iter(record).filter(|m| !m.is_empty()) {
                    if !sink(m.start(), m.end()) {
                        return;
                    }
                }
            }
        }
    }

    /// The matches in `text` as `(offset, length)`: the overlay matcher for a
    /// value that is not in the layout.
    pub(crate) fn overlay_hits(&self, text: &str) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        self.find_in_record(text.as_bytes(), &mut |s, e| {
            out.push((s, e - s));
            true
        });
        out
    }
}
