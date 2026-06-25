//! rustyline `Helper` for the shell: multi-line input (a `Validator` that keeps
//! reading while brackets/quotes are unbalanced) + tab-completion of
//! dot-commands and the live graph's labels / relationship types.
//!
//! The completion candidate list is a snapshot refreshed each prompt by the
//! REPL (via `set_candidates`), so the completer never needs to borrow the
//! graph.

use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Context, Helper};

/// Dot-commands offered for completion (and their bare names).
pub const DOT_COMMANDS: &[&str] = &[
    ".help", ".quit", ".exit", ".labels", ".rels", ".schema", ".indexes", ".mode", ".dump",
    ".read", ".import", ".save", ".timing",
];

#[derive(Default)]
pub struct ShellHelper {
    /// Live-graph completion candidates (labels + relationship types),
    /// refreshed each prompt.
    candidates: Vec<String>,
}

impl ShellHelper {
    pub fn set_candidates(&mut self, candidates: Vec<String>) {
        self.candidates = candidates;
    }
}

impl Completer for ShellHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        // The word under the cursor: from the last whitespace boundary to pos.
        let start = line[..pos]
            .rfind(|c: char| c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        let word = &line[start..pos];

        // A leading-`.` word completes against the dot-commands; any other word
        // completes against the live labels / relationship types.
        let pool: Vec<&str> = if word.starts_with('.') {
            DOT_COMMANDS.to_vec()
        } else {
            self.candidates.iter().map(String::as_str).collect()
        };
        let matches: Vec<Pair> = pool
            .into_iter()
            .filter(|cand| cand.starts_with(word))
            .map(|cand| Pair {
                display: cand.to_string(),
                replacement: cand.to_string(),
            })
            .collect();
        Ok((start, matches))
    }
}

impl Validator for ShellHelper {
    /// sqlite3-style termination: a Cypher statement runs when it ends with `;`
    /// (with brackets/quotes balanced, so a `;` inside a string doesn't count);
    /// otherwise the prompt keeps reading (multi-line). Dot-commands and empty
    /// input run on Enter — they never need a terminator.
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        let input = ctx.input();
        let trimmed = input.trim();
        if trimmed.is_empty() || trimmed.starts_with('.') {
            return Ok(ValidationResult::Valid(None));
        }
        if trimmed.ends_with(';') && is_balanced(input) {
            Ok(ValidationResult::Valid(None))
        } else {
            Ok(ValidationResult::Incomplete)
        }
    }
}

impl Hinter for ShellHelper {
    type Hint = String;
}

impl Highlighter for ShellHelper {}

impl Helper for ShellHelper {}

/// True when all `()`/`[]`/`{}` are matched and quotes (`'` and `"`) are closed,
/// ignoring bracket characters that appear inside a string literal.
fn is_balanced(s: &str) -> bool {
    let mut depth: i32 = 0;
    let mut quote: Option<char> = None;
    let mut prev_backslash = false;
    for c in s.chars() {
        match quote {
            Some(q) => {
                if c == q && !prev_backslash {
                    quote = None;
                }
            }
            None => match c {
                '\'' | '"' => quote = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            },
        }
        prev_backslash = c == '\\' && !prev_backslash;
    }
    quote.is_none() && depth <= 0
}

#[cfg(test)]
mod tests {
    use super::is_balanced;

    #[test]
    fn balanced_single_line() {
        assert!(is_balanced("MATCH (n) RETURN n"));
        assert!(is_balanced("RETURN 1"));
    }

    #[test]
    fn unbalanced_is_incomplete() {
        assert!(!is_balanced("MATCH (n")); // open paren
        assert!(!is_balanced("RETURN [1, 2")); // open bracket
        assert!(!is_balanced("RETURN 'oops")); // open quote
    }

    #[test]
    fn quotes_shield_brackets() {
        assert!(is_balanced("RETURN '(' AS p")); // paren inside string
        assert!(is_balanced("RETURN \"a)b\" AS s"));
    }
}
