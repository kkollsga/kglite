//! Cypher parser: MATCH / OPTIONAL MATCH clause + pattern extraction.

use super::super::ast::*;
use super::super::tokenizer::CypherToken;
use super::CypherParser;

impl CypherParser {
    // ========================================================================
    // MATCH Clause
    // ========================================================================

    pub(super) fn parse_match_clause(&mut self, optional: bool) -> Result<Clause, String> {
        self.expect(&CypherToken::Match)?;

        let mut path_assignments = Vec::new();

        // Check for path assignment: p = shortestPath(...)
        // Pattern: Identifier Equals [Identifier("shortestPath") LParen] pattern [RParen]
        if self.is_path_assignment() {
            let path_var = self.consume_identifier()?;
            self.expect(&CypherToken::Equals)?;

            // Check for shortestPath( wrapper
            let is_shortest = self.is_shortest_path_call();
            if is_shortest {
                self.advance(); // consume "shortestPath" identifier
                self.expect(&CypherToken::LParen)?;
            }

            let patterns = self.parse_match_patterns()?;

            if is_shortest {
                self.expect(&CypherToken::RParen)?;
            }

            path_assignments.push(PathAssignment {
                variable: path_var,
                pattern_index: 0,
                is_shortest_path: is_shortest,
            });

            let clause = MatchClause {
                patterns,
                path_assignments,
                limit_hint: None,
                distinct_node_hint: None,
            };
            return if optional {
                Ok(Clause::OptionalMatch(clause))
            } else {
                Ok(Clause::Match(clause))
            };
        }

        // Normal MATCH clause
        let patterns = self.parse_match_patterns()?;

        let clause = MatchClause {
            patterns,
            path_assignments,
            limit_hint: None,
            distinct_node_hint: None,
        };
        if optional {
            Ok(Clause::OptionalMatch(clause))
        } else {
            Ok(Clause::Match(clause))
        }
    }

    /// Check if current position looks like: Identifier = [shortestPath(] ...
    pub(super) fn is_path_assignment(&self) -> bool {
        matches!(self.peek(), Some(CypherToken::Identifier(_)))
            && self.peek_at(1) == Some(&CypherToken::Equals)
    }

    /// Check if current position is shortestPath( — called AFTER consuming "var ="
    pub(super) fn is_shortest_path_call(&self) -> bool {
        if let Some(CypherToken::Identifier(name)) = self.peek() {
            name.eq_ignore_ascii_case("shortestPath")
                && self.peek_at(1) == Some(&CypherToken::LParen)
        } else {
            false
        }
    }

    /// Consume an identifier token and return the string
    pub(super) fn consume_identifier(&mut self) -> Result<String, String> {
        match self.advance() {
            Some(CypherToken::Identifier(s)) => Ok(s.clone()),
            other => Err(format!("Expected identifier, got {:?}", other)),
        }
    }

    /// Parse one or more comma-separated patterns in MATCH
    pub(super) fn parse_match_patterns(
        &mut self,
    ) -> Result<Vec<crate::graph::core::pattern_matching::Pattern>, String> {
        let mut patterns = Vec::new();

        loop {
            // Reconstruct the pattern string from tokens until we hit a comma (at top-level)
            // or a clause boundary
            let pattern_str = self.extract_pattern_string()?;
            if pattern_str.is_empty() {
                return Err("Expected a pattern in MATCH clause".to_string());
            }

            let pattern = crate::graph::core::pattern_matching::parse_pattern(&pattern_str)
                .map_err(|e| format!("Pattern parse error: {}", e))?;
            patterns.push(pattern);

            // Check for comma to continue with more patterns
            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(patterns)
    }

    /// Parse patterns inside EXISTS { ... } — same as parse_match_patterns but uses
    /// extract_exists_pattern_string which stops at RBrace instead of clause boundaries.
    pub(super) fn parse_exists_patterns(
        &mut self,
    ) -> Result<Vec<crate::graph::core::pattern_matching::Pattern>, String> {
        // Default delimiter for `EXISTS { ... }` / `count { ... }`: closing brace.
        self.parse_pattern_subquery_patterns(&CypherToken::RBrace)
    }

    /// Parse one or more comma/MATCH-separated patterns until a delimiter
    /// token at top level. Used by EXISTS/count (delimiter = `}`) and the
    /// 0.9.0 §6 `size((pattern))` form (delimiter = `)`).
    pub(super) fn parse_pattern_subquery_patterns(
        &mut self,
        end_token: &CypherToken,
    ) -> Result<Vec<crate::graph::core::pattern_matching::Pattern>, String> {
        let mut patterns = Vec::new();

        loop {
            let pattern_str = self.extract_pattern_subquery_string(end_token)?;
            if pattern_str.is_empty() {
                if patterns.is_empty() {
                    return Err("Expected a pattern inside EXISTS { }".to_string());
                }
                break;
            }

            let pattern = crate::graph::core::pattern_matching::parse_pattern(&pattern_str)
                .map_err(|e| format!("Pattern parse error in EXISTS: {}", e))?;
            patterns.push(pattern);

            if self.check(&CypherToken::Comma) {
                self.advance();
            } else if self.check(&CypherToken::Match) {
                // Subquery form: EXISTS { MATCH (a)-[:R]->(b) MATCH (c)-[:R2]->(d) ... }
                // Don't advance — the next iteration's
                // extract_exists_pattern_string will skip the MATCH at its
                // start (same path as the optional first MATCH).
            } else {
                break;
            }
        }

        Ok(patterns)
    }

    /// Extract tokens forming a pattern inside EXISTS { ... }, stopping at RBrace or comma.
    /// Re-serialize an identifier, adding backticks if it contains spaces or special chars.
    pub(super) fn quote_identifier(s: &str) -> String {
        if s.contains(' ')
            || s.contains('-')
            || s.contains('/')
            || s.contains('.')
            || s.contains('(')
            || s.contains(')')
        {
            format!("`{}`", s)
        } else {
            s.to_string()
        }
    }

    /// Re-serialize tokens forming a pattern, stopping at the supplied
    /// delimiter (RBrace for EXISTS/count, RParen for size).
    pub(super) fn extract_pattern_subquery_string(
        &mut self,
        end_token: &CypherToken,
    ) -> Result<String, String> {
        // Skip optional MATCH keyword — standard Cypher allows EXISTS { MATCH (pattern) }
        if self.check(&CypherToken::Match) {
            self.advance();
        }

        let mut parts = Vec::new();
        let mut paren_depth = 0i32;
        let mut bracket_depth = 0i32;

        while self.has_tokens() {
            // Stop at the caller-supplied end-token (RBrace for EXISTS,
            // RParen for size). Only at top level (depth 0).
            if paren_depth == 0 && bracket_depth == 0 && self.check(end_token) {
                break;
            }

            // Stop at comma at top level (pattern separator)
            if paren_depth == 0 && bracket_depth == 0 && self.check(&CypherToken::Comma) {
                break;
            }

            // Stop at WHERE keyword (EXISTS { MATCH ... WHERE ... } subquery)
            if paren_depth == 0 && bracket_depth == 0 && self.check(&CypherToken::Where) {
                break;
            }

            // Stop at MATCH keyword at top level — multi-MATCH subquery
            // form (`EXISTS { MATCH ... MATCH ... }`). The outer
            // parse_exists_patterns loop continues on this case.
            if paren_depth == 0 && bracket_depth == 0 && self.check(&CypherToken::Match) {
                break;
            }

            let token = self.advance().unwrap().clone();

            match &token {
                CypherToken::LParen => {
                    paren_depth += 1;
                    parts.push("(".to_string());
                }
                CypherToken::RParen => {
                    paren_depth -= 1;
                    parts.push(")".to_string());
                }
                CypherToken::LBracket => {
                    bracket_depth += 1;
                    parts.push("[".to_string());
                }
                CypherToken::RBracket => {
                    bracket_depth -= 1;
                    parts.push("]".to_string());
                }
                CypherToken::LBrace => parts.push("{".to_string()),
                CypherToken::RBrace => parts.push("}".to_string()),
                CypherToken::Colon => parts.push(":".to_string()),
                CypherToken::Comma => parts.push(",".to_string()),
                CypherToken::Dash => parts.push("-".to_string()),
                CypherToken::GreaterThan => parts.push(">".to_string()),
                CypherToken::LessThan => parts.push("<".to_string()),
                CypherToken::Star => parts.push("*".to_string()),
                CypherToken::DotDot => parts.push("..".to_string()),
                CypherToken::Dot => parts.push(".".to_string()),
                CypherToken::Identifier(s) => parts.push(Self::quote_identifier(s)),
                CypherToken::StringLit(s) => {
                    // Re-escape quotes so the pattern parser can re-tokenize correctly
                    let escaped = s.replace('\\', "\\\\").replace('\'', "\\'");
                    parts.push(format!("'{}'", escaped));
                }
                CypherToken::IntLit(n) => parts.push(n.to_string()),
                CypherToken::FloatLit(f) => parts.push(f.to_string()),
                CypherToken::True => parts.push("true".to_string()),
                CypherToken::False => parts.push("false".to_string()),
                _ => {
                    return Err(format!("Unexpected token in EXISTS pattern: {:?}", token));
                }
            }
        }

        Ok(parts.join(" "))
    }

    /// Extract tokens forming a single pattern and reconstruct as a string
    /// for the existing pattern_matching parser.
    /// Stops at commas (outside parens/brackets), clause keywords, or end of input.
    pub(super) fn extract_pattern_string(&mut self) -> Result<String, String> {
        let mut parts = Vec::new();
        let mut paren_depth = 0i32;
        let mut bracket_depth = 0i32;

        while self.has_tokens() {
            // Stop at clause boundaries (only at top level)
            if paren_depth == 0 && bracket_depth == 0 && self.at_clause_boundary() {
                break;
            }

            // Stop at comma at top level (pattern separator)
            if paren_depth == 0 && bracket_depth == 0 && self.check(&CypherToken::Comma) {
                break;
            }

            // Stop at AND/OR at top level (boolean operators in WHERE)
            if paren_depth == 0
                && bracket_depth == 0
                && (self.check(&CypherToken::And) || self.check(&CypherToken::Or))
            {
                break;
            }

            // Stop at RParen that would go negative (e.g. closing shortestPath(...))
            if paren_depth == 0 && self.check(&CypherToken::RParen) {
                break;
            }

            let token = self.advance().unwrap().clone();

            match &token {
                CypherToken::LParen => {
                    paren_depth += 1;
                    parts.push("(".to_string());
                }
                CypherToken::RParen => {
                    paren_depth -= 1;
                    parts.push(")".to_string());
                }
                CypherToken::LBracket => {
                    bracket_depth += 1;
                    parts.push("[".to_string());
                }
                CypherToken::RBracket => {
                    bracket_depth -= 1;
                    parts.push("]".to_string());
                }
                CypherToken::LBrace => parts.push("{".to_string()),
                CypherToken::RBrace => parts.push("}".to_string()),
                CypherToken::Colon => parts.push(":".to_string()),
                CypherToken::Comma => parts.push(",".to_string()),
                CypherToken::Dash => parts.push("-".to_string()),
                CypherToken::GreaterThan => parts.push(">".to_string()),
                CypherToken::LessThan => parts.push("<".to_string()),
                CypherToken::Star => parts.push("*".to_string()),
                CypherToken::DotDot => parts.push("..".to_string()),
                CypherToken::Dot => parts.push(".".to_string()),
                CypherToken::Pipe => parts.push("|".to_string()),
                CypherToken::Identifier(s) => parts.push(Self::quote_identifier(s)),
                CypherToken::StringLit(s) => {
                    let escaped = s.replace('\\', "\\\\").replace('\'', "\\'");
                    parts.push(format!("'{}'", escaped));
                }
                CypherToken::IntLit(n) => parts.push(n.to_string()),
                CypherToken::FloatLit(f) => parts.push(f.to_string()),
                CypherToken::True => parts.push("true".to_string()),
                CypherToken::False => parts.push("false".to_string()),
                CypherToken::Parameter(name) => {
                    parts.push(format!("${}", name));
                }
                _ => {
                    return Err(format!("Unexpected token in MATCH pattern: {:?}", token));
                }
            }
        }

        Ok(parts.join(""))
    }

    // ========================================================================
    // WHERE Clause
    // ========================================================================
}
