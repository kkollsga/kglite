//! `extensions.cypher_preprocessor` — rewrite agent Cypher before execution.
//!
//! The retired Python server loaded an arbitrary Python module to rewrite
//! queries. The pure-Rust binary can't host Python, so this offers the same
//! capability in two cross-language shapes — neither of which needs a bespoke
//! FastMCP server:
//!
//! 1. **Declarative `rules:`** — ordered regex substitutions. Covers
//!    id/token normalisation (e.g. Wikidata `'Q42'` → `42`, since the engine
//!    deliberately stopped auto-coercing prefixed ids in 0.10.10).
//! 2. **`command:` hook** — pipe the query to a subprocess (stdin → stdout)
//!    for arbitrary logic in any language.
//!
//! Both are gated on `trust.allow_query_preprocessor: true` (same posture as
//! `extensions.embedder` / `trust.allow_embedder`). Applied to every
//! `cypher_query` and manifest `tools[].cypher` invocation before the query
//! reaches `graph.cypher(...)`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, Result};
use regex::Regex;

/// A compiled query-rewriting preprocessor built from the manifest.
#[derive(Debug)]
pub struct Preprocessor {
    /// Ordered regex → replacement rules (replacement supports `$1` refs).
    rules: Vec<(Regex, String)>,
    /// Optional subprocess: query on stdin, rewritten query on stdout.
    command: Option<Vec<String>>,
    /// Manifest parent dir — the command runs with this as its cwd, so
    /// `./script.py` and relative args resolve against the manifest.
    base_dir: PathBuf,
}

impl Preprocessor {
    /// Parse `extensions.cypher_preprocessor`. `Ok(None)` when absent. Errors
    /// when present without the trust gate, or on a malformed block.
    pub fn from_manifest(
        ext: Option<&serde_json::Value>,
        trust_allowed: bool,
        base_dir: &Path,
    ) -> Result<Option<Self>> {
        let Some(value) = ext else {
            return Ok(None);
        };
        if !trust_allowed {
            return Err(anyhow!(
                "extensions.cypher_preprocessor requires trust.allow_query_preprocessor: true"
            ));
        }
        let map = value
            .as_object()
            .ok_or_else(|| anyhow!("extensions.cypher_preprocessor must be a mapping"))?;

        let mut rules = Vec::new();
        if let Some(rv) = map.get("rules") {
            let arr = rv.as_array().ok_or_else(|| {
                anyhow!("cypher_preprocessor.rules must be a list of {{pattern, replace}}")
            })?;
            for (i, r) in arr.iter().enumerate() {
                let pat = r.get("pattern").and_then(|x| x.as_str()).ok_or_else(|| {
                    anyhow!("cypher_preprocessor.rules[{i}].pattern (string) is required")
                })?;
                let rep = r.get("replace").and_then(|x| x.as_str()).ok_or_else(|| {
                    anyhow!("cypher_preprocessor.rules[{i}].replace (string) is required")
                })?;
                let re = Regex::new(pat).map_err(|e| {
                    anyhow!("cypher_preprocessor.rules[{i}].pattern {pat:?} invalid: {e}")
                })?;
                rules.push((re, rep.to_string()));
            }
        }

        let command = match map.get("command") {
            None | Some(serde_json::Value::Null) => None,
            Some(c) => {
                let arr = c.as_array().ok_or_else(|| {
                    anyhow!("cypher_preprocessor.command must be a list of strings")
                })?;
                let cmd: Option<Vec<String>> =
                    arr.iter().map(|x| x.as_str().map(str::to_string)).collect();
                let cmd = cmd.ok_or_else(|| {
                    anyhow!("cypher_preprocessor.command entries must be strings")
                })?;
                if cmd.is_empty() {
                    return Err(anyhow!("cypher_preprocessor.command must be non-empty"));
                }
                Some(cmd)
            }
        };

        if rules.is_empty() && command.is_none() {
            return Err(anyhow!(
                "extensions.cypher_preprocessor needs `rules` and/or `command`"
            ));
        }
        Ok(Some(Self {
            rules,
            command,
            base_dir: base_dir.to_path_buf(),
        }))
    }

    /// Rewrite a query: apply the regex rules in order, then (if configured)
    /// pipe the result through the command. Errors propagate to the agent as
    /// a `Cypher error:` rather than silently running the original query.
    pub fn rewrite(&self, query: &str) -> Result<String, String> {
        let mut q = query.to_string();
        for (re, rep) in &self.rules {
            q = re.replace_all(&q, rep.as_str()).into_owned();
        }
        if let Some(cmd) = &self.command {
            q = self.run_command(cmd, &q)?;
        }
        Ok(q)
    }

    fn run_command(&self, cmd: &[String], query: &str) -> Result<String, String> {
        let mut child = Command::new(&cmd[0])
            .args(&cmd[1..])
            .current_dir(&self.base_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                format!(
                    "cypher_preprocessor command {:?} failed to start: {e}",
                    cmd[0]
                )
            })?;
        child
            .stdin
            .take()
            .ok_or("cypher_preprocessor: could not open command stdin")?
            .write_all(query.as_bytes())
            .map_err(|e| {
                format!("cypher_preprocessor: writing query to command stdin failed: {e}")
            })?;
        let out = child
            .wait_with_output()
            .map_err(|e| format!("cypher_preprocessor: waiting on command failed: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "cypher_preprocessor command exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn pp(json: serde_json::Value) -> Preprocessor {
        Preprocessor::from_manifest(Some(&json), true, &PathBuf::from("."))
            .unwrap()
            .unwrap()
    }

    #[test]
    fn rules_rewrite_q_numbers_to_ints() {
        let p = pp(serde_json::json!({
            "rules": [{"pattern": "'Q(\\d+)'", "replace": "$1"}]
        }));
        assert_eq!(
            p.rewrite("MATCH (n {nid: 'Q42'}) RETURN n").unwrap(),
            "MATCH (n {nid: 42}) RETURN n"
        );
    }

    #[test]
    fn rules_apply_in_order() {
        let p = pp(serde_json::json!({
            "rules": [
                {"pattern": "FOO", "replace": "BAR"},
                {"pattern": "BAR", "replace": "BAZ"}
            ]
        }));
        assert_eq!(p.rewrite("FOO").unwrap(), "BAZ");
    }

    #[test]
    fn absent_block_is_none() {
        assert!(Preprocessor::from_manifest(None, true, &PathBuf::from("."))
            .unwrap()
            .is_none());
    }

    #[test]
    fn requires_trust_gate() {
        let v = serde_json::json!({"rules": [{"pattern": "a", "replace": "b"}]});
        let err = Preprocessor::from_manifest(Some(&v), false, &PathBuf::from(".")).unwrap_err();
        assert!(err.to_string().contains("allow_query_preprocessor"));
    }

    #[test]
    fn empty_block_errors() {
        let v = serde_json::json!({});
        assert!(Preprocessor::from_manifest(Some(&v), true, &PathBuf::from(".")).is_err());
    }

    #[test]
    fn invalid_regex_errors() {
        let v = serde_json::json!({"rules": [{"pattern": "(", "replace": "x"}]});
        assert!(Preprocessor::from_manifest(Some(&v), true, &PathBuf::from(".")).is_err());
    }
}
