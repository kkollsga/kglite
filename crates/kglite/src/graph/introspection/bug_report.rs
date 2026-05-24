// src/graph/bug_report.rs
//
// Bug report generation — sanitizes user input and writes structured
// markdown reports to a `reported_bugs.md` file.

use chrono::Utc;
use regex::Regex;
use std::fs;
use std::sync::LazyLock;

const BUG_REPORT_FILE: &str = "reported_bugs.md";
const MAX_FIELD_LEN: usize = 10_000;

static HTML_TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"</?[a-zA-Z!][^>]*>").unwrap());

// ── Sanitization ────────────────────────────────────────────────────────────

/// Sanitize user input for safe markdown inclusion.
///
/// - Truncates to [`MAX_FIELD_LEN`] characters.
/// - Strips HTML tags (`<script>`, `<img onerror=...>`, etc.).
/// - Removes `javascript:` protocol strings.
/// - Escapes triple backticks (prevents code-block breakout).
/// - Strips null bytes and non-printable control characters (keeps `\n`, `\r`, `\t`).
fn sanitize(input: &str) -> String {
    // Truncate at a char boundary.
    let truncated = if input.len() > MAX_FIELD_LEN {
        match input.char_indices().nth(MAX_FIELD_LEN) {
            Some((idx, _)) => &input[..idx],
            None => input,
        }
    } else {
        input
    };

    // Strip HTML tags.
    let no_html = HTML_TAG_RE.replace_all(truncated, "");

    let mut result = String::with_capacity(no_html.len());
    for ch in no_html.chars() {
        // Drop control chars except whitespace.
        if ch.is_control() && ch != '\n' && ch != '\r' && ch != '\t' {
            continue;
        }
        result.push(ch);
    }

    // Escape triple backticks → prevent breaking out of fenced code blocks.
    result = result.replace("```", r"\`\`\`");

    // Strip javascript: protocol (case-insensitive).
    result = result.replace("javascript:", "");
    result = result.replace("JAVASCRIPT:", "");
    result = result.replace("Javascript:", "");

    result
}

// ── Formatting ──────────────────────────────────────────────────────────────

/// Format a single bug report entry as markdown.
fn format_report(query: &str, result: &str, expected: &str, description: &str) -> String {
    let now = Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let version = env!("CARGO_PKG_VERSION");

    let query = sanitize(query);
    let result = sanitize(result);
    let expected = sanitize(expected);
    let description = sanitize(description);

    format!(
        "\
---

### Bug Report — {now} | KGLite v{version}

**Query:**
```cypher
{query}
```

**Result:**
```
{result}
```

**Expected:**
```
{expected}
```

**Description:**
{description}

"
    )
}

// ── File I/O ────────────────────────────────────────────────────────────────

/// Write a bug report to `reported_bugs.md`, prepending new entries to the top.
///
/// Creates the file (with a header) if it doesn't exist. Returns a confirmation
/// message on success.
pub fn write_bug_report(
    query: &str,
    result: &str,
    expected: &str,
    description: &str,
    path: Option<&str>,
) -> Result<String, String> {
    let file_path = path.unwrap_or(BUG_REPORT_FILE);
    let report = format_report(query, result, expected, description);

    let existing = fs::read_to_string(file_path).unwrap_or_default();

    let new_content = if existing.is_empty() {
        // New file — add header + report.
        format!("# KGLite Bug Reports\n\n{report}")
    } else if let Some(pos) = existing.find("\n\n") {
        // Existing file — insert after the `# KGLite Bug Reports` header line.
        let header = &existing[..pos];
        let rest = &existing[pos + 2..];
        format!("{header}\n\n{report}{rest}")
    } else {
        // Malformed file — just prepend.
        format!("# KGLite Bug Reports\n\n{report}{existing}")
    };

    fs::write(file_path, new_content).map_err(|e| format!("Failed to write bug report: {e}"))?;

    Ok(format!("Bug report saved to {file_path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_html() {
        let input = "hello <script>alert('xss')</script> world";
        assert_eq!(sanitize(input), "hello alert('xss') world");
    }

    #[test]
    fn sanitize_escapes_triple_backticks() {
        let input = "break out ``` of code";
        assert!(sanitize(input).contains(r"\`\`\`"));
    }

    #[test]
    fn sanitize_strips_javascript_protocol() {
        let input = "click [here](javascript:alert(1))";
        assert!(!sanitize(input).contains("javascript:"));
    }

    #[test]
    fn sanitize_strips_control_chars() {
        let input = "hello\x00\x01\x02world\nnewline";
        let result = sanitize(input);
        assert_eq!(result, "helloworld\nnewline");
    }

    #[test]
    fn sanitize_preserves_normal_text() {
        let input = "MATCH (n:Field) WHERE n.name = 'test' RETURN n";
        assert_eq!(sanitize(input), input);
    }

    #[test]
    fn format_report_has_required_sections() {
        let report = format_report("MATCH (n) RETURN n", "got 5", "got 10", "wrong count");
        assert!(report.contains("### Bug Report"));
        assert!(report.contains("KGLite v"));
        assert!(report.contains("**Query:**"));
        assert!(report.contains("**Result:**"));
        assert!(report.contains("**Expected:**"));
        assert!(report.contains("**Description:**"));
        assert!(report.starts_with("---"));
    }

    #[test]
    fn write_creates_new_file() {
        let dir = std::env::temp_dir().join("kglite_test_bug_report");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_bugs.md");
        let _ = fs::remove_file(&path);

        let result = write_bug_report(
            "MATCH (n) RETURN n",
            "empty",
            "5 rows",
            "no results",
            Some(path.to_str().unwrap()),
        );
        assert!(result.is_ok());

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("# KGLite Bug Reports"));
        assert!(content.contains("### Bug Report"));
        assert!(content.contains("no results"));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn write_prepends_to_existing() {
        let dir = std::env::temp_dir().join("kglite_test_bug_report");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_prepend.md");

        // Write first report.
        write_bug_report("q1", "r1", "e1", "first", Some(path.to_str().unwrap())).unwrap();
        // Write second report.
        write_bug_report("q2", "r2", "e2", "second", Some(path.to_str().unwrap())).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        // "second" should appear before "first".
        let pos_second = content.find("second").unwrap();
        let pos_first = content.find("first").unwrap();
        assert!(pos_second < pos_first, "new report should be prepended");

        let _ = fs::remove_file(&path);
    }
}
