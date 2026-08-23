// Bug report generation — sanitizes user input and writes structured
// markdown reports to a `reported_bugs.md` file.

use chrono::Utc;
use regex::Regex;
use std::fs;
use std::sync::LazyLock;

const BUG_REPORT_FILE: &str = "reported_bugs.md";
const MAX_FIELD_LEN: usize = 10_000;

static HTML_TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"</?[a-zA-Z!][^>]*>").unwrap());

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

    let no_html = HTML_TAG_RE.replace_all(truncated, "");

    let mut result = String::with_capacity(no_html.len());
    for ch in no_html.chars() {
        if ch.is_control() && ch != '\n' && ch != '\r' && ch != '\t' {
            continue;
        }
        result.push(ch);
    }

    result = result.replace("```", r"\`\`\`");

    result = strip_ascii_ci(&result, "javascript:");

    result
}

/// Remove every ASCII-case variant of `needle` from `haystack`.
///
/// Three literal `replace` calls covered `javascript:`, `JAVASCRIPT:` and
/// `Javascript:` and let `JavaScript:` — the spelling in most copy-pasted
/// markup — straight through.
fn strip_ascii_ci(haystack: &str, needle: &str) -> String {
    let needle_lower = needle.to_ascii_lowercase();
    let mut out = String::with_capacity(haystack.len());
    let mut rest = haystack;
    loop {
        // `to_ascii_lowercase` is byte-length preserving, so an index found in
        // the lowered copy is valid in `rest`.
        match rest.to_ascii_lowercase().find(&needle_lower) {
            Some(at) => {
                out.push_str(&rest[..at]);
                rest = &rest[at + needle.len()..];
            }
            None => {
                out.push_str(rest);
                return out;
            }
        }
    }
}

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

/// Write a bug report to `reported_bugs.md`, prepending new entries to the top.
/// Creates the file with a header if it doesn't exist; `Ok` carries a
/// confirmation message.
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
        // Case-insensitively, and asserted case-insensitively: the previous
        // `!contains("javascript:")` check passed for any input this function
        // failed to strip, so it could not fail for `JavaScript:`.
        for input in [
            "click [here](javascript:alert(1))",
            "click [here](JavaScript:alert(1))",
            "click [here](JAVASCRIPT:alert(1))",
            "click [here](jAvAsCrIpT:alert(1))",
        ] {
            let out = sanitize(input);
            assert!(
                !out.to_ascii_lowercase().contains("javascript:"),
                "protocol survived sanitization: {input} -> {out}"
            );
        }
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

        write_bug_report("q1", "r1", "e1", "first", Some(path.to_str().unwrap())).unwrap();
        write_bug_report("q2", "r2", "e2", "second", Some(path.to_str().unwrap())).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let pos_second = content.find("second").unwrap();
        let pos_first = content.find("first").unwrap();
        assert!(pos_second < pos_first, "new report should be prepended");

        let _ = fs::remove_file(&path);
    }
}
