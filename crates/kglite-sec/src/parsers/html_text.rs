//! Shared HTML→text helpers for the Item-section-scanning parsers
//! (SC 13D Item 4, 10-K Item 13, 8-K Item 5.02 / 2.02). Each works
//! on stripped document text.

/// Strip HTML tags, decode the common entities, keep the document's
/// own whitespace (newlines preserved — `extract_item_text` anchors
/// on `\nSIGNATURE`).
pub fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut chars = html.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            _ if in_tag => {}
            '&' => {
                let rest = &html[i..];
                if rest.starts_with("&amp;") {
                    out.push('&');
                    for _ in 0..4 {
                        chars.next();
                    }
                } else if rest.starts_with("&nbsp;") || rest.starts_with("&#160;") {
                    out.push(' ');
                    for _ in 0..5 {
                        chars.next();
                    }
                } else if rest.starts_with("&#8217;") || rest.starts_with("&#8216;") {
                    out.push('\'');
                    for _ in 0..6 {
                        chars.next();
                    }
                } else {
                    out.push('&');
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// Body text of an SEC filing's "Item {item_num}" section —
/// case-insensitive, captured from the heading to the next "Item "
/// marker, a "\nSIGNATURE" line, or 2000 chars, whichever is first.
/// Returns `None` when the item heading is absent or its body is
/// empty.
///
/// Lifted from the SC 13D parser (which now also calls this) so the
/// 10-K Item 13 and 8-K Item 5.02 / 2.02 extractors can reuse it.
pub fn extract_item_text(text: &str, item_num: &str) -> Option<String> {
    let upper = text.to_ascii_uppercase();
    let needle = format!("ITEM {}", item_num);
    let start = upper.find(&needle)?;
    let after = &text[start + needle.len()..];
    let upper_after = after.to_ascii_uppercase();
    let end = upper_after
        .find("ITEM ")
        .or(upper_after.find("\nSIGNATURE"))
        .unwrap_or(after.len().min(2000));
    let body = &after[..end.min(after.len())];
    let trimmed = body
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_tags_and_decodes_entities() {
        let s = strip_html("<p>Foo&amp;Bar&nbsp;Baz</p>");
        assert!(s.contains("Foo&Bar Baz"));
    }

    #[test]
    fn extract_item_text_captures_section_body() {
        let text = "Item 13. Certain Relationships. The Company paid \
                    $50,000 to a related party. Item 14. Exhibits.";
        let body = extract_item_text(text, "13").expect("item 13");
        assert!(body.contains("related party"));
        assert!(!body.contains("Exhibits"));
    }

    #[test]
    fn extract_item_text_missing_returns_none() {
        assert!(extract_item_text("nothing here", "13").is_none());
    }

    #[test]
    fn extract_item_text_case_insensitive() {
        let body = extract_item_text("iTeM 5.02 departure of an officer", "5.02");
        assert!(body.is_some_and(|b| b.to_ascii_lowercase().contains("departure")));
    }
}
