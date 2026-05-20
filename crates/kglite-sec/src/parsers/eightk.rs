//! Parser for SEC Form 8-K (current report) Item codes from the cover
//! page HTML.
//!
//! 8-K cover pages always contain a "Check the appropriate box below
//! if the Form 8-K filing is intended to..." preamble followed by a
//! standardized "Item N.NN" enumeration of the events being reported.
//!
//! We don't try to parse the full HTML — a regex over the raw text
//! catches every well-formed Item code (`Item N.NN <title>`).
//!
//! Item codes carry standardized meaning:
//! - 1.01 — entry into a material agreement
//! - 1.02 — termination of a material agreement
//! - 2.01 — completion of acquisition/disposition of assets
//! - 5.02 — departure/election of officers/directors
//! - 5.07 — submission of matters to a vote of security holders
//! - 7.01 — Regulation FD disclosure
//! - 8.01 — other events
//! - 9.01 — financial statements / exhibits

use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EightKItem {
    /// The dotted item code, e.g. "5.02".
    pub item_code: String,
    /// Optional human-readable description following the code on the
    /// same line. Trimmed; may be empty if the regex didn't capture it.
    pub description: String,
}

/// Scan raw 8-K cover-page text/HTML for Item codes.
///
/// Returns a sorted, deduped list — the same item code may be
/// referenced multiple times in a single filing.
pub fn extract_8k_items(text: &str) -> Vec<EightKItem> {
    // Regex-light: split on "Item " markers and pull out the digits +
    // optional description. Avoids pulling a regex dep just for one
    // pattern.
    let stripped = strip_html(text);
    // SEC 8-K HTML routinely separates "Item" from its code with a
    // non-breaking-space *entity* (`Item&#160;5.07`, `Item&nbsp;1.01`).
    // `strip_html` removes tags but not entities, so normalise the
    // whitespace entities to spaces here — otherwise every
    // entity-separated item code is silently missed.
    let normalized = stripped
        .replace("&nbsp;", " ")
        .replace("&#160;", " ")
        .replace("&#xa0;", " ")
        .replace("&#xA0;", " ");
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut items: Vec<EightKItem> = Vec::new();

    for chunk in normalized.split("Item ") {
        if chunk.is_empty() {
            continue;
        }
        // First whitespace-separated token should be `N.NN` or `N.NN.`
        let first_token: String = chunk
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '.')
            .chain(std::iter::once('.'))
            .chain(
                chunk
                    .chars()
                    .skip_while(|c| !c.is_whitespace() && *c != '.')
                    .skip(1) // the '.'
                    .take_while(|c| c.is_ascii_digit()),
            )
            .collect();
        if !is_item_code(&first_token) {
            continue;
        }
        // `first_token` is exactly `N.NN` (4 ASCII bytes). The text
        // right after it begins the item's Title-Case description for
        // a real heading (`Item 5.02 Departure of...`); a mid-sentence
        // reference (`...furnished under Item 2.02 of Form 8-K`) is
        // followed by a lowercase word. Require an uppercase letter so
        // only reported items count, not back-references.
        let after =
            chunk[first_token.len()..].trim_start_matches(|c: char| c.is_whitespace() || c == '.');
        if !after.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            continue;
        }
        if !seen.insert(first_token.clone()) {
            continue;
        }
        let description = after
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .trim_end_matches('.')
            .trim()
            .to_string();
        items.push(EightKItem {
            item_code: first_token,
            description,
        });
    }
    items.sort();
    items
}

fn is_item_code(s: &str) -> bool {
    // Format: N.NN where N is 1-9 and NN is 01-99.
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 2 {
        return false;
    }
    if parts[0].len() != 1 || !parts[0].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    if parts[1].len() != 2 || !parts[1].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    true
}

/// Minimal HTML-tag stripper. We don't need a full HTML parser since
/// Item codes only appear in body text; tags around them are
/// interchangeable noise.
fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_HTML: &str = r#"<html><body>
<p>This Current Report on Form 8-K is being filed to report:</p>
<p><b>Item 1.01</b> Entry into a Material Definitive Agreement.</p>
<p>On July 30, 2024, ABC Corp entered into ...</p>
<p><b>Item 5.02</b> Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers; Compensatory Arrangements of Certain Officers.</p>
<p>Item 9.01 Financial Statements and Exhibits.</p>
<p>(d) Exhibits</p>
</body></html>"#;

    #[test]
    fn extracts_three_items_from_sample_html() {
        let items = extract_8k_items(SAMPLE_HTML);
        let codes: Vec<&str> = items.iter().map(|i| i.item_code.as_str()).collect();
        assert_eq!(codes, vec!["1.01", "5.02", "9.01"]);
    }

    #[test]
    fn deduplicates_repeated_codes() {
        let s = "Item 5.02 Departure of Officers. Item 5.02 Election of Directors.";
        let items = extract_8k_items(s);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].item_code, "5.02");
    }

    #[test]
    fn decodes_nbsp_entity_separated_codes() {
        // SEC HTML often writes `Item&#160;5.07` / `Item&nbsp;1.01`.
        let s = "Item&#160;5.07 Submission of Matters. Item&nbsp;1.01 Entry into Agreement.";
        let items = extract_8k_items(s);
        let codes: Vec<&str> = items.iter().map(|i| i.item_code.as_str()).collect();
        assert_eq!(codes, vec!["1.01", "5.07"]);
    }

    #[test]
    fn skips_mid_sentence_item_references() {
        // Only 8.01 is a reported heading; the 5.02 is a back-reference
        // ("Item 5.02 of our prior report") — lowercase-followed.
        let s = "Item 8.01 Other Events. As described under Item 5.02 of our prior report.";
        let items = extract_8k_items(s);
        let codes: Vec<&str> = items.iter().map(|i| i.item_code.as_str()).collect();
        assert_eq!(codes, vec!["8.01"]);
    }

    #[test]
    fn ignores_invalid_item_shapes() {
        let s = "Item 1 (bare) and Item 5.0 (short) and Item 5.020 (long).";
        let items = extract_8k_items(s);
        assert_eq!(items.len(), 0);
    }

    #[test]
    fn empty_input_yields_no_items() {
        assert_eq!(extract_8k_items("").len(), 0);
        assert_eq!(extract_8k_items("<html></html>").len(), 0);
    }
}
