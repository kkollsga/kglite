//! 8-K Item 5.02 officer/director-change parser (F13).
//!
//! Item 5.02 ("Departure of Directors or Certain Officers; Election
//! of Directors; Appointment of Certain Officers; …") is free prose.
//! This is the lowest-precision extractor in the set: gated on the
//! 8-K carrying an Item 5.02, it scans the whole (short) filing for
//! "Mr./Ms./Mrs./Dr." person mentions and emits a change only when a
//! change verb (resign / retire / appoint / elect / …) sits in the
//! window around the mention. Mentions missing either signal are
//! skipped rather than guessed.

use super::html_text::strip_html;

/// One officer- or director-change disclosed in 8-K Item 5.02.
#[derive(Debug, Clone, PartialEq)]
pub struct OfficerChange {
    pub person_name: String,
    /// "departure" | "appointment" | "election" | "resignation" |
    /// "retirement" | "compensation".
    pub change_type: String,
    pub position_title: String,
    pub effective_date: String,
    /// The disclosing sentence, truncated.
    pub reason_summary: String,
}

/// Change-verb keyword → canonical `change_type`. Order matters —
/// the more specific verbs are tested first.
const CHANGE_VERBS: &[(&str, &str)] = &[
    ("resign", "resignation"),
    ("retire", "retirement"),
    ("retirement", "retirement"),
    ("appoint", "appointment"),
    ("name", "appointment"),
    ("elect", "election"),
    ("compensat", "compensation"),
    ("step down", "departure"),
    ("stepped down", "departure"),
    ("terminat", "departure"),
    ("depart", "departure"),
    ("transition", "departure"),
];

/// Officer / director titles, longest first so "Chief Executive
/// Officer" wins over "Officer".
const TITLES: &[&str] = &[
    "Chief Executive Officer",
    "Chief Financial Officer",
    "Chief Operating Officer",
    "Chief Technology Officer",
    "Chief Accounting Officer",
    "Chief Legal Officer",
    "Chief Marketing Officer",
    "Executive Vice President",
    "Senior Vice President",
    "Executive Chairman",
    "General Counsel",
    "Vice President",
    "President",
    "Treasurer",
    "Controller",
    "Secretary",
    "Chairman",
    "Director",
];

const MONTHS: &[&str] = &[
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

const HONORIFICS: &[&str] = &["Mr.", "Ms.", "Mrs.", "Dr."];

/// Extract officer/director changes from raw 8-K HTML.
///
/// Gated on the filing carrying an Item 5.02 — but the change detail
/// is often placed in a cross-referenced Item 8.01, so once the gate
/// passes the whole (short) 8-K is scanned. It anchors on
/// "Mr./Ms./Mrs./Dr." person mentions and reads the change verb,
/// title and date from a window around each. Splitting the prose on
/// sentence boundaries would shred those honorifics, so it is avoided.
pub fn extract_officer_changes(html: &str) -> Vec<OfficerChange> {
    let text = strip_html(html);
    if !text.to_ascii_lowercase().contains("item 5.02") {
        return Vec::new();
    }
    let mut out: Vec<OfficerChange> = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();
    for hon in HONORIFICS {
        let mut from = 0;
        while let Some(rel) = text[from..].find(hon) {
            let pos = from + rel;
            from = pos + hon.len();
            let Some(person_name) = name_after(&text[pos + hon.len()..]) else {
                continue;
            };
            let window = window_around(&text, pos, 220, 320);
            let Some(change_type) = change_type_of(window) else {
                continue;
            };
            let key = (person_name.clone(), change_type.clone());
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            out.push(OfficerChange {
                person_name,
                change_type,
                position_title: title_in(window),
                effective_date: effective_date_in(window),
                reason_summary: truncate(&collapse_ws(window), 240),
            });
            if out.len() >= 25 {
                return out;
            }
        }
    }
    out
}

/// A char-boundary-safe substring of `s` spanning `before` bytes
/// before `pos` and `after` bytes after it.
fn window_around(s: &str, pos: usize, before: usize, after: usize) -> &str {
    let mut start = pos.saturating_sub(before);
    while start > 0 && !s.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = (pos + after).min(s.len());
    while end < s.len() && !s.is_char_boundary(end) {
        end += 1;
    }
    &s[start..end]
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Canonical change type from the first change verb in the sentence.
fn change_type_of(sentence: &str) -> Option<String> {
    let lc = sentence.to_ascii_lowercase();
    CHANGE_VERBS
        .iter()
        .find(|(verb, _)| lc.contains(verb))
        .map(|(_, kind)| kind.to_string())
}

/// Recover a person name from the text immediately after a honorific
/// — 1-3 leading capitalised words, with a trailing possessive "'s"
/// stripped.
fn name_after(s: &str) -> Option<String> {
    let words: Vec<&str> = s
        .split_whitespace()
        .take(3)
        .take_while(|w| starts_capitalised(w))
        .collect();
    if words.is_empty() {
        return None;
    }
    let name = words.join(" ");
    let name = name
        .strip_suffix("'s")
        .or_else(|| name.strip_suffix("\u{2019}s"))
        .unwrap_or(&name);
    let trimmed = name.trim_end_matches(|c: char| !c.is_alphanumeric());
    if trimmed.len() >= 2 {
        Some(trimmed.to_string())
    } else {
        None
    }
}

/// True if `w` starts with an uppercase letter (a name token —
/// "Smith", "A.", "O'Brien").
fn starts_capitalised(w: &str) -> bool {
    w.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

/// First officer/director title mentioned in the sentence.
fn title_in(sentence: &str) -> String {
    TITLES
        .iter()
        .find(|t| sentence.contains(*t))
        .map(|t| t.to_string())
        .unwrap_or_default()
}

/// A "Month DD, YYYY" date, preferring one after the word "effective".
fn effective_date_in(sentence: &str) -> String {
    let search_from = sentence
        .to_ascii_lowercase()
        .find("effective")
        .map(|p| p + 9)
        .unwrap_or(0);
    let hay = &sentence[search_from.min(sentence.len())..];
    if let Some(d) = month_date(hay) {
        return d;
    }
    // No "effective" anchor — fall back to any date in the sentence.
    month_date(sentence).unwrap_or_default()
}

/// Find the first "Month DD, YYYY" (day/comma optional) in `s`.
fn month_date(s: &str) -> Option<String> {
    for month in MONTHS {
        if let Some(idx) = s.find(month) {
            let tail: String = s[idx..].chars().take(20).collect();
            // Require a 4-digit year somewhere in the tail.
            if tail
                .split(|c: char| !c.is_ascii_digit())
                .any(|t| t.len() == 4 && t.parse::<u16>().is_ok_and(|y| (1990..=2099).contains(&y)))
            {
                let cut = tail
                    .char_indices()
                    .find(|&(i, c)| i > 0 && c == '.' || i >= 18)
                    .map(|(i, _)| i)
                    .unwrap_or(tail.len());
                return Some(tail[..cut].trim().to_string());
            }
        }
    }
    None
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        s.chars().take(max_chars).collect::<String>() + "…"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_resignation() {
        let html = r#"<html><body>
        <p>Item 5.02 Departure of Directors or Certain Officers.</p>
        <p>On March 5, 2024, John A. Smith notified the Company of his
        decision to resign as Chief Financial Officer, effective
        April 1, 2024. Mr. Smith's departure is not the result of any
        disagreement with the Company.</p>
        <p>Item 9.01 Financial Statements.</p>
        </body></html>"#;
        let rows = extract_officer_changes(html);
        assert!(!rows.is_empty(), "expected ≥1 change");
        let r = &rows[0];
        assert_eq!(r.person_name, "Smith");
        assert_eq!(r.change_type, "resignation");
        assert_eq!(r.position_title, "Chief Financial Officer");
        assert!(r.effective_date.contains("April"));
    }

    #[test]
    fn extracts_appointment() {
        let html = r#"<html><body><p>Item 5.02. On June 10, 2024 the Board
        appointed Ms. Jane Doe as President of the Company.</p>
        <p>Item 9.01.</p></body></html>"#;
        let rows = extract_officer_changes(html);
        let appt = rows.iter().find(|r| r.change_type == "appointment");
        assert!(appt.is_some(), "expected an appointment, got {rows:?}");
        assert_eq!(appt.unwrap().position_title, "President");
    }

    #[test]
    fn no_item_502_yields_nothing() {
        let html = "<html><body><p>Item 8.01 Other Events.</p></body></html>";
        assert!(extract_officer_changes(html).is_empty());
    }

    #[test]
    fn sentence_without_person_is_skipped() {
        let html = "<html><body><p>Item 5.02. The Board approved a new \
                    compensation plan effective immediately.</p>\
                    <p>Item 9.01.</p></body></html>";
        assert!(extract_officer_changes(html).is_empty());
    }
}
