//! The single tokenizer shared by index build and query parsing.
//!
//! Both sides of a BM25 lookup must agree on what a term *is*; if the builder
//! and the query parser tokenize differently, a query can never match a
//! document that plainly contains its words and nothing reports an error. That
//! failure is silent, so the two sides are not allowed to have two
//! implementations: [`analyze`] is the only tokenizer in the retrieval lane.
//!
//! The character rule is the one `text_normalize()` already exposes to Cypher
//! users (`scalar_functions/string.rs`), so a caller can predict tokenization
//! from a function they can run: **`char::is_alphanumeric` is term content,
//! every other character is a separator, and content is lowercased with
//! `char::to_lowercase` (multi-char aware — `İ` lowercases to two chars).**
//! Unicode letters are content, so `Tromsø` is one token and accents survive.
//!
//! Consequences worth knowing before you rely on it:
//!
//! * There is **no CJK segmentation**. Han/Hiragana/Katakana are alphanumeric,
//!   so a run of them with no intervening separator becomes a single term.
//!   CJK retrieval needs a segmenting analyzer, which v1 does not ship.
//! * `it's` tokenizes as `it` + `s`, and `3.14` as `3` + `14`; the apostrophe
//!   and the period are separators like any other punctuation.

use std::borrow::Cow;

/// Tokenize `text` into lowercased terms. Lazy: nothing is allocated for a
/// token that is already lowercase (the overwhelmingly common case), which is
/// why the item type is [`Cow`].
pub fn analyze(text: &str) -> Tokens<'_> {
    Tokens { text, cursor: 0 }
}

/// Iterator returned by [`analyze`].
pub struct Tokens<'a> {
    text: &'a str,
    cursor: usize,
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Cow<'a, str>;

    fn next(&mut self) -> Option<Self::Item> {
        let rest = &self.text[self.cursor..];
        let lead = rest.find(char::is_alphanumeric)?;
        let start = self.cursor + lead;
        let body = &self.text[start..];
        let len = body
            .find(|c: char| !c.is_alphanumeric())
            .unwrap_or(body.len());
        self.cursor = start + len;
        Some(lowercase_token(&self.text[start..self.cursor]))
    }
}

/// Per-*character* lowercasing, deliberately not `str::to_lowercase`: the
/// latter applies the Greek final-sigma rule (`Σ` → `ς` at a word end), which
/// would make a term's normalization depend on its position in the source
/// text. `text_normalize()` lowercases per char, and the two must agree.
fn lowercase_token(raw: &str) -> Cow<'_, str> {
    if raw.chars().all(is_lowercase_identity) {
        return Cow::Borrowed(raw);
    }
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        out.extend(c.to_lowercase());
    }
    Cow::Owned(out)
}

/// Whether lowercasing this char yields exactly itself.
#[inline]
fn is_lowercase_identity(c: char) -> bool {
    if c.is_ascii() {
        return !c.is_ascii_uppercase();
    }
    let mut lowered = c.to_lowercase();
    lowered.next() == Some(c) && lowered.next().is_none()
}
