//! The oracle for everything here is the per-record answer computed with the
//! `regex` crate's `str` API on each record alone, which shares no code with
//! the scan's byte-level paths.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::layout::{BorrowedLayout, LayoutError, OwnedLayout, RecordLayout};
use super::matcher::{required_literal, Matcher, Pattern};
use super::scan::{self, Hit, ScanError, ScanOptions};
use super::scope::{self, ScopeStrategy};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

type Rows = Vec<Option<String>>;

fn layout(rows: &Rows) -> OwnedLayout {
    OwnedLayout::from_records(rows)
}

fn lit(text: &str, ci: bool) -> Pattern {
    Pattern::Literal {
        text: text.into(),
        case_insensitive: ci,
    }
}
fn rx(pattern: &str, ci: bool) -> Pattern {
    Pattern::Regex {
        pattern: pattern.into(),
        case_insensitive: ci,
    }
}
fn phrase(text: &str, ci: bool) -> Pattern {
    Pattern::Phrase {
        text: text.into(),
        case_insensitive: ci,
    }
}

/// Reference: `(record, offset, len)` per present record, `str` regex.
fn naive(rows: &Rows, pattern: &Pattern) -> Vec<(usize, u64, u64)> {
    let build = |p: &str, ci: bool| {
        regex::RegexBuilder::new(p)
            .case_insensitive(ci)
            .build()
            .unwrap()
    };
    let (re, literal) = match pattern {
        Pattern::Literal {
            text,
            case_insensitive: false,
        } => (None, Some(text.clone())),
        Pattern::Literal {
            text,
            case_insensitive,
        } => (Some(build(&regex::escape(text), *case_insensitive)), None),
        Pattern::Phrase {
            text,
            case_insensitive,
        } => {
            let words: Vec<String> = text.split_whitespace().map(regex::escape).collect();
            if words.len() == 1 && !*case_insensitive {
                (None, Some(words[0].clone().replace('\\', "")))
            } else {
                (Some(build(&words.join("\\s+"), *case_insensitive)), None)
            }
        }
        Pattern::Regex {
            pattern,
            case_insensitive,
        } => (Some(build(pattern, *case_insensitive)), None),
    };
    let mut out = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let Some(text) = row else { continue };
        let spans: Vec<(usize, usize)> = match (&re, &literal) {
            (_, Some(l)) => text
                .match_indices(l.as_str())
                .map(|(s, m)| (s, s + m.len()))
                .collect(),
            (Some(re), _) => re
                .find_iter(text)
                .filter(|m| !m.is_empty())
                .map(|m| (m.start(), m.end()))
                .collect(),
            _ => unreachable!(),
        };
        out.extend(
            spans
                .into_iter()
                .map(|(s, e)| (i, s as u64, (e - s) as u64)),
        );
    }
    out
}

fn run(layout: &OwnedLayout, pattern: &Pattern, opts: &ScanOptions<'_>) -> Vec<(usize, u64, u64)> {
    let matcher = Matcher::compile(pattern).unwrap();
    let result = scan::scan(layout, &matcher, opts).unwrap();
    result
        .hits
        .iter()
        .map(|h| (h.record, h.offset, h.len))
        .collect()
}

const ALPHABET: [&str; 10] = ["a", "b", "a", "\n", " ", "é", "ß", "S", "x", "\t"];

/// Small alphabet, short values, nulls and empties: needles keep landing
/// across value boundaries.
fn straddle_corpus(seed: u64, records: usize) -> Rows {
    let mut rng = Rng(seed | 1);
    (0..records)
        .map(|_| match rng.below(10) {
            0 => None,
            1 => Some(String::new()),
            _ => Some(
                (0..1 + rng.below(5))
                    .map(|_| ALPHABET[rng.below(ALPHABET.len())])
                    .collect(),
            ),
        })
        .collect()
}

fn patterns() -> Vec<Pattern> {
    vec![
        lit("a", false),
        lit("aa", false),
        lit("ab", false),
        lit("ba", false),
        lit("a b", false),
        lit("\n", false),
        lit("é", false),
        lit("ßa", false),
        lit("aab", false),
        lit("AA", true),
        lit("ss", true),
        lit("SS", true),
        lit("ß", true),
        lit("é", true),
        phrase("a b", false),
        phrase("a   b", true),
        phrase("aa", false),
        rx("a+", false),
        rx("^a", false),
        rx("b$", false),
        rx("\\bab", false),
        rx("a\\b", false),
        rx("a.*b", false),
        rx("(?s)a.+b", false),
        rx("[^a]b", false),
        rx("a\\s+b", false),
        rx("aba|b", false),
        rx("aab+", false),
        rx("ab[ab]", false),
        rx("\\Aa", false),
        rx("a\\z", false),
        rx("(?m)^b", false),
        rx("(?i)SS", false),
        rx("é|ß", true),
        rx("x{1,3}a", false),
    ]
}

// ── layout ──────────────────────────────────────────────────────────────

#[test]
fn the_checked_constructor_refuses_each_inconsistency() {
    let bytes = b"abcdef";
    assert!(BorrowedLayout::new(&[0, 3, 6], bytes, &[0, 0]).is_ok());
    assert!(
        BorrowedLayout::new(&[0], b"", &[]).is_ok(),
        "an empty layout has one offset"
    );
    assert_eq!(
        BorrowedLayout::new(&[], bytes, &[]).unwrap_err(),
        LayoutError::OffsetsLength
    );
    assert_eq!(
        BorrowedLayout::new(&[0, 3], bytes, &[0, 0]).unwrap_err(),
        LayoutError::OffsetsLength
    );
    assert_eq!(
        BorrowedLayout::new(&[1, 3, 6], bytes, &[0, 0]).unwrap_err(),
        LayoutError::FirstOffset
    );
    assert_eq!(
        BorrowedLayout::new(&[0, 4, 3], bytes, &[0, 0]).unwrap_err(),
        LayoutError::NotMonotone
    );
    assert_eq!(
        BorrowedLayout::new(&[0, 3, 7], bytes, &[0, 0]).unwrap_err(),
        LayoutError::PastData
    );
}

#[test]
fn ranges_nulls_and_record_lookup_follow_the_column_convention() {
    // "ab", null, "", "cde": offsets [0,2,2,2,5], nulls [0,1,0,0].
    let rows = vec![
        Some("ab".to_string()),
        None,
        Some(String::new()),
        Some("cde".to_string()),
    ];
    let l = layout(&rows);
    assert_eq!(l.offsets(), [0, 2, 2, 2, 5]);
    assert_eq!(l.nulls(), [0, 1, 0, 0]);
    assert_eq!(l.len(), 4);
    assert_eq!(l.range(1), 2..2);
    assert!(l.is_present(2) && !l.is_present(1) && !l.is_present(9));
    for (offset, want) in [
        (0, Some(0)),
        (1, Some(0)),
        (2, Some(3)),
        (4, Some(3)),
        (5, None),
        (99, None),
    ] {
        assert_eq!(l.record_of(offset, 0, l.len()), want, "offset {offset}");
    }
    assert_eq!(l.record_of(1, 1, 4), None, "below the window");
    let view = BorrowedLayout::from_validated(l.offsets(), l.bytes(), l.nulls());
    assert_eq!(view.record(3), b"cde");
}

// ── the straddle rule against the per-record oracle ─────────────────────

#[test]
fn every_path_reproduces_the_per_record_answer_on_straddle_corpora() {
    for seed in 1..=25u64 {
        let rows = straddle_corpus(seed, 300);
        let l = layout(&rows);
        for pattern in patterns() {
            let want = naive(&rows, &pattern);
            let default = run(&l, &pattern, &ScanOptions::default());
            assert_eq!(default, want, "seed {seed} {pattern:?}: default path");
            let oracle = ScanOptions {
                oracle_only: true,
                ..Default::default()
            };
            assert_eq!(
                run(&l, &pattern, &oracle),
                want,
                "seed {seed} {pattern:?}: oracle path"
            );
        }
    }
}

#[test]
fn needles_built_from_value_edges_never_hit_across_them() {
    let cases: [(&[&str], &str, usize); 6] = [
        (&["a", "a", "aa"], "aa", 1),
        (&["xa", "ax"], "aa", 0),
        (&["ab", "ab"], "bab", 0),
        (&["a", "aa", "a"], "aa", 1),
        (&["aaa", "a"], "aa", 1),
        (&["", "a", "", "a"], "aa", 0),
    ];
    for (records, needle, hits) in cases {
        let rows: Rows = records.iter().map(|r| Some(r.to_string())).collect();
        let l = layout(&rows);
        for pattern in [lit(needle, false), lit(needle, true), rx(needle, false)] {
            let got = run(&l, &pattern, &ScanOptions::default());
            assert_eq!(got.len(), hits, "{records:?} {needle} {pattern:?}");
            assert_eq!(got, naive(&rows, &pattern));
        }
    }
}

#[test]
fn needle_bytes_split_across_two_values_do_not_match() {
    // "é" is C3 A9; one record ends with C3 and the next starts with A9.
    let l = OwnedLayout::from_records(&[
        Some(&b"x\xC3"[..]),
        Some(&b"\xA9y"[..]),
        Some("é".as_bytes()),
    ]);
    let matcher = Matcher::compile(&lit("é", false)).unwrap();
    let hits = scan::scan(&l, &matcher, &ScanOptions::default())
        .unwrap()
        .hits;
    assert_eq!(
        hits,
        [Hit {
            record: 2,
            offset: 0,
            len: 2
        }]
    );
}

#[test]
fn a_preferred_straddling_regex_match_does_not_hide_an_in_record_one() {
    // Whole-slice `b\s*b?` at record 0's end would prefer "b" + "\nb" across the
    // boundary; the isolated record still has its own match.
    let rows: Rows = vec![Some("ab\n".into()), Some("b ab".into())];
    let l = layout(&rows);
    let pattern = rx("b\\s*b?", false);
    assert_eq!(
        run(&l, &pattern, &ScanOptions::default()),
        naive(&rows, &pattern)
    );
    // Anchors mean the record's own edges, not the neighbours'.
    let rows: Rows = vec![Some("foo".into()), Some("foo".into()), Some("xfoo".into())];
    let l = layout(&rows);
    for p in ["^foo", "foo$", "\\bfoo", "\\Afoo\\z"] {
        let pattern = rx(p, false);
        assert_eq!(
            run(&l, &pattern, &ScanOptions::default()),
            naive(&rows, &pattern),
            "{p}"
        );
    }
}

#[test]
fn case_folding_is_the_regex_crates_per_record_folding() {
    let values = [
        "Straße",
        "STRASSE",
        "strasse",
        "ẞ",
        "ß",
        "İstanbul",
        "istanbul",
        "ISTANBUL",
        "ıi",
        "I",
        "K",
        "\u{212A}",
        "ÆØÅ",
        "æøå",
    ];
    let rows: Rows = values.iter().map(|v| Some(v.to_string())).collect();
    let l = layout(&rows);
    for needle in [
        "ß", "ss", "SS", "i", "I", "ı", "k", "æøå", "ÆØÅ", "straße", "İ",
    ] {
        for pattern in [
            lit(needle, true),
            rx(&regex::escape(needle), true),
            phrase(needle, true),
        ] {
            assert_eq!(
                run(&l, &pattern, &ScanOptions::default()),
                naive(&rows, &pattern),
                "{needle} {pattern:?}"
            );
        }
    }
    let ss = run(&l, &lit("SS", true), &ScanOptions::default());
    assert!(
        ss.iter().all(|h| h.0 != 0 && h.0 != 3 && h.0 != 4),
        "simple folding: ß is not SS: {ss:?}"
    );
    let turkish = run(&l, &lit("i", true), &ScanOptions::default());
    assert!(
        turkish.iter().all(|h| h.0 != 8 || h.2 == 1),
        "dotless ı is not i"
    );
}

#[test]
fn regexes_that_would_match_nothing_useful_are_refused() {
    assert!(Matcher::compile(&rx("a*", false)).is_err());
    assert!(Matcher::compile(&rx("(", false)).is_err());
    assert!(Matcher::compile(&lit("", false)).is_err());
    assert!(Matcher::compile(&phrase("  ", false)).is_err());
}

// ── the regex prefilter ─────────────────────────────────────────────────

#[test]
fn the_required_literal_is_extracted_only_when_it_is_provably_required() {
    let cases: [(&str, bool, Option<&str>); 27] = [
        ("foobar", false, Some("foobar")),
        ("^foo\\d+", false, Some("foo")),
        ("\\bword\\b", false, Some("word")),
        ("a+bcd", false, Some("bcd")),
        ("fooo+", false, Some("fooo")),
        ("fooo?bar", false, Some("foo")),
        ("foo*bar", false, Some("bar")),
        ("ab{1000}cde", false, Some("cde")),
        ("x[a-z]+yzw", false, Some("yzw")),
        ("(abc|def)ghi", false, Some("ghi")),
        ("foo|bar", false, None),
        ("foo(bar|baz)?quux", false, Some("quux")),
        ("(?i)foobar", false, None),
        ("(?s)foo.bar", false, Some("foo")),
        ("(?s).*word.*", false, Some("word")),
        ("(?s).*\\bword\\b.*", false, Some("word")),
        ("(?sm)^abc$", false, Some("abc")),
        ("abc(?U)d*efg", false, Some("abc")),
        ("(?si)foobar", false, None),
        ("(?x)foo bar", false, None),
        ("(?s:foobar)", false, None),
        ("(?s", false, None),
        ("(?)foobar", false, None),
        ("foobar", true, None),
        ("ab", false, None),
        ("\\.\\.\\.", false, None),
        ("[abc]+", false, None),
    ];
    for (pattern, ci, want) in cases {
        assert_eq!(
            required_literal(pattern, ci).as_deref(),
            want,
            "{pattern:?} ci={ci}"
        );
    }
    assert_eq!(
        required_literal("(?:foo)bar", false).as_deref(),
        Some("bar"),
        "plain non-capturing groups are skipped"
    );
}

#[test]
fn a_prefilter_never_changes_the_answer() {
    let rows = straddle_corpus(77, 400);
    let l = layout(&rows);
    for p in [
        "aab+", "ab[ab]", "a ?b", "ab|ba", "x{1,3}a", "aab\\s*a", "\\bab", "b a\\n", "aa.b", "ab+",
    ] {
        let pattern = rx(p, false);
        let matcher = Matcher::compile(&pattern).unwrap();
        let with = scan::scan(&l, &matcher, &ScanOptions::default())
            .unwrap()
            .hits;
        let oracle = scan::scan(
            &l,
            &matcher,
            &ScanOptions {
                oracle_only: true,
                ..Default::default()
            },
        )
        .unwrap()
        .hits;
        assert_eq!(with, oracle, "{p}");
    }
}

// ── controls ────────────────────────────────────────────────────────────

fn big_layout() -> OwnedLayout {
    let rows: Rows = (0..60_000usize)
        .map(|i| {
            Some(format!(
                "record {i} {} fox jumps the quick Ærlig end",
                "padding ".repeat(40 + i % 50)
            ))
        })
        .collect();
    layout(&rows)
}

#[test]
fn parallel_chunked_scans_equal_serial_ones() {
    let l = big_layout();
    assert!(
        l.bytes().len() > 2 * scan::UNIT_BYTES,
        "needs several units"
    );
    for pattern in [
        lit("jumps", false),
        lit("ærlig", true),
        rx("fox\\s+jumps", false),
        rx("[0-9]+ pad", false),
    ] {
        let serial = ScanOptions {
            parallel_min_bytes: Some(u64::MAX),
            ..Default::default()
        };
        let forced = ScanOptions {
            parallel_min_bytes: Some(0),
            ..Default::default()
        };
        let (a, b) = (run(&l, &pattern, &serial), run(&l, &pattern, &forced));
        assert_eq!(a.len(), 60_000);
        assert_eq!(a, b, "{pattern:?}");
    }
}

#[test]
fn the_hit_cap_keeps_the_first_hits_by_position_and_count_only_counts() {
    let l = big_layout();
    let matcher = Matcher::compile(&lit("fox", false)).unwrap();
    let all = scan::scan(&l, &matcher, &ScanOptions::default()).unwrap();
    assert_eq!(
        (all.hits.len(), all.count, all.truncated),
        (60_000, 60_000, false)
    );
    for parallel in [u64::MAX, 0] {
        let opts = ScanOptions {
            max_hits: Some(1234),
            parallel_min_bytes: Some(parallel),
            ..Default::default()
        };
        let capped = scan::scan(&l, &matcher, &opts).unwrap();
        assert!(capped.truncated);
        assert_eq!(capped.hits, all.hits[..1234]);
    }
    let counted = scan::scan(
        &l,
        &matcher,
        &ScanOptions {
            count_only: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(counted.hits.is_empty());
    assert_eq!(counted.count, 60_000);
    let both = scan::scan(
        &l,
        &matcher,
        &ScanOptions {
            count_only: true,
            max_hits: Some(10),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!((both.count, both.truncated), (10, true));
}

#[test]
fn the_interrupt_hook_stops_a_scan_mid_flight_and_the_filter_drops_records() {
    let rows: Rows = (0..3_000).map(|i| Some(format!("row {i} foo"))).collect();
    let l = layout(&rows);
    for pattern in [lit("foo", false), rx("fo+", false), rx("^row", false)] {
        let matcher = Matcher::compile(&pattern).unwrap();
        let polls = AtomicUsize::new(0);
        let trip = || polls.fetch_add(1, Ordering::Relaxed) >= 2;
        let opts = ScanOptions {
            interrupt: Some(&trip),
            ..Default::default()
        };
        assert_eq!(
            scan::scan(&l, &matcher, &opts).unwrap_err(),
            ScanError::Interrupted,
            "{pattern:?}"
        );
        assert!(
            polls.load(Ordering::Relaxed) >= 3,
            "polled inside the unit: {pattern:?}"
        );
        let never = || false;
        let opts = ScanOptions {
            interrupt: Some(&never),
            ..Default::default()
        };
        assert_eq!(scan::scan(&l, &matcher, &opts).unwrap().count, 3_000);
        let keep = |record: usize| record.is_multiple_of(2);
        let opts = ScanOptions {
            record_filter: Some(&keep),
            ..Default::default()
        };
        assert_eq!(scan::scan(&l, &matcher, &opts).unwrap().count, 1_500);
    }
}

#[test]
fn first_per_record_reports_each_records_first_hit_only() {
    let rows = straddle_corpus(13, 300);
    let l = layout(&rows);
    for pattern in patterns() {
        let mut want: Vec<(usize, u64, u64)> = Vec::new();
        for hit in naive(&rows, &pattern) {
            if want.last().is_none_or(|last| last.0 != hit.0) {
                want.push(hit);
            }
        }
        for oracle_only in [false, true] {
            let opts = ScanOptions {
                first_per_record: true,
                oracle_only,
                ..Default::default()
            };
            assert_eq!(
                run(&l, &pattern, &opts),
                want,
                "{pattern:?} oracle_only={oracle_only}"
            );
        }
    }
}

#[test]
fn the_overlay_matcher_is_the_oracle_and_agrees_with_the_scan() {
    let rows = straddle_corpus(5, 200);
    let l = layout(&rows);
    for pattern in patterns() {
        let matcher = Matcher::compile(&pattern).unwrap();
        let scanned = scan::scan(&l, &matcher, &ScanOptions::default())
            .unwrap()
            .hits;
        let mut overlay = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            if let Some(text) = row {
                overlay.extend(matcher.overlay_hits(text).into_iter().map(|(o, n)| Hit {
                    record: i,
                    offset: o as u64,
                    len: n as u64,
                }));
            }
        }
        assert_eq!(scanned, overlay, "{pattern:?}");
    }
}

#[test]
fn snippets_and_character_offsets_respect_utf8() {
    let rows: Rows = vec![Some("line one\nTromsø er ærlig og blå".into())];
    let l = layout(&rows);
    let hit = scan::scan(
        &l,
        &Matcher::compile(&lit("rlig", false)).unwrap(),
        &ScanOptions::default(),
    )
    .unwrap()
    .hits[0];
    assert_eq!(
        scan::snippet(&l, &hit, 1),
        "ærlig ",
        "a one-byte context lands inside æ and is widened"
    );
    assert_eq!(
        scan::char_offset(&l, &hit),
        "line one\nTromsø er æ".chars().count()
    );
}

#[test]
fn token_count_follows_the_analyzer_rule() {
    use crate::graph::algorithms::text_index::analyzer::analyze;
    for row in straddle_corpus(3, 300).into_iter().flatten() {
        assert_eq!(
            scope::token_count(row.as_bytes()) as usize,
            analyze(&row).count(),
            "{row:?}"
        );
    }
    assert_eq!(
        scope::token_count("Tromsø – ærlig 日本語 3.14".as_bytes()),
        5
    );
}

// ── scopes ──────────────────────────────────────────────────────────────

fn scoped(
    l: &OwnedLayout,
    pattern: &Pattern,
    scope: &[u32],
    strategy: ScopeStrategy,
) -> Vec<(usize, u64, u64)> {
    run(
        l,
        pattern,
        &ScanOptions {
            scope: Some(scope),
            scope_strategy: Some(strategy),
            ..Default::default()
        },
    )
}

#[test]
fn both_scope_strategies_return_the_reference_hits_for_every_scope_shape() {
    let rows = straddle_corpus(21, 400);
    let l = layout(&rows);
    let mut rng = Rng(5);
    let scattered: Vec<u32> = (0..60).map(|_| rng.below(400) as u32).collect();
    let scopes: Vec<(&str, Vec<u32>)> = vec![
        ("empty", vec![]),
        ("single", vec![17]),
        ("run", (100..180).collect()),
        ("scattered unsorted with duplicates", scattered),
        ("everything", (0..400).collect()),
        ("out of range", vec![400, 5000, u32::MAX]),
        ("mixed", vec![399, 0, 1, 1, 7, 4000, 6, 5]),
        (
            "nulls and empties only",
            rows.iter()
                .enumerate()
                .filter(|(_, r)| r.as_ref().is_none_or(|s| s.is_empty()))
                .map(|(i, _)| i as u32)
                .collect(),
        ),
    ];
    for (name, scope) in &scopes {
        let members: std::collections::BTreeSet<usize> =
            scope.iter().map(|i| *i as usize).collect();
        for pattern in patterns() {
            let want: Vec<_> = naive(&rows, &pattern)
                .into_iter()
                .filter(|h| members.contains(&h.0))
                .collect();
            for strategy in [ScopeStrategy::RangeDriven, ScopeStrategy::ScanDriven] {
                assert_eq!(
                    scoped(&l, &pattern, scope, strategy),
                    want,
                    "{name} {pattern:?} {strategy:?}"
                );
            }
            let auto = run(
                &l,
                &pattern,
                &ScanOptions {
                    scope: Some(scope),
                    ..Default::default()
                },
            );
            assert_eq!(auto, want, "{name} {pattern:?} automatic");
        }
    }
}

#[test]
fn a_contiguous_range_keeps_its_edges_and_never_joins_adjacent_records() {
    let rows: Rows = ["zzz foo", "alpha foo", "bar beta", "gamma foo", "bar zzz"]
        .iter()
        .map(|s| Some(s.to_string()))
        .collect();
    let l = layout(&rows);
    for strategy in [ScopeStrategy::RangeDriven, ScopeStrategy::ScanDriven] {
        assert_eq!(
            scoped(&l, &lit("foo", false), &[1, 2, 3], strategy),
            [(1, 6, 3), (3, 6, 3)],
            "last byte of a record at the range edge"
        );
        assert_eq!(
            scoped(&l, &lit("alpha", false), &[1, 2, 3], strategy),
            [(1, 0, 5)]
        );
        for pattern in [
            lit("foobar", false),
            lit("foo bar", false),
            rx("foo.?bar", false),
            rx("foo\\s*bar", false),
            phrase("foo bar", false),
        ] {
            for scope in [&[1u32, 2, 3][..], &[0, 1, 2, 3, 4][..]] {
                assert!(
                    scoped(&l, &pattern, scope, strategy).is_empty(),
                    "{pattern:?} {strategy:?} {scope:?}"
                );
            }
        }
    }
}

#[test]
fn scope_statistics_match_brute_force_without_reading_the_data() {
    let rows = straddle_corpus(8, 300);
    let l = layout(&rows);
    let scope: Vec<u32> = (0..300).step_by(3).chain([9999]).collect();
    let stats = scope::scope_stats(&l, Some(&scope));
    let (mut records, mut bytes) = (0u64, 0u64);
    for i in (0..300).step_by(3) {
        if let Some(text) = &rows[i] {
            records += 1;
            bytes += text.len() as u64;
        }
    }
    assert_eq!((stats.records, stats.bytes), (records, bytes));
    let all = scope::scope_stats(&l, None);
    assert_eq!(
        all.bytes,
        rows.iter().flatten().map(|s| s.len() as u64).sum::<u64>()
    );
    assert_eq!(all.records, rows.iter().flatten().count() as u64);
    let zeros = vec![0u8; l.bytes().len()];
    let blind = BorrowedLayout::new(l.offsets(), &zeros, l.nulls()).unwrap();
    assert_eq!(scope::scope_stats(&blind, Some(&scope)), stats);
}

#[test]
fn a_hostile_offsets_array_is_refused_or_scanned_without_panicking() {
    let mut rng = Rng(0xBAD);
    let bytes: Vec<u8> = (0..400).map(|_| b"ab \n"[rng.below(4)]).collect();
    let matchers: Vec<Matcher> = [
        lit("ab", false),
        lit("AB", true),
        rx("a\\s+b", false),
        rx("^ab", false),
    ]
    .iter()
    .map(|p| Matcher::compile(p).unwrap())
    .collect();
    for _ in 0..500 {
        let n = rng.below(20);
        let offsets: Vec<u64> = (0..=n).map(|_| rng.next() % 600).collect();
        let nulls: Vec<u8> = (0..n).map(|_| rng.below(3) as u8).collect();
        let Ok(l) = BorrowedLayout::new(&offsets, &bytes, &nulls) else {
            continue;
        };
        for m in &matchers {
            for oracle_only in [false, true] {
                let _ = scan::scan(
                    &l,
                    m,
                    &ScanOptions {
                        oracle_only,
                        ..Default::default()
                    },
                );
            }
        }
    }
}

/// Debug-build shape of the strategy crossover; run on demand
/// (`-- --ignored --nocapture`). Release numbers are the coordinator's.
#[test]
#[ignore = "prints a timing curve; debug-build shape only"]
fn crossover_curve() {
    use std::time::Instant;
    let rows: Rows = (0..40_000usize)
        .map(|i| {
            Some(format!(
                "rec {i} {} fox jumps over the Ærlig lazy dog {}",
                "lorem ipsum ".repeat(20),
                i % 97
            ))
        })
        .collect();
    let l = layout(&rows);
    let patterns = [
        ("rare", lit("jumps over the Ærlig lazy dog 96", false)),
        ("common", lit("o", false)),
        ("regex", rx("\\bfox\\s+jumps\\b", false)),
    ];
    for (name, pattern) in patterns {
        for clustered in [true, false] {
            for share in [1u32, 5, 10, 25, 50, 75] {
                let n = 40_000 * share / 100;
                let mut rng = Rng(9);
                let scope: Vec<u32> = if clustered {
                    (1000..1000 + n).collect()
                } else {
                    let mut set = std::collections::BTreeSet::new();
                    while (set.len() as u32) < n {
                        set.insert(rng.below(40_000) as u32);
                    }
                    set.into_iter().collect()
                };
                let time = |strategy| {
                    let t = Instant::now();
                    let _ = scoped(&l, &pattern, &scope, strategy);
                    t.elapsed().as_micros()
                };
                println!(
                    "{name:7} {} share {share:3}%  range-driven {:8} us  scan-driven {:8} us",
                    if clustered { "clustered" } else { "scattered" },
                    time(ScopeStrategy::RangeDriven),
                    time(ScopeStrategy::ScanDriven),
                );
            }
        }
    }
}
