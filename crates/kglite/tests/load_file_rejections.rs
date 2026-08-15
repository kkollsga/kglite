//! What `load_file` / `load_kgl_bytes` say about a file they cannot read.
//!
//! The message is the whole product here: a user who points kglite at the
//! wrong path gets nothing else to act on. Until 0.16.1 *every* unrecognised
//! header produced "saved with an older version of kglite" plus rebuild-and-
//! re-save instructions — advice that is actively wrong for a PNG, a CSV, or
//! a truncated download, and that sends people looking for an old binary they
//! never had.
//!
//! These tests live outside `src/graph/io/file.rs` because that file sits at
//! 2485 of its 2500-line ceiling; per the project rule the answer to a full
//! file is not a bigger ceiling.

use std::io::Write;

/// Write `bytes` to a temp file and return `load_file`'s error message.
fn refusal_for(bytes: &[u8], label: &str) -> String {
    let dir =
        std::env::temp_dir().join(format!("kglite-load-reject-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{label}.kgl"));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
    let error = match kglite::api::io::load_file(path.to_str().unwrap()) {
        Ok(_) => panic!("these bytes are not a loadable graph"),
        Err(error) => error,
    };
    let _ = std::fs::remove_dir_all(&dir);
    error.to_string()
}

/// `load_kgl_bytes`'s refusal message for `bytes`.
fn bytes_refusal(bytes: &[u8]) -> String {
    match kglite::api::io::load_kgl_bytes(bytes) {
        Ok(_) => panic!("these bytes are not a loadable graph"),
        Err(error) => error.to_string(),
    }
}

/// A file that is not kglite's at all must be named as such, and must *not*
/// carry the older-version rebuild advice.
#[test]
fn a_non_kglite_file_is_not_reported_as_an_older_version() {
    // Each of these is a real thing a user points at by mistake.
    let cases: [(&str, Vec<u8>); 4] = [
        ("png", b"\x89PNG\r\n\x1a\n".to_vec()),
        ("text", b"id,name\n1,Ada\n".to_vec()),
        ("zeros", vec![0u8; 64]),
        // Truncated download: a real header start, but not kglite's.
        ("zip", b"PK\x03\x04truncated".to_vec()),
    ];
    for (label, bytes) in cases {
        let message = refusal_for(&bytes, label);
        assert!(
            !message.contains("older version of kglite"),
            "{label}: a non-kglite file must not be blamed on an old kglite: {message}"
        );
        assert!(
            message.contains("not a kglite"),
            "{label}: the refusal must say the file is not a kglite file: {message}"
        );
        assert!(
            message.contains(label),
            "{label}: the refusal must name the path it refused: {message}"
        );
    }
}

/// The control: a genuinely older kglite container still gets the
/// upgrade/rebuild path. If this arm ever goes quiet the discriminator has
/// swallowed the case it was carved out of.
#[test]
fn a_genuinely_older_kglite_container_still_gets_rebuild_advice() {
    // "RGF\x02" — the container magic with a version this binary predates.
    // v3..v6 have their own dedicated messages; v0..v2 fall to the generic
    // older-version arm, which is the one under test.
    let message = refusal_for(b"\x52\x47\x46\x02padding-bytes", "older");
    assert!(
        message.contains("older version of kglite"),
        "an RGF-magic container from before v3 is an old kglite file: {message}"
    );
    assert!(
        !message.contains("not a kglite"),
        "it *is* a kglite file — just an unreadable one: {message}"
    );
}

/// The v3 hard break and the newer-container refusal are separately worded
/// and must survive the discriminator untouched.
#[test]
fn the_v3_and_newer_container_arms_are_unchanged() {
    let v3 = refusal_for(b"\x52\x47\x46\x03padding-bytes", "v3");
    assert!(v3.contains("format v3 is not supported"), "{v3}");
    let newer = refusal_for(b"\x52\x47\x46\x7fpadding-bytes", "newer");
    assert!(newer.contains("Please upgrade kglite"), "{newer}");
}

/// Files shorter than the 4-byte magic keep their own message — they cannot
/// be classified at all, so neither branch of the discriminator applies.
#[test]
fn too_short_to_classify_keeps_its_own_message() {
    for (label, bytes) in [("empty", &b""[..]), ("stub", &b"RG"[..])] {
        let message = refusal_for(bytes, label);
        assert!(message.contains("too small"), "{label}: {message}");
    }
}

/// The byte-buffer entry point (`to_bytes()`'s inverse) shares the
/// classification: the same header must produce the same verdict whether it
/// arrives as a file or as bytes.
#[test]
fn the_byte_buffer_entry_point_classifies_identically() {
    let not_kglite = bytes_refusal(b"\x89PNG\r\n\x1a\n");
    assert!(
        not_kglite.contains("not a kglite") && !not_kglite.contains("older version of kglite"),
        "{not_kglite}"
    );

    let older = bytes_refusal(b"\x52\x47\x46\x02padding");
    assert!(older.contains("older version of kglite"), "{older}");
}

/// The large-file branch mmaps instead of reading, and duplicated the whole
/// magic ladder — including the message. Both branches must agree.
#[test]
fn the_mmap_branch_agrees_with_the_small_file_branch() {
    // FILE_MMAP_THRESHOLD is 64 KiB; anything comfortably past it takes the
    // mmap path.
    let mut big = b"\x89PNG\r\n\x1a\n".to_vec();
    big.resize(256 * 1024, 0);
    let mmapped = refusal_for(&big, "bigpng");
    assert!(
        mmapped.contains("not a kglite") && !mmapped.contains("older version of kglite"),
        "the mmap branch must classify like the small-file branch: {mmapped}"
    );

    let mut big_old = b"\x52\x47\x46\x02".to_vec();
    big_old.resize(256 * 1024, 0);
    let mmapped_old = refusal_for(&big_old, "bigold");
    assert!(
        mmapped_old.contains("older version of kglite"),
        "{mmapped_old}"
    );
}
