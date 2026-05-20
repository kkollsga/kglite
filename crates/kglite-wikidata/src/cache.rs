//! Dump cache lifecycle — the staleness / cooldown / resumable-`.part`
//! state machine. Ported from the Python `wikidata.py` `_ensure_dump`.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};

use crate::client::WikidataClient;
use crate::error::Result;
use crate::layout::{Workdir, DUMP_URL};

/// Resolve the local dump, downloading or resuming as needed.
///
/// Returns `(dump_path, mtime_to_record)` — the second value is the
/// timestamp the caller should stamp into `wikidata_source.json`:
/// the remote `Last-Modified` when freshly checked, or the local
/// file's mtime when the dump was kept within cooldown.
pub async fn ensure_dump(
    workdir: &Workdir,
    cooldown_days: i64,
    verbose: bool,
) -> Result<(PathBuf, Option<DateTime<Utc>>)> {
    workdir.ensure_dir()?;
    let local = workdir.dump_path();
    let part = workdir.part_path();

    let client = WikidataClient::new()?;
    // Remote unreachable → `None`; the caller falls back to the local copy.
    let remote_mtime = client
        .head(DUMP_URL)
        .await
        .ok()
        .and_then(|m| m.last_modified);

    if let Some(local_mtime) = file_mtime_utc(&local) {
        let age = age_days(local_mtime);
        if verbose {
            eprintln!("  Local dump present, age {age:.1}d");
        }
        match remote_mtime {
            None => {
                if verbose {
                    eprintln!("  Remote unreachable — using local copy.");
                }
                return Ok((local, None));
            }
            Some(remote) => {
                if remote <= local_mtime {
                    if verbose {
                        eprintln!("  Local dump matches latest remote.");
                    }
                    return Ok((local, Some(remote)));
                }
                if age < cooldown_days as f64 {
                    if verbose {
                        eprintln!(
                            "  Newer dump available, but local is within cooldown \
                             ({age:.1}d < {cooldown_days}d)."
                        );
                    }
                    return Ok((local, Some(local_mtime)));
                }
                if verbose {
                    eprintln!("  Newer dump available + cooldown elapsed. Refreshing.");
                }
                let _ = std::fs::remove_file(&local);
                let _ = std::fs::remove_file(&part);
            }
        }
    }

    // A fresh-enough `.part` from an interrupted download → resume it.
    if let Some(part_mtime) = file_mtime_utc(&part) {
        if age_days(part_mtime) >= cooldown_days as f64 {
            if verbose {
                eprintln!("  Stale partial download — discarding.");
            }
            let _ = std::fs::remove_file(&part);
        } else {
            client.download_resumable(DUMP_URL, &part, verbose).await?;
            std::fs::rename(&part, &local)?;
            return Ok((local, remote_mtime));
        }
    }

    client.download_resumable(DUMP_URL, &part, verbose).await?;
    std::fs::rename(&part, &local)?;
    Ok((local, remote_mtime))
}

/// The remote dump's `Last-Modified`, or `None` if unreachable. Used
/// for the disk-mode "is the cached graph still current?" check.
pub async fn remote_last_modified() -> Option<DateTime<Utc>> {
    let client = WikidataClient::new().ok()?;
    client
        .head(DUMP_URL)
        .await
        .ok()
        .and_then(|m| m.last_modified)
}

fn file_mtime_utc(path: &Path) -> Option<DateTime<Utc>> {
    let mtime = std::fs::metadata(path).ok()?.modified().ok()?;
    Some(DateTime::<Utc>::from(mtime))
}

fn age_days(when: DateTime<Utc>) -> f64 {
    Utc::now().signed_duration_since(when).num_seconds() as f64 / 86_400.0
}
