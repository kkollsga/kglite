//! Whether a `.kgl` load builds its declared indexes or only records them.
//!
//! The eager rebuild at the end of `attach_portable_column_stores` is the
//! single largest *settled* memory term on an index-bearing graph — measured at
//! 65–100 MB, ~40–50% of a 500k-row four-index fixture's footprint
//! (`dev-docs/bench/results/load-rss-2026-08-29.md` §4). A reader that never
//! needs the indexes pays it anyway.
//!
//! This module owns the switch. It is deliberately *not* a `DirGraph` field or
//! a threaded parameter yet: P3 of the load-memory program replaces it with the
//! `LoadOptions.defer_index_rebuild` field it plumbs through the load path, and
//! deletes this module.
//!
//! Two inputs, thread-local first:
//! - [`DeferIndexRebuild`] — a scoped in-process override, test-only until P3
//!   gives the option a real caller.
//! - `KGLITE_DEFER_INDEX_REBUILD` — `1`/`true`/`yes`/`on` enable, `0`/`false`/
//!   `no`/`off` disable, empty/absent means disabled. Anything else warns on
//!   stderr and is treated as disabled, rather than silently doing the opposite
//!   of what the operator asked.

use std::cell::Cell;

/// Environment switch; see the module doc for accepted values.
pub(crate) const DEFER_ENV_VAR: &str = "KGLITE_DEFER_INDEX_REBUILD";

thread_local! {
    static SCOPED_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Scoped override of [`defer_index_rebuild_requested`] for the current
/// thread, restoring the previous value on drop. Nested scopes stack.
///
/// Exists because the alternative — `std::env::set_var` — is process-wide and
/// unsound to call while other threads run, which is every `cargo test`.
#[cfg(test)]
pub(crate) struct DeferIndexRebuild {
    previous: Option<bool>,
}

#[cfg(test)]
impl DeferIndexRebuild {
    /// Force the deferred (`true`) or eager (`false`) load path on this thread
    /// until the returned guard drops.
    pub(crate) fn scoped(defer: bool) -> Self {
        let previous = SCOPED_OVERRIDE.with(|slot| slot.replace(Some(defer)));
        DeferIndexRebuild { previous }
    }
}

#[cfg(test)]
impl Drop for DeferIndexRebuild {
    fn drop(&mut self) {
        SCOPED_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

/// Whether this load should record its declared indexes instead of building
/// them.
pub(crate) fn defer_index_rebuild_requested() -> bool {
    if let Some(scoped) = SCOPED_OVERRIDE.with(|slot| slot.get()) {
        return scoped;
    }
    match std::env::var(DEFER_ENV_VAR) {
        Ok(raw) => parse_flag(&raw),
        Err(_) => false,
    }
}

fn parse_flag(raw: &str) -> bool {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" => false,
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        other => {
            eprintln!(
                "kglite: ignoring {DEFER_ENV_VAR}={other:?} — expected one of \
                 1/true/yes/on or 0/false/no/off; loading with indexes built"
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_override_nests_and_restores() {
        assert!(!defer_index_rebuild_requested());
        let outer = DeferIndexRebuild::scoped(true);
        assert!(defer_index_rebuild_requested());
        {
            let _inner = DeferIndexRebuild::scoped(false);
            assert!(!defer_index_rebuild_requested());
        }
        assert!(defer_index_rebuild_requested());
        drop(outer);
        assert!(!defer_index_rebuild_requested());
    }

    #[test]
    fn unparseable_flag_is_loud_and_falls_back_to_eager() {
        assert!(parse_flag("on"));
        assert!(parse_flag(" TRUE "));
        assert!(!parse_flag("off"));
        assert!(!parse_flag(""));
        // The warning goes to stderr; the value that matters is the refusal to
        // guess. `maybe` must not enable a memory mode nobody asked for.
        assert!(!parse_flag("maybe"));
    }
}
