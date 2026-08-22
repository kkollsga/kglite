//! Where a query warning is *announced*, for this Python process.
//!
//! Query warnings ride two channels, and only one of them is negotiable:
//!
//! * **Structured** — `QueryDiagnostics::warnings`, surfaced as
//!   `ResultView.diagnostics["warnings"]` and `ResultView.warnings`. Computed
//!   by the engine on every execution and **unconditional**: no policy here
//!   can empty it, because a caller reading the field must never have to know
//!   what some other module set process-wide.
//! * **The echo** — the announcement a caller who is *not* reading the field
//!   still sees. That is what this module owns.
//!
//! Three policies, chosen by [`set_query_warning_policy`]:
//!
//! | policy     | engine sink | Python `warnings.warn` |
//! |------------|-------------|------------------------|
//! | `"stderr"` | stderr      | no                     |
//! | `"silent"` | silent      | no                     |
//! | `"pywarn"` | silent      | yes (`UserWarning`)    |
//!
//! `"stderr"` is the default and is byte-for-byte what every release before
//! this one did. `"pywarn"` is **never** the default: `warnings.warn` under
//! `-W error` (or `filterwarnings = error`, which many test suites set) turns
//! an advisory into a raise out of `cypher()`, and flipping that on by default
//! would break those suites on upgrade.
//!
//! ## Why `pywarn` *replaces* stderr rather than adding to it
//!
//! Because `warnings.warn` is itself a routing decision. Its default handler
//! already prints to stderr, so emitting both would double every message for
//! the common case; and a host that installed `logging.captureWarnings(True)`,
//! a `simplefilter("ignore")`, or a custom `showwarning` asked for the
//! warnings to go somewhere specific — writing them to stderr anyway would
//! defeat exactly the routing they set up.
//!
//! ## Why the re-emission happens here and not in the engine
//!
//! The engine cannot depend on Python, and a callback registered *into* it
//! would have to fire from inside `py.detach` — deep in the executor, with the
//! GIL released and graph locks held, in a context that cannot propagate a
//! `PyErr` (so `-W error` would be swallowed). Instead the engine's sink is
//! set to silent and this module re-emits **after** execution, from the
//! diagnostics the boundary already holds: the GIL is held, no lock is, and a
//! raise from `warnings.warn` propagates out of the calling method the way a
//! Python developer expects.
//!
//! Every wheel entry point that receives a `CypherResult` from the engine
//! calls [`announce`] once. That is the complete list — `KnowledgeGraph.cypher`
//! (read and mutation), `Session` (read and write), `Transaction`,
//! `FrozenGraph` — and it is placed *before* the output-shape branch on
//! purpose, so `to_df=True` and `FORMAT CSV`, whose return values have nowhere
//! to carry diagnostics, still get the echo.

use std::sync::atomic::{AtomicU8, Ordering};

use pyo3::prelude::*;

use crate::graph::languages::cypher::QueryDiagnostics;
use kglite_core::api::cypher::{set_query_warning_sink, QueryWarningSink};

const POLICY_STDERR: u8 = 0;
const POLICY_SILENT: u8 = 1;
const POLICY_PYWARN: u8 = 2;

/// Mirrors the engine sink plus the wheel-only `pywarn` arm. Kept as a
/// separate atomic rather than read back from the engine because `silent` and
/// `pywarn` both map onto the engine's `Silent`, and `get_query_warning_policy`
/// must tell the caller which of the two they chose.
static POLICY: AtomicU8 = AtomicU8::new(POLICY_STDERR);

/// Select how query warnings are announced, for the whole process.
///
/// `"stderr"` (default), `"silent"`, or `"pywarn"`. See the module docs for
/// the semantics; the docstring users read is in `kglite/__init__.pyi`.
#[pyfunction]
pub fn set_query_warning_policy(policy: &str) -> PyResult<()> {
    let (code, sink) = match policy {
        "stderr" => (POLICY_STDERR, QueryWarningSink::Stderr),
        "silent" => (POLICY_SILENT, QueryWarningSink::Silent),
        "pywarn" => (POLICY_PYWARN, QueryWarningSink::Silent),
        other => {
            return Err(crate::error_py::ArgumentError::new_err(format!(
                "set_query_warning_policy expects 'stderr', 'silent' or 'pywarn'. Got \
                 '{other}'. 'stderr' prints `warning: ...` (the default), 'silent' prints \
                 nothing, 'pywarn' raises a UserWarning through the warnings module. \
                 ResultView.warnings carries the warnings under every policy."
            )))
        }
    };
    // Order matters only for a concurrent reader, and either order leaves a
    // brief window where one query echoes under the outgoing policy. Set the
    // engine sink first so no window can produce *both* echoes.
    set_query_warning_sink(sink);
    POLICY.store(code, Ordering::Relaxed);
    Ok(())
}

/// The policy currently in effect: `"stderr"`, `"silent"` or `"pywarn"`.
#[pyfunction]
pub fn get_query_warning_policy() -> &'static str {
    match POLICY.load(Ordering::Relaxed) {
        POLICY_SILENT => "silent",
        POLICY_PYWARN => "pywarn",
        _ => "stderr",
    }
}

/// Announce `diagnostics`' warnings the way the current policy asks.
///
/// A no-op under `"stderr"` and `"silent"` — the engine's own sink already
/// did (or deliberately did not do) the echo. Under `"pywarn"` each warning
/// becomes one `UserWarning`; the first one that a filter turns into an error
/// propagates, and the rest are not emitted, matching what a sequence of
/// `warnings.warn` calls in Python would do.
///
/// Called once per engine execution, at the wheel boundary, before the
/// result's output shape is chosen.
pub fn announce(py: Python<'_>, diagnostics: Option<&QueryDiagnostics>) -> PyResult<()> {
    if POLICY.load(Ordering::Relaxed) != POLICY_PYWARN {
        return Ok(());
    }
    let Some(diagnostics) = diagnostics else {
        return Ok(());
    };
    for message in &diagnostics.warnings {
        // A NUL inside a warning would truncate the C string; the engine's
        // messages are formatted from identifiers and never contain one, and
        // `unwrap_or_default` degrades to an empty warning rather than losing
        // the fact that something was flagged.
        let cmsg = std::ffi::CString::new(message.as_str()).unwrap_or_default();
        // stacklevel 1: there is no Python frame for this Rust function, so 1
        // is the caller's `cypher(...)` line — the place the query was written.
        PyErr::warn(
            py,
            py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
            cmsg.as_c_str(),
            1,
        )?;
    }
    Ok(())
}
