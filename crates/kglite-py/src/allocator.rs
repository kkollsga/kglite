//! Process-global allocator maintenance — `kglite.trim_memory()`.
//!
//! Wrapper-specific by construction: the `#[global_allocator]` in `lib.rs`
//! belongs to *this* extension module, not to the `kglite` engine crate (which
//! sets no allocator and inherits whatever its host chose). A sibling binding
//! that does not link mimalloc has nothing to trim, so there is nothing here
//! for `kglite::api` to own.

use pyo3::prelude::*;

// mimalloc's maintenance entry point. `libmimalloc-sys` compiles the whole v2
// library into this extension but binds only the allocation functions, so the
// call is declared here. `mi_collect` is `mi_decl_export`ed in `mimalloc.h`
// and defined in the same translation unit (`src/static.c`) as `mi_malloc`, so
// the object file is unconditionally linked; naming it is what stops the
// linker dead-stripping it out of the final extension.
extern "C" {
    /// `void mi_collect(bool force)`.
    fn mi_collect(force: bool);
}

/// Return allocator-retained memory to the operating system.
///
/// Deliberately opt-in. mimalloc keeps a freed workload's pages so the next
/// one can reuse them without a syscall, which is exactly what makes repeated
/// queries fast; forcing a collect at an internal seam (after `save()`, on
/// graph drop) would trade that away on every call. The lever is exported so
/// a long-lived host can spend the milliseconds where *it* knows a peak is
/// over.
#[pyfunction]
pub(crate) fn trim_memory(py: Python<'_>) {
    // SAFETY: `mi_collect` takes no pointers, is the allocator's own public
    // maintenance entry point, and is safe to call from any thread at any
    // time (mimalloc handles being called by a thread that does not own the
    // heap). The GIL is released because a forced collect walks every page
    // heap and every arena, which takes milliseconds after a large peak.
    py.detach(|| unsafe { mi_collect(true) });
}
