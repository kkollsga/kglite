//! PyO3 entry points for code_tree.

pub mod entry;

use pyo3::prelude::*;

/// Register the `code_tree` submodule on the parent `kglite` module.
/// Called from `src/lib.rs`.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "_kglite_code_tree")?;
    m.add_function(wrap_pyfunction!(entry::build, &m)?)?;
    m.add_function(wrap_pyfunction!(entry::read_manifest, &m)?)?;
    m.add_function(wrap_pyfunction!(entry::repo_tree, &m)?)?;
    parent.add_submodule(&m)?;
    // Register the submodule in sys.modules so `from kglite._kglite_code_tree import ...` works.
    py.import("sys")?
        .getattr("modules")?
        .set_item("kglite._kglite_code_tree", &m)?;
    Ok(())
}
