//! pandas policy belongs to the Python binding, not the engine Value model.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(crate) fn dataframe(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    columns: Option<&Bound<'_, PyList>>,
    dtypes: Option<&Bound<'_, PyDict>>,
    native_dtypes: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let kwargs = PyDict::new(py);
    if let Some(columns) = columns {
        kwargs.set_item("columns", columns)?;
    }
    if let Some(dtypes) = dtypes {
        kwargs.set_item("dtypes", dtypes)?;
    }
    if let Some(native_dtypes) = native_dtypes {
        kwargs.set_item("native_dtypes", native_dtypes)?;
    }
    py.import("kglite._pandas")?
        .getattr("dataframe")?
        .call((data,), Some(&kwargs))
        .map(Bound::unbind)
}
