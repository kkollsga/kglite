//! pandas policy belongs to the Python binding, not the engine Value model.
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyDict, PyList};

/// Wrap one column's raw little/big-endian-native cells as a numpy array of
/// `dtype`, for a caller that has proven every cell shares that layout.
///
/// The buffer is a `bytearray` rather than `bytes` so the array numpy builds
/// over it is **writable**: `frombuffer` inherits the buffer's writability,
/// and a pandas version that adopted the array without copying would hand the
/// caller a frame column that refuses the in-place assignment a boxed-list
/// column accepts.
pub(crate) fn numeric_column<'py>(
    py: Python<'py>,
    numpy: &Bound<'py, PyModule>,
    cells: &[u8],
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let buffer = PyByteArray::new(py, cells);
    numpy.getattr("frombuffer")?.call1((buffer, dtype))
}

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
