//! Python containers have no native graph-value equivalent when recursive.
//! Conversion admits at most 64 active container expansions, including ndarray
//! `tolist()` expansion; repeated acyclic objects in sibling branches are legal.

use super::values::Value;
use pyo3::exceptions::{PyRecursionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDate, PyDateTime, PyDict, PyList, PyTuple, PyTzInfo, PyTzInfoAccess};

const MAX_CONTAINER_DEPTH: usize = 64;

#[derive(Debug, PartialEq, Eq)]
enum ConversionLimit {
    Cycle,
    Depth,
}

impl From<ConversionLimit> for PyErr {
    fn from(limit: ConversionLimit) -> Self {
        match limit {
            ConversionLimit::Cycle => {
                PyValueError::new_err("Recursive Python containers cannot be converted")
            }
            ConversionLimit::Depth => {
                PyRecursionError::new_err("Python value exceeds the 64-container conversion depth")
            }
        }
    }
}

#[derive(Default)]
struct ConversionState {
    active: Vec<usize>,
}

impl ConversionState {
    fn enter(&mut self, identity: usize) -> Result<(), ConversionLimit> {
        if self.active.contains(&identity) {
            return Err(ConversionLimit::Cycle);
        }
        if self.active.len() >= MAX_CONTAINER_DEPTH {
            return Err(ConversionLimit::Depth);
        }
        self.active.push(identity);
        Ok(())
    }

    fn with_container<T>(
        &mut self,
        identity: usize,
        convert: impl FnOnce(&mut Self) -> PyResult<T>,
    ) -> PyResult<T> {
        self.enter(identity)?;
        let result = convert(self);
        self.active.pop();
        result
    }
}

/// True for numpy's ndarray without importing numpy when it is absent.
pub(super) fn is_numpy_ndarray(value: &Bound<'_, PyAny>) -> bool {
    let ty = value.get_type();
    ty.name().map(|n| n == "ndarray").unwrap_or(false)
        && ty
            .getattr("__module__")
            .ok()
            .and_then(|m| m.extract::<String>().ok())
            .is_some_and(|m| m == "numpy" || m.starts_with("numpy."))
}

pub fn py_value_to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    convert_value(value, &mut ConversionState::default())
}

fn convert_value(value: &Bound<'_, PyAny>, state: &mut ConversionState) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    // bool is an int subclass; size-one ndarray can extract as a scalar.
    if value.is_instance_of::<pyo3::types::PyBool>() {
        if let Ok(b) = value.extract::<bool>() {
            return Ok(Value::Boolean(b));
        }
    }
    if is_numpy_ndarray(value) {
        return state.with_container(value.as_ptr() as usize, |state| {
            let as_list = value.call_method0("tolist")?;
            convert_value(&as_list, state)
        });
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Value::Int64(i));
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(Value::Float64(f));
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(u) = value.extract::<u32>() {
        return Ok(Value::UniqueId(u));
    }
    // datetime is a date subclass: failure must not degrade to a date-only value.
    if let Ok(dt) = value.cast::<PyDateTime>() {
        return convert_datetime(dt);
    }
    if value.is_instance_of::<PyDate>() {
        if let Ok(d) = value.extract::<chrono::NaiveDate>() {
            return Ok(Value::DateTime(d));
        }
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        return state.with_container(value.as_ptr() as usize, |state| convert_dict(dict, state));
    }
    if let Ok(list) = value.cast::<PyList>() {
        return state.with_container(value.as_ptr() as usize, |state| {
            list.iter()
                .map(|item| convert_value(&item, state))
                .collect::<PyResult<Vec<_>>>()
                .map(Value::List)
        });
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return state.with_container(value.as_ptr() as usize, |state| {
            tuple
                .iter()
                .map(|item| convert_value(&item, state))
                .collect::<PyResult<Vec<_>>>()
                .map(Value::List)
        });
    }
    Ok(Value::Null)
}

fn convert_dict(dict: &Bound<'_, PyDict>, state: &mut ConversionState) -> PyResult<Value> {
    // Preserve the pair-buffer construction: PropMap sorts once after conversion.
    let mut pairs = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        pairs.push((
            kglite_core::datatypes::PropKey::from(key),
            convert_value(&value, state)?,
        ));
    }
    Ok(Value::Map(kglite_core::datatypes::PropMap::from_pairs(
        pairs,
    )))
}

fn convert_datetime(value: &Bound<'_, PyDateTime>) -> PyResult<Value> {
    if value.get_tzinfo().is_none() {
        return value
            .extract::<chrono::NaiveDateTime>()
            .map(Value::Timestamp);
    }
    // A tzinfo whose utcoffset(dt) is None is naive by Python's definition.
    // Let datetime normalize aware values; extracting FixedOffset from tzinfo
    // alone loses date-dependent offsets and fractional offset seconds.
    let normalized = if value.call_method0("utcoffset")?.is_none() {
        value.clone().into_any()
    } else {
        value.call_method1("astimezone", (PyTzInfo::utc(value.py())?,))?
    };
    let kwargs = PyDict::new(value.py());
    kwargs.set_item("tzinfo", value.py().None())?;
    normalized
        .call_method("replace", (), Some(&kwargs))?
        .extract::<chrono::NaiveDateTime>()
        .map(Value::Timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_identity_refused_without_changing_path() {
        let mut state = ConversionState::default();
        assert_eq!(state.enter(17), Ok(()));
        assert_eq!(state.enter(17), Err(ConversionLimit::Cycle));
        assert_eq!(state.active, [17]);
        state.active.pop();
        assert_eq!(state.enter(17), Ok(()));
    }

    #[test]
    fn synthetic_depth_limit_is_exact_and_does_not_push() {
        let mut state = ConversionState::default();
        for identity in 0..MAX_CONTAINER_DEPTH {
            assert_eq!(state.enter(identity), Ok(()));
        }
        assert_eq!(
            state.enter(MAX_CONTAINER_DEPTH),
            Err(ConversionLimit::Depth)
        );
        assert_eq!(state.active.len(), MAX_CONTAINER_DEPTH);
        state.active.pop();
        assert_eq!(state.enter(MAX_CONTAINER_DEPTH), Ok(()));
    }

    #[test]
    fn container_scope_unwinds_on_success_and_error() {
        Python::initialize();
        Python::attach(|py| {
            let mut state = ConversionState::default();
            let error = state
                .with_container(1, |state| state.with_container(1, |_| Ok(())))
                .unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(state.active.is_empty());
            let error = state
                .with_container(1, |state| {
                    state.with_container(2, |_| {
                        Err::<(), _>(PyValueError::new_err("ordinary conversion error"))
                    })
                })
                .unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(state.active.is_empty());
            assert_eq!(state.with_container(1, |_| Ok(7)).unwrap(), 7);
            assert!(state.active.is_empty());
        });
    }

    #[test]
    fn shallow_visitor_shares_existing_depth_budget() {
        Python::initialize();
        Python::attach(|py| {
            // Synthetic outer frames exercise the real visitor with only two
            // ordinary lists; no deeply nested Python object is constructed.
            let inner = PyList::new(py, [1]).unwrap();
            let outer = PyList::new(py, [inner]).unwrap();
            let synthetic = vec![usize::MAX; MAX_CONTAINER_DEPTH - 1];
            let mut state = ConversionState {
                active: synthetic.clone(),
            };
            let error = convert_value(outer.as_any(), &mut state).unwrap_err();
            assert!(error.is_instance_of::<PyRecursionError>(py));
            assert_eq!(state.active, synthetic);
            state.active.clear();
            assert_eq!(
                convert_value(outer.as_any(), &mut state).unwrap(),
                Value::List(vec![Value::List(vec![Value::Int64(1)])])
            );
            assert!(state.active.is_empty());
        });
    }
}
