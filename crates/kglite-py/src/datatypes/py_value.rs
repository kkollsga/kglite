//! Python containers have no native graph-value equivalent when recursive.
//! Conversion admits at most 64 active container expansions, including ndarray
//! `tolist()` expansion; repeated acyclic objects in sibling branches are legal.

use super::values::Value;
use pyo3::exceptions::{PyOverflowError, PyRecursionError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBytes, PyDate, PyDateTime, PyDelta, PyDict, PyFloat, PyInt, PyList, PyTuple, PyTzInfo,
    PyTzInfoAccess,
};

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

    /// Refuse, as the `tolist()` route would, a decoded ndarray whose nested
    /// lists would push the active path past the depth limit.
    fn admit_nested_lists(&self, levels: usize) -> Result<(), ConversionLimit> {
        if self.active.len() + levels > MAX_CONTAINER_DEPTH {
            return Err(ConversionLimit::Depth);
        }
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

    fn with_query_container<T>(
        &mut self,
        identity: usize,
        convert: impl FnOnce(&mut Self) -> Result<T, QueryConversionError>,
    ) -> Result<T, QueryConversionError> {
        self.enter(identity).map_err(QueryConversionError::Limit)?;
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

/// A numeric ndarray element type the `tobytes()` decoders admit: native (or
/// byte-order-free) float16/32/64, int8–64 and uint8–32. Every other dtype —
/// bool, uint64, object, datetime, a non-native byte order — reads as `None`
/// and keeps its caller's slower route. uint64 is excluded because `tolist()`
/// can yield an int past i64::MAX, which each caller reports in its own way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericElement {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
}

impl NumericElement {
    /// The one dtype check both decoders share.
    fn of_dtype(dtype: &Bound<'_, PyAny>) -> Option<Self> {
        let byteorder: String = dtype.getattr("byteorder").ok()?.extract().ok()?;
        if byteorder != "=" && byteorder != "|" {
            return None;
        }
        let kind: String = dtype.getattr("kind").ok()?.extract().ok()?;
        let itemsize: usize = dtype.getattr("itemsize").ok()?.extract().ok()?;
        Some(match (kind.as_str(), itemsize) {
            ("f", 2) => Self::F16,
            ("f", 4) => Self::F32,
            ("f", 8) => Self::F64,
            ("i", 1) => Self::I8,
            ("i", 2) => Self::I16,
            ("i", 4) => Self::I32,
            ("i", 8) => Self::I64,
            ("u", 1) => Self::U8,
            ("u", 2) => Self::U16,
            ("u", 4) => Self::U32,
            _ => return None,
        })
    }

    fn itemsize(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::F16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 => 8,
        }
    }

    /// Exactly what `tolist()` yields: floats widen to f64, integers to i64.
    fn value_decoder(self) -> fn(&[u8]) -> Value {
        match self {
            Self::F16 => |b| Value::Float64(f16_bits_to_f64(u16::from_ne_bytes([b[0], b[1]]))),
            Self::F32 => |b| Value::Float64(f64::from(f32::from_ne_bytes(b.try_into().unwrap()))),
            Self::F64 => |b| Value::Float64(f64::from_ne_bytes(b.try_into().unwrap())),
            Self::I8 => |b| Value::Int64(i64::from(i8::from_ne_bytes([b[0]]))),
            Self::I16 => |b| Value::Int64(i64::from(i16::from_ne_bytes([b[0], b[1]]))),
            Self::I32 => |b| Value::Int64(i64::from(i32::from_ne_bytes(b.try_into().unwrap()))),
            Self::I64 => |b| Value::Int64(i64::from_ne_bytes(b.try_into().unwrap())),
            Self::U8 => |b| Value::Int64(i64::from(b[0])),
            Self::U16 => |b| Value::Int64(i64::from(u16::from_ne_bytes([b[0], b[1]]))),
            Self::U32 => |b| Value::Int64(i64::from(u32::from_ne_bytes(b.try_into().unwrap()))),
        }
    }

    /// Exactly what extracting each element as a Python float and narrowing
    /// it to f32 yields — the `Vec<f32>` extraction this replaces.
    fn f32_decoder(self) -> fn(&[u8]) -> f32 {
        match self {
            Self::F16 => |b| f16_bits_to_f64(u16::from_ne_bytes([b[0], b[1]])) as f32,
            Self::F32 => |b| f32::from_ne_bytes(b.try_into().unwrap()),
            Self::F64 => |b| f64::from_ne_bytes(b.try_into().unwrap()) as f32,
            Self::I8 => |b| f64::from(i8::from_ne_bytes([b[0]])) as f32,
            Self::I16 => |b| f64::from(i16::from_ne_bytes([b[0], b[1]])) as f32,
            Self::I32 => |b| f64::from(i32::from_ne_bytes(b.try_into().unwrap())) as f32,
            Self::I64 => |b| i64::from_ne_bytes(b.try_into().unwrap()) as f64 as f32,
            Self::U8 => |b| f64::from(b[0]) as f32,
            Self::U16 => |b| f64::from(u16::from_ne_bytes([b[0], b[1]])) as f32,
            Self::U32 => |b| f64::from(u32::from_ne_bytes(b.try_into().unwrap())) as f32,
        }
    }
}

/// A native-order numeric ndarray of rank 1 or 2 decoded from `tobytes()`
/// (C order, so strides do not matter) into what `tolist()` would have
/// produced, plus the number of list levels that result nests. `None` sends
/// the array down the `tolist()` route unchanged: dtypes [`NumericElement`]
/// does not admit, and ranks 0 or 3+.
fn decode_numeric_ndarray(value: &Bound<'_, PyAny>) -> Option<(Value, usize)> {
    let element = NumericElement::of_dtype(&value.getattr("dtype").ok()?)?;
    let itemsize = element.itemsize();
    let decode = element.value_decoder();
    let shape: Vec<usize> = value.getattr("shape").ok()?.extract().ok()?;
    if shape.len() != 1 && shape.len() != 2 {
        return None;
    }
    let bytes = value.call_method0("tobytes").ok()?;
    let bytes = bytes.cast::<PyBytes>().ok()?.as_bytes();
    let row = |chunk: &[u8]| Value::List(chunk.chunks_exact(itemsize).map(decode).collect());
    if shape.len() == 1 {
        return Some((row(bytes), 1));
    }
    // An empty outer axis yields `[]` from `tolist()`: no inner lists nest.
    if shape[0] == 0 {
        return Some((Value::List(Vec::new()), 1));
    }
    let row_bytes = shape[1] * itemsize;
    let rows = if row_bytes == 0 {
        vec![Value::List(Vec::new()); shape[0]]
    } else {
        bytes.chunks_exact(row_bytes).map(row).collect()
    };
    Some((Value::List(rows), 2))
}

/// Decodes many embedding rows to `Vec<f32>`, one Python object at a time,
/// taking 1-D numeric ndarrays through `tobytes()` instead of one float
/// extraction per element. The ndarray type and each distinct dtype object are
/// checked once and remembered (numpy reuses one dtype object per builtin
/// type), so a batch of same-typed rows costs three Python calls per row.
/// Anything else — a list, a 2-D array, an unadmitted dtype — is extracted as
/// before, with the error that extraction has always raised.
#[derive(Default)]
pub struct F32Rows {
    ndarray_type: Option<usize>,
    dtype: Option<(Py<PyAny>, Option<NumericElement>)>,
}

impl F32Rows {
    pub fn extract(&mut self, value: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
        match self.decode_ndarray(value) {
            Some(vector) => Ok(vector),
            None => value.extract(),
        }
    }

    fn decode_ndarray(&mut self, value: &Bound<'_, PyAny>) -> Option<Vec<f32>> {
        let value_type = value.get_type().as_ptr() as usize;
        if self.ndarray_type != Some(value_type) {
            if !is_numpy_ndarray(value) {
                return None;
            }
            self.ndarray_type = Some(value_type);
        }
        let dtype = value.getattr("dtype").ok()?;
        let element = match &self.dtype {
            Some((cached, element)) if cached.is(&dtype) => *element,
            _ => {
                let element = NumericElement::of_dtype(&dtype);
                self.dtype = Some((dtype.unbind(), element));
                element
            }
        }?;
        let ndim: usize = value.getattr("ndim").ok()?.extract().ok()?;
        if ndim != 1 {
            return None;
        }
        let bytes = value.call_method0("tobytes").ok()?;
        let bytes = bytes.cast::<PyBytes>().ok()?.as_bytes();
        let decode = element.f32_decoder();
        Some(bytes.chunks_exact(element.itemsize()).map(decode).collect())
    }
}

/// IEEE half-precision bits to the f64 of the same value, as numpy's
/// `npy_halfbits_to_doublebits` widens (NaN payload shifted, not replaced).
fn f16_bits_to_f64(bits: u16) -> f64 {
    let sign = u64::from(bits & 0x8000) << 48;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u64::from(bits & 0x03ff);
    let magnitude = match exponent {
        0 if mantissa == 0 => 0,
        // Subnormal: exact as mantissa * 2^-24.
        0 => {
            let value = (mantissa as f64) * (-24f64).exp2();
            return if sign == 0 { value } else { -value };
        }
        0x1f => 0x7ff0_0000_0000_0000 | (mantissa << 42),
        _ => ((u64::from(exponent) + 1008) << 52) | (mantissa << 42),
    };
    f64::from_bits(sign | magnitude)
}

pub fn py_value_to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    convert_value(value, &mut ConversionState::default())
}

enum QueryConversionError {
    Limit(ConversionLimit),
    Python(PyErr),
    IntegerOverflow,
    Unsupported(String),
    AtIndex(usize, Box<Self>),
    AtKey(String, Box<Self>),
}

impl From<PyErr> for QueryConversionError {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}

impl QueryConversionError {
    fn into_pyerr(self, parameter: &str) -> PyErr {
        let mut path = format!("${parameter}");
        let mut error = self;
        loop {
            error = match error {
                Self::AtIndex(index, inner) => {
                    path.push_str(&format!("[{index}]"));
                    *inner
                }
                Self::AtKey(key, inner) => {
                    path.push('.');
                    path.push_str(&key);
                    *inner
                }
                Self::IntegerOverflow => {
                    return PyOverflowError::new_err(format!(
                        "Query parameter {path} is outside the signed 64-bit integer range"
                    ));
                }
                Self::Unsupported(type_name) => {
                    return PyTypeError::new_err(format!(
                        "Query parameter {path} has unsupported Python type '{type_name}'"
                    ));
                }
                Self::Limit(limit) => return limit.into(),
                // Re-raise the inner failure under the path the typed arms
                // build, so the stub's promise ("both errors identify the
                // nested parameter path") holds for *any* nested failure and
                // not only the two typed ones. The class is preserved: a
                // caller routing on `TypeError` still sees `TypeError`.
                Self::Python(error) => {
                    return Python::attach(|py| {
                        let detail = error
                            .value(py)
                            .str()
                            .ok()
                            .and_then(|text| text.extract::<String>().ok())
                            .unwrap_or_else(|| error.to_string());
                        PyErr::from_type(
                            error.get_type(py),
                            format!("Query parameter {path} could not be converted: {detail}"),
                        )
                    });
                }
            };
        }
    }
}

/// Convert one Python query parameter without silently losing its value.
pub fn py_query_parameter_to_value(parameter: &str, value: &Bound<'_, PyAny>) -> PyResult<Value> {
    convert_query_value(value, &mut ConversionState::default())
        .map_err(|error| error.into_pyerr(parameter))
}

fn numpy_scalar_type_name(value: &Bound<'_, PyAny>) -> Option<String> {
    let ty = value.get_type();
    let is_numpy = ty
        .getattr("__module__")
        .ok()
        .and_then(|module| module.extract::<String>().ok())
        .is_some_and(|module| module == "numpy" || module.starts_with("numpy."));
    if !is_numpy {
        return None;
    }
    ty.name().ok()?.extract::<String>().ok()
}

/// Whether `value` is pandas' `NaT` singleton, identified by its type rather
/// than by importing pandas (which the wheel does not depend on).
fn is_pandas_nat(value: &Bound<'_, PyAny>) -> bool {
    value
        .get_type()
        .name()
        .ok()
        .and_then(|name| name.extract::<String>().ok())
        .is_some_and(|name| name == "NaTType")
}

fn convert_query_value(
    value: &Bound<'_, PyAny>,
    state: &mut ConversionState,
) -> Result<Value, QueryConversionError> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    // Exact builtin types skip the numpy type-name lookups below, which cost
    // a `__module__` string round trip per element of every list parameter.
    // Subclasses (and `bool`, which is not an exact `int`) take the full chain.
    if value.is_exact_instance_of::<PyFloat>() {
        return value
            .extract::<f64>()
            .map(Value::Float64)
            .map_err(QueryConversionError::Python);
    }
    if value.is_exact_instance_of::<PyInt>() {
        return value
            .extract::<i64>()
            .map(Value::Int64)
            .map_err(|_| QueryConversionError::IntegerOverflow);
    }
    if let Ok(list) = value.cast_exact::<PyList>() {
        return convert_query_list(list, state);
    }
    if let Ok(dict) = value.cast_exact::<PyDict>() {
        return convert_query_dict(dict, state);
    }
    if value.is_instance_of::<pyo3::types::PyBool>() {
        if let Ok(boolean) = value.extract::<bool>() {
            return Ok(Value::Boolean(boolean));
        }
    }
    if is_numpy_ndarray(value) {
        return state.with_query_container(value.as_ptr() as usize, |state| {
            if let Some((decoded, levels)) = decode_numeric_ndarray(value) {
                state
                    .admit_nested_lists(levels)
                    .map_err(QueryConversionError::Limit)?;
                return Ok(decoded);
            }
            let as_list = value.call_method0("tolist")?;
            convert_query_value(&as_list, state)
        });
    }
    let numpy_type = numpy_scalar_type_name(value);
    // `np.bool_` is not a `PyBool`, and every other numpy scalar converts.
    if matches!(numpy_type.as_deref(), Some("bool_" | "bool")) {
        return value
            .extract::<bool>()
            .map(Value::Boolean)
            .map_err(QueryConversionError::Python);
    }
    if value.is_instance_of::<PyInt>()
        || numpy_type
            .as_deref()
            .is_some_and(|name| name.starts_with("int") || name.starts_with("uint"))
    {
        return value
            .extract::<i64>()
            .map(Value::Int64)
            .map_err(|_| QueryConversionError::IntegerOverflow);
    }
    if value.is_instance_of::<PyFloat>()
        || matches!(
            numpy_type.as_deref(),
            Some("float16" | "float32" | "float64")
        )
    {
        return value
            .extract::<f64>()
            .map(Value::Float64)
            .map_err(QueryConversionError::Python);
    }
    if let Ok(string) = value.extract::<String>() {
        return Ok(Value::String(string));
    }
    if let Ok(delta) = value.cast::<PyDelta>() {
        return delta_to_duration(delta)
            .map_err(|message| QueryConversionError::Python(PyValueError::new_err(message)));
    }
    if let Ok(datetime) = value.cast::<PyDateTime>() {
        // `pd.NaT` is a `datetime` subclass, so it arrives here and would fail
        // inside the conversion. It is pandas' missing value: bind it as NULL,
        // the same normalisation NaN and ±inf already get.
        //
        // Tested here rather than ahead of the integer arm because
        // `is_pandas_nat` costs a type-object fetch and a `String` allocation
        // per value, which every element of every list parameter paid for a
        // check only a datetime can pass (+32% on
        // `test_bench_param_list_conversion`, +17% on a 1 000-element
        // `IN $ids`). `PyDateTime_Check` is a subclass check, so NaT reaches
        // this arm and nothing else changes.
        if is_pandas_nat(value) {
            return Ok(Value::Null);
        }
        return datetime_to_utc_naive(datetime)
            .map(Value::Timestamp)
            .map_err(QueryConversionError::Python);
    }
    if value.is_instance_of::<PyDate>() {
        if let Ok(date) = value.extract::<chrono::NaiveDate>() {
            return Ok(Value::DateTime(date));
        }
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        return convert_query_dict(dict, state);
    }
    if let Ok(list) = value.cast::<PyList>() {
        return convert_query_list(list, state);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return state.with_query_container(value.as_ptr() as usize, |state| {
            tuple
                .iter()
                .enumerate()
                .map(|(index, child)| {
                    convert_query_value(&child, state)
                        .map_err(|error| QueryConversionError::AtIndex(index, Box::new(error)))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Value::List)
        });
    }
    // A numpy scalar's bare `__name__` ('complex128') is not a type anyone can
    // look up; qualify it with the module it came from.
    let type_name = numpy_type
        .map(|name| format!("numpy.{name}"))
        .unwrap_or_else(|| {
            value
                .get_type()
                .name()
                .and_then(|name| name.extract::<String>())
                .unwrap_or_else(|_| "<unknown>".to_string())
        });
    Err(QueryConversionError::Unsupported(type_name))
}

fn convert_query_dict(
    dict: &Bound<'_, PyDict>,
    state: &mut ConversionState,
) -> Result<Value, QueryConversionError> {
    state.with_query_container(dict.as_ptr() as usize, |state| {
        let mut pairs = Vec::with_capacity(dict.len());
        for (key, child) in dict.iter() {
            let key: String = key.extract()?;
            let converted = convert_query_value(&child, state)
                .map_err(|error| QueryConversionError::AtKey(key.clone(), Box::new(error)))?;
            pairs.push((kglite_core::datatypes::PropKey::from(key), converted));
        }
        Ok(Value::Map(kglite_core::datatypes::PropMap::from_pairs(
            pairs,
        )))
    })
}

fn convert_query_list(
    list: &Bound<'_, PyList>,
    state: &mut ConversionState,
) -> Result<Value, QueryConversionError> {
    state.with_query_container(list.as_ptr() as usize, |state| {
        list.iter()
            .enumerate()
            .map(|(index, child)| {
                convert_query_value(&child, state)
                    .map_err(|error| QueryConversionError::AtIndex(index, Box::new(error)))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Value::List)
    })
}

fn convert_value(value: &Bound<'_, PyAny>, state: &mut ConversionState) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    // Exact builtin types first, as in `convert_query_value`. An exact int
    // past i64 falls through: the chain below widens it to Float64.
    if value.is_exact_instance_of::<PyFloat>() {
        return value.extract::<f64>().map(Value::Float64);
    }
    if value.is_exact_instance_of::<PyInt>() {
        if let Ok(i) = value.extract::<i64>() {
            return Ok(Value::Int64(i));
        }
    }
    if let Ok(list) = value.cast_exact::<PyList>() {
        return convert_list(list, state);
    }
    if let Ok(dict) = value.cast_exact::<PyDict>() {
        return state.with_container(value.as_ptr() as usize, |state| convert_dict(dict, state));
    }
    // bool is an int subclass; size-one ndarray can extract as a scalar.
    if value.is_instance_of::<pyo3::types::PyBool>() {
        if let Ok(b) = value.extract::<bool>() {
            return Ok(Value::Boolean(b));
        }
    }
    if is_numpy_ndarray(value) {
        return state.with_container(value.as_ptr() as usize, |state| {
            if let Some((decoded, levels)) = decode_numeric_ndarray(value) {
                state.admit_nested_lists(levels)?;
                return Ok(decoded);
            }
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
    if let Ok(delta) = value.cast::<PyDelta>() {
        return delta_to_duration(delta).map_err(PyValueError::new_err);
    }
    // datetime is a date subclass: failure must not degrade to a date-only value.
    if let Ok(dt) = value.cast::<PyDateTime>() {
        return datetime_to_utc_naive(dt).map(Value::Timestamp);
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
        return convert_list(list, state);
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

fn convert_list(list: &Bound<'_, PyList>, state: &mut ConversionState) -> PyResult<Value> {
    state.with_container(list.as_ptr() as usize, |state| {
        list.iter()
            .map(|item| convert_value(&item, state))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::List)
    })
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

pub(crate) fn datetime_to_utc_naive(
    value: &Bound<'_, PyDateTime>,
) -> PyResult<chrono::NaiveDateTime> {
    if value.get_tzinfo().is_none() {
        return value.extract::<chrono::NaiveDateTime>();
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
}

/// A `datetime.timedelta` (or `pd.Timedelta`) as a `Value::Duration`. Its
/// whole seconds split into days and seconds that keep the delta's sign, the
/// components `duration({days, seconds})` builds, so `timedelta(hours=-1)`
/// equals `duration({hours: -1})`. A duration holds whole seconds; a delta
/// with a sub-second part is refused rather than truncated.
pub(crate) fn delta_to_duration(delta: &Bound<'_, PyDelta>) -> Result<Value, String> {
    // The limited API has no timedelta accessors; the attributes are public.
    let field = |name: &str| -> i64 {
        delta
            .getattr(name)
            .and_then(|value| value.extract())
            .unwrap_or(0)
    };
    if field("microseconds") != 0 || field("nanoseconds") != 0 {
        return Err(format!(
            "{} has a sub-second part; a duration holds whole seconds",
            delta
                .repr()
                .map_or_else(|_| "timedelta".to_string(), |r| r.to_string())
        ));
    }
    let total = field("days") * 86_400 + field("seconds");
    let days = i32::try_from(total / 86_400).map_err(|_| "timedelta days exceed the i32 range")?;
    Ok(Value::Duration {
        months: 0,
        days,
        seconds: total % 86_400,
    })
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

    fn eval<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyAny> {
        let code = std::ffi::CString::new(code).unwrap();
        py.eval(&code, None, None).unwrap()
    }

    /// `None` when the embedded interpreter has no numpy (the Rust CI job);
    /// the Python suite covers the same paths there.
    fn numpy(py: Python<'_>) -> Option<Bound<'_, PyModule>> {
        let module = py.import("numpy").ok();
        if module.is_none() {
            eprintln!("numpy not importable in the embedded interpreter; skipping");
        }
        module
    }

    fn query(value: &Bound<'_, PyAny>) -> Result<Value, QueryConversionError> {
        convert_query_value(value, &mut ConversionState::default())
    }

    fn ingest(value: &Bound<'_, PyAny>) -> Value {
        convert_value(value, &mut ConversionState::default()).unwrap()
    }

    #[test]
    fn exact_builtins_convert_on_the_fast_path() {
        Python::initialize();
        Python::attach(|py| {
            let value = eval(py, "[1.5, -2, [0.25], {'k': 3}]");
            let expected = Value::List(vec![
                Value::Float64(1.5),
                Value::Int64(-2),
                Value::List(vec![Value::Float64(0.25)]),
                Value::Map(kglite_core::datatypes::PropMap::from_pairs(vec![(
                    kglite_core::datatypes::PropKey::from("k".to_string()),
                    Value::Int64(3),
                )])),
            ]);
            assert!(query(&value).ok() == Some(expected.clone()));
            assert_eq!(ingest(&value), expected);
        });
    }

    #[test]
    fn subclasses_bool_and_wide_ints_keep_the_generic_chain() {
        Python::initialize();
        Python::attach(|py| {
            let float_subclass = eval(py, "type('F', (float,), {})(2.5)");
            let int_subclass = eval(py, "type('I', (int,), {})(7)");
            let list_subclass = eval(py, "type('L', (list,), {})([1])");
            for (value, expected) in [
                (float_subclass, Value::Float64(2.5)),
                (int_subclass, Value::Int64(7)),
                (list_subclass, Value::List(vec![Value::Int64(1)])),
                (eval(py, "True"), Value::Boolean(true)),
            ] {
                assert!(query(&value).ok() == Some(expected.clone()));
                assert_eq!(ingest(&value), expected);
            }
            let wide = eval(py, "2**63");
            assert!(matches!(
                query(&wide),
                Err(QueryConversionError::IntegerOverflow)
            ));
            assert_eq!(ingest(&wide), Value::Float64(9_223_372_036_854_775_808.0));
            let aware = eval(
                py,
                "__import__('datetime').datetime(2024, 1, 1, 12, tzinfo=__import__('datetime').timezone(__import__('datetime').timedelta(hours=2)))",
            );
            let noon_utc = chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(10, 0, 0)
                .unwrap();
            assert!(query(&aware).ok() == Some(Value::Timestamp(noon_utc)));
            assert_eq!(ingest(&aware), Value::Timestamp(noon_utc));
        });
    }

    #[test]
    fn half_precision_widens_as_numpy_does() {
        assert_eq!(f16_bits_to_f64(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f64(0xc000), -2.0);
        assert_eq!(f16_bits_to_f64(0x7bff), 65504.0);
        assert_eq!(f16_bits_to_f64(0x0001), (-24f64).exp2());
        assert_eq!(f16_bits_to_f64(0x83ff), -1023.0 * (-24f64).exp2());
        assert_eq!(f16_bits_to_f64(0x8000).to_bits(), (-0.0f64).to_bits());
        assert_eq!(f16_bits_to_f64(0x7c00), f64::INFINITY);
        assert_eq!(f16_bits_to_f64(0xfc00), f64::NEG_INFINITY);
        assert_eq!(f16_bits_to_f64(0x7e00).to_bits(), 0x7ff8_0000_0000_0000);
    }

    #[test]
    fn numeric_ndarrays_decode_identically_to_tolist() {
        Python::initialize();
        Python::attach(|py| {
            let Some(np) = numpy(py) else { return };
            let locals = pyo3::types::PyDict::new(py);
            locals.set_item("np", &np).unwrap();
            let arrays = [
                "np.array([[1.5, -0.1], [3e38, float('nan')]], dtype=np.float32)",
                "np.array([0.1, -2.5, float('inf'), 6e-8, -0.0], dtype=np.float16)",
                "np.array([[0.1, 2.0**-1074], [-1e308, 0.0]])",
                "np.array([-128, 127], dtype=np.int8)",
                "np.array([[-32768], [32767]], dtype=np.int16)",
                "np.array([-2**31, 2**31 - 1], dtype=np.int32)",
                "np.array([[-2**63, 2**63 - 1]], dtype=np.int64)",
                "np.array([0, 255], dtype=np.uint8)",
                "np.array([0, 65535], dtype=np.uint16)",
                "np.array([[0, 2**32 - 1]], dtype=np.uint32)",
                "np.zeros((0, 3), dtype=np.float32)",
                "np.zeros((2, 0), dtype=np.float64)",
                "np.zeros(0, dtype=np.int64)",
                // Non-contiguous and Fortran-order views decode in C order.
                "np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]",
                "np.asfortranarray(np.arange(6, dtype=np.int32).reshape(2, 3))",
                "np.arange(10, dtype=np.float64)[::-3]",
            ];
            for code in arrays {
                let code_c = std::ffi::CString::new(code).unwrap();
                let array = py.eval(&code_c, None, Some(&locals)).unwrap();
                assert!(decode_numeric_ndarray(&array).is_some(), "{code}");
                let via_tolist = ingest(&array.call_method0("tolist").unwrap());
                let fast = ingest(&array);
                assert_eq!(format!("{fast:?}"), format!("{via_tolist:?}"), "{code}");
                let queried = query(&array).ok().expect(code);
                assert_eq!(format!("{queried:?}"), format!("{via_tolist:?}"), "{code}");
            }
        });
    }

    #[test]
    fn f32_rows_decode_ndarrays_exactly_as_float_extraction_does() {
        Python::initialize();
        Python::attach(|py| {
            let Some(np) = numpy(py) else { return };
            let locals = pyo3::types::PyDict::new(py);
            locals.set_item("np", &np).unwrap();
            let run = |code: &str| {
                let code = std::ffi::CString::new(code).unwrap();
                py.eval(&code, None, Some(&locals)).unwrap()
            };
            let mut rows = F32Rows::default();
            for code in [
                "np.array([1.5, -0.1, 3e38, float('nan'), float('inf')], dtype=np.float32)",
                "np.array([0.1, 1e-40, -1e308, 2.0**-1074, 1/3])",
                "np.array([0.1, -2.5, 6e-8, -0.0], dtype=np.float16)",
                "np.array([-128, 127], dtype=np.int8)",
                "np.array([-32768, 32767], dtype=np.int16)",
                "np.array([-2**31, 2**31 - 1], dtype=np.int32)",
                "np.array([-2**63, 2**63 - 1, 2**53 + 1], dtype=np.int64)",
                "np.array([0, 255], dtype=np.uint8)",
                "np.array([0, 65535], dtype=np.uint16)",
                "np.array([0, 2**32 - 1], dtype=np.uint32)",
                "np.zeros(0, dtype=np.float32)",
                "np.arange(12, dtype=np.float64)[::-3]",
                // Rows of a 2-D array are 1-D views: the common numpy-rows shape.
                "np.arange(6, dtype=np.float32).reshape(2, 3)[1]",
                // Not decoded — extracted as before.
                "np.array([True, False])",
                "np.array([1, 2], dtype=np.uint64)",
                "np.array([1.5, 2.5], dtype='>f4')",
                "[1.5, 2, -0.25]",
            ] {
                let value = run(code);
                let expected: Vec<f32> = value.extract().expect(code);
                let decoded = rows.extract(&value).expect(code);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&decoded), bits(&expected), "{code}");
            }
            for code in [
                "np.arange(4, dtype=np.float32).reshape(2, 2)",
                "np.array(['a'], dtype=object)",
            ] {
                let value = run(code);
                let expected = value.extract::<Vec<f32>>().unwrap_err().to_string();
                let error = rows.extract(&value).unwrap_err().to_string();
                assert_eq!(error, expected, "{code}");
            }
        });
    }

    #[test]
    fn other_ndarrays_and_numpy_scalars_keep_the_tolist_route() {
        Python::initialize();
        Python::attach(|py| {
            let Some(np) = numpy(py) else { return };
            let locals = pyo3::types::PyDict::new(py);
            locals.set_item("np", &np).unwrap();
            let run = |code: &str| {
                let code = std::ffi::CString::new(code).unwrap();
                py.eval(&code, None, Some(&locals)).unwrap()
            };
            for code in [
                "np.arange(8, dtype=np.float32).reshape(2, 2, 2)",
                "np.array([1, 'a'], dtype=object)",
                "np.array([True, False])",
                "np.array([1, 2], dtype=np.uint64)",
                "np.array([1.5], dtype='>f4')",
                "np.float32(1.5).reshape(())",
            ] {
                let array = run(code);
                assert!(decode_numeric_ndarray(&array).is_none(), "{code}");
            }
            assert!(
                query(&run("np.arange(8, dtype=np.float32).reshape(2, 2, 2)")).ok()
                    == Some(ingest(&run(
                        "np.arange(8, dtype=np.float32).reshape(2, 2, 2).tolist()"
                    )))
            );
            assert!(
                query(&run("np.array([1.5], dtype='>f4')")).ok()
                    == Some(Value::List(vec![Value::Float64(1.5)]))
            );
            assert!(matches!(
                query(&run("np.array([2**63], dtype=np.uint64)")),
                Err(QueryConversionError::AtIndex(0, inner))
                    if matches!(*inner, QueryConversionError::IntegerOverflow)
            ));
            assert!(query(&run("np.bool_(True)")).ok() == Some(Value::Boolean(true)));
            assert!(query(&run("np.int64(-5)")).ok() == Some(Value::Int64(-5)));
            assert!(query(&run("np.float32(0.5)")).ok() == Some(Value::Float64(0.5)));
            assert!(matches!(
                query(&run("np.uint64(2**63)")),
                Err(QueryConversionError::IntegerOverflow)
            ));
            if let Ok(pandas) = py.import("pandas") {
                let nat = pandas.getattr("NaT").unwrap();
                assert!(query(&nat).ok() == Some(Value::Null));
            }
        });
    }

    #[test]
    fn decoded_ndarray_spends_the_depth_budget_of_its_tolist_route() {
        Python::initialize();
        Python::attach(|py| {
            let Some(np) = numpy(py) else { return };
            let locals = pyo3::types::PyDict::new(py);
            locals.set_item("np", &np).unwrap();
            for code in [
                "np.zeros((2, 3), dtype=np.float32)",
                "np.zeros((0, 3), dtype=np.float32)",
                "np.zeros(3, dtype=np.int64)",
            ] {
                let code_c = std::ffi::CString::new(code).unwrap();
                let array = py.eval(&code_c, None, Some(&locals)).unwrap();
                // `[array.tolist()]` nests exactly the containers the
                // ndarray-then-tolist route enters.
                let reference = PyList::new(py, [array.call_method0("tolist").unwrap()]).unwrap();
                for used in MAX_CONTAINER_DEPTH - 4..=MAX_CONTAINER_DEPTH {
                    let outcome = |value: &Bound<'_, PyAny>| {
                        let mut state = ConversionState {
                            active: vec![usize::MAX; used],
                        };
                        let ingest_ok = convert_value(value, &mut state).is_ok();
                        let query_ok = convert_query_value(value, &mut state).is_ok();
                        assert_eq!(state.active.len(), used);
                        (ingest_ok, query_ok)
                    };
                    assert_eq!(
                        outcome(&array),
                        outcome(reference.as_any()),
                        "{code} at depth {used}"
                    );
                }
            }
        });
    }
}
