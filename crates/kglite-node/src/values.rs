//! JS <-> engine `Value` conversion over raw Node-API calls.
//!
//! Raw calls rather than napi's typed wrappers because result conversion is the
//! one part of a query that must run on the JS thread: per-row handle scopes
//! and cached column-key handles are what keep a 100k-row result cheap. Every
//! function here must run on the JS thread of the env it is handed.

use std::collections::HashMap;
use std::ffi::c_char;
use std::ptr;

use kglite::api::{NodeValue, PropMap, RelValue, Value};
use napi::bindgen_prelude::{validate_type_tag, ToNapiValue, TypeTag};
use napi::sys;

use crate::classes::{
    Duration, KgFloat, LocalDate, LocalDateTime, LocalTime, Point, Time, ZonedDateTime,
};
use crate::errors::{JsErr, JsRes};

/// Deepest list/map nesting converted into JS. The engine refuses to build a
/// value past its own ceiling (`MAX_VALUE_DEPTH`, 256), so any value a query
/// returns fits.
pub const MAX_DEPTH: usize = 1024;

/// Deepest nesting accepted in a parameter. Conversion recurses on the JS
/// thread, whose stack is the host's; a debug build overflows it (SIGSEGV, not
/// an error) a little past 600 levels, so the ceiling sits well under that.
pub const MAX_PARAM_DEPTH: usize = 256;

/// Largest integer a JS number holds exactly.
const MAX_SAFE: i64 = (1 << 53) - 1;

/// `integers` option: how `Int64` results surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerMode {
    /// `number` when exact, `bigint` otherwise.
    Safe,
    /// Always `bigint`.
    BigInt,
}

fn ck(status: sys::napi_status, what: &str) -> JsRes<()> {
    if status == sys::Status::napi_ok {
        Ok(())
    } else {
        Err(JsErr::internal(format!("{what} failed (status {status})")))
    }
}

fn nv(r: napi::Result<sys::napi_value>) -> JsRes<sys::napi_value> {
    r.map_err(JsErr::from)
}

// ---------------------------------------------------------------- engine -> JS

/// Converts engine values to JS values on one env.
pub struct ToJs {
    env: sys::napi_env,
    ints: IntegerMode,
}

impl ToJs {
    pub fn new(env: sys::napi_env, ints: IntegerMode) -> Self {
        Self { env, ints }
    }

    pub fn string(&self, s: &str) -> JsRes<sys::napi_value> {
        let mut out = ptr::null_mut();
        ck(
            unsafe {
                sys::napi_create_string_utf8(
                    self.env,
                    s.as_ptr().cast::<c_char>(),
                    s.len() as isize,
                    &mut out,
                )
            },
            "create string",
        )?;
        Ok(out)
    }

    pub fn number(&self, n: f64) -> JsRes<sys::napi_value> {
        let mut out = ptr::null_mut();
        ck(
            unsafe { sys::napi_create_double(self.env, n, &mut out) },
            "create number",
        )?;
        Ok(out)
    }

    pub fn undefined(&self) -> JsRes<sys::napi_value> {
        let mut out = ptr::null_mut();
        ck(
            unsafe { sys::napi_get_undefined(self.env, &mut out) },
            "get undefined",
        )?;
        Ok(out)
    }

    pub fn boolean(&self, b: bool) -> JsRes<sys::napi_value> {
        let mut out = ptr::null_mut();
        ck(
            unsafe { sys::napi_get_boolean(self.env, b, &mut out) },
            "get boolean",
        )?;
        Ok(out)
    }

    pub fn null(&self) -> JsRes<sys::napi_value> {
        let mut out = ptr::null_mut();
        ck(
            unsafe { sys::napi_get_null(self.env, &mut out) },
            "get null",
        )?;
        Ok(out)
    }

    pub fn object(&self) -> JsRes<sys::napi_value> {
        let mut out = ptr::null_mut();
        ck(
            unsafe { sys::napi_create_object(self.env, &mut out) },
            "create object",
        )?;
        Ok(out)
    }

    pub fn array(&self, len: usize) -> JsRes<sys::napi_value> {
        let mut out = ptr::null_mut();
        ck(
            unsafe { sys::napi_create_array_with_length(self.env, len, &mut out) },
            "create array",
        )?;
        Ok(out)
    }

    pub fn push(&self, array: sys::napi_value, index: usize, v: sys::napi_value) -> JsRes<()> {
        ck(
            unsafe { sys::napi_set_element(self.env, array, index as u32, v) },
            "set element",
        )
    }

    /// Own data property `key` on `obj`. `__proto__` goes through
    /// `napi_define_properties` because a plain set would replace the prototype.
    pub fn set(&self, obj: sys::napi_value, key: &str, v: sys::napi_value) -> JsRes<()> {
        if key == "__proto__" {
            let desc = sys::napi_property_descriptor {
                utf8name: c"__proto__".as_ptr(),
                name: ptr::null_mut(),
                method: None,
                getter: None,
                setter: None,
                value: v,
                attributes: sys::PropertyAttributes::default,
                data: ptr::null_mut(),
            };
            return ck(
                unsafe { sys::napi_define_properties(self.env, obj, 1, &desc) },
                "define property",
            );
        }
        let k = self.string(key)?;
        ck(
            unsafe { sys::napi_set_property(self.env, obj, k, v) },
            "set property",
        )
    }

    /// `set` with a key handle made once and reused for every row.
    pub fn set_keyed(
        &self,
        obj: sys::napi_value,
        key: &CachedKey,
        v: sys::napi_value,
    ) -> JsRes<()> {
        if key.proto {
            return self.set(obj, "__proto__", v);
        }
        ck(
            unsafe { sys::napi_set_property(self.env, obj, key.handle, v) },
            "set property",
        )
    }

    pub fn cached_key(&self, key: &str) -> JsRes<CachedKey> {
        Ok(CachedKey {
            handle: self.string(key)?,
            proto: key == "__proto__",
        })
    }

    pub fn scope<T>(&self, f: impl FnOnce() -> JsRes<T>) -> JsRes<T> {
        let mut scope = ptr::null_mut();
        ck(
            unsafe { sys::napi_open_handle_scope(self.env, &mut scope) },
            "open scope",
        )?;
        let r = f();
        ck(
            unsafe { sys::napi_close_handle_scope(self.env, scope) },
            "close scope",
        )?;
        r
    }

    pub fn int(&self, v: i64) -> JsRes<sys::napi_value> {
        let mut out = ptr::null_mut();
        if self.ints == IntegerMode::Safe && (-MAX_SAFE..=MAX_SAFE).contains(&v) {
            ck(
                unsafe { sys::napi_create_int64(self.env, v, &mut out) },
                "create int",
            )?;
        } else {
            ck(
                unsafe { sys::napi_create_bigint_int64(self.env, v, &mut out) },
                "create bigint",
            )?;
        }
        Ok(out)
    }

    fn props(&self, props: &PropMap, depth: usize) -> JsRes<sys::napi_value> {
        let obj = self.object()?;
        for (k, v) in props.iter() {
            let js = self.value(v, depth + 1)?;
            self.set(obj, k, js)?;
        }
        Ok(obj)
    }

    fn node(&self, n: &NodeValue, depth: usize) -> JsRes<sys::napi_value> {
        let obj = self.object()?;
        self.set(obj, "id", self.number(f64::from(n.id))?)?;
        let labels = self.array(n.labels.len())?;
        for (i, l) in n.labels.iter().enumerate() {
            self.push(labels, i, self.string(l)?)?;
        }
        self.set(obj, "labels", labels)?;
        self.set(obj, "properties", self.props(&n.properties, depth)?)?;
        Ok(obj)
    }

    fn rel(&self, r: &RelValue, depth: usize) -> JsRes<sys::napi_value> {
        let obj = self.object()?;
        self.set(obj, "id", self.number(f64::from(r.id))?)?;
        self.set(obj, "type", self.string(&r.rel_type)?)?;
        self.set(obj, "startId", self.number(f64::from(r.start_id))?)?;
        self.set(obj, "endId", self.number(f64::from(r.end_id))?)?;
        self.set(obj, "properties", self.props(&r.properties, depth)?)?;
        Ok(obj)
    }

    fn class<T: ToNapiValue>(&self, v: T) -> JsRes<sys::napi_value> {
        nv(unsafe { T::to_napi_value(self.env, v) })
    }

    pub fn value(&self, v: &Value, depth: usize) -> JsRes<sys::napi_value> {
        if depth > MAX_DEPTH {
            return Err(JsErr::internal(
                "result nests deeper than the supported depth",
            ));
        }
        match v {
            Value::Null => self.null(),
            Value::Boolean(b) => self.boolean(*b),
            Value::Int64(i) => self.int(*i),
            Value::Float64(f) => self.number(*f),
            Value::String(s) => self.string(s),
            Value::UniqueId(n) | Value::NodeRef(n) => self.number(f64::from(*n)),
            Value::DateTime(d) => self.class(LocalDate::from_naive(*d)),
            Value::Timestamp(t) => self.class(LocalDateTime::from_naive(*t)),
            Value::ZonedDateTime {
                utc,
                offset_secs,
                zone,
            } => self.class(ZonedDateTime::from_parts(*utc, *offset_secs, *zone)),
            Value::LocalTime(n) => self.class(LocalTime::from_nanos(*n)),
            Value::Time { nanos, offset_secs } => {
                self.class(Time::from_parts(*nanos, *offset_secs))
            }
            Value::Duration {
                months,
                days,
                seconds,
                nanos,
            } => self.class(Duration {
                months: *months,
                days: *days,
                seconds: *seconds,
                nanoseconds: *nanos,
            }),
            Value::Point { lat, lon } => self.class(Point {
                latitude: *lat,
                longitude: *lon,
            }),
            Value::List(items) => {
                let arr = self.array(items.len())?;
                for (i, item) in items.iter().enumerate() {
                    self.push(arr, i, self.value(item, depth + 1)?)?;
                }
                Ok(arr)
            }
            Value::Map(m) => self.props(m, depth),
            Value::Node(n) => self.node(n, depth),
            Value::Relationship(r) => self.rel(r, depth),
            Value::Path(p) => {
                let obj = self.object()?;
                let nodes = self.array(p.nodes.len())?;
                for (i, n) in p.nodes.iter().enumerate() {
                    self.push(nodes, i, self.node(n, depth + 1)?)?;
                }
                let rels = self.array(p.rels.len())?;
                for (i, r) in p.rels.iter().enumerate() {
                    self.push(rels, i, self.rel(r, depth + 1)?)?;
                }
                self.set(obj, "nodes", nodes)?;
                self.set(obj, "relationships", rels)?;
                Ok(obj)
            }
        }
    }
}

/// A column-name string handle created once and set on every row.
pub struct CachedKey {
    handle: sys::napi_value,
    proto: bool,
}

// ---------------------------------------------------------------- JS -> engine

/// Converts JS parameter values to engine values.
pub struct FromJs {
    env: sys::napi_env,
    object_proto: Option<sys::napi_value>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Undefined,
    Null,
    Boolean,
    Number,
    String,
    Symbol,
    Object,
    Function,
    BigInt,
    External,
}

impl FromJs {
    pub fn new(env: sys::napi_env) -> Self {
        Self {
            env,
            object_proto: None,
        }
    }

    pub fn kind(&self, v: sys::napi_value) -> JsRes<KindOf> {
        let mut t = 0;
        ck(unsafe { sys::napi_typeof(self.env, v, &mut t) }, "typeof")?;
        Ok(KindOf(match t {
            sys::ValueType::napi_undefined => Kind::Undefined,
            sys::ValueType::napi_null => Kind::Null,
            sys::ValueType::napi_boolean => Kind::Boolean,
            sys::ValueType::napi_number => Kind::Number,
            sys::ValueType::napi_string => Kind::String,
            sys::ValueType::napi_symbol => Kind::Symbol,
            sys::ValueType::napi_object => Kind::Object,
            sys::ValueType::napi_function => Kind::Function,
            sys::ValueType::napi_bigint => Kind::BigInt,
            _ => Kind::External,
        }))
    }

    pub fn is_nullish(&self, v: sys::napi_value) -> JsRes<bool> {
        Ok(matches!(self.kind(v)?.0, Kind::Undefined | Kind::Null))
    }

    pub fn is_plain_object(&mut self, v: sys::napi_value) -> JsRes<bool> {
        if self.kind(v)?.0 != Kind::Object
            || self.flag(v, sys::napi_is_array)?
            || self.flag(v, sys::napi_is_date)?
        {
            return Ok(false);
        }
        let mut proto = ptr::null_mut();
        ck(
            unsafe { sys::napi_get_prototype(self.env, v, &mut proto) },
            "get prototype",
        )?;
        if self.kind(proto)?.0 == Kind::Null {
            return Ok(true);
        }
        let object_proto = self.object_proto()?;
        let mut same = false;
        ck(
            unsafe { sys::napi_strict_equals(self.env, proto, object_proto, &mut same) },
            "strict equals",
        )?;
        Ok(same)
    }

    fn object_proto(&mut self) -> JsRes<sys::napi_value> {
        if let Some(p) = self.object_proto {
            return Ok(p);
        }
        let mut global = ptr::null_mut();
        ck(
            unsafe { sys::napi_get_global(self.env, &mut global) },
            "get global",
        )?;
        let mut ctor = ptr::null_mut();
        ck(
            unsafe {
                sys::napi_get_named_property(self.env, global, c"Object".as_ptr(), &mut ctor)
            },
            "get Object",
        )?;
        let mut proto = ptr::null_mut();
        ck(
            unsafe {
                sys::napi_get_named_property(self.env, ctor, c"prototype".as_ptr(), &mut proto)
            },
            "get Object.prototype",
        )?;
        self.object_proto = Some(proto);
        Ok(proto)
    }

    fn flag(
        &self,
        v: sys::napi_value,
        f: unsafe fn(sys::napi_env, sys::napi_value, *mut bool) -> sys::napi_status,
    ) -> JsRes<bool> {
        let mut out = false;
        ck(unsafe { f(self.env, v, &mut out) }, "type check")?;
        Ok(out)
    }

    pub fn get_string(&self, v: sys::napi_value) -> JsRes<String> {
        let mut len = 0usize;
        ck(
            unsafe { sys::napi_get_value_string_utf8(self.env, v, ptr::null_mut(), 0, &mut len) },
            "string length",
        )?;
        let mut buf = vec![0u8; len + 1];
        let mut written = 0usize;
        ck(
            unsafe {
                sys::napi_get_value_string_utf8(
                    self.env,
                    v,
                    buf.as_mut_ptr().cast::<c_char>(),
                    buf.len(),
                    &mut written,
                )
            },
            "string value",
        )?;
        buf.truncate(written);
        String::from_utf8(buf).map_err(|_| JsErr::internal("string was not valid UTF-8"))
    }

    pub fn get_bool(&self, v: sys::napi_value) -> JsRes<bool> {
        let mut out = false;
        ck(
            unsafe { sys::napi_get_value_bool(self.env, v, &mut out) },
            "bool value",
        )?;
        Ok(out)
    }

    pub fn get_f64(&self, v: sys::napi_value) -> JsRes<f64> {
        let mut out = 0.0;
        ck(
            unsafe { sys::napi_get_value_double(self.env, v, &mut out) },
            "number value",
        )?;
        Ok(out)
    }

    pub fn get_property(&self, obj: sys::napi_value, key: &str) -> JsRes<sys::napi_value> {
        let k = ToJs::new(self.env, IntegerMode::Safe).string(key)?;
        let mut out = ptr::null_mut();
        ck(
            unsafe { sys::napi_get_property(self.env, obj, k, &mut out) },
            "get property",
        )?;
        Ok(out)
    }

    /// Own enumerable string keys, in insertion order.
    pub fn own_keys(&self, obj: sys::napi_value) -> JsRes<Vec<String>> {
        let mut names = ptr::null_mut();
        ck(
            unsafe {
                sys::napi_get_all_property_names(
                    self.env,
                    obj,
                    sys::KeyCollectionMode::own_only,
                    sys::KeyFilter::enumerable | sys::KeyFilter::skip_symbols,
                    sys::KeyConversion::numbers_to_strings,
                    &mut names,
                )
            },
            "property names",
        )?;
        let mut len = 0u32;
        ck(
            unsafe { sys::napi_get_array_length(self.env, names, &mut len) },
            "array length",
        )?;
        let mut keys = Vec::with_capacity(len as usize);
        for i in 0..len {
            let mut k = ptr::null_mut();
            ck(
                unsafe { sys::napi_get_element(self.env, names, i, &mut k) },
                "get element",
            )?;
            keys.push(self.get_string(k)?);
        }
        Ok(keys)
    }

    /// Convert one parameter value. `path` names it in error messages.
    pub fn value(&mut self, v: sys::napi_value, path: &str, depth: usize) -> JsRes<Value> {
        if depth > MAX_PARAM_DEPTH {
            return Err(JsErr::arg(format!(
                "{path}: nests deeper than {MAX_PARAM_DEPTH} levels"
            )));
        }
        match self.kind(v)?.0 {
            Kind::Undefined | Kind::Null => Ok(Value::Null),
            Kind::Boolean => {
                let mut b = false;
                ck(
                    unsafe { sys::napi_get_value_bool(self.env, v, &mut b) },
                    "bool value",
                )?;
                Ok(Value::Boolean(b))
            }
            Kind::Number => {
                let n = self.get_f64(v)?;
                if n.fract() == 0.0
                    && n.abs() <= MAX_SAFE as f64
                    && !(n == 0.0 && n.is_sign_negative())
                {
                    Ok(Value::Int64(n as i64))
                } else {
                    Ok(Value::Float64(n))
                }
            }
            Kind::BigInt => {
                let mut out = 0i64;
                let mut lossless = false;
                ck(
                    unsafe {
                        sys::napi_get_value_bigint_int64(self.env, v, &mut out, &mut lossless)
                    },
                    "bigint value",
                )?;
                if lossless {
                    Ok(Value::Int64(out))
                } else {
                    Err(JsErr::arg(format!(
                        "{path}: BigInt is outside the 64-bit signed integer range"
                    )))
                }
            }
            Kind::String => Ok(Value::String(self.get_string(v)?)),
            Kind::Symbol | Kind::Function | Kind::External => {
                Err(JsErr::arg(format!("{path}: unsupported parameter type")))
            }
            Kind::Object => self.object_value(v, path, depth),
        }
    }

    fn object_value(&mut self, v: sys::napi_value, path: &str, depth: usize) -> JsRes<Value> {
        if self.flag(v, sys::napi_is_array)? {
            let mut len = 0u32;
            ck(
                unsafe { sys::napi_get_array_length(self.env, v, &mut len) },
                "array length",
            )?;
            let mut items = Vec::with_capacity(len as usize);
            for i in 0..len {
                let mut e = ptr::null_mut();
                ck(
                    unsafe { sys::napi_get_element(self.env, v, i, &mut e) },
                    "get element",
                )?;
                items.push(self.value(e, &format!("{path}[{i}]"), depth + 1)?);
            }
            return Ok(Value::List(items));
        }
        if self.flag(v, sys::napi_is_date)? {
            let mut ms = 0.0;
            ck(
                unsafe { sys::napi_get_date_value(self.env, v, &mut ms) },
                "date value",
            )?;
            // A JS `Date` is an instant: it binds as a zoned datetime at UTC.
            return chrono::DateTime::from_timestamp_millis(ms as i64)
                .filter(|_| ms.is_finite())
                .map(|t| Value::zoned(t.naive_utc(), 0))
                .ok_or_else(|| JsErr::arg(format!("{path}: invalid Date")));
        }
        if self.flag(v, sys::napi_is_typedarray)?
            || self.flag(v, sys::napi_is_arraybuffer)?
            || self.flag(v, sys::napi_is_dataview)?
        {
            return Err(JsErr::arg(format!(
                "{path}: binary values are not supported"
            )));
        }
        if let Some(value) = self.class_value(v) {
            return Ok(value);
        }
        if !self.is_plain_object(v)? {
            return Err(JsErr::arg(format!(
                "{path}: only plain objects, arrays and the kglite value classes are supported"
            )));
        }
        let mut pairs: Vec<(String, Value)> = Vec::new();
        for key in self.own_keys(v)? {
            let child = self.get_property(v, &key)?;
            if self.kind(child)?.0 == Kind::Undefined {
                continue;
            }
            let value = self.value(child, &format!("{path}.{key}"), depth + 1)?;
            pairs.push((key, value));
        }
        Ok(Value::Map(
            pairs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect(),
        ))
    }

    /// The engine value behind one of the exported value classes, if `v` is one.
    fn class_value(&self, v: sys::napi_value) -> Option<Value> {
        if let Some(d) = self.unwrap_class::<LocalDate>(v) {
            return Some(Value::DateTime(d.to_naive()));
        }
        if let Some(t) = self.unwrap_class::<LocalDateTime>(v) {
            return Some(Value::Timestamp(t.to_naive()));
        }
        if let Some(t) = self.unwrap_class::<ZonedDateTime>(v) {
            let (utc, offset_secs, zone) = t.to_parts();
            return Some(Value::ZonedDateTime {
                utc,
                offset_secs,
                zone,
            });
        }
        if let Some(t) = self.unwrap_class::<LocalTime>(v) {
            return Some(Value::LocalTime(t.to_nanos()));
        }
        if let Some(t) = self.unwrap_class::<Time>(v) {
            return Some(Value::Time {
                nanos: t.to_nanos(),
                offset_secs: t.offset_seconds,
            });
        }
        if let Some(d) = self.unwrap_class::<Duration>(v) {
            return Some(Value::Duration {
                months: d.months,
                days: d.days,
                seconds: d.seconds,
                nanos: d.nanoseconds,
            });
        }
        if let Some(p) = self.unwrap_class::<Point>(v) {
            return Some(Value::Point {
                lat: p.latitude,
                lon: p.longitude,
            });
        }
        self.unwrap_class::<KgFloat>(v)
            .map(|f| Value::Float64(f.value))
    }

    /// The native struct behind a napi class instance, `None` for any other value.
    ///
    /// napi's own `&T` conversion only works inside generated argument glue, and
    /// params arrive as raw objects, so unwrap and type-check by hand. The tag
    /// check rejects an object that merely borrows the class's prototype.
    fn unwrap_class<T: TypeTag>(&self, v: sys::napi_value) -> Option<&T> {
        let mut data = ptr::null_mut();
        if unsafe { sys::napi_unwrap(self.env, v, &mut data) } != sys::Status::napi_ok
            || data.is_null()
        {
            return None;
        }
        // SAFETY: `env` and `v` are valid for the duration of this JS-thread call.
        unsafe { validate_type_tag(self.env, v, &T::type_tag(), std::any::type_name::<T>()) }
            .ok()?;
        // SAFETY: the type tag proves `data` is the `T` napi wrapped into `v`; the
        // reference is used and dropped before control returns to JS.
        Some(unsafe { &*data.cast::<T>() })
    }

    /// `params` argument: nullish or a plain object of named values.
    pub fn params(&mut self, v: Option<sys::napi_value>) -> JsRes<HashMap<String, Value>> {
        let mut out = HashMap::new();
        let Some(v) = v else { return Ok(out) };
        if self.is_nullish(v)? {
            return Ok(out);
        }
        if !self.is_plain_object(v)? {
            return Err(JsErr::arg("params must be a plain object of named values"));
        }
        for key in self.own_keys(v)? {
            let child = self.get_property(v, &key)?;
            if self.kind(child)?.0 == Kind::Undefined {
                continue;
            }
            let value = self.value(child, &format!("${key}"), 0)?;
            out.insert(key, value);
        }
        Ok(out)
    }
}

/// Opaque result of [`FromJs::kind`]; callers only compare through helpers.
pub struct KindOf(Kind);

impl KindOf {
    pub fn is_string(&self) -> bool {
        self.0 == Kind::String
    }
    pub fn is_number(&self) -> bool {
        self.0 == Kind::Number
    }
    pub fn is_boolean(&self) -> bool {
        self.0 == Kind::Boolean
    }
    pub fn is_function(&self) -> bool {
        self.0 == Kind::Function
    }
    pub fn is_object(&self) -> bool {
        self.0 == Kind::Object
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_integer_bounds() {
        assert_eq!(MAX_SAFE, 9_007_199_254_740_991);
    }
}
