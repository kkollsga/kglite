//! Reading one `files` entry's knobs out of its `extra` map.
//!
//! Every format's knobs arrive as untyped JSON, and every format needs the
//! same sentence when one is written with the wrong shape: which entry, which
//! key, what it must be, and what was actually there. These live here so a
//! second and third reader do not each invent their own phrasing for it.

use indexmap::IndexMap;

pub type Extra = IndexMap<String, serde_json::Value>;

/// What a value is, for an error that says what was written instead.
pub fn json_kind(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "a list",
        serde_json::Value::Object(_) => "an object",
    }
}

pub fn get_string(name: &str, extra: &Extra, key: &str) -> Result<Option<String>, String> {
    match extra.get(key) {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(v) => Err(format!(
            "files '{name}': '{key}' must be a string, but it is {}.",
            json_kind(v)
        )),
    }
}

pub fn get_bool(name: &str, extra: &Extra, key: &str) -> Result<Option<bool>, String> {
    match extra.get(key) {
        None => Ok(None),
        Some(serde_json::Value::Bool(b)) => Ok(Some(*b)),
        Some(v) => Err(format!(
            "files '{name}': '{key}' must be true or false, but it is {}.",
            json_kind(v)
        )),
    }
}

/// A non-negative whole number. `unit` names what is being counted, so the
/// error reads as a sentence about the file rather than about JSON.
pub fn get_usize(
    name: &str,
    extra: &Extra,
    key: &str,
    unit: &str,
) -> Result<Option<usize>, String> {
    match extra.get(key) {
        None => Ok(None),
        Some(serde_json::Value::Number(n)) => match n.as_u64() {
            Some(n) => Ok(Some(n as usize)),
            None => Err(format!(
                "files '{name}': '{key}' must be a whole number of {unit} that is not negative, \
                 but it is {n}."
            )),
        },
        Some(v) => Err(format!(
            "files '{name}': '{key}' must be a whole number of {unit}, but it is {}.",
            json_kind(v)
        )),
    }
}

pub fn get_string_list(
    name: &str,
    extra: &Extra,
    key: &str,
) -> Result<Option<Vec<String>>, String> {
    let Some(v) = extra.get(key) else {
        return Ok(None);
    };
    let serde_json::Value::Array(items) = v else {
        return Err(format!(
            "files '{name}': '{key}' must be a list of column names, but it is {}.",
            json_kind(v)
        ));
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            serde_json::Value::String(s) => out.push(s.clone()),
            other => {
                return Err(format!(
                    "files '{name}': every entry of '{key}' must be a column name, but one is {}.",
                    json_kind(other)
                ))
            }
        }
    }
    Ok(Some(out))
}

pub fn get_string_map(
    name: &str,
    extra: &Extra,
    key: &str,
) -> Result<Option<Vec<(String, String)>>, String> {
    let Some(v) = extra.get(key) else {
        return Ok(None);
    };
    let serde_json::Value::Object(map) = v else {
        return Err(format!(
            "files '{name}': '{key}' must be an object of column name → text, but it is {}.",
            json_kind(v)
        ));
    };
    let mut out = Vec::with_capacity(map.len());
    for (column, value) in map {
        match value {
            serde_json::Value::String(s) => out.push((column.clone(), s.clone())),
            other => {
                return Err(format!(
                    "files '{name}': '{key}' entry '{column}' must be text, but it is {}.",
                    json_kind(other)
                ))
            }
        }
    }
    Ok(Some(out))
}

/// The first name in `names` that an earlier entry already used.
pub fn first_duplicate(names: &[String]) -> Option<&str> {
    for (i, name) in names.iter().enumerate() {
        if names[..i].iter().any(|earlier| earlier == name) {
            return Some(name);
        }
    }
    None
}
