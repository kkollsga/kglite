//! The nesting ceiling for runtime values.
//!
//! Clone, Drop, comparison, hashing, serialisation and every binding's
//! converter recurse once per container level, so a value nested deeply enough
//! overflows the stack of whichever thread touches it. The parser already
//! bounds expressions, but a query can build depth at runtime (`reduce` with a
//! wrapping body, repeated `SET n.p = [n.p]`), and decoders read whatever a
//! file holds. Every place that puts a value inside a new container checks
//! [`MAX_VALUE_DEPTH`] with these helpers, so a deeper value never exists.
//!
//! 256 leaves stack headroom on 512 KiB caller threads (C ABI, Java, macOS
//! secondary threads) in release builds and on 2 MiB threads in debug builds;
//! 512 did not.

use super::values::Value;

/// Deepest container nesting a value may have. A scalar has depth 0, `[1]`
/// depth 1; a node or relationship counts its property map as one level.
pub const MAX_VALUE_DEPTH: usize = 256;

/// The error text for a value past [`MAX_VALUE_DEPTH`].
pub fn depth_error() -> String {
    format!("value nesting exceeds {MAX_VALUE_DEPTH} levels; flatten the list or map")
}

/// Container depth of `value`, counting at most `limit + 1` levels: a result
/// above `limit` means "too deep" without walking the rest. The walk recurses
/// at most `limit + 1` frames, so it is safe on a value of any depth.
pub fn container_depth(value: &Value, limit: usize) -> usize {
    depth_at(value, 0, limit)
}

fn depth_at(value: &Value, level: usize, limit: usize) -> usize {
    if level > limit {
        return level;
    }
    let inner = |children: &mut dyn Iterator<Item = &Value>| -> usize {
        let mut deepest = level + 1;
        for child in children {
            if !is_container(child) {
                continue;
            }
            deepest = deepest.max(depth_at(child, level + 1, limit));
            if deepest > limit {
                break;
            }
        }
        deepest
    };
    match value {
        Value::List(items) => inner(&mut items.iter()),
        Value::Map(map) => inner(&mut map.values()),
        Value::Node(node) => inner(&mut node.properties.values()),
        Value::Relationship(rel) => inner(&mut rel.properties.values()),
        Value::Path(path) => {
            let mut deepest = level + 1;
            for props in path
                .nodes
                .iter()
                .map(|n| &n.properties)
                .chain(path.rels.iter().map(|r| &r.properties))
            {
                let node_level = level + 2;
                deepest = deepest.max(node_level);
                for child in props.values().filter(|v| is_container(v)) {
                    deepest = deepest.max(depth_at(child, node_level, limit));
                }
                if deepest > limit {
                    break;
                }
            }
            deepest
        }
        _ => level,
    }
}

/// Whether `value` holds other values.
#[inline]
pub fn is_container(value: &Value) -> bool {
    matches!(
        value,
        Value::List(_) | Value::Map(_) | Value::Node(_) | Value::Relationship(_) | Value::Path(_)
    )
}

/// `Err` with [`depth_error`] when a new list or map holding `children` would
/// exceed [`MAX_VALUE_DEPTH`]. Scalars cost one tag test; only container
/// children are walked, and each walk stops at the ceiling.
pub fn check_wrap<'a>(children: impl IntoIterator<Item = &'a Value>) -> Result<(), String> {
    for child in children {
        if is_container(child) && container_depth(child, MAX_VALUE_DEPTH) >= MAX_VALUE_DEPTH {
            return Err(depth_error());
        }
    }
    Ok(())
}

/// `Err` with [`depth_error`] when `value` itself is deeper than
/// [`MAX_VALUE_DEPTH`]: for values that arrive whole (parameters, property
/// writes) rather than being built one level at a time.
pub fn check_value(value: &Value) -> Result<(), String> {
    if is_container(value) && container_depth(value, MAX_VALUE_DEPTH) > MAX_VALUE_DEPTH {
        return Err(depth_error());
    }
    Ok(())
}

#[cfg(test)]
#[path = "value_depth_tests.rs"]
mod tests;
