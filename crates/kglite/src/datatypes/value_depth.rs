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
    match value {
        Value::List(items) => children_depth(items.iter(), level + 1, limit),
        Value::Map(map) => children_depth(map.values(), level + 1, limit),
        Value::Node(node) => children_depth(node.properties.values(), level + 1, limit),
        Value::Relationship(rel) => children_depth(rel.properties.values(), level + 1, limit),
        Value::Path(path) => {
            let mut deepest = level + 1;
            for props in path
                .nodes
                .iter()
                .map(|n| &n.properties)
                .chain(path.rels.iter().map(|r| &r.properties))
            {
                deepest = deepest.max(children_depth(props.values(), level + 2, limit));
                if deepest > limit {
                    break;
                }
            }
            deepest
        }
        _ => level,
    }
}

/// Depth of a container at `level` holding `children`. Generic rather than
/// `dyn Iterator`: a virtual call per child measured +10 % on a 10k-item
/// comprehension that wraps every element.
#[inline]
fn children_depth<'a>(
    children: impl Iterator<Item = &'a Value>,
    level: usize,
    limit: usize,
) -> usize {
    let mut deepest = level;
    for child in children {
        if !is_container(child) {
            continue;
        }
        deepest = deepest.max(depth_at(child, level, limit));
        if deepest > limit {
            break;
        }
    }
    deepest
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
#[inline]
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
#[inline]
pub fn check_value(value: &Value) -> Result<(), String> {
    if is_container(value) && container_depth(value, MAX_VALUE_DEPTH) > MAX_VALUE_DEPTH {
        return Err(depth_error());
    }
    Ok(())
}

thread_local! {
    static DECODE_LEVEL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static DECODE_EXCEEDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Tracks how many [`Value`]s are being decoded inside one another on this
/// thread. A decoder recurses once per level before it holds any value to
/// measure, so the ceiling is enforced on entry: a payload crafted deeper
/// than the stack allows returns an error instead of overflowing it.
///
/// Entry allows one level past [`MAX_VALUE_DEPTH`] because a scalar leaf sits
/// one level below the deepest container; [`Self::check_container`] then
/// refuses a container at that last level.
pub struct DecodeDepth;

impl DecodeDepth {
    pub fn enter<E: serde::de::Error>() -> Result<Self, E> {
        let level = DECODE_LEVEL.with(|l| {
            l.set(l.get() + 1);
            l.get()
        });
        let guard = DecodeDepth;
        if level > MAX_VALUE_DEPTH + 1 {
            DECODE_EXCEEDED.with(|f| f.set(true));
            return Err(E::custom(depth_error()));
        }
        Ok(guard)
    }

    pub fn check_container<E: serde::de::Error>(&self, value: &Value) -> Result<(), E> {
        if is_container(value) && DECODE_LEVEL.with(|l| l.get()) > MAX_VALUE_DEPTH {
            DECODE_EXCEEDED.with(|f| f.set(true));
            return Err(E::custom(depth_error()));
        }
        Ok(())
    }
}

impl Drop for DecodeDepth {
    fn drop(&mut self) {
        DECODE_LEVEL.with(|l| l.set(l.get() - 1));
    }
}

/// Clear the "a decode hit the ceiling" flag before a decode starts.
pub fn reset_decode_exceeded() {
    DECODE_EXCEEDED.with(|f| f.set(false));
}

/// Whether a decode on this thread since the last reset hit the ceiling.
/// Serde codecs flatten custom errors to a generic message, so this carries
/// the cause to the error that reaches the user.
pub fn decode_exceeded() -> bool {
    DECODE_EXCEEDED.with(|f| f.get())
}

#[cfg(test)]
#[path = "value_depth_tests.rs"]
mod tests;
