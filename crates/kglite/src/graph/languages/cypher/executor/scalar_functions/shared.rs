//! Shared constants and free helpers for the scalar-function modules.
use crate::datatypes::values::Value;

/// Shared error suffix when a spatial function arg can't be resolved to a
/// geometry or point. Names the conventional property names that the
/// fallback inference (in `build_node_spatial_data`) accepts so users have
/// a quick fix. Also surfaced from `resolve_spatial` when a node has no
/// registered spatial config and no inferable conventional fields.
pub(super) const SPATIAL_RESOLUTION_HELP: &str =
    "spatial argument did not resolve to a geometry or point. \
Either pass column_types={'<col>': 'geometry'} (or 'location.lat'/'location.lon') during \
add_nodes(), or store the data under a conventional property name (wkt_geometry, geometry, \
geom, or wkt for WKT; latitude+longitude or lat+lon for points).";

/// Recursively convert a parsed `serde_json::Value` into a kglite `Value`.
/// Objects become `Value::Map`, arrays `Value::List`; integers that fit i64
/// stay `Int64`, other numbers become `Float64`. Backs the `parse_json()`
/// Cypher function.
pub(super) fn json_to_value(j: &serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int64(i)
            } else {
                Value::Float64(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(a) => Value::List(a.iter().map(json_to_value).collect()),
        serde_json::Value::Object(o) => Value::Map(
            o.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect(),
        ),
    }
}

/// Which wall-clock "now" shape a `local*`/`time` function produces.
/// KGLite has no time-of-day Value variant, so these emit ISO-8601
/// strings (see the `localdatetime`/`localtime`/`time` arms).
#[derive(Clone, Copy)]
pub(super) enum LocalTemporalKind {
    /// `localdatetime()` → `YYYY-MM-DDTHH:MM:SS` (no offset).
    DateTime,
    /// `localtime()` / `time()` → `HH:MM:SS`.
    Time,
}

/// Advance the thread-local xorshift64 PRNG one step and return the
/// raw 64-bit state. Shared by `rand()`/`random()` and `randomUUID()`.
///
/// Seeded once per thread from SystemTime mixed with a monotonic
/// per-thread counter; subsequent calls just advance the state. Avoids
/// per-call `SystemTime::now()` overhead and guarantees distinct values
/// within a tight per-row loop. The counter splat ensures parallel
/// rayon workers don't collide on the same nanosecond.
pub(super) fn next_random_u64() -> u64 {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::SystemTime;
    static THREAD_COUNTER: AtomicU64 = AtomicU64::new(0);
    thread_local! {
        static XORSHIFT_STATE: Cell<u64> = {
            let nanos = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            let counter = THREAD_COUNTER.fetch_add(1, Ordering::Relaxed);
            // Mix counter via splitmix64-ish avalanche so adjacent
            // thread IDs produce well-separated seeds.
            let mut seed = nanos.wrapping_add(counter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            seed ^= seed >> 30;
            seed = seed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            seed ^= seed >> 27;
            seed = seed.wrapping_mul(0x94D0_49BB_1331_11EB);
            seed ^= seed >> 31;
            Cell::new(seed | 1)
        };
    }
    XORSHIFT_STATE.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        x
    })
}

/// Draw 128 random bits as two u64 halves (for `randomUUID()`).
pub(super) fn next_random_u128_halves() -> (u64, u64) {
    (next_random_u64(), next_random_u64())
}

/// Coerce a temporal Value to `NaiveDateTime` for cross-type temporal
/// arithmetic (`date_diff`, `duration.between`). A date-only `DateTime`
/// is treated as midnight, so mixing `date()` and `datetime()` operands
/// works. Returns `None` for non-temporal values.
pub(super) fn coerce_naive_datetime(v: &Value) -> Option<chrono::NaiveDateTime> {
    match v {
        Value::Timestamp(dt) => Some(*dt),
        Value::DateTime(d) => d.and_hms_opt(0, 0, 0),
        _ => None,
    }
}
