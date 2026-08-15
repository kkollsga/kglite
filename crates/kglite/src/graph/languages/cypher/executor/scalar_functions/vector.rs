//! Cypher scalar functions — vector category: `dot`, `cosine`, `norm` over
//! **list-valued data** (a stored list property, a list literal, a parameter, a
//! `collect()`), as opposed to the embedding *store* that `vector_score` /
//! `text_score` / `embedding_norm` read. Routed from
//! `super::evaluate_scalar_function`; returns `Ok(None)` when `name` is not one
//! of this category's functions so the dispatcher tries the next.
//!
//! ## Semantics (and why)
//!
//! - **`null` in → `null` out.** A missing property, an unbound variable, or an
//!   explicit `null` argument makes the whole call `null`. This is the
//!   convention every other kglite scalar function follows (`abs`, `sqrt`,
//!   `atan2`, …) and Cypher's own null propagation.
//! - **Length mismatch is an error, not `null`.** Neo4j's `vector.similarity.*`
//!   family only compares vectors of equal dimension and rejects the rest; the
//!   failure a mismatch describes — a 384-dimension vector meeting a
//!   768-dimension one — is a data bug that a silent `null` would hide in a
//!   column of otherwise plausible scores. The message names both lengths.
//! - **A non-numeric element is an error**, naming which vector and which
//!   position. Same reasoning, and it matches `vector_score`'s existing
//!   "query vector elements must be numeric". (Neo4j's GDS
//!   `gds.similarity.cosine` substitutes `0.0` for a null element instead; we
//!   deliberately do not, because a silently-zeroed component changes the
//!   answer without changing its shape.)
//! - **A non-list argument is an error**, even when the *other* argument is
//!   `null`: both arguments are evaluated and type-checked before the null
//!   short-circuit, so an ill-typed call is reported rather than masked.
//! - **`cosine` of a zero-length vector is `null`, not `0.0` or `NaN`.** The
//!   quotient is `0/0` — undefined, and `null` is Cypher's word for that. The
//!   vector-search [`Scorer`](crate::graph::algorithms::vector::Scorer) answers
//!   `0.0` for the same input because a top-k ranking needs a total order over
//!   every candidate; a scalar function carries no such constraint. Any other
//!   non-finite result (an infinite component) is `null` for the same reason.
//! - `norm([])` is `0.0` and `dot([], [])` is `0.0` — the empty sum. Only
//!   `cosine([], [])` is `null`, by the zero-norm rule above.
//!
//! ## Why these do not call the `algorithms::vector` f32 kernels
//!
//! Those kernels ([`dot_product`](crate::graph::algorithms::vector::dot_product)
//! and friends) take two contiguous `&[f32]` — the layout `EmbeddingStore.data`
//! keeps. A list *property* arrives as `&[Value]`, an enum per element, so
//! calling them would mean allocating and filling two temporary `Vec<f32>` per
//! row first; that conversion pass costs more than the arithmetic it feeds and
//! loses precision on the way. [`pair_sums`] instead makes one fused pass
//! straight over the elements, accumulating in `f64` — no allocation, no clone,
//! and the `Float64` the function returns is computed at its own width.

use super::super::helpers::*;
use super::super::*;
use crate::datatypes::values::Value;

/// The three sums a pairwise vector function needs, from one pass over both
/// vectors: `Σaᵢbᵢ`, `Σaᵢ²`, `Σbᵢ²`. `dot` reads the first; `cosine` reads all
/// three.
struct PairSums {
    dot: f64,
    norm_a_sq: f64,
    norm_b_sq: f64,
}

/// Read one vector element as an `f64`. Strictly `Int64` / `Float64` — a
/// numeric-looking *string* is not silently coerced, because a vector whose
/// elements are strings is a storage mistake, not a value to guess at.
fn element(fname: &str, which: &str, index: usize, v: &Value) -> Result<f64, String> {
    match v {
        Value::Int64(i) => Ok(*i as f64),
        Value::Float64(f) => Ok(*f),
        other => Err(format!(
            "{fname}(): {which} vector element {index} must be a number, got {}",
            other.type_name()
        )),
    }
}

/// One fused pass over two equal-length vectors. Callers check the lengths
/// first; `zip` would otherwise truncate silently.
fn pair_sums(fname: &str, a: &[Value], b: &[Value]) -> Result<PairSums, String> {
    let mut sums = PairSums {
        dot: 0.0,
        norm_a_sq: 0.0,
        norm_b_sq: 0.0,
    };
    for (index, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let x = element(fname, "first", index, av)?;
        let y = element(fname, "second", index, bv)?;
        sums.dot += x * y;
        sums.norm_a_sq += x * x;
        sums.norm_b_sq += y * y;
    }
    Ok(sums)
}

impl<'a> CypherExecutor<'a> {
    /// Resolve one argument into the vector's elements.
    ///
    /// `Ok(None)` means "this argument is null" — the caller turns that into a
    /// null result. The native [`Value::List`] arm *moves* the evaluated Vec
    /// out rather than cloning it, so the kernel reads the row's own elements
    /// (in-memory list reads themselves borrow — `ColumnStore::get_cow`).
    fn vector_arg(
        &self,
        fname: &str,
        which: &str,
        expr: &Expression,
        row: &ResultRow,
    ) -> Result<Option<Vec<Value>>, String> {
        match self.evaluate_expression(expr, row)? {
            Value::List(items) => Ok(Some(items)),
            Value::Null => Ok(None),
            // A bracketed string is a list here exactly as it is to
            // `size()` / `head()` / `last()`, parsed by the same helper, so a
            // graph that stored its vectors as JSON text answers too.
            Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    Ok(Some(parse_list_value(&Value::String(s))))
                } else {
                    Err(format!(
                        "{fname}(): {which} argument must be a list of numbers, \
                         got a string that is not a bracketed list"
                    ))
                }
            }
            other => Err(format!(
                "{fname}(): {which} argument must be a list of numbers, got {}",
                other.type_name()
            )),
        }
    }

    /// Evaluate `dot(a, b)`, `cosine(a, b)` and `norm(a)`.
    /// See the module header for the null / mismatch / zero-norm contract.
    pub(super) fn eval_vector_fn(
        &self,
        name: &str,
        args: &[Expression],
        row: &ResultRow,
    ) -> Result<Option<Value>, String> {
        let result: Result<Value, String> = match name {
            "dot" | "cosine" => {
                if args.len() != 2 {
                    return Err(format!("{name}() requires 2 arguments: {name}(a, b)"));
                }
                // Both arguments are evaluated and type-checked before the
                // null short-circuit, so `dot(null, 7)` reports the 7.
                let a = self.vector_arg(name, "first", &args[0], row)?;
                let b = self.vector_arg(name, "second", &args[1], row)?;
                let (Some(a), Some(b)) = (a, b) else {
                    return Ok(Some(Value::Null));
                };
                if a.len() != b.len() {
                    return Err(format!(
                        "{name}(): vectors must have the same length, got {} and {}",
                        a.len(),
                        b.len()
                    ));
                }
                let sums = pair_sums(name, &a, &b)?;
                if name == "dot" {
                    Ok(Value::Float64(sums.dot))
                } else {
                    let denom = (sums.norm_a_sq * sums.norm_b_sq).sqrt();
                    // Zero-length vector: the cosine is 0/0, i.e. undefined.
                    if denom == 0.0 {
                        return Ok(Some(Value::Null));
                    }
                    let cos = sums.dot / denom;
                    // Non-finite components (±inf) leave the quotient
                    // meaningless; say so rather than returning a NaN.
                    if cos.is_finite() {
                        Ok(Value::Float64(cos))
                    } else {
                        Ok(Value::Null)
                    }
                }
            }
            "norm" => {
                if args.len() != 1 {
                    return Err("norm() requires 1 argument: norm(a)".into());
                }
                let Some(a) = self.vector_arg(name, "first", &args[0], row)? else {
                    return Ok(Some(Value::Null));
                };
                let mut sum = 0.0f64;
                for (index, v) in a.iter().enumerate() {
                    let x = element(name, "first", index, v)?;
                    sum += x * x;
                }
                Ok(Value::Float64(sum.sqrt()))
            }
            _ => return Ok(None),
        };
        result.map(Some)
    }
}
