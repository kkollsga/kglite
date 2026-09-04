//! Shared optional metric/exact-policy parsing for scalar and indexed scoring.

use crate::datatypes::values::Value;
use crate::graph::algorithms::vector::DistanceMetric;

#[derive(Default)]
pub(super) struct VectorOptions {
    pub(super) metric: Option<DistanceMetric>,
    pub(super) exact: bool,
}

fn metric(name: &str) -> Result<DistanceMetric, String> {
    DistanceMetric::from_name(name).ok_or_else(|| {
        format!("vector_score(): unknown metric '{name}'. Use 'cosine', 'dot_product', 'euclidean', or 'poincare'.")
    })
}

fn exact(value: &Value) -> Result<bool, String> {
    let Value::Map(options) = value else {
        return Err(
            "vector_score(): options must be a map containing an optional boolean 'exact'".into(),
        );
    };
    let mut exact = false;
    for (key, value) in options.iter() {
        if key != "exact" {
            return Err(format!(
                "vector_score(): unknown options key '{key}'; use 'exact'"
            ));
        }
        let Value::Boolean(requested) = value else {
            return Err("vector_score(): options 'exact' must be a boolean".into());
        };
        exact = *requested;
    }
    Ok(exact)
}

pub(super) fn parse(tail: &[Value]) -> Result<VectorOptions, String> {
    match tail {
        [] => Ok(VectorOptions::default()),
        [Value::String(name)] => Ok(VectorOptions {
            metric: Some(metric(name)?),
            exact: false,
        }),
        [value @ Value::Map(_)] => Ok(VectorOptions {
            metric: None,
            exact: exact(value)?,
        }),
        // Existing non-string fourth metric arguments selected cosine. Keep
        // that behavior while giving maps their new explicit options meaning.
        [_] => Ok(VectorOptions {
            metric: Some(DistanceMetric::Cosine),
            exact: false,
        }),
        [Value::String(name), options] => Ok(VectorOptions {
            metric: Some(metric(name)?),
            exact: exact(options)?,
        }),
        _ => Err("vector_score(): expected (node, property, query [, metric] [, options])".into()),
    }
}
