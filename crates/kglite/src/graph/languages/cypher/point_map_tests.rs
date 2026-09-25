//! `point()` from a map: the geographic keys (`latitude`/`longitude`, or
//! `x`/`y` under the WGS-84 CRS) build the same point as the positional form;
//! a Cartesian or 3D point, which KGLite does not represent, is refused.

use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_read, ExecuteOptions};
use std::collections::HashMap;

fn run(query: &str) -> Result<Vec<Vec<Value>>, String> {
    let graph = DirGraph::new();
    let params: HashMap<String, Value> = [(
        "m".to_string(),
        Value::Map(
            [
                ("latitude".to_string(), Value::Float64(59.9)),
                ("longitude".to_string(), Value::Float64(10.75)),
            ]
            .into_iter()
            .collect(),
        ),
    )]
    .into();
    execute_read(&graph, query, &ExecuteOptions::eager(&params))
        .map(|outcome| outcome.result.rows)
        .map_err(|error| error.to_string())
}

fn oslo() -> Value {
    Value::Point {
        lat: 59.9,
        lon: 10.75,
    }
}

#[test]
fn geographic_map_forms_build_the_positional_point() {
    for query in [
        "RETURN point({latitude: 59.9, longitude: 10.75}) AS p",
        "RETURN point({longitude: 10.75, latitude: 59.9}) AS p",
        "RETURN point({x: 10.75, y: 59.9, crs: 'wgs-84'}) AS p",
        "RETURN point({x: 10.75, y: 59.9, crs: 'WGS-84'}) AS p",
        "RETURN point({x: 10.75, y: 59.9, srid: 4326}) AS p",
        "RETURN point({latitude: 59.9, longitude: 10.75, crs: 'wgs-84', srid: 4326}) AS p",
        "RETURN point($m) AS p",
        "WITH {latitude: 59.9, longitude: 10.75} AS m RETURN point(m) AS p",
        "RETURN point(59.9, 10.75) AS p",
    ] {
        assert_eq!(run(query).unwrap(), vec![vec![oslo()]], "{query}");
    }
    assert_eq!(
        run("RETURN point({latitude: 1, longitude: 2}).y AS lat, point({latitude: 1, longitude: 2}).longitude AS lon")
            .unwrap(),
        vec![vec![Value::Float64(1.0), Value::Float64(2.0)]]
    );
    assert_eq!(
        run("RETURN round(distance(point({latitude: 59.9, longitude: 10.75}), point(59.9, 10.75))) AS d")
            .unwrap(),
        vec![vec![Value::Float64(0.0)]]
    );
}

#[test]
fn null_map_or_null_coordinate_is_null() {
    for query in [
        "RETURN point(null) AS p",
        "RETURN point({latitude: null, longitude: 10.75}) AS p",
        "RETURN point({x: 10.75, y: null, crs: 'wgs-84'}) AS p",
    ] {
        assert_eq!(run(query).unwrap(), vec![vec![Value::Null]], "{query}");
    }
}

#[test]
fn unsupported_or_malformed_maps_are_errors() {
    for (query, needle) in [
        ("RETURN point({x: 1, y: 2}) AS p", "Cartesian"),
        (
            "RETURN point({x: 1, y: 2, crs: 'cartesian'}) AS p",
            "Cartesian",
        ),
        ("RETURN point({x: 1, y: 2, srid: 7203}) AS p", "Cartesian"),
        (
            "RETURN point({latitude: 1, longitude: 2, height: 3}) AS p",
            "3D",
        ),
        (
            "RETURN point({x: 1, y: 2, z: 3, crs: 'wgs-84-3d'}) AS p",
            "3D",
        ),
        ("RETURN point({latitude: 1}) AS p", "longitude"),
        ("RETURN point({latitude: 1, x: 2}) AS p", "latitude"),
        (
            "RETURN point({latitude: 'a', longitude: 2}) AS p",
            "numeric",
        ),
        (
            "RETURN point({latitude: 1, longitude: 2, colour: 'red'}) AS p",
            "colour",
        ),
        (
            "RETURN point({latitude: 1, longitude: 2, crs: 'wgs-84', srid: 7203}) AS p",
            "srid",
        ),
        ("RETURN point('abc') AS p", "map"),
    ] {
        let error = run(query).expect_err(query);
        assert!(error.contains(needle), "{query}: {error}");
    }
}
