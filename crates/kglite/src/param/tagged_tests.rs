use super::super::*;
use chrono::NaiveDate;
use std::collections::HashMap;

fn params(source: &str) -> HashMap<String, Value> {
    json_text_to_query_value_map(source).unwrap()
}

fn rejection(source: &str) -> JsonQueryParameterError {
    match json_text_to_query_value_map(source) {
        Err(JsonQueryTextError::Parameter(error)) => error,
        other => panic!("{source} must be rejected as a parameter, got {other:?}"),
    }
}

#[test]
fn date_tag_becomes_a_date_value() {
    let params = params(r#"{"v":{"$date":"2020-01-01"}}"#);
    assert_eq!(
        params["v"],
        Value::DateTime(NaiveDate::from_ymd_opt(2020, 1, 1).unwrap())
    );
}

#[test]
fn datetime_tag_becomes_a_timestamp_normalised_to_utc() {
    let params = params(
        r#"{"naive":{"$datetime":"2020-01-01T08:00:00.250"},"zoned":{"$datetime":"2020-01-01T10:00:00+02:00"}}"#,
    );
    let expected = NaiveDate::from_ymd_opt(2020, 1, 1)
        .unwrap()
        .and_hms_milli_opt(8, 0, 0, 250)
        .unwrap();
    assert_eq!(params["naive"], Value::Timestamp(expected));
    assert_eq!(
        params["zoned"],
        Value::Timestamp(expected - chrono::Duration::milliseconds(250))
    );
}

#[test]
fn duration_tag_takes_the_result_encoding_object() {
    let params = params(
        r#"{"v":{"$duration":{"months":1,"days":2,"seconds":3}},"w":{"$duration":{"days":7}}}"#,
    );
    assert_eq!(
        params["v"],
        Value::Duration {
            months: 1,
            days: 2,
            seconds: 3
        }
    );
    assert_eq!(
        params["w"],
        Value::Duration {
            months: 0,
            days: 7,
            seconds: 0
        }
    );
}

/// A result cell rendered by `kglite_value_to_json`, wrapped in its tag, reads
/// back as the value it was rendered from.
#[test]
fn result_encoding_round_trips_through_the_tag() {
    let date = NaiveDate::from_ymd_opt(1999, 12, 31).unwrap();
    let values = [
        ("$date", Value::DateTime(date)),
        (
            "$datetime",
            Value::Timestamp(date.and_hms_micro_opt(23, 59, 58, 123_456).unwrap()),
        ),
        (
            "$duration",
            Value::Duration {
                months: -2,
                days: 40,
                seconds: -7,
            },
        ),
    ];
    for (tag, value) in values {
        let source = serde_json::json!({ "v": { tag: kglite_value_to_json(&value) } }).to_string();
        assert_eq!(params(&source)["v"], value, "{source}");
    }
}

#[test]
fn tags_apply_inside_lists_and_maps() {
    let params = params(r#"{"v":[{"$date":"2020-01-01"},{"at":{"$date":"2021-02-03"}}]}"#);
    let Value::List(items) = &params["v"] else {
        panic!("list expected")
    };
    assert!(matches!(items[0], Value::DateTime(_)));
    let Value::Map(map) = &items[1] else {
        panic!("map expected")
    };
    assert!(matches!(map.get("at"), Some(Value::DateTime(_))));
}

#[test]
fn a_tag_key_beside_other_keys_is_an_ordinary_map() {
    let params = params(r#"{"v":{"$date":"2020-01-01","other":1},"w":{"date":"2020-01-01"}}"#);
    assert!(matches!(params["v"], Value::Map(_)));
    assert!(matches!(params["w"], Value::Map(_)));
}

#[test]
fn malformed_tags_are_rejected_with_their_path() {
    for (source, path) in [
        (r#"{"v":{"$date":"2020-13-45"}}"#, "$.v"),
        (r#"{"v":{"$date":20200101}}"#, "$.v"),
        (r#"{"v":[{"$datetime":"yesterday"}]}"#, "$.v[0]"),
        (r#"{"v":{"$duration":{"weeks":1}}}"#, "$.v"),
        (r#"{"v":{"$duration":{"days":1.5}}}"#, "$.v"),
        (r#"{"v":{"$duration":{"days":4294967296}}}"#, "$.v"),
        (r#"{"v":{"$duration":"P1D"}}"#, "$.v"),
    ] {
        let error = rejection(source);
        assert_eq!(
            error.kind(),
            JsonQueryParameterErrorKind::InvalidTemporal,
            "{source}"
        );
        assert_eq!(error.path(), path, "{source}");
    }
}

#[test]
fn tolerant_converter_decodes_the_same_tags_as_the_query_path() {
    let source = r#"{"d":{"$date":"2020-01-01"},"t":{"$datetime":"2020-01-01T10:00:00+02:00"},"s":[{"$duration":{"days":1}}]}"#;
    let parsed: serde_json::Value = serde_json::from_str(source).unwrap();
    let Value::Map(tolerant) = json_value_to_kglite_value(&parsed) else {
        panic!("object must convert to a map");
    };
    let strict = params(source);
    for key in ["d", "t", "s"] {
        assert_eq!(tolerant.get(key), strict.get(key), "{key}");
    }
    assert!(matches!(strict["d"], Value::DateTime(_)));
}

#[test]
fn tolerant_converter_keeps_an_invalid_tag_as_a_map() {
    let parsed: serde_json::Value = serde_json::from_str(r#"{"$date":"2020-13-45"}"#).unwrap();
    let Value::Map(map) = json_value_to_kglite_value(&parsed) else {
        panic!("an invalid tagged payload must stay a map");
    };
    assert_eq!(map.get("$date"), Some(&Value::String("2020-13-45".into())));
}
