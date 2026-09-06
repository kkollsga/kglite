//! Escaping preserves the value seen by the target format's parser.

pub(super) fn escape_xml(text: &str) -> Result<String, String> {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            // Character references avoid XML attribute/line-end normalization.
            '\t' => escaped.push_str("&#9;"),
            '\n' => escaped.push_str("&#10;"),
            '\r' => escaped.push_str("&#13;"),
            '\u{20}'..='\u{d7ff}' | '\u{e000}'..='\u{fffd}' | '\u{10000}'..='\u{10ffff}' => {
                escaped.push(ch)
            }
            _ => {
                return Err(format!(
                    "XML export cannot represent character U+{:04X}",
                    u32::from(ch)
                ))
            }
        }
    }
    Ok(escaped)
}

pub(super) fn escape_csv(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

pub(super) fn json_string(text: &str) -> String {
    // A string serializer has no fallible value cases; the output is UTF-8.
    serde_json::to_string(text).expect("serializing a UTF-8 string cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_coordinates_follow_float_json_policy() {
        use super::super::json_value;
        use crate::datatypes::values::Value;
        for coordinate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let value: serde_json::Value = serde_json::from_str(&json_value(&Value::Point {
                lat: coordinate,
                lon: 7.0,
            }))
            .unwrap();
            assert_eq!(value, serde_json::json!({"lat": null, "lon": 7}));
            let value: serde_json::Value = serde_json::from_str(&json_value(&Value::Point {
                lat: 5.0,
                lon: coordinate,
            }))
            .unwrap();
            assert_eq!(value, serde_json::json!({"lat": 5, "lon": null}));
        }
        let value: serde_json::Value = serde_json::from_str(&json_value(&Value::Point {
            lat: 5.0,
            lon: -7.0,
        }))
        .unwrap();
        assert_eq!(value, serde_json::json!({"lat": 5, "lon": -7}));
    }

    #[test]
    fn json_string_controls_roundtrip_exactly() {
        let controls: String = (0_u8..32).map(char::from).collect();
        for value in [
            "plain",
            "雪",
            "quote\" slash\\",
            "line\ncarriage\r tab\t",
            &controls,
        ] {
            let decoded: String = serde_json::from_str(&json_string(value)).unwrap();
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn csv_carriage_returns_are_quoted_and_preserved() {
        for value in ["plain", "雪", "a,b", "a\"b", "a\rb", "a\r\nb", "a\nb"] {
            let record = format!("{},sentinel\n", escape_csv(value));
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(record.as_bytes());
            let rows = reader.records().collect::<Result<Vec<_>, _>>().unwrap();
            assert_eq!(rows, vec![csv::StringRecord::from(vec![value, "sentinel"])]);
        }
        assert_eq!(escape_csv("a\rb"), "\"a\rb\"");
    }

    #[test]
    fn xml_whitespace_uses_references_and_forbidden_scalars_refuse() {
        assert_eq!(
            escape_xml("\t\n\r<&雪").unwrap(),
            "&#9;&#10;&#13;&lt;&amp;雪"
        );
        // Pure encoder checks only; no malformed XML file or graph is emitted.
        for ch in ['\0', '\u{1f}', '\u{fffe}', '\u{ffff}'] {
            let error = escape_xml(&ch.to_string()).unwrap_err();
            assert!(error.contains("XML export cannot represent character"));
        }
    }
}
