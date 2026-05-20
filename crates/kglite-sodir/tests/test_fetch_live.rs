//! Live integration test against the Sodir FactMaps REST API.
//!
//! Skipped unless `SODIR_LIVE_TEST` is set in the environment — CI and
//! offline runs must not depend on the network. Run with:
//!
//! ```sh
//! SODIR_LIVE_TEST=1 cargo test -p kglite-sodir --test test_fetch_live
//! ```

use kglite_sodir::{fetch, ArcGISClient};

#[tokio::test]
async fn fetch_quadrant_live() {
    if std::env::var("SODIR_LIVE_TEST").is_err() {
        eprintln!("skipping fetch_quadrant_live — set SODIR_LIVE_TEST=1 to run");
        return;
    }

    let client = ArcGISClient::new().expect("client constructs");

    // `quadrant` is one of the smallest datasets — good for a probe.
    let n = fetch::count(&client, "quadrant")
        .await
        .expect("count probe succeeds");
    assert!(n > 0, "quadrant should report a non-zero row count");

    let tmp = tempfile::tempdir().unwrap();
    let csv_path = tmp.path().join("quadrant.csv");
    let written = fetch::fetch_to_csv(&client, "quadrant", &csv_path)
        .await
        .expect("fetch_to_csv succeeds");

    assert!(written > 0, "expected rows written");
    assert!(csv_path.is_file(), "csv file should exist");

    let content = std::fs::read_to_string(&csv_path).unwrap();
    let lines = content.lines().count();
    assert!(lines > 1, "csv should have a header + data rows");
    // Header should carry the WKT geometry column for a geometry layer.
    assert!(
        content.lines().next().unwrap().contains("wkt_geometry"),
        "geometry layer header should include wkt_geometry"
    );
}
