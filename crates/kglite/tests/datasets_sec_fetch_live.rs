//! Live SEC integration test — gated behind the
//! `KGLITE_SEC_INTEGRATION_TEST` env var so the default `cargo test`
//! doesn't hit the network.
//!
//! Run with: `KGLITE_SEC_INTEGRATION_TEST=1 cargo test -p kglite-sec --test test_fetch_live`

use std::env;
use std::path::PathBuf;

use kglite::api::datasets::sec::{
    fetch_company_tickers, fetch_quarterly_master_idx, FetchMode, SecClient, Workdir, YearRange,
};

fn live_tests_enabled() -> bool {
    env::var("KGLITE_SEC_INTEGRATION_TEST").is_ok()
}

fn user_agent() -> String {
    env::var("KGLITE_SEC_USER_AGENT")
        .unwrap_or_else(|_| "kglite-sec test runner kglite-tests@example.com".to_string())
}

fn isolated_workdir(name: &str) -> PathBuf {
    let dir = env::temp_dir().join(format!(
        "kglite-sec-live-{}-{}",
        name,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn fetches_one_quarter_master_idx() {
    if !live_tests_enabled() {
        eprintln!("skipping live test (set KGLITE_SEC_INTEGRATION_TEST=1 to run)");
        return;
    }

    let root = isolated_workdir("master-idx");
    let workdir = Workdir::new(&root);
    let client = SecClient::new(&user_agent()).expect("client");

    // 2023 Q4 is fully closed, so the fetch should hit the cache on
    // the second call.
    let range = YearRange::new(2023, 2023);
    let (dl1, sk1) = fetch_quarterly_master_idx(&client, &workdir, range, 2025, 1)
        .await
        .expect("first fetch");
    assert!(dl1 >= 1, "should download at least one quarter");
    assert_eq!(sk1, 0);

    // Second call must skip — raw/ is immutable for closed quarters.
    let (dl2, sk2) = fetch_quarterly_master_idx(&client, &workdir, range, 2025, 1)
        .await
        .expect("second fetch");
    assert_eq!(dl2, 0, "second call should fetch nothing");
    assert!(sk2 >= 1);

    // The downloaded master.idx file is present on disk for the closed quarter.
    assert!(
        workdir.raw_master_idx(2023, 4).is_file(),
        "fetched master.idx should be on disk"
    );

    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn fetches_company_tickers_json() {
    if !live_tests_enabled() {
        return;
    }

    let root = isolated_workdir("tickers");
    let workdir = Workdir::new(&root);
    let client = SecClient::new(&user_agent()).expect("client");

    let dl = fetch_company_tickers(&client, &workdir, false)
        .await
        .expect("fetch");
    assert!(dl);
    assert!(workdir.raw_company_tickers_json().is_file());

    // Re-fetch should skip
    let dl = fetch_company_tickers(&client, &workdir, false)
        .await
        .expect("re-fetch");
    assert!(!dl);

    std::fs::remove_dir_all(&root).ok();
}

#[tokio::test]
async fn rejects_missing_user_agent_on_construct() {
    if !live_tests_enabled() {
        return;
    }
    let err = SecClient::new("").err().expect("should reject");
    assert!(format!("{err}").contains("User-Agent"));
}

#[tokio::test]
async fn handles_unrelated_assertion_about_fetch_mode_enum() {
    if !live_tests_enabled() {
        return;
    }
    // No-op compile check; ensures FetchMode is publicly exposed
    let _ = FetchMode::OnlyIfMissing;
    let _ = FetchMode::Always;
}
