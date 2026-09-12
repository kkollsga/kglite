use std::path::Path;
use std::process::{Command, Output};

use serde_json::{json, Value};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_kglite")
}

fn invoke(args: &[&str], cache: &Path, cwd: Option<&Path>) -> Output {
    let mut command = Command::new(binary());
    command.args(args).env("KGLITE_AGENT_CACHE_DIR", cache);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command.output().unwrap()
}

fn json_stdout(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap()
}

fn budget(value: &Value) -> Value {
    serde_json::from_str::<Value>(value["content"][0]["text"].as_str().unwrap()).unwrap()
        ["response_budget"]
        .clone()
}

#[test]
fn emitted_commands_expand_late_and_nested_evidence_after_graph_moves() {
    let temp = tempfile::tempdir().unwrap();
    let graph = temp.path().join("evidence.kgl");
    let cache = temp.path().join("cache");
    let initial = invoke(
        &[
            "write",
            graph.to_str().unwrap(),
            "UNWIND range(0, 999) AS i CREATE (:Item {id:i, text:'αβγ界'}) RETURN i, 'αβγ界'",
            "--save",
            "--format",
            "agent",
            "--response-max-bytes",
            "4096",
        ],
        &cache,
        None,
    );
    assert!(
        initial.status.success(),
        "{}",
        String::from_utf8_lossy(&initial.stderr)
    );
    assert!(initial.stdout.len() <= 4097);
    let preview = json_stdout(&initial);
    let initial_budget = budget(&preview);
    let retained_record: Value = std::fs::read_dir(cache.join("entries"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
        .unwrap();
    let commands = initial_budget["domain_commands"].as_array().unwrap();
    let command_for = |purpose: &str| {
        commands
            .iter()
            .find(|item| item["purpose"] == purpose)
            .unwrap()["command"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    let late = command_for("inspect a late positional row");
    let nested = commands
        .iter()
        .find(|item| item["json_pointer"] == "/rows/0/1")
        .unwrap()["command"]
        .as_str()
        .unwrap()
        .to_owned();

    let moved = temp.path().join("moved.kgl");
    std::fs::rename(&graph, &moved).unwrap();
    let before = std::fs::read(&moved).unwrap();
    let alternate = temp.path().join("alternate");
    std::fs::create_dir(&alternate).unwrap();
    for (command, expected) in [(late, json!([37, "αβγ界"])), (nested, json!("αβγ界"))] {
        let args: Vec<_> = command.split_whitespace().skip(1).collect();
        let expanded = invoke(&args, &cache, Some(&alternate));
        assert!(expanded.status.success());
        assert!(expanded.stderr.is_empty());
        let expanded = json_stdout(&expanded);
        assert!(expanded["isError"] == false);
        assert_eq!(budget(&expanded)["preview"]["value"], expected);
    }
    assert_eq!(
        std::fs::read(&moved).unwrap(),
        before,
        "expansion changed graph"
    );

    let full_command = initial_budget["next"]["full_result"].as_str().unwrap();
    let args: Vec<_> = full_command.split_whitespace().skip(1).collect();
    let full = invoke(&args, &cache, Some(&alternate));
    assert_eq!(
        json_stdout(&full)["structuredContent"],
        retained_record["envelope"]
    );

    let long = "αβγ界".repeat(4_000);
    let long_query = format!("RETURN '{long}' AS text");
    let long_result = invoke(
        &[
            "query",
            moved.to_str().unwrap(),
            &long_query,
            "--format",
            "agent",
            "--response-max-bytes",
            "4096",
        ],
        &cache,
        None,
    );
    let long_budget = budget(&json_stdout(&long_result));
    let inspect = long_budget["domain_commands"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["json_pointer"] == "/rows/0/0")
        .unwrap()["command"]
        .as_str()
        .unwrap();
    let args: Vec<_> = inspect.split_whitespace().skip(1).collect();
    let first_page = budget(&json_stdout(&invoke(&args, &cache, Some(&alternate))));
    let end = first_page["preview"]["end"].as_u64().unwrap() as usize;
    let expected_first: String = long.chars().take(end).collect();
    assert_eq!(first_page["preview"]["excerpt"], expected_first);
    let page = first_page["next"]["page"].as_str().unwrap();
    let args: Vec<_> = page.split_whitespace().skip(1).collect();
    let second_page = budget(&json_stdout(&invoke(&args, &cache, Some(&alternate))));
    assert_eq!(second_page["preview"]["offset"], end);
    let second_end = second_page["preview"]["end"].as_u64().unwrap() as usize;
    let expected: String = long.chars().skip(end).take(second_end - end).collect();
    assert_eq!(second_page["preview"]["excerpt"], expected);
}

#[test]
fn agent_failures_keep_nonzero_status_and_warnings_stay_in_bounded_json() {
    let temp = tempfile::tempdir().unwrap();
    let graph = temp.path().join("warnings.kgl");
    let cache = temp.path().join("cache");
    let seed = invoke(
        &[
            "write",
            graph.to_str().unwrap(),
            "CREATE (:Item {id:1})",
            "--save",
        ],
        &cache,
        None,
    );
    assert!(seed.status.success());
    for _ in 0..2 {
        let warned = invoke(
            &[
                "query",
                graph.to_str().unwrap(),
                "MATCH (n:Itm) RETURN n",
                "--format",
                "agent",
                "--response-max-bytes",
                "4096",
            ],
            &cache,
            None,
        );
        assert!(warned.status.success());
        assert!(warned.stderr.is_empty());
        assert!(warned.stdout.len() <= 4097);
        let value = json_stdout(&warned);
        let warnings = value["structuredContent"]["diagnostics"]["warnings"]
            .as_array()
            .unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].as_str().unwrap().contains("Itm"));
    }

    let missing = temp.path().join("missing.kgl");
    let failed = invoke(
        &[
            "query",
            missing.to_str().unwrap(),
            "RETURN 1",
            "--format",
            "agent",
        ],
        &cache,
        None,
    );
    assert!(!failed.status.success());
    assert!(failed.stderr.is_empty());
    assert_eq!(json_stdout(&failed)["isError"], true);

    let rejected = temp.path().join("rejected.kgl");
    let conflict = invoke(
        &[
            "write",
            rejected.to_str().unwrap(),
            "CREATE (:Never)",
            "--save",
            "--format",
            "agent",
            "--response-full",
            "--response-max-bytes",
            "4096",
        ],
        &cache,
        None,
    );
    assert!(!conflict.status.success());
    assert!(!rejected.exists(), "invalid controls executed the mutation");
}

#[cfg(unix)]
#[test]
fn successful_mutation_with_retention_failure_stays_successful_and_complete() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let graph = temp.path().join("retained.kgl");
    let cache = temp.path().join("public-cache");
    std::fs::create_dir(&cache).unwrap();
    std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o755)).unwrap();
    let output = invoke(
        &[
            "write",
            graph.to_str().unwrap(),
            "UNWIND range(0, 4999) AS i CREATE (:Kept {id:i}) RETURN i",
            "--save",
            "--format",
            "agent",
        ],
        &cache,
        None,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value = json_stdout(&output);
    assert_eq!(value["isError"], false);
    assert_eq!(value["retention"]["retained"], false);
    assert_eq!(
        value["structuredContent"]["rows"].as_array().unwrap().len(),
        5000
    );
    assert_eq!(value["structuredContent"]["operation"]["mutation"], true);
}
