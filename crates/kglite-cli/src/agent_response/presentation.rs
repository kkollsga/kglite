use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use mcp_methods::response_budget::{
    Expansion, Mode, ResponseOptions, ResponseStore, DEFAULT_BYTES, MIN_BYTES,
};
use serde_json::{json, Value};

use super::cache::ResultCache;

#[cfg(test)]
const DEFAULT_RESPONSE_BYTES: usize = DEFAULT_BYTES;
#[cfg(test)]
const MIN_RESPONSE_BYTES: usize = MIN_BYTES;
const RENDERER_TOOL: &str = "expand_cli_response";
const RENDERER_OWNER: &str = "cli";

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AgentOptions {
    pub(crate) max_bytes: Option<usize>,
    pub(crate) full: bool,
}

impl AgentOptions {
    pub(crate) fn validate(self) -> Result<()> {
        self.framework().validate().map_err(anyhow::Error::msg)
    }

    fn framework(self) -> ResponseOptions {
        ResponseOptions {
            mode: if self.full { Mode::Full } else { Mode::Bounded },
            max_bytes: self.max_bytes,
        }
    }

    fn limit(self) -> usize {
        self.max_bytes.unwrap_or(DEFAULT_BYTES)
    }
}

pub(crate) fn cache_root() -> Result<PathBuf> {
    if let Some(root) = std::env::var_os("KGLITE_AGENT_CACHE_DIR") {
        return Ok(PathBuf::from(root));
    }
    #[cfg(target_os = "windows")]
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base = std::env::var_os("HOME").map(|home| PathBuf::from(home).join("Library/Caches"));
    #[cfg(all(unix, not(target_os = "macos")))]
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")));
    #[cfg(not(any(unix, target_os = "windows")))]
    let base: Option<PathBuf> = None;
    base.map(|path| path.join("kglite/agent-responses"))
        .ok_or_else(|| anyhow!("cannot determine the per-user cache directory"))
}

pub(crate) fn namespace(graph: &Path) -> String {
    graph
        .canonicalize()
        .unwrap_or_else(|_| graph.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn wrapped(envelope: &Value, error: bool) -> Value {
    json!({
        "content": [{"type": "text", "text": if error { "Cypher operation failed" } else { "Cypher result" }}],
        "structuredContent": envelope,
        "isError": error
    })
}

pub(crate) fn present(
    root: PathBuf,
    namespace: &str,
    envelope: Value,
    error: bool,
    options: AgentOptions,
) -> Value {
    let framework = options.framework();
    let original = wrapped(&envelope, error);
    let mut renderer = ResponseStore::default();
    let rendered = renderer.present(
        RENDERER_OWNER,
        "cypher",
        json!({"namespace": namespace}),
        original.clone(),
        &framework,
        RENDERER_TOOL,
    );
    if temporary_id(&rendered).is_none() {
        return rendered;
    }
    let cache = ResultCache::new(root);
    let handle = match cache.store(namespace, envelope.clone()) {
        Ok(handle) => handle,
        Err(cache_error) => return retention_warning(original, options.limit(), &cache_error),
    };
    fit_translated_preview(original, &envelope, namespace, options, &handle)
}

pub(crate) fn expand(
    root: PathBuf,
    handle: &str,
    path: String,
    offset: usize,
    options: AgentOptions,
) -> Result<Value> {
    options.validate()?;
    let loaded = ResultCache::new(root)
        .load(handle)?
        .ok_or_else(|| anyhow!("result unavailable (expired, evicted, corrupt, or unknown); the operation was not rerun"))?;
    let original = wrapped(&loaded.envelope, envelope_is_error(&loaded.envelope));
    let mut renderer = ResponseStore::default();
    let seed = renderer.present(
        RENDERER_OWNER,
        "cypher",
        json!({"namespace": loaded.namespace}),
        original,
        &ResponseOptions {
            mode: Mode::Bounded,
            max_bytes: Some(MIN_BYTES),
        },
        RENDERER_TOOL,
    );
    let temporary = temporary_id(&seed)
        .ok_or_else(|| anyhow!("retained result no longer requires expansion"))?;
    let expanded = renderer
        .expand(
            &RENDERER_OWNER,
            &Expansion {
                result_id: temporary,
                path,
                offset,
                response: options.framework(),
            },
            RENDERER_TOOL,
        )
        .map_err(anyhow::Error::msg)?;
    Ok(translate_preview(expanded, handle, Some(&loaded.envelope)))
}

pub(crate) fn purge(root: PathBuf) -> Result<()> {
    ResultCache::new(root).purge_all()
}

pub(crate) fn finalize_session(
    mut value: Value,
    op: &str,
    request_id: Option<Value>,
    options: AgentOptions,
) -> Value {
    value["op"] = json!(op);
    if let Some(request_id) = request_id {
        value["id"] = request_id;
    }
    if options.full {
        return value;
    }
    let limit = options.limit();
    shrink_to_limit_with_command_floor(&mut value, limit, 1);
    let actual = serialized_size(&value);
    if actual > limit {
        value["retention"] = json!({
            "complete": false,
            "budget_exceeded": true,
            "max_bytes": limit,
            "actual_bytes": 0,
            "warning": "Mandatory JSONL op/id fields and retrieval guidance cannot fit the requested budget; this response preserves them intact."
        });
        set_final_actual_bytes(&mut value);
    }
    value
}

fn set_final_actual_bytes(value: &mut Value) {
    loop {
        let actual = serialized_size(value);
        if value
            .pointer("/retention/actual_bytes")
            .and_then(Value::as_u64)
            == Some(actual as u64)
        {
            return;
        }
        value["retention"]["actual_bytes"] = json!(actual);
    }
}

fn fit_translated_preview(
    original: Value,
    envelope: &Value,
    namespace: &str,
    options: AgentOptions,
    handle: &str,
) -> Value {
    let limit = options.limit();
    let mut allowance = limit;
    loop {
        let mut renderer = ResponseStore::default();
        let candidate = renderer.present(
            RENDERER_OWNER,
            "cypher",
            json!({"namespace": namespace}),
            original.clone(),
            &ResponseOptions {
                mode: Mode::Bounded,
                max_bytes: Some(allowance),
            },
            RENDERER_TOOL,
        );
        let mut translated = translate_preview(candidate, handle, Some(envelope));
        shrink_to_limit(&mut translated, limit);
        let size = serialized_size(&translated);
        if size <= limit {
            return translated;
        }
        if allowance == MIN_BYTES {
            return translation_overage(translated, limit, size);
        }
        allowance = allowance.saturating_sub(size - limit).max(MIN_BYTES);
    }
}

fn temporary_id(value: &Value) -> Option<String> {
    let text = value.pointer("/content/0/text")?.as_str()?;
    let body: Value = serde_json::from_str(text).ok()?;
    body.pointer("/response_budget/result_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn translate_preview(mut value: Value, handle: &str, envelope: Option<&Value>) -> Value {
    let Some(text) = value.pointer("/content/0/text").and_then(Value::as_str) else {
        return value;
    };
    let Ok(mut body) = serde_json::from_str::<Value>(text) else {
        return value;
    };
    if let Some(budget) = body.get_mut("response_budget") {
        budget["result_id"] = json!(handle);
        if let Some(next) = budget.get_mut("next").and_then(Value::as_object_mut) {
            for (label, action) in next.iter_mut() {
                if action.get("name").and_then(Value::as_str) == Some(RENDERER_TOOL) {
                    *action = json!(command_for(handle, label, action));
                }
            }
        }
        if let Some(envelope) = envelope {
            budget["domain_commands"] = Value::Array(domain_commands(envelope, handle));
        }
    }
    value["content"][0]["text"] = json!(body.to_string());
    value
}

fn domain_commands(envelope: &Value, handle: &str) -> Vec<Value> {
    let navigation = &envelope["navigation"];
    let mut commands = navigation["targets"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|target| command_from_target(target, handle))
        .collect::<Vec<_>>();
    if let Some(rows) = envelope["rows"].as_array().filter(|rows| rows.len() > 16) {
        let index = 37.min(rows.len() - 1);
        commands.push(command_from_parts(
            "inspect a late positional row",
            &format!("/rows/{index}"),
            0,
            handle,
        ));
    }
    commands.extend(
        navigation["observed_value_targets"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|target| command_from_target(target, handle)),
    );
    commands
}

fn command_from_target(target: &Value, handle: &str) -> Option<Value> {
    Some(command_from_parts(
        target["purpose"]
            .as_str()
            .unwrap_or("inspect observed value"),
        target["json_pointer"].as_str()?,
        target["offset"].as_u64().unwrap_or(0),
        handle,
    ))
}

fn command_from_parts(purpose: &str, pointer: &str, offset: u64, handle: &str) -> Value {
    let arguments = json!({
        "result_id": handle,
        "path": pointer,
        "offset": offset,
        "response": {"mode":"bounded","max_bytes":4096}
    });
    json!({
        "purpose": purpose,
        "json_pointer": pointer,
        "offset": offset,
        "command": command_for(handle, "domain_target", &json!({"arguments":arguments}))
    })
}

fn shrink_to_limit(value: &mut Value, limit: usize) {
    shrink_to_limit_with_command_floor(value, limit, 7);
}

fn shrink_to_limit_with_command_floor(value: &mut Value, limit: usize, command_floor: usize) {
    let Some(text) = value.pointer("/content/0/text").and_then(Value::as_str) else {
        return;
    };
    let Ok(mut body) = serde_json::from_str::<Value>(text) else {
        return;
    };
    for field in [
        "content_overview",
        "metadata_overview",
        "tail_excerpt",
        "scope",
        "selection",
        "tool",
        "coverage",
    ] {
        value["content"][0]["text"] = json!(body.to_string());
        if serialized_size(value) <= limit {
            return;
        }
        body["response_budget"]
            .as_object_mut()
            .map(|object| object.remove(field));
    }
    value["content"][0]["text"] = json!(body.to_string());
    while serialized_size(value) > limit {
        let Some(commands) = body["response_budget"]
            .get_mut("domain_commands")
            .and_then(Value::as_array_mut)
        else {
            break;
        };
        if commands.len() <= command_floor {
            break;
        }
        commands.pop();
        value["content"][0]["text"] = json!(body.to_string());
    }
}

fn command_for(handle: &str, label: &str, action: &Value) -> String {
    let arguments = &action["arguments"];
    let mut parts = vec!["kglite response expand".to_string(), shell_word(handle)];
    if let Some(path) = arguments["path"].as_str().filter(|path| !path.is_empty()) {
        parts.extend(["--path".into(), shell_word(path)]);
    }
    if let Some(offset) = arguments["offset"].as_u64().filter(|offset| *offset != 0) {
        parts.extend(["--offset".into(), offset.to_string()]);
    }
    match arguments.pointer("/response/mode").and_then(Value::as_str) {
        _ if label == "selected_value" => {}
        Some("full") => parts.push("--response-full".into()),
        _ => {
            if let Some(bytes) = arguments
                .pointer("/response/max_bytes")
                .and_then(Value::as_u64)
            {
                parts.extend(["--response-max-bytes".into(), bytes.to_string()]);
            }
        }
    }
    parts.join(" ")
}

fn shell_word(value: &str) -> String {
    if value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"_./:-".contains(&byte))
    {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn serialized_size(value: &Value) -> usize {
    serde_json::to_vec(value).expect("JSON value").len()
}

fn retention_warning(mut original: Value, limit: usize, error: &anyhow::Error) -> Value {
    original["retention"] = json!({
        "complete": true,
        "retained": false,
        "max_bytes": limit,
        "warning": format!("Result could not be retained: {error}. The operation succeeded and was not rerun.")
    });
    original
}

fn translation_overage(mut value: Value, limit: usize, _actual: usize) -> Value {
    value["retention"] = json!({
        "complete": false,
        "budget_exceeded": true,
        "max_bytes": limit,
        "actual_bytes": 0,
        "warning": "Executable retrieval commands cannot fit the requested budget; this preview is returned intact."
    });
    loop {
        let actual = serialized_size(&value);
        if value["retention"]["actual_bytes"].as_u64() == Some(actual as u64) {
            break;
        }
        value["retention"]["actual_bytes"] = json!(actual);
    }
    value
}

fn envelope_is_error(envelope: &Value) -> bool {
    envelope
        .pointer("/representation/text_kind")
        .and_then(Value::as_str)
        == Some("error")
}

#[cfg(test)]
mod tests;
