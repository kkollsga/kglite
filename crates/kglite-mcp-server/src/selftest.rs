//! `--selftest`: a positive "did I set it up right?" check.
//!
//! Server failures are silent by design — a missing tool, a hidden github
//! tool (no token), a stale PATH-shadowing binary, or "No active graph" all
//! present as an *absence* of errors, so an operator can't tell "correctly
//! configured" from "quietly half-broken". This harness removes that
//! ambiguity: it re-spawns *this* binary with the operator's own flags
//! (minus the selftest-only flags), speaks a real MCP handshake over the
//! child's stdio (`initialize` → `tools/list` → activate → `cypher_query`),
//! and prints green/red per capability. Self-spawn (not an in-process
//! GraphState poke) is deliberate — only a real `tools/list` reflects the
//! mcp-methods-owned tool registry, which is exactly what "are my tools
//! present?" asks.
//!
//! For `workspace.kind: local` the `workspace.root` is a wide sandbox that
//! agents narrow with `set_root_dir` and is never built as a unit, so the
//! selftest is registration-only by default (building the whole root would be
//! unbounded work → a hang). `--selftest-path <subdir>` opts into a real
//! build + `cypher_query` hydration against a small representative directory.
//!
//! Exit code is 0 when every non-skipped check passes, 1 otherwise, so it
//! doubles as a CI / deployment smoke gate.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use mcp_methods::server::{Manifest, SkillsSource};

use super::{load_manifest, pick_mode, promote_local_workspace, Cli, Mode};

/// How long to wait for any single JSON-RPC response before declaring the
/// child unresponsive. Generous: a first `cypher_query` in local-workspace
/// mode triggers the code-tree build.
const RPC_TIMEOUT: Duration = Duration::from_secs(120);

/// How many trailing stderr lines to keep for quoting as the *cause* of a
/// missing response. Two, not one: a boot failure is usually a tracing line
/// plus the `ERROR:` summary, and the useful half varies.
const STDERR_TAIL_LINES: usize = 2;

/// Minimal JSON-RPC-over-child-stdio client. A background thread reads the
/// child's stdout into a channel so a hung child surfaces as a timeout (or a
/// fast `Disconnected` on child exit) rather than a deadlocked read; a second
/// thread mirrors the child's stderr to ours and keeps the tail so a failed
/// handshake can name what the child said instead of only its own symptom.
struct Rpc {
    child: Child,
    stdin: ChildStdin,
    rx: mpsc::Receiver<Value>,
    next_id: i64,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    stderr_reader: Option<thread::JoinHandle<()>>,
}

/// Resolve the command that launches a fresh server instance for the child
/// handshake, as `(program, leading_args)`.
///
/// The cargo standalone binary is its own `current_exe()`, so re-spawning it
/// directly works. But in the pip wheel the running process is the Python
/// interpreter and `kglite-mcp-server` is a console-script shim — there
/// `current_exe()` is Python, and `python --graph …` fails. The wheel's
/// `kglite.mcp_server.main` therefore exports `KGLITE_MCP_RESPAWN` (a JSON
/// array like `["/…/python", "-m", "kglite.mcp_server"]`) telling us how to
/// relaunch the *server*, not the interpreter. Absent that (standalone
/// binary), fall back to `current_exe()`.
fn respawn_command() -> Result<(OsString, Vec<OsString>)> {
    if let Ok(raw) = std::env::var("KGLITE_MCP_RESPAWN") {
        let parts: Vec<String> = serde_json::from_str(&raw)
            .context("KGLITE_MCP_RESPAWN is not a JSON array of strings")?;
        let mut it = parts.into_iter();
        let program = it
            .next()
            .context("KGLITE_MCP_RESPAWN is an empty array (need at least the program)")?;
        return Ok((OsString::from(program), it.map(OsString::from).collect()));
    }
    let exe =
        std::env::current_exe().context("cannot resolve current executable for --selftest")?;
    Ok((exe.into_os_string(), Vec::new()))
}

impl Rpc {
    fn spawn(child_args: &[OsString]) -> Result<Self> {
        let (program, lead_args) = respawn_command()?;
        // stderr is piped rather than inherited, then mirrored line-by-line to
        // our own stderr below: the operator still sees the child's boot
        // diagnostics (bad manifest, missing .env, PATH-shadow warnings) live
        // and in place, and we additionally keep the tail so a failed
        // handshake can quote the *cause* rather than only reporting that no
        // response arrived.
        let mut child = Command::new(&program)
            .args(&lead_args)
            .args(child_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to spawn child server: {}",
                    program.to_string_lossy()
                )
            })?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        let tail = stderr_tail.clone();
        let stderr_reader = thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                eprintln!("{line}");
                // `Caused by:` is anyhow's chain header — it carries none of
                // the cause it introduces, and keeping it would spend one of
                // the two tail slots on nothing.
                if line.trim().is_empty() || line.trim() == "Caused by:" {
                    continue;
                }
                let mut tail = tail.lock().expect("stderr tail lock");
                if tail.len() == STDERR_TAIL_LINES {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
        });
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
                // Non-JSON stdout lines (stray logging) are ignored.
            }
        });
        Ok(Self {
            child,
            stdin,
            rx,
            next_id: 0,
            stderr_tail,
            stderr_reader: Some(stderr_reader),
        })
    }

    /// The child's last non-empty stderr lines, as a trailing clause for a
    /// handshake failure. `exited` joins the mirror thread first — on child
    /// exit the pipe closes, so the join is bounded and guarantees the final
    /// line (typically the very error that killed the boot) is in the buffer;
    /// on a *hung* child the thread is still running and joining would block
    /// forever, so we read whatever has arrived.
    fn stderr_cause(&mut self, exited: bool) -> String {
        if exited {
            let _ = self.child.wait();
            if let Some(handle) = self.stderr_reader.take() {
                let _ = handle.join();
            }
        }
        let tail = self.stderr_tail.lock().expect("stderr tail lock");
        if tail.is_empty() {
            return " (child printed nothing to stderr)".to_string();
        }
        format!(
            " — last stderr: {}",
            tail.iter()
                .map(|line| truncate(line.trim(), 200))
                .collect::<Vec<_>>()
                .join(" | ")
        )
    }

    fn send(&mut self, payload: &Value) -> Result<()> {
        writeln!(self.stdin, "{}", serde_json::to_string(payload)?)
            .context("write to child stdin failed (child exited?)")?;
        self.stdin.flush().context("flush child stdin failed")?;
        Ok(())
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        loop {
            let msg = match self.rx.recv_timeout(RPC_TIMEOUT) {
                Ok(msg) => msg,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let cause = self.stderr_cause(true);
                    bail!("no `{method}` response — child server exited{cause}");
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    let cause = self.stderr_cause(false);
                    bail!(
                        "no `{method}` response — child server unresponsive after {}s{cause}",
                        RPC_TIMEOUT.as_secs()
                    );
                }
            };
            if msg.get("id").and_then(Value::as_i64) == Some(id) {
                if let Some(err) = msg.get("error") {
                    bail!("`{method}` returned an error: {err}");
                }
                return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
            }
            // A notification or unrelated id — keep waiting for ours.
        }
    }

    fn notify(&mut self, method: &str) -> Result<()> {
        self.send(&json!({"jsonrpc": "2.0", "method": method}))
    }
}

impl Drop for Rpc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Outcome of one capability probe.
enum Check {
    Pass(String),
    Fail(String),
    Skip(String),
}

/// Pull the joined text + `isError` flag out of a `tools/call` result.
fn call_text(result: &Value) -> (String, bool) {
    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let text = result
        .get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    (text, is_error)
}

/// Require the success shape of an MCP tool-call envelope before interpreting
/// its human-readable payload. Tool handlers can report typed failures as
/// ordinary text content, so callers must additionally validate the payload's
/// positive contract rather than treating `isError == false` as success.
fn successful_call_text(result: &Value) -> std::result::Result<String, String> {
    let (text, is_error) = call_text(result);
    let trimmed = text.trim();
    if is_error {
        return Err(if trimmed.is_empty() {
            "tool returned isError without diagnostic text".into()
        } else {
            snippet(trimmed)
        });
    }
    if trimmed.is_empty() {
        return Err("tool returned no text".into());
    }
    Ok(text)
}

/// Match the three documented positive `set_root_dir` outcomes.
///
/// Other successful MCP envelopes can contain negative operational outcomes
/// such as a missing path, superseded request, or abandoned refresh. Those
/// must remain red in a deployment selftest.
fn activation_succeeded(text: &str) -> bool {
    let Some(line) = text.lines().find(|line| !line.trim().is_empty()) else {
        return false;
    };
    let line = line.trim();
    (line.starts_with("Cloned '")
        || line.starts_with("Updated '")
        || line.starts_with("Activated (already up to date) '"))
        && line.contains("' at ")
}

/// Parse the exact one-row scalar projection emitted by
/// `MATCH (n) RETURN count(n) AS n`.
fn hydration_count(text: &str) -> Option<u64> {
    let mut lines = text.lines().filter_map(|line| {
        let line = line.trim();
        (!line.is_empty()).then_some(line)
    });
    if lines.next()? != "1 row(s):" || lines.next()? != "n" {
        return None;
    }
    lines.next()?.parse().ok()
}

/// Truncate a probe detail so multi-line tool output stays a one-liner.
fn snippet(text: &str) -> String {
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    truncate(line, 100)
}

fn truncate(line: &str, max_chars: usize) -> String {
    if line.chars().count() > max_chars {
        format!("{}…", line.chars().take(max_chars).collect::<String>())
    } else {
        line.to_string()
    }
}

/// Child argv = our argv minus the program name and the selftest-only flags
/// (`--selftest`, and `--selftest-path <val>` in both space and `=` forms) —
/// the child is a real server and clap would reject those unknown flags.
fn child_argv(argv: &[OsString]) -> Vec<OsString> {
    let mut child_args: Vec<OsString> = Vec::new();
    let mut skip_next = false;
    for a in argv.iter().skip(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        let s = a.to_string_lossy();
        if s == "--selftest" || s.starts_with("--selftest-path=") {
            continue;
        }
        if s == "--selftest-path" {
            skip_next = true; // also drop its value
            continue;
        }
        child_args.push(a.clone());
    }
    child_args
}

/// Whether the live `tools/list` registry carries this tool name.
fn registered(names: &[String], name: &str) -> bool {
    names.iter().any(|x| x == name)
}

/// 1. initialize. Returns `false` when the handshake failed — there is nothing
///    more to probe against a child that never came up.
fn check_initialize(rpc: &mut Rpc, checks: &mut Vec<(&'static str, Check)>) -> Result<bool> {
    let init = rpc.request(
        "initialize",
        json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "kglite-selftest", "version": env!("CARGO_PKG_VERSION")},
        }),
    );
    match init {
        Ok(result) => {
            rpc.notify("notifications/initialized")?;
            let name = result
                .get("serverInfo")
                .and_then(|s| s.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("(unnamed)");
            checks.push((
                "server initializes",
                Check::Pass(format!("serverInfo.name = {name}")),
            ));
            Ok(true)
        }
        Err(e) => {
            checks.push(("server initializes", Check::Fail(e.to_string())));
            Ok(false)
        }
    }
}

/// 2. tools/list — the graph tools must be present in every mode.
fn check_graph_tools(names: &[String], checks: &mut Vec<(&'static str, Check)>) {
    if registered(names, "cypher_query") && registered(names, "graph_overview") {
        checks.push((
            "graph tools registered",
            Check::Pass(format!(
                "cypher_query + graph_overview present ({} tools total)",
                names.len()
            )),
        ));
    } else {
        checks.push((
            "graph tools registered",
            Check::Fail(format!(
                "missing {}{}(if a code-mode client only sees grep/read_source, search the registry for 'cypher')",
                if registered(names, "cypher_query") { "" } else { "cypher_query " },
                if registered(names, "graph_overview") { "" } else { "graph_overview " },
            )),
        ));
    }
}

/// A non-empty recipe catalog is a two-route ownership unit. Probe the live
/// registry rather than trusting that successful manifest parsing implies both
/// tools were installed.
fn check_recipe_tools(
    manifest: Option<&Manifest>,
    names: &[String],
    checks: &mut Vec<(&'static str, Check)>,
) {
    let recipe_catalog_configured = manifest
        .and_then(|manifest| manifest.extensions.get("cypher_recipes"))
        .and_then(Value::as_object)
        .is_some_and(|catalog| !catalog.is_empty());
    let recipe_tools = (
        registered(names, "list_recipe_queries"),
        registered(names, "run_recipe_query"),
    );
    if recipe_catalog_configured {
        if recipe_tools == (true, true) {
            checks.push((
                "recipe catalog tools",
                Check::Pass("list_recipe_queries + run_recipe_query present".into()),
            ));
        } else {
            checks.push((
                "recipe catalog tools",
                Check::Fail(format!(
                    "configured catalog is missing {}{}",
                    if recipe_tools.0 {
                        ""
                    } else {
                        "list_recipe_queries "
                    },
                    if recipe_tools.1 {
                        ""
                    } else {
                        "run_recipe_query"
                    },
                )),
            ));
        }
    } else if recipe_tools != (false, false) {
        checks.push((
            "recipe catalog tools",
            Check::Fail("recipe routes registered without a non-empty catalog".into()),
        ));
    }
}

/// 2b. declared source roots — informational, never a hard failure.
///
/// A `source_root:` that does not resolve is non-fatal at boot (see
/// `modes::resolve_declared_source_roots`): since mcp-methods 0.4.7 the
/// surviving roots are served and only the missing ones are dropped. That is
/// exactly the "quietly half-broken" state this harness exists to surface —
/// especially in the partial case, where every tool call still succeeds and
/// nothing else says the search covered fewer directories than declared. So it
/// gets its own line — yellow, not red, because the graph capability the
/// deployment is *for* is intact.
///
/// Only manifests that declare a root produce a line; the far more common "no
/// source_root at all" is a configuration choice, not a degradation.
fn check_declared_source_roots(
    manifest: Option<&Manifest>,
    checks: &mut Vec<(&'static str, Check)>,
) {
    let Some(m) = manifest.filter(|m| !m.source_roots.is_empty()) else {
        return;
    };
    let (resolved, unresolved) = mcp_methods::server::resolve_source_roots_lenient(m);
    if unresolved.is_empty() {
        checks.push((
            "manifest source roots",
            Check::Pass(format!("resolved: {}", resolved.join(", "))),
        ));
        return;
    }
    // Name what is missing AND what survived: with a partial resolve the
    // source tools answer normally from the survivors, so an operator reading
    // only "something is missing" cannot tell whether the tools are dead or
    // merely searching less than they declared.
    let missing = unresolved
        .iter()
        .map(|bad| format!("{:?} → {}", bad.declared, bad.path.display()))
        .collect::<Vec<_>>()
        .join(", ");
    let state = if resolved.is_empty() {
        "source tools unavailable — no declared root resolved".to_string()
    } else {
        format!(
            "source tools serving {} of {} declared roots ({})",
            resolved.len(),
            resolved.len() + unresolved.len(),
            resolved.join(", ")
        )
    };
    checks.push((
        "manifest source roots",
        Check::Skip(format!(
            "{state}; unresolved: {missing}. Graph tools unaffected — create the \
             directory or fix source_root, then restart"
        )),
    ));
}

/// 2c. skills — how many the session actually serves.
///
/// Opting in is one line and the payoff is invisible: skills ride tool
/// *descriptions*, so a manifest that opts in and resolves nothing looks
/// exactly like one that resolved everything. The commonest way to get there
/// is a project directory keyed to the wrong basename — it is
/// `<manifest>.skills/`, next to the YAML, not `<graph>.skills/` — and a count
/// is what makes that visible. Zero is legal (opt in now, add files later), so
/// it is yellow, not red.
///
/// Only manifests that opt in produce a line; `skills:` absent or `false` is a
/// configuration choice and has no `prompts/list` to ask.
fn check_skills(
    rpc: &mut Rpc,
    manifest: Option<&Manifest>,
    checks: &mut Vec<(&'static str, Check)>,
) {
    if !manifest.is_some_and(|m| matches!(m.skills, SkillsSource::Sources(_))) {
        return;
    }
    let names = match rpc.request("prompts/list", json!({})) {
        Ok(result) => result
            .get("prompts")
            .and_then(Value::as_array)
            .map(|ps| {
                ps.iter()
                    .filter_map(|p| p.get("name").and_then(Value::as_str).map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        Err(e) => {
            checks.push(("skills", Check::Fail(truncate(&e.to_string(), 200))));
            return;
        }
    };
    if names.is_empty() {
        checks.push((
            "skills",
            Check::Skip(
                "0 served — `skills:` is on but no skill resolved; the project layer is \
                 `<manifest-basename>.skills/` next to the YAML"
                    .into(),
            ),
        ));
        return;
    }
    checks.push((
        "skills",
        Check::Pass(format!("{} served: {}", names.len(), names.join(", "))),
    ));
}

/// 3. github tools — informational, never a hard failure. Honest listing:
///    present iff the manifest opted in with `builtins.github: true` *and* a
///    token is reachable. Absence is the default, not a fault.
fn check_github_tools(names: &[String], checks: &mut Vec<(&'static str, Check)>) {
    let gh: Vec<&str> = ["github_issues", "github_api", "screen_stargazers"]
        .into_iter()
        .filter(|t| registered(names, t))
        .collect();
    if gh.is_empty() {
        checks.push((
            "github tools",
            Check::Skip(
                "none registered (needs `builtins.github: true` in the manifest, then a \
                 reachable GITHUB_TOKEN)"
                    .into(),
            ),
        ));
    } else {
        checks.push((
            "github tools",
            Check::Pass(format!("present: {}", gh.join(", "))),
        ));
    }
}

/// 4. activation — local-workspace. The `workspace.root` can be a wide
///    starting root that agents narrow with `set_root_dir` at runtime; it
///    is never built as a unit. (`sandbox_root`, when configured, owns the
///    containment boundary.) So the selftest must NOT `set_root_dir(root)` —
///    for a broad root (the documented code-review archetype) that builds a
///    code_tree over the whole tree, which is unbounded work and hangs the
///    handshake. Registration-only by default; a real build+hydrate check is
///    opt-in via `--selftest-path <subdir>` pointed at a small representative
///    directory.
///
/// Returns whether a graph was actually built, which gates the hydration probe.
fn check_local_activation(
    rpc: &mut Rpc,
    cli: &Cli,
    mode: &Mode,
    names: &[String],
    checks: &mut Vec<(&'static str, Check)>,
) -> Result<bool> {
    let mut local_activated = false;
    if let Mode::LocalWorkspace { .. } = mode {
        if !registered(names, "set_root_dir") {
            checks.push((
                "workspace activation",
                Check::Fail("set_root_dir tool not registered for local-workspace mode".into()),
            ));
        } else if let Some(path) = cli.selftest_path.as_ref() {
            let r = rpc.request(
                "tools/call",
                json!({"name": "set_root_dir", "arguments": {"path": path.to_string_lossy()}}),
            )?;
            match successful_call_text(&r) {
                Ok(text) if activation_succeeded(&text) => {
                    local_activated = true;
                    checks.push((
                        "workspace activation",
                        Check::Pass(format!(
                            "set_root_dir({}) → {}",
                            path.display(),
                            snippet(&text)
                        )),
                    ));
                }
                Ok(text) => {
                    checks.push(("workspace activation", Check::Fail(snippet(&text))));
                }
                Err(detail) => checks.push(("workspace activation", Check::Fail(detail))),
            }
        } else {
            checks.push((
                "workspace activation",
                Check::Pass(
                    "set_root_dir registered; wide workspace.root not built (built per-\
                     set_root_dir at runtime). Pass --selftest-path <subdir> to verify a build"
                        .into(),
                ),
            ));
        }
    }
    Ok(local_activated)
}

/// 5. graph hydrates — a real cypher_query round-trip.
fn check_hydration(rpc: &mut Rpc, mode: &Mode, local_activated: bool) -> Result<Check> {
    Ok(match mode {
        // github workspace needs a repo_management clone (network) to hydrate —
        // out of scope for a fast selftest.
        Mode::Workspace { .. } => Check::Skip(
            "github workspace: run repo_management(org/repo) then re-check (clone not attempted)"
                .into(),
        ),
        Mode::SourceRoot { .. } | Mode::Bare => {
            Check::Skip("no graph in this mode (file/bare tools only)".into())
        }
        // local-workspace with no `--selftest-path`: nothing was built (the wide
        // root is not built as a unit), so there's no graph to query yet.
        Mode::LocalWorkspace { .. } if !local_activated => Check::Skip(
            "local-workspace: wide root not built; pass --selftest-path <subdir> to build \
             a representative subdir and verify hydration"
                .into(),
        ),
        _ => {
            let r = rpc.request(
                "tools/call",
                json!({"name": "cypher_query", "arguments": {"query": "MATCH (n) RETURN count(n) AS n"}}),
            )?;
            match successful_call_text(&r) {
                Ok(text) => match hydration_count(&text) {
                    Some(count) => {
                        Check::Pass(format!("MATCH (n) RETURN count(n) → {count} node(s)"))
                    }
                    None => Check::Fail(snippet(&text)),
                },
                Err(detail) => Check::Fail(detail),
            }
        }
    })
}

/// Entry point for `--selftest`. `argv` is the full original argv (program
/// name in `[0]`); `cli` is the already-parsed view used to re-derive the
/// mode so the harness knows how to activate and what to expect.
pub fn run_selftest(cli: &Cli, argv: &[OsString]) -> Result<()> {
    // Re-derive the mode exactly as boot does, so we activate correctly and
    // set the right expectations. Manifest load is best-effort here: if it
    // fails, the child hits the same error and `initialize` reports red.
    let mode = pick_mode(cli);
    let manifest = load_manifest(cli, &mode).ok().flatten();
    let mode = promote_local_workspace(mode.clone(), manifest.as_ref()).unwrap_or(mode);

    let child_args = child_argv(argv);

    println!(
        "kglite-mcp-server --selftest  (mode: {})",
        mode_label(&mode)
    );
    println!("  spawning child server for a live MCP handshake …\n");

    let mut rpc = Rpc::spawn(&child_args)?;
    let mut checks: Vec<(&str, Check)> = Vec::new();

    if !check_initialize(&mut rpc, &mut checks)? {
        return report(checks);
    }

    let tools = rpc.request("tools/list", json!({}))?;
    let names: Vec<String> = tools
        .get("tools")
        .and_then(Value::as_array)
        .map(|ts| {
            ts.iter()
                .filter_map(|t| t.get("name").and_then(Value::as_str).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    check_graph_tools(&names, &mut checks);
    check_recipe_tools(manifest.as_ref(), &names, &mut checks);
    check_declared_source_roots(manifest.as_ref(), &mut checks);
    check_skills(&mut rpc, manifest.as_ref(), &mut checks);
    check_github_tools(&names, &mut checks);

    let local_activated = check_local_activation(&mut rpc, cli, &mode, &names, &mut checks)?;
    let hydrate = check_hydration(&mut rpc, &mode, local_activated)?;
    checks.push(("graph hydrates", hydrate));

    report(checks)
}

fn mode_label(mode: &Mode) -> &'static str {
    match mode {
        Mode::Graph { .. } => "single-graph",
        Mode::SourceRoot { .. } => "source-root",
        Mode::Workspace { .. } => "github-workspace",
        Mode::LocalWorkspace { .. } => "local-workspace",
        Mode::Watch { .. } => "watch",
        Mode::Bare => "bare",
    }
}

/// Print the per-capability lines and return `Ok(())` iff nothing failed;
/// a failure returns an error so the process exits non-zero.
fn report(checks: Vec<(&str, Check)>) -> Result<()> {
    let mut failed = 0usize;
    for (label, check) in &checks {
        let (mark, detail) = match check {
            Check::Pass(d) => ("✓", d.as_str()),
            Check::Fail(d) => {
                failed += 1;
                ("✗", d.as_str())
            }
            Check::Skip(d) => ("–", d.as_str()),
        };
        if detail.is_empty() {
            println!("  {mark} {label}");
        } else {
            println!("  {mark} {label}: {detail}");
        }
    }
    println!();
    if failed == 0 {
        println!("Selftest PASSED — the server is configured correctly.");
        Ok(())
    } else {
        bail!("Selftest FAILED — {failed} check(s) did not pass (see above).");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_result(text: Option<&str>, is_error: bool) -> Value {
        let content =
            text.map_or_else(Vec::new, |text| vec![json!({"type": "text", "text": text})]);
        json!({"content": content, "isError": is_error})
    }

    #[test]
    fn call_envelope_requires_nonerror_text() {
        assert_eq!(
            successful_call_text(&tool_result(Some("ok"), false)).as_deref(),
            Ok("ok")
        );
        assert!(successful_call_text(&tool_result(Some("bad input"), true)).is_err());
        assert!(successful_call_text(&tool_result(Some("  \n"), false)).is_err());
        assert!(successful_call_text(&tool_result(None, false)).is_err());
    }

    #[test]
    fn activation_requires_a_positive_contract() {
        for text in [
            "Cloned 'local/ws' at /tmp/ws.",
            "Updated 'local/ws' at /tmp/ws.",
            "Activated (already up to date) 'local/ws' at /tmp/ws.",
        ] {
            assert!(activation_succeeded(text), "expected success for {text:?}");
        }

        for text in [
            "Path does not exist or is not a directory: /missing",
            "set_root_dir failed: parse error",
            "Activation request 1 for 'local/ws' was superseded by request 2.",
            "Refresh request for 'local/ws' was abandoned.",
            "No active graph.",
            "",
        ] {
            assert!(!activation_succeeded(text), "expected failure for {text:?}");
        }
    }

    #[test]
    fn hydration_requires_the_exact_count_projection() {
        assert_eq!(
            hydration_count("1 row(s):\nn\n0\n<active_graph path=\"x.kgl\"/>"),
            Some(0)
        );
        assert_eq!(hydration_count("1 row(s):\nn\n42"), Some(42));

        for text in [
            "No active graph. Pass --graph X.kgl.",
            "No results.",
            "2 row(s):\nn\n1\n2",
            "1 row(s):\ncount\n1",
            "1 row(s):\nn\nnot-a-number",
            "1 row(s):\nn\n-1",
            "Cypher query failed: unknown variable",
        ] {
            assert_eq!(hydration_count(text), None, "expected failure for {text:?}");
        }
    }
}
