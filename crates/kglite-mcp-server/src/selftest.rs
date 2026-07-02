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

use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use super::{load_manifest, pick_mode, promote_local_workspace, Cli, Mode};

/// How long to wait for any single JSON-RPC response before declaring the
/// child unresponsive. Generous: a first `cypher_query` in local-workspace
/// mode triggers the code-tree build.
const RPC_TIMEOUT: Duration = Duration::from_secs(120);

/// Minimal JSON-RPC-over-child-stdio client. A background thread reads the
/// child's stdout into a channel so a hung child surfaces as a timeout (or a
/// fast `Disconnected` on child exit) rather than a deadlocked read.
struct Rpc {
    child: Child,
    stdin: ChildStdin,
    rx: mpsc::Receiver<Value>,
    next_id: i64,
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
        // stderr is inherited so the child's boot diagnostics (bad manifest,
        // missing .env, PATH-shadow warnings) reach the operator directly.
        let mut child = Command::new(&program)
            .args(&lead_args)
            .args(child_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to spawn child server: {}",
                    program.to_string_lossy()
                )
            })?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
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
        })
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
            let msg = self.rx.recv_timeout(RPC_TIMEOUT).map_err(|_| {
                anyhow!("no `{method}` response — child server unresponsive or exited (see stderr above)")
            })?;
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

/// Truncate a probe detail so multi-line tool output stays a one-liner.
fn snippet(text: &str) -> String {
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    if line.chars().count() > 100 {
        format!("{}…", line.chars().take(100).collect::<String>())
    } else {
        line.to_string()
    }
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

    // Child argv = our argv minus the program name and the selftest-only flags
    // (`--selftest`, and `--selftest-path <val>` in both space and `=` forms) —
    // the child is a real server and clap would reject those unknown flags.
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

    println!(
        "kglite-mcp-server --selftest  (mode: {})",
        mode_label(&mode)
    );
    println!("  spawning child server for a live MCP handshake …\n");

    let mut rpc = Rpc::spawn(&child_args)?;
    let mut checks: Vec<(&str, Check)> = Vec::new();

    // 1. initialize — if this fails there's nothing more to probe.
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
        }
        Err(e) => {
            checks.push(("server initializes", Check::Fail(e.to_string())));
            return report(checks);
        }
    }

    // 2. tools/list — the graph tools must be present in every mode.
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
    let has = |n: &str| names.iter().any(|x| x == n);

    if has("cypher_query") && has("graph_overview") {
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
                if has("cypher_query") { "" } else { "cypher_query " },
                if has("graph_overview") { "" } else { "graph_overview " },
            )),
        ));
    }

    // 3. github tools — informational (honest listing: present iff a token is
    //    reachable), never a hard failure.
    let gh: Vec<&str> = ["github_issues", "github_api", "screen_stargazers"]
        .into_iter()
        .filter(|t| has(t))
        .collect();
    if gh.is_empty() {
        checks.push((
            "github tools",
            Check::Skip("none registered (no GITHUB_TOKEN reachable, or disabled)".into()),
        ));
    } else {
        checks.push((
            "github tools",
            Check::Pass(format!("present: {}", gh.join(", "))),
        ));
    }

    // 4. activation — local-workspace. The `workspace.root` is a *wide sandbox
    //    boundary* that agents narrow with `set_root_dir` at runtime; it is
    //    never built as a unit. So the selftest must NOT `set_root_dir(root)` —
    //    for a broad root (the documented code-review archetype) that builds a
    //    code_tree over the whole tree, which is unbounded work and hangs the
    //    handshake. Registration-only by default; a real build+hydrate check is
    //    opt-in via `--selftest-path <subdir>` pointed at a small representative
    //    directory.
    let mut local_activated = false;
    if let Mode::LocalWorkspace { .. } = &mode {
        if !has("set_root_dir") {
            checks.push((
                "workspace activation",
                Check::Fail("set_root_dir tool not registered for local-workspace mode".into()),
            ));
        } else if let Some(path) = cli.selftest_path.as_ref() {
            let r = rpc.request(
                "tools/call",
                json!({"name": "set_root_dir", "arguments": {"path": path.to_string_lossy()}}),
            )?;
            let (text, is_error) = call_text(&r);
            if is_error || text.to_lowercase().contains("failed") {
                checks.push(("workspace activation", Check::Fail(snippet(&text))));
            } else {
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

    // 5. graph hydrates — a real cypher_query round-trip.
    let hydrate = match &mode {
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
            let (text, is_error) = call_text(&r);
            if is_error {
                Check::Fail(snippet(&text))
            } else {
                Check::Pass(format!("MATCH (n) RETURN count(n) → {}", snippet(&text)))
            }
        }
    };
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
