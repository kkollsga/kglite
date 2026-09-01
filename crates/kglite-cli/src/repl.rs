//! The interactive read-eval-print loop.
//!
//! Runs Cypher and a set of `sqlite3`-style dot-commands against an open
//! graph. Ctrl-C cancels a running query (the prompt itself stays alive —
//! rustyline keeps the terminal in raw mode, so SIGINT only fires while a
//! query is executing).

use std::collections::HashMap;
use std::io::{BufRead, IsTerminal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use std::time::Instant;

use anyhow::Result;
use kglite::api::io::{to_csv_dir, GraphFileIdentity, WriteOwnership};
use kglite::api::session::{execute_read, ExecuteOptions};
use kglite::api::{DirGraph, Value};
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::Editor;

use crate::exec::{self, QueryOptions};
use crate::format::{stdout_cell_cap, Mode};
use crate::helper::{self, ShellHelper};

const PROMPT: &str = "kglite> ";

/// Set by the SIGINT handler; polled by the executor via `ExecuteOptions.cancel`.
/// `'static` because the only setter is a process-global signal handler.
static CANCEL: AtomicBool = AtomicBool::new(false);

const HELP: &str = "\
Commands:
  .help                  show this help
  .quit / .exit          leave the shell
  .labels                list node types          (CALL db.labels)
  .rels                  list relationship types  (CALL db.relationshipTypes)
  .schema                per-type property schema (CALL db.schema)
  .indexes               list indexes             (CALL db.indexes)
  .mode table|csv|json   set output format
  .dump <dir>            export a portable CSV + blueprint copy
  .read <file>           run the Cypher statements in a file
  .import <csv> <Type>   load a CSV as nodes  [--id <col>] [--title <col>]
  .save [path]           save the graph to a .kgl file
  .timing on|off         show query wall-time after each statement
Anything else is run as Cypher, e.g.
  MATCH (n) RETURN labels(n), count(*)
Statements can span lines — a statement runs when a trailing ; terminates it
(with brackets and quotes closed). Piped (non-interactive) input also runs a
final, balanced statement at EOF. Tab completes dot-commands + labels.
Ctrl-C cancels a running query; Ctrl-D (or .quit) exits.";

/// Mutable shell state threaded through the loop.
struct Shell {
    graph: Arc<DirGraph>,
    /// The file the graph is associated with (for `.save` with no argument),
    /// together with what is needed to write it back without clobbering a
    /// change somebody else made to it. `None` for a session with no file.
    ownership: Option<WriteOwnership>,
    mode: Mode,
    /// Print wall-time after each statement (`.timing on`).
    timing: bool,
}

/// Run the REPL against `graph` until EOF / `.quit`.
pub fn run(graph: Arc<DirGraph>, ownership: Option<WriteOwnership>) -> Result<()> {
    // Install the SIGINT → cancel handler once. Ignore an error (e.g. handler
    // already set in an odd embedding) — the shell still works, just without
    // mid-query cancellation.
    let _ = ctrlc::set_handler(|| CANCEL.store(true, Ordering::SeqCst));

    let mut shell = Shell {
        graph,
        ownership,
        mode: Mode::default(),
        timing: false,
    };
    match shell.source_path() {
        Some(p) => println!("kglite shell — {p}"),
        None => println!("kglite shell — in-memory graph (not saved)"),
    }
    println!("Type .help for commands, .quit to exit.");

    // rustyline drives a terminal; it must not drive a pipe. Its non-TTY
    // fallback (`readline_direct`) accumulates continuation lines in a local
    // buffer that it *discards* on EOF, so a piped script whose last statement
    // lacked a `;` ran nothing and exited 0 — silently. Reading stdin ourselves
    // is the only way to see that buffer; the termination rule stays shared
    // (`helper::is_terminated`), so the two readers cannot drift.
    if std::io::stdin().is_terminal() {
        run_interactive(&mut shell)
    } else {
        run_piped(&mut shell)
    }
}

/// The terminal shell: rustyline owns line editing, history and multi-line
/// accumulation (via `ShellHelper`'s `Validator`).
fn run_interactive(shell: &mut Shell) -> Result<()> {
    let mut rl: Editor<ShellHelper, DefaultHistory> = Editor::new()?;
    rl.set_helper(Some(ShellHelper::default()));
    loop {
        // Refresh tab-completion candidates (labels + relationship types) from
        // the current graph before each prompt.
        if let Some(h) = rl.helper_mut() {
            h.set_candidates(shell.completion_candidates());
        }
        match rl.readline(PROMPT) {
            Ok(line) => {
                // A multi-line statement arrives as one input; drop a trailing
                // `;` terminator before dispatch.
                let line = line.trim().trim_end_matches(';').trim();
                if line.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(line);
                if !dispatch(shell, line) {
                    break; // .quit / .exit
                }
            }
            // Ctrl-C at the prompt: abandon the line, keep the session.
            Err(ReadlineError::Interrupted) => continue,
            // Ctrl-D: exit cleanly.
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        }
    }
    Ok(())
}

/// The piped shell: read stdin line by line, accumulating continuation lines
/// under the same termination rule the prompt uses (`helper::is_terminated`).
///
/// Two flush points exist here that a terminal does not need:
/// * a dot-command line — a shell command is never part of a Cypher statement,
///   so it terminates a balanced statement still waiting for its `;`;
/// * end of input — a balanced trailing statement runs, which is what the
///   `.help` text promises and what a `kglite g.kgl <<< 'MATCH …'` heredoc
///   expects.
///
/// An *unbalanced* tail at EOF (an unclosed quote or bracket) is a truncated
/// script, not a statement: nothing is run for it and the shell exits non-zero.
fn run_piped(shell: &mut Shell) -> Result<()> {
    let stdin = std::io::stdin();
    let mut pending = String::new();
    for line in stdin.lock().lines() {
        let line = line?;
        if is_dot_line(&line) && !pending.trim().is_empty() && helper::is_balanced(&pending) {
            if !dispatch_buffer(shell, &pending) {
                return Ok(());
            }
            pending.clear();
        }
        if !pending.is_empty() {
            pending.push('\n');
        }
        pending.push_str(&line);
        if helper::is_terminated(&pending) {
            let keep_going = dispatch_buffer(shell, &pending);
            pending.clear();
            if !keep_going {
                return Ok(());
            }
        }
    }
    let tail = pending.trim();
    if tail.is_empty() {
        return Ok(());
    }
    if helper::is_balanced(&pending) {
        dispatch_buffer(shell, &pending);
        return Ok(());
    }
    anyhow::bail!(
        "unterminated input at end of stdin (unclosed quote or bracket) — nothing was run for: {}",
        elide(tail)
    )
}

/// True for a line that opens a dot-command (leading whitespace tolerated).
fn is_dot_line(line: &str) -> bool {
    line.trim_start().starts_with('.')
}

/// Shorten a tail for an error message so a truncated 10 MB script does not
/// become a 10 MB diagnostic.
fn elide(tail: &str) -> String {
    const MAX: usize = 120;
    if tail.chars().count() <= MAX {
        return tail.to_string();
    }
    let kept: String = tail.chars().take(MAX).collect();
    format!("{kept}…")
}

/// Normalise an accumulated buffer the way the prompt does (trim, drop the
/// `;` terminator) and run it. Returns `false` when the shell should exit.
fn dispatch_buffer(shell: &mut Shell, buffer: &str) -> bool {
    let statement = buffer.trim().trim_end_matches(';').trim();
    if statement.is_empty() {
        return true;
    }
    dispatch(shell, statement)
}

/// Run one normalised statement — a dot-command or Cypher. Returns `false`
/// when the shell should exit (`.quit` / `.exit`).
fn dispatch(shell: &mut Shell, statement: &str) -> bool {
    match statement.strip_prefix('.') {
        Some(cmd) => shell.dispatch_dot(cmd),
        None => {
            shell.run_cypher(statement);
            true
        }
    }
}

impl Shell {
    /// Handle a `.`-prefixed command. Returns `false` to exit the loop.
    fn dispatch_dot(&mut self, cmd: &str) -> bool {
        let mut parts = cmd.split_whitespace();
        let name = parts.next().unwrap_or("");
        let arg = cmd[name.len()..].trim();
        match name {
            "help" | "h" | "?" => println!("{HELP}"),
            "quit" | "exit" | "q" => return false,
            "labels" => self.run_cypher("CALL db.labels() YIELD label RETURN label ORDER BY label"),
            "rels" | "relationshiptypes" => self.run_cypher(
                "CALL db.relationshipTypes() YIELD relationshipType \
                 RETURN relationshipType ORDER BY relationshipType",
            ),
            "schema" => self.run_cypher(
                "CALL db.schema() YIELD nodeType, properties \
                 RETURN nodeType, properties ORDER BY nodeType",
            ),
            "indexes" => self.run_cypher(
                "CALL db.indexes() YIELD name, type, properties \
                 RETURN name, type, properties ORDER BY name",
            ),
            "mode" => self.set_mode(arg),
            "timing" => match arg {
                "on" => {
                    self.timing = true;
                    println!("timing on");
                }
                "off" => {
                    self.timing = false;
                    println!("timing off");
                }
                _ => println!(
                    "Usage: .timing on|off (current: {})",
                    if self.timing { "on" } else { "off" }
                ),
            },
            "dump" => self.dump(arg),
            "read" => self.read_file(arg),
            "save" => self.save(arg),
            "import" => self.import_csv(arg),
            other => println!("Unknown command '.{other}'. Try .help."),
        }
        true
    }

    fn set_mode(&mut self, arg: &str) {
        match Mode::parse(arg) {
            Some(m) => {
                self.mode = m;
                println!("output mode: {}", m.name());
            }
            None => println!(
                "Usage: .mode table|csv|json (current: {})",
                self.mode.name()
            ),
        }
    }

    fn dump(&self, dir: &str) {
        if dir.is_empty() {
            println!("Usage: .dump <directory>");
            return;
        }
        match to_csv_dir(&self.graph, dir, None, &self.graph.parent_types) {
            Ok(summary) => {
                for line in &summary.log_lines {
                    println!("{line}");
                }
                println!("exported to {dir}/ — reload with kglite.from_blueprint('{dir}/blueprint.json')");
            }
            Err(e) => eprintln!("error: {e}"),
        }
    }

    fn read_file(&mut self, file: &str) {
        if file.is_empty() {
            println!("Usage: .read <file>");
            return;
        }
        let contents = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("error: cannot read {file}: {e}");
                return;
            }
        };
        for stmt in split_statements(&contents) {
            println!("kglite> {stmt}");
            self.run_cypher(&stmt);
        }
    }

    /// The file this session would save to with a bare `.save`.
    fn source_path(&self) -> Option<String> {
        self.ownership
            .as_ref()
            .map(|o| o.path().to_string_lossy().to_string())
    }

    fn save(&mut self, arg: &str) {
        let target = if arg.is_empty() {
            self.source_path()
        } else {
            Some(arg.to_string())
        };
        let Some(path) = target else {
            println!("Usage: .save <path>  (no path is associated with this session yet)");
            return;
        };
        if let Err(error) = self.retarget(&path) {
            eprintln!("error: {error}");
            return;
        }
        let ownership = self.ownership.as_mut().expect("retarget installs one");
        match ownership.publish(&mut self.graph) {
            Ok(()) => println!("saved to {path}"),
            Err(refusal) => eprintln!(
                "error: {}",
                crate::write_refusal(std::path::Path::new(&path), refusal)
            ),
        }
    }

    /// Point the session at `path`, capturing what is on disk there now so a
    /// save to a file somebody else is also writing is refused rather than
    /// silently winning.
    fn retarget(&mut self, path: &str) -> std::io::Result<()> {
        if self.source_path().as_deref() == Some(path) {
            return Ok(());
        }
        let target = std::path::Path::new(path);
        let identity = GraphFileIdentity::capture(target)?;
        match self.ownership.as_mut() {
            Some(ownership) => ownership.retarget(target.to_path_buf(), identity),
            None => {
                self.ownership = Some(WriteOwnership::new(
                    target.to_path_buf(),
                    identity,
                    &self.graph,
                    None,
                    false,
                ))
            }
        }
        Ok(())
    }

    /// Execute one Cypher statement (no params) and print it in the active mode.
    fn run_cypher(&mut self, query: &str) {
        self.exec(query, HashMap::new());
    }

    /// Execute a Cypher statement with bound params and render the result.
    /// Everything goes through `execute_mut`: a read query run that way
    /// executes as a read (no version bump), so one path serves reads and
    /// writes in this single-user shell.
    fn exec(&mut self, query: &str, params: HashMap<String, Value>) {
        CANCEL.store(false, Ordering::SeqCst);

        let timing = self.timing;
        let start = Instant::now();
        let options = QueryOptions {
            cancel: Some(&CANCEL),
            ..QueryOptions::default()
        };
        match exec::execute(&mut self.graph, query, &params, &options) {
            Ok(outcome) => {
                println!(
                    "{}",
                    exec::render_outcome(self.mode, &outcome, stdout_cell_cap())
                );
                if timing {
                    println!("({:.3} ms)", start.elapsed().as_secs_f64() * 1e3);
                }
            }
            Err(e) => {
                if CANCEL.load(Ordering::SeqCst) {
                    println!("^C query cancelled");
                } else {
                    eprintln!("error: {e}");
                }
            }
        }
    }

    /// Snapshot of completion candidates from the current graph — node labels
    /// and relationship types, fetched via the `db.*` procedures. Best-effort:
    /// an empty list on error just means completion falls back to dot-commands.
    fn completion_candidates(&self) -> Vec<String> {
        let params: HashMap<String, Value> = HashMap::new();
        let opts = ExecuteOptions::new(&params);
        let mut out = Vec::new();
        for q in [
            "CALL db.labels() YIELD label RETURN label",
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
        ] {
            if let Ok(outcome) = execute_read(&self.graph, q, &opts) {
                for row in &outcome.result.rows {
                    if let Some(Value::String(s)) = row.first() {
                        out.push(s.clone());
                    }
                }
            }
        }
        out
    }

    /// `.import <file.csv> <NodeType> [--id <col>] [--title <col>]` — load a CSV
    /// as nodes. Each row is fed through `UNWIND $rows AS r CREATE (:Type {...})`
    /// with the row values bound as a parameter (so cell contents are never
    /// interpolated into the query — no injection); only the node type and
    /// column names appear as identifiers, and those are validated. `id`/`name`
    /// columns become the node identity/title via the usual CREATE rules; `--id`
    /// / `--title` override which column maps to each.
    fn import_csv(&mut self, arg: &str) {
        let toks: Vec<&str> = arg.split_whitespace().collect();
        if toks.len() < 2 {
            println!("Usage: .import <file.csv> <NodeType> [--id <col>] [--title <col>]");
            return;
        }
        let (file, node_type) = (toks[0], toks[1]);
        if !is_ident(node_type) {
            eprintln!("error: '{node_type}' is not a valid node type (letters/digits/_, not starting with a digit)");
            return;
        }
        let (mut id_col, mut title_col) = (None, None);
        let mut i = 2;
        while i < toks.len() {
            match toks[i] {
                "--id" => {
                    id_col = toks.get(i + 1).copied();
                    i += 2;
                }
                "--title" => {
                    title_col = toks.get(i + 1).copied();
                    i += 2;
                }
                other => {
                    eprintln!("error: unknown .import flag '{other}'");
                    return;
                }
            }
        }

        let mut rdr = match csv::Reader::from_path(file) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: cannot read {file}: {e}");
                return;
            }
        };
        let headers: Vec<String> = match rdr.headers() {
            Ok(h) => h.iter().map(str::to_string).collect(),
            Err(e) => {
                eprintln!("error: reading CSV header: {e}");
                return;
            }
        };
        for h in &headers {
            if !is_ident(h) {
                eprintln!("error: column '{h}' is not a valid identifier — rename it in the CSV");
                return;
            }
        }
        for (flag, col) in [("--id", id_col), ("--title", title_col)] {
            if let Some(c) = col {
                if !headers.iter().any(|h| h == c) {
                    eprintln!("error: {flag} column '{c}' is not in the CSV header");
                    return;
                }
            }
        }

        // Build the row params (values parameterized) and the CREATE assignment
        // list (identifiers only). A column chosen as id/title maps to the
        // `id`/`title` property; every other column maps to itself.
        let mut rows: Vec<Value> = Vec::new();
        for rec in rdr.records() {
            let rec = match rec {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: reading CSV row: {e}");
                    return;
                }
            };
            let map: kglite::datatypes::PropMap = headers
                .iter()
                .zip(rec.iter())
                .map(|(h, cell)| (h.as_str(), infer_value(cell)))
                .collect();
            rows.push(Value::Map(map));
        }
        let n = rows.len();

        let assignments: Vec<String> = headers
            .iter()
            .map(|h| {
                if Some(h.as_str()) == id_col {
                    format!("id: r.{h}")
                } else if Some(h.as_str()) == title_col {
                    format!("title: r.{h}")
                } else {
                    format!("{h}: r.{h}")
                }
            })
            .collect();
        let query = format!(
            "UNWIND $rows AS r CREATE (:{node_type} {{{}}})",
            assignments.join(", ")
        );
        let mut params = HashMap::new();
        params.insert("rows".to_string(), Value::List(rows));
        self.exec(&query, params);
        println!("imported {n} {node_type} node(s) from {file}");
    }
}

/// A valid Cypher identifier: `[A-Za-z_][A-Za-z0-9_]*`. Guards the node type
/// and CSV column names that land in the query text (values go through params).
fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Infer a typed `Value` from a raw CSV cell: integer, then float, then bool,
/// else string. Empty cells become `Null`.
fn infer_value(cell: &str) -> Value {
    if cell.is_empty() {
        return Value::Null;
    }
    if let Ok(i) = cell.parse::<i64>() {
        return Value::Int64(i);
    }
    if let Ok(f) = cell.parse::<f64>() {
        return Value::Float64(f);
    }
    match cell {
        "true" | "True" | "TRUE" => Value::Boolean(true),
        "false" | "False" | "FALSE" => Value::Boolean(false),
        _ => Value::String(cell.to_string()),
    }
}

/// Split a Cypher script into individual statements. Statements are
/// `;`-terminated when any *separator* `;` is present (the common `.cypher`
/// convention); otherwise each non-blank, non-`//`-comment line is one
/// statement.
///
/// The scan is quote-aware: a `;` inside a `'…'` or `"…"` literal is data, not
/// a separator, so `CREATE (:Note {body: 'a;b'})` stays one statement. A naive
/// `split(';')` tears it into two invalid fragments. Cypher escapes quotes with
/// a backslash, so an escaped quote does not end the literal.
///
/// Shared by the REPL's `.read` and by `kglite migrate`, where a mis-split
/// would be considerably more costly than in interactive use.
pub(crate) fn split_statements(contents: &str) -> Vec<String> {
    let separators = statement_separator_positions(contents);
    if separators.is_empty() {
        return contents
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty() && !s.starts_with("//"))
            .map(str::to_string)
            .collect();
    }
    let mut statements = Vec::new();
    let mut start = 0;
    for &end in separators.iter().chain(std::iter::once(&contents.len())) {
        if start > contents.len() {
            break;
        }
        let fragment = contents[start..end.min(contents.len())].trim();
        if !fragment.is_empty() {
            statements.push(fragment.to_string());
        }
        start = end + 1;
    }
    statements
}

/// Byte offsets of the `;` characters that actually separate statements —
/// i.e. those outside any string literal.
fn statement_separator_positions(contents: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (offset, ch) in contents.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match quote {
            Some(open) => match ch {
                '\\' => escaped = true,
                c if c == open => quote = None,
                _ => {}
            },
            None => match ch {
                '\'' | '"' => quote = Some(ch),
                ';' => positions.push(offset),
                _ => {}
            },
        }
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::{elide, infer_value, is_dot_line, is_ident, split_statements};
    use kglite::api::Value;

    #[test]
    fn ident_validation() {
        assert!(is_ident("Person"));
        assert!(is_ident("_x9"));
        assert!(!is_ident("9bad")); // leading digit
        assert!(!is_ident("a b")); // space
        assert!(!is_ident("a-b")); // hyphen
        assert!(!is_ident("")); // empty
    }

    #[test]
    fn value_inference() {
        assert_eq!(infer_value("42"), Value::Int64(42));
        assert_eq!(infer_value("-7"), Value::Int64(-7));
        assert_eq!(infer_value("2.5"), Value::Float64(2.5));
        assert_eq!(infer_value("true"), Value::Boolean(true));
        assert_eq!(infer_value("False"), Value::Boolean(false));
        assert_eq!(infer_value("hello"), Value::String("hello".to_string()));
        assert_eq!(infer_value(""), Value::Null);
        // A leading-zero/alpha string stays a string (not coerced to int).
        assert_eq!(infer_value("007x"), Value::String("007x".to_string()));
    }

    #[test]
    fn split_on_semicolons() {
        let got = split_statements("CREATE (:A);  MATCH (n) RETURN n ;\n");
        assert_eq!(got, vec!["CREATE (:A)", "MATCH (n) RETURN n"]);
    }

    #[test]
    fn split_on_lines_when_no_semicolons() {
        let got = split_statements("MATCH (a) RETURN a\n// a comment\n\nMATCH (b) RETURN b\n");
        assert_eq!(got, vec!["MATCH (a) RETURN a", "MATCH (b) RETURN b"]);
    }

    #[test]
    fn semicolon_inside_a_string_literal_is_not_a_separator() {
        // A naive split(';') tears this into two invalid fragments.
        let got = split_statements("CREATE (:Note {body: 'a;b'});\nMATCH (n) RETURN n;");
        assert_eq!(
            got,
            vec!["CREATE (:Note {body: 'a;b'})", "MATCH (n) RETURN n"]
        );

        // Double quotes behave the same, and an escaped quote does not end the
        // literal — so the `;` after it is still data.
        let got = split_statements("CREATE (:N {t: \"x\\\";y\"})");
        assert_eq!(got, vec!["CREATE (:N {t: \"x\\\";y\"})"]);
    }

    #[test]
    fn a_script_whose_only_semicolons_are_quoted_falls_back_to_line_mode() {
        // No separator semicolons at all, so each line is a statement — the
        // quoted `;` must not push the script into semicolon mode and produce
        // one giant fragment.
        let got = split_statements("MATCH (a) WHERE a.t = 'x;y' RETURN a\nMATCH (b) RETURN b");
        assert_eq!(
            got,
            vec!["MATCH (a) WHERE a.t = 'x;y' RETURN a", "MATCH (b) RETURN b"]
        );
    }

    #[test]
    fn dot_lines_are_recognised_with_leading_space() {
        assert!(is_dot_line(".quit"));
        assert!(is_dot_line("   .schema"));
        assert!(!is_dot_line("MATCH (n) RETURN n"));
        assert!(!is_dot_line(""));
    }

    #[test]
    fn elide_bounds_the_reported_tail() {
        assert_eq!(elide("RETURN 'oops"), "RETURN 'oops");
        let long = "x".repeat(500);
        let short = elide(&long);
        assert_eq!(short.chars().count(), 121); // 120 kept + the ellipsis
        assert!(short.ends_with('…'));
    }
}
