//! Bolt-layer query intercepts: the recognizers, column contracts, and
//! stream builders for the verbs `kglite-bolt-server` answers itself —
//! `db.checkpoint()`, the server-facts trio (`dbms.components()`,
//! `dbms.showCurrentUser()`, `SHOW DATABASES`) — plus the EXPLAIN-rows →
//! Bolt `summary.plan` conversion. Split from `backend.rs` (which keeps the
//! Backend impl and the verb *handlers* that need server state).

use super::*;

/// Output columns of `db.checkpoint()`, in declaration order — the shape
/// Neo4j's own `db.checkpoint()` yields, so a client can call it the same way.
pub(super) const CHECKPOINT_COLUMNS: [&str; 2] = ["success", "message"];

/// A recognized `CALL db.checkpoint()` invocation and the columns it asked
/// for (all of [`CHECKPOINT_COLUMNS`] when there is no `YIELD`).
#[derive(Debug, PartialEq, Eq)]
pub(super) struct CheckpointCall {
    pub(super) columns: Vec<&'static str>,
}

/// Strip `keyword` from the front of `input`, case-insensitively, requiring a
/// word boundary after it; returns the remainder with leading whitespace
/// trimmed.
///
/// The boundary check is what stops `CALLS ...` or `YIELDing` from being read
/// as the keyword plus a remainder.
pub(super) fn strip_keyword_ci<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    if !input.get(..keyword.len())?.eq_ignore_ascii_case(keyword) {
        return None;
    }
    let rest = &input[keyword.len()..];
    match rest.chars().next() {
        Some(c) if c.is_alphanumeric() || c == '_' => None,
        _ => Some(rest.trim_start()),
    }
}

/// Recognize exactly `CALL db.checkpoint()`, optionally followed by a `YIELD`
/// naming a subset of [`CHECKPOINT_COLUMNS`]; `None` for anything else.
///
/// **Deliberately narrow.** Keywords and the procedure name are matched
/// case-insensitively (Cypher keywords are, and every driver spells the
/// procedure lowercase anyway), surrounding whitespace and one trailing `;`
/// are tolerated — but arguments inside the parentheses, an unknown or
/// repeated `YIELD` column, an alias (`YIELD success AS ok`), or a differently
/// *cased* column name all fall through to the engine, which answers with its
/// standard "Unknown procedure 'db.checkpoint'". That is the intended
/// behaviour, not a gap: YIELD names are Cypher identifiers, and quietly
/// re-casing one would hand the driver a record whose key is not the key the
/// client asked for. Only the exact verb is a bolt-layer verb.
pub(super) fn parse_checkpoint_call(query: &str) -> Option<CheckpointCall> {
    parse_procedure_call(query, "db.checkpoint", &CHECKPOINT_COLUMNS)
        .map(|columns| CheckpointCall { columns })
}

/// Build the single-record result a `db.checkpoint()` call answers with,
/// projected onto the columns the client yielded.
///
/// `success` is always `true`: every way a checkpoint can fail — refused,
/// or a save error — returns a Bolt FAILURE instead, so a record that reaches
/// the client is a record about a checkpoint that either wrote or was
/// deliberately skipped. `type` distinguishes the two ("w" wrote, "r" didn't).
pub(super) fn checkpoint_stream(
    call: &CheckpointCall,
    message: String,
    type_str: &str,
    started: Instant,
) -> ResultStream {
    let values: Vec<BoltValue> = call
        .columns
        .iter()
        .map(|column| {
            debug_assert!(
                CHECKPOINT_COLUMNS.contains(column),
                "parse_checkpoint_call yielded an unknown column: {column}"
            );
            if *column == CHECKPOINT_COLUMNS[0] {
                BoltValue::Boolean(true)
            } else {
                BoltValue::String(message.clone())
            }
        })
        .collect();
    ResultStream {
        metadata: ResultMetadata {
            columns: call.columns.iter().map(|c| (*c).to_string()).collect(),
            extra: BoltDict::new(),
        },
        records: vec![BoltRecord { values }],
        summary: BoltDict::from([
            ("type".to_string(), BoltValue::String(type_str.to_string())),
            (
                "t_last".to_string(),
                BoltValue::Integer(started.elapsed().as_millis() as i64),
            ),
        ]),
    }
}

/// Declared output columns of `CALL dbms.components()`, in declaration
/// order — the shape Neo4j yields, which Neo4j Browser reads as its very
/// first call (`serverInfoQuery` in the Browser source) to draw the
/// name/version/edition banner.
pub(super) const COMPONENTS_COLUMNS: [&str; 3] = ["name", "versions", "edition"];

/// Declared output columns of `CALL dbms.showCurrentUser()` (Neo4j shape).
pub(super) const SHOW_CURRENT_USER_COLUMNS: [&str; 3] = ["username", "roles", "flags"];

/// Columns of `SHOW DATABASES`, the Neo4j 5 set. Clients key on `name`,
/// `default`, `home`, and `currentStatus`; the rest are present so a client
/// reading the full row does not find columns missing.
pub(super) const SHOW_DATABASES_COLUMNS: [&str; 13] = [
    "name",
    "type",
    "aliases",
    "access",
    "address",
    "role",
    "writer",
    "requestedStatus",
    "currentStatus",
    "statusMessage",
    "default",
    "home",
    "constituents",
];

/// A bolt-layer *server-facts* verb: identity, user, database roster. Like
/// `db.checkpoint()`, these are answered here because the Cypher engine has
/// none of the state they report — no server identity (`--neo4j-compat`
/// lives in this crate), no auth config, no advertised address. An
/// engine-side implementation would make `graph.cypher("CALL
/// dbms.components()")` in the wheel answer questions about a server that
/// is not running.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ServerFactsVerb {
    DbmsComponents,
    DbmsShowCurrentUser,
    ShowDatabases,
}

/// A recognized server-facts invocation and the columns it asked for.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ServerFactsCall {
    pub(super) verb: ServerFactsVerb,
    pub(super) columns: Vec<&'static str>,
}

/// Recognize `CALL <proc>()`, optionally followed by `YIELD` naming a subset
/// of `declared`; `None` for anything else. Shares `parse_checkpoint_call`'s
/// deliberate narrowness: arguments, aliases, unknown or repeated or
/// differently-cased YIELD columns all fall through to the engine and its
/// standard "Unknown procedure" error.
pub(super) fn parse_procedure_call(
    query: &str,
    proc: &str,
    declared: &'static [&'static str],
) -> Option<Vec<&'static str>> {
    let mut body = query.trim();
    if let Some(stripped) = body.strip_suffix(';') {
        body = stripped.trim_end();
    }
    let rest = strip_keyword_ci(body, "call")?;
    let rest = strip_keyword_ci(rest, proc)?;
    let rest = rest.strip_prefix('(')?.trim_start();
    let rest = rest.strip_prefix(')')?.trim_start();
    if rest.is_empty() {
        return Some(declared.to_vec());
    }
    let yielded = strip_keyword_ci(rest, "yield")?;
    let mut columns: Vec<&'static str> = Vec::with_capacity(declared.len());
    for raw in yielded.split(',') {
        let name = raw.trim();
        let canonical = declared.iter().find(|column| **column == name)?;
        if columns.contains(canonical) {
            return None;
        }
        columns.push(*canonical);
    }
    Some(columns)
}

/// Recognize the server-facts verbs. `SHOW DATABASES` is matched in its
/// plain statement form only — a `YIELD`/`WHERE` tail falls through to the
/// engine (and its parse error), which is the honest answer until the
/// engine grows SHOW projections.
pub(super) fn parse_server_facts_call(query: &str) -> Option<ServerFactsCall> {
    if let Some(columns) = parse_procedure_call(query, "dbms.components", &COMPONENTS_COLUMNS) {
        return Some(ServerFactsCall {
            verb: ServerFactsVerb::DbmsComponents,
            columns,
        });
    }
    if let Some(columns) =
        parse_procedure_call(query, "dbms.showcurrentuser", &SHOW_CURRENT_USER_COLUMNS)
    {
        return Some(ServerFactsCall {
            verb: ServerFactsVerb::DbmsShowCurrentUser,
            columns,
        });
    }
    let mut body = query.trim();
    if let Some(stripped) = body.strip_suffix(';') {
        body = stripped.trim_end();
    }
    let rest = strip_keyword_ci(body, "show")?;
    let rest = strip_keyword_ci(rest, "databases")?;
    if rest.is_empty() {
        return Some(ServerFactsCall {
            verb: ServerFactsVerb::ShowDatabases,
            columns: SHOW_DATABASES_COLUMNS.to_vec(),
        });
    }
    None
}

/// Convert the engine's EXPLAIN step rows into Neo4j's nested Bolt plan
/// shape: each node `{operatorType, args, identifiers, children}`, with the
/// FINAL pipeline step as the root (Neo4j's root is the last operator) and
/// each earlier step nested as its single child. `OptimizerPass <name>` rows
/// are not operators — they are collected into the root's
/// `args["optimizer-passes"]` list. Returns `None` when the shape is not the
/// engine's EXPLAIN output (leaving the stream untouched).
pub(super) fn plan_from_explain_rows(
    columns: &[String],
    rows: &[Vec<kglite::datatypes::values::Value>],
) -> Option<BoltValue> {
    use kglite::datatypes::values::Value;

    let step_idx = columns.iter().position(|c| c == "step")?;
    let op_idx = columns.iter().position(|c| c == "operation")?;
    let est_idx = columns.iter().position(|c| c == "estimated_rows")?;

    let mut steps: Vec<(i64, String, Option<i64>)> = Vec::new();
    let mut passes: Vec<BoltValue> = Vec::new();
    for row in rows {
        let step = match row.get(step_idx)? {
            Value::Int64(i) => *i,
            _ => return None,
        };
        let op = match row.get(op_idx)? {
            Value::String(s) => s.clone(),
            _ => return None,
        };
        let est = match row.get(est_idx) {
            Some(Value::Int64(i)) => Some(*i),
            _ => None,
        };
        if let Some(name) = op.strip_prefix("OptimizerPass ") {
            passes.push(BoltValue::String(name.to_string()));
        } else {
            steps.push((step, op, est));
        }
    }
    if steps.is_empty() {
        return None;
    }
    steps.sort_by_key(|(step, _, _)| *step);

    // Text rendering of the plan for `args["string-representation"]` on the
    // root — Neo4j puts its ASCII plan there, and G.V()'s plan tab reads the
    // key UNCONDITIONALLY (`summary.plan().arguments().get("string-
    // representation").asString()`, verified by decompilation after its tab
    // rendered an NPE against a plan without it). Root operator first,
    // matching the tree orientation.
    let text: String = {
        let mut lines = Vec::with_capacity(steps.len() + passes.len());
        for (_, op, est) in steps.iter().rev() {
            match est {
                Some(est) => lines.push(format!("+ {op}  (estimated rows: {est})")),
                None => lines.push(format!("+ {op}")),
            }
        }
        if !passes.is_empty() {
            let names: Vec<&str> = passes
                .iter()
                .filter_map(|p| match p {
                    BoltValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            lines.push(format!("optimizer passes: {}", names.join(", ")));
        }
        lines.join("\n")
    };

    // Fold from the first step outward: the last operator becomes the root.
    let mut node: Option<BoltValue> = None;
    let last = steps.len() - 1;
    for (i, (_, op, est)) in steps.into_iter().enumerate() {
        let mut args = BoltDict::new();
        if let Some(est) = est {
            args.insert("EstimatedRows".to_string(), BoltValue::Float(est as f64));
        }
        if i == last {
            args.insert(
                "string-representation".to_string(),
                BoltValue::String(text.clone()),
            );
            args.insert(
                "runtime".to_string(),
                BoltValue::String("kglite".to_string()),
            );
            if !passes.is_empty() {
                args.insert(
                    "optimizer-passes".to_string(),
                    BoltValue::List(std::mem::take(&mut passes)),
                );
            }
        }
        let children = match node.take() {
            Some(child) => vec![child],
            None => Vec::new(),
        };
        node = Some(BoltValue::Dict(BoltDict::from([
            ("operatorType".to_string(), BoltValue::String(op)),
            ("args".to_string(), BoltValue::Dict(args)),
            ("identifiers".to_string(), BoltValue::List(Vec::new())),
            ("children".to_string(), BoltValue::List(children)),
        ])));
    }
    node
}

/// Project `(column, value)` pairs onto the client's yielded columns and wrap
/// them as a single-record read stream.
pub(super) fn server_facts_stream(
    columns: &[&'static str],
    values: &[(&'static str, BoltValue)],
    started: Instant,
) -> ResultStream {
    let record: Vec<BoltValue> = columns
        .iter()
        .map(|column| {
            values
                .iter()
                .find(|(name, _)| name == column)
                .map(|(_, value)| value.clone())
                .unwrap_or(BoltValue::Null)
        })
        .collect();
    ResultStream {
        metadata: ResultMetadata {
            columns: columns.iter().map(|c| (*c).to_string()).collect(),
            extra: BoltDict::new(),
        },
        records: vec![BoltRecord { values: record }],
        summary: BoltDict::from([
            ("type".to_string(), BoltValue::String("r".to_string())),
            (
                "t_last".to_string(),
                BoltValue::Integer(started.elapsed().as_millis() as i64),
            ),
        ]),
    }
}
