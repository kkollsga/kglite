# Bounded agent responses

KGLite can return a compact response that identifies the complete executed result, describes omitted evidence, and supplies a way to retrieve the needed part. The budget covers the serialized response, including data, navigation, warnings, metadata, escaping, and follow-up actions.

Presentation does not change query semantics. A Cypher `LIMIT`, an executor row limit, and a response byte budget are separate boundaries. Expansion recovers content from the completed retained result; it never reruns the query. It cannot recover rows that the query or executor did not produce.

## MCP response controls

Discover the response-control field and expansion tool through `tools/list`. The default maximum is 16,384 serialized bytes. A custom maximum must be at least 4,096 bytes, and cannot be combined with full mode. For example, when discovery publishes `_response` for `cypher_query`:

```json
{"name":"cypher_query","arguments":{"query":"MATCH (n) RETURN n.name ORDER BY n.name","_response":{"max_bytes":4096}}}
```

Do not hard-code the expansion tool name or result ID. A bounded response contains an actual `next.selected_value` action with the collision-safe tool name and session-local result ID. It also contains selector targets under its domain guidance. To retrieve a target:

1. Copy the complete `next.selected_value` action.
2. Keep its `name` and `arguments.result_id` unchanged.
3. Copy only the target's `json_pointer`, `offset`, and `response` into the action's `path`, `offset`, and `response` arguments.
4. Dispatch the resulting action.

Each supplied target requests a bounded 4,096-byte response. Use an explicitly discovered full response only when the complete selected value should be inline. JSON Pointer uses `~1` for `/` and `~0` for `~`. Offsets continue an array, an ordered object-field page, or a Unicode string.

The retained `navigation` directory lists available row and column counts, ordered columns, selectable sections, omitted column names, and observed nested values. The initial guidance points to `/navigation`, so the directory remains retrievable if it does not fit in the first preview. KGLite advertises only paths and values observed in the result; it does not infer semantic groups.

MCP retention is in memory and local to the current MCP session. A retained result is available for up to ten minutes on access and may be evicted when the session reaches 32 entries or 32 MiB. Full expansion restores the completed MCP result, including structured content, content blocks, metadata, and error status.

## CLI agent output

Agent output is explicit, so JSON and CSV keep their complete machine-output contracts:

```console
kglite query app.kgl "MATCH (n:Evidence) RETURN n.id ORDER BY n.id" --format agent
kglite query app.kgl "MATCH (n:Evidence) RETURN n.id ORDER BY n.id" --format agent --response-max-bytes 32768
kglite query app.kgl "MATCH (n:Evidence) RETURN n.id ORDER BY n.id" --format agent --response-full
kglite write app.kgl "CREATE (:Task {id: 't1'})" --format agent
```

`--response-max-bytes` and `--response-full` are mutually exclusive and valid only with `--format agent`. Invalid controls are rejected before execution, including for writes. The default and minimum are 16,384 and 4,096 serialized bytes respectively. Errors use the same structured response and exit nonzero; agent mode does not duplicate the error or query warnings on stderr.

A bounded result contains executable commands. Copy one instead of constructing a cache path:

```console
kglite response expand RESULT_ID --path /rows/37/0 --offset 0 --response-max-bytes 4096
kglite response expand RESULT_ID --path /rows/37/0 --response-full
```

Expansion works in a later process and from another working directory. It reads the retained canonical result and does not open the graph, so it still works if the graph changed or was removed. Missing, expired, evicted, corrupt, or unknown handles fail explicitly and never cause query replay.

The JSONL session protocol remains complete by default. An individual query or
write opts in with
`"format":"agent"` and may add `"response":{"max_bytes":4096}` or
`"response":{"mode":"full"}`. Its `response_expand` operation uses the
same handle, JSON Pointer, offset, and response controls, with
`{"op":"help"}` publishing the exact request shape.

## CLI retention and cleanup

CLI agent mode uses a private per-user disk cache. Its versioned envelope stores ordered `columns`, positional `rows`, `diagnostics`, `coverage`, `identity`, `operation`, `representation`, and `navigation`. Positional rows keep duplicate column names unambiguous: `/rows/37/0` selects by column position.

The cache is partitioned by canonical workspace for provenance, with global limits of 32 entries and 32 MiB across all workspaces. Entries expire ten minutes after creation when later cache activity runs cleanup. There is no background-deletion promise for an unused cache. Remove all retained responses explicitly with:

```console
kglite response purge --all
```

Retention is best effort. Admission may evict the oldest entries before a new entry is published. If publication fails, KGLite exposes no partial handle and returns the complete operation result with a retention warning; a successful mutation remains successful and is never replayed. The CLI owns expired-entry, oldest-first, stale-temporary-file, and purge cleanup.

## Reading coverage

The envelope reports `executed_rows`, any executor row limit, all literal limits observed in the parsed query, and a conservative `literal_limit_status`. That status says only whether the executed row count matches an observed literal limit. A literal limit may be nested, and database-wide population is reported as `unknown`.

For example, if `LIMIT 120` returns only write callers, the result establishes that its 120 executed rows contain no read callers. It does not establish that the database contains no read callers. Expansion can inspect the retained 120 rows; broader coverage requires a separately reviewed and executed query.
