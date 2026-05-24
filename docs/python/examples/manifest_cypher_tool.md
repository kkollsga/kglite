# Example: a parameterised Cypher tool

A `tools[].cypher` entry that wraps a parameterised query as a
first-class MCP tool. The agent sees `find_decisions_by_year` as a
regular tool with a JSON-Schema-validated `year` argument.

## Manifest

```yaml
# norwegian_law_mcp.yaml — co-located with norwegian_law.kgl
name: Norwegian Law
instructions: |
  Norwegian legal corpus. Use cypher_query for ad-hoc questions and
  find_decisions_by_year / find_law_section_citations for the common
  lookups.

source_root: ./data

tools:
  - name: find_decisions_by_year
    description: All Supreme Court decisions published in a given year.
    parameters:
      type: object
      properties:
        year:
          type: integer
          minimum: 1900
          maximum: 2100
          description: 4-digit publication year (e.g. 2024).
      required: [year]
    cypher: |
      MATCH (d:CourtDecision)
      WHERE d.year = $year
      RETURN d.case_id AS case_id, d.title AS title, d.url AS url
      ORDER BY d.case_id
```

## What the agent sees on `tools/list`

The tool registers alongside the bundled `cypher_query`,
`graph_overview`, `ping`, and the source tools auto-registered by
`source_root: ./data`:

```
- cypher_query
- graph_overview
- ping
- read_source / grep / list_source
- find_decisions_by_year
```

## Calling it

The agent calls `find_decisions_by_year` with a typed argument:

```json
{"name": "find_decisions_by_year", "arguments": {"year": 2024}}
```

The MCP client validates `year` against the schema (rejects strings,
floats outside `1900..2100`) before the tool dispatch reaches
kglite. The Cypher template runs as

```cypher
MATCH (d:CourtDecision) WHERE d.year = $year
RETURN d.case_id AS case_id, d.title AS title, d.url AS url
ORDER BY d.case_id
```

with `$year` bound to the integer `2024` via kglite's typed parameter
binding — no string interpolation, no injection surface.

## Response shape

Inherits the `cypher_query` inline format. With 5 rows:

```
5 row(s):
case_id	title	url
'HR-2024-1234-A'	'Tvist om eierskap til ...'	'https://lovdata.no/...'
'HR-2024-1567-S'	'Skattesak — ...'	'https://lovdata.no/...'
'HR-2024-1890-A'	'Strafferettslig sak — ...'	'https://lovdata.no/...'
'HR-2024-2103-A'	'Avtalerettslig tvist — ...'  'https://lovdata.no/...'
'HR-2024-2456-A'  'Konkurssak — ...'           'https://lovdata.no/...'
```

If the cypher needs to return a large result set, end the template
in `RETURN ... FORMAT CSV` and pair the manifest with
`extensions.csv_http_server:` — the tool then returns a localhost URL
instead of inlining (see `extensions.csv_http_server` in the
reference docs).

## Failure modes

- **Boot**: a `$param` in the template not present in
  `parameters.properties` fails with
  `ERROR: <path>: cypher tool 'find_decisions_by_year': cypher references $params [...] not declared in parameters.properties`.
- **Boot**: a malformed JSON Schema (missing `type`, unknown
  type-flavour) fails with `ERROR: <path>: cypher tool 'X':
  invalid parameters schema: ...`.
- **Runtime**: the MCP client rejects a value that doesn't match the
  schema before the tool dispatches; the agent sees a structured
  error, not a Cypher error.
- **Runtime**: a Cypher engine error (graph mutation in read-only
  mode, syntax error in the template) surfaces as
  `Cypher error: <engine message>` in the response body.
