# Example: a parameterised Cypher tool

A `tools[].cypher` entry that wraps a parameterised query as a
first-class MCP tool. The agent sees `find_decisions_by_year` as a
regular tool whose published MCP input schema describes a `year` argument.

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

The tool schema tells MCP clients that `year` is a required integer between
`1900` and `2100`. Whether a client validates that schema before dispatch is
client-dependent; KGLite's legacy manifest-tool path publishes the schema but
does not enforce it. The Cypher template runs as

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

- **Client-dependent**: an MCP client may reject a value that does not match
  the published schema before dispatch. Raw callers can bypass client-side
  validation.
- **Runtime**: a missing `$param`, an incompatible value, or a Cypher engine
  error (graph mutation in read-only
  mode, syntax error in the template) surfaces as
  `Cypher error: <engine message>` in the response body.
- **Boot**: KGLite does not validate the legacy `parameters:` schema or compare
  its properties with the template's `$param` references.
