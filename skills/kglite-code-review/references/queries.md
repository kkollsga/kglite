# Code-review query patterns

Run `kglite describe <graph> --connections --cypher` first and adapt these
patterns to the labels and properties it reports.

Find a symbol before asking about its relationships:

```cypher
MATCH (n)
WHERE n.name = '<symbol>' OR n.qualified_name = '<qualified_symbol>'
RETURN labels(n) AS labels, n.qualified_name AS symbol,
       n.file_path AS file, n.line_number AS line
LIMIT 20
```

After confirming the connection name, inspect direct callers:

```cypher
MATCH (caller)-[:CALLS]->(target)
WHERE target.qualified_name = '<qualified_symbol>'
RETURN caller.qualified_name AS caller,
       caller.file_path AS file, caller.line_number AS line
ORDER BY file, line
```

Find tests structurally connected to a changed symbol:

```cypher
MATCH (test)-[*1..4]->(changed)
WHERE test.is_test = true AND changed.qualified_name = '<qualified_symbol>'
RETURN DISTINCT test.qualified_name AS test,
       test.file_path AS file, test.line_number AS line
LIMIT 100
```

For a multi-revision graph, prefer the built-in delta procedure shown by
`describe()`:

```cypher
CALL rev_diff({from: '<base>', to: '<head>'})
YIELD bucket, type, qualified_name, name, file, line
RETURN bucket, type, qualified_name, name, file, line
ORDER BY bucket, type, qualified_name
```

CLI one-shot queries currently take literal Cypher rather than a separate
parameter map. Replace the angle-bracket placeholders only with trusted git or
source identifiers and escape Cypher string quotes. For untrusted values, use
the JSONL session API's parameter support.
