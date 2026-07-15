---
name: kglite-code-review
description: Use when reviewing a code change or answering structural questions about a codebase, including definitions, callers, dependencies, routes, affected tests, and history across git revisions. Builds a local Cypher-queryable code graph, uses it alongside the diff and literal search, and verifies every finding against exact source lines.
---

# KGLite code review

Use KGLite for structural evidence during review. The graph complements the git
diff, source reading, and literal-text search; it does not replace them.

## Review workflow

1. Inspect the diff and repository guidance first. Identify changed symbols and
   the base/head revisions.
2. Build or refresh the graph without executing repository code:

   ```bash
   kglite code-tree build . --output .kglite/code-review.kgl --format json
   ```

   For a committed comparison, use one graph spanning both revisions:

   ```bash
   kglite code-tree build . --revs '<base>' '<head>' \
     --output .kglite/code-review.kgl --format json
   ```

3. Always discover the actual schema before writing Cypher:

   ```bash
   kglite describe .kglite/code-review.kgl --connections --cypher
   ```

4. Query the smallest structural question that can confirm or reject a review
   hypothesis. Use JSON for agent parsing:

   ```bash
   kglite query .kglite/code-review.kgl '<cypher>' --format json
   ```

5. Open every implicated file and verify the behavior at exact lines. Report
   only findings supported by source evidence. Do not infer runtime behavior
   from an edge alone.

6. Before reusing an artifact, check freshness:

   ```bash
   kglite code-tree status --output .kglite/code-review.kgl --format json
   ```

See [queries.md](references/queries.md) for query patterns,
[public-repositories.md](references/public-repositories.md) for safe public-repo
review, and [mcp-upgrade.md](references/mcp-upgrade.md) for the persistent MCP
workflow.

## Honesty rules

- Never invent labels, properties, or connection types: `describe()` first.
- Treat unresolved or missing graph edges as absence of evidence, not proof.
- Quote paths and revisions passed through the shell.
- Never build, import, or execute code from a repository merely to review it.
- Use grep/ripgrep for exact tokens and the graph for relationships and impact.
