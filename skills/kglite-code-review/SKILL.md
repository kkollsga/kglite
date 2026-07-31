---
name: kglite-code-review
description: Use when reviewing a code change or answering structural questions about a codebase, including definitions, callers, dependencies, routes, affected tests, and history across git revisions. Builds a local Cypher-queryable code graph, uses it alongside the diff and literal search, and verifies every finding against exact source lines.
---

# KGLite code review

Use KGLite for structural evidence during review. The graph complements the git
diff, source reading, and literal-text search; it does not replace them.

## Review workflow

> **Prerequisite:** the `codingest` CLI must be on PATH — `pip install codingest`
> (bundles the CLI alongside the Python API) or `cargo install codingest-cli`.


1. Inspect the diff and repository guidance first. Identify changed symbols and
   the base/head revisions.
2. Build or refresh the graph without executing repository code:

   ```bash
   codingest build . --output .kglite/code-review.kgl --format json
   ```

   For a committed comparison, use one graph spanning both revisions:

   ```bash
   codingest build . --revs '<base>' '<head>' \
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
   codingest status --output .kglite/code-review.kgl --format json
   ```

See [queries.md](references/queries.md) for query patterns,
[public-repositories.md](references/public-repositories.md) for safe public-repo
review, and [mcp-upgrade.md](references/mcp-upgrade.md) for the persistent MCP
workflow.

## What counts as a finding

The graph and the diff tell you *what changed*. They do not tell you what is
**wrong**, and only what is wrong belongs in a review.

A finding names a concrete failure: the inputs or state, and the wrong
behaviour they produce — wrong result, crash, data loss, security hole, a
broken contract with a caller or a persisted file, a *measured* performance
regression, or a check that cannot fail. **If you cannot write down the case
that breaks, you do not have a finding.**

Do not report, at any confidence or severity:

- Structure and organisation preferences — "extract this", "split this file",
  "this belongs elsewhere", "this would read better as X".
- Naming, ordering, formatting, comment density, idiom preferences.
- "Could be simplified" or "is repetitive", absent a defect it causes.
- Inconsistency with surrounding code, unless the inconsistency is itself the
  defect.
- Speculation — "this won't scale", "this will be hard to extend" — without a
  present, reachable failure.
- Performance opinions with no measurement behind them.
- Anything a formatter, linter, type checker or compiler already decides.

**Exception: a rule the project already declared.** Citing a documented
constraint from the repository's own guidance is legitimate *when you name the
rule and the line that violates it*. The test is whether the rule existed
before you read the diff.

"No findings" is a valid and often correct review. A reviewer that always
returns an action list is measuring its own appetite for restructuring, not the
code — and it buries real defects among preferences.

Structural evidence is especially prone to this failure. An edge showing that
two modules are coupled, or that a function has many callers, is a *fact*, not
a defect. Use it to test a hypothesis about breakage, never as grounds for a
reorganisation suggestion.

## Honesty rules

- Never invent labels, properties, or connection types: `describe()` first.
- Treat unresolved or missing graph edges as absence of evidence, not proof.
- Quote paths and revisions passed through the shell.
- Never build, import, or execute code from a repository merely to review it.
- Use grep/ripgrep for exact tokens and the graph for relationships and impact.
