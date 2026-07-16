# Authoring MCP skills

A **skill** is a markdown file that teaches an agent how and when to use a tool.
At boot the MCP server injects each active skill into the description of the
tool(s) it applies to, so the methodology travels with the tool — no
hand-rolled `instructions:` block required.

This guide is the operator-facing spec for the skill surface: the three text
channels and when to use each, where skills live, the frontmatter schema (and
which keys are load-bearing), how gating works, and the size limits. For
manifests in general (tools, embedders, source roots) see
{doc}`mcp-servers`.

## TL;DR

1. Opt in: `skills: true` in your manifest.
2. Drop a `<tool>.md` (or cross-tool `<topic>.md`) into a `<basename>.skills/`
   directory next to your manifest.
3. Give it frontmatter — at minimum `name`, plus `references_tools` (which
   tools it rides on) and usually `applies_when` (when it should be silent).
4. Put the **routing heuristic** in `description` (one paragraph: when to reach
   for this tool vs a sibling) and the **how-to** in the markdown body.

```yaml
# my_graph_mcp.yaml
name: my_graph
skills: true          # turn the skill system on
```

```markdown
<!-- my_graph.skills/find_papers.md -->
---
name: find_papers
description: "TRIGGER when the user asks to find / list / filter papers by
  author, year, or topic. SKIP for citation-graph traversal (that's a plain
  cypher_query MATCH on the CITES edge)."
references_tools: [cypher_query]
applies_when:
  graph_has_node_type: [Paper]
---

# Finding papers

Papers carry `title`, `year`, `venue`, and an `author` list...
```

## The three text channels — pick the right one

Three manifest mechanisms put text in front of an agent. They have different
lifecycles; putting text in the wrong one is the usual mistake.

| Channel | Manifest key | When the agent sees it | Use for |
|---|---|---|---|
| **Init instructions** | `instructions:` | Once, in the MCP `initialize` handshake. Ages out of a long session's context. | One-time orientation that doesn't need to re-surface. Keep it short. |
| **Overview preamble** | `overview_prefix:` | Prepended to every **bare** `graph_overview()` call — re-surfaces each time. | A sticky reminder tied to schema discovery. |
| **Skills** | `skills:` + skill files | Injected into the **tool description** of every tool a skill applies to. Re-read whenever the agent inspects tools (`tools/list`). Gated per-graph by `applies_when`. | Per-tool and cross-tool methodology + routing. **This is where most guidance belongs.** |

Rule of thumb: if the guidance is *about how to use a tool*, it's a skill. If
it's *one-time setup context*, it's `instructions:`. If it's *a reminder that
should ride the schema*, it's `overview_prefix:`.

Two automatic prefixes also ride the init channel and are **closed** to
operator extension (documented here so you don't go looking for a hook):

- the **mode banner** (`[kglite-mode] …`) — states which conditional tools are
  registered for the active mode; re-surfaced on bare `graph_overview()`.
- the **batch-load hint** (`[kglite-batch-load-hint]`) — tells deferred-loading
  clients to bulk-fetch tool schemas. Include its marker in your
  `instructions:` to suppress it; you cannot add to it.

## Where skills come from (the layers)

Skills load from four layers. On a name collision the **higher** layer wins, so
you can override a bundled skill by shipping one of the same `name`:

1. **kglite-bundled** (lowest) — compiled into the `kglite-mcp-server` binary
   from `crates/kglite-mcp-server/skills/` (registered explicitly in
   `main.rs`). The bundled set today: `cypher_query`, `graph_overview`,
   `read_code_source`, `save_graph`, `explore`, `code_graph_analysis`,
   `code_graph_views`. Adding to this set is a kglite change; **operators add
   their own skills via the project layer below — no rebuild.**
2. **framework defaults** — from the mcp-methods crate.
3. **project layer** — a `<basename>.skills/` directory **next to your
   manifest**. For `my_graph_mcp.yaml` that's `my_graph_mcp.skills/`. This is
   the operator's home: drop skill files here, no code changes.
4. **operator-declared paths** (highest) — extra directories listed in
   `skills:` (see next section).

## The `skills:` manifest value

`skills:` is polymorphic:

| Value | Meaning |
|---|---|
| absent / `false` / `null` | Skills **off**. No injection, `prompts/list` empty. |
| `true` | On: kglite-bundled + framework defaults + the `<basename>.skills/` project layer. |
| `"./path"` | On, and also load skills from `./path` (relative to the manifest). |
| `[true, "./a", "./b"]` | List form: `true` = the bundled/default set, each string = an extra path. Use to combine the defaults with one or more operator packs. |

## Frontmatter schema

Frontmatter is YAML between `---` fences. mcp-methods parses exactly these keys;
**everything else is ignored** (see "load-bearing vs decorative" below).

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | **yes** | Skill identity. Also the tool it injects into by name match (so a skill named `cypher_query` rides the `cypher_query` tool). For a cross-tool skill, use a topic name that is *not* a tool name and rely on `references_tools`. |
| `description` | string | no | The **routing heuristic** — TRIGGER/SKIP guidance. Injected into the tool description under a `## When to use` header (and sent to `prompts/list`). Keep it to a paragraph; it is never truncated. |
| `body` | (the markdown after the frontmatter) | no | The **methodology**. Injected under `## Methodology`, capped (see limits). |
| `references_tools` | list of strings | no | Extra tools this skill injects into, beyond its name match. **Load-bearing.** A `code_graph_analysis` skill with `references_tools: [cypher_query, graph_overview, explore]` rides all three. |
| `auto_inject_hint` | bool (default `true`) | no | `false` keeps the skill out of tool descriptions (it still appears in `prompts/list`). Use to ship a skill for prompt-only clients without bloating `tools/list`. |
| `applies_when` | mapping | no | Gating predicate — see below. Absent = always active. |

### `applies_when` — gate a skill to the graphs it fits

`applies_when` keeps a skill silent on graphs it doesn't apply to (e.g. a
code-graph skill stays off a legal/finance domain graph). Predicates are
AND-combined; an absent predicate is "satisfied". Re-evaluated against the
**live** graph on each request, so it tracks post-boot mutations (a workspace
activating a repo).

| Predicate | True when |
|---|---|
| `graph_has_node_type: [A, B]` | the graph has **any** of these node labels |
| `graph_has_property: {node_type: T, prop_name: p}` | nodes of type `T` carry property `p` |
| `tool_registered: NAME` | tool `NAME` is registered in this mode |
| `extension_enabled: NAME` | manifest extension `NAME` is on |

```yaml
applies_when:
  graph_has_node_type: [Function, Class]   # code graphs only
```

### Load-bearing vs decorative keys

Only the keys in the table above are read. Several keys appear in older
bundled files for human documentation but the loader **ignores** them — copying
them into your skill does nothing:

- `applies_to` (version floors like `mcp_methods: ">=0.3.36"`) — **decorative**.
  Activation is *not* gated on it; `applies_when` + the layer the file lives in
  are what gate a skill.
- `references_arguments`, `references_properties` — **decorative**.

(`references_tools` *is* read — don't confuse it with the decorative
`references_*` keys.)

## How a skill reaches the agent

For each active skill with `auto_inject_hint: true`, its routing + methodology
are appended to the description of every tool it attaches to — its name-match
tool **and** every tool in `references_tools`:

```text
<the tool's own description>

<!-- mcp-skill:find_papers -->

## When to use

<the skill's `description`>

## Methodology

<the skill's body>
```

A tool can carry several skills (its own + any that reference it); each is
injected once. This rides `tools/list`, which **every** MCP client exposes to
the agent.

```{warning}
Do **not** rely on `prompts/get` for agentic retrieval. Skills are also
registered as MCP prompts, but the `prompts/*` plane was designed for
human-invoked slash commands in chat UIs and is **not exposed to the agent**
in Claude Code / Claude Desktop / Cursor / Continue. The tool-description
injection above is the channel agents actually read; the prompt registration
is a fallback for the rare custom integration that surfaces prompts.
```

## Size limits

- The injected **body** is capped at **16 KB** (hard) with a **4 KB** soft
  target — keep bodies tight; 16 KB × N tools is real context cost on every
  `tools/list`. Past 16 KB the body is truncated with a marker.
- The **`description`** (routing) is small by design and never truncated — it
  is the highest-value half, so lead with it.

If your methodology is longer than the cap, that's a signal to split it: a
focused routing `description` plus a tight body beats a wall of text the agent
skims.

## Worked example: a cross-tool orchestration skill

The bundled `code_graph_analysis` skill is the canonical cross-tool pattern —
named after no tool, gated to code graphs, attached to several tools at once:

```markdown
---
name: code_graph_analysis
description: "TRIGGER for any structural question about a codebase — what
  calls / defines / extends / imports X... Map structure with the graph FIRST
  (graph_overview → cypher_query → explore), then drop to grep/read_source
  only to confirm a detail. Never grep to discover what the graph encodes."
references_tools: [cypher_query, graph_overview, explore, grep, read_source]
applies_when:
  graph_has_node_type: [Function, Class]
---

# Code-graph analysis: the sequencing strategy
...
```

On a code graph it rides all five tool descriptions; on a domain graph
(no `Function`/`Class`) it is silent. That is the whole point of skills over
`instructions:`: gated, per-tool, re-surfacing, and zero hand-maintenance.
