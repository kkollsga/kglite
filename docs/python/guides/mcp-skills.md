# Authoring MCP skills

A **skill** is markdown that teaches an agent how and when to use a tool. At boot the MCP server attaches each active skill to the description of the tool(s) it applies to. The methodology travels with the tool, so you need no hand-rolled `instructions:` block.

A skill is usually a file next to the manifest. It can also be **carried inside the graph** as a `KgliteSkill` node, so a `.kgl` explains how to query itself.

This guide is the operator-facing spec for the skill surface. It covers:

- the three text channels and when to use each;
- where skills live;
- how a body reaches the agent (routing eagerly, body on demand);
- the frontmatter schema;
- how gating works;
- the size limits.

For manifests in general (tools, embedders, source roots) see {doc}`mcp-servers`.

## TL;DR

1. Opt in: `skills: true` in your manifest.
2. Drop a `<tool>.md` (or cross-tool `<topic>.md`) into a `<basename>.skills/` directory next to your manifest.
3. Give it frontmatter. You need `name` at minimum. Add `references_tools` (which tools it rides on) and usually `applies_when` (when it should be silent).
4. Put the **routing heuristic** in `description` (one paragraph: when to reach for this tool vs a sibling). Put the **how-to** in the markdown body.

The routing ships in the tool description. The agent fetches the body on demand with the `skill(name)` tool.

If you have no files to ship alongside the graph, store the same skill in the `.kgl` itself with `graph.set_skill(...)`. See [Skills carried in the graph](#skills-carried-in-the-graph).

If you are building a knowledge base rather than only a skill, start with the portable [Knowledge Bases](https://github.com/kkollsga/kglite/blob/main/KNOWLEDGE_BASES.md) guide. Then use {doc}`help-vault` for the runnable vault and MCP lifecycle.

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

Three manifest mechanisms put text in front of an agent. They have different lifecycles. Putting text in the wrong one is the usual mistake.

| Channel | Manifest key | When the agent sees it | Use for |
|---|---|---|---|
| **Init instructions** | `instructions:` | Once, in the MCP `initialize` handshake. Ages out of a long session's context. | One-time orientation that doesn't need to re-surface. Keep it short. |
| **Overview preamble** | `overview_prefix:` | Prepended to every **bare** `graph_overview()` call — re-surfaces each time. | A sticky reminder tied to schema discovery. |
| **Skills** | `skills:` + skill files or `KgliteSkill` nodes | The **routing** (`description`) rides the **tool description** of every tool the skill applies to, re-read whenever the agent inspects tools (`tools/list`). The **body** is handed over when the agent calls `skill(name)`. Gated per-graph by `applies_when`. | Per-tool and cross-tool methodology + routing. **This is where most guidance belongs.** |

Rule of thumb:

- Guidance *about how to use a tool* is a skill.
- *One-time setup context* is `instructions:`.
- *A reminder that should ride the schema* is `overview_prefix:`.

Two automatic prefixes also ride the init channel. They are **closed** to operator extension, so don't go looking for a hook:

- the **mode banner** (`[kglite-mode] …`) — states which conditional tools are registered for the active mode; re-surfaced on bare `graph_overview()`.
- the **batch-load hint** (`[kglite-batch-load-hint]`) — tells deferred-loading clients to bulk-fetch tool schemas. Include its marker in your `instructions:` to suppress it; you cannot add to it.

## Where skills come from (the layers)

Skills load from six layers. There is **one skill per `name`**. On a collision the **higher** layer wins, so you override a lower one by shipping a skill of the same `name`:

1. **bundled defaults** (lowest) — KGLite's compiled skills plus the framework defaults. The KGLite set is registered explicitly from `crates/kglite-mcp-server/skills/`; operators do not need to rebuild it.
2. **producer** — the embedder's own layer, registered once per server with `ServerExtensions::with_skills`. Provenance renders as `owned:producer`. See [Skills registered by the producer](#skills-registered-by-the-producer).
3. **graph-carried** — `KgliteSkill` nodes inside the served `.kgl`. Provenance `owned:graph`. See [Skills carried in the graph](#skills-carried-in-the-graph).
4. **manifest inline** — skill mappings written straight into the `skills:` list. KGLite adds nothing of its own here; the framework underneath supports it.
5. **operator-declared paths** — directories listed in `skills:`, in declaration order. Earlier paths win collisions with later paths.
6. **project layer** (highest) — a `<basename>.skills/` directory next to the manifest. For `my_graph_mcp.yaml` this is `my_graph_mcp.skills/`. The basename is the manifest's, not the graph's. This layer is optional when absent; a declared path that is absent is a boot error.

The ordering is one rule: **closer to the operator wins.** The three lowest layers are the binary's and the data's. The three highest are the operator's files.

- A graph skill named `cypher_query` **replaces** the bundled `cypher_query` methodology. That is the documented way to override framework guidance with something graph-specific.
- A producer skill of that name sits between the two.
- An operator's pack or `<basename>.skills/` file beats all of them. The operator is always the last word.

The bundled, producer and graph-carried layers surface only when skills are switched on for the deployment. The file layers are operator-declared and surface whenever the list is walked.

A manifest turns skills on with `skills: true` (see the next section). A producer turns them on by registering a layer, which is the only way a manifest-less deployment serves any.

## The `skills:` manifest value

`skills:` is polymorphic:

| Value | Meaning |
|---|---|
| absent / `false` / `null` | Skills **off**. No injection, `prompts/list` empty. *(Exception: a producer layer reads an absent `skills:` as silence rather than refusal — see below.)* |
| `true` | On: bundled defaults, the producer layer, the served graph's own skills, and the `<basename>.skills/` project layer. |
| `"./path"` | On, and also load skills from `./path` (relative to the manifest). |
| `[true, "./a", "./b"]` | List form: `true` = the bundled/default set, each string = an extra path. Use to combine the defaults with one or more operator packs. |

A declared path that does not exist **fails the boot**, naming what you wrote and where it resolved to. One bad entry fails the whole registry build. The alternative would be a server that runs with every skill silently gone (bundled ones included) while the graph tools answer normally.

The auto-detected project layer (layer 6) is the opposite: optional, and silent when absent.

`kglite-mcp-server --selftest` prints the number of skills the session actually serves. An opted-in deployment that resolved nothing is therefore visible without reading `prompts/list` by hand.

## Skills registered by the producer

A skill does not have to belong to a graph. A binary that **builds** its graphs (a workspace-mode code indexer, a domain ingester) emits the same node and edge shapes every time. The methodology for querying them is a property of the server, not of each artefact it produces.

`ServerExtensions::with_skills` registers that layer once, at boot, in **every** mode:

```rust
use kglite_mcp_server::{run_with_extensions, Delivery, ServerExtensions, SkillRecord};

let extensions = ServerExtensions::new().with_skills([SkillRecord {
    name: "code_graph_shapes".to_string(),
    description: "TRIGGER for any structural question about an indexed repository.".to_string(),
    body: "Every graph this server builds carries `:Function` and `:Class` nodes keyed by \
           their qualified name.\n".to_string(),
    references_tools: vec!["cypher_query".to_string()],
    delivery: Delivery::Lazy,
}]);
run_with_extensions(std::env::args_os(), extensions)?;
```

`SkillRecord` and `Delivery` are re-exported from `kglite_mcp_server`. An embedder needs no direct `kglite` dependency to describe its methodology. See {doc}`/rust/building-on-kglite` for the full embedder surface.

**It is its own opt-in.** Owned layers surface only when skills are enabled, and the server enables none without a manifest. That is exactly the shape a producer binary ships in. Calling `with_skills` therefore turns them on, and the manifest still overrules:

| Manifest | Result |
|---|---|
| none at all | bundled set + producer layer |
| present, no `skills:` key | bundled set + producer layer — silence is not a refusal |
| `skills: false` / `skills: null` | everything off, producer included; the boot line reports how many records it silenced |
| any explicit `skills:` value | used exactly as written, so a list without `true` leaves the producer layer off along with the bundled one |

Without a producer layer an unset `skills:` still means off, so no existing deployment gains a skill surface by upgrading.

**Always active.** `SkillRecord` carries no `applies_when:` predicate. Nothing in the layer depends on the shape of the active graph, so it resolves once and stays correct through every root swap.

**Validated at boot, and a bad record costs the boot.** These records are the embedder's own code, not data a `CREATE` could have written. A record that fails validation fails the boot naming itself, rather than being skipped the way a graph record is. The boot summary adds a `producer skills: N served (B B), M active as owned:producer` line beside the graph one.

## Skills carried in the graph

A skill does not have to be a file. A `.kgl` can carry its own methodology as nodes under the `KgliteSkill` system label. A graph you hand to someone then arrives knowing how to explain itself. You ship no skills directory alongside it and need no manifest beyond `skills: true`.

```python
graph.set_skill(
    "projects",
    "TRIGGER for any question about projects, their owners or their budgets.",
    body=methodology_markdown,
    references_tools=["cypher_query"],
)
graph.save("projects.kgl")
```

With `skills: true` in the manifest, `kglite-mcp-server --graph projects.kgl` now serves `projects` alongside its bundled methodology.

### The node

| Property | Type | Meaning |
|---|---|---|
| `name` | string | The key. One node per name; also the filename stem on export. A single path-safe token — no whitespace, `/`, `\\`, `:` or `.` traversal. |
| `description` | string | The routing heuristic, required. With lazy delivery this is all an agent sees before deciding to fetch the body. |
| `body` | string | The methodology, as markdown. At most 16,384 bytes. |
| `references_tools` | list of strings | Tools this skill attaches to, beyond its name match. |
| `delivery` | string | `"lazy"` (default) or `"eager"` — see [Delivery](#delivery-routing-now-body-on-demand). |

`KgliteSkill` is a **system label**: conventional, not enforced, and hidden from every surface that enumerates node types.

- `node_types()`, `schema()`, `describe()` and the Cypher `db.labels()` procedure all omit it.
- The counts printed beside those listings omit it too. Storing a skill never changes what the graph reports itself to be about.

It stays an ordinary node to Cypher. `MATCH (s:KgliteSkill) RETURN s.name` works, `MATCH (n) RETURN count(n)` counts it, and saves, exports and diffs carry it.

`KgliteRecipe`, the label graph-carried recipe queries use, behaves the same way. See {doc}`mcp-servers`.

### Managing them

`KnowledgeGraph` has six methods for skills. All of them validate what a hand-written `CREATE` could not:

| Call | Does |
|---|---|
| `list_skills()` | Every skill, sorted by name, bodies omitted. |
| `get_skill(name)` | One skill, body included. |
| `set_skill(name, description, body=..., references_tools=..., delivery=...)` | Create or replace. |
| `delete_skill(name)` | Remove one; `False` when there was nothing to remove. |
| `import_skills(path)` | Read one `.md` file, or every `.md` in a directory (non-recursive, sorted). |
| `export_skills(path)` | Write every skill to a directory as `<name>.md`. |

Import and export speak the same SKILL.md frontmatter dialect an MCP skills directory serves: `name`, `description`, `references_tools`, `delivery`, then the body. An existing skills folder imports unchanged. An exported directory can be hand-edited and imported back.

Import ignores frontmatter keys the graph has no property for (`applies_when`, `auto_inject_hint`, `applies_to`). A graph skill is not gated by a predicate.

`kglite skill <graph>` lists what a graph carries. `kglite skill <graph> <name>` prints one body raw. You can check what a server would serve without starting one. See {doc}`/operators/cli`.

### What the server does with them

- **Opt-in.** The layer surfaces only when the manifest's `skills:` contains `true`. A graph is *data*, and data that can rewrite tool descriptions without an operator saying so is a supply-chain surface.
- **`--graph` and `--watch` modes only.** Only those two modes have the graph open by the time skills are installed. The workspace modes build their graph on first activation, long after the prompt plane is frozen. The source-root and bare modes have no graph at all. In those modes the layer contributes nothing.
- **Validated per node, and one bad node costs only itself.** Every record is held to the same rules `set_skill` applies:
  - non-empty `name` and `description`;
  - `references_tools` a list of strings;
  - `delivery` one of the two tiers;
  - body within 16 KiB.

  A record that fails (a hand-written `CREATE` can store anything) is skipped with a warning naming the skill and the rule. Its siblings still load.
- **Named on the boot summary.** A graph that contributed anything adds a line like `graph skills: 3 served (9412 B), 2 active as owned:graph, 1 skipped: projects: description must not be empty` to stderr.
  - `--selftest` mirrors that stderr. It separately reports how many skills the session serves in total.
  - `--selftest` speaks MCP to a child process, and `prompts/list` carries names and descriptions only. Attribution and byte totals therefore reach you on the boot line, not from the check.
- **Re-resolved on a graph swap.** `reload_graph`, `load_graph` and `create_graph` rebuild the whole skill layer against the graph they just swapped in and send `tools/list_changed`. A `.kgl` rebuilt by another process serves its new methodology on the next reload rather than at the next restart. This is where skills differ from graph-carried recipes, whose catalogue is fixed for the session.

For a vault-backed deployment, `rebuild_graph` performs the same skill refresh after rebuilding the notes. Recipe queries, parameter schemas and descriptions remain fixed at boot. Adding, removing or renaming a recipe `tool:` also changes the boot-time router. The canonical per-mode update table is in
[VAULT.md §8](https://kglite.readthedocs.io/en/latest/reference/vault-format.html#skills-and-recipes-carried-in-the-vault).

## Frontmatter schema

Frontmatter is YAML between `---` fences. Unknown top-level metadata is ignored for compatibility. Unknown keys inside `applies_when` are rejected, so a misspelled gate cannot activate a skill.

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | **yes** | Skill identity. Also the tool it injects into by name match (so a skill named `cypher_query` rides the `cypher_query` tool). For a cross-tool skill, use a topic name that is *not* a tool name and rely on `references_tools`. |
| `description` | string | **yes** | The **routing heuristic** — TRIGGER/SKIP guidance. Injected into the tool description under a `## When to use` header (and sent to `prompts/list`). Keep it to a paragraph; it is never truncated. |
| `body` | (the markdown after the frontmatter) | no | The **methodology**. Injected under `## Methodology`, capped (see limits). |
| `references_tools` | list of strings | no | Extra tools this skill injects into, beyond its name match. **Load-bearing.** A `code_graph_analysis` skill with `references_tools: [cypher_query, graph_overview, explore]` rides all three. A skill that declares targets and whose every target is unregistered in this mode is dropped entirely — from `prompts/list`, from `skill()` and from the overview index. |
| `delivery` | `lazy` \| `eager` (default `lazy`) | no | How much of the skill a tool description carries — see [Delivery](#delivery-routing-now-body-on-demand). |
| `auto_inject_hint` | bool (default `true`) | no | `false` keeps the skill out of tool descriptions (it still appears in `prompts/list`). Use to ship a skill for prompt-only clients without bloating `tools/list`. |
| `applies_when` | mapping | no | Gating predicate — see below. Absent = always active. |

### `applies_when` — gate a skill to the graphs it fits

`applies_when` keeps a skill silent on graphs it doesn't apply to (e.g. a code-graph skill stays off a legal/finance domain graph). Predicates are AND-combined. An absent predicate is "satisfied".

Predicates are evaluated at server boot, after the graph and tool catalogue are ready. They are evaluated again whenever the graph the server serves changes identity:

- a `--graph` or `--watch` server swapping it with `reload_graph`, `load_graph` or `create_graph`;
- a workspace server activating a root with `set_root_dir` or `repo_management`. This one matters because a workspace mode has **no graph at boot**. Every `graph_has_node_type:` predicate is false there until the first activation. That is why the bundled `code_graph_analysis`, `code_graph_views` and `read_code_source` skills only appear once a root has been built.

Either way the whole registry is re-resolved against the new graph and the client is sent `tools/list_changed`.

Three events do **not** re-evaluate anything. Prompt registration and the injected tool descriptions keep the answer they had until the next swap or restart.

- Mutating the graph **in place**.
- The per-call freshness re-read that notices the served file changed on disk.
- The watcher's lazy rebuild of an already-activated root.

The last two replace a graph with one built from the same source, so the node types the last resolution saw are the node types it still sees. Call `reload_graph`, or re-activate the root, to pick up a rebuilt file's skills.

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

`graph_has_node_type:` answers `false` for the system labels `KgliteSkill` and `KgliteRecipe` whatever the graph holds. Those labels are hidden from every type enumeration, so an agent could never discover the shape being gated on.

A **graph-carried** skill carries no `applies_when` at all. The node has no property for one, and an import drops the key. Gate those by what you store in the graph, not by a predicate.

### Load-bearing vs decorative keys

Only the keys in the table above affect behavior. Unknown top-level keys remain decorative and are ignored, including these keys found in older bundled files. Unknown keys nested under `applies_when` are an error. The affected external skill is skipped with a path-and-error warning.

- `applies_to` (version floors like `mcp_methods: ">=0.3.36"`) — **decorative**. Activation is *not* gated on it. `applies_when` + the layer the file lives in are what gate a skill.
- `references_arguments`, `references_properties` — **decorative**.

`references_tools` *is* read. Don't confuse it with the decorative `references_*` keys.

## How a skill reaches the agent

For each active skill with `auto_inject_hint: true`, a block is appended to the description of every tool it attaches to: its name-match tool **and** every tool in `references_tools`. On the default **lazy** tier the block is routing plus a pointer:

```text
<the tool's own description>

<!-- mcp-skill:find_papers -->

## When to use

<the skill's `description`>

Load the full methodology with skill("find_papers") before first use.
```

On the **eager** tier the body itself replaces the pointer, under a `## Methodology` heading.

A tool can carry several skills (its own + any that reference it). Each is appended once. The block rides `tools/list`, which **every** MCP client exposes to the agent.

## Delivery: routing now, body on demand

Skills are delivered **lazily by default** (mcp-methods 0.4.11). The routing `description` still travels eagerly. It is small, it is what the agent reads to decide whether the body is worth having, and it is exempt from the body's size caps. The body does not travel eagerly. A `skill` tool, registered whenever skills are on, returns it verbatim on request.

Why: a skill body used to be copied into every tool it referenced, at `tools/list` time, before any tool was used. One methodology skill across five tools was five copies. A plain domain deployment paid roughly 15–22 KB of description text up front. A code-graph deployment with recipes paid roughly 60 KB. The agent might never need those bodies.

**The `skill` tool.** `skill(name)` returns that skill's body, whole.

- It is exempt from the response budget, so a body is never handed back as a truncated preview. The 16 KiB cap at load is what makes the exemption safe.
- An unknown or inactive name is refused with the list of active skills.
- What a session has loaded is remembered for as long as the session keeps making tool calls. A new session, or one that has made no tool call for ten minutes, starts empty.

**The nudge.** The first call in a session to a tool that advertises a lazy skill the agent has not fetched gets one extra footer line naming it (`Skill "projects" applies to this tool and has not been loaded this session — call skill("projects")`). It is silent afterwards. It comes back only if the skill's body changed under the agent or the session went quiet past that ten-minute window.

The footer shares the text part of a typed tool's reply with the JSON, separated by a blank line. A scripted consumer of such a tool expects the trailer: `json.loads(part.text)` raises "Extra data" on a tool's first call. Decode the leading value with `json.JSONDecoder().raw_decode(part.text)` instead.

```{warning}
**The nudge does not fire for `cypher_query`.** The footer is composed inside the framework's typed-tool dispatch. `cypher_query` is the one KGLite tool registered as a raw route (it needs a custom output schema), and the framework offers no way to opt in. A lazy skill that targets *only* `cypher_query` therefore never nudges. The pointer in the tool description is the agent's only cue.

This is a known upstream limitation in mcp-methods 0.4.11, reported and pinned by a test that goes red when it is fixed. Until then, give a `cypher_query` skill a second target (`graph_overview` is the usual one) if you want the reminder.
```

**When to choose `eager`.** Choose it only when the body has to shape the **first** call's arguments. There is no result to learn from, so a pointer arrives too late.

The bundled `cypher_query` skill is the one KGLite ships eager, for exactly that reason: Cypher has to be written correctly before it returns anything. Every other bundled skill is lazy. So is every graph-carried skill that does not say otherwise.

Bare `graph_overview()` ends with an index of what this server serves, one line per active skill with its tier. An agent that has not inspected `tools/list` can still see what is on offer:

```xml
<skills count="2" get-via="skill(name)">
cypher_query [eager] — Run Cypher against the active knowledge graph.
projects [lazy] — TRIGGER for any question about projects.
</skills>
```

The tier is load-bearing for the reader: a `[lazy]` line is an invitation to call `skill(name)`, not a summary of something already in hand. The index is refreshed on every graph swap, so it never advertises a skill the session has stopped serving. Focused `graph_overview(...)` calls omit it.

```{warning}
Do **not** rely on `prompts/get` for agentic retrieval. Skills are also registered as MCP prompts. The `prompts/*` plane was designed for human-invoked slash commands in chat UIs, and it is **not exposed to the agent** in Claude Code / Claude Desktop / Cursor / Continue. The tool-description injection above is the channel agents actually read. The prompt registration is a fallback for the rare custom integration that surfaces prompts.
```

## Size limits

- A complete skill file above **16,384 bytes** is rejected, not truncated. The limit includes frontmatter, `description`, and markdown body. `set_skill` applies the same 16 KiB ceiling to a graph-carried `body`, and refuses rather than truncating.
- Files above the **4,096-byte** soft target still load, with a warning.
- The resolved session total has a **65,536-byte** soft limit that warns without dropping skills.

Lazy delivery changes what those numbers cost, not what they are. A skill's body counts against the ceilings whichever tier it is on.

- An **eager** body is paid for in `tools/list`, once per tool it attaches to.
- A lazy body is paid for once, by the agent that asks for it.

Sizing a body for the tool plane is therefore an `eager`-only concern.

Keep each file focused. If it exceeds the hard limit, split the methodology into independently routed skills.

## Worked example: a cross-tool orchestration skill

The bundled `code_graph_analysis` skill is the canonical cross-tool pattern. It is named after no tool, gated to code graphs, and attached to several tools at once:

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

On a code graph it rides all five tool descriptions. On a domain graph (no `Function`/`Class`) it is silent. That is the whole point of skills over `instructions:`: gated, per-tool, re-surfacing, and zero hand-maintenance.

## Worked example: a graph that ships its own skill and queries

**A skill that names an operation should ship the operation.** A methodology that says "find a function's callers" is worth much more when the exact, parameter-checked query is in the same file, reachable as `run_recipe_query("code_review", "callers_page", ...)`. The agent does not have to rewrite Cypher the graph's author already got right. Graph-carried recipes are the other half of this feature. See the `extensions.cypher_recipes` section of {doc}`mcp-servers`.

[`examples/code_review_graph_skills.py`](https://github.com/kkollsga/kglite/blob/main/examples/code_review_graph_skills.py) builds a small code graph, writes three `code_review` recipes and one `code_review` skill that names them, saves the `.kgl`, and prints what an MCP server would then serve.

To see the whole surface, run it and point `kglite-mcp-server --graph` at the file it writes, with `skills: true` in the manifest. You see:

- `list_recipe_queries` / `run_recipe_query` registered from the graph alone;
- `code_review [lazy]` in the bare overview index;
- the body arriving on `skill("code_review")`.
