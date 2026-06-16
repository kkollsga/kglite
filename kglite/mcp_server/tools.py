"""Built-in MCP tools: cypher_query, graph_overview, save_graph.

Mirrors `crates/kglite-mcp-server/src/tools.rs` with the same handler
shapes, status messages, and 15-row inline cap. Differences:

- We call `kglite.KnowledgeGraph.cypher(...)`/`.describe(...)`/`.save(...)`
  directly (kglite Python API). The GIL is released inside `cypher()`
  for execution, so the wrapping overhead is just Python attr-lookup
  and argument marshalling — sub-microsecond per call.
- Mutating queries route to `graph.cypher(...)` too (the Python API
  handles read/write internally, unlike the Rust shim which had to
  branch on `is_mutation_query`).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from typing import Any

import kglite
from kglite.mcp_server.csv_http import CsvHttpConfig, write_csv
from kglite.mcp_server.preprocessor import Preprocessor
from kglite.mcp_server.preprocessor import apply as apply_preprocessor

log = logging.getLogger("kglite.mcp_server.tools")

NO_GRAPH = "No active graph. Pass --graph X.kgl, or activate one via repo_management('org/repo')."


@dataclass
class ActiveGraph:
    """The kglite graph + its source path (for save_graph)."""

    graph: kglite.KnowledgeGraph
    source_path: Path | None


class GraphState:
    """Shared active-graph slot, swappable at runtime (workspace mode
    re-activates the slot when the operator switches repos). The lock
    serialises swap + read access; tool calls hold the read view for
    the duration of the call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: ActiveGraph | None = None
        # Deferred-rebuild slot. The watch callback tags the directory
        # it would rebuild here (cheap — just a pointer write under
        # the lock). Each MCP tool entry calls
        # `ensure_code_tree_fresh()` which atomically takes the slot
        # and rebuilds. N FS events between two MCP calls collapse to
        # one rebuild (slot just holds the latest target).
        self._pending_rebuild: Path | None = None

    def tag_code_tree_dirty(self, target: Path) -> None:
        """Tag `target` for rebuild on the next MCP tool call.
        Non-blocking: just sets the slot under the lock."""
        log.debug("code_tree tagged for rebuild: %s", target)
        with self._lock:
            self._pending_rebuild = target

    def ensure_code_tree_fresh(self) -> None:
        """Rebuild the code-tree if the watcher tagged it dirty since
        the last call. Called by each MCP tool entry that reads the
        graph. No-op when nothing's pending.

        On rebuild failure, logs + clears the slot. The next FS change
        re-tags. (Avoids tight rebuild loops if the source dir is
        permanently broken.)
        """
        with self._lock:
            target = self._pending_rebuild
            self._pending_rebuild = None
        if target is None:
            return
        log.info("rebuilding code_tree (lazy, FS changed): %s", target)
        try:
            self.build_code_tree(target)
        except Exception as e:  # noqa: BLE001
            log.warning("lazy code_tree rebuild failed: %s", e)

    def load_kgl(self, path: Path) -> None:
        graph = kglite.load(str(path))
        with self._lock:
            self._active = ActiveGraph(graph=graph, source_path=path)

    def build_code_tree(self, dir_path: Path) -> None:
        try:
            from kglite import code_tree
        except ImportError as e:
            raise RuntimeError(
                "code_tree graph building requires the `tree-sitter` "
                "extra. Install with `pip install 'kglite[mcp,code-tree]'` "
                "(or add `tree-sitter` to your environment)."
            ) from e
        graph = code_tree.build(str(dir_path))
        with self._lock:
            self._active = ActiveGraph(graph=graph, source_path=None)

    def bind_embedder(self, embedder: Any) -> None:
        """Bind an embedder instance to the active graph. No-op if no
        active graph yet (workspace mode loads the graph later)."""
        with self._lock:
            if self._active is None:
                return
            self._active.graph.set_embedder(embedder)

    def active(self) -> ActiveGraph | None:
        with self._lock:
            return self._active

    def schema(self) -> tuple[int, int] | None:
        a = self.active()
        if a is None:
            return None
        s = a.graph.schema()
        return (s["node_count"], s["edge_count"])


def run_cypher(
    state: GraphState,
    query: str,
    params: dict[str, Any] | None = None,
    csv_http: CsvHttpConfig | None = None,
    preprocessor: Preprocessor | None = None,
) -> str:
    """Run a Cypher query against the active graph. Returns the
    rendered tool body — inline 15-row preview by default, CSV (or URL
    when csv_http is configured) when `FORMAT CSV` appears in the
    query.

    0.9.25: when `preprocessor` is set, the query + params are rewritten
    before execution (see `kglite.mcp_server.preprocessor`). Rewrites
    that raise ValueError/TypeError surface as
    `preprocessor: <message>` short-circuit bodies — no traceback
    reaches the agent."""
    active = state.active()
    if active is None:
        return NO_GRAPH

    rewritten = apply_preprocessor(preprocessor, query, params)
    if isinstance(rewritten, str):
        # `apply_preprocessor` returns a str when the preprocessor
        # raised — forward verbatim as the tool body.
        return rewritten
    query, params = rewritten

    try:
        result = active.graph.cypher(query, params=params or {})
    except Exception as e:  # noqa: BLE001 — surface the engine's error verbatim
        return f"Cypher error: {e}"

    # `FORMAT CSV` is recognised inside kglite's cypher() — when set,
    # the call returns a CSV string instead of a ResultView. Use that
    # signal to route URL-vs-inline.
    if isinstance(result, str):
        if csv_http is not None:
            try:
                name = write_csv(csv_http, result)
            except Exception as e:  # noqa: BLE001
                # Best-effort: fall back to inline CSV on write failure.
                return result + f"\n# (csv_http write failed: {e}; CSV inlined)"
            url = csv_http.url_for(name)
            row_count = _count_csv_rows(result)
            return f"FORMAT CSV: {row_count} row(s) written to {url}\nFetch with: curl {url}"
        return result

    return _format_inline(result)


def _count_csv_rows(csv: str) -> int:
    # Strip trailing newline so we don't double-count an empty final line.
    lines = csv.rstrip("\n").splitlines()
    return max(0, len(lines) - 1)  # minus header


def _format_inline(result: Any) -> str:
    """Render a kglite ResultView as a 15-row inline preview. Matches
    the Rust shim's format."""
    rows = list(result)
    total = len(rows)
    if total == 0:
        return "No results."
    columns = result.columns
    header = f"{total} row(s) (showing first 15):\n" if total > 15 else f"{total} row(s):\n"
    out = [header, "\t".join(columns), "\n"]
    for row in rows[:15]:
        # 0.9.22 fix: kglite's ResultView yields rows as DICTS keyed by
        # column name. Iterating a dict gives KEYS (column names), not
        # values — the 0.9.21 regression that returned `'f.name'` in
        # every cell instead of the actual data. Index by column to
        # pull the value.
        out.append("\t".join(_repr(row[col]) for col in columns))
        out.append("\n")
    return "".join(out)


def _repr(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    return str(v)


def run_overview(
    state: GraphState,
    types: list[str] | None,
    connections: Any,
    cypher: Any,
    temp_cleanup_dir: Path | None,
    *,
    overview_prefix: str | None = None,
) -> str:
    active = state.active()
    if active is None:
        return NO_GRAPH

    is_bare = types is None and connections is None and cypher is None

    # Wipe the temp dir on bare overview when temp_cleanup is on. "Bare"
    # = no drill-down args. Matches the Rust shim's P4 behaviour.
    if temp_cleanup_dir is not None and is_bare:
        _wipe_dir(temp_cleanup_dir)

    kwargs: dict[str, Any] = {}
    if types is not None:
        kwargs["types"] = types
    if connections is not None:
        kwargs["connections"] = connections
    if cypher is not None:
        kwargs["cypher"] = cypher
    try:
        body = active.graph.describe(**kwargs)
    except Exception as e:  # noqa: BLE001
        return f"graph_overview error: {e}"

    # 0.9.32: manifest's `overview_prefix:` text is prepended ONLY on
    # bare-overview calls (no drill-down args). Mirrors the FastMCP
    # path's behaviour at mcp_methods/fastmcp/_overview.py and the
    # documented contract in docs/guides/mcp-servers.md. Pre-0.9.32
    # the Python entry parsed the field but never used it — operator
    # bug report 2026-05-14.
    if overview_prefix and is_bare:
        return f"{overview_prefix}\n\n{body}"
    return body


def run_save(state: GraphState) -> str:
    active = state.active()
    if active is None:
        return NO_GRAPH
    if active.source_path is None:
        return "save_graph requires --graph mode (no source path bound)."
    try:
        active.graph.save(str(active.source_path))
    except Exception as e:  # noqa: BLE001
        return f"save_graph error: {e}"
    s = state.schema()
    if s is None:
        return f"Saved {active.source_path}."
    nodes, edges = s
    return f"Saved {active.source_path} ({nodes} nodes, {edges} edges)."


def run_explore(
    state: GraphState,
    query: str,
    max_entities: int = 10,
    max_depth: int = 2,
    include_source: bool = True,
    source_roots: list[str] | None = None,
) -> str:
    """One-call codebase exploration: rank entry points + 2-hop
    neighborhood + grouped source slices. Thin wrapper over the kglite
    `graph.explore()` Python API (Rust-side ranking + traversal +
    slicing); the MCP layer only validates args and forwards source
    roots. Non-code graphs / empty queries get a no-match message from
    the engine, not an error."""
    active = state.active()
    if active is None:
        return NO_GRAPH
    if not isinstance(query, str) or not query.strip():
        return "explore: missing required argument `query` (a free-text topic, e.g. 'authenticate')."
    try:
        return active.graph.explore(
            query=query,
            max_entities=max_entities,
            max_depth=max_depth,
            include_source=include_source,
            source_roots=source_roots or None,
        )
    except Exception as e:  # noqa: BLE001
        return f"explore error: {e}"


EXPLORE_DESCRIPTION = (
    "One-call codebase exploration for code-tree graphs. Lexically ranks "
    "Function/Class/Interface/Struct/Trait/Protocol/Enum nodes against "
    "`query` (name > signature > docstring), takes the top entries, 2-hop "
    "traverses CALLS/USES_TYPE/HAS_METHOD/DEFINES/REFERENCES_FN, and returns "
    "a markdown report: entry points, a 2-hop neighborhood, and grouped "
    "source slices for the entry points — replacing the typical grep + read "
    "chain for 'how does X work' / 'where is Y' / 'trace the Z flow' "
    "questions. Returns a no-match message on non-code graphs."
)

EXPLORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Free-text topic. Short, specific terms beat sentences "
                "('authenticate' beats 'how does login work'); the ranker is "
                "lexical, not semantic. A symbol name you've already seen "
                "always works."
            ),
        },
        "max_entities": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "Top N entry points after ranking (default 10).",
        },
        "max_depth": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": (
                "Neighborhood hops (default 2): 1 = direct neighbors only, 2 = "
                "neighbours' neighbours. Past 3 the neighborhood explodes on "
                "well-connected code."
            ),
        },
        "include_source": {
            "type": ["boolean", "null"],
            "description": (
                "Include source slices for entry points (default true). Set "
                "false for a smaller, faster symbol+location-only response."
            ),
        },
    },
    "required": ["query"],
}


def _wipe_dir(dir_path: Path) -> None:
    """Best-effort directory wipe — log on errors, never raise. Matches
    the Rust shim's `wipe_temp_dir` semantics."""
    import logging
    import shutil

    log = logging.getLogger("kglite.mcp_server.tools")
    if not dir_path.is_dir():
        log.debug("temp_cleanup: directory does not exist: %s", dir_path)
        return
    wiped = 0
    for entry in dir_path.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            wiped += 1
        except OSError as e:
            log.debug("temp_cleanup: remove %s failed: %s", entry, e)
    if wiped > 0:
        log.info("temp_cleanup: wiped %d entries from %s", wiped, dir_path)
