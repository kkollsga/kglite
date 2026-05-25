"""`kglite-mcp-server` — Python entry point (0.9.20+).

Replaces the pre-0.9.20 bundled Rust binary. The architecture is
identical: same modes (`--graph` / `--workspace` / `--watch` /
`--source-root` / bare), same YAML schema, same tool surface
(`cypher_query`, `graph_overview`, `save_graph`, `read_code_source`,
manifest-declared cypher tools, optional `csv_http_server`).

Why Python instead of Rust: a Rust binary that depends on the kglite
crate transitively links libpython at a specific Python version,
forcing per-Python wheel matrices (3 OS × 4 Python = 12 wheels in
0.9.18/0.9.19). The Python entry-point bypasses that entirely — the
wheel is back to 3 abi3 wheels, and kglite's Python `cypher()`
method already releases the GIL so the hot-path performance is
unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys
from typing import Any

log = logging.getLogger("kglite.mcp_server")


def main() -> None:
    """Console-script entry point. Synchronous wrapper that pipes into
    the asyncio loop."""
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging()

    try:
        # Lazy imports so a missing `mcp` extras dep produces an
        # actionable error rather than a cryptic ModuleNotFoundError
        # from somewhere deep in the call chain.
        _check_extras()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"kglite-mcp-server: {e}\n")
        sys.exit(1)

    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        sys.exit(0)


def _check_extras() -> None:
    """Verify the `mcp` extras are installed. Without them the server
    can't function; without a clear error, users hit confusing
    `ModuleNotFoundError` deep in the stack."""
    missing = []
    for mod, pkg in [
        ("mcp", "mcp"),
        ("yaml", "pyyaml"),
        ("aiohttp", "aiohttp"),
        ("fastembed", "fastembed"),
        ("watchdog", "watchdog"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise SystemExit(
            "kglite-mcp-server requires the MCP extras. Install with:\n"
            f"  pip install 'kglite[mcp]'\n"
            f"Missing packages: {', '.join(missing)}"
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kglite-mcp-server",
        description="MCP server for KGLite knowledge graphs (Python entry point).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--graph", type=Path, help="Path to a .kgl knowledge graph file. Loaded at boot.")
    mode.add_argument(
        "--source-root",
        type=Path,
        dest="source_root",
        help="Source-root mode (no graph).",
    )
    mode.add_argument(
        "--workspace",
        type=Path,
        help="Workspace mode: clone GitHub repos and build code-tree graphs.",
    )
    mode.add_argument(
        "--watch",
        type=Path,
        help="Watch mode: rebuild the code-tree graph on file changes.",
    )
    p.add_argument("--mcp-config", type=Path, dest="mcp_config")
    p.add_argument("--name")
    p.add_argument(
        "--trust-tools",
        action="store_true",
        help="Legacy flag (0.9.17 era); no-op in 0.9.20+ since python tools were removed.",
    )
    p.add_argument(
        "--stale-after-days",
        type=int,
        default=7,
        dest="stale_after_days",
        help="Workspace mode: refresh clones older than N days (default 7).",
    )
    return p


def _setup_logging() -> None:
    level = logging.INFO
    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


async def _async_main(args: argparse.Namespace) -> None:
    from kglite.mcp_server.csv_http import (
        CsvHttpConfig,
    )
    from kglite.mcp_server.csv_http import (
        from_manifest_value as csv_http_from_manifest,
    )
    from kglite.mcp_server.csv_http import (
        spawn as csv_http_spawn,
    )
    from kglite.mcp_server.embedder import from_manifest_value as embedder_from_manifest
    from kglite.mcp_server.preprocessor import PreprocessorError
    from kglite.mcp_server.preprocessor import from_manifest_value as preprocessor_from_manifest
    from kglite.mcp_server.tools import GraphState
    from kglite.mcp_server.watch import start as watch_start
    from kglite.mcp_server.workspace import Workspace

    mode = _pick_mode(args)
    _validate_mode_paths(mode)

    manifest = _load_manifest(args, mode)
    _load_env(manifest, mode)

    csv_http_cfg: CsvHttpConfig | None = None
    embedder_adapter = None
    preprocessor = None
    if manifest is not None:
        ext = manifest.extensions
        csv_http_cfg = csv_http_from_manifest(ext.get("csv_http_server"), manifest.base_dir)
        embedder_adapter = embedder_from_manifest(ext.get("embedder"))
        try:
            preprocessor = preprocessor_from_manifest(
                ext.get("cypher_preprocessor"),
                manifest.base_dir,
                manifest.trust.allow_query_preprocessor,
            )
        except PreprocessorError as e:
            sys.stderr.write(f"ERROR: {e}\n")
            sys.exit(3)

    if csv_http_cfg is not None:
        await csv_http_spawn(csv_http_cfg)

    graph_state = GraphState()
    workspace: Workspace | None = None

    # Local-workspace active-root slot. Populated by the
    # `_post_activate` hook each time `set_root_dir(path)` fires;
    # read by the watch callback in `local_workspace + watch:true`
    # mode so rebuilds target the active repo (not the wide
    # `workspace.root` sandbox). Stays `None` until the first
    # `set_root_dir` — the watch callback then no-ops, matching the
    # boot-time deferral. List-cell idiom so the closures can mutate.
    active_root_holder: list[Path | None] = [None]

    # Mode-specific binding.
    if mode["kind"] == "graph":
        graph_state.load_kgl(mode["path"])
    elif mode["kind"] == "watch":
        graph_state.build_code_tree(mode["path"])
    elif mode["kind"] == "workspace":
        workspace = Workspace(root=mode["path"], stale_after_days=args.stale_after_days)
    elif mode["kind"] == "local_workspace":
        workspace = Workspace(
            root=mode["path"],
            kind="local",
            stale_after_days=args.stale_after_days,
        )
        # Operator inbox 2026-05-25: the explicit boot-time
        # `build_code_tree(workspace.root)` was a 0.9.28 stop-gap to
        # ensure cypher_query had something to query before any
        # set_root_dir. For wide `workspace.root` (the documented
        # lateral-swap pattern, e.g. ~360k files), it blocks MCP
        # `initialize` past Claude Desktop's 60s timeout — operator
        # sees only "Could not attach to MCP server." Drop the
        # eager build; first `set_root_dir(path)` becomes the first
        # build (via the `_post_activate` hook wired below).
        # Bare local-workspace boot now returns in ms.

    if embedder_adapter is not None:
        graph_state.bind_embedder(embedder_adapter)

    # Boot summary to stderr (MCP stdio uses stdout for protocol).
    _print_boot_summary(mode, manifest, graph_state)

    # Wire the watcher if applicable. `on_change` now receives the
    # changed paths so the callback can skip rebuilds for events
    # that aren't graph-relevant (a `cargo build` storm changes
    # `target/...` but no `.rs` files; an `npm install` storm
    # changes `node_modules/...` only; etc.).
    from kglite import _kglite_code_tree

    def _any_code_file(paths: list[str]) -> bool:
        return any(_kglite_code_tree.language_for_path(p) is not None for p in paths)

    watch_handle: Any = None
    if mode["kind"] == "watch":

        def _watch_rebuild(paths: list[str]) -> None:
            # Tag for rebuild; the actual rebuild fires on the next
            # MCP tool call via ensure_code_tree_fresh.
            if not _any_code_file(paths):
                return
            graph_state.tag_code_tree_dirty(mode["path"])

        watch_handle = watch_start(mode["path"], _watch_rebuild)
    elif mode["kind"] == "local_workspace" and manifest and manifest.workspace and manifest.workspace.watch:
        # Operator inbox 2026-05-25: previously
        # `lambda: graph_state.build_code_tree(mode["path"])` — i.e.
        # always rebuilt the wide sandbox on every FS event (every
        # cargo build, editor save, etc. anywhere under it). Now:
        # two-stage filter — only rebuild when at least one changed
        # path is (a) inside the active repo AND (b) a file
        # code_tree actually parses. No-op when no set_root_dir has
        # fired yet.
        def _local_workspace_rebuild(paths: list[str]) -> None:
            # Tag for rebuild; the actual rebuild fires on the next
            # MCP tool call via ensure_code_tree_fresh.
            active = active_root_holder[0]
            if active is None:
                return
            active_str = str(active)
            if not any(p.startswith(active_str) and _kglite_code_tree.language_for_path(p) is not None for p in paths):
                return
            graph_state.tag_code_tree_dirty(active)

        watch_handle = watch_start(mode["path"], _local_workspace_rebuild)
    _ = watch_handle  # keep alive until process exit

    # Build the MCP server.
    server = _build_server(
        graph_state=graph_state,
        manifest=manifest,
        mode=mode,
        csv_http_cfg=csv_http_cfg,
        workspace=workspace,
        preprocessor=preprocessor,
        active_root_holder=active_root_holder,
    )

    # Serve.
    from mcp.server.stdio import stdio_server

    fallback_name = _fallback_name(mode["kind"])
    server_name = args.name or (manifest.name if manifest else None) or fallback_name
    instructions = manifest.instructions if manifest else None
    # Banner first (runtime state — which conditional tools are
    # registered in this mode), batch-load hint + operator text after.
    # See `_compose_mode_banner` for the per-mode payload.
    mode_banner = _compose_mode_banner(mode, manifest)
    instructions = f"{mode_banner}\n\n{_compose_instructions(instructions)}"

    async with stdio_server() as (read_stream, write_stream):
        init_opts = server.create_initialization_options(
            notification_options=None,
            experimental_capabilities={},
        )
        # Override the name + instructions populated by
        # `create_initialization_options` so the boot YAML can shape
        # what the agent sees on `initialize`.
        init_opts.server_name = server_name
        if instructions is not None:
            init_opts.instructions = instructions
        await server.run(read_stream, write_stream, init_opts)


# Marker the auto-injected hint carries so we don't duplicate it if an
# operator's manifest already contains the same text (or if the
# injection runs twice for any reason).
_BATCH_LOAD_HINT_MARKER = "[kglite-batch-load-hint]"

_BATCH_LOAD_HINT = f"""\
{_BATCH_LOAD_HINT_MARKER}
This server's tools are deferred-loaded by Claude Code: each tool
appears in ToolSearch but its full schema only arrives after a
search-by-name. To batch-load every tool on this server in one
round trip, call:

  ToolSearch(query="+<server-slug>", max_results=20)

where <server-slug> is the substring between `mcp__` and the next
`__` in this server's tool names (the key you used in your MCP
client config). One ToolSearch call per server, not per tool.
"""


_MODE_BANNER_MARKER = "[kglite-mode]"


def _compose_mode_banner(mode: dict[str, Any], manifest: Any) -> str:
    """Build a per-mode banner naming which conditional tools are
    registered. Prepended to the `instructions` block on `initialize`
    and to the bare `graph_overview()` response so the agent can see
    at a glance whether `repo_management` / `set_root_dir` / `save_graph`
    are available without probing.

    Operator feedback 2026-05-17: with several MCP servers attached the
    agent has no clean way to distinguish "tool not registered (wrong
    mode)" from "tool failed", and ends up fingerprinting mode by trial.
    """
    kind = mode["kind"]
    save_graph_enabled = bool(manifest and manifest.builtins and manifest.builtins.save_graph)
    if kind == "graph":
        path = mode.get("path")
        path_hint = f" ({Path(path).name})" if path else ""
        save_line = (
            "- save_graph: registered. Call to persist CREATE / SET / DELETE mutations."
            if save_graph_enabled
            else "- save_graph: not registered (graph is read-only)."
        )
        return (
            f"{_MODE_BANNER_MARKER} graph{path_hint}\n"
            f"- cypher_query / graph_overview / read_code_source: registered.\n"
            f"{save_line}\n"
            "- repo_management / set_root_dir: not in this mode."
        )
    if kind == "workspace":
        return (
            f"{_MODE_BANNER_MARKER} workspace (clone-and-activate)\n"
            "- repo_management: registered. Start with:\n"
            "    repo_management()             — list known repos\n"
            "    repo_management('org/repo')   — clone + activate\n"
            "- cypher_query / graph_overview: registered (operate on the active repo's graph).\n"
            "- save_graph / set_root_dir: not in this mode."
        )
    if kind == "local_workspace":
        return (
            f"{_MODE_BANNER_MARKER} local-workspace (fixed root, lateral swap)\n"
            "- set_root_dir: registered. Use set_root_dir(path) to swap to "
            "any subdir under the configured root.\n"
            "- cypher_query / graph_overview: registered (operate on the active root's graph).\n"
            "- save_graph / repo_management: not in this mode."
        )
    if kind == "source_root":
        return (
            f"{_MODE_BANNER_MARKER} source-root (file tools only, no graph)\n"
            "- read_source / grep / list_source: registered.\n"
            "- cypher_query / graph_overview / save_graph: not active (no graph)."
        )
    if kind == "watch":
        return (
            f"{_MODE_BANNER_MARKER} watch (code-tree graph rebuilt on file change)\n"
            "- cypher_query / graph_overview: registered.\n"
            "- save_graph: not registered (graph is derived from the source tree)."
        )
    # bare
    return f"{_MODE_BANNER_MARKER} bare\n- Manifest-only / framework-default tools. No graph, no source root."


def _compose_instructions(operator_instructions: str | None) -> str:
    """Combine the kglite batch-load hint with operator-declared
    instructions. The hint teaches agents how to load all of this
    server's tools in one ToolSearch round trip rather than serially
    via per-tool searches (operator-reported friction in 0.9.29's
    post-deploy audit).

    Idempotent: if the marker is already present in operator_instructions
    the hint is skipped (operators who want full control can include the
    marker literally to suppress auto-injection)."""
    if operator_instructions and _BATCH_LOAD_HINT_MARKER in operator_instructions:
        return operator_instructions
    if not operator_instructions:
        return _BATCH_LOAD_HINT
    return f"{_BATCH_LOAD_HINT}\n\n{operator_instructions}"


def _pick_mode(args: argparse.Namespace) -> dict[str, Any]:
    if args.graph:
        return {"kind": "graph", "path": args.graph}
    if args.source_root:
        return {"kind": "source_root", "path": args.source_root}
    if args.workspace:
        return {"kind": "workspace", "path": args.workspace}
    if args.watch:
        return {"kind": "watch", "path": args.watch}
    return {"kind": "bare", "path": None}


def _validate_mode_paths(mode: dict[str, Any]) -> None:
    if mode["kind"] == "graph":
        p = mode["path"]
        # 0.9.26: `--graph` also accepts disk-backed graph directories
        # (built via `kglite.KnowledgeGraph(storage="disk", path=...)`),
        # not just single `.kgl` files. Disk graphs are a directory of
        # column-store files identified by a `disk_graph_meta.json`
        # sentinel — the same marker the Rust loader checks
        # (src/graph/io/file.rs::load_file). Without this branch,
        # operators deploying Wikidata-scale graphs (the documented
        # `storage="disk"` path for graphs >50M nodes) hit a
        # misleading "does not exist" error even though the path is
        # a valid graph directory.
        if not (p.is_file() or (p.is_dir() and (p / "disk_graph_meta.json").is_file())):
            sys.stderr.write(f"--graph path is not a .kgl file or disk-backed graph directory: {p}\n")
            sys.exit(1)
    if mode["kind"] in {"source_root", "watch", "workspace"} and not mode["path"].is_dir():
        sys.stderr.write(f"path does not exist or is not a directory: {mode['path']}\n")
        sys.exit(1)


def _load_manifest(args: argparse.Namespace, mode: dict[str, Any]) -> Any:
    from kglite.mcp_server.manifest import (
        ManifestError,
        find_sibling_manifest,
        find_workspace_manifest,
        load_manifest,
    )

    path: Path | None = None
    if args.mcp_config:
        if not args.mcp_config.is_file():
            sys.stderr.write(f"--mcp-config path does not exist: {args.mcp_config}\n")
            sys.exit(3)
        path = args.mcp_config
    elif mode["kind"] == "graph":
        path = find_sibling_manifest(mode["path"])
    elif mode["kind"] in {"workspace", "watch"}:
        path = find_workspace_manifest(mode["path"])

    if path is None:
        return None
    try:
        manifest = load_manifest(path)
    except ManifestError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(3)

    # Promote `workspace.kind: local` over CLI flags.
    if manifest.workspace and manifest.workspace.kind == "local" and manifest.workspace.root:
        mode["kind"] = "local_workspace"
        mode["path"] = manifest.workspace.root

    return manifest


def _load_env(manifest: Any, mode: dict[str, Any]) -> None:
    """Load `.env` from the manifest's `env_file:` or a walk-up
    discovery rooted at the mode's directory.

    0.9.24: the walk-up case delegates to mcp-methods Rust
    (`kglite._mcp_internal.load_env_walk`) which mirrors the
    framework's `load_env_for_mode` semantics — same quoting rules,
    same no-overwrite-existing-env behaviour, same comment/blank-line
    handling. Explicit `env_file:` paths are still loaded inline
    (small enough that we don't need to round-trip through Rust)."""
    from kglite import _mcp_internal

    if manifest and manifest.env_file:
        env_path = manifest.env_file
        if not env_path.is_file():
            return
        # Minimal .env loader (don't pull python-dotenv as a hard dep).
        import os

        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
        log.info("loaded env file: %s", env_path)
        return

    start = mode["path"] if mode["path"] else Path.cwd()
    if start.is_file():
        start = start.parent
    loaded = _mcp_internal.load_env_walk(str(start.resolve()))
    if loaded:
        log.info("loaded env file: %s", loaded)


def _fallback_name(kind: str) -> str:
    return {
        "graph": "KGLite (single-graph)",
        "source_root": "KGLite (source-root)",
        "workspace": "KGLite (workspace)",
        "local_workspace": "KGLite (local-workspace)",
        "watch": "KGLite (watch)",
        "bare": "KGLite",
    }[kind]


def _print_boot_summary(mode: dict[str, Any], manifest: Any, state: Any) -> None:
    parts = [f"mode: {mode['kind']}"]
    if mode["path"]:
        parts[-1] += f" [{mode['path']}]"
    if manifest is not None:
        parts.append(f"manifest: {manifest.yaml_path}")
    s = state.schema()
    if s is not None:
        nodes, edges = s
        parts.append(f"graph: {nodes} nodes, {edges} edges")
    sys.stderr.write("kglite-mcp-server: " + "; ".join(parts) + "\n")


BUNDLED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "cypher_query",
        "graph_overview",
        "ping",
        "read_code_source",
        "save_graph",
        "read_source",
        "grep",
        "list_source",
        "repo_management",
        "set_root_dir",
        "github_issues",
        "github_api",
    }
)
"""Set of bundled tool names kglite-mcp-server can register. The
manifest's `tools[].bundled:` overrides (mcp-methods 0.3.31+) are
validated against this set at boot: any name not in here surfaces
as a clear `ERROR: unknown bundled tool ...` so operators catch
typos before runtime."""


def _build_server(
    graph_state: Any,
    manifest: Any,
    mode: dict[str, Any],
    csv_http_cfg: Any,
    workspace: Any,
    preprocessor: Any = None,
    active_root_holder: list[Path | None] | None = None,
) -> Any:
    """Construct the mcp.Server and register all tools."""
    from mcp import types
    from mcp.server.lowlevel import Server

    import kglite._mcp_internal as mcp_internal
    from kglite.mcp_server.code_source import DESCRIPTION as CODE_SOURCE_DESC
    from kglite.mcp_server.code_source import SCHEMA as CODE_SOURCE_SCHEMA
    from kglite.mcp_server.code_source import read as read_code_source
    from kglite.mcp_server.cypher_tools import build_tool_attrs as cypher_tool_attrs
    from kglite.mcp_server.cypher_tools import call_cypher_tool
    from kglite.mcp_server.tools import run_cypher, run_overview, run_save

    fallback = _fallback_name(mode["kind"])
    server_name = (manifest.name if manifest else None) or fallback
    server: Any = Server(server_name)

    # 0.9.27: bundled-tool overrides from manifest. Validate names
    # against the catalogue at boot — unknown names error out with
    # the full list of valid bundled tools (matches the strict-key
    # validation pattern operators expect from manifest parsing).
    bundled_overrides_list = manifest.bundled_overrides if manifest else []
    unknown_bundled = {b.name for b in bundled_overrides_list} - BUNDLED_TOOL_NAMES
    if unknown_bundled:
        sys.stderr.write(
            "ERROR: unknown bundled tool name(s) in manifest `tools[].bundled:`: "
            f"{sorted(unknown_bundled)}. Valid names: {sorted(BUNDLED_TOOL_NAMES)}\n"
        )
        sys.exit(3)
    bundled_overrides: dict[str, Any] = {b.name: b for b in bundled_overrides_list}
    hidden_bundled_names: set[str] = {b.name for b in bundled_overrides_list if b.hidden}

    # 0.9.30: collect `rename:` overrides and build a runtime lookup
    # from agent-facing alias → canonical bundled name. mcp-methods
    # 0.3.34+ ships the `rename:` field on bundled override entries;
    # we validate collisions across the manifest's registered tool
    # namespace (other bundled canonical names, other renames, cypher
    # tools) at boot so operators see the error before serving.
    alias_to_canonical: dict[str, str] = {}
    cypher_tool_names: set[str] = {t.name for t in (manifest.tools if manifest else [])}
    for b in bundled_overrides_list:
        if b.rename is None:
            continue
        alias = b.rename
        if alias == b.name:
            # No-op rename — silently allow (operator might generate
            # manifests programmatically and the equality case
            # shouldn't be a hard error).
            continue
        if alias in alias_to_canonical:
            sys.stderr.write(
                f"ERROR: duplicate `tools[].bundled: rename:` value {alias!r} "
                f"(would shadow another rename for {alias_to_canonical[alias]!r}).\n"
            )
            sys.exit(3)
        if alias in BUNDLED_TOOL_NAMES:
            sys.stderr.write(
                f"ERROR: `tools[].bundled: rename:` value {alias!r} shadows "
                f"an existing bundled tool name. Rename targets must not "
                f"collide with canonical bundled names.\n"
            )
            sys.exit(3)
        if alias in cypher_tool_names:
            sys.stderr.write(
                f"ERROR: `tools[].bundled: rename:` value {alias!r} shadows "
                f"a manifest-declared cypher tool of the same name.\n"
            )
            sys.exit(3)
        alias_to_canonical[alias] = b.name

    def _apply_override(tool: types.Tool) -> types.Tool:
        """Apply a manifest `bundled:` override to a freshly-built
        tool, returning a new Tool with the override applied (or the
        original if no override exists for this tool name).

        Both `description:` and `rename:` are honoured — the latter
        rewrites `tool.name` so the agent sees the override-declared
        identifier in `tools/list`. Dispatch in `_call` reverses the
        rename via `alias_to_canonical` before running the tool body."""
        override = bundled_overrides.get(tool.name)
        if override is None:
            return tool
        updates: dict[str, Any] = {}
        if override.description is not None:
            updates["description"] = override.description
        if override.rename is not None and override.rename != tool.name:
            updates["name"] = override.rename
        if not updates:
            return tool
        return tool.model_copy(update=updates)

    builtins = manifest.builtins if manifest else None
    save_graph_enabled = bool(builtins and builtins.save_graph)
    # Banner for the bare `graph_overview()` preamble — same payload
    # as the `initialize` instructions block, computed once at boot.
    overview_mode_banner = _compose_mode_banner(mode, manifest)
    temp_cleanup_dir: Path | None = None
    if builtins and builtins.temp_cleanup_on_overview:
        if csv_http_cfg is not None:
            temp_cleanup_dir = csv_http_cfg.dir
        elif manifest is not None:
            temp_cleanup_dir = manifest.base_dir / "temp"

    source_roots: list[Path] = list(manifest.source_roots) if manifest else []
    if mode["kind"] == "graph" and not source_roots and mode["path"]:
        # Auto-bind the .kgl's parent as the source root, same default
        # as the Rust shim.
        source_roots = [mode["path"].resolve().parent]
    elif mode["kind"] == "source_root" and mode["path"]:
        source_roots = [mode["path"].resolve()]
    elif mode["kind"] == "local_workspace" and mode["path"] and not source_roots:
        # 0.9.23: local-workspace mode binds the workspace root as the
        # source root by default (set_root_dir rebinds it at runtime).
        # Without this the fallback below picked manifest.base_dir which
        # is the YAML's parent — usually NOT what the operator wanted.
        source_roots = [mode["path"].resolve()]
    elif manifest is not None and not source_roots:
        # Fallback: any manifest-driven boot without an explicit
        # source_root binds the manifest's directory. Matches the
        # 0.9.18 Rust binary's behaviour — operator's sodir-style
        # YAMLs (no source_root: declared) registered read_source /
        # grep / list_source under that parent. Without this fallback
        # we'd silently lose three tools per such manifest, which is
        # exactly the 0.9.20 regression class.
        source_roots = [manifest.base_dir]

    manifest_cypher_tools = manifest.tools if manifest else []
    cypher_tool_lookup = {t.name: t for t in manifest_cypher_tools}

    # 0.9.23: source roots must be re-read on each tool call so
    # `set_root_dir` can rebind them at runtime. Mutate `source_roots`
    # in place from the handler; expose a lambda for the source tools
    # to fetch the live list. Previously this was captured once at
    # build time and set_root_dir didn't actually change anything the
    # source tools could see (the regression Cat F2 catches).
    def _live_source_roots() -> list[str]:
        return [str(p) for p in source_roots]

    # 0.9.28: wire workspace post-activate so `repo_management(...)`
    # and local-workspace `set_root_dir(...)` actually build the
    # code-tree graph for the active path AND rebind source_roots so
    # subsequent read_source/grep target the active clone. Prior to
    # 0.9.28 the workspace was instantiated but this hook was never
    # called — the activation tools were path-tracking-only, no graph
    # build, source_roots stayed pointed at the workspace dir. That's
    # why operator's open-source / code-review deployments couldn't
    # migrate from 0.6.x.
    if workspace is not None:

        def _post_activate(repo_path: str, _repo_name: str) -> None:
            # Hook signature: (repo_path, repo_name). The Rust wrapper
            # passes path first (mcp_tools.rs::DeferredPyHook::invoke),
            # which matches mcp-methods' Rust hook contract.
            target = Path(repo_path)
            # Operator inbox 2026-05-25: stash the active root so the
            # local-workspace watch callback (in `_run`) can scope its
            # rebuilds. Before this, the watch callback always
            # rebuilt `workspace.root` (the wide sandbox) on every
            # FS event.
            if active_root_holder is not None:
                active_root_holder[0] = target
            graph_state.build_code_tree(target)
            source_roots[:] = [target]

        workspace.set_post_activate(_post_activate)

    # GithubIssues holds an ElementCache for drill-down across calls.
    # Constructed once per process; lives for the server's lifetime.
    github_token_present = mcp_internal.has_github_token()
    github_issues = mcp_internal.GithubIssues() if github_token_present else None

    # Workspace-mode tools. `repo_management` registers in remote
    # workspace mode; `set_root_dir` in local workspace mode. Matches
    # the 0.9.18 binary's per-mode tool surface.
    is_workspace = mode["kind"] == "workspace"
    is_local_workspace = mode["kind"] == "local_workspace"

    # 0.9.31: load + merge + filter skills. Manifest-gated: bare-mode
    # deployments without a manifest get no skills (the framework's
    # `SkillRegistry.from_manifest` needs a YAML to read `skills:`
    # from). Manifests with `skills: false` (or unset) get the empty
    # set. Manifests with `skills: true` get kglite-bundled + framework
    # defaults; richer values pull in project + operator layers. The
    # set is filtered by `applies_when:` against the *current* graph
    # state — but the closure captures graph_state by reference, so
    # subsequent calls re-evaluate the predicate live (post-activate
    # workspace switches reflect immediately).
    from kglite.mcp_server.skills_loader import Skill, build_active_skill_set

    def _registered_tool_names() -> set[str]:
        names = {"cypher_query", "graph_overview", "read_code_source", "ping"}
        if save_graph_enabled:
            names.add("save_graph")
        if source_roots:
            names.update({"read_source", "grep", "list_source"})
        if github_token_present:
            names.update({"github_issues", "github_api"})
        if is_workspace:
            names.add("repo_management")
        if is_local_workspace:
            names.add("set_root_dir")
        for t in manifest_cypher_tools:
            names.add(t.name)
        return names

    manifest_extensions: dict[str, Any] = manifest.extensions if manifest else {}
    manifest_yaml_path: Path | None = manifest.yaml_path if manifest else None

    def _has_node_type(node_type: str) -> bool:
        active = graph_state.active()
        if active is None:
            return False
        try:
            return node_type in (active.graph.describe(types=[node_type]) or "")
        except Exception:  # noqa: BLE001
            return False

    def _has_property(node_type: str, prop_name: str) -> bool:
        # Probe via a labeled MATCH that returns the property; an
        # exception or empty result indicates the property doesn't
        # exist on that node type (or the type doesn't exist). The
        # check stays cheap because LIMIT 1 short-circuits.
        active = graph_state.active()
        if active is None:
            return False
        try:
            rows = list(active.graph.cypher(f"MATCH (n:{node_type}) WHERE n.{prop_name} IS NOT NULL RETURN n LIMIT 1"))
            return len(rows) > 0
        except Exception:  # noqa: BLE001
            return False

    # Skills are opt-in via manifest `skills:`. `None` or `False` →
    # disabled. Any other value (`True`, a path string, or a list)
    # enables the kglite-bundled + framework-defaults + project layers.
    skills_opted_in: bool = bool(manifest and manifest.skills)
    active_skills: dict[str, Skill] = build_active_skill_set(
        manifest_yaml_path,
        has_node_type=_has_node_type,
        has_property=_has_property,
        registered_tools=_registered_tool_names(),
        extensions=manifest_extensions,
        skills_opted_in=skills_opted_in,
    )
    skill_auto_hint_names: set[str] = {s.name for s in active_skills.values() if s.auto_inject_hint}

    # 0.9.32: cap auto-injected skill bodies to match the framework's
    # lint thresholds — 16 KB hard ceiling, 4 KB soft target. Beyond
    # the hard ceiling we truncate with a trailing marker so the
    # injected body stays a bounded fraction of every tools/list
    # payload. (16 KB × N tools is still meaningful context cost; the
    # 4 KB soft target keeps most well-authored skills under this cap
    # with room to spare.)
    _SKILL_INJECT_HARD_LIMIT = 16 * 1024
    _SKILL_INJECT_TRUNCATE_MARKER = "\n\n[skill body truncated — full text via prompts/get]"

    # Marker the injected `## Methodology` section carries so we don't
    # duplicate it if `_apply_skill_hint` runs twice for any reason.
    _SKILL_INJECT_HEADER = "\n\n## Methodology\n\n"

    def _apply_skill_hint(tool: types.Tool) -> types.Tool:
        """Inject the active skill's full body into a tool's
        description when an active skill with auto_inject_hint=True
        shares the tool's name.

        Embeds the body verbatim under a `## Methodology` header.
        Reachable in every MCP client (every client exposes
        tools/list to the agent). Operators who want the smaller
        payload can set `auto_inject_hint: false` per-skill; the
        skill still surfaces via prompts/list for clients that DO
        expose prompts to agents (custom MCP integrations building
        on the protocol directly).

        This matches the framework's canonical `serve_prompts`
        auto-inject pass (mcp-methods 0.3.37+) which adopted the
        same shape after operator finding that `prompts/get` is
        unreachable by agents in Claude Code / Claude Desktop /
        Cursor / Continue. The MCP `prompts/*` plane was designed
        for human slash commands in chat UIs, not agentic retrieval.
        Pre-0.9.32 kglite emitted a bracketed pointer (`[See
        prompts/get NAME for full methodology.]`) — a dangling
        reference in those clients.

        Idempotent: a header marker in the existing description
        suppresses re-injection."""
        if tool.name not in skill_auto_hint_names:
            return tool
        skill = active_skills.get(tool.name)
        if skill is None:
            return tool
        existing = tool.description or ""
        if _SKILL_INJECT_HEADER in existing:
            return tool
        body = skill.body
        if len(body) > _SKILL_INJECT_HARD_LIMIT:
            body = body[: _SKILL_INJECT_HARD_LIMIT - len(_SKILL_INJECT_TRUNCATE_MARKER)] + _SKILL_INJECT_TRUNCATE_MARKER
        return tool.model_copy(update={"description": existing + _SKILL_INJECT_HEADER + body})

    @server.list_tools()
    async def _list() -> list[types.Tool]:
        # ───────────────────────────────────────────────────────────
        # kglite-specific tools
        # ───────────────────────────────────────────────────────────
        tools: list[types.Tool] = [
            types.Tool(
                name="cypher_query",
                description=(
                    "Run a Cypher query against the active knowledge graph. "
                    "Returns up to 15 rows inline; append FORMAT CSV to "
                    "export results"
                    + (
                        " — large CSVs are written to the csv_http_server directory and returned as a fetch URL."
                        if csv_http_cfg is not None
                        else " to a CSV string."
                    )
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Cypher query string. Append `FORMAT CSV` for CSV-encoded output.",
                        }
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="graph_overview",
                description=(
                    "Inspect the active graph's schema. With no args returns "
                    "the inventory; pass types=[...] / connections=true|[...] "
                    "/ cypher=true|[...] for drill-down."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "types": {"type": ["array", "null"], "items": {"type": "string"}},
                        "connections": {"type": ["array", "boolean", "null"]},
                        "cypher": {"type": ["array", "boolean", "null"]},
                    },
                },
            ),
            types.Tool(
                name="read_code_source",
                description=CODE_SOURCE_DESC,
                inputSchema=CODE_SOURCE_SCHEMA,
            ),
        ]
        if save_graph_enabled:
            tools.append(
                types.Tool(
                    name="save_graph",
                    description="Persist the active graph to its source .kgl file (single-graph mode only).",
                    inputSchema={"type": "object", "properties": {}},
                )
            )

        # ───────────────────────────────────────────────────────────
        # Framework tools — wrapped from mcp-methods Rust crate via
        # kglite._mcp_internal. Same output format as the 0.9.18 binary.
        # ───────────────────────────────────────────────────────────
        tools.append(
            types.Tool(
                name="ping",
                description="Liveness probe. Returns the supplied message (default 'pong').",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Optional message to echo back.",
                        }
                    },
                },
            )
        )
        if _live_source_roots():
            tools.extend(_framework_source_tools())
        if github_issues is not None:
            tools.extend(_framework_github_tools())
        if is_workspace:
            tools.append(_framework_repo_management_tool())
        if is_local_workspace:
            tools.append(_framework_set_root_dir_tool())

        # 0.9.27: apply manifest `tools[].bundled:` overrides.
        # `hidden: true` drops the tool from tools/list (and is also
        # rejected at call time in `_call` below). `description: "..."`
        # replaces the agent-facing description without touching the
        # input schema.
        tools = [_apply_override(t) for t in tools if t.name not in hidden_bundled_names]

        # YAML-declared cypher tools (overrides do NOT apply here —
        # cypher tools carry their own description in the manifest
        # entry; the override block is for the bundled catalogue
        # only).
        tools.extend(cypher_tool_attrs(manifest_cypher_tools))

        # 0.9.31: auto-inject "see prompts/get <name>" hint into
        # matching tool descriptions when a skill exists for the tool
        # name AND has auto_inject_hint=True. Mirrors the framework's
        # auto-inject pass for the Rust binary path. Agents that
        # discover tools first (tools/list) get a discoverability
        # pointer to the skill body.
        if skill_auto_hint_names:
            tools = [_apply_skill_hint(t) for t in tools]

        return tools

    @server.call_tool()
    async def _call(name: str, args: dict[str, Any]) -> list[types.TextContent]:
        # 0.9.30: agents call bundled tools by their agent-facing name,
        # which may be a `rename:` alias rather than the canonical
        # name. Translate before dispatch so the rest of this function
        # operates on canonical names only. The reverse map (canonical
        # → alias) is used in `_list`; here we go alias → canonical.
        if name in alias_to_canonical:
            name = alias_to_canonical[name]
        # 0.9.27: hidden bundled tools (manifest declared
        # `tools[].bundled: NAME / hidden: true`) are absent from
        # `tools/list` but the dispatcher still needs to refuse
        # calls if an agent tries the name directly. Reject early
        # with a clear error rather than falling through to "unknown
        # tool" (which would be confusing — the tool DOES exist in
        # the catalogue, the manifest just hid it).
        if name in hidden_bundled_names:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error: tool {name!r} is hidden by manifest configuration.",
                )
            ]
        # Lazy rebuild gate: every graph-touching tool checks whether
        # the watcher tagged a rebuild since the last call. Cheap
        # (single lock-acquire + None check) when nothing's pending.
        if name in {"cypher_query", "graph_overview", "save_graph", "read_code_source"}:
            graph_state.ensure_code_tree_fresh()
        if name == "cypher_query":
            body = run_cypher(
                graph_state,
                args.get("query", ""),
                csv_http=csv_http_cfg,
                preprocessor=preprocessor,
            )
        elif name == "graph_overview":
            # Echo the mode banner into the bare-call preamble so the
            # mode/write-mode signal re-surfaces after the handshake
            # `instructions` text ages out of context. Operator-supplied
            # `overview_prefix:` text (if any) follows the banner.
            op_prefix = manifest.overview_prefix if manifest else None
            composed_prefix = f"{overview_mode_banner}\n\n{op_prefix}" if op_prefix else overview_mode_banner
            body = run_overview(
                graph_state,
                args.get("types"),
                args.get("connections"),
                args.get("cypher"),
                temp_cleanup_dir,
                overview_prefix=composed_prefix,
            )
        elif name == "save_graph":
            body = run_save(graph_state)
        elif name == "read_code_source":
            body = read_code_source(graph_state, args, source_roots)
        # ── framework tools, forward to Rust wrappers ────────────────
        elif name == "ping":
            body = mcp_internal.ping(args.get("message"))
        elif name == "read_source":
            body = mcp_internal.read_source(
                args["file_path"],
                _live_source_roots(),
                start_line=args.get("start_line"),
                end_line=args.get("end_line"),
                grep=args.get("grep"),
                grep_context=args.get("grep_context"),
                max_matches=args.get("max_matches"),
                max_chars=args.get("max_chars"),
            )
        elif name == "grep":
            body = mcp_internal.grep(
                args["pattern"],
                _live_source_roots(),
                glob=args.get("glob"),
                context=args.get("context", 0),
                max_results=args.get("max_results"),
                case_insensitive=args.get("case_insensitive", False),
            )
        elif name == "list_source":
            body = mcp_internal.list_source(
                _live_source_roots(),
                path=args.get("path", "."),
                depth=args.get("depth", 1),
                glob=args.get("glob"),
                dirs_only=args.get("dirs_only", False),
            )
        elif name == "github_issues":
            if github_issues is None:
                body = "Error: github_issues requires GITHUB_TOKEN / GH_TOKEN to be set."
            else:
                body = _call_github_issues(github_issues, args, workspace)
        elif name == "github_api":
            body = mcp_internal.git_api(
                args.get("repo_name") or _active_repo_for_github(workspace),
                args["path"],
                truncate_at=args.get("truncate_at", 80_000),
            )
        elif name == "repo_management":
            if workspace is None:
                body = "Error: repo_management requires workspace mode."
            else:
                body = workspace.repo_management_tool(args)
        elif name == "set_root_dir":
            if workspace is None:
                body = "Error: set_root_dir requires local-workspace mode."
            else:
                body = workspace.set_root_dir_tool(args.get("path", ""))
                # source_roots rebind + code-tree rebuild happen via
                # the workspace post-activate hook wired above (0.9.28).
                # mcp-methods workspace.rs fires the hook on both
                # repo_management(activate) AND set_root_dir success,
                # so this branch no longer needs an inline rebind.
        # ── manifest-declared cypher tools ───────────────────────────
        elif name in cypher_tool_lookup:
            body = await call_cypher_tool(
                cypher_tool_lookup[name],
                graph_state,
                args,
                csv_http_cfg,
                preprocessor=preprocessor,
            )
        else:
            body = f"unknown tool: {name!r}"
        return [types.TextContent(type="text", text=body)]

    # 0.9.31: prompts/list + prompts/get for skills. The set of active
    # skills was computed once at boot via `build_active_skill_set`;
    # we expose them here as MCP prompts. Empty active set → both
    # handlers return empty / not-found.
    @server.list_prompts()
    async def _list_prompts() -> list[types.Prompt]:
        # Re-resolve at request time so applies_when predicates that
        # depend on the active graph (graph_has_node_type, etc.)
        # reflect any post-boot mutations (e.g. workspace activated
        # a repo after server startup). Cheap — predicate evaluation
        # is one Cypher LIMIT 1 per declared clause in the worst case.
        live = build_active_skill_set(
            manifest_yaml_path,
            has_node_type=_has_node_type,
            has_property=_has_property,
            registered_tools=_registered_tool_names(),
            extensions=manifest_extensions,
            skills_opted_in=skills_opted_in,
        )
        return [
            types.Prompt(name=s.name, description=s.description) for s in sorted(live.values(), key=lambda x: x.name)
        ]

    @server.get_prompt()
    async def _get_prompt(name: str, arguments: dict[str, Any] | None) -> types.GetPromptResult:
        live = build_active_skill_set(
            manifest_yaml_path,
            has_node_type=_has_node_type,
            has_property=_has_property,
            registered_tools=_registered_tool_names(),
            extensions=manifest_extensions,
            skills_opted_in=skills_opted_in,
        )
        skill = live.get(name)
        if skill is None:
            # mcp.server.lowlevel surfaces ValueError as a protocol-
            # level error response to the agent. Use a clear message
            # rather than KeyError so the agent sees actionable text.
            raise ValueError(f"unknown prompt: {name!r} (active skills: {sorted(live.keys())})")
        return types.GetPromptResult(
            description=skill.description,
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=skill.body),
                )
            ],
        )

    return server


def _framework_source_tools() -> list:
    """JSON-Schema definitions for the file-system tools. Source roots
    must be configured for these to be useful — caller gates by
    checking source_roots is non-empty."""
    from mcp import types

    return [
        types.Tool(
            name="read_source",
            description=(
                "Read a file slice from one of the configured source roots. "
                "Line range / regex filter / truncation; path traversal rejected."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "grep": {"type": "string"},
                    "grep_context": {"type": "integer"},
                    "max_matches": {"type": "integer"},
                    "max_chars": {"type": "integer"},
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="grep",
            description=(
                "Recursive regex search across source roots. .gitignore-aware. "
                "Output: `path:lineno:content` for matches, `path-lineno-content` "
                "for context lines."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string"},
                    "context": {"type": "integer"},
                    "max_results": {"type": "integer"},
                    "case_insensitive": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
        ),
        types.Tool(
            name="list_source",
            description="Tree-style directory listing relative to the primary source root.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "depth": {"type": "integer"},
                    "glob": {"type": "string"},
                    "dirs_only": {"type": "boolean"},
                },
            },
        ),
    ]


def _framework_github_tools() -> list:
    from mcp import types

    return [
        types.Tool(
            name="github_issues",
            description=(
                "Fetch / search / list GitHub issues, PRs, and discussions. "
                "Drill-down with `element_id` after a fetch (cb_N for code blocks, "
                "comment_N for review comments, patch_N for diffs)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "repo_name": {"type": "string"},
                    "query": {"type": "string"},
                    "kind": {"type": "string", "enum": ["issue", "pr", "discussion", "all"]},
                    "state": {"type": "string", "enum": ["open", "closed", "all"]},
                    "sort": {"type": "string"},
                    "limit": {"type": "integer"},
                    "labels": {"type": "string"},
                    "element_id": {"type": "string"},
                    "lines": {"type": "string"},
                    "grep": {"type": "string"},
                    "context": {"type": "integer"},
                    "refresh": {"type": "boolean"},
                },
            },
        ),
        types.Tool(
            name="github_api",
            description=(
                "Generic GitHub REST GET. Relative paths auto-prefix with "
                "`/repos/<repo>/`; absolute paths pass through. Returns pretty-"
                "printed JSON, truncated at `truncate_at` chars (default 80k)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "repo_name": {"type": "string"},
                    "truncate_at": {"type": "integer"},
                },
                "required": ["path"],
            },
        ),
    ]


def _call_github_issues(github_issues: Any, args: dict[str, Any], workspace: Any) -> str:
    """Route a github_issues call between FETCH (number given) and
    SEARCH/LIST (no number) via the wrapped Rust ElementCache.

    0.9.23: when `repo_name` isn't passed and a workspace is active,
    default to the workspace's active repo. Without this the agent had
    to repeat `repo_name='org/repo'` on every call even after
    `repo_management('org/repo')` activated one. Operator caught this
    on 0.9.22 redeploy."""
    repo = args.get("repo_name") or _active_repo_for_github(workspace) or None
    number = args.get("number")
    if number is not None:
        return github_issues.fetch(
            repo or "",
            int(number),
            element_id=args.get("element_id"),
            lines=args.get("lines"),
            grep=args.get("grep"),
            context=args.get("context", 3),
            refresh=args.get("refresh", False),
        )
    return github_issues.search_or_list(
        repo=repo,
        query=args.get("query"),
        kind=args.get("kind", "all"),
        state=args.get("state", "open"),
        sort=args.get("sort"),
        limit=args.get("limit", 20),
        labels=args.get("labels"),
    )


def _framework_repo_management_tool() -> Any:
    from mcp import types

    return types.Tool(
        name="repo_management",
        description=(
            "Activate (or update / delete) a cloned GitHub repo in the "
            "workspace. With `name='org/repo'`: clone if missing, build "
            "code-tree, set as active. With `update=true`: refresh active "
            "clone. With `delete=true`: remove clone + inventory entry. "
            "With no args: list known repos."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "delete": {"type": "boolean"},
                "update": {"type": "boolean"},
                "force_rebuild": {"type": "boolean"},
            },
        },
    )


def _framework_set_root_dir_tool() -> Any:
    from mcp import types

    return types.Tool(
        name="set_root_dir",
        description=(
            "Swap the active source root (local-workspace mode only). "
            "Canonicalises the path, rebinds the source tools, and "
            "triggers a code-tree rebuild for the new root."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    )


def _active_repo_for_github(workspace: Any) -> str:
    """Return the active repo (`org/repo`) for github_api when not
    explicitly passed. Falls back to empty string — the Rust function
    handles the empty case with a clear error."""
    if workspace is None:
        return ""
    name = getattr(workspace, "active_repo_name", lambda: None)()
    return name or ""


if __name__ == "__main__":
    main()
