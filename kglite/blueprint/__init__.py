"""Build a KGLite knowledge graph from a JSON blueprint and CSV files.

Usage::

    from kglite.blueprint import from_blueprint

    graph = from_blueprint("blueprint.json")

Implemented entirely in Rust; see ``src/graph/blueprint/`` and the
``from_blueprint_rust`` ``#[pyfunction]`` in ``src/graph/pyapi/blueprint.rs``.
This module is a ~20-line shim that handles optional save + schema lock
on top of the Rust build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from kglite.kglite import KnowledgeGraph
from kglite.kglite import from_blueprint_rust as _from_blueprint_rs
from kglite.kglite import from_records_rust as _from_records_rs


def from_blueprint(
    blueprint_path: Union[str, Path],
    *,
    verbose: bool = False,
    save: bool = True,
    lock_schema: bool = False,
    storage: str = "default",
    path: Optional[str] = None,
) -> KnowledgeGraph:
    """Build a KnowledgeGraph from a JSON blueprint + CSV files.

    Args:
        blueprint_path: Path to the blueprint JSON file.
        verbose: Print a summary line after the build.
        save: If True and the blueprint has an ``output_file`` key, save
            the built graph to that path.
        lock_schema: If True, lock the schema so subsequent Cypher
            mutations are validated against the blueprint types.
        storage: ``"default"`` (in-memory), ``"mapped"`` (mmap columns),
            or ``"disk"`` (CSR + mmap). Disk requires ``path``.
        path: Directory for disk storage (only used with ``storage="disk"``).
    """
    if verbose:
        print(f"Loading blueprint from {blueprint_path}...")
    graph, output_path = _from_blueprint_rs(
        str(blueprint_path),
        verbose=verbose,
        storage=storage if storage else "default",
        path=path,
    )
    if verbose:
        counts = graph.node_type_counts()
        for node_type, n in sorted(counts.items()):
            print(f"  {node_type}: {n} nodes")
    if save and output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        graph.save(str(out))
    if lock_schema:
        graph.lock_schema()
    return graph


def from_records(
    spec: Union[dict, str],
    *,
    save: Optional[str] = None,
    lock_schema: bool = False,
    storage: str = "default",
    path: Optional[str] = None,
) -> KnowledgeGraph:
    """Build a KnowledgeGraph from an inline JSON records spec.

    A JSON-native sibling to :func:`from_blueprint`: instead of pointing at
    CSV files on disk, the spec carries node and connection records inline.
    Agent-authored graphs are JSON-native, so this is the natural ingestion
    path for them. Column types are inferred from the record values, so a JSON
    array becomes a native list property. Missing edge endpoints are
    auto-vivified as provisional stub nodes (same as ``add_connections``).

    Args:
        spec: The records spec, as a ``dict`` or a JSON string. Shape::

            {
              "nodes": [
                {"type": "Person", "id_field": "id", "title_field": "name",
                 "conflict_handling": "update",
                 "records": [{"id": 1, "name": "Alice", "aliases": ["a", "b"]}]}
              ],
              "connections": [
                {"type": "KNOWS", "source_type": "Person", "source_id_field": "from",
                 "target_type": "Person", "target_id_field": "to",
                 "records": [{"from": 1, "to": 2, "since": 2020}]}
              ]
            }

        save: If set, save the built graph to this ``.kgl`` path.
        lock_schema: If True, lock the schema after building.
        storage: ``"default"`` (in-memory), ``"mapped"``, or ``"disk"``.
        path: Directory for disk storage (only used with ``storage="disk"``).
    """
    records_json = spec if isinstance(spec, str) else json.dumps(spec)
    graph = _from_records_rs(
        records_json,
        storage=storage if storage else "default",
        path=path,
    )
    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        graph.save(str(out))
    if lock_schema:
        graph.lock_schema()
    return graph


__all__ = ["from_blueprint", "from_records"]
