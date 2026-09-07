"""Build a KGLite knowledge graph from a JSON blueprint and its inputs.

Usage::

    from kglite.blueprint import from_blueprint

    graph = from_blueprint("blueprint.json")

Implemented entirely in Rust; see ``src/graph/blueprint/`` and the
``from_blueprint_rust`` ``#[pyfunction]`` in ``src/graph/pyapi/blueprint.rs``.
This module is a thin shim that handles optional save + schema lock on
top of the Rust build.
"""

from __future__ import annotations

import datetime as _datetime
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from kglite.kglite import KnowledgeGraph
from kglite.kglite import from_blueprint_rust as _from_blueprint_rs
from kglite.kglite import from_records_rust as _from_records_rs


def _save_destination(output_path: Optional[str], storage: str, path: Optional[str]) -> Optional[str]:
    """Where a built blueprint graph should be written, or None.

    Two destinations, in precedence order:

    1. The blueprint's own ``output`` / ``output_file`` setting, already
       resolved to an absolute path by the Rust loader.
    2. ``storage="disk"`` + ``path``. In disk mode the directory *is* the
       graph, but the build only leaves a working directory there —
       publication (the ``CURRENT`` marker plus ``disk_graph_meta.json``)
       happens at ``save()``. Without it the caller is left with a
       directory that looks like a graph and that ``kglite.load()``
       rejects, so ``path`` is the destination the flag must mean.

    ``storage="mapped"`` ignores ``path`` (mapped columns spill to
    anonymous mmap), so it never yields a destination.
    """
    if output_path:
        return output_path
    if storage == "disk" and path:
        return path
    return None


def from_blueprint(
    blueprint_path: Union[str, Path],
    *,
    verbose: bool = False,
    save: Optional[bool] = None,
    lock_schema: bool = False,
    storage: str = "default",
    path: Optional[str] = None,
    frames: Optional[Mapping[str, Any]] = None,
) -> KnowledgeGraph:
    """Build a KnowledgeGraph from a JSON blueprint + its declared inputs.

    Args:
        blueprint_path: Path to the blueprint JSON file.
        verbose: Print a summary line after the build.
        save: Whether to persist the built graph. ``None`` (the default)
            saves when a destination exists and skips otherwise; ``True``
            requires one and raises ``ValueError`` if there is none;
            ``False`` never saves. The destination is the blueprint's
            ``output`` / ``output_file`` setting, or — with
            ``storage="disk"`` — ``path``.
        lock_schema: If True, lock the schema so subsequent Cypher
            mutations are validated against the blueprint types.
        storage: ``"default"`` (in-memory), ``"mapped"`` (mmap columns),
            or ``"disk"`` (CSR + mmap). Disk requires ``path``.
        path: Directory for disk storage (only used with ``storage="disk"``).
        frames: In-memory tables, keyed by the name of a ``files`` entry
            declaring ``{"format": "frame"}``. Values are pandas
            DataFrames; anything else carrying ``.to_pandas()`` (a polars
            frame, a pyarrow table) is converted through it. Declared scalar
            strings use CSV parsing, including whitespace and exact signed
            integer decimal/scientific forms. Invalid declared cells become
            NULL; undeclared columns retain dtype-based inference.

    Raises:
        ValueError: If ``save=True`` was passed explicitly and neither
            destination exists.
    """
    if verbose:
        print(f"Loading blueprint from {blueprint_path}...")
    if frames is not None:
        # Duck-typed, so polars/pyarrow work without kglite importing either:
        # a pandas DataFrame has no `.to_pandas()`, so this is a no-op for one.
        frames = {name: (frame.to_pandas() if hasattr(frame, "to_pandas") else frame) for name, frame in frames.items()}
    graph, output_path = _from_blueprint_rs(
        str(blueprint_path),
        verbose=verbose,
        storage=storage if storage else "default",
        path=path,
        frames=frames,
    )
    if verbose:
        counts = graph.node_type_counts()
        for node_type, n in sorted(counts.items()):
            print(f"  {node_type}: {n} nodes")
    if save is not False:
        destination = _save_destination(output_path, storage, path)
        if destination is not None:
            out = Path(destination)
            out.parent.mkdir(parents=True, exist_ok=True)
            graph.save(str(out))
        elif save:
            raise ValueError(
                "from_blueprint(save=True) has nowhere to write the graph: the "
                "blueprint declares no 'output' / 'output_file' setting, and this "
                "is not a storage='disk' build with a path. Add \"output\": "
                '"graph.kgl" to the blueprint\'s settings, or pass '
                "storage='disk', path='graph_dir/'. Omit save entirely to build "
                "without persisting."
            )
    if lock_schema:
        graph.lock_schema()
    return graph


def _json_scalar(value: Any) -> Any:
    """Refuse a record value JSON cannot carry, by name.

    ``json.dumps`` raised ``TypeError: Object of type datetime is not JSON
    serializable`` — true, but it named neither the value nor the field, and it
    read as a wrapper bug rather than a spec one. The refusal itself is the
    contract (``tests/test_property_roundtrip_matrix.py``): a records spec is
    JSON, JSON has no temporal type, and silently writing ``datetime`` as
    ISO-8601 text would demote a temporal to a string property — a silent
    degradation, which is the one outcome that matrix exists to keep out. Load
    temporals through ``add_nodes``/``add_connections``, which type them, or
    pass the ISO-8601 string yourself and convert with Cypher ``datetime()``.

    Missing values are the exception: ``pd.NaT`` carries no temporal value to
    degrade, so it becomes ``null`` like any other absent field.
    """
    if isinstance(value, (_datetime.datetime, _datetime.date, _datetime.time)):
        if value.isoformat() == "NaT":  # pandas' missing timestamp
            return None
        raise TypeError(
            f"from_records: a {type(value).__name__} value ({value!r}) cannot be carried in a "
            "JSON records spec — JSON has no temporal type. Pass an ISO-8601 string and convert "
            "with Cypher datetime()/date(), or load the column through add_nodes()."
        )
    raise TypeError(
        f"from_records: a record value of type {type(value).__name__!r} ({value!r}) cannot be "
        "carried in a JSON records spec"
    )


def from_records(
    spec: Union[dict, str],
    *,
    save: Optional[str] = None,
    lock_schema: bool = False,
    storage: str = "default",
    path: Optional[str] = None,
    on_missing_endpoint: Optional[str] = None,
) -> KnowledgeGraph:
    """Build a KnowledgeGraph from an inline JSON records spec.

    A JSON-native sibling to :func:`from_blueprint`: instead of pointing at
    CSV files on disk, the spec carries node and connection records inline.
    Agent-authored graphs are JSON-native, so this is the natural ingestion
    path for them. Column types are inferred from the record values, so a JSON
    array becomes a native list property. Missing edge endpoints can be
    vivified as provisional stubs, dropped, or rejected atomically.

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

            Those key sets are closed: an unknown key at either level raises
            a ``ValueError`` (naming a near-miss where there is one) rather
            than being ignored.

        save: If set, save the built graph to this ``.kgl`` path. With
            ``storage="disk"`` pass ``path`` here too: the disk build
            leaves an unpublished working directory until something calls
            ``save()``, and ``kglite.load()`` rejects that directory.
        lock_schema: If True, lock the schema after building.
        storage: ``"default"`` (in-memory), ``"mapped"``, or ``"disk"``.
        path: Directory for disk storage (only used with ``storage="disk"``).
        on_missing_endpoint: ``"vivify"`` creates provisional stub nodes,
            ``"drop"`` omits affected edges, and ``"error"`` rejects the
            complete build without publishing partial changes. ``None`` (the
            default) takes the spec's own ``on_missing_endpoint`` key, which
            itself defaults to ``"vivify"``; passing the argument overrides
            whatever the spec carries.
    """
    records_json = spec if isinstance(spec, str) else json.dumps(spec, default=_json_scalar)
    graph = _from_records_rs(
        records_json,
        storage=storage if storage else "default",
        path=path,
        on_missing_endpoint=on_missing_endpoint,
    )
    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        graph.save(str(out))
    if lock_schema:
        graph.lock_schema()
    return graph


__all__ = ["from_blueprint", "from_records"]
