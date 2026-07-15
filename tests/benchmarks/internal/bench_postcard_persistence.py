"""Cross-version persistence benchmark for the Postcard migration.

Run this file from an isolated environment outside the checkout so the local
``kglite/`` package cannot shadow the wheel being measured.  It intentionally
uses only public Python APIs shared by 0.13.3 and the migration branch.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import sys
import tempfile
import time

import pandas as pd

import kglite


def _stats(samples: list[float]) -> dict[str, float | int]:
    return {
        "rounds": len(samples),
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.mean(samples),
    }


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the BSDs traditionally report KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


def _rows(result):
    return result.to_list() if hasattr(result, "to_list") else list(result)


def _shape(scale: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "id": range(scale),
            "name": [f"n{i}" for i in range(scale)],
            "group": [i % 8 for i in range(scale)],
            "score": [(i % 101) / 10 for i in range(scale)],
            # Lists exercise the overflow/Mixed-value persistence paths.  The
            # values are intentionally small to expose Postcard's varints.
            "tags": [[i % 5, (i + 1) % 5, (i + 2) % 5] for i in range(scale)],
        }
    )
    edge_count = scale * 3
    edges = pd.DataFrame(
        {
            "source": [i % scale for i in range(edge_count)],
            "target": [(i * 17 + 11) % scale for i in range(edge_count)],
            "rank": [i % 17 for i in range(edge_count)],
        }
    )
    return nodes, edges


def _populate(graph, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    graph.add_nodes(nodes, "Node", "id", "name")
    graph.add_connections(edges, "LINK", "Node", "source", "Node", "target")


def _portable(root: Path, nodes: pd.DataFrame, edges: pd.DataFrame, rounds: int) -> dict:
    graph = kglite.KnowledgeGraph()
    _populate(graph, nodes, edges)
    target = root / "portable.kgl"

    save_samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        graph.save(str(target), fsync=False)
        save_samples.append(time.perf_counter() - start)

    del graph
    gc.collect()
    load_samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        loaded = kglite.load(str(target))
        count = loaded.cypher("MATCH (n:Node) RETURN count(n) AS n").scalar()
        load_samples.append(time.perf_counter() - start)
        assert count == len(nodes)
        del loaded
        gc.collect()

    loaded = kglite.load(str(target))
    query_samples: dict[str, dict] = {}
    queries = {
        "in_memory_filter": "MATCH (n:Node) WHERE n.group = 3 RETURN count(n) AS n",
        "edge_property_scan": "MATCH ()-[r:LINK]->() WHERE r.rank < 8 RETURN count(r) AS n",
        "overflow_projection": "MATCH (n:Node) RETURN n.tags AS tags LIMIT 200",
    }
    for name, query in queries.items():
        loaded.cypher(query)
        samples = []
        for _ in range(max(100, rounds * 20)):
            start = time.perf_counter()
            result = loaded.cypher(query)
            _rows(result)
            samples.append(time.perf_counter() - start)
        query_samples[name] = _stats(samples)

    return {
        "artifact_bytes": target.stat().st_size,
        "save": _stats(save_samples),
        "load_and_count": _stats(load_samples),
        "queries": query_samples,
    }


def _disk(root: Path, nodes: pd.DataFrame, edges: pd.DataFrame, rounds: int) -> dict:
    save_samples = []
    open_samples = []
    sizes = []
    for index in range(rounds):
        path = root / f"disk-{index}"
        graph = kglite.KnowledgeGraph(storage="disk", path=str(path))
        _populate(graph, nodes, edges)
        start = time.perf_counter()
        graph.save(str(path), fsync=False)
        save_samples.append(time.perf_counter() - start)
        del graph
        gc.collect()

        start = time.perf_counter()
        loaded = kglite.load(str(path))
        count = loaded.cypher("MATCH ()-[r:LINK]->() RETURN count(r) AS n").scalar()
        open_samples.append(time.perf_counter() - start)
        assert count == len(edges)
        sizes.append(_dir_size(path))
        del loaded
        gc.collect()

    return {
        "artifact_bytes_min": min(sizes),
        "save": _stats(save_samples),
        "open_and_count": _stats(open_samples),
    }


def _wal(root: Path, mutations: int, rounds: int) -> dict:
    append_samples = []
    recover_samples = []
    sizes = []
    for index in range(rounds):
        path = root / f"wal-{index}.kgl"
        graph = kglite.open(str(path), durable=True)
        start = time.perf_counter()
        for item in range(mutations):
            graph.cypher(
                "CREATE (:Event {id: $id, group: $group})",
                params={"id": item, "group": item % 8},
            )
        append_samples.append(time.perf_counter() - start)
        wal_path = Path(f"{path}-wal")
        sizes.append(wal_path.stat().st_size)
        del graph
        gc.collect()

        start = time.perf_counter()
        recovered = kglite.open(str(path), durable=True)
        count = recovered.cypher("MATCH (n:Event) RETURN count(n) AS n").scalar()
        recover_samples.append(time.perf_counter() - start)
        assert count == mutations
        del recovered
        gc.collect()

    return {
        "mutations": mutations,
        "artifact_bytes_min": min(sizes),
        "append": _stats(append_samples),
        "recover_and_count": _stats(recover_samples),
    }


def _write_ntriples(path: Path, entities: int) -> None:
    with path.open("w", encoding="utf-8") as output:
        for item in range(entities):
            output.write(
                f"<http://www.wikidata.org/entity/Q{item}> "
                f'<http://www.w3.org/2000/01/rdf-schema#label> "Entity {item}"@en .\n'
            )
            if item:
                output.write(
                    f"<http://www.wikidata.org/entity/Q{item}> "
                    "<http://www.wikidata.org/prop/direct/P17> "
                    f"<http://www.wikidata.org/entity/Q{item - 1}> .\n"
                )


def _property_log(root: Path, entities: int, rounds: int) -> dict:
    source = root / "property-log.nt"
    _write_ntriples(source, entities)
    samples = []
    sizes = []
    for index in range(rounds):
        path = root / f"ntriples-{index}"
        graph = kglite.KnowledgeGraph(storage="disk", path=str(path))
        start = time.perf_counter()
        result = graph.load_ntriples(str(source), languages=["en"])
        samples.append(time.perf_counter() - start)
        assert result["entities"] == entities
        sizes.append(_dir_size(path))
        del graph
        gc.collect()
    return {
        "entities": entities,
        "artifact_bytes_min": min(sizes),
        "ingest": _stats(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=20_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--wal-mutations", type=int, default=1_500)
    parser.add_argument("--ntriples-entities", type=int, default=20_000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="kglite-postcard-bench-"))
    try:
        nodes, edges = _shape(args.scale)
        result = {
            "harness_version": 1,
            "kglite_version": getattr(kglite, "__version__", "unknown"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "scale": args.scale,
            "rounds": args.rounds,
            "portable": _portable(work_root, nodes, edges, args.rounds),
            "disk": _disk(work_root, nodes, edges, args.rounds),
            "wal": _wal(work_root, args.wal_mutations, args.rounds),
            "property_log": _property_log(work_root, args.ntriples_entities, args.rounds),
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
