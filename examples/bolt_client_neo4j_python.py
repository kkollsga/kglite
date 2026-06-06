#!/usr/bin/env python3
"""Query kglite-bolt-server with the standard Neo4j Python driver.

Demonstrates: the Bolt ecosystem unlock — any Neo4j-aware client talks to
KGLite over the wire with zero changes. Build a graph, save it, start the
server, and drive it with the exact `neo4j` driver you'd point at real Neo4j.

Shows: handshake (`verify_connectivity`), scalar RETURN, parameters,
variable-length traversal, and a whole-Node return (PackStream `Node`
struct round-tripping over the wire).

Requires: pip install kglite[neo4j]   (for the neo4j driver)
          a built kglite-bolt-server binary — `cargo build -p
          kglite-bolt-server --release`, or set KGLITE_BOLT_SERVER to its path.
"""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from neo4j import GraphDatabase
import pandas as pd

import kglite


def _find_bolt_server() -> str:
    """Locate the kglite-bolt-server binary: $KGLITE_BOLT_SERVER, then PATH,
    then the repo's target/release/ build."""
    env = os.environ.get("KGLITE_BOLT_SERVER")
    if env:
        return env
    on_path = shutil.which("kglite-bolt-server")
    if on_path:
        return on_path
    candidate = Path(__file__).resolve().parent.parent / "target" / "release" / "kglite-bolt-server"
    if candidate.exists():
        return str(candidate)
    print("kglite-bolt-server binary not found. Build it with:", file=sys.stderr)
    print("    cargo build -p kglite-bolt-server --release", file=sys.stderr)
    print("or set KGLITE_BOLT_SERVER to its path.", file=sys.stderr)
    sys.exit(1)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"bolt server never came up on {host}:{port}")


# -- Build a small graph + save it -----------------------------------------

graph = kglite.KnowledgeGraph()
people = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "title": ["Alice", "Bob", "Carol", "Dave"],
        "city": ["Oslo", "Bergen", "Oslo", "Trondheim"],
    }
)
graph.add_nodes(people, "Person", "id", "title")
edges = pd.DataFrame({"src": [1, 2, 3], "dst": [2, 3, 4]})
graph.add_connections(edges, "KNOWS", "Person", "src", "Person", "dst")

tmpdir = tempfile.TemporaryDirectory(prefix="kglite-bolt-example-")
kgl_path = Path(tmpdir.name) / "people.kgl"
graph.save(str(kgl_path))
print(f"Built + saved graph to {kgl_path}")

# -- Start kglite-bolt-server on an ephemeral port -------------------------

binary = _find_bolt_server()
port = _free_port()
proc = subprocess.Popen([binary, "--graph", str(kgl_path), "--bind", "127.0.0.1", "--port", str(port)])
url = f"bolt://127.0.0.1:{port}"

try:
    _wait_for_port("127.0.0.1", port)
    print(f"kglite-bolt-server listening on {url}")

    # -- Drive it with the standard Neo4j driver ---------------------------

    with GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
        driver.verify_connectivity()

        # Scalar RETURN
        print("\n--- People in Oslo ---")
        with driver.session() as session:
            for record in session.run("MATCH (p:Person) WHERE p.city = 'Oslo' RETURN p.title AS name"):
                print(f"  {record['name']}")

        # Parameters + variable-length traversal
        print("\n--- Who Alice knows (1-2 hops) ---")
        with driver.session() as session:
            for record in session.run(
                "MATCH (a:Person {title: $name})-[:KNOWS*1..2]->(b:Person) RETURN DISTINCT b.title AS friend",
                name="Alice",
            ):
                print(f"  {record['friend']}")

        # Whole-node return — PackStream Node struct over the wire
        print("\n--- Return a whole node ---")
        with driver.session() as session:
            node = session.run("MATCH (p:Person {title: 'Bob'}) RETURN p").single()["p"]
            print(f"  labels={list(node.labels)}  props={dict(node)}")
finally:
    proc.terminate()
    proc.wait(timeout=5)
    tmpdir.cleanup()
