#!/usr/bin/env python3
"""Build a code knowledge graph from a source directory (code tree parsing).

Demonstrates: code_tree.build, code-specific queries, graph algorithms.

Parses: Python, Rust, TypeScript, Go, Java, C#, C++.
Generates: Function, Class, Struct, Enum, Module, File nodes
           with CALLS, DEFINES, IMPORTS, HAS_METHOD edges.

Requires: pip install kglite
"""

from pathlib import Path
import sys

from kglite.code_tree import build

# -- Build -----------------------------------------------------------------

src_dir = sys.argv[1] if len(sys.argv) > 1 else "."
src_path = Path(src_dir).resolve()

if not src_path.is_dir():
    print(f"Not a directory: {src_path}", file=sys.stderr)
    sys.exit(1)

print(f"Parsing {src_path} ...")
graph = build(str(src_path), verbose=True)

schema = graph.schema()
print(f"\nBuilt: {schema['node_count']} nodes, {schema['edge_count']} edges")
for nt, info in sorted(schema["node_types"].items()):
    print(f"  {nt}: {info['count']}")

# -- Save ------------------------------------------------------------------

output = f"{src_path.name}.kgl"
graph.save(output)
print(f"\nSaved to {output}")

# -- Example queries -------------------------------------------------------

# Most-called functions (by incoming CALLS edges)
print("\n--- Most-called functions ---")
for row in graph.cypher("""
    MATCH (caller:Function)-[:CALLS]->(f:Function)
    RETURN f.qualified_name, count(caller) AS callers
    ORDER BY callers DESC LIMIT 10
"""):
    print(f"  {row['f.qualified_name']}: {row['callers']} callers")

# Largest files by entity count
print("\n--- Largest files ---")
for row in graph.cypher("""
    MATCH (f:File)-[:DEFINES]->(item)
    RETURN f.file_path, f.loc AS lines, count(item) AS entities
    ORDER BY entities DESC LIMIT 10
"""):
    print(f"  {row['f.file_path']}: {row['entities']} entities, {row['lines']} lines")

# Cross-file dependencies
print("\n--- Cross-file call dependencies ---")
for row in graph.cypher("""
    MATCH (a:Function)-[:CALLS]->(b:Function)
    WHERE a.file_path <> b.file_path
    WITH a.file_path AS source, b.file_path AS target, count(*) AS calls
    RETURN source, target, calls
    ORDER BY calls DESC LIMIT 10
"""):
    print(f"  {row['source']} -> {row['target']}: {row['calls']} calls")

# PageRank on the call graph (most important functions)
print("\n--- PageRank (most important functions) ---")
for row in graph.cypher("""
    CALL pagerank({connection_types: ['CALLS'], top_k: 10})
    YIELD node, score
    RETURN node.qualified_name, round(score, 4) AS score
"""):
    print(f"  {row['node.qualified_name']}: {row['score']}")
