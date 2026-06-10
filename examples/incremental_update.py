#!/usr/bin/env python3
"""Incrementally update a graph with a new data snapshot.

Demonstrates: add_nodes(conflict_handling='update') to merge a v2 snapshot
into an existing graph — existing nodes (matched by unique id) have their
properties updated in place, and brand-new ids are inserted.

'update' is the default conflict_handling; it is passed explicitly here to
make the merge semantics obvious.
"""

import pandas as pd

import kglite

graph = kglite.KnowledgeGraph()

# -- v1 snapshot -----------------------------------------------------------

v1 = pd.DataFrame(
    {
        "product_id": [1, 2, 3],
        "name": ["Widget", "Gadget", "Gizmo"],
        "price": [9.99, 19.99, 4.99],
    }
)
report = graph.add_nodes(v1, "Product", "product_id", "name")
print(f"v1 loaded: {report['nodes_created']} created")

# -- v2 snapshot: price changes on existing products + one new product -----

v2 = pd.DataFrame(
    {
        "product_id": [1, 2, 4],  # 1 & 2 already exist; 4 is new
        "name": ["Widget", "Gadget", "Doohickey"],
        "price": [11.49, 18.99, 14.99],  # updated prices for 1 & 2
    }
)
report = graph.add_nodes(v2, "Product", "product_id", "name", conflict_handling="update")
print(f"v2 merged: {report['nodes_created']} created, {report['nodes_updated']} updated")

# -- Verify the merge ------------------------------------------------------

print("\n--- Catalogue after merge ---")
for row in graph.cypher("MATCH (p:Product) RETURN p.title AS name, p.price AS price ORDER BY name"):
    print(f"  {row['name']}: {row['price']}")
