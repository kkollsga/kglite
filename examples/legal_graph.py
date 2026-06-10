#!/usr/bin/env python3
"""Build a legal knowledge graph from DataFrames (index-based loading).

Demonstrates: add_nodes, add_connections, create_index, create_range_index,
set_parent_type, valid_at, Cypher queries, save/load.

Domain: laws, regulations, court decisions, and citation relationships.
Adapt the DataFrames to your own data source (CSV, database, API).
"""

from pathlib import Path
import tempfile

import pandas as pd

import kglite

graph = kglite.KnowledgeGraph()

# -- Nodes -----------------------------------------------------------------

laws = pd.DataFrame(
    {
        "law_id": [1, 2, 3, 4, 5],
        "name": [
            "Data Protection Act",
            "Consumer Rights Act",
            "Employment Act",
            "Environmental Protection Act",
            "Competition Act",
        ],
        "year": [2018, 2015, 2010, 2020, 2012],
        "status": ["active", "active", "amended", "active", "active"],
        "category": ["privacy", "consumer", "labor", "environment", "competition"],
    }
)
graph.add_nodes(laws, "Law", "law_id", "name")

# Sections are children of laws (supporting type)
sections = pd.DataFrame(
    {
        "section_id": list(range(100, 120)),
        "name": [f"Section {i}" for i in range(1, 21)],
        "text": [f"Provision text for section {i}..." for i in range(1, 21)],
        "law_id": [1] * 5 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 3,
    }
)
graph.add_nodes(sections, "Section", "section_id", "name")
graph.set_parent_type("Section", "Law")

decisions = pd.DataFrame(
    {
        "decision_id": list(range(200, 212)),
        "name": [f"Case {2018 + i // 3}/{100 + i}" for i in range(12)],
        "court": ["Supreme Court"] * 4 + ["Court of Appeal"] * 4 + ["District Court"] * 4,
        "date": [f"{2019 + i // 4}-{1 + (i % 12):02d}-15" for i in range(12)],
        "outcome": ["upheld", "overturned", "upheld", "settled"] * 3,
    }
)
graph.add_nodes(decisions, "CourtDecision", "decision_id", "name")

regulations = pd.DataFrame(
    {
        "reg_id": list(range(300, 306)),
        "name": [
            "Privacy Regulation",
            "Data Breach Notification Rules",
            "Consumer Complaint Procedure",
            "Emission Standards",
            "Waste Management Rules",
            "Merger Review Guidelines",
        ],
        "effective_from": ["2018-05-25", "2019-01-01", "2016-03-01", "2021-01-01", "2020-06-15", "2013-07-01"],
        "effective_to": [None, None, "2023-12-31", None, None, "2024-06-30"],
    }
)
graph.add_nodes(regulations, "Regulation", "reg_id", "name")

# -- Connections -----------------------------------------------------------

# Laws -> Regulations (implements)
graph.add_connections(
    pd.DataFrame({"law_id": [1, 1, 2, 4, 4, 5], "reg_id": [300, 301, 302, 303, 304, 305]}),
    "IMPLEMENTS",
    "Law",
    "law_id",
    "Regulation",
    "reg_id",
)

# Laws -> Sections (has)
graph.add_connections(
    sections[["law_id", "section_id"]],
    "HAS_SECTION",
    "Law",
    "law_id",
    "Section",
    "section_id",
)

# Decisions cite laws and sections
cites = pd.DataFrame(
    {
        "decision_id": [200, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211],
        "law_id": [1, 2, 1, 3, 1, 4, 2, 3, 5, 1, 4, 2, 5],
        "weight": [1.0, 0.5, 1.0, 1.0, 0.8, 1.0, 0.7, 1.0, 1.0, 0.6, 0.9, 1.0, 0.8],
    }
)
graph.add_connections(cites, "CITES", "CourtDecision", "decision_id", "Law", "law_id", columns=["weight"])

# -- Indexes ---------------------------------------------------------------

graph.create_index("Law", "category")
graph.create_range_index("CourtDecision", "date")
graph.create_range_index("Law", "year")

# -- Save / load round-trip ------------------------------------------------
# Saved to a temp dir so running the example leaves nothing behind;
# point the path anywhere to keep the file.

kgl_path = Path(tempfile.mkdtemp(prefix="legal_graph_")) / "legal_graph.kgl"
graph.save(str(kgl_path))
graph = kglite.load(str(kgl_path))
print(f"Built: {graph.schema()['node_count']} nodes, {graph.schema()['edge_count']} edges")

# -- Example queries -------------------------------------------------------

# Most-cited laws
print("\n--- Most-cited laws ---")
for row in graph.cypher("""
    MATCH (d:CourtDecision)-[:CITES]->(l:Law)
    RETURN l.title, count(d) AS citations
    ORDER BY citations DESC
"""):
    print(f"  {row['l.title']}: {row['citations']}")

# Temporal: regulations active at a point in time
print("\n--- Regulations active on 2022-01-01 ---")
for row in graph.cypher("""
    MATCH (r:Regulation)
    WHERE valid_at(r, '2022-01-01', 'effective_from', 'effective_to')
    RETURN r.title
"""):
    print(f"  {row['r.title']}")

# Court decisions citing privacy laws
print("\n--- Privacy-related decisions ---")
for row in graph.cypher("""
    MATCH (d:CourtDecision)-[:CITES]->(l:Law)
    WHERE l.category = 'privacy'
    RETURN d.title, l.title, d.court
    ORDER BY d.date DESC
"""):
    print(f"  {row['d.title']} -> {row['l.title']} ({row['d.court']})")
