"""
Benchmark: SQLite vs DuckDB vs KGLite — graph traversal performance.

Three embedded/in-process engines (no network overhead), all using in-memory storage.
SQLite and DuckDB use recursive CTEs; KGLite uses native graph traversal.

Dataset: ~30k nodes, ~220k edges — heterogeneous academic knowledge graph
  Nodes: Person (10k), Paper (20k), Topic (500), Institution (200)
  Edges: AUTHORED, CITES, COVERS, AFFILIATED, COLLABORATES
  Properties: h_index, pub_count, field, year, citation_count, venue, country, ranking

Queries: 15 traversal benchmarks + 10 analytical + 10 advanced analytical (OPTIONAL MATCH,
  EXISTS/NOT EXISTS, CASE aggregation, UNION, count(DISTINCT), multi-step WITH,
  correlated top-N, out-degree distribution, multi-aggregate).

Usage:
    pip install kglite duckdb
    python bench_graph_traversal.py
"""

import random
import sqlite3
import statistics
import time

import duckdb

import kglite

# ── Config ──────────────────────────────────────────────────────────────────

SEED = 42
N_PERSONS = 10_000
N_PAPERS = 20_000
N_TOPICS = 500
N_INSTITUTIONS = 200
RUNS = 10  # median of N runs per query

FIELDS = ["ML", "NLP", "CV", "Theory", "Systems", "DB", "Security", "HCI"]
VENUES = ["NeurIPS", "ICML", "ACL", "CVPR", "AAAI", "SIGMOD", "VLDB", "KDD", "WWW", "ICLR"]
COUNTRIES = ["US", "UK", "DE", "CN", "JP", "CA", "FR", "KR", "AU", "IN"]

# ── Data generation ─────────────────────────────────────────────────────────


def generate_data():
    """Generate deterministic graph data. Returns dict of edge lists."""
    rng = random.Random(SEED)

    # Edge lists
    authored = []  # (person_id, paper_id)
    cites = []  # (paper_id, cited_paper_id)
    covers = []  # (paper_id, topic_id)
    affiliated = []  # (person_id, institution_id)
    collaborates = []  # (person_id, person_id)

    # Each person authors 2-10 papers
    for pid in range(N_PERSONS):
        n_papers = rng.randint(2, 10)
        for _ in range(n_papers):
            authored.append((pid, rng.randint(0, N_PAPERS - 1)))

    # Each paper cites 1-8 other papers
    for paper_id in range(N_PAPERS):
        n_cited = rng.randint(1, 8)
        for _ in range(n_cited):
            cited = rng.randint(0, N_PAPERS - 1)
            if cited != paper_id:
                cites.append((paper_id, cited))

    # Each paper covers 1-4 topics
    for paper_id in range(N_PAPERS):
        n_topics = rng.randint(1, 4)
        for _ in range(n_topics):
            covers.append((paper_id, rng.randint(0, N_TOPICS - 1)))

    # Each person affiliated with 1 institution
    for pid in range(N_PERSONS):
        affiliated.append((pid, rng.randint(0, N_INSTITUTIONS - 1)))

    # ~3 collaborators per person (undirected, stored both ways)
    for pid in range(N_PERSONS):
        n_collabs = rng.randint(1, 5)
        for _ in range(n_collabs):
            other = rng.randint(0, N_PERSONS - 1)
            if other != pid:
                collaborates.append((pid, other))

    # ── Guaranteed graph structure ──────────────────────────────────────────
    # Random sparse graphs have almost no triangles or shared citations.
    # Add deterministic structure so pattern-matching benchmarks find results.

    # Citation triangles: paper i → i+1 → i+2 → i (100 triangles)
    for i in range(0, 300, 3):
        cites.append((i, i + 1))
        cites.append((i + 1, i + 2))
        cites.append((i + 2, i))

    # Collaboration triangles: person i — i+1 — i+2 — i (100 triangles)
    for i in range(0, 300, 3):
        collaborates.append((i, i + 1))
        collaborates.append((i + 1, i + 2))
        collaborates.append((i + 2, i))

    # Hub citations: papers 0–99 all cite papers 5000, 5001 (shared targets)
    for pid in range(100):
        cites.append((pid, 5000))
        cites.append((pid, 5001))

    # ── Node properties (separate RNG to keep edge data stable) ─────────────
    prop_rng = random.Random(SEED + 100)

    person_props = [
        {
            "id": pid,
            "name": f"Person_{pid}",
            "h_index": prop_rng.randint(1, 100),
            "pub_count": prop_rng.randint(1, 200),
            "field": prop_rng.choice(FIELDS),
            "career_start": prop_rng.randint(1980, 2023),
        }
        for pid in range(N_PERSONS)
    ]
    paper_props = [
        {
            "id": pid,
            "title": f"Paper_{pid}",
            "year": prop_rng.randint(1995, 2024),
            "citation_count": prop_rng.randint(0, 2000),
            "venue": prop_rng.choice(VENUES),
            "field": prop_rng.choice(FIELDS),
        }
        for pid in range(N_PAPERS)
    ]
    topic_props = [
        {"id": tid, "name": f"Topic_{tid}", "parent_field": prop_rng.choice(FIELDS)} for tid in range(N_TOPICS)
    ]
    inst_props = [
        {"id": iid, "name": f"Inst_{iid}", "country": prop_rng.choice(COUNTRIES), "ranking": prop_rng.randint(1, 200)}
        for iid in range(N_INSTITUTIONS)
    ]

    return {
        "authored": authored,
        "cites": cites,
        "covers": covers,
        "affiliated": affiliated,
        "collaborates": collaborates,
        "person_props": person_props,
        "paper_props": paper_props,
        "topic_props": topic_props,
        "inst_props": inst_props,
    }


# ── SQLite setup ────────────────────────────────────────────────────────────


def setup_sqlite(data):
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA cache_size=-64000")  # 64MB cache

    db.executescript("""
        CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, h_index INT,
                             pub_count INT, field TEXT, career_start INT);
        CREATE TABLE paper (id INTEGER PRIMARY KEY, title TEXT, year INT,
                            citation_count INT, venue TEXT, field TEXT);
        CREATE TABLE topic (id INTEGER PRIMARY KEY, name TEXT, parent_field TEXT);
        CREATE TABLE institution (id INTEGER PRIMARY KEY, name TEXT, country TEXT,
                                  ranking INT);

        CREATE TABLE authored (person_id INT, paper_id INT);
        CREATE TABLE cites (paper_id INT, cited_id INT);
        CREATE TABLE covers (paper_id INT, topic_id INT);
        CREATE TABLE affiliated (person_id INT, inst_id INT);
        CREATE TABLE collaborates (pid1 INT, pid2 INT);
    """)

    db.executemany(
        "INSERT INTO person VALUES (?,?,?,?,?,?)",
        [
            (p["id"], p["name"], p["h_index"], p["pub_count"], p["field"], p["career_start"])
            for p in data["person_props"]
        ],
    )
    db.executemany(
        "INSERT INTO paper VALUES (?,?,?,?,?,?)",
        [(p["id"], p["title"], p["year"], p["citation_count"], p["venue"], p["field"]) for p in data["paper_props"]],
    )
    db.executemany(
        "INSERT INTO topic VALUES (?,?,?)", [(t["id"], t["name"], t["parent_field"]) for t in data["topic_props"]]
    )
    db.executemany(
        "INSERT INTO institution VALUES (?,?,?,?)",
        [(i["id"], i["name"], i["country"], i["ranking"]) for i in data["inst_props"]],
    )
    db.executemany("INSERT INTO authored VALUES (?,?)", data["authored"])
    db.executemany("INSERT INTO cites VALUES (?,?)", data["cites"])
    db.executemany("INSERT INTO covers VALUES (?,?)", data["covers"])
    db.executemany("INSERT INTO affiliated VALUES (?,?)", data["affiliated"])
    db.executemany("INSERT INTO collaborates VALUES (?,?)", data["collaborates"])

    db.executescript("""
        CREATE INDEX idx_auth_p ON authored(person_id);
        CREATE INDEX idx_auth_pp ON authored(paper_id);
        CREATE INDEX idx_cite_p ON cites(paper_id);
        CREATE INDEX idx_cite_c ON cites(cited_id);
        CREATE INDEX idx_cov_p ON covers(paper_id);
        CREATE INDEX idx_cov_t ON covers(topic_id);
        CREATE INDEX idx_aff_p ON affiliated(person_id);
        CREATE INDEX idx_aff_i ON affiliated(inst_id);
        CREATE INDEX idx_col_1 ON collaborates(pid1);
        CREATE INDEX idx_col_2 ON collaborates(pid2);
        CREATE INDEX idx_person_field ON person(field);
        CREATE INDEX idx_person_hindex ON person(h_index);
        CREATE INDEX idx_paper_year ON paper(year);
        CREATE INDEX idx_paper_venue ON paper(venue);
        CREATE INDEX idx_paper_field ON paper(field);
        CREATE INDEX idx_paper_citation ON paper(citation_count);
    """)
    db.commit()
    return db


# ── DuckDB setup ───────────────────────────────────────────────────────────


def setup_duckdb(data):
    import pandas as pd

    db = duckdb.connect(":memory:")

    # Node tables with properties — bulk load via DataFrames
    for name, props_key in [
        ("person", "person_props"),
        ("paper", "paper_props"),
        ("topic", "topic_props"),
        ("institution", "inst_props"),
    ]:
        df = pd.DataFrame(data[props_key])  # noqa: F841
        db.execute(f"CREATE TABLE {name} AS SELECT * FROM df")

    for name, cols in [
        ("authored", ["person_id", "paper_id"]),
        ("cites", ["paper_id", "cited_id"]),
        ("covers", ["paper_id", "topic_id"]),
        ("affiliated", ["person_id", "inst_id"]),
        ("collaborates", ["pid1", "pid2"]),
    ]:
        frame = pd.DataFrame(data[name], columns=cols)  # noqa: F841
        db.execute(f"CREATE TABLE {name} AS SELECT * FROM frame")

    db.execute("CREATE INDEX idx_dk_auth_p ON authored(person_id)")
    db.execute("CREATE INDEX idx_dk_auth_pp ON authored(paper_id)")
    db.execute("CREATE INDEX idx_dk_cite_p ON cites(paper_id)")
    db.execute("CREATE INDEX idx_dk_cite_c ON cites(cited_id)")
    db.execute("CREATE INDEX idx_dk_cov_p ON covers(paper_id)")
    db.execute("CREATE INDEX idx_dk_cov_t ON covers(topic_id)")
    db.execute("CREATE INDEX idx_dk_aff_p ON affiliated(person_id)")
    db.execute("CREATE INDEX idx_dk_aff_i ON affiliated(inst_id)")
    db.execute("CREATE INDEX idx_dk_col_1 ON collaborates(pid1)")
    db.execute("CREATE INDEX idx_dk_col_2 ON collaborates(pid2)")
    # Property indexes
    db.execute("CREATE INDEX idx_dk_person_field ON person(field)")
    db.execute("CREATE INDEX idx_dk_person_hindex ON person(h_index)")
    db.execute("CREATE INDEX idx_dk_paper_year ON paper(year)")
    db.execute("CREATE INDEX idx_dk_paper_venue ON paper(venue)")
    db.execute("CREATE INDEX idx_dk_paper_field ON paper(field)")
    db.execute("CREATE INDEX idx_dk_paper_citation ON paper(citation_count)")
    return db


# ── KGLite setup ────────────────────────────────────────────────────────────


def setup_kglite(data):
    import pandas as pd

    g = kglite.KnowledgeGraph()

    g.add_nodes(pd.DataFrame(data["person_props"]), "Person", "id", "id")
    g.add_nodes(pd.DataFrame(data["paper_props"]), "Paper", "id", "id")
    g.add_nodes(pd.DataFrame(data["topic_props"]), "Topic", "id", "id")
    g.add_nodes(pd.DataFrame(data["inst_props"]), "Institution", "id", "id")

    g.add_connections(
        pd.DataFrame(data["authored"], columns=["person_id", "paper_id"]),
        "AUTHORED",
        "Person",
        "person_id",
        "Paper",
        "paper_id",
    )
    g.add_connections(
        pd.DataFrame(data["cites"], columns=["paper_id", "cited_id"]),
        "CITES",
        "Paper",
        "paper_id",
        "Paper",
        "cited_id",
    )
    g.add_connections(
        pd.DataFrame(data["covers"], columns=["paper_id", "topic_id"]),
        "COVERS",
        "Paper",
        "paper_id",
        "Topic",
        "topic_id",
    )
    g.add_connections(
        pd.DataFrame(data["affiliated"], columns=["person_id", "inst_id"]),
        "AFFILIATED",
        "Person",
        "person_id",
        "Institution",
        "inst_id",
    )
    g.add_connections(
        pd.DataFrame(data["collaborates"], columns=["pid1", "pid2"]),
        "COLLABORATES",
        "Person",
        "pid1",
        "Person",
        "pid2",
    )

    # Property indexes — match the SQLite/DuckDB index coverage for fair comparison
    g.create_index("Person", "field")
    g.create_index("Paper", "venue")
    g.create_index("Paper", "field")
    g.create_range_index("Paper", "year")
    g.create_range_index("Paper", "citation_count")
    g.create_range_index("Person", "h_index")

    return g


# ── Benchmark harness ──────────────────────────────────────────────────────


def bench(fn, runs=RUNS):
    """Return median execution time in ms over `runs` iterations."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    if result is None:
        count = 0
    elif isinstance(result, (int, float)):
        count = int(result)
    elif isinstance(result, tuple):
        count = result[0] if result[0] is not None else 0
    else:
        count = len(list(result))
    return statistics.median(times), count


# ── Queries ─────────────────────────────────────────────────────────────────
#
# For each hop depth we follow citation chains (Paper→Paper via CITES).
# This is the purest traversal benchmark: same entity type, same edge type,
# varying only the depth — isolating the join-vs-pointer cost.

START_PAPER = 42  # deterministic start node
START_PERSON = 7


def sqlite_citation_hops(db, depth):
    """N-hop citation traversal using recursive CTE."""
    return db.execute(f"""
        WITH RECURSIVE chain(paper_id, d) AS (
            SELECT cited_id, 1 FROM cites WHERE paper_id = {START_PAPER}
            UNION
            SELECT c.cited_id, chain.d + 1
            FROM cites c JOIN chain ON c.paper_id = chain.paper_id
            WHERE chain.d < {depth}
        )
        SELECT DISTINCT paper_id FROM chain
    """).fetchall()


def kglite_citation_hops(g, depth):
    """N-hop citation traversal — full materialization."""
    return g.cypher(f"""
        MATCH (start:Paper {{id: {START_PAPER}}})-[:CITES*1..{depth}]->(p:Paper)
        RETURN DISTINCT p.id
    """).to_list()


def kglite_citation_hops_raw(g, depth):
    """N-hop citation traversal — raw Cypher (no Python materialization)."""
    result = g.cypher(f"""
        MATCH (start:Paper {{id: {START_PAPER}}})-[:CITES*1..{depth}]->(p:Paper)
        RETURN DISTINCT p.id
    """)
    return len(result)


def sqlite_collab_hops(db, depth):
    """N-hop collaboration traversal using recursive CTE."""
    return db.execute(f"""
        WITH RECURSIVE chain(person_id, d) AS (
            SELECT pid2, 1 FROM collaborates WHERE pid1 = {START_PERSON}
            UNION
            SELECT c.pid2, chain.d + 1
            FROM collaborates c JOIN chain ON c.pid1 = chain.person_id
            WHERE chain.d < {depth}
        )
        SELECT DISTINCT person_id FROM chain
    """).fetchall()


def kglite_collab_hops(g, depth):
    """N-hop collaboration traversal — full materialization."""
    return g.cypher(f"""
        MATCH (start:Person {{id: {START_PERSON}}})-[:COLLABORATES*1..{depth}]->(p:Person)
        RETURN DISTINCT p.id
    """).to_list()


def kglite_collab_hops_raw(g, depth):
    """N-hop collaboration traversal — raw Cypher (no Python materialization)."""
    result = g.cypher(f"""
        MATCH (start:Person {{id: {START_PERSON}}})-[:COLLABORATES*1..{depth}]->(p:Person)
        RETURN DISTINCT p.id
    """)
    return len(result)


def sqlite_hetero_traversal(db):
    """4-hop heterogeneous: Person→Paper→Topic→Paper→Person (via authored/covers)."""
    return db.execute(f"""
        SELECT DISTINCT p2.person_id
        FROM authored a1
        JOIN covers c1 ON a1.paper_id = c1.paper_id
        JOIN covers c2 ON c1.topic_id = c2.topic_id AND c2.paper_id != c1.paper_id
        JOIN authored p2 ON c2.paper_id = p2.paper_id
        WHERE a1.person_id = {START_PERSON}
    """).fetchall()


def kglite_hetero_traversal(g):
    """4-hop heterogeneous: Person→Paper→Topic→Paper→Person — full materialization."""
    return g.cypher(f"""
        MATCH (start:Person {{id: {START_PERSON}}})-[:AUTHORED]->(paper:Paper)
              -[:COVERS]->(t:Topic)<-[:COVERS]-(other_paper:Paper)
              <-[:AUTHORED]-(other:Person)
        WHERE other.id <> {START_PERSON}
        RETURN DISTINCT other.id
    """).to_list()


def kglite_hetero_traversal_raw(g):
    """4-hop heterogeneous — raw Cypher (no Python materialization)."""
    result = g.cypher(f"""
        MATCH (start:Person {{id: {START_PERSON}}})-[:AUTHORED]->(paper:Paper)
              -[:COVERS]->(t:Topic)<-[:COVERS]-(other_paper:Paper)
              <-[:AUTHORED]-(other:Person)
        WHERE other.id <> {START_PERSON}
        RETURN DISTINCT other.id
    """)
    return len(result)


def sqlite_shortest_path(db, source, target):
    """BFS shortest path between two persons via collaborates."""
    return db.execute(f"""
        WITH RECURSIVE bfs(person_id, depth) AS (
            SELECT {source}, 0
            UNION
            SELECT c.pid2, bfs.depth + 1
            FROM collaborates c JOIN bfs ON c.pid1 = bfs.person_id
            WHERE bfs.depth < 8
        )
        SELECT MIN(depth) FROM bfs WHERE person_id = {target}
    """).fetchone()


def kglite_shortest_path(g, source, target):
    """Shortest path between two persons via collaborates."""
    result = g.cypher(f"""
        MATCH p = shortestPath(
            (a:Person {{id: {source}}})-[:COLLABORATES*..8]-(b:Person {{id: {target}}})
        )
        RETURN length(p) AS dist
    """).to_list()
    return result[0]["dist"] if result else None


# ── Triangle detection ─────────────────────────────────────────────────────
# Find citation triangles: A cites B, B cites C, C cites A.
# Classic graph problem — requires 3-way self-join in SQL.

TRIANGLE_PAPERS = list(range(300))  # papers 0-299 have guaranteed triangles


def sqlite_triangles(db):
    """Count citation triangles among sampled papers."""
    placeholders = ",".join(str(p) for p in TRIANGLE_PAPERS)
    return db.execute(f"""
        SELECT count(*) FROM (
            SELECT DISTINCT c1.paper_id AS a, c2.paper_id AS b, c3.paper_id AS c
            FROM cites c1
            JOIN cites c2 ON c1.cited_id = c2.paper_id
            JOIN cites c3 ON c2.cited_id = c3.paper_id AND c3.cited_id = c1.paper_id
            WHERE c1.paper_id IN ({placeholders})
        )
    """).fetchone()


def kglite_triangles_raw(g):
    """Count citation triangles among sampled papers."""
    ids = TRIANGLE_PAPERS
    result = g.cypher(f"""
        MATCH (a:Paper)-[:CITES]->(b:Paper)-[:CITES]->(c:Paper)-[:CITES]->(a)
        WHERE a.id IN {ids} AND a.id < b.id AND b.id < c.id
        RETURN count(*) AS triangles
    """).to_list()
    return result[0]["triangles"] if result else 0


# ── Reachability check (early termination) ────────────────────────────────
# "Can person X reach person Y within N hops?" — graph DB stops on first hit.


def sqlite_reachable(db, source, target, max_hops):
    """Reachability check — SQLite must materialize entire BFS frontier."""
    return db.execute(f"""
        WITH RECURSIVE bfs(person_id, d) AS (
            SELECT {source}, 0
            UNION
            SELECT c.pid2, bfs.d + 1
            FROM collaborates c JOIN bfs ON c.pid1 = bfs.person_id
            WHERE bfs.d < {max_hops}
        )
        SELECT 1 FROM bfs WHERE person_id = {target} LIMIT 1
    """).fetchone()


def kglite_reachable_raw(g, source, target, max_hops):
    """Reachability check — Cypher with LIMIT 1 for early termination."""
    result = g.cypher(f"""
        MATCH (a:Person {{id: {source}}})-[:COLLABORATES*1..{max_hops}]-(b:Person {{id: {target}}})
        RETURN 1 AS found
        LIMIT 1
    """)
    return len(result)


# ── Neighborhood aggregation ──────────────────────────────────────────────
# Count distinct topics covered by a person's N-hop collaboration network.
# Multi-type: Person → COLLABORATES → Person → AUTHORED → Paper → COVERS → Topic


def sqlite_neighbor_topics(db, max_hops):
    """Topics covered by papers authored by N-hop collaborators."""
    return db.execute(f"""
        WITH RECURSIVE collab_net(person_id, d) AS (
            SELECT pid2, 1 FROM collaborates WHERE pid1 = {START_PERSON}
            UNION
            SELECT c.pid2, collab_net.d + 1
            FROM collaborates c JOIN collab_net ON c.pid1 = collab_net.person_id
            WHERE collab_net.d < {max_hops}
        )
        SELECT count(DISTINCT cv.topic_id)
        FROM collab_net cn
        JOIN authored a ON cn.person_id = a.person_id
        JOIN covers cv ON a.paper_id = cv.paper_id
    """).fetchone()


def kglite_neighbor_topics_raw(g, max_hops):
    """Topics covered by papers authored by N-hop collaborators."""
    result = g.cypher(f"""
        MATCH (start:Person {{id: {START_PERSON}}})-[:COLLABORATES*1..{max_hops}]->(collab:Person)
              -[:AUTHORED]->(paper:Paper)-[:COVERS]->(t:Topic)
        RETURN count(DISTINCT t.id) AS topic_count
    """).to_list()
    return result[0]["topic_count"] if result else 0


# ── Fan-out / influence spread ─────────────────────────────────────────────
# How many unique people can each of N start nodes reach in K hops?
# Batch query that exercises the BFS engine repeatedly.


def sqlite_fan_out(db, persons, max_hops):
    """Count reachable persons within K hops for N start nodes."""
    total = 0
    for pid in persons:
        row = db.execute(f"""
            WITH RECURSIVE chain(person_id, d) AS (
                SELECT pid2, 1 FROM collaborates WHERE pid1 = {pid}
                UNION
                SELECT c.pid2, chain.d + 1
                FROM collaborates c JOIN chain ON c.pid1 = chain.person_id
                WHERE chain.d < {max_hops}
            )
            SELECT count(DISTINCT person_id) FROM chain
        """).fetchone()
        total += row[0]
    return total


def kglite_fan_out_raw(g, persons, max_hops):
    """Count reachable persons within K hops for N start nodes."""
    result = g.cypher(f"""
        UNWIND {persons} AS pid
        MATCH (start:Person {{id: pid}})-[:COLLABORATES*1..{max_hops}]->(p:Person)
        RETURN pid, count(DISTINCT p.id) AS reach
    """).to_list()
    return sum(row["reach"] for row in result)


# ── Multi-start deep traversal ────────────────────────────────────────────
# Multiple start nodes each doing deep BFS — stresses both engines.


def sqlite_multi_start_deep(db, starts, depth):
    """Deep citation traversal from multiple start nodes."""
    total = 0
    for sid in starts:
        rows = db.execute(f"""
            WITH RECURSIVE chain(paper_id, d) AS (
                SELECT cited_id, 1 FROM cites WHERE paper_id = {sid}
                UNION
                SELECT c.cited_id, chain.d + 1
                FROM cites c JOIN chain ON c.paper_id = chain.paper_id
                WHERE chain.d < {depth}
            )
            SELECT count(DISTINCT paper_id) FROM chain
        """).fetchone()
        total += rows[0]
    return total


def kglite_multi_start_deep_raw(g, starts, depth):
    """Deep citation traversal from multiple start nodes."""
    result = g.cypher(f"""
        UNWIND {starts} AS sid
        MATCH (start:Paper {{id: sid}})-[:CITES*1..{depth}]->(p:Paper)
        RETURN sid, count(DISTINCT p.id) AS reach
    """).to_list()
    return sum(row["reach"] for row in result)


# ── DuckDB queries ─────────────────────────────────────────────────────────
# Same recursive CTE SQL as SQLite — DuckDB uses standard SQL syntax.
# DuckDB returns results via .fetchall() like SQLite.


def duckdb_citation_hops(db, depth):
    return db.execute(f"""
        WITH RECURSIVE chain(paper_id, d) AS (
            SELECT cited_id, 1 FROM cites WHERE paper_id = {START_PAPER}
            UNION
            SELECT c.cited_id, chain.d + 1
            FROM cites c JOIN chain ON c.paper_id = chain.paper_id
            WHERE chain.d < {depth}
        )
        SELECT DISTINCT paper_id FROM chain
    """).fetchall()


def duckdb_collab_hops(db, depth):
    return db.execute(f"""
        WITH RECURSIVE chain(person_id, d) AS (
            SELECT pid2, 1 FROM collaborates WHERE pid1 = {START_PERSON}
            UNION
            SELECT c.pid2, chain.d + 1
            FROM collaborates c JOIN chain ON c.pid1 = chain.person_id
            WHERE chain.d < {depth}
        )
        SELECT DISTINCT person_id FROM chain
    """).fetchall()


def duckdb_hetero_traversal(db):
    return db.execute(f"""
        SELECT DISTINCT p2.person_id
        FROM authored a1
        JOIN covers c1 ON a1.paper_id = c1.paper_id
        JOIN covers c2 ON c1.topic_id = c2.topic_id AND c2.paper_id != c1.paper_id
        JOIN authored p2 ON c2.paper_id = p2.paper_id
        WHERE a1.person_id = {START_PERSON}
    """).fetchall()


def duckdb_shortest_path(db, source, target):
    return db.execute(f"""
        WITH RECURSIVE bfs(person_id, depth) AS (
            SELECT {source}, 0
            UNION
            SELECT c.pid2, bfs.depth + 1
            FROM collaborates c JOIN bfs ON c.pid1 = bfs.person_id
            WHERE bfs.depth < 8
        )
        SELECT MIN(depth) FROM bfs WHERE person_id = {target}
    """).fetchone()


def duckdb_triangles(db):
    placeholders = ",".join(str(p) for p in TRIANGLE_PAPERS)
    return db.execute(f"""
        SELECT count(*) FROM (
            SELECT DISTINCT c1.paper_id AS a, c2.paper_id AS b, c3.paper_id AS c
            FROM cites c1
            JOIN cites c2 ON c1.cited_id = c2.paper_id
            JOIN cites c3 ON c2.cited_id = c3.paper_id AND c3.cited_id = c1.paper_id
            WHERE c1.paper_id IN ({placeholders})
        )
    """).fetchone()


def duckdb_reachable(db, source, target, max_hops):
    return db.execute(f"""
        WITH RECURSIVE bfs(person_id, d) AS (
            SELECT {source}, 0
            UNION
            SELECT c.pid2, bfs.d + 1
            FROM collaborates c JOIN bfs ON c.pid1 = bfs.person_id
            WHERE bfs.d < {max_hops}
        )
        SELECT 1 FROM bfs WHERE person_id = {target} LIMIT 1
    """).fetchone()


def duckdb_neighbor_topics(db, max_hops):
    return db.execute(f"""
        WITH RECURSIVE collab_net(person_id, d) AS (
            SELECT pid2, 1 FROM collaborates WHERE pid1 = {START_PERSON}
            UNION
            SELECT c.pid2, collab_net.d + 1
            FROM collaborates c JOIN collab_net ON c.pid1 = collab_net.person_id
            WHERE collab_net.d < {max_hops}
        )
        SELECT count(DISTINCT cv.topic_id)
        FROM collab_net cn
        JOIN authored a ON cn.person_id = a.person_id
        JOIN covers cv ON a.paper_id = cv.paper_id
    """).fetchone()


def duckdb_fan_out(db, persons, max_hops):
    total = 0
    for pid in persons:
        row = db.execute(f"""
            WITH RECURSIVE chain(person_id, d) AS (
                SELECT pid2, 1 FROM collaborates WHERE pid1 = {pid}
                UNION
                SELECT c.pid2, chain.d + 1
                FROM collaborates c JOIN chain ON c.pid1 = chain.person_id
                WHERE chain.d < {max_hops}
            )
            SELECT count(DISTINCT person_id) FROM chain
        """).fetchone()
        total += row[0]
    return total


def duckdb_multi_start_deep(db, starts, depth):
    total = 0
    for sid in starts:
        rows = db.execute(f"""
            WITH RECURSIVE chain(paper_id, d) AS (
                SELECT cited_id, 1 FROM cites WHERE paper_id = {sid}
                UNION
                SELECT c.cited_id, chain.d + 1
                FROM cites c JOIN chain ON c.paper_id = chain.paper_id
                WHERE chain.d < {depth}
            )
            SELECT count(DISTINCT paper_id) FROM chain
        """).fetchone()
        total += rows[0]
    return total


# ── DuckDB optimized queries ──────────────────────────────────────────────
# DuckDB is vectorized and optimized for fewer, larger queries.
# These batch multiple BFS traversals into a single CTE.


def duckdb_fan_out_batch(db, persons, max_hops):
    """Single multi-source BFS instead of 50 separate queries."""
    plist = ",".join(str(p) for p in persons)
    rows = db.execute(f"""
        WITH RECURSIVE chain(start_pid, person_id, d) AS (
            SELECT c.pid1, c.pid2, 1
            FROM collaborates c
            WHERE c.pid1 IN ({plist})
            UNION
            SELECT chain.start_pid, c.pid2, chain.d + 1
            FROM collaborates c JOIN chain ON c.pid1 = chain.person_id
            WHERE chain.d < {max_hops}
        )
        SELECT sum(cnt) FROM (
            SELECT count(DISTINCT person_id) AS cnt FROM chain GROUP BY start_pid
        )
    """).fetchone()
    return rows[0]


def duckdb_multi_start_deep_batch(db, starts, depth):
    """Single multi-source deep citation traversal."""
    slist = ",".join(str(s) for s in starts)
    rows = db.execute(f"""
        WITH RECURSIVE chain(start_id, paper_id, d) AS (
            SELECT c.paper_id, c.cited_id, 1
            FROM cites c
            WHERE c.paper_id IN ({slist})
            UNION
            SELECT chain.start_id, c.cited_id, chain.d + 1
            FROM cites c JOIN chain ON c.paper_id = chain.paper_id
            WHERE chain.d < {depth}
        )
        SELECT sum(cnt) FROM (
            SELECT count(DISTINCT paper_id) AS cnt FROM chain GROUP BY start_id
        )
    """).fetchone()
    return rows[0]


# ── Analytical queries ──────────────────────────────────────────────────────
# Property filters, aggregations, window functions, HAVING, WITH chains,
# in-degree centrality, cross-institution joins, string/math ops.


# 1. Property-filtered traversal: ML authors with high h-index → papers


def sqlite_filtered_traversal(db):
    return db.execute("""
        SELECT DISTINCT a.paper_id
        FROM person p JOIN authored a ON p.id = a.person_id
        WHERE p.field = 'ML' AND p.h_index > 80
    """).fetchall()


def duckdb_filtered_traversal(db):
    return db.execute("""
        SELECT DISTINCT a.paper_id
        FROM person p JOIN authored a ON p.id = a.person_id
        WHERE p.field = 'ML' AND p.h_index > 80
    """).fetchall()


def kglite_filtered_traversal(g):
    return len(
        g.cypher("""
        MATCH (p:Person)-[:AUTHORED]->(paper:Paper)
        WHERE p.field = 'ML' AND p.h_index > 80
        RETURN DISTINCT paper.id
    """)
    )


# 2. Multi-hop with intermediate filter: recent papers' 2-hop citations


def sqlite_filtered_multihop(db):
    return db.execute("""
        SELECT DISTINCT c2.cited_id
        FROM paper p
        JOIN cites c1 ON p.id = c1.paper_id
        JOIN cites c2 ON c1.cited_id = c2.paper_id
        WHERE p.year > 2020
    """).fetchall()


def duckdb_filtered_multihop(db):
    return db.execute("""
        SELECT DISTINCT c2.cited_id
        FROM paper p
        JOIN cites c1 ON p.id = c1.paper_id
        JOIN cites c2 ON c1.cited_id = c2.paper_id
        WHERE p.year > 2020
    """).fetchall()


def kglite_filtered_multihop(g):
    return len(
        g.cypher("""
        MATCH (p:Paper)-[:CITES]->(:Paper)-[:CITES]->(c2:Paper)
        WHERE p.year > 2020
        RETURN DISTINCT c2.id
    """)
    )


# 3. Aggregation pipeline: top 20 venue×year by paper count


def sqlite_agg_pipeline(db):
    return db.execute("""
        SELECT venue, year, COUNT(*) AS cnt
        FROM paper GROUP BY venue, year ORDER BY cnt DESC LIMIT 20
    """).fetchall()


def duckdb_agg_pipeline(db):
    return db.execute("""
        SELECT venue, year, COUNT(*) AS cnt
        FROM paper GROUP BY venue, year ORDER BY cnt DESC LIMIT 20
    """).fetchall()


def kglite_agg_pipeline(g):
    return len(
        g.cypher("""
        MATCH (p:Paper)
        RETURN p.venue AS venue, p.year AS year, count(*) AS cnt
        ORDER BY cnt DESC LIMIT 20
    """)
    )


# 4. HAVING clause: prolific venues


def sqlite_having(db):
    return db.execute("""
        SELECT venue, COUNT(*) AS cnt, AVG(citation_count) AS ac
        FROM paper GROUP BY venue
        HAVING COUNT(*) > 200 AND AVG(citation_count) > 100
    """).fetchall()


def duckdb_having(db):
    return db.execute("""
        SELECT venue, COUNT(*) AS cnt, AVG(citation_count) AS ac
        FROM paper GROUP BY venue
        HAVING COUNT(*) > 200 AND AVG(citation_count) > 100
    """).fetchall()


def kglite_having(g):
    return len(
        g.cypher("""
        MATCH (p:Paper)
        RETURN p.venue AS venue, count(*) AS cnt, avg(p.citation_count) AS ac
        HAVING cnt > 200 AND ac > 100
    """)
    )


# 5. Window function: rank authors by h_index within field


def sqlite_window(db):
    return db.execute("""
        SELECT id, field, h_index,
               ROW_NUMBER() OVER (PARTITION BY field ORDER BY h_index DESC) AS rk
        FROM person
    """).fetchall()


def duckdb_window(db):
    return db.execute("""
        SELECT id, field, h_index,
               ROW_NUMBER() OVER (PARTITION BY field ORDER BY h_index DESC) AS rk
        FROM person
    """).fetchall()


def kglite_window(g):
    return len(
        g.cypher("""
        MATCH (p:Person)
        RETURN p.id, p.field, p.h_index,
               row_number() OVER (PARTITION BY p.field ORDER BY p.h_index DESC) AS rk
    """)
    )


# 6. WITH chain: top authors' most-cited papers


def sqlite_with_chain(db):
    return db.execute("""
        WITH top AS (SELECT id FROM person ORDER BY h_index DESC LIMIT 50)
        SELECT p.id, p.citation_count
        FROM top t JOIN authored a ON t.id = a.person_id
        JOIN paper p ON a.paper_id = p.id
        ORDER BY p.citation_count DESC LIMIT 20
    """).fetchall()


def duckdb_with_chain(db):
    return db.execute("""
        WITH top AS (SELECT id FROM person ORDER BY h_index DESC LIMIT 50)
        SELECT p.id, p.citation_count
        FROM top t JOIN authored a ON t.id = a.person_id
        JOIN paper p ON a.paper_id = p.id
        ORDER BY p.citation_count DESC LIMIT 20
    """).fetchall()


def kglite_with_chain(g):
    return len(
        g.cypher("""
        MATCH (a:Person)
        WITH a ORDER BY a.h_index DESC LIMIT 50
        MATCH (a)-[:AUTHORED]->(p:Paper)
        RETURN p.id, p.citation_count
        ORDER BY p.citation_count DESC LIMIT 20
    """)
    )


# 7. In-degree centrality: most-cited papers


def sqlite_indegree(db):
    return db.execute("""
        SELECT cited_id, COUNT(*) AS indegree
        FROM cites GROUP BY cited_id ORDER BY indegree DESC LIMIT 20
    """).fetchall()


def duckdb_indegree(db):
    return db.execute("""
        SELECT cited_id, COUNT(*) AS indegree
        FROM cites GROUP BY cited_id ORDER BY indegree DESC LIMIT 20
    """).fetchall()


def kglite_indegree(g):
    return len(
        g.cypher("""
        MATCH (:Paper)-[:CITES]->(p:Paper)
        RETURN p.id, count(*) AS indegree
        ORDER BY indegree DESC LIMIT 20
    """)
    )


# 8. Cross-institution collaboration


def sqlite_cross_inst(db):
    return db.execute("""
        SELECT a1.inst_id AS i1, a2.inst_id AS i2,
               COUNT(DISTINCT au1.paper_id) AS collabs
        FROM affiliated a1
        JOIN authored au1 ON a1.person_id = au1.person_id
        JOIN authored au2 ON au1.paper_id = au2.paper_id
        JOIN affiliated a2 ON au2.person_id = a2.person_id
        WHERE a1.inst_id < a2.inst_id
        GROUP BY a1.inst_id, a2.inst_id
        ORDER BY collabs DESC LIMIT 20
    """).fetchall()


def duckdb_cross_inst(db):
    return db.execute("""
        SELECT a1.inst_id AS i1, a2.inst_id AS i2,
               COUNT(DISTINCT au1.paper_id) AS collabs
        FROM affiliated a1
        JOIN authored au1 ON a1.person_id = au1.person_id
        JOIN authored au2 ON au1.paper_id = au2.paper_id
        JOIN affiliated a2 ON au2.person_id = a2.person_id
        WHERE a1.inst_id < a2.inst_id
        GROUP BY a1.inst_id, a2.inst_id
        ORDER BY collabs DESC LIMIT 20
    """).fetchall()


def kglite_cross_inst(g):
    return len(
        g.cypher("""
        MATCH (p1:Person)-[:AUTHORED]->(paper:Paper)<-[:AUTHORED]-(p2:Person),
              (p1)-[:AFFILIATED]->(i1:Institution),
              (p2)-[:AFFILIATED]->(i2:Institution)
        WHERE i1.id < i2.id
        RETURN i1.id AS inst1, i2.id AS inst2, count(DISTINCT paper.id) AS collabs
        ORDER BY collabs DESC LIMIT 20
    """)
    )


# 9. Complex filtered scan: multiple predicates


def sqlite_complex_filter(db):
    return db.execute("""
        SELECT COUNT(*) FROM paper
        WHERE year BETWEEN 2015 AND 2022
          AND venue IN ('NeurIPS', 'ICML', 'ICLR')
          AND citation_count > 500
          AND field = 'ML'
    """).fetchone()


def duckdb_complex_filter(db):
    return db.execute("""
        SELECT COUNT(*) FROM paper
        WHERE year BETWEEN 2015 AND 2022
          AND venue IN ('NeurIPS', 'ICML', 'ICLR')
          AND citation_count > 500
          AND field = 'ML'
    """).fetchone()


def kglite_complex_filter(g):
    rows = g.cypher("""
        MATCH (p:Paper)
        WHERE p.year >= 2015 AND p.year <= 2022
          AND p.venue IN ['NeurIPS', 'ICML', 'ICLR']
          AND p.citation_count > 500 AND p.field = 'ML'
        RETURN count(*) AS cnt
    """).to_list()
    return rows[0]["cnt"] if rows else 0


# 10. String + math operations


def sqlite_string_math(db):
    return db.execute("""
        SELECT UPPER(venue) AS v, ROUND(citation_count / 100.0) AS bucket,
               COUNT(*) AS cnt
        FROM paper WHERE citation_count > 200
        GROUP BY UPPER(venue), ROUND(citation_count / 100.0)
        ORDER BY cnt DESC LIMIT 20
    """).fetchall()


def duckdb_string_math(db):
    return db.execute("""
        SELECT UPPER(venue) AS v, ROUND(citation_count / 100.0) AS bucket,
               COUNT(*) AS cnt
        FROM paper WHERE citation_count > 200
        GROUP BY UPPER(venue), ROUND(citation_count / 100.0)
        ORDER BY cnt DESC LIMIT 20
    """).fetchall()


def kglite_string_math(g):
    return len(
        g.cypher("""
        MATCH (p:Paper)
        WHERE p.citation_count > 200
        RETURN toUpper(p.venue) AS v, round(p.citation_count / 100.0) AS bucket,
               count(*) AS cnt
        ORDER BY cnt DESC LIMIT 20
    """)
    )


# ── Advanced analytical queries ───────────────────────────────────────────
# OPTIONAL MATCH, EXISTS/NOT EXISTS, CASE aggregation, UNION, count(DISTINCT),
# multi-step WITH, correlated top-N, out-degree distribution, multi-aggregate.


# 11. OPTIONAL MATCH — papers per person (left outer join)


def sqlite_optional_match(db):
    return db.execute("""
        SELECT p.id, COALESCE(cnt, 0) AS papers
        FROM person p LEFT JOIN (
            SELECT person_id, COUNT(*) AS cnt FROM authored GROUP BY person_id
        ) a ON p.id = a.person_id
        WHERE p.field = 'ML'
        ORDER BY papers DESC LIMIT 20
    """).fetchall()


def duckdb_optional_match(db):
    return db.execute("""
        SELECT p.id, COALESCE(cnt, 0) AS papers
        FROM person p LEFT JOIN (
            SELECT person_id, COUNT(*) AS cnt FROM authored GROUP BY person_id
        ) a ON p.id = a.person_id
        WHERE p.field = 'ML'
        ORDER BY papers DESC LIMIT 20
    """).fetchall()


def kglite_optional_match(g):
    return len(
        g.cypher("""
        MATCH (p:Person)
        WHERE p.field = 'ML'
        OPTIONAL MATCH (p)-[:AUTHORED]->(paper:Paper)
        RETURN p.id, count(paper) AS papers
        ORDER BY papers DESC LIMIT 20
    """)
    )


# 12. EXISTS — NLP persons who collaborate with ML authors


def sqlite_exists(db):
    return db.execute("""
        SELECT DISTINCT p.id FROM person p
        JOIN collaborates c ON p.id = c.pid1
        JOIN person p2 ON c.pid2 = p2.id
        WHERE p.field = 'NLP' AND p2.field = 'ML'
    """).fetchall()


def duckdb_exists(db):
    return db.execute("""
        SELECT DISTINCT p.id FROM person p
        JOIN collaborates c ON p.id = c.pid1
        JOIN person p2 ON c.pid2 = p2.id
        WHERE p.field = 'NLP' AND p2.field = 'ML'
    """).fetchall()


def kglite_exists(g):
    return len(
        g.cypher("""
        MATCH (p:Person)
        WHERE p.field = 'NLP' AND EXISTS {
            MATCH (p)-[:COLLABORATES]->(other:Person)
            WHERE other.field = 'ML'
        }
        RETURN DISTINCT p.id
    """)
    )


# 13. NOT EXISTS — old papers never cited


def sqlite_not_exists(db):
    return db.execute("""
        SELECT p.id FROM paper p
        WHERE p.year < 2000
        AND NOT EXISTS (SELECT 1 FROM cites c WHERE c.cited_id = p.id)
    """).fetchall()


def duckdb_not_exists(db):
    return db.execute("""
        SELECT p.id FROM paper p
        WHERE p.year < 2000
        AND NOT EXISTS (SELECT 1 FROM cites c WHERE c.cited_id = p.id)
    """).fetchall()


def kglite_not_exists(g):
    return len(
        g.cypher("""
        MATCH (p:Paper)
        WHERE p.year < 2000 AND NOT EXISTS {
            MATCH (:Paper)-[:CITES]->(p)
        }
        RETURN p.id
    """)
    )


# 14. Conditional aggregation — ML vs non-ML per venue (CASE in aggregate)


def sqlite_case_agg(db):
    return db.execute("""
        SELECT venue,
               SUM(CASE WHEN field = 'ML' THEN 1 ELSE 0 END) AS ml_count,
               SUM(CASE WHEN field <> 'ML' THEN 1 ELSE 0 END) AS other_count,
               COUNT(*) AS total
        FROM paper GROUP BY venue ORDER BY total DESC LIMIT 10
    """).fetchall()


def duckdb_case_agg(db):
    return db.execute("""
        SELECT venue,
               SUM(CASE WHEN field = 'ML' THEN 1 ELSE 0 END) AS ml_count,
               SUM(CASE WHEN field <> 'ML' THEN 1 ELSE 0 END) AS other_count,
               COUNT(*) AS total
        FROM paper GROUP BY venue ORDER BY total DESC LIMIT 10
    """).fetchall()


def kglite_case_agg(g):
    return len(
        g.cypher("""
        MATCH (p:Paper)
        RETURN p.venue AS venue,
               sum(CASE WHEN p.field = 'ML' THEN 1 ELSE 0 END) AS ml_count,
               sum(CASE WHEN p.field <> 'ML' THEN 1 ELSE 0 END) AS other_count,
               count(*) AS total
        ORDER BY total DESC LIMIT 10
    """)
    )


# 15. count(DISTINCT) — distinct fields per institution


def sqlite_count_distinct(db):
    return db.execute("""
        SELECT i.id, i.name, COUNT(DISTINCT p.field) AS n_fields
        FROM institution i
        JOIN affiliated a ON i.id = a.inst_id
        JOIN person p ON a.person_id = p.id
        GROUP BY i.id, i.name
        ORDER BY n_fields DESC LIMIT 20
    """).fetchall()


def duckdb_count_distinct(db):
    return db.execute("""
        SELECT i.id, i.name, COUNT(DISTINCT p.field) AS n_fields
        FROM institution i
        JOIN affiliated a ON i.id = a.inst_id
        JOIN person p ON a.person_id = p.id
        GROUP BY i.id, i.name
        ORDER BY n_fields DESC LIMIT 20
    """).fetchall()


def kglite_count_distinct(g):
    return len(
        g.cypher("""
        MATCH (i:Institution)<-[:AFFILIATED]-(p:Person)
        RETURN i.id, i.name, count(DISTINCT p.field) AS n_fields
        ORDER BY n_fields DESC LIMIT 20
    """)
    )


# 16. Multi-step WITH — top authors' venue distribution


def sqlite_multistep_with(db):
    return db.execute("""
        WITH top_authors AS (
            SELECT id FROM person ORDER BY h_index DESC LIMIT 100
        ),
        their_papers AS (
            SELECT p.venue, p.citation_count
            FROM top_authors ta
            JOIN authored a ON ta.id = a.person_id
            JOIN paper p ON a.paper_id = p.id
        )
        SELECT venue, COUNT(*) AS cnt, AVG(citation_count) AS avg_cite
        FROM their_papers
        GROUP BY venue ORDER BY cnt DESC LIMIT 10
    """).fetchall()


def duckdb_multistep_with(db):
    return db.execute("""
        WITH top_authors AS (
            SELECT id FROM person ORDER BY h_index DESC LIMIT 100
        ),
        their_papers AS (
            SELECT p.venue, p.citation_count
            FROM top_authors ta
            JOIN authored a ON ta.id = a.person_id
            JOIN paper p ON a.paper_id = p.id
        )
        SELECT venue, COUNT(*) AS cnt, AVG(citation_count) AS avg_cite
        FROM their_papers
        GROUP BY venue ORDER BY cnt DESC LIMIT 10
    """).fetchall()


def kglite_multistep_with(g):
    return len(
        g.cypher("""
        MATCH (a:Person)
        WITH a ORDER BY a.h_index DESC LIMIT 100
        MATCH (a)-[:AUTHORED]->(p:Paper)
        WITH p.venue AS venue, p.citation_count AS cites
        RETURN venue, count(*) AS cnt, avg(cites) AS avg_cite
        ORDER BY cnt DESC LIMIT 10
    """)
    )


# 17. UNION — top by citations UNION top by recency


def sqlite_union(db):
    return db.execute("""
        SELECT * FROM (
            SELECT id, 'most_cited' AS reason FROM paper
            ORDER BY citation_count DESC LIMIT 10
        )
        UNION
        SELECT * FROM (
            SELECT id, 'most_recent' AS reason FROM paper
            ORDER BY year DESC, id DESC LIMIT 10
        )
    """).fetchall()


def duckdb_union(db):
    return db.execute("""
        (SELECT id, 'most_cited' AS reason FROM paper
         ORDER BY citation_count DESC LIMIT 10)
        UNION
        (SELECT id, 'most_recent' AS reason FROM paper
         ORDER BY year DESC, id DESC LIMIT 10)
    """).fetchall()


def kglite_union(g):
    return len(
        g.cypher("""
        MATCH (p:Paper)
        RETURN p.id AS id, 'most_cited' AS reason
        ORDER BY p.citation_count DESC LIMIT 10
        UNION
        MATCH (p:Paper)
        RETURN p.id AS id, 'most_recent' AS reason
        ORDER BY p.year DESC, p.id DESC LIMIT 10
    """)
    )


# 18. Correlated top-N — best author per field (window + filter)


def sqlite_top_per_group(db):
    return db.execute("""
        SELECT field, id, h_index FROM (
            SELECT field, id, h_index,
                   ROW_NUMBER() OVER (PARTITION BY field ORDER BY h_index DESC) AS rn
            FROM person
        ) WHERE rn = 1
    """).fetchall()


def duckdb_top_per_group(db):
    return db.execute("""
        SELECT field, id, h_index FROM (
            SELECT field, id, h_index,
                   ROW_NUMBER() OVER (PARTITION BY field ORDER BY h_index DESC) AS rn
            FROM person
        ) WHERE rn = 1
    """).fetchall()


def kglite_top_per_group(g):
    return len(
        g.cypher("""
        MATCH (p:Person)
        WITH p.field AS field, p.id AS id, p.h_index AS h,
             row_number() OVER (PARTITION BY p.field ORDER BY p.h_index DESC) AS rn
        WHERE rn = 1
        RETURN field, id, h
    """)
    )


# 19. Out-degree distribution — citation fan-out histogram


def sqlite_outdegree_dist(db):
    return db.execute("""
        SELECT out_deg, COUNT(*) AS n_papers FROM (
            SELECT paper_id, COUNT(*) AS out_deg FROM cites GROUP BY paper_id
        ) GROUP BY out_deg ORDER BY out_deg
    """).fetchall()


def duckdb_outdegree_dist(db):
    return db.execute("""
        SELECT out_deg, COUNT(*) AS n_papers FROM (
            SELECT paper_id, COUNT(*) AS out_deg FROM cites GROUP BY paper_id
        ) GROUP BY out_deg ORDER BY out_deg
    """).fetchall()


def kglite_outdegree_dist(g):
    return len(
        g.cypher("""
        MATCH (p:Paper)-[:CITES]->(cited:Paper)
        WITH p, count(cited) AS out_deg
        RETURN out_deg, count(*) AS n_papers
        ORDER BY out_deg
    """)
    )


# 20. Multiple aggregates — comprehensive venue stats


def sqlite_multi_agg(db):
    return db.execute("""
        SELECT venue,
               COUNT(*) AS cnt,
               AVG(citation_count) AS avg_cite,
               SUM(citation_count) AS total_cite,
               MIN(citation_count) AS min_cite,
               MAX(citation_count) AS max_cite
        FROM paper GROUP BY venue ORDER BY avg_cite DESC
    """).fetchall()


def duckdb_multi_agg(db):
    return db.execute("""
        SELECT venue,
               COUNT(*) AS cnt,
               AVG(citation_count) AS avg_cite,
               SUM(citation_count) AS total_cite,
               MIN(citation_count) AS min_cite,
               MAX(citation_count) AS max_cite
        FROM paper GROUP BY venue ORDER BY avg_cite DESC
    """).fetchall()


def kglite_multi_agg(g):
    return len(
        g.cypher("""
        MATCH (p:Paper)
        RETURN p.venue AS venue,
               count(*) AS cnt,
               avg(p.citation_count) AS avg_cite,
               sum(p.citation_count) AS total_cite,
               min(p.citation_count) AS min_cite,
               max(p.citation_count) AS max_cite
        ORDER BY avg_cite DESC
    """)
    )


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    import pandas as pd

    print("Generating dataset...")
    data = generate_data()
    edge_keys = ["authored", "cites", "covers", "affiliated", "collaborates"]
    edge_count = sum(len(data[k]) for k in edge_keys)
    node_count = N_PERSONS + N_PAPERS + N_TOPICS + N_INSTITUTIONS
    print(f"  {node_count:,} nodes, {edge_count:,} edges\n")

    print("Loading into SQLite...", end=" ", flush=True)
    t0 = time.perf_counter()
    db = setup_sqlite(data)
    print(f"{(time.perf_counter() - t0) * 1000:.0f} ms")

    print("Loading into DuckDB...", end=" ", flush=True)
    t0 = time.perf_counter()
    ddb = setup_duckdb(data)
    print(f"{(time.perf_counter() - t0) * 1000:.0f} ms")

    print("Loading into KGLite...", end=" ", flush=True)
    t0 = time.perf_counter()
    g = setup_kglite(data)
    print(f"{(time.perf_counter() - t0) * 1000:.0f} ms\n")

    rows = []

    def run_triple(label, sqlite_fn, duckdb_fn, kglite_fn, kglite_raw_fn=None, duckdb_opt_fn=None, category=""):
        sql_ms, sql_n = bench(sqlite_fn)
        duck_ms, duck_n = bench(duckdb_fn)
        if duckdb_opt_fn:
            duck_opt_ms, _ = bench(duckdb_opt_fn)
            duck_ms = min(duck_ms, duck_opt_ms)
        kg_ms, kg_n = bench(kglite_fn)
        if kglite_raw_fn:
            kg_raw_ms, _ = bench(kglite_raw_fn)
        else:
            kg_raw_ms = kg_ms
        best_sql = min(sql_ms, duck_ms)
        if kg_raw_ms <= best_sql:
            winner = "KGLite"
        elif duck_ms <= sql_ms:
            winner = "DuckDB"
        else:
            winner = "SQLite"
        # Ratio: best SQL engine vs KGLite
        ratio = best_sql / kg_raw_ms if kg_raw_ms > 0 else float("inf")
        rows.append(
            {
                "Category": category,
                "Query": label,
                "SQLite (ms)": round(sql_ms, 4),
                "DuckDB (ms)": round(duck_ms, 4),
                "KGLite (ms)": round(kg_raw_ms, 4),
                "Winner": winner,
                "Ratio": f"{ratio:.1f}x",
                "Rows": max(sql_n, duck_n, kg_n),
            }
        )

    # ── Citation chain (shallow → crossover → deep) ──
    for depth in [5, 8, 15, 20]:
        run_triple(
            f"{depth}-hop citations",
            lambda d=depth: sqlite_citation_hops(db, d),
            lambda d=depth: duckdb_citation_hops(ddb, d),
            lambda d=depth: kglite_citation_hops(g, d),
            lambda d=depth: kglite_citation_hops_raw(g, d),
            category="Citation chain",
        )

    # ── Collaboration chain ──
    for depth in [6, 10]:
        run_triple(
            f"{depth}-hop collaborators",
            lambda d=depth: sqlite_collab_hops(db, d),
            lambda d=depth: duckdb_collab_hops(ddb, d),
            lambda d=depth: kglite_collab_hops(g, d),
            lambda d=depth: kglite_collab_hops_raw(g, d),
            category="Collab chain",
        )

    # ── Heterogeneous traversal ──
    run_triple(
        "Person->Paper->Topic->Paper->Person",
        lambda: sqlite_hetero_traversal(db),
        lambda: duckdb_hetero_traversal(ddb),
        lambda: kglite_hetero_traversal(g),
        lambda: kglite_hetero_traversal_raw(g),
        category="Heterogeneous",
    )

    # ── Shortest path ──
    rng = random.Random(SEED)
    for _ in range(2):
        src = rng.randint(0, N_PERSONS - 1)
        tgt = rng.randint(0, N_PERSONS - 1)
        run_triple(
            f"Person {src} -> {tgt}",
            lambda s=src, t=tgt: sqlite_shortest_path(db, s, t),
            lambda s=src, t=tgt: duckdb_shortest_path(ddb, s, t),
            lambda s=src, t=tgt: kglite_shortest_path(g, s, t),
            category="Shortest path",
        )

    # ── Triangle detection ──
    run_triple(
        f"A->B->C->A in {len(TRIANGLE_PAPERS)} papers",
        lambda: sqlite_triangles(db),
        lambda: duckdb_triangles(ddb),
        lambda: kglite_triangles_raw(g),
        category="Triangles",
    )

    # ── Reachability (early termination) ──
    rng2 = random.Random(SEED + 1)
    for max_hops in [8, 12]:
        src = rng2.randint(0, N_PERSONS - 1)
        tgt = rng2.randint(0, N_PERSONS - 1)
        run_triple(
            f"Person {src} <-> {tgt} ({max_hops} hops)",
            lambda s=src, t=tgt, h=max_hops: sqlite_reachable(db, s, t, h),
            lambda s=src, t=tgt, h=max_hops: duckdb_reachable(ddb, s, t, h),
            lambda s=src, t=tgt, h=max_hops: kglite_reachable_raw(g, s, t, h),
            category="Reachability",
        )

    # ── Neighborhood aggregation (mixed types) ──
    run_triple(
        "Topics via 2-hop collabs",
        lambda: sqlite_neighbor_topics(db, 2),
        lambda: duckdb_neighbor_topics(ddb, 2),
        lambda: kglite_neighbor_topics_raw(g, 2),
        category="Neighborhood agg.",
    )

    # ── Fan-out (batch BFS) ──
    fan_out_50 = list(range(50))
    run_triple(
        "50 x 3-hop reach",
        lambda: sqlite_fan_out(db, fan_out_50, 3),
        lambda: duckdb_fan_out(ddb, fan_out_50, 3),
        lambda: kglite_fan_out_raw(g, fan_out_50, 3),
        duckdb_opt_fn=lambda: duckdb_fan_out_batch(ddb, fan_out_50, 3),
        category="Fan-out",
    )

    # ── Multi-start deep traversal ──
    deep_starts = list(range(0, 100, 10))  # 10 start nodes
    run_triple(
        "10 x 12-hop citations",
        lambda: sqlite_multi_start_deep(db, deep_starts, 12),
        lambda: duckdb_multi_start_deep(ddb, deep_starts, 12),
        lambda: kglite_multi_start_deep_raw(g, deep_starts, 12),
        duckdb_opt_fn=lambda: duckdb_multi_start_deep_batch(ddb, deep_starts, 12),
        category="Multi-start deep",
    )

    # ── Analytical benchmarks ──

    run_triple(
        "ML authors (h>80) → papers",
        lambda: sqlite_filtered_traversal(db),
        lambda: duckdb_filtered_traversal(ddb),
        lambda: kglite_filtered_traversal(g),
        category="Prop. filter",
    )

    run_triple(
        "Recent papers 2-hop cites",
        lambda: sqlite_filtered_multihop(db),
        lambda: duckdb_filtered_multihop(ddb),
        lambda: kglite_filtered_multihop(g),
        category="Filtered hops",
    )

    run_triple(
        "Top 20 venue×year counts",
        lambda: sqlite_agg_pipeline(db),
        lambda: duckdb_agg_pipeline(ddb),
        lambda: kglite_agg_pipeline(g),
        category="Aggregation",
    )

    run_triple(
        "Prolific venues (HAVING)",
        lambda: sqlite_having(db),
        lambda: duckdb_having(ddb),
        lambda: kglite_having(g),
        category="HAVING",
    )

    run_triple(
        "Rank authors in field",
        lambda: sqlite_window(db),
        lambda: duckdb_window(ddb),
        lambda: kglite_window(g),
        category="Window fn",
    )

    run_triple(
        "Top-50 authors' best papers",
        lambda: sqlite_with_chain(db),
        lambda: duckdb_with_chain(ddb),
        lambda: kglite_with_chain(g),
        category="WITH chain",
    )

    run_triple(
        "Most-cited (in-degree)",
        lambda: sqlite_indegree(db),
        lambda: duckdb_indegree(ddb),
        lambda: kglite_indegree(g),
        category="In-degree",
    )

    run_triple(
        "Cross-inst co-authorships",
        lambda: sqlite_cross_inst(db),
        lambda: duckdb_cross_inst(ddb),
        lambda: kglite_cross_inst(g),
        category="Cross-inst",
    )

    run_triple(
        "Multi-predicate paper scan",
        lambda: sqlite_complex_filter(db),
        lambda: duckdb_complex_filter(ddb),
        lambda: kglite_complex_filter(g),
        category="Complex filter",
    )

    run_triple(
        "UPPER + ROUND + GROUP BY",
        lambda: sqlite_string_math(db),
        lambda: duckdb_string_math(ddb),
        lambda: kglite_string_math(g),
        category="String+math",
    )

    # ── Advanced analytical benchmarks ──

    run_triple(
        "OPTIONAL MATCH papers/person",
        lambda: sqlite_optional_match(db),
        lambda: duckdb_optional_match(ddb),
        lambda: kglite_optional_match(g),
        category="OPTIONAL MATCH",
    )

    run_triple(
        "EXISTS semi-join",
        lambda: sqlite_exists(db),
        lambda: duckdb_exists(ddb),
        lambda: kglite_exists(g),
        category="EXISTS",
    )

    run_triple(
        "NOT EXISTS anti-join",
        lambda: sqlite_not_exists(db),
        lambda: duckdb_not_exists(ddb),
        lambda: kglite_not_exists(g),
        category="NOT EXISTS",
    )

    run_triple(
        "CASE conditional agg",
        lambda: sqlite_case_agg(db),
        lambda: duckdb_case_agg(ddb),
        lambda: kglite_case_agg(g),
        category="CASE agg",
    )

    run_triple(
        "count(DISTINCT) per group",
        lambda: sqlite_count_distinct(db),
        lambda: duckdb_count_distinct(ddb),
        lambda: kglite_count_distinct(g),
        category="DISTINCT agg",
    )

    run_triple(
        "3-stage WITH pipeline",
        lambda: sqlite_multistep_with(db),
        lambda: duckdb_multistep_with(ddb),
        lambda: kglite_multistep_with(g),
        category="Multi-WITH",
    )

    run_triple(
        "UNION top-cited + top-recent",
        lambda: sqlite_union(db),
        lambda: duckdb_union(ddb),
        lambda: kglite_union(g),
        category="UNION",
    )

    run_triple(
        "Best author per field",
        lambda: sqlite_top_per_group(db),
        lambda: duckdb_top_per_group(ddb),
        lambda: kglite_top_per_group(g),
        category="Top-per-group",
    )

    run_triple(
        "Citation out-degree histogram",
        lambda: sqlite_outdegree_dist(db),
        lambda: duckdb_outdegree_dist(ddb),
        lambda: kglite_outdegree_dist(g),
        category="Degree dist.",
    )

    run_triple(
        "Venue stats (5 aggregates)",
        lambda: sqlite_multi_agg(db),
        lambda: duckdb_multi_agg(ddb),
        lambda: kglite_multi_agg(g),
        category="Multi-agg",
    )

    # Build DataFrame, sort by SQLite time, print as markdown
    df = pd.DataFrame(rows)
    df = df.sort_values("SQLite (ms)").reset_index(drop=True)
    print(df.to_markdown(index=False, floatfmt=".4f"))

    # Summary
    kg_wins = (df["Winner"] == "KGLite").sum()
    duck_wins = (df["Winner"] == "DuckDB").sum()
    sql_wins = (df["Winner"] == "SQLite").sum()
    total = len(df)
    print(f"\nKGLite wins: {kg_wins}/{total}  |  DuckDB wins: {duck_wins}/{total}  |  SQLite wins: {sql_wins}/{total}")
    print(f"Each query: median of {RUNS} runs. Dataset: {node_count:,} nodes, {edge_count:,} edges.")


if __name__ == "__main__":
    main()
