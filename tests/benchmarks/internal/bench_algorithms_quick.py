"""Quick benchmark for graph algorithm performance (Steps 2,3,4 optimizations)."""

import random
import time

import pandas as pd

import kglite

SEED = 42
N_PERSONS = 10_000
N_PAPERS = 20_000
N_TOPICS = 500
N_INSTITUTIONS = 200


def generate_data():
    rng = random.Random(SEED)
    authored, cites, covers, affiliated, collaborates = [], [], [], [], []

    for pid in range(N_PERSONS):
        for _ in range(rng.randint(2, 10)):
            authored.append((pid, rng.randint(0, N_PAPERS - 1)))

    for paper_id in range(N_PAPERS):
        for _ in range(rng.randint(1, 8)):
            cited = rng.randint(0, N_PAPERS - 1)
            if cited != paper_id:
                cites.append((paper_id, cited))

    for paper_id in range(N_PAPERS):
        for _ in range(rng.randint(1, 4)):
            covers.append((paper_id, rng.randint(0, N_TOPICS - 1)))

    for pid in range(N_PERSONS):
        affiliated.append((pid, rng.randint(0, N_INSTITUTIONS - 1)))

    for pid in range(N_PERSONS):
        for _ in range(rng.randint(1, 5)):
            other = rng.randint(0, N_PERSONS - 1)
            if other != pid:
                collaborates.append((pid, other))

    # Deterministic structure
    for i in range(0, 300, 3):
        cites.extend([(i, i + 1), (i + 1, i + 2), (i + 2, i)])
        collaborates.extend([(i, i + 1), (i + 1, i + 2), (i + 2, i)])
    for pid in range(100):
        cites.extend([(pid, 5000), (pid, 5001)])

    return {
        "authored": authored,
        "cites": cites,
        "covers": covers,
        "affiliated": affiliated,
        "collaborates": collaborates,
    }


def setup_graph(data):
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": range(N_PERSONS)}), "Person", "id", "id")
    g.add_nodes(pd.DataFrame({"id": range(N_PAPERS)}), "Paper", "id", "id")
    g.add_nodes(pd.DataFrame({"id": range(N_TOPICS)}), "Topic", "id", "id")
    g.add_nodes(pd.DataFrame({"id": range(N_INSTITUTIONS)}), "Institution", "id", "id")

    g.add_connections(
        pd.DataFrame(data["authored"], columns=["person_id", "paper_id"]),
        "AUTHORED",
        "Person",
        "person_id",
        "Paper",
        "paper_id",
    )
    g.add_connections(
        pd.DataFrame(data["cites"], columns=["paper_id", "cited_id"]), "CITES", "Paper", "paper_id", "Paper", "cited_id"
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
        pd.DataFrame(data["collaborates"], columns=["pid1", "pid2"]), "COLLABORATES", "Person", "pid1", "Person", "pid2"
    )
    return g


def bench(label, fn, runs=3):
    # Warmup
    fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"  {label:40s} {best * 1000:10.1f} ms")
    return best


print("Building graph...")
data = generate_data()
g = setup_graph(data)
counts = g.node_type_counts()
total_nodes = sum(counts.values())
edge_stats = g.statistics("edges")
total_edges = sum(v for v in edge_stats.values() if isinstance(v, (int, float)))
print(f"Graph: {total_nodes} nodes, {total_edges} edges\n")

print("Algorithm benchmarks (best of 3 runs):")
print("=" * 55)

bench("degree_centrality", lambda: g.cypher("CALL degree_centrality() YIELD node, score"))

bench("pagerank", lambda: g.cypher("CALL pagerank() YIELD node, score"))

bench("label_propagation", lambda: g.cypher("CALL label_propagation() YIELD node, community"))

bench("louvain", lambda: g.cypher("CALL louvain() YIELD node, community"))

bench(
    "betweenness(sample=1000)", lambda: g.cypher("CALL betweenness_centrality({sample_size: 1000}) YIELD node, score")
)

bench("closeness(full)", lambda: g.cypher("CALL closeness_centrality() YIELD node, score"))

bench("betweenness(full)", lambda: g.cypher("CALL betweenness_centrality() YIELD node, score"))

print("\nDone.")
