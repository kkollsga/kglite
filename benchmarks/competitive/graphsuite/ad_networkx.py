"""NetworkX adapter — the pure-Python graph baseline.

Builds three views: a directed `G` carrying every typed edge (for the
pattern-match join), an undirected `K` over KNOWS (for the social-graph
groups), and a directed `D` over DEPENDS_ON (for deep traversal).
"""

from __future__ import annotations

import networkx as nx

from .base import Adapter
from .dataset import Dataset


class NetworkXAdapter(Adapter):
    name = "networkx"

    def version(self) -> str:
        return nx.__version__

    def build(self, ds: Dataset) -> None:
        G = nx.DiGraph()
        for ntype, rows in ds.nodes.items():
            for r in rows:
                G.add_node(r["gid"], **{k: v for k, v in r.items() if k != "gid"}, ntype=ntype)
        for etype, rows in ds.edges.items():
            for r in rows:
                G.add_edge(r["src"], r["dst"], etype=etype)
        self.G = G
        self.K = nx.Graph()
        self.K.add_edges_from((r["src"], r["dst"]) for r in ds.edges["KNOWS"])
        self.D = nx.DiGraph()
        self.D.add_edges_from((r["src"], r["dst"]) for r in ds.edges["DEPENDS_ON"])
        self._mut = 0

    def g_node_scan(self, ds):
        return sum(1 for _, t in self.G.nodes(data="ntype") if t == "Person")

    def g_point_lookup(self, ds):
        G = self.G
        return sum(1 for gid in ds.params["lookup_ids"] if gid in G)

    def g_property_filter(self, ds):
        age, city = ds.params["filter_age"], ds.params["filter_city"]
        return frozenset(
            n for n, d in self.G.nodes(data=True) if d.get("ntype") == "Person" and d["age"] > age and d["city"] == city
        )

    def g_group_aggregation(self, ds):
        acc: dict[str, list[int]] = {}
        for _, d in self.G.nodes(data=True):
            if d.get("ntype") == "Person":
                c = d["city"]
                slot = acc.setdefault(c, [0, 0])
                slot[0] += 1
                slot[1] += d["age"]
        return {c: (n, s / n) for c, (n, s) in acc.items()}

    def g_one_hop(self, ds):
        K = self.K
        out = set()
        for s in ds.params["seed_persons"]:
            if s in K:
                out.update(K.neighbors(s))
        return frozenset(out)

    def _khop(self, seeds, cutoff):
        K = self.K
        out = set()
        for s in seeds:
            if s in K:
                d = nx.single_source_shortest_path_length(K, s, cutoff=cutoff)
                out.update(n for n, dist in d.items() if 1 <= dist <= cutoff)
        return frozenset(out)

    def g_two_hop(self, ds):
        return self._khop(ds.params["seed_persons_small"], 2)

    def g_three_hop(self, ds):
        return self._khop(ds.params["seed_persons_tiny"], 3)

    def g_filtered_traversal(self, ds):
        K, G = self.K, self.G
        out = set()
        for s in ds.params["seed_persons"]:
            if s in K:
                for f in K.neighbors(s):
                    if G.nodes[f]["age"] < 30:
                        out.add(f)
        return frozenset(out)

    def g_deep_traversal(self, ds):
        D = self.D
        out = set()
        for s in ds.params["seed_projects"]:
            if s in D:
                out.update(nx.descendants(D, s))
        return frozenset(out)

    def g_shortest_path(self, ds):
        K = self.K
        lengths = []
        for a, b in ds.params["sp_pairs"]:
            if a in K and b in K:
                try:
                    lengths.append(nx.shortest_path_length(K, a, b))
                except nx.NetworkXNoPath:
                    lengths.append(None)
            else:
                lengths.append(None)
        return tuple(lengths)

    def g_pattern_match(self, ds):
        G = self.G
        count = 0
        p0, p1 = ds.params["person_range"]
        for p in range(p0, p1):
            for c in G.successors(p):
                if G[p][c]["etype"] != "WORKS_AT":
                    continue
                for pr in G.successors(c):
                    if G[c][pr]["etype"] != "OWNS":
                        continue
                    if G.has_edge(p, pr) and G[p][pr]["etype"] == "CONTRIBUTES_TO":
                        count += 1
        return count

    def g_degree_topk(self, ds):
        deg = sorted((d for _, d in self.K.degree()), reverse=True)
        return tuple(deg[: ds.params["topk"]])

    def g_connected_components(self, ds):
        comps = list(nx.connected_components(self.K))
        return (len(comps), max((len(c) for c in comps), default=0))

    def g_mutations(self, ds):
        off = ds.params["mut_new_base"] + self._mut * 100_000
        self._mut += 1
        n = ds.params["mut_new_count"]
        K = self.K
        for i in range(n):
            K.add_node(off + i, age=30 + (i % 40))
        K.add_edges_from((off + i, off + i - 1) for i in range(1, n))
        for i in range(n):
            K.nodes[off + i]["age"] = 99
        K.remove_nodes_from(off + i for i in range(0, n, 3))
        return n
