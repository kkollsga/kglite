"""rustworkx adapter — Rust-backed graph-algorithm library.

rustworkx is index-based and excels at graph algorithms (BFS, shortest
path, components, degree) but has no property/relational query surface,
so node-property groups iterate node payloads in Python and the cyclic
multi-edge `pattern_match` is skipped (not a graph-algo operation).

Two remapped subgraphs are built so vertex sets match the other
backends exactly: a person-only undirected KNOWS graph (vertex i =
person gid - p0) and a project-only directed DEPENDS_ON graph.
"""

from __future__ import annotations

import rustworkx as rx

from .base import Adapter, Skip
from .dataset import DEGREE_MIN, SCORE_MIN, SCORE_RANGE, Dataset


class RustworkxAdapter(Adapter):
    name = "rustworkx"

    def version(self) -> str:
        return rx.__version__

    def build(self, ds: Dataset) -> None:
        self.p0, self.p1 = ds.ranges["Person"]
        self.pr0, self.pr1 = ds.ranges["Project"]
        np = self.p1 - self.p0
        npr = self.pr1 - self.pr0

        K = rx.PyGraph()
        K.add_nodes_from(list(range(np)))
        K.add_edges_from_no_data([(r["src"] - self.p0, r["dst"] - self.p0) for r in ds.edges["KNOWS"]])
        self.K = K

        D = rx.PyDiGraph()
        D.add_nodes_from(list(range(npr)))
        D.add_edges_from_no_data([(r["src"] - self.pr0, r["dst"] - self.pr0) for r in ds.edges["DEPENDS_ON"]])
        self.D = D

        # payloads for property groups (index-based libs hold no property index)
        self.persons = ds.nodes["Person"]
        self.person_by_gid = {r["gid"]: r for r in self.persons}
        self._mut = 0

    # -- property / scan groups (Python iteration over payloads) -----------
    def g_node_scan(self, ds):
        return sum(1 for _ in self.persons)

    def g_point_lookup(self, ds):
        pbg = self.person_by_gid
        return sum(1 for gid in ds.params["lookup_ids"] if gid in pbg)

    def g_property_filter(self, ds):
        age, city = ds.params["filter_age"], ds.params["filter_city"]
        return frozenset(r["gid"] for r in self.persons if r["age"] > age and r["city"] == city)

    def g_group_aggregation(self, ds):
        acc: dict[str, list[int]] = {}
        for r in self.persons:
            slot = acc.setdefault(r["city"], [0, 0])
            slot[0] += 1
            slot[1] += r["age"]
        return {c: (n, s / n) for c, (n, s) in acc.items()}

    def g_edge_scan(self, ds):
        return self.K.num_edges()

    def g_range_filter(self, ds):
        lo, hi = SCORE_RANGE
        return frozenset(r["gid"] for r in self.persons if lo <= r["score"] <= hi)

    def g_year_aggregation(self, ds):
        acc: dict[int, list[float]] = {}
        for r in self.persons:
            slot = acc.setdefault(r["joined_year"], [0, 0.0])
            slot[0] += 1
            slot[1] += r["score"]
        return {y: (n, s / n) for y, (n, s) in acc.items()}

    # -- graph-algo groups (rustworkx strengths) ---------------------------
    # local vertex indices are mapped back to global ids (+ p0) so the result
    # sets are directly comparable with the property-graph backends.
    def _khop(self, seeds, k):
        K = self.K
        out: set[int] = set()
        for s in seeds:
            loc = s - self.p0
            frontier = {loc}
            visited = {loc}
            for _ in range(k):
                nxt: set[int] = set()
                for u in frontier:
                    nxt.update(K.neighbors(u))
                nxt -= visited
                if not nxt:
                    break
                visited |= nxt
                frontier = nxt
            visited.discard(loc)
            out |= visited
        return frozenset(loc + self.p0 for loc in out)

    def g_one_hop(self, ds):
        return self._khop(ds.params["seed_persons"], 1)

    def g_two_hop(self, ds):
        return self._khop(ds.params["seed_persons_small"], 2)

    def g_three_hop(self, ds):
        return self._khop(ds.params["seed_persons_tiny"], 3)

    def g_filtered_traversal(self, ds):
        K = self.K
        out: set[int] = set()
        for s in ds.params["seed_persons"]:
            for f in K.neighbors(s - self.p0):
                if self.persons[f]["age"] < 30:
                    out.add(f)
        return frozenset(f + self.p0 for f in out)

    def g_deep_traversal(self, ds):
        D = self.D
        out: set[int] = set()
        for s in ds.params["seed_projects"]:
            out |= rx.descendants(D, s - self.pr0)
        return frozenset(d + self.pr0 for d in out)

    def g_score_filtered_traversal(self, ds):
        K, persons = self.K, self.persons
        out: set[int] = set()
        for s in ds.params["seed_persons"]:
            for f in K.neighbors(s - self.p0):
                if persons[f]["score"] > SCORE_MIN:
                    out.add(f)
        return frozenset(f + self.p0 for f in out)

    def g_shortest_path(self, ds):
        K = self.K
        lengths = []
        for a, b in ds.params["sp_pairs"]:
            la, lb = a - self.p0, b - self.p0
            if la == lb:
                lengths.append(0)
                continue
            d = rx.dijkstra_shortest_path_lengths(K, la, lambda _e: 1.0, goal=lb)
            lengths.append(int(d[lb]) if lb in d else None)
        return tuple(lengths)

    def g_degree_topk(self, ds):
        K = self.K
        degs = sorted((K.degree(i) for i in range(self.p1 - self.p0)), reverse=True)
        return tuple(degs[: ds.params["topk"]])

    def g_connected_components(self, ds):
        comps = rx.connected_components(self.K)
        return (len(comps), max((len(c) for c in comps), default=0))

    def g_degree_filter(self, ds):
        K = self.K
        return sum(1 for i in range(self.p1 - self.p0) if K.degree(i) >= DEGREE_MIN)

    def g_bulk_update(self, ds):
        pbg = self.person_by_gid
        c = 0
        for gid in ds.params["lookup_ids"]:
            if gid in pbg:
                pbg[gid]["active"] = True
                c += 1
        return c

    def g_pattern_match(self, ds):
        raise Skip("rustworkx has no relational pattern-match surface")

    def g_mutations(self, ds):
        off = (self.p1 - self.p0) + self._mut * 100_000
        self._mut += 1
        n = ds.params["mut_new_count"]
        K = self.K
        new_idx = K.add_nodes_from(list(range(off, off + n)))
        K.add_edges_from_no_data([(new_idx[i], new_idx[i - 1]) for i in range(1, n)])
        for i in range(0, n, 3):
            K.remove_node(new_idx[i])
        return n
