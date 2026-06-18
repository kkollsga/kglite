"""python-igraph adapter — C-backed graph-algorithm library.

Like rustworkx, igraph is index-based and algorithm-focused. It does
carry vertex attributes, so property filter/aggregation use the native
`vs.select()` surface. The cyclic multi-edge pattern_match is skipped.

Subgraphs match the other backends' vertex sets exactly: a person-only
undirected KNOWS graph (vertex i = person gid - p0, with age/city
attributes) and a project-only directed DEPENDS_ON graph.
"""

from __future__ import annotations

import igraph as ig

from .base import Adapter, Skip
from .dataset import DEGREE_MIN, SCORE_MIN, SCORE_RANGE, Dataset


class IgraphAdapter(Adapter):
    name = "igraph"

    def version(self) -> str:
        return ig.__version__

    def build(self, ds: Dataset) -> None:
        self.p0, self.p1 = ds.ranges["Person"]
        self.pr0, self.pr1 = ds.ranges["Project"]
        np = self.p1 - self.p0
        npr = self.pr1 - self.pr0

        persons = ds.nodes["Person"]
        self.K = ig.Graph(
            n=np,
            edges=[(r["src"] - self.p0, r["dst"] - self.p0) for r in ds.edges["KNOWS"]],
            directed=False,
        )
        self.K.vs["age"] = [r["age"] for r in persons]
        self.K.vs["city"] = [r["city"] for r in persons]

        self.D = ig.Graph(
            n=npr,
            edges=[(r["src"] - self.pr0, r["dst"] - self.pr0) for r in ds.edges["DEPENDS_ON"]],
            directed=True,
        )
        self.persons = persons
        self.person_by_gid = {r["gid"]: r for r in persons}
        self._mut = 0

    def g_node_scan(self, ds):
        return self.K.vcount()

    def g_point_lookup(self, ds):
        pbg = self.person_by_gid
        return sum(1 for gid in ds.params["lookup_ids"] if gid in pbg)

    def g_property_filter(self, ds):
        vs = self.K.vs.select(age_gt=ds.params["filter_age"], city_eq=ds.params["filter_city"])
        return frozenset(v.index + self.p0 for v in vs)

    def g_group_aggregation(self, ds):
        acc: dict[str, list[int]] = {}
        for city, age in zip(self.K.vs["city"], self.K.vs["age"]):
            slot = acc.setdefault(city, [0, 0])
            slot[0] += 1
            slot[1] += age
        return {c: (n, s / n) for c, (n, s) in acc.items()}

    def g_edge_scan(self, ds):
        return self.K.ecount()

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

    # local vertex indices mapped back to global ids (+ p0) for parity.
    def _khop(self, seeds, k):
        locs = [s - self.p0 for s in seeds]
        nbhd = self.K.neighborhood(vertices=locs, order=k, mindist=1)
        out: set[int] = set()
        for lst in nbhd:
            out.update(lst)
        return frozenset(loc + self.p0 for loc in out)

    def g_one_hop(self, ds):
        return self._khop(ds.params["seed_persons"], 1)

    def g_two_hop(self, ds):
        return self._khop(ds.params["seed_persons_small"], 2)

    def g_three_hop(self, ds):
        return self._khop(ds.params["seed_persons_tiny"], 3)

    def g_filtered_traversal(self, ds):
        K = self.K
        ages = K.vs["age"]
        out: set[int] = set()
        for s in ds.params["seed_persons"]:
            for f in K.neighbors(s - self.p0):
                if ages[f] < 30:
                    out.add(f)
        return frozenset(f + self.p0 for f in out)

    def g_deep_traversal(self, ds):
        D = self.D
        out: set[int] = set()
        for s in ds.params["seed_projects"]:
            comp = set(D.subcomponent(s - self.pr0, mode="out"))
            comp.discard(s - self.pr0)  # exclude the source itself, not other seeds
            out |= comp
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
        srcs = [a - self.p0 for a, _ in ds.params["sp_pairs"]]
        tgts = [b - self.p0 for _, b in ds.params["sp_pairs"]]
        lengths = []
        for a, b in zip(srcs, tgts):
            d = K.distances(source=[a], target=[b])[0][0]
            lengths.append(int(d) if d != float("inf") else None)
        return tuple(lengths)

    def g_degree_topk(self, ds):
        degs = sorted(self.K.degree(range(self.K.vcount())), reverse=True)
        return tuple(degs[: ds.params["topk"]])

    def g_connected_components(self, ds):
        cc = self.K.connected_components()
        sizes = cc.sizes()
        return (len(sizes), max(sizes) if sizes else 0)

    def g_louvain(self, ds):
        cl = self.K.community_multilevel()
        sizes = cl.sizes()
        return (len(sizes), max(sizes) if sizes else 0)

    def g_degree_filter(self, ds):
        return sum(1 for d in self.K.degree() if d >= DEGREE_MIN)

    def g_bulk_update(self, ds):
        pbg = self.person_by_gid
        c = 0
        for gid in ds.params["lookup_ids"]:
            if gid in pbg:
                pbg[gid]["active"] = True
                c += 1
        return c

    def g_pattern_match(self, ds):
        raise Skip("igraph has no relational pattern-match surface")

    def g_mutations(self, ds):
        n = ds.params["mut_new_count"]
        K = self.K
        base = K.vcount()
        K.add_vertices(n)
        K.add_edges([(base + i, base + i - 1) for i in range(1, n)])
        K.delete_vertices([base + i for i in range(0, n, 3)])
        self._mut += 1
        return n
