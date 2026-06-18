"""Adapter contract + group registry for the graphsuite benchmark.

Each backend subclasses `Adapter` and overrides the group methods it
supports. Unsupported groups raise `Skip` (recorded, not fatal). Every
group method runs the *same* parameterised workload against the backend
and returns a small "sanity" value (a count or a list length) so result
divergence between backends is visible in the report.

A *group* bundles several related operations; the reported number is the
combined wall-time of running the whole group method (min over repeats).
"""

from __future__ import annotations

from .dataset import Dataset


class Skip(Exception):
    """Raised by a group method when the backend cannot express it."""


# Ordered registry. First entry ("build") is special — it maps to
# `adapter.build()`. The rest map to `g_<name>` methods.
#   (group_id, human description, method_name | None)
GROUPS: list[tuple[str, str, str | None]] = [
    ("build", "Graph build — bulk load all nodes + edges", None),
    ("node_scan", "Full node scan — count + collect every Person id", "g_node_scan"),
    ("point_lookup", "Point lookups — fetch 500 Persons by id", "g_point_lookup"),
    ("edge_scan", "Edge scan — count every KNOWS edge", "g_edge_scan"),
    ("property_filter", "Property filter — Persons age>N in one city", "g_property_filter"),
    ("range_filter", "Range filter — Persons with score in [40,60]", "g_range_filter"),
    ("group_aggregation", "Group aggregation — count + avg(age) per city", "g_group_aggregation"),
    ("year_aggregation", "Group aggregation — count + avg(score) per joined_year", "g_year_aggregation"),
    ("one_hop", "1-hop traversal — KNOWS neighbours of 200 seeds", "g_one_hop"),
    ("two_hop", "2-hop traversal — friends-of-friends of 200 seeds", "g_two_hop"),
    ("three_hop", "3-hop traversal — 3-hop neighbourhood of 50 seeds", "g_three_hop"),
    ("filtered_traversal", "Filtered traversal — young KNOWS neighbours", "g_filtered_traversal"),
    ("deep_traversal", "Deep traversal — DEPENDS_ON transitive closure", "g_deep_traversal"),
    ("score_filtered_traversal", "Filtered traversal — high-score KNOWS neighbours", "g_score_filtered_traversal"),
    ("shortest_path", "Shortest path — 100 Person pairs over KNOWS", "g_shortest_path"),
    ("pattern_match", "Pattern match — Person/Company/Project triangle", "g_pattern_match"),
    (
        "industry_aggregation",
        "Heterogeneous aggregation — count + avg(size) per Company industry",
        "g_industry_aggregation",
    ),
    ("two_step_join", "Multi-type join — Person→Company→Project path count", "g_two_step_join"),
    ("degree_topk", "Degree centrality — top-K KNOWS degree", "g_degree_topk"),
    ("connected_components", "Connected components — WCC over KNOWS", "g_connected_components"),
    ("louvain", "Community detection — Louvain modularity over KNOWS", "g_louvain"),
    ("degree_filter", "Degree filter — count Persons with KNOWS-degree ≥ N", "g_degree_filter"),
    # Specialized KG capabilities — kglite's differentiators. kglite-mostly
    # coverage is warranted (vector search / geospatial are KG-engine features
    # the algorithm libraries simply don't have). Read-only, so they run on the
    # pristine graph BEFORE the mutating groups.
    ("vector_knn", "Vector kNN — top-K nearest Persons by embedding", "g_vector_knn"),
    ("geo_within", "Geospatial — Cities within a lat/lon bounding box", "g_geo_within"),
    # bulk_update + mutations run LAST (and on the pristine graph) so their
    # writes don't perturb any read group's measurement.
    ("bulk_update", "Bulk update — set a property on 500 Persons", "g_bulk_update"),
    ("mutations", "Mutations — add/update/delete nodes & edges", "g_mutations"),
]

GROUP_IDS = [g[0] for g in GROUPS]


class Adapter:
    """Base adapter. Subclasses set `name`, implement `build`, and override
    whichever `g_*` group methods they support."""

    #: stable library identifier stored in the results datafile
    name: str = "base"

    def version(self) -> str:
        """Version string of the underlying library (stored per run)."""
        return "unknown"

    def available(self) -> tuple[bool, str]:
        """Return (is_available, reason). Adapters whose backend isn't
        installed / reachable return (False, reason) and are skipped
        wholesale."""
        return True, ""

    # -- build (group 1) ---------------------------------------------------
    def build(self, ds: Dataset) -> None:
        raise NotImplementedError

    def teardown(self) -> None:
        """Release native resources (servers, temp dirs). Best-effort."""

    # -- group methods (override the supported ones) -----------------------
    def g_node_scan(self, ds: Dataset):
        raise Skip("node_scan")

    def g_point_lookup(self, ds: Dataset):
        raise Skip("point_lookup")

    def g_edge_scan(self, ds: Dataset):
        raise Skip("edge_scan")

    def g_property_filter(self, ds: Dataset):
        raise Skip("property_filter")

    def g_range_filter(self, ds: Dataset):
        raise Skip("range_filter")

    def g_group_aggregation(self, ds: Dataset):
        raise Skip("group_aggregation")

    def g_year_aggregation(self, ds: Dataset):
        raise Skip("year_aggregation")

    def g_one_hop(self, ds: Dataset):
        raise Skip("one_hop")

    def g_two_hop(self, ds: Dataset):
        raise Skip("two_hop")

    def g_three_hop(self, ds: Dataset):
        raise Skip("three_hop")

    def g_filtered_traversal(self, ds: Dataset):
        raise Skip("filtered_traversal")

    def g_deep_traversal(self, ds: Dataset):
        raise Skip("deep_traversal")

    def g_score_filtered_traversal(self, ds: Dataset):
        raise Skip("score_filtered_traversal")

    def g_shortest_path(self, ds: Dataset):
        raise Skip("shortest_path")

    def g_pattern_match(self, ds: Dataset):
        raise Skip("pattern_match")

    def g_industry_aggregation(self, ds: Dataset):
        raise Skip("industry_aggregation")

    def g_two_step_join(self, ds: Dataset):
        raise Skip("two_step_join")

    def g_degree_topk(self, ds: Dataset):
        raise Skip("degree_topk")

    def g_connected_components(self, ds: Dataset):
        raise Skip("connected_components")

    def g_louvain(self, ds: Dataset):
        raise Skip("louvain")

    def g_degree_filter(self, ds: Dataset):
        raise Skip("degree_filter")

    def g_mutations(self, ds: Dataset):
        raise Skip("mutations")

    def g_bulk_update(self, ds: Dataset):
        raise Skip("bulk_update")

    def g_vector_knn(self, ds: Dataset):
        raise Skip("vector_knn")

    def g_geo_within(self, ds: Dataset):
        raise Skip("geo_within")
