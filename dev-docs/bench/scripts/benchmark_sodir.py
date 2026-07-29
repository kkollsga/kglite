"""Sodir benchmark: build → save → load → query → report.

Builds the Sodir graph from the blueprint, saves/loads a temp file,
runs Cypher + Fluent API benchmarks, deletes the temp file, and appends
results to benchmark.csv (one column per kglite version).

Requires: kglite, pandas, tqdm
Blueprint: bench/sodir_graph_config.json
CSV root:  /Volumes/EksternalHome/Koding/Python/Faktasider
"""

import contextlib
import csv
import io
import os
from pathlib import Path
import re
import statistics
import sys
import time

import pandas as pd
from tqdm import tqdm

import kglite

SCRIPT_DIR = Path(__file__).parent
BLUEPRINT = str(SCRIPT_DIR / "sodir_graph_config.json")
CSV_ROOT = "/Volumes/EksternalHome/Koding/Python/Faktasider"
TEMP_KGL = str(SCRIPT_DIR / "temp.kgl")
CSV_OUT = str(SCRIPT_DIR / "benchmark.csv")

ITERATIONS = 5
WARMUP = 1

# Benchmarks that are inherently slow (N×N spatial, full-graph algorithms, etc.)
# Run with fewer iterations to keep total time reasonable.
HEAVY_BENCHMARKS = {
    "cypher_spatial_pairs_10km",
    "cypher_spatial_intersects",
    "cypher_algo_betweenness",
    "cypher_algo_closeness",
    "cypher_algo_components",
    "cypher_algo_louvain",
    "cypher_algo_label_prop",
    "cypher_scan_edge_type_counts",
    "cypher_complex_many_small",
    "cypher_window_row_number",
    "cypher_window_rank",
}
HEAVY_ITERATIONS = 2
HEAVY_WARMUP = 0


# ═══════════════════════════════════════════════════════════════════
# Cypher benchmark queries
# ═══════════════════════════════════════════════════════════════════

CYPHER_QUERIES: list[tuple[str, str | None]] = [
    # ── Simple Lookups ─────────────────────────────────────────────
    (
        "cypher_lookup_field_by_title",
        "MATCH (f:Field) WHERE f.title = 'TROLL' RETURN f.title, f.fldStatus, f.fldMainArea",
    ),
    ("cypher_lookup_first_wellbore", "MATCH (w:Wellbore) RETURN w LIMIT 1"),
    ("cypher_lookup_count_fields", "MATCH (f:Field) RETURN count(f)"),
    (
        "cypher_lookup_company_by_title",
        "MATCH (c:Company) WHERE c.title = 'Equinor Energy AS' RETURN c.title, c.cmpOrgNumberBrReg",
    ),
    ("cypher_lookup_node_by_id", "MATCH (f:Field) RETURN id(f), f.title LIMIT 5"),
    # ── WHERE Operators & Predicates ───────────────────────────────
    (
        "cypher_where_and",
        "MATCH (w:Wellbore) WHERE w.wlbMainArea = 'NORTH SEA' AND w.wlbPurpose = 'WILDCAT' RETURN count(w)",
    ),
    (
        "cypher_where_or",
        "MATCH (w:Wellbore) WHERE w.wlbMainArea = 'NORTH SEA' OR w.wlbMainArea = 'NORWEGIAN SEA' RETURN count(w)",
    ),
    ("cypher_where_not", "MATCH (w:Wellbore) WHERE NOT w.wlbPurpose = 'WILDCAT' RETURN count(w)"),
    (
        "cypher_where_range",
        "MATCH (w:Wellbore) WHERE w.wlbEntryYear >= 2020 AND w.wlbEntryYear <= 2025 RETURN w.title, w.wlbEntryYear ORDER BY w.wlbEntryYear DESC LIMIT 20",
    ),
    ("cypher_where_contains", "MATCH (w:Wellbore) WHERE w.wlbContent CONTAINS 'OIL' RETURN count(w)"),
    ("cypher_where_starts_with", "MATCH (w:Wellbore) WHERE w.title STARTS WITH '31/' RETURN w.title LIMIT 20"),
    ("cypher_where_ends_with", "MATCH (w:Wellbore) WHERE w.title ENDS WITH 'S' RETURN count(w)"),
    ("cypher_where_regex", "MATCH (w:Wellbore) WHERE w.title =~ '31/[0-9]+-.*' RETURN count(w)"),
    (
        "cypher_where_in_list",
        "MATCH (f:Field) WHERE f.fldStatus IN ['PRODUCING', 'SHUT DOWN'] RETURN f.title, f.fldStatus",
    ),
    ("cypher_where_is_null", "MATCH (w:Wellbore) WHERE w.wlbContent IS NULL RETURN count(w)"),
    ("cypher_where_is_not_null", "MATCH (w:Wellbore) WHERE w.wlbTotalDepth IS NOT NULL RETURN count(w)"),
    (
        "cypher_where_exists_pattern",
        "MATCH (f:Field) WHERE EXISTS { MATCH (f)-[:INCLUDES_DISCOVERY]->(d:Discovery) } RETURN f.title",
    ),
    # ── Relationship Traversals ────────────────────────────────────
    (
        "cypher_traverse_1hop_out",
        "MATCH (f:Field {title: 'TROLL'})-[:INCLUDES_DISCOVERY]->(d:Discovery) RETURN d.title, d.dscCurrentActivityStatus",
    ),
    ("cypher_traverse_1hop_in", "MATCH (w:Wellbore)-[:IN_FIELD]->(f:Field {title: 'EKOFISK'}) RETURN count(w)"),
    ("cypher_traverse_1hop_undirected", "MATCH (f:Field {title: 'TROLL'})-[:OF_FIELD]-(w:Wellbore) RETURN count(w)"),
    (
        "cypher_traverse_2hop",
        "MATCH (c:Company)<-[:DRILLED_BY]-(w:Wellbore)-[:IN_FIELD]->(f:Field) WHERE c.title = 'Equinor Energy AS' RETURN f.title, count(w) AS wells ORDER BY wells DESC LIMIT 10",
    ),
    (
        "cypher_traverse_3hop",
        "MATCH (p:ProductionProfile)-[:OF_FIELD]->(f:Field)-[:HAS_LICENSEE]->(c:Company) WHERE c.title = 'Equinor Energy AS' RETURN f.title, ts_sum(p.prd_oil_net) AS oil ORDER BY oil DESC",
    ),
    (
        "cypher_traverse_inline_props",
        "MATCH (f:Field {title: 'JOHAN SVERDRUP'})-[:HAS_LICENSEE]->(c:Company) RETURN DISTINCT c.title",
    ),
    # ── Variable-Length Paths & Shortest Path ──────────────────────
    (
        "cypher_varlen_path",
        "MATCH (f:Field {title: 'TROLL'})-[:INCLUDES_DISCOVERY*1..2]->(x) RETURN labels(x)[0] AS type, count(x) AS cnt",
    ),
    (
        "cypher_shortest_path",
        "MATCH p = shortestPath((a:Field {title: 'TROLL'})-[*..6]-(b:Field {title: 'EKOFISK'})) RETURN length(p), [n IN nodes(p) | n.title] AS path",
    ),
    # ── OPTIONAL MATCH ─────────────────────────────────────────────
    (
        "cypher_optional_match",
        "MATCH (f:Field) OPTIONAL MATCH (f)-[:INCLUDES_DISCOVERY]->(d:Discovery) RETURN f.title, count(d) AS discoveries ORDER BY discoveries DESC LIMIT 10",
    ),
    (
        "cypher_optional_match_where",
        "MATCH (f:Field {title: 'TROLL'}) OPTIONAL MATCH (f)-[:INCLUDES_DISCOVERY]->(d:Discovery) WHERE d.dscCurrentActivityStatus = 'PRODUCING' RETURN f.title, count(d) AS discoveries",
    ),
    # ── Aggregations ───────────────────────────────────────────────
    ("cypher_agg_count", "MATCH (w:Wellbore) RETURN count(w) AS total"),
    (
        "cypher_agg_group_count",
        "MATCH (w:Wellbore) RETURN w.wlbMainArea AS area, count(w) AS wells ORDER BY wells DESC",
    ),
    (
        "cypher_agg_group_purpose",
        "MATCH (w:Wellbore) RETURN w.wlbPurpose AS purpose, count(w) AS cnt ORDER BY cnt DESC",
    ),
    (
        "cypher_agg_sum",
        "MATCH (fr:FieldReserves)-[:OF_FIELD]->(f:Field) RETURN f.title, sum(fr.fldRecoverableOil) AS total_oil ORDER BY total_oil DESC LIMIT 10",
    ),
    (
        "cypher_agg_avg_max_min",
        "MATCH (w:Wellbore) WHERE w.wlbTotalDepth IS NOT NULL RETURN avg(w.wlbTotalDepth) AS avg_d, max(w.wlbTotalDepth) AS max_d, min(w.wlbTotalDepth) AS min_d",
    ),
    (
        "cypher_agg_collect",
        "MATCH (f:Field {title: 'TROLL'})-[:INCLUDES_DISCOVERY]->(d:Discovery) RETURN collect(d.title) AS discoveries",
    ),
    (
        "cypher_agg_collect_index",
        "MATCH (f:Field {title: 'TROLL'})-[:INCLUDES_DISCOVERY]->(d:Discovery) WITH f, collect(d.title) AS discs RETURN discs[0] AS first, size(discs) AS total",
    ),
    (
        "cypher_agg_count_distinct",
        "MATCH (c:Company)<-[:DRILLED_BY]-(w:Wellbore)-[:IN_FIELD]->(f:Field) RETURN c.title, count(DISTINCT f) AS fields ORDER BY fields DESC LIMIT 10",
    ),
    (
        "cypher_agg_top_drillers",
        "MATCH (c:Company)<-[:DRILLED_BY]-(w:Wellbore) RETURN c.title AS company, count(w) AS wells ORDER BY wells DESC LIMIT 10",
    ),
    (
        "cypher_agg_yearly",
        "MATCH (w:Wellbore) WHERE w.wlbEntryYear IS NOT NULL RETURN w.wlbEntryYear AS year, count(w) AS wells ORDER BY year DESC LIMIT 20",
    ),
    # ── DISTINCT ───────────────────────────────────────────────────
    ("cypher_distinct", "MATCH (w:Wellbore) RETURN DISTINCT w.wlbPurpose ORDER BY w.wlbPurpose"),
    # ── ORDER BY + SKIP + LIMIT ───────────────────────────────────
    (
        "cypher_order_limit",
        "MATCH (w:Wellbore) WHERE w.wlbTotalDepth IS NOT NULL RETURN w.title, w.wlbTotalDepth ORDER BY w.wlbTotalDepth DESC LIMIT 10",
    ),
    ("cypher_skip_limit", "MATCH (f:Field) RETURN f.title ORDER BY f.title SKIP 10 LIMIT 10"),
    # ── WITH Clause ────────────────────────────────────────────────
    (
        "cypher_with_chain",
        "MATCH (f:Field)<-[:OF_FIELD]-(pp:ProductionProfile) WITH f, ts_sum(pp.prd_oil_net) AS oil WHERE oil > 0 RETURN f.title, oil ORDER BY oil DESC LIMIT 10",
    ),
    (
        "cypher_with_aggregation",
        "MATCH (f:Field)<-[:OF_FIELD]-(fr:FieldReserves) WITH f.title AS field, max(fr.fldRecoverableOil) AS max_oil RETURN field, max_oil ORDER BY max_oil DESC LIMIT 10",
    ),
    # ── HAVING ─────────────────────────────────────────────────────
    (
        "cypher_having",
        "MATCH (w:Wellbore) RETURN w.wlbMainArea AS area, count(w) AS wells HAVING wells > 1000 ORDER BY wells DESC",
    ),
    # ── UNWIND ─────────────────────────────────────────────────────
    (
        "cypher_unwind",
        "UNWIND ['TROLL', 'EKOFISK', 'OSEBERG'] AS name MATCH (f:Field {title: name}) RETURN f.title, f.fldStatus",
    ),
    (
        "cypher_unwind_range",
        "UNWIND range(2015, 2025) AS yr MATCH (w:Wellbore) WHERE w.wlbEntryYear = yr RETURN yr, count(w) AS wells ORDER BY yr",
    ),
    # ── UNION ──────────────────────────────────────────────────────
    (
        "cypher_union_all",
        "MATCH (f:Field) WHERE f.fldStatus = 'PRODUCING' RETURN f.title AS name, 'field' AS type LIMIT 5 UNION ALL MATCH (d:Discovery) RETURN d.title AS name, 'discovery' AS type LIMIT 5",
    ),
    # ── String Functions ───────────────────────────────────────────
    (
        "cypher_fn_toupper_tolower",
        "MATCH (f:Field) RETURN toUpper(f.title) AS upper, toLower(f.title) AS lower LIMIT 5",
    ),
    ("cypher_fn_substring", "MATCH (w:Wellbore) RETURN substring(w.title, 0, 5) AS prefix LIMIT 10"),
    (
        "cypher_fn_split",
        "MATCH (w:Wellbore) RETURN split(w.title, '/')[0] AS quadrant, count(w) AS cnt ORDER BY cnt DESC LIMIT 10",
    ),
    ("cypher_fn_replace", "MATCH (f:Field) RETURN replace(f.title, ' ', '_') AS slug LIMIT 10"),
    ("cypher_fn_trim_reverse", "MATCH (f:Field) RETURN trim(f.title) AS trimmed, reverse(f.title) AS reversed LIMIT 5"),
    ("cypher_fn_left_right", "MATCH (w:Wellbore) RETURN left(w.title, 4) AS l, right(w.title, 2) AS r LIMIT 10"),
    ("cypher_fn_concat", "MATCH (f:Field) RETURN f.title || ' (' || f.fldStatus || ')' AS label LIMIT 10"),
    # ── Math Functions ─────────────────────────────────────────────
    (
        "cypher_fn_abs_round",
        "MATCH (w:Wellbore) WHERE w.wlbTotalDepth IS NOT NULL RETURN w.title, abs(w.wlbTotalDepth - 3000) AS diff, round(w.wlbTotalDepth, 0) AS rounded LIMIT 10",
    ),
    (
        "cypher_fn_sqrt_ceil_floor",
        "MATCH (w:Wellbore) WHERE w.wlbTotalDepth IS NOT NULL RETURN sqrt(w.wlbTotalDepth) AS sq, ceil(w.wlbTotalDepth / 1000.0) AS ceil_km, floor(w.wlbTotalDepth / 1000.0) AS floor_km LIMIT 10",
    ),
    # ── Type Conversion ────────────────────────────────────────────
    (
        "cypher_fn_tostring_tointeger",
        "MATCH (w:Wellbore) WHERE w.wlbEntryYear IS NOT NULL RETURN toString(w.wlbEntryYear) AS yr_str, toInteger('2020') AS parsed LIMIT 5",
    ),
    # ── Introspection Functions ────────────────────────────────────
    (
        "cypher_fn_labels_type",
        "MATCH (f:Field)-[r:INCLUDES_DISCOVERY]->(d:Discovery) RETURN labels(f)[0] AS fl, type(r) AS rt, labels(d)[0] AS dl LIMIT 5",
    ),
    ("cypher_fn_keys", "MATCH (f:Field {title: 'TROLL'}) RETURN keys(f) AS props"),
    (
        "cypher_fn_size_length",
        "MATCH (f:Field {title: 'TROLL'})-[:INCLUDES_DISCOVERY]->(d:Discovery) WITH collect(d.title) AS discs RETURN size(discs) AS count",
    ),
    # ── CASE Expression ────────────────────────────────────────────
    (
        "cypher_case",
        "MATCH (w:Wellbore) RETURN CASE WHEN w.wlbTotalDepth > 5000 THEN 'deep' WHEN w.wlbTotalDepth > 2000 THEN 'medium' ELSE 'shallow' END AS depth_class, count(w) AS cnt ORDER BY cnt DESC",
    ),
    # ── coalesce ───────────────────────────────────────────────────
    ("cypher_coalesce", "MATCH (w:Wellbore) RETURN w.title, coalesce(w.wlbContent, 'UNKNOWN') AS content LIMIT 10"),
    # ── List Comprehension & Slicing ───────────────────────────────
    (
        "cypher_list_comprehension",
        "MATCH (f:Field {title: 'TROLL'})-[:INCLUDES_DISCOVERY]->(d:Discovery) WITH collect(d.title) AS discs RETURN [x IN discs WHERE x STARTS WITH 'T'] AS t_discoveries",
    ),
    (
        "cypher_list_slicing",
        "MATCH (f:Field {title: 'TROLL'})-[:INCLUDES_DISCOVERY]->(d:Discovery) WITH collect(d.title) AS discs RETURN discs[0..3] AS first_three",
    ),
    # ── Map Projection ─────────────────────────────────────────────
    (
        "cypher_map_projection",
        "MATCH (f:Field {title: 'TROLL'}) RETURN f {.title, .fldStatus, .fldMainArea} AS field_map",
    ),
    ("cypher_map_literal", "MATCH (f:Field) RETURN {name: f.title, status: f.fldStatus} AS info LIMIT 5"),
    # ── Date Functions & Arithmetic ────────────────────────────────
    ("cypher_date_parse", "RETURN date('2020-06-15') AS d, datetime('2020-06-15') AS dt"),
    ("cypher_date_arithmetic", "WITH date('2020-01-01') AS d RETURN d + 30 AS plus30, d - 10 AS minus10"),
    (
        "cypher_date_diff",
        "WITH date('2020-01-01') AS a, date('2025-06-15') AS b RETURN date_diff(a, b) AS days_between",
    ),
    ("cypher_date_accessors", "WITH date('2020-06-15') AS d RETURN d.year AS y, d.month AS m, d.day AS dy"),
    # ── Temporal Functions (valid_at / valid_during) ───────────────
    (
        "cypher_valid_at",
        "MATCH (f:Field)-[r:HAS_LICENSEE]->(c:Company) WHERE valid_at(r, '2010', 'fldLicenseeFrom', 'fldLicenseeTo') RETURN f.title, c.title LIMIT 20",
    ),
    (
        "cypher_valid_during",
        "MATCH (f:Field)-[r:HAS_LICENSEE]->(c:Company) WHERE valid_during(r, '2000', '2010', 'fldLicenseeFrom', 'fldLicenseeTo') RETURN f.title, count(c) AS licensees ORDER BY licensees DESC LIMIT 10",
    ),
    # ── Window Functions ───────────────────────────────────────────
    (
        "cypher_window_row_number",
        "MATCH (w:Wellbore) WHERE w.wlbTotalDepth IS NOT NULL RETURN w.title, w.wlbMainArea, w.wlbTotalDepth, row_number() OVER (PARTITION BY w.wlbMainArea ORDER BY w.wlbTotalDepth DESC) AS rn LIMIT 30",
    ),
    (
        "cypher_window_rank",
        "MATCH (w:Wellbore) WHERE w.wlbTotalDepth IS NOT NULL RETURN w.title, w.wlbMainArea, rank() OVER (PARTITION BY w.wlbMainArea ORDER BY w.wlbTotalDepth DESC) AS rnk LIMIT 30",
    ),
    (
        "cypher_window_dense_rank",
        "MATCH (c:Company)<-[:DRILLED_BY]-(w:Wellbore) WITH c.title AS company, count(w) AS wells RETURN company, wells, dense_rank() OVER (ORDER BY wells DESC) AS drnk LIMIT 20",
    ),
    # ── EXISTS Subquery ────────────────────────────────────────────
    (
        "cypher_exists_subquery",
        "MATCH (f:Field) WHERE EXISTS { MATCH (f)-[:INCLUDES_DISCOVERY]->(d:Discovery) WHERE d.dscCurrentActivityStatus = 'PRODUCING' } RETURN f.title",
    ),
    # ── Spatial Queries ────────────────────────────────────────────
    (
        "cypher_spatial_distance",
        "MATCH (a:Field), (b:Field) WHERE a.title = 'TROLL' AND b.title = 'OSEBERG' RETURN distance(a, b) AS dist_m",
    ),
    ("cypher_spatial_area_top10", "MATCH (f:Field) RETURN f.title, area(f) AS area_m2 ORDER BY area_m2 DESC LIMIT 10"),
    (
        "cypher_spatial_nearest_10",
        "MATCH (a:Field {title: 'TROLL'}), (b:Field) WHERE a <> b RETURN b.title, distance(a, b) AS dist_m ORDER BY dist_m ASC LIMIT 10",
    ),
    (
        "cypher_spatial_pairs_10km",
        "MATCH (a:Field), (b:Field) WHERE a <> b AND distance(a, b) < 10000 RETURN a.title, b.title, distance(a, b) AS dist_m ORDER BY dist_m LIMIT 10",
    ),
    ("cypher_spatial_perimeter", "MATCH (f:Field) WHERE f.title = 'TROLL' RETURN perimeter(f) AS perimeter_m"),
    ("cypher_spatial_contains", "MATCH (f:Field), (b:Block) WHERE contains(f, b) AND f.title = 'TROLL' RETURN b.title"),
    ("cypher_spatial_centroid", "MATCH (f:Field {title: 'TROLL'}) RETURN centroid(f) AS center"),
    (
        "cypher_spatial_intersects",
        "MATCH (a:Field), (b:Block) WHERE intersects(a, b) AND a.title = 'TROLL' RETURN b.title",
    ),
    # ── Time Series Queries ────────────────────────────────────────
    (
        "cypher_ts_sum_single",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field {title: 'TROLL'}) RETURN ts_sum(pp.prd_oil_net) AS total_oil_net",
    ),
    (
        "cypher_ts_avg_window",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field {title: 'TROLL'}) RETURN ts_avg(pp.prd_oil_net, '2020', '2024') AS avg_oil",
    ),
    (
        "cypher_ts_min_max",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field {title: 'TROLL'}) RETURN ts_min(pp.prd_oil_net) AS min_oil, ts_max(pp.prd_oil_net) AS peak_oil",
    ),
    (
        "cypher_ts_first_last_count",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field {title: 'JOHAN SVERDRUP'}) RETURN ts_first(pp.prd_oil_net) AS first, ts_last(pp.prd_oil_net) AS last, ts_count(pp.prd_oil_net) AS months",
    ),
    (
        "cypher_ts_at",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field {title: 'TROLL'}) RETURN ts_at(pp.prd_oil_net, '2023-06') AS june_2023",
    ),
    (
        "cypher_ts_delta",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field {title: 'TROLL'}) RETURN ts_delta(pp.prd_oil_net, '2020', '2024') AS change",
    ),
    (
        "cypher_ts_series_extract",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field {title: 'EKOFISK'}) RETURN ts_series(pp.prd_oil_net, '2022', '2024') AS oil_series",
    ),
    (
        "cypher_ts_top10_oil",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field) RETURN f.title, ts_sum(pp.prd_oil_net) AS total_oil ORDER BY total_oil DESC LIMIT 10",
    ),
    (
        "cypher_ts_top10_gas_window",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field) RETURN f.title, ts_sum(pp.prd_gas_net, '2023', '2024') AS gas ORDER BY gas DESC LIMIT 10",
    ),
    (
        "cypher_ts_multi_channel",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field {title: 'TROLL'}) RETURN ts_sum(pp.prd_oil_net) AS oil, ts_sum(pp.prd_gas_net) AS gas, ts_sum(pp.prd_water) AS water",
    ),
    # ── Graph Algorithms ───────────────────────────────────────────
    (
        "cypher_algo_pagerank",
        "CALL pagerank({connection_types: 'INCLUDES_DISCOVERY'}) YIELD node, score RETURN node.title, score ORDER BY score DESC LIMIT 10",
    ),
    (
        "cypher_algo_degree",
        "CALL degree({connection_types: 'DRILLED_BY'}) YIELD node, score RETURN node.title, score ORDER BY score DESC LIMIT 10",
    ),
    (
        "cypher_algo_betweenness",
        "CALL betweenness({node_type: 'Field', sample_size: 50}) YIELD node, score RETURN node.title, score ORDER BY score DESC LIMIT 10",
    ),
    (
        "cypher_algo_closeness",
        "CALL closeness({node_type: 'Field', sample_size: 100}) YIELD node, score RETURN node.title, score ORDER BY score DESC LIMIT 10",
    ),
    (
        "cypher_algo_louvain",
        "CALL louvain({connection_types: 'HAS_LICENSEE'}) YIELD node, community RETURN community, collect(node.title) AS members ORDER BY size(members) DESC LIMIT 5",
    ),
    (
        "cypher_algo_label_prop",
        "CALL label_propagation({connection_types: 'HAS_LICENSEE'}) YIELD node, community RETURN community, count(*) AS size ORDER BY size DESC LIMIT 5",
    ),
    (
        "cypher_algo_components",
        "CALL connected_components() YIELD node, component RETURN component, count(*) AS size ORDER BY size DESC LIMIT 5",
    ),
    # ── EXPLAIN / PROFILE ──────────────────────────────────────────
    (
        "cypher_explain",
        "EXPLAIN MATCH (w:Wellbore)-[:IN_FIELD]->(f:Field) RETURN f.title, count(w) ORDER BY count(w) DESC LIMIT 10",
    ),
    (
        "cypher_profile",
        "PROFILE MATCH (w:Wellbore)-[:IN_FIELD]->(f:Field) RETURN f.title, count(w) ORDER BY count(w) DESC LIMIT 10",
    ),
    # ── Full Scans & Heavy Operations ──────────────────────────────
    ("cypher_scan_filter_count", "MATCH (w:Wellbore) WHERE w.wlbTotalDepth > 5000 RETURN count(w)"),
    (
        "cypher_scan_large_result",
        "MATCH (w:Wellbore) RETURN w.title, w.wlbPurpose, w.wlbTotalDepth, w.wlbContent LIMIT 5000",
    ),
    (
        "cypher_scan_full_pattern",
        "MATCH (f:Field)-[:OF_FIELD]-(w:Wellbore) RETURN f.title, count(w) AS wells ORDER BY wells DESC LIMIT 20",
    ),
    ("cypher_scan_edge_type_counts", "MATCH ()-[r]->() RETURN type(r), count(*) ORDER BY count(*) DESC"),
    (
        "cypher_scan_of_wellbore",
        "MATCH (w:Wellbore)-[:OF_WELLBORE]-(sub) RETURN labels(sub)[0] AS subtype, count(sub) AS cnt ORDER BY cnt DESC",
    ),
    # ── Complex Analytical Queries ─────────────────────────────────
    (
        "cypher_complex_2hop_filter_agg",
        "MATCH (c:Company)<-[:DRILLED_BY]-(w:Wellbore)-[:IN_FIELD]->(f:Field) WHERE w.wlbPurpose = 'WILDCAT' AND w.wlbContent CONTAINS 'OIL' RETURN c.title, count(DISTINCT f) AS fields, count(w) AS wells ORDER BY wells DESC LIMIT 10",
    ),
    (
        "cypher_complex_ts_all_fields",
        "MATCH (f:Field)<-[:OF_FIELD]-(pp:ProductionProfile) WITH f, ts_sum(pp.prd_oil_net) AS oil, ts_sum(pp.prd_gas_net) AS gas RETURN f.title, oil, gas, oil + gas AS total_oe ORDER BY total_oe DESC LIMIT 10",
    ),
    (
        "cypher_complex_2hop_distinct",
        "MATCH (w:Wellbore)-[:IN_LICENCE]->(l:Licence)-[:HAS_LICENSEE]->(c:Company) WHERE w.wlbPurpose = 'WILDCAT' AND w.wlbEntryYear >= 2015 RETURN c.title, count(DISTINCT w) AS wildcats, count(DISTINCT l) AS licences ORDER BY wildcats DESC LIMIT 10",
    ),
    (
        "cypher_complex_spatial_ts",
        "MATCH (f:Field) WITH f, area(f) AS field_area MATCH (f)<-[:OF_FIELD]-(pp:ProductionProfile) WITH f.title AS field, field_area, ts_sum(pp.prd_oil_net) AS oil WHERE field_area > 0 RETURN field, oil, field_area, oil / field_area * 1e6 AS oil_per_km2 ORDER BY oil_per_km2 DESC LIMIT 10",
    ),
    (
        "cypher_complex_formations_oil",
        "MATCH (w:Wellbore)-[:HAS_FORMATION_TOP]->(s:Stratigraphy) WHERE w.wlbPurpose = 'WILDCAT' AND w.wlbContent CONTAINS 'OIL' RETURN s.title AS formation, count(DISTINCT w) AS oil_wells ORDER BY oil_wells DESC LIMIT 10",
    ),
    (
        "cypher_complex_production_decades",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field) RETURN f.title, ts_sum(pp.prd_oil_net, '2000', '2010') AS oil_00s, ts_sum(pp.prd_oil_net, '2010', '2020') AS oil_10s, ts_sum(pp.prd_oil_net, '2020', '2026') AS oil_20s ORDER BY oil_00s DESC LIMIT 20",
    ),
    (
        "cypher_complex_multi_ts_agg",
        "MATCH (pp:ProductionProfile)-[:OF_FIELD]->(f:Field) RETURN f.title, ts_sum(pp.prd_oil_net) AS oil, ts_sum(pp.prd_gas_net) AS gas, ts_sum(pp.prd_water) AS water, ts_first(pp.prd_oil_net) AS first_oil, ts_last(pp.prd_oil_net) AS last_oil ORDER BY oil DESC",
    ),
    ("cypher_complex_many_small", None),  # Special: 100× small queries — handled in runner
    # ── Mutations (on graph copy) ──────────────────────────────────
    ("cypher_mutation_create", None),  # handled specially
    ("cypher_mutation_set", None),
    ("cypher_mutation_merge", None),
    ("cypher_mutation_delete", None),
]

MUTATION_CYPHER = {
    "cypher_mutation_create": "CREATE (n:TestNode {name: 'benchmark', value: 42})",
    "cypher_mutation_set": "MATCH (f:Field {title: 'TROLL'}) SET f._bench_test = 'hello'",
    "cypher_mutation_merge": "MERGE (n:TestNode {name: 'merged'}) ON CREATE SET n.created = true ON MATCH SET n.updated = true",
    "cypher_mutation_delete": "MATCH (n:TestNode) DELETE n",
}


# ═══════════════════════════════════════════════════════════════════
# CSV pre-processing
# ═══════════════════════════════════════════════════════════════════


def csv_path(rel: str) -> str:
    return os.path.join(CSV_ROOT, rel)


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(csv_path(rel), low_memory=False)


def save_csv(df: pd.DataFrame, rel: str) -> None:
    df.to_csv(csv_path(rel), index=False)


def preprocess_csvs():
    """Add integer FK columns that the blueprint expects."""
    df_ptl = read_csv("sodir-data/csv/petreg_licence.csv")
    df_ptl["ptl_id"] = range(1, len(df_ptl) + 1)
    guid_to_id = dict(zip(df_ptl["ptlPetregLicenceID"], df_ptl["ptl_id"]))
    save_csv(df_ptl, "sodir-data/csv/petreg_licence.csv")

    df_msg = read_csv("sodir-data/csv/petreg_licence_message.csv")
    df_msg["ptl_id"] = df_msg["ptlPetregLicenceID"].map(guid_to_id).astype("Int64")
    save_csv(df_msg, "sodir-data/csv/petreg_licence_message.csv")

    for junc_csv in [
        "sodir-data/csv/petreg_licence_licensee.csv",
        "sodir-data/csv/petreg_licence_operator.csv",
    ]:
        df_j = read_csv(junc_csv)
        df_j["ptl_id"] = df_j["ptlPetregLicenceID"].map(guid_to_id).astype("Int64")
        save_csv(df_j, junc_csv)

    df_survey = read_csv("sodir-data/csv/seismic_acquisition.csv")
    name_to_npdid = dict(zip(df_survey["seaName"], df_survey["seaNpdidSurvey"]))
    df_prog = read_csv("sodir-data/csv/seismic_acquisition_progress.csv")
    df_prog["seaNpdidSurvey"] = df_prog["seaSurveyName"].map(name_to_npdid).astype("Int64")
    save_csv(df_prog, "sodir-data/csv/seismic_acquisition_progress.csv")

    df_chrono = read_csv("sodir-data/csv/strat_chrono.csv")
    name_to_npdid = dict(zip(df_chrono["strat_chrono_name"], df_chrono["NPDID_strat_chrono"]))
    df_chrono["strat_chrono_parent_npdid"] = df_chrono["strat_chrono_parent_name"].map(name_to_npdid).astype("Int64")
    save_csv(df_chrono, "sodir-data/csv/strat_chrono.csv")

    df_block = read_csv("sodir-data/csv/block.csv")
    name_to_npdid = dict(zip(df_block["blcName"], df_block["blcNpdidBlock"]))
    df_ah = read_csv("sodir-data/csv/announced_history.csv")
    df_ah["blcNpdidBlock"] = df_ah["block"].map(name_to_npdid).astype("Int64")
    save_csv(df_ah, "sodir-data/csv/announced_history.csv")

    df_prospect = read_csv("sodir-data/csv-extra/prospect.csv")
    if "shape" in df_prospect.columns:
        df_prospect.rename(columns={"shape": "wkt_geometry"}, inplace=True)
        save_csv(df_prospect, "sodir-data/csv-extra/prospect.csv")

    num_to_npdid = dict(zip(df_prospect["prospect_number"], df_prospect["npdid_prospect"]))
    df_wp = read_csv("sodir-data/csv-extra/wellbore_plan_prospect.csv")
    df_wp["npdid_prospect"] = df_wp["npd_prospid"].map(num_to_npdid).astype("Int64")
    save_csv(df_wp, "sodir-data/csv-extra/wellbore_plan_prospect.csv")


# ═══════════════════════════════════════════════════════════════════
# Benchmark helpers
# ═══════════════════════════════════════════════════════════════════


def bench(fn, iterations=ITERATIONS, warmup=WARMUP):
    """Run fn() multiple times, return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
    return statistics.median(times)


def build_fluent_benchmarks(graph):
    """Return list of (name, callable, iterations, warmup) for all fluent benchmarks."""
    B = []  # (name, fn, iters, warmup)

    def add(name, fn, iters=ITERATIONS, wu=WARMUP):
        B.append((name, fn, iters, wu))

    # ── Introspection ──────────────────────────────────────────────
    add("fluent_schema", lambda: graph.schema())
    add("fluent_describe", lambda: graph.describe())
    add("fluent_len", lambda: len(graph))
    add("fluent_node_type_counts", lambda: graph.node_type_counts())
    add("fluent_properties", lambda: graph.properties("Wellbore"))
    add("fluent_connection_types", lambda: graph.connection_types())

    # ── Select ─────────────────────────────────────────────────────
    add("fluent_select_large", lambda: graph.select("Wellbore"))
    add("fluent_select_medium", lambda: graph.select("Field"))
    add("fluent_select_sort", lambda: graph.select("Field", sort="title"))
    add("fluent_select_limit", lambda: graph.select("Wellbore", limit=10))
    add("fluent_select_temporal", lambda: graph.select("FieldStatusHistory"))
    add("fluent_select_temporal_false", lambda: graph.select("FieldStatusHistory", temporal=False))

    # ── Where ──────────────────────────────────────────────────────
    add("fluent_where_eq", lambda: graph.select("Field").where({"title": "TROLL"}))
    add("fluent_where_contains", lambda: graph.select("Wellbore").where({"title": {"contains": "TROLL"}}))
    add("fluent_where_starts_with", lambda: graph.select("Wellbore").where({"title": {"starts_with": "31/"}}))
    add("fluent_where_ends_with", lambda: graph.select("Wellbore").where({"title": {"ends_with": "S"}}))
    add("fluent_where_regex", lambda: graph.select("Wellbore").where({"title": {"=~": "31/[0-9]+-.*"}}))
    add("fluent_where_gt", lambda: graph.select("Wellbore").where({"wlbTotalDepth": {">": 5000}}))
    add("fluent_where_range", lambda: graph.select("Wellbore").where({"wlbEntryYear": {">=": 2020, "<=": 2025}}))
    add("fluent_where_in", lambda: graph.select("Field").where({"fldStatus": {"in": ["PRODUCING", "SHUT DOWN"]}}))
    add("fluent_where_is_not_null", lambda: graph.select("Wellbore").where({"wlbContent": {"is_not_null": True}}))
    add("fluent_where_is_null", lambda: graph.select("Wellbore").where({"wlbContent": {"is_null": True}}))
    add("fluent_where_not_contains", lambda: graph.select("Wellbore").where({"title": {"not_contains": "TROLL"}}))
    add(
        "fluent_where_any",
        lambda: graph.select("Wellbore").where_any([{"wlbMainArea": "NORTH SEA"}, {"wlbMainArea": "NORWEGIAN SEA"}]),
    )
    add("fluent_where_connected", lambda: graph.select("Field").where_connected("INCLUDES_DISCOVERY"))

    # ── Traverse ───────────────────────────────────────────────────
    add("fluent_traverse_1hop", lambda: graph.select("Field").where({"title": "TROLL"}).traverse("HAS_LICENSEE"))
    add("fluent_traverse_all_fields", lambda: graph.select("Field").traverse("HAS_LICENSEE"))
    add("fluent_traverse_large_fanout", lambda: graph.select("Wellbore").traverse("IN_FIELD"))
    add("fluent_traverse_no_temporal", lambda: graph.select("Field").traverse("HAS_LICENSEE", temporal=False))
    add("fluent_traverse_at", lambda: graph.select("Field").traverse("HAS_LICENSEE", at="2005"))
    add("fluent_traverse_during", lambda: graph.select("Field").traverse("HAS_LICENSEE", during=("2000", "2010")))
    add(
        "fluent_traverse_multihop",
        lambda: graph.select("Field").where({"title": "TROLL"}).traverse("INCLUDES_DISCOVERY").traverse("IN_PLAY"),
    )
    add(
        "fluent_traverse_where",
        lambda: graph.select("Field").traverse("HAS_LICENSEE", where={"title": "Equinor Energy AS"}),
    )
    add(
        "fluent_traverse_target_type",
        lambda: graph.select("Field").traverse("OF_FIELD", direction="incoming", target_type="ProductionProfile"),
    )

    # ── Collect / Output ───────────────────────────────────────────
    add("fluent_collect_single", lambda: graph.select("Field").where({"title": "TROLL"}).collect())
    add("fluent_collect_all_fields", lambda: graph.select("Field").collect())
    add("fluent_collect_1000", lambda: graph.select("Wellbore", limit=1000).collect())
    add("fluent_collect_all_wellbores", lambda: graph.select("Wellbore").collect())
    add("fluent_to_df", lambda: graph.select("Field").to_df())
    add("fluent_to_df_large", lambda: graph.select("Wellbore").to_df())
    add("fluent_ids", lambda: graph.select("Field").ids())
    add("fluent_titles", lambda: graph.select("Field").titles())
    add("fluent_sample_small", lambda: graph.sample("Field", 5))
    add("fluent_sample_large", lambda: graph.sample("Wellbore", 100))
    add("fluent_show", lambda: graph.select("Field").where({"title": "TROLL"}).show(["title", "fldStatus"]))

    # ── Statistics & Aggregation ───────────────────────────────────
    add("fluent_statistics", lambda: graph.select("Wellbore").statistics("wlbTotalDepth"))
    add(
        "fluent_statistics_groupby",
        lambda: graph.select("Wellbore").statistics("wlbTotalDepth", group_by="wlbMainArea"),
    )
    add("fluent_count", lambda: graph.select("Wellbore").count())
    add("fluent_count_groupby", lambda: graph.select("Wellbore").count(group_by="wlbMainArea"))
    add("fluent_unique_values", lambda: graph.select("Wellbore").unique_values("wlbPurpose", group_by_parent=False))

    # ── Temporal Context ───────────────────────────────────────────
    add("fluent_date_create", lambda: graph.date("2010"))
    add("fluent_date_select", lambda: graph.date("2010").select("FieldStatusHistory"))
    add("fluent_date_traverse", lambda: graph.date("2005").select("Field").traverse("HAS_LICENSEE"))
    add("fluent_date_range_traverse", lambda: graph.date("2000", "2010").select("Field").traverse("HAS_LICENSEE"))
    add("fluent_date_all_select", lambda: graph.date("all").select("FieldStatusHistory"))
    add("fluent_valid_at", lambda: graph.select("FieldStatusHistory", temporal=False).valid_at("2010"))
    add("fluent_valid_during", lambda: graph.select("FieldStatusHistory", temporal=False).valid_during("2000", "2010"))

    # ── Spatial ────────────────────────────────────────────────────
    add("fluent_near_point_m", lambda: graph.select("Field").near_point_m(60.6, 3.7, 50_000))
    add("fluent_within_bounds", lambda: graph.select("Field").within_bounds(58.0, 62.0, 2.0, 6.0))
    add("fluent_bounds", lambda: graph.select("Field").bounds())
    add("fluent_centroid", lambda: graph.select("Field").centroid())

    # ── Graph Algorithms ───────────────────────────────────────────
    add("fluent_pagerank", lambda: graph.select("Field").pagerank(top_k=10))
    add("fluent_degree_centrality", lambda: graph.select("Field").degree_centrality(top_k=10))
    add("fluent_betweenness", lambda: graph.select("Field").betweenness_centrality(top_k=10, sample_size=50))
    add("fluent_closeness", lambda: graph.select("Field").closeness_centrality(top_k=10, sample_size=100))
    add("fluent_louvain", lambda: graph.select("Field").louvain_communities())
    add("fluent_label_propagation", lambda: graph.select("Field").label_propagation())
    add("fluent_connected_components", lambda: graph.select("Field").connected_components())

    # ── Path Finding ───────────────────────────────────────────────
    troll_ids = graph.select("Field").where({"title": "TROLL"}).ids()
    ekofisk_ids = graph.select("Field").where({"title": "EKOFISK"}).ids()
    if troll_ids and ekofisk_ids:
        troll_id, ekofisk_id = troll_ids[0], ekofisk_ids[0]
        add("fluent_shortest_path", lambda: graph.shortest_path("Field", troll_id, "Field", ekofisk_id))
        add("fluent_shortest_path_length", lambda: graph.shortest_path_length("Field", troll_id, "Field", ekofisk_id))
        add("fluent_are_connected", lambda: graph.are_connected("Field", troll_id, "Field", ekofisk_id))

    # ── Set Operations ─────────────────────────────────────────────
    north_sea = graph.select("Wellbore").where({"wlbMainArea": "NORTH SEA"})
    wildcats = graph.select("Wellbore").where({"wlbPurpose": "WILDCAT"})
    add("fluent_union", lambda: north_sea.union(wildcats))
    add("fluent_intersection", lambda: north_sea.intersection(wildcats))
    add("fluent_difference", lambda: north_sea.difference(wildcats))
    add("fluent_symmetric_difference", lambda: north_sea.symmetric_difference(wildcats))

    # ── Timeseries ─────────────────────────────────────────────────
    pp_ids = graph.select("ProductionProfile").ids()
    if pp_ids:
        pp_id = pp_ids[0]
        add("fluent_timeseries_get", lambda: graph.timeseries(pp_id, "prd_oil_net"))
        add("fluent_timeseries_range", lambda: graph.timeseries(pp_id, "prd_oil_net", start="2020", end="2024"))
        add("fluent_time_index", lambda: graph.time_index(pp_id))

    # ── Chained Pipelines ──────────────────────────────────────────
    add(
        "fluent_pipeline_select_where_traverse_collect",
        lambda: graph.select("Field").where({"title": "TROLL"}).traverse("HAS_LICENSEE").collect(),
    )
    add(
        "fluent_pipeline_date_full",
        lambda: graph.date("2005").select("Field").where({"title": "TROLL"}).traverse("HAS_LICENSEE").collect(),
    )
    add(
        "fluent_pipeline_multihop_collect",
        lambda: (
            graph.select("Field").where({"title": "TROLL"}).traverse("INCLUDES_DISCOVERY").traverse("IN_PLAY").collect()
        ),
    )
    add("fluent_pipeline_large_collect", lambda: graph.date("all").select("Licence").traverse("HAS_LICENSEE").collect())
    add(
        "fluent_pipeline_fanout_collect",
        lambda: graph.select("Wellbore", limit=500).traverse("OF_WELLBORE", direction="incoming").collect(),
    )

    # ── Mutations (deep-copy graph each time, avoids disk I/O) ─────
    def _bench_update():
        g2 = graph.copy()
        g2.select("Field").where({"title": "TROLL"}).update({"_bench_test": "value"})

    add("fluent_update", _bench_update, iters=3, wu=0)

    def _bench_add_nodes():
        g2 = graph.copy()
        df = pd.DataFrame({"id": range(1000), "title": [f"Test_{i}" for i in range(1000)]})
        g2.add_nodes(df, "TestNode", "id", "title")

    add("fluent_add_nodes_1k", _bench_add_nodes, iters=3, wu=0)

    def _bench_add_connections():
        g2 = graph.copy()
        df = pd.DataFrame({"id": range(100), "title": [f"Test_{i}" for i in range(100)]})
        g2.add_nodes(df, "TestNode", "id", "title")
        edge_df = pd.DataFrame({"source": list(range(50)), "target": list(range(50, 100))})
        g2.add_connections(edge_df, "TEST_EDGE", "TestNode", "source", "TestNode", "target")

    add("fluent_add_connections", _bench_add_connections, iters=3, wu=0)

    return B


# ═══════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════


def run_benchmarks():
    version = kglite.__version__
    results: dict[str, float] = {}
    errors: list[tuple[str, str]] = []

    # ── Setup: preprocess + build + save + load ────────────────────
    print(f"KGLite v{version} — Sodir benchmark")
    print()

    with contextlib.redirect_stdout(io.StringIO()):
        preprocess_csvs()

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        graph = kglite.from_blueprint(BLUEPRINT)
    build_ms = (time.perf_counter() - t0) * 1000
    results["build_from_blueprint"] = round(build_ms, 1)
    s = graph.schema()
    print(f"  Build:  {build_ms:>8.0f} ms  ({s['node_count']} nodes, {s['edge_count']} edges)")

    save_ms = bench(lambda: graph.save(TEMP_KGL), iterations=3, warmup=0)
    results["save_kgl"] = round(save_ms, 1)
    size_mb = os.path.getsize(TEMP_KGL) / (1024 * 1024)
    print(f"  Save:   {save_ms:>8.0f} ms  ({size_mb:.1f} MB)")

    load_ms = bench(lambda: kglite.load(TEMP_KGL), iterations=3, warmup=0)
    results["load_kgl"] = round(load_ms, 1)
    print(f"  Load:   {load_ms:>8.0f} ms")
    print()

    graph = kglite.load(TEMP_KGL)

    # ── Collect all benchmark tasks ────────────────────────────────
    tasks: list[tuple[str, callable, int, int]] = []

    # Cypher queries
    for name, query in CYPHER_QUERIES:
        heavy = name in HEAVY_BENCHMARKS
        it = HEAVY_ITERATIONS if heavy else ITERATIONS
        wu = HEAVY_WARMUP if heavy else WARMUP
        if query is not None:
            tasks.append((name, lambda q=query: list(graph.cypher(q)), it, wu))
        elif "many_small" in name:
            q = "MATCH (f:Field {title: 'TROLL'}) RETURN f.title"
            tasks.append((name, lambda: [list(graph.cypher(q)) for _ in range(100)], it, wu))
        elif name in MUTATION_CYPHER:
            mut_q = MUTATION_CYPHER[name]

            def make_mut(q=mut_q):
                g2 = graph.copy()
                g2.cypher(q)

            tasks.append((name, make_mut, 3, 0))

    # Fluent API
    tasks.extend(build_fluent_benchmarks(graph))

    # ── Run with progress bar ──────────────────────────────────────
    pbar = tqdm(tasks, desc="Benchmarking", unit="bench", ncols=90)
    for name, fn, iters, wu in pbar:
        pbar.set_postfix_str(name, refresh=True)
        try:
            ms = bench(fn, iterations=iters, warmup=wu)
            results[name] = round(ms, 2)
        except Exception as e:
            results[name] = -1
            errors.append((name, str(e)))

    # ── Cleanup ────────────────────────────────────────────────────
    if os.path.exists(TEMP_KGL):
        os.remove(TEMP_KGL)

    # ── Summary ────────────────────────────────────────────────────
    print()
    cypher_ok = [v for k, v in results.items() if k.startswith("cypher_") and v >= 0]
    fluent_ok = [v for k, v in results.items() if k.startswith("fluent_") and v >= 0]
    total = len(cypher_ok) + len(fluent_ok)
    print(f"  Build:         {results['build_from_blueprint']:>8.0f} ms")
    print(f"  Save:          {results['save_kgl']:>8.0f} ms")
    print(f"  Load:          {results['load_kgl']:>8.0f} ms")
    print(f"  Cypher:        {sum(cypher_ok):>8.1f} ms  ({len(cypher_ok)} benchmarks)")
    print(f"  Fluent:        {sum(fluent_ok):>8.1f} ms  ({len(fluent_ok)} benchmarks)")
    print(f"  Total:         {sum(cypher_ok) + sum(fluent_ok):>8.1f} ms  ({total} benchmarks)")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for name, msg in errors:
            print(f"    {name}: {msg}")

    print()
    return version, results


# ═══════════════════════════════════════════════════════════════════
# CSV persistence
# ═══════════════════════════════════════════════════════════════════


def load_existing_csv() -> tuple[list[str], list[str], dict[str, dict[str, str]]]:
    """Load existing benchmark.csv → (benchmark_names, col_names, data)."""
    if not os.path.exists(CSV_OUT):
        return [], [], {}

    with open(CSV_OUT, newline="") as f:
        reader = csv.DictReader(f)
        # Support both old header "benchmark" and new "benchmark (ms)"
        bm_key = next((c for c in reader.fieldnames if c.startswith("benchmark")), "benchmark")
        col_names = [c for c in reader.fieldnames if c != bm_key]
        data: dict[str, dict[str, str]] = {}
        benchmark_names: list[str] = []
        for row in reader:
            bm = row[bm_key]
            benchmark_names.append(bm)
            data[bm] = {c: row[c] for c in col_names}

    return benchmark_names, col_names, data


def next_column_name(existing_cols: list[str], version: str) -> str:
    """Determine column name: v0.5.79, v0.5.79_2, v0.5.79_3, ..."""
    base = f"v{version}"
    if base not in existing_cols:
        return base

    pattern = re.compile(re.escape(base) + r"(?:_(\d+))?$")
    max_suffix = 1
    for col in existing_cols:
        m = pattern.match(col)
        if m:
            suffix = int(m.group(1)) if m.group(1) else 1
            max_suffix = max(max_suffix, suffix)

    return f"{base}_{max_suffix + 1}"


def save_to_csv(version: str, results: dict[str, float]):
    """Append results as a new column in benchmark.csv."""
    benchmark_names, col_names, data = load_existing_csv()

    new_col = next_column_name(col_names, version)
    col_names.append(new_col)

    for bm in results:
        if bm not in data:
            data[bm] = {}
            if bm not in benchmark_names:
                benchmark_names.append(bm)
        data[bm][new_col] = str(results[bm])

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark (ms)"] + col_names)
        for bm in benchmark_names:
            row = [bm] + [data.get(bm, {}).get(c, "") for c in col_names]
            writer.writerow(row)

    print(f"Results written to {CSV_OUT}  (column: {new_col})")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not os.path.isdir(CSV_ROOT):
        print(f"ERROR: CSV root not found: {CSV_ROOT}")
        print("This benchmark requires the Sodir CSV data.")
        sys.exit(1)
    if not os.path.exists(BLUEPRINT):
        print(f"ERROR: Blueprint not found: {BLUEPRINT}")
        sys.exit(1)

    version, results = run_benchmarks()
    save_to_csv(version, results)
