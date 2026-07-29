"""Norwegian legal-graph benchmark: build → save → load → query → report.

Builds the Norwegian legal knowledge graph from JSON data (113k nodes,
692k edges), saves/loads a temp file, runs Cypher + Fluent API benchmarks,
and appends results to benchmark_legal.csv (one column per kglite version).

Requires: kglite, pandas, tqdm
Data:     /Volumes/EksternalHome/Koding/Python/Scraping/processed
Build:    /Volumes/EksternalHome/Koding/MCP servers/legal/build_legal_graph.py
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
TEMP_KGL = str(SCRIPT_DIR / "temp_legal.kgl")
CSV_OUT = str(SCRIPT_DIR / "benchmark_legal.csv")

ITERATIONS = 5
WARMUP = 1

# Heavy benchmarks get fewer iterations
HEAVY_BENCHMARKS = {
    "cypher_algo_betweenness",
    "cypher_algo_closeness",
    "cypher_algo_components",
    "cypher_algo_louvain",
    "cypher_algo_label_prop",
    "cypher_scan_edge_type_counts",
    "cypher_complex_many_small",
}
HEAVY_ITERATIONS = 2
HEAVY_WARMUP = 0

# Import the build module for graph construction
LEGAL_DIR = Path("/Volumes/EksternalHome/Koding/MCP servers/legal")
sys.path.insert(0, str(LEGAL_DIR))
import build_legal_graph as blg

# ═══════════════════════════════════════════════════════════════════
# Graph construction (reuses build_legal_graph.py helpers)
# ═══════════════════════════════════════════════════════════════════


def build_legal_graph() -> kglite.KnowledgeGraph:
    """Build the full Norwegian legal graph (phases 1-12, no embed/save)."""
    graph = kglite.KnowledgeGraph()

    # Phase 1: Load raw data
    laws_raw = blg.load_json(blg.LAWS_INDEX)
    regs_raw = blg.load_json(blg.REGS_INDEX)
    hr_raw = blg.load_json(blg.HR_INDEX)
    lr_raw = blg.load_json(blg.LR_INDEX)
    all_cases = hr_raw + lr_raw

    # Phase 2: Department nodes
    dept_names: set[str] = set()
    for item in laws_raw + regs_raw:
        for d in item.get("department", "").split(", "):
            d = d.strip()
            if d:
                dept_names.add(d)
    dept_df = pd.DataFrame({"dept_id": list(dept_names), "name": list(dept_names)})
    graph.add_nodes(dept_df, "Department", "dept_id", "name")

    # Phase 3: Law nodes
    law_records = []
    for item in laws_raw:
        law_id = item.get("law_id", "")
        law_records.append(
            {
                "law_node_id": item["id"],
                "name": item["title"],
                "law_id": law_id,
                "law_date_key": law_id.replace("LOV-", ""),
                "korttittel": item.get("korttittel", ""),
                "url": item.get("url", ""),
                "file_path": item.get("file_path", ""),
                "file_size": item.get("file_size", 0),
                "dato": item.get("dato", ""),
            }
        )
    graph.add_nodes(pd.DataFrame(law_records), "Law", "law_node_id", "name")

    # Phase 4: Regulation nodes
    reg_records = []
    for item in regs_raw:
        law_id = item.get("law_id", "")
        reg_records.append(
            {
                "reg_node_id": item["id"],
                "name": item["title"],
                "law_id": law_id,
                "reg_date_key": law_id.replace("FOR-", ""),
                "korttittel": item.get("korttittel", ""),
                "url": item.get("url", ""),
                "file_path": item.get("file_path", ""),
                "file_size": item.get("file_size", 0),
                "dato": item.get("dato", ""),
            }
        )
    graph.add_nodes(pd.DataFrame(reg_records), "Regulation", "reg_node_id", "name")

    # Phase 5: GOVERNED_BY edges
    gov_law = [
        {"src": item["id"], "tgt": d.strip()}
        for item in laws_raw
        for d in item.get("department", "").split(", ")
        if d.strip()
    ]
    gov_reg = [
        {"src": item["id"], "tgt": d.strip()}
        for item in regs_raw
        for d in item.get("department", "").split(", ")
        if d.strip()
    ]
    if gov_law:
        graph.add_connections(pd.DataFrame(gov_law), "GOVERNED_BY", "Law", "src", "Department", "tgt")
    if gov_reg:
        graph.add_connections(pd.DataFrame(gov_reg), "GOVERNED_BY", "Regulation", "src", "Department", "tgt")

    # Phase 6: CourtDecision nodes
    case_records = []
    for court_level, items in [("hoyesterett", hr_raw), ("lagmannsrett", lr_raw)]:
        for item in items:
            case_key = item["id"].split("/")[-1].upper()
            dato = item.get("dato", "")
            year = int(dato[:4]) if dato and len(dato) >= 4 and dato[:4].isdigit() else 0
            case_records.append(
                {
                    "case_node_id": item["id"],
                    "name": item["title"],
                    "case_key": case_key,
                    "dato": dato,
                    "year": year,
                    "decision_type": item.get("decision_type", ""),
                    "court_level": court_level,
                    "section": item.get("section", ""),
                    "instans": item.get("instans", ""),
                    "sammendrag": item.get("sammendrag", ""),
                    "url": item.get("url", ""),
                    "file_path": item.get("file_path", ""),
                    "saksgang_raw": blg.strip_html(item.get("saksgang", "")),
                }
            )
    graph.add_nodes(pd.DataFrame(case_records), "CourtDecision", "case_node_id", "name")

    # Phase 7: Keyword nodes + HAS_KEYWORD edges
    keyword_set: dict[str, str] = {}
    kw_edges = []
    for item in all_cases:
        for kw in item.get("keywords", []):
            kw_stripped = kw.strip()
            if kw_stripped:
                if kw_stripped not in keyword_set:
                    keyword_set[kw_stripped] = kw_stripped
                kw_edges.append({"src": item["id"], "tgt": kw_stripped})
    graph.add_nodes(pd.DataFrame([{"kw_id": k, "name": v} for k, v in keyword_set.items()]), "Keyword", "kw_id", "name")
    if kw_edges:
        graph.add_connections(pd.DataFrame(kw_edges), "HAS_KEYWORD", "CourtDecision", "src", "Keyword", "tgt")

    # Phase 8: Judge nodes + JUDGED_BY edges
    judge_names: set[str] = set()
    judge_edges = []
    for item in all_cases:
        for j in item.get("dommere", []):
            name = j.strip().rstrip(".")
            if name:
                judge_names.add(name)
                judge_edges.append({"src": item["id"], "tgt": name})
    graph.add_nodes(pd.DataFrame([{"judge_id": n, "name": n} for n in judge_names]), "Judge", "judge_id", "name")
    if judge_edges:
        graph.add_connections(pd.DataFrame(judge_edges), "JUDGED_BY", "CourtDecision", "src", "Judge", "tgt")

    # Phase 9: Representative nodes + REPRESENTED_BY edges
    REP_PARENS_RE = re.compile(r"\(([^)]*)\)")
    REP_TITLES = ("advokat", "statsadvokat", "førstestatsadvokat", "regjeringsadvokaten", "kommuneadvokaten")

    def parse_representatives(parter_str):
        results = []
        for m in REP_PARENS_RE.finditer(parter_str):
            inner = m.group(1).strip()
            inner = (
                re.sub(r"^[^/]*/", "", inner)
                if inner.startswith("Regjeringsadvokaten v/") or inner.startswith("Kommuneadvokaten v/")
                else inner
            )
            lower = inner.lower()
            for title in REP_TITLES:
                if lower.startswith(title):
                    name = inner[len(title) :].strip()
                    name = re.sub(r"\s*–\s*til prøve$", "", name)
                    if name:
                        results.append((title, name))
                    break
        return results

    rep_meta: dict[str, str] = {}
    rep_edges = []
    for item in all_cases:
        parter_str = item.get("parter", "")
        if not parter_str:
            continue
        parts = parter_str.split(" mot ", 1)
        for side_idx, part in enumerate(parts):
            side = "saksoker" if side_idx == 0 else "saksokt"
            for role, name in parse_representatives(part):
                rep_meta[name] = role
                rep_edges.append({"src": item["id"], "tgt": name, "side": side})
    graph.add_nodes(
        pd.DataFrame([{"rep_id": n, "name": n, "most_recent_role": r} for n, r in rep_meta.items()]),
        "Representative",
        "rep_id",
        "name",
    )
    if rep_edges:
        graph.add_connections(
            pd.DataFrame(rep_edges), "REPRESENTED_BY", "CourtDecision", "src", "Representative", "tgt", columns=["side"]
        )

    # Phase 10: LawSection nodes + citation edges
    law_date_to_id = {item.get("law_id", "").replace("LOV-", ""): item["id"] for item in laws_raw if item.get("law_id")}
    reg_date_to_id = {item.get("law_id", "").replace("FOR-", ""): item["id"] for item in regs_raw if item.get("law_id")}

    section_map: dict[str, dict] = {}
    cites_edges = []
    cites_law_direct = []
    cites_reg_direct = []
    REF_TAG_RE = re.compile(r'data-id="([^"]+)"[^>]*>([^<]*)</a>')

    for item in all_cases:
        case_id = item["id"]
        html = item.get("henvisninger_i_teksten", "")
        if not html:
            continue
        seen: set[str] = set()
        for m in REF_TAG_RE.finditer(html):
            rid, display = m.group(1), m.group(2).strip()
            if rid in seen:
                continue
            seen.add(rid)
            parts = rid.split("/")
            ref_type = parts[0]
            if ref_type not in ("lov", "forskrift"):
                continue
            if len(parts) >= 3:
                ldk = parts[1]
                section_part = "/".join(parts[2:])
                if rid not in section_map:
                    section_map[rid] = {
                        "section_id": rid,
                        "name": display or rid,
                        "law_date_key": ldk,
                        "section": section_part,
                        "ref_type": ref_type,
                    }
                cites_edges.append({"src": case_id, "tgt": rid})
            elif len(parts) == 2:
                dk = parts[1]
                if ref_type == "lov" and dk in law_date_to_id:
                    cites_law_direct.append({"src": case_id, "tgt": law_date_to_id[dk]})
                elif ref_type == "forskrift" and dk in reg_date_to_id:
                    cites_reg_direct.append({"src": case_id, "tgt": reg_date_to_id[dk]})

    if section_map:
        graph.add_nodes(pd.DataFrame(list(section_map.values())), "LawSection", "section_id", "name")
    if cites_edges:
        graph.add_connections(pd.DataFrame(cites_edges), "CITES", "CourtDecision", "src", "LawSection", "tgt")
    if cites_law_direct:
        graph.add_connections(pd.DataFrame(cites_law_direct), "CITES_DIRECTLY", "CourtDecision", "src", "Law", "tgt")
    if cites_reg_direct:
        graph.add_connections(
            pd.DataFrame(cites_reg_direct), "CITES_DIRECTLY", "CourtDecision", "src", "Regulation", "tgt"
        )

    # SECTION_OF edges
    section_of_law = [
        {"src": sid, "tgt": law_date_to_id[s["law_date_key"]]}
        for sid, s in section_map.items()
        if s["ref_type"] == "lov" and s["law_date_key"] in law_date_to_id
    ]
    section_of_reg = [
        {"src": sid, "tgt": reg_date_to_id[s["law_date_key"]]}
        for sid, s in section_map.items()
        if s["ref_type"] == "forskrift" and s["law_date_key"] in reg_date_to_id
    ]
    if section_of_law:
        graph.add_connections(pd.DataFrame(section_of_law), "SECTION_OF", "LawSection", "src", "Law", "tgt")
    if section_of_reg:
        graph.add_connections(pd.DataFrame(section_of_reg), "SECTION_OF", "LawSection", "src", "Regulation", "tgt")

    # Phase 11: CASE_PROGRESSION edges
    case_key_to_id = {item["id"].split("/")[-1].upper(): item["id"] for item in all_cases}
    progression_edges = []
    seen_prog: set[tuple[str, str]] = set()
    CASE_ID_RE = blg.CASE_ID_RE
    for item in all_cases:
        sg = blg.strip_html(item.get("saksgang", ""))
        found = CASE_ID_RE.findall(sg)
        unique_ordered = []
        seen_sg: set[str] = set()
        for cid in found:
            cu = cid.upper()
            if cu not in seen_sg:
                seen_sg.add(cu)
                unique_ordered.append(cu)
        for i in range(len(unique_ordered) - 1):
            sk, tk = unique_ordered[i], unique_ordered[i + 1]
            if sk in case_key_to_id and tk in case_key_to_id:
                edge = (case_key_to_id[sk], case_key_to_id[tk])
                if edge not in seen_prog:
                    seen_prog.add(edge)
                    progression_edges.append({"src": edge[0], "tgt": edge[1]})
    if progression_edges:
        graph.add_connections(
            pd.DataFrame(progression_edges), "CASE_PROGRESSION", "CourtDecision", "src", "CourtDecision", "tgt"
        )

    # Phase 12: Create indexes
    graph.create_index("CourtDecision", "court_level")
    graph.create_index("CourtDecision", "decision_type")
    graph.create_index("CourtDecision", "section")
    graph.create_index("CourtDecision", "case_key")
    graph.create_range_index("CourtDecision", "year")
    graph.create_index("LawSection", "law_date_key")
    graph.create_index("LawSection", "ref_type")
    graph.create_index("Law", "law_date_key")
    graph.create_index("Law", "korttittel")
    graph.create_index("Regulation", "reg_date_key")

    return graph


# ═══════════════════════════════════════════════════════════════════
# Cypher benchmark queries
# ═══════════════════════════════════════════════════════════════════

CYPHER_QUERIES: list[tuple[str, str | None]] = [
    # ── Real-World Queries (from MCP conversations) ───────────────
    (
        "cypher_real_law_contains",
        "MATCH (l:Law) WHERE l.korttittel CONTAINS 'folketrygd' RETURN l.name, l.korttittel, l.law_date_key, l.file_path, l.url",
    ),
    (
        "cypher_real_law_compound_filter",
        "MATCH (l:Law) WHERE l.name CONTAINS 'folketrygd' AND NOT l.name CONTAINS 'endring' RETURN l.name, l.korttittel, l.law_date_key, l.file_path, l.url",
    ),
    (
        "cypher_real_sections_by_law",
        "MATCH (s:LawSection)-[:SECTION_OF]->(l:Law {korttittel: 'Folketrygdloven'}) WHERE s.section IN ['§ 8-9', '§ 8-2', '§ 8-15', '§ 8-47'] RETURN s.section, s.name, s.section_id ORDER BY s.section",
    ),
    (
        "cypher_real_sections_contains",
        "MATCH (s:LawSection)-[:SECTION_OF]->(l:Law {korttittel: 'Folketrygdloven'}) WHERE s.section CONTAINS '8-9' OR s.section CONTAINS '8-2' OR s.section CONTAINS '8-15' OR s.section CONTAINS '8-47' RETURN s.section, s.name, s.section_id ORDER BY s.section",
    ),
    (
        "cypher_real_decisions_citing_section",
        "MATCH (c:CourtDecision)-[:CITES]->(s:LawSection {section_id: 'lov/1997-02-28-19/§8-9'}) WHERE c.sammendrag IS NOT NULL RETURN c.name, c.dato, c.court_level, c.url ORDER BY c.dato DESC LIMIT 10",
    ),
    ("cypher_real_decision_file_path", "MATCH (c:CourtDecision {name: 'LG-2023-94623'}) RETURN c.file_path"),
    # ── Simple Lookups ─────────────────────────────────────────────
    (
        "cypher_lookup_law_by_korttittel",
        "MATCH (l:Law {korttittel: 'Folketrygdloven'}) RETURN l.name, l.korttittel, l.law_date_key",
    ),
    ("cypher_lookup_first_decision", "MATCH (d:CourtDecision) RETURN d LIMIT 1"),
    ("cypher_lookup_count_decisions", "MATCH (d:CourtDecision) RETURN count(d)"),
    ("cypher_lookup_count_all_nodes", "MATCH (n) RETURN count(n)"),
    ("cypher_lookup_node_by_id", "MATCH (l:Law) RETURN id(l), l.title LIMIT 5"),
    (
        "cypher_lookup_decision_by_case_key",
        "MATCH (d:CourtDecision {case_key: 'HR-2020-1167-A'}) RETURN d.name, d.dato, d.court_level",
    ),
    # ── WHERE Operators & Predicates ───────────────────────────────
    (
        "cypher_where_and",
        "MATCH (d:CourtDecision) WHERE d.court_level = 'hoyesterett' AND d.year >= 2020 RETURN count(d)",
    ),
    (
        "cypher_where_or",
        "MATCH (d:CourtDecision) WHERE d.court_level = 'hoyesterett' OR d.year >= 2024 RETURN count(d)",
    ),
    (
        "cypher_where_or_same_prop",
        "MATCH (d:CourtDecision) WHERE d.court_level = 'hoyesterett' OR d.court_level = 'lagmannsrett' RETURN d.court_level, count(d)",
    ),
    ("cypher_where_not", "MATCH (d:CourtDecision) WHERE NOT d.court_level = 'hoyesterett' RETURN count(d)"),
    (
        "cypher_where_range",
        "MATCH (d:CourtDecision) WHERE d.year >= 2020 AND d.year <= 2025 RETURN d.name, d.dato ORDER BY d.dato DESC LIMIT 20",
    ),
    ("cypher_where_contains", "MATCH (l:Law) WHERE l.name CONTAINS 'straffelov' RETURN l.name, l.korttittel"),
    ("cypher_where_starts_with", "MATCH (d:CourtDecision) WHERE d.name STARTS WITH 'HR-2024' RETURN d.name LIMIT 20"),
    ("cypher_where_ends_with", "MATCH (d:CourtDecision) WHERE d.name ENDS WITH '-A' RETURN count(d)"),
    ("cypher_where_regex", "MATCH (d:CourtDecision) WHERE d.name =~ 'HR-202[0-9]-.*' RETURN count(d)"),
    (
        "cypher_where_in_list",
        "MATCH (d:CourtDecision) WHERE d.court_level IN ['hoyesterett', 'lagmannsrett'] RETURN d.court_level, count(d) AS cnt",
    ),
    ("cypher_where_is_null", "MATCH (d:CourtDecision) WHERE d.sammendrag IS NULL RETURN count(d)"),
    ("cypher_where_is_not_null", "MATCH (d:CourtDecision) WHERE d.sammendrag IS NOT NULL RETURN count(d)"),
    (
        "cypher_where_exists_pattern",
        "MATCH (l:Law) WHERE EXISTS { MATCH (d:CourtDecision)-[:CITES_DIRECTLY]->(l) } RETURN l.title LIMIT 20",
    ),
    # ── Relationship Traversals ────────────────────────────────────
    (
        "cypher_traverse_law_sections",
        "MATCH (s:LawSection)-[:SECTION_OF]->(l:Law {korttittel: 'Folketrygdloven'}) RETURN count(s)",
    ),
    (
        "cypher_traverse_decisions_citing_law",
        "MATCH (d:CourtDecision)-[:CITES_DIRECTLY]->(l:Law {korttittel: 'Straffeloven'}) RETURN d.name, d.dato ORDER BY d.dato DESC LIMIT 10",
    ),
    (
        "cypher_traverse_law_to_dept",
        "MATCH (l:Law {korttittel: 'Folketrygdloven'})-[:GOVERNED_BY]->(dept:Department) RETURN dept.title",
    ),
    (
        "cypher_traverse_2hop_section_citations",
        "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law) WHERE l.korttittel = 'Folketrygdloven' RETURN s.section, count(d) AS citations ORDER BY citations DESC LIMIT 20",
    ),
    (
        "cypher_traverse_decision_judges",
        "MATCH (d:CourtDecision {case_key: 'HR-2020-1167-A'})-[:JUDGED_BY]->(j:Judge) RETURN j.title",
    ),
    (
        "cypher_traverse_decision_reps",
        "MATCH (d:CourtDecision {case_key: 'HR-2020-1167-A'})-[:REPRESENTED_BY]->(r:Representative) RETURN r.title",
    ),
    (
        "cypher_traverse_case_progression",
        "MATCH (d:CourtDecision {name: 'HR-2020-1167-A'})-[:CASE_PROGRESSION*1..3]-(related:CourtDecision) RETURN related.name, related.dato, related.court_level",
    ),
    (
        "cypher_traverse_inline_props",
        "MATCH (l:Law {korttittel: 'Folketrygdloven'})-[:GOVERNED_BY]->(d:Department) RETURN l.name, d.title",
    ),
    # ── Variable-Length Paths & Shortest Path ──────────────────────
    (
        "cypher_varlen_path",
        "MATCH (l:Law {korttittel: 'Folketrygdloven'})<-[:SECTION_OF*1..2]-(x) RETURN labels(x)[0] AS type, count(x) AS cnt",
    ),
    (
        "cypher_shortest_path",
        "MATCH p = shortestPath((a:Law {korttittel: 'Folketrygdloven'})-[*..4]-(b:Law {korttittel: 'Straffeloven'})) RETURN length(p), [n IN nodes(p) | n.title] AS path",
    ),
    # ── OPTIONAL MATCH ─────────────────────────────────────────────
    (
        "cypher_optional_match",
        "MATCH (l:Law) OPTIONAL MATCH (d:CourtDecision)-[:CITES_DIRECTLY]->(l) RETURN l.korttittel, count(d) AS citations ORDER BY citations DESC LIMIT 15",
    ),
    (
        "cypher_optional_match_where",
        "MATCH (l:Law {korttittel: 'Folketrygdloven'}) OPTIONAL MATCH (d:CourtDecision)-[:CITES_DIRECTLY]->(l) WHERE d.year >= 2020 RETURN l.korttittel, count(d)",
    ),
    # ── Aggregations ───────────────────────────────────────────────
    ("cypher_agg_count", "MATCH (d:CourtDecision) RETURN count(d) AS total"),
    (
        "cypher_agg_group_court",
        "MATCH (d:CourtDecision) RETURN d.court_level, count(d) AS decisions ORDER BY decisions DESC",
    ),
    (
        "cypher_agg_group_year",
        "MATCH (d:CourtDecision) WHERE d.year >= 2015 RETURN d.year, count(d) AS cnt ORDER BY d.year DESC",
    ),
    (
        "cypher_agg_most_cited_laws",
        "MATCH (d:CourtDecision)-[:CITES_DIRECTLY]->(l:Law) RETURN l.korttittel, count(d) AS citations ORDER BY citations DESC LIMIT 10",
    ),
    (
        "cypher_agg_most_cited_sections",
        "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law) RETURN l.korttittel, s.section, count(d) AS citations ORDER BY citations DESC LIMIT 20",
    ),
    (
        "cypher_agg_collect",
        "MATCH (d:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword) WHERE d.case_key = 'HR-2020-1167-A' RETURN collect(k.title) AS keywords",
    ),
    (
        "cypher_agg_count_distinct",
        "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law) RETURN d.court_level, count(DISTINCT l) AS unique_laws ORDER BY unique_laws DESC",
    ),
    (
        "cypher_agg_top_judges",
        "MATCH (d:CourtDecision)-[:JUDGED_BY]->(j:Judge) RETURN j.title, count(d) AS cases ORDER BY cases DESC LIMIT 10",
    ),
    (
        "cypher_agg_top_keywords",
        "MATCH (d:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword) RETURN k.title, count(d) AS cases ORDER BY cases DESC LIMIT 10",
    ),
    # ── DISTINCT ───────────────────────────────────────────────────
    ("cypher_distinct", "MATCH (d:CourtDecision) RETURN DISTINCT d.court_level ORDER BY d.court_level"),
    # ── ORDER BY + SKIP + LIMIT ───────────────────────────────────
    (
        "cypher_order_limit",
        "MATCH (d:CourtDecision) WHERE d.year >= 2020 RETURN d.name, d.dato ORDER BY d.dato DESC LIMIT 10",
    ),
    ("cypher_skip_limit", "MATCH (l:Law) RETURN l.korttittel ORDER BY l.korttittel SKIP 10 LIMIT 10"),
    # ── WITH Clause ────────────────────────────────────────────────
    (
        "cypher_with_chain",
        "MATCH (d:CourtDecision)-[:CITES_DIRECTLY]->(l:Law) WITH l, count(d) AS cites WHERE cites >= 5 RETURN l.korttittel, cites ORDER BY cites DESC LIMIT 10",
    ),
    (
        "cypher_with_aggregation",
        "MATCH (d:CourtDecision)-[:JUDGED_BY]->(j:Judge) WITH j.title AS judge, count(d) AS cases WHERE cases >= 20 RETURN judge, cases ORDER BY cases DESC",
    ),
    # ── HAVING ─────────────────────────────────────────────────────
    (
        "cypher_having",
        "MATCH (d:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword) RETURN k.title, count(d) AS cnt HAVING cnt >= 50 ORDER BY cnt DESC",
    ),
    # ── UNWIND ─────────────────────────────────────────────────────
    (
        "cypher_unwind",
        "UNWIND ['Folketrygdloven', 'Straffeloven', 'Arbeidsmiljøloven'] AS kt MATCH (l:Law {korttittel: kt}) RETURN l.korttittel, l.name",
    ),
    (
        "cypher_unwind_range",
        "UNWIND range(2015, 2025) AS yr MATCH (d:CourtDecision) WHERE d.year = yr AND d.court_level = 'hoyesterett' RETURN yr, count(d) AS decisions ORDER BY yr",
    ),
    # ── UNION ──────────────────────────────────────────────────────
    (
        "cypher_union_all",
        "MATCH (l:Law) RETURN l.korttittel AS name, 'law' AS type LIMIT 5 UNION ALL MATCH (r:Regulation) RETURN r.korttittel AS name, 'regulation' AS type LIMIT 5",
    ),
    # ── String Functions ───────────────────────────────────────────
    (
        "cypher_fn_toupper_tolower",
        "MATCH (l:Law) RETURN toUpper(l.korttittel) AS upper, toLower(l.korttittel) AS lower LIMIT 5",
    ),
    ("cypher_fn_substring", "MATCH (d:CourtDecision) RETURN substring(d.name, 0, 7) AS prefix LIMIT 10"),
    (
        "cypher_fn_split",
        "MATCH (d:CourtDecision) WHERE d.court_level = 'hoyesterett' RETURN split(d.name, '-')[0] AS prefix, count(d) AS cnt ORDER BY cnt DESC LIMIT 5",
    ),
    ("cypher_fn_replace", "MATCH (l:Law) RETURN replace(l.korttittel, ' ', '_') AS slug LIMIT 10"),
    ("cypher_fn_concat", "MATCH (d:CourtDecision) RETURN d.name || ' (' || d.court_level || ')' AS label LIMIT 10"),
    # ── Type Conversion ────────────────────────────────────────────
    (
        "cypher_fn_tostring_tointeger",
        "MATCH (d:CourtDecision) WHERE d.year IS NOT NULL RETURN toString(d.year) AS yr_str, toInteger('2020') AS parsed LIMIT 5",
    ),
    # ── Introspection Functions ────────────────────────────────────
    (
        "cypher_fn_labels_type",
        "MATCH (d:CourtDecision)-[r:CITES]->(s:LawSection) RETURN labels(d)[0] AS dl, type(r) AS rt, labels(s)[0] AS sl LIMIT 5",
    ),
    ("cypher_fn_keys", "MATCH (l:Law {korttittel: 'Folketrygdloven'}) RETURN keys(l) AS props"),
    (
        "cypher_fn_size",
        "MATCH (l:Law {korttittel: 'Folketrygdloven'})<-[:SECTION_OF]-(s:LawSection) WITH collect(s.section) AS sections RETURN size(sections) AS count",
    ),
    # ── CASE Expression ────────────────────────────────────────────
    (
        "cypher_case",
        "MATCH (d:CourtDecision) RETURN CASE WHEN d.year >= 2020 THEN 'recent' WHEN d.year >= 2010 THEN 'modern' ELSE 'older' END AS era, count(d) AS cnt ORDER BY cnt DESC",
    ),
    # ── coalesce ───────────────────────────────────────────────────
    (
        "cypher_coalesce",
        "MATCH (d:CourtDecision) RETURN d.name, coalesce(d.sammendrag, 'no summary') AS summary LIMIT 10",
    ),
    # ── List Comprehension ───────────────────────────────────────────
    (
        "cypher_list_comprehension",
        "MATCH (d:CourtDecision)-[:HAS_KEYWORD]->(k:Keyword) WHERE d.case_key = 'HR-2020-1167-A' WITH collect(k.title) AS kws RETURN [x IN kws WHERE x STARTS WITH 'S'] AS s_keywords",
    ),
    # ── Map Projection ─────────────────────────────────────────────
    (
        "cypher_map_projection",
        "MATCH (l:Law {korttittel: 'Folketrygdloven'}) RETURN l {.title, .korttittel, .law_date_key} AS law_map",
    ),
    (
        "cypher_map_literal",
        "MATCH (d:CourtDecision) WHERE d.year >= 2024 RETURN {name: d.name, court: d.court_level, year: d.year} AS info LIMIT 10",
    ),
    # ── Date Functions & Arithmetic ────────────────────────────────
    ("cypher_date_parse", "RETURN date('2020-06-15') AS d, datetime('2020-06-15') AS dt"),
    ("cypher_date_arithmetic", "WITH date('2020-01-01') AS d RETURN d + 30 AS plus30, d - 10 AS minus10"),
    (
        "cypher_date_diff",
        "WITH date('2020-01-01') AS a, date('2025-06-15') AS b RETURN date_diff(a, b) AS days_between",
    ),
    # ── EXISTS Subquery ────────────────────────────────────────────
    (
        "cypher_exists_subquery",
        "MATCH (l:Law) WHERE EXISTS { MATCH (d:CourtDecision)-[:CITES_DIRECTLY]->(l) WHERE d.court_level = 'hoyesterett' } RETURN l.korttittel LIMIT 20",
    ),
    # ── EXPLAIN / PROFILE ──────────────────────────────────────────
    (
        "cypher_explain",
        "EXPLAIN MATCH (d:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law) RETURN l.korttittel, count(d) ORDER BY count(d) DESC LIMIT 10",
    ),
    (
        "cypher_profile",
        "PROFILE MATCH (d:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law) RETURN l.korttittel, count(d) ORDER BY count(d) DESC LIMIT 10",
    ),
    # ── Full Scans & Heavy Operations ──────────────────────────────
    ("cypher_scan_filter_count", "MATCH (d:CourtDecision) WHERE d.year > 2020 RETURN count(d)"),
    ("cypher_scan_large_result", "MATCH (d:CourtDecision) RETURN d.name, d.court_level, d.year, d.dato LIMIT 5000"),
    ("cypher_scan_edge_type_counts", "MATCH ()-[r]->() RETURN type(r), count(*) ORDER BY count(*) DESC"),
    # ── Complex Analytical Queries ─────────────────────────────────
    (
        "cypher_complex_dept_citations",
        "MATCH (d:CourtDecision)-[:CITES_DIRECTLY]->(l:Law)-[:GOVERNED_BY]->(dept:Department) RETURN dept.title, count(d) AS citations ORDER BY citations DESC LIMIT 10",
    ),
    (
        "cypher_complex_2hop_section_law",
        "MATCH (d:CourtDecision)-[:CITES]->(s:LawSection)-[:SECTION_OF]->(l:Law) RETURN l.korttittel, count(DISTINCT s) AS sections, count(d) AS citations ORDER BY citations DESC LIMIT 10",
    ),
    (
        "cypher_complex_judge_court",
        "MATCH (d:CourtDecision)-[:JUDGED_BY]->(j:Judge) RETURN j.title, d.court_level, count(d) AS cases ORDER BY cases DESC LIMIT 20",
    ),
    (
        "cypher_complex_yearly_decisions",
        "MATCH (d:CourtDecision) WHERE d.year >= 2010 RETURN d.year, d.court_level, count(d) AS cnt ORDER BY d.year DESC, cnt DESC",
    ),
    ("cypher_complex_many_small", None),  # Special: 100× small queries — handled in runner
    # ── Graph Algorithms ───────────────────────────────────────────
    (
        "cypher_algo_pagerank",
        "CALL pagerank({connection_types: 'CITES'}) YIELD node, score RETURN node.title, score ORDER BY score DESC LIMIT 10",
    ),
    (
        "cypher_algo_degree",
        "CALL degree({connection_types: 'CITES'}) YIELD node, score RETURN node.title, score ORDER BY score DESC LIMIT 10",
    ),
    (
        "cypher_algo_betweenness",
        "CALL betweenness({node_type: 'Law', sample_size: 50}) YIELD node, score RETURN node.title, score ORDER BY score DESC LIMIT 10",
    ),
    (
        "cypher_algo_closeness",
        "CALL closeness({node_type: 'Law', sample_size: 100}) YIELD node, score RETURN node.title, score ORDER BY score DESC LIMIT 10",
    ),
    (
        "cypher_algo_louvain",
        "CALL louvain({connection_types: 'GOVERNED_BY'}) YIELD node, community RETURN community, collect(node.title) AS members ORDER BY size(members) DESC LIMIT 5",
    ),
    (
        "cypher_algo_label_prop",
        "CALL label_propagation({connection_types: 'GOVERNED_BY'}) YIELD node, community RETURN community, count(*) AS size ORDER BY size DESC LIMIT 5",
    ),
    (
        "cypher_algo_components",
        "CALL connected_components() YIELD node, component RETURN component, count(*) AS size ORDER BY size DESC LIMIT 5",
    ),
    # ── Mutations (on graph copy) ──────────────────────────────────
    ("cypher_mutation_create", None),
    ("cypher_mutation_set", None),
    ("cypher_mutation_merge", None),
    ("cypher_mutation_delete", None),
]

MUTATION_CYPHER = {
    "cypher_mutation_create": "CREATE (n:TestNode {name: 'benchmark', value: 42})",
    "cypher_mutation_set": "MATCH (l:Law {korttittel: 'Folketrygdloven'}) SET l._bench_test = 'hello'",
    "cypher_mutation_merge": "MERGE (n:TestNode {name: 'merged'}) ON CREATE SET n.created = true ON MATCH SET n.updated = true",
    "cypher_mutation_delete": "MATCH (n:TestNode) DELETE n",
}


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
    B = []

    def add(name, fn, iters=ITERATIONS, wu=WARMUP):
        B.append((name, fn, iters, wu))

    # ── Introspection ──────────────────────────────────────────────
    add("fluent_schema", lambda: graph.schema())
    add("fluent_describe", lambda: graph.describe())
    add("fluent_len", lambda: len(graph))
    add("fluent_node_type_counts", lambda: graph.node_type_counts())
    add("fluent_properties_law", lambda: graph.properties("Law"))
    add("fluent_properties_decision", lambda: graph.properties("CourtDecision"))
    add("fluent_connection_types", lambda: graph.connection_types())

    # ── Select ─────────────────────────────────────────────────────
    add("fluent_select_law", lambda: graph.select("Law"))
    add("fluent_select_decision", lambda: graph.select("CourtDecision"))
    add("fluent_select_section", lambda: graph.select("LawSection"))
    add("fluent_select_sort", lambda: graph.select("Law", sort="korttittel"))
    add("fluent_select_limit", lambda: graph.select("CourtDecision", limit=10))

    # ── Where ──────────────────────────────────────────────────────
    add("fluent_where_eq", lambda: graph.select("Law").where({"korttittel": "Folketrygdloven"}))
    add("fluent_where_contains", lambda: graph.select("Law").where({"name": {"contains": "straffelov"}}))
    add("fluent_where_starts_with", lambda: graph.select("CourtDecision").where({"name": {"starts_with": "HR-2024"}}))
    add("fluent_where_ends_with", lambda: graph.select("CourtDecision").where({"name": {"ends_with": "-A"}}))
    add("fluent_where_gt", lambda: graph.select("CourtDecision").where({"year": {">": 2020}}))
    add("fluent_where_range", lambda: graph.select("CourtDecision").where({"year": {">=": 2020, "<=": 2025}}))
    add(
        "fluent_where_in",
        lambda: graph.select("CourtDecision").where({"court_level": {"in": ["hoyesterett", "lagmannsrett"]}}),
    )
    add("fluent_where_is_null", lambda: graph.select("CourtDecision").where({"sammendrag": {"is_null": True}}))
    add("fluent_where_is_not_null", lambda: graph.select("CourtDecision").where({"sammendrag": {"is_not_null": True}}))
    add(
        "fluent_where_any",
        lambda: graph.select("CourtDecision").where_any([{"court_level": "hoyesterett"}, {"year": 2024}]),
    )
    add("fluent_where_connected", lambda: graph.select("Law").where_connected("CITES_DIRECTLY"))

    # ── Traverse ───────────────────────────────────────────────────
    add(
        "fluent_traverse_law_sections",
        lambda: (
            graph.select("Law").where({"korttittel": "Folketrygdloven"}).traverse("SECTION_OF", direction="incoming")
        ),
    )
    add("fluent_traverse_all_governed", lambda: graph.select("Law").traverse("GOVERNED_BY"))
    add("fluent_traverse_large_cites", lambda: graph.select("CourtDecision").traverse("CITES"))
    add("fluent_traverse_decision_judges", lambda: graph.select("CourtDecision", limit=100).traverse("JUDGED_BY"))
    add(
        "fluent_traverse_multihop",
        lambda: graph.select("CourtDecision", limit=50).traverse("CITES").traverse("SECTION_OF"),
    )

    # ── Collect / Output ───────────────────────────────────────────
    add("fluent_collect_single", lambda: graph.select("Law").where({"korttittel": "Folketrygdloven"}).collect())
    add("fluent_collect_all_laws", lambda: graph.select("Law").collect())
    add("fluent_collect_1000", lambda: graph.select("CourtDecision", limit=1000).collect())
    add("fluent_collect_all_decisions", lambda: graph.select("CourtDecision").collect())
    add("fluent_to_df", lambda: graph.select("Law").to_df())
    add("fluent_to_df_large", lambda: graph.select("CourtDecision").to_df())
    add("fluent_ids", lambda: graph.select("Law").ids())
    add("fluent_titles", lambda: graph.select("Law").titles())
    add("fluent_sample_small", lambda: graph.sample("Law", 5))
    add("fluent_sample_large", lambda: graph.sample("CourtDecision", 100))
    add(
        "fluent_show",
        lambda: (
            graph.select("Law").where({"korttittel": "Folketrygdloven"}).show(["title", "korttittel", "law_date_key"])
        ),
    )

    # ── Statistics & Aggregation ───────────────────────────────────
    add("fluent_statistics", lambda: graph.select("CourtDecision").statistics("year"))
    add("fluent_statistics_groupby", lambda: graph.select("CourtDecision").statistics("year", group_by="court_level"))
    add("fluent_count", lambda: graph.select("CourtDecision").count())
    add("fluent_count_groupby", lambda: graph.select("CourtDecision").count(group_by="court_level"))
    add(
        "fluent_unique_values",
        lambda: graph.select("CourtDecision").unique_values("court_level", group_by_parent=False),
    )

    # ── Graph Algorithms ───────────────────────────────────────────
    add("fluent_pagerank", lambda: graph.select("Law").pagerank(top_k=10))
    add("fluent_degree_centrality", lambda: graph.select("Law").degree_centrality(top_k=10))
    add("fluent_betweenness", lambda: graph.select("Law").betweenness_centrality(top_k=10, sample_size=50))
    add("fluent_closeness", lambda: graph.select("Law").closeness_centrality(top_k=10, sample_size=100))
    add("fluent_louvain", lambda: graph.select("Law").louvain_communities())
    add("fluent_label_propagation", lambda: graph.select("Law").label_propagation())
    add("fluent_connected_components", lambda: graph.select("Law").connected_components())

    # ── Path Finding ───────────────────────────────────────────────
    ft_ids = graph.select("Law").where({"korttittel": "Folketrygdloven"}).ids()
    str_ids = graph.select("Law").where({"korttittel": "Straffeloven"}).ids()
    if ft_ids and str_ids:
        ft_id, str_id = ft_ids[0], str_ids[0]
        add("fluent_shortest_path", lambda: graph.shortest_path("Law", ft_id, "Law", str_id))
        add("fluent_shortest_path_length", lambda: graph.shortest_path_length("Law", ft_id, "Law", str_id))
        add("fluent_are_connected", lambda: graph.are_connected("Law", ft_id, "Law", str_id))

    # ── Set Operations ─────────────────────────────────────────────
    hoyesterett = graph.select("CourtDecision").where({"court_level": "hoyesterett"})
    recent = graph.select("CourtDecision").where({"year": {">": 2020}})
    add("fluent_union", lambda: hoyesterett.union(recent))
    add("fluent_intersection", lambda: hoyesterett.intersection(recent))
    add("fluent_difference", lambda: hoyesterett.difference(recent))
    add("fluent_symmetric_difference", lambda: hoyesterett.symmetric_difference(recent))

    # ── Chained Pipelines ──────────────────────────────────────────
    add(
        "fluent_pipeline_where_traverse_collect",
        lambda: (
            graph.select("Law")
            .where({"korttittel": "Folketrygdloven"})
            .traverse("SECTION_OF", direction="incoming")
            .collect()
        ),
    )
    add(
        "fluent_pipeline_multihop_collect",
        lambda: graph.select("CourtDecision", limit=50).traverse("CITES").traverse("SECTION_OF").collect(),
    )
    add("fluent_pipeline_large_collect", lambda: graph.select("CourtDecision").traverse("HAS_KEYWORD").collect())

    # ── Mutations (deep-copy graph each time) ──────────────────────
    def _bench_update():
        g2 = graph.copy()
        g2.select("Law").where({"korttittel": "Folketrygdloven"}).update({"_bench_test": "value"})

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

    print(f"KGLite v{version} — Norwegian Legal benchmark")
    print()

    # ── Build ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        graph = build_legal_graph()
    build_ms = (time.perf_counter() - t0) * 1000
    results["build_graph"] = round(build_ms, 1)
    s = graph.schema()
    print(f"  Build:  {build_ms:>8.0f} ms  ({s['node_count']} nodes, {s['edge_count']} edges)")

    # ── Save ───────────────────────────────────────────────────────
    save_ms = bench(lambda: graph.save(TEMP_KGL), iterations=3, warmup=0)
    results["save_kgl"] = round(save_ms, 1)
    size_mb = os.path.getsize(TEMP_KGL) / (1024 * 1024)
    print(f"  Save:   {save_ms:>8.0f} ms  ({size_mb:.1f} MB)")

    # ── Load ───────────────────────────────────────────────────────
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
            q = "MATCH (l:Law {korttittel: 'Folketrygdloven'}) RETURN l.title"
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
    print(f"  Build:         {results['build_graph']:>8.0f} ms")
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
    """Load existing benchmark_legal.csv → (benchmark_names, col_names, data)."""
    if not os.path.exists(CSV_OUT):
        return [], [], {}

    with open(CSV_OUT, newline="") as f:
        reader = csv.DictReader(f)
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
    """Determine column name: v0.5.82, v0.5.82_2, v0.5.82_3, ..."""
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
    """Append results as a new column in benchmark_legal.csv."""
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
    if not blg.DATA_SOURCE.is_dir():
        print(f"ERROR: Data source not found: {blg.DATA_SOURCE}")
        print("This benchmark requires the Norwegian legal JSON data.")
        sys.exit(1)

    version, results = run_benchmarks()
    save_to_csv(version, results)
