"""Shared fixtures for kglite test suite."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def empty_graph():
    """Empty graph for edge case testing."""
    return KnowledgeGraph()


@pytest.fixture
def small_graph():
    """Small graph: 3 Person nodes + 3 KNOWS edges.

    Persons: Alice (age=28, city=Oslo), Bob (age=35, city=Bergen), Charlie (age=42, city=Oslo)
    Edges: Alice->Bob, Bob->Charlie, Alice->Charlie
    """
    graph = KnowledgeGraph()

    people = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [28, 35, 42],
            "city": ["Oslo", "Bergen", "Oslo"],
        }
    )
    graph.add_nodes(people, "Person", "person_id", "name")

    edges = pd.DataFrame(
        {
            "from_id": [1, 2, 1],
            "to_id": [2, 3, 3],
            "since": [2020, 2019, 2021],
        }
    )
    graph.add_connections(edges, "KNOWS", "Person", "from_id", "Person", "to_id", columns=["since"])

    return graph


@pytest.fixture
def file_imports_graph():
    """Synthetic code-tree-shaped graph for testing File→File IMPORTS edges
    and the `affected_tests` Cypher procedure.

    Mimics the schema emitted by `kglite.code_tree.build()` without paying
    the tree-sitter parse cost — five Files (three source, two tests) and
    a small import web:

        src/a.py  ─┐
                   ├─►  src/util.py  ◄──  tests/test_util.py  (is_test=True)
        src/b.py  ─┘                      tests/test_a.py     (is_test=True, imports src/a.py)

    Both File→File IMPORTS edges (the 0.9.34 addition) and the existing
    `is_test` File property are populated.
    """
    graph = KnowledgeGraph()
    files = pd.DataFrame(
        {
            "path": [
                "src/a.py",
                "src/b.py",
                "src/util.py",
                "tests/test_util.py",
                "tests/test_a.py",
            ],
            "filename": ["a.py", "b.py", "util.py", "test_util.py", "test_a.py"],
            "is_test": [False, False, False, True, True],
        }
    )
    graph.add_nodes(files, "File", "path", "filename")

    imports = pd.DataFrame(
        {
            "source": [
                "src/a.py",
                "src/b.py",
                "tests/test_util.py",
                "tests/test_a.py",
            ],
            "target": [
                "src/util.py",
                "src/util.py",
                "src/util.py",
                "src/a.py",
            ],
            "import_count": [1, 1, 1, 2],
        }
    )
    graph.add_connections(
        imports,
        "IMPORTS",
        "File",
        "source",
        "File",
        "target",
        columns=["import_count"],
    )
    return graph


@pytest.fixture
def social_graph():
    """Medium graph: 20 Person + 5 Company nodes, KNOWS/WORKS_AT edges.

    Persons: Person_1..Person_20, age=21..40, city in [Oslo, Bergen, Stavanger, Trondheim]
             email is None for odd-numbered persons (nullable field)
    Companies: TechCorp, DataInc, CloudSoft, AILabs, DevHouse
    KNOWS: each person knows next 3 persons (with 'since' property)
    WORKS_AT: each person works at one company (with 'start_year' property)
    """
    graph = KnowledgeGraph()

    people = pd.DataFrame(
        {
            "person_id": list(range(1, 21)),
            "name": [f"Person_{i}" for i in range(1, 21)],
            "age": [20 + i for i in range(1, 21)],
            "city": (["Oslo"] * 5 + ["Bergen"] * 5 + ["Stavanger"] * 5 + ["Trondheim"] * 5),
            "salary": [50000 + i * 5000 for i in range(20)],
            "email": [f"person{i}@test.com" if i % 2 == 0 else None for i in range(1, 21)],
        }
    )
    graph.add_nodes(people, "Person", "person_id", "name")

    companies = pd.DataFrame(
        {
            "company_id": list(range(100, 105)),
            "name": ["TechCorp", "DataInc", "CloudSoft", "AILabs", "DevHouse"],
            "industry": ["Tech", "Data", "Cloud", "AI", "Software"],
        }
    )
    graph.add_nodes(companies, "Company", "company_id", "name")

    knows_edges = []
    for i in range(1, 21):
        for j in range(i + 1, min(i + 4, 21)):
            knows_edges.append({"from_id": i, "to_id": j, "since": 2015 + (i % 5)})
    knows_df = pd.DataFrame(knows_edges)
    graph.add_connections(knows_df, "KNOWS", "Person", "from_id", "Person", "to_id", columns=["since"])

    works_at = pd.DataFrame(
        {
            "person_id": list(range(1, 21)),
            "company_id": [100 + (i % 5) for i in range(20)],
            "start_year": [2018 + (i % 4) for i in range(20)],
        }
    )
    graph.add_connections(works_at, "WORKS_AT", "Person", "person_id", "Company", "company_id", columns=["start_year"])

    return graph


@pytest.fixture
def large_schema_graph():
    """Graph with >15 node types for compact inventory testing.

    Creates 20 types with varying node counts and property counts:
    - Types 0-4:  Large (>1000 nodes), few properties
    - Types 5-14: Medium (101-500 nodes), moderate properties
    - Types 15-19: Small (10-50 nodes), many properties
    """
    graph = KnowledgeGraph()
    for i in range(20):
        if i < 5:
            n_nodes = 1200 + i * 100
            extra_cols = {f"prop_{j}": [j] * n_nodes for j in range(3)}
        elif i < 15:
            n_nodes = 150 + i * 20
            extra_cols = {f"prop_{j}": [j] * n_nodes for j in range(10)}
        else:
            n_nodes = 10 + i
            extra_cols = {f"prop_{j}": [j] * n_nodes for j in range(20)}
        df = pd.DataFrame(
            {
                "item_id": list(range(n_nodes)),
                "name": [f"Type{i}_Item_{j}" for j in range(n_nodes)],
                **extra_cols,
            }
        )
        graph.add_nodes(df, f"Type{i}", "item_id", "name")
    # Add some connections
    edges = pd.DataFrame(
        {
            "from_id": list(range(100)),
            "to_id": list(range(100, 200)),
        }
    )
    graph.add_connections(edges, "LINKS", "Type0", "from_id", "Type1", "to_id")
    return graph


@pytest.fixture
def petroleum_graph():
    """Domain graph: Play/Prospect/Discovery/Estimate with temporal and spatial data.

    3 Plays with lat/lon
    20 Prospects with status, geoprovince, lat/lon, date_from/date_to
    10 Discoveries with resource_type, lat/lon
    50 Estimates with value, confidence, date_from/date_to (datetime)
    Connections: HAS_PROSPECT, BECAME_DISCOVERY (share_pct), HAS_ESTIMATE (weight)
    """
    graph = KnowledgeGraph()

    plays = pd.DataFrame(
        {
            "play_id": [1, 2, 3],
            "name": ["North Sea Play", "Atlantic Play", "Barents Play"],
            "region": ["Norwegian Sea", "Atlantic", "Barents Sea"],
            "latitude": [62.0, 64.5, 71.0],
            "longitude": [5.0, 3.0, 25.0],
        }
    )
    graph.add_nodes(plays, "Play", "play_id", "name")

    prospects = pd.DataFrame(
        {
            "prospect_id": list(range(100, 120)),
            "name": [f"Prospect_{i}" for i in range(20)],
            "status": ["Active"] * 10 + ["Closed"] * 5 + ["Matured"] * 5,
            "geoprovince": ["N3"] * 7 + ["M3"] * 7 + ["B1"] * 6,
            "latitude": [60.0 + i * 0.5 for i in range(20)],
            "longitude": [4.0 + i * 0.3 for i in range(20)],
            "date_from": ["2020-01-01"] * 10 + ["2019-01-01"] * 10,
            "date_to": ["2025-12-31"] * 10 + ["2023-12-31"] * 10,
        }
    )
    graph.add_nodes(prospects, "Prospect", "prospect_id", "name")

    discoveries = pd.DataFrame(
        {
            "discovery_id": list(range(200, 210)),
            "name": [f"Discovery_{i}" for i in range(10)],
            "resource_type": ["Oil"] * 5 + ["Gas"] * 5,
            "latitude": [59.0 + i * 0.4 for i in range(10)],
            "longitude": [2.0 + i * 0.2 for i in range(10)],
        }
    )
    graph.add_nodes(discoveries, "Discovery", "discovery_id", "name")

    estimates = pd.DataFrame(
        {
            "estimate_id": list(range(300, 350)),
            "name": [f"Estimate_{i}" for i in range(50)],
            "value": [10.0 + i * 20.0 for i in range(50)],
            "confidence": [0.5 + (i % 10) * 0.05 for i in range(50)],
            "date_from": ["2020-01-01"] * 25 + ["2021-01-01"] * 25,
            "date_to": ["2020-12-31"] * 25 + ["2021-12-31"] * 25,
        }
    )
    graph.add_nodes(
        estimates, "Estimate", "estimate_id", "name", column_types={"date_from": "datetime", "date_to": "datetime"}
    )

    play_prospect = pd.DataFrame(
        {
            "play_id": [1] * 7 + [2] * 7 + [3] * 6,
            "prospect_id": list(range(100, 120)),
        }
    )
    graph.add_connections(play_prospect, "HAS_PROSPECT", "Play", "play_id", "Prospect", "prospect_id")

    prospect_discovery = pd.DataFrame(
        {
            "prospect_id": [100, 101, 102, 107, 108, 109, 114, 115, 116, 117],
            "discovery_id": list(range(200, 210)),
            "share_pct": [100.0, 75.0, 50.0, 80.0, 60.0, 40.0, 90.0, 70.0, 55.0, 45.0],
        }
    )
    graph.add_connections(
        prospect_discovery,
        "BECAME_DISCOVERY",
        "Prospect",
        "prospect_id",
        "Discovery",
        "discovery_id",
        columns=["share_pct"],
    )

    prospect_estimate = pd.DataFrame(
        {
            "prospect_id": [100 + (i % 20) for i in range(50)],
            "estimate_id": list(range(300, 350)),
            "weight": [0.5 + (i % 10) * 0.05 for i in range(50)],
        }
    )
    graph.add_connections(
        prospect_estimate, "HAS_ESTIMATE", "Prospect", "prospect_id", "Estimate", "estimate_id", columns=["weight"]
    )

    return graph


@pytest.fixture
def tiered_graph():
    """Graph with core/supporting tiers for describe() tier testing.

    Core types: Region (3), Project (100), Facility (50)
    Supporting types: ProjectBudget (200, parent=Project),
                      ProjectPhase (150, parent=Project),
                      FacilitySpec (80, parent=Facility)
    Connections: HAS_PROJECT (Region→Project), HAS_FACILITY (Region→Facility),
                 OF_PROJECT (ProjectBudget→Project, ProjectPhase→Project),
                 OF_FACILITY (FacilitySpec→Facility)

    ProjectBudget has timeseries data (simulated via ts metadata).
    """
    graph = KnowledgeGraph()

    # Core: Region
    regions = pd.DataFrame(
        {
            "region_id": [1, 2, 3],
            "name": ["North", "South", "East"],
        }
    )
    graph.add_nodes(regions, "Region", "region_id", "name")

    # Core: Project
    projects = pd.DataFrame(
        {
            "project_id": list(range(1, 101)),
            "name": [f"Project_{i}" for i in range(1, 101)],
            "status": ["Active"] * 60 + ["Completed"] * 40,
        }
    )
    graph.add_nodes(projects, "Project", "project_id", "name")

    # Core: Facility
    facilities = pd.DataFrame(
        {
            "facility_id": list(range(1, 51)),
            "name": [f"Facility_{i}" for i in range(1, 51)],
            "latitude": [60.0 + i * 0.1 for i in range(50)],
            "longitude": [5.0 + i * 0.1 for i in range(50)],
        }
    )
    graph.add_nodes(
        facilities,
        "Facility",
        "facility_id",
        "name",
        column_types={"latitude": "location.lat", "longitude": "location.lon"},
    )

    # Supporting: ProjectBudget (parent=Project)
    budgets = pd.DataFrame(
        {
            "budget_id": list(range(1, 201)),
            "name": [f"Budget_{i}" for i in range(1, 201)],
            "amount": [1000000.0 * i for i in range(1, 201)],
        }
    )
    graph.add_nodes(budgets, "ProjectBudget", "budget_id", "name")

    # Supporting: ProjectPhase (parent=Project)
    phases = pd.DataFrame(
        {
            "phase_id": list(range(1, 151)),
            "name": [f"Phase_{i}" for i in range(1, 151)],
            "phase_type": ["Planning"] * 50 + ["Execution"] * 50 + ["Closing"] * 50,
        }
    )
    graph.add_nodes(phases, "ProjectPhase", "phase_id", "name")

    # Supporting: FacilitySpec (parent=Facility)
    specs = pd.DataFrame(
        {
            "spec_id": list(range(1, 81)),
            "name": [f"Spec_{i}" for i in range(1, 81)],
            "capacity": [100.0 * i for i in range(1, 81)],
        }
    )
    graph.add_nodes(specs, "FacilitySpec", "spec_id", "name")

    # Connections
    region_project = pd.DataFrame(
        {
            "region_id": [(i % 3) + 1 for i in range(100)],
            "project_id": list(range(1, 101)),
        }
    )
    graph.add_connections(region_project, "HAS_PROJECT", "Region", "region_id", "Project", "project_id")

    region_facility = pd.DataFrame(
        {
            "region_id": [(i % 3) + 1 for i in range(50)],
            "facility_id": list(range(1, 51)),
        }
    )
    graph.add_connections(region_facility, "HAS_FACILITY", "Region", "region_id", "Facility", "facility_id")

    of_project_budget = pd.DataFrame(
        {
            "budget_id": list(range(1, 201)),
            "project_id": [(i % 100) + 1 for i in range(200)],
        }
    )
    graph.add_connections(of_project_budget, "OF_PROJECT", "ProjectBudget", "budget_id", "Project", "project_id")

    of_project_phase = pd.DataFrame(
        {
            "phase_id": list(range(1, 151)),
            "project_id": [(i % 100) + 1 for i in range(150)],
        }
    )
    graph.add_connections(of_project_phase, "OF_PROJECT", "ProjectPhase", "phase_id", "Project", "project_id")

    of_facility_spec = pd.DataFrame(
        {
            "spec_id": list(range(1, 81)),
            "facility_id": [(i % 50) + 1 for i in range(80)],
        }
    )
    graph.add_connections(of_facility_spec, "OF_FACILITY", "FacilitySpec", "spec_id", "Facility", "facility_id")

    # Add filler core types to push above 15 core types (triggers inventory path)
    for i in range(14):
        filler = pd.DataFrame(
            {
                "filler_id": list(range(10 + i * 5)),
                "name": [f"Filler{i}_{j}" for j in range(10 + i * 5)],
            }
        )
        graph.add_nodes(filler, f"Filler{i}", "filler_id", "name")

    # Set parent types (tiers)
    graph.set_parent_type("ProjectBudget", "Project")
    graph.set_parent_type("ProjectPhase", "Project")
    graph.set_parent_type("FacilitySpec", "Facility")

    # Add timeseries metadata to ProjectBudget so capability bubbling can be tested
    graph.set_timeseries("ProjectBudget", resolution="year")

    return graph
