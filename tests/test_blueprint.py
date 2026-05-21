"""Tests for kglite.blueprint.from_blueprint()."""

import json

import pandas as pd
import pytest

import kglite
from kglite.blueprint import from_blueprint

# ── Helpers ──────────────────────────────────────────────────────────


def _write_csv(path, df):
    """Write a DataFrame as CSV."""
    df.to_csv(path, index=False)


def _write_blueprint(path, bp):
    """Write a blueprint dict as JSON."""
    with open(path, "w") as f:
        json.dump(bp, f)


def _minimal_blueprint(tmp_path):
    """Create a minimal blueprint with Person nodes + KNOWS edges."""
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [28, 35, 42],
            "city": ["Oslo", "Bergen", "Oslo"],
        }
    )
    _write_csv(tmp_path / "persons.csv", persons)

    knows = pd.DataFrame({"source_id": [1, 2], "target_id": [2, 3]})
    _write_csv(tmp_path / "knows.csv", knows)

    bp = {
        "settings": {"root": str(tmp_path)},
        "nodes": {
            "Person": {
                "csv": "persons.csv",
                "pk": "person_id",
                "title": "name",
                "properties": {
                    "age": "int",
                    "city": "string",
                },
                "skipped": [],
                "connections": {
                    "junction_edges": {
                        "KNOWS": {
                            "csv": "knows.csv",
                            "source_fk": "source_id",
                            "target": "Person",
                            "target_fk": "target_id",
                            "properties": [],
                        }
                    }
                },
            }
        },
    }
    bp_path = tmp_path / "blueprint.json"
    _write_blueprint(bp_path, bp)
    return bp_path


# ── Tests ────────────────────────────────────────────────────────────


class TestBasicLoading:
    def test_load_nodes_and_edges(self, tmp_path):
        bp_path = _minimal_blueprint(tmp_path)
        graph = from_blueprint(bp_path, save=False)

        # Check nodes
        result = graph.cypher("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        names = [r["p.name"] for r in result]
        assert names == ["Alice", "Bob", "Charlie"]

    def test_node_properties(self, tmp_path):
        bp_path = _minimal_blueprint(tmp_path)
        graph = from_blueprint(bp_path, save=False)

        alice = graph.node("Person", 1)
        assert alice is not None
        assert alice["title"] == "Alice"
        assert alice["age"] == 28
        assert alice["city"] == "Oslo"

    def test_junction_edges(self, tmp_path):
        bp_path = _minimal_blueprint(tmp_path)
        graph = from_blueprint(bp_path, save=False)

        result = graph.cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS src, b.name AS tgt ORDER BY src")
        edges = [(r["src"], r["tgt"]) for r in result]
        assert edges == [("Alice", "Bob"), ("Bob", "Charlie")]

    def test_verbose_mode(self, tmp_path, capfd):
        # capfd captures file-descriptor-level stdout so Rust println!
        # output reaches the buffer (capsys would only see Python-side
        # writes).
        bp_path = _minimal_blueprint(tmp_path)
        from_blueprint(bp_path, save=False, verbose=True)
        captured = capfd.readouterr()
        assert "Loading blueprint" in captured.out
        assert "Person" in captured.out

    def test_verbose_edge_count_matches_graph_truth(self, tmp_path, capfd):
        """0.9.1 #1 — the verbose log must report the actual graph
        edge count (queryable via `MATCH ()-[r]->() RETURN count(r)`),
        not the accumulated input-row count from the blueprint
        pipeline. The two diverge when the blueprint touches the same
        edge type from multiple sections (default Update conflict
        handling collapses repeats), or in any future scenario where
        the report's accumulated count overcounts vs the graph."""
        bp_path = _minimal_blueprint(tmp_path)
        graph = from_blueprint(bp_path, save=False, verbose=True)
        captured = capfd.readouterr()

        # Ground truth from the graph
        rows = list(graph.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS n"))
        graph_count = rows[0]["n"]

        # Verbose log must report exactly graph_count under [KNOWS]
        assert f"[KNOWS]: {graph_count} edges" in captured.out
        # And the summary line must also report graph_count
        assert f"{graph_count} edges (1 types)" in captured.out

    def test_top_level_import(self):
        """Verify from_blueprint is importable from kglite top level."""
        assert hasattr(kglite, "from_blueprint")
        assert kglite.from_blueprint is from_blueprint


class TestWarningCapture:
    """0.9.1 #2 — Rust-emitted PyUserWarnings can be captured via the
    standard Python `logging.captureWarnings(True)` pattern. The
    `from_blueprint` docstring documents this; these tests pin the
    behaviour so the docs can't drift unnoticed.
    """

    def test_logging_capture_warnings_pipeline(self, tmp_path):
        """`logging.captureWarnings(True)` routes the Rust-emitted
        UserWarning into the `py.warnings` logger, where it can be
        sent to a file (or any other handler).
        """
        import logging

        log_path = tmp_path / "warnings.log"
        # Snapshot py.warnings handlers so we can restore them
        py_warnings = logging.getLogger("py.warnings")
        prior_handlers = list(py_warnings.handlers)
        prior_level = py_warnings.level
        prior_capture = logging.getLogger("py.warnings").propagate

        try:
            logging.captureWarnings(True)
            handler = logging.FileHandler(str(log_path))
            handler.setLevel(logging.WARNING)
            py_warnings.addHandler(handler)
            py_warnings.setLevel(logging.WARNING)

            # Trigger a Rust-emitted PyUserWarning. The fluent
            # `create_connections()` chain-discard guard is the
            # cleanest reliable trigger.
            g = kglite.KnowledgeGraph()
            g.add_nodes(
                pd.DataFrame([{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]),
                "P",
                "id",
                "name",
            )
            try:
                g.select("P").create_connections("LINK")
            except Exception:
                pass  # the warning fires regardless of any subsequent error

            handler.flush()
            handler.close()
            py_warnings.removeHandler(handler)
        finally:
            # Restore prior state so other tests aren't affected.
            logging.captureWarnings(False)
            py_warnings.handlers = prior_handlers
            py_warnings.setLevel(prior_level)
            py_warnings.propagate = prior_capture

        log_content = log_path.read_text()
        # The Rust-emitted UserWarning should appear in the log.
        assert "create_connections" in log_content, (
            f"py.warnings logger didn't capture the Rust UserWarning. Log: {log_content!r}"
        )


class TestFKEdges:
    def test_fk_edges(self, tmp_path):
        companies = pd.DataFrame({"company_id": [10, 20], "name": ["Acme", "Globex"]})
        persons = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "company_id": [10, 20, 10],
            }
        )
        _write_csv(tmp_path / "companies.csv", companies)
        _write_csv(tmp_path / "persons.csv", persons)

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Company": {
                    "csv": "companies.csv",
                    "pk": "company_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                },
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "properties": {},
                    "skipped": ["company_id"],
                    "connections": {
                        "fk_edges": {
                            "WORKS_AT": {
                                "target": "Company",
                                "fk": "company_id",
                            }
                        }
                    },
                },
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        result = graph.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name AS person, c.name AS company ORDER BY person"
        )
        edges = [(r["person"], r["company"]) for r in result]
        assert edges == [
            ("Alice", "Acme"),
            ("Bob", "Globex"),
            ("Charlie", "Acme"),
        ]


class TestSubNodes:
    def test_sub_nodes_with_parent_fk(self, tmp_path):
        fields = pd.DataFrame({"field_id": [1, 2], "name": ["Troll", "Ekofisk"]})
        reserves = pd.DataFrame(
            {
                "field_id": [1, 1, 2],
                "year": [2020, 2021, 2020],
                "oil": [100.0, 110.0, 200.0],
            }
        )
        _write_csv(tmp_path / "fields.csv", fields)
        _write_csv(tmp_path / "reserves.csv", reserves)

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Field": {
                    "csv": "fields.csv",
                    "pk": "field_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "sub_nodes": {
                        "Reserve": {
                            "csv": "reserves.csv",
                            "pk": "auto",
                            "title": "year",
                            "parent_fk": "field_id",
                            "properties": {"oil": "float"},
                            "skipped": ["field_id"],
                            "connections": {
                                "fk_edges": {
                                    "OF_FIELD": {
                                        "target": "Field",
                                        "fk": "field_id",
                                    }
                                }
                            },
                        }
                    },
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        # Check sub-nodes created
        result = graph.cypher("MATCH (r:Reserve) RETURN r.oil ORDER BY r.oil")
        oils = [r["r.oil"] for r in result]
        assert oils == [100.0, 110.0, 200.0]

        # Check edges to parent
        result = graph.cypher("MATCH (r:Reserve)-[:OF_FIELD]->(f:Field) RETURN f.title AS field, r.oil ORDER BY r.oil")
        assert len(result) == 3
        assert result[0]["field"] == "Troll"


class TestManualNodes:
    def test_manual_nodes_from_fk_values(self, tmp_path):
        fields = pd.DataFrame(
            {
                "field_id": [1, 2, 3],
                "name": ["Troll", "Ekofisk", "Ormen Lange"],
                "area": ["North Sea", "North Sea", "Norwegian Sea"],
            }
        )
        _write_csv(tmp_path / "fields.csv", fields)

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Ocean": {
                    "pk": "name",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                },
                "Field": {
                    "csv": "fields.csv",
                    "pk": "field_id",
                    "title": "name",
                    "properties": {},
                    "skipped": ["area"],
                    "connections": {
                        "fk_edges": {
                            "IN_OCEAN": {
                                "target": "Ocean",
                                "fk": "area",
                            }
                        }
                    },
                },
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        # Check manual nodes created
        result = graph.cypher("MATCH (o:Ocean) RETURN o.title ORDER BY o.title")
        names = [r["o.title"] for r in result]
        assert names == ["North Sea", "Norwegian Sea"]

        # Check FK edges to manual nodes
        result = graph.cypher(
            "MATCH (f:Field)-[:IN_OCEAN]->(o:Ocean) RETURN f.title AS field, o.title AS ocean ORDER BY field"
        )
        assert len(result) == 3


class TestAutoId:
    def test_pk_auto_generates_sequential_ids(self, tmp_path):
        items = pd.DataFrame({"name": ["A", "B", "C"]})
        _write_csv(tmp_path / "items.csv", items)

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Item": {
                    "csv": "items.csv",
                    "pk": "auto",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        result = graph.cypher("MATCH (i:Item) RETURN i.id, i.title ORDER BY i.id")
        ids = [r["i.id"] for r in result]
        assert ids == [1, 2, 3]


class TestFilter:
    def test_filter_rows(self, tmp_path):
        items = pd.DataFrame(
            {
                "item_id": [1, 2, 3, 4],
                "name": ["A", "B", "C", "D"],
                "status": ["Active", "Inactive", "Active", "Active"],
            }
        )
        _write_csv(tmp_path / "items.csv", items)

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Item": {
                    "csv": "items.csv",
                    "pk": "item_id",
                    "title": "name",
                    "properties": {"status": "string"},
                    "skipped": [],
                    "filter": {"status": "Active"},
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        result = graph.cypher("MATCH (i:Item) RETURN i.title ORDER BY i.title")
        names = [r["i.title"] for r in result]
        assert names == ["A", "C", "D"]


class TestTimeseries:
    def test_timeseries_sub_node(self, tmp_path):
        fields = pd.DataFrame({"field_id": [1, 2], "name": ["Troll", "Ekofisk"]})
        production = pd.DataFrame(
            {
                "field_id": [1, 1, 1, 2, 2, 2],
                "name": ["Troll"] * 3 + ["Ekofisk"] * 3,
                "prfYear": [2020, 2020, 2020, 2020, 2020, 2020],
                "prfMonth": [1, 2, 3, 1, 2, 3],
                "prfOil": [1.0, 1.5, 2.0, 0.5, 0.6, 0.7],
                "prfGas": [0.1, 0.2, 0.3, 0.05, 0.06, 0.07],
            }
        )
        _write_csv(tmp_path / "fields.csv", fields)
        _write_csv(tmp_path / "production.csv", production)

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Field": {
                    "csv": "fields.csv",
                    "pk": "field_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "sub_nodes": {
                        "Production": {
                            "csv": "production.csv",
                            "pk": "field_id",
                            "title": "name",
                            "parent_fk": "field_id",
                            "properties": {},
                            "skipped": ["field_id", "name"],
                            "timeseries": {
                                "time_key": {
                                    "year": "prfYear",
                                    "month": "prfMonth",
                                },
                                "resolution": "month",
                                "channels": {
                                    "oil": "prfOil",
                                    "gas": "prfGas",
                                },
                                "units": {
                                    "oil": "MSm3",
                                    "gas": "BSm3",
                                },
                            },
                            "connections": {
                                "fk_edges": {
                                    "OF_FIELD": {
                                        "target": "Field",
                                        "fk": "field_id",
                                    }
                                }
                            },
                        }
                    },
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        # Check timeseries data is accessible
        result = graph.cypher(
            "MATCH (p:Production) RETURN p.title, ts_sum(p.oil, '2020') AS total_oil ORDER BY total_oil DESC"
        )
        assert len(result) == 2
        # Troll: 1.0 + 1.5 + 2.0 = 4.5
        assert result[0]["total_oil"] == pytest.approx(4.5)
        assert result[0]["p.title"] == "Troll"


class TestSaveOutput:
    def test_save_to_output_path(self, tmp_path):
        bp_path = _minimal_blueprint(tmp_path)

        # Add output to blueprint
        with open(bp_path) as f:
            bp = json.load(f)
        bp["settings"]["output"] = "output/graph.kgl"
        _write_blueprint(bp_path, bp)

        from_blueprint(bp_path, save=True)
        assert (tmp_path / "output" / "graph.kgl").exists()

        # Verify saved graph can be loaded
        loaded = kglite.load(str(tmp_path / "output" / "graph.kgl"))
        result = loaded.cypher("MATCH (p:Person) RETURN count(p) AS n")
        assert result[0]["n"] == 3

    def test_no_save_when_disabled(self, tmp_path):
        bp_path = _minimal_blueprint(tmp_path)
        with open(bp_path) as f:
            bp = json.load(f)
        bp["settings"]["output"] = "output/graph.kgl"
        _write_blueprint(bp_path, bp)

        from_blueprint(bp_path, save=False)
        assert not (tmp_path / "output" / "graph.kgl").exists()


class TestErrorHandling:
    def test_missing_blueprint_file(self):
        with pytest.raises(FileNotFoundError, match="Blueprint file not found"):
            from_blueprint("/nonexistent/blueprint.json")

    def test_missing_csv_is_nonfatal(self, tmp_path):
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Missing": {
                    "csv": "nonexistent.csv",
                    "pk": "id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)
        # Graph should still be created, just empty
        assert graph.cypher("MATCH (n) RETURN count(n) AS n")[0]["n"] == 0

    def test_missing_fk_column_is_nonfatal(self, tmp_path):
        items = pd.DataFrame({"item_id": [1], "name": ["A"]})
        _write_csv(tmp_path / "items.csv", items)

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Item": {
                    "csv": "items.csv",
                    "pk": "item_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "connections": {
                        "fk_edges": {
                            "BAD_EDGE": {
                                "target": "Other",
                                "fk": "nonexistent_col",
                            }
                        }
                    },
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)
        # Node loaded, edge skipped
        assert graph.cypher("MATCH (i:Item) RETURN count(i) AS n")[0]["n"] == 1


class TestJunctionEdgeProperties:
    def test_junction_edge_with_properties(self, tmp_path):
        persons = pd.DataFrame({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        movies = pd.DataFrame({"movie_id": [10, 20], "title": ["Film A", "Film B"]})
        ratings = pd.DataFrame(
            {
                "person_id": [1, 1, 2],
                "movie_id": [10, 20, 10],
                "score": [5, 3, 4],
            }
        )
        _write_csv(tmp_path / "persons.csv", persons)
        _write_csv(tmp_path / "movies.csv", movies)
        _write_csv(tmp_path / "ratings.csv", ratings)

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "connections": {
                        "junction_edges": {
                            "RATED": {
                                "csv": "ratings.csv",
                                "source_fk": "person_id",
                                "target": "Movie",
                                "target_fk": "movie_id",
                                "properties": ["score"],
                            }
                        }
                    },
                },
                "Movie": {
                    "csv": "movies.csv",
                    "pk": "movie_id",
                    "title": "title",
                    "properties": {},
                    "skipped": [],
                },
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        result = graph.cypher(
            "MATCH (p:Person)-[r:RATED]->(m:Movie) RETURN p.name, m.title, r.score ORDER BY p.name, m.title"
        )
        assert len(result) == 3
        assert result[0]["r.score"] == 5
        assert result[0]["p.name"] == "Alice"
        assert result[0]["m.title"] == "Film A"


class TestStreamingNodeLoader:
    """0.9.44 F1 — streaming node-loader parity. The buffered path
    materialised the full CSV before dispatching to add_nodes; the
    streaming path chunks the CSV and calls add_nodes per chunk. Both
    must produce identical graphs for streaming-eligible specs
    (no timeseries, no spatial, pk != 'auto')."""

    def test_multi_chunk_node_load(self, tmp_path, monkeypatch):
        # Set a small chunk size so a modest CSV spans multiple chunks
        # — exercises the per-chunk add_nodes accumulation path.
        monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", "100")
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")

        n = 350  # 4 chunks: 100 + 100 + 100 + 50
        persons = pd.DataFrame(
            {
                "person_id": list(range(n)),
                "name": [f"P{i}" for i in range(n)],
                "age": [20 + (i % 60) for i in range(n)],
            }
        )
        _write_csv(tmp_path / "persons.csv", persons)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "properties": {"age": "int"},
                    "skipped": [],
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        count = graph.cypher("MATCH (p:Person) RETURN count(p) AS n")[0]["n"]
        assert count == n
        # Spot-check a row from each chunk boundary.
        for i in [0, 99, 100, 199, 200, 299, 300, 349]:
            node = graph.node("Person", i)
            assert node is not None, f"missing pk={i}"
            assert node["title"] == f"P{i}"
            assert node["age"] == 20 + (i % 60)

    def test_streamed_node_with_fk_edges(self, tmp_path, monkeypatch):
        """FK edges from a streamed-parent spec still resolve. F1 keeps
        the parent CSV in CsvCache; F3 will switch FK edges to streaming."""
        monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", "50")
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
        n = 120
        persons = pd.DataFrame(
            {
                "person_id": list(range(n)),
                "name": [f"P{i}" for i in range(n)],
                "manager_id": [(i // 10) * 10 for i in range(n)],
            }
        )
        _write_csv(tmp_path / "persons.csv", persons)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "connections": {
                        "fk_edges": {
                            "MANAGED_BY": {
                                "target": "Person",
                                "fk": "manager_id",
                            }
                        }
                    },
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        n_nodes = graph.cypher("MATCH (p:Person) RETURN count(p) AS n")[0]["n"]
        assert n_nodes == n
        n_edges = graph.cypher("MATCH (a:Person)-[r:MANAGED_BY]->(b:Person) RETURN count(r) AS n")[0]["n"]
        # Each person points at a manager (their own id // 10 * 10).
        # Persons 0, 10, 20, ... are self-managed (still creates an edge).
        assert n_edges == n

    def test_streamed_with_filter(self, tmp_path, monkeypatch):
        """Filter applied per-chunk drops the right rows."""
        monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", "30")
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
        n = 100
        items = pd.DataFrame(
            {
                "item_id": list(range(n)),
                "active": ["true" if i % 2 == 0 else "false" for i in range(n)],
                "name": [f"I{i}" for i in range(n)],
            }
        )
        _write_csv(tmp_path / "items.csv", items)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Item": {
                    "csv": "items.csv",
                    "pk": "item_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "filter": {"active": "true"},
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)
        count = graph.cypher("MATCH (i:Item) RETURN count(i) AS n")[0]["n"]
        assert count == 50  # half of n filtered through


class TestStreamingAutoPk:
    """0.9.44 F2 — `pk: "auto"` flows through the streaming path with
    a per-spec counter. Each chunk gets a dense id range; total ids
    span 1..=N matching the buffered path's behaviour."""

    def test_multi_chunk_auto_pk_is_dense_and_monotonic(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", "75")
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
        n = 250  # 4 chunks at chunk_size 75: 75 + 75 + 75 + 25
        items = pd.DataFrame({"name": [f"I{i}" for i in range(n)]})
        _write_csv(tmp_path / "items.csv", items)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Item": {
                    "csv": "items.csv",
                    "pk": "auto",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        # The synthesised id column is `_Item_id` (per-spec naming).
        # `i.id` resolves to the spec's pk via aliasing.
        result = graph.cypher("MATCH (i:Item) RETURN i.id AS id ORDER BY id")
        ids = [r["id"] for r in result]
        assert ids == list(range(1, n + 1)), (
            f"expected dense 1..={n}, got len={len(ids)} first={ids[:5]} last={ids[-5:]}"
        )

    def test_auto_pk_with_filter_keeps_dense_ids(self, tmp_path, monkeypatch):
        """Filter is applied per-chunk; auto-pk counter advances only
        by the post-filter row count. Dense ids over kept rows."""
        monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", "40")
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
        n = 200
        items = pd.DataFrame(
            {
                "name": [f"I{i}" for i in range(n)],
                "active": ["true" if i % 3 != 0 else "false" for i in range(n)],
            }
        )
        _write_csv(tmp_path / "items.csv", items)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Item": {
                    "csv": "items.csv",
                    "pk": "auto",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "filter": {"active": "true"},
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)
        # 2/3 rows survive — ids should be dense 1..=kept_n.
        expected_kept = sum(1 for i in range(n) if i % 3 != 0)
        result = graph.cypher("MATCH (i:Item) RETURN i.id AS id ORDER BY id")
        ids = [r["id"] for r in result]
        assert ids == list(range(1, expected_kept + 1))


class TestStreamingFkEdges:
    """0.9.44 F3 — FK edges from streaming-eligible specs are
    built per-chunk, with `connect()` called once per (chunk, edge)
    pair. The streamed-parent CsvCache is bypassed, so peak RAM
    during the FK phase is bounded by chunk size."""

    def test_multi_chunk_fk_edges(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", "50")
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
        n_companies = 20
        n_employees = 500
        companies = pd.DataFrame(
            {
                "company_id": list(range(n_companies)),
                "name": [f"C{i}" for i in range(n_companies)],
            }
        )
        employees = pd.DataFrame(
            {
                "employee_id": list(range(n_employees)),
                "name": [f"E{i}" for i in range(n_employees)],
                "company_id": [i % n_companies for i in range(n_employees)],
            }
        )
        _write_csv(tmp_path / "companies.csv", companies)
        _write_csv(tmp_path / "employees.csv", employees)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Company": {
                    "csv": "companies.csv",
                    "pk": "company_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                },
                "Employee": {
                    "csv": "employees.csv",
                    "pk": "employee_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "connections": {
                        "fk_edges": {
                            "WORKS_AT": {
                                "target": "Company",
                                "fk": "company_id",
                            }
                        }
                    },
                },
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        edge_count = graph.cypher("MATCH (e:Employee)-[r:WORKS_AT]->(c:Company) RETURN count(r) AS n")[0]["n"]
        assert edge_count == n_employees
        # Each company gets n_employees/n_companies employees.
        per_company = graph.cypher("MATCH (e:Employee)-[:WORKS_AT]->(c:Company {company_id: 0}) RETURN count(e) AS n")[
            0
        ]["n"]
        assert per_company == n_employees // n_companies

    def test_streamed_auto_pk_subnode_fk_edges(self, tmp_path, monkeypatch):
        """Sub-node with `pk:"auto"` + parent_fk emits OF_PARENT
        edges via streaming. Source ids must align between the node
        loader (assigns 1..=N) and the FK loader (also assigns 1..=N
        from independent counter). Edge count = sub-row count."""
        monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", "60")
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
        n_fields = 5
        n_reserves = 200
        fields = pd.DataFrame(
            {
                "field_id": list(range(n_fields)),
                "name": [f"F{i}" for i in range(n_fields)],
            }
        )
        reserves = pd.DataFrame(
            {
                "field_id": [i % n_fields for i in range(n_reserves)],
                "year": [2000 + (i // n_fields) for i in range(n_reserves)],
                "oil": [100.0 + i for i in range(n_reserves)],
            }
        )
        _write_csv(tmp_path / "fields.csv", fields)
        _write_csv(tmp_path / "reserves.csv", reserves)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Field": {
                    "csv": "fields.csv",
                    "pk": "field_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "sub_nodes": {
                        "Reserve": {
                            "csv": "reserves.csv",
                            "pk": "auto",
                            "title": "year",
                            "parent_fk": "field_id",
                            "properties": {"oil": "float"},
                            "skipped": [],
                            "connections": {
                                "fk_edges": {
                                    "OF_FIELD": {
                                        "target": "Field",
                                        "fk": "field_id",
                                    }
                                }
                            },
                        }
                    },
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        n_nodes = graph.cypher("MATCH (r:Reserve) RETURN count(r) AS n")[0]["n"]
        assert n_nodes == n_reserves
        n_edges = graph.cypher("MATCH (r:Reserve)-[:OF_FIELD]->(f:Field) RETURN count(r) AS n")[0]["n"]
        assert n_edges == n_reserves


class TestProvisionalNodes:
    """Auto-vivification: an edge to a missing node creates a
    `_provisional` stub instead of silently dropping the edge."""

    def test_fk_edge_to_missing_node_vivifies_stub(self, tmp_path):
        # Person 2 reports to manager 99, which has no row of its own.
        persons = pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "mgr": [2, 99]})
        _write_csv(tmp_path / "persons.csv", persons)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Person": {
                    "csv": "persons.csv",
                    "pk": "id",
                    "title": "name",
                    "properties": {"name": "string"},
                    "connections": {"fk_edges": {"REPORTS_TO": {"target": "Person", "fk": "mgr"}}},
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        g = from_blueprint(tmp_path / "bp.json", save=False)
        # 2 real Person nodes + 1 vivified stub (id 99).
        assert g.cypher("MATCH (p:Person) RETURN count(p) AS n")[0]["n"] == 3
        # Both REPORTS_TO edges present — none dropped.
        assert g.cypher("MATCH ()-[r:REPORTS_TO]->() RETURN count(r) AS n")[0]["n"] == 2
        stub = g.cypher("MATCH (p:Person) WHERE p._provisional = true RETURN p.id AS id")
        assert [r["id"] for r in stub] == [99]

    def test_junction_edge_to_missing_nodes_vivifies(self, tmp_path):
        # The loading-order case: Class A is loaded, but the friends
        # CSV references Class B ids (4,5,6) that have no rows.
        students = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        _write_csv(tmp_path / "students.csv", students)
        friends = pd.DataFrame({"src": [1, 2, 4], "dst": [2, 5, 6]})
        _write_csv(tmp_path / "friends.csv", friends)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Student": {
                    "csv": "students.csv",
                    "pk": "id",
                    "title": "name",
                    "properties": {"name": "string"},
                    "connections": {
                        "junction_edges": {
                            "FRIEND": {
                                "csv": "friends.csv",
                                "source_fk": "src",
                                "target": "Student",
                                "target_fk": "dst",
                                "properties": [],
                            }
                        }
                    },
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        g = from_blueprint(tmp_path / "bp.json", save=False)
        # 3 real + 3 vivified (4,5,6).
        assert g.cypher("MATCH (s:Student) RETURN count(s) AS n")[0]["n"] == 6
        # All 3 friend edges present — none dropped.
        assert g.cypher("MATCH ()-[r:FRIEND]->() RETURN count(r) AS n")[0]["n"] == 3
        prov = g.cypher("MATCH (s:Student) WHERE s._provisional = true RETURN s.id AS id ORDER BY id")
        assert [r["id"] for r in prov] == [4, 5, 6]

    def test_reserved_provisional_property_name_rejected(self, tmp_path):
        items = pd.DataFrame({"id": [1], "_provisional": ["x"]})
        _write_csv(tmp_path / "items.csv", items)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Item": {
                    "csv": "items.csv",
                    "pk": "id",
                    "title": "id",
                    "properties": {"_provisional": "string"},
                    "connections": {},
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        with pytest.raises(Exception, match="reserved"):
            from_blueprint(tmp_path / "bp.json", save=False)

    def test_same_type_edge_missing_both_endpoints(self, tmp_path):
        # Id 9 has no row and is referenced as both a source and a
        # target — it must be vivified exactly once and stay marked.
        students = pd.DataFrame({"id": [1], "name": ["a"]})
        _write_csv(tmp_path / "students.csv", students)
        friends = pd.DataFrame({"src": [1, 9], "dst": [9, 1]})
        _write_csv(tmp_path / "friends.csv", friends)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Student": {
                    "csv": "students.csv",
                    "pk": "id",
                    "title": "name",
                    "properties": {"name": "string"},
                    "connections": {
                        "junction_edges": {
                            "FRIEND": {
                                "csv": "friends.csv",
                                "source_fk": "src",
                                "target": "Student",
                                "target_fk": "dst",
                                "properties": [],
                            }
                        }
                    },
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        g = from_blueprint(tmp_path / "bp.json", save=False)
        assert g.cypher("MATCH (s:Student) RETURN count(s) AS n")[0]["n"] == 2
        assert g.cypher("MATCH ()-[r:FRIEND]->() RETURN count(r) AS n")[0]["n"] == 2
        prov = g.cypher("MATCH (s:Student) WHERE s._provisional = true RETURN s.id AS id")
        assert [r["id"] for r in prov] == [9]

    def test_promotion_clears_marker_on_real_upsert(self):
        # The loading-order fix end to end: Class A, then Friends
        # (vivifies Class B stubs), then Class B — the real rows
        # promote the stubs and keep their friendships.
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}), "Student", "id", "name")
        g.add_connections(
            pd.DataFrame({"src": [1, 2, 4], "dst": [2, 5, 6]}),
            "FRIEND",
            "Student",
            "src",
            "Student",
            "dst",
        )
        assert g.cypher("MATCH (s:Student) WHERE s._provisional = true RETURN count(s) AS n")[0]["n"] == 3
        # Class B arrives last — its rows promote the stubs.
        g.add_nodes(pd.DataFrame({"id": [4, 5, 6], "name": ["d", "e", "f"]}), "Student", "id", "name")
        assert g.cypher("MATCH (s:Student) WHERE s._provisional = true RETURN count(s) AS n")[0]["n"] == 0
        assert g.cypher("MATCH (s:Student) RETURN count(s) AS n")[0]["n"] == 6
        assert g.cypher("MATCH ()-[r:FRIEND]->() RETURN count(r) AS n")[0]["n"] == 3
        # Class B kept the friendships made before its rows loaded.
        assert g.cypher("MATCH (s:Student {id: 5}) RETURN s.name AS name")[0]["name"] == "e"
