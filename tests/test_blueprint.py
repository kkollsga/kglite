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
    with open(path, "w", encoding="utf-8") as f:
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

        log_content = log_path.read_text(encoding="utf-8")
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


class TestManualNodeIdTypes:
    """A CSV-less type is synthesised from the distinct values of every FK
    column pointing at it. The FK edge that follows resolves its endpoint
    against the same values, so the two must agree on the id's *type* — if
    they don't, the edge finds nothing, vivifies a stub, and the type ends up
    with two nodes per value.
    """

    def _build(self, tmp_path, city_ids):
        persons = pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "city_id": city_ids,
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
                    "connections": {"fk_edges": {"LIVES_IN": {"target": "City", "fk": "city_id"}}},
                },
                "City": {"pk": "city_id"},
            },
        }
        bp_path = tmp_path / "bp.json"
        _write_blueprint(bp_path, bp)
        return kglite.from_blueprint(str(bp_path), save=False)

    def test_numeric_fk_makes_one_node_per_value(self, tmp_path):
        g = self._build(tmp_path, [10, 20, 10])
        assert len(list(g.cypher("MATCH (c:City) RETURN c.id"))) == 2

    def test_numeric_fk_edges_reach_the_synthesised_nodes(self, tmp_path):
        """Every edge must land on a real City, not on a stub the loader had
        to invent because the id types disagreed."""
        g = self._build(tmp_path, [10, 20, 10])
        rows = list(
            g.cypher(
                "MATCH (p:Person)-[:LIVES_IN]->(c:City) WHERE c._provisional IS NULL RETURN p.title AS p ORDER BY p"
            )
        )
        assert [r["p"] for r in rows] == ["Alice", "Bob", "Charlie"]

    def test_text_fk_is_unaffected(self, tmp_path):
        g = self._build(tmp_path, ["Oslo", "Bergen", "Oslo"])
        assert len(list(g.cypher("MATCH (c:City) RETURN c.id"))) == 2


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
        with open(bp_path, encoding="utf-8") as f:
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
        with open(bp_path, encoding="utf-8") as f:
            bp = json.load(f)
        bp["settings"]["output"] = "output/graph.kgl"
        _write_blueprint(bp_path, bp)

        from_blueprint(bp_path, save=False)
        assert not (tmp_path / "output" / "graph.kgl").exists()

    def test_save_defaults_to_on_when_blueprint_declares_output(self, tmp_path):
        """The default (``save`` omitted) still honours ``settings.output``."""
        bp_path = _minimal_blueprint(tmp_path)
        with open(bp_path, encoding="utf-8") as f:
            bp = json.load(f)
        bp["settings"]["output"] = "output/graph.kgl"
        _write_blueprint(bp_path, bp)

        from_blueprint(bp_path)
        assert (tmp_path / "output" / "graph.kgl").exists()

    def test_disk_mode_publishes_the_path_directory(self, tmp_path):
        """``storage="disk"`` + ``path`` is a save destination.

        The build leaves a working directory; publication happens at
        ``save()``. Before this was wired up the directory held only
        ``.kglite.lock`` / ``.working-*`` / a partial ``seg_000/`` and
        ``kglite.load()`` rejected it.
        """
        bp_path = _minimal_blueprint(tmp_path)
        out = tmp_path / "disk-graph"

        from_blueprint(bp_path, storage="disk", path=str(out))

        assert (out / "CURRENT").exists(), sorted(p.name for p in out.iterdir())
        reopened = kglite.load(str(out))
        result = reopened.cypher("MATCH (p:Person) RETURN count(p) AS n")
        assert result[0]["n"] == 3

    def test_disk_mode_save_false_leaves_the_build_unpublished(self, tmp_path):
        """``save=False`` still means "do not publish the build" on disk.

        A disk graph *is* its directory, so creating one always writes: the
        lock, ``seg_000/``, and — since load-or-create has to hold for disk
        too — an empty generation, so that a crash before the first ``save()``
        does not leave a path every later open refuses. What ``save=False``
        withholds is the build. The directory therefore opens, and opens
        *empty*: none of the blueprint's nodes are in it.
        """
        bp_path = _minimal_blueprint(tmp_path)
        out = tmp_path / "disk-graph"

        from_blueprint(bp_path, save=False, storage="disk", path=str(out))

        assert (out / "CURRENT").exists()
        assert kglite.load(str(out)).cypher("MATCH (p:Person) RETURN count(p) AS n")[0]["n"] == 0

    def test_explicit_save_without_destination_raises(self, tmp_path):
        """An explicit ``save=True`` that cannot be honoured must not pass."""
        bp_path = _minimal_blueprint(tmp_path)

        with pytest.raises(ValueError, match="nowhere to write"):
            from_blueprint(bp_path, save=True)

    def test_default_without_destination_builds_in_memory(self, tmp_path):
        """The default is "save if there is somewhere to save" — not an error."""
        bp_path = _minimal_blueprint(tmp_path)
        before = sorted(p.name for p in tmp_path.iterdir())

        graph = from_blueprint(bp_path)

        assert graph.cypher("MATCH (p:Person) RETURN count(p) AS n")[0]["n"] == 3
        assert sorted(p.name for p in tmp_path.iterdir()) == before


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

    def _repeated_fk_bp(self, tmp_path, n_rows):
        """One company, one employee, `n_rows` identical membership rows —
        so every row builds the same (source, target) FK pair."""
        _write_csv(tmp_path / "companies.csv", pd.DataFrame({"company_id": [0], "name": ["C0"]}))
        _write_csv(
            tmp_path / "employees.csv",
            pd.DataFrame(
                {
                    "employee_id": [1] * n_rows,
                    "name": ["E1"] * n_rows,
                    "company_id": [0] * n_rows,
                }
            ),
        )
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Company": {"csv": "companies.csv", "pk": "company_id", "title": "name", "properties": {}},
                "Employee": {
                    "csv": "employees.csv",
                    "pk": "employee_id",
                    "title": "name",
                    "properties": {},
                    "connections": {"fk_edges": {"WORKS_AT": {"target": "Company", "fk": "company_id"}}},
                },
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        return tmp_path / "bp.json"

    def test_repeated_fk_rows_do_not_depend_on_the_chunk_size(self, tmp_path, monkeypatch):
        """Streaming a node CSV bounds peak RAM; it must not decide how many
        edges its FK rows produce. Before the fix the first chunk registered
        the connection type and every later chunk merged its rows onto that
        chunk's edges, so the edge count fell as the chunk size did."""
        n_rows = 20
        bp_path = self._repeated_fk_bp(tmp_path, n_rows)

        def build(threshold_mb, chunk_size):
            monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", str(threshold_mb))
            monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", str(chunk_size))
            graph = from_blueprint(bp_path, save=False)
            return graph.cypher("MATCH (:Employee)-[r:WORKS_AT]->(:Company) RETURN count(r) AS n")[0]["n"]

        # Buffered (one connect() call for the whole spec) is the reference.
        buffered = build(100, 5)
        assert buffered == n_rows
        for chunk_size in (3, 5, 7, n_rows):
            assert build(0, chunk_size) == buffered, f"streamed chunk_size={chunk_size} changed the edge count"

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

    def test_purge_provisional_deletes_unpromoted_stubs(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}), "Student", "id", "name")
        g.add_connections(
            pd.DataFrame({"src": [1, 2], "dst": [2, 9]}),  # 9 has no row -> stub
            "FRIEND",
            "Student",
            "src",
            "Student",
            "dst",
        )
        assert g.cypher("MATCH (s:Student) RETURN count(s) AS n")[0]["n"] == 3
        assert g.cypher("MATCH ()-[r:FRIEND]->() RETURN count(r) AS n")[0]["n"] == 2
        result = g.purge_provisional()
        assert result["nodes_purged"] == 1
        assert result["edges_removed"] == 1
        # Stub 9 + its incident edge gone; real nodes and the 1->2 edge spared.
        ids = sorted(r["id"] for r in g.cypher("MATCH (s:Student) RETURN s.id AS id"))
        assert ids == [1, 2]
        assert g.cypher("MATCH ()-[r:FRIEND]->() RETURN count(r) AS n")[0]["n"] == 1

    def test_purge_provisional_spares_promoted_stubs(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": [1], "name": ["a"]}), "Student", "id", "name")
        g.add_connections(
            pd.DataFrame({"src": [1, 1], "dst": [8, 9]}),  # 8 and 9 both missing
            "FRIEND",
            "Student",
            "src",
            "Student",
            "dst",
        )
        # Stub 8 is promoted by a real row; stub 9 is left dangling.
        g.add_nodes(pd.DataFrame({"id": [8], "name": ["h"]}), "Student", "id", "name")
        result = g.purge_provisional()
        assert result["nodes_purged"] == 1  # only 9
        ids = sorted(r["id"] for r in g.cypher("MATCH (s:Student) RETURN s.id AS id"))
        assert ids == [1, 8]

    def test_blueprint_auto_purge_drops_unpromoted_stubs(self, tmp_path):
        students = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        _write_csv(tmp_path / "students.csv", students)
        friends = pd.DataFrame({"src": [1, 2], "dst": [2, 9]})  # 9 has no row
        _write_csv(tmp_path / "friends.csv", friends)
        bp = {
            "settings": {"root": str(tmp_path), "auto_purge": True},
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
        # Stub 9 and the 2->9 edge purged at build end; real nodes and
        # the 1->2 edge kept.
        assert g.cypher("MATCH (s:Student) RETURN count(s) AS n")[0]["n"] == 2
        assert g.cypher("MATCH ()-[r:FRIEND]->() RETURN count(r) AS n")[0]["n"] == 1
        assert g.cypher("MATCH (s:Student) WHERE s._provisional = true RETURN count(s) AS n")[0]["n"] == 0


class TestUnknownPropertyTypeWarning:
    def test_unknown_type_value_warns(self, tmp_path, capfd):
        """A properties/property_types value that is neither a type keyword
        nor a spatial target was silently ignored (the rename-map trap);
        now the build report warns, naming column and value."""
        bp_path = _minimal_blueprint(tmp_path)
        with open(bp_path, encoding="utf-8") as f:
            bp = json.load(f)
        bp["nodes"]["Person"]["properties"]["age"] = "renamedAge"
        _write_blueprint(bp_path, bp)

        kglite.from_blueprint(str(bp_path), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "renamedAge" in err
        assert "does not rename columns" in err

    def test_known_values_stay_silent(self, tmp_path, capfd):
        bp_path = _minimal_blueprint(tmp_path)
        kglite.from_blueprint(str(bp_path), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "does not rename columns" not in err


class TestJunctionRename:
    def _bp_with_junction_props(self, tmp_path, rename=None, extra=None):
        persons = pd.DataFrame({"person_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        _write_csv(tmp_path / "persons.csv", persons)
        knows = pd.DataFrame(
            {
                "source_id": [1, 2],
                "target_id": [2, 3],
                "fldFrom": ["2001-01-01", "2002-02-02"],
            }
        )
        _write_csv(tmp_path / "knows.csv", knows)
        junction = {
            "csv": "knows.csv",
            "source_fk": "source_id",
            "target": "Person",
            "target_fk": "target_id",
            "properties": ["fldFrom"],
            "property_types": {"fldFrom": "date"},
        }
        if rename is not None:
            junction["rename"] = rename
        if extra:
            junction.update(extra)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "properties": {},
                    "connections": {"junction_edges": {"KNOWS": junction}},
                }
            },
        }
        bp_path = tmp_path / "blueprint.json"
        _write_blueprint(bp_path, bp)
        return bp_path

    def test_rename_lands_property_under_new_name(self, tmp_path):
        bp_path = self._bp_with_junction_props(tmp_path, rename={"fldFrom": "validFrom"})
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN r.validFrom AS vf, r.fldFrom AS old").to_list()
        assert len(rows) == 2
        assert all(r["vf"] is not None for r in rows)
        assert all(r["old"] is None for r in rows)

    def test_without_rename_csv_name_kept(self, tmp_path):
        bp_path = self._bp_with_junction_props(tmp_path)
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN r.fldFrom AS vf").to_list()
        assert all(r["vf"] is not None for r in rows)

    def test_rename_of_fk_column_reports_error(self, tmp_path, capfd):
        bp_path = self._bp_with_junction_props(tmp_path, rename={"source_id": "src"})
        graph = from_blueprint(bp_path, save=False, verbose=True)
        err = capfd.readouterr().err
        assert "rename of fk column" in err
        # The junction is skipped, not half-loaded.
        assert graph.cypher("MATCH ()-[r:KNOWS]->() RETURN count(r) AS c").to_list()[0]["c"] == 0

    def test_rename_key_must_be_declared_property(self, tmp_path, capfd):
        bp_path = self._bp_with_junction_props(tmp_path, rename={"nosuch": "x"})
        from_blueprint(bp_path, save=False, verbose=True)
        err = capfd.readouterr().err
        assert "not in 'properties'" in err


class TestNodeLabels:
    """`labels` on a node spec stamps secondary labels on every node of the
    type. Without it a blueprint can express `:Disease` and `:Phenotype` only
    as separate node types, and a query over the union has to name each one.
    """

    def _bp(self, tmp_path, labels, sub_labels=None, rows=3):
        persons = pd.DataFrame(
            {
                "person_id": list(range(1, rows + 1)),
                "name": [f"P{i}" for i in range(1, rows + 1)],
                "city_id": [10] * rows,
            }
        )
        _write_csv(tmp_path / "persons.csv", persons)
        person = {
            "csv": "persons.csv",
            "pk": "person_id",
            "title": "name",
            "connections": {"fk_edges": {"LIVES_IN": {"target": "City", "fk": "city_id"}}},
        }
        if labels is not None:
            person["labels"] = labels
        if sub_labels is not None:
            aliases = pd.DataFrame(
                {
                    "person_id": list(range(1, rows + 1)),
                    "alias": [f"a{i}" for i in range(1, rows + 1)],
                }
            )
            _write_csv(tmp_path / "aliases.csv", aliases)
            person["sub_nodes"] = {
                "Alias": {
                    "csv": "aliases.csv",
                    "pk": "alias",
                    "parent_fk": "person_id",
                    "labels": sub_labels,
                }
            }
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {"Person": person, "City": {"pk": "city_id"}},
        }
        bp_path = tmp_path / "bp.json"
        _write_blueprint(bp_path, bp)
        return bp_path

    def test_labels_are_visible_to_labels_and_to_a_label_match(self, tmp_path):
        g = kglite.from_blueprint(str(self._bp(tmp_path, ["Human", "Agent"])), save=False)
        rows = list(g.cypher("MATCH (p:Person) RETURN labels(p) AS l ORDER BY p.title"))
        assert all(set(r["l"]) >= {"Person", "Human", "Agent"} for r in rows), rows
        assert len(list(g.cypher("MATCH (n:Human) RETURN n.title"))) == 3
        assert len(list(g.cypher("MATCH (n:Agent) RETURN n.title"))) == 3

    def test_no_labels_key_stamps_nothing(self, tmp_path):
        g = kglite.from_blueprint(str(self._bp(tmp_path, None)), save=False)
        rows = list(g.cypher("MATCH (p:Person) RETURN labels(p) AS l"))
        assert all(r["l"] == ["Person"] for r in rows), rows

    def test_sub_node_labels_are_stamped_too(self, tmp_path):
        g = kglite.from_blueprint(str(self._bp(tmp_path, ["Human"], sub_labels=["Name"])), save=False)
        assert len(list(g.cypher("MATCH (n:Name) RETURN n.title"))) == 3

    def test_provisional_stubs_of_the_type_are_labelled(self, tmp_path):
        """A blueprint owns its type's labels. `City` has a CSV listing only
        city 10, so the reference to 99 is vivified as a stub during the edge
        phase — after the node phases. It must carry the label anyway, or
        `MATCH (:Place)` silently misses exactly the nodes that arrived via an
        edge rather than a row."""
        cities = pd.DataFrame({"city_id": [10], "city_name": ["Oslo"]})
        _write_csv(tmp_path / "cities.csv", cities)
        persons = pd.DataFrame({"person_id": [1, 2], "name": ["A", "B"], "city_id": [10, 99]})
        _write_csv(tmp_path / "persons.csv", persons)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "connections": {"fk_edges": {"LIVES_IN": {"target": "City", "fk": "city_id"}}},
                },
                "City": {
                    "csv": "cities.csv",
                    "pk": "city_id",
                    "title": "city_name",
                    "labels": ["Place"],
                },
            },
        }
        bp_path = tmp_path / "bp.json"
        _write_blueprint(bp_path, bp)
        g = kglite.from_blueprint(str(bp_path), save=False)
        ids = sorted(r["c.id"] for r in g.cypher("MATCH (c:Place) RETURN c.id"))
        assert ids == [10, 99], ids

    def test_streaming_and_buffered_agree(self, tmp_path, monkeypatch):
        """The streaming node loader is a RAM knob, so it must produce the same
        labels as the buffered path."""
        monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", "2")
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
        g = kglite.from_blueprint(str(self._bp(tmp_path, ["Human"], rows=7)), save=False)
        assert len(list(g.cypher("MATCH (n:Human) RETURN n.title"))) == 7

    def test_a_label_equal_to_the_type_is_not_an_error(self, tmp_path):
        g = kglite.from_blueprint(str(self._bp(tmp_path, ["Person", "Human"])), save=False)
        rows = list(g.cypher("MATCH (p:Person) RETURN labels(p) AS l"))
        assert all(r["l"].count("Person") == 1 for r in rows), rows


class TestUnknownSpecKeyWarning:
    """A key the blueprint parser does not read was dropped in silence, and the
    build reported success on a graph the author did not describe. `"lables"`
    for `"labels"`, `"propertes"` for `"properties"` — the misspelling costs
    every property or edge that key was carrying, with no diagnostic anywhere.
    """

    def _bp_with(self, tmp_path, mutate):
        bp_path = _minimal_blueprint(tmp_path)
        with open(bp_path, encoding="utf-8") as f:
            bp = json.load(f)
        mutate(bp)
        _write_blueprint(bp_path, bp)
        return bp_path

    def test_unknown_top_level_key_warns(self, tmp_path, capfd):
        def mutate(bp):
            bp["nodez"] = {}

        kglite.from_blueprint(str(self._bp_with(tmp_path, mutate)), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "unknown key 'nodez'" in err
        assert "Did you mean 'nodes'?" in err

    def test_unknown_node_key_warns_with_a_near_miss_hint(self, tmp_path, capfd):
        def mutate(bp):
            bp["nodes"]["Person"]["propertes"] = {"age": "int"}

        kglite.from_blueprint(str(self._bp_with(tmp_path, mutate)), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "node 'Person'" in err
        assert "unknown key 'propertes'" in err
        assert "Did you mean 'properties'?" in err

    def test_unknown_key_is_a_warning_not_an_error(self, tmp_path):
        """Blueprints in the wild carry stray keys; the build must still run."""

        def mutate(bp):
            bp["nodes"]["Person"]["comment"] = "written by the ETL job"

        g = kglite.from_blueprint(str(self._bp_with(tmp_path, mutate)), save=False)
        assert len(list(g.cypher("MATCH (p:Person) RETURN p.name"))) == 3

    def test_unknown_junction_edge_key_warns(self, tmp_path, capfd):
        def mutate(bp):
            junc = bp["nodes"]["Person"]["connections"]["junction_edges"]["KNOWS"]
            junc["propertie_types"] = {}

        kglite.from_blueprint(str(self._bp_with(tmp_path, mutate)), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "junction 'KNOWS'" in err
        assert "unknown key 'propertie_types'" in err

    def test_unknown_fk_edge_key_warns(self, tmp_path, capfd):
        def mutate(bp):
            bp["nodes"]["Person"]["connections"]["fk_edges"] = {
                "LIVES_IN": {"target": "Person", "fk": "person_id", "targt": "City"}
            }

        kglite.from_blueprint(str(self._bp_with(tmp_path, mutate)), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "fk_edge 'LIVES_IN'" in err
        assert "unknown key 'targt'" in err

    def test_unknown_sub_node_key_warns(self, tmp_path, capfd):
        def mutate(bp):
            bp["nodes"]["Person"]["sub_nodes"] = {"Nickname": {"pk": "name", "parent_fq": "person_id"}}

        kglite.from_blueprint(str(self._bp_with(tmp_path, mutate)), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "node 'Nickname'" in err
        assert "unknown key 'parent_fq'" in err

    def test_a_clean_blueprint_says_nothing(self, tmp_path, capfd):
        kglite.from_blueprint(str(_minimal_blueprint(tmp_path)), verbose=True, save=False)
        assert "unknown key" not in capfd.readouterr().err


class TestListColumnType:
    """`"list"` / `"array"` declares a JSON-array column. Without it, a
    blueprint that stores multi-valued cells (synonyms, aliases, cross-refs)
    has no way to say so: the value lands as one opaque string and every
    downstream query that wants membership has to parse it again.
    """

    def _bp(self, tmp_path, synonyms, declared="list", junction_prop_type=None):
        genes = pd.DataFrame({"gene_id": [1, 2], "name": ["adhE", "pfkA"]})
        genes["synonyms"] = synonyms
        _write_csv(tmp_path / "genes.csv", genes)
        pathways = pd.DataFrame({"pathway_id": [10, 20], "label": ["glycolysis", "fermentation"]})
        _write_csv(tmp_path / "pathways.csv", pathways)
        member = pd.DataFrame(
            {
                "gene_id": [1, 2],
                "pathway_id": [10, 20],
                "evidence": ['["IDA","IMP"]', '["ISS"]'],
            }
        )
        _write_csv(tmp_path / "member.csv", member)

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Gene": {
                    "csv": "genes.csv",
                    "pk": "gene_id",
                    "title": "name",
                    "properties": {"synonyms": declared},
                    "connections": {
                        "junction_edges": {
                            "MEMBER_OF": {
                                "csv": "member.csv",
                                "source_fk": "gene_id",
                                "target": "Pathway",
                                "target_fk": "pathway_id",
                                "properties": ["evidence"],
                                "property_types": ({"evidence": junction_prop_type} if junction_prop_type else {}),
                            }
                        }
                    },
                },
                "Pathway": {"csv": "pathways.csv", "pk": "pathway_id", "title": "label"},
            },
        }
        bp_path = tmp_path / "bp.json"
        _write_blueprint(bp_path, bp)
        return bp_path

    def test_declared_list_column_lands_as_a_list(self, tmp_path):
        bp_path = self._bp(tmp_path, ['["adhC","ADHE"]', '["pfk1"]'])
        g = kglite.from_blueprint(str(bp_path), save=False)
        rows = g.cypher("MATCH (n:Gene) RETURN n.name AS name, n.synonyms AS syn ORDER BY name")
        assert [r["syn"] for r in rows] == [["adhC", "ADHE"], ["pfk1"]]

    def test_list_column_is_queryable_element_wise(self, tmp_path):
        bp_path = self._bp(tmp_path, ['["adhC","ADHE"]', '["pfk1"]'])
        g = kglite.from_blueprint(str(bp_path), save=False)
        rows = g.cypher("MATCH (n:Gene) WHERE 'ADHE' IN n.synonyms RETURN n.name AS name")
        assert [r["name"] for r in rows] == ["adhE"]

    def test_array_is_the_same_keyword(self, tmp_path):
        bp_path = self._bp(tmp_path, ['["adhC"]', '["pfk1"]'], declared="array")
        g = kglite.from_blueprint(str(bp_path), save=False)
        rows = g.cypher("MATCH (n:Gene) RETURN n.synonyms AS syn ORDER BY n.name")
        assert [r["syn"] for r in rows] == [["adhC"], ["pfk1"]]

    def test_list_is_not_an_unknown_property_type(self, tmp_path, capfd):
        bp_path = self._bp(tmp_path, ['["adhC"]', '["pfk1"]'])
        kglite.from_blueprint(str(bp_path), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "does not rename columns" not in err

    def test_junction_property_types_accepts_list(self, tmp_path):
        bp_path = self._bp(tmp_path, ['["adhC"]', '["pfk1"]'], junction_prop_type="list")
        g = kglite.from_blueprint(str(bp_path), save=False)
        rows = g.cypher(
            "MATCH (:Gene)-[r:MEMBER_OF]->(:Pathway) RETURN r.evidence AS ev ORDER BY size(r.evidence) DESC"
        )
        assert [r["ev"] for r in rows] == [["IDA", "IMP"], ["ISS"]]

    def test_separator_in_a_non_json_cell_warns(self, tmp_path, capfd):
        """A `list` column whose cell is `a|b` is not a JSON array. It is
        wrapped as a one-element list holding the whole string — a plausible
        wrong answer, because the author plainly meant two values. Say so,
        naming the column, the row and the cell."""
        bp_path = self._bp(tmp_path, ["adhC|ADHE", '["pfk1"]'])
        kglite.from_blueprint(str(bp_path), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "synonyms" in err
        assert "row 1" in err
        assert "adhC|ADHE" in err

    def test_a_plain_scalar_cell_does_not_warn(self, tmp_path, capfd):
        """No separator, no ambiguity: a lone token is a one-element list and
        the warning would be noise."""
        bp_path = self._bp(tmp_path, ["adhC", '["pfk1"]'])
        kglite.from_blueprint(str(bp_path), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "not a JSON array" not in err


class TestJunctionChunkInvariance:
    """The junction loader streams its CSV in chunks
    (`KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE`, default 100k rows) purely to bound
    peak RAM. The chunk size is a performance knob, so the graph it produces
    must not depend on it: a parallel edge (same endpoints and type, different
    properties) that survives a one-chunk load has to survive a many-chunk one.
    Regression for the silent row loss found downstream 2026-09-02, where a
    117,160-row junction CSV landed 109,065 edges at the default chunk size and
    all 117,160 with the chunking effectively disabled.
    """

    ROWS = 25

    def _bp(self, tmp_path):
        persons = pd.DataFrame({"person_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        _write_csv(tmp_path / "persons.csv", persons)
        # Three endpoint pairs cycling over 25 rows: every pair recurs many
        # times, and its repeats straddle every boundary a chunk size of 10
        # draws. Each row carries a distinct `seq`, so a folded row is visible
        # as a missing value, not just a missing count.
        pairs = [(1, 2), (2, 3), (3, 1)]
        knows = pd.DataFrame(
            {
                "source_id": [pairs[i % 3][0] for i in range(self.ROWS)],
                "target_id": [pairs[i % 3][1] for i in range(self.ROWS)],
                "seq": list(range(self.ROWS)),
            }
        )
        _write_csv(tmp_path / "knows.csv", knows)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "properties": {},
                    "connections": {
                        "junction_edges": {
                            "KNOWS": {
                                "csv": "knows.csv",
                                "source_fk": "source_id",
                                "target": "Person",
                                "target_fk": "target_id",
                                "properties": ["seq"],
                                "property_types": {"seq": "int"},
                            }
                        }
                    },
                }
            },
        }
        bp_path = tmp_path / "blueprint.json"
        _write_blueprint(bp_path, bp)
        return bp_path

    def _build(self, bp_path, monkeypatch, chunk_size):
        monkeypatch.setenv("KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE", str(chunk_size))
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN r.seq AS seq").to_list()
        return sorted(r["seq"] for r in rows)

    def test_parallel_edges_survive_the_chunk_boundary(self, tmp_path, monkeypatch):
        bp_path = self._bp(tmp_path)
        chunked = self._build(bp_path, monkeypatch, 10)
        assert chunked == list(range(self.ROWS))

    def test_edge_set_is_the_same_at_every_chunk_size(self, tmp_path, monkeypatch):
        bp_path = self._bp(tmp_path)
        single = self._build(bp_path, monkeypatch, 1000)
        for chunk_size in (1, 7, 10, 24):
            assert self._build(bp_path, monkeypatch, chunk_size) == single, (
                f"chunk_size={chunk_size} produced a different edge set"
            )


class TestOntologyGate:
    def _bp(self, tmp_path, enforcement):
        persons = pd.DataFrame({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        _write_csv(tmp_path / "persons.csv", persons)
        classes = pd.DataFrame({"class_id": [7], "cname": ["Math"]})
        _write_csv(tmp_path / "classes.csv", classes)
        enrolled = pd.DataFrame({"source_id": [1], "target_id": [7]})
        _write_csv(tmp_path / "enrolled.csv", enrolled)
        (tmp_path / "school.ontology.json").write_text(
            json.dumps(
                {
                    "relationships": {
                        "ENROLLED_IN": {
                            "domain": "Person",
                            "range": "Class",
                            # Bob has no enrollment -> 1 violation
                            "required": True,
                            "enforcement": enforcement,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        bp = {
            "settings": {"root": str(tmp_path), "output": "out.kgl"},
            "ontology": "school.ontology.json",
            "nodes": {
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "connections": {
                        "junction_edges": {
                            "ENROLLED_IN": {
                                "csv": "enrolled.csv",
                                "source_fk": "source_id",
                                "target": "Class",
                                "target_fk": "target_id",
                            }
                        }
                    },
                },
                "Class": {"csv": "classes.csv", "pk": "class_id", "title": "cname"},
            },
        }
        bp_path = tmp_path / "blueprint.json"
        _write_blueprint(bp_path, bp)
        return bp_path

    def test_error_enforcement_fails_build_and_writes_nothing(self, tmp_path):
        bp_path = self._bp(tmp_path, "error")
        with pytest.raises(ValueError, match="ontology gate failed"):
            kglite.from_blueprint(str(bp_path))
        assert not (tmp_path / "out.kgl").exists()

    def test_warn_enforcement_builds_and_reports(self, tmp_path, capfd):
        bp_path = self._bp(tmp_path, "warn")
        graph = kglite.from_blueprint(str(bp_path), save=False, verbose=True)
        err = capfd.readouterr().err
        assert "ENROLLED_IN.required: 1/2" in err
        # The ontology itself is installed and persisted with the graph.
        assert graph.ontology()["relationships"]["ENROLLED_IN"]["required"] is True

    def test_advisory_enforcement_stays_silent(self, tmp_path, capfd):
        bp_path = self._bp(tmp_path, "advisory")
        graph = kglite.from_blueprint(str(bp_path), save=False, verbose=True)
        err = capfd.readouterr().err
        assert "ENROLLED_IN.required" not in err
        # Still available on demand.
        rows = graph.cypher("CALL ontology_audit() YIELD rule, violations RETURN rule, violations").to_list()
        assert {r["rule"]: r["violations"] for r in rows}["ENROLLED_IN.required"] == 1

    def _bp_exempt(self, tmp_path, exempt=None):
        """Two source types on one relationship; only one can carry `since`."""
        _write_csv(tmp_path / "persons.csv", pd.DataFrame({"person_id": [1, 2], "name": ["Alice", "Bob"]}))
        _write_csv(tmp_path / "legacy.csv", pd.DataFrame({"legacy_id": [5], "lname": ["Old"]}))
        _write_csv(tmp_path / "classes.csv", pd.DataFrame({"class_id": [7], "cname": ["Math"]}))
        _write_csv(
            tmp_path / "enrolled.csv",
            pd.DataFrame({"source_id": [1, 2], "target_id": [7, 7], "since": ["2020-01-01", "2021-01-01"]}),
        )
        # Legacy rows genuinely have no `since` — the permanent, legitimate
        # violation the operator wants counted separately.
        _write_csv(tmp_path / "legacy_enrolled.csv", pd.DataFrame({"source_id": [5], "target_id": [7]}))
        rel = {"required_properties": ["since"], "enforcement": "error"}
        if exempt is not None:
            rel["exempt"] = exempt
        (tmp_path / "school.ontology.json").write_text(
            json.dumps(
                {
                    "classes": {"Person": {}, "Legacy": {}, "Class": {}},
                    "relationships": {"ENROLLED_IN": rel},
                }
            ),
            encoding="utf-8",
        )
        bp = {
            "settings": {"root": str(tmp_path), "output": "out.kgl"},
            "ontology": "school.ontology.json",
            "nodes": {
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "connections": {
                        "junction_edges": {
                            "ENROLLED_IN": {
                                "csv": "enrolled.csv",
                                "source_fk": "source_id",
                                "target": "Class",
                                "target_fk": "target_id",
                                "properties": ["since"],
                            }
                        }
                    },
                },
                "Legacy": {
                    "csv": "legacy.csv",
                    "pk": "legacy_id",
                    "title": "lname",
                    "connections": {
                        "junction_edges": {
                            "ENROLLED_IN": {
                                "csv": "legacy_enrolled.csv",
                                "source_fk": "source_id",
                                "target": "Class",
                                "target_fk": "target_id",
                            }
                        }
                    },
                },
                "Class": {"csv": "classes.csv", "pk": "class_id", "title": "cname"},
            },
        }
        bp_path = tmp_path / "blueprint.json"
        _write_blueprint(bp_path, bp)
        return bp_path

    def test_unexempted_source_class_fails_the_error_gate(self, tmp_path):
        bp_path = self._bp_exempt(tmp_path)
        with pytest.raises(ValueError, match="ontology gate failed"):
            kglite.from_blueprint(str(bp_path))

    def test_exempted_source_class_passes_and_reports_the_tail(self, tmp_path, capfd):
        bp_path = self._bp_exempt(tmp_path, {"required_properties": ["Legacy"]})
        graph = kglite.from_blueprint(str(bp_path), save=False, verbose=True)
        err = capfd.readouterr().err
        # Passing the gate must not hide the exempted rows: a zero-violation
        # line with an exempted tail is not the same as a clean graph.
        assert "ENROLLED_IN.required_properties: 0/3 (0.0%) violations (+1 exempted)" in err
        row = graph.cypher(
            "CALL ontology_audit() YIELD rule, violations, exempted RETURN rule, violations, exempted"
        ).to_list()
        by_rule = {r["rule"]: r for r in row}
        assert by_rule["ENROLLED_IN.required_properties"]["violations"] == 0
        assert by_rule["ENROLLED_IN.required_properties"]["exempted"] == 1

    def test_clean_contract_passes_error_enforcement(self, tmp_path):
        bp_path = self._bp(tmp_path, "error")
        # Enroll Bob too -> contract satisfied -> error-level gate passes.
        enrolled = pd.DataFrame({"source_id": [1, 2], "target_id": [7, 7]})
        _write_csv(tmp_path / "enrolled.csv", enrolled)
        graph = kglite.from_blueprint(str(bp_path), save=False)
        assert graph.shape[0] == 3
