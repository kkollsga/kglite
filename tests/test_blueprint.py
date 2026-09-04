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

    def test_two_manual_types_over_three_fk_edges_on_one_csv(self, tmp_path):
        """Two manual types fed by three FK edges of the same source CSV.

        The manual phase reads each referring input once and harvests every
        manual-target FK column from that one table; this pins the graph that
        pass must still produce.
        """
        items = pd.DataFrame(
            {
                "item_id": [1, 2, 3],
                "owner": ["ann", "bob", "ann"],
                "co_owner": ["bob", "cara", ""],
                "tag": ["red", "blue", "red"],
            }
        )
        _write_csv(tmp_path / "items.csv", items)

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Owner": {"pk": "name", "title": "name", "properties": {}, "skipped": []},
                "Tag": {"pk": "name", "title": "name", "properties": {}, "skipped": []},
                "Item": {
                    "csv": "items.csv",
                    "pk": "item_id",
                    "properties": {},
                    "skipped": ["owner", "co_owner", "tag"],
                    "connections": {
                        "fk_edges": {
                            "OWNED_BY": {"target": "Owner", "fk": "owner"},
                            "CO_OWNED_BY": {"target": "Owner", "fk": "co_owner"},
                            "TAGGED": {"target": "Tag", "fk": "tag"},
                        }
                    },
                },
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)

        owners = [r["o.title"] for r in graph.cypher("MATCH (o:Owner) RETURN o.title ORDER BY o.title")]
        assert owners == ["ann", "bob", "cara"]
        tags = [r["t.title"] for r in graph.cypher("MATCH (t:Tag) RETURN t.title ORDER BY t.title")]
        assert tags == ["blue", "red"]
        counts = {
            e: graph.cypher(f"MATCH ()-[r:{e}]->() RETURN count(r) AS c").to_list()[0]["c"]
            for e in ("OWNED_BY", "CO_OWNED_BY", "TAGGED")
        }
        # The empty co_owner cell is a null FK, so that row yields no edge.
        assert counts == {"OWNED_BY": 3, "CO_OWNED_BY": 2, "TAGGED": 3}


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


class TestFkEdgeProperties:
    """`properties` / `property_types` / `rename` on an fk_edge, mirroring a
    junction edge. The columns come from the source node's own CSV row, so the
    property values must follow the rows that survived the null-FK drop."""

    def _bp(self, tmp_path, edge_extra=None, employees=None, skipped=None):
        companies = pd.DataFrame({"company_id": [10, 20], "name": ["Acme", "Globex"]})
        if employees is None:
            employees = pd.DataFrame(
                {
                    "employee_id": [1, 2, 3],
                    "name": ["Alice", "Bob", "Charlie"],
                    "company_id": [10, 20, 10],
                    "since": ["2001-01-01", "2002-02-02", "2003-03-03"],
                    "role": ["Lead", "Member", "Member"],
                    "level": [1, 2, 3],
                }
            )
        _write_csv(tmp_path / "companies.csv", companies)
        _write_csv(tmp_path / "employees.csv", employees)
        edge = {"target": "Company", "fk": "company_id"}
        if edge_extra:
            edge.update(edge_extra)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Company": {"csv": "companies.csv", "pk": "company_id", "title": "name"},
                "Employee": {
                    "csv": "employees.csv",
                    "pk": "employee_id",
                    "title": "name",
                    "skipped": skipped if skipped is not None else ["company_id"],
                    "connections": {"fk_edges": {"WORKS_AT": edge}},
                },
            },
        }
        bp_path = tmp_path / "bp.json"
        _write_blueprint(bp_path, bp)
        return bp_path

    def _edges(self, graph):
        return graph.cypher(
            "MATCH (e:Employee)-[r:WORKS_AT]->(c:Company) "
            "RETURN e.name AS who, r.role AS role, r.since AS since ORDER BY who"
        ).to_list()

    def test_declared_properties_land_on_the_edge(self, tmp_path):
        bp_path = self._bp(tmp_path, {"properties": ["role", "since"]})
        rows = self._edges(from_blueprint(bp_path, save=False))
        assert [r["who"] for r in rows] == ["Alice", "Bob", "Charlie"]
        assert [r["role"] for r in rows] == ["Lead", "Member", "Member"]
        assert [r["since"] for r in rows] == ["2001-01-01", "2002-02-02", "2003-03-03"]

    def test_undeclared_columns_stay_off_the_edge(self, tmp_path):
        bp_path = self._bp(tmp_path, {"properties": ["role"]})
        rows = self._edges(from_blueprint(bp_path, save=False))
        assert [r["role"] for r in rows] == ["Lead", "Member", "Member"]
        assert all(r["since"] is None for r in rows)

    def test_property_types_declare_the_column_type(self, tmp_path):
        """Inference would make `level` an int; the declaration wins."""
        bp_path = self._bp(
            tmp_path,
            {"properties": ["level"], "property_types": {"level": "string"}},
        )
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher(
            "MATCH (e:Employee)-[r:WORKS_AT]->(:Company) RETURN r.level AS lvl ORDER BY e.name"
        ).to_list()
        assert [r["lvl"] for r in rows] == ["1", "2", "3"]

    def test_rename_lands_the_property_under_the_new_name(self, tmp_path):
        bp_path = self._bp(
            tmp_path,
            {"properties": ["since"], "rename": {"since": "validFrom"}},
        )
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher(
            "MATCH (:Employee)-[r:WORKS_AT]->(:Company) RETURN r.validFrom AS vf, r.since AS old"
        ).to_list()
        assert len(rows) == 3
        assert all(r["vf"] is not None for r in rows)
        assert all(r["old"] is None for r in rows)

    def test_rename_of_the_fk_column_reports_an_error(self, tmp_path, capfd):
        bp_path = self._bp(
            tmp_path,
            {"properties": ["role"], "rename": {"company_id": "cid"}},
        )
        graph = from_blueprint(bp_path, save=False, verbose=True)
        assert "rename of fk column" in capfd.readouterr().err
        assert graph.cypher("MATCH ()-[r:WORKS_AT]->() RETURN count(r) AS c").to_list()[0]["c"] == 0

    def test_rename_key_must_be_a_declared_property(self, tmp_path, capfd):
        bp_path = self._bp(tmp_path, {"properties": ["role"], "rename": {"nosuch": "x"}})
        from_blueprint(bp_path, save=False, verbose=True)
        assert "not in 'properties'" in capfd.readouterr().err

    def test_a_property_column_absent_from_the_csv_is_reported(self, tmp_path, capfd):
        bp_path = self._bp(tmp_path, {"properties": ["role", "bonus"]})
        graph = from_blueprint(bp_path, save=False)
        err = capfd.readouterr().err
        assert "bonus" in err
        # The edge still builds, carrying the properties that do exist.
        rows = self._edges(graph)
        assert [r["role"] for r in rows] == ["Lead", "Member", "Member"]

    def test_null_fk_rows_drop_with_their_properties(self, tmp_path):
        """The row Bob sits on has no company; his `role` must not slide onto
        Charlie's edge."""
        employees = pd.DataFrame(
            {
                "employee_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "company_id": [10, None, 20],
                "since": ["2001-01-01", "2002-02-02", "2003-03-03"],
                "role": ["Lead", "Ghost", "Member"],
                "level": [1, 2, 3],
            }
        )
        bp_path = self._bp(tmp_path, {"properties": ["role", "since"]}, employees=employees)
        rows = self._edges(from_blueprint(bp_path, save=False))
        assert [(r["who"], r["role"], r["since"]) for r in rows] == [
            ("Alice", "Lead", "2001-01-01"),
            ("Charlie", "Member", "2003-03-03"),
        ]

    def test_declaring_a_property_does_not_skip_it_from_the_node(self, tmp_path):
        """The two landings are independent: `properties` copies the column onto
        the edge, `skipped` is what keeps it off the node."""
        bp_path = self._bp(tmp_path, {"properties": ["role"]}, skipped=["company_id"])
        graph = from_blueprint(bp_path, save=False)
        nodes = graph.cypher("MATCH (e:Employee) RETURN e.role AS role ORDER BY e.name").to_list()
        assert [r["role"] for r in nodes] == ["Lead", "Member", "Member"]
        edges = self._edges(graph)
        assert [r["role"] for r in edges] == ["Lead", "Member", "Member"]

    def test_a_self_reference_edge_carries_properties(self, tmp_path):
        """`fk == pk` synthesises the target column; the property columns must
        still line up with the rows behind it."""
        bp_path = self._bp(
            tmp_path,
            {"target": "Employee", "fk": "employee_id", "properties": ["role"]},
        )
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher(
            "MATCH (a:Employee)-[r:WORKS_AT]->(b:Employee) RETURN a.name AS who, r.role AS role ORDER BY who"
        ).to_list()
        assert [(r["who"], r["role"]) for r in rows] == [
            ("Alice", "Lead"),
            ("Bob", "Member"),
            ("Charlie", "Member"),
        ]

    def test_streamed_and_buffered_agree(self, tmp_path, monkeypatch):
        employees = pd.DataFrame(
            {
                "employee_id": list(range(1, 41)),
                "name": [f"E{i:02d}" for i in range(1, 41)],
                "company_id": [10 if i % 2 else 20 for i in range(1, 41)],
                "since": [f"20{i:02d}-01-01" for i in range(1, 41)],
                "role": [f"R{i}" for i in range(1, 41)],
                "level": list(range(1, 41)),
            }
        )
        bp_path = self._bp(
            tmp_path,
            {"properties": ["role", "since"], "rename": {"since": "validFrom"}},
            employees=employees,
        )

        def build(threshold_mb, chunk_size):
            monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", str(threshold_mb))
            monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", str(chunk_size))
            graph = from_blueprint(bp_path, save=False)
            return graph.cypher(
                "MATCH (e:Employee)-[r:WORKS_AT]->(:Company) "
                "RETURN e.name AS who, r.role AS role, r.validFrom AS vf ORDER BY who"
            ).to_list()

        buffered = build(100, 1000)
        assert len(buffered) == 40
        assert buffered[0]["role"] == "R1"
        for chunk_size in (7, 13, 40):
            assert build(0, chunk_size) == buffered, f"chunk_size={chunk_size} changed the edge properties"

    def test_streamed_null_fk_rows_drop_with_their_properties(self, tmp_path, monkeypatch):
        """`test_null_fk_rows_drop_with_their_properties` is buffered-only and
        `test_streamed_and_buffered_agree` has no nulls, so the streaming
        loader's own row-subset — indices into a *chunk*, not into the file —
        was never exercised with a hole in it. A row's properties sliding onto
        the next surviving row is a silent wrong answer, not a crash."""
        n = 40
        employees = pd.DataFrame(
            {
                "employee_id": list(range(1, n + 1)),
                "name": [f"E{i:02d}" for i in range(1, n + 1)],
                # Every third row has no company: the holes fall at different
                # offsets in every chunk size below.
                "company_id": [None if i % 3 == 0 else (10 if i % 2 else 20) for i in range(1, n + 1)],
                "since": [f"20{i:02d}-01-01" for i in range(1, n + 1)],
                "role": [f"R{i}" for i in range(1, n + 1)],
                "level": list(range(1, n + 1)),
            }
        )
        bp_path = self._bp(tmp_path, {"properties": ["role", "since"]}, employees=employees)

        def build(threshold_mb, chunk_size):
            monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", str(threshold_mb))
            monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", str(chunk_size))
            return self._edges(from_blueprint(bp_path, save=False))

        buffered = build(100, 1000)
        # Each surviving edge still carries its own row's property values.
        assert [(r["who"], r["role"]) for r in buffered] == [(f"E{i:02d}", f"R{i}") for i in range(1, n + 1) if i % 3]
        for chunk_size in (7, 13, 40):
            assert build(0, chunk_size) == buffered, f"chunk_size={chunk_size} moved the edge properties"

    def _two_source_bp(self, tmp_path, with_properties):
        """Two node types writing the same relationship type, each with repeated
        (source, target) rows. Only the *first* add_connections call per
        relationship keeps parallel edges, so anything that reorders or resizes
        the FK frames shows up as a different count here."""
        _write_csv(tmp_path / "companies.csv", pd.DataFrame({"company_id": [10], "name": ["Acme"]}))
        for csv_name, id_col in (("employees.csv", "employee_id"), ("contractors.csv", "contractor_id")):
            _write_csv(
                tmp_path / csv_name,
                pd.DataFrame(
                    {
                        id_col: [1, 1, 1],
                        "name": ["X", "X", "X"],
                        "company_id": [10, 10, 10],
                        "role": ["a", "b", "c"],
                    }
                ),
            )
        edge = {"target": "Company", "fk": "company_id"}
        if with_properties:
            edge = dict(edge, properties=["role"])
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Company": {"csv": "companies.csv", "pk": "company_id", "title": "name"},
                "Employee": {
                    "csv": "employees.csv",
                    "pk": "employee_id",
                    "title": "name",
                    "connections": {"fk_edges": {"WORKS_AT": dict(edge)}},
                },
                "Contractor": {
                    "csv": "contractors.csv",
                    "pk": "contractor_id",
                    "title": "name",
                    "connections": {"fk_edges": {"WORKS_AT": dict(edge)}},
                },
            },
        }
        bp_path = tmp_path / ("bp_props.json" if with_properties else "bp_plain.json")
        _write_blueprint(bp_path, bp)
        return bp_path

    def test_declaring_properties_does_not_change_parallel_edge_counts(self, tmp_path):
        def count(with_properties):
            graph = from_blueprint(self._two_source_bp(tmp_path, with_properties), save=False)
            return graph.cypher("MATCH ()-[r:WORKS_AT]->(:Company) RETURN count(r) AS n").to_list()[0]["n"]

        plain = count(False)
        assert plain == 4, f"expected 3 parallel edges from the first spec + 1 deduped, got {plain}"
        assert count(True) == plain

        graph = from_blueprint(self._two_source_bp(tmp_path, True), save=False)
        roles = graph.cypher("MATCH ()-[r:WORKS_AT]->(:Company) RETURN r.role AS role").to_list()
        assert sum(1 for r in roles if r["role"] is not None) == plain


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

    def test_a_well_formed_array_does_not_warn_for_its_own_commas(self, tmp_path, capfd):
        """The separator check runs only on cells that are not JSON. Drop that
        guard and every multi-element list column warns about the commas
        inside its own valid arrays — and no other test in this class reads
        stderr on a well-formed multi-element cell."""
        bp_path = self._bp(tmp_path, ['["adhC","ADHE"]', '["pfk1"]'])
        kglite.from_blueprint(str(bp_path), verbose=True, save=False)
        assert "not a JSON array" not in capfd.readouterr().err

    def test_a_plain_scalar_cell_does_not_warn(self, tmp_path, capfd):
        """No separator, no ambiguity: a lone token is a one-element list and
        the warning would be noise."""
        bp_path = self._bp(tmp_path, ["adhC", '["pfk1"]'])
        kglite.from_blueprint(str(bp_path), verbose=True, save=False)
        err = capfd.readouterr().err
        assert "not a JSON array" not in err

    def _row_id_bp(self, tmp_path, filt=None):
        """Six genes with the one malformed cell at *file* row 5. Everything the
        loader does to the row vector between reading and typing — filtering,
        chunking — must leave the reported row number the one the author reads
        in the CSV, or the warning sends them to an innocent line."""
        genes = pd.DataFrame(
            {
                "gene_id": [1, 2, 3, 4, 5, 6],
                "name": [f"g{i}" for i in range(1, 7)],
                "keep": ["no", "no", "no", "no", "yes", "yes"],
                "synonyms": ['["a"]', '["b"]', '["c"]', '["d"]', "adhC|ADHE", '["f"]'],
            }
        )
        _write_csv(tmp_path / "genes.csv", genes)
        spec = {
            "csv": "genes.csv",
            "pk": "gene_id",
            "title": "name",
            "properties": {"synonyms": "list"},
        }
        if filt:
            spec["filter"] = filt
        bp = {"settings": {"root": str(tmp_path)}, "nodes": {"Gene": spec}}
        bp_path = tmp_path / "bp.json"
        _write_blueprint(bp_path, bp)
        return bp_path

    def test_the_reported_row_survives_a_filter(self, tmp_path, capfd):
        """`filter` drops the four rows above the bad cell, leaving it at index
        1 of the surviving rows. The warning must still say row 5."""
        bp_path = self._row_id_bp(tmp_path, filt={"keep": "yes"})
        g = kglite.from_blueprint(str(bp_path), verbose=True, save=False)
        assert g.cypher("MATCH (n:Gene) RETURN count(n) AS n").to_list()[0]["n"] == 2
        err = capfd.readouterr().err
        assert "First at row 5:" in err, err

    def test_the_reported_row_survives_chunking(self, tmp_path, capfd, monkeypatch):
        """Streamed two rows to a chunk, the bad cell is the first row of the
        third chunk; a chunk-local counter would call it row 1. The tally is
        also per-CSV, so the column gets one line, not one per chunk."""
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
        monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", "2")
        bp_path = self._row_id_bp(tmp_path)
        g = kglite.from_blueprint(str(bp_path), verbose=True, save=False)
        assert g.cypher("MATCH (n:Gene) RETURN count(n) AS n").to_list()[0]["n"] == 6
        err = capfd.readouterr().err
        assert err.count("not a JSON array") == 1, err
        assert "1 cell(s)" in err, err
        assert "First at row 5:" in err, err


class TestJunctionUnionTargets:
    """One junction relationship over a union of target types. A relation whose
    range is an abstract class (Disease | Phenotype | Exposure) otherwise needs
    one relationship name per concrete type, which no query and no ontology
    declaration can put back together."""

    TARGET_TYPES = ["Disease", "Phenotype", "Exposure"]

    def _bp(self, tmp_path, target, type_column=None, links=None, properties=None):
        _write_csv(tmp_path / "microbes.csv", pd.DataFrame({"id": ["M1", "M2"], "name": ["Mi1", "Mi2"]}))
        _write_csv(tmp_path / "diseases.csv", pd.DataFrame({"id": ["D1"], "name": ["Dis1"]}))
        _write_csv(tmp_path / "phenotypes.csv", pd.DataFrame({"id": ["P1"], "name": ["Phe1"]}))
        _write_csv(tmp_path / "exposures.csv", pd.DataFrame({"id": ["E1"], "name": ["Exp1"]}))
        if links is None:
            links = [
                {"source_id": "M1", "target_id": "D1", "target_type": "Disease", "score": 1},
                {"source_id": "M1", "target_id": "P1", "target_type": "Phenotype", "score": 2},
                {"source_id": "M2", "target_id": "E1", "target_type": "Exposure", "score": 3},
            ]
        _write_csv(tmp_path / "links.csv", pd.DataFrame(links))
        junction = {
            "csv": "links.csv",
            "source_fk": "source_id",
            "target": target,
            "target_fk": "target_id",
        }
        if type_column is not None:
            junction["target_type_column"] = type_column
        if properties is not None:
            junction["properties"] = properties
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Microbe": {
                    "csv": "microbes.csv",
                    "pk": "id",
                    "title": "name",
                    "connections": {"junction_edges": {"ASSOCIATED_WITH": junction}},
                },
                "Disease": {"csv": "diseases.csv", "pk": "id", "title": "name"},
                "Phenotype": {"csv": "phenotypes.csv", "pk": "id", "title": "name"},
                "Exposure": {"csv": "exposures.csv", "pk": "id", "title": "name"},
            },
        }
        bp_path = tmp_path / "bp.json"
        _write_blueprint(bp_path, bp)
        return bp_path

    def _landed(self, graph):
        return sorted(
            (r["src"], r["tgt"], r["t"])
            for r in graph.cypher(
                "MATCH (m:Microbe)-[:ASSOCIATED_WITH]->(x) RETURN m.id AS src, x.id AS tgt, head(labels(x)) AS t"
            ).to_list()
        )

    EXPECTED = [
        ("M1", "D1", "Disease"),
        ("M1", "P1", "Phenotype"),
        ("M2", "E1", "Exposure"),
    ]

    def test_a_list_target_routes_each_row_by_its_id(self, tmp_path):
        """No type column: the declared types are probed and the one that
        already has the id wins."""
        graph = from_blueprint(self._bp(tmp_path, self.TARGET_TYPES), save=False)
        assert self._landed(graph) == self.EXPECTED

    def test_target_type_column_routes_each_row_explicitly(self, tmp_path):
        graph = from_blueprint(self._bp(tmp_path, self.TARGET_TYPES, type_column="target_type"), save=False)
        assert self._landed(graph) == self.EXPECTED

    def test_the_routing_column_is_not_an_edge_property(self, tmp_path):
        graph = from_blueprint(self._bp(tmp_path, self.TARGET_TYPES, type_column="target_type"), save=False)
        rows = graph.cypher("MATCH ()-[r:ASSOCIATED_WITH]->() RETURN r.target_type AS t").to_list()
        assert all(r["t"] is None for r in rows)

    def test_the_routing_column_is_a_property_when_declared(self, tmp_path):
        graph = from_blueprint(
            self._bp(tmp_path, self.TARGET_TYPES, type_column="target_type", properties=["target_type"]),
            save=False,
        )
        rows = graph.cypher("MATCH ()-[r:ASSOCIATED_WITH]->() RETURN r.target_type AS t").to_list()
        assert sorted(r["t"] for r in rows) == ["Disease", "Exposure", "Phenotype"]

    def test_properties_survive_the_per_type_split(self, tmp_path):
        graph = from_blueprint(
            self._bp(tmp_path, self.TARGET_TYPES, type_column="target_type", properties=["score"]),
            save=False,
        )
        rows = graph.cypher("MATCH (m:Microbe)-[r:ASSOCIATED_WITH]->(x) RETURN x.id AS tgt, r.score AS s").to_list()
        assert sorted((r["tgt"], r["s"]) for r in rows) == [("D1", 1), ("E1", 3), ("P1", 2)]

    def test_a_row_naming_an_undeclared_type_is_reported_and_skipped(self, tmp_path, capfd):
        links = [
            {"source_id": "M1", "target_id": "D1", "target_type": "Disease", "score": 1},
            {"source_id": "M2", "target_id": "X1", "target_type": "Chemical", "score": 2},
        ]
        graph = from_blueprint(
            self._bp(tmp_path, self.TARGET_TYPES, type_column="target_type", links=links),
            save=False,
            verbose=True,
        )
        err = capfd.readouterr().err
        assert "Chemical" in err
        assert self._landed(graph) == [("M1", "D1", "Disease")]

    def test_an_id_no_declared_type_has_falls_to_the_first(self, tmp_path):
        """The probe cannot invent a type, and dropping the row would lose an
        edge — so the row takes the first declared type, where the existing
        missing-endpoint policy vivifies its stub."""
        links = [{"source_id": "M1", "target_id": "Z9", "target_type": "Disease", "score": 1}]
        graph = from_blueprint(self._bp(tmp_path, self.TARGET_TYPES, links=links), save=False)
        assert self._landed(graph) == [("M1", "Z9", "Disease")]

    def _numeric_bp(self, tmp_path):
        """The same union keyed by integer ids, and no type column — so every
        row is routed by probing the declared types' id indices."""
        _write_csv(tmp_path / "microbes.csv", pd.DataFrame({"id": [1, 2], "name": ["Mi1", "Mi2"]}))
        _write_csv(tmp_path / "diseases.csv", pd.DataFrame({"id": [10], "name": ["Dis1"]}))
        _write_csv(tmp_path / "phenotypes.csv", pd.DataFrame({"id": [20], "name": ["Phe1"]}))
        _write_csv(tmp_path / "exposures.csv", pd.DataFrame({"id": [30], "name": ["Exp1"]}))
        _write_csv(
            tmp_path / "links.csv",
            pd.DataFrame(
                [
                    {"source_id": 1, "target_id": 10},
                    {"source_id": 1, "target_id": 20},
                    {"source_id": 2, "target_id": 30},
                ]
            ),
        )
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Microbe": {
                    "csv": "microbes.csv",
                    "pk": "id",
                    "title": "name",
                    "connections": {
                        "junction_edges": {
                            "ASSOCIATED_WITH": {
                                "csv": "links.csv",
                                "source_fk": "source_id",
                                "target": self.TARGET_TYPES,
                                "target_fk": "target_id",
                            }
                        }
                    },
                },
                "Disease": {"csv": "diseases.csv", "pk": "id", "title": "name"},
                "Phenotype": {"csv": "phenotypes.csv", "pk": "id", "title": "name"},
                "Exposure": {"csv": "exposures.csv", "pk": "id", "title": "name"},
            },
        }
        bp_path = tmp_path / "bp_numeric.json"
        _write_blueprint(bp_path, bp)
        return bp_path

    def test_the_probe_routes_numeric_ids(self, tmp_path):
        """A numeric pk keys its id index by `Int64`, so a probe that compares
        the raw CSV cell matches nothing and every row falls to the *first*
        declared type — three edges, all on Disease, no error. Every other
        union test uses text ids, which is the blind spot the CSV-less
        numeric-FK defect sat in on this same branch."""
        graph = from_blueprint(self._numeric_bp(tmp_path), save=False)
        assert self._landed(graph) == [
            (1, 10, "Disease"),
            (1, 20, "Phenotype"),
            (2, 30, "Exposure"),
        ]
        # No stub was invented for a target the probe failed to place.
        assert graph.cypher("MATCH (n) WHERE n._provisional = true RETURN count(n) AS c").to_list()[0]["c"] == 0

    def test_chunking_does_not_change_the_routing(self, tmp_path, monkeypatch):
        bp_path = self._bp(tmp_path, self.TARGET_TYPES, type_column="target_type")

        def build(chunk_size):
            monkeypatch.setenv("KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE", str(chunk_size))
            return self._landed(from_blueprint(bp_path, save=False))

        for chunk_size in (1, 2, 3, 100):
            assert build(chunk_size) == self.EXPECTED, f"chunk_size={chunk_size}"

    def test_a_string_target_is_unchanged(self, tmp_path):
        links = [{"source_id": "M1", "target_id": "D1", "target_type": "Disease", "score": 1}]
        graph = from_blueprint(self._bp(tmp_path, "Disease", links=links), save=False)
        assert self._landed(graph) == [("M1", "D1", "Disease")]

    def test_a_routing_column_that_is_an_fk_is_an_error(self, tmp_path, capfd):
        graph = from_blueprint(self._bp(tmp_path, self.TARGET_TYPES, type_column="target_id"), save=False, verbose=True)
        assert "target_type_column" in capfd.readouterr().err
        assert self._landed(graph) == []

    def test_ontology_range_over_the_abstract_class_sees_no_violation(self, tmp_path):
        """The point of the union form: one relationship whose range is the
        abstract class, audited as one rule instead of three."""
        graph = from_blueprint(self._bp(tmp_path, self.TARGET_TYPES), save=False)
        graph.define_ontology(
            {
                "classes": {
                    "Microbe": {},
                    "Outcome": {"abstract": True},
                    "Disease": {"is_a": "Outcome"},
                    "Phenotype": {"is_a": "Outcome"},
                    "Exposure": {"is_a": "Outcome"},
                },
                "relationships": {"ASSOCIATED_WITH": {"domain": "Microbe", "range": "Outcome", "enforcement": "error"}},
            }
        )
        audit = {
            r["rule"]: r for r in graph.cypher("CALL ontology_audit() YIELD rule, violations, total RETURN *").to_list()
        }
        assert audit["ASSOCIATED_WITH.range"]["violations"] == 0
        assert audit["ASSOCIATED_WITH.range"]["total"] == 3


class TestUndeclaredTypeChunkInvariance:
    """A column with no declared type must get the same type whatever the
    chunk size.

    The streamed loaders used to run `infer_type` per chunk, so an undeclared
    column whose first rows are integers and whose later rows are text became
    Int64 in the early chunks and String in the late ones — the property type
    (and the value) depended on `KGLITE_BLUEPRINT_NODE_CHUNK_SIZE`, a knob
    documented as bounding memory only. Confirmed by the 2026-09-03 probe.
    """

    ROWS = 30

    def _bp(self, tmp_path):
        # `code` is int-shaped for the first 12 rows and text afterwards, so
        # any chunk boundary below 12 splits the two shapes; nothing declares
        # its type. `ref` is the same shape on the FK column.
        codes = [str(i) for i in range(12)] + [f"x{i}" for i in range(12, self.ROWS)]
        items = pd.DataFrame(
            {
                "item_id": list(range(self.ROWS)),
                "name": [f"Item_{i}" for i in range(self.ROWS)],
                "code": codes,
                "ref": codes,
            }
        )
        _write_csv(tmp_path / "items.csv", items)
        links = pd.DataFrame(
            {
                "src": list(range(self.ROWS)),
                "dst": [(i + 1) % self.ROWS for i in range(self.ROWS)],
                "note": codes,
            }
        )
        _write_csv(tmp_path / "links.csv", links)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Ref": {"pk": "name", "title": "name", "properties": {}, "skipped": []},
                "Item": {
                    "csv": "items.csv",
                    "pk": "item_id",
                    "title": "name",
                    # `code`, `ref` and the junction's `note` are deliberately
                    # undeclared — that is what the pre-pass has to resolve.
                    "properties": {},
                    "skipped": [],
                    "connections": {
                        "fk_edges": {"REFERS_TO": {"target": "Ref", "fk": "ref", "properties": ["code"]}},
                        "junction_edges": {
                            "LINKS_TO": {
                                "csv": "links.csv",
                                "source_fk": "src",
                                "target": "Item",
                                "target_fk": "dst",
                                "properties": ["note"],
                            }
                        },
                    },
                },
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        return tmp_path / "bp.json"

    def _load(self, bp_path, monkeypatch, chunk):
        if chunk is None:
            monkeypatch.delenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", raising=False)
            monkeypatch.delenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", raising=False)
            monkeypatch.delenv("KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE", raising=False)
        else:
            monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
            monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", str(chunk))
            monkeypatch.setenv("KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE", str(chunk))
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher("MATCH (i:Item) RETURN i.item_id AS id, i.code AS code ORDER BY id").to_list()
        edges = graph.cypher(
            "MATCH (a:Item)-[r:LINKS_TO]->(b:Item) RETURN a.item_id AS a, b.item_id AS b, r.note AS note ORDER BY a"
        ).to_list()
        fk = graph.cypher(
            "MATCH (i:Item)-[r:REFERS_TO]->(:Ref) RETURN i.item_id AS id, r.code AS code ORDER BY id"
        ).to_list()
        return rows, edges, fk

    @pytest.mark.parametrize("chunk", [5, 10, 100])
    def test_undeclared_column_type_does_not_move_with_the_chunk_size(self, tmp_path, monkeypatch, chunk):
        bp_path = self._bp(tmp_path)
        buffered = self._load(bp_path, monkeypatch, None)
        streamed = self._load(bp_path, monkeypatch, chunk)
        # Values AND their Python types: `1` and `"1"` compare unequal here,
        # which is the whole point.
        assert [(r["id"], r["code"], type(r["code"]).__name__) for r in buffered[0]] == [
            (r["id"], r["code"], type(r["code"]).__name__) for r in streamed[0]
        ]
        assert [(e["a"], e["b"], e["note"], type(e["note"]).__name__) for e in buffered[1]] == [
            (e["a"], e["b"], e["note"], type(e["note"]).__name__) for e in streamed[1]
        ]
        assert [(r["id"], r["code"], type(r["code"]).__name__) for r in buffered[2]] == [
            (r["id"], r["code"], type(r["code"]).__name__) for r in streamed[2]
        ]

    def test_the_prepass_warning_names_the_undeclared_columns(self, tmp_path, monkeypatch, capfd):
        """The extra read is visible and actionable, not silent."""
        bp_path = self._bp(tmp_path)
        monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
        monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", "10")
        monkeypatch.setenv("KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE", "10")
        from_blueprint(bp_path, save=False, verbose=True)
        err = capfd.readouterr().err
        assert "node 'Item'" in err and "code" in err
        assert "junction 'LINKS_TO'" in err and "note" in err
        assert "fk_edge properties (node 'Item')" in err
        assert "Declaring them keeps the type stable" in err


class TestFkIdTypeChunkInvariance:
    """An FK column's id type must not move with the chunk size either.

    `build_edge_df` typed the edge's endpoint columns from the values of the
    chunk in front of it: a column that is integer-shaped for its first rows
    and text afterwards came out `Int64` in the early chunks and `String` in
    the late ones. The target nodes carry one of those two types, so the other
    half of the rows matched nothing and vivified a *second*, provisional node
    per id — a silent duplicate, not a missing edge.
    """

    ROWS = 30

    def _bp(self, tmp_path):
        refs = [str(i) for i in range(12)] + [f"x{i}" for i in range(12, self.ROWS)]
        _write_csv(
            tmp_path / "items.csv",
            pd.DataFrame({"item_id": list(range(self.ROWS)), "name": [f"I{i}" for i in range(self.ROWS)], "ref": refs}),
        )
        _write_csv(
            tmp_path / "refs.csv",
            pd.DataFrame({"rid": refs, "rname": [f"R{r}" for r in refs]}),
        )
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Ref": {"csv": "refs.csv", "pk": "rid", "title": "rname", "properties": {}, "skipped": []},
                # A manual type is fed from the same FK values, so both the
                # loaded-target and synthesised-target routes are covered.
                "Manual": {"pk": "name", "title": "name", "properties": {}, "skipped": []},
                "Item": {
                    "csv": "items.csv",
                    "pk": "item_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "connections": {
                        "fk_edges": {
                            "REFERS_TO": {"target": "Ref", "fk": "ref"},
                            "MANUAL_REF": {"target": "Manual", "fk": "ref"},
                        }
                    },
                },
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        return tmp_path / "bp.json"

    def _load(self, bp_path, monkeypatch, chunk):
        for key in (
            "KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB",
            "KGLITE_BLUEPRINT_NODE_CHUNK_SIZE",
            "KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE",
        ):
            monkeypatch.delenv(key, raising=False)
        if chunk is not None:
            monkeypatch.setenv("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB", "0")
            monkeypatch.setenv("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE", str(chunk))
            monkeypatch.setenv("KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE", str(chunk))
        graph = from_blueprint(bp_path, save=False)

        def count(q):
            return graph.cypher(q).to_list()[0]["c"]

        return {
            "refs": count("MATCH (n:Ref) RETURN count(n) AS c"),
            "manual": count("MATCH (n:Manual) RETURN count(n) AS c"),
            "refers_to": count("MATCH (:Item)-[r:REFERS_TO]->(:Ref) RETURN count(r) AS c"),
            "manual_ref": count("MATCH (:Item)-[r:MANUAL_REF]->(:Manual) RETURN count(r) AS c"),
            "provisional": count("MATCH (n) WHERE n._provisional = true RETURN count(n) AS c"),
        }

    @pytest.mark.parametrize("chunk", [3, 10, 1000])
    def test_fk_id_type_does_not_move_with_the_chunk_size(self, tmp_path, monkeypatch, chunk):
        bp_path = self._bp(tmp_path)
        buffered = self._load(bp_path, monkeypatch, None)
        streamed = self._load(bp_path, monkeypatch, chunk)
        assert streamed == buffered
        # Non-vacuity: the fixture really does resolve every row to a real
        # node, so a divergence shows up as duplicates, not as a shared zero.
        assert buffered == {
            "refs": self.ROWS,
            "manual": self.ROWS,
            "refers_to": self.ROWS,
            "manual_ref": self.ROWS,
            "provisional": 0,
        }

    def test_junction_endpoint_id_type_does_not_move_with_the_chunk_size(self, tmp_path, monkeypatch):
        """The junction loader types its FK columns through the same
        whole-input pre-pass; this pins that it does, endpoint ids included."""
        refs = [str(i) for i in range(12)] + [f"x{i}" for i in range(12, self.ROWS)]
        _write_csv(
            tmp_path / "refs.csv",
            pd.DataFrame({"rid": refs, "rname": [f"R{r}" for r in refs]}),
        )
        _write_csv(
            tmp_path / "items.csv",
            pd.DataFrame({"item_id": list(range(self.ROWS)), "name": [f"I{i}" for i in range(self.ROWS)]}),
        )
        _write_csv(
            tmp_path / "links.csv",
            pd.DataFrame({"src": list(range(self.ROWS)), "dst": refs}),
        )
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Ref": {"csv": "refs.csv", "pk": "rid", "title": "rname", "properties": {}, "skipped": []},
                "Item": {
                    "csv": "items.csv",
                    "pk": "item_id",
                    "title": "name",
                    "properties": {},
                    "skipped": [],
                    "connections": {
                        "junction_edges": {
                            "LINKS_TO": {
                                "csv": "links.csv",
                                "source_fk": "src",
                                "target": "Ref",
                                "target_fk": "dst",
                            }
                        }
                    },
                },
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        bp_path = tmp_path / "bp.json"

        def load(chunk):
            for key in (
                "KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB",
                "KGLITE_BLUEPRINT_NODE_CHUNK_SIZE",
                "KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE",
            ):
                monkeypatch.delenv(key, raising=False)
            if chunk is not None:
                monkeypatch.setenv("KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE", str(chunk))
            graph = from_blueprint(bp_path, save=False)

            def count(q):
                return graph.cypher(q).to_list()[0]["c"]

            return {
                "refs": count("MATCH (n:Ref) RETURN count(n) AS c"),
                "links": count("MATCH (:Item)-[r:LINKS_TO]->(:Ref) RETURN count(r) AS c"),
                "provisional": count("MATCH (n) WHERE n._provisional = true RETURN count(n) AS c"),
            }

        whole = load(None)
        assert whole == {"refs": self.ROWS, "links": self.ROWS, "provisional": 0}
        for chunk in (3, 10, 1000):
            assert load(chunk) == whole, f"chunk={chunk}"


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


# ── P0 status-quo pins ───────────────────────────────────────────────
#
# Everything below records what the CSV loader does *today*, ahead of the
# producer-seam refactor that moves this code. A future input format (a
# delimited file, an xlsx sheet, a pandas frame) has to land on the same
# answers, so a diff here is a deliberate change of meaning — not a test
# detail.


def _write_text(path, text):
    """Write a CSV verbatim — the exact bytes matter to these pins."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


class TestGeometryProperties:
    """`geometry` / `location.lat` / `location.lon` through `from_blueprint`.

    `crates/kglite/src/graph/blueprint/geometry.rs` had no test at any level
    before this class: it parses the `_geometry` column's GeoJSON and writes
    WKT + centroid back into the columns the spec's *property types* name.
    """

    SQUARE = '{"type":"Polygon","coordinates":[[[0,0],[2,0],[2,2],[0,2],[0,0]]]}'
    POINT = '{"type":"Point","coordinates":[10.5,20.25]}'

    def _bp(self, tmp_path, rows, properties):
        _write_csv(tmp_path / "sites.csv", pd.DataFrame(rows))
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Site": {
                    "csv": "sites.csv",
                    "pk": "id",
                    "title": "name",
                    "properties": properties,
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        return tmp_path / "bp.json"

    def test_geojson_becomes_wkt_and_a_centroid(self, tmp_path):
        bp_path = self._bp(
            tmp_path,
            {
                "id": [1, 2],
                "name": ["Square", "Point"],
                "_geometry": [self.SQUARE, self.POINT],
            },
            {"wkt": "geometry", "lat": "location.lat", "lon": "location.lon"},
        )
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher(
            "MATCH (s:Site) RETURN s.id AS i, s.wkt AS w, s.lat AS lat, s.lon AS lon ORDER BY i"
        ).to_list()
        assert rows[0]["w"] == "POLYGON((0 0,2 0,2 2,0 2,0 0))"
        assert rows[0]["lat"] == pytest.approx(1.0)
        assert rows[0]["lon"] == pytest.approx(1.0)
        assert rows[1]["w"] == "POINT(10.5 20.25)"
        assert rows[1]["lat"] == pytest.approx(20.25)
        assert rows[1]["lon"] == pytest.approx(10.5)

    def test_the_geojson_source_column_is_never_a_property(self, tmp_path):
        bp_path = self._bp(
            tmp_path,
            {"id": [1], "name": ["Square"], "_geometry": [self.SQUARE]},
            {"wkt": "geometry"},
        )
        graph = from_blueprint(bp_path, save=False)
        props = graph.cypher("MATCH (s:Site) RETURN s").to_list()[0]["s"]["properties"]
        assert "_geometry" not in props
        assert props["wkt"] == "POLYGON((0 0,2 0,2 2,0 2,0 0))"

    def test_a_missing_or_unparseable_geometry_cell_is_null(self, tmp_path):
        bp_path = self._bp(
            tmp_path,
            {
                "id": [1, 2, 3],
                "name": ["ok", "empty", "junk"],
                "_geometry": [self.SQUARE, "", "not json"],
            },
            {"wkt": "geometry", "lat": "location.lat", "lon": "location.lon"},
        )
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher("MATCH (s:Site) RETURN s.id AS i, s.wkt AS w, s.lat AS lat ORDER BY i").to_list()
        assert rows[0]["w"] is not None
        # A bad cell is dropped silently — no warning, no failed build.
        assert rows[1]["w"] is None and rows[1]["lat"] is None
        assert rows[2]["w"] is None and rows[2]["lat"] is None

    def test_the_centroid_columns_are_inferred_not_declared(self, tmp_path):
        """`location.lat` is not a `ColumnType`, so the synthesised column
        goes through ordinary inference on its own text. Whole-number
        centroids therefore land as ints, not floats."""
        bp_path = self._bp(
            tmp_path,
            {"id": [1], "name": ["Square"], "_geometry": [self.SQUARE]},
            {"lat": "location.lat", "lon": "location.lon"},
        )
        graph = from_blueprint(bp_path, save=False)
        lat = graph.cypher("MATCH (s:Site) RETURN s.lat AS lat").to_list()[0]["lat"]
        assert isinstance(lat, int) and not isinstance(lat, bool)

    def test_without_a_geojson_column_a_wkt_column_passes_through(self, tmp_path):
        """Mode 2 of geometry.rs: the CSV already holds WKT, so nothing is
        converted and the column is stored as the string it is."""
        bp_path = self._bp(
            tmp_path,
            {"id": [1], "name": ["Square"], "wkt": ["POLYGON((0 0,1 0,1 1,0 0))"]},
            {"wkt": "geometry"},
        )
        graph = from_blueprint(bp_path, save=False)
        got = graph.cypher("MATCH (s:Site) RETURN s.wkt AS w").to_list()[0]["w"]
        assert got == "POLYGON((0 0,1 0,1 1,0 0))"


class TestCsvShapeQuirks:
    """Ragged rows, a UTF-8 BOM and CRLF endings, end to end."""

    def _bp(self, tmp_path, text, properties=None):
        _write_text(tmp_path / "p.csv", text)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "P": {
                    "csv": "p.csv",
                    "pk": "id",
                    "title": "name",
                    "properties": properties or {},
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        return tmp_path / "bp.json"

    def test_a_short_row_is_null_padded(self, tmp_path):
        """The readers run the `csv` crate with `flexible(true)`: a row with
        fewer fields than the header is padded with nulls, not rejected."""
        bp_path = self._bp(
            tmp_path,
            "id,name,age\n1,Alice,30\n2,Bob\n",
            {"age": "int"},
        )
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher("MATCH (p:P) RETURN p.id AS i, p.age AS a ORDER BY i").to_list()
        assert rows == [{"i": 1, "a": 30}, {"i": 2, "a": None}]

    def test_fields_past_the_header_width_are_dropped_silently(self, tmp_path):
        """`flexible(true)`'s other half: the loader reads `0..headers.len()`
        cells, so a trailing extra field vanishes with no warning. A producer
        that emits a phantom trailing column (a `\\t|` line terminator, say)
        gets away with it here and must not start failing loudly by accident."""
        bp_path = self._bp(
            tmp_path,
            "id,name\n1,Alice\n2,Bob,EXTRA\n",
        )
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher("MATCH (p:P) RETURN p.id AS i, p.title AS t ORDER BY i").to_list()
        assert rows == [{"i": 1, "t": "Alice"}, {"i": 2, "t": "Bob"}]

    def test_a_utf8_bom_on_the_header_is_stripped(self, tmp_path):
        """`csv` 1.4 strips a leading UTF-8 BOM itself, so `pk: "id"` resolves
        against a BOM'd header. Any replacement reader must keep doing so."""
        _write_text(tmp_path / "p.csv", "﻿id,name\n1,Alice\n")
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {"P": {"csv": "p.csv", "pk": "id", "title": "name", "properties": {}}},
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        assert open(tmp_path / "p.csv", "rb").read(3) == b"\xef\xbb\xbf"
        graph = from_blueprint(tmp_path / "bp.json", save=False)
        assert graph.cypher("MATCH (p:P) RETURN p.id AS i").to_list() == [{"i": 1}]

    def test_crlf_line_endings_load(self, tmp_path):
        bp_path = self._bp(tmp_path, "id,name\r\n1,Alice\r\n2,Bob\r\n")
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher("MATCH (p:P) RETURN p.title AS t ORDER BY t").to_list()
        # No stray \r rides along on the last column of a row.
        assert [r["t"] for r in rows] == ["Alice", "Bob"]


class TestTrimAndNullSemantics:
    """Every typed arm trims its cell; `String` does not. Empty is null
    everywhere; whitespace-only is null everywhere *except* string."""

    def _graph(self, tmp_path, text):
        _write_text(tmp_path / "p.csv", text)
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "P": {
                    "csv": "p.csv",
                    "pk": "id",
                    "title": "id",
                    "properties": {"s": "string", "n": "int"},
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        return from_blueprint(tmp_path / "bp.json", save=False)

    def test_a_string_column_keeps_its_padding_and_an_int_column_does_not(self, tmp_path):
        graph = self._graph(tmp_path, "id,s,n\n1, x , 1 \n")
        row = graph.cypher("MATCH (p:P) RETURN p.s AS s, p.n AS n").to_list()[0]
        assert row["s"] == " x "
        assert row["n"] == 1

    def test_empty_is_null_everywhere_whitespace_only_is_null_except_in_a_string(self, tmp_path):
        graph = self._graph(tmp_path, "id,s,n\n1,,\n2,   ,   \n")
        rows = graph.cypher("MATCH (p:P) RETURN p.id AS i, p.s AS s, p.n AS n ORDER BY i").to_list()
        assert rows[0] == {"i": 1, "s": None, "n": None}
        assert rows[1] == {"i": 2, "s": "   ", "n": None}


class TestFeatureGoldens:
    """`filter` and `timeseries` through `from_blueprint`, with the values
    pinned rather than only the row counts."""

    def test_spec_filter_removes_rows_before_any_node_is_made(self, tmp_path):
        _write_csv(
            tmp_path / "items.csv",
            pd.DataFrame(
                {
                    "item_id": [1, 2, 3],
                    "name": ["A", "B", "C"],
                    "status": ["Active", "Inactive", "Active"],
                    "score": [10, 20, 30],
                }
            ),
        )
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Item": {
                    "csv": "items.csv",
                    "pk": "item_id",
                    "title": "name",
                    "properties": {"status": "string", "score": "int"},
                    "filter": {"status": "Active"},
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)
        rows = graph.cypher("MATCH (i:Item) RETURN i.item_id AS id, i.title AS t, i.score AS s ORDER BY id").to_list()
        assert rows == [{"id": 1, "t": "A", "s": 10}, {"id": 3, "t": "C", "s": 30}]
        # The filtered row leaves no stub behind either.
        assert graph.cypher("MATCH (i:Item) RETURN count(i) AS c").to_list()[0]["c"] == 2

    def test_timeseries_channels_land_and_their_source_columns_do_not(self, tmp_path):
        _write_csv(tmp_path / "fields.csv", pd.DataFrame({"field_id": [1, 2], "name": ["Troll", "Ekofisk"]}))
        _write_csv(
            tmp_path / "production.csv",
            pd.DataFrame(
                {
                    "field_id": [1, 1, 1, 2, 2, 2],
                    "name": ["Troll"] * 3 + ["Ekofisk"] * 3,
                    "prfYear": [2020] * 6,
                    "prfMonth": [1, 2, 3, 1, 2, 3],
                    "prfOil": [1.0, 1.5, 2.0, 0.5, 0.6, 0.7],
                }
            ),
        )
        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Field": {
                    "csv": "fields.csv",
                    "pk": "field_id",
                    "title": "name",
                    "properties": {},
                    "sub_nodes": {
                        "Production": {
                            "csv": "production.csv",
                            "pk": "field_id",
                            "title": "name",
                            "parent_fk": "field_id",
                            "properties": {},
                            "skipped": ["field_id", "name"],
                            "timeseries": {
                                "time_key": {"year": "prfYear", "month": "prfMonth"},
                                "resolution": "month",
                                "channels": {"oil": "prfOil"},
                                "units": {"oil": "MSm3"},
                            },
                        }
                    },
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        graph = from_blueprint(tmp_path / "bp.json", save=False)
        # One node per pk, not one per input row.
        assert graph.cypher("MATCH (p:Production) RETURN count(p) AS c").to_list()[0]["c"] == 2
        rows = graph.cypher(
            "MATCH (p:Production) RETURN p.title AS t, ts_sum(p.oil,'2020') AS year, "
            "ts_sum(p.oil,'2020-02') AS feb ORDER BY t"
        ).to_list()
        assert rows[0]["t"] == "Ekofisk"
        assert rows[0]["year"] == pytest.approx(1.8)
        assert rows[0]["feb"] == pytest.approx(0.6)
        assert rows[1]["year"] == pytest.approx(4.5)
        assert rows[1]["feb"] == pytest.approx(1.5)
        # The channel's source column and the time-key columns are consumed by
        # the timeseries, not left behind as scalar properties.
        raw = graph.cypher("MATCH (p:Production) RETURN p.prfOil AS o, p.prfYear AS y, p.prfMonth AS m").to_list()
        assert all(r["o"] is None and r["y"] is None and r["m"] is None for r in raw)


class TestTopLevelCompute:
    """The top-level `compute` pipeline had no Python test at all. It is a
    second, independent CSV reader/writer (`blueprint/compute/`) that rewrites
    the blueprint to point at files under `<root>/computed/` before the five
    load phases run."""

    def _items(self, tmp_path):
        _write_csv(
            tmp_path / "t.csv",
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "code": ["P", "S", "P"],
                    "shares": [100, 50, 25],
                    "price": [10.0, 20.0, 4.0],
                }
            ),
        )

    def _bp(self, tmp_path, compute, nodes=None):
        bp = {
            "settings": {"root": str(tmp_path)},
            "compute": compute,
            "nodes": nodes
            or {
                "T": {
                    "csv": "t.csv",
                    "pk": "id",
                    "title": "id",
                    "properties": {"code": "string", "shares": "int", "price": "float"},
                }
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        return tmp_path / "bp.json"

    def test_derive_adds_a_computed_property_to_every_node(self, tmp_path):
        self._items(tmp_path)
        bp_path = self._bp(tmp_path, [{"op": "derive", "from": "T", "set": {"total": "shares * price"}}])
        graph = from_blueprint(bp_path, save=False)
        rows = graph.cypher("MATCH (t:T) RETURN t.id AS i, t.total AS total ORDER BY i").to_list()
        assert [r["total"] for r in rows] == [
            pytest.approx(1000.0),
            pytest.approx(1000.0),
            pytest.approx(100.0),
        ]
        # The op writes a real file next to the inputs and repoints the spec.
        assert (tmp_path / "computed" / "T_derived.csv").exists()

    def test_filter_op_copies_matching_rows_into_a_new_type(self, tmp_path):
        self._items(tmp_path)
        bp_path = self._bp(tmp_path, [{"op": "filter", "from": "T", "where": "code == 'P'", "into": "Buy"}])
        graph = from_blueprint(bp_path, save=False)
        # `into` copies: the source type keeps every row.
        assert [r["i"] for r in graph.cypher("MATCH (t:T) RETURN t.id AS i ORDER BY i").to_list()] == [1, 2, 3]
        assert [r["i"] for r in graph.cypher("MATCH (b:Buy) RETURN b.id AS i ORDER BY i").to_list()] == [1, 3]

    def test_a_compute_op_whose_source_csv_is_missing_is_a_silent_no_op(self, tmp_path):
        """Every primitive probes `csv_path.exists()` and returns `Ok(())` if
        it is absent, so the build reports success and the derived property is
        simply never there. Pinned because it is the trap for any input format
        that is not a file on disk."""
        bp_path = self._bp(
            tmp_path,
            [{"op": "derive", "from": "T", "set": {"total": "shares * price"}}],
            nodes={"T": {"csv": "gone.csv", "pk": "id", "title": "id", "properties": {}}},
        )
        graph = from_blueprint(bp_path, save=False)
        assert graph.cypher("MATCH (t:T) RETURN count(t) AS c").to_list()[0]["c"] == 0
        assert not (tmp_path / "computed" / "T_derived.csv").exists()


class TestJunctionDedupeRegime:
    """The 0.16.22 dedupe regime, from the other side.

    `load_one_junction_edge` decides once per junction CSV whether it *owns*
    the edge type (`InitialLoad::Preset(!metadata.contains_key(edge_type))`)
    and reuses that decision for every chunk. `TestJunctionChunkInvariance`
    covers the owning source — its duplicate endpoint pairs stay parallel
    edges at any chunk size. This class covers the half with no test: a
    *second* source into the same edge type does not own it, so its duplicate
    endpoint pairs merge into one edge instead.
    """

    def _bp(self, tmp_path):
        _write_csv(tmp_path / "persons.csv", pd.DataFrame({"person_id": [1, 2, 3], "name": ["A", "B", "C"]}))
        _write_csv(tmp_path / "companies.csv", pd.DataFrame({"company_id": [10], "name": ["Acme"]}))
        # First source: three endpoint pairs, each twice.
        _write_csv(
            tmp_path / "first.csv",
            pd.DataFrame({"s": [1, 1, 2, 2, 3, 3], "t": [2, 2, 3, 3, 1, 1], "seq": [1, 2, 3, 4, 5, 6]}),
        )
        # Second source into the same edge type: two endpoint pairs, each twice.
        _write_csv(
            tmp_path / "second.csv",
            pd.DataFrame({"s": [10, 10, 10, 10], "t": [1, 1, 2, 2], "seq": [7, 8, 9, 10]}),
        )

        def junction(csv, target):
            return {
                "csv": csv,
                "source_fk": "s",
                "target": target,
                "target_fk": "t",
                "properties": ["seq"],
                "property_types": {"seq": "int"},
            }

        bp = {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                # Declaration order decides which source owns LINK.
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "properties": {},
                    "connections": {"junction_edges": {"LINK": junction("first.csv", "Person")}},
                },
                "Company": {
                    "csv": "companies.csv",
                    "pk": "company_id",
                    "title": "name",
                    "properties": {},
                    "connections": {"junction_edges": {"LINK": junction("second.csv", "Person")}},
                },
            },
        }
        _write_blueprint(tmp_path / "bp.json", bp)
        return tmp_path / "bp.json"

    def _edges(self, bp_path, monkeypatch, chunk_size):
        monkeypatch.setenv("KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE", str(chunk_size))
        graph = from_blueprint(bp_path, save=False)
        return sorted(
            (r["src"], r["seq"])
            for r in graph.cypher("MATCH (a)-[r:LINK]->(b) RETURN head(labels(a)) AS src, r.seq AS seq").to_list()
        )

    def test_the_first_source_keeps_parallel_edges_and_the_second_merges(self, tmp_path, monkeypatch):
        edges = self._edges(self._bp(tmp_path), monkeypatch, 1000)
        person = [seq for src, seq in edges if src == "Person"]
        company = [seq for src, seq in edges if src == "Company"]
        # Owning source: one edge per input row, every `seq` present.
        assert person == [1, 2, 3, 4, 5, 6]
        # Second source: four rows, two endpoint pairs, two edges — the later
        # row of each pair wins the property.
        assert company == [8, 10]

    def test_the_regime_does_not_move_with_the_chunk_size(self, tmp_path, monkeypatch):
        bp_path = self._bp(tmp_path)
        single = self._edges(bp_path, monkeypatch, 1000)
        for chunk_size in (1, 3, 5):
            assert self._edges(bp_path, monkeypatch, chunk_size) == single, (
                f"chunk_size={chunk_size} produced a different edge set"
            )


class TestFilesSection:
    """`files:` declares each input once by name; a spec references it with
    `"file"`. `"csv": "x.csv"` stays the shorthand for a `files` entry
    `{"path": "x.csv", "format": "csv"}`, so the two spellings must build the
    same graph and share one registry slot.
    """

    def _data(self, tmp_path):
        _write_csv(
            tmp_path / "persons.csv",
            pd.DataFrame({"person_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [28, 35, 42]}),
        )
        _write_csv(tmp_path / "knows.csv", pd.DataFrame({"source_id": [1, 2], "target_id": [2, 3]}))

    def _build(self, tmp_path, bp, name="bp.json"):
        _write_blueprint(tmp_path / name, bp)
        return from_blueprint(tmp_path / name, save=False)

    def _shorthand_bp(self, tmp_path):
        return {
            "settings": {"root": str(tmp_path)},
            "nodes": {
                "Person": {
                    "csv": "persons.csv",
                    "pk": "person_id",
                    "title": "name",
                    "properties": {"age": "int"},
                    "connections": {
                        "junction_edges": {
                            "KNOWS": {
                                "csv": "knows.csv",
                                "source_fk": "source_id",
                                "target": "Person",
                                "target_fk": "target_id",
                            }
                        }
                    },
                }
            },
        }

    def _files_bp(self, tmp_path):
        bp = self._shorthand_bp(tmp_path)
        bp["files"] = {
            "people": {"path": "persons.csv", "format": "csv"},
            "acquaintances": {"path": "knows.csv"},
        }
        person = bp["nodes"]["Person"]
        del person["csv"]
        person["file"] = "people"
        junc = person["connections"]["junction_edges"]["KNOWS"]
        del junc["csv"]
        junc["file"] = "acquaintances"
        return bp

    def _shape(self, graph):
        nodes = graph.cypher("MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY name").to_list()
        edges = graph.cypher("MATCH (a)-[:KNOWS]->(b) RETURN a.name AS a, b.name AS b ORDER BY a, b").to_list()
        return nodes, edges

    def test_a_files_blueprint_builds_the_same_graph_as_its_csv_shorthand_twin(self, tmp_path):
        self._data(tmp_path)
        shorthand = self._shape(self._build(tmp_path, self._shorthand_bp(tmp_path), "short.json"))
        declared = self._shape(self._build(tmp_path, self._files_bp(tmp_path), "files.json"))
        # Non-vacuity: the twin actually loaded rows, properties and edges.
        assert shorthand[0] == [
            {"name": "Alice", "age": 28},
            {"name": "Bob", "age": 35},
            {"name": "Charlie", "age": 42},
        ]
        assert len(shorthand[1]) == 2
        assert declared == shorthand

    def test_two_specs_read_one_declared_entry(self, tmp_path):
        _write_csv(
            tmp_path / "rows.csv",
            pd.DataFrame({"id": [1, 2], "org": ["a", "b"], "kind": ["x", "y"]}),
        )
        bp = {
            "settings": {"root": str(tmp_path)},
            "files": {"rows": {"path": "rows.csv"}},
            "nodes": {
                "Person": {"file": "rows", "pk": "id", "properties": {"kind": "string"}},
                "Org": {"file": "rows", "pk": "org", "properties": {}},
            },
        }
        graph = self._build(tmp_path, bp)
        assert len(graph.cypher("MATCH (p:Person) RETURN p.id").to_list()) == 2
        assert sorted(r["id"] for r in graph.cypher("MATCH (o:Org) RETURN o.id AS id").to_list()) == ["a", "b"]

    def test_a_junction_reads_a_declared_entry(self, tmp_path):
        self._data(tmp_path)
        bp = self._shorthand_bp(tmp_path)
        bp["files"] = {"acquaintances": {"path": "knows.csv"}}
        junc = bp["nodes"]["Person"]["connections"]["junction_edges"]["KNOWS"]
        del junc["csv"]
        junc["file"] = "acquaintances"
        graph = self._build(tmp_path, bp)
        assert len(graph.cypher("MATCH ()-[r:KNOWS]->() RETURN r").to_list()) == 2

    def test_csv_and_file_on_one_spec_is_an_error(self, tmp_path):
        self._data(tmp_path)
        bp = self._shorthand_bp(tmp_path)
        bp["files"] = {"people": {"path": "persons.csv"}}
        bp["nodes"]["Person"]["file"] = "people"
        with pytest.raises(ValueError) as exc:
            self._build(tmp_path, bp)
        assert "node 'Person'" in str(exc.value)
        assert "'csv' and 'file'" in str(exc.value)

    def test_csv_and_file_on_one_junction_is_an_error(self, tmp_path):
        self._data(tmp_path)
        bp = self._shorthand_bp(tmp_path)
        bp["files"] = {"acquaintances": {"path": "knows.csv"}}
        bp["nodes"]["Person"]["connections"]["junction_edges"]["KNOWS"]["file"] = "acquaintances"
        with pytest.raises(ValueError) as exc:
            self._build(tmp_path, bp)
        assert "junction 'KNOWS'" in str(exc.value)
        assert "'csv' and 'file'" in str(exc.value)

    def test_an_undeclared_file_name_is_an_error_listing_the_declared_ones(self, tmp_path):
        self._data(tmp_path)
        bp = self._files_bp(tmp_path)
        bp["nodes"]["Person"]["file"] = "pepole"
        with pytest.raises(ValueError) as exc:
            self._build(tmp_path, bp)
        message = str(exc.value)
        assert "node 'Person'" in message
        assert '"file": "pepole"' in message
        assert "people, acquaintances" in message

    def test_a_declared_entry_without_a_path_is_an_error(self, tmp_path):
        self._data(tmp_path)
        bp = self._files_bp(tmp_path)
        bp["files"]["people"] = {"format": "csv"}
        with pytest.raises(ValueError) as exc:
            self._build(tmp_path, bp)
        assert "files 'people'" in str(exc.value)
        assert "'path'" in str(exc.value)

    def test_an_entry_name_that_shadows_a_different_shorthand_file_is_an_error(self, tmp_path):
        self._data(tmp_path)
        _write_csv(tmp_path / "other.csv", pd.DataFrame({"person_id": [9], "name": ["Zoe"]}))
        bp = self._shorthand_bp(tmp_path)
        # The shorthand and the entry name are the same string but mean two
        # different files — one registry slot, two claimants.
        bp["files"] = {"persons.csv": {"path": "other.csv"}}
        with pytest.raises(ValueError) as exc:
            self._build(tmp_path, bp)
        message = str(exc.value)
        assert "node 'Person'" in message
        assert "persons.csv" in message
        assert "other.csv" in message

    def test_an_entry_naming_the_same_file_as_a_shorthand_is_one_input(self, tmp_path):
        self._data(tmp_path)
        bp = self._shorthand_bp(tmp_path)
        bp["files"] = {"persons.csv": {"path": "persons.csv", "format": "csv"}}
        graph = self._build(tmp_path, bp)
        assert len(graph.cypher("MATCH (p:Person) RETURN p.name").to_list()) == 3

    def test_an_unknown_format_lists_the_formats_this_build_reads(self, tmp_path):
        self._data(tmp_path)
        bp = self._files_bp(tmp_path)
        bp["files"]["people"]["format"] = "parquet"
        with pytest.raises(ValueError) as exc:
            self._build(tmp_path, bp)
        message = str(exc.value)
        assert "'parquet'" in message
        assert "csv" in message

    def test_a_stray_key_in_a_declared_entry_warns_with_its_format(self, tmp_path, capfd):
        self._data(tmp_path)
        bp = self._files_bp(tmp_path)
        bp["files"]["people"]["delimiter"] = "\t"
        _write_blueprint(tmp_path / "bp.json", bp)
        from_blueprint(tmp_path / "bp.json", save=False, verbose=True)
        err = capfd.readouterr().err
        assert "file 'people'" in err
        assert "format 'csv'" in err
        assert "unknown key 'delimiter'" in err
        assert "'path', 'format'" in err
