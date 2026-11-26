"""Export/import defects found in review: each test failed on the tree before its fix.

Naming is HR/org-chart only. Every round trip goes through the real exporter
and the real importer.
"""

import json
import warnings

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


def _titles(g, label):
    return g.cypher(f"MATCH (n:{label}) RETURN n.id AS id, n.title AS title ORDER BY id").to_list()


def _null_title_graph():
    g = KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": [1, 2, 3], "code": pd.array([5, None, 9], dtype="Int64")}),
        "Badge",
        "id",
        "code",
    )
    g.add_nodes(pd.DataFrame({"id": [1, 2, 3], "name": ["a", None, "c"]}), "Person", "id", "name")
    return g


def test_rdf_round_trip_keeps_a_null_title_null(tmp_path):
    g = _null_title_graph()
    path = str(tmp_path / "g.nq")
    g.export_rdf(path)
    back = kglite.load_rdf(path)
    for label in ("Badge", "Person"):
        assert _titles(back, label) == _titles(g, label), label
        assert back.cypher(f"MATCH (n:{label}) WHERE n.title IS NULL RETURN n.id AS id").to_list() == [{"id": 2}]


def test_empty_export_destinations_are_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = _null_title_graph()
    with pytest.raises(OSError, match="must not be empty"):
        g.export_csv("")
    with pytest.raises((OSError, ValueError), match="must not be empty"):
        g.export_rdf("")
    assert list(tmp_path.iterdir()) == []


def _type_property_graph():
    g = KnowledgeGraph()
    g.cypher("CREATE (:Person {id: 1, title: 'Ada', type: 'manager', name: 'Ada L'})")
    g.cypher("CREATE (:Person {id: 2, title: 'Bo', type: 'contractor'})")
    g.cypher("CREATE (:Person {id: 3, title: 'Cy'})")
    return g


def _type_values(g):
    return g.cypher("MATCH (n:Person) RETURN n.id AS id, n.type AS t ORDER BY id").to_list()


def test_a_property_named_type_survives_csv(tmp_path):
    g = _type_property_graph()
    out = str(tmp_path / "csv")
    g.export_csv(out)
    back = kglite.from_blueprint(out + "/blueprint.json", save=False)
    assert _type_values(back) == _type_values(g)
    assert _type_values(g)[0] == {"id": 1, "t": "manager"}


@pytest.mark.parametrize("fmt", ["nq", "trig"])
def test_a_property_named_type_survives_rdf(tmp_path, fmt):
    g = _type_property_graph()
    path = str(tmp_path / f"g.{fmt}")
    g.export_rdf(path, format=fmt)
    back = kglite.load_rdf(path)
    assert _type_values(back) == _type_values(g)


def _years_blueprint(tmp_path):
    (tmp_path / "n.csv").write_text(
        "id,ts,dt\n"
        "1,10000-01-01T00:00:00,10000-01-01\n"
        "2,2020-01-01T00:00:00,2020-01-01\n"
        "3,0000-01-01T00:00:00,0000-01-01\n",
        encoding="utf-8",
    )
    spec = {
        "settings": {"root": "."},
        "nodes": {
            "T": {
                "csv": "n.csv",
                "pk": "id",
                "title": "id",
                "properties": {"id": "int", "ts": "timestamp", "dt": "date"},
            }
        },
    }
    path = tmp_path / "b.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return str(path)


def test_csv_temporal_cells_outside_years_1_to_9999_are_null_with_a_warning(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = kglite.from_blueprint(_years_blueprint(tmp_path), save=False)
    rows = g.cypher("MATCH (n:T) RETURN n.id AS id, n.ts AS ts, n.dt AS dt ORDER BY id").to_list()
    assert rows[0]["ts"] is None and rows[0]["dt"] is None
    assert rows[1]["ts"] is not None and rows[1]["dt"] is not None
    assert rows[2]["ts"] is None and rows[2]["dt"] is None
    text = " ".join(str(w.message) for w in caught)
    assert "declared timestamp" in text and "declared date" in text


def test_rdf_datetimes_outside_years_1_to_9999_stay_text_with_a_warning(tmp_path):
    xsd = "http://www.w3.org/2001/XMLSchema#"
    lines = []
    for n, (dt, kind) in enumerate(
        [
            ("10000-01-01T00:00:00", "dateTime"),
            ("2020-01-01T00:00:00", "dateTime"),
            ("10000-01-01", "date"),
            ("0000-01-01T00:00:00", "dateTime"),
        ]
    ):
        lines.append(f'<http://e.org/n{n}> <http://e.org/at> "{dt}"^^<{xsd}{kind}> .')
    path = tmp_path / "y.nt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="kept as text"):
        g = kglite.load_rdf(str(path))
    values = sorted((r["uri"], r["at"]) for r in g.cypher("MATCH (n) RETURN n.uri AS uri, n.at AS at").to_list())
    assert values[0][1] == "10000-01-01T00:00:00"
    assert values[2][1] == "10000-01-01"
    assert values[3][1] == "0000-01-01T00:00:00"
    # every typed value must read back through properties()
    g.cypher("MATCH (n) RETURN properties(n) AS p").to_list()
    assert not isinstance(values[1][1], str)


def _round_trip(g, fmt, tmp_path):
    if fmt == "csv":
        out = str(tmp_path / "csv")
        g.export_csv(out)
        return kglite.from_blueprint(out + "/blueprint.json", save=False)
    path = str(tmp_path / f"g.{fmt}")
    g.export_rdf(path, format=fmt)
    return kglite.load_rdf(path)


FORMATS = ["csv", "nq", "trig"]


def _props(g, label):
    return g.cypher(f"MATCH (n:{label}) RETURN n.id AS id, properties(n) AS p ORDER BY id").to_list()


@pytest.mark.parametrize("fmt", FORMATS)
def test_typed_values_inside_lists_and_maps_keep_their_type(tmp_path, fmt):
    g = KnowledgeGraph()
    g.cypher(
        "CREATE (:Person {id: 1, title: 'Ada', "
        "days: [date('2020-01-01'), date('2021-02-03')], "
        "seen: [datetime('2020-01-02T03:04:05.250')], "
        "gaps: [duration({days: 3})], "
        "spots: [point({latitude: 60.1, longitude: 5.2})], "
        "nested: [[1, 2], [date('2020-01-01')]], "
        "info: {hired: date('2020-01-01'), at: datetime('2020-01-02T03:04:05'), "
        "gap: duration({days: 1}), spot: point({latitude: 1.5, longitude: 2.5}), "
        "inner: {d: date('1999-12-31'), l: [duration({days: 2})]}}, "
        "tricky: {`$date`: '2020-01-01'}})"
    )
    back = _round_trip(g, fmt, tmp_path)
    got, want = _props(back, "Person"), _props(g, "Person")
    assert got == want
    assert isinstance(want[0]["p"]["days"][0], __import__("datetime").date)


@pytest.mark.parametrize("fmt", FORMATS)
def test_mixed_kind_titles_keep_their_kind(tmp_path, fmt):
    g = KnowledgeGraph()
    g.cypher("CREATE (:Badge {id: 1, title: 5}), (:Badge {id: 2, title: 'five'}), (:Badge {id: 3, title: 2.5})")
    g.cypher("CREATE (:Badge {id: 4, title: date('2020-01-01')}), (:Badge {id: 5})")
    back = _round_trip(g, fmt, tmp_path)
    assert _titles(back, "Badge") == _titles(g, "Badge")
    assert [type(r["title"]).__name__ for r in _titles(back, "Badge")] == [
        type(r["title"]).__name__ for r in _titles(g, "Badge")
    ]


def _mixed_id_graph():
    g = KnowledgeGraph()
    g.cypher(
        "CREATE (:Person {id: 1, title: 'Ada'}), (:Person {id: '1', title: 'Bo'}), (:Person {id: 'x', title: 'Cy'})"
    )
    g.cypher("MATCH (a:Person {id: 1}), (b:Person {id: '1'}) CREATE (a)-[:REPORTS_TO]->(b)")
    return g


def test_csv_refuses_a_type_whose_ids_mix_kinds(tmp_path):
    g = _mixed_id_graph()
    with pytest.raises(OSError, match="mix kinds"):
        g.export_csv(str(tmp_path / "csv"))


@pytest.mark.parametrize("fmt", ["nq", "trig"])
def test_rdf_keeps_mixed_kind_ids_apart(tmp_path, fmt):
    g = _mixed_id_graph()
    back = _round_trip(g, fmt, tmp_path)
    ids = lambda graph: sorted(  # noqa: E731
        (type(r["id"]).__name__, r["id"], r["title"])
        for r in graph.cypher("MATCH (n:Person) RETURN n.id AS id, n.title AS title").to_list()
    )
    assert ids(back) == ids(g)
    assert len(ids(back)) == 3


def _people_graph(tmp_path, storage):
    g = KnowledgeGraph(storage=storage, path=str(tmp_path / storage)) if storage == "disk" else KnowledgeGraph()
    for n in range(7):
        g.cypher(f"CREATE (:Person {{id: {n}, title: 'p{n}', age: {20 + n}, hired: date('2020-01-0{n + 1}')}})")
    g.cypher("CREATE (:Department {id: 'eng', title: 'Engineering'})")
    g.cypher("MATCH (p:Person), (d:Department) CREATE (p)-[:WORKS_IN {since: date('2021-01-01')}]->(d)")
    return g


def _tree(root):
    import pathlib

    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(pathlib.Path(root).rglob("*")) if p.is_file()}


# A disk graph builds each record it reads on demand; its export must match the
# in-memory graph's byte for byte.
def test_csv_export_of_a_disk_graph_matches_memory(tmp_path):
    _people_graph(tmp_path, "disk").export_csv(str(tmp_path / "from_disk"))
    _people_graph(tmp_path, "default").export_csv(str(tmp_path / "from_memory"))
    assert _tree(tmp_path / "from_disk") == _tree(tmp_path / "from_memory")
    assert len(_tree(tmp_path / "from_disk")) > 3


@pytest.mark.parametrize("fmt", ["nq", "trig"])
def test_rdf_export_of_a_disk_graph_matches_memory(tmp_path, fmt):
    _people_graph(tmp_path, "disk").export_rdf(str(tmp_path / f"disk.{fmt}"), format=fmt)
    _people_graph(tmp_path, "default").export_rdf(str(tmp_path / f"memory.{fmt}"), format=fmt)
    disk = (tmp_path / f"disk.{fmt}").read_text(encoding="utf-8")
    assert disk == (tmp_path / f"memory.{fmt}").read_text(encoding="utf-8")
    assert disk.count("WORKS_IN") >= 7
