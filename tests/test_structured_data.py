"""Structured-data support: table properties, declared shapes."""

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


@pytest.fixture
def g() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:Order {id: 'order-1', name: 'first'})")
    return graph


def _items() -> pd.DataFrame:
    # Column order deliberately non-alphabetical.
    return pd.DataFrame(
        {
            "sku": ["a-1", "b-2", "c-3"],
            "qty": [1, 2, 8],
            "price": [9.5, 12.0, 3.25],
            "note": ["x", None, "z"],
        }
    )


def test_table_property_roundtrip_preserves_order_and_dtypes(g):
    stored = g.set_table_property("Order", "order-1", "line_items", _items())
    assert stored == 3
    df = g.get_table_property("Order", "order-1", "line_items")
    assert list(df.columns) == ["sku", "qty", "price", "note"]
    assert str(df["qty"].dtype) == "int64"
    assert str(df["price"].dtype) == "float64"
    assert df["qty"].tolist() == [1, 2, 8]
    assert df["note"].isna().sum() == 1


def test_table_property_is_plain_cypher_data(g):
    g.set_table_property("Order", "order-1", "line_items", _items())
    rows = g.cypher(
        "MATCH (o:Order {id: 'order-1'}) UNWIND o.line_items AS r RETURN r.sku AS sku, r.qty AS qty ORDER BY sku"
    ).to_list()
    assert [(r["sku"], r["qty"]) for r in rows] == [("a-1", 1), ("b-2", 2), ("c-3", 8)]
    assert g.cypher("MATCH (o:Order {id: 'order-1'}) RETURN o.line_items[2].price AS p").scalar() == 3.25


def test_table_property_meta_persists(g, tmp_path):
    g.set_table_property("Order", "order-1", "line_items", _items())
    path = tmp_path / "t.kgl"
    g.save(str(path))
    loaded = kglite.load(str(path))
    df = loaded.get_table_property("Order", "order-1", "line_items")
    assert list(df.columns) == ["sku", "qty", "price", "note"]
    assert str(df["qty"].dtype) == "int64"


def test_table_property_errors(g):
    with pytest.raises(ValueError, match="no Order node"):
        g.set_table_property("Order", "missing", "line_items", _items())
    with pytest.raises(TypeError, match="pandas DataFrame"):
        g.set_table_property("Order", "order-1", "line_items", [1, 2])
    with pytest.raises(ValueError, match="plain identifier"):
        g.set_table_property("Order", "order-1", "bad prop", _items())


# ─── declared structured shapes ────────────────────────────────────────────

SHAPE_SCHEMA = {
    "nodes": {
        "Order": {
            "types": {
                "line_items": "list<map{sku: string!, qty: int!, price: float}>",
                "status": "string",
            }
        }
    }
}


def test_shape_gates_add_nodes_with_indexed_path(g):
    g.define_schema(SHAPE_SCHEMA)
    good = pd.DataFrame(
        {
            "id": ["o2"],
            "name": ["ok"],
            "line_items": [[{"sku": "a", "qty": 1, "price": 2.0}]],
        }
    )
    g.add_nodes(good, "Order", "id", node_title_field="name")

    bad = pd.DataFrame(
        {
            "id": ["o3"],
            "name": ["bad"],
            "line_items": [[{"sku": "a", "qty": 1}, {"sku": "b", "qty": "two"}]],
        }
    )
    with pytest.raises(Exception, match=r"line_items\[1\]\.qty: expected integer"):
        g.add_nodes(bad, "Order", "id", node_title_field="name")
    # Whole-frame gate: nothing was written.
    assert g.cypher("MATCH (o:Order {id: 'o3'}) RETURN count(o) AS c").scalar() == 0


def test_shape_gates_cypher_set_and_create(g):
    g.define_schema(SHAPE_SCHEMA)
    with pytest.raises(Exception, match=r"line_items\[0\]\.sku: required key is missing"):
        g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.line_items = [{qty: 3}]")
    with pytest.raises(Exception, match=r"line_items\[0\]\.qty: expected integer"):
        g.cypher("CREATE (:Order {id: 'o4', line_items: [{sku: 'x', qty: 'oops'}]})")
    # Good values pass both.
    g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.line_items = [{sku: 'x', qty: 3, price: 1.0}]")
    g.cypher("CREATE (:Order {id: 'o5', line_items: [{sku: 'y', qty: 4}]})")


def test_plain_types_stay_advisory(g):
    g.define_schema(SHAPE_SCHEMA)
    # 'status: string' is advisory as before — a wrong scalar type does not error.
    g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.status = 42")


def test_malformed_shape_fails_define_schema(g):
    with pytest.raises(Exception, match="unknown type 'oops'"):
        g.define_schema({"nodes": {"Order": {"types": {"x": "list<oops>"}}}})


def test_shapes_persist(g, tmp_path):
    g.define_schema(SHAPE_SCHEMA)
    path = tmp_path / "s.kgl"
    g.save(str(path))
    loaded = kglite.load(str(path))
    with pytest.raises(Exception, match=r"expected integer"):
        loaded.cypher("MATCH (o:Order {id: 'order-1'}) SET o.line_items = [{sku: 'a', qty: 'no'}]")


# ─── atomic nested mutations ───────────────────────────────────────────────


def test_nested_set_updates_one_cell(g):
    g.set_table_property("Order", "order-1", "line_items", _items())
    g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.line_items[2].qty = 99")
    assert g.cypher("MATCH (o:Order {id: 'order-1'}) RETURN o.line_items[2].qty AS q").scalar() == 99
    # The rest of the row and table are untouched.
    assert g.cypher("MATCH (o:Order {id: 'order-1'}) RETURN o.line_items[2].sku AS s").scalar() == "c-3"
    assert g.cypher("MATCH (o:Order {id: 'order-1'}) RETURN size(o.line_items) AS n").scalar() == 3


def test_nested_set_map_field_and_creation(g):
    g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.metadata.status = 'approved'")
    assert g.cypher("MATCH (o:Order {id: 'order-1'}) RETURN o.metadata.status AS s").scalar() == "approved"
    g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.metadata.status = 'shipped'")
    assert g.cypher("MATCH (o:Order {id: 'order-1'}) RETURN o.metadata.status AS s").scalar() == "shipped"


def test_nested_set_errors_name_the_path(g):
    g.set_table_property("Order", "order-1", "line_items", _items())
    with pytest.raises(Exception, match=r"line_items\[7\]: index out of bounds"):
        g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.line_items[7].qty = 1")
    with pytest.raises(Exception, match="expected a list"):
        g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.name[0] = 'x'")


def test_nested_set_revalidates_declared_shape(g):
    g.define_schema(SHAPE_SCHEMA)
    g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.line_items = [{sku: 'a', qty: 1, price: 2.0}]")
    with pytest.raises(Exception, match=r"line_items\[0\]\.qty: expected integer"):
        g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.line_items[0].qty = 'nope'")


def test_list_append_via_plus(g):
    g.set_table_property("Order", "order-1", "line_items", _items())
    g.cypher("MATCH (o:Order {id: 'order-1'}) SET o.line_items = o.line_items + [{sku: 'd-4', qty: 5}]")
    assert g.cypher("MATCH (o:Order {id: 'order-1'}) RETURN size(o.line_items) AS n").scalar() == 4


def test_table_upsert_and_delete(g):
    g.set_table_property("Order", "order-1", "line_items", _items())
    r = g.cypher(
        "CALL table.upsert({type: 'Order', id: 'order-1', property: 'line_items', "
        "key: 'sku', row: {sku: 'b-2', qty: 42}}) YIELD action, rows RETURN action, rows"
    ).to_list()[0]
    assert (r["action"], r["rows"]) == ("updated", 3)
    assert (
        g.cypher(
            "MATCH (o:Order {id: 'order-1'}) UNWIND o.line_items AS r WITH r WHERE r.sku = 'b-2' RETURN r.qty AS q"
        ).scalar()
        == 42
    )

    r = g.cypher(
        "CALL table.upsert({type: 'Order', id: 'order-1', property: 'line_items', "
        "key: 'sku', row: {sku: 'e-5', qty: 1}}) YIELD action, rows RETURN action, rows"
    ).to_list()[0]
    assert (r["action"], r["rows"]) == ("inserted", 4)

    r = g.cypher(
        "CALL table.delete({type: 'Order', id: 'order-1', property: 'line_items', "
        "key: 'sku', value: 'a-1'}) YIELD removed, rows RETURN removed, rows"
    ).to_list()[0]
    assert (r["removed"], r["rows"]) == (1, 3)


def test_table_procs_are_write_gated(g):
    g.set_table_property("Order", "order-1", "line_items", _items())
    ro = g  # read_only flag route
    ro.read_only(True)
    try:
        with pytest.raises(Exception):
            ro.cypher(
                "CALL table.upsert({type: 'Order', id: 'order-1', property: "
                "'line_items', key: 'sku', row: {sku: 'z', qty: 1}}) YIELD rows RETURN rows"
            )
    finally:
        ro.read_only(False)


# ─── attach_rows + describe integration ────────────────────────────────────


def test_attach_rows_normalized_modeling(g):
    n = kglite.attach_rows(g, "Order", "order-1", _items(), row_type="LineItem", edge_type="HAS_LINE", key="sku")
    assert n == 3
    rows = g.cypher(
        "MATCH (:Order {id: 'order-1'})-[:HAS_LINE]->(r:LineItem) RETURN r.sku AS sku, r.qty AS qty ORDER BY sku"
    ).to_list()
    assert [(r["sku"], r["qty"]) for r in rows] == [("a-1", 1), ("b-2", 2), ("c-3", 8)]
    with pytest.raises(ValueError, match="no Order node"):
        kglite.attach_rows(g, "Order", "nope", _items(), row_type="LineItem", edge_type="HAS_LINE", key="sku")
    dup = _items()
    dup.loc[1, "sku"] = "a-1"
    with pytest.raises(ValueError, match="duplicate"):
        kglite.attach_rows(g, "Order", "order-1", dup, row_type="LineItem", edge_type="HAS_LINE", key="sku")


def test_describe_reports_declared_and_inferred_shapes(g):
    g.set_table_property("Order", "order-1", "line_items", _items())
    text = g.describe(types=["Order"])
    assert 'shape="list&lt;map{' in text or 'shape="list<map{' in text
    assert "shape_inferred" in text
    g.define_schema(SHAPE_SCHEMA)
    text = g.describe(types=["Order"])
    assert "qty: integer!" in text
    assert "shape_inferred" not in text.split('name="line_items"')[1].split("/>")[0]
