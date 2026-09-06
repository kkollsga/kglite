"""Every completed SET/REMOVE item is visible to later RHS reads in the same statement."""

import pytest

import kglite

MODES = ["memory", "mapped", "disk"]


def graph_for(tmp_path, mode):
    graph = kglite.KnowledgeGraph(storage=mode, path=str(tmp_path / "disk") if mode == "disk" else None)
    graph.cypher("CREATE(:T{id:1,title:'before',a:0,b:0,items:[{qty:0}]})")
    # Publish disk columns before exercising the staged mutation read path.
    if mode == "disk":
        graph.save(str(tmp_path / "disk"))
    return graph


def values(graph):
    return graph.cypher("MATCH(n:T) RETURN n.id AS id,n.a AS a,n.b AS b ORDER BY id").to_list()


def no_staged_read(capfd):
    assert "staged Map-typed write" not in capfd.readouterr().err


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("indexed", [False, True])
def test_string_composite_set_remove_matches_unindexed_values(tmp_path, mode, indexed, capfd):
    graph = kglite.KnowledgeGraph(storage=mode, path=str(tmp_path / "disk") if mode == "disk" else None)
    graph.cypher("CREATE(:T{id:1,a:'old',b:'fixed'}),(:T{id:2,a:'other',b:'fixed'})").to_list()
    if mode == "disk":
        graph.save(str(tmp_path / "disk"))
    if indexed:
        graph.create_composite_index("T", ["a", "b"])

    def ids(predicate):
        return graph.cypher(f"MATCH(n:T) WHERE {predicate} RETURN n.id AS id ORDER BY id").to_list()

    assert ids("n.a='old' AND n.b='fixed'") == [{"id": 1}]
    graph.cypher("MATCH(n:T{id:1}) SET n.a='new'").to_list()
    assert ids("n.a='new' AND n.b='fixed'") == [{"id": 1}]
    assert ids("n.a='old' AND n.b='fixed'") == []
    graph.cypher("MATCH(n:T{id:1}) REMOVE n.a").to_list()
    assert ids("n.a='new' AND n.b='fixed'") == []
    assert ids("n.a IS NULL AND n.b='fixed'") == [{"id": 1}]
    assert graph.cypher("MATCH(n:T) RETURN n.id AS id,n.a AS a,n.b AS b ORDER BY id").to_list() == [
        {"id": 1, "a": None, "b": "fixed"},
        {"id": 2, "a": "other", "b": "fixed"},
    ]
    no_staged_read(capfd)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "operation,expected",
    [
        ("SET n.a=1,n.b=n.a+1", {"id": 1, "a": 1, "b": 2}),
        ("SET n.a=n.a+1,n.b=n.a", {"id": 1, "a": 1, "b": 1}),
        ("SET n.a=1 WITH n SET n.b=n.a+1", {"id": 1, "a": 1, "b": 2}),
        ("SET n.a=1,n.b=2", {"id": 1, "a": 1, "b": 2}),
        ("SET n += {a:1},n.b=n.a+1", {"id": 1, "a": 1, "b": 2}),
        ("SET n = {a:1},n.b=n.a+1", {"id": 1, "a": 1, "b": 2}),
    ],
)
def test_sequential_set_items_match_exact_values(tmp_path, mode, operation, expected, capfd):
    graph = graph_for(tmp_path, mode)
    graph.cypher(f"MATCH(n:T) {operation}").to_list()
    assert values(graph) == [expected]
    no_staged_read(capfd)


@pytest.mark.parametrize("mode", MODES)
def test_set_items_compose_across_repeated_rows_and_nested_rhs(tmp_path, mode, capfd):
    graph = graph_for(tmp_path, mode)
    graph.cypher("MATCH(n:T) UNWIND [1,2,3] AS i SET n.a=n.a+1,n.b=n.a").to_list()
    assert values(graph) == [{"id": 1, "a": 3, "b": 3}]
    graph.cypher("MATCH(n:T) SET n.items[0].qty=7,n.b=n.items[0].qty").to_list()
    assert values(graph) == [{"id": 1, "a": 3, "b": 7}]
    assert graph.cypher("MATCH(n:T) RETURN n.items AS items").scalar() == [{"qty": 7}]
    no_staged_read(capfd)


@pytest.mark.parametrize("mode", MODES)
def test_set_aliases_labels_and_null_target_do_not_hide_completed_item(tmp_path, mode, capfd):
    graph = graph_for(tmp_path, mode)
    graph.cypher("MATCH(n:T) SET n.title='after',n.copy=n.name").to_list()
    assert graph.cypher("MATCH(n:T) RETURN n.copy AS copy").scalar() == "after"
    graph.cypher("MATCH(n:T) OPTIONAL MATCH(m:Absent) SET n.a=4,m.a=9,n:Visible,n.b=n.a+1").to_list()
    assert values(graph) == [{"id": 1, "a": 4, "b": 5}]
    assert graph.cypher("MATCH(n:Visible) RETURN n.id AS id").scalar() == 1
    no_staged_read(capfd)


@pytest.mark.parametrize("mode", MODES)
def test_merge_and_foreach_reach_set_item_visibility_boundary(tmp_path, mode, capfd):
    graph = graph_for(tmp_path, mode)
    graph.cypher("MERGE(n:T{id:1}) ON MATCH SET n.a=1,n.b=n.a+1").to_list()
    graph.cypher("MERGE(n:T{id:2}) ON CREATE SET n.a=5,n.b=n.a+1").to_list()
    assert values(graph) == [{"id": 1, "a": 1, "b": 2}, {"id": 2, "a": 5, "b": 6}]
    graph.cypher("MATCH(n:T{id:1}) FOREACH(i IN [1,2] | SET n.a=n.a+1,n.b=n.a)").to_list()
    assert values(graph) == [{"id": 1, "a": 3, "b": 3}, {"id": 2, "a": 5, "b": 6}]
    no_staged_read(capfd)


@pytest.mark.parametrize("mode", MODES)
def test_error_after_completed_item_restores_exact_statement_state(tmp_path, mode, capfd):
    graph = graph_for(tmp_path, mode)
    query = "MATCH(n:T) RETURN n.id AS id,elementId(n) AS slot,properties(n) AS props,labels(n) AS labels"
    before = graph.cypher(query).to_list()
    with pytest.raises(kglite.CypherExecutionError, match="immutable"):
        graph.cypher("MATCH(n:T) SET n.a=9,n.id=2").to_list()
    assert graph.cypher(query).to_list() == before
    assert values(graph) == [{"id": 1, "a": 0, "b": 0}]
    no_staged_read(capfd)


@pytest.mark.parametrize("mode", MODES)
def test_session_set_reads_own_writes_without_changing_held_snapshot(tmp_path, mode, capfd):
    graph = graph_for(tmp_path, mode)
    session = graph.session()
    held = session.snapshot()
    before = values(held)
    session.execute("MATCH(n:T) SET n.a=1,n.b=n.a+1")
    assert values(session) == [{"id": 1, "a": 1, "b": 2}]
    assert values(held) == before == [{"id": 1, "a": 0, "b": 0}]
    assert values(graph) == before
    no_staged_read(capfd)


@pytest.mark.parametrize("mode", MODES)
def test_repeated_title_remove_counts_one_change_and_clears_title(tmp_path, mode, capfd):
    graph = graph_for(tmp_path, mode)
    graph.cypher("MATCH(n:T) REMOVE n.title,n.title").to_list()
    assert graph.last_mutation_stats["properties_removed"] == 1
    assert graph.cypher("MATCH(n:T) RETURN n.title AS title").scalar() is None
    assert values(graph) == [{"id": 1, "a": 0, "b": 0}]
    no_staged_read(capfd)


@pytest.mark.parametrize("mode", MODES)
def test_remove_properties_and_label_preserves_held_view_and_rolls_back_error(tmp_path, mode, capfd):
    graph = graph_for(tmp_path, mode)
    graph.cypher("MATCH(n:T) SET n:Visible").to_list()
    session = graph.session()
    held = session.snapshot()
    snapshot_query = (
        "MATCH(n:T) RETURN n.id AS id,elementId(n) AS slot,n.title AS title,properties(n) AS props,labels(n) AS labels"
    )
    before = held.cypher(snapshot_query).to_list()
    session.execute("MATCH(n:T) REMOVE n.a,n.b,n:Visible")
    assert values(session) == [{"id": 1, "a": None, "b": None}]
    assert session.cypher("MATCH(n:Visible) RETURN count(n) AS c").scalar() == 0
    assert held.cypher(snapshot_query).to_list() == before
    with pytest.raises(kglite.CypherExecutionError, match="immutable"):
        graph.cypher("MATCH(n:T) REMOVE n.a,n.id").to_list()
    assert graph.cypher(snapshot_query).to_list() == before
    no_staged_read(capfd)


def test_held_session_title_remove_saves_without_an_absent_alias_column(tmp_path):
    graph = graph_for(tmp_path, "memory")
    session = graph.session()
    held = session.snapshot()
    session.execute("MATCH(n:T) REMOVE n.title")
    assert session.cypher("MATCH(n:T) RETURN n.title AS title").scalar() is None
    assert held.cypher("MATCH(n:T) RETURN n.title AS title").scalar() == "before"
    path = tmp_path / "session.kgl"
    session.cursor().save(str(path))
    loaded = kglite.load(str(path))
    assert loaded.cypher("MATCH(n:T) RETURN n.title AS title").scalar() is None
    assert values(loaded) == [{"id": 1, "a": 0, "b": 0}]
