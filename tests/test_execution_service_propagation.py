"""An ordinary Python callback remains available across supported execution paths."""

import pytest

import kglite


class FakeEmbedder:
    dimension = 2
    model_id = "execution-service-contract"

    def __init__(self, reverse=False):
        self.calls = []
        self.reverse = reverse
        self.fail = False

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("ordinary fake callback failure")
        return [[0.0, 1.0] if (text == "beta") != self.reverse else [1.0, 0.0] for text in texts]


def fixture():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Doc {id:1,title:'A',body:'alpha'}),(:Doc {id:2,title:'B',body:'beta'})")
    model = FakeEmbedder()
    graph.set_embedder(model)
    graph.embed_texts("Doc", "body", show_progress=False)
    model.calls.clear()
    return graph, model


READ = "MATCH (d:Doc) RETURN d.id AS id,text_score(d,'body','query') AS score ORDER BY id"
WRITE = "MATCH (d:Doc) SET d.score=text_score(d,'body','query') RETURN d.id AS id,d.score AS score ORDER BY id"
EXPECTED = [{"id": 1, "score": 1.0}, {"id": 2, "score": 0.0}]


def handle(graph, kind):
    if kind == "graph":
        return graph
    return {
        "frozen": graph.freeze,
        "session": graph.session,
        "transaction": graph.begin,
        "read_transaction": graph.begin_read,
    }[kind]()


def run(target, kind, query, **options):
    method = target.execute if kind == "session" else target.cypher
    return method(query, **options).to_list()


def close(target, kind):
    if kind in {"transaction", "read_transaction"}:
        target.rollback()


@pytest.mark.parametrize("kind", ["graph", "frozen", "session", "transaction", "read_transaction"])
def test_registered_embedder_remains_available_for_reads(kind):
    graph, model = fixture()
    target = handle(graph, kind)
    try:
        assert run(target, kind, READ) == EXPECTED
        assert model.calls == [["query"]]
    finally:
        close(target, kind)


@pytest.mark.parametrize("kind", ["graph", "session", "transaction"])
def test_registered_embedder_remains_available_for_writes(kind):
    graph, model = fixture()
    target = handle(graph, kind)
    try:
        assert run(target, kind, WRITE) == EXPECTED
        assert run(target, kind, "MATCH (d:Doc) RETURN d.id AS id,d.score AS score ORDER BY id") == EXPECTED
        assert model.calls == [["query"]]
    finally:
        close(target, kind)


@pytest.mark.parametrize("kind", ["frozen", "session", "transaction", "read_transaction"])
def test_derived_handle_captures_binding_at_creation(kind):
    graph, original = fixture()
    target = handle(graph, kind)
    replacement = FakeEmbedder(reverse=True)
    graph.set_embedder(replacement)
    try:
        assert run(target, kind, READ) == EXPECTED
        assert original.calls == [["query"]]
        assert replacement.calls == []
        assert graph.cypher(READ).to_list() == [{"id": 1, "score": 0.0}, {"id": 2, "score": 1.0}]
        assert replacement.calls == [["query"]]
    finally:
        close(target, kind)


@pytest.mark.parametrize("kind", ["graph", "session", "transaction"])
def test_callback_failure_keeps_statement_state(kind):
    graph, model = fixture()
    target = handle(graph, kind)
    state_query = "MATCH (d:Doc) RETURN d.id AS id,properties(d) AS props ORDER BY id"
    # A transaction already has a working fork; failed callback must preserve it.
    run(target, kind, "MATCH (d:Doc) SET d.prior=7")
    before = run(target, kind, state_query)
    model.fail = True
    try:
        with pytest.raises(kglite.KgError, match="ordinary fake callback failure"):
            run(target, kind, WRITE)
        assert model.calls == [["query"]]
        assert run(target, kind, state_query) == before
        model.fail = False
        assert run(target, kind, WRITE) == EXPECTED
    finally:
        close(target, kind)
