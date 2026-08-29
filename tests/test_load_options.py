"""`kglite.load` / `from_bytes` / `open_session` load options.

Two knobs with very different jobs. ``storage=`` decides the backend the loaded
graph *continues* in — it is not a memory lever, because mapped and memory cost
the same for a loaded ``.kgl``. ``defer_index_rebuild=`` is the memory lever:
it records the file's declared indexes instead of building them, which is only
safe because every reader of the index maps then sees what a no-index graph
sees. Both halves of that are asserted here — the listing shows the
declarations, and the predicates keep saying "no index" — because a listing
that reported nothing would hide the indexes and a predicate that reported them
would drop rows.
"""

import pytest

import kglite


def _seed(path, *, storage=None):
    """A graph with all four index families, saved at ``path``."""
    with kglite.open(str(path), storage=storage) as g:
        for i in range(40):
            g.cypher(
                "CREATE (:Item {id: $i, sku: $sku, category: $cat, region: $reg, score: $i})",
                params={
                    "i": i,
                    "sku": f"sku-{i}",
                    "cat": f"cat-{i % 5}",
                    "reg": f"reg-{i % 3}",
                },
            )
        g.cypher("CREATE CONSTRAINT item_sku FOR (n:Item) REQUIRE n.sku IS UNIQUE")
        g.cypher("CREATE INDEX FOR (n:Item) ON (n.category)")
        g.cypher("CREATE INDEX FOR (n:Item) ON (n.category, n.region)")
        g.cypher("CREATE RANGE INDEX FOR (n:Item) ON (n.score)")
    return str(path)


def _mode(graph):
    return graph.graph_info()["storage_mode"]


class TestStorageRequest:
    def test_recorded_mode_is_honoured_without_a_request(self, tmp_path):
        """The control the two override tests need: no argument, no change."""
        assert _mode(kglite.load(_seed(tmp_path / "m.kgl"))) == "memory"
        assert _mode(kglite.load(_seed(tmp_path / "p.kgl", storage="mapped"))) == "mapped"

    def test_request_overrides_the_recorded_mode_in_both_directions(self, tmp_path):
        memory_file = _seed(tmp_path / "m.kgl")
        mapped_file = _seed(tmp_path / "p.kgl", storage="mapped")

        promoted = kglite.load(memory_file, storage="mapped")
        assert _mode(promoted) == "mapped"
        assert promoted.cypher("MATCH (n:Item) RETURN count(n) AS c").scalar() == 40

        demoted = kglite.load(mapped_file, storage="memory")
        assert _mode(demoted) == "memory"
        assert demoted.cypher("MATCH (n:Item) RETURN count(n) AS c").scalar() == 40

    def test_disk_request_on_a_kgl_is_refused_naming_the_alternative(self, tmp_path):
        with pytest.raises(kglite.ArgumentError) as excinfo:
            kglite.load(_seed(tmp_path / "m.kgl"), storage="disk")
        message = str(excinfo.value)
        assert "enable_disk_mode()" in message
        assert "directory" in message

    def test_unknown_mode_is_rejected(self, tmp_path):
        with pytest.raises(kglite.ArgumentError, match="Unknown storage mode"):
            kglite.load(_seed(tmp_path / "m.kgl"), storage="banana")

    def test_from_bytes_takes_the_same_request(self, tmp_path):
        data = kglite.load(_seed(tmp_path / "m.kgl")).to_bytes()
        assert _mode(kglite.from_bytes(data, storage="mapped")) == "mapped"
        with pytest.raises(kglite.ArgumentError):
            kglite.from_bytes(data, storage="disk")


class TestDeferIndexRebuild:
    """The deferral is invisible to answers and visible to listings."""

    QUERIES = [
        "MATCH (n:Item {category: 'cat-2'}) RETURN count(n) AS c",
        "MATCH (n:Item {category: 'cat-1', region: 'reg-2'}) RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.score > 30 RETURN count(n) AS c",
        "MATCH (n:Item {sku: 'sku-17'}) RETURN n.id AS id",
        # A value nothing carries: the arm a wrongly-present index turns into
        # a proven-empty answer.
        "MATCH (n:Item {category: 'nope'}) RETURN count(n) AS c",
    ]

    def test_answers_are_identical(self, tmp_path):
        path = _seed(tmp_path / "m.kgl")
        eager = kglite.load(path, defer_index_rebuild=False)
        deferred = kglite.load(path, defer_index_rebuild=True)

        for query in self.QUERIES:
            assert deferred.cypher(query).to_list() == eager.cypher(query).to_list(), query

    def test_listings_show_the_declarations_while_predicates_stay_absent(self, tmp_path):
        path = _seed(tmp_path / "m.kgl")
        eager = kglite.load(path, defer_index_rebuild=False)
        deferred = kglite.load(path, defer_index_rebuild=True)

        def names(graph):
            return sorted(f"{i['node_type']}.{i['property']}" for i in graph.list_indexes())

        assert names(deferred) == names(eager)
        assert names(deferred), "the fixture declares equality indexes"
        assert {i["state"] for i in deferred.list_indexes()} == {"DEFERRED"}
        assert {i["state"] for i in eager.list_indexes()} == {"ONLINE"}
        assert {i["state"] for i in deferred.list_composite_indexes()} == {"DEFERRED"}

        # SHOW INDEXES projects the same collector.
        states = {row["state"] for row in deferred.cypher("SHOW INDEXES").to_list()}
        assert states == {"DEFERRED"}
        assert {row["state"] for row in eager.cypher("SHOW INDEXES").to_list()} == {"ONLINE"}

        # The constraint is listed too — it carries no state because there is
        # nothing to distinguish: enforcement materializes before any write.
        assert deferred.cypher("SHOW CONSTRAINTS").to_list() == (eager.cypher("SHOW CONSTRAINTS").to_list())

        # …and every predicate on the very same graph still says "no index",
        # which is what keeps the matcher scanning instead of proving empty.
        assert deferred.has_index("Item", "category") is False
        assert deferred.has_composite_index("Item", ["category", "region"]) is False
        assert eager.has_index("Item", "category") is True

    def test_the_first_write_materializes(self, tmp_path):
        path = _seed(tmp_path / "m.kgl")
        deferred = kglite.load(path, defer_index_rebuild=True)
        assert deferred.has_index("Item", "category") is False

        deferred.cypher("CREATE (:Item {id: 900, sku: 'sku-900', category: 'cat-2', region: 'reg-1', score: 99})")

        assert deferred.has_index("Item", "category") is True
        assert {i["state"] for i in deferred.list_indexes()} == {"ONLINE"}
        assert deferred.cypher("MATCH (n:Item {category: 'cat-2'}) RETURN count(n) AS c").scalar() == 9
        # The unique constraint the deferred load never built still enforces.
        with pytest.raises(kglite.ConstraintViolationError):
            deferred.cypher("CREATE (:Item {id: 901, sku: 'sku-900'})")

    def test_open_session_and_from_bytes_take_the_option(self, tmp_path):
        path = _seed(tmp_path / "m.kgl")
        session = kglite.open_session(path, defer_index_rebuild=True)
        assert session.cypher("MATCH (n:Item {category: 'cat-2'}) RETURN count(n) AS c").scalar() == 8

        data = kglite.load(path).to_bytes()
        from_buffer = kglite.from_bytes(data, defer_index_rebuild=True)
        assert from_buffer.has_index("Item", "category") is False
        assert {i["state"] for i in from_buffer.list_indexes()} == {"DEFERRED"}

    def test_a_deferred_graph_saves_its_declarations(self, tmp_path):
        """Nothing is lost by never building: a re-save keeps every index."""
        path = _seed(tmp_path / "m.kgl")
        deferred = kglite.load(path, defer_index_rebuild=True)
        deferred.save(str(tmp_path / "again.kgl"))

        reloaded = kglite.load(str(tmp_path / "again.kgl"))
        assert sorted(f"{i['node_type']}.{i['property']}" for i in reloaded.list_indexes()) == (
            sorted(
                f"{i['node_type']}.{i['property']}" for i in kglite.load(path, defer_index_rebuild=False).list_indexes()
            )
        )
        assert reloaded.has_index("Item", "category") is True
