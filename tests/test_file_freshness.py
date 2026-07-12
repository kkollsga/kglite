"""File-freshness helpers — the binding-layer answer to "is a node's linked file
stale?" The engine never reads the filesystem (that's the no-fs posture line);
these run in Python (which has fs access): `stamp_file_freshness` captures the
file state into properties, `check_file_freshness` re-checks for drift.
"""

import hashlib
import os
import threading

import pytest

import kglite


def _graph(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello")
    b.write_text("world")
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Artifact {id: 'x', file_path: $a}), (:Artifact {id: 'y', file_path: $b})",
        params={"a": str(a), "b": str(b)},
    )
    return g, a, b


def test_stamp_then_no_drift(tmp_path):
    g, a, b = _graph(tmp_path)
    assert kglite.stamp_file_freshness(g, node_type="Artifact") == 2
    assert kglite.check_file_freshness(g, node_type="Artifact") == []


def test_detects_changed_and_missing(tmp_path):
    g, a, b = _graph(tmp_path)
    kglite.stamp_file_freshness(g, node_type="Artifact")
    a.write_text("CHANGED")  # content differs
    b.unlink()  # gone
    drift = {d["id"]: d["status"] for d in kglite.check_file_freshness(g, node_type="Artifact")}
    assert drift == {"x": "changed", "y": "missing"}


def test_stamped_mtime_is_queryable(tmp_path):
    g, a, b = _graph(tmp_path)
    kglite.stamp_file_freshness(g, node_type="Artifact")
    # file_mtime lands as a real (queryable) Timestamp value, not a string.
    m = g.cypher("MATCH (n:Artifact {id: 'x'}) RETURN n.file_mtime AS m").to_dicts()[0]["m"]
    assert m is not None


def test_check_is_read_only(tmp_path):
    """The drift check never mutates the graph (no updated_at churn)."""
    g, a, b = _graph(tmp_path)
    kglite.stamp_file_freshness(g, node_type="Artifact")
    before = g.cypher("MATCH (n:Artifact) RETURN count(n) AS c").to_dicts()[0]["c"]
    kglite.check_file_freshness(g, node_type="Artifact")
    after = g.cypher("MATCH (n:Artifact) RETURN count(n) AS c").to_dicts()[0]["c"]
    assert before == after == 2


def test_custom_keyword_property_names_are_escaped(tmp_path):
    artifact = tmp_path / "keyword.txt"
    artifact.write_text("content")
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Artifact {id: 'x', `order`: $path})",
        params={"path": str(artifact)},
    )
    assert (
        kglite.stamp_file_freshness(
            g,
            node_type="Artifact",
            path_property="order",
            mtime_property="contains",
            hash_property="return",
        )
        == 1
    )
    assert g.cypher(
        "MATCH (n:Artifact {id: 'x'}) RETURN n.`contains` IS NOT NULL AS has_mtime, n.`return` IS NOT NULL AS has_hash"
    ).to_dicts() == [{"has_mtime": True, "has_hash": True}]


def test_custom_label_and_property_names_with_spaces(tmp_path):
    artifact = tmp_path / "spaced.txt"
    artifact.write_text("content")
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:`Build Artifact` {id: 'x', `file path`: $path})",
        params={"path": str(artifact)},
    )

    assert (
        kglite.stamp_file_freshness(
            g,
            node_type="Build Artifact",
            path_property="file path",
            mtime_property="file mtime",
            hash_property=None,
        )
        == 1
    )
    assert g.cypher("MATCH (n:`Build Artifact` {id: 'x'}) RETURN n.`file mtime` IS NOT NULL AS stamped").to_dicts() == [
        {"stamped": True}
    ]
    assert (
        kglite.check_file_freshness(
            g,
            node_type="Build Artifact",
            path_property="file path",
            mtime_property="file mtime",
            hash_property=None,
        )
        == []
    )


def test_identifier_with_backtick_is_rejected(tmp_path):
    g, _, _ = _graph(tmp_path)
    import pytest

    with pytest.raises(kglite.ArgumentError, match="backtick"):
        kglite.stamp_file_freshness(g, path_property="file_path` SET n.hacked = true //")


def test_stamp_rejects_non_positive_batch_size(tmp_path):
    g, _, _ = _graph(tmp_path)
    with pytest.raises(kglite.ArgumentError, match="batch_size must be greater than zero"):
        kglite.stamp_file_freshness(g, batch_size=0)


def test_stamp_uses_one_batched_mutation_query(tmp_path):
    g, _, _ = _graph(tmp_path)

    class RecordingGraph:
        def __init__(self, inner):
            self.inner = inner
            self.queries = []

        def cypher(self, query, **kwargs):
            self.queries.append(query)
            return self.inner.cypher(query, **kwargs)

        def begin(self):
            inner_tx = self.inner.begin()
            queries = self.queries

            class RecordingTransaction:
                def __enter__(self):
                    inner_tx.__enter__()
                    return self

                def cypher(self, query, **kwargs):
                    queries.append(query)
                    return inner_tx.cypher(query, **kwargs)

                def __exit__(self, *args):
                    return inner_tx.__exit__(*args)

            return RecordingTransaction()

    recording = RecordingGraph(g)
    assert kglite.stamp_file_freshness(recording, node_type="Artifact") == 2
    mutation_queries = [query for query in recording.queries if " SET " in query]
    assert len(mutation_queries) == 1
    assert mutation_queries[0].startswith("UNWIND $rows AS row")


def test_same_id_across_types_updates_only_the_matching_type(tmp_path):
    artifact = tmp_path / "shared.txt"
    artifact.write_text("content")
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Artifact {id: 'same', file_path: $path}), (:Other {id: 'same', file_path: $path})",
        params={"path": str(artifact)},
    )

    assert kglite.stamp_file_freshness(g, node_type="Artifact") == 1
    rows = g.cypher("MATCH (n) RETURN labels(n)[0] AS type, n.file_mtime AS mtime ORDER BY type").to_dicts()
    assert rows == [
        {"type": "Artifact", "mtime": rows[0]["mtime"]},
        {"type": "Other", "mtime": None},
    ]
    assert rows[0]["mtime"].endswith("Z")


def test_duplicate_resolved_paths_are_snapshotted_once(tmp_path, monkeypatch):
    artifact = tmp_path / "shared.txt"
    artifact.write_text("content")
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Artifact {id: 1, file_path: $a}), (:Artifact {id: 2, file_path: $b})",
        params={"a": str(artifact), "b": str(tmp_path / "." / "shared.txt")},
    )

    real_snapshot = kglite._snapshot_file
    calls = []

    def recording_snapshot(path, *, include_hash):
        calls.append(path)
        return real_snapshot(path, include_hash=include_hash)

    monkeypatch.setattr(kglite, "_snapshot_file", recording_snapshot)
    assert kglite.stamp_file_freshness(g) == 2
    assert calls == [artifact.resolve()]


def test_snapshot_cache_has_a_fixed_entry_bound(tmp_path, monkeypatch):
    from collections import OrderedDict

    monkeypatch.setattr(kglite, "_FILE_SNAPSHOT_CACHE_SIZE", 2)
    monkeypatch.setattr(kglite, "_snapshot_file", lambda path, *, include_hash: {"path": path})
    cache = OrderedDict()
    paths = [tmp_path / name for name in ("a", "b", "c")]
    for path in paths:
        kglite._cached_file_snapshot(cache, path, include_hash=False)
    assert list(cache) == paths[1:]


def test_descriptor_snapshot_retries_path_replacement(tmp_path, monkeypatch):
    target = tmp_path / "target.txt"
    replacement = tmp_path / "replacement.txt"
    target.write_text("old")
    replacement.write_text("new")
    real_stat = os.stat
    replaced = False

    def replace_before_identity_check(path, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            os.replace(replacement, target)
            replaced = True
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", replace_before_identity_check)
    snapshot = kglite._snapshot_file(target, include_hash=True)
    assert snapshot["hash"] == hashlib.sha256(b"new").hexdigest()


def test_descriptor_snapshot_has_bounded_reads(tmp_path, monkeypatch):
    target = tmp_path / "large.bin"
    target.write_bytes(b"x" * (kglite._FILE_SNAPSHOT_CHUNK_BYTES * 2 + 17))
    real_read = os.read
    requested = []

    def recording_read(descriptor, count):
        requested.append(count)
        return real_read(descriptor, count)

    monkeypatch.setattr(os, "read", recording_read)
    assert kglite._snapshot_file(target, include_hash=True)["hash"] is not None
    assert requested
    assert max(requested) == kglite._FILE_SNAPSHOT_CHUNK_BYTES


def test_descriptor_snapshot_retries_concurrent_in_place_mutation(tmp_path, monkeypatch):
    target = tmp_path / "mutating.bin"
    original = b"a" * (kglite._FILE_SNAPSHOT_CHUNK_BYTES + 8)
    changed = b"b" + original[1:]
    target.write_bytes(original)
    first_read = threading.Event()
    mutation_done = threading.Event()
    real_read = os.read
    read_calls = 0

    def coordinated_read(descriptor, count):
        nonlocal read_calls
        chunk = real_read(descriptor, count)
        read_calls += 1
        if read_calls == 1:
            first_read.set()
            assert mutation_done.wait(timeout=5)
        return chunk

    def mutate():
        assert first_read.wait(timeout=5)
        with target.open("r+b") as stream:
            stream.write(b"b")
            stream.flush()
            os.fsync(stream.fileno())
        mutation_done.set()

    monkeypatch.setattr(os, "read", coordinated_read)
    writer = threading.Thread(target=mutate)
    writer.start()
    snapshot = kglite._snapshot_file(target, include_hash=True)
    writer.join(timeout=5)
    assert not writer.is_alive()
    assert snapshot["hash"] == hashlib.sha256(changed).hexdigest()


def test_descriptor_snapshot_reports_persistently_unstable_file(tmp_path, monkeypatch):
    target = tmp_path / "unstable.txt"
    target.write_text("content")
    real_stat = os.stat

    class ChangedIdentity:
        def __init__(self, value):
            self._value = value
            self.st_dev = value.st_dev
            self.st_ino = value.st_ino + 1
            self.st_mode = value.st_mode
            self.st_size = value.st_size
            self.st_mtime_ns = value.st_mtime_ns
            self.st_ctime_ns = value.st_ctime_ns

    monkeypatch.setattr(os, "stat", lambda path: ChangedIdentity(real_stat(path)))
    with pytest.raises(RuntimeError, match=r"unstable across 3 snapshot attempts"):
        kglite._snapshot_file(target, include_hash=True)


def test_hash_none_detects_nanosecond_mtime_change(tmp_path):
    target = tmp_path / "precise.txt"
    target.write_text("unchanged")
    initial_ns = 1_800_000_000_123_456_100
    os.utime(target, ns=(initial_ns, initial_ns))
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Artifact {id: 1, file_path: $path})", params={"path": str(target)})

    assert kglite.stamp_file_freshness(g, hash_property=None) == 1
    stored = g.cypher("MATCH (n:Artifact) RETURN n.file_mtime AS m").scalar()
    assert stored == "2027-01-15T08:00:00.123456100Z"
    assert kglite.check_file_freshness(g, hash_property=None) == []

    changed_ns = initial_ns + 100
    os.utime(target, ns=(changed_ns, changed_ns))
    assert kglite.check_file_freshness(g, hash_property=None) == [{"id": 1, "path": str(target), "status": "changed"}]


def test_later_batch_failure_rolls_back_all_updates(tmp_path):
    g, _, _ = _graph(tmp_path)

    class FailingGraph:
        def cypher(self, query, **kwargs):
            return g.cypher(query, **kwargs)

        def begin(self):
            inner_tx = g.begin()

            class FailingTransaction:
                calls = 0

                def __enter__(self):
                    inner_tx.__enter__()
                    return self

                def cypher(self, query, **kwargs):
                    self.calls += 1
                    if self.calls == 2:
                        raise RuntimeError("injected later-batch failure")
                    return inner_tx.cypher(query, **kwargs)

                def __exit__(self, *args):
                    return inner_tx.__exit__(*args)

            return FailingTransaction()

    with pytest.raises(RuntimeError, match="injected later-batch failure"):
        kglite.stamp_file_freshness(FailingGraph(), node_type="Artifact", batch_size=1)
    assert g.cypher("MATCH (n:Artifact) RETURN n.file_mtime AS m ORDER BY n.id").to_dicts() == [
        {"m": None},
        {"m": None},
    ]
