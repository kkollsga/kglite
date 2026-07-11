"""Tests for GIL release / concurrency (Phase 5).

Verifies that read-only Cypher queries release the GIL and can run
concurrently from multiple Python threads.
"""

import threading
import time

import pandas as pd
import pytest

import kglite


@pytest.fixture
def large_graph():
    """Graph with enough nodes to make queries non-trivial."""
    g = kglite.KnowledgeGraph()
    n = 5000
    df = pd.DataFrame(
        {
            "id": list(range(n)),
            "title": [f"Person_{i}" for i in range(n)],
            "age": [20 + (i % 80) for i in range(n)],
            "city": [["Oslo", "Bergen", "Trondheim", "Stavanger"][i % 4] for i in range(n)],
        }
    )
    g.add_nodes(df, "Person", "id", "title")

    # Add some edges
    edges = pd.DataFrame(
        {
            "source": list(range(0, n - 1)),
            "target": list(range(1, n)),
            "type": ["KNOWS"] * (n - 1),
        }
    )
    g.add_connections(edges, "KNOWS", "Person", "source", "Person", "target")
    return g


class TestConcurrentReads:
    """Multiple threads can read the graph concurrently."""

    def test_concurrent_cypher_reads(self, large_graph):
        """Multiple threads running read-only Cypher should all complete correctly."""
        results = {}
        errors = []

        def query_thread(thread_id, city):
            try:
                result = large_graph.cypher(f"MATCH (n:Person) WHERE n.city = '{city}' RETURN count(n) AS cnt")
                results[thread_id] = result[0]["cnt"]
            except Exception as e:
                errors.append((thread_id, str(e)))

        cities = ["Oslo", "Bergen", "Trondheim", "Stavanger"]
        threads = []
        for i, city in enumerate(cities):
            t = threading.Thread(target=query_thread, args=(i, city))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 4
        # Each city has ~1250 nodes (5000 / 4)
        for count in results.values():
            assert count == 1250

    def test_concurrent_reads_produce_correct_results(self, large_graph):
        """Results from concurrent reads match sequential reads."""
        # Get sequential baseline
        sequential = large_graph.cypher("MATCH (n:Person) WHERE n.age > 50 RETURN count(n) AS cnt")[0]["cnt"]

        results = []
        errors = []

        def query_thread():
            try:
                result = large_graph.cypher("MATCH (n:Person) WHERE n.age > 50 RETURN count(n) AS cnt")
                results.append(result[0]["cnt"])
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=query_thread) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 8
        assert all(r == sequential for r in results)

    def test_concurrent_reads_result_equivalence(self, large_graph):
        """Phase 10: 4 ThreadPoolExecutor workers returning full result sets
        must equal the sequentially-computed baseline row-for-row."""
        from concurrent.futures import ThreadPoolExecutor

        query = "MATCH (p:Person)-[:KNOWS]->(q:Person) WHERE p.age > 30 AND q.age > 30 RETURN p.id AS pid, q.id AS qid"

        def normalise(rows):
            return sorted(
                (dict(r) for r in rows),
                key=lambda r: (r["pid"], r["qid"]),
            )

        baseline = normalise(large_graph.cypher(query))
        assert baseline, "query must produce non-empty baseline"

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(lambda: large_graph.cypher(query)) for _ in range(4)]
            worker_results = [normalise(f.result(timeout=10)) for f in futures]

        for i, got in enumerate(worker_results):
            assert got == baseline, f"worker {i} diverged from sequential baseline"

    def test_concurrent_disk_reads_keep_materialized_nodes_alive(self, tmp_path):
        """Overlapping disk queries must not reclaim another query's node arena."""
        from concurrent.futures import ThreadPoolExecutor

        graph = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "concurrent-disk"))
        node_count = 400
        graph.add_nodes(
            pd.DataFrame(
                {
                    "id": range(node_count),
                    "title": [f"Node {i}" for i in range(node_count)],
                    "payload": [f"payload-{i:04d}-" * 8 for i in range(node_count)],
                }
            ),
            "Item",
            "id",
            "title",
        )

        query = "MATCH (n:Item) RETURN n ORDER BY n.id"
        baseline = list(graph.cypher(query))
        assert len(baseline) == node_count

        workers = 8
        rounds = 12
        barrier = threading.Barrier(workers)

        def read_repeatedly():
            for _ in range(rounds):
                barrier.wait(timeout=10)
                assert list(graph.cypher(query)) == baseline

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(read_repeatedly) for _ in range(workers)]
            for future in futures:
                future.result(timeout=30)


class TestReadWriteIsolation:
    """Reads and writes don't interfere."""

    def test_read_during_no_mutation(self, large_graph):
        """Simple sanity: reads work fine when no mutations happening."""
        result = large_graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
        assert result[0]["cnt"] == 5000


class TestGILReleasePerformance:
    """GIL release should allow other Python code to run during heavy computations."""

    def test_gil_released_during_centrality(self, large_graph):
        """While a centrality computation runs, other Python threads can make progress."""
        counter = {"value": 0}
        stop_event = threading.Event()

        def increment_counter():
            """Simple Python thread that increments a counter."""
            while not stop_event.is_set():
                counter["value"] += 1
                time.sleep(0.001)

        # Start the counter thread
        counter_thread = threading.Thread(target=increment_counter, daemon=True)
        counter_thread.start()

        # Run pagerank (releases GIL during computation, allowing counter to increment)
        large_graph.pagerank()

        stop_event.set()
        counter_thread.join(timeout=2)

        # The counter should have incremented at least once during the computation
        # (if GIL wasn't released, counter would stay at 0)
        assert counter["value"] > 0


# ── Phase A.3 / 0.9.53 — Bolt-scale concurrency stress + documented quirks ──


class TestBoltScaleConcurrency:
    """16+ concurrent readers (the expected lower-bound for a Bolt server's
    session count) must not panic, deadlock, or produce inconsistent results."""

    def test_16_concurrent_readers_complete_correctly(self, large_graph):
        """16 threads × 4 queries each = 64 reads completing within budget,
        all returning the sequential baseline."""
        from concurrent.futures import ThreadPoolExecutor

        baseline = large_graph.cypher("MATCH (n:Person) WHERE n.city = 'Oslo' RETURN count(n) AS cnt")[0]["cnt"]
        query = "MATCH (n:Person) WHERE n.city = 'Oslo' RETURN count(n) AS cnt"

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(lambda: large_graph.cypher(query)) for _ in range(64)]
            results = [f.result(timeout=20)[0]["cnt"] for f in futures]

        assert len(results) == 64
        assert all(r == baseline for r in results), "concurrent reads diverged"

    def test_no_panic_under_high_contention(self, large_graph):
        """32 threads, half readers / half mutators, run for 500 ms over a
        shared ``Session`` — the supported concurrent handle (what the Bolt
        server uses). Pin that there are no errors and no panic.

        Note: this deliberately shares a ``Session``, not a bare
        ``KnowledgeGraph``. Sharing a live ``KnowledgeGraph`` across mutating
        threads is unsupported and is *correctly* rejected by the single-owner
        guard — under a free-threaded (no-GIL) interpreter that guard fires on
        every real overlap; under the GIL it was masked by serialization. The
        Session path (lock-free reads + serialized composable writes) is the
        contract we actually guarantee, so that's what we stress here.
        """
        from concurrent.futures import ThreadPoolExecutor

        session = large_graph.session()
        stop_event = threading.Event()
        errors: list[str] = []

        def reader():
            while not stop_event.is_set():
                try:
                    session.cypher("MATCH (n:Person) RETURN count(n) AS cnt")
                except Exception as e:
                    errors.append(f"reader: {e!r}")

        def mutator(idx):
            while not stop_event.is_set():
                try:
                    session.execute(f"CREATE (:Marker {{tid: {idx}}})")
                except Exception as e:
                    errors.append(f"mutator-{idx}: {e!r}")

        with ThreadPoolExecutor(max_workers=32) as pool:
            for _ in range(16):
                pool.submit(reader)
            for i in range(16):
                pool.submit(mutator, i)
            time.sleep(0.5)
            stop_event.set()

        assert errors == [], f"errors during contention: {errors[:5]}"


class TestDocumentedQuirks:
    """Pin the two contention quirks documented in concurrency.md so future
    refactors don't silently change the behavior shape."""

    def test_arc_makemut_cow_isolates_reader_from_mutation(self, large_graph):
        """While a read-only transaction holds an Arc snapshot of the graph,
        an outside-the-tx mutation triggers a CoW clone in Arc::make_mut.
        The reader's snapshot is preserved (pre-mutation state); the outside
        view reflects the new state."""
        # Open a read-only tx — holds an Arc reference.
        tx = large_graph.begin_read()
        baseline_count = tx.cypher("MATCH (n:Person) RETURN count(n) AS cnt")[0]["cnt"]
        assert baseline_count == 5000

        # Outside mutation while reader holds the snapshot.
        large_graph.cypher("CREATE (:Person {id: 99999, title: 'Latecomer'})")

        # The reader still sees the original count (snapshot isolation).
        post_count = tx.cypher("MATCH (n:Person) RETURN count(n) AS cnt")[0]["cnt"]
        assert post_count == 5000, "read-only snapshot must not see post-begin mutations"

        # Outside view reflects the mutation.
        outside_count = large_graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt")[0]["cnt"]
        assert outside_count == 5001, "outside graph must reflect the mutation"

    def test_wkt_cache_warmup_is_safe_under_contention(self):
        """Spatial cypher hits an Arc<RwLock<HashMap>> wkt_cache that takes a
        write lock on first encounter with each WKT string. Pin that 8 threads
        each parsing the same novel WKT complete without panic or wrong
        results (the read-after-warmup path stays parallel)."""
        from concurrent.futures import ThreadPoolExecutor

        # Build a tiny spatial graph (separate from `large_graph` because the
        # area needs to be created via add_nodes() to register the id field
        # in the schema; CREATE on a fresh graph auto-assigns numeric ids).
        g = kglite.KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame(
                [
                    {
                        "id": "unit",
                        "title": "UNIT_SQUARE",
                        "wkt_geometry": "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                    }
                ]
            ),
            "Area",
            "id",
            "title",
        )

        # Multi-thread evaluation of the same WKT-touching query — first
        # thread warms the cache, the rest hit it from the read path.
        query = "MATCH (a:Area {id: 'unit'}) RETURN contains(a, point(0.5, 0.5)) AS inside"
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(lambda: g.cypher(query)) for _ in range(8)]
            results = [f.result(timeout=10)[0]["inside"] for f in futures]
        assert all(r is True for r in results), "spatial reads under contention diverged"
