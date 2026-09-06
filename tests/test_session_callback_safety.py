"""Bounded child processes detect callback lock cycles without hanging pytest."""

import subprocess
import sys
import textwrap

import pytest

CHILD = textwrap.dedent(
    r"""
    import json
    import sys
    import threading

    import kglite
    from tests.test_execution_service_propagation import fixture

    mode, reader_kind = sys.argv[1:]
    graph, model = fixture()
    graph.cypher("MATCH(d:Doc) SET d.marker=0")
    session = graph.session()
    version = session.version()
    entered = threading.Event()
    reader_started = threading.Event()
    reader_done = threading.Event()
    failures = []
    original_embed = model.embed
    def read_committed():
        if reader_kind == 'cypher':
            assert session.cypher('MATCH(d:Doc) RETURN d.marker AS m ORDER BY d.id').to_list() == [{'m':0}, {'m':0}]
        elif reader_kind == 'execute_read':
            assert session.execute('RETURN 1 AS value').to_list() == [{'value':1}]
        elif reader_kind == 'snapshot':
            rows = session.snapshot().cypher(
                'MATCH(d:Doc) RETURN d.marker AS m ORDER BY d.id'
            ).to_list()
            assert rows == [{'m':0}, {'m':0}]
        elif reader_kind == 'cursor':
            rows = session.cursor().cypher(
                'MATCH(d:Doc) RETURN d.marker AS m ORDER BY d.id'
            ).to_list()
            assert rows == [{'m':0}, {'m':0}]
        elif reader_kind == 'version':
            assert session.version() == version
        elif reader_kind == 'node_count':
            assert session.node_count() == 2
        elif reader_kind == 'node_types':
            assert session.node_types == ['Doc']
        else:
            assert f'nodes=2, types=1, version={version}' in repr(session)
    def callback(texts):
        print('callback_entered', flush=True)
        if mode == 'concurrent':
            entered.set()
            assert reader_started.wait(2), 'reader did not enter'
            assert reader_done.wait(2), 'reader did not finish'
        elif mode == 'reentrant_write':
            try:
                session.execute('MATCH(d:Doc) SET d.marker=99')
            except kglite.ArgumentError as error:
                assert error.code == 'InvalidArgument'
                assert 'same Session' in str(error)
            else:
                raise AssertionError('reentrant write was accepted')
        else:
            read_committed()
        return original_embed(texts)
    model.embed = callback
    def reader():
        try:
            assert entered.wait(2), 'callback did not enter'
            reader_started.set()
            print('reader_entered', flush=True)
            read_committed()
        except BaseException as error:
            failures.append(repr(error))
        finally:
            reader_done.set()
    thread = None
    if mode == 'concurrent':
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
    before = session.snapshot()
    rows = session.execute(
        "MATCH(d:Doc) SET d.marker=1,d.score=text_score(d,'body','query') "
        "RETURN d.id AS id,d.score AS score ORDER BY id"
    ).to_list()
    if thread:
        thread.join(2)
        assert not thread.is_alive()
    assert failures == [], failures
    assert rows == [{'id':1,'score':1.0}, {'id':2,'score':0.0}], rows
    assert session.cypher('MATCH(d:Doc) RETURN d.marker AS m ORDER BY d.id').to_list() == [{'m':1}, {'m':1}]
    assert before.cypher('MATCH(d:Doc) RETURN d.marker AS m ORDER BY d.id').to_list() == [{'m':0}, {'m':0}]
    assert session.version() == version + 1
    # An explained write takes the write route but must not publish its working fork.
    after = session.version()
    explained = session.execute('EXPLAIN CREATE (:NoOp)').to_list()
    assert explained[0]['operation'] == 'Create'
    assert session.version() == after
    # Successful unwinding clears the per-thread write guard.
    session.execute('MATCH(d:Doc) SET d.marker=2')
    assert session.cypher('MATCH(d:Doc) RETURN d.marker AS m ORDER BY d.id').to_list() == [{'m':2}, {'m':2}]
    print(json.dumps({'complete':True,'mode':mode,'reader':reader_kind}), flush=True)
    """
)


@pytest.mark.parametrize("mode", ["self_read", "concurrent"])
@pytest.mark.parametrize(
    "reader", ["cypher", "execute_read", "snapshot", "cursor", "version", "node_count", "node_types", "repr"]
)
def test_callback_reads_committed_session_without_lock_cycle(mode, reader):
    run_child(mode, reader)


def test_callback_reentrant_write_refuses_before_lock_and_leaves_outer_write_usable():
    run_child("reentrant_write", "cypher")


def run_child(mode, reader):
    process = subprocess.Popen(
        [sys.executable, "-c", CHILD, mode, reader],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=2)
        pytest.fail(f"Session callback lock cycle: {mode}/{reader}\n{stdout}\n{stderr}")
    assert process.returncode == 0, (stdout, stderr)
    assert "callback_entered" in stdout
    if mode == "concurrent":
        assert "reader_entered" in stdout
    assert '"complete": true' in stdout
