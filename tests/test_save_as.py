"""Save-as transfers one writer while preserving both files' recovery state."""

import os
import subprocess
import sys
import textwrap

import pytest

import kglite


def ids(graph):
    return [row["id"] for row in graph.cypher("MATCH (n:Doc) RETURN n.id AS id ORDER BY id")]


@pytest.mark.parametrize("storage", ["memory", "mapped"])
@pytest.mark.parametrize("level", ["full", "normal"])
def test_save_as_crash_preserves_source_and_recovers_destination(tmp_path, storage, level):
    source = str(tmp_path / "source.kgl")
    target = str(tmp_path / "target.kgl")
    script = textwrap.dedent(f"""
        import os, kglite
        g = kglite.open({source!r}, storage={storage!r}, durable={level!r})
        g.cypher('CREATE (:Doc {{id: 1}})')
        g.save()
        g.cypher('CREATE (:Doc {{id: 2}})')
        g.save({target!r})
        g.cypher('CREATE (:Doc {{id: 3}})')
        os._exit(0)
    """)
    subprocess.run([sys.executable, "-c", script], check=True, env=dict(os.environ), timeout=30)
    assert ids(kglite.open(source)) == [1, 2]
    assert ids(kglite.open(target)) == [1, 2, 3]


@pytest.mark.parametrize("level", ["off", "full", "normal"])
def test_failed_save_as_keeps_original_home_and_lease(tmp_path, level):
    source = str(tmp_path / "source.kgl")
    g = kglite.open(source, durable=level)
    g.cypher("CREATE (:Doc {id: 1})")
    g.save()
    with pytest.raises(kglite.FileIoError):
        g.save(str(tmp_path / "missing" / "target.kgl"))
    with pytest.raises(kglite.FileIoError, match="open for writing"):
        kglite.open(source)
    g.cypher("CREATE (:Doc {id: 2})")
    g.save()
    assert ids(kglite.load(source)) == [1, 2]


@pytest.mark.parametrize("level", ["off", "full"])
def test_save_as_transfers_lease_and_refuses_live_target(tmp_path, level):
    source = str(tmp_path / "source.kgl")
    target = str(tmp_path / "target.kgl")
    g = kglite.open(source, durable=level)
    g.cypher("CREATE (:Doc {id: 1})")
    g.save()
    other = kglite.open(target, durable=level)
    other.cypher("CREATE (:Doc {id: 9})")
    other.save()
    with pytest.raises(kglite.FileIoError, match="open for writing"):
        g.save(target)
    assert ids(kglite.load(target)) == [9]
    other.close()
    g.save(target)
    with pytest.raises(kglite.FileIoError, match="open for writing"):
        kglite.open(target)
    assert ids(kglite.open(source)) == [1]
    assert ids(kglite.load(target)) == [1]


@pytest.mark.parametrize("source_writes", [1, 2, 5])
def test_save_as_checks_destination_checkpoint_not_source_lsn(tmp_path, source_writes):
    source = str(tmp_path / "source.kgl")
    target = str(tmp_path / "target.kgl")
    g = kglite.open(source)
    for node_id in range(source_writes):
        g.cypher("CREATE (:Doc {id: $id})", params={"id": node_id})
    g.save()
    script = textwrap.dedent(f"""
        import os, kglite
        g = kglite.open({target!r})
        g.cypher('CREATE (:Doc {{id: 9}})')
        g.save()
        g.cypher('CREATE (:Doc {{id: 10}})')
        os._exit(0)
    """)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
    with pytest.raises(ValueError, match="write-ahead"):
        g.save(target)
    assert ids(kglite.open(target)) == [9, 10]
    g.cypher("CREATE (:Doc {id: 20})")
    g.save()
    assert ids(kglite.load(source)) == [*range(source_writes), 20]


def test_same_path_alias_keeps_ownership_and_log(tmp_path):
    source = tmp_path / "source.kgl"
    g = kglite.open(str(source))
    g.cypher("CREATE (:Doc {id: 1})")
    g.save()
    g.save(str(tmp_path) + "/./source.kgl")
    g.cypher("CREATE (:Doc {id: 2})")
    g.save()
    assert ids(kglite.load(str(source))) == [1, 2]
    with pytest.raises(kglite.FileIoError, match="open for writing"):
        kglite.open(str(source))


def test_lock_opt_out_stays_detached_on_save_as(tmp_path):
    source = str(tmp_path / "source.kgl")
    target = str(tmp_path / "target.kgl")
    g = kglite.open(source, lock=False, durable="off")
    g.cypher("CREATE (:Doc {id: 1})")
    g.save(target)
    assert ids(kglite.open(target)) == [1]
