"""Tombstone errors for the 0.14 removals (code_tree / datasets).

Not a compat shim — nothing old works. These pin that a migrating user
gets the two-line remediation instead of a bare AttributeError, and that
graceful feature-probing (`hasattr`) still sees the names as absent.
DELETE this file together with the tombstones in 0.15.
"""

import pytest

import kglite


@pytest.mark.parametrize("name", ["code_tree", "build_code_tree", "repo_tree", "datasets"])
def test_tombstone_raises_with_remediation(name):
    with pytest.raises(AttributeError) as exc:
        getattr(kglite, name)
    msg = str(exc.value)
    assert "kglite<0.14" in msg, f"pin-back escape missing: {msg}"
    assert "codingest" in msg or "kglite-datasets" in msg
    assert "migrations/0.13-to-0.14" in msg


@pytest.mark.parametrize("name", ["code_tree", "datasets"])
def test_feature_probing_stays_graceful(name):
    # hasattr() swallows AttributeError — probers take their False branch
    # silently instead of crashing on the tombstone.
    assert not hasattr(kglite, name)


def test_from_import_still_raises():
    # CPython's from-import machinery replaces the __getattr__ message with
    # its own generic ImportError — the guided text only appears on
    # attribute access (`kglite.code_tree`), which is the common runtime
    # break. Pin that from-import at least still fails cleanly.
    with pytest.raises(ImportError):
        from kglite import code_tree  # noqa: F401


def test_unknown_attribute_error_is_untouched():
    with pytest.raises(AttributeError) as exc:
        kglite.definitely_not_a_thing
    assert "kglite<0.14" not in str(exc.value)
