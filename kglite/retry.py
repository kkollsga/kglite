"""Optimistic-concurrency retry helper.

KGLite commits a transaction by publishing its working copy with a pointer
swap, so *any* commit that lands between ``begin()`` and ``commit()`` makes
this transaction's snapshot stale — even one that touched entirely different
nodes. Conflicts are therefore an ordinary outcome under concurrency rather
than a rare edge case, and every correct writer needs a retry loop.

The loop is short but easy to get subtly wrong: a conflicted transaction is
already spent, so it must be *rebuilt* (not reused) on each attempt, the work
must be re-run against the fresh snapshot rather than replayed from the stale
one, and unbounded immediate retries livelock under contention.
:func:`retry_on_conflict` is that loop, done once.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar

from .kglite import TransactionConflictError

__all__ = ["retry_on_conflict"]

T = TypeVar("T")


def retry_on_conflict(
    graph: Any,
    work: Callable[[Any], T],
    *,
    attempts: int = 5,
    base_delay: float = 0.005,
    max_delay: float = 0.5,
    jitter: bool = True,
) -> T:
    """Run ``work`` in a transaction, retrying the whole unit on conflict.

    ``work`` is called as ``work(tx)`` with a fresh :class:`Transaction` and is
    re-invoked from the start on each attempt, so it must be safe to run more
    than once — read what you need *inside* it rather than closing over values
    read before the call. The transaction is committed when ``work`` returns
    and rolled back if it raises.

    Args:
        graph: The :class:`KnowledgeGraph` to transact against.
        work: Callable taking the transaction and returning the result.
        attempts: Maximum number of tries, including the first.
        base_delay: Seconds to wait before the second attempt. Doubles each
            further attempt (exponential backoff).
        max_delay: Upper bound on any single wait.
        jitter: Spread the backoff randomly over ``[0, delay]`` (full jitter).
            Keeps competing writers from retrying in lockstep; turn it off only
            for deterministic tests.

    Returns:
        Whatever ``work`` returned on the successful attempt.

    Raises:
        TransactionConflictError: Every attempt conflicted. The final error is
            re-raised unchanged, so ``.code`` and the version gap survive.
        ValueError: ``attempts`` is less than 1.
        Exception: Anything ``work`` raises is propagated immediately and the
            transaction rolled back — only conflicts are retried.

    Example:
        >>> def signup(tx):
        ...     tx.cypher("CREATE (u:User {email: $e})", params={"e": email})
        ...     return "created"
        >>> kglite.retry_on_conflict(graph, signup)
        'created'
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    for attempt in range(1, attempts + 1):
        try:
            with graph.begin() as tx:
                result = work(tx)
            return result
        except TransactionConflictError:
            # The transaction is spent either way; the last attempt re-raises
            # so the caller sees a real conflict rather than a silent failure.
            if attempt == attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            time.sleep(random.uniform(0, delay) if jitter else delay)

    # Unreachable: the final attempt either returns or re-raises.
    raise AssertionError("retry_on_conflict exhausted its loop without returning")
