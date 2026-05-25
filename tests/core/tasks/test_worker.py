"""Tests for engine.core.tasks.worker — WorkerPool and work-stealing."""

import threading
import time

import pytest

from engine.core.tasks.worker import (
    TaskPriority,
    TaskAffinity,
    WorkItem,
    WorkerPool,
    WorkerThread,
)
from engine.core.tasks.sync import Future


@pytest.fixture
def pool():
    p = WorkerPool(num_workers=2)
    p.start()
    yield p
    p.shutdown(timeout=2)


class TestWorkerPoolLifecycle:
    def test_start_stop(self):
        p = WorkerPool(num_workers=2)
        assert not p.running
        p.start()
        assert p.running
        assert p.num_workers == 2
        p.shutdown()
        assert not p.running

    def test_min_one_worker(self):
        p = WorkerPool(num_workers=0)
        assert p.num_workers == 1
        p.start()
        p.shutdown()

    def test_double_start_noop(self):
        p = WorkerPool(num_workers=1)
        p.start()
        count_before = p.num_workers
        p.start()  # should not raise or add workers
        assert p.num_workers == count_before
        p.shutdown()

    def test_shutdown_idempotent(self):
        p = WorkerPool(num_workers=1)
        p.start()
        p.shutdown()
        assert not p.running
        p.shutdown()  # should not raise
        assert not p.running


class TestTaskSubmission:
    def test_submit_and_get_result(self, pool):
        f = Future()
        item = WorkItem(
            priority=TaskPriority.NORMAL,
            seq=0,
            func=lambda: 42,
            future=f,
        )
        pool.submit(item)
        result = f.get(timeout=2)
        assert result == 42

    def test_submit_with_args(self, pool):
        f = Future()
        item = WorkItem(
            priority=TaskPriority.NORMAL,
            seq=0,
            func=lambda a, b: a + b,
            args=(3, 4),
            future=f,
        )
        pool.submit(item)
        assert f.get(timeout=2) == 7

    def test_exception_captured(self, pool):
        f = Future()

        def bad():
            raise ValueError("oops")

        item = WorkItem(priority=TaskPriority.NORMAL, seq=0, func=bad, future=f)
        pool.submit(item)
        with pytest.raises(ValueError, match="oops"):
            f.get(timeout=2)

    def test_multiple_submissions(self, pool):
        futures = []
        for i in range(20):
            f = Future()
            item = WorkItem(
                priority=TaskPriority.NORMAL,
                seq=0,
                func=lambda x=i: x * 2,
                future=f,
            )
            pool.submit(item)
            futures.append(f)

        results = [f.get(timeout=3) for f in futures]
        assert sorted(results) == sorted(i * 2 for i in range(20))


class TestPriorityOrdering:
    def test_priority_enum_ordering(self):
        assert TaskPriority.CRITICAL < TaskPriority.HIGH
        assert TaskPriority.HIGH < TaskPriority.NORMAL
        assert TaskPriority.NORMAL < TaskPriority.LOW
        assert TaskPriority.LOW < TaskPriority.IDLE

    def test_work_item_sorts_by_priority(self):
        items = [
            WorkItem(priority=TaskPriority.LOW, seq=0, func=lambda: None),
            WorkItem(priority=TaskPriority.CRITICAL, seq=1, func=lambda: None),
            WorkItem(priority=TaskPriority.NORMAL, seq=2, func=lambda: None),
        ]
        items.sort()
        assert items[0].priority == TaskPriority.CRITICAL
        assert items[1].priority == TaskPriority.NORMAL
        assert items[2].priority == TaskPriority.LOW


class TestWorkStealing:
    def test_steal_from_worker(self):
        """Directly test the steal mechanism on WorkerThread."""
        p = WorkerPool(num_workers=2)
        w = p._workers[0]
        item = WorkItem(priority=TaskPriority.NORMAL, seq=0, func=lambda: 99)
        w.push(item)
        stolen = w.steal()
        assert stolen is item

    def test_steal_empty_returns_none(self):
        p = WorkerPool(num_workers=2)
        w = p._workers[0]
        assert w.steal() is None

    def test_pop_empty_returns_none(self):
        p = WorkerPool(num_workers=1)
        w = p._workers[0]
        assert w.pop() is None


class TestTaskAffinity:
    def test_affinity_values_exist_and_cover_expected(self):
        expected = {"any", "main", "worker", "io"}
        actual = {member.value for member in TaskAffinity}
        assert actual == expected
        assert len(TaskAffinity) == 4
