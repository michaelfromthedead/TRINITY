"""Tests for engine.core.tasks.scheduler — TaskScheduler."""

import threading
import time

import pytest

from engine.core.tasks.scheduler import TaskScheduler, TaskHandle
from engine.core.tasks.worker import TaskPriority


@pytest.fixture
def scheduler():
    s = TaskScheduler(worker_count=2)
    yield s
    s.shutdown()


class TestTaskSchedulerLifecycle:
    def test_initialize_auto_detect(self):
        s = TaskScheduler()
        s.initialize(0)
        assert s.worker_count >= 1
        assert s.initialized
        s.shutdown()
        assert not s.initialized

    def test_initialize_explicit(self):
        s = TaskScheduler(worker_count=3)
        assert s.worker_count == 3
        s.shutdown()

    def test_shutdown_idempotent(self):
        s = TaskScheduler(worker_count=1)
        s.shutdown()
        assert not s.initialized
        s.shutdown()  # should not raise
        assert not s.initialized

    def test_double_initialize_noop(self):
        s = TaskScheduler(worker_count=2)
        s.initialize(4)  # second call ignored
        assert s.worker_count == 2
        s.shutdown()


class TestSubmitAndWait:
    def test_submit_returns_handle(self, scheduler):
        h = scheduler.submit(lambda: 42)
        assert isinstance(h, TaskHandle)

    def test_wait_returns_result(self, scheduler):
        h = scheduler.submit(lambda: 7 + 3)
        result = scheduler.wait(h)
        assert result == 10

    def test_is_complete(self, scheduler):
        event = threading.Event()
        h = scheduler.submit(lambda: event.wait(2))
        assert not scheduler.is_complete(h)
        event.set()
        scheduler.wait(h)
        assert scheduler.is_complete(h)

    def test_submit_with_args(self, scheduler):
        h = scheduler.submit(lambda a, b: a * b, 6, 7)
        assert scheduler.wait(h) == 42

    def test_submit_with_priority(self, scheduler):
        h = scheduler.submit(lambda: 1, priority=TaskPriority.HIGH)
        assert h.priority == TaskPriority.HIGH
        scheduler.wait(h)

    def test_exception_propagates(self, scheduler):
        def bad():
            raise RuntimeError("boom")

        h = scheduler.submit(bad)
        with pytest.raises(RuntimeError, match="boom"):
            scheduler.wait(h)


class TestSubmitAfter:
    def test_respects_single_dependency(self, scheduler):
        order = []

        def first():
            time.sleep(0.05)
            order.append("first")

        def second():
            order.append("second")

        h1 = scheduler.submit(first)
        h2 = scheduler.submit_after(second, [h1])
        scheduler.wait(h2)
        assert order == ["first", "second"]

    def test_respects_multiple_dependencies(self, scheduler):
        results = []
        lock = threading.Lock()

        def task(val):
            time.sleep(0.02)
            with lock:
                results.append(val)

        def final():
            with lock:
                results.append("final")

        h1 = scheduler.submit(task, "a")
        h2 = scheduler.submit(task, "b")
        h3 = scheduler.submit_after(final, [h1, h2])
        scheduler.wait(h3)
        assert "final" in results
        assert results.index("final") == len(results) - 1


class TestWaitAll:
    def test_wait_all(self, scheduler):
        handles = [scheduler.submit(lambda i=i: i * 2) for i in range(5)]
        results = scheduler.wait_all(handles)
        assert sorted(results) == [0, 2, 4, 6, 8]


class TestParallelFor:
    def test_basic(self, scheduler):
        results = {}
        lock = threading.Lock()

        def process(start, end):
            for i in range(start, end):
                with lock:
                    results[i] = i * i

        h = scheduler.parallel_for(10, 3, process)
        scheduler.wait(h)
        assert len(results) == 10
        for i in range(10):
            assert results[i] == i * i

    def test_empty_count(self, scheduler):
        h = scheduler.parallel_for(0, 4, lambda s, e: None)
        assert isinstance(h, TaskHandle)
        scheduler.wait(h)  # should not raise
        assert scheduler.is_complete(h)

    def test_chunk_larger_than_count(self, scheduler):
        called = []

        def process(start, end):
            called.append((start, end))

        h = scheduler.parallel_for(3, 100, process)
        scheduler.wait(h)
        assert called == [(0, 3)]
