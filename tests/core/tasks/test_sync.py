"""Tests for engine.core.tasks.sync — synchronization primitives."""

import threading
import time

import pytest

from engine.core.tasks.sync import TaskCounter, Future, Promise, Latch, Barrier


class TestTaskCounter:
    def test_initial_zero(self):
        c = TaskCounter(0)
        assert c.value == 0
        assert c.wait_until_zero(timeout=0.01)

    def test_increment_decrement(self):
        c = TaskCounter(0)
        c.increment(3)
        assert c.value == 3
        c.decrement(2)
        assert c.value == 1
        c.decrement(1)
        assert c.value == 0

    def test_wait_blocks_until_zero(self):
        c = TaskCounter(2)
        results = []

        def waiter():
            c.wait_until_zero(timeout=2)
            results.append("done")

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.02)
        assert results == []
        c.decrement(1)
        time.sleep(0.02)
        assert results == []
        c.decrement(1)
        t.join(timeout=1)
        assert results == ["done"]

    def test_decrement_below_zero_clamps(self):
        c = TaskCounter(1)
        c.decrement(5)
        assert c.value == 0
        assert c.wait_until_zero(timeout=0.01)


class TestFuturePromise:
    def test_promise_set_value(self):
        p = Promise()
        f = p.future
        assert not f.is_ready()
        p.set_value(42)
        assert f.is_ready()
        assert f.get() == 42

    def test_promise_set_exception(self):
        p = Promise()
        f = p.future
        p.set_exception(RuntimeError("fail"))
        with pytest.raises(RuntimeError, match="fail"):
            f.get()

    def test_future_wait(self):
        p = Promise()
        f = p.future

        def setter():
            time.sleep(0.03)
            p.set_value("ok")

        t = threading.Thread(target=setter)
        t.start()
        assert f.wait(timeout=2)
        assert f.get() == "ok"
        t.join()

    def test_future_timeout(self):
        f = Future()
        with pytest.raises(TimeoutError):
            f.get(timeout=0.01)


class TestLatch:
    def test_zero_count_already_open(self):
        l = Latch(0)
        assert l.try_wait()

    def test_count_down(self):
        l = Latch(3)
        assert not l.try_wait()
        l.count_down()
        l.count_down()
        assert not l.try_wait()
        l.count_down()
        assert l.try_wait()

    def test_wait_blocks(self):
        l = Latch(1)
        result = []

        def waiter():
            l.wait(timeout=2)
            result.append("released")

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.02)
        assert result == []
        l.count_down()
        t.join(timeout=1)
        assert result == ["released"]

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            Latch(-1)

    def test_count_property(self):
        l = Latch(5)
        assert l.count == 5
        l.count_down(3)
        assert l.count == 2


class TestBarrier:
    def test_single_party(self):
        b = Barrier(1)
        idx = b.arrive_and_wait(timeout=1)
        assert idx == 0

    def test_multiple_parties(self):
        b = Barrier(3)
        results = []
        lock = threading.Lock()

        def participant(pid):
            idx = b.arrive_and_wait(timeout=2)
            with lock:
                results.append(pid)

        threads = [threading.Thread(target=participant, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3)
        assert sorted(results) == [0, 1, 2]

    def test_reusable(self):
        b = Barrier(2)
        results = []

        def run():
            b.arrive_and_wait(timeout=2)
            results.append("phase1")
            b.arrive_and_wait(timeout=2)
            results.append("phase2")

        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start()
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)
        assert results.count("phase1") == 2
        assert results.count("phase2") == 2

    def test_invalid_count_raises(self):
        with pytest.raises(ValueError):
            Barrier(0)
