"""Tests for ObjectPool."""

from engine.core.memory.object_pool import ObjectPool


class _Dummy:
    def __init__(self):
        self.value = 0


class TestObjectPool:
    def test_acquire_creates_new(self):
        pool = ObjectPool(factory=_Dummy)
        obj = pool.acquire()
        assert isinstance(obj, _Dummy)

    def test_release_and_reuse(self):
        pool = ObjectPool(factory=_Dummy)
        obj = pool.acquire()
        obj.value = 42
        pool.release(obj)
        reused = pool.acquire()
        assert reused is obj
        # Stale state is preserved (no reset_func); document this behavior
        assert reused.value == 42

    def test_release_and_reuse_with_manual_reset(self):
        """Verify that callers can manually reset state before release."""
        pool = ObjectPool(factory=_Dummy)
        obj = pool.acquire()
        obj.value = 99
        # Manually reset before release
        obj.value = 0
        pool.release(obj)
        reused = pool.acquire()
        assert reused is obj
        assert reused.value == 0

    def test_initial_size(self):
        pool = ObjectPool(factory=_Dummy, initial_size=5)
        assert pool.available == 5
        assert pool.total_created == 5

    def test_available_count(self):
        pool = ObjectPool(factory=_Dummy, initial_size=3)
        pool.acquire()
        assert pool.available == 2

    def test_grows_on_demand(self):
        pool = ObjectPool(factory=_Dummy, initial_size=0)
        objs = [pool.acquire() for _ in range(10)]
        assert pool.total_created == 10
        for o in objs:
            pool.release(o)
        assert pool.available == 10
