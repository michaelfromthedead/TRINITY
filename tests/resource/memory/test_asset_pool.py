"""Tests for AssetPool."""

import pytest

from engine.resource.memory.asset_pool import AssetPool, DEFAULT_POOL_CAPACITY


SMALL_POOL_SIZE = 4


class TestAssetPool:
    def test_default_capacity(self) -> None:
        pool: AssetPool[str] = AssetPool()
        assert pool.capacity() == DEFAULT_POOL_CAPACITY

    def test_custom_capacity(self) -> None:
        pool: AssetPool[str] = AssetPool(capacity=SMALL_POOL_SIZE)
        assert pool.capacity() == SMALL_POOL_SIZE
        assert pool.active_count() == 0

    def test_acquire_returns_slot_and_object(self) -> None:
        pool: AssetPool[str] = AssetPool(capacity=SMALL_POOL_SIZE)
        slot_id, obj = pool.acquire("texture_01")
        assert isinstance(slot_id, int)
        assert obj == "texture_01"
        assert pool.active_count() == 1

    def test_release_frees_slot(self) -> None:
        pool: AssetPool[str] = AssetPool(capacity=SMALL_POOL_SIZE)
        slot_id, _ = pool.acquire("mesh_01")
        pool.release(slot_id)
        assert pool.active_count() == 0
        assert pool.get(slot_id) is None

    def test_get_returns_object_or_none(self) -> None:
        pool: AssetPool[int] = AssetPool(capacity=SMALL_POOL_SIZE)
        slot_id, _ = pool.acquire(42)
        assert pool.get(slot_id) == 42
        assert pool.get(slot_id + SMALL_POOL_SIZE) is None
        assert pool.get(-1) is None

    def test_full_pool_raises(self) -> None:
        pool: AssetPool[int] = AssetPool(capacity=2)
        pool.acquire(1)
        pool.acquire(2)
        assert pool.is_full()
        with pytest.raises(RuntimeError):
            pool.acquire(3)

    def test_reset_clears_all(self) -> None:
        pool: AssetPool[str] = AssetPool(capacity=SMALL_POOL_SIZE)
        ids = [pool.acquire(f"a{i}")[0] for i in range(SMALL_POOL_SIZE)]
        assert pool.is_full()
        pool.reset()
        assert pool.active_count() == 0
        assert not pool.is_full()
        for sid in ids:
            assert pool.get(sid) is None

    def test_release_inactive_raises(self) -> None:
        pool: AssetPool[str] = AssetPool(capacity=SMALL_POOL_SIZE)
        slot_id, _ = pool.acquire("x")
        pool.release(slot_id)
        with pytest.raises(ValueError):
            pool.release(slot_id)

    def test_invalid_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            AssetPool(capacity=0)
