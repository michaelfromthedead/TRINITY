"""Tests for ResidencyManager."""

from engine.resource.memory.budget_manager import AssetCategory, BudgetManager
from engine.resource.memory.eviction import LRUEviction
from engine.resource.memory.residency_manager import ResidencyManager, ResidencyState

_BUDGET = 1000
_ASSET_SIZE = 200
_ASSET_LARGE = 600


class TestResidencyManager:
    def _make_manager(self) -> ResidencyManager:
        bm = BudgetManager()
        bm.set_budget(AssetCategory.OTHER, _BUDGET)
        clock = _FakeClock()
        return ResidencyManager(bm, LRUEviction(), AssetCategory.OTHER, time_fn=clock)

    def test_request_residency_success(self) -> None:
        mgr = self._make_manager()
        assert mgr.request_residency(1, _ASSET_SIZE)
        assert mgr.get_state(1) == ResidencyState.RESIDENT

    def test_request_residency_over_budget(self) -> None:
        mgr = self._make_manager()
        assert mgr.request_residency(1, _BUDGET)
        assert not mgr.request_residency(2, 1)

    def test_release_residency(self) -> None:
        mgr = self._make_manager()
        mgr.request_residency(1, _ASSET_SIZE)
        mgr.release_residency(1)
        assert mgr.get_state(1) == ResidencyState.NON_RESIDENT
        assert mgr.get_resident_count() == 0

    def test_touch_updates_access(self) -> None:
        mgr = self._make_manager()
        mgr.request_residency(1, _ASSET_SIZE)
        old_access = mgr._assets[1].last_access
        # Advance clock so touch produces a different timestamp
        mgr._time_fn.advance(1.0)
        mgr.touch(1)
        assert mgr.get_state(1) == ResidencyState.RESIDENT
        assert mgr._assets[1].last_access > old_access

    def test_get_state_unknown_asset(self) -> None:
        mgr = self._make_manager()
        assert mgr.get_state(999) == ResidencyState.NON_RESIDENT

    def test_resident_count_and_bytes(self) -> None:
        mgr = self._make_manager()
        mgr.request_residency(1, _ASSET_SIZE)
        mgr.request_residency(2, _ASSET_SIZE)
        assert mgr.get_resident_count() == 2
        assert mgr.get_resident_bytes() == 2 * _ASSET_SIZE

    def test_eviction_on_update(self) -> None:
        bm = BudgetManager()
        bm.set_budget(AssetCategory.OTHER, _BUDGET)
        clock = _FakeClock()
        mgr = ResidencyManager(bm, LRUEviction(), AssetCategory.OTHER, time_fn=clock)
        # Fill budget
        mgr.request_residency(1, _ASSET_LARGE, priority=0)
        clock.advance(1.0)
        mgr.request_residency(2, _BUDGET - _ASSET_LARGE, priority=0)
        # Force over-budget
        entry = bm.get_usage(AssetCategory.OTHER)
        entry.used_bytes = _BUDGET + _ASSET_SIZE
        evicted = mgr.update()
        assert len(evicted) >= 1

    def test_request_already_resident_is_noop(self) -> None:
        mgr = self._make_manager()
        mgr.request_residency(1, _ASSET_SIZE)
        assert mgr.request_residency(1, _ASSET_SIZE)
        assert mgr.get_resident_count() == 1


class _FakeClock:
    """Deterministic clock for testing."""

    def __init__(self) -> None:
        self._time = 0.0

    def __call__(self) -> float:
        return self._time

    def advance(self, dt: float) -> None:
        self._time += dt
