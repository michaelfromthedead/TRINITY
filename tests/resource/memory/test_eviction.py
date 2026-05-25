"""Tests for eviction policies and manager."""

from engine.resource.memory.eviction import (
    EvictionCandidate,
    EvictionManager,
    LFUEviction,
    LRUEviction,
    PriorityEviction,
    SizeEviction,
)

_SIZE_SMALL = 100
_SIZE_MEDIUM = 200
_SIZE_LARGE = 500


def _make_candidates() -> list[EvictionCandidate]:
    return [
        EvictionCandidate(asset_id=1, size_bytes=_SIZE_SMALL, last_access_time=3.0, access_count=10, priority=5),
        EvictionCandidate(asset_id=2, size_bytes=_SIZE_MEDIUM, last_access_time=1.0, access_count=2, priority=1),
        EvictionCandidate(asset_id=3, size_bytes=_SIZE_LARGE, last_access_time=2.0, access_count=5, priority=3),
    ]


class TestLRUEviction:
    def test_selects_least_recently_used(self) -> None:
        policy = LRUEviction()
        candidates = _make_candidates()
        result = policy.select_for_eviction(candidates, _SIZE_MEDIUM)
        assert result[0] == 2  # last_access_time=1.0 is oldest

    def test_collects_enough_bytes(self) -> None:
        policy = LRUEviction()
        candidates = _make_candidates()
        result = policy.select_for_eviction(candidates, _SIZE_SMALL + _SIZE_MEDIUM + 1)
        assert len(result) >= 2


class TestLFUEviction:
    def test_selects_least_frequently_used(self) -> None:
        policy = LFUEviction()
        candidates = _make_candidates()
        result = policy.select_for_eviction(candidates, _SIZE_SMALL)
        assert result[0] == 2  # access_count=2 is lowest


class TestSizeEviction:
    def test_selects_largest_first(self) -> None:
        policy = SizeEviction()
        candidates = _make_candidates()
        result = policy.select_for_eviction(candidates, _SIZE_LARGE)
        assert result[0] == 3  # size=500 is largest


class TestPriorityEviction:
    def test_selects_lowest_priority_first(self) -> None:
        policy = PriorityEviction()
        candidates = _make_candidates()
        result = policy.select_for_eviction(candidates, _SIZE_SMALL)
        assert result[0] == 2  # priority=1 is lowest


class TestEvictionManager:
    def test_add_and_run_eviction(self) -> None:
        mgr = EvictionManager(LRUEviction())
        for c in _make_candidates():
            mgr.add_candidate(c)
        assert mgr.candidate_count() == 3
        evicted = mgr.run_eviction(_SIZE_MEDIUM)
        assert len(evicted) >= 1
        assert mgr.candidate_count() < 3

    def test_remove_candidate(self) -> None:
        mgr = EvictionManager(LRUEviction())
        mgr.add_candidate(EvictionCandidate(asset_id=10, size_bytes=_SIZE_SMALL))
        mgr.remove_candidate(10)
        assert mgr.candidate_count() == 0

    def test_empty_candidates_returns_empty(self) -> None:
        mgr = EvictionManager(LRUEviction())
        result = mgr.run_eviction(_SIZE_LARGE)
        assert result == []

    def test_eviction_removes_from_candidates(self) -> None:
        mgr = EvictionManager(SizeEviction())
        mgr.add_candidate(EvictionCandidate(asset_id=1, size_bytes=_SIZE_LARGE))
        mgr.add_candidate(EvictionCandidate(asset_id=2, size_bytes=_SIZE_SMALL))
        evicted = mgr.run_eviction(_SIZE_LARGE)
        assert 1 in evicted
        assert mgr.candidate_count() == 1
