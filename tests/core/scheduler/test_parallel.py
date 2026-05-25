"""Tests for parallel dispatch and conflict detection."""

import threading

import pytest

from engine.core.scheduler.parallel import (
    ParallelDispatcher,
    SystemAccess,
    can_run_parallel,
    compute_parallel_groups,
)


class _Position:
    pass


class _Velocity:
    pass


class _Health:
    pass


class TestCanRunParallel:
    def test_read_read_ok(self):
        a = SystemAccess(reads={_Position}, writes=set())
        b = SystemAccess(reads={_Position}, writes=set())
        assert can_run_parallel(a, b) is True

    def test_read_write_conflict(self):
        a = SystemAccess(reads={_Position}, writes=set())
        b = SystemAccess(reads=set(), writes={_Position})
        assert can_run_parallel(a, b) is False

    def test_write_write_conflict(self):
        a = SystemAccess(reads=set(), writes={_Position})
        b = SystemAccess(reads=set(), writes={_Position})
        assert can_run_parallel(a, b) is False

    def test_disjoint_writes_ok(self):
        a = SystemAccess(reads=set(), writes={_Position})
        b = SystemAccess(reads=set(), writes={_Velocity})
        assert can_run_parallel(a, b) is True

    def test_no_access_ok(self):
        assert can_run_parallel(SystemAccess(), SystemAccess()) is True


class TestComputeParallelGroups:
    def test_non_conflicting_same_group(self):
        access = {
            0: SystemAccess(reads={_Position}),
            1: SystemAccess(reads={_Velocity}),
        }
        groups = compute_parallel_groups([0, 1], access)
        assert groups == [[0, 1]]

    def test_conflicting_separate_groups(self):
        access = {
            0: SystemAccess(writes={_Position}),
            1: SystemAccess(reads={_Position}),
        }
        groups = compute_parallel_groups([0, 1], access)
        assert groups == [[0], [1]]


class TestParallelDispatcher:
    def test_executes_all_systems(self):
        results = []
        lock = threading.Lock()

        def run(sid):
            with lock:
                results.append(sid)

        dispatcher = ParallelDispatcher(max_workers=2)
        dispatcher.dispatch([[0, 1], [2]], run)
        assert sorted(results) == [0, 1, 2]

    def test_single_system_groups(self):
        results = []
        dispatcher = ParallelDispatcher()
        dispatcher.dispatch([[0], [1]], lambda sid: results.append(sid))
        assert results == [0, 1]

    def test_exception_propagation(self):
        def run(sid):
            if sid == 1:
                raise ValueError("boom")

        dispatcher = ParallelDispatcher(max_workers=2)
        with pytest.raises(ValueError, match="boom"):
            dispatcher.dispatch([[0, 1]], run)
