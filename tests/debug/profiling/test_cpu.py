"""Tests for CPU profiler."""

import threading
import time

import pytest

from engine.debug.profiling.cpu import (
    CPUProfiler,
    FlatProfileEntry,
    ProfileSample,
    get_default_profiler,
    profile,
    profile_scope,
    set_default_profiler,
)


class TestProfileSample:
    """Tests for ProfileSample dataclass."""

    def test_duration_calculation(self) -> None:
        """Test duration is calculated correctly."""
        sample = ProfileSample(
            name="test",
            start_ns=1_000_000_000,
            end_ns=1_016_666_666
        )
        assert sample.duration_ns == 16_666_666
        assert abs(sample.duration_ms - 16.666666) < 0.001

    def test_duration_incomplete_sample(self) -> None:
        """Test duration for incomplete sample returns 0."""
        sample = ProfileSample(name="test", start_ns=1_000_000_000)
        assert sample.duration_ns == 0
        assert sample.duration_ms == 0.0

    def test_self_time_calculation(self) -> None:
        """Test self time excludes children."""
        parent = ProfileSample(
            name="parent",
            start_ns=0,
            end_ns=100_000_000
        )
        child = ProfileSample(
            name="child",
            start_ns=10_000_000,
            end_ns=60_000_000,
            parent=parent
        )
        parent.children.append(child)

        # Parent: 100ms total, child: 50ms
        # Self time should be 100 - 50 = 50ms
        assert parent.duration_ms == 100.0
        assert child.duration_ms == 50.0
        assert parent.self_time_ms == 50.0


class TestCPUProfiler:
    """Tests for CPUProfiler class."""

    def test_begin_end_profiling(self) -> None:
        """Test basic begin/end profiling."""
        profiler = CPUProfiler()

        sample = profiler.begin("test_scope")
        assert sample is not None
        assert sample.name == "test_scope"
        assert sample.start_ns > 0

        time.sleep(0.001)  # 1ms delay

        ended = profiler.end()
        assert ended is sample
        assert sample.end_ns > sample.start_ns
        assert sample.duration_ms >= 1.0

    def test_scoped_profiling(self) -> None:
        """Test context manager scoped profiling."""
        profiler = CPUProfiler()

        with profiler.scope("scoped_test") as sample:
            assert sample is not None
            assert sample.name == "scoped_test"
            time.sleep(0.001)

        assert sample.end_ns > 0
        assert sample.duration_ms >= 1.0

    def test_nested_scopes(self) -> None:
        """Test nested profiling scopes create hierarchy."""
        profiler = CPUProfiler()

        with profiler.scope("outer") as outer:
            time.sleep(0.001)
            with profiler.scope("inner") as inner:
                time.sleep(0.001)

        assert outer is not None
        assert inner is not None
        assert inner.parent is outer
        assert inner in outer.children
        assert outer.duration_ms >= inner.duration_ms

    def test_get_hierarchy(self) -> None:
        """Test getting hierarchical timing tree."""
        profiler = CPUProfiler()

        with profiler.scope("root"):
            with profiler.scope("child1"):
                time.sleep(0.001)
            with profiler.scope("child2"):
                time.sleep(0.001)

        hierarchy = profiler.get_hierarchy()
        assert len(hierarchy) == 1

        root = hierarchy[0]
        assert root.name == "root"
        assert len(root.children) == 2

        child_names = [c.name for c in root.children]
        assert "child1" in child_names
        assert "child2" in child_names

    def test_get_flat(self) -> None:
        """Test getting flat aggregated view."""
        profiler = CPUProfiler()

        # Create multiple samples of same name
        for _ in range(3):
            with profiler.scope("repeated"):
                time.sleep(0.001)

        flat = profiler.get_flat()
        assert len(flat) == 1

        entry = flat[0]
        assert entry.name == "repeated"
        assert entry.call_count == 3
        assert entry.total_time_ms >= 3.0

    def test_flat_sorted_by_total_time(self) -> None:
        """Test flat view is sorted by total time descending."""
        profiler = CPUProfiler()

        with profiler.scope("short"):
            time.sleep(0.001)

        with profiler.scope("long"):
            time.sleep(0.005)

        flat = profiler.get_flat()
        assert len(flat) == 2
        assert flat[0].name == "long"
        assert flat[1].name == "short"

    def test_reset(self) -> None:
        """Test reset clears all samples."""
        profiler = CPUProfiler()

        with profiler.scope("test"):
            pass

        assert len(profiler.get_hierarchy()) == 1

        profiler.reset()

        assert len(profiler.get_hierarchy()) == 0
        assert len(profiler.get_flat()) == 0

    def test_disabled_profiler(self) -> None:
        """Test disabled profiler returns None."""
        profiler = CPUProfiler(enabled=False)

        assert profiler.begin("test") is None
        assert profiler.end() is None

        with profiler.scope("test") as sample:
            assert sample is None

    def test_enable_disable(self) -> None:
        """Test enabling and disabling profiler."""
        profiler = CPUProfiler()

        profiler.enabled = False
        with profiler.scope("disabled") as sample:
            assert sample is None

        profiler.enabled = True
        with profiler.scope("enabled") as sample:
            assert sample is not None

    def test_current_depth(self) -> None:
        """Test getting current nesting depth."""
        profiler = CPUProfiler()

        assert profiler.get_current_depth() == 0

        with profiler.scope("level1"):
            assert profiler.get_current_depth() == 1

            with profiler.scope("level2"):
                assert profiler.get_current_depth() == 2

            assert profiler.get_current_depth() == 1

        assert profiler.get_current_depth() == 0

    def test_format_hierarchy(self) -> None:
        """Test formatting hierarchy as string."""
        profiler = CPUProfiler()

        with profiler.scope("root"):
            with profiler.scope("child"):
                time.sleep(0.001)

        output = profiler.format_hierarchy()
        assert "root:" in output
        assert "child:" in output

    def test_thread_safety(self) -> None:
        """Test profiler works correctly with multiple threads."""
        profiler = CPUProfiler()
        results = []

        def profile_in_thread(name: str) -> None:
            with profiler.scope(name):
                time.sleep(0.001)
            results.append(name)

        threads = [
            threading.Thread(target=profile_in_thread, args=(f"thread_{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        hierarchy = profiler.get_hierarchy()
        # Each thread creates its own root
        assert len(hierarchy) == 5


class TestProfileDecorator:
    """Tests for @profile decorator."""

    def test_profile_decorator(self) -> None:
        """Test basic decorator usage."""
        profiler = CPUProfiler()
        set_default_profiler(profiler)

        @profile()
        def test_func() -> int:
            time.sleep(0.001)
            return 42

        result = test_func()
        assert result == 42

        flat = profiler.get_flat()
        assert any(e.name == "test_func" for e in flat)

    def test_profile_decorator_custom_name(self) -> None:
        """Test decorator with custom name."""
        profiler = CPUProfiler()
        set_default_profiler(profiler)

        @profile(name="custom_name")
        def test_func() -> None:
            pass

        test_func()

        flat = profiler.get_flat()
        assert any(e.name == "custom_name" for e in flat)

    def test_profile_decorator_warning_threshold(self) -> None:
        """Test decorator warning when threshold exceeded."""
        profiler = CPUProfiler()
        set_default_profiler(profiler)

        @profile(name="slow_func", warn_ms=0.1)
        def slow_func() -> None:
            time.sleep(0.002)  # 2ms, exceeds 0.1ms threshold

        # Should log warning but not raise
        slow_func()


class TestProfileScope:
    """Tests for profile_scope convenience function."""

    def test_profile_scope_function(self) -> None:
        """Test profile_scope convenience function."""
        profiler = CPUProfiler()
        set_default_profiler(profiler)

        with profile_scope("convenience_test"):
            time.sleep(0.001)

        flat = profiler.get_flat()
        assert any(e.name == "convenience_test" for e in flat)


class TestFlatProfileEntry:
    """Tests for FlatProfileEntry dataclass."""

    def test_property_calculations(self) -> None:
        """Test property calculations."""
        entry = FlatProfileEntry(
            name="test",
            total_time_ns=100_000_000,  # 100ms
            self_time_ns=50_000_000,    # 50ms
            call_count=10,
            min_time_ns=5_000_000,      # 5ms
            max_time_ns=20_000_000      # 20ms
        )

        assert entry.total_time_ms == 100.0
        assert entry.self_time_ms == 50.0
        assert entry.avg_time_ms == 10.0
        assert entry.min_time_ms == 5.0
        assert entry.max_time_ms == 20.0

    def test_avg_time_zero_calls(self) -> None:
        """Test avg time with zero calls."""
        entry = FlatProfileEntry(
            name="test",
            call_count=0
        )
        assert entry.avg_time_ms == 0.0
