"""Tests for memory profiler."""

import time

import pytest

from engine.debug.profiling.memory import (
    AllocationRecord,
    LeakCandidate,
    MemoryDiff,
    MemoryProfiler,
    MemorySnapshot,
    MemoryTag,
)


class TestMemoryTag:
    """Tests for MemoryTag enum."""

    def test_all_tags_exist(self) -> None:
        """Test all expected tags exist."""
        expected = [
            "RENDERING", "PHYSICS", "AUDIO", "GAMEPLAY",
            "AI", "NETWORK", "UI", "RESOURCES",
            "SCRIPTING", "DEBUG", "SYSTEM", "UNKNOWN"
        ]
        for tag_name in expected:
            assert hasattr(MemoryTag, tag_name)


class TestAllocationRecord:
    """Tests for AllocationRecord dataclass."""

    def test_lifetime_calculation(self) -> None:
        """Test lifetime calculation for active allocation."""
        record = AllocationRecord(
            ptr=1,
            size=1024,
            tag=MemoryTag.RENDERING,
            timestamp=time.time() - 5.0  # 5 seconds ago
        )
        assert record.lifetime_seconds >= 5.0

    def test_lifetime_freed_allocation(self) -> None:
        """Test lifetime calculation for freed allocation."""
        start = time.time() - 10.0
        end = time.time() - 5.0
        record = AllocationRecord(
            ptr=1,
            size=1024,
            tag=MemoryTag.RENDERING,
            timestamp=start,
            freed=True,
            freed_timestamp=end
        )
        assert abs(record.lifetime_seconds - 5.0) < 0.1


class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""

    def test_track_allocation(self) -> None:
        """Test basic allocation tracking."""
        profiler = MemoryProfiler()

        ptr = profiler.track_allocation(1024, MemoryTag.RENDERING)
        assert ptr > 0
        assert profiler.total_allocated == 1024
        assert profiler.allocation_count == 1

    def test_track_multiple_allocations(self) -> None:
        """Test tracking multiple allocations."""
        profiler = MemoryProfiler()

        ptr1 = profiler.track_allocation(1024, MemoryTag.RENDERING)
        ptr2 = profiler.track_allocation(2048, MemoryTag.PHYSICS)

        assert ptr1 != ptr2
        assert profiler.total_allocated == 3072
        assert profiler.allocation_count == 2

    def test_track_free(self) -> None:
        """Test freeing tracked allocations."""
        profiler = MemoryProfiler()

        ptr = profiler.track_allocation(1024, MemoryTag.RENDERING)
        assert profiler.total_allocated == 1024

        result = profiler.track_free(ptr)
        assert result is True
        assert profiler.total_allocated == 0
        assert profiler.allocation_count == 0

    def test_track_free_invalid_ptr(self) -> None:
        """Test freeing invalid pointer returns False."""
        profiler = MemoryProfiler()

        result = profiler.track_free(999)
        assert result is False

    def test_track_free_zero_ptr(self) -> None:
        """Test freeing zero pointer is no-op."""
        profiler = MemoryProfiler()

        result = profiler.track_free(0)
        assert result is False

    def test_get_usage_by_tag(self) -> None:
        """Test getting usage grouped by tag."""
        profiler = MemoryProfiler()

        profiler.track_allocation(1024, MemoryTag.RENDERING)
        profiler.track_allocation(2048, MemoryTag.RENDERING)
        profiler.track_allocation(512, MemoryTag.PHYSICS)

        usage = profiler.get_usage_by_tag()

        assert usage[MemoryTag.RENDERING] == 3072
        assert usage[MemoryTag.PHYSICS] == 512
        assert usage[MemoryTag.AUDIO] == 0

    def test_peak_allocated(self) -> None:
        """Test peak allocation tracking."""
        profiler = MemoryProfiler()

        ptr1 = profiler.track_allocation(1000, MemoryTag.RENDERING)
        ptr2 = profiler.track_allocation(2000, MemoryTag.RENDERING)
        assert profiler.peak_allocated == 3000

        profiler.track_free(ptr1)
        assert profiler.total_allocated == 2000
        assert profiler.peak_allocated == 3000  # Peak unchanged

    def test_disabled_profiler(self) -> None:
        """Test disabled profiler returns 0."""
        profiler = MemoryProfiler(enabled=False)

        ptr = profiler.track_allocation(1024, MemoryTag.RENDERING)
        assert ptr == 0
        assert profiler.total_allocated == 0


class TestMemorySnapshot:
    """Tests for memory snapshots."""

    def test_create_snapshot(self) -> None:
        """Test creating a memory snapshot."""
        profiler = MemoryProfiler()

        profiler.track_allocation(1024, MemoryTag.RENDERING)
        profiler.track_allocation(2048, MemoryTag.PHYSICS)

        snapshot = profiler.snapshot("test_snapshot")

        assert snapshot.name == "test_snapshot"
        assert snapshot.total_allocated == 3072
        assert snapshot.allocation_count == 2
        assert snapshot.usage_by_tag[MemoryTag.RENDERING] == 1024
        assert snapshot.usage_by_tag[MemoryTag.PHYSICS] == 2048

    def test_get_snapshot(self) -> None:
        """Test retrieving a saved snapshot."""
        profiler = MemoryProfiler()

        profiler.track_allocation(1024, MemoryTag.RENDERING)
        profiler.snapshot("my_snapshot")

        retrieved = profiler.get_snapshot("my_snapshot")
        assert retrieved is not None
        assert retrieved.name == "my_snapshot"
        assert retrieved.total_allocated == 1024

    def test_get_nonexistent_snapshot(self) -> None:
        """Test retrieving non-existent snapshot returns None."""
        profiler = MemoryProfiler()

        assert profiler.get_snapshot("does_not_exist") is None


class TestMemoryDiff:
    """Tests for memory snapshot comparison."""

    def test_diff_snapshots(self) -> None:
        """Test comparing two snapshots."""
        profiler = MemoryProfiler()

        profiler.track_allocation(1024, MemoryTag.RENDERING)
        profiler.snapshot("before")

        profiler.track_allocation(2048, MemoryTag.PHYSICS)
        profiler.snapshot("after")

        diff = profiler.diff("before", "after")

        assert diff is not None
        assert diff.snapshot_a == "before"
        assert diff.snapshot_b == "after"
        assert diff.delta_total == 2048
        assert diff.delta_count == 1
        assert diff.is_memory_increase is True
        assert diff.is_memory_decrease is False

    def test_diff_with_freed_allocations(self) -> None:
        """Test diff tracks freed allocations."""
        profiler = MemoryProfiler()

        ptr = profiler.track_allocation(1024, MemoryTag.RENDERING)
        profiler.snapshot("before")

        profiler.track_free(ptr)
        profiler.snapshot("after")

        diff = profiler.diff("before", "after")

        assert diff is not None
        assert diff.delta_total == -1024
        assert len(diff.freed_allocations) == 1
        assert len(diff.new_allocations) == 0

    def test_diff_nonexistent_snapshot(self) -> None:
        """Test diff with non-existent snapshot returns None."""
        profiler = MemoryProfiler()

        profiler.snapshot("exists")
        diff = profiler.diff("exists", "does_not_exist")

        assert diff is None


class TestLeakDetection:
    """Tests for memory leak detection."""

    def test_detect_leaks_no_leaks(self) -> None:
        """Test no leaks detected for fresh allocations."""
        profiler = MemoryProfiler()

        profiler.track_allocation(1024, MemoryTag.RENDERING)

        # With default 60s threshold, fresh allocation is not a leak
        leaks = profiler.detect_leaks(min_age_seconds=60.0)
        assert len(leaks) == 0

    def test_detect_leaks_with_old_allocation(self) -> None:
        """Test detecting old allocations as potential leaks."""
        profiler = MemoryProfiler()

        # Manually create an old allocation
        ptr = profiler.track_allocation(1024, MemoryTag.RENDERING)
        record = profiler.get_allocation(ptr)
        assert record is not None
        record.timestamp = time.time() - 100  # 100 seconds old

        leaks = profiler.detect_leaks(min_age_seconds=60.0)
        assert len(leaks) == 1
        assert leaks[0].allocation.ptr == ptr
        assert leaks[0].age_seconds >= 100.0

    def test_detect_leaks_ignore_tags(self) -> None:
        """Test ignoring specific tags in leak detection."""
        profiler = MemoryProfiler()

        ptr = profiler.track_allocation(1024, MemoryTag.SYSTEM)
        record = profiler.get_allocation(ptr)
        assert record is not None
        record.timestamp = time.time() - 100

        # Ignore SYSTEM tag
        leaks = profiler.detect_leaks(
            min_age_seconds=60.0,
            ignore_tags={MemoryTag.SYSTEM}
        )
        assert len(leaks) == 0

    def test_leak_confidence_scoring(self) -> None:
        """Test leak candidates have confidence scores."""
        profiler = MemoryProfiler()

        # Create a large, old allocation (high confidence)
        ptr = profiler.track_allocation(10 * 1024 * 1024, MemoryTag.RENDERING)
        record = profiler.get_allocation(ptr)
        assert record is not None
        record.timestamp = time.time() - 1000

        leaks = profiler.detect_leaks(min_age_seconds=60.0)
        assert len(leaks) == 1
        assert leaks[0].confidence > 0.5  # Should have high confidence


class TestMemoryProfilerHelpers:
    """Tests for helper methods."""

    def test_get_allocation(self) -> None:
        """Test getting a specific allocation."""
        profiler = MemoryProfiler()

        ptr = profiler.track_allocation(1024, MemoryTag.RENDERING)
        record = profiler.get_allocation(ptr)

        assert record is not None
        assert record.ptr == ptr
        assert record.size == 1024
        assert record.tag == MemoryTag.RENDERING

    def test_get_allocations_by_tag(self) -> None:
        """Test filtering allocations by tag."""
        profiler = MemoryProfiler()

        profiler.track_allocation(1024, MemoryTag.RENDERING)
        profiler.track_allocation(2048, MemoryTag.RENDERING)
        profiler.track_allocation(512, MemoryTag.PHYSICS)

        rendering_allocs = profiler.get_allocations_by_tag(MemoryTag.RENDERING)
        assert len(rendering_allocs) == 2
        assert all(a.tag == MemoryTag.RENDERING for a in rendering_allocs)

    def test_get_largest_allocations(self) -> None:
        """Test getting largest allocations."""
        profiler = MemoryProfiler()

        profiler.track_allocation(100, MemoryTag.RENDERING)
        profiler.track_allocation(1000, MemoryTag.PHYSICS)
        profiler.track_allocation(500, MemoryTag.AUDIO)

        largest = profiler.get_largest_allocations(2)
        assert len(largest) == 2
        assert largest[0].size == 1000
        assert largest[1].size == 500

    def test_reset(self) -> None:
        """Test reset clears all data."""
        profiler = MemoryProfiler()

        profiler.track_allocation(1024, MemoryTag.RENDERING)
        profiler.snapshot("test")

        profiler.reset()

        assert profiler.total_allocated == 0
        assert profiler.allocation_count == 0
        assert profiler.peak_allocated == 0
        assert profiler.get_snapshot("test") is None

    def test_format_usage_report(self) -> None:
        """Test formatting usage report."""
        profiler = MemoryProfiler()

        profiler.track_allocation(1024 * 1024, MemoryTag.RENDERING)
        profiler.track_allocation(2 * 1024 * 1024, MemoryTag.PHYSICS)

        report = profiler.format_usage_report()

        assert "Memory Usage Report" in report
        assert "RENDERING" in report
        assert "PHYSICS" in report


class TestStackTraceCapture:
    """Tests for stack trace capture."""

    def test_stack_trace_captured(self) -> None:
        """Test stack traces are captured when enabled."""
        profiler = MemoryProfiler(capture_stack_traces=True)

        ptr = profiler.track_allocation(1024, MemoryTag.RENDERING)
        record = profiler.get_allocation(ptr)

        assert record is not None
        assert record.stack_trace is not None
        assert "test_stack_trace_captured" in record.stack_trace

    def test_stack_trace_not_captured(self) -> None:
        """Test stack traces are not captured when disabled."""
        profiler = MemoryProfiler(capture_stack_traces=False)

        ptr = profiler.track_allocation(1024, MemoryTag.RENDERING)
        record = profiler.get_allocation(ptr)

        assert record is not None
        assert record.stack_trace is None
