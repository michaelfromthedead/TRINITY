"""Tests for the memory profiler module."""

from __future__ import annotations

import time

import pytest

from engine.tooling.profiling.memory_profiler import (
    MemoryProfiler,
    MemoryProfilerState,
    AllocationRecord,
    MemorySnapshot,
    SnapshotDiff,
    MemoryCategory,
    LeakReport,
    FragmentationStats,
    MemoryBudget,
)


class TestAllocationRecord:
    """Tests for AllocationRecord."""

    def test_creation(self):
        """Test basic creation."""
        record = AllocationRecord(
            address=0x1000,
            size=1024,
            category=MemoryCategory.GAMEPLAY,
            timestamp=time.time(),
        )
        assert record.address == 0x1000
        assert record.size == 1024
        assert record.category == MemoryCategory.GAMEPLAY

    def test_size_conversions(self):
        """Test size conversion properties."""
        record = AllocationRecord(
            address=0x1000,
            size=1024 * 1024,  # 1 MB
            category=MemoryCategory.ASSETS,
            timestamp=time.time(),
        )
        assert record.size_kb == pytest.approx(1024.0, rel=1e-3)
        assert record.size_mb == pytest.approx(1.0, rel=1e-3)

    def test_lifetime_calculation(self):
        """Test lifetime calculation."""
        past = time.time() - 10.0
        record = AllocationRecord(
            address=0x1000,
            size=1024,
            category=MemoryCategory.UNKNOWN,
            timestamp=past,
        )
        assert record.lifetime_seconds >= 10.0

    def test_freed_lifetime(self):
        """Test lifetime calculation for freed allocation."""
        start = time.time() - 10.0
        end = time.time() - 5.0
        record = AllocationRecord(
            address=0x1000,
            size=1024,
            category=MemoryCategory.UNKNOWN,
            timestamp=start,
            freed=True,
            free_timestamp=end,
        )
        assert record.lifetime_seconds == pytest.approx(5.0, rel=1e-1)

    def test_to_dict(self):
        """Test dictionary conversion."""
        record = AllocationRecord(
            address=0x1000,
            size=1024,
            category=MemoryCategory.RENDERING,
            timestamp=time.time(),
            tag="texture_data",
        )
        data = record.to_dict()

        assert data["address"] == 0x1000
        assert data["size"] == 1024
        assert data["category"] == "rendering"
        assert data["tag"] == "texture_data"


class TestMemorySnapshot:
    """Tests for MemorySnapshot."""

    def test_creation(self):
        """Test basic creation."""
        snapshot = MemorySnapshot(
            snapshot_id=1,
            timestamp=time.time(),
            frame_number=100,
        )
        assert snapshot.snapshot_id == 1
        assert snapshot.frame_number == 100

    def test_mb_conversions(self):
        """Test MB conversion properties."""
        snapshot = MemorySnapshot(
            snapshot_id=1,
            timestamp=time.time(),
            frame_number=0,
            current_usage=1024 * 1024 * 100,  # 100 MB
            peak_usage=1024 * 1024 * 150,      # 150 MB
        )
        assert snapshot.current_usage_mb == pytest.approx(100.0, rel=1e-3)
        assert snapshot.peak_usage_mb == pytest.approx(150.0, rel=1e-3)

    def test_diff(self):
        """Test snapshot diffing."""
        snapshot1 = MemorySnapshot(
            snapshot_id=1,
            timestamp=time.time(),
            frame_number=0,
            current_usage=1000,
            allocation_count=10,
        )
        snapshot2 = MemorySnapshot(
            snapshot_id=2,
            timestamp=time.time(),
            frame_number=1,
            current_usage=1500,
            allocation_count=15,
        )

        diff = snapshot1.diff(snapshot2)

        assert diff.memory_delta == 500
        assert diff.allocation_delta == 5

    def test_to_dict(self):
        """Test dictionary conversion."""
        snapshot = MemorySnapshot(
            snapshot_id=1,
            timestamp=time.time(),
            frame_number=50,
            current_usage=1024 * 1024,
        )
        data = snapshot.to_dict()

        assert data["snapshot_id"] == 1
        assert data["frame_number"] == 50
        assert "current_usage_mb" in data


class TestSnapshotDiff:
    """Tests for SnapshotDiff."""

    def test_creation(self):
        """Test basic creation."""
        diff = SnapshotDiff(
            from_snapshot_id=1,
            to_snapshot_id=2,
            memory_delta=1024,
            allocation_delta=5,
        )
        assert diff.memory_delta == 1024
        assert diff.allocation_delta == 5

    def test_mb_conversion(self):
        """Test MB conversion."""
        diff = SnapshotDiff(
            from_snapshot_id=1,
            to_snapshot_id=2,
            memory_delta=1024 * 1024,  # 1 MB
            allocation_delta=0,
        )
        assert diff.memory_delta_mb == pytest.approx(1.0, rel=1e-3)

    def test_from_snapshots(self):
        """Test creation from snapshots."""
        alloc1 = AllocationRecord(
            address=1, size=100, category=MemoryCategory.GAMEPLAY,
            timestamp=time.time()
        )
        alloc2 = AllocationRecord(
            address=2, size=200, category=MemoryCategory.RENDERING,
            timestamp=time.time()
        )

        snapshot1 = MemorySnapshot(
            snapshot_id=1,
            timestamp=time.time(),
            frame_number=0,
            current_usage=100,
            allocation_count=1,
            allocations=[alloc1],
            category_breakdown={MemoryCategory.GAMEPLAY: 100},
        )
        snapshot2 = MemorySnapshot(
            snapshot_id=2,
            timestamp=time.time(),
            frame_number=1,
            current_usage=300,
            allocation_count=2,
            allocations=[alloc1, alloc2],
            category_breakdown={
                MemoryCategory.GAMEPLAY: 100,
                MemoryCategory.RENDERING: 200,
            },
        )

        diff = SnapshotDiff.from_snapshots(snapshot1, snapshot2)

        assert diff.memory_delta == 200
        assert diff.allocation_delta == 1


class TestLeakReport:
    """Tests for LeakReport."""

    def test_creation(self):
        """Test basic creation."""
        report = LeakReport(
            timestamp=time.time(),
            frame_number=1000,
            total_leaked_bytes=1024 * 1024,
            leak_count=5,
        )
        assert report.leak_count == 5
        assert report.total_leaked_mb == pytest.approx(1.0, rel=1e-3)

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = LeakReport(
            timestamp=time.time(),
            frame_number=100,
            total_leaked_bytes=2048,
            leak_count=3,
        )
        data = report.to_dict()

        assert data["leak_count"] == 3
        assert "total_leaked_mb" in data


class TestFragmentationStats:
    """Tests for FragmentationStats."""

    def test_creation(self):
        """Test basic creation."""
        stats = FragmentationStats(
            total_blocks=100,
            free_blocks=20,
            largest_free_block=1024 * 1024,
        )
        assert stats.total_blocks == 100
        assert stats.largest_free_block_mb == pytest.approx(1.0, rel=1e-3)

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = FragmentationStats(
            fragmentation_ratio=0.25,
            average_allocation_size=512,
        )
        data = stats.to_dict()

        assert data["fragmentation_ratio"] == 0.25
        assert data["average_allocation_size"] == 512


class TestMemoryBudget:
    """Tests for MemoryBudget."""

    def test_creation(self):
        """Test basic creation."""
        budget = MemoryBudget(
            category=MemoryCategory.RENDERING,
            max_bytes=100 * 1024 * 1024,
        )
        assert budget.category == MemoryCategory.RENDERING
        assert budget.max_bytes == 100 * 1024 * 1024

    def test_usage_percentage(self):
        """Test usage percentage calculation."""
        budget = MemoryBudget(
            category=MemoryCategory.ASSETS,
            max_bytes=1000,
            current_usage=250,
        )
        assert budget.usage_percentage == pytest.approx(25.0, rel=1e-3)

    def test_over_budget(self):
        """Test over budget detection."""
        budget = MemoryBudget(
            category=MemoryCategory.AUDIO,
            max_bytes=1000,
            current_usage=1500,
        )
        assert budget.is_over_budget

    def test_warning_threshold(self):
        """Test warning threshold detection."""
        budget = MemoryBudget(
            category=MemoryCategory.UI,
            max_bytes=1000,
            warn_at_percentage=80.0,
            current_usage=850,
        )
        assert budget.is_warning
        assert not budget.is_over_budget


class TestMemoryProfiler:
    """Tests for MemoryProfiler."""

    @pytest.fixture
    def profiler(self):
        """Create a fresh profiler instance."""
        return MemoryProfiler(track_stack_traces=False)

    def test_initial_state(self, profiler):
        """Test initial profiler state."""
        assert profiler.state == MemoryProfilerState.DISABLED
        assert not profiler.is_enabled

    def test_enable_disable(self, profiler):
        """Test enable/disable operations."""
        profiler.enable()
        assert profiler.is_enabled

        profiler.disable()
        assert not profiler.is_enabled

    def test_pause_resume(self, profiler):
        """Test pause/resume operations."""
        profiler.enable()
        profiler.pause()
        assert profiler.state == MemoryProfilerState.PAUSED

        profiler.resume()
        assert profiler.state == MemoryProfilerState.ENABLED

    def test_record_allocation(self, profiler):
        """Test allocation recording."""
        profiler.enable()

        addr = profiler.record_allocation(
            size=1024,
            category=MemoryCategory.GAMEPLAY,
            tag="test_alloc",
        )

        assert addr > 0
        assert profiler.current_usage == 1024

    def test_record_free(self, profiler):
        """Test free recording."""
        profiler.enable()

        addr = profiler.record_allocation(1024)
        assert profiler.current_usage == 1024

        profiler.record_free(addr)
        assert profiler.current_usage == 0

    def test_allocation_disabled(self, profiler):
        """Test allocation not recorded when disabled."""
        addr = profiler.record_allocation(1024)
        assert addr == 0
        assert profiler.current_usage == 0

    def test_peak_usage(self, profiler):
        """Test peak usage tracking."""
        profiler.enable()

        addr1 = profiler.record_allocation(1000)
        addr2 = profiler.record_allocation(500)

        assert profiler.peak_usage == 1500

        profiler.record_free(addr1)

        assert profiler.current_usage == 500
        assert profiler.peak_usage == 1500

    def test_scope_allocation(self, profiler):
        """Test scope-based allocation tracking."""
        profiler.enable()

        with profiler.scope(1024, MemoryCategory.RENDERING, "temp") as addr:
            assert profiler.current_usage == 1024

        assert profiler.current_usage == 0

    def test_take_snapshot(self, profiler):
        """Test snapshot creation."""
        profiler.enable()

        profiler.record_allocation(1000, MemoryCategory.GAMEPLAY)
        profiler.record_allocation(500, MemoryCategory.RENDERING)

        snapshot = profiler.take_snapshot()

        assert snapshot.snapshot_id == 1
        assert snapshot.current_usage == 1500
        assert MemoryCategory.GAMEPLAY in snapshot.category_breakdown

    def test_get_snapshot(self, profiler):
        """Test snapshot retrieval."""
        profiler.enable()

        snapshot1 = profiler.take_snapshot()
        snapshot2 = profiler.take_snapshot()

        retrieved = profiler.get_snapshot(1)
        assert retrieved is not None
        assert retrieved.snapshot_id == 1

    def test_diff_snapshots(self, profiler):
        """Test snapshot diffing."""
        profiler.enable()

        profiler.record_allocation(1000)
        snapshot1 = profiler.take_snapshot()

        profiler.record_allocation(500)
        snapshot2 = profiler.take_snapshot()

        diff = profiler.diff_snapshots(1, 2)

        assert diff is not None
        assert diff.memory_delta == 500

    def test_detect_leaks(self, profiler):
        """Test leak detection."""
        profiler.enable()
        profiler.set_leak_threshold(0.0)  # Detect immediately

        profiler.record_allocation(1024, MemoryCategory.GAMEPLAY, "leaked")

        report = profiler.detect_leaks(threshold_seconds=0.0)

        assert report.leak_count == 1
        assert report.total_leaked_bytes == 1024

    def test_get_fragmentation_stats(self, profiler):
        """Test fragmentation stats calculation."""
        profiler.enable()

        for i in range(10):
            profiler.record_allocation(100 * (i + 1))

        stats = profiler.get_fragmentation_stats()

        assert stats.total_blocks == 10
        assert stats.average_allocation_size > 0

    def test_category_breakdown(self, profiler):
        """Test category breakdown tracking."""
        profiler.enable()

        profiler.record_allocation(1000, MemoryCategory.RENDERING)
        profiler.record_allocation(500, MemoryCategory.GAMEPLAY)
        profiler.record_allocation(200, MemoryCategory.RENDERING)

        breakdown = profiler.get_category_breakdown()

        assert breakdown[MemoryCategory.RENDERING] == 1200
        assert breakdown[MemoryCategory.GAMEPLAY] == 500

    def test_get_allocations_filtered(self, profiler):
        """Test filtered allocation retrieval."""
        profiler.enable()

        profiler.record_allocation(100, MemoryCategory.GAMEPLAY, "small")
        profiler.record_allocation(1000, MemoryCategory.RENDERING, "large")
        profiler.record_allocation(500, MemoryCategory.GAMEPLAY, "medium")

        gameplay_allocs = profiler.get_allocations(category=MemoryCategory.GAMEPLAY)
        assert len(gameplay_allocs) == 2

        large_allocs = profiler.get_allocations(min_size=500)
        assert len(large_allocs) == 2

        tagged = profiler.get_allocations(tag="large")
        assert len(tagged) == 1

    def test_get_top_allocations(self, profiler):
        """Test top allocations retrieval."""
        profiler.enable()

        profiler.record_allocation(100)
        profiler.record_allocation(1000)
        profiler.record_allocation(500)

        top = profiler.get_top_allocations(top_n=2, sort_by="size")

        assert len(top) == 2
        assert top[0].size == 1000
        assert top[1].size == 500

    def test_memory_budget(self, profiler):
        """Test memory budget system."""
        profiler.enable()

        profiler.set_budget(MemoryCategory.RENDERING, max_bytes=1000, warn_at_percentage=80.0)

        profiler.record_allocation(500, MemoryCategory.RENDERING)

        status = profiler.get_budget_status()

        assert MemoryCategory.RENDERING in status
        assert status[MemoryCategory.RENDERING]["usage_percentage"] == pytest.approx(50.0, rel=1e-3)

    def test_clear(self, profiler):
        """Test clearing profiler data."""
        profiler.enable()

        profiler.record_allocation(1024)
        profiler.take_snapshot()

        profiler.clear()

        assert profiler.current_usage == 0
        assert profiler.peak_usage == 0
        assert len(profiler.get_allocations()) == 0

    def test_listener_callback(self, profiler):
        """Test allocation listener callbacks."""
        profiler.enable()
        allocations_received = []

        def on_alloc(record):
            allocations_received.append(record)

        profiler.add_listener(on_alloc)

        profiler.record_allocation(1024)

        assert len(allocations_received) == 1

        profiler.remove_listener(on_alloc)

    def test_to_dict(self, profiler):
        """Test dictionary export."""
        profiler.enable()

        profiler.record_allocation(1024, MemoryCategory.GAMEPLAY)

        data = profiler.to_dict()

        assert "state" in data
        assert "current_usage" in data
        assert data["active_allocations"] == 1

    def test_frame_lifecycle(self, profiler):
        """Test frame begin/end."""
        profiler.enable()

        profiler.begin_frame()
        profiler.record_allocation(512)
        profiler.end_frame()

        # Verify allocation was recorded
        assert profiler.current_usage == 512
