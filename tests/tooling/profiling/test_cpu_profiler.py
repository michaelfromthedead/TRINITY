"""Tests for the CPU profiler module."""

from __future__ import annotations

import threading
import time

import pytest

from engine.tooling.profiling.cpu_profiler import (
    CPUProfiler,
    CPUProfileSample,
    CallTreeNode,
    FlameGraphData,
    HotPath,
    ProfilerState,
    ProfilerStats,
)


class TestCPUProfileSample:
    """Tests for CPUProfileSample."""

    def test_sample_creation(self):
        """Test basic sample creation."""
        sample = CPUProfileSample(
            name="test",
            start_time=1.0,
            end_time=1.001,
            thread_id=1,
            depth=0,
        )
        assert sample.name == "test"
        assert sample.thread_id == 1
        assert sample.depth == 0

    def test_duration_calculations(self):
        """Test duration property calculations."""
        sample = CPUProfileSample(
            name="test",
            start_time=1.0,
            end_time=1.001,  # 1ms duration
            thread_id=1,
            depth=0,
        )
        assert sample.duration_ms == pytest.approx(1.0, rel=1e-3)
        assert sample.duration_us == pytest.approx(1000.0, rel=1e-3)
        assert sample.duration_ns == pytest.approx(1_000_000.0, rel=1e-3)

    def test_sample_with_tags(self):
        """Test sample with custom tags."""
        sample = CPUProfileSample(
            name="test",
            start_time=1.0,
            end_time=1.001,
            thread_id=1,
            depth=0,
            tags={"custom": "value"},
        )
        assert sample.tags["custom"] == "value"

    def test_sample_with_parent(self):
        """Test sample with parent relationship."""
        sample = CPUProfileSample(
            name="child",
            start_time=1.0,
            end_time=1.001,
            thread_id=1,
            depth=1,
            parent_id=0,
            sample_id=1,
        )
        assert sample.parent_id == 0
        assert sample.sample_id == 1


class TestCallTreeNode:
    """Tests for CallTreeNode."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = CallTreeNode(name="test")
        assert node.name == "test"
        assert node.call_count == 0
        assert node.inclusive_time_ms == 0.0
        assert node.exclusive_time_ms == 0.0

    def test_add_sample(self):
        """Test adding samples to a node."""
        node = CallTreeNode(name="test")
        sample = CPUProfileSample(
            name="test",
            start_time=0.0,
            end_time=0.001,
            thread_id=1,
            depth=0,
        )
        node.add_sample(sample)

        assert node.call_count == 1
        assert node.inclusive_time_ms == pytest.approx(1.0, rel=1e-3)

    def test_avg_time_calculation(self):
        """Test average time calculation."""
        node = CallTreeNode(name="test")
        for i in range(5):
            sample = CPUProfileSample(
                name="test",
                start_time=i,
                end_time=i + 0.002,
                thread_id=1,
                depth=0,
            )
            node.add_sample(sample)

        assert node.call_count == 5
        assert node.avg_time_ms == pytest.approx(2.0, rel=1e-3)

    def test_exclusive_time_calculation(self):
        """Test exclusive time calculation."""
        parent = CallTreeNode(name="parent", inclusive_time_ms=10.0)
        child = CallTreeNode(name="child", inclusive_time_ms=3.0)
        parent.children["child"] = child

        parent.calculate_exclusive_time()

        assert parent.exclusive_time_ms == pytest.approx(7.0, rel=1e-3)
        assert child.exclusive_time_ms == pytest.approx(3.0, rel=1e-3)

    def test_to_dict(self):
        """Test dictionary conversion."""
        node = CallTreeNode(name="test", inclusive_time_ms=5.0, call_count=3)
        data = node.to_dict()

        assert data["name"] == "test"
        assert data["inclusive_time_ms"] == 5.0
        assert data["call_count"] == 3


class TestFlameGraphData:
    """Tests for FlameGraphData."""

    def test_creation(self):
        """Test flame graph data creation."""
        data = FlameGraphData(name="root", value=10.0)
        assert data.name == "root"
        assert data.value == 10.0

    def test_with_children(self):
        """Test flame graph with children."""
        child1 = FlameGraphData(name="child1", value=3.0)
        child2 = FlameGraphData(name="child2", value=5.0)
        parent = FlameGraphData(name="parent", value=2.0, children=[child1, child2])

        assert len(parent.children) == 2
        assert parent.children[0].name == "child1"

    def test_from_call_tree(self):
        """Test creation from call tree."""
        root = CallTreeNode(name="root", exclusive_time_ms=5.0)
        child = CallTreeNode(name="child", exclusive_time_ms=3.0)
        root.children["child"] = child

        flame = FlameGraphData.from_call_tree(root)
        assert flame.name == "root"
        assert flame.value == 5.0
        assert len(flame.children) == 1

    def test_to_dict(self):
        """Test dictionary conversion."""
        data = FlameGraphData(name="test", value=10.0)
        result = data.to_dict()

        assert result["name"] == "test"
        assert result["value"] == 10.0
        assert "children" in result


class TestHotPath:
    """Tests for HotPath."""

    def test_creation(self):
        """Test hot path creation."""
        path = HotPath(
            path=["main", "update", "physics"],
            total_time_ms=15.0,
            call_count=100,
            percentage=25.0,
        )
        assert len(path.path) == 3
        assert path.total_time_ms == 15.0
        assert path.percentage == 25.0

    def test_string_representation(self):
        """Test string representation."""
        path = HotPath(
            path=["main", "update"],
            total_time_ms=10.0,
            call_count=50,
            percentage=20.0,
        )
        s = str(path)
        assert "main" in s
        assert "update" in s
        assert "10.00ms" in s
        assert "20.0%" in s


class TestProfilerStats:
    """Tests for ProfilerStats."""

    def test_creation(self):
        """Test stats creation."""
        stats = ProfilerStats(name="test")
        assert stats.name == "test"
        assert stats.call_count == 0

    def test_update(self):
        """Test stats update."""
        stats = ProfilerStats(name="test")
        stats.update(5.0)
        stats.update(3.0)
        stats.update(7.0)

        assert stats.call_count == 3
        assert stats.total_time_ms == 15.0
        assert stats.min_time_ms == 3.0
        assert stats.max_time_ms == 7.0
        assert stats.avg_time_ms == pytest.approx(5.0, rel=1e-3)

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = ProfilerStats(name="test")
        stats.update(5.0)
        data = stats.to_dict()

        assert data["name"] == "test"
        assert data["call_count"] == 1
        assert data["total_time_ms"] == 5.0


class TestCPUProfiler:
    """Tests for CPUProfiler."""

    @pytest.fixture
    def profiler(self):
        """Create a fresh profiler instance."""
        return CPUProfiler()

    def test_initial_state(self, profiler):
        """Test initial profiler state."""
        assert profiler.state == ProfilerState.DISABLED
        assert not profiler.is_enabled

    def test_enable_disable(self, profiler):
        """Test enable/disable operations."""
        profiler.enable()
        assert profiler.is_enabled
        assert profiler.state == ProfilerState.ENABLED

        profiler.disable()
        assert not profiler.is_enabled
        assert profiler.state == ProfilerState.DISABLED

    def test_pause_resume(self, profiler):
        """Test pause/resume operations."""
        profiler.enable()
        profiler.pause()
        assert profiler.state == ProfilerState.PAUSED

        profiler.resume()
        assert profiler.state == ProfilerState.ENABLED

    def test_scope_profiling_disabled(self, profiler):
        """Test that scope doesn't collect when disabled."""
        with profiler.scope("test"):
            pass

        assert len(profiler.get_samples()) == 0

    def test_scope_profiling_enabled(self, profiler):
        """Test scope profiling when enabled."""
        profiler.enable()

        with profiler.scope("test"):
            time.sleep(0.001)

        samples = profiler.get_samples()
        assert len(samples) == 1
        assert samples[0].name == "test"
        assert samples[0].duration_ms > 0

    def test_nested_scopes(self, profiler):
        """Test nested profiling scopes."""
        profiler.enable()

        with profiler.scope("outer"):
            with profiler.scope("inner"):
                pass

        samples = profiler.get_samples()
        assert len(samples) == 2

        inner_samples = [s for s in samples if s.name == "inner"]
        outer_samples = [s for s in samples if s.name == "outer"]

        assert len(inner_samples) == 1
        assert len(outer_samples) == 1
        assert inner_samples[0].depth == 1
        assert outer_samples[0].depth == 0

    def test_get_stats(self, profiler):
        """Test statistics collection."""
        profiler.enable()

        for _ in range(5):
            with profiler.scope("test"):
                pass

        stats = profiler.get_stats("test")
        assert "test" in stats
        assert stats["test"].call_count == 5

    def test_get_stats_filtered(self, profiler):
        """Test filtered statistics."""
        profiler.enable()

        with profiler.scope("test1"):
            pass
        with profiler.scope("test2"):
            pass

        stats = profiler.get_stats("test1")
        assert "test1" in stats
        assert "test2" not in stats

    def test_clear(self, profiler):
        """Test clearing profiler data."""
        profiler.enable()

        with profiler.scope("test"):
            pass

        profiler.clear()

        assert len(profiler.get_samples()) == 0
        assert len(profiler.get_stats()) == 0

    def test_warn_threshold(self, profiler):
        """Test warning threshold setting."""
        profiler.enable()
        profiler.set_warn_threshold("slow_op", 1.0)

        with profiler.scope("slow_op"):
            time.sleep(0.002)  # 2ms > 1ms threshold

        profiler.remove_warn_threshold("slow_op")
        # Threshold should be removed without error

    def test_frame_timing(self, profiler):
        """Test frame begin/end."""
        profiler.enable()

        profiler.begin_frame()
        time.sleep(0.001)
        frame_time = profiler.end_frame()

        assert frame_time > 0

    def test_build_call_tree(self, profiler):
        """Test call tree building."""
        profiler.enable()

        with profiler.scope("root"):
            with profiler.scope("child1"):
                pass
            with profiler.scope("child2"):
                pass

        tree = profiler.build_call_tree()

        assert tree.name == "[root]"
        assert "root" in tree.children
        root_node = tree.children["root"]
        assert "child1" in root_node.children
        assert "child2" in root_node.children

    def test_get_flame_graph(self, profiler):
        """Test flame graph generation."""
        profiler.enable()

        with profiler.scope("root"):
            with profiler.scope("child"):
                pass

        flame = profiler.get_flame_graph()

        assert flame.name == "[root]"
        assert len(flame.children) > 0

    def test_get_hot_paths(self, profiler):
        """Test hot path detection."""
        profiler.enable()

        for _ in range(10):
            with profiler.scope("hot"):
                time.sleep(0.001)
            with profiler.scope("cold"):
                pass

        paths = profiler.get_hot_paths(top_n=5)
        assert len(paths) > 0
        assert paths[0].path[-1] == "hot"

    def test_thread_breakdown(self, profiler):
        """Test thread-based breakdown."""
        profiler.enable()

        with profiler.scope("main_thread"):
            pass

        breakdown = profiler.get_thread_breakdown()
        assert len(breakdown) > 0
        assert threading.get_ident() in breakdown

    def test_get_hotspots(self, profiler):
        """Test hotspot detection."""
        profiler.enable()

        with profiler.scope("slow"):
            time.sleep(0.002)
        with profiler.scope("fast"):
            pass

        hotspots = profiler.get_hotspots(top_n=2, sort_by="exclusive")
        assert len(hotspots) > 0

    def test_sample_filtering(self, profiler):
        """Test sample filtering."""
        profiler.enable()

        with profiler.scope("slow"):
            time.sleep(0.002)
        with profiler.scope("fast"):
            pass

        slow_samples = profiler.get_samples(name="slow")
        assert len(slow_samples) == 1
        assert slow_samples[0].name == "slow"

        filtered = profiler.get_samples(min_duration_ms=1.0)
        assert all(s.duration_ms >= 1.0 for s in filtered)

    def test_to_dict(self, profiler):
        """Test dictionary export."""
        profiler.enable()

        with profiler.scope("test"):
            pass

        data = profiler.to_dict()

        assert "state" in data
        assert "sample_count" in data
        assert "stats" in data
        assert data["sample_count"] == 1

    def test_listener_callback(self, profiler):
        """Test sample listener callbacks."""
        profiler.enable()
        samples_received = []

        def on_sample(sample):
            samples_received.append(sample)

        profiler.add_listener(on_sample)

        with profiler.scope("test"):
            pass

        assert len(samples_received) == 1
        assert samples_received[0].name == "test"

        profiler.remove_listener(on_sample)

    def test_max_samples_trimming(self):
        """Test sample trimming at max capacity."""
        profiler = CPUProfiler(max_samples=10)
        profiler.enable()

        for i in range(20):
            with profiler.scope(f"test_{i}"):
                pass

        samples = profiler.get_samples()
        assert len(samples) <= 10

    def test_scope_with_tags(self, profiler):
        """Test scope with custom tags."""
        profiler.enable()

        with profiler.scope("test", category="render", priority=1):
            pass

        samples = profiler.get_samples()
        assert samples[0].tags["category"] == "render"
        assert samples[0].tags["priority"] == 1
