"""Tests for the GPU profiler module."""

from __future__ import annotations

import time

import pytest

from engine.tooling.profiling.gpu_profiler import (
    GPUProfiler,
    GPUProfileSample,
    GPUProfilerState,
    DrawCallStats,
    ShaderStats,
    GPUMemoryStats,
    RenderPassTiming,
    RenderPassType,
    GPUFrameStats,
)


class TestDrawCallStats:
    """Tests for DrawCallStats."""

    def test_creation(self):
        """Test basic creation."""
        stats = DrawCallStats()
        assert stats.total_draw_calls == 0
        assert stats.total_triangles == 0

    def test_add(self):
        """Test adding stats together."""
        stats1 = DrawCallStats(total_draw_calls=10, total_triangles=1000)
        stats2 = DrawCallStats(total_draw_calls=5, total_triangles=500)

        stats1.add(stats2)

        assert stats1.total_draw_calls == 15
        assert stats1.total_triangles == 1500

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = DrawCallStats(
            total_draw_calls=10,
            instanced_draw_calls=3,
            indexed_draw_calls=8,
            total_triangles=5000,
        )
        data = stats.to_dict()

        assert data["total_draw_calls"] == 10
        assert data["instanced_draw_calls"] == 3
        assert data["total_triangles"] == 5000


class TestShaderStats:
    """Tests for ShaderStats."""

    def test_creation(self):
        """Test basic creation."""
        stats = ShaderStats(name="test_shader")
        assert stats.name == "test_shader"
        assert stats.invocations == 0

    def test_avg_time(self):
        """Test average time calculation."""
        stats = ShaderStats(name="test", invocations=10, total_time_ms=20.0)
        assert stats.avg_time_ms == pytest.approx(2.0, rel=1e-3)

    def test_avg_time_zero_invocations(self):
        """Test average time with zero invocations."""
        stats = ShaderStats(name="test", invocations=0, total_time_ms=0.0)
        assert stats.avg_time_ms == 0.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = ShaderStats(
            name="pbr_shader",
            invocations=100,
            total_time_ms=5.0,
            vertex_invocations=50000,
            fragment_invocations=200000,
        )
        data = stats.to_dict()

        assert data["name"] == "pbr_shader"
        assert data["invocations"] == 100
        assert data["avg_time_ms"] == pytest.approx(0.05, rel=1e-3)


class TestGPUMemoryStats:
    """Tests for GPUMemoryStats."""

    def test_creation(self):
        """Test basic creation."""
        stats = GPUMemoryStats()
        assert stats.total_vram_bytes == 0
        assert stats.used_vram_bytes == 0

    def test_mb_conversions(self):
        """Test MB conversion properties."""
        stats = GPUMemoryStats(
            total_vram_bytes=8 * 1024 * 1024 * 1024,  # 8 GB
            used_vram_bytes=4 * 1024 * 1024 * 1024,    # 4 GB
        )
        assert stats.total_vram_mb == pytest.approx(8192.0, rel=1e-3)
        assert stats.used_vram_mb == pytest.approx(4096.0, rel=1e-3)

    def test_usage_percentage(self):
        """Test usage percentage calculation."""
        stats = GPUMemoryStats(
            total_vram_bytes=1000,
            used_vram_bytes=250,
        )
        assert stats.usage_percentage == pytest.approx(25.0, rel=1e-3)

    def test_usage_percentage_zero_total(self):
        """Test usage percentage with zero total."""
        stats = GPUMemoryStats(total_vram_bytes=0)
        assert stats.usage_percentage == 0.0

    def test_bandwidth_total(self):
        """Test bandwidth total calculation."""
        stats = GPUMemoryStats(
            bandwidth_read_bytes=1000,
            bandwidth_write_bytes=500,
        )
        assert stats.bandwidth_total_bytes == 1500

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = GPUMemoryStats(
            total_vram_bytes=1024 * 1024,
            used_vram_bytes=512 * 1024,
        )
        data = stats.to_dict()

        assert "total_vram_bytes" in data
        assert "used_vram_mb" in data
        assert "usage_percentage" in data


class TestRenderPassTiming:
    """Tests for RenderPassTiming."""

    def test_creation(self):
        """Test basic creation."""
        timing = RenderPassTiming(
            name="shadow_pass",
            pass_type=RenderPassType.SHADOW,
        )
        assert timing.name == "shadow_pass"
        assert timing.pass_type == RenderPassType.SHADOW
        assert timing.gpu_time_ms == 0.0

    def test_total_time(self):
        """Test total time calculation."""
        timing = RenderPassTiming(
            name="test",
            pass_type=RenderPassType.CUSTOM,
            gpu_time_ms=5.0,
            cpu_time_ms=2.0,
        )
        assert timing.total_time_ms == pytest.approx(7.0, rel=1e-3)

    def test_add_sample(self):
        """Test adding samples."""
        timing = RenderPassTiming(
            name="test",
            pass_type=RenderPassType.LIGHTING,
        )

        sample = GPUProfileSample(
            name="test",
            category="lighting",
            start_time=0.0,
            end_time=0.001,
            gpu_time_ms=1.0,
            depth=0,
            draw_calls=10,
            triangles=5000,
        )

        timing.add_sample(sample)

        assert timing.gpu_time_ms == pytest.approx(1.0, rel=1e-3)
        assert timing.draw_calls == 10
        assert timing.triangles == 5000
        assert len(timing.samples) == 1

    def test_to_dict(self):
        """Test dictionary conversion."""
        timing = RenderPassTiming(
            name="gbuffer",
            pass_type=RenderPassType.GBUFFER,
            gpu_time_ms=3.0,
            draw_calls=50,
        )
        data = timing.to_dict()

        assert data["name"] == "gbuffer"
        assert data["pass_type"] == "GBUFFER"
        assert data["gpu_time_ms"] == 3.0


class TestGPUProfileSample:
    """Tests for GPUProfileSample."""

    def test_creation(self):
        """Test basic creation."""
        sample = GPUProfileSample(
            name="test",
            category="rendering",
            start_time=1.0,
            end_time=1.002,
            gpu_time_ms=1.5,
            depth=0,
        )
        assert sample.name == "test"
        assert sample.category == "rendering"
        assert sample.gpu_time_ms == 1.5

    def test_cpu_time(self):
        """Test CPU time calculation."""
        sample = GPUProfileSample(
            name="test",
            category="test",
            start_time=0.0,
            end_time=0.002,  # 2ms
            gpu_time_ms=1.5,
            depth=0,
        )
        assert sample.cpu_time_ms == pytest.approx(2.0, rel=1e-3)

    def test_with_draw_stats(self):
        """Test sample with draw stats."""
        sample = GPUProfileSample(
            name="test",
            category="test",
            start_time=0.0,
            end_time=0.001,
            gpu_time_ms=1.0,
            depth=0,
            draw_calls=100,
            triangles=50000,
            vertices=150000,
        )
        assert sample.draw_calls == 100
        assert sample.triangles == 50000
        assert sample.vertices == 150000


class TestGPUProfiler:
    """Tests for GPUProfiler."""

    @pytest.fixture
    def profiler(self):
        """Create a fresh profiler instance."""
        return GPUProfiler()

    def test_initial_state(self, profiler):
        """Test initial profiler state."""
        assert profiler.state == GPUProfilerState.DISABLED
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
        assert profiler.state == GPUProfilerState.PAUSED

        profiler.resume()
        assert profiler.state == GPUProfilerState.ENABLED

    def test_scope_disabled(self, profiler):
        """Test scope doesn't collect when disabled."""
        with profiler.scope("test", "rendering"):
            pass

        assert len(profiler.get_samples()) == 0

    def test_scope_enabled(self, profiler):
        """Test scope profiling when enabled."""
        profiler.enable()
        profiler.set_simulated_gpu_time(1.5)

        with profiler.scope("test_pass", "shadows"):
            time.sleep(0.001)

        samples = profiler.get_samples()
        assert len(samples) == 1
        assert samples[0].name == "test_pass"
        assert samples[0].category == "shadows"
        assert samples[0].gpu_time_ms == pytest.approx(1.5, rel=1e-3)

    def test_nested_scopes(self, profiler):
        """Test nested profiling scopes."""
        profiler.enable()
        profiler.set_simulated_gpu_time(1.0)

        with profiler.scope("outer", "rendering"):
            with profiler.scope("inner", "rendering"):
                pass

        samples = profiler.get_samples()
        assert len(samples) == 2

    def test_record_draw_call(self, profiler):
        """Test draw call recording."""
        profiler.enable()

        profiler.record_draw_call(triangles=1000, vertices=3000, instanced=True)
        profiler.record_draw_call(triangles=500, vertices=1500)

        stats = profiler.get_draw_stats()
        assert stats.total_draw_calls == 2
        assert stats.total_triangles == 1500
        assert stats.instanced_draw_calls == 1

    def test_record_state_change(self, profiler):
        """Test state change recording."""
        profiler.enable()

        profiler.record_state_change()
        profiler.record_state_change()
        profiler.record_state_change()

        stats = profiler.get_draw_stats()
        assert stats.state_changes == 3

    def test_record_shader_usage(self, profiler):
        """Test shader usage recording."""
        profiler.enable()

        profiler.record_shader_usage(
            "pbr_shader",
            time_ms=0.5,
            vertex_invocations=10000,
            fragment_invocations=50000,
        )

        stats = profiler.get_shader_stats()
        assert "pbr_shader" in stats
        assert stats["pbr_shader"].invocations == 1
        assert stats["pbr_shader"].total_time_ms == pytest.approx(0.5, rel=1e-3)

    def test_update_memory_stats(self, profiler):
        """Test memory stats update."""
        profiler.update_memory_stats(
            total_vram_bytes=8 * 1024 * 1024 * 1024,
            used_vram_bytes=2 * 1024 * 1024 * 1024,
            texture_memory_bytes=1 * 1024 * 1024 * 1024,
        )

        stats = profiler.get_memory_stats()
        assert stats.total_vram_bytes == 8 * 1024 * 1024 * 1024
        assert stats.used_vram_bytes == 2 * 1024 * 1024 * 1024

    def test_frame_lifecycle(self, profiler):
        """Test frame begin/end lifecycle."""
        profiler.enable()

        profiler.begin_frame()

        with profiler.scope("test", "rendering"):
            pass

        stats = profiler.end_frame()

        assert stats is not None
        assert stats.frame_number == 1

    def test_get_frame_stats(self, profiler):
        """Test getting frame stats."""
        profiler.enable()

        profiler.begin_frame()
        profiler.end_frame()

        stats = profiler.get_frame_stats(1)
        assert stats is not None
        assert stats.frame_number == 1

    def test_get_hottest_passes(self, profiler):
        """Test getting hottest render passes."""
        profiler.enable()
        profiler.set_simulated_gpu_time(5.0)

        with profiler.scope("shadow_pass", "shadows", RenderPassType.SHADOW):
            pass

        profiler.set_simulated_gpu_time(1.0)

        with profiler.scope("ui_pass", "ui", RenderPassType.UI):
            pass

        hottest = profiler.get_hottest_passes(top_n=2)
        assert len(hottest) == 2
        assert hottest[0][0] == "shadow_pass"

    def test_get_hottest_shaders(self, profiler):
        """Test getting hottest shaders."""
        profiler.enable()

        profiler.record_shader_usage("slow_shader", time_ms=10.0)
        profiler.record_shader_usage("fast_shader", time_ms=1.0)

        hottest = profiler.get_hottest_shaders(top_n=2)
        assert len(hottest) == 2
        assert hottest[0][0] == "slow_shader"

    def test_get_pass_timings(self, profiler):
        """Test getting pass timings."""
        profiler.enable()
        profiler.set_simulated_gpu_time(2.0)

        with profiler.scope("gbuffer", "rendering", RenderPassType.GBUFFER):
            pass

        timings = profiler.get_pass_timings()
        assert "gbuffer" in timings
        assert timings["gbuffer"].pass_type == RenderPassType.GBUFFER

    def test_clear(self, profiler):
        """Test clearing profiler data."""
        profiler.enable()

        with profiler.scope("test", "test"):
            pass

        profiler.clear()

        assert len(profiler.get_samples()) == 0
        assert len(profiler.get_pass_timings()) == 0

    def test_sample_filtering(self, profiler):
        """Test sample filtering by category."""
        profiler.enable()
        profiler.set_simulated_gpu_time(1.0)

        with profiler.scope("pass1", "shadows"):
            pass
        with profiler.scope("pass2", "lighting"):
            pass

        shadow_samples = profiler.get_samples(category="shadows")
        assert len(shadow_samples) == 1
        assert shadow_samples[0].category == "shadows"

    def test_to_dict(self, profiler):
        """Test dictionary export."""
        profiler.enable()

        with profiler.scope("test", "test"):
            pass

        data = profiler.to_dict()

        assert "state" in data
        assert "sample_count" in data
        assert data["sample_count"] == 1

    def test_listener_callback(self, profiler):
        """Test sample listener callbacks."""
        profiler.enable()
        samples_received = []

        def on_sample(sample):
            samples_received.append(sample)

        profiler.add_listener(on_sample)

        with profiler.scope("test", "test"):
            pass

        assert len(samples_received) == 1

        profiler.remove_listener(on_sample)
