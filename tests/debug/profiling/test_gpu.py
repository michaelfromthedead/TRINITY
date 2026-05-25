"""Tests for GPU profiler."""

import time

import pytest

from engine.debug.profiling.gpu import (
    GPUFrameTiming,
    GPUPassTiming,
    GPUPassType,
    GPUProfiler,
    get_default_gpu_profiler,
    set_default_gpu_profiler,
)


class TestGPUPassType:
    """Tests for GPUPassType enum."""

    def test_all_pass_types_exist(self) -> None:
        """Test all expected pass types exist."""
        expected = [
            "SHADOW", "DEPTH_PREPASS", "GBUFFER", "LIGHTING",
            "FORWARD", "TRANSPARENT", "POST_PROCESS", "UI",
            "COMPUTE", "CUSTOM"
        ]
        for type_name in expected:
            assert hasattr(GPUPassType, type_name)


class TestGPUPassTiming:
    """Tests for GPUPassTiming dataclass."""

    def test_duration_calculation(self) -> None:
        """Test duration is calculated correctly."""
        timing = GPUPassTiming(
            name="test",
            start_ns=1_000_000_000,
            end_ns=1_016_666_666
        )
        assert timing.duration_ns == 16_666_666
        assert abs(timing.duration_ms - 16.666666) < 0.001

    def test_duration_incomplete(self) -> None:
        """Test duration for incomplete timing returns 0."""
        timing = GPUPassTiming(name="test", start_ns=1_000_000_000)
        assert timing.duration_ns == 0

    def test_is_complete(self) -> None:
        """Test completion status."""
        incomplete = GPUPassTiming(name="test", start_ns=1000)
        complete = GPUPassTiming(name="test", start_ns=1000, end_ns=2000)

        assert not incomplete.is_complete
        assert complete.is_complete


class TestGPUFrameTiming:
    """Tests for GPUFrameTiming dataclass."""

    def test_total_gpu_time(self) -> None:
        """Test total GPU time calculation."""
        frame = GPUFrameTiming(frame_index=0)
        frame.passes = [
            GPUPassTiming(name="a", start_ns=0, end_ns=10_000_000),
            GPUPassTiming(name="b", start_ns=10_000_000, end_ns=25_000_000),
        ]

        assert frame.total_gpu_time_ns == 25_000_000
        assert frame.total_gpu_time_ms == 25.0

    def test_frame_time(self) -> None:
        """Test frame time calculation."""
        frame = GPUFrameTiming(
            frame_index=0,
            frame_start_ns=0,
            frame_end_ns=16_666_666
        )
        assert abs(frame.frame_time_ms - 16.666666) < 0.001


class TestGPUProfiler:
    """Tests for GPUProfiler class."""

    def test_begin_end_frame(self) -> None:
        """Test basic frame profiling."""
        profiler = GPUProfiler()

        profiler.begin_frame()
        time.sleep(0.001)
        frame = profiler.end_frame()

        assert frame is not None
        assert frame.frame_index == 0
        assert frame.frame_time_ms >= 1.0

    def test_begin_end_pass(self) -> None:
        """Test render pass profiling."""
        profiler = GPUProfiler()

        profiler.begin_frame()

        pass_timing = profiler.begin_pass("shadow", GPUPassType.SHADOW)
        assert pass_timing is not None
        assert pass_timing.name == "shadow"
        assert pass_timing.pass_type == GPUPassType.SHADOW

        time.sleep(0.001)

        ended = profiler.end_pass()
        assert ended is pass_timing
        assert ended.duration_ms >= 1.0

        profiler.end_frame()

    def test_multiple_passes(self) -> None:
        """Test multiple render passes in a frame."""
        profiler = GPUProfiler()

        profiler.begin_frame()

        profiler.begin_pass("shadow", GPUPassType.SHADOW)
        time.sleep(0.001)
        profiler.end_pass()

        profiler.begin_pass("forward", GPUPassType.FORWARD)
        time.sleep(0.001)
        profiler.end_pass()

        frame = profiler.end_frame()

        assert frame is not None
        assert len(frame.passes) == 2
        assert frame.passes[0].name == "shadow"
        assert frame.passes[1].name == "forward"

    def test_auto_frame_creation(self) -> None:
        """Test that begin_pass auto-creates frame if needed."""
        profiler = GPUProfiler()

        profiler.begin_pass("test")
        profiler.end_pass()
        frame = profiler.end_frame()

        assert frame is not None
        assert len(frame.passes) == 1

    def test_frame_history(self) -> None:
        """Test frame history is maintained."""
        profiler = GPUProfiler(history_size=5)

        for i in range(7):
            profiler.begin_frame()
            profiler.end_frame()

        # History should be limited to 5 frames
        frame = profiler.get_frame_timing(0)  # Most recent
        assert frame is not None
        assert frame.frame_index == 6

        # Can't access beyond history
        assert profiler.get_frame_timing(5) is None

    def test_get_pass_timings(self) -> None:
        """Test getting pass timings for a frame."""
        profiler = GPUProfiler()

        profiler.begin_frame()
        profiler.begin_pass("test", GPUPassType.SHADOW)
        profiler.end_pass()
        profiler.end_frame()

        timings = profiler.get_pass_timings(0)
        assert len(timings) == 1
        assert timings[0].name == "test"

    def test_get_average_pass_times(self) -> None:
        """Test averaging pass times across frames."""
        profiler = GPUProfiler()

        for _ in range(3):
            profiler.begin_frame()
            profiler.begin_pass("test")
            time.sleep(0.001)
            profiler.end_pass()
            profiler.end_frame()

        averages = profiler.get_average_pass_times(3)
        assert "test" in averages
        assert averages["test"] >= 1.0

    def test_get_average_frame_time(self) -> None:
        """Test averaging frame times."""
        profiler = GPUProfiler()

        for _ in range(3):
            profiler.begin_frame()
            time.sleep(0.001)
            profiler.end_frame()

        avg = profiler.get_average_frame_time(3)
        assert avg >= 1.0

    def test_disabled_profiler(self) -> None:
        """Test disabled profiler returns None."""
        profiler = GPUProfiler(enabled=False)

        profiler.begin_frame()
        assert profiler.end_frame() is None

        assert profiler.begin_pass("test") is None
        assert profiler.end_pass() is None

    def test_reset(self) -> None:
        """Test reset clears all data."""
        profiler = GPUProfiler()

        profiler.begin_frame()
        profiler.begin_pass("test")
        profiler.end_pass()
        profiler.end_frame()

        profiler.reset()

        assert profiler.get_frame_timing(0) is None
        assert profiler.get_pass_timings(0) == []

    def test_format_frame_breakdown(self) -> None:
        """Test formatting frame breakdown."""
        profiler = GPUProfiler()

        profiler.begin_frame()
        profiler.begin_pass("shadow", GPUPassType.SHADOW)
        profiler.end_pass()
        profiler.begin_pass("forward", GPUPassType.FORWARD)
        profiler.end_pass()
        profiler.end_frame()

        output = profiler.format_frame_breakdown(0)

        assert "Frame 0" in output
        assert "shadow" in output
        assert "forward" in output

    def test_format_no_data(self) -> None:
        """Test formatting when no data available."""
        profiler = GPUProfiler()
        output = profiler.format_frame_breakdown(0)
        assert "No frame data" in output


class TestDefaultGPUProfiler:
    """Tests for default GPU profiler instance."""

    def test_get_default_profiler(self) -> None:
        """Test getting default profiler instance."""
        profiler = get_default_gpu_profiler()
        assert profiler is not None
        assert isinstance(profiler, GPUProfiler)

    def test_set_default_profiler(self) -> None:
        """Test setting default profiler instance."""
        original = get_default_gpu_profiler()
        new_profiler = GPUProfiler()

        set_default_gpu_profiler(new_profiler)
        assert get_default_gpu_profiler() is new_profiler

        # Restore original
        set_default_gpu_profiler(original)
