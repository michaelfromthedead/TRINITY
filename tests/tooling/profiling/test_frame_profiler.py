"""Tests for the frame profiler module."""

from __future__ import annotations

import time

import pytest

from engine.tooling.profiling.frame_profiler import (
    FrameProfiler,
    FrameProfilerState,
    FrameData,
    FrameTimeline,
    FramePhase,
    PhaseTimestamp,
    SpikeDetector,
    FrameBudget,
)


class TestPhaseTimestamp:
    """Tests for PhaseTimestamp."""

    def test_creation(self):
        """Test basic creation."""
        phase = PhaseTimestamp(
            phase=FramePhase.PHYSICS,
            start_time=0.0,
            end_time=0.002,
        )
        assert phase.phase == FramePhase.PHYSICS
        assert phase.duration_ms == pytest.approx(2.0, rel=1e-3)

    def test_custom_name(self):
        """Test with custom name."""
        phase = PhaseTimestamp(
            phase=FramePhase.CUSTOM,
            start_time=0.0,
            end_time=0.001,
            custom_name="my_custom_phase",
        )
        assert phase.custom_name == "my_custom_phase"

    def test_to_dict(self):
        """Test dictionary conversion."""
        phase = PhaseTimestamp(
            phase=FramePhase.RENDERING,
            start_time=0.0,
            end_time=0.005,
        )
        data = phase.to_dict()

        assert data["phase"] == "rendering"
        assert data["duration_ms"] == pytest.approx(5.0, rel=1e-3)


class TestFrameData:
    """Tests for FrameData."""

    def test_creation(self):
        """Test basic creation."""
        frame = FrameData(
            frame_number=1,
            start_time=0.0,
        )
        assert frame.frame_number == 1
        assert frame.is_spike is False

    def test_frame_time_ms(self):
        """Test frame time calculation."""
        frame = FrameData(
            frame_number=1,
            start_time=0.0,
            end_time=0.016667,  # ~60 FPS
        )
        assert frame.frame_time_ms == pytest.approx(16.667, rel=1e-3)

    def test_fps_calculation(self):
        """Test FPS calculation."""
        frame = FrameData(
            frame_number=1,
            start_time=0.0,
            end_time=0.016667,
        )
        assert frame.fps == pytest.approx(60.0, rel=1e-1)

    def test_add_phase(self):
        """Test adding phases."""
        frame = FrameData(
            frame_number=1,
            start_time=0.0,
        )

        frame.add_phase(FramePhase.INPUT, 0.0, 0.001)
        frame.add_phase(FramePhase.PHYSICS, 0.001, 0.005)

        assert len(frame.phases) == 2

    def test_phase_breakdown(self):
        """Test phase breakdown calculation."""
        frame = FrameData(
            frame_number=1,
            start_time=0.0,
        )

        frame.add_phase(FramePhase.INPUT, 0.0, 0.001)
        frame.add_phase(FramePhase.PHYSICS, 0.001, 0.005)
        frame.add_phase(FramePhase.INPUT, 0.005, 0.006)  # Second input phase

        breakdown = frame.phase_breakdown

        assert breakdown["input"] == pytest.approx(2.0, rel=1e-3)
        assert breakdown["physics"] == pytest.approx(4.0, rel=1e-3)

    def test_finalize(self):
        """Test frame finalization."""
        frame = FrameData(
            frame_number=1,
            start_time=time.perf_counter(),
        )

        time.sleep(0.001)
        frame.finalize()

        assert frame.end_time > frame.start_time

    def test_to_dict(self):
        """Test dictionary conversion."""
        frame = FrameData(
            frame_number=100,
            start_time=0.0,
            end_time=0.016,
            cpu_time_ms=10.0,
            gpu_time_ms=5.0,
            draw_calls=1000,
        )
        data = frame.to_dict()

        assert data["frame_number"] == 100
        assert data["draw_calls"] == 1000
        assert "fps" in data


class TestFrameTimeline:
    """Tests for FrameTimeline."""

    def test_creation(self):
        """Test basic creation."""
        timeline = FrameTimeline()
        assert len(timeline.frames) == 0
        assert timeline.avg_fps == 0.0

    def test_add_frame(self):
        """Test adding frames."""
        timeline = FrameTimeline()

        for i in range(5):
            frame = FrameData(
                frame_number=i,
                start_time=i * 0.016,
                end_time=(i + 1) * 0.016,
            )
            timeline.add_frame(frame)

        assert len(timeline.frames) == 5
        assert timeline.avg_frame_time_ms == pytest.approx(16.0, rel=1e-3)

    def test_spike_count(self):
        """Test spike counting."""
        timeline = FrameTimeline()

        for i in range(3):
            frame = FrameData(
                frame_number=i,
                start_time=0.0,
                end_time=0.016,
                is_spike=(i == 1),
            )
            timeline.add_frame(frame)

        assert timeline.spike_count == 1

    def test_get_frames_filtered(self):
        """Test filtered frame retrieval."""
        timeline = FrameTimeline()

        for i in range(5):
            frame = FrameData(
                frame_number=i,
                start_time=0.0,
                end_time=0.016,
                is_spike=(i % 2 == 0),
            )
            timeline.add_frame(frame)

        spikes = timeline.get_frames(spikes_only=True)
        assert len(spikes) == 3

        range_frames = timeline.get_frames(start_frame=2, end_frame=4)
        assert len(range_frames) == 3

    def test_get_phase_averages(self):
        """Test phase average calculation."""
        timeline = FrameTimeline()

        for i in range(3):
            frame = FrameData(frame_number=i, start_time=0.0, end_time=0.016)
            frame.add_phase(FramePhase.PHYSICS, 0.0, 0.004)
            frame.add_phase(FramePhase.RENDERING, 0.004, 0.010)
            timeline.add_frame(frame)

        averages = timeline.get_phase_averages()

        assert "physics" in averages
        assert averages["physics"] == pytest.approx(4.0, rel=1e-3)

    def test_to_dict(self):
        """Test dictionary conversion."""
        timeline = FrameTimeline()
        frame = FrameData(frame_number=1, start_time=0.0, end_time=0.016)
        timeline.add_frame(frame)

        data = timeline.to_dict()

        assert data["frame_count"] == 1
        assert "avg_fps" in data


class TestSpikeDetector:
    """Tests for SpikeDetector."""

    def test_creation(self):
        """Test basic creation."""
        detector = SpikeDetector(threshold_ms=33.3)
        assert detector.threshold_ms == pytest.approx(33.3, rel=1e-3)

    def test_update_threshold(self):
        """Test threshold update from FPS."""
        detector = SpikeDetector()
        detector.update_threshold(60.0)
        assert detector.threshold_ms == pytest.approx(16.67, rel=1e-2)

    def test_check_spike_fixed(self):
        """Test spike detection with fixed threshold."""
        detector = SpikeDetector(threshold_ms=16.0, adaptive=False)

        assert detector.check_spike(10.0) is False
        assert detector.check_spike(20.0) is True

    def test_check_spike_adaptive(self):
        """Test spike detection with adaptive threshold."""
        detector = SpikeDetector(threshold_ms=100.0, adaptive=True, sensitivity=2.0)

        # Build up history
        for _ in range(10):
            detector.check_spike(10.0)

        # This should be detected as a spike (2x average)
        assert detector.check_spike(25.0) is True

    def test_get_threshold(self):
        """Test getting current threshold."""
        detector = SpikeDetector(threshold_ms=16.0, adaptive=False)
        assert detector.get_threshold() == pytest.approx(16.0, rel=1e-3)


class TestFrameBudget:
    """Tests for FrameBudget."""

    def test_creation(self):
        """Test basic creation."""
        budget = FrameBudget(target_fps=60.0)
        assert budget.target_fps == 60.0
        assert budget.total_budget_ms == pytest.approx(16.67, rel=1e-2)

    def test_set_phase_budget(self):
        """Test setting phase budgets."""
        budget = FrameBudget()

        budget.set_phase_budget(FramePhase.PHYSICS, 4.0)
        budget.set_phase_budget(FramePhase.RENDERING, 8.0)

        assert budget.phase_budgets[FramePhase.PHYSICS] == 4.0
        assert budget.phase_budgets[FramePhase.RENDERING] == 8.0

    def test_check_budget_within(self):
        """Test budget check when within budget."""
        budget = FrameBudget(target_fps=60.0)
        budget.set_phase_budget(FramePhase.PHYSICS, 5.0)

        frame = FrameData(
            frame_number=1,
            start_time=0.0,
            end_time=0.010,  # 10ms total
        )
        frame.add_phase(FramePhase.PHYSICS, 0.0, 0.003)

        result = budget.check_budget(frame)

        assert result["within_budget"] is True
        assert len(result["phase_violations"]) == 0

    def test_check_budget_over(self):
        """Test budget check when over budget."""
        budget = FrameBudget(target_fps=60.0)
        budget.set_phase_budget(FramePhase.PHYSICS, 2.0)

        frame = FrameData(
            frame_number=1,
            start_time=0.0,
            end_time=0.020,  # 20ms - over 16.67ms budget
        )
        frame.add_phase(FramePhase.PHYSICS, 0.0, 0.005)  # 5ms - over 2ms budget

        result = budget.check_budget(frame)

        assert result["total_over"] is True
        assert len(result["phase_violations"]) == 1


class TestFrameProfiler:
    """Tests for FrameProfiler."""

    @pytest.fixture
    def profiler(self):
        """Create a fresh profiler instance."""
        return FrameProfiler(target_fps=60.0)

    def test_initial_state(self, profiler):
        """Test initial profiler state."""
        assert profiler.state == FrameProfilerState.DISABLED
        assert not profiler.is_enabled
        assert profiler.current_frame_number == 0

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
        assert profiler.state == FrameProfilerState.PAUSED

        profiler.resume()
        assert profiler.state == FrameProfilerState.ENABLED

    def test_begin_frame(self, profiler):
        """Test frame begin."""
        profiler.enable()

        frame_num = profiler.begin_frame()

        assert frame_num == 1
        assert profiler.current_frame_number == 1

    def test_end_frame(self, profiler):
        """Test frame end."""
        profiler.enable()

        profiler.begin_frame()
        time.sleep(0.001)
        frame = profiler.end_frame()

        assert frame is not None
        assert frame.frame_number == 1
        assert frame.frame_time_ms > 0

    def test_frame_lifecycle(self, profiler):
        """Test complete frame lifecycle."""
        profiler.enable()

        profiler.begin_frame()
        profiler.begin_phase(FramePhase.INPUT)
        profiler.end_phase()
        profiler.begin_phase(FramePhase.PHYSICS)
        profiler.end_phase()
        frame = profiler.end_frame()

        assert len(frame.phases) == 2

    def test_record_phase_direct(self, profiler):
        """Test direct phase recording."""
        profiler.enable()

        profiler.begin_frame()
        profiler.record_phase(FramePhase.RENDERING, duration_ms=5.0)
        frame = profiler.end_frame()

        assert len(frame.phases) == 1
        assert frame.phases[0].duration_ms == pytest.approx(5.0, rel=1e-3)

    def test_set_frame_stats(self, profiler):
        """Test setting frame statistics."""
        profiler.enable()

        profiler.begin_frame()
        profiler.set_frame_gpu_time(8.0)
        profiler.set_frame_stats(draw_calls=500, triangles=50000, memory_mb=256.0)
        frame = profiler.end_frame()

        assert frame.gpu_time_ms == 8.0
        assert frame.draw_calls == 500
        assert frame.triangles == 50000
        assert frame.memory_used_mb == 256.0

    def test_add_custom_data(self, profiler):
        """Test adding custom frame data."""
        profiler.enable()

        profiler.begin_frame()
        profiler.add_custom_data("my_metric", 42)
        frame = profiler.end_frame()

        assert frame.custom_data["my_metric"] == 42

    def test_spike_detection(self, profiler):
        """Test spike detection."""
        profiler.enable()
        profiler.set_adaptive_spike_detection(False)
        profiler.set_target_fps(60.0)  # 16.67ms threshold

        # Normal frame
        profiler.begin_frame()
        time.sleep(0.001)
        frame1 = profiler.end_frame()

        # Force a spike by sleeping longer (simulated)
        profiler._spike_detector.threshold_ms = 5.0
        profiler.begin_frame()
        time.sleep(0.01)  # 10ms > 5ms threshold
        frame2 = profiler.end_frame()

        assert frame2.is_spike is True

    def test_get_frame(self, profiler):
        """Test getting specific frame."""
        profiler.enable()

        profiler.begin_frame()
        profiler.end_frame()
        profiler.begin_frame()
        profiler.end_frame()

        frame = profiler.get_frame(1)
        assert frame is not None
        assert frame.frame_number == 1

    def test_get_frames(self, profiler):
        """Test getting frame history."""
        profiler.enable()

        for _ in range(5):
            profiler.begin_frame()
            profiler.end_frame()

        frames = profiler.get_frames(count=3)
        assert len(frames) == 3

    def test_get_timeline(self, profiler):
        """Test getting frame timeline."""
        profiler.enable()

        for _ in range(3):
            profiler.begin_frame()
            time.sleep(0.001)
            profiler.end_frame()

        timeline = profiler.get_timeline()

        assert len(timeline.frames) == 3
        assert timeline.avg_frame_time_ms > 0

    def test_get_current_fps(self, profiler):
        """Test getting current FPS."""
        profiler.enable()

        for _ in range(10):
            profiler.begin_frame()
            time.sleep(0.001)  # ~1ms frame time
            profiler.end_frame()

        fps = profiler.get_current_fps()
        assert fps > 0

    def test_get_budget_status(self, profiler):
        """Test getting budget status."""
        profiler.enable()
        profiler.set_phase_budget(FramePhase.PHYSICS, 5.0)

        profiler.begin_frame()
        profiler.begin_phase(FramePhase.PHYSICS)
        time.sleep(0.01)  # 10ms > 5ms budget
        profiler.end_phase()
        profiler.end_frame()

        status = profiler.get_budget_status()
        assert "within_budget" in status

    def test_get_spike_frames(self, profiler):
        """Test getting spike frames."""
        profiler.enable()
        profiler._spike_detector.threshold_ms = 5.0
        profiler._spike_detector.adaptive = False

        profiler.begin_frame()
        time.sleep(0.001)
        profiler.end_frame()

        profiler.begin_frame()
        time.sleep(0.01)  # Spike
        profiler.end_frame()

        profiler.begin_frame()
        time.sleep(0.001)
        profiler.end_frame()

        spikes = profiler.get_spike_frames()
        assert len(spikes) == 1

    def test_get_stats(self, profiler):
        """Test getting profiler stats."""
        profiler.enable()

        for _ in range(3):
            profiler.begin_frame()
            time.sleep(0.001)
            profiler.end_frame()

        stats = profiler.get_stats()

        assert stats["frame_count"] == 3
        assert "current_fps" in stats
        assert "avg_frame_time_ms" in stats

    def test_clear(self, profiler):
        """Test clearing profiler data."""
        profiler.enable()

        for _ in range(3):
            profiler.begin_frame()
            profiler.end_frame()

        profiler.clear()

        assert profiler.current_frame_number == 0
        assert len(profiler.get_frames()) == 0

    def test_listener_callback(self, profiler):
        """Test frame completion callbacks."""
        profiler.enable()
        frames_received = []

        def on_frame(frame):
            frames_received.append(frame)

        profiler.add_listener(on_frame)

        profiler.begin_frame()
        profiler.end_frame()

        assert len(frames_received) == 1

        profiler.remove_listener(on_frame)

    def test_nested_phases(self, profiler):
        """Test nested phase handling."""
        profiler.enable()

        profiler.begin_frame()
        profiler.begin_phase(FramePhase.RENDERING)
        profiler.begin_phase(FramePhase.GPU_SUBMIT, "shadow_pass")
        profiler.end_phase()
        profiler.begin_phase(FramePhase.GPU_SUBMIT, "main_pass")
        profiler.end_phase()
        profiler.end_phase()
        frame = profiler.end_frame()

        assert len(frame.phases) == 3

    def test_to_dict(self, profiler):
        """Test dictionary export."""
        profiler.enable()

        profiler.begin_frame()
        profiler.end_frame()

        data = profiler.to_dict()

        assert "state" in data
        assert "current_frame_number" in data
        assert "stats" in data
        assert "timeline" in data
