"""Tests for engine.core.frame -- FrameTimer, FrameAllocator, FrameContext, FixedTimestepAccumulator."""

import time

import pytest

from engine.core.frame import (
    FramePhase,
    FrameTimer,
    FrameAllocator,
    FrameContext,
    FixedTimestepAccumulator,
)
from engine.core.constants import DEFAULT_FRAME_ALLOCATOR_SIZE, MAX_DELTA_TIME


# ---------------------------------------------------------------------------
# FrameTimer
# ---------------------------------------------------------------------------

class TestFrameTimer:
    def test_initial_state(self):
        ft = FrameTimer()
        assert ft.frame_count == 0
        assert ft.delta_time == 0.0
        assert ft.total_time == 0.0

    def test_begin_end_increments_count(self):
        ft = FrameTimer()
        ft.begin_frame()
        ft.end_frame()
        assert ft.frame_count == 1

    def test_delta_time_positive_after_sleep(self):
        ft = FrameTimer()
        ft.begin_frame()
        ft.end_frame()
        time.sleep(0.02)
        ft.begin_frame()
        assert ft.unscaled_delta_time >= 0.01
        ft.end_frame()

    def test_time_scale_affects_delta(self):
        ft = FrameTimer(time_scale=2.0)
        ft.begin_frame()
        ft.end_frame()
        time.sleep(0.02)
        ft.begin_frame()
        # delta_time should be approximately 2x unscaled
        ratio = ft.delta_time / ft.unscaled_delta_time if ft.unscaled_delta_time > 0 else 0
        assert abs(ratio - 2.0) < 0.01
        ft.end_frame()

    def test_time_scale_does_not_affect_unscaled(self):
        ft = FrameTimer(time_scale=0.5)
        ft.begin_frame()
        ft.end_frame()
        time.sleep(0.02)
        ft.begin_frame()
        assert ft.unscaled_delta_time >= 0.01
        ft.end_frame()

    def test_time_scale_setter_clamps_negative(self):
        ft = FrameTimer()
        ft.time_scale = -1.0
        assert ft.time_scale == 0.0

    def test_delta_clamped_to_max(self):
        ft = FrameTimer()
        # Simulate a long gap by manually setting _last_time far in the past
        ft.begin_frame()
        ft._last_time = time.perf_counter() - 1.0  # 1 second ago
        ft.begin_frame()
        assert ft.unscaled_delta_time <= MAX_DELTA_TIME

    def test_total_time_accumulates(self):
        ft = FrameTimer()
        for _ in range(3):
            ft.begin_frame()
            time.sleep(0.001)
            ft.end_frame()
        assert ft.total_time > 0.0


# ---------------------------------------------------------------------------
# FixedTimestepAccumulator
# ---------------------------------------------------------------------------

class TestFixedTimestepAccumulator:
    def test_no_tick_initially(self):
        acc = FixedTimestepAccumulator(fixed_dt=1.0 / 60.0)
        assert not acc.should_tick()

    def test_tick_after_accumulate(self):
        acc = FixedTimestepAccumulator(fixed_dt=1.0 / 60.0)
        acc.accumulate(1.0 / 60.0)
        assert acc.should_tick()

    def test_consume_reduces_accumulator(self):
        acc = FixedTimestepAccumulator(fixed_dt=0.1)
        acc.accumulate(0.25)
        ticks = 0
        while acc.should_tick():
            acc.consume_tick()
            ticks += 1
        assert ticks == 2
        assert abs(acc.accumulator - 0.05) < 1e-9

    def test_alpha_interpolation(self):
        acc = FixedTimestepAccumulator(fixed_dt=0.1)
        acc.accumulate(0.05)
        assert abs(acc.alpha - 0.5) < 1e-9

    def test_reset(self):
        acc = FixedTimestepAccumulator(fixed_dt=0.1)
        acc.accumulate(0.5)
        acc.reset()
        assert acc.accumulator == 0.0


# ---------------------------------------------------------------------------
# FrameAllocator
# ---------------------------------------------------------------------------

class TestFrameAllocator:
    def test_initial_state(self):
        fa = FrameAllocator(size=256)
        assert fa.size == 256
        assert fa.used == 0
        assert fa.remaining == 256

    def test_allocate_returns_offset(self):
        fa = FrameAllocator(size=256)
        off1 = fa.allocate(64)
        assert off1 == 0
        off2 = fa.allocate(32)
        assert off2 == 64
        assert fa.used == 96

    def test_allocate_exhausted_raises(self):
        fa = FrameAllocator(size=64)
        fa.allocate(64)
        with pytest.raises(MemoryError):
            fa.allocate(1)

    def test_allocate_negative_raises(self):
        fa = FrameAllocator(size=64)
        with pytest.raises(ValueError):
            fa.allocate(-1)

    def test_reset_reclaims(self):
        fa = FrameAllocator(size=128)
        fa.allocate(128)
        fa.reset()
        assert fa.used == 0
        assert fa.remaining == 128
        # Can allocate again
        off = fa.allocate(64)
        assert off == 0

    def test_default_size(self):
        fa = FrameAllocator()
        assert fa.size == DEFAULT_FRAME_ALLOCATOR_SIZE


# ---------------------------------------------------------------------------
# FrameContext
# ---------------------------------------------------------------------------

class TestFrameContext:
    def test_creation(self):
        ctx = FrameContext(
            frame_number=42,
            delta_time=0.016,
            total_time=1.0,
            time_scale=1.0,
            phase=FramePhase.UPDATE,
        )
        assert ctx.frame_number == 42
        assert ctx.delta_time == 0.016
        assert ctx.phase == FramePhase.UPDATE

    def test_frozen(self):
        ctx = FrameContext(
            frame_number=0,
            delta_time=0.0,
            total_time=0.0,
            time_scale=1.0,
            phase=FramePhase.PRE_UPDATE,
        )
        with pytest.raises(AttributeError):
            ctx.frame_number = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FramePhase enum
# ---------------------------------------------------------------------------

class TestFramePhase:
    def test_ordering(self):
        phases = list(FramePhase)
        assert phases == [
            FramePhase.PRE_UPDATE,
            FramePhase.UPDATE,
            FramePhase.POST_UPDATE,
            FramePhase.PRE_RENDER,
            FramePhase.RENDER,
            FramePhase.POST_RENDER,
        ]

    def test_values_sequential(self):
        for i, phase in enumerate(FramePhase):
            assert phase.value == i
