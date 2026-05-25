"""Stress and verification tests for GPU crowd rendering system.

Tests cover InstanceBuffer overflow protection, memory layout correctness,
dynamic buffer growth, and 10,000-instance stress capacity.

Source: engine/animation/crowds/crowd_renderer.py
Architecture reference: docs/gap_sets/GAPSET_2_FRAME_GRAPH/PHASE_1_ARCH.md
"""

from __future__ import annotations

import pytest

from engine.animation.crowds.crowd_renderer import (
    CrowdInstance,
    InstanceBuffer,
    InstanceBufferOverflowError,
)
from engine.core.math import Vec3, Vec4, Quat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_instance(index: int, visible: bool = True) -> CrowdInstance:
    """Create a deterministic crowd instance for testing."""
    return CrowdInstance(
        position=Vec3(float(index * 10), float(index), 0.0),
        rotation=Quat.identity(),
        scale=1.0,
        animation_index=index % 16,
        animation_time=float(index) * 0.1,
        animation_speed=1.0,
        tint_color=Vec4(1.0, 1.0, 1.0, 1.0),
        lod_level=0,
        visible=visible,
        instance_id=index + 1,  # explicit to avoid _next_id side effects
    )


@pytest.fixture(autouse=True)
def reset_next_id() -> None:
    """Reset the class-level auto-increment ID before each test."""
    CrowdInstance._next_id = 0


# ---------------------------------------------------------------------------
# Overflow Protection
# ---------------------------------------------------------------------------

class TestOverflowProtection:
    """InstanceBuffer must reject instances past max_capacity."""

    def test_buffer_overflow_raises_at_max(self) -> None:
        """Adding past max_capacity raises InstanceBufferOverflowError."""
        buffer = InstanceBuffer()
        buffer.max_capacity = 100
        buffer.reserve(100)

        for i in range(100):
            buffer.add_instance(make_instance(i))

        with pytest.raises(InstanceBufferOverflowError):
            buffer.add_instance(make_instance(101))

    def test_reserve_past_max_raises(self) -> None:
        """Reserving more than max_capacity raises immediately."""
        buffer = InstanceBuffer()
        buffer.max_capacity = 50
        with pytest.raises(InstanceBufferOverflowError):
            buffer.reserve(100)

    def test_reserve_exact_max_succeeds(self) -> None:
        """Reserving exactly max_capacity is allowed."""
        buffer = InstanceBuffer()
        buffer.max_capacity = 50
        buffer.reserve(50)
        assert buffer.capacity == 50

    def test_overflow_error_is_exception_subclass(self) -> None:
        """InstanceBufferOverflowError inherits from Exception."""
        assert issubclass(InstanceBufferOverflowError, Exception)

    def test_overflow_error_message_includes_capacity(self) -> None:
        """Overflow error message contains the max capacity value."""
        buffer = InstanceBuffer()
        buffer.max_capacity = 5
        buffer.reserve(5)
        for i in range(5):
            buffer.add_instance(make_instance(i))
        with pytest.raises(InstanceBufferOverflowError, match="5"):
            buffer.add_instance(make_instance(99))


# ---------------------------------------------------------------------------
# Memory Layout
# ---------------------------------------------------------------------------

class TestMemoryLayout:
    """Per-instance packed data must match expected byte layout.

    Architecture reference layout:
      Transform  (4x4 matrix)  16 floats  = 64 bytes
      Animation  (4 floats)     4 floats  = 16 bytes
      Color (RGBA tint)         4 floats  = 16 bytes
      Total                     24 floats  = 96 bytes
    """

    def test_single_instance_byte_size(self) -> None:
        """One instance must occupy exactly 96 bytes (24 floats x 4)."""
        buffer = InstanceBuffer()
        buffer.add_instance(make_instance(0))
        assert buffer.get_memory_size_bytes() == 96

    def test_multiple_instances_scale_linear(self) -> None:
        """N instances must produce N * 96 bytes."""
        buffer = InstanceBuffer()
        n = 10
        for i in range(n):
            buffer.add_instance(make_instance(i))
        assert buffer.get_memory_size_bytes() == n * 96

    def test_memory_size_after_clear_is_zero(self) -> None:
        """Clearing the buffer must zero its memory footprint."""
        buffer = InstanceBuffer()
        buffer.add_instance(make_instance(0))
        buffer.clear()
        assert buffer.get_memory_size_bytes() == 0
        assert buffer.instance_count == 0
        assert buffer.dirty is True

    def test_float_counts_per_data_array(self) -> None:
        """Each data array must have correct float count per instance.

        After reserve(10) + add_instance(0):
          transform_data: 10 * 16 = 160 floats
          animation_data:  10 * 4 =  40 floats
          color_data:      10 * 4 =  40 floats
        """
        buffer = InstanceBuffer()
        buffer.max_capacity = 100
        buffer.reserve(10)
        buffer.add_instance(make_instance(0))

        assert len(buffer.transform_data) == 160
        assert len(buffer.animation_data) == 40
        assert len(buffer.color_data) == 40

    def test_transform_data_is_16_floats_per_instance(self) -> None:
        """Transform matrix uses exactly 16 floats per instance."""
        buffer = InstanceBuffer()
        for i in range(5):
            buffer.add_instance(make_instance(i))

        # After 5 instances, transform_data should have 5 * 16 = 80 floats
        # (extended dynamically since no reserve was called)
        assert len(buffer.transform_data) == 5 * 16
        assert len(buffer.animation_data) == 5 * 4
        assert len(buffer.color_data) == 5 * 4


# ---------------------------------------------------------------------------
# Dynamic Growth
# ---------------------------------------------------------------------------

class TestDynamicGrowth:
    """InstanceBuffer must auto-grow when reserved capacity is exceeded."""

    def test_grow_when_capacity_exceeded(self) -> None:
        """Buffer extends capacity via _grow when hitting capacity < max."""
        buffer = InstanceBuffer()
        buffer.max_capacity = 200
        buffer.capacity = 10  # small initial capacity

        # Fill the initial capacity
        for i in range(10):
            buffer.add_instance(make_instance(i))

        assert buffer.instance_count == 10
        assert buffer.capacity == 10

        # 11th instance triggers _grow:
        #   new_capacity = max(10 * 2, 64) = 64
        #   growth = 64 - 10 = 54
        before_transform_len = len(buffer.transform_data)
        buffer.add_instance(make_instance(10))

        expected_growth = 54  # max(10*2, 64) - 10 = 64 - 10
        assert buffer.capacity == 64
        assert buffer.instance_count == 11
        assert len(buffer.transform_data) == before_transform_len + expected_growth * 16

    def test_growth_respects_max_capacity(self) -> None:
        """_grow must not exceed max_capacity."""
        buffer = InstanceBuffer()
        buffer.max_capacity = 70  # smaller than theoretical growth target
        buffer.capacity = 10

        for i in range(10):
            buffer.add_instance(make_instance(i))

        # _grow would target 64 (max(20, 64)), clamped to 70
        # growth = 70 - 10 = 60
        buffer.add_instance(make_instance(10))
        assert buffer.capacity == 64  # max(10*2, 64)=64, min(64, 70)=64
        assert buffer.capacity <= buffer.max_capacity

    def test_multiple_growth_steps(self) -> None:
        """Buffer must survive multiple growth cycles."""
        buffer = InstanceBuffer()
        buffer.max_capacity = 500
        buffer.capacity = 10

        for i in range(10):
            buffer.add_instance(make_instance(i))

        # First growth: 10 -> 64
        for i in range(10, 64):
            buffer.add_instance(make_instance(i))

        assert buffer.capacity == 64
        assert buffer.instance_count == 64

        # Second growth: 64 -> 128
        buffer.add_instance(make_instance(64))
        assert buffer.capacity == 128
        assert buffer.instance_count == 65

    def test_growth_maintains_data_integrity(self) -> None:
        """After growth, previously written data must be intact."""
        buffer = InstanceBuffer()
        buffer.max_capacity = 200
        buffer.capacity = 5

        # Store instances with identifiable data
        instances = [make_instance(i) for i in range(5)]
        for inst in instances:
            buffer.add_instance(inst)

        # Trigger growth
        new_inst = make_instance(5)
        buffer.add_instance(new_inst)

        # Verify first instance's transform data is still correct
        # (transform_data[0:16] should match the first instance's matrix)
        transform0 = instances[0].get_transform_matrix()
        for j in range(16):
            assert buffer.transform_data[j] == transform0.m[j], (
                f"Transform[{j}] corrupted after growth"
            )


# ---------------------------------------------------------------------------
# 10,000-Instance Stress Tests
# ---------------------------------------------------------------------------

class TestStress10K:
    """InstanceBuffer must handle 10,000 instances without crash.

    Target: 10,000+ instances at 60 fps.
    Default max_capacity = MAX_INSTANCES_PER_BATCH (1000) * 10 = 10000.
    """

    def test_10000_instances_without_crash(self) -> None:
        """Buffer must accept and hold 10,000 instances."""
        buffer = InstanceBuffer()
        assert buffer.max_capacity == 10000  # from config default

        buffer.reserve(10000)
        for i in range(10000):
            buffer.add_instance(make_instance(i))

        assert buffer.instance_count == 10000
        assert buffer.capacity == 10000

    def test_10000_instance_memory_size(self) -> None:
        """10,000 instances must produce expected memory footprint."""
        buffer = InstanceBuffer()
        buffer.reserve(10000)
        for i in range(10000):
            buffer.add_instance(make_instance(i))

        # 10000 * (16 + 4 + 4) floats * 4 bytes = 960,000 bytes
        expected = 10000 * (16 + 4 + 4) * 4
        assert buffer.get_memory_size_bytes() == expected

    def test_10000_instances_dirty_flag(self) -> None:
        """After adding instances, dirty flag must be True."""
        buffer = InstanceBuffer()
        buffer.reserve(10000)
        for i in range(10000):
            buffer.add_instance(make_instance(i))
        assert buffer.dirty is True

    def test_clear_after_10000_instances(self) -> None:
        """Clearing after 10K instances must reset state correctly."""
        buffer = InstanceBuffer()
        buffer.reserve(10000)
        for i in range(10000):
            buffer.add_instance(make_instance(i))

        buffer.clear()
        assert buffer.instance_count == 0
        assert buffer.get_memory_size_bytes() == 0
        assert buffer.dirty is True

    def test_buffer_reuse_after_clear(self) -> None:
        """Buffer must accept new instances after a full clear."""
        buffer = InstanceBuffer()
        buffer.reserve(5000)

        # Fill and clear
        for i in range(5000):
            buffer.add_instance(make_instance(i))
        buffer.clear()

        # Reuse
        for i in range(5000):
            buffer.add_instance(make_instance(i))
        assert buffer.instance_count == 5000

    def test_stress_update_does_not_corrupt(self) -> None:
        """Updating instance data at scale must not corrupt data."""
        buffer = InstanceBuffer()
        buffer.reserve(100)
        for i in range(100):
            buffer.add_instance(make_instance(i))

        # Update instances (simulate per-frame update)
        for i in range(100):
            inst = make_instance(i + 100)
            buffer.update_instance(i, inst)

        # Check first instance was overwritten correctly
        updated = make_instance(100)
        matrix = updated.get_transform_matrix()
        for j in range(16):
            assert buffer.transform_data[j] == matrix.m[j], (
                f"Transform[{j}] corrupted after update at index 0"
            )

        assert buffer.dirty is True
        assert buffer.instance_count == 100
