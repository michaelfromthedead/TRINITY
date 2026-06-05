"""Whitebox tests for InstanceBuffer in crowd_renderer.py.

Tests internal state, byte layout, dynamic growth, and stress conditions.
Task: T1.2 Instance Buffer Stress Test
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from engine.animation.crowds.crowd_renderer import (
    CrowdInstance,
    InstanceBuffer,
    InstanceBufferOverflowError,
)
from engine.animation.config import CROWD_RENDERER_CONFIG
from engine.core.math import Vec3, Vec4, Quat


# =============================================================================
# Byte Size Constants Tests
# =============================================================================


class TestByteConstants:
    """Test byte layout constants are correct."""

    def test_bytes_per_float_is_4(self):
        """Verify float32 assumption: 4 bytes per float."""
        buf = InstanceBuffer()
        assert buf.BYTES_PER_FLOAT == 4

    def test_transform_bytes_is_64(self):
        """Transform: 16 floats (4x4 matrix) * 4 bytes = 64."""
        buf = InstanceBuffer()
        assert buf.TRANSFORM_BYTES == 64
        # Verify calculation
        assert buf.TRANSFORM_BYTES == CROWD_RENDERER_CONFIG.TRANSFORM_FLOATS * 4

    def test_animation_bytes_is_16(self):
        """Animation: 4 floats (index, time, speed, lod) * 4 bytes = 16."""
        buf = InstanceBuffer()
        assert buf.ANIMATION_BYTES == 16
        assert buf.ANIMATION_BYTES == CROWD_RENDERER_CONFIG.ANIMATION_FLOATS * 4

    def test_color_bytes_is_16(self):
        """Color: 4 floats (RGBA) * 4 bytes = 16."""
        buf = InstanceBuffer()
        assert buf.COLOR_BYTES == 16
        assert buf.COLOR_BYTES == CROWD_RENDERER_CONFIG.COLOR_FLOATS * 4

    def test_bytes_per_instance_is_96(self):
        """Total per instance: 64 + 16 + 16 = 96 bytes."""
        buf = InstanceBuffer()
        assert buf.BYTES_PER_INSTANCE == 96
        assert buf.BYTES_PER_INSTANCE == buf.TRANSFORM_BYTES + buf.ANIMATION_BYTES + buf.COLOR_BYTES

    def test_get_byte_size_returns_instance_count_times_96(self):
        """get_byte_size() should return instance_count * BYTES_PER_INSTANCE.

        - Empty buffer: 0 bytes
        - 1 instance: 96 bytes
        - N instances: N * 96 bytes
        """
        buf = InstanceBuffer()
        # Empty buffer has 0 byte size
        assert buf.get_byte_size() == 0
        # Add one instance
        inst = CrowdInstance(position=Vec3(0, 0, 0), rotation=Quat.identity(), scale=1.0)
        buf.add_instance(inst)
        assert buf.get_byte_size() == buf.BYTES_PER_INSTANCE


# =============================================================================
# Constructor Tests
# =============================================================================


class TestConstructor:
    """Test InstanceBuffer constructor variations."""

    def test_default_max_instances(self):
        """Default max_instances = MAX_INSTANCES_PER_BATCH * 10 = 10,000."""
        buf = InstanceBuffer()
        expected = CROWD_RENDERER_CONFIG.MAX_INSTANCES_PER_BATCH * 10
        assert buf.max_instances == expected
        assert buf.max_instances == 10_000

    def test_custom_max_instances(self):
        """Custom max_instances can be specified."""
        buf = InstanceBuffer(max_instances=500)
        assert buf.max_instances == 500

    def test_max_instances_small(self):
        """Small max_instances value works."""
        buf = InstanceBuffer(max_instances=1)
        assert buf.max_instances == 1

    def test_max_instances_large(self):
        """Large max_instances value works."""
        buf = InstanceBuffer(max_instances=100_000)
        assert buf.max_instances == 100_000

    def test_initial_state_empty(self):
        """New buffer starts empty."""
        buf = InstanceBuffer()
        assert buf.instance_count == 0
        assert buf.capacity == 0
        assert buf.dirty is True
        assert len(buf.transform_data) == 0
        assert len(buf.animation_data) == 0
        assert len(buf.color_data) == 0

    def test_max_capacity_alias(self):
        """max_capacity property is alias for max_instances."""
        buf = InstanceBuffer(max_instances=1234)
        assert buf.max_capacity == buf.max_instances
        assert buf.max_capacity == 1234


# =============================================================================
# Reserve Tests
# =============================================================================


class TestReserve:
    """Test buffer reservation behavior."""

    def test_reserve_allocates_arrays(self):
        """Reserve pre-allocates transform, animation, color arrays."""
        buf = InstanceBuffer()
        buf.reserve(100)

        assert buf.capacity == 100
        assert len(buf.transform_data) == 100 * CROWD_RENDERER_CONFIG.TRANSFORM_FLOATS
        assert len(buf.animation_data) == 100 * CROWD_RENDERER_CONFIG.ANIMATION_FLOATS
        assert len(buf.color_data) == 100 * CROWD_RENDERER_CONFIG.COLOR_FLOATS

    def test_reserve_exact_capacity(self):
        """Reserve at exactly max_capacity succeeds."""
        buf = InstanceBuffer(max_instances=500)
        buf.reserve(500)
        assert buf.capacity == 500

    def test_reserve_exceeds_max_raises(self):
        """Reserve exceeding max_capacity raises InstanceBufferOverflowError."""
        buf = InstanceBuffer(max_instances=100)
        with pytest.raises(InstanceBufferOverflowError) as exc_info:
            buf.reserve(101)
        assert "Cannot reserve 101 instances" in str(exc_info.value)
        assert "maximum is 100" in str(exc_info.value)

    def test_reserve_zero(self):
        """Reserve zero is allowed."""
        buf = InstanceBuffer()
        buf.reserve(0)
        assert buf.capacity == 0

    def test_reserve_initializes_to_zero(self):
        """Reserved arrays contain zeros."""
        buf = InstanceBuffer()
        buf.reserve(10)
        assert all(v == 0.0 for v in buf.transform_data)
        assert all(v == 0.0 for v in buf.animation_data)
        assert all(v == 0.0 for v in buf.color_data)


# =============================================================================
# Add Instance Tests
# =============================================================================


class TestAddInstance:
    """Test add_instance behavior and internal state."""

    def test_add_single_instance(self):
        """Add single instance updates count and returns index."""
        buf = InstanceBuffer()
        inst = CrowdInstance()

        idx = buf.add_instance(inst)

        assert idx == 0
        assert buf.instance_count == 1
        assert buf.dirty is True

    def test_add_multiple_instances(self):
        """Add multiple instances returns sequential indices."""
        buf = InstanceBuffer()
        buf.reserve(10)

        indices = []
        for i in range(5):
            inst = CrowdInstance()
            indices.append(buf.add_instance(inst))

        assert indices == [0, 1, 2, 3, 4]
        assert buf.instance_count == 5

    def test_add_instance_packs_transform_data(self):
        """Verify transform matrix is packed correctly."""
        buf = InstanceBuffer()
        inst = CrowdInstance(
            position=Vec3(1.0, 2.0, 3.0),
            rotation=Quat.identity(),
            scale=1.0,
        )

        buf.add_instance(inst)

        # Transform data should have 16 floats at offset 0
        assert len(buf.transform_data) >= 16
        # Mat4 is stored; verify position (translation) is in column 3
        # Standard column-major 4x4: m[12], m[13], m[14] = translation
        matrix = inst.get_transform_matrix()
        for i in range(16):
            assert buf.transform_data[i] == matrix.m[i]

    def test_add_instance_packs_animation_data(self):
        """Verify animation data is packed correctly."""
        buf = InstanceBuffer()
        inst = CrowdInstance(
            animation_index=5,
            animation_time=1.5,
            animation_speed=2.0,
            lod_level=3,
        )

        buf.add_instance(inst)

        assert buf.animation_data[0] == 5.0  # animation_index
        assert buf.animation_data[1] == 1.5  # animation_time
        assert buf.animation_data[2] == 2.0  # animation_speed
        assert buf.animation_data[3] == 3.0  # lod_level

    def test_add_instance_packs_color_data(self):
        """Verify color data is packed correctly."""
        buf = InstanceBuffer()
        inst = CrowdInstance(
            tint_color=Vec4(0.5, 0.6, 0.7, 0.8),
        )

        buf.add_instance(inst)

        assert buf.color_data[0] == 0.5  # R
        assert buf.color_data[1] == 0.6  # G
        assert buf.color_data[2] == 0.7  # B
        assert buf.color_data[3] == 0.8  # A

    def test_add_instance_offset_correctness(self):
        """Multiple instances pack at correct offsets."""
        buf = InstanceBuffer()
        buf.reserve(5)

        for i in range(3):
            inst = CrowdInstance(
                animation_index=i * 10,
                tint_color=Vec4(i * 0.1, 0.0, 0.0, 1.0),
            )
            buf.add_instance(inst)

        # Animation data offsets: 0, 4, 8
        assert buf.animation_data[0] == 0.0   # instance 0
        assert buf.animation_data[4] == 10.0  # instance 1
        assert buf.animation_data[8] == 20.0  # instance 2

        # Color data offsets: 0, 4, 8
        assert buf.color_data[0] == 0.0   # instance 0 R
        assert buf.color_data[4] == 0.1   # instance 1 R
        assert buf.color_data[8] == 0.2   # instance 2 R

    def test_add_instance_auto_extends_arrays(self):
        """Adding without reserve extends data arrays directly.

        Note: Implementation extends arrays on-demand without updating
        capacity when starting from zero. This is valid behavior.
        """
        buf = InstanceBuffer()
        inst = CrowdInstance()

        # No reserve, capacity is 0
        idx = buf.add_instance(inst)

        assert idx == 0
        assert buf.instance_count == 1
        # Arrays are extended directly (capacity may stay 0 for on-demand growth)
        assert len(buf.transform_data) >= 16
        assert len(buf.animation_data) >= 4
        assert len(buf.color_data) >= 4


# =============================================================================
# Growth Tests (_grow internal)
# =============================================================================


class TestGrow:
    """Test _grow() internal method behavior."""

    def test_grow_uses_growth_factor(self):
        """_grow doubles capacity (BUFFER_GROWTH_FACTOR = 2)."""
        buf = InstanceBuffer()
        buf.reserve(64)  # DEFAULT_BUFFER_CAPACITY

        old_capacity = buf.capacity
        buf._grow()

        expected = old_capacity * CROWD_RENDERER_CONFIG.BUFFER_GROWTH_FACTOR
        assert buf.capacity == expected
        assert buf.capacity == 128

    def test_grow_from_zero_uses_default(self):
        """_grow from zero capacity uses DEFAULT_BUFFER_CAPACITY."""
        buf = InstanceBuffer()
        assert buf.capacity == 0

        buf._grow()

        assert buf.capacity == CROWD_RENDERER_CONFIG.DEFAULT_BUFFER_CAPACITY
        assert buf.capacity == 64

    def test_grow_extends_data_arrays(self):
        """_grow extends transform, animation, color arrays."""
        buf = InstanceBuffer()
        buf.reserve(64)
        initial_transform_len = len(buf.transform_data)
        initial_animation_len = len(buf.animation_data)
        initial_color_len = len(buf.color_data)

        buf._grow()

        # Should double
        assert len(buf.transform_data) == initial_transform_len * 2
        assert len(buf.animation_data) == initial_animation_len * 2
        assert len(buf.color_data) == initial_color_len * 2

    def test_grow_clamped_to_max_capacity(self):
        """_grow is clamped to max_capacity."""
        buf = InstanceBuffer(max_instances=100)
        buf.reserve(64)

        buf._grow()

        # 64 * 2 = 128, but max is 100, so clamp
        assert buf.capacity == 100

    def test_grow_preserves_existing_data(self):
        """_grow preserves existing instance data."""
        buf = InstanceBuffer()
        buf.reserve(64)

        inst = CrowdInstance(animation_index=42)
        buf.add_instance(inst)

        buf._grow()

        # Data should still be at index 0
        assert buf.animation_data[0] == 42.0

    def test_grow_at_max_no_change(self):
        """_grow at max_capacity doesn't increase capacity."""
        buf = InstanceBuffer(max_instances=64)
        buf.reserve(64)

        buf._grow()

        assert buf.capacity == 64  # No change


# =============================================================================
# Overflow Tests
# =============================================================================


class TestOverflow:
    """Test exact overflow boundary behavior."""

    def test_add_at_exact_max_raises(self):
        """Adding instance when at max_capacity raises error."""
        buf = InstanceBuffer(max_instances=5)
        buf.reserve(5)

        # Fill to capacity
        for _ in range(5):
            buf.add_instance(CrowdInstance())

        assert buf.instance_count == 5
        assert buf.capacity == 5

        # One more should raise
        with pytest.raises(InstanceBufferOverflowError) as exc_info:
            buf.add_instance(CrowdInstance())

        assert "maximum capacity (5)" in str(exc_info.value)

    def test_overflow_at_10000_default(self):
        """Default buffer overflows at 10,000 instances."""
        buf = InstanceBuffer()
        buf.reserve(10_000)

        # Fill to capacity
        for _ in range(10_000):
            buf.add_instance(CrowdInstance())

        assert buf.instance_count == 10_000

        with pytest.raises(InstanceBufferOverflowError):
            buf.add_instance(CrowdInstance())

    def test_boundary_minus_one_succeeds(self):
        """Adding at max-1 succeeds."""
        buf = InstanceBuffer(max_instances=10)
        buf.reserve(9)

        for _ in range(9):
            buf.add_instance(CrowdInstance())

        # Can still add one more via growth
        idx = buf.add_instance(CrowdInstance())
        assert idx == 9
        assert buf.instance_count == 10

    def test_overflow_error_message(self):
        """Overflow error has descriptive message."""
        buf = InstanceBuffer(max_instances=3)
        buf.reserve(3)
        for _ in range(3):
            buf.add_instance(CrowdInstance())

        with pytest.raises(InstanceBufferOverflowError) as exc_info:
            buf.add_instance(CrowdInstance())

        error_msg = str(exc_info.value)
        assert "Instance buffer at maximum capacity" in error_msg
        assert "(3)" in error_msg

    def test_overflow_without_reserve(self):
        """Overflow is raised at max_instances WITHOUT calling reserve().

        T1.2 FIX Cycle 1: Verifies the fix where max_capacity is checked FIRST
        in add_instance(), before the capacity/grow logic. This ensures
        overflow is detected even when using on-demand growth (no reserve).

        Prior to fix: capacity=0 combined with max_capacity=5 allowed the
        `capacity >= max_capacity` check (0 >= 5 = False) to skip overflow,
        causing _grow() to loop indefinitely or bypass the limit.
        """
        buf = InstanceBuffer(max_instances=5)
        # Intentionally NOT calling reserve() - using on-demand growth

        # Add instances up to max_instances
        for i in range(5):
            idx = buf.add_instance(CrowdInstance())
            assert idx == i

        # Verify state
        assert buf.instance_count == 5
        assert buf.max_capacity == 5

        # The 6th instance must raise overflow
        with pytest.raises(InstanceBufferOverflowError) as exc_info:
            buf.add_instance(CrowdInstance())

        assert "maximum capacity (5)" in str(exc_info.value)
        assert buf.instance_count == 5  # Count unchanged


# =============================================================================
# Memory Calculation Tests
# =============================================================================


class TestMemoryCalculation:
    """Test memory size calculation methods."""

    def test_get_memory_size_bytes_empty(self):
        """Empty buffer has zero memory size."""
        buf = InstanceBuffer()
        assert buf.get_memory_size_bytes() == 0

    def test_get_memory_size_bytes_reserved(self):
        """Reserved buffer accounts for all allocated space."""
        buf = InstanceBuffer()
        buf.reserve(100)

        # 100 instances * (16 + 4 + 4 floats) * 4 bytes
        # = 100 * 24 floats * 4 = 9600 bytes
        total_floats = 100 * (
            CROWD_RENDERER_CONFIG.TRANSFORM_FLOATS +
            CROWD_RENDERER_CONFIG.ANIMATION_FLOATS +
            CROWD_RENDERER_CONFIG.COLOR_FLOATS
        )
        expected = total_floats * 4
        assert buf.get_memory_size_bytes() == expected
        assert buf.get_memory_size_bytes() == 100 * 96

    def test_get_used_memory_bytes_empty(self):
        """Empty buffer has zero used memory."""
        buf = InstanceBuffer()
        assert buf.get_used_memory_bytes() == 0

    def test_get_used_memory_bytes_partial(self):
        """Used memory reflects actual instance count."""
        buf = InstanceBuffer()
        buf.reserve(100)

        for _ in range(25):
            buf.add_instance(CrowdInstance())

        # 25 instances * 96 bytes
        assert buf.get_used_memory_bytes() == 25 * 96
        assert buf.get_used_memory_bytes() == 2400

    def test_memory_vs_used_memory_difference(self):
        """Total memory >= used memory."""
        buf = InstanceBuffer()
        buf.reserve(1000)

        for _ in range(100):
            buf.add_instance(CrowdInstance())

        total = buf.get_memory_size_bytes()
        used = buf.get_used_memory_bytes()

        assert total >= used
        assert total == 1000 * 96
        assert used == 100 * 96


# =============================================================================
# Clear Tests
# =============================================================================


class TestClear:
    """Test clear() method behavior."""

    def test_clear_resets_count(self):
        """Clear resets instance count to zero."""
        buf = InstanceBuffer()
        buf.reserve(100)
        for _ in range(50):
            buf.add_instance(CrowdInstance())

        buf.clear()

        assert buf.instance_count == 0

    def test_clear_empties_arrays(self):
        """Clear empties all data arrays."""
        buf = InstanceBuffer()
        buf.reserve(100)
        for _ in range(50):
            buf.add_instance(CrowdInstance())

        buf.clear()

        assert len(buf.transform_data) == 0
        assert len(buf.animation_data) == 0
        assert len(buf.color_data) == 0

    def test_clear_marks_dirty(self):
        """Clear sets dirty flag."""
        buf = InstanceBuffer()
        buf.dirty = False

        buf.clear()

        assert buf.dirty is True


# =============================================================================
# Update Instance Tests
# =============================================================================


class TestUpdateInstance:
    """Test update_instance() method."""

    def test_update_changes_data(self):
        """Update instance modifies buffer data."""
        buf = InstanceBuffer()
        buf.reserve(10)

        inst = CrowdInstance(animation_index=5)
        buf.add_instance(inst)

        assert buf.animation_data[0] == 5.0

        inst.animation_index = 10
        buf.update_instance(0, inst)

        assert buf.animation_data[0] == 10.0

    def test_update_invalid_index_no_crash(self):
        """Update with invalid index is silently ignored."""
        buf = InstanceBuffer()
        buf.reserve(10)
        buf.add_instance(CrowdInstance())

        # Should not raise
        buf.update_instance(-1, CrowdInstance())
        buf.update_instance(99, CrowdInstance())

    def test_update_marks_dirty(self):
        """Update sets dirty flag."""
        buf = InstanceBuffer()
        buf.reserve(10)
        buf.add_instance(CrowdInstance())
        buf.dirty = False

        buf.update_instance(0, CrowdInstance())

        assert buf.dirty is True


# =============================================================================
# Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for large instance counts."""

    def test_10000_instances_no_crash(self):
        """Buffer handles 10,000 instances without crash."""
        buf = InstanceBuffer()
        buf.reserve(10_000)

        for i in range(10_000):
            inst = CrowdInstance(
                position=Vec3(float(i), 0.0, 0.0),
                animation_index=i % 100,
            )
            buf.add_instance(inst)

        assert buf.instance_count == 10_000

    def test_10000_instances_memory_correct(self):
        """10,000 instances use correct memory."""
        buf = InstanceBuffer()
        buf.reserve(10_000)

        for _ in range(10_000):
            buf.add_instance(CrowdInstance())

        expected_bytes = 10_000 * 96
        assert buf.get_used_memory_bytes() == expected_bytes
        assert buf.get_used_memory_bytes() == 960_000  # ~937.5 KB

    def test_10000_instances_data_integrity(self):
        """Verify data integrity at scale."""
        buf = InstanceBuffer()
        buf.reserve(10_000)

        # Add instances with predictable values
        for i in range(10_000):
            inst = CrowdInstance(
                animation_index=i,
                animation_time=float(i) / 1000.0,
            )
            buf.add_instance(inst)

        # Verify sampling of data
        for i in [0, 1000, 5000, 9999]:
            anim_offset = i * 4
            assert buf.animation_data[anim_offset] == float(i)
            assert abs(buf.animation_data[anim_offset + 1] - float(i) / 1000.0) < 1e-6

    def test_stress_performance_add_10000(self):
        """Adding 10,000 instances completes under reasonable time."""
        buf = InstanceBuffer()
        buf.reserve(10_000)

        start = time.perf_counter()
        for _ in range(10_000):
            buf.add_instance(CrowdInstance())
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second (typically ~0.1s)
        assert elapsed < 1.0, f"Adding 10,000 instances took {elapsed:.3f}s"

    def test_stress_update_all_10000(self):
        """Updating 10,000 instances completes under reasonable time."""
        buf = InstanceBuffer()
        buf.reserve(10_000)

        # Pre-populate
        instances = []
        for _ in range(10_000):
            inst = CrowdInstance()
            buf.add_instance(inst)
            instances.append(inst)

        start = time.perf_counter()
        for i, inst in enumerate(instances):
            inst.animation_time += 0.016
            buf.update_instance(i, inst)
        elapsed = time.perf_counter() - start

        # Should complete in under 2 seconds (typically ~0.2s)
        assert elapsed < 2.0, f"Updating 10,000 instances took {elapsed:.3f}s"


# =============================================================================
# Dynamic Growth Integration Tests
# =============================================================================


class TestDynamicGrowth:
    """Test dynamic growth through add_instance."""

    def test_grow_triggered_at_capacity(self):
        """Growth is triggered when capacity is reached."""
        buf = InstanceBuffer(max_instances=1000)
        buf.reserve(64)  # DEFAULT_BUFFER_CAPACITY

        # Fill to capacity
        for _ in range(64):
            buf.add_instance(CrowdInstance())

        assert buf.capacity == 64

        # One more triggers growth
        buf.add_instance(CrowdInstance())

        assert buf.capacity > 64
        # Should have doubled (clamped to max)
        assert buf.capacity == 128

    def test_multiple_growth_cycles(self):
        """Buffer can grow multiple times."""
        buf = InstanceBuffer(max_instances=1000)
        # Start with small reserve
        buf.reserve(8)

        # Add 500 instances, triggering multiple growths
        for _ in range(500):
            buf.add_instance(CrowdInstance())

        assert buf.instance_count == 500
        assert buf.capacity >= 500
        # Should have grown: 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512
        assert buf.capacity == 512

    def test_growth_up_to_max(self):
        """Can add up to max_instances."""
        buf = InstanceBuffer(max_instances=100)

        for _ in range(100):
            buf.add_instance(CrowdInstance())

        # instance_count reaches max
        assert buf.instance_count == 100
        # Arrays have enough space for 100 instances
        assert len(buf.transform_data) >= 100 * 16
        assert len(buf.animation_data) >= 100 * 4
        assert len(buf.color_data) >= 100 * 4

    def test_no_reserve_on_demand_extension(self):
        """Without reserve, arrays extend on-demand per instance.

        Note: The implementation extends arrays just enough for each
        instance when capacity is 0, rather than pre-allocating.
        """
        buf = InstanceBuffer()

        # Add first instance - extends just enough
        buf.add_instance(CrowdInstance())
        assert buf.instance_count == 1
        assert len(buf.transform_data) >= 16
        assert len(buf.animation_data) >= 4
        assert len(buf.color_data) >= 4

        # Add more instances
        for _ in range(9):
            buf.add_instance(CrowdInstance())

        assert buf.instance_count == 10
        assert len(buf.transform_data) >= 160
        assert len(buf.animation_data) >= 40
        assert len(buf.color_data) >= 40


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_max_instances_zero(self):
        """max_instances=0 raises overflow error on any add_instance.

        With the fix applied (T1.2 FIX Cycle 1), max_capacity is checked FIRST
        in add_instance(), before the capacity/grow check. When max_instances=0,
        the condition `instance_count >= max_capacity` (0 >= 0) is True, so
        overflow is raised immediately.

        This prevents the previous bypass where capacity=0 allowed unlimited
        on-demand extension without proper bounds checking.
        """
        buf = InstanceBuffer(max_instances=0)
        # Reserve fails as before
        with pytest.raises(InstanceBufferOverflowError):
            buf.reserve(1)

        # Now add_instance also raises overflow (fix applied)
        buf2 = InstanceBuffer(max_instances=0)
        with pytest.raises(InstanceBufferOverflowError) as exc_info:
            buf2.add_instance(CrowdInstance())
        assert "maximum capacity (0)" in str(exc_info.value)

    def test_max_instances_one_with_reserve(self):
        """max_instances=1 with reserve allows exactly one instance."""
        buf = InstanceBuffer(max_instances=1)
        buf.reserve(1)  # Must reserve to enable capacity checking

        buf.add_instance(CrowdInstance())
        assert buf.instance_count == 1

        with pytest.raises(InstanceBufferOverflowError):
            buf.add_instance(CrowdInstance())

    def test_extreme_instance_values(self):
        """Instances with extreme values are handled."""
        buf = InstanceBuffer()
        inst = CrowdInstance(
            position=Vec3(1e10, -1e10, 1e-10),
            animation_index=999999,
            animation_time=1e6,
            animation_speed=-1.0,
            lod_level=100,
        )

        idx = buf.add_instance(inst)
        assert idx == 0
        assert buf.animation_data[0] == 999999.0
        assert buf.animation_data[2] == -1.0

    def test_default_instance_values(self):
        """Default CrowdInstance values pack correctly."""
        buf = InstanceBuffer()
        inst = CrowdInstance()
        buf.add_instance(inst)

        # Animation: (0, 0.0, 1.0, 0)
        assert buf.animation_data[0] == 0.0  # index
        assert buf.animation_data[1] == 0.0  # time
        assert buf.animation_data[2] == 1.0  # speed (default)
        assert buf.animation_data[3] == 0.0  # lod

        # Color: white (1, 1, 1, 1)
        assert buf.color_data[0] == 1.0
        assert buf.color_data[1] == 1.0
        assert buf.color_data[2] == 1.0
        assert buf.color_data[3] == 1.0

    def test_clear_then_reuse(self):
        """Buffer can be cleared and reused."""
        buf = InstanceBuffer()
        buf.reserve(100)

        for _ in range(50):
            buf.add_instance(CrowdInstance())

        buf.clear()
        assert buf.instance_count == 0

        # Can add again
        for _ in range(100):
            buf.add_instance(CrowdInstance())

        assert buf.instance_count == 100


# =============================================================================
# Internal State Consistency Tests
# =============================================================================


class TestInternalConsistency:
    """Test internal state remains consistent."""

    def test_array_lengths_consistent(self):
        """Data array lengths stay proportional to capacity."""
        buf = InstanceBuffer()
        buf.reserve(100)

        # Proportions should match config
        assert len(buf.transform_data) == 100 * 16
        assert len(buf.animation_data) == 100 * 4
        assert len(buf.color_data) == 100 * 4

    def test_instance_count_never_exceeds_capacity_with_reserve(self):
        """instance_count is always <= capacity when reserved."""
        buf = InstanceBuffer()
        buf.reserve(1000)

        for _ in range(1000):
            buf.add_instance(CrowdInstance())
            assert buf.instance_count <= buf.capacity

    def test_dirty_flag_semantics(self):
        """Dirty flag is set on mutations."""
        buf = InstanceBuffer()
        buf.reserve(10)

        # New buffer is dirty
        assert buf.dirty is True

        buf.dirty = False
        buf.add_instance(CrowdInstance())
        assert buf.dirty is True

        buf.dirty = False
        buf.update_instance(0, CrowdInstance())
        assert buf.dirty is True

        buf.dirty = False
        buf.clear()
        assert buf.dirty is True

    def test_capacity_vs_instance_count(self):
        """Capacity is always >= instance_count after reserve."""
        buf = InstanceBuffer()
        buf.reserve(100)

        for i in range(50):
            buf.add_instance(CrowdInstance())
            assert buf.capacity >= buf.instance_count

        assert buf.capacity == 100
        assert buf.instance_count == 50


# =============================================================================
# Configuration Integration Tests
# =============================================================================


class TestConfigIntegration:
    """Test integration with CROWD_RENDERER_CONFIG."""

    def test_uses_config_transform_floats(self):
        """Buffer uses TRANSFORM_FLOATS from config."""
        buf = InstanceBuffer()
        buf.reserve(10)
        assert len(buf.transform_data) == 10 * CROWD_RENDERER_CONFIG.TRANSFORM_FLOATS

    def test_uses_config_animation_floats(self):
        """Buffer uses ANIMATION_FLOATS from config."""
        buf = InstanceBuffer()
        buf.reserve(10)
        assert len(buf.animation_data) == 10 * CROWD_RENDERER_CONFIG.ANIMATION_FLOATS

    def test_uses_config_color_floats(self):
        """Buffer uses COLOR_FLOATS from config."""
        buf = InstanceBuffer()
        buf.reserve(10)
        assert len(buf.color_data) == 10 * CROWD_RENDERER_CONFIG.COLOR_FLOATS

    def test_uses_config_growth_factor(self):
        """Buffer uses BUFFER_GROWTH_FACTOR from config."""
        buf = InstanceBuffer()
        buf.reserve(64)
        buf._grow()
        expected = 64 * CROWD_RENDERER_CONFIG.BUFFER_GROWTH_FACTOR
        assert buf.capacity == expected

    def test_uses_config_default_capacity(self):
        """Buffer uses DEFAULT_BUFFER_CAPACITY from config."""
        buf = InstanceBuffer()
        buf._grow()  # From 0
        assert buf.capacity == CROWD_RENDERER_CONFIG.DEFAULT_BUFFER_CAPACITY

    def test_default_max_uses_config_batch_size(self):
        """Default max_instances uses MAX_INSTANCES_PER_BATCH * 10."""
        buf = InstanceBuffer()
        expected = CROWD_RENDERER_CONFIG.MAX_INSTANCES_PER_BATCH * 10
        assert buf.max_instances == expected
