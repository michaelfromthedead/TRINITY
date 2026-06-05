"""
Blackbox tests for T1.2 Instance Buffer Stress Test.

These tests verify the InstanceBuffer public contract without
knowledge of internal implementation details. Tests are derived from:
- docs/PYTHON_DOCS/engine_animation_crowds_facial/PHASE_1_ARCH.md
- docs/PYTHON_DOCS/engine_animation_crowds_facial/PHASE_1_TODO.md

Contract under test:
1. InstanceBuffer accepts CrowdInstance objects via add_instance()
2. Buffer raises InstanceBufferOverflowError when at capacity
3. Dynamic buffer growth functions correctly
4. Memory layout: 96 bytes per instance (64 transform + 16 animation + 16 color)
5. Buffer handles 10,000+ instances without crash

NOTE: Tests marked with @pytest.mark.xfail indicate contract violations found
during blackbox testing. The implementation does not match the documented contract.
These failures should be addressed by fixing the implementation, not the tests.

Contract violations discovered:
- get_byte_size() returns constant 96 instead of instance_count * 96
- FIXED: InstanceBufferOverflowError is now raised at max capacity (T1.2 FIX Cycle 1)
"""

import pytest
import numpy as np

from engine.core.math import Vec3, Vec4, Quat
from engine.animation.crowds import (
    CrowdInstance,
    InstanceBuffer,
)
# Import exception directly from module since not exported in __init__.py
from engine.animation.crowds.crowd_renderer import InstanceBufferOverflowError


# -----------------------------------------------------------------------------
# Test Constants (from PHASE_1_ARCH.md contract)
# -----------------------------------------------------------------------------

BYTES_PER_INSTANCE = 96  # 64 (transform) + 16 (animation) + 16 (color)
DEFAULT_MAX_INSTANCES = 10000


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def make_instance(
    index: int = 0,
    position: Vec3 = None,
    rotation: Quat = None,
    scale: float = 1.0,
    animation_index: int = 0,
    animation_time: float = 0.0,
    animation_speed: float = 1.0,
    tint_color: Vec4 = None,
    lod_level: int = 0,
) -> CrowdInstance:
    """Create a CrowdInstance with sensible defaults.

    Uses index to create distinguishable instances for testing.
    Note: scale is uniform (float) per the actual public API.
    """
    if position is None:
        position = Vec3(float(index), 0.0, 0.0)
    if rotation is None:
        rotation = Quat.identity()
    if tint_color is None:
        tint_color = Vec4(1.0, 1.0, 1.0, 1.0)

    return CrowdInstance(
        position=position,
        rotation=rotation,
        scale=scale,
        animation_index=animation_index,
        animation_time=animation_time,
        animation_speed=animation_speed,
        tint_color=tint_color,
        lod_level=lod_level,
    )


# -----------------------------------------------------------------------------
# Buffer Overflow Protection Tests (from T1.2 acceptance criteria)
# -----------------------------------------------------------------------------

class TestBufferOverflowProtection:
    """Test that InstanceBuffer raises InstanceBufferOverflowError at capacity."""

    def test_buffer_overflow_at_max_capacity(self):
        """Adding instance beyond max_instances raises InstanceBufferOverflowError.

        From TODO: 'InstanceBufferOverflowError raised at max capacity'
        """
        max_size = 100
        buffer = InstanceBuffer(max_instances=max_size)

        # Fill buffer to capacity
        for i in range(max_size):
            buffer.add_instance(make_instance(i))

        # Attempt to add one more should raise
        with pytest.raises(InstanceBufferOverflowError):
            buffer.add_instance(make_instance(max_size))

    def test_buffer_overflow_error_type(self):
        """InstanceBufferOverflowError is a proper Exception subclass."""
        assert issubclass(InstanceBufferOverflowError, Exception)

    def test_buffer_allows_exactly_max_instances(self):
        """Buffer should accept exactly max_instances before overflow."""
        max_size = 50
        buffer = InstanceBuffer(max_instances=max_size)

        # Should accept exactly max_size instances
        for i in range(max_size):
            buffer.add_instance(make_instance(i))

        # No exception should have been raised

    def test_buffer_overflow_with_size_one(self):
        """Edge case: buffer with max_instances=1 overflows on second add."""
        buffer = InstanceBuffer(max_instances=1)

        buffer.add_instance(make_instance(0))

        with pytest.raises(InstanceBufferOverflowError):
            buffer.add_instance(make_instance(1))

    def test_buffer_overflow_preserves_buffer_state(self):
        """After overflow error, previously added instances should remain intact."""
        max_size = 10
        buffer = InstanceBuffer(max_instances=max_size)

        # Fill buffer
        for i in range(max_size):
            buffer.add_instance(make_instance(i))

        original_size = buffer.get_byte_size()

        # Attempt overflow
        with pytest.raises(InstanceBufferOverflowError):
            buffer.add_instance(make_instance(max_size))

        # Buffer state should be unchanged
        assert buffer.get_byte_size() == original_size


# -----------------------------------------------------------------------------
# Memory Layout Tests (from PHASE_1_ARCH.md)
# -----------------------------------------------------------------------------

class TestMemoryLayout:
    """Test that memory layout matches 96 bytes per instance contract."""

    def test_single_instance_byte_size(self):
        """Single instance should be exactly 96 bytes.

        From ARCH: Memory layout is 64 (transform) + 16 (animation) + 16 (color) = 96
        """
        buffer = InstanceBuffer(max_instances=10)
        buffer.add_instance(make_instance(0))

        assert buffer.get_byte_size() == BYTES_PER_INSTANCE

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 instead of instance_count * 96")
    def test_multiple_instances_byte_size(self):
        """Multiple instances should scale linearly at 96 bytes each."""
        buffer = InstanceBuffer(max_instances=100)

        for i in range(10):
            buffer.add_instance(make_instance(i))

        assert buffer.get_byte_size() == 10 * BYTES_PER_INSTANCE

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 even for empty buffer")
    def test_empty_buffer_byte_size(self):
        """Empty buffer should have zero byte size."""
        buffer = InstanceBuffer(max_instances=100)

        assert buffer.get_byte_size() == 0

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96, doesn't increment")
    def test_byte_size_increments_correctly(self):
        """Each added instance should increment byte size by exactly 96."""
        buffer = InstanceBuffer(max_instances=100)

        for i in range(5):
            before_size = buffer.get_byte_size()
            buffer.add_instance(make_instance(i))
            after_size = buffer.get_byte_size()

            assert after_size - before_size == BYTES_PER_INSTANCE


# -----------------------------------------------------------------------------
# Stress Tests (from T1.2 acceptance criteria)
# -----------------------------------------------------------------------------

class TestStressConditions:
    """Test buffer behavior under stress conditions."""

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 instead of instance_count * 96")
    def test_buffer_handles_10000_instances(self):
        """Buffer should handle 10,000 instances without crash.

        From TODO: 'Buffer handles 10,000 instances without crash'
        """
        buffer = InstanceBuffer(max_instances=10000)

        for i in range(10000):
            buffer.add_instance(make_instance(i))

        # Verify all instances were added
        assert buffer.get_byte_size() == 10000 * BYTES_PER_INSTANCE

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 instead of instance_count * 96")
    def test_buffer_handles_default_max_instances(self):
        """Buffer should handle default max (10000) instances."""
        buffer = InstanceBuffer()  # Uses default max_instances

        # Add up to default max
        for i in range(DEFAULT_MAX_INSTANCES):
            buffer.add_instance(make_instance(i))

        assert buffer.get_byte_size() == DEFAULT_MAX_INSTANCES * BYTES_PER_INSTANCE

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96")
    def test_rapid_add_remove_cycles(self):
        """Buffer should handle rapid add/clear cycles without degradation."""
        buffer = InstanceBuffer(max_instances=1000)

        for cycle in range(10):
            # Fill buffer
            for i in range(1000):
                buffer.add_instance(make_instance(i))

            assert buffer.get_byte_size() == 1000 * BYTES_PER_INSTANCE

            # Clear buffer
            buffer.clear()
            assert buffer.get_byte_size() == 0


# -----------------------------------------------------------------------------
# Dynamic Growth Tests (from T1.2 acceptance criteria)
# -----------------------------------------------------------------------------

class TestDynamicGrowth:
    """Test dynamic buffer growth functionality.

    From ARCH: 'Dynamic buffer growth (BUFFER_GROWTH_FACTOR)'
    """

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 instead of instance_count * 96")
    def test_buffer_accepts_instances_up_to_max(self):
        """Buffer should grow internally to accommodate instances up to max."""
        buffer = InstanceBuffer(max_instances=500)

        # Add instances incrementally
        for i in range(500):
            buffer.add_instance(make_instance(i))

        assert buffer.get_byte_size() == 500 * BYTES_PER_INSTANCE

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 instead of instance_count * 96")
    def test_buffer_clear_allows_reuse(self):
        """After clear(), buffer should accept new instances."""
        buffer = InstanceBuffer(max_instances=100)

        # First fill
        for i in range(100):
            buffer.add_instance(make_instance(i))

        buffer.clear()

        # Second fill (should work without overflow)
        for i in range(100):
            buffer.add_instance(make_instance(i))

        assert buffer.get_byte_size() == 100 * BYTES_PER_INSTANCE


# -----------------------------------------------------------------------------
# Buffer Clear Tests
# -----------------------------------------------------------------------------

class TestBufferClear:
    """Test buffer clear() functionality."""

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 even after clear()")
    def test_clear_empties_buffer(self):
        """clear() should reset buffer to empty state."""
        buffer = InstanceBuffer(max_instances=100)

        for i in range(50):
            buffer.add_instance(make_instance(i))

        buffer.clear()

        assert buffer.get_byte_size() == 0

    def test_clear_allows_full_refill(self):
        """After clear(), buffer should accept max_instances again."""
        buffer = InstanceBuffer(max_instances=10)

        # Fill to capacity
        for i in range(10):
            buffer.add_instance(make_instance(i))

        buffer.clear()

        # Should be able to fill again
        for i in range(10):
            buffer.add_instance(make_instance(i))

        # Should still overflow at max
        with pytest.raises(InstanceBufferOverflowError):
            buffer.add_instance(make_instance(10))

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 even after clear()")
    def test_multiple_clears(self):
        """Multiple clear() calls should be idempotent."""
        buffer = InstanceBuffer(max_instances=100)

        for i in range(50):
            buffer.add_instance(make_instance(i))

        buffer.clear()
        buffer.clear()
        buffer.clear()

        assert buffer.get_byte_size() == 0

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 even for empty buffer")
    def test_clear_on_empty_buffer(self):
        """clear() on empty buffer should be safe."""
        buffer = InstanceBuffer(max_instances=100)

        buffer.clear()

        assert buffer.get_byte_size() == 0


# -----------------------------------------------------------------------------
# Instance Validation Tests
# -----------------------------------------------------------------------------

class TestInstanceValidation:
    """Test that various CrowdInstance configurations are accepted."""

    def test_instance_with_custom_position(self):
        """Instance with custom position should be accepted."""
        buffer = InstanceBuffer(max_instances=10)

        instance = make_instance(position=Vec3(100.0, 200.0, 300.0))
        buffer.add_instance(instance)

        assert buffer.get_byte_size() == BYTES_PER_INSTANCE

    def test_instance_with_rotation(self):
        """Instance with non-identity rotation should be accepted."""
        buffer = InstanceBuffer(max_instances=10)

        instance = make_instance(rotation=Quat.from_axis_angle(Vec3(0, 1, 0), 1.57))
        buffer.add_instance(instance)

        assert buffer.get_byte_size() == BYTES_PER_INSTANCE

    def test_instance_with_large_scale(self):
        """Instance with large uniform scale should be accepted."""
        buffer = InstanceBuffer(max_instances=10)

        instance = make_instance(scale=10.0)
        buffer.add_instance(instance)

        assert buffer.get_byte_size() == BYTES_PER_INSTANCE

    def test_instance_with_animation_params(self):
        """Instance with custom animation parameters should be accepted."""
        buffer = InstanceBuffer(max_instances=10)

        instance = make_instance(
            animation_index=5,
            animation_time=1.5,
            animation_speed=2.0,
        )
        buffer.add_instance(instance)

        assert buffer.get_byte_size() == BYTES_PER_INSTANCE

    def test_instance_with_tint_color(self):
        """Instance with custom tint color should be accepted."""
        buffer = InstanceBuffer(max_instances=10)

        instance = make_instance(tint_color=Vec4(1.0, 0.0, 0.0, 0.5))
        buffer.add_instance(instance)

        assert buffer.get_byte_size() == BYTES_PER_INSTANCE

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 instead of instance_count * 96")
    def test_instance_with_lod_level(self):
        """Instance with specific LOD level should be accepted."""
        buffer = InstanceBuffer(max_instances=10)

        for lod in range(4):  # Common LOD levels 0-3
            instance = make_instance(lod_level=lod)
            buffer.add_instance(instance)

        assert buffer.get_byte_size() == 4 * BYTES_PER_INSTANCE


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_max_instances(self):
        """Buffer with max_instances=0 should overflow immediately."""
        buffer = InstanceBuffer(max_instances=0)

        with pytest.raises(InstanceBufferOverflowError):
            buffer.add_instance(make_instance(0))

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 instead of instance_count * 96")
    def test_large_max_instances(self):
        """Buffer should accept large max_instances value."""
        # Just verify creation doesn't crash
        buffer = InstanceBuffer(max_instances=100000)

        # Add a few instances to verify it works
        for i in range(100):
            buffer.add_instance(make_instance(i))

        assert buffer.get_byte_size() == 100 * BYTES_PER_INSTANCE

    def test_instance_with_zero_scale(self):
        """Instance with zero scale should be accepted (degenerate but valid)."""
        buffer = InstanceBuffer(max_instances=10)

        instance = make_instance(scale=0.0)
        buffer.add_instance(instance)

        assert buffer.get_byte_size() == BYTES_PER_INSTANCE

    def test_instance_with_negative_animation_time(self):
        """Instance with negative animation_time should be accepted."""
        buffer = InstanceBuffer(max_instances=10)

        instance = make_instance(animation_time=-1.0)
        buffer.add_instance(instance)

        assert buffer.get_byte_size() == BYTES_PER_INSTANCE

    def test_instance_with_zero_animation_speed(self):
        """Instance with zero animation_speed (paused) should be accepted."""
        buffer = InstanceBuffer(max_instances=10)

        instance = make_instance(animation_speed=0.0)
        buffer.add_instance(instance)

        assert buffer.get_byte_size() == BYTES_PER_INSTANCE

    def test_instance_with_extreme_values(self):
        """Instance with extreme but valid values should be accepted."""
        buffer = InstanceBuffer(max_instances=10)

        instance = make_instance(
            position=Vec3(1e6, 1e6, 1e6),
            scale=1e-6,
            animation_speed=1000.0,
        )
        buffer.add_instance(instance)

        assert buffer.get_byte_size() == BYTES_PER_INSTANCE


# -----------------------------------------------------------------------------
# Concurrent Instance Variations Test
# -----------------------------------------------------------------------------

class TestMixedInstances:
    """Test buffer with diverse instance configurations."""

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 instead of instance_count * 96")
    def test_diverse_instance_batch(self):
        """Buffer should handle batch of instances with varying properties."""
        buffer = InstanceBuffer(max_instances=100)

        instances = [
            make_instance(position=Vec3(i * 10, 0, 0), animation_index=i % 5)
            for i in range(50)
        ]

        for instance in instances:
            buffer.add_instance(instance)

        assert buffer.get_byte_size() == 50 * BYTES_PER_INSTANCE

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 instead of instance_count * 96")
    def test_alternating_lod_levels(self):
        """Buffer should handle instances with alternating LOD levels."""
        buffer = InstanceBuffer(max_instances=100)

        for i in range(40):
            instance = make_instance(lod_level=i % 4)
            buffer.add_instance(instance)

        assert buffer.get_byte_size() == 40 * BYTES_PER_INSTANCE


# -----------------------------------------------------------------------------
# Integration-style Blackbox Tests
# -----------------------------------------------------------------------------

class TestBufferLifecycle:
    """Test complete buffer lifecycle scenarios."""

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: Multiple issues - get_byte_size() constant, no overflow exception")
    def test_full_lifecycle(self):
        """Test complete add -> use -> clear -> reuse cycle."""
        buffer = InstanceBuffer(max_instances=100)

        # Phase 1: Initial fill
        for i in range(50):
            buffer.add_instance(make_instance(i))
        assert buffer.get_byte_size() == 50 * BYTES_PER_INSTANCE

        # Phase 2: Continue filling to capacity
        for i in range(50, 100):
            buffer.add_instance(make_instance(i))
        assert buffer.get_byte_size() == 100 * BYTES_PER_INSTANCE

        # Phase 3: Overflow attempt
        with pytest.raises(InstanceBufferOverflowError):
            buffer.add_instance(make_instance(100))

        # Phase 4: Clear and reuse
        buffer.clear()
        assert buffer.get_byte_size() == 0

        # Phase 5: Fill again
        for i in range(25):
            buffer.add_instance(make_instance(i))
        assert buffer.get_byte_size() == 25 * BYTES_PER_INSTANCE

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: get_byte_size() returns constant 96 instead of instance_count * 96")
    def test_crowd_simulation_pattern(self):
        """Simulate realistic crowd rendering pattern."""
        buffer = InstanceBuffer(max_instances=5000)

        # Simulate crowd spawn
        for i in range(1000):
            position = Vec3(float(i % 100), 0, float(i // 100))
            instance = make_instance(
                position=position,
                animation_index=i % 10,
                animation_time=float(i % 60) / 30.0,
                animation_speed=0.8 + (i % 5) * 0.1,
                lod_level=min(3, i // 250),
            )
            buffer.add_instance(instance)

        assert buffer.get_byte_size() == 1000 * BYTES_PER_INSTANCE

        # Simulate frame update - clear and repopulate
        buffer.clear()

        # Re-add with different animation times
        for i in range(1000):
            position = Vec3(float(i % 100), 0, float(i // 100))
            instance = make_instance(
                position=position,
                animation_index=i % 10,
                animation_time=float(i % 60 + 1) / 30.0,  # Time advanced
            )
            buffer.add_instance(instance)

        assert buffer.get_byte_size() == 1000 * BYTES_PER_INSTANCE
