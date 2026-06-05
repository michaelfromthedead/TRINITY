"""Tests for suspension physics.

Tests cover:
- SuspensionState and SuspensionGeometry dataclasses
- Suspension spring force calculations
- Damper force calculations (compression/rebound)
- Bump stop forces at travel limits
- Progressive spring rates
- AntiRollBar torque transfer
- SuspensionSystem axle management
"""

import math
import pytest

from engine.simulation.vehicles.suspension import (
    SuspensionType,
    SuspensionState,
    SuspensionGeometry,
    Suspension,
    AntiRollBar,
    SuspensionSystem,
)
from engine.simulation.vehicles.vehicle_system import Vector3


# =============================================================================
# SuspensionState Tests
# =============================================================================


class TestSuspensionState:
    """Tests for SuspensionState dataclass."""

    def test_default_values(self):
        """SuspensionState should have sensible defaults."""
        state = SuspensionState()
        assert state.compression == 0.0
        assert state.velocity == 0.0
        assert state.force == 0.0
        assert not state.is_grounded

    def test_custom_values(self):
        """SuspensionState should accept custom values."""
        state = SuspensionState(
            compression=0.05,
            velocity=-0.5,
            force=5000.0,
            is_grounded=True,
        )
        assert state.compression == 0.05
        assert state.velocity == -0.5
        assert state.force == 5000.0
        assert state.is_grounded


# =============================================================================
# SuspensionGeometry Tests
# =============================================================================


class TestSuspensionGeometry:
    """Tests for SuspensionGeometry dataclass."""

    def test_default_values(self):
        """SuspensionGeometry should have sensible defaults."""
        geom = SuspensionGeometry()
        assert geom.camber == 0.0
        assert geom.caster == 0.0
        assert geom.toe == 0.0

    def test_camber_gain(self):
        """Camber gain should change with compression."""
        geom = SuspensionGeometry()
        gain_zero = geom.get_camber_gain(0.0)
        gain_compressed = geom.get_camber_gain(0.05)  # 5cm compression

        assert gain_zero == 0.0
        # Typically negative camber gain (wheel leans in under compression)
        assert gain_compressed != 0.0

    def test_toe_change(self):
        """Toe change (bump steer) should change with compression."""
        geom = SuspensionGeometry()
        toe_zero = geom.get_toe_change(0.0)
        toe_compressed = geom.get_toe_change(0.05)

        assert toe_zero == 0.0
        assert toe_compressed != 0.0


# =============================================================================
# Suspension Tests
# =============================================================================


class TestSuspension:
    """Tests for Suspension physics simulation."""

    @pytest.fixture
    def suspension(self):
        """Create a standard suspension."""
        return Suspension(
            rest_length=0.5,
            spring_strength=35000.0,
            damper_compression=4500.0,
            damper_rebound=4000.0,
            travel=0.2,
        )

    def test_initialization(self, suspension):
        """Suspension should initialize with correct values."""
        assert suspension.rest_length == 0.5
        assert suspension.spring_strength == 35000.0
        assert suspension.min_length == 0.4  # 0.5 - 0.2/2
        assert suspension.max_length == 0.6  # 0.5 + 0.2/2

    def test_spring_strength_setter(self, suspension):
        """Spring strength setter should accept positive values."""
        suspension.spring_strength = 40000.0
        assert suspension.spring_strength == 40000.0

    def test_spring_strength_rejects_negative(self, suspension):
        """Spring strength setter should reject negative values."""
        with pytest.raises(ValueError, match="non-negative"):
            suspension.spring_strength = -1000

    def test_damper_setters_positive(self, suspension):
        """Damper setters should accept positive values."""
        suspension.damper_compression = 5000.0
        suspension.damper_rebound = 4500.0
        assert suspension.damper_compression == 5000.0
        assert suspension.damper_rebound == 4500.0

    def test_damper_setters_reject_negative(self, suspension):
        """Damper setters should reject negative values."""
        with pytest.raises(ValueError, match="non-negative"):
            suspension.damper_compression = -100
        with pytest.raises(ValueError, match="non-negative"):
            suspension.damper_rebound = -100

    def test_spring_force_at_rest(self, suspension):
        """At rest length, spring force should be zero."""
        force = suspension.spring_force(0.0)  # No compression
        assert force == 0.0

    def test_spring_force_compressed(self, suspension):
        """Compressed suspension should push back."""
        force = suspension.spring_force(0.05)  # 5cm compression
        expected = 35000.0 * 0.05
        assert force == expected

    def test_spring_force_extended(self, suspension):
        """Extended suspension should pull back."""
        force = suspension.spring_force(-0.05)  # 5cm extension
        expected = 35000.0 * -0.05
        assert force == expected

    def test_damper_force_compression(self, suspension):
        """Compressing suspension should create damping force."""
        force = suspension.damper_force(1.0)  # Compressing at 1 m/s
        expected = 4500.0 * 1.0  # Uses compression damping
        assert force == expected

    def test_damper_force_rebound(self, suspension):
        """Extending suspension should create rebound damping."""
        force = suspension.damper_force(-1.0)  # Extending at 1 m/s
        expected = 4000.0 * -1.0  # Uses rebound damping
        assert force == expected

    def test_bump_stop_not_engaged(self, suspension):
        """Bump stop should not engage in normal travel."""
        force = suspension.bump_stop_force(0.5)  # At rest
        assert force == 0.0

    def test_bump_stop_compression(self, suspension):
        """Bump stop should engage near minimum length."""
        # Just past bump stop threshold
        force = suspension.bump_stop_force(0.41)  # Near min (0.4)
        assert force > 0  # Pushes back

    def test_bump_stop_extension(self, suspension):
        """Bump stop should engage near maximum length (droop)."""
        force = suspension.bump_stop_force(0.59)  # Near max (0.6)
        assert force < 0  # Pulls back

    def test_update_grounded(self, suspension):
        """Update should calculate forces for grounded wheel."""
        force = suspension.update(length=0.45, dt=0.016)  # Compressed
        assert force > 0  # Spring pushes back
        assert suspension.state.is_grounded
        assert abs(suspension.state.compression - 0.05) < 0.001  # rest - current

    def test_update_airborne(self, suspension):
        """Update should handle airborne wheel."""
        force = suspension.update(length=0.7, dt=0.016)  # Beyond max
        assert not suspension.state.is_grounded

    def test_compression_ratio(self, suspension):
        """Compression ratio should reflect position in travel."""
        suspension.update(length=0.5, dt=0.016)  # At rest
        ratio_rest = suspension.compression_ratio
        assert 0.4 < ratio_rest < 0.6  # Around 0.5

        suspension.update(length=0.4, dt=0.016)  # Fully compressed
        ratio_compressed = suspension.compression_ratio
        assert ratio_compressed > 0.9  # Near 1.0

        suspension.update(length=0.6, dt=0.016)  # Fully extended
        ratio_extended = suspension.compression_ratio
        assert ratio_extended < 0.1  # Near 0.0

    def test_reset(self, suspension):
        """Reset should return to initial state."""
        suspension.update(length=0.45, dt=0.016)
        suspension.reset()
        assert suspension.compression == 0.0
        assert suspension.velocity == 0.0

    def test_progressive_spring_rate(self, suspension):
        """Progressive rate should increase force at higher compression."""
        suspension.set_progressive_rate(50000.0)  # Progressive rate

        force_small = suspension.spring_force(0.02)  # Small compression
        force_large = suspension.spring_force(0.08)  # Large compression

        # With progressive rate, force grows faster than linear
        linear_ratio = 0.08 / 0.02  # = 4
        actual_ratio = force_large / force_small
        assert actual_ratio > linear_ratio  # Progressive effect


# =============================================================================
# AntiRollBar Tests
# =============================================================================


class TestAntiRollBar:
    """Tests for anti-roll bar simulation."""

    @pytest.fixture
    def arb(self):
        """Create a standard anti-roll bar."""
        return AntiRollBar(stiffness=5000.0)

    def test_initialization(self, arb):
        """ARB should initialize with correct stiffness."""
        assert arb.stiffness == 5000.0

    def test_stiffness_setter_positive(self, arb):
        """Stiffness setter should accept positive values."""
        arb.stiffness = 6000.0
        assert arb.stiffness == 6000.0

    def test_stiffness_setter_rejects_negative(self, arb):
        """Stiffness setter should reject negative values."""
        with pytest.raises(ValueError, match="non-negative"):
            arb.stiffness = -1000

    def test_equal_compression_no_force(self, arb):
        """Equal compression should produce no ARB force."""
        left, right = arb.calculate_force(
            left_compression=0.05,
            right_compression=0.05,
            wheel_base_half=0.8,
        )
        assert left == 0.0
        assert right == 0.0

    def test_unequal_compression_produces_force(self, arb):
        """Unequal compression should produce ARB forces."""
        left, right = arb.calculate_force(
            left_compression=0.08,
            right_compression=0.02,
            wheel_base_half=0.8,
        )
        # Left more compressed - ARB pushes left down, right up
        assert left < 0  # Additional compression force
        assert right > 0  # Extension force
        # Forces should be equal and opposite
        assert abs(left + right) < 0.01

    def test_arb_force_proportional_to_difference(self, arb):
        """ARB force should be proportional to compression difference."""
        left1, _ = arb.calculate_force(
            left_compression=0.06,
            right_compression=0.04,  # diff = 0.02
            wheel_base_half=0.8,
        )
        left2, _ = arb.calculate_force(
            left_compression=0.08,
            right_compression=0.04,  # diff = 0.04
            wheel_base_half=0.8,
        )
        # Double difference should give approximately double force
        assert abs(left2 / left1 - 2.0) < 0.1


# =============================================================================
# SuspensionSystem Tests
# =============================================================================


class TestSuspensionSystem:
    """Tests for complete axle suspension system."""

    @pytest.fixture
    def system(self):
        """Create a suspension system for one axle."""
        return SuspensionSystem(
            suspension_type=SuspensionType.DOUBLE_WISHBONE,
            track_width=1.6,
            rest_length=0.5,
            spring_strength=35000.0,
        )

    def test_initialization(self, system):
        """Suspension system should initialize both sides."""
        assert system.left is not None
        assert system.right is not None
        assert system.track_width == 1.6

    def test_update_symmetric(self, system):
        """Symmetric compression should produce symmetric forces."""
        left_force, right_force = system.update(
            left_length=0.45,
            right_length=0.45,
            dt=0.016,
        )
        assert abs(left_force - right_force) < 0.01

    def test_update_asymmetric(self, system):
        """Asymmetric compression should produce different forces."""
        left_force, right_force = system.update(
            left_length=0.42,
            right_length=0.48,
            dt=0.016,
        )
        assert left_force > right_force  # More compressed = more force

    def test_with_anti_roll_bar(self, system):
        """Anti-roll bar should affect force distribution."""
        # Without ARB
        left1, right1 = system.update(
            left_length=0.42,
            right_length=0.48,
            dt=0.016,
        )

        # Reset and add ARB
        system.reset()
        system.set_anti_roll_bar(AntiRollBar(stiffness=5000.0))

        left2, right2 = system.update(
            left_length=0.42,
            right_length=0.48,
            dt=0.016,
        )

        # ARB should reduce difference between sides
        diff_without_arb = abs(left1 - right1)
        diff_with_arb = abs(left2 - right2)
        assert diff_with_arb < diff_without_arb

    def test_roll_angle_estimate(self, system):
        """Should estimate body roll from compressions."""
        system.update(
            left_length=0.42,  # More compressed
            right_length=0.48,  # Less compressed
            dt=0.016,
        )

        roll = system.get_roll_angle()
        # Left more compressed = rolling right (positive)
        assert roll != 0.0

    def test_reset(self, system):
        """Reset should restore both suspensions."""
        system.update(left_length=0.42, right_length=0.48, dt=0.016)
        system.reset()
        assert system.left.compression == 0.0
        assert system.right.compression == 0.0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestSuspensionEdgeCases:
    """Tests for edge cases in suspension behavior."""

    def test_zero_travel(self):
        """Zero travel should still work."""
        susp = Suspension(rest_length=0.5, travel=0.0)
        # compression_ratio should handle this
        _ = susp.compression_ratio  # Should not error

    def test_very_high_velocity(self):
        """Very high velocity should still produce reasonable damping."""
        susp = Suspension(
            damper_compression=4500.0,
            damper_rebound=4000.0,
        )
        force = susp.damper_force(100.0)  # 100 m/s - extreme
        # Should be large but finite
        assert force > 0
        assert force < float('inf')

    def test_zero_dt_update(self):
        """Zero dt should not cause division by zero."""
        susp = Suspension()
        # This should handle dt=0 gracefully
        force = susp.update(length=0.45, dt=0.0)
        # Velocity calculation should handle zero dt
        assert susp.velocity == 0.0

    def test_suspension_types(self):
        """All suspension types should be valid."""
        for susp_type in SuspensionType:
            susp = Suspension(suspension_type=susp_type)
            assert susp.suspension_type == susp_type
