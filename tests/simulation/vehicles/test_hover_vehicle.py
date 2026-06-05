"""Tests for hover vehicle simulation (hovercraft, hovercars).

Tests cover:
- HoverVehicle initialization
- Air cushion physics
- Lift fan thrust
- Thrust vectoring
- Skirt contact physics
- Drag calculations
- Edge cases
"""

import math
import pytest

from engine.simulation.vehicles.hover_vehicle import (
    HoverMode,
    LiftFan,
    ThrustVector,
    SkirtState,
    HoverVehicle,
)
from engine.simulation.vehicles.vehicle_system import Vector3, Transform, VehicleState


# =============================================================================
# LiftFan Tests
# =============================================================================


class TestLiftFan:
    """Tests for LiftFan dataclass."""

    def test_default_values(self):
        """LiftFan should have sensible defaults."""
        fan = LiftFan()
        assert fan.max_thrust > 0
        assert fan.efficiency > 0
        assert fan.current_power == 1.0
        assert fan.is_active

    def test_custom_values(self):
        """LiftFan should accept custom values."""
        fan = LiftFan(
            max_thrust=60000.0,
            efficiency=0.9,
            current_power=0.8,
            is_active=False,
        )
        assert fan.max_thrust == 60000.0
        assert fan.efficiency == 0.9
        assert fan.current_power == 0.8
        assert not fan.is_active


# =============================================================================
# ThrustVector Tests
# =============================================================================


class TestThrustVector:
    """Tests for ThrustVector dataclass."""

    def test_default_values(self):
        """ThrustVector should have sensible defaults."""
        tv = ThrustVector()
        assert tv.max_thrust > 0
        assert tv.gimbal_range > 0
        assert tv.is_active

    def test_thrust_vector_gimbal(self):
        """get_thrust_vector should account for gimbal."""
        tv = ThrustVector(
            direction=Vector3(0, 0, 1),
            gimbal_angle=0.0,
        )
        vec = tv.get_thrust_vector()
        assert vec.z == pytest.approx(1.0, abs=0.01)

        tv.gimbal_angle = 90.0  # Full gimbal
        vec = tv.get_thrust_vector()
        # Should be rotated
        assert vec.x != 0 or vec.z != 1.0


# =============================================================================
# SkirtState Tests
# =============================================================================


class TestSkirtState:
    """Tests for SkirtState dataclass."""

    def test_default_values(self):
        """SkirtState should have sensible defaults."""
        state = SkirtState()
        assert state.compression == 0.0
        assert state.pressure == 0.0
        assert state.contact_points == 0


# =============================================================================
# HoverVehicle Initialization Tests
# =============================================================================


class TestHoverVehicleInit:
    """Tests for HoverVehicle initialization."""

    @pytest.fixture
    def hovercraft(self):
        """Create a standard hover vehicle."""
        return HoverVehicle(
            mass=2000.0,
            length=5.0,
            width=3.0,
            hover_height=0.5,
        )

    def test_initialization(self, hovercraft):
        """Hover vehicle should initialize correctly."""
        assert hovercraft.mass == 2000.0
        assert hovercraft.hover_height == 0.5
        assert hovercraft.vehicle_type.name == "HOVER"
        assert hovercraft.state == VehicleState.ACTIVE

    def test_lift_fans_created(self, hovercraft):
        """Should have lift fans."""
        assert len(hovercraft._lift_fans) > 0

    def test_thrust_vectors_created(self, hovercraft):
        """Should have thrust vectors."""
        assert len(hovercraft._thrust_vectors) > 0

    def test_vehicle_id_generated(self, hovercraft):
        """Should have generated ID."""
        assert hovercraft.vehicle_id is not None

    def test_custom_vehicle_id(self):
        """Should accept custom ID."""
        hv = HoverVehicle(vehicle_id="hover-001")
        assert hv.vehicle_id == "hover-001"

    def test_multiple_lift_fans(self):
        """Should support multiple lift fans."""
        hv = HoverVehicle(num_lift_fans=4)
        assert len(hv._lift_fans) == 4

    def test_multiple_thrust_vectors(self):
        """Should support multiple thrust vectors."""
        hv = HoverVehicle(num_thrust_vectors=2)
        assert len(hv._thrust_vectors) == 2


# =============================================================================
# Input Tests
# =============================================================================


class TestInputs:
    """Tests for control inputs."""

    @pytest.fixture
    def hovercraft(self):
        """Create a hover vehicle."""
        return HoverVehicle()

    def test_throttle_clamp(self, hovercraft):
        """Throttle should be clamped to [0, 1]."""
        hovercraft.throttle = 2.0
        assert hovercraft.throttle == 1.0

        hovercraft.throttle = -1.0
        assert hovercraft.throttle == 0.0

    def test_lift_power_clamp(self, hovercraft):
        """Lift power should be clamped to [0, 1]."""
        hovercraft.lift_power = 2.0
        assert hovercraft.lift_power == 1.0

        hovercraft.lift_power = -1.0
        assert hovercraft.lift_power == 0.0

    def test_rudder_clamp(self, hovercraft):
        """Rudder should be clamped to [-1, 1]."""
        hovercraft.rudder = 2.0
        assert hovercraft.rudder == 1.0

        hovercraft.rudder = -2.0
        assert hovercraft.rudder == -1.0

    def test_hover_height_setter(self, hovercraft):
        """Hover height should be settable with minimum."""
        hovercraft.hover_height = 0.8
        assert hovercraft.hover_height == 0.8

        hovercraft.hover_height = 0.0
        assert hovercraft.hover_height >= 0.1  # Minimum enforced


# =============================================================================
# Cushion Pressure Tests
# =============================================================================


class TestCushionPressure:
    """Tests for air cushion pressure calculations."""

    @pytest.fixture
    def hovercraft(self):
        """Create a hover vehicle."""
        return HoverVehicle(
            mass=2000.0,
            hover_height=0.5,
        )

    def test_pressure_at_hover_height(self, hovercraft):
        """Pressure should be correct at hover height."""
        pressure = hovercraft.calculate_cushion_pressure(0.5)
        assert pressure > 0

    def test_pressure_increases_lower(self, hovercraft):
        """Pressure should increase when closer to ground."""
        pressure_high = hovercraft.calculate_cushion_pressure(0.4)
        pressure_low = hovercraft.calculate_cushion_pressure(0.2)
        assert pressure_low > pressure_high

    def test_pressure_zero_too_high(self, hovercraft):
        """Pressure should be zero when too high."""
        # Way above hover height + skirt depth
        pressure = hovercraft.calculate_cushion_pressure(2.0)
        assert pressure == 0.0

    def test_pressure_with_lift_power(self, hovercraft):
        """Pressure should scale with lift power."""
        hovercraft.lift_power = 1.0
        pressure_full = hovercraft.calculate_cushion_pressure(0.3)

        hovercraft.lift_power = 0.5
        pressure_half = hovercraft.calculate_cushion_pressure(0.3)

        assert abs(pressure_half - pressure_full * 0.5) < pressure_full * 0.1


# =============================================================================
# Lift Tests
# =============================================================================


class TestLift:
    """Tests for lift force calculations."""

    @pytest.fixture
    def hovercraft(self):
        """Create a hover vehicle."""
        return HoverVehicle(mass=2000.0)

    def test_hover_force_property(self, hovercraft):
        """Should report total lift force."""
        hovercraft.transform.position.y = 0.5
        hovercraft.update_lift(dt=0.016)
        assert hovercraft.hover_force >= 0

    def test_lift_near_ground(self, hovercraft):
        """Should generate lift near ground."""
        hovercraft.transform.position.y = 0.3
        hovercraft._ground_height = 0.0
        hovercraft._accumulated_force = Vector3.zero()

        hovercraft.update_lift(dt=0.016)

        # Should have positive Y force
        assert hovercraft._accumulated_force.y > 0

    def test_lift_fan_inactive(self, hovercraft):
        """Inactive fan should not contribute."""
        for fan in hovercraft._lift_fans:
            fan.is_active = False

        hovercraft.transform.position.y = 0.3
        hovercraft.update_lift(dt=0.016)

        # Fans should show zero thrust
        for fan in hovercraft._lift_fans:
            assert fan.current_thrust == 0.0


# =============================================================================
# Thrust Tests
# =============================================================================


class TestThrust:
    """Tests for propulsion thrust."""

    @pytest.fixture
    def hovercraft(self):
        """Create a hover vehicle."""
        return HoverVehicle()

    def test_thrust_with_throttle(self, hovercraft):
        """Throttle should produce thrust force."""
        hovercraft.throttle = 1.0
        hovercraft._accumulated_force = Vector3.zero()

        hovercraft.update_thrust(dt=0.016)

        # Should have forward force
        assert hovercraft._accumulated_force.magnitude() > 0

    def test_no_thrust_zero_throttle(self, hovercraft):
        """Zero throttle should produce no thrust."""
        hovercraft.throttle = 0.0
        hovercraft._accumulated_force = Vector3.zero()

        hovercraft.update_thrust(dt=0.016)

        # Force should be zero (or very small)
        assert hovercraft._accumulated_force.magnitude() < 0.1

    def test_rudder_affects_gimbal(self, hovercraft):
        """Rudder input should affect thrust gimbal."""
        hovercraft.throttle = 1.0
        hovercraft.rudder = 1.0

        hovercraft.update_thrust(dt=0.016)

        # Thrusters should have gimbal angle
        for thruster in hovercraft._thrust_vectors:
            assert thruster.gimbal_angle != 0.0


# =============================================================================
# Drag Tests
# =============================================================================


class TestDrag:
    """Tests for drag forces."""

    @pytest.fixture
    def hovercraft(self):
        """Create a hover vehicle."""
        return HoverVehicle()

    def test_drag_at_speed(self, hovercraft):
        """Should have drag when moving."""
        hovercraft.velocity = Vector3(0, 0, 20.0)
        hovercraft._accumulated_force = Vector3.zero()

        hovercraft.update_drag(dt=0.016)

        # Drag should oppose motion
        assert hovercraft._accumulated_force.z < 0

    def test_no_drag_stationary(self, hovercraft):
        """No drag when stationary."""
        hovercraft.velocity = Vector3.zero()
        hovercraft._accumulated_force = Vector3.zero()

        hovercraft.update_drag(dt=0.016)

        # Minimal force
        assert hovercraft._accumulated_force.magnitude() < 1.0


# =============================================================================
# Skirt Contact Tests
# =============================================================================


class TestSkirtContact:
    """Tests for skirt ground contact."""

    @pytest.fixture
    def hovercraft(self):
        """Create a hover vehicle."""
        return HoverVehicle(skirt_depth=0.3)

    def test_skirt_compression_near_ground(self, hovercraft):
        """Skirt should compress near ground."""
        hovercraft.transform.position.y = 0.2  # Below skirt bottom
        hovercraft._ground_height = 0.0
        hovercraft._skirt_state = SkirtState()

        hovercraft.update_lift(dt=0.016)

        assert hovercraft._skirt_state.compression > 0
        assert hovercraft._skirt_state.contact_points > 0

    def test_no_skirt_contact_high(self, hovercraft):
        """Skirt should not contact when high."""
        hovercraft.transform.position.y = 1.0  # Well above ground
        hovercraft._ground_height = 0.0
        hovercraft._skirt_state = SkirtState()

        hovercraft.update_lift(dt=0.016)

        assert hovercraft._skirt_state.compression == 0
        assert hovercraft._skirt_state.contact_points == 0


# =============================================================================
# Update Cycle Tests
# =============================================================================


class TestUpdateCycle:
    """Tests for complete update cycle."""

    @pytest.fixture
    def hovercraft(self):
        """Create a hover vehicle."""
        return HoverVehicle()

    def test_zero_dt_no_crash(self, hovercraft):
        """Zero dt should not cause errors."""
        hovercraft.update(dt=0.0)

    def test_update_cycle_completes(self, hovercraft):
        """Full update cycle should complete."""
        hovercraft.throttle = 0.5
        hovercraft.rudder = 0.2
        hovercraft.lift_power = 1.0

        for _ in range(100):
            hovercraft.update(dt=0.016)

        # Should not crash

    def test_speed_property(self, hovercraft):
        """Speed property should return correct magnitude."""
        hovercraft.velocity = Vector3(3, 0, 4)
        assert abs(hovercraft.speed - 5.0) < 0.01

    def test_ground_clamping(self, hovercraft):
        """Should not go below ground."""
        hovercraft.transform.position.y = -1.0
        hovercraft._ground_height = 0.0

        hovercraft.update(dt=0.016)

        # Should clamp to ground level
        assert hovercraft.transform.position.y >= 0.0


# =============================================================================
# Force Application Tests
# =============================================================================


class TestForceApplication:
    """Tests for force and torque application."""

    @pytest.fixture
    def hovercraft(self):
        """Create a hover vehicle."""
        return HoverVehicle()

    def test_apply_force(self, hovercraft):
        """Should accumulate forces."""
        hovercraft._accumulated_force = Vector3.zero()
        hovercraft.apply_force(Vector3(1000, 0, 0))
        assert hovercraft._accumulated_force.x == 1000

    def test_apply_force_with_offset(self, hovercraft):
        """Off-center force should create torque."""
        hovercraft._accumulated_force = Vector3.zero()
        hovercraft._accumulated_torque = Vector3.zero()

        hovercraft.apply_force(Vector3(0, 0, 1000), Vector3(1, 0, 0))
        assert hovercraft._accumulated_torque.magnitude() > 0

    def test_apply_torque(self, hovercraft):
        """Should accumulate torques."""
        hovercraft._accumulated_torque = Vector3.zero()
        hovercraft.apply_torque(Vector3(0, 100, 0))
        assert hovercraft._accumulated_torque.y == 100


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Tests for vehicle reset."""

    @pytest.fixture
    def hovercraft(self):
        """Create a hover vehicle."""
        return HoverVehicle(hover_height=0.5)

    def test_reset_position(self, hovercraft):
        """Reset should set position to hover height."""
        hovercraft.transform.position = Vector3(100, 0, 200)
        hovercraft.reset()
        assert hovercraft.transform.position.y == 0.5  # hover_height

    def test_reset_velocity(self, hovercraft):
        """Reset should clear velocity."""
        hovercraft.velocity = Vector3(30, 5, 20)
        hovercraft.reset()
        assert hovercraft.velocity.magnitude() == 0

    def test_reset_inputs(self, hovercraft):
        """Reset should clear inputs."""
        hovercraft.throttle = 0.8
        hovercraft.rudder = -0.5
        hovercraft.reset()

        assert hovercraft.throttle == 0.0
        assert hovercraft.lift_power == 1.0
        assert hovercraft.rudder == 0.0


# =============================================================================
# Ground Data Tests
# =============================================================================


class TestGroundData:
    """Tests for ground raycast data."""

    @pytest.fixture
    def hovercraft(self):
        """Create a hover vehicle."""
        return HoverVehicle()

    def test_set_ground_data(self, hovercraft):
        """Should set ground height and normal."""
        hovercraft.set_ground_data(
            height=1.0,
            normal=Vector3(0, 1, 0),
        )
        assert hovercraft._ground_height == 1.0
        assert hovercraft._ground_normal.y == 1.0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual situations."""

    def test_all_hover_modes(self):
        """All hover modes should work."""
        for mode in HoverMode:
            hv = HoverVehicle(hover_mode=mode)
            assert hv._hover_mode == mode

    def test_very_heavy_hovercraft(self):
        """Very heavy hovercraft should still work."""
        hv = HoverVehicle(mass=50000.0)
        hv.update(dt=0.016)

    def test_very_light_hovercraft(self):
        """Very light hovercraft should still work."""
        hv = HoverVehicle(mass=100.0)
        hv.update(dt=0.016)

    def test_zero_lift_fans(self):
        """Zero lift fans should still initialize."""
        # This might create at least 1 internally
        hv = HoverVehicle(num_lift_fans=0)
        hv.update(dt=0.016)  # Should not crash

    def test_single_thrust_vector(self):
        """Single thrust vector should work."""
        hv = HoverVehicle(num_thrust_vectors=1)
        assert len(hv._thrust_vectors) == 1

    def test_very_low_hover_height(self):
        """Very low hover height should be clamped."""
        hv = HoverVehicle(hover_height=0.05)
        hv.hover_height = 0.01
        assert hv.hover_height >= 0.1  # Should be minimum
