"""Tests for watercraft simulation (boats, ships).

Tests cover:
- Watercraft initialization
- Buoyancy calculations
- Hull hydrodynamics
- Propeller thrust
- Rudder steering
- Wave interaction
- Edge cases
"""

import math
import pytest

from engine.simulation.vehicles.watercraft import (
    HullType,
    WatercraftType,
    BuoyancySamplePoint,
    Propeller,
    Rudder,
    WaveState,
    Watercraft,
)
from engine.simulation.vehicles.vehicle_system import Vector3, Transform, VehicleState


# =============================================================================
# BuoyancySamplePoint Tests
# =============================================================================


class TestBuoyancySamplePoint:
    """Tests for BuoyancySamplePoint dataclass."""

    def test_default_values(self):
        """BuoyancySamplePoint should have sensible defaults."""
        point = BuoyancySamplePoint()
        assert point.volume == 1.0
        assert point.submerged_ratio == 0.0
        assert point.buoyancy_force == 0.0

    def test_custom_values(self):
        """BuoyancySamplePoint should accept custom values."""
        point = BuoyancySamplePoint(
            local_position=Vector3(-1, -0.5, 2),
            volume=2.5,
            submerged_ratio=0.8,
        )
        assert point.volume == 2.5
        assert point.submerged_ratio == 0.8


# =============================================================================
# Propeller Tests
# =============================================================================


class TestPropeller:
    """Tests for Propeller dataclass."""

    def test_default_values(self):
        """Propeller should have sensible defaults."""
        prop = Propeller()
        assert prop.max_thrust > 0
        assert prop.efficiency > 0
        assert prop.current_throttle == 0.0

    def test_custom_values(self):
        """Propeller should accept custom values."""
        prop = Propeller(
            max_thrust=15000.0,
            efficiency=0.75,
            diameter=0.6,
        )
        assert prop.max_thrust == 15000.0
        assert prop.efficiency == 0.75
        assert prop.diameter == 0.6


# =============================================================================
# Rudder Tests
# =============================================================================


class TestRudder:
    """Tests for Rudder dataclass."""

    def test_default_values(self):
        """Rudder should have sensible defaults."""
        rudder = Rudder()
        assert rudder.area > 0
        assert rudder.max_angle > 0
        assert rudder.current_angle == 0.0


# =============================================================================
# WaveState Tests
# =============================================================================


class TestWaveState:
    """Tests for WaveState dataclass."""

    def test_default_values(self):
        """WaveState should have sensible defaults."""
        waves = WaveState()
        assert waves.amplitude >= 0
        assert waves.frequency > 0


# =============================================================================
# Watercraft Initialization Tests
# =============================================================================


class TestWatercraftInit:
    """Tests for Watercraft initialization."""

    @pytest.fixture
    def boat(self):
        """Create a standard watercraft."""
        return Watercraft(
            mass=1000.0,
            length=8.0,
            beam=2.5,
            draft=0.8,
        )

    def test_initialization(self, boat):
        """Watercraft should initialize correctly."""
        assert boat.mass == 1000.0
        assert boat._length == 8.0
        assert boat._beam == 2.5
        assert boat._draft == 0.8
        assert boat.vehicle_type.name == "WATERCRAFT"

    def test_displaced_volume_calculated(self, boat):
        """Displaced volume should be calculated for neutral buoyancy."""
        expected = 1000.0 / 1025.0  # mass / water_density
        assert abs(boat._displaced_volume - expected) < 0.1

    def test_custom_displaced_volume(self):
        """Should accept custom displaced volume."""
        boat = Watercraft(mass=1000.0, displaced_volume=2.0)
        assert boat._displaced_volume == 2.0

    def test_invalid_displaced_volume(self):
        """Zero or negative volume should raise error."""
        with pytest.raises(ValueError):
            Watercraft(mass=1000.0, displaced_volume=0.0)

        with pytest.raises(ValueError):
            Watercraft(mass=1000.0, displaced_volume=-1.0)

    def test_has_buoyancy_points(self, boat):
        """Should have buoyancy sample points."""
        assert len(boat._buoyancy_points) > 0

    def test_has_propeller(self, boat):
        """Should have propeller(s)."""
        assert len(boat._propellers) > 0

    def test_has_rudder(self, boat):
        """Should have rudder."""
        assert boat._rudder is not None

    def test_vehicle_id_generated(self, boat):
        """Should have generated ID."""
        assert boat.vehicle_id is not None

    def test_custom_hull_type(self):
        """Should accept custom hull type."""
        boat = Watercraft(hull_type=HullType.PLANING)
        assert boat._hull_type == HullType.PLANING


# =============================================================================
# Input Tests
# =============================================================================


class TestInputs:
    """Tests for control inputs."""

    @pytest.fixture
    def boat(self):
        """Create a watercraft."""
        return Watercraft()

    def test_throttle_clamp(self, boat):
        """Throttle should be clamped to [-1, 1]."""
        boat.throttle = 2.0
        assert boat.throttle == 1.0

        boat.throttle = -2.0
        assert boat.throttle == -1.0

    def test_steering_clamp(self, boat):
        """Steering should be clamped to [-1, 1]."""
        boat.steering = 2.0
        assert boat.steering == 1.0

        boat.steering = -2.0
        assert boat.steering == -1.0


# =============================================================================
# Buoyancy Tests
# =============================================================================


class TestBuoyancy:
    """Tests for buoyancy calculations."""

    @pytest.fixture
    def boat(self):
        """Create a watercraft at waterline."""
        boat = Watercraft(mass=1000.0, draft=0.8)
        boat._water_height = 0.0
        boat.transform.position.y = 0.4  # Approximately at waterline
        return boat

    def test_calculate_buoyancy_returns_force(self, boat):
        """Should return buoyancy force and torque."""
        force, torque = boat.calculate_buoyancy()
        assert isinstance(force, Vector3)
        assert isinstance(torque, Vector3)

    def test_buoyancy_force_upward(self, boat):
        """Buoyancy should be upward (positive Y)."""
        boat.transform.position.y = 0.0  # At water level
        force, _ = boat.calculate_buoyancy()
        assert force.y > 0

    def test_buoyancy_increases_when_lower(self, boat):
        """More submerged should give more buoyancy."""
        boat.transform.position.y = 0.0
        force_high, _ = boat.calculate_buoyancy()

        boat.transform.position.y = -0.2  # More submerged
        force_low, _ = boat.calculate_buoyancy()

        assert force_low.y > force_high.y

    def test_no_buoyancy_above_water(self, boat):
        """Above water should have minimal buoyancy."""
        boat.transform.position.y = 5.0  # Well above water
        force, _ = boat.calculate_buoyancy()
        # Should be zero or very small
        assert force.y < 100  # Small residual


# =============================================================================
# Wave Tests
# =============================================================================


class TestWaves:
    """Tests for wave interaction."""

    @pytest.fixture
    def boat(self):
        """Create a watercraft with waves."""
        boat = Watercraft()
        boat.set_wave_conditions(amplitude=1.0, frequency=0.5)
        return boat

    def test_water_height_varies_with_waves(self, boat):
        """Water height should vary based on position and time."""
        boat._waves.phase = 0.0
        h1 = boat.get_water_height_at(Vector3(0, 0, 0))

        boat._waves.phase = math.pi  # Half cycle
        h2 = boat.get_water_height_at(Vector3(0, 0, 0))

        assert h1 != h2

    def test_set_wave_conditions(self, boat):
        """Should set wave amplitude and frequency."""
        boat.set_wave_conditions(amplitude=2.0, frequency=1.0)
        assert boat._waves.amplitude == 2.0
        assert boat._waves.frequency == 1.0

    def test_wave_amplitude_clamped(self, boat):
        """Amplitude should not be negative."""
        boat.set_wave_conditions(amplitude=-1.0)
        assert boat._waves.amplitude >= 0

    def test_wave_direction(self, boat):
        """Should set wave direction."""
        boat.set_wave_conditions(direction=Vector3(0, 0, 1))
        assert boat._waves.direction.z != 0


# =============================================================================
# Hull Drag Tests
# =============================================================================


class TestHullDrag:
    """Tests for hull hydrodynamic drag."""

    @pytest.fixture
    def boat(self):
        """Create a watercraft."""
        return Watercraft()

    def test_drag_at_speed(self, boat):
        """Should have drag when moving."""
        boat.velocity = Vector3(0, 0, 10.0)
        drag_force, _ = boat.calculate_hull_drag()

        # Drag should oppose motion
        assert drag_force.z < 0

    def test_no_drag_stationary(self, boat):
        """No drag when stationary."""
        boat.velocity = Vector3.zero()
        drag_force, _ = boat.calculate_hull_drag()
        assert drag_force.magnitude() < 0.01

    def test_lateral_drag_higher(self, boat):
        """Lateral drag should be higher than forward drag."""
        boat.velocity = Vector3(10, 0, 0)  # Lateral
        drag_lat, _ = boat.calculate_hull_drag()

        boat.velocity = Vector3(0, 0, 10)  # Forward
        drag_fwd, _ = boat.calculate_hull_drag()

        # Lateral drag should be higher (boats resist sideways motion)
        assert abs(drag_lat.x) > abs(drag_fwd.z)

    def test_planing_hull_reduces_drag(self):
        """Planing hull should have reduced drag at high speed."""
        displacement = Watercraft(hull_type=HullType.DISPLACEMENT)
        planing = Watercraft(hull_type=HullType.PLANING)

        displacement.velocity = Vector3(0, 0, 15.0)
        planing.velocity = Vector3(0, 0, 15.0)

        drag_d, _ = displacement.calculate_hull_drag()
        drag_p, _ = planing.calculate_hull_drag()

        assert abs(drag_p.z) < abs(drag_d.z)


# =============================================================================
# Propulsion Tests
# =============================================================================


class TestPropulsion:
    """Tests for propeller thrust."""

    @pytest.fixture
    def boat(self):
        """Create a watercraft."""
        return Watercraft()

    def test_thrust_with_throttle(self, boat):
        """Throttle should produce thrust."""
        boat.throttle = 1.0
        boat._accumulated_force = Vector3.zero()

        boat.update_propulsion(dt=0.016)

        assert boat._accumulated_force.magnitude() > 0

    def test_no_thrust_zero_throttle(self, boat):
        """Zero throttle should produce no thrust."""
        boat.throttle = 0.0
        boat._accumulated_force = Vector3.zero()

        boat.update_propulsion(dt=0.016)

        assert boat._accumulated_force.magnitude() < 0.1

    def test_reverse_thrust(self, boat):
        """Negative throttle should produce reverse thrust."""
        boat.throttle = -1.0
        boat._accumulated_force = Vector3.zero()

        boat.update_propulsion(dt=0.016)

        # Should have negative Z force
        assert boat._accumulated_force.z < 0


# =============================================================================
# Steering Tests
# =============================================================================


class TestSteering:
    """Tests for rudder steering."""

    @pytest.fixture
    def boat(self):
        """Create a watercraft."""
        return Watercraft()

    def test_rudder_angle_set(self, boat):
        """Steering input should set rudder angle."""
        boat.steering = 0.5
        boat.update_steering(dt=0.016)

        expected = 0.5 * boat._rudder.max_angle
        assert abs(boat._rudder.current_angle - expected) < 0.1

    def test_no_steering_stationary(self, boat):
        """No steering force when stationary."""
        boat.velocity = Vector3.zero()
        boat.steering = 1.0
        boat._accumulated_force = Vector3.zero()

        boat.update_steering(dt=0.016)

        assert boat._accumulated_force.magnitude() < 1.0

    def test_steering_at_speed(self, boat):
        """Should have steering force when moving."""
        boat.velocity = Vector3(0, 0, 10.0)
        boat.steering = 1.0
        boat._accumulated_force = Vector3.zero()

        boat.update_steering(dt=0.016)

        # Should have lateral force
        assert boat._accumulated_force.magnitude() > 0


# =============================================================================
# Update Cycle Tests
# =============================================================================


class TestUpdateCycle:
    """Tests for complete update cycle."""

    @pytest.fixture
    def boat(self):
        """Create a watercraft."""
        return Watercraft()

    def test_zero_dt_no_crash(self, boat):
        """Zero dt should not cause errors."""
        boat.update(dt=0.0)

    def test_update_cycle_completes(self, boat):
        """Full update cycle should complete."""
        boat.throttle = 0.8
        boat.steering = 0.3

        for _ in range(100):
            boat.update(dt=0.016)

        # Should not crash

    def test_speed_property(self, boat):
        """Speed property should return correct magnitude."""
        boat.velocity = Vector3(3, 0, 4)
        assert abs(boat.speed - 5.0) < 0.01

    def test_speed_knots(self, boat):
        """Speed in knots should be correct."""
        boat.velocity = Vector3(0, 0, 10.0)
        expected = 10.0 * 1.944
        assert abs(boat.speed_knots - expected) < 0.1


# =============================================================================
# Force Application Tests
# =============================================================================


class TestForceApplication:
    """Tests for force and torque application."""

    @pytest.fixture
    def boat(self):
        """Create a watercraft."""
        return Watercraft()

    def test_apply_force(self, boat):
        """Should accumulate forces."""
        boat._accumulated_force = Vector3.zero()
        boat.apply_force(Vector3(1000, 0, 0))
        assert boat._accumulated_force.x == 1000

    def test_apply_force_with_offset(self, boat):
        """Off-center force should create torque."""
        boat._accumulated_force = Vector3.zero()
        boat._accumulated_torque = Vector3.zero()

        boat.apply_force(Vector3(0, 0, 1000), Vector3(1, 0, 0))
        assert boat._accumulated_torque.magnitude() > 0

    def test_apply_torque(self, boat):
        """Should accumulate torques."""
        boat._accumulated_torque = Vector3.zero()
        boat.apply_torque(Vector3(0, 100, 0))
        assert boat._accumulated_torque.y == 100


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Tests for watercraft reset."""

    @pytest.fixture
    def boat(self):
        """Create a watercraft."""
        return Watercraft(draft=0.8)

    def test_reset_position(self, boat):
        """Reset should set position at waterline."""
        boat.transform.position = Vector3(100, -5, 200)
        boat._water_height = 0.0
        boat.reset()
        # Should be at water height + half draft
        assert boat.transform.position.y == pytest.approx(0.4, abs=0.1)

    def test_reset_velocity(self, boat):
        """Reset should clear velocity."""
        boat.velocity = Vector3(10, 0, 5)
        boat.reset()
        assert boat.velocity.magnitude() == 0

    def test_reset_inputs(self, boat):
        """Reset should clear inputs."""
        boat.throttle = 0.8
        boat.steering = -0.5
        boat.reset()

        assert boat.throttle == 0.0
        assert boat.steering == 0.0

    def test_reset_rudder(self, boat):
        """Reset should center rudder."""
        boat._rudder.current_angle = 30.0
        boat.reset()
        assert boat._rudder.current_angle == 0.0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual situations."""

    def test_all_hull_types(self):
        """All hull types should work."""
        for hull_type in HullType:
            boat = Watercraft(hull_type=hull_type)
            assert boat._hull_type == hull_type

    def test_all_watercraft_types(self):
        """All watercraft types should work."""
        for wc_type in WatercraftType:
            boat = Watercraft(watercraft_type=wc_type)
            assert boat._watercraft_type == wc_type

    def test_very_heavy_ship(self):
        """Very heavy ship should still work."""
        ship = Watercraft(mass=100000.0)  # 100 tons
        ship.update(dt=0.016)

    def test_very_light_boat(self):
        """Very light boat should still work."""
        boat = Watercraft(mass=50.0)
        boat.update(dt=0.016)

    def test_set_water_height(self):
        """Should set water height."""
        boat = Watercraft()
        boat.set_water_height(10.0)
        assert boat._water_height == 10.0

    def test_large_waves(self):
        """Large waves should still work."""
        boat = Watercraft()
        boat.set_wave_conditions(amplitude=5.0, frequency=0.2)
        for _ in range(50):
            boat.update(dt=0.016)

    def test_calm_water(self):
        """Zero wave amplitude should work."""
        boat = Watercraft()
        boat.set_wave_conditions(amplitude=0.0)
        boat.update(dt=0.016)
