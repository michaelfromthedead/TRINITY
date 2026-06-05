"""Tests for aircraft simulation.

Tests cover:
- Aircraft initialization
- Aerodynamic force calculations (lift, drag)
- Control surfaces (aileron, elevator, rudder)
- Engine thrust
- Stall behavior
- Flight phases
- Edge cases (inverted flight, extreme angles)
"""

import math
import pytest

from engine.simulation.vehicles.aircraft import (
    AircraftType,
    FlightPhase,
    AerodynamicSurface,
    ControlSurface,
    AircraftEngine,
    Aircraft,
)
from engine.simulation.vehicles.vehicle_system import Vector3, Transform, VehicleState


# =============================================================================
# AerodynamicSurface Tests
# =============================================================================


class TestAerodynamicSurface:
    """Tests for AerodynamicSurface calculations."""

    @pytest.fixture
    def wing(self):
        """Create a standard wing."""
        return AerodynamicSurface(
            name="main_wing",
            area=16.0,
            span=12.0,
            stall_angle=15.0,
        )

    def test_initialization(self, wing):
        """Surface should initialize correctly."""
        assert wing.name == "main_wing"
        assert wing.area == 16.0
        assert wing.span == 12.0

    def test_aspect_ratio_calculated(self, wing):
        """Aspect ratio should be calculated from span and area."""
        expected = 12.0 ** 2 / 16.0  # = 9
        assert wing.aspect_ratio == expected

    def test_lift_coefficient_zero_aoa(self, wing):
        """Zero AoA should have near-zero lift coefficient."""
        # Account for zero-lift AoA offset
        cl = wing.compute_lift_coefficient(wing.zero_lift_aoa)
        assert abs(cl) < 0.1

    def test_lift_coefficient_increases_with_aoa(self, wing):
        """Lift coefficient should increase with AoA."""
        cl_low = wing.compute_lift_coefficient(2.0)
        cl_high = wing.compute_lift_coefficient(8.0)
        assert cl_high > cl_low

    def test_lift_coefficient_stall(self, wing):
        """Lift coefficient should reduce after stall."""
        cl_pre_stall = wing.compute_lift_coefficient(14.0)  # Just before stall
        cl_post_stall = wing.compute_lift_coefficient(25.0)  # Well past stall

        # Post-stall lift should be reduced
        assert cl_post_stall < cl_pre_stall

    def test_lift_coefficient_clamped(self, wing):
        """Lift coefficient should be within limits."""
        cl_max = wing.compute_lift_coefficient(20.0)
        assert cl_max <= wing.max_lift_coeff

        cl_min = wing.compute_lift_coefficient(-20.0)
        assert cl_min >= wing.min_lift_coeff

    def test_drag_coefficient_with_lift(self, wing):
        """Drag should increase with lift squared."""
        cd_low = wing.compute_drag_coefficient(0.5, 5.0)
        cd_high = wing.compute_drag_coefficient(1.0, 10.0)
        assert cd_high > cd_low

    def test_drag_coefficient_post_stall(self, wing):
        """Drag should increase significantly post-stall."""
        cd_normal = wing.compute_drag_coefficient(0.5, 10.0)
        cd_stall = wing.compute_drag_coefficient(0.5, 25.0)  # Post stall
        assert cd_stall > cd_normal


# =============================================================================
# ControlSurface Tests
# =============================================================================


class TestControlSurface:
    """Tests for ControlSurface dataclass."""

    def test_default_values(self):
        """Control surface should have zero deflections by default."""
        cs = ControlSurface()
        assert cs.aileron == 0.0
        assert cs.elevator == 0.0
        assert cs.rudder == 0.0
        assert cs.flaps == 0.0

    def test_custom_values(self):
        """Control surface should accept custom values."""
        cs = ControlSurface(
            aileron=0.5,
            elevator=-0.3,
            rudder=0.2,
            flaps=0.4,
        )
        assert cs.aileron == 0.5
        assert cs.elevator == -0.3
        assert cs.rudder == 0.2
        assert cs.flaps == 0.4


# =============================================================================
# AircraftEngine Tests
# =============================================================================


class TestAircraftEngine:
    """Tests for AircraftEngine."""

    @pytest.fixture
    def engine(self):
        """Create a standard aircraft engine."""
        return AircraftEngine(
            max_thrust=10000.0,
            is_propeller=True,
            propeller_efficiency=0.85,
        )

    def test_initialization(self, engine):
        """Engine should initialize correctly."""
        assert engine.max_thrust == 10000.0
        assert engine.is_propeller
        assert engine.is_running

    def test_thrust_at_zero_throttle(self, engine):
        """Zero throttle should produce zero thrust."""
        engine.current_throttle = 0.0
        thrust = engine.compute_thrust(airspeed=50.0)
        assert thrust == 0.0

    def test_thrust_at_full_throttle(self, engine):
        """Full throttle should produce near-max thrust."""
        engine.current_throttle = 1.0
        thrust = engine.compute_thrust(airspeed=50.0)
        assert thrust > 0
        assert thrust <= engine.max_thrust

    def test_thrust_decreases_with_altitude(self, engine):
        """Thrust should decrease at altitude."""
        engine.current_throttle = 1.0
        thrust_sea_level = engine.compute_thrust(airspeed=50.0, altitude=0.0)
        thrust_high = engine.compute_thrust(airspeed=50.0, altitude=10000.0)
        assert thrust_high < thrust_sea_level

    def test_stopped_engine_no_thrust(self, engine):
        """Stopped engine should produce no thrust."""
        engine.is_running = False
        engine.current_throttle = 1.0
        thrust = engine.compute_thrust(airspeed=50.0)
        assert thrust == 0.0


# =============================================================================
# Aircraft Initialization Tests
# =============================================================================


class TestAircraftInit:
    """Tests for Aircraft initialization."""

    @pytest.fixture
    def aircraft(self):
        """Create a standard aircraft."""
        return Aircraft(
            mass=1000.0,
            wing_area=16.0,
            wing_span=12.0,
            max_thrust=5000.0,
        )

    def test_initialization(self, aircraft):
        """Aircraft should initialize correctly."""
        assert aircraft.mass == 1000.0
        assert aircraft.vehicle_type.name == "AIRCRAFT"
        assert aircraft.state == VehicleState.ACTIVE

    def test_has_main_wing(self, aircraft):
        """Should have main wing."""
        assert aircraft._main_wing is not None
        assert aircraft._main_wing.area == 16.0

    def test_has_tail_surfaces(self, aircraft):
        """Should have horizontal and vertical tail."""
        assert aircraft._h_tail is not None
        assert aircraft._v_tail is not None

    def test_has_engine(self, aircraft):
        """Should have at least one engine."""
        assert len(aircraft._engines) > 0

    def test_starts_grounded(self, aircraft):
        """Aircraft should start grounded."""
        assert aircraft.flight_phase == FlightPhase.GROUNDED
        assert aircraft._is_grounded

    def test_vehicle_id_generated(self, aircraft):
        """Should have generated ID."""
        assert aircraft.vehicle_id is not None

    def test_custom_aircraft_type(self):
        """Should accept custom aircraft type."""
        ac = Aircraft(aircraft_type=AircraftType.HELICOPTER)
        assert ac.aircraft_type == AircraftType.HELICOPTER


# =============================================================================
# Lift and Drag Tests
# =============================================================================


class TestLiftAndDrag:
    """Tests for lift and drag force calculations."""

    @pytest.fixture
    def aircraft(self):
        """Create an aircraft for aero tests."""
        return Aircraft(
            mass=1000.0,
            wing_area=16.0,
            wing_span=12.0,
        )

    def test_compute_lift_with_speed(self, aircraft):
        """Should compute lift at speed."""
        aircraft._angle_of_attack = 5.0
        q = 0.5 * 1.225 * 50.0 ** 2  # Dynamic pressure

        lift = aircraft.compute_lift(q)
        assert lift > 0

    def test_compute_lift_zero_speed(self, aircraft):
        """Zero speed should produce zero lift."""
        lift = aircraft.compute_lift(0.0)
        assert lift == 0.0

    def test_flaps_increase_lift(self, aircraft):
        """Flaps should increase lift."""
        aircraft._angle_of_attack = 5.0
        q = 0.5 * 1.225 * 30.0 ** 2

        aircraft._controls.flaps = 0.0
        lift_no_flaps = aircraft.compute_lift(q)

        aircraft._controls.flaps = 1.0
        lift_full_flaps = aircraft.compute_lift(q)

        assert lift_full_flaps > lift_no_flaps

    def test_compute_drag_increases_with_lift(self, aircraft):
        """Drag should increase with lift (induced drag)."""
        q = 0.5 * 1.225 * 50.0 ** 2

        drag_low = aircraft.compute_drag(q, 1000.0)
        drag_high = aircraft.compute_drag(q, 5000.0)

        assert drag_high > drag_low


# =============================================================================
# Control Tests
# =============================================================================


class TestControls:
    """Tests for control surface inputs."""

    @pytest.fixture
    def aircraft(self):
        """Create an aircraft for control tests."""
        return Aircraft()

    def test_set_throttle(self, aircraft):
        """Should set engine throttle."""
        aircraft.set_throttle(0.75)
        for engine in aircraft._engines:
            assert engine.current_throttle == 0.75

    def test_throttle_clamped(self, aircraft):
        """Throttle should be clamped to [0, 1]."""
        aircraft.set_throttle(2.0)
        for engine in aircraft._engines:
            assert engine.current_throttle == 1.0

        aircraft.set_throttle(-1.0)
        for engine in aircraft._engines:
            assert engine.current_throttle == 0.0

    def test_set_control_inputs(self, aircraft):
        """Should set control surface deflections."""
        aircraft.set_control_inputs(
            pitch=0.5,
            roll=-0.3,
            yaw=0.2,
            flaps=0.6,
        )
        assert aircraft.controls.elevator == 0.5
        assert aircraft.controls.aileron == -0.3
        assert aircraft.controls.rudder == 0.2
        assert aircraft.controls.flaps == 0.6

    def test_control_inputs_clamped(self, aircraft):
        """Control inputs should be clamped."""
        aircraft.set_control_inputs(pitch=2.0, roll=-2.0, yaw=2.0, flaps=2.0)
        assert aircraft.controls.elevator == 1.0
        assert aircraft.controls.aileron == -1.0
        assert aircraft.controls.rudder == 1.0
        assert aircraft.controls.flaps == 1.0


# =============================================================================
# Stall Tests
# =============================================================================


class TestStall:
    """Tests for stall behavior."""

    @pytest.fixture
    def aircraft(self):
        """Create an aircraft for stall tests."""
        return Aircraft()

    def test_is_stalled_property(self, aircraft):
        """is_stalled should reflect AoA vs stall angle."""
        aircraft._angle_of_attack = 5.0  # Below stall
        assert not aircraft.is_stalled

        aircraft._angle_of_attack = 20.0  # Above stall
        assert aircraft.is_stalled

    def test_negative_stall(self, aircraft):
        """Should detect negative AoA stall."""
        aircraft._angle_of_attack = -20.0  # Negative stall
        assert aircraft.is_stalled


# =============================================================================
# Flight Phase Tests
# =============================================================================


class TestFlightPhase:
    """Tests for flight phase detection."""

    @pytest.fixture
    def aircraft(self):
        """Create an aircraft."""
        return Aircraft()

    def test_grounded_phase(self, aircraft):
        """Should be grounded when on ground."""
        aircraft._is_grounded = True
        aircraft.velocity = Vector3.zero()

        aircraft._update_flight_state()
        assert aircraft.flight_phase == FlightPhase.GROUNDED

    def test_climb_phase(self, aircraft):
        """Should detect climb."""
        aircraft._is_grounded = False
        aircraft.velocity = Vector3(0, 5.0, 50.0)  # Climbing

        aircraft._update_flight_state()
        assert aircraft.flight_phase == FlightPhase.CLIMB

    def test_descent_phase(self, aircraft):
        """Should detect descent."""
        aircraft._is_grounded = False
        aircraft.velocity = Vector3(0, -5.0, 50.0)  # Descending

        aircraft._update_flight_state()
        assert aircraft.flight_phase == FlightPhase.DESCENT

    def test_cruise_phase(self, aircraft):
        """Should detect cruise."""
        aircraft._is_grounded = False
        aircraft.velocity = Vector3(0, 0, 50.0)  # Level flight

        aircraft._update_flight_state()
        assert aircraft.flight_phase == FlightPhase.CRUISE


# =============================================================================
# Update Cycle Tests
# =============================================================================


class TestUpdateCycle:
    """Tests for complete update cycle."""

    @pytest.fixture
    def aircraft(self):
        """Create an aircraft."""
        return Aircraft()

    def test_zero_dt_no_crash(self, aircraft):
        """Zero dt should not cause errors."""
        aircraft.update(dt=0.0)

    def test_update_cycle_completes(self, aircraft):
        """Full update cycle should complete."""
        aircraft.set_throttle(0.8)
        aircraft.set_control_inputs(pitch=0.1)

        for _ in range(100):
            aircraft.update(dt=0.016)

        # Should not crash

    def test_airspeed_property(self, aircraft):
        """Airspeed should reflect velocity magnitude."""
        aircraft.velocity = Vector3(0, 0, 100.0)
        aircraft._update_flight_state()
        assert abs(aircraft.airspeed - 100.0) < 1.0

    def test_altitude_property(self, aircraft):
        """Altitude should reflect height above ground."""
        aircraft.transform.position.y = 500.0
        aircraft._ground_height = 0.0
        aircraft._update_flight_state()
        assert abs(aircraft.altitude - 500.0) < 1.0


# =============================================================================
# Force Application Tests
# =============================================================================


class TestForceApplication:
    """Tests for force and torque application."""

    @pytest.fixture
    def aircraft(self):
        """Create an aircraft."""
        return Aircraft()

    def test_apply_force(self, aircraft):
        """Should accumulate forces."""
        aircraft._accumulated_force = Vector3.zero()
        aircraft.apply_force(Vector3(1000, 0, 0))
        assert aircraft._accumulated_force.x == 1000

    def test_apply_force_with_offset(self, aircraft):
        """Off-center force should create torque."""
        aircraft._accumulated_force = Vector3.zero()
        aircraft._accumulated_torque = Vector3.zero()

        aircraft.apply_force(Vector3(0, 0, 1000), Vector3(1, 0, 0))
        assert aircraft._accumulated_torque.magnitude() > 0

    def test_apply_torque(self, aircraft):
        """Should accumulate torques."""
        aircraft._accumulated_torque = Vector3.zero()
        aircraft.apply_torque(Vector3(0, 100, 0))
        assert aircraft._accumulated_torque.y == 100


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Tests for aircraft reset."""

    @pytest.fixture
    def aircraft(self):
        """Create an aircraft."""
        return Aircraft()

    def test_reset_position(self, aircraft):
        """Reset should clear position."""
        aircraft.transform.position = Vector3(1000, 500, 2000)
        aircraft.reset()
        assert aircraft.transform.position.magnitude() == 0

    def test_reset_velocity(self, aircraft):
        """Reset should clear velocity."""
        aircraft.velocity = Vector3(100, 10, 50)
        aircraft.reset()
        assert aircraft.velocity.magnitude() == 0

    def test_reset_controls(self, aircraft):
        """Reset should clear controls."""
        aircraft.set_control_inputs(pitch=0.5, roll=0.3)
        aircraft.reset()
        assert aircraft.controls.elevator == 0.0
        assert aircraft.controls.aileron == 0.0

    def test_reset_throttle(self, aircraft):
        """Reset should clear throttle."""
        aircraft.set_throttle(0.8)
        aircraft.reset()
        for engine in aircraft._engines:
            assert engine.current_throttle == 0.0

    def test_reset_grounded(self, aircraft):
        """Reset should set grounded."""
        aircraft._is_grounded = False
        aircraft.reset()
        assert aircraft._is_grounded


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual situations."""

    def test_all_aircraft_types(self):
        """All aircraft types should work."""
        for ac_type in AircraftType:
            ac = Aircraft(aircraft_type=ac_type)
            assert ac.aircraft_type == ac_type

    def test_very_heavy_aircraft(self):
        """Very heavy aircraft should still work."""
        ac = Aircraft(mass=100000.0)
        ac.update(dt=0.016)

    def test_very_light_aircraft(self):
        """Very light aircraft should still work."""
        ac = Aircraft(mass=50.0)
        ac.update(dt=0.016)

    def test_high_altitude(self):
        """High altitude should reduce air density effects."""
        ac = Aircraft()
        ac.transform.position.y = 30000.0  # 30km
        ac._ground_height = 0.0
        ac._update_flight_state()

        # Should not crash
        ac.update(dt=0.016)

    def test_inverted_flight(self):
        """Inverted flight should still work."""
        ac = Aircraft()
        ac.transform.rotation.x = 180.0  # Inverted
        ac.update(dt=0.016)

    def test_extreme_aoa(self):
        """Extreme AoA should not crash."""
        ac = Aircraft()
        ac._angle_of_attack = 90.0  # Vertical
        ac._update_flight_state()

        # Lift coefficient should still be calculated
        q = 0.5 * 1.225 * 50.0 ** 2
        lift = ac.compute_lift(q)
        # Should be some value, likely reduced
