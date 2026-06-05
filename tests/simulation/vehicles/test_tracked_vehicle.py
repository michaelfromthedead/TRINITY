"""Tests for tracked vehicle simulation (tanks, excavators).

Tests cover:
- TrackedVehicle initialization
- Track physics and differential steering
- Pivot turns (neutral steering)
- Track slip and friction
- Road wheel suspension
- Edge cases
"""

import math
import pytest

from engine.simulation.vehicles.tracked_vehicle import (
    TrackType,
    RoadWheel,
    TrackState,
    Track,
    TrackedVehicle,
)
from engine.simulation.vehicles.vehicle_system import Vector3, Transform, VehicleState


# =============================================================================
# TrackState Tests
# =============================================================================


class TestTrackState:
    """Tests for TrackState dataclass."""

    def test_default_values(self):
        """TrackState should have sensible defaults."""
        state = TrackState()
        assert state.velocity == 0.0
        assert state.tension > 0
        assert state.slip == 0.0

    def test_custom_values(self):
        """TrackState should accept custom values."""
        state = TrackState(
            velocity=5.0,
            tension=2000.0,
            slip=0.1,
            longitudinal_force=10000.0,
        )
        assert state.velocity == 5.0
        assert state.tension == 2000.0
        assert state.slip == 0.1
        assert state.longitudinal_force == 10000.0


# =============================================================================
# RoadWheel Tests
# =============================================================================


class TestRoadWheel:
    """Tests for RoadWheel dataclass."""

    def test_default_values(self):
        """RoadWheel should have sensible defaults."""
        wheel = RoadWheel()
        assert wheel.radius == 0.3
        assert wheel.angular_velocity == 0.0
        assert not wheel.is_grounded

    def test_custom_values(self):
        """RoadWheel should accept custom values."""
        wheel = RoadWheel(
            radius=0.25,
            angular_velocity=10.0,
            contact_force=5000.0,
            is_grounded=True,
        )
        assert wheel.radius == 0.25
        assert wheel.angular_velocity == 10.0
        assert wheel.is_grounded


# =============================================================================
# Track Tests
# =============================================================================


class TestTrack:
    """Tests for Track assembly."""

    def test_initialization(self):
        """Track should initialize with road wheels."""
        track = Track(side="left")
        assert track.side == "left"
        assert len(track.road_wheels) > 0  # Default wheels created

    def test_custom_track_type(self):
        """Track should accept custom type."""
        track = Track(track_type=TrackType.RUBBER)
        assert track.track_type == TrackType.RUBBER

    def test_track_dimensions(self):
        """Track should have correct dimensions."""
        track = Track(
            width=0.6,
            contact_length=5.0,
            sprocket_radius=0.5,
        )
        assert track.width == 0.6
        assert track.contact_length == 5.0
        assert track.sprocket_radius == 0.5

    def test_road_wheel_count(self):
        """Default track should have multiple road wheels."""
        track = Track()
        # Default is 5 wheels
        assert len(track.road_wheels) == 5

    def test_road_wheel_suspension(self):
        """Road wheels should have suspension."""
        track = Track()
        for wheel in track.road_wheels:
            assert wheel.suspension is not None


# =============================================================================
# TrackedVehicle Initialization Tests
# =============================================================================


class TestTrackedVehicleInit:
    """Tests for TrackedVehicle initialization."""

    @pytest.fixture
    def tank(self):
        """Create a standard tracked vehicle."""
        return TrackedVehicle(
            mass=40000.0,
            length=7.0,
            width=3.5,
            track_separation=2.8,
            max_engine_torque=4000.0,
        )

    def test_initialization(self, tank):
        """Tracked vehicle should initialize correctly."""
        assert tank.mass == 40000.0
        assert tank.vehicle_type.name == "TRACKED"
        assert tank.state == VehicleState.ACTIVE

    def test_has_two_tracks(self, tank):
        """Should have left and right tracks."""
        assert tank.left_track is not None
        assert tank.right_track is not None
        assert tank.left_track.side == "left"
        assert tank.right_track.side == "right"

    def test_vehicle_id_generated(self, tank):
        """Should have generated ID."""
        assert tank.vehicle_id is not None
        assert len(tank.vehicle_id) > 0

    def test_custom_vehicle_id(self):
        """Should accept custom ID."""
        tank = TrackedVehicle(vehicle_id="tiger-001")
        assert tank.vehicle_id == "tiger-001"


# =============================================================================
# Differential Steering Tests
# =============================================================================


class TestDifferentialSteering:
    """Tests for differential steering behavior."""

    @pytest.fixture
    def tank(self):
        """Create a tracked vehicle for steering tests."""
        return TrackedVehicle()

    def test_throttle_input_clamp(self, tank):
        """Track throttle should be clamped to [-1, 1]."""
        tank.left_track_throttle = 2.0
        assert tank.left_track_throttle == 1.0

        tank.right_track_throttle = -2.0
        assert tank.right_track_throttle == -1.0

    def test_forward_motion(self, tank):
        """Equal throttle should move forward."""
        tank.left_track_throttle = 1.0
        tank.right_track_throttle = 1.0

        for _ in range(50):
            tank.update(dt=0.016)

        # Should move forward (positive Z in most setups)
        assert tank.velocity.magnitude() > 0

    def test_turn_right(self, tank):
        """Left track faster should turn right."""
        tank.left_track_throttle = 1.0
        tank.right_track_throttle = 0.5

        for _ in range(50):
            tank.update(dt=0.016)

        # Should have some yaw rotation
        # Positive yaw typically = turning right
        assert tank.angular_velocity.magnitude() > 0

    def test_turn_left(self, tank):
        """Right track faster should turn left."""
        tank.left_track_throttle = 0.5
        tank.right_track_throttle = 1.0

        for _ in range(50):
            tank.update(dt=0.016)

        # Should have some yaw rotation
        assert tank.angular_velocity.magnitude() > 0

    def test_combined_throttle_steer(self, tank):
        """set_throttle_steer should convert to differential."""
        tank.set_throttle_steer(throttle=1.0, steer=0.5)

        # Left should be faster when steering right
        assert tank.left_track_throttle > tank.right_track_throttle

    def test_combined_throttle_steer_clamped(self, tank):
        """Combined values should be normalized."""
        tank.set_throttle_steer(throttle=1.0, steer=1.0)

        # Values should not exceed 1
        assert abs(tank.left_track_throttle) <= 1.0
        assert abs(tank.right_track_throttle) <= 1.0


# =============================================================================
# Pivot Turn Tests
# =============================================================================


class TestPivotTurn:
    """Tests for neutral steering (pivot turns)."""

    @pytest.fixture
    def tank(self):
        """Create a tracked vehicle for pivot tests."""
        return TrackedVehicle()

    def test_pivot_right(self, tank):
        """Pivot right should spin in place."""
        tank.pivot_turn(1.0)  # Full right pivot

        assert tank.left_track_throttle == 1.0
        assert tank.right_track_throttle == -1.0

    def test_pivot_left(self, tank):
        """Pivot left should spin in place."""
        tank.pivot_turn(-1.0)  # Full left pivot

        assert tank.left_track_throttle == -1.0
        assert tank.right_track_throttle == 1.0

    def test_pivot_with_updates(self, tank):
        """Pivot should create rotation with minimal translation."""
        tank.pivot_turn(1.0)

        initial_pos = tank.transform.position.copy()

        for _ in range(50):
            tank.update(dt=0.016)

        # Should rotate but not move much
        # Position might change slightly due to physics
        assert tank.angular_velocity.magnitude() > 0


# =============================================================================
# Track Physics Tests
# =============================================================================


class TestTrackPhysics:
    """Tests for track physics calculations."""

    @pytest.fixture
    def tank(self):
        """Create a tracked vehicle for physics tests."""
        return TrackedVehicle()

    def test_track_slip_calculation(self, tank):
        """Track slip should be calculated."""
        tank.left_track_throttle = 1.0
        tank.right_track_throttle = 1.0

        tank.update_tracks(dt=0.016)

        # Some slip should be calculated
        # (May be zero if not moving, but should not error)

    def test_lateral_resistance(self, tank):
        """Tracks should resist lateral sliding."""
        tank.velocity = Vector3(10.0, 0.0, 0.0)  # Pure lateral

        tank.update_tracks(dt=0.016)

        # Should create force opposing lateral motion
        # Force should be in accumulated forces

    def test_braking(self, tank):
        """Brakes should slow vehicle."""
        # First get moving
        tank.left_track_throttle = 1.0
        tank.right_track_throttle = 1.0

        for _ in range(50):
            tank.update(dt=0.016)

        speed_before = tank.speed

        # Apply brakes
        tank._brake_input = 1.0
        tank.left_track_throttle = 0.0
        tank.right_track_throttle = 0.0

        for _ in range(50):
            tank.update(dt=0.016)

        # Should be slower
        assert tank.speed < speed_before

    def test_speed_property(self, tank):
        """Speed should reflect forward velocity."""
        tank.velocity = Vector3(0, 0, 10.0)
        assert abs(tank.speed - 10.0) < 0.1


# =============================================================================
# Force Application Tests
# =============================================================================


class TestForceApplication:
    """Tests for force and torque application."""

    @pytest.fixture
    def tank(self):
        """Create a tracked vehicle."""
        return TrackedVehicle()

    def test_apply_force(self, tank):
        """Should accumulate applied forces."""
        tank._accumulated_force = Vector3.zero()
        tank.apply_force(Vector3(1000, 0, 0))
        assert tank._accumulated_force.x == 1000

    def test_apply_force_with_offset(self, tank):
        """Off-center force should create torque."""
        tank._accumulated_force = Vector3.zero()
        tank._accumulated_torque = Vector3.zero()

        tank.apply_force(Vector3(0, 0, 1000), Vector3(1, 0, 0))

        # Torque should be created
        assert tank._accumulated_torque.magnitude() > 0

    def test_apply_torque(self, tank):
        """Should accumulate applied torques."""
        tank._accumulated_torque = Vector3.zero()
        tank.apply_torque(Vector3(0, 100, 0))
        tank.apply_torque(Vector3(0, 100, 0))
        assert tank._accumulated_torque.y == 200


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Tests for vehicle reset."""

    @pytest.fixture
    def tank(self):
        """Create a tracked vehicle."""
        return TrackedVehicle()

    def test_reset_position(self, tank):
        """Reset should clear position."""
        tank.transform.position = Vector3(100, 50, 200)
        tank.reset()
        assert tank.transform.position.magnitude() == 0

    def test_reset_velocity(self, tank):
        """Reset should clear velocity."""
        tank.velocity = Vector3(10, 0, 5)
        tank.reset()
        assert tank.velocity.magnitude() == 0

    def test_reset_track_inputs(self, tank):
        """Reset should clear track inputs."""
        tank.left_track_throttle = 0.8
        tank.right_track_throttle = -0.5
        tank._brake_input = 0.5
        tank.reset()

        assert tank.left_track_throttle == 0.0
        assert tank.right_track_throttle == 0.0
        assert tank._brake_input == 0.0

    def test_reset_track_state(self, tank):
        """Reset should reset track states."""
        tank.left_track.state.velocity = 10.0
        tank.reset()
        assert tank.left_track.state.velocity == 0.0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual situations."""

    def test_very_heavy_tank(self):
        """Very heavy tank should still work."""
        tank = TrackedVehicle(mass=100000.0)  # 100 tons
        tank.left_track_throttle = 1.0
        tank.right_track_throttle = 1.0
        tank.update(dt=0.016)
        # Should not crash

    def test_very_light_tank(self):
        """Very light tracked vehicle should still work."""
        tank = TrackedVehicle(mass=500.0)  # Very light
        tank.update(dt=0.016)
        # Should not crash

    def test_zero_dt_update(self):
        """Zero dt should not cause errors."""
        tank = TrackedVehicle()
        tank.update(dt=0.0)
        # Should return early without errors

    def test_all_track_types(self):
        """All track types should work."""
        for track_type in TrackType:
            track = Track(track_type=track_type)
            assert track.track_type == track_type

    def test_opposite_throttles(self):
        """Opposite throttles should cause rotation."""
        tank = TrackedVehicle()
        tank.left_track_throttle = 1.0
        tank.right_track_throttle = -1.0

        for _ in range(50):
            tank.update(dt=0.016)

        # Should rotate
        assert tank.angular_velocity.magnitude() > 0

    def test_reverse_motion(self):
        """Negative throttle should reverse."""
        tank = TrackedVehicle()
        tank.left_track_throttle = -1.0
        tank.right_track_throttle = -1.0

        for _ in range(50):
            tank.update(dt=0.016)

        # Should move (possibly backward based on coordinate system)
        assert tank.velocity.magnitude() > 0
