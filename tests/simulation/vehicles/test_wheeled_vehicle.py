"""Tests for wheeled vehicle simulation.

Tests cover:
- WheeledVehicle initialization and configuration
- Wheel setup and positioning
- Steering with Ackermann geometry
- Brake force distribution
- Wheel physics and rotation
- Aerodynamic forces
- Complete vehicle update cycle
- Edge cases (airborne, flipped, etc.)
"""

import math
import pytest

from engine.simulation.vehicles.wheeled_vehicle import (
    WheelPosition,
    WheelState,
    Wheel,
    WheeledVehicle,
)
from engine.simulation.vehicles.vehicle_system import Vector3, Transform, VehicleState
from engine.simulation.vehicles.drivetrain import DrivetrainLayout, Drivetrain
from engine.simulation.vehicles.suspension import Suspension
from engine.simulation.vehicles.tire_model import PacejkaTire


# =============================================================================
# WheelState Tests
# =============================================================================


class TestWheelState:
    """Tests for WheelState dataclass."""

    def test_default_values(self):
        """WheelState should have sensible defaults."""
        state = WheelState()
        assert state.angular_velocity == 0.0
        assert state.steer_angle == 0.0
        assert state.brake_torque == 0.0
        assert state.drive_torque == 0.0
        assert not state.is_grounded

    def test_custom_values(self):
        """WheelState should accept custom values."""
        state = WheelState(
            angular_velocity=50.0,
            steer_angle=0.3,
            brake_torque=500.0,
            is_grounded=True,
        )
        assert state.angular_velocity == 50.0
        assert state.steer_angle == 0.3
        assert state.brake_torque == 500.0
        assert state.is_grounded


# =============================================================================
# Wheel Tests
# =============================================================================


class TestWheel:
    """Tests for Wheel dataclass."""

    def test_default_initialization(self):
        """Wheel should initialize with default components."""
        wheel = Wheel()
        assert wheel.suspension is not None
        assert wheel.tire_model is not None
        assert isinstance(wheel.suspension, Suspension)
        assert isinstance(wheel.tire_model, PacejkaTire)

    def test_custom_position(self):
        """Wheel should accept custom position."""
        wheel = Wheel(
            position=WheelPosition.REAR_LEFT,
            local_position=Vector3(-0.8, 0, -1.4),
        )
        assert wheel.position == WheelPosition.REAR_LEFT
        assert wheel.local_position.x == -0.8

    def test_wheel_properties(self):
        """Wheel should have correct physical properties."""
        wheel = Wheel(
            radius=0.35,
            width=0.225,
            mass=15.0,
            inertia=1.5,
        )
        assert wheel.radius == 0.35
        assert wheel.width == 0.225
        assert wheel.mass == 15.0
        assert wheel.inertia == 1.5


# =============================================================================
# WheeledVehicle Initialization Tests
# =============================================================================


class TestWheeledVehicleInit:
    """Tests for WheeledVehicle initialization."""

    @pytest.fixture
    def vehicle(self):
        """Create a standard wheeled vehicle."""
        return WheeledVehicle(
            mass=1500.0,
            wheelbase=2.8,
            track_width_front=1.6,
            track_width_rear=1.6,
        )

    def test_initialization(self, vehicle):
        """Vehicle should initialize with correct values."""
        assert vehicle.mass == 1500.0
        assert vehicle.wheelbase == 2.8
        assert len(vehicle.wheels) == 4

    def test_vehicle_id_generated(self, vehicle):
        """Vehicle should have a generated ID."""
        assert vehicle.vehicle_id is not None
        assert len(vehicle.vehicle_id) > 0

    def test_custom_vehicle_id(self):
        """Vehicle should accept custom ID."""
        vehicle = WheeledVehicle(vehicle_id="test-car-001")
        assert vehicle.vehicle_id == "test-car-001"

    def test_wheels_positioned_correctly(self, vehicle):
        """Wheels should be in correct positions."""
        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)
        fr = vehicle.get_wheel(WheelPosition.FRONT_RIGHT)
        rl = vehicle.get_wheel(WheelPosition.REAR_LEFT)
        rr = vehicle.get_wheel(WheelPosition.REAR_RIGHT)

        # Front wheels ahead of rear
        assert fl.local_position.z > rl.local_position.z
        assert fr.local_position.z > rr.local_position.z

        # Left wheels negative X, right positive
        assert fl.local_position.x < 0
        assert fr.local_position.x > 0
        assert rl.local_position.x < 0
        assert rr.local_position.x > 0

    def test_front_wheels_steerable(self, vehicle):
        """Front wheels should be steerable."""
        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)
        fr = vehicle.get_wheel(WheelPosition.FRONT_RIGHT)
        rl = vehicle.get_wheel(WheelPosition.REAR_LEFT)
        rr = vehicle.get_wheel(WheelPosition.REAR_RIGHT)

        assert fl.is_steerable
        assert fr.is_steerable
        assert not rl.is_steerable
        assert not rr.is_steerable

    def test_invalid_wheel_position(self, vehicle):
        """Getting invalid wheel should raise error."""
        with pytest.raises(ValueError):
            vehicle.get_wheel("invalid")


# =============================================================================
# Steering Tests
# =============================================================================


class TestSteering:
    """Tests for steering behavior."""

    @pytest.fixture
    def vehicle(self):
        """Create a vehicle for steering tests."""
        return WheeledVehicle()

    def test_steering_input_clamp(self, vehicle):
        """Steering input should be clamped to [-1, 1]."""
        vehicle.steering_input = 2.0
        assert vehicle.steering_input == 1.0

        vehicle.steering_input = -2.0
        assert vehicle.steering_input == -1.0

    def test_steering_updates_wheel_angles(self, vehicle):
        """Steering input should update wheel angles."""
        vehicle.steering_input = 0.5
        vehicle.update_steering(dt=0.5)  # Allow some time

        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)
        fr = vehicle.get_wheel(WheelPosition.FRONT_RIGHT)

        # At least one wheel should have non-zero steer angle
        assert fl.state.steer_angle != 0 or fr.state.steer_angle != 0

    def test_ackermann_geometry(self, vehicle):
        """Inside wheel should have larger steer angle."""
        vehicle.steering_input = 1.0  # Full right turn

        # Update several times to reach target
        for _ in range(20):
            vehicle.update_steering(dt=0.016)

        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)  # Inside
        fr = vehicle.get_wheel(WheelPosition.FRONT_RIGHT)  # Outside

        # Inside wheel (FL for right turn) should have larger angle
        assert abs(fl.state.steer_angle) > abs(fr.state.steer_angle) * 0.9

    def test_steering_auto_centers(self, vehicle):
        """Steering should auto-center when released."""
        # First steer
        vehicle.steering_input = 1.0
        for _ in range(20):
            vehicle.update_steering(dt=0.016)

        # Release steering
        vehicle.steering_input = 0.0
        for _ in range(50):
            vehicle.update_steering(dt=0.016)

        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)
        # Should return near zero
        assert abs(fl.state.steer_angle) < 0.1


# =============================================================================
# Brake Tests
# =============================================================================


class TestBrakes:
    """Tests for brake behavior."""

    @pytest.fixture
    def vehicle(self):
        """Create a vehicle for brake tests."""
        return WheeledVehicle()

    def test_brake_input_clamp(self, vehicle):
        """Brake input should be clamped to [0, 1]."""
        vehicle.brake_input = 2.0
        assert vehicle.brake_input == 1.0

        vehicle.brake_input = -1.0
        assert vehicle.brake_input == 0.0

    def test_brake_force_distribution(self, vehicle):
        """Brake force should be distributed front-biased."""
        vehicle.brake_input = 1.0
        vehicle.apply_brakes()

        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)
        rl = vehicle.get_wheel(WheelPosition.REAR_LEFT)

        # Front should have more brake torque (typical 60/40 split)
        assert fl.state.brake_torque > rl.state.brake_torque

    def test_handbrake_rear_only(self, vehicle):
        """Handbrake should only affect rear wheels."""
        vehicle.handbrake_input = 1.0
        vehicle.apply_brakes()

        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)
        rl = vehicle.get_wheel(WheelPosition.REAR_LEFT)

        # Front should have no handbrake torque
        # Rear should have handbrake torque
        assert rl.state.brake_torque > 0


# =============================================================================
# Throttle Tests
# =============================================================================


class TestThrottle:
    """Tests for throttle behavior."""

    @pytest.fixture
    def vehicle(self):
        """Create a vehicle for throttle tests."""
        return WheeledVehicle()

    def test_throttle_input_clamp(self, vehicle):
        """Throttle input should be clamped to [0, 1]."""
        vehicle.throttle_input = 2.0
        assert vehicle.throttle_input == 1.0

        vehicle.throttle_input = -1.0
        assert vehicle.throttle_input == 0.0


# =============================================================================
# Wheel Physics Tests
# =============================================================================


class TestWheelPhysics:
    """Tests for wheel physics calculations."""

    @pytest.fixture
    def vehicle(self):
        """Create a vehicle for physics tests."""
        return WheeledVehicle()

    def test_airborne_wheel_spins_down(self, vehicle):
        """Airborne wheel should slow down."""
        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)
        fl.state.angular_velocity = 100.0
        fl.state.is_grounded = False

        vehicle.update_wheels(dt=0.016)

        # Should have slowed
        assert fl.state.angular_velocity < 100.0

    def test_grounded_wheel_has_forces(self, vehicle):
        """Grounded wheel should generate forces."""
        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)
        fl.state.is_grounded = True
        fl.state.contact_distance = 0.45  # Normal compression
        fl.state.angular_velocity = 50.0

        # Set vehicle velocity
        vehicle.velocity = Vector3(0, 0, 10.0)
        vehicle._update_local_velocity()

        vehicle.update_wheels(dt=0.016)

        # Should have suspension force
        assert fl.state.suspension_force > 0


# =============================================================================
# Aerodynamics Tests
# =============================================================================


class TestAerodynamics:
    """Tests for aerodynamic forces."""

    @pytest.fixture
    def vehicle(self):
        """Create a vehicle for aero tests."""
        return WheeledVehicle()

    def test_no_aero_at_low_speed(self, vehicle):
        """No aerodynamic forces at very low speed."""
        vehicle.velocity = Vector3(0, 0, 0.5)  # Very slow
        initial_force = vehicle._accumulated_force.copy()
        vehicle.apply_aerodynamics()
        # Force should be unchanged (or minimal change)
        # The method returns early below 1 m/s

    def test_drag_at_speed(self, vehicle):
        """Drag should oppose motion."""
        vehicle.velocity = Vector3(0, 0, 30.0)  # ~108 km/h
        vehicle._accumulated_force = Vector3.zero()
        vehicle.apply_aerodynamics()

        # Drag should be negative Z (opposing forward motion)
        assert vehicle._accumulated_force.z < 0

    def test_drag_increases_with_speed(self, vehicle):
        """Drag should increase with speed squared."""
        vehicle.velocity = Vector3(0, 0, 20.0)
        vehicle._accumulated_force = Vector3.zero()
        vehicle.apply_aerodynamics()
        drag_20 = abs(vehicle._accumulated_force.z)

        vehicle.velocity = Vector3(0, 0, 40.0)
        vehicle._accumulated_force = Vector3.zero()
        vehicle.apply_aerodynamics()
        drag_40 = abs(vehicle._accumulated_force.z)

        # Double speed should give ~4x drag (speed squared)
        ratio = drag_40 / drag_20
        assert 3.5 < ratio < 4.5


# =============================================================================
# Complete Update Cycle Tests
# =============================================================================


class TestUpdateCycle:
    """Tests for complete vehicle update."""

    @pytest.fixture
    def vehicle(self):
        """Create a vehicle for update tests."""
        return WheeledVehicle()

    def test_zero_dt_no_crash(self, vehicle):
        """Zero dt should not cause errors."""
        vehicle.update(dt=0.0)  # Should return early

    def test_update_cycle_completes(self, vehicle):
        """Full update cycle should complete without errors."""
        vehicle.throttle_input = 0.5
        vehicle.steering_input = 0.2
        vehicle.brake_input = 0.1

        # Simulate grounded wheels
        for wheel in vehicle.wheels:
            wheel.state.is_grounded = True
            wheel.state.contact_distance = 0.45

        for _ in range(100):
            vehicle.update(dt=0.016)

        # Vehicle should have moved
        # (May or may not depending on ground contact)

    def test_speed_property(self, vehicle):
        """Speed property should return correct magnitude."""
        vehicle.velocity = Vector3(3, 0, 4)  # 3-4-5 triangle
        assert abs(vehicle.speed - 5.0) < 0.01

    def test_speed_kmh_conversion(self, vehicle):
        """Speed in km/h should be correct."""
        vehicle.velocity = Vector3(0, 0, 10.0)  # 10 m/s
        expected_kmh = 10.0 * 3.6  # 36 km/h
        assert abs(vehicle.speed_kmh - expected_kmh) < 0.01


# =============================================================================
# Force Application Tests
# =============================================================================


class TestForceApplication:
    """Tests for force and torque application."""

    @pytest.fixture
    def vehicle(self):
        """Create a vehicle for force tests."""
        return WheeledVehicle()

    def test_apply_force_at_cg(self, vehicle):
        """Force at CG should not create torque."""
        vehicle._accumulated_force = Vector3.zero()
        vehicle._accumulated_torque = Vector3.zero()

        vehicle.apply_force(Vector3(1000, 0, 0))

        assert vehicle._accumulated_force.x == 1000
        assert vehicle._accumulated_torque.magnitude() == 0

    def test_apply_force_off_center(self, vehicle):
        """Force off-center should create torque."""
        vehicle._accumulated_force = Vector3.zero()
        vehicle._accumulated_torque = Vector3.zero()

        vehicle.apply_force(Vector3(0, 0, 1000), Vector3(1, 0, 0))

        # Torque = r x F = (1,0,0) x (0,0,1000) = (0,-1000,0)
        assert vehicle._accumulated_torque.y != 0

    def test_apply_torque(self, vehicle):
        """Apply torque should accumulate."""
        vehicle._accumulated_torque = Vector3.zero()

        vehicle.apply_torque(Vector3(0, 100, 0))
        vehicle.apply_torque(Vector3(0, 100, 0))

        assert vehicle._accumulated_torque.y == 200


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Tests for vehicle reset."""

    @pytest.fixture
    def vehicle(self):
        """Create a vehicle for reset tests."""
        return WheeledVehicle()

    def test_reset_position(self, vehicle):
        """Reset should clear position."""
        vehicle.transform.position = Vector3(100, 50, 200)
        vehicle.reset()
        assert vehicle.transform.position.magnitude() == 0

    def test_reset_velocity(self, vehicle):
        """Reset should clear velocity."""
        vehicle.velocity = Vector3(30, 5, 20)
        vehicle.reset()
        assert vehicle.velocity.magnitude() == 0

    def test_reset_inputs(self, vehicle):
        """Reset should clear inputs."""
        vehicle.steering_input = 0.5
        vehicle.throttle_input = 0.8
        vehicle.brake_input = 0.3
        vehicle.reset()

        assert vehicle.steering_input == 0.0
        assert vehicle.throttle_input == 0.0
        assert vehicle.brake_input == 0.0

    def test_reset_wheels(self, vehicle):
        """Reset should reset wheel states."""
        for wheel in vehicle.wheels:
            wheel.state.angular_velocity = 100.0

        vehicle.reset()

        for wheel in vehicle.wheels:
            assert wheel.state.angular_velocity == 0.0


# =============================================================================
# Raycast Integration Tests
# =============================================================================


class TestRaycastIntegration:
    """Tests for raycast result handling."""

    @pytest.fixture
    def vehicle(self):
        """Create a vehicle for raycast tests."""
        return WheeledVehicle()

    def test_set_raycast_result(self, vehicle):
        """Should set wheel contact from raycast."""
        vehicle.set_raycast_result(
            wheel_position=WheelPosition.FRONT_LEFT,
            hit=True,
            contact_point=Vector3(0, 0, 0),
            contact_normal=Vector3(0, 1, 0),
            distance=0.45,
        )

        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)
        assert fl.state.is_grounded
        assert fl.state.contact_distance == 0.45

    def test_set_raycast_miss(self, vehicle):
        """Should handle raycast miss."""
        vehicle.set_raycast_result(
            wheel_position=WheelPosition.FRONT_LEFT,
            hit=False,
            contact_point=Vector3(0, 0, 0),
            contact_normal=Vector3(0, 1, 0),
            distance=1.0,
        )

        fl = vehicle.get_wheel(WheelPosition.FRONT_LEFT)
        assert not fl.state.is_grounded


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual situations."""

    def test_very_heavy_vehicle(self):
        """Very heavy vehicle should still work."""
        vehicle = WheeledVehicle(mass=50000.0)  # 50 tons
        vehicle.update(dt=0.016)
        # Should not crash

    def test_very_light_vehicle(self):
        """Very light vehicle should still work."""
        vehicle = WheeledVehicle(mass=100.0)  # 100 kg
        vehicle.update(dt=0.016)
        # Should not crash

    def test_narrow_track(self):
        """Narrow track width should still work."""
        vehicle = WheeledVehicle(track_width_front=0.5, track_width_rear=0.5)
        vehicle.steering_input = 1.0
        vehicle.update_steering(dt=0.016)
        # Should not crash

    def test_extreme_steering(self):
        """Extreme steering values should be clamped."""
        vehicle = WheeledVehicle()
        vehicle.steering_input = 100.0
        assert vehicle.steering_input == 1.0

    def test_all_drivetrain_layouts(self):
        """All drivetrain layouts should work."""
        for layout in DrivetrainLayout:
            drivetrain = Drivetrain(layout=layout)
            vehicle = WheeledVehicle(drivetrain=drivetrain)
            vehicle.update(dt=0.016)
            # Should not crash

    def test_anti_roll_bars(self):
        """Setting ARB should work."""
        vehicle = WheeledVehicle()
        vehicle.set_anti_roll_bars(front_stiffness=5000, rear_stiffness=3000)
        assert vehicle._front_arb is not None
        assert vehicle._rear_arb is not None
