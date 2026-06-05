"""
Whitebox tests for VehicleComponent.

Tests cover:
- Vehicle creation and configuration
- Wheel setup and management
- Engine and gearbox configuration
- Input handling
- Gear shifting
- Driver assists
- Serialization
"""

import pytest

from engine.simulation.character.character_controller import Vector3
from engine.simulation.components.vehicle_component import (
    DriveType,
    EngineConfig,
    GearboxConfig,
    VehicleComponent,
    VehicleType,
    WheelConfig,
    WheelState,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def vehicle() -> VehicleComponent:
    """Create a basic vehicle component."""
    return VehicleComponent(
        entity_id=1,
        vehicle_type=VehicleType.CAR,
        drive_type=DriveType.REAR_WHEEL,
    )


@pytest.fixture
def car_with_wheels() -> VehicleComponent:
    """Create a car with wheels set up."""
    vehicle = VehicleComponent(
        entity_id=2,
        vehicle_type=VehicleType.CAR,
        drive_type=DriveType.ALL_WHEEL,
    )
    vehicle.setup_car(wheelbase=2.5, track=1.6)
    return vehicle


@pytest.fixture
def engine_config() -> EngineConfig:
    """Custom engine configuration."""
    return EngineConfig(
        max_rpm=8000.0,
        idle_rpm=1000.0,
        max_torque=500.0,
    )


@pytest.fixture
def gearbox_config() -> GearboxConfig:
    """Custom gearbox configuration."""
    return GearboxConfig(
        gear_ratios=[-3.0, 0.0, 3.0, 2.0, 1.5, 1.0],
        final_drive=4.0,
        shift_time=0.15,
        auto_shift=False,
    )


# =============================================================================
# VehicleType Tests
# =============================================================================


class TestVehicleType:
    """Tests for VehicleType enum."""

    def test_all_types(self):
        """Test all vehicle types exist."""
        assert VehicleType.CAR.value == "car"
        assert VehicleType.MOTORCYCLE.value == "motorcycle"
        assert VehicleType.TRUCK.value == "truck"
        assert VehicleType.TANK.value == "tank"
        assert VehicleType.HOVERCRAFT.value == "hovercraft"
        assert VehicleType.BOAT.value == "boat"


class TestDriveType:
    """Tests for DriveType enum."""

    def test_all_types(self):
        """Test all drive types exist."""
        assert DriveType.FRONT_WHEEL.value == "fwd"
        assert DriveType.REAR_WHEEL.value == "rwd"
        assert DriveType.ALL_WHEEL.value == "awd"
        assert DriveType.TANK.value == "tank"


# =============================================================================
# WheelConfig Tests
# =============================================================================


class TestWheelConfig:
    """Tests for WheelConfig dataclass."""

    def test_default_values(self):
        """Test default wheel configuration."""
        config = WheelConfig()

        assert config.position.magnitude() == 0.0
        assert config.radius == 0.4
        assert config.width == 0.2
        assert config.suspension_travel == 0.3
        assert config.suspension_stiffness == 5000.0
        assert config.suspension_damping == 500.0
        assert config.friction == 1.0
        assert config.is_steering is False
        assert config.is_powered is True

    def test_custom_values(self):
        """Test custom wheel configuration."""
        config = WheelConfig(
            position=Vector3(1.0, 0.0, 2.0),
            radius=0.5,
            is_steering=True,
            is_powered=False,
        )

        assert config.position.x == 1.0
        assert config.radius == 0.5
        assert config.is_steering is True
        assert config.is_powered is False


# =============================================================================
# WheelState Tests
# =============================================================================


class TestWheelState:
    """Tests for WheelState dataclass."""

    def test_default_values(self):
        """Test default wheel state."""
        state = WheelState()

        assert state.rotation == 0.0
        assert state.rpm == 0.0
        assert state.slip_angle == 0.0
        assert state.slip_ratio == 0.0
        assert state.is_grounded is False
        assert state.suspension_compression == 0.0

    def test_custom_values(self):
        """Test custom wheel state."""
        state = WheelState(
            rotation=1.5,
            rpm=1000.0,
            slip_angle=0.1,
            is_grounded=True,
            suspension_compression=0.15,
        )

        assert state.rotation == 1.5
        assert state.rpm == 1000.0
        assert state.is_grounded is True


# =============================================================================
# EngineConfig Tests
# =============================================================================


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_default_values(self):
        """Test default engine configuration."""
        config = EngineConfig()

        assert config.max_rpm == 7000.0
        assert config.idle_rpm == 800.0
        assert config.max_torque == 400.0
        assert config.inertia == 0.3
        assert len(config.torque_curve) == 5

    def test_custom_values(self, engine_config):
        """Test custom engine configuration."""
        assert engine_config.max_rpm == 8000.0
        assert engine_config.idle_rpm == 1000.0
        assert engine_config.max_torque == 500.0


# =============================================================================
# GearboxConfig Tests
# =============================================================================


class TestGearboxConfig:
    """Tests for GearboxConfig dataclass."""

    def test_default_values(self):
        """Test default gearbox configuration."""
        config = GearboxConfig()

        assert len(config.gear_ratios) == 8
        assert config.gear_ratios[0] < 0  # Reverse
        assert config.gear_ratios[1] == 0.0  # Neutral
        assert config.final_drive == 3.5
        assert config.shift_time == 0.2
        assert config.auto_shift is True

    def test_custom_values(self, gearbox_config):
        """Test custom gearbox configuration."""
        assert len(gearbox_config.gear_ratios) == 6
        assert gearbox_config.final_drive == 4.0
        assert gearbox_config.auto_shift is False


# =============================================================================
# Vehicle Creation Tests
# =============================================================================


class TestVehicleCreation:
    """Tests for vehicle creation."""

    def test_create_with_defaults(self, vehicle):
        """Test creating vehicle with defaults."""
        assert vehicle.entity_id == 1
        assert vehicle.vehicle_type == VehicleType.CAR
        assert vehicle.drive_type == DriveType.REAR_WHEEL
        assert vehicle.enabled is True

    def test_initial_state(self, vehicle):
        """Test initial vehicle state."""
        assert vehicle.speed == 0.0
        assert vehicle.engine_rpm == 0.0
        assert vehicle.current_gear == 1
        assert vehicle.wheel_count == 0


# =============================================================================
# Car Setup Tests
# =============================================================================


class TestCarSetup:
    """Tests for car setup."""

    def test_setup_car(self, vehicle):
        """Test setting up standard car."""
        vehicle.setup_car(wheelbase=2.5, track=1.6)

        assert vehicle.wheel_count == 4

    def test_wheel_positions(self, car_with_wheels):
        """Test wheel positions are correct."""
        # Front left
        fl_pos = car_with_wheels.get_wheel_position(0)
        assert fl_pos is not None
        assert fl_pos.x < 0  # Left side
        assert fl_pos.z > 0  # Front

        # Front right
        fr_pos = car_with_wheels.get_wheel_position(1)
        assert fr_pos is not None
        assert fr_pos.x > 0  # Right side
        assert fr_pos.z > 0  # Front

    def test_rwd_wheel_power(self):
        """Test RWD has powered rear wheels."""
        vehicle = VehicleComponent(
            entity_id=1,
            drive_type=DriveType.REAR_WHEEL,
        )
        vehicle.setup_car()

        # Front wheels not powered
        assert vehicle._wheel_configs[0].is_powered is False
        assert vehicle._wheel_configs[1].is_powered is False

        # Rear wheels powered
        assert vehicle._wheel_configs[2].is_powered is True
        assert vehicle._wheel_configs[3].is_powered is True

    def test_fwd_wheel_power(self):
        """Test FWD has powered front wheels."""
        vehicle = VehicleComponent(
            entity_id=1,
            drive_type=DriveType.FRONT_WHEEL,
        )
        vehicle.setup_car()

        # Front wheels powered
        assert vehicle._wheel_configs[0].is_powered is True
        assert vehicle._wheel_configs[1].is_powered is True

        # Rear wheels not powered
        assert vehicle._wheel_configs[2].is_powered is False
        assert vehicle._wheel_configs[3].is_powered is False

    def test_awd_wheel_power(self, car_with_wheels):
        """Test AWD has all wheels powered."""
        for i in range(4):
            assert car_with_wheels._wheel_configs[i].is_powered is True

    def test_steering_wheels(self, car_with_wheels):
        """Test front wheels steer."""
        # Front wheels steer
        assert car_with_wheels._wheel_configs[0].is_steering is True
        assert car_with_wheels._wheel_configs[1].is_steering is True

        # Rear wheels don't steer
        assert car_with_wheels._wheel_configs[2].is_steering is False
        assert car_with_wheels._wheel_configs[3].is_steering is False


# =============================================================================
# Wheel Management Tests
# =============================================================================


class TestWheelManagement:
    """Tests for wheel management."""

    def test_add_wheel(self, vehicle):
        """Test adding individual wheel."""
        config = WheelConfig(
            position=Vector3(0.0, 0.0, 1.0),
            radius=0.5,
        )

        index = vehicle.add_wheel(config)

        assert index == 0
        assert vehicle.wheel_count == 1

    def test_set_wheel_config(self, car_with_wheels):
        """Test setting wheel configuration."""
        new_config = WheelConfig(radius=0.6, friction=1.5)

        car_with_wheels.set_wheel_config(0, new_config)

        assert car_with_wheels._wheel_configs[0].radius == 0.6

    def test_get_wheel_state(self, car_with_wheels):
        """Test getting wheel state."""
        state = car_with_wheels.get_wheel_state(0)

        assert state is not None
        assert isinstance(state, WheelState)

    def test_get_wheel_state_invalid(self, car_with_wheels):
        """Test getting invalid wheel state."""
        state = car_with_wheels.get_wheel_state(99)

        assert state is None

    def test_is_any_wheel_grounded(self, car_with_wheels):
        """Test checking if any wheel is grounded."""
        # Initially not grounded
        assert car_with_wheels.is_any_wheel_grounded() is False

        # Set one wheel grounded
        car_with_wheels._wheel_states[0].is_grounded = True
        assert car_with_wheels.is_any_wheel_grounded() is True

    def test_get_grounded_wheel_count(self, car_with_wheels):
        """Test getting grounded wheel count."""
        assert car_with_wheels.get_grounded_wheel_count() == 0

        car_with_wheels._wheel_states[0].is_grounded = True
        car_with_wheels._wheel_states[2].is_grounded = True

        assert car_with_wheels.get_grounded_wheel_count() == 2

    def test_average_suspension_compression(self, car_with_wheels):
        """Test getting average suspension compression."""
        car_with_wheels._wheel_states[0].suspension_compression = 0.1
        car_with_wheels._wheel_states[1].suspension_compression = 0.2
        car_with_wheels._wheel_states[2].suspension_compression = 0.1
        car_with_wheels._wheel_states[3].suspension_compression = 0.2

        avg = car_with_wheels.get_average_suspension_compression()

        assert abs(avg - 0.15) < 0.001


# =============================================================================
# Engine Configuration Tests
# =============================================================================


class TestEngineConfiguration:
    """Tests for engine configuration."""

    def test_set_engine_config(self, vehicle, engine_config):
        """Test setting engine configuration."""
        vehicle.set_engine_config(engine_config)

        assert vehicle._engine.max_rpm == 8000.0
        assert vehicle._engine.max_torque == 500.0

    def test_get_engine_torque(self, vehicle):
        """Test getting engine torque."""
        vehicle._throttle = 1.0

        # At mid RPM
        torque = vehicle.get_engine_torque(3500.0)
        assert torque > 0.0

    def test_engine_torque_at_idle(self, vehicle):
        """Test engine torque at idle."""
        vehicle._throttle = 1.0

        torque = vehicle.get_engine_torque(800.0)
        assert torque >= 0.0

    def test_engine_torque_below_idle(self, vehicle):
        """Test engine torque below idle is zero."""
        vehicle._throttle = 1.0

        torque = vehicle.get_engine_torque(500.0)
        assert torque == 0.0

    def test_engine_torque_above_max(self, vehicle):
        """Test engine torque above max RPM is zero."""
        vehicle._throttle = 1.0

        torque = vehicle.get_engine_torque(8000.0)
        assert torque == 0.0

    def test_engine_torque_scales_with_throttle(self, vehicle):
        """Test torque scales with throttle."""
        vehicle._throttle = 0.5
        torque_half = vehicle.get_engine_torque(3500.0)

        vehicle._throttle = 1.0
        torque_full = vehicle.get_engine_torque(3500.0)

        assert abs(torque_full - torque_half * 2) < 1.0


# =============================================================================
# Gearbox Configuration Tests
# =============================================================================


class TestGearboxConfiguration:
    """Tests for gearbox configuration."""

    def test_set_gearbox_config(self, vehicle, gearbox_config):
        """Test setting gearbox configuration."""
        vehicle.set_gearbox_config(gearbox_config)

        assert len(vehicle._gearbox.gear_ratios) == 6
        assert vehicle._gearbox.auto_shift is False

    def test_get_wheel_torque(self, vehicle):
        """Test getting wheel torque."""
        vehicle._throttle = 1.0
        vehicle._engine_rpm = 3500.0
        vehicle._current_gear = 2  # First gear

        torque = vehicle.get_wheel_torque()

        assert torque > 0.0

    def test_wheel_torque_in_neutral(self, vehicle):
        """Test no wheel torque in neutral."""
        vehicle._throttle = 1.0
        vehicle._engine_rpm = 3500.0
        vehicle._current_gear = 0  # Neutral

        torque = vehicle.get_wheel_torque()

        assert torque == 0.0

    def test_wheel_torque_while_shifting(self, vehicle):
        """Test no wheel torque while shifting."""
        vehicle._throttle = 1.0
        vehicle._engine_rpm = 3500.0
        vehicle._current_gear = 2
        vehicle._shifting = True

        torque = vehicle.get_wheel_torque()

        assert torque == 0.0


# =============================================================================
# Input Tests
# =============================================================================


class TestInput:
    """Tests for input handling."""

    def test_set_input(self, vehicle):
        """Test setting vehicle input."""
        vehicle.set_input(
            throttle=0.8,
            brake=0.2,
            steering=-0.5,
            handbrake=True,
        )

        assert vehicle._throttle == 0.8
        assert vehicle._brake == 0.2
        assert vehicle._steering == -0.5
        assert vehicle._handbrake is True

    def test_input_clamped(self, vehicle):
        """Test input is clamped to valid range."""
        vehicle.set_input(
            throttle=2.0,
            brake=-0.5,
            steering=3.0,
        )

        assert vehicle._throttle == 1.0
        assert vehicle._brake == 0.0
        assert vehicle._steering == 1.0


# =============================================================================
# Gear Shifting Tests
# =============================================================================


class TestGearShifting:
    """Tests for gear shifting."""

    def test_shift_up(self, vehicle):
        """Test shifting up."""
        vehicle._current_gear = 2

        result = vehicle.shift_up()

        assert result is True
        assert vehicle._current_gear == 3
        assert vehicle._shifting is True

    def test_shift_up_at_max(self, vehicle):
        """Test can't shift up at max gear."""
        vehicle._current_gear = len(vehicle._gearbox.gear_ratios) - 2

        result = vehicle.shift_up()

        assert result is False

    def test_shift_up_while_shifting(self, vehicle):
        """Test can't shift up while already shifting."""
        vehicle._shifting = True

        result = vehicle.shift_up()

        assert result is False

    def test_shift_down(self, vehicle):
        """Test shifting down."""
        vehicle._current_gear = 3

        result = vehicle.shift_down()

        assert result is True
        assert vehicle._current_gear == 2

    def test_shift_down_at_reverse(self, vehicle):
        """Test can't shift down past reverse."""
        vehicle._current_gear = -1

        result = vehicle.shift_down()

        assert result is False

    def test_shift_to(self, vehicle):
        """Test shifting to specific gear."""
        result = vehicle.shift_to(4)

        assert result is True
        assert vehicle._current_gear == 4

    def test_shift_to_invalid(self, vehicle):
        """Test shifting to invalid gear fails."""
        result = vehicle.shift_to(99)

        assert result is False

    def test_gear_change_callback(self, vehicle):
        """Test gear change callback."""
        gears_changed = []
        vehicle.set_gear_change_callback(lambda g: gears_changed.append(g))

        vehicle.shift_up()

        assert len(gears_changed) == 1


# =============================================================================
# Mass and Configuration Tests
# =============================================================================


class TestMassAndConfig:
    """Tests for mass and configuration."""

    def test_set_mass(self, vehicle):
        """Test setting vehicle mass."""
        vehicle.set_mass(2000.0)

        assert vehicle._mass == 2000.0

    def test_set_mass_clamped(self, vehicle):
        """Test mass is clamped to minimum."""
        vehicle.set_mass(50.0)

        assert vehicle._mass == 100.0

    def test_set_center_of_mass(self, vehicle):
        """Test setting center of mass."""
        com = Vector3(0.0, 0.5, -0.2)
        vehicle.set_center_of_mass(com)

        assert vehicle._center_of_mass.y == 0.5
        assert vehicle._center_of_mass.z == -0.2


# =============================================================================
# Driver Assists Tests
# =============================================================================


class TestDriverAssists:
    """Tests for driver assists."""

    def test_default_assists(self, vehicle):
        """Test default assists are enabled."""
        assert vehicle._traction_control is True
        assert vehicle._stability_control is True
        assert vehicle._abs_enabled is True

    def test_set_assists(self, vehicle):
        """Test setting assists."""
        vehicle.set_assists(
            traction_control=False,
            stability_control=False,
            abs_enabled=False,
        )

        assert vehicle._traction_control is False
        assert vehicle._stability_control is False
        assert vehicle._abs_enabled is False


# =============================================================================
# Speed Tests
# =============================================================================


class TestSpeed:
    """Tests for speed properties."""

    def test_speed_properties(self, vehicle):
        """Test speed conversion properties."""
        vehicle._speed = 10.0  # m/s

        assert vehicle.speed == 10.0
        assert abs(vehicle.speed_kmh - 36.0) < 0.1
        assert abs(vehicle.speed_mph - 22.37) < 0.1


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for component lifecycle."""

    def test_initialize(self, vehicle):
        """Test initialization with physics IDs."""
        vehicle.initialize(vehicle_id=42, body_id=100)

        assert vehicle._vehicle_id == 42
        assert vehicle._body_id == 100

    def test_cleanup(self, vehicle):
        """Test cleanup."""
        vehicle.initialize(vehicle_id=42, body_id=100)
        vehicle.cleanup()

        assert vehicle._vehicle_id is None
        assert vehicle._body_id is None

    def test_enabled_property(self, vehicle):
        """Test enabled property."""
        assert vehicle.enabled is True

        vehicle.enabled = False
        assert vehicle.enabled is False


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for callbacks."""

    def test_wheel_contact_callback(self, car_with_wheels):
        """Test wheel contact callback."""
        contacts = []
        car_with_wheels.set_wheel_contact_callback(
            lambda idx, pos: contacts.append((idx, pos))
        )

        # Callback would be triggered by physics system
        # Just verify it's set
        assert car_with_wheels._on_wheel_contact is not None


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for state serialization."""

    def test_get_state(self, car_with_wheels):
        """Test getting serializable state."""
        car_with_wheels.set_input(throttle=0.5, steering=0.3)
        car_with_wheels._speed = 15.0
        car_with_wheels._engine_rpm = 4000.0

        state = car_with_wheels.get_state()

        assert state["entity_id"] == 2
        assert state["vehicle_type"] == "car"
        assert state["drive_type"] == "awd"
        assert state["wheel_count"] == 4
        assert state["speed"] == 15.0
        assert state["engine_rpm"] == 4000.0
        assert state["input"]["throttle"] == 0.5
        assert state["input"]["steering"] == 0.3


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_vehicle(self):
        """Test vehicle with no wheels."""
        vehicle = VehicleComponent(entity_id=1)

        assert vehicle.wheel_count == 0
        assert vehicle.get_wheel_state(0) is None
        assert vehicle.is_any_wheel_grounded() is False

    def test_single_wheel(self):
        """Test single-wheel vehicle (unicycle)."""
        vehicle = VehicleComponent(entity_id=1)
        vehicle.add_wheel(WheelConfig(is_steering=True, is_powered=True))

        assert vehicle.wheel_count == 1

    def test_many_wheels(self):
        """Test vehicle with many wheels (truck)."""
        vehicle = VehicleComponent(
            entity_id=1,
            vehicle_type=VehicleType.TRUCK,
        )

        for i in range(8):
            vehicle.add_wheel(WheelConfig(
                position=Vector3(float(i % 2 - 0.5), 0.0, float(i // 2)),
            ))

        assert vehicle.wheel_count == 8

    def test_motorcycle(self):
        """Test motorcycle vehicle type."""
        vehicle = VehicleComponent(
            entity_id=1,
            vehicle_type=VehicleType.MOTORCYCLE,
            drive_type=DriveType.REAR_WHEEL,
        )

        # Add two wheels
        vehicle.add_wheel(WheelConfig(position=Vector3(0.0, 0.0, 1.0), is_steering=True))
        vehicle.add_wheel(WheelConfig(position=Vector3(0.0, 0.0, -1.0), is_powered=True))

        assert vehicle.vehicle_type == VehicleType.MOTORCYCLE
        assert vehicle.wheel_count == 2

    def test_tank_drive(self):
        """Test tank drive type."""
        vehicle = VehicleComponent(
            entity_id=1,
            vehicle_type=VehicleType.TANK,
            drive_type=DriveType.TANK,
        )

        assert vehicle.drive_type == DriveType.TANK

    def test_empty_wheel_states(self):
        """Test average suspension with no wheels."""
        vehicle = VehicleComponent(entity_id=1)

        avg = vehicle.get_average_suspension_compression()

        assert avg == 0.0

    def test_zero_throttle_torque(self, vehicle):
        """Test zero throttle produces no torque."""
        vehicle._throttle = 0.0
        vehicle._engine_rpm = 3500.0

        torque = vehicle.get_engine_torque(3500.0)

        assert torque == 0.0
