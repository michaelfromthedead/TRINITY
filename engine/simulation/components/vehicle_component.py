"""
Vehicle Component.

Provides vehicle physics simulation component supporting wheeled vehicles,
suspension, engine simulation, and vehicle dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..character.character_controller import Quaternion, Vector3


class VehicleType(str, Enum):
    """Type of vehicle."""
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    TRUCK = "truck"
    TANK = "tank"
    HOVERCRAFT = "hovercraft"
    BOAT = "boat"


class DriveType(str, Enum):
    """Drive configuration."""
    FRONT_WHEEL = "fwd"
    REAR_WHEEL = "rwd"
    ALL_WHEEL = "awd"
    TANK = "tank"


@dataclass
class WheelConfig:
    """
    Configuration for a single wheel.

    Attributes:
        position: Local position relative to vehicle
        radius: Wheel radius
        width: Wheel width
        suspension_travel: Maximum suspension travel
        suspension_stiffness: Spring stiffness
        suspension_damping: Damping coefficient
        friction: Tire friction
        is_steering: Whether wheel steers
        is_powered: Whether wheel receives power
    """
    position: Vector3 = field(default_factory=Vector3.zero)
    radius: float = 0.4
    width: float = 0.2
    suspension_travel: float = 0.3
    suspension_stiffness: float = 5000.0
    suspension_damping: float = 500.0
    friction: float = 1.0
    is_steering: bool = False
    is_powered: bool = True


@dataclass
class WheelState:
    """
    Runtime state of a wheel.

    Attributes:
        rotation: Current rotation angle
        rpm: Rotations per minute
        slip_angle: Lateral slip angle
        slip_ratio: Longitudinal slip ratio
        contact_position: Ground contact position
        contact_normal: Ground contact normal
        is_grounded: Whether wheel is touching ground
        suspension_compression: Current suspension compression
    """
    rotation: float = 0.0
    rpm: float = 0.0
    slip_angle: float = 0.0
    slip_ratio: float = 0.0
    contact_position: Vector3 = field(default_factory=Vector3.zero)
    contact_normal: Vector3 = field(default_factory=Vector3.up)
    is_grounded: bool = False
    suspension_compression: float = 0.0


@dataclass
class EngineConfig:
    """
    Engine configuration.

    Attributes:
        max_rpm: Maximum engine RPM
        idle_rpm: Idle RPM
        max_torque: Maximum torque (Nm)
        torque_curve: RPM to torque multiplier curve
        inertia: Engine rotational inertia
    """
    max_rpm: float = 7000.0
    idle_rpm: float = 800.0
    max_torque: float = 400.0
    torque_curve: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.5), (0.33, 0.8), (0.5, 1.0), (0.75, 0.95), (1.0, 0.7)
    ])
    inertia: float = 0.3


@dataclass
class GearboxConfig:
    """
    Gearbox configuration.

    Attributes:
        gear_ratios: Gear ratios (index 0 = reverse)
        final_drive: Final drive ratio
        shift_time: Time to shift gears
        auto_shift: Enable automatic shifting
    """
    gear_ratios: list[float] = field(default_factory=lambda: [
        -3.5, 0.0, 3.5, 2.5, 1.8, 1.4, 1.1, 0.9
    ])
    final_drive: float = 3.5
    shift_time: float = 0.2
    auto_shift: bool = True


class VehicleComponent:
    """
    Component for vehicle physics simulation.

    Provides:
    - Wheeled vehicle dynamics
    - Suspension simulation
    - Engine and transmission
    - Steering and braking
    - Stability control
    """

    def __init__(
        self,
        entity_id: int,
        vehicle_type: VehicleType = VehicleType.CAR,
        drive_type: DriveType = DriveType.REAR_WHEEL,
    ):
        self._entity_id = entity_id
        self._vehicle_type = vehicle_type
        self._drive_type = drive_type

        # Configuration
        self._mass = 1500.0
        self._center_of_mass = Vector3(0.0, 0.3, 0.0)
        self._engine = EngineConfig()
        self._gearbox = GearboxConfig()

        # Wheels
        self._wheel_configs: list[WheelConfig] = []
        self._wheel_states: list[WheelState] = []

        # Input state
        self._throttle = 0.0
        self._brake = 0.0
        self._steering = 0.0
        self._handbrake = False

        # Engine state
        self._engine_rpm = 0.0
        self._current_gear = 1
        self._shifting = False
        self._shift_timer = 0.0

        # Vehicle state
        self._velocity = Vector3.zero()
        self._angular_velocity = Vector3.zero()
        self._speed = 0.0
        self._forward_speed = 0.0

        # Assists
        self._traction_control = True
        self._stability_control = True
        self._abs_enabled = True

        # State
        self._vehicle_id: Optional[int] = None
        self._body_id: Optional[int] = None
        self._enabled = True

        # Callbacks
        self._on_gear_change: Optional[Callable[[int], None]] = None
        self._on_wheel_contact: Optional[Callable[[int, Vector3], None]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def entity_id(self) -> int:
        """Entity this vehicle belongs to."""
        return self._entity_id

    @property
    def vehicle_type(self) -> VehicleType:
        """Type of vehicle."""
        return self._vehicle_type

    @property
    def drive_type(self) -> DriveType:
        """Drive configuration."""
        return self._drive_type

    @property
    def speed(self) -> float:
        """Current speed in m/s."""
        return self._speed

    @property
    def speed_kmh(self) -> float:
        """Current speed in km/h."""
        return self._speed * 3.6

    @property
    def speed_mph(self) -> float:
        """Current speed in mph."""
        return self._speed * 2.237

    @property
    def forward_speed(self) -> float:
        """Forward velocity component."""
        return self._forward_speed

    @property
    def engine_rpm(self) -> float:
        """Current engine RPM."""
        return self._engine_rpm

    @property
    def current_gear(self) -> int:
        """Current gear (0 = neutral, -1 = reverse)."""
        return self._current_gear

    @property
    def wheel_count(self) -> int:
        """Number of wheels."""
        return len(self._wheel_configs)

    @property
    def enabled(self) -> bool:
        """Whether vehicle simulation is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def setup_car(self, wheelbase: float = 2.5, track: float = 1.6) -> None:
        """
        Setup a standard 4-wheel car.

        Args:
            wheelbase: Distance between front and rear axles
            track: Distance between left and right wheels
        """
        self._wheel_configs = [
            # Front left
            WheelConfig(
                position=Vector3(-track / 2, 0, wheelbase / 2),
                is_steering=True,
                is_powered=self._drive_type in (DriveType.FRONT_WHEEL, DriveType.ALL_WHEEL),
            ),
            # Front right
            WheelConfig(
                position=Vector3(track / 2, 0, wheelbase / 2),
                is_steering=True,
                is_powered=self._drive_type in (DriveType.FRONT_WHEEL, DriveType.ALL_WHEEL),
            ),
            # Rear left
            WheelConfig(
                position=Vector3(-track / 2, 0, -wheelbase / 2),
                is_steering=False,
                is_powered=self._drive_type in (DriveType.REAR_WHEEL, DriveType.ALL_WHEEL),
            ),
            # Rear right
            WheelConfig(
                position=Vector3(track / 2, 0, -wheelbase / 2),
                is_steering=False,
                is_powered=self._drive_type in (DriveType.REAR_WHEEL, DriveType.ALL_WHEEL),
            ),
        ]

        self._wheel_states = [WheelState() for _ in self._wheel_configs]

    def add_wheel(self, config: WheelConfig) -> int:
        """
        Add a wheel to the vehicle.

        Returns:
            Index of added wheel
        """
        self._wheel_configs.append(config)
        self._wheel_states.append(WheelState())
        return len(self._wheel_configs) - 1

    def set_wheel_config(self, index: int, config: WheelConfig) -> None:
        """Set configuration for a specific wheel."""
        if 0 <= index < len(self._wheel_configs):
            self._wheel_configs[index] = config

    def set_engine_config(self, config: EngineConfig) -> None:
        """Set engine configuration."""
        self._engine = config

    def set_gearbox_config(self, config: GearboxConfig) -> None:
        """Set gearbox configuration."""
        self._gearbox = config

    def set_mass(self, mass: float) -> None:
        """Set vehicle mass."""
        self._mass = max(100.0, mass)

    def set_center_of_mass(self, com: Vector3) -> None:
        """Set center of mass offset."""
        self._center_of_mass = com

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------

    def set_input(
        self,
        throttle: float = 0.0,
        brake: float = 0.0,
        steering: float = 0.0,
        handbrake: bool = False,
    ) -> None:
        """
        Set vehicle input.

        Args:
            throttle: Throttle (0-1)
            brake: Brake (0-1)
            steering: Steering (-1 to 1)
            handbrake: Handbrake engaged
        """
        self._throttle = max(0.0, min(1.0, throttle))
        self._brake = max(0.0, min(1.0, brake))
        self._steering = max(-1.0, min(1.0, steering))
        self._handbrake = handbrake

    def shift_up(self) -> bool:
        """Shift to higher gear."""
        max_gear = len(self._gearbox.gear_ratios) - 2  # Exclude reverse
        if self._current_gear < max_gear and not self._shifting:
            self._start_shift(self._current_gear + 1)
            return True
        return False

    def shift_down(self) -> bool:
        """Shift to lower gear."""
        if self._current_gear > -1 and not self._shifting:
            self._start_shift(self._current_gear - 1)
            return True
        return False

    def shift_to(self, gear: int) -> bool:
        """Shift to specific gear."""
        if -1 <= gear <= len(self._gearbox.gear_ratios) - 2 and not self._shifting:
            self._start_shift(gear)
            return True
        return False

    def _start_shift(self, target_gear: int) -> None:
        """Start gear shift."""
        self._shifting = True
        self._shift_timer = self._gearbox.shift_time
        self._current_gear = target_gear

        if self._on_gear_change:
            self._on_gear_change(target_gear)

    # -------------------------------------------------------------------------
    # Assists
    # -------------------------------------------------------------------------

    def set_assists(
        self,
        traction_control: bool = True,
        stability_control: bool = True,
        abs_enabled: bool = True,
    ) -> None:
        """Set driver assists."""
        self._traction_control = traction_control
        self._stability_control = stability_control
        self._abs_enabled = abs_enabled

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_wheel_state(self, index: int) -> Optional[WheelState]:
        """Get state of a specific wheel."""
        if 0 <= index < len(self._wheel_states):
            return self._wheel_states[index]
        return None

    def get_wheel_position(self, index: int) -> Optional[Vector3]:
        """Get world position of wheel."""
        if 0 <= index < len(self._wheel_configs):
            # Would transform local position to world
            return self._wheel_configs[index].position
        return None

    def is_any_wheel_grounded(self) -> bool:
        """Check if any wheel is touching ground."""
        return any(ws.is_grounded for ws in self._wheel_states)

    def get_grounded_wheel_count(self) -> int:
        """Get number of grounded wheels."""
        return sum(1 for ws in self._wheel_states if ws.is_grounded)

    def get_average_suspension_compression(self) -> float:
        """Get average suspension compression across all wheels."""
        if not self._wheel_states:
            return 0.0
        return sum(ws.suspension_compression for ws in self._wheel_states) / len(self._wheel_states)

    # -------------------------------------------------------------------------
    # Engine Calculations
    # -------------------------------------------------------------------------

    def get_engine_torque(self, rpm: float) -> float:
        """Calculate engine torque at given RPM."""
        if rpm < self._engine.idle_rpm or rpm > self._engine.max_rpm:
            return 0.0

        # Normalize RPM
        normalized = (rpm - self._engine.idle_rpm) / (self._engine.max_rpm - self._engine.idle_rpm)

        # Interpolate torque curve
        multiplier = 1.0
        for i in range(len(self._engine.torque_curve) - 1):
            if self._engine.torque_curve[i][0] <= normalized <= self._engine.torque_curve[i + 1][0]:
                t = (normalized - self._engine.torque_curve[i][0]) / (
                    self._engine.torque_curve[i + 1][0] - self._engine.torque_curve[i][0]
                )
                multiplier = (
                    self._engine.torque_curve[i][1] * (1 - t) +
                    self._engine.torque_curve[i + 1][1] * t
                )
                break

        return self._engine.max_torque * multiplier * self._throttle

    def get_wheel_torque(self) -> float:
        """Calculate torque delivered to driven wheels."""
        if self._current_gear == 0 or self._shifting:
            return 0.0

        gear_index = self._current_gear if self._current_gear >= 0 else 0
        gear_ratio = self._gearbox.gear_ratios[gear_index]
        engine_torque = self.get_engine_torque(self._engine_rpm)

        return engine_torque * gear_ratio * self._gearbox.final_drive

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_gear_change_callback(
        self, callback: Optional[Callable[[int], None]]
    ) -> None:
        """Set callback for gear changes."""
        self._on_gear_change = callback

    def set_wheel_contact_callback(
        self, callback: Optional[Callable[[int, Vector3], None]]
    ) -> None:
        """Set callback for wheel contact events."""
        self._on_wheel_contact = callback

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self, vehicle_id: int, body_id: int) -> None:
        """Initialize with physics IDs."""
        self._vehicle_id = vehicle_id
        self._body_id = body_id

    def cleanup(self) -> None:
        """Cleanup component."""
        self._vehicle_id = None
        self._body_id = None

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return {
            "entity_id": self._entity_id,
            "vehicle_type": self._vehicle_type.value,
            "drive_type": self._drive_type.value,
            "mass": self._mass,
            "wheel_count": len(self._wheel_configs),
            "engine_rpm": self._engine_rpm,
            "current_gear": self._current_gear,
            "speed": self._speed,
            "enabled": self._enabled,
            "input": {
                "throttle": self._throttle,
                "brake": self._brake,
                "steering": self._steering,
                "handbrake": self._handbrake,
            },
        }


__all__ = [
    "VehicleType",
    "DriveType",
    "WheelConfig",
    "WheelState",
    "EngineConfig",
    "GearboxConfig",
    "VehicleComponent",
]
