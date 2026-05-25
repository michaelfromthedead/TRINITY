"""
Vehicle simulation system manager.

This module provides the main VehicleSystem class that manages all vehicles
in the simulation, handling registration, updates, and lifecycle management.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
)

from .config import (
    GRAVITY,
    PHYSICS_SUBSTEPS,
    VELOCITY_SLEEP_THRESHOLD,
    ANGULAR_SLEEP_THRESHOLD,
)

if TYPE_CHECKING:
    from .wheeled_vehicle import WheeledVehicle
    from .tracked_vehicle import TrackedVehicle
    from .hover_vehicle import HoverVehicle
    from .aircraft import Aircraft
    from .watercraft import Watercraft


class VehicleType(Enum):
    """Types of vehicles supported by the simulation."""

    WHEELED = auto()
    TRACKED = auto()
    HOVER = auto()
    AIRCRAFT = auto()
    WATERCRAFT = auto()


class VehicleState(Enum):
    """Vehicle simulation states."""

    ACTIVE = auto()
    SLEEPING = auto()
    DISABLED = auto()
    DESTROYED = auto()


@dataclass
class Vector3:
    """Simple 3D vector for physics calculations."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def dot(self, other: "Vector3") -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        """Cross product."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        """Vector magnitude (length)."""
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def magnitude_squared(self) -> float:
        """Squared magnitude (avoids sqrt)."""
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def normalized(self) -> "Vector3":
        """Return unit vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return self / mag

    def copy(self) -> "Vector3":
        """Create a copy of this vector."""
        return Vector3(self.x, self.y, self.z)

    @classmethod
    def zero(cls) -> "Vector3":
        """Return zero vector."""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def up(cls) -> "Vector3":
        """Return up vector (Y-up convention)."""
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def forward(cls) -> "Vector3":
        """Return forward vector (Z-forward convention)."""
        return cls(0.0, 0.0, 1.0)

    @classmethod
    def right(cls) -> "Vector3":
        """Return right vector (X-right convention)."""
        return cls(1.0, 0.0, 0.0)


@dataclass
class Transform:
    """3D transform with position, rotation (Euler), and scale."""

    position: Vector3 = field(default_factory=Vector3.zero)
    rotation: Vector3 = field(default_factory=Vector3.zero)  # Euler angles in degrees
    scale: Vector3 = field(default_factory=lambda: Vector3(1.0, 1.0, 1.0))

    def copy(self) -> "Transform":
        """Create a deep copy."""
        return Transform(
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            scale=self.scale.copy(),
        )


class VehicleBase(Protocol):
    """Protocol defining the vehicle interface."""

    vehicle_id: str
    vehicle_type: VehicleType
    state: VehicleState
    transform: Transform
    velocity: Vector3
    angular_velocity: Vector3
    mass: float

    def update(self, dt: float) -> None:
        """Update vehicle physics for one frame."""
        ...

    def apply_force(self, force: Vector3, position: Optional[Vector3] = None) -> None:
        """Apply force to vehicle at optional position."""
        ...

    def apply_torque(self, torque: Vector3) -> None:
        """Apply torque to vehicle."""
        ...

    def reset(self) -> None:
        """Reset vehicle to initial state."""
        ...


T = TypeVar("T", bound=VehicleBase)


@dataclass
class VehicleGroup:
    """Group of vehicles for batch processing."""

    name: str
    vehicle_ids: Set[str] = field(default_factory=set)
    enabled: bool = True

    def add(self, vehicle_id: str) -> None:
        """Add vehicle to group."""
        self.vehicle_ids.add(vehicle_id)

    def remove(self, vehicle_id: str) -> None:
        """Remove vehicle from group."""
        self.vehicle_ids.discard(vehicle_id)


@dataclass
class CollisionInfo:
    """Information about a vehicle collision."""

    vehicle_a_id: str
    vehicle_b_id: Optional[str]  # None if collision with static geometry
    contact_point: Vector3
    contact_normal: Vector3
    penetration_depth: float
    relative_velocity: float


class VehicleSystem:
    """
    Main vehicle simulation system.

    Manages all vehicles in the simulation, handling registration,
    physics updates, collision detection, and lifecycle management.
    """

    def __init__(
        self,
        gravity: float = GRAVITY,
        substeps: int = PHYSICS_SUBSTEPS,
        enable_sleeping: bool = True,
    ):
        """
        Initialize the vehicle system.

        Args:
            gravity: Gravitational acceleration (m/s^2).
            substeps: Number of physics substeps per update.
            enable_sleeping: Whether to enable sleeping for inactive vehicles.
        """
        self._vehicles: Dict[str, VehicleBase] = {}
        self._vehicle_types: Dict[str, VehicleType] = {}
        self._groups: Dict[str, VehicleGroup] = {}

        self._gravity = gravity
        self._substeps = substeps
        self._enable_sleeping = enable_sleeping

        # Performance tracking
        self._total_updates = 0
        self._active_vehicle_count = 0

        # Collision handling
        self._collision_callbacks: List[Callable[[CollisionInfo], None]] = []
        self._collision_pairs: Set[Tuple[str, str]] = set()

        # Vehicle factory registry
        self._factories: Dict[VehicleType, Type[VehicleBase]] = {}

        # Events
        self._on_vehicle_added: List[Callable[[str], None]] = []
        self._on_vehicle_removed: List[Callable[[str], None]] = []
        self._on_vehicle_state_changed: List[Callable[[str, VehicleState], None]] = []

    @property
    def gravity(self) -> float:
        """Gravitational acceleration."""
        return self._gravity

    @gravity.setter
    def gravity(self, value: float) -> None:
        """Set gravitational acceleration."""
        if value < 0:
            raise ValueError("Gravity must be non-negative")
        self._gravity = value

    @property
    def substeps(self) -> int:
        """Number of physics substeps."""
        return self._substeps

    @substeps.setter
    def substeps(self, value: int) -> None:
        """Set number of substeps."""
        if value < 1:
            raise ValueError("Substeps must be at least 1")
        self._substeps = value

    @property
    def vehicle_count(self) -> int:
        """Total number of registered vehicles."""
        return len(self._vehicles)

    @property
    def active_vehicle_count(self) -> int:
        """Number of active (non-sleeping) vehicles."""
        return self._active_vehicle_count

    def register_factory(
        self,
        vehicle_type: VehicleType,
        factory_class: Type[VehicleBase],
    ) -> None:
        """
        Register a vehicle factory class for a type.

        Args:
            vehicle_type: The vehicle type.
            factory_class: The class to instantiate for this type.
        """
        self._factories[vehicle_type] = factory_class

    def create_vehicle(
        self,
        vehicle_type: VehicleType,
        **kwargs: Any,
    ) -> str:
        """
        Create a new vehicle using registered factory.

        Args:
            vehicle_type: Type of vehicle to create.
            **kwargs: Arguments passed to vehicle constructor.

        Returns:
            The vehicle ID.

        Raises:
            ValueError: If no factory registered for vehicle type.
        """
        if vehicle_type not in self._factories:
            raise ValueError(f"No factory registered for vehicle type: {vehicle_type}")

        factory = self._factories[vehicle_type]
        vehicle = factory(**kwargs)
        return self.register_vehicle(vehicle)

    def register_vehicle(self, vehicle: VehicleBase) -> str:
        """
        Register a vehicle with the system.

        Args:
            vehicle: The vehicle to register.

        Returns:
            The vehicle's unique ID.

        Raises:
            ValueError: If vehicle with same ID already registered.
        """
        vehicle_id = vehicle.vehicle_id
        if vehicle_id in self._vehicles:
            raise ValueError(f"Vehicle with ID '{vehicle_id}' already registered")

        self._vehicles[vehicle_id] = vehicle
        self._vehicle_types[vehicle_id] = vehicle.vehicle_type

        # Notify listeners
        for callback in self._on_vehicle_added:
            callback(vehicle_id)

        return vehicle_id

    def unregister_vehicle(self, vehicle_id: str) -> bool:
        """
        Unregister a vehicle from the system.

        Args:
            vehicle_id: The vehicle's ID.

        Returns:
            True if vehicle was removed, False if not found.
        """
        if vehicle_id not in self._vehicles:
            return False

        del self._vehicles[vehicle_id]
        del self._vehicle_types[vehicle_id]

        # Remove from all groups
        for group in self._groups.values():
            group.remove(vehicle_id)

        # Notify listeners
        for callback in self._on_vehicle_removed:
            callback(vehicle_id)

        return True

    def get_vehicle(self, vehicle_id: str) -> Optional[VehicleBase]:
        """
        Get a vehicle by ID.

        Args:
            vehicle_id: The vehicle's ID.

        Returns:
            The vehicle, or None if not found.
        """
        return self._vehicles.get(vehicle_id)

    def get_vehicles_by_type(self, vehicle_type: VehicleType) -> List[VehicleBase]:
        """
        Get all vehicles of a specific type.

        Args:
            vehicle_type: The vehicle type to filter by.

        Returns:
            List of vehicles matching the type.
        """
        return [
            v for v_id, v in self._vehicles.items()
            if self._vehicle_types.get(v_id) == vehicle_type
        ]

    def get_vehicles_in_group(self, group_name: str) -> List[VehicleBase]:
        """
        Get all vehicles in a group.

        Args:
            group_name: Name of the group.

        Returns:
            List of vehicles in the group.
        """
        group = self._groups.get(group_name)
        if group is None:
            return []
        return [
            self._vehicles[v_id]
            for v_id in group.vehicle_ids
            if v_id in self._vehicles
        ]

    def create_group(self, name: str) -> VehicleGroup:
        """
        Create a new vehicle group.

        Args:
            name: Name of the group.

        Returns:
            The created group.

        Raises:
            ValueError: If group with name already exists.
        """
        if name in self._groups:
            raise ValueError(f"Group '{name}' already exists")
        group = VehicleGroup(name=name)
        self._groups[name] = group
        return group

    def add_to_group(self, vehicle_id: str, group_name: str) -> bool:
        """
        Add a vehicle to a group.

        Args:
            vehicle_id: The vehicle's ID.
            group_name: Name of the group.

        Returns:
            True if added, False if vehicle or group not found.
        """
        if vehicle_id not in self._vehicles:
            return False
        group = self._groups.get(group_name)
        if group is None:
            return False
        group.add(vehicle_id)
        return True

    def remove_from_group(self, vehicle_id: str, group_name: str) -> bool:
        """
        Remove a vehicle from a group.

        Args:
            vehicle_id: The vehicle's ID.
            group_name: Name of the group.

        Returns:
            True if removed, False if not in group.
        """
        group = self._groups.get(group_name)
        if group is None:
            return False
        if vehicle_id not in group.vehicle_ids:
            return False
        group.remove(vehicle_id)
        return True

    def update(self, dt: float) -> None:
        """
        Update all vehicles for one frame.

        Performs physics substeps for stability.

        Args:
            dt: Delta time in seconds.
        """
        if dt <= 0:
            return

        substep_dt = dt / self._substeps
        self._active_vehicle_count = 0

        for _ in range(self._substeps):
            self._update_substep(substep_dt)

        self._total_updates += 1

    def _update_substep(self, dt: float) -> None:
        """
        Perform a single physics substep.

        Args:
            dt: Substep delta time.
        """
        for vehicle in self._vehicles.values():
            # Skip inactive vehicles
            if vehicle.state != VehicleState.ACTIVE:
                continue

            # Apply gravity
            gravity_force = Vector3(0, -self._gravity * vehicle.mass, 0)
            vehicle.apply_force(gravity_force)

            # Update vehicle physics
            vehicle.update(dt)

            # Check for sleeping
            if self._enable_sleeping:
                self._check_sleeping(vehicle)

            self._active_vehicle_count += 1

    def _check_sleeping(self, vehicle: VehicleBase) -> None:
        """
        Check if vehicle should go to sleep.

        Args:
            vehicle: The vehicle to check.
        """
        vel_mag = vehicle.velocity.magnitude()
        ang_vel_mag = vehicle.angular_velocity.magnitude()

        if (
            vel_mag < VELOCITY_SLEEP_THRESHOLD and
            ang_vel_mag < ANGULAR_SLEEP_THRESHOLD
        ):
            self._set_vehicle_state(vehicle.vehicle_id, VehicleState.SLEEPING)

    def wake_vehicle(self, vehicle_id: str) -> bool:
        """
        Wake a sleeping vehicle.

        Args:
            vehicle_id: The vehicle's ID.

        Returns:
            True if woken, False if not found or not sleeping.
        """
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None or vehicle.state != VehicleState.SLEEPING:
            return False
        return self._set_vehicle_state(vehicle_id, VehicleState.ACTIVE)

    def _set_vehicle_state(
        self,
        vehicle_id: str,
        new_state: VehicleState,
    ) -> bool:
        """
        Set vehicle state and notify listeners.

        Args:
            vehicle_id: The vehicle's ID.
            new_state: The new state.

        Returns:
            True if state changed.
        """
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            return False

        old_state = vehicle.state
        if old_state == new_state:
            return False

        vehicle.state = new_state

        for callback in self._on_vehicle_state_changed:
            callback(vehicle_id, new_state)

        return True

    def register_collision_callback(
        self,
        callback: Callable[[CollisionInfo], None],
    ) -> None:
        """
        Register a callback for collision events.

        Args:
            callback: Function called when collision occurs.
        """
        self._collision_callbacks.append(callback)

    def notify_collision(self, collision: CollisionInfo) -> None:
        """
        Notify listeners of a collision.

        Args:
            collision: Collision information.
        """
        for callback in self._collision_callbacks:
            callback(collision)

    def on_vehicle_added(self, callback: Callable[[str], None]) -> None:
        """Register callback for vehicle added events."""
        self._on_vehicle_added.append(callback)

    def on_vehicle_removed(self, callback: Callable[[str], None]) -> None:
        """Register callback for vehicle removed events."""
        self._on_vehicle_removed.append(callback)

    def on_vehicle_state_changed(
        self,
        callback: Callable[[str, VehicleState], None],
    ) -> None:
        """Register callback for vehicle state change events."""
        self._on_vehicle_state_changed.append(callback)

    def iter_vehicles(self) -> Iterator[VehicleBase]:
        """Iterate over all vehicles."""
        yield from self._vehicles.values()

    def iter_active_vehicles(self) -> Iterator[VehicleBase]:
        """Iterate over active vehicles only."""
        for vehicle in self._vehicles.values():
            if vehicle.state == VehicleState.ACTIVE:
                yield vehicle

    def clear(self) -> None:
        """Remove all vehicles from the system."""
        vehicle_ids = list(self._vehicles.keys())
        for vehicle_id in vehicle_ids:
            self.unregister_vehicle(vehicle_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary of statistics.
        """
        type_counts = {}
        state_counts = {}

        for v_id, vehicle in self._vehicles.items():
            v_type = self._vehicle_types.get(v_id)
            if v_type:
                type_counts[v_type.name] = type_counts.get(v_type.name, 0) + 1
            state_counts[vehicle.state.name] = (
                state_counts.get(vehicle.state.name, 0) + 1
            )

        return {
            "total_vehicles": len(self._vehicles),
            "active_vehicles": self._active_vehicle_count,
            "total_updates": self._total_updates,
            "groups": len(self._groups),
            "vehicles_by_type": type_counts,
            "vehicles_by_state": state_counts,
            "gravity": self._gravity,
            "substeps": self._substeps,
            "sleeping_enabled": self._enable_sleeping,
        }


def generate_vehicle_id() -> str:
    """Generate a unique vehicle ID."""
    return str(uuid.uuid4())
