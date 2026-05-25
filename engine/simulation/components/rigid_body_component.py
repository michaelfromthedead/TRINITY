"""
Rigid Body Component.

Provides a component wrapper for rigid body physics entities,
supporting both kinematic and dynamic simulation modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..character.character_controller import Quaternion, Transform, Vector3


class RigidBodyType(str, Enum):
    """Type of rigid body simulation."""
    STATIC = "static"       # Immovable
    KINEMATIC = "kinematic" # Controlled by code, affects dynamics
    DYNAMIC = "dynamic"     # Fully simulated


class ActivationState(str, Enum):
    """Activation state of a rigid body."""
    ACTIVE = "active"
    SLEEPING = "sleeping"
    WANTS_DEACTIVATION = "wants_deactivation"
    DISABLE_DEACTIVATION = "disable_deactivation"
    DISABLE_SIMULATION = "disable_simulation"


@dataclass
class RigidBodyConfig:
    """
    Configuration for a rigid body.

    Attributes:
        mass: Mass in kg (0 for static/kinematic)
        friction: Coefficient of friction
        restitution: Coefficient of restitution (bounciness)
        linear_damping: Linear velocity damping
        angular_damping: Angular velocity damping
        gravity_scale: Gravity multiplier
        body_type: Type of rigid body
        continuous_collision: Use CCD for fast objects
        collision_group: Collision group bitmask
        collision_mask: Collision mask bitmask
    """
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.0
    linear_damping: float = 0.0
    angular_damping: float = 0.05
    gravity_scale: float = 1.0
    body_type: RigidBodyType = RigidBodyType.DYNAMIC
    continuous_collision: bool = False
    collision_group: int = 1
    collision_mask: int = 0xFFFF


@dataclass
class CollisionEvent:
    """
    Data for a collision event.

    Attributes:
        other_entity: ID of other entity
        contact_point: World position of contact
        contact_normal: Normal pointing from other to this
        impulse: Collision impulse magnitude
        relative_velocity: Relative velocity at contact
    """
    other_entity: int = 0
    contact_point: Vector3 = field(default_factory=Vector3.zero)
    contact_normal: Vector3 = field(default_factory=Vector3.up)
    impulse: float = 0.0
    relative_velocity: Vector3 = field(default_factory=Vector3.zero)


class RigidBodyComponent:
    """
    Component for rigid body physics.

    Provides:
    - Mass and inertia configuration
    - Linear and angular velocity control
    - Force and impulse application
    - Collision callbacks
    - Sleep management
    """

    def __init__(
        self,
        entity_id: int,
        config: Optional[RigidBodyConfig] = None,
    ):
        self._entity_id = entity_id
        self._config = config or RigidBodyConfig()

        # Physics state
        self._body_id: Optional[int] = None
        self._position = Vector3.zero()
        self._rotation = Quaternion.identity()
        self._linear_velocity = Vector3.zero()
        self._angular_velocity = Vector3.zero()

        # State
        self._activation_state = ActivationState.ACTIVE
        self._is_initialized = False

        # Callbacks
        self._on_collision_enter: Optional[Callable[[CollisionEvent], None]] = None
        self._on_collision_stay: Optional[Callable[[CollisionEvent], None]] = None
        self._on_collision_exit: Optional[Callable[[int], None]] = None

        # Tracking
        self._colliding_entities: set[int] = set()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def entity_id(self) -> int:
        """Entity this component belongs to."""
        return self._entity_id

    @property
    def body_id(self) -> Optional[int]:
        """Physics body ID."""
        return self._body_id

    @property
    def config(self) -> RigidBodyConfig:
        """Rigid body configuration."""
        return self._config

    @property
    def body_type(self) -> RigidBodyType:
        """Type of rigid body."""
        return self._config.body_type

    @property
    def mass(self) -> float:
        """Mass of the body."""
        return self._config.mass

    @property
    def is_kinematic(self) -> bool:
        """Whether body is kinematic."""
        return self._config.body_type == RigidBodyType.KINEMATIC

    @property
    def is_static(self) -> bool:
        """Whether body is static."""
        return self._config.body_type == RigidBodyType.STATIC

    @property
    def is_dynamic(self) -> bool:
        """Whether body is dynamic."""
        return self._config.body_type == RigidBodyType.DYNAMIC

    @property
    def is_sleeping(self) -> bool:
        """Whether body is sleeping."""
        return self._activation_state == ActivationState.SLEEPING

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------

    @property
    def position(self) -> Vector3:
        """World position."""
        return self._position

    @position.setter
    def position(self, value: Vector3) -> None:
        self._position = value

    @property
    def rotation(self) -> Quaternion:
        """World rotation."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Quaternion) -> None:
        self._rotation = value

    def get_transform(self) -> Transform:
        """Get full transform."""
        return Transform(position=self._position, rotation=self._rotation)

    def set_transform(self, transform: Transform) -> None:
        """Set full transform."""
        self._position = transform.position
        self._rotation = transform.rotation

    def teleport(self, position: Vector3, rotation: Optional[Quaternion] = None) -> None:
        """Teleport body to new transform."""
        self._position = position
        if rotation is not None:
            self._rotation = rotation
        self._linear_velocity = Vector3.zero()
        self._angular_velocity = Vector3.zero()

    # -------------------------------------------------------------------------
    # Velocity
    # -------------------------------------------------------------------------

    @property
    def linear_velocity(self) -> Vector3:
        """Linear velocity."""
        return self._linear_velocity

    @linear_velocity.setter
    def linear_velocity(self, value: Vector3) -> None:
        self._linear_velocity = value

    @property
    def angular_velocity(self) -> Vector3:
        """Angular velocity."""
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value: Vector3) -> None:
        self._angular_velocity = value

    def get_velocity_at_point(self, world_point: Vector3) -> Vector3:
        """Get velocity at a world point (includes angular contribution)."""
        to_point = world_point - self._position
        angular_contribution = self._angular_velocity.cross(to_point)
        return self._linear_velocity + angular_contribution

    # -------------------------------------------------------------------------
    # Forces
    # -------------------------------------------------------------------------

    def add_force(self, force: Vector3, mode: str = "force") -> None:
        """
        Add a force to the body.

        Args:
            force: Force vector
            mode: "force" (continuous), "impulse" (instant), "acceleration"
        """
        if self._config.body_type != RigidBodyType.DYNAMIC:
            return

        if mode == "impulse":
            self._linear_velocity = self._linear_velocity + force / self._config.mass
        elif mode == "acceleration":
            self._linear_velocity = self._linear_velocity + force
        # "force" would be accumulated and applied during physics step

    def add_force_at_position(
        self,
        force: Vector3,
        position: Vector3,
        mode: str = "force",
    ) -> None:
        """Add a force at a world position, creating torque."""
        self.add_force(force, mode)

        # Calculate torque
        to_point = position - self._position
        torque = to_point.cross(force)
        self.add_torque(torque, mode)

    def add_torque(self, torque: Vector3, mode: str = "force") -> None:
        """Add torque to the body."""
        if self._config.body_type != RigidBodyType.DYNAMIC:
            return

        if mode == "impulse":
            # Simplified - would use inertia tensor in real implementation
            self._angular_velocity = self._angular_velocity + torque

    def add_explosive_force(
        self,
        force: float,
        explosion_position: Vector3,
        radius: float,
        upward_modifier: float = 0.0,
    ) -> None:
        """Add explosive force from a point."""
        to_body = self._position - explosion_position
        distance = to_body.magnitude()

        if distance > radius or distance < 0.001:
            return

        # Falloff with distance
        falloff = 1.0 - (distance / radius)
        direction = to_body.normalized()

        # Add upward component
        direction = Vector3(
            direction.x,
            direction.y + upward_modifier,
            direction.z,
        ).normalized()

        impulse = direction * force * falloff
        self.add_force(impulse, "impulse")

    # -------------------------------------------------------------------------
    # Sleep Management
    # -------------------------------------------------------------------------

    def wake_up(self) -> None:
        """Wake up the body from sleep."""
        if self._activation_state == ActivationState.SLEEPING:
            self._activation_state = ActivationState.ACTIVE

    def put_to_sleep(self) -> None:
        """Put the body to sleep."""
        if self._activation_state != ActivationState.DISABLE_DEACTIVATION:
            self._activation_state = ActivationState.SLEEPING
            self._linear_velocity = Vector3.zero()
            self._angular_velocity = Vector3.zero()

    def set_sleep_allowed(self, allowed: bool) -> None:
        """Set whether sleeping is allowed."""
        if allowed:
            if self._activation_state == ActivationState.DISABLE_DEACTIVATION:
                self._activation_state = ActivationState.ACTIVE
        else:
            self._activation_state = ActivationState.DISABLE_DEACTIVATION

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_mass(self, mass: float) -> None:
        """Set body mass."""
        self._config.mass = max(0.0, mass)

    def set_gravity_scale(self, scale: float) -> None:
        """Set gravity scale."""
        self._config.gravity_scale = scale

    def set_friction(self, friction: float) -> None:
        """Set friction coefficient."""
        self._config.friction = max(0.0, friction)

    def set_restitution(self, restitution: float) -> None:
        """Set restitution (bounciness)."""
        self._config.restitution = max(0.0, min(1.0, restitution))

    def set_kinematic(self, kinematic: bool) -> None:
        """Set kinematic mode."""
        if kinematic:
            self._config.body_type = RigidBodyType.KINEMATIC
        else:
            self._config.body_type = RigidBodyType.DYNAMIC

    # -------------------------------------------------------------------------
    # Collision Callbacks
    # -------------------------------------------------------------------------

    def set_collision_callbacks(
        self,
        on_enter: Optional[Callable[[CollisionEvent], None]] = None,
        on_stay: Optional[Callable[[CollisionEvent], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Set collision callbacks."""
        self._on_collision_enter = on_enter
        self._on_collision_stay = on_stay
        self._on_collision_exit = on_exit

    def handle_collision(self, event: CollisionEvent) -> None:
        """Handle a collision event."""
        if event.other_entity in self._colliding_entities:
            # Collision stay
            if self._on_collision_stay:
                self._on_collision_stay(event)
        else:
            # Collision enter
            self._colliding_entities.add(event.other_entity)
            if self._on_collision_enter:
                self._on_collision_enter(event)

    def handle_collision_end(self, other_entity: int) -> None:
        """Handle end of collision."""
        if other_entity in self._colliding_entities:
            self._colliding_entities.discard(other_entity)
            if self._on_collision_exit:
                self._on_collision_exit(other_entity)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self, body_id: int) -> None:
        """Initialize with physics body ID."""
        self._body_id = body_id
        self._is_initialized = True

    def cleanup(self) -> None:
        """Cleanup component."""
        self._body_id = None
        self._is_initialized = False
        self._colliding_entities.clear()

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return {
            "entity_id": self._entity_id,
            "position": (self._position.x, self._position.y, self._position.z),
            "rotation": (
                self._rotation.x, self._rotation.y,
                self._rotation.z, self._rotation.w
            ),
            "linear_velocity": (
                self._linear_velocity.x,
                self._linear_velocity.y,
                self._linear_velocity.z,
            ),
            "angular_velocity": (
                self._angular_velocity.x,
                self._angular_velocity.y,
                self._angular_velocity.z,
            ),
            "activation_state": self._activation_state.value,
            "config": {
                "mass": self._config.mass,
                "friction": self._config.friction,
                "restitution": self._config.restitution,
                "body_type": self._config.body_type.value,
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load from serialized state."""
        pos = state.get("position", (0, 0, 0))
        self._position = Vector3(pos[0], pos[1], pos[2])

        rot = state.get("rotation", (0, 0, 0, 1))
        self._rotation = Quaternion(rot[0], rot[1], rot[2], rot[3])

        vel = state.get("linear_velocity", (0, 0, 0))
        self._linear_velocity = Vector3(vel[0], vel[1], vel[2])

        ang = state.get("angular_velocity", (0, 0, 0))
        self._angular_velocity = Vector3(ang[0], ang[1], ang[2])

        self._activation_state = ActivationState(
            state.get("activation_state", "active")
        )

        config = state.get("config", {})
        self._config.mass = config.get("mass", 1.0)
        self._config.friction = config.get("friction", 0.5)
        self._config.restitution = config.get("restitution", 0.0)
        self._config.body_type = RigidBodyType(config.get("body_type", "dynamic"))
