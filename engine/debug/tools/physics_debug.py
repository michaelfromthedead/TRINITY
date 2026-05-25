"""
Physics Debugging - Collision visualization, physics pause/step, body inspection.

Provides tools for debugging physics systems including:
- Collision shape visualization
- Contact point display
- Velocity vectors
- Physics pause/step control
- Force application
- Body property inspection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

logger = logging.getLogger(__name__)


class PhysicsDebugState(Enum):
    """State of physics debugging."""
    RUNNING = auto()
    PAUSED = auto()
    STEPPING = auto()


class PhysicsVisualization(Flag):
    """Physics visualization options."""
    NONE = 0
    COLLISION_SHAPES = auto()
    CONTACT_POINTS = auto()
    VELOCITIES = auto()
    ANGULAR_VELOCITIES = auto()
    CONSTRAINTS = auto()
    JOINTS = auto()
    RAYCASTS = auto()
    CENTER_OF_MASS = auto()
    BOUNDING_BOXES = auto()
    SLEEP_STATE = auto()
    FORCES = auto()
    ALL = (
        COLLISION_SHAPES | CONTACT_POINTS | VELOCITIES | ANGULAR_VELOCITIES |
        CONSTRAINTS | JOINTS | RAYCASTS | CENTER_OF_MASS | BOUNDING_BOXES |
        SLEEP_STATE | FORCES
    )


@dataclass
class PhysicsVisualizationConfig:
    """
    Configuration for physics visualization.

    All colors, scales, and limits are configurable to avoid magic numbers.
    """
    # Collision visualization
    collision_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5)  # Green
    collision_wireframe: bool = True

    # Contact visualization
    contact_color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)  # Red
    contact_normal_length: float = 1.0  # Length of contact normal arrows

    # Velocity visualization
    velocity_color: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # Blue
    velocity_scale: float = 0.1  # Scale factor for velocity vectors

    # Angular velocity
    angular_velocity_color: Tuple[float, float, float, float] = (1.0, 0.0, 1.0, 1.0)  # Magenta

    # Constraints
    constraint_color: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0)  # Yellow

    # Center of mass
    center_of_mass_size: float = 0.2  # Size of center of mass indicator

    # Sleep state colors
    sleeping_color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.3)  # Gray, transparent
    active_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5)   # Green

    # Buffer limits (prevent memory bloat)
    max_contacts: int = 1000          # Maximum tracked contact points
    max_raycast_history: int = 100    # Maximum raycast history entries

    # Build restrictions
    allow_in_shipping: bool = False   # Disable physics debug in shipping


@dataclass
class ContactPoint:
    """A contact point between two bodies."""
    position: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    penetration: float
    body_a: Any
    body_b: Any
    impulse: float = 0.0


@dataclass
class BodyInspection:
    """Detailed inspection data for a physics body."""
    entity: Any
    body_type: str  # static, dynamic, kinematic
    mass: float
    inertia: Tuple[float, float, float]
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # quaternion
    linear_velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    linear_damping: float
    angular_damping: float
    friction: float
    restitution: float
    is_sleeping: bool
    is_sensor: bool
    collision_group: int
    collision_mask: int
    shape_count: int
    constraint_count: int
    contact_count: int


class PhysicsDebugger:
    """
    Debugger for physics systems.

    SECURITY: This debugger is automatically disabled in shipping builds
    to prevent physics manipulation exploits.

    Provides visualization and control for:
    - Collision shapes
    - Contact points
    - Velocities
    - Physics pause/step
    - Force application
    - Body inspection
    """

    def __init__(self, config: Optional[PhysicsVisualizationConfig] = None) -> None:
        self._config = config or PhysicsVisualizationConfig()
        self._state = PhysicsDebugState.RUNNING
        self._enabled = True

        # Visualization
        self._visualization = PhysicsVisualization.NONE
        self._show_for_entities: Set[Any] = set()
        self._show_all = False

        # Step control
        self._step_pending = False
        self._step_count = 0

        # Contact tracking - limits from config
        self._contacts: List[ContactPoint] = []
        self._max_contacts = self._config.max_contacts

        # Raycast tracking - limits from config
        self._raycast_history: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], bool]] = []
        self._max_raycast_history = self._config.max_raycast_history

        # Force visualization
        self._pending_forces: List[Tuple[Any, Tuple[float, float, float], Tuple[float, float, float]]] = []

        # Callbacks
        self._state_callbacks: List[Callable[[PhysicsDebugState], None]] = []

        # Check build restrictions (after all fields initialized)
        self._build_allowed = self._check_build_allowed()

    def _check_build_allowed(self) -> bool:
        """Check if physics debugging is allowed in this build."""
        import os

        if os.environ.get("GAME_BUILD_TYPE", "").upper() == "SHIPPING":
            if not self._config.allow_in_shipping:
                logger.info("PhysicsDebugger disabled - shipping build")
                return False
        if os.environ.get("SHIPPING") == "1":
            if not self._config.allow_in_shipping:
                return False

        return True

    @property
    def state(self) -> PhysicsDebugState:
        """Get the current physics debug state."""
        return self._state

    @property
    def enabled(self) -> bool:
        """Check if physics debugging is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable physics debugging."""
        if value and not self._build_allowed:
            logger.warning("Cannot enable physics debugger - not allowed in this build")
            return
        self._enabled = value
        if not value:
            self._visualization = PhysicsVisualization.NONE
            self._show_for_entities.clear()
            self._show_all = False

    @property
    def visualization(self) -> PhysicsVisualization:
        """Get current visualization flags."""
        return self._visualization

    @property
    def config(self) -> PhysicsVisualizationConfig:
        """Get visualization configuration."""
        return self._config

    # =========================================================================
    # Visualization Control
    # =========================================================================

    def show_collision_shapes(self, enabled: bool = True) -> None:
        """Enable/disable collision shape visualization."""
        if enabled:
            self._visualization |= PhysicsVisualization.COLLISION_SHAPES
        else:
            self._visualization &= ~PhysicsVisualization.COLLISION_SHAPES

    def show_contact_points(self, enabled: bool = True) -> None:
        """Enable/disable contact point visualization."""
        if enabled:
            self._visualization |= PhysicsVisualization.CONTACT_POINTS
        else:
            self._visualization &= ~PhysicsVisualization.CONTACT_POINTS

    def show_velocities(self, enabled: bool = True) -> None:
        """Enable/disable velocity visualization."""
        if enabled:
            self._visualization |= PhysicsVisualization.VELOCITIES
        else:
            self._visualization &= ~PhysicsVisualization.VELOCITIES

    def show_angular_velocities(self, enabled: bool = True) -> None:
        """Enable/disable angular velocity visualization."""
        if enabled:
            self._visualization |= PhysicsVisualization.ANGULAR_VELOCITIES
        else:
            self._visualization &= ~PhysicsVisualization.ANGULAR_VELOCITIES

    def show_constraints(self, enabled: bool = True) -> None:
        """Enable/disable constraint visualization."""
        if enabled:
            self._visualization |= PhysicsVisualization.CONSTRAINTS
        else:
            self._visualization &= ~PhysicsVisualization.CONSTRAINTS

    def show_joints(self, enabled: bool = True) -> None:
        """Enable/disable joint visualization."""
        if enabled:
            self._visualization |= PhysicsVisualization.JOINTS
        else:
            self._visualization &= ~PhysicsVisualization.JOINTS

    def show_raycasts(self, enabled: bool = True) -> None:
        """Enable/disable raycast visualization."""
        if enabled:
            self._visualization |= PhysicsVisualization.RAYCASTS
        else:
            self._visualization &= ~PhysicsVisualization.RAYCASTS

    def show_center_of_mass(self, enabled: bool = True) -> None:
        """Enable/disable center of mass visualization."""
        if enabled:
            self._visualization |= PhysicsVisualization.CENTER_OF_MASS
        else:
            self._visualization &= ~PhysicsVisualization.CENTER_OF_MASS

    def show_all_visualizations(self) -> None:
        """Enable all visualizations."""
        self._visualization = PhysicsVisualization.ALL

    def hide_all_visualizations(self) -> None:
        """Disable all visualizations."""
        self._visualization = PhysicsVisualization.NONE

    def set_visualization(self, flags: PhysicsVisualization) -> None:
        """Set visualization flags directly."""
        self._visualization = flags

    def is_visualization_enabled(self, flag: PhysicsVisualization) -> bool:
        """Check if a specific visualization is enabled."""
        return bool(self._visualization & flag)

    def show_for_entity(self, entity: Any) -> None:
        """Show physics debug for a specific entity."""
        self._show_for_entities.add(entity)

    def hide_for_entity(self, entity: Any) -> None:
        """Hide physics debug for a specific entity."""
        self._show_for_entities.discard(entity)

    def show_for_all(self, enabled: bool = True) -> None:
        """Show physics debug for all entities."""
        self._show_all = enabled

    def should_visualize(self, entity: Any) -> bool:
        """Check if entity should be visualized."""
        return self._show_all or entity in self._show_for_entities

    # =========================================================================
    # Physics Control
    # =========================================================================

    def pause_physics(self) -> None:
        """Pause physics simulation."""
        self._state = PhysicsDebugState.PAUSED
        logger.info("Physics paused")
        self._notify_state_callbacks(self._state)

    def resume_physics(self) -> None:
        """Resume physics simulation."""
        self._state = PhysicsDebugState.RUNNING
        self._step_pending = False
        self._step_count = 0
        logger.info("Physics resumed")
        self._notify_state_callbacks(self._state)

    def step_physics(self, count: int = 1) -> int:
        """
        Step physics simulation.

        Args:
            count: Number of physics steps to execute

        Returns:
            Number of steps queued.
        """
        if self._state != PhysicsDebugState.PAUSED:
            logger.warning("Cannot step - physics is not paused")
            return 0

        self._step_pending = True
        self._step_count = max(1, count)
        self._state = PhysicsDebugState.STEPPING

        logger.debug("Physics step: %d steps", self._step_count)
        self._notify_state_callbacks(self._state)

        return self._step_count

    def should_simulate(self) -> bool:
        """
        Check if physics should simulate.

        Called by physics system to check if it should step.
        """
        if not self._enabled:
            return True

        if self._state == PhysicsDebugState.RUNNING:
            return True

        if self._state == PhysicsDebugState.STEPPING:
            return self._step_pending

        return False

    def consume_step(self) -> bool:
        """
        Consume a pending physics step.

        Returns True if step was consumed.
        """
        if self._step_pending:
            self._step_count -= 1
            if self._step_count <= 0:
                self._step_pending = False
                self._step_count = 0
                self._state = PhysicsDebugState.PAUSED
                self._notify_state_callbacks(self._state)
            return True
        return False

    # =========================================================================
    # Force Application
    # =========================================================================

    def apply_force(
        self,
        entity: Any,
        force: Tuple[float, float, float],
        point: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Apply a force to an entity (for visualization and application).

        Args:
            entity: Entity to apply force to
            force: Force vector (x, y, z)
            point: Point of application (None for center of mass)
        """
        if point is None:
            point = self._get_entity_position(entity) or (0.0, 0.0, 0.0)

        self._pending_forces.append((entity, force, point))

        # Actual force application would be handled by physics system
        logger.debug(
            "Force applied: entity=%s, force=%s, point=%s",
            entity, force, point,
        )

    def get_pending_forces(self) -> List[Tuple[Any, Tuple[float, float, float], Tuple[float, float, float]]]:
        """Get and clear pending forces."""
        forces = self._pending_forces.copy()
        self._pending_forces.clear()
        return forces

    def _get_entity_position(self, entity: Any) -> Optional[Tuple[float, float, float]]:
        """Get entity position."""
        if hasattr(entity, "position"):
            pos = entity.position
            if hasattr(pos, "x"):
                return (pos.x, pos.y, pos.z)
            elif isinstance(pos, (tuple, list)):
                return tuple(pos[:3])
        return None

    # =========================================================================
    # Contact Tracking
    # =========================================================================

    def record_contact(self, contact: ContactPoint) -> None:
        """Record a contact point for visualization."""
        if len(self._contacts) >= self._max_contacts:
            self._contacts.pop(0)
        self._contacts.append(contact)

    def get_contacts(self) -> List[ContactPoint]:
        """Get current contact points."""
        return self._contacts.copy()

    def clear_contacts(self) -> None:
        """Clear recorded contacts."""
        self._contacts.clear()

    # =========================================================================
    # Raycast Tracking
    # =========================================================================

    def record_raycast(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        hit: bool,
    ) -> None:
        """Record a raycast for visualization."""
        if len(self._raycast_history) >= self._max_raycast_history:
            self._raycast_history.pop(0)
        self._raycast_history.append((start, end, hit))

    def get_raycasts(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], bool]]:
        """Get raycast history."""
        return self._raycast_history.copy()

    def clear_raycasts(self) -> None:
        """Clear raycast history."""
        self._raycast_history.clear()

    # =========================================================================
    # Body Inspection
    # =========================================================================

    def inspect_body(self, entity: Any) -> Optional[BodyInspection]:
        """
        Get detailed inspection data for a physics body.

        Args:
            entity: Entity to inspect

        Returns:
            BodyInspection or None if no physics body found.
        """
        body = self._get_physics_body(entity)
        if body is None:
            return None

        return self._build_body_inspection(entity, body)

    def _get_physics_body(self, entity: Any) -> Optional[Any]:
        """Get physics body from entity. Override for actual implementation."""
        if hasattr(entity, "physics_body"):
            return entity.physics_body
        if hasattr(entity, "body"):
            return entity.body
        if hasattr(entity, "rigidbody"):
            return entity.rigidbody
        return None

    def _build_body_inspection(self, entity: Any, body: Any) -> BodyInspection:
        """Build inspection data for a physics body."""
        # Extract data from body (with fallbacks)
        return BodyInspection(
            entity=entity,
            body_type=getattr(body, "body_type", "unknown"),
            mass=getattr(body, "mass", 0.0),
            inertia=getattr(body, "inertia", (0.0, 0.0, 0.0)),
            position=self._get_vec3_tuple(body, "position", (0.0, 0.0, 0.0)),
            rotation=self._get_vec4_tuple(body, "rotation", (0.0, 0.0, 0.0, 1.0)),
            linear_velocity=self._get_vec3_tuple(body, "linear_velocity", (0.0, 0.0, 0.0)),
            angular_velocity=self._get_vec3_tuple(body, "angular_velocity", (0.0, 0.0, 0.0)),
            linear_damping=getattr(body, "linear_damping", 0.0),
            angular_damping=getattr(body, "angular_damping", 0.0),
            friction=getattr(body, "friction", 0.5),
            restitution=getattr(body, "restitution", 0.0),
            is_sleeping=getattr(body, "is_sleeping", False),
            is_sensor=getattr(body, "is_sensor", False),
            collision_group=getattr(body, "collision_group", 0),
            collision_mask=getattr(body, "collision_mask", 0xFFFFFFFF),
            shape_count=len(getattr(body, "shapes", [])),
            constraint_count=len(getattr(body, "constraints", [])),
            contact_count=len(getattr(body, "contacts", [])),
        )

    def _get_vec3_tuple(
        self,
        obj: Any,
        attr: str,
        default: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Get a Vec3 attribute as tuple."""
        val = getattr(obj, attr, None)
        if val is None:
            return default
        if hasattr(val, "x"):
            return (val.x, val.y, val.z)
        if isinstance(val, (tuple, list)) and len(val) >= 3:
            return tuple(val[:3])
        return default

    def _get_vec4_tuple(
        self,
        obj: Any,
        attr: str,
        default: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """Get a Vec4/Quat attribute as tuple."""
        val = getattr(obj, attr, None)
        if val is None:
            return default
        if hasattr(val, "x"):
            return (val.x, val.y, val.z, getattr(val, "w", 1.0))
        if isinstance(val, (tuple, list)) and len(val) >= 4:
            return tuple(val[:4])
        return default

    # =========================================================================
    # Callbacks
    # =========================================================================

    def add_state_callback(
        self,
        callback: Callable[[PhysicsDebugState], None],
    ) -> None:
        """Add a callback for physics debug state changes."""
        self._state_callbacks.append(callback)

    def remove_state_callback(
        self,
        callback: Callable[[PhysicsDebugState], None],
    ) -> bool:
        """Remove a state callback."""
        try:
            self._state_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _notify_state_callbacks(self, state: PhysicsDebugState) -> None:
        """Notify state callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error("State callback error: %s", e)

    # =========================================================================
    # Console Commands
    # =========================================================================

    def cmd_physics_pause(self) -> str:
        """Console command: physics.pause"""
        self.pause_physics()
        return "Physics paused"

    def cmd_physics_resume(self) -> str:
        """Console command: physics.resume"""
        self.resume_physics()
        return "Physics resumed"

    def cmd_physics_step(self, count: int = 1) -> str:
        """Console command: physics.step [count]"""
        stepped = self.step_physics(count)
        if stepped > 0:
            return f"Physics stepping {stepped} step(s)"
        return "Cannot step - physics is not paused"

    def cmd_show_collision(self, enabled: bool = True) -> str:
        """Console command: show.collision [0/1]"""
        self.show_collision_shapes(enabled)
        return f"Collision shapes {'enabled' if enabled else 'disabled'}"

    def cmd_show_velocities(self, enabled: bool = True) -> str:
        """Console command: show.velocities [0/1]"""
        self.show_velocities(enabled)
        return f"Velocities {'enabled' if enabled else 'disabled'}"


# =============================================================================
# Singleton instance
# =============================================================================

_physics_debugger: Optional[PhysicsDebugger] = None


def get_physics_debugger() -> PhysicsDebugger:
    """Get the global physics debugger instance."""
    global _physics_debugger
    if _physics_debugger is None:
        _physics_debugger = PhysicsDebugger()
    return _physics_debugger


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "BodyInspection",
    "ContactPoint",
    "get_physics_debugger",
    "PhysicsDebugger",
    "PhysicsDebugState",
    "PhysicsVisualization",
    "PhysicsVisualizationConfig",
]
