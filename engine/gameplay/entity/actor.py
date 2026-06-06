"""
Actor System
=============
Base actor classes following UE5-inspired architecture:
- Actor: Base class for all world entities
- StaticActor: Non-moving actors
- DynamicActor: Physics-enabled actors
- Pawn: Possessable actors
- Character: Humanoid pawns with movement

Uses the Trinity Pattern with:
- ActorMeta metaclass for registration and validation
- Component composition over inheritance
- Lifecycle integration
"""
from __future__ import annotations

import threading
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from trinity.decorators.ops import Op, Step, make_decorator, run_steps
from trinity.descriptors.base import BaseDescriptor
from trinity.metaclasses.engine_meta import EngineMeta

def _get_entity_event_log():
    """Get the EntityEventLog singleton (lazy import to avoid circular imports)."""
    from .eventlog_integration import EntityEventLog
    return EntityEventLog()

from .constants import (
    CROUCH_SPEED_MULTIPLIER,
    DEFAULT_JUMP_VELOCITY,
    DEFAULT_MAX_RUN_SPEED,
    DEFAULT_MAX_WALK_SPEED,
    ENTITY_ID_INVALID,
    ENTITY_ID_START,
    ENTITY_NAME_MAX_LENGTH,
    MAX_COMPONENTS_PER_ENTITY,
    ActorType,
    LifecycleState,
    TickGroup,
)
from .lifecycle import LifecycleCallback, LifecycleEvent, LifecycleManager, LifecycleMixin

if TYPE_CHECKING:
    from .possession import Controller

T = TypeVar("T")
ComponentT = TypeVar("ComponentT")


# =============================================================================
# ACTOR METACLASS
# =============================================================================


class ActorMeta(EngineMeta):
    """
    Metaclass for Actor types.

    Responsibilities:
    - Assign unique actor type IDs
    - Register actor types in global registry
    - Validate actor class definitions
    - Collect component declarations
    """

    _registry: ClassVar[Dict[int, Type["Actor"]]] = {}
    _name_to_id: ClassVar[Dict[str, int]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # Base class names to skip registration
    _BASE_CLASS_NAMES: ClassVar[frozenset[str]] = frozenset({
        "Actor",
        "StaticActor",
        "DynamicActor",
        "Pawn",
        "Character",
    })

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "ActorMeta":
        """Create a new actor type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base classes
        if name in mcs._BASE_CLASS_NAMES:
            cls._actor_type_id = 0
            cls._actor_type_name = name
            return cls

        with mcs._lock:
            # Assign unique type ID
            cls._actor_type_id = mcs._next_id
            mcs._next_id += 1
            cls._actor_type_name = f"{cls.__module__}.{name}"

            # Record metaclass steps
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "actor_type_id", "value": cls._actor_type_id})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "actor_type_name", "value": cls._actor_type_name})
            )

            # Collect component declarations
            mcs._collect_components(cls)

            # Validate actor definition
            mcs._validate_actor(cls)
            cls._metaclass_steps.append(
                Step(Op.VALIDATE, {"constraint": "actor_rules"})
            )

            # Register
            mcs._registry[cls._actor_type_id] = cls
            mcs._name_to_id[cls._actor_type_name] = cls._actor_type_id
            cls._metaclass_steps.append(
                Step(Op.REGISTER, {"registry": "actor_registry", "id": cls._actor_type_id})
            )

        return cls

    @classmethod
    def _collect_components(mcs, cls: type) -> None:
        """Collect component class declarations."""
        cls._declared_components: Dict[str, type] = {}

        # Look for component declarations in class annotations
        annotations = getattr(cls, "__annotations__", {})
        for field_name, field_type in annotations.items():
            if field_name.startswith("_"):
                continue
            # Check if this is a component type (has _component marker)
            if hasattr(field_type, "_component"):
                cls._declared_components[field_name] = field_type
                cls._metaclass_steps.append(
                    Step(Op.TAG, {"key": f"component_{field_name}", "value": field_type.__name__})
                )

    @classmethod
    def _validate_actor(mcs, cls: type) -> None:
        """Validate actor class definition."""
        # Validate component count
        if len(getattr(cls, "_declared_components", {})) > MAX_COMPONENTS_PER_ENTITY:
            raise ValueError(
                f"{cls.__name__}: Too many components declared ({len(cls._declared_components)}). "
                f"Maximum is {MAX_COMPONENTS_PER_ENTITY}."
            )

    @classmethod
    def get_by_id(mcs, actor_type_id: int) -> Optional[Type["Actor"]]:
        """Get actor class by type ID."""
        return mcs._registry.get(actor_type_id)

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[Type["Actor"]]:
        """Get actor class by qualified name."""
        actor_type_id = mcs._name_to_id.get(name)
        return mcs._registry.get(actor_type_id) if actor_type_id else None

    @classmethod
    def all_actor_types(mcs) -> List[Type["Actor"]]:
        """Get all registered actor types."""
        return list(mcs._registry.values())

    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the actor registry (for testing)."""
        with mcs._lock:
            mcs._registry.clear()
            mcs._name_to_id.clear()
            mcs._next_id = 1
        super().clear_registry()


# =============================================================================
# COMPONENT CONTAINER
# =============================================================================


class ComponentContainer:
    """
    Container for entity components.

    Provides:
    - Fast component lookup by type
    - Component iteration
    - Component add/remove with callbacks
    """

    __slots__ = ("_components", "_owner", "_type_to_name")

    def __init__(self, owner: "Actor") -> None:
        self._owner = weakref.ref(owner)
        self._components: Dict[str, Any] = {}
        self._type_to_name: Dict[type, str] = {}

    def add(self, name: str, component: Any) -> None:
        """Add a component to the container."""
        if name in self._components:
            raise ValueError(f"Component '{name}' already exists")
        if len(self._components) >= MAX_COMPONENTS_PER_ENTITY:
            raise ValueError(f"Maximum component count ({MAX_COMPONENTS_PER_ENTITY}) reached")

        self._components[name] = component
        self._type_to_name[type(component)] = name

        # Notify owner
        owner = self._owner()
        if owner and hasattr(owner, "_on_component_added"):
            owner._on_component_added(name, component)

    def remove(self, name: str) -> Optional[Any]:
        """Remove a component from the container."""
        component = self._components.pop(name, None)
        if component is not None:
            del self._type_to_name[type(component)]

            # Notify owner
            owner = self._owner()
            if owner and hasattr(owner, "_on_component_removed"):
                owner._on_component_removed(name, component)

        return component

    def get(self, name: str) -> Optional[Any]:
        """Get a component by name."""
        return self._components.get(name)

    def get_by_type(self, component_type: Type[T]) -> Optional[T]:
        """Get a component by type."""
        name = self._type_to_name.get(component_type)
        return self._components.get(name) if name else None

    def has(self, name: str) -> bool:
        """Check if a component exists by name."""
        return name in self._components

    def has_type(self, component_type: type) -> bool:
        """Check if a component exists by type."""
        return component_type in self._type_to_name

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Iterate over (name, component) pairs."""
        return iter(self._components.items())

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self._components)

    def clear(self) -> None:
        """Remove all components."""
        for name in list(self._components.keys()):
            self.remove(name)


# =============================================================================
# TRANSFORM DATA
# =============================================================================


@dataclass
class Transform:
    """3D transform data."""

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # Quaternion
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def copy(self) -> "Transform":
        """Create a copy of this transform."""
        return Transform(
            position=self.position,
            rotation=self.rotation,
            scale=self.scale,
        )


# =============================================================================
# BASE ACTOR CLASS
# =============================================================================


class Actor(LifecycleMixin, metaclass=ActorMeta):
    """
    Base class for all world entities.

    Features:
    - Unique entity ID
    - Component composition
    - Lifecycle management
    - Transform hierarchy
    - Tag system
    """

    # Class-level configuration
    _actor_type: ClassVar[ActorType] = ActorType.STATIC
    _default_tick_group: ClassVar[TickGroup] = TickGroup.UPDATE
    _can_tick: ClassVar[bool] = True

    # Instance ID generation
    _next_entity_id: ClassVar[int] = ENTITY_ID_START
    _id_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        name: Optional[str] = None,
        transform: Optional[Transform] = None,
        tags: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the actor.

        Args:
            name: Optional display name
            transform: Initial transform (position, rotation, scale)
            tags: Optional set of tags for filtering
            **kwargs: Additional initialization parameters
        """
        # Generate unique entity ID
        with self._id_lock:
            self._entity_id = Actor._next_entity_id
            Actor._next_entity_id += 1

        # Basic properties
        self._name = name or f"{self.__class__.__name__}_{self._entity_id}"
        if len(self._name) > ENTITY_NAME_MAX_LENGTH:
            self._name = self._name[:ENTITY_NAME_MAX_LENGTH]

        # Transform
        self._transform = transform.copy() if transform else Transform()
        self._parent: Optional[weakref.ref["Actor"]] = None
        self._children: List[weakref.ref["Actor"]] = []

        # Components
        self._components = ComponentContainer(self)

        # Tags
        self._tags: Set[str] = tags.copy() if tags else set()

        # Tick configuration
        self._tick_enabled = self._can_tick
        self._tick_group = self._default_tick_group

        # Initialize lifecycle
        self._init_lifecycle()
        self._lifecycle_state = LifecycleState.CREATED

        # Log spawn event to EventLog
        try:
            event_log = _get_entity_event_log()
            position = self._transform.position if self._transform else (0.0, 0.0, 0.0)
            spawn_event = event_log.record_spawn(
                self._entity_id,
                prefab_name=self._name,
                position=position,
                entity_type=type(self).__name__,
            )
            # Store spawn event ID for causal chain tracking
            self._spawn_event_id = spawn_event.id
        except Exception:
            self._spawn_event_id = None

        # Log state change from UNINITIALIZED to CREATED
        try:
            event_log = _get_entity_event_log()
            event_log.record_state_change(
                self._entity_id,
                LifecycleState.UNINITIALIZED,
                LifecycleState.CREATED,
            )
        except Exception:
            pass

        # Initialize declared components
        self._init_declared_components()

    def _init_declared_components(self) -> None:
        """Initialize components declared in class definition."""
        for name, component_type in getattr(self.__class__, "_declared_components", {}).items():
            if not self._components.has(name):
                # Check if there's a default value
                default = getattr(self, name, None)
                if default is not None and isinstance(default, component_type):
                    self._components.add(name, default)
                else:
                    # Create default instance
                    try:
                        instance = component_type()
                        self._components.add(name, instance)
                    except TypeError:
                        pass  # Component requires arguments

    # =========================================================================
    # IDENTIFICATION
    # =========================================================================

    @property
    def entity_id(self) -> int:
        """Get the unique entity ID."""
        return self._entity_id

    @property
    def name(self) -> str:
        """Get the display name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the display name."""
        if len(value) > ENTITY_NAME_MAX_LENGTH:
            value = value[:ENTITY_NAME_MAX_LENGTH]
        self._name = value

    @property
    def actor_type(self) -> ActorType:
        """Get the actor type."""
        return self._actor_type

    # =========================================================================
    # TRANSFORM
    # =========================================================================

    @property
    def transform(self) -> Transform:
        """Get the local transform."""
        return self._transform

    @transform.setter
    def transform(self, value: Transform) -> None:
        """Set the local transform."""
        self._transform = value

    @property
    def position(self) -> Tuple[float, float, float]:
        """Get the local position."""
        return self._transform.position

    @position.setter
    def position(self, value: Tuple[float, float, float]) -> None:
        """Set the local position."""
        self._transform.position = value

    @property
    def rotation(self) -> Tuple[float, float, float, float]:
        """Get the local rotation (quaternion)."""
        return self._transform.rotation

    @rotation.setter
    def rotation(self, value: Tuple[float, float, float, float]) -> None:
        """Set the local rotation (quaternion)."""
        self._transform.rotation = value

    @property
    def scale(self) -> Tuple[float, float, float]:
        """Get the local scale."""
        return self._transform.scale

    @scale.setter
    def scale(self, value: Tuple[float, float, float]) -> None:
        """Set the local scale."""
        self._transform.scale = value

    def get_world_position(self) -> Tuple[float, float, float]:
        """Get the world position (accounting for parent hierarchy)."""
        if self._parent is None:
            return self._transform.position
        parent = self._parent()
        if parent is None:
            return self._transform.position
        # Simple additive for now (proper matrix transform would be more accurate)
        parent_pos = parent.get_world_position()
        return (
            parent_pos[0] + self._transform.position[0],
            parent_pos[1] + self._transform.position[1],
            parent_pos[2] + self._transform.position[2],
        )

    # =========================================================================
    # HIERARCHY
    # =========================================================================

    @property
    def parent(self) -> Optional["Actor"]:
        """Get the parent actor."""
        return self._parent() if self._parent else None

    def set_parent(self, parent: Optional["Actor"]) -> None:
        """Set the parent actor."""
        # Remove from old parent
        if self._parent is not None:
            old_parent = self._parent()
            if old_parent is not None:
                old_parent._children = [
                    ref for ref in old_parent._children if ref() is not self
                ]

        # Set new parent
        if parent is not None:
            self._parent = weakref.ref(parent)
            parent._children.append(weakref.ref(self))
        else:
            self._parent = None

    @property
    def children(self) -> List["Actor"]:
        """Get all child actors."""
        return [ref() for ref in self._children if ref() is not None]

    def add_child(self, child: "Actor") -> None:
        """Add a child actor."""
        child.set_parent(self)

    def remove_child(self, child: "Actor") -> None:
        """Remove a child actor."""
        if child.parent is self:
            child.set_parent(None)

    # =========================================================================
    # COMPONENTS
    # =========================================================================

    @property
    def components(self) -> ComponentContainer:
        """Get the component container."""
        return self._components

    def add_component(self, name: str, component: Any) -> None:
        """Add a component to this actor."""
        self._components.add(name, component)

    def remove_component(self, name: str) -> Optional[Any]:
        """Remove a component from this actor."""
        return self._components.remove(name)

    def get_component(self, name: str) -> Optional[Any]:
        """Get a component by name."""
        return self._components.get(name)

    def get_component_by_type(self, component_type: Type[T]) -> Optional[T]:
        """Get a component by type."""
        return self._components.get_by_type(component_type)

    def has_component(self, name: str) -> bool:
        """Check if a component exists by name."""
        return self._components.has(name)

    def has_component_type(self, component_type: type) -> bool:
        """Check if a component exists by type."""
        return self._components.has_type(component_type)

    def _on_component_added(self, name: str, component: Any) -> None:
        """Called when a component is added."""
        # Log to EventLog
        try:
            event_log = _get_entity_event_log()
            from .eventlog_integration import CausalChain

            # Link to spawn event if available
            causal_chain = None
            spawn_event_id = getattr(self, "_spawn_event_id", None)
            if spawn_event_id is not None:
                causal_chain = CausalChain(
                    root_event_id=spawn_event_id,
                    parent_event_id=spawn_event_id,
                    depth=1,
                )

            event_log.record_component_added(
                self._entity_id,
                component_type=type(component).__name__,
                component_name=name,
                causal_chain=causal_chain,
            )
        except Exception:
            pass  # Don't let logging errors break component addition

    def _on_component_removed(self, name: str, component: Any) -> None:
        """Called when a component is removed."""
        # Log to EventLog
        try:
            event_log = _get_entity_event_log()
            event_log.record_component_removed(
                self._entity_id,
                component_type=type(component).__name__,
                component_name=name,
            )
        except Exception:
            pass  # Don't let logging errors break component removal

    # =========================================================================
    # TAGS
    # =========================================================================

    def add_tag(self, tag: str) -> None:
        """Add a tag to this actor."""
        self._tags.add(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this actor."""
        self._tags.discard(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if this actor has a tag."""
        return tag in self._tags

    def has_any_tag(self, tags: Set[str]) -> bool:
        """Check if this actor has any of the given tags."""
        return bool(self._tags & tags)

    def has_all_tags(self, tags: Set[str]) -> bool:
        """Check if this actor has all of the given tags."""
        return tags <= self._tags

    @property
    def tags(self) -> frozenset[str]:
        """Get all tags (immutable view)."""
        return frozenset(self._tags)

    # =========================================================================
    # TICK
    # =========================================================================

    @property
    def tick_enabled(self) -> bool:
        """Check if tick is enabled."""
        return self._tick_enabled

    @tick_enabled.setter
    def tick_enabled(self, value: bool) -> None:
        """Enable or disable tick."""
        self._tick_enabled = value

    @property
    def tick_group(self) -> TickGroup:
        """Get the tick group."""
        return self._tick_group

    @tick_group.setter
    def tick_group(self, value: TickGroup) -> None:
        """Set the tick group."""
        self._tick_group = value

    def tick(self, delta_time: float) -> None:
        """
        Called every frame when tick is enabled.

        Override this method to implement per-frame behavior.

        Args:
            delta_time: Time since last tick in seconds
        """
        pass

    # =========================================================================
    # LIFECYCLE CALLBACKS
    # =========================================================================

    def on_spawn(self) -> None:
        """Called when the actor is spawned."""
        pass

    def begin_play(self) -> None:
        """Called when the actor begins playing."""
        pass

    def end_play(self) -> None:
        """Called when the actor ends playing."""
        pass

    def on_destroy(self) -> None:
        """Called when the actor is destroyed."""
        # Clean up children
        for child in self.children:
            child.set_parent(None)

        # Clean up components
        self._components.clear()

    # =========================================================================
    # DESTRUCTION
    # =========================================================================

    def destroy(self, immediate: bool = False) -> None:
        """
        Request destruction of this actor.

        Args:
            immediate: If True, destroy immediately; otherwise defer
        """
        self.transition_to(LifecycleState.DESTROYING, immediate=immediate)

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"id={self._entity_id} "
            f"name='{self._name}' "
            f"state={self._lifecycle_state.name}>"
        )

    @classmethod
    def reset_entity_ids(cls) -> None:
        """Reset entity ID generation (for testing)."""
        with cls._id_lock:
            cls._next_entity_id = ENTITY_ID_START


# =============================================================================
# STATIC ACTOR
# =============================================================================


class StaticActor(Actor):
    """
    Actor that does not move.

    Features:
    - No physics simulation
    - Optimized for static geometry
    - Can still have components and children
    """

    _actor_type: ClassVar[ActorType] = ActorType.STATIC
    _can_tick: ClassVar[bool] = False  # Static actors don't tick by default

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._is_static = True

    @property
    def is_static(self) -> bool:
        """Check if this is a static actor."""
        return True


# =============================================================================
# DYNAMIC ACTOR
# =============================================================================


class DynamicActor(Actor):
    """
    Actor with physics simulation.

    Features:
    - Physics body integration
    - Velocity and forces
    - Collision detection
    """

    _actor_type: ClassVar[ActorType] = ActorType.DYNAMIC
    _can_tick: ClassVar[bool] = True
    _default_tick_group: ClassVar[TickGroup] = TickGroup.POST_PHYSICS

    def __init__(
        self,
        simulate_physics: bool = True,
        mass: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._simulate_physics = simulate_physics
        self._mass = mass
        self._velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def simulate_physics(self) -> bool:
        """Check if physics simulation is enabled."""
        return self._simulate_physics

    @simulate_physics.setter
    def simulate_physics(self, value: bool) -> None:
        """Enable or disable physics simulation."""
        self._simulate_physics = value

    @property
    def mass(self) -> float:
        """Get the mass."""
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        """Set the mass."""
        if value <= 0:
            raise ValueError("Mass must be positive")
        self._mass = value

    @property
    def velocity(self) -> Tuple[float, float, float]:
        """Get the linear velocity."""
        return self._velocity

    @velocity.setter
    def velocity(self, value: Tuple[float, float, float]) -> None:
        """Set the linear velocity."""
        self._velocity = value

    @property
    def angular_velocity(self) -> Tuple[float, float, float]:
        """Get the angular velocity."""
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value: Tuple[float, float, float]) -> None:
        """Set the angular velocity."""
        self._angular_velocity = value

    def add_force(self, force: Tuple[float, float, float]) -> None:
        """Add a force to the actor."""
        # F = ma, so a = F/m
        acceleration = (
            force[0] / self._mass,
            force[1] / self._mass,
            force[2] / self._mass,
        )
        self._velocity = (
            self._velocity[0] + acceleration[0],
            self._velocity[1] + acceleration[1],
            self._velocity[2] + acceleration[2],
        )

    def add_impulse(self, impulse: Tuple[float, float, float]) -> None:
        """Add an impulse (instant velocity change) to the actor."""
        # J = mv, so dv = J/m
        delta_v = (
            impulse[0] / self._mass,
            impulse[1] / self._mass,
            impulse[2] / self._mass,
        )
        self._velocity = (
            self._velocity[0] + delta_v[0],
            self._velocity[1] + delta_v[1],
            self._velocity[2] + delta_v[2],
        )


# =============================================================================
# PAWN
# =============================================================================


class Pawn(DynamicActor):
    """
    Actor that can be possessed by a Controller.

    Features:
    - Controller possession/unpossession
    - Input handling through controller
    - AI or player controlled
    """

    _actor_type: ClassVar[ActorType] = ActorType.PAWN
    _can_tick: ClassVar[bool] = True

    def __init__(
        self,
        auto_possess: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._controller: Optional[weakref.ref["Controller"]] = None
        self._auto_possess = auto_possess
        self._pending_controller: Optional[weakref.ref["Controller"]] = None

    @property
    def controller(self) -> Optional["Controller"]:
        """Get the possessing controller."""
        return self._controller() if self._controller else None

    @property
    def is_possessed(self) -> bool:
        """Check if this pawn is possessed."""
        return self._controller is not None and self._controller() is not None

    @property
    def is_player_controlled(self) -> bool:
        """Check if this pawn is controlled by a player."""
        controller = self.controller
        if controller is None:
            return False
        # Check controller type
        from .possession import PlayerController
        return isinstance(controller, PlayerController)

    def possess(self, controller: "Controller") -> bool:
        """
        Called by controller to possess this pawn.

        Args:
            controller: The controller taking possession

        Returns:
            True if possession was successful
        """
        if self._controller is not None:
            # Already possessed - unpossess first
            current = self._controller()
            if current is not None:
                current.unpossess()

        self._controller = weakref.ref(controller)
        self._on_possessed(controller)
        return True

    def unpossess(self) -> Optional["Controller"]:
        """
        Called to release possession.

        Returns:
            The previous controller, if any
        """
        old_controller = self._controller() if self._controller else None
        self._controller = None
        if old_controller is not None:
            self._on_unpossessed(old_controller)
        return old_controller

    def _on_possessed(self, controller: "Controller") -> None:
        """Called when possessed by a controller."""
        pass

    def _on_unpossessed(self, controller: "Controller") -> None:
        """Called when unpossessed."""
        pass

    def setup_player_input(self) -> None:
        """
        Override to bind player input actions.

        Called when a player controller possesses this pawn.
        """
        pass

    def on_restart(self) -> None:
        """Called when the pawn needs to be reset/restarted."""
        pass


# =============================================================================
# CHARACTER
# =============================================================================


class Character(Pawn):
    """
    Humanoid pawn with movement capabilities.

    Features:
    - Walking/running movement
    - Jumping
    - Crouching
    - Movement state tracking
    """

    _actor_type: ClassVar[ActorType] = ActorType.CHARACTER
    _can_tick: ClassVar[bool] = True

    def __init__(
        self,
        max_walk_speed: float = DEFAULT_MAX_WALK_SPEED,
        max_run_speed: float = DEFAULT_MAX_RUN_SPEED,
        jump_velocity: float = DEFAULT_JUMP_VELOCITY,
        can_crouch: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Movement configuration
        self._max_walk_speed = max_walk_speed
        self._max_run_speed = max_run_speed
        self._jump_velocity = jump_velocity
        self._can_crouch = can_crouch

        # Movement state
        self._is_walking = False
        self._is_running = False
        self._is_jumping = False
        self._is_falling = False
        self._is_crouching = False
        self._is_grounded = True

        # Movement input
        self._movement_input: Tuple[float, float] = (0.0, 0.0)  # Forward, Right
        self._look_input: Tuple[float, float] = (0.0, 0.0)  # Yaw, Pitch

    # =========================================================================
    # MOVEMENT PROPERTIES
    # =========================================================================

    @property
    def max_walk_speed(self) -> float:
        """Get maximum walk speed."""
        return self._max_walk_speed

    @max_walk_speed.setter
    def max_walk_speed(self, value: float) -> None:
        """Set maximum walk speed."""
        self._max_walk_speed = max(0.0, value)

    @property
    def max_run_speed(self) -> float:
        """Get maximum run speed."""
        return self._max_run_speed

    @max_run_speed.setter
    def max_run_speed(self, value: float) -> None:
        """Set maximum run speed."""
        self._max_run_speed = max(0.0, value)

    @property
    def jump_velocity(self) -> float:
        """Get jump velocity."""
        return self._jump_velocity

    @jump_velocity.setter
    def jump_velocity(self, value: float) -> None:
        """Set jump velocity."""
        self._jump_velocity = max(0.0, value)

    # =========================================================================
    # MOVEMENT STATE
    # =========================================================================

    @property
    def is_walking(self) -> bool:
        """Check if character is walking."""
        return self._is_walking

    @property
    def is_running(self) -> bool:
        """Check if character is running."""
        return self._is_running

    @property
    def is_jumping(self) -> bool:
        """Check if character is jumping."""
        return self._is_jumping

    @property
    def is_falling(self) -> bool:
        """Check if character is falling."""
        return self._is_falling

    @property
    def is_crouching(self) -> bool:
        """Check if character is crouching."""
        return self._is_crouching

    @property
    def is_grounded(self) -> bool:
        """Check if character is on the ground."""
        return self._is_grounded

    @property
    def current_max_speed(self) -> float:
        """Get current maximum speed based on state."""
        if self._is_crouching:
            return self._max_walk_speed * CROUCH_SPEED_MULTIPLIER
        if self._is_running:
            return self._max_run_speed
        return self._max_walk_speed

    # =========================================================================
    # MOVEMENT INPUT
    # =========================================================================

    def add_movement_input(self, forward: float, right: float) -> None:
        """
        Add movement input.

        Args:
            forward: Forward/backward movement (-1 to 1)
            right: Left/right movement (-1 to 1)
        """
        self._movement_input = (
            max(-1.0, min(1.0, forward)),
            max(-1.0, min(1.0, right)),
        )

    def add_look_input(self, yaw: float, pitch: float) -> None:
        """
        Add look/rotation input.

        Args:
            yaw: Horizontal rotation
            pitch: Vertical rotation
        """
        self._look_input = (yaw, pitch)

    # =========================================================================
    # MOVEMENT ACTIONS
    # =========================================================================

    def jump(self) -> bool:
        """
        Attempt to jump.

        Returns:
            True if jump was initiated
        """
        if not self._is_grounded or self._is_jumping:
            return False

        self._is_jumping = True
        self._is_grounded = False
        self._velocity = (
            self._velocity[0],
            self._velocity[1] + self._jump_velocity,
            self._velocity[2],
        )
        return True

    def crouch(self) -> bool:
        """
        Attempt to crouch.

        Returns:
            True if crouch was successful
        """
        if not self._can_crouch or self._is_crouching:
            return False

        self._is_crouching = True
        return True

    def uncrouch(self, force: bool = False) -> bool:
        """
        Attempt to uncrouch.

        Args:
            force: If True, skip collision check (for when clearance is known)

        Returns:
            True if uncrouch was successful
        """
        if not self._is_crouching:
            return False

        # Collision check would be performed by physics system if available
        # For now, uncrouch is always allowed - physics integration will
        # handle clearance checks when a physics component is attached
        self._is_crouching = False
        return True

    def start_running(self) -> None:
        """Start running."""
        self._is_running = True

    def stop_running(self) -> None:
        """Stop running."""
        self._is_running = False

    # =========================================================================
    # TICK
    # =========================================================================

    def tick(self, delta_time: float) -> None:
        """Process character movement each frame."""
        super().tick(delta_time)

        # Update movement state
        if self._movement_input != (0.0, 0.0):
            self._is_walking = True
        else:
            self._is_walking = False

        # Apply movement based on input
        if self._is_grounded and self._movement_input != (0.0, 0.0):
            speed = self.current_max_speed
            # Simple movement application (would normally consider rotation)
            self._velocity = (
                self._movement_input[0] * speed,
                self._velocity[1],
                self._movement_input[1] * speed,
            )

        # Clear input for next frame
        self._movement_input = (0.0, 0.0)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Metaclass
    "ActorMeta",
    # Container
    "ComponentContainer",
    # Transform
    "Transform",
    # Actor classes
    "Actor",
    "StaticActor",
    "DynamicActor",
    "Pawn",
    "Character",
]
