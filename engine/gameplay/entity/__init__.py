"""Entity subsystem: Actor/Pawn framework, prefabs, lifecycle, and possession."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from engine.gameplay.constants import (
    ActorType,
    EntityState,
    ACTOR_TYPE_STATIC,
    ACTOR_TYPE_DYNAMIC,
    ACTOR_TYPE_PAWN,
    ACTOR_TYPE_CHARACTER,
)
from engine.gameplay.entity.constants import (
    CROUCH_SPEED_MULTIPLIER,
    DEFAULT_CHARACTER_HEALTH,
    DEFAULT_CHARACTER_MAX_HEALTH,
    DEFAULT_JUMP_FORCE,
    DEFAULT_MAX_WALK_SPEED,
    SPRINT_SPEED_MULTIPLIER,
)

if TYPE_CHECKING:
    from engine.core.ecs import Entity, World


# === Base Actor Classes ===

@dataclass
class ActorConfig:
    """Configuration for actor spawning."""

    actor_type: ActorType = ActorType.STATIC
    spawn_transform: Optional[Tuple[float, float, float]] = None
    spawn_rotation: Optional[Tuple[float, float, float, float]] = None
    initial_components: List[Any] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    replicates: bool = False
    can_be_damaged: bool = True
    auto_possess_player: int = -1  # -1 means no auto possess


class Actor:
    """Base class for all gameplay entities in the world."""

    _id_counter: int = 0

    def __init__(
        self,
        name: str = "",
        actor_type: ActorType = ActorType.STATIC,
    ) -> None:
        Actor._id_counter += 1
        self._actor_id: int = Actor._id_counter
        self._name: str = name or f"Actor_{self._actor_id}"
        self._actor_type: ActorType = actor_type
        self._state: EntityState = EntityState.CREATING
        self._entity: Optional[Entity] = None
        self._world: Optional[World] = None
        self._components: Dict[Type, Any] = {}
        self._tags: Set[str] = set()
        self._owner: Optional[Actor] = None
        self._children: List[Actor] = []
        self._root_component: Optional[Any] = None
        self._is_active: bool = True
        self._pending_destroy: bool = False
        self._lifespan: float = 0.0  # 0 = infinite
        self._instigator: Optional[Actor] = None

    @property
    def actor_id(self) -> int:
        """Unique actor identifier."""
        return self._actor_id

    @property
    def name(self) -> str:
        """Actor name."""
        return self._name

    @property
    def actor_type(self) -> ActorType:
        """Actor type classification."""
        return self._actor_type

    @property
    def state(self) -> EntityState:
        """Current lifecycle state."""
        return self._state

    @property
    def entity(self) -> Optional[Entity]:
        """Underlying ECS entity."""
        return self._entity

    @property
    def is_active(self) -> bool:
        """Whether actor is active in the world."""
        return self._is_active and self._state == EntityState.ACTIVE

    def begin_play(self) -> None:
        """Called when actor begins play in the world."""
        self._state = EntityState.ACTIVE

    def end_play(self) -> None:
        """Called when actor is removed from the world."""
        self._state = EntityState.DEACTIVATING

    def tick(self, delta_time: float) -> None:
        """Called every frame when actor is active."""
        if self._lifespan > 0:
            self._lifespan -= delta_time
            if self._lifespan <= 0:
                self.destroy()

    def destroy(self) -> None:
        """Mark actor for destruction."""
        if not self._pending_destroy:
            self._pending_destroy = True
            self._state = EntityState.DESTROYING

    def add_component(self, component: Any) -> None:
        """Add component to actor."""
        component_type = type(component)
        self._components[component_type] = component

    def get_component(self, component_type: Type[T]) -> Optional[T]:
        """Get component by type."""
        return self._components.get(component_type)

    def has_component(self, component_type: Type) -> bool:
        """Check if actor has component type."""
        return component_type in self._components

    def remove_component(self, component_type: Type) -> bool:
        """Remove component by type."""
        if component_type in self._components:
            del self._components[component_type]
            return True
        return False

    def add_tag(self, tag: str) -> None:
        """Add tag to actor."""
        self._tags.add(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove tag from actor."""
        self._tags.discard(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if actor has tag."""
        return tag in self._tags

    def set_owner(self, owner: Optional[Actor]) -> None:
        """Set actor owner."""
        if self._owner:
            self._owner._children.remove(self)
        self._owner = owner
        if owner:
            owner._children.append(self)

    def get_owner(self) -> Optional[Actor]:
        """Get actor owner."""
        return self._owner


T = TypeVar("T")


class Pawn(Actor):
    """Actor that can be possessed by a controller."""

    def __init__(
        self,
        name: str = "",
    ) -> None:
        super().__init__(name, ActorType.PAWN)
        self._controller: Optional[Controller] = None
        self._auto_possess_player: int = -1
        self._can_be_possessed: bool = True

    @property
    def controller(self) -> Optional[Controller]:
        """Current possessing controller."""
        return self._controller

    @property
    def is_possessed(self) -> bool:
        """Whether pawn is currently possessed."""
        return self._controller is not None

    def possess(self, controller: Controller) -> bool:
        """Possess this pawn with controller."""
        if not self._can_be_possessed:
            return False

        if self._controller:
            self._controller.unpossess()

        self._controller = controller
        controller._possessed_pawn = self
        self.on_possessed(controller)
        return True

    def unpossess(self) -> None:
        """Release pawn from controller."""
        if self._controller:
            controller = self._controller
            self._controller._possessed_pawn = None
            self._controller = None
            self.on_unpossessed(controller)

    def on_possessed(self, controller: Controller) -> None:
        """Called when possessed by controller."""
        pass

    def on_unpossessed(self, controller: Controller) -> None:
        """Called when released from controller."""
        pass

    def get_movement_input(self) -> Tuple[float, float, float]:
        """Get movement input from controller."""
        if self._controller:
            return self._controller.get_movement_input()
        return (0.0, 0.0, 0.0)

    def get_look_input(self) -> Tuple[float, float]:
        """Get look input from controller."""
        if self._controller:
            return self._controller.get_look_input()
        return (0.0, 0.0)


class Character(Pawn):
    """Humanoid pawn with built-in movement capabilities."""

    def __init__(
        self,
        name: str = "",
    ) -> None:
        super().__init__(name)
        self._actor_type = ActorType.CHARACTER
        self._movement_speed: float = DEFAULT_MAX_WALK_SPEED
        self._jump_force: float = DEFAULT_JUMP_FORCE
        self._is_crouched: bool = False
        self._is_sprinting: bool = False
        self._is_jumping: bool = False
        self._is_falling: bool = False
        self._health: float = DEFAULT_CHARACTER_HEALTH
        self._max_health: float = DEFAULT_CHARACTER_MAX_HEALTH

    @property
    def movement_speed(self) -> float:
        """Current movement speed."""
        speed = self._movement_speed
        if self._is_crouched:
            speed *= CROUCH_SPEED_MULTIPLIER
        elif self._is_sprinting:
            speed *= SPRINT_SPEED_MULTIPLIER
        return speed

    @property
    def is_alive(self) -> bool:
        """Whether character is alive."""
        return self._health > 0

    def jump(self) -> bool:
        """Initiate jump if possible."""
        if not self._is_jumping and not self._is_falling:
            self._is_jumping = True
            return True
        return False

    def crouch(self) -> None:
        """Enter crouch state."""
        self._is_crouched = True
        self._is_sprinting = False

    def uncrouch(self) -> None:
        """Exit crouch state."""
        self._is_crouched = False

    def sprint(self, enable: bool = True) -> None:
        """Set sprint state."""
        if enable and not self._is_crouched:
            self._is_sprinting = True
        else:
            self._is_sprinting = False

    def take_damage(
        self,
        amount: float,
        damage_type: int = 0,
        instigator: Optional[Actor] = None,
    ) -> float:
        """Apply damage to character. Returns actual damage dealt."""
        if amount <= 0 or not self.is_alive:
            return 0.0

        actual_damage = min(amount, self._health)
        self._health -= actual_damage

        if self._health <= 0:
            self._health = 0
            self.on_death(instigator)

        return actual_damage

    def heal(self, amount: float) -> float:
        """Heal character. Returns actual healing done."""
        if amount <= 0 or not self.is_alive:
            return 0.0

        actual_heal = min(amount, self._max_health - self._health)
        self._health += actual_heal
        return actual_heal

    def on_death(self, killer: Optional[Actor]) -> None:
        """Called when character dies."""
        pass

    def on_landed(self) -> None:
        """Called when character lands after falling/jumping."""
        self._is_jumping = False
        self._is_falling = False


# === Controller System ===

class Controller(ABC):
    """Base class for pawn controllers."""

    _id_counter: int = 0

    def __init__(self) -> None:
        Controller._id_counter += 1
        self._controller_id: int = Controller._id_counter
        self._possessed_pawn: Optional[Pawn] = None
        self._is_active: bool = True

    @property
    def controller_id(self) -> int:
        """Unique controller identifier."""
        return self._controller_id

    @property
    def pawn(self) -> Optional[Pawn]:
        """Currently possessed pawn."""
        return self._possessed_pawn

    def possess(self, pawn: Pawn) -> bool:
        """Possess a pawn."""
        return pawn.possess(self)

    def unpossess(self) -> None:
        """Release current pawn."""
        if self._possessed_pawn:
            self._possessed_pawn.unpossess()

    @abstractmethod
    def get_movement_input(self) -> Tuple[float, float, float]:
        """Get movement input vector."""
        pass

    @abstractmethod
    def get_look_input(self) -> Tuple[float, float]:
        """Get look input (yaw, pitch)."""
        pass

    def tick(self, delta_time: float) -> None:
        """Update controller state."""
        pass


class PlayerController(Controller):
    """Controller driven by player input."""

    def __init__(self, player_index: int = 0) -> None:
        super().__init__()
        self._player_index: int = player_index
        self._movement_input: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._look_input: Tuple[float, float] = (0.0, 0.0)
        self._input_enabled: bool = True

    @property
    def player_index(self) -> int:
        """Player index for local multiplayer."""
        return self._player_index

    def get_movement_input(self) -> Tuple[float, float, float]:
        """Get movement input from player."""
        if self._input_enabled:
            return self._movement_input
        return (0.0, 0.0, 0.0)

    def get_look_input(self) -> Tuple[float, float]:
        """Get look input from player."""
        if self._input_enabled:
            return self._look_input
        return (0.0, 0.0)

    def set_movement_input(self, x: float, y: float, z: float) -> None:
        """Set movement input values."""
        self._movement_input = (x, y, z)

    def set_look_input(self, yaw: float, pitch: float) -> None:
        """Set look input values."""
        self._look_input = (yaw, pitch)

    def enable_input(self, enabled: bool = True) -> None:
        """Enable or disable input processing."""
        self._input_enabled = enabled


class AIController(Controller):
    """Controller driven by AI systems."""

    def __init__(self) -> None:
        super().__init__()
        self._movement_input: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._look_target: Optional[Tuple[float, float, float]] = None
        self._blackboard: Dict[str, Any] = {}
        self._behavior_tree: Optional[Any] = None

    def get_movement_input(self) -> Tuple[float, float, float]:
        """Get AI-computed movement input."""
        return self._movement_input

    def get_look_input(self) -> Tuple[float, float]:
        """Get AI-computed look input."""
        # AI typically looks at targets rather than using delta inputs
        return (0.0, 0.0)

    def set_movement_input(self, x: float, y: float, z: float) -> None:
        """Set AI movement input."""
        self._movement_input = (x, y, z)

    def set_look_target(self, target: Optional[Tuple[float, float, float]]) -> None:
        """Set position to look at."""
        self._look_target = target

    def set_blackboard_value(self, key: str, value: Any) -> None:
        """Set blackboard value."""
        self._blackboard[key] = value

    def get_blackboard_value(self, key: str, default: Any = None) -> Any:
        """Get blackboard value."""
        return self._blackboard.get(key, default)

    def set_behavior_tree(self, tree: Any) -> None:
        """Set behavior tree for this controller."""
        self._behavior_tree = tree


# === Prefab System ===

@dataclass
class PrefabComponent:
    """Component definition for prefab."""

    component_type: Type
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prefab:
    """Reusable actor template."""

    name: str
    actor_class: Type[Actor] = Actor
    components: List[PrefabComponent] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    children: List[Prefab] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def spawn(
        self,
        world: Optional[World] = None,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None,
        override_properties: Optional[Dict[str, Any]] = None,
    ) -> Actor:
        """Spawn actor instance from prefab."""
        # Merge properties with overrides
        props = {**self.properties}
        if override_properties:
            props.update(override_properties)

        # Create actor instance
        actor = self.actor_class(name=self.name)

        # Add components
        for comp_def in self.components:
            component = comp_def.component_type(**comp_def.properties)
            actor.add_component(component)

        # Add tags
        for tag in self.tags:
            actor.add_tag(tag)

        # Apply properties
        for key, value in props.items():
            if hasattr(actor, key):
                setattr(actor, key, value)

        # Spawn children
        for child_prefab in self.children:
            child_actor = child_prefab.spawn(world, override_properties=override_properties)
            child_actor.set_owner(actor)

        return actor


class PrefabRegistry:
    """Registry for prefab templates."""

    _instance: Optional[PrefabRegistry] = None

    def __new__(cls) -> PrefabRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._prefabs = {}
        return cls._instance

    def register(self, prefab: Prefab) -> None:
        """Register a prefab template."""
        self._prefabs[prefab.name] = prefab

    def unregister(self, name: str) -> bool:
        """Unregister a prefab template."""
        if name in self._prefabs:
            del self._prefabs[name]
            return True
        return False

    def get(self, name: str) -> Optional[Prefab]:
        """Get prefab by name."""
        return self._prefabs.get(name)

    def spawn(
        self,
        name: str,
        world: Optional[World] = None,
        **kwargs: Any,
    ) -> Optional[Actor]:
        """Spawn actor from registered prefab."""
        prefab = self.get(name)
        if prefab:
            return prefab.spawn(world, **kwargs)
        return None

    def list_prefabs(self) -> List[str]:
        """List all registered prefab names."""
        return list(self._prefabs.keys())


# === Lifecycle Management ===

class LifecycleEvent(IntEnum):
    """Entity lifecycle events."""
    CREATED = auto()
    INITIALIZED = auto()
    ACTIVATED = auto()
    DEACTIVATED = auto()
    DESTROYED = auto()


LifecycleCallback = Callable[[Actor, LifecycleEvent], None]


class LifecycleManager:
    """Manages actor lifecycle events and transitions."""

    _instance: Optional[LifecycleManager] = None

    def __new__(cls) -> LifecycleManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._actors = {}
            cls._instance._pending_spawn = []
            cls._instance._pending_destroy = []
            cls._instance._callbacks = []
        return cls._instance

    def register_actor(self, actor: Actor) -> None:
        """Register actor with lifecycle manager."""
        self._actors[actor.actor_id] = actor
        self._notify(actor, LifecycleEvent.CREATED)

    def unregister_actor(self, actor: Actor) -> None:
        """Unregister actor from lifecycle manager."""
        if actor.actor_id in self._actors:
            del self._actors[actor.actor_id]

    def queue_spawn(self, actor: Actor) -> None:
        """Queue actor for deferred spawning."""
        self._pending_spawn.append(actor)

    def queue_destroy(self, actor: Actor) -> None:
        """Queue actor for deferred destruction."""
        self._pending_destroy.append(actor)

    def process_pending(self) -> None:
        """Process pending spawn and destroy operations."""
        # Process spawns
        for actor in self._pending_spawn:
            actor._state = EntityState.INITIALIZING
            self._notify(actor, LifecycleEvent.INITIALIZED)
            actor.begin_play()
            self._notify(actor, LifecycleEvent.ACTIVATED)
        self._pending_spawn.clear()

        # Process destroys
        for actor in self._pending_destroy:
            actor.end_play()
            self._notify(actor, LifecycleEvent.DEACTIVATED)
            self._notify(actor, LifecycleEvent.DESTROYED)
            self.unregister_actor(actor)
        self._pending_destroy.clear()

    def add_callback(self, callback: LifecycleCallback) -> None:
        """Add lifecycle event callback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: LifecycleCallback) -> None:
        """Remove lifecycle event callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify(self, actor: Actor, event: LifecycleEvent) -> None:
        """Notify callbacks of lifecycle event."""
        for callback in self._callbacks:
            callback(actor, event)

    def get_actor(self, actor_id: int) -> Optional[Actor]:
        """Get actor by ID."""
        return self._actors.get(actor_id)

    def get_actors_by_type(self, actor_type: ActorType) -> List[Actor]:
        """Get all actors of given type."""
        return [a for a in self._actors.values() if a.actor_type == actor_type]

    def get_actors_with_tag(self, tag: str) -> List[Actor]:
        """Get all actors with given tag."""
        return [a for a in self._actors.values() if a.has_tag(tag)]

    def get_all_actors(self) -> List[Actor]:
        """Get all registered actors."""
        return list(self._actors.values())


from engine.gameplay.entity.eventlog_integration import (
    EntitySpawned,
    EntityDestroyed,
    ComponentAdded,
    ComponentRemoved,
    EntityStateChanged,
    EntityEventLog,
    get_entity_event_log,
    clear_entity_event_log,
)


__all__ = [
    # Config
    "ActorConfig",
    # Actors
    "Actor",
    "Pawn",
    "Character",
    # Controllers
    "Controller",
    "PlayerController",
    "AIController",
    # Prefabs
    "PrefabComponent",
    "Prefab",
    "PrefabRegistry",
    # Lifecycle
    "LifecycleEvent",
    "LifecycleCallback",
    "LifecycleManager",
    # EventLog Integration
    "EntitySpawned",
    "EntityDestroyed",
    "ComponentAdded",
    "ComponentRemoved",
    "EntityStateChanged",
    "EntityEventLog",
    "get_entity_event_log",
    "clear_entity_event_log",
]
