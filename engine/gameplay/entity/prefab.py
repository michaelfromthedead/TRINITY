"""
Prefab/Blueprint System
=======================
Prefab templates for entity instantiation with:
- Component composition
- Inheritance/extension
- Deferred instantiation
- Property overrides

Uses the Trinity Pattern with:
- PrefabMeta metaclass for registration
- @prefab decorator for template definition
- @extends decorator for inheritance
"""
from __future__ import annotations

import copy
import logging
import threading
import weakref
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
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

from trinity.decorators.ops import Op, Step, make_decorator, run_steps
from trinity.metaclasses.engine_meta import EngineMeta

from .actor import Actor, Transform
from .constants import (
    DEFAULT_PREFAB_CACHE_SIZE,
    MAX_PREFAB_INHERITANCE_DEPTH,
    PREFAB_INSTANCE_BATCH_SIZE,
    LifecycleState,
)

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=Actor)
_logger = logging.getLogger(__name__)


# =============================================================================
# PREFAB DATA STRUCTURES
# =============================================================================


@dataclass
class ComponentDefinition:
    """Definition of a component to add to a prefab instance."""

    name: str
    component_type: type
    properties: Dict[str, Any] = field(default_factory=dict)
    factory: Optional[Callable[[], Any]] = None

    def create_instance(self) -> Any:
        """Create an instance of this component."""
        if self.factory:
            instance = self.factory()
        else:
            instance = self.component_type()

        # Apply property overrides
        for name, value in self.properties.items():
            if hasattr(instance, name):
                setattr(instance, name, copy.deepcopy(value))

        return instance


@dataclass
class PropertyOverride:
    """Property override for prefab instances."""

    property_path: str  # e.g., "transform.position" or "health.current"
    value: Any


@dataclass
class PrefabDefinition:
    """Complete definition of a prefab template."""

    name: str
    actor_class: Type[Actor]
    components: Dict[str, ComponentDefinition] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    parent_prefab: Optional[str] = None
    transform: Optional[Transform] = None


# =============================================================================
# PREFAB REGISTRY
# =============================================================================


class PrefabRegistry:
    """
    Global registry for prefab templates.

    Features:
    - Prefab registration and lookup
    - Inheritance resolution
    - Caching of resolved prefabs
    """

    _instance: ClassVar[Optional["PrefabRegistry"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "PrefabRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._prefabs: Dict[str, PrefabDefinition] = {}
        self._resolved_cache: Dict[str, PrefabDefinition] = {}
        self._cache_size = DEFAULT_PREFAB_CACHE_SIZE
        self._initialized = True

    def register(self, prefab: PrefabDefinition) -> None:
        """Register a prefab definition."""
        self._prefabs[prefab.name] = prefab
        # Invalidate cache for this prefab and its children
        self._invalidate_cache(prefab.name)

    def unregister(self, name: str) -> Optional[PrefabDefinition]:
        """Unregister a prefab definition."""
        prefab = self._prefabs.pop(name, None)
        if prefab:
            self._invalidate_cache(name)
        return prefab

    def get(self, name: str) -> Optional[PrefabDefinition]:
        """Get a prefab definition by name."""
        return self._prefabs.get(name)

    def get_resolved(self, name: str) -> Optional[PrefabDefinition]:
        """Get a fully resolved prefab (with inheritance applied)."""
        if name in self._resolved_cache:
            return self._resolved_cache[name]

        prefab = self._prefabs.get(name)
        if prefab is None:
            return None

        resolved = self._resolve_inheritance(prefab)
        if resolved:
            self._resolved_cache[name] = resolved
        return resolved

    def _resolve_inheritance(
        self,
        prefab: PrefabDefinition,
        depth: int = 0,
    ) -> Optional[PrefabDefinition]:
        """Resolve prefab inheritance chain."""
        if depth > MAX_PREFAB_INHERITANCE_DEPTH:
            raise RecursionError(
                f"Prefab inheritance depth exceeded ({MAX_PREFAB_INHERITANCE_DEPTH})"
            )

        if prefab.parent_prefab is None:
            # No parent, return copy
            return PrefabDefinition(
                name=prefab.name,
                actor_class=prefab.actor_class,
                components=dict(prefab.components),
                properties=dict(prefab.properties),
                tags=set(prefab.tags),
                parent_prefab=None,
                transform=prefab.transform.copy() if prefab.transform else None,
            )

        # Resolve parent first
        parent = self._prefabs.get(prefab.parent_prefab)
        if parent is None:
            raise ValueError(
                f"Parent prefab '{prefab.parent_prefab}' not found for '{prefab.name}'"
            )

        resolved_parent = self._resolve_inheritance(parent, depth + 1)
        if resolved_parent is None:
            return None

        # Merge child onto parent
        return PrefabDefinition(
            name=prefab.name,
            actor_class=prefab.actor_class or resolved_parent.actor_class,
            components={**resolved_parent.components, **prefab.components},
            properties={**resolved_parent.properties, **prefab.properties},
            tags=resolved_parent.tags | prefab.tags,
            parent_prefab=None,  # Already resolved
            transform=prefab.transform or resolved_parent.transform,
        )

    def _invalidate_cache(self, name: str) -> None:
        """Invalidate cache for a prefab and all its children."""
        # Remove from cache
        self._resolved_cache.pop(name, None)

        # Find and invalidate children
        for prefab_name, prefab in self._prefabs.items():
            if prefab.parent_prefab == name:
                self._invalidate_cache(prefab_name)

    def list_prefabs(self) -> List[str]:
        """List all registered prefab names."""
        return list(self._prefabs.keys())

    def get_children(self, name: str) -> List[str]:
        """Get all prefabs that extend the given prefab."""
        return [
            prefab_name
            for prefab_name, prefab in self._prefabs.items()
            if prefab.parent_prefab == name
        ]

    def clear(self) -> None:
        """Clear all registered prefabs (for testing)."""
        self._prefabs.clear()
        self._resolved_cache.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
            cls._instance = None


# =============================================================================
# PREFAB INSTANTIATOR
# =============================================================================


class PrefabInstantiator:
    """
    Handles instantiation of prefab templates.

    Features:
    - Deferred instantiation (batched to end of frame)
    - Property overrides at instantiation time
    - Instance caching for frequently used prefabs
    """

    _instance: ClassVar[Optional["PrefabInstantiator"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "PrefabInstantiator":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._registry = PrefabRegistry()
        self._pending_instantiations: List[Tuple[str, Dict[str, Any], Callable[[Actor], None]]] = []
        self._instantiation_lock = threading.Lock()
        self._initialized = True

    def instantiate(
        self,
        prefab_name: str,
        overrides: Optional[Dict[str, Any]] = None,
        transform: Optional[Transform] = None,
        immediate: bool = False,
    ) -> Optional[Actor]:
        """
        Instantiate a prefab.

        Args:
            prefab_name: Name of the prefab to instantiate
            overrides: Optional property overrides
            transform: Optional transform override
            immediate: If True, create immediately; otherwise defer

        Returns:
            The created actor (only if immediate=True)
        """
        if immediate:
            return self._create_instance(prefab_name, overrides or {}, transform)

        # Defer instantiation
        def callback(actor: Actor) -> None:
            pass  # No-op callback for basic deferred instantiation

        with self._instantiation_lock:
            self._pending_instantiations.append((prefab_name, overrides or {}, callback))

        return None

    def instantiate_async(
        self,
        prefab_name: str,
        callback: Callable[[Actor], None],
        overrides: Optional[Dict[str, Any]] = None,
        transform: Optional[Transform] = None,
    ) -> None:
        """
        Instantiate a prefab with a callback when complete.

        Args:
            prefab_name: Name of the prefab to instantiate
            callback: Called with the created actor
            overrides: Optional property overrides
            transform: Optional transform override
        """
        with self._instantiation_lock:
            self._pending_instantiations.append((prefab_name, overrides or {}, callback))

    def _create_instance(
        self,
        prefab_name: str,
        overrides: Dict[str, Any],
        transform: Optional[Transform] = None,
    ) -> Optional[Actor]:
        """Create an actor instance from a prefab definition."""
        prefab = self._registry.get_resolved(prefab_name)
        if prefab is None:
            return None

        # Create base actor
        actor_transform = transform or prefab.transform or Transform()
        actor = prefab.actor_class(
            name=f"{prefab.name}_instance",
            transform=actor_transform,
            tags=prefab.tags.copy(),
        )

        # Add components
        for name, component_def in prefab.components.items():
            component = component_def.create_instance()
            actor.add_component(name, component)

        # Apply prefab properties
        for prop_path, value in prefab.properties.items():
            self._apply_property(actor, prop_path, value)

        # Apply overrides
        for prop_path, value in overrides.items():
            self._apply_property(actor, prop_path, value)

        return actor

    def _apply_property(self, actor: Actor, path: str, value: Any) -> None:
        """Apply a property value using dot-notation path."""
        parts = path.split(".")
        obj = actor

        # Navigate to parent of target property
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif hasattr(obj, "_components") and obj._components.has(part):
                obj = obj._components.get(part)
            else:
                return  # Path not found, skip

        # Set the final property
        final_part = parts[-1]
        if hasattr(obj, final_part):
            setattr(obj, final_part, copy.deepcopy(value))

    def process_pending(self) -> List[Actor]:
        """
        Process all pending instantiations.

        Called at the end of each frame.

        Returns:
            List of created actors
        """
        created = []

        with self._instantiation_lock:
            batch = self._pending_instantiations[:PREFAB_INSTANCE_BATCH_SIZE]
            self._pending_instantiations = self._pending_instantiations[PREFAB_INSTANCE_BATCH_SIZE:]

        for prefab_name, overrides, callback in batch:
            actor = self._create_instance(prefab_name, overrides)
            if actor is not None:
                created.append(actor)
                try:
                    callback(actor)
                except Exception as e:
                    # Log error but don't let callback errors break instantiation
                    _logger.warning(
                        "Prefab instantiation callback failed for '%s': %s",
                        prefab_name,
                        e,
                    )

        return created

    def clear_pending(self) -> int:
        """Clear all pending instantiations."""
        with self._instantiation_lock:
            count = len(self._pending_instantiations)
            self._pending_instantiations.clear()
        return count

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear_pending()
            cls._instance = None


# =============================================================================
# PREFAB DECORATOR
# =============================================================================


def _build_prefab_steps(params: dict) -> list[Step]:
    """Build steps for prefab decorator."""
    name = params.get("name", "")
    return [
        Step(Op.TAG, {"key": "prefab", "value": True}),
        Step(Op.TAG, {"key": "prefab_name", "value": name}),
        Step(Op.REGISTER, {"registry": "prefabs"}),
    ]


def _validate_prefab_params(name: str = "", **kwargs: Any) -> None:
    """Validate prefab decorator parameters."""
    if not name:
        raise ValueError("'name' parameter is required for @prefab decorator")


def _after_prefab_steps(target: Any, params: dict) -> Any:
    """Post-processing for prefab decorator."""
    name = params.get("name", "")

    # Store prefab metadata
    target._prefab = True
    target._prefab_name = name

    # Create and register prefab definition
    if isinstance(target, type) and issubclass(target, Actor):
        definition = PrefabDefinition(
            name=name,
            actor_class=target,
            components={},
            properties={},
            tags=set(),
            parent_prefab=None,
            transform=None,
        )

        # Collect component declarations from class
        for attr_name in dir(target):
            if attr_name.startswith("_"):
                continue
            attr = getattr(target, attr_name, None)
            if attr is not None and hasattr(attr, "_component"):
                definition.components[attr_name] = ComponentDefinition(
                    name=attr_name,
                    component_type=type(attr),
                    properties={},
                )

        # Register with global registry
        registry = PrefabRegistry()
        registry.register(definition)

    return target


prefab = make_decorator(
    name="prefab",
    steps=_build_prefab_steps,
    doc="Register a class as a prefab template.",
    validate=_validate_prefab_params,
    after_steps=_after_prefab_steps,
)


# =============================================================================
# EXTENDS DECORATOR
# =============================================================================


def _build_extends_steps(params: dict) -> list[Step]:
    """Build steps for extends decorator."""
    parent = params.get("parent", "")
    return [
        Step(Op.TAG, {"key": "extends", "value": True}),
        Step(Op.TAG, {"key": "extends_parent", "value": parent}),
        Step(Op.REGISTER, {"registry": "prefabs"}),
    ]


def _validate_extends_params(parent: str = "", **kwargs: Any) -> None:
    """Validate extends decorator parameters."""
    if not parent:
        raise ValueError("'parent' parameter is required for @extends decorator")


def _after_extends_steps(target: Any, params: dict) -> Any:
    """Post-processing for extends decorator."""
    parent = params.get("parent", "")

    # Store extends metadata
    target._extends = True
    target._extends_parent = parent

    # Update prefab definition if this is also a prefab
    if hasattr(target, "_prefab_name"):
        registry = PrefabRegistry()
        existing = registry.get(target._prefab_name)
        if existing:
            existing.parent_prefab = parent
            # Re-register to invalidate cache
            registry.register(existing)

    return target


extends = make_decorator(
    name="extends",
    steps=_build_extends_steps,
    doc="Extend a parent prefab template.",
    validate=_validate_extends_params,
    after_steps=_after_extends_steps,
)


# =============================================================================
# PREFAB BUILDER (FLUENT API)
# =============================================================================


class PrefabBuilder(Generic[T]):
    """
    Fluent API for building prefab definitions.

    Usage:
        player = (PrefabBuilder("player", Character)
            .with_component("health", HealthComponent, max_health=100)
            .with_component("inventory", InventoryComponent)
            .with_property("max_walk_speed", 6.0)
            .with_tag("player")
            .with_transform(position=(0, 0, 0))
            .build())
    """

    def __init__(self, name: str, actor_class: Type[T]) -> None:
        self._name = name
        self._actor_class = actor_class
        self._components: Dict[str, ComponentDefinition] = {}
        self._properties: Dict[str, Any] = {}
        self._tags: Set[str] = set()
        self._parent: Optional[str] = None
        self._transform: Optional[Transform] = None

    def with_component(
        self,
        name: str,
        component_type: type,
        factory: Optional[Callable[[], Any]] = None,
        **properties: Any,
    ) -> "PrefabBuilder[T]":
        """Add a component definition."""
        self._components[name] = ComponentDefinition(
            name=name,
            component_type=component_type,
            properties=properties,
            factory=factory,
        )
        return self

    def with_property(self, path: str, value: Any) -> "PrefabBuilder[T]":
        """Set a property override."""
        self._properties[path] = value
        return self

    def with_tag(self, tag: str) -> "PrefabBuilder[T]":
        """Add a tag."""
        self._tags.add(tag)
        return self

    def with_tags(self, *tags: str) -> "PrefabBuilder[T]":
        """Add multiple tags."""
        self._tags.update(tags)
        return self

    def extends(self, parent_prefab: str) -> "PrefabBuilder[T]":
        """Set parent prefab for inheritance."""
        self._parent = parent_prefab
        return self

    def with_transform(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None,
        scale: Optional[Tuple[float, float, float]] = None,
    ) -> "PrefabBuilder[T]":
        """Set the default transform."""
        self._transform = Transform(
            position=position or (0.0, 0.0, 0.0),
            rotation=rotation or (0.0, 0.0, 0.0, 1.0),
            scale=scale or (1.0, 1.0, 1.0),
        )
        return self

    def build(self) -> PrefabDefinition:
        """Build and register the prefab definition."""
        definition = PrefabDefinition(
            name=self._name,
            actor_class=self._actor_class,
            components=self._components,
            properties=self._properties,
            tags=self._tags,
            parent_prefab=self._parent,
            transform=self._transform,
        )

        # Register with global registry
        registry = PrefabRegistry()
        registry.register(definition)

        return definition

    def instantiate(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        transform: Optional[Transform] = None,
    ) -> T:
        """Build the prefab and create an instance."""
        self.build()
        instantiator = PrefabInstantiator()
        actor = instantiator.instantiate(
            self._name,
            overrides=overrides,
            transform=transform,
            immediate=True,
        )
        return actor  # type: ignore


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def spawn_prefab(
    prefab_name: str,
    position: Optional[Tuple[float, float, float]] = None,
    rotation: Optional[Tuple[float, float, float, float]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    immediate: bool = True,
) -> Optional[Actor]:
    """
    Spawn an actor from a prefab.

    Args:
        prefab_name: Name of the prefab to spawn
        position: Optional spawn position
        rotation: Optional spawn rotation
        overrides: Optional property overrides
        immediate: If True, spawn immediately; otherwise defer

    Returns:
        The spawned actor (only if immediate=True)
    """
    transform = None
    if position or rotation:
        transform = Transform(
            position=position or (0.0, 0.0, 0.0),
            rotation=rotation or (0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
        )

    instantiator = PrefabInstantiator()
    return instantiator.instantiate(
        prefab_name,
        overrides=overrides,
        transform=transform,
        immediate=immediate,
    )


def register_prefab(
    name: str,
    actor_class: Type[Actor],
    components: Optional[Dict[str, ComponentDefinition]] = None,
    properties: Optional[Dict[str, Any]] = None,
    tags: Optional[Set[str]] = None,
    parent: Optional[str] = None,
) -> PrefabDefinition:
    """
    Register a prefab programmatically.

    Args:
        name: Unique prefab name
        actor_class: Base actor class
        components: Component definitions
        properties: Default property values
        tags: Default tags
        parent: Parent prefab name for inheritance

    Returns:
        The registered prefab definition
    """
    definition = PrefabDefinition(
        name=name,
        actor_class=actor_class,
        components=components or {},
        properties=properties or {},
        tags=tags or set(),
        parent_prefab=parent,
        transform=None,
    )

    registry = PrefabRegistry()
    registry.register(definition)
    return definition


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "ComponentDefinition",
    "PropertyOverride",
    "PrefabDefinition",
    # Registry
    "PrefabRegistry",
    # Instantiator
    "PrefabInstantiator",
    # Decorators
    "prefab",
    "extends",
    # Builder
    "PrefabBuilder",
    # Convenience functions
    "spawn_prefab",
    "register_prefab",
]
