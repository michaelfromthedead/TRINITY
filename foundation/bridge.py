"""
Foundation Bridge - Integration between Trinity and Foundation systems.

Provides utilities for:
    - Creating ShellLang worlds from Trinity's component registry
    - Syncing Trinity instances with ShellLang entities
    - Unified querying across both systems
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from foundation.shelllang.core import World, Entity
    from foundation.shelllang.ai import AIInterface
    from foundation.shelllang.repl import Shell


def get_trinity_registry() -> Dict[str, Type]:
    """
    Get all Trinity components as a ShellLang-compatible registry.

    Returns:
        Dict mapping component names to component classes.
    """
    try:
        from trinity.metaclasses.component_meta import ComponentMeta
        return {cls.__name__: cls for cls in ComponentMeta.all_components()}
    except ImportError:
        return {}


def create_world_from_trinity() -> "World":
    """
    Create a ShellLang World pre-populated with Trinity's component registry.

    Returns:
        A new World instance with all Trinity components registered.
    """
    from foundation.shelllang.core import World

    world = World()
    registry = get_trinity_registry()

    # Register all Trinity component types
    for name, cls in registry.items():
        world.register_component(cls)

    return world


def create_ai_interface() -> "AIInterface":
    """
    Create an AIInterface connected to Trinity's component registry.

    Returns:
        An AIInterface instance ready for AI agent commands.
    """
    from foundation.shelllang.ai import AIInterface
    from foundation.shelllang.core import World

    registry = get_trinity_registry()
    world = World()
    for cls in registry.values():
        world.register_component(cls)
    return AIInterface(world, registry)


def create_shell() -> "Shell":
    """
    Create an interactive Shell connected to Trinity's component registry.

    Returns:
        A Shell instance ready for human interaction.
    """
    from foundation.shelllang.repl import Shell
    from foundation.shelllang.core import World

    registry = get_trinity_registry()
    world = World()
    for cls in registry.values():
        world.register_component(cls)
    return Shell(world, registry)


class TrinityWorldAdapter:
    """
    Adapter that syncs Trinity component instances with a ShellLang World.

    Provides bidirectional mapping between Trinity instances and ShellLang entities.
    """

    def __init__(self, world: Optional["World"] = None) -> None:
        """
        Initialize the adapter.

        Args:
            world: Optional existing World. If None, creates a new one.
        """
        if world is None:
            world = create_world_from_trinity()
        self._world = world
        self._registry = get_trinity_registry()
        self._instance_to_entity: Dict[int, "Entity"] = {}  # id(instance) -> Entity
        self._entity_to_instances: Dict[int, Dict[str, Any]] = {}  # entity.id -> {comp_name: instance}

    @property
    def world(self) -> "World":
        """Get the underlying ShellLang World."""
        return self._world

    @property
    def registry(self) -> Dict[str, Type]:
        """Get the component registry."""
        return self._registry

    def add_instance(self, instance: Any) -> "Entity":
        """
        Add a Trinity component instance to the World.

        If the instance is already tracked, returns its existing entity.

        Args:
            instance: A Trinity component instance.

        Returns:
            The Entity associated with this instance.
        """
        instance_id = id(instance)

        # Return existing entity if already tracked
        if instance_id in self._instance_to_entity:
            return self._instance_to_entity[instance_id]

        # Create new entity
        entity = self._world.create()
        self._world.attach(entity, instance)

        # Track mappings
        self._instance_to_entity[instance_id] = entity
        comp_name = type(instance).__name__
        if entity.id not in self._entity_to_instances:
            self._entity_to_instances[entity.id] = {}
        self._entity_to_instances[entity.id][comp_name] = instance

        return entity

    def get_entity(self, instance: Any) -> Optional["Entity"]:
        """
        Get the Entity for a Trinity component instance.

        Args:
            instance: A Trinity component instance.

        Returns:
            The associated Entity, or None if not tracked.
        """
        return self._instance_to_entity.get(id(instance))

    def get_instance(self, entity: "Entity", component_type: Type) -> Optional[Any]:
        """
        Get the Trinity instance for an entity and component type.

        Args:
            entity: A ShellLang Entity.
            component_type: The component class to retrieve.

        Returns:
            The component instance, or None if not found.
        """
        instances = self._entity_to_instances.get(entity.id, {})
        return instances.get(component_type.__name__)

    def remove_instance(self, instance: Any) -> None:
        """
        Remove a Trinity component instance from tracking.

        Args:
            instance: The instance to remove.
        """
        instance_id = id(instance)
        entity = self._instance_to_entity.pop(instance_id, None)

        if entity is not None:
            comp_name = type(instance).__name__
            instances = self._entity_to_instances.get(entity.id, {})
            instances.pop(comp_name, None)

            # Detach from world
            self._world.detach(entity, type(instance))

            # Clean up empty entity tracking
            if not instances:
                self._entity_to_instances.pop(entity.id, None)

    def sync_from_foundation_registry(self) -> None:
        """
        Sync tracked instances from Foundation's registry.

        For each Trinity component type that has track_instances=True,
        adds any live instances to the World.
        """
        try:
            from foundation import registry

            for cls in self._registry.values():
                if registry.is_registered(cls):
                    for instance in registry.instances(cls):
                        self.add_instance(instance)
        except ImportError:
            pass

    def all_instances(self, component_type: Type) -> Iterator[Any]:
        """
        Iterate over all tracked instances of a component type.

        Args:
            component_type: The component class to query.

        Yields:
            Component instances of the specified type.
        """
        comp_name = component_type.__name__
        for instances in self._entity_to_instances.values():
            instance = instances.get(comp_name)
            if instance is not None:
                yield instance


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "get_trinity_registry",
    "create_world_from_trinity",
    "create_ai_interface",
    "create_shell",
    "TrinityWorldAdapter",
]
