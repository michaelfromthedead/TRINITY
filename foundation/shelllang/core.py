"""
ShellLang Core - The 5 semantic primitives.

ENTITY      uint64 identifier for game objects
COMPONENT   typed data attached to entity
QUERY       entity predicate -> [entity]
MUTATE      (entity, field, value) -> tracked change
SNAPSHOT    frozen world state
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Type, TypeVar

# =============================================================================
# CONSTANTS
# =============================================================================

ENTITY_ID_START = 1
DEFAULT_HISTORY_COUNT = 10


# =============================================================================
# CORE TYPES
# =============================================================================

T = TypeVar("T")


@dataclass
class Entity:
    """
    An entity is just an ID. Components give it meaning.

    Entities are lightweight handles - the actual data lives in components
    attached to the entity via the World.
    """
    id: int

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Entity):
            return self.id == other.id
        return False

    def __repr__(self) -> str:
        return f"Entity({self.id})"


# Component is just a type alias - any class can be a component
Component = Any


@dataclass
class Change:
    """
    A recorded mutation for undo/redo and debugging.
    """
    entity_id: int
    component_type: str
    field_name: str
    old_value: Any
    new_value: Any
    timestamp: float = 0.0

    def __repr__(self) -> str:
        return f"{self.component_type}.{self.field_name}: {self.old_value!r} → {self.new_value!r}"


@dataclass
class Snapshot:
    """
    A frozen world state that can be restored.
    """
    name: Optional[str]
    entities: Dict[int, Set[str]]  # entity_id -> set of component type names
    components: Dict[int, Dict[str, Any]]  # entity_id -> {type_name: component_copy}
    next_entity_id: int

    def __repr__(self) -> str:
        name_str = f'"{self.name}"' if self.name else "unnamed"
        return f"Snapshot({name_str}, {len(self.entities)} entities)"


# =============================================================================
# WORLD
# =============================================================================


class World:
    """
    The World holds all entities and their components.

    This is the core ECS container that ShellLang operates on.
    All mutations go through the World and are tracked for undo/redo.
    """

    def __init__(self) -> None:
        self._next_entity_id: int = ENTITY_ID_START
        self._entities: Dict[int, Set[str]] = {}  # id -> set of component type names
        self._components: Dict[int, Dict[str, Any]] = {}  # id -> {type_name: component}
        self._component_registry: Dict[str, Type] = {}  # name -> type
        self._changes: List[Change] = []
        self._time: float = 0.0

    # =========================================================================
    # ENTITY OPERATIONS
    # =========================================================================

    def create(self) -> Entity:
        """Create a new entity and return its handle."""
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        self._entities[entity_id] = set()
        self._components[entity_id] = {}
        return Entity(entity_id)

    def destroy(self, e: Entity) -> None:
        """Destroy an entity and all its components."""
        if e.id in self._entities:
            del self._entities[e.id]
            del self._components[e.id]

    def exists(self, e: Entity) -> bool:
        """Check if an entity exists."""
        return e.id in self._entities

    @property
    def entities(self) -> Iterator[Entity]:
        """Iterate over all entities."""
        for entity_id in self._entities:
            yield Entity(entity_id)

    def entity_count(self) -> int:
        """Return the number of entities."""
        return len(self._entities)

    # =========================================================================
    # COMPONENT OPERATIONS
    # =========================================================================

    def attach(self, e: Entity, c: Component) -> None:
        """Attach a component to an entity."""
        if e.id not in self._entities:
            raise ValueError(f"Entity {e.id} does not exist")

        type_name = type(c).__name__
        self._entities[e.id].add(type_name)
        self._components[e.id][type_name] = c

        # Register the component type if not already known
        if type_name not in self._component_registry:
            self._component_registry[type_name] = type(c)

    def detach(self, e: Entity, C: Type) -> None:
        """Detach a component type from an entity."""
        if e.id not in self._entities:
            return

        type_name = C.__name__
        self._entities[e.id].discard(type_name)
        self._components[e.id].pop(type_name, None)

    def get(self, e: Entity, C: Type[T]) -> Optional[T]:
        """Get a component from an entity, or None if not present."""
        if e.id not in self._components:
            return None
        return self._components[e.id].get(C.__name__)

    def has(self, e: Entity, C: Type) -> bool:
        """Check if an entity has a component type."""
        if e.id not in self._entities:
            return False
        return C.__name__ in self._entities[e.id]

    def set(self, e: Entity, C: Type, field_name: str, value: Any) -> None:
        """Set a field on a component, recording the change."""
        component = self.get(e, C)
        if component is None:
            raise ValueError(f"Entity {e.id} does not have component {C.__name__}")

        old_value = getattr(component, field_name, None)
        setattr(component, field_name, value)

        # Record the change
        change = Change(
            entity_id=e.id,
            component_type=C.__name__,
            field_name=field_name,
            old_value=old_value,
            new_value=value,
            timestamp=self._time,
        )
        self._changes.append(change)

    def components_of(self, e: Entity) -> List[Type]:
        """Get all component types attached to an entity."""
        if e.id not in self._entities:
            return []
        return [
            self._component_registry[name]
            for name in self._entities[e.id]
            if name in self._component_registry
        ]

    def register_component(self, C: Type) -> None:
        """Register a component type for lookup by name."""
        self._component_registry[C.__name__] = C

    def get_component_type(self, name: str) -> Optional[Type]:
        """Get a component type by name."""
        return self._component_registry.get(name)

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def query(self, *Cs: Type) -> List[Entity]:
        """
        Query for entities that have ALL specified component types.

        Returns a list of Entity objects.
        """
        if not Cs:
            return [Entity(eid) for eid in self._entities]

        required = {C.__name__ for C in Cs}
        results = []

        for entity_id, component_names in self._entities.items():
            if required <= component_names:
                results.append(Entity(entity_id))

        return results

    # =========================================================================
    # SNAPSHOT OPERATIONS
    # =========================================================================

    def snap(self, name: Optional[str] = None) -> Snapshot:
        """Create a snapshot of the current world state."""
        # Deep copy all components
        entities_copy = {
            eid: set(names) for eid, names in self._entities.items()
        }
        components_copy = {
            eid: {
                name: copy.deepcopy(comp)
                for name, comp in comps.items()
            }
            for eid, comps in self._components.items()
        }

        return Snapshot(
            name=name,
            entities=entities_copy,
            components=components_copy,
            next_entity_id=self._next_entity_id,
        )

    def restore(self, s: Snapshot) -> None:
        """Restore world state from a snapshot."""
        # Deep copy back to avoid sharing references
        self._entities = {
            eid: set(names) for eid, names in s.entities.items()
        }
        self._components = {
            eid: {
                name: copy.deepcopy(comp)
                for name, comp in comps.items()
            }
            for eid, comps in s.components.items()
        }
        self._next_entity_id = s.next_entity_id

    def diff(self, a: Snapshot, b: Snapshot) -> List[Change]:
        """
        Compute the differences between two snapshots.

        Returns a list of Change objects representing what changed
        from snapshot a to snapshot b.
        """
        changes: List[Change] = []

        # Find entities in b but not in a (created)
        for eid in b.entities:
            if eid not in a.entities:
                for comp_name in b.entities[eid]:
                    changes.append(Change(
                        entity_id=eid,
                        component_type=comp_name,
                        field_name="<attached>",
                        old_value=None,
                        new_value=b.components[eid][comp_name],
                    ))

        # Find entities in a but not in b (destroyed)
        for eid in a.entities:
            if eid not in b.entities:
                for comp_name in a.entities[eid]:
                    changes.append(Change(
                        entity_id=eid,
                        component_type=comp_name,
                        field_name="<detached>",
                        old_value=a.components[eid][comp_name],
                        new_value=None,
                    ))

        # Find component changes in entities that exist in both
        for eid in a.entities:
            if eid not in b.entities:
                continue

            # Components added
            for comp_name in b.entities[eid] - a.entities[eid]:
                changes.append(Change(
                    entity_id=eid,
                    component_type=comp_name,
                    field_name="<attached>",
                    old_value=None,
                    new_value=b.components[eid][comp_name],
                ))

            # Components removed
            for comp_name in a.entities[eid] - b.entities[eid]:
                changes.append(Change(
                    entity_id=eid,
                    component_type=comp_name,
                    field_name="<detached>",
                    old_value=a.components[eid][comp_name],
                    new_value=None,
                ))

            # Components changed (in both snapshots)
            for comp_name in a.entities[eid] & b.entities[eid]:
                comp_a = a.components[eid][comp_name]
                comp_b = b.components[eid][comp_name]

                # Compare field by field
                for field_name in dir(comp_a):
                    if field_name.startswith("_"):
                        continue
                    try:
                        val_a = getattr(comp_a, field_name)
                        val_b = getattr(comp_b, field_name)
                        if callable(val_a):
                            continue
                        if val_a != val_b:
                            changes.append(Change(
                                entity_id=eid,
                                component_type=comp_name,
                                field_name=field_name,
                                old_value=val_a,
                                new_value=val_b,
                            ))
                    except (AttributeError, TypeError):
                        pass

        return changes

    # =========================================================================
    # CHANGE TRACKING
    # =========================================================================

    def recent_changes(self, n: int = DEFAULT_HISTORY_COUNT) -> List[Change]:
        """Get the n most recent changes."""
        return self._changes[-n:]

    def clear_changes(self) -> None:
        """Clear the change history."""
        self._changes.clear()

    def tick(self, delta: float) -> None:
        """Advance world time."""
        self._time += delta


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Entity",
    "Component",
    "Change",
    "Snapshot",
    "World",
    # Constants
    "ENTITY_ID_START",
    "DEFAULT_HISTORY_COUNT",
]
