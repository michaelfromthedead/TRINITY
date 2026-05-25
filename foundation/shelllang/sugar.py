"""
ShellLang Sugar Layer - Ergonomic interfaces for humans.

Provides:
    EntityProxy      Dot access to components (e.health.current)
    ComponentProxy   Tracks field access and mutation
    QueryResult      Chainable queries (Enemy.all.where(...).near(...))
    TypeQuery        Type-based query entry (Enemy.all)
    TimeManager      Named snapshots and undo/redo
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from foundation.shelllang.core import Entity, World
    from foundation.shelllang.repl import Feedback

# =============================================================================
# CONSTANTS
# =============================================================================

MAX_DISPLAY_ENTITIES = 5
MAX_UNDO_STACK = 100
MAX_REDO_STACK = 100
DEFAULT_HISTORY_COUNT = 10


# =============================================================================
# MODULE-LEVEL STATE (set by Shell)
# =============================================================================

# These are set by the Shell when it initializes
_world: Optional["World"] = None
_echo: Optional["Feedback"] = None
_registry: Dict[str, Type] = {}


def set_world(world: "World") -> None:
    """Set the active world for sugar operations."""
    global _world
    _world = world


def set_echo(echo: "Feedback") -> None:
    """Set the feedback function for sugar operations."""
    global _echo
    _echo = echo


def set_registry(registry: Dict[str, Type]) -> None:
    """Set the component registry for sugar operations."""
    global _registry
    _registry = registry


def _get_world() -> "World":
    """Get the active world, raising if not set."""
    if _world is None:
        raise RuntimeError("No world set. Call set_world() first.")
    return _world


def _get_echo() -> Callable[[str], None]:
    """Get the echo function, or a no-op if not set."""
    if _echo is None:
        return lambda msg: None
    return _echo


# =============================================================================
# ENTITY PROXY
# =============================================================================


class ComponentProxy:
    """
    Proxy that tracks field access and mutation on a component.

    Enables syntax like:
        e.health.current = 50
        value = e.health.current
    """

    def __init__(self, entity: "Entity", component_type: Type) -> None:
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_type", component_type)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        world = _get_world()
        component = world.get(self._entity, self._type)
        if component is None:
            raise AttributeError(f"Entity {self._entity.id} has no {self._type.__name__}")
        return getattr(component, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        world = _get_world()
        echo = _get_echo()

        # Get old value for feedback
        component = world.get(self._entity, self._type)
        if component is None:
            raise AttributeError(f"Entity {self._entity.id} has no {self._type.__name__}")

        old = getattr(component, name, None)
        world.set(self._entity, self._type, name, value)

        # Echo feedback
        echo(f"{self._type.__name__}.{name}: {old!r} → {value!r}")

    def __repr__(self) -> str:
        world = _get_world()
        component = world.get(self._entity, self._type)
        if component is None:
            return f"<{self._type.__name__}: not attached>"
        return repr(component)


class EntityProxy:
    """
    Proxy that provides dot access to components.

    Enables syntax like:
        e.health.current
        e.position.x = 10
    """

    def __init__(self, entity: "Entity") -> None:
        object.__setattr__(self, "_entity", entity)

    @property
    def id(self) -> int:
        """Get the entity ID."""
        return self._entity.id

    def __getattr__(self, name: str) -> ComponentProxy:
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        # Look up component type by name
        type_name = name.title()
        component_type = _registry.get(type_name)

        if component_type is None:
            # Try exact name
            component_type = _registry.get(name)

        if component_type is None:
            raise AttributeError(f"Unknown component type: {name}")

        return ComponentProxy(self._entity, component_type)

    def __repr__(self) -> str:
        world = _get_world()
        if not world.exists(self._entity):
            return f"Entity({self._entity.id}, [DESTROYED])"

        components = world.components_of(self._entity)
        comp_names = [C.__name__ for C in components]
        return f"Entity({self._entity.id}, [{', '.join(comp_names)}])"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EntityProxy):
            return self._entity.id == other._entity.id
        if isinstance(other, int):
            return self._entity.id == other
        return False

    def __hash__(self) -> int:
        return hash(self._entity.id)


# =============================================================================
# QUERY RESULT
# =============================================================================


class QueryResult:
    """
    Chainable query result with fluent interface.

    Enables syntax like:
        Enemy.all.where(lambda e: e.health.current < 50).near(player, 10)
    """

    def __init__(self, entities: List["Entity"]) -> None:
        self._entities = entities

    # =========================================================================
    # FILTERS
    # =========================================================================

    def where(self, predicate: Callable[["EntityProxy"], bool]) -> "QueryResult":
        """Filter entities by a predicate."""
        filtered = []
        for e in self._entities:
            try:
                proxy = EntityProxy(e)
                if predicate(proxy):
                    filtered.append(e)
            except (AttributeError, TypeError):
                pass
        return QueryResult(filtered)

    def without(self, *Cs: Type) -> "QueryResult":
        """Exclude entities that have any of the specified components."""
        world = _get_world()
        filtered = [
            e for e in self._entities
            if not any(world.has(e, C) for C in Cs)
        ]
        return QueryResult(filtered)

    def with_all(self, *Cs: Type) -> "QueryResult":
        """Include only entities that have all specified components."""
        world = _get_world()
        filtered = [
            e for e in self._entities
            if all(world.has(e, C) for C in Cs)
        ]
        return QueryResult(filtered)

    def near(self, target: "EntityProxy", distance: float) -> "QueryResult":
        """Filter entities within distance of target (requires Position component)."""
        world = _get_world()

        # Get Position component type
        Position = _registry.get("Position")
        if Position is None:
            return QueryResult([])

        target_entity = target._entity if isinstance(target, EntityProxy) else target
        target_pos = world.get(target_entity, Position)
        if target_pos is None:
            return QueryResult([])

        def in_range(e: "Entity") -> bool:
            pos = world.get(e, Position)
            if pos is None:
                return False
            # Calculate distance (assumes Position has x, y, z or x, y)
            dx = getattr(pos, "x", 0) - getattr(target_pos, "x", 0)
            dy = getattr(pos, "y", 0) - getattr(target_pos, "y", 0)
            dz = getattr(pos, "z", 0) - getattr(target_pos, "z", 0)
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            return dist <= distance

        filtered = [e for e in self._entities if in_range(e)]
        return QueryResult(filtered)

    # =========================================================================
    # ACCESSORS
    # =========================================================================

    def first(self) -> Optional[EntityProxy]:
        """Get the first entity, or None if empty."""
        if not self._entities:
            return None
        return EntityProxy(self._entities[0])

    def count(self) -> int:
        """Count the number of entities."""
        return len(self._entities)

    def exists(self) -> bool:
        """Check if any entities match."""
        return len(self._entities) > 0

    def ids(self) -> List[int]:
        """Get list of entity IDs."""
        return [e.id for e in self._entities]

    # =========================================================================
    # BULK MUTATIONS
    # =========================================================================

    def set(self, **fields: Any) -> None:
        """
        Set fields on all matching entities.

        Field format: component__field=value
        Example: health__current=100

        Args:
            **fields: Keyword arguments in component__field=value format.

        Raises:
            ValueError: If field_spec doesn't contain '__' separator.
        """
        world = _get_world()
        echo = _get_echo()

        count = 0
        for e in self._entities:
            for field_spec, value in fields.items():
                if "__" not in field_spec:
                    raise ValueError(
                        f"Invalid field spec '{field_spec}': use 'component__field' format"
                    )
                component_name, field_name = field_spec.split("__", 1)

                # Look up component type
                C = _registry.get(component_name.title()) or _registry.get(component_name)
                if C and world.has(e, C):
                    world.set(e, C, field_name, value)
                    count += 1

        echo(f"Set {len(fields)} field(s) on {len(self._entities)} entities ({count} mutations)")

    def destroy(self) -> None:
        """Destroy all matching entities."""
        world = _get_world()
        echo = _get_echo()

        entity_count = len(self._entities)
        ids = [e.id for e in self._entities[:MAX_DISPLAY_ENTITIES]]

        for e in self._entities:
            world.destroy(e)

        suffix = f"... and {entity_count - MAX_DISPLAY_ENTITIES} more" if entity_count > MAX_DISPLAY_ENTITIES else ""
        echo(f"Destroyed {entity_count} entities: {ids}{suffix}")

    def each(self, fn: Callable[["EntityProxy"], None]) -> None:
        """Apply a function to each entity."""
        echo = _get_echo()

        for e in self._entities:
            fn(EntityProxy(e))

        echo(f"Applied to {len(self._entities)} entities")

    # =========================================================================
    # ITERATION
    # =========================================================================

    def __iter__(self) -> Iterator[EntityProxy]:
        for e in self._entities:
            yield EntityProxy(e)

    def __len__(self) -> int:
        return len(self._entities)

    def __getitem__(self, index: int) -> EntityProxy:
        return EntityProxy(self._entities[index])

    def __repr__(self) -> str:
        if len(self._entities) <= MAX_DISPLAY_ENTITIES:
            return repr([EntityProxy(e) for e in self._entities])
        shown = [EntityProxy(e) for e in self._entities[:MAX_DISPLAY_ENTITIES]]
        return f"[{', '.join(repr(e) for e in shown)}, ... +{len(self._entities) - MAX_DISPLAY_ENTITIES} more]"


# =============================================================================
# TYPE QUERY
# =============================================================================


class TypeQuery:
    """
    Entry point for type-based queries.

    Enables syntax like:
        Enemy.all
        Enemy.where(lambda e: e.health.current < 50)
    """

    def __init__(self, component_type: Type) -> None:
        self._type = component_type

    @property
    def all(self) -> QueryResult:
        """Get all entities with this component."""
        world = _get_world()
        entities = world.query(self._type)
        return QueryResult(entities)

    def where(self, predicate: Callable[["EntityProxy"], bool]) -> QueryResult:
        """Query with filter."""
        return self.all.where(predicate)

    def without(self, *Cs: Type) -> QueryResult:
        """Query excluding components."""
        return self.all.without(*Cs)

    def near(self, target: "EntityProxy", distance: float) -> QueryResult:
        """Query within distance of target."""
        return self.all.near(target, distance)

    def count(self) -> int:
        """Count entities with this component."""
        return self.all.count()

    def __repr__(self) -> str:
        count = self.count()
        return f"<{self._type.__name__}: {count} entities>"


# =============================================================================
# TIME MANAGER
# =============================================================================


class TimeManager:
    """
    Named snapshots and undo/redo functionality.

    Enables syntax like:
        mark("before_fight")
        rewind("before_fight")
        undo()
        redo()
    """

    def __init__(self) -> None:
        self._marks: Dict[str, Any] = {}  # name -> Snapshot
        self._undo_stack: List[Any] = []
        self._redo_stack: List[Any] = []

    def mark(self, name: str) -> None:
        """Create a named snapshot."""
        world = _get_world()
        echo = _get_echo()

        self._marks[name] = world.snap(name)
        echo(f'Marked "{name}"')

    def rewind(self, name: str) -> None:
        """Restore to a named snapshot."""
        world = _get_world()
        echo = _get_echo()

        if name not in self._marks:
            echo(f'No mark named "{name}"')
            return

        # Save current state for potential redo
        current = world.snap()
        target = self._marks[name]
        changes = world.diff(target, current)

        self._undo_stack.append(current)
        if len(self._undo_stack) > MAX_UNDO_STACK:
            self._undo_stack.pop(0)

        world.restore(target)
        echo(f'Rewound to "{name}" ({len(changes)} changes reverted)')

    def checkpoint(self) -> None:
        """Create an automatic checkpoint before mutation."""
        world = _get_world()

        self._undo_stack.append(world.snap())
        if len(self._undo_stack) > MAX_UNDO_STACK:
            self._undo_stack.pop(0)

        self._redo_stack.clear()

    def undo(self) -> None:
        """Undo the last change."""
        world = _get_world()
        echo = _get_echo()

        if not self._undo_stack:
            echo("Nothing to undo")
            return

        current = world.snap()
        previous = self._undo_stack.pop()
        changes = world.diff(previous, current)

        self._redo_stack.append(current)
        if len(self._redo_stack) > MAX_REDO_STACK:
            self._redo_stack.pop(0)

        world.restore(previous)
        echo(f"Undid {len(changes)} changes")

    def redo(self) -> None:
        """Redo the last undone change."""
        world = _get_world()
        echo = _get_echo()

        if not self._redo_stack:
            echo("Nothing to redo")
            return

        current = world.snap()
        next_state = self._redo_stack.pop()
        changes = world.diff(current, next_state)

        self._undo_stack.append(current)
        if len(self._undo_stack) > MAX_UNDO_STACK:
            self._undo_stack.pop(0)

        world.restore(next_state)
        echo(f"Redid {len(changes)} changes")

    def history(self, n: int = DEFAULT_HISTORY_COUNT) -> None:
        """Show recent changes."""
        world = _get_world()
        echo = _get_echo()

        changes = world.recent_changes(n)
        for change in changes:
            echo(f"  {change}")

    def marks(self) -> List[str]:
        """List all mark names."""
        return list(self._marks.keys())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ComponentProxy",
    "EntityProxy",
    "QueryResult",
    "TypeQuery",
    "TimeManager",
    # Setup functions
    "set_world",
    "set_echo",
    "set_registry",
    # Constants
    "MAX_DISPLAY_ENTITIES",
    "MAX_UNDO_STACK",
    "MAX_REDO_STACK",
    "DEFAULT_HISTORY_COUNT",
]
