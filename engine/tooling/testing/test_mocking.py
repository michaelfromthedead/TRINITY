"""
Mocking framework for game objects.

Provides comprehensive mocking support for testing game engine
components including entities, components, systems, and resources.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
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
from unittest.mock import MagicMock, patch as _patch

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CallRecord:
    """Record of a mock call."""

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    return_value: Any = None
    exception: Optional[Exception] = None
    timestamp: float = 0.0

    def matches(
        self,
        *args,
        **kwargs,
    ) -> bool:
        """Check if this call matches the given arguments."""
        if args and self.args != args:
            return False
        for key, value in kwargs.items():
            if self.kwargs.get(key) != value:
                return False
        return True


class Mock:
    """
    Flexible mock object for testing.

    Features:
    - Call tracking
    - Return value configuration
    - Side effects
    - Attribute mocking
    - Call assertions

    Example:
        mock = Mock()
        mock.configure_return("get_health", 100)
        mock.get_health()  # Returns 100
        assert mock.was_called("get_health")
    """

    def __init__(
        self,
        name: str = "Mock",
        spec: Optional[Type] = None,
        **defaults: Any,
    ):
        self._name = name
        self._spec = spec
        self._calls: Dict[str, List[CallRecord]] = {}
        self._return_values: Dict[str, Any] = {}
        self._side_effects: Dict[str, Callable] = {}
        self._attributes: Dict[str, Any] = dict(defaults)
        self._child_mocks: Dict[str, "Mock"] = {}

    def __getattr__(self, name: str) -> Any:
        # Check configured attributes
        if name.startswith("_"):
            raise AttributeError(name)

        if name in self._attributes:
            return self._attributes[name]

        # Return child mock or create new one
        if name not in self._child_mocks:
            self._child_mocks[name] = MockMethod(self, name)

        return self._child_mocks[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._attributes[name] = value

    def __call__(self, *args, **kwargs) -> Any:
        return self._record_call("__call__", args, kwargs)

    def _record_call(
        self,
        method: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Record a call and return the configured value."""
        import time

        record = CallRecord(
            args=args,
            kwargs=kwargs,
            timestamp=time.time(),
        )

        if method not in self._calls:
            self._calls[method] = []
        self._calls[method].append(record)

        # Check for side effect
        if method in self._side_effects:
            try:
                result = self._side_effects[method](*args, **kwargs)
                record.return_value = result
                return result
            except Exception as e:
                record.exception = e
                raise

        # Return configured value
        if method in self._return_values:
            value = self._return_values[method]
            record.return_value = value
            return value

        return None

    def configure_return(self, method: str, value: Any) -> "Mock":
        """Configure return value for a method."""
        self._return_values[method] = value
        return self

    def configure_side_effect(
        self,
        method: str,
        effect: Union[Callable, Exception, List[Any]],
    ) -> "Mock":
        """Configure side effect for a method."""
        if isinstance(effect, Exception):
            def raise_error(*args, **kwargs):
                raise effect
            self._side_effects[method] = raise_error
        elif isinstance(effect, list):
            iterator = iter(effect)
            def return_next(*args, **kwargs):
                return next(iterator)
            self._side_effects[method] = return_next
        else:
            self._side_effects[method] = effect
        return self

    def was_called(self, method: str = "__call__") -> bool:
        """Check if a method was called."""
        return len(self._calls.get(method, [])) > 0

    def call_count(self, method: str = "__call__") -> int:
        """Get number of times a method was called."""
        return len(self._calls.get(method, []))

    def get_calls(self, method: str = "__call__") -> List[CallRecord]:
        """Get all calls to a method."""
        return self._calls.get(method, [])

    def last_call(self, method: str = "__call__") -> Optional[CallRecord]:
        """Get the last call to a method."""
        calls = self._calls.get(method, [])
        return calls[-1] if calls else None

    def assert_called(self, method: str = "__call__") -> None:
        """Assert method was called at least once."""
        if not self.was_called(method):
            raise AssertionError(f"{self._name}.{method} was not called")

    def assert_called_with(
        self,
        method: str,
        *args,
        **kwargs,
    ) -> None:
        """Assert method was called with specific arguments."""
        calls = self._calls.get(method, [])
        for call in calls:
            if call.matches(*args, **kwargs):
                return
        raise AssertionError(
            f"{self._name}.{method} was not called with args={args}, kwargs={kwargs}"
        )

    def assert_call_count(self, method: str, expected: int) -> None:
        """Assert method was called a specific number of times."""
        actual = self.call_count(method)
        if actual != expected:
            raise AssertionError(
                f"{self._name}.{method} call count: expected {expected}, got {actual}"
            )

    def reset(self) -> None:
        """Reset all call records."""
        self._calls.clear()
        for child in self._child_mocks.values():
            child.reset()


class MockMethod:
    """A callable mock method."""

    def __init__(self, parent: Mock, name: str):
        self._parent = parent
        self._name = name

    def __call__(self, *args, **kwargs) -> Any:
        return self._parent._record_call(self._name, args, kwargs)

    def reset(self) -> None:
        """Reset call records for this method."""
        self._parent._calls.pop(self._name, None)


class MockEntity:
    """
    Mock entity for testing entity-related code.

    Example:
        entity = MockEntity(id=1)
        entity.add_component("Position", {"x": 0, "y": 0})
        assert entity.has_component("Position")
    """

    def __init__(
        self,
        id: int = 0,
        name: str = "MockEntity",
        **components: Any,
    ):
        self.id = id
        self.name = name
        self._components: Dict[str, Any] = dict(components)
        self._tags: Set[str] = set()
        self._active = True

    def add_component(self, component_type: str, data: Any = None) -> None:
        """Add a component to this entity."""
        self._components[component_type] = data or {}

    def remove_component(self, component_type: str) -> None:
        """Remove a component from this entity."""
        self._components.pop(component_type, None)

    def get_component(self, component_type: str) -> Any:
        """Get a component by type."""
        return self._components.get(component_type)

    def has_component(self, component_type: str) -> bool:
        """Check if entity has a component."""
        return component_type in self._components

    def add_tag(self, tag: str) -> None:
        """Add a tag to this entity."""
        self._tags.add(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this entity."""
        self._tags.discard(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if entity has a tag."""
        return tag in self._tags

    @property
    def active(self) -> bool:
        """Check if entity is active."""
        return self._active

    def activate(self) -> None:
        """Activate this entity."""
        self._active = True

    def deactivate(self) -> None:
        """Deactivate this entity."""
        self._active = False


class MockComponent:
    """
    Mock component for testing component-related code.

    Example:
        comp = MockComponent("Health", current=100, max=100)
        assert comp.current == 100
    """

    def __init__(self, type_name: str, **data: Any):
        self._type_name = type_name
        self._data = dict(data)
        self._dirty = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value
            self._dirty = True

    @property
    def type_name(self) -> str:
        """Get component type name."""
        return self._type_name

    @property
    def dirty(self) -> bool:
        """Check if component was modified."""
        return self._dirty

    def mark_clean(self) -> None:
        """Mark component as clean."""
        self._dirty = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return dict(self._data)


class MockSystem:
    """
    Mock system for testing system-related code.

    Example:
        system = MockSystem("MovementSystem")
        system.update(world, 0.016)
        assert system.update_count == 1
    """

    def __init__(
        self,
        name: str,
        phase: str = "update",
        priority: int = 0,
    ):
        self.name = name
        self.phase = phase
        self.priority = priority
        self.update_count = 0
        self.entities_processed: List[int] = []
        self._enabled = True
        self._on_update: Optional[Callable] = None

    def update(self, world: Any, delta_time: float) -> None:
        """Execute system update."""
        if not self._enabled:
            return

        self.update_count += 1

        if self._on_update:
            self._on_update(world, delta_time)

    def configure_update(self, callback: Callable) -> "MockSystem":
        """Configure update behavior."""
        self._on_update = callback
        return self

    def enable(self) -> None:
        """Enable this system."""
        self._enabled = True

    def disable(self) -> None:
        """Disable this system."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if system is enabled."""
        return self._enabled

    def reset(self) -> None:
        """Reset system state."""
        self.update_count = 0
        self.entities_processed.clear()


class MockWorld:
    """
    Mock game world for testing world-related code.

    Example:
        world = MockWorld()
        entity = world.create_entity()
        world.add_component(entity, "Position", {"x": 0, "y": 0})
    """

    def __init__(self):
        self._entities: Dict[int, MockEntity] = {}
        self._systems: List[MockSystem] = []
        self._resources: Dict[str, Any] = {}
        self._events: List[Any] = []
        self._next_entity_id = 1

    def create_entity(self, name: str = "") -> int:
        """Create a new entity."""
        entity_id = self._next_entity_id
        self._next_entity_id += 1

        self._entities[entity_id] = MockEntity(
            id=entity_id,
            name=name or f"Entity_{entity_id}",
        )

        return entity_id

    def destroy_entity(self, entity_id: int) -> None:
        """Destroy an entity."""
        self._entities.pop(entity_id, None)

    def get_entity(self, entity_id: int) -> Optional[MockEntity]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    def add_component(
        self,
        entity_id: int,
        component_type: str,
        data: Any = None,
    ) -> None:
        """Add a component to an entity."""
        entity = self._entities.get(entity_id)
        if entity:
            entity.add_component(component_type, data)

    def remove_component(self, entity_id: int, component_type: str) -> None:
        """Remove a component from an entity."""
        entity = self._entities.get(entity_id)
        if entity:
            entity.remove_component(component_type)

    def get_component(self, entity_id: int, component_type: str) -> Any:
        """Get a component from an entity."""
        entity = self._entities.get(entity_id)
        return entity.get_component(component_type) if entity else None

    def has_component(self, entity_id: int, component_type: str) -> bool:
        """Check if an entity has a component."""
        entity = self._entities.get(entity_id)
        return entity.has_component(component_type) if entity else False

    def add_system(self, system: MockSystem) -> None:
        """Add a system to the world."""
        self._systems.append(system)
        self._systems.sort(key=lambda s: s.priority, reverse=True)

    def set_resource(self, name: str, value: Any) -> None:
        """Set a global resource."""
        self._resources[name] = value

    def get_resource(self, name: str) -> Any:
        """Get a global resource."""
        return self._resources.get(name)

    def fire_event(self, event: Any) -> None:
        """Fire an event."""
        self._events.append(event)

    def get_events(self) -> List[Any]:
        """Get all fired events."""
        return list(self._events)

    def clear_events(self) -> None:
        """Clear event queue."""
        self._events.clear()

    def update(self, delta_time: float = 0.016) -> None:
        """Run one update tick."""
        for system in self._systems:
            system.update(self, delta_time)

    def query(self, *component_types: str) -> List[int]:
        """Query entities with specified components."""
        result = []
        for entity_id, entity in self._entities.items():
            if all(entity.has_component(ct) for ct in component_types):
                result.append(entity_id)
        return result

    @property
    def entity_count(self) -> int:
        """Get number of entities."""
        return len(self._entities)

    @property
    def entities(self) -> Dict[int, MockEntity]:
        """Get all entities."""
        return self._entities

    @property
    def components(self) -> Dict[int, Dict[str, Any]]:
        """Get components by entity."""
        return {
            eid: entity._components
            for eid, entity in self._entities.items()
        }

    def reset(self) -> None:
        """Reset the world."""
        self._entities.clear()
        self._systems.clear()
        self._resources.clear()
        self._events.clear()
        self._next_entity_id = 1


class MockResource:
    """
    Mock resource for testing resource-related code.

    Example:
        time_resource = MockResource("Time", delta=0.016, elapsed=0.0)
        assert time_resource.delta == 0.016
    """

    def __init__(self, name: str, **data: Any):
        self._name = name
        self._data = dict(data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    @property
    def name(self) -> str:
        """Get resource name."""
        return self._name


def patch(target: str, **kwargs) -> Any:
    """
    Patch a target for testing.

    Wrapper around unittest.mock.patch with game-specific defaults.

    Example:
        with patch("game.physics.gravity", 0.0):
            # Test zero gravity
            ...
    """
    return _patch(target, **kwargs)


def spy(obj: Any, method: str) -> Mock:
    """
    Create a spy on an object method.

    The spy calls the original method but tracks calls.

    Example:
        player = Player()
        spy_mock = spy(player, "take_damage")
        player.take_damage(10)
        assert spy_mock.was_called("take_damage")
    """
    original = getattr(obj, method)
    mock = Mock(name=f"spy({obj.__class__.__name__}.{method})")

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        mock._record_call(method, args, kwargs)
        return original(*args, **kwargs)

    setattr(obj, method, wrapper)
    return mock


def stub(return_value: Any = None) -> Callable[[F], F]:
    """
    Decorator to stub a function.

    Example:
        @stub(return_value=100)
        def get_player_health(player):
            # Original implementation replaced
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return return_value
        return wrapper  # type: ignore
    return decorator


class MockContext:
    """
    Context manager for managing multiple mocks.

    Example:
        with MockContext() as ctx:
            player = ctx.mock("player")
            enemy = ctx.mock("enemy")
            ctx.patch("game.gravity", 0.0)
            # Test code
    """

    def __init__(self):
        self._mocks: Dict[str, Mock] = {}
        self._patches: List[Any] = []
        self._original_values: Dict[str, Any] = {}

    def __enter__(self) -> "MockContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def mock(self, name: str, spec: Optional[Type] = None, **defaults) -> Mock:
        """Create and register a mock."""
        m = Mock(name=name, spec=spec, **defaults)
        self._mocks[name] = m
        return m

    def patch(self, target: str, value: Any = None) -> Any:
        """Patch a target."""
        p = _patch(target, value)
        mock = p.start()
        self._patches.append(p)
        return mock

    def get_mock(self, name: str) -> Optional[Mock]:
        """Get a registered mock by name."""
        return self._mocks.get(name)

    def reset_all(self) -> None:
        """Reset all mocks."""
        for mock in self._mocks.values():
            mock.reset()

    def cleanup(self) -> None:
        """Clean up all patches."""
        for p in self._patches:
            p.stop()
        self._patches.clear()


def create_mock(
    spec: Optional[Type[T]] = None,
    **defaults: Any,
) -> Union[Mock, T]:
    """
    Create a mock, optionally based on a spec.

    Example:
        mock_player = create_mock(Player, health=100, name="TestPlayer")
    """
    if spec is None:
        return Mock(**defaults)

    mock = Mock(name=spec.__name__, spec=spec, **defaults)

    # Add methods from spec
    for name in dir(spec):
        if not name.startswith("_") and callable(getattr(spec, name, None)):
            mock.configure_return(name, None)

    return mock
