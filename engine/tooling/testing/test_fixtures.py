"""
Test fixtures and setup/teardown management.

Provides fixture management for test setup and teardown,
supporting various scopes and dependency injection.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class FixtureScope(Enum):
    """Scope of a fixture's lifecycle."""

    FUNCTION = auto()  # Created/destroyed per test function
    CLASS = auto()      # Created/destroyed per test class
    MODULE = auto()     # Created/destroyed per test module
    SESSION = auto()    # Created once per test session


@dataclass
class Fixture(Generic[T]):
    """
    A test fixture that provides resources to tests.

    Fixtures can have setup and teardown logic, and can depend
    on other fixtures through dependency injection.
    """

    name: str
    factory: Callable[..., T]
    scope: FixtureScope = FixtureScope.FUNCTION
    dependencies: List[str] = field(default_factory=list)
    autouse: bool = False
    params: Optional[List[Any]] = None
    _cached_value: Optional[T] = None
    _is_cached: bool = False

    def create(self, **kwargs) -> T:
        """Create the fixture value."""
        if self._is_cached and self.scope != FixtureScope.FUNCTION:
            return self._cached_value  # type: ignore

        value = self.factory(**kwargs)

        if self.scope != FixtureScope.FUNCTION:
            self._cached_value = value
            self._is_cached = True

        return value

    def destroy(self) -> None:
        """Destroy the cached fixture value."""
        if self._is_cached and hasattr(self._cached_value, "close"):
            self._cached_value.close()  # type: ignore
        self._cached_value = None
        self._is_cached = False

    def reset(self) -> None:
        """Reset the fixture for reuse."""
        self._cached_value = None
        self._is_cached = False


def fixture(
    func: Optional[F] = None,
    *,
    scope: Union[FixtureScope, str] = FixtureScope.FUNCTION,
    autouse: bool = False,
    params: Optional[List[Any]] = None,
    name: Optional[str] = None,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to define a test fixture.

    Args:
        func: The fixture factory function
        scope: Fixture scope (function, class, module, session)
        autouse: Whether to automatically use this fixture
        params: Parameters for parameterized fixtures
        name: Custom fixture name

    Example:
        @fixture
        def database():
            db = Database()
            db.connect()
            yield db
            db.disconnect()

        @fixture(scope="module")
        def config():
            return load_config()
    """
    if isinstance(scope, str):
        scope = FixtureScope[scope.upper()]

    def decorator(f: F) -> F:
        fixture_name = name or f.__name__

        # Check if it's a generator (has yield)
        is_generator = inspect.isgeneratorfunction(f)

        @functools.wraps(f)
        def wrapper(**kwargs) -> Any:
            if is_generator:
                gen = f(**kwargs)
                return next(gen)
            return f(**kwargs)

        # Get dependencies from function signature
        sig = inspect.signature(f)
        deps = [
            param.name for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty
        ]

        fixture_obj = Fixture(
            name=fixture_name,
            factory=wrapper,
            scope=scope,
            dependencies=deps,
            autouse=autouse,
            params=params,
        )

        wrapper._fixture = fixture_obj
        wrapper._is_fixture = True
        wrapper._fixture_generator = f if is_generator else None

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


def setup(func: F) -> F:
    """
    Decorator to mark a function as test setup.

    Example:
        class TestExample:
            @setup
            def prepare(self):
                self.data = [1, 2, 3]
    """
    func._is_setup = True
    func._setup_scope = FixtureScope.FUNCTION
    return func


def teardown(func: F) -> F:
    """
    Decorator to mark a function as test teardown.

    Example:
        class TestExample:
            @teardown
            def cleanup(self):
                self.data.clear()
    """
    func._is_teardown = True
    func._teardown_scope = FixtureScope.FUNCTION
    return func


def before_all(func: F) -> F:
    """
    Decorator to mark a function to run before all tests.

    Example:
        @before_all
        def setup_database():
            global db
            db = Database()
    """
    func._is_setup = True
    func._setup_scope = FixtureScope.MODULE
    return func


def after_all(func: F) -> F:
    """
    Decorator to mark a function to run after all tests.

    Example:
        @after_all
        def teardown_database():
            db.close()
    """
    func._is_teardown = True
    func._teardown_scope = FixtureScope.MODULE
    return func


class FixtureManager:
    """
    Manages fixture lifecycle and dependency injection.

    Features:
    - Automatic dependency resolution
    - Scope-based caching
    - Generator fixture support (for teardown)
    - Parameterized fixtures
    """

    def __init__(self):
        self._fixtures: Dict[str, Fixture] = {}
        self._active_fixtures: Dict[str, Any] = {}
        self._generators: Dict[str, Generator] = {}
        self._scope_order = [
            FixtureScope.SESSION,
            FixtureScope.MODULE,
            FixtureScope.CLASS,
            FixtureScope.FUNCTION,
        ]

    def register(self, fixture: Fixture) -> None:
        """Register a fixture."""
        self._fixtures[fixture.name] = fixture

    def register_function(self, func: Callable) -> None:
        """Register a fixture from a decorated function."""
        if hasattr(func, "_fixture"):
            self.register(func._fixture)

    def get_fixture(self, name: str) -> Any:
        """Get or create a fixture value."""
        if name in self._active_fixtures:
            return self._active_fixtures[name]

        fixture = self._fixtures.get(name)
        if fixture is None:
            raise KeyError(f"Unknown fixture: {name}")

        # Resolve dependencies
        deps = {}
        for dep_name in fixture.dependencies:
            deps[dep_name] = self.get_fixture(dep_name)

        # Check if it's a generator fixture
        gen_func = getattr(fixture.factory, "_fixture_generator", None)
        if gen_func:
            gen = gen_func(**deps)
            value = next(gen)
            self._generators[name] = gen
        else:
            value = fixture.create(**deps)

        self._active_fixtures[name] = value
        return value

    def teardown_fixture(self, name: str) -> None:
        """Teardown a specific fixture."""
        if name in self._generators:
            gen = self._generators.pop(name)
            try:
                next(gen)
            except StopIteration:
                pass

        if name in self._active_fixtures:
            value = self._active_fixtures.pop(name)
            if hasattr(value, "close"):
                value.close()

        if name in self._fixtures:
            self._fixtures[name].reset()

    def teardown_scope(self, scope: FixtureScope) -> None:
        """Teardown all fixtures of a given scope."""
        to_teardown = [
            name for name, fixture in self._fixtures.items()
            if fixture.scope == scope and name in self._active_fixtures
        ]

        for name in to_teardown:
            self.teardown_fixture(name)

    def teardown_all(self) -> None:
        """Teardown all active fixtures."""
        for scope in reversed(self._scope_order):
            self.teardown_scope(scope)

    def get_autouse_fixtures(self, scope: FixtureScope) -> List[str]:
        """Get all autouse fixtures for a scope."""
        return [
            name for name, fixture in self._fixtures.items()
            if fixture.autouse and fixture.scope == scope
        ]

    @contextmanager
    def fixture_context(self, *fixture_names: str):
        """Context manager for fixture lifecycle."""
        values = {}
        try:
            for name in fixture_names:
                values[name] = self.get_fixture(name)
            yield values
        finally:
            for name in reversed(fixture_names):
                self.teardown_fixture(name)

    def inject_fixtures(self, func: Callable) -> Callable:
        """
        Decorator to inject fixtures into a function.

        Example:
            @manager.inject_fixtures
            def test_something(database, config):
                # database and config are injected
                ...
        """
        sig = inspect.signature(func)
        fixture_params = [
            param.name for param in sig.parameters.values()
            if param.name in self._fixtures
        ]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.fixture_context(*fixture_params) as fixtures:
                kwargs.update(fixtures)
                return func(*args, **kwargs)

        return wrapper


# Game-specific fixtures

class GameWorldFixture:
    """
    Fixture for creating isolated game worlds for testing.

    Example:
        @fixture
        def game_world():
            return GameWorldFixture()
    """

    def __init__(self):
        self.entities: Dict[int, Any] = {}
        self.components: Dict[int, Dict[str, Any]] = {}
        self.systems: List[Any] = []
        self.resources: Dict[str, Any] = {}
        self._next_entity_id = 1
        self._events: List[Any] = []

    def create_entity(self) -> int:
        """Create a new entity and return its ID."""
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        self.entities[entity_id] = {"id": entity_id, "active": True}
        self.components[entity_id] = {}
        return entity_id

    def add_component(self, entity_id: int, component_type: str, data: Any) -> None:
        """Add a component to an entity."""
        if entity_id not in self.components:
            raise ValueError(f"Entity {entity_id} does not exist")
        self.components[entity_id][component_type] = data

    def get_component(self, entity_id: int, component_type: str) -> Any:
        """Get a component from an entity."""
        return self.components.get(entity_id, {}).get(component_type)

    def has_component(self, entity_id: int, component_type: str) -> bool:
        """Check if an entity has a component."""
        return component_type in self.components.get(entity_id, {})

    def remove_entity(self, entity_id: int) -> None:
        """Remove an entity."""
        self.entities.pop(entity_id, None)
        self.components.pop(entity_id, None)

    def add_system(self, system: Any) -> None:
        """Add a system to the world."""
        self.systems.append(system)

    def set_resource(self, name: str, value: Any) -> None:
        """Set a global resource."""
        self.resources[name] = value

    def get_resource(self, name: str) -> Any:
        """Get a global resource."""
        return self.resources.get(name)

    def fire_event(self, event: Any) -> None:
        """Fire an event."""
        self._events.append(event)

    def get_events(self) -> List[Any]:
        """Get all fired events."""
        return self._events.copy()

    def clear_events(self) -> None:
        """Clear event history."""
        self._events.clear()

    def tick(self, delta_time: float = 0.016) -> None:
        """Simulate a game tick."""
        for system in self.systems:
            if hasattr(system, "update"):
                system.update(self, delta_time)

    def reset(self) -> None:
        """Reset the world to initial state."""
        self.entities.clear()
        self.components.clear()
        self.systems.clear()
        self.resources.clear()
        self._events.clear()
        self._next_entity_id = 1

    def close(self) -> None:
        """Clean up the world."""
        self.reset()


class EntityFixture:
    """
    Fixture for creating test entities with components.

    Example:
        @fixture
        def player_entity(game_world):
            entity = EntityFixture(game_world)
            entity.add_component("Position", {"x": 0, "y": 0})
            entity.add_component("Health", {"current": 100, "max": 100})
            return entity
    """

    def __init__(self, world: GameWorldFixture):
        self.world = world
        self.entity_id = world.create_entity()

    def add_component(self, component_type: str, data: Any) -> "EntityFixture":
        """Add a component and return self for chaining."""
        self.world.add_component(self.entity_id, component_type, data)
        return self

    def get_component(self, component_type: str) -> Any:
        """Get a component."""
        return self.world.get_component(self.entity_id, component_type)

    def has_component(self, component_type: str) -> bool:
        """Check if entity has a component."""
        return self.world.has_component(self.entity_id, component_type)

    def destroy(self) -> None:
        """Remove this entity from the world."""
        self.world.remove_entity(self.entity_id)


class ResourceFixture:
    """
    Fixture for managing test resources (files, connections, etc.).

    Example:
        @fixture
        def temp_file():
            resource = ResourceFixture()
            path = resource.create_temp_file("test.txt", "hello")
            yield path
            resource.cleanup()
    """

    def __init__(self):
        self._temp_files: List[str] = []
        self._temp_dirs: List[str] = []
        self._open_handles: List[Any] = []

    def create_temp_file(
        self,
        name: str,
        content: Union[str, bytes] = "",
        dir: Optional[str] = None,
    ) -> str:
        """Create a temporary file."""
        import tempfile
        import os

        if dir:
            path = os.path.join(dir, name)
        else:
            fd, path = tempfile.mkstemp(suffix=f"_{name}")
            os.close(fd)

        mode = "wb" if isinstance(content, bytes) else "w"
        with open(path, mode) as f:
            f.write(content)

        self._temp_files.append(path)
        return path

    def create_temp_dir(self, name: Optional[str] = None) -> str:
        """Create a temporary directory."""
        import tempfile

        path = tempfile.mkdtemp(suffix=f"_{name}" if name else None)
        self._temp_dirs.append(path)
        return path

    def register_handle(self, handle: Any) -> Any:
        """Register a handle to be closed on cleanup."""
        self._open_handles.append(handle)
        return handle

    def cleanup(self) -> None:
        """Clean up all resources."""
        import os
        import shutil

        for handle in self._open_handles:
            try:
                handle.close()
            except Exception:
                pass

        for path in self._temp_files:
            try:
                os.unlink(path)
            except Exception:
                pass

        for path in self._temp_dirs:
            try:
                shutil.rmtree(path)
            except Exception:
                pass

        self._temp_files.clear()
        self._temp_dirs.clear()
        self._open_handles.clear()

    def close(self) -> None:
        """Alias for cleanup."""
        self.cleanup()
