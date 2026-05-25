"""
Test fixtures for the engine testing framework.

Provides base classes for test fixtures that manage setup and teardown
of test resources. Supports per-test and per-class lifecycle methods.

Usage:
    from engine.debug.testing.fixtures import TestFixture, SharedFixture

    class DatabaseFixture(TestFixture):
        def setUp(self):
            self.db = create_test_database()

        def tearDown(self):
            self.db.close()

    class MyTests(TestSuite):
        fixture = DatabaseFixture()
"""

from __future__ import annotations

import contextlib
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
)


__all__ = [
    "TestFixture",
    "SharedFixture",
    "CompositeFixture",
    "FixtureContext",
    "fixture",
    "shared_fixture",
]


T = TypeVar("T")


@dataclass
class FixtureContext:
    """
    Context information passed to fixture lifecycle methods.

    Provides metadata about the current test execution context.

    Attributes:
        test_name: Name of the current test method (None during class setup)
        suite_name: Name of the test suite class
        fixture_name: Name of the fixture
        is_class_level: True if running setUpClass/tearDownClass
    """

    test_name: Optional[str] = None
    suite_name: Optional[str] = None
    fixture_name: Optional[str] = None
    is_class_level: bool = False

    def __repr__(self) -> str:
        if self.is_class_level:
            return f"FixtureContext(suite={self.suite_name!r}, class_level=True)"
        return f"FixtureContext(suite={self.suite_name!r}, test={self.test_name!r})"


class TestFixture(ABC):
    """
    Base class for test fixtures that manage test resources.

    A fixture provides setup and teardown hooks that run before and after
    each test (setUp/tearDown) or once for all tests in a suite
    (setUpClass/tearDownClass).

    Subclass this to create reusable test infrastructure:
    - Database connections
    - Temporary files/directories
    - Mock objects
    - Game world instances
    - Network connections

    Example:
        class GameWorldFixture(TestFixture):
            def setUpClass(self, context):
                # Called once before all tests
                self.asset_cache = AssetCache()

            def tearDownClass(self, context):
                # Called once after all tests
                self.asset_cache.clear()

            def setUp(self, context):
                # Called before each test
                self.world = World()
                self.world.initialize()

            def tearDown(self, context):
                # Called after each test
                self.world.shutdown()
    """

    # Prevent pytest from collecting this class as a test
    __test__ = False

    # Registry of active fixtures for cleanup
    _active_fixtures: ClassVar[Set["TestFixture"]] = set()

    def __init__(self, name: Optional[str] = None) -> None:
        """
        Initialize the fixture.

        Args:
            name: Optional fixture name for logging/debugging
        """
        self._name = name or self.__class__.__name__
        self._is_class_setup_done = False
        self._setup_count = 0
        self._teardown_count = 0
        self._error_count = 0
        self._last_error: Optional[Exception] = None

    @property
    def name(self) -> str:
        """Get the fixture name."""
        return self._name

    @property
    def is_active(self) -> bool:
        """Check if the fixture has been set up but not torn down."""
        return self._setup_count > self._teardown_count

    @property
    def stats(self) -> Dict[str, Any]:
        """Get fixture usage statistics."""
        return {
            "name": self._name,
            "setup_count": self._setup_count,
            "teardown_count": self._teardown_count,
            "error_count": self._error_count,
            "is_active": self.is_active,
        }

    def setUpClass(self, context: FixtureContext) -> None:
        """
        Called once before any tests in the suite run.

        Override this to perform expensive one-time setup.

        Args:
            context: The fixture context with suite information
        """
        pass

    def tearDownClass(self, context: FixtureContext) -> None:
        """
        Called once after all tests in the suite have run.

        Override this to clean up resources from setUpClass.
        Always called even if setUpClass raised an exception.

        Args:
            context: The fixture context with suite information
        """
        pass

    def setUp(self, context: FixtureContext) -> None:
        """
        Called before each test method.

        Override this to set up per-test resources.

        Args:
            context: The fixture context with test information
        """
        pass

    def tearDown(self, context: FixtureContext) -> None:
        """
        Called after each test method.

        Override this to clean up per-test resources.
        Always called even if setUp or the test raised an exception.

        Args:
            context: The fixture context with test information
        """
        pass

    def _do_class_setup(self, context: FixtureContext) -> None:
        """Internal: Execute class-level setup with tracking."""
        if self._is_class_setup_done:
            return

        TestFixture._active_fixtures.add(self)
        context = FixtureContext(
            suite_name=context.suite_name,
            fixture_name=self._name,
            is_class_level=True,
        )

        try:
            self.setUpClass(context)
            self._is_class_setup_done = True
        except Exception as e:
            self._error_count += 1
            self._last_error = e
            raise

    def _do_class_teardown(self, context: FixtureContext) -> None:
        """Internal: Execute class-level teardown with tracking."""
        if not self._is_class_setup_done:
            return

        context = FixtureContext(
            suite_name=context.suite_name,
            fixture_name=self._name,
            is_class_level=True,
        )

        try:
            self.tearDownClass(context)
        except Exception as e:
            self._error_count += 1
            self._last_error = e
            raise
        finally:
            self._is_class_setup_done = False
            TestFixture._active_fixtures.discard(self)

    def _do_setup(self, context: FixtureContext) -> None:
        """Internal: Execute per-test setup with tracking."""
        context = FixtureContext(
            test_name=context.test_name,
            suite_name=context.suite_name,
            fixture_name=self._name,
            is_class_level=False,
        )

        try:
            self.setUp(context)
            self._setup_count += 1
        except Exception as e:
            self._error_count += 1
            self._last_error = e
            raise

    def _do_teardown(self, context: FixtureContext) -> None:
        """Internal: Execute per-test teardown with tracking."""
        context = FixtureContext(
            test_name=context.test_name,
            suite_name=context.suite_name,
            fixture_name=self._name,
            is_class_level=False,
        )

        try:
            self.tearDown(context)
            self._teardown_count += 1
        except Exception as e:
            self._error_count += 1
            self._last_error = e
            raise

    @contextlib.contextmanager
    def apply(self, context: FixtureContext) -> Iterator[None]:
        """
        Context manager to apply the fixture for a single test.

        Args:
            context: The fixture context

        Yields:
            None

        Example:
            with fixture.apply(context):
                run_test()
        """
        self._do_setup(context)
        try:
            yield
        finally:
            self._do_teardown(context)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, active={self.is_active})"


class SharedFixture(TestFixture):
    """
    A fixture that is shared across multiple test suites.

    Unlike TestFixture, SharedFixture maintains a single instance
    that is reused across test suites. Class-level setup is only
    performed once for the first suite that uses the fixture.

    Use this for expensive resources that should be shared:
    - Asset caches
    - Database connection pools
    - Precomputed test data

    Example:
        # Define at module level
        asset_fixture = SharedFixture.create(
            "assets",
            setup_class=lambda ctx: AssetCache(),
            teardown_class=lambda ctx, cache: cache.clear(),
        )

        class TestA(TestSuite):
            fixtures = [asset_fixture]
            # Uses shared asset cache

        class TestB(TestSuite):
            fixtures = [asset_fixture]
            # Same asset cache instance
    """

    # Global registry of shared fixtures
    _registry: ClassVar[Dict[str, "SharedFixture"]] = {}

    def __init__(self, name: str) -> None:
        """
        Initialize a shared fixture.

        Args:
            name: Unique name for the shared fixture
        """
        super().__init__(name)
        self._ref_count = 0
        self._shared_data: Dict[str, Any] = {}

    @classmethod
    def get(cls, name: str) -> Optional["SharedFixture"]:
        """
        Get a shared fixture by name.

        Args:
            name: The fixture name

        Returns:
            The shared fixture or None if not found
        """
        return cls._registry.get(name)

    @classmethod
    def create(
        cls,
        name: str,
        setup_class: Optional[Callable[[FixtureContext], Any]] = None,
        teardown_class: Optional[Callable[[FixtureContext, Any], None]] = None,
        setup: Optional[Callable[[FixtureContext], Any]] = None,
        teardown: Optional[Callable[[FixtureContext, Any], None]] = None,
    ) -> "SharedFixture":
        """
        Create or get a shared fixture with custom callbacks.

        Args:
            name: Unique name for the fixture
            setup_class: Callback for class-level setup, returns shared data
            teardown_class: Callback for class-level teardown
            setup: Callback for per-test setup
            teardown: Callback for per-test teardown

        Returns:
            The shared fixture instance
        """
        if name in cls._registry:
            return cls._registry[name]

        fixture = _CallbackSharedFixture(
            name,
            setup_class=setup_class,
            teardown_class=teardown_class,
            setup=setup,
            teardown=teardown,
        )
        cls._registry[name] = fixture
        return fixture

    @property
    def data(self) -> Dict[str, Any]:
        """
        Get the shared data dictionary.

        Store data here that should be accessible across all tests.
        """
        return self._shared_data

    def _do_class_setup(self, context: FixtureContext) -> None:
        """Only perform class setup on first reference."""
        self._ref_count += 1
        if self._ref_count == 1:
            super()._do_class_setup(context)

    def _do_class_teardown(self, context: FixtureContext) -> None:
        """Only perform class teardown when last reference is released."""
        self._ref_count -= 1
        if self._ref_count == 0:
            super()._do_class_teardown(context)
            self._shared_data.clear()


class _CallbackSharedFixture(SharedFixture):
    """Internal: SharedFixture implementation using callbacks."""

    def __init__(
        self,
        name: str,
        setup_class: Optional[Callable[[FixtureContext], Any]] = None,
        teardown_class: Optional[Callable[[FixtureContext, Any], None]] = None,
        setup: Optional[Callable[[FixtureContext], Any]] = None,
        teardown: Optional[Callable[[FixtureContext, Any], None]] = None,
    ) -> None:
        super().__init__(name)
        self._setup_class_cb = setup_class
        self._teardown_class_cb = teardown_class
        self._setup_cb = setup
        self._teardown_cb = teardown
        self._class_result: Any = None

    def setUpClass(self, context: FixtureContext) -> None:
        if self._setup_class_cb:
            self._class_result = self._setup_class_cb(context)

    def tearDownClass(self, context: FixtureContext) -> None:
        if self._teardown_class_cb:
            self._teardown_class_cb(context, self._class_result)
        self._class_result = None

    def setUp(self, context: FixtureContext) -> None:
        if self._setup_cb:
            self._setup_cb(context)

    def tearDown(self, context: FixtureContext) -> None:
        if self._teardown_cb:
            self._teardown_cb(context, self._class_result)


class CompositeFixture(TestFixture):
    """
    A fixture that combines multiple fixtures into one.

    Fixtures are set up in order and torn down in reverse order.

    Example:
        database = DatabaseFixture()
        cache = CacheFixture()

        combined = CompositeFixture("full_stack", [database, cache])

        class MyTests(TestSuite):
            fixtures = [combined]
    """

    def __init__(self, name: str, fixtures: List[TestFixture]) -> None:
        """
        Initialize a composite fixture.

        Args:
            name: Name for the composite fixture
            fixtures: List of fixtures to combine (in setup order)
        """
        super().__init__(name)
        self._fixtures = fixtures

    @property
    def fixtures(self) -> List[TestFixture]:
        """Get the list of contained fixtures."""
        return self._fixtures.copy()

    def setUpClass(self, context: FixtureContext) -> None:
        """Set up all fixtures in order."""
        for fixture in self._fixtures:
            fixture._do_class_setup(context)

    def tearDownClass(self, context: FixtureContext) -> None:
        """Tear down all fixtures in reverse order."""
        errors = []
        for fixture in reversed(self._fixtures):
            try:
                fixture._do_class_teardown(context)
            except Exception as e:
                errors.append(e)

        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise Exception(f"Multiple teardown errors: {errors}")

    def setUp(self, context: FixtureContext) -> None:
        """Set up all fixtures in order."""
        setup_fixtures = []
        try:
            for fixture in self._fixtures:
                fixture._do_setup(context)
                setup_fixtures.append(fixture)
        except Exception:
            # Teardown fixtures that were set up
            for fixture in reversed(setup_fixtures):
                try:
                    fixture._do_teardown(context)
                except Exception:
                    pass
            raise

    def tearDown(self, context: FixtureContext) -> None:
        """Tear down all fixtures in reverse order."""
        errors = []
        for fixture in reversed(self._fixtures):
            try:
                fixture._do_teardown(context)
            except Exception as e:
                errors.append(e)

        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise Exception(f"Multiple teardown errors: {errors}")


def fixture(
    setup: Optional[Callable[[], T]] = None,
    teardown: Optional[Callable[[T], None]] = None,
    setup_class: Optional[Callable[[], T]] = None,
    teardown_class: Optional[Callable[[T], None]] = None,
    name: Optional[str] = None,
) -> TestFixture:
    """
    Factory function to create a simple fixture from callbacks.

    Args:
        setup: Callback for per-test setup
        teardown: Callback for per-test teardown
        setup_class: Callback for class-level setup
        teardown_class: Callback for class-level teardown
        name: Optional fixture name

    Returns:
        A TestFixture instance

    Example:
        temp_dir_fixture = fixture(
            setup=lambda: tempfile.mkdtemp(),
            teardown=lambda d: shutil.rmtree(d),
            name="temp_dir",
        )
    """

    class CallbackFixture(TestFixture):
        _result: Any = None
        _class_result: Any = None

        def setUpClass(self, context: FixtureContext) -> None:
            if setup_class:
                self._class_result = setup_class()

        def tearDownClass(self, context: FixtureContext) -> None:
            if teardown_class:
                teardown_class(self._class_result)
            self._class_result = None

        def setUp(self, context: FixtureContext) -> None:
            if setup:
                self._result = setup()

        def tearDown(self, context: FixtureContext) -> None:
            if teardown:
                teardown(self._result)
            self._result = None

    return CallbackFixture(name or "callback_fixture")


def shared_fixture(
    name: str,
    setup_class: Optional[Callable[[], T]] = None,
    teardown_class: Optional[Callable[[T], None]] = None,
) -> SharedFixture:
    """
    Factory function to create a shared fixture from callbacks.

    Args:
        name: Unique name for the shared fixture
        setup_class: Callback for class-level setup
        teardown_class: Callback for class-level teardown

    Returns:
        A SharedFixture instance

    Example:
        asset_cache = shared_fixture(
            "asset_cache",
            setup_class=lambda: AssetCache(),
            teardown_class=lambda cache: cache.clear(),
        )
    """
    return SharedFixture.create(
        name,
        setup_class=lambda ctx: setup_class() if setup_class else None,
        teardown_class=lambda ctx, result: teardown_class(result) if teardown_class else None,
    )
