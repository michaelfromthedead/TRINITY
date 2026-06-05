"""
Whitebox tests for the generic backend registry system.

Tests the BackendRegistry class for thread safety, registration,
creation, default handling, and edge cases.
"""

import pytest
import sys
import threading
import time
from typing import Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/home/user/dev/USER/PROJECTS_VOID/TRINITY')

from engine.platform.registry import BackendRegistry


# ============================================================================
# Test Protocols and Mock Backends
# ============================================================================

@runtime_checkable
class TestBackendProtocol(Protocol):
    """Protocol for test backends."""
    def process(self) -> str: ...


class MockBackendA:
    """Mock backend implementation A."""
    def __init__(self, value: int = 0):
        self.value = value

    def process(self) -> str:
        return f"A:{self.value}"


class MockBackendB:
    """Mock backend implementation B."""
    def __init__(self, name: str = "default"):
        self.name = name

    def process(self) -> str:
        return f"B:{self.name}"


class MockBackendC:
    """Mock backend implementation C with multiple args."""
    def __init__(self, x: int, y: int, z: str = "default"):
        self.x = x
        self.y = y
        self.z = z

    def process(self) -> str:
        return f"C:{self.x},{self.y},{self.z}"


class SlowInitBackend:
    """Backend with slow initialization for concurrency tests."""
    def __init__(self, delay: float = 0.01):
        time.sleep(delay)
        self.initialized = True

    def process(self) -> str:
        return "slow"


class FailingBackend:
    """Backend that raises on initialization."""
    def __init__(self, should_fail: bool = True):
        if should_fail:
            raise RuntimeError("Intentional failure")
        self.working = True

    def process(self) -> str:
        return "fail"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def registry():
    """Provide a fresh registry for each test."""
    return BackendRegistry[TestBackendProtocol]()


@pytest.fixture
def populated_registry():
    """Provide a registry with multiple backends registered."""
    reg = BackendRegistry[TestBackendProtocol]()
    reg.register("a", MockBackendA)
    reg.register("b", MockBackendB)
    reg.register("c", MockBackendC)
    return reg


# ============================================================================
# Basic Registration Tests
# ============================================================================

class TestRegistration:
    """Tests for backend registration functionality."""

    def test_register_single_backend(self, registry):
        """Verify registering a single backend."""
        registry.register("test", MockBackendA)
        assert registry.get("test") == MockBackendA

    def test_register_multiple_backends(self, registry):
        """Verify registering multiple backends."""
        registry.register("a", MockBackendA)
        registry.register("b", MockBackendB)
        registry.register("c", MockBackendC)

        assert registry.get("a") == MockBackendA
        assert registry.get("b") == MockBackendB
        assert registry.get("c") == MockBackendC

    def test_register_overwrite_existing(self, registry):
        """Verify re-registering overwrites existing backend."""
        registry.register("test", MockBackendA)
        registry.register("test", MockBackendB)
        assert registry.get("test") == MockBackendB

    def test_register_with_empty_name(self, registry):
        """Verify registering with empty name is allowed."""
        registry.register("", MockBackendA)
        assert registry.get("") == MockBackendA

    def test_register_with_special_characters(self, registry):
        """Verify registering with special characters in name."""
        registry.register("test-backend_v2.0", MockBackendA)
        assert registry.get("test-backend_v2.0") == MockBackendA

    def test_register_with_unicode_name(self, registry):
        """Verify registering with unicode characters."""
        registry.register("backend_", MockBackendA)
        assert registry.get("backend_") == MockBackendA


# ============================================================================
# Default Backend Tests
# ============================================================================

class TestDefaultBackend:
    """Tests for default backend functionality."""

    def test_no_default_initially(self, registry):
        """Verify no default backend initially."""
        assert registry.default() is None

    def test_set_default_on_register(self, registry):
        """Verify setting default during registration."""
        registry.register("test", MockBackendA, set_default=True)
        assert registry.default() == "test"

    def test_set_default_overwrites_previous(self, registry):
        """Verify setting new default overwrites previous."""
        registry.register("a", MockBackendA, set_default=True)
        registry.register("b", MockBackendB, set_default=True)
        assert registry.default() == "b"

    def test_register_without_default_preserves_existing(self, registry):
        """Verify registering without set_default preserves existing default."""
        registry.register("a", MockBackendA, set_default=True)
        registry.register("b", MockBackendB)  # No set_default
        assert registry.default() == "a"

    def test_default_not_set_by_first_registration(self, registry):
        """Verify first registration doesn't auto-set default."""
        registry.register("a", MockBackendA)
        assert registry.default() is None


# ============================================================================
# Get Backend Tests
# ============================================================================

class TestGetBackend:
    """Tests for backend retrieval functionality."""

    def test_get_registered_backend(self, populated_registry):
        """Verify getting a registered backend."""
        assert populated_registry.get("a") == MockBackendA

    def test_get_nonexistent_backend(self, populated_registry):
        """Verify getting nonexistent backend returns None."""
        assert populated_registry.get("nonexistent") is None

    def test_get_after_overwrite(self, registry):
        """Verify getting backend after overwrite returns new one."""
        registry.register("test", MockBackendA)
        registry.register("test", MockBackendB)
        assert registry.get("test") == MockBackendB

    def test_get_with_empty_name(self, registry):
        """Verify getting backend with empty name."""
        registry.register("", MockBackendA)
        assert registry.get("") == MockBackendA

    def test_get_preserves_class_identity(self, populated_registry):
        """Verify getting backend returns exact class."""
        backend_class = populated_registry.get("a")
        assert backend_class is MockBackendA


# ============================================================================
# Create Instance Tests
# ============================================================================

class TestCreateInstance:
    """Tests for backend instance creation."""

    def test_create_with_explicit_name(self, populated_registry):
        """Verify creating instance with explicit name."""
        instance = populated_registry.create("a")
        assert isinstance(instance, MockBackendA)
        assert instance.value == 0  # Default value

    def test_create_with_default(self, registry):
        """Verify creating instance uses default backend."""
        registry.register("test", MockBackendA, set_default=True)
        instance = registry.create()
        assert isinstance(instance, MockBackendA)

    def test_create_with_args(self, populated_registry):
        """Verify creating instance with positional args."""
        instance = populated_registry.create("a", 42)
        assert isinstance(instance, MockBackendA)
        assert instance.value == 42

    def test_create_with_kwargs(self, populated_registry):
        """Verify creating instance with keyword args to backend."""
        # Using kwargs that go to the backend class, not to create()
        instance = populated_registry.create("c", x=1, y=2, z="custom")
        assert isinstance(instance, MockBackendC)
        assert instance.z == "custom"

    def test_create_with_mixed_args(self, populated_registry):
        """Verify creating instance with mixed args and kwargs."""
        instance = populated_registry.create("c", 1, 2, z="custom")
        assert isinstance(instance, MockBackendC)
        assert instance.x == 1
        assert instance.y == 2
        assert instance.z == "custom"

    def test_create_unknown_backend_raises(self, populated_registry):
        """Verify creating unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            populated_registry.create("nonexistent")

    def test_create_without_default_raises(self, populated_registry):
        """Verify creating without name and no default raises ValueError."""
        with pytest.raises(ValueError, match="No default backend set"):
            populated_registry.create()

    def test_create_returns_new_instances(self, populated_registry):
        """Verify each create call returns new instance."""
        instance1 = populated_registry.create("a")
        instance2 = populated_registry.create("a")
        assert instance1 is not instance2

    def test_create_with_failing_backend(self, registry):
        """Verify create propagates backend initialization errors."""
        registry.register("fail", FailingBackend)
        with pytest.raises(RuntimeError, match="Intentional failure"):
            registry.create("fail")

    def test_create_failing_backend_with_args(self, registry):
        """Verify create with args can avoid failure."""
        registry.register("fail", FailingBackend)
        instance = registry.create("fail", should_fail=False)
        assert instance.working is True


# ============================================================================
# List Backends Tests
# ============================================================================

class TestListBackends:
    """Tests for listing registered backends."""

    def test_list_empty_registry(self, registry):
        """Verify listing empty registry returns empty list."""
        assert registry.list() == []

    def test_list_single_backend(self, registry):
        """Verify listing single backend."""
        registry.register("test", MockBackendA)
        assert registry.list() == ["test"]

    def test_list_multiple_backends(self, populated_registry):
        """Verify listing multiple backends."""
        backends = populated_registry.list()
        assert sorted(backends) == ["a", "b", "c"]

    def test_list_returns_sorted(self, registry):
        """Verify list returns sorted backend names."""
        registry.register("c", MockBackendC)
        registry.register("a", MockBackendA)
        registry.register("b", MockBackendB)
        assert registry.list() == ["a", "b", "c"]

    def test_list_returns_snapshot(self, registry):
        """Verify list returns copy, not internal state."""
        registry.register("test", MockBackendA)
        backends = registry.list()
        backends.append("fake")
        assert registry.list() == ["test"]


# ============================================================================
# Thread Safety Tests
# ============================================================================

class TestThreadSafety:
    """Tests for thread safety of registry operations."""

    def test_concurrent_registration(self, registry):
        """Verify concurrent registration is thread-safe."""
        num_threads = 20
        backends_per_thread = 10

        def register_backends(thread_id):
            for i in range(backends_per_thread):
                name = f"backend_{thread_id}_{i}"
                registry.register(name, MockBackendA)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_backends, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        # All backends should be registered
        assert len(registry.list()) == num_threads * backends_per_thread

    def test_concurrent_get(self, populated_registry):
        """Verify concurrent get operations are thread-safe."""
        num_threads = 50
        results = []

        def get_backend():
            backend = populated_registry.get("a")
            results.append(backend)

        threads = [threading.Thread(target=get_backend) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == num_threads
        assert all(r == MockBackendA for r in results)

    def test_concurrent_create(self, registry):
        """Verify concurrent create operations are thread-safe."""
        registry.register("slow", SlowInitBackend, set_default=True)
        num_threads = 10
        instances = []
        lock = threading.Lock()

        def create_instance():
            instance = registry.create()
            with lock:
                instances.append(instance)

        threads = [threading.Thread(target=create_instance) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == num_threads
        assert all(instance.initialized for instance in instances)

    def test_concurrent_list(self, populated_registry):
        """Verify concurrent list operations are thread-safe."""
        num_threads = 50
        results = []

        def list_backends():
            backends = populated_registry.list()
            results.append(backends)

        threads = [threading.Thread(target=list_backends) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == num_threads
        expected = ["a", "b", "c"]
        assert all(r == expected for r in results)

    def test_concurrent_default_access(self, registry):
        """Verify concurrent default() access is thread-safe."""
        registry.register("test", MockBackendA, set_default=True)
        num_threads = 50
        results = []

        def get_default():
            default = registry.default()
            results.append(default)

        threads = [threading.Thread(target=get_default) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == num_threads
        assert all(r == "test" for r in results)

    def test_concurrent_read_write(self, registry):
        """Verify concurrent read/write operations are thread-safe."""
        num_iterations = 100
        errors = []

        def writer():
            for i in range(num_iterations):
                registry.register(f"writer_{i}", MockBackendA)

        def reader():
            for _ in range(num_iterations):
                try:
                    registry.list()
                    registry.default()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_register_none_class(self, registry):
        """Verify registering None class."""
        # This should work but create() would fail
        registry.register("none", None)
        assert registry.get("none") is None

    def test_register_lambda(self, registry):
        """Verify registering a lambda/callable."""
        registry.register("lambda", lambda: "test")
        result = registry.create("lambda")
        assert result == "test"

    def test_register_class_with_no_args(self, registry):
        """Verify registering class with no __init__ args."""
        class NoArgsBackend:
            pass

        registry.register("noargs", NoArgsBackend)
        instance = registry.create("noargs")
        assert isinstance(instance, NoArgsBackend)

    def test_large_number_of_backends(self, registry):
        """Verify registry handles large number of backends."""
        num_backends = 1000

        for i in range(num_backends):
            registry.register(f"backend_{i:04d}", MockBackendA)

        assert len(registry.list()) == num_backends

        # Verify we can still get and create
        assert registry.get("backend_0500") == MockBackendA
        instance = registry.create("backend_0999")
        assert isinstance(instance, MockBackendA)

    def test_backend_with_star_args(self, registry):
        """Verify backend with *args works."""
        class StarArgsBackend:
            def __init__(self, *args):
                self.args = args

        registry.register("star", StarArgsBackend)
        instance = registry.create("star", 1, 2, 3, 4, 5)
        assert instance.args == (1, 2, 3, 4, 5)

    def test_backend_with_star_kwargs(self, registry):
        """Verify backend with **kwargs works."""
        class StarKwargsBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        registry.register("kwargs", StarKwargsBackend)
        instance = registry.create("kwargs", a=1, b=2, c=3)
        assert instance.kwargs == {"a": 1, "b": 2, "c": 3}

    def test_case_sensitive_names(self, registry):
        """Verify backend names are case-sensitive."""
        registry.register("Test", MockBackendA)
        registry.register("test", MockBackendB)
        registry.register("TEST", MockBackendC)

        assert registry.get("Test") == MockBackendA
        assert registry.get("test") == MockBackendB
        assert registry.get("TEST") == MockBackendC
        assert len(registry.list()) == 3

    def test_whitespace_in_names(self, registry):
        """Verify whitespace in names is preserved."""
        registry.register("  spaces  ", MockBackendA)
        registry.register("tabs\t", MockBackendB)

        assert registry.get("  spaces  ") == MockBackendA
        assert registry.get("tabs\t") == MockBackendB
        assert registry.get("spaces") is None


# ============================================================================
# Type Safety Tests
# ============================================================================

class TestTypeSafety:
    """Tests for type parameter behavior."""

    def test_registry_accepts_subclass(self):
        """Verify registry accepts subclasses of type parameter."""
        class BaseBackend:
            pass

        class DerivedBackend(BaseBackend):
            pass

        registry = BackendRegistry[BaseBackend]()
        registry.register("derived", DerivedBackend)
        instance = registry.create("derived")
        assert isinstance(instance, DerivedBackend)
        assert isinstance(instance, BaseBackend)

    def test_multiple_inheritance(self):
        """Verify registry handles multiple inheritance."""
        class MixinA:
            pass

        class MixinB:
            pass

        class MultiBackend(MixinA, MixinB):
            pass

        registry = BackendRegistry[MixinA]()
        registry.register("multi", MultiBackend)
        instance = registry.create("multi")
        assert isinstance(instance, MixinA)
        assert isinstance(instance, MixinB)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_registration_performance(self, registry):
        """Verify registration performance is acceptable."""
        num_backends = 10000

        start = time.perf_counter()
        for i in range(num_backends):
            registry.register(f"backend_{i}", MockBackendA)
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"Registration too slow: {elapsed:.2f}s"

    def test_get_performance(self, registry):
        """Verify get performance is acceptable."""
        # Register many backends
        for i in range(1000):
            registry.register(f"backend_{i}", MockBackendA)

        num_gets = 100000
        start = time.perf_counter()
        for i in range(num_gets):
            registry.get(f"backend_{i % 1000}")
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"Get too slow: {elapsed:.2f}s"

    def test_list_performance(self, registry):
        """Verify list performance is acceptable."""
        # Register many backends
        for i in range(1000):
            registry.register(f"backend_{i}", MockBackendA)

        num_lists = 10000
        start = time.perf_counter()
        for _ in range(num_lists):
            registry.list()
        elapsed = time.perf_counter() - start

        # Should complete in under 2 seconds
        assert elapsed < 2.0, f"List too slow: {elapsed:.2f}s"


# ============================================================================
# State Isolation Tests
# ============================================================================

class TestStateIsolation:
    """Tests for registry instance isolation."""

    def test_separate_registries_isolated(self):
        """Verify separate registry instances are isolated."""
        registry1 = BackendRegistry[TestBackendProtocol]()
        registry2 = BackendRegistry[TestBackendProtocol]()

        registry1.register("test", MockBackendA)
        registry2.register("test", MockBackendB)

        assert registry1.get("test") == MockBackendA
        assert registry2.get("test") == MockBackendB

    def test_separate_registries_default_isolated(self):
        """Verify defaults are isolated between registries."""
        registry1 = BackendRegistry[TestBackendProtocol]()
        registry2 = BackendRegistry[TestBackendProtocol]()

        registry1.register("a", MockBackendA, set_default=True)
        registry2.register("b", MockBackendB, set_default=True)

        assert registry1.default() == "a"
        assert registry2.default() == "b"

    def test_separate_registries_list_isolated(self):
        """Verify list() is isolated between registries."""
        registry1 = BackendRegistry[TestBackendProtocol]()
        registry2 = BackendRegistry[TestBackendProtocol]()

        registry1.register("a", MockBackendA)
        registry1.register("b", MockBackendB)
        registry2.register("c", MockBackendC)

        assert registry1.list() == ["a", "b"]
        assert registry2.list() == ["c"]
