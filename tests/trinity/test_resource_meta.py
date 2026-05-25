"""
Comprehensive tests for ResourceMeta - Metaclass for singleton resources.

Tests cover:
- Resource ID assignment
- Singleton enforcement (same instance returned)
- initialize_all / shutdown_all lifecycle
- Dependency-ordered initialization
- get_instance / has_instance
- reset_instance
- Registry clearing
- Resources with dependencies
"""
import pytest

from trinity.metaclasses import ResourceMeta
from trinity.constants import DEFAULT_RESOURCE_PRIORITY


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    ResourceMeta.clear_registry()
    yield
    ResourceMeta.clear_registry()


def test_resource_id_assignment():
    """Test that resource IDs are assigned sequentially."""

    class Res1(metaclass=ResourceMeta):
        pass

    class Res2(metaclass=ResourceMeta):
        pass

    class Res3(metaclass=ResourceMeta):
        pass

    assert Res1._resource_id == 1
    assert Res2._resource_id == 2
    assert Res3._resource_id == 3


def test_singleton_enforcement():
    """Test that only one instance per resource class is created."""

    class TestResource(metaclass=ResourceMeta):
        def __init__(self):
            self.value = 42

    # Create first instance
    instance1 = TestResource()

    # Create second instance - should get same object
    instance2 = TestResource()

    assert instance1 is instance2
    assert instance1.value == 42


def test_singleton_with_args_raises():
    """Test that calling singleton with args after creation raises TypeError."""

    class TestResource(metaclass=ResourceMeta):
        def __init__(self, value=0):
            self.value = value

    # First call with arg
    instance1 = TestResource(42)

    # Second call with different arg should raise
    with pytest.raises(TypeError, match="singleton resource"):
        TestResource(99)


def test_priority_default():
    """Test that resources default to DEFAULT_RESOURCE_PRIORITY."""

    class TestResource(metaclass=ResourceMeta):
        pass

    assert TestResource._resource_priority == DEFAULT_RESOURCE_PRIORITY


def test_priority_custom():
    """Test that custom priority can be set."""

    class HighPriorityResource(metaclass=ResourceMeta):
        _resource_priority = 10

    class LowPriorityResource(metaclass=ResourceMeta):
        _resource_priority = 200

    assert HighPriorityResource._resource_priority == 10
    assert LowPriorityResource._resource_priority == 200


def test_get_instance():
    """Test get_instance retrieves the singleton instance."""

    class TestResource(metaclass=ResourceMeta):
        pass

    instance = TestResource()

    retrieved = ResourceMeta.get_instance(TestResource)

    assert retrieved is instance


def test_get_instance_not_created():
    """Test get_instance returns None if not yet instantiated."""

    class TestResource(metaclass=ResourceMeta):
        pass

    assert ResourceMeta.get_instance(TestResource) is None


def test_has_instance():
    """Test has_instance checks if resource was instantiated."""

    class TestResource(metaclass=ResourceMeta):
        pass

    assert ResourceMeta.has_instance(TestResource) is False

    TestResource()

    assert ResourceMeta.has_instance(TestResource) is True


def test_initialize_all_basic():
    """Test that initialize_all creates all resources."""

    class Res1(metaclass=ResourceMeta):
        def __init__(self):
            self.initialized = True

    class Res2(metaclass=ResourceMeta):
        def __init__(self):
            self.initialized = True

    # Neither instantiated yet
    assert not ResourceMeta.has_instance(Res1)
    assert not ResourceMeta.has_instance(Res2)

    # Initialize all
    ResourceMeta.initialize_all()

    # Both should be instantiated
    assert ResourceMeta.has_instance(Res1)
    assert ResourceMeta.has_instance(Res2)


def test_initialize_all_respects_priority():
    """Test that initialize_all initializes in priority order."""

    initialization_order = []

    class LowPriority(metaclass=ResourceMeta):
        _resource_priority = 200

        def __init__(self):
            initialization_order.append("Low")

    class HighPriority(metaclass=ResourceMeta):
        _resource_priority = 10

        def __init__(self):
            initialization_order.append("High")

    class MedPriority(metaclass=ResourceMeta):
        _resource_priority = 100

        def __init__(self):
            initialization_order.append("Med")

    ResourceMeta.initialize_all()

    # Should be initialized in priority order (lower number first)
    assert initialization_order == ["High", "Med", "Low"]


def test_initialize_all_respects_dependencies():
    """Test that initialize_all initializes dependencies first."""

    initialization_order = []

    class BaseResource(metaclass=ResourceMeta):
        def __init__(self):
            initialization_order.append("Base")

    class DependentResource(metaclass=ResourceMeta):
        _resource_dependencies = (BaseResource,)

        def __init__(self):
            initialization_order.append("Dependent")

    ResourceMeta.initialize_all()

    # Base should be initialized before Dependent
    assert initialization_order == ["Base", "Dependent"]


def test_initialize_all_circular_dependency():
    """Test that circular dependencies raise RuntimeError."""

    class Res1(metaclass=ResourceMeta):
        _resource_dependencies = ()  # Will be set manually

    class Res2(metaclass=ResourceMeta):
        _resource_dependencies = (Res1,)

    # Manually create circular dependency
    Res1._resource_dependencies = (Res2,)

    with pytest.raises(RuntimeError, match="circular or unsatisfied dependencies"):
        ResourceMeta.initialize_all()


def test_shutdown_all():
    """Test that shutdown_all calls shutdown on all resources."""

    shutdown_order = []

    class Res1(metaclass=ResourceMeta):
        def shutdown(self):
            shutdown_order.append("Res1")

    class Res2(metaclass=ResourceMeta):
        def shutdown(self):
            shutdown_order.append("Res2")

    # Create instances
    Res1()
    Res2()

    ResourceMeta.shutdown_all()

    # Both should have been shut down
    assert "Res1" in shutdown_order
    assert "Res2" in shutdown_order


def test_shutdown_all_reverse_order():
    """Test that shutdown_all shuts down in reverse ID order."""

    shutdown_ids = []

    class Res1(metaclass=ResourceMeta):
        def shutdown(self):
            shutdown_ids.append(1)

    class Res2(metaclass=ResourceMeta):
        def shutdown(self):
            shutdown_ids.append(2)

    class Res3(metaclass=ResourceMeta):
        def shutdown(self):
            shutdown_ids.append(3)

    # Create instances
    Res1()
    Res2()
    Res3()

    ResourceMeta.shutdown_all()

    # Should be in reverse order
    assert shutdown_ids == [3, 2, 1]


def test_shutdown_all_clears_instances():
    """Test that shutdown_all clears all instances."""

    class TestResource(metaclass=ResourceMeta):
        pass

    TestResource()

    assert ResourceMeta.has_instance(TestResource)

    ResourceMeta.shutdown_all()

    assert not ResourceMeta.has_instance(TestResource)


def test_reset_instance():
    """Test that reset_instance removes and shuts down a specific resource."""

    shutdown_called = []

    class TestResource(metaclass=ResourceMeta):
        def shutdown(self):
            shutdown_called.append(True)

    instance = TestResource()

    assert ResourceMeta.has_instance(TestResource)

    ResourceMeta.reset_instance(TestResource)

    assert not ResourceMeta.has_instance(TestResource)
    assert len(shutdown_called) == 1


def test_reset_instance_allows_recreation():
    """Test that after reset_instance, resource can be recreated."""

    class TestResource(metaclass=ResourceMeta):
        def __init__(self, value=0):
            self.value = value

    instance1 = TestResource(42)

    ResourceMeta.reset_instance(TestResource)

    # Should be able to create with different value
    instance2 = TestResource(99)

    assert instance1 is not instance2
    assert instance2.value == 99


def test_reset_instance_not_a_resource():
    """Test that reset_instance raises TypeError for non-resource."""

    class NotAResource:
        pass

    with pytest.raises(TypeError):
        ResourceMeta.reset_instance(NotAResource)


def test_clear_registry():
    """Test that clear_registry removes all resources and instances."""

    class Res1(metaclass=ResourceMeta):
        pass

    class Res2(metaclass=ResourceMeta):
        pass

    # Create instances
    Res1()
    Res2()

    assert len(ResourceMeta.all_resources()) == 2
    assert ResourceMeta.has_instance(Res1)

    ResourceMeta.clear_registry()

    assert len(ResourceMeta.all_resources()) == 0
    assert not ResourceMeta.has_instance(Res1)


def test_clear_registry_resets_id():
    """Test that clear_registry resets ID counter."""

    class Res1(metaclass=ResourceMeta):
        pass

    assert Res1._resource_id == 1

    ResourceMeta.clear_registry()

    class Res2(metaclass=ResourceMeta):
        pass

    assert Res2._resource_id == 1


def test_base_resource_class_skipped():
    """Test that base Resource class is not registered."""

    class Resource(metaclass=ResourceMeta):
        pass

    assert len(ResourceMeta.all_resources()) == 0


def test_get_by_id():
    """Test retrieving resource by ID."""

    class TestResource(metaclass=ResourceMeta):
        pass

    retrieved = ResourceMeta.get_by_id(TestResource._resource_id)
    assert retrieved is TestResource


def test_get_by_name():
    """Test retrieving resource by qualified name."""

    class TestResource(metaclass=ResourceMeta):
        pass

    retrieved = ResourceMeta.get_by_name(TestResource._resource_name)
    assert retrieved is TestResource


def test_all_resources():
    """Test all_resources returns all registered resources."""

    class Res1(metaclass=ResourceMeta):
        pass

    class Res2(metaclass=ResourceMeta):
        pass

    all_res = ResourceMeta.all_resources()

    assert len(all_res) == 2
    assert Res1 in all_res
    assert Res2 in all_res


def test_resource_qualified_name():
    """Test that resource qualified name includes module."""

    class TestResource(metaclass=ResourceMeta):
        pass

    assert "." in TestResource._resource_name
    assert TestResource._resource_name.endswith(".TestResource")


def test_dependencies_tuple_default():
    """Test that _resource_dependencies defaults to empty tuple."""

    class TestResource(metaclass=ResourceMeta):
        pass

    assert TestResource._resource_dependencies == ()


def test_post_init_hook():
    """Test that _on_resource_created hook is called after instantiation."""

    hook_called = []

    class TestResource(metaclass=ResourceMeta):
        def _on_resource_created(self):
            hook_called.append(True)

    TestResource()

    assert len(hook_called) == 1


def test_initialize_all_skip_already_initialized():
    """Test that initialize_all skips already-initialized resources."""

    init_count = []

    class TestResource(metaclass=ResourceMeta):
        def __init__(self):
            init_count.append(1)

    # Manually create instance
    TestResource()

    assert len(init_count) == 1

    # Call initialize_all
    ResourceMeta.initialize_all()

    # Should not re-initialize
    assert len(init_count) == 1


def test_lazy_resource_with_dependencies():
    """Test that lazy resources with dependencies are handled correctly."""

    class BaseResource(metaclass=ResourceMeta):
        def __init__(self):
            self.value = 42

    class LazyResource(metaclass=ResourceMeta):
        _resource_lazy = True
        _resource_dependencies = (BaseResource,)

        def __init__(self):
            self.base = ResourceMeta.get_instance(BaseResource)

    # Initialize all (should skip lazy resource)
    ResourceMeta.initialize_all()

    # Base should be initialized
    assert ResourceMeta.has_instance(BaseResource)

    # Lazy should not be initialized
    assert not ResourceMeta.has_instance(LazyResource)

    # Now create lazy resource via get_or_create
    lazy = ResourceMeta.get_or_create(LazyResource)

    # Should succeed since dependency is satisfied
    assert lazy.base.value == 42


def test_get_or_create_unsatisfied_dependency():
    """Test that get_or_create raises RuntimeError for unsatisfied dependencies."""

    class BaseResource(metaclass=ResourceMeta):
        pass

    class DependentResource(metaclass=ResourceMeta):
        _resource_dependencies = (BaseResource,)

    # Try to create without initializing dependency
    with pytest.raises(RuntimeError, match="must be initialized first"):
        ResourceMeta.get_or_create(DependentResource)


def test_get_or_create_concurrent_safety():
    """Test that get_or_create is thread-safe."""
    import threading

    instances = []

    class TestResource(metaclass=ResourceMeta):
        def __init__(self):
            import time

            time.sleep(0.01)  # Simulate slow initialization

    def create_resource():
        instances.append(ResourceMeta.get_or_create(TestResource))

    # Create multiple threads trying to get/create the resource
    threads = [threading.Thread(target=create_resource) for _ in range(5)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # All threads should get the same instance
    assert len(set(id(inst) for inst in instances)) == 1


def test_initialize_all_with_exception():
    """Test that initialize_all handles initialization exceptions properly."""

    class GoodResource(metaclass=ResourceMeta):
        def __init__(self):
            self.initialized = True

    class BadResource(metaclass=ResourceMeta):
        def __init__(self):
            raise ValueError("Initialization failed")

    class AnotherGoodResource(metaclass=ResourceMeta):
        def __init__(self):
            self.initialized = True

    # Should raise RuntimeError with details about failed resource
    with pytest.raises(RuntimeError, match="Failed to initialize resources"):
        ResourceMeta.initialize_all()

    # Good resources should still be initialized despite BadResource failure
    assert ResourceMeta.has_instance(GoodResource)
    assert ResourceMeta.has_instance(AnotherGoodResource)


def test_shutdown_with_exception():
    """Test that shutdown_all continues even if shutdown() raises exception."""

    shutdown_order = []

    class GoodResource1(metaclass=ResourceMeta):
        def shutdown(self):
            shutdown_order.append("Good1")

    class BadResource(metaclass=ResourceMeta):
        def shutdown(self):
            shutdown_order.append("Bad")
            raise RuntimeError("Shutdown failed")

    class GoodResource2(metaclass=ResourceMeta):
        def shutdown(self):
            shutdown_order.append("Good2")

    # Create all instances
    GoodResource1()
    BadResource()
    GoodResource2()

    # Shutdown should continue despite exception
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ResourceMeta.shutdown_all()

        # Should have warnings about failed shutdown
        assert any("Error shutting down" in str(warning.message) for warning in w)

    # All resources should have attempted shutdown
    assert "Good1" in shutdown_order or "Good2" in shutdown_order
    assert "Bad" in shutdown_order

    # All instances should be cleared
    assert not ResourceMeta.has_instance(GoodResource1)
    assert not ResourceMeta.has_instance(BadResource)
    assert not ResourceMeta.has_instance(GoodResource2)


def test_get_or_create_init_exception():
    """Test that get_or_create raises RuntimeError if __init__ raises exception."""

    class FailingResource(metaclass=ResourceMeta):
        def __init__(self):
            raise ValueError("Init failed")

    with pytest.raises(RuntimeError, match="Failed to create resource"):
        ResourceMeta.get_or_create(FailingResource)


def test_is_lazy_consistency():
    """Test that is_lazy raises TypeError for non-resources (consistent with get_instance)."""

    class NotAResource:
        pass

    # is_lazy should raise TypeError (not return False)
    with pytest.raises(TypeError, match="not a resource type"):
        ResourceMeta.is_lazy(NotAResource)


def test_lazy_resource_skipped_by_initialize_all():
    """Test that lazy resources are not initialized by initialize_all."""

    init_count = []

    class LazyRes(metaclass=ResourceMeta):
        _resource_lazy = True

        def __init__(self):
            init_count.append(1)

    ResourceMeta.initialize_all()

    # Lazy resource should not be initialized
    assert len(init_count) == 0
    assert not ResourceMeta.has_instance(LazyRes)


def test_get_or_create_invalid_dependency():
    """Test that get_or_create validates dependency types."""

    class BadResource(metaclass=ResourceMeta):
        _resource_dependencies = ("not_a_class",)

    with pytest.raises(RuntimeError, match="not a valid resource type"):
        ResourceMeta.get_or_create(BadResource)


def test_initialize_all_complex_dependency_chain():
    """Test initialize_all with complex dependency chain."""

    init_order = []

    class ResA(metaclass=ResourceMeta):
        _resource_priority = 100

        def __init__(self):
            init_order.append("A")

    class ResB(metaclass=ResourceMeta):
        _resource_priority = 50
        _resource_dependencies = (ResA,)

        def __init__(self):
            init_order.append("B")

    class ResC(metaclass=ResourceMeta):
        _resource_priority = 10
        _resource_dependencies = (ResB,)

        def __init__(self):
            init_order.append("C")

    ResourceMeta.initialize_all()

    # Should be initialized in dependency order (A -> B -> C)
    assert init_order == ["A", "B", "C"]
