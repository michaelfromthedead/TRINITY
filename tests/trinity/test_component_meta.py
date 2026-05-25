"""
Comprehensive tests for ComponentMeta - Metaclass for ECS components.

Tests cover:
- Component ID assignment (sequential, unique)
- Field processing from annotations
- Descriptor installation and chaining
- Mutable default detection and rejection
- Registry access (get_by_id, get_by_name, all_components, component_count)
- Registry clearing
- Inheritance handling
- Qualified name format
"""
import pytest

from trinity.metaclasses import ComponentMeta


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    ComponentMeta.clear_registry()
    yield
    ComponentMeta.clear_registry()


def test_component_id_assignment():
    """Test that component IDs are assigned sequentially."""

    class Player(metaclass=ComponentMeta):
        pass

    class Enemy(metaclass=ComponentMeta):
        pass

    class Item(metaclass=ComponentMeta):
        pass

    assert Player._component_id == 1
    assert Enemy._component_id == 2
    assert Item._component_id == 3


def test_component_id_unique():
    """Test that each component gets a unique ID."""

    components = []
    for i in range(20):
        cls = ComponentMeta(f"Comp{i}", (), {})
        components.append(cls)

    ids = [c._component_id for c in components]

    # All IDs should be unique
    assert len(ids) == len(set(ids))


def test_component_name_qualified():
    """Test that _component_name is the qualified name."""

    class TestComponent(metaclass=ComponentMeta):
        pass

    assert TestComponent._component_name == f"{TestComponent.__module__}.TestComponent"
    assert "." in TestComponent._component_name


def test_field_processing_basic():
    """Test that field annotations are processed into _field_types."""

    class Position(metaclass=ComponentMeta):
        x: float
        y: float
        z: float

    assert "x" in Position._field_types
    assert "y" in Position._field_types
    assert "z" in Position._field_types
    assert Position._field_types["x"] == float
    assert Position._field_types["y"] == float
    assert Position._field_types["z"] == float


def test_field_processing_ignores_private():
    """Test that private fields (starting with _) are ignored."""

    class TestComponent(metaclass=ComponentMeta):
        public_field: int
        _private_field: str
        __dunder_field: float

    assert "public_field" in TestComponent._field_types
    assert "_private_field" not in TestComponent._field_types
    assert "__dunder_field" not in TestComponent._field_types


def test_field_offsets_assigned():
    """Test that field offsets are assigned sequentially."""

    class TestComponent(metaclass=ComponentMeta):
        a: int
        b: int
        c: int

    assert TestComponent._field_offsets["a"] == 0
    assert TestComponent._field_offsets["b"] == 4
    assert TestComponent._field_offsets["c"] == 8


def test_field_defaults_captured():
    """Test that default values are captured."""

    class TestComponent(metaclass=ComponentMeta):
        x: int = 10
        y: str = "hello"
        z: float = 3.14

    assert TestComponent._field_defaults["x"] == 10
    assert TestComponent._field_defaults["y"] == "hello"
    assert TestComponent._field_defaults["z"] == 3.14


def test_mutable_default_raises_error():
    """Test that mutable defaults (list, dict, set) raise TypeError."""

    with pytest.raises(TypeError, match="Mutable default values are forbidden"):
        class BadComponent1(metaclass=ComponentMeta):
            items: list = []

    with pytest.raises(TypeError, match="Mutable default values are forbidden"):
        class BadComponent2(metaclass=ComponentMeta):
            data: dict = {}

    with pytest.raises(TypeError, match="Mutable default values are forbidden"):
        class BadComponent3(metaclass=ComponentMeta):
            tags: set = set()


def test_descriptor_installation():
    """Test that descriptors are installed for fields."""

    class Position(metaclass=ComponentMeta):
        x: float
        y: float

    # Fields should have descriptors installed
    assert hasattr(Position, "x")
    assert hasattr(Position, "y")

    # Descriptors should be in _field_descriptors
    assert "x" in Position._field_descriptors
    assert "y" in Position._field_descriptors

    # Descriptors should have __get__ and __set__
    assert hasattr(Position._field_descriptors["x"], "__get__")
    assert hasattr(Position._field_descriptors["x"], "__set__")


def test_get_by_id():
    """Test retrieving component by ID."""

    class TestComponent(metaclass=ComponentMeta):
        pass

    component_id = TestComponent._component_id

    retrieved = ComponentMeta.get_by_id(component_id)
    assert retrieved is TestComponent


def test_get_by_id_nonexistent():
    """Test get_by_id with non-existent ID returns None."""

    result = ComponentMeta.get_by_id(9999)
    assert result is None


def test_get_by_name():
    """Test retrieving component by qualified name."""

    class TestComponent(metaclass=ComponentMeta):
        pass

    name = TestComponent._component_name
    retrieved = ComponentMeta.get_by_name(name)

    assert retrieved is TestComponent


def test_get_by_name_nonexistent():
    """Test get_by_name with non-existent name returns None."""

    result = ComponentMeta.get_by_name("nonexistent.Component")
    assert result is None


def test_all_components():
    """Test all_components returns all registered components."""

    class Comp1(metaclass=ComponentMeta):
        pass

    class Comp2(metaclass=ComponentMeta):
        pass

    class Comp3(metaclass=ComponentMeta):
        pass

    all_comps = ComponentMeta.all_components()

    assert len(all_comps) == 3
    assert Comp1 in all_comps
    assert Comp2 in all_comps
    assert Comp3 in all_comps


def test_component_count():
    """Test component_count returns correct count."""

    assert ComponentMeta.component_count() == 0

    class Comp1(metaclass=ComponentMeta):
        pass

    assert ComponentMeta.component_count() == 1

    class Comp2(metaclass=ComponentMeta):
        pass

    assert ComponentMeta.component_count() == 2


def test_clear_registry():
    """Test that clear_registry clears all components."""

    class Comp1(metaclass=ComponentMeta):
        pass

    class Comp2(metaclass=ComponentMeta):
        pass

    assert ComponentMeta.component_count() == 2

    ComponentMeta.clear_registry()

    assert ComponentMeta.component_count() == 0
    assert ComponentMeta.all_components() == []


def test_clear_registry_resets_id():
    """Test that clear_registry resets ID counter to 1."""

    class Comp1(metaclass=ComponentMeta):
        pass

    assert Comp1._component_id == 1

    ComponentMeta.clear_registry()

    class Comp2(metaclass=ComponentMeta):
        pass

    # ID should restart from 1
    assert Comp2._component_id == 1


def test_inheritance_gets_new_id():
    """Test that subclasses get their own unique ID."""

    class BaseComponent(metaclass=ComponentMeta):
        x: int

    class DerivedComponent(BaseComponent):
        y: int

    assert BaseComponent._component_id != DerivedComponent._component_id
    assert BaseComponent._component_id == 1
    assert DerivedComponent._component_id == 2


def test_inheritance_inherits_fields():
    """Test that subclasses inherit parent fields."""

    class BaseComponent(metaclass=ComponentMeta):
        x: int

    class DerivedComponent(BaseComponent):
        y: int

    # Derived should have both x and y
    assert "x" in DerivedComponent._field_types
    assert "y" in DerivedComponent._field_types


def test_duplicate_name_different_module():
    """Test components with same name in different modules are distinct."""

    # Simulate different modules by creating with different __module__
    class Comp1(metaclass=ComponentMeta):
        pass

    # Second component with same class name but will have different qualified name
    Comp2 = ComponentMeta("Comp1", (), {"__module__": "other_module"})

    # Should be different IDs
    assert Comp1._component_id != Comp2._component_id

    # Both should be in registry
    assert ComponentMeta.component_count() == 2


def test_base_component_class_skipped():
    """Test that base Component class itself is skipped from registration."""

    # Create a class literally named "Component"
    class Component(metaclass=ComponentMeta):
        pass

    # Should not be registered
    assert ComponentMeta.component_count() == 0


def test_field_descriptors_storage():
    """Test that all field descriptors are stored correctly."""

    class TestComponent(metaclass=ComponentMeta):
        a: int
        b: float
        c: str

    assert len(TestComponent._field_descriptors) == 3
    assert "a" in TestComponent._field_descriptors
    assert "b" in TestComponent._field_descriptors
    assert "c" in TestComponent._field_descriptors


def test_no_fields_component():
    """Test component with no fields is valid."""

    class EmptyComponent(metaclass=ComponentMeta):
        pass

    assert EmptyComponent._field_types == {}
    assert EmptyComponent._field_offsets == {}
    assert EmptyComponent._field_defaults == {}
    assert EmptyComponent._component_id == 1


def test_immutable_defaults_allowed():
    """Test that immutable defaults (int, float, str, None, tuple) are allowed."""

    class TestComponent(metaclass=ComponentMeta):
        int_val: int = 42
        float_val: float = 3.14
        str_val: str = "test"
        none_val: int = None
        tuple_val: tuple = (1, 2, 3)

    # Should not raise any errors
    assert TestComponent._field_defaults["int_val"] == 42
    assert TestComponent._field_defaults["float_val"] == 3.14
    assert TestComponent._field_defaults["str_val"] == "test"
    assert TestComponent._field_defaults["none_val"] is None
    assert TestComponent._field_defaults["tuple_val"] == (1, 2, 3)


def test_registry_thread_safety():
    """Test that component registration is thread-safe."""
    import threading

    components = []
    errors = []

    def create_component(index):
        try:
            cls = ComponentMeta(f"ThreadComp{index}", (), {"x": int})
            components.append(cls)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=create_component, args=(i,)) for i in range(50)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(components) == 50

    # All should have unique IDs
    ids = [c._component_id for c in components]
    assert len(ids) == len(set(ids))


def test_field_types_dict_exists():
    """Test that _field_types dict is always created."""

    class TestComponent(metaclass=ComponentMeta):
        pass

    assert hasattr(TestComponent, "_field_types")
    assert isinstance(TestComponent._field_types, dict)


# =============================================================================
# POOL & BUDGET TESTS (Edge Cases)
# =============================================================================


def test_pool_allocation_basic():
    """Test basic pool allocation and reuse."""

    class PooledComponent(metaclass=ComponentMeta):
        x: int = 0

    # Configure pooling
    PooledComponent._pooled_config = {"max_size": 5}
    PooledComponent._pool = []

    # Create instance normally (pool empty)
    inst1 = PooledComponent()
    inst1.x = 42

    # Return to pool
    PooledComponent.return_to_pool(inst1)
    assert len(PooledComponent._pool) == 1

    # Create new instance (should reuse from pool)
    inst2 = PooledComponent()
    # Should be same object reinitialized
    assert inst2 is inst1


def test_pool_exhaustion():
    """Test creating more instances than max_size pool."""

    class PooledComponent(metaclass=ComponentMeta):
        value: int = 0

    # Configure pooling with small max
    PooledComponent._pooled_config = {"max_size": 2}
    PooledComponent._pool = []

    # Create 3 instances (more than max_size)
    inst1 = PooledComponent()
    inst2 = PooledComponent()
    inst3 = PooledComponent()

    # All should succeed (pool doesn't limit creation, only reuse)
    assert inst1 is not None
    assert inst2 is not None
    assert inst3 is not None


def test_pool_return_when_full():
    """Test returning instances when pool is at max_size."""

    class PooledComponent(metaclass=ComponentMeta):
        x: int = 0

    # Configure pooling
    PooledComponent._pooled_config = {"max_size": 2}
    PooledComponent._pool = []

    # Create 3 instances
    instances = [PooledComponent() for _ in range(3)]

    # Return all 3 (but max_size is 2)
    for inst in instances:
        PooledComponent.return_to_pool(inst)

    # Pool should be capped at max_size
    assert len(PooledComponent._pool) == 2


def test_pool_empty_allocation():
    """Test that empty pool falls through to normal creation."""

    class PooledComponent(metaclass=ComponentMeta):
        x: int = 10

    # Configure pooling with empty pool
    PooledComponent._pooled_config = {"max_size": 5}
    PooledComponent._pool = []

    # Create instance (pool empty, should create new)
    inst = PooledComponent()
    assert inst is not None
    assert inst.x == 10


def test_pool_stats_when_disabled():
    """Test pool_stats returns correct schema when pooling disabled."""

    class NonPooledComponent(metaclass=ComponentMeta):
        x: int

    stats = NonPooledComponent.pool_stats()

    # Should have consistent schema
    assert stats["enabled"] is False
    assert stats["available"] == 0
    assert stats["max_size"] is None
    assert stats["config"] is None


def test_pool_stats_when_enabled():
    """Test pool_stats returns correct values when pooling enabled."""

    class PooledComponent(metaclass=ComponentMeta):
        x: int

    # Configure pooling
    PooledComponent._pooled_config = {"max_size": 10, "custom_key": "value"}
    PooledComponent._pool = [object(), object(), object()]

    stats = PooledComponent.pool_stats()

    assert stats["enabled"] is True
    assert stats["available"] == 3
    assert stats["max_size"] == 10
    assert stats["config"]["custom_key"] == "value"


def test_budget_enforcement_at_limit():
    """Test that budget enforcement raises error at max_instances."""

    class BudgetedComponent(metaclass=ComponentMeta):
        x: int

    # Configure budget
    BudgetedComponent._budget_config = {"max_instances": 3}
    BudgetedComponent._instance_count = 0

    # Create up to limit
    inst1 = BudgetedComponent()
    inst2 = BudgetedComponent()
    inst3 = BudgetedComponent()

    assert BudgetedComponent._instance_count == 3

    # 4th should raise
    with pytest.raises(RuntimeError, match="Budget exceeded"):
        BudgetedComponent()


def test_budget_at_zero_limit():
    """Test budget with max_instances=0 (edge case)."""

    class BudgetedComponent(metaclass=ComponentMeta):
        x: int

    # Configure budget at 0
    BudgetedComponent._budget_config = {"max_instances": 0}
    BudgetedComponent._instance_count = 0

    # First creation should fail
    with pytest.raises(RuntimeError, match="Budget exceeded"):
        BudgetedComponent()


def test_budget_decrement_on_pool_return():
    """Test that returning to pool decrements budget counter."""

    class BudgetedPooledComponent(metaclass=ComponentMeta):
        x: int

    # Configure both budget and pool
    BudgetedPooledComponent._budget_config = {"max_instances": 5}
    BudgetedPooledComponent._pooled_config = {"max_size": 5}
    BudgetedPooledComponent._instance_count = 0
    BudgetedPooledComponent._pool = []

    # Create instance (increments budget)
    inst = BudgetedPooledComponent()
    assert BudgetedPooledComponent._instance_count == 1

    # Return to pool (should decrement budget)
    BudgetedPooledComponent.return_to_pool(inst)
    assert BudgetedPooledComponent._instance_count == 0


def test_budget_double_free_warning():
    """Test that double-free (decrement below 0) triggers warning."""
    import warnings as warn_module

    class BudgetedComponent(metaclass=ComponentMeta):
        x: int

    # Configure budget
    BudgetedComponent._budget_config = {"max_instances": 5}
    BudgetedComponent._pooled_config = {"max_size": 5}
    BudgetedComponent._instance_count = 0
    BudgetedComponent._pool = []

    inst = BudgetedComponent()
    assert BudgetedComponent._instance_count == 1

    # Return once (OK)
    BudgetedComponent.return_to_pool(inst)
    assert BudgetedComponent._instance_count == 0

    # Return again (should warn)
    with pytest.warns(RuntimeWarning, match="double-free"):
        BudgetedComponent.return_to_pool(inst)


def test_instance_count_when_not_budgeted():
    """Test instance_count returns 0 when budget tracking disabled."""

    class NonBudgetedComponent(metaclass=ComponentMeta):
        x: int

    count = NonBudgetedComponent.instance_count()
    assert count == 0


def test_instance_count_when_budgeted():
    """Test instance_count returns accurate count when budgeted."""

    class BudgetedComponent(metaclass=ComponentMeta):
        x: int

    BudgetedComponent._budget_config = {"max_instances": 10}
    BudgetedComponent._instance_count = 0

    assert BudgetedComponent.instance_count() == 0

    inst1 = BudgetedComponent()
    assert BudgetedComponent.instance_count() == 1

    inst2 = BudgetedComponent()
    assert BudgetedComponent.instance_count() == 2


# =============================================================================
# LAYOUT OPTIMIZATION TESTS (Edge Cases)
# =============================================================================


def test_layout_mode_when_not_packed():
    """Test get_layout_mode returns 'aos' when not packed."""

    class RegularComponent(metaclass=ComponentMeta):
        x: int

    assert RegularComponent.get_layout_mode() == "aos"


def test_layout_mode_when_packed():
    """Test get_layout_mode returns 'soa' when packed."""

    class PackedComponent(metaclass=ComponentMeta):
        x: int

    PackedComponent._packed_layout = True

    assert PackedComponent.get_layout_mode() == "soa"


def test_layout_arrays_empty_instances():
    """Test get_layout_arrays with empty instance list."""

    class PackedComponent(metaclass=ComponentMeta):
        x: int
        y: float

    PackedComponent._packed_layout = True

    arrays = PackedComponent.get_layout_arrays([])
    assert arrays == {}


def test_layout_arrays_no_fields():
    """Test get_layout_arrays with component that has no fields."""

    class EmptyPackedComponent(metaclass=ComponentMeta):
        pass

    EmptyPackedComponent._packed_layout = True

    # Create instances (no fields to extract)
    instances = [EmptyPackedComponent(), EmptyPackedComponent()]

    arrays = EmptyPackedComponent.get_layout_arrays(instances)
    assert arrays == {}


def test_layout_arrays_not_packed():
    """Test get_layout_arrays returns empty dict when not packed."""

    class RegularComponent(metaclass=ComponentMeta):
        x: int
        y: float

    # Not packed
    instances = [RegularComponent(), RegularComponent()]

    arrays = RegularComponent.get_layout_arrays(instances)
    assert arrays == {}


def test_layout_arrays_extracts_correctly():
    """Test get_layout_arrays correctly extracts SoA structure."""

    class PackedComponent(metaclass=ComponentMeta):
        x: int
        y: float

    PackedComponent._packed_layout = True

    # Create test instances (need to manually set since no __init__)
    inst1 = object.__new__(PackedComponent)
    inst1.x = 10
    inst1.y = 1.5

    inst2 = object.__new__(PackedComponent)
    inst2.x = 20
    inst2.y = 2.5

    inst3 = object.__new__(PackedComponent)
    inst3.x = 30
    inst3.y = 3.5

    arrays = PackedComponent.get_layout_arrays([inst1, inst2, inst3])

    assert arrays["x"] == [10, 20, 30]
    assert arrays["y"] == [1.5, 2.5, 3.5]


# =============================================================================
# THREAD SAFETY TESTS (Pool & Budget)
# =============================================================================


def test_pool_thread_safety():
    """Test concurrent pool allocation and return."""
    import threading

    class ThreadPoolComponent(metaclass=ComponentMeta):
        x: int = 0

    ThreadPoolComponent._pooled_config = {"max_size": 50}
    ThreadPoolComponent._pool = []

    instances_created = []
    errors = []

    def worker():
        try:
            # Create instance
            inst = ThreadPoolComponent()
            instances_created.append(inst)
            # Immediately return
            ThreadPoolComponent.return_to_pool(inst)
        except Exception as e:
            errors.append(e)

    # Run 100 concurrent workers
    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have no errors
    assert len(errors) == 0
    # Should have created 100 instances
    assert len(instances_created) == 100
    # Pool size should not exceed max_size
    assert len(ThreadPoolComponent._pool) <= 50


def test_budget_thread_safety_at_limit():
    """Test concurrent instance creation at budget limit."""
    import threading

    class ThreadBudgetComponent(metaclass=ComponentMeta):
        x: int

    ThreadBudgetComponent._budget_config = {"max_instances": 10}
    ThreadBudgetComponent._instance_count = 0

    successes = []
    errors = []

    def worker():
        try:
            inst = ThreadBudgetComponent()
            successes.append(inst)
        except RuntimeError as e:
            errors.append(e)

    # Try to create 20 instances (only 10 should succeed)
    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 10 should succeed
    assert len(successes) == 10
    # Exactly 10 should fail
    assert len(errors) == 10
    # All errors should be budget exceeded
    assert all("Budget exceeded" in str(e) for e in errors)
