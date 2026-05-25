"""
Tests for read-tracking in descriptors.

Verifies:
- Computation protocol
- Read recording during computation context
- No recording outside computation context
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from trinity.descriptors.base import (
    BaseDescriptor,
    Computation,
    set_current_computation,
    get_current_computation,
)


class MockComputation:
    """Mock computation that records reads."""

    def __init__(self):
        self.reads: list[tuple] = []

    def record_read(self, obj, field_name: str) -> None:
        self.reads.append((id(obj), field_name))


@pytest.fixture
def descriptor():
    """Create a descriptor for testing."""
    desc = BaseDescriptor[int](field_type=int)
    desc._name = "value"
    return desc


@pytest.fixture
def obj():
    """Create an object for testing."""
    class TestObj:
        pass
    return TestObj()


@pytest.fixture(autouse=True)
def reset_computation():
    """Reset computation context between tests."""
    set_current_computation(None)
    yield
    set_current_computation(None)


class TestComputationProtocol:
    """Test the Computation protocol."""

    def test_mock_implements_protocol(self):
        """MockComputation should implement Computation protocol."""
        comp = MockComputation()
        # Should have record_read method
        assert hasattr(comp, 'record_read')
        assert callable(comp.record_read)

    def test_record_read_captures_info(self):
        """record_read should capture object and field name."""
        comp = MockComputation()
        obj = object()
        comp.record_read(obj, "field_name")

        assert len(comp.reads) == 1
        assert comp.reads[0][0] == id(obj)
        assert comp.reads[0][1] == "field_name"


class TestComputationContext:
    """Test computation context management."""

    def test_get_returns_none_by_default(self):
        """get_current_computation should return None when not set."""
        assert get_current_computation() is None

    def test_set_and_get(self):
        """Setting computation should be retrievable."""
        comp = MockComputation()
        set_current_computation(comp)

        assert get_current_computation() is comp

    def test_set_to_none(self):
        """Setting to None should clear the computation."""
        comp = MockComputation()
        set_current_computation(comp)
        set_current_computation(None)

        assert get_current_computation() is None


class TestDescriptorReadTracking:
    """Test descriptor read tracking."""

    def test_read_recorded_in_computation(self, descriptor, obj):
        """Reads should be recorded when inside a computation."""
        obj.value = 42
        comp = MockComputation()
        set_current_computation(comp)

        _ = descriptor.__get__(obj, type(obj))

        assert len(comp.reads) == 1
        assert comp.reads[0][0] == id(obj)
        assert comp.reads[0][1] == "value"

    def test_no_record_outside_computation(self, descriptor, obj):
        """Reads should not be recorded outside a computation."""
        obj.value = 42

        _ = descriptor.__get__(obj, type(obj))

        # No computation set, so no recording
        assert get_current_computation() is None

    def test_multiple_reads_recorded(self, descriptor, obj):
        """Multiple reads should all be recorded."""
        obj.value = 42
        comp = MockComputation()
        set_current_computation(comp)

        _ = descriptor.__get__(obj, type(obj))
        _ = descriptor.__get__(obj, type(obj))
        _ = descriptor.__get__(obj, type(obj))

        assert len(comp.reads) == 3

    def test_different_objects_recorded_separately(self, descriptor):
        """Reads on different objects should be recorded separately."""
        class TestObj:
            pass

        obj1 = TestObj()
        obj2 = TestObj()
        obj1.value = 1
        obj2.value = 2

        comp = MockComputation()
        set_current_computation(comp)

        _ = descriptor.__get__(obj1, type(obj1))
        _ = descriptor.__get__(obj2, type(obj2))

        assert len(comp.reads) == 2
        assert comp.reads[0][0] == id(obj1)
        assert comp.reads[1][0] == id(obj2)

    def test_class_level_access_not_recorded(self, descriptor):
        """Class-level access (obj=None) should not record reads."""
        comp = MockComputation()
        set_current_computation(comp)

        class TestClass:
            pass

        # Class-level access returns descriptor
        result = descriptor.__get__(None, TestClass)

        assert result is descriptor
        assert len(comp.reads) == 0


class TestReadTrackingWithRealDescriptor:
    """Test read tracking with a complete class setup."""

    def test_full_class_read_tracking(self):
        """Read tracking works with a real class using descriptors."""

        class TrackedValue(BaseDescriptor[int]):
            descriptor_id = "tracked_value"

        class Entity:
            x = TrackedValue(field_type=int)
            y = TrackedValue(field_type=int)

        # Set names manually (normally done by metaclass)
        Entity.x._name = "x"
        Entity.y._name = "y"

        entity = Entity()
        entity.x = 10
        entity.y = 20

        comp = MockComputation()
        set_current_computation(comp)

        # Access both fields
        _ = entity.x
        _ = entity.y
        _ = entity.x  # Access x again

        assert len(comp.reads) == 3
        # Verify field names recorded
        field_names = [r[1] for r in comp.reads]
        assert field_names == ["x", "y", "x"]


class TestProvenanceIntegration:
    """Test integration between Trinity descriptors and foundation provenance."""

    def test_descriptor_reads_captured_in_provenance(self):
        """Descriptor reads should be captured in provenance tracking."""
        from foundation.provenance import (
            track_provenance,
            provenance,
            clear_provenance,
        )

        clear_provenance()

        class TrackedValue(BaseDescriptor[int]):
            descriptor_id = "tracked_value"

        class Entity:
            x = TrackedValue(field_type=int)
            y = TrackedValue(field_type=int)

            @track_provenance
            def sum_values(self):
                return self.x + self.y

        # Set names manually
        Entity.x._name = "x"
        Entity.y._name = "y"

        entity = Entity()
        entity.x = 10
        entity.y = 20

        result = entity.sum_values()

        assert result == 30
        prov = provenance(entity, "sum_values")
        assert prov is not None
        assert prov.value == 30

        # Check that reads were captured
        assert len(prov.reads) == 2
        read_fields = [r.field for r in prov.reads]
        assert "x" in read_fields
        assert "y" in read_fields

        # Check values were captured
        read_values = {r.field: r.value for r in prov.reads}
        assert read_values["x"] == 10
        assert read_values["y"] == 20

        clear_provenance()

    def test_derivation_tree_with_descriptors(self):
        """derivation_tree should work with descriptor reads."""
        from foundation.provenance import (
            track_provenance,
            derivation_tree,
            clear_provenance,
        )

        clear_provenance()

        class TrackedValue(BaseDescriptor[int]):
            descriptor_id = "tracked_value"

        class Entity:
            health = TrackedValue(field_type=int)
            armor = TrackedValue(field_type=int)

            @track_provenance
            def effective_health(self):
                return self.health + self.armor * 2

        Entity.health._name = "health"
        Entity.armor._name = "armor"

        entity = Entity()
        entity.health = 100
        entity.armor = 50

        result = entity.effective_health()
        assert result == 200

        tree = derivation_tree(entity, "effective_health")
        assert tree is not None
        assert tree.field == "effective_health"
        assert tree.value == 200
        assert len(tree.children) == 2

        child_fields = [c.field for c in tree.children]
        assert "health" in child_fields
        assert "armor" in child_fields

        clear_provenance()

    def test_nested_provenance_with_descriptors(self):
        """Nested provenance tracking with descriptors."""
        from foundation.provenance import (
            track_provenance,
            provenance,
            clear_provenance,
        )

        clear_provenance()

        class TrackedValue(BaseDescriptor[int]):
            descriptor_id = "tracked_value"

        class Entity:
            base = TrackedValue(field_type=int)
            multiplier = TrackedValue(field_type=int)

            @track_provenance
            def scaled(self):
                return self.base * self.multiplier

            @track_provenance
            def doubled_scaled(self):
                return self.scaled() * 2

        Entity.base._name = "base"
        Entity.multiplier._name = "multiplier"

        entity = Entity()
        entity.base = 10
        entity.multiplier = 3

        result = entity.doubled_scaled()
        assert result == 60

        # Inner computation should have its own reads
        scaled_prov = provenance(entity, "scaled")
        assert len(scaled_prov.reads) == 2

        # Outer computation should have no descriptor reads
        # (it only called scaled(), which is a method)
        doubled_prov = provenance(entity, "doubled_scaled")
        assert len(doubled_prov.reads) == 0

        clear_provenance()
