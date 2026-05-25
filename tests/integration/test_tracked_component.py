"""
Prime Integration Test: Tracked Component Flow

This test exercises the full decorator → metaclass → descriptor pipeline
for a tracked component. It verifies:

1. DECORATOR LAYER
   - @component decorator marks the class correctly
   - _track_changes attribute is set (manually, simulating what a @tracked decorator would do)

2. METACLASS LAYER
   - ComponentMeta.__new__ processes the class
   - _process_fields() collects field annotations and offsets
   - _install_descriptors() composes the descriptor chain
   - Component is registered with unique ID

3. DESCRIPTOR LAYER
   - TrackedDescriptor wraps StorageDescriptor
   - __set_name__ binds descriptors to field names
   - __get__ retrieves values correctly
   - __set__ stores values and triggers post_set hook
   - post_set marks fields as dirty only on actual change

4. INTEGRATION BEHAVIORS
   - Dirty tracking only fires when values actually change
   - Multiple fields tracked independently
   - clear_dirty resets tracking state
   - Descriptor chain order is correct (tracked wraps storage)
"""

import pytest

from trinity import Component, ComponentMeta
from trinity.descriptors import StorageDescriptor, TrackedDescriptor
from trinity.descriptors.tracking import clear_dirty, get_dirty_fields, is_dirty


class TestTrackedComponentFlow:
    """
    Integration tests for tracked components.

    These tests define classes dynamically inside test methods to ensure
    metaclass code runs fresh for each test (not at import time).
    """

    # =========================================================================
    # DECORATOR + METACLASS: Class Creation
    # =========================================================================

    def test_component_gets_unique_id(self):
        """ComponentMeta assigns unique IDs to each component class."""

        class Position(Component):
            x: float = 0.0
            y: float = 0.0

        class Velocity(Component):
            dx: float = 0.0
            dy: float = 0.0

        assert hasattr(Position, "_component_id")
        assert hasattr(Velocity, "_component_id")
        assert Position._component_id != Velocity._component_id

    def test_component_registered_in_metaclass(self, component_meta):
        """Components are registered and retrievable by ID."""

        class Health(Component):
            current: float = 100.0
            maximum: float = 100.0

        retrieved = component_meta.get_by_id(Health._component_id)
        assert retrieved is Health

    def test_field_types_collected(self):
        """ComponentMeta._process_fields collects field annotations."""

        class Transform(Component):
            x: float = 0.0
            y: float = 0.0
            rotation: float = 0.0

        assert Transform._field_types == {
            "x": float,
            "y": float,
            "rotation": float,
        }

    def test_field_offsets_assigned(self):
        """Each field gets a unique offset for bitmask tracking."""

        class Data(Component):
            a: int = 0
            b: int = 0
            c: int = 0

        offsets = Data._field_offsets
        assert set(offsets.keys()) == {"a", "b", "c"}
        # Offsets should be unique integers
        assert len(set(offsets.values())) == 3

    def test_field_defaults_captured(self):
        """Default values are captured for descriptor initialization."""

        class Defaults(Component):
            with_default: float = 42.0
            another: str = "hello"

        assert Defaults._field_defaults["with_default"] == 42.0
        assert Defaults._field_defaults["another"] == "hello"

    def test_private_fields_skipped(self):
        """Fields starting with underscore are not processed."""

        class Private(Component):
            public: int = 1
            _private: int = 2

        assert "public" in Private._field_types
        assert "_private" not in Private._field_types

    def test_mutable_default_rejected(self):
        """Mutable defaults (list, dict, set) raise TypeError."""
        with pytest.raises(TypeError, match="Mutable default"):

            class BadComponent(Component):
                items: list = []

    # =========================================================================
    # DESCRIPTOR INSTALLATION: Chain Composition
    # =========================================================================

    def test_descriptors_installed_on_fields(self):
        """ComponentMeta installs descriptors on each field."""

        class Simple(Component):
            value: int = 0

        descriptor = Simple.__dict__["value"]
        assert hasattr(descriptor, "__get__")
        assert hasattr(descriptor, "__set__")

    def test_tracked_component_gets_tracked_descriptor(self):
        """When _track_changes=True, TrackedDescriptor is installed."""

        class Tracked(Component):
            _track_changes = True
            x: float = 0.0
            y: float = 0.0

        descriptor = Tracked.__dict__["x"]
        # The outermost descriptor should be TrackedDescriptor
        assert isinstance(descriptor, TrackedDescriptor)

    def test_descriptor_chain_structure(self):
        """Tracked descriptor wraps storage descriptor."""

        class Tracked(Component):
            _track_changes = True
            value: int = 0

        descriptor = Tracked.__dict__["value"]

        # Should be TrackedDescriptor
        assert descriptor.descriptor_id == "tracked"

        # Inner should be StorageDescriptor
        assert descriptor._inner is not None
        assert descriptor._inner.descriptor_id == "storage"

    def test_descriptor_set_name_called(self):
        """__set_name__ binds descriptor to field name."""

        class Named(Component):
            _track_changes = True
            my_field: int = 0

        descriptor = Named.__dict__["my_field"]
        assert descriptor._name == "my_field"
        assert descriptor._owner is Named

    # =========================================================================
    # DESCRIPTOR BEHAVIOR: Get/Set
    # =========================================================================

    def test_get_returns_default_on_fresh_instance(self):
        """First access returns the default value."""

        class WithDefault(Component):
            value: int = 42

        obj = WithDefault()
        assert obj.value == 42

    def test_set_stores_value(self):
        """Setting a value persists it."""

        class Settable(Component):
            value: int = 0

        obj = Settable()
        obj.value = 99
        assert obj.value == 99

    def test_set_via_init_kwargs(self):
        """Values can be set via __init__ kwargs."""

        class Initable(Component):
            x: float = 0.0
            y: float = 0.0

        obj = Initable(x=10.0, y=20.0)
        assert obj.x == 10.0
        assert obj.y == 20.0

    def test_invalid_kwarg_raises(self):
        """Unknown kwargs raise TypeError."""

        class Strict(Component):
            known: int = 0

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            Strict(unknown=1)

    # =========================================================================
    # TRACKED DESCRIPTOR: Dirty Tracking
    # =========================================================================

    def test_dirty_flag_not_set_initially(self):
        """Fresh instance has no dirty fields."""

        class Tracked(Component):
            _track_changes = True
            x: float = 0.0

        obj = Tracked()
        assert not is_dirty(obj, "x")
        assert get_dirty_fields(obj) == set()

    def test_setting_value_marks_dirty(self):
        """Changing a value marks the field as dirty."""

        class Tracked(Component):
            _track_changes = True
            x: float = 0.0

        obj = Tracked()
        obj.x = 10.0

        assert is_dirty(obj, "x")
        assert "x" in get_dirty_fields(obj)

    def test_same_value_does_not_mark_dirty(self):
        """Setting the same value does NOT mark dirty."""

        class Tracked(Component):
            _track_changes = True
            x: float = 5.0

        obj = Tracked()
        # First, access to initialize
        _ = obj.x
        # Set to same value
        obj.x = 5.0

        # Should NOT be dirty (value didn't change)
        assert not is_dirty(obj, "x")

    def test_multiple_fields_tracked_independently(self):
        """Each field's dirty state is independent."""

        class Multi(Component):
            _track_changes = True
            a: int = 0
            b: int = 0
            c: int = 0

        obj = Multi()
        obj.a = 1  # Change a
        obj.c = 3  # Change c, leave b alone

        assert is_dirty(obj, "a")
        assert not is_dirty(obj, "b")
        assert is_dirty(obj, "c")
        assert get_dirty_fields(obj) == {"a", "c"}

    def test_clear_dirty_resets_all(self):
        """clear_dirty removes all dirty flags."""

        class Clearable(Component):
            _track_changes = True
            x: float = 0.0
            y: float = 0.0

        obj = Clearable()
        obj.x = 1.0
        obj.y = 2.0
        assert get_dirty_fields(obj) == {"x", "y"}

        clear_dirty(obj)
        assert get_dirty_fields(obj) == set()

    def test_dirty_after_clear_and_modify(self):
        """After clearing, modifications mark dirty again."""

        class Recleanable(Component):
            _track_changes = True
            value: int = 0

        obj = Recleanable()
        obj.value = 1
        clear_dirty(obj)
        assert not is_dirty(obj, "value")

        obj.value = 2
        assert is_dirty(obj, "value")

    # =========================================================================
    # INTEGRATION: Full Pipeline
    # =========================================================================

    def test_full_tracked_component_lifecycle(self):
        """
        End-to-end test of the full pipeline:
        1. Define component with tracking
        2. Verify metaclass processed it
        3. Create instance
        4. Modify fields
        5. Verify dirty tracking
        6. Clear and re-modify
        """

        # 1. Define component
        class Position(Component):
            _track_changes = True
            x: float = 0.0
            y: float = 0.0
            z: float = 0.0

        # 2. Metaclass processed correctly
        assert hasattr(Position, "_component_id")
        assert Position._field_types == {"x": float, "y": float, "z": float}
        assert all(
            isinstance(Position.__dict__[f], TrackedDescriptor) for f in ("x", "y", "z")
        )

        # 3. Create instance with kwargs
        # Note: kwargs passed to __init__ also trigger __set__, marking dirty
        pos = Position(x=1.0, y=2.0)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 0.0  # default

        # Init kwargs mark those fields dirty (they went through __set__)
        assert is_dirty(pos, "x")
        assert is_dirty(pos, "y")
        assert not is_dirty(pos, "z")  # z was not set, just accessed default

        # 4. Clear and start fresh
        clear_dirty(pos)
        assert get_dirty_fields(pos) == set()

        # 5. Modify specific fields
        pos.x = 10.0
        pos.z = 5.0

        # 6. Verify dirty tracking
        assert is_dirty(pos, "x")
        assert not is_dirty(pos, "y")  # not modified since clear
        assert is_dirty(pos, "z")
        assert get_dirty_fields(pos) == {"x", "z"}

        # 7. Clear and re-modify
        clear_dirty(pos)
        assert get_dirty_fields(pos) == set()

        pos.y = 100.0
        assert get_dirty_fields(pos) == {"y"}

    def test_multiple_instances_independent(self):
        """Each instance has independent dirty state."""

        class Shared(Component):
            _track_changes = True
            value: int = 0

        a = Shared()
        b = Shared()

        a.value = 1
        # Only 'a' should be dirty
        assert is_dirty(a, "value")
        assert not is_dirty(b, "value")

        b.value = 2
        # Now both dirty
        assert is_dirty(a, "value")
        assert is_dirty(b, "value")

        clear_dirty(a)
        # Only 'a' cleared
        assert not is_dirty(a, "value")
        assert is_dirty(b, "value")

    def test_repr_shows_field_values(self):
        """Component __repr__ shows current field values."""

        class Repr(Component):
            x: int = 0
            y: int = 0

        obj = Repr(x=10, y=20)
        r = repr(obj)
        assert "Repr" in r
        assert "x=10" in r
        assert "y=20" in r


class TestDescriptorChainOrder:
    """Tests specifically for descriptor composition order."""

    def test_storage_is_innermost(self):
        """StorageDescriptor should be at the end of the chain."""

        class Tracked(Component):
            _track_changes = True
            value: int = 0

        desc = Tracked.__dict__["value"]

        # Walk the chain
        chain = []
        current = desc
        while current is not None:
            chain.append(current.descriptor_id)
            current = getattr(current, "_inner", None)

        # Should be: tracked -> storage
        assert chain == ["tracked", "storage"]

    def test_get_chain_method(self):
        """Descriptor.get_chain() returns the full chain."""

        class Tracked(Component):
            _track_changes = True
            field: float = 0.0

        desc = Tracked.__dict__["field"]
        chain = desc.get_chain()

        assert len(chain) == 2
        assert chain[0].descriptor_id == "tracked"
        assert chain[1].descriptor_id == "storage"


class TestRegistryIsolation:
    """Tests that verify registry isolation works correctly."""

    def test_fresh_ids_each_test(self, component_meta):
        """Component IDs start fresh in each test (via fixture)."""

        class First(Component):
            x: int = 0

        # In an isolated test, this should be ID 1
        # (assuming clear_registry works)
        assert First._component_id == 1

    def test_registry_empty_after_clear(self, component_meta):
        """After clear_registry, no components are registered."""

        # Define a component
        class Temp(Component):
            x: int = 0

        assert component_meta.component_count() == 1

        # Clear
        component_meta.clear_registry()

        assert component_meta.component_count() == 0
        assert component_meta.get_by_id(Temp._component_id) is None
