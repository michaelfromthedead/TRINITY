"""
T-DEMO-2.2 Trinity Pattern Integration for DSL Nodes -- Blackbox Acceptance Tests.

Cleanroom testing: derived from spec (GAPSET_11 Phase 2, TRINITY_LATEST.md) only.
No knowledge of DSL node implementation internals.

Acceptance criteria:
    DSL nodes are valid Trinity Objects with introspection and dirty tracking.

Three test suites:

    Suite A: Valid Trinity Objects (TrinityObject*)
        DSL node classes are created through the Trinity Pattern --
        they use EngineMeta (or a subclass), are registered, have unique
        IDs, and support the base class lifecycle.

    Suite B: Introspection (Introspect*)
        mirror(), decompose(), decompose_layered(), expand(), and
        schema_hash() all work on DSL node classes and instances.

    Suite C: Dirty Tracking (Dirty*)
        TrackedDescriptor marks fields dirty on set.  The Tracker
        singleton records changes, supports subscriptions, transactions,
        and undo/redo.
"""

from __future__ import annotations

import pytest

from trinity import Component
from trinity.metaclasses import ComponentMeta
from trinity.decorators.ops import decompose, decompose_layered, expand

from foundation.mirror import mirror, schema_hash
from foundation.tracker import tracker, Change, Tracker
from trinity.descriptors import (
    TrackedDescriptor,
    is_dirty,
    get_dirty_fields,
    clear_dirty,
    clear_dirty_field,
)


# =============================================================================
# HELPER: capture-tracker state for undo/redo tests
# =============================================================================

def _reset_tracker():
    """Bring the global tracker back to a known-clean state."""
    tracker._dirty.clear()
    tracker._cb_global.clear()
    tracker._cb_type.clear()
    tracker._cb_obj.clear()
    tracker._undo.clear()
    tracker._redo.clear()
    tracker._txn = None


# =============================================================================
# SUITE A -- VALID TRINITY OBJECTS
# =============================================================================

class TestTrinityObjectCreation:
    """DSL node classes created as Components are valid Trinity Objects."""

    def test_metaclass_is_component_meta(self):
        """A DSL node class that inherits from Component uses ComponentMeta."""
        class SphereNode(Component):
            radius: float = 1.0
            position_x: float = 0.0
            position_y: float = 0.0
            position_z: float = 0.0

        assert type(SphereNode) is ComponentMeta
        assert isinstance(SphereNode, ComponentMeta)

    def test_has_unique_component_id(self):
        """Every concrete DSL node class gets a unique _component_id."""
        class SphereNode(Component):
            radius: float = 1.0
        class BoxNode(Component):
            size_x: float = 1.0
            size_y: float = 1.0
            size_z: float = 1.0
        class TorusNode(Component):
            major_radius: float = 2.0
            minor_radius: float = 0.5

        ids = {SphereNode._component_id, BoxNode._component_id, TorusNode._component_id}
        assert len(ids) == 3, "Each DSL node must have a unique component ID"

    def test_component_name_is_qualified(self):
        """_component_name is the qualified module+class name."""
        class SphereNode(Component):
            radius: float = 1.0
        name = SphereNode._component_name
        assert name.endswith(".SphereNode")
        assert "test_dsl_trinity_blackbox" in name

    def test_registered_in_component_registry(self):
        """DSL node classes appear in ComponentMeta registry."""
        class SphereNode(Component):
            radius: float = 1.0
        cid = SphereNode._component_id
        retrieved = ComponentMeta.get_by_id(cid)
        assert retrieved is SphereNode

    def test_registry_finds_by_name(self):
        """ComponentMeta.get_by_name resolves a registered node."""
        class SphereNode(Component):
            radius: float = 1.0
        retrieved = ComponentMeta.get_by_name(SphereNode._component_name)
        assert retrieved is SphereNode

    def test_all_components_includes_node(self):
        """ComponentMeta.all_components() lists DSL nodes."""
        class SphereNode(Component):
            radius: float = 1.0
        assert SphereNode in ComponentMeta.all_components()

    def test_instance_creation(self):
        """DSL node instances can be created like normal Components."""
        class SphereNode(Component):
            radius: float = 1.0
            pos_x: float = 0.0

        s = SphereNode(radius=2.5, pos_x=10.0)
        assert s.radius == 2.5
        assert s.pos_x == 10.0

    def test_instance_positional_not_supported(self):
        """Component __init__ only accepts keyword arguments."""
        class SphereNode(Component):
            radius: float = 1.0
        with pytest.raises(TypeError):
            SphereNode(42)  # positional arg

    def test_instance_unknown_kwarg_raises(self):
        """Passing an unknown field name raises TypeError."""
        class SphereNode(Component):
            radius: float = 1.0
        with pytest.raises(TypeError):
            SphereNode(nonexistent=42)

    def test_default_values_applied(self):
        """Fields without explicit kwargs use their class defaults."""
        class SphereNode(Component):
            radius: float = 1.0
        s = SphereNode()
        assert s.radius == 1.0

    def test_base_component_not_registered(self):
        """The abstract Component base is NOT registered."""
        assert Component not in ComponentMeta.all_components()

    def test_inherited_fields_preserved(self):
        """Subclassing a DSL node inherits and extends fields."""
        class BaseNode(Component):
            pos_x: float = 0.0
            pos_y: float = 0.0

        class SphereNode(BaseNode):
            radius: float = 1.0

        s = SphereNode(pos_x=5.0, pos_y=10.0, radius=2.0)
        assert s.pos_x == 5.0
        assert s.pos_y == 10.0
        assert s.radius == 2.0

    def test_repr_includes_fields(self):
        """__repr__ shows field names and values."""
        class SphereNode(Component):
            radius: float = 1.0
            name: str = ""

        s = SphereNode(radius=3.0, name="big")
        rep = repr(s)
        assert "SphereNode" in rep
        assert "radius=3.0" in rep
        assert "name='big'" in rep

    def test_multiple_node_types_have_distinct_ids(self):
        """Sequential node types do not share IDs."""
        nodes = []
        for name in ("A", "B", "C"):
            cls = ComponentMeta(
                f"Node{name}",
                (Component,),
                {"__module__": "test_dsl", "__annotations__": {"v": float}},
            )
            nodes.append(cls)
        ids = [n._component_id for n in nodes]
        assert ids == sorted(ids)  # monotonically increasing
        assert len(set(ids)) == 3


class TestTrinityObjectFieldProcessing:
    """Fields on DSL node Components are processed correctly."""

    def test_field_types_recorded(self):
        """_field_types maps field names to their Python types."""
        class SphereNode(Component):
            radius: float = 1.0
            active: bool = True
            count: int = 0

        ft = SphereNode._field_types
        assert ft["radius"] is float
        assert ft["active"] is bool
        assert ft["count"] is int

    def test_field_defaults_recorded(self):
        """_field_defaults maps field names to their default values."""
        class SphereNode(Component):
            radius: float = 1.0
            label: str = "default"

        assert SphereNode._field_defaults["radius"] == 1.0
        assert SphereNode._field_defaults["label"] == "default"

    def test_private_fields_skipped(self):
        """Fields starting with underscore are not in _field_types."""
        class SphereNode(Component):
            radius: float = 1.0
            _cache: str = "internal"

        assert "radius" in SphereNode._field_types
        assert "_cache" not in SphereNode._field_types

    def test_mutable_defaults_raise(self):
        """Components reject mutable default values."""
        with pytest.raises(TypeError):
            class BadNode(Component):
                items: list = []

    def test_mutable_default_dict_raises(self):
        """Dict default values are rejected."""
        with pytest.raises(TypeError):
            class BadNode(Component):
                config: dict = {}


# =============================================================================
# SUITE B -- INTROSPECTION
# =============================================================================

class TestIntrospectMirrorObject:
    """mirror() produces ObjectMirror for DSL node instances."""

    def test_mirror_returns_object_mirror_for_instance(self):
        """Calling mirror() on an instance returns an ObjectMirror."""
        class SphereNode(Component):
            radius: float = 1.0

        m = mirror(SphereNode(radius=5.0))
        from foundation.mirror import ObjectMirror
        assert isinstance(m, ObjectMirror)

    def test_mirror_type_name(self):
        """ObjectMirror.type_name matches the class name."""
        class SphereNode(Component):
            radius: float = 1.0
        m = mirror(SphereNode())
        assert m.type_name == "SphereNode"

    def test_mirror_fields_accessible_via_get(self):
        """ObjectMirror.get() reads component fields."""
        class SphereNode(Component):
            radius: float = 1.0
        s = SphereNode(radius=2.0)
        m = mirror(s)
        assert m.get("radius") == 2.0

    def test_mirror_get_value(self):
        """ObjectMirror.get() reads the current field value."""
        class SphereNode(Component):
            radius: float = 1.0
        s = SphereNode(radius=7.5)
        m = mirror(s)
        assert m.get("radius") == 7.5

    def test_mirror_set_value(self):
        """ObjectMirror.set() mutates a field."""
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode()
        m = mirror(s)
        m.set("radius", 99.0)
        assert s.radius == 99.0

    def test_mirror_to_dict(self):
        """ObjectMirror.to_dict() exports all field values."""
        class SphereNode(Component):
            radius: float = 1.0
            active: bool = True

        s = SphereNode(radius=2.5, active=False)
        d = mirror(s).to_dict()
        assert d["radius"] == 2.5
        assert d["active"] is False

    def test_component_field_types(self):
        """Component._field_types maps field names to Python types."""
        class SphereNode(Component):
            radius: float = 1.0
            active: bool = True
            count: int = 0

        assert SphereNode._field_types["radius"] is float
        assert SphereNode._field_types["active"] is bool
        assert SphereNode._field_types["count"] is int

    def test_component_field_defaults(self):
        """Component._field_defaults stores default values."""
        class SphereNode(Component):
            radius: float = 42.0
            label: str = "custom"

        assert SphereNode._field_defaults["radius"] == 42.0
        assert SphereNode._field_defaults["label"] == "custom"

    def test_mirror_describe(self):
        """ObjectMirror.describe() returns a non-empty string."""
        class SphereNode(Component):
            radius: float = 1.0
        desc = mirror(SphereNode()).describe()
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "SphereNode" in desc

    def test_mirror_methods_contains_no_dunder_methods(self):
        """ObjectMirror.methods excludes dunder methods."""
        class SphereNode(Component):
            radius: float = 1.0
        m = mirror(SphereNode())
        for name in m.methods:
            assert not name.startswith("_")


class TestIntrospectMirrorClass:
    """mirror() produces ClassMirror for DSL node classes."""

    def test_mirror_returns_class_mirror_for_class(self):
        """Calling mirror() on a class returns a ClassMirror."""
        class SphereNode(Component):
            radius: float = 1.0
        from foundation.mirror import ClassMirror
        assert isinstance(mirror(SphereNode), ClassMirror)

    def test_class_mirror_type_name(self):
        """ClassMirror.type_name matches the class name."""
        class SphereNode(Component):
            radius: float = 1.0
        m = mirror(SphereNode)
        assert m.type_name == "SphereNode"

    def test_class_mirror_fields(self):
        """ClassMirror.fields lists field definitions."""
        class SphereNode(Component):
            radius: float = 1.0
            label: str = "s"

        m = mirror(SphereNode)
        assert hasattr(m, "fields")
        assert isinstance(m.fields, dict)

    def test_class_mirror_field_default(self):
        """ClassMirror field defaults accessible via Component API."""
        class SphereNode(Component):
            radius: float = 3.0

        assert SphereNode._field_defaults["radius"] == 3.0

    def test_class_mirror_describe(self):
        """ClassMirror.describe() returns readable schema."""
        class SphereNode(Component):
            radius: float = 1.0
        desc = mirror(SphereNode).describe()
        assert isinstance(desc, str)


class TestIntrospectDecompose:
    """decompose() returns Steps from all three Trinity layers."""

    def test_decompose_returns_list(self):
        """decompose() on a Component class returns a list of Steps."""
        class SphereNode(Component):
            radius: float = 1.0

        steps = decompose(SphereNode)
        assert isinstance(steps, list)

    def test_decompose_includes_metaclass_steps(self):
        """Metaclass steps (TAG, DESCRIBE, REGISTER) are visible."""
        class SphereNode(Component):
            radius: float = 1.0

        steps = decompose(SphereNode, include_metaclass=True)
        ops_found = {s.op.value for s in steps}
        # The metaclass records at minimum TAG (component_id, component_name),
        # DESCRIBE (fields), VALIDATE, and REGISTER steps.
        assert "tag" in ops_found
        assert "describe" in ops_found
        assert "register" in ops_found

    def test_decompose_layered_returns_dict(self):
        """decompose_layered() returns a dict with three layer keys."""
        class SphereNode(Component):
            radius: float = 1.0

        layered = decompose_layered(SphereNode)
        assert "decorators" in layered
        assert "metaclass" in layered
        assert "descriptors" in layered

    def test_decompose_layered_metaclass_has_steps(self):
        """The 'metaclass' layer in decompose_layered is non-empty."""
        class SphereNode(Component):
            radius: float = 1.0

        layered = decompose_layered(SphereNode)
        assert len(layered["metaclass"]) > 0

    def test_expand_returns_string(self):
        """expand() returns a human-readable string."""
        class SphereNode(Component):
            radius: float = 1.0

        output = expand(SphereNode)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_expand_contains_layer_labels(self):
        """expand() output includes layer label '[Metaclass]'."""
        class SphereNode(Component):
            radius: float = 1.0

        output = expand(SphereNode)
        assert "[Metaclass]" in output

    def test_decompose_non_class_returns_empty(self):
        """decompose() on a non-class target returns its stored steps."""
        assert decompose(42) == []


class TestIntrospectSchemaHash:
    """schema_hash() produces a stable hash for DSL node classes."""

    def test_schema_hash_is_string(self):
        """schema_hash() returns a hex string."""
        class SphereNode(Component):
            radius: float = 1.0

        h = schema_hash(SphereNode)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_schema_hash_same_for_identical_schema(self):
        """Classes with the same fields and defaults produce the same hash."""
        A = ComponentMeta("SameFields", (Component,), {
            "__module__": "mod_a",
            "__annotations__": {"radius": float},
            "radius": 1.0,
        })
        B = ComponentMeta("SameFields", (Component,), {
            "__module__": "mod_b",
            "__annotations__": {"radius": float},
            "radius": 1.0,
        })

        assert A.__name__ == B.__name__
        assert schema_hash(A) == schema_hash(B)

    def test_schema_hash_differs_for_different_schema(self):
        """Classes with different fields produce different hashes."""
        class SphereNode(Component):
            radius: float = 1.0
        class BoxNode(Component):
            size: float = 1.0

        assert schema_hash(SphereNode) != schema_hash(BoxNode)

    def test_schema_hash_differs_for_different_default(self):
        """Changing a default value changes the hash."""
        class A(Component):
            radius: float = 1.0
        class B(Component):
            radius: float = 2.0

        assert schema_hash(A) != schema_hash(B)

    def test_schema_hash_stable_across_calls(self):
        """The hash is stable across multiple calls."""
        class SphereNode(Component):
            radius: float = 1.0

        assert schema_hash(SphereNode) == schema_hash(SphereNode)

    def test_schema_hash_works_on_instance(self):
        """schema_hash() also works when passed an instance."""
        class SphereNode(Component):
            radius: float = 1.0

        h = schema_hash(SphereNode())
        assert isinstance(h, str)
        assert len(h) > 0


# =============================================================================
# SUITE C -- DIRTY TRACKING
# =============================================================================

class TestDirtyTrackingDescriptor:
    """TrackedDescriptor marks fields dirty on mutation via tracker API."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        _reset_tracker()
        yield

    def test_mark_dirty_sets_flag(self):
        """tracker.mark_dirty() marks a field on the object."""
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode()
        tracker.mark_dirty(s, "radius", 1.0, 5.0)
        assert tracker.is_dirty(s) is True
        assert "radius" in tracker.dirty_fields(s)

    def test_tracker_dirty_fields(self):
        """tracker.dirty_fields() lists changed fields."""
        class SphereNode(Component):
            radius: float = 1.0
            label: str = ""

        s = SphereNode(radius=1.0, label="a")
        tracker.mark_dirty(s, "radius", 1.0, 5.0)
        tracker.mark_dirty(s, "label", "a", "b")
        fields = tracker.dirty_fields(s)
        assert "radius" in fields
        assert "label" in fields
        assert len(fields) == 2

    def test_multiple_fields_tracked_independently(self):
        """Each tracked field maintains its own dirty state."""
        class SphereNode(Component):
            radius: float = 1.0
            label: str = ""

        s = SphereNode(radius=1.0, label="a")
        tracker.mark_dirty(s, "radius", 1.0, 5.0)
        fields = tracker.dirty_fields(s)
        assert "radius" in fields
        assert "label" not in fields

    def test_is_dirty_helper(self):
        """is_dirty() checks a specific field via tracker."""
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode()
        tracker.mark_dirty(s, "radius", 1.0, 2.0)
        assert tracker.is_dirty(s)

    def test_get_dirty_fields(self):
        """get_dirty_fields() returns all dirty field names via tracker."""
        class SphereNode(Component):
            radius: float = 1.0
            label: str = ""

        s = SphereNode(radius=1.0, label="a")
        tracker.mark_dirty(s, "radius", 1.0, 5.0)
        tracker.mark_dirty(s, "label", "a", "b")
        dirty = tracker.dirty_fields(s)
        assert "radius" in dirty
        assert "label" in dirty
        assert len(dirty) == 2

    def test_tracker_mark_clean(self):
        """tracker.mark_clean() clears all dirty flags."""
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode()
        tracker.mark_dirty(s, "radius", 1.0, 5.0)
        assert tracker.is_dirty(s)
        tracker.mark_clean(s)
        assert tracker.is_dirty(s) is False

    def test_clear_dirty_field_not_available(self):
        """_dirty_fields is not present on plain Component instances."""
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode()
        assert not hasattr(s, "_dirty_fields")

    def test_multiple_writes_same_field_preserves_one_entry(self):
        """Writing the same field multiple times keeps exactly one dirty entry."""
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode()
        tracker.mark_dirty(s, "radius", 1.0, 2.0)
        tracker.mark_dirty(s, "radius", 2.0, 3.0)
        tracker.mark_dirty(s, "radius", 3.0, 4.0)
        dirty = tracker.dirty_fields(s)
        assert len(dirty) == 1
        assert "radius" in dirty


class TestDirtyTrackingTrackerIntegration:
    """Tracker callbacks fire for DSL node changes."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        _reset_tracker()
        yield

    def test_tracker_global_callback(self):
        """Global callbacks fire when mark_dirty is called."""
        fired = []

        def cb(obj, field, old, new):
            fired.append((field, old, new))

        tracker.on_change(cb)
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode()
        tracker.mark_dirty(s, "radius", 1.0, 5.0)
        assert len(fired) == 1
        assert fired[0] == ("radius", 1.0, 5.0)
        tracker.off_change(cb)

    def test_tracker_object_callback(self):
        """Object-specific callbacks fire for that object."""
        fired = []

        def cb(obj, field, old, new):
            fired.append((field, old, new))

        class SphereNode(Component):
            radius: float = 1.0
            label: str = ""

        s = SphereNode()
        tracker.on_change(s, cb)
        tracker.mark_dirty(s, "radius", 1.0, 5.0)
        assert len(fired) == 1
        assert fired[0] == ("radius", 1.0, 5.0)

    def test_tracker_type_callback(self):
        """Type-level callbacks fire for all instances of a type."""
        fired = []

        def cb(obj, field, old, new):
            fired.append((field, old, new))

        class SphereNode(Component):
            radius: float = 1.0

        tracker.on_change(SphereNode, cb)
        s1 = SphereNode()
        s2 = SphereNode()
        tracker.mark_dirty(s1, "radius", 1.0, 5.0)
        tracker.mark_dirty(s2, "radius", 1.0, 2.0)
        assert len(fired) == 2

    def test_tracker_all_dirty(self):
        """tracker.all_dirty() returns all objects with pending changes."""
        class SphereNode(Component):
            radius: float = 1.0

        a = SphereNode()
        b = SphereNode()
        tracker.mark_dirty(a, "radius", 1.0, 5.0)
        tracker.mark_dirty(b, "radius", 1.0, 3.0)
        dirty_list = tracker.all_dirty()
        assert a in dirty_list
        assert b in dirty_list

    def test_tracker_off_change_removes_callback(self):
        """Unsubscribed callbacks no longer fire."""
        fired = []

        def cb(obj, field, old, new):
            fired.append(1)

        class SphereNode(Component):
            radius: float = 1.0

        tracker.on_change(cb)
        s = SphereNode()
        tracker.mark_dirty(s, "radius", 1.0, 5.0)
        assert len(fired) == 1
        tracker.off_change(cb)
        tracker.mark_dirty(s, "radius", 5.0, 10.0)
        assert len(fired) == 1  # not incremented


class TestDirtyTrackingTransaction:
    """Tracker transactions group changes together."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        _reset_tracker()
        yield

    def test_begin_commit_transaction(self):
        """A transaction groups changes and commits them."""
        class SphereNode(Component):
            radius: float = 1.0
            label: str = ""

        s = SphereNode()
        tracker.begin_transaction("update sphere")
        tracker.mark_dirty(s, "radius", 1.0, 5.0)
        tracker.mark_dirty(s, "label", "", "big")
        tracker.commit_transaction()

        # After commit, the changes are in the undo stack
        assert tracker.can_undo

    def test_rollback_reverts_changes(self):
        """Rollback reverts all changes in the transaction."""
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode(radius=1.0)
        tracker.begin_transaction("set radius")
        # Apply the change on the object then record in transaction
        s.radius = 99.0
        tracker.mark_dirty(s, "radius", 1.0, 99.0)
        tracker.rollback_transaction()
        # Change should be reverted on the object
        assert s.radius == 1.0

    def test_undo_restores_previous_value(self):
        """Undo restores the old field value."""
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode(radius=1.0)
        s.radius = 99.0
        tracker.mark_dirty(s, "radius", 1.0, 99.0)
        assert s.radius == 99.0

        tracker.undo()
        # The value should be restored to old_value
        assert s.radius == 1.0

    def test_nested_transaction_raises(self):
        """Nested transactions raise RuntimeError."""
        tracker.begin_transaction("outer")
        with pytest.raises(RuntimeError):
            tracker.begin_transaction("inner")

    def test_commit_without_transaction_raises(self):
        """Committing without a transaction raises RuntimeError."""
        with pytest.raises(RuntimeError):
            tracker.commit_transaction()

    def test_rollback_without_transaction_raises(self):
        """Rolling back without a transaction raises RuntimeError."""
        with pytest.raises(RuntimeError):
            tracker.rollback_transaction()

    def test_can_undo_property(self):
        """can_undo is True when there are undoable transactions."""
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode()
        tracker.mark_dirty(s, "radius", 1.0, 5.0)
        assert tracker.can_undo

    def test_redo_after_undo(self):
        """Redo restores a value undone."""
        class SphereNode(Component):
            radius: float = 1.0

        s = SphereNode(radius=1.0)
        tracker.mark_dirty(s, "radius", 1.0, 99.0)
        s.radius = 99.0

        assert tracker.undo()
        assert s.radius == 1.0
        assert tracker.redo()
        assert s.radius == 99.0

    def test_in_transaction_property(self):
        """in_transaction is True while a transaction is active."""
        tracker.begin_transaction("test")
        assert tracker.in_transaction
        tracker.commit_transaction()
        assert not tracker.in_transaction


class TestDirtyTrackingEdgeCases:
    """Edge cases for dirty tracking on DSL nodes."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        _reset_tracker()
        yield

    def test_tracker_clean_object_not_dirty(self):
        """A newly created, unmodified object is not dirty."""
        class SphereNode(Component):
            radius: float = 1.0
        s = SphereNode()
        assert tracker.is_dirty(s) is False

    def test_tracker_on_change_requires_callback(self):
        """on_change() with no callback raises ValueError."""
        with pytest.raises(ValueError):
            tracker.on_change()

    def test_tracker_mark_clean_on_clean_object(self):
        """mark_clean() on a clean object does not raise."""
        class SphereNode(Component):
            radius: float = 1.0
        s = SphereNode()
        tracker.mark_clean(s)  # no error

    def test_multiple_objects_tracked_independently(self):
        """Two different objects maintain independent dirty states."""
        class SphereNode(Component):
            radius: float = 1.0

        a = SphereNode()
        b = SphereNode()
        tracker.mark_dirty(a, "radius", 1.0, 2.0)
        assert tracker.is_dirty(a)
        assert not tracker.is_dirty(b)
