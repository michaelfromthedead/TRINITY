"""Tests for Phase 8: Annotated field syntax support in ComponentMeta."""

from __future__ import annotations

import warnings
from typing import Annotated

import pytest

from trinity.descriptors.base import BaseDescriptor
from trinity.descriptors.storage import StorageDescriptor
from trinity.descriptors.tracking import TrackedDescriptor, is_dirty, get_dirty_fields
from trinity.descriptors.validation import RangeDescriptor, ValidatedDescriptor
from trinity.descriptors.observable import ObservableDescriptor
from trinity.metaclasses.component_meta import ComponentMeta
from trinity.metaclasses.engine_meta import EngineMeta


# Counter for unique class names across tests
_counter = 0


def _unique_name(prefix: str = "Comp") -> str:
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class _Base:
    """Mixin to silence the 'Components should be data-only' warnings when
    ComponentMeta encounters methods on test helper classes."""
    pass


# =========================================================================
# Fixtures / setup-teardown
# =========================================================================

@pytest.fixture(autouse=True)
def _clear_registries():
    """Clear ComponentMeta and EngineMeta registries before and after each test."""
    ComponentMeta.clear_registry()
    yield
    ComponentMeta.clear_registry()


# =========================================================================
# TestAnnotatedFieldDetection
# =========================================================================

class TestAnnotatedFieldDetection:

    def test_annotated_field_unwraps_type(self):
        """Annotated[float, TrackedDescriptor] stores float in _field_types."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"value": Annotated[float, TrackedDescriptor]},
            "__module__": __name__,
        })
        assert cls._field_types["value"] is float

    def test_plain_field_unchanged(self):
        """hp: float = 100 still stores float in _field_types."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100,
            "__module__": __name__,
        })
        assert cls._field_types["hp"] is float

    def test_annotated_with_non_descriptor_metadata_ignored(self):
        """Annotated[float, 'some_string', 42] stores float, no crash."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"x": Annotated[float, "some_string", 42]},
            "__module__": __name__,
        })
        assert cls._field_types["x"] is float

    def test_mixed_annotated_and_plain_fields(self):
        """Class with both annotated and plain fields works correctly."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {
                "hp": float,
                "mp": Annotated[int, TrackedDescriptor],
            },
            "hp": 100,
            "mp": 50,
            "__module__": __name__,
        })
        assert cls._field_types["hp"] is float
        assert cls._field_types["mp"] is int


# =========================================================================
# TestAnnotatedDescriptorInstallation
# =========================================================================

class TestAnnotatedDescriptorInstallation:

    def test_descriptor_class_instantiated(self):
        """Annotated[float, TrackedDescriptor] creates a TrackedDescriptor instance."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"value": Annotated[float, TrackedDescriptor]},
            "__module__": __name__,
        })
        desc = cls._field_descriptors.get("value")
        assert desc is not None
        # The chain should contain a TrackedDescriptor
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        assert "tracked" in chain_ids

    def test_descriptor_instance_used_directly(self):
        """Annotated[float, RangeDescriptor(...)] uses the instance directly."""
        name = _unique_name()
        range_desc = RangeDescriptor(field_type=float, min_val=0, max_val=100)
        cls = ComponentMeta(name, (), {
            "__annotations__": {"value": Annotated[float, range_desc]},
            "__module__": __name__,
        })
        desc = cls._field_descriptors.get("value")
        assert desc is not None
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        assert "range" in chain_ids

    def test_multiple_descriptors_composed(self):
        """Annotated[float, TrackedDescriptor, RangeDescriptor(...)] creates a composed chain.

        Order matters for composition rules: TrackedDescriptor (outer) can wrap
        RangeDescriptor (inner), which wraps StorageDescriptor (innermost).
        In the Annotated metadata, descriptors are listed outer-to-inner.
        """
        name = _unique_name()
        range_desc = RangeDescriptor(field_type=float, min_val=0, max_val=100)
        # TrackedDescriptor accepts_inner includes "range", so tracked wraps range
        cls = ComponentMeta(name, (), {
            "__annotations__": {"value": Annotated[float, TrackedDescriptor, range_desc]},
            "__module__": __name__,
        })
        desc = cls._field_descriptors.get("value")
        assert desc is not None
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        assert "tracked" in chain_ids
        assert "range" in chain_ids
        assert "storage" in chain_ids

    def test_annotated_field_in_field_descriptors(self):
        """Annotated field appears in cls._field_descriptors."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"value": Annotated[float, TrackedDescriptor]},
            "__module__": __name__,
        })
        assert "value" in cls._field_descriptors

    def test_annotated_field_skipped_by_install_descriptors(self):
        """Annotated field's descriptor is NOT replaced by _install_descriptors."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"value": Annotated[float, TrackedDescriptor]},
            "__module__": __name__,
        })
        desc = cls._field_descriptors["value"]
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        # Should have tracked from Annotated, NOT a plain StorageDescriptor alone
        assert "tracked" in chain_ids
        # The descriptor installed by _process_fields should remain (not overwritten)
        assert desc is cls._field_descriptors["value"]


# =========================================================================
# TestAnnotatedFieldBehavior
# =========================================================================

class TestAnnotatedFieldBehavior:

    def test_annotated_tracked_field_tracks_changes(self):
        """Field declared with TrackedDescriptor actually tracks dirty state."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"value": Annotated[float, TrackedDescriptor]},
            "value": 0.0,
            "__module__": __name__,
        })
        obj = object.__new__(cls)
        # Read to initialize
        _ = obj.value
        assert not is_dirty(obj, "value")
        obj.value = 42.0
        assert is_dirty(obj, "value")
        assert "value" in get_dirty_fields(obj)

    def test_annotated_range_field_clamps(self):
        """Field declared with RangeDescriptor actually clamps values."""
        name = _unique_name()
        range_desc = RangeDescriptor(field_type=float, min_val=0, max_val=100)
        cls = ComponentMeta(name, (), {
            "__annotations__": {"value": Annotated[float, range_desc]},
            "value": 50.0,
            "__module__": __name__,
        })
        obj = object.__new__(cls)
        obj.value = 200.0
        assert obj.value == 100.0
        obj.value = -10.0
        assert obj.value == 0.0

    def test_annotated_field_default_value(self):
        """current: Annotated[float, TrackedDescriptor] = 50 respects the default."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"current": Annotated[float, TrackedDescriptor]},
            "current": 50.0,
            "__module__": __name__,
        })
        obj = object.__new__(cls)
        assert obj.current == 50.0

    def test_annotated_field_get_set(self):
        """Basic get/set works through annotated descriptor."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"value": Annotated[float, TrackedDescriptor]},
            "value": 0.0,
            "__module__": __name__,
        })
        obj = object.__new__(cls)
        obj.value = 99.0
        assert obj.value == 99.0
        obj.value = -5.0
        assert obj.value == -5.0


# =========================================================================
# TestAnnotatedEdgeCases
# =========================================================================

class TestAnnotatedEdgeCases:

    def test_annotated_no_descriptors_in_metadata(self):
        """Annotated[float, 'just_a_string'] works like a plain field."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"x": Annotated[float, "just_a_string"]},
            "x": 10.0,
            "__module__": __name__,
        })
        assert cls._field_types["x"] is float
        # Should still have a descriptor from _install_descriptors (StorageDescriptor)
        desc = cls._field_descriptors.get("x")
        assert desc is not None
        # Verify it works
        obj = object.__new__(cls)
        obj.x = 42.0
        assert obj.x == 42.0

    def test_annotated_private_field_skipped(self):
        """_internal: Annotated[int, TrackedDescriptor] is skipped (private)."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"_internal": Annotated[int, TrackedDescriptor]},
            "__module__": __name__,
        })
        assert "_internal" not in cls._field_types
        assert "_internal" not in cls._field_descriptors

    def test_annotated_bare_descriptor_class(self):
        """Annotated[float, RangeDescriptor] with bare class (not instance) gets instantiated."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"value": Annotated[float, TrackedDescriptor]},
            "value": 0.0,
            "__module__": __name__,
        })
        desc = cls._field_descriptors["value"]
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        assert "tracked" in chain_ids
        # Verify the instantiated descriptor actually works
        obj = object.__new__(cls)
        _ = obj.value
        obj.value = 10.0
        assert is_dirty(obj, "value")
