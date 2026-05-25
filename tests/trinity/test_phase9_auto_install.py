"""Tests for Phase 9: Metaclass Auto-Install of Descriptors."""

from __future__ import annotations

import warnings

import pytest

from trinity.decorators.ops import Op, Step
from trinity.descriptors.persistence import SerializableDescriptor
from trinity.descriptors.tracking import TrackedDescriptor, is_dirty
from trinity.descriptors.validation import ValidatedDescriptor
from trinity.metaclasses.component_meta import ComponentMeta


_counter = 0


def _unique_name(prefix: str = "P9") -> str:
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


@pytest.fixture(autouse=True)
def _clear_registries():
    ComponentMeta.clear_registry()
    yield
    ComponentMeta.clear_registry()


# =========================================================================
# TestAutoInstallTrack
# =========================================================================


class TestAutoInstallTrack:

    def test_track_step_adds_tracked_descriptor(self):
        """_applied_steps=[Step(Op.TRACK)] adds TrackedDescriptor to field chain."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100.0,
            "_applied_steps": [Step(Op.TRACK)],
            "__module__": __name__,
        })
        desc = cls._field_descriptors["hp"]
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        assert "tracked" in chain_ids

    def test_track_step_skips_already_tracked(self):
        """If _track_changes=True is set, auto-install does not double-add."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100.0,
            "_track_changes": True,
            "_applied_steps": [Step(Op.TRACK)],
            "__module__": __name__,
        })
        desc = cls._field_descriptors["hp"]
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        # Should have exactly one "tracked" (from the normal path)
        assert chain_ids.count("tracked") == 1

    def test_track_step_functional(self):
        """Auto-installed TrackedDescriptor actually tracks dirty state."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100.0,
            "_applied_steps": [Step(Op.TRACK)],
            "__module__": __name__,
        })
        obj = object.__new__(cls)
        _ = obj.hp  # initialize
        assert not is_dirty(obj, "hp")
        obj.hp = 50.0
        assert is_dirty(obj, "hp")

    def test_track_step_all_fields_get_tracked(self):
        """Multiple fields all get TrackedDescriptor from auto-install."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float, "mp": int, "sp": float},
            "hp": 100.0,
            "mp": 50,
            "sp": 25.0,
            "_applied_steps": [Step(Op.TRACK)],
            "__module__": __name__,
        })
        for field in ("hp", "mp", "sp"):
            desc = cls._field_descriptors[field]
            chain_ids = [d.descriptor_id for d in desc.get_chain()]
            assert "tracked" in chain_ids, f"{field} missing tracked descriptor"


# =========================================================================
# TestAutoInstallValidate
# =========================================================================


class TestAutoInstallValidate:

    def test_validate_step_adds_validated_descriptor(self):
        """_applied_steps with VALIDATE adds ValidatedDescriptor to fields."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100.0,
            "_applied_steps": [Step(Op.VALIDATE, {"constraint": "positive"})],
            "__module__": __name__,
        })
        desc = cls._field_descriptors["hp"]
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        assert "validated" in chain_ids

    def test_validate_step_skips_already_validated(self):
        """Field with existing RangeDescriptor (a validation type) is not double-wrapped."""
        from typing import Annotated
        from trinity.descriptors.validation import RangeDescriptor
        name = _unique_name()
        range_desc = RangeDescriptor(field_type=float, min_val=0, max_val=100)
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": Annotated[float, range_desc]},
            "hp": 50.0,
            "_applied_steps": [Step(Op.VALIDATE, {"constraint": "positive"})],
            "__module__": __name__,
        })
        desc = cls._field_descriptors["hp"]
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        # "range" is in chain_ids, so auto-install skips adding "validated"
        assert "range" in chain_ids
        assert "validated" not in chain_ids


# =========================================================================
# TestAutoInstallIntercept
# =========================================================================


class TestAutoInstallIntercept:

    def test_intercept_step_emits_warning(self):
        """_applied_steps with INTERCEPT triggers a UserWarning."""
        name = _unique_name()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cls = ComponentMeta(name, (), {
                "__annotations__": {"hp": float},
                "hp": 100.0,
                "_applied_steps": [Step(Op.INTERCEPT, {"set": "deny"})],
                "__module__": __name__,
            })
        intercept_warnings = [x for x in w if "INTERCEPT" in str(x.message)]
        assert len(intercept_warnings) == 1

    def test_no_intercept_no_warning(self):
        """No INTERCEPT step means no INTERCEPT warning."""
        name = _unique_name()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cls = ComponentMeta(name, (), {
                "__annotations__": {"hp": float},
                "hp": 100.0,
                "_applied_steps": [Step(Op.TRACK)],
                "__module__": __name__,
            })
        intercept_warnings = [x for x in w if "INTERCEPT" in str(x.message)]
        assert len(intercept_warnings) == 0


# =========================================================================
# TestAutoInstallSerialize
# =========================================================================


class TestAutoInstallSerialize:

    def test_serialize_hook_adds_serializable(self):
        """HOOK(on_serialize) auto-adds SerializableDescriptor."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100.0,
            "_applied_steps": [Step(Op.HOOK, {"event": "on_serialize"})],
            "__module__": __name__,
        })
        desc = cls._field_descriptors["hp"]
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        assert "serializable" in chain_ids

    def test_serialize_hook_skips_if_config_set(self):
        """If _serialization_config is set, auto-install does not add SerializableDescriptor."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100.0,
            "_serialization_config": {"format": "json"},
            "_applied_steps": [Step(Op.HOOK, {"event": "on_serialize"})],
            "__module__": __name__,
        })
        desc = cls._field_descriptors["hp"]
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        assert "serializable" not in chain_ids


# =========================================================================
# TestAutoInstallMetaclassSteps
# =========================================================================


class TestAutoInstallMetaclassSteps:

    def test_auto_install_records_metaclass_steps(self):
        """Each auto-installed descriptor adds entry with source=auto_install."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100.0,
            "_applied_steps": [Step(Op.TRACK)],
            "__module__": __name__,
        })
        auto_steps = [s for s in cls._metaclass_steps if s.args.get("source") == "auto_install"]
        assert len(auto_steps) == 1
        assert auto_steps[0].args["descriptor"] == "tracked"

    def test_no_applied_steps_no_auto_install(self):
        """Class without _applied_steps gets no auto-installed descriptors."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100.0,
            "__module__": __name__,
        })
        auto_steps = [s for s in cls._metaclass_steps if s.args.get("source") == "auto_install"]
        assert len(auto_steps) == 0


# =========================================================================
# TestAutoInstallEdgeCases
# =========================================================================


class TestAutoInstallEdgeCases:

    def test_empty_applied_steps(self):
        """_applied_steps=[] does nothing."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100.0,
            "_applied_steps": [],
            "__module__": __name__,
        })
        auto_steps = [s for s in cls._metaclass_steps if s.args.get("source") == "auto_install"]
        assert len(auto_steps) == 0

    def test_multiple_ops_in_applied_steps(self):
        """[Step(Op.TRACK), Step(Op.HOOK, on_serialize)] auto-installs both."""
        name = _unique_name()
        cls = ComponentMeta(name, (), {
            "__annotations__": {"hp": float},
            "hp": 100.0,
            "_applied_steps": [
                Step(Op.TRACK),
                Step(Op.HOOK, {"event": "on_serialize"}),
            ],
            "__module__": __name__,
        })
        desc = cls._field_descriptors["hp"]
        chain_ids = [d.descriptor_id for d in desc.get_chain()]
        assert "tracked" in chain_ids
        assert "serializable" in chain_ids
