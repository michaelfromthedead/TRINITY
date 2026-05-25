"""
Blackbox tests for T-CORE-5.3a: Rust SoA Storage integration.

CLEANROOM -- tests only use the public ComponentMeta API.
No knowledge of internal descriptor implementation or _omega module.

AC1: Component field access reads/writes correctly via Rust SoA storage
AC2: Fallback to Python storage for non-components
AC3: No __delete__ orphan (field delete cleans up storage)
"""

from __future__ import annotations

import pytest

from trinity.metaclasses import ComponentMeta


# =============================================================================
# AC1: Component field access reads/writes correctly
# =============================================================================


class TestAC1_FieldReadWrite:
    """AC1: Component field access reads/writes correctly."""

    def test_write_then_read(self):
        """Writing a field then reading returns the same value."""
        cls = ComponentMeta("Comp1", (), {
            "__annotations__": {"x": int},
            "__module__": __name__,
        })
        inst = cls()
        inst.x = 42
        assert inst.x == 42

    def test_multiple_fields(self):
        """Multiple typed fields all work."""
        cls = ComponentMeta("Comp2", (), {
            "__annotations__": {"x": int, "y": float, "name": str},
            "__module__": __name__,
        })
        inst = cls()
        inst.x = 10
        inst.y = 3.14
        inst.name = "test"
        assert inst.x == 10
        assert inst.y == 3.14
        assert inst.name == "test"

    def test_default_value(self):
        """Field default values are applied correctly."""
        cls = ComponentMeta("Comp3", (), {
            "__annotations__": {"x": int},
            "x": 100,
            "__module__": __name__,
        })
        inst = cls()
        assert inst.x == 100

    def test_default_bool(self):
        """Bool default works."""
        cls = ComponentMeta("Comp3b", (), {
            "__annotations__": {"flag": bool},
            "flag": True,
            "__module__": __name__,
        })
        inst = cls()
        assert inst.flag is True

    def test_default_str(self):
        """Str default works."""
        cls = ComponentMeta("Comp3c", (), {
            "__annotations__": {"tag": str},
            "tag": "hello",
            "__module__": __name__,
        })
        inst = cls()
        assert inst.tag == "hello"

    def test_overwrite_default(self):
        """Field update replaces the default."""
        cls = ComponentMeta("Comp4", (), {
            "__annotations__": {"x": int},
            "x": 0,
            "__module__": __name__,
        })
        inst = cls()
        assert inst.x == 0
        inst.x = 999
        assert inst.x == 999

    def test_separate_instances_isolated(self):
        """Each component instance has its own field storage."""
        cls = ComponentMeta("Comp5", (), {
            "__annotations__": {"x": int},
            "x": 0,
            "__module__": __name__,
        })
        a = cls()
        b = cls()
        a.x = 10
        b.x = 20
        assert a.x == 10
        assert b.x == 20
        assert a.x != b.x


# =============================================================================
# AC2: Fallback to Python storage for non-components
# =============================================================================


class TestAC2_FallbackToPython:
    """AC2: Fallback to Python storage for non-components."""

    def test_plain_object_field_access(self):
        """Plain class instances (non-components) work for field access."""
        class Plain:
            pass
        p = Plain()
        p.x = 42
        assert p.x == 42

    def test_component_without_entity_metadata(self):
        """Components work when created without entity metadata."""
        cls = ComponentMeta("Comp6", (), {
            "__annotations__": {"val": str},
            "val": "hello",
            "__module__": __name__,
        })
        inst = cls()
        assert inst.val == "hello"
        inst.val = "world"
        assert inst.val == "world"

    def test_multiple_types_fallback(self):
        """Component with multiple field types works via fallback."""
        cls = ComponentMeta("Comp7", (), {
            "__annotations__": {"a": int, "b": float, "c": bool, "d": str},
            "c": True,
            "__module__": __name__,
        })
        inst = cls()
        inst.a = 99
        inst.b = -1.5
        inst.d = "text"
        assert inst.a == 99
        assert inst.b == -1.5
        assert inst.c is True
        assert inst.d == "text"


# =============================================================================
# AC3: No __delete__ orphan (field delete cleans up storage)
# =============================================================================


class TestAC3_DeleteCleanup:
    """AC3: No __delete__ orphan (field delete cleans up storage)."""

    def test_delete_restores_default(self):
        """After deleting a field, reading returns the default value."""
        cls = ComponentMeta("Comp8", (), {
            "__annotations__": {"x": int},
            "x": 42,
            "__module__": __name__,
        })
        inst = cls()
        inst.x = 100
        del inst.x
        # After delete, should get default back, not orphan
        assert inst.x == 42

    def test_delete_no_default_returns_none(self):
        """Deleting a field with no default returns None."""
        cls = ComponentMeta("Comp8b", (), {
            "__annotations__": {"x": int},
            "__module__": __name__,
        })
        inst = cls()
        inst.x = 5
        del inst.x
        assert inst.x is None

    def test_delete_then_write(self):
        """After delete, writing a new value works correctly."""
        cls = ComponentMeta("Comp9", (), {
            "__annotations__": {"x": int},
            "__module__": __name__,
        })
        inst = cls()
        inst.x = 1
        del inst.x
        inst.x = 2
        assert inst.x == 2

    def test_delete_one_field_others_untouched(self):
        """Deleting one field does not affect other fields."""
        cls = ComponentMeta("Comp10", (), {
            "__annotations__": {"a": int, "b": int},
            "a": 10,
            "b": 20,
            "__module__": __name__,
        })
        inst = cls()
        inst.a = 99
        inst.b = 88
        del inst.a
        assert inst.b == 88
        assert inst.a == 10


# =============================================================================
# ComponentMeta integration verification
# =============================================================================


class TestComponentMetaIntegration:
    """Verifies ComponentMeta creates well-formed components."""

    def test_component_id_assigned(self):
        cls = ComponentMeta("Comp11", (), {"__module__": __name__})
        assert hasattr(cls, "_component_id")
        assert isinstance(cls._component_id, int)

    def test_field_offsets_present(self):
        cls = ComponentMeta("Comp12", (), {
            "__annotations__": {"x": int, "y": float},
            "__module__": __name__,
        })
        assert "x" in cls._field_offsets
        assert "y" in cls._field_offsets
        # Offsets are size-based: int=4 bytes, float=4 bytes
        assert cls._field_offsets["x"] == 0
        assert cls._field_offsets["y"] == 4

    def test_field_types_present(self):
        cls = ComponentMeta("Comp13", (), {
            "__annotations__": {"x": int, "y": str},
            "__module__": __name__,
        })
        assert cls._field_types["x"] is int
        assert cls._field_types["y"] is str

    def test_field_descriptors_installed(self):
        cls = ComponentMeta("Comp14", (), {
            "__annotations__": {"x": int},
            "__module__": __name__,
        })
        assert "x" in cls._field_descriptors
        desc = cls._field_descriptors["x"]
        assert hasattr(desc, "__get__")
        assert hasattr(desc, "__set__")
        assert hasattr(desc, "__delete__")
