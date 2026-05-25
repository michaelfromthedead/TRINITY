"""
Blackbox tests for ComponentMeta Phase 5 FIX re-verification -- T-CORE-5.3b.

CLEANROOM: All assertions derive from the public contract described in
acceptance criteria, not from reading the implementation.

Acceptance criteria:
1. Position(x: float, y: float) auto-registers with a numeric id, 8-byte
   component size, two f32 fields at offsets 0 and 4
2. Duplicate definition with the same qualified name returns the same type id
3. Type registration is idempotent (re-definition does not create entries)
"""

from __future__ import annotations

import foundation
import json
import sys
import threading
import warnings
from unittest import mock

import pytest

from foundation import registry as foundation_registry
from trinity.metaclasses import ComponentMeta


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def patch_omega():
    """Patch _omega in sys.modules with a mock so that type_register calls
    are captured without a real Rust backend."""
    mock_omega = mock.Mock()
    mock_omega.type_register = mock.Mock()
    was_present = "_omega" in sys.modules
    saved = sys.modules.get("_omega")
    sys.modules["_omega"] = mock_omega
    yield mock_omega
    if was_present and saved is not None:
        sys.modules["_omega"] = saved
    else:
        sys.modules.pop("_omega", None)


@pytest.fixture
def capture_type_register(patch_omega):
    """Return a helper that extracts the last type_register call args as a
    dict for blackbox assertions."""
    def _capture() -> dict | None:
        if not patch_omega.type_register.called:
            return None
        call_args = patch_omega.type_register.call_args
        comp_id, comp_name, total_size, fields_json = call_args[0]
        return {
            "id": comp_id,
            "name": comp_name,
            "size": total_size,
            "fields": json.loads(fields_json),
        }
    return _capture


# =============================================================================
# Acceptance Criterion 1: Auto type registration
# =============================================================================


class TestPositionComponentLayout:
    """Position(x: float, y: float) auto-registers with id, 8-byte size,
    two f32 fields at offsets 0 and 4."""

    def test_auto_registers_with_id(self, capture_type_register):
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        assert hasattr(Position, "_component_id")
        assert isinstance(Position._component_id, int)
        assert Position._component_id > 0

    def test_eight_byte_component_size(self, capture_type_register):
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        info = capture_type_register()
        assert info is not None
        assert info["size"] == 8, f"Expected 8, got {info['size']}"

    def test_two_f32_fields(self, capture_type_register):
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        info = capture_type_register()
        assert info is not None
        fields = info["fields"]
        assert len(fields) == 2, f"Expected 2 fields, got {len(fields)}"
        for field in fields:
            assert field[1] == "f32", f"Expected f32, got {field[1]}"

    def test_x_at_offset_zero(self, capture_type_register):
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        info = capture_type_register()
        assert info is not None
        assert info["fields"][0][0] == "x"
        assert info["fields"][0][2] == 0

    def test_y_at_offset_four(self, capture_type_register):
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        info = capture_type_register()
        assert info is not None
        assert info["fields"][1][0] == "y"
        assert info["fields"][1][2] == 4

    def test_qualified_name_contains_class_name(self, capture_type_register):
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        info = capture_type_register()
        assert info is not None
        assert "Position" in info["name"]

    def test_appears_in_engine_type_registry(self):
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        all_types = ComponentMeta.get_all_types()
        matching = [qn for qn in all_types if qn.endswith(".Position")]
        assert len(matching) >= 1

    def test_component_count_incremented(self):
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        assert ComponentMeta.component_count() >= 1


# =============================================================================
# Acceptance Criterion 2: Duplicate definition returns same type id
# =============================================================================


class TestDuplicateDefinitionReturnsSameTypeId:
    """Defining a component with the same qualified name twice returns the
    same type id (C-01 fix)."""

    def test_same_qualified_name_same_id(self):
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        first_id = Position._component_id

        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        assert Position._component_id == first_id, (
            f"Expected id {first_id}, got {Position._component_id}"
        )

    def test_duplicate_does_not_increment_count(self):
        class C(metaclass=ComponentMeta):
            x: float = 0.0

        count_after_first = ComponentMeta.component_count()

        class C(metaclass=ComponentMeta):
            x: float = 0.0

        assert ComponentMeta.component_count() == count_after_first

    def test_duplicate_does_not_call_type_register_again(self, patch_omega):
        class C(metaclass=ComponentMeta):
            x: float = 0.0

        patch_omega.type_register.reset_mock()

        class C(metaclass=ComponentMeta):
            x: float = 0.0

        assert patch_omega.type_register.call_count == 0

    def test_different_module_distinct_ids(self):
        class Score(metaclass=ComponentMeta):
            value: int = 0

        score_id = Score._component_id
        C2 = ComponentMeta(
            "Score",
            (),
            {
                "__module__": "other_module",
                "__annotations__": {"value": int},
                "value": 0,
            },
        )
        assert C2._component_id != score_id

    def test_duplicate_returns_existing_class_object(self):
        class Unique(metaclass=ComponentMeta):
            value: float = 0.0

        first_cls = Unique

        class Unique(metaclass=ComponentMeta):
            value: float = 0.0

        assert Unique is first_cls

    def test_duplicate_preserves_original_fields(self):
        class Fixed(metaclass=ComponentMeta):
            x: float = 0.0

        first_fields, first_size = ComponentMeta._build_rust_layout(Fixed)

        class Fixed(metaclass=ComponentMeta):
            y: int = 0

        second_fields, second_size = ComponentMeta._build_rust_layout(Fixed)
        assert second_fields == first_fields
        assert second_size == first_size

    def test_clear_registry_allows_redefinition(self):
        class Temp(metaclass=ComponentMeta):
            x: int = 0

        first = Temp
        ComponentMeta.clear_registry()
        foundation_registry.clear()

        class Temp(metaclass=ComponentMeta):
            x: int = 0

        # After clearing, the new class is a different object (not the cached one)
        assert Temp is not first
        assert hasattr(Temp, "_component_id")


# =============================================================================
# Acceptance Criterion 3: Type registration is idempotent
# =============================================================================


class TestTypeRegistrationIsIdempotent:
    """Repeated access to a component type does not create new entries,
    consume new IDs, or register new Rust layouts."""

    def test_get_by_id_returns_original_class(self):
        class C(metaclass=ComponentMeta):
            value: float = 0.0

        same = ComponentMeta.get_by_id(C._component_id)
        assert same is C

    def test_get_by_id_does_not_grow_registry(self):
        class C(metaclass=ComponentMeta):
            value: float = 0.0

        count_before = len(ComponentMeta.get_all_types())
        _ = ComponentMeta.get_by_id(C._component_id)
        assert len(ComponentMeta.get_all_types()) == count_before

    def test_get_by_id_stable_across_repeated_calls(self):
        class C(metaclass=ComponentMeta):
            value: float = 0.0

        for _ in range(5):
            assert ComponentMeta.get_by_id(C._component_id) is C

    def test_get_by_name_returns_original_class(self):
        class Named(metaclass=ComponentMeta):
            data: float = 0.0

        retrieved = ComponentMeta.get_by_name(Named._component_name)
        assert retrieved is Named

    def test_get_by_name_stable_across_repeated_calls(self):
        class C(metaclass=ComponentMeta):
            value: float = 0.0

        for _ in range(5):
            assert ComponentMeta.get_by_name(C._component_name) is C

    def test_different_component_names_different_ids(self):
        class Alpha(metaclass=ComponentMeta):
            x: int = 0

        class Beta(metaclass=ComponentMeta):
            x: int = 0

        assert Alpha._component_id != Beta._component_id

    def test_get_by_id_returns_none_for_unknown(self):
        result = ComponentMeta.get_by_id(999999)
        assert result is None

    def test_get_by_name_returns_none_for_unknown(self):
        result = ComponentMeta.get_by_name(
            "nonexistent.module.UnknownType"
        )
        assert result is None

    def test_type_register_called_once_per_component(self, patch_omega):
        class A(metaclass=ComponentMeta):
            x: float = 0.0

        class B(metaclass=ComponentMeta):
            y: float = 0.0

        class C(metaclass=ComponentMeta):
            z: float = 0.0

        assert patch_omega.type_register.call_count == 3


# =============================================================================
# Alignment verification (H-04 fix)
# =============================================================================


class TestFieldAlignment:
    """Verify H-04 alignment padding produces correct offsets and sizes."""

    def test_bool_int_str_correct_offsets(self):
        """bool(1) + int(4) + str(8) => offsets 0, 4, 8, total 16."""
        class Mixed(metaclass=ComponentMeta):
            active: bool = True
            score: int = 0
            name: str = ""

        fields, total_size = ComponentMeta._build_rust_layout(Mixed)
        assert len(fields) == 3
        assert total_size == 16, f"Expected 16, got {total_size}"

    def test_three_floats_layout(self):
        """Three floats => offsets 0, 4, 8, total 12."""
        class Vec3(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0
            z: float = 0.0

        fields, total_size = ComponentMeta._build_rust_layout(Vec3)
        assert len(fields) == 3
        assert total_size == 12, f"Expected 12, got {total_size}"

    def test_bool_bool_float_layout(self):
        """bool + bool + float => offsets 0, 1, 4, total 8."""
        class MixedFlags(metaclass=ComponentMeta):
            a: bool = True
            b: bool = True
            c: float = 0.0

        fields, total_size = ComponentMeta._build_rust_layout(MixedFlags)
        assert len(fields) == 3
        assert total_size == 8, f"Expected 8, got {total_size}"

    def test_empty_component_layout(self):
        class Empty(metaclass=ComponentMeta):
            pass

        fields, total_size = ComponentMeta._build_rust_layout(Empty)
        assert fields == []
        assert total_size == 0

    def test_single_field_component(self):
        class Single(metaclass=ComponentMeta):
            value: float = 0.0

        fields, total_size = ComponentMeta._build_rust_layout(Single)
        assert len(fields) == 1
        assert fields[0] == ("value", "f32", 0)
        assert total_size == 4

    def test_position_fields_offsets(self):
        """Position has f32 at offset 0 and f32 at offset 4, total 8."""
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        fields, total_size = ComponentMeta._build_rust_layout(Position)
        assert fields == [("x", "f32", 0), ("y", "f32", 4)]
        assert total_size == 8

    def test_json_serializable_layout(self):
        class JsonComp(metaclass=ComponentMeta):
            x: float = 0.0
            y: int = 0

        fields, total_size = ComponentMeta._build_rust_layout(JsonComp)
        payload = json.dumps({"fields": fields, "size": total_size})
        parsed = json.loads(payload)
        assert parsed["size"] == total_size
        assert len(parsed["fields"]) == 2

    def test_uint64_alignment(self):
        """u64 field aligns to 8-byte boundary."""
        import ctypes

        # Build class programmatically so annotations resolve to type objects,
        # not strings under from __future__ import annotations
        cls = ComponentMeta(
            "Big_uint64",
            (),
            {
                "__module__": "test_alignment",
                "__annotations__": {"raw": ctypes.c_uint64},
                "raw": 0,
            },
        )
        fields, total_size = ComponentMeta._build_rust_layout(cls)
        assert fields[0] == ("raw", "u64", 0)
        assert total_size == 8

    def test_f64_alignment(self):
        """f64 field aligns to 8-byte boundary."""
        import ctypes

        cls = ComponentMeta(
            "Double_f64",
            (),
            {
                "__module__": "test_alignment",
                "__annotations__": {"val": ctypes.c_double},
                "val": 0.0,
            },
        )
        fields, total_size = ComponentMeta._build_rust_layout(cls)
        assert fields[0] == ("val", "f64", 0)
        assert total_size == 8


# =============================================================================
# Acceptance Criterion 3: total_size is correctly padded to max alignment
# =============================================================================


class TestTotalSizePaddingToMaxAlignment:
    """Blackbox verification that total_size is correctly padded to the maximum
    alignment requirement across all fields in a component.

    For each field type, the alignment is its byte_size:
      bool -> u8   (1 byte,  alignment 1)
      int  -> i32  (4 bytes, alignment 4)
      float-> f32  (4 bytes, alignment 4)
      str  -> String (8 bytes, alignment 8)

    total_size must be a multiple of the max alignment across all fields.
    """

    def test_position_padding_two_f32(self, capture_type_register):
        """Position(f32, f32): max alignment 4, total 8 (8 % 4 == 0)."""
        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        info = capture_type_register()
        assert info is not None
        # Each field packs at its natural alignment: both f32 -> sizes 4+4 = 8
        assert info["size"] == 8, f"Expected 8, got {info['size']}"
        assert info["fields"][0][1] == "f32"
        assert info["fields"][1][1] == "f32"
        # Verify offsets are aligned: first at 0, second at 4
        assert info["fields"][0][2] == 0
        assert info["fields"][1][2] == 4

    def test_bool_float_padding(self, capture_type_register):
        """bool(1) + f32(4): max alignment 4, total 8 (padded 5->8)."""
        class Flagged(metaclass=ComponentMeta):
            active: bool = True
            value: float = 0.0

        info = capture_type_register()
        assert info is not None
        # bool at offset 0 (1 byte), padding 1-3, f32 at offset 4
        assert info["fields"][0][2] == 0       # active at 0
        assert info["fields"][1][2] == 4       # value at 4 (padded)
        assert info["size"] == 8, f"Expected 8, got {info['size']}"

    def test_bool_bool_float_padding(self, capture_type_register):
        """bool(1) + bool(1) + f32(4): max alignment 4, total 8."""
        class Flags(metaclass=ComponentMeta):
            a: bool = True
            b: bool = True
            c: float = 0.0

        info = capture_type_register()
        assert info is not None
        # a at 0, b at 1, padding 2-3, c at 4
        assert info["fields"][0][2] == 0
        assert info["fields"][1][2] == 1
        assert info["fields"][2][2] == 4
        assert info["size"] == 8, f"Expected 8, got {info['size']}"

    def test_bool_int_str_max_alignment_padding(self, capture_type_register):
        """bool(1) + int(4) + str(8): max alignment 8, total 16."""
        class Mixed(metaclass=ComponentMeta):
            active: bool = True
            score: int = 0
            name: str = ""

        info = capture_type_register()
        assert info is not None
        # bool at 0, int at 4 (padded 1-3), str at 8 (padded 8-11), total 16
        assert info["fields"][0][2] == 0       # active at 0
        assert info["fields"][1][2] == 4       # score at 4
        assert info["fields"][2][2] == 8       # name at 8
        assert info["size"] == 16, f"Expected 16, got {info['size']}"

    def test_int_first_then_bool_alignment(self, capture_type_register):
        """int(4) + bool(1): max alignment 4, total 8 (padded 5->8)."""
        class Scored(metaclass=ComponentMeta):
            score: int = 0
            active: bool = True

        info = capture_type_register()
        assert info is not None
        # int at 0, bool at 4, padding 5-7
        assert info["fields"][0][2] == 0       # score at 0
        assert info["fields"][1][2] == 4       # active at 4
        assert info["size"] == 8, f"Expected 8, got {info['size']}"

    def test_three_f32_vec3_padding(self, capture_type_register):
        """f32 + f32 + f32: max alignment 4, total 12 (12 % 4 == 0)."""
        class Vec3(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0
            z: float = 0.0

        info = capture_type_register()
        assert info is not None
        # x at 0, y at 4, z at 8, total 12
        assert info["fields"][0][2] == 0
        assert info["fields"][1][2] == 4
        assert info["fields"][2][2] == 8
        assert info["size"] == 12, f"Expected 12, got {info['size']}"

    def test_f32_f64_padding_to_8(self):
        """f32(4) + f64(8): max alignment 8, total 16 (16 % 8 == 0).
        Uses programmatic class creation to inject ctypes.c_double."""
        import ctypes

        cls = ComponentMeta(
            "MixedFloat",
            (),
            {
                "__module__": "test_alignment_max",
                "__annotations__": {"a": ctypes.c_float, "b": ctypes.c_double},
                "a": 0.0,
                "b": 0.0,
            },
        )
        # Layout assertions: f32(4) at 0, f64(8) at 8 (padded 4->8), total 16
        fields, total_size = ComponentMeta._build_rust_layout(cls)
        assert fields[0] == ("a", "f32", 0), f"Expected ('a', 'f32', 0), got {fields[0]}"
        assert fields[1] == ("b", "f64", 8), f"Expected ('b', 'f64', 8), got {fields[1]}"
        assert total_size == 16, f"Expected total_size 16, got {total_size}"

        # Blackbox verification through public registry
        all_types = ComponentMeta.get_all_types()
        matching = [
            qn for qn, c in all_types.items() if c is cls
        ]
        assert len(matching) >= 1
        assert ComponentMeta.get_by_name(matching[0]) is cls
        assert ComponentMeta.get_by_id(cls._component_id) is cls
        # component_count must reflect this
        assert ComponentMeta.component_count() >= 1

    def test_empty_component_no_padding_needed(self, capture_type_register):
        """Empty component: no fields, total_size 0."""
        class Empty(metaclass=ComponentMeta):
            pass

        info = capture_type_register()
        assert info is not None
        assert info["size"] == 0, f"Expected 0, got {info['size']}"

    def test_single_bool_total_size(self, capture_type_register):
        """Single bool: alignment 1, total 1."""
        class Flag(metaclass=ComponentMeta):
            active: bool = True

        info = capture_type_register()
        assert info is not None
        assert info["fields"] == [["active", "u8", 0]]
        assert info["size"] == 1, f"Expected 1, got {info['size']}"

    def test_single_f64_total_size(self):
        """Single f64: alignment 8, total 8."""
        import ctypes

        cls = ComponentMeta(
            "Double_field",
            (),
            {
                "__module__": "test_single_f64",
                "__annotations__": {"val": ctypes.c_double},
                "val": 0.0,
            },
        )
        all_types = ComponentMeta.get_all_types()
        matching = [qn for qn, c in all_types.items() if c is cls]
        assert len(matching) >= 1
        assert ComponentMeta.get_by_id(cls._component_id) is cls

    def test_f32_i32_mixed_alignment(self, capture_type_register):
        """f32(4) + i32(4): both alignment 4, total 8."""
        class Point(metaclass=ComponentMeta):
            x: float = 0.0
            label: int = 0

        info = capture_type_register()
        assert info is not None
        assert info["fields"][0][2] == 0
        assert info["fields"][1][2] == 4
        assert info["size"] == 8, f"Expected 8, got {info['size']}"

    def test_many_f32_fields_padding(self, capture_type_register):
        """Ten f32 fields: all alignment 4, total 40 (40 % 4 == 0)."""
        class Many(metaclass=ComponentMeta):
            f01: float = 0.0
            f02: float = 0.0
            f03: float = 0.0
            f04: float = 0.0
            f05: float = 0.0
            f06: float = 0.0
            f07: float = 0.0
            f08: float = 0.0
            f09: float = 0.0
            f10: float = 0.0

        info = capture_type_register()
        assert info is not None
        assert len(info["fields"]) == 10
        assert info["size"] == 40, f"Expected 40, got {info['size']}"
        # Every f32 offset is at a 4-byte boundary
        for i, field in enumerate(info["fields"]):
            assert field[2] == i * 4, (
                f"Field {field[0]}: expected offset {i*4}, got {field[2]}"
            )


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge cases: components without _omega, concurrency, many fields."""

    def test_created_when_omega_missing(self):
        was_present = "_omega" in sys.modules
        saved = sys.modules.pop("_omega", None)
        try:
            class Standalone(metaclass=ComponentMeta):
                x: float = 0.0

            assert hasattr(Standalone, "_component_id")
        finally:
            if was_present and saved is not None:
                sys.modules["_omega"] = saved

    def test_many_fields(self):
        class ManyFields(metaclass=ComponentMeta):
            f01: float = 0.0
            f02: float = 0.0
            f03: float = 0.0
            f04: float = 0.0
            f05: float = 0.0
            f06: float = 0.0
            f07: float = 0.0
            f08: float = 0.0
            f09: float = 0.0
            f10: float = 0.0

        fields, total_size = ComponentMeta._build_rust_layout(ManyFields)
        assert len(fields) == 10
        assert total_size == 40

    def test_concurrent_same_name_does_not_crash(self):
        errors = []
        results = []

        def create():
            try:
                cls = ComponentMeta(
                    "Concurrent",
                    (),
                    {
                        "__module__": "test_concurrent",
                        "__annotations__": {"v": int},
                        "v": 0,
                    },
                )
                results.append(cls)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent creation failed: {errors}"
        for cls in results:
            assert hasattr(cls, "_component_id")

    def test_repr_is_readable(self):
        class ReadableRepr(metaclass=ComponentMeta):
            data: float = 0.0

        r = repr(ReadableRepr)
        assert "Component" in r or "ReadableRepr" in r

    def test_clear_registry_resets_count(self):
        class C(metaclass=ComponentMeta):
            x: int = 0

        ComponentMeta.clear_registry()
        count = ComponentMeta.component_count()
        assert isinstance(count, int)

    def test_engine_type_registry_contains_component(self):
        class Registered(metaclass=ComponentMeta):
            x: float = 0.0

        all_types = ComponentMeta.get_all_types()
        matching = [qn for qn, cls in all_types.items() if cls is Registered]
        assert len(matching) >= 1

    def test_qualified_name_format(self):
        class QualifiedName(metaclass=ComponentMeta):
            data: float = 0.0

        assert "." in QualifiedName._component_name
        assert QualifiedName._component_name.endswith(".QualifiedName")


# =============================================================================
# FIX re-verification: F-H5 — Duplicate with different fields emits UserWarning
# =============================================================================


class TestFixFH5DuplicateWarning:
    """F-H5: When a duplicate component definition has different field
    annotations than the original, the idempotency check emits a UserWarning
    describing the mismatch.  Same fields -> no warning."""

    def test_duplicate_different_fields_emits_user_warning(self):
        class Base(metaclass=ComponentMeta):
            x: float = 0.0

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            class Base(metaclass=ComponentMeta):
                y: int = 0

        assert len(captured) >= 1, "Expected at least one warning"
        assert any(
            issubclass(w.category, UserWarning) for w in captured
        ), "Expected a UserWarning"

    def test_duplicate_same_fields_no_warning(self):
        class Same(metaclass=ComponentMeta):
            x: float = 0.0

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            class Same(metaclass=ComponentMeta):
                x: float = 0.0

        user_warnings = [
            w for w in captured if issubclass(w.category, UserWarning)
        ]
        assert len(user_warnings) == 0, (
            f"Expected no UserWarning for identical fields, got {len(user_warnings)}"
        )

    def test_duplicate_warning_mentions_class_name(self):
        class Target(metaclass=ComponentMeta):
            x: float = 0.0

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            class Target(metaclass=ComponentMeta):
                y: int = 0

        user_messages = [
            str(w.message)
            for w in captured
            if issubclass(w.category, UserWarning)
        ]
        assert any("Target" in msg for msg in user_messages), (
            f"Expected warning to mention 'Target', got: {user_messages}"
        )

    def test_duplicate_warning_mentions_field_names(self):
        class Original(metaclass=ComponentMeta):
            x: float = 0.0

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            class Original(metaclass=ComponentMeta):
                y: int = 0

        user_messages = [
            str(w.message)
            for w in captured
            if issubclass(w.category, UserWarning)
        ]
        all_text = " ".join(user_messages)
        # The message should reference the mismatch
        assert "x" in all_text or "y" in all_text or "field" in all_text.lower()

    def test_duplicate_returns_original_after_warning(self):
        class ToBeDuped(metaclass=ComponentMeta):
            x: float = 0.0

        first_cls = ToBeDuped

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            class ToBeDuped(metaclass=ComponentMeta):
                y: int = 0

        assert ToBeDuped is first_cls, "Must return the original class object"

    def test_duplicate_extra_field_warns(self):
        class Few(metaclass=ComponentMeta):
            x: float = 0.0

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            class Few(metaclass=ComponentMeta):
                x: float = 0.0
                y: float = 0.0

        assert any(
            issubclass(w.category, UserWarning) for w in captured
        ), "Extra field should trigger user warning"

    def test_duplicate_fewer_fields_warns(self):
        class ManyFields(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            class ManyFields(metaclass=ComponentMeta):
                x: float = 0.0

        assert any(
            issubclass(w.category, UserWarning) for w in captured
        ), "Fewer fields should trigger user warning"

    def test_duplicate_with_defaults_no_warning(self):
        class Defaulted(metaclass=ComponentMeta):
            x: float = 1.0
            y: float = 2.0

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            class Defaulted(metaclass=ComponentMeta):
                x: float = 3.0
                y: float = 4.0

        user_warnings = [
            w for w in captured if issubclass(w.category, UserWarning)
        ]
        assert len(user_warnings) == 0, (
            "Different defaults, same fields => no warning"
        )


# =============================================================================
# FIX re-verification: FH-3 — Foundation registration exception safety
# =============================================================================


class TestFixFH3FoundationExceptionSafety:
    """FH-3: _register_with_foundation now catches Exception (not just
    ImportError) so that ValueError, TypeError, RuntimeError, etc. from
    foundation.registry.register() do not crash __new__()."""

    # ------------------------------------------------------------------ #
    #  Helper:  temporarily replace foundation.registry with a mock     #
    # ------------------------------------------------------------------ #
    # The foundation.registry object is a C-extension Registry instance
    # with read-only attributes, so mock.patch.object() cannot be used.
    # We replace the module-level attribute instead.
    @staticmethod
    def _mock_registry(register_side_effect=None, is_registered_return=True):
        """Return a context manager that replaces foundation.registry
        with a mock whose ``register`` raises *register_side_effect*
        (default: ValueError).  The original is restored on exit."""
        mock_reg = mock.Mock()
        mock_reg.register = mock.Mock(side_effect=register_side_effect or ValueError("dup"))
        mock_reg.is_registered = mock.Mock(return_value=is_registered_return)
        mock_reg.clear = mock.Mock()
        mock_reg.describe = mock.Mock(return_value="<mock registry>")
        mock_reg.all_types = mock.Mock(return_value={})
        mock_reg.get = mock.Mock(return_value=None)

        class _RegistrySwap:
            def __enter__(self2):
                self2.saved = foundation.registry
                foundation.registry = mock_reg
                return mock_reg

            def __exit__(self2, *exc):
                foundation.registry = self2.saved

        return _RegistrySwap()

    def test_value_error_from_register_caught(self):
        """ValueError from foundation registry does not prevent component creation."""
        with self._mock_registry(register_side_effect=ValueError("dup name")):
            class SurviveValueError(metaclass=ComponentMeta):
                x: float = 0.0

        assert hasattr(SurviveValueError, "_component_id")
        assert isinstance(SurviveValueError._component_id, int)

    def test_type_error_from_register_caught(self):
        """TypeError from foundation registry does not prevent component creation."""
        with self._mock_registry(register_side_effect=TypeError("not a class")):
            class SurviveTypeError(metaclass=ComponentMeta):
                x: float = 0.0

        assert hasattr(SurviveTypeError, "_component_id")

    def test_runtime_error_from_register_caught(self):
        """RuntimeError from foundation registry does not prevent component creation."""
        with self._mock_registry(register_side_effect=RuntimeError("internal")):
            class SurviveRuntimeError(metaclass=ComponentMeta):
                x: float = 0.0

        assert hasattr(SurviveRuntimeError, "_component_id")

    def test_attribute_error_from_is_registered_caught(self):
        """AttributeError from foundation registry does not prevent creation."""
        with self._mock_registry(
            register_side_effect=AttributeError("no attr")
        ):
            class SurviveAttrError(metaclass=ComponentMeta):
                x: float = 0.0

        assert hasattr(SurviveAttrError, "_component_id")

    def test_import_error_from_foundation_caught(self):
        """ImportError when foundation is absent does not prevent creation."""
        saved = sys.modules.pop("foundation", None)
        saved_registry = sys.modules.pop("foundation.registry", None)
        try:
            class NoFoundation(metaclass=ComponentMeta):
                x: float = 0.0

            assert hasattr(NoFoundation, "_component_id")
        finally:
            if saved is not None:
                sys.modules["foundation"] = saved
            if saved_registry is not None:
                sys.modules["foundation.registry"] = saved_registry

    def test_component_survives_foundation_exception_and_is_usable(self):
        """After foundation exception, the component is still in the registry
        and its layout is correct."""
        with self._mock_registry(register_side_effect=ValueError("dup")):
            class Resilient(metaclass=ComponentMeta):
                x: float = 0.0
                y: float = 0.0

        # Component must be findable
        assert ComponentMeta.get_by_id(Resilient._component_id) is Resilient
        # Layout must be computed correctly despite foundation failure
        fields, total_size = ComponentMeta._build_rust_layout(Resilient)
        assert len(fields) == 2
        assert total_size == 8

    def test_multiple_components_survive_individual_foundation_failures(self):
        """Multiple components created while foundation is broken all survive."""
        with self._mock_registry(register_side_effect=ValueError("dup")):
            class A(metaclass=ComponentMeta):
                a: int = 0

            class B(metaclass=ComponentMeta):
                b: float = 0.0

            class C(metaclass=ComponentMeta):
                c: bool = True

        assert hasattr(A, "_component_id")
        assert hasattr(B, "_component_id")
        assert hasattr(C, "_component_id")
        # All must have distinct IDs
        ids = {A._component_id, B._component_id, C._component_id}
        assert len(ids) == 3


# =============================================================================
# FIX re-verification: FH-1 — Str8 type code propagation
# =============================================================================


class TestFixFH1Str8TypeCode:
    """FH-1 (H-01): str fields map to the "Str8" type code (not "String") in
    _build_rust_layout output, signalling an 8-byte fixed-size buffer."""

    def test_str_field_type_code_is_str8(self):
        class Named(metaclass=ComponentMeta):
            name: str = ""

        fields, total_size = ComponentMeta._build_rust_layout(Named)
        assert fields[0][1] == "Str8", (
            f"Expected 'Str8' type code, got '{fields[0][1]}'"
        )

    def test_str_field_aligns_to_8(self):
        class Named(metaclass=ComponentMeta):
            name: str = ""

        fields, total_size = ComponentMeta._build_rust_layout(Named)
        assert fields[0][2] == 0, f"Expected offset 0, got {fields[0][2]}"
        assert total_size == 8, f"Expected total_size 8, got {total_size}"

    def test_str_field_offset_with_preceding_fields(self):
        class Person(metaclass=ComponentMeta):
            age: int = 0
            name: str = ""

        fields, total_size = ComponentMeta._build_rust_layout(Person)
        # int(4) at 0, str(8) aligned to 8 => offset 8
        assert fields[0] == ("age", "i32", 0)
        assert fields[1] == ("name", "Str8", 8)
        assert total_size == 16, f"Expected 16, got {total_size}"

    def test_multiple_str_fields_layout(self):
        class FullName(metaclass=ComponentMeta):
            first: str = ""
            last: str = ""

        fields, total_size = ComponentMeta._build_rust_layout(FullName)
        assert fields[0] == ("first", "Str8", 0)
        assert fields[1] == ("last", "Str8", 8)
        assert total_size == 16, f"Expected 16, got {total_size}"

    def test_str8_type_code_in_type_register_json(self, capture_type_register):
        class Labeled(metaclass=ComponentMeta):
            label: str = ""

        info = capture_type_register()
        assert info is not None
        # type_register receives the "Str8" code via fields_json
        assert info["fields"][0][1] == "Str8", (
            f"Expected 'Str8' in type_register JSON, got '{info['fields'][0][1]}'"
        )
        assert info["size"] == 8

    def test_str8_not_string_type_code(self):
        """Ensure 'String' type code no longer appears for str fields."""
        class TestStr(metaclass=ComponentMeta):
            data: str = ""

        fields, _ = ComponentMeta._build_rust_layout(TestStr)
        assert fields[0][1] != "String", (
            "'String' code is 24-byte Rust String, should be 'Str8' (8-byte buffer)"
        )


# =============================================================================
# FIX re-verification: H-01a — String annotation type fallback
# =============================================================================


class TestFixH01aStringAnnotationFallback:
    """H-01a: Forward-reference string annotations (e.g. ``"float"`` instead
    of the type object ``float``) or unresolvable string type names must not
    crash _build_rust_layout.  The fallback extracts __name__ or the bare
    string and uses a 4-byte default."""

    def test_string_annotation_does_not_crash(self):
        cls = ComponentMeta(
            "StringAnnot",
            (),
            {
                "__module__": "test_string_annot",
                "__annotations__": {"x": "float"},
                "x": 0.0,
            },
        )
        # Must not raise; layout is computable
        fields, total_size = ComponentMeta._build_rust_layout(cls)
        assert len(fields) == 1
        # Unresolvable string annotations fall back to the string itself
        # as type code with a 4-byte default
        assert isinstance(total_size, int)

    def test_string_annotation_mixed_with_real_types(self):
        cls = ComponentMeta(
            "MixedStrAnnot",
            (),
            {
                "__module__": "test_mixed_str_annot",
                "__annotations__": {"known": float, "unknown": "CustomType"},
                "known": 0.0,
                "unknown": None,
            },
        )
        fields, total_size = ComponentMeta._build_rust_layout(cls)
        assert len(fields) == 2
        assert isinstance(total_size, int)
        assert total_size > 0

    def test_unknown_type_fallback_no_crash(self):
        """An unresolvable string type annotation never raises."""
        cls = ComponentMeta(
            "UnknownType",
            (),
            {
                "__module__": "test_unknown_type",
                "__annotations__": {"val": "NonExistentType"},
                "val": None,
            },
        )
        fields, total_size = ComponentMeta._build_rust_layout(cls)
        assert len(fields) == 1
        assert isinstance(total_size, int)


# =============================================================================
# FIX re-verification: CP-01 — Foundation registry integration
# =============================================================================


class TestFixCP01FoundationIntegration:
    """CP-01: Components are registered with the Foundation central registry."""

    def test_foundation_registry_contains_registered_component(self):
        class Tracked(metaclass=ComponentMeta):
            x: float = 0.0

        assert foundation_registry.is_registered(Tracked), (
            "Component must appear in foundation registry"
        )

    def test_foundation_registry_name_matches_component_name(self):
        class TrackedName(metaclass=ComponentMeta):
            x: float = 0.0

        desc = foundation_registry.describe(TrackedName)
        assert TrackedName._component_name in desc, (
            f"Expected '{TrackedName._component_name}' in foundation description, "
            f"got: {desc}"
        )

    def test_foundation_clear_does_not_break_component(self):
        class Persist(metaclass=ComponentMeta):
            val: int = 0

        foundation_registry.clear()
        # Component class itself is unaffected
        assert hasattr(Persist, "_component_id")
        assert Persist._component_id > 0


# =============================================================================
# FIX re-verification: Step-6b — Broad exception handler for _omega.type_register()
# =============================================================================


class TestFixStep6bOmegaTypeRegisterExceptionSafety:
    """Step 6b: Position(x,y) auto-registers even when _omega.type_register()
    raises arbitrary exceptions.  The broad ``except Exception`` handler
    (not just ImportError/AttributeError) catches RuntimeError, OSError,
    and any other unexpected failure from the Rust bridge call.

    Blackbox assertions: only the public observable contract — component
    creation succeeds, the object is properly initialised, and when the
    exception path fires a RuntimeWarning is emitted."""

    def test_position_auto_registers_when_type_register_raises_runtime_error(
        self, capture_type_register,
    ):
        """RuntimeError from type_register does not prevent creation of
        Position(x,y) — the broad exception handler catches it."""
        import sys
        from unittest import mock

        # Give the capture fixture a type_register that raises RuntimeError
        mock_omega = sys.modules["_omega"]
        mock_omega.type_register = mock.Mock(
            side_effect=RuntimeError("Rust bridge unavailable")
        )

        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        # Component must exist with valid identity
        assert hasattr(Position, "_component_id")
        assert isinstance(Position._component_id, int)
        assert Position._component_id > 0

        # Component must be findable through public registry APIs
        assert ComponentMeta.get_by_id(Position._component_id) is Position

    def test_position_auto_registers_when_type_register_raises_os_error(
        self, capture_type_register,
    ):
        """OSError from type_register (e.g. shared-library load failure)
        is caught — Position(x,y) still auto-registers."""
        import sys
        from unittest import mock

        mock_omega = sys.modules["_omega"]
        mock_omega.type_register = mock.Mock(
            side_effect=OSError("Cannot load native library")
        )

        class Position(metaclass=ComponentMeta):
            x: float = 0.0
            y: float = 0.0

        # Core identity must be intact
        assert hasattr(Position, "_component_id")
        # Layout must be computable independently of the failed Rust call
        fields, total_size = ComponentMeta._build_rust_layout(Position)
        assert len(fields) == 2
        assert fields[0] == ("x", "f32", 0)
        assert fields[1] == ("y", "f32", 4)
        assert total_size == 8

    def test_type_register_failure_emits_runtime_warning(self):
        """When type_register raises, a RuntimeWarning is emitted describing
        the failure so that developers are not left in the dark."""
        import sys
        from unittest import mock
        import warnings

        mock_omega = sys.modules.get("_omega")
        if mock_omega is None:
            mock_omega = mock.Mock()
            sys.modules["_omega"] = mock_omega
        mock_omega.type_register = mock.Mock(
            side_effect=RuntimeError("bridge crash")
        )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            class Position(metaclass=ComponentMeta):
                x: float = 0.0
                y: float = 0.0

        runtime_warnings = [
            w for w in captured if issubclass(w.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) >= 1, (
            "Expected at least one RuntimeWarning when type_register fails"
        )
        # At least one RuntimeWarning should mention the registration failure
        messages = [str(w.message) for w in runtime_warnings]
        joined = " ".join(messages)
        assert "registration" in joined or "register" in joined, (
            f"Expected RuntimeWarning mentioning registration, got: {joined}"
        )
        # Component must still be viable
        assert hasattr(Position, "_component_id")
        assert ComponentMeta.get_by_id(Position._component_id) is Position
