"""Tests for Phase 4 introspection functions: _collect_descriptor_steps,
decompose (class path), decompose_layered, and expand (class path)."""

from trinity.decorators.ops import (
    Op,
    Step,
    _collect_descriptor_steps,
    decompose,
    decompose_layered,
    expand,
    make_decorator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_descriptor(steps):
    """Return a simple object with a descriptor_steps property."""

    class _Desc:
        descriptor_steps = steps

    return _Desc()


def _make_descriptor_no_steps():
    """Descriptor that lacks descriptor_steps entirely."""
    return object()


TAG_A = Step(Op.TAG, {"key": "a"})
TAG_B = Step(Op.TAG, {"key": "b"})
HOOK_C = Step(Op.HOOK, {"event": "on_create"})
REG_D = Step(Op.REGISTER, {"registry": "R"})
TRACK_E = Step(Op.TRACK, {"field": "hp"})
VAL_F = Step(Op.VALIDATE, {"constraint": "positive"})


# =========================================================================
# _collect_descriptor_steps
# =========================================================================


class TestCollectDescriptorSteps:
    def test_no_field_descriptors(self):
        class Empty:
            pass

        assert _collect_descriptor_steps(Empty) == []

    def test_descriptors_with_steps(self):
        class Cls:
            _field_descriptors = {
                "x": _make_descriptor([TAG_A, TAG_B]),
                "y": _make_descriptor([HOOK_C]),
            }

        result = _collect_descriptor_steps(Cls)
        assert result == [TAG_A, TAG_B, HOOK_C]

    def test_descriptor_without_steps_attr(self):
        class Cls:
            _field_descriptors = {
                "x": _make_descriptor_no_steps(),
            }

        assert _collect_descriptor_steps(Cls) == []

    def test_mixed_descriptors(self):
        class Cls:
            _field_descriptors = {
                "a": _make_descriptor([TAG_A]),
                "b": _make_descriptor_no_steps(),
                "c": _make_descriptor([HOOK_C, REG_D]),
            }

        result = _collect_descriptor_steps(Cls)
        assert result == [TAG_A, HOOK_C, REG_D]


# =========================================================================
# decompose — class path
# =========================================================================


class TestDecomposeClass:
    def test_applied_steps_only(self):
        class Cls:
            _applied_steps = [TAG_A, TAG_B]

        assert decompose(Cls) == [TAG_A, TAG_B]

    def test_metaclass_steps_only(self):
        class Cls:
            _metaclass_steps = [HOOK_C]

        assert decompose(Cls) == [HOOK_C]

    def test_all_three_layers(self):
        class Cls:
            _applied_steps = [TAG_A]
            _metaclass_steps = [HOOK_C]
            _field_descriptors = {"f": _make_descriptor([REG_D])}

        result = decompose(Cls)
        assert result == [TAG_A, HOOK_C, REG_D]

    def test_exclude_metaclass(self):
        class Cls:
            _applied_steps = [TAG_A]
            _metaclass_steps = [HOOK_C]
            _field_descriptors = {"f": _make_descriptor([REG_D])}

        result = decompose(Cls, include_metaclass=False)
        assert result == [TAG_A, REG_D]

    def test_exclude_descriptors(self):
        class Cls:
            _applied_steps = [TAG_A]
            _metaclass_steps = [HOOK_C]
            _field_descriptors = {"f": _make_descriptor([REG_D])}

        result = decompose(Cls, include_descriptors=False)
        assert result == [TAG_A, HOOK_C]

    def test_empty_class(self):
        class Cls:
            pass

        assert decompose(Cls) == []


# =========================================================================
# decompose_layered
# =========================================================================


class TestDecomposeLayered:
    def test_non_class_target(self):
        dec = make_decorator(name="layered_dec", steps=[TAG_A, TAG_B])
        result = decompose_layered(dec)
        assert result["decorators"] == [TAG_A, TAG_B]
        assert result["metaclass"] == []
        assert result["descriptors"] == []

    def test_class_all_layers(self):
        class Cls:
            _applied_steps = [TAG_A]
            _metaclass_steps = [HOOK_C]
            _field_descriptors = {"f": _make_descriptor([REG_D])}

        result = decompose_layered(Cls)
        assert result["decorators"] == [TAG_A]
        assert result["metaclass"] == [HOOK_C]
        assert result["descriptors"] == [REG_D]

    def test_class_no_steps(self):
        class Cls:
            pass

        result = decompose_layered(Cls)
        assert set(result.keys()) == {"decorators", "metaclass", "descriptors"}
        assert result["decorators"] == []
        assert result["metaclass"] == []
        assert result["descriptors"] == []


# =========================================================================
# expand — class path
# =========================================================================


class TestExpandClass:
    def test_all_layers(self):
        class Cls:
            _applied_steps = [TAG_A]
            _metaclass_steps = [HOOK_C]
            _field_descriptors = {"f": _make_descriptor([REG_D])}

        result = expand(Cls)
        assert "[Decorators]" in result
        assert "[Metaclass]" in result
        assert "[Descriptors]" in result

    def test_single_layer(self):
        class Cls:
            _metaclass_steps = [HOOK_C]

        result = expand(Cls)
        assert "[Metaclass]" in result
        assert "[Decorators]" not in result
        assert "[Descriptors]" not in result

    def test_no_steps(self):
        class Cls:
            pass

        result = expand(Cls)
        assert "no steps defined" in result

    def test_decorator_target_flat(self):
        dec = make_decorator(name="expand_flat", steps=[TAG_A, TAG_B])
        result = expand(dec)
        # Flat format, no layer labels
        assert "[Decorators]" not in result
        assert "TAG" in result
