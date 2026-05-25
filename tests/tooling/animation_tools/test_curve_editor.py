"""Tests for animation curve editor with tangent editing and easing."""

import math
import pytest

from engine.tooling.animation_tools.curve_editor import (
    AnimationCurve,
    BezierCurve,
    CurveEditor,
    CurveKey,
    CurveSelection,
    CurveType,
    EasingFunction,
    EasingType,
    HermiteCurve,
    LinearCurve,
    SteppedCurve,
    TangentHandle,
    TangentMode,
)


# =============================================================================
# TANGENT HANDLE TESTS
# =============================================================================


class TestTangentHandle:
    def test_basic_handle(self):
        handle = TangentHandle(x=1.0, y=2.0)
        assert handle.x == 1.0
        assert handle.y == 2.0

    def test_slope(self):
        handle = TangentHandle(x=2.0, y=4.0)
        assert handle.slope == 2.0

    def test_slope_zero_x(self):
        handle = TangentHandle(x=0.0, y=1.0)
        assert handle.slope == 0.0

    def test_length(self):
        handle = TangentHandle(x=3.0, y=4.0)
        assert handle.length == 5.0

    def test_normalize(self):
        handle = TangentHandle(x=3.0, y=4.0)
        normalized = handle.normalize()
        assert abs(normalized.length - 1.0) < 0.001

    def test_scale(self):
        handle = TangentHandle(x=1.0, y=2.0)
        scaled = handle.scale(2.0)
        assert scaled.x == 2.0
        assert scaled.y == 4.0

    def test_copy(self):
        handle = TangentHandle(x=1.0, y=2.0, weight=0.5)
        copy = handle.copy()
        assert copy.x == handle.x
        assert copy is not handle


# =============================================================================
# CURVE KEY TESTS
# =============================================================================


class TestCurveKey:
    def test_basic_key(self):
        key = CurveKey(time=1.0, value=5.0)
        assert key.time == 1.0
        assert key.value == 5.0
        assert key.tangent_mode == TangentMode.AUTO

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="time must be >= 0"):
            CurveKey(time=-1.0, value=0.0)

    def test_set_flat_tangents(self):
        key = CurveKey(time=1.0, value=5.0)
        key.set_flat_tangents()
        assert key.tangent_mode == TangentMode.FLAT
        assert key.tangent_in.y == 0.0
        assert key.tangent_out.y == 0.0

    def test_set_linear_tangents(self):
        prev_key = CurveKey(time=0.0, value=0.0)
        next_key = CurveKey(time=2.0, value=10.0)
        key = CurveKey(time=1.0, value=5.0)
        key.set_linear_tangents(prev_key, next_key)
        assert key.tangent_mode == TangentMode.LINEAR

    def test_copy_key(self):
        key = CurveKey(
            time=1.0,
            value=5.0,
            tangent_mode=TangentMode.FREE,
            interpolation=CurveType.BEZIER,
        )
        copy = key.copy()
        assert copy.time == key.time
        assert copy.tangent_mode == key.tangent_mode
        assert copy is not key


# =============================================================================
# EASING FUNCTION TESTS
# =============================================================================


class TestEasingFunction:
    def test_linear(self):
        assert EasingFunction.linear(0.0) == 0.0
        assert EasingFunction.linear(0.5) == 0.5
        assert EasingFunction.linear(1.0) == 1.0

    def test_ease_in(self):
        assert EasingFunction.ease_in(0.0) == 0.0
        assert EasingFunction.ease_in(1.0) == 1.0
        # Ease in is slower at start
        assert EasingFunction.ease_in(0.5) < 0.5

    def test_ease_out(self):
        assert EasingFunction.ease_out(0.0) == 0.0
        assert EasingFunction.ease_out(1.0) == 1.0
        # Ease out is faster at start
        assert EasingFunction.ease_out(0.5) > 0.5

    def test_ease_in_out(self):
        assert EasingFunction.ease_in_out(0.0) == 0.0
        assert EasingFunction.ease_in_out(1.0) == 1.0
        assert abs(EasingFunction.ease_in_out(0.5) - 0.5) < 0.001

    def test_bounce_out(self):
        assert EasingFunction.bounce_out(0.0) == 0.0
        assert EasingFunction.bounce_out(1.0) == 1.0

    def test_elastic_out(self):
        assert EasingFunction.elastic_out(0.0) == 0.0
        assert EasingFunction.elastic_out(1.0) == 1.0

    def test_get_function(self):
        func = EasingFunction.get_function(EasingType.CUBIC_IN)
        assert func == EasingFunction.cubic_in
        assert func(0.5) == EasingFunction.cubic_in(0.5)

    def test_all_easing_types_bounds(self):
        # All easing functions should return 0 at t=0 and 1 at t=1
        for easing_type in EasingType:
            func = EasingFunction.get_function(easing_type)
            assert abs(func(0.0)) < 0.01, f"{easing_type} failed at t=0"
            assert abs(func(1.0) - 1.0) < 0.01, f"{easing_type} failed at t=1"


# =============================================================================
# LINEAR CURVE TESTS
# =============================================================================


class TestLinearCurve:
    def test_basic_curve(self):
        curve = LinearCurve("Test")
        assert curve.name == "Test"
        assert curve.curve_type == CurveType.LINEAR
        assert curve.key_count == 0

    def test_add_keys(self):
        curve = LinearCurve()
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 10.0)
        assert curve.key_count == 2

    def test_evaluate_linear(self):
        curve = LinearCurve()
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 10.0)
        assert abs(curve.evaluate(0.5) - 5.0) < 0.001

    def test_evaluate_before_first_key(self):
        curve = LinearCurve()
        curve.add_key_at(1.0, 5.0)
        assert curve.evaluate(0.0) == 5.0

    def test_evaluate_after_last_key(self):
        curve = LinearCurve()
        curve.add_key_at(0.0, 5.0)
        assert curve.evaluate(10.0) == 5.0

    def test_evaluate_empty_curve(self):
        curve = LinearCurve()
        assert curve.evaluate(0.5) == 0.0

    def test_remove_key(self):
        curve = LinearCurve()
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 10.0)
        assert curve.remove_key(0)
        assert curve.key_count == 1


# =============================================================================
# STEPPED CURVE TESTS
# =============================================================================


class TestSteppedCurve:
    def test_basic_curve(self):
        curve = SteppedCurve()
        assert curve.curve_type == CurveType.STEPPED

    def test_evaluate_stepped(self):
        curve = SteppedCurve()
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 10.0)
        # Stepped should return previous key value
        assert curve.evaluate(0.5) == 0.0
        assert curve.evaluate(1.0) == 10.0
        assert curve.evaluate(1.5) == 10.0


# =============================================================================
# BEZIER CURVE TESTS
# =============================================================================


class TestBezierCurve:
    def test_basic_curve(self):
        curve = BezierCurve()
        assert curve.curve_type == CurveType.BEZIER

    def test_evaluate_bezier(self):
        curve = BezierCurve()
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 10.0)
        # Bezier should interpolate smoothly
        mid = curve.evaluate(0.5)
        assert 0.0 < mid < 10.0


# =============================================================================
# HERMITE CURVE TESTS
# =============================================================================


class TestHermiteCurve:
    def test_basic_curve(self):
        curve = HermiteCurve()
        assert curve.curve_type == CurveType.HERMITE

    def test_evaluate_hermite(self):
        curve = HermiteCurve()
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 10.0)
        mid = curve.evaluate(0.5)
        assert 0.0 < mid < 10.0


# =============================================================================
# CURVE SELECTION TESTS
# =============================================================================


class TestCurveSelection:
    def test_basic_selection(self):
        sel = CurveSelection()
        assert not sel.has_selection
        assert sel.curve_index == -1

    def test_add_key(self):
        sel = CurveSelection()
        sel.curve_index = 0
        sel.add_key(0)
        sel.add_key(1)
        assert len(sel.key_indices) == 2
        assert sel.has_selection

    def test_remove_key(self):
        sel = CurveSelection()
        sel.add_key(0)
        sel.remove_key(0)
        assert len(sel.key_indices) == 0

    def test_clear(self):
        sel = CurveSelection(curve_index=0, key_indices=[0, 1])
        sel.clear()
        assert sel.curve_index == -1
        assert len(sel.key_indices) == 0


# =============================================================================
# CURVE EDITOR TESTS
# =============================================================================


class TestCurveEditor:
    def test_basic_editor(self):
        editor = CurveEditor()
        assert editor.curve_count == 0

    def test_add_curve(self):
        editor = CurveEditor()
        curve = LinearCurve("Test")
        idx = editor.add_curve(curve)
        assert idx == 0
        assert editor.curve_count == 1

    def test_remove_curve(self):
        editor = CurveEditor()
        curve = LinearCurve("Test")
        editor.add_curve(curve)
        assert editor.remove_curve(0)
        assert editor.curve_count == 0

    def test_get_curve(self):
        editor = CurveEditor()
        curve = LinearCurve("Test")
        editor.add_curve(curve)
        assert editor.get_curve(0) is curve
        assert editor.get_curve(5) is None

    def test_select_curve(self):
        editor = CurveEditor()
        editor.add_curve(LinearCurve("Test"))
        editor.select_curve(0)
        assert editor.selection.curve_index == 0

    def test_select_key(self):
        editor = CurveEditor()
        curve = LinearCurve("Test")
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 1.0)
        editor.add_curve(curve)
        editor.select_key(0, 0)
        assert editor.selection.curve_index == 0
        assert 0 in editor.selection.key_indices

    def test_add_key_at_time(self):
        editor = CurveEditor()
        curve = LinearCurve("Test")
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 10.0)
        editor.add_curve(curve)

        key = editor.add_key_at_time(0, 0.5)
        assert key is not None
        assert abs(key.value - 5.0) < 0.001
        assert curve.key_count == 3

    def test_delete_selected_keys(self):
        editor = CurveEditor()
        curve = LinearCurve("Test")
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 1.0)
        curve.add_key_at(2.0, 2.0)
        editor.add_curve(curve)
        editor.select_key(0, 1)
        deleted = editor.delete_selected_keys()
        assert deleted == 1
        assert curve.key_count == 2

    def test_set_tangent_mode(self):
        editor = CurveEditor()
        curve = BezierCurve("Test")
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 1.0)
        editor.add_curve(curve)
        editor.select_key(0, 0)
        editor.set_tangent_mode(TangentMode.FLAT)
        key = curve.get_key(0)
        assert key.tangent_mode == TangentMode.FLAT

    def test_flatten_tangents(self):
        editor = CurveEditor()
        curve = BezierCurve("Test")
        curve.add_key_at(1.0, 5.0)
        editor.add_curve(curve)
        editor.select_key(0, 0)
        editor.flatten_tangents()
        key = curve.get_key(0)
        assert key.tangent_mode == TangentMode.FLAT

    def test_frame_all(self):
        editor = CurveEditor()
        curve = LinearCurve("Test")
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(10.0, 100.0)
        editor.add_curve(curve)
        editor.frame_all()
        # View range should encompass all keys
        assert editor._view_range_x[0] <= 0.0
        assert editor._view_range_x[1] >= 10.0

    def test_normalize_curve(self):
        editor = CurveEditor()
        curve = LinearCurve("Test")
        curve.add_key_at(0.0, 10.0)
        curve.add_key_at(1.0, 30.0)
        editor.add_curve(curve)
        editor.normalize_curve(0)
        # Values should now be 0-1
        assert abs(curve.get_key(0).value - 0.0) < 0.001
        assert abs(curve.get_key(1).value - 1.0) < 0.001

    def test_bake_curve(self):
        editor = CurveEditor()
        curve = BezierCurve("Test")
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 10.0)
        editor.add_curve(curve)

        baked = editor.bake_curve(0, interval=0.1)
        assert baked is not None
        assert baked.key_count == 11  # 0.0 to 1.0 in 0.1 increments

    def test_apply_easing(self):
        editor = CurveEditor()
        curve = LinearCurve("Test")
        curve.add_key_at(0.0, 0.0)
        curve.add_key_at(1.0, 10.0)
        curve.add_key_at(0.5, 5.0)
        editor.add_curve(curve)
        editor.apply_easing(0, EasingType.EASE_IN)
        # Middle key value should be adjusted by easing
        mid_value = curve.get_key(1).value
        assert mid_value < 5.0  # Ease in is slower at start

    def test_view_range(self):
        editor = CurveEditor()
        editor.set_view_range(0.0, 10.0, -5.0, 5.0)
        assert editor._view_range_x == (0.0, 10.0)
        assert editor._view_range_y == (-5.0, 5.0)
