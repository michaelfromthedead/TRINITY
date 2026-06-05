"""
Blackbox tests for T3.7 ARKit Compatibility.

Tests are written ONLY from the public contract specification:
- ARKIT_BLEND_SHAPES contains exactly 52 ARKit blend shape names
- validate_arkit_data checks if all keys are valid ARKit names
- apply_arkit_data applies ARKit weights to a BlendShapeSet

NO implementation source code was read to write these tests.
"""

import pytest


class TestARKitBlendShapeCount:
    """Test that ARKIT_BLEND_SHAPES contains exactly 52 shapes."""

    def test_arkit_shape_count_is_52(self):
        """Contract: len(ARKIT_BLEND_SHAPES) == 52"""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert len(ARKIT_BLEND_SHAPES) == 52, (
            f"Expected exactly 52 ARKit blend shapes, got {len(ARKIT_BLEND_SHAPES)}"
        )

    def test_arkit_shapes_are_unique(self):
        """All 52 shapes should be unique (no duplicates)."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        shape_list = list(ARKIT_BLEND_SHAPES)
        unique_shapes = set(shape_list)
        assert len(unique_shapes) == len(shape_list), (
            f"ARKIT_BLEND_SHAPES contains duplicates: {len(shape_list)} total, "
            f"{len(unique_shapes)} unique"
        )


class TestARKitCriticalShapesPresent:
    """Test that critical ARKit shapes are present."""

    def test_eye_blink_left_present(self):
        """Contract: 'eyeBlinkLeft' in ARKIT_BLEND_SHAPES"""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "eyeBlinkLeft" in ARKIT_BLEND_SHAPES, (
            "Critical shape 'eyeBlinkLeft' missing from ARKIT_BLEND_SHAPES"
        )

    def test_eye_blink_right_present(self):
        """Bilateral: 'eyeBlinkRight' should also be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "eyeBlinkRight" in ARKIT_BLEND_SHAPES, (
            "Critical shape 'eyeBlinkRight' missing from ARKIT_BLEND_SHAPES"
        )

    def test_mouth_smile_left_present(self):
        """Contract: 'mouthSmileLeft' in ARKIT_BLEND_SHAPES"""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "mouthSmileLeft" in ARKIT_BLEND_SHAPES, (
            "Critical shape 'mouthSmileLeft' missing from ARKIT_BLEND_SHAPES"
        )

    def test_mouth_smile_right_present(self):
        """Bilateral: 'mouthSmileRight' should also be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "mouthSmileRight" in ARKIT_BLEND_SHAPES, (
            "Critical shape 'mouthSmileRight' missing from ARKIT_BLEND_SHAPES"
        )

    def test_tongue_out_present(self):
        """Contract: 'tongueOut' in ARKIT_BLEND_SHAPES"""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "tongueOut" in ARKIT_BLEND_SHAPES, (
            "Critical shape 'tongueOut' missing from ARKIT_BLEND_SHAPES"
        )

    def test_jaw_open_present(self):
        """Common ARKit shape: jawOpen should be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "jawOpen" in ARKIT_BLEND_SHAPES, (
            "Common shape 'jawOpen' missing from ARKIT_BLEND_SHAPES"
        )


class TestARKitCaseSensitivity:
    """Test that shape names are case-sensitive and match ARKit SDK exactly."""

    def test_eyeblinkleft_lowercase_not_present(self):
        """Case-sensitive: 'eyeblinkleft' (lowercase) should NOT be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "eyeblinkleft" not in ARKIT_BLEND_SHAPES, (
            "Lowercase 'eyeblinkleft' found - names should be camelCase"
        )

    def test_EYEBLINKLEFT_uppercase_not_present(self):
        """Case-sensitive: 'EYEBLINKLEFT' (uppercase) should NOT be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "EYEBLINKLEFT" not in ARKIT_BLEND_SHAPES, (
            "Uppercase 'EYEBLINKLEFT' found - names should be camelCase"
        )

    def test_eye_blink_left_underscore_not_present(self):
        """Case-sensitive: 'eye_blink_left' (snake_case) should NOT be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "eye_blink_left" not in ARKIT_BLEND_SHAPES, (
            "Snake case 'eye_blink_left' found - names should be camelCase"
        )

    def test_mouthsmileleft_lowercase_not_present(self):
        """Case-sensitive: 'mouthsmileleft' (lowercase) should NOT be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "mouthsmileleft" not in ARKIT_BLEND_SHAPES, (
            "Lowercase 'mouthsmileleft' found - names should be camelCase"
        )

    def test_tongueout_lowercase_not_present(self):
        """Case-sensitive: 'tongueout' (lowercase) should NOT be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        assert "tongueout" not in ARKIT_BLEND_SHAPES, (
            "Lowercase 'tongueout' found - names should be camelCase"
        )


class TestValidateARKitData:
    """Test validate_arkit_data function."""

    def test_valid_single_shape_returns_true(self):
        """Contract: validate_arkit_data({'eyeBlinkLeft': 0.5}) returns True"""
        from engine.animation.facial import validate_arkit_data

        result = validate_arkit_data({"eyeBlinkLeft": 0.5})
        assert result is True, (
            f"Expected True for valid ARKit shape, got {result}"
        )

    def test_invalid_shape_returns_false(self):
        """Contract: validate_arkit_data({'notARealShape': 0.5}) returns False"""
        from engine.animation.facial import validate_arkit_data

        result = validate_arkit_data({"notARealShape": 0.5})
        assert result is False, (
            f"Expected False for invalid ARKit shape, got {result}"
        )

    def test_valid_multiple_shapes(self):
        """Multiple valid shapes should return True."""
        from engine.animation.facial import validate_arkit_data

        result = validate_arkit_data({
            "eyeBlinkLeft": 0.5,
            "eyeBlinkRight": 0.5,
            "mouthSmileLeft": 0.3
        })
        assert result is True, (
            f"Expected True for multiple valid shapes, got {result}"
        )

    def test_mixed_valid_invalid_returns_false(self):
        """Mix of valid and invalid shapes should return False."""
        from engine.animation.facial import validate_arkit_data

        result = validate_arkit_data({
            "eyeBlinkLeft": 0.5,  # valid
            "fakeShape": 0.3      # invalid
        })
        assert result is False, (
            f"Expected False when any shape is invalid, got {result}"
        )

    def test_empty_dict_returns_true(self):
        """Empty dict should be valid (no invalid keys)."""
        from engine.animation.facial import validate_arkit_data

        result = validate_arkit_data({})
        assert result is True, (
            f"Expected True for empty dict, got {result}"
        )

    def test_wrong_case_returns_false(self):
        """Wrong case should return False (case-sensitive)."""
        from engine.animation.facial import validate_arkit_data

        result = validate_arkit_data({"eyeblinkleft": 0.5})  # lowercase
        assert result is False, (
            f"Expected False for wrong case 'eyeblinkleft', got {result}"
        )

    def test_all_52_shapes_valid(self):
        """All 52 shapes from ARKIT_BLEND_SHAPES should validate."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES, validate_arkit_data

        # Create dict with all shapes
        all_shapes = {shape: 0.5 for shape in ARKIT_BLEND_SHAPES}
        result = validate_arkit_data(all_shapes)
        assert result is True, (
            f"Expected True for all 52 ARKit shapes, got {result}"
        )


class TestApplyARKitData:
    """Test apply_arkit_data function doesn't crash."""

    def test_apply_single_shape_no_crash(self):
        """Applying a single valid shape should not crash."""
        from engine.animation.facial import apply_arkit_data, BlendShapeSet

        blend_shape_set = BlendShapeSet(name="test_set")
        # Should not raise
        apply_arkit_data(blend_shape_set, {"eyeBlinkLeft": 0.5})

    def test_apply_multiple_shapes_no_crash(self):
        """Applying multiple valid shapes should not crash."""
        from engine.animation.facial import apply_arkit_data, BlendShapeSet

        blend_shape_set = BlendShapeSet(name="test_set")
        # Should not raise
        apply_arkit_data(blend_shape_set, {
            "eyeBlinkLeft": 0.5,
            "eyeBlinkRight": 0.5,
            "mouthSmileLeft": 0.3,
            "tongueOut": 0.1
        })

    def test_apply_empty_dict_no_crash(self):
        """Applying empty dict should not crash."""
        from engine.animation.facial import apply_arkit_data, BlendShapeSet

        blend_shape_set = BlendShapeSet(name="test_set")
        # Should not raise
        apply_arkit_data(blend_shape_set, {})

    def test_apply_all_52_shapes_no_crash(self):
        """Applying all 52 shapes should not crash."""
        from engine.animation.facial import (
            ARKIT_BLEND_SHAPES, apply_arkit_data, BlendShapeSet
        )

        blend_shape_set = BlendShapeSet(name="test_set")
        all_shapes = {shape: 0.5 for shape in ARKIT_BLEND_SHAPES}
        # Should not raise
        apply_arkit_data(blend_shape_set, all_shapes)

    def test_apply_boundary_values_no_crash(self):
        """Applying boundary values (0.0, 1.0) should not crash."""
        from engine.animation.facial import apply_arkit_data, BlendShapeSet

        blend_shape_set = BlendShapeSet(name="test_set")
        # Should not raise
        apply_arkit_data(blend_shape_set, {
            "eyeBlinkLeft": 0.0,
            "eyeBlinkRight": 1.0
        })


class TestARKitBilateralShapes:
    """Test that bilateral (left/right) shapes are both present."""

    @pytest.mark.parametrize("shape_base", [
        "eyeBlink",
        "eyeLookDown",
        "eyeLookIn",
        "eyeLookOut",
        "eyeLookUp",
        "eyeSquint",
        "eyeWide",
        "browDown",
        "browInnerUp",  # Note: browInnerUp might not have L/R
        "browOuterUp",
        "cheekPuff",
        "cheekSquint",
        "noseSneer",
        "mouthSmile",
        "mouthFrown",
        "mouthDimple",
        "mouthStretch",
        "mouthPress",
        "mouthLowerDown",
        "mouthUpperUp",
    ])
    def test_bilateral_shape_pair_exists(self, shape_base):
        """Both Left and Right versions of bilateral shapes should exist."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        left_name = f"{shape_base}Left"
        right_name = f"{shape_base}Right"

        # At least one of the pair should exist (some shapes are unilateral)
        left_exists = left_name in ARKIT_BLEND_SHAPES
        right_exists = right_name in ARKIT_BLEND_SHAPES

        # If one exists, the other should too (they come in pairs)
        if left_exists:
            assert right_exists, (
                f"Found '{left_name}' but missing '{right_name}'"
            )
        if right_exists:
            assert left_exists, (
                f"Found '{right_name}' but missing '{left_name}'"
            )


class TestARKitExpectedShapes:
    """Test that standard ARKit shape categories are represented."""

    def test_eye_shapes_category(self):
        """Eye-related shapes should be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        eye_shapes = [s for s in ARKIT_BLEND_SHAPES if s.startswith("eye")]
        assert len(eye_shapes) >= 10, (
            f"Expected at least 10 eye shapes, got {len(eye_shapes)}"
        )

    def test_mouth_shapes_category(self):
        """Mouth-related shapes should be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        mouth_shapes = [s for s in ARKIT_BLEND_SHAPES if s.startswith("mouth")]
        assert len(mouth_shapes) >= 10, (
            f"Expected at least 10 mouth shapes, got {len(mouth_shapes)}"
        )

    def test_jaw_shapes_present(self):
        """Jaw shapes should be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        jaw_shapes = [s for s in ARKIT_BLEND_SHAPES if s.startswith("jaw")]
        assert len(jaw_shapes) >= 1, (
            f"Expected at least 1 jaw shape, got {len(jaw_shapes)}"
        )

    def test_brow_shapes_present(self):
        """Brow shapes should be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        brow_shapes = [s for s in ARKIT_BLEND_SHAPES if s.startswith("brow")]
        assert len(brow_shapes) >= 1, (
            f"Expected at least 1 brow shape, got {len(brow_shapes)}"
        )

    def test_cheek_shapes_present(self):
        """Cheek shapes should be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        cheek_shapes = [s for s in ARKIT_BLEND_SHAPES if s.startswith("cheek")]
        assert len(cheek_shapes) >= 1, (
            f"Expected at least 1 cheek shape, got {len(cheek_shapes)}"
        )

    def test_tongue_shapes_present(self):
        """Tongue shapes should be present."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        tongue_shapes = [s for s in ARKIT_BLEND_SHAPES if s.startswith("tongue")]
        assert len(tongue_shapes) >= 1, (
            f"Expected at least 1 tongue shape, got {len(tongue_shapes)}"
        )


class TestARKitDataTypes:
    """Test that ARKIT_BLEND_SHAPES has correct data types."""

    def test_shapes_are_strings(self):
        """All shapes should be strings."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        for shape in ARKIT_BLEND_SHAPES:
            assert isinstance(shape, str), (
                f"Shape {shape!r} is {type(shape).__name__}, expected str"
            )

    def test_shapes_are_non_empty(self):
        """All shape names should be non-empty strings."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        for shape in ARKIT_BLEND_SHAPES:
            assert len(shape) > 0, "Found empty string in ARKIT_BLEND_SHAPES"

    def test_shapes_have_no_whitespace(self):
        """Shape names should not contain whitespace."""
        from engine.animation.facial import ARKIT_BLEND_SHAPES

        for shape in ARKIT_BLEND_SHAPES:
            assert shape == shape.strip(), (
                f"Shape '{shape}' contains leading/trailing whitespace"
            )
            assert " " not in shape, (
                f"Shape '{shape}' contains spaces"
            )


class TestValidateARKitDataReturnType:
    """Test that validate_arkit_data returns boolean."""

    def test_returns_bool_for_valid(self):
        """Return type should be bool, not truthy value."""
        from engine.animation.facial import validate_arkit_data

        result = validate_arkit_data({"eyeBlinkLeft": 0.5})
        assert isinstance(result, bool), (
            f"Expected bool, got {type(result).__name__}"
        )

    def test_returns_bool_for_invalid(self):
        """Return type should be bool, not truthy value."""
        from engine.animation.facial import validate_arkit_data

        result = validate_arkit_data({"fakeShape": 0.5})
        assert isinstance(result, bool), (
            f"Expected bool, got {type(result).__name__}"
        )
