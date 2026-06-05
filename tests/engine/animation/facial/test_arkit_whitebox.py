"""
Whitebox Tests for ARKit 52 Blend Shape Compatibility.

Task: T3.7 ARKit Compatibility
Tests internal implementation details of ARKit blend shape support:
- ARKIT_BLEND_SHAPES constant validation
- Case-sensitive name matching
- validate_arkit_data() function behavior
- apply_arkit_data() function behavior
- create_arkit_compatible_set() factory function
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.animation.facial.blend_shapes import (
    ARKIT_BLEND_SHAPES,
    _ARKIT_BLEND_SHAPES_SET,
    BlendShape,
    BlendShapeSet,
    apply_arkit_data,
    create_arkit_compatible_set,
    validate_arkit_data,
)


# =============================================================================
# Test Constants and Fixtures
# =============================================================================

# Official ARKit blend shape names grouped by category for validation
EXPECTED_EYE_SHAPES = [
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
]

EXPECTED_BROW_SHAPES = [
    "browDownLeft", "browDownRight",
    "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
]

EXPECTED_JAW_SHAPES = [
    "jawForward", "jawLeft", "jawRight", "jawOpen",
]

EXPECTED_MOUTH_SHAPES = [
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
]

EXPECTED_CHEEK_NOSE_SHAPES = [
    "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
]

EXPECTED_TONGUE_SHAPE = ["tongueOut"]


@pytest.fixture
def arkit_blend_shape_set() -> BlendShapeSet:
    """Create a BlendShapeSet with ARKit shapes containing actual deltas."""
    vertex_count = 100
    base_vertices = np.zeros((vertex_count, 3), dtype=np.float32)

    shapes = {}
    for i, shape_name in enumerate(ARKIT_BLEND_SHAPES):
        # Create sparse deltas affecting vertices based on shape index
        affected_vertices = np.array([i % vertex_count], dtype=np.int32)
        deltas = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        shapes[shape_name] = BlendShape(
            name=shape_name,
            vertex_indices=affected_vertices,
            deltas=deltas,
        )

    return BlendShapeSet(
        name="arkit_test_set",
        base_vertices=base_vertices,
        blend_shapes=shapes,
    )


@pytest.fixture
def partial_arkit_set() -> BlendShapeSet:
    """Create a BlendShapeSet with only some ARKit shapes."""
    vertex_count = 50
    base_vertices = np.zeros((vertex_count, 3), dtype=np.float32)

    # Only include eye blink shapes
    shapes = {}
    for shape_name in ["eyeBlinkLeft", "eyeBlinkRight"]:
        shapes[shape_name] = BlendShape(
            name=shape_name,
            vertex_indices=np.array([0], dtype=np.int32),
            deltas=np.array([[0.5, 0.5, 0.0]], dtype=np.float32),
        )

    return BlendShapeSet(
        name="partial_arkit_set",
        base_vertices=base_vertices,
        blend_shapes=shapes,
    )


# =============================================================================
# Test 1: ARKIT_BLEND_SHAPES has exactly 52 entries
# =============================================================================

class TestArkitBlendShapesCount:
    """Test that ARKIT_BLEND_SHAPES contains exactly 52 shapes."""

    def test_arkit_blend_shapes_count_is_52(self) -> None:
        """ARKIT_BLEND_SHAPES must have exactly 52 entries per ARKit spec."""
        assert len(ARKIT_BLEND_SHAPES) == 52, (
            f"Expected 52 ARKit blend shapes, got {len(ARKIT_BLEND_SHAPES)}"
        )

    def test_arkit_blend_shapes_set_count_matches(self) -> None:
        """Internal set must also have 52 entries."""
        assert len(_ARKIT_BLEND_SHAPES_SET) == 52

    def test_no_duplicate_shapes(self) -> None:
        """All shape names must be unique."""
        assert len(ARKIT_BLEND_SHAPES) == len(set(ARKIT_BLEND_SHAPES)), (
            "Duplicate entries found in ARKIT_BLEND_SHAPES"
        )


# =============================================================================
# Test 2: All eye shapes present (14 shapes)
# =============================================================================

class TestArkitEyeShapes:
    """Test that all 14 ARKit eye blend shapes are present."""

    def test_eye_shapes_count(self) -> None:
        """There should be exactly 14 eye shapes."""
        eye_shapes_in_list = [s for s in ARKIT_BLEND_SHAPES if s.startswith("eye")]
        assert len(eye_shapes_in_list) == 14, (
            f"Expected 14 eye shapes, got {len(eye_shapes_in_list)}"
        )

    @pytest.mark.parametrize("shape_name", EXPECTED_EYE_SHAPES)
    def test_eye_shape_present(self, shape_name: str) -> None:
        """Each expected eye shape must be in ARKIT_BLEND_SHAPES."""
        assert shape_name in ARKIT_BLEND_SHAPES, (
            f"Missing eye shape: {shape_name}"
        )

    def test_eye_blink_shapes(self) -> None:
        """eyeBlinkLeft and eyeBlinkRight must be present."""
        assert "eyeBlinkLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "eyeBlinkRight" in _ARKIT_BLEND_SHAPES_SET

    def test_eye_look_down_shapes(self) -> None:
        """eyeLookDownLeft and eyeLookDownRight must be present."""
        assert "eyeLookDownLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "eyeLookDownRight" in _ARKIT_BLEND_SHAPES_SET

    def test_eye_look_in_shapes(self) -> None:
        """eyeLookInLeft and eyeLookInRight must be present."""
        assert "eyeLookInLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "eyeLookInRight" in _ARKIT_BLEND_SHAPES_SET

    def test_eye_look_out_shapes(self) -> None:
        """eyeLookOutLeft and eyeLookOutRight must be present."""
        assert "eyeLookOutLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "eyeLookOutRight" in _ARKIT_BLEND_SHAPES_SET

    def test_eye_look_up_shapes(self) -> None:
        """eyeLookUpLeft and eyeLookUpRight must be present."""
        assert "eyeLookUpLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "eyeLookUpRight" in _ARKIT_BLEND_SHAPES_SET

    def test_eye_squint_shapes(self) -> None:
        """eyeSquintLeft and eyeSquintRight must be present."""
        assert "eyeSquintLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "eyeSquintRight" in _ARKIT_BLEND_SHAPES_SET

    def test_eye_wide_shapes(self) -> None:
        """eyeWideLeft and eyeWideRight must be present."""
        assert "eyeWideLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "eyeWideRight" in _ARKIT_BLEND_SHAPES_SET


# =============================================================================
# Test 3: All brow shapes present (5 shapes)
# =============================================================================

class TestArkitBrowShapes:
    """Test that all 5 ARKit brow blend shapes are present."""

    def test_brow_shapes_count(self) -> None:
        """There should be exactly 5 brow shapes."""
        brow_shapes_in_list = [s for s in ARKIT_BLEND_SHAPES if s.startswith("brow")]
        assert len(brow_shapes_in_list) == 5, (
            f"Expected 5 brow shapes, got {len(brow_shapes_in_list)}"
        )

    @pytest.mark.parametrize("shape_name", EXPECTED_BROW_SHAPES)
    def test_brow_shape_present(self, shape_name: str) -> None:
        """Each expected brow shape must be in ARKIT_BLEND_SHAPES."""
        assert shape_name in ARKIT_BLEND_SHAPES, (
            f"Missing brow shape: {shape_name}"
        )

    def test_brow_down_shapes(self) -> None:
        """browDownLeft and browDownRight must be present."""
        assert "browDownLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "browDownRight" in _ARKIT_BLEND_SHAPES_SET

    def test_brow_inner_up_shape(self) -> None:
        """browInnerUp (singular, affects both brows) must be present."""
        assert "browInnerUp" in _ARKIT_BLEND_SHAPES_SET

    def test_brow_outer_up_shapes(self) -> None:
        """browOuterUpLeft and browOuterUpRight must be present."""
        assert "browOuterUpLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "browOuterUpRight" in _ARKIT_BLEND_SHAPES_SET


# =============================================================================
# Test 4: All jaw shapes present (4 shapes)
# =============================================================================

class TestArkitJawShapes:
    """Test that all 4 ARKit jaw blend shapes are present."""

    def test_jaw_shapes_count(self) -> None:
        """There should be exactly 4 jaw shapes."""
        jaw_shapes_in_list = [s for s in ARKIT_BLEND_SHAPES if s.startswith("jaw")]
        assert len(jaw_shapes_in_list) == 4, (
            f"Expected 4 jaw shapes, got {len(jaw_shapes_in_list)}"
        )

    @pytest.mark.parametrize("shape_name", EXPECTED_JAW_SHAPES)
    def test_jaw_shape_present(self, shape_name: str) -> None:
        """Each expected jaw shape must be in ARKIT_BLEND_SHAPES."""
        assert shape_name in ARKIT_BLEND_SHAPES, (
            f"Missing jaw shape: {shape_name}"
        )

    def test_jaw_forward_shape(self) -> None:
        """jawForward must be present."""
        assert "jawForward" in _ARKIT_BLEND_SHAPES_SET

    def test_jaw_left_shape(self) -> None:
        """jawLeft must be present."""
        assert "jawLeft" in _ARKIT_BLEND_SHAPES_SET

    def test_jaw_right_shape(self) -> None:
        """jawRight must be present."""
        assert "jawRight" in _ARKIT_BLEND_SHAPES_SET

    def test_jaw_open_shape(self) -> None:
        """jawOpen must be present."""
        assert "jawOpen" in _ARKIT_BLEND_SHAPES_SET


# =============================================================================
# Test 5: All mouth shapes present (23 shapes)
# =============================================================================

class TestArkitMouthShapes:
    """Test that all 23 ARKit mouth blend shapes are present."""

    def test_mouth_shapes_count(self) -> None:
        """There should be exactly 23 mouth shapes."""
        mouth_shapes_in_list = [s for s in ARKIT_BLEND_SHAPES if s.startswith("mouth")]
        assert len(mouth_shapes_in_list) == 23, (
            f"Expected 23 mouth shapes, got {len(mouth_shapes_in_list)}"
        )

    @pytest.mark.parametrize("shape_name", EXPECTED_MOUTH_SHAPES)
    def test_mouth_shape_present(self, shape_name: str) -> None:
        """Each expected mouth shape must be in ARKIT_BLEND_SHAPES."""
        assert shape_name in ARKIT_BLEND_SHAPES, (
            f"Missing mouth shape: {shape_name}"
        )

    def test_mouth_close_funnel_pucker(self) -> None:
        """mouthClose, mouthFunnel, mouthPucker must be present."""
        assert "mouthClose" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthFunnel" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthPucker" in _ARKIT_BLEND_SHAPES_SET

    def test_mouth_left_right(self) -> None:
        """mouthLeft and mouthRight must be present."""
        assert "mouthLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthRight" in _ARKIT_BLEND_SHAPES_SET

    def test_mouth_smile_shapes(self) -> None:
        """mouthSmileLeft and mouthSmileRight must be present."""
        assert "mouthSmileLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthSmileRight" in _ARKIT_BLEND_SHAPES_SET

    def test_mouth_frown_shapes(self) -> None:
        """mouthFrownLeft and mouthFrownRight must be present."""
        assert "mouthFrownLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthFrownRight" in _ARKIT_BLEND_SHAPES_SET

    def test_mouth_dimple_shapes(self) -> None:
        """mouthDimpleLeft and mouthDimpleRight must be present."""
        assert "mouthDimpleLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthDimpleRight" in _ARKIT_BLEND_SHAPES_SET

    def test_mouth_stretch_shapes(self) -> None:
        """mouthStretchLeft and mouthStretchRight must be present."""
        assert "mouthStretchLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthStretchRight" in _ARKIT_BLEND_SHAPES_SET

    def test_mouth_roll_shapes(self) -> None:
        """mouthRollLower and mouthRollUpper must be present."""
        assert "mouthRollLower" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthRollUpper" in _ARKIT_BLEND_SHAPES_SET

    def test_mouth_shrug_shapes(self) -> None:
        """mouthShrugLower and mouthShrugUpper must be present."""
        assert "mouthShrugLower" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthShrugUpper" in _ARKIT_BLEND_SHAPES_SET

    def test_mouth_press_shapes(self) -> None:
        """mouthPressLeft and mouthPressRight must be present."""
        assert "mouthPressLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthPressRight" in _ARKIT_BLEND_SHAPES_SET

    def test_mouth_lower_down_shapes(self) -> None:
        """mouthLowerDownLeft and mouthLowerDownRight must be present."""
        assert "mouthLowerDownLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthLowerDownRight" in _ARKIT_BLEND_SHAPES_SET

    def test_mouth_upper_up_shapes(self) -> None:
        """mouthUpperUpLeft and mouthUpperUpRight must be present."""
        assert "mouthUpperUpLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthUpperUpRight" in _ARKIT_BLEND_SHAPES_SET


# =============================================================================
# Test 6: All cheek/nose shapes present (5 shapes)
# =============================================================================

class TestArkitCheekNoseShapes:
    """Test that all 5 ARKit cheek and nose blend shapes are present."""

    def test_cheek_nose_shapes_count(self) -> None:
        """There should be exactly 5 cheek/nose shapes."""
        cheek_shapes = [s for s in ARKIT_BLEND_SHAPES if s.startswith("cheek")]
        nose_shapes = [s for s in ARKIT_BLEND_SHAPES if s.startswith("nose")]
        total = len(cheek_shapes) + len(nose_shapes)
        assert total == 5, (
            f"Expected 5 cheek/nose shapes, got {total}"
        )

    @pytest.mark.parametrize("shape_name", EXPECTED_CHEEK_NOSE_SHAPES)
    def test_cheek_nose_shape_present(self, shape_name: str) -> None:
        """Each expected cheek/nose shape must be in ARKIT_BLEND_SHAPES."""
        assert shape_name in ARKIT_BLEND_SHAPES, (
            f"Missing cheek/nose shape: {shape_name}"
        )

    def test_cheek_puff_shape(self) -> None:
        """cheekPuff must be present."""
        assert "cheekPuff" in _ARKIT_BLEND_SHAPES_SET

    def test_cheek_squint_shapes(self) -> None:
        """cheekSquintLeft and cheekSquintRight must be present."""
        assert "cheekSquintLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "cheekSquintRight" in _ARKIT_BLEND_SHAPES_SET

    def test_nose_sneer_shapes(self) -> None:
        """noseSneerLeft and noseSneerRight must be present."""
        assert "noseSneerLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "noseSneerRight" in _ARKIT_BLEND_SHAPES_SET


# =============================================================================
# Test 7: Tongue shape present
# =============================================================================

class TestArkitTongueShape:
    """Test that the tongue blend shape is present."""

    def test_tongue_shape_count(self) -> None:
        """There should be exactly 1 tongue shape."""
        tongue_shapes = [s for s in ARKIT_BLEND_SHAPES if s.startswith("tongue")]
        assert len(tongue_shapes) == 1, (
            f"Expected 1 tongue shape, got {len(tongue_shapes)}"
        )

    def test_tongue_out_shape(self) -> None:
        """tongueOut must be present."""
        assert "tongueOut" in _ARKIT_BLEND_SHAPES_SET
        assert "tongueOut" in ARKIT_BLEND_SHAPES


# =============================================================================
# Test 8: Case-sensitivity
# =============================================================================

class TestArkitCaseSensitivity:
    """Test that ARKit blend shape names are case-sensitive."""

    def test_case_sensitive_eye_blink_left(self) -> None:
        """eyeBlinkLeft must be exact case; lowercase variant should not exist."""
        assert "eyeBlinkLeft" in _ARKIT_BLEND_SHAPES_SET
        assert "eyebinkleft" not in _ARKIT_BLEND_SHAPES_SET
        assert "EYEBLINKLEFT" not in _ARKIT_BLEND_SHAPES_SET
        assert "EyeBlinkLeft" not in _ARKIT_BLEND_SHAPES_SET

    def test_case_sensitive_mouth_smile_right(self) -> None:
        """mouthSmileRight must be exact case."""
        assert "mouthSmileRight" in _ARKIT_BLEND_SHAPES_SET
        assert "mouthsmileright" not in _ARKIT_BLEND_SHAPES_SET
        assert "MOUTHSMILERIGHT" not in _ARKIT_BLEND_SHAPES_SET

    def test_case_sensitive_jaw_open(self) -> None:
        """jawOpen must be exact case."""
        assert "jawOpen" in _ARKIT_BLEND_SHAPES_SET
        assert "jawopen" not in _ARKIT_BLEND_SHAPES_SET
        assert "JawOpen" not in _ARKIT_BLEND_SHAPES_SET

    def test_case_sensitive_brow_inner_up(self) -> None:
        """browInnerUp must be exact case."""
        assert "browInnerUp" in _ARKIT_BLEND_SHAPES_SET
        assert "browinnerup" not in _ARKIT_BLEND_SHAPES_SET
        assert "BrowInnerUp" not in _ARKIT_BLEND_SHAPES_SET

    def test_validate_arkit_data_case_sensitive(self) -> None:
        """validate_arkit_data must reject incorrect case."""
        # Correct case should pass
        assert validate_arkit_data({"eyeBlinkLeft": 0.5}) is True

        # Incorrect case should fail
        assert validate_arkit_data({"eyebinkleft": 0.5}) is False
        assert validate_arkit_data({"EYEBLINKLEFT": 0.5}) is False

    @pytest.mark.parametrize("shape_name", ARKIT_BLEND_SHAPES)
    def test_all_shapes_camel_case(self, shape_name: str) -> None:
        """All ARKit shapes should follow camelCase convention."""
        # First character should be lowercase
        assert shape_name[0].islower(), f"{shape_name} should start with lowercase"
        # Should not be all lowercase or all uppercase
        assert shape_name != shape_name.lower() or shape_name in ("tongueOut",), (
            f"{shape_name} should have mixed case"
        )


# =============================================================================
# Test 9: validate_arkit_data() function behavior
# =============================================================================

class TestValidateArkitData:
    """Test validate_arkit_data() function returns True/False correctly."""

    def test_validate_empty_data(self) -> None:
        """Empty dictionary should be valid."""
        assert validate_arkit_data({}) is True

    def test_validate_single_valid_shape(self) -> None:
        """Single valid ARKit shape should be valid."""
        assert validate_arkit_data({"eyeBlinkLeft": 0.5}) is True

    def test_validate_multiple_valid_shapes(self) -> None:
        """Multiple valid ARKit shapes should be valid."""
        data = {
            "eyeBlinkLeft": 0.8,
            "eyeBlinkRight": 0.8,
            "mouthSmileLeft": 0.5,
            "mouthSmileRight": 0.5,
        }
        assert validate_arkit_data(data) is True

    def test_validate_all_52_shapes(self) -> None:
        """All 52 ARKit shapes should be valid together."""
        data = {name: 0.5 for name in ARKIT_BLEND_SHAPES}
        assert validate_arkit_data(data) is True

    def test_validate_single_invalid_shape(self) -> None:
        """Single invalid shape name should fail validation."""
        assert validate_arkit_data({"invalidShape": 0.5}) is False

    def test_validate_mixed_valid_invalid(self) -> None:
        """Mix of valid and invalid shapes should fail validation."""
        data = {
            "eyeBlinkLeft": 0.5,  # valid
            "customShape": 0.3,   # invalid
        }
        assert validate_arkit_data(data) is False

    def test_validate_with_zero_weight(self) -> None:
        """Zero weight should still be valid if name is valid."""
        assert validate_arkit_data({"eyeBlinkLeft": 0.0}) is True

    def test_validate_with_negative_weight(self) -> None:
        """Negative weight should still be valid if name is valid (weight checked elsewhere)."""
        assert validate_arkit_data({"eyeBlinkLeft": -0.5}) is True

    def test_validate_with_weight_over_one(self) -> None:
        """Weight over 1.0 should still be valid if name is valid (weight clamped elsewhere)."""
        assert validate_arkit_data({"eyeBlinkLeft": 1.5}) is True

    def test_validate_typo_in_shape_name(self) -> None:
        """Common typos should fail validation."""
        # Missing 'Left' suffix
        assert validate_arkit_data({"eyeBlink": 0.5}) is False
        # Wrong capitalization
        assert validate_arkit_data({"eyeblinkLeft": 0.5}) is False
        # Extra character
        assert validate_arkit_data({"eyeBlinkLeftt": 0.5}) is False


# =============================================================================
# Test 10: apply_arkit_data() function behavior
# =============================================================================

class TestApplyArkitData:
    """Test apply_arkit_data() applies weights to BlendShapeSet correctly."""

    def test_apply_returns_numpy_array(self, arkit_blend_shape_set: BlendShapeSet) -> None:
        """apply_arkit_data should return a numpy array."""
        arkit_data = {"eyeBlinkLeft": 0.5}
        result = apply_arkit_data(arkit_blend_shape_set, arkit_data)
        assert isinstance(result, np.ndarray)

    def test_apply_returns_correct_shape(self, arkit_blend_shape_set: BlendShapeSet) -> None:
        """Result should have same shape as base vertices."""
        arkit_data = {"eyeBlinkLeft": 0.5}
        result = apply_arkit_data(arkit_blend_shape_set, arkit_data)
        assert result.shape == arkit_blend_shape_set.base_vertices.shape

    def test_apply_empty_data_returns_base(self, arkit_blend_shape_set: BlendShapeSet) -> None:
        """Empty ARKit data should return unmodified base vertices."""
        result = apply_arkit_data(arkit_blend_shape_set, {})
        np.testing.assert_array_almost_equal(result, arkit_blend_shape_set.base_vertices)

    def test_apply_modifies_correct_vertex(self, arkit_blend_shape_set: BlendShapeSet) -> None:
        """ARKit data should modify the correct vertex based on blend shape."""
        # eyeBlinkLeft is at index 0 in ARKIT_BLEND_SHAPES, so affects vertex 0
        arkit_data = {"eyeBlinkLeft": 1.0}
        result = apply_arkit_data(arkit_blend_shape_set, arkit_data)

        # Vertex 0 should be modified (delta is [1.0, 0.0, 0.0])
        expected_vertex_0 = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result[0], expected_vertex_0)

    def test_apply_weight_scaling(self, arkit_blend_shape_set: BlendShapeSet) -> None:
        """Weights should scale the delta application."""
        arkit_data = {"eyeBlinkLeft": 0.5}
        result = apply_arkit_data(arkit_blend_shape_set, arkit_data)

        # Vertex 0 should be modified at half strength
        expected_vertex_0 = np.array([0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result[0], expected_vertex_0)

    def test_apply_clamps_weight_max(self, arkit_blend_shape_set: BlendShapeSet) -> None:
        """Weights above 1.0 should be clamped to 1.0."""
        arkit_data = {"eyeBlinkLeft": 2.0}
        result = apply_arkit_data(arkit_blend_shape_set, arkit_data)

        # Weight clamped to 1.0, so delta should be full strength
        expected_vertex_0 = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result[0], expected_vertex_0)

    def test_apply_clamps_weight_min(self, arkit_blend_shape_set: BlendShapeSet) -> None:
        """Weights below 0.0 should be clamped to 0.0."""
        arkit_data = {"eyeBlinkLeft": -0.5}
        result = apply_arkit_data(arkit_blend_shape_set, arkit_data)

        # Weight clamped to 0.0, so no modification
        expected_vertex_0 = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result[0], expected_vertex_0)

    def test_apply_ignores_invalid_shapes(self, arkit_blend_shape_set: BlendShapeSet) -> None:
        """Invalid ARKit shape names should be silently ignored."""
        arkit_data = {
            "eyeBlinkLeft": 1.0,      # valid
            "invalidShape": 1.0,       # invalid - ignored
        }
        result = apply_arkit_data(arkit_blend_shape_set, arkit_data)

        # Only eyeBlinkLeft should be applied
        expected_vertex_0 = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result[0], expected_vertex_0)

    def test_apply_ignores_shapes_not_in_set(self, partial_arkit_set: BlendShapeSet) -> None:
        """ARKit shapes not in the BlendShapeSet should be ignored."""
        arkit_data = {
            "eyeBlinkLeft": 1.0,    # in partial set
            "mouthSmileLeft": 1.0,  # valid ARKit but not in partial set
        }
        result = apply_arkit_data(partial_arkit_set, arkit_data)

        # Only eyeBlinkLeft should be applied
        expected_vertex_0 = np.array([0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result[0], expected_vertex_0)

    def test_apply_multiple_shapes(self, arkit_blend_shape_set: BlendShapeSet) -> None:
        """Multiple ARKit shapes should be applied additively."""
        # eyeBlinkLeft affects vertex 0, eyeBlinkRight affects vertex 1
        arkit_data = {
            "eyeBlinkLeft": 0.5,
            "eyeBlinkRight": 0.5,
        }
        result = apply_arkit_data(arkit_blend_shape_set, arkit_data)

        expected_vertex_0 = np.array([0.5, 0.0, 0.0])
        expected_vertex_1 = np.array([0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result[0], expected_vertex_0)
        np.testing.assert_array_almost_equal(result[1], expected_vertex_1)

    def test_apply_does_not_mutate_base_vertices(self, arkit_blend_shape_set: BlendShapeSet) -> None:
        """apply_arkit_data should not mutate the original base vertices."""
        original_base = arkit_blend_shape_set.base_vertices.copy()
        arkit_data = {"eyeBlinkLeft": 1.0}

        apply_arkit_data(arkit_blend_shape_set, arkit_data)

        np.testing.assert_array_equal(
            arkit_blend_shape_set.base_vertices,
            original_base
        )


# =============================================================================
# Additional: create_arkit_compatible_set() tests
# =============================================================================

class TestCreateArkitCompatibleSet:
    """Test the create_arkit_compatible_set() factory function."""

    def test_creates_blend_shape_set(self) -> None:
        """Should create a BlendShapeSet instance."""
        result = create_arkit_compatible_set("test", 100)
        assert isinstance(result, BlendShapeSet)

    def test_set_has_correct_name(self) -> None:
        """Created set should have the specified name."""
        result = create_arkit_compatible_set("my_face", 100)
        assert result.name == "my_face"

    def test_set_has_correct_vertex_count(self) -> None:
        """Created set should have the specified vertex count."""
        result = create_arkit_compatible_set("test", 500)
        assert result.vertex_count == 500
        assert result.base_vertices.shape == (500, 3)

    def test_set_has_52_shapes(self) -> None:
        """Created set should have all 52 ARKit shapes."""
        result = create_arkit_compatible_set("test", 100)
        assert result.shape_count == 52

    def test_set_has_all_arkit_shapes(self) -> None:
        """Created set should contain every ARKit blend shape name."""
        result = create_arkit_compatible_set("test", 100)
        for shape_name in ARKIT_BLEND_SHAPES:
            assert result.has_shape(shape_name), f"Missing shape: {shape_name}"

    def test_shapes_are_empty_initially(self) -> None:
        """Created shapes should have no vertices/deltas initially."""
        result = create_arkit_compatible_set("test", 100)
        for shape_name in ARKIT_BLEND_SHAPES:
            shape = result.get_shape(shape_name)
            assert shape is not None
            assert shape.vertex_count == 0
            assert len(shape.deltas) == 0

    def test_base_vertices_are_zeroed(self) -> None:
        """Base vertices should be initialized to zeros."""
        result = create_arkit_compatible_set("test", 100)
        np.testing.assert_array_equal(
            result.base_vertices,
            np.zeros((100, 3), dtype=np.float32)
        )


# =============================================================================
# Additional: Internal set consistency
# =============================================================================

class TestInternalSetConsistency:
    """Test consistency between ARKIT_BLEND_SHAPES list and internal set."""

    def test_list_and_set_match(self) -> None:
        """The list and frozenset should contain the same elements."""
        list_set = set(ARKIT_BLEND_SHAPES)
        assert list_set == _ARKIT_BLEND_SHAPES_SET

    def test_frozenset_is_immutable(self) -> None:
        """Internal set should be a frozenset (immutable)."""
        assert isinstance(_ARKIT_BLEND_SHAPES_SET, frozenset)

    def test_list_is_ordered(self) -> None:
        """ARKIT_BLEND_SHAPES should be a list with consistent ordering."""
        assert isinstance(ARKIT_BLEND_SHAPES, list)
        # Eye shapes should come before jaw shapes in the list
        eye_blink_idx = ARKIT_BLEND_SHAPES.index("eyeBlinkLeft")
        jaw_open_idx = ARKIT_BLEND_SHAPES.index("jawOpen")
        assert eye_blink_idx < jaw_open_idx


# =============================================================================
# Full ARKit shape validation
# =============================================================================

class TestFullArkitShapeSet:
    """Comprehensive test validating all 52 shapes match ARKit SDK exactly."""

    # Complete list of all 52 ARKit blend shapes as defined by Apple
    OFFICIAL_ARKIT_SHAPES = [
        # Eyes (14)
        "eyeBlinkLeft", "eyeBlinkRight",
        "eyeLookDownLeft", "eyeLookDownRight",
        "eyeLookInLeft", "eyeLookInRight",
        "eyeLookOutLeft", "eyeLookOutRight",
        "eyeLookUpLeft", "eyeLookUpRight",
        "eyeSquintLeft", "eyeSquintRight",
        "eyeWideLeft", "eyeWideRight",
        # Jaw (4)
        "jawForward", "jawLeft", "jawRight", "jawOpen",
        # Mouth (23)
        "mouthClose", "mouthFunnel", "mouthPucker",
        "mouthLeft", "mouthRight",
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "mouthDimpleLeft", "mouthDimpleRight",
        "mouthStretchLeft", "mouthStretchRight",
        "mouthRollLower", "mouthRollUpper",
        "mouthShrugLower", "mouthShrugUpper",
        "mouthPressLeft", "mouthPressRight",
        "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthUpperUpLeft", "mouthUpperUpRight",
        # Brow (5)
        "browDownLeft", "browDownRight",
        "browInnerUp",
        "browOuterUpLeft", "browOuterUpRight",
        # Cheek/Nose (5)
        "cheekPuff",
        "cheekSquintLeft", "cheekSquintRight",
        "noseSneerLeft", "noseSneerRight",
        # Tongue (1)
        "tongueOut",
    ]

    def test_all_official_shapes_present(self) -> None:
        """Every official ARKit shape must be in ARKIT_BLEND_SHAPES."""
        for shape in self.OFFICIAL_ARKIT_SHAPES:
            assert shape in ARKIT_BLEND_SHAPES, f"Missing official ARKit shape: {shape}"

    def test_no_extra_shapes(self) -> None:
        """ARKIT_BLEND_SHAPES should not contain any extra shapes."""
        official_set = set(self.OFFICIAL_ARKIT_SHAPES)
        implementation_set = set(ARKIT_BLEND_SHAPES)
        extra = implementation_set - official_set
        assert len(extra) == 0, f"Extra shapes found: {extra}"

    def test_exact_shape_match(self) -> None:
        """Sets should be identical."""
        assert set(ARKIT_BLEND_SHAPES) == set(self.OFFICIAL_ARKIT_SHAPES)
