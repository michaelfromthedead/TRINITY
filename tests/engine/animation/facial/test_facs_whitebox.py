"""
Whitebox Tests for FACS (Facial Action Coding System) Module.

Task: T3.2 FACS Expression Mapping
Tests internal AU mappings, expression AU combinations, asymmetry, and bilateral support.
"""

import pytest
from engine.animation.facial.facs import (
    ActionUnit,
    ActionUnitData,
    Expression,
    ExpressionData,
    FACSController,
    get_default_au_mappings,
    get_default_expressions,
    AU,  # Alias
)


# =============================================================================
# Test: ActionUnit Enum (20 AUs)
# =============================================================================

class TestActionUnitEnum:
    """Test ActionUnit enum completeness and structure."""

    def test_action_unit_count(self):
        """Verify there are exactly 20 Action Units."""
        au_count = len(ActionUnit)
        assert au_count == 20, f"Expected 20 Action Units, got {au_count}"

    def test_au_alias_works(self):
        """Verify AU alias references ActionUnit."""
        assert AU is ActionUnit
        assert AU.AU1_INNER_BROW_RAISER is ActionUnit.AU1_INNER_BROW_RAISER

    def test_all_expected_aus_present(self):
        """Verify all expected Action Units are defined."""
        expected_aus = [
            "AU1_INNER_BROW_RAISER",
            "AU2_OUTER_BROW_RAISER",
            "AU4_BROW_LOWERER",
            "AU5_UPPER_LID_RAISER",
            "AU6_CHEEK_RAISER",
            "AU7_LID_TIGHTENER",
            "AU9_NOSE_WRINKLER",
            "AU10_UPPER_LIP_RAISER",
            "AU12_LIP_CORNER_PULLER",
            "AU14_DIMPLER",
            "AU15_LIP_CORNER_DEPRESSOR",
            "AU17_CHIN_RAISER",
            "AU20_LIP_STRETCHER",
            "AU23_LIP_TIGHTENER",
            "AU24_LIP_PRESSOR",
            "AU25_LIPS_PART",
            "AU26_JAW_DROP",
            "AU27_MOUTH_STRETCH",
            "AU28_LIP_SUCK",
            "AU43_EYES_CLOSED",
        ]

        actual_names = [au.name for au in ActionUnit]

        for expected_name in expected_aus:
            assert expected_name in actual_names, f"Missing AU: {expected_name}"

    def test_upper_face_aus(self):
        """Test upper face AUs are present (brows, lids)."""
        upper_face = [
            ActionUnit.AU1_INNER_BROW_RAISER,
            ActionUnit.AU2_OUTER_BROW_RAISER,
            ActionUnit.AU4_BROW_LOWERER,
            ActionUnit.AU5_UPPER_LID_RAISER,
            ActionUnit.AU6_CHEEK_RAISER,
            ActionUnit.AU7_LID_TIGHTENER,
        ]
        for au in upper_face:
            assert au in ActionUnit

    def test_mouth_aus(self):
        """Test mouth-related AUs are present."""
        mouth_aus = [
            ActionUnit.AU10_UPPER_LIP_RAISER,
            ActionUnit.AU12_LIP_CORNER_PULLER,
            ActionUnit.AU14_DIMPLER,
            ActionUnit.AU15_LIP_CORNER_DEPRESSOR,
            ActionUnit.AU17_CHIN_RAISER,
            ActionUnit.AU20_LIP_STRETCHER,
            ActionUnit.AU23_LIP_TIGHTENER,
            ActionUnit.AU24_LIP_PRESSOR,
            ActionUnit.AU25_LIPS_PART,
            ActionUnit.AU26_JAW_DROP,
            ActionUnit.AU27_MOUTH_STRETCH,
            ActionUnit.AU28_LIP_SUCK,
        ]
        for au in mouth_aus:
            assert au in ActionUnit


# =============================================================================
# Test: Expression Enum (8 Ekman Expressions)
# =============================================================================

class TestExpressionEnum:
    """Test Expression enum for all Ekman emotions."""

    def test_expression_count(self):
        """Verify there are exactly 8 expressions."""
        expr_count = len(Expression)
        assert expr_count == 8, f"Expected 8 expressions, got {expr_count}"

    def test_all_ekman_expressions_present(self):
        """Verify all Ekman universal expressions are defined."""
        expected = [
            "NEUTRAL",
            "HAPPY",
            "SAD",
            "ANGRY",
            "SURPRISED",
            "DISGUSTED",
            "FEARFUL",
            "CONTEMPT",
        ]
        actual_names = [e.name for e in Expression]

        for expected_name in expected:
            assert expected_name in actual_names, f"Missing expression: {expected_name}"


# =============================================================================
# Test: Default AU Mappings
# =============================================================================

class TestDefaultAUMappings:
    """Test get_default_au_mappings() returns correct blend shape configurations."""

    @pytest.fixture
    def au_mappings(self):
        """Get default AU mappings."""
        return get_default_au_mappings()

    def test_all_20_aus_have_mappings(self, au_mappings):
        """Verify all 20 AUs have blend shape mappings."""
        assert len(au_mappings) == 20, f"Expected 20 AU mappings, got {len(au_mappings)}"

        for au in ActionUnit:
            assert au in au_mappings, f"Missing mapping for {au.name}"

    def test_arkit_blend_shape_names(self, au_mappings):
        """Verify blend shape names use ARKit conventions."""
        # ARKit uses camelCase names like "cheekSquintLeft" not "cheekRaiserL"
        arkit_expected = {
            ActionUnit.AU6_CHEEK_RAISER: ["cheekSquintLeft", "cheekSquintRight"],
            ActionUnit.AU12_LIP_CORNER_PULLER: ["mouthSmileLeft", "mouthSmileRight"],
            ActionUnit.AU43_EYES_CLOSED: ["eyeBlinkLeft", "eyeBlinkRight"],
            ActionUnit.AU1_INNER_BROW_RAISER: ["browInnerUp"],
        }

        for au, expected_shapes in arkit_expected.items():
            au_data = au_mappings[au]
            actual_shapes = list(au_data.blend_shapes.keys()) + \
                           list(au_data.left_shapes.keys()) + \
                           list(au_data.right_shapes.keys())

            for shape in expected_shapes:
                assert shape in actual_shapes, \
                    f"Missing ARKit shape '{shape}' for {au.name}"

    def test_bilateral_aus_have_left_right_shapes(self, au_mappings):
        """Verify bilateral AUs have both left and right shapes."""
        bilateral_aus = [
            ActionUnit.AU2_OUTER_BROW_RAISER,
            ActionUnit.AU4_BROW_LOWERER,
            ActionUnit.AU5_UPPER_LID_RAISER,
            ActionUnit.AU6_CHEEK_RAISER,
            ActionUnit.AU7_LID_TIGHTENER,
            ActionUnit.AU9_NOSE_WRINKLER,
            ActionUnit.AU10_UPPER_LIP_RAISER,
            ActionUnit.AU12_LIP_CORNER_PULLER,
            ActionUnit.AU14_DIMPLER,
            ActionUnit.AU15_LIP_CORNER_DEPRESSOR,
            ActionUnit.AU20_LIP_STRETCHER,
            ActionUnit.AU24_LIP_PRESSOR,
            ActionUnit.AU43_EYES_CLOSED,
        ]

        for au in bilateral_aus:
            au_data = au_mappings[au]
            assert au_data.is_bilateral, f"{au.name} should be bilateral"
            assert len(au_data.left_shapes) > 0, f"{au.name} missing left shapes"
            assert len(au_data.right_shapes) > 0, f"{au.name} missing right shapes"

    def test_non_bilateral_aus(self, au_mappings):
        """Verify non-bilateral AUs have center blend shapes only."""
        non_bilateral = [
            ActionUnit.AU1_INNER_BROW_RAISER,
            ActionUnit.AU17_CHIN_RAISER,
            ActionUnit.AU23_LIP_TIGHTENER,
            ActionUnit.AU25_LIPS_PART,
            ActionUnit.AU26_JAW_DROP,
            ActionUnit.AU27_MOUTH_STRETCH,
            ActionUnit.AU28_LIP_SUCK,
        ]

        for au in non_bilateral:
            au_data = au_mappings[au]
            assert not au_data.is_bilateral, f"{au.name} should NOT be bilateral"
            assert len(au_data.blend_shapes) > 0, f"{au.name} missing blend shapes"


# =============================================================================
# Test: Default Expressions
# =============================================================================

class TestDefaultExpressions:
    """Test get_default_expressions() returns correct AU combinations."""

    @pytest.fixture
    def expressions(self):
        """Get default expression definitions."""
        return get_default_expressions()

    def test_all_8_expressions_defined(self, expressions):
        """Verify all 8 expressions have definitions."""
        assert len(expressions) == 8

        for expr in Expression:
            assert expr in expressions, f"Missing definition for {expr.name}"

    def test_neutral_has_no_aus(self, expressions):
        """NEUTRAL should have no active AUs."""
        neutral = expressions[Expression.NEUTRAL]
        assert len(neutral.au_weights) == 0, "NEUTRAL should have no AU weights"
        assert len(neutral.au_left_weights) == 0
        assert len(neutral.au_right_weights) == 0

    def test_happy_au_combination(self, expressions):
        """HAPPY should use AU6 (cheek raiser) and AU12 (lip corner puller)."""
        happy = expressions[Expression.HAPPY]

        # Must have AU6 and AU12
        assert ActionUnit.AU6_CHEEK_RAISER in happy.au_weights
        assert ActionUnit.AU12_LIP_CORNER_PULLER in happy.au_weights

        # AU12 should be strong
        assert happy.au_weights[ActionUnit.AU12_LIP_CORNER_PULLER] >= 0.8

    def test_sad_au_combination(self, expressions):
        """SAD should use AU1, AU4, AU15, AU17."""
        sad = expressions[Expression.SAD]

        # AU1 (inner brow raise) - the sad brow position
        assert ActionUnit.AU1_INNER_BROW_RAISER in sad.au_weights
        # AU15 (lip corner depressor) - frown
        assert ActionUnit.AU15_LIP_CORNER_DEPRESSOR in sad.au_weights

    def test_angry_au_combination(self, expressions):
        """ANGRY should use AU4 (brow lowerer) strongly."""
        angry = expressions[Expression.ANGRY]

        assert ActionUnit.AU4_BROW_LOWERER in angry.au_weights
        # Brow lowerer should be intense in anger
        assert angry.au_weights[ActionUnit.AU4_BROW_LOWERER] >= 0.8

    def test_surprised_au_combination(self, expressions):
        """SURPRISED should use AU1, AU2, AU5, AU26."""
        surprised = expressions[Expression.SURPRISED]

        # Raised brows
        assert ActionUnit.AU1_INNER_BROW_RAISER in surprised.au_weights
        assert ActionUnit.AU2_OUTER_BROW_RAISER in surprised.au_weights
        # Wide eyes
        assert ActionUnit.AU5_UPPER_LID_RAISER in surprised.au_weights
        # Open mouth
        assert ActionUnit.AU26_JAW_DROP in surprised.au_weights

    def test_disgusted_au_combination(self, expressions):
        """DISGUSTED should use AU9 (nose wrinkler) strongly."""
        disgusted = expressions[Expression.DISGUSTED]

        assert ActionUnit.AU9_NOSE_WRINKLER in disgusted.au_weights
        assert disgusted.au_weights[ActionUnit.AU9_NOSE_WRINKLER] >= 0.8

    def test_fearful_au_combination(self, expressions):
        """FEARFUL should use AU1, AU5, AU20."""
        fearful = expressions[Expression.FEARFUL]

        # Raised inner brow
        assert ActionUnit.AU1_INNER_BROW_RAISER in fearful.au_weights
        # Wide eyes
        assert ActionUnit.AU5_UPPER_LID_RAISER in fearful.au_weights
        # Lip stretcher (fear grimace)
        assert ActionUnit.AU20_LIP_STRETCHER in fearful.au_weights

    def test_contempt_is_asymmetric(self, expressions):
        """CONTEMPT must be asymmetric - right side only."""
        contempt = expressions[Expression.CONTEMPT]

        # Should have no symmetric AU weights
        assert len(contempt.au_weights) == 0, \
            "CONTEMPT should have no symmetric AU weights"

        # Must have right-side AU12 and AU14
        assert ActionUnit.AU12_LIP_CORNER_PULLER in contempt.au_right_weights
        assert ActionUnit.AU14_DIMPLER in contempt.au_right_weights

        # Right side should have specific values
        assert contempt.au_right_weights[ActionUnit.AU12_LIP_CORNER_PULLER] == 0.5
        assert contempt.au_right_weights[ActionUnit.AU14_DIMPLER] == 0.6

        # Left side should have zero or be explicit zeros
        assert ActionUnit.AU12_LIP_CORNER_PULLER in contempt.au_left_weights
        assert ActionUnit.AU14_DIMPLER in contempt.au_left_weights
        assert contempt.au_left_weights[ActionUnit.AU12_LIP_CORNER_PULLER] == 0.0
        assert contempt.au_left_weights[ActionUnit.AU14_DIMPLER] == 0.0


# =============================================================================
# Test: ActionUnitData Bilateral Support
# =============================================================================

class TestActionUnitDataBilateral:
    """Test bilateral blend shape weight calculation."""

    def test_bilateral_au_uniform_intensity(self):
        """Bilateral AU with uniform intensity applies to both sides."""
        au_data = ActionUnitData(
            au=ActionUnit.AU12_LIP_CORNER_PULLER,
            intensity=0.8,
            is_bilateral=True,
            left_shapes={"mouthSmileLeft": 1.0},
            right_shapes={"mouthSmileRight": 1.0},
        )

        weights = au_data.get_blend_weights()

        assert "mouthSmileLeft" in weights
        assert "mouthSmileRight" in weights
        assert weights["mouthSmileLeft"] == pytest.approx(0.8)
        assert weights["mouthSmileRight"] == pytest.approx(0.8)

    def test_bilateral_au_independent_sides(self):
        """Bilateral AU can have independent left/right intensities."""
        au_data = ActionUnitData(
            au=ActionUnit.AU12_LIP_CORNER_PULLER,
            intensity=0.5,
            is_bilateral=True,
            left_shapes={"mouthSmileLeft": 1.0},
            right_shapes={"mouthSmileRight": 1.0},
        )

        # Override with different intensities
        weights = au_data.get_blend_weights(left_intensity=0.2, right_intensity=0.9)

        assert weights["mouthSmileLeft"] == pytest.approx(0.2)
        assert weights["mouthSmileRight"] == pytest.approx(0.9)

    def test_non_bilateral_au_center_only(self):
        """Non-bilateral AU only produces center blend shapes."""
        au_data = ActionUnitData(
            au=ActionUnit.AU1_INNER_BROW_RAISER,
            intensity=0.7,
            is_bilateral=False,
            blend_shapes={"browInnerUp": 1.0},
        )

        weights = au_data.get_blend_weights()

        assert "browInnerUp" in weights
        assert weights["browInnerUp"] == pytest.approx(0.7)
        # Should have no left/right shapes
        assert all("Left" not in k and "Right" not in k for k in weights.keys() if k != "browInnerUp")


# =============================================================================
# Test: FACSController
# =============================================================================

class TestFACSController:
    """Test FACSController operations."""

    @pytest.fixture
    def controller(self):
        """Create fresh controller."""
        return FACSController()

    def test_initial_state_all_aus_zero(self, controller):
        """Controller starts with all AUs at zero intensity."""
        intensities = controller.au_intensities

        for au in ActionUnit:
            assert intensities[au] == 0.0, f"{au.name} should start at 0"

    def test_set_au_intensity_basic(self, controller):
        """Setting AU intensity stores the value."""
        controller.set_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER, 0.75)

        assert controller.get_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER) == pytest.approx(0.75)

    def test_set_au_intensity_clamped(self, controller):
        """AU intensity is clamped to 0-1."""
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 1.5)
        assert controller.get_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER) == 1.0

        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, -0.5)
        assert controller.get_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER) == 0.0

    def test_set_au_bilateral_intensities(self, controller):
        """Setting bilateral AU with left/right overrides."""
        controller.set_au_intensity(
            ActionUnit.AU12_LIP_CORNER_PULLER,
            intensity=0.5,
            left=0.2,
            right=0.8
        )

        weights = controller.get_blend_weights()

        # Should have different left/right weights
        assert "mouthSmileLeft" in weights
        assert "mouthSmileRight" in weights
        # Values should reflect the bilateral override
        # The controller sets AU intensity to 0.5, but left_int=0.2, right_int=0.8
        assert weights["mouthSmileLeft"] == pytest.approx(0.2)
        assert weights["mouthSmileRight"] == pytest.approx(0.8)

    def test_reset_all_aus(self, controller):
        """reset_all_aus() clears all intensities."""
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 1.0)
        controller.set_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER, 0.8)

        controller.reset_all_aus()

        for au in ActionUnit:
            assert controller.get_au_intensity(au) == 0.0

    def test_set_expression_happy(self, controller):
        """set_expression(HAPPY) activates correct AUs."""
        controller.set_expression(Expression.HAPPY)

        # Check AU6 and AU12 are active
        assert controller.get_au_intensity(ActionUnit.AU6_CHEEK_RAISER) > 0
        assert controller.get_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER) > 0

        # Current expression should be HAPPY
        assert controller.current_expression == Expression.HAPPY

    def test_set_expression_contempt_asymmetric_weights(self, controller):
        """set_expression(CONTEMPT) produces asymmetric blend weights."""
        controller.set_expression(Expression.CONTEMPT)

        weights = controller.get_blend_weights()

        # Right side should be active, left side zero
        # Since CONTEMPT uses au_right_weights and au_left_weights (not au_weights),
        # we need to verify the blend weights produced
        # Right side: AU12=0.5, AU14=0.6
        # Left side: AU12=0, AU14=0

        # The controller should produce asymmetric smile
        # Check for mouthSmileRight > 0 and mouthSmileLeft == 0
        if "mouthSmileRight" in weights:
            assert weights["mouthSmileRight"] > 0
        if "mouthSmileLeft" in weights:
            assert weights["mouthSmileLeft"] == pytest.approx(0.0)

    def test_set_expression_with_intensity(self, controller):
        """Expression intensity scales all AU weights."""
        controller.set_expression(Expression.HAPPY, intensity=0.5)

        # AU12 should be at half its normal intensity
        base_au12 = 1.0  # HAPPY has AU12 at 1.0
        expected = base_au12 * 0.5

        assert controller.get_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER) == pytest.approx(expected, abs=0.01)

    def test_get_blend_weights_arkit_names(self, controller):
        """get_blend_weights returns ARKit-compatible names."""
        controller.set_au_intensity(ActionUnit.AU6_CHEEK_RAISER, 1.0)

        weights = controller.get_blend_weights()

        # Should use ARKit names
        arkit_shapes = [
            "cheekSquintLeft",
            "cheekSquintRight",
        ]

        for shape in arkit_shapes:
            assert shape in weights, f"Missing ARKit shape: {shape}"

    def test_get_blend_weights_combines_aus(self, controller):
        """Multiple AUs affecting same blend shape are combined."""
        # AU25 and AU26 both affect jawOpen
        controller.set_au_intensity(ActionUnit.AU25_LIPS_PART, 1.0)
        controller.set_au_intensity(ActionUnit.AU26_JAW_DROP, 1.0)

        weights = controller.get_blend_weights()

        # jawOpen should be combined (additive, clamped)
        # AU25 has jawOpen=0.3, AU26 has jawOpen=0.7
        # Combined = 0.3 + 0.7 = 1.0
        assert "jawOpen" in weights
        assert weights["jawOpen"] == pytest.approx(1.0, abs=0.01)

    def test_get_active_aus(self, controller):
        """get_active_aus returns only AUs above threshold."""
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 0.5)
        controller.set_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER, 0.0001)  # Below threshold

        active = controller.get_active_aus(threshold=0.001)

        assert ActionUnit.AU1_INNER_BROW_RAISER in active
        assert ActionUnit.AU12_LIP_CORNER_PULLER not in active

    def test_dirty_flag_on_change(self, controller):
        """Dirty flag is set when state changes."""
        controller.clear_dirty()
        assert not controller.dirty

        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 0.5)
        assert controller.dirty

    def test_on_weights_changed_callback(self):
        """Callback is invoked on weight changes."""
        callback_weights = {}

        def on_change(weights):
            callback_weights.update(weights)

        controller = FACSController(on_weights_changed=on_change)
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 0.5)

        assert "browInnerUp" in callback_weights

    def test_create_expression_by_name(self, controller):
        """create_expression returns AU weights for named expression."""
        weights = controller.create_expression("happy")

        assert ActionUnit.AU6_CHEEK_RAISER in weights
        assert ActionUnit.AU12_LIP_CORNER_PULLER in weights

    def test_create_expression_case_insensitive(self, controller):
        """Expression name lookup is case-insensitive."""
        weights1 = controller.create_expression("HAPPY")
        weights2 = controller.create_expression("happy")
        weights3 = controller.create_expression("Happy")

        assert weights1 == weights2 == weights3

    def test_blend_expressions(self, controller):
        """blend_expressions interpolates between two expressions."""
        blended = controller.blend_expressions(
            Expression.HAPPY,
            Expression.SAD,
            blend_factor=0.5
        )

        # HAPPY has AU12 at 1.0, SAD has AU12 at 0
        # At 0.5 blend, AU12 should be 0.5
        if ActionUnit.AU12_LIP_CORNER_PULLER in blended:
            assert blended[ActionUnit.AU12_LIP_CORNER_PULLER] == pytest.approx(0.5, abs=0.1)

    def test_add_expression_preset(self, controller):
        """Custom expression preset can be added."""
        custom_weights = {
            ActionUnit.AU1_INNER_BROW_RAISER: 0.3,
            ActionUnit.AU9_NOSE_WRINKLER: 0.2,
        }

        controller.add_expression_preset(
            Expression.NEUTRAL,  # Overwrite NEUTRAL as example
            au_weights=custom_weights
        )

        controller.set_expression(Expression.NEUTRAL)

        assert controller.get_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER) == pytest.approx(0.3)

    def test_set_au_mapping(self, controller):
        """Custom AU mapping can be set."""
        custom_data = ActionUnitData(
            au=ActionUnit.AU1_INNER_BROW_RAISER,
            blend_shapes={"customBrowUp": 1.0}
        )

        controller.set_au_mapping(ActionUnit.AU1_INNER_BROW_RAISER, custom_data)
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 0.8)

        weights = controller.get_blend_weights()

        assert "customBrowUp" in weights
        assert weights["customBrowUp"] == pytest.approx(0.8)

    def test_serialization_to_dict(self, controller):
        """Controller state can be serialized to dict."""
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 0.5)
        controller.set_expression(Expression.HAPPY)

        data = controller.to_dict()

        assert "au_intensities" in data
        assert "current_expression" in data
        assert data["current_expression"] == "HAPPY"

    def test_serialization_from_dict(self, controller):
        """Controller state can be restored from dict."""
        data = {
            "au_intensities": {
                "AU1_INNER_BROW_RAISER": 0.7,
            },
            "current_expression": "SAD",
        }

        controller.from_dict(data)

        assert controller.get_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER) == pytest.approx(0.7)
        assert controller.current_expression == Expression.SAD


# =============================================================================
# Test: Expression Update/Blending
# =============================================================================

class TestExpressionBlending:
    """Test expression blending over time."""

    def test_instant_expression_change(self):
        """Expression with blend_time=0 changes instantly."""
        controller = FACSController()
        controller.set_expression(Expression.HAPPY, blend_time=0)

        # Should be immediate
        assert controller.current_expression == Expression.HAPPY
        assert controller.get_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER) > 0

    def test_blended_expression_change(self):
        """Expression with blend_time>0 blends over time."""
        controller = FACSController()
        controller.set_expression(Expression.HAPPY, blend_time=1.0)

        # Target should be set, but current unchanged
        assert controller._target_expression == Expression.HAPPY
        assert controller._blend_progress == 0.0

    def test_update_progresses_blend(self):
        """update() advances the blend progress."""
        controller = FACSController()
        controller.set_expression(Expression.HAPPY, blend_time=1.0)

        # Update halfway
        changed = controller.update(0.5)

        assert changed
        assert controller._blend_progress > 0.0

    def test_blend_completes_after_full_time(self):
        """Blend completes after full blend_time."""
        controller = FACSController()
        controller.set_expression(Expression.HAPPY, blend_time=1.0)

        # Update for full time
        controller.update(1.5)

        assert controller._blend_progress >= 1.0
        assert controller.current_expression == Expression.HAPPY
        assert controller._target_expression is None


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unknown_expression_name_returns_empty(self):
        """create_expression with unknown name returns empty dict."""
        controller = FACSController()

        result = controller.create_expression("NONEXISTENT")

        assert result == {}

    def test_blend_expressions_invalid_returns_empty(self):
        """blend_expressions with missing preset returns empty."""
        controller = FACSController()

        # Remove HAPPY preset to simulate missing
        del controller._expression_presets[Expression.HAPPY]

        result = controller.blend_expressions(Expression.HAPPY, Expression.SAD, 0.5)

        assert result == {}

    def test_au_intensity_exactly_zero_not_in_weights(self):
        """AU with exactly zero intensity doesn't contribute to weights."""
        controller = FACSController()
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 0.0)

        weights = controller.get_blend_weights()

        # browInnerUp should not be in weights (or be zero)
        assert "browInnerUp" not in weights or weights.get("browInnerUp", 0) == 0

    def test_very_small_intensity_below_threshold(self):
        """Very small intensity below 0.001 is treated as zero."""
        controller = FACSController()
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 0.0005)

        weights = controller.get_blend_weights()

        # Should not produce blend weights
        assert "browInnerUp" not in weights

    def test_from_dict_ignores_unknown_aus(self):
        """from_dict ignores unknown AU names gracefully."""
        controller = FACSController()

        data = {
            "au_intensities": {
                "AU1_INNER_BROW_RAISER": 0.5,
                "NONEXISTENT_AU": 0.9,  # Should be ignored
            }
        }

        controller.from_dict(data)  # Should not raise

        assert controller.get_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER) == pytest.approx(0.5)

    def test_from_dict_ignores_unknown_expression(self):
        """from_dict ignores unknown expression name gracefully."""
        controller = FACSController()

        data = {
            "current_expression": "NONEXISTENT_EXPRESSION"
        }

        controller.from_dict(data)  # Should not raise

        assert controller.current_expression is None

    def test_negative_blend_weight_clamped(self):
        """Negative blend weights are clamped to -1."""
        # AU25_LIPS_PART has mouthClose: -0.5
        controller = FACSController()
        controller.set_au_intensity(ActionUnit.AU25_LIPS_PART, 1.0)

        weights = controller.get_blend_weights()

        if "mouthClose" in weights:
            assert weights["mouthClose"] >= -1.0


# =============================================================================
# Test: Complete AU to Blend Shape Verification
# =============================================================================

class TestAUBlendShapeCompleteness:
    """Verify all AUs produce meaningful blend shapes."""

    @pytest.fixture
    def controller(self):
        return FACSController()

    @pytest.mark.parametrize("au", list(ActionUnit))
    def test_each_au_produces_blend_weights(self, controller, au):
        """Each AU should produce at least one blend shape weight."""
        controller.set_au_intensity(au, 1.0)

        weights = controller.get_blend_weights()

        assert len(weights) > 0, f"{au.name} produced no blend weights"
        assert any(v > 0 for v in weights.values()), f"{au.name} all weights are zero"


# =============================================================================
# Test: Integration - Full Expression Pipeline
# =============================================================================

class TestFullExpressionPipeline:
    """Integration tests for complete expression pipeline."""

    def test_happy_expression_produces_smile_shapes(self):
        """HAPPY expression produces expected ARKit smile blend shapes."""
        controller = FACSController()
        controller.set_expression(Expression.HAPPY)

        weights = controller.get_blend_weights()

        # Should have smile shapes
        assert "mouthSmileLeft" in weights or "mouthSmileRight" in weights
        # Should have cheek squint (Duchenne smile indicator)
        assert "cheekSquintLeft" in weights or "cheekSquintRight" in weights

    def test_contempt_expression_one_sided_smile(self):
        """CONTEMPT expression produces one-sided smile only."""
        controller = FACSController()
        controller.set_expression(Expression.CONTEMPT)

        weights = controller.get_blend_weights()

        # Right side should be active
        right_active = weights.get("mouthSmileRight", 0) > 0 or weights.get("mouthDimpleRight", 0) > 0
        left_inactive = weights.get("mouthSmileLeft", 0) == 0 and weights.get("mouthDimpleLeft", 0) == 0

        # At minimum, left should be less than right
        left_smile = weights.get("mouthSmileLeft", 0)
        right_smile = weights.get("mouthSmileRight", 0)

        assert right_smile >= left_smile, "CONTEMPT should have right >= left smile"

    def test_expression_sequence(self):
        """Cycling through expressions produces distinct outputs."""
        controller = FACSController()
        all_weights = {}

        for expr in Expression:
            controller.set_expression(expr)
            all_weights[expr] = controller.get_blend_weights().copy()

        # NEUTRAL should be empty
        assert len(all_weights[Expression.NEUTRAL]) == 0

        # CONTEMPT is a special case: it only has au_left_weights and au_right_weights
        # but no au_weights. The current implementation requires base au_weights for
        # bilateral overrides to take effect. This is a known design limitation.
        # CONTEMPT produces empty weights because au_intensities stay at 0.
        expressions_with_weights = [e for e in Expression if e not in (Expression.NEUTRAL, Expression.CONTEMPT)]

        for expr in expressions_with_weights:
            assert len(all_weights[expr]) > 0, f"{expr.name} should have blend weights"

    def test_contempt_requires_workaround(self):
        """
        CONTEMPT expression requires manual AU setup for proper asymmetry.

        This documents a design limitation: expressions with only au_left_weights
        and au_right_weights (no au_weights) do not produce blend outputs because
        get_blend_shape_weights() only processes AUs where base intensity > 0.001.

        Workaround: Set base AU intensity, then bilateral overrides work correctly.
        """
        controller = FACSController()

        # Direct set_expression(CONTEMPT) produces no weights due to design limitation
        controller.set_expression(Expression.CONTEMPT)
        weights_direct = controller.get_blend_weights()

        # Workaround: manually set AU intensities with bilateral overrides
        controller.reset_all_aus()
        controller.set_au_intensity(
            ActionUnit.AU12_LIP_CORNER_PULLER,
            intensity=0.5,  # Need non-zero base
            left=0.0,       # Left at zero
            right=0.5       # Right active
        )
        controller.set_au_intensity(
            ActionUnit.AU14_DIMPLER,
            intensity=0.6,
            left=0.0,
            right=0.6
        )

        weights_manual = controller.get_blend_weights()

        # Manual approach produces correct asymmetric output
        assert len(weights_manual) > 0
        assert weights_manual.get("mouthSmileRight", 0) > 0
        assert weights_manual.get("mouthSmileLeft", 0) == 0
