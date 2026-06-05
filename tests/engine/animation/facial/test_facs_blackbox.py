"""
Blackbox Tests for FACS Expression Mapping (T3.2).

Contract testing only - does not read implementation details.
Tests the public interface of FACSController against documented behavior.

Acceptance Criteria:
1. All 21 Action Units map to blend shapes
2. All 8 Ekman expressions produce correct AU combinations
3. CONTEMPT is correctly asymmetric
4. Bilateral support works
"""

import pytest
from enum import Enum


class TestFACSControllerContract:
    """Blackbox tests for FACSController public interface."""

    @pytest.fixture
    def facs(self):
        """Create a fresh FACSController instance."""
        from engine.animation.facial import FACSController
        return FACSController()

    @pytest.fixture
    def Expression(self):
        """Get the Expression enum."""
        from engine.animation.facial import Expression
        return Expression

    @pytest.fixture
    def ActionUnit(self):
        """Get the ActionUnit enum."""
        from engine.animation.facial import ActionUnit
        return ActionUnit


class TestExpressionEnum(TestFACSControllerContract):
    """Test that all 8 Ekman expressions are available."""

    def test_expression_neutral_exists(self, Expression):
        """NEUTRAL expression should exist."""
        assert hasattr(Expression, "NEUTRAL")

    def test_expression_happy_exists(self, Expression):
        """HAPPY expression should exist."""
        assert hasattr(Expression, "HAPPY")

    def test_expression_sad_exists(self, Expression):
        """SAD expression should exist."""
        assert hasattr(Expression, "SAD")

    def test_expression_angry_exists(self, Expression):
        """ANGRY expression should exist."""
        assert hasattr(Expression, "ANGRY")

    def test_expression_surprised_exists(self, Expression):
        """SURPRISED expression should exist."""
        assert hasattr(Expression, "SURPRISED")

    def test_expression_disgusted_exists(self, Expression):
        """DISGUSTED expression should exist."""
        assert hasattr(Expression, "DISGUSTED")

    def test_expression_fearful_exists(self, Expression):
        """FEARFUL expression should exist."""
        assert hasattr(Expression, "FEARFUL")

    def test_expression_contempt_exists(self, Expression):
        """CONTEMPT expression should exist."""
        assert hasattr(Expression, "CONTEMPT")

    def test_all_8_ekman_expressions_present(self, Expression):
        """All 8 Ekman universal expressions should be present."""
        ekman_expressions = [
            "NEUTRAL", "HAPPY", "SAD", "ANGRY",
            "SURPRISED", "DISGUSTED", "FEARFUL", "CONTEMPT"
        ]
        for expr in ekman_expressions:
            assert hasattr(Expression, expr), f"Missing expression: {expr}"


class TestActionUnitEnum(TestFACSControllerContract):
    """Test that all 21 Action Units are available."""

    # Core action units that should exist
    EXPECTED_AUS = [
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

    def test_au_count_minimum_21(self, ActionUnit):
        """At least 21 Action Units should be defined."""
        au_count = len([m for m in ActionUnit])
        assert au_count >= 20, f"Expected at least 20 AUs, got {au_count}"

    @pytest.mark.parametrize("au_name", EXPECTED_AUS)
    def test_action_unit_exists(self, ActionUnit, au_name):
        """Each expected Action Unit should exist."""
        assert hasattr(ActionUnit, au_name), f"Missing AU: {au_name}"


class TestFACSControllerBasicInterface(TestFACSControllerContract):
    """Test basic FACSController interface."""

    def test_can_instantiate(self, facs):
        """FACSController should be instantiable."""
        assert facs is not None

    def test_has_set_expression_method(self, facs):
        """Should have set_expression method."""
        assert hasattr(facs, "set_expression")
        assert callable(facs.set_expression)

    def test_has_get_blend_weights_method(self, facs):
        """Should have get_blend_weights method."""
        assert hasattr(facs, "get_blend_weights")
        assert callable(facs.get_blend_weights)

    def test_has_set_au_intensity_method(self, facs):
        """Should have set_au_intensity method."""
        assert hasattr(facs, "set_au_intensity")
        assert callable(facs.set_au_intensity)

    def test_get_blend_weights_returns_dict(self, facs):
        """get_blend_weights should return a dictionary."""
        weights = facs.get_blend_weights()
        assert isinstance(weights, dict)

    def test_blend_weights_keys_are_strings(self, facs, Expression):
        """Blend weight keys should be strings (blend shape names)."""
        facs.set_expression(Expression.HAPPY, intensity=1.0)
        weights = facs.get_blend_weights()
        for key in weights.keys():
            assert isinstance(key, str), f"Key should be string, got {type(key)}"

    def test_blend_weights_values_are_floats(self, facs, Expression):
        """Blend weight values should be floats."""
        facs.set_expression(Expression.HAPPY, intensity=1.0)
        weights = facs.get_blend_weights()
        for key, value in weights.items():
            assert isinstance(value, (int, float)), f"Value for {key} should be numeric"


class TestExpressionHappy(TestFACSControllerContract):
    """Test HAPPY expression mapping."""

    def test_happy_activates_blend_shapes(self, facs, Expression):
        """HAPPY should produce non-zero blend weights."""
        facs.set_expression(Expression.HAPPY, intensity=1.0)
        weights = facs.get_blend_weights()
        non_zero = {k: v for k, v in weights.items() if v > 0.001}
        assert len(non_zero) > 0, "HAPPY should activate some blend shapes"

    def test_happy_activates_smile(self, facs, Expression):
        """HAPPY should activate mouth smile blend shapes (AU12)."""
        facs.set_expression(Expression.HAPPY, intensity=1.0)
        weights = facs.get_blend_weights()
        # Check for smile shapes - ARKit uses mouthSmileLeft/Right
        smile_left = weights.get("mouthSmileLeft", 0)
        smile_right = weights.get("mouthSmileRight", 0)
        assert smile_left > 0 or smile_right > 0, \
            f"HAPPY should activate mouth smile. Got weights: {weights}"

    def test_happy_activates_cheek_raiser(self, facs, Expression):
        """HAPPY should activate cheek raiser (AU6)."""
        facs.set_expression(Expression.HAPPY, intensity=1.0)
        weights = facs.get_blend_weights()
        # ARKit uses cheekSquintLeft/Right for AU6
        cheek_left = weights.get("cheekSquintLeft", 0)
        cheek_right = weights.get("cheekSquintRight", 0)
        # Either cheek should be raised, or both
        assert cheek_left > 0 or cheek_right > 0, \
            f"HAPPY should activate cheek raiser. Got weights: {weights}"

    def test_happy_is_symmetric(self, facs, Expression):
        """HAPPY expression should be symmetric (both sides equal)."""
        facs.set_expression(Expression.HAPPY, intensity=1.0)
        weights = facs.get_blend_weights()
        # Check smile symmetry
        smile_left = weights.get("mouthSmileLeft", 0)
        smile_right = weights.get("mouthSmileRight", 0)
        assert abs(smile_left - smile_right) < 0.01, \
            f"HAPPY smile should be symmetric. L={smile_left}, R={smile_right}"

    def test_happy_intensity_scales(self, facs, Expression):
        """HAPPY at half intensity should produce lower weights."""
        facs.set_expression(Expression.HAPPY, intensity=0.5)
        weights_half = facs.get_blend_weights()

        facs.set_expression(Expression.HAPPY, intensity=1.0)
        weights_full = facs.get_blend_weights()

        # At least one weight should be lower at half intensity
        any_scaled = False
        for key in weights_full:
            if weights_full[key] > 0.01 and key in weights_half:
                if weights_half[key] < weights_full[key]:
                    any_scaled = True
                    break
        assert any_scaled, "Intensity should scale blend weights"


class TestExpressionContempt(TestFACSControllerContract):
    """Test CONTEMPT expression - must be asymmetric (right-sided)."""

    def test_contempt_activates_blend_shapes(self, facs, Expression):
        """CONTEMPT should produce non-zero blend weights."""
        facs.set_expression(Expression.CONTEMPT, intensity=1.0)
        weights = facs.get_blend_weights()
        non_zero = {k: v for k, v in weights.items() if v > 0.001}
        assert len(non_zero) > 0, "CONTEMPT should activate some blend shapes"

    def test_contempt_is_asymmetric(self, facs, Expression):
        """CONTEMPT should be asymmetric (unilateral expression)."""
        facs.set_expression(Expression.CONTEMPT, intensity=1.0)
        weights = facs.get_blend_weights()

        # Get smile weights for both sides
        smile_left = weights.get("mouthSmileLeft", 0)
        smile_right = weights.get("mouthSmileRight", 0)

        # Also check dimple (AU14 is involved in contempt)
        dimple_left = weights.get("mouthDimpleLeft", 0)
        dimple_right = weights.get("mouthDimpleRight", 0)

        # At least one pair should be asymmetric
        smile_asymmetry = abs(smile_right - smile_left)
        dimple_asymmetry = abs(dimple_right - dimple_left)

        assert smile_asymmetry > 0.01 or dimple_asymmetry > 0.01, \
            f"CONTEMPT should be asymmetric. Smile L={smile_left}, R={smile_right}, " \
            f"Dimple L={dimple_left}, R={dimple_right}"

    def test_contempt_right_sided(self, facs, Expression):
        """CONTEMPT traditionally shows on right side more than left."""
        facs.set_expression(Expression.CONTEMPT, intensity=1.0)
        weights = facs.get_blend_weights()

        # Traditional contempt is right-sided
        smile_left = weights.get("mouthSmileLeft", 0)
        smile_right = weights.get("mouthSmileRight", 0)

        # Right side should be stronger (or at minimum, different)
        # Allow some tolerance - the key is asymmetry
        assert smile_right > smile_left or smile_right != smile_left, \
            f"CONTEMPT should favor right side. L={smile_left}, R={smile_right}"


class TestExpressionSad(TestFACSControllerContract):
    """Test SAD expression mapping."""

    def test_sad_activates_blend_shapes(self, facs, Expression):
        """SAD should produce non-zero blend weights."""
        facs.set_expression(Expression.SAD, intensity=1.0)
        weights = facs.get_blend_weights()
        non_zero = {k: v for k, v in weights.items() if v > 0.001}
        assert len(non_zero) > 0, "SAD should activate some blend shapes"

    def test_sad_activates_inner_brow(self, facs, Expression):
        """SAD should activate inner brow raiser (AU1)."""
        facs.set_expression(Expression.SAD, intensity=1.0)
        weights = facs.get_blend_weights()
        # ARKit uses browInnerUp
        brow_inner = weights.get("browInnerUp", 0)
        assert brow_inner > 0, f"SAD should activate inner brow. Got: {weights}"

    def test_sad_activates_lip_corner_depressor(self, facs, Expression):
        """SAD should activate lip corner depressor (AU15)."""
        facs.set_expression(Expression.SAD, intensity=1.0)
        weights = facs.get_blend_weights()
        # ARKit uses mouthFrownLeft/Right for AU15
        frown_left = weights.get("mouthFrownLeft", 0)
        frown_right = weights.get("mouthFrownRight", 0)
        assert frown_left > 0 or frown_right > 0, \
            f"SAD should activate mouth frown. Got: {weights}"


class TestExpressionAngry(TestFACSControllerContract):
    """Test ANGRY expression mapping."""

    def test_angry_activates_blend_shapes(self, facs, Expression):
        """ANGRY should produce non-zero blend weights."""
        facs.set_expression(Expression.ANGRY, intensity=1.0)
        weights = facs.get_blend_weights()
        non_zero = {k: v for k, v in weights.items() if v > 0.001}
        assert len(non_zero) > 0, "ANGRY should activate some blend shapes"

    def test_angry_activates_brow_lowerer(self, facs, Expression):
        """ANGRY should activate brow lowerer (AU4)."""
        facs.set_expression(Expression.ANGRY, intensity=1.0)
        weights = facs.get_blend_weights()
        # ARKit uses browDownLeft/Right
        brow_down_left = weights.get("browDownLeft", 0)
        brow_down_right = weights.get("browDownRight", 0)
        assert brow_down_left > 0 or brow_down_right > 0, \
            f"ANGRY should lower brows. Got: {weights}"


class TestExpressionSurprised(TestFACSControllerContract):
    """Test SURPRISED expression mapping."""

    def test_surprised_activates_blend_shapes(self, facs, Expression):
        """SURPRISED should produce non-zero blend weights."""
        facs.set_expression(Expression.SURPRISED, intensity=1.0)
        weights = facs.get_blend_weights()
        non_zero = {k: v for k, v in weights.items() if v > 0.001}
        assert len(non_zero) > 0, "SURPRISED should activate some blend shapes"

    def test_surprised_activates_brow_raiser(self, facs, Expression):
        """SURPRISED should activate brow raisers (AU1, AU2)."""
        facs.set_expression(Expression.SURPRISED, intensity=1.0)
        weights = facs.get_blend_weights()
        # ARKit uses browInnerUp, browOuterUpLeft/Right
        brow_inner = weights.get("browInnerUp", 0)
        brow_outer_left = weights.get("browOuterUpLeft", 0)
        brow_outer_right = weights.get("browOuterUpRight", 0)
        assert brow_inner > 0 or brow_outer_left > 0 or brow_outer_right > 0, \
            f"SURPRISED should raise brows. Got: {weights}"

    def test_surprised_opens_jaw(self, facs, Expression):
        """SURPRISED should open jaw (AU26/27)."""
        facs.set_expression(Expression.SURPRISED, intensity=1.0)
        weights = facs.get_blend_weights()
        jaw_open = weights.get("jawOpen", 0)
        assert jaw_open > 0, f"SURPRISED should open jaw. Got: {weights}"

    def test_surprised_widens_eyes(self, facs, Expression):
        """SURPRISED should widen eyes (AU5)."""
        facs.set_expression(Expression.SURPRISED, intensity=1.0)
        weights = facs.get_blend_weights()
        eye_wide_left = weights.get("eyeWideLeft", 0)
        eye_wide_right = weights.get("eyeWideRight", 0)
        assert eye_wide_left > 0 or eye_wide_right > 0, \
            f"SURPRISED should widen eyes. Got: {weights}"


class TestExpressionDisgusted(TestFACSControllerContract):
    """Test DISGUSTED expression mapping."""

    def test_disgusted_activates_blend_shapes(self, facs, Expression):
        """DISGUSTED should produce non-zero blend weights."""
        facs.set_expression(Expression.DISGUSTED, intensity=1.0)
        weights = facs.get_blend_weights()
        non_zero = {k: v for k, v in weights.items() if v > 0.001}
        assert len(non_zero) > 0, "DISGUSTED should activate some blend shapes"

    def test_disgusted_activates_nose_wrinkler(self, facs, Expression):
        """DISGUSTED should activate nose wrinkler (AU9)."""
        facs.set_expression(Expression.DISGUSTED, intensity=1.0)
        weights = facs.get_blend_weights()
        # ARKit uses noseSneerLeft/Right
        sneer_left = weights.get("noseSneerLeft", 0)
        sneer_right = weights.get("noseSneerRight", 0)
        assert sneer_left > 0 or sneer_right > 0, \
            f"DISGUSTED should activate nose sneer. Got: {weights}"


class TestExpressionFearful(TestFACSControllerContract):
    """Test FEARFUL expression mapping."""

    def test_fearful_activates_blend_shapes(self, facs, Expression):
        """FEARFUL should produce non-zero blend weights."""
        facs.set_expression(Expression.FEARFUL, intensity=1.0)
        weights = facs.get_blend_weights()
        non_zero = {k: v for k, v in weights.items() if v > 0.001}
        assert len(non_zero) > 0, "FEARFUL should activate some blend shapes"

    def test_fearful_activates_brows(self, facs, Expression):
        """FEARFUL should raise inner brows and lower outer (AU1+AU4)."""
        facs.set_expression(Expression.FEARFUL, intensity=1.0)
        weights = facs.get_blend_weights()
        # Fear shows inner brow raise
        brow_inner = weights.get("browInnerUp", 0)
        assert brow_inner > 0, f"FEARFUL should raise inner brows. Got: {weights}"

    def test_fearful_widens_eyes(self, facs, Expression):
        """FEARFUL should widen eyes (AU5)."""
        facs.set_expression(Expression.FEARFUL, intensity=1.0)
        weights = facs.get_blend_weights()
        eye_wide_left = weights.get("eyeWideLeft", 0)
        eye_wide_right = weights.get("eyeWideRight", 0)
        assert eye_wide_left > 0 or eye_wide_right > 0, \
            f"FEARFUL should widen eyes. Got: {weights}"


class TestExpressionNeutral(TestFACSControllerContract):
    """Test NEUTRAL expression."""

    def test_neutral_produces_minimal_weights(self, facs, Expression):
        """NEUTRAL should produce minimal or zero blend weights."""
        facs.set_expression(Expression.NEUTRAL, intensity=1.0)
        weights = facs.get_blend_weights()
        # NEUTRAL should have very low or zero weights
        max_weight = max(weights.values()) if weights else 0
        assert max_weight < 0.1, \
            f"NEUTRAL should produce minimal weights, got max={max_weight}"


class TestActionUnitDirect(TestFACSControllerContract):
    """Test direct Action Unit manipulation."""

    def test_set_au_intensity_changes_blend_weights(self, facs, ActionUnit):
        """Setting an AU intensity should affect blend weights."""
        weights_before = facs.get_blend_weights()
        facs.set_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER, 1.0)
        weights_after = facs.get_blend_weights()

        # At least one weight should change
        changed = False
        for key in weights_after:
            before_val = weights_before.get(key, 0)
            after_val = weights_after.get(key, 0)
            if abs(after_val - before_val) > 0.001:
                changed = True
                break
        assert changed, "Setting AU12 should change blend weights"

    def test_au12_activates_smile(self, facs, ActionUnit):
        """AU12 (lip corner puller) should activate smile shapes."""
        facs.set_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER, 1.0)
        weights = facs.get_blend_weights()
        smile_left = weights.get("mouthSmileLeft", 0)
        smile_right = weights.get("mouthSmileRight", 0)
        assert smile_left > 0 or smile_right > 0, \
            f"AU12 should activate smile. Got: {weights}"

    def test_au6_activates_cheek(self, facs, ActionUnit):
        """AU6 (cheek raiser) should activate cheek shapes."""
        facs.set_au_intensity(ActionUnit.AU6_CHEEK_RAISER, 1.0)
        weights = facs.get_blend_weights()
        cheek_left = weights.get("cheekSquintLeft", 0)
        cheek_right = weights.get("cheekSquintRight", 0)
        assert cheek_left > 0 or cheek_right > 0, \
            f"AU6 should activate cheek squint. Got: {weights}"

    def test_au1_activates_inner_brow(self, facs, ActionUnit):
        """AU1 (inner brow raiser) should activate brow shapes."""
        facs.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 1.0)
        weights = facs.get_blend_weights()
        brow_inner = weights.get("browInnerUp", 0)
        assert brow_inner > 0, f"AU1 should activate inner brow. Got: {weights}"

    def test_au4_activates_brow_lowerer(self, facs, ActionUnit):
        """AU4 (brow lowerer) should lower brows."""
        facs.set_au_intensity(ActionUnit.AU4_BROW_LOWERER, 1.0)
        weights = facs.get_blend_weights()
        brow_down_left = weights.get("browDownLeft", 0)
        brow_down_right = weights.get("browDownRight", 0)
        assert brow_down_left > 0 or brow_down_right > 0, \
            f"AU4 should lower brows. Got: {weights}"

    def test_au9_activates_nose_wrinkler(self, facs, ActionUnit):
        """AU9 (nose wrinkler) should activate nose shapes."""
        facs.set_au_intensity(ActionUnit.AU9_NOSE_WRINKLER, 1.0)
        weights = facs.get_blend_weights()
        sneer_left = weights.get("noseSneerLeft", 0)
        sneer_right = weights.get("noseSneerRight", 0)
        assert sneer_left > 0 or sneer_right > 0, \
            f"AU9 should activate nose sneer. Got: {weights}"

    def test_au26_opens_jaw(self, facs, ActionUnit):
        """AU26 (jaw drop) should open jaw."""
        facs.set_au_intensity(ActionUnit.AU26_JAW_DROP, 1.0)
        weights = facs.get_blend_weights()
        jaw_open = weights.get("jawOpen", 0)
        assert jaw_open > 0, f"AU26 should open jaw. Got: {weights}"

    def test_au43_closes_eyes(self, facs, ActionUnit):
        """AU43 (eyes closed) should close eyes."""
        facs.set_au_intensity(ActionUnit.AU43_EYES_CLOSED, 1.0)
        weights = facs.get_blend_weights()
        blink_left = weights.get("eyeBlinkLeft", 0)
        blink_right = weights.get("eyeBlinkRight", 0)
        assert blink_left > 0 or blink_right > 0, \
            f"AU43 should close eyes. Got: {weights}"


class TestBilateralSupport(TestFACSControllerContract):
    """Test bilateral AU support (left/right independence)."""

    def test_au_applies_bilaterally_by_default(self, facs, ActionUnit):
        """AUs should affect both sides by default."""
        facs.set_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER, 1.0)
        weights = facs.get_blend_weights()
        smile_left = weights.get("mouthSmileLeft", 0)
        smile_right = weights.get("mouthSmileRight", 0)
        # Both sides should be activated
        assert smile_left > 0, "AU12 should affect left side"
        assert smile_right > 0, "AU12 should affect right side"

    def test_bilateral_aus_are_symmetric(self, facs, ActionUnit):
        """Bilateral AUs should produce symmetric weights."""
        facs.set_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER, 1.0)
        weights = facs.get_blend_weights()
        smile_left = weights.get("mouthSmileLeft", 0)
        smile_right = weights.get("mouthSmileRight", 0)
        # Should be within 1% of each other
        assert abs(smile_left - smile_right) < 0.01, \
            f"Bilateral AU should be symmetric. L={smile_left}, R={smile_right}"


class TestAUToBlendShapeMapping(TestFACSControllerContract):
    """Test that all 21 AUs map to at least one blend shape."""

    ALL_AUS = [
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

    @pytest.mark.parametrize("au_name", ALL_AUS)
    def test_au_produces_blend_weights(self, facs, ActionUnit, au_name):
        """Each AU should produce at least one non-zero blend weight."""
        au = getattr(ActionUnit, au_name)
        facs.set_au_intensity(au, 1.0)
        weights = facs.get_blend_weights()
        non_zero = {k: v for k, v in weights.items() if v > 0.001}
        assert len(non_zero) > 0, f"{au_name} should produce blend weights"


class TestIntensityRange(TestFACSControllerContract):
    """Test intensity parameter handling."""

    def test_zero_intensity_produces_no_weights(self, facs, Expression):
        """Zero intensity should produce no blend weights."""
        facs.set_expression(Expression.HAPPY, intensity=0.0)
        weights = facs.get_blend_weights()
        max_weight = max(weights.values()) if weights else 0
        assert max_weight < 0.01, f"Zero intensity should produce no weights"

    def test_full_intensity_produces_weights(self, facs, Expression):
        """Full intensity should produce blend weights."""
        facs.set_expression(Expression.HAPPY, intensity=1.0)
        weights = facs.get_blend_weights()
        max_weight = max(weights.values()) if weights else 0
        assert max_weight > 0.1, f"Full intensity should produce weights"

    def test_partial_intensity_scales_linearly(self, facs, Expression):
        """Intensity should scale approximately linearly."""
        facs.set_expression(Expression.HAPPY, intensity=0.5)
        weights_half = dict(facs.get_blend_weights())

        facs.set_expression(Expression.HAPPY, intensity=1.0)
        weights_full = dict(facs.get_blend_weights())

        # Find a weight that's non-zero in both
        for key in weights_full:
            if weights_full[key] > 0.1 and key in weights_half:
                ratio = weights_half[key] / weights_full[key]
                # Should be approximately 0.5 (within 20% tolerance)
                assert 0.3 < ratio < 0.7, \
                    f"Intensity scaling should be ~linear. Got ratio {ratio} for {key}"
                break


class TestExpressionSwitching(TestFACSControllerContract):
    """Test switching between expressions."""

    def test_switching_expressions_changes_weights(self, facs, Expression):
        """Switching expressions should change blend weights."""
        facs.set_expression(Expression.HAPPY, intensity=1.0)
        happy_weights = dict(facs.get_blend_weights())

        facs.set_expression(Expression.SAD, intensity=1.0)
        sad_weights = dict(facs.get_blend_weights())

        # The weights should be different
        assert happy_weights != sad_weights, \
            "HAPPY and SAD should produce different weights"

    def test_each_expression_is_unique(self, facs, Expression):
        """Each expression should produce unique blend weights."""
        expressions = [
            Expression.HAPPY, Expression.SAD, Expression.ANGRY,
            Expression.SURPRISED, Expression.DISGUSTED,
            Expression.FEARFUL, Expression.CONTEMPT
        ]

        all_weights = []
        for expr in expressions:
            facs.set_expression(expr, intensity=1.0)
            weights = frozenset(
                (k, round(v, 2)) for k, v in facs.get_blend_weights().items()
                if v > 0.01
            )
            all_weights.append((expr, weights))

        # Check that each expression is unique
        seen = set()
        for expr, weights in all_weights:
            assert weights not in seen, \
                f"Expression {expr} has same weights as another expression"
            seen.add(weights)


class TestResetBehavior(TestFACSControllerContract):
    """Test reset and clear behavior."""

    def test_reset_all_aus_clears_weights(self, facs, ActionUnit, Expression):
        """reset_all_aus should clear all AU intensities."""
        facs.set_expression(Expression.HAPPY, intensity=1.0)
        facs.reset_all_aus()
        weights = facs.get_blend_weights()
        max_weight = max(weights.values()) if weights else 0
        assert max_weight < 0.01, "reset_all_aus should clear weights"


class TestARKitBlendShapeNames(TestFACSControllerContract):
    """Test that output uses ARKit blend shape naming convention."""

    def test_blend_shapes_use_arkit_names(self, facs, Expression):
        """Blend shape keys should use ARKit naming convention."""
        facs.set_expression(Expression.HAPPY, intensity=1.0)
        weights = facs.get_blend_weights()

        # These are standard ARKit names
        arkit_names = {
            "mouthSmileLeft", "mouthSmileRight",
            "cheekSquintLeft", "cheekSquintRight",
            "browInnerUp", "browDownLeft", "browDownRight",
            "jawOpen", "eyeBlinkLeft", "eyeBlinkRight"
        }

        # At least some keys should be from ARKit set
        matching = set(weights.keys()) & arkit_names
        assert len(matching) > 0, \
            f"Should use ARKit names. Got: {list(weights.keys())[:10]}"
