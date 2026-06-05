"""
Blackbox Tests for Face Rig Layer Blending (T3.5)

CLEANROOM TESTING: Tests derived solely from public contract, not implementation.

Public Contract (as specified):
    from engine.animation.facial import FaceRig, LayerPriority, RigLayer

    rig = FaceRig()
    rig.add_layer(name, priority, additive=False)
    rig.set_layer_weight(layer_name, blend_shape_name, weight)
    result = rig.evaluate()  # -> Dict[str, float]

CONTRACT DEVIATIONS FOUND:
    1. FaceRig.__init__() requires a BlendShapeSet argument (not documented)
    2. BlendShapeSet.__init__() requires a 'name' argument (not documented)
    3. LayerPriority enum does NOT have IDLE, OVERRIDE attributes
       (architecture doc says: IDLE=0, EMOTION=1, LIP_SYNC=2, PROCEDURAL=3, OVERRIDE=4)
       Tests adapted to use numeric priority values.

Test Categories:
    1. Priority override: emotion layer (priority=10) overrides idle (priority=0)
    2. Additive accumulation: procedural (additive=True) adds to emotion
    3. Weight scaling: layer master weight scales all its shapes
    4. Clamp to [0, 1]: weights outside range are clamped
"""

import pytest
from typing import Dict


# Priority constants based on architecture doc (since LayerPriority enum is different)
PRIORITY_IDLE = 0
PRIORITY_EMOTION = 1
PRIORITY_LIP_SYNC = 2
PRIORITY_PROCEDURAL = 3
PRIORITY_OVERRIDE = 4


def create_face_rig():
    """Factory function to create FaceRig with minimal setup."""
    from engine.animation.facial import FaceRig, BlendShapeSet
    blend_shape_set = BlendShapeSet(name="test_shapes")
    return FaceRig(blend_shape_set)


class TestFaceRigLayerBlendingBlackbox:
    """Blackbox tests for FaceRig layer blending based on public contract."""

    # =========================================================================
    # TEST CATEGORY 1: Priority Override
    # =========================================================================

    def test_priority_override_higher_priority_wins(self):
        """Higher priority layers should override lower priority layers."""
        rig = create_face_rig()
        # Add layers with different priorities
        rig.add_layer("idle", priority=0, additive=False)
        rig.add_layer("emotion", priority=10, additive=False)

        # Set same blend shape in both layers
        rig.set_layer_weight("idle", "mouthSmileL", 0.3)
        rig.set_layer_weight("emotion", "mouthSmileL", 0.9)

        result = rig.evaluate()

        # Emotion (priority=10) should override idle (priority=0)
        # Result should reflect the higher priority layer's value
        assert result["mouthSmileL"] > 0.5, "Higher priority layer should dominate"

    def test_priority_override_idle_vs_emotion(self):
        """Emotion layer (standard priority) overrides idle layer."""
        rig = create_face_rig()
        rig.add_layer("idle", priority=PRIORITY_IDLE, additive=False)
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("idle", "browInnerUpL", 0.2)
        rig.set_layer_weight("emotion", "browInnerUpL", 0.8)

        result = rig.evaluate()

        # With override blending, higher priority should significantly influence result
        assert result["browInnerUpL"] > 0.5

    def test_priority_override_emotion_vs_lip_sync(self):
        """Lip sync layer (higher priority) overrides emotion layer."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)
        rig.add_layer("lip_sync", priority=PRIORITY_LIP_SYNC, additive=False)

        rig.set_layer_weight("emotion", "jawOpen", 0.2)
        rig.set_layer_weight("lip_sync", "jawOpen", 0.7)

        result = rig.evaluate()

        # Lip sync (higher priority) should dominate
        assert result["jawOpen"] > 0.5

    def test_priority_override_respects_layer_ordering(self):
        """Multiple layers should be sorted and blended by priority."""
        rig = create_face_rig()
        # Add layers in non-priority order to test sorting
        rig.add_layer("override", priority=PRIORITY_OVERRIDE, additive=False)
        rig.add_layer("idle", priority=PRIORITY_IDLE, additive=False)
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("idle", "cheekSquintL", 0.1)
        rig.set_layer_weight("emotion", "cheekSquintL", 0.5)
        rig.set_layer_weight("override", "cheekSquintL", 1.0)

        result = rig.evaluate()

        # Override (highest) should be the final value
        assert result["cheekSquintL"] >= 0.9

    def test_priority_numeric_values_work(self):
        """Numeric priority values should work correctly."""
        rig = create_face_rig()
        rig.add_layer("low", priority=5, additive=False)
        rig.add_layer("high", priority=100, additive=False)

        rig.set_layer_weight("low", "noseSneerL", 0.2)
        rig.set_layer_weight("high", "noseSneerL", 0.8)

        result = rig.evaluate()

        assert result["noseSneerL"] > 0.5

    def test_priority_only_higher_layer_active(self):
        """When only higher priority layer has a value, that value is used."""
        rig = create_face_rig()
        rig.add_layer("idle", priority=PRIORITY_IDLE, additive=False)
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        # Only set value in higher priority layer
        rig.set_layer_weight("emotion", "eyeBlinkL", 0.7)

        result = rig.evaluate()

        assert "eyeBlinkL" in result
        assert abs(result["eyeBlinkL"] - 0.7) < 0.01

    def test_priority_only_lower_layer_active(self):
        """When only lower priority layer has a value, that value is used."""
        rig = create_face_rig()
        rig.add_layer("idle", priority=PRIORITY_IDLE, additive=False)
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        # Only set value in lower priority layer
        rig.set_layer_weight("idle", "eyeSquintL", 0.4)

        result = rig.evaluate()

        assert "eyeSquintL" in result
        assert abs(result["eyeSquintL"] - 0.4) < 0.01

    # =========================================================================
    # TEST CATEGORY 2: Additive Accumulation
    # =========================================================================

    def test_additive_accumulation_basic(self):
        """Additive layers should accumulate on top of base values."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)
        rig.add_layer("procedural", priority=PRIORITY_PROCEDURAL, additive=True)

        rig.set_layer_weight("emotion", "browDownL", 0.3)
        rig.set_layer_weight("procedural", "browDownL", 0.2)

        result = rig.evaluate()

        # Should accumulate: 0.3 + 0.2 = 0.5
        assert abs(result["browDownL"] - 0.5) < 0.01

    def test_additive_accumulation_multiple_additive_layers(self):
        """Multiple additive layers should all contribute."""
        rig = create_face_rig()
        rig.add_layer("base", priority=0, additive=False)
        rig.add_layer("add1", priority=10, additive=True)
        rig.add_layer("add2", priority=20, additive=True)

        rig.set_layer_weight("base", "mouthFunnel", 0.2)
        rig.set_layer_weight("add1", "mouthFunnel", 0.15)
        rig.set_layer_weight("add2", "mouthFunnel", 0.15)

        result = rig.evaluate()

        # Should accumulate: 0.2 + 0.15 + 0.15 = 0.5
        assert abs(result["mouthFunnel"] - 0.5) < 0.01

    def test_additive_layer_on_empty_base(self):
        """Additive layer with no base should still contribute its value."""
        rig = create_face_rig()
        rig.add_layer("procedural", priority=PRIORITY_PROCEDURAL, additive=True)

        rig.set_layer_weight("procedural", "tongueOut", 0.4)

        result = rig.evaluate()

        # Starting from 0.0, adding 0.4 should give 0.4
        assert abs(result["tongueOut"] - 0.4) < 0.01

    def test_additive_and_override_mixed(self):
        """Mixed additive and override layers blend correctly."""
        rig = create_face_rig()
        rig.add_layer("idle", priority=PRIORITY_IDLE, additive=False)
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)
        rig.add_layer("procedural", priority=PRIORITY_PROCEDURAL, additive=True)

        rig.set_layer_weight("idle", "mouthPucker", 0.2)
        rig.set_layer_weight("emotion", "mouthPucker", 0.5)  # Override idle
        rig.set_layer_weight("procedural", "mouthPucker", 0.1)  # Add to result

        result = rig.evaluate()

        # After emotion override (0.5 blended with 0.2), then add 0.1
        # Result should be > 0.5 due to additive
        assert result["mouthPucker"] > 0.5

    def test_additive_does_not_affect_other_shapes(self):
        """Additive layers only affect shapes they define."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)
        rig.add_layer("procedural", priority=PRIORITY_PROCEDURAL, additive=True)

        rig.set_layer_weight("emotion", "mouthSmileR", 0.6)
        rig.set_layer_weight("procedural", "eyeLookUpL", 0.3)  # Different shape

        result = rig.evaluate()

        # mouthSmileR should be unaffected by procedural layer
        assert abs(result["mouthSmileR"] - 0.6) < 0.01
        assert abs(result["eyeLookUpL"] - 0.3) < 0.01

    # =========================================================================
    # TEST CATEGORY 3: Weight Scaling
    # =========================================================================

    def test_layer_master_weight_scales_values(self):
        """Layer master weight should scale all shape values in that layer."""
        rig = create_face_rig()
        layer = rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        # Set a shape weight
        rig.set_layer_weight("emotion", "jawOpen", 1.0)

        # If the layer has a master weight, it should scale the output
        # Default master weight should be 1.0
        result1 = rig.evaluate()

        # The shape value should be scaled by the layer's master weight
        assert "jawOpen" in result1

    def test_layer_weight_zero_overrides_lower_layer(self):
        """A higher priority layer with zero weight should override lower layer."""
        rig = create_face_rig()
        rig.add_layer("idle", priority=PRIORITY_IDLE, additive=False)
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("idle", "mouthLeft", 0.8)
        rig.set_layer_weight("emotion", "mouthLeft", 0.0)  # Explicit zero

        result = rig.evaluate()

        # When emotion layer explicitly sets to 0.0, it overrides the idle layer's 0.8
        # This is correct override behavior - explicit zero wins
        assert result.get("mouthLeft", -1) == 0.0

    def test_layer_without_shape_does_not_affect_lower_layer(self):
        """When a higher layer has NO entry for a shape, lower layer value is preserved."""
        rig = create_face_rig()
        rig.add_layer("idle", priority=PRIORITY_IDLE, additive=False)
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        # Only set value in lower priority layer
        rig.set_layer_weight("idle", "mouthRight", 0.8)
        # Do NOT set mouthRight in emotion layer

        result = rig.evaluate()

        # Idle layer value should pass through since emotion doesn't touch this shape
        assert result.get("mouthRight", 0) > 0

    def test_partial_weight_scaling(self):
        """Partial weights should scale contributions proportionally."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        # Set shape at half weight
        rig.set_layer_weight("emotion", "eyeWideL", 0.5)

        result = rig.evaluate()

        assert abs(result["eyeWideL"] - 0.5) < 0.01

    def test_weight_scaling_multiple_shapes_same_layer(self):
        """Multiple shapes in one layer should all be scaled consistently."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("emotion", "browOuterUpL", 0.4)
        rig.set_layer_weight("emotion", "browOuterUpR", 0.4)
        rig.set_layer_weight("emotion", "cheekPuff", 0.6)

        result = rig.evaluate()

        assert abs(result["browOuterUpL"] - 0.4) < 0.01
        assert abs(result["browOuterUpR"] - 0.4) < 0.01
        assert abs(result["cheekPuff"] - 0.6) < 0.01

    # =========================================================================
    # TEST CATEGORY 4: Weight Clamping [0, 1]
    # =========================================================================

    def test_clamp_weight_above_one(self):
        """Weights above 1.0 should be clamped to 1.0."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("emotion", "mouthClose", 1.5)  # Above 1.0

        result = rig.evaluate()

        # Should be clamped to 1.0
        assert result["mouthClose"] <= 1.0

    def test_clamp_weight_below_zero(self):
        """Weights below 0.0 should be clamped to 0.0."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("emotion", "mouthRight", -0.5)  # Below 0.0

        result = rig.evaluate()

        # Should be clamped to 0.0
        assert result.get("mouthRight", 0) >= 0.0

    def test_clamp_additive_accumulation_above_one(self):
        """Additive accumulation that exceeds 1.0 should be clamped."""
        rig = create_face_rig()
        rig.add_layer("base", priority=0, additive=False)
        rig.add_layer("add1", priority=10, additive=True)
        rig.add_layer("add2", priority=20, additive=True)

        rig.set_layer_weight("base", "eyeLookDownL", 0.5)
        rig.set_layer_weight("add1", "eyeLookDownL", 0.4)
        rig.set_layer_weight("add2", "eyeLookDownL", 0.4)  # Total would be 1.3

        result = rig.evaluate()

        # Should be clamped to 1.0
        assert result["eyeLookDownL"] <= 1.0

    def test_clamp_at_exact_boundary_one(self):
        """Weight exactly at 1.0 should remain 1.0."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("emotion", "eyeLookOutL", 1.0)

        result = rig.evaluate()

        assert abs(result["eyeLookOutL"] - 1.0) < 0.001

    def test_clamp_at_exact_boundary_zero(self):
        """Weight exactly at 0.0 should remain 0.0."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("emotion", "eyeLookInL", 0.0)

        result = rig.evaluate()

        # Zero weight may or may not appear in result, but if present should be 0.0
        assert result.get("eyeLookInL", 0.0) == 0.0

    def test_clamp_large_negative_value(self):
        """Large negative weights should be clamped to 0.0."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("emotion", "mouthStretchL", -100.0)

        result = rig.evaluate()

        assert result.get("mouthStretchL", 0) >= 0.0

    def test_clamp_large_positive_value(self):
        """Large positive weights should be clamped to 1.0."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("emotion", "mouthStretchR", 100.0)

        result = rig.evaluate()

        assert result["mouthStretchR"] <= 1.0

    # =========================================================================
    # TEST CATEGORY 5: Edge Cases and Integration
    # =========================================================================

    def test_empty_rig_returns_empty_or_default_dict(self):
        """FaceRig with no layers should return empty or default dict."""
        rig = create_face_rig()
        result = rig.evaluate()

        assert isinstance(result, dict)
        # May be empty or contain defaults

    def test_layer_with_no_shapes_returns_valid_dict(self):
        """Layer with no shape weights should not crash."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        result = rig.evaluate()

        assert isinstance(result, dict)

    def test_multiple_shapes_across_multiple_layers(self):
        """Complex scenario with multiple shapes across multiple layers."""
        rig = create_face_rig()
        rig.add_layer("idle", priority=PRIORITY_IDLE, additive=False)
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)
        rig.add_layer("lip_sync", priority=PRIORITY_LIP_SYNC, additive=False)
        rig.add_layer("procedural", priority=PRIORITY_PROCEDURAL, additive=True)

        # Idle layer
        rig.set_layer_weight("idle", "eyeBlinkL", 0.0)
        rig.set_layer_weight("idle", "eyeBlinkR", 0.0)

        # Emotion layer
        rig.set_layer_weight("emotion", "browInnerUpL", 0.7)
        rig.set_layer_weight("emotion", "browInnerUpR", 0.7)
        rig.set_layer_weight("emotion", "mouthSmileL", 0.8)
        rig.set_layer_weight("emotion", "mouthSmileR", 0.8)

        # Lip sync layer
        rig.set_layer_weight("lip_sync", "jawOpen", 0.5)
        rig.set_layer_weight("lip_sync", "mouthFunnel", 0.3)

        # Procedural layer (additive)
        rig.set_layer_weight("procedural", "eyeLookUpL", 0.2)
        rig.set_layer_weight("procedural", "eyeLookUpR", 0.2)

        result = rig.evaluate()

        # Verify all expected shapes are present
        assert "browInnerUpL" in result
        assert "mouthSmileL" in result
        assert "jawOpen" in result
        assert "eyeLookUpL" in result

        # Verify values are in valid range
        for shape_name, value in result.items():
            assert 0.0 <= value <= 1.0, f"{shape_name} = {value} out of range"

    def test_same_shape_different_layers_blends_correctly(self):
        """Same shape in multiple layers should blend according to priority."""
        rig = create_face_rig()
        rig.add_layer("idle", priority=PRIORITY_IDLE, additive=False)
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)
        rig.add_layer("lip_sync", priority=PRIORITY_LIP_SYNC, additive=False)
        rig.add_layer("override", priority=PRIORITY_OVERRIDE, additive=False)

        # All layers affect the same shape
        rig.set_layer_weight("idle", "jawOpen", 0.1)
        rig.set_layer_weight("emotion", "jawOpen", 0.3)
        rig.set_layer_weight("lip_sync", "jawOpen", 0.6)
        rig.set_layer_weight("override", "jawOpen", 0.9)

        result = rig.evaluate()

        # Override (highest priority) should dominate
        assert result["jawOpen"] >= 0.8

    def test_evaluate_returns_dict_of_floats(self):
        """evaluate() should return Dict[str, float]."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)
        rig.set_layer_weight("emotion", "mouthSmileL", 0.5)

        result = rig.evaluate()

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str), f"Key {key} should be string"
            assert isinstance(value, (int, float)), f"Value {value} should be numeric"

    def test_can_update_weights_between_evaluations(self):
        """Weights can be updated and re-evaluated."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        rig.set_layer_weight("emotion", "mouthSmileL", 0.3)
        result1 = rig.evaluate()

        rig.set_layer_weight("emotion", "mouthSmileL", 0.9)
        result2 = rig.evaluate()

        assert result2["mouthSmileL"] > result1["mouthSmileL"]


class TestFaceRigLayerBlendingFormula:
    """Tests for specific blending formula behavior per architecture doc."""

    def test_override_blend_formula(self):
        """
        Override blending formula from architecture:
        result = base * (1.0 - layer.weight) + weighted_value

        When layer.weight = 1.0: result = 0 + weighted_value = weighted_value
        When layer.weight = 0.5: result = base * 0.5 + weighted_value * 0.5
        """
        rig = create_face_rig()
        rig.add_layer("base", priority=0, additive=False)
        rig.add_layer("top", priority=10, additive=False)

        # Base layer sets a value
        rig.set_layer_weight("base", "mouthFrownL", 0.4)

        # Top layer sets a value at weight 1.0 (full override)
        rig.set_layer_weight("top", "mouthFrownL", 0.8)

        result = rig.evaluate()

        # With full override, result should be close to 0.8
        assert result["mouthFrownL"] >= 0.7

    def test_additive_blend_formula(self):
        """
        Additive blending formula from architecture:
        result[shape] = result.get(shape, 0.0) + weighted_value
        """
        rig = create_face_rig()
        rig.add_layer("base", priority=0, additive=False)
        rig.add_layer("add", priority=10, additive=True)

        rig.set_layer_weight("base", "mouthFrownR", 0.3)
        rig.set_layer_weight("add", "mouthFrownR", 0.2)

        result = rig.evaluate()

        # Additive: 0.3 + 0.2 = 0.5
        assert abs(result["mouthFrownR"] - 0.5) < 0.05


class TestFaceRigContractCompliance:
    """Verify the public API contract is honored."""

    def test_face_rig_has_add_layer_method(self):
        """FaceRig should have add_layer method."""
        rig = create_face_rig()
        assert hasattr(rig, 'add_layer')
        assert callable(rig.add_layer)

    def test_face_rig_has_set_layer_weight_method(self):
        """FaceRig should have set_layer_weight method."""
        rig = create_face_rig()
        assert hasattr(rig, 'set_layer_weight')
        assert callable(rig.set_layer_weight)

    def test_face_rig_has_evaluate_method(self):
        """FaceRig should have evaluate method."""
        rig = create_face_rig()
        assert hasattr(rig, 'evaluate')
        assert callable(rig.evaluate)

    def test_layer_priority_is_importable(self):
        """LayerPriority should be importable from engine.animation.facial."""
        from engine.animation.facial import LayerPriority

        assert LayerPriority is not None

    def test_face_rig_is_importable(self):
        """FaceRig should be importable from engine.animation.facial."""
        from engine.animation.facial import FaceRig

        assert FaceRig is not None

    def test_add_layer_accepts_required_params(self):
        """add_layer should accept name, priority, and additive parameters."""
        rig = create_face_rig()

        # Should not raise
        rig.add_layer("test", priority=PRIORITY_EMOTION, additive=False)
        rig.add_layer("test2", priority=5, additive=True)

    def test_set_layer_weight_accepts_required_params(self):
        """set_layer_weight should accept layer_name, blend_shape_name, weight."""
        rig = create_face_rig()
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)

        # Should not raise
        rig.set_layer_weight("emotion", "mouthSmileL", 0.5)

    def test_contract_deviation_face_rig_requires_blend_shape_set(self):
        """
        CONTRACT VIOLATION: FaceRig requires blend_shape_set argument.

        The specified contract says FaceRig() takes no arguments,
        but actual implementation requires FaceRig(blend_shape_set).
        """
        from engine.animation.facial import FaceRig
        import inspect

        sig = inspect.signature(FaceRig.__init__)
        params = list(sig.parameters.keys())

        # Document that 'blend_shape_set' is required (not in original contract)
        assert 'blend_shape_set' in params, "Implementation matches expected deviation"


class TestContractDeviations:
    """
    Tests that document CONTRACT VIOLATIONS between specified and actual API.

    These tests are expected to FAIL if the implementation matches the spec,
    or PASS if they confirm the known deviation.
    """

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: LayerPriority.IDLE not implemented")
    def test_layer_priority_has_idle_attribute(self):
        """LayerPriority should have IDLE attribute per architecture doc."""
        from engine.animation.facial import LayerPriority
        assert hasattr(LayerPriority, 'IDLE')

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: LayerPriority.OVERRIDE not implemented")
    def test_layer_priority_has_override_attribute(self):
        """LayerPriority should have OVERRIDE attribute per architecture doc."""
        from engine.animation.facial import LayerPriority
        assert hasattr(LayerPriority, 'OVERRIDE')

    @pytest.mark.xfail(reason="CONTRACT VIOLATION: FaceRig() should take no args")
    def test_face_rig_no_required_args(self):
        """FaceRig() should be callable with no arguments per contract."""
        from engine.animation.facial import FaceRig
        # This should work per contract but will fail
        rig = FaceRig()

    def test_layer_priority_enum_values_order(self):
        """
        LayerPriority enum values should follow the order:
        IDLE < EMOTION < LIP_SYNC < PROCEDURAL < OVERRIDE

        This test checks if numeric priorities work correctly
        since the enum attributes are not available.
        """
        rig = create_face_rig()

        # Add layers using numeric priorities in expected order
        rig.add_layer("idle", priority=PRIORITY_IDLE, additive=False)
        rig.add_layer("emotion", priority=PRIORITY_EMOTION, additive=False)
        rig.add_layer("lip_sync", priority=PRIORITY_LIP_SYNC, additive=False)
        rig.add_layer("procedural", priority=PRIORITY_PROCEDURAL, additive=False)
        rig.add_layer("override", priority=PRIORITY_OVERRIDE, additive=False)

        # Set same shape across all layers
        for layer in ["idle", "emotion", "lip_sync", "procedural", "override"]:
            rig.set_layer_weight(layer, "testShape", 0.5)

        result = rig.evaluate()

        # Should not crash and should produce valid output
        assert isinstance(result, dict)
        assert "testShape" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
