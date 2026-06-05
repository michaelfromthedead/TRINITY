"""
Whitebox Tests for Face Rig Layer Blending System (T3.5).

Tests internal implementation details of:
1. Higher priority layers override lower priority layers
2. Additive layers accumulate correctly
3. Layer master weights scale contributions
4. Blend shape clamping to [0, 1]
5. Edge cases: empty layers, missing shapes, duplicate layers
6. Integration with FACSController, LipSyncController, EyeController
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from engine.animation.facial.blend_shapes import (
    BlendShape,
    BlendShapeController,
    BlendShapeSet,
    create_arkit_compatible_set,
)
from engine.animation.facial.eye_animation import EyeController
from engine.animation.facial.facs import (
    ActionUnit,
    Expression,
    FACSController,
)
from engine.animation.facial.face_rig import (
    AnimationLayer,
    AnimationPriority,
    EmotionState,
    FaceRig,
    LayerPriority,
    RigLayer,
    create_face_rig,
)
from engine.animation.facial.lip_sync import LipSyncController


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_shape_set() -> BlendShapeSet:
    """Create a minimal blend shape set for testing."""
    return BlendShapeSet(
        name="test",
        base_vertices=np.zeros((100, 3), dtype=np.float32),
    )


@pytest.fixture
def arkit_shape_set() -> BlendShapeSet:
    """Create an ARKit-compatible blend shape set."""
    return create_arkit_compatible_set("face", 100)


@pytest.fixture
def face_rig(minimal_shape_set: BlendShapeSet) -> FaceRig:
    """Create a FaceRig instance for testing."""
    return FaceRig(blend_shape_set=minimal_shape_set)


@pytest.fixture
def face_rig_with_arkit(arkit_shape_set: BlendShapeSet) -> FaceRig:
    """Create a FaceRig with ARKit shapes."""
    return FaceRig(blend_shape_set=arkit_shape_set)


@pytest.fixture
def mock_facs_controller() -> MagicMock:
    """Create a mock FACSController."""
    mock = MagicMock(spec=FACSController)
    mock.get_blend_shape_weights.return_value = {}
    return mock


@pytest.fixture
def mock_eye_controller() -> MagicMock:
    """Create a mock EyeController."""
    mock = MagicMock(spec=EyeController)
    mock.get_blend_shape_weights.return_value = {}
    return mock


@pytest.fixture
def mock_lip_sync_controller() -> MagicMock:
    """Create a mock LipSyncController."""
    mock = MagicMock(spec=LipSyncController)
    mock.update.return_value = {}
    mock.is_playing = False
    return mock


# =============================================================================
# Acceptance Criteria Tests
# =============================================================================


class TestAC1_HigherPriorityOverridesLower:
    """AC1: Higher priority layers override lower priority layers."""

    def test_emotion_overrides_base_layer(self, face_rig: FaceRig) -> None:
        """EMOTION (10) should override BASE (0) layer."""
        face_rig.set_layer_weight("idle", "jawOpen", 0.3)
        face_rig.set_layer_weight("emotion", "jawOpen", 0.9)

        result = face_rig.evaluate()

        # Higher priority emotion layer completely replaces idle value
        assert abs(result["jawOpen"] - 0.9) < 0.001

    def test_lip_sync_overrides_emotion_when_not_additive(
        self, face_rig: FaceRig
    ) -> None:
        """When lip_sync is override mode, it should replace emotion."""
        # Make lip_sync non-additive for this test
        lip_sync_layer = face_rig.get_rig_layer("lip_sync")
        lip_sync_layer.additive = False

        face_rig.set_layer_weight("emotion", "mouthOpen", 0.5)
        face_rig.set_layer_weight("lip_sync", "mouthOpen", 0.8)

        result = face_rig.evaluate()

        # lip_sync (20) overrides emotion (10)
        assert abs(result["mouthOpen"] - 0.8) < 0.001

    def test_multiple_override_layers_highest_wins(self, face_rig: FaceRig) -> None:
        """When multiple override layers set same shape, highest priority wins."""
        face_rig.add_layer("custom_low", priority=5, additive=False)
        face_rig.add_layer("custom_mid", priority=15, additive=False)
        face_rig.add_layer("custom_high", priority=25, additive=False)

        face_rig.set_layer_weight("custom_low", "testShape", 0.2)
        face_rig.set_layer_weight("custom_mid", "testShape", 0.5)
        face_rig.set_layer_weight("custom_high", "testShape", 0.8)

        result = face_rig.evaluate()

        # Highest priority (25) wins
        assert abs(result["testShape"] - 0.8) < 0.001

    def test_lower_priority_value_completely_replaced(
        self, face_rig: FaceRig
    ) -> None:
        """Lower priority values should be completely replaced, not blended."""
        face_rig.set_layer_weight("idle", "browInnerUp", 1.0)  # Maximum value
        face_rig.set_layer_weight("emotion", "browInnerUp", 0.1)  # Small value

        result = face_rig.evaluate()

        # Should be 0.1, not a blend between 1.0 and 0.1
        assert abs(result["browInnerUp"] - 0.1) < 0.001

    def test_priority_order_is_numeric_not_insertion_order(
        self, face_rig: FaceRig
    ) -> None:
        """Layers should be evaluated by numeric priority, not insertion order."""
        # Add in reverse priority order
        face_rig.add_layer("p30", priority=30, additive=False)
        face_rig.add_layer("p10", priority=10, additive=False)
        face_rig.add_layer("p20", priority=20, additive=False)

        face_rig.set_layer_weight("p10", "shape", 0.1)
        face_rig.set_layer_weight("p20", "shape", 0.2)
        face_rig.set_layer_weight("p30", "shape", 0.3)

        result = face_rig.evaluate()

        # p30 (highest priority) wins
        assert abs(result["shape"] - 0.3) < 0.001


class TestAC2_AdditiveLayersAccumulate:
    """AC2: Additive layers accumulate correctly."""

    def test_additive_adds_to_override_value(self, face_rig: FaceRig) -> None:
        """Additive layer should add to the override layer value."""
        face_rig.set_layer_weight("emotion", "browDownL", 0.3)
        face_rig.set_layer_weight("procedural", "browDownL", 0.2)

        result = face_rig.evaluate()

        # 0.3 (emotion) + 0.2 (procedural additive) = 0.5
        assert abs(result["browDownL"] - 0.5) < 0.001

    def test_multiple_additive_layers_accumulate(self, face_rig: FaceRig) -> None:
        """Multiple additive layers should all accumulate."""
        face_rig.add_layer("additive1", priority=15, additive=True)
        face_rig.add_layer("additive2", priority=25, additive=True)
        face_rig.add_layer("additive3", priority=35, additive=True)

        face_rig.set_layer_weight("emotion", "noseSneerLeft", 0.1)
        face_rig.set_layer_weight("additive1", "noseSneerLeft", 0.1)
        face_rig.set_layer_weight("additive2", "noseSneerLeft", 0.1)
        face_rig.set_layer_weight("additive3", "noseSneerLeft", 0.1)

        result = face_rig.evaluate()

        # All should accumulate: 0.1 + 0.1 + 0.1 + 0.1 = 0.4
        assert abs(result["noseSneerLeft"] - 0.4) < 0.001

    def test_additive_layer_on_zero_base(self, face_rig: FaceRig) -> None:
        """Additive layer should work when no base value exists."""
        face_rig.set_layer_weight("procedural", "newShape", 0.5)

        result = face_rig.evaluate()

        # Should be 0.5 (0 base + 0.5 additive)
        assert abs(result["newShape"] - 0.5) < 0.001

    def test_additive_layer_order_does_not_matter(self, face_rig: FaceRig) -> None:
        """Additive layers should produce same result regardless of evaluation order."""
        face_rig.add_layer("add1", priority=5, additive=True)
        face_rig.add_layer("add2", priority=25, additive=True)

        face_rig.set_layer_weight("add1", "shape", 0.3)
        face_rig.set_layer_weight("add2", "shape", 0.2)

        result = face_rig.evaluate()

        # Both additive, so 0.3 + 0.2 = 0.5
        assert abs(result["shape"] - 0.5) < 0.001

    def test_override_then_additive_chain(self, face_rig: FaceRig) -> None:
        """Override layer sets base, then additive layers add to it."""
        face_rig.add_layer("override1", priority=5, additive=False)
        face_rig.add_layer("override2", priority=15, additive=False)
        face_rig.add_layer("additive1", priority=25, additive=True)

        face_rig.set_layer_weight("override1", "shape", 0.2)
        face_rig.set_layer_weight("override2", "shape", 0.4)
        face_rig.set_layer_weight("additive1", "shape", 0.3)

        result = face_rig.evaluate()

        # override2 (0.4) replaces override1 (0.2), then additive1 adds 0.3
        # Result: 0.4 + 0.3 = 0.7
        assert abs(result["shape"] - 0.7) < 0.001


class TestAC3_LayerWeightsScale:
    """AC3: Layer weights scale contributions."""

    def test_master_weight_scales_all_blend_shapes(self, face_rig: FaceRig) -> None:
        """Master weight should scale all blend shapes in the layer."""
        face_rig.set_layer_weight("emotion", "jawOpen", 1.0)
        face_rig.set_layer_weight("emotion", "mouthSmile", 0.8)
        face_rig.set_rig_layer_master_weight("emotion", 0.5)

        result = face_rig.evaluate()

        # Both should be scaled by 0.5
        assert abs(result["jawOpen"] - 0.5) < 0.001
        assert abs(result["mouthSmile"] - 0.4) < 0.001

    def test_zero_master_weight_produces_zero(self, face_rig: FaceRig) -> None:
        """Zero master weight should produce zero contribution."""
        face_rig.set_layer_weight("emotion", "testShape", 1.0)
        face_rig.set_rig_layer_master_weight("emotion", 0.0)

        result = face_rig.evaluate()

        assert result.get("testShape", 0.0) == 0.0

    def test_very_small_master_weight_ignored(self, face_rig: FaceRig) -> None:
        """Very small master weight (< 0.001) should be ignored."""
        face_rig.set_layer_weight("emotion", "testShape", 1.0)
        face_rig.set_rig_layer_master_weight("emotion", 0.0005)

        result = face_rig.evaluate()

        # Should be ignored (weight <= 0.001)
        assert "testShape" not in result or result["testShape"] == 0.0

    def test_master_weight_combined_with_blend_weight(
        self, face_rig: FaceRig
    ) -> None:
        """Master weight should multiply with individual blend shape weight."""
        face_rig.set_layer_weight("emotion", "shape", 0.8)
        face_rig.set_rig_layer_master_weight("emotion", 0.25)

        result = face_rig.evaluate()

        # 0.8 * 0.25 = 0.2
        assert abs(result["shape"] - 0.2) < 0.001

    def test_additive_layer_master_weight_scales(self, face_rig: FaceRig) -> None:
        """Additive layer contributions should also be scaled by master weight."""
        face_rig.set_layer_weight("emotion", "shape", 0.4)
        face_rig.set_layer_weight("procedural", "shape", 0.8)
        face_rig.set_rig_layer_master_weight("procedural", 0.5)

        result = face_rig.evaluate()

        # emotion: 0.4, procedural additive: 0.8 * 0.5 = 0.4
        # Total: 0.4 + 0.4 = 0.8
        assert abs(result["shape"] - 0.8) < 0.001


class TestAC4_BlendShapeClamping:
    """AC4: Blend shape clamping to [0, 1]."""

    def test_result_clamped_to_max_one(self, face_rig: FaceRig) -> None:
        """Accumulated values exceeding 1.0 should be clamped."""
        face_rig.set_layer_weight("emotion", "shape", 0.7)
        face_rig.set_layer_weight("procedural", "shape", 0.5)

        result = face_rig.evaluate()

        # 0.7 + 0.5 = 1.2, should be clamped to 1.0
        assert result["shape"] == 1.0

    def test_result_clamped_to_min_zero(self, face_rig: FaceRig) -> None:
        """Negative values should be clamped to 0.0."""
        # Directly set negative value in layer's blend_shapes
        face_rig._rig_layers["emotion"].blend_shapes["shape"] = -0.5

        result = face_rig.evaluate()

        # Should be clamped to 0.0
        assert result["shape"] == 0.0

    def test_input_weight_clamped_on_set(self, face_rig: FaceRig) -> None:
        """Input weights should be clamped when set via set_layer_weight."""
        face_rig.set_layer_weight("emotion", "shape", 1.5)
        layer = face_rig.get_rig_layer("emotion")

        assert layer.blend_shapes["shape"] == 1.0

        face_rig.set_layer_weight("emotion", "shape2", -0.5)
        assert layer.blend_shapes["shape2"] == 0.0

    def test_master_weight_clamped_on_set(self, face_rig: FaceRig) -> None:
        """Master weight should be clamped to [0, 1]."""
        result1 = face_rig.set_rig_layer_master_weight("emotion", 1.5)
        assert result1 is True
        assert face_rig.get_rig_layer("emotion").weight == 1.0

        result2 = face_rig.set_rig_layer_master_weight("emotion", -0.5)
        assert result2 is True
        assert face_rig.get_rig_layer("emotion").weight == 0.0

    def test_many_additive_layers_still_clamp(self, face_rig: FaceRig) -> None:
        """Even many additive layers should clamp final result."""
        for i in range(10):
            face_rig.add_layer(f"add{i}", priority=i * 2 + 5, additive=True)
            face_rig.set_layer_weight(f"add{i}", "shape", 0.2)

        result = face_rig.evaluate()

        # 10 * 0.2 = 2.0, should be clamped to 1.0
        assert result["shape"] == 1.0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCasesEmptyLayers:
    """Edge cases: empty layers."""

    def test_evaluate_with_no_blend_shapes(self, face_rig: FaceRig) -> None:
        """Evaluating with no blend shapes set should return empty dict."""
        result = face_rig.evaluate()
        assert result == {}

    def test_layer_with_empty_blend_shapes(self, face_rig: FaceRig) -> None:
        """Layer with empty blend_shapes dict should contribute nothing."""
        face_rig.get_rig_layer("emotion").blend_shapes = {}
        face_rig.set_layer_weight("idle", "shape", 0.5)

        result = face_rig.evaluate()

        assert abs(result["shape"] - 0.5) < 0.001

    def test_all_layers_empty(self, face_rig: FaceRig) -> None:
        """All layers having empty blend_shapes should return empty result."""
        for layer in face_rig._rig_layers.values():
            layer.blend_shapes = {}

        result = face_rig.evaluate()
        assert result == {}

    def test_clear_layer_removes_all_shapes(self, face_rig: FaceRig) -> None:
        """clear_rig_layer should remove all blend shapes."""
        face_rig.set_layer_weight("emotion", "shape1", 0.5)
        face_rig.set_layer_weight("emotion", "shape2", 0.7)

        face_rig.clear_rig_layer("emotion")

        layer = face_rig.get_rig_layer("emotion")
        assert len(layer.blend_shapes) == 0


class TestEdgeCasesMissingShapes:
    """Edge cases: missing shapes."""

    def test_set_weight_for_new_shape_creates_it(self, face_rig: FaceRig) -> None:
        """Setting weight for a shape that doesn't exist should create it."""
        face_rig.set_layer_weight("emotion", "newShape", 0.5)

        layer = face_rig.get_rig_layer("emotion")
        assert "newShape" in layer.blend_shapes
        assert layer.blend_shapes["newShape"] == 0.5

    def test_get_nonexistent_layer_returns_none(self, face_rig: FaceRig) -> None:
        """Getting a non-existent layer should return None."""
        result = face_rig.get_rig_layer("nonexistent")
        assert result is None

    def test_set_weight_nonexistent_layer_returns_false(
        self, face_rig: FaceRig
    ) -> None:
        """Setting weight for non-existent layer should return False."""
        result = face_rig.set_layer_weight("nonexistent", "shape", 0.5)
        assert result is False

    def test_clear_nonexistent_layer_returns_false(self, face_rig: FaceRig) -> None:
        """Clearing non-existent layer should return False."""
        result = face_rig.clear_rig_layer("nonexistent")
        assert result is False

    def test_remove_nonexistent_layer_returns_false(self, face_rig: FaceRig) -> None:
        """Removing non-existent layer should return False."""
        result = face_rig.remove_rig_layer("nonexistent")
        assert result is False


class TestEdgeCasesDuplicateLayers:
    """Edge cases: duplicate layers."""

    def test_add_layer_same_name_overwrites(self, face_rig: FaceRig) -> None:
        """Adding layer with same name should overwrite existing."""
        face_rig.add_layer("custom", priority=10, additive=False)
        face_rig.set_layer_weight("custom", "shape", 0.5)

        # Overwrite with new layer
        face_rig.add_layer("custom", priority=20, additive=True)

        layer = face_rig.get_rig_layer("custom")
        assert layer.priority == 20
        assert layer.additive is True
        assert "shape" not in layer.blend_shapes  # New layer, no shapes

    def test_same_priority_layers_last_evaluated_wins(
        self, face_rig: FaceRig
    ) -> None:
        """Layers with same priority: iteration order determines winner."""
        face_rig.add_layer("layer1", priority=15, additive=False)
        face_rig.add_layer("layer2", priority=15, additive=False)

        face_rig.set_layer_weight("layer1", "shape", 0.3)
        face_rig.set_layer_weight("layer2", "shape", 0.7)

        result = face_rig.evaluate()

        # Result should be one of them, not blended
        assert result["shape"] in [0.3, 0.7]

    def test_remove_then_add_same_name_layer(self, face_rig: FaceRig) -> None:
        """Removing then adding layer with same name should work."""
        face_rig.add_layer("temp", priority=50)
        face_rig.set_layer_weight("temp", "shape", 0.5)
        face_rig.remove_rig_layer("temp")

        face_rig.add_layer("temp", priority=60, additive=True)
        layer = face_rig.get_rig_layer("temp")

        assert layer is not None
        assert layer.priority == 60
        assert layer.additive is True


# =============================================================================
# Integration with Subsystems
# =============================================================================


class TestFACSControllerIntegration:
    """Integration tests with FACSController."""

    def test_face_rig_creates_facs_controller(
        self, minimal_shape_set: BlendShapeSet
    ) -> None:
        """FaceRig should create a FACSController if not provided."""
        rig = FaceRig(blend_shape_set=minimal_shape_set)
        assert rig.facs_controller is not None
        assert isinstance(rig.facs_controller, FACSController)

    def test_face_rig_uses_provided_facs_controller(
        self,
        minimal_shape_set: BlendShapeSet,
        mock_facs_controller: MagicMock,
    ) -> None:
        """FaceRig should use provided FACSController."""
        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            facs_controller=mock_facs_controller,
        )
        assert rig.facs_controller is mock_facs_controller

    def test_set_expression_updates_emotion_state(
        self, face_rig_with_arkit: FaceRig
    ) -> None:
        """set_expression should update emotion state and FACS weights."""
        face_rig_with_arkit.set_expression(Expression.HAPPY, intensity=0.8, blend_time=0)

        # Verify emotion state was updated
        assert face_rig_with_arkit.current_emotion.expression == Expression.HAPPY
        assert face_rig_with_arkit.current_emotion.intensity == 0.8

        # Verify emotion layer has blend shapes from FACS
        emotion_layer = face_rig_with_arkit.get_layer("emotion")
        assert len(emotion_layer.blend_shapes) > 0

    def test_emotion_layer_gets_facs_weights(
        self, face_rig_with_arkit: FaceRig
    ) -> None:
        """Emotion layer should receive weights from FACS controller."""
        face_rig_with_arkit.set_expression(Expression.HAPPY, intensity=1.0, blend_time=0)

        # Check emotion layer has the FACS-generated blend shapes
        emotion_layer = face_rig_with_arkit.get_layer("emotion")
        assert emotion_layer is not None
        # HAPPY expression includes mouthSmileLeft/Right from FACS
        assert "mouthSmileLeft" in emotion_layer.blend_shapes or \
               "mouthSmileRight" in emotion_layer.blend_shapes

    def test_set_action_unit_calls_facs_controller(
        self, face_rig_with_arkit: FaceRig
    ) -> None:
        """set_action_unit should call FACS controller's set_au_intensity."""
        with patch.object(
            face_rig_with_arkit.facs_controller,
            "set_au_intensity"
        ) as mock_set_au:
            face_rig_with_arkit.set_action_unit(ActionUnit.AU12_LIP_CORNER_PULLER, 0.8)

            mock_set_au.assert_called_once_with(
                ActionUnit.AU12_LIP_CORNER_PULLER, 0.8, None, None
            )


class TestLipSyncControllerIntegration:
    """Integration tests with LipSyncController."""

    def test_face_rig_creates_lip_sync_controller(
        self, minimal_shape_set: BlendShapeSet
    ) -> None:
        """FaceRig should create a LipSyncController if not provided."""
        rig = FaceRig(blend_shape_set=minimal_shape_set)
        assert rig.lip_sync_controller is not None
        assert isinstance(rig.lip_sync_controller, LipSyncController)

    def test_face_rig_uses_provided_lip_sync_controller(
        self,
        minimal_shape_set: BlendShapeSet,
        mock_lip_sync_controller: MagicMock,
    ) -> None:
        """FaceRig should use provided LipSyncController."""
        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            lip_sync_controller=mock_lip_sync_controller,
        )
        assert rig.lip_sync_controller is mock_lip_sync_controller

    def test_update_calls_lip_sync_update(
        self,
        minimal_shape_set: BlendShapeSet,
        mock_lip_sync_controller: MagicMock,
    ) -> None:
        """update() should call lip_sync_controller.update()."""
        mock_lip_sync_controller.update.return_value = {"jawOpen": 0.5}

        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            lip_sync_controller=mock_lip_sync_controller,
        )
        rig.update(0.016)

        mock_lip_sync_controller.update.assert_called_once_with(0.016)

    def test_lip_sync_weights_go_to_lip_sync_layer(
        self,
        minimal_shape_set: BlendShapeSet,
        mock_lip_sync_controller: MagicMock,
    ) -> None:
        """Lip sync weights should be set on the lip_sync layer."""
        mock_lip_sync_controller.update.return_value = {
            "jawOpen": 0.6,
            "mouthFunnel": 0.3,
        }

        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            lip_sync_controller=mock_lip_sync_controller,
        )
        rig.update(0.016)

        lip_sync_layer = rig.get_layer("lip_sync")
        assert lip_sync_layer.blend_shapes.get("jawOpen") == 0.6
        assert lip_sync_layer.blend_shapes.get("mouthFunnel") == 0.3

    def test_is_speaking_reflects_lip_sync_state(
        self,
        minimal_shape_set: BlendShapeSet,
        mock_lip_sync_controller: MagicMock,
    ) -> None:
        """is_speaking should reflect lip_sync_controller.is_playing."""
        mock_lip_sync_controller.is_playing = True

        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            lip_sync_controller=mock_lip_sync_controller,
        )

        assert rig.is_speaking is True


class TestEyeControllerIntegration:
    """Integration tests with EyeController."""

    def test_face_rig_creates_eye_controller(
        self, minimal_shape_set: BlendShapeSet
    ) -> None:
        """FaceRig should create an EyeController if not provided."""
        rig = FaceRig(blend_shape_set=minimal_shape_set)
        assert rig.eye_controller is not None
        assert isinstance(rig.eye_controller, EyeController)

    def test_face_rig_uses_provided_eye_controller(
        self,
        minimal_shape_set: BlendShapeSet,
        mock_eye_controller: MagicMock,
    ) -> None:
        """FaceRig should use provided EyeController."""
        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            eye_controller=mock_eye_controller,
        )
        assert rig.eye_controller is mock_eye_controller

    def test_update_calls_eye_controller_update(
        self,
        minimal_shape_set: BlendShapeSet,
        mock_eye_controller: MagicMock,
    ) -> None:
        """update() should call eye_controller.update()."""
        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            eye_controller=mock_eye_controller,
        )
        rig.update(0.016)

        mock_eye_controller.update.assert_called_once_with(0.016)

    def test_eye_weights_go_to_eyes_layer(
        self,
        minimal_shape_set: BlendShapeSet,
        mock_eye_controller: MagicMock,
    ) -> None:
        """Eye weights should be set on the eyes layer."""
        mock_eye_controller.get_blend_shape_weights.return_value = {
            "eyeBlinkLeft": 0.5,
            "eyeBlinkRight": 0.5,
        }

        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            eye_controller=mock_eye_controller,
        )
        rig.update(0.016)

        eyes_layer = rig.get_layer("eyes")
        assert eyes_layer.blend_shapes.get("eyeBlinkLeft") == 0.5
        assert eyes_layer.blend_shapes.get("eyeBlinkRight") == 0.5

    def test_look_at_calls_eye_controller(
        self,
        minimal_shape_set: BlendShapeSet,
        mock_eye_controller: MagicMock,
    ) -> None:
        """look_at() should call eye_controller.look_at()."""
        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            eye_controller=mock_eye_controller,
        )

        rig.look_at((1.0, 0.0, 2.0), weight=0.8, smooth_speed=5.0)

        mock_eye_controller.look_at.assert_called_once_with(
            (1.0, 0.0, 2.0), 0.8, 5.0
        )

    def test_blink_calls_eye_controller(
        self,
        minimal_shape_set: BlendShapeSet,
        mock_eye_controller: MagicMock,
    ) -> None:
        """blink() should call eye_controller.blink()."""
        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            eye_controller=mock_eye_controller,
        )

        rig.blink(intensity=0.7)

        mock_eye_controller.blink.assert_called_once_with(0.7)


# =============================================================================
# Internal Implementation Tests
# =============================================================================


class TestEvaluateMethod:
    """Tests for the evaluate() method internals."""

    def test_evaluate_sorts_layers_by_priority(self, face_rig: FaceRig) -> None:
        """evaluate() should process layers in ascending priority order."""
        # Track evaluation order by setting unique values
        face_rig.add_layer("p5", priority=5, additive=False)
        face_rig.add_layer("p25", priority=25, additive=False)
        face_rig.add_layer("p15", priority=15, additive=False)

        face_rig.set_layer_weight("p5", "shape", 0.1)
        face_rig.set_layer_weight("p15", "shape", 0.3)
        face_rig.set_layer_weight("p25", "shape", 0.5)

        result = face_rig.evaluate()

        # Highest priority (p25) should win
        assert abs(result["shape"] - 0.5) < 0.001

    def test_evaluate_skips_zero_weight_layers(self, face_rig: FaceRig) -> None:
        """evaluate() should skip layers with zero or near-zero weight."""
        face_rig.set_layer_weight("emotion", "shape", 0.8)
        face_rig.set_rig_layer_master_weight("emotion", 0.0)

        face_rig.set_layer_weight("idle", "shape", 0.3)

        result = face_rig.evaluate()

        # emotion skipped, idle should provide the value
        assert abs(result["shape"] - 0.3) < 0.001

    def test_evaluate_returns_copy_not_reference(self, face_rig: FaceRig) -> None:
        """evaluate() should return a new dict, not internal reference."""
        face_rig.set_layer_weight("emotion", "shape", 0.5)

        result1 = face_rig.evaluate()
        result2 = face_rig.evaluate()

        # Modify result1 should not affect result2
        result1["shape"] = 999.0

        assert result2["shape"] != 999.0


class TestBlendLayersMethod:
    """Tests for the _blend_layers() method (legacy system)."""

    def test_blend_layers_handles_additive(self, face_rig: FaceRig) -> None:
        """_blend_layers should handle additive layers correctly."""
        face_rig._layers["emotion"].blend_shapes = {"shape": 0.3}
        face_rig._layers["lip_sync"].blend_shapes = {"shape": 0.2}
        face_rig._layers["lip_sync"].is_additive = True

        result = face_rig._blend_layers()

        # lip_sync is additive, should add to emotion
        assert abs(result["shape"] - 0.5) < 0.001

    def test_blend_layers_clamps_output(self, face_rig: FaceRig) -> None:
        """_blend_layers should clamp all output values to [0, 1]."""
        face_rig._layers["emotion"].blend_shapes = {"shape": 0.8}
        face_rig._layers["lip_sync"].blend_shapes = {"shape": 0.5}
        face_rig._layers["lip_sync"].is_additive = True

        result = face_rig._blend_layers()

        # 0.8 + 0.5 = 1.3, should be clamped
        assert result["shape"] == 1.0


class TestRigLayerDataclass:
    """Tests for RigLayer dataclass."""

    def test_riglayer_default_values(self) -> None:
        """RigLayer should have correct default values."""
        layer = RigLayer(name="test", priority=0)

        assert layer.name == "test"
        assert layer.priority == 0
        assert layer.weight == 1.0
        assert layer.additive is False
        assert layer.blend_shapes == {}

    def test_riglayer_with_all_args(self) -> None:
        """RigLayer should accept all arguments."""
        layer = RigLayer(
            name="full",
            priority=50,
            weight=0.7,
            blend_shapes={"a": 0.5, "b": 0.3},
            additive=True,
        )

        assert layer.name == "full"
        assert layer.priority == 50
        assert layer.weight == 0.7
        assert layer.additive is True
        assert layer.blend_shapes == {"a": 0.5, "b": 0.3}

    def test_riglayer_blend_shapes_mutable(self) -> None:
        """RigLayer.blend_shapes should be mutable."""
        layer = RigLayer(name="test", priority=0)
        layer.blend_shapes["newShape"] = 0.8

        assert layer.blend_shapes["newShape"] == 0.8


class TestAnimationLayerDataclass:
    """Tests for AnimationLayer dataclass (legacy system)."""

    def test_animation_layer_defaults(self) -> None:
        """AnimationLayer should have correct defaults."""
        layer = AnimationLayer(name="test", priority=AnimationPriority.IDLE)

        assert layer.name == "test"
        assert layer.priority == AnimationPriority.IDLE
        assert layer.weight == 1.0
        assert layer.is_additive is False
        assert layer.blend_shapes == {}


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Tests for state management (dirty flag, reset, etc.)."""

    def test_dirty_flag_set_on_update(self, face_rig: FaceRig) -> None:
        """Dirty flag should be set after update()."""
        face_rig.clear_dirty()
        assert face_rig.dirty is False

        face_rig.update(0.016)

        assert face_rig.dirty is True

    def test_clear_dirty_clears_flag(self, face_rig: FaceRig) -> None:
        """clear_dirty() should clear the dirty flag."""
        face_rig.update(0.016)
        assert face_rig.dirty is True

        face_rig.clear_dirty()

        assert face_rig.dirty is False

    def test_reset_clears_all_state(self, face_rig: FaceRig) -> None:
        """reset() should clear all facial animation state."""
        face_rig.set_layer_weight("emotion", "shape", 0.5)
        face_rig.set_expression(Expression.HAPPY)

        face_rig.reset()

        # Check layers are cleared
        emotion_layer = face_rig.get_layer("emotion")
        assert len(emotion_layer.blend_shapes) == 0

        # Check emotion is reset
        assert face_rig.current_emotion.expression == Expression.NEUTRAL

    def test_on_weights_changed_callback(
        self, minimal_shape_set: BlendShapeSet
    ) -> None:
        """on_weights_changed callback should be called on update."""
        callback = MagicMock()
        rig = FaceRig(
            blend_shape_set=minimal_shape_set,
            on_weights_changed=callback,
        )

        rig.set_layer_weight("emotion", "shape", 0.5)
        rig.update(0.016)

        callback.assert_called()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateFaceRig:
    """Tests for create_face_rig factory function."""

    def test_create_face_rig_default(self) -> None:
        """create_face_rig should create a working FaceRig."""
        rig = create_face_rig(vertex_count=100)

        assert rig is not None
        assert isinstance(rig, FaceRig)

    def test_create_face_rig_with_arkit(self) -> None:
        """create_face_rig with ARKit shapes should have all ARKit shapes."""
        rig = create_face_rig(vertex_count=100, use_arkit_shapes=True)

        # Check for some ARKit shape names
        shape_set = rig.blend_controller.shape_set
        assert shape_set.has_shape("jawOpen")
        assert shape_set.has_shape("eyeBlinkLeft")
        assert shape_set.has_shape("mouthSmileRight")

    def test_create_face_rig_without_arkit(self) -> None:
        """create_face_rig without ARKit should create minimal shape set."""
        rig = create_face_rig(vertex_count=100, use_arkit_shapes=False)

        shape_set = rig.blend_controller.shape_set
        assert shape_set.shape_count == 0  # No pre-defined shapes


# =============================================================================
# Performance / Stress Tests
# =============================================================================


class TestPerformance:
    """Performance and stress tests."""

    def test_many_layers_evaluate_correctly(self, face_rig: FaceRig) -> None:
        """Many layers should still evaluate correctly."""
        # Add 20 override layers with increasing priority
        for i in range(20):
            face_rig.add_layer(f"layer{i}", priority=i * 10, additive=False)
            face_rig.set_layer_weight(f"layer{i}", "shape", i * 0.05)

        result = face_rig.evaluate()

        # Highest priority (layer19 at priority 190) should win
        # Value: 19 * 0.05 = 0.95
        assert abs(result["shape"] - 0.95) < 0.001

    def test_many_blend_shapes_per_layer(self, face_rig: FaceRig) -> None:
        """Many blend shapes per layer should work correctly."""
        for i in range(100):
            face_rig.set_layer_weight("emotion", f"shape{i}", i * 0.01)

        result = face_rig.evaluate()

        assert len(result) == 100
        assert abs(result["shape50"] - 0.5) < 0.001
        assert result["shape99"] == 0.99

    def test_evaluate_is_repeatable(self, face_rig: FaceRig) -> None:
        """Multiple evaluate() calls should produce same result."""
        face_rig.set_layer_weight("emotion", "shape", 0.5)
        face_rig.set_layer_weight("procedural", "shape", 0.2)

        result1 = face_rig.evaluate()
        result2 = face_rig.evaluate()
        result3 = face_rig.evaluate()

        assert result1["shape"] == result2["shape"] == result3["shape"]
