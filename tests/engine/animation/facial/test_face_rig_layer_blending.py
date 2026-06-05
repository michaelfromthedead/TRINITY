"""
Tests for Face Rig Layer Blending System (T3.5).

Validates:
1. Higher priority layers override lower priority layers
2. Additive layers accumulate correctly
3. Layer master weights scale contributions
4. All subsystems integrate correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.animation.facial.blend_shapes import BlendShapeSet
from engine.animation.facial.face_rig import (
    FaceRig,
    LayerPriority,
    RigLayer,
)


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
def face_rig(minimal_shape_set: BlendShapeSet) -> FaceRig:
    """Create a FaceRig instance for testing."""
    return FaceRig(blend_shape_set=minimal_shape_set)


# =============================================================================
# LayerPriority Tests
# =============================================================================


class TestLayerPriority:
    """Tests for LayerPriority constants."""

    def test_priority_values(self) -> None:
        """Verify priority values are as specified."""
        assert LayerPriority.BASE == 0
        assert LayerPriority.EMOTION == 10
        assert LayerPriority.LIP_SYNC == 20
        assert LayerPriority.PROCEDURAL == 30

    def test_priority_ordering(self) -> None:
        """Verify priority ordering is correct."""
        assert LayerPriority.BASE < LayerPriority.EMOTION
        assert LayerPriority.EMOTION < LayerPriority.LIP_SYNC
        assert LayerPriority.LIP_SYNC < LayerPriority.PROCEDURAL


# =============================================================================
# RigLayer Tests
# =============================================================================


class TestRigLayer:
    """Tests for RigLayer dataclass."""

    def test_layer_creation(self) -> None:
        """Test creating a rig layer."""
        layer = RigLayer(
            name="test",
            priority=LayerPriority.EMOTION,
            weight=0.8,
            additive=True,
        )
        assert layer.name == "test"
        assert layer.priority == LayerPriority.EMOTION
        assert layer.weight == 0.8
        assert layer.additive is True
        assert layer.blend_shapes == {}

    def test_layer_defaults(self) -> None:
        """Test default values for rig layer."""
        layer = RigLayer(name="default", priority=0)
        assert layer.weight == 1.0
        assert layer.additive is False
        assert layer.blend_shapes == {}


# =============================================================================
# Layer Priority Override Tests
# =============================================================================


class TestLayerPriorityOverride:
    """Tests for layer priority override behavior."""

    def test_layer_priority_override(self, face_rig: FaceRig) -> None:
        """Higher priority layers should override lower priority layers.

        Test contract from acceptance criteria.
        """
        face_rig.set_layer_weight("idle", "mouthSmileL", 0.5)
        face_rig.set_layer_weight("emotion", "mouthSmileL", 1.0)  # Higher priority
        result = face_rig.evaluate()
        assert result["mouthSmileL"] > 0.5  # emotion overrides

    def test_emotion_completely_overrides_idle(self, face_rig: FaceRig) -> None:
        """Emotion layer (priority 10) should completely override idle (priority 0)."""
        face_rig.set_layer_weight("idle", "browInnerUp", 0.3)
        face_rig.set_layer_weight("emotion", "browInnerUp", 0.8)
        result = face_rig.evaluate()
        assert abs(result["browInnerUp"] - 0.8) < 0.001

    def test_lower_priority_ignored_when_higher_exists(
        self, face_rig: FaceRig
    ) -> None:
        """Lower priority values should be completely replaced by higher priority."""
        face_rig.set_layer_weight("idle", "jawOpen", 1.0)
        face_rig.set_layer_weight("emotion", "jawOpen", 0.2)
        result = face_rig.evaluate()
        # emotion layer (higher priority) replaces idle layer
        assert abs(result["jawOpen"] - 0.2) < 0.001

    def test_no_blending_for_override_layers(self, face_rig: FaceRig) -> None:
        """Override layers should not blend; they should replace entirely."""
        face_rig.set_layer_weight("idle", "eyeBlinkLeft", 0.9)
        face_rig.set_layer_weight("emotion", "eyeBlinkLeft", 0.1)
        result = face_rig.evaluate()
        # Not 0.5 (average), but 0.1 (higher priority wins)
        assert abs(result["eyeBlinkLeft"] - 0.1) < 0.001


# =============================================================================
# Additive Layer Tests
# =============================================================================


class TestAdditiveLayers:
    """Tests for additive layer blending."""

    def test_additive_layers(self, face_rig: FaceRig) -> None:
        """Additive layers should accumulate correctly.

        Test contract from acceptance criteria.
        """
        face_rig.add_layer(
            "procedural", priority=LayerPriority.PROCEDURAL, additive=True
        )
        face_rig.set_layer_weight("emotion", "browDownL", 0.3)
        face_rig.set_layer_weight("procedural", "browDownL", 0.2)
        result = face_rig.evaluate()
        assert abs(result["browDownL"] - 0.5) < 0.001  # 0.3 + 0.2

    def test_multiple_additive_layers(self, face_rig: FaceRig) -> None:
        """Multiple additive layers should all accumulate."""
        face_rig.add_layer("additive1", priority=15, additive=True)
        face_rig.add_layer("additive2", priority=25, additive=True)

        face_rig.set_layer_weight("emotion", "noseSneerLeft", 0.2)
        face_rig.set_layer_weight("additive1", "noseSneerLeft", 0.1)
        face_rig.set_layer_weight("additive2", "noseSneerLeft", 0.15)

        result = face_rig.evaluate()
        # emotion (override) sets to 0.2, then additive1 adds 0.1, additive2 adds 0.15
        assert abs(result["noseSneerLeft"] - 0.45) < 0.001

    def test_additive_clamps_to_one(self, face_rig: FaceRig) -> None:
        """Additive results should be clamped to [0, 1]."""
        face_rig.add_layer("add1", priority=15, additive=True)
        face_rig.add_layer("add2", priority=25, additive=True)

        face_rig.set_layer_weight("emotion", "cheekPuff", 0.5)
        face_rig.set_layer_weight("add1", "cheekPuff", 0.4)
        face_rig.set_layer_weight("add2", "cheekPuff", 0.3)  # Total would be 1.2

        result = face_rig.evaluate()
        assert result["cheekPuff"] == 1.0  # Clamped to 1.0

    def test_additive_on_top_of_override(self, face_rig: FaceRig) -> None:
        """Additive layer should add to override layer value."""
        face_rig.add_layer("procedural", priority=LayerPriority.PROCEDURAL, additive=True)

        # idle sets to 0.2, emotion overrides to 0.4, procedural adds 0.3
        face_rig.set_layer_weight("idle", "mouthPucker", 0.2)
        face_rig.set_layer_weight("emotion", "mouthPucker", 0.4)
        face_rig.set_layer_weight("procedural", "mouthPucker", 0.3)

        result = face_rig.evaluate()
        # emotion overrides idle (0.4), then procedural adds (0.4 + 0.3 = 0.7)
        assert abs(result["mouthPucker"] - 0.7) < 0.001


# =============================================================================
# Layer Weight Tests
# =============================================================================


class TestLayerWeights:
    """Tests for layer master weight scaling."""

    def test_layer_master_weight_scales_contribution(
        self, face_rig: FaceRig
    ) -> None:
        """Layer master weight should scale all blend shape contributions."""
        face_rig.set_layer_weight("emotion", "jawOpen", 1.0)
        face_rig.set_rig_layer_master_weight("emotion", 0.5)
        result = face_rig.evaluate()
        assert abs(result["jawOpen"] - 0.5) < 0.001

    def test_zero_weight_layer_ignored(self, face_rig: FaceRig) -> None:
        """Layers with zero weight should not contribute."""
        face_rig.set_layer_weight("emotion", "eyeSquintLeft", 1.0)
        face_rig.set_rig_layer_master_weight("emotion", 0.0)
        result = face_rig.evaluate()
        assert result.get("eyeSquintLeft", 0.0) == 0.0

    def test_partial_layer_weight(self, face_rig: FaceRig) -> None:
        """Partial layer weight should scale proportionally."""
        face_rig.set_layer_weight("emotion", "browOuterUpLeft", 0.8)
        face_rig.set_rig_layer_master_weight("emotion", 0.25)
        result = face_rig.evaluate()
        # 0.8 * 0.25 = 0.2
        assert abs(result["browOuterUpLeft"] - 0.2) < 0.001

    def test_additive_with_scaled_weight(self, face_rig: FaceRig) -> None:
        """Additive layers should also respect master weight scaling."""
        face_rig.add_layer("add", priority=LayerPriority.PROCEDURAL, additive=True)

        face_rig.set_layer_weight("emotion", "tongueOut", 0.3)
        face_rig.set_layer_weight("add", "tongueOut", 0.4)
        face_rig.set_rig_layer_master_weight("add", 0.5)

        result = face_rig.evaluate()
        # emotion sets 0.3, add layer contributes 0.4 * 0.5 = 0.2
        # Total: 0.3 + 0.2 = 0.5
        assert abs(result["tongueOut"] - 0.5) < 0.001


# =============================================================================
# Layer Management Tests
# =============================================================================


class TestLayerManagement:
    """Tests for layer add/remove/clear operations."""

    def test_add_layer_with_numeric_priority(self, face_rig: FaceRig) -> None:
        """Adding a layer with numeric priority should create RigLayer."""
        layer = face_rig.add_layer("custom", priority=15, additive=True)
        assert isinstance(layer, RigLayer)
        assert layer.name == "custom"
        assert layer.priority == 15
        assert layer.additive is True

    def test_get_rig_layer(self, face_rig: FaceRig) -> None:
        """Should be able to retrieve a rig layer by name."""
        face_rig.add_layer("test_layer", priority=5)
        layer = face_rig.get_rig_layer("test_layer")
        assert layer is not None
        assert layer.name == "test_layer"

    def test_get_nonexistent_layer(self, face_rig: FaceRig) -> None:
        """Getting nonexistent layer should return None."""
        layer = face_rig.get_rig_layer("nonexistent")
        assert layer is None

    def test_clear_rig_layer(self, face_rig: FaceRig) -> None:
        """Clearing a layer should remove all blend shapes."""
        face_rig.set_layer_weight("emotion", "mouthLeft", 0.5)
        face_rig.set_layer_weight("emotion", "mouthRight", 0.5)

        result = face_rig.clear_rig_layer("emotion")

        assert result is True
        layer = face_rig.get_rig_layer("emotion")
        assert layer is not None
        assert len(layer.blend_shapes) == 0

    def test_remove_rig_layer(self, face_rig: FaceRig) -> None:
        """Removing a layer should delete it entirely."""
        face_rig.add_layer("temp", priority=99)
        face_rig.set_layer_weight("temp", "jawForward", 0.5)

        result = face_rig.remove_rig_layer("temp")

        assert result is True
        assert face_rig.get_rig_layer("temp") is None


# =============================================================================
# Subsystem Integration Tests
# =============================================================================


class TestSubsystemIntegration:
    """Tests for integration with other facial subsystems."""

    def test_default_layers_exist(self, face_rig: FaceRig) -> None:
        """Default layers should be created on init."""
        assert face_rig.get_rig_layer("idle") is not None
        assert face_rig.get_rig_layer("emotion") is not None
        assert face_rig.get_rig_layer("lip_sync") is not None
        assert face_rig.get_rig_layer("procedural") is not None

    def test_default_layer_priorities(self, face_rig: FaceRig) -> None:
        """Default layers should have correct priorities."""
        assert face_rig.get_rig_layer("idle").priority == LayerPriority.BASE
        assert face_rig.get_rig_layer("emotion").priority == LayerPriority.EMOTION
        assert face_rig.get_rig_layer("lip_sync").priority == LayerPriority.LIP_SYNC
        assert face_rig.get_rig_layer("procedural").priority == LayerPriority.PROCEDURAL

    def test_lip_sync_layer_is_additive(self, face_rig: FaceRig) -> None:
        """Lip sync layer should be additive by default."""
        layer = face_rig.get_rig_layer("lip_sync")
        assert layer.additive is True

    def test_procedural_layer_is_additive(self, face_rig: FaceRig) -> None:
        """Procedural layer should be additive by default."""
        layer = face_rig.get_rig_layer("procedural")
        assert layer.additive is True

    def test_idle_layer_is_override(self, face_rig: FaceRig) -> None:
        """Idle layer should be override (not additive)."""
        layer = face_rig.get_rig_layer("idle")
        assert layer.additive is False

    def test_emotion_layer_is_override(self, face_rig: FaceRig) -> None:
        """Emotion layer should be override (not additive)."""
        layer = face_rig.get_rig_layer("emotion")
        assert layer.additive is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_layers_return_empty_dict(self, face_rig: FaceRig) -> None:
        """Evaluating with no blend shapes should return empty dict."""
        result = face_rig.evaluate()
        assert result == {}

    def test_weight_clamping_to_zero(self, face_rig: FaceRig) -> None:
        """Negative weights should be clamped to 0."""
        face_rig._rig_layers["emotion"].blend_shapes["test"] = -0.5
        result = face_rig.evaluate()
        assert result.get("test", 0.0) >= 0.0

    def test_set_layer_weight_clamps_value(self, face_rig: FaceRig) -> None:
        """set_layer_weight should clamp values to [0, 1]."""
        face_rig.set_layer_weight("emotion", "test", 1.5)
        layer = face_rig.get_rig_layer("emotion")
        assert layer.blend_shapes["test"] == 1.0

        face_rig.set_layer_weight("emotion", "test2", -0.3)
        assert layer.blend_shapes["test2"] == 0.0

    def test_custom_priority_between_defaults(self, face_rig: FaceRig) -> None:
        """Custom layers can use priorities between default levels."""
        face_rig.add_layer("custom_emotion", priority=5)  # Between BASE and EMOTION
        face_rig.set_layer_weight("idle", "testShape", 0.2)
        face_rig.set_layer_weight("custom_emotion", "testShape", 0.6)
        face_rig.set_layer_weight("emotion", "testShape", 0.9)

        result = face_rig.evaluate()
        # emotion (10) > custom_emotion (5) > idle (0)
        # emotion wins
        assert abs(result["testShape"] - 0.9) < 0.001

    def test_same_priority_last_added_wins(self, face_rig: FaceRig) -> None:
        """For layers with same priority, iteration order determines winner."""
        face_rig.add_layer("same_priority_1", priority=5)
        face_rig.add_layer("same_priority_2", priority=5)

        face_rig.set_layer_weight("same_priority_1", "testShape", 0.3)
        face_rig.set_layer_weight("same_priority_2", "testShape", 0.7)

        result = face_rig.evaluate()
        # Both have same priority, result depends on dict iteration order
        # The value should be one of them (0.3 or 0.7), not blended
        assert result["testShape"] in [0.3, 0.7]
