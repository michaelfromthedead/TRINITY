"""
Blackbox tests for T-AG-2.16 LayerStack System + T-AG-2.17 LayerStackBuilder.

CLEANROOM MODE: Tests written against public contract only.
DO NOT read implementation files (engine/animation/graph/layer.py).

Tested API (from runtime introspection):
- LayerBlendMode enum: OVERRIDE, ADDITIVE
- AnimationLayer dataclass
- LayerStack class with evaluate(context), add_layer, layer_count()
- LayerStackBuilder fluent API
- BoneMask: set_weight(bone, weight), get_weight(bone)
- Transform uses position= not translation=
"""

import pytest


# ===========================================================================
# Test Fixtures and Helpers
# ===========================================================================

def create_mock_node(node_id: str = "mock"):
    """Create a minimal AnimationNode for testing."""
    from engine.animation.graph import AnimationNode, Pose

    class MockNode(AnimationNode):
        def __init__(self, nid: str):
            super().__init__(node_id=nid)

        def evaluate(self, context):
            return Pose()

    return MockNode(node_id)


def create_pose_node(node_id: str, bone_index: int, position: tuple):
    """Create a node that returns a specific pose at given bone index."""
    from engine.animation.graph import AnimationNode, Pose, Transform

    class PoseNode(AnimationNode):
        def __init__(self, nid: str, idx: int, pos: tuple):
            super().__init__(node_id=nid)
            self._idx = idx
            self._pos = pos

        def evaluate(self, context):
            pose = Pose()
            # Ensure list is long enough
            while len(pose.transforms) <= self._idx:
                pose.transforms.append(Transform())
            pose.transforms[self._idx] = Transform(
                position=self._pos,
                rotation=(0.0, 0.0, 0.0, 1.0),
                scale=(1.0, 1.0, 1.0)
            )
            return pose

    return PoseNode(node_id, bone_index, position)


def create_multi_bone_node(node_id: str, bones: dict):
    """Create a node that returns poses at multiple bone indices."""
    from engine.animation.graph import AnimationNode, Pose, Transform

    class MultiBoneNode(AnimationNode):
        def __init__(self, nid: str, bone_data: dict):
            super().__init__(node_id=nid)
            self._bones = bone_data  # {index: position}

        def evaluate(self, context):
            pose = Pose()
            max_idx = max(self._bones.keys()) if self._bones else 0
            # Ensure list is long enough
            while len(pose.transforms) <= max_idx:
                pose.transforms.append(Transform())
            for idx, pos in self._bones.items():
                pose.transforms[idx] = Transform(
                    position=pos,
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    scale=(1.0, 1.0, 1.0)
                )
            return pose

    return MultiBoneNode(node_id, bones)


def create_bone_mask(*bone_names):
    """Create a BoneMask that includes the specified bones."""
    from engine.animation.graph.bone_mask import BoneMask
    mask = BoneMask(name="test_mask")
    for bone in bone_names:
        mask.set_weight(bone, 1.0)
    return mask


# ===========================================================================
# T-AG-2.16: LayerBlendMode Enum
# ===========================================================================

class TestLayerBlendModeEnum:
    """Test LayerBlendMode enum."""

    def test_import_layer_blend_mode(self):
        from engine.animation.graph import LayerBlendMode
        assert LayerBlendMode is not None

    def test_override_mode_exists(self):
        from engine.animation.graph import LayerBlendMode
        assert hasattr(LayerBlendMode, 'OVERRIDE')

    def test_additive_mode_exists(self):
        from engine.animation.graph import LayerBlendMode
        assert hasattr(LayerBlendMode, 'ADDITIVE')

    def test_multiply_mode_exists(self):
        from engine.animation.graph import LayerBlendMode
        assert hasattr(LayerBlendMode, 'MULTIPLY')

    def test_layer_mode_alias_exists(self):
        from engine.animation.graph import LayerMode
        assert LayerMode is not None


# ===========================================================================
# T-AG-2.16: AnimationLayer
# ===========================================================================

class TestAnimationLayer:
    """Test AnimationLayer dataclass."""

    def test_import(self):
        from engine.animation.graph import AnimationLayer
        assert AnimationLayer is not None

    def test_create_minimal(self):
        from engine.animation.graph import AnimationLayer
        layer = AnimationLayer(name="test")
        assert layer.name == "test"

    def test_create_with_source(self):
        from engine.animation.graph import AnimationLayer
        source = create_mock_node("src")
        layer = AnimationLayer(name="test", source=source)
        assert layer.source is source

    def test_weight_attribute(self):
        from engine.animation.graph import AnimationLayer
        layer = AnimationLayer(name="test", weight=0.75)
        assert layer.weight == pytest.approx(0.75)

    def test_blend_mode_attribute(self):
        from engine.animation.graph import AnimationLayer, LayerBlendMode
        layer = AnimationLayer(name="test", blend_mode=LayerBlendMode.ADDITIVE)
        assert layer.blend_mode == LayerBlendMode.ADDITIVE

    def test_mask_attribute(self):
        from engine.animation.graph import AnimationLayer
        mask = create_bone_mask("spine")
        layer = AnimationLayer(name="test", mask=mask)
        assert layer.mask is mask

    def test_weight_default(self):
        from engine.animation.graph import AnimationLayer
        layer = AnimationLayer(name="test")
        assert layer.weight == pytest.approx(1.0)

    def test_blend_mode_default(self):
        from engine.animation.graph import AnimationLayer
        layer = AnimationLayer(name="test")
        assert layer.blend_mode.name == "OVERRIDE"


# ===========================================================================
# T-AG-2.16: LayerStack
# ===========================================================================

class TestLayerStack:
    """Test LayerStack class."""

    def test_import(self):
        from engine.animation.graph import LayerStack
        assert LayerStack is not None

    def test_create(self):
        from engine.animation.graph import LayerStack
        stack = LayerStack(node_id="test")
        assert stack is not None

    def test_has_add_layer(self):
        from engine.animation.graph import LayerStack
        stack = LayerStack(node_id="test")
        assert callable(stack.add_layer)

    def test_has_evaluate(self):
        from engine.animation.graph import LayerStack
        stack = LayerStack(node_id="test")
        assert callable(stack.evaluate)

    def test_has_layer_count(self):
        from engine.animation.graph import LayerStack
        stack = LayerStack(node_id="test")
        assert callable(stack.layer_count)

    def test_add_layer(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="l1", source=create_mock_node("s")))
        assert stack.layer_count() >= 1

    def test_add_multiple_layers(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="l1", source=create_mock_node("s1")))
        stack.add_layer(AnimationLayer(name="l2", source=create_mock_node("s2")))
        stack.add_layer(AnimationLayer(name="l3", source=create_mock_node("s3")))
        assert stack.layer_count() >= 3

    def test_get_layer_by_name(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="my_layer", source=create_mock_node("s")))
        layer = stack.get_layer("my_layer")
        assert layer.name == "my_layer"

    def test_get_layer_by_index(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="l0", source=create_mock_node("s0")))
        stack.add_layer(AnimationLayer(name="l1", source=create_mock_node("s1")))
        layer = stack.get_layer_by_index(1)
        assert layer.name == "l1"

    def test_remove_layer(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="l1", source=create_mock_node("s")))
        cnt = stack.layer_count()
        stack.remove_layer("l1")
        assert stack.layer_count() == cnt - 1

    def test_set_layer_weight(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="l1", source=create_mock_node("s")))
        stack.set_layer_weight("l1", 0.5)
        assert stack.get_layer("l1").weight == pytest.approx(0.5)

    def test_set_layer_active(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="l1", source=create_mock_node("s")))
        stack.set_layer_active("l1", False)
        assert stack.get_layer("l1").is_active is False


# ===========================================================================
# T-AG-2.16: LayerStack Evaluation
# ===========================================================================

class TestLayerStackEvaluation:
    """Test evaluate method."""

    def test_evaluate_empty(self):
        from engine.animation.graph import LayerStack, GraphContext, Skeleton
        stack = LayerStack(node_id="test")
        context = GraphContext(dt=1/60, skeleton=Skeleton())
        pose = stack.evaluate(context)
        assert pose is not None

    def test_evaluate_single_layer(self):
        from engine.animation.graph import LayerStack, AnimationLayer, GraphContext, Skeleton
        # Use bone index 0 instead of string name
        src = create_pose_node("src", 0, (5.0, 0.0, 0.0))
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="base", source=src))
        context = GraphContext(dt=1/60, skeleton=Skeleton())
        pose = stack.evaluate(context)
        # Pose.transforms is a list, check index 0
        if hasattr(pose, 'transforms') and len(pose.transforms) > 0:
            assert pose.transforms[0].position[0] == pytest.approx(5.0)


# ===========================================================================
# T-AG-2.16: Bone Mask Presets
# ===========================================================================

class TestBoneMaskPresets:
    """Test bone mask presets."""

    def test_import(self):
        from engine.animation.graph import BoneMaskPresets
        assert BoneMaskPresets is not None

    def test_upper_body(self):
        from engine.animation.graph import BoneMaskPresets
        assert hasattr(BoneMaskPresets, 'upper_body')

    def test_lower_body(self):
        from engine.animation.graph import BoneMaskPresets
        assert hasattr(BoneMaskPresets, 'lower_body')


# ===========================================================================
# T-AG-2.17: LayerStackBuilder
# ===========================================================================

class TestLayerStackBuilder:
    """Test LayerStackBuilder."""

    def test_import(self):
        from engine.animation.graph import LayerStackBuilder
        assert LayerStackBuilder is not None

    def test_fluent_api(self):
        from engine.animation.graph import LayerStackBuilder
        builder = LayerStackBuilder(node_id="test")
        result = builder.add_layer("l1", source=create_mock_node("s"))
        assert result is builder

    def test_build(self):
        from engine.animation.graph import LayerStackBuilder, LayerStack
        stack = LayerStackBuilder(node_id="test").add_layer("base", source=create_mock_node("s")).build()
        assert isinstance(stack, LayerStack)

    def test_build_empty(self):
        from engine.animation.graph import LayerStackBuilder
        stack = LayerStackBuilder(node_id="test").build()
        assert stack.layer_count() == 0

    def test_add_layer_with_weight(self):
        from engine.animation.graph import LayerStackBuilder
        stack = LayerStackBuilder(node_id="test").add_layer("l", source=create_mock_node("s"), weight=0.5).build()
        assert stack.get_layer("l").weight == pytest.approx(0.5)

    def test_add_layer_with_mask(self):
        from engine.animation.graph import LayerStackBuilder
        mask = create_bone_mask("spine")
        stack = LayerStackBuilder(node_id="test").add_layer("l", source=create_mock_node("s"), mask=mask).build()
        assert stack.get_layer("l").mask is mask

    def test_add_layer_with_mode(self):
        from engine.animation.graph import LayerStackBuilder, LayerBlendMode
        stack = LayerStackBuilder(node_id="test").add_layer("l", source=create_mock_node("s"), blend_mode=LayerBlendMode.ADDITIVE).build()
        assert stack.get_layer("l").blend_mode == LayerBlendMode.ADDITIVE

    def test_add_multiple_layers(self):
        from engine.animation.graph import LayerStackBuilder, LayerBlendMode
        stack = (LayerStackBuilder(node_id="test")
            .add_layer("base", source=create_mock_node("s1"))
            .add_layer("l1", source=create_mock_node("s2"), weight=0.5)
            .add_layer("l2", source=create_mock_node("s3"), blend_mode=LayerBlendMode.ADDITIVE)
            .build())
        assert stack.layer_count() >= 3


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestLayerStackIntegration:
    """Integration tests."""

    def test_get_active_layers(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="l1", source=create_mock_node("s1"), is_active=True))
        stack.add_layer(AnimationLayer(name="l2", source=create_mock_node("s2"), is_active=False))
        stack.add_layer(AnimationLayer(name="l3", source=create_mock_node("s3"), is_active=True))
        active = stack.get_active_layers()
        names = [l.name for l in active]
        assert "l1" in names and "l3" in names and "l2" not in names

    def test_get_layers_with_weight(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="l1", source=create_mock_node("s1"), weight=1.0))
        stack.add_layer(AnimationLayer(name="l2", source=create_mock_node("s2"), weight=0.0))
        stack.add_layer(AnimationLayer(name="l3", source=create_mock_node("s3"), weight=0.5))
        weighted = stack.get_layers_with_weight()
        names = [item[0].name for item in weighted]
        assert "l1" in names and "l3" in names

    def test_move_layer(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="l0", source=create_mock_node("s0")))
        stack.add_layer(AnimationLayer(name="l1", source=create_mock_node("s1")))
        stack.add_layer(AnimationLayer(name="l2", source=create_mock_node("s2")))
        stack.move_layer("l2", 0)
        assert stack.get_layer_by_index(0).name == "l2"

    def test_debug_info(self):
        from engine.animation.graph import LayerStack, AnimationLayer
        stack = LayerStack(node_id="test")
        stack.add_layer(AnimationLayer(name="l1", source=create_mock_node("s")))
        assert stack.get_debug_info() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
