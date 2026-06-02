"""WHITEBOX tests for engine/animation/graph/layer.py.

Tests for T-AG-2.16 (LayerStack System) and T-AG-2.17 (LayerStackBuilder).

WHITEBOX coverage plan:
  [LayerBlendMode Enum]
    Path A1:  OVERRIDE mode exists with auto value
    Path A2:  ADDITIVE mode exists with auto value
    Path A3:  OVERRIDE_ADDITIVE mode exists (MULTIPLY alternative)

  [AnimationLayer dataclass]
    Path B1:  __init__ default values (weight=1.0, mask=None, blend_mode=OVERRIDE)
    Path B2:  is_active=True by default
    Path B3:  weight_parameter binding (None by default)
    Path B4:  sync_group and is_synced fields

  [AnimationLayer.get_effective_weight]
    Path C1:  inactive layer returns 0.0
    Path C2:  active layer with no parameter returns clamped weight
    Path C3:  weight parameter binding retrieves from context
    Path C4:  weight clamping below 0
    Path C5:  weight clamping above 1

  [AnimationLayer.evaluate]
    Path D1:  inactive layer returns None
    Path D2:  no source returns None
    Path D3:  active layer with source returns evaluated pose
    Path D4:  layer context carries sync_group

  [AnimationLayer.apply_to_pose]
    Path E1:  zero effective weight returns base pose unchanged
    Path E2:  layer with mask calls _apply_masked
    Path E3:  layer without mask calls _apply_full

  [AnimationLayer._apply_masked]
    Path F1:  OVERRIDE mode with mask applies lerp per bone
    Path F2:  ADDITIVE mode with mask applies additive blend
    Path F3:  OVERRIDE_ADDITIVE mode blends override+additive
    Path F4:  bone index bounds checking (min of bone_count)
    Path F5:  zero mask weight skips bone

  [AnimationLayer._apply_full]
    Path G1:  OVERRIDE mode applies lerp
    Path G2:  ADDITIVE mode applies additive_blend
    Path G3:  OVERRIDE_ADDITIVE mode blends both

  [LayerStack class]
    Path H1:  __init__ creates empty layers list and dict
    Path H2:  extends AnimationNode (inheritance)
    Path H3:  _abstract = False (registered in metaclass)

  [LayerStack.add_layer]
    Path I1:  add_layer appends to end when index=None
    Path I2:  add_layer inserts at specific index
    Path I3:  add_layer raises on duplicate name
    Path I4:  add_layer returns actual index
    Path I5:  index clamping (negative, beyond length)

  [LayerStack.remove_layer]
    Path J1:  remove by name succeeds
    Path J2:  remove by index succeeds
    Path J3:  remove non-existent name returns False
    Path J4:  remove invalid index returns False
    Path J5:  removal updates internal dict

  [LayerStack.get_layer / get_layer_by_index]
    Path K1:  get_layer by name returns layer
    Path K2:  get_layer non-existent returns None
    Path K3:  get_layer_by_index valid returns layer
    Path K4:  get_layer_by_index out of bounds returns None

  [LayerStack.move_layer]
    Path L1:  move_layer repositions correctly
    Path L2:  move_layer non-existent returns False
    Path L3:  move_layer index clamping

  [LayerStack.set_layer_weight / set_layer_active]
    Path M1:  set_layer_weight updates existing layer
    Path M2:  set_layer_weight clamps to [0,1]
    Path M3:  set_layer_weight non-existent returns False
    Path M4:  set_layer_active enables/disables layer
    Path M5:  set_layer_active non-existent returns False

  [LayerStack.evaluate]
    Path N1:  empty stack returns empty Pose
    Path N2:  first active layer becomes base
    Path N3:  subsequent layers apply via apply_to_pose
    Path N4:  inactive layers skipped
    Path N5:  zero weight layers skipped
    Path N6:  all inactive returns empty Pose

  [LayerStack.get_active_layers / get_layers_with_weight]
    Path O1:  get_active_layers filters by is_active
    Path O2:  get_layers_with_weight returns tuples

  [LayerStackBuilder class]
    Path P1:  __init__ stores node_id and empty layers
    Path P2:  add_layer appends AnimationLayer
    Path P3:  add_layer returns self (fluent)
    Path P4:  add_override_layer sets OVERRIDE mode
    Path P5:  add_additive_layer sets ADDITIVE mode

  [LayerStackBuilder.build]
    Path Q1:  build creates LayerStack with node_id
    Path Q2:  build adds all layers to stack
    Path Q3:  fluent chaining works

  [BoneMaskPresets factory methods]
    Path R1:  upper_body creates mask with upper body bones
    Path R2:  lower_body creates mask with lower body bones
    Path R3:  left_arm / right_arm / arms create arm masks
    Path R4:  left_leg / right_leg / legs create leg masks
    Path R5:  head / spine create targeted masks
    Path R6:  full_body creates mask with all bones
    Path R7:  gradient_upper_lower creates gradient mask

  [Edge cases]
    Path S1:  layer with None source in stack evaluation
    Path S2:  empty layer list in builder.build()
    Path S3:  duplicate layer name in builder (allowed, stack raises)
    Path S4:  pose bone_count mismatch in masked apply
    Path S5:  BoneMaskPresets with skeleton missing bones
"""

from __future__ import annotations

import math
import pytest
from typing import Optional, Dict, List

from engine.animation.graph.layer import (
    LayerBlendMode,
    AnimationLayer,
    LayerStack,
    LayerStackBuilder,
    BoneMaskPresets,
)
from engine.animation.graph.animation_graph import (
    AnimationNode,
    BoneMask,
    GraphContext,
    GraphParameter,
    ParameterType,
    Pose,
    Skeleton,
    Transform,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def make_skeleton(bone_names: List[str]) -> Skeleton:
    """Create a simple test skeleton with named bones."""
    skeleton = Skeleton()
    for name in bone_names:
        skeleton.add_bone(name, parent_index=-1)
    return skeleton


def make_pose(transforms: List[Transform]) -> Pose:
    """Create a pose from a list of transforms."""
    return Pose(transforms=transforms)


def make_identity_pose(bone_count: int) -> Pose:
    """Create an identity pose with N bones."""
    return Pose(transforms=[Transform.identity() for _ in range(bone_count)])


def make_test_transform(x: float = 1.0, y: float = 2.0, z: float = 3.0) -> Transform:
    """Create a test transform with distinct position."""
    return Transform(position=(x, y, z))


def make_context(
    skeleton: Optional[Skeleton] = None,
    parameters: Optional[Dict[str, GraphParameter]] = None,
) -> GraphContext:
    """Create a test GraphContext."""
    return GraphContext(
        skeleton=skeleton,
        parameters=parameters or {},
        dt=0.016,
    )


class MockAnimationNode(AnimationNode):
    """Mock AnimationNode for testing layer evaluation."""

    _abstract = True  # Don't register in metaclass

    def __init__(self, node_id: str, pose: Optional[Pose] = None) -> None:
        super().__init__(node_id)
        self._pose = pose

    def evaluate(self, context: GraphContext) -> Pose:
        return self._pose if self._pose else Pose()


def transform_approx_equal(t1: Transform, t2: Transform, tol: float = 1e-6) -> bool:
    """Check if two transforms are approximately equal."""
    for i in range(3):
        if abs(t1.position[i] - t2.position[i]) > tol:
            return False
        if abs(t1.scale[i] - t2.scale[i]) > tol:
            return False
    for i in range(4):
        if abs(t1.rotation[i] - t2.rotation[i]) > tol:
            return False
    return True


# =============================================================================
# Path A: LayerBlendMode Enum
# =============================================================================


class TestLayerBlendModeEnum:
    """Tests for LayerBlendMode enum (Path A)."""

    def test_override_mode_exists(self) -> None:
        """Path A1: OVERRIDE mode exists with auto value."""
        assert hasattr(LayerBlendMode, "OVERRIDE")
        assert isinstance(LayerBlendMode.OVERRIDE, LayerBlendMode)

    def test_additive_mode_exists(self) -> None:
        """Path A2: ADDITIVE mode exists with auto value."""
        assert hasattr(LayerBlendMode, "ADDITIVE")
        assert isinstance(LayerBlendMode.ADDITIVE, LayerBlendMode)

    def test_override_additive_mode_exists(self) -> None:
        """Path A3: OVERRIDE_ADDITIVE mode exists (alternative to MULTIPLY)."""
        assert hasattr(LayerBlendMode, "OVERRIDE_ADDITIVE")
        assert isinstance(LayerBlendMode.OVERRIDE_ADDITIVE, LayerBlendMode)

    def test_all_modes_unique(self) -> None:
        """All blend modes have unique values."""
        modes = [LayerBlendMode.OVERRIDE, LayerBlendMode.ADDITIVE, LayerBlendMode.OVERRIDE_ADDITIVE]
        values = [m.value for m in modes]
        assert len(values) == len(set(values))


# =============================================================================
# Path B: AnimationLayer dataclass initialization
# =============================================================================


class TestAnimationLayerInit:
    """Tests for AnimationLayer initialization (Path B)."""

    def test_default_values(self) -> None:
        """Path B1: Default values - weight=1.0, mask=None, blend_mode=OVERRIDE."""
        layer = AnimationLayer(name="test")
        assert layer.name == "test"
        assert layer.weight == 1.0
        assert layer.mask is None
        assert layer.blend_mode == LayerBlendMode.OVERRIDE
        assert layer.source is None

    def test_is_active_default_true(self) -> None:
        """Path B2: is_active=True by default."""
        layer = AnimationLayer(name="test")
        assert layer.is_active is True

    def test_weight_parameter_default_none(self) -> None:
        """Path B3: weight_parameter binding is None by default."""
        layer = AnimationLayer(name="test")
        assert layer.weight_parameter is None

    def test_sync_fields(self) -> None:
        """Path B4: sync_group and is_synced fields exist."""
        layer = AnimationLayer(name="test")
        assert layer.is_synced is False
        assert layer.sync_group is None

    def test_custom_initialization(self) -> None:
        """Custom values are properly set."""
        mask = BoneMask(name="test_mask")
        layer = AnimationLayer(
            name="custom",
            weight=0.5,
            mask=mask,
            blend_mode=LayerBlendMode.ADDITIVE,
            is_active=False,
            sync_group="group1",
        )
        assert layer.name == "custom"
        assert layer.weight == 0.5
        assert layer.mask is mask
        assert layer.blend_mode == LayerBlendMode.ADDITIVE
        assert layer.is_active is False
        assert layer.sync_group == "group1"


# =============================================================================
# Path C: AnimationLayer.get_effective_weight
# =============================================================================


class TestAnimationLayerGetEffectiveWeight:
    """Tests for AnimationLayer.get_effective_weight (Path C)."""

    def test_inactive_layer_returns_zero(self) -> None:
        """Path C1: Inactive layer returns 0.0."""
        layer = AnimationLayer(name="test", weight=1.0, is_active=False)
        context = make_context()
        assert layer.get_effective_weight(context) == 0.0

    def test_active_layer_returns_clamped_weight(self) -> None:
        """Path C2: Active layer with no parameter returns clamped weight."""
        layer = AnimationLayer(name="test", weight=0.75)
        context = make_context()
        assert layer.get_effective_weight(context) == 0.75

    def test_weight_parameter_binding(self) -> None:
        """Path C3: weight parameter binding retrieves from context."""
        param = GraphParameter.float_param("layer_weight", default=0.3)
        layer = AnimationLayer(name="test", weight=1.0, weight_parameter="layer_weight")
        context = make_context(parameters={"layer_weight": param})
        assert layer.get_effective_weight(context) == 0.3

    def test_weight_clamping_below_zero(self) -> None:
        """Path C4: Weight clamping below 0."""
        layer = AnimationLayer(name="test", weight=-0.5)
        context = make_context()
        assert layer.get_effective_weight(context) == 0.0

    def test_weight_clamping_above_one(self) -> None:
        """Path C5: Weight clamping above 1."""
        layer = AnimationLayer(name="test", weight=1.5)
        context = make_context()
        assert layer.get_effective_weight(context) == 1.0

    def test_weight_boundary_values(self) -> None:
        """Weight boundary values 0.0 and 1.0 pass through."""
        layer_zero = AnimationLayer(name="test", weight=0.0)
        layer_one = AnimationLayer(name="test", weight=1.0)
        context = make_context()
        assert layer_zero.get_effective_weight(context) == 0.0
        assert layer_one.get_effective_weight(context) == 1.0


# =============================================================================
# Path D: AnimationLayer.evaluate
# =============================================================================


class TestAnimationLayerEvaluate:
    """Tests for AnimationLayer.evaluate (Path D)."""

    def test_inactive_layer_returns_none(self) -> None:
        """Path D1: Inactive layer returns None."""
        pose = make_identity_pose(3)
        source = MockAnimationNode("source", pose)
        layer = AnimationLayer(name="test", source=source, is_active=False)
        context = make_context()
        assert layer.evaluate(context) is None

    def test_no_source_returns_none(self) -> None:
        """Path D2: No source returns None."""
        layer = AnimationLayer(name="test", source=None)
        context = make_context()
        assert layer.evaluate(context) is None

    def test_active_layer_with_source_returns_pose(self) -> None:
        """Path D3: Active layer with source returns evaluated pose."""
        pose = make_identity_pose(3)
        source = MockAnimationNode("source", pose)
        layer = AnimationLayer(name="test", source=source)
        context = make_context()
        result = layer.evaluate(context)
        assert result is not None
        assert result.bone_count() == 3

    def test_layer_context_carries_sync_group(self) -> None:
        """Path D4: Layer context carries sync_group."""
        captured_context = []

        class CapturingNode(AnimationNode):
            _abstract = True

            def evaluate(self, context: GraphContext) -> Pose:
                captured_context.append(context)
                return Pose()

        source = CapturingNode("capture")
        layer = AnimationLayer(name="test", source=source, sync_group="sync_test")
        context = make_context()
        layer.evaluate(context)

        assert len(captured_context) == 1
        assert captured_context[0].sync_group == "sync_test"


# =============================================================================
# Path E: AnimationLayer.apply_to_pose
# =============================================================================


class TestAnimationLayerApplyToPose:
    """Tests for AnimationLayer.apply_to_pose (Path E)."""

    def test_zero_weight_returns_base_unchanged(self) -> None:
        """Path E1: Zero effective weight returns base pose unchanged."""
        base_pose = make_pose([make_test_transform(1, 2, 3)])
        source = MockAnimationNode("source", make_pose([make_test_transform(10, 20, 30)]))
        layer = AnimationLayer(name="test", source=source, weight=0.0)
        context = make_context()

        result = layer.apply_to_pose(base_pose, context)
        assert transform_approx_equal(result.transforms[0], base_pose.transforms[0])

    def test_layer_with_mask_uses_apply_masked(self) -> None:
        """Path E2: Layer with mask calls _apply_masked (tested via behavior)."""
        skeleton = make_skeleton(["bone0", "bone1"])
        mask = BoneMask(name="test_mask")
        mask.set_weight(0, 1.0)  # Only bone 0
        mask.set_weight(1, 0.0)  # Exclude bone 1

        base_pose = make_pose([make_test_transform(0, 0, 0), make_test_transform(0, 0, 0)])
        layer_pose = make_pose([make_test_transform(10, 10, 10), make_test_transform(20, 20, 20)])
        source = MockAnimationNode("source", layer_pose)

        layer = AnimationLayer(name="test", source=source, mask=mask, weight=1.0)
        context = make_context(skeleton=skeleton)

        result = layer.apply_to_pose(base_pose, context)
        # Bone 0 should be affected, bone 1 should not
        assert result.transforms[0].position[0] == 10.0  # Full override
        assert result.transforms[1].position[0] == 0.0   # Unchanged

    def test_layer_without_mask_uses_apply_full(self) -> None:
        """Path E3: Layer without mask calls _apply_full (tested via behavior)."""
        base_pose = make_pose([make_test_transform(0, 0, 0)])
        layer_pose = make_pose([make_test_transform(10, 10, 10)])
        source = MockAnimationNode("source", layer_pose)

        layer = AnimationLayer(name="test", source=source, mask=None, weight=1.0)
        context = make_context()

        result = layer.apply_to_pose(base_pose, context)
        # Full override should happen
        assert result.transforms[0].position[0] == 10.0


# =============================================================================
# Path F: AnimationLayer._apply_masked
# =============================================================================


class TestAnimationLayerApplyMasked:
    """Tests for AnimationLayer._apply_masked (Path F)."""

    def test_override_mode_with_mask(self) -> None:
        """Path F1: OVERRIDE mode with mask applies lerp per bone."""
        mask = BoneMask(name="mask")
        mask.set_weight(0, 0.5)

        base_pose = make_pose([make_test_transform(0, 0, 0)])
        layer_pose = make_pose([make_test_transform(10, 10, 10)])

        layer = AnimationLayer(name="test", mask=mask, blend_mode=LayerBlendMode.OVERRIDE)
        result = layer._apply_masked(base_pose, layer_pose, 1.0)

        # 0.5 lerp between (0,0,0) and (10,10,10) = (5,5,5)
        assert abs(result.transforms[0].position[0] - 5.0) < 0.001

    def test_additive_mode_with_mask(self) -> None:
        """Path F2: ADDITIVE mode with mask applies additive blend."""
        mask = BoneMask(name="mask")
        mask.set_weight(0, 1.0)

        base_pose = make_pose([make_test_transform(5, 5, 5)])
        layer_pose = make_pose([make_test_transform(3, 3, 3)])

        layer = AnimationLayer(name="test", mask=mask, blend_mode=LayerBlendMode.ADDITIVE)
        result = layer._apply_masked(base_pose, layer_pose, 1.0)

        # Additive: 5 + 3 = 8
        assert abs(result.transforms[0].position[0] - 8.0) < 0.001

    def test_override_additive_mode_with_mask(self) -> None:
        """Path F3: OVERRIDE_ADDITIVE mode blends override+additive."""
        mask = BoneMask(name="mask")
        mask.set_weight(0, 0.5)

        base_pose = make_pose([make_test_transform(5, 5, 5)])
        layer_pose = make_pose([make_test_transform(3, 3, 3)])

        layer = AnimationLayer(name="test", mask=mask, blend_mode=LayerBlendMode.OVERRIDE_ADDITIVE)
        result = layer._apply_masked(base_pose, layer_pose, 1.0)

        # additive_result = base + layer = (8,8,8)
        # lerp(base, additive_result, 0.5) = lerp((5,5,5), (8,8,8), 0.5) = (6.5,6.5,6.5)
        assert abs(result.transforms[0].position[0] - 6.5) < 0.001

    def test_bone_count_bounds_checking(self) -> None:
        """Path F4: Bone index bounds checking (min of bone_count)."""
        mask = BoneMask(name="mask")
        mask.set_weight(0, 1.0)
        mask.set_weight(1, 1.0)
        mask.set_weight(2, 1.0)  # Index 2 doesn't exist in shorter pose

        base_pose = make_pose([make_test_transform(0, 0, 0), make_test_transform(0, 0, 0)])
        layer_pose = make_pose([make_test_transform(10, 10, 10)])  # Only 1 bone

        layer = AnimationLayer(name="test", mask=mask)
        # Should not raise even with mismatched counts
        result = layer._apply_masked(base_pose, layer_pose, 1.0)
        assert result is not None

    def test_zero_mask_weight_skips_bone(self) -> None:
        """Path F5: Zero mask weight skips bone."""
        mask = BoneMask(name="mask")
        mask.set_weight(0, 0.0)  # Zero weight

        base_pose = make_pose([make_test_transform(5, 5, 5)])
        layer_pose = make_pose([make_test_transform(100, 100, 100)])

        layer = AnimationLayer(name="test", mask=mask)
        result = layer._apply_masked(base_pose, layer_pose, 1.0)

        # Should be unchanged from base
        assert result.transforms[0].position[0] == 5.0


# =============================================================================
# Path G: AnimationLayer._apply_full
# =============================================================================


class TestAnimationLayerApplyFull:
    """Tests for AnimationLayer._apply_full (Path G)."""

    def test_override_mode_applies_lerp(self) -> None:
        """Path G1: OVERRIDE mode applies lerp."""
        base_pose = make_pose([make_test_transform(0, 0, 0)])
        layer_pose = make_pose([make_test_transform(10, 10, 10)])

        layer = AnimationLayer(name="test", blend_mode=LayerBlendMode.OVERRIDE)
        result = layer._apply_full(base_pose, layer_pose, 0.5)

        assert abs(result.transforms[0].position[0] - 5.0) < 0.001

    def test_additive_mode_applies_additive_blend(self) -> None:
        """Path G2: ADDITIVE mode applies additive_blend."""
        base_pose = make_pose([make_test_transform(5, 5, 5)])
        layer_pose = make_pose([make_test_transform(3, 3, 3)])

        layer = AnimationLayer(name="test", blend_mode=LayerBlendMode.ADDITIVE)
        result = layer._apply_full(base_pose, layer_pose, 1.0)

        # Additive: 5 + 3 = 8
        assert abs(result.transforms[0].position[0] - 8.0) < 0.001

    def test_override_additive_mode_blends_both(self) -> None:
        """Path G3: OVERRIDE_ADDITIVE mode blends both."""
        base_pose = make_pose([make_test_transform(5, 5, 5)])
        layer_pose = make_pose([make_test_transform(3, 3, 3)])

        layer = AnimationLayer(name="test", blend_mode=LayerBlendMode.OVERRIDE_ADDITIVE)
        result = layer._apply_full(base_pose, layer_pose, 0.5)

        # additive_result = base + layer at full weight = (8,8,8)
        # lerp(base, additive_result, 0.5) = lerp((5,5,5), (8,8,8), 0.5) = (6.5,6.5,6.5)
        assert abs(result.transforms[0].position[0] - 6.5) < 0.001


# =============================================================================
# Path H: LayerStack class
# =============================================================================


class TestLayerStackClass:
    """Tests for LayerStack class (Path H)."""

    def test_init_creates_empty_layers(self) -> None:
        """Path H1: __init__ creates empty layers list and dict."""
        stack = LayerStack("test_stack")
        assert stack.layers == []
        assert stack._layer_by_name == {}

    def test_extends_animation_node(self) -> None:
        """Path H2: LayerStack extends AnimationNode (inheritance)."""
        assert issubclass(LayerStack, AnimationNode)
        stack = LayerStack("test")
        assert isinstance(stack, AnimationNode)

    def test_abstract_false_registered(self) -> None:
        """Path H3: _abstract = False (registered in metaclass)."""
        assert LayerStack._abstract is False


# =============================================================================
# Path I: LayerStack.add_layer
# =============================================================================


class TestLayerStackAddLayer:
    """Tests for LayerStack.add_layer (Path I)."""

    def test_add_layer_appends_to_end(self) -> None:
        """Path I1: add_layer appends to end when index=None."""
        stack = LayerStack("test")
        layer1 = AnimationLayer(name="layer1")
        layer2 = AnimationLayer(name="layer2")

        idx1 = stack.add_layer(layer1)
        idx2 = stack.add_layer(layer2)

        assert idx1 == 0
        assert idx2 == 1
        assert stack.layers[0].name == "layer1"
        assert stack.layers[1].name == "layer2"

    def test_add_layer_inserts_at_index(self) -> None:
        """Path I2: add_layer inserts at specific index."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="layer0"))
        stack.add_layer(AnimationLayer(name="layer2"))

        idx = stack.add_layer(AnimationLayer(name="layer1"), index=1)

        assert idx == 1
        assert stack.layers[1].name == "layer1"
        assert stack.layers[2].name == "layer2"

    def test_add_layer_raises_on_duplicate(self) -> None:
        """Path I3: add_layer raises on duplicate name."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="duplicate"))

        with pytest.raises(ValueError, match="already exists"):
            stack.add_layer(AnimationLayer(name="duplicate"))

    def test_add_layer_returns_actual_index(self) -> None:
        """Path I4: add_layer returns actual index."""
        stack = LayerStack("test")
        idx = stack.add_layer(AnimationLayer(name="test"))
        assert idx == 0

    def test_index_clamping(self) -> None:
        """Path I5: Index clamping (negative, beyond length)."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="middle"))

        # Negative index clamps to 0
        idx_neg = stack.add_layer(AnimationLayer(name="first"), index=-10)
        assert idx_neg == 0
        assert stack.layers[0].name == "first"

        # Beyond length clamps to end
        idx_beyond = stack.add_layer(AnimationLayer(name="last"), index=100)
        assert idx_beyond == 2


# =============================================================================
# Path J: LayerStack.remove_layer
# =============================================================================


class TestLayerStackRemoveLayer:
    """Tests for LayerStack.remove_layer (Path J)."""

    def test_remove_by_name_succeeds(self) -> None:
        """Path J1: Remove by name succeeds."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="to_remove"))

        result = stack.remove_layer("to_remove")
        assert result is True
        assert len(stack.layers) == 0

    def test_remove_by_index_succeeds(self) -> None:
        """Path J2: Remove by index succeeds."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="layer"))

        result = stack.remove_layer(0)
        assert result is True
        assert len(stack.layers) == 0

    def test_remove_nonexistent_name_returns_false(self) -> None:
        """Path J3: Remove non-existent name returns False."""
        stack = LayerStack("test")
        result = stack.remove_layer("nonexistent")
        assert result is False

    def test_remove_invalid_index_returns_false(self) -> None:
        """Path J4: Remove invalid index returns False."""
        stack = LayerStack("test")
        assert stack.remove_layer(-1) is False
        assert stack.remove_layer(100) is False

    def test_removal_updates_internal_dict(self) -> None:
        """Path J5: Removal updates internal dict."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="layer"))

        stack.remove_layer("layer")
        assert "layer" not in stack._layer_by_name


# =============================================================================
# Path K: LayerStack.get_layer / get_layer_by_index
# =============================================================================


class TestLayerStackGetLayer:
    """Tests for LayerStack.get_layer and get_layer_by_index (Path K)."""

    def test_get_layer_by_name_returns_layer(self) -> None:
        """Path K1: get_layer by name returns layer."""
        stack = LayerStack("test")
        layer = AnimationLayer(name="target")
        stack.add_layer(layer)

        result = stack.get_layer("target")
        assert result is layer

    def test_get_layer_nonexistent_returns_none(self) -> None:
        """Path K2: get_layer non-existent returns None."""
        stack = LayerStack("test")
        assert stack.get_layer("nonexistent") is None

    def test_get_layer_by_index_valid(self) -> None:
        """Path K3: get_layer_by_index valid returns layer."""
        stack = LayerStack("test")
        layer = AnimationLayer(name="layer")
        stack.add_layer(layer)

        result = stack.get_layer_by_index(0)
        assert result is layer

    def test_get_layer_by_index_out_of_bounds(self) -> None:
        """Path K4: get_layer_by_index out of bounds returns None."""
        stack = LayerStack("test")
        assert stack.get_layer_by_index(-1) is None
        assert stack.get_layer_by_index(0) is None
        assert stack.get_layer_by_index(100) is None


# =============================================================================
# Path L: LayerStack.move_layer
# =============================================================================


class TestLayerStackMoveLayer:
    """Tests for LayerStack.move_layer (Path L)."""

    def test_move_layer_repositions(self) -> None:
        """Path L1: move_layer repositions correctly."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="a"))
        stack.add_layer(AnimationLayer(name="b"))
        stack.add_layer(AnimationLayer(name="c"))

        result = stack.move_layer("c", 0)
        assert result is True
        assert stack.layers[0].name == "c"
        assert stack.layers[1].name == "a"
        assert stack.layers[2].name == "b"

    def test_move_layer_nonexistent_returns_false(self) -> None:
        """Path L2: move_layer non-existent returns False."""
        stack = LayerStack("test")
        assert stack.move_layer("nonexistent", 0) is False

    def test_move_layer_index_clamping(self) -> None:
        """Path L3: move_layer index clamping."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="layer"))

        # Both should succeed even with extreme indices
        assert stack.move_layer("layer", -100) is True
        assert stack.move_layer("layer", 100) is True


# =============================================================================
# Path M: LayerStack.set_layer_weight / set_layer_active
# =============================================================================


class TestLayerStackSetters:
    """Tests for LayerStack weight and active setters (Path M)."""

    def test_set_layer_weight_updates(self) -> None:
        """Path M1: set_layer_weight updates existing layer."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="layer", weight=1.0))

        result = stack.set_layer_weight("layer", 0.5)
        assert result is True
        assert stack.get_layer("layer").weight == 0.5

    def test_set_layer_weight_clamps(self) -> None:
        """Path M2: set_layer_weight clamps to [0,1]."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="layer"))

        stack.set_layer_weight("layer", -0.5)
        assert stack.get_layer("layer").weight == 0.0

        stack.set_layer_weight("layer", 1.5)
        assert stack.get_layer("layer").weight == 1.0

    def test_set_layer_weight_nonexistent(self) -> None:
        """Path M3: set_layer_weight non-existent returns False."""
        stack = LayerStack("test")
        assert stack.set_layer_weight("nonexistent", 0.5) is False

    def test_set_layer_active(self) -> None:
        """Path M4: set_layer_active enables/disables layer."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="layer", is_active=True))

        result = stack.set_layer_active("layer", False)
        assert result is True
        assert stack.get_layer("layer").is_active is False

        stack.set_layer_active("layer", True)
        assert stack.get_layer("layer").is_active is True

    def test_set_layer_active_nonexistent(self) -> None:
        """Path M5: set_layer_active non-existent returns False."""
        stack = LayerStack("test")
        assert stack.set_layer_active("nonexistent", True) is False


# =============================================================================
# Path N: LayerStack.evaluate
# =============================================================================


class TestLayerStackEvaluate:
    """Tests for LayerStack.evaluate (Path N)."""

    def test_empty_stack_returns_empty_pose(self) -> None:
        """Path N1: Empty stack returns empty Pose."""
        stack = LayerStack("test")
        context = make_context()
        result = stack.evaluate(context)
        assert result.bone_count() == 0

    def test_first_active_layer_becomes_base(self) -> None:
        """Path N2: First active layer becomes base."""
        pose = make_pose([make_test_transform(10, 10, 10)])
        source = MockAnimationNode("source", pose)
        layer = AnimationLayer(name="base", source=source)

        stack = LayerStack("test")
        stack.add_layer(layer)
        context = make_context()

        result = stack.evaluate(context)
        assert result.transforms[0].position[0] == 10.0

    def test_subsequent_layers_apply(self) -> None:
        """Path N3: Subsequent layers apply via apply_to_pose."""
        base_pose = make_pose([make_test_transform(5, 5, 5)])
        overlay_pose = make_pose([make_test_transform(10, 10, 10)])

        base_source = MockAnimationNode("base", base_pose)
        overlay_source = MockAnimationNode("overlay", overlay_pose)

        base_layer = AnimationLayer(name="base", source=base_source)
        overlay_layer = AnimationLayer(name="overlay", source=overlay_source, weight=0.5)

        stack = LayerStack("test")
        stack.add_layer(base_layer)
        stack.add_layer(overlay_layer)
        context = make_context()

        result = stack.evaluate(context)
        # Should be lerp of (5,5,5) and (10,10,10) at 0.5 = (7.5,7.5,7.5)
        assert abs(result.transforms[0].position[0] - 7.5) < 0.001

    def test_inactive_layers_skipped(self) -> None:
        """Path N4: Inactive layers skipped."""
        active_pose = make_pose([make_test_transform(10, 10, 10)])
        inactive_pose = make_pose([make_test_transform(100, 100, 100)])

        active_source = MockAnimationNode("active", active_pose)
        inactive_source = MockAnimationNode("inactive", inactive_pose)

        active_layer = AnimationLayer(name="active", source=active_source)
        inactive_layer = AnimationLayer(name="inactive", source=inactive_source, is_active=False)

        stack = LayerStack("test")
        stack.add_layer(active_layer)
        stack.add_layer(inactive_layer)
        context = make_context()

        result = stack.evaluate(context)
        # Should only reflect active layer
        assert result.transforms[0].position[0] == 10.0

    def test_zero_weight_layers_skipped(self) -> None:
        """Path N5: Zero weight layers skipped."""
        base_pose = make_pose([make_test_transform(5, 5, 5)])
        zero_pose = make_pose([make_test_transform(100, 100, 100)])

        base_source = MockAnimationNode("base", base_pose)
        zero_source = MockAnimationNode("zero", zero_pose)

        base_layer = AnimationLayer(name="base", source=base_source)
        zero_layer = AnimationLayer(name="zero", source=zero_source, weight=0.0)

        stack = LayerStack("test")
        stack.add_layer(base_layer)
        stack.add_layer(zero_layer)
        context = make_context()

        result = stack.evaluate(context)
        assert result.transforms[0].position[0] == 5.0

    def test_all_inactive_returns_empty(self) -> None:
        """Path N6: All inactive returns empty Pose."""
        pose = make_pose([make_test_transform(10, 10, 10)])
        source = MockAnimationNode("source", pose)
        layer = AnimationLayer(name="inactive", source=source, is_active=False)

        stack = LayerStack("test")
        stack.add_layer(layer)
        context = make_context()

        result = stack.evaluate(context)
        assert result.bone_count() == 0


# =============================================================================
# Path O: LayerStack.get_active_layers / get_layers_with_weight
# =============================================================================


class TestLayerStackHelpers:
    """Tests for LayerStack helper methods (Path O)."""

    def test_get_active_layers_filters(self) -> None:
        """Path O1: get_active_layers filters by is_active."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="active1"))
        stack.add_layer(AnimationLayer(name="inactive", is_active=False))
        stack.add_layer(AnimationLayer(name="active2"))

        active = stack.get_active_layers()
        assert len(active) == 2
        names = [l.name for l in active]
        assert "active1" in names
        assert "active2" in names
        assert "inactive" not in names

    def test_get_layers_with_weight_returns_tuples(self) -> None:
        """Path O2: get_layers_with_weight returns tuples."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="layer1", weight=0.5))
        stack.add_layer(AnimationLayer(name="layer2", weight=0.75))

        result = stack.get_layers_with_weight()
        assert len(result) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        assert result[0][1] == 0.5
        assert result[1][1] == 0.75


# =============================================================================
# Path P: LayerStackBuilder class
# =============================================================================


class TestLayerStackBuilderClass:
    """Tests for LayerStackBuilder class (Path P)."""

    def test_init_stores_node_id(self) -> None:
        """Path P1: __init__ stores node_id and empty layers."""
        builder = LayerStackBuilder("test_id")
        assert builder._node_id == "test_id"
        assert builder._layers == []

    def test_add_layer_appends(self) -> None:
        """Path P2: add_layer appends AnimationLayer."""
        builder = LayerStackBuilder("test")
        builder.add_layer("layer1")
        assert len(builder._layers) == 1
        assert builder._layers[0].name == "layer1"

    def test_add_layer_returns_self(self) -> None:
        """Path P3: add_layer returns self (fluent)."""
        builder = LayerStackBuilder("test")
        result = builder.add_layer("layer")
        assert result is builder

    def test_add_override_layer(self) -> None:
        """Path P4: add_override_layer sets OVERRIDE mode."""
        builder = LayerStackBuilder("test")
        builder.add_override_layer("layer")
        assert builder._layers[0].blend_mode == LayerBlendMode.OVERRIDE

    def test_add_additive_layer(self) -> None:
        """Path P5: add_additive_layer sets ADDITIVE mode."""
        builder = LayerStackBuilder("test")
        builder.add_additive_layer("layer")
        assert builder._layers[0].blend_mode == LayerBlendMode.ADDITIVE


# =============================================================================
# Path Q: LayerStackBuilder.build
# =============================================================================


class TestLayerStackBuilderBuild:
    """Tests for LayerStackBuilder.build (Path Q)."""

    def test_build_creates_stack_with_node_id(self) -> None:
        """Path Q1: build creates LayerStack with node_id."""
        builder = LayerStackBuilder("my_stack")
        stack = builder.build()
        assert stack.node_id == "my_stack"
        assert isinstance(stack, LayerStack)

    def test_build_adds_all_layers(self) -> None:
        """Path Q2: build adds all layers to stack."""
        builder = LayerStackBuilder("test")
        builder.add_layer("layer1")
        builder.add_layer("layer2")
        builder.add_layer("layer3")

        stack = builder.build()
        assert len(stack.layers) == 3
        assert stack.get_layer("layer1") is not None
        assert stack.get_layer("layer2") is not None
        assert stack.get_layer("layer3") is not None

    def test_fluent_chaining(self) -> None:
        """Path Q3: Fluent chaining works."""
        stack = (
            LayerStackBuilder("fluent")
            .add_layer("base")
            .add_additive_layer("overlay", weight=0.5)
            .add_override_layer("final", weight=1.0)
            .build()
        )

        assert len(stack.layers) == 3
        assert stack.layers[0].name == "base"
        assert stack.layers[1].name == "overlay"
        assert stack.layers[1].blend_mode == LayerBlendMode.ADDITIVE
        assert stack.layers[2].name == "final"


# =============================================================================
# Path R: BoneMaskPresets factory methods
# =============================================================================


class TestBoneMaskPresets:
    """Tests for BoneMaskPresets factory methods (Path R)."""

    @pytest.fixture
    def humanoid_skeleton(self) -> Skeleton:
        """Create a humanoid-like skeleton for testing."""
        skeleton = Skeleton()
        # Upper body
        for name in ["Spine", "Spine1", "Spine2", "Chest", "Neck", "Head"]:
            skeleton.add_bone(name)
        # Arms
        for side in ["Left", "Right"]:
            for part in ["Shoulder", "Arm", "ForeArm", "Hand"]:
                skeleton.add_bone(f"{side}{part}")
        # Lower body
        for name in ["Hips", "Pelvis"]:
            skeleton.add_bone(name)
        for side in ["Left", "Right"]:
            for part in ["UpLeg", "Leg", "Foot", "ToeBase"]:
                skeleton.add_bone(f"{side}{part}")
        return skeleton

    def test_upper_body_creates_mask(self, humanoid_skeleton: Skeleton) -> None:
        """Path R1: upper_body creates mask with upper body bones."""
        mask = BoneMaskPresets.upper_body(humanoid_skeleton)
        assert mask.name == "UpperBody"
        # Check some upper body bones are included
        spine = humanoid_skeleton.get_bone_by_name("Spine")
        if spine:
            assert mask.get_weight(spine.index) == 1.0

    def test_lower_body_creates_mask(self, humanoid_skeleton: Skeleton) -> None:
        """Path R2: lower_body creates mask with lower body bones."""
        mask = BoneMaskPresets.lower_body(humanoid_skeleton)
        assert mask.name == "LowerBody"
        hips = humanoid_skeleton.get_bone_by_name("Hips")
        if hips:
            assert mask.get_weight(hips.index) == 1.0

    def test_arm_masks(self, humanoid_skeleton: Skeleton) -> None:
        """Path R3: left_arm / right_arm / arms create arm masks."""
        left_mask = BoneMaskPresets.left_arm(humanoid_skeleton)
        right_mask = BoneMaskPresets.right_arm(humanoid_skeleton)
        arms_mask = BoneMaskPresets.arms(humanoid_skeleton)

        assert left_mask.name == "LeftArm"
        assert right_mask.name == "RightArm"
        assert arms_mask.name == "Arms"

    def test_leg_masks(self, humanoid_skeleton: Skeleton) -> None:
        """Path R4: left_leg / right_leg / legs create leg masks."""
        left_mask = BoneMaskPresets.left_leg(humanoid_skeleton)
        right_mask = BoneMaskPresets.right_leg(humanoid_skeleton)
        legs_mask = BoneMaskPresets.legs(humanoid_skeleton)

        assert left_mask.name == "LeftLeg"
        assert right_mask.name == "RightLeg"
        assert legs_mask.name == "Legs"

    def test_head_spine_masks(self, humanoid_skeleton: Skeleton) -> None:
        """Path R5: head / spine create targeted masks."""
        head_mask = BoneMaskPresets.head(humanoid_skeleton)
        spine_mask = BoneMaskPresets.spine(humanoid_skeleton)

        assert head_mask.name == "Head"
        assert spine_mask.name == "Spine"

    def test_full_body_mask(self, humanoid_skeleton: Skeleton) -> None:
        """Path R6: full_body creates mask with all bones."""
        mask = BoneMaskPresets.full_body(humanoid_skeleton)
        assert mask.name == "FullBody"
        # All bones should be at 1.0
        for bone in humanoid_skeleton.bones:
            assert mask.get_weight(bone.index) == 1.0

    def test_gradient_upper_lower_mask(self, humanoid_skeleton: Skeleton) -> None:
        """Path R7: gradient_upper_lower creates gradient mask."""
        mask = BoneMaskPresets.gradient_upper_lower(humanoid_skeleton)
        assert mask.name == "GradientUpperLower"

        # Lower body should be 0.0
        hips = humanoid_skeleton.get_bone_by_name("Hips")
        if hips:
            assert mask.get_weight(hips.index) == 0.0


# =============================================================================
# Path S: Edge Cases
# =============================================================================


class TestLayerEdgeCases:
    """Tests for edge cases (Path S)."""

    def test_layer_with_none_source_in_stack(self) -> None:
        """Path S1: Layer with None source in stack evaluation."""
        layer_with_source = AnimationLayer(
            name="with_source",
            source=MockAnimationNode("src", make_identity_pose(3)),
        )
        layer_no_source = AnimationLayer(name="no_source", source=None)

        stack = LayerStack("test")
        stack.add_layer(layer_no_source)  # First layer has no source
        stack.add_layer(layer_with_source)

        context = make_context()
        result = stack.evaluate(context)
        # Should still work, using the layer with source
        assert result.bone_count() == 3

    def test_empty_builder_creates_empty_stack(self) -> None:
        """Path S2: Empty layer list in builder.build()."""
        builder = LayerStackBuilder("empty")
        stack = builder.build()
        assert len(stack.layers) == 0

    def test_duplicate_name_in_builder_raises_in_stack(self) -> None:
        """Path S3: Duplicate layer name in builder (allowed in builder, stack raises)."""
        builder = LayerStackBuilder("test")
        # Builder allows duplicates
        builder.add_layer("dup")
        builder.add_layer("dup")

        # But stack.add_layer will raise on second add
        with pytest.raises(ValueError, match="already exists"):
            builder.build()

    def test_pose_bone_count_mismatch_masked(self) -> None:
        """Path S4: Pose bone_count mismatch in masked apply."""
        mask = BoneMask(name="mask")
        mask.set_weight(0, 1.0)
        mask.set_weight(1, 1.0)

        # Base has 3 bones, layer has 1
        base_pose = make_pose([
            make_test_transform(0, 0, 0),
            make_test_transform(0, 0, 0),
            make_test_transform(0, 0, 0),
        ])
        layer_pose = make_pose([make_test_transform(10, 10, 10)])

        layer = AnimationLayer(name="test", mask=mask)
        # Should not raise, uses min of bone_count
        result = layer._apply_masked(base_pose, layer_pose, 1.0)
        assert result is not None
        # Only first bone should be modified (min overlap)
        assert result.transforms[0].position[0] == 10.0

    def test_bone_mask_presets_missing_bones(self) -> None:
        """Path S5: BoneMaskPresets with skeleton missing bones."""
        # Skeleton with only a few bones
        skeleton = Skeleton()
        skeleton.add_bone("CustomBone1")
        skeleton.add_bone("CustomBone2")

        # Should not raise, just creates empty/partial mask
        mask = BoneMaskPresets.upper_body(skeleton)
        assert mask is not None
        # No standard bones found
        assert all(mask.get_weight(i) == 0.0 for i in range(2))


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestLayerStackLayerCount:
    """Tests for LayerStack.layer_count method."""

    def test_layer_count_empty(self) -> None:
        """Layer count is 0 for empty stack."""
        stack = LayerStack("test")
        assert stack.layer_count() == 0

    def test_layer_count_with_layers(self) -> None:
        """Layer count reflects added layers."""
        stack = LayerStack("test")
        stack.add_layer(AnimationLayer(name="a"))
        stack.add_layer(AnimationLayer(name="b"))
        assert stack.layer_count() == 2


class TestLayerBuilderWithParameters:
    """Tests for LayerStackBuilder with all parameters."""

    def test_add_layer_with_all_params(self) -> None:
        """Builder add_layer accepts all parameters."""
        source = MockAnimationNode("src", make_identity_pose(3))
        mask = BoneMask(name="mask")

        builder = LayerStackBuilder("test")
        builder.add_layer(
            name="full_params",
            source=source,
            weight=0.75,
            mask=mask,
            blend_mode=LayerBlendMode.ADDITIVE,
        )

        layer = builder._layers[0]
        assert layer.name == "full_params"
        assert layer.source is source
        assert layer.weight == 0.75
        assert layer.mask is mask
        assert layer.blend_mode == LayerBlendMode.ADDITIVE


class TestLayerStackMultipleLayers:
    """Integration tests for multi-layer evaluation."""

    def test_three_layer_blend(self) -> None:
        """Three layers blend correctly in sequence."""
        pose1 = make_pose([make_test_transform(0, 0, 0)])
        pose2 = make_pose([make_test_transform(10, 0, 0)])
        pose3 = make_pose([make_test_transform(0, 10, 0)])

        source1 = MockAnimationNode("s1", pose1)
        source2 = MockAnimationNode("s2", pose2)
        source3 = MockAnimationNode("s3", pose3)

        stack = (
            LayerStackBuilder("multi")
            .add_layer("base", source=source1)
            .add_layer("x_overlay", source=source2, weight=0.5)
            .add_layer("y_overlay", source=source3, weight=0.5)
            .build()
        )

        context = make_context()
        result = stack.evaluate(context)

        # Base = (0,0,0)
        # After x_overlay at 0.5: lerp((0,0,0), (10,0,0), 0.5) = (5,0,0)
        # After y_overlay at 0.5: lerp((5,0,0), (0,10,0), 0.5) = (2.5,5,0)
        assert abs(result.transforms[0].position[0] - 2.5) < 0.001
        assert abs(result.transforms[0].position[1] - 5.0) < 0.001

    def test_additive_and_override_mix(self) -> None:
        """Mix of additive and override layers."""
        base_pose = make_pose([make_test_transform(5, 5, 5)])
        additive_pose = make_pose([make_test_transform(2, 2, 2)])
        override_pose = make_pose([make_test_transform(10, 10, 10)])

        base_source = MockAnimationNode("base", base_pose)
        additive_source = MockAnimationNode("add", additive_pose)
        override_source = MockAnimationNode("override", override_pose)

        stack = (
            LayerStackBuilder("mix")
            .add_layer("base", source=base_source)
            .add_additive_layer("additive", source=additive_source, weight=1.0)
            .add_override_layer("override", source=override_source, weight=0.5)
            .build()
        )

        context = make_context()
        result = stack.evaluate(context)

        # After base: (5,5,5)
        # After additive: (5+2, 5+2, 5+2) = (7,7,7)
        # After override at 0.5: lerp((7,7,7), (10,10,10), 0.5) = (8.5,8.5,8.5)
        assert abs(result.transforms[0].position[0] - 8.5) < 0.001
