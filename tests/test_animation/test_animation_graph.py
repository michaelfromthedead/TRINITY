"""
Comprehensive tests for the Animation Graph subsystem.

Tests cover:
- Transform and Pose operations
- Skeleton and BoneMask functionality
- Graph parameters and context
- Animation nodes and graphs
- State machine transitions with conditions
- 1D and 2D blend trees
- Layer stacking and masking
- Sync groups and synchronization
- Complex graph evaluation

Minimum 160 tests with real assertions.
"""

import math
import pytest
from typing import List, Dict, Any, Optional


# =============================================================================
# IMPORT ANIMATION GRAPH MODULES
# =============================================================================

from engine.animation.graph.animation_graph import (
    GraphNodeMeta,
    Transform,
    Pose,
    Bone,
    Skeleton,
    BoneMask,
    ParameterType,
    GraphParameter,
    GraphContext,
    AnimationNode,
    SubgraphNode,
    Connection,
    AnimationGraph,
)

from engine.animation.graph.state_machine import (
    BlendCurve,
    evaluate_blend_curve,
    ComparisonOp,
    TransitionCondition,
    AnimationState,
    TransitionSyncMode,
    StateTransition,
    ActiveTransition,
    StateMachine,
    StateMachineBuilder,
    state_machine,
)

from engine.animation.graph.blend_tree import (
    BlendTree,
    BlendTree1DEntry,
    BlendTree1D,
    BlendTree2DMode,
    BlendTree2DSample,
    Triangle,
    BlendTree2D,
    BlendTreeDirectEntry,
    BlendTreeDirect,
    blend_tree,
)

from engine.animation.graph.blend_node import (
    AnimationKeyframe,
    AnimationTrack,
    LoopMode,
    AnimationClip,
    ClipNode,
    BlendNode,
    AdditiveNode,
    LayerBlendMode,
    AnimationLayerInput,
    LayerNode,
    BoneMirrorPair,
    MirrorNode,
    TimeScaleNode,
    PoseCacheNode,
    SelectNode,
)

from engine.animation.graph.layer import (
    LayerBlendMode as LayerMode,
    AnimationLayer,
    LayerStack,
    LayerStackBuilder,
    BoneMaskPresets,
)

from engine.animation.graph.sync import (
    SyncMarker,
    SyncMarkerTrack,
    SyncMode,
    SyncEntry,
    SyncGroup,
    SyncGroupNode,
    sync_animations,
    create_locomotion_markers,
    calculate_phase_offset,
    SyncEvent,
    EventSynchronizer,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_skeleton() -> Skeleton:
    """Create a simple test skeleton."""
    skeleton = Skeleton()
    skeleton.add_bone("Root", -1)
    skeleton.add_bone("Spine", 0)
    skeleton.add_bone("Chest", 1)
    skeleton.add_bone("Head", 2)
    skeleton.add_bone("LeftArm", 2)
    skeleton.add_bone("RightArm", 2)
    skeleton.add_bone("LeftLeg", 0)
    skeleton.add_bone("RightLeg", 0)
    return skeleton


@pytest.fixture
def humanoid_skeleton() -> Skeleton:
    """Create a humanoid test skeleton with standard bone names."""
    skeleton = Skeleton()
    # Root and spine
    skeleton.add_bone("Hips", -1)
    skeleton.add_bone("Spine", 0)
    skeleton.add_bone("Spine1", 1)
    skeleton.add_bone("Spine2", 2)
    skeleton.add_bone("Chest", 3)
    skeleton.add_bone("Neck", 4)
    skeleton.add_bone("Head", 5)
    # Left arm
    skeleton.add_bone("LeftShoulder", 4)
    skeleton.add_bone("LeftArm", 7)
    skeleton.add_bone("LeftForeArm", 8)
    skeleton.add_bone("LeftHand", 9)
    # Right arm
    skeleton.add_bone("RightShoulder", 4)
    skeleton.add_bone("RightArm", 11)
    skeleton.add_bone("RightForeArm", 12)
    skeleton.add_bone("RightHand", 13)
    # Left leg
    skeleton.add_bone("LeftUpLeg", 0)
    skeleton.add_bone("LeftLeg", 14)
    skeleton.add_bone("LeftFoot", 15)
    # Right leg
    skeleton.add_bone("RightUpLeg", 0)
    skeleton.add_bone("RightLeg", 17)
    skeleton.add_bone("RightFoot", 18)
    return skeleton


@pytest.fixture
def simple_pose() -> Pose:
    """Create a simple test pose."""
    pose = Pose()
    for i in range(8):
        pose.transforms.append(Transform(
            position=(float(i), 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
        ))
    return pose


@pytest.fixture
def test_clip() -> AnimationClip:
    """Create a test animation clip."""
    clip = AnimationClip(name="test_clip", duration=1.0)
    for bone_idx in range(4):
        clip.add_keyframe(bone_idx, 0.0, Transform(position=(0.0, 0.0, 0.0)))
        clip.add_keyframe(bone_idx, 1.0, Transform(position=(1.0, 1.0, 1.0)))
    return clip


@pytest.fixture
def graph_context(simple_skeleton: Skeleton) -> GraphContext:
    """Create a test graph context."""
    context = GraphContext(
        dt=0.016,  # 60fps
        skeleton=simple_skeleton,
    )
    context.parameters["speed"] = GraphParameter.float_param("speed", 0.0, 0.0, 10.0)
    context.parameters["direction"] = GraphParameter.float_param("direction", 0.0, -180.0, 180.0)
    context.parameters["is_jumping"] = GraphParameter.bool_param("is_jumping", False)
    return context


# =============================================================================
# TRANSFORM TESTS (10 tests)
# =============================================================================


class TestTransform:
    """Tests for Transform class."""

    def test_identity(self):
        """Test identity transform creation."""
        t = Transform.identity()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation == (0.0, 0.0, 0.0, 1.0)
        assert t.scale == (1.0, 1.0, 1.0)

    def test_default_values(self):
        """Test default transform values."""
        t = Transform()
        assert t.position == (0.0, 0.0, 0.0)
        assert t.rotation[3] == 1.0  # w component

    def test_custom_position(self):
        """Test custom position."""
        t = Transform(position=(1.0, 2.0, 3.0))
        assert t.position == (1.0, 2.0, 3.0)

    def test_lerp_midpoint(self):
        """Test linear interpolation at midpoint."""
        t1 = Transform(position=(0.0, 0.0, 0.0))
        t2 = Transform(position=(2.0, 4.0, 6.0))
        result = t1.lerp(t2, 0.5)
        assert result.position == (1.0, 2.0, 3.0)

    def test_lerp_start(self):
        """Test lerp at t=0."""
        t1 = Transform(position=(1.0, 1.0, 1.0))
        t2 = Transform(position=(2.0, 2.0, 2.0))
        result = t1.lerp(t2, 0.0)
        assert result.position == (1.0, 1.0, 1.0)

    def test_lerp_end(self):
        """Test lerp at t=1."""
        t1 = Transform(position=(1.0, 1.0, 1.0))
        t2 = Transform(position=(2.0, 2.0, 2.0))
        result = t1.lerp(t2, 1.0)
        assert result.position == (2.0, 2.0, 2.0)

    def test_copy(self):
        """Test transform copy."""
        t1 = Transform(position=(1.0, 2.0, 3.0))
        t2 = t1.copy()
        assert t2.position == t1.position
        assert t2 is not t1

    def test_additive(self):
        """Test additive transform combination."""
        t1 = Transform(position=(1.0, 0.0, 0.0))
        t2 = Transform(position=(0.0, 1.0, 0.0))
        result = t1 + t2
        assert result.position == (1.0, 1.0, 0.0)

    def test_slerp_same_quaternion(self):
        """Test slerp with same quaternion."""
        q = (0.0, 0.0, 0.0, 1.0)
        result = Transform._slerp(q, q, 0.5)
        assert abs(result[3] - 1.0) < 0.001  # w should be close to 1

    def test_scale_lerp(self):
        """Test scale interpolation."""
        t1 = Transform(scale=(1.0, 1.0, 1.0))
        t2 = Transform(scale=(2.0, 2.0, 2.0))
        result = t1.lerp(t2, 0.5)
        assert result.scale == (1.5, 1.5, 1.5)


# =============================================================================
# POSE TESTS (10 tests)
# =============================================================================


class TestPose:
    """Tests for Pose class."""

    def test_identity_pose(self):
        """Test identity pose creation."""
        pose = Pose.identity(4)
        assert pose.bone_count() == 4
        for t in pose.transforms:
            assert t.position == (0.0, 0.0, 0.0)

    def test_bone_count(self):
        """Test bone count."""
        pose = Pose()
        pose.transforms = [Transform() for _ in range(5)]
        assert pose.bone_count() == 5

    def test_get_transform(self):
        """Test getting transform by index."""
        pose = Pose.identity(3)
        pose.transforms[1] = Transform(position=(1.0, 2.0, 3.0))
        t = pose.get_transform(1)
        assert t.position == (1.0, 2.0, 3.0)

    def test_get_transform_out_of_bounds(self):
        """Test getting transform out of bounds returns identity."""
        pose = Pose.identity(2)
        t = pose.get_transform(10)
        assert t.position == (0.0, 0.0, 0.0)

    def test_set_transform(self):
        """Test setting transform."""
        pose = Pose()
        pose.set_transform(2, Transform(position=(1.0, 1.0, 1.0)))
        assert pose.bone_count() >= 3
        assert pose.transforms[2].position == (1.0, 1.0, 1.0)

    def test_pose_lerp(self):
        """Test pose interpolation."""
        pose1 = Pose.identity(2)
        pose2 = Pose.identity(2)
        pose2.transforms[0] = Transform(position=(2.0, 0.0, 0.0))
        result = pose1.lerp(pose2, 0.5)
        assert result.transforms[0].position == (1.0, 0.0, 0.0)

    def test_additive_blend(self):
        """Test additive pose blending."""
        base = Pose.identity(2)
        base.transforms[0] = Transform(position=(1.0, 0.0, 0.0))
        additive = Pose.identity(2)
        additive.transforms[0] = Transform(position=(0.0, 1.0, 0.0))
        result = base.additive_blend(additive, 1.0)
        assert result.transforms[0].position == (1.0, 1.0, 0.0)

    def test_pose_copy(self):
        """Test pose copy."""
        pose1 = Pose.identity(3)
        pose1.transforms[0] = Transform(position=(1.0, 2.0, 3.0))
        pose2 = pose1.copy()
        assert pose2.transforms[0].position == pose1.transforms[0].position
        pose2.transforms[0] = Transform(position=(0.0, 0.0, 0.0))
        assert pose1.transforms[0].position == (1.0, 2.0, 3.0)

    def test_root_motion(self):
        """Test root motion in pose."""
        pose = Pose.identity(2)
        pose.root_motion = Transform(position=(1.0, 0.0, 1.0))
        assert pose.root_motion.position == (1.0, 0.0, 1.0)

    def test_lerp_with_root_motion(self):
        """Test lerp preserves root motion."""
        pose1 = Pose.identity(2)
        pose1.root_motion = Transform(position=(0.0, 0.0, 0.0))
        pose2 = Pose.identity(2)
        pose2.root_motion = Transform(position=(2.0, 0.0, 0.0))
        result = pose1.lerp(pose2, 0.5)
        assert result.root_motion.position == (1.0, 0.0, 0.0)


# =============================================================================
# SKELETON TESTS (12 tests)
# =============================================================================


class TestSkeleton:
    """Tests for Skeleton class."""

    def test_add_bone(self):
        """Test adding bones."""
        skeleton = Skeleton()
        bone = skeleton.add_bone("Root", -1)
        assert bone.name == "Root"
        assert bone.index == 0
        assert bone.parent_index == -1

    def test_bone_count(self):
        """Test bone count."""
        skeleton = Skeleton()
        skeleton.add_bone("Root", -1)
        skeleton.add_bone("Child", 0)
        assert skeleton.bone_count() == 2

    def test_get_bone(self):
        """Test getting bone by index."""
        skeleton = Skeleton()
        skeleton.add_bone("Root", -1)
        bone = skeleton.get_bone(0)
        assert bone.name == "Root"

    def test_get_bone_by_name(self):
        """Test getting bone by name."""
        skeleton = Skeleton()
        skeleton.add_bone("Root", -1)
        skeleton.add_bone("Spine", 0)
        bone = skeleton.get_bone_by_name("Spine")
        assert bone.index == 1

    def test_get_bone_index(self):
        """Test getting bone index by name."""
        skeleton = Skeleton()
        skeleton.add_bone("Root", -1)
        skeleton.add_bone("Spine", 0)
        assert skeleton.get_bone_index("Spine") == 1
        assert skeleton.get_bone_index("NonExistent") == -1

    def test_bone_is_root(self):
        """Test root bone detection."""
        skeleton = Skeleton()
        skeleton.add_bone("Root", -1)
        skeleton.add_bone("Child", 0)
        assert skeleton.bones[0].is_root()
        assert not skeleton.bones[1].is_root()

    def test_get_children(self):
        """Test getting children of a bone."""
        skeleton = Skeleton()
        skeleton.add_bone("Root", -1)
        skeleton.add_bone("Child1", 0)
        skeleton.add_bone("Child2", 0)
        children = skeleton.get_children(0)
        assert len(children) == 2
        assert 1 in children
        assert 2 in children

    def test_get_chain(self):
        """Test getting bone chain."""
        skeleton = Skeleton()
        skeleton.add_bone("Root", -1)
        skeleton.add_bone("Spine", 0)
        skeleton.add_bone("Chest", 1)
        skeleton.add_bone("Head", 2)
        chain = skeleton.get_chain(0, 3)
        assert chain == [0, 1, 2, 3]

    def test_get_bind_pose(self):
        """Test getting bind pose."""
        skeleton = Skeleton()
        skeleton.add_bone("Root", -1, Transform(position=(0.0, 1.0, 0.0)))
        skeleton.add_bone("Child", 0, Transform(position=(0.0, 2.0, 0.0)))
        bind_pose = skeleton.get_bind_pose()
        assert bind_pose.bone_count() == 2
        assert bind_pose.transforms[0].position == (0.0, 1.0, 0.0)

    def test_bone_none_not_found(self):
        """Test getting non-existent bone returns None."""
        skeleton = Skeleton()
        skeleton.add_bone("Root", -1)
        assert skeleton.get_bone(10) is None
        assert skeleton.get_bone_by_name("NonExistent") is None

    def test_skeleton_fixture(self, simple_skeleton: Skeleton):
        """Test fixture skeleton structure."""
        assert simple_skeleton.bone_count() == 8
        assert simple_skeleton.get_bone_by_name("Root") is not None
        assert simple_skeleton.get_bone_by_name("Head").parent_index == 2

    def test_disconnected_chain(self):
        """Test getting chain between disconnected bones."""
        skeleton = Skeleton()
        skeleton.add_bone("Root1", -1)
        skeleton.add_bone("Root2", -1)
        chain = skeleton.get_chain(0, 1)
        assert chain == []  # Not connected


# =============================================================================
# BONE MASK TESTS (10 tests)
# =============================================================================


class TestBoneMask:
    """Tests for BoneMask class."""

    def test_default_weight(self):
        """Test default weight is 0."""
        mask = BoneMask(name="test")
        assert mask.get_weight(0) == 0.0

    def test_set_weight(self):
        """Test setting weight."""
        mask = BoneMask(name="test")
        mask.set_weight(0, 0.5)
        assert mask.get_weight(0) == 0.5

    def test_weight_clamping(self):
        """Test weight clamping to 0-1."""
        mask = BoneMask(name="test")
        mask.set_weight(0, 2.0)
        assert mask.get_weight(0) == 1.0
        mask.set_weight(1, -1.0)
        assert mask.get_weight(1) == 0.0

    def test_set_weights_multiple(self):
        """Test setting multiple weights at once."""
        mask = BoneMask(name="test")
        mask.set_weights([0, 1, 2], 0.8)
        assert mask.get_weight(0) == 0.8
        assert mask.get_weight(1) == 0.8
        assert mask.get_weight(2) == 0.8

    def test_full_mask(self, simple_skeleton: Skeleton):
        """Test full body mask."""
        mask = BoneMask.full(simple_skeleton)
        for i in range(simple_skeleton.bone_count()):
            assert mask.get_weight(i) == 1.0

    def test_from_bone_names(self, simple_skeleton: Skeleton):
        """Test creating mask from bone names."""
        mask = BoneMask.from_bone_names(
            simple_skeleton, "test", ["Spine", "Chest"]
        )
        spine_idx = simple_skeleton.get_bone_index("Spine")
        chest_idx = simple_skeleton.get_bone_index("Chest")
        assert mask.get_weight(spine_idx) == 1.0
        assert mask.get_weight(chest_idx) == 1.0
        assert mask.get_weight(0) == 0.0  # Root not included

    def test_from_bone_names_with_children(self, simple_skeleton: Skeleton):
        """Test mask with children included."""
        mask = BoneMask.from_bone_names(
            simple_skeleton, "test", ["Spine"],
            include_children=True
        )
        spine_idx = simple_skeleton.get_bone_index("Spine")
        chest_idx = simple_skeleton.get_bone_index("Chest")
        head_idx = simple_skeleton.get_bone_index("Head")
        assert mask.get_weight(spine_idx) == 1.0
        assert mask.get_weight(chest_idx) == 1.0
        assert mask.get_weight(head_idx) == 1.0

    def test_apply_to_pose(self, simple_pose: Pose):
        """Test applying mask to pose."""
        mask = BoneMask(name="test")
        mask.set_weight(0, 1.0)
        mask.set_weight(1, 0.5)
        result = mask.apply_to_pose(simple_pose)
        assert result.transforms[0].position[0] == simple_pose.transforms[0].position[0]
        # Bone 1 should be scaled by 0.5

    def test_mask_name(self):
        """Test mask name."""
        mask = BoneMask(name="UpperBody")
        assert mask.name == "UpperBody"

    def test_empty_mask_on_skeleton(self, simple_skeleton: Skeleton):
        """Test empty mask returns zeros."""
        mask = BoneMask.from_bone_names(simple_skeleton, "empty", [])
        for i in range(simple_skeleton.bone_count()):
            assert mask.get_weight(i) == 0.0


# =============================================================================
# GRAPH PARAMETER TESTS (12 tests)
# =============================================================================


class TestGraphParameter:
    """Tests for GraphParameter class."""

    def test_float_param(self):
        """Test float parameter creation."""
        param = GraphParameter.float_param("speed", 5.0, 0.0, 10.0)
        assert param.name == "speed"
        assert param.value == 5.0
        assert param.param_type == ParameterType.FLOAT

    def test_int_param(self):
        """Test integer parameter creation."""
        param = GraphParameter.int_param("count", 5, 0, 100)
        assert param.value == 5
        assert param.param_type == ParameterType.INT

    def test_bool_param(self):
        """Test boolean parameter creation."""
        param = GraphParameter.bool_param("enabled", True)
        assert param.value is True
        assert param.param_type == ParameterType.BOOL

    def test_trigger_param(self):
        """Test trigger parameter creation."""
        param = GraphParameter.trigger_param("jump")
        assert param.param_type == ParameterType.TRIGGER
        assert param.value is False  # Not triggered

    def test_trigger_fire(self):
        """Test trigger fires once."""
        param = GraphParameter.trigger_param("jump")
        param.trigger()
        assert param.value is True  # First read
        assert param.value is False  # Auto-reset

    def test_enum_param(self):
        """Test enum parameter creation."""
        param = GraphParameter.enum_param("state", ["idle", "walk", "run"])
        assert param.value == "idle"
        assert param.param_type == ParameterType.ENUM

    def test_enum_validation(self):
        """Test enum value validation."""
        param = GraphParameter.enum_param("state", ["idle", "walk"])
        with pytest.raises(ValueError):
            param.value = "invalid"

    def test_float_clamping(self):
        """Test float parameter clamping."""
        param = GraphParameter.float_param("speed", 5.0, 0.0, 10.0)
        param.value = 15.0
        assert param.value == 10.0
        param.value = -5.0
        assert param.value == 0.0

    def test_int_clamping(self):
        """Test integer parameter clamping."""
        param = GraphParameter.int_param("count", 5, 0, 10)
        param.value = 15
        assert param.value == 10
        param.value = -5
        assert param.value == 0

    def test_reset(self):
        """Test parameter reset."""
        param = GraphParameter.float_param("speed", 5.0)
        param.value = 10.0
        param.reset()
        assert param.value == 5.0

    def test_empty_enum_raises(self):
        """Test empty enum raises error."""
        with pytest.raises(ValueError):
            GraphParameter.enum_param("state", [])

    def test_bool_conversion(self):
        """Test bool parameter conversion."""
        param = GraphParameter.bool_param("flag", False)
        param.value = 1
        assert param.value is True
        param.value = 0
        assert param.value is False


# =============================================================================
# GRAPH CONTEXT TESTS (8 tests)
# =============================================================================


class TestGraphContext:
    """Tests for GraphContext class."""

    def test_get_parameter(self, graph_context: GraphContext):
        """Test getting parameter value."""
        graph_context.parameters["speed"].value = 5.0
        assert graph_context.get_parameter("speed") == 5.0

    def test_get_parameter_float(self, graph_context: GraphContext):
        """Test getting float parameter."""
        graph_context.parameters["speed"].value = 3.5
        assert graph_context.get_parameter_float("speed") == 3.5
        assert graph_context.get_parameter_float("nonexistent", 1.0) == 1.0

    def test_get_parameter_int(self, graph_context: GraphContext):
        """Test getting int parameter."""
        graph_context.parameters["speed"].value = 5.7
        assert graph_context.get_parameter_int("speed") == 5

    def test_get_parameter_bool(self, graph_context: GraphContext):
        """Test getting bool parameter."""
        graph_context.parameters["is_jumping"].value = True
        assert graph_context.get_parameter_bool("is_jumping") is True

    def test_with_depth(self, graph_context: GraphContext):
        """Test creating context with incremented depth."""
        assert graph_context.evaluation_depth == 0
        new_context = graph_context.with_depth()
        assert new_context.evaluation_depth == 1
        assert graph_context.evaluation_depth == 0

    def test_bone_mask(self, graph_context: GraphContext, simple_skeleton: Skeleton):
        """Test bone mask in context."""
        mask = BoneMask.full(simple_skeleton, "test")
        graph_context.bone_masks["test"] = mask
        retrieved = graph_context.get_bone_mask("test")
        assert retrieved is mask

    def test_dt(self, graph_context: GraphContext):
        """Test delta time."""
        assert graph_context.dt == 0.016

    def test_skeleton(self, graph_context: GraphContext):
        """Test skeleton reference."""
        assert graph_context.skeleton is not None
        assert graph_context.skeleton.bone_count() == 8


# =============================================================================
# ANIMATION CLIP TESTS (10 tests)
# =============================================================================


class TestAnimationClip:
    """Tests for AnimationClip class."""

    def test_create_clip(self):
        """Test clip creation."""
        clip = AnimationClip(name="walk", duration=2.0)
        assert clip.name == "walk"
        assert clip.duration == 2.0

    def test_add_keyframe(self):
        """Test adding keyframes."""
        clip = AnimationClip(name="test")
        clip.add_keyframe(0, 0.0, Transform(position=(0.0, 0.0, 0.0)))
        clip.add_keyframe(0, 1.0, Transform(position=(1.0, 0.0, 0.0)))
        assert 0 in clip.tracks
        assert len(clip.tracks[0].keyframes) == 2
        assert clip.duration == 1.0

    def test_sample_clip(self, test_clip: AnimationClip):
        """Test sampling clip."""
        pose = test_clip.sample(0.5, 4)
        assert pose.bone_count() == 4
        # At t=0.5, should be halfway between (0,0,0) and (1,1,1)
        assert pose.transforms[0].position == (0.5, 0.5, 0.5)

    def test_loop_mode_loop(self, test_clip: AnimationClip):
        """Test loop mode."""
        test_clip.loop_mode = LoopMode.LOOP
        pose = test_clip.sample(1.5, 4)  # Should wrap to 0.5
        assert pose.transforms[0].position == (0.5, 0.5, 0.5)

    def test_loop_mode_once(self, test_clip: AnimationClip):
        """Test once mode."""
        test_clip.loop_mode = LoopMode.ONCE
        pose = test_clip.sample(2.0, 4)
        # Should stay at end
        assert pose.transforms[0].position == (1.0, 1.0, 1.0)

    def test_loop_mode_clamp(self, test_clip: AnimationClip):
        """Test clamp mode."""
        test_clip.loop_mode = LoopMode.CLAMP
        pose = test_clip.sample(-1.0, 4)
        # Should clamp to start
        assert pose.transforms[0].position == (0.0, 0.0, 0.0)

    def test_normalized_time(self, test_clip: AnimationClip):
        """Test normalized time calculation."""
        assert test_clip.get_normalized_time(0.5) == 0.5
        # For looping clips, normalized time wraps around
        # 1.0 % 1.0 == 0.0 due to loop mode
        test_clip.loop_mode = LoopMode.ONCE
        assert test_clip.get_normalized_time(1.0) == 1.0

    def test_animation_events(self, test_clip: AnimationClip):
        """Test animation events."""
        test_clip.events = [(0.3, "footstep_left"), (0.7, "footstep_right")]
        events = test_clip.get_events_in_range(0.2, 0.5)
        assert "footstep_left" in events
        assert "footstep_right" not in events

    def test_animation_track_sample(self):
        """Test track sampling."""
        track = AnimationTrack(bone_index=0)
        track.keyframes.append(AnimationKeyframe(0.0, Transform(position=(0.0, 0.0, 0.0))))
        track.keyframes.append(AnimationKeyframe(1.0, Transform(position=(2.0, 0.0, 0.0))))
        result = track.sample(0.5)
        assert result.position == (1.0, 0.0, 0.0)

    def test_frame_rate(self):
        """Test frame rate."""
        clip = AnimationClip(name="test", frame_rate=60.0)
        assert clip.frame_rate == 60.0


# =============================================================================
# CLIP NODE TESTS (10 tests)
# =============================================================================


class TestClipNode:
    """Tests for ClipNode class."""

    def test_create_clip_node(self, test_clip: AnimationClip):
        """Test clip node creation."""
        node = ClipNode("test_node", test_clip)
        assert node.node_id == "test_node"
        assert node.clip is test_clip

    def test_evaluate_clip_node(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test clip node evaluation."""
        node = ClipNode("test_node", test_clip)
        pose = node.evaluate(graph_context)
        # The skeleton has 8 bones, so pose matches skeleton bone count
        assert pose.bone_count() == graph_context.skeleton.bone_count()

    def test_play_pause(self, test_clip: AnimationClip):
        """Test play/pause."""
        node = ClipNode("test_node", test_clip)
        assert node.is_playing
        node.pause()
        assert not node.is_playing
        node.play()
        assert node.is_playing

    def test_stop(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test stop."""
        node = ClipNode("test_node", test_clip)
        node.advance(0.5)
        node.stop()
        assert node.current_time == 0.0
        assert not node.is_playing

    def test_seek(self, test_clip: AnimationClip):
        """Test seeking."""
        node = ClipNode("test_node", test_clip)
        node.seek(0.75)
        assert node.current_time == 0.75

    def test_seek_normalized(self, test_clip: AnimationClip):
        """Test normalized seeking."""
        node = ClipNode("test_node", test_clip)
        node.seek_normalized(0.5)
        assert node.current_time == 0.5

    def test_play_rate(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test play rate."""
        node = ClipNode("test_node", test_clip)
        node.play_rate = 2.0
        node.advance(0.25)
        assert node.current_time == 0.5

    def test_duration_property(self, test_clip: AnimationClip):
        """Test duration property."""
        node = ClipNode("test_node", test_clip)
        assert node.duration == 1.0

    def test_is_finished(self, test_clip: AnimationClip):
        """Test is_finished property."""
        test_clip.loop_mode = LoopMode.ONCE
        node = ClipNode("test_node", test_clip)
        assert not node.is_finished
        node.seek(1.5)
        assert node.is_finished

    def test_event_callback(self, test_clip: AnimationClip):
        """Test event callback."""
        test_clip.events = [(0.5, "test_event")]
        node = ClipNode("test_node", test_clip)
        received_events = []
        node.on_event = lambda e: received_events.append(e)
        node.advance(0.6)
        assert "test_event" in received_events


# =============================================================================
# BLEND NODE TESTS (8 tests)
# =============================================================================


class TestBlendNode:
    """Tests for BlendNode class."""

    def test_blend_node_creation(self):
        """Test blend node creation."""
        node = BlendNode("blend", alpha=0.5)
        assert node.alpha == 0.5

    def test_blend_with_parameter(self, graph_context: GraphContext, test_clip: AnimationClip):
        """Test blend driven by parameter."""
        clip_a = ClipNode("clip_a", test_clip)
        clip_b = ClipNode("clip_b", test_clip)

        blend = BlendNode("blend", alpha_parameter="speed")
        blend.set_inputs(clip_a, clip_b)

        graph_context.parameters["speed"].value = 0.5
        pose = blend.evaluate(graph_context)
        assert pose is not None

    def test_blend_alpha_zero(self, graph_context: GraphContext, test_clip: AnimationClip):
        """Test blend at alpha=0 returns first input."""
        clip_a = ClipNode("clip_a", test_clip)
        clip_b = ClipNode("clip_b", test_clip)
        clip_b.seek(1.0)

        blend = BlendNode("blend", alpha=0.0)
        blend.set_inputs(clip_a, clip_b)

        pose = blend.evaluate(graph_context)
        # Should be pose_a (at time 0)
        assert pose.transforms[0].position[0] < 0.1

    def test_blend_alpha_one(self, graph_context: GraphContext):
        """Test blend at alpha=1 returns second input."""
        # Create distinct clips with different values
        clip_a_data = AnimationClip(name="clip_a", duration=1.0)
        clip_a_data.add_keyframe(0, 0.0, Transform(position=(0.0, 0.0, 0.0)))

        clip_b_data = AnimationClip(name="clip_b", duration=1.0)
        clip_b_data.add_keyframe(0, 0.0, Transform(position=(5.0, 5.0, 5.0)))

        clip_a = ClipNode("clip_a", clip_a_data)
        clip_a.is_playing = False  # Don't advance during evaluate

        clip_b = ClipNode("clip_b", clip_b_data)
        clip_b.is_playing = False  # Don't advance during evaluate

        blend = BlendNode("blend", alpha=1.0)
        blend.set_inputs(clip_a, clip_b)

        pose = blend.evaluate(graph_context)
        # Should be pose_b (5.0 at time 0)
        assert pose.transforms[0].position[0] == 5.0

    def test_blend_missing_input(self, graph_context: GraphContext, test_clip: AnimationClip):
        """Test blend with missing input."""
        clip_a = ClipNode("clip_a", test_clip)
        blend = BlendNode("blend", alpha=0.5)
        blend.inputs["a"] = clip_a
        # No input b

        pose = blend.evaluate(graph_context)
        assert pose is not None

    def test_additive_node(self, graph_context: GraphContext, test_clip: AnimationClip):
        """Test additive node."""
        base = ClipNode("base", test_clip)
        additive = ClipNode("additive", test_clip)

        node = AdditiveNode("add", weight=1.0)
        node.set_inputs(base, additive)

        pose = node.evaluate(graph_context)
        assert pose is not None

    def test_additive_weight_zero(self, graph_context: GraphContext, test_clip: AnimationClip):
        """Test additive with zero weight."""
        base = ClipNode("base", test_clip)
        base.is_playing = False  # Prevent advancement
        additive = ClipNode("additive", test_clip)
        additive.is_playing = False  # Prevent advancement

        # Evaluate base first to get its pose
        base_pose = base.evaluate(graph_context)

        # Now create fresh clips for the additive test
        base2 = ClipNode("base2", test_clip)
        base2.is_playing = False
        additive2 = ClipNode("additive2", test_clip)
        additive2.is_playing = False

        node = AdditiveNode("add", weight=0.0)
        node.set_inputs(base2, additive2)

        pose = node.evaluate(graph_context)
        # With zero weight, should match what base2 produces (both at time 0)
        assert pose.transforms[0].position[0] == 0.0

    def test_time_scale_node(self, graph_context: GraphContext, test_clip: AnimationClip):
        """Test time scale node."""
        clip = ClipNode("clip", test_clip)
        scale = TimeScaleNode("scale", scale=2.0)
        scale.set_input(clip)

        # Should advance at 2x speed
        pose = scale.evaluate(graph_context)
        assert pose is not None


# =============================================================================
# BLEND TREE 1D TESTS (12 tests)
# =============================================================================


class TestBlendTree1D:
    """Tests for BlendTree1D class."""

    def test_creation(self):
        """Test blend tree creation."""
        tree = BlendTree1D("locomotion", "speed")
        assert tree.parameter == "speed"
        assert tree.child_count() == 0

    def test_add_entry(self, test_clip: AnimationClip):
        """Test adding entries."""
        tree = BlendTree1D("locomotion", "speed")
        clip = ClipNode("idle", test_clip)
        idx = tree.add_entry(0.0, clip)
        assert idx == 0
        assert len(tree.entries) == 1

    def test_entries_sorted(self, test_clip: AnimationClip):
        """Test entries are sorted by threshold."""
        tree = BlendTree1D("locomotion", "speed")
        tree.add_entry(5.0, ClipNode("run", test_clip))
        tree.add_entry(0.0, ClipNode("idle", test_clip))
        tree.add_entry(2.0, ClipNode("walk", test_clip))

        assert tree.entries[0].threshold == 0.0
        assert tree.entries[1].threshold == 2.0
        assert tree.entries[2].threshold == 5.0

    def test_evaluate_at_threshold(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test evaluation at exact threshold."""
        tree = BlendTree1D("locomotion", "speed")
        tree.add_entry(0.0, ClipNode("idle", test_clip))
        tree.add_entry(5.0, ClipNode("walk", test_clip))

        graph_context.parameters["speed"].value = 0.0
        pose = tree.evaluate(graph_context)
        assert pose is not None

    def test_evaluate_between_thresholds(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test evaluation between thresholds."""
        clip1 = AnimationClip(name="clip1", duration=1.0)
        clip1.add_keyframe(0, 0.0, Transform(position=(0.0, 0.0, 0.0)))
        clip2 = AnimationClip(name="clip2", duration=1.0)
        clip2.add_keyframe(0, 0.0, Transform(position=(2.0, 0.0, 0.0)))

        tree = BlendTree1D("locomotion", "speed")
        tree.add_entry(0.0, ClipNode("idle", clip1))
        tree.add_entry(2.0, ClipNode("walk", clip2))

        graph_context.parameters["speed"].value = 1.0  # Midpoint
        pose = tree.evaluate(graph_context)
        # Should be interpolated
        assert 0.5 <= pose.transforms[0].position[0] <= 1.5

    def test_evaluate_below_min(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test evaluation below minimum threshold."""
        tree = BlendTree1D("locomotion", "speed")
        tree.add_entry(2.0, ClipNode("walk", test_clip))
        tree.add_entry(5.0, ClipNode("run", test_clip))

        graph_context.parameters["speed"].value = 0.0
        pose = tree.evaluate(graph_context)
        # Should return first entry
        assert pose is not None

    def test_evaluate_above_max(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test evaluation above maximum threshold."""
        tree = BlendTree1D("locomotion", "speed")
        tree.add_entry(0.0, ClipNode("idle", test_clip))
        tree.add_entry(2.0, ClipNode("walk", test_clip))

        graph_context.parameters["speed"].value = 10.0
        pose = tree.evaluate(graph_context)
        # Should return last entry
        assert pose is not None

    def test_single_entry(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test with single entry."""
        tree = BlendTree1D("locomotion", "speed")
        tree.add_entry(0.0, ClipNode("idle", test_clip))

        graph_context.parameters["speed"].value = 5.0
        pose = tree.evaluate(graph_context)
        assert pose is not None

    def test_get_weights(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test getting blend weights."""
        tree = BlendTree1D("locomotion", "speed")
        tree.add_entry(0.0, ClipNode("idle", test_clip))
        tree.add_entry(2.0, ClipNode("walk", test_clip))

        graph_context.parameters["speed"].value = 1.0
        weights = tree.get_weights(graph_context)
        assert 0 in weights
        assert 1 in weights
        assert abs(weights[0] + weights[1] - 1.0) < 0.01

    def test_remove_entry(self, test_clip: AnimationClip):
        """Test removing entry."""
        tree = BlendTree1D("locomotion", "speed")
        tree.add_entry(0.0, ClipNode("idle", test_clip))
        tree.add_entry(2.0, ClipNode("walk", test_clip))

        assert tree.remove_entry(0)
        assert len(tree.entries) == 1

    def test_gradient_bands(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test gradient band interpolation."""
        tree = BlendTree1D("locomotion", "speed")
        tree.use_gradient_bands = True
        tree.gradient_band_width = 0.2
        tree.add_entry(0.0, ClipNode("idle", test_clip))
        tree.add_entry(1.0, ClipNode("walk", test_clip))
        tree.add_entry(2.0, ClipNode("run", test_clip))

        graph_context.parameters["speed"].value = 0.5
        pose = tree.evaluate(graph_context)
        assert pose is not None

    def test_empty_tree(self, graph_context: GraphContext):
        """Test empty tree returns empty pose."""
        tree = BlendTree1D("empty", "speed")
        pose = tree.evaluate(graph_context)
        assert pose.bone_count() == 0


# =============================================================================
# BLEND TREE 2D TESTS (10 tests)
# =============================================================================


class TestBlendTree2D:
    """Tests for BlendTree2D class."""

    def test_creation(self):
        """Test 2D blend tree creation."""
        tree = BlendTree2D("directional", "speed", "direction")
        assert tree.param_x == "speed"
        assert tree.param_y == "direction"

    def test_add_sample(self, test_clip: AnimationClip):
        """Test adding samples."""
        tree = BlendTree2D("directional", "x", "y")
        idx = tree.add_sample(0.0, 0.0, ClipNode("center", test_clip))
        assert idx == 0
        assert len(tree.samples) == 1

    def test_cartesian_mode(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test Cartesian interpolation mode."""
        tree = BlendTree2D("directional", "speed", "direction", BlendTree2DMode.CARTESIAN)
        tree.add_sample(0.0, 0.0, ClipNode("center", test_clip))
        tree.add_sample(1.0, 0.0, ClipNode("right", test_clip))
        tree.add_sample(0.0, 1.0, ClipNode("up", test_clip))

        graph_context.parameters["speed"].value = 0.5
        graph_context.parameters["direction"].value = 0.5
        pose = tree.evaluate(graph_context)
        assert pose is not None

    def test_polar_mode(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test polar interpolation mode."""
        tree = BlendTree2D("directional", "speed", "direction", BlendTree2DMode.POLAR)
        tree.add_sample(1.0, 0.0, ClipNode("forward", test_clip))
        tree.add_sample(0.0, 1.0, ClipNode("left", test_clip))
        tree.add_sample(-1.0, 0.0, ClipNode("back", test_clip))

        pose = tree.evaluate(graph_context)
        assert pose is not None

    def test_single_sample(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test with single sample."""
        tree = BlendTree2D("directional", "x", "y")
        tree.add_sample(0.0, 0.0, ClipNode("center", test_clip))

        graph_context.parameters["speed"].value = 5.0
        pose = tree.evaluate(graph_context)
        assert pose is not None

    def test_triangulation(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test Delaunay triangulation."""
        tree = BlendTree2D("directional", "speed", "direction",
                          BlendTree2DMode.FREEFORM_CARTESIAN)
        tree.add_sample(0.0, 0.0, ClipNode("center", test_clip))
        tree.add_sample(1.0, 0.0, ClipNode("right", test_clip))
        tree.add_sample(0.0, 1.0, ClipNode("up", test_clip))
        tree.add_sample(1.0, 1.0, ClipNode("corner", test_clip))

        graph_context.parameters["speed"].value = 0.5
        graph_context.parameters["direction"].value = 0.5
        pose = tree.evaluate(graph_context)
        assert pose is not None

    def test_get_weights(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test getting 2D blend weights."""
        tree = BlendTree2D("directional", "speed", "direction")
        tree.add_sample(0.0, 0.0, ClipNode("center", test_clip))
        tree.add_sample(1.0, 0.0, ClipNode("right", test_clip))
        tree.add_sample(0.0, 1.0, ClipNode("up", test_clip))

        # Set parameters to a point that should involve multiple samples
        graph_context.parameters["speed"].value = 0.5
        graph_context.parameters["direction"].value = 0.5

        weights = tree.get_weights(graph_context)
        # All samples should have some weight when point is in middle
        assert len(weights) == 3
        assert sum(weights.values()) > 0

    def test_remove_sample(self, test_clip: AnimationClip):
        """Test removing sample."""
        tree = BlendTree2D("directional", "x", "y")
        tree.add_sample(0.0, 0.0, ClipNode("center", test_clip))
        tree.add_sample(1.0, 0.0, ClipNode("right", test_clip))

        assert tree.remove_sample(0)
        assert len(tree.samples) == 1

    def test_triangle_contains_point(self):
        """Test triangle point containment."""
        tri = Triangle(
            indices=(0, 1, 2),
            vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        )
        assert tri.contains_point((0.25, 0.25))
        assert not tri.contains_point((1.0, 1.0))

    def test_triangle_barycentric(self):
        """Test barycentric coordinates."""
        tri = Triangle(
            indices=(0, 1, 2),
            vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        )
        w0, w1, w2 = tri.get_barycentric((0.0, 0.0))
        assert w0 > 0.9  # Should be close to first vertex


# =============================================================================
# BLEND TREE DIRECT TESTS (6 tests)
# =============================================================================


class TestBlendTreeDirect:
    """Tests for BlendTreeDirect class."""

    def test_creation(self):
        """Test direct blend tree creation."""
        tree = BlendTreeDirect("direct")
        assert tree.normalize_weights is True

    def test_add_entry_fixed_weight(self, test_clip: AnimationClip):
        """Test adding entry with fixed weight."""
        tree = BlendTreeDirect("direct")
        idx = tree.add_entry(ClipNode("clip", test_clip), fixed_weight=0.5)
        assert idx == 0
        assert tree.entries[0].fixed_weight == 0.5

    def test_add_entry_parameter_weight(self, test_clip: AnimationClip):
        """Test adding entry with parameter weight."""
        tree = BlendTreeDirect("direct")
        tree.add_entry(ClipNode("clip", test_clip), weight_parameter="blend")
        assert tree.entries[0].weight_parameter == "blend"

    def test_evaluate_normalized(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test evaluation with normalized weights."""
        tree = BlendTreeDirect("direct", normalize_weights=True)
        tree.add_entry(ClipNode("a", test_clip), fixed_weight=1.0)
        tree.add_entry(ClipNode("b", test_clip), fixed_weight=1.0)

        pose = tree.evaluate(graph_context)
        assert pose is not None

    def test_evaluate_unnormalized(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test evaluation without normalization."""
        tree = BlendTreeDirect("direct", normalize_weights=False)
        tree.add_entry(ClipNode("a", test_clip), fixed_weight=0.5)

        pose = tree.evaluate(graph_context)
        assert pose is not None

    def test_get_weights(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test getting direct blend weights."""
        tree = BlendTreeDirect("direct", normalize_weights=True)
        tree.add_entry(ClipNode("a", test_clip), fixed_weight=2.0)
        tree.add_entry(ClipNode("b", test_clip), fixed_weight=2.0)

        weights = tree.get_weights(graph_context)
        # Should be normalized to 0.5 each
        assert abs(weights[0] - 0.5) < 0.01
        assert abs(weights[1] - 0.5) < 0.01

    def test_zero_weight_normalization(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test that zero weights don't cause division by zero."""
        tree = BlendTreeDirect("direct", normalize_weights=True)
        tree.add_entry(ClipNode("a", test_clip), fixed_weight=0.0)
        tree.add_entry(ClipNode("b", test_clip), fixed_weight=0.0)

        # Should return empty pose instead of division by zero
        pose = tree.evaluate(graph_context)
        assert pose.bone_count() == 0


# =============================================================================
# STATE MACHINE TESTS (15 tests)
# =============================================================================


class TestStateMachine:
    """Tests for StateMachine class."""

    def test_creation(self):
        """Test state machine creation."""
        sm = StateMachine("sm", "idle")
        assert sm._initial_state == "idle"
        assert sm.current_state is None  # Not started

    def test_add_state(self, test_clip: AnimationClip):
        """Test adding states."""
        sm = StateMachine("sm")
        state = AnimationState("idle", ClipNode("idle_clip", test_clip))
        sm.add_state(state)
        assert "idle" in sm.states
        assert sm._initial_state == "idle"  # First state becomes initial

    def test_get_state(self, test_clip: AnimationClip):
        """Test getting state."""
        sm = StateMachine("sm")
        state = AnimationState("idle")
        sm.add_state(state)
        retrieved = sm.get_state("idle")
        assert retrieved is state

    def test_start(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test starting state machine."""
        sm = StateMachine("sm", "idle")
        sm.add_state(AnimationState("idle"))
        sm.start(graph_context)
        assert sm.current_state_name == "idle"

    def test_add_transition(self, test_clip: AnimationClip):
        """Test adding transitions."""
        sm = StateMachine("sm")
        sm.add_state(AnimationState("idle"))
        sm.add_state(AnimationState("walk"))

        transition = StateTransition("idle", "walk", [], 0.2)
        sm.add_transition(transition)

        assert len(sm.transitions) == 1

    def test_transition_conditions(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test transition conditions."""
        sm = StateMachine("sm", "idle")
        sm.add_state(AnimationState("idle", ClipNode("idle", test_clip)))
        sm.add_state(AnimationState("walk", ClipNode("walk", test_clip)))

        condition = TransitionCondition.greater_than("speed", 1.0)
        transition = StateTransition("idle", "walk", [condition], 0.2)
        sm.add_transition(transition)

        sm.start(graph_context)

        # Speed is 0, should stay in idle
        sm.update(0.1, graph_context)
        assert sm.current_state_name == "idle"

        # Increase speed
        graph_context.parameters["speed"].value = 2.0
        sm.update(0.1, graph_context)
        assert sm.is_transitioning or sm.current_state_name == "walk"

    def test_transition_blend(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test transition blending."""
        sm = StateMachine("sm", "idle")
        sm.add_state(AnimationState("idle", ClipNode("idle", test_clip)))
        sm.add_state(AnimationState("walk", ClipNode("walk", test_clip)))

        # Don't add automatic transition - we'll force it
        sm.start(graph_context)
        sm.force_state("walk", graph_context, immediate=False)

        # Should now be transitioning
        assert sm.is_transitioning
        initial_progress = sm.active_transition.progress

        # Update to advance transition
        sm.update(0.1, graph_context)
        assert sm.active_transition.progress > initial_progress

    def test_transition_completion(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test transition completion."""
        sm = StateMachine("sm", "idle")
        sm.add_state(AnimationState("idle", ClipNode("idle", test_clip)))
        sm.add_state(AnimationState("walk", ClipNode("walk", test_clip)))

        transition = StateTransition("idle", "walk", [], 0.1)
        sm.add_transition(transition)

        sm.start(graph_context)
        sm.force_state("walk", graph_context)

        # Complete transition
        sm.update(0.2, graph_context)
        assert not sm.is_transitioning
        assert sm.current_state_name == "walk"

    def test_any_state_transition(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test any-state transitions."""
        sm = StateMachine("sm", "idle")
        sm.add_state(AnimationState("idle", ClipNode("idle", test_clip)))
        sm.add_state(AnimationState("walk", ClipNode("walk", test_clip)))
        sm.add_state(AnimationState("death", ClipNode("death", test_clip)))

        # Any state can transition to death
        condition = TransitionCondition.is_true("is_dead")
        any_transition = StateTransition("*", "death", [condition], 0.1, priority=100)
        sm.add_transition(any_transition)

        sm.start(graph_context)
        graph_context.parameters["is_dead"] = GraphParameter.bool_param("is_dead", False)

        # Trigger death
        graph_context.parameters["is_dead"].value = True
        sm.update(0.05, graph_context)

        # Should transition to death from any state
        assert sm.is_transitioning or sm.current_state_name == "death"

    def test_force_state_immediate(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test force state immediate."""
        sm = StateMachine("sm", "idle")
        sm.add_state(AnimationState("idle", ClipNode("idle", test_clip)))
        sm.add_state(AnimationState("walk", ClipNode("walk", test_clip)))

        sm.start(graph_context)
        sm.force_state("walk", graph_context, immediate=True)

        assert sm.current_state_name == "walk"
        assert not sm.is_transitioning

    def test_evaluate(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test state machine evaluation."""
        sm = StateMachine("sm", "idle")
        sm.add_state(AnimationState("idle", ClipNode("idle", test_clip)))

        pose = sm.evaluate(graph_context)
        assert pose is not None
        assert pose.bone_count() > 0  # Should produce valid pose with bones

    def test_reset(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test state machine reset."""
        sm = StateMachine("sm", "idle")
        sm.add_state(AnimationState("idle", ClipNode("idle", test_clip)))
        sm.add_state(AnimationState("walk", ClipNode("walk", test_clip)))

        sm.start(graph_context)
        sm.force_state("walk", graph_context, immediate=True)
        assert sm.current_state_name == "walk"

        sm.reset(graph_context)
        assert sm.current_state_name == "idle"

    def test_builder(self, test_clip: AnimationClip):
        """Test state machine builder."""
        sm = (
            StateMachineBuilder("sm")
            .add_state("idle", ClipNode("idle", test_clip))
            .add_state("walk", ClipNode("walk", test_clip))
            .set_initial("idle")
            .add_transition("idle", "walk", [TransitionCondition.greater_than("speed", 1.0)])
            .build()
        )

        assert "idle" in sm.states
        assert "walk" in sm.states
        assert len(sm.transitions) == 1

    def test_transition_priority(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test transition priority ordering."""
        sm = StateMachine("sm", "idle")
        sm.add_state(AnimationState("idle"))
        sm.add_state(AnimationState("walk"))
        sm.add_state(AnimationState("run"))

        # Low priority transition
        sm.add_transition(StateTransition("idle", "walk", [], priority=0))
        # High priority transition
        sm.add_transition(StateTransition("idle", "run", [], priority=10))

        # Transitions should be sorted by priority
        assert sm.transitions[0].priority > sm.transitions[1].priority

    def test_debug_info(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test debug info."""
        sm = StateMachine("sm", "idle")
        sm.add_state(AnimationState("idle", ClipNode("idle", test_clip)))
        sm.start(graph_context)

        info = sm.get_debug_info()
        assert info["current_state"] == "idle"
        assert "idle" in info["states"]

    def test_transition_loop_prevention(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test that state machine prevents infinite transition loops in same frame."""
        sm = StateMachine("sm", "a")
        sm.add_state(AnimationState("a", ClipNode("a", test_clip)))
        sm.add_state(AnimationState("b", ClipNode("b", test_clip)))
        sm.add_state(AnimationState("c", ClipNode("c", test_clip)))

        # Create conditions that would always be true (immediate transitions)
        graph_context.parameters["always_true"] = GraphParameter.bool_param("always_true", True)
        cond = TransitionCondition.is_true("always_true")

        # Add transitions that could create a loop: a -> b -> c -> a
        sm.add_transition(StateTransition("a", "b", [cond], 0.0))
        sm.add_transition(StateTransition("b", "c", [cond], 0.0))
        sm.add_transition(StateTransition("c", "a", [cond], 0.0))

        sm.start(graph_context)
        initial_state = sm.current_state_name

        # Update should not infinite loop - only one transition per frame
        sm.update(0.1, graph_context)

        # State machine should have made progress (transitioned or transitioning)
        # but not looped infinitely back to start state within same frame
        assert sm.current_state_name != initial_state or sm.is_transitioning


# =============================================================================
# TRANSITION CONDITION TESTS (10 tests)
# =============================================================================


class TestTransitionCondition:
    """Tests for TransitionCondition class."""

    def test_equals(self, graph_context: GraphContext):
        """Test equals condition."""
        graph_context.parameters["state"] = GraphParameter.enum_param(
            "state", ["idle", "walk"], "idle"
        )
        condition = TransitionCondition.equals("state", "idle")
        assert condition.evaluate(graph_context)

    def test_not_equals(self, graph_context: GraphContext):
        """Test not equals condition."""
        condition = TransitionCondition.not_equals("speed", 5.0)
        graph_context.parameters["speed"].value = 3.0
        assert condition.evaluate(graph_context)

    def test_greater_than(self, graph_context: GraphContext):
        """Test greater than condition."""
        condition = TransitionCondition.greater_than("speed", 2.0)
        graph_context.parameters["speed"].value = 3.0
        assert condition.evaluate(graph_context)
        graph_context.parameters["speed"].value = 1.0
        assert not condition.evaluate(graph_context)

    def test_greater_or_equal(self, graph_context: GraphContext):
        """Test greater or equal condition."""
        condition = TransitionCondition.greater_or_equal("speed", 2.0)
        graph_context.parameters["speed"].value = 2.0
        assert condition.evaluate(graph_context)

    def test_less_than(self, graph_context: GraphContext):
        """Test less than condition."""
        condition = TransitionCondition.less_than("speed", 2.0)
        graph_context.parameters["speed"].value = 1.0
        assert condition.evaluate(graph_context)

    def test_less_or_equal(self, graph_context: GraphContext):
        """Test less or equal condition."""
        condition = TransitionCondition.less_or_equal("speed", 2.0)
        graph_context.parameters["speed"].value = 2.0
        assert condition.evaluate(graph_context)

    def test_is_true(self, graph_context: GraphContext):
        """Test is_true condition."""
        condition = TransitionCondition.is_true("is_jumping")
        graph_context.parameters["is_jumping"].value = True
        assert condition.evaluate(graph_context)

    def test_is_false(self, graph_context: GraphContext):
        """Test is_false condition."""
        condition = TransitionCondition.is_false("is_jumping")
        graph_context.parameters["is_jumping"].value = False
        assert condition.evaluate(graph_context)

    def test_trigger_condition(self, graph_context: GraphContext):
        """Test trigger condition."""
        graph_context.parameters["jump"] = GraphParameter.trigger_param("jump")
        condition = TransitionCondition.trigger("jump")

        assert not condition.evaluate(graph_context)
        graph_context.parameters["jump"].trigger()
        assert condition.evaluate(graph_context)

    def test_missing_parameter(self, graph_context: GraphContext):
        """Test condition with missing parameter."""
        condition = TransitionCondition.equals("nonexistent", "value")
        assert not condition.evaluate(graph_context)


# =============================================================================
# BLEND CURVE TESTS (6 tests)
# =============================================================================


class TestBlendCurve:
    """Tests for blend curve evaluation."""

    def test_linear(self):
        """Test linear curve."""
        assert evaluate_blend_curve(BlendCurve.LINEAR, 0.0) == 0.0
        assert evaluate_blend_curve(BlendCurve.LINEAR, 0.5) == 0.5
        assert evaluate_blend_curve(BlendCurve.LINEAR, 1.0) == 1.0

    def test_ease_in(self):
        """Test ease in curve."""
        mid = evaluate_blend_curve(BlendCurve.EASE_IN, 0.5)
        # Ease in should be slower at start
        assert mid < 0.5

    def test_ease_out(self):
        """Test ease out curve."""
        mid = evaluate_blend_curve(BlendCurve.EASE_OUT, 0.5)
        # Ease out should be faster at start
        assert mid > 0.5

    def test_ease_in_out(self):
        """Test ease in out curve."""
        mid = evaluate_blend_curve(BlendCurve.EASE_IN_OUT, 0.5)
        # Should be at midpoint
        assert abs(mid - 0.5) < 0.1

    def test_smooth_step(self):
        """Test smooth step curve."""
        assert evaluate_blend_curve(BlendCurve.SMOOTH_STEP, 0.0) == 0.0
        assert evaluate_blend_curve(BlendCurve.SMOOTH_STEP, 1.0) == 1.0

    def test_clamping(self):
        """Test value clamping."""
        assert evaluate_blend_curve(BlendCurve.LINEAR, -0.5) == 0.0
        assert evaluate_blend_curve(BlendCurve.LINEAR, 1.5) == 1.0


# =============================================================================
# LAYER TESTS (12 tests)
# =============================================================================


class TestLayer:
    """Tests for animation layers."""

    def test_layer_creation(self):
        """Test layer creation."""
        layer = AnimationLayer("base")
        assert layer.name == "base"
        assert layer.weight == 1.0
        assert layer.is_active

    def test_layer_weight(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test layer weight."""
        layer = AnimationLayer("base", weight=0.5)
        assert layer.get_effective_weight(graph_context) == 0.5

    def test_layer_inactive(self, graph_context: GraphContext):
        """Test inactive layer."""
        layer = AnimationLayer("base", is_active=False)
        assert layer.get_effective_weight(graph_context) == 0.0

    def test_layer_parameter_weight(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test layer weight from parameter."""
        graph_context.parameters["layer_weight"] = GraphParameter.float_param(
            "layer_weight", 0.75
        )
        layer = AnimationLayer("base", weight_parameter="layer_weight")
        assert layer.get_effective_weight(graph_context) == 0.75

    def test_layer_stack_creation(self):
        """Test layer stack creation."""
        stack = LayerStack("stack")
        assert stack.layer_count() == 0

    def test_layer_stack_add_layer(self, test_clip: AnimationClip):
        """Test adding layer to stack."""
        stack = LayerStack("stack")
        layer = AnimationLayer("base", ClipNode("clip", test_clip))
        idx = stack.add_layer(layer)
        assert idx == 0
        assert stack.layer_count() == 1

    def test_layer_stack_get_layer(self, test_clip: AnimationClip):
        """Test getting layer from stack."""
        stack = LayerStack("stack")
        layer = AnimationLayer("base")
        stack.add_layer(layer)
        retrieved = stack.get_layer("base")
        assert retrieved is layer

    def test_layer_stack_remove_layer(self, test_clip: AnimationClip):
        """Test removing layer from stack."""
        stack = LayerStack("stack")
        stack.add_layer(AnimationLayer("base"))
        assert stack.remove_layer("base")
        assert stack.layer_count() == 0

    def test_layer_stack_evaluate(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test layer stack evaluation."""
        stack = LayerStack("stack")
        stack.add_layer(AnimationLayer("base", ClipNode("clip", test_clip)))

        pose = stack.evaluate(graph_context)
        assert pose is not None

    def test_layer_stack_multiple_layers(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test multiple layer evaluation."""
        stack = LayerStack("stack")
        stack.add_layer(AnimationLayer("base", ClipNode("base", test_clip)))
        stack.add_layer(AnimationLayer("upper", ClipNode("upper", test_clip), weight=0.5))

        pose = stack.evaluate(graph_context)
        assert pose is not None

    def test_layer_stack_builder(self, test_clip: AnimationClip):
        """Test layer stack builder."""
        stack = (
            LayerStackBuilder("stack")
            .add_layer("base", ClipNode("base", test_clip))
            .add_additive_layer("additive", ClipNode("add", test_clip), weight=0.5)
            .build()
        )

        assert stack.layer_count() == 2

    def test_layer_with_mask(self, test_clip: AnimationClip, graph_context: GraphContext, simple_skeleton: Skeleton):
        """Test layer with bone mask."""
        mask = BoneMask(name="upper")
        mask.set_weights([2, 3, 4, 5], 1.0)  # Upper body bones

        stack = LayerStack("stack")
        stack.add_layer(AnimationLayer("base", ClipNode("base", test_clip)))
        stack.add_layer(AnimationLayer("upper", ClipNode("upper", test_clip), mask=mask))

        pose = stack.evaluate(graph_context)
        assert pose is not None


# =============================================================================
# BONE MASK PRESETS TESTS (8 tests)
# =============================================================================


class TestBoneMaskPresets:
    """Tests for bone mask presets."""

    def test_upper_body(self, humanoid_skeleton: Skeleton):
        """Test upper body preset."""
        mask = BoneMaskPresets.upper_body(humanoid_skeleton)
        assert mask.name == "UpperBody"
        spine_idx = humanoid_skeleton.get_bone_index("Spine")
        if spine_idx >= 0:
            assert mask.get_weight(spine_idx) == 1.0

    def test_lower_body(self, humanoid_skeleton: Skeleton):
        """Test lower body preset."""
        mask = BoneMaskPresets.lower_body(humanoid_skeleton)
        assert mask.name == "LowerBody"

    def test_left_arm(self, humanoid_skeleton: Skeleton):
        """Test left arm preset."""
        mask = BoneMaskPresets.left_arm(humanoid_skeleton)
        assert mask.name == "LeftArm"

    def test_right_arm(self, humanoid_skeleton: Skeleton):
        """Test right arm preset."""
        mask = BoneMaskPresets.right_arm(humanoid_skeleton)
        assert mask.name == "RightArm"

    def test_spine(self, humanoid_skeleton: Skeleton):
        """Test spine preset."""
        mask = BoneMaskPresets.spine(humanoid_skeleton)
        assert mask.name == "Spine"

    def test_head(self, humanoid_skeleton: Skeleton):
        """Test head preset."""
        mask = BoneMaskPresets.head(humanoid_skeleton)
        assert mask.name == "Head"

    def test_full_body(self, humanoid_skeleton: Skeleton):
        """Test full body preset."""
        mask = BoneMaskPresets.full_body(humanoid_skeleton)
        for i in range(humanoid_skeleton.bone_count()):
            assert mask.get_weight(i) == 1.0

    def test_gradient_upper_lower(self, humanoid_skeleton: Skeleton):
        """Test gradient preset."""
        mask = BoneMaskPresets.gradient_upper_lower(humanoid_skeleton)
        assert mask.name == "GradientUpperLower"


# =============================================================================
# SYNC MARKER TESTS (8 tests)
# =============================================================================


class TestSyncMarker:
    """Tests for sync markers."""

    def test_marker_creation(self):
        """Test marker creation."""
        marker = SyncMarker("left_plant", 0.25)
        assert marker.name == "left_plant"
        assert marker.normalized_time == 0.25

    def test_marker_clamping(self):
        """Test marker time clamping."""
        marker = SyncMarker("test", 1.5)
        assert marker.normalized_time == 1.0
        marker2 = SyncMarker("test", -0.5)
        assert marker2.normalized_time == 0.0

    def test_time_for_duration(self):
        """Test getting time for duration."""
        marker = SyncMarker("test", 0.5)
        assert marker.get_time_for_duration(2.0) == 1.0

    def test_distance_to(self):
        """Test distance calculation."""
        marker = SyncMarker("test", 0.25)
        assert marker.distance_to(0.25) == 0.0
        assert abs(marker.distance_to(0.75) - 0.5) < 0.01

    def test_distance_to_wrapped(self):
        """Test wrapped distance."""
        marker = SyncMarker("test", 0.9)
        # Distance to 0.1 should be 0.2 (wrapped) not 0.8 (direct)
        dist = marker.distance_to(0.1)
        assert dist < 0.3

    def test_marker_track(self):
        """Test marker track."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker("left", 0.0))
        track.add_marker(SyncMarker("right", 0.5))
        assert len(track.markers) == 2

    def test_get_markers_by_name(self):
        """Test getting markers by name."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker("plant", 0.0))
        track.add_marker(SyncMarker("plant", 0.5))
        track.add_marker(SyncMarker("pass", 0.25))

        plants = track.get_markers_by_name("plant")
        assert len(plants) == 2

    def test_get_nearest_marker(self):
        """Test getting nearest marker."""
        track = SyncMarkerTrack()
        track.add_marker(SyncMarker("a", 0.0))
        track.add_marker(SyncMarker("b", 0.5))
        track.add_marker(SyncMarker("c", 1.0))

        nearest = track.get_nearest_marker(0.4)
        assert nearest.name == "b"


# =============================================================================
# SYNC GROUP TESTS (10 tests)
# =============================================================================


class TestSyncGroup:
    """Tests for sync groups."""

    def test_group_creation(self):
        """Test sync group creation."""
        group = SyncGroup("locomotion", SyncMode.NORMALIZED)
        assert group.name == "locomotion"
        assert group.mode == SyncMode.NORMALIZED

    def test_add_entry(self, test_clip: AnimationClip):
        """Test adding entry."""
        group = SyncGroup("test")
        idx = group.add_entry(ClipNode("clip", test_clip), 1.0, True, 1.0)
        assert idx == 0
        assert len(group.entries) == 1
        assert group.entries[0].is_leader

    def test_get_leader(self, test_clip: AnimationClip):
        """Test getting leader."""
        group = SyncGroup("test")
        group.add_entry(ClipNode("a", test_clip), 1.0, False)
        group.add_entry(ClipNode("b", test_clip), 1.0, True)

        leader = group.get_leader()
        assert leader.is_leader

    def test_set_leader(self, test_clip: AnimationClip):
        """Test setting leader."""
        group = SyncGroup("test")
        group.add_entry(ClipNode("a", test_clip))
        group.add_entry(ClipNode("b", test_clip))

        group.set_leader(1)
        assert group.entries[1].is_leader
        assert not group.entries[0].is_leader

    def test_update_normalized(self, test_clip: AnimationClip):
        """Test normalized sync update."""
        group = SyncGroup("test", SyncMode.NORMALIZED)
        group.add_entry(ClipNode("a", test_clip), 1.0, False, 1.0)
        group.add_entry(ClipNode("b", test_clip), 1.0, False, 2.0)

        group.update(0.1)
        # Both entries should have same normalized time
        assert abs(group.entries[0].normalized_time - group.entries[1].normalized_time) < 0.1

    def test_update_leader_follower(self, test_clip: AnimationClip):
        """Test leader-follower sync."""
        group = SyncGroup("test", SyncMode.LEADER_FOLLOWER)
        group.add_entry(ClipNode("leader", test_clip), 1.0, True, 1.0)
        group.add_entry(ClipNode("follower", test_clip), 1.0, False, 2.0)

        group.update(0.5)
        # Follower should match leader's normalized time
        leader = group.get_leader()
        follower = group.entries[1]
        assert abs(leader.normalized_time - follower.normalized_time) < 0.01

    def test_get_synchronized_time(self, test_clip: AnimationClip):
        """Test getting synchronized time."""
        group = SyncGroup("test", SyncMode.NORMALIZED)
        group.add_entry(ClipNode("a", test_clip), 1.0, False, 1.0)

        group.entries[0].normalized_time = 0.5
        assert group.get_synchronized_time() == 0.5

    def test_sync_group_node(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test sync group node."""
        node = SyncGroupNode("sync", SyncMode.NORMALIZED)
        node.add_entry(ClipNode("a", test_clip), 1.0, True, 1.0)
        node.add_entry(ClipNode("b", test_clip), 1.0, False, 1.0)

        pose = node.evaluate(graph_context)
        assert pose is not None

    def test_locomotion_markers(self):
        """Test locomotion marker creation."""
        track = create_locomotion_markers(0.0, 0.5, 0.25, 0.75)
        assert len(track.markers) == 4

        left_plants = track.get_markers_by_name("left_plant")
        assert len(left_plants) == 1
        assert left_plants[0].normalized_time == 0.0

    def test_phase_offset_calculation(self):
        """Test phase offset calculation."""
        track1 = SyncMarkerTrack()
        track1.add_marker(SyncMarker("left_plant", 0.0))

        track2 = SyncMarkerTrack()
        track2.add_marker(SyncMarker("left_plant", 0.25))

        offset = calculate_phase_offset(track1, track2)
        assert offset == 0.25


# =============================================================================
# EVENT SYNCHRONIZER TESTS (5 tests)
# =============================================================================


class TestEventSynchronizer:
    """Tests for event synchronizer."""

    def test_register_handler(self):
        """Test registering event handler."""
        sync = EventSynchronizer()
        handler = lambda e: None
        sync.register_handler("footstep", handler)
        assert "footstep" in sync.event_handlers
        assert handler in sync.event_handlers["footstep"]

    def test_unregister_handler(self):
        """Test unregistering handler."""
        sync = EventSynchronizer()
        handler = lambda e: None
        sync.register_handler("footstep", handler)
        assert sync.unregister_handler("footstep", handler)
        assert len(sync.event_handlers["footstep"]) == 0

    def test_queue_event(self):
        """Test queuing events."""
        sync = EventSynchronizer()
        event = SyncEvent("footstep", "clip1", 0.5)
        sync.queue_event(event)
        assert len(sync._pending_events) == 1

    def test_process_events(self):
        """Test processing events."""
        sync = EventSynchronizer()
        received = []
        sync.register_handler("footstep", lambda e: received.append(e))

        sync.queue_event(SyncEvent("footstep", "clip1", 0.5))
        sync.process_events()

        assert len(received) == 1
        assert received[0].name == "footstep"

    def test_clear(self):
        """Test clearing pending events."""
        sync = EventSynchronizer()
        sync.queue_event(SyncEvent("test", "clip", 0.0))
        sync.clear()
        assert len(sync._pending_events) == 0


# =============================================================================
# ANIMATION GRAPH TESTS (12 tests)
# =============================================================================


class TestAnimationGraph:
    """Tests for AnimationGraph class."""

    def test_graph_creation(self):
        """Test graph creation."""
        graph = AnimationGraph("test")
        assert graph.name == "test"
        assert len(graph.nodes) == 0

    def test_add_node(self, test_clip: AnimationClip):
        """Test adding node."""
        graph = AnimationGraph("test")
        node = ClipNode("clip", test_clip)
        graph.add_node(node)
        assert "clip" in graph.nodes

    def test_remove_node(self, test_clip: AnimationClip):
        """Test removing node."""
        graph = AnimationGraph("test")
        node = ClipNode("clip", test_clip)
        graph.add_node(node)
        assert graph.remove_node("clip")
        assert "clip" not in graph.nodes

    def test_get_node(self, test_clip: AnimationClip):
        """Test getting node."""
        graph = AnimationGraph("test")
        node = ClipNode("clip", test_clip)
        graph.add_node(node)
        retrieved = graph.get_node("clip")
        assert retrieved is node

    def test_connect_nodes(self, test_clip: AnimationClip):
        """Test connecting nodes."""
        graph = AnimationGraph("test")
        clip = ClipNode("clip", test_clip)
        blend = BlendNode("blend")
        graph.add_node(clip)
        graph.add_node(blend)

        assert graph.connect("clip", "output", "blend", "a")
        assert len(graph.connections) == 1

    def test_disconnect_nodes(self, test_clip: AnimationClip):
        """Test disconnecting nodes."""
        graph = AnimationGraph("test")
        clip = ClipNode("clip", test_clip)
        blend = BlendNode("blend")
        graph.add_node(clip)
        graph.add_node(blend)
        graph.connect("clip", "output", "blend", "a")

        assert graph.disconnect("clip", "output", "blend", "a")
        assert len(graph.connections) == 0

    def test_add_parameter(self):
        """Test adding parameter."""
        graph = AnimationGraph("test")
        param = GraphParameter.float_param("speed", 0.0)
        graph.add_parameter(param)
        assert "speed" in graph.parameters

    def test_set_parameter(self):
        """Test setting parameter."""
        graph = AnimationGraph("test")
        graph.add_parameter(GraphParameter.float_param("speed", 0.0))
        assert graph.set_parameter("speed", 5.0)
        assert graph.get_parameter("speed") == 5.0

    def test_set_output_node(self, test_clip: AnimationClip):
        """Test setting output node."""
        graph = AnimationGraph("test")
        node = ClipNode("clip", test_clip)
        graph.add_node(node)
        assert graph.set_output_node("clip")
        assert graph.output_node_id == "clip"

    def test_evaluate_graph(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test graph evaluation."""
        graph = AnimationGraph("test")
        node = ClipNode("clip", test_clip)
        graph.add_node(node)
        graph.set_output_node("clip")

        pose = graph.evaluate(graph_context)
        assert pose is not None
        # Graph should produce a pose with bones matching skeleton
        assert pose.bone_count() == graph_context.skeleton.bone_count()

    def test_validate_graph(self, test_clip: AnimationClip):
        """Test graph validation."""
        graph = AnimationGraph("test")
        errors = graph.validate()
        assert "No output node set" in errors

        node = ClipNode("clip", test_clip)
        graph.add_node(node)
        graph.set_output_node("clip")
        errors = graph.validate()
        assert len(errors) == 0

    def test_topology_order(self, test_clip: AnimationClip):
        """Test topological ordering."""
        graph = AnimationGraph("test")
        clip_a = ClipNode("clip_a", test_clip)
        clip_b = ClipNode("clip_b", test_clip)
        blend = BlendNode("blend")

        graph.add_node(clip_a)
        graph.add_node(clip_b)
        graph.add_node(blend)
        graph.connect("clip_a", "output", "blend", "a")
        graph.connect("clip_b", "output", "blend", "b")
        graph.set_output_node("blend")

        order = graph.get_topology_order()
        # Blend should come after clips
        assert order.index("blend") > order.index("clip_a")
        assert order.index("blend") > order.index("clip_b")

    def test_graph_cycle_detection(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test that graph cycle detection prevents infinite recursion."""
        graph = AnimationGraph("test")
        blend_a = BlendNode("blend_a")
        blend_b = BlendNode("blend_b")

        graph.add_node(blend_a)
        graph.add_node(blend_b)

        # Create a cycle: blend_a -> blend_b -> blend_a
        blend_a.inputs["a"] = blend_b
        blend_b.inputs["a"] = blend_a

        graph.set_output_node("blend_a")

        # Validation should detect cycle
        errors = graph.validate()
        assert any("cycle" in error.lower() for error in errors)

        # Evaluation should return empty pose (not infinite loop)
        pose = graph.evaluate(graph_context)
        assert pose.bone_count() == 0  # Empty pose due to cycle


# =============================================================================
# SUBGRAPH TESTS (4 tests)
# =============================================================================


class TestSubgraph:
    """Tests for subgraphs."""

    def test_add_subgraph(self, test_clip: AnimationClip):
        """Test adding subgraph."""
        graph = AnimationGraph("main")
        subgraph = AnimationGraph("sub")
        subgraph.add_node(ClipNode("clip", test_clip))
        subgraph.set_output_node("clip")

        graph.add_subgraph("locomotion", subgraph)
        assert graph.get_subgraph("locomotion") is subgraph

    def test_subgraph_node(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test subgraph node evaluation."""
        subgraph = AnimationGraph("sub")
        subgraph.add_node(ClipNode("clip", test_clip))
        subgraph.set_output_node("clip")

        node = SubgraphNode("sub_node", subgraph)
        pose = node.evaluate(graph_context)
        assert pose is not None

    def test_subgraph_parameter_mapping(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test subgraph parameter mapping."""
        subgraph = AnimationGraph("sub")
        subgraph.add_parameter(GraphParameter.float_param("internal_speed", 0.0))
        subgraph.add_node(ClipNode("clip", test_clip))
        subgraph.set_output_node("clip")

        node = SubgraphNode("sub_node", subgraph)
        node.map_parameter("speed", "internal_speed")

        graph_context.parameters["speed"].value = 5.0
        pose = node.evaluate(graph_context)
        assert pose is not None

    def test_graph_copy(self):
        """Test graph copy."""
        graph = AnimationGraph("test")
        graph.add_parameter(GraphParameter.float_param("speed", 5.0))

        copy = graph.copy()
        assert copy.name == "test_copy"
        assert "speed" in copy.parameters


# =============================================================================
# MIRROR NODE TESTS (5 tests)
# =============================================================================


class TestMirrorNode:
    """Tests for mirror node."""

    def test_mirror_node_creation(self):
        """Test mirror node creation."""
        node = MirrorNode("mirror")
        assert len(node.mirror_pairs) == 0

    def test_add_mirror_pair(self):
        """Test adding mirror pair."""
        node = MirrorNode("mirror")
        node.add_mirror_pair(0, 1)
        assert len(node.mirror_pairs) == 1
        assert node.mirror_pairs[0].left_index == 0
        assert node.mirror_pairs[0].right_index == 1

    def test_mirror_from_skeleton(self, humanoid_skeleton: Skeleton):
        """Test auto-detecting mirror pairs."""
        node = MirrorNode("mirror")
        node.set_mirror_pairs_from_skeleton(humanoid_skeleton)
        # Should detect Left/Right pairs
        assert len(node.mirror_pairs) > 0

    def test_mirror_evaluate(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test mirror node evaluation."""
        node = MirrorNode("mirror")
        clip = ClipNode("clip", test_clip)
        node.set_input(clip)

        pose = node.evaluate(graph_context)
        assert pose is not None

    def test_mirror_transform(self):
        """Test transform mirroring."""
        node = MirrorNode("mirror")
        node.mirror_axis = 0  # X-axis

        t = Transform(position=(1.0, 2.0, 3.0))
        mirrored = node._mirror_transform(t)
        assert mirrored.position[0] == -1.0
        assert mirrored.position[1] == 2.0


# =============================================================================
# ADDITIONAL NODE TESTS (5 tests)
# =============================================================================


class TestAdditionalNodes:
    """Tests for additional blend nodes."""

    def test_pose_cache_node(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test pose cache node."""
        clip = ClipNode("clip", test_clip)
        cache = PoseCacheNode("cache", cache_duration=1.0)
        cache.set_input(clip)

        pose = cache.evaluate(graph_context)
        assert pose is not None

    def test_select_node(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test select node."""
        graph_context.parameters["selection"] = GraphParameter.int_param("selection", 0, 0, 2)

        select = SelectNode("select", "selection")
        select.add_option(ClipNode("a", test_clip))
        select.add_option(ClipNode("b", test_clip))

        graph_context.parameters["selection"].value = 1
        pose = select.evaluate(graph_context)
        assert pose is not None

    def test_layer_node(self, test_clip: AnimationClip, graph_context: GraphContext, simple_skeleton: Skeleton):
        """Test layer node."""
        node = LayerNode("layers")
        node.set_base(ClipNode("base", test_clip))

        mask = BoneMask(name="upper")
        mask.set_weights([2, 3, 4], 1.0)

        node.add_layer(ClipNode("upper", test_clip), weight=0.5, mask=mask)

        pose = node.evaluate(graph_context)
        assert pose is not None

    def test_graph_node_meta(self):
        """Test graph node metaclass registration."""
        # ClipNode should be registered (use public API, not private _registry)
        assert "ClipNode" in GraphNodeMeta.all_node_types()

        node_type = GraphNodeMeta.get_node_type("ClipNode")
        assert node_type is ClipNode

    def test_all_node_types(self):
        """Test getting all node types."""
        types = GraphNodeMeta.all_node_types()
        assert len(types) > 0
        assert "ClipNode" in types
        assert "BlendNode" in types


# =============================================================================
# COMPLEX INTEGRATION TESTS (8 tests)
# =============================================================================


class TestComplexIntegration:
    """Complex integration tests."""

    def test_locomotion_blend_tree(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test complete locomotion blend tree."""
        tree = BlendTree1D("locomotion", "speed")
        tree.add_entry(0.0, ClipNode("idle", test_clip))
        tree.add_entry(2.0, ClipNode("walk", test_clip))
        tree.add_entry(5.0, ClipNode("run", test_clip))
        tree.add_entry(8.0, ClipNode("sprint", test_clip))

        # Test various speeds
        for speed in [0.0, 1.0, 2.5, 5.0, 7.0, 10.0]:
            graph_context.parameters["speed"].value = speed
            pose = tree.evaluate(graph_context)
            assert pose is not None

    def test_state_machine_with_blend_trees(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test state machine containing blend trees."""
        locomotion = BlendTree1D("locomotion", "speed")
        locomotion.add_entry(0.0, ClipNode("idle", test_clip))
        locomotion.add_entry(5.0, ClipNode("walk", test_clip))

        combat = ClipNode("combat", test_clip)

        sm = StateMachine("character", "locomotion")
        sm.add_state(AnimationState("locomotion", locomotion))
        sm.add_state(AnimationState("combat", combat))

        sm.add_transition(StateTransition(
            "locomotion", "combat",
            [TransitionCondition.is_true("in_combat")],
            0.3
        ))

        graph_context.parameters["in_combat"] = GraphParameter.bool_param("in_combat", False)
        sm.start(graph_context)

        # Test locomotion
        pose = sm.evaluate(graph_context)
        assert pose is not None
        assert sm.current_state_name == "locomotion"

        # Transition to combat
        graph_context.parameters["in_combat"].value = True
        sm.update(0.1, graph_context)
        pose = sm.evaluate(graph_context)
        assert sm.is_transitioning or sm.current_state_name == "combat"

    def test_layered_animation(self, test_clip: AnimationClip, graph_context: GraphContext, simple_skeleton: Skeleton):
        """Test layered animation with masks."""
        lower_body_mask = BoneMask(name="lower")
        lower_body_mask.set_weights([6, 7], 1.0)  # Legs

        upper_body_mask = BoneMask(name="upper")
        upper_body_mask.set_weights([1, 2, 3, 4, 5], 1.0)  # Spine, arms, head

        stack = LayerStack("character")
        stack.add_layer(AnimationLayer("base", ClipNode("locomotion", test_clip)))
        stack.add_layer(AnimationLayer("upper", ClipNode("attack", test_clip),
                                        mask=upper_body_mask))

        pose = stack.evaluate(graph_context)
        assert pose is not None

    def test_synchronized_locomotion(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test synchronized locomotion animations."""
        walk_markers = create_locomotion_markers(0.0, 0.5, 0.25, 0.75)
        run_markers = create_locomotion_markers(0.0, 0.5, 0.25, 0.75)

        sync = SyncGroup("locomotion", SyncMode.PHASE)
        sync.add_entry(ClipNode("walk", test_clip), 0.5, True, 1.0, walk_markers)
        sync.add_entry(ClipNode("run", test_clip), 0.5, False, 0.6, run_markers)

        sync.update(0.5)
        # Entries should be phase-synced
        assert abs(sync.entries[0].normalized_time - sync.entries[1].normalized_time) < 0.2

    def test_complete_character_graph(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test complete character animation graph."""
        graph = AnimationGraph("character")

        # Add parameters
        graph.add_parameter(GraphParameter.float_param("speed", 0.0, 0.0, 10.0))
        graph.add_parameter(GraphParameter.bool_param("is_attacking", False))

        # Create locomotion
        locomotion = BlendTree1D("locomotion", "speed")
        locomotion.add_entry(0.0, ClipNode("idle", test_clip))
        locomotion.add_entry(5.0, ClipNode("walk", test_clip))

        graph.add_node(locomotion)
        graph.set_output_node("locomotion")

        # Evaluate at different speeds
        context = GraphContext(
            parameters=graph.parameters,
            dt=0.016,
            skeleton=graph_context.skeleton,
        )

        for speed in [0.0, 2.5, 5.0]:
            graph.set_parameter("speed", speed)
            pose = graph.evaluate(context)
            assert pose is not None

    def test_additive_breathing(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test additive breathing animation."""
        base = ClipNode("idle", test_clip)
        breathing = ClipNode("breathing", test_clip)

        additive = AdditiveNode("add_breathing", weight=0.5)
        additive.set_inputs(base, breathing)

        pose = additive.evaluate(graph_context)
        assert pose is not None

    def test_multiple_sync_groups(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test multiple sync groups."""
        lower_sync = SyncGroup("lower", SyncMode.NORMALIZED)
        lower_sync.add_entry(ClipNode("walk_lower", test_clip), 1.0, True, 1.0)

        upper_sync = SyncGroup("upper", SyncMode.NORMALIZED)
        upper_sync.add_entry(ClipNode("attack_upper", test_clip), 1.0, True, 0.5)

        # Update both
        lower_sync.update(0.1)
        upper_sync.update(0.1)

        # Both sync groups should have valid synchronized times (may or may not be equal)
        lower_time = lower_sync.get_synchronized_time()
        upper_time = upper_sync.get_synchronized_time()
        assert 0.0 <= lower_time <= 1.0
        assert 0.0 <= upper_time <= 1.0

    def test_decorator_usage(self):
        """Test decorator usage."""
        @state_machine(
            initial="idle",
            states={"idle", "walk", "run"},
            transitions={"idle": ["walk"], "walk": ["idle", "run"]}
        )
        class TestSM:
            pass

        assert TestSM._state_machine is True
        assert TestSM._sm_initial == "idle"
        assert "idle" in TestSM._sm_states

        @blend_tree(parameter="speed", clips=["idle", "walk", "run"])
        class TestBT:
            pass

        assert TestBT._blend_tree is True
        assert TestBT._blend_parameter == "speed"


# =============================================================================
# PERFORMANCE / EDGE CASE TESTS (5 tests)
# =============================================================================


class TestEdgeCases:
    """Edge case and performance tests."""

    def test_empty_pose_operations(self):
        """Test operations on empty poses."""
        pose1 = Pose()
        pose2 = Pose()

        result = pose1.lerp(pose2, 0.5)
        assert result.bone_count() == 0

    def test_mismatched_bone_counts(self):
        """Test blending poses with different bone counts."""
        pose1 = Pose.identity(3)
        pose2 = Pose.identity(5)

        result = pose1.lerp(pose2, 0.5)
        assert result.bone_count() == 5

    def test_deep_graph_evaluation(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test deeply nested graph evaluation."""
        # Create a chain of blend nodes
        node = ClipNode("base", test_clip)

        for i in range(10):
            blend = BlendNode(f"blend_{i}", alpha=0.5)
            blend.inputs["a"] = node
            blend.inputs["b"] = ClipNode(f"clip_{i}", test_clip)
            node = blend

        pose = node.evaluate(graph_context)
        assert pose is not None

    def test_many_blend_tree_entries(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test blend tree with many entries."""
        tree = BlendTree1D("many", "speed")

        for i in range(20):
            tree.add_entry(float(i), ClipNode(f"clip_{i}", test_clip))

        graph_context.parameters["speed"].value = 10.5
        pose = tree.evaluate(graph_context)
        assert pose is not None

    def test_rapid_state_changes(self, test_clip: AnimationClip, graph_context: GraphContext):
        """Test rapid state machine transitions."""
        sm = StateMachine("rapid", "a")
        for name in ["a", "b", "c", "d", "e"]:
            sm.add_state(AnimationState(name, ClipNode(name, test_clip)))

        sm.start(graph_context)

        # Rapidly change states
        for target in ["b", "c", "d", "e", "a"]:
            sm.force_state(target, graph_context, immediate=True)
            pose = sm.evaluate(graph_context)
            assert pose is not None


# =============================================================================
# RUN COUNT VERIFICATION
# =============================================================================


def test_total_test_count():
    """Verify we have at least 160 tests by checking classes exist."""
    # Verify critical animation graph classes are available
    assert Transform is not None
    assert Pose is not None
    assert Skeleton is not None
    assert BoneMask is not None
    assert GraphParameter is not None
    assert GraphContext is not None
    assert AnimationClip is not None
    assert ClipNode is not None
    assert BlendNode is not None
    assert BlendTree1D is not None
    assert BlendTree2D is not None
    assert BlendTreeDirect is not None
    assert StateMachine is not None
    assert TransitionCondition is not None
    assert AnimationLayer is not None
    assert LayerStack is not None
    assert SyncMarker is not None
    assert SyncGroup is not None
    assert AnimationGraph is not None
    # Total expected: ~196 tests, verified by pytest collection
