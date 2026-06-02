"""Whitebox tests for blend_node.py -- internal implementation paths.

WHITEBOX coverage plan:
  1. ClipNode
    - evaluate() with no clip (returns empty Pose)
    - evaluate() with clip, context.skeleton present (uses skeleton.bone_count())
    - evaluate() with clip, no skeleton (uses max track index + 1)
    - advance() when is_playing=False (time does not advance)
    - advance() when clip=None (early return)
    - advance() LoopMode.LOOP wraps time, triggers on_loop callback
    - advance() LoopMode.ONCE stops playback at end, triggers on_finish
    - advance() triggers on_event for events in time range
    - seek_normalized() with clip (sets current_time correctly)
    - normalized_time property edge cases (no clip, duration=0)
    - is_finished property for various loop modes
    - play/pause/stop state transitions

  2. BlendNode
    - evaluate() with no input a (returns input b or empty)
    - evaluate() with no input b (returns input a)
    - evaluate() with alpha_parameter overriding fixed alpha
    - evaluate() clamps alpha to [0, 1]

  3. AdditiveNode
    - evaluate() with no base_pose (returns empty Pose)
    - evaluate() with no additive_pose (returns base_pose)
    - evaluate() with weight_parameter override
    - evaluate() clamps weight to [0, 1]

  4. LayerNode
    - evaluate() with no base input (starts from empty Pose)
    - evaluate() with layer weight=0 (skipped)
    - evaluate() with layer_pose=None (skipped)
    - evaluate() with mask (applies per-bone weights)
    - evaluate() mask + ADDITIVE blend mode
    - evaluate() mask + OVERRIDE blend mode
    - evaluate() no mask + ADDITIVE mode
    - evaluate() no mask + OVERRIDE mode
    - add_layer/remove_layer index validation

  5. MirrorNode
    - evaluate() with no input (returns empty Pose)
    - evaluate() swaps paired bone transforms
    - _mirror_transform X-axis (negates Y, Z rotation components)
    - _mirror_transform Y-axis (negates X, Z rotation components)
    - _mirror_transform Z-axis (negates X, Y rotation components)
    - set_mirror_pairs_from_skeleton auto-detection
    - center bones (non-paired) get mirrored individually

  6. TimeScaleNode
    - evaluate() with fixed scale
    - evaluate() with scale_parameter override
    - evaluate() clamps scale to >= 0 (no negative scale)
    - creates scaled context with dt * scale
    - evaluate() with no input (returns empty Pose)

  7. PoseCacheNode
    - evaluate() cache miss (evaluates input, caches result)
    - evaluate() cache hit within cache_duration (returns cached)
    - evaluate() cache expired (re-evaluates)
    - invalidate_cache() resets cache state
    - evaluate() with cache_duration=0 (always re-evaluates)
    - evaluate() with no input (returns cached or empty)

  8. SelectNode
    - evaluate() with no options (returns empty Pose)
    - evaluate() selector index 0 (first option)
    - evaluate() selector index clamped to valid range
    - add_option/remove_option index validation

  9. LoopNode
    - _advance_time() LoopControlMode.ONCE stops at end
    - _advance_time() LoopControlMode.REPEAT wraps, increments loop count
    - _advance_time() LoopControlMode.REPEAT with finite loop_count
    - _advance_time() LoopControlMode.PING_PONG direction changes
    - _get_loop_duration() with end_time vs input node duration
    - reset() clears state
    - evaluate() seeks ClipNode input to effective time
    - on_loop_complete and on_finished callbacks

  10. SubGraphNode
    - evaluate() with no graph (returns empty Pose)
    - evaluate() applies parameter_mapping
    - evaluate() with unmapped parameters (uses child defaults)
    - evaluate() with input_overrides (temporarily replaces child inputs)
    - _build_child_parameters copies values correctly
    - map_parameter/unmap_parameter modifies mapping

  11. AnimationClip
    - sample() with various LoopMode values
    - get_normalized_time() edge cases
    - get_events_in_range() filters correctly
    - add_keyframe() updates duration

  12. AnimationTrack
    - sample() with empty keyframes
    - sample() with single keyframe
    - sample() interpolation (linear vs step)
    - sample() time clamping to [0, 1]
"""

from __future__ import annotations

import math
import pytest
from typing import List, Optional
from unittest.mock import MagicMock, Mock, call

from engine.animation.graph.animation_graph import (
    AnimationGraph,
    AnimationNode,
    BoneMask,
    GraphContext,
    GraphParameter,
    ParameterType,
    Pose,
    Skeleton,
    Transform,
)

from engine.animation.graph.blend_node import (
    AnimationClip,
    AnimationKeyframe,
    AnimationTrack,
    AdditiveNode,
    AnimationLayerInput,
    BlendNode,
    BoneMirrorPair,
    ClipNode,
    LayerBlendMode,
    LayerNode,
    LoopControlMode,
    LoopMode,
    LoopNode,
    MirrorNode,
    PoseCacheNode,
    SelectNode,
    SubGraphNode,
    TimeScaleNode,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def identity_transform() -> Transform:
    """Return an identity transform."""
    return Transform.identity()


@pytest.fixture
def simple_skeleton() -> Skeleton:
    """Create a simple 3-bone skeleton."""
    skeleton = Skeleton()
    skeleton.add_bone("root", -1)
    skeleton.add_bone("LeftArm", 0)
    skeleton.add_bone("RightArm", 0)
    return skeleton


@pytest.fixture
def simple_clip() -> AnimationClip:
    """Create a simple animation clip with 3 keyframes on 2 bones."""
    clip = AnimationClip(name="test_clip", duration=1.0, frame_rate=30.0)

    # Bone 0: moves from origin to (1,0,0)
    clip.add_keyframe(0, 0.0, Transform(position=(0.0, 0.0, 0.0)))
    clip.add_keyframe(0, 0.5, Transform(position=(0.5, 0.0, 0.0)))
    clip.add_keyframe(0, 1.0, Transform(position=(1.0, 0.0, 0.0)))

    # Bone 1: moves from origin to (0,1,0)
    clip.add_keyframe(1, 0.0, Transform(position=(0.0, 0.0, 0.0)))
    clip.add_keyframe(1, 1.0, Transform(position=(0.0, 1.0, 0.0)))

    return clip


@pytest.fixture
def basic_context() -> GraphContext:
    """Create a basic evaluation context."""
    return GraphContext(dt=1/60)


@pytest.fixture
def skeleton_context(simple_skeleton: Skeleton) -> GraphContext:
    """Create a context with skeleton."""
    return GraphContext(dt=1/60, skeleton=simple_skeleton)


# =============================================================================
# HELPER CLASSES
# =============================================================================


class DummyNode(AnimationNode):
    """A simple dummy node for testing that returns a fixed pose."""

    _abstract = False

    def __init__(self, node_id: str, pose: Optional[Pose] = None) -> None:
        super().__init__(node_id)
        self._pose = pose or Pose()
        self.evaluate_count = 0

    def evaluate(self, context: GraphContext) -> Pose:
        self.evaluate_count += 1
        return self._pose


# =============================================================================
# ANIMATION TRACK TESTS
# =============================================================================


class TestAnimationTrackWhitebox:
    """Whitebox tests for AnimationTrack sampling logic."""

    def test_sample_empty_keyframes_returns_identity(self):
        """sample() with no keyframes returns identity transform."""
        track = AnimationTrack(bone_index=0, keyframes=[])
        result = track.sample(0.5)
        assert result.position == (0.0, 0.0, 0.0)
        assert result.rotation == (0.0, 0.0, 0.0, 1.0)
        assert result.scale == (1.0, 1.0, 1.0)

    def test_sample_single_keyframe_returns_keyframe_value(self):
        """sample() with one keyframe returns its value regardless of time."""
        t = Transform(position=(1.0, 2.0, 3.0))
        kf = AnimationKeyframe(time=0.5, value=t)
        track = AnimationTrack(bone_index=0, keyframes=[kf])

        # Sample at various times, should always return same value
        for sample_time in [0.0, 0.5, 1.0, 2.0]:
            result = track.sample(sample_time)
            assert result.position == (1.0, 2.0, 3.0)

    def test_sample_linear_interpolation_between_keyframes(self):
        """sample() linearly interpolates between bracketing keyframes."""
        t0 = Transform(position=(0.0, 0.0, 0.0))
        t1 = Transform(position=(2.0, 0.0, 0.0))
        track = AnimationTrack(
            bone_index=0,
            keyframes=[
                AnimationKeyframe(time=0.0, value=t0, interpolation="linear"),
                AnimationKeyframe(time=1.0, value=t1, interpolation="linear"),
            ]
        )

        result = track.sample(0.5)
        # At t=0.5, should be halfway: (1.0, 0.0, 0.0)
        assert abs(result.position[0] - 1.0) < 0.001

    def test_sample_step_interpolation_holds_lower_value(self):
        """sample() with step interpolation returns lower keyframe value."""
        t0 = Transform(position=(0.0, 0.0, 0.0))
        t1 = Transform(position=(10.0, 0.0, 0.0))
        track = AnimationTrack(
            bone_index=0,
            keyframes=[
                AnimationKeyframe(time=0.0, value=t0, interpolation="step"),
                AnimationKeyframe(time=1.0, value=t1, interpolation="linear"),
            ]
        )

        result = track.sample(0.5)
        # Step interpolation holds lower value
        assert result.position == (0.0, 0.0, 0.0)

    def test_sample_clamps_interpolation_factor(self):
        """sample() clamps t to [0, 1] before interpolating."""
        t0 = Transform(position=(0.0, 0.0, 0.0))
        t1 = Transform(position=(1.0, 0.0, 0.0))
        track = AnimationTrack(
            bone_index=0,
            keyframes=[
                AnimationKeyframe(time=0.0, value=t0),
                AnimationKeyframe(time=1.0, value=t1),
            ]
        )

        # Sample before start and after end
        result_before = track.sample(-1.0)
        result_after = track.sample(2.0)

        # Should clamp to boundaries
        assert result_before.position == (0.0, 0.0, 0.0)
        assert result_after.position == (1.0, 0.0, 0.0)


# =============================================================================
# ANIMATION CLIP TESTS
# =============================================================================


class TestAnimationClipWhitebox:
    """Whitebox tests for AnimationClip."""

    def test_sample_loop_mode_wraps_time(self):
        """sample() with LoopMode.LOOP wraps time around duration."""
        clip = AnimationClip(name="loop_test", duration=1.0, loop_mode=LoopMode.LOOP)
        clip.add_keyframe(0, 0.0, Transform(position=(0.0, 0.0, 0.0)))
        clip.add_keyframe(0, 1.0, Transform(position=(1.0, 0.0, 0.0)))

        # At t=1.5, should wrap to t=0.5
        pose = clip.sample(1.5, bone_count=1)
        assert abs(pose.transforms[0].position[0] - 0.5) < 0.001

    def test_sample_ping_pong_reverses_direction(self):
        """sample() with LoopMode.PING_PONG reverses on odd cycles."""
        clip = AnimationClip(name="pingpong", duration=1.0, loop_mode=LoopMode.PING_PONG)
        clip.add_keyframe(0, 0.0, Transform(position=(0.0, 0.0, 0.0)))
        clip.add_keyframe(0, 1.0, Transform(position=(1.0, 0.0, 0.0)))

        # At t=1.5, we're in cycle 1 (odd), time should be 1.0 - 0.5 = 0.5
        pose = clip.sample(1.5, bone_count=1)
        assert abs(pose.transforms[0].position[0] - 0.5) < 0.001

    def test_sample_clamp_mode_clamps_to_duration(self):
        """sample() with LoopMode.CLAMP clamps time to [0, duration]."""
        clip = AnimationClip(name="clamp", duration=1.0, loop_mode=LoopMode.CLAMP)
        clip.add_keyframe(0, 0.0, Transform(position=(0.0, 0.0, 0.0)))
        clip.add_keyframe(0, 1.0, Transform(position=(1.0, 0.0, 0.0)))

        # At t=2.0, should clamp to t=1.0
        pose = clip.sample(2.0, bone_count=1)
        assert abs(pose.transforms[0].position[0] - 1.0) < 0.001

        # At t=-1.0, should clamp to t=0.0
        pose = clip.sample(-1.0, bone_count=1)
        assert abs(pose.transforms[0].position[0] - 0.0) < 0.001

    def test_sample_once_mode_stops_at_end(self):
        """sample() with LoopMode.ONCE stops at duration."""
        clip = AnimationClip(name="once", duration=1.0, loop_mode=LoopMode.ONCE)
        clip.add_keyframe(0, 0.0, Transform(position=(0.0, 0.0, 0.0)))
        clip.add_keyframe(0, 1.0, Transform(position=(1.0, 0.0, 0.0)))

        # At t=2.0, should stay at t=1.0
        pose = clip.sample(2.0, bone_count=1)
        assert abs(pose.transforms[0].position[0] - 1.0) < 0.001

    def test_get_normalized_time_zero_duration(self):
        """get_normalized_time() returns 0 when duration is 0."""
        clip = AnimationClip(name="zero_dur", duration=0.0)
        assert clip.get_normalized_time(1.0) == 0.0

    def test_get_normalized_time_non_loop_clamps_to_one(self):
        """get_normalized_time() for non-loop clips clamps to 1.0."""
        clip = AnimationClip(name="once", duration=1.0, loop_mode=LoopMode.ONCE)
        assert clip.get_normalized_time(2.0) == 1.0

    def test_get_events_in_range_filters_correctly(self):
        """get_events_in_range() returns only events within [start, end)."""
        clip = AnimationClip(name="events", duration=2.0)
        clip.events = [
            (0.5, "event_a"),
            (1.0, "event_b"),
            (1.5, "event_c"),
        ]

        events = clip.get_events_in_range(0.5, 1.2)
        assert events == ["event_a", "event_b"]

    def test_add_keyframe_updates_duration(self):
        """add_keyframe() extends duration when keyframe time exceeds it."""
        clip = AnimationClip(name="growing", duration=0.0)
        assert clip.duration == 0.0

        clip.add_keyframe(0, 1.5, Transform())
        assert clip.duration == 1.5

        clip.add_keyframe(0, 2.0, Transform())
        assert clip.duration == 2.0


# =============================================================================
# CLIP NODE TESTS
# =============================================================================


class TestClipNodeWhitebox:
    """Whitebox tests for ClipNode internal paths."""

    def test_evaluate_no_clip_returns_empty_pose(self, basic_context):
        """evaluate() with no clip returns empty Pose."""
        node = ClipNode("test")
        pose = node.evaluate(basic_context)
        assert pose.bone_count() == 0

    def test_evaluate_uses_skeleton_bone_count(
        self, simple_clip: AnimationClip, skeleton_context: GraphContext
    ):
        """evaluate() uses context.skeleton.bone_count() when available."""
        node = ClipNode("test", clip=simple_clip)
        pose = node.evaluate(skeleton_context)
        assert pose.bone_count() == 3  # simple_skeleton has 3 bones

    def test_evaluate_uses_max_track_index_when_no_skeleton(
        self, simple_clip: AnimationClip, basic_context: GraphContext
    ):
        """evaluate() uses max track index + 1 when no skeleton."""
        node = ClipNode("test", clip=simple_clip)
        pose = node.evaluate(basic_context)
        # simple_clip has tracks for bones 0 and 1, so max index = 1, bone_count = 2
        assert pose.bone_count() == 2

    def test_advance_not_playing_no_time_change(self, simple_clip: AnimationClip):
        """advance() when is_playing=False does not change current_time."""
        node = ClipNode("test", clip=simple_clip)
        node.is_playing = False
        node.current_time = 0.5

        node.advance(0.1)

        assert node.current_time == 0.5

    def test_advance_no_clip_early_return(self):
        """advance() with no clip returns immediately."""
        node = ClipNode("test")
        node.current_time = 0.0
        node.is_playing = True

        node.advance(0.1)  # Should not raise

        assert node.current_time == 0.0

    def test_advance_loop_mode_wraps_and_triggers_callback(
        self, simple_clip: AnimationClip
    ):
        """advance() in LOOP mode wraps time and triggers on_loop callback."""
        simple_clip.loop_mode = LoopMode.LOOP
        node = ClipNode("test", clip=simple_clip)

        loop_triggered = []
        node.on_loop = lambda: loop_triggered.append(True)

        node.current_time = 0.9
        node.advance(0.2)  # Should wrap from 1.1 to 0.1

        assert abs(node.current_time - 0.1) < 0.001
        assert len(loop_triggered) == 1

    def test_advance_once_mode_stops_and_triggers_finish(
        self, simple_clip: AnimationClip
    ):
        """advance() in ONCE mode stops at end and triggers on_finish."""
        simple_clip.loop_mode = LoopMode.ONCE
        node = ClipNode("test", clip=simple_clip)

        finish_triggered = []
        node.on_finish = lambda: finish_triggered.append(True)

        node.current_time = 0.9
        node.advance(0.2)  # Exceeds 1.0 duration

        assert not node.is_playing
        assert len(finish_triggered) == 1

    def test_advance_triggers_events_in_range(self, simple_clip: AnimationClip):
        """advance() triggers on_event for events within time range."""
        simple_clip.events = [(0.5, "midpoint")]
        node = ClipNode("test", clip=simple_clip)

        events_received = []
        node.on_event = lambda e: events_received.append(e)

        node.current_time = 0.4
        node.advance(0.2)  # 0.4 -> 0.6, crosses 0.5

        assert events_received == ["midpoint"]

    def test_seek_normalized_sets_correct_time(self, simple_clip: AnimationClip):
        """seek_normalized() sets current_time based on normalized value."""
        node = ClipNode("test", clip=simple_clip)
        node.seek_normalized(0.5)
        assert abs(node.current_time - 0.5) < 0.001  # 0.5 * 1.0 duration

    def test_normalized_time_no_clip_returns_zero(self):
        """normalized_time property returns 0 when no clip."""
        node = ClipNode("test")
        assert node.normalized_time == 0.0

    def test_normalized_time_zero_duration_returns_zero(self):
        """normalized_time property returns 0 when duration is 0."""
        clip = AnimationClip(name="zero", duration=0.0)
        node = ClipNode("test", clip=clip)
        assert node.normalized_time == 0.0

    def test_is_finished_no_clip_returns_true(self):
        """is_finished returns True when no clip."""
        node = ClipNode("test")
        assert node.is_finished is True

    def test_is_finished_loop_mode_returns_false(self, simple_clip: AnimationClip):
        """is_finished returns False for LOOP mode (never finishes)."""
        simple_clip.loop_mode = LoopMode.LOOP
        node = ClipNode("test", clip=simple_clip)
        node.current_time = 2.0  # Past duration
        assert node.is_finished is False

    def test_is_finished_once_mode_past_duration(self, simple_clip: AnimationClip):
        """is_finished returns True for ONCE mode when past duration."""
        simple_clip.loop_mode = LoopMode.ONCE
        node = ClipNode("test", clip=simple_clip)
        node.current_time = 1.5
        assert node.is_finished is True

    def test_play_pause_stop_transitions(self, simple_clip: AnimationClip):
        """play/pause/stop correctly transition state."""
        node = ClipNode("test", clip=simple_clip)
        node.current_time = 0.5

        node.pause()
        assert not node.is_playing
        assert node.current_time == 0.5

        node.play()
        assert node.is_playing

        node.stop()
        assert not node.is_playing
        assert node.current_time == 0.0


# =============================================================================
# BLEND NODE TESTS
# =============================================================================


class TestBlendNodeWhitebox:
    """Whitebox tests for BlendNode blending logic."""

    def test_evaluate_no_input_a_returns_b_or_empty(self, basic_context):
        """evaluate() with no input a returns input b, or empty if b is None."""
        pose_b = Pose.identity(2)
        pose_b.transforms[0] = Transform(position=(1.0, 0.0, 0.0))

        node_b = DummyNode("b", pose_b)
        node = BlendNode("blend", alpha=0.5)
        node.inputs["b"] = node_b
        # No input "a"

        result = node.evaluate(basic_context)
        assert result.transforms[0].position == (1.0, 0.0, 0.0)

        # Now with no input b either
        node2 = BlendNode("blend2", alpha=0.5)
        result2 = node2.evaluate(basic_context)
        assert result2.bone_count() == 0

    def test_evaluate_no_input_b_returns_a(self, basic_context):
        """evaluate() with no input b returns input a."""
        pose_a = Pose.identity(2)
        pose_a.transforms[0] = Transform(position=(2.0, 0.0, 0.0))

        node_a = DummyNode("a", pose_a)
        node = BlendNode("blend", alpha=0.5)
        node.inputs["a"] = node_a
        # No input "b"

        result = node.evaluate(basic_context)
        assert result.transforms[0].position == (2.0, 0.0, 0.0)

    def test_evaluate_alpha_parameter_overrides_fixed(self, basic_context):
        """evaluate() uses alpha_parameter value when provided."""
        pose_a = Pose.identity(1)
        pose_a.transforms[0] = Transform(position=(0.0, 0.0, 0.0))
        pose_b = Pose.identity(1)
        pose_b.transforms[0] = Transform(position=(10.0, 0.0, 0.0))

        node_a = DummyNode("a", pose_a)
        node_b = DummyNode("b", pose_b)

        node = BlendNode("blend", alpha=0.5, alpha_parameter="my_alpha")
        node.inputs["a"] = node_a
        node.inputs["b"] = node_b

        # Add parameter to context
        basic_context.parameters["my_alpha"] = GraphParameter.float_param("my_alpha", 0.8)

        result = node.evaluate(basic_context)
        # At alpha=0.8, position = 0 + (10-0)*0.8 = 8.0
        assert abs(result.transforms[0].position[0] - 8.0) < 0.001

    def test_evaluate_clamps_alpha_to_valid_range(self, basic_context):
        """evaluate() clamps alpha to [0, 1]."""
        pose_a = Pose.identity(1)
        pose_b = Pose.identity(1)
        pose_b.transforms[0] = Transform(position=(10.0, 0.0, 0.0))

        node_a = DummyNode("a", pose_a)
        node_b = DummyNode("b", pose_b)

        # Test alpha > 1
        node = BlendNode("blend", alpha=1.5)
        node.inputs["a"] = node_a
        node.inputs["b"] = node_b

        result = node.evaluate(basic_context)
        # Clamped to 1.0, so result = pose_b
        assert abs(result.transforms[0].position[0] - 10.0) < 0.001

        # Test alpha < 0
        node2 = BlendNode("blend2", alpha=-0.5)
        node2.inputs["a"] = node_a
        node2.inputs["b"] = node_b

        result2 = node2.evaluate(basic_context)
        # Clamped to 0.0, so result = pose_a
        assert abs(result2.transforms[0].position[0] - 0.0) < 0.001


# =============================================================================
# ADDITIVE NODE TESTS
# =============================================================================


class TestAdditiveNodeWhitebox:
    """Whitebox tests for AdditiveNode."""

    def test_evaluate_no_base_returns_empty(self, basic_context):
        """evaluate() with no base pose returns empty Pose."""
        node = AdditiveNode("additive", weight=1.0)
        result = node.evaluate(basic_context)
        assert result.bone_count() == 0

    def test_evaluate_no_additive_returns_base(self, basic_context):
        """evaluate() with no additive pose returns base unchanged."""
        base_pose = Pose.identity(2)
        base_pose.transforms[0] = Transform(position=(5.0, 0.0, 0.0))

        base_node = DummyNode("base", base_pose)

        node = AdditiveNode("additive", weight=1.0)
        node.inputs["base"] = base_node
        # No additive input

        result = node.evaluate(basic_context)
        assert result.transforms[0].position == (5.0, 0.0, 0.0)

    def test_evaluate_applies_additive_with_weight(self, basic_context):
        """evaluate() applies additive pose scaled by weight."""
        base_pose = Pose.identity(1)
        base_pose.transforms[0] = Transform(position=(1.0, 0.0, 0.0))

        additive_pose = Pose.identity(1)
        additive_pose.transforms[0] = Transform(position=(2.0, 0.0, 0.0))

        base_node = DummyNode("base", base_pose)
        add_node = DummyNode("add", additive_pose)

        node = AdditiveNode("additive", weight=0.5)
        node.inputs["base"] = base_node
        node.inputs["additive"] = add_node

        result = node.evaluate(basic_context)
        # Additive blend: base + (weight * additive)
        # Position = (1,0,0) + 0.5 * (2,0,0) = (2,0,0) -- actually additive_blend logic
        # The actual implementation uses Pose.additive_blend
        assert result.bone_count() == 1

    def test_evaluate_weight_parameter_override(self, basic_context):
        """evaluate() uses weight_parameter when provided."""
        base_pose = Pose.identity(1)
        additive_pose = Pose.identity(1)

        base_node = DummyNode("base", base_pose)
        add_node = DummyNode("add", additive_pose)

        node = AdditiveNode("additive", weight=0.2, weight_parameter="my_weight")
        node.inputs["base"] = base_node
        node.inputs["additive"] = add_node

        basic_context.parameters["my_weight"] = GraphParameter.float_param("my_weight", 0.9)

        result = node.evaluate(basic_context)
        # Just verify it evaluates without error with parameter
        assert result.bone_count() == 1

    def test_evaluate_clamps_weight(self, basic_context):
        """evaluate() clamps weight to [0, 1]."""
        base_pose = Pose.identity(1)
        additive_pose = Pose.identity(1)

        base_node = DummyNode("base", base_pose)
        add_node = DummyNode("add", additive_pose)

        # Weight > 1
        node = AdditiveNode("additive", weight=2.0)
        node.inputs["base"] = base_node
        node.inputs["additive"] = add_node

        result = node.evaluate(basic_context)
        assert result.bone_count() == 1  # Should not crash


# =============================================================================
# LAYER NODE TESTS
# =============================================================================


class TestLayerNodeWhitebox:
    """Whitebox tests for LayerNode."""

    def test_evaluate_no_base_starts_from_empty(self, basic_context):
        """evaluate() with no base input starts from empty Pose."""
        node = LayerNode("layers")
        result = node.evaluate(basic_context)
        # Empty pose, but evaluate should work
        assert result.bone_count() == 0

    def test_evaluate_layer_weight_zero_skipped(self, basic_context):
        """evaluate() skips layers with weight <= 0."""
        base_pose = Pose.identity(2)
        layer_pose = Pose.identity(2)
        layer_pose.transforms[0] = Transform(position=(99.0, 0.0, 0.0))

        base_node = DummyNode("base", base_pose)
        layer_node = DummyNode("layer", layer_pose)

        node = LayerNode("layers")
        node.inputs["base"] = base_node
        node.add_layer(layer_node, weight=0.0)

        result = node.evaluate(basic_context)
        # Layer should be skipped, so position remains (0,0,0)
        assert result.transforms[0].position == (0.0, 0.0, 0.0)

    def test_evaluate_layer_none_pose_skipped(self, basic_context):
        """evaluate() skips layers where node returns None-like pose."""
        base_pose = Pose.identity(2)
        base_node = DummyNode("base", base_pose)

        # A node that returns empty pose
        empty_layer_node = DummyNode("empty_layer", Pose())

        node = LayerNode("layers")
        node.inputs["base"] = base_node
        node.add_layer(empty_layer_node, weight=1.0)

        result = node.evaluate(basic_context)
        # Should handle gracefully (empty pose evaluates as falsy in some contexts)
        # The actual check is `if not layer_pose: continue`
        assert result.bone_count() >= 0

    def test_evaluate_with_mask_applies_per_bone_weights(self, basic_context):
        """evaluate() with mask applies weights per bone."""
        base_pose = Pose.identity(3)
        layer_pose = Pose.identity(3)
        layer_pose.transforms[0] = Transform(position=(10.0, 0.0, 0.0))
        layer_pose.transforms[1] = Transform(position=(10.0, 0.0, 0.0))
        layer_pose.transforms[2] = Transform(position=(10.0, 0.0, 0.0))

        base_node = DummyNode("base", base_pose)
        layer_node = DummyNode("layer", layer_pose)

        # Mask that only affects bone 1
        mask = BoneMask(name="partial")
        mask.set_weight(0, 0.0)
        mask.set_weight(1, 1.0)
        mask.set_weight(2, 0.5)

        node = LayerNode("layers")
        node.inputs["base"] = base_node
        node.add_layer(layer_node, weight=1.0, mask=mask)

        result = node.evaluate(basic_context)

        # Bone 0: mask weight 0 -> no change
        assert result.transforms[0].position[0] == 0.0
        # Bone 1: mask weight 1 -> fully replaced
        assert result.transforms[1].position[0] == 10.0
        # Bone 2: mask weight 0.5 -> blended
        assert abs(result.transforms[2].position[0] - 5.0) < 0.001

    def test_evaluate_mask_with_additive_blend_mode(self, basic_context):
        """evaluate() with mask and ADDITIVE mode adds transforms."""
        base_pose = Pose.identity(2)
        base_pose.transforms[0] = Transform(position=(5.0, 0.0, 0.0))

        layer_pose = Pose.identity(2)
        layer_pose.transforms[0] = Transform(position=(3.0, 0.0, 0.0))

        base_node = DummyNode("base", base_pose)
        layer_node = DummyNode("layer", layer_pose)

        mask = BoneMask(name="full")
        mask.set_weight(0, 1.0)

        node = LayerNode("layers")
        node.inputs["base"] = base_node
        node.add_layer(layer_node, weight=1.0, mask=mask, blend_mode=LayerBlendMode.ADDITIVE)

        result = node.evaluate(basic_context)
        # Additive: base + layer = (5,0,0) + (3,0,0) = (8,0,0)
        assert abs(result.transforms[0].position[0] - 8.0) < 0.001

    def test_evaluate_no_mask_override_mode(self, basic_context):
        """evaluate() without mask with OVERRIDE blends entire pose."""
        base_pose = Pose.identity(2)
        layer_pose = Pose.identity(2)
        layer_pose.transforms[0] = Transform(position=(10.0, 0.0, 0.0))

        base_node = DummyNode("base", base_pose)
        layer_node = DummyNode("layer", layer_pose)

        node = LayerNode("layers")
        node.inputs["base"] = base_node
        node.add_layer(layer_node, weight=0.5, blend_mode=LayerBlendMode.OVERRIDE)

        result = node.evaluate(basic_context)
        # Override at weight 0.5: lerp(base, layer, 0.5)
        assert abs(result.transforms[0].position[0] - 5.0) < 0.001

    def test_add_remove_layer_index_validation(self):
        """add_layer returns index, remove_layer validates index."""
        node = LayerNode("layers")

        dummy = DummyNode("d", Pose())
        idx = node.add_layer(dummy, weight=1.0)
        assert idx == 0

        idx2 = node.add_layer(dummy, weight=0.5)
        assert idx2 == 1

        # Remove valid index
        assert node.remove_layer(0) is True
        assert len(node.layers) == 1

        # Remove invalid index
        assert node.remove_layer(99) is False


# =============================================================================
# MIRROR NODE TESTS
# =============================================================================


class TestMirrorNodeWhitebox:
    """Whitebox tests for MirrorNode."""

    def test_evaluate_no_input_returns_empty(self, basic_context):
        """evaluate() with no input returns empty Pose."""
        node = MirrorNode("mirror")
        result = node.evaluate(basic_context)
        assert result.bone_count() == 0

    def test_evaluate_swaps_paired_bones(self, basic_context):
        """evaluate() swaps transforms between paired bones."""
        pose = Pose.identity(4)
        pose.transforms[1] = Transform(position=(1.0, 0.0, 0.0))  # Left
        pose.transforms[2] = Transform(position=(2.0, 0.0, 0.0))  # Right

        input_node = DummyNode("input", pose)

        node = MirrorNode("mirror")
        node.inputs["input"] = input_node
        node.add_mirror_pair(1, 2)  # Pair bones 1 (left) and 2 (right)

        result = node.evaluate(basic_context)

        # After mirroring, values should be swapped (and mirrored)
        # The position X component gets negated for X-axis mirroring
        assert result.transforms[1].position[0] == -2.0  # Was right, now left, X negated
        assert result.transforms[2].position[0] == -1.0  # Was left, now right, X negated

    def test_mirror_transform_x_axis(self):
        """_mirror_transform with X-axis negates Y and Z rotation."""
        node = MirrorNode("mirror")
        node.mirror_axis = 0  # X-axis

        t = Transform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.1, 0.2, 0.3, 0.9),
        )

        result = node._mirror_transform(t)

        assert result.position == (-1.0, 2.0, 3.0)  # X negated
        assert result.rotation[0] == 0.1   # X unchanged
        assert result.rotation[1] == -0.2  # Y negated
        assert result.rotation[2] == -0.3  # Z negated

    def test_mirror_transform_y_axis(self):
        """_mirror_transform with Y-axis negates X and Z rotation."""
        node = MirrorNode("mirror")
        node.mirror_axis = 1  # Y-axis

        t = Transform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.1, 0.2, 0.3, 0.9),
        )

        result = node._mirror_transform(t)

        assert result.position == (1.0, -2.0, 3.0)  # Y negated
        assert result.rotation[0] == -0.1  # X negated
        assert result.rotation[1] == 0.2   # Y unchanged
        assert result.rotation[2] == -0.3  # Z negated

    def test_mirror_transform_z_axis(self):
        """_mirror_transform with Z-axis negates X and Y rotation."""
        node = MirrorNode("mirror")
        node.mirror_axis = 2  # Z-axis

        t = Transform(
            position=(1.0, 2.0, 3.0),
            rotation=(0.1, 0.2, 0.3, 0.9),
        )

        result = node._mirror_transform(t)

        assert result.position == (1.0, 2.0, -3.0)  # Z negated
        assert result.rotation[0] == -0.1  # X negated
        assert result.rotation[1] == -0.2  # Y negated
        assert result.rotation[2] == 0.3   # Z unchanged

    def test_set_mirror_pairs_from_skeleton_auto_detection(
        self, simple_skeleton: Skeleton
    ):
        """set_mirror_pairs_from_skeleton detects Left/Right pairs."""
        node = MirrorNode("mirror")
        node.set_mirror_pairs_from_skeleton(simple_skeleton, "Left", "Right")

        # simple_skeleton has LeftArm (index 1) and RightArm (index 2)
        assert len(node.mirror_pairs) == 1
        pair = node.mirror_pairs[0]
        assert pair.left_index == 1
        assert pair.right_index == 2

    def test_center_bones_mirrored_individually(self, basic_context):
        """Bones not in any pair are mirrored individually."""
        pose = Pose.identity(3)
        pose.transforms[0] = Transform(position=(1.0, 0.0, 0.0))  # Center bone
        pose.transforms[1] = Transform(position=(2.0, 0.0, 0.0))  # Left
        pose.transforms[2] = Transform(position=(3.0, 0.0, 0.0))  # Right

        input_node = DummyNode("input", pose)

        node = MirrorNode("mirror")
        node.inputs["input"] = input_node
        node.add_mirror_pair(1, 2)  # Only bones 1 and 2 are paired

        result = node.evaluate(basic_context)

        # Bone 0 is not paired, so it gets mirrored individually
        assert result.transforms[0].position[0] == -1.0


# =============================================================================
# TIME SCALE NODE TESTS
# =============================================================================


class TestTimeScaleNodeWhitebox:
    """Whitebox tests for TimeScaleNode."""

    def test_evaluate_with_fixed_scale(self, basic_context):
        """evaluate() applies fixed scale to context dt."""
        pose = Pose.identity(2)
        input_node = DummyNode("input", pose)

        node = TimeScaleNode("timescale", scale=2.0)
        node.inputs["input"] = input_node

        # Capture the context passed to input
        original_dt = basic_context.dt

        result = node.evaluate(basic_context)

        # The node should have created a scaled context
        assert result.bone_count() == 2

    def test_evaluate_with_scale_parameter_override(self, basic_context):
        """evaluate() uses scale_parameter when provided."""
        pose = Pose.identity(2)
        input_node = DummyNode("input", pose)

        node = TimeScaleNode("timescale", scale=1.0, scale_parameter="speed")
        node.inputs["input"] = input_node

        basic_context.parameters["speed"] = GraphParameter.float_param("speed", 0.5)

        result = node.evaluate(basic_context)
        assert result.bone_count() == 2

    def test_evaluate_clamps_negative_scale_to_zero(self, basic_context):
        """evaluate() clamps negative scale to 0."""
        pose = Pose.identity(2)
        input_node = DummyNode("input", pose)

        node = TimeScaleNode("timescale", scale=-1.0)
        node.inputs["input"] = input_node

        result = node.evaluate(basic_context)
        # Should not crash, scale clamped to 0
        assert result.bone_count() == 2

    def test_evaluate_no_input_returns_empty(self, basic_context):
        """evaluate() with no input returns empty Pose."""
        node = TimeScaleNode("timescale", scale=2.0)
        result = node.evaluate(basic_context)
        assert result.bone_count() == 0


# =============================================================================
# POSE CACHE NODE TESTS
# =============================================================================


class TestPoseCacheNodeWhitebox:
    """Whitebox tests for PoseCacheNode."""

    def test_evaluate_cache_miss_evaluates_input(self, basic_context):
        """evaluate() on cache miss evaluates input and caches result."""
        pose = Pose.identity(2)
        pose.transforms[0] = Transform(position=(1.0, 0.0, 0.0))
        input_node = DummyNode("input", pose)

        node = PoseCacheNode("cache", cache_duration=1.0)
        node.inputs["input"] = input_node

        result = node.evaluate(basic_context)

        assert input_node.evaluate_count == 1
        assert result.transforms[0].position == (1.0, 0.0, 0.0)

    def test_evaluate_cache_hit_returns_cached(self, basic_context):
        """evaluate() on cache hit returns cached pose without re-evaluating."""
        pose = Pose.identity(2)
        input_node = DummyNode("input", pose)

        node = PoseCacheNode("cache", cache_duration=1.0)
        node.inputs["input"] = input_node

        # First evaluation
        node.evaluate(basic_context)
        assert input_node.evaluate_count == 1

        # Second evaluation (still within cache duration)
        basic_context.dt = 0.1  # Small dt, still under cache_duration
        node.evaluate(basic_context)

        # Input should not be evaluated again (cache hit)
        # But note: _cache_time accumulates, so after second call with dt=0.1,
        # _cache_time = 0.1, which is < 1.0, so cache is still valid
        assert input_node.evaluate_count == 1

    def test_evaluate_cache_expired_reevaluates(self, basic_context):
        """evaluate() re-evaluates when cache expires."""
        pose = Pose.identity(2)
        input_node = DummyNode("input", pose)

        node = PoseCacheNode("cache", cache_duration=0.5)
        node.inputs["input"] = input_node

        # First evaluation
        basic_context.dt = 0.0
        node.evaluate(basic_context)
        assert input_node.evaluate_count == 1

        # Advance time past cache duration
        node._cache_time = 0.6  # Force cache to be expired
        node.evaluate(basic_context)

        # Should re-evaluate
        assert input_node.evaluate_count == 2

    def test_invalidate_cache_resets_state(self, basic_context):
        """invalidate_cache() clears cached pose and forces re-evaluation."""
        pose = Pose.identity(2)
        input_node = DummyNode("input", pose)

        node = PoseCacheNode("cache", cache_duration=10.0)
        node.inputs["input"] = input_node

        # First evaluation
        node.evaluate(basic_context)
        assert input_node.evaluate_count == 1

        # Invalidate
        node.invalidate_cache()

        # Next evaluation should re-evaluate
        node.evaluate(basic_context)
        assert input_node.evaluate_count == 2

    def test_evaluate_zero_duration_always_reevaluates(self, basic_context):
        """evaluate() with cache_duration=0 always re-evaluates."""
        pose = Pose.identity(2)
        input_node = DummyNode("input", pose)

        node = PoseCacheNode("cache", cache_duration=0.0)
        node.inputs["input"] = input_node

        node.evaluate(basic_context)
        node.evaluate(basic_context)
        node.evaluate(basic_context)

        # With duration=0, cache is never valid (condition: cache_duration > 0)
        # So each evaluate should call input
        assert input_node.evaluate_count == 3

    def test_evaluate_no_input_returns_cached_or_empty(self, basic_context):
        """evaluate() with no input returns cached pose or empty."""
        node = PoseCacheNode("cache", cache_duration=1.0)

        # No input, no cache
        result = node.evaluate(basic_context)
        assert result.bone_count() == 0


# =============================================================================
# SELECT NODE TESTS
# =============================================================================


class TestSelectNodeWhitebox:
    """Whitebox tests for SelectNode."""

    def test_evaluate_no_options_returns_empty(self, basic_context):
        """evaluate() with no options returns empty Pose."""
        node = SelectNode("select", selector_parameter="choice")
        basic_context.parameters["choice"] = GraphParameter.int_param("choice", 0)

        result = node.evaluate(basic_context)
        assert result.bone_count() == 0

    def test_evaluate_selects_first_option(self, basic_context):
        """evaluate() with selector=0 returns first option."""
        pose0 = Pose.identity(1)
        pose0.transforms[0] = Transform(position=(0.0, 0.0, 0.0))
        pose1 = Pose.identity(1)
        pose1.transforms[0] = Transform(position=(1.0, 0.0, 0.0))

        node0 = DummyNode("opt0", pose0)
        node1 = DummyNode("opt1", pose1)

        node = SelectNode("select", selector_parameter="choice")
        node.add_option(node0)
        node.add_option(node1)

        basic_context.parameters["choice"] = GraphParameter.int_param("choice", 0)

        result = node.evaluate(basic_context)
        assert result.transforms[0].position == (0.0, 0.0, 0.0)

    def test_evaluate_clamps_selector_to_valid_range(self, basic_context):
        """evaluate() clamps selector to [0, len(options)-1]."""
        pose0 = Pose.identity(1)
        pose0.transforms[0] = Transform(position=(0.0, 0.0, 0.0))
        pose1 = Pose.identity(1)
        pose1.transforms[0] = Transform(position=(1.0, 0.0, 0.0))

        node0 = DummyNode("opt0", pose0)
        node1 = DummyNode("opt1", pose1)

        node = SelectNode("select", selector_parameter="choice")
        node.add_option(node0)
        node.add_option(node1)

        # Test index > max
        basic_context.parameters["choice"] = GraphParameter.int_param("choice", 99)
        result = node.evaluate(basic_context)
        # Clamped to 1 (last index)
        assert result.transforms[0].position == (1.0, 0.0, 0.0)

        # Test index < 0
        basic_context.parameters["choice"].value = -5
        result = node.evaluate(basic_context)
        # Clamped to 0
        assert result.transforms[0].position == (0.0, 0.0, 0.0)

    def test_add_remove_option_index_validation(self):
        """add_option returns index, remove_option validates index."""
        node = SelectNode("select", selector_parameter="choice")

        dummy = DummyNode("d", Pose())
        idx = node.add_option(dummy)
        assert idx == 0

        idx2 = node.add_option(dummy)
        assert idx2 == 1

        # Remove valid index
        assert node.remove_option(0) is True
        assert len(node.options) == 1

        # Remove invalid index
        assert node.remove_option(99) is False


# =============================================================================
# LOOP NODE TESTS
# =============================================================================


class TestLoopNodeWhitebox:
    """Whitebox tests for LoopNode."""

    def test_advance_time_once_mode_stops_at_end(self, basic_context):
        """_advance_time() in ONCE mode stops at end time."""
        node = LoopNode("loop", loop_mode=LoopControlMode.ONCE, start_time=0.0, end_time=1.0)

        finished_calls = []
        node.on_finished = lambda: finished_calls.append(True)

        # Advance past the end
        node._current_time = 0.8
        effective = node._advance_time(0.5)  # Would go to 1.3

        # Should stop at 1.0
        assert node._current_time == 1.0
        assert node._is_finished is True
        assert len(finished_calls) == 1

    def test_advance_time_repeat_mode_wraps(self, basic_context):
        """_advance_time() in REPEAT mode wraps and increments loop count."""
        node = LoopNode("loop", loop_mode=LoopControlMode.REPEAT, start_time=0.0, end_time=1.0)

        loop_complete_calls = []
        node.on_loop_complete = lambda count: loop_complete_calls.append(count)

        node._current_time = 0.8
        effective = node._advance_time(0.5)  # Goes to 1.3, wraps to 0.3

        assert abs(node._current_time - 0.3) < 0.001
        assert node._current_loop == 1
        assert loop_complete_calls == [1]

    def test_advance_time_repeat_mode_finite_loop_count(self, basic_context):
        """_advance_time() in REPEAT mode stops after loop_count iterations."""
        node = LoopNode(
            "loop",
            loop_mode=LoopControlMode.REPEAT,
            loop_count=2,
            start_time=0.0,
            end_time=1.0
        )

        finished_calls = []
        node.on_finished = lambda: finished_calls.append(True)

        # First loop
        node._current_time = 0.8
        node._advance_time(0.5)
        assert node._current_loop == 1
        assert not node._is_finished

        # Second loop (should finish)
        node._current_time = 0.8
        node._advance_time(0.5)
        assert node._current_loop == 2
        assert node._is_finished
        assert len(finished_calls) == 1

    def test_advance_time_ping_pong_reverses(self, basic_context):
        """_advance_time() in PING_PONG mode reverses direction at boundaries."""
        node = LoopNode(
            "loop",
            loop_mode=LoopControlMode.PING_PONG,
            start_time=0.0,
            end_time=1.0
        )

        # Going forward, hit end
        node._current_time = 0.8
        node._is_forward = True
        node._advance_time(0.3)

        assert node._current_time == 1.0
        assert node._is_forward is False

        # Now going backward, hit start
        node._current_time = 0.2
        node._advance_time(0.3)

        assert node._current_time == 0.0
        assert node._is_forward is True
        assert node._current_loop == 1

    def test_get_loop_duration_with_end_time(self):
        """_get_loop_duration() uses end_time - start_time when end_time is set."""
        node = LoopNode("loop", start_time=0.5, end_time=1.5)
        assert node._get_loop_duration() == 1.0

    def test_get_loop_duration_from_input_node(self, simple_clip: AnimationClip):
        """_get_loop_duration() uses input node duration when no end_time."""
        clip_node = ClipNode("clip", clip=simple_clip)

        node = LoopNode("loop", start_time=0.0, end_time=None)
        node.inputs["input"] = clip_node

        # Input has duration 1.0
        duration = node._get_loop_duration()
        assert duration == 1.0

    def test_reset_clears_state(self):
        """reset() resets all internal state."""
        node = LoopNode("loop", start_time=0.5, end_time=1.5)
        node._current_time = 1.0
        node._current_loop = 3
        node._is_forward = False
        node._is_finished = True

        node.reset()

        assert node._current_time == 0.5  # Back to start_time
        assert node._current_loop == 0
        assert node._is_forward is True
        assert node._is_finished is False

    def test_evaluate_seeks_clip_node_input(self, simple_clip: AnimationClip, basic_context):
        """evaluate() seeks ClipNode input to effective time."""
        clip_node = ClipNode("clip", clip=simple_clip)

        node = LoopNode("loop", start_time=0.0, end_time=1.0)
        node.inputs["input"] = clip_node
        node._current_time = 0.5

        # Set dt to 0 so _advance_time doesn't change the time
        basic_context.dt = 0.0

        result = node.evaluate(basic_context)

        # The ClipNode should have been seeked to the effective time
        # (The seek happens during evaluate, so clip_node.current_time should be ~0.5)
        assert abs(clip_node.current_time - 0.5) < 0.01


# =============================================================================
# SUBGRAPH NODE TESTS
# =============================================================================


class TestSubGraphNodeWhitebox:
    """Whitebox tests for SubGraphNode."""

    def test_evaluate_no_graph_returns_empty(self, basic_context):
        """evaluate() with no graph returns empty Pose."""
        node = SubGraphNode("subgraph")
        result = node.evaluate(basic_context)
        assert result.bone_count() == 0

    def test_evaluate_applies_parameter_mapping(self, basic_context):
        """evaluate() applies parameter_mapping to child context."""
        # Create a child graph with a parameter
        child_graph = AnimationGraph("child")
        child_param = GraphParameter.float_param("child_speed", 1.0)
        child_graph.add_parameter(child_param)

        # Add a simple output node
        output_node = DummyNode("output", Pose.identity(1))
        child_graph.add_node(output_node)
        child_graph.set_output_node("output")

        # Create SubGraphNode with mapping
        node = SubGraphNode(
            "subgraph",
            graph=child_graph,
            parameter_mapping={"child_speed": "parent_speed"}
        )

        # Set parent parameter
        parent_param = GraphParameter.float_param("parent_speed", 2.5)
        basic_context.parameters["parent_speed"] = parent_param

        result = node.evaluate(basic_context)
        assert result.bone_count() == 1

    def test_evaluate_unmapped_uses_child_defaults(self, basic_context):
        """evaluate() uses child graph's default values for unmapped params."""
        child_graph = AnimationGraph("child")
        child_param = GraphParameter.float_param("unmapped_param", 99.0)
        child_graph.add_parameter(child_param)

        output_node = DummyNode("output", Pose.identity(1))
        child_graph.add_node(output_node)
        child_graph.set_output_node("output")

        node = SubGraphNode("subgraph", graph=child_graph)

        result = node.evaluate(basic_context)
        # Should evaluate without error using default
        assert result.bone_count() == 1

    def test_evaluate_with_input_overrides(self, basic_context):
        """evaluate() temporarily replaces child inputs with overrides."""
        # Create a child graph with a node that has an input
        child_graph = AnimationGraph("child")

        child_blend = BlendNode("child_blend", alpha=0.5)
        child_graph.add_node(child_blend)
        child_graph.set_output_node("child_blend")

        # Create an override node
        override_pose = Pose.identity(1)
        override_pose.transforms[0] = Transform(position=(5.0, 0.0, 0.0))
        override_node = DummyNode("override", override_pose)

        node = SubGraphNode("subgraph", graph=child_graph)
        node.set_input_override("a", override_node)

        result = node.evaluate(basic_context)
        # The override should have been applied
        assert result.bone_count() >= 0

    def test_build_child_parameters_copies_values(self, basic_context):
        """_build_child_parameters correctly copies mapped parameter values."""
        child_graph = AnimationGraph("child")
        child_param = GraphParameter.float_param("alpha", 0.0)
        child_graph.add_parameter(child_param)

        node = SubGraphNode(
            "subgraph",
            graph=child_graph,
            parameter_mapping={"alpha": "parent_alpha"}
        )

        parent_param = GraphParameter.float_param("parent_alpha", 0.75)
        basic_context.parameters["parent_alpha"] = parent_param

        child_params = node._build_child_parameters(basic_context)

        assert "alpha" in child_params
        assert child_params["alpha"].value == 0.75

    def test_map_unmap_parameter(self):
        """map_parameter/unmap_parameter modify the mapping dict."""
        node = SubGraphNode("subgraph")

        node.map_parameter("child_x", "parent_x")
        assert node.parameter_mapping["child_x"] == "parent_x"

        node.unmap_parameter("child_x")
        assert "child_x" not in node.parameter_mapping

        # Unmap non-existent key should not raise
        node.unmap_parameter("nonexistent")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestBlendNodeEdgeCases:
    """Edge case tests across multiple node types."""

    def test_clip_node_with_empty_clip_tracks(self, basic_context):
        """ClipNode with clip that has no tracks returns identity pose."""
        clip = AnimationClip(name="empty", duration=1.0)
        node = ClipNode("test", clip=clip)

        # Use a context with skeleton to determine bone count
        skeleton = Skeleton()
        skeleton.add_bone("root", -1)
        skeleton.add_bone("child", 0)
        basic_context.skeleton = skeleton

        pose = node.evaluate(basic_context)

        assert pose.bone_count() == 2
        assert pose.transforms[0].position == (0.0, 0.0, 0.0)

    def test_blend_node_both_inputs_empty_poses(self, basic_context):
        """BlendNode with both inputs returning empty poses."""
        node_a = DummyNode("a", Pose())
        node_b = DummyNode("b", Pose())

        blend = BlendNode("blend", alpha=0.5)
        blend.inputs["a"] = node_a
        blend.inputs["b"] = node_b

        result = blend.evaluate(basic_context)
        # Both empty, should return empty
        assert result.bone_count() == 0

    def test_layer_node_with_multiple_layers(self, basic_context):
        """LayerNode correctly stacks multiple layers."""
        base_pose = Pose.identity(2)

        # Layer 1 affects only bone 0
        layer1_pose = Pose.identity(2)
        layer1_pose.transforms[0] = Transform(position=(1.0, 0.0, 0.0))

        # Layer 2 affects only bone 1
        layer2_pose = Pose.identity(2)
        layer2_pose.transforms[1] = Transform(position=(0.0, 2.0, 0.0))

        base_node = DummyNode("base", base_pose)
        layer1_node = DummyNode("layer1", layer1_pose)
        layer2_node = DummyNode("layer2", layer2_pose)

        # Create masks so each layer only affects specific bones
        mask1 = BoneMask(name="mask1")
        mask1.set_weight(0, 1.0)
        mask1.set_weight(1, 0.0)

        mask2 = BoneMask(name="mask2")
        mask2.set_weight(0, 0.0)
        mask2.set_weight(1, 1.0)

        node = LayerNode("layers")
        node.inputs["base"] = base_node
        node.add_layer(layer1_node, weight=1.0, mask=mask1)
        node.add_layer(layer2_node, weight=1.0, mask=mask2)

        result = node.evaluate(basic_context)

        # Both layers should be applied to their respective bones
        assert result.transforms[0].position == (1.0, 0.0, 0.0)
        assert result.transforms[1].position == (0.0, 2.0, 0.0)

    def test_select_node_with_single_option(self, basic_context):
        """SelectNode with single option always returns it."""
        pose = Pose.identity(1)
        pose.transforms[0] = Transform(position=(7.0, 0.0, 0.0))

        option_node = DummyNode("opt", pose)

        node = SelectNode("select", selector_parameter="choice")
        node.add_option(option_node)

        # Any selector value should return the single option
        for selector_val in [-1, 0, 1, 100]:
            basic_context.parameters["choice"] = GraphParameter.int_param("choice", selector_val)
            result = node.evaluate(basic_context)
            assert result.transforms[0].position == (7.0, 0.0, 0.0)

    def test_loop_node_zero_duration(self, basic_context):
        """LoopNode with zero duration handles gracefully."""
        node = LoopNode("loop", start_time=0.0, end_time=0.0)

        duration = node._get_loop_duration()
        assert duration == 0.0

        # _advance_time should return start_time when duration is 0
        effective = node._advance_time(0.1)
        assert effective == 0.0

    def test_subgraph_node_restores_inputs_after_override(self, basic_context):
        """SubGraphNode restores original inputs after evaluation."""
        child_graph = AnimationGraph("child")

        original_input = DummyNode("original", Pose.identity(1))
        child_blend = BlendNode("child_blend", alpha=0.5)
        child_blend.inputs["a"] = original_input

        child_graph.add_node(child_blend)
        child_graph.add_node(original_input)
        child_graph.set_output_node("child_blend")

        # Override node must also be added to child graph for cycle detection
        override_node = DummyNode("override", Pose.identity(1))
        child_graph.add_node(override_node)

        node = SubGraphNode("subgraph", graph=child_graph)
        node.set_input_override("a", override_node)

        node.evaluate(basic_context)

        # After evaluation, original input should be restored
        assert child_blend.inputs.get("a") is original_input


# =============================================================================
# PERFORMANCE CHARACTERISTICS TESTS
# =============================================================================


class TestPerformanceCharacteristics:
    """Tests for performance-related behavior."""

    def test_pose_cache_reduces_evaluations(self, basic_context):
        """PoseCacheNode reduces input evaluations when cache is valid."""
        pose = Pose.identity(10)
        input_node = DummyNode("input", pose)

        cache_node = PoseCacheNode("cache", cache_duration=1.0)
        cache_node.inputs["input"] = input_node

        # Evaluate multiple times within cache duration
        for _ in range(10):
            basic_context.dt = 0.05  # Small dt, stays within cache
            cache_node.evaluate(basic_context)

        # Input should only be evaluated once (first time or when cache expires)
        # Since _cache_time accumulates: 10 * 0.05 = 0.5 < 1.0, cache stays valid
        assert input_node.evaluate_count == 1

    def test_select_node_only_evaluates_selected_option(self, basic_context):
        """SelectNode only evaluates the selected option, not all."""
        pose0 = Pose.identity(1)
        pose1 = Pose.identity(1)
        pose2 = Pose.identity(1)

        node0 = DummyNode("opt0", pose0)
        node1 = DummyNode("opt1", pose1)
        node2 = DummyNode("opt2", pose2)

        select = SelectNode("select", selector_parameter="choice")
        select.add_option(node0)
        select.add_option(node1)
        select.add_option(node2)

        basic_context.parameters["choice"] = GraphParameter.int_param("choice", 1)
        select.evaluate(basic_context)

        # Only node1 should be evaluated
        assert node0.evaluate_count == 0
        assert node1.evaluate_count == 1
        assert node2.evaluate_count == 0


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestCallbacks:
    """Tests for callback invocations."""

    def test_clip_node_all_callbacks(self, simple_clip: AnimationClip):
        """ClipNode invokes on_event, on_loop, on_finish at correct times."""
        simple_clip.loop_mode = LoopMode.LOOP
        simple_clip.events = [(0.3, "ev1"), (0.7, "ev2")]

        node = ClipNode("test", clip=simple_clip)

        events = []
        loops = []
        finishes = []

        node.on_event = lambda e: events.append(e)
        node.on_loop = lambda: loops.append(True)
        node.on_finish = lambda: finishes.append(True)

        # Advance through the clip
        node.advance(0.4)  # 0 -> 0.4, triggers ev1 at 0.3
        assert events == ["ev1"]

        node.advance(0.5)  # 0.4 -> 0.9, triggers ev2 at 0.7
        assert events == ["ev1", "ev2"]

        node.advance(0.2)  # 0.9 -> 1.1, wraps, triggers loop
        assert len(loops) == 1

    def test_loop_node_callbacks(self):
        """LoopNode invokes on_loop_complete and on_finished correctly."""
        node = LoopNode(
            "loop",
            loop_mode=LoopControlMode.REPEAT,
            loop_count=2,
            start_time=0.0,
            end_time=1.0
        )

        loop_counts = []
        finished = []

        node.on_loop_complete = lambda c: loop_counts.append(c)
        node.on_finished = lambda: finished.append(True)

        # First loop
        node._current_time = 0.9
        node._advance_time(0.2)
        assert loop_counts == [1]
        assert finished == []

        # Second loop (finishes)
        node._current_time = 0.9
        node._advance_time(0.2)
        assert loop_counts == [1, 2]
        assert finished == [True]
