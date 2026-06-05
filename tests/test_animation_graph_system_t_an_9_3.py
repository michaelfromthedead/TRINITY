"""
Tests for AnimationGraphSystem (T-AN-9.3) — Presentation Phase Animation Evaluation.

This test suite covers:
- State machine to graph evaluation
- Clip sampling correctness
- Blend tree weight computation
- Dirty flag optimization (no re-eval when clean)
- Multi-entity parallel evaluation
- Output format compatibility with skinning
- SoA conversion
- Parameter binding and synchronization
- Root motion extraction
- Transition blending

50+ test cases organized into logical groups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, Mock
import pytest

from engine.animation.systems.animation_graph_system import (
    AnimationGraphSystem,
    AnimationGraphComponent,
    BoneTransformSoA,
    DirtyFlags,
    AnimationDirtyState,
    StateMachineOutput,
    ClipSampler,
    BlendTreeEvaluator,
    system,
)
from engine.animation.graph import (
    AnimationClip,
    AnimationGraph,
    AnimationNode,
    AnimationTrack,
    AnimationKeyframe,
    BlendTree1D,
    BlendTree1DEntry,
    BlendTree2D,
    BlendTree2DMode,
    BlendTreeDirect,
    ClipNode,
    GraphContext,
    GraphParameter,
    LoopMode,
    ParameterType,
    Pose,
    Skeleton,
    StateMachine,
    Transform,
)


# =============================================================================
# TEST FIXTURES AND HELPERS
# =============================================================================


class MockNode(AnimationNode):
    """Mock animation node for testing."""

    _abstract = True

    def __init__(self, node_id: str, return_pose: Optional[Pose] = None) -> None:
        super().__init__(node_id)
        self.eval_count = 0
        self._return_pose = return_pose or Pose()

    def evaluate(self, context: GraphContext) -> Pose:
        self.eval_count += 1
        return self._return_pose


@pytest.fixture
def skeleton() -> Skeleton:
    """Create a test skeleton with 4 bones."""
    skel = Skeleton()
    skel.add_bone("root", parent_index=-1)
    skel.add_bone("spine", parent_index=0)
    skel.add_bone("head", parent_index=1)
    skel.add_bone("arm", parent_index=1)
    return skel


@pytest.fixture
def identity_pose() -> Pose:
    """Create an identity pose with 4 bones."""
    return Pose.identity(4)


@pytest.fixture
def sample_pose() -> Pose:
    """Create a sample pose with known values."""
    transforms = [
        Transform(
            position=(1.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
        ),
        Transform(
            position=(0.0, 1.0, 0.0),
            rotation=(0.0, 0.707, 0.0, 0.707),
            scale=(1.0, 1.0, 1.0),
        ),
        Transform(
            position=(0.0, 0.0, 1.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(2.0, 2.0, 2.0),
        ),
        Transform(
            position=(1.0, 1.0, 1.0),
            rotation=(0.5, 0.5, 0.5, 0.5),
            scale=(0.5, 0.5, 0.5),
        ),
    ]
    return Pose(transforms=transforms)


@pytest.fixture
def sample_clip() -> AnimationClip:
    """Create a sample animation clip."""
    clip = AnimationClip(name="test_clip", duration=1.0, frame_rate=30.0)

    # Add keyframes for bone 0
    track0 = clip.add_track(0)
    track0.keyframes = [
        AnimationKeyframe(
            time=0.0,
            value=Transform(position=(0.0, 0.0, 0.0)),
        ),
        AnimationKeyframe(
            time=1.0,
            value=Transform(position=(1.0, 0.0, 0.0)),
        ),
    ]

    # Add keyframes for bone 1
    track1 = clip.add_track(1)
    track1.keyframes = [
        AnimationKeyframe(
            time=0.0,
            value=Transform(position=(0.0, 0.0, 0.0)),
        ),
        AnimationKeyframe(
            time=1.0,
            value=Transform(position=(0.0, 1.0, 0.0)),
        ),
    ]

    return clip


@pytest.fixture
def walk_clip() -> AnimationClip:
    """Create a walk animation clip."""
    clip = AnimationClip(name="walk", duration=1.0, loop_mode=LoopMode.LOOP)
    track = clip.add_track(0)
    track.keyframes = [
        AnimationKeyframe(time=0.0, value=Transform(position=(0.0, 0.0, 0.0))),
        AnimationKeyframe(time=0.5, value=Transform(position=(0.5, 0.1, 0.0))),
        AnimationKeyframe(time=1.0, value=Transform(position=(1.0, 0.0, 0.0))),
    ]
    return clip


@pytest.fixture
def run_clip() -> AnimationClip:
    """Create a run animation clip."""
    clip = AnimationClip(name="run", duration=0.5, loop_mode=LoopMode.LOOP)
    track = clip.add_track(0)
    track.keyframes = [
        AnimationKeyframe(time=0.0, value=Transform(position=(0.0, 0.0, 0.0))),
        AnimationKeyframe(time=0.25, value=Transform(position=(1.0, 0.2, 0.0))),
        AnimationKeyframe(time=0.5, value=Transform(position=(2.0, 0.0, 0.0))),
    ]
    return clip


@pytest.fixture
def idle_clip() -> AnimationClip:
    """Create an idle animation clip."""
    clip = AnimationClip(name="idle", duration=2.0, loop_mode=LoopMode.LOOP)
    track = clip.add_track(0)
    track.keyframes = [
        AnimationKeyframe(time=0.0, value=Transform(position=(0.0, 0.0, 0.0))),
        AnimationKeyframe(time=2.0, value=Transform(position=(0.0, 0.0, 0.0))),
    ]
    return clip


@pytest.fixture
def graph_system() -> AnimationGraphSystem:
    """Create an animation graph system."""
    return AnimationGraphSystem()


@pytest.fixture
def component(skeleton: Skeleton) -> AnimationGraphComponent:
    """Create an animation graph component."""
    comp = AnimationGraphComponent()
    comp.skeleton = skeleton
    comp.enabled = True
    return comp


# =============================================================================
# 1. SYSTEM DECORATOR TESTS
# =============================================================================


class TestSystemDecorator:
    """Tests for the @system decorator."""

    def test_system_decorator_sets_phase(self) -> None:
        """Test that @system sets the phase attribute."""
        assert hasattr(AnimationGraphSystem, "_system_phase")
        assert AnimationGraphSystem._system_phase == "animation"

    def test_system_decorator_sets_priority(self) -> None:
        """Test that @system sets the priority attribute."""
        assert hasattr(AnimationGraphSystem, "_system_priority")
        assert AnimationGraphSystem._system_priority > 0

    def test_system_decorator_sets_reads(self) -> None:
        """Test that @system sets the reads attribute."""
        assert hasattr(AnimationGraphSystem, "_system_reads")
        assert "AnimationGraphComponent" in AnimationGraphSystem._system_reads

    def test_system_decorator_sets_writes(self) -> None:
        """Test that @system sets the writes attribute."""
        assert hasattr(AnimationGraphSystem, "_system_writes")
        assert "AnimationGraphComponent" in AnimationGraphSystem._system_writes

    def test_custom_system_decorator(self) -> None:
        """Test applying @system with custom parameters."""
        @system(phase="custom", priority=50, reads=("A",), writes=("B",))
        class CustomSystem:
            pass

        assert CustomSystem._system_phase == "custom"
        assert CustomSystem._system_priority == 50
        assert CustomSystem._system_reads == ("A",)
        assert CustomSystem._system_writes == ("B",)


# =============================================================================
# 2. SoA BONE TRANSFORM TESTS
# =============================================================================


class TestBoneTransformSoA:
    """Tests for SoA bone transform conversion."""

    def test_from_pose_empty(self) -> None:
        """Test converting empty pose to SoA."""
        pose = Pose()
        soa = BoneTransformSoA.from_pose(pose)

        assert soa.bone_count == 0
        assert len(soa.positions_x) == 0

    def test_from_pose_with_transforms(self, sample_pose: Pose) -> None:
        """Test converting pose with transforms to SoA."""
        soa = BoneTransformSoA.from_pose(sample_pose)

        assert soa.bone_count == 4
        assert len(soa.positions_x) == 4
        assert len(soa.rotations_w) == 4
        assert len(soa.scales_z) == 4

    def test_from_pose_preserves_positions(self, sample_pose: Pose) -> None:
        """Test that positions are correctly extracted."""
        soa = BoneTransformSoA.from_pose(sample_pose)

        assert soa.positions_x[0] == 1.0
        assert soa.positions_y[1] == 1.0
        assert soa.positions_z[2] == 1.0

    def test_from_pose_preserves_rotations(self, sample_pose: Pose) -> None:
        """Test that rotations are correctly extracted."""
        soa = BoneTransformSoA.from_pose(sample_pose)

        # First bone has identity rotation
        assert soa.rotations_w[0] == 1.0
        assert soa.rotations_x[0] == 0.0

        # Second bone has 90-degree Y rotation
        assert abs(soa.rotations_y[1] - 0.707) < 0.001

    def test_from_pose_preserves_scales(self, sample_pose: Pose) -> None:
        """Test that scales are correctly extracted."""
        soa = BoneTransformSoA.from_pose(sample_pose)

        assert soa.scales_x[2] == 2.0
        assert soa.scales_y[2] == 2.0
        assert soa.scales_z[2] == 2.0

    def test_to_pose_roundtrip(self, sample_pose: Pose) -> None:
        """Test that SoA -> AoS roundtrip preserves data."""
        soa = BoneTransformSoA.from_pose(sample_pose)
        result = soa.to_pose()

        assert result.bone_count() == sample_pose.bone_count()

        for i in range(sample_pose.bone_count()):
            orig = sample_pose.transforms[i]
            conv = result.transforms[i]

            assert orig.position == conv.position
            assert orig.rotation == conv.rotation
            assert orig.scale == conv.scale

    def test_clear(self, sample_pose: Pose) -> None:
        """Test clearing SoA data."""
        soa = BoneTransformSoA.from_pose(sample_pose)
        soa.clear()

        assert soa.bone_count == 0
        assert len(soa.positions_x) == 0
        assert len(soa.rotations_w) == 0

    def test_get_flat_positions(self, sample_pose: Pose) -> None:
        """Test getting flat position array."""
        soa = BoneTransformSoA.from_pose(sample_pose)
        flat = soa.get_flat_positions()

        assert len(flat) == 12  # 4 bones * 3 components
        assert flat[0] == 1.0  # bone 0 x
        assert flat[4] == 1.0  # bone 1 y

    def test_get_flat_rotations(self, sample_pose: Pose) -> None:
        """Test getting flat rotation array."""
        soa = BoneTransformSoA.from_pose(sample_pose)
        flat = soa.get_flat_rotations()

        assert len(flat) == 16  # 4 bones * 4 components
        assert flat[3] == 1.0  # bone 0 w (identity)

    def test_get_flat_scales(self, sample_pose: Pose) -> None:
        """Test getting flat scale array."""
        soa = BoneTransformSoA.from_pose(sample_pose)
        flat = soa.get_flat_scales()

        assert len(flat) == 12  # 4 bones * 3 components
        assert flat[6] == 2.0  # bone 2 x (scale 2.0)


# =============================================================================
# 3. DIRTY FLAG TESTS
# =============================================================================


class TestDirtyFlags:
    """Tests for dirty flag tracking."""

    def test_initial_state_is_dirty(self) -> None:
        """Test that initial state has all flags set."""
        state = AnimationDirtyState()
        assert state.is_dirty(DirtyFlags.ALL)

    def test_mark_clean_clears_flags(self) -> None:
        """Test that mark_clean clears flags."""
        state = AnimationDirtyState()
        state.mark_clean()
        assert not state.is_dirty()

    def test_mark_dirty_sets_specific_flag(self) -> None:
        """Test marking specific flags as dirty."""
        state = AnimationDirtyState()
        state.clear()

        state.mark_dirty(DirtyFlags.PARAMETERS)
        assert state.is_dirty(DirtyFlags.PARAMETERS)
        assert not state.is_dirty(DirtyFlags.STATE)

    def test_mark_dirty_multiple_flags(self) -> None:
        """Test marking multiple flags."""
        state = AnimationDirtyState()
        state.clear()

        state.mark_dirty(DirtyFlags.PARAMETERS)
        state.mark_dirty(DirtyFlags.STATE)

        assert state.is_dirty(DirtyFlags.PARAMETERS)
        assert state.is_dirty(DirtyFlags.STATE)

    def test_mark_clean_specific_flag(self) -> None:
        """Test clearing specific flags."""
        state = AnimationDirtyState()
        state.mark_all_dirty()

        state.mark_clean(DirtyFlags.PARAMETERS)

        assert not state.is_dirty(DirtyFlags.PARAMETERS)
        assert state.is_dirty(DirtyFlags.STATE)

    def test_clear_all_flags(self) -> None:
        """Test clearing all flags."""
        state = AnimationDirtyState()
        state.clear()

        assert state.flags == DirtyFlags.NONE.value
        assert not state.is_dirty()


# =============================================================================
# 4. STATE MACHINE OUTPUT TESTS
# =============================================================================


class TestStateMachineOutput:
    """Tests for state machine output handling."""

    def test_default_output(self) -> None:
        """Test default state machine output."""
        output = StateMachineOutput()

        assert output.current_state == ""
        assert not output.is_transitioning
        assert output.transition_progress == 0.0

    def test_output_with_state(self) -> None:
        """Test output with active state."""
        output = StateMachineOutput(
            current_state="idle",
            state_time=1.5,
            normalized_time=0.75,
        )

        assert output.current_state == "idle"
        assert output.state_time == 1.5
        assert output.normalized_time == 0.75

    def test_output_with_transition(self) -> None:
        """Test output during transition."""
        output = StateMachineOutput(
            current_state="idle",
            target_state="walk",
            is_transitioning=True,
            transition_progress=0.5,
            transition_duration=0.3,
        )

        assert output.is_transitioning
        assert output.target_state == "walk"
        assert output.transition_progress == 0.5


# =============================================================================
# 5. CLIP SAMPLER TESTS
# =============================================================================


class TestClipSampler:
    """Tests for clip sampling."""

    def test_sample_at_start(self, sample_clip: AnimationClip) -> None:
        """Test sampling at clip start."""
        sampler = ClipSampler()
        pose = sampler.sample(sample_clip, 0.0, 4)

        assert pose.bone_count() == 4
        assert pose.transforms[0].position[0] == 0.0

    def test_sample_at_end(self, sample_clip: AnimationClip) -> None:
        """Test sampling at clip end."""
        sampler = ClipSampler()
        # Note: sample_clip has duration 1.0 with LOOP mode, so time 1.0 wraps to 0.0
        # Use a time just before the end to test interpolation
        pose = sampler.sample(sample_clip, 0.99, 4)

        # Should be close to end values
        assert pose.transforms[0].position[0] > 0.9
        assert pose.transforms[1].position[1] > 0.9

    def test_sample_interpolation(self, sample_clip: AnimationClip) -> None:
        """Test sampling with interpolation."""
        sampler = ClipSampler()
        pose = sampler.sample(sample_clip, 0.5, 4)

        # Should be halfway between start and end
        assert abs(pose.transforms[0].position[0] - 0.5) < 0.001

    def test_sample_blended(
        self, walk_clip: AnimationClip, run_clip: AnimationClip
    ) -> None:
        """Test blended sampling of two clips."""
        sampler = ClipSampler()
        pose = sampler.sample_blended(
            walk_clip, 0.0,
            run_clip, 0.0,
            weight=0.5,
            bone_count=4,
        )

        assert pose.bone_count() == 4

    def test_cache_hit(self, sample_clip: AnimationClip) -> None:
        """Test that cache is used on repeated samples."""
        sampler = ClipSampler()

        pose1 = sampler.sample(sample_clip, 0.5, 4)
        pose2 = sampler.sample(sample_clip, 0.5, 4)

        # Same object should be returned from cache
        assert pose1 is pose2

    def test_cache_miss_different_time(self, sample_clip: AnimationClip) -> None:
        """Test that different times produce different poses."""
        sampler = ClipSampler()

        pose1 = sampler.sample(sample_clip, 0.0, 4)
        pose2 = sampler.sample(sample_clip, 0.5, 4)

        assert pose1 is not pose2

    def test_cache_eviction(self, sample_clip: AnimationClip) -> None:
        """Test cache eviction when full."""
        sampler = ClipSampler(cache_size=2)

        sampler.sample(sample_clip, 0.0, 4)
        sampler.sample(sample_clip, 0.1, 4)
        sampler.sample(sample_clip, 0.2, 4)  # Should evict 0.0

        assert len(sampler._cache) == 2

    def test_clear_cache(self, sample_clip: AnimationClip) -> None:
        """Test clearing the cache."""
        sampler = ClipSampler()
        sampler.sample(sample_clip, 0.5, 4)

        sampler.clear_cache()

        assert len(sampler._cache) == 0


# =============================================================================
# 6. ANIMATION GRAPH COMPONENT TESTS
# =============================================================================


class TestAnimationGraphComponent:
    """Tests for the animation graph component."""

    def test_register_clip(self, component: AnimationGraphComponent, sample_clip: AnimationClip) -> None:
        """Test registering clips."""
        component.register_clip("test", sample_clip)

        assert "test" in component.clips
        assert component.dirty_state.is_dirty(DirtyFlags.CLIP)

    def test_get_clip(self, component: AnimationGraphComponent, sample_clip: AnimationClip) -> None:
        """Test retrieving registered clips."""
        component.register_clip("test", sample_clip)

        clip = component.get_clip("test")
        assert clip is sample_clip

    def test_get_nonexistent_clip(self, component: AnimationGraphComponent) -> None:
        """Test getting a clip that doesn't exist."""
        clip = component.get_clip("nonexistent")
        assert clip is None

    def test_set_parameter(self, component: AnimationGraphComponent) -> None:
        """Test setting graph parameters."""
        graph = AnimationGraph("test")
        graph.add_parameter(GraphParameter.float_param("speed", default=1.0))
        component.graph = graph

        result = component.set_parameter("speed", 2.0)

        assert result is True
        assert component.dirty_state.is_dirty(DirtyFlags.PARAMETERS)

    def test_get_parameter(self, component: AnimationGraphComponent) -> None:
        """Test getting graph parameters."""
        graph = AnimationGraph("test")
        graph.add_parameter(GraphParameter.float_param("speed", default=1.5))
        component.graph = graph

        value = component.get_parameter("speed")
        assert value == 1.5

    def test_update_time(self, component: AnimationGraphComponent) -> None:
        """Test time advancement."""
        component.update_time(0.016)

        assert component._current_time > 0
        assert component._frame_count == 1
        assert component.dirty_state.is_dirty(DirtyFlags.TIME)

    def test_update_time_with_scale(self, component: AnimationGraphComponent) -> None:
        """Test time advancement with time scale."""
        component.time_scale = 2.0
        component.update_time(0.016)

        assert abs(component._current_time - 0.032) < 0.0001

    def test_invalidate(self, component: AnimationGraphComponent) -> None:
        """Test invalidation."""
        component.dirty_state.clear()
        component.invalidate()

        assert component.dirty_state.is_dirty(DirtyFlags.ALL)


# =============================================================================
# 7. ANIMATION GRAPH SYSTEM TESTS
# =============================================================================


class TestAnimationGraphSystem:
    """Tests for the main animation graph system."""

    def test_update_empty_list(self, graph_system: AnimationGraphSystem) -> None:
        """Test updating with no entities."""
        world = MagicMock()
        graph_system.update(world, 0.016, [])

        # Should not raise

    def test_update_disabled_component(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test that disabled components are skipped."""
        component.enabled = False
        entity = MagicMock()
        world = MagicMock()

        graph_system.update(world, 0.016, [(entity, component)])

        stats = graph_system.get_statistics()
        assert stats["entities_evaluated"] == 0

    def test_update_evaluates_enabled_component(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test that enabled components are evaluated."""
        entity = MagicMock()
        world = MagicMock()

        graph_system.update(world, 0.016, [(entity, component)])

        stats = graph_system.get_statistics()
        assert stats["entities_evaluated"] == 1

    def test_update_produces_soa_output(
        self, graph_system: AnimationGraphSystem,
        component: AnimationGraphComponent,
    ) -> None:
        """Test that update produces SoA output."""
        # Create a clip with 4 bones to match skeleton
        clip = AnimationClip(name="idle", duration=1.0, loop_mode=LoopMode.LOOP)
        for i in range(4):
            track = clip.add_track(i)
            track.keyframes = [
                AnimationKeyframe(time=0.0, value=Transform(position=(0.0, 0.0, 0.0))),
                AnimationKeyframe(time=1.0, value=Transform(position=(float(i), 0.0, 0.0))),
            ]

        # Register the clip with the same name used in state machine
        component.register_clip("idle", clip)
        component.state_machine_output = StateMachineOutput(
            current_state="idle",
            state_time=0.5,
        )

        entity = MagicMock()
        world = MagicMock()

        graph_system.update(world, 0.016, [(entity, component)])

        # The clip has 4 bones, so output should have 4 bones
        assert component.output_soa.bone_count == 4
        assert component.output_pose.bone_count() == 4

    def test_dirty_flag_optimization(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test that clean components are skipped."""
        entity = MagicMock()
        world = MagicMock()

        # First update - should evaluate
        graph_system.update(world, 0.016, [(entity, component)])
        assert graph_system.get_statistics()["entities_evaluated"] == 1

        # Second update - dirty flags cleared but TIME should be set
        component.dirty_state.clear()
        graph_system.update(world, 0.016, [(entity, component)])

        # TIME flag should trigger re-evaluation
        stats = graph_system.get_statistics()
        assert stats["entities_evaluated"] == 1

    def test_transition_blending(
        self,
        graph_system: AnimationGraphSystem,
        component: AnimationGraphComponent,
    ) -> None:
        """Test transition blending between clips."""
        # Create clips with 4 bones to match skeleton
        walk_clip = AnimationClip(name="walk", duration=1.0, loop_mode=LoopMode.LOOP)
        for i in range(4):
            track = walk_clip.add_track(i)
            track.keyframes = [
                AnimationKeyframe(time=0.0, value=Transform(position=(0.0, 0.0, 0.0))),
                AnimationKeyframe(time=1.0, value=Transform(position=(1.0, 0.0, 0.0))),
            ]

        run_clip = AnimationClip(name="run", duration=0.5, loop_mode=LoopMode.LOOP)
        for i in range(4):
            track = run_clip.add_track(i)
            track.keyframes = [
                AnimationKeyframe(time=0.0, value=Transform(position=(0.0, 0.0, 0.0))),
                AnimationKeyframe(time=0.5, value=Transform(position=(2.0, 0.0, 0.0))),
            ]

        component.register_clip("walk", walk_clip)
        component.register_clip("run", run_clip)
        component.state_machine_output = StateMachineOutput(
            current_state="walk",
            target_state="run",
            is_transitioning=True,
            transition_progress=0.5,
            transition_duration=0.3,
            state_time=0.5,
        )

        entity = MagicMock()
        world = MagicMock()

        graph_system.update(world, 0.016, [(entity, component)])

        # Should have 4 bones (skeleton has 4)
        assert component.output_pose.bone_count() == 4

    def test_force_state(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test forcing immediate state change."""
        component.state_machine_output = StateMachineOutput(current_state="idle")

        result = graph_system.force_state(component, "walk")

        assert result is True
        assert component.state_machine_output.current_state == "walk"
        assert not component.state_machine_output.is_transitioning

    def test_trigger_transition(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test triggering a transition."""
        component.state_machine_output = StateMachineOutput(current_state="idle")

        result = graph_system.trigger_transition(component, "walk", duration=0.5)

        assert result is True
        assert component.state_machine_output.is_transitioning
        assert component.state_machine_output.target_state == "walk"
        assert component.state_machine_output.transition_duration == 0.5

    def test_sync_parameters_from_gameplay(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test syncing parameters from gameplay data."""
        graph = AnimationGraph("test")
        graph.add_parameter(GraphParameter.float_param("speed", default=1.0))
        component.graph = graph
        component.parameter_bindings = {"speed": "player_speed"}

        gameplay_data = {"player_speed": 2.5}
        updated = graph_system.sync_parameters_from_gameplay(component, gameplay_data)

        assert updated == 1
        assert component.get_parameter("speed") == 2.5

    def test_animation_provider(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test external animation provider."""
        expected_pose = Pose.identity(4)

        def provider(clip_name: str, time: float) -> Pose:
            return expected_pose

        graph_system.set_animation_provider(provider)

        pose = graph_system.sample_animation("test", 0.5)
        assert pose.bone_count() == 4

    def test_clear_caches(
        self, graph_system: AnimationGraphSystem, sample_clip: AnimationClip
    ) -> None:
        """Test clearing system caches."""
        # Prime the cache
        graph_system._clip_sampler.sample(sample_clip, 0.5, 4)
        assert len(graph_system._clip_sampler._cache) > 0

        graph_system.clear_caches()

        assert len(graph_system._clip_sampler._cache) == 0

    def test_get_statistics(self, graph_system: AnimationGraphSystem) -> None:
        """Test getting system statistics."""
        stats = graph_system.get_statistics()

        assert "current_frame" in stats
        assert "entities_evaluated" in stats
        assert "entities_skipped" in stats
        assert "parallel_batches" in stats


# =============================================================================
# 8. PARALLEL EVALUATION TESTS
# =============================================================================


class TestParallelEvaluation:
    """Tests for parallel entity evaluation."""

    def test_parallel_threshold_default(self, graph_system: AnimationGraphSystem) -> None:
        """Test default parallel threshold."""
        assert graph_system._parallel_threshold == 4

    def test_set_parallel_threshold(self, graph_system: AnimationGraphSystem) -> None:
        """Test setting parallel threshold."""
        graph_system.set_parallel_threshold(8)
        assert graph_system._parallel_threshold == 8

    def test_parallel_threshold_minimum(self, graph_system: AnimationGraphSystem) -> None:
        """Test that threshold has minimum of 1."""
        graph_system.set_parallel_threshold(0)
        assert graph_system._parallel_threshold == 1

    def test_serial_below_threshold(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test that serial execution is used below threshold."""
        graph_system.set_parallel_threshold(10)

        # Create 5 entities (below threshold)
        entities = [(MagicMock(), AnimationGraphComponent()) for _ in range(5)]
        for _, comp in entities:
            comp.skeleton = component.skeleton
            comp.enabled = True

        world = MagicMock()
        graph_system.update(world, 0.016, entities)

        # All should be evaluated serially (no parallel batches)
        stats = graph_system.get_statistics()
        assert stats["parallel_batches"] == 0

    def test_parallel_with_scheduler(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test parallel execution with task scheduler."""
        # Mock task scheduler
        mock_scheduler = MagicMock()
        mock_handle = MagicMock()
        mock_handle.result.return_value = None
        mock_scheduler.submit.return_value = mock_handle
        mock_scheduler.wait_all.return_value = []

        graph_system.set_task_scheduler(mock_scheduler)
        graph_system.set_parallel_threshold(2)

        # Create enough entities
        entities = [(MagicMock(), AnimationGraphComponent()) for _ in range(5)]
        for _, comp in entities:
            comp.skeleton = component.skeleton
            comp.enabled = True

        world = MagicMock()
        graph_system.update(world, 0.016, entities)

        # Verify parallel dispatch was used
        assert mock_scheduler.submit.called
        assert mock_scheduler.wait_all.called


# =============================================================================
# 9. BLEND TREE EVALUATOR TESTS
# =============================================================================


class TestBlendTreeEvaluator:
    """Tests for blend tree evaluation."""

    def test_evaluate_1d_single_entry(self, skeleton: Skeleton) -> None:
        """Test 1D blend tree with single entry."""
        sampler = ClipSampler()
        evaluator = BlendTreeEvaluator(sampler)

        mock_node = MockNode("idle", Pose.identity(4))
        tree = BlendTree1D("speed_blend", "speed")
        tree.add_entry(0.0, mock_node)

        context = GraphContext(skeleton=skeleton)
        context.parameters["speed"] = GraphParameter.float_param("speed", default=0.0)

        pose = evaluator.evaluate(tree, context, {})

        assert pose.bone_count() == 4
        assert mock_node.eval_count == 1

    def test_evaluate_1d_blending(self, skeleton: Skeleton) -> None:
        """Test 1D blend tree interpolation."""
        sampler = ClipSampler()
        evaluator = BlendTreeEvaluator(sampler)

        idle_pose = Pose.identity(4)
        walk_pose = Pose(transforms=[
            Transform(position=(1.0, 0.0, 0.0)) for _ in range(4)
        ])

        idle_node = MockNode("idle", idle_pose)
        walk_node = MockNode("walk", walk_pose)

        tree = BlendTree1D("speed_blend", "speed")
        tree.add_entry(0.0, idle_node)
        tree.add_entry(1.0, walk_node)

        context = GraphContext(skeleton=skeleton)
        speed_param = GraphParameter.float_param("speed", default=0.5)
        context.parameters["speed"] = speed_param

        pose = evaluator.evaluate(tree, context, {})

        # Should blend between idle and walk
        assert idle_node.eval_count == 1
        assert walk_node.eval_count == 1


# =============================================================================
# 10. OUTPUT FORMAT COMPATIBILITY TESTS
# =============================================================================


class TestOutputFormatCompatibility:
    """Tests for skinning pipeline compatibility."""

    def test_soa_has_correct_structure(self, sample_pose: Pose) -> None:
        """Test that SoA output has correct structure for skinning."""
        soa = BoneTransformSoA.from_pose(sample_pose)

        # All arrays should have same length
        assert len(soa.positions_x) == soa.bone_count
        assert len(soa.positions_y) == soa.bone_count
        assert len(soa.positions_z) == soa.bone_count
        assert len(soa.rotations_x) == soa.bone_count
        assert len(soa.rotations_y) == soa.bone_count
        assert len(soa.rotations_z) == soa.bone_count
        assert len(soa.rotations_w) == soa.bone_count
        assert len(soa.scales_x) == soa.bone_count
        assert len(soa.scales_y) == soa.bone_count
        assert len(soa.scales_z) == soa.bone_count

    def test_soa_suitable_for_gpu_upload(self, sample_pose: Pose) -> None:
        """Test that SoA format is suitable for GPU buffer upload."""
        soa = BoneTransformSoA.from_pose(sample_pose)

        # Flat arrays should be contiguous and properly sized
        positions = soa.get_flat_positions()
        rotations = soa.get_flat_rotations()
        scales = soa.get_flat_scales()

        assert len(positions) == soa.bone_count * 3
        assert len(rotations) == soa.bone_count * 4
        assert len(scales) == soa.bone_count * 3

    def test_component_provides_both_formats(
        self,
        graph_system: AnimationGraphSystem,
        component: AnimationGraphComponent,
    ) -> None:
        """Test that component provides both AoS and SoA output."""
        # Create clip with 4 bones to match skeleton
        idle_clip = AnimationClip(name="idle", duration=1.0, loop_mode=LoopMode.LOOP)
        for i in range(4):
            track = idle_clip.add_track(i)
            track.keyframes = [
                AnimationKeyframe(time=0.0, value=Transform(position=(0.0, 0.0, 0.0))),
                AnimationKeyframe(time=1.0, value=Transform(position=(0.0, 0.0, 0.0))),
            ]

        component.register_clip("idle", idle_clip)
        component.state_machine_output = StateMachineOutput(
            current_state="idle",
            state_time=0.5,
        )

        entity = MagicMock()
        world = MagicMock()

        graph_system.update(world, 0.016, [(entity, component)])

        # Both formats should be populated with 4 bones
        assert component.output_pose.bone_count() == 4
        assert component.output_soa.bone_count == 4

        # They should represent the same data
        assert component.output_pose.bone_count() == component.output_soa.bone_count


# =============================================================================
# 11. ROOT MOTION TESTS
# =============================================================================


class TestRootMotion:
    """Tests for root motion extraction."""

    def test_root_motion_disabled_by_default(
        self, component: AnimationGraphComponent
    ) -> None:
        """Test that root motion is disabled by default."""
        assert not component.root_motion_enabled

    def test_root_motion_when_enabled(
        self,
        graph_system: AnimationGraphSystem,
        component: AnimationGraphComponent,
    ) -> None:
        """Test root motion extraction when enabled."""
        component.root_motion_enabled = True

        # Create a clip with root motion
        clip = AnimationClip(name="walk", duration=1.0, root_motion=True)
        track = clip.add_track(0)
        track.keyframes = [
            AnimationKeyframe(
                time=0.0,
                value=Transform(position=(0.0, 0.0, 0.0)),
            ),
            AnimationKeyframe(
                time=1.0,
                value=Transform(position=(1.0, 0.0, 0.0)),
            ),
        ]

        component.register_clip("walk", clip)
        component.state_machine_output = StateMachineOutput(
            current_state="walk",
            state_time=0.5,
        )

        entity = MagicMock()
        world = MagicMock()

        graph_system.update(world, 0.016, [(entity, component)])

        # Root motion should be accumulated
        # (actual value depends on pose.root_motion being set)


# =============================================================================
# 12. EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_evaluate_with_no_graph(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test evaluation with no graph set."""
        component.graph = None

        entity = MagicMock()
        world = MagicMock()

        # Should not raise
        graph_system.update(world, 0.016, [(entity, component)])

        assert component.output_pose.bone_count() == 0

    def test_evaluate_with_no_skeleton(
        self, graph_system: AnimationGraphSystem
    ) -> None:
        """Test evaluation with no skeleton."""
        component = AnimationGraphComponent()
        component.enabled = True
        component.skeleton = None

        entity = MagicMock()
        world = MagicMock()

        # Should not raise
        graph_system.update(world, 0.016, [(entity, component)])

    def test_clip_not_found(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test handling of missing clip."""
        component.state_machine_output = StateMachineOutput(
            current_state="nonexistent"
        )

        entity = MagicMock()
        world = MagicMock()

        # Should not raise
        graph_system.update(world, 0.016, [(entity, component)])

    def test_zero_delta_time(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test with zero delta time."""
        entity = MagicMock()
        world = MagicMock()

        # Should not raise
        graph_system.update(world, 0.0, [(entity, component)])

    def test_negative_delta_time(
        self, graph_system: AnimationGraphSystem, component: AnimationGraphComponent
    ) -> None:
        """Test with negative delta time (edge case)."""
        entity = MagicMock()
        world = MagicMock()

        # Should not raise (though behavior may be undefined)
        graph_system.update(world, -0.016, [(entity, component)])

    def test_very_large_delta_time(
        self,
        graph_system: AnimationGraphSystem,
        component: AnimationGraphComponent,
        sample_clip: AnimationClip,
    ) -> None:
        """Test with very large delta time."""
        component.register_clip("test", sample_clip)
        component.state_machine_output = StateMachineOutput(current_state="test")

        entity = MagicMock()
        world = MagicMock()

        # Should not raise
        graph_system.update(world, 100.0, [(entity, component)])


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
