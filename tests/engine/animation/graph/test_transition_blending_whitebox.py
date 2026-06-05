"""
Whitebox tests for transition blending functionality.

Tests with full source access for:
- ActiveTransition class: progress tracking, blend_weight, is_complete
- Pose blending: source_pose.lerp(target_pose, weight) in evaluate()
- Curve application: evaluate_blend_curve() with different curves
- Completion detection: is_complete property behavior
- Sync modes: TransitionSyncMode enum and _apply_sync_mode() method

Task: T-AG-2.6 Transition Blending
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from engine.animation.graph.state_machine import (
    ActiveTransition,
    StateTransition,
    AnimationState,
    BlendCurve,
    evaluate_blend_curve,
    StateMachine,
    TransitionCondition,
    ConditionOperator,
    InterruptMode,
    TransitionSyncMode,
    MotionMode,
)
from engine.animation.graph.animation_graph import (
    GraphContext,
    Pose,
    Transform,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_graph_context() -> GraphContext:
    """Create a mock GraphContext for testing."""
    context = Mock(spec=GraphContext)
    context.parameters = {}
    context.dt = 0.016
    context.skeleton = None
    context.bone_masks = None
    context.normalized_time = 0.0
    context.sync_group = None
    context.layer_weight = 1.0

    def get_parameter(name: str) -> Any:
        return context.parameters.get(name)

    def set_parameter(name: str, value: Any) -> None:
        context.parameters[name] = value

    context.get_parameter = get_parameter
    context.set_parameter = set_parameter
    return context


@pytest.fixture
def source_animation_state() -> AnimationState:
    """Create a source AnimationState for testing."""
    state = AnimationState(
        name="source_state",
        motion_mode=MotionMode.LOOP,
        speed=1.0,
    )
    state._normalized_time = 0.5
    state.current_time = 0.5
    return state


@pytest.fixture
def target_animation_state() -> AnimationState:
    """Create a target AnimationState for testing."""
    state = AnimationState(
        name="target_state",
        motion_mode=MotionMode.LOOP,
        speed=1.0,
    )
    state._normalized_time = 0.0
    state.current_time = 0.0
    return state


@pytest.fixture
def basic_transition() -> StateTransition:
    """Create a basic StateTransition for testing."""
    return StateTransition(
        source="source_state",
        target="target_state",
        conditions=[],
        duration=1.0,
        duration_mode="fixed",
        blend_curve=BlendCurve.LINEAR,
        priority=0,
        interrupt_mode=InterruptMode.HIGHER_PRIORITY,
    )


@pytest.fixture
def source_pose() -> Pose:
    """Create a source pose for blending tests."""
    return Pose(
        transforms=[
            Transform(position=(0.0, 0.0, 0.0)),
            Transform(position=(1.0, 0.0, 0.0)),
            Transform(position=(2.0, 0.0, 0.0)),
        ]
    )


@pytest.fixture
def target_pose() -> Pose:
    """Create a target pose for blending tests."""
    return Pose(
        transforms=[
            Transform(position=(10.0, 0.0, 0.0)),
            Transform(position=(11.0, 0.0, 0.0)),
            Transform(position=(12.0, 0.0, 0.0)),
        ]
    )


@pytest.fixture
def active_transition(
    basic_transition: StateTransition,
    source_animation_state: AnimationState,
    target_animation_state: AnimationState,
) -> ActiveTransition:
    """Create an ActiveTransition for testing."""
    return ActiveTransition(
        transition=basic_transition,
        source_state=source_animation_state,
        target_state=target_animation_state,
        progress=0.0,
        source_pose=None,
        duration=0.0,  # Will be calculated in __post_init__
    )


# =============================================================================
# SECTION 1: ActiveTransition Class Tests
# =============================================================================


class TestActiveTransitionDataclass:
    """Tests for ActiveTransition dataclass fields and initialization."""

    def test_transition_field(
        self,
        basic_transition: StateTransition,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test transition field holds the StateTransition reference."""
        active = ActiveTransition(
            transition=basic_transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        assert active.transition is basic_transition

    def test_source_state_field(
        self,
        basic_transition: StateTransition,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test source_state field holds the source AnimationState."""
        active = ActiveTransition(
            transition=basic_transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        assert active.source_state is source_animation_state
        assert active.source_state.name == "source_state"

    def test_target_state_field(
        self,
        basic_transition: StateTransition,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test target_state field holds the target AnimationState."""
        active = ActiveTransition(
            transition=basic_transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        assert active.target_state is target_animation_state
        assert active.target_state.name == "target_state"

    def test_progress_default_zero(
        self,
        basic_transition: StateTransition,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test progress field defaults to 0.0."""
        active = ActiveTransition(
            transition=basic_transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        assert active.progress == 0.0

    def test_progress_explicit_value(
        self,
        basic_transition: StateTransition,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test progress field can be set explicitly."""
        active = ActiveTransition(
            transition=basic_transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=0.5,
        )
        assert active.progress == 0.5

    def test_source_pose_field_none_default(
        self,
        basic_transition: StateTransition,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test source_pose field defaults to None."""
        active = ActiveTransition(
            transition=basic_transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        assert active.source_pose is None

    def test_source_pose_explicit_value(
        self,
        basic_transition: StateTransition,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
        source_pose: Pose,
    ) -> None:
        """Test source_pose field can be set explicitly."""
        active = ActiveTransition(
            transition=basic_transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            source_pose=source_pose,
        )
        assert active.source_pose is source_pose

    def test_duration_post_init_fixed_mode(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test duration is computed correctly in __post_init__ for fixed mode."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=0.5,
            duration_mode="fixed",
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        assert active.duration == 0.5

    def test_duration_post_init_percentage_mode(
        self,
        target_animation_state: AnimationState,
    ) -> None:
        """Test duration is computed correctly for percentage mode."""
        # Create source state with a clip that has duration
        source_state = AnimationState(name="source_state")
        mock_clip = Mock()
        mock_clip.duration = 2.0
        source_state.clip = mock_clip

        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=0.25,  # 25% of source animation
            duration_mode="percentage",
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_state,
            target_state=target_animation_state,
        )
        # 0.25 * 2.0 = 0.5 seconds
        assert active.duration == 0.5


# =============================================================================
# SECTION 2: Progress Tracking Tests
# =============================================================================


class TestActiveTransitionProgressTracking:
    """Tests for ActiveTransition progress tracking (0.0 to duration)."""

    def test_update_increments_progress(
        self,
        basic_transition: StateTransition,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test update() method increments progress by dt."""
        active = ActiveTransition(
            transition=basic_transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        initial_progress = active.progress
        active.update(0.1)
        assert active.progress == initial_progress + 0.1

    def test_update_accumulates_progress(
        self,
        basic_transition: StateTransition,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test multiple update() calls accumulate progress."""
        active = ActiveTransition(
            transition=basic_transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        active.update(0.1)
        active.update(0.2)
        active.update(0.3)
        assert abs(active.progress - 0.6) < 1e-9

    def test_progress_starts_at_zero(
        self,
        basic_transition: StateTransition,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test progress starts at 0.0."""
        active = ActiveTransition(
            transition=basic_transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        assert active.progress == 0.0

    def test_progress_can_exceed_duration(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test progress can exceed duration (not clamped internally)."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=0.5,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        active.update(0.6)
        assert active.progress == 0.6
        assert active.progress > active.duration


# =============================================================================
# SECTION 3: Blend Weight Property Tests
# =============================================================================


class TestActiveTransitionBlendWeight:
    """Tests for ActiveTransition blend_weight property with curve application."""

    def test_blend_weight_at_zero_progress(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test blend_weight is 0.0 at progress = 0."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
            blend_curve=BlendCurve.LINEAR,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=0.0,
        )
        assert active.blend_weight == 0.0

    def test_blend_weight_at_full_progress(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test blend_weight is 1.0 at progress = duration."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
            blend_curve=BlendCurve.LINEAR,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=1.0,
        )
        assert active.blend_weight == 1.0

    def test_blend_weight_mid_progress_linear(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test blend_weight at 50% progress with LINEAR curve."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
            blend_curve=BlendCurve.LINEAR,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=0.5,
        )
        assert abs(active.blend_weight - 0.5) < 1e-9

    def test_blend_weight_zero_duration_returns_one(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test blend_weight returns 1.0 when duration is zero."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=0.0,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
        )
        # duration should be 0.0 since it's fixed mode with 0 duration
        active.duration = 0.0
        assert active.blend_weight == 1.0

    def test_blend_weight_with_ease_in_curve(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test blend_weight at 50% progress with EASE_IN curve (t*t)."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
            blend_curve=BlendCurve.EASE_IN,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=0.5,
        )
        # EASE_IN at t=0.5 is 0.5*0.5 = 0.25
        assert abs(active.blend_weight - 0.25) < 1e-9

    def test_blend_weight_with_ease_out_curve(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test blend_weight at 50% progress with EASE_OUT curve (t*(2-t))."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
            blend_curve=BlendCurve.EASE_OUT,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=0.5,
        )
        # EASE_OUT at t=0.5 is 0.5*(2-0.5) = 0.5*1.5 = 0.75
        assert abs(active.blend_weight - 0.75) < 1e-9

    def test_blend_weight_with_smooth_step_curve(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test blend_weight at 50% progress with SMOOTH_STEP curve."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
            blend_curve=BlendCurve.SMOOTH_STEP,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=0.5,
        )
        # SMOOTH_STEP at t=0.5 is 0.5*0.5*(3 - 2*0.5) = 0.25*2 = 0.5
        assert abs(active.blend_weight - 0.5) < 1e-9


# =============================================================================
# SECTION 4: Is Complete Property Tests
# =============================================================================


class TestActiveTransitionIsComplete:
    """Tests for ActiveTransition is_complete property."""

    def test_is_complete_false_at_start(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test is_complete is False when progress is 0."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=0.0,
        )
        assert active.is_complete is False

    def test_is_complete_false_mid_transition(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test is_complete is False when progress < duration."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=0.5,
        )
        assert active.is_complete is False

    def test_is_complete_true_at_duration(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test is_complete is True when progress equals duration."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=1.0,
        )
        assert active.is_complete is True

    def test_is_complete_true_when_exceeds_duration(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test is_complete is True when progress exceeds duration."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=1.5,
        )
        assert active.is_complete is True

    def test_is_complete_with_zero_duration(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test is_complete is True when duration is zero."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=0.0,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=0.0,
        )
        active.duration = 0.0  # Force zero duration
        assert active.is_complete is True


# =============================================================================
# SECTION 5: Evaluate Blend Curve Function Tests
# =============================================================================


class TestEvaluateBlendCurve:
    """Tests for evaluate_blend_curve() function with different curves."""

    def test_linear_curve_at_zero(self) -> None:
        """Test LINEAR curve returns 0.0 at t=0."""
        assert evaluate_blend_curve(BlendCurve.LINEAR, 0.0) == 0.0

    def test_linear_curve_at_one(self) -> None:
        """Test LINEAR curve returns 1.0 at t=1."""
        assert evaluate_blend_curve(BlendCurve.LINEAR, 1.0) == 1.0

    def test_linear_curve_at_mid(self) -> None:
        """Test LINEAR curve returns 0.5 at t=0.5."""
        assert evaluate_blend_curve(BlendCurve.LINEAR, 0.5) == 0.5

    def test_ease_in_curve_at_zero(self) -> None:
        """Test EASE_IN curve returns 0.0 at t=0."""
        assert evaluate_blend_curve(BlendCurve.EASE_IN, 0.0) == 0.0

    def test_ease_in_curve_at_one(self) -> None:
        """Test EASE_IN curve returns 1.0 at t=1."""
        assert evaluate_blend_curve(BlendCurve.EASE_IN, 1.0) == 1.0

    def test_ease_in_curve_formula(self) -> None:
        """Test EASE_IN follows t*t formula."""
        t = 0.5
        expected = t * t  # 0.25
        assert abs(evaluate_blend_curve(BlendCurve.EASE_IN, t) - expected) < 1e-9

    def test_ease_out_curve_at_zero(self) -> None:
        """Test EASE_OUT curve returns 0.0 at t=0."""
        assert evaluate_blend_curve(BlendCurve.EASE_OUT, 0.0) == 0.0

    def test_ease_out_curve_at_one(self) -> None:
        """Test EASE_OUT curve returns 1.0 at t=1."""
        assert abs(evaluate_blend_curve(BlendCurve.EASE_OUT, 1.0) - 1.0) < 1e-9

    def test_ease_out_curve_formula(self) -> None:
        """Test EASE_OUT follows t*(2-t) formula."""
        t = 0.5
        expected = t * (2.0 - t)  # 0.75
        assert abs(evaluate_blend_curve(BlendCurve.EASE_OUT, t) - expected) < 1e-9

    def test_ease_in_out_curve_at_zero(self) -> None:
        """Test EASE_IN_OUT curve returns 0.0 at t=0."""
        assert evaluate_blend_curve(BlendCurve.EASE_IN_OUT, 0.0) == 0.0

    def test_ease_in_out_curve_at_one(self) -> None:
        """Test EASE_IN_OUT curve returns 1.0 at t=1."""
        assert abs(evaluate_blend_curve(BlendCurve.EASE_IN_OUT, 1.0) - 1.0) < 1e-9

    def test_ease_in_out_curve_formula(self) -> None:
        """Test EASE_IN_OUT follows t*t*(3-2t) formula."""
        t = 0.5
        expected = t * t * (3.0 - 2.0 * t)  # 0.5
        assert abs(evaluate_blend_curve(BlendCurve.EASE_IN_OUT, t) - expected) < 1e-9

    def test_smooth_step_curve_at_zero(self) -> None:
        """Test SMOOTH_STEP curve returns 0.0 at t=0."""
        assert evaluate_blend_curve(BlendCurve.SMOOTH_STEP, 0.0) == 0.0

    def test_smooth_step_curve_at_one(self) -> None:
        """Test SMOOTH_STEP curve returns 1.0 at t=1."""
        assert abs(evaluate_blend_curve(BlendCurve.SMOOTH_STEP, 1.0) - 1.0) < 1e-9

    def test_smooth_step_curve_formula(self) -> None:
        """Test SMOOTH_STEP follows t*t*(3-2t) formula (same as EASE_IN_OUT)."""
        t = 0.25
        expected = t * t * (3.0 - 2.0 * t)
        assert abs(evaluate_blend_curve(BlendCurve.SMOOTH_STEP, t) - expected) < 1e-9

    def test_smoother_step_curve_at_zero(self) -> None:
        """Test SMOOTHER_STEP curve returns 0.0 at t=0."""
        assert evaluate_blend_curve(BlendCurve.SMOOTHER_STEP, 0.0) == 0.0

    def test_smoother_step_curve_at_one(self) -> None:
        """Test SMOOTHER_STEP curve returns 1.0 at t=1."""
        assert abs(evaluate_blend_curve(BlendCurve.SMOOTHER_STEP, 1.0) - 1.0) < 1e-9

    def test_smoother_step_curve_formula(self) -> None:
        """Test SMOOTHER_STEP follows 6t^5 - 15t^4 + 10t^3 formula."""
        t = 0.5
        expected = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)  # 0.5
        assert abs(evaluate_blend_curve(BlendCurve.SMOOTHER_STEP, t) - expected) < 1e-9

    def test_curve_clamps_negative_t(self) -> None:
        """Test curves clamp t to 0 when negative."""
        assert evaluate_blend_curve(BlendCurve.LINEAR, -0.5) == 0.0
        assert evaluate_blend_curve(BlendCurve.EASE_IN, -0.5) == 0.0

    def test_curve_clamps_t_above_one(self) -> None:
        """Test curves clamp t to 1 when above 1."""
        assert evaluate_blend_curve(BlendCurve.LINEAR, 1.5) == 1.0
        assert abs(evaluate_blend_curve(BlendCurve.EASE_IN, 1.5) - 1.0) < 1e-9


# =============================================================================
# SECTION 6: Pose Blending Tests
# =============================================================================


class TestPoseBlending:
    """Tests for pose blending: source_pose.lerp(target_pose, weight)."""

    def test_lerp_weight_zero_returns_source(
        self,
        source_pose: Pose,
        target_pose: Pose,
    ) -> None:
        """Test lerp with weight=0 returns source pose."""
        result = source_pose.lerp(target_pose, 0.0)
        for i in range(len(source_pose.transforms)):
            assert result.transforms[i].position == source_pose.transforms[i].position

    def test_lerp_weight_one_returns_target(
        self,
        source_pose: Pose,
        target_pose: Pose,
    ) -> None:
        """Test lerp with weight=1 returns target pose."""
        result = source_pose.lerp(target_pose, 1.0)
        for i in range(len(target_pose.transforms)):
            assert result.transforms[i].position == target_pose.transforms[i].position

    def test_lerp_mid_weight_interpolates(
        self,
        source_pose: Pose,
        target_pose: Pose,
    ) -> None:
        """Test lerp with weight=0.5 interpolates positions."""
        result = source_pose.lerp(target_pose, 0.5)
        # First bone: (0,0,0) + ((10,0,0) - (0,0,0)) * 0.5 = (5,0,0)
        assert abs(result.transforms[0].position[0] - 5.0) < 1e-9
        # Second bone: (1,0,0) + ((11,0,0) - (1,0,0)) * 0.5 = (6,0,0)
        assert abs(result.transforms[1].position[0] - 6.0) < 1e-9

    def test_lerp_different_bone_counts(self) -> None:
        """Test lerp handles poses with different bone counts."""
        source = Pose(transforms=[Transform(position=(0.0, 0.0, 0.0))])
        target = Pose(
            transforms=[
                Transform(position=(10.0, 0.0, 0.0)),
                Transform(position=(20.0, 0.0, 0.0)),
            ]
        )
        result = source.lerp(target, 0.5)
        # Result should have max(1, 2) = 2 bones
        assert len(result.transforms) == 2
        # First bone: interpolated
        assert abs(result.transforms[0].position[0] - 5.0) < 1e-9
        # Second bone: identity lerped with target
        assert abs(result.transforms[1].position[0] - 10.0) < 1e-9

    def test_lerp_preserves_rotation(self) -> None:
        """Test lerp interpolates rotation correctly."""
        source = Pose(
            transforms=[
                Transform(rotation=(0.0, 0.0, 0.0, 1.0))  # Identity
            ]
        )
        target = Pose(
            transforms=[
                Transform(rotation=(0.0, 0.0, 0.707107, 0.707107))  # 90 deg Z
            ]
        )
        result = source.lerp(target, 0.5)
        # Should be approximately 45 deg rotation
        assert len(result.transforms) == 1
        # Check rotation is interpolated (not exactly 45 deg due to slerp)
        rot = result.transforms[0].rotation
        assert abs(rot[3]) < 1.0  # w component should be less than 1

    def test_lerp_preserves_scale(self) -> None:
        """Test lerp interpolates scale correctly."""
        source = Pose(transforms=[Transform(scale=(1.0, 1.0, 1.0))])
        target = Pose(transforms=[Transform(scale=(2.0, 2.0, 2.0))])
        result = source.lerp(target, 0.5)
        # Scale should be (1.5, 1.5, 1.5)
        assert abs(result.transforms[0].scale[0] - 1.5) < 1e-9
        assert abs(result.transforms[0].scale[1] - 1.5) < 1e-9
        assert abs(result.transforms[0].scale[2] - 1.5) < 1e-9


# =============================================================================
# SECTION 7: TransitionSyncMode Enum Tests
# =============================================================================


class TestTransitionSyncModeEnum:
    """Tests for TransitionSyncMode enum values."""

    def test_none_mode_exists(self) -> None:
        """Test TransitionSyncMode.NONE exists."""
        assert hasattr(TransitionSyncMode, "NONE")

    def test_normalized_mode_exists(self) -> None:
        """Test TransitionSyncMode.NORMALIZED exists."""
        assert hasattr(TransitionSyncMode, "NORMALIZED")

    def test_proportional_mode_exists(self) -> None:
        """Test TransitionSyncMode.PROPORTIONAL exists."""
        assert hasattr(TransitionSyncMode, "PROPORTIONAL")

    def test_marker_mode_exists(self) -> None:
        """Test TransitionSyncMode.MARKER exists."""
        assert hasattr(TransitionSyncMode, "MARKER")

    def test_all_sync_modes_count(self) -> None:
        """Test there are exactly 4 sync modes."""
        assert len(list(TransitionSyncMode)) == 4

    def test_sync_mode_identity(self) -> None:
        """Test sync mode identity comparison."""
        mode1 = TransitionSyncMode.NONE
        mode2 = TransitionSyncMode.NONE
        assert mode1 is mode2

    def test_sync_mode_inequality(self) -> None:
        """Test different sync modes are not equal."""
        assert TransitionSyncMode.NONE != TransitionSyncMode.NORMALIZED
        assert TransitionSyncMode.NORMALIZED != TransitionSyncMode.PROPORTIONAL


# =============================================================================
# SECTION 8: Apply Sync Mode Method Tests
# =============================================================================


class TestApplySyncMode:
    """Tests for _apply_sync_mode() method."""

    def test_none_sync_mode_does_nothing(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test NONE sync mode does not modify target state time."""
        sm = StateMachine("test_sm")

        source = AnimationState(name="source")
        source._normalized_time = 0.5

        target = AnimationState(name="target")
        target._normalized_time = 0.0

        transition = StateTransition(
            source="source",
            target="target",
            sync_mode=TransitionSyncMode.NONE,
        )

        sm._apply_sync_mode(transition, source, target)

        # Target time should be unchanged
        assert target._normalized_time == 0.0

    def test_normalized_sync_mode_copies_time(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test NORMALIZED sync mode copies normalized time from source to target."""
        sm = StateMachine("test_sm")

        source = AnimationState(name="source")
        source._normalized_time = 0.75

        target = AnimationState(name="target")
        target._normalized_time = 0.0

        transition = StateTransition(
            source="source",
            target="target",
            sync_mode=TransitionSyncMode.NORMALIZED,
        )

        sm._apply_sync_mode(transition, source, target)

        # Target time should match source normalized time
        assert target._normalized_time == 0.75

    def test_proportional_sync_mode_calculates_remaining(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test PROPORTIONAL sync mode calculates based on remaining time ratio."""
        sm = StateMachine("test_sm")

        source = AnimationState(name="source")
        source_clip = Mock()
        source_clip.duration = 2.0
        source.clip = source_clip
        source._normalized_time = 0.5  # 50% through

        target = AnimationState(name="target")
        target_clip = Mock()
        target_clip.duration = 1.0
        target.clip = target_clip
        target._normalized_time = 0.0

        transition = StateTransition(
            source="source",
            target="target",
            sync_mode=TransitionSyncMode.PROPORTIONAL,
        )

        sm._apply_sync_mode(transition, source, target)

        # Source is 50% through, remaining_ratio = 0.5
        # Target normalized time = 1.0 - 0.5 = 0.5
        assert target._normalized_time == 0.5

    def test_marker_sync_mode_falls_back_to_normalized(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test MARKER sync mode currently falls back to normalized sync."""
        sm = StateMachine("test_sm")

        source = AnimationState(name="source")
        source._normalized_time = 0.3

        target = AnimationState(name="target")
        target._normalized_time = 0.0

        transition = StateTransition(
            source="source",
            target="target",
            sync_mode=TransitionSyncMode.MARKER,
        )

        sm._apply_sync_mode(transition, source, target)

        # Currently falls back to normalized sync
        assert target._normalized_time == 0.3


# =============================================================================
# SECTION 9: State Machine Transition Evaluation Tests
# =============================================================================


class TestStateMachineTransitionEvaluation:
    """Tests for StateMachine evaluate() method during transitions."""

    def test_evaluate_blends_poses_during_transition(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test evaluate() blends source and target poses during active transition."""
        sm = StateMachine("test_sm")

        # Create states with mock nodes that return predictable poses
        source = AnimationState(name="source")
        source_node = Mock()
        source_pose = Pose(
            transforms=[Transform(position=(0.0, 0.0, 0.0))]
        )
        source_node.evaluate.return_value = source_pose
        source.animation_node = source_node

        target = AnimationState(name="target")
        target_node = Mock()
        target_pose = Pose(
            transforms=[Transform(position=(10.0, 0.0, 0.0))]
        )
        target_node.evaluate.return_value = target_pose
        target.animation_node = target_node

        sm.add_state(source)
        sm.add_state(target)
        sm.start(mock_graph_context)

        # Create an active transition at 50%
        transition = StateTransition(
            source="source",
            target="target",
            duration=1.0,
            blend_curve=BlendCurve.LINEAR,
        )
        sm._active_transition = ActiveTransition(
            transition=transition,
            source_state=source,
            target_state=target,
            progress=0.5,
            source_pose=source_pose,
        )
        sm._active_transition.duration = 1.0

        result = sm.evaluate(mock_graph_context)

        # Result should be 50% between (0,0,0) and (10,0,0) = (5,0,0)
        assert abs(result.transforms[0].position[0] - 5.0) < 1e-9

    def test_evaluate_returns_target_pose_at_complete_transition(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test evaluate() returns target pose when blend_weight is 1.0."""
        sm = StateMachine("test_sm")

        source = AnimationState(name="source")
        source_node = Mock()
        source_pose = Pose(transforms=[Transform(position=(0.0, 0.0, 0.0))])
        source_node.evaluate.return_value = source_pose
        source.animation_node = source_node

        target = AnimationState(name="target")
        target_node = Mock()
        target_pose = Pose(transforms=[Transform(position=(10.0, 0.0, 0.0))])
        target_node.evaluate.return_value = target_pose
        target.animation_node = target_node

        sm.add_state(source)
        sm.add_state(target)
        sm.start(mock_graph_context)

        # Create an active transition at 100%
        transition = StateTransition(
            source="source",
            target="target",
            duration=1.0,
            blend_curve=BlendCurve.LINEAR,
        )
        sm._active_transition = ActiveTransition(
            transition=transition,
            source_state=source,
            target_state=target,
            progress=1.0,
            source_pose=source_pose,
        )
        sm._active_transition.duration = 1.0

        result = sm.evaluate(mock_graph_context)

        # Result should be 100% target = (10,0,0)
        assert abs(result.transforms[0].position[0] - 10.0) < 1e-9

    def test_evaluate_uses_source_pose_if_available(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test evaluate() uses cached source_pose instead of re-evaluating."""
        sm = StateMachine("test_sm")

        source = AnimationState(name="source")
        source_node = Mock()
        # Source node should not be called since we have cached pose
        source_node.evaluate.return_value = Pose(
            transforms=[Transform(position=(100.0, 0.0, 0.0))]
        )
        source.animation_node = source_node

        target = AnimationState(name="target")
        target_node = Mock()
        target_pose = Pose(transforms=[Transform(position=(10.0, 0.0, 0.0))])
        target_node.evaluate.return_value = target_pose
        target.animation_node = target_node

        sm.add_state(source)
        sm.add_state(target)
        sm.start(mock_graph_context)

        # Create cached source pose with specific position
        cached_source_pose = Pose(transforms=[Transform(position=(0.0, 0.0, 0.0))])

        transition = StateTransition(
            source="source",
            target="target",
            duration=1.0,
            blend_curve=BlendCurve.LINEAR,
        )
        sm._active_transition = ActiveTransition(
            transition=transition,
            source_state=source,
            target_state=target,
            progress=0.5,
            source_pose=cached_source_pose,  # Cached pose
        )
        sm._active_transition.duration = 1.0

        result = sm.evaluate(mock_graph_context)

        # Should blend cached (0,0,0) with target (10,0,0) at 50% = (5,0,0)
        # Not (100,0,0) which would be from re-evaluating source
        assert abs(result.transforms[0].position[0] - 5.0) < 1e-9

    def test_evaluate_re_evaluates_source_if_no_cached_pose(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test evaluate() re-evaluates source state if source_pose is None."""
        sm = StateMachine("test_sm")

        source = AnimationState(name="source")
        source_node = Mock()
        source_pose = Pose(transforms=[Transform(position=(0.0, 0.0, 0.0))])
        source_node.evaluate.return_value = source_pose
        source.animation_node = source_node

        target = AnimationState(name="target")
        target_node = Mock()
        target_pose = Pose(transforms=[Transform(position=(10.0, 0.0, 0.0))])
        target_node.evaluate.return_value = target_pose
        target.animation_node = target_node

        sm.add_state(source)
        sm.add_state(target)
        sm.start(mock_graph_context)

        transition = StateTransition(
            source="source",
            target="target",
            duration=1.0,
            blend_curve=BlendCurve.LINEAR,
        )
        sm._active_transition = ActiveTransition(
            transition=transition,
            source_state=source,
            target_state=target,
            progress=0.5,
            source_pose=None,  # No cached pose
        )
        sm._active_transition.duration = 1.0

        result = sm.evaluate(mock_graph_context)

        # Should blend evaluated source (0,0,0) with target (10,0,0) at 50%
        assert abs(result.transforms[0].position[0] - 5.0) < 1e-9


# =============================================================================
# SECTION 10: Transition Completion Detection Tests
# =============================================================================


class TestTransitionCompletionDetection:
    """Tests for transition completion detection and clearing."""

    def test_transition_cleared_on_completion(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test active transition is cleared when complete."""
        sm = StateMachine("test_sm")

        source = AnimationState(name="source")
        target = AnimationState(name="target")
        sm.add_state(source)
        sm.add_state(target)
        sm.start(mock_graph_context)

        transition = StateTransition(
            source="source",
            target="target",
            duration=0.5,
        )
        sm._active_transition = ActiveTransition(
            transition=transition,
            source_state=source,
            target_state=target,
            progress=0.0,
        )
        sm._active_transition.duration = 0.5

        # Update with enough time to complete
        sm.update(0.6, mock_graph_context)

        # Transition should be cleared
        assert sm._active_transition is None
        assert sm._current_state is target

    def test_source_exit_called_on_completion(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test source state exit callback is called on transition completion."""
        sm = StateMachine("test_sm")

        exit_called = []

        def on_exit(state, context):
            exit_called.append(state.name)

        source = AnimationState(name="source", on_exit=on_exit)
        target = AnimationState(name="target")
        sm.add_state(source)
        sm.add_state(target)
        sm.start(mock_graph_context)

        transition = StateTransition(
            source="source",
            target="target",
            duration=0.5,
        )
        sm._active_transition = ActiveTransition(
            transition=transition,
            source_state=source,
            target_state=target,
            progress=0.4,
        )
        sm._active_transition.duration = 0.5

        sm.update(0.2, mock_graph_context)

        assert "source" in exit_called

    def test_current_state_set_to_target_on_completion(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test current state is set to target on transition completion."""
        sm = StateMachine("test_sm")

        source = AnimationState(name="source")
        target = AnimationState(name="target")
        sm.add_state(source)
        sm.add_state(target)
        sm.start(mock_graph_context)

        assert sm._current_state is source

        transition = StateTransition(
            source="source",
            target="target",
            duration=0.5,
        )
        sm._active_transition = ActiveTransition(
            transition=transition,
            source_state=source,
            target_state=target,
            progress=0.5,  # Already at completion
        )
        sm._active_transition.duration = 0.5

        sm.update(0.1, mock_graph_context)

        assert sm._current_state is target


# =============================================================================
# SECTION 11: Edge Cases and Boundary Conditions
# =============================================================================


class TestTransitionEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_blend_weight_with_very_small_duration(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test blend_weight handles very small but positive duration."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=0.001,  # 1ms
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=0.0005,
        )
        active.duration = 0.001
        # Weight should be ~0.5
        assert 0.4 < active.blend_weight < 0.6

    def test_blend_weight_with_very_large_progress(
        self,
        source_animation_state: AnimationState,
        target_animation_state: AnimationState,
    ) -> None:
        """Test blend_weight is clamped when progress far exceeds duration."""
        transition = StateTransition(
            source="source_state",
            target="target_state",
            duration=1.0,
            blend_curve=BlendCurve.LINEAR,
        )
        active = ActiveTransition(
            transition=transition,
            source_state=source_animation_state,
            target_state=target_animation_state,
            progress=100.0,  # Way over duration
        )
        active.duration = 1.0
        # Should clamp to 1.0
        assert active.blend_weight == 1.0

    def test_lerp_empty_poses(self) -> None:
        """Test lerp handles empty poses."""
        source = Pose(transforms=[])
        target = Pose(transforms=[])
        result = source.lerp(target, 0.5)
        assert len(result.transforms) == 0

    def test_lerp_single_bone(self) -> None:
        """Test lerp with single bone pose."""
        source = Pose(transforms=[Transform(position=(0.0, 0.0, 0.0))])
        target = Pose(transforms=[Transform(position=(2.0, 4.0, 6.0))])
        result = source.lerp(target, 0.5)
        assert len(result.transforms) == 1
        assert abs(result.transforms[0].position[0] - 1.0) < 1e-9
        assert abs(result.transforms[0].position[1] - 2.0) < 1e-9
        assert abs(result.transforms[0].position[2] - 3.0) < 1e-9

    def test_all_blend_curves_produce_valid_output(self) -> None:
        """Test all blend curves produce values in [0, 1] range."""
        curves = list(BlendCurve)
        test_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

        for curve in curves:
            for t in test_values:
                result = evaluate_blend_curve(curve, t)
                assert 0.0 <= result <= 1.0, f"{curve} at t={t} produced {result}"

    def test_evaluate_blend_curve_is_monotonic(self) -> None:
        """Test evaluate_blend_curve produces monotonically increasing output."""
        curves = [
            BlendCurve.LINEAR,
            BlendCurve.EASE_IN,
            BlendCurve.EASE_OUT,
            BlendCurve.SMOOTH_STEP,
            BlendCurve.SMOOTHER_STEP,
        ]
        for curve in curves:
            prev = 0.0
            for i in range(101):
                t = i / 100.0
                value = evaluate_blend_curve(curve, t)
                assert value >= prev - 1e-9, f"{curve} not monotonic at t={t}"
                prev = value


# =============================================================================
# SECTION 12: Integration Tests
# =============================================================================


class TestTransitionBlendingIntegration:
    """Integration tests for full transition blending flow."""

    def test_full_transition_lifecycle(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test complete transition from start to finish."""
        sm = StateMachine("test_sm")

        source_node = Mock()
        source_pose = Pose(transforms=[Transform(position=(0.0, 0.0, 0.0))])
        source_node.evaluate.return_value = source_pose
        source = AnimationState(name="idle", animation_node=source_node)

        target_node = Mock()
        target_pose = Pose(transforms=[Transform(position=(10.0, 0.0, 0.0))])
        target_node.evaluate.return_value = target_pose
        target = AnimationState(name="walk", animation_node=target_node)

        sm.add_state(source)
        sm.add_state(target)
        sm.add_transition(
            StateTransition(
                source="idle",
                target="walk",
                conditions=[TransitionCondition.equals("move", True)],
                duration=0.4,
                blend_curve=BlendCurve.SMOOTH_STEP,
            )
        )

        sm.start(mock_graph_context)
        assert sm._current_state is source

        # Trigger transition
        mock_graph_context.parameters["move"] = True
        sm.update(0.016, mock_graph_context)

        # Should now be transitioning
        assert sm._active_transition is not None
        assert sm._active_transition.transition.target == "walk"

        # Progress through transition
        for _ in range(25):  # ~0.4 seconds at 16ms per frame
            sm.update(0.016, mock_graph_context)

        # Should be complete
        assert sm._active_transition is None
        assert sm._current_state is target

    def test_transition_blending_produces_smooth_values(
        self, mock_graph_context: GraphContext
    ) -> None:
        """Test transition produces smoothly interpolated values."""
        sm = StateMachine("test_sm")

        source_node = Mock()
        source_node.evaluate.return_value = Pose(
            transforms=[Transform(position=(0.0, 0.0, 0.0))]
        )
        source = AnimationState(name="idle", animation_node=source_node)

        target_node = Mock()
        target_node.evaluate.return_value = Pose(
            transforms=[Transform(position=(10.0, 0.0, 0.0))]
        )
        target = AnimationState(name="walk", animation_node=target_node)

        sm.add_state(source)
        sm.add_state(target)
        sm.start(mock_graph_context)

        transition = StateTransition(
            source="idle",
            target="walk",
            duration=1.0,
            blend_curve=BlendCurve.LINEAR,
        )
        sm._active_transition = ActiveTransition(
            transition=transition,
            source_state=source,
            target_state=target,
            progress=0.0,
            source_pose=Pose(transforms=[Transform(position=(0.0, 0.0, 0.0))]),
        )
        sm._active_transition.duration = 1.0

        # Sample values during transition
        values = []
        for i in range(11):
            sm._active_transition.progress = i / 10.0
            pose = sm.evaluate(mock_graph_context)
            values.append(pose.transforms[0].position[0])

        # Values should be monotonically increasing
        for i in range(len(values) - 1):
            assert values[i + 1] >= values[i], f"Not monotonic at {i}"

        # Values should range from 0 to 10
        assert abs(values[0] - 0.0) < 1e-9
        assert abs(values[-1] - 10.0) < 1e-9
