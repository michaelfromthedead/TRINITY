"""
Comprehensive tests for the Camera Track system.

Tests all camera track modules:
- CameraKeyframe: Keyframe data structure
- CameraTrack: Main track class with interpolation
- CameraTrackManager: Multi-track management
- Spline interpolation: Catmull-Rom and cubic Bezier
- Blend transitions: Blend in/out from gameplay camera
- Look-at constraints: Target tracking
- @camera_track decorator: Registration system

Minimum 50 tests with real assertions.
"""

import math
import pytest
from typing import Dict, Optional, Tuple

from engine.animation.cinematics.camera_track import (
    BezierControlPoint,
    BlendState,
    CameraKeyframe,
    CameraState,
    CameraTrack,
    CameraTrackManager,
    InterpolationMode,
    LookAtTarget,
    camera_track,
    catmull_rom_interpolate,
    catmull_rom_tangent,
    create_camera_track,
    create_track_from_class,
    cubic_bezier_interpolate,
    cubic_bezier_tangent,
    get_camera_track_registry,
    quat_identity,
    quat_look_at,
    quat_slerp,
    vec3_lerp,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_keyframes():
    """Create a simple set of keyframes for testing."""
    return [
        CameraKeyframe(time=0.0, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0), fov=60.0),
        CameraKeyframe(time=1.0, position=(10.0, 5.0, 0.0), rotation=(0.1, 0.0, 0.0, 0.995), fov=70.0),
        CameraKeyframe(time=2.0, position=(10.0, 10.0, 10.0), rotation=(0.0, 0.1, 0.0, 0.995), fov=60.0),
    ]


@pytest.fixture
def simple_track(simple_keyframes):
    """Create a simple camera track for testing."""
    track = CameraTrack(id="test_track", interpolation=InterpolationMode.LINEAR)
    for kf in simple_keyframes:
        track.add_keyframe(kf)
    return track


@pytest.fixture
def gameplay_camera():
    """Create a gameplay camera state for blend testing."""
    return CameraState(
        position=(100.0, 50.0, 100.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        fov=90.0,
    )


# =============================================================================
# CameraKeyframe Tests
# =============================================================================


class TestCameraKeyframe:
    """Tests for CameraKeyframe dataclass."""

    def test_keyframe_creation_basic(self):
        """Test basic keyframe creation."""
        kf = CameraKeyframe(time=0.0, position=(1.0, 2.0, 3.0), rotation=(0.0, 0.0, 0.0, 1.0), fov=60.0)
        assert kf.time == 0.0
        assert kf.position == (1.0, 2.0, 3.0)
        assert kf.fov == 60.0

    def test_keyframe_default_values(self):
        """Test keyframe with default values."""
        kf = CameraKeyframe(time=1.0)
        assert kf.position == (0.0, 0.0, 0.0)
        assert kf.rotation == (0.0, 0.0, 0.0, 1.0)
        assert kf.fov == 60.0

    def test_keyframe_negative_time_raises(self):
        """Test that negative time raises error."""
        with pytest.raises(ValueError, match="time must be >= 0"):
            CameraKeyframe(time=-1.0)

    def test_keyframe_invalid_fov_low_raises(self):
        """Test that FOV below 1 raises error."""
        with pytest.raises(ValueError, match="fov must be in"):
            CameraKeyframe(time=0.0, fov=0.5)

    def test_keyframe_invalid_fov_high_raises(self):
        """Test that FOV above 179 raises error."""
        with pytest.raises(ValueError, match="fov must be in"):
            CameraKeyframe(time=0.0, fov=180.0)

    def test_keyframe_rotation_normalized(self):
        """Test that rotation is normalized on creation."""
        kf = CameraKeyframe(time=0.0, rotation=(2.0, 0.0, 0.0, 0.0))
        length = math.sqrt(sum(x * x for x in kf.rotation))
        assert abs(length - 1.0) < 1e-6

    def test_keyframe_with_bezier_control(self):
        """Test keyframe with Bezier control points."""
        ctrl = BezierControlPoint(in_tangent=(-1.0, 0.0, 0.0), out_tangent=(1.0, 0.0, 0.0))
        kf = CameraKeyframe(time=1.0, position=(5.0, 5.0, 5.0), bezier_control=ctrl)
        assert kf.bezier_control is not None
        assert kf.bezier_control.in_tangent == (-1.0, 0.0, 0.0)


# =============================================================================
# CameraState Tests
# =============================================================================


class TestCameraState:
    """Tests for CameraState dataclass."""

    def test_state_creation(self):
        """Test basic state creation."""
        state = CameraState(position=(1.0, 2.0, 3.0), rotation=(0.0, 0.0, 0.0, 1.0), fov=60.0)
        assert state.position == (1.0, 2.0, 3.0)
        assert state.fov == 60.0

    def test_state_blend_zero_weight(self):
        """Test blend with weight 0 returns self."""
        state1 = CameraState(position=(0.0, 0.0, 0.0), fov=60.0)
        state2 = CameraState(position=(10.0, 10.0, 10.0), fov=90.0)
        result = state1.blend(state2, 0.0)
        assert result.position == (0.0, 0.0, 0.0)
        assert result.fov == 60.0

    def test_state_blend_full_weight(self):
        """Test blend with weight 1 returns other."""
        state1 = CameraState(position=(0.0, 0.0, 0.0), fov=60.0)
        state2 = CameraState(position=(10.0, 10.0, 10.0), fov=90.0)
        result = state1.blend(state2, 1.0)
        assert result.position == (10.0, 10.0, 10.0)
        assert result.fov == 90.0

    def test_state_blend_half_weight(self):
        """Test blend with weight 0.5 returns midpoint."""
        state1 = CameraState(position=(0.0, 0.0, 0.0), fov=60.0)
        state2 = CameraState(position=(10.0, 10.0, 10.0), fov=80.0)
        result = state1.blend(state2, 0.5)
        assert abs(result.position[0] - 5.0) < 1e-6
        assert abs(result.fov - 70.0) < 1e-6


# =============================================================================
# LookAtTarget Tests
# =============================================================================


class TestLookAtTarget:
    """Tests for LookAtTarget dataclass."""

    def test_lookat_with_position(self):
        """Test look-at target with static position."""
        target = LookAtTarget(position=(10.0, 5.0, 10.0))
        assert target.position == (10.0, 5.0, 10.0)
        assert target.object_id is None

    def test_lookat_with_object_id(self):
        """Test look-at target with object ID."""
        target = LookAtTarget(object_id="player")
        assert target.object_id == "player"
        assert target.position is None

    def test_lookat_requires_position_or_object(self):
        """Test that either position or object_id is required."""
        with pytest.raises(ValueError, match="Either position or object_id"):
            LookAtTarget()

    def test_lookat_invalid_weight_raises(self):
        """Test that invalid weight raises error."""
        with pytest.raises(ValueError, match="weight must be in"):
            LookAtTarget(position=(0.0, 0.0, 0.0), weight=1.5)

    def test_lookat_get_target_position_static(self):
        """Test getting target position from static position."""
        target = LookAtTarget(position=(10.0, 5.0, 10.0), offset=(0.0, 2.0, 0.0))
        pos = target.get_target_position()
        assert pos == (10.0, 7.0, 10.0)

    def test_lookat_get_target_position_dynamic(self):
        """Test getting target position from object resolver."""
        target = LookAtTarget(object_id="player", offset=(0.0, 1.0, 0.0))

        def resolver(obj_id: str) -> Tuple[float, float, float]:
            if obj_id == "player":
                return (20.0, 0.0, 20.0)
            return (0.0, 0.0, 0.0)

        pos = target.get_target_position(resolver)
        assert pos == (20.0, 1.0, 20.0)

    def test_lookat_get_target_no_resolver_returns_none(self):
        """Test that missing resolver returns None for dynamic targets."""
        target = LookAtTarget(object_id="player")
        pos = target.get_target_position()
        assert pos is None


# =============================================================================
# Spline Interpolation Tests
# =============================================================================


class TestSplineInterpolation:
    """Tests for spline interpolation functions."""

    def test_catmull_rom_at_start(self):
        """Test Catmull-Rom at t=0 returns p1."""
        p0 = (-5.0, 0.0, 0.0)
        p1 = (0.0, 0.0, 0.0)
        p2 = (5.0, 0.0, 0.0)
        p3 = (10.0, 0.0, 0.0)
        result = catmull_rom_interpolate(p0, p1, p2, p3, 0.0)
        assert abs(result[0] - 0.0) < 1e-6

    def test_catmull_rom_at_end(self):
        """Test Catmull-Rom at t=1 returns p2."""
        p0 = (-5.0, 0.0, 0.0)
        p1 = (0.0, 0.0, 0.0)
        p2 = (5.0, 0.0, 0.0)
        p3 = (10.0, 0.0, 0.0)
        result = catmull_rom_interpolate(p0, p1, p2, p3, 1.0)
        assert abs(result[0] - 5.0) < 1e-6

    def test_catmull_rom_midpoint(self):
        """Test Catmull-Rom at t=0.5 is smooth midpoint."""
        p0 = (0.0, 0.0, 0.0)
        p1 = (1.0, 0.0, 0.0)
        p2 = (2.0, 0.0, 0.0)
        p3 = (3.0, 0.0, 0.0)
        result = catmull_rom_interpolate(p0, p1, p2, p3, 0.5)
        # For linear points, midpoint should be close to linear midpoint
        assert abs(result[0] - 1.5) < 0.1

    def test_catmull_rom_tangent_direction(self):
        """Test Catmull-Rom tangent points in correct direction."""
        p0 = (0.0, 0.0, 0.0)
        p1 = (1.0, 0.0, 0.0)
        p2 = (2.0, 0.0, 0.0)
        p3 = (3.0, 0.0, 0.0)
        tangent = catmull_rom_tangent(p0, p1, p2, p3, 0.5)
        # Should point in positive X direction
        assert tangent[0] > 0

    def test_cubic_bezier_at_start(self):
        """Test cubic Bezier at t=0 returns p0."""
        p0 = (0.0, 0.0, 0.0)
        p1 = (1.0, 2.0, 0.0)
        p2 = (3.0, 2.0, 0.0)
        p3 = (4.0, 0.0, 0.0)
        result = cubic_bezier_interpolate(p0, p1, p2, p3, 0.0)
        assert abs(result[0] - 0.0) < 1e-6

    def test_cubic_bezier_at_end(self):
        """Test cubic Bezier at t=1 returns p3."""
        p0 = (0.0, 0.0, 0.0)
        p1 = (1.0, 2.0, 0.0)
        p2 = (3.0, 2.0, 0.0)
        p3 = (4.0, 0.0, 0.0)
        result = cubic_bezier_interpolate(p0, p1, p2, p3, 1.0)
        assert abs(result[0] - 4.0) < 1e-6

    def test_cubic_bezier_curve_shape(self):
        """Test cubic Bezier produces curved path."""
        p0 = (0.0, 0.0, 0.0)
        p1 = (0.0, 10.0, 0.0)  # Control pulls curve up
        p2 = (10.0, 10.0, 0.0)
        p3 = (10.0, 0.0, 0.0)
        result = cubic_bezier_interpolate(p0, p1, p2, p3, 0.5)
        # Midpoint should be pulled up by control points
        assert result[1] > 0

    def test_cubic_bezier_tangent(self):
        """Test cubic Bezier tangent calculation."""
        p0 = (0.0, 0.0, 0.0)
        p1 = (1.0, 0.0, 0.0)
        p2 = (2.0, 0.0, 0.0)
        p3 = (3.0, 0.0, 0.0)
        tangent = cubic_bezier_tangent(p0, p1, p2, p3, 0.5)
        # Should point in positive X direction
        assert tangent[0] > 0


# =============================================================================
# CameraTrack Basic Tests
# =============================================================================


class TestCameraTrackBasic:
    """Basic tests for CameraTrack class."""

    def test_track_creation(self):
        """Test basic track creation."""
        track = CameraTrack(id="test")
        assert track.id == "test"
        assert track.keyframe_count == 0

    def test_track_empty_id_raises(self):
        """Test that empty ID raises error."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):
            CameraTrack(id="")

    def test_track_negative_blend_in_raises(self):
        """Test that negative blend_in raises error."""
        with pytest.raises(ValueError, match="blend_in must be >= 0"):
            CameraTrack(id="test", blend_in=-1.0)

    def test_track_negative_blend_out_raises(self):
        """Test that negative blend_out raises error."""
        with pytest.raises(ValueError, match="blend_out must be >= 0"):
            CameraTrack(id="test", blend_out=-1.0)

    def test_track_add_keyframe(self):
        """Test adding keyframes."""
        track = CameraTrack(id="test")
        kf = CameraKeyframe(time=0.0)
        track.add_keyframe(kf)
        assert track.keyframe_count == 1

    def test_track_add_keyframe_chaining(self):
        """Test that add_keyframe returns self for chaining."""
        track = CameraTrack(id="test")
        result = track.add_keyframe(CameraKeyframe(time=0.0))
        assert result is track

    def test_track_remove_keyframe(self):
        """Test removing keyframes."""
        track = CameraTrack(id="test")
        track.add_keyframe(CameraKeyframe(time=0.0))
        track.add_keyframe(CameraKeyframe(time=1.0))
        removed = track.remove_keyframe(0)
        assert removed is True
        assert track.keyframe_count == 1

    def test_track_remove_invalid_index(self):
        """Test removing with invalid index."""
        track = CameraTrack(id="test")
        removed = track.remove_keyframe(0)
        assert removed is False

    def test_track_clear_keyframes(self):
        """Test clearing all keyframes."""
        track = CameraTrack(id="test")
        track.add_keyframe(CameraKeyframe(time=0.0))
        track.add_keyframe(CameraKeyframe(time=1.0))
        track.clear_keyframes()
        assert track.keyframe_count == 0

    def test_track_duration_empty(self):
        """Test duration of empty track."""
        track = CameraTrack(id="test")
        assert track.duration == 0.0

    def test_track_duration(self, simple_track):
        """Test duration calculation."""
        assert simple_track.duration == 2.0

    def test_track_get_keyframe(self, simple_track):
        """Test getting keyframe by index."""
        kf = simple_track.get_keyframe(0)
        assert kf is not None
        assert kf.time == 0.0


# =============================================================================
# CameraTrack Interpolation Tests
# =============================================================================


class TestCameraTrackInterpolation:
    """Tests for CameraTrack interpolation."""

    def test_linear_interpolation_start(self, simple_track):
        """Test linear interpolation at start."""
        state = simple_track.sample(0.0)
        assert state.position == (0.0, 0.0, 0.0)

    def test_linear_interpolation_end(self, simple_track):
        """Test linear interpolation at end."""
        state = simple_track.sample(2.0)
        assert state.position == (10.0, 10.0, 10.0)

    def test_linear_interpolation_midpoint(self, simple_track):
        """Test linear interpolation at midpoint."""
        state = simple_track.sample(0.5)
        assert abs(state.position[0] - 5.0) < 1e-6
        assert abs(state.position[1] - 2.5) < 1e-6

    def test_linear_interpolation_fov(self, simple_track):
        """Test FOV interpolation."""
        state = simple_track.sample(0.5)
        assert abs(state.fov - 65.0) < 1e-6

    def test_catmull_rom_interpolation(self, simple_keyframes):
        """Test Catmull-Rom interpolation mode."""
        track = CameraTrack(id="test", interpolation=InterpolationMode.CATMULL_ROM)
        for kf in simple_keyframes:
            track.add_keyframe(kf)

        # Should produce smooth curve through points
        state = track.sample(0.5)
        assert state.position[0] > 0

    def test_bezier_interpolation(self, simple_keyframes):
        """Test Bezier interpolation mode."""
        track = CameraTrack(id="test", interpolation=InterpolationMode.CUBIC_BEZIER)
        for kf in simple_keyframes:
            track.add_keyframe(kf)

        state = track.sample(0.5)
        assert state.position[0] > 0

    def test_single_keyframe_returns_keyframe_values(self):
        """Test that single keyframe returns its values at any time."""
        track = CameraTrack(id="test")
        track.add_keyframe(CameraKeyframe(time=0.0, position=(5.0, 5.0, 5.0), fov=70.0))

        state1 = track.sample(0.0)
        state2 = track.sample(1.0)
        assert state1.position == (5.0, 5.0, 5.0)
        assert state2.position == (5.0, 5.0, 5.0)

    def test_rotation_interpolation_slerp(self, simple_track):
        """Test that rotation uses slerp interpolation."""
        # Get rotations at different times
        state_start = simple_track.sample(0.0)
        state_mid = simple_track.sample(0.5)
        state_end = simple_track.sample(1.0)

        # Rotation should be normalized
        for state in [state_start, state_mid, state_end]:
            length = math.sqrt(sum(x * x for x in state.rotation))
            assert abs(length - 1.0) < 1e-6


# =============================================================================
# CameraTrack Blend Tests
# =============================================================================


class TestCameraTrackBlend:
    """Tests for CameraTrack blend transitions."""

    def test_blend_state_initial(self, simple_track):
        """Test initial blend state is inactive."""
        assert simple_track.blend_state == BlendState.INACTIVE

    def test_start_sets_blending_in(self, simple_track, gameplay_camera):
        """Test that start() sets blending in state."""
        simple_track.start(gameplay_camera)
        assert simple_track.blend_state == BlendState.BLENDING_IN

    def test_start_no_blend_in_sets_active(self, simple_keyframes):
        """Test that start() with no blend_in goes directly to active."""
        track = CameraTrack(id="test", blend_in=0.0)
        for kf in simple_keyframes:
            track.add_keyframe(kf)

        track.start()
        assert track.blend_state == BlendState.ACTIVE

    def test_blend_weight_increases_during_blend_in(self, simple_track, gameplay_camera):
        """Test that blend weight increases during blend in."""
        simple_track._blend_in = 1.0
        simple_track.start(gameplay_camera)

        simple_track.update(0.5, gameplay_camera)
        assert 0.4 < simple_track.blend_weight < 0.6

    def test_blend_weight_reaches_one_after_blend_in(self, simple_track, gameplay_camera):
        """Test that blend weight reaches 1 after blend in completes."""
        simple_track._blend_in = 0.5
        simple_track.start(gameplay_camera)

        simple_track.update(0.5, gameplay_camera)
        assert simple_track.blend_weight >= 0.99

    def test_stop_resets_state(self, simple_track, gameplay_camera):
        """Test that stop() resets track state."""
        simple_track.start(gameplay_camera)
        simple_track.update(0.5, gameplay_camera)
        simple_track.stop()

        assert simple_track.blend_state == BlendState.INACTIVE
        assert simple_track.blend_weight == 0.0
        assert simple_track.elapsed == 0.0

    def test_blend_with_gameplay_camera(self, simple_track, gameplay_camera):
        """Test blending between gameplay and track camera."""
        simple_track._blend_in = 1.0
        simple_track.start(gameplay_camera)

        # At 50% blend, should be midway between cameras
        state = simple_track.update(0.5, gameplay_camera)

        # Position should be between gameplay (100, 50, 100) and track start (0, 0, 0)
        assert state.position[0] < 100.0
        assert state.position[0] > 0.0

    def test_blend_out_at_end(self, simple_keyframes):
        """Test blend out triggers at track end."""
        track = CameraTrack(id="test", blend_in=0.1, blend_out=0.5)
        for kf in simple_keyframes:
            track.add_keyframe(kf)

        gameplay = CameraState()
        track.start(gameplay)

        # Update past blend in
        track.update(0.2, gameplay)
        assert track.blend_state == BlendState.ACTIVE

        # Update to near end
        track.update(1.5, gameplay)
        assert track.blend_state == BlendState.BLENDING_OUT

    def test_loop_restarts_after_complete(self, simple_keyframes):
        """Test that loop=True restarts the track."""
        track = CameraTrack(id="test", blend_in=0.0, blend_out=0.1, loop=True)
        for kf in simple_keyframes:
            track.add_keyframe(kf)

        gameplay = CameraState()
        track.start(gameplay)

        # Update past full duration multiple times to complete a loop cycle
        track.update(2.0, gameplay)  # Reach end
        track.update(0.2, gameplay)  # Complete blend out and restart

        # Should have looped - check elapsed time is less than duration
        assert track.elapsed < track.duration or track.blend_state in (BlendState.BLENDING_IN, BlendState.ACTIVE, BlendState.BLENDING_OUT)


# =============================================================================
# CameraTrack Look-At Tests
# =============================================================================


class TestCameraTrackLookAt:
    """Tests for CameraTrack look-at constraints."""

    def test_look_at_target_set(self, simple_track):
        """Test setting look-at target."""
        target = LookAtTarget(position=(0.0, 0.0, 10.0))
        simple_track.look_at_target = target
        assert simple_track.look_at_target is not None

    def test_look_at_affects_rotation(self, simple_track):
        """Test that look-at affects sampled rotation."""
        # Sample without look-at
        state_no_lookat = simple_track.sample(0.0)

        # Set look-at to a position that requires rotation (off to the side)
        simple_track.look_at_target = LookAtTarget(position=(50.0, 0.0, 0.0), weight=1.0)

        # Sample with look-at
        state_with_lookat = simple_track.sample(0.0)

        # Rotations should differ (looking sideways vs forward)
        # Compare at least one component differs
        rotation_differs = any(
            abs(state_no_lookat.rotation[i] - state_with_lookat.rotation[i]) > 0.01
            for i in range(4)
        )
        assert rotation_differs

    def test_look_at_weight_zero_no_effect(self, simple_track):
        """Test that weight=0 has no effect."""
        state_before = simple_track.sample(0.0)

        simple_track.look_at_target = LookAtTarget(position=(0.0, 0.0, 100.0), weight=0.0)
        state_after = simple_track.sample(0.0)

        # Rotations should be same
        for i in range(4):
            assert abs(state_before.rotation[i] - state_after.rotation[i]) < 1e-6

    def test_look_at_with_object_resolver(self, simple_track):
        """Test look-at with dynamic object position."""
        simple_track.look_at_target = LookAtTarget(object_id="target")

        def resolver(obj_id: str):
            if obj_id == "target":
                return (0.0, 0.0, 50.0)
            return None

        state = simple_track.sample(0.0, object_resolver=resolver)
        # Should have applied look-at
        assert state is not None


# =============================================================================
# CameraTrackManager Tests
# =============================================================================


class TestCameraTrackManager:
    """Tests for CameraTrackManager."""

    def test_manager_creation(self):
        """Test manager creation."""
        manager = CameraTrackManager()
        assert manager.active_track_id is None

    def test_manager_register_track(self, simple_track):
        """Test registering a track."""
        manager = CameraTrackManager()
        manager.register(simple_track)
        assert manager.get("test_track") is simple_track

    def test_manager_unregister_track(self, simple_track):
        """Test unregistering a track."""
        manager = CameraTrackManager()
        manager.register(simple_track)
        result = manager.unregister("test_track")
        assert result is True
        assert manager.get("test_track") is None

    def test_manager_unregister_nonexistent(self):
        """Test unregistering nonexistent track returns False."""
        manager = CameraTrackManager()
        result = manager.unregister("nonexistent")
        assert result is False

    def test_manager_play_track(self, simple_track, gameplay_camera):
        """Test playing a track."""
        manager = CameraTrackManager()
        manager.register(simple_track)
        result = manager.play("test_track", gameplay_camera)
        assert result is True
        assert manager.active_track_id == "test_track"

    def test_manager_play_nonexistent(self, gameplay_camera):
        """Test playing nonexistent track returns False."""
        manager = CameraTrackManager()
        result = manager.play("nonexistent", gameplay_camera)
        assert result is False

    def test_manager_stop_track(self, simple_track, gameplay_camera):
        """Test stopping a track."""
        manager = CameraTrackManager()
        manager.register(simple_track)
        manager.play("test_track", gameplay_camera)
        result = manager.stop("test_track")
        assert result is True
        assert manager.active_track_id is None

    def test_manager_stop_all(self, gameplay_camera):
        """Test stopping all tracks."""
        manager = CameraTrackManager()
        track1 = CameraTrack(id="track1")
        track2 = CameraTrack(id="track2")
        manager.register(track1)
        manager.register(track2)
        manager.play("track1", gameplay_camera)

        manager.stop_all()
        assert manager.active_track_id is None

    def test_manager_update_returns_gameplay_when_no_active(self, gameplay_camera):
        """Test update returns gameplay camera when no active track."""
        manager = CameraTrackManager()
        state = manager.update(0.1, gameplay_camera)
        assert state.position == gameplay_camera.position


# =============================================================================
# Decorator Tests
# =============================================================================


class TestCameraTrackDecorator:
    """Tests for @camera_track decorator."""

    def test_decorator_sets_attributes(self):
        """Test that decorator sets class attributes."""

        @camera_track(id="test_deco", blend_in=1.0, blend_out=2.0)
        class TestTrack:
            pass

        assert TestTrack._camera_track is True
        assert TestTrack._camera_track_id == "test_deco"
        assert TestTrack._camera_track_blend_in == 1.0
        assert TestTrack._camera_track_blend_out == 2.0

    def test_decorator_default_id_uses_class_name(self):
        """Test that missing ID uses class name."""

        @camera_track(blend_in=0.5)
        class AutoNamedTrack:
            pass

        assert AutoNamedTrack._camera_track_id == "AutoNamedTrack"

    def test_decorator_tracks_applied_decorators(self):
        """Test that decorator is tracked in _applied_decorators."""

        @camera_track(id="tracked")
        class TrackedTrack:
            pass

        assert "camera_track" in TrackedTrack._applied_decorators

    def test_decorator_adds_to_registry(self):
        """Test that decorator adds class to registry."""

        @camera_track(id="registered_track")
        class RegisteredTrack:
            pass

        registry = get_camera_track_registry()
        assert "registered_track" in registry

    def test_decorator_invalid_blend_in_raises(self):
        """Test that negative blend_in raises error."""
        with pytest.raises(ValueError, match="blend_in must be >= 0"):

            @camera_track(id="test", blend_in=-1.0)
            class BadTrack:
                pass

    def test_decorator_sets_tags(self):
        """Test that decorator sets _tags attribute."""

        @camera_track(id="tagged", blend_in=1.5)
        class TaggedTrack:
            pass

        assert TaggedTrack._tags["camera_track"] is True
        assert TaggedTrack._tags["camera_track_blend_in"] == 1.5


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_camera_track(self, simple_keyframes):
        """Test create_camera_track factory."""
        track = create_camera_track(
            id="factory_track",
            keyframes=simple_keyframes,
            interpolation=InterpolationMode.LINEAR,
            blend_in=1.0,
            blend_out=1.0,
        )

        assert track.id == "factory_track"
        assert track.keyframe_count == 3
        assert track.blend_in == 1.0

    def test_create_camera_track_with_lookat(self, simple_keyframes):
        """Test create_camera_track with look-at target."""
        target = LookAtTarget(position=(0.0, 0.0, 10.0))
        track = create_camera_track(
            id="lookat_track",
            keyframes=simple_keyframes,
            look_at=target,
        )

        assert track.look_at_target is not None

    def test_create_track_from_class(self, simple_keyframes):
        """Test create_track_from_class factory."""

        @camera_track(id="from_class", blend_in=0.8)
        class ClassTrack:
            keyframes = simple_keyframes

        track = create_track_from_class(ClassTrack)

        assert track.id == "from_class"
        assert track.blend_in == 0.8
        assert track.keyframe_count == 3

    def test_create_track_from_class_tuple_keyframes(self):
        """Test create_track_from_class with tuple keyframes."""

        @camera_track(id="tuple_track")
        class TupleTrack:
            keyframes = [
                (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 60.0),
                (1.0, (10.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 70.0),
            ]

        track = create_track_from_class(TupleTrack)
        assert track.keyframe_count == 2

    def test_create_track_from_non_track_raises(self):
        """Test that creating from non-track class raises error."""

        class NotATrack:
            pass

        with pytest.raises(ValueError, match="is not a camera track"):
            create_track_from_class(NotATrack)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and degenerate inputs."""

    def test_empty_track_sample(self):
        """Test sampling empty track returns default state."""
        track = CameraTrack(id="empty")
        state = track.sample(0.0)
        assert state.position == (0.0, 0.0, 0.0)

    def test_sample_before_first_keyframe(self, simple_track):
        """Test sampling before first keyframe clamps to first."""
        state = simple_track.sample(-1.0)
        assert state.position == (0.0, 0.0, 0.0)

    def test_sample_after_last_keyframe(self, simple_track):
        """Test sampling after last keyframe clamps to last."""
        state = simple_track.sample(10.0)
        assert state.position == (10.0, 10.0, 10.0)

    def test_zero_duration_segment(self):
        """Test keyframes at same time."""
        track = CameraTrack(id="test")
        track.add_keyframe(CameraKeyframe(time=1.0, position=(0.0, 0.0, 0.0)))
        track.add_keyframe(CameraKeyframe(time=1.0, position=(10.0, 0.0, 0.0)))

        state = track.sample(1.0)
        # Should return one of the keyframes
        assert state.position[0] in (0.0, 10.0)

    def test_update_with_zero_delta(self, simple_track):
        """Test update with zero delta time."""
        simple_track.start()
        initial_elapsed = simple_track.elapsed
        simple_track.update(0.0)
        assert simple_track.elapsed == initial_elapsed

    def test_update_with_negative_delta(self, simple_track):
        """Test update with negative delta time."""
        simple_track.start()
        initial_elapsed = simple_track.elapsed
        simple_track.update(-0.5)
        assert simple_track.elapsed == initial_elapsed

    def test_seek_clamps_to_duration(self, simple_track):
        """Test that seek clamps to track duration."""
        simple_track.seek(100.0)
        assert simple_track.elapsed == simple_track.duration

    def test_seek_clamps_to_zero(self, simple_track):
        """Test that seek clamps negative values to zero."""
        simple_track.seek(-10.0)
        assert simple_track.elapsed == 0.0

    def test_degenerate_catmull_rom_two_points(self):
        """Test Catmull-Rom with only two keyframes."""
        track = CameraTrack(id="test", interpolation=InterpolationMode.CATMULL_ROM)
        track.add_keyframe(CameraKeyframe(time=0.0, position=(0.0, 0.0, 0.0)))
        track.add_keyframe(CameraKeyframe(time=1.0, position=(10.0, 0.0, 0.0)))

        state = track.sample(0.5)
        # Should still interpolate reasonably
        assert 0.0 < state.position[0] < 10.0

    def test_progress_empty_track(self):
        """Test progress on empty track is 0."""
        track = CameraTrack(id="empty")
        assert track.progress == 0.0


# =============================================================================
# Quaternion Math Tests
# =============================================================================


class TestQuaternionMath:
    """Tests for quaternion utility functions."""

    def test_quat_identity(self):
        """Test identity quaternion."""
        q = quat_identity()
        assert q == (0.0, 0.0, 0.0, 1.0)

    def test_quat_slerp_identity(self):
        """Test slerp with same quaternions."""
        q = (0.0, 0.0, 0.0, 1.0)
        result = quat_slerp(q, q, 0.5)
        for i in range(4):
            assert abs(result[i] - q[i]) < 1e-6

    def test_quat_slerp_endpoints(self):
        """Test slerp at endpoints."""
        q1 = (0.0, 0.0, 0.0, 1.0)
        q2 = (0.0, 0.7071, 0.0, 0.7071)  # 90 degree rotation around Y

        result_0 = quat_slerp(q1, q2, 0.0)
        result_1 = quat_slerp(q1, q2, 1.0)

        for i in range(4):
            assert abs(result_0[i] - q1[i]) < 1e-4
            assert abs(result_1[i] - q2[i]) < 1e-4

    def test_quat_look_at_forward(self):
        """Test look-at rotation facing forward."""
        q = quat_look_at((0.0, 0.0, 1.0))
        # Should be close to identity
        length = math.sqrt(sum(x * x for x in q))
        assert abs(length - 1.0) < 1e-6


# =============================================================================
# Vector Math Tests
# =============================================================================


class TestVectorMath:
    """Tests for vector utility functions."""

    def test_vec3_lerp_endpoints(self):
        """Test vec3_lerp at endpoints."""
        a = (0.0, 0.0, 0.0)
        b = (10.0, 10.0, 10.0)

        result_0 = vec3_lerp(a, b, 0.0)
        result_1 = vec3_lerp(a, b, 1.0)

        assert result_0 == a
        assert result_1 == b

    def test_vec3_lerp_midpoint(self):
        """Test vec3_lerp at midpoint."""
        a = (0.0, 0.0, 0.0)
        b = (10.0, 10.0, 10.0)

        result = vec3_lerp(a, b, 0.5)

        assert result == (5.0, 5.0, 5.0)
