"""Tests for XR reprojection module."""

import pytest
import math
import time

from engine.xr.rendering.reprojection import (
    ReprojectionMode,
    PredictionMethod,
    Pose,
    PoseVelocity,
    MotionVector,
    ReprojectionConfig,
    ReprojectionMetrics,
    ReprojectedFrame,
    XRReprojection,
    ATWReprojection,
    ASWReprojection,
    HybridReprojection,
    create_reprojection,
)
from engine.xr.utils.math_utils import multiply_quaternions


class TestPose:
    """Tests for Pose dataclass."""

    def test_default_pose(self):
        """Test default pose values."""
        pose = Pose()

        assert pose.position == (0.0, 0.0, 0.0)
        assert pose.orientation == (0.0, 0.0, 0.0, 1.0)  # Identity quaternion
        assert pose.timestamp_ns == 0

    def test_custom_pose(self):
        """Test custom pose values."""
        pose = Pose(
            position=(1.0, 1.6, 0.5),
            orientation=(0.0, 0.707, 0.0, 0.707),
            timestamp_ns=1000000
        )

        assert pose.position == (1.0, 1.6, 0.5)
        assert pose.orientation[1] == pytest.approx(0.707)


class TestPoseVelocity:
    """Tests for PoseVelocity dataclass."""

    def test_default_velocity(self):
        """Test default velocity values."""
        vel = PoseVelocity()

        assert vel.linear == (0.0, 0.0, 0.0)
        assert vel.angular == (0.0, 0.0, 0.0)

    def test_custom_velocity(self):
        """Test custom velocity values."""
        vel = PoseVelocity(
            linear=(1.0, 0.0, -0.5),
            angular=(0.1, 0.2, 0.0)
        )

        assert vel.linear == (1.0, 0.0, -0.5)
        assert vel.angular == (0.1, 0.2, 0.0)


class TestReprojectionConfig:
    """Tests for ReprojectionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReprojectionConfig()

        assert config.mode == ReprojectionMode.ATW
        assert config.enabled is True
        assert config.target_frame_time_ms == pytest.approx(11.11, rel=0.01)
        assert config.prediction_method == PredictionMethod.LINEAR

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReprojectionConfig(
            mode=ReprojectionMode.ASW,
            prediction_method=PredictionMethod.QUADRATIC,
            prediction_horizon_ms=30.0
        )

        assert config.mode == ReprojectionMode.ASW
        assert config.prediction_method == PredictionMethod.QUADRATIC
        assert config.prediction_horizon_ms == pytest.approx(30.0)


class TestATWReprojection:
    """Tests for ATWReprojection."""

    def test_creation_default_config(self):
        """Test ATW creation with default config."""
        atw = ATWReprojection()

        assert atw.config.mode == ReprojectionMode.ATW

    def test_creation_custom_config(self):
        """Test ATW creation with custom config."""
        config = ReprojectionConfig(
            atw_rotation_limit=0.2,
            prediction_horizon_ms=25.0
        )
        atw = ATWReprojection(config)

        assert atw.config.atw_rotation_limit == pytest.approx(0.2)

    def test_configure(self):
        """Test configuration update."""
        atw = ATWReprojection()

        new_config = ReprojectionConfig(atw_rotation_limit=0.15)
        atw.configure(new_config)

        assert atw.config.mode == ReprojectionMode.ATW
        assert atw.config.atw_rotation_limit == pytest.approx(0.15)

    def test_submit_pose(self):
        """Test pose submission."""
        atw = ATWReprojection()

        pose = Pose(
            position=(0.0, 1.6, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            timestamp_ns=1000000
        )

        # Should not raise
        atw.submit_pose(pose)

    def test_submit_pose_with_velocity(self):
        """Test pose submission with velocity."""
        atw = ATWReprojection()

        pose = Pose(
            position=(0.0, 1.6, 0.0),
            timestamp_ns=1000000
        )
        velocity = PoseVelocity(
            linear=(0.0, 0.0, -1.0),
            angular=(0.0, 0.5, 0.0)
        )

        atw.submit_pose(pose, velocity)

    def test_predict_pose_no_motion(self):
        """Test pose prediction with stationary pose."""
        atw = ATWReprojection()

        pose = Pose(
            position=(0.0, 1.6, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            timestamp_ns=1000000
        )

        atw.submit_pose(pose)

        # Predict at same time
        predicted = atw.predict_pose(1000000)

        assert predicted.position[1] == pytest.approx(1.6)

    def test_predict_pose_with_velocity(self):
        """Test pose prediction with linear velocity."""
        config = ReprojectionConfig(
            prediction_method=PredictionMethod.LINEAR,
            max_prediction_ms=200.0  # Allow longer prediction for test
        )
        atw = ATWReprojection(config)

        pose = Pose(
            position=(0.0, 1.6, 0.0),
            timestamp_ns=0
        )
        velocity = PoseVelocity(
            linear=(0.0, 0.0, -1.0),  # Moving forward 1 m/s
            angular=(0.0, 0.0, 0.0)
        )

        atw.submit_pose(pose, velocity)

        # Predict 100ms into future
        predicted = atw.predict_pose(100_000_000)  # 100ms in ns

        # Should have moved -0.1m in Z
        assert predicted.position[2] == pytest.approx(-0.1, rel=0.01)

    def test_predict_pose_with_rotation(self):
        """Test pose prediction with angular velocity."""
        config = ReprojectionConfig(prediction_method=PredictionMethod.LINEAR)
        atw = ATWReprojection(config)

        pose = Pose(
            position=(0.0, 1.6, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),  # Identity
            timestamp_ns=0
        )
        velocity = PoseVelocity(
            linear=(0.0, 0.0, 0.0),
            angular=(0.0, 1.0, 0.0)  # Rotating around Y at 1 rad/s
        )

        atw.submit_pose(pose, velocity)

        # Predict 100ms into future
        predicted = atw.predict_pose(100_000_000)

        # Orientation should have changed
        assert predicted.orientation != (0.0, 0.0, 0.0, 1.0)

    def test_submit_frame(self):
        """Test frame submission."""
        atw = ATWReprojection()

        pose = Pose(timestamp_ns=1000000)
        atw.submit_frame(pose, frame_id=1)

        # Should not raise

    def test_reproject_without_data(self):
        """Test reprojection without submitted data."""
        atw = ATWReprojection()

        result = atw.reproject(2000000)

        assert result.success is False

    def test_reproject_with_data(self):
        """Test reprojection with submitted data."""
        atw = ATWReprojection()

        # Submit pose
        pose = Pose(
            position=(0.0, 1.6, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            timestamp_ns=0
        )
        atw.submit_pose(pose)

        # Submit frame
        atw.submit_frame(pose, frame_id=1)

        # Reproject
        result = atw.reproject(10_000_000)  # 10ms later

        assert result.success is True
        assert result.mode_used == ReprojectionMode.ATW

    def test_reproject_returns_rotation_delta(self):
        """Test that reprojection returns rotation delta."""
        atw = ATWReprojection()

        pose = Pose(
            position=(0.0, 1.6, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            timestamp_ns=0
        )
        velocity = PoseVelocity(angular=(0.0, 0.5, 0.0))

        atw.submit_pose(pose, velocity)
        atw.submit_frame(pose, frame_id=1)

        result = atw.reproject(50_000_000)  # 50ms later

        assert result.success is True
        # Rotation delta should be non-identity
        assert result.rotation_delta != (0.0, 0.0, 0.0, 1.0)

    def test_get_late_latch_pose(self):
        """Test late latch pose retrieval."""
        atw = ATWReprojection()

        pose = Pose(
            position=(0.0, 1.6, 0.0),
            timestamp_ns=time.time_ns()
        )
        atw.submit_pose(pose)

        late_pose = atw.get_late_latch_pose()

        # Should return a pose predicted slightly into future
        assert late_pose.position[1] == pytest.approx(1.6, rel=0.01)

    def test_get_metrics(self):
        """Test metrics retrieval."""
        atw = ATWReprojection()

        # Do some operations
        pose = Pose(timestamp_ns=0)
        atw.submit_pose(pose)
        atw.submit_frame(pose, 1)
        atw.reproject(10_000_000)

        metrics = atw.get_metrics()

        assert isinstance(metrics, ReprojectionMetrics)
        assert metrics.frames_reprojected >= 1
        assert metrics.atw_corrections >= 1

    def test_rotation_clamping(self):
        """Test that rotation corrections are clamped."""
        config = ReprojectionConfig(atw_rotation_limit=0.1)  # Small limit
        atw = ATWReprojection(config)

        pose = Pose(
            orientation=(0.0, 0.0, 0.0, 1.0),
            timestamp_ns=0
        )
        # Large angular velocity
        velocity = PoseVelocity(angular=(0.0, 10.0, 0.0))  # Very fast rotation

        atw.submit_pose(pose, velocity)
        atw.submit_frame(pose, 1)

        # Long time delta would cause large rotation
        result = atw.reproject(100_000_000)  # 100ms

        # Rotation should be clamped
        # Extract angle from quaternion
        w = result.rotation_delta[3]
        angle = 2.0 * math.acos(min(1.0, abs(w)))

        assert angle <= config.atw_rotation_limit + 0.01  # Small tolerance


class TestASWReprojection:
    """Tests for ASWReprojection."""

    def test_creation(self):
        """Test ASW creation."""
        asw = ASWReprojection()

        assert asw.config.mode == ReprojectionMode.ASW

    def test_submit_motion_vectors(self):
        """Test motion vector submission."""
        asw = ASWReprojection()

        # Create small motion vector grid
        width, height = 8, 8
        vectors = []
        for y in range(height):
            row = []
            for x in range(width):
                row.append(MotionVector(dx=1.0, dy=0.5, depth=0.5))
            vectors.append(row)

        asw.submit_motion_vectors(vectors, width, height)

    def test_reproject_uses_motion_vectors(self):
        """Test that ASW uses motion vectors when available."""
        asw = ASWReprojection()

        pose = Pose(timestamp_ns=0)
        asw.submit_pose(pose)
        asw.submit_frame(pose, 1)

        # Submit motion vectors with significant motion
        width, height = 8, 8
        vectors = []
        for y in range(height):
            row = []
            for x in range(width):
                row.append(MotionVector(dx=10.0, dy=5.0, depth=0.5))
            vectors.append(row)
        asw.submit_motion_vectors(vectors, width, height)

        result = asw.reproject(10_000_000)

        assert result.success is True
        assert result.mode_used == ReprojectionMode.ASW

    def test_fallback_to_atw_for_small_motion(self):
        """Test ASW falls back to ATW for small motion."""
        config = ReprojectionConfig(
            mode=ReprojectionMode.ASW,
            asw_motion_threshold=1.0
        )
        asw = ASWReprojection(config)

        pose = Pose(timestamp_ns=0)
        asw.submit_pose(pose)
        asw.submit_frame(pose, 1)

        # Submit motion vectors with tiny motion
        width, height = 8, 8
        vectors = []
        for y in range(height):
            row = []
            for x in range(width):
                row.append(MotionVector(dx=0.1, dy=0.1, depth=0.5))
            vectors.append(row)
        asw.submit_motion_vectors(vectors, width, height)

        result = asw.reproject(10_000_000)

        # Should fall back to ATW
        assert result.success is True

    def test_metrics_tracks_asw_generations(self):
        """Test that metrics track ASW frame generations."""
        asw = ASWReprojection()

        pose = Pose(timestamp_ns=0)
        asw.submit_pose(pose)
        asw.submit_frame(pose, 1)

        # Large motion vectors
        vectors = [[MotionVector(dx=10.0, dy=10.0)] * 8 for _ in range(8)]
        asw.submit_motion_vectors(vectors, 8, 8)

        asw.reproject(10_000_000)

        metrics = asw.get_metrics()
        assert metrics.asw_generations >= 1


class TestHybridReprojection:
    """Tests for HybridReprojection."""

    def test_creation(self):
        """Test Hybrid creation."""
        hybrid = HybridReprojection()

        assert hybrid.config.mode == ReprojectionMode.HYBRID

    def test_combines_atw_and_asw(self):
        """Test that hybrid combines ATW rotation and ASW translation."""
        hybrid = HybridReprojection()

        pose = Pose(
            position=(0.0, 1.6, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            timestamp_ns=0
        )
        velocity = PoseVelocity(angular=(0.0, 0.5, 0.0))

        hybrid.submit_pose(pose, velocity)
        hybrid.submit_frame(pose, 1)

        # Submit motion vectors
        vectors = [[MotionVector(dx=5.0, dy=2.0, depth=0.5)] * 8 for _ in range(8)]
        hybrid.submit_motion_vectors(vectors, 8, 8)

        result = hybrid.reproject(20_000_000)

        assert result.success is True
        # Should have rotation from ATW
        assert result.rotation_delta != (0.0, 0.0, 0.0, 1.0)


class TestReprojectionFactory:
    """Tests for create_reprojection factory function."""

    def test_create_default(self):
        """Test default reprojection creation."""
        reproj = create_reprojection()

        assert isinstance(reproj, ATWReprojection)

    def test_create_atw(self):
        """Test ATW creation."""
        config = ReprojectionConfig(mode=ReprojectionMode.ATW)
        reproj = create_reprojection(config)

        assert isinstance(reproj, ATWReprojection)

    def test_create_asw(self):
        """Test ASW creation."""
        config = ReprojectionConfig(mode=ReprojectionMode.ASW)
        reproj = create_reprojection(config)

        assert isinstance(reproj, ASWReprojection)

    def test_create_hybrid(self):
        """Test Hybrid creation."""
        config = ReprojectionConfig(mode=ReprojectionMode.HYBRID)
        reproj = create_reprojection(config)

        assert isinstance(reproj, HybridReprojection)

    def test_create_disabled(self):
        """Test disabled reprojection."""
        config = ReprojectionConfig(enabled=False)
        reproj = create_reprojection(config)

        assert not reproj.config.enabled

    def test_create_none_mode(self):
        """Test NONE mode reprojection."""
        config = ReprojectionConfig(mode=ReprojectionMode.NONE)
        reproj = create_reprojection(config)

        assert not reproj.config.enabled


class TestQuaternionMath:
    """Tests for quaternion math operations in reprojection."""

    def test_quaternion_inverse(self):
        """Test quaternion inverse calculation."""
        atw = ATWReprojection()

        q = (0.0, 0.707, 0.0, 0.707)  # 90 deg Y rotation
        inv = atw._quaternion_inverse(q)

        # For unit quaternion, inverse is conjugate
        assert inv == (-0.0, -0.707, -0.0, 0.707)

    def test_quaternion_multiply_identity(self):
        """Test quaternion multiplication with identity."""
        identity = (0.0, 0.0, 0.0, 1.0)
        q = (0.1, 0.2, 0.3, 0.9)

        result = multiply_quaternions(q, identity)

        for i in range(4):
            assert result[i] == pytest.approx(q[i], rel=1e-5)

    def test_angular_velocity_integration(self):
        """Test angular velocity integration."""
        atw = ATWReprojection()

        orientation = (0.0, 0.0, 0.0, 1.0)  # Identity
        angular_vel = (0.0, 1.0, 0.0)  # 1 rad/s around Y
        dt = 0.1  # 100ms

        result = atw._integrate_angular_velocity(orientation, angular_vel, dt)

        # Should have rotated about 0.1 radians around Y
        # Check quaternion is valid (normalized)
        mag = math.sqrt(sum(x * x for x in result))
        assert mag == pytest.approx(1.0, rel=1e-5)

        # Y component should be non-zero (rotation around Y)
        assert abs(result[1]) > 0.01
