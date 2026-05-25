"""XR reprojection techniques for latency compensation.

Implements:
- ATW (Asynchronous Timewarp): Rotation-only correction
- ASW (Asynchronous Spacewarp): Frame generation with motion vectors
- Pose prediction and late latching

Target: Maintain 90Hz/120Hz even when rendering drops frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Deque
from collections import deque
import math
import threading
import time

from engine.xr.utils.math_utils import multiply_quaternions


class ReprojectionMode(Enum):
    """Reprojection technique mode."""
    NONE = auto()           # No reprojection
    ATW = auto()            # Asynchronous Timewarp (rotation only)
    ASW = auto()            # Asynchronous Spacewarp (full motion)
    HYBRID = auto()         # ATW + partial ASW


class PredictionMethod(Enum):
    """Pose prediction method."""
    NONE = auto()           # No prediction
    LINEAR = auto()         # Linear extrapolation
    QUADRATIC = auto()      # Quadratic (velocity + acceleration)
    KALMAN = auto()         # Kalman filter


@dataclass
class Pose:
    """6DOF pose with timestamp."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # quaternion
    timestamp_ns: int = 0


@dataclass
class PoseVelocity:
    """Pose velocity data."""
    linear: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # m/s
    angular: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # rad/s


@dataclass
class MotionVector:
    """Per-pixel motion vector."""
    dx: float = 0.0  # Horizontal motion in pixels
    dy: float = 0.0  # Vertical motion in pixels
    depth: float = 1.0  # Normalized depth


@dataclass
class ReprojectionConfig:
    """Reprojection configuration."""
    mode: ReprojectionMode = ReprojectionMode.ATW
    enabled: bool = True

    # Timing targets
    target_frame_time_ms: float = 11.11  # ~90Hz
    photon_time_offset_ms: float = 5.0   # Time from submit to photon

    # Prediction settings
    prediction_method: PredictionMethod = PredictionMethod.LINEAR
    prediction_horizon_ms: float = 20.0  # How far ahead to predict
    max_prediction_ms: float = 50.0      # Safety limit

    # ATW settings
    atw_rotation_limit: float = 0.1      # Max rotation correction (radians)

    # ASW settings
    asw_motion_threshold: float = 0.5    # Min motion for ASW activation
    asw_depth_threshold: float = 0.1     # Depth discontinuity threshold
    asw_interpolation_factor: float = 0.5  # Blend between frames

    # Quality settings
    late_latch_enabled: bool = True
    motion_smoothing: float = 0.8        # Motion vector smoothing


@dataclass
class ReprojectionMetrics:
    """Reprojection performance metrics."""
    frames_reprojected: int = 0
    atw_corrections: int = 0
    asw_generations: int = 0
    average_latency_ms: float = 0.0
    prediction_error_deg: float = 0.0
    dropped_frames: int = 0


@dataclass
class ReprojectedFrame:
    """Result of frame reprojection."""
    success: bool = False
    mode_used: ReprojectionMode = ReprojectionMode.NONE
    predicted_pose: Optional[Pose] = None
    rotation_delta: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    translation_delta: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    timestamp_ns: int = 0


class XRReprojection(ABC):
    """Abstract XR reprojection interface."""

    @property
    @abstractmethod
    def config(self) -> ReprojectionConfig:
        """Get current reprojection configuration."""
        pass

    @abstractmethod
    def configure(self, config: ReprojectionConfig) -> None:
        """Update reprojection configuration."""
        pass

    @abstractmethod
    def submit_pose(self, pose: Pose, velocity: Optional[PoseVelocity] = None) -> None:
        """Submit tracked pose sample for prediction.

        Args:
            pose: Current tracked pose
            velocity: Optional velocity data
        """
        pass

    @abstractmethod
    def predict_pose(self, target_time_ns: int) -> Pose:
        """Predict pose at target time.

        Args:
            target_time_ns: Target timestamp in nanoseconds

        Returns:
            Predicted pose
        """
        pass

    @abstractmethod
    def submit_frame(self, render_pose: Pose, frame_id: int) -> None:
        """Submit rendered frame for reprojection.

        Args:
            render_pose: Pose used for rendering this frame
            frame_id: Unique frame identifier
        """
        pass

    @abstractmethod
    def submit_motion_vectors(self, vectors: List[List[MotionVector]],
                               width: int, height: int) -> None:
        """Submit motion vectors for ASW.

        Args:
            vectors: 2D array of motion vectors
            width: Frame width
            height: Frame height
        """
        pass

    @abstractmethod
    def reproject(self, display_time_ns: int) -> ReprojectedFrame:
        """Perform reprojection for display time.

        Args:
            display_time_ns: Target display timestamp

        Returns:
            Reprojection result
        """
        pass

    @abstractmethod
    def get_late_latch_pose(self) -> Pose:
        """Get latest pose for late latching.

        Called just before compositor submission.

        Returns:
            Latest available pose
        """
        pass

    @abstractmethod
    def get_metrics(self) -> ReprojectionMetrics:
        """Get reprojection metrics."""
        pass


class ATWReprojection(XRReprojection):
    """Asynchronous Timewarp implementation.

    Corrects for head rotation between render time and display time.
    Fast and robust but cannot handle translation.
    """

    def __init__(self, config: Optional[ReprojectionConfig] = None):
        """Initialize ATW reprojection."""
        self._config = config or ReprojectionConfig(mode=ReprojectionMode.ATW)
        self._pose_history: Deque[Tuple[Pose, PoseVelocity]] = deque(maxlen=100)
        self._latest_pose: Optional[Pose] = None
        self._latest_velocity: Optional[PoseVelocity] = None
        self._render_pose: Optional[Pose] = None
        self._render_frame_id: int = 0
        self._metrics = ReprojectionMetrics()
        self._lock = threading.Lock()

    @property
    def config(self) -> ReprojectionConfig:
        return self._config

    def configure(self, config: ReprojectionConfig) -> None:
        with self._lock:
            self._config = config
            self._config.mode = ReprojectionMode.ATW

    def submit_pose(self, pose: Pose, velocity: Optional[PoseVelocity] = None) -> None:
        with self._lock:
            vel = velocity or PoseVelocity()
            self._pose_history.append((pose, vel))
            self._latest_pose = pose
            self._latest_velocity = vel

    def predict_pose(self, target_time_ns: int) -> Pose:
        with self._lock:
            if not self._latest_pose:
                return Pose(timestamp_ns=target_time_ns)

            return self._predict_pose_internal(target_time_ns)

    def _predict_pose_internal(self, target_time_ns: int) -> Pose:
        """Internal pose prediction without lock."""
        if not self._latest_pose:
            return Pose(timestamp_ns=target_time_ns)

        # Calculate time delta
        dt_ns = target_time_ns - self._latest_pose.timestamp_ns
        dt_s = dt_ns / 1_000_000_000.0

        # Clamp prediction horizon
        max_dt = self._config.max_prediction_ms / 1000.0
        dt_s = max(-max_dt, min(max_dt, dt_s))

        if self._config.prediction_method == PredictionMethod.NONE:
            return Pose(
                position=self._latest_pose.position,
                orientation=self._latest_pose.orientation,
                timestamp_ns=target_time_ns
            )

        # Linear prediction
        vel = self._latest_velocity or PoseVelocity()

        # Predict position
        predicted_pos = (
            self._latest_pose.position[0] + vel.linear[0] * dt_s,
            self._latest_pose.position[1] + vel.linear[1] * dt_s,
            self._latest_pose.position[2] + vel.linear[2] * dt_s
        )

        # Predict orientation
        predicted_orient = self._integrate_angular_velocity(
            self._latest_pose.orientation,
            vel.angular,
            dt_s
        )

        return Pose(
            position=predicted_pos,
            orientation=predicted_orient,
            timestamp_ns=target_time_ns
        )

    def submit_frame(self, render_pose: Pose, frame_id: int) -> None:
        with self._lock:
            self._render_pose = render_pose
            self._render_frame_id = frame_id

    def submit_motion_vectors(self, vectors: List[List[MotionVector]],
                               width: int, height: int) -> None:
        # ATW doesn't use motion vectors
        pass

    def reproject(self, display_time_ns: int) -> ReprojectedFrame:
        with self._lock:
            if not self._render_pose or not self._latest_pose:
                return ReprojectedFrame(success=False)

            # Predict pose at display time
            predicted_pose = self._predict_pose_internal(display_time_ns)

            # Calculate rotation delta from render pose to predicted pose
            rotation_delta = self._calculate_rotation_delta(
                self._render_pose.orientation,
                predicted_pose.orientation
            )

            # Clamp rotation correction
            rotation_delta = self._clamp_rotation(rotation_delta)

            self._metrics.frames_reprojected += 1
            self._metrics.atw_corrections += 1

            return ReprojectedFrame(
                success=True,
                mode_used=ReprojectionMode.ATW,
                predicted_pose=predicted_pose,
                rotation_delta=rotation_delta,
                translation_delta=(0.0, 0.0, 0.0),  # ATW doesn't handle translation
                timestamp_ns=display_time_ns
            )

    def get_late_latch_pose(self) -> Pose:
        with self._lock:
            if not self._latest_pose:
                return Pose()
            # Predict slightly into future for photon time
            photon_ns = int(time.time_ns() +
                          self._config.photon_time_offset_ms * 1_000_000)
            return self._predict_pose_internal(photon_ns)

    def get_metrics(self) -> ReprojectionMetrics:
        return self._metrics

    def _integrate_angular_velocity(
        self, orientation: Tuple[float, float, float, float],
        angular_vel: Tuple[float, float, float],
        dt: float
    ) -> Tuple[float, float, float, float]:
        """Integrate angular velocity to update orientation."""
        wx, wy, wz = angular_vel
        mag = math.sqrt(wx * wx + wy * wy + wz * wz)

        if mag < 1e-8:
            return orientation

        # Rotation axis and angle
        half_angle = mag * dt / 2.0
        sin_half = math.sin(half_angle)
        cos_half = math.cos(half_angle)

        # Delta quaternion
        dq = (
            sin_half * wx / mag,
            sin_half * wy / mag,
            sin_half * wz / mag,
            cos_half
        )

        # Multiply quaternions using shared utility
        return multiply_quaternions(orientation, dq)

    def _calculate_rotation_delta(
        self, from_orient: Tuple[float, float, float, float],
        to_orient: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Calculate rotation quaternion from one orientation to another."""
        # delta = to * inverse(from)
        inv_from = self._quaternion_inverse(from_orient)
        return multiply_quaternions(to_orient, inv_from)

    def _quaternion_inverse(
        self, q: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Calculate quaternion inverse (conjugate for unit quaternions)."""
        return (-q[0], -q[1], -q[2], q[3])

    def _clamp_rotation(
        self, q: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """Clamp rotation magnitude."""
        # Extract rotation angle
        w = min(1.0, max(-1.0, q[3]))
        angle = 2.0 * math.acos(abs(w))

        if angle > self._config.atw_rotation_limit:
            # Scale down rotation
            scale = self._config.atw_rotation_limit / angle
            half_angle = angle * scale / 2.0
            sin_half = math.sin(half_angle)
            cos_half = math.cos(half_angle)

            # Normalize axis
            mag = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2])
            if mag > 1e-8:
                return (
                    sin_half * q[0] / mag,
                    sin_half * q[1] / mag,
                    sin_half * q[2] / mag,
                    cos_half
                )

        return q


class ASWReprojection(XRReprojection):
    """Asynchronous Spacewarp implementation.

    Full frame generation using motion vectors.
    Handles both rotation and translation but more expensive.
    """

    def __init__(self, config: Optional[ReprojectionConfig] = None):
        """Initialize ASW reprojection."""
        self._config = config or ReprojectionConfig(mode=ReprojectionMode.ASW)
        self._atw = ATWReprojection(config)  # Fallback to ATW
        self._motion_vectors: Optional[List[List[MotionVector]]] = None
        self._motion_width = 0
        self._motion_height = 0
        # Cache for motion analysis to avoid per-frame calculations
        self._cached_avg_motion: float = 0.0
        self._cached_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._motion_dirty: bool = True
        self._metrics = ReprojectionMetrics()
        self._lock = threading.Lock()

    @property
    def config(self) -> ReprojectionConfig:
        return self._config

    def configure(self, config: ReprojectionConfig) -> None:
        with self._lock:
            self._config = config
            self._config.mode = ReprojectionMode.ASW
            self._atw.configure(config)

    def submit_pose(self, pose: Pose, velocity: Optional[PoseVelocity] = None) -> None:
        self._atw.submit_pose(pose, velocity)

    def predict_pose(self, target_time_ns: int) -> Pose:
        return self._atw.predict_pose(target_time_ns)

    def submit_frame(self, render_pose: Pose, frame_id: int) -> None:
        self._atw.submit_frame(render_pose, frame_id)

    def submit_motion_vectors(self, vectors: List[List[MotionVector]],
                               width: int, height: int) -> None:
        with self._lock:
            self._motion_vectors = vectors
            self._motion_width = width
            self._motion_height = height
            # Mark cache as dirty to recalculate on next access
            self._motion_dirty = True

    def reproject(self, display_time_ns: int) -> ReprojectedFrame:
        with self._lock:
            # Check if motion vectors available
            has_motion = (self._motion_vectors is not None and
                         self._motion_width > 0 and self._motion_height > 0)

            # Check if motion magnitude exceeds threshold
            if has_motion:
                avg_motion = self._calculate_average_motion()
                if avg_motion < self._config.asw_motion_threshold:
                    # Fall back to ATW for small motion
                    result = self._atw.reproject(display_time_ns)
                    return result

        # Perform ASW reprojection
        atw_result = self._atw.reproject(display_time_ns)

        if not atw_result.success:
            return atw_result

        # Calculate translation delta from motion vectors
        translation_delta = self._estimate_translation()

        self._metrics.frames_reprojected += 1
        self._metrics.asw_generations += 1

        return ReprojectedFrame(
            success=True,
            mode_used=ReprojectionMode.ASW,
            predicted_pose=atw_result.predicted_pose,
            rotation_delta=atw_result.rotation_delta,
            translation_delta=translation_delta,
            timestamp_ns=display_time_ns
        )

    def get_late_latch_pose(self) -> Pose:
        return self._atw.get_late_latch_pose()

    def get_metrics(self) -> ReprojectionMetrics:
        # Combine metrics
        atw_metrics = self._atw.get_metrics()
        return ReprojectionMetrics(
            frames_reprojected=self._metrics.frames_reprojected,
            atw_corrections=atw_metrics.atw_corrections,
            asw_generations=self._metrics.asw_generations,
            average_latency_ms=atw_metrics.average_latency_ms,
            prediction_error_deg=atw_metrics.prediction_error_deg,
            dropped_frames=self._metrics.dropped_frames
        )

    def _calculate_average_motion(self) -> float:
        """Calculate average motion magnitude from vectors."""
        if not self._motion_vectors:
            return 0.0

        # Return cached value if motion vectors haven't changed
        if not self._motion_dirty:
            return self._cached_avg_motion

        total = 0.0
        count = 0

        # Calculate both average motion and translation in single pass
        total_x = 0.0
        total_y = 0.0
        total_z = 0.0

        for row in self._motion_vectors:
            for mv in row:
                # Motion magnitude calculation
                total += math.sqrt(mv.dx * mv.dx + mv.dy * mv.dy)
                # Translation estimation (combined from _estimate_translation)
                depth_scale = max(0.1, mv.depth)
                total_x += mv.dx * depth_scale
                total_y += mv.dy * depth_scale
                total_z += (1.0 - mv.depth) * 0.1
                count += 1

        if count > 0:
            self._cached_avg_motion = total / count
            # Convert pixel motion to world units (approximate)
            scale = 0.001
            self._cached_translation = (
                total_x / count * scale,
                total_y / count * scale,
                total_z / count * scale
            )
        else:
            self._cached_avg_motion = 0.0
            self._cached_translation = (0.0, 0.0, 0.0)

        self._motion_dirty = False
        return self._cached_avg_motion

    def _estimate_translation(self) -> Tuple[float, float, float]:
        """Estimate world translation from motion vectors."""
        if not self._motion_vectors:
            return (0.0, 0.0, 0.0)

        # Ensure cache is populated by calling _calculate_average_motion
        # which now computes both values in a single pass
        if self._motion_dirty:
            self._calculate_average_motion()

        return self._cached_translation


class HybridReprojection(XRReprojection):
    """Hybrid ATW + ASW reprojection.

    Uses ATW for rotation and partial ASW for translation.
    Balance between quality and performance.
    """

    def __init__(self, config: Optional[ReprojectionConfig] = None):
        """Initialize hybrid reprojection."""
        self._config = config or ReprojectionConfig(mode=ReprojectionMode.HYBRID)
        self._atw = ATWReprojection(config)
        self._asw = ASWReprojection(config)
        self._metrics = ReprojectionMetrics()

    @property
    def config(self) -> ReprojectionConfig:
        return self._config

    def configure(self, config: ReprojectionConfig) -> None:
        self._config = config
        self._config.mode = ReprojectionMode.HYBRID
        self._atw.configure(config)
        self._asw.configure(config)

    def submit_pose(self, pose: Pose, velocity: Optional[PoseVelocity] = None) -> None:
        self._atw.submit_pose(pose, velocity)
        self._asw.submit_pose(pose, velocity)

    def predict_pose(self, target_time_ns: int) -> Pose:
        return self._atw.predict_pose(target_time_ns)

    def submit_frame(self, render_pose: Pose, frame_id: int) -> None:
        self._atw.submit_frame(render_pose, frame_id)
        self._asw.submit_frame(render_pose, frame_id)

    def submit_motion_vectors(self, vectors: List[List[MotionVector]],
                               width: int, height: int) -> None:
        self._asw.submit_motion_vectors(vectors, width, height)

    def reproject(self, display_time_ns: int) -> ReprojectedFrame:
        # Always do ATW
        atw_result = self._atw.reproject(display_time_ns)

        if not atw_result.success:
            return atw_result

        # Attempt partial ASW
        asw_result = self._asw.reproject(display_time_ns)

        # Blend results
        if asw_result.success and asw_result.mode_used == ReprojectionMode.ASW:
            # Use ATW rotation with ASW translation
            blend = self._config.asw_interpolation_factor
            translation = (
                asw_result.translation_delta[0] * blend,
                asw_result.translation_delta[1] * blend,
                asw_result.translation_delta[2] * blend
            )

            self._metrics.frames_reprojected += 1

            return ReprojectedFrame(
                success=True,
                mode_used=ReprojectionMode.HYBRID,
                predicted_pose=atw_result.predicted_pose,
                rotation_delta=atw_result.rotation_delta,
                translation_delta=translation,
                timestamp_ns=display_time_ns
            )

        return atw_result

    def get_late_latch_pose(self) -> Pose:
        return self._atw.get_late_latch_pose()

    def get_metrics(self) -> ReprojectionMetrics:
        atw_metrics = self._atw.get_metrics()
        asw_metrics = self._asw.get_metrics()
        return ReprojectionMetrics(
            frames_reprojected=self._metrics.frames_reprojected,
            atw_corrections=atw_metrics.atw_corrections,
            asw_generations=asw_metrics.asw_generations,
            average_latency_ms=atw_metrics.average_latency_ms,
            prediction_error_deg=atw_metrics.prediction_error_deg,
            dropped_frames=self._metrics.dropped_frames
        )


def create_reprojection(config: Optional[ReprojectionConfig] = None) -> XRReprojection:
    """Factory function to create appropriate reprojection handler.

    Args:
        config: Reprojection configuration

    Returns:
        Configured reprojection instance
    """
    if config is None:
        config = ReprojectionConfig()

    if not config.enabled or config.mode == ReprojectionMode.NONE:
        return ATWReprojection(ReprojectionConfig(enabled=False))

    if config.mode == ReprojectionMode.ATW:
        return ATWReprojection(config)
    elif config.mode == ReprojectionMode.ASW:
        return ASWReprojection(config)
    elif config.mode == ReprojectionMode.HYBRID:
        return HybridReprojection(config)
    else:
        return ATWReprojection(config)
