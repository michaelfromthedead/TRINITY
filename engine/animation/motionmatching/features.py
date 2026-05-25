"""
Motion Matching Features - Feature extraction for motion matching.

This module provides feature extraction for motion matching:
- FeatureSet: Container for feature values and weights
- FeatureExtractor: Extracts features from poses and trajectories
- Standard features: bone positions/velocities, trajectory, foot contacts
- Feature normalization and weighting

Usage:
    from engine.animation.motionmatching.features import (
        FeatureSet, FeatureExtractor, FeatureConfig
    )

    # Configure extractor
    config = FeatureConfig()
    extractor = FeatureExtractor(config)

    # Extract features from a clip frame
    features = extractor.extract(clip, frame)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)
import numpy as np


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================


class FeatureType(Enum):
    """Types of features that can be extracted."""
    BONE_POSITION = auto()       # 3D position relative to root
    BONE_VELOCITY = auto()       # 3D velocity
    BONE_ROTATION = auto()       # Quaternion or euler
    TRAJECTORY_POSITION = auto() # Future root position
    TRAJECTORY_FACING = auto()   # Future facing direction
    TRAJECTORY_VELOCITY = auto() # Future velocity
    FOOT_CONTACT = auto()        # Foot contact state (0/1)
    FOOT_POSITION = auto()       # Foot position
    FOOT_VELOCITY = auto()       # Foot velocity
    ROOT_VELOCITY = auto()       # Root motion velocity
    ROOT_ANGULAR_VEL = auto()    # Root angular velocity


# Import centralized config
from engine.animation.motionmatching.config import (
    DEFAULT_FEATURE_WEIGHTS,
    DEFAULT_TRAJECTORY_TIMES as CONFIG_TRAJECTORY_TIMES,
    DEFAULT_IDLE_DETECTION,
)

# Default trajectory time points (seconds ahead)
DEFAULT_TRAJECTORY_TIMES = CONFIG_TRAJECTORY_TIMES.copy()

# Standard bone names for feature extraction
STANDARD_BONES = [
    'hips', 'spine', 'chest', 'neck', 'head',
    'left_shoulder', 'left_arm', 'left_forearm', 'left_hand',
    'right_shoulder', 'right_arm', 'right_forearm', 'right_hand',
    'left_upleg', 'left_leg', 'left_foot', 'left_toe',
    'right_upleg', 'right_leg', 'right_foot', 'right_toe',
]

# Key bones typically used for motion matching (reduced set)
KEY_BONES = ['hips', 'left_foot', 'right_foot', 'left_hand', 'right_hand']


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FeatureSet:
    """Container for extracted feature values and their weights.

    Attributes:
        values: Flattened feature vector (numpy array)
        weights: Per-feature importance weights
        labels: Optional labels for each feature dimension
        feature_ranges: Mapping of feature name to (start, end) indices
    """
    values: np.ndarray
    weights: Optional[np.ndarray] = None
    labels: Optional[List[str]] = None
    feature_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values, dtype=np.float32)

        if self.weights is None:
            self.weights = np.ones(len(self.values), dtype=np.float32)
        elif not isinstance(self.weights, np.ndarray):
            self.weights = np.array(self.weights, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Feature vector dimension."""
        return len(self.values)

    def get_feature(self, name: str) -> Optional[np.ndarray]:
        """Get feature values by name."""
        if name in self.feature_ranges:
            start, end = self.feature_ranges[name]
            return self.values[start:end]
        return None

    def get_feature_weight(self, name: str) -> Optional[np.ndarray]:
        """Get feature weights by name."""
        if name in self.feature_ranges:
            start, end = self.feature_ranges[name]
            return self.weights[start:end]
        return None

    def weighted_values(self) -> np.ndarray:
        """Get values multiplied by weights."""
        return self.values * self.weights

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction.

    Attributes:
        use_bone_positions: Extract bone position features
        use_bone_velocities: Extract bone velocity features
        use_trajectory: Extract future trajectory features
        use_foot_contacts: Extract foot contact states
        bone_names: List of bones to extract features from
        trajectory_times: Time points for trajectory prediction (seconds)
        position_weight: Weight for position features
        velocity_weight: Weight for velocity features
        trajectory_weight: Weight for trajectory features
        contact_weight: Weight for contact features
        local_space: Use local (root-relative) coordinates
    """
    use_bone_positions: bool = True
    use_bone_velocities: bool = True
    use_trajectory: bool = True
    use_foot_contacts: bool = True

    bone_names: List[str] = field(default_factory=lambda: KEY_BONES.copy())
    trajectory_times: List[float] = field(
        default_factory=lambda: DEFAULT_TRAJECTORY_TIMES.copy()
    )

    position_weight: float = DEFAULT_FEATURE_WEIGHTS.position_weight
    velocity_weight: float = DEFAULT_FEATURE_WEIGHTS.velocity_weight
    trajectory_weight: float = DEFAULT_FEATURE_WEIGHTS.trajectory_weight
    contact_weight: float = DEFAULT_FEATURE_WEIGHTS.contact_weight

    local_space: bool = True
    include_root: bool = True


@dataclass
class BoneData:
    """Data for a single bone at a frame.

    Attributes:
        position: World or local position (x, y, z)
        velocity: Velocity (x, y, z)
        rotation: Rotation as quaternion (x, y, z, w)
    """
    position: np.ndarray
    velocity: np.ndarray
    rotation: Optional[np.ndarray] = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        if self.rotation is not None:
            self.rotation = np.asarray(self.rotation, dtype=np.float32)


@dataclass
class TrajectoryPoint:
    """A point on the predicted trajectory.

    Attributes:
        time_offset: Time offset from current frame (seconds)
        position: Position (x, y, z) or (x, z) in local space
        facing: Facing direction as angle or vector
        velocity: Velocity at this point
    """
    time_offset: float
    position: np.ndarray
    facing: Union[float, np.ndarray]
    velocity: Optional[np.ndarray] = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        if isinstance(self.facing, (int, float)):
            self.facing = float(self.facing)
        else:
            self.facing = np.asarray(self.facing, dtype=np.float32)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float32)


@dataclass
class FootContact:
    """Foot contact state.

    Attributes:
        left_contact: Left foot contact (0.0-1.0)
        right_contact: Right foot contact (0.0-1.0)
    """
    left_contact: float = 0.0
    right_contact: float = 0.0


# =============================================================================
# PROTOCOLS
# =============================================================================


class PoseProvider(Protocol):
    """Protocol for objects that can provide pose data."""

    def get_bone_position(self, bone_name: str) -> np.ndarray: ...
    def get_bone_velocity(self, bone_name: str) -> np.ndarray: ...
    def get_bone_rotation(self, bone_name: str) -> np.ndarray: ...
    def get_root_transform(self) -> Tuple[np.ndarray, np.ndarray]: ...


class TrajectoryProvider(Protocol):
    """Protocol for objects that can provide trajectory data."""

    def get_future_position(self, time_offset: float) -> np.ndarray: ...
    def get_future_facing(self, time_offset: float) -> float: ...
    def get_future_velocity(self, time_offset: float) -> np.ndarray: ...


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================


class FeatureExtractor:
    """Extracts features from animation poses and trajectories.

    The extractor combines multiple feature types into a single
    flattened vector suitable for motion matching search.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize feature extractor.

        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config or FeatureConfig()
        self._feature_dimension: Optional[int] = None
        self._feature_layout: List[Tuple[str, int, int]] = []

    @property
    def feature_dimension(self) -> int:
        """Total dimension of extracted feature vectors."""
        if self._feature_dimension is None:
            self._compute_feature_layout()
        return self._feature_dimension

    def _compute_feature_layout(self) -> None:
        """Compute the layout of feature dimensions."""
        self._feature_layout = []
        offset = 0

        # Bone positions (3 per bone)
        if self.config.use_bone_positions:
            for bone in self.config.bone_names:
                name = f"pos_{bone}"
                self._feature_layout.append((name, offset, offset + 3))
                offset += 3

        # Bone velocities (3 per bone)
        if self.config.use_bone_velocities:
            for bone in self.config.bone_names:
                name = f"vel_{bone}"
                self._feature_layout.append((name, offset, offset + 3))
                offset += 3

        # Trajectory (3 pos + 2 facing per point)
        if self.config.use_trajectory:
            for t in self.config.trajectory_times:
                name = f"traj_pos_{t:.2f}"
                self._feature_layout.append((name, offset, offset + 3))
                offset += 3

                name = f"traj_face_{t:.2f}"
                self._feature_layout.append((name, offset, offset + 2))
                offset += 2

        # Foot contacts (2 values)
        if self.config.use_foot_contacts:
            self._feature_layout.append(("foot_contacts", offset, offset + 2))
            offset += 2

        self._feature_dimension = offset

    def extract(
        self,
        clip: Any,
        frame: int,
        trajectory_override: Optional[List[TrajectoryPoint]] = None,
    ) -> FeatureSet:
        """Extract features from an animation clip at a specific frame.

        Args:
            clip: Animation clip with pose data
            frame: Frame number to extract from
            trajectory_override: Optional override for trajectory prediction

        Returns:
            FeatureSet containing extracted features
        """
        if self._feature_dimension is None:
            self._compute_feature_layout()

        values = np.zeros(self._feature_dimension, dtype=np.float32)
        weights = np.ones(self._feature_dimension, dtype=np.float32)
        feature_ranges: Dict[str, Tuple[int, int]] = {}

        # Get root transform for local space conversion
        root_pos, root_rot = self._get_root_transform(clip, frame)

        offset = 0

        # Extract bone positions
        if self.config.use_bone_positions:
            for bone in self.config.bone_names:
                pos = self._get_bone_position(clip, frame, bone)
                if self.config.local_space:
                    pos = self._to_local_space(pos, root_pos, root_rot)

                values[offset:offset+3] = pos
                weights[offset:offset+3] = self.config.position_weight
                feature_ranges[f"pos_{bone}"] = (offset, offset + 3)
                offset += 3

        # Extract bone velocities
        if self.config.use_bone_velocities:
            for bone in self.config.bone_names:
                vel = self._get_bone_velocity(clip, frame, bone)
                if self.config.local_space:
                    vel = self._rotate_to_local(vel, root_rot)

                values[offset:offset+3] = vel
                weights[offset:offset+3] = self.config.velocity_weight
                feature_ranges[f"vel_{bone}"] = (offset, offset + 3)
                offset += 3

        # Extract trajectory
        if self.config.use_trajectory:
            trajectory = trajectory_override or self._predict_trajectory(clip, frame)

            for i, t in enumerate(self.config.trajectory_times):
                if i < len(trajectory):
                    point = trajectory[i]
                else:
                    # Use last known point
                    point = trajectory[-1] if trajectory else TrajectoryPoint(
                        time_offset=t,
                        position=np.zeros(3),
                        facing=0.0,
                    )

                # Position (3D)
                traj_pos = point.position
                if self.config.local_space and len(traj_pos) >= 3:
                    traj_pos = self._to_local_space(traj_pos, root_pos, root_rot)

                values[offset:offset+3] = traj_pos[:3] if len(traj_pos) >= 3 else np.pad(traj_pos, (0, 3-len(traj_pos)))
                weights[offset:offset+3] = self.config.trajectory_weight
                feature_ranges[f"traj_pos_{t:.2f}"] = (offset, offset + 3)
                offset += 3

                # Facing (2D direction vector)
                if isinstance(point.facing, (int, float)):
                    facing_vec = np.array([
                        np.cos(point.facing),
                        np.sin(point.facing),
                    ], dtype=np.float32)
                elif isinstance(point.facing, np.ndarray) and point.facing.ndim == 0:
                    # Scalar numpy array
                    facing_angle = float(point.facing)
                    facing_vec = np.array([
                        np.cos(facing_angle),
                        np.sin(facing_angle),
                    ], dtype=np.float32)
                else:
                    facing_arr = np.asarray(point.facing)
                    if facing_arr.ndim == 0:
                        facing_angle = float(facing_arr)
                        facing_vec = np.array([np.cos(facing_angle), np.sin(facing_angle)], dtype=np.float32)
                    else:
                        facing_vec = facing_arr[:2] if len(facing_arr) >= 2 else np.pad(facing_arr, (0, 2-len(facing_arr)))

                values[offset:offset+2] = facing_vec
                weights[offset:offset+2] = self.config.trajectory_weight
                feature_ranges[f"traj_face_{t:.2f}"] = (offset, offset + 2)
                offset += 2

        # Extract foot contacts
        if self.config.use_foot_contacts:
            contacts = self._get_foot_contacts(clip, frame)
            values[offset] = contacts.left_contact
            values[offset+1] = contacts.right_contact
            weights[offset:offset+2] = self.config.contact_weight
            feature_ranges["foot_contacts"] = (offset, offset + 2)
            offset += 2

        return FeatureSet(
            values=values,
            weights=weights,
            feature_ranges=feature_ranges,
        )

    def extract_from_pose(
        self,
        bone_data: Dict[str, BoneData],
        trajectory: List[TrajectoryPoint],
        foot_contacts: FootContact,
        root_transform: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> FeatureSet:
        """Extract features from explicit pose data.

        Args:
            bone_data: Dictionary mapping bone names to BoneData
            trajectory: Future trajectory points
            foot_contacts: Foot contact states
            root_transform: Optional (position, rotation) tuple

        Returns:
            FeatureSet containing extracted features
        """
        if self._feature_dimension is None:
            self._compute_feature_layout()

        values = np.zeros(self._feature_dimension, dtype=np.float32)
        weights = np.ones(self._feature_dimension, dtype=np.float32)
        feature_ranges: Dict[str, Tuple[int, int]] = {}

        # Get root transform
        if root_transform:
            root_pos, root_rot = root_transform
        else:
            root_pos = np.zeros(3, dtype=np.float32)
            root_rot = np.array([0, 0, 0, 1], dtype=np.float32)  # Identity quaternion

        offset = 0

        # Extract bone positions
        if self.config.use_bone_positions:
            for bone in self.config.bone_names:
                if bone in bone_data:
                    pos = bone_data[bone].position.copy()
                else:
                    pos = np.zeros(3, dtype=np.float32)

                if self.config.local_space:
                    pos = self._to_local_space(pos, root_pos, root_rot)

                values[offset:offset+3] = pos
                weights[offset:offset+3] = self.config.position_weight
                feature_ranges[f"pos_{bone}"] = (offset, offset + 3)
                offset += 3

        # Extract bone velocities
        if self.config.use_bone_velocities:
            for bone in self.config.bone_names:
                if bone in bone_data:
                    vel = bone_data[bone].velocity.copy()
                else:
                    vel = np.zeros(3, dtype=np.float32)

                if self.config.local_space:
                    vel = self._rotate_to_local(vel, root_rot)

                values[offset:offset+3] = vel
                weights[offset:offset+3] = self.config.velocity_weight
                feature_ranges[f"vel_{bone}"] = (offset, offset + 3)
                offset += 3

        # Extract trajectory
        if self.config.use_trajectory:
            for i, t in enumerate(self.config.trajectory_times):
                if i < len(trajectory):
                    point = trajectory[i]
                else:
                    point = trajectory[-1] if trajectory else TrajectoryPoint(
                        time_offset=t,
                        position=np.zeros(3),
                        facing=0.0,
                    )

                traj_pos = point.position
                if self.config.local_space and len(traj_pos) >= 3:
                    traj_pos = self._to_local_space(traj_pos, root_pos, root_rot)

                values[offset:offset+3] = traj_pos[:3] if len(traj_pos) >= 3 else np.pad(traj_pos, (0, 3-len(traj_pos)))
                weights[offset:offset+3] = self.config.trajectory_weight
                feature_ranges[f"traj_pos_{t:.2f}"] = (offset, offset + 3)
                offset += 3

                if isinstance(point.facing, (int, float)):
                    facing_vec = np.array([
                        np.cos(point.facing),
                        np.sin(point.facing),
                    ], dtype=np.float32)
                elif isinstance(point.facing, np.ndarray) and point.facing.ndim == 0:
                    facing_angle = float(point.facing)
                    facing_vec = np.array([np.cos(facing_angle), np.sin(facing_angle)], dtype=np.float32)
                else:
                    facing_arr = np.asarray(point.facing)
                    if facing_arr.ndim == 0:
                        facing_angle = float(facing_arr)
                        facing_vec = np.array([np.cos(facing_angle), np.sin(facing_angle)], dtype=np.float32)
                    else:
                        facing_vec = facing_arr[:2] if len(facing_arr) >= 2 else np.pad(facing_arr, (0, 2-len(facing_arr)))

                values[offset:offset+2] = facing_vec
                weights[offset:offset+2] = self.config.trajectory_weight
                feature_ranges[f"traj_face_{t:.2f}"] = (offset, offset + 2)
                offset += 2

        # Extract foot contacts
        if self.config.use_foot_contacts:
            values[offset] = foot_contacts.left_contact
            values[offset+1] = foot_contacts.right_contact
            weights[offset:offset+2] = self.config.contact_weight
            feature_ranges["foot_contacts"] = (offset, offset + 2)
            offset += 2

        return FeatureSet(
            values=values,
            weights=weights,
            feature_ranges=feature_ranges,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_root_transform(
        self, clip: Any, frame: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get root bone transform.

        Args:
            clip: Animation clip
            frame: Frame number

        Returns:
            Tuple of (position, rotation quaternion)
        """
        if hasattr(clip, 'get_root_transform'):
            return clip.get_root_transform(frame)

        # Try to get from pose
        if hasattr(clip, 'get_frame_pose'):
            pose = clip.get_frame_pose(frame)
            if hasattr(pose, 'root_position') and hasattr(pose, 'root_rotation'):
                return pose.root_position, pose.root_rotation

        # Default: origin with identity rotation
        return np.zeros(3, dtype=np.float32), np.array([0, 0, 0, 1], dtype=np.float32)

    def _get_bone_position(
        self, clip: Any, frame: int, bone_name: str
    ) -> np.ndarray:
        """Get bone position at frame.

        Args:
            clip: Animation clip
            frame: Frame number
            bone_name: Bone name

        Returns:
            3D position
        """
        if hasattr(clip, 'get_bone_position'):
            pos = clip.get_bone_position(frame, bone_name)
            return np.asarray(pos, dtype=np.float32)

        if hasattr(clip, 'get_frame_pose'):
            pose = clip.get_frame_pose(frame)
            if hasattr(pose, 'get_bone_position'):
                pos = pose.get_bone_position(bone_name)
                return np.asarray(pos, dtype=np.float32)

        return np.zeros(3, dtype=np.float32)

    def _get_bone_velocity(
        self, clip: Any, frame: int, bone_name: str
    ) -> np.ndarray:
        """Get bone velocity at frame.

        Args:
            clip: Animation clip
            frame: Frame number
            bone_name: Bone name

        Returns:
            3D velocity
        """
        if hasattr(clip, 'get_bone_velocity'):
            vel = clip.get_bone_velocity(frame, bone_name)
            return np.asarray(vel, dtype=np.float32)

        # Compute from positions if not available
        if frame > 0:
            pos_prev = self._get_bone_position(clip, frame - 1, bone_name)
            pos_curr = self._get_bone_position(clip, frame, bone_name)
            dt = 1.0 / getattr(clip, 'frame_rate', 30.0)
            return (pos_curr - pos_prev) / dt

        return np.zeros(3, dtype=np.float32)

    def _get_foot_contacts(self, clip: Any, frame: int) -> FootContact:
        """Get foot contact states at frame.

        Args:
            clip: Animation clip
            frame: Frame number

        Returns:
            FootContact state
        """
        if hasattr(clip, 'get_foot_contacts'):
            contacts = clip.get_foot_contacts(frame)
            return FootContact(
                left_contact=contacts[0] if len(contacts) > 0 else 0.0,
                right_contact=contacts[1] if len(contacts) > 1 else 0.0,
            )

        # Try to detect from foot positions/velocities
        left_vel = self._get_bone_velocity(clip, frame, 'left_foot')
        right_vel = self._get_bone_velocity(clip, frame, 'right_foot')

        # Simple heuristic: low velocity = contact
        vel_threshold = DEFAULT_IDLE_DETECTION.velocity_threshold
        return FootContact(
            left_contact=1.0 if np.linalg.norm(left_vel) < vel_threshold else 0.0,
            right_contact=1.0 if np.linalg.norm(right_vel) < vel_threshold else 0.0,
        )

    def _predict_trajectory(
        self, clip: Any, frame: int
    ) -> List[TrajectoryPoint]:
        """Predict future trajectory from clip.

        Args:
            clip: Animation clip
            frame: Current frame number

        Returns:
            List of trajectory points
        """
        trajectory = []
        frame_rate = getattr(clip, 'frame_rate', 30.0)
        frame_count = getattr(clip, 'frame_count', 0)

        for t in self.config.trajectory_times:
            future_frame = frame + int(t * frame_rate)

            if future_frame >= frame_count:
                # Extrapolate if past clip end
                if getattr(clip, 'is_looping', False):
                    future_frame = future_frame % frame_count
                else:
                    future_frame = frame_count - 1

            # Get position
            root_pos, root_rot = self._get_root_transform(clip, future_frame)

            # Get facing (from rotation quaternion)
            facing = self._quaternion_to_facing(root_rot)

            # Get velocity
            if hasattr(clip, 'get_root_velocity'):
                velocity = clip.get_root_velocity(future_frame)
            else:
                velocity = np.zeros(3, dtype=np.float32)

            trajectory.append(TrajectoryPoint(
                time_offset=t,
                position=root_pos,
                facing=facing,
                velocity=velocity,
            ))

        return trajectory

    def _to_local_space(
        self,
        position: np.ndarray,
        root_position: np.ndarray,
        root_rotation: np.ndarray,
    ) -> np.ndarray:
        """Convert world position to root-local space.

        Args:
            position: World position
            root_position: Root bone world position
            root_rotation: Root bone rotation quaternion

        Returns:
            Local space position
        """
        # Translate to root origin
        local = position - root_position

        # Rotate by inverse of root rotation
        inv_rot = self._quaternion_inverse(root_rotation)
        local = self._rotate_vector(local, inv_rot)

        return local

    def _rotate_to_local(
        self, vector: np.ndarray, root_rotation: np.ndarray
    ) -> np.ndarray:
        """Rotate vector to local space (no translation).

        Args:
            vector: World space vector
            root_rotation: Root rotation quaternion

        Returns:
            Local space vector
        """
        inv_rot = self._quaternion_inverse(root_rotation)
        return self._rotate_vector(vector, inv_rot)

    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse (conjugate for unit quaternions)."""
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

    def _rotate_vector(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate vector by quaternion.

        Args:
            v: 3D vector
            q: Rotation quaternion (x, y, z, w)

        Returns:
            Rotated vector
        """
        # Extract quaternion components
        qx, qy, qz, qw = q

        # Quaternion-vector rotation: v' = q * v * q^-1
        # Optimized formula
        t = 2.0 * np.cross(q[:3], v)
        return v + qw * t + np.cross(q[:3], t)

    def _quaternion_to_facing(self, q: np.ndarray) -> float:
        """Extract facing angle from quaternion (rotation around Y axis).

        Args:
            q: Rotation quaternion (x, y, z, w)

        Returns:
            Facing angle in radians
        """
        # Extract yaw from quaternion
        qx, qy, qz, qw = q
        siny_cosp = 2.0 * (qw * qy + qz * qx)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp)


# =============================================================================
# NORMALIZATION
# =============================================================================


class FeatureNormalizer:
    """Normalizes features using statistics from a dataset.

    Supports z-score normalization (mean/std) and min-max scaling.
    """

    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        min_val: Optional[np.ndarray] = None,
        max_val: Optional[np.ndarray] = None,
    ):
        """Initialize normalizer.

        Args:
            mean: Feature means
            std: Feature standard deviations
            min_val: Feature minimums
            max_val: Feature maximums
        """
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self._fitted = mean is not None

    def fit(
        self,
        features: np.ndarray,
        epsilon: float = 1e-8,
    ) -> FeatureNormalizer:
        """Compute normalization statistics from features.

        Args:
            features: Matrix of shape (num_samples, feature_dim)
            epsilon: Small value to prevent division by zero

        Returns:
            self for chaining
        """
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        self.std = np.maximum(self.std, epsilon)
        self.min_val = np.min(features, axis=0)
        self.max_val = np.max(features, axis=0)
        self._fitted = True
        return self

    def normalize(
        self,
        features: np.ndarray,
        method: str = 'zscore',
    ) -> np.ndarray:
        """Normalize features.

        Args:
            features: Features to normalize
            method: 'zscore' or 'minmax'

        Returns:
            Normalized features
        """
        if not self._fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if method == 'zscore':
            return (features - self.mean) / self.std
        elif method == 'minmax':
            range_val = self.max_val - self.min_val
            range_val = np.maximum(range_val, 1e-8)
            return (features - self.min_val) / range_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def denormalize(
        self,
        features: np.ndarray,
        method: str = 'zscore',
    ) -> np.ndarray:
        """Denormalize features back to original scale.

        Args:
            features: Normalized features
            method: 'zscore' or 'minmax'

        Returns:
            Original scale features
        """
        if not self._fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        if method == 'zscore':
            return features * self.std + self.mean
        elif method == 'minmax':
            range_val = self.max_val - self.min_val
            return features * range_val + self.min_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# FEATURE WEIGHTING
# =============================================================================


@dataclass
class FeatureWeights:
    """Configurable weights for different feature categories."""

    pose_weight: float = 1.0
    velocity_weight: float = 0.5
    trajectory_weight: float = 1.0
    contact_weight: float = 2.0

    # Per-bone weights (optional override)
    bone_weights: Dict[str, float] = field(default_factory=dict)

    # Per-trajectory-point weights (optional override)
    trajectory_point_weights: Dict[float, float] = field(default_factory=dict)

    def get_bone_weight(self, bone_name: str, default: float = 1.0) -> float:
        """Get weight for a specific bone."""
        return self.bone_weights.get(bone_name, default)

    def get_trajectory_weight(
        self, time_offset: float, default: float = 1.0
    ) -> float:
        """Get weight for a specific trajectory point."""
        return self.trajectory_point_weights.get(time_offset, default)

    def apply_to_feature_set(self, feature_set: FeatureSet) -> FeatureSet:
        """Apply weights to a feature set.

        Args:
            feature_set: Feature set to weight

        Returns:
            New feature set with applied weights
        """
        weights = feature_set.weights.copy()

        for name, (start, end) in feature_set.feature_ranges.items():
            if name.startswith('pos_'):
                bone = name[4:]
                weights[start:end] *= self.pose_weight * self.get_bone_weight(bone)
            elif name.startswith('vel_'):
                bone = name[4:]
                weights[start:end] *= self.velocity_weight * self.get_bone_weight(bone)
            elif name.startswith('traj_'):
                # Extract time from name like "traj_pos_0.20"
                parts = name.split('_')
                if len(parts) >= 3:
                    try:
                        t = float(parts[2])
                        weights[start:end] *= self.trajectory_weight * self.get_trajectory_weight(t)
                    except ValueError:
                        weights[start:end] *= self.trajectory_weight
            elif name == 'foot_contacts':
                weights[start:end] *= self.contact_weight

        return FeatureSet(
            values=feature_set.values,
            weights=weights,
            labels=feature_set.labels,
            feature_ranges=feature_set.feature_ranges,
        )
