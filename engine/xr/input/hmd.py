"""
Head-Mounted Display (HMD) tracking component.

Provides 6-DOF pose tracking with prediction for VR/AR headsets.
Integrates with the Trinity Pattern descriptors for change detection
and thread-safe updates from tracking subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from trinity.descriptors import (
    TrackedDescriptor,
    AtomicDescriptor,
    ObservableDescriptor,
    RangeDescriptor,
    TransientDescriptor,
    ImmutableDescriptor,
    clear_dirty,
)

if TYPE_CHECKING:
    from engine.core.math.quat import Quat
    from engine.core.math.mat import Mat4


# Type aliases
Vec3Tuple = Tuple[float, float, float]
QuatTuple = Tuple[float, float, float, float]


class HMDTrackingState(Enum):
    """HMD tracking state machine states."""
    INITIALIZING = auto()  # System starting up
    TRACKING = auto()      # Full 6-DOF tracking active
    LIMITED = auto()       # Partial tracking (e.g., orientation only)
    LOST = auto()          # Tracking lost
    DISABLED = auto()      # Tracking disabled by user/system


# Valid state transitions
HMD_STATE_TRANSITIONS: Dict[HMDTrackingState, set[HMDTrackingState]] = {
    HMDTrackingState.INITIALIZING: {
        HMDTrackingState.TRACKING,
        HMDTrackingState.LOST,
        HMDTrackingState.DISABLED,
    },
    HMDTrackingState.TRACKING: {
        HMDTrackingState.LIMITED,
        HMDTrackingState.LOST,
        HMDTrackingState.DISABLED,
    },
    HMDTrackingState.LIMITED: {
        HMDTrackingState.TRACKING,
        HMDTrackingState.LOST,
        HMDTrackingState.DISABLED,
    },
    HMDTrackingState.LOST: {
        HMDTrackingState.TRACKING,
        HMDTrackingState.LIMITED,
        HMDTrackingState.DISABLED,
    },
    HMDTrackingState.DISABLED: {
        HMDTrackingState.INITIALIZING,
    },
}


@dataclass(slots=True)
class HMDDisplayInfo:
    """Display specifications for an HMD."""
    resolution_per_eye: Tuple[int, int] = (1920, 1080)
    refresh_rate: float = 90.0
    field_of_view: Tuple[float, float] = (90.0, 90.0)  # Horizontal, vertical in degrees
    ipd: float = 0.063  # Inter-pupillary distance in meters


@dataclass(slots=True)
class PredictionConfig:
    """Configuration for pose prediction."""
    enabled: bool = True
    prediction_time_ms: float = 11.1  # ~1 frame at 90Hz
    max_prediction_time_ms: float = 33.3  # ~3 frames max
    velocity_smoothing: float = 0.8  # EMA smoothing factor


class HeadMountedDisplay:
    """
    Head-mounted display tracking component.

    Provides 6-DOF pose tracking with prediction for low-latency rendering.
    Uses atomic descriptors for thread-safe updates from tracking threads.

    Features:
    - Position and orientation tracking
    - Velocity tracking for prediction
    - Pose prediction for ATW/ASW
    - Tracking state machine
    - View matrix computation

    Attributes:
        position: 3D position in stage space (meters)
        orientation: Rotation as quaternion (x, y, z, w)
        linear_velocity: Velocity in m/s
        angular_velocity: Angular velocity in rad/s
        tracking_state: Current tracking state
        confidence: Tracking confidence (0-1)
    """

    # Tracked + Atomic descriptors for pose data
    position = TrackedDescriptor(
        field_type=tuple,
        use_bitmask=True,
        field_offset=0,
    )
    orientation = TrackedDescriptor(
        field_type=tuple,
        use_bitmask=True,
        field_offset=1,
    )
    linear_velocity = TrackedDescriptor(
        field_type=tuple,
        use_bitmask=True,
        field_offset=2,
    )
    angular_velocity = TrackedDescriptor(
        field_type=tuple,
        use_bitmask=True,
        field_offset=3,
    )

    # Observable for tracking state (UI can react)
    tracking_state = TrackedDescriptor(
        field_type=HMDTrackingState,
        use_bitmask=True,
        field_offset=4,
    )

    # Range-clamped confidence
    confidence = TrackedDescriptor(
        field_type=float,
        use_bitmask=True,
        field_offset=5,
    )

    __slots__ = (
        "__dict__",
        "__weakref__",
        "_device_id",
        "_display_info",
        "_prediction_config",
        "_last_update_time",
        "_predicted_position",
        "_predicted_orientation",
        "_view_matrix_left",
        "_view_matrix_right",
        "_on_tracking_state_changed",
        "_on_tracking_lost",
        "_on_tracking_restored",
        "_entity_id",
    )

    def __init__(
        self,
        device_id: str = "",
        display_info: Optional[HMDDisplayInfo] = None,
        prediction_config: Optional[PredictionConfig] = None,
        entity_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the HMD component.

        Args:
            device_id: Unique device identifier
            display_info: Display specifications
            prediction_config: Prediction settings
            entity_id: Optional entity ID for ECS
        """
        self._device_id = device_id
        self._display_info = display_info or HMDDisplayInfo()
        self._prediction_config = prediction_config or PredictionConfig()
        self._last_update_time: float = 0.0
        self._entity_id = entity_id

        # Predicted pose (computed)
        self._predicted_position: Vec3Tuple = (0.0, 0.0, 0.0)
        self._predicted_orientation: QuatTuple = (0.0, 0.0, 0.0, 1.0)

        # Cached view matrices (transient)
        self._view_matrix_left: Optional[Any] = None
        self._view_matrix_right: Optional[Any] = None

        # Callbacks
        self._on_tracking_state_changed: List[Callable[[HMDTrackingState, HMDTrackingState], None]] = []
        self._on_tracking_lost: List[Callable[[HeadMountedDisplay], None]] = []
        self._on_tracking_restored: List[Callable[[HeadMountedDisplay], None]] = []

        # Initialize tracked fields
        self.position = (0.0, 0.0, 0.0)
        self.orientation = (0.0, 0.0, 0.0, 1.0)
        self.linear_velocity = (0.0, 0.0, 0.0)
        self.angular_velocity = (0.0, 0.0, 0.0)
        self.tracking_state = HMDTrackingState.INITIALIZING
        self.confidence = 0.0

        clear_dirty(self)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def device_id(self) -> str:
        """Get the device identifier."""
        return self._device_id

    @property
    def display_info(self) -> HMDDisplayInfo:
        """Get display specifications."""
        return self._display_info

    @property
    def prediction_config(self) -> PredictionConfig:
        """Get prediction configuration."""
        return self._prediction_config

    @property
    def is_tracking(self) -> bool:
        """Check if HMD is actively tracking."""
        return self.tracking_state in (HMDTrackingState.TRACKING, HMDTrackingState.LIMITED)

    @property
    def predicted_position(self) -> Vec3Tuple:
        """Get predicted position for next frame."""
        return self._predicted_position

    @property
    def predicted_orientation(self) -> QuatTuple:
        """Get predicted orientation for next frame."""
        return self._predicted_orientation

    # =========================================================================
    # POSE UPDATE
    # =========================================================================

    def update_pose(
        self,
        position: Vec3Tuple,
        orientation: QuatTuple,
        linear_velocity: Optional[Vec3Tuple] = None,
        angular_velocity: Optional[Vec3Tuple] = None,
        confidence: float = 1.0,
        timestamp: float = 0.0,
    ) -> None:
        """
        Update HMD pose from tracking system.

        Args:
            position: New position (x, y, z) in meters
            orientation: New orientation quaternion (x, y, z, w)
            linear_velocity: Optional velocity override
            angular_velocity: Optional angular velocity override
            confidence: Tracking confidence (0-1)
            timestamp: Update timestamp
        """
        old_pos = self.position
        old_orient = self.orientation

        # Update position and orientation
        self.position = position
        self.orientation = orientation

        # Calculate velocity if not provided
        if linear_velocity is not None:
            self.linear_velocity = linear_velocity
        elif timestamp > self._last_update_time:
            dt = timestamp - self._last_update_time
            self.linear_velocity = (
                (position[0] - old_pos[0]) / dt,
                (position[1] - old_pos[1]) / dt,
                (position[2] - old_pos[2]) / dt,
            )

        if angular_velocity is not None:
            self.angular_velocity = angular_velocity

        # Update confidence and tracking state
        self.confidence = max(0.0, min(1.0, confidence))
        self._update_tracking_state(confidence)

        # Update prediction
        self._update_prediction()

        # Invalidate view matrices
        self._view_matrix_left = None
        self._view_matrix_right = None

        self._last_update_time = timestamp

    def _update_tracking_state(self, confidence: float) -> None:
        """Update tracking state based on confidence."""
        old_state = self.tracking_state

        if confidence >= 0.9:
            new_state = HMDTrackingState.TRACKING
        elif confidence >= 0.5:
            new_state = HMDTrackingState.LIMITED
        elif confidence > 0.0:
            new_state = HMDTrackingState.LIMITED
        else:
            new_state = HMDTrackingState.LOST

        if new_state != old_state and new_state in HMD_STATE_TRANSITIONS.get(old_state, set()):
            self._transition_to_state(new_state)

    def _transition_to_state(self, new_state: HMDTrackingState) -> None:
        """Transition to a new tracking state."""
        old_state = self.tracking_state
        self.tracking_state = new_state

        # Fire callbacks
        for callback in self._on_tracking_state_changed:
            callback(old_state, new_state)

        # Fire specific callbacks
        if new_state == HMDTrackingState.LOST:
            for callback in self._on_tracking_lost:
                callback(self)
        elif old_state == HMDTrackingState.LOST and new_state in (
            HMDTrackingState.TRACKING,
            HMDTrackingState.LIMITED,
        ):
            for callback in self._on_tracking_restored:
                callback(self)

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def _update_prediction(self) -> None:
        """Update predicted pose based on velocity."""
        if not self._prediction_config.enabled:
            self._predicted_position = self.position
            self._predicted_orientation = self.orientation
            return

        # Predict position
        prediction_time = self._prediction_config.prediction_time_ms / 1000.0
        vel = self.linear_velocity
        self._predicted_position = (
            self.position[0] + vel[0] * prediction_time,
            self.position[1] + vel[1] * prediction_time,
            self.position[2] + vel[2] * prediction_time,
        )

        # Predict orientation (simple linear extrapolation)
        ang_vel = self.angular_velocity
        half_dt = prediction_time * 0.5
        q = self.orientation

        # Small angle approximation for angular velocity integration
        dq_x = ang_vel[0] * half_dt
        dq_y = ang_vel[1] * half_dt
        dq_z = ang_vel[2] * half_dt

        # Apply rotation delta
        self._predicted_orientation = self._normalize_quaternion((
            q[0] + (-q[1] * dq_x - q[2] * dq_y - q[3] * dq_z),
            q[1] + (q[0] * dq_x + q[2] * dq_z - q[3] * dq_y),
            q[2] + (q[0] * dq_y + q[3] * dq_x - q[1] * dq_z),
            q[3] + (q[0] * dq_z + q[1] * dq_y - q[2] * dq_x),
        ))

    @staticmethod
    def _normalize_quaternion(q: QuatTuple) -> QuatTuple:
        """Normalize a quaternion."""
        length = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
        if length < 1e-10:
            return (0.0, 0.0, 0.0, 1.0)
        inv_length = 1.0 / length
        return (q[0] * inv_length, q[1] * inv_length, q[2] * inv_length, q[3] * inv_length)

    def get_predicted_pose(self, prediction_time_ms: float = 0.0) -> Tuple[Vec3Tuple, QuatTuple]:
        """
        Get predicted pose for a specific prediction time.

        Args:
            prediction_time_ms: Time to predict ahead (ms). 0 uses default.

        Returns:
            Tuple of (position, orientation)
        """
        if prediction_time_ms <= 0.0:
            return self._predicted_position, self._predicted_orientation

        # Custom prediction time
        prediction_time = min(
            prediction_time_ms / 1000.0,
            self._prediction_config.max_prediction_time_ms / 1000.0,
        )

        vel = self.linear_velocity
        position = (
            self.position[0] + vel[0] * prediction_time,
            self.position[1] + vel[1] * prediction_time,
            self.position[2] + vel[2] * prediction_time,
        )

        # Simplified orientation prediction
        ang_vel = self.angular_velocity
        half_dt = prediction_time * 0.5
        q = self.orientation

        dq_x = ang_vel[0] * half_dt
        dq_y = ang_vel[1] * half_dt
        dq_z = ang_vel[2] * half_dt

        orientation = self._normalize_quaternion((
            q[0] + (-q[1] * dq_x - q[2] * dq_y - q[3] * dq_z),
            q[1] + (q[0] * dq_x + q[2] * dq_z - q[3] * dq_y),
            q[2] + (q[0] * dq_y + q[3] * dq_x - q[1] * dq_z),
            q[3] + (q[0] * dq_z + q[1] * dq_y - q[2] * dq_x),
        ))

        return position, orientation

    # =========================================================================
    # VIEW MATRICES
    # =========================================================================

    def get_left_view_matrix(self) -> List[List[float]]:
        """
        Compute left eye view matrix from pose.

        Returns:
            4x4 view matrix as nested lists
        """
        if self._view_matrix_left is None:
            self._view_matrix_left = self._compute_view_matrix(-self._display_info.ipd / 2)
        return self._view_matrix_left

    def get_right_view_matrix(self) -> List[List[float]]:
        """
        Compute right eye view matrix from pose.

        Returns:
            4x4 view matrix as nested lists
        """
        if self._view_matrix_right is None:
            self._view_matrix_right = self._compute_view_matrix(self._display_info.ipd / 2)
        return self._view_matrix_right

    def _compute_view_matrix(self, ipd_offset: float) -> List[List[float]]:
        """
        Compute view matrix with IPD offset.

        Args:
            ipd_offset: Inter-pupillary distance offset in meters

        Returns:
            4x4 view matrix
        """
        # Convert quaternion to rotation matrix
        q = self.orientation
        x, y, z, w = q

        # Rotation matrix from quaternion
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        r00 = 1 - 2 * (yy + zz)
        r01 = 2 * (xy - wz)
        r02 = 2 * (xz + wy)
        r10 = 2 * (xy + wz)
        r11 = 1 - 2 * (xx + zz)
        r12 = 2 * (yz - wx)
        r20 = 2 * (xz - wy)
        r21 = 2 * (yz + wx)
        r22 = 1 - 2 * (xx + yy)

        # Apply IPD offset in local space
        pos = self.position
        eye_x = pos[0] + r00 * ipd_offset
        eye_y = pos[1] + r10 * ipd_offset
        eye_z = pos[2] + r20 * ipd_offset

        # View matrix is inverse of camera transform
        # Transpose rotation, negate position in rotated space
        t_x = -(r00 * eye_x + r01 * eye_y + r02 * eye_z)
        t_y = -(r10 * eye_x + r11 * eye_y + r12 * eye_z)
        t_z = -(r20 * eye_x + r21 * eye_y + r22 * eye_z)

        return [
            [r00, r10, r20, 0.0],
            [r01, r11, r21, 0.0],
            [r02, r12, r22, 0.0],
            [t_x, t_y, t_z, 1.0],
        ]

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_tracking_state_changed(
        self,
        callback: Callable[[HMDTrackingState, HMDTrackingState], None]
    ) -> None:
        """Register callback for tracking state changes (old_state, new_state)."""
        self._on_tracking_state_changed.append(callback)

    def on_tracking_lost(self, callback: Callable[[HeadMountedDisplay], None]) -> None:
        """Register callback for when tracking is lost."""
        self._on_tracking_lost.append(callback)

    def on_tracking_restored(self, callback: Callable[[HeadMountedDisplay], None]) -> None:
        """Register callback for when tracking is restored."""
        self._on_tracking_restored.append(callback)

    def remove_tracking_state_callback(
        self,
        callback: Callable[[HMDTrackingState, HMDTrackingState], None]
    ) -> bool:
        """Remove a tracking state change callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        try:
            self._on_tracking_state_changed.remove(callback)
            return True
        except ValueError:
            return False

    def remove_tracking_lost_callback(
        self,
        callback: Callable[[HeadMountedDisplay], None]
    ) -> bool:
        """Remove a tracking lost callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        try:
            self._on_tracking_lost.remove(callback)
            return True
        except ValueError:
            return False

    def remove_tracking_restored_callback(
        self,
        callback: Callable[[HeadMountedDisplay], None]
    ) -> bool:
        """Remove a tracking restored callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        try:
            self._on_tracking_restored.remove(callback)
            return True
        except ValueError:
            return False

    def clear_callbacks(self) -> None:
        """Clear all registered callbacks."""
        self._on_tracking_state_changed.clear()
        self._on_tracking_lost.clear()
        self._on_tracking_restored.clear()

    # =========================================================================
    # STATE CONTROL
    # =========================================================================

    def enable(self) -> None:
        """Enable HMD tracking."""
        if self.tracking_state == HMDTrackingState.DISABLED:
            self._transition_to_state(HMDTrackingState.INITIALIZING)

    def disable(self) -> None:
        """Disable HMD tracking."""
        if self.tracking_state != HMDTrackingState.DISABLED:
            self._transition_to_state(HMDTrackingState.DISABLED)

    def recenter(self) -> None:
        """Recenter the HMD (reset position to origin, keep height)."""
        current_y = self.position[1]
        self.position = (0.0, current_y, 0.0)
        self.orientation = (0.0, 0.0, 0.0, 1.0)
        self._view_matrix_left = None
        self._view_matrix_right = None

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize HMD state to dictionary."""
        return {
            "device_id": self._device_id,
            "position": list(self.position),
            "orientation": list(self.orientation),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
            "tracking_state": self.tracking_state.name,
            "confidence": self.confidence,
            "display_info": {
                "resolution_per_eye": list(self._display_info.resolution_per_eye),
                "refresh_rate": self._display_info.refresh_rate,
                "field_of_view": list(self._display_info.field_of_view),
                "ipd": self._display_info.ipd,
            },
            "entity_id": self._entity_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HeadMountedDisplay:
        """Deserialize HMD state from dictionary."""
        display_data = data.get("display_info", {})
        display_info = HMDDisplayInfo(
            resolution_per_eye=tuple(display_data.get("resolution_per_eye", [1920, 1080])),
            refresh_rate=display_data.get("refresh_rate", 90.0),
            field_of_view=tuple(display_data.get("field_of_view", [90.0, 90.0])),
            ipd=display_data.get("ipd", 0.063),
        )

        hmd = cls(
            device_id=data.get("device_id", ""),
            display_info=display_info,
            entity_id=data.get("entity_id"),
        )

        hmd.position = tuple(data.get("position", [0.0, 0.0, 0.0]))
        hmd.orientation = tuple(data.get("orientation", [0.0, 0.0, 0.0, 1.0]))
        hmd.linear_velocity = tuple(data.get("linear_velocity", [0.0, 0.0, 0.0]))
        hmd.angular_velocity = tuple(data.get("angular_velocity", [0.0, 0.0, 0.0]))
        hmd.confidence = data.get("confidence", 0.0)

        state_name = data.get("tracking_state", "INITIALIZING")
        hmd.tracking_state = HMDTrackingState[state_name]

        return hmd

    def __repr__(self) -> str:
        return (
            f"HeadMountedDisplay(device_id={self._device_id!r}, "
            f"state={self.tracking_state.name}, confidence={self.confidence:.2f})"
        )


# Descriptor setup
HeadMountedDisplay.position.__set_name__(HeadMountedDisplay, "position")
HeadMountedDisplay.orientation.__set_name__(HeadMountedDisplay, "orientation")
HeadMountedDisplay.linear_velocity.__set_name__(HeadMountedDisplay, "linear_velocity")
HeadMountedDisplay.angular_velocity.__set_name__(HeadMountedDisplay, "angular_velocity")
HeadMountedDisplay.tracking_state.__set_name__(HeadMountedDisplay, "tracking_state")
HeadMountedDisplay.confidence.__set_name__(HeadMountedDisplay, "confidence")


__all__ = [
    "HeadMountedDisplay",
    "HMDTrackingState",
    "HMD_STATE_TRANSITIONS",
    "HMDDisplayInfo",
    "PredictionConfig",
]
