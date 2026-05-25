"""Eye tracking module for XR input with gaze, pupil, and fixation detection.

This module provides comprehensive eye tracking support following the Trinity Pattern
with Tracked and Range descriptors for eye data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable, Optional

from engine.xr.config import XR_CONFIG

logger = logging.getLogger(__name__)

# Type aliases for Trinity descriptors (to be replaced with actual imports)
Tracked = "Tracked"
Range = "Range"
Observable = "Observable"
Immutable = "Immutable"


class EyeId(IntEnum):
    """Identifier for individual eyes."""
    LEFT = 0
    RIGHT = 1
    COMBINED = 2  # Combined/cyclops gaze


class CalibrationState(IntEnum):
    """Eye tracking calibration state."""
    UNCALIBRATED = 0
    INITIAL = auto()       # Basic calibration complete
    DYNAMIC = auto()       # Dynamic calibration active
    PROFILE_LOADED = auto()  # User profile calibration loaded


class GazeState(IntEnum):
    """Current gaze behavior state."""
    UNKNOWN = 0
    FIXATION = auto()      # Eyes focused on a point
    SACCADE = auto()       # Rapid eye movement between points
    SMOOTH_PURSUIT = auto()  # Tracking a moving object
    BLINK = auto()         # Eyes closed during blink


@dataclass
class EyeData:
    """Data for a single eye.

    Attributes:
        pupil_position: 2D position of pupil in normalized eye coordinates
        pupil_diameter: Diameter of the pupil in millimeters
        openness: How open the eye is (0.0 = closed, 1.0 = fully open)
        gaze_origin: 3D origin point of the gaze ray in tracking space
        gaze_direction: 3D direction vector of the gaze
        is_valid: Whether the eye tracking data is valid
        confidence: Tracking confidence for this eye (0.0 to 1.0)
    """
    pupil_position: tuple[float, float] = (0.0, 0.0)
    pupil_diameter: float = XR_CONFIG.runtime.DEFAULT_PUPIL_DIAMETER_MM  # mm, typical range 2-8mm
    openness: float = 1.0
    gaze_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gaze_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    is_valid: bool = False
    confidence: float = 0.0


@dataclass
class FixationData:
    """Data about a gaze fixation.

    Attributes:
        position: 3D world position of the fixation point
        start_time: Timestamp when fixation started
        duration: How long the fixation has lasted (seconds)
        is_active: Whether the fixation is currently ongoing
        stability: How stable the fixation is (0.0 to 1.0)
    """
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_time: float = 0.0
    duration: float = 0.0
    is_active: bool = False
    stability: float = 0.0


@dataclass
class SaccadeData:
    """Data about a saccade (rapid eye movement).

    Attributes:
        start_position: 3D world position where saccade started
        end_position: 3D world position where saccade ended
        start_time: Timestamp when saccade started
        duration: Saccade duration in seconds
        amplitude: Angular amplitude of the saccade in degrees
        peak_velocity: Peak angular velocity in degrees/second
    """
    start_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    end_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_time: float = 0.0
    duration: float = 0.0
    amplitude: float = 0.0
    peak_velocity: float = 0.0


@dataclass
class BlinkData:
    """Data about an eye blink.

    Attributes:
        start_time: Timestamp when blink started
        duration: Blink duration in seconds
        is_complete: Whether the blink has completed
        eye: Which eye blinked (or COMBINED for both)
    """
    start_time: float = 0.0
    duration: float = 0.0
    is_complete: bool = False
    eye: EyeId = EyeId.COMBINED


@dataclass
class EyeTrackingData:
    """Eye tracking component with gaze, pupil, and fixation data.

    This component follows the Trinity Pattern with appropriate descriptors:
    - Tracked for gaze and pupil data
    - Range for bounded values like openness and confidence

    Attributes:
        gaze_origin: Combined gaze ray origin (between both eyes)
        gaze_direction: Combined gaze ray direction
        left_eye: Data for the left eye
        right_eye: Data for the right eye
        is_fixating: Whether currently fixating on a point
        fixation_point: 3D world position of current fixation
        calibration_state: Current calibration level
        confidence: Overall tracking confidence
    """
    # Combined gaze (Tracked)
    gaze_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gaze_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    # Per-eye data (Tracked)
    left_eye: EyeData = field(default_factory=EyeData)
    right_eye: EyeData = field(default_factory=EyeData)

    # Pupil data shortcuts (Tracked)
    left_pupil_position: tuple[float, float] = (0.0, 0.0)
    right_pupil_position: tuple[float, float] = (0.0, 0.0)
    left_pupil_diameter: float = XR_CONFIG.runtime.DEFAULT_PUPIL_DIAMETER_MM
    right_pupil_diameter: float = XR_CONFIG.runtime.DEFAULT_PUPIL_DIAMETER_MM
    left_openness: float = 1.0  # Range(0, 1)
    right_openness: float = 1.0  # Range(0, 1)

    # Fixation detection (Tracked)
    is_fixating: bool = False
    fixation_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixation_duration: float = 0.0

    # Gaze state (Tracked + Observable)
    gaze_state: GazeState = GazeState.UNKNOWN

    # Calibration (Tracked + Observable)
    calibration_state: CalibrationState = CalibrationState.UNCALIBRATED
    is_calibrated: bool = False

    # Tracking state (Tracked + Range)
    confidence: float = 0.0  # Range(0, 1)
    is_tracked: bool = False

    # Internal state for detection algorithms
    _gaze_history: list[tuple[float, float, float]] = field(default_factory=list, repr=False)
    _last_update_time: float = field(default=0.0, repr=False)
    _fixation_start_time: float = field(default=0.0, repr=False)
    _blink_start_time: float = field(default=0.0, repr=False)

    @property
    def convergence_distance(self) -> float:
        """Calculate the vergence/convergence distance of both eyes.

        This is the distance at which both eye gaze rays intersect,
        indicating where the user is focusing.

        Returns:
            Distance in meters, or float('inf') if eyes are parallel
        """
        if not (self.left_eye.is_valid and self.right_eye.is_valid):
            return float('inf')

        # Get gaze rays for both eyes
        left_origin = self.left_eye.gaze_origin
        left_dir = self.left_eye.gaze_direction
        right_origin = self.right_eye.gaze_origin
        right_dir = self.right_eye.gaze_direction

        # Find closest point between the two rays
        # Using the algorithm for closest approach of two 3D lines
        w0 = (
            left_origin[0] - right_origin[0],
            left_origin[1] - right_origin[1],
            left_origin[2] - right_origin[2],
        )

        a = _dot(left_dir, left_dir)
        b = _dot(left_dir, right_dir)
        c = _dot(right_dir, right_dir)
        d = _dot(left_dir, w0)
        e = _dot(right_dir, w0)

        denom = a * c - b * b
        if abs(denom) < 1e-6:
            return float('inf')  # Parallel rays

        t = (b * e - c * d) / denom

        # Calculate the intersection point on the left ray
        intersection = (
            left_origin[0] + t * left_dir[0],
            left_origin[1] + t * left_dir[1],
            left_origin[2] + t * left_dir[2],
        )

        # Distance from midpoint between eyes to intersection
        midpoint = (
            (left_origin[0] + right_origin[0]) / 2,
            (left_origin[1] + right_origin[1]) / 2,
            (left_origin[2] + right_origin[2]) / 2,
        )

        return math.sqrt(
            (intersection[0] - midpoint[0])**2 +
            (intersection[1] - midpoint[1])**2 +
            (intersection[2] - midpoint[2])**2
        )

    @property
    def average_pupil_diameter(self) -> float:
        """Get the average pupil diameter of both eyes.

        Returns:
            Average diameter in millimeters
        """
        if self.left_eye.is_valid and self.right_eye.is_valid:
            return (self.left_pupil_diameter + self.right_pupil_diameter) / 2
        elif self.left_eye.is_valid:
            return self.left_pupil_diameter
        elif self.right_eye.is_valid:
            return self.right_pupil_diameter
        return 0.0

    @property
    def is_blinking(self) -> bool:
        """Check if the user is currently blinking.

        Returns:
            True if either eye openness is below threshold
        """
        return self.left_openness < XR_CONFIG.runtime.BLINK_THRESHOLD or self.right_openness < XR_CONFIG.runtime.BLINK_THRESHOLD

    def get_gaze_point_at_distance(self, distance: float) -> tuple[float, float, float]:
        """Calculate the 3D point where gaze ray hits at a given distance.

        Args:
            distance: Distance from gaze origin in meters

        Returns:
            3D world position
        """
        return (
            self.gaze_origin[0] + self.gaze_direction[0] * distance,
            self.gaze_origin[1] + self.gaze_direction[1] * distance,
            self.gaze_origin[2] + self.gaze_direction[2] * distance,
        )

    def update(
        self,
        gaze_origin: tuple[float, float, float],
        gaze_direction: tuple[float, float, float],
        left_pupil_position: Optional[tuple[float, float]] = None,
        right_pupil_position: Optional[tuple[float, float]] = None,
        left_pupil_diameter: Optional[float] = None,
        right_pupil_diameter: Optional[float] = None,
        left_openness: Optional[float] = None,
        right_openness: Optional[float] = None,
        confidence: float = 1.0,
        timestamp: float = 0.0,
    ) -> None:
        """Update eye tracking data.

        Args:
            gaze_origin: Combined gaze ray origin
            gaze_direction: Combined gaze ray direction (should be normalized)
            left_pupil_position: Optional left eye pupil position
            right_pupil_position: Optional right eye pupil position
            left_pupil_diameter: Optional left pupil diameter in mm
            right_pupil_diameter: Optional right pupil diameter in mm
            left_openness: Optional left eye openness (0-1)
            right_openness: Optional right eye openness (0-1)
            confidence: Overall tracking confidence
            timestamp: Update timestamp
        """
        self.gaze_origin = gaze_origin
        self.gaze_direction = _normalize(gaze_direction)
        self.confidence = max(0.0, min(1.0, confidence))
        self.is_tracked = confidence > 0.0
        self._last_update_time = timestamp

        # Update per-eye data
        if left_pupil_position is not None:
            self.left_pupil_position = left_pupil_position
            self.left_eye.pupil_position = left_pupil_position
            self.left_eye.is_valid = True

        if right_pupil_position is not None:
            self.right_pupil_position = right_pupil_position
            self.right_eye.pupil_position = right_pupil_position
            self.right_eye.is_valid = True

        if left_pupil_diameter is not None:
            self.left_pupil_diameter = left_pupil_diameter
            self.left_eye.pupil_diameter = left_pupil_diameter

        if right_pupil_diameter is not None:
            self.right_pupil_diameter = right_pupil_diameter
            self.right_eye.pupil_diameter = right_pupil_diameter

        if left_openness is not None:
            self.left_openness = max(0.0, min(1.0, left_openness))
            self.left_eye.openness = self.left_openness

        if right_openness is not None:
            self.right_openness = max(0.0, min(1.0, right_openness))
            self.right_eye.openness = self.right_openness

        # Store gaze history for fixation/saccade detection
        self._gaze_history.append(gaze_direction)
        if len(self._gaze_history) > 30:  # Keep ~0.5s at 60Hz
            self._gaze_history.pop(0)


class FixationDetector:
    """Detects fixations and saccades from eye tracking data.

    Uses velocity-threshold identification (I-VT) algorithm for
    detecting fixations and saccades.

    Attributes:
        velocity_threshold: Angular velocity threshold for fixation (deg/s)
        min_fixation_duration: Minimum duration for valid fixation (seconds)
        min_saccade_duration: Minimum duration for valid saccade (seconds)
    """

    __slots__ = (
        'velocity_threshold',
        'min_fixation_duration',
        'min_saccade_duration',
        '_current_fixation',
        '_current_saccade',
        '_gaze_velocity_history',
        '_last_gaze_direction',
        '_last_timestamp',
    )

    def __init__(
        self,
        velocity_threshold: float = XR_CONFIG.runtime.SACCADE_VELOCITY_THRESHOLD,  # deg/s
        min_fixation_duration: float = XR_CONFIG.runtime.FIXATION_DURATION_MS / 1000.0,  # 100ms
        min_saccade_duration: float = 0.02,  # 20ms
    ):
        """Initialize the fixation detector.

        Args:
            velocity_threshold: Angular velocity threshold for fixation detection
            min_fixation_duration: Minimum time to count as fixation
            min_saccade_duration: Minimum time to count as saccade
        """
        self.velocity_threshold = velocity_threshold
        self.min_fixation_duration = min_fixation_duration
        self.min_saccade_duration = min_saccade_duration
        self._current_fixation: Optional[FixationData] = None
        self._current_saccade: Optional[SaccadeData] = None
        self._gaze_velocity_history: list[float] = []
        self._last_gaze_direction: Optional[tuple[float, float, float]] = None
        self._last_timestamp: float = 0.0

    def update(self, eye_data: EyeTrackingData, timestamp: float) -> GazeState:
        """Update fixation detection with new eye data.

        Args:
            eye_data: Current eye tracking data
            timestamp: Current timestamp

        Returns:
            Current gaze state (FIXATION, SACCADE, or UNKNOWN)
        """
        if not eye_data.is_tracked:
            return GazeState.UNKNOWN

        # Calculate angular velocity
        velocity = self._calculate_gaze_velocity(eye_data.gaze_direction, timestamp)

        # Store velocity history for analysis
        self._gaze_velocity_history.append(velocity)
        if len(self._gaze_velocity_history) > 30:
            self._gaze_velocity_history.pop(0)

        self._last_gaze_direction = eye_data.gaze_direction
        self._last_timestamp = timestamp

        # Determine state based on velocity
        if velocity < self.velocity_threshold:
            # Potentially in fixation
            return self._process_fixation(eye_data, timestamp)
        else:
            # Potentially in saccade
            return self._process_saccade(eye_data, timestamp, velocity)

    def get_current_fixation(self) -> Optional[FixationData]:
        """Get the current fixation if active.

        Returns:
            FixationData if fixating, None otherwise
        """
        return self._current_fixation

    def get_current_saccade(self) -> Optional[SaccadeData]:
        """Get the current saccade if active.

        Returns:
            SaccadeData if in saccade, None otherwise
        """
        return self._current_saccade

    def _calculate_gaze_velocity(
        self,
        current_direction: tuple[float, float, float],
        timestamp: float,
    ) -> float:
        """Calculate angular velocity of gaze.

        Args:
            current_direction: Current gaze direction
            timestamp: Current timestamp

        Returns:
            Angular velocity in degrees per second
        """
        if self._last_gaze_direction is None or timestamp <= self._last_timestamp:
            return 0.0

        # Calculate angle between directions
        dot = _dot(self._last_gaze_direction, current_direction)
        dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability
        angle_rad = math.acos(dot)
        angle_deg = math.degrees(angle_rad)

        # Calculate time delta
        dt = timestamp - self._last_timestamp
        if dt > 0:
            return angle_deg / dt
        return 0.0

    def _process_fixation(self, eye_data: EyeTrackingData, timestamp: float) -> GazeState:
        """Process potential fixation state.

        Args:
            eye_data: Current eye tracking data
            timestamp: Current timestamp

        Returns:
            GazeState.FIXATION if valid fixation
        """
        # End any active saccade
        if self._current_saccade is not None:
            self._current_saccade = None

        # Start or continue fixation
        if self._current_fixation is None:
            # Start new fixation
            gaze_point = eye_data.get_gaze_point_at_distance(1.0)
            self._current_fixation = FixationData(
                position=gaze_point,
                start_time=timestamp,
                duration=0.0,
                is_active=True,
                stability=1.0,
            )
        else:
            # Continue existing fixation
            self._current_fixation.duration = timestamp - self._current_fixation.start_time

            # Update stability based on velocity variance
            if len(self._gaze_velocity_history) > 5:
                avg_velocity = sum(self._gaze_velocity_history[-5:]) / 5
                self._current_fixation.stability = max(0.0, 1.0 - avg_velocity / self.velocity_threshold)

        # Update eye data
        if self._current_fixation.duration >= self.min_fixation_duration:
            eye_data.is_fixating = True
            eye_data.fixation_point = self._current_fixation.position
            eye_data.fixation_duration = self._current_fixation.duration
            eye_data.gaze_state = GazeState.FIXATION
            return GazeState.FIXATION

        return GazeState.UNKNOWN

    def _process_saccade(
        self,
        eye_data: EyeTrackingData,
        timestamp: float,
        velocity: float,
    ) -> GazeState:
        """Process potential saccade state.

        Args:
            eye_data: Current eye tracking data
            timestamp: Current timestamp
            velocity: Current gaze velocity

        Returns:
            GazeState.SACCADE if valid saccade
        """
        # End any active fixation
        if self._current_fixation is not None:
            eye_data.is_fixating = False
            self._current_fixation = None

        # Start or continue saccade
        gaze_point = eye_data.get_gaze_point_at_distance(1.0)

        if self._current_saccade is None:
            # Start new saccade
            self._current_saccade = SaccadeData(
                start_position=gaze_point,
                end_position=gaze_point,
                start_time=timestamp,
                duration=0.0,
                amplitude=0.0,
                peak_velocity=velocity,
            )
        else:
            # Continue existing saccade
            self._current_saccade.duration = timestamp - self._current_saccade.start_time
            self._current_saccade.end_position = gaze_point
            self._current_saccade.peak_velocity = max(self._current_saccade.peak_velocity, velocity)

            # Calculate amplitude
            start = self._current_saccade.start_position
            end = self._current_saccade.end_position
            # Approximate angle from positions at 1m distance
            self._current_saccade.amplitude = math.degrees(math.atan2(
                math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2),
                1.0
            ))

        if self._current_saccade.duration >= self.min_saccade_duration:
            eye_data.gaze_state = GazeState.SACCADE
            return GazeState.SACCADE

        return GazeState.UNKNOWN


class BlinkDetector:
    """Detects eye blinks from eye tracking data.

    Attributes:
        openness_threshold: Eye openness below which counts as closed
        min_blink_duration: Minimum duration for valid blink (seconds)
        max_blink_duration: Maximum duration for valid blink (seconds)
    """

    __slots__ = (
        'openness_threshold',
        'min_blink_duration',
        'max_blink_duration',
        '_blink_start_time',
        '_is_blinking',
        '_blink_callbacks',
    )

    def __init__(
        self,
        openness_threshold: float = 0.2,
        min_blink_duration: float = 0.05,  # 50ms
        max_blink_duration: float = 0.4,   # 400ms
    ):
        """Initialize the blink detector.

        Args:
            openness_threshold: Eye openness below this is considered closed
            min_blink_duration: Minimum time for valid blink
            max_blink_duration: Maximum time for valid blink
        """
        self.openness_threshold = openness_threshold
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration
        self._blink_start_time: float = 0.0
        self._is_blinking: bool = False
        self._blink_callbacks: list[Callable[[BlinkData], None]] = []

    def update(self, eye_data: EyeTrackingData, timestamp: float) -> Optional[BlinkData]:
        """Update blink detection with new eye data.

        Args:
            eye_data: Current eye tracking data
            timestamp: Current timestamp

        Returns:
            BlinkData if a blink was completed this frame, None otherwise
        """
        is_closed = (
            eye_data.left_openness < self.openness_threshold and
            eye_data.right_openness < self.openness_threshold
        )

        if is_closed and not self._is_blinking:
            # Start of blink
            self._is_blinking = True
            self._blink_start_time = timestamp
            return None

        elif not is_closed and self._is_blinking:
            # End of blink
            self._is_blinking = False
            duration = timestamp - self._blink_start_time

            # Check if valid blink duration
            if self.min_blink_duration <= duration <= self.max_blink_duration:
                blink = BlinkData(
                    start_time=self._blink_start_time,
                    duration=duration,
                    is_complete=True,
                    eye=EyeId.COMBINED,
                )
                self._fire_blink_event(blink)
                return blink

        return None

    def add_blink_callback(self, callback: Callable[[BlinkData], None]) -> None:
        """Register a callback for blink events.

        Args:
            callback: Function to call when blink is detected
        """
        self._blink_callbacks.append(callback)

    def remove_blink_callback(self, callback: Callable[[BlinkData], None]) -> bool:
        """Remove a blink callback.

        Args:
            callback: The callback to remove

        Returns:
            True if callback was removed
        """
        try:
            self._blink_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _fire_blink_event(self, blink: BlinkData) -> None:
        """Fire blink event to all callbacks.

        Args:
            blink: The blink data
        """
        for callback in self._blink_callbacks:
            try:
                callback(blink)
            except Exception as e:
                logger.warning(f"Blink callback error: {e}")


@dataclass
class CalibrationPoint:
    """A calibration target point.

    Attributes:
        target_position: Where the user should look
        measured_gaze: Where the user actually looked
        error: Angular error in degrees
        is_valid: Whether this point was successfully calibrated
    """
    target_position: tuple[float, float, float]
    measured_gaze: Optional[tuple[float, float, float]] = None
    error: float = 0.0
    is_valid: bool = False


class EyeCalibration:
    """Eye tracking calibration system.

    Manages the calibration process for eye tracking, supporting:
    - Initial multi-point calibration
    - Dynamic/continuous calibration
    - User profile loading/saving

    Attributes:
        state: Current calibration state
        calibration_points: List of calibration target points
        average_error: Average calibration error in degrees
    """

    __slots__ = (
        'state',
        'calibration_points',
        'average_error',
        '_current_point_index',
        '_samples_per_point',
        '_collected_samples',
        '_calibration_data',
    )

    def __init__(self, num_points: int = 9, samples_per_point: int = 10):
        """Initialize the calibration system.

        Args:
            num_points: Number of calibration points (typically 5, 9, or 13)
            samples_per_point: Number of gaze samples to collect per point
        """
        self.state = CalibrationState.UNCALIBRATED
        self.calibration_points: list[CalibrationPoint] = []
        self.average_error = float('inf')
        self._current_point_index = 0
        self._samples_per_point = samples_per_point
        self._collected_samples: list[tuple[float, float, float]] = []
        self._calibration_data: dict[str, Any] = {}

        # Generate default calibration point positions (3x3 grid)
        self._generate_calibration_points(num_points)

    def start_calibration(self) -> CalibrationPoint:
        """Start the calibration process.

        Returns:
            The first calibration target point
        """
        self.state = CalibrationState.INITIAL
        self._current_point_index = 0
        self._collected_samples.clear()

        for point in self.calibration_points:
            point.measured_gaze = None
            point.error = 0.0
            point.is_valid = False

        return self.calibration_points[0]

    def add_gaze_sample(
        self,
        gaze_direction: tuple[float, float, float],
    ) -> Optional[CalibrationPoint]:
        """Add a gaze sample for the current calibration point.

        Args:
            gaze_direction: The user's current gaze direction

        Returns:
            The next calibration point, or None if calibration is complete
        """
        if self.state != CalibrationState.INITIAL:
            return None

        # Check if we're past the last point
        if self._current_point_index >= len(self.calibration_points):
            return None

        self._collected_samples.append(gaze_direction)

        # Check if we have enough samples for this point
        if len(self._collected_samples) >= self._samples_per_point:
            # Calculate average gaze for this point
            avg_gaze = self._average_directions(self._collected_samples)

            current_point = self.calibration_points[self._current_point_index]
            current_point.measured_gaze = avg_gaze
            current_point.is_valid = True

            # Calculate error (angle between target and measured)
            target_dir = _normalize(current_point.target_position)
            dot = _dot(target_dir, avg_gaze)
            dot = max(-1.0, min(1.0, dot))
            current_point.error = math.degrees(math.acos(dot))

            # Move to next point
            self._collected_samples.clear()
            self._current_point_index += 1

            if self._current_point_index < len(self.calibration_points):
                return self.calibration_points[self._current_point_index]
            else:
                # Calibration complete
                self._finalize_calibration()
                return None

        # Return current target if still collecting samples
        if self._current_point_index < len(self.calibration_points):
            return self.calibration_points[self._current_point_index]
        return None

    def get_current_target(self) -> Optional[CalibrationPoint]:
        """Get the current calibration target.

        Returns:
            Current target point, or None if not calibrating
        """
        if self.state != CalibrationState.INITIAL:
            return None
        if self._current_point_index >= len(self.calibration_points):
            return None
        return self.calibration_points[self._current_point_index]

    def is_calibrating(self) -> bool:
        """Check if calibration is in progress.

        Returns:
            True if currently calibrating
        """
        return self.state == CalibrationState.INITIAL

    def is_calibrated(self) -> bool:
        """Check if calibration is complete.

        Returns:
            True if calibration is complete and valid
        """
        return self.state in (CalibrationState.INITIAL, CalibrationState.DYNAMIC, CalibrationState.PROFILE_LOADED) and self.average_error < 5.0

    def save_profile(self) -> dict[str, Any]:
        """Save calibration data as a profile.

        Returns:
            Dictionary containing calibration profile data
        """
        return {
            'state': self.state.value,
            'average_error': self.average_error,
            'calibration_data': self._calibration_data.copy(),
            'points': [
                {
                    'target': p.target_position,
                    'measured': p.measured_gaze,
                    'error': p.error,
                    'valid': p.is_valid,
                }
                for p in self.calibration_points
            ],
        }

    def load_profile(self, profile: dict[str, Any]) -> bool:
        """Load a calibration profile.

        Args:
            profile: Profile data from save_profile()

        Returns:
            True if profile loaded successfully
        """
        try:
            self.state = CalibrationState.PROFILE_LOADED
            self.average_error = profile.get('average_error', float('inf'))
            self._calibration_data = profile.get('calibration_data', {})

            points_data = profile.get('points', [])
            self.calibration_points = [
                CalibrationPoint(
                    target_position=p['target'],
                    measured_gaze=p.get('measured'),
                    error=p.get('error', 0.0),
                    is_valid=p.get('valid', False),
                )
                for p in points_data
            ]
            return True
        except (KeyError, TypeError):
            return False

    def _generate_calibration_points(self, num_points: int) -> None:
        """Generate calibration point positions.

        Args:
            num_points: Number of points to generate
        """
        self.calibration_points.clear()

        # Generate points on a grid at 1m distance
        if num_points == 5:
            # Center and 4 corners
            positions = [
                (0.0, 0.0, -1.0),
                (-0.3, 0.2, -1.0),
                (0.3, 0.2, -1.0),
                (-0.3, -0.2, -1.0),
                (0.3, -0.2, -1.0),
            ]
        elif num_points == 9:
            # 3x3 grid
            positions = []
            for y in [0.2, 0.0, -0.2]:
                for x in [-0.3, 0.0, 0.3]:
                    positions.append((x, y, -1.0))
        elif num_points == 13:
            # 3x3 grid + 4 mid-edge points
            positions = []
            for y in [0.2, 0.0, -0.2]:
                for x in [-0.3, 0.0, 0.3]:
                    positions.append((x, y, -1.0))
            # Add mid-edge points
            positions.extend([
                (-0.15, 0.1, -1.0),
                (0.15, 0.1, -1.0),
                (-0.15, -0.1, -1.0),
                (0.15, -0.1, -1.0),
            ])
        else:
            # Default to single center point
            positions = [(0.0, 0.0, -1.0)]

        for pos in positions:
            self.calibration_points.append(CalibrationPoint(target_position=pos))

    def _average_directions(
        self,
        directions: list[tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Calculate average direction from multiple samples.

        Args:
            directions: List of direction vectors

        Returns:
            Normalized average direction
        """
        if not directions:
            return (0.0, 0.0, -1.0)

        avg = [0.0, 0.0, 0.0]
        for d in directions:
            avg[0] += d[0]
            avg[1] += d[1]
            avg[2] += d[2]

        return _normalize((avg[0], avg[1], avg[2]))

    def _finalize_calibration(self) -> None:
        """Finalize the calibration process."""
        # Calculate average error
        valid_errors = [p.error for p in self.calibration_points if p.is_valid]
        if valid_errors:
            self.average_error = sum(valid_errors) / len(valid_errors)
        else:
            self.average_error = float('inf')

        # Store calibration data for correction
        self._calibration_data = {
            'points': [(p.target_position, p.measured_gaze) for p in self.calibration_points if p.is_valid],
            'error': self.average_error,
        }


class EyeTracker:
    """High-level eye tracking manager.

    This class manages eye tracking state, fixation detection,
    blink detection, and calibration.

    Attributes:
        eye_data: Current eye tracking data
        calibration: Calibration system
        fixation_detector: Fixation/saccade detection
        blink_detector: Blink detection
    """

    __slots__ = (
        'eye_data',
        'calibration',
        'fixation_detector',
        'blink_detector',
        '_update_callbacks',
    )

    def __init__(
        self,
        calibration: Optional[EyeCalibration] = None,
        fixation_detector: Optional[FixationDetector] = None,
        blink_detector: Optional[BlinkDetector] = None,
    ):
        """Initialize the eye tracker.

        Args:
            calibration: Optional custom calibration system
            fixation_detector: Optional custom fixation detector
            blink_detector: Optional custom blink detector
        """
        self.eye_data = EyeTrackingData()
        self.calibration = calibration or EyeCalibration()
        self.fixation_detector = fixation_detector or FixationDetector()
        self.blink_detector = blink_detector or BlinkDetector()
        self._update_callbacks: list[Callable[[EyeTrackingData], None]] = []

    def update(
        self,
        gaze_origin: tuple[float, float, float],
        gaze_direction: tuple[float, float, float],
        timestamp: float,
        left_pupil_position: Optional[tuple[float, float]] = None,
        right_pupil_position: Optional[tuple[float, float]] = None,
        left_pupil_diameter: Optional[float] = None,
        right_pupil_diameter: Optional[float] = None,
        left_openness: Optional[float] = None,
        right_openness: Optional[float] = None,
        confidence: float = 1.0,
    ) -> None:
        """Update eye tracking with new data.

        Args:
            gaze_origin: Combined gaze ray origin
            gaze_direction: Combined gaze ray direction
            timestamp: Current timestamp
            left_pupil_position: Optional left pupil position
            right_pupil_position: Optional right pupil position
            left_pupil_diameter: Optional left pupil diameter
            right_pupil_diameter: Optional right pupil diameter
            left_openness: Optional left eye openness
            right_openness: Optional right eye openness
            confidence: Tracking confidence
        """
        # Update eye data
        self.eye_data.update(
            gaze_origin=gaze_origin,
            gaze_direction=gaze_direction,
            left_pupil_position=left_pupil_position,
            right_pupil_position=right_pupil_position,
            left_pupil_diameter=left_pupil_diameter,
            right_pupil_diameter=right_pupil_diameter,
            left_openness=left_openness,
            right_openness=right_openness,
            confidence=confidence,
            timestamp=timestamp,
        )

        # Update calibration state
        self.eye_data.calibration_state = self.calibration.state
        self.eye_data.is_calibrated = self.calibration.is_calibrated()

        # Detect fixations/saccades
        self.fixation_detector.update(self.eye_data, timestamp)

        # Detect blinks
        self.blink_detector.update(self.eye_data, timestamp)

        # Fire callbacks
        for callback in self._update_callbacks:
            try:
                callback(self.eye_data)
            except Exception as e:
                logger.warning(f"Eye tracking update callback error: {e}")

    def add_update_callback(self, callback: Callable[[EyeTrackingData], None]) -> None:
        """Register a callback for eye tracking updates.

        Args:
            callback: Function to call on each update
        """
        self._update_callbacks.append(callback)

    def remove_update_callback(self, callback: Callable[[EyeTrackingData], None]) -> bool:
        """Remove an update callback.

        Args:
            callback: The callback to remove

        Returns:
            True if callback was removed
        """
        try:
            self._update_callbacks.remove(callback)
            return True
        except ValueError:
            return False


# Helper functions

def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Calculate dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize a 3D vector."""
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length > 1e-6:
        return (v[0] / length, v[1] / length, v[2] / length)
    return (0.0, 0.0, -1.0)  # Default forward direction


# Type alias for convenience
Any = object  # Placeholder for typing.Any to avoid import
