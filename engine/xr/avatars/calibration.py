"""Avatar calibration for XR.

Provides height, arm span, and floor level calibration
for accurate avatar representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.xr.config import XR_CONFIG


class CalibrationState(Enum):
    """Calibration process state."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()


class CalibrationStep(Enum):
    """Individual calibration steps."""
    FLOOR_DETECTION = auto()
    HEIGHT_MEASUREMENT = auto()
    ARM_SPAN_MEASUREMENT = auto()
    T_POSE = auto()
    A_POSE = auto()


@dataclass(slots=True)
class CalibrationData:
    """Stores calibration measurements.

    Attributes:
        height: Player height in meters
        arm_span: Player arm span in meters
        floor_level: Y position of floor
        eye_height: Eye level relative to floor
        shoulder_width: Distance between shoulders
        arm_length: Length of arm (shoulder to wrist)
        leg_length: Length of leg (hip to ankle)
        torso_length: Length of torso (shoulder to hip)
    """
    height: float = XR_CONFIG.avatar.DEFAULT_AVATAR_HEIGHT_M
    arm_span: float = XR_CONFIG.avatar.DEFAULT_AVATAR_HEIGHT_M
    floor_level: float = 0.0
    eye_height: float = 1.6
    shoulder_width: float = 0.4
    arm_length: float = 0.6
    leg_length: float = 0.85
    torso_length: float = 0.55

    def calculate_proportions(self) -> None:
        """Calculate body proportions from height and arm span."""
        # Standard human proportions (approximations)
        self.eye_height = self.height * 0.94
        self.shoulder_width = self.arm_span * 0.24
        self.arm_length = self.arm_span * 0.35
        self.leg_length = self.height * 0.5
        self.torso_length = self.height * 0.32

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary of calibration values
        """
        return {
            "height": self.height,
            "arm_span": self.arm_span,
            "floor_level": self.floor_level,
            "eye_height": self.eye_height,
            "shoulder_width": self.shoulder_width,
            "arm_length": self.arm_length,
            "leg_length": self.leg_length,
            "torso_length": self.torso_length,
        }

    @staticmethod
    def from_dict(data: dict[str, float]) -> CalibrationData:
        """Create from dictionary.

        Args:
            data: Dictionary of calibration values

        Returns:
            CalibrationData instance
        """
        return CalibrationData(
            height=data.get("height", 1.7),
            arm_span=data.get("arm_span", 1.7),
            floor_level=data.get("floor_level", 0.0),
            eye_height=data.get("eye_height", 1.6),
            shoulder_width=data.get("shoulder_width", 0.4),
            arm_length=data.get("arm_length", 0.6),
            leg_length=data.get("leg_length", 0.85),
            torso_length=data.get("torso_length", 0.55),
        )


class AvatarCalibration:
    """Handles avatar calibration process.

    Provides guided calibration for measuring player dimensions
    and adjusting avatar proportions accordingly.
    """
    __slots__ = (
        '_state', '_current_step', '_data',
        '_samples', '_sample_count', '_on_step_complete',
        '_on_calibration_complete', '_min_samples',
        '_hmd_samples', '_left_hand_samples', '_right_hand_samples'
    )

    def __init__(
        self,
        min_samples: int = XR_CONFIG.avatar.CALIBRATION_SAMPLE_COUNT,
        on_step_complete: Optional[Callable[[CalibrationStep], None]] = None,
        on_calibration_complete: Optional[Callable[[CalibrationData], None]] = None,
    ) -> None:
        """Initialize calibration.

        Args:
            min_samples: Minimum samples per measurement
            on_step_complete: Callback when step completes
            on_calibration_complete: Callback when calibration completes
        """
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")

        self._state = CalibrationState.NOT_STARTED
        self._current_step: Optional[CalibrationStep] = None
        self._data = CalibrationData()
        self._min_samples = min_samples

        # Sample storage
        self._hmd_samples: list[Vec3] = []
        self._left_hand_samples: list[Vec3] = []
        self._right_hand_samples: list[Vec3] = []
        self._sample_count = 0

        # Callbacks
        self._on_step_complete = on_step_complete
        self._on_calibration_complete = on_calibration_complete

    @property
    def state(self) -> CalibrationState:
        """Get current calibration state."""
        return self._state

    @property
    def current_step(self) -> Optional[CalibrationStep]:
        """Get current calibration step."""
        return self._current_step

    @property
    def data(self) -> CalibrationData:
        """Get calibration data."""
        return self._data

    @property
    def progress(self) -> float:
        """Get calibration progress (0-1)."""
        if self._state == CalibrationState.NOT_STARTED:
            return 0.0
        elif self._state == CalibrationState.COMPLETED:
            return 1.0
        elif self._current_step is None:
            return 0.0
        else:
            # Progress through current step
            step_progress = min(1.0, self._sample_count / self._min_samples)

            # Total steps completed
            step_order = [
                CalibrationStep.FLOOR_DETECTION,
                CalibrationStep.HEIGHT_MEASUREMENT,
                CalibrationStep.ARM_SPAN_MEASUREMENT,
            ]

            try:
                step_idx = step_order.index(self._current_step)
                steps_complete = step_idx / len(step_order)
                current_step_weight = 1.0 / len(step_order)
                return steps_complete + step_progress * current_step_weight
            except ValueError:
                return step_progress

    @property
    def height(self) -> float:
        """Get calibrated height."""
        return self._data.height

    @property
    def arm_span(self) -> float:
        """Get calibrated arm span."""
        return self._data.arm_span

    @property
    def floor_level(self) -> float:
        """Get calibrated floor level."""
        return self._data.floor_level

    def start(self) -> None:
        """Start the calibration process."""
        self._state = CalibrationState.IN_PROGRESS
        self._current_step = CalibrationStep.FLOOR_DETECTION
        self._clear_samples()

    def cancel(self) -> None:
        """Cancel the calibration process."""
        self._state = CalibrationState.NOT_STARTED
        self._current_step = None
        self._clear_samples()

    def _clear_samples(self) -> None:
        """Clear all collected samples."""
        self._hmd_samples.clear()
        self._left_hand_samples.clear()
        self._right_hand_samples.clear()
        self._sample_count = 0

    def add_sample(
        self,
        hmd_position: Vec3,
        left_hand_position: Optional[Vec3] = None,
        right_hand_position: Optional[Vec3] = None,
    ) -> None:
        """Add a tracking sample for calibration.

        Args:
            hmd_position: Current HMD position
            left_hand_position: Current left hand position
            right_hand_position: Current right hand position
        """
        if self._state != CalibrationState.IN_PROGRESS:
            return

        self._hmd_samples.append(hmd_position)
        if left_hand_position:
            self._left_hand_samples.append(left_hand_position)
        if right_hand_position:
            self._right_hand_samples.append(right_hand_position)

        self._sample_count += 1

        # Check if we have enough samples for current step
        if self._sample_count >= self._min_samples:
            self._complete_current_step()

    def _complete_current_step(self) -> None:
        """Complete the current calibration step."""
        if self._current_step == CalibrationStep.FLOOR_DETECTION:
            self._calculate_floor_level()
            self._advance_step(CalibrationStep.HEIGHT_MEASUREMENT)

        elif self._current_step == CalibrationStep.HEIGHT_MEASUREMENT:
            self._calculate_height()
            self._advance_step(CalibrationStep.ARM_SPAN_MEASUREMENT)

        elif self._current_step == CalibrationStep.ARM_SPAN_MEASUREMENT:
            self._calculate_arm_span()
            self._finish_calibration()

    def _advance_step(self, next_step: CalibrationStep) -> None:
        """Advance to the next calibration step.

        Args:
            next_step: Next step to perform
        """
        if self._on_step_complete and self._current_step:
            self._on_step_complete(self._current_step)

        self._current_step = next_step
        self._clear_samples()

    def _calculate_floor_level(self) -> None:
        """Calculate floor level from samples.

        Uses the lowest HMD position minus expected eye height
        to estimate floor level.
        """
        if not self._hmd_samples:
            return

        # Find average Y position (user standing normally)
        total_y = sum(s.y for s in self._hmd_samples)
        if not self._hmd_samples:
            return
        avg_y = total_y / len(self._hmd_samples)

        # Floor is approximately 1.6m below eye level for average height
        # Will be refined in height measurement step
        self._data.floor_level = avg_y - 1.6

    def _calculate_height(self) -> None:
        """Calculate player height from samples.

        Uses HMD position relative to floor level.
        """
        if not self._hmd_samples:
            return

        # Average HMD height
        total_y = sum(s.y for s in self._hmd_samples)
        avg_y = total_y / len(self._hmd_samples)

        # Eye height is HMD position relative to floor
        eye_height = avg_y - self._data.floor_level

        # Total height is approximately eye height / EYE_HEIGHT_RATIO
        EYE_HEIGHT_RATIO = 0.94  # Standard human proportion
        if abs(EYE_HEIGHT_RATIO) < 0.001:
            raise ValueError("Invalid eye height ratio")
        self._data.eye_height = eye_height
        self._data.height = eye_height / EYE_HEIGHT_RATIO

    def _calculate_arm_span(self) -> None:
        """Calculate arm span from T-pose samples.

        Measures distance between left and right hand positions.
        """
        if not self._left_hand_samples or not self._right_hand_samples:
            # No hand data - estimate from height
            self._data.arm_span = self._data.height
            return

        # Average hand positions
        left_avg = Vec3(
            sum(s.x for s in self._left_hand_samples) / len(self._left_hand_samples),
            sum(s.y for s in self._left_hand_samples) / len(self._left_hand_samples),
            sum(s.z for s in self._left_hand_samples) / len(self._left_hand_samples),
        )
        right_avg = Vec3(
            sum(s.x for s in self._right_hand_samples) / len(self._right_hand_samples),
            sum(s.y for s in self._right_hand_samples) / len(self._right_hand_samples),
            sum(s.z for s in self._right_hand_samples) / len(self._right_hand_samples),
        )

        # Arm span is horizontal distance between hands
        self._data.arm_span = left_avg.distance(right_avg)

    def _finish_calibration(self) -> None:
        """Finish the calibration process."""
        # Calculate derived proportions
        self._data.calculate_proportions()

        # Update state
        self._state = CalibrationState.COMPLETED
        self._current_step = None

        if self._on_step_complete:
            self._on_step_complete(CalibrationStep.ARM_SPAN_MEASUREMENT)

        if self._on_calibration_complete:
            self._on_calibration_complete(self._data)

    def quick_calibrate(
        self,
        hmd_position: Vec3,
        left_hand_position: Optional[Vec3] = None,
        right_hand_position: Optional[Vec3] = None,
        floor_level: Optional[float] = None,
    ) -> CalibrationData:
        """Perform quick single-sample calibration.

        Args:
            hmd_position: Current HMD position
            left_hand_position: Current left hand position
            right_hand_position: Current right hand position
            floor_level: Known floor level (if available)

        Returns:
            Calibration data
        """
        # Floor level
        if floor_level is not None:
            self._data.floor_level = floor_level
        else:
            # Estimate from HMD position
            self._data.floor_level = hmd_position.y - 1.6

        # Height from HMD
        self._data.eye_height = hmd_position.y - self._data.floor_level
        self._data.height = self._data.eye_height / 0.94

        # Arm span from hands or estimate from height
        if left_hand_position and right_hand_position:
            self._data.arm_span = left_hand_position.distance(right_hand_position)
        else:
            self._data.arm_span = self._data.height

        # Calculate proportions
        self._data.calculate_proportions()

        self._state = CalibrationState.COMPLETED
        return self._data

    def set_manual(
        self,
        height: float,
        arm_span: Optional[float] = None,
        floor_level: float = 0.0,
    ) -> CalibrationData:
        """Set calibration values manually.

        Args:
            height: Player height in meters
            arm_span: Player arm span (defaults to height)
            floor_level: Floor Y position

        Returns:
            Calibration data
        """
        if height <= 0:
            raise ValueError("Height must be positive")

        self._data.height = height
        self._data.arm_span = arm_span if arm_span is not None else height
        self._data.floor_level = floor_level
        self._data.calculate_proportions()

        self._state = CalibrationState.COMPLETED
        return self._data

    def reset(self) -> None:
        """Reset calibration to defaults."""
        self._state = CalibrationState.NOT_STARTED
        self._current_step = None
        self._data = CalibrationData()
        self._clear_samples()

    def get_instruction(self) -> str:
        """Get instruction text for current calibration step.

        Returns:
            Instruction string for user
        """
        if self._state == CalibrationState.NOT_STARTED:
            return "Press a button to start calibration."

        elif self._state == CalibrationState.COMPLETED:
            return f"Calibration complete. Height: {self._data.height:.2f}m"

        elif self._current_step == CalibrationStep.FLOOR_DETECTION:
            return "Stand naturally in a neutral position."

        elif self._current_step == CalibrationStep.HEIGHT_MEASUREMENT:
            return "Stand straight with your head in a neutral position."

        elif self._current_step == CalibrationStep.ARM_SPAN_MEASUREMENT:
            return "Extend your arms out to the sides in a T-pose."

        elif self._current_step == CalibrationStep.T_POSE:
            return "Hold the T-pose with arms extended horizontally."

        elif self._current_step == CalibrationStep.A_POSE:
            return "Stand with arms at a 45-degree angle from your body."

        else:
            return "Follow the on-screen instructions."

    def save(self) -> dict:
        """Save calibration data for persistence.

        Returns:
            Serializable dictionary
        """
        return {
            "version": 1,
            "data": self._data.to_dict(),
            "completed": self._state == CalibrationState.COMPLETED,
        }

    def load(self, saved_data: dict) -> bool:
        """Load calibration data from saved state.

        Args:
            saved_data: Previously saved calibration data

        Returns:
            True if loaded successfully
        """
        try:
            if saved_data is None or not isinstance(saved_data, dict):
                return False
            if saved_data.get("version") != 1:
                return False

            self._data = CalibrationData.from_dict(saved_data.get("data", {}))

            if saved_data.get("completed", False):
                self._state = CalibrationState.COMPLETED
            else:
                self._state = CalibrationState.NOT_STARTED

            return True
        except (KeyError, TypeError, ValueError):
            return False
