"""Face and expression tracking for XR avatars.

Provides face tracking with blend shapes, eye tracking, and lip sync
for expressive avatar faces in social XR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.xr.config import XR_CONFIG


class BlendShapeType(Enum):
    """Standard blend shape types for facial animation.

    Based on ARKit/XR face tracking standards.
    """
    # Eyebrows
    BROW_DOWN_LEFT = auto()
    BROW_DOWN_RIGHT = auto()
    BROW_INNER_UP = auto()
    BROW_OUTER_UP_LEFT = auto()
    BROW_OUTER_UP_RIGHT = auto()

    # Eyes
    EYE_BLINK_LEFT = auto()
    EYE_BLINK_RIGHT = auto()
    EYE_LOOK_DOWN_LEFT = auto()
    EYE_LOOK_DOWN_RIGHT = auto()
    EYE_LOOK_IN_LEFT = auto()
    EYE_LOOK_IN_RIGHT = auto()
    EYE_LOOK_OUT_LEFT = auto()
    EYE_LOOK_OUT_RIGHT = auto()
    EYE_LOOK_UP_LEFT = auto()
    EYE_LOOK_UP_RIGHT = auto()
    EYE_SQUINT_LEFT = auto()
    EYE_SQUINT_RIGHT = auto()
    EYE_WIDE_LEFT = auto()
    EYE_WIDE_RIGHT = auto()

    # Jaw
    JAW_FORWARD = auto()
    JAW_LEFT = auto()
    JAW_OPEN = auto()
    JAW_RIGHT = auto()

    # Mouth
    MOUTH_CLOSE = auto()
    MOUTH_DIMPLE_LEFT = auto()
    MOUTH_DIMPLE_RIGHT = auto()
    MOUTH_FROWN_LEFT = auto()
    MOUTH_FROWN_RIGHT = auto()
    MOUTH_FUNNEL = auto()
    MOUTH_LEFT = auto()
    MOUTH_LOWER_DOWN_LEFT = auto()
    MOUTH_LOWER_DOWN_RIGHT = auto()
    MOUTH_PRESS_LEFT = auto()
    MOUTH_PRESS_RIGHT = auto()
    MOUTH_PUCKER = auto()
    MOUTH_RIGHT = auto()
    MOUTH_ROLL_LOWER = auto()
    MOUTH_ROLL_UPPER = auto()
    MOUTH_SHRUG_LOWER = auto()
    MOUTH_SHRUG_UPPER = auto()
    MOUTH_SMILE_LEFT = auto()
    MOUTH_SMILE_RIGHT = auto()
    MOUTH_STRETCH_LEFT = auto()
    MOUTH_STRETCH_RIGHT = auto()
    MOUTH_UPPER_UP_LEFT = auto()
    MOUTH_UPPER_UP_RIGHT = auto()

    # Nose
    NOSE_SNEER_LEFT = auto()
    NOSE_SNEER_RIGHT = auto()

    # Cheeks
    CHEEK_PUFF = auto()
    CHEEK_SQUINT_LEFT = auto()
    CHEEK_SQUINT_RIGHT = auto()

    # Tongue
    TONGUE_OUT = auto()


class ExpressionType(Enum):
    """Predefined facial expression types."""
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    SURPRISED = auto()
    DISGUSTED = auto()
    SCARED = auto()
    THINKING = auto()


class FaceDrivingMode(Enum):
    """Face animation driving modes."""
    BLEND_SHAPES = auto()  # Direct blend shape control
    BONE_DRIVEN = auto()   # Bone-based facial rig
    ML_DRIVEN = auto()     # Machine learning based


@dataclass(slots=True)
class EyeGazeData:
    """Eye gaze tracking data.

    Attributes:
        gaze_origin: Origin point of gaze ray
        gaze_direction: Direction of gaze
        left_openness: Left eye openness (0=closed, 1=open)
        right_openness: Right eye openness
        left_pupil_diameter: Left pupil diameter in mm
        right_pupil_diameter: Right pupil diameter in mm
        is_fixating: Whether eyes are fixating on a point
        fixation_point: World position of fixation point
        confidence: Tracking confidence (0-1)
    """
    gaze_origin: Vec3 = field(default_factory=Vec3.zero)
    gaze_direction: Vec3 = field(default_factory=lambda: Vec3(0, 0, -1))
    left_openness: float = 1.0
    right_openness: float = 1.0
    left_pupil_diameter: float = XR_CONFIG.runtime.DEFAULT_PUPIL_DIAMETER_MM  # mm
    right_pupil_diameter: float = XR_CONFIG.runtime.DEFAULT_PUPIL_DIAMETER_MM
    is_fixating: bool = False
    fixation_point: Vec3 = field(default_factory=Vec3.zero)
    confidence: float = 0.0

    def get_average_openness(self) -> float:
        """Get average eye openness."""
        return (self.left_openness + self.right_openness) / 2.0

    def is_blinking(self, threshold: float = XR_CONFIG.runtime.BLINK_THRESHOLD) -> bool:
        """Check if either eye is blinking.

        Args:
            threshold: Openness threshold for blink detection

        Returns:
            True if blinking
        """
        return self.left_openness < threshold or self.right_openness < threshold


@dataclass(slots=True)
class LipSyncData:
    """Lip sync viseme data.

    Attributes:
        current_viseme: Current viseme index
        viseme_weights: Weights for all visemes
        audio_amplitude: Current audio amplitude
        is_speaking: Whether the user is speaking
    """
    current_viseme: int = 0
    viseme_weights: list[float] = field(default_factory=lambda: [0.0] * 15)
    audio_amplitude: float = 0.0
    is_speaking: bool = False

    # Standard viseme indices
    VISEME_SILENCE = 0
    VISEME_AA = 1  # as in "father"
    VISEME_AH = 2  # as in "but"
    VISEME_AO = 3  # as in "dog"
    VISEME_AW = 4  # as in "now"
    VISEME_CH = 5  # as in "chair"
    VISEME_EE = 6  # as in "see"
    VISEME_EH = 7  # as in "bed"
    VISEME_ER = 8  # as in "bird"
    VISEME_IH = 9  # as in "sit"
    VISEME_OO = 10 # as in "too"
    VISEME_OH = 11 # as in "go"
    VISEME_PP = 12 # as in "pop"
    VISEME_TH = 13 # as in "think"
    VISEME_WW = 14 # as in "win"


class BlendShapeController:
    """Controller for facial blend shapes.

    Manages blend shape weights and provides interpolation
    for smooth facial animation.
    """
    __slots__ = ('_weights', '_target_weights', '_blend_speed')

    def __init__(self, blend_speed: float = XR_CONFIG.runtime.EYE_BLEND_SPEED):
        """Initialize blend shape controller.

        Args:
            blend_speed: Interpolation speed (weights per second)
        """
        if blend_speed <= 0:
            raise ValueError("blend_speed must be positive")

        self._weights: dict[BlendShapeType, float] = {
            shape: 0.0 for shape in BlendShapeType
        }
        self._target_weights: dict[BlendShapeType, float] = {
            shape: 0.0 for shape in BlendShapeType
        }
        self._blend_speed = blend_speed

    def get_weight(self, shape: BlendShapeType) -> float:
        """Get current weight for a blend shape.

        Args:
            shape: Blend shape type

        Returns:
            Current weight (0-1)
        """
        return self._weights.get(shape, 0.0)

    def set_weight(self, shape: BlendShapeType, weight: float) -> None:
        """Set target weight for a blend shape.

        Args:
            shape: Blend shape type
            weight: Target weight (0-1)
        """
        self._target_weights[shape] = max(0.0, min(1.0, weight))

    def set_weights(self, weights: dict[BlendShapeType, float]) -> None:
        """Set multiple target weights.

        Args:
            weights: Dictionary of blend shape weights
        """
        for shape, weight in weights.items():
            self._target_weights[shape] = max(0.0, min(1.0, weight))

    def reset(self) -> None:
        """Reset all weights to zero."""
        for shape in BlendShapeType:
            self._weights[shape] = 0.0
            self._target_weights[shape] = 0.0

    def update(self, delta_time: float) -> None:
        """Update weight interpolation.

        Args:
            delta_time: Time since last update in seconds
        """
        if delta_time <= 0:
            return

        t = min(1.0, delta_time * self._blend_speed)

        for shape in BlendShapeType:
            current = self._weights[shape]
            target = self._target_weights[shape]
            self._weights[shape] = current + (target - current) * t

    def snap_to_target(self) -> None:
        """Immediately set current weights to target weights."""
        for shape in BlendShapeType:
            self._weights[shape] = self._target_weights[shape]

    def get_all_weights(self) -> dict[BlendShapeType, float]:
        """Get all current weights.

        Returns:
            Dictionary of all blend shape weights
        """
        return self._weights.copy()

    def apply_expression(self, expression: ExpressionType) -> None:
        """Apply a predefined expression.

        Args:
            expression: Expression type to apply
        """
        # Reset all weights
        for shape in BlendShapeType:
            self._target_weights[shape] = 0.0

        # Apply expression-specific weights
        if expression == ExpressionType.HAPPY:
            self._target_weights[BlendShapeType.MOUTH_SMILE_LEFT] = 0.7
            self._target_weights[BlendShapeType.MOUTH_SMILE_RIGHT] = 0.7
            self._target_weights[BlendShapeType.CHEEK_SQUINT_LEFT] = 0.3
            self._target_weights[BlendShapeType.CHEEK_SQUINT_RIGHT] = 0.3

        elif expression == ExpressionType.SAD:
            self._target_weights[BlendShapeType.MOUTH_FROWN_LEFT] = 0.6
            self._target_weights[BlendShapeType.MOUTH_FROWN_RIGHT] = 0.6
            self._target_weights[BlendShapeType.BROW_INNER_UP] = 0.4

        elif expression == ExpressionType.ANGRY:
            self._target_weights[BlendShapeType.BROW_DOWN_LEFT] = 0.7
            self._target_weights[BlendShapeType.BROW_DOWN_RIGHT] = 0.7
            self._target_weights[BlendShapeType.MOUTH_PRESS_LEFT] = 0.4
            self._target_weights[BlendShapeType.MOUTH_PRESS_RIGHT] = 0.4
            self._target_weights[BlendShapeType.NOSE_SNEER_LEFT] = 0.3
            self._target_weights[BlendShapeType.NOSE_SNEER_RIGHT] = 0.3

        elif expression == ExpressionType.SURPRISED:
            self._target_weights[BlendShapeType.EYE_WIDE_LEFT] = 0.8
            self._target_weights[BlendShapeType.EYE_WIDE_RIGHT] = 0.8
            self._target_weights[BlendShapeType.BROW_OUTER_UP_LEFT] = 0.6
            self._target_weights[BlendShapeType.BROW_OUTER_UP_RIGHT] = 0.6
            self._target_weights[BlendShapeType.JAW_OPEN] = 0.5

        elif expression == ExpressionType.DISGUSTED:
            self._target_weights[BlendShapeType.NOSE_SNEER_LEFT] = 0.6
            self._target_weights[BlendShapeType.NOSE_SNEER_RIGHT] = 0.6
            self._target_weights[BlendShapeType.MOUTH_UPPER_UP_LEFT] = 0.3
            self._target_weights[BlendShapeType.MOUTH_UPPER_UP_RIGHT] = 0.3

        elif expression == ExpressionType.SCARED:
            self._target_weights[BlendShapeType.EYE_WIDE_LEFT] = 0.9
            self._target_weights[BlendShapeType.EYE_WIDE_RIGHT] = 0.9
            self._target_weights[BlendShapeType.BROW_INNER_UP] = 0.7
            self._target_weights[BlendShapeType.MOUTH_STRETCH_LEFT] = 0.4
            self._target_weights[BlendShapeType.MOUTH_STRETCH_RIGHT] = 0.4

        elif expression == ExpressionType.THINKING:
            self._target_weights[BlendShapeType.BROW_INNER_UP] = 0.3
            self._target_weights[BlendShapeType.EYE_LOOK_UP_LEFT] = 0.4
            self._target_weights[BlendShapeType.EYE_LOOK_UP_RIGHT] = 0.4
            self._target_weights[BlendShapeType.MOUTH_PUCKER] = 0.2


class FaceTracking:
    """Face tracking system for XR avatars.

    Combines eye tracking, lip sync, and expression blend shapes
    for expressive avatar faces.
    """
    __slots__ = (
        '_blend_shapes', '_eye_gaze', '_lip_sync',
        '_driving_mode', '_is_calibrated', '_blend_speed',
        '_auto_blink_enabled', '_auto_blink_timer',
        '_lip_sync_enabled', '_expression_detection_enabled'
    )

    def __init__(
        self,
        driving_mode: FaceDrivingMode = FaceDrivingMode.BLEND_SHAPES,
        blend_speed: float = XR_CONFIG.runtime.EYE_BLEND_SPEED,
    ) -> None:
        """Initialize face tracking.

        Args:
            driving_mode: Face animation driving mode
            blend_speed: Blend shape interpolation speed
        """
        self._blend_shapes = BlendShapeController(blend_speed)
        self._eye_gaze = EyeGazeData()
        self._lip_sync = LipSyncData()
        self._driving_mode = driving_mode
        self._is_calibrated = False
        self._blend_speed = blend_speed

        # Auto-blink for when eye tracking unavailable
        self._auto_blink_enabled = True
        self._auto_blink_timer = 0.0

        # Feature toggles
        self._lip_sync_enabled = True
        self._expression_detection_enabled = True

    @property
    def driving_mode(self) -> FaceDrivingMode:
        """Get face driving mode."""
        return self._driving_mode

    @property
    def eye_gaze(self) -> EyeGazeData:
        """Get current eye gaze data."""
        return self._eye_gaze

    @property
    def lip_sync(self) -> LipSyncData:
        """Get current lip sync data."""
        return self._lip_sync

    @property
    def is_calibrated(self) -> bool:
        """Check if face tracking is calibrated."""
        return self._is_calibrated

    def calibrate(self) -> bool:
        """Run face tracking calibration.

        Returns:
            True if calibration succeeded
        """
        # Reset blend shapes to neutral
        self._blend_shapes.reset()
        self._is_calibrated = True
        return True

    def update_eye_tracking(
        self,
        gaze_origin: Vec3,
        gaze_direction: Vec3,
        left_openness: float,
        right_openness: float,
        confidence: float = 1.0,
    ) -> None:
        """Update eye tracking data.

        Args:
            gaze_origin: Gaze ray origin
            gaze_direction: Gaze direction vector
            left_openness: Left eye openness (0-1)
            right_openness: Right eye openness (0-1)
            confidence: Tracking confidence
        """
        self._eye_gaze.gaze_origin = gaze_origin
        self._eye_gaze.gaze_direction = gaze_direction.normalized()
        self._eye_gaze.left_openness = max(0.0, min(1.0, left_openness))
        self._eye_gaze.right_openness = max(0.0, min(1.0, right_openness))
        self._eye_gaze.confidence = max(0.0, min(1.0, confidence))

        # Update eye blend shapes
        blink_left = 1.0 - self._eye_gaze.left_openness
        blink_right = 1.0 - self._eye_gaze.right_openness

        self._blend_shapes.set_weight(BlendShapeType.EYE_BLINK_LEFT, blink_left)
        self._blend_shapes.set_weight(BlendShapeType.EYE_BLINK_RIGHT, blink_right)

        # Convert gaze direction to look blend shapes
        # Assuming gaze_direction is in local face space
        look_right = max(0.0, gaze_direction.x)
        look_left = max(0.0, -gaze_direction.x)
        look_up = max(0.0, gaze_direction.y)
        look_down = max(0.0, -gaze_direction.y)

        self._blend_shapes.set_weight(BlendShapeType.EYE_LOOK_OUT_LEFT, look_left)
        self._blend_shapes.set_weight(BlendShapeType.EYE_LOOK_IN_LEFT, look_right)
        self._blend_shapes.set_weight(BlendShapeType.EYE_LOOK_OUT_RIGHT, look_right)
        self._blend_shapes.set_weight(BlendShapeType.EYE_LOOK_IN_RIGHT, look_left)
        self._blend_shapes.set_weight(BlendShapeType.EYE_LOOK_UP_LEFT, look_up)
        self._blend_shapes.set_weight(BlendShapeType.EYE_LOOK_UP_RIGHT, look_up)
        self._blend_shapes.set_weight(BlendShapeType.EYE_LOOK_DOWN_LEFT, look_down)
        self._blend_shapes.set_weight(BlendShapeType.EYE_LOOK_DOWN_RIGHT, look_down)

    def update_lip_sync(
        self,
        viseme_weights: list[float],
        audio_amplitude: float = 0.0,
    ) -> None:
        """Update lip sync from viseme data.

        Args:
            viseme_weights: Weights for each viseme
            audio_amplitude: Current audio amplitude
        """
        if not self._lip_sync_enabled:
            return

        self._lip_sync.viseme_weights = viseme_weights[:15]
        self._lip_sync.audio_amplitude = max(0.0, min(1.0, audio_amplitude))
        self._lip_sync.is_speaking = audio_amplitude > 0.1

        # Find dominant viseme
        if viseme_weights:
            max_weight = 0.0
            max_idx = 0
            for i, w in enumerate(viseme_weights):
                if w > max_weight:
                    max_weight = w
                    max_idx = i
            self._lip_sync.current_viseme = max_idx

        # Map visemes to blend shapes
        self._apply_viseme_to_blend_shapes(viseme_weights)

    def _apply_viseme_to_blend_shapes(self, viseme_weights: list[float]) -> None:
        """Map viseme weights to mouth blend shapes.

        Args:
            viseme_weights: Weights for each viseme
        """
        if len(viseme_weights) < 15:
            return

        # Reset mouth shapes
        mouth_shapes = [
            BlendShapeType.JAW_OPEN,
            BlendShapeType.MOUTH_FUNNEL,
            BlendShapeType.MOUTH_PUCKER,
            BlendShapeType.MOUTH_SMILE_LEFT,
            BlendShapeType.MOUTH_SMILE_RIGHT,
        ]
        for shape in mouth_shapes:
            self._blend_shapes.set_weight(shape, 0.0)

        # Map each viseme to blend shapes
        # AA - wide open mouth
        if viseme_weights[LipSyncData.VISEME_AA] > 0:
            w = viseme_weights[LipSyncData.VISEME_AA]
            self._blend_shapes.set_weight(BlendShapeType.JAW_OPEN, w * 0.7)

        # OO - pursed lips
        if viseme_weights[LipSyncData.VISEME_OO] > 0:
            w = viseme_weights[LipSyncData.VISEME_OO]
            self._blend_shapes.set_weight(BlendShapeType.MOUTH_PUCKER, w * 0.8)

        # EE - wide smile
        if viseme_weights[LipSyncData.VISEME_EE] > 0:
            w = viseme_weights[LipSyncData.VISEME_EE]
            self._blend_shapes.set_weight(BlendShapeType.MOUTH_SMILE_LEFT, w * 0.5)
            self._blend_shapes.set_weight(BlendShapeType.MOUTH_SMILE_RIGHT, w * 0.5)

        # PP/WW - closed lips
        if viseme_weights[LipSyncData.VISEME_PP] > 0:
            w = viseme_weights[LipSyncData.VISEME_PP]
            self._blend_shapes.set_weight(BlendShapeType.MOUTH_CLOSE, w * 0.9)

    def update_expression(
        self,
        blend_shape_weights: dict[BlendShapeType, float],
    ) -> None:
        """Update from tracked expression blend shapes.

        Args:
            blend_shape_weights: Tracked blend shape weights
        """
        if not self._expression_detection_enabled:
            return

        for shape, weight in blend_shape_weights.items():
            # Don't override eye/mouth shapes if eye/lip tracking is active
            if shape in (
                BlendShapeType.EYE_BLINK_LEFT,
                BlendShapeType.EYE_BLINK_RIGHT,
            ) and self._eye_gaze.confidence > 0.5:
                continue

            self._blend_shapes.set_weight(shape, weight)

    def set_expression(self, expression: ExpressionType) -> None:
        """Set a predefined expression.

        Args:
            expression: Expression type to apply
        """
        self._blend_shapes.apply_expression(expression)

    def update(self, delta_time: float) -> None:
        """Update face tracking state.

        Args:
            delta_time: Time since last update in seconds
        """
        # Update blend shape interpolation
        self._blend_shapes.update(delta_time)

        # Auto-blink if eye tracking not available
        if self._auto_blink_enabled and self._eye_gaze.confidence < 0.3:
            self._update_auto_blink(delta_time)

    def _update_auto_blink(self, delta_time: float) -> None:
        """Update automatic blinking.

        Args:
            delta_time: Time since last update
        """
        import random

        self._auto_blink_timer += delta_time

        # Blink approximately every 3-5 seconds
        if self._auto_blink_timer > XR_CONFIG.runtime.AUTO_BLINK_MIN_INTERVAL_SECONDS + random.random() * XR_CONFIG.runtime.AUTO_BLINK_RANDOM_RANGE_SECONDS:
            # Trigger blink
            self._blend_shapes.set_weight(BlendShapeType.EYE_BLINK_LEFT, 1.0)
            self._blend_shapes.set_weight(BlendShapeType.EYE_BLINK_RIGHT, 1.0)
            self._auto_blink_timer = -XR_CONFIG.runtime.BLINK_DURATION_SECONDS  # Blink duration
        elif self._auto_blink_timer > 0:
            # End blink
            self._blend_shapes.set_weight(BlendShapeType.EYE_BLINK_LEFT, 0.0)
            self._blend_shapes.set_weight(BlendShapeType.EYE_BLINK_RIGHT, 0.0)

    def get_blend_shape_weight(self, shape: BlendShapeType) -> float:
        """Get current weight for a blend shape.

        Args:
            shape: Blend shape type

        Returns:
            Current weight (0-1)
        """
        return self._blend_shapes.get_weight(shape)

    def get_all_blend_shapes(self) -> dict[BlendShapeType, float]:
        """Get all current blend shape weights.

        Returns:
            Dictionary of blend shape weights
        """
        return self._blend_shapes.get_all_weights()

    def get_network_state(self) -> dict:
        """Get state for network synchronization.

        Returns:
            Dictionary of networked state
        """
        # Only send non-zero blend shapes to reduce bandwidth
        non_zero_shapes = {
            shape.name: weight
            for shape, weight in self._blend_shapes.get_all_weights().items()
            if weight > 0.01
        }

        return {
            "blend_shapes": non_zero_shapes,
            "gaze_direction": (
                self._eye_gaze.gaze_direction.x,
                self._eye_gaze.gaze_direction.y,
                self._eye_gaze.gaze_direction.z,
            ),
            "is_speaking": self._lip_sync.is_speaking,
        }

    def apply_network_state(self, state: dict) -> None:
        """Apply state from network synchronization.

        Args:
            state: Dictionary of networked state
        """
        if "blend_shapes" in state:
            for shape_name, weight in state["blend_shapes"].items():
                try:
                    shape = BlendShapeType[shape_name]
                    self._blend_shapes.set_weight(shape, weight)
                except KeyError:
                    pass

        if "gaze_direction" in state:
            gd = state["gaze_direction"]
            self._eye_gaze.gaze_direction = Vec3(gd[0], gd[1], gd[2])

        if "is_speaking" in state:
            self._lip_sync.is_speaking = state["is_speaking"]
