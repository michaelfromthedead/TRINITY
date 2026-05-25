"""
Facial Animation Configuration.

Centralizes all magic numbers and configurable parameters
for the facial animation system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# =============================================================================
# Blend Shape Configuration
# =============================================================================


@dataclass
class BlendShapeConfig:
    """Configuration for blend shape system."""

    # Weight limits
    weight_min: float = 0.0
    weight_max: float = 1.0
    weight_epsilon: float = 0.001  # Threshold for "active" detection

    # Transition settings
    default_transition_speed: float = 10.0  # Units per second
    transition_completion_threshold: float = 0.0001

    # Corrective settings
    corrective_weight_threshold: float = 0.001  # Min weight to apply corrective

    # Normalization
    normalize_total_weight: bool = False


# =============================================================================
# FACS Configuration
# =============================================================================


@dataclass
class FACSConfig:
    """Configuration for FACS system."""

    # Intensity limits
    intensity_min: float = 0.0
    intensity_max: float = 1.0
    intensity_threshold: float = 0.001  # Min intensity to consider active

    # Expression blending
    default_blend_speed: float = 5.0
    blend_completion_threshold: float = 0.0001

    # AU combination limits (prevent over-saturation)
    max_combined_au_weight: float = 1.0
    min_combined_au_weight: float = -1.0


# =============================================================================
# Lip Sync Configuration
# =============================================================================


@dataclass
class LipSyncConfig:
    """Configuration for lip synchronization."""

    # Viseme timing
    default_blend_time: float = 0.05  # Time to blend between visemes (seconds)
    min_viseme_duration: float = 0.01  # Minimum viseme duration (seconds)

    # Coarticulation
    default_anticipation_time: float = 0.05  # Anticipation window (seconds)
    default_carryover_time: float = 0.03  # Carryover window (seconds)

    # Intensity
    default_intensity: float = 1.0
    intensity_min: float = 0.0
    intensity_max: float = 1.0

    # Phoneme to viseme conversion
    default_phoneme_duration: float = 0.08  # Default phoneme length (seconds)
    silence_phoneme_scale: float = 0.5  # Scale factor for silence durations


# =============================================================================
# Eye Animation Configuration
# =============================================================================


@dataclass
class EyeConfig:
    """Configuration for eye animation."""

    # Rotation limits (degrees)
    max_yaw: float = 35.0
    max_pitch_up: float = 25.0
    max_pitch_down: float = 30.0
    max_vergence: float = 15.0

    # Eye separation
    default_eye_separation: float = 0.065  # ~6.5cm between eyes (meters)

    # Tracking
    default_smooth_speed: float = 10.0
    min_smooth_speed: float = 0.1
    tracking_target_threshold: float = 0.001  # Min distance to consider tracking

    # Vergence
    min_vergence_distance: float = 0.1  # Min distance for vergence calculation


@dataclass
class BlinkConfig:
    """Configuration for blink behavior."""

    # Timing (seconds)
    min_interval: float = 2.0
    max_interval: float = 6.0
    blink_duration: float = 0.15

    # Blink curve timing (fraction of total duration)
    close_phase_duration: float = 0.3  # 30% closing
    open_phase_duration: float = 0.7  # 70% opening

    # Blink types
    half_blink_chance: float = 0.1
    double_blink_chance: float = 0.15
    half_blink_min_intensity: float = 0.5
    half_blink_max_intensity: float = 0.8
    full_blink_min_intensity: float = 0.9
    full_blink_max_intensity: float = 1.0


@dataclass
class SaccadeConfig:
    """Configuration for saccadic eye movements."""

    # Micro-saccades
    micro_saccade_interval: float = 0.5  # Time between micro-saccades (seconds)
    micro_saccade_magnitude: float = 0.5  # Amplitude (degrees)
    micro_saccade_decay_rate: float = 5.0  # Decay multiplier per second

    # Full saccades
    saccade_speed: float = 500.0  # Speed (degrees/second) - very fast
    fixation_duration: float = 0.2  # Time to fixate before saccade (seconds)


@dataclass
class PupilConfig:
    """Configuration for pupil dilation."""

    # Size limits (normalized 0-1)
    base_size: float = 0.5
    min_size: float = 0.2
    max_size: float = 0.9

    # Response rates
    dilation_speed: float = 2.0  # Speed of size changes

    # Response sensitivity (0-1)
    light_response: float = 0.5
    emotional_response: float = 0.3


# =============================================================================
# Face Rig Configuration
# =============================================================================


@dataclass
class FaceRigConfig:
    """Configuration for face rig integration."""

    # Jaw control
    jaw_max_rotation: float = 25.0  # Maximum jaw rotation (degrees)

    # Emotion blending
    default_emotion_blend_time: float = 0.3  # Default transition time (seconds)
    emotion_blend_min_time: float = 0.001  # Minimum blend time (seconds)

    # Layer weights
    layer_weight_min: float = 0.0
    layer_weight_max: float = 1.0
    layer_activation_threshold: float = 0.001  # Min weight to apply layer

    # Final weight clamping
    final_weight_min: float = 0.0
    final_weight_max: float = 1.0


# =============================================================================
# Face Capture Configuration
# =============================================================================


@dataclass
class FaceCaptureConfig:
    """Configuration for face capture playback."""

    # Playback
    default_frame_rate: float = 30.0
    min_speed: float = -10.0  # Minimum playback speed (allows reverse)
    max_speed: float = 10.0  # Maximum playback speed

    # Interpolation
    default_interpolation_mode: str = "linear"

    # Blend times
    default_blend_in_time: float = 0.0
    default_blend_out_time: float = 0.0

    # Curve sampling
    min_keyframe_delta: float = 0.0001  # Minimum time between keyframes


# =============================================================================
# Global Configuration
# =============================================================================


@dataclass
class FacialAnimationConfig:
    """Master configuration for all facial animation systems."""

    blend_shapes: BlendShapeConfig = field(default_factory=BlendShapeConfig)
    facs: FACSConfig = field(default_factory=FACSConfig)
    lip_sync: LipSyncConfig = field(default_factory=LipSyncConfig)
    eye: EyeConfig = field(default_factory=EyeConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    saccade: SaccadeConfig = field(default_factory=SaccadeConfig)
    pupil: PupilConfig = field(default_factory=PupilConfig)
    face_rig: FaceRigConfig = field(default_factory=FaceRigConfig)
    face_capture: FaceCaptureConfig = field(default_factory=FaceCaptureConfig)


# Global default configuration instance
_default_config: Optional[FacialAnimationConfig] = None


def get_config() -> FacialAnimationConfig:
    """Get the global facial animation configuration."""
    global _default_config
    if _default_config is None:
        _default_config = FacialAnimationConfig()
    return _default_config


def set_config(config: FacialAnimationConfig) -> None:
    """Set the global facial animation configuration."""
    global _default_config
    _default_config = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _default_config
    _default_config = None
