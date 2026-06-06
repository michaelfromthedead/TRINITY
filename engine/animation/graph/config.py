"""
Animation graph configuration constants.

Provides centralized configuration for animation system parameters.
All magic numbers should be defined here for easy tuning and maintenance.

Environment Variable Overrides:
    Set environment variables to override defaults for testing:
    - TRINITY_ANIM_MAX_EVAL_DEPTH: Override MAX_EVALUATION_DEPTH
    - TRINITY_ANIM_CYCLE_DETECTION: Override CYCLE_DETECTION_ENABLED (0/1)
    - TRINITY_ANIM_DEFAULT_BLEND_TIME: Override DEFAULT_BLEND_TIME (seconds)
    - TRINITY_ANIM_DEFAULT_TIME_SCALE: Override DEFAULT_TIME_SCALE
    - TRINITY_ANIM_SLERP_THRESHOLD: Override SLERP_DOT_THRESHOLD

Units Convention:
    - Time values: seconds (float)
    - Angles: radians (float)
    - Weights: normalized 0.0-1.0 (float)
    - Thresholds: dimensionless (float)
    - Depths/counts: integer
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import os


def _env_float(name: str, default: float) -> float:
    """Get float from environment variable or return default."""
    val = os.environ.get(name)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _env_int(name: str, default: int) -> int:
    """Get int from environment variable or return default."""
    val = os.environ.get(name)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _env_bool(name: str, default: bool) -> bool:
    """Get bool from environment variable or return default."""
    val = os.environ.get(name)
    if val is not None:
        return val.lower() in ("1", "true", "yes", "on")
    return default


@dataclass
class TransitionConfig:
    """Configuration for state machine transitions.

    Attributes:
        DEFAULT_TRANSITION_DURATION: Default blend duration for state
            transitions. Unit: seconds.
        FORCED_TRANSITION_DURATION: Blend duration when a transition is
            forced (e.g., interrupt). Unit: seconds.
        ANY_STATE_PRIORITY: Priority level for any-state transitions.
            Higher values take precedence. Unit: dimensionless integer.
    """

    # Default transition blend duration in seconds
    DEFAULT_TRANSITION_DURATION: float = 0.2

    # Transition blend duration for forced transitions (seconds)
    FORCED_TRANSITION_DURATION: float = 0.2

    # High priority for any-state transitions (dimensionless)
    ANY_STATE_PRIORITY: int = 100


@dataclass
class BlendTreeConfig:
    """Configuration for blend trees.

    Attributes:
        DEFAULT_GRADIENT_BAND_WIDTH: Width of the gradient band for smooth
            blending in 1D blend trees. Larger values create smoother
            transitions between clips. Unit: parameter space units (typically
            normalized 0-1 or physical units like m/s for locomotion).
        INVERSE_DISTANCE_POWER: Exponent for inverse distance weighting in
            2D blend trees. Higher values create sharper transitions.
            Unit: dimensionless.
        DISTANCE_EPSILON: Epsilon threshold for distance comparisons.
            Samples closer than this are considered coincident.
            Unit: parameter space units.
    """

    # Gradient band width for smooth blending in 1D blend trees
    DEFAULT_GRADIENT_BAND_WIDTH: float = 0.1

    # Power for inverse distance weighting in 2D blend trees (dimensionless)
    INVERSE_DISTANCE_POWER: float = 2.0

    # Epsilon for distance comparisons (very close to sample)
    DISTANCE_EPSILON: float = 1e-10


@dataclass
class SyncConfig:
    """Configuration for animation synchronization.

    Attributes:
        SYNC_TOLERANCE: Tolerance for comparing normalized times during
            synchronization. Times within this tolerance are considered
            equal. Unit: normalized time (0.0-1.0).
        EVENT_DEDUP_PRECISION: Number of decimal places for rounding
            event times during deduplication. Unit: decimal places.
    """

    # Tolerance for comparing normalized times (sync precision, normalized 0-1)
    SYNC_TOLERANCE: float = 0.01

    # Tolerance for event deduplication (decimal places)
    EVENT_DEDUP_PRECISION: int = 2


@dataclass
class LayerConfig:
    """Configuration for animation layers.

    Attributes:
        DEFAULT_LAYER_WEIGHT: Default weight applied to animation layers.
            Unit: normalized weight (0.0-1.0).
    """

    # Default layer weight (normalized 0.0-1.0)
    DEFAULT_LAYER_WEIGHT: float = 1.0


@dataclass
class QuaternionConfig:
    """Configuration for quaternion operations.

    Attributes:
        SLERP_DOT_THRESHOLD: When the dot product of two quaternions exceeds
            this threshold, slerp falls back to linear interpolation (nlerp)
            for numerical stability. Unit: dimensionless (cos of angle).
        SLERP_MIN_SIN_THETA: Minimum sin(theta) for slerp denominator to
            avoid division by near-zero. Unit: dimensionless.
        NORMALIZATION_EPSILON: Epsilon for quaternion length checks during
            normalization. Quaternions with squared length below this are
            treated as zero-length and returned as identity.
            Unit: dimensionless (squared length).
        EPSILON: General-purpose epsilon for quaternion numerical stability
            in comparisons and equality checks. Unit: dimensionless.

    Environment Overrides:
        TRINITY_ANIM_SLERP_THRESHOLD: Override SLERP_DOT_THRESHOLD
    """

    # Threshold for slerp to switch to linear interpolation (dimensionless)
    # Higher values = more aggressive fallback to nlerp
    SLERP_DOT_THRESHOLD: float = field(
        default_factory=lambda: _env_float("TRINITY_ANIM_SLERP_THRESHOLD", 0.9995)
    )

    # Minimum sin theta for slerp denominator (dimensionless)
    SLERP_MIN_SIN_THETA: float = 0.0001

    # Epsilon for quaternion length checks during normalization (squared length)
    # A quaternion whose squared length is below this threshold
    # is treated as zero-length and returned as identity.
    NORMALIZATION_EPSILON: float = 1e-6

    # General epsilon for numerical stability in quaternion operations
    EPSILON: float = 1e-7


@dataclass
class GraphConfig:
    """Configuration for the AnimationGraph DAG container.

    Attributes:
        MAX_EVALUATION_DEPTH: Maximum allowed recursion depth during graph
            evaluation to prevent stack overflow from undetected cycles.
            Unit: stack frames (integer).
        CYCLE_DETECTION_ENABLED: When True, the graph checks for cycles
            using three-colour DFS before each evaluate() call. Can be
            disabled in hot loops where topology is known to be acyclic.
            Unit: boolean flag.
        DEFAULT_TIME_SCALE: Default playback speed multiplier for the
            animation graph. 1.0 = normal speed, 0.5 = half speed, etc.
            Unit: dimensionless multiplier.

    Environment Overrides:
        TRINITY_ANIM_MAX_EVAL_DEPTH: Override MAX_EVALUATION_DEPTH
        TRINITY_ANIM_CYCLE_DETECTION: Override CYCLE_DETECTION_ENABLED (0/1)
        TRINITY_ANIM_DEFAULT_TIME_SCALE: Override DEFAULT_TIME_SCALE
    """

    # Maximum allowed evaluation depth to prevent runaway recursion (integer)
    # when a cycle somehow slips past detection.
    MAX_EVALUATION_DEPTH: int = field(
        default_factory=lambda: _env_int("TRINITY_ANIM_MAX_EVAL_DEPTH", 100)
    )

    # When True, the graph checks for cycles using three-colour DFS
    # before each evaluate() call.  Can be disabled in hot loops where
    # the topology is known to be static and acyclic.
    CYCLE_DETECTION_ENABLED: bool = field(
        default_factory=lambda: _env_bool("TRINITY_ANIM_CYCLE_DETECTION", True)
    )

    # Default playback speed multiplier (dimensionless)
    # 1.0 = normal, 0.5 = half speed, 2.0 = double speed
    DEFAULT_TIME_SCALE: float = field(
        default_factory=lambda: _env_float("TRINITY_ANIM_DEFAULT_TIME_SCALE", 1.0)
    )


@dataclass
class BlendConfig:
    """Configuration for pose and animation blending.

    Attributes:
        DEFAULT_BLEND_TIME: Default duration for blend operations when
            no explicit duration is specified. Unit: seconds.
        MIN_BLEND_WEIGHT: Minimum valid blend weight. Weights below this
            are clamped to zero. Unit: normalized weight (0.0-1.0).
        MAX_BLEND_WEIGHT: Maximum valid blend weight. Weights above this
            are clamped. Unit: normalized weight (0.0-1.0).
        WEIGHT_EPSILON: Weights below this threshold are treated as zero
            to avoid unnecessary computation. Unit: normalized weight.
        NORMALIZE_WEIGHTS: When True, per-bone blend weights are normalized
            so that the sum equals 1.0 before blending.

    Environment Overrides:
        TRINITY_ANIM_DEFAULT_BLEND_TIME: Override DEFAULT_BLEND_TIME
    """

    # Default blend duration when not specified (seconds)
    DEFAULT_BLEND_TIME: float = field(
        default_factory=lambda: _env_float("TRINITY_ANIM_DEFAULT_BLEND_TIME", 0.25)
    )

    # Minimum blend weight (normalized 0.0-1.0)
    # Weights below this are clamped to 0
    MIN_BLEND_WEIGHT: float = 0.0

    # Maximum blend weight (normalized 0.0-1.0)
    # Weights above this are clamped to 1
    MAX_BLEND_WEIGHT: float = 1.0

    # Minimum blend weight epsilon.  Weights below this threshold are
    # treated as zero to avoid unnecessary computation and
    # near-zero-weight artefacts. Unit: normalized weight.
    WEIGHT_EPSILON: float = 0.001

    # When True, per-bone blend weights are normalised so that
    # the sum of all active weights equals 1.0 before blending.
    # Disable when the caller already provides normalised weights.
    NORMALIZE_WEIGHTS: bool = True


@dataclass
class AnimationGraphConfig:
    """Master configuration for animation graph system."""

    transition: TransitionConfig = field(default_factory=TransitionConfig)
    blend_tree: BlendTreeConfig = field(default_factory=BlendTreeConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    layer: LayerConfig = field(default_factory=LayerConfig)
    quaternion: QuaternionConfig = field(default_factory=QuaternionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)


# Global default configuration instance
DEFAULT_CONFIG = AnimationGraphConfig()


def get_config() -> AnimationGraphConfig:
    """Get the default animation graph configuration."""
    return DEFAULT_CONFIG


def reset_config() -> None:
    """Reset global config to defaults (useful for testing).

    This re-reads environment variables to create a fresh config instance.
    """
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = AnimationGraphConfig()


__all__ = [
    "TransitionConfig",
    "BlendTreeConfig",
    "SyncConfig",
    "LayerConfig",
    "QuaternionConfig",
    "GraphConfig",
    "BlendConfig",
    "AnimationGraphConfig",
    "DEFAULT_CONFIG",
    "get_config",
    "reset_config",
]
