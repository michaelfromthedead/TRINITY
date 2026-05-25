"""
Animation graph configuration constants.

Provides centralized configuration for animation system parameters.
All magic numbers should be defined here for easy tuning and maintenance.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TransitionConfig:
    """Configuration for state machine transitions."""

    # Default transition blend duration in seconds
    DEFAULT_TRANSITION_DURATION: float = 0.2

    # Transition blend duration for forced transitions
    FORCED_TRANSITION_DURATION: float = 0.2

    # High priority for any-state transitions
    ANY_STATE_PRIORITY: int = 100


@dataclass
class BlendTreeConfig:
    """Configuration for blend trees."""

    # Gradient band width for smooth blending in 1D blend trees
    DEFAULT_GRADIENT_BAND_WIDTH: float = 0.1

    # Power for inverse distance weighting in 2D blend trees
    INVERSE_DISTANCE_POWER: float = 2.0

    # Epsilon for distance comparisons (very close to sample)
    DISTANCE_EPSILON: float = 1e-10


@dataclass
class SyncConfig:
    """Configuration for animation synchronization."""

    # Tolerance for comparing normalized times (sync precision)
    SYNC_TOLERANCE: float = 0.01

    # Tolerance for event deduplication (rounded to 2 decimal places)
    EVENT_DEDUP_PRECISION: int = 2


@dataclass
class LayerConfig:
    """Configuration for animation layers."""

    # Default layer weight
    DEFAULT_LAYER_WEIGHT: float = 1.0


@dataclass
class QuaternionConfig:
    """Configuration for quaternion operations."""

    # Threshold for slerp to switch to linear interpolation
    SLERP_DOT_THRESHOLD: float = 0.9995

    # Minimum sin theta for slerp denominator
    SLERP_MIN_SIN_THETA: float = 0.0001

    # Epsilon for quaternion length checks during normalization.
    # A quaternion whose squared length is below this threshold
    # is treated as zero-length and returned as identity.
    NORMALIZATION_EPSILON: float = 1e-6


@dataclass
class GraphConfig:
    """Configuration for the AnimationGraph DAG container."""

    # Maximum allowed evaluation depth to prevent runaway recursion
    # when a cycle somehow slips past detection.
    MAX_EVALUATION_DEPTH: int = 100

    # When True, the graph checks for cycles using three-colour DFS
    # before each evaluate() call.  Can be disabled in hot loops where
    # the topology is known to be static and acyclic.
    CYCLE_DETECTION_ENABLED: bool = True


@dataclass
class BlendConfig:
    """Configuration for pose and animation blending."""

    # Minimum blend weight.  Weights below this threshold are
    # treated as zero to avoid unnecessary computation and
    # near-zero-weight artefacts.
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
]
