"""
Motion Matching Configuration - Centralized configuration constants.

This module provides centralized configuration for motion matching:
- Feature extraction weights and thresholds
- Search parameters
- Transition/inertialization parameters
- Contact detection thresholds
- Database quantization levels

All magic numbers should be defined here for easy tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================


@dataclass
class FeatureWeightConfig:
    """Weights for different feature types in cost computation.

    These weights control the relative importance of different
    feature types when computing match costs.
    """
    # Position features (bone positions relative to root)
    position_weight: float = 1.0

    # Velocity features (bone velocities)
    velocity_weight: float = 0.5

    # Trajectory features (future position/facing predictions)
    trajectory_weight: float = 1.0

    # Foot contact features
    contact_weight: float = 2.0

    # Per-bone weight overrides (bone_name -> weight)
    bone_weight_overrides: Dict[str, float] = field(default_factory=dict)

    # Per-trajectory-point weight overrides (time_offset -> weight)
    trajectory_point_overrides: Dict[float, float] = field(default_factory=dict)


# Default trajectory time points (seconds ahead)
DEFAULT_TRAJECTORY_TIMES: List[float] = [0.2, 0.4, 0.6]


# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================


@dataclass
class SearchParameterConfig:
    """Parameters controlling motion matching search behavior."""

    # Maximum number of results to return from search
    max_results: int = 5

    # Default cost threshold (infinity means no threshold)
    default_cost_threshold: float = float('inf')

    # KD-tree leaf size (smaller = more accurate, larger = faster build)
    kd_tree_leaf_size: int = 40

    # LSH parameters
    lsh_num_tables: int = 10
    lsh_num_hashes: int = 8
    lsh_bucket_width: float = 1.0

    # Minimum improvement ratio to trigger transition
    cost_improvement_threshold: float = 0.1

    # Epsilon for numerical stability in cost computation
    cost_epsilon: float = 1e-10


# =============================================================================
# TRANSITION/INERTIALIZATION CONFIGURATION
# =============================================================================


@dataclass
class TransitionParameterConfig:
    """Parameters controlling animation transitions."""

    # Default blend duration in seconds
    default_blend_duration: float = 0.15

    # Minimum blend duration (to prevent instant snapping)
    min_blend_duration: float = 0.001

    # Spring half-life for inertialization decay (seconds)
    spring_halflife: float = 0.1

    # Minimum spring half-life (to prevent numerical issues)
    min_spring_halflife: float = 1e-8

    # Threshold for considering inertialization complete
    position_threshold: float = 0.001
    rotation_threshold: float = 0.001


# =============================================================================
# CONTACT DETECTION CONFIGURATION
# =============================================================================


@dataclass
class ContactDetectionConfig:
    """Parameters for automatic foot contact detection."""

    # Maximum foot height above ground to be considered in contact (meters)
    height_threshold: float = 0.05

    # Maximum foot velocity to be considered in contact (m/s)
    velocity_threshold: float = 0.5

    # Default ground height (Y coordinate)
    ground_height: float = 0.0

    # Minimum frames for a contact event to be valid
    min_contact_frames: int = 3

    # Contact smoothing threshold
    contact_smoothing_threshold: float = 0.5


# =============================================================================
# IDLE DETECTION CONFIGURATION
# =============================================================================


@dataclass
class IdleDetectionConfig:
    """Parameters for idle state detection."""

    # Velocity below this is considered stationary (m/s)
    velocity_threshold: float = 0.1

    # Time velocity must be low before entering idle (seconds)
    hold_time: float = 0.1


# =============================================================================
# CONTROLLER TIMING CONFIGURATION
# =============================================================================


@dataclass
class ControllerTimingConfig:
    """Timing parameters for the motion matching controller."""

    # Minimum time in clip before searching for new match (seconds)
    min_time_in_clip: float = 0.1

    # Interval between database searches (seconds)
    search_interval: float = 0.05

    # Frame distance exclusion for search (frames before/after current)
    exclude_frames_before: int = 5
    exclude_frames_after: int = 10

    # Minimum remaining frames for transition candidate
    min_remaining_frames: int = 5


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================


@dataclass
class DatabaseConfig:
    """Configuration for motion database."""

    # Quantization scale for INT16 storage
    int16_quant_scale: float = 10000.0

    # Quantization scale divisor for INT8 storage
    int8_quant_divisor: float = 128.0

    # Epsilon for normalization (prevents division by zero)
    normalization_epsilon: float = 1e-8


# =============================================================================
# LOCOMOTION TAG SPEED THRESHOLDS
# =============================================================================


@dataclass
class LocomotionSpeedConfig:
    """Speed thresholds for auto-detecting locomotion tags."""

    # (min_speed, max_speed) for each locomotion type (m/s)
    idle_speed: tuple = (0.0, 0.1)
    walk_speed: tuple = (0.1, 2.0)
    run_speed: tuple = (2.0, 5.0)
    sprint_speed: tuple = (5.0, float('inf'))

    # Minimum frames for a locomotion region to be tagged
    min_region_frames: int = 10


# =============================================================================
# TURN DETECTION CONFIGURATION
# =============================================================================


@dataclass
class TurnDetectionConfig:
    """Parameters for automatic turn tag detection."""

    # Minimum angular velocity for turn detection (radians/s)
    turn_threshold: float = 0.5

    # Minimum frames for a turn to be tagged
    min_turn_frames: int = 10


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================


# Create default configuration instances
DEFAULT_FEATURE_WEIGHTS = FeatureWeightConfig()
DEFAULT_SEARCH_PARAMS = SearchParameterConfig()
DEFAULT_TRANSITION_PARAMS = TransitionParameterConfig()
DEFAULT_CONTACT_DETECTION = ContactDetectionConfig()
DEFAULT_IDLE_DETECTION = IdleDetectionConfig()
DEFAULT_CONTROLLER_TIMING = ControllerTimingConfig()
DEFAULT_DATABASE_CONFIG = DatabaseConfig()
DEFAULT_LOCOMOTION_SPEEDS = LocomotionSpeedConfig()
DEFAULT_TURN_DETECTION = TurnDetectionConfig()
