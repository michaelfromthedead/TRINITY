"""
HLOD System Constants.

Centralized constants for the HLOD (Hierarchical Level of Detail) system.
This module consolidates magic numbers from generator, layers, and transitions modules.

References:
- WORLD_CONTEXT.md Section 7 HLOD System
"""

from __future__ import annotations


# =============================================================================
# FLOATING POINT CONSTANTS
# =============================================================================


class FloatingPointConstants:
    """Floating point comparison and precision constants."""
    # General epsilon for floating point comparisons
    EPSILON: float = 1e-8

    # Larger epsilon for transition calculations (more forgiving)
    TRANSITION_EPSILON: float = 1e-6

    # Hash rounding precision (decimal places)
    HASH_ROUNDING_PRECISION: int = 6


# =============================================================================
# MESH SIMPLIFICATION CONSTANTS
# =============================================================================


class SimplificationConstants:
    """Constants for mesh simplification via edge collapse."""
    # Default simplification settings
    DEFAULT_TARGET_RATIO: float = 0.5
    DEFAULT_MAX_ERROR: float = 0.01

    # Minimum triangles to maintain mesh validity
    MIN_TRIANGLES: int = 1
    MIN_VERTICES: int = 3

    # Edge collapse weight factors for QEM
    EDGE_COLLAPSE_WEIGHT_POSITION: float = 1.0
    EDGE_COLLAPSE_WEIGHT_NORMAL: float = 0.5
    EDGE_COLLAPSE_WEIGHT_UV: float = 0.3

    # Degenerate triangle detection
    MIN_TRIANGLE_AREA: float = 1e-10
    COLLINEAR_THRESHOLD: float = 1e-8


# =============================================================================
# MESH MERGING CONSTANTS
# =============================================================================


class MergeConstants:
    """Constants for mesh merging operations."""
    # Default merge thresholds
    DEFAULT_MERGE_DISTANCE: float = 0.001
    DEFAULT_UV_TOLERANCE: float = 0.0001

    # Interior face detection
    OPPOSING_NORMAL_THRESHOLD: float = -0.99
    INTERIOR_FACE_DISTANCE_MULTIPLIER: float = 10.0


# =============================================================================
# IMPOSTOR GENERATION CONSTANTS
# =============================================================================


class ImpostorConstants:
    """Constants for billboard/impostor generation."""
    # Default impostor settings
    DEFAULT_RESOLUTION: int = 512
    DEFAULT_VIEW_COUNT: int = 8

    # Minimum and maximum resolution bounds
    MIN_RESOLUTION: int = 16
    MAX_RESOLUTION: int = 4096

    # Hemi-octahedron view generation
    HEMI_OCTAHEDRON_Y_ELEVATION: float = 0.3

    # Atlas packing
    MIN_ATLAS_SIZE: int = 64


# =============================================================================
# HLOD METHOD SELECTION CONSTANTS
# =============================================================================


class MethodSelectionConstants:
    """Constants for automatic HLOD method selection."""
    # Triangle count thresholds for method selection
    SMALL_MESH_TRIANGLE_THRESHOLD: int = 1000
    MEDIUM_MESH_TRIANGLE_THRESHOLD: int = 10000
    LARGE_MESH_TRIANGLE_THRESHOLD: int = 100000

    # Mesh count threshold for impostor selection
    MANY_MESHES_THRESHOLD: int = 50


# =============================================================================
# LAYER MANAGEMENT CONSTANTS
# =============================================================================


class LayerConstants:
    """Constants for HLOD layer management."""
    # Default distance thresholds (in world units)
    DEFAULT_LOD0_DISTANCE: float = 500.0
    DEFAULT_LOD1_DISTANCE: float = 1000.0
    DEFAULT_LOD2_DISTANCE: float = 2000.0
    DEFAULT_LOD3_DISTANCE: float = 4000.0

    # Default simplification ratios per layer
    DEFAULT_LOD0_RATIO: float = 1.0      # Original quality
    DEFAULT_LOD1_RATIO: float = 0.5      # 50% triangles
    DEFAULT_LOD2_RATIO: float = 0.25     # 25% triangles
    DEFAULT_LOD3_RATIO: float = 0.1      # 10% triangles

    # Layer limits
    MAX_LAYERS: int = 8
    MIN_THRESHOLD_GAP: float = 100.0


# =============================================================================
# TRANSITION CONSTANTS
# =============================================================================


class TransitionConstantsConfig:
    """Constants for LOD transitions."""
    # Default transition settings
    DEFAULT_TRANSITION_RANGE: float = 50.0
    DEFAULT_DITHER_SCALE: float = 1.0
    DEFAULT_MORPH_SPEED: float = 5.0

    # Screen space error defaults
    DEFAULT_ERROR_THRESHOLD: float = 2.0  # pixels
    DEFAULT_MIN_SCREEN_SIZE: float = 1.0  # pixels

    # Dither pattern configuration
    DITHER_PATTERN_SIZE: int = 4

    # Hysteresis
    DEFAULT_HYSTERESIS_FACTOR: float = 0.1  # 10% hysteresis band
    MIN_HYSTERESIS_FACTOR: float = 0.0
    MAX_HYSTERESIS_FACTOR: float = 1.0


# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================


class ValidationConstants:
    """Constants for input validation."""
    # Ratio bounds
    MIN_RATIO: float = 0.0
    MAX_RATIO: float = 1.0

    # Distance bounds
    MIN_DISTANCE: float = 0.0

    # Screen space bounds
    MIN_SCREEN_HEIGHT: int = 1
    MIN_FOV_RADIANS: float = 0.001
    MAX_FOV_RADIANS: float = 3.14159  # ~180 degrees


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "FloatingPointConstants",
    "SimplificationConstants",
    "MergeConstants",
    "ImpostorConstants",
    "MethodSelectionConstants",
    "LayerConstants",
    "TransitionConstantsConfig",
    "ValidationConstants",
]
