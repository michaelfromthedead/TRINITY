"""
Particle System Constants.

Centralized configuration constants for the Particles & VFX subsystem.
All magic numbers and configurable defaults should be defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ParticleConstants:
    """
    Configuration constants for the particle system.

    Centralizes all magic numbers for easy configuration and tuning.
    """

    # ==========================================================================
    # PARTICLE SYSTEM DEFAULTS
    # ==========================================================================

    # Particle count thresholds
    DEFAULT_MAX_PARTICLES: int = 1000
    DEFAULT_GPU_THRESHOLD: int = 10000  # Switch to GPU above this count
    GPU_MAX_PARTICLES: int = 100000  # Default max for GPU systems

    # Timing defaults
    DEFAULT_WARMUP_TIME: float = 0.0
    DEFAULT_LIFETIME: float = 5.0
    PREWARM_FPS: float = 60.0  # Simulation FPS for prewarm
    COMPACT_INTERVAL: int = 60  # Frames between pool compaction

    # LOD ranges
    DEFAULT_LOD_MIN: int = 0
    DEFAULT_LOD_MAX: int = 3

    # ==========================================================================
    # BUDGET ALLOCATIONS
    # ==========================================================================

    # Category: (max_particles, priority)
    BUDGET_AMBIENT: Tuple[int, int] = (50000, 10)
    BUDGET_GAMEPLAY: Tuple[int, int] = (100000, 50)
    BUDGET_CRITICAL: Tuple[int, int] = (200000, 100)
    BUDGET_DEFAULT: Tuple[int, int] = (100000, 25)
    BUDGET_TOTAL_LIMIT: int = 500000

    # ==========================================================================
    # GPU PARTICLE DEFAULTS
    # ==========================================================================

    # Workgroup sizes for compute shaders
    WORKGROUP_SIZE_X: int = 64
    WORKGROUP_SIZE_Y: int = 1
    WORKGROUP_SIZE_Z: int = 1

    # Buffer sizes
    INDIRECT_BUFFER_SIZE: int = 16  # bytes (4 uint32s)
    COUNTER_BUFFER_SIZE: int = 4  # bytes (1 uint32)

    # Simulation bounds
    DEFAULT_BOUNDS_MIN: float = -1000.0
    DEFAULT_BOUNDS_MAX: float = 1000.0

    # ==========================================================================
    # TRAIL RENDERER DEFAULTS
    # ==========================================================================

    TRAIL_DEFAULT_WIDTH: float = 0.1
    TRAIL_DEFAULT_FADE_TIME: float = 1.0
    TRAIL_DEFAULT_MAX_POINTS: int = 100
    TRAIL_DEFAULT_MIN_DISTANCE: float = 0.01
    TRAIL_CAP_SEGMENTS: int = 4  # Segments for round caps
    TRAIL_ARROW_WIDTH_FACTOR: float = 0.75  # Arrow base width relative to trail

    # ==========================================================================
    # DECAL SYSTEM DEFAULTS
    # ==========================================================================

    DECAL_DEFAULT_LIFETIME: float = None  # None = infinite
    DECAL_DEFAULT_FADE_TIME: float = 1.0
    DECAL_DEFAULT_CHANNEL: int = 0
    DECAL_DEFAULT_PRIORITY: int = 0
    DECAL_DEFAULT_SIZE: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    DECAL_SYSTEM_MAX_DECALS: int = 1000

    # Atlas defaults
    DECAL_ATLAS_DEFAULT_WIDTH: int = 2048
    DECAL_ATLAS_DEFAULT_HEIGHT: int = 2048
    DECAL_ATLAS_DEFAULT_PADDING: int = 2

    # ==========================================================================
    # PHYSICS CONSTANTS
    # ==========================================================================

    DEFAULT_GRAVITY_Y: float = -9.81
    EPSILON: float = 1e-8  # Small value for floating point comparisons
    COLLISION_EPSILON: float = 0.001  # Minimum distance for collision checks
    ALPHA_DEATH_THRESHOLD: float = 0.001  # Alpha below which particle is considered dead

    # ==========================================================================
    # SPAWN RATE LIMITS
    # ==========================================================================

    MAX_SPAWN_RATE: float = 100000.0  # Maximum particles per second
    MAX_BURST_COUNT: int = 1000000  # Maximum particles per burst

    # ==========================================================================
    # VFX GRAPH DEFAULTS
    # ==========================================================================

    VFX_DEFAULT_SPAWN_RATE: float = 100.0
    VFX_MAX_SPAWN_RATE: float = 100000.0
    VFX_DEFAULT_BURST_COUNT: int = 10
    VFX_MAX_EMITTER_PARTICLES: int = 1000000


# Singleton instance for easy access
PARTICLE_CONSTANTS = ParticleConstants()


# Backwards compatibility - export individual constants
DEFAULT_GPU_THRESHOLD = PARTICLE_CONSTANTS.DEFAULT_GPU_THRESHOLD
DEFAULT_MAX_PARTICLES = PARTICLE_CONSTANTS.DEFAULT_MAX_PARTICLES
DEFAULT_WARMUP_TIME = PARTICLE_CONSTANTS.DEFAULT_WARMUP_TIME
DEFAULT_LIFETIME = PARTICLE_CONSTANTS.DEFAULT_LIFETIME

DEFAULT_TRAIL_WIDTH = PARTICLE_CONSTANTS.TRAIL_DEFAULT_WIDTH
DEFAULT_FADE_TIME = PARTICLE_CONSTANTS.TRAIL_DEFAULT_FADE_TIME
DEFAULT_MAX_POINTS = PARTICLE_CONSTANTS.TRAIL_DEFAULT_MAX_POINTS
DEFAULT_MIN_DISTANCE = PARTICLE_CONSTANTS.TRAIL_DEFAULT_MIN_DISTANCE

DEFAULT_DECAL_LIFETIME = PARTICLE_CONSTANTS.DECAL_DEFAULT_LIFETIME
DEFAULT_DECAL_FADE_TIME = PARTICLE_CONSTANTS.DECAL_DEFAULT_FADE_TIME
DEFAULT_DECAL_CHANNEL = PARTICLE_CONSTANTS.DECAL_DEFAULT_CHANNEL
DEFAULT_DECAL_PRIORITY = PARTICLE_CONSTANTS.DECAL_DEFAULT_PRIORITY


__all__ = [
    "ParticleConstants",
    "PARTICLE_CONSTANTS",
    # Legacy exports
    "DEFAULT_GPU_THRESHOLD",
    "DEFAULT_MAX_PARTICLES",
    "DEFAULT_WARMUP_TIME",
    "DEFAULT_LIFETIME",
    "DEFAULT_TRAIL_WIDTH",
    "DEFAULT_FADE_TIME",
    "DEFAULT_MAX_POINTS",
    "DEFAULT_MIN_DISTANCE",
    "DEFAULT_DECAL_LIFETIME",
    "DEFAULT_DECAL_FADE_TIME",
    "DEFAULT_DECAL_CHANNEL",
    "DEFAULT_DECAL_PRIORITY",
]
