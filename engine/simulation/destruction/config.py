"""
Destruction System Configuration.

This module defines all configurable constants for the destruction simulation system
including fracture parameters, debris management, and damage propagation settings.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Final


# =============================================================================
# FRACTURE CONFIGURATION
# =============================================================================

DEFAULT_FRACTURE_SEED: Final[int] = 42
"""Default random seed for deterministic fracture pattern generation."""

MIN_CHUNK_VOLUME: Final[float] = 0.001
"""Minimum volume (cubic units) for a valid chunk. Smaller chunks are merged or discarded."""

MAX_CHUNKS_PER_OBJECT: Final[int] = 64
"""Maximum number of chunks generated from a single fracture operation."""

MIN_VORONOI_SITES: Final[int] = 4
"""Minimum number of Voronoi sites for cell generation."""

MAX_VORONOI_SITES: Final[int] = 128
"""Maximum number of Voronoi sites to prevent excessive computation."""

DEFAULT_VORONOI_SITES: Final[int] = 16
"""Default number of Voronoi sites for fracture generation."""

RADIAL_MIN_SLICES: Final[int] = 4
"""Minimum number of radial slices for radial fracture pattern."""

RADIAL_MAX_SLICES: Final[int] = 32
"""Maximum number of radial slices."""

RADIAL_DEFAULT_SLICES: Final[int] = 8
"""Default number of radial slices."""

RADIAL_MIN_RINGS: Final[int] = 1
"""Minimum number of concentric rings for radial fracture."""

RADIAL_MAX_RINGS: Final[int] = 8
"""Maximum number of concentric rings."""

RADIAL_DEFAULT_RINGS: Final[int] = 3
"""Default number of concentric rings."""

SLICE_MAX_PLANES: Final[int] = 16
"""Maximum number of slice planes in a single operation."""


# =============================================================================
# DEBRIS CONFIGURATION
# =============================================================================

DEBRIS_LIFETIME: Final[float] = 10.0
"""Default lifetime in seconds for debris before cleanup."""

DEBRIS_MIN_LIFETIME: Final[float] = 0.5
"""Minimum debris lifetime to prevent instant cleanup."""

DEBRIS_MAX_LIFETIME: Final[float] = 60.0
"""Maximum debris lifetime to prevent memory leaks."""

MAX_ACTIVE_DEBRIS: Final[int] = 1000
"""Maximum number of active debris pieces in the simulation."""

DEBRIS_POOL_INITIAL_SIZE: Final[int] = 256
"""Initial pool size for debris object allocation."""

DEBRIS_POOL_GROWTH_FACTOR: Final[float] = 1.5
"""Growth factor when debris pool needs expansion."""

DEBRIS_MERGE_DISTANCE: Final[float] = 0.1
"""Distance threshold for merging small debris pieces."""

DEBRIS_MIN_VELOCITY: Final[float] = 0.01
"""Minimum velocity threshold; debris below this is considered at rest."""

DEBRIS_SLEEP_TIME: Final[float] = 1.0
"""Time in seconds of low velocity before debris can be cleaned up early."""


# =============================================================================
# DAMAGE CONFIGURATION
# =============================================================================

DAMAGE_PROPAGATION_FACTOR: Final[float] = 0.5
"""Factor applied when damage propagates to neighboring chunks (0.0-1.0)."""

DAMAGE_MIN_THRESHOLD: Final[float] = 0.1
"""Minimum damage amount to process (filters noise)."""

DAMAGE_ACCUMULATION_RATE: Final[float] = 1.0
"""Rate at which damage accumulates (can be used for time-based effects)."""

DAMAGE_DECAY_RATE: Final[float] = 0.0
"""Rate at which accumulated damage decays over time (0.0 = no decay)."""


# =============================================================================
# SUPPORT STRUCTURE CONFIGURATION
# =============================================================================

SUPPORT_STRESS_THRESHOLD: Final[float] = 1000.0
"""Stress threshold above which support connections break."""

SUPPORT_MAX_CONNECTIONS: Final[int] = 8
"""Maximum number of support connections per chunk."""

SUPPORT_MIN_CONTACT_AREA: Final[float] = 0.01
"""Minimum contact area for a valid support connection."""

SUPPORT_STRESS_PROPAGATION_RATE: Final[float] = 0.8
"""Rate at which stress propagates through support graph."""

SUPPORT_GRAPH_MAX_DEPTH: Final[int] = 50
"""Maximum depth for damage propagation to prevent infinite loops in cycles."""


# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

FRACTURE_BATCH_SIZE: Final[int] = 16
"""Number of fractures to process per frame to avoid spikes."""

DEBRIS_UPDATE_BATCH_SIZE: Final[int] = 64
"""Number of debris pieces to update per batch."""

SUPPORT_GRAPH_MAX_DEPTH: Final[int] = 32
"""Maximum depth for support graph traversal."""

CHUNK_CACHE_SIZE: Final[int] = 256
"""Size of the chunk mesh cache for reuse."""


# =============================================================================
# DEBRIS SPAWN CONFIGURATION
# =============================================================================

DEBRIS_ANGULAR_VELOCITY_MIN: Final[float] = -2.0
"""Minimum angular velocity component for spawned debris."""

DEBRIS_ANGULAR_VELOCITY_MAX: Final[float] = 2.0
"""Maximum angular velocity component for spawned debris."""

DEBRIS_IMPORTANCE_VOLUME_MULTIPLIER: Final[float] = 10.0
"""Multiplier for computing debris importance from volume."""

DEBRIS_LOD_DISTANCE_FULL: Final[float] = 10.0
"""Distance threshold for full LOD debris rendering."""

DEBRIS_LOD_DISTANCE_REDUCED: Final[float] = 25.0
"""Distance threshold for reduced LOD debris rendering."""

DEBRIS_LOD_DISTANCE_SIMPLE: Final[float] = 50.0
"""Distance threshold for simple LOD debris rendering."""


# =============================================================================
# FRACTURE SPAWN CONFIGURATION
# =============================================================================

FRACTURE_VELOCITY_MULTIPLIER: Final[float] = 5.0
"""Multiplier for debris velocity based on fracture intensity."""

FRACTURE_SPREAD_MULTIPLIER: Final[float] = 3.0
"""Multiplier for debris spread based on fracture intensity."""

SURFACE_SAMPLE_ITERATIONS: Final[int] = 20
"""Number of iterations for surface point sampling."""

DEGENERATE_TRIANGLE_AREA_THRESHOLD: Final[float] = 1e-10
"""Minimum triangle area to avoid degenerate triangles."""


# =============================================================================
# ENUMERATIONS
# =============================================================================

class FracturePattern(IntEnum):
    """Supported fracture pattern types."""
    VORONOI = 0
    RADIAL = 1
    SLICE = 2
    CUSTOM = 3


class DebrisState(IntEnum):
    """State of a debris piece."""
    ACTIVE = 0
    SLEEPING = 1
    PENDING_CLEANUP = 2
    POOLED = 3


class SupportType(IntEnum):
    """Types of support connections."""
    FIXED = 0       # Anchored to world
    STRUCTURAL = 1  # Connected to another chunk
    TEMPORARY = 2   # Breakable connection


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass(frozen=True, slots=True)
class FractureConfig:
    """Configuration for fracture operations."""
    pattern: FracturePattern = FracturePattern.VORONOI
    seed: int = DEFAULT_FRACTURE_SEED
    max_chunks: int = MAX_CHUNKS_PER_OBJECT
    min_chunk_volume: float = MIN_CHUNK_VOLUME
    num_sites: int = DEFAULT_VORONOI_SITES
    num_slices: int = RADIAL_DEFAULT_SLICES
    num_rings: int = RADIAL_DEFAULT_RINGS
    preserve_surface: bool = True
    generate_interior_uvs: bool = True

    def __post_init__(self) -> None:
        if self.max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        if self.min_chunk_volume <= 0:
            raise ValueError("min_chunk_volume must be > 0")
        if self.num_sites < MIN_VORONOI_SITES:
            raise ValueError(f"num_sites must be >= {MIN_VORONOI_SITES}")
        if self.num_slices < RADIAL_MIN_SLICES:
            raise ValueError(f"num_slices must be >= {RADIAL_MIN_SLICES}")


@dataclass(frozen=True, slots=True)
class DebrisConfig:
    """Configuration for debris management."""
    lifetime: float = DEBRIS_LIFETIME
    max_active: int = MAX_ACTIVE_DEBRIS
    pool_size: int = DEBRIS_POOL_INITIAL_SIZE
    merge_distance: float = DEBRIS_MERGE_DISTANCE
    sleep_velocity: float = DEBRIS_MIN_VELOCITY
    sleep_time: float = DEBRIS_SLEEP_TIME

    def __post_init__(self) -> None:
        if self.lifetime < DEBRIS_MIN_LIFETIME:
            raise ValueError(f"lifetime must be >= {DEBRIS_MIN_LIFETIME}")
        if self.lifetime > DEBRIS_MAX_LIFETIME:
            raise ValueError(f"lifetime must be <= {DEBRIS_MAX_LIFETIME}")
        if self.max_active < 1:
            raise ValueError("max_active must be >= 1")


@dataclass(frozen=True, slots=True)
class DamageConfig:
    """Configuration for damage system."""
    propagation_factor: float = DAMAGE_PROPAGATION_FACTOR
    min_threshold: float = DAMAGE_MIN_THRESHOLD
    accumulation_rate: float = DAMAGE_ACCUMULATION_RATE
    decay_rate: float = DAMAGE_DECAY_RATE

    def __post_init__(self) -> None:
        if not 0.0 <= self.propagation_factor <= 1.0:
            raise ValueError("propagation_factor must be in [0.0, 1.0]")
        if self.min_threshold < 0:
            raise ValueError("min_threshold must be >= 0")


@dataclass(frozen=True, slots=True)
class SupportConfig:
    """Configuration for support structure system."""
    stress_threshold: float = SUPPORT_STRESS_THRESHOLD
    max_connections: int = SUPPORT_MAX_CONNECTIONS
    min_contact_area: float = SUPPORT_MIN_CONTACT_AREA
    propagation_rate: float = SUPPORT_STRESS_PROPAGATION_RATE

    def __post_init__(self) -> None:
        if self.stress_threshold <= 0:
            raise ValueError("stress_threshold must be > 0")
        if self.max_connections < 1:
            raise ValueError("max_connections must be >= 1")


@dataclass(slots=True)
class DestructionSystemConfig:
    """Master configuration for the destruction system."""
    fracture: FractureConfig = None  # type: ignore
    debris: DebrisConfig = None  # type: ignore
    damage: DamageConfig = None  # type: ignore
    support: SupportConfig = None  # type: ignore

    def __post_init__(self) -> None:
        if self.fracture is None:
            object.__setattr__(self, 'fracture', FractureConfig())
        if self.debris is None:
            object.__setattr__(self, 'debris', DebrisConfig())
        if self.damage is None:
            object.__setattr__(self, 'damage', DamageConfig())
        if self.support is None:
            object.__setattr__(self, 'support', SupportConfig())


# Default configuration instance
DEFAULT_CONFIG = DestructionSystemConfig()
