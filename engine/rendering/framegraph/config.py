"""
Configuration constants for the Frame Graph subsystem.

This module centralizes all configurable values that were previously
hardcoded throughout the frame graph implementation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AsyncSchedulerConfig:
    """Configuration for async compute scheduling."""

    # Number of recent graphics passes to check for write dependencies
    # when determining if a compute pass can run async
    recent_graphics_write_window: int = 3

    # Maximum benefit ratio from async compute overlap (0.0 - 1.0)
    # Caps the estimated performance improvement
    max_overlap_benefit: float = 0.3

    # Multiplier for calculating overlap benefit from async ratio
    overlap_benefit_multiplier: float = 0.5

    # Maximum number of read dependencies for a pass to be considered
    # a good async compute candidate
    async_candidate_max_reads: int = 3


@dataclass(frozen=True)
class ResourceManagerConfig:
    """Configuration for resource management."""

    # Minimum buffer size in bytes (must be positive)
    min_buffer_size: int = 1

    # Default texture dimensions when not specified
    default_texture_width: int = 0
    default_texture_height: int = 0


@dataclass(frozen=True)
class FrameGraphConfig:
    """Configuration for the frame graph."""

    # Enable async compute scheduling by default
    default_async_compute_enabled: bool = True

    # Enable unused pass culling by default
    default_pass_culling_enabled: bool = True

    # Enable resource memory aliasing by default
    default_resource_aliasing_enabled: bool = True


# Default configuration instances
ASYNC_SCHEDULER_CONFIG = AsyncSchedulerConfig()
RESOURCE_MANAGER_CONFIG = ResourceManagerConfig()
FRAME_GRAPH_CONFIG = FrameGraphConfig()
