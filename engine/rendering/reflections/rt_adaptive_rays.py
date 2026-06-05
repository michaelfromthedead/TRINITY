"""Roughness-Based Ray Count Adaptation for RT Reflections (T-GIR-P8.3).

This module implements adaptive ray tracing based on surface roughness:
- Smooth surfaces: 1 ray/pixel at full resolution
- Rough surfaces: multiple rays at lower resolution with heavy denoising

Components:
- RoughnessRayMapping: Maps roughness to ray count, resolution, denoise strength
- ResolutionHierarchy: Multi-resolution tier management
- AdaptiveRayScheduler: Per-pixel ray scheduling with temporal stratification
- RayBudgetManager: Global ray budget distribution
- AdaptiveRTConfig: Configuration for adaptive ray tracing

Ray Count Strategy:
    roughness | rays/pixel | resolution | denoise
    ----------|------------|------------|--------
    0.0-0.1   | 1          | 1x         | none
    0.1-0.3   | 2          | 0.5x       | light
    0.3-0.5   | 4          | 0.25x      | medium
    0.5-0.7   | 8          | 0.25x      | heavy

References:
    - T-GIR-P8.1 RT Reflection Ray Generation (rt_reflections.py)
    - Section 6.11 Ray Tracing Architecture in RENDERING_CONTEXT.md
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from engine.core.math.vec import Vec2, Vec3, Vec4

if TYPE_CHECKING:
    from engine.rendering.reflections.rt_reflections import (
        GBufferReader,
        ReflectionRay,
        RTReflectionConfig,
        RTReflectionPass,
        RTReflectionTracer,
    )


# =============================================================================
# Constants
# =============================================================================

# Default roughness thresholds for resolution tiers
DEFAULT_TIER_THRESHOLDS = [0.1, 0.3, 0.5, 0.7]

# Resolution scales for each tier
DEFAULT_RESOLUTION_SCALES = [1.0, 0.5, 0.25, 0.25]

# Ray counts for each tier
DEFAULT_RAY_COUNTS = [1, 2, 4, 8]

# Denoise strengths for each tier
DEFAULT_DENOISE_STRENGTHS = [0.0, 0.25, 0.5, 1.0]

# Default ray budget per frame (for 1080p target)
DEFAULT_RAY_BUDGET = 2_073_600  # 1920 * 1080

# Default temporal accumulation frames
DEFAULT_TEMPORAL_FRAMES = 4

# Maximum roughness for RT (above this, skip tracing)
MAX_RT_ROUGHNESS = 0.7


# =============================================================================
# Resolution Tier Enum
# =============================================================================


class ResolutionTier(Enum):
    """Resolution tiers for adaptive ray tracing."""

    FULL = auto()
    """Full resolution (1.0x) - smoothest surfaces."""

    HALF = auto()
    """Half resolution (0.5x) - slightly rough surfaces."""

    QUARTER = auto()
    """Quarter resolution (0.25x) - rough surfaces."""

    EIGHTH = auto()
    """Eighth resolution (0.125x) - very rough surfaces."""


_TIER_SCALES = {
    ResolutionTier.FULL: 1.0,
    ResolutionTier.HALF: 0.5,
    ResolutionTier.QUARTER: 0.25,
    ResolutionTier.EIGHTH: 0.125,
}


class DenoiseLevel(Enum):
    """Denoising intensity levels."""

    NONE = auto()
    """No denoising (smooth surfaces)."""

    LIGHT = auto()
    """Light denoising (low roughness)."""

    MEDIUM = auto()
    """Medium denoising (moderate roughness)."""

    HEAVY = auto()
    """Heavy denoising (high roughness)."""


_DENOISE_STRENGTHS = {
    DenoiseLevel.NONE: 0.0,
    DenoiseLevel.LIGHT: 0.25,
    DenoiseLevel.MEDIUM: 0.5,
    DenoiseLevel.HEAVY: 1.0,
}


# =============================================================================
# Roughness Ray Mapping
# =============================================================================


@dataclass
class RoughnessMapping:
    """Result of roughness to ray parameter mapping.

    Attributes:
        ray_count: Number of rays to trace for this pixel.
        resolution_scale: Resolution scaling factor [0.125, 1.0].
        denoise_strength: Denoising strength [0.0, 1.0].
        tier: Resolution tier enum.
        denoise_level: Denoising level enum.
        should_trace: Whether to trace rays at all.
    """

    ray_count: int = 1
    resolution_scale: float = 1.0
    denoise_strength: float = 0.0
    tier: ResolutionTier = ResolutionTier.FULL
    denoise_level: DenoiseLevel = DenoiseLevel.NONE
    should_trace: bool = True


class RoughnessRayMapping:
    """Maps surface roughness to ray tracing parameters.

    Provides methods to determine:
    - Number of rays per pixel based on roughness
    - Resolution scale based on roughness
    - Denoising strength based on roughness

    The mapping follows a tiered approach:
    - Roughness 0.0-0.1: 1 ray at full resolution, no denoise
    - Roughness 0.1-0.3: 2 rays at half resolution, light denoise
    - Roughness 0.3-0.5: 4 rays at quarter resolution, medium denoise
    - Roughness 0.5-0.7: 8 rays at quarter resolution, heavy denoise
    - Roughness >0.7: skip RT entirely

    Usage:
        mapping = RoughnessRayMapping()
        result = mapping.get_mapping(roughness=0.2)
        ray_count = mapping.get_ray_count(roughness=0.2)
        scale = mapping.get_resolution_scale(roughness=0.2)
    """

    def __init__(
        self,
        tier_thresholds: Optional[List[float]] = None,
        ray_counts: Optional[List[int]] = None,
        resolution_scales: Optional[List[float]] = None,
        denoise_strengths: Optional[List[float]] = None,
        max_roughness: float = MAX_RT_ROUGHNESS,
    ) -> None:
        """Initialize the roughness ray mapping.

        Args:
            tier_thresholds: Roughness thresholds for each tier.
            ray_counts: Ray count for each tier.
            resolution_scales: Resolution scale for each tier.
            denoise_strengths: Denoise strength for each tier.
            max_roughness: Maximum roughness for RT (skip if above).
        """
        self._tier_thresholds = tier_thresholds or list(DEFAULT_TIER_THRESHOLDS)
        self._ray_counts = ray_counts or list(DEFAULT_RAY_COUNTS)
        self._resolution_scales = resolution_scales or list(DEFAULT_RESOLUTION_SCALES)
        self._denoise_strengths = denoise_strengths or list(DEFAULT_DENOISE_STRENGTHS)
        self._max_roughness = max_roughness

        # Ensure lists have same length
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate that configuration lists have consistent lengths."""
        n = len(self._tier_thresholds)
        if len(self._ray_counts) != n:
            raise ValueError(
                f"ray_counts length ({len(self._ray_counts)}) "
                f"must match tier_thresholds length ({n})"
            )
        if len(self._resolution_scales) != n:
            raise ValueError(
                f"resolution_scales length ({len(self._resolution_scales)}) "
                f"must match tier_thresholds length ({n})"
            )
        if len(self._denoise_strengths) != n:
            raise ValueError(
                f"denoise_strengths length ({len(self._denoise_strengths)}) "
                f"must match tier_thresholds length ({n})"
            )

        # Ensure thresholds are sorted
        for i in range(1, n):
            if self._tier_thresholds[i] <= self._tier_thresholds[i - 1]:
                raise ValueError("tier_thresholds must be strictly increasing")

    @property
    def tier_count(self) -> int:
        """Get the number of tiers."""
        return len(self._tier_thresholds)

    @property
    def max_roughness(self) -> float:
        """Get maximum roughness for RT."""
        return self._max_roughness

    @max_roughness.setter
    def max_roughness(self, value: float) -> None:
        """Set maximum roughness for RT."""
        self._max_roughness = max(0.0, min(1.0, value))

    @property
    def tier_thresholds(self) -> List[float]:
        """Get tier threshold values."""
        return list(self._tier_thresholds)

    def get_tier_index(self, roughness: float) -> int:
        """Get the tier index for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            Tier index (0 = smoothest tier).
        """
        roughness = max(0.0, min(1.0, roughness))

        for i, threshold in enumerate(self._tier_thresholds):
            if roughness <= threshold:
                return i

        return len(self._tier_thresholds) - 1

    def get_ray_count(self, roughness: float) -> int:
        """Get the number of rays for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            Number of rays to trace.
        """
        if roughness > self._max_roughness:
            return 0

        tier_idx = self.get_tier_index(roughness)
        return self._ray_counts[tier_idx]

    def get_resolution_scale(self, roughness: float) -> float:
        """Get the resolution scale for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            Resolution scale [0.125, 1.0].
        """
        if roughness > self._max_roughness:
            return 0.0

        tier_idx = self.get_tier_index(roughness)
        return self._resolution_scales[tier_idx]

    def get_denoise_strength(self, roughness: float) -> float:
        """Get the denoising strength for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            Denoise strength [0.0, 1.0].
        """
        if roughness > self._max_roughness:
            return 1.0

        tier_idx = self.get_tier_index(roughness)
        return self._denoise_strengths[tier_idx]

    def get_tier(self, roughness: float) -> ResolutionTier:
        """Get the resolution tier for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            Resolution tier enum.
        """
        scale = self.get_resolution_scale(roughness)

        if scale >= 1.0:
            return ResolutionTier.FULL
        elif scale >= 0.5:
            return ResolutionTier.HALF
        elif scale >= 0.25:
            return ResolutionTier.QUARTER
        else:
            return ResolutionTier.EIGHTH

    def get_denoise_level(self, roughness: float) -> DenoiseLevel:
        """Get the denoise level for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            Denoise level enum.
        """
        strength = self.get_denoise_strength(roughness)

        if strength <= 0.0:
            return DenoiseLevel.NONE
        elif strength <= 0.25:
            return DenoiseLevel.LIGHT
        elif strength <= 0.5:
            return DenoiseLevel.MEDIUM
        else:
            return DenoiseLevel.HEAVY

    def should_trace(self, roughness: float) -> bool:
        """Check if roughness is low enough to trace.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            True if roughness <= max_roughness.
        """
        return roughness <= self._max_roughness

    def get_mapping(self, roughness: float) -> RoughnessMapping:
        """Get complete mapping for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            RoughnessMapping with all parameters.
        """
        should_trace = self.should_trace(roughness)

        if not should_trace:
            return RoughnessMapping(
                ray_count=0,
                resolution_scale=0.0,
                denoise_strength=1.0,
                tier=ResolutionTier.QUARTER,
                denoise_level=DenoiseLevel.HEAVY,
                should_trace=False,
            )

        return RoughnessMapping(
            ray_count=self.get_ray_count(roughness),
            resolution_scale=self.get_resolution_scale(roughness),
            denoise_strength=self.get_denoise_strength(roughness),
            tier=self.get_tier(roughness),
            denoise_level=self.get_denoise_level(roughness),
            should_trace=True,
        )

    def interpolate_parameters(
        self, roughness: float, smooth_blend: bool = False
    ) -> RoughnessMapping:
        """Get interpolated parameters between tiers (for smooth transitions).

        Args:
            roughness: Surface roughness [0, 1].
            smooth_blend: Whether to interpolate between adjacent tiers.

        Returns:
            RoughnessMapping with potentially interpolated values.
        """
        if not smooth_blend:
            return self.get_mapping(roughness)

        if not self.should_trace(roughness):
            return self.get_mapping(roughness)

        tier_idx = self.get_tier_index(roughness)

        # Get current tier values
        ray_count = self._ray_counts[tier_idx]
        res_scale = self._resolution_scales[tier_idx]
        denoise = self._denoise_strengths[tier_idx]

        # Check if we need to blend with next tier
        if tier_idx < len(self._tier_thresholds) - 1:
            current_threshold = self._tier_thresholds[tier_idx]
            if tier_idx > 0:
                prev_threshold = self._tier_thresholds[tier_idx - 1]
            else:
                prev_threshold = 0.0

            # Calculate blend factor within current tier
            tier_range = current_threshold - prev_threshold
            if tier_range > 0:
                t = (roughness - prev_threshold) / tier_range
                t = max(0.0, min(1.0, t))

                # Blend with next tier if near edge
                if t > 0.7 and tier_idx < len(self._tier_thresholds) - 1:
                    next_idx = tier_idx + 1
                    blend = (t - 0.7) / 0.3  # Map 0.7-1.0 to 0-1
                    blend = max(0.0, min(1.0, blend))

                    # Interpolate resolution and denoise (not ray count)
                    res_scale = res_scale * (1 - blend) + self._resolution_scales[next_idx] * blend
                    denoise = denoise * (1 - blend) + self._denoise_strengths[next_idx] * blend

        return RoughnessMapping(
            ray_count=ray_count,
            resolution_scale=res_scale,
            denoise_strength=denoise,
            tier=self.get_tier(roughness),
            denoise_level=self.get_denoise_level(roughness),
            should_trace=True,
        )


# =============================================================================
# Resolution Hierarchy
# =============================================================================


@dataclass
class ResolutionLevel:
    """A single level in the resolution hierarchy.

    Attributes:
        scale: Resolution scale factor.
        width: Width in pixels at this level.
        height: Height in pixels at this level.
        tier: Resolution tier enum.
        roughness_min: Minimum roughness for this level.
        roughness_max: Maximum roughness for this level.
    """

    scale: float
    width: int
    height: int
    tier: ResolutionTier
    roughness_min: float
    roughness_max: float


class ResolutionHierarchy:
    """Manages multi-resolution rendering for adaptive ray tracing.

    Creates a hierarchy of resolution levels:
    - Full resolution (1x): roughness 0.0-0.1
    - Half resolution (0.5x): roughness 0.1-0.3
    - Quarter resolution (0.25x): roughness 0.3+

    Provides upscaling from lower resolution levels to full resolution.

    Usage:
        hierarchy = ResolutionHierarchy(1920, 1080)
        level = hierarchy.get_level_for_roughness(0.2)
        result = hierarchy.upscale_result(low_res_data, level)
    """

    def __init__(
        self,
        base_width: int,
        base_height: int,
        scales: Optional[List[float]] = None,
        roughness_thresholds: Optional[List[float]] = None,
    ) -> None:
        """Initialize the resolution hierarchy.

        Args:
            base_width: Base (full) resolution width.
            base_height: Base (full) resolution height.
            scales: Resolution scale factors for each level.
            roughness_thresholds: Roughness boundaries for each level.
        """
        self._base_width = base_width
        self._base_height = base_height
        self._scales = scales or [1.0, 0.5, 0.25]
        self._thresholds = roughness_thresholds or [0.1, 0.3, 1.0]

        self._levels: List[ResolutionLevel] = []
        self._build_hierarchy()

    def _build_hierarchy(self) -> None:
        """Build the resolution level hierarchy."""
        self._levels.clear()

        tiers = [ResolutionTier.FULL, ResolutionTier.HALF, ResolutionTier.QUARTER]

        for i, scale in enumerate(self._scales):
            width = max(1, int(self._base_width * scale))
            height = max(1, int(self._base_height * scale))

            roughness_min = self._thresholds[i - 1] if i > 0 else 0.0
            roughness_max = self._thresholds[i] if i < len(self._thresholds) else 1.0

            tier = tiers[min(i, len(tiers) - 1)]

            self._levels.append(
                ResolutionLevel(
                    scale=scale,
                    width=width,
                    height=height,
                    tier=tier,
                    roughness_min=roughness_min,
                    roughness_max=roughness_max,
                )
            )

    @property
    def base_width(self) -> int:
        """Get base resolution width."""
        return self._base_width

    @property
    def base_height(self) -> int:
        """Get base resolution height."""
        return self._base_height

    @property
    def level_count(self) -> int:
        """Get number of resolution levels."""
        return len(self._levels)

    @property
    def levels(self) -> List[ResolutionLevel]:
        """Get all resolution levels."""
        return list(self._levels)

    def set_base_resolution(self, width: int, height: int) -> None:
        """Update base resolution and rebuild hierarchy.

        Args:
            width: New base width.
            height: New base height.
        """
        self._base_width = width
        self._base_height = height
        self._build_hierarchy()

    def get_level(self, index: int) -> ResolutionLevel:
        """Get resolution level by index.

        Args:
            index: Level index (0 = full resolution).

        Returns:
            ResolutionLevel for that index.
        """
        if index < 0 or index >= len(self._levels):
            return self._levels[-1]
        return self._levels[index]

    def get_level_for_roughness(self, roughness: float) -> ResolutionLevel:
        """Get the appropriate resolution level for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            ResolutionLevel for that roughness.
        """
        roughness = max(0.0, min(1.0, roughness))

        for level in self._levels:
            if roughness <= level.roughness_max:
                return level

        return self._levels[-1]

    def get_tier(self, roughness: float) -> ResolutionTier:
        """Get the resolution tier for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            ResolutionTier enum.
        """
        return self.get_level_for_roughness(roughness).tier

    def get_scale_for_roughness(self, roughness: float) -> float:
        """Get the resolution scale for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            Resolution scale factor.
        """
        return self.get_level_for_roughness(roughness).scale

    def get_dimensions_for_roughness(self, roughness: float) -> Tuple[int, int]:
        """Get output dimensions for a roughness value.

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            (width, height) tuple for the appropriate level.
        """
        level = self.get_level_for_roughness(roughness)
        return (level.width, level.height)

    def uv_to_level_pixel(
        self, uv: Vec2, roughness: float
    ) -> Tuple[int, int, ResolutionLevel]:
        """Convert UV coordinates to pixel coordinates at appropriate level.

        Args:
            uv: UV coordinates [0, 1].
            roughness: Surface roughness.

        Returns:
            (x, y, level) tuple.
        """
        level = self.get_level_for_roughness(roughness)
        x = int(uv.x * (level.width - 1))
        y = int(uv.y * (level.height - 1))
        return (x, y, level)

    def level_pixel_to_uv(self, x: int, y: int, level: ResolutionLevel) -> Vec2:
        """Convert level pixel coordinates to UV.

        Args:
            x: Pixel X at level.
            y: Pixel Y at level.
            level: Resolution level.

        Returns:
            UV coordinates [0, 1].
        """
        u = (x + 0.5) / level.width
        v = (y + 0.5) / level.height
        return Vec2(u, v)

    def upscale_result(
        self,
        data: List[Vec3],
        source_level: ResolutionLevel,
        method: str = "bilinear",
    ) -> List[Vec3]:
        """Upscale data from a lower resolution level to full resolution.

        Args:
            data: Source data at lower resolution.
            source_level: Source resolution level.
            method: Upscaling method ("nearest", "bilinear").

        Returns:
            Upscaled data at full resolution.
        """
        if source_level.scale >= 1.0:
            return list(data)

        target_size = self._base_width * self._base_height
        result = [Vec3.zero() for _ in range(target_size)]

        for y in range(self._base_height):
            for x in range(self._base_width):
                # Map to source coordinates
                src_x = x * source_level.scale
                src_y = y * source_level.scale

                if method == "nearest":
                    color = self._sample_nearest(data, source_level, src_x, src_y)
                else:
                    color = self._sample_bilinear(data, source_level, src_x, src_y)

                result[y * self._base_width + x] = color

        return result

    def _sample_nearest(
        self, data: List[Vec3], level: ResolutionLevel, x: float, y: float
    ) -> Vec3:
        """Nearest-neighbor sampling.

        Args:
            data: Source data.
            level: Source resolution level.
            x: X coordinate in source space.
            y: Y coordinate in source space.

        Returns:
            Sampled color.
        """
        ix = max(0, min(level.width - 1, int(x)))
        iy = max(0, min(level.height - 1, int(y)))
        idx = iy * level.width + ix

        if idx < len(data):
            return data[idx]
        return Vec3.zero()

    def _sample_bilinear(
        self, data: List[Vec3], level: ResolutionLevel, x: float, y: float
    ) -> Vec3:
        """Bilinear sampling.

        Args:
            data: Source data.
            level: Source resolution level.
            x: X coordinate in source space.
            y: Y coordinate in source space.

        Returns:
            Bilinearly interpolated color.
        """
        x0 = max(0, min(level.width - 1, int(x)))
        y0 = max(0, min(level.height - 1, int(y)))
        x1 = min(level.width - 1, x0 + 1)
        y1 = min(level.height - 1, y0 + 1)

        fx = x - x0
        fy = y - y0

        def get_sample(ix: int, iy: int) -> Vec3:
            idx = iy * level.width + ix
            if idx < len(data):
                return data[idx]
            return Vec3.zero()

        c00 = get_sample(x0, y0)
        c10 = get_sample(x1, y0)
        c01 = get_sample(x0, y1)
        c11 = get_sample(x1, y1)

        # Bilinear interpolation
        c0 = Vec3(
            c00.x * (1 - fx) + c10.x * fx,
            c00.y * (1 - fx) + c10.y * fx,
            c00.z * (1 - fx) + c10.z * fx,
        )
        c1 = Vec3(
            c01.x * (1 - fx) + c11.x * fx,
            c01.y * (1 - fx) + c11.y * fx,
            c01.z * (1 - fx) + c11.z * fx,
        )

        return Vec3(
            c0.x * (1 - fy) + c1.x * fy,
            c0.y * (1 - fy) + c1.y * fy,
            c0.z * (1 - fy) + c1.z * fy,
        )

    def estimate_memory_usage(self) -> int:
        """Estimate total memory for all levels.

        Returns:
            Memory estimate in bytes.
        """
        total = 0
        # RGBA16F (8 bytes) + depth (4 bytes) per pixel per level
        bytes_per_pixel = 12

        for level in self._levels:
            total += level.width * level.height * bytes_per_pixel

        return total


# =============================================================================
# Adaptive Ray Scheduler
# =============================================================================


@dataclass
class ScheduledRays:
    """Rays scheduled for a pixel.

    Attributes:
        pixel_x: Pixel X coordinate.
        pixel_y: Pixel Y coordinate.
        ray_count: Number of rays to trace.
        ray_directions: Optional list of specific directions (for stratified sampling).
        frame_index: Frame index for temporal stratification.
        accumulation_weight: Weight for this frame's contribution.
    """

    pixel_x: int
    pixel_y: int
    ray_count: int
    ray_directions: List[Vec3] = field(default_factory=list)
    frame_index: int = 0
    accumulation_weight: float = 1.0


@dataclass
class AccumulatedResult:
    """Accumulated result over multiple frames.

    Attributes:
        color: Accumulated color.
        hit_distance: Average hit distance.
        sample_count: Number of accumulated samples.
        confidence: Confidence value [0, 1].
        frames_accumulated: Number of frames accumulated.
    """

    color: Vec3 = field(default_factory=Vec3.zero)
    hit_distance: float = 0.0
    sample_count: int = 0
    confidence: float = 0.0
    frames_accumulated: int = 0


class AdaptiveRayScheduler:
    """Schedules rays per-pixel based on roughness with temporal stratification.

    Manages:
    - Per-pixel ray budgets based on surface roughness
    - Temporal stratification for noise-free accumulation
    - Frame-to-frame sample distribution

    Usage:
        scheduler = AdaptiveRayScheduler(config)
        scheduled = scheduler.schedule_rays(pixel_x, pixel_y, roughness, frame)
        result = trace_rays(scheduled)
        accumulated = scheduler.accumulate_results(pixel_idx, result)
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        temporal_frames: int = DEFAULT_TEMPORAL_FRAMES,
        roughness_mapping: Optional[RoughnessRayMapping] = None,
    ) -> None:
        """Initialize the adaptive ray scheduler.

        Args:
            width: Output width in pixels.
            height: Output height in pixels.
            temporal_frames: Number of frames for temporal accumulation.
            roughness_mapping: Roughness to ray parameter mapping.
        """
        self._width = width
        self._height = height
        self._temporal_frames = temporal_frames
        self._roughness_mapping = roughness_mapping or RoughnessRayMapping()
        self._current_frame = 0

        # Accumulation buffers
        self._accumulated: Dict[int, AccumulatedResult] = {}

        # Random state for stratified sampling
        self._rng_seed = 42

    @property
    def width(self) -> int:
        """Get output width."""
        return self._width

    @property
    def height(self) -> int:
        """Get output height."""
        return self._height

    @property
    def temporal_frames(self) -> int:
        """Get number of temporal accumulation frames."""
        return self._temporal_frames

    @temporal_frames.setter
    def temporal_frames(self, value: int) -> None:
        """Set temporal accumulation frames."""
        self._temporal_frames = max(1, value)

    @property
    def current_frame(self) -> int:
        """Get current frame index."""
        return self._current_frame

    def set_resolution(self, width: int, height: int) -> None:
        """Update output resolution.

        Args:
            width: New width.
            height: New height.
        """
        if width != self._width or height != self._height:
            self._width = width
            self._height = height
            self.reset_accumulation()

    def advance_frame(self) -> None:
        """Advance to next frame."""
        self._current_frame += 1

    def reset_accumulation(self) -> None:
        """Clear all accumulated results."""
        self._accumulated.clear()
        self._current_frame = 0

    def get_pixel_index(self, x: int, y: int) -> int:
        """Convert pixel coordinates to linear index.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            Linear pixel index.
        """
        return y * self._width + x

    def schedule_rays(
        self,
        pixel_x: int,
        pixel_y: int,
        roughness: float,
        use_stratification: bool = True,
    ) -> ScheduledRays:
        """Schedule rays for a pixel.

        Args:
            pixel_x: Pixel X coordinate.
            pixel_y: Pixel Y coordinate.
            roughness: Surface roughness.
            use_stratification: Whether to use temporal stratification.

        Returns:
            ScheduledRays with ray configuration.
        """
        mapping = self._roughness_mapping.get_mapping(roughness)

        if not mapping.should_trace:
            return ScheduledRays(
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                ray_count=0,
                frame_index=self._current_frame,
                accumulation_weight=0.0,
            )

        ray_count = mapping.ray_count

        # For temporal stratification, distribute rays across frames
        if use_stratification and self._temporal_frames > 1:
            frame_in_cycle = self._current_frame % self._temporal_frames
            # Weight based on accumulation history
            pixel_idx = self.get_pixel_index(pixel_x, pixel_y)
            if pixel_idx in self._accumulated:
                frames_so_far = self._accumulated[pixel_idx].frames_accumulated
                weight = 1.0 / max(1, frames_so_far + 1)
            else:
                weight = 1.0
        else:
            frame_in_cycle = 0
            weight = 1.0

        # Generate stratified ray directions if multiple rays
        directions: List[Vec3] = []
        if ray_count > 1:
            directions = self._generate_stratified_directions(
                pixel_x, pixel_y, ray_count, frame_in_cycle
            )

        return ScheduledRays(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            ray_count=ray_count,
            ray_directions=directions,
            frame_index=self._current_frame,
            accumulation_weight=weight,
        )

    def get_frame_rays(
        self, roughness_map: List[float], frame_index: Optional[int] = None
    ) -> List[ScheduledRays]:
        """Schedule rays for all pixels in a frame.

        Args:
            roughness_map: Roughness value per pixel.
            frame_index: Optional frame index override.

        Returns:
            List of ScheduledRays for each pixel.
        """
        if frame_index is not None:
            self._current_frame = frame_index

        result = []
        for y in range(self._height):
            for x in range(self._width):
                idx = y * self._width + x
                if idx < len(roughness_map):
                    roughness = roughness_map[idx]
                else:
                    roughness = 0.5

                scheduled = self.schedule_rays(x, y, roughness)
                result.append(scheduled)

        return result

    def accumulate_results(
        self,
        pixel_x: int,
        pixel_y: int,
        color: Vec3,
        hit_distance: float,
        confidence: float = 1.0,
    ) -> AccumulatedResult:
        """Accumulate a result for a pixel.

        Args:
            pixel_x: Pixel X coordinate.
            pixel_y: Pixel Y coordinate.
            color: New color sample.
            hit_distance: New hit distance.
            confidence: Confidence of sample.

        Returns:
            Updated accumulated result.
        """
        pixel_idx = self.get_pixel_index(pixel_x, pixel_y)

        if pixel_idx not in self._accumulated:
            self._accumulated[pixel_idx] = AccumulatedResult()

        acc = self._accumulated[pixel_idx]

        # Exponential moving average for smooth accumulation
        n = acc.sample_count + 1
        weight = 1.0 / n

        acc.color = Vec3(
            acc.color.x * (1 - weight) + color.x * weight,
            acc.color.y * (1 - weight) + color.y * weight,
            acc.color.z * (1 - weight) + color.z * weight,
        )
        acc.hit_distance = acc.hit_distance * (1 - weight) + hit_distance * weight
        acc.sample_count = n
        acc.confidence = min(1.0, acc.confidence + confidence * weight)
        acc.frames_accumulated += 1

        return acc

    def get_accumulated(self, pixel_x: int, pixel_y: int) -> AccumulatedResult:
        """Get accumulated result for a pixel.

        Args:
            pixel_x: Pixel X coordinate.
            pixel_y: Pixel Y coordinate.

        Returns:
            Accumulated result or empty result.
        """
        pixel_idx = self.get_pixel_index(pixel_x, pixel_y)
        return self._accumulated.get(pixel_idx, AccumulatedResult())

    def is_converged(self, pixel_x: int, pixel_y: int, min_samples: int = 4) -> bool:
        """Check if a pixel has converged.

        Args:
            pixel_x: Pixel X coordinate.
            pixel_y: Pixel Y coordinate.
            min_samples: Minimum samples for convergence.

        Returns:
            True if converged.
        """
        acc = self.get_accumulated(pixel_x, pixel_y)
        return acc.sample_count >= min_samples and acc.confidence >= 0.9

    def _generate_stratified_directions(
        self,
        pixel_x: int,
        pixel_y: int,
        ray_count: int,
        frame_index: int,
    ) -> List[Vec3]:
        """Generate stratified ray direction offsets.

        Args:
            pixel_x: Pixel X coordinate.
            pixel_y: Pixel Y coordinate.
            ray_count: Number of rays.
            frame_index: Current frame in cycle.

        Returns:
            List of direction offset vectors.
        """
        # Use deterministic random based on pixel and frame
        seed = (pixel_y * self._width + pixel_x) * 1000 + frame_index + self._rng_seed
        rng = random.Random(seed)

        directions = []
        sqrt_n = int(math.sqrt(ray_count))

        for i in range(ray_count):
            # Stratified jitter in a grid
            grid_x = i % sqrt_n
            grid_y = i // sqrt_n

            # Add jitter within cell
            jitter_x = (grid_x + rng.random()) / sqrt_n - 0.5
            jitter_y = (grid_y + rng.random()) / sqrt_n - 0.5

            # Small perturbation vector (not full direction - just offset)
            directions.append(Vec3(jitter_x * 0.1, jitter_y * 0.1, 0.0))

        return directions

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dict with scheduling statistics.
        """
        total_samples = sum(acc.sample_count for acc in self._accumulated.values())
        converged_count = sum(
            1 for idx in range(self._width * self._height)
            if self.is_converged(idx % self._width, idx // self._width)
        )

        return {
            "current_frame": self._current_frame,
            "temporal_frames": self._temporal_frames,
            "pixels_tracked": len(self._accumulated),
            "total_samples": total_samples,
            "converged_pixels": converged_count,
            "convergence_rate": converged_count / max(1, self._width * self._height),
        }


# =============================================================================
# Ray Budget Manager
# =============================================================================


@dataclass
class BudgetAllocation:
    """Ray budget allocation result.

    Attributes:
        tier_budgets: Ray budget per resolution tier.
        tier_pixel_counts: Pixel count per tier.
        total_rays: Total rays allocated.
        utilization: Budget utilization ratio.
    """

    tier_budgets: Dict[ResolutionTier, int] = field(default_factory=dict)
    tier_pixel_counts: Dict[ResolutionTier, int] = field(default_factory=dict)
    total_rays: int = 0
    utilization: float = 0.0


class RayBudgetManager:
    """Manages global ray budget distribution across pixels.

    Distributes a fixed ray budget based on:
    - Roughness distribution across the frame
    - Priority to smooth surfaces (fewer rays needed, high quality)
    - Budget cap to maintain frame time

    Usage:
        manager = RayBudgetManager(budget=2_000_000)
        allocation = manager.allocate_rays(roughness_distribution)
        rays_for_pixel = manager.get_rays_for_pixel(roughness)
    """

    def __init__(
        self,
        budget: int = DEFAULT_RAY_BUDGET,
        roughness_mapping: Optional[RoughnessRayMapping] = None,
    ) -> None:
        """Initialize the ray budget manager.

        Args:
            budget: Total ray budget per frame.
            roughness_mapping: Roughness to ray parameter mapping.
        """
        self._budget = budget
        self._roughness_mapping = roughness_mapping or RoughnessRayMapping()
        self._current_allocation: Optional[BudgetAllocation] = None
        self._budget_scale = 1.0

    @property
    def budget(self) -> int:
        """Get total ray budget."""
        return self._budget

    def set_budget(self, budget: int) -> None:
        """Set total ray budget.

        Args:
            budget: New budget value.
        """
        self._budget = max(0, budget)
        self._current_allocation = None

    @property
    def budget_scale(self) -> float:
        """Get current budget scale factor."""
        return self._budget_scale

    def get_utilization(self) -> float:
        """Get current budget utilization.

        Returns:
            Utilization ratio [0, 1].
        """
        if self._current_allocation is None:
            return 0.0
        return self._current_allocation.utilization

    def allocate_rays(
        self,
        roughness_distribution: List[float],
        width: int,
        height: int,
    ) -> BudgetAllocation:
        """Allocate rays based on roughness distribution.

        Args:
            roughness_distribution: Roughness value per pixel.
            width: Frame width.
            height: Frame height.

        Returns:
            BudgetAllocation with per-tier budgets.
        """
        # Count pixels per tier
        tier_counts: Dict[ResolutionTier, int] = {tier: 0 for tier in ResolutionTier}
        tier_ray_totals: Dict[ResolutionTier, int] = {tier: 0 for tier in ResolutionTier}

        for roughness in roughness_distribution:
            if not self._roughness_mapping.should_trace(roughness):
                continue

            mapping = self._roughness_mapping.get_mapping(roughness)
            tier = mapping.tier
            tier_counts[tier] += 1
            tier_ray_totals[tier] += mapping.ray_count

        # Calculate total rays needed
        total_needed = sum(tier_ray_totals.values())

        # Scale if over budget
        if total_needed > self._budget and total_needed > 0:
            self._budget_scale = self._budget / total_needed
        else:
            self._budget_scale = 1.0

        # Apply scaling to get actual allocation
        tier_budgets: Dict[ResolutionTier, int] = {}
        total_allocated = 0

        for tier in ResolutionTier:
            scaled = int(tier_ray_totals[tier] * self._budget_scale)
            tier_budgets[tier] = scaled
            total_allocated += scaled

        utilization = total_allocated / max(1, self._budget)

        self._current_allocation = BudgetAllocation(
            tier_budgets=tier_budgets,
            tier_pixel_counts=tier_counts,
            total_rays=total_allocated,
            utilization=utilization,
        )

        return self._current_allocation

    def get_rays_for_pixel(self, roughness: float) -> int:
        """Get scaled ray count for a pixel based on budget.

        Args:
            roughness: Surface roughness.

        Returns:
            Number of rays to trace (scaled by budget).
        """
        base_count = self._roughness_mapping.get_ray_count(roughness)
        return max(1, int(base_count * self._budget_scale)) if base_count > 0 else 0

    def estimate_frame_time(
        self, rays_per_second: float = 1_000_000_000.0
    ) -> float:
        """Estimate frame time based on allocated rays.

        Args:
            rays_per_second: Ray tracing throughput.

        Returns:
            Estimated time in milliseconds.
        """
        if self._current_allocation is None:
            return 0.0

        rays = self._current_allocation.total_rays
        return (rays / rays_per_second) * 1000.0

    def adjust_budget_for_target_time(
        self,
        target_ms: float,
        rays_per_second: float = 1_000_000_000.0,
    ) -> None:
        """Adjust budget to meet target frame time.

        Args:
            target_ms: Target time in milliseconds.
            rays_per_second: Ray tracing throughput.
        """
        max_rays = int((target_ms / 1000.0) * rays_per_second)
        self.set_budget(max_rays)

    def get_statistics(self) -> Dict[str, Any]:
        """Get budget manager statistics.

        Returns:
            Dict with budget statistics.
        """
        if self._current_allocation is None:
            return {
                "budget": self._budget,
                "allocated": 0,
                "utilization": 0.0,
                "scale": self._budget_scale,
            }

        return {
            "budget": self._budget,
            "allocated": self._current_allocation.total_rays,
            "utilization": self._current_allocation.utilization,
            "scale": self._budget_scale,
            "tier_budgets": {
                tier.name: count
                for tier, count in self._current_allocation.tier_budgets.items()
            },
            "tier_pixels": {
                tier.name: count
                for tier, count in self._current_allocation.tier_pixel_counts.items()
            },
        }


# =============================================================================
# Adaptive RT Configuration
# =============================================================================


@dataclass
class AdaptiveRTConfig:
    """Configuration for adaptive ray tracing.

    Attributes:
        min_rays: Minimum rays per pixel (smooth surfaces).
        max_rays: Maximum rays per pixel (rough surfaces).
        roughness_thresholds: Tier boundary roughness values.
        temporal_accumulation: Frames for temporal accumulation.
        ray_budget: Total ray budget per frame.
        enable_stratification: Enable temporal stratification.
        smooth_tier_transitions: Interpolate between tiers.
        denoise_enabled: Enable denoising.
    """

    min_rays: int = 1
    max_rays: int = 8
    roughness_thresholds: List[float] = field(
        default_factory=lambda: list(DEFAULT_TIER_THRESHOLDS)
    )
    temporal_accumulation: int = DEFAULT_TEMPORAL_FRAMES
    ray_budget: int = DEFAULT_RAY_BUDGET
    enable_stratification: bool = True
    smooth_tier_transitions: bool = False
    denoise_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.min_rays = max(1, self.min_rays)
        self.max_rays = max(self.min_rays, self.max_rays)
        self.temporal_accumulation = max(1, self.temporal_accumulation)
        self.ray_budget = max(1000, self.ray_budget)

        # Ensure thresholds are sorted and in valid range
        self.roughness_thresholds = sorted(
            max(0.0, min(1.0, t)) for t in self.roughness_thresholds
        )

    def validate(self) -> List[str]:
        """Validate configuration and return errors.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []

        if self.min_rays < 1:
            errors.append("min_rays must be at least 1")

        if self.max_rays < self.min_rays:
            errors.append("max_rays must be >= min_rays")

        if len(self.roughness_thresholds) < 2:
            errors.append("roughness_thresholds must have at least 2 values")

        if self.temporal_accumulation < 1:
            errors.append("temporal_accumulation must be at least 1")

        if self.ray_budget < 1000:
            errors.append("ray_budget should be at least 1000")

        return errors

    def create_roughness_mapping(self) -> RoughnessRayMapping:
        """Create a RoughnessRayMapping from this config.

        Returns:
            Configured RoughnessRayMapping.
        """
        # Generate ray counts based on min/max
        n_tiers = len(self.roughness_thresholds)
        ray_counts = []
        for i in range(n_tiers):
            t = i / max(1, n_tiers - 1)
            count = int(self.min_rays + t * (self.max_rays - self.min_rays))
            ray_counts.append(max(1, count))

        return RoughnessRayMapping(
            tier_thresholds=self.roughness_thresholds,
            ray_counts=ray_counts,
        )

    @staticmethod
    def low_quality() -> "AdaptiveRTConfig":
        """Low quality preset (fast, lower quality)."""
        return AdaptiveRTConfig(
            min_rays=1,
            max_rays=2,
            temporal_accumulation=2,
            ray_budget=500_000,
            denoise_enabled=True,
        )

    @staticmethod
    def medium_quality() -> "AdaptiveRTConfig":
        """Medium quality preset (balanced)."""
        return AdaptiveRTConfig(
            min_rays=1,
            max_rays=4,
            temporal_accumulation=4,
            ray_budget=1_000_000,
            denoise_enabled=True,
        )

    @staticmethod
    def high_quality() -> "AdaptiveRTConfig":
        """High quality preset (best quality, slower)."""
        return AdaptiveRTConfig(
            min_rays=1,
            max_rays=8,
            temporal_accumulation=8,
            ray_budget=2_000_000,
            enable_stratification=True,
            smooth_tier_transitions=True,
            denoise_enabled=True,
        )


# =============================================================================
# Adaptive RT Pass Integration
# =============================================================================


@dataclass
class AdaptiveRayResult:
    """Result from adaptive ray tracing for a pixel.

    Attributes:
        color: Final reflected color.
        hit_distance: Distance to reflection.
        confidence: Result confidence.
        rays_traced: Number of rays traced.
        resolution_scale: Resolution scale used.
        denoise_strength: Denoising strength applied.
    """

    color: Vec3 = field(default_factory=Vec3.zero)
    hit_distance: float = 0.0
    confidence: float = 0.0
    rays_traced: int = 0
    resolution_scale: float = 1.0
    denoise_strength: float = 0.0


class AdaptiveRTPass:
    """Adaptive ray tracing pass with roughness-based adaptation.

    Integrates:
    - RoughnessRayMapping for parameter lookup
    - ResolutionHierarchy for multi-resolution rendering
    - AdaptiveRayScheduler for temporal stratification
    - RayBudgetManager for budget control

    Usage:
        config = AdaptiveRTConfig.medium_quality()
        adaptive_pass = AdaptiveRTPass(config, width=1920, height=1080)
        adaptive_pass.configure_pass(rt_pass)
        result = adaptive_pass.execute_adaptive(roughness_map)
    """

    def __init__(
        self,
        config: AdaptiveRTConfig,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        """Initialize the adaptive RT pass.

        Args:
            config: Adaptive RT configuration.
            width: Output width.
            height: Output height.
        """
        self._config = config
        self._width = width
        self._height = height

        # Create components
        self._roughness_mapping = config.create_roughness_mapping()
        self._resolution_hierarchy = ResolutionHierarchy(width, height)
        self._scheduler = AdaptiveRayScheduler(
            width, height, config.temporal_accumulation, self._roughness_mapping
        )
        self._budget_manager = RayBudgetManager(
            config.ray_budget, self._roughness_mapping
        )

        # Output buffer
        self._results: List[AdaptiveRayResult] = []

        # Statistics
        self._total_rays_traced = 0
        self._pixels_processed = 0

    @property
    def config(self) -> AdaptiveRTConfig:
        """Get configuration."""
        return self._config

    @property
    def width(self) -> int:
        """Get output width."""
        return self._width

    @property
    def height(self) -> int:
        """Get output height."""
        return self._height

    @property
    def roughness_mapping(self) -> RoughnessRayMapping:
        """Get roughness mapping."""
        return self._roughness_mapping

    @property
    def resolution_hierarchy(self) -> ResolutionHierarchy:
        """Get resolution hierarchy."""
        return self._resolution_hierarchy

    @property
    def scheduler(self) -> AdaptiveRayScheduler:
        """Get ray scheduler."""
        return self._scheduler

    @property
    def budget_manager(self) -> RayBudgetManager:
        """Get budget manager."""
        return self._budget_manager

    def set_resolution(self, width: int, height: int) -> None:
        """Update output resolution.

        Args:
            width: New width.
            height: New height.
        """
        self._width = width
        self._height = height
        self._resolution_hierarchy.set_base_resolution(width, height)
        self._scheduler.set_resolution(width, height)

    def configure_pass(self, rt_pass: Any) -> None:
        """Configure an RT pass with adaptive settings.

        Args:
            rt_pass: RTReflectionPass to configure.
        """
        # This would configure the RT pass with adaptive settings
        # In practice, would set resolution scale, enable features, etc.
        pass

    def execute_adaptive(
        self,
        roughness_map: List[float],
        trace_func: Optional[Callable[[int, int, int], Vec3]] = None,
    ) -> List[AdaptiveRayResult]:
        """Execute adaptive ray tracing.

        Args:
            roughness_map: Per-pixel roughness values.
            trace_func: Optional function to trace rays (x, y, ray_count) -> color.

        Returns:
            List of AdaptiveRayResult per pixel.
        """
        # Reset statistics
        self._total_rays_traced = 0
        self._pixels_processed = 0

        # Allocate budget
        self._budget_manager.allocate_rays(roughness_map, self._width, self._height)

        # Schedule rays
        scheduled_list = self._scheduler.get_frame_rays(roughness_map)

        # Process each pixel
        self._results = []

        for i, scheduled in enumerate(scheduled_list):
            x = scheduled.pixel_x
            y = scheduled.pixel_y

            if i < len(roughness_map):
                roughness = roughness_map[i]
            else:
                roughness = 0.5

            mapping = self._roughness_mapping.get_mapping(roughness)

            if not mapping.should_trace or scheduled.ray_count == 0:
                self._results.append(
                    AdaptiveRayResult(
                        color=Vec3.zero(),
                        rays_traced=0,
                        resolution_scale=0.0,
                        denoise_strength=1.0,
                    )
                )
                continue

            # Get scaled ray count from budget manager
            ray_count = self._budget_manager.get_rays_for_pixel(roughness)

            # Trace rays (use provided function or simulate)
            if trace_func is not None:
                color = trace_func(x, y, ray_count)
            else:
                # Simulate result for testing
                color = Vec3(0.5, 0.5, 0.5)

            self._total_rays_traced += ray_count
            self._pixels_processed += 1

            # Accumulate result
            self._scheduler.accumulate_results(x, y, color, 10.0, 1.0)
            accumulated = self._scheduler.get_accumulated(x, y)

            self._results.append(
                AdaptiveRayResult(
                    color=accumulated.color,
                    hit_distance=accumulated.hit_distance,
                    confidence=accumulated.confidence,
                    rays_traced=ray_count,
                    resolution_scale=mapping.resolution_scale,
                    denoise_strength=mapping.denoise_strength,
                )
            )

        # Advance frame for temporal accumulation
        self._scheduler.advance_frame()

        return self._results

    def get_result_at(self, x: int, y: int) -> AdaptiveRayResult:
        """Get result at pixel coordinates.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            AdaptiveRayResult at that pixel.
        """
        if not self._results:
            return AdaptiveRayResult()

        idx = y * self._width + x
        if 0 <= idx < len(self._results):
            return self._results[idx]

        return AdaptiveRayResult()

    def get_statistics(self) -> Dict[str, Any]:
        """Get pass statistics.

        Returns:
            Dict with execution statistics.
        """
        return {
            "total_rays_traced": self._total_rays_traced,
            "pixels_processed": self._pixels_processed,
            "average_rays_per_pixel": (
                self._total_rays_traced / max(1, self._pixels_processed)
            ),
            "budget_utilization": self._budget_manager.get_utilization(),
            "scheduler_stats": self._scheduler.get_statistics(),
            "budget_stats": self._budget_manager.get_statistics(),
        }


# =============================================================================
# WGSL Shader Generation
# =============================================================================


def generate_adaptive_rays_wgsl(config: AdaptiveRTConfig) -> str:
    """Generate WGSL shader for adaptive ray count determination.

    Args:
        config: Adaptive RT configuration.

    Returns:
        WGSL shader source.
    """
    thresholds = config.roughness_thresholds
    threshold_str = ", ".join(f"{t:.3f}" for t in thresholds)

    return f"""// Adaptive Ray Count Shader (rt_adaptive_rays.wgsl)
// Generated for T-GIR-P8.3 Roughness-Based Ray Count Adaptation

// Configuration
const MIN_RAYS: u32 = {config.min_rays}u;
const MAX_RAYS: u32 = {config.max_rays}u;
const TEMPORAL_FRAMES: u32 = {config.temporal_accumulation}u;
const ROUGHNESS_THRESHOLDS: array<f32, {len(thresholds)}> = array<f32, {len(thresholds)}>({threshold_str});

struct AdaptiveParams {{
    ray_count: u32,
    resolution_scale: f32,
    denoise_strength: f32,
    should_trace: u32,
}}

// Get tier index for roughness
fn get_tier_index(roughness: f32) -> u32 {{
    for (var i: u32 = 0u; i < {len(thresholds)}u; i++) {{
        if (roughness <= ROUGHNESS_THRESHOLDS[i]) {{
            return i;
        }}
    }}
    return {len(thresholds) - 1}u;
}}

// Get ray count for roughness
fn get_ray_count(roughness: f32) -> u32 {{
    if (roughness > 0.7) {{
        return 0u;
    }}
    let tier = get_tier_index(roughness);
    let t = f32(tier) / f32({len(thresholds) - 1}u);
    return u32(mix(f32(MIN_RAYS), f32(MAX_RAYS), t));
}}

// Get resolution scale for roughness
fn get_resolution_scale(roughness: f32) -> f32 {{
    if (roughness <= 0.1) {{
        return 1.0;
    }} else if (roughness <= 0.3) {{
        return 0.5;
    }} else {{
        return 0.25;
    }}
}}

// Get denoise strength for roughness
fn get_denoise_strength(roughness: f32) -> f32 {{
    if (roughness <= 0.1) {{
        return 0.0;
    }} else if (roughness <= 0.3) {{
        return 0.25;
    }} else if (roughness <= 0.5) {{
        return 0.5;
    }} else {{
        return 1.0;
    }}
}}

// Get complete adaptive parameters
fn get_adaptive_params(roughness: f32) -> AdaptiveParams {{
    var params: AdaptiveParams;
    params.ray_count = get_ray_count(roughness);
    params.resolution_scale = get_resolution_scale(roughness);
    params.denoise_strength = get_denoise_strength(roughness);
    params.should_trace = select(0u, 1u, roughness <= 0.7);
    return params;
}}

// Compute shader for adaptive ray scheduling
@group(0) @binding(0) var<storage, read> roughness_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> ray_counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> resolution_scales: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec2<u32>;

@compute @workgroup_size(8, 8, 1)
fn schedule_rays(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {{
        return;
    }}

    let idx = global_id.y * dimensions.x + global_id.x;
    let roughness = roughness_buffer[idx];
    let params = get_adaptive_params(roughness);

    ray_counts[idx] = params.ray_count;
    resolution_scales[idx] = params.resolution_scale;
}}
"""


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_adaptive_memory(
    width: int,
    height: int,
    config: AdaptiveRTConfig,
) -> int:
    """Estimate memory usage for adaptive RT.

    Args:
        width: Output width.
        height: Output height.
        config: Adaptive RT configuration.

    Returns:
        Memory estimate in bytes.
    """
    base_pixels = width * height

    # Per-pixel storage:
    # - Color accumulation: RGBA32F (16 bytes)
    # - Hit distance: R32F (4 bytes)
    # - Confidence: R32F (4 bytes)
    # - Sample count: U32 (4 bytes)
    bytes_per_pixel = 16 + 4 + 4 + 4

    # Temporal history (N frames)
    temporal_bytes = bytes_per_pixel * base_pixels * config.temporal_accumulation

    # Resolution hierarchy (sum of all levels)
    hierarchy = ResolutionHierarchy(width, height)
    hierarchy_bytes = hierarchy.estimate_memory_usage()

    return temporal_bytes + hierarchy_bytes


def create_test_roughness_map(
    width: int,
    height: int,
    pattern: str = "gradient",
) -> List[float]:
    """Create a test roughness map.

    Args:
        width: Map width.
        height: Map height.
        pattern: Pattern type ("gradient", "random", "uniform", "bands").

    Returns:
        List of roughness values.
    """
    result = []

    for y in range(height):
        for x in range(width):
            if pattern == "gradient":
                # Horizontal gradient from smooth to rough
                roughness = x / max(1, width - 1) * 0.7
            elif pattern == "random":
                roughness = random.random() * 0.7
            elif pattern == "uniform":
                roughness = 0.3
            elif pattern == "bands":
                # Vertical bands of different roughness
                band = (x * 4) // width
                roughness = [0.0, 0.2, 0.4, 0.6][band % 4]
            else:
                roughness = 0.5

            result.append(roughness)

    return result


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Constants
    "DEFAULT_TIER_THRESHOLDS",
    "DEFAULT_RESOLUTION_SCALES",
    "DEFAULT_RAY_COUNTS",
    "DEFAULT_DENOISE_STRENGTHS",
    "DEFAULT_RAY_BUDGET",
    "DEFAULT_TEMPORAL_FRAMES",
    "MAX_RT_ROUGHNESS",
    # Enums
    "ResolutionTier",
    "DenoiseLevel",
    # Data structures
    "RoughnessMapping",
    "ResolutionLevel",
    "ScheduledRays",
    "AccumulatedResult",
    "BudgetAllocation",
    "AdaptiveRayResult",
    # Core classes
    "RoughnessRayMapping",
    "ResolutionHierarchy",
    "AdaptiveRayScheduler",
    "RayBudgetManager",
    "AdaptiveRTPass",
    # Config
    "AdaptiveRTConfig",
    # Utilities
    "generate_adaptive_rays_wgsl",
    "estimate_adaptive_memory",
    "create_test_roughness_map",
]
