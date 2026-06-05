"""Per-Pixel Reflection Probe Blending System.

Implements smooth transitions between multiple reflection probes through:
- Per-pixel influence weight calculation (distance, normal alignment, visibility)
- Probe collection and priority sorting
- Weight normalization to sum to 1.0
- Multi-probe cubemap sampling and blending
- Full-screen probe blend pass

Reference: RENDERING_CONTEXT.md Section 6.4 - Reflection Probe System
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, List

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3
from engine.rendering.lighting.baked_probes import (
    CubemapData,
    CubemapFace,
    HDRPixel,
)
from engine.rendering.lighting.reflection_probes import (
    RealtimeReflectionProbe,
    RealtimeProbeManager,
)

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

class ProbeBlendConstants:
    """Constants for probe blending."""
    # Default maximum probes per pixel
    DEFAULT_MAX_PROBES: int = 4
    # Maximum allowed probes per pixel
    MAX_PROBES_LIMIT: int = 16
    # Minimum weight threshold (below this, probe is discarded)
    MIN_WEIGHT_THRESHOLD: float = 0.001
    # Default blend distance (transition zone size)
    DEFAULT_BLEND_DISTANCE: float = 2.0
    # Minimum blend distance
    MIN_BLEND_DISTANCE: float = 0.1
    # Maximum blend distance
    MAX_BLEND_DISTANCE: float = 50.0
    # Default normal weight factor
    DEFAULT_NORMAL_WEIGHT: float = 0.3
    # Default visibility weight factor
    DEFAULT_VISIBILITY_WEIGHT: float = 0.2
    # Epsilon for numerical stability
    EPSILON: float = 1e-6


class FalloffType(Enum):
    """Distance falloff calculation types."""
    LINEAR = auto()      # 1 - distance/max_distance
    QUADRATIC = auto()   # 1 - (distance/max_distance)^2
    SMOOTH = auto()      # Smoothstep-based falloff
    INVERSE = auto()     # 1 / (1 + distance)
    EXPONENTIAL = auto() # exp(-distance * k)


# -----------------------------------------------------------------------------
# Probe Influence Calculator
# -----------------------------------------------------------------------------

@dataclass
class ProbeInfluence:
    """Calculates influence weight for a probe at a world position.

    Combines distance-based falloff, normal alignment, and visibility
    to compute a final influence weight.

    Attributes:
        probe: The reflection probe
        world_position: Position being shaded
        surface_normal: Surface normal at position (optional)
        is_visible: Whether probe is visible from position
        falloff_type: Type of distance falloff
        blend_distance: Transition zone size
        normal_weight: Weight factor for normal alignment (0-1)
        visibility_weight: Weight factor for visibility (0-1)
    """
    probe: RealtimeReflectionProbe
    world_position: Vec3
    surface_normal: Optional[Vec3] = None
    is_visible: bool = True
    falloff_type: FalloffType = FalloffType.SMOOTH
    blend_distance: float = ProbeBlendConstants.DEFAULT_BLEND_DISTANCE
    normal_weight: float = ProbeBlendConstants.DEFAULT_NORMAL_WEIGHT
    visibility_weight: float = ProbeBlendConstants.DEFAULT_VISIBILITY_WEIGHT

    _cached_weight: Optional[float] = field(default=None, repr=False)
    _cached_distance: Optional[float] = field(default=None, repr=False)

    def calculate_weight(self) -> float:
        """Calculate the total influence weight.

        Combines distance falloff, normal alignment, and visibility
        into a single weight value.

        Returns:
            Influence weight (0-1), higher means more influence
        """
        if self._cached_weight is not None:
            return self._cached_weight

        # Check if point is inside probe bounds
        if not self.probe.contains(self.world_position):
            self._cached_weight = 0.0
            return 0.0

        # Calculate component weights
        dist_weight = self.distance_falloff()
        normal_weight = self.normal_alignment()
        vis_weight = self.visibility_factor()

        # Combine weights: distance is primary, others are modifiers
        # Formula: dist * (1 - normal_factor + normal_factor * normal_align) * (1 - vis_factor + vis_factor * vis)
        normal_modifier = (1.0 - self.normal_weight) + self.normal_weight * normal_weight
        vis_modifier = (1.0 - self.visibility_weight) + self.visibility_weight * vis_weight

        total = dist_weight * normal_modifier * vis_modifier
        self._cached_weight = max(0.0, min(1.0, total))
        return self._cached_weight

    def distance_falloff(self) -> float:
        """Calculate distance-based falloff weight.

        Uses the configured falloff type to compute how much
        the probe influences based on distance from center.

        Returns:
            Falloff weight (0-1), 1 at center, decreasing with distance
        """
        probe_center = self.probe.bounds.center
        extent = self.probe.bounds.max - self.probe.bounds.min
        max_dist = extent.length() * 0.5

        # Clamp max_dist to avoid division by zero
        max_dist = max(max_dist, ProbeBlendConstants.EPSILON)

        # Calculate distance from center
        if self._cached_distance is None:
            self._cached_distance = self.world_position.distance(probe_center)
        distance = self._cached_distance

        # Normalize distance
        normalized_dist = min(distance / max_dist, 1.0)

        # Apply falloff based on type
        if self.falloff_type == FalloffType.LINEAR:
            return max(0.0, 1.0 - normalized_dist)

        elif self.falloff_type == FalloffType.QUADRATIC:
            return max(0.0, 1.0 - normalized_dist * normalized_dist)

        elif self.falloff_type == FalloffType.SMOOTH:
            # Smoothstep: 3t^2 - 2t^3 where t = 1 - normalized_dist
            t = 1.0 - normalized_dist
            return t * t * (3.0 - 2.0 * t)

        elif self.falloff_type == FalloffType.INVERSE:
            return 1.0 / (1.0 + distance / self.blend_distance)

        elif self.falloff_type == FalloffType.EXPONENTIAL:
            k = 1.0 / max(self.blend_distance, ProbeBlendConstants.EPSILON)
            return math.exp(-distance * k)

        return 0.0

    def normal_alignment(self) -> float:
        """Calculate normal alignment factor.

        Probes facing the surface normal have higher influence.
        Uses dot product between surface normal and direction to probe.

        Returns:
            Alignment factor (0-1), 1 when aligned, 0 when perpendicular
        """
        if self.surface_normal is None:
            return 1.0  # No normal provided, full weight

        # Direction from position to probe center
        probe_center = self.probe.bounds.center
        to_probe = probe_center - self.world_position
        dist = to_probe.length()

        if dist < ProbeBlendConstants.EPSILON:
            return 1.0  # At probe center

        to_probe_normalized = to_probe * (1.0 / dist)
        normal_normalized = self.surface_normal.normalized()

        # Dot product: positive when normal faces probe
        dot = normal_normalized.dot(to_probe_normalized)

        # Remap from [-1, 1] to [0, 1]
        # Probes behind surface (dot < 0) get reduced weight
        return max(0.0, (dot + 1.0) * 0.5)

    def visibility_factor(self) -> float:
        """Calculate visibility factor.

        Returns 1.0 if visible, 0.0 if not visible.
        Could be extended for partial visibility (occlusion).

        Returns:
            Visibility factor (0-1)
        """
        return 1.0 if self.is_visible else 0.0

    def get_distance(self) -> float:
        """Get distance from position to probe center.

        Returns:
            Distance in world units
        """
        if self._cached_distance is None:
            self._cached_distance = self.world_position.distance(self.probe.bounds.center)
        return self._cached_distance

    def invalidate_cache(self) -> None:
        """Invalidate cached calculations."""
        self._cached_weight = None
        self._cached_distance = None


# -----------------------------------------------------------------------------
# Probe Collector
# -----------------------------------------------------------------------------

@dataclass
class ProbeCollectorConfig:
    """Configuration for probe collection.

    Attributes:
        max_probes: Maximum number of probes to return
        min_weight: Minimum weight threshold
        falloff_type: Default falloff type for influences
        blend_distance: Default blend distance
        normal_weight: Default normal weight factor
        visibility_weight: Default visibility weight factor
    """
    max_probes: int = ProbeBlendConstants.DEFAULT_MAX_PROBES
    min_weight: float = ProbeBlendConstants.MIN_WEIGHT_THRESHOLD
    falloff_type: FalloffType = FalloffType.SMOOTH
    blend_distance: float = ProbeBlendConstants.DEFAULT_BLEND_DISTANCE
    normal_weight: float = ProbeBlendConstants.DEFAULT_NORMAL_WEIGHT
    visibility_weight: float = ProbeBlendConstants.DEFAULT_VISIBILITY_WEIGHT

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.max_probes = max(1, min(self.max_probes, ProbeBlendConstants.MAX_PROBES_LIMIT))
        self.min_weight = max(0.0, min(self.min_weight, 0.5))
        self.blend_distance = max(
            ProbeBlendConstants.MIN_BLEND_DISTANCE,
            min(self.blend_distance, ProbeBlendConstants.MAX_BLEND_DISTANCE)
        )
        self.normal_weight = max(0.0, min(self.normal_weight, 1.0))
        self.visibility_weight = max(0.0, min(self.visibility_weight, 1.0))


class ProbeCollector:
    """Collects and sorts probes influencing a world position.

    Finds all probes that contain the position, calculates their
    influence weights, and returns the top N most influential probes.
    """

    def __init__(
        self,
        probe_manager: RealtimeProbeManager,
        config: Optional[ProbeCollectorConfig] = None,
    ) -> None:
        """Initialize probe collector.

        Args:
            probe_manager: Manager containing registered probes
            config: Collection configuration
        """
        self._probe_manager = probe_manager
        self._config = config or ProbeCollectorConfig()

    @property
    def config(self) -> ProbeCollectorConfig:
        """Get collector configuration."""
        return self._config

    @config.setter
    def config(self, value: ProbeCollectorConfig) -> None:
        """Set collector configuration."""
        self._config = value

    @property
    def probe_manager(self) -> RealtimeProbeManager:
        """Get probe manager."""
        return self._probe_manager

    def collect_probes(
        self,
        world_position: Vec3,
        surface_normal: Optional[Vec3] = None,
        visibility_func: Optional[Callable[[RealtimeReflectionProbe, Vec3], bool]] = None,
    ) -> List[ProbeInfluence]:
        """Collect all probes influencing a position.

        Args:
            world_position: Position being shaded
            surface_normal: Surface normal at position (optional)
            visibility_func: Optional function to check probe visibility

        Returns:
            List of ProbeInfluence objects, unsorted
        """
        influences = []

        for probe in self._probe_manager.get_all_probes():
            # Skip probes that don't contain the position
            if not probe.contains(world_position):
                continue

            # Check visibility if function provided
            is_visible = True
            if visibility_func is not None:
                is_visible = visibility_func(probe, world_position)

            # Create influence object
            influence = ProbeInfluence(
                probe=probe,
                world_position=world_position,
                surface_normal=surface_normal,
                is_visible=is_visible,
                falloff_type=self._config.falloff_type,
                blend_distance=self._config.blend_distance,
                normal_weight=self._config.normal_weight,
                visibility_weight=self._config.visibility_weight,
            )

            # Calculate weight and check threshold
            weight = influence.calculate_weight()
            if weight >= self._config.min_weight:
                influences.append(influence)

        return influences

    def sort_by_influence(
        self,
        influences: List[ProbeInfluence],
    ) -> List[ProbeInfluence]:
        """Sort influences by weight (descending).

        Args:
            influences: List of probe influences

        Returns:
            Sorted list (highest weight first)
        """
        return sorted(influences, key=lambda inf: inf.calculate_weight(), reverse=True)

    def get_top_n(
        self,
        world_position: Vec3,
        n: Optional[int] = None,
        surface_normal: Optional[Vec3] = None,
        visibility_func: Optional[Callable[[RealtimeReflectionProbe, Vec3], bool]] = None,
    ) -> List[ProbeInfluence]:
        """Get top N most influential probes.

        Args:
            world_position: Position being shaded
            n: Number of probes (defaults to config.max_probes)
            surface_normal: Surface normal at position
            visibility_func: Optional visibility check function

        Returns:
            List of top N ProbeInfluence objects, sorted by weight
        """
        n = n if n is not None else self._config.max_probes

        # Collect all influences
        influences = self.collect_probes(world_position, surface_normal, visibility_func)

        # Sort by weight
        sorted_influences = self.sort_by_influence(influences)

        # Return top N
        return sorted_influences[:n]


# -----------------------------------------------------------------------------
# Probe Blender
# -----------------------------------------------------------------------------

@dataclass
class BlendResult:
    """Result of probe blending operation.

    Attributes:
        color: Blended HDR color
        weight_sum: Sum of all weights before normalization
        probe_count: Number of probes that contributed
        dominant_probe: Probe with highest weight (if any)
    """
    color: HDRPixel
    weight_sum: float
    probe_count: int
    dominant_probe: Optional[RealtimeReflectionProbe] = None


class ProbeBlender:
    """Blends samples from multiple reflection probes.

    Normalizes weights to sum to 1.0 and blends cubemap samples
    from multiple probes to produce a final reflection color.
    """

    def __init__(self) -> None:
        """Initialize probe blender."""
        pass

    def normalize_weights(
        self,
        influences: List[ProbeInfluence],
    ) -> List[Tuple[ProbeInfluence, float]]:
        """Normalize influence weights to sum to 1.0.

        Args:
            influences: List of probe influences

        Returns:
            List of (influence, normalized_weight) tuples
        """
        if not influences:
            return []

        # Calculate total weight
        total_weight = sum(inf.calculate_weight() for inf in influences)

        if total_weight < ProbeBlendConstants.EPSILON:
            # All weights near zero, distribute equally
            equal_weight = 1.0 / len(influences)
            return [(inf, equal_weight) for inf in influences]

        # Normalize weights
        inv_total = 1.0 / total_weight
        return [(inf, inf.calculate_weight() * inv_total) for inf in influences]

    def blend_samples(
        self,
        influences: List[ProbeInfluence],
        direction: Vec3,
        roughness: float = 0.0,
    ) -> BlendResult:
        """Blend cubemap samples from multiple probes.

        Args:
            influences: List of probe influences
            direction: Sample direction (normalized)
            roughness: Surface roughness (0-1)

        Returns:
            BlendResult with final color and metadata
        """
        # Handle edge cases
        if not influences:
            return BlendResult(
                color=HDRPixel(0.0, 0.0, 0.0),
                weight_sum=0.0,
                probe_count=0,
                dominant_probe=None,
            )

        # Single probe: no blending needed
        if len(influences) == 1:
            inf = influences[0]
            sample = inf.probe.sample(direction, roughness)
            return BlendResult(
                color=HDRPixel(sample.x, sample.y, sample.z),
                weight_sum=inf.calculate_weight(),
                probe_count=1,
                dominant_probe=inf.probe,
            )

        # Normalize weights
        normalized = self.normalize_weights(influences)

        # Calculate total weight before normalization
        total_weight = sum(inf.calculate_weight() for inf in influences)

        # Blend samples
        result_r = 0.0
        result_g = 0.0
        result_b = 0.0
        max_weight = 0.0
        dominant_probe = None

        for inf, weight in normalized:
            sample = inf.probe.sample(direction, roughness)
            result_r += sample.x * weight
            result_g += sample.y * weight
            result_b += sample.z * weight

            # Track dominant probe
            if weight > max_weight:
                max_weight = weight
                dominant_probe = inf.probe

        return BlendResult(
            color=HDRPixel(result_r, result_g, result_b),
            weight_sum=total_weight,
            probe_count=len(influences),
            dominant_probe=dominant_probe,
        )

    def sample_blended(
        self,
        collector: ProbeCollector,
        world_position: Vec3,
        direction: Vec3,
        roughness: float = 0.0,
        surface_normal: Optional[Vec3] = None,
    ) -> BlendResult:
        """Sample blended probe color at a position.

        Convenience method that combines collection and blending.

        Args:
            collector: Probe collector
            world_position: Position being shaded
            direction: Sample direction
            roughness: Surface roughness
            surface_normal: Surface normal (optional)

        Returns:
            BlendResult with blended color
        """
        influences = collector.get_top_n(world_position, surface_normal=surface_normal)
        return self.blend_samples(influences, direction, roughness)


# -----------------------------------------------------------------------------
# Probe Blend Configuration
# -----------------------------------------------------------------------------

@dataclass
class ProbeBlendConfig:
    """Configuration for probe blending system.

    Attributes:
        max_probes: Maximum probes per pixel (default 4)
        distance_falloff_type: Type of distance falloff
        normal_weight: Weight factor for normal alignment (0-1)
        visibility_weight: Weight factor for visibility (0-1)
        blend_distance: Transition zone size
        min_weight_threshold: Minimum weight to include probe
        roughness_levels: Number of roughness mip levels
    """
    max_probes: int = ProbeBlendConstants.DEFAULT_MAX_PROBES
    distance_falloff_type: FalloffType = FalloffType.SMOOTH
    normal_weight: float = ProbeBlendConstants.DEFAULT_NORMAL_WEIGHT
    visibility_weight: float = ProbeBlendConstants.DEFAULT_VISIBILITY_WEIGHT
    blend_distance: float = ProbeBlendConstants.DEFAULT_BLEND_DISTANCE
    min_weight_threshold: float = ProbeBlendConstants.MIN_WEIGHT_THRESHOLD
    roughness_levels: int = 6

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.max_probes = max(1, min(self.max_probes, ProbeBlendConstants.MAX_PROBES_LIMIT))
        self.normal_weight = max(0.0, min(self.normal_weight, 1.0))
        self.visibility_weight = max(0.0, min(self.visibility_weight, 1.0))
        self.blend_distance = max(
            ProbeBlendConstants.MIN_BLEND_DISTANCE,
            min(self.blend_distance, ProbeBlendConstants.MAX_BLEND_DISTANCE)
        )
        self.min_weight_threshold = max(0.0, min(self.min_weight_threshold, 0.5))
        self.roughness_levels = max(1, min(self.roughness_levels, 12))

    def to_collector_config(self) -> ProbeCollectorConfig:
        """Convert to ProbeCollectorConfig.

        Returns:
            Equivalent ProbeCollectorConfig
        """
        return ProbeCollectorConfig(
            max_probes=self.max_probes,
            min_weight=self.min_weight_threshold,
            falloff_type=self.distance_falloff_type,
            blend_distance=self.blend_distance,
            normal_weight=self.normal_weight,
            visibility_weight=self.visibility_weight,
        )


# -----------------------------------------------------------------------------
# Probe Blend Pass
# -----------------------------------------------------------------------------

@dataclass
class GBufferSample:
    """Sample from G-buffer at a pixel.

    Attributes:
        position: World position
        normal: Surface normal (normalized)
        roughness: Surface roughness (0-1)
        metallic: Surface metallic (0-1)
        is_valid: Whether sample is valid (not sky)
    """
    position: Vec3
    normal: Vec3
    roughness: float
    metallic: float
    is_valid: bool = True


@dataclass
class ReflectionBuffer:
    """Buffer storing reflection results.

    Attributes:
        width: Buffer width in pixels
        height: Buffer height in pixels
        data: Pixel data as list of HDRPixel
    """
    width: int
    height: int
    data: List[HDRPixel] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize buffer data."""
        if not self.data:
            self.data = [HDRPixel(0.0, 0.0, 0.0) for _ in range(self.width * self.height)]

    def get_pixel(self, x: int, y: int) -> HDRPixel:
        """Get pixel at coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y * self.width + x]
        return HDRPixel(0.0, 0.0, 0.0)

    def set_pixel(self, x: int, y: int, color: HDRPixel) -> None:
        """Set pixel at coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y * self.width + x] = color

    def clear(self, color: Optional[HDRPixel] = None) -> None:
        """Clear buffer to a color."""
        clear_color = color or HDRPixel(0.0, 0.0, 0.0)
        self.data = [HDRPixel(clear_color.r, clear_color.g, clear_color.b)
                     for _ in range(self.width * self.height)]


class ProbeBlendPass:
    """Full-screen pass computing blended probe reflections.

    Reads G-buffer data (position, normal, roughness) and outputs
    blended reflection colors from multiple probes.
    """

    def __init__(
        self,
        probe_manager: RealtimeProbeManager,
        config: Optional[ProbeBlendConfig] = None,
    ) -> None:
        """Initialize blend pass.

        Args:
            probe_manager: Manager containing registered probes
            config: Blend configuration
        """
        self._probe_manager = probe_manager
        self._config = config or ProbeBlendConfig()
        self._collector = ProbeCollector(probe_manager, self._config.to_collector_config())
        self._blender = ProbeBlender()
        self._reflection_buffer: Optional[ReflectionBuffer] = None
        self._last_execution_time_ms: float = 0.0
        self._pixel_count: int = 0

    @property
    def config(self) -> ProbeBlendConfig:
        """Get blend configuration."""
        return self._config

    @config.setter
    def config(self, value: ProbeBlendConfig) -> None:
        """Set blend configuration."""
        self._config = value
        self._collector.config = value.to_collector_config()

    @property
    def reflection_buffer(self) -> Optional[ReflectionBuffer]:
        """Get reflection buffer."""
        return self._reflection_buffer

    @property
    def last_execution_time_ms(self) -> float:
        """Get last execution time in milliseconds."""
        return self._last_execution_time_ms

    def execute(
        self,
        width: int,
        height: int,
        gbuffer_sampler: Callable[[int, int], GBufferSample],
        camera_position: Vec3,
    ) -> ReflectionBuffer:
        """Execute the blend pass.

        Args:
            width: Output width in pixels
            height: Output height in pixels
            gbuffer_sampler: Function to sample G-buffer at pixel coordinates
            camera_position: Camera world position for reflection direction

        Returns:
            ReflectionBuffer containing blended reflections
        """
        import time
        start_time = time.perf_counter()

        # Create or resize buffer
        if (self._reflection_buffer is None or
            self._reflection_buffer.width != width or
            self._reflection_buffer.height != height):
            self._reflection_buffer = ReflectionBuffer(width, height)
        else:
            self._reflection_buffer.clear()

        self._pixel_count = 0

        # Process each pixel
        for y in range(height):
            for x in range(width):
                # Sample G-buffer
                sample = gbuffer_sampler(x, y)

                if not sample.is_valid:
                    # Sky or invalid pixel, skip
                    continue

                # Calculate reflection direction
                view_dir = (camera_position - sample.position).normalized()
                reflect_dir = view_dir.reflect(sample.normal)

                # Get blended reflection
                result = self._blender.sample_blended(
                    self._collector,
                    sample.position,
                    reflect_dir,
                    sample.roughness,
                    sample.normal,
                )

                # Store result
                self._reflection_buffer.set_pixel(x, y, result.color)
                self._pixel_count += 1

        self._last_execution_time_ms = (time.perf_counter() - start_time) * 1000.0
        return self._reflection_buffer

    def get_reflection_buffer(self) -> Optional[ReflectionBuffer]:
        """Get the current reflection buffer.

        Returns:
            ReflectionBuffer or None if not yet executed
        """
        return self._reflection_buffer

    def get_stats(self) -> dict:
        """Get execution statistics.

        Returns:
            Dictionary with execution stats
        """
        return {
            "execution_time_ms": self._last_execution_time_ms,
            "pixel_count": self._pixel_count,
            "probe_count": self._probe_manager.probe_count,
            "max_probes_per_pixel": self._config.max_probes,
        }


# -----------------------------------------------------------------------------
# WGSL Shader Generation
# -----------------------------------------------------------------------------

def generate_probe_blend_wgsl(config: ProbeBlendConfig) -> str:
    """Generate WGSL compute shader for probe blending.

    Args:
        config: Blend configuration

    Returns:
        WGSL shader source code
    """
    # Map falloff type to WGSL code
    falloff_code = {
        FalloffType.LINEAR: "max(0.0, 1.0 - normalized_dist)",
        FalloffType.QUADRATIC: "max(0.0, 1.0 - normalized_dist * normalized_dist)",
        FalloffType.SMOOTH: "smoothstep(0.0, 1.0, 1.0 - normalized_dist)",
        FalloffType.INVERSE: "1.0 / (1.0 + distance / blend_distance)",
        FalloffType.EXPONENTIAL: "exp(-distance / blend_distance)",
    }.get(config.distance_falloff_type, "smoothstep(0.0, 1.0, 1.0 - normalized_dist)")

    shader = f"""// Probe Blend Compute Shader
// Generated for TRINITY Engine - Reflection Probe System
// Configuration: max_probes={config.max_probes}, falloff={config.distance_falloff_type.name}

// Constants
const MAX_PROBES: u32 = {config.max_probes}u;
const MIN_WEIGHT: f32 = {config.min_weight_threshold:.6f};
const BLEND_DISTANCE: f32 = {config.blend_distance:.6f};
const NORMAL_WEIGHT: f32 = {config.normal_weight:.6f};
const VISIBILITY_WEIGHT: f32 = {config.visibility_weight:.6f};
const EPSILON: f32 = 1e-6;

// Probe data structure
struct ProbeData {{
    position: vec3<f32>,
    radius: f32,
    bounds_min: vec3<f32>,
    _pad0: f32,
    bounds_max: vec3<f32>,
    _pad1: f32,
}};

// G-buffer input
struct GBufferSample {{
    position: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metallic: f32,
}};

// Uniforms
struct Uniforms {{
    camera_position: vec3<f32>,
    probe_count: u32,
    output_size: vec2<u32>,
    roughness_levels: u32,
    _pad: u32,
}};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> probes: array<ProbeData>;
@group(0) @binding(2) var gbuffer_position: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_material: texture_2d<f32>;
@group(0) @binding(5) var probe_cubemaps: texture_cube_array<f32>;
@group(0) @binding(6) var cubemap_sampler: sampler;
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;

// Check if point is inside probe bounds
fn point_in_probe(point: vec3<f32>, probe: ProbeData) -> bool {{
    return all(point >= probe.bounds_min) && all(point <= probe.bounds_max);
}}

// Calculate distance falloff
fn distance_falloff(distance: f32, max_dist: f32) -> f32 {{
    let normalized_dist = min(distance / max_dist, 1.0);
    let blend_distance = BLEND_DISTANCE;
    return {falloff_code};
}}

// Calculate normal alignment factor
fn normal_alignment(normal: vec3<f32>, to_probe: vec3<f32>) -> f32 {{
    let to_probe_norm = normalize(to_probe);
    let dot_product = dot(normal, to_probe_norm);
    return max(0.0, (dot_product + 1.0) * 0.5);
}}

// Calculate probe influence weight
fn calculate_weight(
    position: vec3<f32>,
    normal: vec3<f32>,
    probe: ProbeData,
) -> f32 {{
    if (!point_in_probe(position, probe)) {{
        return 0.0;
    }}

    let extent = probe.bounds_max - probe.bounds_min;
    let max_dist = length(extent) * 0.5;
    let to_probe = probe.position - position;
    let distance = length(to_probe);

    // Distance weight
    let dist_weight = distance_falloff(distance, max_dist);

    // Normal alignment
    let normal_align = normal_alignment(normal, to_probe);
    let normal_modifier = (1.0 - NORMAL_WEIGHT) + NORMAL_WEIGHT * normal_align;

    // Visibility assumed 1.0 (would need raytracing for occlusion)
    let vis_modifier = 1.0;

    return max(0.0, min(1.0, dist_weight * normal_modifier * vis_modifier));
}}

// Sample cubemap with roughness
fn sample_probe(
    probe_index: u32,
    direction: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {{
    let mip_level = roughness * f32(uniforms.roughness_levels - 1u);
    return textureSampleLevel(
        probe_cubemaps,
        cubemap_sampler,
        direction,
        probe_index,
        mip_level
    ).rgb;
}}

// Main compute kernel
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let pixel = global_id.xy;
    if (pixel.x >= uniforms.output_size.x || pixel.y >= uniforms.output_size.y) {{
        return;
    }}

    let output_index = pixel.y * uniforms.output_size.x + pixel.x;

    // Sample G-buffer
    let position = textureLoad(gbuffer_position, pixel, 0).xyz;
    let normal = normalize(textureLoad(gbuffer_normal, pixel, 0).xyz);
    let material = textureLoad(gbuffer_material, pixel, 0);
    let roughness = material.x;
    let metallic = material.y;

    // Calculate reflection direction
    let view_dir = normalize(uniforms.camera_position - position);
    let reflect_dir = reflect(-view_dir, normal);

    // Collect probe influences
    var weights: array<f32, MAX_PROBES>;
    var probe_indices: array<u32, MAX_PROBES>;
    var weight_count: u32 = 0u;
    var total_weight: f32 = 0.0;

    for (var i: u32 = 0u; i < uniforms.probe_count && weight_count < MAX_PROBES; i = i + 1u) {{
        let probe = probes[i];
        let weight = calculate_weight(position, normal, probe);

        if (weight >= MIN_WEIGHT) {{
            weights[weight_count] = weight;
            probe_indices[weight_count] = i;
            total_weight = total_weight + weight;
            weight_count = weight_count + 1u;
        }}
    }}

    // Blend samples
    var result = vec3<f32>(0.0);

    if (weight_count > 0u) {{
        let inv_total = select(1.0 / total_weight, 1.0, total_weight < EPSILON);

        for (var i: u32 = 0u; i < weight_count; i = i + 1u) {{
            let normalized_weight = weights[i] * inv_total;
            let sample = sample_probe(probe_indices[i], reflect_dir, roughness);
            result = result + sample * normalized_weight;
        }}
    }}

    output[output_index] = vec4<f32>(result, 1.0);
}}
"""
    return shader


def generate_probe_blend_shader(
    config: Optional[ProbeBlendConfig] = None,
    output_path: Optional[str] = None,
) -> str:
    """Generate and optionally save probe blend shader.

    Args:
        config: Blend configuration (uses defaults if None)
        output_path: Optional path to save shader file

    Returns:
        Generated WGSL shader source
    """
    config = config or ProbeBlendConfig()
    shader = generate_probe_blend_wgsl(config)

    if output_path:
        with open(output_path, "w") as f:
            f.write(shader)

    return shader
