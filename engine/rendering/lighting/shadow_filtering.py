"""Shadow filtering techniques.

Implements shadow filtering from Section 6.4 of RENDERING_CONTEXT.md:
- PCF (Percentage-Closer Filtering)
- PCSS (Percentage-Closer Soft Shadows)
- VSM (Variance Shadow Maps)
- ESM (Exponential Shadow Maps)
- Contact Shadows (screen-space trace)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec2, Vec3, Vec4


class ShadowFilterType(Enum):
    """Available shadow filtering techniques."""
    HARD = auto()          # No filtering (sharp edges)
    PCF = auto()           # Percentage-Closer Filtering
    PCSS = auto()          # Percentage-Closer Soft Shadows
    VSM = auto()           # Variance Shadow Maps
    ESM = auto()           # Exponential Shadow Maps
    CONTACT = auto()       # Screen-space contact shadows


@dataclass
class ShadowSample:
    """Result of a shadow map sample.

    Attributes:
        visibility: Shadow visibility (0=fully shadowed, 1=fully lit)
        distance: Distance to occluder (if available)
        penumbra_size: Estimated penumbra size (for soft shadows)
    """
    visibility: float = 1.0
    distance: float = 0.0
    penumbra_size: float = 0.0


class ShadowFilter(ABC):
    """Base class for shadow filtering techniques."""

    @property
    @abstractmethod
    def filter_type(self) -> ShadowFilterType:
        """Return the type of this filter."""
        ...

    @abstractmethod
    def sample(
        self,
        shadow_map: ShadowMapSampler,
        shadow_coord: Vec3,
        receiver_depth: float,
    ) -> ShadowSample:
        """Sample the shadow map with filtering.

        Args:
            shadow_map: Shadow map sampler interface
            shadow_coord: Shadow map coordinates (UV + depth)
            receiver_depth: Depth of the receiving surface

        Returns:
            Shadow sample result
        """
        ...


class ShadowMapSampler:
    """Interface for sampling shadow maps.

    This abstraction allows filters to work with different shadow map
    implementations and GPU backends.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        sample_func: Optional[Callable[[float, float], float]] = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            resolution: Shadow map resolution (width, height)
            sample_func: Optional custom sample function
        """
        self.resolution = resolution
        self._sample_func = sample_func
        self._depth_data: Optional[list[list[float]]] = None

    def set_depth_data(self, data: list[list[float]]) -> None:
        """Set depth data for CPU-based sampling.

        Args:
            data: 2D array of depth values
        """
        self._depth_data = data

    def sample_depth(self, u: float, v: float) -> float:
        """Sample the shadow map depth at UV coordinates.

        Args:
            u: U coordinate [0, 1]
            v: V coordinate [0, 1]

        Returns:
            Depth value at the sample location
        """
        if self._sample_func:
            return self._sample_func(u, v)

        if self._depth_data:
            # Bilinear sampling from depth data
            w, h = self.resolution
            x = u * (w - 1)
            y = v * (h - 1)
            x0, y0 = int(x), int(y)
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)
            fx, fy = x - x0, y - y0

            d00 = self._depth_data[y0][x0]
            d10 = self._depth_data[y0][x1]
            d01 = self._depth_data[y1][x0]
            d11 = self._depth_data[y1][x1]

            return (
                d00 * (1 - fx) * (1 - fy) +
                d10 * fx * (1 - fy) +
                d01 * (1 - fx) * fy +
                d11 * fx * fy
            )

        return 0.0

    def get_texel_size(self) -> Vec2:
        """Get the size of a texel in UV space.

        Returns:
            Texel size as (1/width, 1/height)
        """
        return Vec2(1.0 / self.resolution[0], 1.0 / self.resolution[1])


@dataclass
class PCFConfig:
    """Configuration for PCF shadow filtering.

    Attributes:
        kernel_size: Size of the filter kernel (e.g., 3 for 3x3)
        sample_pattern: Pattern for taking samples ("grid", "poisson", "vogel")
        bias: Depth comparison bias
    """
    kernel_size: int = 3
    sample_pattern: str = "grid"
    bias: float = 0.001


class PCFFilter(ShadowFilter):
    """Percentage-Closer Filtering for smooth shadow edges.

    PCF averages multiple depth comparisons around each pixel
    to produce soft shadow edges.
    """

    def __init__(self, config: PCFConfig = None) -> None:
        """Initialize PCF filter.

        Args:
            config: PCF configuration
        """
        self.config = config or PCFConfig()
        self._kernel = self._generate_kernel()

    @property
    def filter_type(self) -> ShadowFilterType:
        return ShadowFilterType.PCF

    def _generate_kernel(self) -> list[Vec2]:
        """Generate sample kernel based on configuration.

        Returns:
            List of sample offsets
        """
        if self.config.sample_pattern == "poisson":
            return self._poisson_disk()
        elif self.config.sample_pattern == "vogel":
            return self._vogel_disk()
        else:
            return self._grid_pattern()

    def _grid_pattern(self) -> list[Vec2]:
        """Generate grid sample pattern."""
        samples = []
        half = self.config.kernel_size // 2
        for y in range(-half, half + 1):
            for x in range(-half, half + 1):
                samples.append(Vec2(float(x), float(y)))
        return samples

    def _poisson_disk(self) -> list[Vec2]:
        """Generate Poisson disk sample pattern.

        Pre-computed for common kernel sizes.
        """
        # Pre-computed Poisson disk samples
        poisson_samples = [
            Vec2(-0.94201624, -0.39906216),
            Vec2(0.94558609, -0.76890725),
            Vec2(-0.09418410, -0.92938870),
            Vec2(0.34495938, 0.29387760),
            Vec2(-0.91588581, 0.45771432),
            Vec2(-0.81544232, -0.87912464),
            Vec2(-0.38277543, 0.27676845),
            Vec2(0.97484398, 0.75648379),
            Vec2(0.44323325, -0.97511554),
            Vec2(0.53742981, -0.47373420),
            Vec2(-0.26496911, -0.41893023),
            Vec2(0.79197514, 0.19090188),
            Vec2(-0.24188840, 0.99706507),
            Vec2(-0.81409955, 0.91437590),
            Vec2(0.19984126, 0.78641367),
            Vec2(0.14383161, -0.14100790),
        ]
        n = min(self.config.kernel_size * self.config.kernel_size, len(poisson_samples))
        return poisson_samples[:n]

    def _vogel_disk(self) -> list[Vec2]:
        """Generate Vogel disk sample pattern.

        Vogel disk provides good distribution for varying sample counts.
        """
        samples = []
        n = self.config.kernel_size * self.config.kernel_size
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))

        for i in range(n):
            r = math.sqrt((i + 0.5) / n)
            theta = i * golden_angle
            samples.append(Vec2(r * math.cos(theta), r * math.sin(theta)))

        return samples

    def sample(
        self,
        shadow_map: ShadowMapSampler,
        shadow_coord: Vec3,
        receiver_depth: float,
    ) -> ShadowSample:
        """Sample with PCF filtering.

        Args:
            shadow_map: Shadow map sampler
            shadow_coord: Shadow UV + depth
            receiver_depth: Receiver surface depth

        Returns:
            Filtered shadow sample
        """
        texel_size = shadow_map.get_texel_size()
        visibility = 0.0
        sample_count = len(self._kernel)

        for offset in self._kernel:
            u = shadow_coord.x + offset.x * texel_size.x
            v = shadow_coord.y + offset.y * texel_size.y

            # Clamp to valid UV range
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))

            shadow_depth = shadow_map.sample_depth(u, v)

            # Depth comparison with bias
            if receiver_depth - self.config.bias <= shadow_depth:
                visibility += 1.0

        visibility /= sample_count

        return ShadowSample(visibility=visibility)


@dataclass
class PCSSConfig:
    """Configuration for PCSS shadow filtering.

    Attributes:
        blocker_search_samples: Number of samples for blocker search
        pcf_samples: Number of samples for final PCF
        light_size: Size of the light source (affects penumbra)
        near_plane: Near plane for depth linearization
        max_filter_radius: Maximum filter radius in UV space
    """
    blocker_search_samples: int = 16
    pcf_samples: int = 32
    light_size: float = 1.0
    near_plane: float = 0.1
    max_filter_radius: float = 0.1


class PCSSFilter(ShadowFilter):
    """Percentage-Closer Soft Shadows with variable penumbra.

    PCSS estimates the average blocker depth to compute penumbra size,
    then applies PCF with a kernel sized to match the penumbra.
    """

    def __init__(self, config: PCSSConfig = None) -> None:
        """Initialize PCSS filter.

        Args:
            config: PCSS configuration
        """
        self.config = config or PCSSConfig()
        self._blocker_kernel = self._generate_vogel_disk(config.blocker_search_samples if config else 16)
        self._pcf_kernel = self._generate_vogel_disk(config.pcf_samples if config else 32)

    @property
    def filter_type(self) -> ShadowFilterType:
        return ShadowFilterType.PCSS

    def _generate_vogel_disk(self, n: int) -> list[Vec2]:
        """Generate Vogel disk samples."""
        samples = []
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))

        for i in range(n):
            r = math.sqrt((i + 0.5) / n)
            theta = i * golden_angle
            samples.append(Vec2(r * math.cos(theta), r * math.sin(theta)))

        return samples

    def sample(
        self,
        shadow_map: ShadowMapSampler,
        shadow_coord: Vec3,
        receiver_depth: float,
    ) -> ShadowSample:
        """Sample with PCSS filtering.

        Args:
            shadow_map: Shadow map sampler
            shadow_coord: Shadow UV + depth
            receiver_depth: Receiver surface depth

        Returns:
            Filtered shadow sample with variable penumbra
        """
        texel_size = shadow_map.get_texel_size()

        # Step 1: Blocker search
        avg_blocker_depth, blocker_count = self._find_blockers(
            shadow_map, shadow_coord, receiver_depth, texel_size
        )

        if blocker_count == 0:
            # No blockers found - fully lit
            return ShadowSample(visibility=1.0)

        # Step 2: Estimate penumbra size
        penumbra_size = self._estimate_penumbra(
            receiver_depth, avg_blocker_depth
        )

        # Step 3: Variable-size PCF
        filter_radius = min(
            penumbra_size * self.config.light_size,
            self.config.max_filter_radius
        )

        visibility = self._filtered_pcf(
            shadow_map, shadow_coord, receiver_depth,
            texel_size, filter_radius
        )

        return ShadowSample(
            visibility=visibility,
            distance=avg_blocker_depth,
            penumbra_size=penumbra_size,
        )

    def _find_blockers(
        self,
        shadow_map: ShadowMapSampler,
        shadow_coord: Vec3,
        receiver_depth: float,
        texel_size: Vec2,
    ) -> tuple[float, int]:
        """Find average blocker depth.

        Args:
            shadow_map: Shadow map sampler
            shadow_coord: Shadow coordinates
            receiver_depth: Receiver depth
            texel_size: Texel size

        Returns:
            Tuple of (average blocker depth, blocker count)
        """
        blocker_sum = 0.0
        blocker_count = 0
        search_radius = self.config.light_size * 0.5

        for offset in self._blocker_kernel:
            u = shadow_coord.x + offset.x * search_radius * texel_size.x
            v = shadow_coord.y + offset.y * search_radius * texel_size.y

            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))

            shadow_depth = shadow_map.sample_depth(u, v)

            if shadow_depth < receiver_depth:
                blocker_sum += shadow_depth
                blocker_count += 1

        if blocker_count > 0:
            return blocker_sum / blocker_count, blocker_count
        return 0.0, 0

    def _estimate_penumbra(
        self,
        receiver_depth: float,
        blocker_depth: float,
    ) -> float:
        """Estimate penumbra size based on blocker and receiver depths.

        Args:
            receiver_depth: Depth of the receiving surface
            blocker_depth: Average depth of blockers

        Returns:
            Estimated penumbra width
        """
        if blocker_depth <= 0 or blocker_depth >= receiver_depth:
            return 0.0

        # Penumbra width = (d_receiver - d_blocker) * light_size / d_blocker
        return (receiver_depth - blocker_depth) / blocker_depth

    def _filtered_pcf(
        self,
        shadow_map: ShadowMapSampler,
        shadow_coord: Vec3,
        receiver_depth: float,
        texel_size: Vec2,
        filter_radius: float,
    ) -> float:
        """Apply PCF with specified filter radius.

        Args:
            shadow_map: Shadow map sampler
            shadow_coord: Shadow coordinates
            receiver_depth: Receiver depth
            texel_size: Texel size
            filter_radius: Filter radius in UV space

        Returns:
            Visibility value [0, 1]
        """
        visibility = 0.0
        bias = 0.001

        for offset in self._pcf_kernel:
            u = shadow_coord.x + offset.x * filter_radius
            v = shadow_coord.y + offset.y * filter_radius

            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))

            shadow_depth = shadow_map.sample_depth(u, v)

            if receiver_depth - bias <= shadow_depth:
                visibility += 1.0

        return visibility / len(self._pcf_kernel)


@dataclass
class VSMConfig:
    """Configuration for Variance Shadow Maps.

    Attributes:
        min_variance: Minimum variance to avoid numerical issues
        light_bleeding_reduction: Light bleeding reduction factor
        use_moment2: Use second moment for better precision
    """
    min_variance: float = 0.00001
    light_bleeding_reduction: float = 0.2
    use_moment2: bool = True


class VSMFilter(ShadowFilter):
    """Variance Shadow Maps for pre-filtered soft shadows.

    VSM stores depth and depth^2 in the shadow map, allowing
    for hardware texture filtering to create soft shadows.
    """

    def __init__(self, config: VSMConfig = None) -> None:
        """Initialize VSM filter.

        Args:
            config: VSM configuration
        """
        self.config = config or VSMConfig()

    @property
    def filter_type(self) -> ShadowFilterType:
        return ShadowFilterType.VSM

    def sample(
        self,
        shadow_map: ShadowMapSampler,
        shadow_coord: Vec3,
        receiver_depth: float,
    ) -> ShadowSample:
        """Sample with VSM filtering.

        Note: VSM requires a two-channel shadow map storing (depth, depth^2).
        This implementation assumes the sampler returns these packed values.

        Args:
            shadow_map: Shadow map sampler (returns depth, depth^2)
            shadow_coord: Shadow UV + depth
            receiver_depth: Receiver surface depth

        Returns:
            VSM shadow sample
        """
        # Sample first and second moments
        # In a real implementation, these would come from a 2-channel texture
        moment1 = shadow_map.sample_depth(shadow_coord.x, shadow_coord.y)
        # Approximate second moment (would normally be stored in texture)
        moment2 = moment1 * moment1 * 1.001

        # Compute variance
        variance = moment2 - moment1 * moment1
        variance = max(variance, self.config.min_variance)

        # Chebyshev's inequality
        d = receiver_depth - moment1
        p_max = variance / (variance + d * d) if d > 0 else 1.0

        # Light bleeding reduction
        visibility = self._reduce_light_bleeding(
            p_max, self.config.light_bleeding_reduction
        )

        return ShadowSample(visibility=visibility)

    def _reduce_light_bleeding(
        self, p_max: float, amount: float
    ) -> float:
        """Reduce light bleeding artifacts.

        Args:
            p_max: Maximum probability from Chebyshev
            amount: Light bleeding reduction amount

        Returns:
            Adjusted visibility
        """
        # Linearly rescale probability
        return max(0.0, (p_max - amount) / (1.0 - amount))


@dataclass
class ESMConfig:
    """Configuration for Exponential Shadow Maps.

    Attributes:
        exponent: Exponential constant (higher = sharper shadows)
        bias: Depth bias
    """
    exponent: float = 80.0
    bias: float = 0.001


class ESMFilter(ShadowFilter):
    """Exponential Shadow Maps for filtered soft shadows.

    ESM uses exponential depth representation which can be
    safely filtered with hardware texture filtering.
    """

    def __init__(self, config: ESMConfig = None) -> None:
        """Initialize ESM filter.

        Args:
            config: ESM configuration
        """
        self.config = config or ESMConfig()

    @property
    def filter_type(self) -> ShadowFilterType:
        return ShadowFilterType.ESM

    def sample(
        self,
        shadow_map: ShadowMapSampler,
        shadow_coord: Vec3,
        receiver_depth: float,
    ) -> ShadowSample:
        """Sample with ESM filtering.

        Note: ESM requires shadow map to store exp(c * depth).
        This implementation approximates for demonstration.

        Args:
            shadow_map: Shadow map sampler
            shadow_coord: Shadow UV + depth
            receiver_depth: Receiver surface depth

        Returns:
            ESM shadow sample
        """
        # Sample exponential occluder depth
        shadow_depth = shadow_map.sample_depth(shadow_coord.x, shadow_coord.y)

        # In production: shadow map stores exp(c * d_occluder)
        # exp_occluder = texture_sample
        # visibility = clamp(exp_occluder * exp(-c * d_receiver), 0, 1)

        c = self.config.exponent
        exp_occluder = math.exp(c * shadow_depth)
        exp_receiver = math.exp(-c * (receiver_depth - self.config.bias))

        visibility = min(1.0, exp_occluder * exp_receiver)
        visibility = max(0.0, visibility)

        return ShadowSample(visibility=visibility)


@dataclass
class ContactShadowConfig:
    """Configuration for screen-space contact shadows.

    Attributes:
        max_distance: Maximum ray march distance (world units)
        step_count: Number of ray march steps
        thickness: Assumed occluder thickness
        fade_start: Distance at which shadows start fading
        fade_end: Distance at which shadows fully fade
    """
    max_distance: float = 0.5
    step_count: int = 16
    thickness: float = 0.01
    fade_start: float = 0.3
    fade_end: float = 0.5


class ContactShadowFilter(ShadowFilter):
    """Screen-space contact shadows for fine shadow detail.

    Contact shadows ray march in screen space to find occluders
    that might be missed by the shadow map due to resolution.
    """

    def __init__(self, config: ContactShadowConfig = None) -> None:
        """Initialize contact shadow filter.

        Args:
            config: Contact shadow configuration
        """
        self.config = config or ContactShadowConfig()

    @property
    def filter_type(self) -> ShadowFilterType:
        return ShadowFilterType.CONTACT

    def sample(
        self,
        shadow_map: ShadowMapSampler,
        shadow_coord: Vec3,
        receiver_depth: float,
    ) -> ShadowSample:
        """Sample contact shadows (placeholder).

        Note: Contact shadows require access to the depth buffer,
        not the shadow map. This is a placeholder showing the interface.

        Args:
            shadow_map: Unused for contact shadows
            shadow_coord: Screen-space position
            receiver_depth: Receiver depth

        Returns:
            Contact shadow sample
        """
        # In production, this would:
        # 1. Ray march from shading point toward light in screen space
        # 2. Sample depth buffer at each step
        # 3. Check if ray is occluded by comparing depths

        # Placeholder return
        return ShadowSample(visibility=1.0)

    def trace(
        self,
        start_pos: Vec3,
        light_dir: Vec3,
        depth_buffer_sample: Callable[[Vec2], float],
        view_proj: any,  # Mat4 in production
    ) -> ShadowSample:
        """Trace contact shadows in screen space.

        Args:
            start_pos: Starting world position
            light_dir: Direction to light
            depth_buffer_sample: Function to sample depth buffer
            view_proj: View-projection matrix

        Returns:
            Contact shadow result
        """
        ray_dir = light_dir.normalized()
        step_size = self.config.max_distance / self.config.step_count

        for i in range(self.config.step_count):
            # Advance along ray
            t = (i + 1) * step_size
            sample_pos = start_pos + ray_dir * t

            # Project to screen space (simplified)
            # In production: use view_proj matrix
            screen_pos = Vec2(
                sample_pos.x * 0.5 + 0.5,
                sample_pos.y * 0.5 + 0.5,
            )

            # Sample depth buffer
            buffer_depth = depth_buffer_sample(screen_pos)

            # Compare depths
            ray_depth = sample_pos.z
            if ray_depth > buffer_depth + self.config.thickness:
                # Hit occluder
                fade = self._compute_fade(t)
                return ShadowSample(visibility=fade, distance=t)

        return ShadowSample(visibility=1.0)

    def _compute_fade(self, distance: float) -> float:
        """Compute shadow fade based on distance.

        Args:
            distance: Distance along ray

        Returns:
            Fade factor (0 = no shadow, 1 = full shadow)
        """
        if distance <= self.config.fade_start:
            return 0.0
        if distance >= self.config.fade_end:
            return 1.0

        # Guard against division by zero when fade_start equals fade_end
        fade_range = self.config.fade_end - self.config.fade_start
        if fade_range <= 1e-6:
            return 1.0

        t = (distance - self.config.fade_start) / fade_range
        # Smooth fade
        return t * t * (3.0 - 2.0 * t)


def create_shadow_filter(filter_type: ShadowFilterType) -> ShadowFilter:
    """Factory function to create shadow filters.

    Args:
        filter_type: Type of filter to create

    Returns:
        Configured shadow filter
    """
    if filter_type == ShadowFilterType.PCF:
        return PCFFilter()
    elif filter_type == ShadowFilterType.PCSS:
        return PCSSFilter()
    elif filter_type == ShadowFilterType.VSM:
        return VSMFilter()
    elif filter_type == ShadowFilterType.ESM:
        return ESMFilter()
    elif filter_type == ShadowFilterType.CONTACT:
        return ContactShadowFilter()
    else:
        # Default to simple PCF
        return PCFFilter(PCFConfig(kernel_size=1))
