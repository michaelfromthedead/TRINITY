"""
SSR Roughness-Driven Blur System

Provides Bloomberg-style roughness-based reflection blurring:
- SSRRoughnessBlur: Main blur processor with multi-level downsample
- GaussianBlur: Separable Gaussian convolution
- BilateralUpscale: Edge-aware bilateral upsampling
- MaterialReflectionParams: Per-material reflection configuration

The blur kernel radius scales with roughness squared:
    kernel_radius = roughness^2 * max_radius

This provides smooth surfaces with sharp reflections and rough
surfaces with appropriately blurry reflections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class BlurTechnique(Enum):
    """Blur algorithm selection for SSR."""

    GAUSSIAN = auto()  # Traditional separable Gaussian
    KAWASE = auto()  # Dual Kawase (faster, lower quality)
    BOX = auto()  # Simple box filter (fastest)


class SSRBlurQuality(Enum):
    """Quality presets for SSR blur."""

    LOW = auto()  # 2 downsample levels, box blur
    MEDIUM = auto()  # 3 downsample levels, Gaussian blur
    HIGH = auto()  # 4 downsample levels, Gaussian blur
    ULTRA = auto()  # 5 downsample levels, Gaussian blur with bilateral


@dataclass(frozen=True)
class SSRBlurConstants:
    """Constants for SSR roughness blur."""

    # Roughness thresholds
    ROUGHNESS_MIRROR: float = 0.01  # Below this = perfect mirror
    ROUGHNESS_MAX_BLUR: float = 1.0  # Maximum roughness value
    ROUGHNESS_CUTOFF: float = 0.9  # Above this = no SSR, use probes

    # Blur radius settings
    MAX_BLUR_RADIUS_DEFAULT: float = 32.0
    MAX_BLUR_RADIUS_MIN: float = 1.0
    MAX_BLUR_RADIUS_MAX: float = 64.0

    # Downsample chain settings
    DOWNSAMPLE_LEVELS_LOW: int = 2
    DOWNSAMPLE_LEVELS_MEDIUM: int = 3
    DOWNSAMPLE_LEVELS_HIGH: int = 4
    DOWNSAMPLE_LEVELS_ULTRA: int = 5
    DOWNSAMPLE_LEVELS_MAX: int = 6

    # Gaussian settings
    GAUSSIAN_SIGMA_DEFAULT: float = 2.0
    GAUSSIAN_RADIUS_DEFAULT: int = 4
    GAUSSIAN_ITERATIONS_DEFAULT: int = 2

    # Bilateral upscale settings
    BILATERAL_SIGMA_SPATIAL: float = 2.0
    BILATERAL_SIGMA_RANGE: float = 0.1
    BILATERAL_RADIUS_DEFAULT: int = 3

    # Edge preservation
    DEPTH_THRESHOLD: float = 0.01  # Normalized depth difference
    NORMAL_THRESHOLD: float = 0.9  # Dot product threshold

    # Roughness to blur mapping power
    ROUGHNESS_POWER: float = 2.0  # kernel = roughness^power * max_radius


SSR_BLUR = SSRBlurConstants()


@dataclass
class MaterialReflectionParams:
    """Per-material reflection configuration.

    Controls how reflections appear for a specific material,
    allowing artists to fine-tune the reflection behavior.

    Attributes:
        intensity: Reflection intensity multiplier [0, 2].
        roughness_offset: Offset added to roughness [-0.5, 0.5].
        technique_override: Optional technique override.
        use_contact_hardening: Enable distance-based blur falloff.
        fresnel_power: Fresnel effect exponent [1, 10].
        anisotropy: Anisotropic reflection stretching [-1, 1].
        anisotropy_rotation: Rotation of anisotropy in radians.
    """

    intensity: float = 1.0
    roughness_offset: float = 0.0
    technique_override: Optional[BlurTechnique] = None
    use_contact_hardening: bool = False
    fresnel_power: float = 5.0
    anisotropy: float = 0.0
    anisotropy_rotation: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.intensity <= 2.0:
            raise ValueError(f"intensity must be in [0, 2], got {self.intensity}")
        if not -0.5 <= self.roughness_offset <= 0.5:
            raise ValueError(
                f"roughness_offset must be in [-0.5, 0.5], got {self.roughness_offset}"
            )
        if not 1.0 <= self.fresnel_power <= 10.0:
            raise ValueError(
                f"fresnel_power must be in [1, 10], got {self.fresnel_power}"
            )
        if not -1.0 <= self.anisotropy <= 1.0:
            raise ValueError(
                f"anisotropy must be in [-1, 1], got {self.anisotropy}"
            )

    def get_effective_roughness(self, base_roughness: float) -> float:
        """Calculate effective roughness with offset applied.

        Args:
            base_roughness: Base material roughness [0, 1].

        Returns:
            Effective roughness clamped to [0, 1].
        """
        return max(0.0, min(1.0, base_roughness + self.roughness_offset))

    def compute_fresnel(self, cos_theta: float, base_reflectivity: float = 0.04) -> float:
        """Compute Fresnel reflectance using Schlick's approximation.

        Args:
            cos_theta: Cosine of angle between view and normal.
            base_reflectivity: F0 reflectivity at normal incidence.

        Returns:
            Fresnel reflectance factor [0, 1].
        """
        cos_theta = max(0.0, min(1.0, cos_theta))
        one_minus_cos = 1.0 - cos_theta
        return base_reflectivity + (1.0 - base_reflectivity) * (
            one_minus_cos ** self.fresnel_power
        )


class GaussianBlur:
    """Separable Gaussian blur processor.

    Implements efficient 2-pass separable convolution with
    pre-calculated weights for consistent quality.
    """

    def __init__(
        self,
        radius: int = SSR_BLUR.GAUSSIAN_RADIUS_DEFAULT,
        sigma: float = SSR_BLUR.GAUSSIAN_SIGMA_DEFAULT,
    ) -> None:
        """Initialize Gaussian blur.

        Args:
            radius: Blur radius in pixels.
            sigma: Gaussian standard deviation.
        """
        self._radius: int = max(1, radius)
        self._sigma: float = max(0.1, sigma)
        self._weights: List[float] = []
        self._offsets: List[float] = []
        self._calculate_weights()

    @property
    def radius(self) -> int:
        """Blur radius in pixels."""
        return self._radius

    @radius.setter
    def radius(self, value: int) -> None:
        self._radius = max(1, value)
        self._calculate_weights()

    @property
    def sigma(self) -> float:
        """Gaussian standard deviation."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        self._sigma = max(0.1, value)
        self._calculate_weights()

    @property
    def weights(self) -> List[float]:
        """Pre-calculated Gaussian weights."""
        return self._weights.copy()

    @property
    def offsets(self) -> List[float]:
        """Pre-calculated sample offsets."""
        return self._offsets.copy()

    def _calculate_weights(self) -> None:
        """Pre-calculate normalized Gaussian weights."""
        self._weights = []
        self._offsets = []
        total = 0.0

        for i in range(self._radius + 1):
            weight = math.exp(-(i * i) / (2.0 * self._sigma * self._sigma))
            self._weights.append(weight)
            self._offsets.append(float(i))
            total += weight if i == 0 else weight * 2.0

        # Normalize weights
        for i in range(len(self._weights)):
            self._weights[i] /= total

    def blur_horizontal(
        self,
        source: List[float],
        width: int,
        height: int,
    ) -> List[float]:
        """Apply horizontal blur pass.

        Args:
            source: Source buffer (flat RGBA list).
            width: Buffer width in pixels.
            height: Buffer height in pixels.

        Returns:
            Horizontally blurred buffer.
        """
        if not source or width < 1 or height < 1:
            return source if source else []

        result = [0.0] * len(source)
        stride = width * 4

        for y in range(height):
            row = y * stride
            for x in range(width):
                idx = row + x * 4
                for c in range(4):
                    total = source[idx + c] * self._weights[0]
                    for k in range(1, len(self._weights)):
                        off = int(self._offsets[k])
                        w = self._weights[k]
                        # Sample left
                        if x - off >= 0:
                            total += source[idx - off * 4 + c] * w
                        else:
                            total += source[row + c] * w  # Clamp to edge
                        # Sample right
                        if x + off < width:
                            total += source[idx + off * 4 + c] * w
                        else:
                            total += source[row + (width - 1) * 4 + c] * w
                    result[idx + c] = total

        return result

    def blur_vertical(
        self,
        source: List[float],
        width: int,
        height: int,
    ) -> List[float]:
        """Apply vertical blur pass.

        Args:
            source: Source buffer (flat RGBA list).
            width: Buffer width in pixels.
            height: Buffer height in pixels.

        Returns:
            Vertically blurred buffer.
        """
        if not source or width < 1 or height < 1:
            return source if source else []

        result = [0.0] * len(source)
        stride = width * 4

        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 4
                for c in range(4):
                    total = source[idx + c] * self._weights[0]
                    for k in range(1, len(self._weights)):
                        off = int(self._offsets[k])
                        w = self._weights[k]
                        # Sample up
                        if y - off >= 0:
                            up_idx = ((y - off) * width + x) * 4
                            total += source[up_idx + c] * w
                        else:
                            total += source[x * 4 + c] * w  # Clamp to edge
                        # Sample down
                        if y + off < height:
                            down_idx = ((y + off) * width + x) * 4
                            total += source[down_idx + c] * w
                        else:
                            total += source[((height - 1) * width + x) * 4 + c] * w
                    result[idx + c] = total

        return result

    def blur(
        self,
        source: List[float],
        width: int,
        height: int,
        iterations: int = 1,
    ) -> List[float]:
        """Apply full separable Gaussian blur.

        Args:
            source: Source buffer (flat RGBA list).
            width: Buffer width in pixels.
            height: Buffer height in pixels.
            iterations: Number of blur passes.

        Returns:
            Blurred buffer.
        """
        if not source or width < 1 or height < 1:
            return source if source else []

        current = source
        for _ in range(iterations):
            current = self.blur_horizontal(current, width, height)
            current = self.blur_vertical(current, width, height)

        return current


class BilateralUpscale:
    """Edge-aware bilateral upsampler.

    Preserves silhouettes and sharp edges during upscaling by
    considering both spatial distance and value similarity.
    """

    def __init__(
        self,
        sigma_spatial: float = SSR_BLUR.BILATERAL_SIGMA_SPATIAL,
        sigma_range: float = SSR_BLUR.BILATERAL_SIGMA_RANGE,
        radius: int = SSR_BLUR.BILATERAL_RADIUS_DEFAULT,
    ) -> None:
        """Initialize bilateral upsampler.

        Args:
            sigma_spatial: Spatial Gaussian sigma.
            sigma_range: Range/color Gaussian sigma.
            radius: Filter radius in pixels.
        """
        self._sigma_spatial: float = max(0.1, sigma_spatial)
        self._sigma_range: float = max(0.001, sigma_range)
        self._radius: int = max(1, radius)
        self._spatial_weights: List[List[float]] = []
        self._calculate_spatial_weights()

    @property
    def sigma_spatial(self) -> float:
        """Spatial Gaussian sigma."""
        return self._sigma_spatial

    @sigma_spatial.setter
    def sigma_spatial(self, value: float) -> None:
        self._sigma_spatial = max(0.1, value)
        self._calculate_spatial_weights()

    @property
    def sigma_range(self) -> float:
        """Range/color Gaussian sigma."""
        return self._sigma_range

    @sigma_range.setter
    def sigma_range(self, value: float) -> None:
        self._sigma_range = max(0.001, value)

    @property
    def radius(self) -> int:
        """Filter radius in pixels."""
        return self._radius

    @radius.setter
    def radius(self, value: int) -> None:
        self._radius = max(1, value)
        self._calculate_spatial_weights()

    def _calculate_spatial_weights(self) -> None:
        """Pre-calculate spatial Gaussian weights."""
        size = 2 * self._radius + 1
        self._spatial_weights = []

        for dy in range(-self._radius, self._radius + 1):
            row = []
            for dx in range(-self._radius, self._radius + 1):
                dist_sq = dx * dx + dy * dy
                weight = math.exp(-dist_sq / (2.0 * self._sigma_spatial ** 2))
                row.append(weight)
            self._spatial_weights.append(row)

    def _range_weight(self, diff: float) -> float:
        """Calculate range weight based on color/depth difference.

        Args:
            diff: Absolute value difference.

        Returns:
            Range weight [0, 1].
        """
        return math.exp(-(diff * diff) / (2.0 * self._sigma_range ** 2))

    def upscale(
        self,
        low_res: List[float],
        low_width: int,
        low_height: int,
        high_width: int,
        high_height: int,
        depth_buffer: Optional[List[float]] = None,
        normal_buffer: Optional[List[float]] = None,
    ) -> List[float]:
        """Upscale low-resolution buffer with edge awareness.

        Args:
            low_res: Low-resolution color buffer (flat RGBA).
            low_width: Low-resolution width.
            low_height: Low-resolution height.
            high_width: Target high-resolution width.
            high_height: Target high-resolution height.
            depth_buffer: Optional high-res depth for edge detection.
            normal_buffer: Optional high-res normals for edge detection.

        Returns:
            Upscaled buffer at high resolution.
        """
        if not low_res or low_width < 1 or low_height < 1:
            return [0.0] * (high_width * high_height * 4)

        result = [0.0] * (high_width * high_height * 4)

        scale_x = low_width / high_width
        scale_y = low_height / high_height

        for hy in range(high_height):
            for hx in range(high_width):
                # Map to low-res coordinates
                lx = hx * scale_x
                ly = hy * scale_y

                # Get center pixel in low-res
                cx = int(lx)
                cy = int(ly)

                # Gather bilateral weighted samples
                total_weight = 0.0
                color = [0.0, 0.0, 0.0, 0.0]

                # Get reference depth/normal at high-res position
                ref_depth = 0.0
                if depth_buffer and len(depth_buffer) > hy * high_width + hx:
                    ref_depth = depth_buffer[hy * high_width + hx]

                for dy in range(-self._radius, self._radius + 1):
                    for dx in range(-self._radius, self._radius + 1):
                        sx = cx + dx
                        sy = cy + dy

                        if 0 <= sx < low_width and 0 <= sy < low_height:
                            s_idx = (sy * low_width + sx) * 4

                            # Spatial weight
                            spatial_w = self._spatial_weights[dy + self._radius][
                                dx + self._radius
                            ]

                            # Range weight (based on color similarity)
                            center_idx = (cy * low_width + cx) * 4
                            color_diff = 0.0
                            for c in range(3):
                                diff = low_res[s_idx + c] - low_res[center_idx + c]
                                color_diff += diff * diff
                            color_diff = math.sqrt(color_diff)
                            range_w = self._range_weight(color_diff)

                            # Depth edge weight
                            depth_w = 1.0
                            if depth_buffer:
                                # Map sample position to high-res depth
                                hsx = int(sx / scale_x)
                                hsy = int(sy / scale_y)
                                if 0 <= hsx < high_width and 0 <= hsy < high_height:
                                    sample_depth = depth_buffer[hsy * high_width + hsx]
                                    depth_diff = abs(sample_depth - ref_depth)
                                    if depth_diff > SSR_BLUR.DEPTH_THRESHOLD:
                                        depth_w = 0.1  # Reduce weight across edges

                            weight = spatial_w * range_w * depth_w
                            total_weight += weight

                            for c in range(4):
                                color[c] += low_res[s_idx + c] * weight

                # Normalize
                if total_weight > 1e-6:
                    for c in range(4):
                        color[c] /= total_weight

                out_idx = (hy * high_width + hx) * 4
                for c in range(4):
                    result[out_idx + c] = color[c]

        return result

    def upscale_simple(
        self,
        low_res: List[float],
        low_width: int,
        low_height: int,
        high_width: int,
        high_height: int,
    ) -> List[float]:
        """Simple bilinear upscale without edge awareness.

        Args:
            low_res: Low-resolution color buffer.
            low_width: Low-resolution width.
            low_height: Low-resolution height.
            high_width: Target high-resolution width.
            high_height: Target high-resolution height.

        Returns:
            Upscaled buffer.
        """
        if not low_res or low_width < 1 or low_height < 1:
            return [0.0] * (high_width * high_height * 4)

        result = [0.0] * (high_width * high_height * 4)

        for hy in range(high_height):
            for hx in range(high_width):
                # Map to low-res coordinates (center of pixel)
                lx = (hx + 0.5) * low_width / high_width - 0.5
                ly = (hy + 0.5) * low_height / high_height - 0.5

                # Bilinear interpolation coordinates
                x0 = max(0, min(low_width - 1, int(lx)))
                y0 = max(0, min(low_height - 1, int(ly)))
                x1 = max(0, min(low_width - 1, x0 + 1))
                y1 = max(0, min(low_height - 1, y0 + 1))

                fx = lx - x0
                fy = ly - y0
                fx = max(0.0, min(1.0, fx))
                fy = max(0.0, min(1.0, fy))

                # Sample four corners
                i00 = (y0 * low_width + x0) * 4
                i10 = (y0 * low_width + x1) * 4
                i01 = (y1 * low_width + x0) * 4
                i11 = (y1 * low_width + x1) * 4

                out_idx = (hy * high_width + hx) * 4
                for c in range(4):
                    v00 = low_res[i00 + c]
                    v10 = low_res[i10 + c]
                    v01 = low_res[i01 + c]
                    v11 = low_res[i11 + c]

                    # Bilinear interpolation
                    top = v00 * (1.0 - fx) + v10 * fx
                    bottom = v01 * (1.0 - fx) + v11 * fx
                    result[out_idx + c] = top * (1.0 - fy) + bottom * fy

        return result


@dataclass
class DownsampleLevel:
    """Single level in the downsample chain."""

    width: int
    height: int
    buffer: Optional[List[float]] = None
    blurred: Optional[List[float]] = None


class DownsampleChain:
    """Multi-level downsample chain for Bloomberg-style blur.

    Creates progressively smaller versions of the reflection buffer
    for efficient roughness-based blur lookup.
    """

    def __init__(self, max_levels: int = SSR_BLUR.DOWNSAMPLE_LEVELS_HIGH) -> None:
        """Initialize downsample chain.

        Args:
            max_levels: Maximum number of downsample levels.
        """
        self._max_levels: int = max(1, min(max_levels, SSR_BLUR.DOWNSAMPLE_LEVELS_MAX))
        self._levels: List[DownsampleLevel] = []
        self._base_width: int = 0
        self._base_height: int = 0

    @property
    def max_levels(self) -> int:
        """Maximum number of levels."""
        return self._max_levels

    @property
    def level_count(self) -> int:
        """Current number of active levels."""
        return len(self._levels)

    @property
    def levels(self) -> List[DownsampleLevel]:
        """All downsample levels."""
        return self._levels.copy()

    def setup(self, width: int, height: int) -> None:
        """Initialize the downsample chain.

        Args:
            width: Base resolution width.
            height: Base resolution height.
        """
        self._base_width = width
        self._base_height = height
        self._levels.clear()

        w, h = width, height
        for _ in range(self._max_levels):
            w = max(1, w // 2)
            h = max(1, h // 2)

            if w < 4 or h < 4:
                break

            self._levels.append(DownsampleLevel(width=w, height=h))

    def downsample(
        self,
        source: List[float],
        source_width: int,
        source_height: int,
    ) -> None:
        """Generate all downsample levels from source.

        Args:
            source: Full-resolution source buffer.
            source_width: Source width.
            source_height: Source height.
        """
        if not self._levels:
            self.setup(source_width, source_height)

        current = source
        current_w = source_width
        current_h = source_height

        for level in self._levels:
            level.buffer = self._downsample_2x(
                current, current_w, current_h, level.width, level.height
            )
            current = level.buffer
            current_w = level.width
            current_h = level.height

    def _downsample_2x(
        self,
        source: List[float],
        src_w: int,
        src_h: int,
        dst_w: int,
        dst_h: int,
    ) -> List[float]:
        """Downsample by approximately 2x with box filter.

        Args:
            source: Source buffer.
            src_w: Source width.
            src_h: Source height.
            dst_w: Destination width.
            dst_h: Destination height.

        Returns:
            Downsampled buffer.
        """
        result = [0.0] * (dst_w * dst_h * 4)

        for dy in range(dst_h):
            for dx in range(dst_w):
                # Map destination to source coordinates
                sx = int(dx * src_w / dst_w)
                sy = int(dy * src_h / dst_h)

                # 2x2 box filter
                color = [0.0, 0.0, 0.0, 0.0]
                count = 0

                for oy in range(2):
                    for ox in range(2):
                        px = min(src_w - 1, sx + ox)
                        py = min(src_h - 1, sy + oy)
                        idx = (py * src_w + px) * 4
                        for c in range(4):
                            color[c] += source[idx + c]
                        count += 1

                out_idx = (dy * dst_w + dx) * 4
                for c in range(4):
                    result[out_idx + c] = color[c] / count

        return result

    def get_level(self, index: int) -> Optional[DownsampleLevel]:
        """Get a specific downsample level.

        Args:
            index: Level index (0 = first downsample).

        Returns:
            Level or None if out of range.
        """
        if 0 <= index < len(self._levels):
            return self._levels[index]
        return None

    def get_level_for_roughness(self, roughness: float) -> Tuple[int, float]:
        """Get the appropriate level for a roughness value.

        Args:
            roughness: Material roughness [0, 1].

        Returns:
            Tuple of (level_index, blend_factor).
        """
        if roughness <= SSR_BLUR.ROUGHNESS_MIRROR:
            return 0, 0.0

        if not self._levels:
            return 0, 0.0

        # Map roughness^2 to level
        blur_amount = roughness ** SSR_BLUR.ROUGHNESS_POWER
        level_float = blur_amount * (len(self._levels) - 1)
        level_int = int(level_float)
        blend = level_float - level_int

        level_int = max(0, min(len(self._levels) - 1, level_int))

        return level_int, blend

    def clear(self) -> None:
        """Clear all level buffers."""
        for level in self._levels:
            level.buffer = None
            level.blurred = None


@dataclass
class SSRRoughnessBlurSettings:
    """Settings for SSR roughness blur effect."""

    enabled: bool = True
    quality: SSRBlurQuality = SSRBlurQuality.HIGH
    technique: BlurTechnique = BlurTechnique.GAUSSIAN
    max_blur_radius: float = SSR_BLUR.MAX_BLUR_RADIUS_DEFAULT
    blur_iterations: int = SSR_BLUR.GAUSSIAN_ITERATIONS_DEFAULT
    use_bilateral_upscale: bool = True
    roughness_power: float = SSR_BLUR.ROUGHNESS_POWER
    edge_preservation: bool = True
    depth_threshold: float = SSR_BLUR.DEPTH_THRESHOLD
    normal_threshold: float = SSR_BLUR.NORMAL_THRESHOLD

    def __post_init__(self) -> None:
        if not SSR_BLUR.MAX_BLUR_RADIUS_MIN <= self.max_blur_radius <= SSR_BLUR.MAX_BLUR_RADIUS_MAX:
            raise ValueError(
                f"max_blur_radius must be in [{SSR_BLUR.MAX_BLUR_RADIUS_MIN}, "
                f"{SSR_BLUR.MAX_BLUR_RADIUS_MAX}], got {self.max_blur_radius}"
            )
        if self.blur_iterations < 1:
            raise ValueError(
                f"blur_iterations must be >= 1, got {self.blur_iterations}"
            )
        if not 1.0 <= self.roughness_power <= 4.0:
            raise ValueError(
                f"roughness_power must be in [1, 4], got {self.roughness_power}"
            )

    def lerp(self, other: "SSRRoughnessBlurSettings", t: float) -> "SSRRoughnessBlurSettings":
        """Interpolate between two settings.

        Args:
            other: Target settings.
            t: Interpolation factor [0, 1].

        Returns:
            Interpolated settings.
        """
        return SSRRoughnessBlurSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            quality=self.quality if t < 0.5 else other.quality,
            technique=self.technique if t < 0.5 else other.technique,
            max_blur_radius=self.max_blur_radius + (other.max_blur_radius - self.max_blur_radius) * t,
            blur_iterations=int(
                self.blur_iterations + (other.blur_iterations - self.blur_iterations) * t
            ),
            use_bilateral_upscale=self.use_bilateral_upscale if t < 0.5 else other.use_bilateral_upscale,
            roughness_power=self.roughness_power + (other.roughness_power - self.roughness_power) * t,
            edge_preservation=self.edge_preservation if t < 0.5 else other.edge_preservation,
            depth_threshold=self.depth_threshold + (other.depth_threshold - self.depth_threshold) * t,
            normal_threshold=self.normal_threshold + (other.normal_threshold - self.normal_threshold) * t,
        )


class SSRRoughnessBlur:
    """Roughness-driven SSR blur processor.

    Implements Bloomberg-style multi-level blur where the blur kernel
    radius scales with roughness squared:

        kernel_radius = roughness^power * max_radius

    This ensures:
    - Smooth surfaces (low roughness) get sharp reflections
    - Rough surfaces (high roughness) get appropriately blurry reflections

    The implementation uses:
    1. Multi-level downsample chain for efficient blur lookup
    2. Separable Gaussian blur at each level
    3. Edge-aware bilateral upscaling to preserve silhouettes
    """

    def __init__(
        self,
        settings: Optional[SSRRoughnessBlurSettings] = None,
    ) -> None:
        """Initialize the SSR roughness blur processor.

        Args:
            settings: Blur configuration settings.
        """
        self._settings: SSRRoughnessBlurSettings = settings or SSRRoughnessBlurSettings()
        self._downsample_chain: DownsampleChain = DownsampleChain()
        self._gaussian_blur: GaussianBlur = GaussianBlur()
        self._bilateral_upscale: BilateralUpscale = BilateralUpscale()

        self._width: int = 0
        self._height: int = 0
        self._is_setup: bool = False

        self._configure_from_settings()

    @property
    def settings(self) -> SSRRoughnessBlurSettings:
        """Current blur settings."""
        return self._settings

    @settings.setter
    def settings(self, value: SSRRoughnessBlurSettings) -> None:
        self._settings = value
        self._configure_from_settings()

    @property
    def width(self) -> int:
        """Current buffer width."""
        return self._width

    @property
    def height(self) -> int:
        """Current buffer height."""
        return self._height

    @property
    def is_setup(self) -> bool:
        """Whether the processor is initialized."""
        return self._is_setup

    @property
    def downsample_chain(self) -> DownsampleChain:
        """The internal downsample chain."""
        return self._downsample_chain

    def _configure_from_settings(self) -> None:
        """Configure components from current settings."""
        # Set downsample levels based on quality
        levels = {
            SSRBlurQuality.LOW: SSR_BLUR.DOWNSAMPLE_LEVELS_LOW,
            SSRBlurQuality.MEDIUM: SSR_BLUR.DOWNSAMPLE_LEVELS_MEDIUM,
            SSRBlurQuality.HIGH: SSR_BLUR.DOWNSAMPLE_LEVELS_HIGH,
            SSRBlurQuality.ULTRA: SSR_BLUR.DOWNSAMPLE_LEVELS_ULTRA,
        }.get(self._settings.quality, SSR_BLUR.DOWNSAMPLE_LEVELS_HIGH)

        self._downsample_chain = DownsampleChain(max_levels=levels)

        # Configure Gaussian blur
        radius = int(self._settings.max_blur_radius / 4)
        self._gaussian_blur = GaussianBlur(
            radius=max(2, radius),
            sigma=radius / 2.0,
        )

    def setup(self, width: int, height: int) -> None:
        """Initialize blur resources.

        Args:
            width: Render width in pixels.
            height: Render height in pixels.
        """
        if width < 1 or height < 1:
            raise ValueError(f"Invalid dimensions: {width}x{height}")

        self._width = width
        self._height = height
        self._downsample_chain.setup(width, height)
        self._is_setup = True

    def calculate_blur_radius(self, roughness: float) -> float:
        """Calculate blur radius for a given roughness.

        Uses the formula: radius = roughness^power * max_radius

        Args:
            roughness: Material roughness [0, 1].

        Returns:
            Blur radius in pixels.
        """
        if roughness <= SSR_BLUR.ROUGHNESS_MIRROR:
            return 0.0

        roughness = max(0.0, min(1.0, roughness))
        blur_amount = roughness ** self._settings.roughness_power
        return blur_amount * self._settings.max_blur_radius

    def blur_reflection(
        self,
        ssr_buffer: List[float],
        roughness_buffer: List[float],
        width: int,
        height: int,
        depth_buffer: Optional[List[float]] = None,
        normal_buffer: Optional[List[float]] = None,
    ) -> List[float]:
        """Apply roughness-driven blur to SSR buffer.

        Args:
            ssr_buffer: SSR color buffer (flat RGBA).
            roughness_buffer: Per-pixel roughness values.
            width: Buffer width.
            height: Buffer height.
            depth_buffer: Optional depth for edge preservation.
            normal_buffer: Optional normals for edge preservation.

        Returns:
            Blurred SSR buffer.
        """
        if not self._settings.enabled:
            return ssr_buffer

        if not self._is_setup or width != self._width or height != self._height:
            self.setup(width, height)

        # Generate downsample chain
        self._downsample_chain.downsample(ssr_buffer, width, height)

        # Blur each level
        for level in self._downsample_chain.levels:
            if level.buffer:
                level.blurred = self._gaussian_blur.blur(
                    level.buffer,
                    level.width,
                    level.height,
                    self._settings.blur_iterations,
                )

        # Sample from appropriate levels based on roughness
        result = self._sample_by_roughness(
            ssr_buffer,
            roughness_buffer,
            width,
            height,
            depth_buffer,
            normal_buffer,
        )

        return result

    def _sample_by_roughness(
        self,
        original: List[float],
        roughness: List[float],
        width: int,
        height: int,
        depth_buffer: Optional[List[float]],
        normal_buffer: Optional[List[float]],
    ) -> List[float]:
        """Sample from downsample chain based on per-pixel roughness.

        Args:
            original: Original SSR buffer (for smooth surfaces).
            roughness: Per-pixel roughness values.
            width: Buffer width.
            height: Buffer height.
            depth_buffer: Optional depth buffer.
            normal_buffer: Optional normal buffer.

        Returns:
            Roughness-sampled result buffer.
        """
        result = [0.0] * (width * height * 4)

        for y in range(height):
            for x in range(width):
                pixel_idx = y * width + x
                r = roughness[pixel_idx] if pixel_idx < len(roughness) else 0.0

                if r <= SSR_BLUR.ROUGHNESS_MIRROR:
                    # Perfect mirror - use original
                    src_idx = pixel_idx * 4
                    for c in range(4):
                        result[src_idx + c] = original[src_idx + c]
                else:
                    # Get level and blend factor
                    level_idx, blend = self._downsample_chain.get_level_for_roughness(r)

                    # Sample from level(s)
                    color = self._sample_level(
                        x, y, level_idx, blend, width, height, original
                    )

                    out_idx = pixel_idx * 4
                    for c in range(4):
                        result[out_idx + c] = color[c]

        return result

    def _sample_level(
        self,
        x: int,
        y: int,
        level_idx: int,
        blend: float,
        width: int,
        height: int,
        original: List[float],
    ) -> List[float]:
        """Sample from a specific downsample level.

        Args:
            x: X coordinate in full resolution.
            y: Y coordinate in full resolution.
            level_idx: Downsample level index.
            blend: Blend factor to next level.
            width: Full resolution width.
            height: Full resolution height.
            original: Original buffer for level 0.

        Returns:
            RGBA color sample.
        """
        color = [0.0, 0.0, 0.0, 0.0]

        # Get current level
        level = self._downsample_chain.get_level(level_idx)
        if level is None or level.blurred is None:
            # Fallback to original
            idx = (y * width + x) * 4
            return [original[idx + c] for c in range(4)]

        # Sample from current level
        lx = x * level.width // width
        ly = y * level.height // height
        lx = max(0, min(level.width - 1, lx))
        ly = max(0, min(level.height - 1, ly))
        idx = (ly * level.width + lx) * 4

        for c in range(4):
            color[c] = level.blurred[idx + c]

        # Blend with next level if needed
        if blend > 0.0:
            next_level = self._downsample_chain.get_level(level_idx + 1)
            if next_level is not None and next_level.blurred is not None:
                nlx = x * next_level.width // width
                nly = y * next_level.height // height
                nlx = max(0, min(next_level.width - 1, nlx))
                nly = max(0, min(next_level.height - 1, nly))
                n_idx = (nly * next_level.width + nlx) * 4

                for c in range(4):
                    color[c] = color[c] * (1.0 - blend) + next_level.blurred[n_idx + c] * blend

        return color

    def blur_uniform(
        self,
        ssr_buffer: List[float],
        roughness: float,
        width: int,
        height: int,
    ) -> List[float]:
        """Apply uniform blur for a single roughness value.

        Optimized path when roughness is constant across the image.

        Args:
            ssr_buffer: SSR color buffer.
            roughness: Uniform roughness value.
            width: Buffer width.
            height: Buffer height.

        Returns:
            Blurred buffer.
        """
        if not self._settings.enabled or roughness <= SSR_BLUR.ROUGHNESS_MIRROR:
            return ssr_buffer

        if not self._is_setup or width != self._width or height != self._height:
            self.setup(width, height)

        # Calculate blur radius
        radius = self.calculate_blur_radius(roughness)
        if radius < 1.0:
            return ssr_buffer

        # Configure blur for this radius
        blur_radius = max(2, int(radius / 2))
        blur = GaussianBlur(radius=blur_radius, sigma=blur_radius / 2.0)

        return blur.blur(ssr_buffer, width, height, self._settings.blur_iterations)

    def upscale_with_edges(
        self,
        low_res: List[float],
        low_width: int,
        low_height: int,
        high_width: int,
        high_height: int,
        depth_buffer: Optional[List[float]] = None,
        normal_buffer: Optional[List[float]] = None,
    ) -> List[float]:
        """Upscale buffer with edge-aware filtering.

        Args:
            low_res: Low-resolution buffer.
            low_width: Low-resolution width.
            low_height: Low-resolution height.
            high_width: Target width.
            high_height: Target height.
            depth_buffer: High-res depth for edge detection.
            normal_buffer: High-res normals for edge detection.

        Returns:
            Upscaled buffer.
        """
        if self._settings.use_bilateral_upscale:
            return self._bilateral_upscale.upscale(
                low_res,
                low_width,
                low_height,
                high_width,
                high_height,
                depth_buffer,
                normal_buffer,
            )
        else:
            return self._bilateral_upscale.upscale_simple(
                low_res,
                low_width,
                low_height,
                high_width,
                high_height,
            )

    def cleanup(self) -> None:
        """Release all resources."""
        self._downsample_chain.clear()
        self._is_setup = False
        self._width = 0
        self._height = 0


__all__ = [
    "BlurTechnique",
    "SSRBlurQuality",
    "SSRBlurConstants",
    "SSR_BLUR",
    "MaterialReflectionParams",
    "GaussianBlur",
    "BilateralUpscale",
    "DownsampleChain",
    "DownsampleLevel",
    "SSRRoughnessBlur",
    "SSRRoughnessBlurSettings",
]
