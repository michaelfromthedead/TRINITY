"""RT Reflection Denoising System (T-GIR-P8.4).

Implements specialized denoising for ray-traced reflections using:
- A-Trous wavelet spatial filter (4-5 iterations, depth/normal/luminance edge-stopping)
- Temporal accumulation (8-16 frames, velocity buffer reprojection)
- Bilateral upscale from half-resolution to full-resolution

Key Components:
- ReflectionEdgeStopFunctions: Edge-stopping weights (depth, normal, luminance)
- ReflectionATrousFilter: Multi-pass A-Trous wavelet spatial filter
- ReflectionTemporalAccumulator: Velocity-based temporal accumulation
- ReflectionBilateralUpscale: Edge-aware upsampling
- RTReflectionDenoisePipeline: Complete denoise pipeline

WGSL Shader Generated:
- rt_reflections_denoise.comp.wgsl

Performance Target: <2ms total denoise at 1080p

References:
- "Edge-Avoiding A-Trous Wavelet Transform" Dammertz et al., HPG 2010
- "SVGF: Spatiotemporal Variance-Guided Filtering" Schied et al., HPG 2017
- Section 6.11 Ray Tracing Architecture in RENDERING_CONTEXT.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

from engine.core.math.vec import Vec2, Vec3

if TYPE_CHECKING:
    from engine.platform.rhi.device import Device
    from engine.platform.rhi.resources import Texture


# =============================================================================
# Constants
# =============================================================================

# Numerical safety
EPSILON = 1e-6
LUMINANCE_EPSILON = 1e-4
DEPTH_EPSILON = 1e-5

# Default edge-stopping parameters
DEFAULT_SIGMA_DEPTH = 1.0
DEFAULT_SIGMA_NORMAL = 128.0
DEFAULT_SIGMA_LUMINANCE = 4.0

# Temporal accumulation defaults
DEFAULT_TEMPORAL_ALPHA = 0.1  # EMA blend: 10% current, 90% history
FAST_TEMPORAL_ALPHA = 0.2    # More responsive
SLOW_TEMPORAL_ALPHA = 0.05   # More stable
MIN_TEMPORAL_ALPHA = 0.02    # Minimum for very stable scenarios
MAX_TEMPORAL_ALPHA = 0.5     # Maximum for fast response

# History length bounds
MIN_HISTORY_FRAMES = 1
MAX_HISTORY_FRAMES = 64
DEFAULT_HISTORY_FRAMES = 16
CONVERGENCE_HISTORY_FRAMES = 12  # Frames to reach stable output

# A-Trous filter parameters
DEFAULT_ATROUS_ITERATIONS = 4
MAX_ATROUS_ITERATIONS = 5
DEFAULT_DILATIONS: Tuple[int, ...] = (1, 2, 4, 8, 16)

# Standard 5x5 Gaussian kernel for A-Trous wavelet
GAUSSIAN_5X5_KERNEL: Tuple[float, ...] = (
    1.0 / 256.0,  4.0 / 256.0,  6.0 / 256.0,  4.0 / 256.0, 1.0 / 256.0,
    4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
    6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
    4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
    1.0 / 256.0,  4.0 / 256.0,  6.0 / 256.0,  4.0 / 256.0, 1.0 / 256.0,
)

# Bilateral upscale parameters
BILATERAL_RADIUS = 2
BILATERAL_SIGMA_SPATIAL = 1.0
BILATERAL_SIGMA_RANGE = 0.1

# Disocclusion thresholds
DEPTH_REJECT_THRESHOLD = 0.1    # 10% relative depth difference
NORMAL_REJECT_THRESHOLD = 0.9   # Dot product threshold
VELOCITY_REJECT_THRESHOLD = 2.0  # Pixels per frame

# Workgroup size for compute shaders
WORKGROUP_SIZE_X = 8
WORKGROUP_SIZE_Y = 8


# =============================================================================
# Quality Presets
# =============================================================================


class ReflectionDenoiseQuality(IntEnum):
    """Quality presets for RT reflection denoising.

    Each level controls iteration count, temporal stability, and filter extent.
    """

    LOW = 1
    """2 A-Trous iterations, 8 frame history - fast but noisy."""

    MEDIUM = 2
    """3 A-Trous iterations, 12 frame history - balanced."""

    HIGH = 3
    """4 A-Trous iterations, 16 frame history - high quality."""

    ULTRA = 4
    """5 A-Trous iterations, 24 frame history - maximum quality."""


@dataclass(frozen=True)
class QualityPresetParams:
    """Parameters for a quality preset.

    Attributes:
        spatial_iterations: Number of A-Trous filter passes.
        temporal_alpha: EMA blend factor.
        history_frames: Target frames for convergence.
        sigma_depth: Depth edge-stopping sensitivity.
        sigma_normal: Normal edge-stopping power.
        sigma_luminance: Luminance edge-stopping sensitivity.
    """

    spatial_iterations: int
    temporal_alpha: float
    history_frames: int
    sigma_depth: float
    sigma_normal: float
    sigma_luminance: float


QUALITY_PRESETS: Dict[ReflectionDenoiseQuality, QualityPresetParams] = {
    ReflectionDenoiseQuality.LOW: QualityPresetParams(
        spatial_iterations=2,
        temporal_alpha=0.2,
        history_frames=8,
        sigma_depth=1.5,
        sigma_normal=64.0,
        sigma_luminance=6.0,
    ),
    ReflectionDenoiseQuality.MEDIUM: QualityPresetParams(
        spatial_iterations=3,
        temporal_alpha=0.12,
        history_frames=12,
        sigma_depth=1.0,
        sigma_normal=128.0,
        sigma_luminance=4.0,
    ),
    ReflectionDenoiseQuality.HIGH: QualityPresetParams(
        spatial_iterations=4,
        temporal_alpha=0.08,
        history_frames=16,
        sigma_depth=0.8,
        sigma_normal=128.0,
        sigma_luminance=3.0,
    ),
    ReflectionDenoiseQuality.ULTRA: QualityPresetParams(
        spatial_iterations=5,
        temporal_alpha=0.05,
        history_frames=24,
        sigma_depth=0.5,
        sigma_normal=256.0,
        sigma_luminance=2.0,
    ),
}


# =============================================================================
# Color Space Conversion (YCoCg)
# =============================================================================


class YCoCgConverter:
    """RGB to YCoCg color space conversion for luminance-based edge detection.

    YCoCg separates luminance (Y) from chrominance (Co, Cg),
    allowing more perceptually accurate edge detection.

    Conversion matrices:
        Y  =  0.25 * R + 0.5 * G + 0.25 * B
        Co =  0.5  * R           - 0.5  * B
        Cg = -0.25 * R + 0.5 * G - 0.25 * B
    """

    @staticmethod
    def rgb_to_ycocg(r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Convert RGB to YCoCg color space.

        Args:
            r: Red channel [0, 1].
            g: Green channel [0, 1].
            b: Blue channel [0, 1].

        Returns:
            Tuple of (Y, Co, Cg) values.
        """
        y = 0.25 * r + 0.5 * g + 0.25 * b
        co = 0.5 * r - 0.5 * b
        cg = -0.25 * r + 0.5 * g - 0.25 * b
        return (y, co, cg)

    @staticmethod
    def ycocg_to_rgb(y: float, co: float, cg: float) -> Tuple[float, float, float]:
        """Convert YCoCg to RGB color space.

        Args:
            y: Luminance.
            co: Orange chrominance.
            cg: Green chrominance.

        Returns:
            Tuple of (R, G, B) values.
        """
        r = y + co - cg
        g = y + cg
        b = y - co - cg
        return (r, g, b)

    @staticmethod
    def luminance(r: float, g: float, b: float) -> float:
        """Extract luminance from RGB (Y channel of YCoCg).

        Args:
            r: Red channel [0, 1].
            g: Green channel [0, 1].
            b: Blue channel [0, 1].

        Returns:
            Luminance value.
        """
        return 0.25 * r + 0.5 * g + 0.25 * b

    @staticmethod
    def luminance_vec3(color: Vec3) -> float:
        """Extract luminance from Vec3 RGB.

        Args:
            color: RGB color as Vec3.

        Returns:
            Luminance value.
        """
        return 0.25 * color.x + 0.5 * color.y + 0.25 * color.z

    @staticmethod
    def bt709_luminance(r: float, g: float, b: float) -> float:
        """Extract luminance using BT.709 coefficients.

        Args:
            r: Red channel [0, 1].
            g: Green channel [0, 1].
            b: Blue channel [0, 1].

        Returns:
            BT.709 luminance value.
        """
        return 0.2126 * r + 0.7152 * g + 0.0722 * b


# =============================================================================
# Edge-Stopping Functions for Reflections
# =============================================================================


@dataclass
class EdgeStopWeights:
    """Combined edge-stopping weights for a reflection sample.

    Stores individual weights from each edge-stopping function
    and provides the combined weight.

    Attributes:
        depth: Weight from depth edge-stopping [0, 1].
        normal: Weight from normal edge-stopping [0, 1].
        luminance: Weight from luminance edge-stopping [0, 1].
        kernel: Weight from spatial kernel.
    """

    depth: float = 1.0
    normal: float = 1.0
    luminance: float = 1.0
    kernel: float = 1.0

    def combined(self) -> float:
        """Calculate combined edge-stopping weight.

        Returns:
            Product of all weights.
        """
        return self.depth * self.normal * self.luminance * self.kernel

    def is_valid(self) -> bool:
        """Check if weights are valid (non-negative and bounded).

        Returns:
            True if all weights are in valid range.
        """
        return (
            0.0 <= self.depth <= 1.0
            and 0.0 <= self.normal <= 1.0
            and 0.0 <= self.luminance <= 1.0
            and self.kernel >= 0.0
        )


class ReflectionEdgeStopFunctions:
    """Edge-stopping functions specialized for RT reflection denoising.

    Provides depth, normal, and luminance edge-stopping weights
    to preserve reflection edges while removing Monte Carlo noise.

    Usage:
        edge_stop = ReflectionEdgeStopFunctions(sigma_depth=1.0, sigma_normal=128.0)
        w_depth = edge_stop.depth_weight(z0, z1)
        w_normal = edge_stop.normal_weight(n0, n1)
        w_lum = edge_stop.luminance_weight(L0, L1)
        w_combined = edge_stop.combined_weight(z0, z1, n0, n1, L0, L1)
    """

    def __init__(
        self,
        sigma_depth: float = DEFAULT_SIGMA_DEPTH,
        sigma_normal: float = DEFAULT_SIGMA_NORMAL,
        sigma_luminance: float = DEFAULT_SIGMA_LUMINANCE,
    ) -> None:
        """Initialize edge-stopping functions.

        Args:
            sigma_depth: Depth sensitivity (higher = more tolerant of depth diff).
            sigma_normal: Normal power exponent (higher = sharper normal falloff).
            sigma_luminance: Luminance sensitivity (higher = more tolerant).

        Raises:
            ValueError: If any sigma is not positive.
        """
        if sigma_depth <= 0.0:
            raise ValueError(f"sigma_depth must be positive, got {sigma_depth}")
        if sigma_normal <= 0.0:
            raise ValueError(f"sigma_normal must be positive, got {sigma_normal}")
        if sigma_luminance <= 0.0:
            raise ValueError(f"sigma_luminance must be positive, got {sigma_luminance}")

        self._sigma_depth = sigma_depth
        self._sigma_normal = sigma_normal
        self._sigma_luminance = sigma_luminance
        self._converter = YCoCgConverter()

    @property
    def sigma_depth(self) -> float:
        """Get depth sensitivity."""
        return self._sigma_depth

    @sigma_depth.setter
    def sigma_depth(self, value: float) -> None:
        """Set depth sensitivity."""
        if value <= 0.0:
            raise ValueError(f"sigma_depth must be positive, got {value}")
        self._sigma_depth = value

    @property
    def sigma_normal(self) -> float:
        """Get normal power exponent."""
        return self._sigma_normal

    @sigma_normal.setter
    def sigma_normal(self, value: float) -> None:
        """Set normal power exponent."""
        if value <= 0.0:
            raise ValueError(f"sigma_normal must be positive, got {value}")
        self._sigma_normal = value

    @property
    def sigma_luminance(self) -> float:
        """Get luminance sensitivity."""
        return self._sigma_luminance

    @sigma_luminance.setter
    def sigma_luminance(self, value: float) -> None:
        """Set luminance sensitivity."""
        if value <= 0.0:
            raise ValueError(f"sigma_luminance must be positive, got {value}")
        self._sigma_luminance = value

    def depth_weight(
        self,
        depth_center: float,
        depth_sample: float,
        gradient: float = 1.0,
    ) -> float:
        """Calculate depth edge-stopping weight.

        Formula: w = exp(-|z1 - z0| / (sigma_z * gradient))

        Args:
            depth_center: Depth at center pixel.
            depth_sample: Depth at sample pixel.
            gradient: Local depth gradient for adaptive scaling.

        Returns:
            Edge-stopping weight [0, 1].
        """
        if gradient < DEPTH_EPSILON:
            gradient = DEPTH_EPSILON

        depth_diff = abs(depth_sample - depth_center)
        denominator = self._sigma_depth * gradient

        if denominator < EPSILON:
            denominator = EPSILON

        exponent = -depth_diff / denominator

        # Clamp exponent to avoid underflow
        exponent = max(exponent, -88.0)

        return math.exp(exponent)

    def depth_weight_relative(
        self,
        depth_center: float,
        depth_sample: float,
    ) -> float:
        """Calculate depth weight using relative depth difference.

        More robust for varying scene depths.

        Args:
            depth_center: Depth at center pixel.
            depth_sample: Depth at sample pixel.

        Returns:
            Edge-stopping weight [0, 1].
        """
        if abs(depth_center) < DEPTH_EPSILON:
            return 1.0 if abs(depth_sample) < DEPTH_EPSILON else 0.0

        relative_diff = abs(depth_sample - depth_center) / max(
            abs(depth_center), DEPTH_EPSILON
        )

        exponent = -relative_diff / self._sigma_depth
        exponent = max(exponent, -88.0)

        return math.exp(exponent)

    def normal_weight(
        self,
        normal_center: Tuple[float, float, float],
        normal_sample: Tuple[float, float, float],
    ) -> float:
        """Calculate normal edge-stopping weight.

        Formula: w = max(0, dot(n0, n1))^sigma_n

        Args:
            normal_center: Surface normal at center pixel (normalized).
            normal_sample: Surface normal at sample pixel (normalized).

        Returns:
            Edge-stopping weight [0, 1].
        """
        # Dot product of normals
        dot = (
            normal_center[0] * normal_sample[0]
            + normal_center[1] * normal_sample[1]
            + normal_center[2] * normal_sample[2]
        )

        # Clamp to [0, 1]
        dot = max(0.0, min(1.0, dot))

        # Apply power exponent
        return math.pow(dot, self._sigma_normal)

    def normal_weight_vec3(self, normal_center: Vec3, normal_sample: Vec3) -> float:
        """Calculate normal weight using Vec3 normals.

        Args:
            normal_center: Surface normal at center pixel.
            normal_sample: Surface normal at sample pixel.

        Returns:
            Edge-stopping weight [0, 1].
        """
        dot = (
            normal_center.x * normal_sample.x
            + normal_center.y * normal_sample.y
            + normal_center.z * normal_sample.z
        )
        dot = max(0.0, min(1.0, dot))
        return math.pow(dot, self._sigma_normal)

    def luminance_weight(
        self,
        luminance_center: float,
        luminance_sample: float,
        variance: float = 0.0,
    ) -> float:
        """Calculate luminance edge-stopping weight.

        Formula: w = exp(-|L1 - L0| / (sigma_l + variance))

        Args:
            luminance_center: Luminance at center pixel.
            luminance_sample: Luminance at sample pixel.
            variance: Local variance for adaptive sigma.

        Returns:
            Edge-stopping weight [0, 1].
        """
        lum_diff = abs(luminance_sample - luminance_center)
        denominator = self._sigma_luminance + max(0.0, variance)

        if denominator < LUMINANCE_EPSILON:
            denominator = LUMINANCE_EPSILON

        exponent = -lum_diff / denominator
        exponent = max(exponent, -88.0)

        return math.exp(exponent)

    def luminance_weight_rgb(
        self,
        color_center: Tuple[float, float, float],
        color_sample: Tuple[float, float, float],
        variance: float = 0.0,
    ) -> float:
        """Calculate luminance weight from RGB colors.

        Args:
            color_center: RGB color at center pixel.
            color_sample: RGB color at sample pixel.
            variance: Local variance for adaptive sigma.

        Returns:
            Edge-stopping weight [0, 1].
        """
        lum_center = self._converter.luminance(*color_center)
        lum_sample = self._converter.luminance(*color_sample)
        return self.luminance_weight(lum_center, lum_sample, variance)

    def combined_weight(
        self,
        depth_center: float,
        depth_sample: float,
        normal_center: Tuple[float, float, float],
        normal_sample: Tuple[float, float, float],
        luminance_center: float,
        luminance_sample: float,
        depth_gradient: float = 1.0,
        variance: float = 0.0,
    ) -> float:
        """Calculate combined edge-stopping weight.

        Combines depth, normal, and luminance weights multiplicatively.

        Args:
            depth_center: Depth at center pixel.
            depth_sample: Depth at sample pixel.
            normal_center: Normal at center pixel.
            normal_sample: Normal at sample pixel.
            luminance_center: Luminance at center pixel.
            luminance_sample: Luminance at sample pixel.
            depth_gradient: Local depth gradient.
            variance: Local luminance variance.

        Returns:
            Combined edge-stopping weight [0, 1].
        """
        w_depth = self.depth_weight(depth_center, depth_sample, depth_gradient)
        w_normal = self.normal_weight(normal_center, normal_sample)
        w_luminance = self.luminance_weight(
            luminance_center, luminance_sample, variance
        )

        return w_depth * w_normal * w_luminance

    def combined_weight_full(
        self,
        depth_center: float,
        depth_sample: float,
        normal_center: Tuple[float, float, float],
        normal_sample: Tuple[float, float, float],
        luminance_center: float,
        luminance_sample: float,
        kernel_weight: float,
        depth_gradient: float = 1.0,
        variance: float = 0.0,
    ) -> EdgeStopWeights:
        """Calculate all edge-stopping weights with details.

        Args:
            depth_center: Depth at center pixel.
            depth_sample: Depth at sample pixel.
            normal_center: Normal at center pixel.
            normal_sample: Normal at sample pixel.
            luminance_center: Luminance at center pixel.
            luminance_sample: Luminance at sample pixel.
            kernel_weight: Spatial kernel weight.
            depth_gradient: Local depth gradient.
            variance: Local luminance variance.

        Returns:
            EdgeStopWeights with individual and combined weights.
        """
        return EdgeStopWeights(
            depth=self.depth_weight(depth_center, depth_sample, depth_gradient),
            normal=self.normal_weight(normal_center, normal_sample),
            luminance=self.luminance_weight(
                luminance_center, luminance_sample, variance
            ),
            kernel=kernel_weight,
        )


# =============================================================================
# A-Trous Wavelet Filter for Reflections
# =============================================================================


@dataclass
class ATrousIterationResult:
    """Result of a single A-Trous filter iteration.

    Attributes:
        iteration: Iteration index (0-based).
        dilation: Dilation factor used.
        output_buffer: Index of output buffer (0 or 1 for ping-pong).
        samples_processed: Number of pixels processed.
        edge_stops_triggered: Count of edge-stop rejections.
    """

    iteration: int
    dilation: int
    output_buffer: int
    samples_processed: int = 0
    edge_stops_triggered: int = 0


@dataclass
class ATrousFilterResult:
    """Result of complete A-Trous filtering.

    Attributes:
        iterations: List of per-iteration results.
        final_buffer: Index of final output buffer.
        total_samples: Total samples processed across all iterations.
        total_edge_stops: Total edge-stop rejections.
        elapsed_ms: Time taken in milliseconds.
    """

    iterations: List[ATrousIterationResult]
    final_buffer: int
    total_samples: int = 0
    total_edge_stops: int = 0
    elapsed_ms: float = 0.0


class ReflectionATrousFilter:
    """A-Trous wavelet spatial filter specialized for RT reflections.

    Implements 4-5 iteration A-Trous filter with increasing dilation (1,2,4,8,16)
    and edge-stopping functions to preserve reflection edges.

    Usage:
        filter = ReflectionATrousFilter(iterations=4)
        filter.set_edge_functions(edge_stop)
        filter.set_buffers(ping, pong)
        result = filter.filter_full(noisy_reflections, gbuffer)
    """

    def __init__(
        self,
        iterations: int = DEFAULT_ATROUS_ITERATIONS,
        kernel: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """Initialize A-Trous filter.

        Args:
            iterations: Number of filter passes (2-5).
            kernel: 5x5 filter kernel weights (default: Gaussian).

        Raises:
            ValueError: If iterations is out of valid range.
        """
        if not (1 <= iterations <= MAX_ATROUS_ITERATIONS):
            raise ValueError(
                f"iterations must be 1-{MAX_ATROUS_ITERATIONS}, got {iterations}"
            )

        self._iterations = iterations
        self._kernel = kernel if kernel is not None else GAUSSIAN_5X5_KERNEL
        self._dilations = DEFAULT_DILATIONS[:iterations]

        # Edge-stopping functions (optional)
        self._edge_functions: Optional[ReflectionEdgeStopFunctions] = None

        # Ping-pong buffers for in-place filtering
        self._buffer_a: Optional[List[List[Vec3]]] = None
        self._buffer_b: Optional[List[List[Vec3]]] = None
        self._current_buffer = 0

        # G-buffer references
        self._depth_buffer: Optional[List[List[float]]] = None
        self._normal_buffer: Optional[List[List[Vec3]]] = None

        # Dimensions
        self._width = 0
        self._height = 0

    @property
    def iterations(self) -> int:
        """Get number of filter iterations."""
        return self._iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        """Set number of filter iterations."""
        if not (1 <= value <= MAX_ATROUS_ITERATIONS):
            raise ValueError(
                f"iterations must be 1-{MAX_ATROUS_ITERATIONS}, got {value}"
            )
        self._iterations = value
        self._dilations = DEFAULT_DILATIONS[:value]

    @property
    def dilations(self) -> Tuple[int, ...]:
        """Get dilation sequence."""
        return self._dilations

    def get_dilation(self, iteration: int) -> int:
        """Get dilation factor for a specific iteration.

        Args:
            iteration: Iteration index (0-based).

        Returns:
            Dilation factor (power of 2).

        Raises:
            IndexError: If iteration is out of range.
        """
        if not (0 <= iteration < self._iterations):
            raise IndexError(
                f"iteration {iteration} out of range [0, {self._iterations})"
            )
        return self._dilations[iteration]

    def set_edge_functions(
        self, edge_functions: ReflectionEdgeStopFunctions
    ) -> None:
        """Set edge-stopping functions.

        Args:
            edge_functions: Edge-stopping function calculator.
        """
        self._edge_functions = edge_functions

    def set_buffers(
        self,
        buffer_a: List[List[Vec3]],
        buffer_b: List[List[Vec3]],
    ) -> None:
        """Set ping-pong buffers for in-place filtering.

        Args:
            buffer_a: First buffer (input initially).
            buffer_b: Second buffer (scratch).

        Raises:
            ValueError: If buffers have different dimensions.
        """
        if len(buffer_a) != len(buffer_b):
            raise ValueError("Buffers must have same height")
        if buffer_a and buffer_b and len(buffer_a[0]) != len(buffer_b[0]):
            raise ValueError("Buffers must have same width")

        self._buffer_a = buffer_a
        self._buffer_b = buffer_b
        self._height = len(buffer_a)
        self._width = len(buffer_a[0]) if buffer_a else 0
        self._current_buffer = 0

    def set_gbuffer(
        self,
        depth_buffer: List[List[float]],
        normal_buffer: List[List[Vec3]],
    ) -> None:
        """Set G-buffer data for edge-stopping.

        Args:
            depth_buffer: Depth values per pixel.
            normal_buffer: Surface normals per pixel.
        """
        self._depth_buffer = depth_buffer
        self._normal_buffer = normal_buffer

    def get_kernel_weight(self, dx: int, dy: int) -> float:
        """Get kernel weight at offset.

        Args:
            dx: Horizontal offset from center [-2, 2].
            dy: Vertical offset from center [-2, 2].

        Returns:
            Kernel weight at position.
        """
        if not (-2 <= dx <= 2 and -2 <= dy <= 2):
            return 0.0
        ix = dx + 2
        iy = dy + 2
        return self._kernel[iy * 5 + ix]

    def filter_iteration(
        self,
        iteration: int,
        input_buffer: List[List[Vec3]],
        output_buffer: List[List[Vec3]],
    ) -> ATrousIterationResult:
        """Execute single A-Trous filter iteration.

        Args:
            iteration: Iteration index (0-based).
            input_buffer: Source color buffer.
            output_buffer: Destination color buffer.

        Returns:
            ATrousIterationResult with statistics.
        """
        dilation = self.get_dilation(iteration)
        samples_processed = 0
        edge_stops_triggered = 0

        for y in range(self._height):
            for x in range(self._width):
                result, edge_stops = self._filter_pixel(
                    x, y, dilation, input_buffer
                )
                output_buffer[y][x] = result
                samples_processed += 1
                edge_stops_triggered += edge_stops

        return ATrousIterationResult(
            iteration=iteration,
            dilation=dilation,
            output_buffer=1 if self._current_buffer == 0 else 0,
            samples_processed=samples_processed,
            edge_stops_triggered=edge_stops_triggered,
        )

    def _filter_pixel(
        self,
        x: int,
        y: int,
        dilation: int,
        input_buffer: List[List[Vec3]],
    ) -> Tuple[Vec3, int]:
        """Filter single pixel with edge-stopping.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.
            dilation: Current dilation factor.
            input_buffer: Source color buffer.

        Returns:
            Tuple of (filtered color, edge stop count).
        """
        center_color = input_buffer[y][x]
        center_depth = (
            self._depth_buffer[y][x]
            if self._depth_buffer
            else 0.0
        )
        center_normal = (
            self._normal_buffer[y][x]
            if self._normal_buffer
            else Vec3(0.0, 1.0, 0.0)
        )
        center_luminance = YCoCgConverter.luminance_vec3(center_color)

        sum_color = Vec3(0.0, 0.0, 0.0)
        sum_weight = 0.0
        edge_stops = 0

        # 5x5 kernel with dilation
        for ky in range(-2, 3):
            for kx in range(-2, 3):
                # Dilated sample position
                sx = x + kx * dilation
                sy = y + ky * dilation

                # Skip out-of-bounds
                if not (0 <= sx < self._width and 0 <= sy < self._height):
                    continue

                # Get kernel weight
                kernel_weight = self.get_kernel_weight(kx, ky)

                # Get sample data
                sample_color = input_buffer[sy][sx]
                sample_depth = (
                    self._depth_buffer[sy][sx]
                    if self._depth_buffer
                    else 0.0
                )
                sample_normal = (
                    self._normal_buffer[sy][sx]
                    if self._normal_buffer
                    else Vec3(0.0, 1.0, 0.0)
                )
                sample_luminance = YCoCgConverter.luminance_vec3(sample_color)

                # Calculate edge-stopping weight
                if self._edge_functions:
                    edge_weight = self._edge_functions.combined_weight(
                        center_depth,
                        sample_depth,
                        (center_normal.x, center_normal.y, center_normal.z),
                        (sample_normal.x, sample_normal.y, sample_normal.z),
                        center_luminance,
                        sample_luminance,
                    )
                    if edge_weight < 0.01:
                        edge_stops += 1
                else:
                    edge_weight = 1.0

                # Combined weight
                weight = kernel_weight * edge_weight

                # Accumulate
                sum_color = Vec3(
                    sum_color.x + sample_color.x * weight,
                    sum_color.y + sample_color.y * weight,
                    sum_color.z + sample_color.z * weight,
                )
                sum_weight += weight

        # Normalize
        if sum_weight > EPSILON:
            return (
                Vec3(
                    sum_color.x / sum_weight,
                    sum_color.y / sum_weight,
                    sum_color.z / sum_weight,
                ),
                edge_stops,
            )
        else:
            return center_color, edge_stops

    def filter_full(
        self,
        input_buffer: List[List[Vec3]],
    ) -> ATrousFilterResult:
        """Execute complete A-Trous filtering pipeline.

        Performs all iterations using ping-pong buffers.

        Args:
            input_buffer: Initial noisy reflection buffer.

        Returns:
            ATrousFilterResult with filtered output in appropriate buffer.
        """
        import time

        start_time = time.perf_counter()

        if self._buffer_a is None or self._buffer_b is None:
            raise RuntimeError("Ping-pong buffers not set, call set_buffers() first")

        # Copy input to buffer A
        for y in range(self._height):
            for x in range(self._width):
                self._buffer_a[y][x] = Vec3(
                    input_buffer[y][x].x,
                    input_buffer[y][x].y,
                    input_buffer[y][x].z,
                )

        iterations_results: List[ATrousIterationResult] = []
        total_samples = 0
        total_edge_stops = 0

        # Ping-pong filtering
        current_input = self._buffer_a
        current_output = self._buffer_b

        for i in range(self._iterations):
            result = self.filter_iteration(i, current_input, current_output)
            iterations_results.append(result)
            total_samples += result.samples_processed
            total_edge_stops += result.edge_stops_triggered

            # Swap buffers
            current_input, current_output = current_output, current_input

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Final result is in current_input after last swap
        final_buffer = 0 if current_input is self._buffer_a else 1

        return ATrousFilterResult(
            iterations=iterations_results,
            final_buffer=final_buffer,
            total_samples=total_samples,
            total_edge_stops=total_edge_stops,
            elapsed_ms=elapsed_ms,
        )

    def get_output_buffer(self, final_buffer: int) -> List[List[Vec3]]:
        """Get reference to output buffer after filtering.

        Args:
            final_buffer: Buffer index from ATrousFilterResult.

        Returns:
            Reference to the output buffer.
        """
        if final_buffer == 0:
            return self._buffer_a if self._buffer_a else []
        else:
            return self._buffer_b if self._buffer_b else []


# =============================================================================
# Temporal Accumulator for Reflections
# =============================================================================


@dataclass
class ReprojectionResult:
    """Result of temporal reprojection.

    Attributes:
        prev_uv: Previous frame UV coordinates.
        valid: Whether reprojection is valid.
        confidence: Confidence in history sample [0, 1].
        disoccluded: Whether pixel is disoccluded.
        out_of_bounds: Whether previous UV is out of bounds.
    """

    prev_uv: Vec2
    valid: bool = True
    confidence: float = 1.0
    disoccluded: bool = False
    out_of_bounds: bool = False


@dataclass
class TemporalAccumulationResult:
    """Result of temporal accumulation for a frame.

    Attributes:
        converged_pixels: Count of pixels that have converged.
        disoccluded_pixels: Count of disoccluded pixels.
        average_history_length: Average history length across pixels.
        elapsed_ms: Time taken in milliseconds.
    """

    converged_pixels: int = 0
    disoccluded_pixels: int = 0
    average_history_length: float = 0.0
    elapsed_ms: float = 0.0


class ReflectionTemporalAccumulator:
    """Temporal accumulator for RT reflections with velocity-based reprojection.

    Implements:
    - Velocity buffer reprojection to find previous frame samples
    - History rejection based on depth/normal discontinuity
    - Exponential moving average (alpha = 0.05-0.1)
    - 8-16 frame convergence tracking

    Usage:
        accumulator = ReflectionTemporalAccumulator(alpha=0.1, history_frames=16)
        accumulator.set_velocity_buffer(velocity)
        accumulator.set_gbuffer(depth_curr, depth_prev, normal_curr, normal_prev)
        result = accumulator.accumulate(current_frame, history_frame)
    """

    def __init__(
        self,
        alpha: float = DEFAULT_TEMPORAL_ALPHA,
        history_frames: int = DEFAULT_HISTORY_FRAMES,
    ) -> None:
        """Initialize temporal accumulator.

        Args:
            alpha: EMA blend factor (0 = all history, 1 = all current).
            history_frames: Target frames for convergence.

        Raises:
            ValueError: If alpha is not in valid range.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not (MIN_HISTORY_FRAMES <= history_frames <= MAX_HISTORY_FRAMES):
            raise ValueError(
                f"history_frames must be in [{MIN_HISTORY_FRAMES}, {MAX_HISTORY_FRAMES}], "
                f"got {history_frames}"
            )

        self._alpha = alpha
        self._history_frames = history_frames

        # Buffers
        self._velocity_buffer: Optional[List[List[Vec2]]] = None
        self._depth_current: Optional[List[List[float]]] = None
        self._depth_previous: Optional[List[List[float]]] = None
        self._normal_current: Optional[List[List[Vec3]]] = None
        self._normal_previous: Optional[List[List[Vec3]]] = None

        # History length per pixel
        self._history_length: Optional[List[List[int]]] = None

        # Dimensions
        self._width = 0
        self._height = 0

        # Rejection thresholds
        self._depth_reject_threshold = DEPTH_REJECT_THRESHOLD
        self._normal_reject_threshold = NORMAL_REJECT_THRESHOLD

    @property
    def alpha(self) -> float:
        """Get EMA blend factor."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set EMA blend factor."""
        if not (0.0 < value <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {value}")
        self._alpha = value

    @property
    def history_frames(self) -> int:
        """Get target history frames."""
        return self._history_frames

    @history_frames.setter
    def history_frames(self, value: int) -> None:
        """Set target history frames."""
        if not (MIN_HISTORY_FRAMES <= value <= MAX_HISTORY_FRAMES):
            raise ValueError(
                f"history_frames must be in [{MIN_HISTORY_FRAMES}, {MAX_HISTORY_FRAMES}], "
                f"got {value}"
            )
        self._history_frames = value

    def set_velocity_buffer(self, velocity: List[List[Vec2]]) -> None:
        """Set velocity buffer for reprojection.

        Args:
            velocity: Screen-space motion vectors per pixel.
        """
        self._velocity_buffer = velocity
        self._height = len(velocity)
        self._width = len(velocity[0]) if velocity else 0

    def set_gbuffer(
        self,
        depth_current: List[List[float]],
        depth_previous: List[List[float]],
        normal_current: List[List[Vec3]],
        normal_previous: List[List[Vec3]],
    ) -> None:
        """Set G-buffer data for history validation.

        Args:
            depth_current: Current frame depth.
            depth_previous: Previous frame depth.
            normal_current: Current frame normals.
            normal_previous: Previous frame normals.
        """
        self._depth_current = depth_current
        self._depth_previous = depth_previous
        self._normal_current = normal_current
        self._normal_previous = normal_previous

    def initialize_history(self, width: int, height: int) -> None:
        """Initialize history length buffer.

        Args:
            width: Buffer width.
            height: Buffer height.
        """
        self._width = width
        self._height = height
        self._history_length = [[0 for _ in range(width)] for _ in range(height)]

    def reproject(self, x: int, y: int) -> ReprojectionResult:
        """Reproject pixel to previous frame using velocity buffer.

        Args:
            x: Current frame x coordinate.
            y: Current frame y coordinate.

        Returns:
            ReprojectionResult with previous frame UV and validity.
        """
        if self._velocity_buffer is None:
            return ReprojectionResult(
                prev_uv=Vec2(float(x), float(y)),
                valid=True,
                confidence=1.0,
            )

        # Get velocity at pixel
        velocity = self._velocity_buffer[y][x]

        # Calculate previous UV
        prev_x = float(x) - velocity.x
        prev_y = float(y) - velocity.y

        # Check bounds
        out_of_bounds = (
            prev_x < 0.0
            or prev_x >= self._width
            or prev_y < 0.0
            or prev_y >= self._height
        )

        if out_of_bounds:
            return ReprojectionResult(
                prev_uv=Vec2(prev_x, prev_y),
                valid=False,
                confidence=0.0,
                out_of_bounds=True,
            )

        return ReprojectionResult(
            prev_uv=Vec2(prev_x, prev_y),
            valid=True,
            confidence=1.0,
        )

    def validate_history(
        self,
        x: int,
        y: int,
        prev_x: int,
        prev_y: int,
    ) -> Tuple[bool, float]:
        """Validate history sample based on depth/normal discontinuity.

        Args:
            x: Current frame x coordinate.
            y: Current frame y coordinate.
            prev_x: Previous frame x coordinate.
            prev_y: Previous frame y coordinate.

        Returns:
            Tuple of (valid, confidence).
        """
        # Check bounds
        if not (0 <= prev_x < self._width and 0 <= prev_y < self._height):
            return False, 0.0

        confidence = 1.0

        # Depth validation
        if self._depth_current and self._depth_previous:
            depth_curr = self._depth_current[y][x]
            depth_prev = self._depth_previous[prev_y][prev_x]

            if abs(depth_curr) > DEPTH_EPSILON:
                relative_diff = abs(depth_curr - depth_prev) / abs(depth_curr)
                if relative_diff > self._depth_reject_threshold:
                    return False, 0.0
                confidence *= 1.0 - min(1.0, relative_diff / self._depth_reject_threshold)

        # Normal validation
        if self._normal_current and self._normal_previous:
            normal_curr = self._normal_current[y][x]
            normal_prev = self._normal_previous[prev_y][prev_x]

            dot = (
                normal_curr.x * normal_prev.x
                + normal_curr.y * normal_prev.y
                + normal_curr.z * normal_prev.z
            )

            if dot < self._normal_reject_threshold:
                return False, 0.0
            confidence *= dot

        return True, confidence

    def accumulate(
        self,
        current_frame: List[List[Vec3]],
        history_frame: List[List[Vec3]],
        output_frame: List[List[Vec3]],
    ) -> TemporalAccumulationResult:
        """Accumulate current frame with history using EMA.

        Args:
            current_frame: Current frame reflection colors.
            history_frame: Previous accumulated history.
            output_frame: Output buffer for accumulated result.

        Returns:
            TemporalAccumulationResult with statistics.
        """
        import time

        start_time = time.perf_counter()

        if self._history_length is None:
            self.initialize_history(self._width, self._height)

        converged_pixels = 0
        disoccluded_pixels = 0
        total_history_length = 0

        for y in range(self._height):
            for x in range(self._width):
                # Reproject
                reproj = self.reproject(x, y)

                if not reproj.valid:
                    # Disocclusion: use current frame only
                    output_frame[y][x] = Vec3(
                        current_frame[y][x].x,
                        current_frame[y][x].y,
                        current_frame[y][x].z,
                    )
                    self._history_length[y][x] = 1
                    disoccluded_pixels += 1
                    continue

                # Bilinear sample coordinates
                prev_x = int(reproj.prev_uv.x)
                prev_y = int(reproj.prev_uv.y)

                # Validate history
                valid, confidence = self.validate_history(x, y, prev_x, prev_y)

                if not valid:
                    # Disocclusion: use current frame only
                    output_frame[y][x] = Vec3(
                        current_frame[y][x].x,
                        current_frame[y][x].y,
                        current_frame[y][x].z,
                    )
                    self._history_length[y][x] = 1
                    disoccluded_pixels += 1
                    continue

                # Sample history (simple nearest-neighbor for now)
                history_sample = history_frame[prev_y][prev_x]
                current_sample = current_frame[y][x]

                # Adaptive alpha based on history length
                history_len = self._history_length[y][x]
                adaptive_alpha = max(self._alpha, 1.0 / (history_len + 1))
                adaptive_alpha *= confidence

                # EMA blend
                output_frame[y][x] = Vec3(
                    current_sample.x * adaptive_alpha
                    + history_sample.x * (1.0 - adaptive_alpha),
                    current_sample.y * adaptive_alpha
                    + history_sample.y * (1.0 - adaptive_alpha),
                    current_sample.z * adaptive_alpha
                    + history_sample.z * (1.0 - adaptive_alpha),
                )

                # Update history length
                self._history_length[y][x] = min(
                    history_len + 1, self._history_frames
                )
                total_history_length += self._history_length[y][x]

                if self._history_length[y][x] >= self._history_frames:
                    converged_pixels += 1

        total_pixels = self._width * self._height
        average_history = (
            total_history_length / total_pixels if total_pixels > 0 else 0.0
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return TemporalAccumulationResult(
            converged_pixels=converged_pixels,
            disoccluded_pixels=disoccluded_pixels,
            average_history_length=average_history,
            elapsed_ms=elapsed_ms,
        )

    def get_history_length(self, x: int, y: int) -> int:
        """Get history length at pixel.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            History length (number of accumulated frames).
        """
        if self._history_length is None:
            return 0
        return self._history_length[y][x]

    def reset_pixel_history(self, x: int, y: int) -> None:
        """Reset history length for a pixel.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.
        """
        if self._history_length is not None:
            self._history_length[y][x] = 0


# =============================================================================
# Bilateral Upscale for Reflections
# =============================================================================


@dataclass
class BilateralUpscaleResult:
    """Result of bilateral upscaling.

    Attributes:
        elapsed_ms: Time taken in milliseconds.
        pixels_upscaled: Number of pixels processed.
    """

    elapsed_ms: float = 0.0
    pixels_upscaled: int = 0


class ReflectionBilateralUpscale:
    """Bilateral upsampler for half/quarter resolution RT reflections.

    Uses depth and normal from full-resolution G-buffer to guide
    edge-preserving upscaling of low-resolution reflection data.

    Usage:
        upscaler = ReflectionBilateralUpscale(scale_factor=2)
        upscaler.set_gbuffer_full(depth_full, normal_full)
        upscaler.set_gbuffer_low(depth_low, normal_low)
        result = upscaler.upscale(low_res_reflections, full_res_output)
    """

    def __init__(
        self,
        scale_factor: int = 2,
        radius: int = BILATERAL_RADIUS,
        sigma_spatial: float = BILATERAL_SIGMA_SPATIAL,
        sigma_depth: float = DEFAULT_SIGMA_DEPTH,
        sigma_normal: float = DEFAULT_SIGMA_NORMAL,
    ) -> None:
        """Initialize bilateral upscaler.

        Args:
            scale_factor: Upscale factor (2 for half-res, 4 for quarter-res).
            radius: Bilateral filter radius in low-res pixels.
            sigma_spatial: Spatial Gaussian sigma.
            sigma_depth: Depth edge-stopping sensitivity.
            sigma_normal: Normal edge-stopping power.

        Raises:
            ValueError: If parameters are invalid.
        """
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1, got {scale_factor}")
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")

        self._scale_factor = scale_factor
        self._radius = radius
        self._sigma_spatial = sigma_spatial
        self._sigma_depth = sigma_depth
        self._sigma_normal = sigma_normal

        # Precompute spatial weights
        self._spatial_weights = self._compute_spatial_weights()

        # G-buffer references
        self._depth_full: Optional[List[List[float]]] = None
        self._normal_full: Optional[List[List[Vec3]]] = None
        self._depth_low: Optional[List[List[float]]] = None
        self._normal_low: Optional[List[List[Vec3]]] = None

        # Dimensions
        self._width_full = 0
        self._height_full = 0
        self._width_low = 0
        self._height_low = 0

    def _compute_spatial_weights(self) -> List[List[float]]:
        """Compute spatial Gaussian weights.

        Returns:
            2D array of spatial weights.
        """
        size = 2 * self._radius + 1
        weights = []
        sigma_sq = 2.0 * self._sigma_spatial * self._sigma_spatial

        for dy in range(-self._radius, self._radius + 1):
            row = []
            for dx in range(-self._radius, self._radius + 1):
                dist_sq = dx * dx + dy * dy
                w = math.exp(-dist_sq / sigma_sq) if sigma_sq > 0 else 1.0
                row.append(w)
            weights.append(row)

        return weights

    @property
    def scale_factor(self) -> int:
        """Get upscale factor."""
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value: int) -> None:
        """Set upscale factor."""
        if value < 1:
            raise ValueError(f"scale_factor must be >= 1, got {value}")
        self._scale_factor = value

    def set_gbuffer_full(
        self,
        depth: List[List[float]],
        normal: List[List[Vec3]],
    ) -> None:
        """Set full-resolution G-buffer data.

        Args:
            depth: Full-res depth buffer.
            normal: Full-res normal buffer.
        """
        self._depth_full = depth
        self._normal_full = normal
        self._height_full = len(depth)
        self._width_full = len(depth[0]) if depth else 0

    def set_gbuffer_low(
        self,
        depth: List[List[float]],
        normal: List[List[Vec3]],
    ) -> None:
        """Set low-resolution G-buffer data.

        Args:
            depth: Low-res depth buffer.
            normal: Low-res normal buffer.
        """
        self._depth_low = depth
        self._normal_low = normal
        self._height_low = len(depth)
        self._width_low = len(depth[0]) if depth else 0

    def bilateral_sample(
        self,
        x_full: int,
        y_full: int,
        low_res_buffer: List[List[Vec3]],
    ) -> Vec3:
        """Sample low-res buffer with bilateral filtering.

        Args:
            x_full: Full-res x coordinate.
            y_full: Full-res y coordinate.
            low_res_buffer: Low-resolution color buffer.

        Returns:
            Bilaterally filtered color.
        """
        # Map to low-res coordinates
        x_low = x_full // self._scale_factor
        y_low = y_full // self._scale_factor

        # Get full-res reference values
        depth_ref = (
            self._depth_full[y_full][x_full]
            if self._depth_full
            else 0.0
        )
        normal_ref = (
            self._normal_full[y_full][x_full]
            if self._normal_full
            else Vec3(0.0, 1.0, 0.0)
        )

        sum_color = Vec3(0.0, 0.0, 0.0)
        sum_weight = 0.0

        for dy in range(-self._radius, self._radius + 1):
            for dx in range(-self._radius, self._radius + 1):
                # Sample coordinates in low-res
                sx = x_low + dx
                sy = y_low + dy

                # Skip out-of-bounds
                if not (0 <= sx < self._width_low and 0 <= sy < self._height_low):
                    continue

                # Spatial weight
                spatial_w = self._spatial_weights[dy + self._radius][dx + self._radius]

                # Edge-stopping weights
                if self._depth_low:
                    depth_sample = self._depth_low[sy][sx]
                    depth_diff = abs(depth_sample - depth_ref)
                    if abs(depth_ref) > DEPTH_EPSILON:
                        depth_diff /= abs(depth_ref)
                    depth_w = math.exp(-depth_diff / self._sigma_depth)
                else:
                    depth_w = 1.0

                if self._normal_low:
                    normal_sample = self._normal_low[sy][sx]
                    dot = (
                        normal_ref.x * normal_sample.x
                        + normal_ref.y * normal_sample.y
                        + normal_ref.z * normal_sample.z
                    )
                    dot = max(0.0, min(1.0, dot))
                    normal_w = math.pow(dot, self._sigma_normal)
                else:
                    normal_w = 1.0

                # Combined weight
                weight = spatial_w * depth_w * normal_w

                # Get color
                color = low_res_buffer[sy][sx]

                # Accumulate
                sum_color = Vec3(
                    sum_color.x + color.x * weight,
                    sum_color.y + color.y * weight,
                    sum_color.z + color.z * weight,
                )
                sum_weight += weight

        # Normalize
        if sum_weight > EPSILON:
            return Vec3(
                sum_color.x / sum_weight,
                sum_color.y / sum_weight,
                sum_color.z / sum_weight,
            )
        else:
            # Fallback to nearest
            return low_res_buffer[y_low][x_low]

    def upscale(
        self,
        low_res_buffer: List[List[Vec3]],
        full_res_output: List[List[Vec3]],
    ) -> BilateralUpscaleResult:
        """Upscale low-resolution reflections to full resolution.

        Args:
            low_res_buffer: Low-resolution reflection colors.
            full_res_output: Full-resolution output buffer.

        Returns:
            BilateralUpscaleResult with statistics.
        """
        import time

        start_time = time.perf_counter()

        pixels_upscaled = 0

        for y in range(self._height_full):
            for x in range(self._width_full):
                full_res_output[y][x] = self.bilateral_sample(
                    x, y, low_res_buffer
                )
                pixels_upscaled += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return BilateralUpscaleResult(
            elapsed_ms=elapsed_ms,
            pixels_upscaled=pixels_upscaled,
        )


# =============================================================================
# RT Reflection Denoise Configuration
# =============================================================================


@dataclass
class RTReflectionDenoiseConfig:
    """Configuration for RT reflection denoising.

    Attributes:
        spatial_iterations: Number of A-Trous filter passes (1-5).
        temporal_alpha: EMA blend factor for temporal accumulation.
        sigma_depth: Depth edge-stopping sensitivity.
        sigma_normal: Normal edge-stopping power.
        sigma_luminance: Luminance edge-stopping sensitivity.
        history_frames: Target frames for convergence (8-16).
        enable_temporal: Enable temporal accumulation.
        enable_spatial: Enable spatial filtering.
        enable_upscale: Enable bilateral upscaling.
        input_scale: Input resolution scale (0.5 for half, 0.25 for quarter).
    """

    spatial_iterations: int = DEFAULT_ATROUS_ITERATIONS
    temporal_alpha: float = DEFAULT_TEMPORAL_ALPHA
    sigma_depth: float = DEFAULT_SIGMA_DEPTH
    sigma_normal: float = DEFAULT_SIGMA_NORMAL
    sigma_luminance: float = DEFAULT_SIGMA_LUMINANCE
    history_frames: int = DEFAULT_HISTORY_FRAMES
    enable_temporal: bool = True
    enable_spatial: bool = True
    enable_upscale: bool = True
    input_scale: float = 0.5  # Half resolution by default

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (1 <= self.spatial_iterations <= MAX_ATROUS_ITERATIONS):
            raise ValueError(
                f"spatial_iterations must be 1-{MAX_ATROUS_ITERATIONS}"
            )
        if not (0.0 < self.temporal_alpha <= 1.0):
            raise ValueError("temporal_alpha must be in (0, 1]")
        if not (MIN_HISTORY_FRAMES <= self.history_frames <= MAX_HISTORY_FRAMES):
            raise ValueError(
                f"history_frames must be in [{MIN_HISTORY_FRAMES}, {MAX_HISTORY_FRAMES}]"
            )
        if not (0.0 < self.input_scale <= 1.0):
            raise ValueError("input_scale must be in (0, 1]")

    @classmethod
    def from_quality(cls, quality: ReflectionDenoiseQuality) -> "RTReflectionDenoiseConfig":
        """Create configuration from quality preset.

        Args:
            quality: Quality preset.

        Returns:
            RTReflectionDenoiseConfig with preset parameters.
        """
        params = QUALITY_PRESETS[quality]
        return cls(
            spatial_iterations=params.spatial_iterations,
            temporal_alpha=params.temporal_alpha,
            sigma_depth=params.sigma_depth,
            sigma_normal=params.sigma_normal,
            sigma_luminance=params.sigma_luminance,
            history_frames=params.history_frames,
        )


# =============================================================================
# RT Reflection Denoise Pipeline
# =============================================================================


@dataclass
class DenoisePipelineResult:
    """Result of complete denoise pipeline.

    Attributes:
        temporal_result: Temporal accumulation statistics.
        spatial_result: A-Trous filter statistics.
        upscale_result: Bilateral upscale statistics.
        total_elapsed_ms: Total time in milliseconds.
        converged: Whether image has converged.
    """

    temporal_result: Optional[TemporalAccumulationResult] = None
    spatial_result: Optional[ATrousFilterResult] = None
    upscale_result: Optional[BilateralUpscaleResult] = None
    total_elapsed_ms: float = 0.0
    converged: bool = False


class RTReflectionDenoisePipeline:
    """Complete denoise pipeline for RT reflections.

    Implements the full pipeline: temporal -> spatial -> upscale

    Pipeline stages:
    1. Temporal Accumulation: Blend with reprojected history
    2. Spatial Filtering: A-Trous wavelet with edge-stopping
    3. Bilateral Upscale: Edge-aware upsampling to full resolution

    Usage:
        pipeline = RTReflectionDenoisePipeline(config)
        pipeline.set_buffers(...)
        pipeline.set_gbuffer(...)
        result = pipeline.denoise(noisy_reflections)
    """

    def __init__(self, config: Optional[RTReflectionDenoiseConfig] = None) -> None:
        """Initialize denoise pipeline.

        Args:
            config: Pipeline configuration (default: medium quality).
        """
        self._config = config or RTReflectionDenoiseConfig()

        # Create components
        self._edge_functions = ReflectionEdgeStopFunctions(
            sigma_depth=self._config.sigma_depth,
            sigma_normal=self._config.sigma_normal,
            sigma_luminance=self._config.sigma_luminance,
        )

        self._temporal = ReflectionTemporalAccumulator(
            alpha=self._config.temporal_alpha,
            history_frames=self._config.history_frames,
        )

        self._spatial = ReflectionATrousFilter(
            iterations=self._config.spatial_iterations,
        )
        self._spatial.set_edge_functions(self._edge_functions)

        scale_factor = int(1.0 / self._config.input_scale)
        self._upscale = ReflectionBilateralUpscale(
            scale_factor=max(1, scale_factor),
        )

        # Intermediate buffers
        self._temporal_output: Optional[List[List[Vec3]]] = None
        self._spatial_ping: Optional[List[List[Vec3]]] = None
        self._spatial_pong: Optional[List[List[Vec3]]] = None
        self._history_buffer: Optional[List[List[Vec3]]] = None

        # Dimensions
        self._width_low = 0
        self._height_low = 0
        self._width_full = 0
        self._height_full = 0

    @property
    def config(self) -> RTReflectionDenoiseConfig:
        """Get current configuration."""
        return self._config

    def set_config(self, config: RTReflectionDenoiseConfig) -> None:
        """Update pipeline configuration.

        Args:
            config: New configuration.
        """
        self._config = config

        self._edge_functions.sigma_depth = config.sigma_depth
        self._edge_functions.sigma_normal = config.sigma_normal
        self._edge_functions.sigma_luminance = config.sigma_luminance

        self._temporal.alpha = config.temporal_alpha
        self._temporal.history_frames = config.history_frames

        self._spatial.iterations = config.spatial_iterations

        scale_factor = int(1.0 / config.input_scale)
        self._upscale.scale_factor = max(1, scale_factor)

    def initialize_buffers(
        self,
        width_low: int,
        height_low: int,
        width_full: int,
        height_full: int,
    ) -> None:
        """Initialize intermediate buffers.

        Args:
            width_low: Low-resolution width.
            height_low: Low-resolution height.
            width_full: Full-resolution width.
            height_full: Full-resolution height.
        """
        self._width_low = width_low
        self._height_low = height_low
        self._width_full = width_full
        self._height_full = height_full

        # Allocate buffers
        self._temporal_output = [
            [Vec3(0.0, 0.0, 0.0) for _ in range(width_low)]
            for _ in range(height_low)
        ]
        self._spatial_ping = [
            [Vec3(0.0, 0.0, 0.0) for _ in range(width_low)]
            for _ in range(height_low)
        ]
        self._spatial_pong = [
            [Vec3(0.0, 0.0, 0.0) for _ in range(width_low)]
            for _ in range(height_low)
        ]
        self._history_buffer = [
            [Vec3(0.0, 0.0, 0.0) for _ in range(width_low)]
            for _ in range(height_low)
        ]

        # Initialize components
        self._temporal.initialize_history(width_low, height_low)
        self._spatial.set_buffers(self._spatial_ping, self._spatial_pong)

    def set_gbuffer(
        self,
        depth_low: List[List[float]],
        normal_low: List[List[Vec3]],
        depth_full: List[List[float]],
        normal_full: List[List[Vec3]],
        depth_prev: Optional[List[List[float]]] = None,
        normal_prev: Optional[List[List[Vec3]]] = None,
    ) -> None:
        """Set G-buffer data for all components.

        Args:
            depth_low: Low-res current frame depth.
            normal_low: Low-res current frame normals.
            depth_full: Full-res current frame depth.
            normal_full: Full-res current frame normals.
            depth_prev: Low-res previous frame depth (for temporal).
            normal_prev: Low-res previous frame normals (for temporal).
        """
        # Spatial filter
        self._spatial.set_gbuffer(depth_low, normal_low)

        # Temporal accumulator
        self._temporal.set_gbuffer(
            depth_low,
            depth_prev or depth_low,
            normal_low,
            normal_prev or normal_low,
        )

        # Bilateral upscale
        self._upscale.set_gbuffer_full(depth_full, normal_full)
        self._upscale.set_gbuffer_low(depth_low, normal_low)

    def set_velocity_buffer(self, velocity: List[List[Vec2]]) -> None:
        """Set velocity buffer for temporal reprojection.

        Args:
            velocity: Screen-space motion vectors.
        """
        self._temporal.set_velocity_buffer(velocity)

    def denoise(
        self,
        noisy_reflections: List[List[Vec3]],
        output_buffer: List[List[Vec3]],
    ) -> DenoisePipelineResult:
        """Execute complete denoise pipeline.

        Pipeline: temporal -> spatial -> upscale

        Args:
            noisy_reflections: Noisy RT reflection input (low-res).
            output_buffer: Denoised output (full-res).

        Returns:
            DenoisePipelineResult with statistics.
        """
        import time

        start_time = time.perf_counter()

        if self._temporal_output is None:
            raise RuntimeError(
                "Buffers not initialized, call initialize_buffers() first"
            )

        temporal_result: Optional[TemporalAccumulationResult] = None
        spatial_result: Optional[ATrousFilterResult] = None
        upscale_result: Optional[BilateralUpscaleResult] = None

        current_input = noisy_reflections

        # Stage 1: Temporal Accumulation
        if self._config.enable_temporal:
            temporal_result = self._temporal.accumulate(
                current_input,
                self._history_buffer,
                self._temporal_output,
            )

            # Update history for next frame
            for y in range(self._height_low):
                for x in range(self._width_low):
                    self._history_buffer[y][x] = Vec3(
                        self._temporal_output[y][x].x,
                        self._temporal_output[y][x].y,
                        self._temporal_output[y][x].z,
                    )

            current_input = self._temporal_output

        # Stage 2: Spatial Filtering
        if self._config.enable_spatial:
            spatial_result = self._spatial.filter_full(current_input)
            current_input = self._spatial.get_output_buffer(spatial_result.final_buffer)

        # Stage 3: Bilateral Upscale
        if self._config.enable_upscale and self._config.input_scale < 1.0:
            upscale_result = self._upscale.upscale(current_input, output_buffer)
        else:
            # No upscaling needed, copy directly
            for y in range(min(self._height_low, len(output_buffer))):
                for x in range(min(self._width_low, len(output_buffer[0]))):
                    output_buffer[y][x] = Vec3(
                        current_input[y][x].x,
                        current_input[y][x].y,
                        current_input[y][x].z,
                    )

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Check convergence
        converged = False
        if temporal_result:
            total_pixels = self._width_low * self._height_low
            converged = (
                temporal_result.converged_pixels > total_pixels * 0.9
            )

        return DenoisePipelineResult(
            temporal_result=temporal_result,
            spatial_result=spatial_result,
            upscale_result=upscale_result,
            total_elapsed_ms=total_elapsed_ms,
            converged=converged,
        )

    def get_quality_preset(self, quality: ReflectionDenoiseQuality) -> RTReflectionDenoiseConfig:
        """Get configuration for a quality preset.

        Args:
            quality: Quality preset.

        Returns:
            Configuration matching the preset.
        """
        return RTReflectionDenoiseConfig.from_quality(quality)


# =============================================================================
# WGSL Shader Generation
# =============================================================================


def generate_rt_reflections_denoise_wgsl() -> str:
    """Generate WGSL compute shader for RT reflection denoising.

    Returns:
        WGSL shader source code.
    """
    return '''// RT Reflection Denoise Compute Shader
// Generated by TRINITY T-GIR-P8.4

// Bindings
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var depth_texture: texture_2d<f32>;
@group(0) @binding(3) var normal_texture: texture_2d<f32>;
@group(0) @binding(4) var history_texture: texture_2d<f32>;
@group(0) @binding(5) var velocity_texture: texture_2d<f32>;
@group(0) @binding(6) var linear_sampler: sampler;

struct DenoiseParams {
    resolution: vec2<f32>,
    sigma_depth: f32,
    sigma_normal: f32,
    sigma_luminance: f32,
    temporal_alpha: f32,
    iteration: u32,
    dilation: u32,
}

@group(0) @binding(7) var<uniform> params: DenoiseParams;

// Constants
const EPSILON: f32 = 1e-6;
const KERNEL_SIZE: i32 = 5;
const KERNEL_HALF: i32 = 2;

// 5x5 Gaussian kernel weights
const KERNEL: array<f32, 25> = array<f32, 25>(
    0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
    0.015625,   0.0625,   0.09375,   0.0625,   0.015625,
    0.0234375,  0.09375,  0.140625,  0.09375,  0.0234375,
    0.015625,   0.0625,   0.09375,   0.0625,   0.015625,
    0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625
);

// Luminance extraction (YCoCg Y channel)
fn luminance(color: vec3<f32>) -> f32 {
    return 0.25 * color.r + 0.5 * color.g + 0.25 * color.b;
}

// Depth edge-stopping: exp(-|z1-z0| / sigma_z)
fn depth_edge_stop(depth_center: f32, depth_sample: f32, sigma: f32) -> f32 {
    let diff = abs(depth_sample - depth_center);
    let normalized = diff / max(abs(depth_center), EPSILON);
    return exp(-normalized / sigma);
}

// Normal edge-stopping: max(0, dot(n0, n1))^sigma_n
fn normal_edge_stop(normal_center: vec3<f32>, normal_sample: vec3<f32>, sigma: f32) -> f32 {
    let d = max(0.0, dot(normal_center, normal_sample));
    return pow(d, sigma);
}

// Luminance edge-stopping: exp(-|L1-L0| / sigma_l)
fn luminance_edge_stop(lum_center: f32, lum_sample: f32, sigma: f32) -> f32 {
    let diff = abs(lum_sample - lum_center);
    return exp(-diff / sigma);
}

// Combined edge-stopping weight
fn edge_stop_weight(
    depth_center: f32, depth_sample: f32,
    normal_center: vec3<f32>, normal_sample: vec3<f32>,
    lum_center: f32, lum_sample: f32
) -> f32 {
    let w_depth = depth_edge_stop(depth_center, depth_sample, params.sigma_depth);
    let w_normal = normal_edge_stop(normal_center, normal_sample, params.sigma_normal);
    let w_lum = luminance_edge_stop(lum_center, lum_sample, params.sigma_luminance);
    return w_depth * w_normal * w_lum;
}

// A-Trous wavelet filter pass
@compute @workgroup_size(8, 8, 1)
fn atrous_filter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = vec2<i32>(global_id.xy);
    let res = vec2<i32>(params.resolution);

    if (pixel.x >= res.x || pixel.y >= res.y) {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) / params.resolution;

    // Center pixel data
    let center_color = textureSampleLevel(input_texture, linear_sampler, uv, 0.0).rgb;
    let center_depth = textureSampleLevel(depth_texture, linear_sampler, uv, 0.0).r;
    let center_normal = textureSampleLevel(normal_texture, linear_sampler, uv, 0.0).xyz * 2.0 - 1.0;
    let center_lum = luminance(center_color);

    var sum_color = vec3<f32>(0.0);
    var sum_weight = 0.0;
    let dilation = i32(params.dilation);

    // 5x5 kernel with dilation
    for (var ky = -KERNEL_HALF; ky <= KERNEL_HALF; ky++) {
        for (var kx = -KERNEL_HALF; kx <= KERNEL_HALF; kx++) {
            let offset = vec2<i32>(kx, ky) * dilation;
            let sample_pixel = pixel + offset;

            // Bounds check
            if (sample_pixel.x < 0 || sample_pixel.x >= res.x ||
                sample_pixel.y < 0 || sample_pixel.y >= res.y) {
                continue;
            }

            let sample_uv = (vec2<f32>(sample_pixel) + 0.5) / params.resolution;

            // Sample data
            let sample_color = textureSampleLevel(input_texture, linear_sampler, sample_uv, 0.0).rgb;
            let sample_depth = textureSampleLevel(depth_texture, linear_sampler, sample_uv, 0.0).r;
            let sample_normal = textureSampleLevel(normal_texture, linear_sampler, sample_uv, 0.0).xyz * 2.0 - 1.0;
            let sample_lum = luminance(sample_color);

            // Kernel weight
            let kernel_idx = (ky + KERNEL_HALF) * KERNEL_SIZE + (kx + KERNEL_HALF);
            let kernel_w = KERNEL[kernel_idx];

            // Edge-stopping weight
            let edge_w = edge_stop_weight(
                center_depth, sample_depth,
                center_normal, sample_normal,
                center_lum, sample_lum
            );

            let weight = kernel_w * edge_w;
            sum_color += sample_color * weight;
            sum_weight += weight;
        }
    }

    // Normalize
    let result = select(center_color, sum_color / sum_weight, sum_weight > EPSILON);
    textureStore(output_texture, pixel, vec4<f32>(result, 1.0));
}

// Temporal accumulation pass
@compute @workgroup_size(8, 8, 1)
fn temporal_accumulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = vec2<i32>(global_id.xy);
    let res = vec2<i32>(params.resolution);

    if (pixel.x >= res.x || pixel.y >= res.y) {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) / params.resolution;

    // Current frame
    let current_color = textureSampleLevel(input_texture, linear_sampler, uv, 0.0).rgb;
    let current_depth = textureSampleLevel(depth_texture, linear_sampler, uv, 0.0).r;
    let current_normal = textureSampleLevel(normal_texture, linear_sampler, uv, 0.0).xyz * 2.0 - 1.0;

    // Velocity for reprojection
    let velocity = textureSampleLevel(velocity_texture, linear_sampler, uv, 0.0).xy;
    let prev_uv = uv - velocity;

    // Check bounds
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        textureStore(output_texture, pixel, vec4<f32>(current_color, 1.0));
        return;
    }

    // Sample history
    let history_color = textureSampleLevel(history_texture, linear_sampler, prev_uv, 0.0).rgb;

    // Validate history (depth/normal test)
    let prev_depth = textureSampleLevel(depth_texture, linear_sampler, prev_uv, 0.0).r;
    let prev_normal = textureSampleLevel(normal_texture, linear_sampler, prev_uv, 0.0).xyz * 2.0 - 1.0;

    let depth_valid = abs(current_depth - prev_depth) / max(abs(current_depth), EPSILON) < 0.1;
    let normal_valid = dot(current_normal, prev_normal) > 0.9;

    // Blend
    var alpha = params.temporal_alpha;
    if (!depth_valid || !normal_valid) {
        alpha = 1.0; // Disoccluded, use current only
    }

    let result = mix(history_color, current_color, alpha);
    textureStore(output_texture, pixel, vec4<f32>(result, 1.0));
}

// Bilateral upscale pass
@compute @workgroup_size(8, 8, 1)
fn bilateral_upscale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = vec2<i32>(global_id.xy);
    let res = vec2<i32>(params.resolution);

    if (pixel.x >= res.x || pixel.y >= res.y) {
        return;
    }

    let uv = (vec2<f32>(pixel) + 0.5) / params.resolution;

    // Full-res reference
    let ref_depth = textureSampleLevel(depth_texture, linear_sampler, uv, 0.0).r;
    let ref_normal = textureSampleLevel(normal_texture, linear_sampler, uv, 0.0).xyz * 2.0 - 1.0;

    var sum_color = vec3<f32>(0.0);
    var sum_weight = 0.0;

    // Bilateral filter (radius 2)
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy)) / params.resolution;
            let sample_uv = uv + offset;

            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                continue;
            }

            let sample_color = textureSampleLevel(input_texture, linear_sampler, sample_uv, 0.0).rgb;
            let sample_depth = textureSampleLevel(depth_texture, linear_sampler, sample_uv, 0.0).r;
            let sample_normal = textureSampleLevel(normal_texture, linear_sampler, sample_uv, 0.0).xyz * 2.0 - 1.0;

            // Spatial weight
            let dist_sq = f32(dx * dx + dy * dy);
            let spatial_w = exp(-dist_sq * 0.5);

            // Range weights
            let depth_w = depth_edge_stop(ref_depth, sample_depth, params.sigma_depth);
            let normal_w = normal_edge_stop(ref_normal, sample_normal, params.sigma_normal);

            let weight = spatial_w * depth_w * normal_w;
            sum_color += sample_color * weight;
            sum_weight += weight;
        }
    }

    let result = select(
        textureSampleLevel(input_texture, linear_sampler, uv, 0.0).rgb,
        sum_color / sum_weight,
        sum_weight > EPSILON
    );
    textureStore(output_texture, pixel, vec4<f32>(result, 1.0));
}
'''


# =============================================================================
# Factory Functions
# =============================================================================


def create_reflection_denoiser(
    quality: ReflectionDenoiseQuality = ReflectionDenoiseQuality.HIGH,
) -> RTReflectionDenoisePipeline:
    """Create RT reflection denoiser with quality preset.

    Args:
        quality: Quality preset.

    Returns:
        Configured RTReflectionDenoisePipeline.
    """
    config = RTReflectionDenoiseConfig.from_quality(quality)
    return RTReflectionDenoisePipeline(config)


def create_fast_reflection_denoiser() -> RTReflectionDenoisePipeline:
    """Create fast RT reflection denoiser (LOW quality).

    Returns:
        Fast RTReflectionDenoisePipeline.
    """
    return create_reflection_denoiser(ReflectionDenoiseQuality.LOW)


def create_quality_reflection_denoiser() -> RTReflectionDenoisePipeline:
    """Create high-quality RT reflection denoiser (ULTRA quality).

    Returns:
        High-quality RTReflectionDenoisePipeline.
    """
    return create_reflection_denoiser(ReflectionDenoiseQuality.ULTRA)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Constants
    "EPSILON",
    "LUMINANCE_EPSILON",
    "DEPTH_EPSILON",
    "DEFAULT_SIGMA_DEPTH",
    "DEFAULT_SIGMA_NORMAL",
    "DEFAULT_SIGMA_LUMINANCE",
    "DEFAULT_TEMPORAL_ALPHA",
    "DEFAULT_HISTORY_FRAMES",
    "DEFAULT_ATROUS_ITERATIONS",
    "MAX_ATROUS_ITERATIONS",
    "DEFAULT_DILATIONS",
    "GAUSSIAN_5X5_KERNEL",
    "BILATERAL_RADIUS",
    # Quality
    "ReflectionDenoiseQuality",
    "QualityPresetParams",
    "QUALITY_PRESETS",
    # Color Space
    "YCoCgConverter",
    # Edge-Stopping
    "EdgeStopWeights",
    "ReflectionEdgeStopFunctions",
    # A-Trous Filter
    "ATrousIterationResult",
    "ATrousFilterResult",
    "ReflectionATrousFilter",
    # Temporal Accumulation
    "ReprojectionResult",
    "TemporalAccumulationResult",
    "ReflectionTemporalAccumulator",
    # Bilateral Upscale
    "BilateralUpscaleResult",
    "ReflectionBilateralUpscale",
    # Configuration
    "RTReflectionDenoiseConfig",
    # Pipeline
    "DenoisePipelineResult",
    "RTReflectionDenoisePipeline",
    # WGSL Generation
    "generate_rt_reflections_denoise_wgsl",
    # Factory Functions
    "create_reflection_denoiser",
    "create_fast_reflection_denoiser",
    "create_quality_reflection_denoiser",
]
