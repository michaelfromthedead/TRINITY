"""
A-Trous Wavelet Spatial Denoiser

Implements edge-preserving spatial filtering using the A-Trous wavelet
algorithm with multiple edge-stopping functions for denoising ray-traced
signals (GI, reflections, shadows).

Key Features:
- 4-5 iteration A-Trous filter with increasing dilation (1, 2, 4, 8, 16)
- Edge-stopping functions: depth exponential, normal dot product, YCoCg luminance
- Ping-pong buffers for efficient multi-pass iteration
- PSNR measurement for quality evaluation
- Configurable per signal type (GI, reflections, shadows)

References:
- "Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering"
  Dammertz et al., HPG 2010
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.platform.rhi.device import Device
    from engine.platform.rhi.resources import Texture


# =============================================================================
# Constants
# =============================================================================


# Standard 5x5 Gaussian kernel for A-Trous wavelet
# Weights for the 5-tap filter: [1/16, 1/4, 3/8, 1/4, 1/16]
GAUSSIAN_5X5_KERNEL: Tuple[float, ...] = (
    1.0 / 256.0,  4.0 / 256.0,  6.0 / 256.0,  4.0 / 256.0, 1.0 / 256.0,
    4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
    6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
    4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
    1.0 / 256.0,  4.0 / 256.0,  6.0 / 256.0,  4.0 / 256.0, 1.0 / 256.0,
)

# Default dilation sequence: powers of 2
DEFAULT_DILATIONS: Tuple[int, ...] = (1, 2, 4, 8, 16)

# Numerical safety
EPSILON = 1e-6
LUMINANCE_EPSILON = 1e-4


# =============================================================================
# Quality Presets
# =============================================================================


class DenoiseQuality(IntEnum):
    """Denoiser quality presets controlling iteration count.

    Each level represents the number of A-Trous filter passes.
    Higher iteration counts provide smoother results at increased cost.
    """

    LOW = 2       # 2 iterations - dilation 1, 2
    MEDIUM = 3    # 3 iterations - dilation 1, 2, 4
    HIGH = 4      # 4 iterations - dilation 1, 2, 4, 8
    ULTRA = 5     # 5 iterations - dilation 1, 2, 4, 8, 16


class DenoiseTarget(Enum):
    """Target signal type for denoising.

    Different signal types require different edge-stopping sensitivities.
    """

    GI = auto()           # Global illumination (diffuse indirect)
    REFLECTIONS = auto()  # Specular reflections
    SHADOWS = auto()      # Shadow visibility
    AO = auto()           # Ambient occlusion
    CUSTOM = auto()       # User-defined parameters


# =============================================================================
# YCoCg Color Space Conversion
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

    # RGB to YCoCg transformation coefficients
    RGB_TO_Y_COEFF: Tuple[float, float, float] = (0.25, 0.5, 0.25)
    RGB_TO_CO_COEFF: Tuple[float, float, float] = (0.5, 0.0, -0.5)
    RGB_TO_CG_COEFF: Tuple[float, float, float] = (-0.25, 0.5, -0.25)

    # YCoCg to RGB transformation coefficients
    YCOCG_TO_R_COEFF: Tuple[float, float, float] = (1.0, 1.0, -1.0)
    YCOCG_TO_G_COEFF: Tuple[float, float, float] = (1.0, 0.0, 1.0)
    YCOCG_TO_B_COEFF: Tuple[float, float, float] = (1.0, -1.0, -1.0)

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
        """Extract luminance from RGB.

        Args:
            r: Red channel [0, 1].
            g: Green channel [0, 1].
            b: Blue channel [0, 1].

        Returns:
            Luminance value (Y channel of YCoCg).
        """
        return 0.25 * r + 0.5 * g + 0.25 * b

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
# Wavelet Kernel
# =============================================================================


@dataclass
class WaveletKernel:
    """5x5 A-Trous wavelet filter kernel.

    Provides weights for the separable or 2D wavelet filter.
    Supports different kernel configurations.

    Attributes:
        weights: 25 weights for 5x5 kernel (row-major).
        size: Kernel size (5 for 5x5).
    """

    weights: Tuple[float, ...]
    size: int = 5

    def __post_init__(self) -> None:
        """Validate kernel configuration."""
        expected_count = self.size * self.size
        if len(self.weights) != expected_count:
            raise ValueError(
                f"Expected {expected_count} weights for {self.size}x{self.size} kernel, "
                f"got {len(self.weights)}"
            )

        # Verify weights sum to approximately 1
        total = sum(self.weights)
        if not (0.99 < total < 1.01):
            raise ValueError(f"Kernel weights should sum to 1.0, got {total}")

    def get_weight(self, x: int, y: int) -> float:
        """Get kernel weight at position.

        Args:
            x: Horizontal offset from center [-2, 2].
            y: Vertical offset from center [-2, 2].

        Returns:
            Weight at the specified position.
        """
        half = self.size // 2
        if not (-half <= x <= half and -half <= y <= half):
            raise IndexError(f"Position ({x}, {y}) out of kernel bounds")

        ix = x + half
        iy = y + half
        return self.weights[iy * self.size + ix]

    def get_center_weight(self) -> float:
        """Get the center weight of the kernel.

        Returns:
            Center weight value.
        """
        center = (self.size * self.size) // 2
        return self.weights[center]

    @classmethod
    def create_gaussian(cls) -> "WaveletKernel":
        """Create standard Gaussian 5x5 kernel.

        Returns:
            WaveletKernel with Gaussian weights.
        """
        return cls(weights=GAUSSIAN_5X5_KERNEL)

    @classmethod
    def create_box(cls) -> "WaveletKernel":
        """Create box filter 5x5 kernel (uniform weights).

        Returns:
            WaveletKernel with uniform weights.
        """
        weight = 1.0 / 25.0
        return cls(weights=tuple([weight] * 25))


# =============================================================================
# Edge-Stopping Functions
# =============================================================================


@dataclass
class EdgeStopWeights:
    """Combined edge-stopping weights for a sample.

    Stores individual weights from each edge-stopping function
    and provides the combined weight.

    Attributes:
        depth: Weight from depth edge-stopping.
        normal: Weight from normal edge-stopping.
        luminance: Weight from luminance edge-stopping.
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
        """Check if weights are valid (non-negative).

        Returns:
            True if all weights are non-negative.
        """
        return (
            self.depth >= 0.0
            and self.normal >= 0.0
            and self.luminance >= 0.0
            and self.kernel >= 0.0
        )


class DepthEdgeStop:
    """Depth-based edge-stopping function using exponential falloff.

    Prevents blurring across depth discontinuities by reducing weights
    for samples with significantly different depth values.

    Formula: w = exp(-|z1 - z0| / (sigma * gradient))
    """

    def __init__(self, sigma: float = 1.0) -> None:
        """Initialize depth edge-stopping.

        Args:
            sigma: Depth sensitivity (higher = more tolerant of depth differences).

        Raises:
            ValueError: If sigma is not positive.
        """
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self._sigma = sigma

    @property
    def sigma(self) -> float:
        """Get depth sensitivity."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        """Set depth sensitivity."""
        if value <= 0.0:
            raise ValueError(f"sigma must be positive, got {value}")
        self._sigma = value

    def calculate_weight(
        self,
        depth_center: float,
        depth_sample: float,
        gradient: float = 1.0,
    ) -> float:
        """Calculate depth edge-stopping weight.

        Args:
            depth_center: Depth at center pixel.
            depth_sample: Depth at sample pixel.
            gradient: Local depth gradient for adaptive scaling.

        Returns:
            Edge-stopping weight [0, 1].
        """
        if gradient < EPSILON:
            gradient = EPSILON

        depth_diff = abs(depth_sample - depth_center)
        exponent = -depth_diff / (self._sigma * gradient)

        # Clamp exponent to avoid underflow
        exponent = max(exponent, -88.0)  # exp(-88) ~ 0

        return math.exp(exponent)

    def calculate_weight_linear(
        self,
        depth_center: float,
        depth_sample: float,
    ) -> float:
        """Calculate depth weight using linear depth (simplified).

        Args:
            depth_center: Linear depth at center.
            depth_sample: Linear depth at sample.

        Returns:
            Edge-stopping weight [0, 1].
        """
        depth_diff = abs(depth_sample - depth_center)
        normalized = depth_diff * self._sigma

        # Gaussian-like falloff
        return math.exp(-normalized * normalized)


class NormalEdgeStop:
    """Normal-based edge-stopping using dot product falloff.

    Prevents blurring across normal discontinuities by reducing weights
    for samples with different surface orientations.

    Formula: w = max(0, dot(n0, n1))^power
    """

    def __init__(self, power: float = 128.0) -> None:
        """Initialize normal edge-stopping.

        Args:
            power: Falloff power (higher = sharper falloff).

        Raises:
            ValueError: If power is not positive.
        """
        if power <= 0.0:
            raise ValueError(f"power must be positive, got {power}")
        self._power = power

    @property
    def power(self) -> float:
        """Get falloff power."""
        return self._power

    @power.setter
    def power(self, value: float) -> None:
        """Set falloff power."""
        if value <= 0.0:
            raise ValueError(f"power must be positive, got {value}")
        self._power = value

    def calculate_weight(
        self,
        normal_center: Tuple[float, float, float],
        normal_sample: Tuple[float, float, float],
    ) -> float:
        """Calculate normal edge-stopping weight.

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

        # Clamp to [0, 1] and apply power
        dot = max(0.0, min(1.0, dot))

        return math.pow(dot, self._power)

    def calculate_weight_threshold(
        self,
        normal_center: Tuple[float, float, float],
        normal_sample: Tuple[float, float, float],
        threshold: float = 0.9,
    ) -> float:
        """Calculate normal weight with hard threshold.

        Args:
            normal_center: Surface normal at center pixel.
            normal_sample: Surface normal at sample pixel.
            threshold: Minimum dot product for full weight.

        Returns:
            Edge-stopping weight [0, 1].
        """
        dot = (
            normal_center[0] * normal_sample[0]
            + normal_center[1] * normal_sample[1]
            + normal_center[2] * normal_sample[2]
        )

        if dot < threshold:
            return 0.0

        # Smooth transition above threshold
        t = (dot - threshold) / (1.0 - threshold + EPSILON)
        return t * t


class LuminanceEdgeStop:
    """Luminance-based edge-stopping using YCoCg color space.

    Prevents blurring across color/luminance edges by reducing weights
    for samples with significantly different luminance values.

    Formula: w = exp(-|L1 - L0| / (sigma + variance))
    """

    def __init__(self, sigma: float = 4.0) -> None:
        """Initialize luminance edge-stopping.

        Args:
            sigma: Luminance sensitivity (higher = more tolerant).

        Raises:
            ValueError: If sigma is not positive.
        """
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self._sigma = sigma
        self._converter = YCoCgConverter()

    @property
    def sigma(self) -> float:
        """Get luminance sensitivity."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float) -> None:
        """Set luminance sensitivity."""
        if value <= 0.0:
            raise ValueError(f"sigma must be positive, got {value}")
        self._sigma = value

    def calculate_weight(
        self,
        luminance_center: float,
        luminance_sample: float,
        variance: float = 0.0,
    ) -> float:
        """Calculate luminance edge-stopping weight.

        Args:
            luminance_center: Luminance at center pixel.
            luminance_sample: Luminance at sample pixel.
            variance: Local luminance variance for adaptive scaling.

        Returns:
            Edge-stopping weight [0, 1].
        """
        lum_diff = abs(luminance_sample - luminance_center)

        # Adaptive sigma based on local variance
        adaptive_sigma = self._sigma + math.sqrt(max(0.0, variance))

        exponent = -lum_diff / (adaptive_sigma + LUMINANCE_EPSILON)
        exponent = max(exponent, -88.0)

        return math.exp(exponent)

    def calculate_weight_rgb(
        self,
        color_center: Tuple[float, float, float],
        color_sample: Tuple[float, float, float],
        variance: float = 0.0,
    ) -> float:
        """Calculate luminance weight from RGB colors.

        Args:
            color_center: RGB color at center pixel.
            color_sample: RGB color at sample pixel.
            variance: Local luminance variance.

        Returns:
            Edge-stopping weight [0, 1].
        """
        lum_center = self._converter.luminance(*color_center)
        lum_sample = self._converter.luminance(*color_sample)

        return self.calculate_weight(lum_center, lum_sample, variance)

    def calculate_ycocg_weight(
        self,
        ycocg_center: Tuple[float, float, float],
        ycocg_sample: Tuple[float, float, float],
        variance: float = 0.0,
    ) -> float:
        """Calculate weight from YCoCg colors (all channels).

        Args:
            ycocg_center: YCoCg color at center.
            ycocg_sample: YCoCg color at sample.
            variance: Local variance.

        Returns:
            Edge-stopping weight [0, 1].
        """
        # Weight based on all YCoCg channels
        y_diff = abs(ycocg_sample[0] - ycocg_center[0])
        co_diff = abs(ycocg_sample[1] - ycocg_center[1])
        cg_diff = abs(ycocg_sample[2] - ycocg_center[2])

        # Weighted combination (luminance most important)
        total_diff = y_diff + 0.25 * co_diff + 0.25 * cg_diff

        adaptive_sigma = self._sigma + math.sqrt(max(0.0, variance))
        exponent = -total_diff / (adaptive_sigma + LUMINANCE_EPSILON)
        exponent = max(exponent, -88.0)

        return math.exp(exponent)


class EdgeStopFunctions:
    """Combined edge-stopping function manager.

    Provides a unified interface for all edge-stopping functions
    and calculates combined weights.
    """

    def __init__(
        self,
        depth_sigma: float = 1.0,
        normal_power: float = 128.0,
        luminance_sigma: float = 4.0,
    ) -> None:
        """Initialize edge-stopping functions.

        Args:
            depth_sigma: Depth edge-stopping sensitivity.
            normal_power: Normal edge-stopping power.
            luminance_sigma: Luminance edge-stopping sensitivity.
        """
        self._depth = DepthEdgeStop(depth_sigma)
        self._normal = NormalEdgeStop(normal_power)
        self._luminance = LuminanceEdgeStop(luminance_sigma)

    @property
    def depth(self) -> DepthEdgeStop:
        """Get depth edge-stopping function."""
        return self._depth

    @property
    def normal(self) -> NormalEdgeStop:
        """Get normal edge-stopping function."""
        return self._normal

    @property
    def luminance(self) -> LuminanceEdgeStop:
        """Get luminance edge-stopping function."""
        return self._luminance

    def calculate_weights(
        self,
        depth_center: float,
        depth_sample: float,
        normal_center: Tuple[float, float, float],
        normal_sample: Tuple[float, float, float],
        luminance_center: float,
        luminance_sample: float,
        kernel_weight: float,
        depth_gradient: float = 1.0,
        luminance_variance: float = 0.0,
    ) -> EdgeStopWeights:
        """Calculate all edge-stopping weights.

        Args:
            depth_center: Depth at center pixel.
            depth_sample: Depth at sample pixel.
            normal_center: Normal at center pixel.
            normal_sample: Normal at sample pixel.
            luminance_center: Luminance at center pixel.
            luminance_sample: Luminance at sample pixel.
            kernel_weight: Spatial kernel weight.
            depth_gradient: Local depth gradient.
            luminance_variance: Local luminance variance.

        Returns:
            EdgeStopWeights with individual and combined weights.
        """
        return EdgeStopWeights(
            depth=self._depth.calculate_weight(
                depth_center, depth_sample, depth_gradient
            ),
            normal=self._normal.calculate_weight(normal_center, normal_sample),
            luminance=self._luminance.calculate_weight(
                luminance_center, luminance_sample, luminance_variance
            ),
            kernel=kernel_weight,
        )

    def get_shader_params(self) -> Dict[str, float]:
        """Get parameters for shader binding.

        Returns:
            Dictionary of edge-stopping parameters.
        """
        return {
            "depth_sigma": self._depth.sigma,
            "normal_power": self._normal.power,
            "luminance_sigma": self._luminance.sigma,
        }


# =============================================================================
# Denoiser Configuration
# =============================================================================


@dataclass
class DenoiseConfig:
    """Full configuration for A-Trous wavelet denoising.

    Attributes:
        quality: Quality preset controlling iteration count.
        target: Signal type being denoised.
        depth_sigma: Depth edge-stopping sensitivity.
        normal_power: Normal edge-stopping power.
        luminance_sigma: Luminance edge-stopping sensitivity.
        iterations: Override iteration count (None = use quality preset).
        dilations: Custom dilation sequence (None = use defaults).
        use_ycocg: Use YCoCg luminance for edge detection.
        use_variance: Use local variance for adaptive filtering.
        preserve_details: Detail preservation strength [0, 1].
    """

    quality: DenoiseQuality = DenoiseQuality.HIGH
    target: DenoiseTarget = DenoiseTarget.GI

    # Edge-stopping parameters
    depth_sigma: float = 1.0
    normal_power: float = 128.0
    luminance_sigma: float = 4.0

    # Override settings
    iterations: Optional[int] = None
    dilations: Optional[Tuple[int, ...]] = None

    # Advanced options
    use_ycocg: bool = True
    use_variance: bool = True
    preserve_details: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.quality, DenoiseQuality):
            raise TypeError(
                f"quality must be DenoiseQuality, got {type(self.quality).__name__}"
            )
        if not isinstance(self.target, DenoiseTarget):
            raise TypeError(
                f"target must be DenoiseTarget, got {type(self.target).__name__}"
            )
        if self.depth_sigma <= 0.0:
            raise ValueError(f"depth_sigma must be positive, got {self.depth_sigma}")
        if self.normal_power <= 0.0:
            raise ValueError(f"normal_power must be positive, got {self.normal_power}")
        if self.luminance_sigma <= 0.0:
            raise ValueError(
                f"luminance_sigma must be positive, got {self.luminance_sigma}"
            )
        if self.iterations is not None and self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if not (0.0 <= self.preserve_details <= 1.0):
            raise ValueError(
                f"preserve_details must be in [0, 1], got {self.preserve_details}"
            )

        # Validate custom dilations
        if self.dilations is not None:
            if len(self.dilations) < 1:
                raise ValueError("dilations must have at least one element")
            for d in self.dilations:
                if d < 1:
                    raise ValueError(f"dilation values must be >= 1, got {d}")

    def get_iteration_count(self) -> int:
        """Get the number of filter iterations.

        Returns:
            Number of A-Trous passes.
        """
        if self.iterations is not None:
            return self.iterations
        return int(self.quality)

    def get_dilations(self) -> Tuple[int, ...]:
        """Get the dilation sequence for iterations.

        Returns:
            Tuple of dilation values (one per iteration).
        """
        if self.dilations is not None:
            return self.dilations

        iteration_count = self.get_iteration_count()
        return DEFAULT_DILATIONS[:iteration_count]

    def create_edge_functions(self) -> EdgeStopFunctions:
        """Create edge-stopping functions from config.

        Returns:
            Configured EdgeStopFunctions instance.
        """
        return EdgeStopFunctions(
            depth_sigma=self.depth_sigma,
            normal_power=self.normal_power,
            luminance_sigma=self.luminance_sigma,
        )


# =============================================================================
# PSNR Quality Metrics
# =============================================================================


@dataclass
class PSNRMetrics:
    """Peak Signal-to-Noise Ratio metrics for denoiser evaluation.

    Attributes:
        psnr: PSNR in decibels (higher is better).
        mse: Mean Squared Error.
        max_value: Maximum signal value used for PSNR calculation.
        improvement: PSNR improvement vs unfiltered (dB).
    """

    psnr: float
    mse: float
    max_value: float = 1.0
    improvement: float = 0.0

    @staticmethod
    def calculate_mse(
        reference: List[float],
        filtered: List[float],
    ) -> float:
        """Calculate Mean Squared Error.

        Args:
            reference: Reference signal values.
            filtered: Filtered signal values.

        Returns:
            MSE value.

        Raises:
            ValueError: If signal lengths don't match.
        """
        if len(reference) != len(filtered):
            raise ValueError(
                f"Signal lengths must match: {len(reference)} vs {len(filtered)}"
            )
        if len(reference) == 0:
            raise ValueError("Cannot calculate MSE of empty signals")

        sum_squared_diff = sum(
            (r - f) ** 2 for r, f in zip(reference, filtered)
        )
        return sum_squared_diff / len(reference)

    @classmethod
    def calculate(
        cls,
        reference: List[float],
        filtered: List[float],
        max_value: float = 1.0,
    ) -> "PSNRMetrics":
        """Calculate PSNR metrics.

        Args:
            reference: Reference (ground truth) signal.
            filtered: Filtered signal.
            max_value: Maximum signal value.

        Returns:
            PSNRMetrics instance.
        """
        mse = cls.calculate_mse(reference, filtered)

        if mse < EPSILON:
            psnr = float("inf")
        else:
            psnr = 10.0 * math.log10((max_value ** 2) / mse)

        return cls(psnr=psnr, mse=mse, max_value=max_value)

    @classmethod
    def calculate_improvement(
        cls,
        reference: List[float],
        noisy: List[float],
        filtered: List[float],
        max_value: float = 1.0,
    ) -> "PSNRMetrics":
        """Calculate PSNR with improvement measurement.

        Args:
            reference: Reference (ground truth) signal.
            noisy: Noisy input signal.
            filtered: Filtered signal.
            max_value: Maximum signal value.

        Returns:
            PSNRMetrics with improvement value.
        """
        noisy_metrics = cls.calculate(reference, noisy, max_value)
        filtered_metrics = cls.calculate(reference, filtered, max_value)

        improvement = filtered_metrics.psnr - noisy_metrics.psnr

        return cls(
            psnr=filtered_metrics.psnr,
            mse=filtered_metrics.mse,
            max_value=max_value,
            improvement=improvement,
        )

    def is_improved(self) -> bool:
        """Check if filtering improved quality.

        Returns:
            True if PSNR improved.
        """
        return self.improvement > 0.0


@dataclass
class DenoiseStats:
    """Statistics from a denoising operation.

    Attributes:
        iterations: Number of iterations performed.
        total_time_ms: Total processing time.
        per_iteration_ms: Average time per iteration.
        pixels_processed: Total pixels processed.
        psnr: Optional PSNR metrics.
    """

    iterations: int
    total_time_ms: float = 0.0
    per_iteration_ms: float = 0.0
    pixels_processed: int = 0
    psnr: Optional[PSNRMetrics] = None


# =============================================================================
# Ping-Pong Buffers
# =============================================================================


@dataclass
class PingPongBuffers:
    """Ping-pong buffer pair for iterative filtering.

    Attributes:
        ping: First buffer.
        pong: Second buffer.
        width: Buffer width.
        height: Buffer height.
    """

    ping: "Texture"
    pong: "Texture"
    width: int
    height: int

    def is_valid(self) -> bool:
        """Check if buffers are valid.

        Returns:
            True if both buffers are valid.
        """
        return (
            self.ping is not None
            and self.pong is not None
            and self.ping.is_valid()
            and self.pong.is_valid()
        )

    def matches_dimensions(self, width: int, height: int) -> bool:
        """Check if buffers match specified dimensions.

        Args:
            width: Expected width.
            height: Expected height.

        Returns:
            True if dimensions match.
        """
        return self.width == width and self.height == height


@dataclass
class DenoiseGBuffer:
    """G-Buffer data required for edge-aware denoising.

    Attributes:
        depth: Linear depth buffer.
        normal: World-space normal buffer.
        albedo: Optional surface albedo.
        velocity: Optional motion vector buffer.
    """

    depth: "Texture"
    normal: "Texture"
    albedo: Optional["Texture"] = None
    velocity: Optional["Texture"] = None

    def is_valid(self) -> bool:
        """Check if required G-Buffer textures are valid.

        Returns:
            True if depth and normal are valid.
        """
        return (
            self.depth is not None
            and self.normal is not None
            and self.depth.is_valid()
            and self.normal.is_valid()
        )

    def has_albedo(self) -> bool:
        """Check if albedo texture is available.

        Returns:
            True if albedo is present and valid.
        """
        return self.albedo is not None and self.albedo.is_valid()

    def has_velocity(self) -> bool:
        """Check if velocity texture is available.

        Returns:
            True if velocity is present and valid.
        """
        return self.velocity is not None and self.velocity.is_valid()


# =============================================================================
# A-Trous Pass
# =============================================================================


@dataclass
class ATrousPass:
    """Single A-Trous filter pass configuration.

    Attributes:
        iteration: Pass iteration index (0-based).
        dilation: Step size (2^iteration typically).
        source: Source texture for this pass.
        destination: Destination texture for this pass.
    """

    iteration: int
    dilation: int
    source: "Texture"
    destination: "Texture"

    def get_step_size(self) -> int:
        """Get the step size for kernel sampling.

        Returns:
            Step size (same as dilation).
        """
        return self.dilation

    def get_kernel_offsets(self) -> List[Tuple[int, int]]:
        """Get 5x5 kernel sample offsets for this dilation.

        Returns:
            List of (x, y) offsets relative to center.
        """
        offsets = []
        step = self.dilation

        for y in range(-2, 3):
            for x in range(-2, 3):
                offsets.append((x * step, y * step))

        return offsets


# =============================================================================
# A-Trous Denoiser
# =============================================================================


class ATrousDenoiser:
    """A-Trous wavelet spatial denoiser for ray-traced signals.

    Implements edge-preserving spatial filtering using the A-Trous wavelet
    algorithm with configurable edge-stopping functions.

    Features:
    - 4-5 iteration filter with increasing dilation (1, 2, 4, 8, 16)
    - Edge-stopping: depth exponential, normal dot product, YCoCg luminance
    - Ping-pong buffers for efficient iteration
    - PSNR measurement for quality evaluation
    - Works with GI, reflections, shadows, and AO

    Example:
        config = DenoiseConfig(quality=DenoiseQuality.HIGH, target=DenoiseTarget.GI)
        denoiser = ATrousDenoiser(device, config)
        denoiser.denoise(noisy_gi, g_buffer, output)
    """

    def __init__(
        self,
        device: "Device",
        config: Optional[DenoiseConfig] = None,
    ) -> None:
        """Initialize the A-Trous denoiser.

        Args:
            device: RHI device for resource creation.
            config: Denoiser configuration (uses defaults if None).
        """
        self._device = device
        self._config = config or DenoiseConfig()
        self._edge_functions = self._config.create_edge_functions()
        self._kernel = WaveletKernel.create_gaussian()

        self._ping_pong: Optional[PingPongBuffers] = None
        self._initialized = False

    @property
    def device(self) -> "Device":
        """Get the RHI device."""
        return self._device

    @property
    def config(self) -> DenoiseConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, value: DenoiseConfig) -> None:
        """Set configuration and update edge functions."""
        self._config = value
        self._edge_functions = value.create_edge_functions()

    @property
    def edge_functions(self) -> EdgeStopFunctions:
        """Get edge-stopping functions."""
        return self._edge_functions

    @property
    def kernel(self) -> WaveletKernel:
        """Get wavelet kernel."""
        return self._kernel

    @property
    def is_initialized(self) -> bool:
        """Check if denoiser is initialized with buffers."""
        return self._initialized

    def get_iteration_count(self) -> int:
        """Get number of filter iterations from config.

        Returns:
            Number of A-Trous passes.
        """
        return self._config.get_iteration_count()

    def get_dilations(self) -> Tuple[int, ...]:
        """Get dilation sequence from config.

        Returns:
            Tuple of dilation values.
        """
        return self._config.get_dilations()

    def create_ping_pong_buffers(
        self,
        width: int,
        height: int,
    ) -> PingPongBuffers:
        """Create or reuse ping-pong buffers.

        Args:
            width: Buffer width.
            height: Buffer height.

        Returns:
            PingPongBuffers instance.

        Raises:
            ValueError: If width or height is not positive.
        """
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")

        # Check if existing buffers can be reused
        if (
            self._ping_pong is not None
            and self._ping_pong.matches_dimensions(width, height)
        ):
            return self._ping_pong

        # Import here to avoid circular imports
        from engine.platform.rhi.resources import (
            Format,
            TextureDesc,
            TextureType,
            TextureUsage,
        )

        desc = TextureDesc(
            type=TextureType.TEXTURE_2D,
            format=Format.RGBA16_FLOAT,
            width=width,
            height=height,
            usage=TextureUsage.SHADER_RESOURCE | TextureUsage.UNORDERED_ACCESS,
        )

        ping = self._device.create_texture(desc)
        pong = self._device.create_texture(desc)

        self._ping_pong = PingPongBuffers(
            ping=ping,
            pong=pong,
            width=width,
            height=height,
        )
        self._initialized = True

        return self._ping_pong

    def denoise(
        self,
        noisy_input: "Texture",
        g_buffer: DenoiseGBuffer,
        output: "Texture",
        config: Optional[DenoiseConfig] = None,
    ) -> DenoiseStats:
        """Perform spatial denoising.

        Executes multi-pass A-Trous wavelet filtering with edge-stopping.

        Args:
            noisy_input: Input texture with noisy signal.
            g_buffer: G-Buffer for edge-aware filtering.
            output: Output texture for denoised result.
            config: Override configuration (uses instance config if None).

        Returns:
            DenoiseStats with operation statistics.

        Raises:
            ValueError: If inputs are invalid or dimensions mismatch.
        """
        if config is not None:
            self._config = config
            self._edge_functions = config.create_edge_functions()

        # Validate inputs
        self._validate_inputs(noisy_input, g_buffer, output)

        # Get dimensions
        input_desc = noisy_input.desc
        width = input_desc.width
        height = input_desc.height

        # Create ping-pong buffers
        ping_pong = self.create_ping_pong_buffers(width, height)

        # Build pass list
        passes = self._build_passes(noisy_input, output, ping_pong)

        # Dispatch passes
        stats = self._dispatch_passes(passes, g_buffer, width, height)

        return stats

    def _validate_inputs(
        self,
        noisy_input: "Texture",
        g_buffer: DenoiseGBuffer,
        output: "Texture",
    ) -> None:
        """Validate input textures.

        Raises:
            ValueError: If any input is invalid.
        """
        if noisy_input is None or not noisy_input.is_valid():
            raise ValueError("noisy_input texture is invalid")
        if output is None or not output.is_valid():
            raise ValueError("output texture is invalid")
        if not g_buffer.is_valid():
            raise ValueError("g_buffer is invalid (missing depth or normal)")

        # Check dimension match
        input_desc = noisy_input.desc
        output_desc = output.desc

        if input_desc.width != output_desc.width:
            raise ValueError(
                f"Output width ({output_desc.width}) does not match "
                f"input width ({input_desc.width})"
            )
        if input_desc.height != output_desc.height:
            raise ValueError(
                f"Output height ({output_desc.height}) does not match "
                f"input height ({input_desc.height})"
            )

    def _build_passes(
        self,
        noisy_input: "Texture",
        output: "Texture",
        ping_pong: PingPongBuffers,
    ) -> List[ATrousPass]:
        """Build list of filter passes.

        Args:
            noisy_input: Input texture.
            output: Output texture.
            ping_pong: Ping-pong buffers.

        Returns:
            List of ATrousPass configurations.
        """
        dilations = self.get_dilations()
        passes = []

        current_src = noisy_input
        use_ping = True

        for i, dilation in enumerate(dilations):
            is_last = i == len(dilations) - 1

            # Determine destination
            if is_last:
                current_dst = output
            else:
                current_dst = ping_pong.ping if use_ping else ping_pong.pong

            passes.append(
                ATrousPass(
                    iteration=i,
                    dilation=dilation,
                    source=current_src,
                    destination=current_dst,
                )
            )

            # Swap for next iteration
            current_src = current_dst
            use_ping = not use_ping

        return passes

    def _dispatch_passes(
        self,
        passes: List[ATrousPass],
        g_buffer: DenoiseGBuffer,
        width: int,
        height: int,
    ) -> DenoiseStats:
        """Dispatch filter passes.

        Args:
            passes: List of passes to execute.
            g_buffer: G-Buffer for edge-stopping.
            width: Texture width.
            height: Texture height.

        Returns:
            DenoiseStats from the operation.
        """
        # In real implementation, this would:
        # 1. Bind compute pipeline for A-Trous filter
        # 2. For each pass:
        #    a. Bind source and destination textures
        #    b. Bind G-Buffer textures
        #    c. Set shader uniforms (dilation, edge-stop params)
        #    d. Dispatch compute shader

        for pass_config in passes:
            self._dispatch_single_pass(pass_config, g_buffer, width, height)

        return DenoiseStats(
            iterations=len(passes),
            pixels_processed=width * height * len(passes),
        )

    def _dispatch_single_pass(
        self,
        pass_config: ATrousPass,
        g_buffer: DenoiseGBuffer,
        width: int,
        height: int,
    ) -> None:
        """Dispatch a single A-Trous filter pass.

        Args:
            pass_config: Pass configuration.
            g_buffer: G-Buffer textures.
            width: Texture width.
            height: Texture height.
        """
        # Stub: In real implementation, dispatch compute shader
        _ = (
            pass_config.source,
            pass_config.destination,
            pass_config.dilation,
            g_buffer.depth,
            g_buffer.normal,
            self._edge_functions.get_shader_params(),
            self._kernel.weights,
            width,
            height,
        )

    def destroy(self) -> None:
        """Release denoiser resources."""
        if self._ping_pong is not None:
            if self._ping_pong.ping is not None:
                self._ping_pong.ping.destroy()
            if self._ping_pong.pong is not None:
                self._ping_pong.pong.destroy()
            self._ping_pong = None
        self._initialized = False

    def __del__(self) -> None:
        """Clean up on deletion."""
        self.destroy()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_default_config() -> DenoiseConfig:
    """Create default denoiser configuration.

    Returns:
        DenoiseConfig with balanced defaults.
    """
    return DenoiseConfig()


def create_quality_config(quality: DenoiseQuality) -> DenoiseConfig:
    """Create configuration for a specific quality level.

    Args:
        quality: Desired quality level.

    Returns:
        DenoiseConfig tuned for the quality level.
    """
    # Adjust parameters based on quality
    sigma_scale = {
        DenoiseQuality.LOW: 1.2,
        DenoiseQuality.MEDIUM: 1.0,
        DenoiseQuality.HIGH: 0.9,
        DenoiseQuality.ULTRA: 0.8,
    }

    scale = sigma_scale.get(quality, 1.0)

    return DenoiseConfig(
        quality=quality,
        depth_sigma=scale,
        luminance_sigma=4.0 * scale,
    )


def create_gi_denoiser(device: "Device") -> ATrousDenoiser:
    """Create denoiser optimized for global illumination.

    GI typically has low-frequency noise and benefits from
    aggressive spatial filtering with strong edge preservation.

    Args:
        device: RHI device.

    Returns:
        Configured ATrousDenoiser.
    """
    config = DenoiseConfig(
        quality=DenoiseQuality.HIGH,
        target=DenoiseTarget.GI,
        depth_sigma=1.0,
        normal_power=64.0,  # Lower power for softer GI transitions
        luminance_sigma=4.0,
        use_variance=True,
    )
    return ATrousDenoiser(device, config)


def create_reflection_denoiser(device: "Device") -> ATrousDenoiser:
    """Create denoiser optimized for specular reflections.

    Reflections require tighter edge preservation to maintain
    sharp reflection boundaries.

    Args:
        device: RHI device.

    Returns:
        Configured ATrousDenoiser.
    """
    config = DenoiseConfig(
        quality=DenoiseQuality.HIGH,
        target=DenoiseTarget.REFLECTIONS,
        depth_sigma=0.5,  # Tighter depth sensitivity
        normal_power=256.0,  # Higher power for sharper reflection edges
        luminance_sigma=2.0,  # Tighter luminance sensitivity
        preserve_details=0.7,
    )
    return ATrousDenoiser(device, config)


def create_shadow_denoiser(device: "Device") -> ATrousDenoiser:
    """Create denoiser optimized for shadow visibility.

    Shadows benefit from strong spatial filtering while
    preserving shadow boundaries.

    Args:
        device: RHI device.

    Returns:
        Configured ATrousDenoiser.
    """
    config = DenoiseConfig(
        quality=DenoiseQuality.MEDIUM,  # Shadows often need fewer iterations
        target=DenoiseTarget.SHADOWS,
        depth_sigma=2.0,  # More tolerant of depth variation
        normal_power=128.0,
        luminance_sigma=8.0,  # More tolerant of luminance variation
        use_ycocg=False,  # Shadows are typically single-channel
    )
    return ATrousDenoiser(device, config)


__all__ = [
    # Core Denoiser
    "ATrousDenoiser",
    "ATrousPass",
    # Configuration
    "DenoiseConfig",
    "DenoiseQuality",
    "DenoiseTarget",
    # Edge-Stopping Functions
    "EdgeStopFunctions",
    "EdgeStopWeights",
    "DepthEdgeStop",
    "NormalEdgeStop",
    "LuminanceEdgeStop",
    # Color Space Conversion
    "YCoCgConverter",
    # Filter Kernel
    "WaveletKernel",
    "GAUSSIAN_5X5_KERNEL",
    # Buffers
    "PingPongBuffers",
    "DenoiseGBuffer",
    # Quality Metrics
    "PSNRMetrics",
    "DenoiseStats",
    # Convenience Functions
    "create_gi_denoiser",
    "create_reflection_denoiser",
    "create_shadow_denoiser",
    "create_default_config",
    "create_quality_config",
]
