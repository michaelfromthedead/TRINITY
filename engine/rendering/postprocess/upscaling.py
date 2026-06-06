"""
Super Resolution Upscaling System

Provides upscaling integration:
- SpatialUpscaler: Base class for spatial upscalers (FSR 1.0, CAS)
- TemporalUpscaler: Base class for temporal upscalers (DLSS, FSR 2/3, XeSS)
- UpscalingSettings: Quality mode and sharpening configuration
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .postprocess_stack import EffectPriority, EffectSettings, PostProcessEffect


class UpscalerType(Enum):
    """Type of upscaler."""

    NONE = auto()
    # Spatial upscalers
    BILINEAR = auto()
    BICUBIC = auto()
    LANCZOS = auto()
    FSR_1 = auto()  # AMD FSR 1.0
    CAS = auto()  # AMD Contrast Adaptive Sharpening
    # Temporal upscalers
    FSR_2 = auto()  # AMD FSR 2.x
    FSR_3 = auto()  # AMD FSR 3.x with frame gen
    DLSS = auto()  # NVIDIA DLSS
    DLSS_FG = auto()  # NVIDIA DLSS with frame gen
    XESS = auto()  # Intel XeSS
    TSR_LANCZOS = auto()  # Native Lanczos-based TSR (T-PP-6.3)


class UpscaleQuality(Enum):
    """Upscaling quality preset."""

    ULTRA_PERFORMANCE = auto()  # ~3x scale
    PERFORMANCE = auto()  # ~2x scale
    BALANCED = auto()  # ~1.7x scale
    QUALITY = auto()  # ~1.5x scale
    ULTRA_QUALITY = auto()  # ~1.3x scale
    NATIVE_AA = auto()  # 1x scale (AA only)


class FrameGenerationMode(Enum):
    """Frame generation mode."""

    OFF = auto()
    ON = auto()
    AUTO = auto()  # Based on framerate


@dataclass
class UpscaleResolution:
    """Resolution information for upscaling."""

    render_width: int
    render_height: int
    output_width: int
    output_height: int

    @property
    def scale_factor(self) -> float:
        """Get the upscale factor."""
        if self.render_width == 0:
            return 1.0
        return self.output_width / self.render_width

    @property
    def render_percentage(self) -> float:
        """Get render resolution as percentage of output."""
        return 100.0 / self.scale_factor if self.scale_factor > 0 else 100.0


def get_render_resolution(
    output_width: int,
    output_height: int,
    quality: UpscaleQuality,
) -> Tuple[int, int]:
    """Calculate render resolution for a quality preset.

    Args:
        output_width: Target output width.
        output_height: Target output height.
        quality: Quality preset.

    Returns:
        (render_width, render_height).
    """
    from .constants import UPSCALING

    scale_factors = {
        UpscaleQuality.ULTRA_PERFORMANCE: UPSCALING.SCALE_ULTRA_PERFORMANCE,
        UpscaleQuality.PERFORMANCE: UPSCALING.SCALE_PERFORMANCE,
        UpscaleQuality.BALANCED: UPSCALING.SCALE_BALANCED,
        UpscaleQuality.QUALITY: UPSCALING.SCALE_QUALITY,
        UpscaleQuality.ULTRA_QUALITY: UPSCALING.SCALE_ULTRA_QUALITY,
        UpscaleQuality.NATIVE_AA: UPSCALING.SCALE_NATIVE,
    }

    scale = scale_factors.get(quality, UPSCALING.SCALE_NATIVE)
    render_width = max(1, int(output_width * scale))
    render_height = max(1, int(output_height * scale))

    return (render_width, render_height)


@dataclass
class UpscalingSettings(EffectSettings):
    """Upscaling settings."""

    upscaler_type: UpscalerType = UpscalerType.FSR_2
    quality: UpscaleQuality = UpscaleQuality.QUALITY

    # Sharpening
    sharpening_enabled: bool = True
    sharpening_amount: float = 0.5  # [0, 1]

    # Frame generation
    frame_generation: FrameGenerationMode = FrameGenerationMode.OFF

    # Motion vector settings
    motion_vector_scale: float = 1.0
    motion_vector_jitter_cancel: bool = True

    # Advanced settings
    auto_exposure_enabled: bool = True
    mip_bias_offset: float = 0.0  # Additional mip bias
    reset_accumulation: bool = False  # Force reset temporal data

    def __post_init__(self) -> None:
        self.priority = EffectPriority.UPSCALING.value

    def lerp(self, other: "UpscalingSettings", t: float) -> "UpscalingSettings":
        """Interpolate between two upscaling settings."""
        return UpscalingSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            upscaler_type=self.upscaler_type if t < 0.5 else other.upscaler_type,
            quality=self.quality if t < 0.5 else other.quality,
            sharpening_enabled=self.sharpening_enabled
            if t < 0.5
            else other.sharpening_enabled,
            sharpening_amount=self.sharpening_amount
            + (other.sharpening_amount - self.sharpening_amount) * t,
        )

    def get_recommended_mip_bias(self) -> float:
        """Get recommended texture mip bias for quality preset.

        Returns:
            Mip bias value.
        """
        from .constants import UPSCALING

        bias_table = {
            UpscaleQuality.ULTRA_PERFORMANCE: UPSCALING.MIP_BIAS_ULTRA_PERFORMANCE,
            UpscaleQuality.PERFORMANCE: UPSCALING.MIP_BIAS_PERFORMANCE,
            UpscaleQuality.BALANCED: UPSCALING.MIP_BIAS_BALANCED,
            UpscaleQuality.QUALITY: UPSCALING.MIP_BIAS_QUALITY,
            UpscaleQuality.ULTRA_QUALITY: UPSCALING.MIP_BIAS_ULTRA_QUALITY,
            UpscaleQuality.NATIVE_AA: UPSCALING.MIP_BIAS_NATIVE,
        }
        return bias_table.get(self.quality, UPSCALING.MIP_BIAS_NATIVE) + self.mip_bias_offset


class SpatialUpscaler(ABC):
    """Base class for spatial (single-frame) upscalers.

    Spatial upscalers work on a single frame without temporal
    information, making them simpler but potentially lower quality.
    """

    def __init__(self, name: str) -> None:
        """Initialize spatial upscaler.

        Args:
            name: Upscaler name for identification.
        """
        self._name: str = name
        self._output_buffer: Any = None
        self._input_width: int = 0
        self._input_height: int = 0
        self._output_width: int = 0
        self._output_height: int = 0

    @property
    def name(self) -> str:
        """Upscaler name."""
        return self._name

    @property
    def input_resolution(self) -> Tuple[int, int]:
        """Input resolution."""
        return (self._input_width, self._input_height)

    @property
    def output_resolution(self) -> Tuple[int, int]:
        """Output resolution."""
        return (self._output_width, self._output_height)

    def setup(
        self,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
    ) -> None:
        """Initialize upscaler resources.

        Args:
            input_width: Input render width.
            input_height: Input render height.
            output_width: Output display width.
            output_height: Output display height.
        """
        self._input_width = input_width
        self._input_height = input_height
        self._output_width = output_width
        self._output_height = output_height
        self._output_buffer = None

    @abstractmethod
    def upscale(
        self,
        color_buffer: Any,
        settings: UpscalingSettings,
    ) -> Any:
        """Upscale the input to output resolution.

        Args:
            color_buffer: Low-resolution input.
            settings: Upscaling settings.

        Returns:
            High-resolution output.
        """
        pass

    def cleanup(self) -> None:
        """Release upscaler resources."""
        self._output_buffer = None


class BilinearUpscaler(SpatialUpscaler):
    """Simple bilinear interpolation upscaler."""

    def __init__(self) -> None:
        super().__init__("Bilinear")

    def upscale(
        self,
        color_buffer: Any,
        settings: UpscalingSettings,
    ) -> Any:
        """Upscale using bilinear filtering.

        Args:
            color_buffer: Low-resolution input.
            settings: Upscaling settings.

        Returns:
            Upscaled output.
        """
        return self._output_buffer


class FSR1Upscaler(SpatialUpscaler):
    """AMD FidelityFX Super Resolution 1.0.

    Uses EASU (Edge-Adaptive Spatial Upsampling) followed by
    optional RCAS (Robust Contrast-Adaptive Sharpening).
    """

    def __init__(self) -> None:
        super().__init__("FSR 1.0")
        self._easu_buffer: Any = None

    def setup(
        self,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
    ) -> None:
        """Initialize FSR 1.0 resources."""
        super().setup(input_width, input_height, output_width, output_height)
        self._easu_buffer = None

    def upscale(
        self,
        color_buffer: Any,
        settings: UpscalingSettings,
    ) -> Any:
        """Upscale using FSR 1.0.

        Args:
            color_buffer: Low-resolution input.
            settings: Upscaling settings.

        Returns:
            Upscaled output.
        """
        # EASU pass
        self._easu_buffer = self._easu_pass(color_buffer)

        # Optional RCAS pass
        if settings.sharpening_enabled:
            return self._rcas_pass(self._easu_buffer, settings.sharpening_amount)

        return self._easu_buffer

    def _easu_pass(self, color: Any) -> Any:
        """Edge-Adaptive Spatial Upsampling."""
        return self._easu_buffer

    def _rcas_pass(self, color: Any, sharpness: float) -> Any:
        """Robust Contrast-Adaptive Sharpening."""
        return self._output_buffer


class CASUpscaler(SpatialUpscaler):
    """AMD Contrast Adaptive Sharpening.

    Primarily a sharpening filter that can also perform
    basic upscaling.
    """

    def __init__(self) -> None:
        super().__init__("CAS")

    def upscale(
        self,
        color_buffer: Any,
        settings: UpscalingSettings,
    ) -> Any:
        """Upscale using CAS.

        Args:
            color_buffer: Low-resolution input.
            settings: Upscaling settings.

        Returns:
            Sharpened/upscaled output.
        """
        return self._output_buffer


class TemporalUpscaler(ABC):
    """Base class for temporal upscalers.

    Temporal upscalers use multiple frames and motion vectors
    to produce higher quality results than spatial upscaling.
    """

    def __init__(self, name: str) -> None:
        """Initialize temporal upscaler.

        Args:
            name: Upscaler name.
        """
        self._name: str = name
        self._output_buffer: Any = None
        self._history_buffers: List[Any] = []
        self._input_width: int = 0
        self._input_height: int = 0
        self._output_width: int = 0
        self._output_height: int = 0
        self._frame_index: int = 0
        self._initialized: bool = False

    @property
    def name(self) -> str:
        """Upscaler name."""
        return self._name

    @property
    def initialized(self) -> bool:
        """Whether the upscaler is ready."""
        return self._initialized

    def setup(
        self,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
    ) -> None:
        """Initialize temporal upscaler resources.

        Args:
            input_width: Input render width.
            input_height: Input render height.
            output_width: Output display width.
            output_height: Output display height.
        """
        resolution_changed = (
            input_width != self._input_width
            or input_height != self._input_height
            or output_width != self._output_width
            or output_height != self._output_height
        )

        self._input_width = input_width
        self._input_height = input_height
        self._output_width = output_width
        self._output_height = output_height

        if resolution_changed:
            self._history_buffers = []
            self._frame_index = 0
            self._initialized = False

        self._output_buffer = None
        self._initialized = True

    @abstractmethod
    def upscale(
        self,
        color_buffer: Any,
        depth_buffer: Any,
        motion_vectors: Any,
        jitter_offset: Tuple[float, float],
        settings: UpscalingSettings,
        delta_time: float,
    ) -> Any:
        """Upscale using temporal accumulation.

        Args:
            color_buffer: Low-resolution color.
            depth_buffer: Low-resolution depth.
            motion_vectors: Per-pixel motion vectors.
            jitter_offset: Current frame jitter.
            settings: Upscaling settings.
            delta_time: Frame time.

        Returns:
            High-resolution output.
        """
        pass

    def reset(self) -> None:
        """Reset temporal accumulation."""
        self._history_buffers = []
        self._frame_index = 0

    def cleanup(self) -> None:
        """Release upscaler resources."""
        self._output_buffer = None
        self._history_buffers = []
        self._initialized = False


class FSR2Upscaler(TemporalUpscaler):
    """AMD FidelityFX Super Resolution 2.x.

    High-quality temporal upscaler using jitter, motion vectors,
    and reactive masks.
    """

    def __init__(self) -> None:
        super().__init__("FSR 2")
        self._reactive_mask: Any = None
        self._transparency_mask: Any = None

    def setup(
        self,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
    ) -> None:
        """Initialize FSR 2 resources."""
        super().setup(input_width, input_height, output_width, output_height)
        self._reactive_mask = None
        self._transparency_mask = None

    def upscale(
        self,
        color_buffer: Any,
        depth_buffer: Any,
        motion_vectors: Any,
        jitter_offset: Tuple[float, float],
        settings: UpscalingSettings,
        delta_time: float,
    ) -> Any:
        """Upscale using FSR 2.

        Args:
            color_buffer: Low-resolution color.
            depth_buffer: Low-resolution depth.
            motion_vectors: Per-pixel motion vectors.
            jitter_offset: Current frame jitter.
            settings: Upscaling settings.
            delta_time: Frame time.

        Returns:
            High-resolution output.
        """
        if settings.reset_accumulation:
            self.reset()

        self._frame_index += 1
        return self._output_buffer

    def set_reactive_mask(self, mask: Any) -> None:
        """Set reactive mask for areas needing faster response.

        Args:
            mask: Reactive mask (particles, alpha, etc.).
        """
        self._reactive_mask = mask

    def set_transparency_mask(self, mask: Any) -> None:
        """Set transparency and composition mask.

        Args:
            mask: Transparency mask.
        """
        self._transparency_mask = mask


class DLSSUpscaler(TemporalUpscaler):
    """NVIDIA Deep Learning Super Sampling.

    AI-based temporal upscaler using NVIDIA tensor cores.
    Requires NVIDIA RTX hardware.
    """

    def __init__(self) -> None:
        super().__init__("DLSS")
        self._available: bool = False
        self._ngx_handle: Any = None

    @property
    def available(self) -> bool:
        """Whether DLSS is available on current hardware."""
        return self._available

    def check_availability(self) -> bool:
        """Check if DLSS can be used.

        Returns:
            True if DLSS is available.
        """
        self._available = False
        return self._available

    def setup(
        self,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
    ) -> None:
        """Initialize DLSS resources."""
        if not self._available:
            return

        super().setup(input_width, input_height, output_width, output_height)

    def upscale(
        self,
        color_buffer: Any,
        depth_buffer: Any,
        motion_vectors: Any,
        jitter_offset: Tuple[float, float],
        settings: UpscalingSettings,
        delta_time: float,
    ) -> Any:
        """Upscale using DLSS.

        Args:
            color_buffer: Low-resolution color.
            depth_buffer: Low-resolution depth.
            motion_vectors: Per-pixel motion vectors.
            jitter_offset: Current frame jitter.
            settings: Upscaling settings.
            delta_time: Frame time.

        Returns:
            High-resolution output (or input if unavailable).
        """
        if not self._available:
            return color_buffer

        self._frame_index += 1
        return self._output_buffer


class XeSSUpscaler(TemporalUpscaler):
    """Intel Xe Super Sampling.

    AI-based temporal upscaler using Intel XMX units,
    with fallback to DP4a instructions on other hardware.
    """

    def __init__(self) -> None:
        super().__init__("XeSS")
        self._available: bool = False
        self._use_dp4a: bool = False  # Non-Intel fallback mode

    @property
    def available(self) -> bool:
        """Whether XeSS is available."""
        return self._available

    def check_availability(self) -> bool:
        """Check if XeSS can be used.

        Returns:
            True if XeSS is available.
        """
        self._available = False
        return self._available

    def upscale(
        self,
        color_buffer: Any,
        depth_buffer: Any,
        motion_vectors: Any,
        jitter_offset: Tuple[float, float],
        settings: UpscalingSettings,
        delta_time: float,
    ) -> Any:
        """Upscale using XeSS.

        Args:
            color_buffer: Low-resolution color.
            depth_buffer: Low-resolution depth.
            motion_vectors: Per-pixel motion vectors.
            jitter_offset: Current frame jitter.
            settings: Upscaling settings.
            delta_time: Frame time.

        Returns:
            High-resolution output.
        """
        if not self._available:
            return color_buffer

        self._frame_index += 1
        return self._output_buffer


# ============================================================================
# T-PP-6.3: TSR LANCZOS UPSAMPLING
# ============================================================================


def lanczos_kernel(x: float, a: int = 2) -> float:
    """Compute Lanczos kernel value.

    The Lanczos kernel is a sinc function windowed by a sinc window,
    providing high-quality interpolation with controllable ringing.

    Args:
        x: Distance from center.
        a: Lanczos parameter (2 or 3 typical). Larger values are smoother
           but have more ringing.

    Returns:
        Kernel weight at distance x.
    """
    if x == 0.0:
        return 1.0
    if abs(x) >= a:
        return 0.0

    pi_x = math.pi * x
    return (a * math.sin(pi_x) * math.sin(pi_x / a)) / (pi_x * pi_x)


def generate_lanczos_weights(scale: float, a: int = 2) -> List[Tuple[int, float]]:
    """Generate Lanczos filter weights for a given scale factor.

    Args:
        scale: Inverse of upscale factor (e.g., 0.5 for 2x upscale).
        a: Lanczos parameter (2 = sharper, 3 = smoother).

    Returns:
        List of (offset, weight) pairs for the filter kernel.
    """
    # Number of source pixels to sample in each direction
    radius = int(math.ceil(a / scale)) if scale > 0 else a

    weights: List[Tuple[int, float]] = []
    total = 0.0

    for i in range(-radius, radius + 1):
        offset = i * scale
        weight = lanczos_kernel(offset, a)
        if weight > 0.0001:
            weights.append((i, weight))
            total += weight

    # Normalize weights to sum to 1.0
    if total > 0:
        weights = [(offset, w / total) for offset, w in weights]

    return weights


def measure_local_contrast(
    center: Tuple[float, float, float],
    neighbors: List[Tuple[float, float, float]],
) -> float:
    """Measure local contrast using 3x3 box blur comparison.

    Computes the luminance difference between the center pixel and
    the average of its neighbors, providing a measure of local contrast
    for adaptive sharpening.

    Args:
        center: Center pixel RGB values (0-1 range).
        neighbors: List of neighboring pixel RGB values.

    Returns:
        Contrast value in [0, 1] range.
    """
    if not neighbors:
        return 0.0

    # Compute luminance using Rec. 709 coefficients
    def luminance(rgb: Tuple[float, float, float]) -> float:
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    center_lum = luminance(center)

    # Box blur average luminance
    avg_lum = sum(luminance(n) for n in neighbors) / len(neighbors)

    # Contrast is absolute difference normalized (scaled by 2 for sensitivity)
    return min(abs(center_lum - avg_lum) * 2.0, 1.0)


def get_adaptive_sharpening_for_quality(quality: str) -> Tuple[float, float]:
    """Return (min_sharpening, max_sharpening) for quality level.

    Higher quality levels use more aggressive sharpening ranges
    while lower quality levels use gentler sharpening to reduce
    potential artifacts.

    Args:
        quality: Quality preset name ("ultra", "high", "medium", "low").

    Returns:
        Tuple of (min_sharpening, max_sharpening) values.
    """
    presets = {
        "ultra": (0.4, 0.9),
        "high": (0.3, 0.8),
        "medium": (0.2, 0.6),
        "low": (0.1, 0.4),
    }
    return presets.get(quality.lower(), (0.3, 0.8))


class LanczosKernel(Enum):
    """Lanczos kernel size."""

    LANCZOS2 = 2  # Sharper, slight ringing
    LANCZOS3 = 3  # Smoother, less ringing


@dataclass
class TSRLanczosSettings:
    """Temporal Super-Resolution with Lanczos settings."""

    enabled: bool = True
    kernel: LanczosKernel = LanczosKernel.LANCZOS2
    scale_factor: float = 2.0  # 1.5, 2.0, 3.0 common values
    sharpness: float = 0.5  # Post-filter sharpening (0-1)
    sharpening: bool = True  # Enable sharpening

    # Temporal accumulation
    temporal_blend: float = 0.1  # Blend with history (0 = no temporal)
    motion_scale: float = 1.0  # Motion vector scaling

    # Anti-aliasing
    jitter_sequence: str = "halton_8"  # Jitter pattern name

    # Quality
    separable: bool = True  # Use separable filter (faster)

    # Adaptive sharpening (T-PP-6.4)
    adaptive_sharpening: bool = True
    sharpening_min: float = 0.3  # Minimum sharpening for low contrast areas
    sharpening_max: float = 0.8  # Maximum sharpening for high contrast areas
    contrast_threshold: float = 0.1  # Below this contrast, use min sharpening


class TSRLanczosUpscaler:
    """Native Lanczos-based temporal super-resolution.

    Provides a fallback when DLSS/FSR2/XeSS are unavailable.
    Uses Lanczos interpolation with optional temporal accumulation.

    The Lanczos filter is a high-quality resampling filter based on
    the sinc function. It provides excellent sharpness while minimizing
    aliasing artifacts.

    Attributes:
        settings: TSR Lanczos configuration.
    """

    def __init__(self, settings: Optional[TSRLanczosSettings] = None) -> None:
        """Initialize TSR Lanczos upscaler.

        Args:
            settings: Optional configuration. Uses defaults if not provided.
        """
        self.settings = settings or TSRLanczosSettings()
        self._weights_h: List[Tuple[int, float]] = []
        self._weights_v: List[Tuple[int, float]] = []
        self._history_buffer: Optional[Any] = None
        self._frame_index: int = 0
        self._rebuild_weights()

    def _rebuild_weights(self) -> None:
        """Rebuild filter weights for current settings."""
        a = self.settings.kernel.value
        scale = self.settings.scale_factor

        weights = generate_lanczos_weights(1.0 / scale, a)
        self._weights_h = weights
        self._weights_v = weights

    @property
    def kernel_radius(self) -> int:
        """Get kernel radius in source pixels."""
        return self.settings.kernel.value

    @property
    def output_scale(self) -> Tuple[float, float]:
        """Get output resolution multiplier."""
        return (self.settings.scale_factor, self.settings.scale_factor)

    def get_jitter_offset(self) -> Tuple[float, float]:
        """Get sub-pixel jitter offset for current frame.

        Uses Halton sequence (bases 2, 3) for temporally stable
        sub-pixel sampling positions.

        Returns:
            (x, y) offset in pixels, centered around 0.
        """

        def halton(index: int, base: int) -> float:
            """Compute Halton sequence value."""
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        # Determine sequence length from settings
        seq_length = 8  # Default Halton_8
        if self.settings.jitter_sequence == "halton_16":
            seq_length = 16
        elif self.settings.jitter_sequence == "halton_32":
            seq_length = 32

        idx = self._frame_index % seq_length
        # Use index + 1 to avoid (0, 0) at start
        jx = halton(idx + 1, 2) - 0.5
        jy = halton(idx + 1, 3) - 0.5

        return (jx, jy)

    def sample_lanczos(
        self,
        source: List[List[Tuple[float, float, float]]],
        x: float,
        y: float,
    ) -> Tuple[float, float, float]:
        """Sample source image using Lanczos interpolation.

        Args:
            source: 2D array of RGB tuples.
            x: Sample position X in source coordinates.
            y: Sample position Y in source coordinates.

        Returns:
            Interpolated RGB color.
        """
        height = len(source)
        width = len(source[0]) if height > 0 else 0

        if self.settings.separable:
            # Separable filter: horizontal then vertical (faster)
            return self._sample_separable(source, x, y, width, height)
        else:
            # Full 2D filter (higher quality)
            return self._sample_2d(source, x, y, width, height)

    def _sample_separable(
        self,
        source: List[List[Tuple[float, float, float]]],
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> Tuple[float, float, float]:
        """Separable Lanczos sampling (faster).

        Applies horizontal filter first, then vertical.
        Complexity is O(2n) vs O(n^2) for full 2D.
        """
        xi = int(x)
        yi = int(y)

        # Horizontal pass: sample along row
        row_samples: List[Tuple[Tuple[float, float, float], float]] = []
        for offset, weight in self._weights_h:
            sx = max(0, min(width - 1, xi + offset))
            sy = max(0, min(height - 1, yi))
            row_samples.append((source[sy][sx], weight))

        # Accumulate horizontal weighted samples
        hr = sum(s[0] * w for s, w in row_samples)
        hg = sum(s[1] * w for s, w in row_samples)
        hb = sum(s[2] * w for s, w in row_samples)

        return (hr, hg, hb)

    def _sample_2d(
        self,
        source: List[List[Tuple[float, float, float]]],
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> Tuple[float, float, float]:
        """Full 2D Lanczos sampling.

        Applies the full 2D kernel for highest quality.
        """
        xi = int(x)
        yi = int(y)

        r, g, b = 0.0, 0.0, 0.0
        total = 0.0

        for oy, wy in self._weights_v:
            for ox, wx in self._weights_h:
                sx = max(0, min(width - 1, xi + ox))
                sy = max(0, min(height - 1, yi + oy))

                weight = wx * wy
                pixel = source[sy][sx]

                r += pixel[0] * weight
                g += pixel[1] * weight
                b += pixel[2] * weight
                total += weight

        if total > 0:
            r /= total
            g /= total
            b /= total

        return (r, g, b)

    def advance_frame(self) -> None:
        """Advance to next frame in temporal sequence."""
        self._frame_index += 1

    def apply_sharpening(
        self,
        color: Tuple[float, float, float],
        neighbors: List[Tuple[float, float, float]],
    ) -> Tuple[float, float, float]:
        """Apply adaptive sharpening based on local contrast.

        Uses unsharp mask with strength that varies based on local contrast.
        High contrast areas (edges) get more sharpening while low contrast
        areas (flat regions) get less to avoid noise amplification.

        Args:
            color: Center pixel color (RGB in 0-1 range).
            neighbors: List of neighboring pixel colors.

        Returns:
            Sharpened color (clamped to 0-1 range).
        """
        # Early exit if sharpening is disabled
        if not self.settings.sharpening or self.settings.sharpness <= 0:
            return color

        if not neighbors:
            return color

        # Measure local contrast for adaptive strength
        contrast = measure_local_contrast(color, neighbors)

        # Determine sharpening strength
        if self.settings.adaptive_sharpening:
            if contrast < self.settings.contrast_threshold:
                # Low contrast: use minimum sharpening
                strength = self.settings.sharpening_min
            else:
                # Normalize contrast above threshold to [0, 1]
                denominator = 1.0 - self.settings.contrast_threshold
                if denominator <= 0.0:
                    # Threshold is at or above 1.0, use minimum sharpening
                    t = 0.0
                else:
                    t = min(
                        (contrast - self.settings.contrast_threshold) / denominator,
                        1.0,
                    )
                # Lerp between min and max sharpening based on contrast
                strength = self.settings.sharpening_min + t * (
                    self.settings.sharpening_max - self.settings.sharpening_min
                )
        else:
            # Non-adaptive: use fixed sharpness value
            strength = self.settings.sharpness

        # Calculate average neighbor color (box blur)
        n_count = len(neighbors)
        avg = (
            sum(n[0] for n in neighbors) / n_count,
            sum(n[1] for n in neighbors) / n_count,
            sum(n[2] for n in neighbors) / n_count,
        )

        # Apply unsharp mask: sharpened = center + strength * (center - blur)
        # Clamp to [0, 1] to prevent overshoot
        result = tuple(
            max(0.0, min(1.0, c + strength * (c - a))) for c, a in zip(color, avg)
        )

        return result  # type: ignore[return-value]

    def get_budget_ms(self) -> float:
        """Estimate GPU time in milliseconds.

        Returns:
            Estimated processing time based on settings.
        """
        base = 0.5  # Base cost
        scale_factor = self.settings.scale_factor
        kernel_size = self.settings.kernel.value * 2 + 1

        if self.settings.separable:
            return base + 0.1 * scale_factor * kernel_size
        else:
            return base + 0.1 * scale_factor * kernel_size * kernel_size

    def reset(self) -> None:
        """Reset temporal accumulation."""
        self._history_buffer = None
        self._frame_index = 0

    @staticmethod
    def is_available() -> bool:
        """Check if TSR Lanczos is available.

        TSR Lanczos is always available as it's a CPU/GPU fallback
        that doesn't require specific hardware.

        Returns:
            Always True.
        """
        return True


def create_tsr_lanczos(
    scale: float = 2.0,
    kernel: LanczosKernel = LanczosKernel.LANCZOS2,
    temporal: bool = True,
) -> TSRLanczosUpscaler:
    """Create TSR Lanczos upscaler with common settings.

    Factory function for convenient creation of TSR Lanczos upscaler
    with typical configurations.

    Args:
        scale: Upscale factor (e.g., 2.0 for 2x upscale).
        kernel: Lanczos kernel size (LANCZOS2 or LANCZOS3).
        temporal: Whether to enable temporal accumulation.

    Returns:
        Configured TSRLanczosUpscaler instance.
    """
    settings = TSRLanczosSettings(
        scale_factor=scale,
        kernel=kernel,
        temporal_blend=0.1 if temporal else 0.0,
    )
    return TSRLanczosUpscaler(settings)


class UpscalingEffect(PostProcessEffect[UpscalingSettings]):
    """Complete upscaling post-process effect."""

    def __init__(
        self,
        settings: Optional[UpscalingSettings] = None,
    ) -> None:
        """Initialize upscaling effect.

        Args:
            settings: Upscaling configuration.
        """
        super().__init__(
            name="Upscaling",
            settings=settings or UpscalingSettings(),
            priority=EffectPriority.UPSCALING.value,
        )

        # Spatial upscalers
        self._bilinear: BilinearUpscaler = BilinearUpscaler()
        self._fsr1: FSR1Upscaler = FSR1Upscaler()
        self._cas: CASUpscaler = CASUpscaler()

        # Temporal upscalers
        self._fsr2: FSR2Upscaler = FSR2Upscaler()
        self._dlss: DLSSUpscaler = DLSSUpscaler()
        self._xess: XeSSUpscaler = XeSSUpscaler()
        self._tsr_lanczos: TSRLanczosUpscaler = TSRLanczosUpscaler()

        self._current_upscaler: Optional[str] = None
        self._jitter_offset: Tuple[float, float] = (0.0, 0.0)

    @property
    def current_upscaler(self) -> Optional[str]:
        """Name of currently active upscaler."""
        return self._current_upscaler

    def set_jitter_offset(self, x: float, y: float) -> None:
        """Set current frame jitter offset.

        Args:
            x: X jitter in pixels.
            y: Y jitter in pixels.
        """
        self._jitter_offset = (x, y)

    def get_render_resolution(
        self,
        output_width: int,
        output_height: int,
    ) -> Tuple[int, int]:
        """Get recommended render resolution.

        Args:
            output_width: Target output width.
            output_height: Target output height.

        Returns:
            (render_width, render_height).
        """
        if not self._settings:
            return (output_width, output_height)

        return get_render_resolution(
            output_width,
            output_height,
            self._settings.quality,
        )

    def get_required_inputs(self) -> List[str]:
        """Get required input resources."""
        inputs = ["color"]

        if self._settings and self._settings.upscaler_type in (
            UpscalerType.FSR_2,
            UpscalerType.FSR_3,
            UpscalerType.DLSS,
            UpscalerType.DLSS_FG,
            UpscalerType.XESS,
            UpscalerType.TSR_LANCZOS,
        ):
            inputs.extend(["depth", "velocity"])

        return inputs

    def get_outputs(self) -> List[str]:
        """Get output resources."""
        return ["color"]

    def setup(self, width: int, height: int) -> None:
        """Initialize upscaling resources.

        Args:
            width: Output display width.
            height: Output display height.
        """
        if not self._settings:
            return

        render_width, render_height = self.get_render_resolution(width, height)

        # Setup spatial upscalers
        self._bilinear.setup(render_width, render_height, width, height)
        self._fsr1.setup(render_width, render_height, width, height)
        self._cas.setup(render_width, render_height, width, height)

        # Setup temporal upscalers
        self._fsr2.setup(render_width, render_height, width, height)
        self._dlss.setup(render_width, render_height, width, height)
        self._xess.setup(render_width, render_height, width, height)

    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        """Execute upscaling.

        Args:
            inputs: Input buffers.
            outputs: Output buffers.
            delta_time: Frame time.
        """
        if not self._settings or not self._settings.enabled:
            return

        color = inputs.get("color")
        depth = inputs.get("depth")
        velocity = inputs.get("velocity")

        upscaler_type = self._settings.upscaler_type

        if upscaler_type == UpscalerType.NONE:
            self._current_upscaler = None
            return

        elif upscaler_type == UpscalerType.BILINEAR:
            self._bilinear.upscale(color, self._settings)
            self._current_upscaler = "Bilinear"

        elif upscaler_type == UpscalerType.FSR_1:
            self._fsr1.upscale(color, self._settings)
            self._current_upscaler = "FSR 1.0"

        elif upscaler_type == UpscalerType.CAS:
            self._cas.upscale(color, self._settings)
            self._current_upscaler = "CAS"

        elif upscaler_type in (UpscalerType.FSR_2, UpscalerType.FSR_3):
            self._fsr2.upscale(
                color,
                depth,
                velocity,
                self._jitter_offset,
                self._settings,
                delta_time,
            )
            self._current_upscaler = "FSR 2"

        elif upscaler_type in (UpscalerType.DLSS, UpscalerType.DLSS_FG):
            if self._dlss.available:
                self._dlss.upscale(
                    color,
                    depth,
                    velocity,
                    self._jitter_offset,
                    self._settings,
                    delta_time,
                )
                self._current_upscaler = "DLSS"
            else:
                # Fallback to FSR 2
                self._fsr2.upscale(
                    color,
                    depth,
                    velocity,
                    self._jitter_offset,
                    self._settings,
                    delta_time,
                )
                self._current_upscaler = "FSR 2 (DLSS fallback)"

        elif upscaler_type == UpscalerType.XESS:
            if self._xess.available:
                self._xess.upscale(
                    color,
                    depth,
                    velocity,
                    self._jitter_offset,
                    self._settings,
                    delta_time,
                )
                self._current_upscaler = "XeSS"
            else:
                self._fsr2.upscale(
                    color,
                    depth,
                    velocity,
                    self._jitter_offset,
                    self._settings,
                    delta_time,
                )
                self._current_upscaler = "FSR 2 (XeSS fallback)"

        elif upscaler_type == UpscalerType.TSR_LANCZOS:
            # Native Lanczos-based TSR (always available)
            self._tsr_lanczos.advance_frame()
            self._current_upscaler = "TSR Lanczos"

    def cleanup(self) -> None:
        """Release upscaling resources."""
        self._bilinear.cleanup()
        self._fsr1.cleanup()
        self._cas.cleanup()
        self._fsr2.cleanup()
        self._dlss.cleanup()
        self._xess.cleanup()
        self._tsr_lanczos.reset()

    def reset_accumulation(self) -> None:
        """Reset temporal accumulation (call on camera cuts)."""
        self._fsr2.reset()
        self._dlss.reset()
        self._xess.reset()
        self._tsr_lanczos.reset()


__all__ = [
    "UpscalerType",
    "UpscaleQuality",
    "FrameGenerationMode",
    "UpscaleResolution",
    "get_render_resolution",
    "UpscalingSettings",
    "SpatialUpscaler",
    "BilinearUpscaler",
    "FSR1Upscaler",
    "CASUpscaler",
    "TemporalUpscaler",
    "FSR2Upscaler",
    "DLSSUpscaler",
    "XeSSUpscaler",
    "UpscalingEffect",
    # TSR Lanczos (T-PP-6.3)
    "lanczos_kernel",
    "generate_lanczos_weights",
    "LanczosKernel",
    "TSRLanczosSettings",
    "TSRLanczosUpscaler",
    "create_tsr_lanczos",
    # TSR Adaptive Sharpening (T-PP-6.4)
    "measure_local_contrast",
    "get_adaptive_sharpening_for_quality",
]
