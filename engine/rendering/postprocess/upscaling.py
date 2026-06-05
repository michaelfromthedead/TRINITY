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


class LanczosKernel(Enum):
    """Lanczos kernel size enum."""

    LANCZOS2 = 2  # Lanczos-2: radius 2, good balance of sharpness/ringing
    LANCZOS3 = 3  # Lanczos-3: radius 3, sharper but more ringing


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
    TSR_LANCZOS = auto()  # Native Lanczos-based temporal super-resolution


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


def lanczos_kernel(x: float, a: int = 2) -> float:
    """Compute Lanczos kernel value at position x.

    The Lanczos kernel is defined as:
        L(x) = sinc(x) * sinc(x/a)  if |x| < a
        L(x) = 0                     otherwise

    where sinc(x) = sin(pi*x) / (pi*x) for x != 0, and sinc(0) = 1.

    Args:
        x: Position to evaluate kernel at.
        a: Kernel size (2 for Lanczos-2, 3 for Lanczos-3).

    Returns:
        Kernel value at position x.
    """
    if x == 0.0:
        return 1.0

    if abs(x) >= a:
        return 0.0

    # sinc(x) = sin(pi*x) / (pi*x)
    pi_x = math.pi * x
    sinc_x = math.sin(pi_x) / pi_x

    # sinc(x/a)
    pi_x_a = math.pi * x / a
    sinc_x_a = math.sin(pi_x_a) / pi_x_a

    return sinc_x * sinc_x_a


def generate_lanczos_weights(
    scale: float,
    a: int = 2,
    threshold: float = 0.0001,
) -> List[Tuple[int, float]]:
    """Generate normalized Lanczos filter weights for a given scale factor.

    Args:
        scale: Scale factor (< 1 means upscaling, e.g., 0.5 = 2x upscale).
        a: Lanczos kernel size (2 or 3).
        threshold: Minimum weight magnitude to include.

    Returns:
        List of (offset, weight) tuples, normalized to sum to 1.0.
    """
    if scale <= 0:
        scale = 1.0

    # Calculate support radius
    radius = int(math.ceil(a / scale))

    # Generate raw weights
    weights: List[Tuple[int, float]] = []
    for i in range(-radius, radius + 1):
        w = lanczos_kernel(i * scale, a)
        if abs(w) > threshold:
            weights.append((i, w))

    # Normalize weights to sum to 1.0
    if weights:
        total = sum(w for _, w in weights)
        if total != 0:
            weights = [(offset, w / total) for offset, w in weights]

    return weights


def measure_local_contrast(
    center: Tuple[float, float, float],
    neighbors: List[Tuple[float, float, float]],
) -> float:
    """Measure local contrast between center pixel and neighbors.

    Uses luminance-weighted contrast calculation (Rec. 709).

    Args:
        center: Center pixel RGB (0-1 range).
        neighbors: List of neighbor pixel RGB values.

    Returns:
        Contrast value in [0, 1] range.
    """
    if not neighbors:
        return 0.0

    # Rec. 709 luminance weights
    def luminance(rgb: Tuple[float, float, float]) -> float:
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    center_lum = luminance(center)

    # Calculate average neighbor luminance
    neighbor_lums = [luminance(n) for n in neighbors]
    avg_lum = sum(neighbor_lums) / len(neighbor_lums)

    # Contrast is the absolute difference, scaled by 2 and clamped
    contrast = abs(center_lum - avg_lum) * 2.0
    return min(1.0, max(0.0, contrast))


def get_adaptive_sharpening_for_quality(quality: str) -> Tuple[float, float]:
    """Get recommended adaptive sharpening min/max for a quality preset.

    Args:
        quality: Quality preset name ("ultra", "high", "medium", "low").

    Returns:
        Tuple of (min_sharpening, max_sharpening) values.
    """
    sharpening_table = {
        "ultra": (0.4, 0.9),
        "high": (0.3, 0.8),
        "medium": (0.2, 0.6),
        "low": (0.1, 0.4),
    }
    return sharpening_table.get(quality.lower(), (0.3, 0.8))


@dataclass
class TSRLanczosSettings:
    """Settings for TSR Lanczos upscaler.

    Attributes:
        enabled: Whether the upscaler is enabled.
        kernel: Lanczos kernel size (LANCZOS2 or LANCZOS3).
        scale_factor: Target upscale factor (e.g., 2.0 for 2x).
        sharpness: Post-upscale sharpening amount [0, 1].
        temporal_blend: Temporal blending factor [0, 1] (0 = disabled).
        separable: Use separable (faster) vs 2D (higher quality) filtering.
        jitter_sequence: Jitter pattern name ("halton_8" or "halton_16").
        sharpening: Enable sharpening.
        adaptive_sharpening: Use contrast-adaptive sharpening.
        sharpening_min: Minimum sharpening strength for adaptive mode.
        sharpening_max: Maximum sharpening strength for adaptive mode.
        contrast_threshold: Contrast level below which min sharpening is used.
    """

    enabled: bool = True
    kernel: LanczosKernel = LanczosKernel.LANCZOS2
    scale_factor: float = 2.0
    sharpness: float = 0.5
    temporal_blend: float = 0.1
    separable: bool = True
    jitter_sequence: str = "halton_8"
    sharpening: bool = True
    adaptive_sharpening: bool = True
    sharpening_min: float = 0.3
    sharpening_max: float = 0.8
    contrast_threshold: float = 0.1


class TSRLanczosUpscaler:
    """Temporal Super-Resolution Lanczos Upscaler.

    A native Lanczos-based temporal super-resolution implementation
    that serves as a fallback when DLSS/FSR2/XeSS are unavailable.

    Features:
        - Lanczos-2 or Lanczos-3 kernel for high-quality spatial upscaling
        - Halton jitter sequence for temporal anti-aliasing
        - Separable filter option for better performance
        - Adaptive sharpening based on local contrast
        - Temporal blending with previous frames

    Example:
        >>> upscaler = TSRLanczosUpscaler()
        >>> for frame in frames:
        ...     jitter = upscaler.get_jitter_offset()
        ...     apply_jitter(camera, jitter)
        ...     render()
        ...     output = upscaler.upscale(color, depth, motion)
        ...     upscaler.advance_frame()
    """

    # Halton sequences (base 2 and 3, centered at 0)
    _HALTON_8_X = [0.0, -0.25, 0.25, -0.375, 0.125, -0.125, 0.375, -0.4375]
    _HALTON_8_Y = [
        1 / 3 - 0.5,
        2 / 3 - 0.5,
        1 / 9 - 0.5,
        4 / 9 - 0.5,
        7 / 9 - 0.5,
        2 / 9 - 0.5,
        5 / 9 - 0.5,
        8 / 9 - 0.5,
    ]

    def __init__(self, settings: Optional[TSRLanczosSettings] = None) -> None:
        """Initialize TSR Lanczos upscaler.

        Args:
            settings: Upscaler configuration, or None for defaults.
        """
        self._settings: TSRLanczosSettings = settings or TSRLanczosSettings()
        self._frame_index: int = 0
        self._history_buffer: Any = None

        # Generate Halton-16 sequence (extend from Halton-8)
        self._halton_16_x: List[float] = []
        self._halton_16_y: List[float] = []
        self._generate_halton_16()

        # Pre-compute Lanczos weights for horizontal and vertical passes
        self._weights_h: List[Tuple[int, float]] = []
        self._weights_v: List[Tuple[int, float]] = []
        self._compute_weights()

    def _generate_halton_16(self) -> None:
        """Generate 16-sample Halton sequence."""
        # Base-2 Halton sequence for X
        for i in range(1, 17):
            x = 0.0
            f = 0.5
            j = i
            while j > 0:
                x += f * (j % 2)
                j //= 2
                f *= 0.5
            self._halton_16_x.append(x - 0.5)

        # Base-3 Halton sequence for Y
        for i in range(1, 17):
            y = 0.0
            f = 1 / 3
            j = i
            while j > 0:
                y += f * (j % 3)
                j //= 3
                f /= 3
            self._halton_16_y.append(y - 0.5)

    def _compute_weights(self) -> None:
        """Pre-compute Lanczos weights for horizontal and vertical passes."""
        scale = 1.0 / self._settings.scale_factor if self._settings.scale_factor > 0 else 1.0
        a = self._settings.kernel.value

        self._weights_h = generate_lanczos_weights(scale, a)
        self._weights_v = generate_lanczos_weights(scale, a)

    @staticmethod
    def is_available() -> bool:
        """Check if TSR Lanczos is available.

        Always returns True as it's a pure software implementation.

        Returns:
            Always True.
        """
        return True

    @property
    def settings(self) -> TSRLanczosSettings:
        """Get current settings."""
        return self._settings

    @property
    def kernel_radius(self) -> int:
        """Get kernel radius (2 for Lanczos-2, 3 for Lanczos-3)."""
        return self._settings.kernel.value

    @property
    def output_scale(self) -> Tuple[float, float]:
        """Get output scale factor (x, y)."""
        return (self._settings.scale_factor, self._settings.scale_factor)

    def get_jitter_offset(self) -> Tuple[float, float]:
        """Get current frame jitter offset for temporal AA.

        Returns:
            (x, y) jitter offset in [-0.5, 0.5] range.
        """
        if self._settings.jitter_sequence == "halton_16":
            idx = self._frame_index % 16
            return (self._halton_16_x[idx], self._halton_16_y[idx])
        else:  # halton_8
            idx = self._frame_index % 8
            return (self._HALTON_8_X[idx], self._HALTON_8_Y[idx])

    def advance_frame(self) -> None:
        """Advance to next frame in jitter sequence."""
        self._frame_index += 1

    def reset(self) -> None:
        """Reset temporal accumulation and frame index."""
        self._frame_index = 0
        self._history_buffer = None

    def sample_lanczos(
        self,
        image: List[List[Tuple[float, float, float]]],
        x: float,
        y: float,
    ) -> Tuple[float, float, float]:
        """Sample image at fractional position using Lanczos filter.

        Args:
            image: 2D image as list of rows of RGB tuples.
            x: X coordinate (fractional).
            y: Y coordinate (fractional).

        Returns:
            Interpolated RGB color.
        """
        if not image or not image[0]:
            return (0.0, 0.0, 0.0)

        height = len(image)
        width = len(image[0])
        a = self._settings.kernel.value

        if self._settings.separable:
            return self._sample_separable(image, x, y, width, height, a)
        else:
            return self._sample_2d(image, x, y, width, height, a)

    def _sample_separable(
        self,
        image: List[List[Tuple[float, float, float]]],
        x: float,
        y: float,
        width: int,
        height: int,
        a: int,
    ) -> Tuple[float, float, float]:
        """Sample using separable (horizontal then vertical) Lanczos."""
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        fx = x - ix
        fy = y - iy

        # First pass: horizontal filter for each row
        h_samples: List[Tuple[float, float, float]] = []
        for row_offset in range(-a + 1, a + 1):
            row_idx = min(max(iy + row_offset, 0), height - 1)
            row = image[row_idx]

            r, g, b = 0.0, 0.0, 0.0
            total_weight = 0.0

            for col_offset in range(-a + 1, a + 1):
                col_idx = min(max(ix + col_offset, 0), width - 1)
                dist = col_offset - fx
                w = lanczos_kernel(dist, a)
                pixel = row[col_idx]
                r += pixel[0] * w
                g += pixel[1] * w
                b += pixel[2] * w
                total_weight += w

            if total_weight > 0:
                h_samples.append((r / total_weight, g / total_weight, b / total_weight))
            else:
                h_samples.append((0.0, 0.0, 0.0))

        # Second pass: vertical filter
        r, g, b = 0.0, 0.0, 0.0
        total_weight = 0.0

        for i, sample in enumerate(h_samples):
            row_offset = i - a + 1
            dist = row_offset - fy
            w = lanczos_kernel(dist, a)
            r += sample[0] * w
            g += sample[1] * w
            b += sample[2] * w
            total_weight += w

        if total_weight > 0:
            return (r / total_weight, g / total_weight, b / total_weight)
        return (0.0, 0.0, 0.0)

    def _sample_2d(
        self,
        image: List[List[Tuple[float, float, float]]],
        x: float,
        y: float,
        width: int,
        height: int,
        a: int,
    ) -> Tuple[float, float, float]:
        """Sample using full 2D Lanczos kernel."""
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        fx = x - ix
        fy = y - iy

        r, g, b = 0.0, 0.0, 0.0
        total_weight = 0.0

        for row_offset in range(-a + 1, a + 1):
            row_idx = min(max(iy + row_offset, 0), height - 1)
            row = image[row_idx]
            dy = row_offset - fy

            for col_offset in range(-a + 1, a + 1):
                col_idx = min(max(ix + col_offset, 0), width - 1)
                dx = col_offset - fx

                # 2D Lanczos is product of 1D Lanczos
                w = lanczos_kernel(dx, a) * lanczos_kernel(dy, a)
                pixel = row[col_idx]
                r += pixel[0] * w
                g += pixel[1] * w
                b += pixel[2] * w
                total_weight += w

        if total_weight > 0:
            return (r / total_weight, g / total_weight, b / total_weight)
        return (0.0, 0.0, 0.0)

    def apply_sharpening(
        self,
        color: Tuple[float, float, float],
        neighbors: List[Tuple[float, float, float]],
    ) -> Tuple[float, float, float]:
        """Apply contrast-adaptive sharpening.

        When adaptive_sharpening is enabled, the sharpening strength is
        interpolated between sharpening_min and sharpening_max based on
        the local contrast. Low contrast areas get less sharpening to
        avoid amplifying noise.

        Args:
            color: Center pixel RGB.
            neighbors: List of neighbor pixel RGB values.

        Returns:
            Sharpened RGB color.
        """
        # Early exit conditions
        if not self._settings.sharpening:
            return color
        if self._settings.sharpness <= 0:
            # Zero sharpness means no sharpening, regardless of adaptive mode
            return color
        if not neighbors:
            return color

        # Calculate neighbor average
        avg_r = sum(n[0] for n in neighbors) / len(neighbors)
        avg_g = sum(n[1] for n in neighbors) / len(neighbors)
        avg_b = sum(n[2] for n in neighbors) / len(neighbors)

        # Determine sharpening strength
        if self._settings.adaptive_sharpening:
            # Calculate local contrast
            contrast = measure_local_contrast(color, neighbors)

            # Interpolate between min and max based on contrast
            threshold = self._settings.contrast_threshold
            if contrast <= threshold:
                sharpness = self._settings.sharpening_min
            else:
                # Linear interpolation from min to max
                t = min(1.0, (contrast - threshold) / (1.0 - threshold))
                sharpness = (
                    self._settings.sharpening_min
                    + t * (self._settings.sharpening_max - self._settings.sharpening_min)
                )
        else:
            # Use fixed sharpness value
            sharpness = self._settings.sharpness

        if sharpness <= 0:
            return color

        # Unsharp mask: output = center + sharpness * (center - avg)
        r = color[0] + sharpness * (color[0] - avg_r)
        g = color[1] + sharpness * (color[1] - avg_g)
        b = color[2] + sharpness * (color[2] - avg_b)

        # Clamp to valid range
        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))

        return (r, g, b)

    def get_budget_ms(self) -> float:
        """Estimate performance budget in milliseconds.

        Returns:
            Estimated GPU time in milliseconds.
        """
        # Base cost depends on kernel size and filter type
        kernel_cost = 0.2 if self._settings.kernel == LanczosKernel.LANCZOS2 else 0.4

        # Separable is roughly half the cost
        if self._settings.separable:
            kernel_cost *= 0.5

        # Scale factor affects output pixel count
        scale_cost = self._settings.scale_factor * 0.1

        return kernel_cost + scale_cost


def create_tsr_lanczos(
    scale: float = 2.0,
    kernel: LanczosKernel = LanczosKernel.LANCZOS2,
    temporal: bool = True,
) -> TSRLanczosUpscaler:
    """Factory function to create a TSR Lanczos upscaler.

    Args:
        scale: Target upscale factor.
        kernel: Lanczos kernel size.
        temporal: Enable temporal blending.

    Returns:
        Configured TSRLanczosUpscaler instance.
    """
    settings = TSRLanczosSettings(
        kernel=kernel,
        scale_factor=scale,
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

    def cleanup(self) -> None:
        """Release upscaling resources."""
        self._bilinear.cleanup()
        self._fsr1.cleanup()
        self._cas.cleanup()
        self._fsr2.cleanup()
        self._dlss.cleanup()
        self._xess.cleanup()

    def reset_accumulation(self) -> None:
        """Reset temporal accumulation (call on camera cuts)."""
        self._fsr2.reset()
        self._dlss.reset()
        self._xess.reset()


__all__ = [
    # Enums
    "LanczosKernel",
    "UpscalerType",
    "UpscaleQuality",
    "FrameGenerationMode",
    # Data classes
    "UpscaleResolution",
    "UpscalingSettings",
    "TSRLanczosSettings",
    # Functions
    "get_render_resolution",
    "lanczos_kernel",
    "generate_lanczos_weights",
    "measure_local_contrast",
    "get_adaptive_sharpening_for_quality",
    "create_tsr_lanczos",
    # Spatial upscalers
    "SpatialUpscaler",
    "BilinearUpscaler",
    "FSR1Upscaler",
    "CASUpscaler",
    # Temporal upscalers
    "TemporalUpscaler",
    "FSR2Upscaler",
    "DLSSUpscaler",
    "XeSSUpscaler",
    "TSRLanczosUpscaler",
    # Effect
    "UpscalingEffect",
]
