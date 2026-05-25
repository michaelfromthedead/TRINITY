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
]
