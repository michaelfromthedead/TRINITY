"""
Upscaler Plugin Interface (T-PP-6.5)

Provides an abstract plugin interface for vendor upscalers (DLSS, FSR, XeSS)
with runtime auto-detection and graceful fallback to TSR Lanczos.

The plugin system enables:
- Unified interface for all vendor upscaling SDKs
- Runtime detection of available SDKs/hardware
- Automatic fallback chain: DLSS -> XeSS -> FSR -> TSR
- Easy integration of new upscaler technologies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Type

from .upscaling import TSRLanczosUpscaler, TSRLanczosSettings, LanczosKernel


# Type alias for texture handles (placeholder for actual GPU texture type)
TextureHandle = Any


# ==============================================================================
# UPSCALER CAPABILITIES
# ==============================================================================


@dataclass
class UpscalerCapabilities:
    """Capabilities reported by an upscaler plugin.

    Attributes:
        name: Human-readable name of the upscaler.
        version: Version string of the upscaler SDK/implementation.
        supports_sharpening: Whether the upscaler has built-in sharpening.
        supports_hdr: Whether HDR input/output is supported.
        min_scale: Minimum supported upscale factor.
        max_scale: Maximum supported upscale factor.
        requires_motion_vectors: Whether motion vectors are required.
        requires_depth: Whether depth buffer is required.
        supports_frame_generation: Whether frame generation is available.
        vendor: Hardware vendor requirement (empty = any).
    """

    name: str
    version: str
    supports_sharpening: bool = True
    supports_hdr: bool = True
    min_scale: float = 1.0
    max_scale: float = 3.0
    requires_motion_vectors: bool = True
    requires_depth: bool = True
    supports_frame_generation: bool = False
    vendor: str = ""  # "NVIDIA", "AMD", "Intel", or "" for any


class QualityPreset(Enum):
    """Standard quality presets for upscaling."""

    ULTRA_PERFORMANCE = "ultra_performance"  # ~3x upscale
    PERFORMANCE = "performance"              # ~2x upscale
    BALANCED = "balanced"                    # ~1.7x upscale
    QUALITY = "quality"                      # ~1.5x upscale
    ULTRA_QUALITY = "ultra_quality"          # ~1.3x upscale
    NATIVE_AA = "native_aa"                  # 1x (AA only)


# ==============================================================================
# UPSCALER PLUGIN ABC
# ==============================================================================


class UpscalerPlugin(ABC):
    """Abstract base class for vendor upscaler plugins.

    This interface defines the contract that all upscaler implementations
    must follow, enabling unified handling of DLSS, FSR, XeSS, and future
    technologies.

    Lifecycle:
        1. Check is_available() to verify SDK/hardware support
        2. Call initialize() with resolution settings
        3. Call evaluate() each frame to perform upscaling
        4. Call shutdown() when done

    Example:
        >>> if DLSSPlugin.is_available():
        ...     plugin = DLSSPlugin()
        ...     plugin.initialize((1280, 720), (2560, 1440))
        ...     output = plugin.evaluate(color, depth, motion)
        ...     plugin.shutdown()
    """

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Check if this upscaler's SDK/DLL is available.

        This method should check for:
        - Required DLLs/shared libraries
        - Compatible GPU hardware
        - Required driver versions

        Returns:
            True if the upscaler can be used, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the upscaler.

        Returns:
            Name string (e.g., "NVIDIA DLSS", "AMD FSR 2.2").
        """
        pass

    @property
    @abstractmethod
    def capabilities(self) -> UpscalerCapabilities:
        """Return the upscaler's capabilities.

        Returns:
            UpscalerCapabilities dataclass with feature support.
        """
        pass

    @abstractmethod
    def initialize(
        self,
        input_resolution: Tuple[int, int],
        output_resolution: Tuple[int, int],
        quality_preset: str = "balanced"
    ) -> bool:
        """Initialize the upscaler context.

        Must be called before evaluate(). Can be called again to
        reconfigure for different resolutions.

        Args:
            input_resolution: (width, height) of input render target.
            output_resolution: (width, height) of output display.
            quality_preset: Quality mode ("ultra_performance", "performance",
                           "balanced", "quality", "ultra_quality").

        Returns:
            True on successful initialization, False on failure.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        color_input: TextureHandle,
        depth: Optional[TextureHandle] = None,
        motion_vectors: Optional[TextureHandle] = None,
        exposure: float = 1.0,
        sharpness: float = 0.5,
        reset: bool = False
    ) -> TextureHandle:
        """Execute upscaling for one frame.

        Args:
            color_input: Low-resolution color input.
            depth: Low-resolution depth buffer (optional for some upscalers).
            motion_vectors: Per-pixel motion vectors in screen space.
            exposure: Current exposure value for auto-exposure aware upscaling.
            sharpness: Post-upscale sharpening amount [0, 1].
            reset: True to reset temporal history (e.g., on camera cut).

        Returns:
            High-resolution upscaled output texture.

        Raises:
            NotImplementedError: If SDK is not available.
        """
        pass

    @abstractmethod
    def get_optimal_render_resolution(
        self,
        target_resolution: Tuple[int, int],
        quality_preset: str = "balanced"
    ) -> Tuple[int, int]:
        """Calculate optimal input resolution for target output.

        Different upscalers may have slightly different optimal scales
        for each quality preset. This method returns the recommended
        render resolution.

        Args:
            target_resolution: Desired output (width, height).
            quality_preset: Quality mode to use.

        Returns:
            Optimal render (width, height).
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release all resources.

        Must be called when the upscaler is no longer needed.
        After shutdown, initialize() must be called again before use.
        """
        pass

    def supports_preset(self, preset: str) -> bool:
        """Check if a quality preset is supported.

        Args:
            preset: Quality preset name.

        Returns:
            True if supported.
        """
        valid_presets = {
            "ultra_performance", "performance", "balanced",
            "quality", "ultra_quality", "native_aa"
        }
        return preset.lower() in valid_presets


# ==============================================================================
# DLSS PLUGIN STUB
# ==============================================================================


class DLSSPlugin(UpscalerPlugin):
    """NVIDIA DLSS implementation (NGX SDK wrapper).

    DLSS (Deep Learning Super Sampling) uses NVIDIA tensor cores
    for AI-based temporal upscaling. Requires:
    - NVIDIA RTX series GPU (Turing or newer)
    - nvngx_dlss.dll in system or application path
    - NVIDIA driver with NGX support

    Quality presets map to DLSS modes:
    - ultra_performance: DLSS Ultra Performance (3x)
    - performance: DLSS Performance (2x)
    - balanced: DLSS Balanced (1.7x)
    - quality: DLSS Quality (1.5x)
    - ultra_quality: DLSS Ultra Quality / DLAA (1.3x / 1x)
    """

    _ngx_initialized: bool = False
    _ngx_handle: Any = None
    _input_resolution: Tuple[int, int] = (0, 0)
    _output_resolution: Tuple[int, int] = (0, 0)
    _current_preset: str = "balanced"

    def __init__(self) -> None:
        """Initialize DLSS plugin (does not initialize NGX)."""
        self._ngx_initialized = False
        self._ngx_handle = None
        self._input_resolution = (0, 0)
        self._output_resolution = (0, 0)
        self._current_preset = "balanced"

    @staticmethod
    def is_available() -> bool:
        """Check for NGX SDK and compatible GPU.

        In a real implementation, this would:
        1. Check for nvngx_dlss.dll
        2. Query NVIDIA driver for RTX capability
        3. Verify tensor core availability

        Returns:
            False (stub - no actual SDK).
        """
        # Stub implementation - would check for NGX SDK
        # import ctypes
        # try:
        #     ngx_dll = ctypes.windll.LoadLibrary("nvngx_dlss.dll")
        #     return True
        # except OSError:
        #     return False
        return False

    @property
    def name(self) -> str:
        """Return DLSS name."""
        return "NVIDIA DLSS"

    @property
    def capabilities(self) -> UpscalerCapabilities:
        """Return DLSS capabilities."""
        return UpscalerCapabilities(
            name="NVIDIA DLSS",
            version="3.5",
            supports_sharpening=True,
            supports_hdr=True,
            min_scale=1.0,
            max_scale=3.0,
            requires_motion_vectors=True,
            requires_depth=True,
            supports_frame_generation=True,  # DLSS 3 Frame Generation
            vendor="NVIDIA",
        )

    def initialize(
        self,
        input_resolution: Tuple[int, int],
        output_resolution: Tuple[int, int],
        quality_preset: str = "balanced"
    ) -> bool:
        """Initialize NGX DLSS context.

        Args:
            input_resolution: Render resolution.
            output_resolution: Display resolution.
            quality_preset: DLSS quality mode.

        Returns:
            False (stub - NGX SDK not available).
        """
        if not self.is_available():
            return False

        self._input_resolution = input_resolution
        self._output_resolution = output_resolution
        self._current_preset = quality_preset

        # Would initialize NGX context here:
        # NVSDK_NGX_D3D12_Init(...)
        # NVSDK_NGX_D3D12_CreateFeature(NVSDK_NGX_Feature_SuperSampling, ...)

        self._ngx_initialized = True
        return True

    def evaluate(
        self,
        color_input: TextureHandle,
        depth: Optional[TextureHandle] = None,
        motion_vectors: Optional[TextureHandle] = None,
        exposure: float = 1.0,
        sharpness: float = 0.5,
        reset: bool = False
    ) -> TextureHandle:
        """Execute DLSS upscaling.

        Raises:
            NotImplementedError: DLSS SDK not available.
        """
        if not self._ngx_initialized:
            raise NotImplementedError("DLSS SDK not available - use fallback upscaler")

        # Would call NGX evaluate here:
        # NVSDK_NGX_D3D12_EvaluateFeature(...)

        return color_input  # Stub return

    def get_optimal_render_resolution(
        self,
        target_resolution: Tuple[int, int],
        quality_preset: str = "balanced"
    ) -> Tuple[int, int]:
        """Calculate optimal DLSS render resolution.

        DLSS quality presets use specific scale factors:
        - ultra_performance: 3.0x (33% render scale)
        - performance: 2.0x (50% render scale)
        - balanced: 1.7x (58% render scale)
        - quality: 1.5x (67% render scale)
        - ultra_quality: 1.3x (77% render scale)

        Args:
            target_resolution: Display resolution.
            quality_preset: DLSS quality mode.

        Returns:
            Recommended render resolution.
        """
        scale_factors = {
            "ultra_performance": 3.0,
            "performance": 2.0,
            "balanced": 1.7,
            "quality": 1.5,
            "ultra_quality": 1.3,
            "native_aa": 1.0,  # DLAA mode
        }
        scale = scale_factors.get(quality_preset.lower(), 1.7)
        return (
            max(1, int(target_resolution[0] / scale)),
            max(1, int(target_resolution[1] / scale)),
        )

    def shutdown(self) -> None:
        """Release NGX resources."""
        if self._ngx_initialized:
            # Would release NGX context here:
            # NVSDK_NGX_D3D12_ReleaseFeature(...)
            # NVSDK_NGX_D3D12_Shutdown()
            pass

        self._ngx_initialized = False
        self._ngx_handle = None


# ==============================================================================
# FSR PLUGIN STUB
# ==============================================================================


class FSRPlugin(UpscalerPlugin):
    """AMD FidelityFX Super Resolution 2.x implementation.

    FSR 2 is an open-source temporal upscaler that works on any GPU.
    Unlike FSR 1 (spatial only), FSR 2 uses temporal accumulation
    and motion vectors for higher quality.

    Requirements:
    - AMD FidelityFX SDK
    - Any GPU supporting compute shaders

    Quality presets:
    - ultra_performance: FSR 2 Ultra Performance (3x)
    - performance: FSR 2 Performance (2x)
    - balanced: FSR 2 Balanced (1.7x)
    - quality: FSR 2 Quality (1.5x)
    - ultra_quality: FSR 2 Ultra Quality (1.3x)
    """

    _fsr_context: Any = None
    _initialized: bool = False
    _input_resolution: Tuple[int, int] = (0, 0)
    _output_resolution: Tuple[int, int] = (0, 0)
    _current_preset: str = "balanced"

    def __init__(self) -> None:
        """Initialize FSR plugin."""
        self._fsr_context = None
        self._initialized = False
        self._input_resolution = (0, 0)
        self._output_resolution = (0, 0)
        self._current_preset = "balanced"

    @staticmethod
    def is_available() -> bool:
        """Check for FidelityFX SDK.

        In a real implementation, this would:
        1. Check for ffx_fsr2_api_*.dll
        2. Verify compute shader support

        Returns:
            False (stub - no actual SDK).
        """
        # Stub - would check for FidelityFX SDK presence
        return False

    @property
    def name(self) -> str:
        """Return FSR name."""
        return "AMD FSR 2"

    @property
    def capabilities(self) -> UpscalerCapabilities:
        """Return FSR 2 capabilities."""
        return UpscalerCapabilities(
            name="AMD FSR 2",
            version="2.2",
            supports_sharpening=True,
            supports_hdr=True,
            min_scale=1.0,
            max_scale=3.0,
            requires_motion_vectors=True,
            requires_depth=True,
            supports_frame_generation=False,  # FSR 3 has FG, FSR 2 does not
            vendor="AMD",
        )

    def initialize(
        self,
        input_resolution: Tuple[int, int],
        output_resolution: Tuple[int, int],
        quality_preset: str = "balanced"
    ) -> bool:
        """Initialize FSR 2 context.

        Args:
            input_resolution: Render resolution.
            output_resolution: Display resolution.
            quality_preset: FSR 2 quality mode.

        Returns:
            False (stub - SDK not available).
        """
        if not self.is_available():
            return False

        self._input_resolution = input_resolution
        self._output_resolution = output_resolution
        self._current_preset = quality_preset

        # Would initialize FSR 2 context:
        # ffxFsr2ContextCreate(...)

        self._initialized = True
        return True

    def evaluate(
        self,
        color_input: TextureHandle,
        depth: Optional[TextureHandle] = None,
        motion_vectors: Optional[TextureHandle] = None,
        exposure: float = 1.0,
        sharpness: float = 0.5,
        reset: bool = False
    ) -> TextureHandle:
        """Execute FSR 2 upscaling.

        Raises:
            NotImplementedError: FSR SDK not available.
        """
        if not self._initialized:
            raise NotImplementedError("FSR SDK not available - use fallback upscaler")

        # Would call FSR 2 dispatch:
        # ffxFsr2ContextDispatch(...)

        return color_input  # Stub return

    def get_optimal_render_resolution(
        self,
        target_resolution: Tuple[int, int],
        quality_preset: str = "balanced"
    ) -> Tuple[int, int]:
        """Calculate optimal FSR 2 render resolution.

        FSR 2 uses similar scale factors to DLSS:
        - ultra_performance: 3.0x
        - performance: 2.0x
        - balanced: 1.7x
        - quality: 1.5x
        - ultra_quality: 1.3x

        Args:
            target_resolution: Display resolution.
            quality_preset: FSR 2 quality mode.

        Returns:
            Recommended render resolution.
        """
        scale_factors = {
            "ultra_performance": 3.0,
            "performance": 2.0,
            "balanced": 1.7,
            "quality": 1.5,
            "ultra_quality": 1.3,
            "native_aa": 1.0,
        }
        scale = scale_factors.get(quality_preset.lower(), 1.7)
        return (
            max(1, int(target_resolution[0] / scale)),
            max(1, int(target_resolution[1] / scale)),
        )

    def shutdown(self) -> None:
        """Release FSR 2 resources."""
        if self._initialized:
            # Would destroy FSR 2 context:
            # ffxFsr2ContextDestroy(...)
            pass

        self._initialized = False
        self._fsr_context = None


# ==============================================================================
# XeSS PLUGIN STUB
# ==============================================================================


class XeSSPlugin(UpscalerPlugin):
    """Intel Xe Super Sampling implementation.

    XeSS uses Intel's XMX (Xe Matrix eXtensions) for AI-based upscaling
    on Intel Arc GPUs, with DP4a fallback for other hardware.

    Requirements:
    - Intel XeSS SDK
    - Intel Arc GPU (native) or any GPU with DP4a (fallback)

    Quality presets:
    - ultra_performance: XeSS Ultra Performance (3x)
    - performance: XeSS Performance (2x)
    - balanced: XeSS Balanced (1.7x)
    - quality: XeSS Quality (1.5x)
    - ultra_quality: XeSS Ultra Quality (1.3x)
    """

    _xess_context: Any = None
    _initialized: bool = False
    _use_dp4a_fallback: bool = False
    _input_resolution: Tuple[int, int] = (0, 0)
    _output_resolution: Tuple[int, int] = (0, 0)
    _current_preset: str = "balanced"

    def __init__(self) -> None:
        """Initialize XeSS plugin."""
        self._xess_context = None
        self._initialized = False
        self._use_dp4a_fallback = False
        self._input_resolution = (0, 0)
        self._output_resolution = (0, 0)
        self._current_preset = "balanced"

    @staticmethod
    def is_available() -> bool:
        """Check for XeSS SDK and DP4a support.

        In a real implementation, this would:
        1. Check for libxess.dll
        2. Query GPU for XMX or DP4a support

        Returns:
            False (stub - no actual SDK).
        """
        # Stub - would check for XeSS SDK
        return False

    @property
    def name(self) -> str:
        """Return XeSS name."""
        return "Intel XeSS"

    @property
    def capabilities(self) -> UpscalerCapabilities:
        """Return XeSS capabilities."""
        return UpscalerCapabilities(
            name="Intel XeSS",
            version="1.3",
            supports_sharpening=True,
            supports_hdr=True,
            min_scale=1.0,
            max_scale=3.0,
            requires_motion_vectors=True,
            requires_depth=True,
            supports_frame_generation=False,
            vendor="Intel",
        )

    @property
    def using_dp4a_fallback(self) -> bool:
        """Check if using DP4a fallback mode.

        Returns:
            True if using DP4a (non-Intel GPU), False if using native XMX.
        """
        return self._use_dp4a_fallback

    def initialize(
        self,
        input_resolution: Tuple[int, int],
        output_resolution: Tuple[int, int],
        quality_preset: str = "balanced"
    ) -> bool:
        """Initialize XeSS context.

        Args:
            input_resolution: Render resolution.
            output_resolution: Display resolution.
            quality_preset: XeSS quality mode.

        Returns:
            False (stub - SDK not available).
        """
        if not self.is_available():
            return False

        self._input_resolution = input_resolution
        self._output_resolution = output_resolution
        self._current_preset = quality_preset

        # Would initialize XeSS context:
        # xessD3D12CreateContext(...)

        self._initialized = True
        return True

    def evaluate(
        self,
        color_input: TextureHandle,
        depth: Optional[TextureHandle] = None,
        motion_vectors: Optional[TextureHandle] = None,
        exposure: float = 1.0,
        sharpness: float = 0.5,
        reset: bool = False
    ) -> TextureHandle:
        """Execute XeSS upscaling.

        Raises:
            NotImplementedError: XeSS SDK not available.
        """
        if not self._initialized:
            raise NotImplementedError("XeSS SDK not available - use fallback upscaler")

        # Would call XeSS execute:
        # xessD3D12Execute(...)

        return color_input  # Stub return

    def get_optimal_render_resolution(
        self,
        target_resolution: Tuple[int, int],
        quality_preset: str = "balanced"
    ) -> Tuple[int, int]:
        """Calculate optimal XeSS render resolution.

        XeSS quality presets:
        - ultra_performance: 3.0x
        - performance: 2.0x
        - balanced: 1.7x
        - quality: 1.5x
        - ultra_quality: 1.3x

        Args:
            target_resolution: Display resolution.
            quality_preset: XeSS quality mode.

        Returns:
            Recommended render resolution.
        """
        scale_factors = {
            "ultra_performance": 3.0,
            "performance": 2.0,
            "balanced": 1.7,
            "quality": 1.5,
            "ultra_quality": 1.3,
            "native_aa": 1.0,
        }
        scale = scale_factors.get(quality_preset.lower(), 1.7)
        return (
            max(1, int(target_resolution[0] / scale)),
            max(1, int(target_resolution[1] / scale)),
        )

    def shutdown(self) -> None:
        """Release XeSS resources."""
        if self._initialized:
            # Would destroy XeSS context:
            # xessDestroyContext(...)
            pass

        self._initialized = False
        self._xess_context = None


# ==============================================================================
# TSR LANCZOS PLUGIN ADAPTER
# ==============================================================================


class TSRLanczosPlugin(UpscalerPlugin):
    """TSR Lanczos adapter implementing UpscalerPlugin interface.

    This adapter wraps TSRLanczosUpscaler to conform to the plugin
    interface, enabling it to serve as the always-available fallback.
    """

    _upscaler: Optional[TSRLanczosUpscaler] = None
    _input_resolution: Tuple[int, int] = (0, 0)
    _output_resolution: Tuple[int, int] = (0, 0)

    def __init__(self) -> None:
        """Initialize TSR Lanczos plugin."""
        self._upscaler = None
        self._input_resolution = (0, 0)
        self._output_resolution = (0, 0)

    @staticmethod
    def is_available() -> bool:
        """TSR Lanczos is always available.

        Returns:
            Always True.
        """
        return True

    @property
    def name(self) -> str:
        """Return TSR Lanczos name."""
        return "TSR Lanczos"

    @property
    def capabilities(self) -> UpscalerCapabilities:
        """Return TSR Lanczos capabilities."""
        return UpscalerCapabilities(
            name="TSR Lanczos",
            version="1.0",
            supports_sharpening=True,
            supports_hdr=True,
            min_scale=1.0,
            max_scale=4.0,  # Lanczos can go higher
            requires_motion_vectors=False,  # Optional for temporal
            requires_depth=False,
            supports_frame_generation=False,
            vendor="",  # Works on any hardware
        )

    def initialize(
        self,
        input_resolution: Tuple[int, int],
        output_resolution: Tuple[int, int],
        quality_preset: str = "balanced"
    ) -> bool:
        """Initialize TSR Lanczos.

        Args:
            input_resolution: Render resolution.
            output_resolution: Display resolution.
            quality_preset: Quality mode (affects kernel size).

        Returns:
            Always True.
        """
        self._input_resolution = input_resolution
        self._output_resolution = output_resolution

        # Calculate scale factor
        if input_resolution[0] > 0:
            scale = output_resolution[0] / input_resolution[0]
        else:
            scale = 2.0

        # Use Lanczos3 for higher quality presets
        kernel = LanczosKernel.LANCZOS3 if quality_preset in (
            "quality", "ultra_quality", "native_aa"
        ) else LanczosKernel.LANCZOS2

        settings = TSRLanczosSettings(
            kernel=kernel,
            scale_factor=scale,
            sharpening=True,
            adaptive_sharpening=True,
        )

        self._upscaler = TSRLanczosUpscaler(settings)
        return True

    def evaluate(
        self,
        color_input: TextureHandle,
        depth: Optional[TextureHandle] = None,
        motion_vectors: Optional[TextureHandle] = None,
        exposure: float = 1.0,
        sharpness: float = 0.5,
        reset: bool = False
    ) -> TextureHandle:
        """Execute TSR Lanczos upscaling.

        Args:
            color_input: Low-resolution color.
            depth: Depth buffer (optional, improves quality if provided).
            motion_vectors: Motion vectors (optional, enables temporal).
            exposure: Exposure value.
            sharpness: Sharpening amount.
            reset: Reset temporal history.

        Returns:
            Upscaled output.
        """
        if self._upscaler is None:
            # Auto-initialize with defaults
            self.initialize(
                self._input_resolution or (1920, 1080),
                self._output_resolution or (3840, 2160)
            )

        if reset and self._upscaler:
            self._upscaler.reset()

        if self._upscaler:
            self._upscaler.settings.sharpness = sharpness
            self._upscaler.advance_frame()

        # In real implementation, would process color_input
        return color_input

    def get_optimal_render_resolution(
        self,
        target_resolution: Tuple[int, int],
        quality_preset: str = "balanced"
    ) -> Tuple[int, int]:
        """Calculate optimal TSR Lanczos render resolution.

        Args:
            target_resolution: Display resolution.
            quality_preset: Quality mode.

        Returns:
            Recommended render resolution.
        """
        scale_factors = {
            "ultra_performance": 3.0,
            "performance": 2.0,
            "balanced": 1.7,
            "quality": 1.5,
            "ultra_quality": 1.3,
            "native_aa": 1.0,
        }
        scale = scale_factors.get(quality_preset.lower(), 1.7)
        return (
            max(1, int(target_resolution[0] / scale)),
            max(1, int(target_resolution[1] / scale)),
        )

    def shutdown(self) -> None:
        """Release TSR Lanczos resources."""
        if self._upscaler:
            self._upscaler.reset()
        self._upscaler = None


# ==============================================================================
# UPSCALER MANAGER
# ==============================================================================


class UpscalerManager:
    """Manages upscaler plugin selection and fallback.

    Provides automatic detection of available upscalers and manages
    the fallback chain when preferred upscalers are unavailable.

    Priority order (configurable):
        1. NVIDIA DLSS (best on RTX GPUs)
        2. Intel XeSS (good on any GPU, best on Arc)
        3. AMD FSR 2 (good on any GPU)
        4. TSR Lanczos (always available fallback)

    Example:
        >>> manager = UpscalerManager()
        >>> upscaler = manager.detect_best_upscaler()
        >>> upscaler.initialize((1280, 720), (2560, 1440))
        >>> output = upscaler.evaluate(color, depth, motion)
    """

    # Priority order for auto-detection (highest to lowest)
    PRIORITY_ORDER: List[Type[UpscalerPlugin]] = [
        DLSSPlugin,
        XeSSPlugin,
        FSRPlugin,
        TSRLanczosPlugin,
    ]

    def __init__(self) -> None:
        """Initialize upscaler manager."""
        self._active_plugin: Optional[UpscalerPlugin] = None
        self._tsr_fallback: TSRLanczosPlugin = TSRLanczosPlugin()
        self._detected_plugins: Dict[str, UpscalerPlugin] = {}

    @property
    def active_plugin(self) -> Optional[UpscalerPlugin]:
        """Get currently active upscaler plugin."""
        return self._active_plugin

    def detect_best_upscaler(self) -> UpscalerPlugin:
        """Auto-detect best available upscaler.

        Checks each upscaler in priority order and returns the
        first one that is available.

        Returns:
            Best available upscaler plugin.
        """
        for plugin_class in self.PRIORITY_ORDER:
            if plugin_class.is_available():
                plugin = plugin_class()
                self._active_plugin = plugin
                return plugin

        # Fallback to TSR Lanczos (always available)
        self._active_plugin = self._tsr_fallback
        return self._tsr_fallback

    def get_available_upscalers(self) -> List[str]:
        """List all available upscaler names.

        Returns:
            List of available upscaler names.
        """
        available = []
        for plugin_class in self.PRIORITY_ORDER:
            if plugin_class.is_available():
                plugin = plugin_class()
                available.append(plugin.name)
        return available

    def select_upscaler(self, name: str) -> bool:
        """Manually select an upscaler by name.

        Args:
            name: Upscaler name to select (e.g., "NVIDIA DLSS").

        Returns:
            True if upscaler was selected, False if unavailable.
        """
        name_lower = name.lower()

        for plugin_class in self.PRIORITY_ORDER:
            plugin = plugin_class()
            if plugin.name.lower() == name_lower:
                if plugin_class.is_available():
                    self._active_plugin = plugin
                    return True
                return False

        return False

    def get_upscaler_by_name(self, name: str) -> Optional[UpscalerPlugin]:
        """Get upscaler plugin by name.

        Args:
            name: Upscaler name.

        Returns:
            Plugin instance or None if not found.
        """
        name_lower = name.lower()

        for plugin_class in self.PRIORITY_ORDER:
            plugin = plugin_class()
            if plugin.name.lower() == name_lower:
                return plugin

        return None

    def get_fallback(self) -> TSRLanczosPlugin:
        """Get the fallback upscaler (TSR Lanczos).

        Returns:
            TSR Lanczos plugin instance.
        """
        return self._tsr_fallback

    def shutdown_active(self) -> None:
        """Shutdown the currently active upscaler."""
        if self._active_plugin:
            self._active_plugin.shutdown()
            self._active_plugin = None

    def get_capabilities(self) -> Dict[str, UpscalerCapabilities]:
        """Get capabilities of all registered upscalers.

        Returns:
            Dict mapping upscaler names to their capabilities.
        """
        caps = {}
        for plugin_class in self.PRIORITY_ORDER:
            plugin = plugin_class()
            caps[plugin.name] = plugin.capabilities
        return caps


# ==============================================================================
# EXPORTS
# ==============================================================================


__all__ = [
    # Capabilities
    "UpscalerCapabilities",
    "QualityPreset",
    "TextureHandle",
    # Plugin ABC
    "UpscalerPlugin",
    # Plugin implementations
    "DLSSPlugin",
    "FSRPlugin",
    "XeSSPlugin",
    "TSRLanczosPlugin",
    # Manager
    "UpscalerManager",
]
