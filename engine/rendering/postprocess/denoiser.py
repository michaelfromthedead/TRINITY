"""
Denoiser System for Ray Tracing

Provides spatial denoising using A-Trous wavelet filtering:
- DenoiserQuality: Quality presets controlling iteration count
- DenoiserParams: Denoising configuration parameters
- GBuffer: Geometry buffer for edge-aware filtering
- Denoiser: Main denoiser dispatch class
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.platform.rhi.device import Device
    from engine.platform.rhi.resources import Texture


# =============================================================================
# Quality Presets
# =============================================================================


class DenoiserQuality(IntEnum):
    """Denoiser quality preset.

    Controls the number of A-Trous filter iterations.
    Higher iteration counts provide better denoising at increased cost.
    """

    LOW = 2       # 2 iterations - fast, some residual noise
    MEDIUM = 3    # 3 iterations - balanced quality/performance
    HIGH = 4      # 4 iterations - high quality, more expensive
    ULTRA = 5     # 5 iterations - maximum quality


# =============================================================================
# GBuffer
# =============================================================================


@dataclass
class GBuffer:
    """Geometry buffer for edge-aware denoising.

    Contains auxiliary textures used by the denoiser to preserve edges
    and prevent blurring across geometric discontinuities.

    Attributes:
        depth: Linear depth buffer for depth-based edge detection.
        normal: World-space normal buffer for normal-based edge detection.
        albedo: Optional surface albedo for modulated denoising.
    """

    depth: "Texture"
    normal: "Texture"
    albedo: Optional["Texture"] = None

    def is_valid(self) -> bool:
        """Check if G-Buffer has valid required textures.

        Returns:
            True if depth and normal textures are valid.
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
            True if albedo texture is present and valid.
        """
        return self.albedo is not None and self.albedo.is_valid()


# =============================================================================
# Denoiser Parameters
# =============================================================================


@dataclass
class DenoiserParams:
    """Denoiser configuration parameters.

    Controls the A-Trous wavelet filter behavior for edge-preserving
    denoising of ray-traced images.

    Attributes:
        quality: Quality preset controlling iteration count.
        sigma_color: Color similarity weight (higher = more blur).
        sigma_depth: Depth similarity weight (higher = more edge preservation).
        sigma_normal: Normal similarity weight (higher = more edge preservation).
    """

    quality: DenoiserQuality = DenoiserQuality.MEDIUM
    sigma_color: float = 1.0
    sigma_depth: float = 1.0
    sigma_normal: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not isinstance(self.quality, DenoiserQuality):
            raise TypeError(
                f"quality must be DenoiserQuality, got {type(self.quality).__name__}"
            )
        if self.sigma_color <= 0.0:
            raise ValueError(f"sigma_color must be positive, got {self.sigma_color}")
        if self.sigma_depth <= 0.0:
            raise ValueError(f"sigma_depth must be positive, got {self.sigma_depth}")
        if self.sigma_normal <= 0.0:
            raise ValueError(f"sigma_normal must be positive, got {self.sigma_normal}")

    def get_shader_params(self) -> Tuple[float, float, float]:
        """Get sigma values as tuple for shader binding.

        Returns:
            Tuple of (sigma_color, sigma_depth, sigma_normal).
        """
        return (self.sigma_color, self.sigma_depth, self.sigma_normal)


# =============================================================================
# Denoiser
# =============================================================================


class Denoiser:
    """A-Trous wavelet spatial denoiser for ray-traced images.

    Implements edge-aware spatial filtering using G-Buffer guidance.
    Uses ping-pong buffers for multi-pass iterative denoising.

    Example:
        denoiser = Denoiser(device)
        params = DenoiserParams(quality=DenoiserQuality.HIGH)
        denoiser.spatial_denoise(noisy_image, g_buffer, output, params)
    """

    def __init__(self, device: "Device") -> None:
        """Initialize the denoiser.

        Args:
            device: RHI device for resource creation.
        """
        self._device = device
        self._ping_buffer: Optional["Texture"] = None
        self._pong_buffer: Optional["Texture"] = None
        self._initialized = False

    @property
    def device(self) -> "Device":
        """Get the RHI device.

        Returns:
            The device used for resource creation.
        """
        return self._device

    @property
    def is_initialized(self) -> bool:
        """Check if denoiser has been initialized with buffers.

        Returns:
            True if ping-pong buffers have been created.
        """
        return self._initialized

    def get_iteration_count(self, quality: DenoiserQuality) -> int:
        """Return iteration count for quality level.

        The iteration count determines how many A-Trous filter passes
        are executed. Each pass uses a larger step size (2^iteration).

        Args:
            quality: Quality preset to query.

        Returns:
            Number of filter iterations for the given quality.
        """
        return int(quality)

    def create_ping_pong_buffers(
        self, width: int, height: int
    ) -> Tuple["Texture", "Texture"]:
        """Create temporary buffers for multi-pass denoising.

        Creates two textures of equal size for ping-pong rendering.
        The buffers are cached and reused if dimensions match.

        Args:
            width: Buffer width in pixels.
            height: Buffer height in pixels.

        Returns:
            Tuple of (ping_buffer, pong_buffer) textures.

        Raises:
            ValueError: If width or height is not positive.
        """
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")

        # Import here to avoid circular imports
        from engine.platform.rhi.resources import (
            Format,
            TextureDesc,
            TextureType,
            TextureUsage,
        )

        # Check if we can reuse existing buffers
        if self._ping_buffer is not None and self._pong_buffer is not None:
            ping_desc = self._ping_buffer.desc
            if ping_desc.width == width and ping_desc.height == height:
                return (self._ping_buffer, self._pong_buffer)

        # Create new buffers
        desc = TextureDesc(
            type=TextureType.TEXTURE_2D,
            format=Format.RGBA16_FLOAT,
            width=width,
            height=height,
            usage=TextureUsage.SHADER_RESOURCE | TextureUsage.UNORDERED_ACCESS,
        )

        self._ping_buffer = self._device.create_texture(desc)
        self._pong_buffer = self._device.create_texture(desc)
        self._initialized = True

        return (self._ping_buffer, self._pong_buffer)

    def spatial_denoise(
        self,
        noisy_input: "Texture",
        g_buffer: GBuffer,
        output: "Texture",
        params: Optional[DenoiserParams] = None,
    ) -> None:
        """Dispatch spatial denoising passes.

        Performs multi-pass A-Trous wavelet filtering guided by G-Buffer
        data for edge preservation. Uses ping-pong rendering for iteration.

        Args:
            noisy_input: Input texture with noisy ray-traced result.
            g_buffer: Geometry buffer with depth/normal/albedo.
            output: Output texture for denoised result.
            params: Denoising parameters. Uses defaults if None.

        Raises:
            ValueError: If input textures are invalid or dimensions mismatch.
        """
        if params is None:
            params = DenoiserParams()

        # Validate inputs
        if noisy_input is None or not noisy_input.is_valid():
            raise ValueError("noisy_input texture is invalid")
        if output is None or not output.is_valid():
            raise ValueError("output texture is invalid")
        if not g_buffer.is_valid():
            raise ValueError("g_buffer is invalid (missing depth or normal)")

        # Get dimensions from input
        input_desc = noisy_input.desc
        width = input_desc.width
        height = input_desc.height

        # Ensure output dimensions match
        output_desc = output.desc
        if output_desc.width != width or output_desc.height != height:
            raise ValueError(
                f"Output dimensions ({output_desc.width}x{output_desc.height}) "
                f"do not match input ({width}x{height})"
            )

        # Create or retrieve ping-pong buffers
        ping, pong = self.create_ping_pong_buffers(width, height)

        # Get iteration count from quality
        iterations = self.get_iteration_count(params.quality)

        # Dispatch passes
        self._dispatch_passes(
            noisy_input=noisy_input,
            g_buffer=g_buffer,
            output=output,
            ping=ping,
            pong=pong,
            params=params,
            iterations=iterations,
        )

    def _dispatch_passes(
        self,
        noisy_input: "Texture",
        g_buffer: GBuffer,
        output: "Texture",
        ping: "Texture",
        pong: "Texture",
        params: DenoiserParams,
        iterations: int,
    ) -> None:
        """Internal dispatch of A-Trous filter passes.

        Args:
            noisy_input: Input texture.
            g_buffer: Geometry buffer.
            output: Final output texture.
            ping: Ping buffer for ping-pong.
            pong: Pong buffer for ping-pong.
            params: Filter parameters.
            iterations: Number of passes to run.
        """
        # Stub implementation: In real implementation, would:
        # 1. Bind compute pipeline for A-Trous filter
        # 2. For each iteration:
        #    a. Set step size = 2^iteration
        #    b. Bind source (input or previous ping/pong)
        #    c. Bind destination (ping or pong, alternating)
        #    d. Bind G-Buffer for edge weights
        #    e. Set sigma parameters
        #    f. Dispatch compute shader
        # 3. Copy final result to output

        # Track current source/destination for ping-pong
        current_src = noisy_input
        use_ping = True

        for i in range(iterations):
            step_size = 1 << i  # 2^i

            # Choose destination buffer
            current_dst = ping if use_ping else pong

            # In real impl: dispatch compute shader here
            _ = (current_src, current_dst, step_size, params.get_shader_params())

            # Swap for next iteration
            current_src = current_dst
            use_ping = not use_ping

        # Final result is in current_src
        # In real impl: copy current_src to output
        _ = (current_src, output)

    def destroy(self) -> None:
        """Release denoiser resources.

        Destroys ping-pong buffers if they exist.
        """
        if self._ping_buffer is not None:
            self._ping_buffer.destroy()
            self._ping_buffer = None
        if self._pong_buffer is not None:
            self._pong_buffer.destroy()
            self._pong_buffer = None
        self._initialized = False

    def __del__(self) -> None:
        """Clean up on deletion."""
        self.destroy()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_default_params() -> DenoiserParams:
    """Create default denoiser parameters.

    Returns:
        DenoiserParams with balanced defaults.
    """
    return DenoiserParams()


def create_quality_params(quality: DenoiserQuality) -> DenoiserParams:
    """Create parameters for a specific quality level.

    Args:
        quality: Desired quality level.

    Returns:
        DenoiserParams configured for the given quality.
    """
    # Adjust sigmas based on quality for optimal results
    sigma_scale = {
        DenoiserQuality.LOW: 1.2,     # More aggressive blur for fewer passes
        DenoiserQuality.MEDIUM: 1.0,  # Balanced
        DenoiserQuality.HIGH: 0.9,    # Slightly tighter for more passes
        DenoiserQuality.ULTRA: 0.8,   # Tightest for maximum passes
    }

    scale = sigma_scale.get(quality, 1.0)

    return DenoiserParams(
        quality=quality,
        sigma_color=scale,
        sigma_depth=scale,
        sigma_normal=scale,
    )
