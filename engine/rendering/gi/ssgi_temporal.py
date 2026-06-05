"""
SSGI Temporal Accumulation (T-GIR-P3.2).

Provides temporal accumulation for Screen-Space Global Illumination to reduce
noise and improve stability. Uses motion vector reprojection, neighbourhood
clamping, and disocclusion detection.

Key Features:
    - Reprojection via velocity buffer for motion compensation
    - Neighbourhood clamping (AABB/variance) to reject ghosting
    - Disocclusion reset when velocity magnitude exceeds threshold
    - Configurable accumulation blend factor
    - Exponential moving average variance tracking

Architecture:
    - TemporalConfig: Configuration dataclass for all temporal parameters
    - TemporalHistory: Manages ping-pong history buffers
    - SSGITemporalAccumulator: Main accumulation dispatch class
    - NeighbourhoodStats: Per-pixel neighbourhood statistics

References:
    - "Temporal Reprojection Anti-Aliasing in INSIDE" (GDC 2016)
    - "High-Quality Temporal Supersampling" (Karis, SIGGRAPH 2014)
    - "Temporal AA and the quest for the Holy Trail" (Wihlidal, GDC 2017)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.platform.rhi.device import Device
    from engine.platform.rhi.resources import Texture


# =============================================================================
# Constants
# =============================================================================

# Default blend factor for temporal accumulation (higher = more history)
DEFAULT_BLEND_FACTOR: float = 0.9

# Default velocity threshold for disocclusion detection (in NDC units/frame)
DEFAULT_VELOCITY_THRESHOLD: float = 0.01

# Default neighbourhood clamp expand factor (reduces clamping strictness)
DEFAULT_CLAMP_EXPAND: float = 1.25

# Minimum variance before clamping is applied
DEFAULT_MIN_VARIANCE: float = 0.0001

# Maximum frames for variance calculation
MAX_VARIANCE_FRAMES: int = 32

# Workgroup size for temporal accumulation compute shader
WORKGROUP_SIZE: int = 8


# =============================================================================
# Enums
# =============================================================================


class ClampMode(IntEnum):
    """Neighbourhood clamping mode for ghosting rejection.

    Controls how the reprojected history is constrained to the current
    frame's neighbourhood to prevent ghosting artifacts.
    """

    NONE = 0       # No clamping (fastest, most ghosting)
    AABB = 1       # AABB clamp to min/max of 3x3 neighbourhood
    VARIANCE = 2   # Variance-based AABB (tighter, less ghosting)
    CLIPPED = 3    # Clip to convex hull (most accurate, expensive)


class ResetCondition(IntEnum):
    """Conditions that trigger history reset.

    Determines when the temporal history should be invalidated
    and reset to the current frame.
    """

    NEVER = 0            # Never auto-reset (manual only)
    VELOCITY = 1         # Reset on high velocity (disocclusion)
    DEPTH_DISCONTINUITY = 2  # Reset on depth edges
    BOTH = 3             # Reset on velocity OR depth discontinuity


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TemporalConfig:
    """Configuration for SSGI temporal accumulation.

    Controls the behavior of temporal reprojection, neighbourhood clamping,
    and disocclusion handling for SSGI.

    Attributes:
        enabled: Whether temporal accumulation is enabled.
        blend_factor: Weight for history blend [0.0, 1.0]. Higher = more history.
        velocity_threshold: Velocity magnitude (NDC/frame) that triggers reset.
        clamp_mode: Neighbourhood clamping strategy.
        clamp_expand: Factor to expand clamping bounds (reduces aggressiveness).
        reset_condition: When to reset history automatically.
        depth_threshold: Depth difference threshold for discontinuity detection.
        variance_gamma: Exponential decay for variance tracking.
        min_history_weight: Minimum history weight even at high velocity.
        max_history_weight: Maximum history weight for static scenes.
        anti_flicker: Enable anti-flicker filtering.
        luminance_weight: Weight blending by luminance difference.
    """

    enabled: bool = True
    blend_factor: float = DEFAULT_BLEND_FACTOR
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD
    clamp_mode: ClampMode = ClampMode.VARIANCE
    clamp_expand: float = DEFAULT_CLAMP_EXPAND
    reset_condition: ResetCondition = ResetCondition.VELOCITY
    depth_threshold: float = 0.05
    variance_gamma: float = 0.1
    min_history_weight: float = 0.5
    max_history_weight: float = 0.98
    anti_flicker: bool = True
    luminance_weight: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.blend_factor <= 1.0:
            raise ValueError(
                f"blend_factor must be in [0.0, 1.0], got {self.blend_factor}"
            )
        if self.velocity_threshold < 0.0:
            raise ValueError(
                f"velocity_threshold must be >= 0, got {self.velocity_threshold}"
            )
        if self.clamp_expand < 1.0:
            raise ValueError(
                f"clamp_expand must be >= 1.0, got {self.clamp_expand}"
            )
        if self.depth_threshold <= 0.0:
            raise ValueError(
                f"depth_threshold must be > 0, got {self.depth_threshold}"
            )
        if not 0.0 < self.variance_gamma <= 1.0:
            raise ValueError(
                f"variance_gamma must be in (0.0, 1.0], got {self.variance_gamma}"
            )
        if not 0.0 <= self.min_history_weight <= 1.0:
            raise ValueError(
                f"min_history_weight must be in [0.0, 1.0], got {self.min_history_weight}"
            )
        if not 0.0 <= self.max_history_weight <= 1.0:
            raise ValueError(
                f"max_history_weight must be in [0.0, 1.0], got {self.max_history_weight}"
            )
        if self.min_history_weight > self.max_history_weight:
            raise ValueError(
                f"min_history_weight ({self.min_history_weight}) must be <= "
                f"max_history_weight ({self.max_history_weight})"
            )

    def get_adaptive_blend(self, velocity_magnitude: float) -> float:
        """Calculate adaptive blend factor based on velocity.

        Reduces history weight for fast-moving pixels to reduce ghosting
        while maintaining high weight for static pixels.

        Args:
            velocity_magnitude: Motion vector magnitude in NDC units.

        Returns:
            Adaptive blend factor [min_history_weight, max_history_weight].
        """
        if velocity_magnitude <= 0.0:
            return self.max_history_weight

        # Normalize velocity by threshold
        t = min(velocity_magnitude / self.velocity_threshold, 1.0)

        # Smooth interpolation from max to min weight
        t = t * t * (3.0 - 2.0 * t)  # smoothstep

        return self.max_history_weight - t * (
            self.max_history_weight - self.min_history_weight
        )

    def should_reset(
        self,
        velocity_magnitude: float,
        depth_diff: float,
    ) -> bool:
        """Determine if history should be reset for a pixel.

        Args:
            velocity_magnitude: Motion vector magnitude in NDC units.
            depth_diff: Absolute depth difference from reprojected position.

        Returns:
            True if history should be reset.
        """
        if self.reset_condition == ResetCondition.NEVER:
            return False
        elif self.reset_condition == ResetCondition.VELOCITY:
            return velocity_magnitude > self.velocity_threshold
        elif self.reset_condition == ResetCondition.DEPTH_DISCONTINUITY:
            return depth_diff > self.depth_threshold
        elif self.reset_condition == ResetCondition.BOTH:
            return (
                velocity_magnitude > self.velocity_threshold
                or depth_diff > self.depth_threshold
            )
        return False

    def with_blend_factor(self, blend_factor: float) -> "TemporalConfig":
        """Create a copy with modified blend factor.

        Args:
            blend_factor: New blend factor value.

        Returns:
            New TemporalConfig with updated blend factor.
        """
        return TemporalConfig(
            enabled=self.enabled,
            blend_factor=blend_factor,
            velocity_threshold=self.velocity_threshold,
            clamp_mode=self.clamp_mode,
            clamp_expand=self.clamp_expand,
            reset_condition=self.reset_condition,
            depth_threshold=self.depth_threshold,
            variance_gamma=self.variance_gamma,
            min_history_weight=self.min_history_weight,
            max_history_weight=self.max_history_weight,
            anti_flicker=self.anti_flicker,
            luminance_weight=self.luminance_weight,
        )

    def with_clamp_mode(self, clamp_mode: ClampMode) -> "TemporalConfig":
        """Create a copy with modified clamp mode.

        Args:
            clamp_mode: New clamping mode.

        Returns:
            New TemporalConfig with updated clamp mode.
        """
        return TemporalConfig(
            enabled=self.enabled,
            blend_factor=self.blend_factor,
            velocity_threshold=self.velocity_threshold,
            clamp_mode=clamp_mode,
            clamp_expand=self.clamp_expand,
            reset_condition=self.reset_condition,
            depth_threshold=self.depth_threshold,
            variance_gamma=self.variance_gamma,
            min_history_weight=self.min_history_weight,
            max_history_weight=self.max_history_weight,
            anti_flicker=self.anti_flicker,
            luminance_weight=self.luminance_weight,
        )


# =============================================================================
# Neighbourhood Statistics
# =============================================================================


@dataclass
class NeighbourhoodStats:
    """Statistics computed from a pixel's neighbourhood.

    Used for variance-based clamping and ghosting rejection.

    Attributes:
        mean: Mean color of the neighbourhood.
        variance: Variance of the neighbourhood colors.
        min_val: Minimum color in neighbourhood.
        max_val: Maximum color in neighbourhood.
        sample_count: Number of valid samples in neighbourhood.
    """

    mean: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    variance: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    min_val: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    max_val: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    sample_count: int = 0

    def aabb_min(self) -> Tuple[float, float, float, float]:
        """Get the AABB minimum for clamping.

        Returns:
            Minimum color bounds (RGBA).
        """
        return self.min_val

    def aabb_max(self) -> Tuple[float, float, float, float]:
        """Get the AABB maximum for clamping.

        Returns:
            Maximum color bounds (RGBA).
        """
        return self.max_val

    def variance_min(self, expand: float = 1.25) -> Tuple[float, float, float, float]:
        """Get variance-based minimum for clamping.

        Args:
            expand: Factor to expand bounds beyond variance.

        Returns:
            Lower bound based on mean - expand * sqrt(variance).
        """
        return tuple(
            self.mean[i] - expand * math.sqrt(max(self.variance[i], 0.0))
            for i in range(4)
        )

    def variance_max(self, expand: float = 1.25) -> Tuple[float, float, float, float]:
        """Get variance-based maximum for clamping.

        Args:
            expand: Factor to expand bounds beyond variance.

        Returns:
            Upper bound based on mean + expand * sqrt(variance).
        """
        return tuple(
            self.mean[i] + expand * math.sqrt(max(self.variance[i], 0.0))
            for i in range(4)
        )


def compute_neighbourhood_stats(
    samples: List[Tuple[float, float, float, float]],
) -> NeighbourhoodStats:
    """Compute neighbourhood statistics from color samples.

    Args:
        samples: List of RGBA color tuples from the neighbourhood.

    Returns:
        NeighbourhoodStats with computed mean, variance, min, max.
    """
    if not samples:
        return NeighbourhoodStats()

    n = len(samples)

    # Compute min/max
    min_val = list(samples[0])
    max_val = list(samples[0])
    for sample in samples[1:]:
        for i in range(4):
            min_val[i] = min(min_val[i], sample[i])
            max_val[i] = max(max_val[i], sample[i])

    # Compute mean
    mean = [0.0, 0.0, 0.0, 0.0]
    for sample in samples:
        for i in range(4):
            mean[i] += sample[i]
    mean = [m / n for m in mean]

    # Compute variance
    variance = [0.0, 0.0, 0.0, 0.0]
    for sample in samples:
        for i in range(4):
            diff = sample[i] - mean[i]
            variance[i] += diff * diff
    variance = [v / n for v in variance]

    return NeighbourhoodStats(
        mean=tuple(mean),
        variance=tuple(variance),
        min_val=tuple(min_val),
        max_val=tuple(max_val),
        sample_count=n,
    )


# =============================================================================
# Clamping Functions
# =============================================================================


def clamp_color(
    color: Tuple[float, float, float, float],
    min_val: Tuple[float, float, float, float],
    max_val: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Clamp a color to the given bounds.

    Args:
        color: Input RGBA color.
        min_val: Minimum bounds.
        max_val: Maximum bounds.

    Returns:
        Clamped color.
    """
    return tuple(max(min_val[i], min(max_val[i], color[i])) for i in range(4))


def clip_color_to_aabb(
    color: Tuple[float, float, float, float],
    center: Tuple[float, float, float, float],
    half_extent: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Clip color towards center of AABB.

    Clips the color along the ray from center to color, finding the
    point where it exits the AABB. This produces better results than
    simple clamping for colors outside the bounds.

    Args:
        color: Input RGBA color to clip.
        center: Center of the AABB (typically neighbourhood mean).
        half_extent: Half-size of the AABB in each dimension.

    Returns:
        Clipped color on the AABB surface or original if inside.
    """
    # Direction from center to color
    direction = tuple(color[i] - center[i] for i in range(4))

    # Find the t value where we exit the AABB
    t_max = 1.0
    for i in range(4):
        if abs(direction[i]) > 1e-6:
            t_pos = half_extent[i] / abs(direction[i])
            t_max = min(t_max, t_pos)

    # Clip to t_max if needed
    if t_max < 1.0:
        return tuple(center[i] + direction[i] * t_max for i in range(4))
    return color


# =============================================================================
# Reprojection
# =============================================================================


def reproject_uv(
    uv: Tuple[float, float],
    velocity: Tuple[float, float],
) -> Tuple[float, float]:
    """Reproject UV coordinates using velocity.

    Args:
        uv: Current frame UV coordinates [0, 1].
        velocity: Motion vector (velocity buffer value).

    Returns:
        Reprojected UV coordinates for history lookup.
    """
    return (uv[0] - velocity[0], uv[1] - velocity[1])


def velocity_magnitude(velocity: Tuple[float, float]) -> float:
    """Compute the magnitude of a velocity vector.

    Args:
        velocity: Motion vector (vx, vy).

    Returns:
        Magnitude in NDC units.
    """
    return math.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1])


def is_valid_uv(uv: Tuple[float, float]) -> bool:
    """Check if UV coordinates are within valid range.

    Args:
        uv: UV coordinates to check.

    Returns:
        True if UV is in [0, 1] range.
    """
    return 0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0


# =============================================================================
# Temporal History Buffer Management
# =============================================================================


@dataclass
class TemporalHistory:
    """Manages temporal history buffers for SSGI accumulation.

    Implements ping-pong buffering for temporal accumulation with
    support for multiple history frames for variance calculation.

    Attributes:
        width: Buffer width in pixels.
        height: Buffer height in pixels.
        frame_count: Number of accumulated frames.
        is_valid: Whether history contains valid data.
    """

    width: int = 0
    height: int = 0
    frame_count: int = 0
    is_valid: bool = False

    # Internal buffers (texture references)
    _current: Optional["Texture"] = field(default=None, repr=False)
    _history: Optional["Texture"] = field(default=None, repr=False)
    _variance: Optional["Texture"] = field(default=None, repr=False)

    def create_buffers(
        self,
        device: "Device",
        width: int,
        height: int,
    ) -> None:
        """Create or resize history buffers.

        Args:
            device: RHI device for texture creation.
            width: Buffer width.
            height: Buffer height.
        """
        # Check if resize is needed
        if width == self.width and height == self.height and self._current is not None:
            return

        # Release old buffers
        self.destroy()

        # Import here to avoid circular imports
        from engine.platform.rhi.resources import (
            Format,
            TextureDesc,
            TextureType,
            TextureUsage,
        )

        self.width = width
        self.height = height

        # Create RGBA16F buffers for color history
        desc = TextureDesc(
            type=TextureType.TEXTURE_2D,
            format=Format.RGBA16_FLOAT,
            width=width,
            height=height,
            usage=TextureUsage.SHADER_RESOURCE | TextureUsage.UNORDERED_ACCESS,
        )

        self._current = device.create_texture(desc)
        self._history = device.create_texture(desc)

        # Create RG16F buffer for variance (mean luminance, variance)
        variance_desc = TextureDesc(
            type=TextureType.TEXTURE_2D,
            format=Format.RG16_FLOAT,
            width=width,
            height=height,
            usage=TextureUsage.SHADER_RESOURCE | TextureUsage.UNORDERED_ACCESS,
        )
        self._variance = device.create_texture(variance_desc)

        self.is_valid = False
        self.frame_count = 0

    def swap_buffers(self) -> None:
        """Swap current and history buffers (ping-pong)."""
        self._current, self._history = self._history, self._current
        self.frame_count += 1

    def invalidate(self) -> None:
        """Mark history as invalid (e.g., on camera cut)."""
        self.is_valid = False
        self.frame_count = 0

    def destroy(self) -> None:
        """Release buffer resources."""
        if self._current is not None:
            self._current.destroy()
            self._current = None
        if self._history is not None:
            self._history.destroy()
            self._history = None
        if self._variance is not None:
            self._variance.destroy()
            self._variance = None
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.is_valid = False

    @property
    def current_buffer(self) -> Optional["Texture"]:
        """Get the current output buffer."""
        return self._current

    @property
    def history_buffer(self) -> Optional["Texture"]:
        """Get the history input buffer."""
        return self._history

    @property
    def variance_buffer(self) -> Optional["Texture"]:
        """Get the variance tracking buffer."""
        return self._variance

    def get_convergence_ratio(self) -> float:
        """Get temporal convergence ratio based on frame count.

        Returns:
            Convergence ratio [0, 1] based on accumulated frames.
        """
        if not self.is_valid or self.frame_count == 0:
            return 0.0
        # Exponential convergence curve
        return 1.0 - math.exp(-self.frame_count / 8.0)


# =============================================================================
# GPU Uniform Structures
# =============================================================================


@dataclass
class TemporalUniforms:
    """GPU-side temporal accumulation uniforms.

    Matches the WGSL TemporalConfig struct layout for compute shader.

    Attributes:
        blend_factor: History blend weight.
        velocity_threshold: Disocclusion threshold.
        clamp_mode: Clamping strategy (0-3).
        clamp_expand: AABB expansion factor.
        depth_threshold: Depth discontinuity threshold.
        variance_gamma: Variance decay factor.
        min_weight: Minimum history weight.
        max_weight: Maximum history weight.
        frame_index: Current frame number.
        flags: Bitfield (bit 0: anti_flicker, bit 1: luminance_weight).
    """

    blend_factor: float = DEFAULT_BLEND_FACTOR
    velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD
    clamp_mode: int = ClampMode.VARIANCE.value
    clamp_expand: float = DEFAULT_CLAMP_EXPAND
    depth_threshold: float = 0.05
    variance_gamma: float = 0.1
    min_weight: float = 0.5
    max_weight: float = 0.98
    frame_index: int = 0
    flags: int = 0  # bit 0: anti_flicker, bit 1: luminance_weight
    _pad: Tuple[int, int] = (0, 0)  # Padding for 48-byte alignment

    @classmethod
    def from_config(cls, config: TemporalConfig, frame_index: int) -> "TemporalUniforms":
        """Create uniforms from a TemporalConfig.

        Args:
            config: Configuration to convert.
            frame_index: Current frame number.

        Returns:
            TemporalUniforms matching the config.
        """
        flags = 0
        if config.anti_flicker:
            flags |= 1
        if config.luminance_weight:
            flags |= 2

        return cls(
            blend_factor=config.blend_factor,
            velocity_threshold=config.velocity_threshold,
            clamp_mode=config.clamp_mode.value,
            clamp_expand=config.clamp_expand,
            depth_threshold=config.depth_threshold,
            variance_gamma=config.variance_gamma,
            min_weight=config.min_history_weight,
            max_weight=config.max_history_weight,
            frame_index=frame_index,
            flags=flags,
        )

    def to_bytes(self) -> bytes:
        """Convert to byte buffer for GPU upload.

        Returns:
            48 bytes of packed uniform data.
        """
        import struct

        return struct.pack(
            "<ffffffff II II",
            self.blend_factor,
            self.velocity_threshold,
            float(self.clamp_mode),  # Stored as float for WGSL compat
            self.clamp_expand,
            self.depth_threshold,
            self.variance_gamma,
            self.min_weight,
            self.max_weight,
            self.frame_index,
            self.flags,
            self._pad[0],
            self._pad[1],
        )


# =============================================================================
# SSGI Temporal Accumulator
# =============================================================================


class SSGITemporalAccumulator:
    """SSGI temporal accumulation processor.

    Manages temporal reprojection and accumulation for SSGI to reduce
    noise and improve stability over multiple frames.

    Example:
        accumulator = SSGITemporalAccumulator(device)
        config = TemporalConfig(blend_factor=0.9)
        accumulator.setup(1920, 1080)
        accumulator.accumulate(
            ssgi_input,
            velocity_buffer,
            depth_buffer,
            output,
            config,
        )
    """

    def __init__(self, device: Optional["Device"] = None) -> None:
        """Initialize the temporal accumulator.

        Args:
            device: Optional RHI device for resource creation.
        """
        self._device = device
        self._history: TemporalHistory = TemporalHistory()
        self._config: TemporalConfig = TemporalConfig()
        self._frame_index: int = 0
        self._initialized: bool = False

    @property
    def device(self) -> Optional["Device"]:
        """Get the RHI device."""
        return self._device

    @property
    def config(self) -> TemporalConfig:
        """Get the current configuration."""
        return self._config

    @config.setter
    def config(self, value: TemporalConfig) -> None:
        """Set the configuration."""
        self._config = value

    @property
    def history(self) -> TemporalHistory:
        """Get the temporal history manager."""
        return self._history

    @property
    def frame_index(self) -> int:
        """Get the current frame index."""
        return self._frame_index

    @property
    def is_initialized(self) -> bool:
        """Check if accumulator is initialized."""
        return self._initialized

    @property
    def is_history_valid(self) -> bool:
        """Check if history buffer contains valid data."""
        return self._history.is_valid

    def setup(self, width: int, height: int) -> None:
        """Initialize or resize accumulator resources.

        Args:
            width: Render width in pixels.
            height: Render height in pixels.
        """
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"height must be positive, got {height}")

        # Check if resize is needed
        needs_resize = width != self._history.width or height != self._history.height

        if needs_resize and self._device is not None:
            self._history.create_buffers(self._device, width, height)

        self._initialized = True

    def reset_history(self) -> None:
        """Reset temporal history (call on camera cuts, scene changes)."""
        self._history.invalidate()
        self._frame_index = 0

    def accumulate(
        self,
        ssgi_input: "Texture",
        velocity_buffer: "Texture",
        depth_buffer: "Texture",
        output: "Texture",
        config: Optional[TemporalConfig] = None,
    ) -> None:
        """Perform temporal accumulation.

        Args:
            ssgi_input: Current frame SSGI buffer.
            velocity_buffer: Motion vectors from TAA/motion pass.
            depth_buffer: Scene depth buffer.
            output: Output accumulated SSGI buffer.
            config: Optional config override.
        """
        if config is not None:
            self._config = config

        if not self._config.enabled:
            # Temporal disabled - just copy input to output
            self._copy_texture(ssgi_input, output)
            return

        # First frame - initialize history
        if not self._history.is_valid:
            self._copy_texture(ssgi_input, output)
            self._copy_texture(ssgi_input, self._history._current)
            self._history.is_valid = True
            self._frame_index += 1
            return

        # Dispatch temporal accumulation
        self._dispatch_accumulate(
            ssgi_input,
            velocity_buffer,
            depth_buffer,
            output,
        )

        # Swap history buffers
        self._history.swap_buffers()
        self._frame_index += 1

    def get_convergence(self) -> float:
        """Get current temporal convergence ratio.

        Returns:
            Convergence ratio [0, 1] based on accumulated frames.
        """
        return self._history.get_convergence_ratio()

    def get_variance_reduction(self, frames: int = 8) -> float:
        """Estimate variance reduction factor for given frame count.

        The variance of temporal accumulation decreases as 1/N where N
        is the effective number of accumulated samples.

        Args:
            frames: Number of frames for estimation.

        Returns:
            Estimated variance reduction factor.
        """
        if frames <= 0:
            return 1.0

        # Account for blend factor - effective samples is less than frames
        effective_samples = 0.0
        weight = 1.0
        for _ in range(frames):
            effective_samples += weight
            weight *= self._config.blend_factor

        if effective_samples <= 0:
            return 1.0

        return 1.0 / effective_samples

    def _dispatch_accumulate(
        self,
        ssgi_input: "Texture",
        velocity_buffer: "Texture",
        depth_buffer: "Texture",
        output: "Texture",
    ) -> None:
        """Internal dispatch of temporal accumulation compute shader.

        Args:
            ssgi_input: Current SSGI buffer.
            velocity_buffer: Motion vectors.
            depth_buffer: Scene depth.
            output: Output buffer.
        """
        # In real implementation: dispatch compute shader
        # This is a stub that would:
        # 1. Bind ssgi_input, velocity_buffer, depth_buffer, history as inputs
        # 2. Bind output as UAV
        # 3. Upload TemporalUniforms to constant buffer
        # 4. Dispatch compute shader with ceil(width/8) x ceil(height/8) groups
        pass

    def _copy_texture(self, src: "Texture", dst: "Texture") -> None:
        """Copy texture contents.

        Args:
            src: Source texture.
            dst: Destination texture.
        """
        # In real implementation: blit/copy src to dst
        pass

    def destroy(self) -> None:
        """Release all resources."""
        self._history.destroy()
        self._initialized = False

    def __del__(self) -> None:
        """Clean up on deletion."""
        self.destroy()


# =============================================================================
# Temporal Blend Computation
# =============================================================================


def compute_temporal_blend(
    current: Tuple[float, float, float, float],
    history: Tuple[float, float, float, float],
    blend_factor: float,
) -> Tuple[float, float, float, float]:
    """Compute temporally blended color.

    Blends current and history colors using exponential moving average.

    Args:
        current: Current frame color (RGBA).
        history: History buffer color (RGBA).
        blend_factor: Blend weight [0, 1]. Higher = more history.

    Returns:
        Blended color.
    """
    alpha = blend_factor
    return tuple(
        history[i] * alpha + current[i] * (1.0 - alpha)
        for i in range(4)
    )


def compute_luminance(color: Tuple[float, float, float, float]) -> float:
    """Compute perceptual luminance of a color.

    Uses Rec. 709 coefficients.

    Args:
        color: RGBA color tuple.

    Returns:
        Luminance value.
    """
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def compute_luminance_weight(
    current: Tuple[float, float, float, float],
    history: Tuple[float, float, float, float],
) -> float:
    """Compute weight adjustment based on luminance difference.

    Reduces history weight when there's a significant luminance
    change to reduce ghosting.

    Args:
        current: Current frame color.
        history: History buffer color.

    Returns:
        Weight multiplier [0, 1].
    """
    current_lum = compute_luminance(current)
    history_lum = compute_luminance(history)

    if history_lum < 1e-6:
        return 1.0

    # Compute relative difference
    diff = abs(current_lum - history_lum) / (history_lum + 1e-6)

    # Soft transition
    return max(0.0, 1.0 - diff * 2.0)


# =============================================================================
# Variance Tracking
# =============================================================================


@dataclass
class VarianceTracker:
    """Tracks temporal variance for convergence monitoring.

    Uses exponential moving average to track variance over time,
    useful for adaptive sampling and quality metrics.

    Attributes:
        mean: Running mean value.
        variance: Running variance estimate.
        sample_count: Number of accumulated samples.
        gamma: Decay factor for exponential moving average.
    """

    mean: float = 0.0
    variance: float = 0.0
    sample_count: int = 0
    gamma: float = 0.1

    def update(self, value: float) -> None:
        """Update tracker with a new sample.

        Uses Welford's algorithm with exponential decay.

        Args:
            value: New sample value.
        """
        self.sample_count += 1

        if self.sample_count == 1:
            self.mean = value
            self.variance = 0.0
            return

        # Exponential moving average
        delta = value - self.mean
        self.mean += self.gamma * delta
        self.variance = (1.0 - self.gamma) * (
            self.variance + self.gamma * delta * delta
        )

    def get_standard_deviation(self) -> float:
        """Get current standard deviation estimate.

        Returns:
            Standard deviation (sqrt of variance).
        """
        return math.sqrt(max(0.0, self.variance))

    def is_converged(self, threshold: float = 0.01) -> bool:
        """Check if variance has converged below threshold.

        Args:
            threshold: Variance threshold for convergence.

        Returns:
            True if converged.
        """
        return self.sample_count >= 8 and self.variance < threshold

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.mean = 0.0
        self.variance = 0.0
        self.sample_count = 0


# =============================================================================
# Quality Presets
# =============================================================================


def create_low_quality_config() -> TemporalConfig:
    """Create a low-quality temporal config for performance.

    Returns:
        TemporalConfig optimized for speed.
    """
    return TemporalConfig(
        enabled=True,
        blend_factor=0.85,
        clamp_mode=ClampMode.AABB,
        reset_condition=ResetCondition.VELOCITY,
        anti_flicker=False,
        luminance_weight=False,
    )


def create_medium_quality_config() -> TemporalConfig:
    """Create a medium-quality temporal config (default).

    Returns:
        Balanced TemporalConfig.
    """
    return TemporalConfig(
        enabled=True,
        blend_factor=0.9,
        clamp_mode=ClampMode.VARIANCE,
        reset_condition=ResetCondition.VELOCITY,
        anti_flicker=True,
        luminance_weight=True,
    )


def create_high_quality_config() -> TemporalConfig:
    """Create a high-quality temporal config for maximum quality.

    Returns:
        TemporalConfig optimized for quality.
    """
    return TemporalConfig(
        enabled=True,
        blend_factor=0.95,
        clamp_mode=ClampMode.CLIPPED,
        reset_condition=ResetCondition.BOTH,
        anti_flicker=True,
        luminance_weight=True,
        variance_gamma=0.05,  # Slower decay for smoother results
    )


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Constants
    "DEFAULT_BLEND_FACTOR",
    "DEFAULT_VELOCITY_THRESHOLD",
    "DEFAULT_CLAMP_EXPAND",
    "DEFAULT_MIN_VARIANCE",
    "MAX_VARIANCE_FRAMES",
    "WORKGROUP_SIZE",
    # Enums
    "ClampMode",
    "ResetCondition",
    # Config
    "TemporalConfig",
    # Statistics
    "NeighbourhoodStats",
    "compute_neighbourhood_stats",
    # Clamping
    "clamp_color",
    "clip_color_to_aabb",
    # Reprojection
    "reproject_uv",
    "velocity_magnitude",
    "is_valid_uv",
    # History
    "TemporalHistory",
    "TemporalUniforms",
    # Main class
    "SSGITemporalAccumulator",
    # Blend computation
    "compute_temporal_blend",
    "compute_luminance",
    "compute_luminance_weight",
    # Variance
    "VarianceTracker",
    # Presets
    "create_low_quality_config",
    "create_medium_quality_config",
    "create_high_quality_config",
]
