"""
SSR Temporal Reprojection System

Implements temporal reprojection for Screen-Space Reflections (SSR):
- TemporalBuffer: Ping-pong history buffers with confidence tracking
- TemporalBufferSet: Manages history/current buffer pairs
- SSRTemporalReprojection: Main reprojection and accumulation pass
- DisocclusionMode: Rejection strategies for invalid history
- SSRTemporalConfig: Configuration for temporal filtering

The temporal reprojection algorithm:
1. Sample current SSR ray march result
2. Reproject history using velocity buffer
3. Detect disocclusion via depth delta, normal dot, velocity magnitude
4. Compute confidence weight for history sample
5. Blend current and history with confidence-weighted factor
6. Output accumulated result and update history

Convergence: After ~8 accumulated frames, result should be stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from engine.platform.rhi.device import Device
    from engine.platform.rhi.resources import Texture


# =============================================================================
# Enumerations
# =============================================================================


class DisocclusionMode(IntEnum):
    """Disocclusion detection strategies for rejecting invalid history.

    DEPTH_ONLY: Reject based on depth discontinuity only.
    NORMAL_ONLY: Reject based on surface normal change.
    VELOCITY_ONLY: Reject based on motion vector magnitude.
    COMBINED: Use all three criteria (recommended for quality).
    ADAPTIVE: Dynamically weight criteria based on scene content.
    """

    DEPTH_ONLY = 0
    NORMAL_ONLY = 1
    VELOCITY_ONLY = 2
    COMBINED = 3
    ADAPTIVE = 4


class TemporalQuality(Enum):
    """Quality presets for temporal reprojection.

    LOW: Fast, aggressive rejection, may flicker.
    MEDIUM: Balanced quality and performance.
    HIGH: High quality, more stable, slower.
    ULTRA: Maximum quality, most accumulated frames.
    """

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    ULTRA = auto()


# Quality preset parameters: (history_weight, frames_to_converge, variance_gamma)
_QUALITY_PARAMS: Dict[TemporalQuality, Tuple[float, int, float]] = {
    TemporalQuality.LOW: (0.85, 4, 1.5),
    TemporalQuality.MEDIUM: (0.90, 6, 1.25),
    TemporalQuality.HIGH: (0.93, 8, 1.0),
    TemporalQuality.ULTRA: (0.96, 12, 0.75),
}


# =============================================================================
# Temporal Sample
# =============================================================================


@dataclass
class TemporalSample:
    """A single temporal sample with metadata.

    Attributes:
        color: RGB color value of the sample.
        alpha: Alpha/hit confidence of the sample.
        depth: Linear depth at sample location.
        normal: World-space surface normal at sample.
        velocity: Screen-space motion vector.
        confidence: Accumulated confidence weight [0, 1].
        frame_count: Number of frames accumulated into this sample.
    """

    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha: float = 0.0
    depth: float = 0.0
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    frame_count: int = 0

    def is_valid(self) -> bool:
        """Check if sample has valid data.

        Returns:
            True if sample has non-zero confidence and depth.
        """
        return self.confidence > 0.0 and self.depth > 0.0

    def luminance(self) -> float:
        """Compute luminance of the sample color.

        Returns:
            Luminance value using standard rec.709 coefficients.
        """
        r, g, b = self.color
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def velocity_magnitude(self) -> float:
        """Compute magnitude of velocity vector.

        Returns:
            Velocity magnitude in screen-space units.
        """
        vx, vy = self.velocity
        return math.sqrt(vx * vx + vy * vy)

    def blend_with(
        self,
        other: "TemporalSample",
        weight: float,
    ) -> "TemporalSample":
        """Blend this sample with another.

        Args:
            other: Sample to blend with.
            weight: Weight for other sample [0, 1].

        Returns:
            Blended sample.
        """
        inv_weight = 1.0 - weight

        return TemporalSample(
            color=(
                self.color[0] * inv_weight + other.color[0] * weight,
                self.color[1] * inv_weight + other.color[1] * weight,
                self.color[2] * inv_weight + other.color[2] * weight,
            ),
            alpha=self.alpha * inv_weight + other.alpha * weight,
            depth=self.depth * inv_weight + other.depth * weight,
            normal=(
                self.normal[0] * inv_weight + other.normal[0] * weight,
                self.normal[1] * inv_weight + other.normal[1] * weight,
                self.normal[2] * inv_weight + other.normal[2] * weight,
            ),
            velocity=(
                self.velocity[0] * inv_weight + other.velocity[0] * weight,
                self.velocity[1] * inv_weight + other.velocity[1] * weight,
            ),
            confidence=self.confidence * inv_weight + other.confidence * weight,
            frame_count=max(self.frame_count, other.frame_count) + 1,
        )


# =============================================================================
# Temporal Buffer
# =============================================================================


@dataclass
class TemporalBuffer:
    """A single temporal history buffer with metadata.

    Manages a texture buffer for storing temporal history along with
    associated metadata like frame index and validity.

    Attributes:
        texture: The GPU texture resource (may be None if not allocated).
        width: Buffer width in pixels.
        height: Buffer height in pixels.
        frame_index: Frame index when this buffer was last written.
        valid: Whether the buffer contains valid data.
        format: Texture format string (e.g., 'rgba16f').
    """

    texture: Optional["Texture"] = None
    width: int = 0
    height: int = 0
    frame_index: int = -1
    valid: bool = False
    format: str = "rgba16f"

    def is_allocated(self) -> bool:
        """Check if texture is allocated.

        Returns:
            True if texture exists.
        """
        return self.texture is not None

    def matches_dimensions(self, width: int, height: int) -> bool:
        """Check if buffer matches given dimensions.

        Args:
            width: Expected width.
            height: Expected height.

        Returns:
            True if dimensions match.
        """
        return self.width == width and self.height == height

    def invalidate(self) -> None:
        """Mark buffer as invalid."""
        self.valid = False
        self.frame_index = -1

    def mark_written(self, frame_index: int) -> None:
        """Mark buffer as written at given frame.

        Args:
            frame_index: Current frame index.
        """
        self.valid = True
        self.frame_index = frame_index

    def age(self, current_frame: int) -> int:
        """Get age of buffer in frames.

        Args:
            current_frame: Current frame index.

        Returns:
            Number of frames since buffer was written, or -1 if invalid.
        """
        if not self.valid or self.frame_index < 0:
            return -1
        return current_frame - self.frame_index


@dataclass
class TemporalBufferSet:
    """Ping-pong buffer pair for temporal accumulation.

    Manages two buffers that alternate between read (history) and
    write (current) roles each frame.

    Attributes:
        buffer_a: First buffer in ping-pong pair.
        buffer_b: Second buffer in ping-pong pair.
        read_index: Index of buffer to read (0 or 1).
        frame_count: Total frames processed.
    """

    buffer_a: TemporalBuffer = field(default_factory=TemporalBuffer)
    buffer_b: TemporalBuffer = field(default_factory=TemporalBuffer)
    read_index: int = 0
    frame_count: int = 0

    @property
    def history_buffer(self) -> TemporalBuffer:
        """Get the history (read) buffer.

        Returns:
            Buffer containing previous frame's accumulated result.
        """
        return self.buffer_a if self.read_index == 0 else self.buffer_b

    @property
    def current_buffer(self) -> TemporalBuffer:
        """Get the current (write) buffer.

        Returns:
            Buffer to write this frame's accumulated result.
        """
        return self.buffer_b if self.read_index == 0 else self.buffer_a

    def swap(self) -> None:
        """Swap read and write buffers for next frame."""
        self.read_index = 1 - self.read_index
        self.frame_count += 1

    def invalidate_all(self) -> None:
        """Invalidate both buffers (e.g., on resolution change)."""
        self.buffer_a.invalidate()
        self.buffer_b.invalidate()
        self.frame_count = 0

    def needs_resize(self, width: int, height: int) -> bool:
        """Check if buffers need resizing.

        Args:
            width: Target width.
            height: Target height.

        Returns:
            True if either buffer needs resize.
        """
        return not (
            self.buffer_a.matches_dimensions(width, height)
            and self.buffer_b.matches_dimensions(width, height)
        )

    def get_convergence_progress(self, target_frames: int = 8) -> float:
        """Get convergence progress toward stable result.

        Args:
            target_frames: Number of frames for full convergence.

        Returns:
            Progress value [0, 1] where 1 = fully converged.
        """
        if target_frames <= 0:
            return 1.0
        return min(1.0, self.frame_count / target_frames)

    def is_converged(self, target_frames: int = 8) -> bool:
        """Check if temporal accumulation has converged.

        Args:
            target_frames: Number of frames for full convergence.

        Returns:
            True if frame_count >= target_frames.
        """
        return self.frame_count >= target_frames


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SSRTemporalConfig:
    """Configuration for SSR temporal reprojection.

    Attributes:
        quality: Quality preset (affects history weight, convergence).
        disocclusion_mode: Strategy for detecting invalid history.
        history_weight: Base weight for history samples [0.8, 0.98].
        frames_to_converge: Target frames for stable result.
        depth_threshold: Depth difference threshold for rejection.
        normal_threshold: Normal dot product threshold for rejection.
        velocity_threshold: Velocity magnitude threshold for rejection.
        variance_gamma: Variance clipping aggressiveness.
        neighborhood_size: Size of color box for clamping (3, 5, or 7).
        use_variance_clipping: Enable AABB variance clipping.
        use_ycocg_space: Convert to YCoCg for better clamping.
        anti_flicker: Enable anti-flicker filter.
        luminance_weight: Weight history by luminance difference.
        motion_weight_scale: Scale factor for motion-based weight reduction.
    """

    quality: TemporalQuality = TemporalQuality.HIGH
    disocclusion_mode: DisocclusionMode = DisocclusionMode.COMBINED

    # History blend weights (can override quality preset)
    history_weight: Optional[float] = None
    frames_to_converge: Optional[int] = None

    # Disocclusion thresholds
    depth_threshold: float = 0.01
    normal_threshold: float = 0.9
    velocity_threshold: float = 0.02

    # Variance clipping
    variance_gamma: Optional[float] = None
    neighborhood_size: int = 3
    use_variance_clipping: bool = True
    use_ycocg_space: bool = True

    # Anti-flicker
    anti_flicker: bool = True
    luminance_weight: bool = True

    # Motion handling
    motion_weight_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.neighborhood_size not in (3, 5, 7):
            raise ValueError(
                f"neighborhood_size must be 3, 5, or 7, got {self.neighborhood_size}"
            )
        if self.depth_threshold <= 0:
            raise ValueError(
                f"depth_threshold must be positive, got {self.depth_threshold}"
            )
        if not 0 <= self.normal_threshold <= 1:
            raise ValueError(
                f"normal_threshold must be in [0, 1], got {self.normal_threshold}"
            )
        if self.velocity_threshold < 0:
            raise ValueError(
                f"velocity_threshold must be non-negative, got {self.velocity_threshold}"
            )

    def get_history_weight(self) -> float:
        """Get effective history weight.

        Returns:
            History weight from config or quality preset.
        """
        if self.history_weight is not None:
            return max(0.0, min(1.0, self.history_weight))
        return _QUALITY_PARAMS[self.quality][0]

    def get_frames_to_converge(self) -> int:
        """Get target frames for convergence.

        Returns:
            Frame count from config or quality preset.
        """
        if self.frames_to_converge is not None:
            return max(1, self.frames_to_converge)
        return _QUALITY_PARAMS[self.quality][1]

    def get_variance_gamma(self) -> float:
        """Get variance clipping gamma.

        Returns:
            Gamma value from config or quality preset.
        """
        if self.variance_gamma is not None:
            return max(0.1, self.variance_gamma)
        return _QUALITY_PARAMS[self.quality][2]


# =============================================================================
# Temporal Statistics
# =============================================================================


@dataclass
class TemporalStats:
    """Statistics for temporal reprojection pass.

    Attributes:
        total_pixels: Total pixels processed.
        valid_history_pixels: Pixels with valid history.
        rejected_pixels: Pixels where history was rejected.
        clamped_pixels: Pixels where history was clamped.
        converged_pixels: Pixels that have reached convergence.
        average_confidence: Average confidence across all pixels.
        average_velocity: Average velocity magnitude.
        max_velocity: Maximum velocity magnitude.
        frame_index: Frame index for these stats.
    """

    total_pixels: int = 0
    valid_history_pixels: int = 0
    rejected_pixels: int = 0
    clamped_pixels: int = 0
    converged_pixels: int = 0
    average_confidence: float = 0.0
    average_velocity: float = 0.0
    max_velocity: float = 0.0
    frame_index: int = 0

    @property
    def rejection_rate(self) -> float:
        """Get fraction of pixels with rejected history.

        Returns:
            Rejection rate [0, 1].
        """
        if self.total_pixels == 0:
            return 0.0
        return self.rejected_pixels / self.total_pixels

    @property
    def convergence_rate(self) -> float:
        """Get fraction of converged pixels.

        Returns:
            Convergence rate [0, 1].
        """
        if self.total_pixels == 0:
            return 0.0
        return self.converged_pixels / self.total_pixels

    @property
    def history_usage_rate(self) -> float:
        """Get fraction of pixels using valid history.

        Returns:
            History usage rate [0, 1].
        """
        if self.total_pixels == 0:
            return 0.0
        return self.valid_history_pixels / self.total_pixels

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.total_pixels = 0
        self.valid_history_pixels = 0
        self.rejected_pixels = 0
        self.clamped_pixels = 0
        self.converged_pixels = 0
        self.average_confidence = 0.0
        self.average_velocity = 0.0
        self.max_velocity = 0.0


# =============================================================================
# SSR Temporal Reprojection
# =============================================================================


class SSRTemporalReprojection:
    """SSR Temporal Reprojection pass for flickering elimination.

    Accumulates SSR results over time using velocity buffer reprojection.
    Implements disocclusion rejection to prevent ghosting on fast motion.

    The algorithm:
    1. For each pixel, sample current SSR result
    2. Compute reprojected UV using velocity buffer
    3. Sample history at reprojected location
    4. Test disocclusion criteria (depth, normal, velocity)
    5. If valid, clamp history to current neighborhood (variance clip)
    6. Blend current and clamped history with confidence weight
    7. Update confidence based on history validity
    8. Write to current buffer, swap for next frame

    Example:
        config = SSRTemporalConfig(quality=TemporalQuality.HIGH)
        reprojection = SSRTemporalReprojection(device, config)
        reprojection.setup(1920, 1080)

        # Each frame:
        reprojection.execute(
            current_ssr=ssr_output,
            velocity_buffer=velocity,
            depth_buffer=depth,
            normal_buffer=normal,
            output=final_ssr,
        )
    """

    def __init__(
        self,
        device: Optional["Device"] = None,
        config: Optional[SSRTemporalConfig] = None,
    ) -> None:
        """Initialize SSR temporal reprojection.

        Args:
            device: RHI device for resource creation (can be None for testing).
            config: Configuration parameters.
        """
        self._device = device
        self._config = config or SSRTemporalConfig()
        self._buffers = TemporalBufferSet()
        self._stats = TemporalStats()
        self._width = 0
        self._height = 0
        self._initialized = False
        self._frame_index = 0

        # Shader resources (placeholder)
        self._compute_pipeline: Any = None
        self._bind_group: Any = None
        self._uniform_buffer: Any = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def device(self) -> Optional["Device"]:
        """Get the RHI device."""
        return self._device

    @property
    def config(self) -> SSRTemporalConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, value: SSRTemporalConfig) -> None:
        """Set configuration (does not invalidate history)."""
        self._config = value

    @property
    def buffers(self) -> TemporalBufferSet:
        """Get the temporal buffer set."""
        return self._buffers

    @property
    def stats(self) -> TemporalStats:
        """Get statistics from last execution."""
        return self._stats

    @property
    def is_initialized(self) -> bool:
        """Check if reprojection pass is initialized."""
        return self._initialized

    @property
    def width(self) -> int:
        """Get buffer width."""
        return self._width

    @property
    def height(self) -> int:
        """Get buffer height."""
        return self._height

    @property
    def frame_index(self) -> int:
        """Get current frame index."""
        return self._frame_index

    @property
    def convergence_progress(self) -> float:
        """Get progress toward convergence [0, 1]."""
        return self._buffers.get_convergence_progress(
            self._config.get_frames_to_converge()
        )

    @property
    def is_converged(self) -> bool:
        """Check if temporal accumulation has converged."""
        return self._buffers.is_converged(self._config.get_frames_to_converge())

    # -------------------------------------------------------------------------
    # Setup and Resource Management
    # -------------------------------------------------------------------------

    def setup(self, width: int, height: int) -> None:
        """Initialize or resize temporal buffers.

        Args:
            width: Buffer width in pixels.
            height: Buffer height in pixels.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")

        needs_resize = self._buffers.needs_resize(width, height)

        if needs_resize or not self._initialized:
            self._width = width
            self._height = height
            self._create_buffers(width, height)
            self._buffers.invalidate_all()
            self._initialized = True

    def _create_buffers(self, width: int, height: int) -> None:
        """Create GPU texture resources for temporal buffers.

        Args:
            width: Buffer width.
            height: Buffer height.
        """
        # Update buffer metadata
        self._buffers.buffer_a.width = width
        self._buffers.buffer_a.height = height
        self._buffers.buffer_a.format = "rgba16f"

        self._buffers.buffer_b.width = width
        self._buffers.buffer_b.height = height
        self._buffers.buffer_b.format = "rgba16f"

        # GPU texture creation would happen here with real device
        if self._device is not None:
            # self._buffers.buffer_a.texture = self._device.create_texture(...)
            # self._buffers.buffer_b.texture = self._device.create_texture(...)
            pass

    def invalidate_history(self) -> None:
        """Invalidate all history buffers.

        Call this on camera cuts, resolution changes, or other
        discontinuities that make history invalid.
        """
        self._buffers.invalidate_all()
        self._stats.reset()

    def destroy(self) -> None:
        """Release GPU resources."""
        self._buffers.buffer_a.texture = None
        self._buffers.buffer_b.texture = None
        self._compute_pipeline = None
        self._bind_group = None
        self._uniform_buffer = None
        self._initialized = False

    # -------------------------------------------------------------------------
    # Disocclusion Detection
    # -------------------------------------------------------------------------

    def compute_depth_rejection(
        self,
        current_depth: float,
        history_depth: float,
    ) -> float:
        """Compute depth-based rejection weight.

        Args:
            current_depth: Current frame linear depth.
            history_depth: History sample depth.

        Returns:
            Rejection weight [0, 1] where 1 = no rejection.
        """
        if current_depth <= 0.0 or history_depth <= 0.0:
            return 0.0

        # Relative depth difference
        depth_diff = abs(current_depth - history_depth) / max(current_depth, 1e-6)

        threshold = self._config.depth_threshold
        if depth_diff > threshold:
            return 0.0

        # Smooth falloff near threshold
        return 1.0 - (depth_diff / threshold)

    def compute_normal_rejection(
        self,
        current_normal: Tuple[float, float, float],
        history_normal: Tuple[float, float, float],
    ) -> float:
        """Compute normal-based rejection weight.

        Args:
            current_normal: Current frame surface normal.
            history_normal: History sample normal.

        Returns:
            Rejection weight [0, 1] where 1 = no rejection.
        """
        # Dot product of normals
        dot = (
            current_normal[0] * history_normal[0]
            + current_normal[1] * history_normal[1]
            + current_normal[2] * history_normal[2]
        )

        threshold = self._config.normal_threshold
        if dot < threshold:
            return 0.0

        # Remap [threshold, 1] to [0, 1]
        return (dot - threshold) / (1.0 - threshold + 1e-6)

    def compute_velocity_rejection(
        self,
        velocity: Tuple[float, float],
    ) -> float:
        """Compute velocity-based rejection weight.

        High velocity indicates fast motion where ghosting is likely.

        Args:
            velocity: Screen-space motion vector.

        Returns:
            Rejection weight [0, 1] where 1 = no rejection.
        """
        magnitude = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

        threshold = self._config.velocity_threshold
        if magnitude > threshold:
            # Scale down weight for high velocity
            scale = self._config.motion_weight_scale
            excess = (magnitude - threshold) * scale
            return max(0.0, 1.0 - excess)

        return 1.0

    def compute_disocclusion_weight(
        self,
        current_depth: float,
        history_depth: float,
        current_normal: Tuple[float, float, float],
        history_normal: Tuple[float, float, float],
        velocity: Tuple[float, float],
    ) -> float:
        """Compute combined disocclusion rejection weight.

        Args:
            current_depth: Current frame linear depth.
            history_depth: History sample depth.
            current_normal: Current frame surface normal.
            history_normal: History sample normal.
            velocity: Screen-space motion vector.

        Returns:
            Combined rejection weight [0, 1] where 1 = valid history.
        """
        mode = self._config.disocclusion_mode

        if mode == DisocclusionMode.DEPTH_ONLY:
            return self.compute_depth_rejection(current_depth, history_depth)

        elif mode == DisocclusionMode.NORMAL_ONLY:
            return self.compute_normal_rejection(current_normal, history_normal)

        elif mode == DisocclusionMode.VELOCITY_ONLY:
            return self.compute_velocity_rejection(velocity)

        elif mode == DisocclusionMode.COMBINED:
            depth_w = self.compute_depth_rejection(current_depth, history_depth)
            normal_w = self.compute_normal_rejection(current_normal, history_normal)
            velocity_w = self.compute_velocity_rejection(velocity)
            return depth_w * normal_w * velocity_w

        elif mode == DisocclusionMode.ADAPTIVE:
            # Weight criteria by velocity magnitude
            velocity_mag = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

            depth_w = self.compute_depth_rejection(current_depth, history_depth)
            normal_w = self.compute_normal_rejection(current_normal, history_normal)
            velocity_w = self.compute_velocity_rejection(velocity)

            # At high velocity, rely more on velocity rejection
            velocity_importance = min(1.0, velocity_mag * 10.0)
            geometry_importance = 1.0 - velocity_importance

            geometry_weight = (depth_w * normal_w) ** geometry_importance
            motion_weight = velocity_w ** velocity_importance

            return geometry_weight * motion_weight

        return 1.0

    # -------------------------------------------------------------------------
    # Color Space Conversion
    # -------------------------------------------------------------------------

    def rgb_to_ycocg(
        self,
        rgb: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Convert RGB to YCoCg color space.

        YCoCg provides better decorrelation for variance clipping.

        Args:
            rgb: RGB color tuple.

        Returns:
            YCoCg color tuple (Y, Co, Cg).
        """
        r, g, b = rgb
        y = 0.25 * r + 0.5 * g + 0.25 * b
        co = 0.5 * r - 0.5 * b
        cg = -0.25 * r + 0.5 * g - 0.25 * b
        return (y, co, cg)

    def ycocg_to_rgb(
        self,
        ycocg: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Convert YCoCg back to RGB color space.

        Args:
            ycocg: YCoCg color tuple (Y, Co, Cg).

        Returns:
            RGB color tuple.
        """
        y, co, cg = ycocg
        r = y + co - cg
        g = y + cg
        b = y - co - cg
        return (r, g, b)

    # -------------------------------------------------------------------------
    # Variance Clipping
    # -------------------------------------------------------------------------

    def compute_neighborhood_stats(
        self,
        samples: List[Tuple[float, float, float]],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute mean and standard deviation of neighborhood.

        Args:
            samples: List of RGB color samples from neighborhood.

        Returns:
            Tuple of (mean, std_dev) as RGB tuples.
        """
        if not samples:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        n = len(samples)

        # Compute mean
        mean_r = sum(s[0] for s in samples) / n
        mean_g = sum(s[1] for s in samples) / n
        mean_b = sum(s[2] for s in samples) / n

        if n < 2:
            return ((mean_r, mean_g, mean_b), (0.0, 0.0, 0.0))

        # Compute variance
        var_r = sum((s[0] - mean_r) ** 2 for s in samples) / (n - 1)
        var_g = sum((s[1] - mean_g) ** 2 for s in samples) / (n - 1)
        var_b = sum((s[2] - mean_b) ** 2 for s in samples) / (n - 1)

        std_r = math.sqrt(max(0.0, var_r))
        std_g = math.sqrt(max(0.0, var_g))
        std_b = math.sqrt(max(0.0, var_b))

        return ((mean_r, mean_g, mean_b), (std_r, std_g, std_b))

    def variance_clip(
        self,
        history_color: Tuple[float, float, float],
        current_samples: List[Tuple[float, float, float]],
    ) -> Tuple[Tuple[float, float, float], bool]:
        """Clip history color to neighborhood AABB.

        Uses variance-based AABB clipping to constrain history to
        plausible values given the current frame's neighborhood.

        Args:
            history_color: History sample color.
            current_samples: Current frame neighborhood samples.

        Returns:
            Tuple of (clipped_color, was_clipped).
        """
        if not self._config.use_variance_clipping:
            return (history_color, False)

        if not current_samples:
            return (history_color, False)

        gamma = self._config.get_variance_gamma()

        # Convert to YCoCg if enabled
        if self._config.use_ycocg_space:
            history_ycocg = self.rgb_to_ycocg(history_color)
            samples_ycocg = [self.rgb_to_ycocg(s) for s in current_samples]
            mean, std = self.compute_neighborhood_stats(samples_ycocg)

            # Clip to AABB
            clipped = (
                max(mean[0] - gamma * std[0], min(mean[0] + gamma * std[0], history_ycocg[0])),
                max(mean[1] - gamma * std[1], min(mean[1] + gamma * std[1], history_ycocg[1])),
                max(mean[2] - gamma * std[2], min(mean[2] + gamma * std[2], history_ycocg[2])),
            )

            was_clipped = clipped != history_ycocg
            result = self.ycocg_to_rgb(clipped)

        else:
            mean, std = self.compute_neighborhood_stats(current_samples)

            # Clip to AABB in RGB space
            clipped = (
                max(mean[0] - gamma * std[0], min(mean[0] + gamma * std[0], history_color[0])),
                max(mean[1] - gamma * std[1], min(mean[1] + gamma * std[1], history_color[1])),
                max(mean[2] - gamma * std[2], min(mean[2] + gamma * std[2], history_color[2])),
            )

            was_clipped = clipped != history_color
            result = clipped

        return (result, was_clipped)

    # -------------------------------------------------------------------------
    # Reprojection
    # -------------------------------------------------------------------------

    def reproject_uv(
        self,
        uv: Tuple[float, float],
        velocity: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Reproject UV coordinates using velocity.

        Args:
            uv: Current frame UV coordinates [0, 1].
            velocity: Screen-space velocity vector.

        Returns:
            Reprojected UV for history lookup.
        """
        return (uv[0] - velocity[0], uv[1] - velocity[1])

    def is_uv_valid(self, uv: Tuple[float, float]) -> bool:
        """Check if UV coordinates are within valid range.

        Args:
            uv: UV coordinates to validate.

        Returns:
            True if UV is in [0, 1] range.
        """
        return 0.0 <= uv[0] <= 1.0 and 0.0 <= uv[1] <= 1.0

    # -------------------------------------------------------------------------
    # Temporal Blend
    # -------------------------------------------------------------------------

    def compute_blend_weight(
        self,
        disocclusion_weight: float,
        velocity: Tuple[float, float],
        history_confidence: float,
    ) -> float:
        """Compute final blend weight for history sample.

        Args:
            disocclusion_weight: Weight from disocclusion detection.
            velocity: Screen-space motion vector.
            history_confidence: Confidence from history buffer.

        Returns:
            Final blend weight for history [0, history_weight].
        """
        base_weight = self._config.get_history_weight()

        # Scale by disocclusion weight
        weight = base_weight * disocclusion_weight

        # Reduce weight for high velocity (less history = less ghosting)
        if self._config.motion_weight_scale > 0:
            velocity_mag = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
            velocity_factor = 1.0 - min(1.0, velocity_mag * self._config.motion_weight_scale)
            weight *= velocity_factor

        # Scale by history confidence for adaptive blending
        if self._config.luminance_weight:
            weight *= min(1.0, history_confidence + 0.1)

        return max(0.0, min(base_weight, weight))

    def blend_samples(
        self,
        current: TemporalSample,
        history: TemporalSample,
        weight: float,
    ) -> TemporalSample:
        """Blend current and history samples.

        Args:
            current: Current frame sample.
            history: History sample (possibly clamped).
            weight: Weight for history sample [0, 1].

        Returns:
            Blended result sample.
        """
        if weight <= 0.0 or not history.is_valid():
            # No history contribution
            result = TemporalSample(
                color=current.color,
                alpha=current.alpha,
                depth=current.depth,
                normal=current.normal,
                velocity=current.velocity,
                confidence=0.1,  # Fresh sample, low confidence
                frame_count=1,
            )
        else:
            # Blend with history
            inv_weight = 1.0 - weight

            blended_color = (
                current.color[0] * inv_weight + history.color[0] * weight,
                current.color[1] * inv_weight + history.color[1] * weight,
                current.color[2] * inv_weight + history.color[2] * weight,
            )

            blended_alpha = current.alpha * inv_weight + history.alpha * weight

            # Update confidence based on history validity
            new_confidence = min(1.0, history.confidence + 0.1 * (1.0 - history.confidence))

            result = TemporalSample(
                color=blended_color,
                alpha=blended_alpha,
                depth=current.depth,  # Always use current depth
                normal=current.normal,  # Always use current normal
                velocity=current.velocity,  # Always use current velocity
                confidence=new_confidence * weight + 0.1 * inv_weight,
                frame_count=history.frame_count + 1,
            )

        return result

    # -------------------------------------------------------------------------
    # Anti-Flicker
    # -------------------------------------------------------------------------

    def apply_anti_flicker(
        self,
        current: Tuple[float, float, float],
        history: Tuple[float, float, float],
        blended: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Apply anti-flicker filter to reduce temporal instability.

        Uses luminance-based weighting to reduce high-frequency flicker.

        Args:
            current: Current frame color.
            history: History color.
            blended: Preliminary blended result.

        Returns:
            Anti-flickered color.
        """
        if not self._config.anti_flicker:
            return blended

        # Compute luminances
        def luminance(c: Tuple[float, float, float]) -> float:
            return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

        lum_current = luminance(current)
        lum_history = luminance(history)
        lum_blended = luminance(blended)

        # Detect flicker: large luminance change between current and history
        lum_diff = abs(lum_current - lum_history)

        if lum_diff < 0.01:
            return blended

        # Bias toward history when flicker is detected
        flicker_weight = min(1.0, lum_diff * 2.0)
        bias_toward_history = 0.1 * flicker_weight

        return (
            blended[0] + (history[0] - blended[0]) * bias_toward_history,
            blended[1] + (history[1] - blended[1]) * bias_toward_history,
            blended[2] + (history[2] - blended[2]) * bias_toward_history,
        )

    # -------------------------------------------------------------------------
    # Main Execution
    # -------------------------------------------------------------------------

    def process_pixel(
        self,
        uv: Tuple[float, float],
        current_color: Tuple[float, float, float],
        current_alpha: float,
        current_depth: float,
        current_normal: Tuple[float, float, float],
        velocity: Tuple[float, float],
        history_color: Tuple[float, float, float],
        history_alpha: float,
        history_depth: float,
        history_normal: Tuple[float, float, float],
        history_confidence: float,
        history_frame_count: int,
        neighborhood_samples: List[Tuple[float, float, float]],
    ) -> TemporalSample:
        """Process a single pixel through temporal reprojection.

        This is the core algorithm that would run per-pixel in the compute shader.

        Args:
            uv: Pixel UV coordinates.
            current_color: Current frame SSR color.
            current_alpha: Current frame SSR alpha/hit.
            current_depth: Current frame depth.
            current_normal: Current frame normal.
            velocity: Motion vector for this pixel.
            history_color: Reprojected history color.
            history_alpha: Reprojected history alpha.
            history_depth: Reprojected history depth.
            history_normal: Reprojected history normal.
            history_confidence: History sample confidence.
            history_frame_count: Frames accumulated in history.
            neighborhood_samples: Samples from current frame neighborhood.

        Returns:
            Processed temporal sample for output.
        """
        current_sample = TemporalSample(
            color=current_color,
            alpha=current_alpha,
            depth=current_depth,
            normal=current_normal,
            velocity=velocity,
            confidence=0.1,
            frame_count=1,
        )

        # Check if reprojected UV is valid
        reprojected_uv = self.reproject_uv(uv, velocity)
        if not self.is_uv_valid(reprojected_uv):
            # Invalid history location - use current only
            return current_sample

        history_sample = TemporalSample(
            color=history_color,
            alpha=history_alpha,
            depth=history_depth,
            normal=history_normal,
            velocity=(0.0, 0.0),  # History velocity not tracked
            confidence=history_confidence,
            frame_count=history_frame_count,
        )

        # Compute disocclusion weight
        disocclusion_weight = self.compute_disocclusion_weight(
            current_depth,
            history_depth,
            current_normal,
            history_normal,
            velocity,
        )

        if disocclusion_weight <= 0.0:
            # History rejected - use current only
            return current_sample

        # Variance clip history to neighborhood
        clipped_history_color, was_clipped = self.variance_clip(
            history_color,
            neighborhood_samples,
        )

        # Update history sample with clipped color
        history_sample = TemporalSample(
            color=clipped_history_color,
            alpha=history_alpha,
            depth=history_depth,
            normal=history_normal,
            velocity=(0.0, 0.0),
            confidence=history_confidence * (0.9 if was_clipped else 1.0),
            frame_count=history_frame_count,
        )

        # Compute final blend weight
        blend_weight = self.compute_blend_weight(
            disocclusion_weight,
            velocity,
            history_confidence,
        )

        # Blend samples
        result = self.blend_samples(current_sample, history_sample, blend_weight)

        # Apply anti-flicker
        if self._config.anti_flicker:
            anti_flickered = self.apply_anti_flicker(
                current_color,
                clipped_history_color,
                result.color,
            )
            result = TemporalSample(
                color=anti_flickered,
                alpha=result.alpha,
                depth=result.depth,
                normal=result.normal,
                velocity=result.velocity,
                confidence=result.confidence,
                frame_count=result.frame_count,
            )

        return result

    def execute(
        self,
        current_ssr: Any,
        velocity_buffer: Any,
        depth_buffer: Any,
        normal_buffer: Any,
        output: Any,
    ) -> None:
        """Execute temporal reprojection pass.

        In production, this dispatches the compute shader. For testing,
        the actual work happens in process_pixel.

        Args:
            current_ssr: Current frame SSR result texture.
            velocity_buffer: Per-pixel motion vectors.
            depth_buffer: Scene depth buffer.
            normal_buffer: World-space normal buffer.
            output: Output texture to write accumulated result.
        """
        if not self._initialized:
            raise RuntimeError("SSRTemporalReprojection not initialized. Call setup() first.")

        self._stats.reset()
        self._stats.frame_index = self._frame_index

        # In real implementation:
        # 1. Bind resources to compute pipeline
        # 2. Set uniforms (config, frame index, etc.)
        # 3. Dispatch compute shader
        # 4. Copy stats from GPU readback

        # Swap buffers for next frame
        self._buffers.swap()
        self._buffers.current_buffer.mark_written(self._frame_index)
        self._frame_index += 1

    def get_shader_uniforms(self) -> Dict[str, Any]:
        """Get uniform values for compute shader.

        Returns:
            Dictionary of uniform name to value.
        """
        return {
            "history_weight": self._config.get_history_weight(),
            "variance_gamma": self._config.get_variance_gamma(),
            "depth_threshold": self._config.depth_threshold,
            "normal_threshold": self._config.normal_threshold,
            "velocity_threshold": self._config.velocity_threshold,
            "motion_weight_scale": self._config.motion_weight_scale,
            "disocclusion_mode": int(self._config.disocclusion_mode),
            "neighborhood_size": self._config.neighborhood_size,
            "use_variance_clipping": int(self._config.use_variance_clipping),
            "use_ycocg_space": int(self._config.use_ycocg_space),
            "anti_flicker": int(self._config.anti_flicker),
            "luminance_weight": int(self._config.luminance_weight),
            "frame_index": self._frame_index,
            "resolution": (self._width, self._height),
        }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "DisocclusionMode",
    "SSRTemporalConfig",
    "SSRTemporalReprojection",
    "TemporalBuffer",
    "TemporalBufferSet",
    "TemporalQuality",
    "TemporalSample",
    "TemporalStats",
]
