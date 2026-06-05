"""
Temporal Anti-Aliasing via World-Space Hit Position Reprojection (T-DEMO-8.5).

This module implements TAA for ray marching using world-space hit positions
instead of traditional motion vectors. This approach is ideal for SDF rendering:

1. **World-Space Hit Position Storage**
   - Store the world-space hit position per pixel (RGB32F)
   - Use this for reprojection to previous frame's screen space
   - No explicit motion vectors needed

2. **Reprojection Without Motion Vectors**
   - Transform previous frame's hit position to current screen space
   - Sample history buffer at the reprojected location
   - Reject invalid samples (disocclusion detection)

3. **Stable Image Output**
   - Blend current + history with adaptive feedback weight
   - YCoCg neighborhood clamping for ghosting reduction
   - Converges to high-quality anti-aliased result

Formula:
    prev_uv = project(current_view_proj * prev_hit_position)
    history_color = sample(history_buffer, prev_uv)
    output = lerp(clamped_history, current, 0.1)

Usage:
    >>> from engine.rendering.demoscene.taa_reprojection import (
    ...     TAAReprojection, ReprojectionConfig, HitPositionBuffer
    ... )
    >>> config = ReprojectionConfig(blend_factor=0.1)
    >>> taa = TAAReprojection(width=1920, height=1080, config=config)
    >>> # After ray marching, store hit positions
    >>> taa.update_hit_positions(hit_buffer)
    >>> # Reproject and blend
    >>> result = taa.accumulate(current_color, current_view_proj, prev_view_proj)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec2, Vec3, Vec4


# =============================================================================
# Constants
# =============================================================================

# Default blend factor for temporal accumulation (lower = smoother, slower)
DEFAULT_BLEND_FACTOR = 0.1

# Minimum blend factor to prevent stale history
MIN_BLEND_FACTOR = 0.02

# Maximum blend factor for fast updates on disocclusion
MAX_BLEND_FACTOR = 0.8

# Default disocclusion threshold (world-space distance)
DEFAULT_DISOCCLUSION_THRESHOLD = 0.5

# Depth rejection threshold (relative depth difference)
DEFAULT_DEPTH_THRESHOLD = 0.1

# Invalid hit position marker (for misses)
INVALID_HIT_DISTANCE = -1.0


# =============================================================================
# Enums and Configuration
# =============================================================================


class DisocclusionMode(Enum):
    """Mode for detecting disoccluded pixels."""

    NONE = auto()
    """No disocclusion detection (fast but may ghost)."""

    DEPTH = auto()
    """Detect by comparing reprojected depth to current depth."""

    POSITION = auto()
    """Detect by comparing reprojected world position to current position."""

    COMBINED = auto()
    """Use both depth and position checks (most robust)."""


class ClampingMode(Enum):
    """Color clamping mode for ghosting reduction."""

    NONE = auto()
    """No clamping (maximum blur, may ghost on edges)."""

    RGB = auto()
    """Clamp in RGB color space (simple but can cause saturation issues)."""

    YCOCG = auto()
    """Clamp in YCoCg color space (perceptually better, standard TAA approach)."""

    VARIANCE = auto()
    """Use variance-based clamping (adaptive, handles HDR better)."""


@dataclass
class ReprojectionConfig:
    """Configuration for TAA reprojection.

    Attributes:
        blend_factor: Base blend factor for EMA (0.0-1.0). Lower = smoother.
        min_blend: Minimum blend factor to prevent stale history.
        max_blend: Maximum blend factor for fast reset on disocclusion.
        disocclusion_mode: How to detect disoccluded pixels.
        clamping_mode: Color space for neighborhood clamping.
        disocclusion_threshold: World-space distance threshold for disocclusion.
        depth_threshold: Relative depth difference threshold.
        enable_velocity_weighting: Weight blend by velocity magnitude.
        neighborhood_size: Size of neighborhood for clamping (1=3x3, 2=5x5).
    """

    blend_factor: float = DEFAULT_BLEND_FACTOR
    min_blend: float = MIN_BLEND_FACTOR
    max_blend: float = MAX_BLEND_FACTOR
    disocclusion_mode: DisocclusionMode = DisocclusionMode.COMBINED
    clamping_mode: ClampingMode = ClampingMode.YCOCG
    disocclusion_threshold: float = DEFAULT_DISOCCLUSION_THRESHOLD
    depth_threshold: float = DEFAULT_DEPTH_THRESHOLD
    enable_velocity_weighting: bool = True
    neighborhood_size: int = 1

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 < self.blend_factor <= 1.0:
            raise ValueError(f"blend_factor must be in (0, 1], got {self.blend_factor}")
        if not 0.0 < self.min_blend <= self.blend_factor:
            raise ValueError(f"min_blend must be in (0, blend_factor], got {self.min_blend}")
        if not self.blend_factor <= self.max_blend <= 1.0:
            raise ValueError(f"max_blend must be in [blend_factor, 1], got {self.max_blend}")
        if self.disocclusion_threshold <= 0:
            raise ValueError(f"disocclusion_threshold must be positive, got {self.disocclusion_threshold}")
        if self.depth_threshold <= 0:
            raise ValueError(f"depth_threshold must be positive, got {self.depth_threshold}")
        if self.neighborhood_size < 1:
            raise ValueError(f"neighborhood_size must be >= 1, got {self.neighborhood_size}")


# =============================================================================
# Hit Position Buffer
# =============================================================================


@dataclass
class HitPositionBuffer:
    """Buffer storing world-space hit positions from ray marching.

    This replaces traditional motion vectors for TAA in ray marching scenarios.
    Each pixel stores the 3D world-space position where the ray hit a surface,
    or a marker value for ray misses (sky, background).

    Attributes:
        width: Buffer width in pixels.
        height: Buffer height in pixels.
        positions: Flat list of Vec3 positions (width * height).
        valid: Flat list of booleans indicating valid hits.
    """

    width: int
    height: int
    positions: List[Vec3] = field(default_factory=list, repr=False)
    valid: List[bool] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Initialize buffers if not provided."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid dimensions: {self.width}x{self.height}")
        size = self.width * self.height
        if not self.positions:
            self.positions = [Vec3.zero() for _ in range(size)]
        if not self.valid:
            self.valid = [False] * size

    def set_hit(self, x: int, y: int, position: Vec3) -> None:
        """Set a valid hit position at pixel (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = y * self.width + x
            self.positions[idx] = position
            self.valid[idx] = True

    def set_miss(self, x: int, y: int) -> None:
        """Mark pixel (x, y) as a miss (no surface hit)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = y * self.width + x
            self.positions[idx] = Vec3.zero()
            self.valid[idx] = False

    def get_position(self, x: int, y: int) -> Tuple[Vec3, bool]:
        """Get hit position and validity at pixel (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = y * self.width + x
            return self.positions[idx], self.valid[idx]
        return Vec3.zero(), False

    def sample_bilinear(self, uv: Vec2) -> Tuple[Vec3, float]:
        """Sample position with bilinear interpolation.

        Returns:
            Tuple of (interpolated position, validity weight 0-1).
        """
        # Convert UV to pixel coordinates
        px = uv.x * self.width - 0.5
        py = uv.y * self.height - 0.5

        x0 = int(math.floor(px))
        y0 = int(math.floor(py))
        x1 = x0 + 1
        y1 = y0 + 1

        fx = px - x0
        fy = py - y0

        # Sample 4 positions
        p00, v00 = self.get_position(x0, y0)
        p10, v10 = self.get_position(x1, y0)
        p01, v01 = self.get_position(x0, y1)
        p11, v11 = self.get_position(x1, y1)

        # Compute weights
        w00 = (1 - fx) * (1 - fy) * (1.0 if v00 else 0.0)
        w10 = fx * (1 - fy) * (1.0 if v10 else 0.0)
        w01 = (1 - fx) * fy * (1.0 if v01 else 0.0)
        w11 = fx * fy * (1.0 if v11 else 0.0)

        total_weight = w00 + w10 + w01 + w11

        if total_weight < 1e-6:
            return Vec3.zero(), 0.0

        # Weighted average
        result = Vec3(
            (p00.x * w00 + p10.x * w10 + p01.x * w01 + p11.x * w11) / total_weight,
            (p00.y * w00 + p10.y * w10 + p01.y * w01 + p11.y * w11) / total_weight,
            (p00.z * w00 + p10.z * w10 + p01.z * w01 + p11.z * w11) / total_weight,
        )

        return result, total_weight

    def clear(self) -> None:
        """Clear all positions to invalid."""
        for i in range(len(self.positions)):
            self.positions[i] = Vec3.zero()
            self.valid[i] = False

    def copy_from(self, other: "HitPositionBuffer") -> None:
        """Copy data from another buffer."""
        if self.width != other.width or self.height != other.height:
            raise ValueError("Buffer dimensions must match for copy")
        for i in range(len(self.positions)):
            self.positions[i] = Vec3(
                other.positions[i].x,
                other.positions[i].y,
                other.positions[i].z,
            )
            self.valid[i] = other.valid[i]

    def clone(self) -> "HitPositionBuffer":
        """Create a deep copy of this buffer."""
        new_buf = HitPositionBuffer(self.width, self.height)
        new_buf.copy_from(self)
        return new_buf


# =============================================================================
# Color Space Conversion (YCoCg)
# =============================================================================


def rgb_to_ycocg(color: Vec4) -> Vec4:
    """Convert RGB color to YCoCg color space.

    YCoCg provides better perceptual uniformity for color clamping.
    Y = luminance, Co = orange chroma, Cg = green chroma.
    """
    y = 0.25 * color.x + 0.5 * color.y + 0.25 * color.z
    co = 0.5 * color.x - 0.5 * color.z
    cg = -0.25 * color.x + 0.5 * color.y - 0.25 * color.z
    return Vec4(y, co, cg, color.w)


def ycocg_to_rgb(ycocg: Vec4) -> Vec4:
    """Convert YCoCg color back to RGB."""
    y, co, cg = ycocg.x, ycocg.y, ycocg.z
    r = y + co - cg
    g = y + cg
    b = y - co - cg
    return Vec4(r, g, b, ycocg.w)


# =============================================================================
# Reprojection Functions
# =============================================================================


def project_to_screen(
    world_pos: Vec3,
    view_proj: Mat4,
    width: int,
    height: int,
) -> Tuple[Vec2, float, bool]:
    """Project a world-space position to screen UV coordinates.

    Args:
        world_pos: World-space position to project.
        view_proj: Combined view-projection matrix.
        width: Screen width in pixels.
        height: Screen height in pixels.

    Returns:
        Tuple of (uv, depth, is_valid).
        UV is in [0, 1] range, depth is linear depth, is_valid is False
        if the point is behind the camera.
    """
    # Transform to clip space
    clip = Vec4(world_pos.x, world_pos.y, world_pos.z, 1.0)
    m = view_proj.m

    cx = m[0]*clip.x + m[4]*clip.y + m[8]*clip.z + m[12]
    cy = m[1]*clip.x + m[5]*clip.y + m[9]*clip.z + m[13]
    cz = m[2]*clip.x + m[6]*clip.y + m[10]*clip.z + m[14]
    cw = m[3]*clip.x + m[7]*clip.y + m[11]*clip.z + m[15]

    # Behind camera check
    if cw <= 0:
        return Vec2(0.5, 0.5), 0.0, False

    # Perspective divide to NDC
    ndc_x = cx / cw
    ndc_y = cy / cw
    ndc_z = cz / cw

    # NDC to UV ([-1, 1] -> [0, 1])
    u = (ndc_x + 1.0) * 0.5
    v = (ndc_y + 1.0) * 0.5

    # Check if within screen bounds
    is_valid = 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0

    return Vec2(u, v), ndc_z, is_valid


def calculate_reprojected_uv(
    current_hit_pos: Vec3,
    prev_view_proj: Mat4,
    width: int,
    height: int,
) -> Tuple[Vec2, bool]:
    """Calculate where a world-space hit would appear in the previous frame.

    This is the core of TAA reprojection for ray marching: we use the
    world-space hit position (which is stable) to find where that same
    surface point was visible in the previous frame.

    Args:
        current_hit_pos: World-space hit position from current frame.
        prev_view_proj: Previous frame's view-projection matrix.
        width: Screen width.
        height: Screen height.

    Returns:
        Tuple of (prev_uv, is_valid).
    """
    prev_uv, _, is_valid = project_to_screen(
        current_hit_pos, prev_view_proj, width, height
    )
    return prev_uv, is_valid


def detect_disocclusion_depth(
    current_depth: float,
    reprojected_depth: float,
    threshold: float,
) -> bool:
    """Detect disocclusion by comparing depth values.

    Args:
        current_depth: Depth at current pixel.
        reprojected_depth: Depth from reprojected history.
        threshold: Relative depth difference threshold.

    Returns:
        True if disoccluded (should reject history).
    """
    if current_depth <= 0 or reprojected_depth <= 0:
        return True

    rel_diff = abs(current_depth - reprojected_depth) / max(current_depth, reprojected_depth)
    return rel_diff > threshold


def detect_disocclusion_position(
    current_pos: Vec3,
    reprojected_pos: Vec3,
    threshold: float,
) -> bool:
    """Detect disocclusion by comparing world positions.

    Args:
        current_pos: Current frame's hit position.
        reprojected_pos: Position from reprojected history.
        threshold: World-space distance threshold.

    Returns:
        True if disoccluded (should reject history).
    """
    dist = current_pos.distance(reprojected_pos)
    return dist > threshold


# =============================================================================
# Neighborhood Clamping
# =============================================================================


def compute_neighborhood_bounds_rgb(
    current_tex: "ColorBuffer",
    x: int,
    y: int,
    radius: int = 1,
) -> Tuple[Vec4, Vec4]:
    """Compute min/max color bounds in RGB space from neighborhood.

    Args:
        current_tex: Current frame color buffer.
        x, y: Center pixel coordinates.
        radius: Neighborhood radius (1 = 3x3).

    Returns:
        Tuple of (min_color, max_color).
    """
    min_color = Vec4(float('inf'), float('inf'), float('inf'), 1.0)
    max_color = Vec4(float('-inf'), float('-inf'), float('-inf'), 1.0)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            c = current_tex.get_pixel(x + dx, y + dy)
            min_color = Vec4(
                min(min_color.x, c.x),
                min(min_color.y, c.y),
                min(min_color.z, c.z),
                1.0,
            )
            max_color = Vec4(
                max(max_color.x, c.x),
                max(max_color.y, c.y),
                max(max_color.z, c.z),
                1.0,
            )

    return min_color, max_color


def compute_neighborhood_bounds_ycocg(
    current_tex: "ColorBuffer",
    x: int,
    y: int,
    radius: int = 1,
) -> Tuple[Vec4, Vec4]:
    """Compute min/max color bounds in YCoCg space from neighborhood.

    YCoCg provides better perceptual bounds for clamping.
    """
    min_ycocg = Vec4(float('inf'), float('inf'), float('inf'), 1.0)
    max_ycocg = Vec4(float('-inf'), float('-inf'), float('-inf'), 1.0)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            c = current_tex.get_pixel(x + dx, y + dy)
            yc = rgb_to_ycocg(c)
            min_ycocg = Vec4(
                min(min_ycocg.x, yc.x),
                min(min_ycocg.y, yc.y),
                min(min_ycocg.z, yc.z),
                1.0,
            )
            max_ycocg = Vec4(
                max(max_ycocg.x, yc.x),
                max(max_ycocg.y, yc.y),
                max(max_ycocg.z, yc.z),
                1.0,
            )

    return min_ycocg, max_ycocg


def compute_neighborhood_variance(
    current_tex: "ColorBuffer",
    x: int,
    y: int,
    radius: int = 1,
) -> Tuple[Vec4, Vec4, Vec4]:
    """Compute mean and variance of neighborhood colors.

    Returns:
        Tuple of (mean, variance, std_dev).
    """
    samples = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            c = current_tex.get_pixel(x + dx, y + dy)
            samples.append(c)

    n = len(samples)

    # Mean
    mean = Vec4(0, 0, 0, 1)
    for c in samples:
        mean = Vec4(mean.x + c.x, mean.y + c.y, mean.z + c.z, 1.0)
    mean = Vec4(mean.x / n, mean.y / n, mean.z / n, 1.0)

    # Variance
    var = Vec4(0, 0, 0, 1)
    for c in samples:
        dx = c.x - mean.x
        dy = c.y - mean.y
        dz = c.z - mean.z
        var = Vec4(var.x + dx*dx, var.y + dy*dy, var.z + dz*dz, 1.0)
    var = Vec4(var.x / n, var.y / n, var.z / n, 1.0)

    # Standard deviation
    std = Vec4(math.sqrt(var.x), math.sqrt(var.y), math.sqrt(var.z), 1.0)

    return mean, var, std


def clamp_color_rgb(
    color: Vec4,
    min_color: Vec4,
    max_color: Vec4,
) -> Vec4:
    """Clamp color to bounds in RGB space."""
    return Vec4(
        max(min_color.x, min(max_color.x, color.x)),
        max(min_color.y, min(max_color.y, color.y)),
        max(min_color.z, min(max_color.z, color.z)),
        color.w,
    )


def clamp_color_ycocg(
    color: Vec4,
    min_ycocg: Vec4,
    max_ycocg: Vec4,
) -> Vec4:
    """Clamp color to bounds in YCoCg space.

    Converts to YCoCg, clamps, then converts back to RGB.
    """
    yc = rgb_to_ycocg(color)
    clamped = Vec4(
        max(min_ycocg.x, min(max_ycocg.x, yc.x)),
        max(min_ycocg.y, min(max_ycocg.y, yc.y)),
        max(min_ycocg.z, min(max_ycocg.z, yc.z)),
        yc.w,
    )
    return ycocg_to_rgb(clamped)


def clamp_color_variance(
    color: Vec4,
    mean: Vec4,
    std: Vec4,
    gamma: float = 1.5,
) -> Vec4:
    """Clamp color using variance-based bounds.

    Bounds are mean +/- gamma * std_dev.
    """
    min_c = Vec4(
        mean.x - gamma * std.x,
        mean.y - gamma * std.y,
        mean.z - gamma * std.z,
        1.0,
    )
    max_c = Vec4(
        mean.x + gamma * std.x,
        mean.y + gamma * std.y,
        mean.z + gamma * std.z,
        1.0,
    )
    return clamp_color_rgb(color, min_c, max_c)


# =============================================================================
# Color Buffer (for CPU-side TAA testing)
# =============================================================================


@dataclass
class ColorBuffer:
    """Color buffer for storing rendered frames.

    Similar to Texture from temporal_aa.py but optimized for TAA reprojection.
    """

    width: int
    height: int
    data: List[Vec4] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Initialize pixel data if not provided."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid dimensions: {self.width}x{self.height}")
        if not self.data:
            self.data = [Vec4.zero() for _ in range(self.width * self.height)]

    def get_pixel(self, x: int, y: int) -> Vec4:
        """Get pixel value at (x, y), returning black for out-of-bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y * self.width + x]
        return Vec4.zero()

    def set_pixel(self, x: int, y: int, color: Vec4) -> None:
        """Set pixel value at (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y * self.width + x] = color

    def sample_bilinear(self, uv: Vec2) -> Vec4:
        """Sample with bilinear interpolation at UV coordinates."""
        px = uv.x * self.width - 0.5
        py = uv.y * self.height - 0.5

        x0 = int(math.floor(px))
        y0 = int(math.floor(py))
        x1 = x0 + 1
        y1 = y0 + 1

        fx = px - x0
        fy = py - y0

        c00 = self.get_pixel(x0, y0)
        c10 = self.get_pixel(x1, y0)
        c01 = self.get_pixel(x0, y1)
        c11 = self.get_pixel(x1, y1)

        c0 = c00.lerp(c10, fx)
        c1 = c01.lerp(c11, fx)
        return c0.lerp(c1, fy)

    def clear(self, color: Vec4 = None) -> None:
        """Clear buffer to a solid color."""
        fill = color if color else Vec4.zero()
        for i in range(len(self.data)):
            self.data[i] = Vec4(fill.x, fill.y, fill.z, fill.w)

    def copy_from(self, other: "ColorBuffer") -> None:
        """Copy data from another buffer."""
        if self.width != other.width or self.height != other.height:
            raise ValueError("Buffer dimensions must match")
        for i in range(len(self.data)):
            src = other.data[i]
            self.data[i] = Vec4(src.x, src.y, src.z, src.w)

    def clone(self) -> "ColorBuffer":
        """Create a deep copy."""
        new_buf = ColorBuffer(self.width, self.height)
        new_buf.copy_from(self)
        return new_buf


# =============================================================================
# TAA Reprojection Main Class
# =============================================================================


class TAAReprojection:
    """
    Temporal Anti-Aliasing using world-space hit position reprojection.

    This is specifically designed for ray marching / SDF rendering where
    we have world-space hit positions available but not traditional
    per-object motion vectors.

    The algorithm:
    1. For each pixel, get current hit position
    2. Project that position to previous frame's screen space
    3. Sample history buffer at that location
    4. Clamp history to current neighborhood bounds (anti-ghosting)
    5. Blend current + clamped_history with adaptive factor

    Example:
        >>> taa = TAAReprojection(1920, 1080)
        >>> # After ray marching frame N:
        >>> hit_buffer.set_hit(x, y, world_pos)
        >>> # Apply TAA
        >>> result = taa.accumulate(
        ...     current_color=render_result,
        ...     current_hit_positions=hit_buffer,
        ...     current_view_proj=view_proj,
        ...     prev_view_proj=last_frame_view_proj,
        ... )
    """

    def __init__(
        self,
        width: int,
        height: int,
        config: Optional[ReprojectionConfig] = None,
    ) -> None:
        """Initialize TAA reprojection system.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            config: Optional configuration (uses defaults if None).
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")

        self.width = width
        self.height = height
        self.config = config if config else ReprojectionConfig()

        # History buffers
        self._history_color = ColorBuffer(width, height)
        self._history_positions = HitPositionBuffer(width, height)

        # Frame tracking
        self._frame_count = 0
        self._converged = False

    @property
    def frame_count(self) -> int:
        """Number of frames accumulated."""
        return self._frame_count

    @property
    def is_converged(self) -> bool:
        """Whether accumulation has converged."""
        return self._converged

    def reset(self) -> None:
        """Reset all history buffers."""
        self._history_color.clear()
        self._history_positions.clear()
        self._frame_count = 0
        self._converged = False

    def resize(self, width: int, height: int) -> None:
        """Resize buffers, clearing history."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")
        self.width = width
        self.height = height
        self._history_color = ColorBuffer(width, height)
        self._history_positions = HitPositionBuffer(width, height)
        self.reset()

    def _compute_blend_factor(
        self,
        is_disoccluded: bool,
        velocity_magnitude: float = 0.0,
    ) -> float:
        """Compute adaptive blend factor for a pixel.

        Args:
            is_disoccluded: Whether the pixel is disoccluded.
            velocity_magnitude: Screen-space velocity magnitude.

        Returns:
            Blend factor in [min_blend, max_blend].
        """
        cfg = self.config

        if self._frame_count == 0:
            return 1.0  # First frame: use current directly

        if is_disoccluded:
            return cfg.max_blend  # Fast reset

        # Base blend
        blend = cfg.blend_factor

        # Adaptive based on frame count (faster initial convergence)
        adaptive = 1.0 / (self._frame_count + 1)
        blend = max(blend, adaptive)

        # Velocity weighting (optional)
        if cfg.enable_velocity_weighting and velocity_magnitude > 0.01:
            # Higher blend for fast-moving pixels
            velocity_factor = min(1.0, velocity_magnitude * 2.0)
            blend = blend + (cfg.max_blend - blend) * velocity_factor

        return max(cfg.min_blend, min(cfg.max_blend, blend))

    def _clamp_history(
        self,
        history_color: Vec4,
        current: ColorBuffer,
        x: int,
        y: int,
    ) -> Vec4:
        """Clamp history color to current neighborhood bounds.

        This is the key anti-ghosting mechanism in TAA.
        """
        cfg = self.config
        radius = cfg.neighborhood_size

        if cfg.clamping_mode == ClampingMode.NONE:
            return history_color

        elif cfg.clamping_mode == ClampingMode.RGB:
            min_c, max_c = compute_neighborhood_bounds_rgb(current, x, y, radius)
            return clamp_color_rgb(history_color, min_c, max_c)

        elif cfg.clamping_mode == ClampingMode.YCOCG:
            min_yc, max_yc = compute_neighborhood_bounds_ycocg(current, x, y, radius)
            return clamp_color_ycocg(history_color, min_yc, max_yc)

        elif cfg.clamping_mode == ClampingMode.VARIANCE:
            mean, var, std = compute_neighborhood_variance(current, x, y, radius)
            return clamp_color_variance(history_color, mean, std)

        return history_color

    def _is_disoccluded(
        self,
        current_pos: Vec3,
        current_valid: bool,
        reprojected_pos: Vec3,
        reprojected_valid: bool,
        current_depth: float,
        reprojected_depth: float,
    ) -> bool:
        """Check if a pixel is disoccluded.

        Disocclusion occurs when:
        - Current pixel has no valid hit but history does
        - History pixel has no valid hit
        - World positions are too far apart
        - Depth values differ significantly
        """
        cfg = self.config

        if cfg.disocclusion_mode == DisocclusionMode.NONE:
            return False

        # Invalid hit in current or history
        if not current_valid or not reprojected_valid:
            return True

        if cfg.disocclusion_mode == DisocclusionMode.DEPTH:
            return detect_disocclusion_depth(
                current_depth, reprojected_depth, cfg.depth_threshold
            )

        elif cfg.disocclusion_mode == DisocclusionMode.POSITION:
            return detect_disocclusion_position(
                current_pos, reprojected_pos, cfg.disocclusion_threshold
            )

        elif cfg.disocclusion_mode == DisocclusionMode.COMBINED:
            depth_reject = detect_disocclusion_depth(
                current_depth, reprojected_depth, cfg.depth_threshold
            )
            pos_reject = detect_disocclusion_position(
                current_pos, reprojected_pos, cfg.disocclusion_threshold
            )
            return depth_reject or pos_reject

        return False

    def accumulate(
        self,
        current_color: ColorBuffer,
        current_hit_positions: HitPositionBuffer,
        current_view_proj: Mat4,
        prev_view_proj: Mat4,
    ) -> ColorBuffer:
        """Accumulate current frame with history using reprojection.

        This is the main TAA function. For each pixel:
        1. Get current hit position
        2. Reproject to previous screen space
        3. Sample and clamp history
        4. Blend with current color

        Args:
            current_color: Current frame's rendered colors.
            current_hit_positions: Current frame's hit positions.
            current_view_proj: Current frame's view-projection matrix.
            prev_view_proj: Previous frame's view-projection matrix.

        Returns:
            Accumulated color buffer.
        """
        if current_color.width != self.width or current_color.height != self.height:
            raise ValueError(
                f"Current color buffer {current_color.width}x{current_color.height} "
                f"doesn't match TAA {self.width}x{self.height}"
            )

        output = ColorBuffer(self.width, self.height)

        for y in range(self.height):
            for x in range(self.width):
                # Get current pixel data
                current_c = current_color.get_pixel(x, y)
                current_pos, current_valid = current_hit_positions.get_position(x, y)

                # First frame: use current directly
                if self._frame_count == 0:
                    output.set_pixel(x, y, current_c)
                    continue

                # Calculate reprojection
                if current_valid:
                    prev_uv, uv_valid = calculate_reprojected_uv(
                        current_pos, prev_view_proj, self.width, self.height
                    )
                else:
                    # No hit: use current pixel UV
                    prev_uv = Vec2((x + 0.5) / self.width, (y + 0.5) / self.height)
                    uv_valid = True

                # Sample history
                if uv_valid:
                    history_c = self._history_color.sample_bilinear(prev_uv)
                    history_pos, history_weight = self._history_positions.sample_bilinear(prev_uv)
                    history_valid = history_weight > 0.5
                else:
                    history_c = Vec4.zero()
                    history_pos = Vec3.zero()
                    history_valid = False

                # Compute depths for disocclusion check
                current_depth = current_pos.z if current_valid else 0.0
                history_depth = history_pos.z if history_valid else 0.0

                # Check disocclusion
                is_disoccluded = self._is_disoccluded(
                    current_pos, current_valid,
                    history_pos, history_valid,
                    current_depth, history_depth,
                )

                # Clamp history to reduce ghosting
                if not is_disoccluded:
                    history_c = self._clamp_history(history_c, current_color, x, y)

                # Compute blend factor
                velocity = 0.0  # Could compute from UV difference
                if uv_valid:
                    current_uv = Vec2((x + 0.5) / self.width, (y + 0.5) / self.height)
                    velocity = current_uv.distance(prev_uv)

                blend = self._compute_blend_factor(is_disoccluded, velocity)

                # Blend
                result = history_c.lerp(current_c, blend)
                output.set_pixel(x, y, result)

        # Update history
        self._history_color.copy_from(output)
        self._history_positions.copy_from(current_hit_positions)
        self._frame_count += 1

        # Check convergence
        if self._frame_count >= int(1.0 / self.config.blend_factor):
            self._converged = True

        return output

    def get_history_color(self) -> ColorBuffer:
        """Get the current history color buffer."""
        return self._history_color.clone()

    def get_history_positions(self) -> HitPositionBuffer:
        """Get the current history positions buffer."""
        return self._history_positions.clone()


# =============================================================================
# WGSL Code Generation
# =============================================================================


def generate_reprojection_wgsl() -> str:
    """Generate WGSL code for world-space reprojection.

    Returns:
        WGSL function for reprojecting hit positions.
    """
    return """\
/// Projects a world-space position to screen UV coordinates.
/// Returns vec4(uv.x, uv.y, depth, valid) where valid is 0.0 or 1.0.
fn project_to_screen(
    world_pos: vec3<f32>,
    view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
) -> vec4<f32> {
    // Transform to clip space
    let clip = view_proj * vec4<f32>(world_pos, 1.0);

    // Behind camera check
    if (clip.w <= 0.0) {
        return vec4<f32>(0.5, 0.5, 0.0, 0.0);
    }

    // Perspective divide to NDC
    let ndc = clip.xyz / clip.w;

    // NDC to UV ([-1, 1] -> [0, 1])
    let uv = (ndc.xy + 1.0) * 0.5;

    // Check bounds
    let in_bounds = all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
    let valid = select(0.0, 1.0, in_bounds);

    return vec4<f32>(uv.x, uv.y, ndc.z, valid);
}

/// Calculates the reprojected UV for TAA.
/// Uses world-space hit position instead of motion vectors.
fn calculate_reprojected_uv(
    current_hit_pos: vec3<f32>,
    prev_view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
) -> vec4<f32> {
    return project_to_screen(current_hit_pos, prev_view_proj, resolution);
}
"""


def generate_ycocg_wgsl() -> str:
    """Generate WGSL code for YCoCg color space conversion."""
    return """\
/// Converts RGB to YCoCg color space.
fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let y = 0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b;
    let co = 0.5 * rgb.r - 0.5 * rgb.b;
    let cg = -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b;
    return vec3<f32>(y, co, cg);
}

/// Converts YCoCg back to RGB.
fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let y = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;
    return vec3<f32>(r, g, b);
}
"""


def generate_neighborhood_clamping_wgsl() -> str:
    """Generate WGSL code for neighborhood clamping in YCoCg space."""
    return """\
/// Computes neighborhood min/max in YCoCg space for anti-ghosting.
fn compute_neighborhood_bounds_ycocg(
    current_texture: texture_2d<f32>,
    pixel: vec2<i32>,
    radius: i32,
) -> array<vec3<f32>, 2> {
    var min_yc = vec3<f32>(1e10);
    var max_yc = vec3<f32>(-1e10);

    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let sample_pos = pixel + vec2<i32>(dx, dy);
            let rgb = textureLoad(current_texture, sample_pos, 0).rgb;
            let yc = rgb_to_ycocg(rgb);
            min_yc = min(min_yc, yc);
            max_yc = max(max_yc, yc);
        }
    }

    return array<vec3<f32>, 2>(min_yc, max_yc);
}

/// Clamps color in YCoCg space to neighborhood bounds.
fn clamp_history_ycocg(
    history_rgb: vec3<f32>,
    min_yc: vec3<f32>,
    max_yc: vec3<f32>,
) -> vec3<f32> {
    let yc = rgb_to_ycocg(history_rgb);
    let clamped = clamp(yc, min_yc, max_yc);
    return ycocg_to_rgb(clamped);
}
"""


def generate_disocclusion_wgsl() -> str:
    """Generate WGSL code for disocclusion detection."""
    return """\
/// Detects disocclusion by comparing world positions.
fn detect_disocclusion_position(
    current_pos: vec3<f32>,
    history_pos: vec3<f32>,
    threshold: f32,
) -> bool {
    let dist = distance(current_pos, history_pos);
    return dist > threshold;
}

/// Detects disocclusion by comparing depths.
fn detect_disocclusion_depth(
    current_depth: f32,
    history_depth: f32,
    threshold: f32,
) -> bool {
    let max_depth = max(current_depth, history_depth);
    if (max_depth <= 0.0) {
        return true;
    }
    let rel_diff = abs(current_depth - history_depth) / max_depth;
    return rel_diff > threshold;
}
"""


def generate_taa_reprojection_wgsl() -> str:
    """Generate complete WGSL code for TAA reprojection compute shader."""
    return f"""\
{generate_ycocg_wgsl()}

{generate_reprojection_wgsl()}

{generate_disocclusion_wgsl()}

{generate_neighborhood_clamping_wgsl()}

/// TAA parameters uniform buffer.
struct TAAReprojectionParams {{
    blend_factor: f32,
    disocclusion_threshold: f32,
    depth_threshold: f32,
    frame_count: u32,
    prev_view_proj: mat4x4<f32>,
    current_view_proj: mat4x4<f32>,
}}

@group(0) @binding(0) var current_color: texture_2d<f32>;
@group(0) @binding(1) var current_hit_positions: texture_2d<f32>;  // RGB32F
@group(0) @binding(2) var history_color: texture_2d<f32>;
@group(0) @binding(3) var history_positions: texture_2d<f32>;
@group(0) @binding(4) var output_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> params: TAAReprojectionParams;
@group(0) @binding(6) var bilinear_sampler: sampler;

/// Main TAA reprojection compute shader.
@compute @workgroup_size(8, 8, 1)
fn taa_reprojection(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let dimensions = textureDimensions(current_color);
    let pixel = vec2<i32>(global_id.xy);

    // Bounds check
    if (pixel.x >= i32(dimensions.x) || pixel.y >= i32(dimensions.y)) {{
        return;
    }}

    let resolution = vec2<f32>(dimensions);
    let uv = (vec2<f32>(pixel) + 0.5) / resolution;

    // Load current frame data
    let current_c = textureLoad(current_color, pixel, 0);
    let current_pos_data = textureLoad(current_hit_positions, pixel, 0);
    let current_pos = current_pos_data.xyz;
    let current_valid = current_pos_data.w > 0.5;

    var result: vec4<f32>;

    // First frame: use current directly
    if (params.frame_count == 0u) {{
        result = current_c;
    }} else {{
        // Calculate reprojection
        var prev_uv = uv;
        var uv_valid = false;

        if (current_valid) {{
            let reproj = calculate_reprojected_uv(current_pos, params.prev_view_proj, resolution);
            prev_uv = reproj.xy;
            uv_valid = reproj.w > 0.5;
        }}

        // Sample history with bilinear
        var history_c: vec4<f32>;
        var history_pos: vec3<f32>;
        var history_valid = false;

        if (uv_valid) {{
            history_c = textureSampleLevel(history_color, bilinear_sampler, prev_uv, 0.0);
            let history_pos_data = textureSampleLevel(history_positions, bilinear_sampler, prev_uv, 0.0);
            history_pos = history_pos_data.xyz;
            history_valid = history_pos_data.w > 0.5;
        }} else {{
            history_c = vec4<f32>(0.0);
            history_pos = vec3<f32>(0.0);
        }}

        // Disocclusion detection
        let depth_reject = detect_disocclusion_depth(
            current_pos.z, history_pos.z, params.depth_threshold
        );
        let pos_reject = detect_disocclusion_position(
            current_pos, history_pos, params.disocclusion_threshold
        );
        let is_disoccluded = !current_valid || !history_valid || depth_reject || pos_reject;

        // Neighborhood clamping (anti-ghosting)
        var clamped_history = history_c.rgb;
        if (!is_disoccluded) {{
            let bounds = compute_neighborhood_bounds_ycocg(current_color, pixel, 1);
            clamped_history = clamp_history_ycocg(history_c.rgb, bounds[0], bounds[1]);
        }}

        // Adaptive blend factor
        var blend = params.blend_factor;
        if (is_disoccluded) {{
            blend = 0.8;  // Fast reset
        }} else {{
            let adaptive = 1.0 / f32(params.frame_count + 1u);
            blend = max(blend, adaptive);
        }}
        blend = clamp(blend, 0.02, 0.8);

        // Final blend
        result = vec4<f32>(
            mix(clamped_history, current_c.rgb, blend),
            current_c.a
        );
    }}

    textureStore(output_color, pixel, result);
}}
"""


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Configuration
    "ReprojectionConfig",
    "DisocclusionMode",
    "ClampingMode",
    # Buffers
    "HitPositionBuffer",
    "ColorBuffer",
    # Color space conversion
    "rgb_to_ycocg",
    "ycocg_to_rgb",
    # Reprojection functions
    "project_to_screen",
    "calculate_reprojected_uv",
    "detect_disocclusion_depth",
    "detect_disocclusion_position",
    # Clamping functions
    "compute_neighborhood_bounds_rgb",
    "compute_neighborhood_bounds_ycocg",
    "compute_neighborhood_variance",
    "clamp_color_rgb",
    "clamp_color_ycocg",
    "clamp_color_variance",
    # Main class
    "TAAReprojection",
    # WGSL generation
    "generate_reprojection_wgsl",
    "generate_ycocg_wgsl",
    "generate_neighborhood_clamping_wgsl",
    "generate_disocclusion_wgsl",
    "generate_taa_reprojection_wgsl",
]
