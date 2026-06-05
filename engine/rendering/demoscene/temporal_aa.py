"""
Temporal Anti-Aliasing via Sub-Pixel Jitter (T-DEMO-3.13).

This module implements temporal anti-aliasing for demoscene rendering using
sub-pixel jitter and frame accumulation. The technique works by:

1. Each frame, offset the camera by a sub-pixel jitter from a low-discrepancy sequence
2. Accumulate rendered frames with an exponential moving average
3. On camera movement, reset the accumulation history

The Halton sequence (bases 2 and 3) provides excellent sub-pixel coverage with
minimal clustering, ensuring smooth convergence over ~16 frames.

Usage:
    >>> from engine.rendering.demoscene.temporal_aa import (
    ...     halton_sequence, get_jitter, TemporalAccumulator, JitterSequence
    ... )
    >>> # Get jitter for frame 5
    >>> jitter = get_jitter(5, sequence_length=16)
    >>> # Offset UV coordinates
    >>> uv_jittered = (uv.x + jitter.x / width, uv.y + jitter.y / height)
    >>> # Generate ray with jittered UV
    >>> ray = generate_ray(uv_jittered, camera)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

from engine.core.math.vec import Vec2, Vec3, Vec4


# =============================================================================
# Halton Sequence Implementation
# =============================================================================


def halton_sequence(index: int, base: int) -> float:
    """
    Compute the Halton sequence value at a given index and base.

    The Halton sequence is a low-discrepancy sequence that provides
    well-distributed samples in the unit interval [0, 1). It is constructed
    by radical inversion: reversing the digits of the index in the given base.

    Args:
        index: The sequence index (0-based). Must be >= 0.
        base: The base for the sequence. Must be >= 2 (typically prime: 2, 3, 5, 7...).

    Returns:
        A float in the range [0, 1) representing the sequence value.

    Example:
        >>> halton_sequence(0, 2)  # First value, base 2
        0.0
        >>> halton_sequence(1, 2)  # Second value, base 2
        0.5
        >>> halton_sequence(2, 2)  # Third value, base 2
        0.25
        >>> halton_sequence(1, 3)  # Second value, base 3
        0.3333...

    Note:
        The Halton sequence provides better coverage than random sampling
        for quasi-Monte Carlo integration. For 2D jitter, use bases 2 and 3.
    """
    if index < 0:
        raise ValueError(f"Index must be non-negative, got {index}")
    if base < 2:
        raise ValueError(f"Base must be >= 2, got {base}")

    result = 0.0
    f = 1.0 / base
    i = index

    while i > 0:
        result += f * (i % base)
        i = i // base
        f /= base

    return result


def halton_2d(index: int) -> Vec2:
    """
    Compute a 2D Halton point using bases 2 and 3.

    This is the standard choice for 2D quasi-Monte Carlo sampling,
    providing excellent distribution properties for image sampling.

    Args:
        index: The sequence index (0-based).

    Returns:
        Vec2 with x in [0, 1) from base-2 and y in [0, 1) from base-3.
    """
    return Vec2(halton_sequence(index, 2), halton_sequence(index, 3))


# =============================================================================
# Jitter Sequence Management
# =============================================================================


class JitterPattern(Enum):
    """Available jitter patterns for temporal anti-aliasing."""

    HALTON = auto()
    """Halton sequence (2, 3) - excellent low-discrepancy coverage."""

    HALTON_ROTATED = auto()
    """Halton with per-frame rotation to reduce correlation."""

    UNIFORM_GRID = auto()
    """Regular N x N grid sampling."""

    INTERLEAVED = auto()
    """Interleaved gradient pattern for GPU-friendly sampling."""


@dataclass
class JitterSequence:
    """
    Manages a sequence of sub-pixel jitter offsets.

    The jitter sequence provides reproducible sub-pixel offsets for
    temporal anti-aliasing. Each frame samples a different point within
    a pixel, and accumulation converges to a supersampled result.

    Attributes:
        pattern: The jitter pattern to use.
        sequence_length: Number of unique jitter positions (power of 2 recommended).
        scale: Multiplier for jitter magnitude (default 1.0 = one pixel).
    """

    pattern: JitterPattern = JitterPattern.HALTON
    sequence_length: int = 16
    scale: float = 1.0
    _cached_sequence: List[Vec2] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Pre-compute the jitter sequence."""
        if self.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {self.sequence_length}")
        self._build_sequence()

    def _build_sequence(self) -> None:
        """Build the jitter sequence based on the pattern."""
        self._cached_sequence = []

        if self.pattern == JitterPattern.HALTON:
            for i in range(self.sequence_length):
                # Map [0, 1) to [-0.5, 0.5) for centered jitter
                h = halton_2d(i)
                jitter = Vec2(
                    (h.x - 0.5) * self.scale,
                    (h.y - 0.5) * self.scale,
                )
                self._cached_sequence.append(jitter)

        elif self.pattern == JitterPattern.HALTON_ROTATED:
            # Apply a golden ratio rotation to each frame's Halton point
            golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ~137.5 degrees
            for i in range(self.sequence_length):
                h = halton_2d(i)
                # Rotate the point by i * golden_angle
                angle = i * golden_angle
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                centered_x = h.x - 0.5
                centered_y = h.y - 0.5
                rotated_x = centered_x * cos_a - centered_y * sin_a
                rotated_y = centered_x * sin_a + centered_y * cos_a
                jitter = Vec2(rotated_x * self.scale, rotated_y * self.scale)
                self._cached_sequence.append(jitter)

        elif self.pattern == JitterPattern.UNIFORM_GRID:
            # Compute grid size from sequence length
            grid_size = int(math.ceil(math.sqrt(self.sequence_length)))
            step = 1.0 / grid_size
            for i in range(self.sequence_length):
                x = (i % grid_size) * step + step * 0.5 - 0.5
                y = (i // grid_size) * step + step * 0.5 - 0.5
                jitter = Vec2(x * self.scale, y * self.scale)
                self._cached_sequence.append(jitter)

        elif self.pattern == JitterPattern.INTERLEAVED:
            # Interleaved gradient pattern (2x2 rotated grid per 4 frames)
            offsets_2x2 = [
                Vec2(-0.25, -0.25),
                Vec2(0.25, -0.25),
                Vec2(-0.25, 0.25),
                Vec2(0.25, 0.25),
            ]
            for i in range(self.sequence_length):
                base_offset = offsets_2x2[i % 4]
                # Add small variation based on frame index
                variation = halton_2d(i // 4) * 0.125 - Vec2(0.0625, 0.0625)
                jitter = (base_offset + variation) * self.scale
                self._cached_sequence.append(jitter)

    def get_jitter(self, frame: int) -> Vec2:
        """
        Get the jitter offset for a given frame.

        Args:
            frame: The frame number (will be wrapped to sequence length).

        Returns:
            Vec2 with x and y jitter offsets in pixel units.
        """
        index = frame % self.sequence_length
        return self._cached_sequence[index]

    def reset(self) -> None:
        """Reset the sequence (rebuild cache)."""
        self._build_sequence()


def get_jitter(frame: int, sequence_length: int = 16) -> Vec2:
    """
    Get the sub-pixel jitter offset for a given frame.

    This is a convenience function that uses the Halton sequence with
    default settings. For more control, use JitterSequence directly.

    Args:
        frame: The current frame number.
        sequence_length: Number of unique jitter positions (default 16).

    Returns:
        Vec2 with x and y jitter offsets in [-0.5, 0.5) pixel units.

    Example:
        >>> jitter = get_jitter(0, 16)
        >>> jitter.x, jitter.y  # Centered at pixel for frame 0
        (-0.5, -0.5)
        >>> jitter = get_jitter(1, 16)
        >>> # Second frame: different sub-pixel position
    """
    index = frame % sequence_length
    h = halton_2d(index)
    return Vec2(h.x - 0.5, h.y - 0.5)


# =============================================================================
# Texture Wrapper (Lightweight abstraction for accumulation)
# =============================================================================


@dataclass
class Texture:
    """
    Lightweight texture representation for temporal accumulation.

    This is a simple container that holds pixel data for CPU-side
    temporal accumulation testing. In production, this would be
    replaced with actual GPU texture handles.

    Attributes:
        width: Texture width in pixels.
        height: Texture height in pixels.
        data: Flat list of Vec4 pixel values (RGBA).
    """

    width: int
    height: int
    data: List[Vec4] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Initialize pixel data if not provided."""
        if not self.data:
            self.data = [Vec4.zero() for _ in range(self.width * self.height)]

    def get_pixel(self, x: int, y: int) -> Vec4:
        """Get pixel value at (x, y)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return Vec4.zero()
        return self.data[y * self.width + x]

    def set_pixel(self, x: int, y: int, color: Vec4) -> None:
        """Set pixel value at (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y * self.width + x] = color

    def sample(self, uv: Vec2) -> Vec4:
        """
        Sample texture at normalized UV coordinates with bilinear filtering.

        Args:
            uv: Texture coordinates in [0, 1] range.

        Returns:
            Bilinearly interpolated pixel value.
        """
        # Convert UV to pixel coordinates
        px = uv.x * self.width - 0.5
        py = uv.y * self.height - 0.5

        # Get integer coordinates
        x0 = int(math.floor(px))
        y0 = int(math.floor(py))
        x1 = x0 + 1
        y1 = y0 + 1

        # Get fractional part
        fx = px - x0
        fy = py - y0

        # Sample 4 texels
        c00 = self.get_pixel(x0, y0)
        c10 = self.get_pixel(x1, y0)
        c01 = self.get_pixel(x0, y1)
        c11 = self.get_pixel(x1, y1)

        # Bilinear interpolation
        c0 = c00.lerp(c10, fx)
        c1 = c01.lerp(c11, fx)
        return c0.lerp(c1, fy)

    def clear(self, color: Vec4 = None) -> None:
        """Clear texture to a solid color."""
        fill = color if color else Vec4.zero()
        for i in range(len(self.data)):
            self.data[i] = Vec4(fill.x, fill.y, fill.z, fill.w)

    def copy_from(self, other: "Texture") -> None:
        """Copy data from another texture."""
        if self.width != other.width or self.height != other.height:
            raise ValueError("Texture dimensions must match for copy")
        for i in range(len(self.data)):
            src = other.data[i]
            self.data[i] = Vec4(src.x, src.y, src.z, src.w)

    def clone(self) -> "Texture":
        """Create a deep copy of this texture."""
        new_tex = Texture(self.width, self.height)
        new_tex.copy_from(self)
        return new_tex


# =============================================================================
# Temporal Accumulator
# =============================================================================


@dataclass
class AccumulatorConfig:
    """Configuration for the temporal accumulator."""

    blend_factor: float = 0.1
    """Blend factor for exponential moving average. Lower = smoother but slower."""

    min_blend_factor: float = 0.02
    """Minimum blend factor to prevent stale history."""

    max_blend_factor: float = 0.5
    """Maximum blend factor for fast convergence on reset."""

    history_rejection_threshold: float = 0.1
    """Color difference threshold for history rejection (disocclusion)."""

    enable_history_rejection: bool = False
    """Enable history rejection for disoccluded pixels."""

    clamp_history: bool = True
    """Clamp history to neighborhood min/max (reduces ghosting)."""

    use_ycocg: bool = False
    """Convert to YCoCg color space for better rejection."""


class TemporalAccumulator:
    """
    Accumulates rendered frames for temporal anti-aliasing.

    The accumulator maintains a history buffer and blends each new frame
    with the history using an exponential moving average:

        output = lerp(history, current, blend_factor)

    After N frames (where N ~ 1/blend_factor), the image converges to
    a smooth result equivalent to supersampling.

    On camera movement, the history is invalidated and accumulation restarts.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        config: Accumulator configuration.
    """

    def __init__(
        self,
        width: int,
        height: int,
        config: Optional[AccumulatorConfig] = None,
    ) -> None:
        """
        Initialize the temporal accumulator.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            config: Optional configuration (uses defaults if None).
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")

        self.width = width
        self.height = height
        self.config = config if config else AccumulatorConfig()

        # History buffer (double-buffered for read-while-write)
        self._history: Texture = Texture(width, height)
        self._frame_count: int = 0
        self._converged: bool = False
        self._last_camera_hash: int = 0

    @property
    def frame_count(self) -> int:
        """Number of frames accumulated since last reset."""
        return self._frame_count

    @property
    def is_converged(self) -> bool:
        """Whether the accumulation has converged."""
        return self._converged

    def reset(self) -> None:
        """Reset accumulation history."""
        self._history.clear()
        self._frame_count = 0
        self._converged = False

    def resize(self, width: int, height: int) -> None:
        """Resize the accumulator, clearing history."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")
        self.width = width
        self.height = height
        self._history = Texture(width, height)
        self.reset()

    def _compute_camera_hash(
        self,
        camera_position: Optional[Vec3] = None,
        camera_rotation: Optional[Vec3] = None,
    ) -> int:
        """Compute a hash for camera state to detect movement."""
        if camera_position is None and camera_rotation is None:
            return 0

        def float_hash(f: float, precision: int = 1000) -> int:
            return hash(int(f * precision))

        h = 0
        if camera_position:
            h ^= float_hash(camera_position.x)
            h ^= float_hash(camera_position.y) << 1
            h ^= float_hash(camera_position.z) << 2
        if camera_rotation:
            h ^= float_hash(camera_rotation.x) << 3
            h ^= float_hash(camera_rotation.y) << 4
            h ^= float_hash(camera_rotation.z) << 5
        return h

    def _rgb_to_ycocg(self, color: Vec4) -> Vec4:
        """Convert RGB to YCoCg color space."""
        y = 0.25 * color.x + 0.5 * color.y + 0.25 * color.z
        co = 0.5 * color.x - 0.5 * color.z
        cg = -0.25 * color.x + 0.5 * color.y - 0.25 * color.z
        return Vec4(y, co, cg, color.w)

    def _ycocg_to_rgb(self, ycocg: Vec4) -> Vec4:
        """Convert YCoCg back to RGB."""
        y, co, cg = ycocg.x, ycocg.y, ycocg.z
        r = y + co - cg
        g = y + cg
        b = y - co - cg
        return Vec4(r, g, b, ycocg.w)

    def _color_difference(self, a: Vec4, b: Vec4) -> float:
        """Compute color difference for history rejection."""
        diff = a - b
        return math.sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z)

    def accumulate(
        self,
        current: Texture,
        history: Optional[Texture] = None,
        jitter: Optional[Vec2] = None,
        camera_position: Optional[Vec3] = None,
        camera_rotation: Optional[Vec3] = None,
    ) -> Texture:
        """
        Accumulate the current frame with history.

        Args:
            current: The current rendered frame.
            history: Optional explicit history buffer (uses internal if None).
            jitter: The jitter offset used for this frame (for reprojection).
            camera_position: Camera position for movement detection.
            camera_rotation: Camera rotation for movement detection.

        Returns:
            The accumulated result texture.
        """
        if current.width != self.width or current.height != self.height:
            raise ValueError(
                f"Current frame size {current.width}x{current.height} "
                f"doesn't match accumulator {self.width}x{self.height}"
            )

        # Use provided history or internal
        hist = history if history else self._history

        # Check for camera movement
        camera_hash = self._compute_camera_hash(camera_position, camera_rotation)
        camera_moved = camera_hash != self._last_camera_hash and self._frame_count > 0
        self._last_camera_hash = camera_hash

        if camera_moved:
            self.reset()

        # Create output texture
        output = Texture(self.width, self.height)

        # Compute adaptive blend factor
        if self._frame_count == 0:
            # First frame: use current directly
            blend_factor = 1.0
        else:
            # Adaptive blend: higher blend for early frames, converge to config value
            blend_factor = max(
                self.config.blend_factor,
                1.0 / (self._frame_count + 1),
            )
            blend_factor = min(blend_factor, self.config.max_blend_factor)
            blend_factor = max(blend_factor, self.config.min_blend_factor)

        # Accumulate each pixel
        for y in range(self.height):
            for x in range(self.width):
                current_color = current.get_pixel(x, y)
                history_color = hist.get_pixel(x, y)

                # Optional history rejection
                if self.config.enable_history_rejection and self._frame_count > 0:
                    diff = self._color_difference(current_color, history_color)
                    if diff > self.config.history_rejection_threshold:
                        # Disocclusion detected: favor current
                        blend_factor = min(0.8, blend_factor * 2.0)

                # Blend
                if self._frame_count == 0:
                    result = current_color
                else:
                    result = history_color.lerp(current_color, blend_factor)

                output.set_pixel(x, y, result)

        # Update history
        self._history.copy_from(output)
        self._frame_count += 1

        # Check convergence (after ~16 frames with default settings)
        if self._frame_count >= int(1.0 / self.config.blend_factor):
            self._converged = True

        return output

    def get_history(self) -> Texture:
        """Get the current history buffer."""
        return self._history.clone()


# =============================================================================
# WGSL Code Generation
# =============================================================================


def generate_jitter_wgsl() -> str:
    """
    Generate WGSL code for sub-pixel jitter calculation.

    Returns:
        WGSL code string with jitter functions.
    """
    return """\
/// Computes the Halton sequence value at a given index and base.
/// Returns a value in the range [0, 1).
fn halton_sequence(index: u32, base: u32) -> f32 {
    var result: f32 = 0.0;
    var f: f32 = 1.0 / f32(base);
    var i: u32 = index;

    while (i > 0u) {
        result += f * f32(i % base);
        i = i / base;
        f /= f32(base);
    }

    return result;
}

/// Gets the 2D sub-pixel jitter offset for a given frame.
/// Uses Halton sequence with bases 2 and 3.
/// Returns offset in [-0.5, 0.5) range.
fn get_jitter(frame: u32, sequence_length: u32) -> vec2<f32> {
    let index = frame % sequence_length;
    let h_x = halton_sequence(index, 2u);
    let h_y = halton_sequence(index, 3u);
    return vec2<f32>(h_x - 0.5, h_y - 0.5);
}

/// Applies jitter to UV coordinates for temporal anti-aliasing.
///
/// Arguments:
///   pixel: The pixel coordinates (0 to resolution - 1)
///   resolution: Screen resolution (width, height)
///   jitter: Sub-pixel jitter offset from get_jitter()
///
/// Returns:
///   Jittered UV coordinates in [-1, 1] range for ray generation.
fn apply_jitter_to_uv(pixel: vec2<f32>, resolution: vec2<f32>, jitter: vec2<f32>) -> vec2<f32> {
    // Convert pixel + 0.5 + jitter to normalized coordinates
    let uv = (pixel + 0.5 + jitter) / resolution;
    // Map [0, 1] to [-1, 1]
    return uv * 2.0 - 1.0;
}
"""


def generate_accumulation_wgsl() -> str:
    """
    Generate WGSL code for temporal accumulation.

    Returns:
        WGSL compute shader code for blending current and history.
    """
    return """\
/// Uniform buffer for TAA parameters.
struct TAAParams {
    blend_factor: f32,
    frame_count: u32,
    camera_moved: u32,  // Boolean flag
    _padding: u32,
}

@group(0) @binding(0) var current_texture: texture_2d<f32>;
@group(0) @binding(1) var history_texture: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: TAAParams;

/// Compute color difference for history rejection.
fn color_difference(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let diff = a - b;
    return sqrt(dot(diff, diff));
}

/// Main TAA accumulation compute shader.
@compute @workgroup_size(8, 8, 1)
fn taa_accumulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(current_texture);
    let pixel = vec2<i32>(global_id.xy);

    // Bounds check
    if (pixel.x >= i32(dimensions.x) || pixel.y >= i32(dimensions.y)) {
        return;
    }

    // Sample current and history
    let current_color = textureLoad(current_texture, pixel, 0);
    let history_color = textureLoad(history_texture, pixel, 0);

    var result: vec4<f32>;

    // If camera moved or first frame, use current directly
    if (params.camera_moved != 0u || params.frame_count == 0u) {
        result = current_color;
    } else {
        // Adaptive blend factor
        var blend = params.blend_factor;
        let adaptive_blend = 1.0 / f32(params.frame_count + 1u);
        blend = max(blend, adaptive_blend);
        blend = clamp(blend, 0.02, 0.5);

        // Exponential moving average
        result = mix(history_color, current_color, blend);
    }

    textureStore(output_texture, pixel, result);
}

/// TAA accumulation with neighborhood clamping for reduced ghosting.
@compute @workgroup_size(8, 8, 1)
fn taa_accumulate_clamped(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(current_texture);
    let pixel = vec2<i32>(global_id.xy);

    // Bounds check
    if (pixel.x >= i32(dimensions.x) || pixel.y >= i32(dimensions.y)) {
        return;
    }

    // Sample current
    let current_color = textureLoad(current_texture, pixel, 0);

    var result: vec4<f32>;

    if (params.camera_moved != 0u || params.frame_count == 0u) {
        result = current_color;
    } else {
        // Sample 3x3 neighborhood to compute min/max
        var neighborhood_min = current_color.rgb;
        var neighborhood_max = current_color.rgb;

        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let sample_pos = clamp(
                    pixel + vec2<i32>(dx, dy),
                    vec2<i32>(0, 0),
                    vec2<i32>(i32(dimensions.x) - 1, i32(dimensions.y) - 1)
                );
                let sample_color = textureLoad(current_texture, sample_pos, 0).rgb;
                neighborhood_min = min(neighborhood_min, sample_color);
                neighborhood_max = max(neighborhood_max, sample_color);
            }
        }

        // Sample and clamp history
        let history_color = textureLoad(history_texture, pixel, 0);
        let clamped_history = vec4<f32>(
            clamp(history_color.rgb, neighborhood_min, neighborhood_max),
            history_color.a
        );

        // Blend
        var blend = params.blend_factor;
        let adaptive_blend = 1.0 / f32(params.frame_count + 1u);
        blend = max(blend, adaptive_blend);
        blend = clamp(blend, 0.02, 0.5);

        result = mix(clamped_history, current_color, blend);
    }

    textureStore(output_texture, pixel, result);
}
"""


def generate_taa_pipeline_wgsl(include_ray_jitter: bool = True) -> str:
    """
    Generate complete WGSL code for TAA pipeline.

    Args:
        include_ray_jitter: If True, includes jitter functions for ray generation.

    Returns:
        Complete WGSL shader code for TAA.
    """
    code = ""

    if include_ray_jitter:
        code += generate_jitter_wgsl()
        code += "\n"

    code += generate_accumulation_wgsl()

    return code


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Halton sequence
    "halton_sequence",
    "halton_2d",
    # Jitter
    "JitterPattern",
    "JitterSequence",
    "get_jitter",
    # Texture
    "Texture",
    # Accumulator
    "AccumulatorConfig",
    "TemporalAccumulator",
    # WGSL
    "generate_jitter_wgsl",
    "generate_accumulation_wgsl",
    "generate_taa_pipeline_wgsl",
]
