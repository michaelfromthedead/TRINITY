"""Animation baked to textures for GPU crowd rendering.

This module provides functionality to encode animation clips as texture data,
enabling efficient GPU-based crowd animation where bone transforms are sampled
directly from textures in vertex shaders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

from engine.core.math import Vec3, Vec4, Quat, Transform
from engine.animation.config import ANIMATION_TEXTURE_CONFIG


class TextureFormat(Enum):
    """Texture format for animation data encoding."""
    FLOAT32 = auto()  # Full precision float32 RGBA
    FLOAT16 = auto()  # Half precision float16 RGBA
    RGBA8_UNORM = auto()  # 8-bit normalized (lossy but compact)
    RGBA8_SNORM = auto()  # 8-bit signed normalized


@dataclass
class AnimationTexture:
    """Animation data baked into texture format for GPU sampling.

    Each row represents one frame of animation.
    Each bone uses 2 pixels: one for position (xyz) + scale(w),
    one for rotation quaternion (xyzw).

    Attributes:
        texture_data: Raw pixel data as list of floats (RGBA per pixel)
        bone_count: Number of bones in the animation
        frame_count: Number of animation frames
        width: Texture width in pixels
        height: Texture height in pixels
        format: Texture format used for encoding
        clip_name: Name of the source animation clip
        duration: Duration of the animation in seconds
        frame_rate: Frames per second of the animation
    """
    texture_data: list[float] = field(default_factory=list)
    bone_count: int = 0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    format: TextureFormat = TextureFormat.FLOAT32
    clip_name: str = ""
    duration: float = 0.0
    frame_rate: float = 30.0

    def get_pixel(self, x: int, y: int) -> tuple[float, float, float, float]:
        """Get RGBA values at pixel coordinates."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return (0.0, 0.0, 0.0, 0.0)
        idx = (y * self.width + x) * 4
        if idx + 4 > len(self.texture_data):
            return (0.0, 0.0, 0.0, 0.0)
        return (
            self.texture_data[idx],
            self.texture_data[idx + 1],
            self.texture_data[idx + 2],
            self.texture_data[idx + 3],
        )

    def set_pixel(self, x: int, y: int, r: float, g: float, b: float, a: float) -> None:
        """Set RGBA values at pixel coordinates."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        idx = (y * self.width + x) * 4
        if idx + 4 > len(self.texture_data):
            return
        self.texture_data[idx] = r
        self.texture_data[idx + 1] = g
        self.texture_data[idx + 2] = b
        self.texture_data[idx + 3] = a

    def get_bone_transform(self, bone_index: int, frame: int) -> Transform:
        """Extract bone transform at given frame.

        Args:
            bone_index: Index of the bone
            frame: Frame number

        Returns:
            Transform for the bone at that frame
        """
        if bone_index >= self.bone_count or frame >= self.frame_count:
            return Transform.identity()

        # Each bone uses 2 pixels
        pixel_x = bone_index * 2
        pixel_y = frame

        # First pixel: position (xyz) + uniform scale (w)
        pos_data = self.get_pixel(pixel_x, pixel_y)
        position = Vec3(pos_data[0], pos_data[1], pos_data[2])
        uniform_scale = pos_data[3]

        # Second pixel: rotation quaternion (xyzw)
        rot_data = self.get_pixel(pixel_x + 1, pixel_y)
        rotation = Quat(rot_data[0], rot_data[1], rot_data[2], rot_data[3])

        return Transform(
            translation=position,
            rotation=rotation.normalized(),
            scale=Vec3(uniform_scale, uniform_scale, uniform_scale),
        )

    def sample_bone_transform(self, bone_index: int, time: float) -> Transform:
        """Sample bone transform at arbitrary time with interpolation.

        Args:
            bone_index: Index of the bone
            time: Time in seconds

        Returns:
            Interpolated transform
        """
        if self.duration <= 0 or self.frame_count <= 1:
            return self.get_bone_transform(bone_index, 0)

        # Normalize time to frame
        normalized_time = (time % self.duration) / self.duration
        frame_float = normalized_time * (self.frame_count - 1)

        frame_a = int(frame_float)
        frame_b = min(frame_a + 1, self.frame_count - 1)
        blend = frame_float - frame_a

        transform_a = self.get_bone_transform(bone_index, frame_a)
        transform_b = self.get_bone_transform(bone_index, frame_b)

        return transform_a.lerp(transform_b, blend)

    def sample_bone_transform_cubic(self, bone_index: int, time: float) -> Transform:
        """Sample bone transform at arbitrary time with cubic interpolation.

        Uses Catmull-Rom for translation/scale and SQUAD for rotation.
        Falls back to linear interpolation if fewer than 4 frames.

        Args:
            bone_index: Index of the bone
            time: Time in seconds

        Returns:
            Interpolated transform using cubic interpolation
        """
        if bone_index >= self.bone_count:
            return Transform.identity()

        if self.duration <= 0 or self.frame_count <= 1:
            return self.get_bone_transform(bone_index, 0)

        # Fall back to linear for 2-3 frames
        if self.frame_count < 4:
            return self.sample_bone_transform(bone_index, time)

        # Normalize time to frame
        normalized_time = (time % self.duration) / self.duration
        frame_float = normalized_time * (self.frame_count - 1)

        frame_1 = int(frame_float)
        blend = frame_float - frame_1

        # Get four frames for cubic interpolation
        frame_0 = max(0, frame_1 - 1)
        frame_2 = min(frame_1 + 1, self.frame_count - 1)
        frame_3 = min(frame_1 + 2, self.frame_count - 1)

        t0 = self.get_bone_transform(bone_index, frame_0)
        t1 = self.get_bone_transform(bone_index, frame_1)
        t2 = self.get_bone_transform(bone_index, frame_2)
        t3 = self.get_bone_transform(bone_index, frame_3)

        return cubic_hermite_interpolate_transform(t0, t1, t2, t3, blend)

    def get_memory_size_bytes(self) -> int:
        """Calculate memory size in bytes."""
        bytes_per_component = {
            TextureFormat.FLOAT32: 4,
            TextureFormat.FLOAT16: 2,
            TextureFormat.RGBA8_UNORM: 1,
            TextureFormat.RGBA8_SNORM: 1,
        }
        return self.width * self.height * 4 * bytes_per_component.get(self.format, 4)


@dataclass
class AnimationTextureAtlas:
    """Atlas containing multiple animation clips in a single texture.

    Allows multiple animations to share a single GPU texture for efficiency.
    """
    texture_data: list[float] = field(default_factory=list)
    width: int = 0
    height: int = 0
    format: TextureFormat = TextureFormat.FLOAT32
    bone_count: int = 0
    clips: dict[str, tuple[int, int, int]] = field(default_factory=dict)  # name -> (start_row, frame_count, frame_rate)

    def add_clip(self, name: str, clip_texture: AnimationTexture) -> bool:
        """Add an animation clip to the atlas.

        Args:
            name: Name identifier for the clip
            clip_texture: Animation texture to add

        Returns:
            True if successfully added, False otherwise
        """
        if clip_texture.bone_count != self.bone_count and self.bone_count > 0:
            return False  # Bone count mismatch

        if self.bone_count == 0:
            self.bone_count = clip_texture.bone_count
            self.width = clip_texture.width

        start_row = self.height
        self.clips[name] = (start_row, clip_texture.frame_count, int(clip_texture.frame_rate))

        # Append texture data
        self.texture_data.extend(clip_texture.texture_data)
        self.height += clip_texture.frame_count

        return True

    def get_clip_info(self, name: str) -> tuple[int, int, int] | None:
        """Get clip info: (start_row, frame_count, frame_rate)."""
        return self.clips.get(name)

    def get_clip_uv_range(self, name: str) -> tuple[float, float] | None:
        """Get normalized UV Y-range for a clip."""
        info = self.clips.get(name)
        if info is None or self.height == 0:
            return None
        start_row, frame_count, _ = info
        return (start_row / self.height, (start_row + frame_count) / self.height)

    def sample_clip(self, name: str, bone_index: int, time: float) -> Transform:
        """Sample a specific clip at given time.

        Args:
            name: Clip name
            bone_index: Bone index
            time: Time in seconds

        Returns:
            Sampled transform
        """
        info = self.clips.get(name)
        if info is None:
            return Transform.identity()

        start_row, frame_count, frame_rate = info
        if frame_count <= 1 or frame_rate <= 0:
            return self._get_bone_transform(bone_index, start_row)

        duration = frame_count / frame_rate
        normalized_time = (time % duration) / duration
        frame_float = normalized_time * (frame_count - 1)

        frame_a = int(frame_float)
        frame_b = min(frame_a + 1, frame_count - 1)
        blend = frame_float - frame_a

        transform_a = self._get_bone_transform(bone_index, start_row + frame_a)
        transform_b = self._get_bone_transform(bone_index, start_row + frame_b)

        return transform_a.lerp(transform_b, blend)

    def sample_clip_cubic(self, name: str, bone_index: int, time: float) -> Transform:
        """Sample a specific clip at given time with cubic interpolation.

        Uses Catmull-Rom for translation/scale and SQUAD for rotation.
        Falls back to linear interpolation if fewer than 4 frames.

        Args:
            name: Clip name
            bone_index: Bone index
            time: Time in seconds

        Returns:
            Sampled transform using cubic interpolation
        """
        info = self.clips.get(name)
        if info is None:
            return Transform.identity()

        start_row, frame_count, frame_rate = info
        if frame_count <= 1 or frame_rate <= 0:
            return self._get_bone_transform(bone_index, start_row)

        # Fall back to linear for 2-3 frames
        if frame_count < 4:
            return self.sample_clip(name, bone_index, time)

        duration = frame_count / frame_rate
        normalized_time = (time % duration) / duration
        frame_float = normalized_time * (frame_count - 1)

        frame_1 = int(frame_float)
        blend = frame_float - frame_1

        # Get four frames for cubic interpolation
        frame_0 = max(0, frame_1 - 1)
        frame_2 = min(frame_1 + 1, frame_count - 1)
        frame_3 = min(frame_1 + 2, frame_count - 1)

        t0 = self._get_bone_transform(bone_index, start_row + frame_0)
        t1 = self._get_bone_transform(bone_index, start_row + frame_1)
        t2 = self._get_bone_transform(bone_index, start_row + frame_2)
        t3 = self._get_bone_transform(bone_index, start_row + frame_3)

        return cubic_hermite_interpolate_transform(t0, t1, t2, t3, blend)

    def _get_bone_transform(self, bone_index: int, row: int) -> Transform:
        """Get transform for a bone at a specific row."""
        if bone_index >= self.bone_count or row >= self.height:
            return Transform.identity()

        pixel_x = bone_index * 2
        idx = (row * self.width + pixel_x) * 4

        if idx + 8 > len(self.texture_data):
            return Transform.identity()

        # Position + scale
        pos = Vec3(
            self.texture_data[idx],
            self.texture_data[idx + 1],
            self.texture_data[idx + 2],
        )
        scale_val = self.texture_data[idx + 3]

        # Rotation
        rot = Quat(
            self.texture_data[idx + 4],
            self.texture_data[idx + 5],
            self.texture_data[idx + 6],
            self.texture_data[idx + 7],
        )

        return Transform(
            translation=pos,
            rotation=rot.normalized(),
            scale=Vec3(scale_val, scale_val, scale_val),
        )


def encode_transform_to_pixels(transform: Transform) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Encode a transform into two RGBA pixels.

    First pixel: position (xyz) + uniform scale (w)
    Second pixel: rotation quaternion (xyzw)

    Args:
        transform: Transform to encode

    Returns:
        Tuple of two RGBA pixel tuples
    """
    # Use average scale for uniform scale encoding
    uniform_scale = (transform.scale.x + transform.scale.y + transform.scale.z) / 3.0

    pixel1 = (
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
        uniform_scale,
    )

    rot = transform.rotation.normalized()
    pixel2 = (rot.x, rot.y, rot.z, rot.w)

    return pixel1, pixel2


def decode_pixels_to_transform(pixel1: tuple[float, float, float, float],
                                pixel2: tuple[float, float, float, float]) -> Transform:
    """Decode two RGBA pixels back to a transform.

    Args:
        pixel1: Position (xyz) + scale (w)
        pixel2: Rotation quaternion (xyzw)

    Returns:
        Decoded transform
    """
    position = Vec3(pixel1[0], pixel1[1], pixel1[2])
    scale_val = pixel1[3]
    rotation = Quat(pixel2[0], pixel2[1], pixel2[2], pixel2[3]).normalized()

    return Transform(
        translation=position,
        rotation=rotation,
        scale=Vec3(scale_val, scale_val, scale_val),
    )


@dataclass
class Skeleton:
    """Simple skeleton representation for animation baking."""
    bone_names: list[str] = field(default_factory=list)
    bone_parents: list[int] = field(default_factory=list)  # -1 for root
    bind_poses: list[Transform] = field(default_factory=list)

    @property
    def bone_count(self) -> int:
        return len(self.bone_names)


@dataclass
class AnimationClip:
    """Simple animation clip representation."""
    name: str = ""
    duration: float = 0.0
    frame_rate: float = 30.0
    bone_tracks: dict[int, list[Transform]] = field(default_factory=dict)  # bone_index -> frames

    @property
    def frame_count(self) -> int:
        if not self.bone_tracks:
            return 0
        return max(len(frames) for frames in self.bone_tracks.values())

    def get_bone_transform(self, bone_index: int, frame: int) -> Transform | None:
        """Get transform for a bone at a specific frame."""
        track = self.bone_tracks.get(bone_index)
        if track is None or frame >= len(track):
            return None
        return track[frame]

    def sample_bone(self, bone_index: int, time: float) -> Transform | None:
        """Sample bone transform at arbitrary time."""
        track = self.bone_tracks.get(bone_index)
        if track is None or len(track) == 0:
            return None

        if self.duration <= 0 or len(track) <= 1:
            return track[0]

        normalized_time = (time % self.duration) / self.duration
        frame_float = normalized_time * (len(track) - 1)

        frame_a = int(frame_float)
        frame_b = min(frame_a + 1, len(track) - 1)
        blend = frame_float - frame_a

        return track[frame_a].lerp(track[frame_b], blend)


class AnimationTextureOverflowError(Exception):
    """Raised when animation texture exceeds maximum dimensions."""
    pass


def bake_clip_to_texture(
    clip: AnimationClip,
    skeleton: Skeleton,
    format: TextureFormat = TextureFormat.FLOAT32,
) -> AnimationTexture:
    """Bake an animation clip into a texture for GPU sampling.

    Each row of the texture represents one frame.
    Each bone requires 2 pixels (position+scale, rotation).

    Args:
        clip: Animation clip to bake
        skeleton: Skeleton definition
        format: Texture format to use

    Returns:
        AnimationTexture containing the baked animation

    Raises:
        AnimationTextureOverflowError: If bone/frame count exceeds texture limits
    """
    bone_count = skeleton.bone_count
    frame_count = clip.frame_count

    if bone_count == 0 or frame_count == 0:
        return AnimationTexture(
            texture_data=[],
            bone_count=0,
            frame_count=0,
            width=0,
            height=0,
            format=format,
            clip_name=clip.name,
            duration=clip.duration,
            frame_rate=clip.frame_rate,
        )

    # Validate against maximum dimensions to prevent texture overflow
    if bone_count > ANIMATION_TEXTURE_CONFIG.MAX_BONES_PER_TEXTURE:
        raise AnimationTextureOverflowError(
            f"Bone count {bone_count} exceeds maximum {ANIMATION_TEXTURE_CONFIG.MAX_BONES_PER_TEXTURE}"
        )
    if frame_count > ANIMATION_TEXTURE_CONFIG.MAX_FRAMES_PER_ANIMATION:
        raise AnimationTextureOverflowError(
            f"Frame count {frame_count} exceeds maximum {ANIMATION_TEXTURE_CONFIG.MAX_FRAMES_PER_ANIMATION}"
        )

    # Width = 2 pixels per bone (position+scale, rotation)
    width = bone_count * 2
    height = frame_count

    # Validate final dimensions
    if width > ANIMATION_TEXTURE_CONFIG.MAX_TEXTURE_WIDTH:
        raise AnimationTextureOverflowError(
            f"Texture width {width} exceeds maximum {ANIMATION_TEXTURE_CONFIG.MAX_TEXTURE_WIDTH}"
        )
    if height > ANIMATION_TEXTURE_CONFIG.MAX_TEXTURE_HEIGHT:
        raise AnimationTextureOverflowError(
            f"Texture height {height} exceeds maximum {ANIMATION_TEXTURE_CONFIG.MAX_TEXTURE_HEIGHT}"
        )

    # Initialize texture data
    texture_data: list[float] = [0.0] * (width * height * 4)

    for frame_idx in range(frame_count):
        for bone_idx in range(bone_count):
            # Get transform from clip or use bind pose
            transform = clip.get_bone_transform(bone_idx, frame_idx)
            if transform is None:
                transform = skeleton.bind_poses[bone_idx] if bone_idx < len(skeleton.bind_poses) else Transform.identity()

            # Encode to pixels
            pixel1, pixel2 = encode_transform_to_pixels(transform)

            # Write to texture
            pixel_x = bone_idx * 2

            # First pixel (position + scale)
            idx1 = (frame_idx * width + pixel_x) * 4
            texture_data[idx1:idx1 + 4] = [pixel1[0], pixel1[1], pixel1[2], pixel1[3]]

            # Second pixel (rotation)
            idx2 = (frame_idx * width + pixel_x + 1) * 4
            texture_data[idx2:idx2 + 4] = [pixel2[0], pixel2[1], pixel2[2], pixel2[3]]

    return AnimationTexture(
        texture_data=texture_data,
        bone_count=bone_count,
        frame_count=frame_count,
        width=width,
        height=height,
        format=format,
        clip_name=clip.name,
        duration=clip.duration,
        frame_rate=clip.frame_rate,
    )


def pack_float_to_rgba8(
    value: float,
    min_val: float = ANIMATION_TEXTURE_CONFIG.PACK_MIN_VALUE,
    max_val: float = ANIMATION_TEXTURE_CONFIG.PACK_MAX_VALUE
) -> tuple[int, int, int, int]:
    """Pack a float value into RGBA8 format using 32 bits total.

    This provides better precision by using all 4 channels for a single value.
    """
    # Avoid division by zero
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1.0

    # Normalize to 0-1 range
    normalized = (value - min_val) / range_val
    normalized = max(0.0, min(1.0, normalized))

    # Convert to 32-bit integer
    int_val = int(normalized * 0xFFFFFFFF)

    r = (int_val >> 24) & 0xFF
    g = (int_val >> 16) & 0xFF
    b = (int_val >> 8) & 0xFF
    a = int_val & 0xFF

    return (r, g, b, a)


def unpack_rgba8_to_float(
    r: int, g: int, b: int, a: int,
    min_val: float = ANIMATION_TEXTURE_CONFIG.PACK_MIN_VALUE,
    max_val: float = ANIMATION_TEXTURE_CONFIG.PACK_MAX_VALUE
) -> float:
    """Unpack RGBA8 back to float."""
    int_val = (r << 24) | (g << 16) | (b << 8) | a
    normalized = int_val / 0xFFFFFFFF
    return min_val + normalized * (max_val - min_val)


# =============================================================================
# CUBIC HERMITE (CATMULL-ROM) INTERPOLATION
# =============================================================================

def cubic_hermite_interpolate(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """Catmull-Rom spline interpolation for scalar values.

    Interpolates between p1 and p2, using p0 and p3 as control points.
    At t=0 returns p1, at t=1 returns p2.

    Args:
        p0: Control point before p1
        p1: Start point (t=0)
        p2: End point (t=1)
        p3: Control point after p2
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated value
    """
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom basis functions
    return 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def cubic_hermite_interpolate_vec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
    """Catmull-Rom spline interpolation for Vec3 values.

    Interpolates between p1 and p2, using p0 and p3 as control points.

    Args:
        p0: Control point before p1
        p1: Start point (t=0)
        p2: End point (t=1)
        p3: Control point after p2
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated Vec3
    """
    return Vec3(
        cubic_hermite_interpolate(p0.x, p1.x, p2.x, p3.x, t),
        cubic_hermite_interpolate(p0.y, p1.y, p2.y, p3.y, t),
        cubic_hermite_interpolate(p0.z, p1.z, p2.z, p3.z, t),
    )


# =============================================================================
# SQUAD QUATERNION INTERPOLATION (PRIVATE HELPERS)
# =============================================================================

def _quat_log(q: Quat) -> Vec3:
    """Compute quaternion logarithm.

    For a unit quaternion q = (v*sin(theta), cos(theta)) where v is unit axis,
    log(q) = v * theta (a 3D vector).

    Args:
        q: Unit quaternion

    Returns:
        3D vector representing the logarithm
    """
    # Handle identity quaternion
    sin_half_angle = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z)

    if sin_half_angle < 1e-8:
        # Near identity, use small angle approximation
        return Vec3(q.x, q.y, q.z)

    # Compute half-angle
    half_angle = math.atan2(sin_half_angle, q.w)

    # Scale vector part by half_angle / sin(half_angle)
    scale = half_angle / sin_half_angle
    return Vec3(q.x * scale, q.y * scale, q.z * scale)


def _quat_exp(v: Vec3) -> Quat:
    """Compute quaternion exponential.

    exp(v) = (v/|v| * sin(|v|), cos(|v|)) for 3D vector v.

    Args:
        v: 3D vector (half-angle * axis)

    Returns:
        Unit quaternion
    """
    half_angle = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    if half_angle < 1e-8:
        # Near zero, return identity
        return Quat(v.x, v.y, v.z, 1.0).normalized()

    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)
    scale = sin_half / half_angle

    return Quat(
        v.x * scale,
        v.y * scale,
        v.z * scale,
        cos_half,
    ).normalized()


def _compute_squad_intermediate(q_prev: Quat, q_curr: Quat, q_next: Quat) -> Quat:
    """Compute SQUAD intermediate control point.

    For smooth quaternion interpolation, we need intermediate control points
    s_i = q_i * exp(-(log(q_i^-1 * q_{i+1}) + log(q_i^-1 * q_{i-1})) / 4)

    Args:
        q_prev: Previous quaternion in sequence
        q_curr: Current quaternion
        q_next: Next quaternion in sequence

    Returns:
        Intermediate control quaternion
    """
    # Ensure quaternions are in same hemisphere
    if q_prev.dot(q_curr) < 0:
        q_prev = Quat(-q_prev.x, -q_prev.y, -q_prev.z, -q_prev.w)
    if q_next.dot(q_curr) < 0:
        q_next = Quat(-q_next.x, -q_next.y, -q_next.z, -q_next.w)

    # Compute q_curr^-1 (conjugate for unit quaternion)
    q_curr_inv = Quat(-q_curr.x, -q_curr.y, -q_curr.z, q_curr.w)

    # q_curr^-1 * q_next
    q_to_next = Quat(
        q_curr_inv.w * q_next.x + q_curr_inv.x * q_next.w + q_curr_inv.y * q_next.z - q_curr_inv.z * q_next.y,
        q_curr_inv.w * q_next.y - q_curr_inv.x * q_next.z + q_curr_inv.y * q_next.w + q_curr_inv.z * q_next.x,
        q_curr_inv.w * q_next.z + q_curr_inv.x * q_next.y - q_curr_inv.y * q_next.x + q_curr_inv.z * q_next.w,
        q_curr_inv.w * q_next.w - q_curr_inv.x * q_next.x - q_curr_inv.y * q_next.y - q_curr_inv.z * q_next.z,
    )

    # q_curr^-1 * q_prev
    q_to_prev = Quat(
        q_curr_inv.w * q_prev.x + q_curr_inv.x * q_prev.w + q_curr_inv.y * q_prev.z - q_curr_inv.z * q_prev.y,
        q_curr_inv.w * q_prev.y - q_curr_inv.x * q_prev.z + q_curr_inv.y * q_prev.w + q_curr_inv.z * q_prev.x,
        q_curr_inv.w * q_prev.z + q_curr_inv.x * q_prev.y - q_curr_inv.y * q_prev.x + q_curr_inv.z * q_prev.w,
        q_curr_inv.w * q_prev.w - q_curr_inv.x * q_prev.x - q_curr_inv.y * q_prev.y - q_curr_inv.z * q_prev.z,
    )

    # Logarithms
    log_next = _quat_log(q_to_next)
    log_prev = _quat_log(q_to_prev)

    # -(log_next + log_prev) / 4
    neg_sum = Vec3(
        -(log_next.x + log_prev.x) / 4.0,
        -(log_next.y + log_prev.y) / 4.0,
        -(log_next.z + log_prev.z) / 4.0,
    )

    # exp and multiply by q_curr
    exp_result = _quat_exp(neg_sum)

    # q_curr * exp_result
    result = Quat(
        q_curr.w * exp_result.x + q_curr.x * exp_result.w + q_curr.y * exp_result.z - q_curr.z * exp_result.y,
        q_curr.w * exp_result.y - q_curr.x * exp_result.z + q_curr.y * exp_result.w + q_curr.z * exp_result.x,
        q_curr.w * exp_result.z + q_curr.x * exp_result.y - q_curr.y * exp_result.x + q_curr.z * exp_result.w,
        q_curr.w * exp_result.w - q_curr.x * exp_result.x - q_curr.y * exp_result.y - q_curr.z * exp_result.z,
    )

    return result.normalized()


def _slerp(q1: Quat, q2: Quat, t: float) -> Quat:
    """Spherical linear interpolation between two quaternions.

    Args:
        q1: Start quaternion
        q2: End quaternion
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion
    """
    # Ensure shortest path
    dot = q1.dot(q2)
    if dot < 0:
        q2 = Quat(-q2.x, -q2.y, -q2.z, -q2.w)
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        result = Quat(
            q1.x + t * (q2.x - q1.x),
            q1.y + t * (q2.y - q1.y),
            q1.z + t * (q2.z - q1.z),
            q1.w + t * (q2.w - q1.w),
        )
        return result.normalized()

    # Standard slerp
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return Quat(
        s0 * q1.x + s1 * q2.x,
        s0 * q1.y + s1 * q2.y,
        s0 * q1.z + s1 * q2.z,
        s0 * q1.w + s1 * q2.w,
    ).normalized()


def _squad_interpolate(q0: Quat, q1: Quat, q2: Quat, q3: Quat, t: float) -> Quat:
    """SQUAD (Spherical Quadrangle) interpolation for quaternions.

    Provides C1-continuous cubic interpolation on the unit sphere.
    Interpolates between q1 and q2, using q0 and q3 as control points.

    Args:
        q0: Control point before q1
        q1: Start quaternion (t=0)
        q2: End quaternion (t=1)
        q3: Control point after q2
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion
    """
    # Ensure all quaternions are in same hemisphere relative to q1
    if q0.dot(q1) < 0:
        q0 = Quat(-q0.x, -q0.y, -q0.z, -q0.w)
    if q2.dot(q1) < 0:
        q2 = Quat(-q2.x, -q2.y, -q2.z, -q2.w)
    if q3.dot(q2) < 0:
        q3 = Quat(-q3.x, -q3.y, -q3.z, -q3.w)

    # Compute intermediate control points
    s1 = _compute_squad_intermediate(q0, q1, q2)
    s2 = _compute_squad_intermediate(q1, q2, q3)

    # SQUAD formula: slerp(slerp(q1, q2, t), slerp(s1, s2, t), 2t(1-t))
    slerp_q = _slerp(q1, q2, t)
    slerp_s = _slerp(s1, s2, t)

    return _slerp(slerp_q, slerp_s, 2.0 * t * (1.0 - t))


def cubic_hermite_interpolate_transform(
    t0: Transform, t1: Transform, t2: Transform, t3: Transform, t: float
) -> Transform:
    """Catmull-Rom spline interpolation for Transforms.

    Uses Catmull-Rom for translation and scale, SQUAD for rotation.
    Interpolates between t1 and t2, using t0 and t3 as control points.

    Args:
        t0: Control transform before t1
        t1: Start transform (t=0)
        t2: End transform (t=1)
        t3: Control transform after t2
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated transform
    """
    # Interpolate translation with Catmull-Rom
    translation = cubic_hermite_interpolate_vec3(
        t0.translation, t1.translation, t2.translation, t3.translation, t
    )

    # Interpolate scale with Catmull-Rom
    scale = cubic_hermite_interpolate_vec3(
        t0.scale, t1.scale, t2.scale, t3.scale, t
    )

    # Interpolate rotation with SQUAD
    rotation = _squad_interpolate(
        t0.rotation, t1.rotation, t2.rotation, t3.rotation, t
    )

    return Transform(
        translation=translation,
        rotation=rotation.normalized(),
        scale=scale,
    )


# =============================================================================
# ATLAS VALIDATION
# =============================================================================

def validate_atlas_uv_ranges(atlas: AnimationTextureAtlas) -> tuple[bool, list[tuple[str, str]]]:
    """Validate that all clips in an atlas have non-overlapping UV ranges.

    Args:
        atlas: Animation texture atlas to validate

    Returns:
        Tuple of (is_valid, list of overlapping clip name pairs)
    """
    overlaps: list[tuple[str, str]] = []

    clip_names = list(atlas.clips.keys())
    for i, name_a in enumerate(clip_names):
        info_a = atlas.clips[name_a]
        start_a, count_a, _ = info_a
        end_a = start_a + count_a

        for name_b in clip_names[i + 1:]:
            info_b = atlas.clips[name_b]
            start_b, count_b, _ = info_b
            end_b = start_b + count_b

            # Check for overlap (exclusive ends)
            if start_a < end_b and start_b < end_a:
                overlaps.append((name_a, name_b))

    return (len(overlaps) == 0, overlaps)
