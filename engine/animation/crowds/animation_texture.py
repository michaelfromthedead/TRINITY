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
        """Sample bone transform at arbitrary time with linear interpolation.

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
        """Sample bone transform at arbitrary time with Cubic Hermite interpolation.

        Provides smoother animation curves than linear interpolation by using
        Catmull-Rom splines for position/scale and SQUAD for rotation.

        Args:
            bone_index: Index of the bone
            time: Time in seconds

        Returns:
            Smoothly interpolated transform
        """
        # Edge case: single frame or invalid animation
        if self.duration <= 0 or self.frame_count <= 1:
            return self.get_bone_transform(bone_index, 0)

        # Edge case: only 2 frames - fall back to linear interpolation
        if self.frame_count == 2:
            return self.sample_bone_transform(bone_index, time)

        # Normalize time to frame
        normalized_time = (time % self.duration) / self.duration
        frame_float = normalized_time * (self.frame_count - 1)

        # Get the four frames needed for cubic interpolation
        frame1 = int(frame_float)
        frame2 = min(frame1 + 1, self.frame_count - 1)

        # Clamp frame0 and frame3 to valid range
        frame0 = max(frame1 - 1, 0)
        frame3 = min(frame2 + 1, self.frame_count - 1)

        blend = frame_float - frame1

        t0 = self.get_bone_transform(bone_index, frame0)
        t1 = self.get_bone_transform(bone_index, frame1)
        t2 = self.get_bone_transform(bone_index, frame2)
        t3 = self.get_bone_transform(bone_index, frame3)

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
        """Sample a specific clip at given time with linear interpolation.

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
        """Sample a specific clip with Cubic Hermite interpolation.

        Provides smoother animation curves than linear interpolation.

        Args:
            name: Clip name
            bone_index: Bone index
            time: Time in seconds

        Returns:
            Smoothly interpolated transform
        """
        info = self.clips.get(name)
        if info is None:
            return Transform.identity()

        start_row, frame_count, frame_rate = info

        # Edge case: single frame or invalid frame rate
        if frame_count <= 1 or frame_rate <= 0:
            return self._get_bone_transform(bone_index, start_row)

        # Edge case: only 2 frames - fall back to linear
        if frame_count == 2:
            return self.sample_clip(name, bone_index, time)

        duration = frame_count / frame_rate
        normalized_time = (time % duration) / duration
        frame_float = normalized_time * (frame_count - 1)

        # Get the four frames needed for cubic interpolation
        frame1 = int(frame_float)
        frame2 = min(frame1 + 1, frame_count - 1)
        frame0 = max(frame1 - 1, 0)
        frame3 = min(frame2 + 1, frame_count - 1)

        blend = frame_float - frame1

        t0 = self._get_bone_transform(bone_index, start_row + frame0)
        t1 = self._get_bone_transform(bone_index, start_row + frame1)
        t2 = self._get_bone_transform(bone_index, start_row + frame2)
        t3 = self._get_bone_transform(bone_index, start_row + frame3)

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


def cubic_hermite_interpolate(
    p0: float, p1: float, p2: float, p3: float, t: float
) -> float:
    """Cubic Hermite (Catmull-Rom) interpolation between p1 and p2.

    Uses p0 and p3 as control points to compute tangents.
    The result interpolates between p1 (t=0) and p2 (t=1).

    Args:
        p0: Point before p1 (for tangent calculation)
        p1: Start point (t=0)
        p2: End point (t=1)
        p3: Point after p2 (for tangent calculation)
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated value
    """
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom coefficients
    # m0 = (p2 - p0) / 2, m1 = (p3 - p1) / 2
    # Result = (2t^3 - 3t^2 + 1)*p1 + (t^3 - 2t^2 + t)*m0 + (-2t^3 + 3t^2)*p2 + (t^3 - t^2)*m1
    m0 = (p2 - p0) * 0.5
    m1 = (p3 - p1) * 0.5

    a = 2.0 * t3 - 3.0 * t2 + 1.0
    b = t3 - 2.0 * t2 + t
    c = -2.0 * t3 + 3.0 * t2
    d = t3 - t2

    return a * p1 + b * m0 + c * p2 + d * m1


def cubic_hermite_interpolate_vec3(
    p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float
) -> Vec3:
    """Cubic Hermite interpolation for Vec3."""
    return Vec3(
        cubic_hermite_interpolate(p0.x, p1.x, p2.x, p3.x, t),
        cubic_hermite_interpolate(p0.y, p1.y, p2.y, p3.y, t),
        cubic_hermite_interpolate(p0.z, p1.z, p2.z, p3.z, t),
    )


def cubic_hermite_interpolate_transform(
    t0: Transform, t1: Transform, t2: Transform, t3: Transform, blend: float
) -> Transform:
    """Cubic Hermite interpolation for transforms.

    Interpolates position and scale with Catmull-Rom spline,
    rotation with squad (spherical cubic interpolation approximation).

    Args:
        t0: Transform before t1 (control point)
        t1: Start transform (blend=0)
        t2: End transform (blend=1)
        t3: Transform after t2 (control point)
        blend: Interpolation parameter [0, 1]

    Returns:
        Smoothly interpolated transform
    """
    # Cubic Hermite for position
    pos = cubic_hermite_interpolate_vec3(
        t0.translation, t1.translation, t2.translation, t3.translation, blend
    )

    # Cubic Hermite for scale
    scale = cubic_hermite_interpolate_vec3(
        t0.scale, t1.scale, t2.scale, t3.scale, blend
    )

    # For rotation, use squad (spherical and quadrangle) approximation
    # This provides C1 continuity for quaternion interpolation
    rot = _squad_interpolate(t0.rotation, t1.rotation, t2.rotation, t3.rotation, blend)

    return Transform(translation=pos, rotation=rot.normalized(), scale=scale)


def _squad_interpolate(q0: Quat, q1: Quat, q2: Quat, q3: Quat, t: float) -> Quat:
    """Spherical cubic interpolation (SQUAD) for quaternions.

    Provides smooth, C1 continuous quaternion interpolation.

    Args:
        q0: Quaternion before q1
        q1: Start quaternion (t=0)
        q2: End quaternion (t=1)
        q3: Quaternion after q2
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion
    """
    # Ensure quaternions are in the same hemisphere
    if q0.dot(q1) < 0:
        q0 = Quat(-q0.x, -q0.y, -q0.z, -q0.w)
    if q1.dot(q2) < 0:
        q2 = Quat(-q2.x, -q2.y, -q2.z, -q2.w)
    if q2.dot(q3) < 0:
        q3 = Quat(-q3.x, -q3.y, -q3.z, -q3.w)

    # Calculate intermediate control quaternions
    s1 = _compute_squad_intermediate(q0, q1, q2)
    s2 = _compute_squad_intermediate(q1, q2, q3)

    # SQUAD: slerp(slerp(q1, q2, t), slerp(s1, s2, t), 2t(1-t))
    slerp1 = q1.slerp(q2, t)
    slerp2 = s1.slerp(s2, t)

    return slerp1.slerp(slerp2, 2.0 * t * (1.0 - t))


def _compute_squad_intermediate(q_prev: Quat, q_curr: Quat, q_next: Quat) -> Quat:
    """Compute intermediate quaternion for SQUAD.

    Returns the intermediate control quaternion for q_curr given
    its neighbors q_prev and q_next.
    """
    q_curr_inv = q_curr.inverse()

    # Log of relative rotations
    log_prev = _quat_log(q_curr_inv * q_prev)
    log_next = _quat_log(q_curr_inv * q_next)

    # Average in tangent space
    avg = Vec3(
        -(log_prev.x + log_next.x) * 0.25,
        -(log_prev.y + log_next.y) * 0.25,
        -(log_prev.z + log_next.z) * 0.25,
    )

    return q_curr * _quat_exp(avg)


def _quat_log(q: Quat) -> Vec3:
    """Quaternion logarithm (returns rotation axis * half-angle)."""
    # Ensure unit quaternion
    q = q.normalized()

    # Handle identity quaternion
    sin_half_angle = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z)
    if sin_half_angle < 1e-10:
        return Vec3(0.0, 0.0, 0.0)

    half_angle = math.atan2(sin_half_angle, q.w)
    k = half_angle / sin_half_angle

    return Vec3(q.x * k, q.y * k, q.z * k)


def _quat_exp(v: Vec3) -> Quat:
    """Quaternion exponential (converts axis * half-angle to quaternion)."""
    half_angle = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    if half_angle < 1e-10:
        return Quat(0.0, 0.0, 0.0, 1.0)

    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)
    k = sin_half / half_angle

    return Quat(v.x * k, v.y * k, v.z * k, cos_half)


def validate_atlas_uv_ranges(atlas: AnimationTextureAtlas) -> tuple[bool, list[tuple[str, str]]]:
    """Validate that all clip UV ranges in an atlas are non-overlapping.

    Args:
        atlas: The animation texture atlas to validate

    Returns:
        Tuple of (is_valid, list of overlapping clip name pairs)
        If is_valid is True, the overlap list will be empty.
    """
    if atlas.height == 0:
        return (True, [])

    overlaps: list[tuple[str, str]] = []
    clip_names = list(atlas.clips.keys())

    for i, name_a in enumerate(clip_names):
        range_a = atlas.get_clip_uv_range(name_a)
        if range_a is None:
            continue

        start_a, end_a = range_a

        for name_b in clip_names[i + 1:]:
            range_b = atlas.get_clip_uv_range(name_b)
            if range_b is None:
                continue

            start_b, end_b = range_b

            # Check for overlap: ranges overlap if one starts before the other ends
            # and ends after the other starts
            if start_a < end_b and end_a > start_b:
                overlaps.append((name_a, name_b))

    return (len(overlaps) == 0, overlaps)
