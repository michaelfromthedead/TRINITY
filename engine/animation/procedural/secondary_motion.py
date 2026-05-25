"""
General Secondary Motion System.

Provides various secondary motion effects for bones:
- DelayedMotion: Bone follows source with time delay
- OscillatingMotion: Sine wave offset
- NoiseMotion: Perlin noise displacement
- ImpulseResponse: React to sudden movements

Usage:
    motion = DelayedMotion(affected_bones=[10, 11], delay=0.1)
    modified_pose = motion.update(pose, dt)
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Protocol, Dict, Callable

# Type aliases
Vec3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


class Pose(Protocol):
    """Protocol for pose data."""

    def get_bone_position(self, bone_index: int) -> Vec3:
        """Get world position of a bone."""
        ...

    def set_bone_position(self, bone_index: int, position: Vec3) -> None:
        """Set world position of a bone."""
        ...

    def get_bone_rotation(self, bone_index: int) -> Quaternion:
        """Get world rotation of a bone."""
        ...

    def set_bone_rotation(self, bone_index: int, rotation: Quaternion) -> None:
        """Set world rotation of a bone."""
        ...

    def copy(self) -> "Pose":
        """Create a copy of this pose."""
        ...


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def vec3_length(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def quat_slerp(a: Quaternion, b: Quaternion, t: float) -> Quaternion:
    """Spherical linear interpolation."""
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

    if dot < 0:
        b = (-b[0], -b[1], -b[2], -b[3])
        dot = -dot

    dot = min(dot, 1.0)

    if dot > 0.9995:
        result = (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        )
        length = math.sqrt(sum(x * x for x in result))
        return tuple(x / length for x in result)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (
        a[0] * s0 + b[0] * s1,
        a[1] * s0 + b[1] * s1,
        a[2] * s0 + b[2] * s1,
        a[3] * s0 + b[3] * s1,
    )


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quaternion:
    """Create quaternion from axis-angle."""
    length = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    if length < 1e-10:
        return (0.0, 0.0, 0.0, 1.0)

    inv_length = 1.0 / length
    axis = (axis[0] * inv_length, axis[1] * inv_length, axis[2] * inv_length)

    half_angle = angle * 0.5
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)

    return (axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half)


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Multiply two quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


class PerlinNoise:
    """Simple 1D Perlin noise implementation for procedural motion."""

    def __init__(self, seed: int = 0):
        self._permutation = list(range(256))
        random.seed(seed)
        random.shuffle(self._permutation)
        self._permutation = self._permutation * 2

    def _fade(self, t: float) -> float:
        """Smooth interpolation curve."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def _grad(self, hash_val: int, x: float) -> float:
        """Compute gradient."""
        # Use simpler gradient that produces values in [-1, 1]
        h = hash_val & 1
        return x if h == 0 else -x

    def noise(self, x: float) -> float:
        """
        Generate 1D Perlin noise.

        Args:
            x: Input value

        Returns:
            Noise value in range [-1, 1]
        """
        X = int(math.floor(x)) & 255
        x -= math.floor(x)
        u = self._fade(x)

        A = self._permutation[X]
        B = self._permutation[X + 1]

        # The gradients are applied to x and (x-1), both in [0,1) and [-1,0) respectively
        # This naturally bounds the output
        result = self._lerp(self._grad(A, x), self._grad(B, x - 1), u)
        return max(-1.0, min(1.0, result))

    def fbm(self, x: float, octaves: int = 4, persistence: float = 0.5) -> float:
        """
        Fractal Brownian Motion (layered noise).

        Args:
            x: Input value
            octaves: Number of noise layers
            persistence: Amplitude decay per octave

        Returns:
            Noise value
        """
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            total += self.noise(x * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2

        return total / max_value if max_value > 0 else 0.0


@dataclass
class SecondaryMotion(ABC):
    """
    Base class for secondary motion effects.

    All secondary motion effects inherit from this and implement
    the update method.
    """

    affected_bones: List[int]
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        if not self.affected_bones:
            raise ValueError("affected_bones must not be empty")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("weight must be in [0, 1]")

    @abstractmethod
    def update(self, pose: Pose, dt: float) -> Pose:
        """
        Update secondary motion and return modified pose.

        Args:
            pose: Current animation pose
            dt: Time step in seconds

        Returns:
            Modified pose with secondary motion applied
        """
        pass

    def reset(self) -> None:
        """Reset internal state."""
        pass


@dataclass
class DelayedMotion(SecondaryMotion):
    """
    Bone follows source position/rotation with time delay.

    Creates a "dragging" effect where bones lag behind their
    animated position.
    """

    delay: float = 0.1  # Delay time in seconds
    source_bone: int = -1  # Source to follow, -1 = same bone's animation

    # Internal state
    _position_buffer: Dict[int, List[Tuple[float, Vec3]]] = field(
        default_factory=dict, repr=False
    )
    _rotation_buffer: Dict[int, List[Tuple[float, Quaternion]]] = field(
        default_factory=dict, repr=False
    )
    _time: float = field(default=0.0, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.delay < 0:
            raise ValueError("delay must be >= 0")

        # Initialize buffers
        for bone in self.affected_bones:
            self._position_buffer[bone] = []
            self._rotation_buffer[bone] = []

    def update(self, pose: Pose, dt: float) -> Pose:
        if not self.enabled or dt <= 0:
            return pose

        self._time += dt
        result = pose.copy()

        for bone in self.affected_bones:
            source = self.source_bone if self.source_bone >= 0 else bone

            # Get current source position/rotation
            current_pos = pose.get_bone_position(source)
            current_rot = pose.get_bone_rotation(source)

            # Add to buffer
            self._position_buffer[bone].append((self._time, current_pos))
            self._rotation_buffer[bone].append((self._time, current_rot))

            # Find delayed values
            target_time = self._time - self.delay

            # Get position at target time
            delayed_pos = self._sample_buffer(
                self._position_buffer[bone], target_time, current_pos
            )
            delayed_rot = self._sample_rotation_buffer(
                self._rotation_buffer[bone], target_time, current_rot
            )

            # Blend with weight
            final_pos = vec3_lerp(current_pos, delayed_pos, self.weight)
            final_rot = quat_slerp(current_rot, delayed_rot, self.weight)

            result.set_bone_position(bone, final_pos)
            result.set_bone_rotation(bone, final_rot)

            # Clean old buffer entries
            self._clean_buffer(self._position_buffer[bone], target_time)
            self._clean_buffer(self._rotation_buffer[bone], target_time)

        return result

    def _sample_buffer(
        self, buffer: List[Tuple[float, Vec3]], target_time: float, default: Vec3
    ) -> Vec3:
        """Sample position from time buffer."""
        if not buffer:
            return default

        # Find surrounding samples
        prev_sample = buffer[0]
        for sample in buffer:
            if sample[0] > target_time:
                # Interpolate between prev and current
                if prev_sample[0] >= target_time:
                    return prev_sample[1]

                t = (target_time - prev_sample[0]) / (sample[0] - prev_sample[0])
                return vec3_lerp(prev_sample[1], sample[1], t)
            prev_sample = sample

        return buffer[-1][1] if buffer else default

    def _sample_rotation_buffer(
        self, buffer: List[Tuple[float, Quaternion]], target_time: float, default: Quaternion
    ) -> Quaternion:
        """Sample rotation from time buffer."""
        if not buffer:
            return default

        prev_sample = buffer[0]
        for sample in buffer:
            if sample[0] > target_time:
                if prev_sample[0] >= target_time:
                    return prev_sample[1]

                t = (target_time - prev_sample[0]) / (sample[0] - prev_sample[0])
                return quat_slerp(prev_sample[1], sample[1], t)
            prev_sample = sample

        return buffer[-1][1] if buffer else default

    def _clean_buffer(self, buffer: List, min_time: float) -> None:
        """Remove old buffer entries."""
        while len(buffer) > 2 and buffer[0][0] < min_time:
            buffer.pop(0)

    def reset(self) -> None:
        """Reset delay buffers."""
        self._time = 0.0
        for bone in self.affected_bones:
            self._position_buffer[bone] = []
            self._rotation_buffer[bone] = []


@dataclass
class OscillatingMotion(SecondaryMotion):
    """
    Applies sine wave offset to bones.

    Creates rhythmic oscillating motion like swaying or bobbing.
    """

    frequency: float = 1.0  # Oscillations per second
    amplitude: Vec3 = (0.0, 0.05, 0.0)  # Offset amplitude per axis
    rotation_amplitude: Vec3 = (0.0, 0.0, 0.0)  # Rotation in radians per axis
    phase_offset: float = 0.0  # Phase offset in radians
    per_bone_phase_offset: float = 0.0  # Additional phase per bone in chain

    # Internal state
    _time: float = field(default=0.0, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.frequency < 0:
            raise ValueError("frequency must be >= 0")

    def update(self, pose: Pose, dt: float) -> Pose:
        if not self.enabled or dt <= 0:
            return pose

        self._time += dt
        result = pose.copy()

        for i, bone in enumerate(self.affected_bones):
            # Calculate phase for this bone
            phase = (
                self._time * self.frequency * 2 * math.pi
                + self.phase_offset
                + i * self.per_bone_phase_offset
            )

            sine_val = math.sin(phase)

            # Apply position offset
            offset = vec3_scale(self.amplitude, sine_val * self.weight)
            current_pos = pose.get_bone_position(bone)
            result.set_bone_position(bone, vec3_add(current_pos, offset))

            # Apply rotation offset
            if any(r != 0 for r in self.rotation_amplitude):
                rot_angles = vec3_scale(self.rotation_amplitude, sine_val * self.weight)

                current_rot = pose.get_bone_rotation(bone)
                x_rot = quat_from_axis_angle((1.0, 0.0, 0.0), rot_angles[0])
                y_rot = quat_from_axis_angle((0.0, 1.0, 0.0), rot_angles[1])
                z_rot = quat_from_axis_angle((0.0, 0.0, 1.0), rot_angles[2])

                new_rot = quat_multiply(
                    quat_multiply(quat_multiply(current_rot, x_rot), y_rot), z_rot
                )
                result.set_bone_rotation(bone, new_rot)

        return result

    def reset(self) -> None:
        """Reset oscillation phase."""
        self._time = 0.0


@dataclass
class NoiseMotion(SecondaryMotion):
    """
    Applies Perlin noise displacement to bones.

    Creates organic, random-looking motion without visible patterns.
    """

    amplitude: Vec3 = (0.01, 0.01, 0.01)  # Noise amplitude per axis
    rotation_amplitude: Vec3 = (0.0, 0.0, 0.0)  # Rotation noise in radians
    frequency: float = 1.0  # Base noise frequency
    octaves: int = 2  # Noise detail layers
    persistence: float = 0.5  # Amplitude decay per octave
    seed: int = 0  # Random seed for reproducibility

    # Internal state
    _noise_generators: List[PerlinNoise] = field(default_factory=list, repr=False)
    _time: float = field(default=0.0, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.frequency <= 0:
            raise ValueError("frequency must be > 0")
        if self.octaves < 1:
            raise ValueError("octaves must be >= 1")

        # Create noise generators for each axis and bone
        self._noise_generators = [
            PerlinNoise(self.seed + i) for i in range(6 * len(self.affected_bones))
        ]

    def update(self, pose: Pose, dt: float) -> Pose:
        if not self.enabled or dt <= 0:
            return pose

        self._time += dt
        result = pose.copy()

        for i, bone in enumerate(self.affected_bones):
            # Get noise values for this bone
            base_idx = i * 6
            t = self._time * self.frequency

            noise_x = self._noise_generators[base_idx].fbm(
                t + bone * 0.1, self.octaves, self.persistence
            )
            noise_y = self._noise_generators[base_idx + 1].fbm(
                t + bone * 0.2, self.octaves, self.persistence
            )
            noise_z = self._noise_generators[base_idx + 2].fbm(
                t + bone * 0.3, self.octaves, self.persistence
            )

            # Apply position offset
            offset = (
                self.amplitude[0] * noise_x * self.weight,
                self.amplitude[1] * noise_y * self.weight,
                self.amplitude[2] * noise_z * self.weight,
            )

            current_pos = pose.get_bone_position(bone)
            result.set_bone_position(bone, vec3_add(current_pos, offset))

            # Apply rotation if specified
            if any(r != 0 for r in self.rotation_amplitude):
                rot_noise_x = self._noise_generators[base_idx + 3].fbm(
                    t + bone * 0.4, self.octaves, self.persistence
                )
                rot_noise_y = self._noise_generators[base_idx + 4].fbm(
                    t + bone * 0.5, self.octaves, self.persistence
                )
                rot_noise_z = self._noise_generators[base_idx + 5].fbm(
                    t + bone * 0.6, self.octaves, self.persistence
                )

                rot_offset = (
                    self.rotation_amplitude[0] * rot_noise_x * self.weight,
                    self.rotation_amplitude[1] * rot_noise_y * self.weight,
                    self.rotation_amplitude[2] * rot_noise_z * self.weight,
                )

                current_rot = pose.get_bone_rotation(bone)
                x_rot = quat_from_axis_angle((1.0, 0.0, 0.0), rot_offset[0])
                y_rot = quat_from_axis_angle((0.0, 1.0, 0.0), rot_offset[1])
                z_rot = quat_from_axis_angle((0.0, 0.0, 1.0), rot_offset[2])

                new_rot = quat_multiply(
                    quat_multiply(quat_multiply(current_rot, x_rot), y_rot), z_rot
                )
                result.set_bone_rotation(bone, new_rot)

        return result

    def reset(self) -> None:
        """Reset noise time."""
        self._time = 0.0


@dataclass
class ImpulseResponse(SecondaryMotion):
    """
    React to sudden movements with bounce/shake.

    Detects acceleration and applies a damped spring response.
    """

    stiffness: float = 50.0  # Spring stiffness
    damping: float = 0.7  # Damping ratio (0-1)
    threshold: float = 0.5  # Minimum acceleration to trigger
    max_response: float = 0.1  # Maximum displacement

    # Internal state per bone
    _velocities: Dict[int, Vec3] = field(default_factory=dict, repr=False)
    _offsets: Dict[int, Vec3] = field(default_factory=dict, repr=False)
    _prev_positions: Dict[int, Vec3] = field(default_factory=dict, repr=False)
    _prev_velocities: Dict[int, Vec3] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.stiffness <= 0:
            raise ValueError("stiffness must be > 0")
        if not (0.0 <= self.damping <= 1.0):
            raise ValueError("damping must be in [0, 1]")
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")

        # Initialize state
        for bone in self.affected_bones:
            self._velocities[bone] = (0.0, 0.0, 0.0)
            self._offsets[bone] = (0.0, 0.0, 0.0)
            self._prev_positions[bone] = (0.0, 0.0, 0.0)
            self._prev_velocities[bone] = (0.0, 0.0, 0.0)

    def update(self, pose: Pose, dt: float) -> Pose:
        if not self.enabled or dt <= 0:
            return pose

        result = pose.copy()

        for bone in self.affected_bones:
            current_pos = pose.get_bone_position(bone)

            # Calculate velocity
            if self._prev_positions[bone] != (0.0, 0.0, 0.0):
                velocity = vec3_scale(
                    vec3_sub(current_pos, self._prev_positions[bone]),
                    1.0 / dt
                )

                # Calculate acceleration
                acceleration = vec3_scale(
                    vec3_sub(velocity, self._prev_velocities[bone]),
                    1.0 / dt
                )

                accel_magnitude = vec3_length(acceleration)

                # Apply impulse if above threshold
                if accel_magnitude > self.threshold:
                    # Add impulse in opposite direction of acceleration
                    impulse_strength = min(accel_magnitude / 100.0, self.max_response)
                    if accel_magnitude > 0:
                        impulse = vec3_scale(acceleration, -impulse_strength / accel_magnitude)
                        self._velocities[bone] = vec3_add(self._velocities[bone], impulse)

                self._prev_velocities[bone] = velocity
            else:
                self._prev_velocities[bone] = (0.0, 0.0, 0.0)

            self._prev_positions[bone] = current_pos

            # Spring physics: F = -kx - cv
            spring_force = vec3_scale(self._offsets[bone], -self.stiffness)
            damping_force = vec3_scale(
                self._velocities[bone], -self.damping * 2 * math.sqrt(self.stiffness)
            )

            # Update velocity and position
            acceleration = vec3_add(spring_force, damping_force)
            self._velocities[bone] = vec3_add(
                self._velocities[bone],
                vec3_scale(acceleration, dt)
            )
            self._offsets[bone] = vec3_add(
                self._offsets[bone],
                vec3_scale(self._velocities[bone], dt)
            )

            # Clamp offset (avoid division by zero)
            offset_length = vec3_length(self._offsets[bone])
            if offset_length > self.max_response and offset_length > 1e-10:
                self._offsets[bone] = vec3_scale(
                    self._offsets[bone], self.max_response / offset_length
                )

            # Apply offset
            final_offset = vec3_scale(self._offsets[bone], self.weight)
            result.set_bone_position(bone, vec3_add(current_pos, final_offset))

        return result

    def apply_impulse(self, bone: int, impulse: Vec3) -> None:
        """
        Manually apply an impulse to a bone.

        Args:
            bone: Bone index
            impulse: Impulse vector
        """
        if bone in self._velocities:
            self._velocities[bone] = vec3_add(self._velocities[bone], impulse)

    def reset(self) -> None:
        """Reset all impulse state."""
        for bone in self.affected_bones:
            self._velocities[bone] = (0.0, 0.0, 0.0)
            self._offsets[bone] = (0.0, 0.0, 0.0)
            self._prev_positions[bone] = (0.0, 0.0, 0.0)
            self._prev_velocities[bone] = (0.0, 0.0, 0.0)


@dataclass
class MotionComposer:
    """
    Composes multiple secondary motion effects.

    Allows stacking and managing multiple motion effects on bones.
    """

    motions: List[SecondaryMotion] = field(default_factory=list)

    def add(self, motion: SecondaryMotion) -> None:
        """Add a motion effect."""
        self.motions.append(motion)

    def remove(self, motion: SecondaryMotion) -> bool:
        """Remove a motion effect."""
        if motion in self.motions:
            self.motions.remove(motion)
            return True
        return False

    def clear(self) -> None:
        """Remove all motion effects."""
        self.motions.clear()

    def update(self, pose: Pose, dt: float) -> Pose:
        """
        Apply all motion effects in sequence.

        Args:
            pose: Input pose
            dt: Time step

        Returns:
            Pose with all effects applied
        """
        result = pose
        for motion in self.motions:
            if motion.enabled:
                result = motion.update(result, dt)
        return result

    def reset(self) -> None:
        """Reset all motion effects."""
        for motion in self.motions:
            motion.reset()

    def set_all_weights(self, weight: float) -> None:
        """Set weight for all motion effects."""
        for motion in self.motions:
            motion.weight = max(0.0, min(1.0, weight))

    def enable_all(self, enabled: bool = True) -> None:
        """Enable or disable all motion effects."""
        for motion in self.motions:
            motion.enabled = enabled

    def get_motion_count(self) -> int:
        """Get number of motion effects."""
        return len(self.motions)
