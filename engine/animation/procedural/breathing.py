"""
Procedural Breathing Animation.

Adds realistic breathing motion to characters by animating
spine, chest, and shoulder bones.

Usage:
    controller = BreathingController(
        spine_bones=[5, 6, 7],
        chest_bone=8,
        breath_rate=0.25  # 15 breaths per minute
    )
    modified_pose = controller.update(pose, dt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Protocol

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

    def get_bone_local_rotation(self, bone_index: int) -> Quaternion:
        """Get local rotation of a bone."""
        ...

    def set_bone_local_rotation(self, bone_index: int, rotation: Quaternion) -> None:
        """Set local rotation of a bone."""
        ...

    def copy(self) -> "Pose":
        """Create a copy of this pose."""
        ...


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quaternion:
    """Create quaternion from axis and angle."""
    length = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    if length < 1e-10:
        return (0.0, 0.0, 0.0, 1.0)

    inv_length = 1.0 / length
    axis = (axis[0] * inv_length, axis[1] * inv_length, axis[2] * inv_length)

    half_angle = angle * 0.5
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)

    return (
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        cos_half,
    )


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


class BreathPhase(Enum):
    """Phases of the breathing cycle."""

    INHALE = auto()  # Breathing in
    INHALE_HOLD = auto()  # Brief pause at top
    EXHALE = auto()  # Breathing out
    EXHALE_HOLD = auto()  # Brief pause at bottom


class ExertionLevel(Enum):
    """Exertion levels affecting breathing rate."""

    RELAXED = auto()  # Slow, deep breaths
    NORMAL = auto()  # Normal breathing
    ACTIVE = auto()  # Faster breathing
    HEAVY = auto()  # Rapid, shallow breathing
    EXHAUSTED = auto()  # Very rapid, gasping


@dataclass
class BreathingParams:
    """Parameters for a specific exertion level."""

    breath_rate: float  # Breaths per second (Hz)
    chest_expansion: float  # Chest scale factor
    shoulder_rise: float  # Shoulder lift amount
    spine_curve: float  # Spine bend amount (radians)
    inhale_ratio: float  # Ratio of cycle spent inhaling

    # Advanced
    asymmetry: float = 0.0  # Left-right asymmetry
    noise_amount: float = 0.0  # Randomness in breathing

    def __post_init__(self):
        if self.breath_rate <= 0:
            raise ValueError("breath_rate must be > 0")
        if not (0.0 < self.inhale_ratio < 1.0):
            raise ValueError("inhale_ratio must be in (0, 1)")


# Preset breathing parameters for each exertion level
BREATHING_PRESETS = {
    ExertionLevel.RELAXED: BreathingParams(
        breath_rate=0.2,  # 12 breaths per minute
        chest_expansion=0.02,
        shoulder_rise=0.005,
        spine_curve=math.radians(2.0),
        inhale_ratio=0.4,
    ),
    ExertionLevel.NORMAL: BreathingParams(
        breath_rate=0.25,  # 15 breaths per minute
        chest_expansion=0.025,
        shoulder_rise=0.008,
        spine_curve=math.radians(3.0),
        inhale_ratio=0.4,
    ),
    ExertionLevel.ACTIVE: BreathingParams(
        breath_rate=0.4,  # 24 breaths per minute
        chest_expansion=0.04,
        shoulder_rise=0.015,
        spine_curve=math.radians(5.0),
        inhale_ratio=0.45,
    ),
    ExertionLevel.HEAVY: BreathingParams(
        breath_rate=0.6,  # 36 breaths per minute
        chest_expansion=0.06,
        shoulder_rise=0.025,
        spine_curve=math.radians(8.0),
        inhale_ratio=0.5,
    ),
    ExertionLevel.EXHAUSTED: BreathingParams(
        breath_rate=0.8,  # 48 breaths per minute
        chest_expansion=0.08,
        shoulder_rise=0.04,
        spine_curve=math.radians(12.0),
        inhale_ratio=0.55,
        noise_amount=0.1,
    ),
}


@dataclass
class BreathingController:
    """
    Controller for procedural breathing animation.

    Animates spine, chest, and shoulder bones to create
    realistic breathing motion.
    """

    spine_bones: List[int]
    chest_bone: int
    breath_rate: float = 0.25  # Default 15 breaths/min

    # Optional additional bones
    shoulder_bones: List[int] = field(default_factory=list)  # Left, right
    clavicle_bones: List[int] = field(default_factory=list)  # Left, right
    neck_bone: int = -1

    # Motion amplitudes
    chest_expansion: float = 0.025
    shoulder_rise: float = 0.008
    spine_curve_amount: float = math.radians(3.0)
    neck_extension: float = math.radians(1.0)

    # Timing
    inhale_ratio: float = 0.4
    hold_ratio: float = 0.05  # Brief pause between phases

    # Current state
    _current_phase: float = field(default=0.0, repr=False)
    _current_breath_value: float = field(default=0.0, repr=False)
    _target_exertion: ExertionLevel = field(default=ExertionLevel.NORMAL, repr=False)
    _current_params: BreathingParams = field(default=None, repr=False)

    def __post_init__(self):
        if self.chest_bone < 0:
            raise ValueError("chest_bone must be >= 0")
        if self.breath_rate <= 0:
            raise ValueError("breath_rate must be > 0")
        if not (0.0 < self.inhale_ratio < 1.0):
            raise ValueError("inhale_ratio must be in (0, 1)")

        # Initialize with current params
        self._current_params = BreathingParams(
            breath_rate=self.breath_rate,
            chest_expansion=self.chest_expansion,
            shoulder_rise=self.shoulder_rise,
            spine_curve=self.spine_curve_amount,
            inhale_ratio=self.inhale_ratio,
        )

    def _calculate_breath_value(self, phase: float) -> Tuple[float, BreathPhase]:
        """
        Calculate breath value and phase from cycle position.

        Args:
            phase: Cycle position (0-1)

        Returns:
            (breath_value 0-1, current_phase)
        """
        inhale_end = self.inhale_ratio
        inhale_hold_end = inhale_end + self.hold_ratio
        exhale_end = 1.0 - self.hold_ratio

        if phase < inhale_end:
            # Inhaling: smooth rise
            t = phase / inhale_end
            value = self._ease_in_out(t)
            return (value, BreathPhase.INHALE)

        elif phase < inhale_hold_end:
            # Brief hold at top
            return (1.0, BreathPhase.INHALE_HOLD)

        elif phase < exhale_end:
            # Exhaling: smooth fall
            t = (phase - inhale_hold_end) / (exhale_end - inhale_hold_end)
            value = 1.0 - self._ease_in_out(t)
            return (value, BreathPhase.EXHALE)

        else:
            # Brief hold at bottom
            return (0.0, BreathPhase.EXHALE_HOLD)

    def _ease_in_out(self, t: float) -> float:
        """Smooth ease-in-out curve."""
        if t < 0.5:
            return 2 * t * t
        return 1 - pow(-2 * t + 2, 2) / 2

    def update(
        self,
        pose: Pose,
        dt: float,
        exertion: Optional[ExertionLevel] = None,
    ) -> Pose:
        """
        Update breathing and return modified pose.

        Args:
            pose: Current animation pose
            dt: Time step in seconds
            exertion: Optional exertion level override

        Returns:
            Modified pose with breathing applied
        """
        if dt <= 0:
            return pose

        result = pose.copy()

        # Update exertion level if specified
        if exertion is not None and exertion != self._target_exertion:
            self._target_exertion = exertion
            target_params = BREATHING_PRESETS[exertion]
            # Smooth transition would be handled here
            self._current_params = target_params

        # Update phase
        params = self._current_params
        phase_delta = dt * params.breath_rate
        self._current_phase = (self._current_phase + phase_delta) % 1.0

        # Calculate breath value
        breath_value, breath_phase = self._calculate_breath_value(self._current_phase)
        self._current_breath_value = breath_value

        # Apply to chest
        self._apply_chest_expansion(result, pose, breath_value, params)

        # Apply to spine
        self._apply_spine_curve(result, pose, breath_value, params)

        # Apply to shoulders
        self._apply_shoulder_motion(result, pose, breath_value, params)

        # Apply to neck
        if self.neck_bone >= 0:
            self._apply_neck_extension(result, pose, breath_value, params)

        return result

    def _apply_chest_expansion(
        self,
        result: Pose,
        base: Pose,
        breath_value: float,
        params: BreathingParams,
    ) -> None:
        """Apply chest expansion/contraction."""
        chest_pos = base.get_bone_position(self.chest_bone)

        # Expand forward and up during inhale
        expansion = params.chest_expansion * breath_value
        offset = (expansion * 0.3, expansion * 0.7, 0.0)

        new_pos = vec3_add(chest_pos, offset)
        result.set_bone_position(self.chest_bone, new_pos)

        # Slight rotation for more natural motion
        chest_rot = base.get_bone_rotation(self.chest_bone)
        pitch = quat_from_axis_angle((1.0, 0.0, 0.0), -params.spine_curve * 0.3 * breath_value)
        result.set_bone_rotation(self.chest_bone, quat_multiply(chest_rot, pitch))

    def _apply_spine_curve(
        self,
        result: Pose,
        base: Pose,
        breath_value: float,
        params: BreathingParams,
    ) -> None:
        """Apply spine curvature during breathing."""
        if not self.spine_bones:
            return

        # Distribute curve across spine bones
        curve_per_bone = params.spine_curve * breath_value / len(self.spine_bones)

        for i, spine_idx in enumerate(self.spine_bones):
            spine_rot = base.get_bone_rotation(spine_idx)

            # Upper spine curves more than lower
            weight = (i + 1) / len(self.spine_bones)
            bone_curve = curve_per_bone * weight

            # Curve backward slightly during inhale (chest opens)
            pitch = quat_from_axis_angle((1.0, 0.0, 0.0), -bone_curve)
            result.set_bone_rotation(spine_idx, quat_multiply(spine_rot, pitch))

    def _apply_shoulder_motion(
        self,
        result: Pose,
        base: Pose,
        breath_value: float,
        params: BreathingParams,
    ) -> None:
        """Apply shoulder rise/fall during breathing."""
        shoulder_rise = params.shoulder_rise * breath_value

        for i, shoulder_idx in enumerate(self.shoulder_bones):
            shoulder_pos = base.get_bone_position(shoulder_idx)

            # Rise up and slightly back
            offset = (-shoulder_rise * 0.2, shoulder_rise, 0.0)
            new_pos = vec3_add(shoulder_pos, offset)
            result.set_bone_position(shoulder_idx, new_pos)

        # Apply to clavicles if present
        for i, clavicle_idx in enumerate(self.clavicle_bones):
            clavicle_rot = base.get_bone_rotation(clavicle_idx)

            # Rotate up during inhale
            roll_angle = shoulder_rise * 2.0  # Amplify for rotation
            side = -1.0 if i == 0 else 1.0  # Left vs right
            roll = quat_from_axis_angle((0.0, 0.0, side), roll_angle)

            result.set_bone_rotation(clavicle_idx, quat_multiply(clavicle_rot, roll))

    def _apply_neck_extension(
        self,
        result: Pose,
        base: Pose,
        breath_value: float,
        params: BreathingParams,
    ) -> None:
        """Apply subtle neck extension during deep breaths."""
        if self.neck_bone < 0:
            return

        neck_rot = base.get_bone_rotation(self.neck_bone)

        # Slight extension (head back) during inhale
        extension = self.neck_extension * breath_value
        pitch = quat_from_axis_angle((1.0, 0.0, 0.0), -extension)

        result.set_bone_rotation(self.neck_bone, quat_multiply(neck_rot, pitch))

    def set_exertion(self, level: ExertionLevel) -> None:
        """
        Set exertion level.

        Args:
            level: New exertion level
        """
        self._target_exertion = level
        self._current_params = BREATHING_PRESETS[level]

    def set_breath_rate(self, rate: float) -> None:
        """
        Set breathing rate directly.

        Args:
            rate: Breaths per second (Hz)
        """
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self._current_params = BreathingParams(
            breath_rate=rate,
            chest_expansion=self._current_params.chest_expansion,
            shoulder_rise=self._current_params.shoulder_rise,
            spine_curve=self._current_params.spine_curve,
            inhale_ratio=self._current_params.inhale_ratio,
        )

    def get_current_phase(self) -> BreathPhase:
        """Get current breath phase."""
        _, phase = self._calculate_breath_value(self._current_phase)
        return phase

    def get_breath_value(self) -> float:
        """Get current breath value (0 = exhaled, 1 = inhaled)."""
        return self._current_breath_value

    def is_inhaling(self) -> bool:
        """Check if currently inhaling."""
        phase = self.get_current_phase()
        return phase in (BreathPhase.INHALE, BreathPhase.INHALE_HOLD)

    def reset(self) -> None:
        """Reset breathing state."""
        self._current_phase = 0.0
        self._current_breath_value = 0.0

    def sync_to_phase(self, phase: float) -> None:
        """
        Sync breathing to a specific phase.

        Useful for syncing with other animations or audio.

        Args:
            phase: Target phase (0-1)
        """
        self._current_phase = phase % 1.0
