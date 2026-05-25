"""
Procedural Locomotion.

Generates walk and run cycles procedurally based on parameters.
Useful for multi-legged creatures, terrain adaptation, and style variation.

Usage:
    gait = GaitConfig(step_height=0.15, step_length=0.6, cycle_duration=0.5)
    locomotion = ProceduralLocomotion(skeleton, gait)
    pose = locomotion.generate_walk_cycle(speed=1.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Protocol, Dict

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


class Skeleton(Protocol):
    """Protocol for skeleton data."""

    def get_bone_count(self) -> int:
        """Get number of bones."""
        ...

    def get_bone_name(self, bone_index: int) -> str:
        """Get bone name."""
        ...

    def get_parent_index(self, bone_index: int) -> int:
        """Get parent bone index."""
        ...


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def vec3_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


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


class GaitType(Enum):
    """Types of locomotion gaits."""

    WALK = auto()  # Alternating legs, always one foot on ground
    RUN = auto()  # Flight phase, both feet off ground
    TROT = auto()  # Diagonal pairs move together
    GALLOP = auto()  # Asymmetric four-beat gait
    PACE = auto()  # Lateral pairs move together
    CRAWL = auto()  # Slow four-beat gait


@dataclass
class FootTrajectory:
    """Defines the trajectory of a foot during a step."""

    # Arc parameters
    step_height: float = 0.15  # Maximum height of foot arc
    step_length: float = 0.6  # Length of step forward
    heel_strike_angle: float = math.radians(-15.0)  # Foot angle at heel strike
    toe_off_angle: float = math.radians(25.0)  # Foot angle at toe off

    # Timing
    stance_ratio: float = 0.6  # Ratio of cycle spent on ground (walk)
    swing_ratio: float = 0.4  # Ratio of cycle in air

    # Shape
    arc_exponent: float = 2.0  # Controls arc shape (2 = parabola)
    forward_offset: float = 0.0  # Offset from hip in rest position

    def __post_init__(self):
        if self.step_height < 0:
            raise ValueError("step_height must be >= 0")
        if self.step_length < 0:
            raise ValueError("step_length must be >= 0")
        if not (0.0 < self.stance_ratio < 1.0):
            raise ValueError("stance_ratio must be in (0, 1)")
        self.swing_ratio = 1.0 - self.stance_ratio

    def sample_stance(self, t: float) -> Tuple[Vec3, float]:
        """
        Sample foot position during stance phase.

        Args:
            t: Normalized time in stance phase (0-1)

        Returns:
            (relative_position, foot_rotation_angle)
        """
        # Foot slides back relative to body moving forward
        forward = self.step_length * 0.5 - self.step_length * t
        height = 0.0  # On ground
        lateral = 0.0

        # Foot angle transitions from heel strike to toe off
        angle = self.heel_strike_angle + (self.toe_off_angle - self.heel_strike_angle) * t

        return ((forward, height, lateral), angle)

    def sample_swing(self, t: float) -> Tuple[Vec3, float]:
        """
        Sample foot position during swing phase.

        Args:
            t: Normalized time in swing phase (0-1)

        Returns:
            (relative_position, foot_rotation_angle)
        """
        # Foot moves forward in arc
        forward = -self.step_length * 0.5 + self.step_length * t

        # Parabolic arc: h(t) = 4 * h_max * t * (1 - t)
        height = 4.0 * self.step_height * t * (1.0 - t)

        # Apply arc exponent for different shapes
        if self.arc_exponent != 2.0:
            arc_t = math.sin(t * math.pi)  # Smooth 0-1-0
            height = self.step_height * pow(arc_t, 2.0 / self.arc_exponent)

        lateral = 0.0

        # Foot angle returns to heel strike position
        angle = self.toe_off_angle + (self.heel_strike_angle - self.toe_off_angle) * t

        return ((forward, height, lateral), angle)

    def sample(self, phase: float) -> Tuple[Vec3, float, bool]:
        """
        Sample foot position at any phase.

        Args:
            phase: Cycle phase (0-1)

        Returns:
            (relative_position, foot_rotation_angle, is_on_ground)
        """
        if phase < self.stance_ratio:
            # Stance phase
            t = phase / self.stance_ratio
            pos, angle = self.sample_stance(t)
            return (pos, angle, True)
        else:
            # Swing phase
            t = (phase - self.stance_ratio) / self.swing_ratio
            pos, angle = self.sample_swing(t)
            return (pos, angle, False)


@dataclass
class BodyDynamics:
    """Body motion parameters during locomotion."""

    # Vertical bob
    bob_amplitude: float = 0.03  # Up/down motion amplitude
    bob_frequency: float = 2.0  # Bobs per cycle (2 for biped)

    # Lateral sway
    sway_amplitude: float = 0.02  # Side-to-side amplitude
    sway_offset: float = 0.0  # Phase offset from step

    # Forward lean
    lean_angle: float = math.radians(5.0)  # Base forward lean
    speed_lean_factor: float = 0.05  # Additional lean per unit speed

    # Hip rotation
    hip_rotation_amplitude: float = math.radians(10.0)
    hip_rotation_offset: float = 0.0

    # Spine twist
    spine_twist_amplitude: float = math.radians(5.0)

    def __post_init__(self):
        if self.bob_amplitude < 0:
            raise ValueError("bob_amplitude must be >= 0")
        if self.sway_amplitude < 0:
            raise ValueError("sway_amplitude must be >= 0")

    def sample(self, phase: float, speed: float = 1.0) -> Dict[str, float]:
        """
        Sample body dynamics at a given phase.

        Args:
            phase: Cycle phase (0-1)
            speed: Movement speed multiplier

        Returns:
            Dictionary of body adjustments
        """
        # Vertical bob follows sinusoidal pattern
        bob_phase = phase * self.bob_frequency * 2 * math.pi
        vertical_offset = self.bob_amplitude * math.cos(bob_phase)

        # Lateral sway is 90 degrees offset from bob
        sway_phase = (phase + self.sway_offset) * 2 * math.pi
        lateral_offset = self.sway_amplitude * math.sin(sway_phase)

        # Forward lean increases with speed
        total_lean = self.lean_angle + speed * self.speed_lean_factor

        # Hip rotation counter to leg movement
        hip_phase = (phase + self.hip_rotation_offset) * 2 * math.pi
        hip_rotation = self.hip_rotation_amplitude * math.sin(hip_phase)

        # Spine twist for natural motion
        spine_twist = self.spine_twist_amplitude * math.sin(hip_phase)

        return {
            "vertical_offset": vertical_offset,
            "lateral_offset": lateral_offset,
            "forward_lean": total_lean,
            "hip_rotation": hip_rotation,
            "spine_twist": spine_twist,
        }


@dataclass
class GaitConfig:
    """Complete gait configuration."""

    # Timing
    step_height: float = 0.15
    step_length: float = 0.6
    cycle_duration: float = 0.5  # Seconds for full cycle

    # Foot offsets from body
    foot_offset: Vec3 = (0.0, 0.0, 0.15)  # (forward, up, lateral)

    # Gait type
    gait_type: GaitType = GaitType.WALK

    # Phase offsets for each foot (normalized 0-1)
    foot_phases: Dict[str, float] = field(default_factory=dict)

    # Trajectories
    foot_trajectory: FootTrajectory = field(default_factory=FootTrajectory)
    body_dynamics: BodyDynamics = field(default_factory=BodyDynamics)

    # Speed adaptation
    min_speed: float = 0.0
    max_speed: float = 5.0
    walk_to_run_speed: float = 2.0  # Speed at which to transition to run

    def __post_init__(self):
        if self.cycle_duration <= 0:
            raise ValueError("cycle_duration must be > 0")
        if self.min_speed < 0:
            raise ValueError("min_speed must be >= 0")
        if self.max_speed <= self.min_speed:
            raise ValueError("max_speed must be > min_speed")

        # Set default foot phases for biped
        if not self.foot_phases:
            self.foot_phases = {
                "left_foot": 0.0,
                "right_foot": 0.5,  # 180 degrees out of phase
            }

        # Update trajectory with config values
        self.foot_trajectory.step_height = self.step_height
        self.foot_trajectory.step_length = self.step_length

    @classmethod
    def create_walk(cls) -> "GaitConfig":
        """Create a standard walk gait."""
        return cls(
            step_height=0.1,
            step_length=0.5,
            cycle_duration=0.6,
            gait_type=GaitType.WALK,
            foot_trajectory=FootTrajectory(stance_ratio=0.6),
        )

    @classmethod
    def create_run(cls) -> "GaitConfig":
        """Create a standard run gait."""
        return cls(
            step_height=0.2,
            step_length=1.0,
            cycle_duration=0.35,
            gait_type=GaitType.RUN,
            foot_trajectory=FootTrajectory(stance_ratio=0.4),
            body_dynamics=BodyDynamics(
                bob_amplitude=0.05,
                lean_angle=math.radians(10.0),
            ),
        )

    @classmethod
    def create_quadruped_walk(cls) -> "GaitConfig":
        """Create a quadruped walk gait."""
        return cls(
            step_height=0.1,
            step_length=0.4,
            cycle_duration=0.8,
            gait_type=GaitType.WALK,
            foot_phases={
                "front_left": 0.0,
                "back_right": 0.25,
                "front_right": 0.5,
                "back_left": 0.75,
            },
        )

    @classmethod
    def create_quadruped_trot(cls) -> "GaitConfig":
        """Create a quadruped trot gait."""
        return cls(
            step_height=0.15,
            step_length=0.6,
            cycle_duration=0.5,
            gait_type=GaitType.TROT,
            foot_phases={
                "front_left": 0.0,
                "back_right": 0.0,  # Diagonal pair
                "front_right": 0.5,
                "back_left": 0.5,
            },
        )


@dataclass
class LegConfig:
    """Configuration for a single leg."""

    hip_bone: int
    thigh_bone: int
    calf_bone: int
    foot_bone: int
    is_left: bool = True
    phase_offset: float = 0.0

    def __post_init__(self):
        if self.hip_bone < 0:
            raise ValueError("hip_bone must be >= 0")
        if self.thigh_bone < 0:
            raise ValueError("thigh_bone must be >= 0")
        if self.calf_bone < 0:
            raise ValueError("calf_bone must be >= 0")
        if self.foot_bone < 0:
            raise ValueError("foot_bone must be >= 0")


@dataclass
class ArmConfig:
    """Configuration for a single arm."""

    shoulder_bone: int
    upper_arm_bone: int
    lower_arm_bone: int
    is_left: bool = True
    swing_amplitude: float = math.radians(30.0)
    phase_offset: float = 0.5  # Counter to opposite leg

    def __post_init__(self):
        if self.shoulder_bone < 0:
            raise ValueError("shoulder_bone must be >= 0")


@dataclass
class ProceduralLocomotion:
    """
    Generates procedural locomotion animations.

    Creates walk/run cycles based on gait parameters without
    pre-authored animation data.
    """

    skeleton: Skeleton
    gait_config: GaitConfig

    # Bone configuration
    hips_bone: int = -1
    spine_bones: List[int] = field(default_factory=list)
    legs: List[LegConfig] = field(default_factory=list)
    arms: List[ArmConfig] = field(default_factory=list)

    # Runtime state
    _current_phase: float = 0.0
    _current_speed: float = 0.0
    _accumulated_time: float = 0.0

    def __post_init__(self):
        if self.gait_config is None:
            raise ValueError("gait_config must be provided")

    def configure_biped(
        self,
        hips: int,
        spine: List[int],
        left_leg: LegConfig,
        right_leg: LegConfig,
        left_arm: Optional[ArmConfig] = None,
        right_arm: Optional[ArmConfig] = None,
    ) -> None:
        """
        Configure for a biped character.

        Args:
            hips: Hip bone index
            spine: List of spine bone indices
            left_leg: Left leg configuration
            right_leg: Right leg configuration
            left_arm: Optional left arm configuration
            right_arm: Optional right arm configuration
        """
        self.hips_bone = hips
        self.spine_bones = spine
        self.legs = [left_leg, right_leg]

        left_leg.phase_offset = 0.0
        right_leg.phase_offset = 0.5

        if left_arm and right_arm:
            self.arms = [left_arm, right_arm]
            left_arm.phase_offset = 0.5  # Counter to left leg
            right_arm.phase_offset = 0.0  # Counter to right leg

    def update(self, dt: float, speed: float) -> float:
        """
        Update locomotion phase.

        Args:
            dt: Time step in seconds
            speed: Current movement speed

        Returns:
            Current cycle phase (0-1)
        """
        self._current_speed = max(0.0, speed)

        if self._current_speed < 0.01:
            return self._current_phase

        # Adjust cycle duration based on speed
        speed_factor = self._current_speed / (
            (self.gait_config.max_speed + self.gait_config.min_speed) / 2
        )
        adjusted_duration = self.gait_config.cycle_duration / max(0.1, speed_factor)

        # Update phase
        phase_delta = dt / adjusted_duration
        self._current_phase = (self._current_phase + phase_delta) % 1.0
        self._accumulated_time += dt

        return self._current_phase

    def generate_walk_cycle(self, speed: float, base_pose: Optional[Pose] = None) -> Optional[Pose]:
        """
        Generate a walk cycle pose.

        Args:
            speed: Movement speed
            base_pose: Base pose to modify (or None for new pose)

        Returns:
            Generated walk pose
        """
        if base_pose is None:
            return None

        result = base_pose.copy()

        # Get body dynamics
        body = self.gait_config.body_dynamics.sample(self._current_phase, speed)

        # Apply body motion to hips
        if self.hips_bone >= 0:
            hip_pos = base_pose.get_bone_position(self.hips_bone)
            hip_pos = vec3_add(
                hip_pos,
                (0.0, body["vertical_offset"], body["lateral_offset"])
            )
            result.set_bone_position(self.hips_bone, hip_pos)

            # Apply hip rotation
            hip_rot = base_pose.get_bone_rotation(self.hips_bone)
            yaw_rotation = quat_from_axis_angle((0.0, 1.0, 0.0), body["hip_rotation"])
            lean_rotation = quat_from_axis_angle((1.0, 0.0, 0.0), body["forward_lean"])
            hip_rot = quat_multiply(quat_multiply(hip_rot, lean_rotation), yaw_rotation)
            result.set_bone_rotation(self.hips_bone, hip_rot)

        # Apply spine motion
        if self.spine_bones:
            twist_per_bone = body["spine_twist"] / len(self.spine_bones)
            for spine_bone in self.spine_bones:
                spine_rot = base_pose.get_bone_rotation(spine_bone)
                twist = quat_from_axis_angle((0.0, 1.0, 0.0), twist_per_bone)
                result.set_bone_rotation(spine_bone, quat_multiply(spine_rot, twist))

        # Apply leg motion
        for leg in self.legs:
            phase = (self._current_phase + leg.phase_offset) % 1.0
            pos, angle, on_ground = self.gait_config.foot_trajectory.sample(phase)

            # Get base foot position
            foot_pos = base_pose.get_bone_position(leg.foot_bone)

            # Apply trajectory offset
            lateral_sign = -1.0 if leg.is_left else 1.0
            offset = (pos[0], pos[1], pos[2] * lateral_sign)
            new_foot_pos = vec3_add(foot_pos, offset)
            result.set_bone_position(leg.foot_bone, new_foot_pos)

            # Apply foot rotation
            foot_rot = base_pose.get_bone_rotation(leg.foot_bone)
            pitch = quat_from_axis_angle((1.0, 0.0, 0.0), angle)
            result.set_bone_rotation(leg.foot_bone, quat_multiply(foot_rot, pitch))

        # Apply arm swing (counter to legs)
        for arm in self.arms:
            phase = (self._current_phase + arm.phase_offset) % 1.0
            swing_angle = arm.swing_amplitude * math.sin(phase * 2 * math.pi)

            upper_arm_rot = base_pose.get_bone_rotation(arm.upper_arm_bone)
            swing = quat_from_axis_angle((1.0, 0.0, 0.0), swing_angle)
            result.set_bone_rotation(arm.upper_arm_bone, quat_multiply(upper_arm_rot, swing))

        return result

    def generate_run_cycle(self, speed: float, base_pose: Optional[Pose] = None) -> Optional[Pose]:
        """
        Generate a run cycle pose.

        Args:
            speed: Movement speed
            base_pose: Base pose to modify

        Returns:
            Generated run pose
        """
        # Run is similar to walk but with different parameters
        # Store original config
        original_stance_ratio = self.gait_config.foot_trajectory.stance_ratio

        # Adjust for run
        self.gait_config.foot_trajectory.stance_ratio = 0.4

        result = self.generate_walk_cycle(speed, base_pose)

        # Restore
        self.gait_config.foot_trajectory.stance_ratio = original_stance_ratio

        return result

    def generate_adaptive_cycle(
        self,
        speed: float,
        base_pose: Optional[Pose] = None,
    ) -> Optional[Pose]:
        """
        Generate a cycle that adapts between walk and run.

        Args:
            speed: Movement speed
            base_pose: Base pose to modify

        Returns:
            Generated pose blending walk/run based on speed
        """
        if speed < self.gait_config.walk_to_run_speed:
            return self.generate_walk_cycle(speed, base_pose)
        else:
            return self.generate_run_cycle(speed, base_pose)

    def get_foot_contacts(self) -> Dict[str, bool]:
        """
        Get current foot contact states.

        Returns:
            Dictionary mapping foot names to contact state
        """
        contacts = {}

        for i, leg in enumerate(self.legs):
            phase = (self._current_phase + leg.phase_offset) % 1.0
            _, _, on_ground = self.gait_config.foot_trajectory.sample(phase)

            name = "left_foot" if leg.is_left else "right_foot"
            if i > 1:
                name = f"foot_{i}"

            contacts[name] = on_ground

        return contacts

    def get_current_phase(self) -> float:
        """Get current cycle phase (0-1)."""
        return self._current_phase

    def set_phase(self, phase: float) -> None:
        """Set current cycle phase."""
        self._current_phase = phase % 1.0

    def reset(self) -> None:
        """Reset locomotion state."""
        self._current_phase = 0.0
        self._current_speed = 0.0
        self._accumulated_time = 0.0
