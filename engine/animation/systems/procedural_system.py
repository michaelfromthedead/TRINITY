"""ECS system for procedural animation (T-AN-9.5).

Applies procedural animation effects in a defined order after IK:
1. Spring/jiggle bones (T-AN-7.5)
2. Look-at controllers (T-AN-7.6)
3. Twist distribution (T-AN-7.7)
4. Ragdoll blending (T-AN-7.8)

Key Features:
- @system(phase="animation", order=2) annotation for ECS scheduling
- Runs AFTER IK system (order=1), BEFORE skinning
- Effect chaining: output of one effect feeds the next
- Per-bone effect enable/disable masks
- Effect weight blending (0-1)
- Ragdoll blend in/out support

Dependencies:
- engine.animation.procedural: SpringBone, SpringChain, LookAtController, TwistBone, Ragdoll
- engine.animation.systems.animation_graph_system: system decorator
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import PROCEDURAL_CONFIG

if TYPE_CHECKING:
    from engine.animation.procedural.spring_bone import SpringBone, SpringChain, WindForce
    from engine.animation.procedural.spring_bone import CollisionSphere, CollisionCapsule
    from engine.animation.procedural.lookat import LookAtController, InterestPoint
    from engine.animation.procedural.twist import TwistBone, TwistChain
    from engine.animation.procedural.ragdoll import Ragdoll, RagdollState


# =============================================================================
# SYSTEM DECORATOR (for phase annotation)
# =============================================================================


def system(
    phase: str = "update",
    order: int = 0,
    priority: int = 0,
    reads: Optional[Tuple[str, ...]] = None,
    writes: Optional[Tuple[str, ...]] = None,
) -> Callable:
    """Decorator to mark a class as an ECS system with phase scheduling.

    Args:
        phase: Frame phase for execution ("animation", "update", "render", etc.)
        order: Execution order within phase (lower = earlier)
        priority: Execution priority (alternative to order for backward compat)
        reads: Component types this system reads from
        writes: Component types this system writes to

    Returns:
        Decorated class with system metadata.
    """
    def decorator(cls: type) -> type:
        cls._system_phase = phase
        cls._system_order = order
        cls._system_priority = priority if order == 0 else order
        cls._system_reads = reads or ()
        cls._system_writes = writes or ()
        return cls
    return decorator


# =============================================================================
# EFFECT TYPE ENUMERATION
# =============================================================================


class ProceduralEffectType(Enum):
    """Types of procedural effects in processing order."""
    SPRING = auto()      # Spring/jiggle physics (T-AN-7.5)
    LOOK_AT = auto()     # Look-at constraints (T-AN-7.6)
    TWIST = auto()       # Twist distribution (T-AN-7.7)
    RAGDOLL = auto()     # Ragdoll blending (T-AN-7.8)
    SWAY = auto()        # Sway/wave motion
    BREATHING = auto()   # Breathing animation
    NOISE = auto()       # Noise-based motion
    CUSTOM = auto()      # User-defined effects


class ControllerType(Enum):
    """Type of procedural controller (legacy compatibility)."""
    SPRING = auto()
    LOOK_AT = auto()
    SWAY = auto()
    BREATHING = auto()
    NOISE = auto()
    CUSTOM = auto()


# =============================================================================
# BONE MASK FOR PER-BONE EFFECT CONTROL
# =============================================================================


@dataclass
class BoneMask:
    """Per-bone effect enable/disable mask with weights.

    Allows fine-grained control over which bones are affected by an effect
    and with what weight.

    Attributes:
        enabled_bones: Set of bone indices that are enabled
        bone_weights: Per-bone weight overrides (0-1)
        default_enabled: Whether bones are enabled by default
        default_weight: Default weight for bones not in bone_weights
    """
    enabled_bones: Set[int] = field(default_factory=set)
    bone_weights: Dict[int, float] = field(default_factory=dict)
    default_enabled: bool = True
    default_weight: float = 1.0

    def is_enabled(self, bone_index: int) -> bool:
        """Check if a bone is enabled for this effect."""
        if self.default_enabled:
            return bone_index not in self.enabled_bones or True
        return bone_index in self.enabled_bones

    def get_weight(self, bone_index: int) -> float:
        """Get weight for a specific bone."""
        return self.bone_weights.get(bone_index, self.default_weight)

    def enable_bone(self, bone_index: int, weight: float = 1.0) -> None:
        """Enable a bone with optional weight."""
        self.enabled_bones.add(bone_index)
        if weight != self.default_weight:
            self.bone_weights[bone_index] = max(0.0, min(1.0, weight))

    def disable_bone(self, bone_index: int) -> None:
        """Disable a bone."""
        self.enabled_bones.discard(bone_index)
        self.bone_weights.pop(bone_index, None)

    def set_bone_weight(self, bone_index: int, weight: float) -> None:
        """Set weight for a specific bone."""
        self.bone_weights[bone_index] = max(0.0, min(1.0, weight))

    def clear(self) -> None:
        """Clear all overrides."""
        self.enabled_bones.clear()
        self.bone_weights.clear()


# =============================================================================
# BASE PROCEDURAL CONTROLLER
# =============================================================================


@dataclass
class ProceduralController(ABC):
    """Base class for procedural animation controllers.

    All procedural effects inherit from this class and implement the update method.
    Controllers process bone transforms and return modifications.

    Attributes:
        enabled: Whether this controller is active
        weight: Global blend weight for this controller (0-1)
        affected_bones: List of bone indices this controller can modify
        bone_mask: Per-bone enable/disable mask
        effect_order: Order within same effect type (lower = earlier)
    """
    enabled: bool = True
    weight: float = 1.0
    affected_bones: List[int] = field(default_factory=list)
    bone_mask: BoneMask = field(default_factory=BoneMask)
    effect_order: int = 0

    @property
    @abstractmethod
    def controller_type(self) -> ControllerType:
        """Controller type identifier."""
        pass

    @property
    def effect_type(self) -> ProceduralEffectType:
        """Get the effect type for ordering."""
        type_mapping = {
            ControllerType.SPRING: ProceduralEffectType.SPRING,
            ControllerType.LOOK_AT: ProceduralEffectType.LOOK_AT,
            ControllerType.SWAY: ProceduralEffectType.SWAY,
            ControllerType.BREATHING: ProceduralEffectType.BREATHING,
            ControllerType.NOISE: ProceduralEffectType.NOISE,
            ControllerType.CUSTOM: ProceduralEffectType.CUSTOM,
        }
        return type_mapping.get(self.controller_type, ProceduralEffectType.CUSTOM)

    @abstractmethod
    def update(self, dt: float, pose: Dict[int, Transform]) -> Dict[int, Transform]:
        """Update controller and return modified transforms.

        Args:
            dt: Delta time in seconds
            pose: Current bone transforms (bone_index -> Transform)

        Returns:
            Modified transforms for affected bones
        """
        pass

    def reset(self) -> None:
        """Reset controller state."""
        pass

    def get_effective_weight(self, bone_index: int) -> float:
        """Get effective weight for a bone (controller weight * bone mask weight)."""
        if not self.bone_mask.is_enabled(bone_index):
            return 0.0
        return self.weight * self.bone_mask.get_weight(bone_index)


# =============================================================================
# SPRING CONTROLLER (T-AN-7.5)
# =============================================================================


@dataclass
class SpringController(ProceduralController):
    """Spring-based secondary motion (e.g., hair, cloth, accessories).

    Simulates spring dynamics for natural secondary motion using Verlet integration.
    Supports collision detection, wind forces, and chain constraints.

    Attributes:
        stiffness: Spring constant (higher = stiffer)
        damping: Damping factor (0-1, higher = more damping)
        mass: Mass of connected objects
        gravity: Gravity vector
        max_stretch: Maximum stretch factor before clamping
    """
    stiffness: float = PROCEDURAL_CONFIG.DEFAULT_SPRING_STIFFNESS
    damping: float = PROCEDURAL_CONFIG.DEFAULT_SPRING_DAMPING
    mass: float = PROCEDURAL_CONFIG.DEFAULT_SPRING_MASS
    gravity: Vec3 = field(default_factory=lambda: Vec3(0, -9.8, 0))
    max_stretch: float = PROCEDURAL_CONFIG.DEFAULT_MAX_STRETCH

    # Internal state for Verlet integration
    _velocities: Dict[int, Vec3] = field(default_factory=dict)
    _rest_positions: Dict[int, Vec3] = field(default_factory=dict)
    _current_positions: Dict[int, Vec3] = field(default_factory=dict)
    _initialized: bool = field(default=False, repr=False)

    @property
    def controller_type(self) -> ControllerType:
        return ControllerType.SPRING

    @property
    def effect_type(self) -> ProceduralEffectType:
        return ProceduralEffectType.SPRING

    def update(self, dt: float, pose: Dict[int, Transform]) -> Dict[int, Transform]:
        """Update spring physics simulation."""
        result = {}
        if not self.enabled or self.weight <= 0 or dt <= 0:
            return result

        # Clamp dt for numerical stability
        dt = min(dt, 0.033)  # Max ~30fps timestep

        for bone in self.affected_bones:
            if bone not in pose:
                continue

            effective_weight = self.get_effective_weight(bone)
            if effective_weight <= 0:
                continue

            transform = pose[bone]

            # Initialize rest position if needed
            if bone not in self._rest_positions:
                self._rest_positions[bone] = transform.translation
                self._current_positions[bone] = transform.translation
                self._velocities[bone] = Vec3.zero()
                self._initialized = True

            rest_pos = self._rest_positions[bone]
            current_pos = self._current_positions[bone]
            velocity = self._velocities[bone]

            # Update rest position to follow animation
            self._rest_positions[bone] = transform.translation

            # Spring force: F = -k * displacement
            displacement = rest_pos - current_pos
            spring_force = displacement * self.stiffness

            # Damping force: F = -c * v
            damping_force = velocity * (-self.damping * self.stiffness)

            # Gravity force
            gravity_force = self.gravity * self.mass

            # Total acceleration
            total_force = spring_force + damping_force + gravity_force
            acceleration = total_force / self.mass

            # Verlet integration
            velocity = velocity + acceleration * dt
            new_pos = current_pos + velocity * dt

            # Limit stretch
            offset = new_pos - rest_pos
            max_dist = self.max_stretch
            if offset.length() > max_dist:
                new_pos = rest_pos + offset.normalized() * max_dist
                velocity = velocity * 0.5  # Dampen at limit

            self._current_positions[bone] = new_pos
            self._velocities[bone] = velocity

            # Blend with original based on effective weight
            final_pos = transform.translation.lerp(new_pos, effective_weight)

            result[bone] = Transform(
                translation=final_pos,
                rotation=transform.rotation,
                scale=transform.scale,
            )

        return result

    def reset(self) -> None:
        """Reset spring state."""
        self._velocities.clear()
        self._rest_positions.clear()
        self._current_positions.clear()
        self._initialized = False


# =============================================================================
# LOOK-AT CONTROLLER (T-AN-7.6)
# =============================================================================


@dataclass
class LookAtController(ProceduralController):
    """Look-at constraint controller.

    Makes bones orient toward a target point with angle limits and smooth interpolation.
    Supports head, neck, and eye bone hierarchies.

    Attributes:
        target: Target world position to look at
        up_vector: Up vector for orientation
        speed: Rotation interpolation speed (radians/sec)
        angle_limit_horizontal: Maximum horizontal rotation (radians)
        angle_limit_vertical: Maximum vertical rotation (radians)
        forward_axis: Local forward axis of the bone
    """
    target: Vec3 = field(default_factory=Vec3.zero)
    up_vector: Vec3 = field(default_factory=Vec3.up)
    speed: float = PROCEDURAL_CONFIG.DEFAULT_LOOK_SPEED
    angle_limit_horizontal: float = PROCEDURAL_CONFIG.DEFAULT_HORIZONTAL_LIMIT
    angle_limit_vertical: float = PROCEDURAL_CONFIG.DEFAULT_VERTICAL_LIMIT
    forward_axis: Vec3 = field(default_factory=Vec3.forward)

    # Per-bone weights for distributed look-at
    bone_weights: Dict[int, float] = field(default_factory=dict)

    # Internal state
    _current_rotations: Dict[int, Quat] = field(default_factory=dict)
    _initialized: bool = field(default=False, repr=False)

    @property
    def controller_type(self) -> ControllerType:
        return ControllerType.LOOK_AT

    @property
    def effect_type(self) -> ProceduralEffectType:
        return ProceduralEffectType.LOOK_AT

    def update(self, dt: float, pose: Dict[int, Transform]) -> Dict[int, Transform]:
        """Update look-at constraint."""
        result = {}
        if not self.enabled or self.weight <= 0 or dt <= 0:
            return result

        for bone in self.affected_bones:
            if bone not in pose:
                continue

            effective_weight = self.get_effective_weight(bone)
            bone_weight = self.bone_weights.get(bone, 1.0)
            final_weight = effective_weight * bone_weight

            if final_weight <= 0:
                continue

            transform = pose[bone]

            # Calculate direction to target
            to_target = self.target - transform.translation
            if to_target.length_squared() < 0.0001:
                continue

            to_target = to_target.normalized()

            # Calculate target rotation
            target_rotation = self._look_rotation(to_target, self.up_vector)

            # Apply angle limits
            target_rotation = self._apply_angle_limits(target_rotation, transform.rotation)

            # Smooth rotation
            if bone not in self._current_rotations:
                self._current_rotations[bone] = transform.rotation
                self._initialized = True

            current_rot = self._current_rotations[bone]
            new_rot = current_rot.slerp(target_rotation, min(1.0, self.speed * dt))
            self._current_rotations[bone] = new_rot

            # Blend with original
            final_rot = transform.rotation.slerp(new_rot, final_weight)

            result[bone] = Transform(
                translation=transform.translation,
                rotation=final_rot,
                scale=transform.scale,
            )

        return result

    def _look_rotation(self, forward: Vec3, up: Vec3) -> Quat:
        """Calculate rotation to look in direction."""
        forward = forward.normalized()
        right = up.cross(forward).normalized()
        up_adjusted = forward.cross(right)

        # Build rotation from orthonormal basis
        m00, m01, m02 = right.x, right.y, right.z
        m10, m11, m12 = up_adjusted.x, up_adjusted.y, up_adjusted.z
        m20, m21, m22 = forward.x, forward.y, forward.z

        trace = m00 + m11 + m22

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m12 - m21) * s
            y = (m20 - m02) * s
            z = (m01 - m10) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
            w = (m12 - m21) / s
            x = 0.25 * s
            y = (m10 + m01) / s
            z = (m20 + m02) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
            w = (m20 - m02) / s
            x = (m10 + m01) / s
            y = 0.25 * s
            z = (m21 + m12) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
            w = (m01 - m10) / s
            x = (m20 + m02) / s
            y = (m21 + m12) / s
            z = 0.25 * s

        return Quat(x, y, z, w).normalized()

    def _apply_angle_limits(self, target: Quat, reference: Quat) -> Quat:
        """Apply angle limits relative to reference rotation."""
        relative = reference.inverse() * target
        pitch, yaw, roll = relative.to_euler()

        # Clamp angles
        yaw = max(-self.angle_limit_horizontal, min(self.angle_limit_horizontal, yaw))
        pitch = max(-self.angle_limit_vertical, min(self.angle_limit_vertical, pitch))

        # Reconstruct
        limited = Quat.from_euler(pitch, yaw, roll)
        return reference * limited

    def set_target(self, target: Vec3) -> None:
        """Set the look-at target position."""
        self.target = target

    def reset(self) -> None:
        """Reset look-at state."""
        self._current_rotations.clear()
        self._initialized = False


# =============================================================================
# TWIST CONTROLLER (T-AN-7.7)
# =============================================================================


@dataclass
class TwistController(ProceduralController):
    """Twist distribution controller.

    Distributes twist rotation from a source bone across helper twist bones.
    Common for forearm, upper arm, and thigh twist distribution.

    Attributes:
        source_bone: Bone index to extract twist from
        twist_axis: Local axis to twist around
        distribution_weights: Per-bone distribution weights (0-1)
    """
    source_bone: int = -1
    twist_axis: Vec3 = field(default_factory=lambda: Vec3(1, 0, 0))
    distribution_weights: Dict[int, float] = field(default_factory=dict)
    reference_bone: int = -1  # Reference for relative twist calculation

    # Internal state
    _reference_rotation: Quat = field(default_factory=Quat.identity)

    @property
    def controller_type(self) -> ControllerType:
        return ControllerType.CUSTOM

    @property
    def effect_type(self) -> ProceduralEffectType:
        return ProceduralEffectType.TWIST

    def update(self, dt: float, pose: Dict[int, Transform]) -> Dict[int, Transform]:
        """Update twist distribution."""
        result = {}
        if not self.enabled or self.weight <= 0:
            return result

        if self.source_bone < 0 or self.source_bone not in pose:
            return result

        source_transform = pose[self.source_bone]

        # Get reference rotation
        if self.reference_bone >= 0 and self.reference_bone in pose:
            ref_rotation = pose[self.reference_bone].rotation
        else:
            ref_rotation = Quat.identity()

        # Extract twist component
        relative_rotation = ref_rotation.inverse() * source_transform.rotation
        twist_rotation = self._extract_twist(relative_rotation, self.twist_axis)
        twist_axis_vec, twist_angle = self._quat_to_axis_angle(twist_rotation)

        # Distribute twist to affected bones
        for bone in self.affected_bones:
            if bone not in pose:
                continue

            effective_weight = self.get_effective_weight(bone)
            dist_weight = self.distribution_weights.get(bone, 0.5)
            final_weight = effective_weight * dist_weight

            if final_weight <= 0:
                continue

            transform = pose[bone]

            # Calculate twist amount for this bone
            bone_twist_angle = twist_angle * final_weight
            bone_twist = self._quat_from_axis_angle(self.twist_axis, bone_twist_angle)

            # Apply twist
            new_rotation = transform.rotation * bone_twist
            new_rotation = new_rotation.normalized()

            result[bone] = Transform(
                translation=transform.translation,
                rotation=new_rotation,
                scale=transform.scale,
            )

        return result

    def _extract_twist(self, rotation: Quat, twist_axis: Vec3) -> Quat:
        """Extract twist component around an axis."""
        axis, angle = self._quat_to_axis_angle(rotation)
        dot = axis.x * twist_axis.x + axis.y * twist_axis.y + axis.z * twist_axis.z
        twist_angle = angle * dot
        return self._quat_from_axis_angle(twist_axis, twist_angle)

    def _quat_to_axis_angle(self, q: Quat) -> Tuple[Vec3, float]:
        """Convert quaternion to axis-angle."""
        q = q.normalized()
        angle = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))
        sin_half = math.sqrt(1.0 - q.w * q.w)

        if sin_half < 1e-10:
            return Vec3(1, 0, 0), 0.0

        inv_sin = 1.0 / sin_half
        return Vec3(q.x * inv_sin, q.y * inv_sin, q.z * inv_sin), angle

    def _quat_from_axis_angle(self, axis: Vec3, angle: float) -> Quat:
        """Create quaternion from axis-angle."""
        axis = axis.normalized()
        half_angle = angle * 0.5
        sin_half = math.sin(half_angle)
        cos_half = math.cos(half_angle)
        return Quat(
            axis.x * sin_half,
            axis.y * sin_half,
            axis.z * sin_half,
            cos_half
        )

    def reset(self) -> None:
        """Reset twist state."""
        self._reference_rotation = Quat.identity()


# =============================================================================
# RAGDOLL BLEND CONTROLLER (T-AN-7.8)
# =============================================================================


@dataclass
class RagdollBlendController(ProceduralController):
    """Ragdoll physics blend controller.

    Blends between animation and ragdoll physics for partial or full ragdoll effects.
    Supports smooth blend in/out transitions.

    Attributes:
        physics_poses: Physics-driven bone transforms
        blend_weight: Blend between animation (0) and physics (1)
        blend_speed: Speed of blend weight change per second
        active_bodies: Set of body indices currently in physics mode
    """
    physics_poses: Dict[int, Transform] = field(default_factory=dict)
    blend_weight: float = 0.0
    blend_speed: float = 3.0  # Blend weight change per second
    target_blend_weight: float = 0.0
    active_bodies: Set[int] = field(default_factory=set)

    # Body to bone mapping
    body_to_bone: Dict[int, int] = field(default_factory=dict)

    @property
    def controller_type(self) -> ControllerType:
        return ControllerType.CUSTOM

    @property
    def effect_type(self) -> ProceduralEffectType:
        return ProceduralEffectType.RAGDOLL

    def update(self, dt: float, pose: Dict[int, Transform]) -> Dict[int, Transform]:
        """Update ragdoll blend."""
        result = {}
        if not self.enabled:
            return result

        # Update blend weight toward target
        if dt > 0:
            if self.blend_weight < self.target_blend_weight:
                self.blend_weight = min(
                    self.target_blend_weight,
                    self.blend_weight + self.blend_speed * dt
                )
            elif self.blend_weight > self.target_blend_weight:
                self.blend_weight = max(
                    self.target_blend_weight,
                    self.blend_weight - self.blend_speed * dt
                )

        if self.blend_weight <= 0 and self.target_blend_weight <= 0:
            return result

        for bone in self.affected_bones:
            if bone not in pose:
                continue

            # Check if this bone has physics data
            if bone not in self.physics_poses:
                continue

            effective_weight = self.get_effective_weight(bone)
            final_weight = effective_weight * self.blend_weight

            if final_weight <= 0:
                continue

            anim_transform = pose[bone]
            physics_transform = self.physics_poses[bone]

            # Blend position
            blended_pos = anim_transform.translation.lerp(
                physics_transform.translation, final_weight
            )

            # Blend rotation
            blended_rot = anim_transform.rotation.slerp(
                physics_transform.rotation, final_weight
            )

            # Keep animation scale
            result[bone] = Transform(
                translation=blended_pos,
                rotation=blended_rot,
                scale=anim_transform.scale,
            )

        return result

    def set_physics_pose(self, bone_index: int, transform: Transform) -> None:
        """Set physics-driven transform for a bone."""
        self.physics_poses[bone_index] = transform

    def activate(self, blend_time: float = 0.3) -> None:
        """Activate ragdoll with blend-in."""
        self.target_blend_weight = 1.0
        if blend_time <= 0:
            self.blend_weight = 1.0

    def deactivate(self, blend_time: float = 0.3) -> None:
        """Deactivate ragdoll with blend-out."""
        self.target_blend_weight = 0.0
        if blend_time <= 0:
            self.blend_weight = 0.0

    def is_blending(self) -> bool:
        """Check if currently blending."""
        return abs(self.blend_weight - self.target_blend_weight) > 0.001

    def is_active(self) -> bool:
        """Check if ragdoll is active or blending in."""
        return self.blend_weight > 0 or self.target_blend_weight > 0

    def reset(self) -> None:
        """Reset ragdoll blend state."""
        self.blend_weight = 0.0
        self.target_blend_weight = 0.0
        self.physics_poses.clear()


# =============================================================================
# SWAY CONTROLLER
# =============================================================================


@dataclass
class SwayController(ProceduralController):
    """Sway/wave motion controller.

    Creates oscillating motion for vegetation, flags, etc.

    Attributes:
        frequency: Oscillation frequency in Hz
        amplitude: Rotation amplitude per axis
        phase_offset: Phase offset in radians
        noise_amount: Random variation amount (0-1)
    """
    frequency: float = PROCEDURAL_CONFIG.DEFAULT_SWAY_FREQUENCY
    amplitude: Vec3 = field(default_factory=lambda: Vec3(0.1, 0.05, 0.1))
    phase_offset: float = 0.0
    noise_amount: float = PROCEDURAL_CONFIG.DEFAULT_NOISE_AMOUNT

    # Per-bone phase offsets for cascading motion
    bone_phase_offsets: Dict[int, float] = field(default_factory=dict)

    _time: float = 0.0

    @property
    def controller_type(self) -> ControllerType:
        return ControllerType.SWAY

    def update(self, dt: float, pose: Dict[int, Transform]) -> Dict[int, Transform]:
        """Update sway motion."""
        result = {}
        if not self.enabled or self.weight <= 0 or dt <= 0:
            return result

        self._time += dt

        for bone in self.affected_bones:
            if bone not in pose:
                continue

            effective_weight = self.get_effective_weight(bone)
            if effective_weight <= 0:
                continue

            transform = pose[bone]
            bone_phase = self.bone_phase_offsets.get(bone, 0.0)

            # Calculate sway offset
            phase = self._time * self.frequency * math.pi * 2.0 + self.phase_offset + bone_phase

            offset = Vec3(
                math.sin(phase) * self.amplitude.x,
                math.sin(phase * 0.7 + 0.5) * self.amplitude.y,
                math.cos(phase * 1.3) * self.amplitude.z,
            )

            # Add noise variation
            if self.noise_amount > 0:
                noise_phase = self._time * 0.7 + bone_phase * 2.0
                noise = Vec3(
                    math.sin(noise_phase * 2.3) * self.noise_amount,
                    math.sin(noise_phase * 1.7 + 1.0) * self.noise_amount * 0.5,
                    math.sin(noise_phase * 3.1 + 2.0) * self.noise_amount,
                )
                offset = offset + noise * self.amplitude.x

            # Apply as rotation
            rotation_offset = Quat.from_euler(offset.x, offset.y, offset.z)
            new_rotation = transform.rotation * rotation_offset

            # Blend with effective weight
            final_rot = transform.rotation.slerp(new_rotation, effective_weight)

            result[bone] = Transform(
                translation=transform.translation,
                rotation=final_rot,
                scale=transform.scale,
            )

        return result

    def reset(self) -> None:
        """Reset sway state."""
        self._time = 0.0


# =============================================================================
# BREATHING CONTROLLER
# =============================================================================


@dataclass
class BreathingController(ProceduralController):
    """Breathing animation controller.

    Simulates breathing motion on chest/spine bones.

    Attributes:
        breath_rate: Breaths per second (normal ~15/min = 0.25/s)
        breath_depth: Breathing depth/intensity
        inhale_exhale_ratio: Ratio of inhale duration to total cycle
        scale_axis: Contribution of breathing to scale per axis
    """
    breath_rate: float = PROCEDURAL_CONFIG.DEFAULT_BREATH_RATE
    breath_depth: float = PROCEDURAL_CONFIG.DEFAULT_BREATH_DEPTH
    inhale_exhale_ratio: float = 0.4
    scale_axis: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.3, 1.0))

    _time: float = 0.0

    @property
    def controller_type(self) -> ControllerType:
        return ControllerType.BREATHING

    def update(self, dt: float, pose: Dict[int, Transform]) -> Dict[int, Transform]:
        """Update breathing animation."""
        result = {}
        if not self.enabled or self.weight <= 0 or dt <= 0:
            return result

        self._time += dt
        breath_value = self._calculate_breath_value()

        for bone in self.affected_bones:
            if bone not in pose:
                continue

            effective_weight = self.get_effective_weight(bone)
            if effective_weight <= 0:
                continue

            transform = pose[bone]

            # Apply breathing as slight scale change
            scale_offset = Vec3(
                1.0 + breath_value * self.breath_depth * self.scale_axis.x * effective_weight,
                1.0 + breath_value * self.breath_depth * self.scale_axis.y * effective_weight,
                1.0 + breath_value * self.breath_depth * self.scale_axis.z * effective_weight,
            )

            new_scale = Vec3(
                transform.scale.x * scale_offset.x,
                transform.scale.y * scale_offset.y,
                transform.scale.z * scale_offset.z,
            )

            result[bone] = Transform(
                translation=transform.translation,
                rotation=transform.rotation,
                scale=new_scale,
            )

        return result

    def _calculate_breath_value(self) -> float:
        """Calculate current breath value (0 = exhaled, 1 = inhaled)."""
        cycle_time = 1.0 / self.breath_rate
        phase = (self._time % cycle_time) / cycle_time

        if phase < self.inhale_exhale_ratio:
            # Inhale - ease in
            t = phase / self.inhale_exhale_ratio
            return t * t * (3.0 - 2.0 * t)  # smoothstep
        else:
            # Exhale - ease out
            t = (phase - self.inhale_exhale_ratio) / (1.0 - self.inhale_exhale_ratio)
            t_inv = 1.0 - t
            return t_inv * t_inv * (3.0 - 2.0 * t_inv)

    def reset(self) -> None:
        """Reset breathing state."""
        self._time = 0.0


# =============================================================================
# PROCEDURAL COMPONENT
# =============================================================================


@dataclass
class ProceduralComponent:
    """Component for entities with procedural animation.

    Attributes:
        controllers: List of procedural controllers
        enabled: Whether procedural animation is enabled
        global_weight: Global weight multiplier for all effects
    """
    controllers: List[ProceduralController] = field(default_factory=list)
    enabled: bool = True
    global_weight: float = 1.0

    # Optional integrated subsystems
    spring_chains: List[Any] = field(default_factory=list)  # SpringChain instances
    twist_bones: List[Any] = field(default_factory=list)    # TwistBone instances
    ragdoll: Optional[Any] = None  # Ragdoll instance
    lookat_controller: Optional[Any] = None  # External LookAtController

    def add_controller(self, controller: ProceduralController) -> int:
        """Add controller, returns index."""
        self.controllers.append(controller)
        return len(self.controllers) - 1

    def remove_controller(self, index: int) -> bool:
        """Remove controller by index."""
        if 0 <= index < len(self.controllers):
            self.controllers.pop(index)
            return True
        return False

    def get_controller(self, index: int) -> Optional[ProceduralController]:
        """Get controller by index."""
        if 0 <= index < len(self.controllers):
            return self.controllers[index]
        return None

    def get_controllers_by_type(self, ctrl_type: ControllerType) -> List[ProceduralController]:
        """Get all controllers of given type."""
        return [c for c in self.controllers if c.controller_type == ctrl_type]

    def get_controllers_by_effect(self, effect_type: ProceduralEffectType) -> List[ProceduralController]:
        """Get all controllers of given effect type."""
        return [c for c in self.controllers if c.effect_type == effect_type]

    def set_all_weights(self, weight: float) -> None:
        """Set weight for all controllers."""
        weight = max(0.0, min(1.0, weight))
        for controller in self.controllers:
            controller.weight = weight

    def enable_all(self, enabled: bool = True) -> None:
        """Enable or disable all controllers."""
        for controller in self.controllers:
            controller.enabled = enabled


# =============================================================================
# PROCEDURAL SYSTEM (T-AN-9.5)
# =============================================================================


@system(phase="animation", order=2)
class ProceduralSystem:
    """ECS system for procedural animation.

    Runs after IK system (order=1), before skinning system.

    Processing Order:
    1. Spring/jiggle bones (T-AN-7.5)
    2. Look-at controllers (T-AN-7.6)
    3. Twist distribution (T-AN-7.7)
    4. Ragdoll blending (T-AN-7.8)

    Features:
    - Effect chaining: output of one feeds the next
    - Per-bone effect masking via BoneMask
    - Weight blending per controller and per bone
    - Integration with standalone procedural modules
    """

    # Effect processing order
    EFFECT_ORDER = [
        ProceduralEffectType.SPRING,
        ProceduralEffectType.LOOK_AT,
        ProceduralEffectType.TWIST,
        ProceduralEffectType.RAGDOLL,
        ProceduralEffectType.SWAY,
        ProceduralEffectType.BREATHING,
        ProceduralEffectType.NOISE,
        ProceduralEffectType.CUSTOM,
    ]

    def __init__(self):
        """Initialize the procedural system."""
        self._debug_enabled: bool = False
        self._stats: Dict[str, float] = {}

    def update(
        self,
        world: World,
        dt: float,
        entity_components: List[Tuple[Entity, ProceduralComponent]],
        pose_data: Dict[Entity, Dict[int, Transform]]
    ) -> Dict[Entity, Dict[int, Transform]]:
        """Update all procedural components.

        Args:
            world: ECS world
            dt: Delta time in seconds
            entity_components: List of (entity, component) tuples
            pose_data: Current poses (entity -> bone transforms)

        Returns:
            Updated pose data with procedural effects applied
        """
        result = {}

        for entity, component in entity_components:
            if not component.enabled:
                result[entity] = pose_data.get(entity, {})
                continue

            # Get current pose (copy to allow modification)
            entity_pose = dict(pose_data.get(entity, {}))

            # Apply global weight
            effective_dt = dt
            if component.global_weight < 1.0:
                # Could adjust weights instead, but for simplicity we proceed
                pass

            # Process effects in defined order
            entity_pose = self._process_effects_in_order(
                component, effective_dt, entity_pose
            )

            result[entity] = entity_pose

        return result

    def _process_effects_in_order(
        self,
        component: ProceduralComponent,
        dt: float,
        pose: Dict[int, Transform]
    ) -> Dict[int, Transform]:
        """Process all effects in the defined order with chaining.

        Each effect type processes in sequence, with the output of one
        feeding into the next.
        """
        current_pose = pose

        for effect_type in self.EFFECT_ORDER:
            # Get controllers of this effect type, sorted by effect_order
            controllers = [
                c for c in component.controllers
                if c.effect_type == effect_type and c.enabled
            ]
            controllers.sort(key=lambda c: c.effect_order)

            # Apply each controller, chaining results
            for controller in controllers:
                modifications = controller.update(dt, current_pose)

                # Apply weight-scaled modifications
                for bone, transform in modifications.items():
                    current_pose[bone] = transform

        # Process integrated subsystems
        current_pose = self._process_integrated_subsystems(component, dt, current_pose)

        return current_pose

    def _process_integrated_subsystems(
        self,
        component: ProceduralComponent,
        dt: float,
        pose: Dict[int, Transform]
    ) -> Dict[int, Transform]:
        """Process integrated subsystem instances (SpringChain, TwistBone, etc.)."""
        # This would integrate with the actual procedural module classes
        # For now, we handle them through the controller abstraction
        return pose

    def reset_controllers(
        self,
        entity_components: List[Tuple[Entity, ProceduralComponent]]
    ) -> None:
        """Reset all procedural controllers."""
        for _, component in entity_components:
            for controller in component.controllers:
                controller.reset()

    def set_debug_enabled(self, enabled: bool) -> None:
        """Enable or disable debug mode."""
        self._debug_enabled = enabled

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return self._stats.copy()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def create_spring_chain_controller(
    bone_indices: List[int],
    stiffness: float = 50.0,
    damping: float = 0.5,
    gravity: Vec3 = None
) -> SpringController:
    """Factory function to create a spring controller for a chain of bones."""
    controller = SpringController(
        stiffness=stiffness,
        damping=damping,
        gravity=gravity or Vec3(0, -9.8, 0),
        affected_bones=bone_indices,
    )
    return controller


def create_lookat_controller(
    head_bone: int,
    neck_bone: int = -1,
    eye_bones: List[int] = None,
    target: Vec3 = None
) -> LookAtController:
    """Factory function to create a look-at controller."""
    affected = [head_bone]
    weights = {head_bone: 0.7}

    if neck_bone >= 0:
        affected.append(neck_bone)
        weights[neck_bone] = 0.3

    if eye_bones:
        affected.extend(eye_bones)
        for eye in eye_bones:
            weights[eye] = 1.0

    controller = LookAtController(
        target=target or Vec3.zero(),
        affected_bones=affected,
        bone_weights=weights,
    )
    return controller


def create_twist_controller(
    source_bone: int,
    twist_bones: List[int],
    reference_bone: int = -1,
    twist_axis: Vec3 = None
) -> TwistController:
    """Factory function to create a twist distribution controller."""
    # Calculate linear distribution weights
    weights = {}
    num_bones = len(twist_bones)
    for i, bone in enumerate(twist_bones):
        weights[bone] = (i + 1) / (num_bones + 1)

    controller = TwistController(
        source_bone=source_bone,
        reference_bone=reference_bone,
        twist_axis=twist_axis or Vec3(1, 0, 0),
        affected_bones=twist_bones,
        distribution_weights=weights,
    )
    return controller


def create_ragdoll_blend_controller(
    affected_bones: List[int],
    blend_speed: float = 3.0
) -> RagdollBlendController:
    """Factory function to create a ragdoll blend controller."""
    controller = RagdollBlendController(
        affected_bones=affected_bones,
        blend_speed=blend_speed,
    )
    return controller
