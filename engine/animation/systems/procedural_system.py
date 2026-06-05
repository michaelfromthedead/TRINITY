"""ECS system for procedural animation.

Applies secondary motion effects like springs, look-at, sway, and breathing.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Any, Sequence

from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import PROCEDURAL_CONFIG


# Forward reference for Pose type
class Pose:
    """Placeholder for pose type used in ProceduralModifier."""
    pass


class ProceduralModifier(ABC):
    """Base class for procedural animation modifiers."""

    @abstractmethod
    def apply(self, pose: 'Pose', dt: float) -> 'Pose':
        """Apply modifier to pose, return modified pose."""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Execution priority (lower = earlier)."""
        pass


@dataclass
class BreathingModifier(ProceduralModifier):
    """Adds breathing motion to spine/chest bones."""
    spine_bones: list[int] = field(default_factory=list)
    amplitude: float = 0.02
    frequency: float = 0.25
    _phase: float = 0.0

    def apply(self, pose: 'Pose', dt: float) -> 'Pose':
        self._phase += dt * self.frequency * 2 * math.pi
        # Apply sinusoidal offset to spine bones
        return pose

    @property
    def priority(self) -> int:
        return 100


@dataclass
class SpringBoneModifier(ProceduralModifier):
    """Physics-based secondary motion for hair, cloth, accessories."""
    bone_indices: list[int] = field(default_factory=list)
    stiffness: float = 100.0
    damping: float = 5.0
    gravity: float = -9.8

    def apply(self, pose: 'Pose', dt: float) -> 'Pose':
        # Spring physics simulation would go here
        return pose

    @property
    def priority(self) -> int:
        return 200


@dataclass
class LookAtModifier(ProceduralModifier):
    """Makes bones orient toward a target point."""
    head_bone: int = 0
    target_position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    speed: float = 5.0
    angle_limit: float = 1.5708  # 90 degrees in radians

    def apply(self, pose: 'Pose', dt: float) -> 'Pose':
        # Look-at calculation would go here
        return pose

    @property
    def priority(self) -> int:
        return 50


class ControllerType(Enum):
    """Type of procedural controller."""
    SPRING = auto()
    LOOK_AT = auto()
    SWAY = auto()
    BREATHING = auto()
    NOISE = auto()
    CUSTOM = auto()


@dataclass
class ProceduralController(ABC):
    """Base class for procedural animation controllers."""
    enabled: bool = True
    weight: float = 1.0
    affected_bones: list[int] = field(default_factory=list)

    @property
    @abstractmethod
    def controller_type(self) -> ControllerType:
        """Controller type identifier."""
        pass

    @abstractmethod
    def update(self, dt: float, pose: dict[int, Transform]) -> dict[int, Transform]:
        """Update controller and return modified transforms.

        Args:
            dt: Delta time
            pose: Current bone transforms

        Returns:
            Modified transforms for affected bones
        """
        pass

    def reset(self) -> None:
        """Reset controller state."""
        pass


@dataclass
class SpringController(ProceduralController):
    """Spring-based secondary motion (e.g., hair, cloth, accessories).

    Simulates spring dynamics for natural secondary motion.
    """
    stiffness: float = PROCEDURAL_CONFIG.DEFAULT_SPRING_STIFFNESS  # Spring stiffness
    damping: float = PROCEDURAL_CONFIG.DEFAULT_SPRING_DAMPING  # Damping factor (0-1)
    mass: float = PROCEDURAL_CONFIG.DEFAULT_SPRING_MASS  # Mass of connected object
    gravity: Vec3 = field(default_factory=lambda: Vec3(0, -9.8, 0))
    max_stretch: float = PROCEDURAL_CONFIG.DEFAULT_MAX_STRETCH  # Maximum stretch factor

    # Internal state
    _velocities: dict[int, Vec3] = field(default_factory=dict)
    _rest_positions: dict[int, Vec3] = field(default_factory=dict)
    _current_positions: dict[int, Vec3] = field(default_factory=dict)

    @property
    def controller_type(self) -> ControllerType:
        return ControllerType.SPRING

    def update(self, dt: float, pose: dict[int, Transform]) -> dict[int, Transform]:
        result = {}
        if not self.enabled or self.weight <= 0:
            return result

        for bone in self.affected_bones:
            if bone not in pose:
                continue

            transform = pose[bone]

            # Initialize rest position if needed
            if bone not in self._rest_positions:
                self._rest_positions[bone] = transform.translation
                self._current_positions[bone] = transform.translation
                self._velocities[bone] = Vec3.zero()

            rest_pos = self._rest_positions[bone]
            current_pos = self._current_positions[bone]
            velocity = self._velocities[bone]

            # Update rest position to follow animation
            self._rest_positions[bone] = transform.translation

            # Spring force
            displacement = rest_pos - current_pos
            spring_force = displacement * self.stiffness

            # Damping force
            damping_force = velocity * (-self.damping * self.stiffness)

            # Gravity
            gravity_force = self.gravity * self.mass

            # Total acceleration
            total_force = spring_force + damping_force + gravity_force
            acceleration = total_force / self.mass

            # Integrate
            velocity = velocity + acceleration * dt
            new_pos = current_pos + velocity * dt

            # Limit stretch
            offset = new_pos - rest_pos
            max_dist = self.max_stretch
            if offset.length() > max_dist:
                new_pos = rest_pos + offset.normalized() * max_dist
                # Dampen velocity when at limit
                velocity = velocity * 0.5

            self._current_positions[bone] = new_pos
            self._velocities[bone] = velocity

            # Blend with original
            final_pos = transform.translation.lerp(new_pos, self.weight)

            result[bone] = Transform(
                translation=final_pos,
                rotation=transform.rotation,
                scale=transform.scale,
            )

        return result

    def reset(self) -> None:
        self._velocities.clear()
        self._rest_positions.clear()
        self._current_positions.clear()


@dataclass
class LookAtController(ProceduralController):
    """Look-at constraint controller.

    Makes bones orient toward a target point.
    """
    target: Vec3 = field(default_factory=Vec3.zero)
    up_vector: Vec3 = field(default_factory=Vec3.up)
    speed: float = PROCEDURAL_CONFIG.DEFAULT_LOOK_SPEED  # Rotation speed
    angle_limit_horizontal: float = PROCEDURAL_CONFIG.DEFAULT_HORIZONTAL_LIMIT  # 90 degrees
    angle_limit_vertical: float = PROCEDURAL_CONFIG.DEFAULT_VERTICAL_LIMIT  # 60 degrees
    forward_axis: Vec3 = field(default_factory=Vec3.forward)

    # Per-bone weights
    bone_weights: dict[int, float] = field(default_factory=dict)

    # Internal state
    _current_rotations: dict[int, Quat] = field(default_factory=dict)

    @property
    def controller_type(self) -> ControllerType:
        return ControllerType.LOOK_AT

    def update(self, dt: float, pose: dict[int, Transform]) -> dict[int, Transform]:
        result = {}
        if not self.enabled or self.weight <= 0:
            return result

        for bone in self.affected_bones:
            if bone not in pose:
                continue

            transform = pose[bone]
            bone_weight = self.bone_weights.get(bone, 1.0)

            if bone_weight <= 0:
                continue

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

            current_rot = self._current_rotations[bone]
            new_rot = current_rot.slerp(target_rotation, min(1.0, self.speed * dt))
            self._current_rotations[bone] = new_rot

            # Blend with original
            effective_weight = self.weight * bone_weight
            final_rot = transform.rotation.slerp(new_rot, effective_weight)

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
        # This is a simplified version
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
        # Get relative rotation
        relative = reference.inverse() * target
        pitch, yaw, roll = relative.to_euler()

        # Clamp angles
        yaw = max(-self.angle_limit_horizontal, min(self.angle_limit_horizontal, yaw))
        pitch = max(-self.angle_limit_vertical, min(self.angle_limit_vertical, pitch))

        # Reconstruct
        limited = Quat.from_euler(pitch, yaw, roll)
        return reference * limited

    def reset(self) -> None:
        self._current_rotations.clear()


@dataclass
class SwayController(ProceduralController):
    """Sway/wave motion controller.

    Creates oscillating motion for vegetation, flags, etc.
    """
    frequency: float = PROCEDURAL_CONFIG.DEFAULT_SWAY_FREQUENCY  # Oscillation frequency
    amplitude: Vec3 = field(default_factory=lambda: Vec3(0.1, 0.05, 0.1))
    phase_offset: float = 0.0  # Phase offset in radians
    noise_amount: float = PROCEDURAL_CONFIG.DEFAULT_NOISE_AMOUNT  # Random variation

    # Per-bone phase offsets (for cascading motion)
    bone_phase_offsets: dict[int, float] = field(default_factory=dict)

    _time: float = 0.0

    @property
    def controller_type(self) -> ControllerType:
        return ControllerType.SWAY

    def update(self, dt: float, pose: dict[int, Transform]) -> dict[int, Transform]:
        result = {}
        if not self.enabled or self.weight <= 0:
            return result

        self._time += dt

        for bone in self.affected_bones:
            if bone not in pose:
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

            # Blend
            final_rot = transform.rotation.slerp(new_rotation, self.weight)

            result[bone] = Transform(
                translation=transform.translation,
                rotation=final_rot,
                scale=transform.scale,
            )

        return result

    def reset(self) -> None:
        self._time = 0.0


@dataclass
class BreathingController(ProceduralController):
    """Breathing animation controller.

    Simulates breathing motion on chest/spine bones.
    """
    breath_rate: float = PROCEDURAL_CONFIG.DEFAULT_BREATH_RATE  # Breaths per second (normal ~15/min = 0.25/s)
    breath_depth: float = PROCEDURAL_CONFIG.DEFAULT_BREATH_DEPTH  # Breathing depth
    inhale_exhale_ratio: float = 0.4  # Inhale takes 40% of breath cycle
    scale_axis: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.3, 1.0))  # Scale contribution

    _time: float = 0.0

    @property
    def controller_type(self) -> ControllerType:
        return ControllerType.BREATHING

    def update(self, dt: float, pose: dict[int, Transform]) -> dict[int, Transform]:
        result = {}
        if not self.enabled or self.weight <= 0:
            return result

        self._time += dt
        breath_value = self._calculate_breath_value()

        for bone in self.affected_bones:
            if bone not in pose:
                continue

            transform = pose[bone]

            # Apply breathing as slight scale change
            scale_offset = Vec3(
                1.0 + breath_value * self.breath_depth * self.scale_axis.x,
                1.0 + breath_value * self.breath_depth * self.scale_axis.y,
                1.0 + breath_value * self.breath_depth * self.scale_axis.z,
            )

            new_scale = Vec3(
                transform.scale.x * scale_offset.x,
                transform.scale.y * scale_offset.y,
                transform.scale.z * scale_offset.z,
            )

            # Blend
            final_scale = transform.scale.lerp(new_scale, self.weight)

            result[bone] = Transform(
                translation=transform.translation,
                rotation=transform.rotation,
                scale=final_scale,
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
        self._time = 0.0


@dataclass
class ProceduralComponent:
    """Component for entities with procedural animation.

    Attributes:
        controllers: List of procedural controllers
        enabled: Whether procedural animation is enabled
    """
    controllers: list[ProceduralController] = field(default_factory=list)
    enabled: bool = True

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

    def get_controller(self, index: int) -> ProceduralController | None:
        """Get controller by index."""
        if 0 <= index < len(self.controllers):
            return self.controllers[index]
        return None

    def get_controllers_by_type(self, ctrl_type: ControllerType) -> list[ProceduralController]:
        """Get all controllers of given type."""
        return [c for c in self.controllers if c.controller_type == ctrl_type]


class ProceduralSystem:
    """ECS system for procedural animation.

    Runs after IK system, before skinning.
    """

    def __init__(self):
        pass

    def update(
        self,
        world: World,
        dt: float,
        entity_components: list[tuple[Entity, ProceduralComponent]],
        pose_data: dict[Entity, dict[int, Transform]]
    ) -> dict[Entity, dict[int, Transform]]:
        """Update all procedural components.

        Args:
            world: ECS world
            dt: Delta time
            entity_components: List of (entity, component) tuples
            pose_data: Current poses

        Returns:
            Updated pose data with procedural effects applied
        """
        result = {}

        for entity, component in entity_components:
            if not component.enabled:
                result[entity] = pose_data.get(entity, {})
                continue

            entity_pose = dict(pose_data.get(entity, {}))

            for controller in component.controllers:
                if not controller.enabled:
                    continue

                modifications = controller.update(dt, entity_pose)

                # Merge modifications
                for bone, transform in modifications.items():
                    entity_pose[bone] = transform

            result[entity] = entity_pose

        return result

    def reset_controllers(
        self,
        entity_components: list[tuple[Entity, ProceduralComponent]]
    ) -> None:
        """Reset all procedural controllers."""
        for _, component in entity_components:
            for controller in component.controllers:
                controller.reset()


def system(func: Callable) -> Callable:
    """Decorator to mark a function as an ECS system."""
    func._is_system = True
    return func
