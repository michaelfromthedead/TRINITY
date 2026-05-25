"""
Ragdoll Physics Integration.

Provides ragdoll physics for characters, supporting:
- Full ragdoll (physics-driven)
- Partial ragdoll (animation-physics blend)
- Kinematic mode (animation drives physics)
- Active ragdoll (motor-assisted physics)

Usage:
    config = RagdollConfig.create_humanoid(skeleton)
    ragdoll = Ragdoll(skeleton, config, physics_world)
    ragdoll.activate(blend_time=0.3)
    pose = ragdoll.sync_from_physics(physics_world)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Protocol, Any, Set

# Type aliases
Vec3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (x, y, z, w)
Matrix4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


class PhysicsWorld(Protocol):
    """Protocol for physics world interface."""

    def create_rigid_body(
        self,
        shape: "CollisionShape",
        transform: "Transform",
        mass: float,
        is_kinematic: bool,
    ) -> int:
        """Create a rigid body and return its ID."""
        ...

    def destroy_rigid_body(self, body_id: int) -> None:
        """Destroy a rigid body."""
        ...

    def get_body_transform(self, body_id: int) -> "Transform":
        """Get transform of a rigid body."""
        ...

    def set_body_transform(self, body_id: int, transform: "Transform") -> None:
        """Set transform of a rigid body."""
        ...

    def set_body_kinematic(self, body_id: int, is_kinematic: bool) -> None:
        """Set whether a body is kinematic."""
        ...

    def create_joint(
        self,
        body_a: int,
        body_b: int,
        joint_type: str,
        config: Dict[str, Any],
    ) -> int:
        """Create a joint between two bodies and return its ID."""
        ...

    def destroy_joint(self, joint_id: int) -> None:
        """Destroy a joint."""
        ...

    def set_joint_motor(
        self,
        joint_id: int,
        target_rotation: Quaternion,
        max_torque: float,
    ) -> None:
        """Set joint motor target."""
        ...

    def apply_impulse(self, body_id: int, impulse: Vec3, position: Vec3) -> None:
        """Apply an impulse to a body."""
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
        """Get parent bone index, -1 for root."""
        ...

    def get_bone_bind_pose(self, bone_index: int) -> "Transform":
        """Get bone bind pose transform."""
        ...


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


@dataclass
class Transform:
    """3D transform with position and rotation."""

    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    scale: Vec3 = (1.0, 1.0, 1.0)


class CollisionShapeType(Enum):
    """Types of collision shapes for ragdoll bodies."""

    SPHERE = auto()
    CAPSULE = auto()
    BOX = auto()


@dataclass
class CollisionShape:
    """Collision shape for a ragdoll body."""

    shape_type: CollisionShapeType
    radius: float = 0.1  # For sphere and capsule
    height: float = 0.5  # For capsule
    half_extents: Vec3 = (0.1, 0.1, 0.1)  # For box
    offset: Vec3 = (0.0, 0.0, 0.0)  # Offset from bone

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("radius must be > 0")
        if self.height < 0:
            raise ValueError("height must be >= 0")


class CollisionGroup(Enum):
    """Collision groups for filtering ragdoll collisions."""

    DEFAULT = 0
    RAGDOLL = 1
    ENVIRONMENT = 2
    CHARACTER = 3
    PROJECTILE = 4


@dataclass
class JointLimits:
    """Angular limits for a ragdoll joint."""

    # Twist limits (rotation around joint axis)
    twist_lower: float = math.radians(-45.0)
    twist_upper: float = math.radians(45.0)

    # Swing limits (cone of allowed directions)
    swing1_limit: float = math.radians(45.0)  # Y-axis swing
    swing2_limit: float = math.radians(45.0)  # Z-axis swing

    # Contact distance (degrees before limit is hit)
    contact_distance: float = math.radians(5.0)

    def __post_init__(self):
        if self.twist_lower > self.twist_upper:
            raise ValueError("twist_lower must be <= twist_upper")
        if self.swing1_limit < 0 or self.swing2_limit < 0:
            raise ValueError("swing limits must be >= 0")


@dataclass
class JointMotor:
    """Motor configuration for active ragdoll."""

    enabled: bool = False
    max_torque: float = 100.0
    spring_stiffness: float = 1000.0
    spring_damping: float = 100.0
    target_rotation: Quaternion = field(default=(0.0, 0.0, 0.0, 1.0))

    def __post_init__(self):
        if self.max_torque < 0:
            raise ValueError("max_torque must be >= 0")
        if self.spring_stiffness < 0:
            raise ValueError("spring_stiffness must be >= 0")
        if self.spring_damping < 0:
            raise ValueError("spring_damping must be >= 0")


@dataclass
class RagdollBody:
    """Configuration for a single ragdoll body."""

    bone_index: int
    shape: CollisionShape
    mass: float = 1.0
    collision_group: CollisionGroup = CollisionGroup.RAGDOLL
    collision_mask: Set[CollisionGroup] = field(
        default_factory=lambda: {CollisionGroup.ENVIRONMENT, CollisionGroup.RAGDOLL}
    )

    # Runtime state
    physics_body_id: int = -1

    def __post_init__(self):
        if self.bone_index < 0:
            raise ValueError("bone_index must be >= 0")
        if self.mass <= 0:
            raise ValueError("mass must be > 0")

    def is_active(self) -> bool:
        """Check if physics body is created."""
        return self.physics_body_id >= 0


@dataclass
class RagdollJoint:
    """Configuration for a ragdoll joint between two bodies."""

    parent_body: int  # Index in bodies list
    child_body: int  # Index in bodies list
    limits: JointLimits = field(default_factory=JointLimits)
    motor: JointMotor = field(default_factory=JointMotor)

    # Local transforms for joint attachment
    parent_local_transform: Transform = field(default_factory=Transform)
    child_local_transform: Transform = field(default_factory=Transform)

    # Runtime state
    physics_joint_id: int = -1

    def __post_init__(self):
        if self.parent_body < 0:
            raise ValueError("parent_body must be >= 0")
        if self.child_body < 0:
            raise ValueError("child_body must be >= 0")
        if self.parent_body == self.child_body:
            raise ValueError("parent_body and child_body must be different")

    def is_active(self) -> bool:
        """Check if physics joint is created."""
        return self.physics_joint_id >= 0


class RagdollState(Enum):
    """Ragdoll operation state."""

    INACTIVE = auto()  # Not simulating
    KINEMATIC = auto()  # Animation drives physics
    DYNAMIC = auto()  # Physics drives animation
    BLENDING = auto()  # Transitioning between states


@dataclass
class RagdollConfig:
    """Complete ragdoll configuration."""

    bodies: List[RagdollBody]
    joints: List[RagdollJoint]

    # Collision configuration
    self_collision: bool = False
    collision_groups: Dict[int, CollisionGroup] = field(default_factory=dict)

    # Physics parameters
    linear_damping: float = 0.1
    angular_damping: float = 0.1

    def __post_init__(self):
        if not self.bodies:
            raise ValueError("bodies must not be empty")

        # Validate joint references
        num_bodies = len(self.bodies)
        for joint in self.joints:
            if joint.parent_body >= num_bodies:
                raise ValueError(
                    f"Joint parent_body {joint.parent_body} out of range"
                )
            if joint.child_body >= num_bodies:
                raise ValueError(
                    f"Joint child_body {joint.child_body} out of range"
                )

    @classmethod
    def create_humanoid(
        cls,
        skeleton: Skeleton,
        bone_mapping: Optional[Dict[str, int]] = None,
    ) -> "RagdollConfig":
        """
        Create a humanoid ragdoll configuration.

        Args:
            skeleton: Skeleton to create ragdoll for
            bone_mapping: Optional bone name to index mapping

        Returns:
            Configured RagdollConfig for humanoid
        """
        # Default humanoid bone names
        humanoid_bones = {
            "hips": {"mass": 15.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.15, "height": 0.2},
            "spine": {"mass": 10.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.12, "height": 0.15},
            "chest": {"mass": 10.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.15, "height": 0.2},
            "head": {"mass": 5.0, "shape": CollisionShapeType.SPHERE, "radius": 0.12},
            "left_upper_arm": {"mass": 3.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.05, "height": 0.25},
            "left_lower_arm": {"mass": 2.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.04, "height": 0.22},
            "right_upper_arm": {"mass": 3.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.05, "height": 0.25},
            "right_lower_arm": {"mass": 2.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.04, "height": 0.22},
            "left_thigh": {"mass": 8.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.08, "height": 0.4},
            "left_calf": {"mass": 5.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.06, "height": 0.35},
            "right_thigh": {"mass": 8.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.08, "height": 0.4},
            "right_calf": {"mass": 5.0, "shape": CollisionShapeType.CAPSULE, "radius": 0.06, "height": 0.35},
        }

        bodies = []
        bone_to_body_idx = {}

        # Create bodies for each bone
        for bone_name, params in humanoid_bones.items():
            bone_idx = -1
            if bone_mapping and bone_name in bone_mapping:
                bone_idx = bone_mapping[bone_name]
            else:
                # Try to find bone by name in skeleton
                for i in range(skeleton.get_bone_count()):
                    if skeleton.get_bone_name(i).lower() == bone_name.lower():
                        bone_idx = i
                        break

            if bone_idx >= 0:
                shape = CollisionShape(
                    shape_type=params["shape"],
                    radius=params.get("radius", 0.1),
                    height=params.get("height", 0.0),
                )
                body = RagdollBody(
                    bone_index=bone_idx,
                    shape=shape,
                    mass=params["mass"],
                )
                bone_to_body_idx[bone_name] = len(bodies)
                bodies.append(body)

        # Create joints between bodies
        joints = []
        joint_connections = [
            ("hips", "spine", JointLimits(
                twist_lower=math.radians(-30), twist_upper=math.radians(30),
                swing1_limit=math.radians(30), swing2_limit=math.radians(20)
            )),
            ("spine", "chest", JointLimits(
                twist_lower=math.radians(-30), twist_upper=math.radians(30),
                swing1_limit=math.radians(30), swing2_limit=math.radians(20)
            )),
            ("chest", "head", JointLimits(
                twist_lower=math.radians(-60), twist_upper=math.radians(60),
                swing1_limit=math.radians(40), swing2_limit=math.radians(30)
            )),
            ("chest", "left_upper_arm", JointLimits(
                twist_lower=math.radians(-90), twist_upper=math.radians(90),
                swing1_limit=math.radians(90), swing2_limit=math.radians(90)
            )),
            ("left_upper_arm", "left_lower_arm", JointLimits(
                twist_lower=math.radians(-5), twist_upper=math.radians(140),
                swing1_limit=math.radians(5), swing2_limit=math.radians(5)
            )),
            ("chest", "right_upper_arm", JointLimits(
                twist_lower=math.radians(-90), twist_upper=math.radians(90),
                swing1_limit=math.radians(90), swing2_limit=math.radians(90)
            )),
            ("right_upper_arm", "right_lower_arm", JointLimits(
                twist_lower=math.radians(-5), twist_upper=math.radians(140),
                swing1_limit=math.radians(5), swing2_limit=math.radians(5)
            )),
            ("hips", "left_thigh", JointLimits(
                twist_lower=math.radians(-45), twist_upper=math.radians(45),
                swing1_limit=math.radians(90), swing2_limit=math.radians(30)
            )),
            ("left_thigh", "left_calf", JointLimits(
                twist_lower=math.radians(-5), twist_upper=math.radians(150),
                swing1_limit=math.radians(5), swing2_limit=math.radians(5)
            )),
            ("hips", "right_thigh", JointLimits(
                twist_lower=math.radians(-45), twist_upper=math.radians(45),
                swing1_limit=math.radians(90), swing2_limit=math.radians(30)
            )),
            ("right_thigh", "right_calf", JointLimits(
                twist_lower=math.radians(-5), twist_upper=math.radians(150),
                swing1_limit=math.radians(5), swing2_limit=math.radians(5)
            )),
        ]

        for parent_name, child_name, limits in joint_connections:
            if parent_name in bone_to_body_idx and child_name in bone_to_body_idx:
                joint = RagdollJoint(
                    parent_body=bone_to_body_idx[parent_name],
                    child_body=bone_to_body_idx[child_name],
                    limits=limits,
                )
                joints.append(joint)

        return cls(bodies=bodies, joints=joints)


@dataclass
class Ragdoll:
    """
    Ragdoll physics controller.

    Manages the creation, synchronization, and blending of ragdoll physics.
    """

    skeleton: Skeleton
    config: RagdollConfig
    physics_world: Optional[PhysicsWorld] = None

    # State
    state: RagdollState = RagdollState.INACTIVE
    blend_weight: float = 0.0  # 0 = animation, 1 = physics
    blend_time: float = 0.0
    blend_duration: float = 0.3

    # Partial ragdoll: set of body indices that are dynamic
    active_bodies: Set[int] = field(default_factory=set)

    # Cached animation pose for blending
    _animation_pose: Optional[Pose] = field(default=None, repr=False)
    _physics_pose: Optional[Pose] = field(default=None, repr=False)
    _created: bool = field(default=False, repr=False)

    def create(self, physics_world: PhysicsWorld, pose: Pose) -> None:
        """
        Create physics bodies and joints.

        Args:
            physics_world: Physics world to create bodies in
            pose: Initial pose for body positions
        """
        if self._created:
            return

        self.physics_world = physics_world

        # Create bodies
        for i, body in enumerate(self.config.bodies):
            transform = Transform(
                position=pose.get_bone_position(body.bone_index),
                rotation=pose.get_bone_rotation(body.bone_index),
            )

            body.physics_body_id = physics_world.create_rigid_body(
                shape=body.shape,
                transform=transform,
                mass=body.mass,
                is_kinematic=True,  # Start kinematic
            )

        # Create joints
        for joint in self.config.joints:
            parent_body = self.config.bodies[joint.parent_body]
            child_body = self.config.bodies[joint.child_body]

            joint_config = {
                "limits": {
                    "twist_lower": joint.limits.twist_lower,
                    "twist_upper": joint.limits.twist_upper,
                    "swing1_limit": joint.limits.swing1_limit,
                    "swing2_limit": joint.limits.swing2_limit,
                },
                "motor": {
                    "enabled": joint.motor.enabled,
                    "max_torque": joint.motor.max_torque,
                    "spring_stiffness": joint.motor.spring_stiffness,
                    "spring_damping": joint.motor.spring_damping,
                },
            }

            joint.physics_joint_id = physics_world.create_joint(
                body_a=parent_body.physics_body_id,
                body_b=child_body.physics_body_id,
                joint_type="d6",
                config=joint_config,
            )

        self._created = True
        self.state = RagdollState.KINEMATIC

    def destroy(self) -> None:
        """Destroy all physics bodies and joints."""
        if not self._created or self.physics_world is None:
            return

        # Destroy joints first
        for joint in self.config.joints:
            if joint.is_active():
                self.physics_world.destroy_joint(joint.physics_joint_id)
                joint.physics_joint_id = -1

        # Destroy bodies
        for body in self.config.bodies:
            if body.is_active():
                self.physics_world.destroy_rigid_body(body.physics_body_id)
                body.physics_body_id = -1

        self._created = False
        self.state = RagdollState.INACTIVE

    def activate(
        self,
        blend_time: float = 0.3,
        partial_bodies: Optional[Set[int]] = None,
    ) -> None:
        """
        Activate ragdoll (switch to dynamic physics).

        Args:
            blend_time: Time to blend from animation to physics
            partial_bodies: Optional set of body indices to activate (None = all)
        """
        if not self._created or self.physics_world is None:
            return

        self.blend_duration = max(0.0, blend_time)
        self.blend_time = 0.0
        self.state = RagdollState.BLENDING if blend_time > 0 else RagdollState.DYNAMIC

        # Determine which bodies to activate
        if partial_bodies is not None:
            self.active_bodies = partial_bodies.copy()
        else:
            self.active_bodies = set(range(len(self.config.bodies)))

        # Set active bodies to dynamic
        for i in self.active_bodies:
            body = self.config.bodies[i]
            if body.is_active():
                self.physics_world.set_body_kinematic(body.physics_body_id, False)

    def deactivate(self, blend_time: float = 0.3) -> None:
        """
        Deactivate ragdoll (switch back to animation).

        Args:
            blend_time: Time to blend from physics to animation
        """
        if not self._created or self.physics_world is None:
            return

        self.blend_duration = max(0.0, blend_time)
        self.blend_time = 0.0
        self.state = RagdollState.BLENDING

        # Reset all bodies to kinematic
        for body in self.config.bodies:
            if body.is_active():
                self.physics_world.set_body_kinematic(body.physics_body_id, True)

        self.active_bodies.clear()

    def sync_to_physics(self, pose: Pose) -> None:
        """
        Sync animation pose to physics bodies (kinematic mode).

        Args:
            pose: Animation pose to sync from
        """
        if not self._created or self.physics_world is None:
            return

        self._animation_pose = pose.copy()

        # Only sync non-active (kinematic) bodies
        for i, body in enumerate(self.config.bodies):
            if body.is_active() and i not in self.active_bodies:
                transform = Transform(
                    position=pose.get_bone_position(body.bone_index),
                    rotation=pose.get_bone_rotation(body.bone_index),
                )
                self.physics_world.set_body_transform(body.physics_body_id, transform)

    def sync_from_physics(self, dt: float = 0.0) -> Optional[Pose]:
        """
        Get pose from physics bodies (dynamic mode).

        Args:
            dt: Time step for blend update

        Returns:
            Pose driven by physics, or None if not active
        """
        if not self._created or self.physics_world is None:
            return None

        if self._animation_pose is None:
            return None

        result = self._animation_pose.copy()

        # Update blend state
        if self.state == RagdollState.BLENDING and dt > 0:
            self.blend_time += dt
            if self.blend_duration > 0:
                self.blend_weight = min(1.0, self.blend_time / self.blend_duration)
            else:
                self.blend_weight = 1.0

            if self.blend_weight >= 1.0:
                self.state = RagdollState.DYNAMIC if self.active_bodies else RagdollState.KINEMATIC

        # Get physics transforms for active bodies
        for i in self.active_bodies:
            body = self.config.bodies[i]
            if body.is_active():
                transform = self.physics_world.get_body_transform(body.physics_body_id)

                if self.state == RagdollState.BLENDING:
                    # Blend between animation and physics
                    anim_pos = self._animation_pose.get_bone_position(body.bone_index)
                    anim_rot = self._animation_pose.get_bone_rotation(body.bone_index)

                    blended_pos = self._lerp_vec3(anim_pos, transform.position, self.blend_weight)
                    blended_rot = self._slerp_quat(anim_rot, transform.rotation, self.blend_weight)

                    result.set_bone_position(body.bone_index, blended_pos)
                    result.set_bone_rotation(body.bone_index, blended_rot)
                else:
                    # Full physics
                    result.set_bone_position(body.bone_index, transform.position)
                    result.set_bone_rotation(body.bone_index, transform.rotation)

        self._physics_pose = result
        return result

    def apply_impulse(
        self,
        body_index: int,
        impulse: Vec3,
        world_position: Optional[Vec3] = None,
    ) -> None:
        """
        Apply an impulse to a ragdoll body.

        Args:
            body_index: Index of body to apply impulse to
            impulse: Impulse vector
            world_position: World position to apply impulse at (None = center of mass)
        """
        if not self._created or self.physics_world is None:
            return

        if body_index < 0 or body_index >= len(self.config.bodies):
            return

        body = self.config.bodies[body_index]
        if not body.is_active():
            return

        if world_position is None:
            # Apply at center of mass
            transform = self.physics_world.get_body_transform(body.physics_body_id)
            world_position = transform.position

        self.physics_world.apply_impulse(body.physics_body_id, impulse, world_position)

    def set_motor_targets(self, pose: Pose) -> None:
        """
        Set joint motor targets from animation pose (active ragdoll).

        Args:
            pose: Target pose for motors
        """
        if not self._created or self.physics_world is None:
            return

        for joint in self.config.joints:
            if not joint.motor.enabled or not joint.is_active():
                continue

            child_body = self.config.bodies[joint.child_body]
            target_rotation = pose.get_bone_rotation(child_body.bone_index)

            self.physics_world.set_joint_motor(
                joint.physics_joint_id,
                target_rotation,
                joint.motor.max_torque,
            )

    def enable_active_ragdoll(self, enabled: bool = True) -> None:
        """
        Enable/disable active ragdoll (motor-assisted physics).

        Args:
            enabled: Whether to enable joint motors
        """
        for joint in self.config.joints:
            joint.motor.enabled = enabled

    def get_state(self) -> RagdollState:
        """Get current ragdoll state."""
        return self.state

    def get_blend_weight(self) -> float:
        """Get current blend weight (0 = animation, 1 = physics)."""
        return self.blend_weight

    def is_active(self) -> bool:
        """Check if ragdoll is simulating."""
        return self.state in (RagdollState.DYNAMIC, RagdollState.BLENDING)

    def _lerp_vec3(self, a: Vec3, b: Vec3, t: float) -> Vec3:
        """Linear interpolation between vectors."""
        return (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        )

    def _slerp_quat(self, a: Quaternion, b: Quaternion, t: float) -> Quaternion:
        """Spherical interpolation between quaternions."""
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


def create_from_skeleton(
    skeleton: Skeleton,
    physics_world: PhysicsWorld,
    pose: Pose,
    bone_mapping: Optional[Dict[str, int]] = None,
) -> Ragdoll:
    """
    Factory function to create a ragdoll from a skeleton.

    Args:
        skeleton: Skeleton to create ragdoll for
        physics_world: Physics world to create bodies in
        pose: Initial pose
        bone_mapping: Optional bone name to index mapping

    Returns:
        Configured and created Ragdoll
    """
    config = RagdollConfig.create_humanoid(skeleton, bone_mapping)
    ragdoll = Ragdoll(skeleton=skeleton, config=config)
    ragdoll.create(physics_world, pose)
    return ragdoll
