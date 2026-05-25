"""
Ragdoll Physics System.

Provides ragdoll creation, activation, and pose recovery for physics-driven
character animations during impacts, falls, and deaths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .character_controller import Quaternion, Transform, Vector3
from .config import (
    LAYER_RAGDOLL,
    MASK_RAGDOLL,
    RAGDOLL_BLEND_TIME_MS,
    RAGDOLL_MIN_VELOCITY,
    RAGDOLL_RECOVERY_TIME_MS,
    RAGDOLL_SETTLED_TIME_MS,
)


# =============================================================================
# Ragdoll Data Structures
# =============================================================================

class RagdollState(str, Enum):
    """State of the ragdoll system."""
    INACTIVE = "inactive"
    ACTIVATING = "activating"
    ACTIVE = "active"
    SETTLING = "settling"
    RECOVERING = "recovering"


class BodyPartType(str, Enum):
    """Types of ragdoll body parts."""
    PELVIS = "pelvis"
    SPINE_LOWER = "spine_lower"
    SPINE_UPPER = "spine_upper"
    CHEST = "chest"
    NECK = "neck"
    HEAD = "head"
    SHOULDER_L = "shoulder_l"
    UPPER_ARM_L = "upper_arm_l"
    LOWER_ARM_L = "lower_arm_l"
    HAND_L = "hand_l"
    SHOULDER_R = "shoulder_r"
    UPPER_ARM_R = "upper_arm_r"
    LOWER_ARM_R = "lower_arm_r"
    HAND_R = "hand_r"
    UPPER_LEG_L = "upper_leg_l"
    LOWER_LEG_L = "lower_leg_l"
    FOOT_L = "foot_l"
    UPPER_LEG_R = "upper_leg_r"
    LOWER_LEG_R = "lower_leg_r"
    FOOT_R = "foot_r"


@dataclass
class RagdollBodyDef:
    """
    Definition for a ragdoll body part.

    Attributes:
        part_type: Type of body part
        bone_name: Name of corresponding skeleton bone
        mass: Mass of the body
        shape_type: Collision shape (capsule, box, sphere)
        dimensions: Shape dimensions (radius, height for capsule)
        local_offset: Offset from bone transform
    """
    part_type: BodyPartType
    bone_name: str
    mass: float = 1.0
    shape_type: str = "capsule"
    dimensions: tuple[float, ...] = (0.05, 0.2)
    local_offset: Vector3 = field(default_factory=Vector3.zero)
    local_rotation: Quaternion = field(default_factory=Quaternion.identity)


@dataclass
class RagdollJointDef:
    """
    Definition for a ragdoll joint.

    Attributes:
        parent_part: Parent body part type
        child_part: Child body part type
        joint_type: Type of joint (cone, hinge, fixed)
        swing_limit: Swing limit in degrees
        twist_limit: Twist limit in degrees
        local_anchor_parent: Joint anchor in parent space
        local_anchor_child: Joint anchor in child space
    """
    parent_part: BodyPartType
    child_part: BodyPartType
    joint_type: str = "cone"
    swing_limit: float = 45.0
    twist_limit: float = 30.0
    local_anchor_parent: Vector3 = field(default_factory=Vector3.zero)
    local_anchor_child: Vector3 = field(default_factory=Vector3.zero)


@dataclass
class RagdollSetup:
    """
    Complete ragdoll configuration.

    Attributes:
        bodies_per_bone: Mapping of bone names to body definitions
        joints: Joint definitions between bodies
        total_mass: Total mass of ragdoll
        collision_group: Collision group for self-collision
        self_collision: Whether bodies collide with each other
    """
    bodies_per_bone: dict[str, RagdollBodyDef] = field(default_factory=dict)
    joints: list[RagdollJointDef] = field(default_factory=list)
    total_mass: float = 70.0
    collision_group: int = LAYER_RAGDOLL
    collision_mask: int = MASK_RAGDOLL
    self_collision: bool = False


@dataclass
class RagdollBodyState:
    """Runtime state of a ragdoll body."""
    body_id: int = 0
    position: Vector3 = field(default_factory=Vector3.zero)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    velocity: Vector3 = field(default_factory=Vector3.zero)
    angular_velocity: Vector3 = field(default_factory=Vector3.zero)
    is_kinematic: bool = False


@dataclass
class RagdollPose:
    """Complete ragdoll pose."""
    body_states: dict[BodyPartType, RagdollBodyState] = field(default_factory=dict)
    root_position: Vector3 = field(default_factory=Vector3.zero)
    root_rotation: Quaternion = field(default_factory=Quaternion.identity)


# =============================================================================
# Skeleton Interface
# =============================================================================

class SkeletonInterface:
    """Interface for skeleton access. Override in implementation."""

    def get_bone_names(self) -> list[str]:
        """Get all bone names in the skeleton."""
        return []

    def get_bone_transform(self, bone_name: str) -> Transform:
        """Get world transform of a bone."""
        return Transform()

    def get_bone_parent(self, bone_name: str) -> Optional[str]:
        """Get parent bone name."""
        return None

    def set_bone_transform(self, bone_name: str, transform: Transform) -> None:
        """Set bone transform."""
        pass


# =============================================================================
# Physics World Interface for Ragdoll
# =============================================================================

class RagdollPhysicsInterface:
    """Interface for ragdoll physics operations."""

    def create_body(
        self,
        position: Vector3,
        rotation: Quaternion,
        shape_type: str,
        dimensions: tuple[float, ...],
        mass: float,
        collision_group: int,
        collision_mask: int,
    ) -> int:
        """Create a physics body. Returns body ID."""
        return 0

    def destroy_body(self, body_id: int) -> None:
        """Destroy a physics body."""
        pass

    def create_joint(
        self,
        parent_body: int,
        child_body: int,
        joint_type: str,
        anchor_parent: Vector3,
        anchor_child: Vector3,
        limits: tuple[float, float],
    ) -> int:
        """Create a joint between bodies. Returns joint ID."""
        return 0

    def destroy_joint(self, joint_id: int) -> None:
        """Destroy a joint."""
        pass

    def get_body_transform(self, body_id: int) -> tuple[Vector3, Quaternion]:
        """Get body position and rotation."""
        return Vector3.zero(), Quaternion.identity()

    def get_body_velocity(self, body_id: int) -> tuple[Vector3, Vector3]:
        """Get body linear and angular velocity."""
        return Vector3.zero(), Vector3.zero()

    def set_body_transform(
        self,
        body_id: int,
        position: Vector3,
        rotation: Quaternion,
    ) -> None:
        """Set body transform."""
        pass

    def set_body_velocity(
        self,
        body_id: int,
        linear: Vector3,
        angular: Vector3,
    ) -> None:
        """Set body velocities."""
        pass

    def set_body_kinematic(self, body_id: int, kinematic: bool) -> None:
        """Set body kinematic state."""
        pass

    def apply_impulse(
        self,
        body_id: int,
        impulse: Vector3,
        point: Optional[Vector3] = None,
    ) -> None:
        """Apply impulse to body."""
        pass


# =============================================================================
# Default Humanoid Setup
# =============================================================================

def create_default_humanoid_setup() -> RagdollSetup:
    """Create a default humanoid ragdoll setup."""
    setup = RagdollSetup()

    # Define bodies
    body_defs = [
        RagdollBodyDef(BodyPartType.PELVIS, "Pelvis", mass=10.0, dimensions=(0.12, 0.15)),
        RagdollBodyDef(BodyPartType.SPINE_LOWER, "Spine", mass=8.0, dimensions=(0.1, 0.12)),
        RagdollBodyDef(BodyPartType.SPINE_UPPER, "Spine1", mass=8.0, dimensions=(0.1, 0.12)),
        RagdollBodyDef(BodyPartType.CHEST, "Spine2", mass=10.0, dimensions=(0.12, 0.15)),
        RagdollBodyDef(BodyPartType.NECK, "Neck", mass=2.0, dimensions=(0.05, 0.08)),
        RagdollBodyDef(BodyPartType.HEAD, "Head", mass=5.0, shape_type="sphere", dimensions=(0.1,)),

        # Left arm
        RagdollBodyDef(BodyPartType.SHOULDER_L, "LeftShoulder", mass=2.0, dimensions=(0.05, 0.1)),
        RagdollBodyDef(BodyPartType.UPPER_ARM_L, "LeftArm", mass=3.0, dimensions=(0.05, 0.25)),
        RagdollBodyDef(BodyPartType.LOWER_ARM_L, "LeftForeArm", mass=2.0, dimensions=(0.04, 0.22)),
        RagdollBodyDef(BodyPartType.HAND_L, "LeftHand", mass=0.5, dimensions=(0.03, 0.1)),

        # Right arm
        RagdollBodyDef(BodyPartType.SHOULDER_R, "RightShoulder", mass=2.0, dimensions=(0.05, 0.1)),
        RagdollBodyDef(BodyPartType.UPPER_ARM_R, "RightArm", mass=3.0, dimensions=(0.05, 0.25)),
        RagdollBodyDef(BodyPartType.LOWER_ARM_R, "RightForeArm", mass=2.0, dimensions=(0.04, 0.22)),
        RagdollBodyDef(BodyPartType.HAND_R, "RightHand", mass=0.5, dimensions=(0.03, 0.1)),

        # Left leg
        RagdollBodyDef(BodyPartType.UPPER_LEG_L, "LeftUpLeg", mass=6.0, dimensions=(0.06, 0.4)),
        RagdollBodyDef(BodyPartType.LOWER_LEG_L, "LeftLeg", mass=4.0, dimensions=(0.05, 0.35)),
        RagdollBodyDef(BodyPartType.FOOT_L, "LeftFoot", mass=1.0, shape_type="box", dimensions=(0.08, 0.05, 0.2)),

        # Right leg
        RagdollBodyDef(BodyPartType.UPPER_LEG_R, "RightUpLeg", mass=6.0, dimensions=(0.06, 0.4)),
        RagdollBodyDef(BodyPartType.LOWER_LEG_R, "RightLeg", mass=4.0, dimensions=(0.05, 0.35)),
        RagdollBodyDef(BodyPartType.FOOT_R, "RightFoot", mass=1.0, shape_type="box", dimensions=(0.08, 0.05, 0.2)),
    ]

    for body_def in body_defs:
        setup.bodies_per_bone[body_def.bone_name] = body_def

    # Define joints
    setup.joints = [
        # Spine chain
        RagdollJointDef(BodyPartType.PELVIS, BodyPartType.SPINE_LOWER, swing_limit=30, twist_limit=20),
        RagdollJointDef(BodyPartType.SPINE_LOWER, BodyPartType.SPINE_UPPER, swing_limit=30, twist_limit=20),
        RagdollJointDef(BodyPartType.SPINE_UPPER, BodyPartType.CHEST, swing_limit=30, twist_limit=20),
        RagdollJointDef(BodyPartType.CHEST, BodyPartType.NECK, swing_limit=40, twist_limit=30),
        RagdollJointDef(BodyPartType.NECK, BodyPartType.HEAD, swing_limit=60, twist_limit=45),

        # Left arm
        RagdollJointDef(BodyPartType.CHEST, BodyPartType.SHOULDER_L, swing_limit=30, twist_limit=15),
        RagdollJointDef(BodyPartType.SHOULDER_L, BodyPartType.UPPER_ARM_L, swing_limit=90, twist_limit=45),
        RagdollJointDef(BodyPartType.UPPER_ARM_L, BodyPartType.LOWER_ARM_L, joint_type="hinge", swing_limit=140, twist_limit=0),
        RagdollJointDef(BodyPartType.LOWER_ARM_L, BodyPartType.HAND_L, swing_limit=60, twist_limit=30),

        # Right arm
        RagdollJointDef(BodyPartType.CHEST, BodyPartType.SHOULDER_R, swing_limit=30, twist_limit=15),
        RagdollJointDef(BodyPartType.SHOULDER_R, BodyPartType.UPPER_ARM_R, swing_limit=90, twist_limit=45),
        RagdollJointDef(BodyPartType.UPPER_ARM_R, BodyPartType.LOWER_ARM_R, joint_type="hinge", swing_limit=140, twist_limit=0),
        RagdollJointDef(BodyPartType.LOWER_ARM_R, BodyPartType.HAND_R, swing_limit=60, twist_limit=30),

        # Left leg
        RagdollJointDef(BodyPartType.PELVIS, BodyPartType.UPPER_LEG_L, swing_limit=80, twist_limit=30),
        RagdollJointDef(BodyPartType.UPPER_LEG_L, BodyPartType.LOWER_LEG_L, joint_type="hinge", swing_limit=140, twist_limit=0),
        RagdollJointDef(BodyPartType.LOWER_LEG_L, BodyPartType.FOOT_L, swing_limit=40, twist_limit=20),

        # Right leg
        RagdollJointDef(BodyPartType.PELVIS, BodyPartType.UPPER_LEG_R, swing_limit=80, twist_limit=30),
        RagdollJointDef(BodyPartType.UPPER_LEG_R, BodyPartType.LOWER_LEG_R, joint_type="hinge", swing_limit=140, twist_limit=0),
        RagdollJointDef(BodyPartType.LOWER_LEG_R, BodyPartType.FOOT_R, swing_limit=40, twist_limit=20),
    ]

    setup.total_mass = sum(b.mass for b in body_defs)

    return setup


# =============================================================================
# Ragdoll Class
# =============================================================================

class Ragdoll:
    """
    Ragdoll physics system for a character.

    Handles:
    - Ragdoll creation from skeleton
    - Activation with initial velocity
    - Pose reading for animation
    - Deactivation and recovery
    """

    def __init__(
        self,
        physics: RagdollPhysicsInterface,
        skeleton: SkeletonInterface,
        setup: Optional[RagdollSetup] = None,
    ):
        self._physics = physics
        self._skeleton = skeleton
        self._setup = setup or create_default_humanoid_setup()

        # Runtime state
        self._state = RagdollState.INACTIVE
        self._body_ids: dict[BodyPartType, int] = {}
        self._joint_ids: list[int] = []
        self._bone_to_part: dict[str, BodyPartType] = {}

        # Timing
        self._activation_time: float = 0.0
        self._settled_time: float = 0.0
        self._blend_progress: float = 0.0

        # Callbacks
        self._on_activate: Optional[Callable[[], None]] = None
        self._on_deactivate: Optional[Callable[[], None]] = None
        self._on_settled: Optional[Callable[[], None]] = None

        # Build bone mapping
        for bone_name, body_def in self._setup.bodies_per_bone.items():
            self._bone_to_part[bone_name] = body_def.part_type

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> RagdollState:
        """Current ragdoll state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether ragdoll is currently active."""
        return self._state in (RagdollState.ACTIVE, RagdollState.ACTIVATING, RagdollState.SETTLING)

    @property
    def is_settled(self) -> bool:
        """Whether ragdoll has settled."""
        return self._state == RagdollState.SETTLING

    @property
    def blend_progress(self) -> float:
        """Progress of blend (0 = animation, 1 = ragdoll)."""
        return self._blend_progress

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_activate_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for ragdoll activation."""
        self._on_activate = callback

    def set_deactivate_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for ragdoll deactivation."""
        self._on_deactivate = callback

    def set_settled_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for when ragdoll settles."""
        self._on_settled = callback

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def setup_from_skeleton(self) -> bool:
        """
        Create ragdoll bodies from current skeleton pose.

        Returns:
            True if setup was successful
        """
        if self._body_ids:
            # Already set up
            return True

        # Create bodies
        for bone_name, body_def in self._setup.bodies_per_bone.items():
            bone_transform = self._skeleton.get_bone_transform(bone_name)

            body_id = self._physics.create_body(
                position=bone_transform.position + body_def.local_offset,
                rotation=bone_transform.rotation,
                shape_type=body_def.shape_type,
                dimensions=body_def.dimensions,
                mass=body_def.mass,
                collision_group=self._setup.collision_group,
                collision_mask=self._setup.collision_mask,
            )

            self._body_ids[body_def.part_type] = body_id

            # Start as kinematic
            self._physics.set_body_kinematic(body_id, True)

        # Create joints with validated limits
        for joint_def in self._setup.joints:
            parent_id = self._body_ids.get(joint_def.parent_part)
            child_id = self._body_ids.get(joint_def.child_part)

            if parent_id is None or child_id is None:
                continue

            # Validate joint limits to prevent invalid configurations
            swing_limit = max(0.0, min(180.0, joint_def.swing_limit))
            twist_limit = max(0.0, min(180.0, joint_def.twist_limit))

            joint_id = self._physics.create_joint(
                parent_body=parent_id,
                child_body=child_id,
                joint_type=joint_def.joint_type,
                anchor_parent=joint_def.local_anchor_parent,
                anchor_child=joint_def.local_anchor_child,
                limits=(swing_limit, twist_limit),
            )

            self._joint_ids.append(joint_id)

        return True

    def destroy(self) -> None:
        """Destroy all ragdoll bodies and joints."""
        # Destroy joints first
        for joint_id in self._joint_ids:
            self._physics.destroy_joint(joint_id)
        self._joint_ids.clear()

        # Destroy bodies
        for body_id in self._body_ids.values():
            self._physics.destroy_body(body_id)
        self._body_ids.clear()

        self._state = RagdollState.INACTIVE

    # -------------------------------------------------------------------------
    # Activation
    # -------------------------------------------------------------------------

    def activate(
        self,
        initial_velocity: Optional[Vector3] = None,
        impulse_point: Optional[Vector3] = None,
        impulse: Optional[Vector3] = None,
        time: float = 0.0,
    ) -> None:
        """
        Activate ragdoll physics.

        Args:
            initial_velocity: Initial velocity to apply to all bodies
            impulse_point: Point to apply impulse (e.g., hit location)
            impulse: Impulse vector to apply at point
            time: Current game time
        """
        if not self._body_ids:
            self.setup_from_skeleton()

        # Match current skeleton pose
        self._match_skeleton_pose()

        # Switch bodies to dynamic
        for body_id in self._body_ids.values():
            self._physics.set_body_kinematic(body_id, False)

        # Apply initial velocity from animation
        if initial_velocity is not None:
            for body_id in self._body_ids.values():
                self._physics.set_body_velocity(
                    body_id, initial_velocity, Vector3.zero()
                )

        # Apply impulse at specific point
        if impulse is not None and impulse_point is not None:
            self._apply_impulse_at_point(impulse_point, impulse)

        self._state = RagdollState.ACTIVATING
        self._activation_time = time
        self._blend_progress = 0.0

        if self._on_activate:
            self._on_activate()

    def _match_skeleton_pose(self) -> None:
        """Match ragdoll bodies to current skeleton pose."""
        for bone_name, body_def in self._setup.bodies_per_bone.items():
            body_id = self._body_ids.get(body_def.part_type)
            if body_id is None:
                continue

            bone_transform = self._skeleton.get_bone_transform(bone_name)
            self._physics.set_body_transform(
                body_id,
                bone_transform.position + body_def.local_offset,
                bone_transform.rotation,
            )

    def _apply_impulse_at_point(self, point: Vector3, impulse: Vector3) -> None:
        """Apply impulse at a specific world point."""
        # Find closest body
        closest_body: Optional[int] = None
        closest_dist = float("inf")

        for body_id in self._body_ids.values():
            pos, _ = self._physics.get_body_transform(body_id)
            dist = (pos - point).magnitude()
            if dist < closest_dist:
                closest_dist = dist
                closest_body = body_id

        if closest_body is not None:
            self._physics.apply_impulse(closest_body, impulse, point)

    # -------------------------------------------------------------------------
    # Deactivation
    # -------------------------------------------------------------------------

    def deactivate(self) -> None:
        """
        Deactivate ragdoll and switch back to animation.
        """
        if not self.is_active:
            return

        # Switch bodies back to kinematic
        for body_id in self._body_ids.values():
            self._physics.set_body_kinematic(body_id, True)

        self._state = RagdollState.RECOVERING
        self._blend_progress = 1.0

        if self._on_deactivate:
            self._on_deactivate()

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------

    def update(self, dt: float, current_time: float) -> None:
        """
        Update ragdoll state.

        Args:
            dt: Delta time
            current_time: Current game time
        """
        if self._state == RagdollState.INACTIVE:
            return

        if self._state == RagdollState.ACTIVATING:
            # Blend to ragdoll
            elapsed = (current_time - self._activation_time) * 1000.0
            self._blend_progress = min(1.0, elapsed / RAGDOLL_BLEND_TIME_MS)

            if self._blend_progress >= 1.0:
                self._state = RagdollState.ACTIVE

        elif self._state == RagdollState.ACTIVE:
            # Check if settled
            if self._is_settled():
                self._settled_time += dt * 1000.0
                if self._settled_time >= RAGDOLL_SETTLED_TIME_MS:
                    self._state = RagdollState.SETTLING
                    if self._on_settled:
                        self._on_settled()
            else:
                self._settled_time = 0.0

        elif self._state == RagdollState.RECOVERING:
            # Blend back to animation
            elapsed = (current_time - self._activation_time) * 1000.0
            self._blend_progress = max(0.0, 1.0 - elapsed / RAGDOLL_RECOVERY_TIME_MS)

            if self._blend_progress <= 0.0:
                self._state = RagdollState.INACTIVE

    def _is_settled(self) -> bool:
        """Check if ragdoll has settled (low velocity)."""
        total_velocity = 0.0

        for body_id in self._body_ids.values():
            linear, angular = self._physics.get_body_velocity(body_id)
            total_velocity += linear.magnitude() + angular.magnitude() * 0.1

        avg_velocity = total_velocity / len(self._body_ids) if self._body_ids else 0.0
        return avg_velocity < RAGDOLL_MIN_VELOCITY

    # -------------------------------------------------------------------------
    # Pose Reading
    # -------------------------------------------------------------------------

    def get_pose(self) -> RagdollPose:
        """
        Get current ragdoll pose.

        Returns:
            Current pose with all body transforms
        """
        pose = RagdollPose()

        for part_type, body_id in self._body_ids.items():
            pos, rot = self._physics.get_body_transform(body_id)
            linear, angular = self._physics.get_body_velocity(body_id)

            pose.body_states[part_type] = RagdollBodyState(
                body_id=body_id,
                position=pos,
                rotation=rot,
                velocity=linear,
                angular_velocity=angular,
            )

        # Root is pelvis
        if BodyPartType.PELVIS in pose.body_states:
            pelvis = pose.body_states[BodyPartType.PELVIS]
            pose.root_position = pelvis.position
            pose.root_rotation = pelvis.rotation

        return pose

    def get_bone_transform(self, bone_name: str) -> Optional[Transform]:
        """
        Get the ragdoll transform for a specific bone.

        Args:
            bone_name: Name of the bone

        Returns:
            Transform if bone has a ragdoll body
        """
        part_type = self._bone_to_part.get(bone_name)
        if part_type is None:
            return None

        body_id = self._body_ids.get(part_type)
        if body_id is None:
            return None

        pos, rot = self._physics.get_body_transform(body_id)

        # Subtract local offset
        body_def = self._setup.bodies_per_bone.get(bone_name)
        if body_def:
            pos = pos - body_def.local_offset

        return Transform(position=pos, rotation=rot)

    def write_to_skeleton(self) -> None:
        """Write ragdoll pose to skeleton bones."""
        for bone_name, part_type in self._bone_to_part.items():
            body_id = self._body_ids.get(part_type)
            if body_id is None:
                continue

            pos, rot = self._physics.get_body_transform(body_id)

            # Subtract local offset
            body_def = self._setup.bodies_per_bone.get(bone_name)
            if body_def:
                pos = pos - body_def.local_offset

            self._skeleton.set_bone_transform(
                bone_name, Transform(position=pos, rotation=rot)
            )

    # -------------------------------------------------------------------------
    # Physics Queries
    # -------------------------------------------------------------------------

    def get_center_of_mass(self) -> Vector3:
        """Calculate ragdoll center of mass."""
        total_mass = 0.0
        weighted_pos = Vector3.zero()

        for bone_name, body_def in self._setup.bodies_per_bone.items():
            body_id = self._body_ids.get(body_def.part_type)
            if body_id is None:
                continue

            pos, _ = self._physics.get_body_transform(body_id)
            weighted_pos = weighted_pos + pos * body_def.mass
            total_mass += body_def.mass

        if total_mass > 0:
            return weighted_pos / total_mass
        return Vector3.zero()

    def get_average_velocity(self) -> Vector3:
        """Get mass-weighted average velocity."""
        total_mass = 0.0
        weighted_vel = Vector3.zero()

        for bone_name, body_def in self._setup.bodies_per_bone.items():
            body_id = self._body_ids.get(body_def.part_type)
            if body_id is None:
                continue

            linear, _ = self._physics.get_body_velocity(body_id)
            weighted_vel = weighted_vel + linear * body_def.mass
            total_mass += body_def.mass

        if total_mass > 0:
            return weighted_vel / total_mass
        return Vector3.zero()

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def get_debug_info(self) -> dict[str, Any]:
        """Get debug information."""
        return {
            "state": self._state.value,
            "body_count": len(self._body_ids),
            "joint_count": len(self._joint_ids),
            "blend_progress": self._blend_progress,
            "settled_time": self._settled_time,
            "center_of_mass": (
                c := self.get_center_of_mass(),
                (c.x, c.y, c.z),
            )[1],
            "is_settled": self._is_settled() if self.is_active else False,
        }
