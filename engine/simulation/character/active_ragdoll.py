"""
Active Ragdoll Physics System.

Provides powered ragdoll functionality with PD controllers for joints,
balance control, and recovery behaviors for physics-driven character animation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .character_controller import Quaternion, Transform, Vector3
from .config import (
    BALANCE_THRESHOLD,
    DEFAULT_PD_KD,
    DEFAULT_PD_KP,
    MAX_TORQUE,
)
from .ragdoll import (
    BodyPartType,
    Ragdoll,
    RagdollPhysicsInterface,
    RagdollSetup,
    SkeletonInterface,
)


# =============================================================================
# Active Ragdoll Data Structures
# =============================================================================

class ActiveRagdollState(str, Enum):
    """State of the active ragdoll system."""
    INACTIVE = "inactive"
    BALANCED = "balanced"
    RECOVERING = "recovering"
    STUMBLING = "stumbling"
    FALLING = "falling"


class RecoveryBehavior(str, Enum):
    """Recovery behaviors for active ragdoll."""
    NONE = "none"
    STEP = "step"          # Take a recovery step
    STUMBLE = "stumble"    # Stumble forward
    FALL = "fall"          # Give up and fall
    BRACE = "brace"        # Brace for impact


@dataclass
class PDController:
    """
    Proportional-Derivative controller for joint control.

    Attributes:
        kp: Proportional gain
        kd: Derivative gain
        target_rotation: Target rotation to track
        max_torque: Maximum torque output
    """
    kp: float = DEFAULT_PD_KP
    kd: float = DEFAULT_PD_KD
    target_rotation: Quaternion = field(default_factory=Quaternion.identity)
    max_torque: float = MAX_TORQUE

    def compute_torque(
        self,
        current_rotation: Quaternion,
        current_angular_velocity: Vector3,
    ) -> Vector3:
        """
        Compute torque to drive joint to target.

        Args:
            current_rotation: Current joint rotation
            current_angular_velocity: Current angular velocity

        Returns:
            Torque vector to apply
        """
        # Calculate rotation error
        error = self._quaternion_error(current_rotation, self.target_rotation)

        # PD control: torque = kp * error - kd * velocity
        torque = error * self.kp - current_angular_velocity * self.kd

        # Clamp magnitude with zero-magnitude protection
        mag = torque.magnitude()
        if mag > self.max_torque and mag > 1e-10:
            torque = torque * (self.max_torque / mag)

        return torque

    def _quaternion_error(
        self,
        current: Quaternion,
        target: Quaternion,
    ) -> Vector3:
        """Calculate rotation error as axis-angle."""
        # Relative rotation: target * inverse(current)
        inv_current = Quaternion(-current.x, -current.y, -current.z, current.w)

        # Multiply quaternions
        rel_x = target.w * inv_current.x + target.x * inv_current.w + target.y * inv_current.z - target.z * inv_current.y
        rel_y = target.w * inv_current.y - target.x * inv_current.z + target.y * inv_current.w + target.z * inv_current.x
        rel_z = target.w * inv_current.z + target.x * inv_current.y - target.y * inv_current.x + target.z * inv_current.w
        rel_w = target.w * inv_current.w - target.x * inv_current.x - target.y * inv_current.y - target.z * inv_current.z

        # Convert to axis-angle
        # Ensure w is positive (shortest path)
        if rel_w < 0:
            rel_x, rel_y, rel_z, rel_w = -rel_x, -rel_y, -rel_z, -rel_w

        # Approximate for small angles: axis * angle ~= 2 * (x, y, z)
        return Vector3(rel_x * 2.0, rel_y * 2.0, rel_z * 2.0)


@dataclass
class JointController:
    """
    Controller configuration for a specific joint.

    Attributes:
        part_type: Body part this controller affects
        pd_controller: PD controller for this joint
        strength: Overall strength multiplier (0-1)
        enabled: Whether this joint is actively controlled
    """
    part_type: BodyPartType
    pd_controller: PDController = field(default_factory=PDController)
    strength: float = 1.0
    enabled: bool = True


@dataclass
class BalanceConfig:
    """
    Configuration for balance control.

    Attributes:
        com_target: Target center of mass position (relative to feet)
        com_threshold: Maximum allowed COM offset before recovery
        ankle_gain: Gain for ankle strategy
        hip_gain: Gain for hip strategy
        step_threshold: COM offset to trigger stepping
    """
    com_target: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.9, 0.0))
    com_threshold: float = BALANCE_THRESHOLD
    ankle_gain: float = 100.0
    hip_gain: float = 50.0
    step_threshold: float = 0.4


# =============================================================================
# Active Ragdoll
# =============================================================================

class ActiveRagdoll:
    """
    Powered ragdoll system with balance control.

    Features:
    - PD controllers for each joint
    - Center of mass tracking
    - Ankle/hip balance strategies
    - Recovery behaviors (step, stumble, fall)
    - Strength modulation
    """

    def __init__(
        self,
        ragdoll: Ragdoll,
        physics: RagdollPhysicsInterface,
    ):
        self._ragdoll = ragdoll
        self._physics = physics

        # State
        self._state = ActiveRagdollState.INACTIVE
        self._balance_config = BalanceConfig()

        # Joint controllers
        self._controllers: dict[BodyPartType, JointController] = {}
        self._initialize_controllers()

        # Balance tracking
        self._center_of_mass = Vector3.zero()
        self._com_velocity = Vector3.zero()
        self._support_position = Vector3.zero()
        self._balance_error = Vector3.zero()

        # Recovery state
        self._recovery_behavior = RecoveryBehavior.NONE
        self._recovery_time: float = 0.0
        self._step_direction = Vector3.zero()

        # Callbacks
        self._on_lose_balance: Optional[Callable[[], None]] = None
        self._on_recover_balance: Optional[Callable[[], None]] = None
        self._on_fall: Optional[Callable[[], None]] = None

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _initialize_controllers(self) -> None:
        """Initialize PD controllers for all body parts."""
        # Spine - high strength for posture
        spine_parts = [
            BodyPartType.PELVIS,
            BodyPartType.SPINE_LOWER,
            BodyPartType.SPINE_UPPER,
            BodyPartType.CHEST,
        ]
        for part in spine_parts:
            self._controllers[part] = JointController(
                part_type=part,
                pd_controller=PDController(kp=400.0, kd=40.0, max_torque=600.0),
                strength=1.0,
            )

        # Neck and head - medium strength
        for part in [BodyPartType.NECK, BodyPartType.HEAD]:
            self._controllers[part] = JointController(
                part_type=part,
                pd_controller=PDController(kp=200.0, kd=20.0, max_torque=300.0),
                strength=0.8,
            )

        # Arms - lower strength, more reactive
        arm_parts = [
            BodyPartType.SHOULDER_L, BodyPartType.UPPER_ARM_L,
            BodyPartType.LOWER_ARM_L, BodyPartType.HAND_L,
            BodyPartType.SHOULDER_R, BodyPartType.UPPER_ARM_R,
            BodyPartType.LOWER_ARM_R, BodyPartType.HAND_R,
        ]
        for part in arm_parts:
            self._controllers[part] = JointController(
                part_type=part,
                pd_controller=PDController(kp=150.0, kd=15.0, max_torque=200.0),
                strength=0.6,
            )

        # Legs - high strength for balance
        leg_parts = [
            BodyPartType.UPPER_LEG_L, BodyPartType.LOWER_LEG_L, BodyPartType.FOOT_L,
            BodyPartType.UPPER_LEG_R, BodyPartType.LOWER_LEG_R, BodyPartType.FOOT_R,
        ]
        for part in leg_parts:
            self._controllers[part] = JointController(
                part_type=part,
                pd_controller=PDController(kp=350.0, kd=35.0, max_torque=500.0),
                strength=1.0,
            )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> ActiveRagdollState:
        """Current active ragdoll state."""
        return self._state

    @property
    def is_balanced(self) -> bool:
        """Whether character is currently balanced."""
        return self._state == ActiveRagdollState.BALANCED

    @property
    def center_of_mass(self) -> Vector3:
        """Current center of mass."""
        return self._center_of_mass

    @property
    def balance_error(self) -> Vector3:
        """Current balance error (COM offset from support)."""
        return self._balance_error

    @property
    def recovery_behavior(self) -> RecoveryBehavior:
        """Current recovery behavior."""
        return self._recovery_behavior

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_lose_balance_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for when balance is lost."""
        self._on_lose_balance = callback

    def set_recover_balance_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for when balance is recovered."""
        self._on_recover_balance = callback

    def set_fall_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for when character falls."""
        self._on_fall = callback

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_balance_config(self, config: BalanceConfig) -> None:
        """Set balance configuration."""
        self._balance_config = config

    def set_joint_strength(self, part_type: BodyPartType, strength: float) -> None:
        """Set strength for a specific joint (0-1)."""
        if part_type in self._controllers:
            self._controllers[part_type].strength = max(0.0, min(1.0, strength))

    def set_global_strength(self, strength: float) -> None:
        """Set strength for all joints (0-1)."""
        for controller in self._controllers.values():
            controller.strength = max(0.0, min(1.0, strength))

    def enable_joint(self, part_type: BodyPartType, enabled: bool = True) -> None:
        """Enable or disable a joint controller."""
        if part_type in self._controllers:
            self._controllers[part_type].enabled = enabled

    def set_target_pose(self, pose: dict[BodyPartType, Quaternion]) -> None:
        """
        Set target pose for all joints.

        Args:
            pose: Dictionary mapping body parts to target rotations
        """
        for part_type, rotation in pose.items():
            if part_type in self._controllers:
                self._controllers[part_type].pd_controller.target_rotation = rotation

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """
        Update active ragdoll physics.

        Args:
            dt: Delta time in seconds
        """
        if self._state == ActiveRagdollState.INACTIVE:
            return

        if not self._ragdoll.is_active:
            self._state = ActiveRagdollState.INACTIVE
            return

        # Update center of mass
        self._update_center_of_mass()

        # Update balance
        self._update_balance(dt)

        # Apply joint torques
        self._apply_joint_torques()

        # Apply balance corrections
        if self._state in (ActiveRagdollState.BALANCED, ActiveRagdollState.RECOVERING):
            self._apply_balance_control()

        # Update recovery
        if self._recovery_behavior != RecoveryBehavior.NONE:
            self._update_recovery(dt)

    def _update_center_of_mass(self) -> None:
        """Update center of mass tracking."""
        old_com = self._center_of_mass
        self._center_of_mass = self._ragdoll.get_center_of_mass()
        self._com_velocity = self._center_of_mass - old_com

    def _update_balance(self, dt: float) -> None:
        """Update balance state."""
        # Calculate support position (average of feet)
        pose = self._ragdoll.get_pose()

        foot_l = pose.body_states.get(BodyPartType.FOOT_L)
        foot_r = pose.body_states.get(BodyPartType.FOOT_R)

        if foot_l and foot_r:
            self._support_position = (foot_l.position + foot_r.position) * 0.5
        elif foot_l:
            self._support_position = foot_l.position
        elif foot_r:
            self._support_position = foot_r.position

        # Calculate balance error (horizontal offset)
        self._balance_error = Vector3(
            self._center_of_mass.x - self._support_position.x,
            0.0,
            self._center_of_mass.z - self._support_position.z,
        )

        error_magnitude = self._balance_error.magnitude()

        # State transitions
        if self._state == ActiveRagdollState.BALANCED:
            if error_magnitude > self._balance_config.com_threshold:
                self._state = ActiveRagdollState.RECOVERING
                self._determine_recovery_behavior()
                if self._on_lose_balance:
                    self._on_lose_balance()

        elif self._state == ActiveRagdollState.RECOVERING:
            if error_magnitude < self._balance_config.com_threshold * 0.5:
                self._state = ActiveRagdollState.BALANCED
                self._recovery_behavior = RecoveryBehavior.NONE
                if self._on_recover_balance:
                    self._on_recover_balance()
            elif error_magnitude > self._balance_config.step_threshold * 1.5:
                self._state = ActiveRagdollState.FALLING
                self._recovery_behavior = RecoveryBehavior.FALL
                if self._on_fall:
                    self._on_fall()

    def _determine_recovery_behavior(self) -> None:
        """Determine appropriate recovery behavior."""
        error_mag = self._balance_error.magnitude()

        if error_mag < self._balance_config.step_threshold:
            # Small error - try ankle/hip strategy
            self._recovery_behavior = RecoveryBehavior.NONE

        elif error_mag < self._balance_config.step_threshold * 1.2:
            # Medium error - take a recovery step
            self._recovery_behavior = RecoveryBehavior.STEP
            self._step_direction = self._balance_error.normalized()
            self._recovery_time = 0.0

        elif error_mag < self._balance_config.step_threshold * 1.5:
            # Large error - stumble
            self._recovery_behavior = RecoveryBehavior.STUMBLE
            self._step_direction = self._balance_error.normalized()
            self._recovery_time = 0.0

        else:
            # Too large - fall
            self._recovery_behavior = RecoveryBehavior.FALL

    # -------------------------------------------------------------------------
    # Joint Control
    # -------------------------------------------------------------------------

    def _apply_joint_torques(self) -> None:
        """Apply PD control torques to all joints."""
        pose = self._ragdoll.get_pose()

        for part_type, controller in self._controllers.items():
            if not controller.enabled or controller.strength <= 0:
                continue

            body_state = pose.body_states.get(part_type)
            if body_state is None:
                continue

            # Compute torque
            torque = controller.pd_controller.compute_torque(
                body_state.rotation,
                body_state.angular_velocity,
            )

            # Apply strength
            torque = torque * controller.strength

            # Apply to body
            self._apply_torque(body_state.body_id, torque)

    def compute_torque(self, part_type: BodyPartType) -> Vector3:
        """
        Compute torque for a specific body part.

        Args:
            part_type: Body part to compute torque for

        Returns:
            Computed torque vector
        """
        controller = self._controllers.get(part_type)
        if controller is None or not controller.enabled:
            return Vector3.zero()

        pose = self._ragdoll.get_pose()
        body_state = pose.body_states.get(part_type)
        if body_state is None:
            return Vector3.zero()

        torque = controller.pd_controller.compute_torque(
            body_state.rotation,
            body_state.angular_velocity,
        )

        return torque * controller.strength

    def _apply_torque(self, body_id: int, torque: Vector3) -> None:
        """Apply torque to a physics body."""
        # Convert to impulse for this frame
        # This is a simplified approach - real implementation would use
        # the physics engine's torque API
        pass  # Placeholder - implement based on physics backend

    # -------------------------------------------------------------------------
    # Balance Control
    # -------------------------------------------------------------------------

    def _apply_balance_control(self) -> None:
        """Apply balance control strategies."""
        # Ankle strategy - push against ground
        self._ankle_strategy()

        # Hip strategy - adjust torso lean
        self._hip_strategy()

    def ankle_strategy(self) -> Vector3:
        """
        Calculate ankle correction torque.

        Returns:
            Torque to apply to ankles
        """
        # Push COM back toward support
        correction = -self._balance_error * self._balance_config.ankle_gain
        return correction

    def _ankle_strategy(self) -> None:
        """Apply ankle strategy for balance."""
        correction = self.ankle_strategy()

        # Apply to both feet
        for foot_part in [BodyPartType.FOOT_L, BodyPartType.FOOT_R]:
            controller = self._controllers.get(foot_part)
            if controller and controller.enabled:
                pose = self._ragdoll.get_pose()
                body_state = pose.body_states.get(foot_part)
                if body_state:
                    self._apply_torque(body_state.body_id, correction * 0.5)

    def _hip_strategy(self) -> None:
        """Apply hip strategy for balance."""
        # Lean torso opposite to error
        correction = -self._balance_error * self._balance_config.hip_gain

        # Apply to spine
        for spine_part in [BodyPartType.PELVIS, BodyPartType.SPINE_LOWER]:
            controller = self._controllers.get(spine_part)
            if controller and controller.enabled:
                pose = self._ragdoll.get_pose()
                body_state = pose.body_states.get(spine_part)
                if body_state:
                    self._apply_torque(body_state.body_id, correction * 0.3)

    # -------------------------------------------------------------------------
    # Recovery Behaviors
    # -------------------------------------------------------------------------

    def _update_recovery(self, dt: float) -> None:
        """Update recovery behavior."""
        self._recovery_time += dt

        if self._recovery_behavior == RecoveryBehavior.STEP:
            self._execute_step_recovery()
        elif self._recovery_behavior == RecoveryBehavior.STUMBLE:
            self._execute_stumble_recovery()
        elif self._recovery_behavior == RecoveryBehavior.BRACE:
            self._execute_brace_recovery()

    def _execute_step_recovery(self) -> None:
        """Execute recovery step."""
        # Determine which foot to step with
        step_foot = BodyPartType.FOOT_R
        if self._step_direction.x < 0:
            step_foot = BodyPartType.FOOT_L

        # Reduce strength on stepping leg to allow movement
        self.set_joint_strength(step_foot, 0.3)

        # Increase strength on supporting leg
        support_foot = BodyPartType.FOOT_L if step_foot == BodyPartType.FOOT_R else BodyPartType.FOOT_R
        self.set_joint_strength(support_foot, 1.0)

        # Reset after step
        if self._recovery_time > 0.5:
            self.set_joint_strength(step_foot, 1.0)
            self._recovery_behavior = RecoveryBehavior.NONE

    def _execute_stumble_recovery(self) -> None:
        """Execute stumble recovery."""
        # Reduce leg strength to allow stumbling
        for leg_part in [
            BodyPartType.UPPER_LEG_L, BodyPartType.LOWER_LEG_L,
            BodyPartType.UPPER_LEG_R, BodyPartType.LOWER_LEG_R,
        ]:
            self.set_joint_strength(leg_part, 0.5)

        # Arms out for balance
        for arm_part in [
            BodyPartType.UPPER_ARM_L, BodyPartType.UPPER_ARM_R,
        ]:
            self.set_joint_strength(arm_part, 0.8)

        if self._recovery_time > 1.0:
            self.set_global_strength(1.0)
            self._recovery_behavior = RecoveryBehavior.NONE

    def _execute_brace_recovery(self) -> None:
        """Execute brace for impact."""
        # Extend arms
        for arm_part in [
            BodyPartType.UPPER_ARM_L, BodyPartType.LOWER_ARM_L,
            BodyPartType.UPPER_ARM_R, BodyPartType.LOWER_ARM_R,
        ]:
            self.set_joint_strength(arm_part, 1.0)

        # Reduce spine flexibility for rigid impact
        for spine_part in [BodyPartType.SPINE_LOWER, BodyPartType.SPINE_UPPER]:
            self.set_joint_strength(spine_part, 1.0)

    # -------------------------------------------------------------------------
    # Activation / Deactivation
    # -------------------------------------------------------------------------

    def activate(self) -> None:
        """Activate active ragdoll control."""
        if not self._ragdoll.is_active:
            return

        self._state = ActiveRagdollState.BALANCED
        self._recovery_behavior = RecoveryBehavior.NONE
        self.set_global_strength(1.0)

    def deactivate(self) -> None:
        """Deactivate active ragdoll control."""
        self._state = ActiveRagdollState.INACTIVE
        self._recovery_behavior = RecoveryBehavior.NONE

    def set_falling(self) -> None:
        """Force transition to falling state."""
        self._state = ActiveRagdollState.FALLING
        self._recovery_behavior = RecoveryBehavior.FALL
        self.set_global_strength(0.3)

        if self._on_fall:
            self._on_fall()

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def get_debug_info(self) -> dict[str, Any]:
        """Get debug information."""
        return {
            "state": self._state.value,
            "recovery_behavior": self._recovery_behavior.value,
            "center_of_mass": (
                self._center_of_mass.x,
                self._center_of_mass.y,
                self._center_of_mass.z,
            ),
            "balance_error": (
                self._balance_error.x,
                self._balance_error.y,
                self._balance_error.z,
            ),
            "balance_error_magnitude": self._balance_error.magnitude(),
            "support_position": (
                self._support_position.x,
                self._support_position.y,
                self._support_position.z,
            ),
            "recovery_time": self._recovery_time,
            "controller_count": len(self._controllers),
        }
