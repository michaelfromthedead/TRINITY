"""
Character Physical Interactions.

Provides functionality for character-to-character and character-to-object
physical interactions including pushing, grabbing, carrying, throwing,
climbing, and vaulting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .character_controller import (
    CollisionHit,
    PhysicsWorldInterface,
    Quaternion,
    Transform,
    Vector3,
)
from .config import (
    CARRY_MASS_LIMIT,
    CLIMB_MAX_HEIGHT,
    GRAB_DISTANCE,
    PUSH_FORCE,
    THROW_FORCE_MULTIPLIER,
    VAULT_MAX_HEIGHT,
)


# =============================================================================
# Interaction Types
# =============================================================================

class InteractionType(str, Enum):
    """Types of physical interactions."""
    NONE = "none"
    PUSH = "push"
    GRAB = "grab"
    CARRY = "carry"
    THROW = "throw"
    CLIMB = "climb"
    VAULT = "vault"
    LEDGE_GRAB = "ledge_grab"
    ROPE_CLIMB = "rope_climb"
    LADDER = "ladder"


class GrabState(str, Enum):
    """State of a grab interaction."""
    NONE = "none"
    REACHING = "reaching"
    HOLDING = "holding"
    RELEASING = "releasing"


# =============================================================================
# Interaction Data
# =============================================================================

@dataclass
class InteractionTarget:
    """
    Target for a physical interaction.

    Attributes:
        entity_id: ID of target entity
        body_id: Physics body ID
        position: World position
        mass: Mass of target
        is_character: Whether target is another character
        can_be_grabbed: Whether target can be grabbed
        can_be_carried: Whether target can be carried
    """
    entity_id: int = 0
    body_id: int = 0
    position: Vector3 = field(default_factory=Vector3.zero)
    mass: float = 1.0
    is_character: bool = False
    can_be_grabbed: bool = True
    can_be_carried: bool = True
    grab_offset: Vector3 = field(default_factory=Vector3.zero)


@dataclass
class GrabInfo:
    """
    Information about a grab interaction.

    Attributes:
        target: Target being grabbed
        grab_point: Local grab point on target
        hand: Which hand is grabbing ("left", "right", "both")
        state: Current grab state
        hold_time: Time held
    """
    target: Optional[InteractionTarget] = None
    grab_point: Vector3 = field(default_factory=Vector3.zero)
    hand: str = "right"
    state: GrabState = GrabState.NONE
    hold_time: float = 0.0
    constraint_id: Optional[int] = None


@dataclass
class ClimbInfo:
    """
    Information about a climb interaction.

    Attributes:
        surface_normal: Normal of climb surface
        surface_position: Position on surface
        climb_direction: Direction of climbing
        progress: Climb progress (0-1)
        height: Total climb height
    """
    surface_normal: Vector3 = field(default_factory=Vector3.forward)
    surface_position: Vector3 = field(default_factory=Vector3.zero)
    climb_direction: Vector3 = field(default_factory=Vector3.up)
    progress: float = 0.0
    height: float = 0.0
    start_position: Vector3 = field(default_factory=Vector3.zero)
    end_position: Vector3 = field(default_factory=Vector3.zero)


@dataclass
class VaultInfo:
    """
    Information about a vault interaction.

    Attributes:
        obstacle_position: Position of obstacle
        obstacle_height: Height of obstacle
        vault_direction: Direction of vault
        progress: Vault progress (0-1)
        trajectory: List of positions in vault trajectory
    """
    obstacle_position: Vector3 = field(default_factory=Vector3.zero)
    obstacle_height: float = 0.0
    vault_direction: Vector3 = field(default_factory=Vector3.forward)
    progress: float = 0.0
    trajectory: list[Vector3] = field(default_factory=list)
    start_position: Vector3 = field(default_factory=Vector3.zero)


# =============================================================================
# Character Interaction Manager
# =============================================================================

class CharacterInteractionManager:
    """
    Manages physical interactions for a character.

    Handles:
    - Pushing characters and objects
    - Character vs character collisions
    - Grabbing and carrying objects
    - Throwing objects
    - Climbing ledges
    - Vaulting obstacles
    """

    def __init__(self, physics_world: PhysicsWorldInterface):
        self._physics = physics_world

        # Current interaction state
        self._current_interaction = InteractionType.NONE
        self._grab_info = GrabInfo()
        self._climb_info = ClimbInfo()
        self._vault_info = VaultInfo()

        # Configuration
        self._push_force = PUSH_FORCE
        self._grab_distance = GRAB_DISTANCE
        self._carry_mass_limit = CARRY_MASS_LIMIT
        self._throw_multiplier = THROW_FORCE_MULTIPLIER
        self._vault_max_height = VAULT_MAX_HEIGHT
        self._climb_max_height = CLIMB_MAX_HEIGHT

        # Character state
        self._character_position = Vector3.zero()
        self._character_forward = Vector3.forward()
        self._character_velocity = Vector3.zero()
        self._character_mass = 70.0
        self._character_body_id: Optional[int] = None

        # Callbacks
        self._on_interaction_start: Optional[Callable[[InteractionType], None]] = None
        self._on_interaction_end: Optional[Callable[[InteractionType], None]] = None
        self._on_grab: Optional[Callable[[InteractionTarget], None]] = None
        self._on_throw: Optional[Callable[[InteractionTarget, Vector3], None]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def current_interaction(self) -> InteractionType:
        """Current interaction type."""
        return self._current_interaction

    @property
    def is_interacting(self) -> bool:
        """Whether any interaction is active."""
        return self._current_interaction != InteractionType.NONE

    @property
    def is_grabbing(self) -> bool:
        """Whether currently grabbing something."""
        return self._grab_info.state in (GrabState.HOLDING, GrabState.REACHING)

    @property
    def is_carrying(self) -> bool:
        """Whether currently carrying something."""
        return (
            self._current_interaction == InteractionType.CARRY and
            self._grab_info.state == GrabState.HOLDING
        )

    @property
    def is_climbing(self) -> bool:
        """Whether currently climbing."""
        return self._current_interaction in (
            InteractionType.CLIMB,
            InteractionType.LEDGE_GRAB,
            InteractionType.ROPE_CLIMB,
            InteractionType.LADDER,
        )

    @property
    def is_vaulting(self) -> bool:
        """Whether currently vaulting."""
        return self._current_interaction == InteractionType.VAULT

    @property
    def grab_info(self) -> GrabInfo:
        """Current grab information."""
        return self._grab_info

    @property
    def climb_info(self) -> ClimbInfo:
        """Current climb information."""
        return self._climb_info

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_interaction_callbacks(
        self,
        on_start: Optional[Callable[[InteractionType], None]] = None,
        on_end: Optional[Callable[[InteractionType], None]] = None,
    ) -> None:
        """Set interaction start/end callbacks."""
        self._on_interaction_start = on_start
        self._on_interaction_end = on_end

    def set_grab_callback(
        self, callback: Optional[Callable[[InteractionTarget], None]]
    ) -> None:
        """Set callback for grab events."""
        self._on_grab = callback

    def set_throw_callback(
        self, callback: Optional[Callable[[InteractionTarget, Vector3], None]]
    ) -> None:
        """Set callback for throw events."""
        self._on_throw = callback

    # -------------------------------------------------------------------------
    # State Updates
    # -------------------------------------------------------------------------

    def update_character_state(
        self,
        position: Vector3,
        forward: Vector3,
        velocity: Vector3,
        body_id: Optional[int] = None,
    ) -> None:
        """Update character state for interactions."""
        self._character_position = position
        self._character_forward = forward.normalized()
        self._character_velocity = velocity
        self._character_body_id = body_id

    # -------------------------------------------------------------------------
    # Pushing
    # -------------------------------------------------------------------------

    def push_character(
        self,
        target: InteractionTarget,
        push_direction: Optional[Vector3] = None,
        force_multiplier: float = 1.0,
    ) -> bool:
        """
        Push another character.

        Args:
            target: Target to push
            push_direction: Direction to push (defaults to forward)
            force_multiplier: Force multiplier

        Returns:
            True if push was applied
        """
        direction = push_direction if push_direction else self._character_forward
        direction = direction.normalized()

        # Calculate force based on mass ratio
        mass_ratio = self._character_mass / max(target.mass, 1.0)
        force = direction * self._push_force * force_multiplier * min(mass_ratio, 2.0)

        # Apply impulse
        self._physics.apply_impulse(
            target.body_id,
            force,
            target.position,
        )

        return True

    def character_vs_character(
        self,
        other_target: InteractionTarget,
        collision_normal: Vector3,
    ) -> tuple[Vector3, Vector3]:
        """
        Resolve character vs character collision.

        Args:
            other_target: Other character
            collision_normal: Collision normal

        Returns:
            Tuple of (self_impulse, other_impulse)
        """
        # Relative velocity
        rel_velocity = self._character_velocity  # Simplified

        # Coefficient of restitution
        restitution = 0.1  # Characters don't bounce much

        # Calculate impulse magnitude (simplified)
        total_mass = self._character_mass + other_target.mass
        impulse_mag = (
            -(1 + restitution) * rel_velocity.dot(collision_normal) /
            (1.0 / self._character_mass + 1.0 / other_target.mass)
        )

        impulse = collision_normal * impulse_mag

        self_impulse = -impulse / self._character_mass
        other_impulse = impulse / other_target.mass

        return self_impulse, other_impulse

    # -------------------------------------------------------------------------
    # Grabbing
    # -------------------------------------------------------------------------

    def grab_object(
        self,
        target: InteractionTarget,
        hand: str = "right",
    ) -> bool:
        """
        Grab an object.

        Args:
            target: Object to grab
            hand: Which hand to use

        Returns:
            True if grab was initiated
        """
        if not target.can_be_grabbed:
            return False

        # Check distance
        to_target = target.position - self._character_position
        distance = to_target.magnitude()

        if distance > self._grab_distance:
            return False

        # Start grab
        self._grab_info = GrabInfo(
            target=target,
            grab_point=target.grab_offset,
            hand=hand,
            state=GrabState.REACHING,
            hold_time=0.0,
        )

        self._current_interaction = InteractionType.GRAB

        if self._on_interaction_start:
            self._on_interaction_start(InteractionType.GRAB)
        if self._on_grab:
            self._on_grab(target)

        return True

    def release_grab(self) -> Optional[InteractionTarget]:
        """
        Release currently grabbed object.

        Returns:
            The released target, or None
        """
        if self._grab_info.state == GrabState.NONE:
            return None

        target = self._grab_info.target

        # Remove constraint if any
        if self._grab_info.constraint_id is not None:
            # Remove physics constraint
            pass

        old_interaction = self._current_interaction

        self._grab_info = GrabInfo()
        self._current_interaction = InteractionType.NONE

        if self._on_interaction_end:
            self._on_interaction_end(old_interaction)

        return target

    def confirm_grab(self) -> bool:
        """
        Confirm grab after reaching (transition from REACHING to HOLDING).

        Returns:
            True if grab was confirmed
        """
        if self._grab_info.state != GrabState.REACHING:
            return False

        self._grab_info.state = GrabState.HOLDING

        # Create physics constraint to hold object
        # This would be implemented based on physics backend

        return True

    # -------------------------------------------------------------------------
    # Carrying
    # -------------------------------------------------------------------------

    def carry_object(self, target: InteractionTarget) -> bool:
        """
        Start carrying a grabbed object.

        Args:
            target: Object to carry

        Returns:
            True if carrying started
        """
        if not target.can_be_carried:
            return False

        if target.mass > self._carry_mass_limit:
            return False

        # Must already be grabbed
        if self._grab_info.target != target:
            if not self.grab_object(target, "both"):
                return False

        self._current_interaction = InteractionType.CARRY

        if self._on_interaction_start:
            self._on_interaction_start(InteractionType.CARRY)

        return True

    def update_carried_object(self, carry_position: Vector3) -> None:
        """
        Update position of carried object.

        Args:
            carry_position: World position to hold object
        """
        if not self.is_carrying or self._grab_info.target is None:
            return

        target = self._grab_info.target

        # Set target position (with interpolation for smoothness)
        # This would use physics constraints in real implementation
        target.position = carry_position

    # -------------------------------------------------------------------------
    # Throwing
    # -------------------------------------------------------------------------

    def throw_object(
        self,
        throw_direction: Vector3,
        force: float,
    ) -> bool:
        """
        Throw currently held object.

        Args:
            throw_direction: Direction to throw
            force: Throw force

        Returns:
            True if throw was executed
        """
        if not self.is_carrying and not self.is_grabbing:
            return False

        target = self._grab_info.target
        if target is None:
            return False

        # Validate throw direction
        if throw_direction.magnitude() < 0.001:
            return False

        # Calculate throw velocity
        direction = throw_direction.normalized()
        throw_force = force * self._throw_multiplier
        velocity = direction * throw_force / max(target.mass, 0.1)

        # Store target info before release
        target_body_id = target.body_id
        target_position = target.position

        # Release grab first
        self.release_grab()

        # Apply impulse to the released object
        if target_body_id > 0:
            self._physics.apply_impulse(
                target_body_id,
                direction * throw_force,
                target_position,
            )

        if self._on_throw:
            self._on_throw(target, velocity)

        return True

    # -------------------------------------------------------------------------
    # Climbing
    # -------------------------------------------------------------------------

    def climb_ledge(
        self,
        ledge_position: Vector3,
        ledge_normal: Vector3,
        climb_height: float,
    ) -> bool:
        """
        Start climbing a ledge.

        Args:
            ledge_position: Position of ledge
            ledge_normal: Normal pointing away from wall
            climb_height: Height to climb

        Returns:
            True if climb was initiated
        """
        if climb_height > self._climb_max_height:
            return False

        if self.is_interacting:
            return False

        # Calculate climb trajectory
        start_pos = self._character_position
        end_pos = Vector3(
            ledge_position.x - ledge_normal.x * 0.5,
            ledge_position.y,
            ledge_position.z - ledge_normal.z * 0.5,
        )

        self._climb_info = ClimbInfo(
            surface_normal=ledge_normal,
            surface_position=ledge_position,
            climb_direction=Vector3.up(),
            progress=0.0,
            height=climb_height,
            start_position=start_pos,
            end_position=end_pos,
        )

        self._current_interaction = InteractionType.CLIMB

        if self._on_interaction_start:
            self._on_interaction_start(InteractionType.CLIMB)

        return True

    def update_climb(self, dt: float, climb_speed: float = 2.0) -> Vector3:
        """
        Update climb progress.

        Args:
            dt: Delta time
            climb_speed: Climb speed

        Returns:
            Current position during climb
        """
        if not self.is_climbing:
            return self._character_position

        # Update progress
        climb_rate = climb_speed / max(self._climb_info.height, 0.1)
        self._climb_info.progress += climb_rate * dt
        self._climb_info.progress = min(1.0, self._climb_info.progress)

        # Calculate position along trajectory
        t = self._climb_info.progress

        # Smooth curve: start slow, accelerate, end slow
        t_smooth = t * t * (3.0 - 2.0 * t)

        position = Vector3.lerp(
            self._climb_info.start_position,
            self._climb_info.end_position,
            t_smooth,
        )

        # Check if complete
        if self._climb_info.progress >= 1.0:
            self._end_interaction()

        return position

    def cancel_climb(self) -> None:
        """Cancel current climb."""
        if self.is_climbing:
            self._end_interaction()

    # -------------------------------------------------------------------------
    # Vaulting
    # -------------------------------------------------------------------------

    def vault_obstacle(
        self,
        obstacle_position: Vector3,
        obstacle_height: float,
        vault_direction: Vector3,
    ) -> bool:
        """
        Start vaulting over an obstacle.

        Args:
            obstacle_position: Position of obstacle
            obstacle_height: Height of obstacle
            vault_direction: Direction of vault

        Returns:
            True if vault was initiated
        """
        if obstacle_height > self._vault_max_height:
            return False

        if self.is_interacting:
            return False

        direction = vault_direction.horizontal().normalized()
        start_pos = self._character_position

        # Generate vault trajectory
        trajectory = self._generate_vault_trajectory(
            start_pos, obstacle_position, obstacle_height, direction
        )

        self._vault_info = VaultInfo(
            obstacle_position=obstacle_position,
            obstacle_height=obstacle_height,
            vault_direction=direction,
            progress=0.0,
            trajectory=trajectory,
            start_position=start_pos,
        )

        self._current_interaction = InteractionType.VAULT

        if self._on_interaction_start:
            self._on_interaction_start(InteractionType.VAULT)

        return True

    def _generate_vault_trajectory(
        self,
        start: Vector3,
        obstacle: Vector3,
        height: float,
        direction: Vector3,
    ) -> list[Vector3]:
        """Generate vault trajectory points."""
        points = []

        # Start
        points.append(start)

        # Approach
        approach = Vector3(
            obstacle.x - direction.x * 0.3,
            start.y,
            obstacle.z - direction.z * 0.3,
        )
        points.append(approach)

        # Over obstacle
        over = Vector3(
            obstacle.x,
            obstacle.y + height + 0.2,  # Clear obstacle
            obstacle.z,
        )
        points.append(over)

        # Landing
        landing = Vector3(
            obstacle.x + direction.x * 0.8,
            start.y,
            obstacle.z + direction.z * 0.8,
        )
        points.append(landing)

        return points

    def update_vault(self, dt: float, vault_speed: float = 4.0) -> Vector3:
        """
        Update vault progress.

        Args:
            dt: Delta time
            vault_speed: Vault speed

        Returns:
            Current position during vault
        """
        if not self.is_vaulting:
            return self._character_position

        # Update progress
        self._vault_info.progress += vault_speed * dt
        self._vault_info.progress = min(1.0, self._vault_info.progress)

        # Interpolate along trajectory
        position = self._interpolate_trajectory(
            self._vault_info.trajectory,
            self._vault_info.progress,
        )

        # Check if complete
        if self._vault_info.progress >= 1.0:
            self._end_interaction()

        return position

    def _interpolate_trajectory(
        self,
        points: list[Vector3],
        t: float,
    ) -> Vector3:
        """Interpolate along a trajectory."""
        if not points:
            return Vector3.zero()

        if len(points) == 1:
            return points[0]

        # Scale t to segment count
        segments = len(points) - 1
        scaled_t = t * segments

        # Find segment
        segment = int(scaled_t)
        segment = min(segment, segments - 1)

        # Local t within segment
        local_t = scaled_t - segment

        # Smooth interpolation
        local_t = local_t * local_t * (3.0 - 2.0 * local_t)

        return Vector3.lerp(points[segment], points[segment + 1], local_t)

    # -------------------------------------------------------------------------
    # Interaction Management
    # -------------------------------------------------------------------------

    def _end_interaction(self) -> None:
        """End current interaction."""
        old_interaction = self._current_interaction

        self._current_interaction = InteractionType.NONE
        self._climb_info = ClimbInfo()
        self._vault_info = VaultInfo()

        if self._on_interaction_end:
            self._on_interaction_end(old_interaction)

    def cancel_interaction(self) -> None:
        """Cancel any current interaction."""
        if self._current_interaction == InteractionType.GRAB:
            self.release_grab()
        elif self._current_interaction == InteractionType.CARRY:
            self.release_grab()
        else:
            self._end_interaction()

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def find_grabbable_objects(
        self,
        max_results: int = 5,
    ) -> list[InteractionTarget]:
        """
        Find objects that can be grabbed.

        Args:
            max_results: Maximum number of results

        Returns:
            List of grabbable targets
        """
        # This would query the physics world for nearby objects
        # Placeholder implementation
        return []

    def find_climbable_surfaces(
        self,
        search_direction: Vector3,
    ) -> Optional[ClimbInfo]:
        """
        Find climbable surfaces in a direction.

        Args:
            search_direction: Direction to search

        Returns:
            Climb info if found
        """
        # Cast rays to find ledges
        # Placeholder implementation
        return None

    def find_vaultable_obstacles(
        self,
        search_direction: Vector3,
    ) -> Optional[VaultInfo]:
        """
        Find vaultable obstacles in a direction.

        Args:
            search_direction: Direction to search

        Returns:
            Vault info if found
        """
        # Cast rays to find obstacles
        # Placeholder implementation
        return None

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def get_debug_info(self) -> dict[str, Any]:
        """Get debug information."""
        return {
            "current_interaction": self._current_interaction.value,
            "is_grabbing": self.is_grabbing,
            "is_carrying": self.is_carrying,
            "is_climbing": self.is_climbing,
            "is_vaulting": self.is_vaulting,
            "grab_state": self._grab_info.state.value,
            "grab_target": (
                self._grab_info.target.entity_id
                if self._grab_info.target
                else None
            ),
            "climb_progress": self._climb_info.progress,
            "vault_progress": self._vault_info.progress,
        }
