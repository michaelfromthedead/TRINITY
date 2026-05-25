"""
Character Controller Component.

Provides a component wrapper for character physics controllers,
integrating character movement, ground detection, and interaction
systems into the entity-component framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..character.character_controller import (
    CharacterController,
    CharacterControllerConfig,
    ControllerType,
    PhysicsWorldInterface,
    Quaternion,
    Vector3,
)
from ..character.ground_detection import GroundDetector, GroundInfo
from ..character.movement_modes import (
    MovementContext,
    MovementMode,
    MovementModeManager,
    MovementState,
)
from ..character.slope_handling import SlopeHandler
from ..character.platform_handling import PlatformHandler, PlatformProvider


@dataclass
class CharacterComponentConfig:
    """
    Configuration for character controller component.

    Attributes:
        radius: Capsule radius
        height: Capsule height
        step_height: Maximum step height
        slope_limit: Maximum walkable slope (degrees)
        controller_type: Type of controller
        mass: Character mass for interactions
        push_force_multiplier: Multiplier for push interactions
    """
    radius: float = 0.35
    height: float = 1.8
    step_height: float = 0.35
    slope_limit: float = 45.0
    controller_type: ControllerType = ControllerType.KINEMATIC
    mass: float = 70.0
    push_force_multiplier: float = 1.0
    enable_ground_snapping: bool = True
    enable_platform_handling: bool = True


class CharacterControllerComponent:
    """
    Component for character physics.

    Integrates:
    - Character controller for movement
    - Ground detection
    - Movement modes
    - Slope handling
    - Platform attachment

    This component provides a high-level interface for character
    physics in an entity-component system.
    """

    def __init__(
        self,
        entity_id: int,
        physics_world: PhysicsWorldInterface,
        platform_provider: Optional[PlatformProvider] = None,
        config: Optional[CharacterComponentConfig] = None,
    ):
        self._entity_id = entity_id
        self._config = config or CharacterComponentConfig()

        # Create controller configuration
        ctrl_config = CharacterControllerConfig(
            radius=self._config.radius,
            height=self._config.height,
            step_height=self._config.step_height,
            slope_limit=self._config.slope_limit,
            controller_type=self._config.controller_type,
        )

        # Initialize systems
        self._controller = CharacterController(physics_world, ctrl_config)
        self._ground_detector = GroundDetector(
            physics_world, self._config.radius, self._config.height
        )
        self._movement_manager = MovementModeManager()
        self._slope_handler = SlopeHandler(
            physics_world, self._config.slope_limit, self._config.step_height
        )
        self._platform_handler: Optional[PlatformHandler] = None
        if platform_provider and self._config.enable_platform_handling:
            self._platform_handler = PlatformHandler(platform_provider)

        # State
        self._enabled = True
        self._input_direction = Vector3.zero()
        self._look_direction = Vector3.forward()
        self._ground_info = GroundInfo()
        self._wants_jump = False
        self._last_velocity = Vector3.zero()

        # Callbacks
        self._on_landed: Optional[Callable[[], None]] = None
        self._on_jump: Optional[Callable[[], None]] = None
        self._on_fell: Optional[Callable[[float], None]] = None

        # Tracking
        self._was_grounded = False
        self._fall_start_height: Optional[float] = None
        self._time_in_air = 0.0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def entity_id(self) -> int:
        """Entity this component belongs to."""
        return self._entity_id

    @property
    def controller(self) -> CharacterController:
        """Underlying character controller."""
        return self._controller

    @property
    def position(self) -> Vector3:
        """Current world position."""
        return self._controller.position

    @position.setter
    def position(self, value: Vector3) -> None:
        self._controller.position = value

    @property
    def rotation(self) -> Quaternion:
        """Current rotation."""
        return self._controller.rotation

    @rotation.setter
    def rotation(self, value: Quaternion) -> None:
        self._controller.rotation = value

    @property
    def velocity(self) -> Vector3:
        """Current velocity."""
        return self._controller.velocity

    @property
    def is_grounded(self) -> bool:
        """Whether character is on ground."""
        return self._controller.is_grounded

    @property
    def ground_info(self) -> GroundInfo:
        """Detailed ground information."""
        return self._ground_info

    @property
    def movement_mode(self) -> MovementMode:
        """Current movement mode."""
        return self._movement_manager.current_mode

    @property
    def movement_state(self) -> MovementState:
        """Current movement state."""
        return self._movement_manager.state

    @property
    def can_jump(self) -> bool:
        """Whether character can currently jump."""
        if not self._movement_manager.can_jump:
            return False
        return self._ground_detector.can_jump(self._controller.is_grounded)

    @property
    def enabled(self) -> bool:
        """Whether component is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_landed_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for landing events."""
        self._on_landed = callback

    def set_jump_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for jump events."""
        self._on_jump = callback

    def set_fell_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        """Set callback for fall events (receives fall height)."""
        self._on_fell = callback

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------

    def set_input(
        self,
        direction: Vector3,
        look_direction: Optional[Vector3] = None,
    ) -> None:
        """
        Set movement input.

        Args:
            direction: Movement direction (can be non-normalized for speed control)
            look_direction: Direction character is facing
        """
        self._input_direction = direction
        if look_direction is not None:
            self._look_direction = look_direction.normalized()

    def jump(self) -> bool:
        """
        Request a jump.

        Returns:
            True if jump was initiated
        """
        if not self.can_jump:
            # Buffer the input
            self._ground_detector.register_jump_input()
            return False

        if self._controller.jump():
            self._wants_jump = False
            self._ground_detector.clear_jump_buffer()

            if self._on_jump:
                self._on_jump()

            return True

        return False

    # -------------------------------------------------------------------------
    # Movement Mode
    # -------------------------------------------------------------------------

    def set_movement_mode(self, mode: MovementMode, force: bool = False) -> bool:
        """
        Change movement mode.

        Args:
            mode: Target movement mode
            force: Force transition ignoring rules

        Returns:
            True if transition was started
        """
        return self._movement_manager.transition_to_mode(mode, force)

    def start_sprinting(self) -> bool:
        """Start sprinting."""
        return self.set_movement_mode(MovementMode.SPRINTING)

    def stop_sprinting(self) -> None:
        """Stop sprinting, return to running or walking."""
        if self._movement_manager.current_mode == MovementMode.SPRINTING:
            self.set_movement_mode(MovementMode.RUNNING)

    def crouch(self) -> bool:
        """Start crouching."""
        success = self.set_movement_mode(MovementMode.CROUCHING)
        if success:
            # Resize capsule
            self._controller.resize(
                self._config.height * self._movement_manager.height_modifier
            )
        return success

    def stand_up(self) -> bool:
        """Stand up from crouching."""
        if self._movement_manager.current_mode in (
            MovementMode.CROUCHING, MovementMode.PRONE
        ):
            # Check if we can stand up
            success = self._controller.resize(self._config.height)
            if success:
                self.set_movement_mode(MovementMode.WALKING)
            return success
        return True

    def go_prone(self) -> bool:
        """Go prone."""
        success = self.set_movement_mode(MovementMode.PRONE)
        if success:
            self._controller.resize(
                self._config.height * self._movement_manager.height_modifier
            )
        return success

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------

    def update(self, dt: float) -> Vector3:
        """
        Update character physics.

        Args:
            dt: Delta time in seconds

        Returns:
            Actual displacement
        """
        if not self._enabled or dt <= 0:
            return Vector3.zero()

        # Store previous state
        self._was_grounded = self._controller.is_grounded

        # Update ground detection
        self._ground_info = self._ground_detector.detect_ground(self._controller.position)

        # Update movement context
        self._update_movement_context()

        # Handle buffered jump
        if self._ground_detector.is_jump_buffered() and self.can_jump:
            self.jump()

        # Calculate movement from movement mode
        movement = self._movement_manager.apply_movement(
            self._input_direction,
            dt,
            self._controller.is_grounded,
        )

        # Apply platform velocity
        if self._platform_handler and self._platform_handler.is_attached:
            platform_vel = self._platform_handler.get_platform_velocity()
            self._controller.set_external_velocity(platform_vel)

        # Move character
        displacement = self._controller.move(movement / dt, dt)

        # Update platform attachment
        if self._platform_handler:
            self._update_platform_attachment()

        # Handle ground snapping
        if (
            self._config.enable_ground_snapping and
            self._slope_handler.should_step_down(
                self._controller.velocity,
                self._controller.is_grounded,
                self._was_grounded,
            )
        ):
            step_info = self._slope_handler.step_down(
                self._controller.position,
                self._config.radius,
                self._config.height,
            )
            if step_info and step_info.can_step:
                self._controller.position = step_info.landing_position

        # Handle landing/falling events
        self._handle_ground_state_change(dt)

        self._last_velocity = self._controller.velocity

        return displacement

    def _update_movement_context(self) -> None:
        """Update movement context based on environment."""
        if self._ground_info.ground_type.value == "water":
            self._movement_manager.set_context(MovementContext.WATER)
        elif self._controller.is_grounded:
            self._movement_manager.set_context(MovementContext.GROUND)
        else:
            self._movement_manager.set_context(MovementContext.AIR)

    def _update_platform_attachment(self) -> None:
        """Update moving platform attachment."""
        if self._platform_handler is None:
            return

        # Check for platform beneath us
        if self._controller.is_grounded and self._ground_info.collider_id != 0:
            if not self._platform_handler.is_attached:
                # Try to attach
                self._platform_handler.attach_to_platform(
                    self._ground_info.collider_id,
                    self._controller.position,
                    self._controller.rotation,
                )
        elif not self._controller.is_grounded and self._platform_handler.is_attached:
            # Detach when leaving ground
            exit_vel = self._platform_handler.detach_from_platform()
            self._controller.velocity = self._controller.velocity + exit_vel

    def _handle_ground_state_change(self, dt: float) -> None:
        """Handle landing and falling events."""
        is_grounded = self._controller.is_grounded

        if is_grounded and not self._was_grounded:
            # Just landed
            if self._fall_start_height is not None:
                fall_height = self._fall_start_height - self._controller.position.y
                if fall_height > 0.5 and self._on_fell:
                    self._on_fell(fall_height)

            self._fall_start_height = None
            self._time_in_air = 0.0

            if self._on_landed:
                self._on_landed()

            # Transition out of falling mode
            if self._movement_manager.current_mode == MovementMode.FALLING:
                self.set_movement_mode(MovementMode.WALKING)

        elif not is_grounded and self._was_grounded:
            # Just left ground
            self._fall_start_height = self._controller.position.y
            self._time_in_air = 0.0

            # Transition to falling mode
            if self._movement_manager.current_mode not in (
                MovementMode.FLYING, MovementMode.SWIMMING
            ):
                self.set_movement_mode(MovementMode.FALLING, force=True)

        elif not is_grounded:
            # Continuing to fall
            self._time_in_air += dt

    # -------------------------------------------------------------------------
    # Teleport
    # -------------------------------------------------------------------------

    def teleport(self, position: Vector3, rotation: Optional[Quaternion] = None) -> None:
        """Teleport character to new position."""
        self._controller.teleport(position, rotation)
        if self._platform_handler and self._platform_handler.is_attached:
            self._platform_handler.detach_from_platform(preserve_velocity=False)

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_forward(self) -> Vector3:
        """Get forward direction."""
        return self._look_direction.horizontal().normalized()

    def get_right(self) -> Vector3:
        """Get right direction."""
        forward = self.get_forward()
        return Vector3(forward.z, 0.0, -forward.x)

    def is_moving(self) -> bool:
        """Check if character is moving."""
        return self._movement_manager.is_moving()

    def is_falling(self) -> bool:
        """Check if character is falling."""
        return not self._controller.is_grounded and self._controller.velocity.y < 0

    def get_time_in_air(self) -> float:
        """Get time spent in air."""
        return self._time_in_air

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def cleanup(self) -> None:
        """Cleanup component."""
        if self._platform_handler and self._platform_handler.is_attached:
            self._platform_handler.detach_from_platform(preserve_velocity=False)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return {
            "entity_id": self._entity_id,
            "position": (
                self._controller.position.x,
                self._controller.position.y,
                self._controller.position.z,
            ),
            "velocity": (
                self._controller.velocity.x,
                self._controller.velocity.y,
                self._controller.velocity.z,
            ),
            "is_grounded": self._controller.is_grounded,
            "movement_mode": self._movement_manager.current_mode.name,
            "movement_state": self._movement_manager.get_state_dict(),
            "enabled": self._enabled,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load from serialized state."""
        pos = state.get("position", (0, 0, 0))
        self._controller.position = Vector3(pos[0], pos[1], pos[2])

        vel = state.get("velocity", (0, 0, 0))
        self._controller.velocity = Vector3(vel[0], vel[1], vel[2])

        self._enabled = state.get("enabled", True)

        mode_name = state.get("movement_mode", "WALKING")
        try:
            mode = MovementMode[mode_name]
            self._movement_manager.transition_to_mode(mode, force=True)
        except KeyError:
            pass

        movement_state = state.get("movement_state")
        if movement_state:
            self._movement_manager.load_state_dict(movement_state)


__all__ = [
    "CharacterComponentConfig",
    "CharacterControllerComponent",
]
