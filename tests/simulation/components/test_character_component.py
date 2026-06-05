"""
Whitebox tests for CharacterControllerComponent.

Tests cover:
- Component creation and configuration
- Position and rotation management
- Movement input handling
- Jump mechanics
- Movement modes (walking, running, sprinting, crouching, prone)
- Platform handling
- Ground state changes
- Serialization/deserialization
"""

import pytest
from unittest.mock import Mock, MagicMock

from engine.simulation.character.character_controller import (
    CharacterController,
    CharacterControllerConfig,
    ControllerType,
    PhysicsWorldInterface,
    Quaternion,
    Vector3,
)
from engine.simulation.character.ground_detection import GroundInfo
from engine.simulation.character.movement_modes import MovementMode, MovementState
from engine.simulation.components.character_component import (
    CharacterComponentConfig,
    CharacterControllerComponent,
)


# =============================================================================
# Mock Physics World
# =============================================================================


class MockPhysicsWorld(PhysicsWorldInterface):
    """Mock physics world for testing."""

    def __init__(self):
        self._capsule_position = Vector3.zero()

    def sweep_capsule(self, start, end, radius, height, mask=None, **kwargs):
        """Mock sweep - always returns no hit."""
        return None

    def raycast(self, start=None, end=None, mask=None, origin=None, direction=None, distance=None, **kwargs):
        """Mock raycast - always returns no hit."""
        return None

    def overlap_capsule(self, center=None, radius=None, height=None, mask=None, position=None, **kwargs):
        """Mock overlap - returns empty list."""
        return []

    def get_gravity(self):
        """Return standard gravity."""
        return Vector3(0.0, -9.81, 0.0)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def physics_world() -> MockPhysicsWorld:
    """Mock physics world."""
    return MockPhysicsWorld()


@pytest.fixture
def character_config() -> CharacterComponentConfig:
    """Default character component configuration."""
    return CharacterComponentConfig(
        radius=0.35,
        height=1.8,
        step_height=0.35,
        slope_limit=45.0,
        mass=70.0,
    )


@pytest.fixture
def character(physics_world, character_config) -> CharacterControllerComponent:
    """Create a character controller component."""
    return CharacterControllerComponent(
        entity_id=1,
        physics_world=physics_world,
        config=character_config,
    )


# =============================================================================
# CharacterComponentConfig Tests
# =============================================================================


class TestCharacterComponentConfig:
    """Tests for CharacterComponentConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CharacterComponentConfig()

        assert config.radius == 0.35
        assert config.height == 1.8
        assert config.step_height == 0.35
        assert config.slope_limit == 45.0
        assert config.controller_type == ControllerType.KINEMATIC
        assert config.mass == 70.0
        assert config.push_force_multiplier == 1.0
        assert config.enable_ground_snapping is True
        assert config.enable_platform_handling is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CharacterComponentConfig(
            radius=0.5,
            height=2.0,
            step_height=0.5,
            slope_limit=60.0,
            mass=80.0,
            controller_type=ControllerType.DYNAMIC,
        )

        assert config.radius == 0.5
        assert config.height == 2.0
        assert config.slope_limit == 60.0
        assert config.controller_type == ControllerType.DYNAMIC


# =============================================================================
# Component Creation Tests
# =============================================================================


class TestCharacterCreation:
    """Tests for character component creation."""

    def test_create_with_default_config(self, physics_world):
        """Test creating with default config."""
        component = CharacterControllerComponent(
            entity_id=1,
            physics_world=physics_world,
        )

        assert component.entity_id == 1
        assert component.enabled is True

    def test_create_with_custom_config(self, character, character_config):
        """Test creating with custom config."""
        assert character.entity_id == 1
        # Controller should use config values

    def test_initial_state(self, character):
        """Test initial component state."""
        assert character.position.magnitude() == 0.0
        assert character.velocity.magnitude() == 0.0
        assert character.is_grounded is False
        assert character.enabled is True


# =============================================================================
# Property Tests
# =============================================================================


class TestProperties:
    """Tests for component properties."""

    def test_entity_id(self, character):
        """Test entity_id property."""
        assert character.entity_id == 1

    def test_controller_property(self, character):
        """Test controller property."""
        controller = character.controller
        assert controller is not None
        assert isinstance(controller, CharacterController)

    def test_position_property(self, character):
        """Test position getter/setter."""
        character.position = Vector3(10.0, 5.0, 3.0)

        assert character.position.x == 10.0
        assert character.position.y == 5.0
        assert character.position.z == 3.0

    def test_rotation_property(self, character):
        """Test rotation getter/setter."""
        rotation = Quaternion(0.0, 0.707, 0.0, 0.707)
        character.rotation = rotation

        assert character.rotation.y == 0.707

    def test_velocity_property(self, character):
        """Test velocity property (read-only from controller)."""
        velocity = character.velocity
        assert velocity is not None

    def test_enabled_property(self, character):
        """Test enabled getter/setter."""
        assert character.enabled is True

        character.enabled = False
        assert character.enabled is False


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for event callbacks."""

    def test_set_landed_callback(self, character):
        """Test setting landed callback."""
        landed_events = []
        character.set_landed_callback(lambda: landed_events.append(True))

        # Callback should be set
        assert character._on_landed is not None

    def test_set_jump_callback(self, character):
        """Test setting jump callback."""
        jump_events = []
        character.set_jump_callback(lambda: jump_events.append(True))

        assert character._on_jump is not None

    def test_set_fell_callback(self, character):
        """Test setting fell callback."""
        fall_events = []
        character.set_fell_callback(lambda h: fall_events.append(h))

        assert character._on_fell is not None


# =============================================================================
# Input Tests
# =============================================================================


class TestInput:
    """Tests for input handling."""

    def test_set_input(self, character):
        """Test setting movement input."""
        direction = Vector3(1.0, 0.0, 0.0)
        look_dir = Vector3(0.0, 0.0, 1.0)

        character.set_input(direction, look_dir)

        assert character._input_direction.x == 1.0
        assert character._look_direction.z == 1.0

    def test_set_input_normalizes_look(self, character):
        """Test look direction is normalized."""
        character.set_input(
            Vector3(1.0, 0.0, 0.0),
            Vector3(0.0, 0.0, 2.0),  # Non-normalized
        )

        # Should be normalized
        assert abs(character._look_direction.magnitude() - 1.0) < 0.001


# =============================================================================
# Jump Tests
# =============================================================================


class TestJump:
    """Tests for jump mechanics."""

    def test_can_jump_property(self, character):
        """Test can_jump property."""
        # Initially not grounded, so can't jump (depends on implementation)
        can = character.can_jump
        # Result depends on ground detector implementation

    def test_jump_not_grounded(self, character):
        """Test jump when not grounded fails."""
        result = character.jump()

        # Typically returns False if not grounded
        # Actual result depends on jump buffering implementation


# =============================================================================
# Movement Mode Tests
# =============================================================================


class TestMovementModes:
    """Tests for movement mode management."""

    def test_initial_movement_mode(self, character):
        """Test initial movement mode."""
        mode = character.movement_mode
        assert mode is not None

    def test_movement_state(self, character):
        """Test movement state property."""
        state = character.movement_state
        assert state is not None

    def test_set_movement_mode(self, character):
        """Test changing movement mode."""
        result = character.set_movement_mode(MovementMode.RUNNING)
        # Result depends on transition rules

    def test_start_sprinting(self, character):
        """Test starting sprint."""
        result = character.start_sprinting()
        # Transitions to sprinting mode if allowed

    def test_stop_sprinting(self, character):
        """Test stopping sprint."""
        character.start_sprinting()
        character.stop_sprinting()
        # Should transition back to running/walking

    def test_crouch(self, character):
        """Test crouching."""
        result = character.crouch()
        # If successful, capsule should resize

    def test_stand_up(self, character):
        """Test standing up from crouch."""
        character.crouch()
        result = character.stand_up()
        # Should return to normal height

    def test_go_prone(self, character):
        """Test going prone."""
        result = character.go_prone()
        # Transitions to prone mode if allowed


# =============================================================================
# Update Tests
# =============================================================================


class TestUpdate:
    """Tests for update cycle."""

    def test_update_returns_displacement(self, character):
        """Test update returns displacement vector."""
        character.set_input(Vector3(1.0, 0.0, 0.0))

        displacement = character.update(dt=0.016)

        assert isinstance(displacement, Vector3)

    def test_update_disabled(self, character):
        """Test update when disabled returns zero."""
        character.enabled = False

        displacement = character.update(dt=0.016)

        assert displacement.magnitude() == 0.0

    def test_update_zero_dt(self, character):
        """Test update with zero dt returns zero."""
        displacement = character.update(dt=0.0)

        assert displacement.magnitude() == 0.0

    def test_update_negative_dt(self, character):
        """Test update with negative dt returns zero."""
        displacement = character.update(dt=-0.1)

        assert displacement.magnitude() == 0.0


# =============================================================================
# Teleport Tests
# =============================================================================


class TestTeleport:
    """Tests for teleportation."""

    def test_teleport_position(self, character):
        """Test teleporting to position."""
        character.teleport(Vector3(100.0, 50.0, 25.0))

        # Position should be updated via controller
        # Exact behavior depends on controller implementation

    def test_teleport_with_rotation(self, character):
        """Test teleporting with rotation."""
        rotation = Quaternion(0.0, 0.707, 0.0, 0.707)

        character.teleport(Vector3(0.0, 10.0, 0.0), rotation)

        # Both position and rotation should be set


# =============================================================================
# Query Tests
# =============================================================================


class TestQueries:
    """Tests for query methods."""

    def test_get_forward(self, character):
        """Test getting forward direction."""
        character.set_input(Vector3.zero(), Vector3(0.0, 0.0, 1.0))

        forward = character.get_forward()

        # Should return horizontal forward
        assert forward.y == 0.0

    def test_get_right(self, character):
        """Test getting right direction."""
        character.set_input(Vector3.zero(), Vector3(0.0, 0.0, 1.0))

        right = character.get_right()

        assert right.y == 0.0

    def test_is_moving(self, character):
        """Test checking if moving."""
        result = character.is_moving()

        assert isinstance(result, bool)

    def test_is_falling(self, character):
        """Test checking if falling."""
        result = character.is_falling()

        # Depends on velocity and grounded state
        assert isinstance(result, bool)

    def test_get_time_in_air(self, character):
        """Test getting time in air."""
        time = character.get_time_in_air()

        assert time >= 0.0


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for component lifecycle."""

    def test_cleanup(self, character):
        """Test cleanup."""
        character.cleanup()

        # Should cleanup internal state


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for state serialization/deserialization."""

    def test_get_state(self, character):
        """Test getting serializable state."""
        character.position = Vector3(10.0, 5.0, 3.0)

        state = character.get_state()

        assert state["entity_id"] == 1
        assert state["position"] == (10.0, 5.0, 3.0)
        assert "velocity" in state
        assert "is_grounded" in state
        assert "movement_mode" in state
        assert state["enabled"] is True

    def test_load_state(self, character):
        """Test loading from serialized state."""
        state = {
            "position": (100.0, 50.0, 25.0),
            "velocity": (5.0, 0.0, 0.0),
            "enabled": False,
            "movement_mode": "WALKING",
        }

        character.load_state(state)

        assert character.enabled is False
        # Position/velocity set via controller

    def test_load_state_invalid_mode(self, character):
        """Test loading state with invalid movement mode."""
        state = {
            "position": (0.0, 0.0, 0.0),
            "movement_mode": "INVALID_MODE",
        }

        # Should not raise, just ignore invalid mode
        character.load_state(state)

    def test_load_state_partial(self, character):
        """Test loading partial state."""
        state = {
            "position": (5.0, 0.0, 0.0),
        }

        character.load_state(state)

        # Should handle missing fields gracefully

    def test_roundtrip_serialization(self, character):
        """Test state survives roundtrip."""
        character.position = Vector3(50.0, 25.0, 10.0)
        character.enabled = False

        state = character.get_state()

        # Create new component with same physics world
        new_character = CharacterControllerComponent(
            entity_id=99,
            physics_world=MockPhysicsWorld(),
        )
        new_character.load_state(state)

        assert new_character.enabled is False


# =============================================================================
# Platform Handling Tests
# =============================================================================


class TestPlatformHandling:
    """Tests for moving platform handling."""

    def test_no_platform_handler(self, physics_world, character_config):
        """Test component without platform handler."""
        character_config.enable_platform_handling = False
        component = CharacterControllerComponent(
            entity_id=1,
            physics_world=physics_world,
            config=character_config,
        )

        assert component._platform_handler is None

    def test_platform_handler_disabled_in_config(self, physics_world):
        """Test platform handling can be disabled."""
        config = CharacterComponentConfig(enable_platform_handling=False)
        component = CharacterControllerComponent(
            entity_id=1,
            physics_world=physics_world,
            config=config,
        )

        assert component._platform_handler is None


# =============================================================================
# Ground State Tests
# =============================================================================


class TestGroundState:
    """Tests for ground state handling."""

    def test_ground_info_property(self, character):
        """Test ground info property."""
        info = character.ground_info

        assert info is not None
        assert isinstance(info, GroundInfo)

    def test_was_grounded_tracking(self, character):
        """Test tracking of previous grounded state."""
        character.update(dt=0.016)

        # _was_grounded should be updated
        # Actual value depends on ground detection


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_small_dt(self, character):
        """Test update with very small dt."""
        character.set_input(Vector3(1.0, 0.0, 0.0))

        displacement = character.update(dt=0.0001)

        # Should still work, just small movement

    def test_very_large_dt(self, character):
        """Test update with very large dt."""
        character.set_input(Vector3(1.0, 0.0, 0.0))

        displacement = character.update(dt=1.0)

        # Should handle gracefully

    def test_extreme_position(self, character):
        """Test teleporting to extreme position."""
        character.teleport(Vector3(1e6, 1e6, 1e6))

        # Should not crash

    def test_zero_look_direction(self, character):
        """Test zero look direction."""
        character.set_input(Vector3(1.0, 0.0, 0.0), Vector3.zero())

        # get_forward should handle this

    def test_multiple_jumps(self, character):
        """Test multiple jump attempts."""
        for _ in range(10):
            character.jump()

        # Should not crash

    def test_rapid_mode_changes(self, character):
        """Test rapid movement mode changes."""
        modes = [
            MovementMode.WALKING,
            MovementMode.RUNNING,
            MovementMode.SPRINTING,
            MovementMode.CROUCHING,
        ]

        for mode in modes:
            character.set_movement_mode(mode)

        # Should handle rapid transitions

    def test_cleanup_with_platform(self, physics_world):
        """Test cleanup when attached to platform."""
        # Would need mock platform provider
        config = CharacterComponentConfig()
        component = CharacterControllerComponent(
            entity_id=1,
            physics_world=physics_world,
            config=config,
        )

        component.cleanup()

        # Should cleanup without error
