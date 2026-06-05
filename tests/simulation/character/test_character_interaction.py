"""
Whitebox tests for engine/simulation/character/character_interaction.py

Tests CharacterInteractionManager, grabbing, climbing, vaulting, and pushing.
"""

import pytest
from engine.simulation.character.character_interaction import (
    CharacterInteractionManager,
    ClimbInfo,
    GrabInfo,
    GrabState,
    InteractionTarget,
    InteractionType,
    VaultInfo,
)
from engine.simulation.character.character_controller import (
    PhysicsWorldInterface,
    Vector3,
)


class TestInteractionType:
    """Tests for InteractionType enum."""

    def test_none_value(self):
        """NONE should have expected value."""
        assert InteractionType.NONE.value == "none"

    def test_push_value(self):
        """PUSH should have expected value."""
        assert InteractionType.PUSH.value == "push"

    def test_grab_value(self):
        """GRAB should have expected value."""
        assert InteractionType.GRAB.value == "grab"

    def test_carry_value(self):
        """CARRY should have expected value."""
        assert InteractionType.CARRY.value == "carry"

    def test_throw_value(self):
        """THROW should have expected value."""
        assert InteractionType.THROW.value == "throw"

    def test_climb_value(self):
        """CLIMB should have expected value."""
        assert InteractionType.CLIMB.value == "climb"

    def test_vault_value(self):
        """VAULT should have expected value."""
        assert InteractionType.VAULT.value == "vault"


class TestGrabState:
    """Tests for GrabState enum."""

    def test_none_value(self):
        """NONE should have expected value."""
        assert GrabState.NONE.value == "none"

    def test_reaching_value(self):
        """REACHING should have expected value."""
        assert GrabState.REACHING.value == "reaching"

    def test_holding_value(self):
        """HOLDING should have expected value."""
        assert GrabState.HOLDING.value == "holding"

    def test_releasing_value(self):
        """RELEASING should have expected value."""
        assert GrabState.RELEASING.value == "releasing"


class TestInteractionTarget:
    """Tests for InteractionTarget dataclass."""

    def test_default_construction(self):
        """Default InteractionTarget should have default values."""
        target = InteractionTarget()
        assert target.entity_id == 0
        assert target.body_id == 0
        assert target.mass == 1.0
        assert target.is_character is False
        assert target.can_be_grabbed is True
        assert target.can_be_carried is True

    def test_custom_construction(self):
        """InteractionTarget should accept custom values."""
        target = InteractionTarget(
            entity_id=42,
            body_id=123,
            position=Vector3(1.0, 2.0, 3.0),
            mass=50.0,
            is_character=True,
            can_be_grabbed=False,
        )
        assert target.entity_id == 42
        assert target.mass == 50.0
        assert target.is_character is True
        assert target.can_be_grabbed is False


class TestGrabInfo:
    """Tests for GrabInfo dataclass."""

    def test_default_construction(self):
        """Default GrabInfo should have default values."""
        info = GrabInfo()
        assert info.target is None
        assert info.hand == "right"
        assert info.state == GrabState.NONE
        assert info.hold_time == 0.0

    def test_custom_construction(self):
        """GrabInfo should accept custom values."""
        target = InteractionTarget(entity_id=1)
        info = GrabInfo(
            target=target,
            grab_point=Vector3(0.0, 1.0, 0.0),
            hand="both",
            state=GrabState.HOLDING,
            hold_time=2.5,
        )
        assert info.target is target
        assert info.hand == "both"
        assert info.state == GrabState.HOLDING


class TestClimbInfo:
    """Tests for ClimbInfo dataclass."""

    def test_default_construction(self):
        """Default ClimbInfo should have default values."""
        info = ClimbInfo()
        assert info.progress == 0.0
        assert info.height == 0.0

    def test_custom_construction(self):
        """ClimbInfo should accept custom values."""
        info = ClimbInfo(
            surface_normal=Vector3(0.0, 0.0, -1.0),
            surface_position=Vector3(0.0, 2.0, 1.0),
            climb_direction=Vector3.up(),
            progress=0.5,
            height=2.0,
        )
        assert info.progress == 0.5
        assert info.height == 2.0


class TestVaultInfo:
    """Tests for VaultInfo dataclass."""

    def test_default_construction(self):
        """Default VaultInfo should have default values."""
        info = VaultInfo()
        assert info.obstacle_height == 0.0
        assert info.progress == 0.0
        assert len(info.trajectory) == 0

    def test_custom_construction(self):
        """VaultInfo should accept custom values."""
        info = VaultInfo(
            obstacle_position=Vector3(2.0, 0.5, 0.0),
            obstacle_height=1.0,
            vault_direction=Vector3.forward(),
            progress=0.3,
            trajectory=[Vector3.zero(), Vector3(1.0, 1.0, 0.0)],
        )
        assert info.obstacle_height == 1.0
        assert len(info.trajectory) == 2


class MockInteractionPhysics(PhysicsWorldInterface):
    """Mock physics for interaction testing."""

    def __init__(self):
        self.applied_impulses = []

    def apply_impulse(self, body_id, impulse, point):
        self.applied_impulses.append({
            "body_id": body_id,
            "impulse": impulse,
            "point": point,
        })


class TestCharacterInteractionManager:
    """Tests for CharacterInteractionManager class."""

    def test_construction(self):
        """CharacterInteractionManager should be constructible."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        assert manager is not None

    def test_current_interaction_initial(self):
        """current_interaction should be NONE initially."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        assert manager.current_interaction == InteractionType.NONE

    def test_is_interacting_false(self):
        """is_interacting should be False initially."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        assert manager.is_interacting is False

    def test_is_grabbing_false(self):
        """is_grabbing should be False initially."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        assert manager.is_grabbing is False

    def test_is_carrying_false(self):
        """is_carrying should be False initially."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        assert manager.is_carrying is False

    def test_is_climbing_false(self):
        """is_climbing should be False initially."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        assert manager.is_climbing is False

    def test_is_vaulting_false(self):
        """is_vaulting should be False initially."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        assert manager.is_vaulting is False

    def test_grab_info_property(self):
        """grab_info should return current grab info."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        info = manager.grab_info
        assert isinstance(info, GrabInfo)

    def test_climb_info_property(self):
        """climb_info should return current climb info."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        info = manager.climb_info
        assert isinstance(info, ClimbInfo)


class TestCallbackSetup:
    """Tests for callback setup."""

    def test_set_interaction_callbacks(self):
        """set_interaction_callbacks should set callbacks."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        starts = []
        ends = []
        manager.set_interaction_callbacks(
            on_start=lambda t: starts.append(t),
            on_end=lambda t: ends.append(t),
        )
        # Callbacks tested through interaction usage

    def test_set_grab_callback(self):
        """set_grab_callback should set callback."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        grabs = []
        manager.set_grab_callback(lambda t: grabs.append(t))

    def test_set_throw_callback(self):
        """set_throw_callback should set callback."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        throws = []
        manager.set_throw_callback(lambda t, v: throws.append((t, v)))


class TestCharacterStateUpdate:
    """Tests for character state updates."""

    def test_update_character_state(self):
        """update_character_state should update internal state."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            position=Vector3(1.0, 0.0, 0.0),
            forward=Vector3(0.0, 0.0, 1.0),
            velocity=Vector3(2.0, 0.0, 0.0),
            body_id=42,
        )
        assert manager._character_position.x == 1.0
        assert manager._character_forward.z == 1.0
        assert manager._character_body_id == 42


class TestPushing:
    """Tests for pushing interactions."""

    def test_push_character(self):
        """push_character should apply impulse to target."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(
            body_id=123,
            position=Vector3(0.0, 0.0, 1.0),
            mass=70.0,
        )
        result = manager.push_character(target)
        assert result is True
        assert len(physics.applied_impulses) == 1
        assert physics.applied_impulses[0]["body_id"] == 123

    def test_push_character_custom_direction(self):
        """push_character should accept custom direction."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(body_id=123, mass=70.0)
        manager.push_character(target, push_direction=Vector3(1.0, 0.0, 0.0))
        assert physics.applied_impulses[0]["impulse"].x > 0

    def test_character_vs_character(self):
        """character_vs_character should calculate impulses."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager._character_mass = 70.0
        manager._character_velocity = Vector3(5.0, 0.0, 0.0)
        other = InteractionTarget(mass=70.0)
        collision_normal = Vector3(1.0, 0.0, 0.0)
        self_impulse, other_impulse = manager.character_vs_character(
            other, collision_normal
        )
        # Impulses should be opposite
        assert self_impulse.x * other_impulse.x <= 0


class TestGrabbing:
    """Tests for grab interactions."""

    def test_grab_object_too_far(self):
        """grab_object should fail if target too far."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(
            entity_id=1,
            position=Vector3(0.0, 0.0, 10.0),  # Far away
        )
        result = manager.grab_object(target)
        assert result is False

    def test_grab_object_not_grabbable(self):
        """grab_object should fail if target not grabbable."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(
            entity_id=1,
            position=Vector3(0.0, 0.0, 0.5),
            can_be_grabbed=False,
        )
        result = manager.grab_object(target)
        assert result is False

    def test_grab_object_success(self):
        """grab_object should succeed for valid target."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(
            entity_id=1,
            position=Vector3(0.0, 0.0, 0.5),
        )
        result = manager.grab_object(target)
        assert result is True
        assert manager.is_grabbing is True
        assert manager.current_interaction == InteractionType.GRAB

    def test_grab_object_callback(self):
        """grab_object should trigger callbacks."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        grabs = []
        starts = []
        manager.set_grab_callback(lambda t: grabs.append(t))
        manager.set_interaction_callbacks(on_start=lambda t: starts.append(t))
        target = InteractionTarget(entity_id=1, position=Vector3(0.0, 0.0, 0.5))
        manager.grab_object(target)
        assert len(grabs) == 1
        assert len(starts) == 1

    def test_release_grab_not_grabbing(self):
        """release_grab when not grabbing should return None."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        result = manager.release_grab()
        assert result is None

    def test_release_grab(self):
        """release_grab should release and return target."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(entity_id=1, position=Vector3(0.0, 0.0, 0.5))
        manager.grab_object(target)
        released = manager.release_grab()
        assert released is target
        assert manager.is_grabbing is False

    def test_confirm_grab_not_reaching(self):
        """confirm_grab should fail if not reaching."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        result = manager.confirm_grab()
        assert result is False

    def test_confirm_grab(self):
        """confirm_grab should transition to holding."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(entity_id=1, position=Vector3(0.0, 0.0, 0.5))
        manager.grab_object(target)
        result = manager.confirm_grab()
        assert result is True
        assert manager._grab_info.state == GrabState.HOLDING


class TestCarrying:
    """Tests for carry interactions."""

    def test_carry_object_too_heavy(self):
        """carry_object should fail if target too heavy."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(
            entity_id=1,
            position=Vector3(0.0, 0.0, 0.5),
            mass=100.0,  # Over limit
        )
        result = manager.carry_object(target)
        assert result is False

    def test_carry_object_not_carryable(self):
        """carry_object should fail if target not carryable."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        target = InteractionTarget(
            entity_id=1,
            position=Vector3(0.0, 0.0, 0.5),
            can_be_carried=False,
        )
        result = manager.carry_object(target)
        assert result is False

    def test_carry_object_success(self):
        """carry_object should succeed for valid target."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(
            entity_id=1,
            position=Vector3(0.0, 0.0, 0.5),
            mass=10.0,
        )
        # First grab, then confirm, then carry
        manager.grab_object(target)
        manager.confirm_grab()
        result = manager.carry_object(target)
        assert result is True
        assert manager.current_interaction == InteractionType.CARRY


class TestThrowing:
    """Tests for throw interactions."""

    def test_throw_object_not_holding(self):
        """throw_object should fail if not holding."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        result = manager.throw_object(Vector3.forward(), 10.0)
        assert result is False

    def test_throw_object_zero_direction(self):
        """throw_object should fail with zero direction."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(
            entity_id=1,
            body_id=42,
            position=Vector3(0.0, 0.0, 0.5),
            mass=5.0,
        )
        manager.carry_object(target)
        result = manager.throw_object(Vector3.zero(), 10.0)
        assert result is False

    def test_throw_object_success(self):
        """throw_object should succeed when holding."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(
            entity_id=1,
            body_id=42,
            position=Vector3(0.0, 0.0, 0.5),
            mass=5.0,
        )
        manager.carry_object(target)
        result = manager.throw_object(Vector3.forward(), 10.0)
        assert result is True
        assert manager.is_carrying is False
        assert len(physics.applied_impulses) == 1


class TestClimbing:
    """Tests for climb interactions."""

    def test_climb_ledge_too_high(self):
        """climb_ledge should fail if too high."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        result = manager.climb_ledge(
            ledge_position=Vector3(0.0, 5.0, 1.0),
            ledge_normal=Vector3(0.0, 0.0, -1.0),
            climb_height=5.0,  # Over limit
        )
        assert result is False

    def test_climb_ledge_already_interacting(self):
        """climb_ledge should fail if already interacting."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(entity_id=1, position=Vector3(0.0, 0.0, 0.5))
        manager.grab_object(target)
        result = manager.climb_ledge(
            Vector3(0.0, 2.0, 1.0),
            Vector3(0.0, 0.0, -1.0),
            1.5,
        )
        assert result is False

    def test_climb_ledge_success(self):
        """climb_ledge should succeed for valid ledge."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        result = manager.climb_ledge(
            ledge_position=Vector3(0.0, 2.0, 1.0),
            ledge_normal=Vector3(0.0, 0.0, -1.0),
            climb_height=1.5,
        )
        assert result is True
        assert manager.is_climbing is True
        assert manager.current_interaction == InteractionType.CLIMB

    def test_update_climb(self):
        """update_climb should progress the climb."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        manager.climb_ledge(
            Vector3(0.0, 2.0, 1.0),
            Vector3(0.0, 0.0, -1.0),
            1.5,
        )
        initial_progress = manager._climb_info.progress
        manager.update_climb(0.5)
        assert manager._climb_info.progress > initial_progress

    def test_update_climb_completes(self):
        """update_climb should complete when progress reaches 1."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        manager.climb_ledge(
            Vector3(0.0, 2.0, 1.0),
            Vector3(0.0, 0.0, -1.0),
            1.5,
        )
        # Update until complete
        for _ in range(20):
            manager.update_climb(0.1)
        assert manager.is_climbing is False

    def test_cancel_climb(self):
        """cancel_climb should end climb interaction."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        manager.climb_ledge(
            Vector3(0.0, 2.0, 1.0),
            Vector3(0.0, 0.0, -1.0),
            1.5,
        )
        manager.cancel_climb()
        assert manager.is_climbing is False


class TestVaulting:
    """Tests for vault interactions."""

    def test_vault_obstacle_too_high(self):
        """vault_obstacle should fail if too high."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        result = manager.vault_obstacle(
            obstacle_position=Vector3(1.0, 1.0, 0.0),
            obstacle_height=2.0,  # Over limit
            vault_direction=Vector3.forward(),
        )
        assert result is False

    def test_vault_obstacle_already_interacting(self):
        """vault_obstacle should fail if already interacting."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(entity_id=1, position=Vector3(0.0, 0.0, 0.5))
        manager.grab_object(target)
        result = manager.vault_obstacle(
            Vector3(1.0, 0.5, 0.0),
            0.8,
            Vector3.forward(),
        )
        assert result is False

    def test_vault_obstacle_success(self):
        """vault_obstacle should succeed for valid obstacle."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        result = manager.vault_obstacle(
            obstacle_position=Vector3(1.0, 0.5, 0.0),
            obstacle_height=0.8,
            vault_direction=Vector3.forward(),
        )
        assert result is True
        assert manager.is_vaulting is True
        assert manager.current_interaction == InteractionType.VAULT

    def test_update_vault(self):
        """update_vault should progress the vault."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        manager.vault_obstacle(
            Vector3(1.0, 0.5, 0.0),
            0.8,
            Vector3.forward(),
        )
        initial_progress = manager._vault_info.progress
        manager.update_vault(0.1)
        assert manager._vault_info.progress > initial_progress

    def test_update_vault_completes(self):
        """update_vault should complete when progress reaches 1."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        manager.vault_obstacle(
            Vector3(1.0, 0.5, 0.0),
            0.8,
            Vector3.forward(),
        )
        # Update until complete
        for _ in range(10):
            manager.update_vault(0.2, vault_speed=8.0)
        assert manager.is_vaulting is False

    def test_vault_trajectory_generated(self):
        """vault_obstacle should generate trajectory."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        manager.vault_obstacle(
            Vector3(1.0, 0.5, 0.0),
            0.8,
            Vector3.forward(),
        )
        assert len(manager._vault_info.trajectory) >= 3


class TestInteractionManagement:
    """Tests for interaction management."""

    def test_cancel_interaction_grab(self):
        """cancel_interaction should release grab."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(entity_id=1, position=Vector3(0.0, 0.0, 0.5))
        manager.grab_object(target)
        manager.cancel_interaction()
        assert manager.is_grabbing is False

    def test_cancel_interaction_climb(self):
        """cancel_interaction should end climb."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        manager.climb_ledge(Vector3(0.0, 2.0, 1.0), Vector3.forward(), 1.5)
        manager.cancel_interaction()
        assert manager.is_climbing is False


class TestQueries:
    """Tests for query methods."""

    def test_find_grabbable_objects(self):
        """find_grabbable_objects should return list."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        result = manager.find_grabbable_objects()
        assert isinstance(result, list)

    def test_find_climbable_surfaces(self):
        """find_climbable_surfaces should return optional ClimbInfo."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        result = manager.find_climbable_surfaces(Vector3.forward())
        # Placeholder returns None
        assert result is None

    def test_find_vaultable_obstacles(self):
        """find_vaultable_obstacles should return optional VaultInfo."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        result = manager.find_vaultable_obstacles(Vector3.forward())
        # Placeholder returns None
        assert result is None


class TestDebugInfo:
    """Tests for debug info."""

    def test_get_debug_info(self):
        """get_debug_info should return debug dictionary."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        info = manager.get_debug_info()
        assert "current_interaction" in info
        assert "is_grabbing" in info
        assert "is_carrying" in info
        assert "is_climbing" in info
        assert "is_vaulting" in info
        assert "grab_state" in info

    def test_get_debug_info_with_grab(self):
        """get_debug_info should include grab target when grabbing."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        manager.update_character_state(
            Vector3.zero(), Vector3.forward(), Vector3.zero()
        )
        target = InteractionTarget(entity_id=42, position=Vector3(0.0, 0.0, 0.5))
        manager.grab_object(target)
        info = manager.get_debug_info()
        assert info["grab_target"] == 42


class TestTrajectoryInterpolation:
    """Tests for trajectory interpolation."""

    def test_interpolate_trajectory_empty(self):
        """_interpolate_trajectory with empty should return zero."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        result = manager._interpolate_trajectory([], 0.5)
        assert result.x == 0.0

    def test_interpolate_trajectory_single(self):
        """_interpolate_trajectory with single point should return it."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        point = Vector3(1.0, 2.0, 3.0)
        result = manager._interpolate_trajectory([point], 0.5)
        assert result.x == 1.0
        assert result.y == 2.0
        assert result.z == 3.0

    def test_interpolate_trajectory_midpoint(self):
        """_interpolate_trajectory should interpolate between points."""
        physics = MockInteractionPhysics()
        manager = CharacterInteractionManager(physics)
        points = [
            Vector3(0.0, 0.0, 0.0),
            Vector3(2.0, 2.0, 0.0),
        ]
        result = manager._interpolate_trajectory(points, 0.5)
        # Should be around midpoint
        assert 0.5 < result.x < 1.5
        assert 0.5 < result.y < 1.5
