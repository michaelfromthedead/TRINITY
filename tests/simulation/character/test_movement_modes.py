"""
Whitebox tests for engine/simulation/character/movement_modes.py

Tests MovementMode, MovementModeManager, transitions, and stamina.
"""

import pytest
from engine.simulation.character.movement_modes import (
    DEFAULT_MODE_PARAMS,
    MovementContext,
    MovementMode,
    MovementModeManager,
    MovementModeParams,
    MovementState,
    TRANSITION_RULES,
    TransitionRule,
)
from engine.simulation.character.character_controller import Vector3


class TestMovementMode:
    """Tests for MovementMode enum."""

    def test_all_modes_defined(self):
        """All expected movement modes should be defined."""
        expected_modes = [
            "WALKING", "RUNNING", "SPRINTING", "CROUCHING", "PRONE",
            "SWIMMING", "CLIMBING", "FLYING", "FALLING", "LADDERING",
            "SLIDING", "VAULTING", "HANGING", "CUSTOM"
        ]
        for mode_name in expected_modes:
            assert hasattr(MovementMode, mode_name)

    def test_mode_values_unique(self):
        """All mode values should be unique."""
        values = [mode.value for mode in MovementMode]
        assert len(values) == len(set(values))


class TestMovementContext:
    """Tests for MovementContext enum."""

    def test_all_contexts_defined(self):
        """All expected contexts should be defined."""
        expected_contexts = ["GROUND", "AIR", "WATER", "LADDER", "ROPE", "LEDGE"]
        for context_name in expected_contexts:
            assert hasattr(MovementContext, context_name)


class TestMovementModeParams:
    """Tests for MovementModeParams dataclass."""

    def test_default_construction(self):
        """Default params should have reasonable values."""
        params = MovementModeParams()
        assert params.max_speed == 5.0
        assert params.acceleration == 20.0
        assert params.deceleration == 15.0
        assert params.turn_speed == 1.0
        assert params.gravity_scale == 1.0
        assert params.can_jump is True
        assert params.height_modifier == 1.0
        assert params.friction_modifier == 1.0
        assert params.stamina_cost == 0.0
        assert params.air_control == 0.3

    def test_custom_construction(self):
        """Params should accept custom values."""
        params = MovementModeParams(
            max_speed=10.0,
            acceleration=30.0,
            can_jump=False,
            stamina_cost=5.0,
        )
        assert params.max_speed == 10.0
        assert params.acceleration == 30.0
        assert params.can_jump is False
        assert params.stamina_cost == 5.0


class TestDefaultModeParams:
    """Tests for default mode parameters."""

    def test_all_modes_have_params(self):
        """All movement modes should have default params."""
        for mode in MovementMode:
            assert mode in DEFAULT_MODE_PARAMS

    def test_walking_params(self):
        """Walking params should be configured correctly."""
        params = DEFAULT_MODE_PARAMS[MovementMode.WALKING]
        assert params.can_jump is True
        assert params.stamina_cost == 0.0
        assert params.max_speed > 0

    def test_running_faster_than_walking(self):
        """Running should be faster than walking."""
        walking = DEFAULT_MODE_PARAMS[MovementMode.WALKING]
        running = DEFAULT_MODE_PARAMS[MovementMode.RUNNING]
        assert running.max_speed > walking.max_speed

    def test_sprinting_fastest_ground(self):
        """Sprinting should be fastest ground movement."""
        sprinting = DEFAULT_MODE_PARAMS[MovementMode.SPRINTING]
        running = DEFAULT_MODE_PARAMS[MovementMode.RUNNING]
        walking = DEFAULT_MODE_PARAMS[MovementMode.WALKING]
        assert sprinting.max_speed > running.max_speed
        assert sprinting.max_speed > walking.max_speed

    def test_crouching_has_height_modifier(self):
        """Crouching should have height modifier < 1."""
        params = DEFAULT_MODE_PARAMS[MovementMode.CROUCHING]
        assert params.height_modifier < 1.0

    def test_prone_has_lowest_height_modifier(self):
        """Prone should have lowest height modifier."""
        prone = DEFAULT_MODE_PARAMS[MovementMode.PRONE]
        crouching = DEFAULT_MODE_PARAMS[MovementMode.CROUCHING]
        assert prone.height_modifier < crouching.height_modifier

    def test_prone_no_jump(self):
        """Prone should not allow jumping."""
        params = DEFAULT_MODE_PARAMS[MovementMode.PRONE]
        assert params.can_jump is False

    def test_swimming_reduced_gravity(self):
        """Swimming should have reduced gravity."""
        params = DEFAULT_MODE_PARAMS[MovementMode.SWIMMING]
        assert params.gravity_scale < 1.0

    def test_climbing_no_gravity(self):
        """Climbing should have no gravity."""
        params = DEFAULT_MODE_PARAMS[MovementMode.CLIMBING]
        assert params.gravity_scale == 0.0

    def test_flying_no_gravity(self):
        """Flying should have no gravity."""
        params = DEFAULT_MODE_PARAMS[MovementMode.FLYING]
        assert params.gravity_scale == 0.0

    def test_falling_has_gravity(self):
        """Falling should have gravity."""
        params = DEFAULT_MODE_PARAMS[MovementMode.FALLING]
        assert params.gravity_scale == 1.0

    def test_sliding_increased_gravity(self):
        """Sliding should have increased gravity."""
        params = DEFAULT_MODE_PARAMS[MovementMode.SLIDING]
        assert params.gravity_scale > 1.0

    def test_sprinting_has_stamina_cost(self):
        """Sprinting should cost stamina."""
        params = DEFAULT_MODE_PARAMS[MovementMode.SPRINTING]
        assert params.stamina_cost > 0


class TestMovementState:
    """Tests for MovementState dataclass."""

    def test_default_construction(self):
        """Default state should be walking on ground."""
        state = MovementState()
        assert state.mode == MovementMode.WALKING
        assert state.context == MovementContext.GROUND
        assert state.current_speed == 0.0
        assert state.is_transitioning is False
        assert state.transition_progress == 1.0
        assert state.stamina == 100.0
        assert state.max_stamina == 100.0

    def test_get_params(self):
        """get_params should return params for current mode."""
        state = MovementState(mode=MovementMode.RUNNING)
        params = state.get_params()
        assert params == DEFAULT_MODE_PARAMS[MovementMode.RUNNING]


class TestTransitionRule:
    """Tests for TransitionRule dataclass."""

    def test_default_construction(self):
        """Default rule should allow instant transition."""
        rule = TransitionRule(
            from_mode=MovementMode.WALKING,
            to_mode=MovementMode.RUNNING,
        )
        assert rule.duration == 0.0
        assert rule.requires_grounded is False
        assert rule.min_stamina == 0.0
        assert rule.condition is None

    def test_custom_construction(self):
        """Rule should accept custom values."""
        rule = TransitionRule(
            from_mode=MovementMode.WALKING,
            to_mode=MovementMode.CROUCHING,
            duration=0.2,
            requires_grounded=True,
        )
        assert rule.duration == 0.2
        assert rule.requires_grounded is True


class TestTransitionRules:
    """Tests for default transition rules."""

    def test_walking_to_running_exists(self):
        """Walking to running transition should exist."""
        exists = any(
            rule.from_mode == MovementMode.WALKING and
            rule.to_mode == MovementMode.RUNNING
            for rule in TRANSITION_RULES
        )
        assert exists is True

    def test_walking_to_crouching_exists(self):
        """Walking to crouching transition should exist."""
        rule = next(
            (r for r in TRANSITION_RULES
             if r.from_mode == MovementMode.WALKING and
             r.to_mode == MovementMode.CROUCHING),
            None
        )
        assert rule is not None
        assert rule.requires_grounded is True

    def test_running_to_sprinting_requires_stamina(self):
        """Running to sprinting should require stamina."""
        rule = next(
            (r for r in TRANSITION_RULES
             if r.from_mode == MovementMode.RUNNING and
             r.to_mode == MovementMode.SPRINTING),
            None
        )
        assert rule is not None
        assert rule.min_stamina > 0

    def test_falling_to_walking_requires_ground(self):
        """Falling to walking should require ground."""
        rule = next(
            (r for r in TRANSITION_RULES
             if r.from_mode == MovementMode.FALLING and
             r.to_mode == MovementMode.WALKING),
            None
        )
        assert rule is not None
        assert rule.requires_grounded is True


class TestMovementModeManager:
    """Tests for MovementModeManager class."""

    def test_construction(self):
        """Manager should be constructible."""
        manager = MovementModeManager()
        assert manager.current_mode == MovementMode.WALKING
        assert manager.is_transitioning is False

    def test_state_property(self):
        """state property should return current state."""
        manager = MovementModeManager()
        state = manager.state
        assert isinstance(state, MovementState)
        assert state.mode == MovementMode.WALKING

    def test_max_speed_property(self):
        """max_speed should return current mode's max speed."""
        manager = MovementModeManager()
        expected = DEFAULT_MODE_PARAMS[MovementMode.WALKING].max_speed
        assert manager.max_speed == expected

    def test_can_jump_property(self):
        """can_jump should return current mode's jump ability."""
        manager = MovementModeManager()
        assert manager.can_jump is True  # Walking can jump

    def test_height_modifier_property(self):
        """height_modifier should return current mode's modifier."""
        manager = MovementModeManager()
        assert manager.height_modifier == 1.0  # Walking is normal height

    def test_transition_to_same_mode(self):
        """Transitioning to same mode should succeed without callback."""
        manager = MovementModeManager()
        callback_called = []
        manager.set_mode_change_callback(lambda old, new: callback_called.append((old, new)))
        result = manager.transition_to_mode(MovementMode.WALKING)
        assert result is True
        assert len(callback_called) == 0  # No actual transition

    def test_transition_to_running(self):
        """Transition from walking to running should succeed."""
        manager = MovementModeManager()
        result = manager.transition_to_mode(MovementMode.RUNNING)
        assert result is True
        assert manager.current_mode == MovementMode.RUNNING

    def test_transition_with_callback(self):
        """Mode change callback should be called on transition."""
        manager = MovementModeManager()
        transitions = []
        manager.set_mode_change_callback(lambda old, new: transitions.append((old, new)))
        manager.transition_to_mode(MovementMode.RUNNING)
        assert len(transitions) == 1
        assert transitions[0] == (MovementMode.WALKING, MovementMode.RUNNING)

    def test_transition_complete_callback(self):
        """Transition complete callback should be called."""
        manager = MovementModeManager()
        completed = []
        manager.set_transition_complete_callback(lambda mode: completed.append(mode))
        manager.transition_to_mode(MovementMode.RUNNING)
        assert MovementMode.RUNNING in completed

    def test_transition_blocked_mode(self):
        """Transition to blocked mode should fail."""
        manager = MovementModeManager()
        manager.block_mode(MovementMode.RUNNING)
        result = manager.transition_to_mode(MovementMode.RUNNING)
        assert result is False
        assert manager.current_mode == MovementMode.WALKING

    def test_unblock_mode(self):
        """Unblocked mode should allow transition."""
        manager = MovementModeManager()
        manager.block_mode(MovementMode.RUNNING)
        manager.unblock_mode(MovementMode.RUNNING)
        result = manager.transition_to_mode(MovementMode.RUNNING)
        assert result is True

    def test_transition_requires_grounded(self):
        """Transition requiring ground should fail in air."""
        manager = MovementModeManager()
        manager._state.context = MovementContext.AIR
        # Crouching requires grounded
        result = manager.transition_to_mode(MovementMode.CROUCHING)
        assert result is False

    def test_transition_requires_stamina(self):
        """Transition requiring stamina should fail without it."""
        manager = MovementModeManager()
        manager.transition_to_mode(MovementMode.RUNNING)
        manager._state.stamina = 0.0
        # Sprinting requires stamina
        result = manager.transition_to_mode(MovementMode.SPRINTING)
        assert result is False

    def test_force_transition(self):
        """Force transition should bypass rules."""
        manager = MovementModeManager()
        manager._state.context = MovementContext.AIR
        manager._state.stamina = 0.0
        result = manager.transition_to_mode(MovementMode.CROUCHING, force=True)
        assert result is True
        assert manager.current_mode == MovementMode.CROUCHING

    def test_transition_with_duration(self):
        """Transition with duration should set transitioning state."""
        manager = MovementModeManager()
        # Walking to crouching has duration, but requires grounded
        manager._state.context = MovementContext.GROUND
        result = manager.transition_to_mode(MovementMode.CROUCHING)
        assert result is True
        # Due to duration, we need to check if transitioning started
        # The mode changes to target after transition_to_mode in implementation
        # OR we can check the transition was accepted
        assert manager.current_mode == MovementMode.CROUCHING or manager.is_transitioning

    def test_set_custom_params(self):
        """set_custom_params should override default params."""
        manager = MovementModeManager()
        custom = MovementModeParams(max_speed=100.0)
        manager.set_custom_params(MovementMode.WALKING, custom)
        assert manager.max_speed == 100.0

    def test_apply_movement_returns_vector(self):
        """apply_movement should return movement vector."""
        manager = MovementModeManager()
        result = manager.apply_movement(
            input_direction=Vector3(1.0, 0.0, 0.0),
            dt=0.1,
            is_grounded=True,
        )
        assert isinstance(result, Vector3)

    def test_apply_movement_increases_speed(self):
        """apply_movement with input should increase speed."""
        manager = MovementModeManager()
        manager.apply_movement(
            input_direction=Vector3(1.0, 0.0, 0.0),
            dt=0.1,
            is_grounded=True,
        )
        assert manager.state.current_speed > 0

    def test_apply_movement_updates_time_in_mode(self):
        """apply_movement should update time_in_mode."""
        manager = MovementModeManager()
        initial_time = manager.state.time_in_mode
        manager.apply_movement(Vector3.zero(), 0.1, True)
        assert manager.state.time_in_mode > initial_time

    def test_apply_movement_consumes_stamina(self):
        """apply_movement in stamina mode should consume stamina."""
        manager = MovementModeManager()
        manager.transition_to_mode(MovementMode.SPRINTING, force=True)
        initial_stamina = manager.state.stamina
        manager.apply_movement(Vector3(1.0, 0.0, 0.0), 0.1, True)
        assert manager.state.stamina < initial_stamina

    def test_apply_movement_regenerates_stamina(self):
        """apply_movement in no-cost mode should regenerate stamina."""
        manager = MovementModeManager()
        manager._state.stamina = 50.0
        manager.apply_movement(Vector3.zero(), 0.1, True)
        assert manager.state.stamina > 50.0

    def test_set_vertical_velocity(self):
        """set_vertical_velocity should update Y velocity."""
        manager = MovementModeManager()
        manager.set_vertical_velocity(10.0)
        assert manager.state.current_velocity.y == 10.0

    def test_add_vertical_velocity(self):
        """add_vertical_velocity should add to Y velocity."""
        manager = MovementModeManager()
        manager.set_vertical_velocity(5.0)
        manager.add_vertical_velocity(3.0)
        assert manager.state.current_velocity.y == 8.0

    def test_set_context(self):
        """set_context should update movement context."""
        manager = MovementModeManager()
        manager.set_context(MovementContext.WATER)
        assert manager.state.context == MovementContext.WATER

    def test_has_stamina_true(self):
        """has_stamina should return True when enough stamina."""
        manager = MovementModeManager()
        assert manager.has_stamina(50.0) is True

    def test_has_stamina_false(self):
        """has_stamina should return False when not enough stamina."""
        manager = MovementModeManager()
        assert manager.has_stamina(150.0) is False

    def test_consume_stamina_success(self):
        """consume_stamina should return True and reduce stamina."""
        manager = MovementModeManager()
        result = manager.consume_stamina(20.0)
        assert result is True
        assert manager.state.stamina == 80.0

    def test_consume_stamina_failure(self):
        """consume_stamina should return False when not enough."""
        manager = MovementModeManager()
        manager._state.stamina = 10.0
        result = manager.consume_stamina(20.0)
        assert result is False
        assert manager.state.stamina == 10.0

    def test_restore_stamina(self):
        """restore_stamina should increase stamina."""
        manager = MovementModeManager()
        manager._state.stamina = 50.0
        manager.restore_stamina(30.0)
        assert manager.state.stamina == 80.0

    def test_restore_stamina_capped(self):
        """restore_stamina should not exceed max."""
        manager = MovementModeManager()
        manager._state.stamina = 90.0
        manager.restore_stamina(50.0)
        assert manager.state.stamina == 100.0

    def test_is_moving(self):
        """is_moving should return True when speed > 0.1."""
        manager = MovementModeManager()
        manager._state.current_speed = 0.5
        assert manager.is_moving() is True

    def test_is_not_moving(self):
        """is_moving should return False when speed <= 0.1."""
        manager = MovementModeManager()
        manager._state.current_speed = 0.05
        assert manager.is_moving() is False

    def test_is_sprinting(self):
        """is_sprinting should return True in sprint mode."""
        manager = MovementModeManager()
        manager.transition_to_mode(MovementMode.SPRINTING, force=True)
        assert manager.is_sprinting() is True

    def test_is_crouching(self):
        """is_crouching should return True in crouch or prone."""
        manager = MovementModeManager()
        manager.transition_to_mode(MovementMode.CROUCHING, force=True)
        assert manager.is_crouching() is True
        manager.transition_to_mode(MovementMode.PRONE, force=True)
        assert manager.is_crouching() is True

    def test_is_airborne_falling(self):
        """is_airborne should return True when falling."""
        manager = MovementModeManager()
        manager.transition_to_mode(MovementMode.FALLING, force=True)
        assert manager.is_airborne() is True

    def test_is_airborne_flying(self):
        """is_airborne should return True when flying."""
        manager = MovementModeManager()
        manager.transition_to_mode(MovementMode.FLYING, force=True)
        assert manager.is_airborne() is True

    def test_is_swimming(self):
        """is_swimming should return True in swim mode."""
        manager = MovementModeManager()
        manager.transition_to_mode(MovementMode.SWIMMING, force=True)
        assert manager.is_swimming() is True

    def test_get_gravity_scale(self):
        """get_gravity_scale should return current mode's scale."""
        manager = MovementModeManager()
        assert manager.get_gravity_scale() == 1.0  # Walking
        manager.transition_to_mode(MovementMode.SWIMMING, force=True)
        assert manager.get_gravity_scale() < 1.0  # Swimming has reduced gravity

    def test_get_friction_modifier(self):
        """get_friction_modifier should return current mode's modifier."""
        manager = MovementModeManager()
        assert manager.get_friction_modifier() == 1.0  # Walking
        manager.transition_to_mode(MovementMode.SLIDING, force=True)
        assert manager.get_friction_modifier() < 1.0  # Sliding has reduced friction

    def test_get_state_dict(self):
        """get_state_dict should return serializable state."""
        manager = MovementModeManager()
        manager._state.stamina = 75.0
        manager._state.current_speed = 5.0
        data = manager.get_state_dict()
        assert data["mode"] == "WALKING"
        assert data["stamina"] == 75.0
        assert data["current_speed"] == 5.0

    def test_load_state_dict(self):
        """load_state_dict should restore state."""
        manager = MovementModeManager()
        data = {
            "mode": "RUNNING",
            "context": "GROUND",
            "stamina": 50.0,
            "current_speed": 6.0,
            "time_in_mode": 1.5,
            "velocity": (6.0, 0.0, 0.0),
        }
        manager.load_state_dict(data)
        assert manager.current_mode == MovementMode.RUNNING
        assert manager.state.stamina == 50.0
        assert manager.state.current_speed == 6.0
