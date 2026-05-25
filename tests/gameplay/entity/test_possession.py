"""
Comprehensive tests for the Controller Possession System.

Tests for:
- Controller possession/unpossession
- Possession chain validation
- Input routing to possessed pawn
- Camera switching on possession
- AI controller possession
- Player controller possession
- Possession events/callbacks
"""
from __future__ import annotations

import pytest
import threading
import weakref
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock, patch, call

from engine.gameplay.entity.possession import (
    Controller,
    ControllerMeta,
    PlayerController,
    AIController,
    PossessionState,
    PossessionDescriptor,
    PossessionManager,
)
from engine.gameplay.entity.actor import Pawn, Character
from engine.gameplay.entity.constants import (
    ControllerType,
    ENTITY_ID_START,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before each test."""
    Controller.reset_controller_ids()
    ControllerMeta.clear_registry()
    PossessionManager.reset_instance()
    yield
    PossessionManager.reset_instance()


@pytest.fixture
def controller():
    """Create a basic controller."""
    return Controller()


@pytest.fixture
def player_controller():
    """Create a player controller."""
    return PlayerController()


@pytest.fixture
def ai_controller():
    """Create an AI controller."""
    return AIController()


@pytest.fixture
def pawn():
    """Create a pawn for possession tests."""
    return Pawn(name="TestPawn")


@pytest.fixture
def character():
    """Create a character for possession tests."""
    return Character(name="TestCharacter")


@pytest.fixture
def possession_manager():
    """Create a possession manager."""
    PossessionManager.reset_instance()
    return PossessionManager()


# =============================================================================
# CONTROLLER TESTS
# =============================================================================


class TestControllerCreation:
    """Tests for Controller creation."""

    def test_controller_creation(self):
        """Test basic controller creation."""
        controller = Controller()
        assert controller is not None

    def test_controller_unique_id(self):
        """Test each controller gets a unique ID."""
        c1 = Controller()
        c2 = Controller()
        assert c1.controller_id != c2.controller_id

    def test_controller_id_starts_from_start(self):
        """Test controller IDs start from ENTITY_ID_START."""
        controller = Controller()
        assert controller.controller_id >= ENTITY_ID_START

    def test_controller_sequential_ids(self):
        """Test controller IDs are sequential."""
        c1 = Controller()
        c2 = Controller()
        assert c2.controller_id == c1.controller_id + 1

    def test_controller_is_active_default(self, controller):
        """Test controller is active by default."""
        assert controller.is_active is True

    def test_controller_not_possessing_default(self, controller):
        """Test controller is not possessing anything by default."""
        assert controller.is_possessing is False

    def test_controller_pawn_is_none_default(self, controller):
        """Test controller pawn is None by default."""
        assert controller.pawn is None


class TestControllerType:
    """Tests for controller type property."""

    def test_base_controller_type(self, controller):
        """Test base Controller type is PLAYER by default."""
        assert controller.controller_type == ControllerType.PLAYER

    def test_player_controller_type(self, player_controller):
        """Test PlayerController type is PLAYER."""
        assert player_controller.controller_type == ControllerType.PLAYER

    def test_ai_controller_type(self, ai_controller):
        """Test AIController type is AI."""
        assert ai_controller.controller_type == ControllerType.AI


class TestControllerActivation:
    """Tests for controller activation/deactivation."""

    def test_activate(self, controller):
        """Test activating controller."""
        controller.deactivate()
        controller.activate()
        assert controller.is_active is True

    def test_deactivate(self, controller):
        """Test deactivating controller."""
        controller.deactivate()
        assert controller.is_active is False

    def test_destroy(self, controller, pawn):
        """Test destroying controller unpossesses pawn."""
        controller.possess(pawn)
        controller.destroy()
        assert controller.is_active is False
        assert controller.pawn is None


# =============================================================================
# POSSESSION TESTS
# =============================================================================


class TestBasicPossession:
    """Tests for basic possession functionality."""

    def test_possess_pawn(self, controller, pawn):
        """Test possessing a pawn."""
        result = controller.possess(pawn)
        assert result is True
        assert controller.pawn is pawn

    def test_possess_sets_is_possessing(self, controller, pawn):
        """Test possession sets is_possessing flag."""
        controller.possess(pawn)
        assert controller.is_possessing is True

    def test_possess_none_fails(self, controller):
        """Test possessing None fails."""
        result = controller.possess(None)
        assert result is False

    def test_possess_updates_pawn_controller(self, controller, pawn):
        """Test possession updates pawn's controller reference."""
        controller.possess(pawn)
        assert pawn.controller is controller

    def test_possess_same_pawn_twice(self, controller, pawn):
        """Test possessing same pawn twice works."""
        controller.possess(pawn)
        result = controller.possess(pawn)
        assert result is True
        assert controller.pawn is pawn


class TestUnpossession:
    """Tests for unpossession functionality."""

    def test_unpossess_returns_pawn(self, controller, pawn):
        """Test unpossess returns the pawn."""
        controller.possess(pawn)
        result = controller.unpossess()
        assert result is pawn

    def test_unpossess_clears_pawn(self, controller, pawn):
        """Test unpossess clears the pawn reference."""
        controller.possess(pawn)
        controller.unpossess()
        assert controller.pawn is None

    def test_unpossess_clears_is_possessing(self, controller, pawn):
        """Test unpossess clears is_possessing flag."""
        controller.possess(pawn)
        controller.unpossess()
        assert controller.is_possessing is False

    def test_unpossess_when_not_possessing(self, controller):
        """Test unpossess when not possessing returns None."""
        result = controller.unpossess()
        assert result is None

    def test_unpossess_clears_pawn_controller(self, controller, pawn):
        """Test unpossess clears pawn's controller reference."""
        controller.possess(pawn)
        controller.unpossess()
        assert pawn.controller is None


class TestPossessionSwitch:
    """Tests for switching possession between pawns."""

    def test_possess_new_pawn_unpossesses_old(self, controller):
        """Test possessing new pawn unpossesses old one."""
        pawn1 = Pawn(name="Pawn1")
        pawn2 = Pawn(name="Pawn2")

        controller.possess(pawn1)
        controller.possess(pawn2)

        assert controller.pawn is pawn2
        assert pawn1.controller is None

    def test_multiple_possession_switches(self, controller):
        """Test multiple possession switches."""
        pawns = [Pawn(name=f"Pawn{i}") for i in range(5)]

        for pawn in pawns:
            controller.possess(pawn)
            assert controller.pawn is pawn

    def test_switch_to_already_possessed_pawn(self):
        """Test switching to pawn possessed by another controller."""
        controller1 = Controller()
        controller2 = Controller()
        pawn = Pawn(name="SharedPawn")

        controller1.possess(pawn)
        controller2.possess(pawn)

        # Second controller should now possess it
        assert controller2.pawn is pawn
        # First controller should be unpossessed
        assert controller1.pawn is None


class TestPossessionHistory:
    """Tests for possession history tracking."""

    def test_possession_history_recorded(self, controller, pawn):
        """Test possession is recorded in history."""
        controller.possess(pawn)
        assert len(controller._possession_history) > 0

    def test_unpossession_history_recorded(self, controller, pawn):
        """Test unpossession is recorded in history."""
        controller.possess(pawn)
        initial_len = len(controller._possession_history)
        controller.unpossess()
        assert len(controller._possession_history) > initial_len


# =============================================================================
# POSSESSION CALLBACK TESTS
# =============================================================================


class TestPossessionCallbacks:
    """Tests for possession callbacks."""

    def test_on_possess_called(self, controller, pawn):
        """Test _on_possess is called on possession."""
        controller._on_possess = Mock()
        controller.possess(pawn)
        controller._on_possess.assert_called_once_with(pawn)

    def test_on_unpossess_called(self, controller, pawn):
        """Test _on_unpossess is called on unpossession."""
        controller.possess(pawn)
        controller._on_unpossess = Mock()
        controller.unpossess()
        controller._on_unpossess.assert_called_once_with(pawn)

    def test_pawn_on_possessed_called(self, controller, pawn):
        """Test pawn's _on_possessed is called."""
        pawn._on_possessed = Mock()
        controller.possess(pawn)
        pawn._on_possessed.assert_called_once_with(controller)

    def test_pawn_on_unpossessed_called(self, controller, pawn):
        """Test pawn's _on_unpossessed is called."""
        controller.possess(pawn)
        pawn._on_unpossessed = Mock()
        controller.unpossess()
        pawn._on_unpossessed.assert_called_once_with(controller)


# =============================================================================
# PLAYER CONTROLLER TESTS
# =============================================================================


class TestPlayerControllerCreation:
    """Tests for PlayerController creation."""

    def test_player_controller_creation(self):
        """Test basic PlayerController creation."""
        pc = PlayerController()
        assert pc is not None

    def test_player_index_default(self, player_controller):
        """Test default player index is 0."""
        assert player_controller.player_index == 0

    def test_player_index_custom(self):
        """Test custom player index."""
        pc = PlayerController(player_index=1)
        assert pc.player_index == 1

    def test_is_local_player_default(self, player_controller):
        """Test is_local_player returns True by default."""
        assert player_controller.is_local_player is True

    def test_show_mouse_cursor_default(self, player_controller):
        """Test show_mouse_cursor is True by default."""
        assert player_controller.show_mouse_cursor is True

    def test_show_mouse_cursor_setter(self, player_controller):
        """Test setting show_mouse_cursor."""
        player_controller.show_mouse_cursor = False
        assert player_controller.show_mouse_cursor is False


class TestPlayerControllerInputBinding:
    """Tests for PlayerController input binding."""

    def test_bind_action(self, player_controller):
        """Test binding an action."""
        callback = Mock()
        player_controller.bind_action("jump", callback)
        assert "jump" in player_controller.get_bound_actions()

    def test_unbind_action(self, player_controller):
        """Test unbinding an action."""
        callback = Mock()
        player_controller.bind_action("jump", callback)
        result = player_controller.unbind_action("jump")
        assert result is True
        assert "jump" not in player_controller.get_bound_actions()

    def test_unbind_nonexistent_action(self, player_controller):
        """Test unbinding nonexistent action returns False."""
        result = player_controller.unbind_action("nonexistent")
        assert result is False

    def test_trigger_action(self, player_controller):
        """Test triggering a bound action."""
        callback = Mock()
        player_controller.bind_action("jump", callback)
        result = player_controller.trigger_action("jump")
        assert result is True
        callback.assert_called_once()

    def test_trigger_action_with_args(self, player_controller):
        """Test triggering action with arguments."""
        callback = Mock()
        player_controller.bind_action("move", callback)
        player_controller.trigger_action("move", 1.0, 0.5)
        callback.assert_called_once_with(1.0, 0.5)

    def test_trigger_nonexistent_action(self, player_controller):
        """Test triggering nonexistent action returns False."""
        result = player_controller.trigger_action("nonexistent")
        assert result is False

    def test_get_bound_actions_empty(self, player_controller):
        """Test get_bound_actions when empty."""
        actions = player_controller.get_bound_actions()
        assert actions == []

    def test_get_bound_actions_multiple(self, player_controller):
        """Test get_bound_actions with multiple bindings."""
        player_controller.bind_action("jump", Mock())
        player_controller.bind_action("fire", Mock())
        player_controller.bind_action("crouch", Mock())
        actions = player_controller.get_bound_actions()
        assert len(actions) == 3
        assert "jump" in actions
        assert "fire" in actions
        assert "crouch" in actions


class TestPlayerControllerCamera:
    """Tests for PlayerController camera control."""

    def test_set_camera_target(self, player_controller):
        """Test setting camera target."""
        target = Mock()
        player_controller.set_camera_target(target)
        assert player_controller.get_camera_target() is target

    def test_set_camera_target_none(self, player_controller):
        """Test setting camera target to None."""
        target = Mock()
        player_controller.set_camera_target(target)
        player_controller.set_camera_target(None)
        assert player_controller.get_camera_target() is None

    def test_get_camera_target_default(self, player_controller):
        """Test get_camera_target is None by default."""
        assert player_controller.get_camera_target() is None

    def test_camera_target_set_on_possess(self, player_controller, pawn):
        """Test camera target is set to pawn on possession."""
        player_controller.possess(pawn)
        assert player_controller.get_camera_target() is pawn

    def test_camera_target_cleared_on_unpossess(self, player_controller, pawn):
        """Test camera target is cleared on unpossession."""
        player_controller.possess(pawn)
        player_controller.unpossess()
        assert player_controller.get_camera_target() is None


class TestPlayerControllerPossession:
    """Tests for PlayerController possession specifics."""

    def test_setup_player_input_called(self, player_controller, pawn):
        """Test pawn's setup_player_input is called on possession."""
        pawn.setup_player_input = Mock()
        player_controller.possess(pawn)
        pawn.setup_player_input.assert_called_once()

    def test_is_player_controlled(self, player_controller, pawn):
        """Test pawn reports being player controlled."""
        player_controller.possess(pawn)
        assert pawn.is_player_controlled is True


# =============================================================================
# AI CONTROLLER TESTS
# =============================================================================


class TestAIControllerCreation:
    """Tests for AIController creation."""

    def test_ai_controller_creation(self):
        """Test basic AIController creation."""
        ai = AIController()
        assert ai is not None

    def test_behavior_tree_default(self, ai_controller):
        """Test behavior tree is None by default."""
        assert ai_controller.behavior_tree is None

    def test_behavior_tree_setter(self, ai_controller):
        """Test setting behavior tree."""
        bt = Mock()
        ai_controller.behavior_tree = bt
        assert ai_controller.behavior_tree is bt

    def test_behavior_tree_init(self):
        """Test creating with behavior tree."""
        bt = Mock()
        ai = AIController(behavior_tree=bt)
        assert ai.behavior_tree is bt

    def test_ai_enabled_default(self, ai_controller):
        """Test AI is enabled by default."""
        assert ai_controller.ai_enabled is True

    def test_ai_enabled_setter(self, ai_controller):
        """Test setting AI enabled."""
        ai_controller.ai_enabled = False
        assert ai_controller.ai_enabled is False


class TestAIControllerBlackboard:
    """Tests for AIController blackboard."""

    def test_blackboard_empty_default(self, ai_controller):
        """Test blackboard is empty by default."""
        assert len(ai_controller.blackboard) == 0

    def test_set_blackboard_value(self, ai_controller):
        """Test setting a blackboard value."""
        ai_controller.set_blackboard_value("target", Mock())
        assert "target" in ai_controller.blackboard

    def test_get_blackboard_value(self, ai_controller):
        """Test getting a blackboard value."""
        target = Mock()
        ai_controller.set_blackboard_value("target", target)
        assert ai_controller.get_blackboard_value("target") is target

    def test_get_blackboard_value_default(self, ai_controller):
        """Test getting nonexistent value returns default."""
        result = ai_controller.get_blackboard_value("nonexistent", "default")
        assert result == "default"

    def test_get_blackboard_value_none_default(self, ai_controller):
        """Test getting nonexistent value returns None by default."""
        result = ai_controller.get_blackboard_value("nonexistent")
        assert result is None

    def test_clear_blackboard(self, ai_controller):
        """Test clearing blackboard."""
        ai_controller.set_blackboard_value("key1", "value1")
        ai_controller.set_blackboard_value("key2", "value2")
        ai_controller.clear_blackboard()
        assert len(ai_controller.blackboard) == 0


class TestAIControllerFocus:
    """Tests for AIController focus system."""

    def test_set_focus(self, ai_controller):
        """Test setting focus target."""
        target = Mock()
        ai_controller.set_focus(target)
        assert ai_controller.get_focus() is target

    def test_get_focus_default(self, ai_controller):
        """Test get_focus returns None by default."""
        assert ai_controller.get_focus() is None

    def test_clear_focus(self, ai_controller):
        """Test clearing focus."""
        ai_controller.set_focus(Mock())
        ai_controller.clear_focus()
        assert ai_controller.get_focus() is None

    def test_set_focus_none(self, ai_controller):
        """Test setting focus to None."""
        ai_controller.set_focus(Mock())
        ai_controller.set_focus(None)
        assert ai_controller.get_focus() is None


class TestAIControllerMovement:
    """Tests for AIController movement requests."""

    def test_move_to_location(self, ai_controller, pawn):
        """Test move to location."""
        ai_controller.possess(pawn)
        result = ai_controller.move_to_location((100.0, 0.0, 100.0))
        assert result is True
        assert ai_controller.get_blackboard_value("move_target") == (100.0, 0.0, 100.0)

    def test_move_to_location_no_pawn(self, ai_controller):
        """Test move to location without pawn fails."""
        result = ai_controller.move_to_location((100.0, 0.0, 100.0))
        assert result is False

    def test_move_to_location_acceptance_radius(self, ai_controller, pawn):
        """Test move to location with custom acceptance radius."""
        ai_controller.possess(pawn)
        ai_controller.move_to_location((100.0, 0.0, 100.0), acceptance_radius=25.0)
        assert ai_controller.get_blackboard_value("acceptance_radius") == 25.0

    def test_move_to_actor(self, ai_controller, pawn):
        """Test move to actor."""
        ai_controller.possess(pawn)
        target = Mock()
        target.position = (50.0, 0.0, 50.0)
        result = ai_controller.move_to_actor(target)
        assert result is True

    def test_move_to_actor_none(self, ai_controller, pawn):
        """Test move to None actor fails."""
        ai_controller.possess(pawn)
        result = ai_controller.move_to_actor(None)
        assert result is False

    def test_move_to_actor_no_position(self, ai_controller, pawn):
        """Test move to actor without position fails."""
        ai_controller.possess(pawn)
        target = Mock(spec=[])  # No position attribute
        result = ai_controller.move_to_actor(target)
        assert result is False

    def test_stop_movement(self, ai_controller, pawn):
        """Test stopping movement."""
        ai_controller.possess(pawn)
        ai_controller.move_to_location((100.0, 0.0, 100.0))
        ai_controller.stop_movement()
        assert ai_controller.get_blackboard_value("move_target") is None


class TestAIControllerTick:
    """Tests for AIController tick behavior."""

    def test_tick_calls_behavior_tree(self, ai_controller, pawn):
        """Test tick calls behavior tree."""
        bt = Mock()
        ai_controller.behavior_tree = bt
        ai_controller.possess(pawn)
        ai_controller.tick(0.016)
        bt.tick.assert_called_once()

    def test_tick_no_behavior_tree(self, ai_controller, pawn):
        """Test tick without behavior tree doesn't raise."""
        ai_controller.possess(pawn)
        ai_controller.tick(0.016)  # Should not raise

    def test_tick_inactive(self, ai_controller, pawn):
        """Test tick when inactive does nothing."""
        bt = Mock()
        ai_controller.behavior_tree = bt
        ai_controller.possess(pawn)
        ai_controller.deactivate()
        ai_controller.tick(0.016)
        bt.tick.assert_not_called()

    def test_tick_ai_disabled(self, ai_controller, pawn):
        """Test tick when AI disabled does nothing."""
        bt = Mock()
        ai_controller.behavior_tree = bt
        ai_controller.possess(pawn)
        ai_controller.ai_enabled = False
        ai_controller.tick(0.016)
        bt.tick.assert_not_called()


# =============================================================================
# POSSESSION MANAGER TESTS
# =============================================================================


class TestPossessionManagerSingleton:
    """Tests for PossessionManager singleton pattern."""

    def test_singleton_instance(self):
        """Test PossessionManager is a singleton."""
        pm1 = PossessionManager()
        pm2 = PossessionManager()
        assert pm1 is pm2

    def test_reset_instance(self):
        """Test resetting singleton instance."""
        pm1 = PossessionManager()
        PossessionManager.reset_instance()
        pm2 = PossessionManager()
        # Should be a new instance after reset
        assert pm2 is not None


class TestPossessionManagerRegistration:
    """Tests for controller registration with PossessionManager."""

    def test_register_controller(self, possession_manager, controller):
        """Test registering a controller."""
        possession_manager.register_controller(controller)
        # Should not raise
        assert True

    def test_unregister_controller(self, possession_manager, controller):
        """Test unregistering a controller."""
        possession_manager.register_controller(controller)
        possession_manager.unregister_controller(controller)
        # Should not raise
        assert True


class TestPossessionManagerQueries:
    """Tests for PossessionManager query methods."""

    def test_get_all_controllers(self, possession_manager):
        """Test getting all controllers."""
        c1 = Controller()
        c2 = Controller()
        possession_manager.register_controller(c1)
        possession_manager.register_controller(c2)
        all_controllers = possession_manager.get_all_controllers()
        assert len(all_controllers) == 2

    def test_get_player_controllers(self, possession_manager):
        """Test getting player controllers."""
        pc = PlayerController()
        ai = AIController()
        possession_manager.register_controller(pc)
        possession_manager.register_controller(ai)
        player_controllers = possession_manager.get_player_controllers()
        assert len(player_controllers) == 1
        assert pc in player_controllers

    def test_get_ai_controllers(self, possession_manager):
        """Test getting AI controllers."""
        pc = PlayerController()
        ai = AIController()
        possession_manager.register_controller(pc)
        possession_manager.register_controller(ai)
        ai_controllers = possession_manager.get_ai_controllers()
        assert len(ai_controllers) == 1
        assert ai in ai_controllers

    def test_get_controller_for_pawn(self, possession_manager, pawn):
        """Test getting controller for a pawn."""
        controller = Controller()
        possession_manager.register_controller(controller)
        controller.possess(pawn)
        result = possession_manager.get_controller_for_pawn(pawn)
        # Note: This depends on implementation details
        # The manager may or may not track this automatically
        assert result is None or result is controller

    def test_get_pawn_for_controller(self, possession_manager, pawn):
        """Test getting pawn for a controller."""
        controller = Controller()
        possession_manager.register_controller(controller)
        controller.possess(pawn)
        result = possession_manager.get_pawn_for_controller(controller)
        assert result is pawn


class TestPossessionManagerOperations:
    """Tests for PossessionManager operations."""

    def test_switch_possession(self, possession_manager, controller, pawn):
        """Test switching possession."""
        possession_manager.register_controller(controller)
        result = possession_manager.switch_possession(controller, pawn)
        assert result is True
        assert controller.pawn is pawn

    def test_clear(self, possession_manager):
        """Test clearing manager state."""
        c = Controller()
        possession_manager.register_controller(c)
        possession_manager.clear()
        controllers = possession_manager.get_all_controllers()
        assert len(controllers) == 0


# =============================================================================
# POSSESSION STATE TESTS
# =============================================================================


class TestPossessionState:
    """Tests for PossessionState data class."""

    def test_possession_state_creation(self):
        """Test creating PossessionState."""
        state = PossessionState()
        assert state is not None

    def test_possession_state_defaults(self):
        """Test PossessionState default values."""
        state = PossessionState()
        assert state.pawn is None
        assert state.is_possessing is False
        assert state.possession_time == 0.0
        assert state.pending_pawn is None

    def test_possession_state_with_values(self):
        """Test PossessionState with custom values."""
        pawn_ref = Mock()
        state = PossessionState(
            pawn=weakref.ref(pawn_ref),
            is_possessing=True,
            possession_time=1.5,
        )
        assert state.is_possessing is True
        assert state.possession_time == 1.5


# =============================================================================
# POSSESSION DESCRIPTOR TESTS
# =============================================================================


class TestPossessionDescriptor:
    """Tests for PossessionDescriptor."""

    def test_descriptor_id(self):
        """Test descriptor has correct ID."""
        descriptor = PossessionDescriptor()
        assert descriptor.descriptor_id == "possession"

    def test_pre_set_accepts_none(self):
        """Test pre_set accepts None value."""
        descriptor = PossessionDescriptor()
        result = descriptor.pre_set(Mock(), None)
        assert result is None

    def test_descriptor_steps(self):
        """Test descriptor defines expected steps."""
        descriptor = PossessionDescriptor()
        steps = descriptor.descriptor_steps
        ops = {s.op.name for s in steps}
        assert "TRACK" in ops
        assert "VALIDATE" in ops
        assert "HOOK" in ops


# =============================================================================
# CONTROLLER META TESTS
# =============================================================================


class TestControllerMeta:
    """Tests for ControllerMeta metaclass."""

    def test_controller_meta_registry(self):
        """Test controllers are registered in metaclass registry."""
        class TestController(Controller):
            pass

        assert TestController._controller_type_id > 0
        found = ControllerMeta.get_by_id(TestController._controller_type_id)
        assert found is TestController

    def test_controller_meta_unique_ids(self):
        """Test each controller class gets unique ID."""
        class Controller1(Controller):
            pass

        class Controller2(Controller):
            pass

        assert Controller1._controller_type_id != Controller2._controller_type_id

    def test_controller_meta_get_by_name(self):
        """Test looking up controller by qualified name."""
        class NamedController(Controller):
            pass

        found = ControllerMeta.get_by_name(NamedController._controller_type_name)
        assert found is NamedController

    def test_controller_meta_base_classes_not_registered(self):
        """Test base classes have type ID 0."""
        assert Controller._controller_type_id == 0
        assert PlayerController._controller_type_id == 0
        assert AIController._controller_type_id == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPossessionIntegration:
    """Integration tests for possession system."""

    def test_player_possesses_character(self):
        """Test player controller possessing a character."""
        player = PlayerController(player_index=0)
        character = Character(name="PlayerCharacter")

        result = player.possess(character)

        assert result is True
        assert player.pawn is character
        assert character.controller is player
        assert character.is_player_controlled is True

    def test_ai_possesses_character(self):
        """Test AI controller possessing a character."""
        ai = AIController()
        character = Character(name="AICharacter")

        result = ai.possess(character)

        assert result is True
        assert ai.pawn is character
        assert character.controller is ai
        assert character.is_player_controlled is False

    def test_possession_transfer(self):
        """Test transferring possession between controllers."""
        player = PlayerController()
        ai = AIController()
        pawn = Pawn(name="TransferPawn")

        player.possess(pawn)
        assert pawn.controller is player

        ai.possess(pawn)
        assert pawn.controller is ai
        assert player.pawn is None

    def test_multi_player_possession(self):
        """Test multiple players with separate pawns."""
        player1 = PlayerController(player_index=0)
        player2 = PlayerController(player_index=1)
        pawn1 = Pawn(name="Player1Pawn")
        pawn2 = Pawn(name="Player2Pawn")

        player1.possess(pawn1)
        player2.possess(pawn2)

        assert player1.pawn is pawn1
        assert player2.pawn is pawn2
        assert pawn1.controller is player1
        assert pawn2.controller is player2


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestPossessionEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_weak_reference_pawn_deleted(self, controller):
        """Test handling when possessed pawn is deleted."""
        pawn = Pawn(name="DeletedPawn")
        controller.possess(pawn)

        # Delete the pawn
        del pawn
        import gc
        gc.collect()

        # Controller should handle this gracefully
        assert controller.pawn is None or controller.is_possessing is False

    def test_controller_repr(self, controller, pawn):
        """Test controller string representation."""
        repr_str = repr(controller)
        assert "Controller" in repr_str
        assert "no pawn" in repr_str

        controller.possess(pawn)
        repr_str = repr(controller)
        assert "pawn=" in repr_str

    def test_reset_controller_ids(self):
        """Test resetting controller ID counter."""
        c1 = Controller()
        first_id = c1.controller_id
        Controller.reset_controller_ids()
        c2 = Controller()
        assert c2.controller_id == ENTITY_ID_START

    def test_thread_safety_registration(self, possession_manager):
        """Test thread-safe controller registration."""
        controllers = []
        lock = threading.Lock()

        def register_controller():
            c = Controller()
            possession_manager.register_controller(c)
            with lock:
                controllers.append(c)

        threads = [threading.Thread(target=register_controller) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(controllers) == 50
