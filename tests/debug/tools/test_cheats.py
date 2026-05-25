"""
Tests for the cheat system.

Tests verify:
1. Cheat state management
2. Command registration and execution
3. Build-type security guards
4. Multiplayer restrictions
5. Actual game state impact (not just flag setting)
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from engine.debug.tools.cheats import (
    CheatCommand,
    CheatConfig,
    CheatFlags,
    CheatManager,
    CheatState,
    cheats_allowed,
    get_cheat_config,
    get_cheat_manager,
    is_debug_build,
    is_shipping_build,
    reset_cheat_manager,
    set_cheat_config,
)


class TestCheatState:
    """Tests for CheatState dataclass."""

    def test_default_state(self):
        """Test default cheat state values."""
        state = CheatState()
        assert state.god_mode is False
        assert state.fly_mode is False
        assert state.ghost_mode is False
        assert state.infinite_ammo is False
        assert state.invisible is False
        assert state.speed_multiplier == 1.0
        assert state.damage_multiplier == 1.0

    def test_custom_state(self):
        """Test cheat state with custom values."""
        state = CheatState(
            god_mode=True,
            fly_mode=True,
            speed_multiplier=2.0,
        )
        assert state.god_mode is True
        assert state.fly_mode is True
        assert state.ghost_mode is False
        assert state.speed_multiplier == 2.0

    def test_is_any_active_false(self):
        """Test is_any_active returns False when no cheats active."""
        state = CheatState()
        assert state.is_any_active() is False

    def test_is_any_active_true_flags(self):
        """Test is_any_active returns True for boolean flags."""
        state = CheatState(god_mode=True)
        assert state.is_any_active() is True

        state = CheatState(fly_mode=True)
        assert state.is_any_active() is True

        state = CheatState(invisible=True)
        assert state.is_any_active() is True

    def test_is_any_active_true_multipliers(self):
        """Test is_any_active returns True for non-default multipliers."""
        state = CheatState(speed_multiplier=2.0)
        assert state.is_any_active() is True

        state = CheatState(damage_multiplier=0.5)
        assert state.is_any_active() is True


class TestCheatCommand:
    """Tests for CheatCommand dataclass."""

    def test_create_command(self):
        """Test creating a cheat command."""
        handler = Mock()
        cmd = CheatCommand(
            name="test",
            handler=handler,
            description="Test command",
        )
        assert cmd.name == "test"
        assert cmd.handler == handler
        assert cmd.description == "Test command"
        assert cmd.flags == CheatFlags.NONE
        assert cmd.aliases == []
        assert cmd.parameters == []

    def test_command_with_aliases(self):
        """Test command with aliases."""
        handler = Mock()
        cmd = CheatCommand(
            name="god",
            handler=handler,
            aliases=["godmode", "invincible"],
        )
        assert cmd.aliases == ["godmode", "invincible"]

    def test_command_with_flags(self):
        """Test command with flags."""
        handler = Mock()
        cmd = CheatCommand(
            name="test",
            handler=handler,
            flags=CheatFlags.LOGGED | CheatFlags.REQUIRES_DEBUG_BUILD,
        )
        assert CheatFlags.LOGGED in cmd.flags
        assert CheatFlags.REQUIRES_DEBUG_BUILD in cmd.flags
        assert CheatFlags.REQUIRES_ADMIN not in cmd.flags

    def test_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError):
            CheatCommand(name="", handler=Mock())


class TestCheatManager:
    """Tests for CheatManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh CheatManager."""
        return CheatManager()

    def test_enabled_by_default(self, manager):
        """Test that cheats are enabled by default."""
        assert manager.enabled is True

    def test_disable_cheats(self, manager):
        """Test disabling cheats."""
        # Enable some cheats
        manager.god_mode(True)
        assert manager.get_state().god_mode is True

        # Disable cheats
        manager.enabled = False
        assert manager.enabled is False

        # State should be reset
        assert manager.get_state().god_mode is False

    def test_builtin_cheats_registered(self, manager):
        """Test that built-in cheats are registered."""
        commands = manager.list_commands()
        names = [cmd.name for cmd in commands]

        assert "god" in names
        assert "fly" in names
        assert "ghost" in names
        assert "teleport" in names
        assert "spawn" in names
        assert "kill" in names
        assert "sethealth" in names
        assert "give" in names
        assert "infiniteammo" in names

    def test_register_custom_command(self, manager):
        """Test registering a custom command."""
        handler = Mock(return_value="success")
        cmd = CheatCommand(
            name="custom",
            handler=handler,
            description="Custom cheat",
        )
        manager.register(cmd)

        # Execute the custom command
        success, result = manager.execute("custom")
        assert success is True
        assert result == "success"
        handler.assert_called_once()

    def test_unregister_command(self, manager):
        """Test unregistering a command."""
        handler = Mock()
        cmd = CheatCommand(name="removeme", handler=handler)
        manager.register(cmd)

        assert manager.unregister("removeme") is True
        assert manager.unregister("removeme") is False  # Already removed

        success, _ = manager.execute("removeme")
        assert success is False

    def test_execute_with_alias(self, manager):
        """Test executing command via alias."""
        # 'godmode' is an alias for 'god'
        success, result = manager.execute("godmode")
        assert success is True
        assert "enabled" in result.lower() or "disabled" in result.lower()

    def test_execute_unknown_command(self, manager):
        """Test executing unknown command."""
        success, result = manager.execute("unknowncommand")
        assert success is False
        assert "unknown" in result.lower()

    def test_execute_when_disabled(self, manager):
        """Test executing command when cheats are disabled."""
        manager.enabled = False
        success, result = manager.execute("god")
        assert success is False
        assert "disabled" in result.lower()

    def test_god_mode(self, manager):
        """Test god mode cheat."""
        # Enable
        result = manager.god_mode(True)
        assert result is True
        assert manager.get_state().god_mode is True

        # Disable
        result = manager.god_mode(False)
        assert result is False
        assert manager.get_state().god_mode is False

    def test_fly_mode(self, manager):
        """Test fly mode cheat."""
        result = manager.fly_mode(True)
        assert result is True
        assert manager.get_state().fly_mode is True

    def test_ghost_mode(self, manager):
        """Test ghost mode cheat."""
        result = manager.ghost_mode(True)
        assert result is True
        assert manager.get_state().ghost_mode is True

    def test_teleport(self, manager):
        """Test teleport cheat."""
        pos = manager.teleport(100.0, 50.0, 200.0)
        assert pos == (100.0, 50.0, 200.0)

    def test_set_health(self, manager):
        """Test set health cheat."""
        health = manager.set_health(75.0)
        assert health == 75.0

    def test_give_item(self, manager):
        """Test give item cheat."""
        count = manager.give_item("sword", 5)
        assert count == 5

    def test_infinite_ammo(self, manager):
        """Test infinite ammo cheat."""
        result = manager.infinite_ammo(True)
        assert result is True
        assert manager.get_state().infinite_ammo is True

    def test_entity_specific_state(self, manager):
        """Test cheat state for specific entity."""
        entity1 = Mock()
        entity2 = Mock()

        # Enable god mode for entity1 only
        manager.god_mode(True, entity=entity1)

        assert manager.get_state(entity1).god_mode is True
        assert manager.get_state(entity2).god_mode is False
        assert manager.get_state(None).god_mode is False  # Global state

    def test_callback_on_cheat_execution(self, manager):
        """Test callback is called when cheat is executed."""
        callback = Mock()
        manager.add_callback("god", callback)

        manager.execute("god", entity=None)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "god"

    def test_remove_callback(self, manager):
        """Test removing a callback."""
        callback = Mock()
        manager.add_callback("god", callback)
        assert manager.remove_callback("god", callback) is True
        assert manager.remove_callback("god", callback) is False  # Already removed

        manager.execute("god")
        callback.assert_not_called()

    def test_console_command_toggle(self, manager):
        """Test console command toggling."""
        # First call enables
        result = manager._cmd_god()
        assert "enabled" in result.lower()
        assert manager.get_state().god_mode is True

        # Second call disables
        result = manager._cmd_god()
        assert "disabled" in result.lower()
        assert manager.get_state().god_mode is False

    def test_console_teleport_command(self, manager):
        """Test console teleport command."""
        result = manager._cmd_teleport(10.0, 20.0, 30.0)
        assert "10" in result and "20" in result and "30" in result

    def test_console_spawn_command(self, manager):
        """Test console spawn command."""
        result = manager._cmd_spawn("enemy")
        # spawn returns None (placeholder), so should indicate failure or attempt
        assert "enemy" in result.lower()

    def test_console_give_command(self, manager):
        """Test console give command."""
        result = manager._cmd_give_item("sword", 3)
        assert "3" in result and "sword" in result.lower()

    def test_cheat_flags(self):
        """Test cheat flag combinations."""
        flags = CheatFlags.LOGGED | CheatFlags.PERSISTENT
        assert CheatFlags.LOGGED in flags
        assert CheatFlags.PERSISTENT in flags
        assert CheatFlags.REQUIRES_ADMIN not in flags

        flags |= CheatFlags.REQUIRES_ADMIN
        assert CheatFlags.REQUIRES_ADMIN in flags


class TestGetCheatManager:
    """Tests for get_cheat_manager singleton."""

    def test_singleton(self):
        """Test that get_cheat_manager returns singleton."""
        # Reset the singleton
        reset_cheat_manager()

        manager1 = get_cheat_manager()
        manager2 = get_cheat_manager()
        assert manager1 is manager2


class TestBuildTypeGuards:
    """Tests for build-type security guards."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset singleton and config before each test."""
        reset_cheat_manager()
        set_cheat_config(CheatConfig())
        yield
        # Cleanup
        reset_cheat_manager()
        set_cheat_config(CheatConfig())
        # Clear environment variables
        for var in ["GAME_BUILD_TYPE", "SHIPPING", "NDEBUG", "DEBUG", "_DEBUG"]:
            os.environ.pop(var, None)

    def test_cheats_allowed_in_debug_build(self):
        """Test cheats are allowed in debug builds."""
        os.environ["GAME_BUILD_TYPE"] = "DEBUG"
        assert cheats_allowed() is True

    def test_cheats_blocked_in_shipping_build(self):
        """Test cheats are blocked in shipping builds by default."""
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"
        assert cheats_allowed() is False

    def test_cheats_blocked_with_shipping_env_var(self):
        """Test cheats blocked when SHIPPING=1."""
        os.environ["SHIPPING"] = "1"
        assert is_shipping_build() is True
        assert cheats_allowed() is False

    def test_cheats_blocked_with_ndebug(self):
        """Test cheats blocked when NDEBUG=1."""
        os.environ["NDEBUG"] = "1"
        assert is_shipping_build() is True
        assert cheats_allowed() is False

    def test_is_debug_build_detection(self):
        """Test debug build detection."""
        os.environ["DEBUG"] = "1"
        assert is_debug_build() is True

        os.environ.pop("DEBUG", None)
        os.environ["_DEBUG"] = "1"
        assert is_debug_build() is True

    def test_manager_disabled_in_shipping(self):
        """Test CheatManager is disabled in shipping builds."""
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"
        manager = CheatManager()
        assert manager.enabled is False

    def test_cannot_enable_cheats_in_shipping(self):
        """Test cannot enable cheats in shipping builds."""
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"
        manager = CheatManager()
        manager.enabled = True  # Try to enable
        assert manager.enabled is False  # Should remain disabled

    def test_execute_blocked_in_shipping(self):
        """Test cheat execution is blocked in shipping builds."""
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"
        manager = CheatManager()
        success, message = manager.execute("god")
        assert success is False
        assert "disabled" in message.lower()

    def test_config_allows_shipping_override(self):
        """Test config can allow cheats in shipping (for testing)."""
        config = CheatConfig(allow_in_shipping=True)
        set_cheat_config(config)
        os.environ["GAME_BUILD_TYPE"] = "SHIPPING"
        assert cheats_allowed() is True


class TestMultiplayerRestrictions:
    """Tests for multiplayer cheat restrictions."""

    @pytest.fixture
    def manager(self):
        """Create a fresh CheatManager."""
        reset_cheat_manager()
        return CheatManager()

    def test_multiplayer_mode_tracking(self, manager):
        """Test multiplayer mode is tracked."""
        assert manager.is_multiplayer is False
        manager.set_multiplayer_mode(True)
        assert manager.is_multiplayer is True

    def test_disabled_in_multiplayer_flag(self, manager):
        """Test DISABLED_IN_MULTIPLAYER flag is enforced."""
        # Register a command with multiplayer restriction
        handler = Mock(return_value="success")
        cmd = CheatCommand(
            name="mp_blocked",
            handler=handler,
            flags=CheatFlags.DISABLED_IN_MULTIPLAYER,
        )
        manager.register(cmd)

        # Works in single player
        success, _ = manager.execute("mp_blocked")
        assert success is True
        handler.assert_called_once()

        # Blocked in multiplayer
        handler.reset_mock()
        manager.set_multiplayer_mode(True)
        success, message = manager.execute("mp_blocked")
        assert success is False
        assert "multiplayer" in message.lower()
        handler.assert_not_called()


class TestCheatGameImpact:
    """
    Tests that verify cheats actually impact game state,
    not just that flags are set.

    These tests use mock game systems to verify the cheats
    would have the intended effect in a real game.
    """

    @pytest.fixture
    def manager(self):
        """Create a fresh CheatManager."""
        reset_cheat_manager()
        return CheatManager()

    def test_god_mode_affects_entity_damage(self, manager):
        """Test god mode actually prevents damage to entity."""
        # Create mock entity with health system
        entity = Mock()
        entity.health = 100.0

        def apply_damage(amount):
            state = manager.get_state(entity)
            if state.god_mode:
                return  # Damage blocked
            entity.health -= amount

        entity.apply_damage = apply_damage

        # Without god mode, entity takes damage
        entity.apply_damage(50)
        assert entity.health == 50.0

        # Enable god mode
        manager.god_mode(True, entity=entity)
        assert manager.get_state(entity).god_mode is True

        # With god mode, damage is blocked
        entity.apply_damage(50)
        assert entity.health == 50.0  # Health unchanged

    def test_speed_multiplier_affects_movement(self, manager):
        """Test speed multiplier actually affects entity movement."""
        entity = Mock()
        base_speed = 10.0

        def get_effective_speed():
            state = manager.get_state(entity)
            return base_speed * state.speed_multiplier

        # Default speed
        assert get_effective_speed() == 10.0

        # Modify speed via state
        state = manager._get_or_create_state(entity)
        state.speed_multiplier = 2.0

        # Speed is now doubled
        assert get_effective_speed() == 20.0

    def test_infinite_ammo_affects_weapon(self, manager):
        """Test infinite ammo actually prevents ammo consumption."""
        entity = Mock()
        entity.ammo = 30

        def fire_weapon():
            state = manager.get_state(entity)
            if not state.infinite_ammo:
                entity.ammo -= 1

        # Normal firing consumes ammo
        fire_weapon()
        assert entity.ammo == 29

        # Enable infinite ammo
        manager.infinite_ammo(True, entity=entity)

        # Firing no longer consumes ammo
        fire_weapon()
        assert entity.ammo == 29  # Unchanged
