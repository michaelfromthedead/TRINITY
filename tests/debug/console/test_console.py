"""Tests for Console - modes, history, and execution.

Tests cover:
- Console modes (OVERLAY, FULLSCREEN, MINI)
- Command history navigation
- Command execution
- CVar handling
- Output management
"""

import pytest

from engine.debug.console.aliases import AliasRegistry
from engine.debug.console.commands import CommandRegistry
from engine.debug.console.console import Console, ConsoleMode, ConsoleOutput
from engine.debug.console.cvar import CVar, CVarFlags, CVarRegistry


@pytest.fixture(autouse=True)
def reset_cvar_registry():
    """Reset the CVarRegistry singleton before each test."""
    CVarRegistry.reset_instance()
    yield
    CVarRegistry.reset_instance()


@pytest.fixture
def console():
    """Create a fresh console for testing."""
    return Console()


class TestConsoleModes:
    """Test console display modes."""

    def test_default_mode_is_overlay(self, console):
        """Test default mode is OVERLAY."""
        assert console.mode == ConsoleMode.OVERLAY

    def test_set_mode(self, console):
        """Test setting console mode."""
        console.mode = ConsoleMode.FULLSCREEN
        assert console.mode == ConsoleMode.FULLSCREEN

        console.mode = ConsoleMode.MINI
        assert console.mode == ConsoleMode.MINI

    def test_cycle_mode(self, console):
        """Test cycling through modes."""
        assert console.mode == ConsoleMode.OVERLAY

        console.cycle_mode()
        assert console.mode == ConsoleMode.FULLSCREEN

        console.cycle_mode()
        assert console.mode == ConsoleMode.MINI

        console.cycle_mode()
        assert console.mode == ConsoleMode.OVERLAY


class TestConsoleVisibility:
    """Test console visibility."""

    def test_default_hidden(self, console):
        """Test console is hidden by default."""
        assert not console.visible

    def test_toggle_visibility(self, console):
        """Test toggling visibility."""
        assert not console.visible

        result = console.toggle()
        assert result is True
        assert console.visible

        result = console.toggle()
        assert result is False
        assert not console.visible

    def test_visibility_callback(self, console):
        """Test visibility change callback."""
        callback_data = {"visible": None}

        def callback(visible):
            callback_data["visible"] = visible

        console.on_visibility_change(callback)
        console.visible = True

        assert callback_data["visible"] is True


class TestConsoleHistory:
    """Test command history navigation."""

    def test_history_starts_empty(self, console):
        """Test history is empty initially."""
        assert console.get_history() == []

    def test_commands_added_to_history(self, console):
        """Test executed commands are added to history."""
        console.execute("echo test1")
        console.execute("echo test2")

        history = console.get_history()
        assert len(history) == 2
        assert "echo test1" in history
        assert "echo test2" in history

    def test_history_navigation_up(self, console):
        """Test navigating up through history."""
        console.execute("cmd1")
        console.execute("cmd2")
        console.execute("cmd3")

        assert console.history_up() == "cmd3"
        assert console.history_up() == "cmd2"
        assert console.history_up() == "cmd1"
        assert console.history_up() == "cmd1"  # At beginning, stays

    def test_history_navigation_down(self, console):
        """Test navigating down through history."""
        console.execute("cmd1")
        console.execute("cmd2")

        console.history_up()  # cmd2
        console.history_up()  # cmd1

        assert console.history_down() == "cmd2"
        assert console.history_down() is None  # Back to current

    def test_history_no_duplicates(self, console):
        """Test consecutive duplicates aren't added."""
        console.execute("same")
        console.execute("same")
        console.execute("same")

        assert len(console.get_history()) == 1

    def test_clear_history(self, console):
        """Test clearing history."""
        console.execute("cmd1")
        console.execute("cmd2")

        console.clear_history()
        assert console.get_history() == []

    def test_history_limit(self):
        """Test history is limited to max_history."""
        console = Console(max_history=5)

        for i in range(10):
            console.execute(f"cmd{i}")

        history = console.get_history()
        assert len(history) == 5
        assert "cmd5" in history
        assert "cmd9" in history
        assert "cmd0" not in history


class TestConsoleExecution:
    """Test command execution."""

    def test_execute_command(self, console):
        """Test executing a registered command."""
        result = console.execute("echo hello world")

        assert "hello world" in result

    def test_execute_empty_string(self, console):
        """Test executing empty string returns None."""
        result = console.execute("")
        assert result is None

        result = console.execute("   ")
        assert result is None

    def test_execute_unknown_command(self, console):
        """Test executing unknown command."""
        result = console.execute("nonexistent")

        assert "Unknown" in result

    def test_execute_cvar_get(self, console):
        """Test getting CVar value."""
        cvar = CVar("test.value", default=42)

        result = console.execute("test.value")
        assert "42" in result

    def test_execute_cvar_set(self, console):
        """Test setting CVar value."""
        cvar = CVar("test.setter", default=0)

        result = console.execute("test.setter 100")
        assert "100" in result
        assert cvar.value == 100

    def test_execute_with_quotes(self, console):
        """Test executing command with quoted arguments."""
        result = console.execute('echo "hello world"')

        assert "hello world" in result

    def test_execute_adds_to_output(self, console):
        """Test execution adds to output."""
        initial_count = len(console.get_output())
        console.execute("echo test")

        assert len(console.get_output()) > initial_count

    def test_register_custom_command(self, console):
        """Test registering custom command."""
        console.register_command(
            "custom",
            lambda x: f"got {x}",
            description="Custom command"
        )

        result = console.execute("custom argument")
        assert "got argument" in result


class TestConsoleAliases:
    """Test alias handling."""

    def test_register_and_execute_alias(self, console):
        """Test registering and executing an alias."""
        console.register_alias("hw", "echo hello world")

        result = console.execute("hw")
        assert "hello world" in result

    def test_multi_command_alias(self, console):
        """Test alias with multiple commands."""
        console.register_command("set_a", lambda: "A", description="")
        console.register_command("set_b", lambda: "B", description="")
        console.register_alias("both", "set_a; set_b")

        result = console.execute("both")
        # Both commands should execute
        assert result is not None


class TestConsoleOutput:
    """Test console output management."""

    def test_add_output(self, console):
        """Test adding output."""
        console.add_output("Test message", "info")

        output = console.get_output()
        assert any("Test message" in o.text for o in output)

    def test_output_levels(self, console):
        """Test output with different levels."""
        console.add_output("Info", "info")
        console.add_output("Warning", "warning")
        console.add_output("Error", "error")

        output = console.get_output()
        levels = [o.level for o in output[-3:]]
        assert "info" in levels
        assert "warning" in levels
        assert "error" in levels

    def test_clear_output(self, console):
        """Test clearing output."""
        console.add_output("Message 1")
        console.add_output("Message 2")
        console.clear()

        assert len(console.get_output()) == 0

    def test_output_limit(self):
        """Test output is limited to max_output."""
        console = Console(max_output=10)

        for i in range(20):
            console.add_output(f"Message {i}")

        output = console.get_output()
        assert len(output) == 10

    def test_get_output_with_limit(self, console):
        """Test getting limited output lines."""
        for i in range(10):
            console.add_output(f"Line {i}")

        output = console.get_output(limit=3)
        assert len(output) == 3

    def test_output_callback(self, console):
        """Test output callback."""
        received = []

        def callback(output):
            received.append(output)

        console.on_output(callback)
        console.add_output("Test")

        assert len(received) == 1
        assert received[0].text == "Test"

    def test_clear_callback(self, console):
        """Test clear callback."""
        cleared = {"called": False}

        def callback():
            cleared["called"] = True

        console.on_clear(callback)
        console.clear()

        assert cleared["called"]


class TestConsoleCheats:
    """Test cheat mode handling."""

    def test_cheats_default_disabled(self, console):
        """Test cheats are disabled by default."""
        assert not console.cheats_enabled

    def test_enable_cheats(self, console):
        """Test enabling cheats."""
        console.cheats_enabled = True

        assert console.cheats_enabled
        # Check output message
        output = console.get_output()
        assert any("enabled" in o.text.lower() for o in output)

    def test_cheats_propagate_to_cvar_registry(self, console):
        """Test that cheats state propagates to CVarRegistry."""
        console.cheats_enabled = True

        assert CVarRegistry.instance().cheats_enabled

    def test_cheat_cvar_with_cheats_enabled(self, console):
        """Test cheat CVar access with cheats enabled."""
        cvar = CVar("cheat.test", default=False, flags=CVarFlags.CHEAT)
        console.cheats_enabled = True

        result = console.execute("cheat.test true")
        assert cvar.value is True


class TestConsoleStats:
    """Test console statistics."""

    def test_get_stats(self, console):
        """Test getting console stats."""
        console.execute("echo test1")
        console.execute("echo test2")
        console.register_alias("test", "echo test")

        stats = console.get_stats()

        assert "mode" in stats
        assert stats["history_count"] == 2
        assert stats["alias_count"] == 1


class TestConsoleOutputDataclass:
    """Test ConsoleOutput dataclass."""

    def test_console_output_defaults(self):
        """Test ConsoleOutput default values."""
        output = ConsoleOutput(text="Test")

        assert output.text == "Test"
        assert output.level == "info"
        assert output.timestamp > 0

    def test_console_output_custom_level(self):
        """Test ConsoleOutput with custom level."""
        output = ConsoleOutput(text="Error!", level="error")

        assert output.level == "error"
