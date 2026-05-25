"""Tests for Command system - registration and execution.

Tests cover:
- Command registration with handlers
- Command execution with arguments
- Built-in commands (help, list, clear, etc.)
- Command flags (CHEAT, HIDDEN, BUILTIN)
"""

import pytest

from engine.debug.console.commands import (
    Command,
    CommandAccessError,
    CommandExecutionError,
    CommandFlags,
    CommandNotFoundError,
    CommandRegistry,
)


@pytest.fixture
def registry():
    """Create a fresh CommandRegistry without builtins for testing."""
    return CommandRegistry(register_builtins=False)


@pytest.fixture
def registry_with_builtins():
    """Create a CommandRegistry with built-in commands."""
    return CommandRegistry(register_builtins=True)


class TestCommandRegistration:
    """Test command registration."""

    def test_register_simple_command(self, registry):
        """Test registering a simple command."""
        cmd = registry.register(
            name="ping",
            handler=lambda: "pong",
            description="Simple ping command"
        )

        assert cmd.name == "ping"
        assert cmd.description == "Simple ping command"
        assert registry.get("ping") is cmd

    def test_register_command_with_args(self, registry):
        """Test registering a command with arguments."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        cmd = registry.register(
            name="greet",
            handler=greet,
            min_args=1,
            max_args=2,
            usage="greet <name> [greeting]"
        )

        assert cmd.min_args == 1
        assert cmd.max_args == 2

    def test_register_duplicate_error(self, registry):
        """Test that duplicate registration raises ValueError."""
        registry.register("test", lambda: None)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", lambda: None)

    def test_register_with_flags(self, registry):
        """Test registering with various flags."""
        registry.register(
            "cheat_cmd",
            lambda: "cheated",
            flags=CommandFlags.CHEAT
        )
        registry.register(
            "hidden_cmd",
            lambda: "hidden",
            flags=CommandFlags.HIDDEN
        )
        registry.register(
            "combined_cmd",
            lambda: "combined",
            flags=CommandFlags.CHEAT | CommandFlags.HIDDEN
        )

        cmd = registry.get("combined_cmd")
        assert CommandFlags.CHEAT in cmd.flags
        assert CommandFlags.HIDDEN in cmd.flags

    def test_unregister_command(self, registry):
        """Test unregistering a command."""
        registry.register("temp", lambda: None)
        assert "temp" in registry

        assert registry.unregister("temp")
        assert "temp" not in registry
        assert not registry.unregister("temp")  # Already gone

    def test_empty_name_error(self):
        """Test that empty command name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Command(name="", handler=lambda: None)

    def test_non_callable_handler_error(self):
        """Test that non-callable handler raises ValueError."""
        with pytest.raises(ValueError, match="must be callable"):
            Command(name="test", handler="not a function")


class TestCommandExecution:
    """Test command execution."""

    def test_execute_no_args(self, registry):
        """Test executing a command without arguments."""
        registry.register("test", lambda: "result")

        result = registry.execute("test", [])
        assert result == "result"

    def test_execute_with_args(self, registry):
        """Test executing a command with arguments."""
        registry.register("add", lambda a, b: int(a) + int(b))

        result = registry.execute("add", ["10", "20"])
        assert result == 30

    def test_execute_unknown_command(self, registry):
        """Test executing an unknown command raises error."""
        with pytest.raises(CommandNotFoundError, match="Unknown command"):
            registry.execute("nonexistent", [])

    def test_execute_cheat_without_cheats(self, registry):
        """Test that cheat commands are blocked without cheats."""
        registry.register("cheat", lambda: "cheated", flags=CommandFlags.CHEAT)
        registry.cheats_enabled = False

        with pytest.raises(CommandAccessError, match="requires cheats"):
            registry.execute("cheat", [])

    def test_execute_cheat_with_cheats(self, registry):
        """Test that cheat commands work with cheats enabled."""
        registry.register("cheat", lambda: "cheated", flags=CommandFlags.CHEAT)
        registry.cheats_enabled = True

        result = registry.execute("cheat", [])
        assert result == "cheated"

    def test_execute_too_few_args(self, registry):
        """Test execution with too few arguments."""
        registry.register(
            "need_args",
            lambda a, b: None,
            min_args=2
        )

        with pytest.raises(CommandExecutionError, match="requires at least"):
            registry.execute("need_args", ["one"])

    def test_execute_too_many_args(self, registry):
        """Test execution with too many arguments."""
        registry.register(
            "limited_args",
            lambda a: a,
            max_args=1
        )

        with pytest.raises(CommandExecutionError, match="at most"):
            registry.execute("limited_args", ["one", "two", "three"])

    def test_execute_handler_exception(self, registry):
        """Test that handler exceptions are wrapped."""
        def failing_handler():
            raise RuntimeError("Handler failed")

        registry.register("fail", failing_handler)

        with pytest.raises(CommandExecutionError, match="Error executing"):
            registry.execute("fail", [])

    def test_execute_with_return_none(self, registry):
        """Test commands that return None."""
        registry.register("silent", lambda: None)

        result = registry.execute("silent", [])
        assert result is None


class TestBuiltinCommands:
    """Test built-in commands."""

    def test_help_no_args(self, registry_with_builtins):
        """Test help command without arguments."""
        result = registry_with_builtins.execute("help", [])

        assert "Available commands:" in result
        assert "help" in result
        assert "list" in result

    def test_help_with_command(self, registry_with_builtins):
        """Test help command with specific command."""
        result = registry_with_builtins.execute("help", ["echo"])

        assert "Command: echo" in result
        assert "Print a message" in result

    def test_help_unknown_command(self, registry_with_builtins):
        """Test help for unknown command."""
        result = registry_with_builtins.execute("help", ["nonexistent"])

        assert "Unknown command" in result

    def test_list_command(self, registry_with_builtins):
        """Test list command."""
        result = registry_with_builtins.execute("list", [])

        assert "Commands:" in result
        assert "help" in result

    def test_list_with_filter(self, registry_with_builtins):
        """Test list command with filter."""
        registry_with_builtins.register("test_a", lambda: None)
        registry_with_builtins.register("test_b", lambda: None)

        result = registry_with_builtins.execute("list", ["test_*"])

        assert "test_a" in result
        assert "test_b" in result
        assert "help" not in result

    def test_echo_command(self, registry_with_builtins):
        """Test echo command."""
        result = registry_with_builtins.execute("echo", ["Hello", "World"])

        assert result == "Hello World"

    def test_clear_command(self, registry_with_builtins):
        """Test clear command returns marker."""
        result = registry_with_builtins.execute("clear", [])

        assert result == "__CLEAR__"

    def test_version_command(self, registry_with_builtins):
        """Test version command."""
        result = registry_with_builtins.execute("version", [])

        assert "Engine" in result
        assert "v" in result.lower()

    def test_quit_command(self, registry_with_builtins):
        """Test quit command returns marker."""
        result = registry_with_builtins.execute("quit", [])

        assert result == "__QUIT__"


class TestCommandUtilities:
    """Test command utility methods."""

    def test_all_returns_sorted(self, registry):
        """Test that all() returns sorted commands."""
        registry.register("zzz", lambda: None)
        registry.register("aaa", lambda: None)
        registry.register("mmm", lambda: None)

        commands = registry.all()
        names = [cmd.name for cmd in commands]

        assert names == sorted(names)

    def test_visible_excludes_hidden(self, registry):
        """Test that visible() excludes hidden commands."""
        registry.register("visible_cmd", lambda: None)
        registry.register("hidden_cmd", lambda: None, flags=CommandFlags.HIDDEN)

        visible = registry.visible()
        names = [cmd.name for cmd in visible]

        assert "visible_cmd" in names
        assert "hidden_cmd" not in names

    def test_get_completions(self, registry):
        """Test command name completions."""
        registry.register("teleport", lambda: None)
        registry.register("tell", lambda: None)
        registry.register("test", lambda: None)
        registry.register("spawn", lambda: None)

        completions = registry.get_completions("te")

        assert "teleport" in completions
        assert "tell" in completions
        assert "test" in completions
        assert "spawn" not in completions

    def test_get_completions_excludes_hidden(self, registry):
        """Test that completions exclude hidden commands."""
        registry.register("visible", lambda: None)
        registry.register("hidden", lambda: None, flags=CommandFlags.HIDDEN)

        completions = registry.get_completions("")

        assert "visible" in completions
        assert "hidden" not in completions

    def test_len_and_contains(self, registry):
        """Test len and contains operations."""
        registry.register("cmd1", lambda: None)
        registry.register("cmd2", lambda: None)

        assert len(registry) == 2
        assert "cmd1" in registry
        assert "nonexistent" not in registry


class TestCommandHelp:
    """Test command help formatting."""

    def test_get_help_basic(self):
        """Test basic help formatting."""
        cmd = Command(
            name="test",
            handler=lambda: None,
            description="A test command"
        )

        help_text = cmd.get_help()
        assert "Command: test" in help_text
        assert "A test command" in help_text

    def test_get_help_with_usage(self):
        """Test help with usage string."""
        cmd = Command(
            name="spawn",
            handler=lambda: None,
            description="Spawn an actor",
            usage="spawn <actor_type> [x] [y] [z]"
        )

        help_text = cmd.get_help()
        assert "Usage: spawn <actor_type>" in help_text

    def test_get_help_with_flags(self):
        """Test help shows flags."""
        cmd = Command(
            name="godmode",
            handler=lambda: None,
            flags=CommandFlags.CHEAT
        )

        help_text = cmd.get_help()
        assert "CHEAT" in help_text
