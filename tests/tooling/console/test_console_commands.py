"""Tests for the console command system.

Tests command registration, parsing, execution, and decorators.
"""

import pytest

from engine.tooling.console.console_commands import (
    CommandRegistry,
    Command,
    CommandResult,
    CommandContext,
    CommandArg,
    CommandStatus,
    PermissionLevel,
    command,
    cheat,
    admin,
    developer,
)


class TestCommandArg:
    """Tests for CommandArg validation."""

    def test_validate_required_missing(self):
        arg = CommandArg("name", str, required=True)
        is_valid, value, error = arg.validate(None)
        assert not is_valid
        assert "Missing required" in error

    def test_validate_optional_missing(self):
        arg = CommandArg("name", str, default="default", required=False)
        is_valid, value, error = arg.validate(None)
        assert is_valid
        assert value == "default"

    def test_validate_int_conversion(self):
        arg = CommandArg("count", int)
        is_valid, value, error = arg.validate("42")
        assert is_valid
        assert value == 42

    def test_validate_float_conversion(self):
        arg = CommandArg("amount", float)
        is_valid, value, error = arg.validate("3.14")
        assert is_valid
        assert value == pytest.approx(3.14)

    def test_validate_bool_conversion(self):
        arg = CommandArg("flag", bool)
        is_valid, value, error = arg.validate("true")
        assert is_valid
        assert value is True

        is_valid, value, error = arg.validate("false")
        assert is_valid
        assert value is False

    def test_validate_invalid_type(self):
        arg = CommandArg("count", int)
        is_valid, value, error = arg.validate("not_a_number")
        assert not is_valid
        assert "Invalid value" in error

    def test_validate_choices(self):
        arg = CommandArg("mode", str, choices=["easy", "medium", "hard"])
        is_valid, value, error = arg.validate("medium")
        assert is_valid
        assert value == "medium"

        is_valid, value, error = arg.validate("impossible")
        assert not is_valid
        assert "not valid" in error


class TestCommand:
    """Tests for Command class."""

    def test_basic_command(self):
        def handler(x: int) -> str:
            return f"Got {x}"

        cmd = Command(name="test", handler=handler, description="Test command")
        assert cmd.name == "test"
        assert cmd.description == "Test command"
        assert cmd.permission == PermissionLevel.USER

    def test_auto_extract_args(self):
        def handler(name: str, count: int = 1) -> None:
            pass

        cmd = Command(name="test", handler=handler)
        assert len(cmd.args) == 2
        assert cmd.args[0].name == "name"
        assert cmd.args[0].required is True
        assert cmd.args[1].name == "count"
        assert cmd.args[1].required is False
        assert cmd.args[1].default == 1

    def test_get_usage(self):
        def handler(name: str, count: int = 1) -> None:
            pass

        cmd = Command(name="test", handler=handler)
        usage = cmd.get_usage()
        assert "test" in usage
        assert "<name>" in usage
        assert "[count]" in usage

    def test_get_help(self):
        def handler(name: str) -> None:
            pass

        cmd = Command(
            name="test",
            handler=handler,
            description="Test description",
            aliases=("t", "tst")
        )
        help_text = cmd.get_help()
        assert "test" in help_text
        assert "Test description" in help_text
        assert "Aliases" in help_text
        assert "t, tst" in help_text


class TestCommandResult:
    """Tests for CommandResult."""

    def test_ok_result(self):
        result = CommandResult.ok("Success", return_value=42)
        assert result.success is True
        assert result.status == CommandStatus.SUCCESS
        assert result.message == "Success"
        assert result.return_value == 42

    def test_error_result(self):
        result = CommandResult.error("Something failed")
        assert result.success is False
        assert result.status == CommandStatus.ERROR
        assert result.message == "Something failed"

    def test_not_found_result(self):
        result = CommandResult.not_found("foo")
        assert result.status == CommandStatus.NOT_FOUND
        assert "foo" in result.message

    def test_permission_denied_result(self):
        result = CommandResult.permission_denied("secret")
        assert result.status == CommandStatus.PERMISSION_DENIED
        assert "secret" in result.message

    def test_invalid_args_result(self):
        result = CommandResult.invalid_args("Missing argument")
        assert result.status == CommandStatus.INVALID_ARGS


class TestCommandContext:
    """Tests for CommandContext."""

    def test_default_context(self):
        ctx = CommandContext()
        assert ctx.permission_level == PermissionLevel.USER
        assert ctx.cheats_enabled is False
        assert ctx.is_server is False
        assert ctx.source == "console"

    def test_custom_context(self):
        ctx = CommandContext(
            permission_level=PermissionLevel.ADMIN,
            cheats_enabled=True,
            user_id="user123"
        )
        assert ctx.permission_level == PermissionLevel.ADMIN
        assert ctx.cheats_enabled is True
        assert ctx.user_id == "user123"


class TestCommandRegistry:
    """Tests for CommandRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before and after each test."""
        CommandRegistry.reset_instance()
        yield
        CommandRegistry.reset_instance()

    def test_register_command(self):
        registry = CommandRegistry()

        def handler():
            pass

        cmd = Command(name="test", handler=handler)
        registry.register(cmd)

        assert registry.has_command("test")

    def test_register_duplicate_raises(self):
        registry = CommandRegistry()

        def handler():
            pass

        cmd1 = Command(name="test", handler=handler)
        cmd2 = Command(name="test", handler=handler)

        registry.register(cmd1)
        with pytest.raises(ValueError, match="already exists"):
            registry.register(cmd2)

    def test_register_alias_conflict_raises(self):
        registry = CommandRegistry()

        def handler():
            pass

        cmd1 = Command(name="test1", handler=handler, aliases=("t",))
        cmd2 = Command(name="test2", handler=handler, aliases=("t",))

        registry.register(cmd1)
        with pytest.raises(ValueError, match="conflicts"):
            registry.register(cmd2)

    def test_get_command(self):
        registry = CommandRegistry()

        def handler():
            return "hello"

        cmd = Command(name="test", handler=handler)
        registry.register(cmd)

        retrieved = registry.get("test")
        assert retrieved is cmd

    def test_get_by_alias(self):
        registry = CommandRegistry()

        def handler():
            pass

        cmd = Command(name="test", handler=handler, aliases=("t",))
        registry.register(cmd)

        assert registry.get("t") is cmd

    def test_case_insensitive(self):
        registry = CommandRegistry()

        def handler():
            pass

        cmd = Command(name="Test", handler=handler)
        registry.register(cmd)

        assert registry.get("test") is cmd
        assert registry.get("TEST") is cmd

    def test_unregister(self):
        registry = CommandRegistry()

        def handler():
            pass

        cmd = Command(name="test", handler=handler, aliases=("t",))
        registry.register(cmd)

        removed = registry.unregister("test")
        assert removed is cmd
        assert registry.get("test") is None
        assert registry.get("t") is None

    def test_all_commands(self):
        registry = CommandRegistry()

        def handler():
            pass

        cmd1 = Command(name="cmd1", handler=handler)
        cmd2 = Command(name="cmd2", handler=handler)

        registry.register(cmd1)
        registry.register(cmd2)

        all_cmds = registry.all_commands()
        assert len(all_cmds) == 2

    def test_by_category(self):
        registry = CommandRegistry()

        def handler():
            pass

        cmd1 = Command(name="cmd1", handler=handler, category="debug")
        cmd2 = Command(name="cmd2", handler=handler, category="debug")
        cmd3 = Command(name="cmd3", handler=handler, category="game")

        registry.register(cmd1)
        registry.register(cmd2)
        registry.register(cmd3)

        debug_cmds = registry.by_category("debug")
        assert len(debug_cmds) == 2

    def test_find_pattern(self):
        registry = CommandRegistry()

        def handler():
            pass

        registry.register(Command(name="sv_cheats", handler=handler))
        registry.register(Command(name="sv_gravity", handler=handler))
        registry.register(Command(name="cl_fov", handler=handler))

        matches = registry.find("sv_*")
        assert len(matches) == 2

    def test_parse_command_line(self):
        registry = CommandRegistry()

        cmd, args = registry.parse_command_line("test arg1 arg2")
        assert cmd == "test"
        assert args == ["arg1", "arg2"]

    def test_parse_command_line_quoted(self):
        registry = CommandRegistry()

        cmd, args = registry.parse_command_line('say "hello world" player1')
        assert cmd == "say"
        assert args == ["hello world", "player1"]

    def test_parse_empty_command(self):
        registry = CommandRegistry()

        cmd, args = registry.parse_command_line("")
        assert cmd == ""
        assert args == []


class TestCommandExecution:
    """Tests for command execution."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        CommandRegistry.reset_instance()
        yield
        CommandRegistry.reset_instance()

    def test_execute_simple(self):
        registry = CommandRegistry()

        def handler():
            return "success"

        cmd = Command(name="test", handler=handler)
        registry.register(cmd)

        result = registry.execute("test")
        assert result.success
        assert result.return_value == "success"

    def test_execute_with_args(self):
        registry = CommandRegistry()

        def handler(name: str, count: int):
            return f"{name}: {count}"

        cmd = Command(name="test", handler=handler)
        registry.register(cmd)

        result = registry.execute("test hello 42")
        assert result.success
        assert result.return_value == "hello: 42"

    def test_execute_with_context(self):
        registry = CommandRegistry()

        def handler(ctx: CommandContext, message: str):
            return f"User {ctx.user_id}: {message}"

        cmd = Command(name="test", handler=handler)
        registry.register(cmd)

        ctx = CommandContext(user_id="player1")
        result = registry.execute("test hello", ctx)
        assert result.success
        assert "player1" in result.return_value

    def test_execute_not_found(self):
        registry = CommandRegistry()
        result = registry.execute("nonexistent")
        assert result.status == CommandStatus.NOT_FOUND

    def test_execute_missing_args(self):
        registry = CommandRegistry()

        def handler(name: str, count: int):
            pass

        cmd = Command(name="test", handler=handler)
        registry.register(cmd)

        result = registry.execute("test")
        assert result.status == CommandStatus.INVALID_ARGS

    def test_execute_exception(self):
        registry = CommandRegistry()

        def handler():
            raise ValueError("Test error")

        cmd = Command(name="test", handler=handler)
        registry.register(cmd)

        result = registry.execute("test")
        assert result.status == CommandStatus.ERROR
        assert "Test error" in result.message

    def test_execute_returns_command_result(self):
        registry = CommandRegistry()

        def handler():
            return CommandResult.error("Custom error")

        cmd = Command(name="test", handler=handler)
        registry.register(cmd)

        result = registry.execute("test")
        assert result.status == CommandStatus.ERROR
        assert result.message == "Custom error"

    def test_execution_time_tracked(self):
        import time
        registry = CommandRegistry()

        def handler():
            time.sleep(0.01)
            return "done"

        cmd = Command(name="test", handler=handler)
        registry.register(cmd)

        result = registry.execute("test")
        assert result.execution_time > 0


class TestCommandPermissions:
    """Tests for command permission checking."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        CommandRegistry.reset_instance()
        yield
        CommandRegistry.reset_instance()

    def test_user_command(self):
        registry = CommandRegistry()

        def handler():
            return "ok"

        cmd = Command(name="test", handler=handler, permission=PermissionLevel.USER)
        registry.register(cmd)

        ctx = CommandContext(permission_level=PermissionLevel.USER)
        result = registry.execute("test", ctx)
        assert result.success

    def test_developer_command_denied(self):
        registry = CommandRegistry()

        def handler():
            return "ok"

        cmd = Command(name="debug", handler=handler, permission=PermissionLevel.DEVELOPER)
        registry.register(cmd)

        ctx = CommandContext(permission_level=PermissionLevel.USER)
        result = registry.execute("debug", ctx)
        assert result.status == CommandStatus.PERMISSION_DENIED

    def test_developer_command_allowed(self):
        registry = CommandRegistry()

        def handler():
            return "ok"

        cmd = Command(name="debug", handler=handler, permission=PermissionLevel.DEVELOPER)
        registry.register(cmd)

        ctx = CommandContext(permission_level=PermissionLevel.DEVELOPER)
        result = registry.execute("debug", ctx)
        assert result.success

    def test_cheat_command_requires_cheats_enabled(self):
        registry = CommandRegistry()

        def handler():
            return "ok"

        cmd = Command(name="god", handler=handler, permission=PermissionLevel.CHEAT)
        registry.register(cmd)

        # Developer without cheats enabled
        ctx = CommandContext(
            permission_level=PermissionLevel.DEVELOPER,
            cheats_enabled=False
        )
        result = registry.execute("god", ctx)
        assert result.status == CommandStatus.PERMISSION_DENIED

        # Developer with cheats enabled
        ctx = CommandContext(
            permission_level=PermissionLevel.DEVELOPER,
            cheats_enabled=True
        )
        result = registry.execute("god", ctx)
        assert result.success

    def test_admin_command(self):
        registry = CommandRegistry()

        def handler():
            return "ok"

        cmd = Command(name="kick", handler=handler, permission=PermissionLevel.ADMIN)
        registry.register(cmd)

        ctx = CommandContext(permission_level=PermissionLevel.DEVELOPER)
        result = registry.execute("kick", ctx)
        assert result.status == CommandStatus.PERMISSION_DENIED

        ctx = CommandContext(permission_level=PermissionLevel.ADMIN)
        result = registry.execute("kick", ctx)
        assert result.success


class TestCommandCompletions:
    """Tests for command autocompletion."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        CommandRegistry.reset_instance()
        yield
        CommandRegistry.reset_instance()

    def test_get_completions(self):
        registry = CommandRegistry()

        def handler():
            pass

        registry.register(Command(name="test", handler=handler))
        registry.register(Command(name="teleport", handler=handler))
        registry.register(Command(name="give", handler=handler))

        completions = registry.get_completions("te")
        assert "test" in completions
        assert "teleport" in completions
        assert "give" not in completions

    def test_completions_include_aliases(self):
        registry = CommandRegistry()

        def handler():
            pass

        registry.register(Command(name="teleport", handler=handler, aliases=("tp",)))

        completions = registry.get_completions("t")
        assert "teleport" in completions
        assert "tp" in completions

    def test_hidden_commands_excluded(self):
        registry = CommandRegistry()

        def handler():
            pass

        registry.register(Command(name="test", handler=handler))
        registry.register(Command(name="hidden", handler=handler, hidden=True))

        completions = registry.get_completions("")
        assert "test" in completions
        assert "hidden" not in completions

    def test_completions_respect_permissions(self):
        registry = CommandRegistry()

        def handler():
            pass

        registry.register(Command(name="public", handler=handler))
        registry.register(Command(
            name="admin_only",
            handler=handler,
            permission=PermissionLevel.ADMIN
        ))

        ctx = CommandContext(permission_level=PermissionLevel.USER)
        completions = registry.get_completions("", ctx)
        assert "public" in completions
        assert "admin_only" not in completions


class TestCommandDecorators:
    """Tests for command decorator functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        CommandRegistry.reset_instance()
        yield
        CommandRegistry.reset_instance()

    def test_command_decorator(self):
        @command(name="greet", description="Say hello")
        def greet_handler(name: str):
            return f"Hello, {name}!"

        assert hasattr(greet_handler, '_console_command')
        cmd = greet_handler._console_command
        assert cmd.name == "greet"
        assert cmd.description == "Say hello"
        assert cmd.permission == PermissionLevel.USER

    def test_command_decorator_auto_name(self):
        @command()
        def my_command():
            return "ok"

        cmd = my_command._console_command
        assert cmd.name == "my_command"

    def test_cheat_decorator(self):
        @cheat(name="god_mode", description="Toggle god mode")
        def god_mode_handler():
            return "God mode enabled"

        cmd = god_mode_handler._console_command
        assert cmd.name == "god_mode"
        assert cmd.permission == PermissionLevel.CHEAT
        assert cmd.category == "cheats"

    def test_admin_decorator(self):
        @admin(name="kick", description="Kick a player")
        def kick_handler(player: str):
            return f"Kicked {player}"

        cmd = kick_handler._console_command
        assert cmd.name == "kick"
        assert cmd.permission == PermissionLevel.ADMIN
        assert cmd.category == "admin"
        assert cmd.requires_confirmation is True

    def test_developer_decorator(self):
        @developer(name="debug_info", description="Show debug info")
        def debug_info_handler():
            return "Debug info"

        cmd = debug_info_handler._console_command
        assert cmd.name == "debug_info"
        assert cmd.permission == PermissionLevel.DEVELOPER
        assert cmd.hidden is True

    def test_decorator_with_aliases(self):
        @command(name="teleport", aliases=("tp", "goto"))
        def teleport_handler():
            pass

        cmd = teleport_handler._console_command
        assert cmd.aliases == ("tp", "goto")

    def test_decorator_auto_registers(self):
        registry = CommandRegistry.get_instance()

        @command(name="unique_test_cmd")
        def test_handler():
            return "ok"

        assert registry.has_command("unique_test_cmd")

    def test_decorated_function_still_callable(self):
        @command(name="calc")
        def calc_handler(a: int, b: int) -> int:
            return a + b

        result = calc_handler(3, 4)
        assert result == 7
