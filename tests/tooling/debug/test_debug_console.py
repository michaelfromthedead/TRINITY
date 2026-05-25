"""Tests for debug console - command parsing, execution."""

import pytest
from engine.tooling.debug.debug_console import (
    DebugConsole,
    ConsoleCommand,
    CommandArg,
    CommandCategory,
    CommandResult,
    CommandExecutionResult,
    cheat,
)


class TestCommandArg:
    """Tests for CommandArg class."""

    def test_arg_creation(self):
        arg = CommandArg(
            name="value",
            arg_type=int,
            description="A numeric value",
            required=True,
        )
        assert arg.name == "value"
        assert arg.arg_type == int
        assert arg.required is True

    def test_arg_with_default(self):
        arg = CommandArg(
            name="count",
            arg_type=int,
            required=False,
            default=10,
        )
        assert arg.default == 10


class TestConsoleCommand:
    """Tests for ConsoleCommand class."""

    def test_command_creation(self):
        def test_func():
            return "test"

        cmd = ConsoleCommand(
            name="test",
            callback=test_func,
            description="A test command",
            category=CommandCategory.DEBUG,
        )
        assert cmd.name == "test"
        assert cmd.category == CommandCategory.DEBUG

    def test_command_usage(self):
        cmd = ConsoleCommand(
            name="give",
            callback=lambda: None,
            args=[
                CommandArg("item", str, required=True),
                CommandArg("count", int, required=False),
            ],
        )
        usage = cmd.get_usage()
        assert "give" in usage
        assert "<item>" in usage
        assert "[count]" in usage


class TestDebugConsole:
    """Tests for DebugConsole class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        DebugConsole.reset_instance()
        yield
        DebugConsole.reset_instance()

    def test_singleton(self):
        c1 = DebugConsole.get_instance()
        c2 = DebugConsole.get_instance()
        assert c1 is c2

    def test_enable_disable(self):
        console = DebugConsole.get_instance()
        console.enable()
        assert console.is_enabled
        console.disable()
        assert not console.is_enabled

    def test_show_hide(self):
        console = DebugConsole.get_instance()
        console.show()
        assert console.is_visible
        console.hide()
        assert not console.is_visible

    def test_toggle(self):
        console = DebugConsole.get_instance()
        console.hide()
        result = console.toggle()
        assert result is True
        assert console.is_visible
        result = console.toggle()
        assert result is False

    def test_register_command(self):
        console = DebugConsole.get_instance()
        cmd = ConsoleCommand(
            name="mycommand",
            callback=lambda: "result",
            description="My command",
        )
        console.register_command(cmd)
        assert console.get_command("mycommand") is cmd

    def test_unregister_command(self):
        console = DebugConsole.get_instance()
        cmd = ConsoleCommand(
            name="mycommand",
            callback=lambda: "result",
        )
        console.register_command(cmd)
        result = console.unregister_command("mycommand")
        assert result is True
        assert console.get_command("mycommand") is None

    def test_command_aliases(self):
        console = DebugConsole.get_instance()
        cmd = ConsoleCommand(
            name="mycommand",
            callback=lambda: "result",
            aliases=["mc", "mycmd"],
        )
        console.register_command(cmd)
        assert console.get_command("mc") is cmd
        assert console.get_command("mycmd") is cmd

    def test_execute_simple_command(self):
        console = DebugConsole.get_instance()
        result = console.execute("echo Hello World")
        assert result.status == CommandResult.SUCCESS
        assert "Hello World" in result.message

    def test_execute_command_with_args(self):
        console = DebugConsole.get_instance()
        executed = [False]

        def my_cmd(name: str, count: int):
            executed[0] = True
            return f"name={name}, count={count}"

        console.register_command(ConsoleCommand(
            name="mycmd",
            callback=my_cmd,
            args=[
                CommandArg("name", str, required=True),
                CommandArg("count", int, required=True),
            ],
        ))

        result = console.execute("mycmd test 42")
        assert result.status == CommandResult.SUCCESS
        assert executed[0] is True
        assert "name=test" in result.message

    def test_execute_unknown_command(self):
        console = DebugConsole.get_instance()
        result = console.execute("unknown_command")
        assert result.status == CommandResult.NOT_FOUND

    def test_execute_missing_args(self):
        console = DebugConsole.get_instance()
        console.register_command(ConsoleCommand(
            name="mycmd",
            callback=lambda x: x,
            args=[CommandArg("required_arg", str, required=True)],
        ))

        result = console.execute("mycmd")
        assert result.status == CommandResult.INVALID_ARGS

    def test_execute_invalid_arg_type(self):
        console = DebugConsole.get_instance()
        console.register_command(ConsoleCommand(
            name="mycmd",
            callback=lambda x: x,
            args=[CommandArg("number", int, required=True)],
        ))

        result = console.execute("mycmd not_a_number")
        assert result.status == CommandResult.INVALID_ARGS

    def test_execute_optional_args(self):
        console = DebugConsole.get_instance()

        def my_cmd(required: str, optional: str = "default"):
            return f"{required},{optional}"

        console.register_command(ConsoleCommand(
            name="mycmd",
            callback=my_cmd,
            args=[
                CommandArg("required", str, required=True),
                CommandArg("optional", str, required=False, default="default"),
            ],
        ))

        result = console.execute("mycmd value1")
        assert result.status == CommandResult.SUCCESS
        assert "value1,default" in result.message

    def test_execute_disabled(self):
        console = DebugConsole.get_instance()
        console.disable()
        result = console.execute("echo test")
        assert result.status == CommandResult.ERROR

    def test_execute_empty_command(self):
        console = DebugConsole.get_instance()
        result = console.execute("")
        assert result.status == CommandResult.ERROR

    def test_execute_command_error(self):
        console = DebugConsole.get_instance()

        def failing_cmd():
            raise RuntimeError("Command failed")

        console.register_command(ConsoleCommand(
            name="fail",
            callback=failing_cmd,
        ))

        result = console.execute("fail")
        assert result.status == CommandResult.ERROR
        assert "Command failed" in result.message

    def test_builtin_help_command(self):
        console = DebugConsole.get_instance()
        result = console.execute("help echo")
        assert result.status == CommandResult.SUCCESS
        assert "echo" in result.message.lower()

    def test_builtin_clear_command(self):
        console = DebugConsole.get_instance()
        console.print("some output")
        result = console.execute("clear")
        assert result.status == CommandResult.SUCCESS

    def test_builtin_history_command(self):
        console = DebugConsole.get_instance()
        console.execute("echo test1")
        console.execute("echo test2")
        result = console.execute("history")
        assert result.status == CommandResult.SUCCESS

    def test_builtin_set_get_commands(self):
        console = DebugConsole.get_instance()
        result = console.execute("set myvar 42")
        assert result.status == CommandResult.SUCCESS

        result = console.execute("get myvar")
        assert result.status == CommandResult.SUCCESS
        assert "42" in result.message

    def test_builtin_commands_list(self):
        console = DebugConsole.get_instance()
        result = console.execute("commands")
        assert result.status == CommandResult.SUCCESS
        # Should list available commands

    def test_cheats_disabled(self):
        console = DebugConsole.get_instance()
        console.register_command(ConsoleCommand(
            name="godmode",
            callback=lambda: "god mode enabled",
            is_cheat=True,
        ))

        result = console.execute("godmode")
        assert result.status == CommandResult.PERMISSION_DENIED

    def test_cheats_enabled(self):
        console = DebugConsole.get_instance()
        console.execute("sv_cheats 1")

        console.register_command(ConsoleCommand(
            name="godmode",
            callback=lambda: "god mode enabled",
            is_cheat=True,
        ))

        result = console.execute("godmode")
        assert result.status == CommandResult.SUCCESS

    def test_permission_level(self):
        console = DebugConsole.get_instance()
        console.register_command(ConsoleCommand(
            name="admin_cmd",
            callback=lambda: "admin",
            min_permission_level=5,
        ))

        result = console.execute("admin_cmd")
        assert result.status == CommandResult.PERMISSION_DENIED

        console.set_permission_level(5)
        result = console.execute("admin_cmd")
        assert result.status == CommandResult.SUCCESS

    def test_command_history(self):
        console = DebugConsole.get_instance()
        console.execute("echo test1")
        console.execute("echo test2")

        history = console.get_history()
        assert len(history) >= 2

    def test_print_output(self):
        console = DebugConsole.get_instance()
        console.print("Normal message")
        console.print_error("Error message")
        console.print_warning("Warning message")

        output = console.get_output()
        assert len(output) >= 3

    def test_output_callback(self):
        console = DebugConsole.get_instance()
        received = []

        def on_output(text, output_type):
            received.append((text, output_type))

        console.on_output(on_output)
        console.print("test message")

        assert len(received) == 1
        assert received[0][0] == "test message"

    def test_autocomplete(self):
        console = DebugConsole.get_instance()
        suggestions = console.get_autocomplete("ec")
        assert "echo" in suggestions

    def test_variables(self):
        console = DebugConsole.get_instance()
        console.set_variable("test_var", 123)
        assert console.get_variable("test_var") == 123
        assert console.get_variable("nonexistent", "default") == "default"

    def test_commands_by_category(self):
        console = DebugConsole.get_instance()
        system_cmds = console.get_commands_by_category(CommandCategory.SYSTEM)
        assert len(system_cmds) > 0

    def test_quoted_args(self):
        console = DebugConsole.get_instance()
        result = console.execute('echo "Hello World"')
        assert result.status == CommandResult.SUCCESS
        assert "Hello World" in result.message

    def test_execution_time(self):
        console = DebugConsole.get_instance()
        result = console.execute("echo test")
        assert result.execution_time >= 0


class TestCheatDecorator:
    """Tests for @cheat decorator."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        DebugConsole.reset_instance()
        yield
        DebugConsole.reset_instance()

    def test_cheat_decorator_registers_command(self):
        @cheat(name="test_cheat", description="Test cheat command")
        def test_cheat_func():
            return "cheated"

        console = DebugConsole.get_instance()
        cmd = console.get_command("test_cheat")
        assert cmd is not None
        assert cmd.is_cheat is True

    def test_cheat_decorator_with_args(self):
        @cheat(name="give_item")
        def give_item(item: str, count: int = 1):
            return f"Gave {count} {item}"

        console = DebugConsole.get_instance()
        console.execute("sv_cheats 1")
        result = console.execute("give_item sword 5")
        assert result.status == CommandResult.SUCCESS
        assert "5 sword" in result.message

    def test_cheat_decorator_auto_name(self):
        @cheat(description="Auto-named cheat")
        def my_cheat_command():
            return "auto"

        console = DebugConsole.get_instance()
        cmd = console.get_command("my_cheat_command")
        assert cmd is not None

    def test_cheat_decorator_preserves_function(self):
        @cheat(name="preserved")
        def preserved_func():
            """Original docstring."""
            return "preserved"

        # Can still call directly
        assert preserved_func() == "preserved"

    def test_cheat_decorator_with_aliases(self):
        @cheat(name="noclip", aliases=["nc", "fly"])
        def noclip_mode():
            return "noclip enabled"

        console = DebugConsole.get_instance()
        console.execute("sv_cheats 1")

        result = console.execute("nc")
        assert result.status == CommandResult.SUCCESS

    def test_cheat_blocked_without_sv_cheats(self):
        @cheat(name="blocked_cheat")
        def blocked():
            return "should not run"

        console = DebugConsole.get_instance()
        # Don't enable cheats
        result = console.execute("blocked_cheat")
        assert result.status == CommandResult.PERMISSION_DENIED
