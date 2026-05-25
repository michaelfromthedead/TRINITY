"""
Debug Console - In-game debug console with command execution.

Provides a command-line interface for runtime debugging, cheats,
and system manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Callable, ClassVar, Optional, Any
import threading
import time
import re
import shlex


class CommandCategory(Enum):
    """Categories for console commands."""
    SYSTEM = auto()
    GAMEPLAY = auto()
    RENDERING = auto()
    PHYSICS = auto()
    AI = auto()
    AUDIO = auto()
    DEBUG = auto()
    CHEAT = auto()
    CUSTOM = auto()


class CommandResult(Enum):
    """Result of command execution."""
    SUCCESS = auto()
    ERROR = auto()
    INVALID_ARGS = auto()
    NOT_FOUND = auto()
    PERMISSION_DENIED = auto()


@dataclass
class CommandArg:
    """Describes a command argument."""
    name: str
    arg_type: type
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass
class ConsoleCommand:
    """Represents a console command."""
    name: str
    callback: Callable[..., Any]
    description: str = ""
    category: CommandCategory = CommandCategory.CUSTOM
    args: list[CommandArg] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    is_cheat: bool = False
    min_permission_level: int = 0

    def get_usage(self) -> str:
        """Get usage string for command."""
        arg_strs = []
        for arg in self.args:
            if arg.required:
                arg_strs.append(f"<{arg.name}>")
            else:
                arg_strs.append(f"[{arg.name}]")

        args_str = " ".join(arg_strs)
        return f"{self.name} {args_str}".strip()


@dataclass
class CommandExecutionResult:
    """Result of executing a command."""
    status: CommandResult
    message: str = ""
    return_value: Any = None
    execution_time: float = 0.0


@dataclass
class ConsoleHistoryEntry:
    """An entry in console history."""
    command: str
    timestamp: float
    result: CommandExecutionResult


class DebugConsole:
    """In-game debug console with command execution."""

    _instance: ClassVar[Optional["DebugConsole"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    __slots__ = (
        '_commands',
        '_aliases',
        '_history',
        '_max_history',
        '_enabled',
        '_visible',
        '_cheats_enabled',
        '_permission_level',
        '_output_buffer',
        '_max_output_lines',
        '_on_output',
        '_variables',
    )

    def __init__(self, max_history: int = 100, max_output_lines: int = 500):
        self._commands: dict[str, ConsoleCommand] = {}
        self._aliases: dict[str, str] = {}
        self._history: list[ConsoleHistoryEntry] = []
        self._max_history = max_history
        self._enabled = True
        self._visible = False
        self._cheats_enabled = False
        self._permission_level = 0
        self._output_buffer: list[tuple[str, str]] = []  # (text, color/type)
        self._max_output_lines = max_output_lines
        self._on_output: list[Callable[[str, str], None]] = []
        self._variables: dict[str, Any] = {}

        self._register_builtin_commands()

    @classmethod
    def get_instance(cls) -> "DebugConsole":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _register_builtin_commands(self) -> None:
        """Register built-in commands."""
        # Help command
        self.register_command(
            ConsoleCommand(
                name="help",
                callback=self._cmd_help,
                description="Show available commands or help for a specific command",
                category=CommandCategory.SYSTEM,
                args=[
                    CommandArg("command", str, "Command to get help for", required=False),
                ],
            )
        )

        # Clear command
        self.register_command(
            ConsoleCommand(
                name="clear",
                callback=self._cmd_clear,
                description="Clear the console output",
                category=CommandCategory.SYSTEM,
            )
        )

        # History command
        self.register_command(
            ConsoleCommand(
                name="history",
                callback=self._cmd_history,
                description="Show command history",
                category=CommandCategory.SYSTEM,
                args=[
                    CommandArg("count", int, "Number of entries to show", required=False, default=10),
                ],
            )
        )

        # Echo command
        self.register_command(
            ConsoleCommand(
                name="echo",
                callback=self._cmd_echo,
                description="Print a message to the console",
                category=CommandCategory.SYSTEM,
                args=[],  # Echo handles raw args specially
            )
        )

        # Set variable command
        self.register_command(
            ConsoleCommand(
                name="set",
                callback=self._cmd_set,
                description="Set a console variable",
                category=CommandCategory.SYSTEM,
                args=[
                    CommandArg("name", str, "Variable name"),
                    CommandArg("value", str, "Value to set"),
                ],
            )
        )

        # Get variable command
        self.register_command(
            ConsoleCommand(
                name="get",
                callback=self._cmd_get,
                description="Get a console variable value",
                category=CommandCategory.SYSTEM,
                args=[
                    CommandArg("name", str, "Variable name"),
                ],
            )
        )

        # List commands
        self.register_command(
            ConsoleCommand(
                name="commands",
                callback=self._cmd_list_commands,
                description="List all commands, optionally filtered by category",
                category=CommandCategory.SYSTEM,
                args=[
                    CommandArg("category", str, "Category to filter by", required=False),
                ],
                aliases=["cmds"],
            )
        )

        # Enable cheats
        self.register_command(
            ConsoleCommand(
                name="sv_cheats",
                callback=self._cmd_sv_cheats,
                description="Enable or disable cheat commands",
                category=CommandCategory.SYSTEM,
                args=[
                    CommandArg("enabled", int, "1 to enable, 0 to disable"),
                ],
            )
        )

    def _cmd_help(self, command: Optional[str] = None) -> str:
        """Help command implementation."""
        if command:
            cmd = self._commands.get(command)
            if not cmd:
                return f"Unknown command: {command}"

            result = [
                f"Command: {cmd.name}",
                f"Description: {cmd.description}",
                f"Category: {cmd.category.name}",
                f"Usage: {cmd.get_usage()}",
            ]

            if cmd.aliases:
                result.append(f"Aliases: {', '.join(cmd.aliases)}")

            if cmd.args:
                result.append("Arguments:")
                for arg in cmd.args:
                    req = "required" if arg.required else "optional"
                    result.append(f"  {arg.name} ({arg.arg_type.__name__}, {req}): {arg.description}")

            return "\n".join(result)
        else:
            return "Type 'help <command>' for help on a specific command, or 'commands' to list all commands."

    def _cmd_clear(self) -> str:
        """Clear command implementation."""
        self._output_buffer.clear()
        return "Console cleared."

    def _cmd_history(self, count: int = 10) -> str:
        """History command implementation."""
        entries = self._history[-count:]
        if not entries:
            return "No command history."

        result = ["Command history:"]
        for i, entry in enumerate(entries, 1):
            status = "OK" if entry.result.status == CommandResult.SUCCESS else "ERR"
            result.append(f"  {i}. [{status}] {entry.command}")

        return "\n".join(result)

    def _cmd_echo(self, *args) -> str:
        """Echo command implementation."""
        return " ".join(str(a) for a in args)

    def _cmd_set(self, name: str, value: str) -> str:
        """Set variable command implementation."""
        # Try to parse value type
        try:
            if value.lower() in ("true", "false"):
                parsed_value = value.lower() == "true"
            elif "." in value:
                parsed_value = float(value)
            else:
                parsed_value = int(value)
        except ValueError:
            parsed_value = value

        self._variables[name] = parsed_value
        return f"{name} = {parsed_value}"

    def _cmd_get(self, name: str) -> str:
        """Get variable command implementation."""
        if name in self._variables:
            return f"{name} = {self._variables[name]}"
        return f"Variable '{name}' not found."

    def _cmd_list_commands(self, category: Optional[str] = None) -> str:
        """List commands implementation."""
        commands = list(self._commands.values())

        if category:
            try:
                cat_enum = CommandCategory[category.upper()]
                commands = [c for c in commands if c.category == cat_enum]
            except KeyError:
                return f"Unknown category: {category}"

        if not commands:
            return "No commands found."

        # Group by category
        by_category: dict[CommandCategory, list[ConsoleCommand]] = {}
        for cmd in commands:
            if cmd.category not in by_category:
                by_category[cmd.category] = []
            by_category[cmd.category].append(cmd)

        result = ["Available commands:"]
        for cat, cmds in sorted(by_category.items(), key=lambda x: x[0].name):
            result.append(f"\n[{cat.name}]")
            for cmd in sorted(cmds, key=lambda c: c.name):
                cheat_marker = " [CHEAT]" if cmd.is_cheat else ""
                result.append(f"  {cmd.name}{cheat_marker} - {cmd.description}")

        return "\n".join(result)

    def _cmd_sv_cheats(self, enabled: int) -> str:
        """Enable/disable cheats command implementation."""
        self._cheats_enabled = enabled == 1
        return f"Cheats {'enabled' if self._cheats_enabled else 'disabled'}."

    def enable(self) -> None:
        """Enable the console."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the console."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def show(self) -> None:
        """Show the console."""
        self._visible = True

    def hide(self) -> None:
        """Hide the console."""
        self._visible = False

    def toggle(self) -> bool:
        """Toggle console visibility. Returns new state."""
        self._visible = not self._visible
        return self._visible

    @property
    def is_visible(self) -> bool:
        return self._visible

    @property
    def cheats_enabled(self) -> bool:
        return self._cheats_enabled

    def set_permission_level(self, level: int) -> None:
        """Set the current permission level."""
        self._permission_level = level

    def register_command(self, command: ConsoleCommand) -> None:
        """Register a command."""
        self._commands[command.name] = command
        for alias in command.aliases:
            self._aliases[alias] = command.name

    def unregister_command(self, name: str) -> bool:
        """Unregister a command."""
        if name in self._commands:
            cmd = self._commands.pop(name)
            for alias in cmd.aliases:
                self._aliases.pop(alias, None)
            return True
        return False

    def get_command(self, name: str) -> Optional[ConsoleCommand]:
        """Get a command by name or alias."""
        if name in self._commands:
            return self._commands[name]
        if name in self._aliases:
            return self._commands.get(self._aliases[name])
        return None

    def execute(self, command_line: str) -> CommandExecutionResult:
        """Execute a command string."""
        if not self._enabled:
            return CommandExecutionResult(
                status=CommandResult.ERROR,
                message="Console is disabled.",
            )

        command_line = command_line.strip()
        if not command_line:
            return CommandExecutionResult(
                status=CommandResult.ERROR,
                message="Empty command.",
            )

        # Parse command and arguments
        try:
            parts = shlex.split(command_line)
        except ValueError as e:
            return CommandExecutionResult(
                status=CommandResult.ERROR,
                message=f"Parse error: {e}",
            )

        cmd_name = parts[0].lower()
        args = parts[1:]

        # Find command
        command = self.get_command(cmd_name)
        if not command:
            return CommandExecutionResult(
                status=CommandResult.NOT_FOUND,
                message=f"Unknown command: {cmd_name}",
            )

        # Check cheat status
        if command.is_cheat and not self._cheats_enabled:
            return CommandExecutionResult(
                status=CommandResult.PERMISSION_DENIED,
                message=f"Cheat commands are disabled. Use 'sv_cheats 1' to enable.",
            )

        # Check permission level
        if command.min_permission_level > self._permission_level:
            return CommandExecutionResult(
                status=CommandResult.PERMISSION_DENIED,
                message=f"Insufficient permission level.",
            )

        # Parse arguments
        parsed_args = []
        for i, arg_spec in enumerate(command.args):
            if i < len(args):
                try:
                    if arg_spec.arg_type == bool:
                        parsed_args.append(args[i].lower() in ("true", "1", "yes"))
                    else:
                        parsed_args.append(arg_spec.arg_type(args[i]))
                except (ValueError, TypeError) as e:
                    return CommandExecutionResult(
                        status=CommandResult.INVALID_ARGS,
                        message=f"Invalid argument '{arg_spec.name}': {e}",
                    )
            elif arg_spec.required:
                return CommandExecutionResult(
                    status=CommandResult.INVALID_ARGS,
                    message=f"Missing required argument: {arg_spec.name}",
                )
            else:
                parsed_args.append(arg_spec.default)

        # For commands with no defined args (like echo), pass all raw args
        if not command.args and args:
            parsed_args = args

        # Execute command
        start_time = time.time()
        try:
            result = command.callback(*parsed_args)
            execution_time = time.time() - start_time

            exec_result = CommandExecutionResult(
                status=CommandResult.SUCCESS,
                message=str(result) if result is not None else "",
                return_value=result,
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            exec_result = CommandExecutionResult(
                status=CommandResult.ERROR,
                message=f"Error: {e}",
                execution_time=execution_time,
            )

        # Add to history
        self._add_to_history(command_line, exec_result)

        # Output result
        if exec_result.message:
            self._add_output(exec_result.message, "output" if exec_result.status == CommandResult.SUCCESS else "error")

        return exec_result

    def _add_to_history(self, command: str, result: CommandExecutionResult) -> None:
        """Add entry to history."""
        entry = ConsoleHistoryEntry(
            command=command,
            timestamp=time.time(),
            result=result,
        )
        self._history.append(entry)

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def _add_output(self, text: str, output_type: str = "output") -> None:
        """Add text to output buffer."""
        self._output_buffer.append((text, output_type))

        # Trim output
        if len(self._output_buffer) > self._max_output_lines:
            self._output_buffer = self._output_buffer[-self._max_output_lines:]

        # Notify listeners
        for callback in self._on_output:
            callback(text, output_type)

    def print(self, text: str, output_type: str = "output") -> None:
        """Print text to the console."""
        self._add_output(text, output_type)

    def print_error(self, text: str) -> None:
        """Print error text to the console."""
        self._add_output(text, "error")

    def print_warning(self, text: str) -> None:
        """Print warning text to the console."""
        self._add_output(text, "warning")

    def on_output(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for console output."""
        self._on_output.append(callback)

    def get_output(self) -> list[tuple[str, str]]:
        """Get the output buffer."""
        return self._output_buffer.copy()

    def get_history(self) -> list[ConsoleHistoryEntry]:
        """Get command history."""
        return self._history.copy()

    def get_autocomplete(self, partial: str) -> list[str]:
        """Get autocomplete suggestions for partial input."""
        partial = partial.lower()
        suggestions = []

        for name in self._commands.keys():
            if name.startswith(partial):
                suggestions.append(name)

        for alias in self._aliases.keys():
            if alias.startswith(partial):
                suggestions.append(alias)

        return sorted(set(suggestions))

    def set_variable(self, name: str, value: Any) -> None:
        """Set a console variable."""
        self._variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a console variable."""
        return self._variables.get(name, default)

    @property
    def command_count(self) -> int:
        return len(self._commands)

    def get_commands_by_category(self, category: CommandCategory) -> list[ConsoleCommand]:
        """Get all commands in a category."""
        return [c for c in self._commands.values() if c.category == category]


def cheat(
    name: Optional[str] = None,
    description: str = "",
    category: CommandCategory = CommandCategory.CHEAT,
    args: Optional[list[CommandArg]] = None,
    aliases: Optional[list[str]] = None,
) -> Callable:
    """
    Decorator to register a function as a cheat command.

    Usage:
        @cheat(name="god_mode", description="Toggle god mode")
        def god_mode(enabled: bool = True):
            ...
    """
    def decorator(func: Callable) -> Callable:
        cmd_name = name or func.__name__
        cmd_args = args or []

        # Auto-generate args from function signature if not provided
        if not cmd_args:
            import inspect
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                arg_type = param.annotation if param.annotation != inspect.Parameter.empty else str
                required = param.default == inspect.Parameter.empty
                default = param.default if param.default != inspect.Parameter.empty else None
                cmd_args.append(CommandArg(
                    name=param_name,
                    arg_type=arg_type,
                    required=required,
                    default=default,
                ))

        command = ConsoleCommand(
            name=cmd_name,
            callback=func,
            description=description or func.__doc__ or "",
            category=category,
            args=cmd_args,
            aliases=aliases or [],
            is_cheat=True,
        )

        # Register with console
        console = DebugConsole.get_instance()
        console.register_command(command)

        @wraps(func)
        def wrapper(*a, **kw):
            return func(*a, **kw)

        wrapper._console_command = command
        return wrapper

    return decorator
