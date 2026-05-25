"""Console command system for registration, parsing, and execution.

Provides a comprehensive command system with permission levels, argument parsing,
and decorators for easy command registration.
"""

from __future__ import annotations

import functools
import inspect
import shlex
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Union


class PermissionLevel(Enum):
    """Permission levels for console commands."""
    USER = 0  # Available to all users
    DEVELOPER = 1  # Developer-only commands
    CHEAT = 2  # Cheat commands (requires sv_cheats)
    ADMIN = 3  # Admin-only commands


class CommandStatus(Enum):
    """Status of command execution."""
    SUCCESS = auto()
    ERROR = auto()
    NOT_FOUND = auto()
    PERMISSION_DENIED = auto()
    INVALID_ARGS = auto()
    CANCELLED = auto()


@dataclass
class CommandResult:
    """Result of executing a console command."""
    status: CommandStatus
    message: str = ""
    return_value: Any = None
    execution_time: float = 0.0

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.status == CommandStatus.SUCCESS

    @classmethod
    def ok(cls, message: str = "", return_value: Any = None) -> CommandResult:
        """Create a success result."""
        return cls(CommandStatus.SUCCESS, message, return_value)

    @classmethod
    def error(cls, message: str) -> CommandResult:
        """Create an error result."""
        return cls(CommandStatus.ERROR, message)

    @classmethod
    def not_found(cls, command: str) -> CommandResult:
        """Create a not found result."""
        return cls(CommandStatus.NOT_FOUND, f"Unknown command: {command}")

    @classmethod
    def permission_denied(cls, command: str) -> CommandResult:
        """Create a permission denied result."""
        return cls(
            CommandStatus.PERMISSION_DENIED,
            f"Permission denied for command: {command}"
        )

    @classmethod
    def invalid_args(cls, message: str) -> CommandResult:
        """Create an invalid arguments result."""
        return cls(CommandStatus.INVALID_ARGS, message)


@dataclass
class CommandContext:
    """Context for command execution."""
    permission_level: PermissionLevel = PermissionLevel.USER
    cheats_enabled: bool = False
    is_server: bool = False
    user_id: Optional[str] = None
    source: str = "console"  # "console", "script", "network"
    metadata: dict = field(default_factory=dict)


@dataclass
class CommandArg:
    """Definition of a command argument."""
    name: str
    arg_type: type
    default: Any = None
    required: bool = True
    description: str = ""
    choices: Optional[list] = None

    def validate(self, value: Any) -> tuple[bool, Any, str]:
        """Validate and convert an argument value.

        Args:
            value: The value to validate

        Returns:
            Tuple of (is_valid, converted_value, error_message)
        """
        if value is None:
            if self.required:
                return False, None, f"Missing required argument: {self.name}"
            return True, self.default, ""

        try:
            if self.arg_type == bool:
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                else:
                    value = bool(value)
            else:
                value = self.arg_type(value)
        except (TypeError, ValueError) as e:
            return False, None, f"Invalid value for {self.name}: {e}"

        if self.choices is not None and value not in self.choices:
            return False, None, (
                f"Value '{value}' not valid for {self.name}. "
                f"Choices: {', '.join(str(c) for c in self.choices)}"
            )

        return True, value, ""


@dataclass
class Command:
    """A registered console command."""
    name: str
    handler: Callable
    description: str = ""
    permission: PermissionLevel = PermissionLevel.USER
    category: str = "general"
    aliases: tuple[str, ...] = ()
    hidden: bool = False
    args: list[CommandArg] = field(default_factory=list)
    requires_confirmation: bool = False

    def __post_init__(self):
        """Extract argument info from handler signature."""
        if not self.args:
            sig = inspect.signature(self.handler)
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'ctx', 'context'):
                    continue

                arg_type = param.annotation if param.annotation != inspect.Parameter.empty else str
                default = param.default if param.default != inspect.Parameter.empty else None
                required = param.default == inspect.Parameter.empty

                self.args.append(CommandArg(
                    name=param_name,
                    arg_type=arg_type,
                    default=default,
                    required=required
                ))

    def get_usage(self) -> str:
        """Get command usage string.

        Returns:
            Usage string showing arguments
        """
        parts = [self.name]
        for arg in self.args:
            if arg.required:
                parts.append(f"<{arg.name}>")
            else:
                parts.append(f"[{arg.name}]")
        return " ".join(parts)

    def get_help(self) -> str:
        """Get full help text for command.

        Returns:
            Help text with description and arguments
        """
        lines = [
            f"Command: {self.name}",
            f"Description: {self.description}",
            f"Usage: {self.get_usage()}",
            f"Permission: {self.permission.name}",
        ]

        if self.aliases:
            lines.append(f"Aliases: {', '.join(self.aliases)}")

        if self.args:
            lines.append("\nArguments:")
            for arg in self.args:
                default_str = f" (default: {arg.default})" if not arg.required else ""
                lines.append(f"  {arg.name}: {arg.arg_type.__name__}{default_str}")
                if arg.description:
                    lines.append(f"    {arg.description}")
                if arg.choices:
                    lines.append(f"    Choices: {', '.join(str(c) for c in arg.choices)}")

        return "\n".join(lines)


class CommandRegistry:
    """Registry for console commands.

    Handles command registration, lookup, and execution.
    """
    __slots__ = ('_commands', '_aliases', '_categories', '_lock')

    _instance: Optional[CommandRegistry] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        """Initialize the command registry."""
        self._commands: dict[str, Command] = {}
        self._aliases: dict[str, str] = {}
        self._categories: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> CommandRegistry:
        """Get the singleton instance.

        Returns:
            The global CommandRegistry instance
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._instance_lock:
            cls._instance = None

    def register(self, command: Command) -> None:
        """Register a command.

        Args:
            command: The command to register

        Raises:
            ValueError: If command name conflicts with existing
        """
        with self._lock:
            name_lower = command.name.lower()

            if name_lower in self._commands or name_lower in self._aliases:
                raise ValueError(f"Command or alias '{command.name}' already exists")

            self._commands[name_lower] = command

            # Register aliases
            for alias in command.aliases:
                alias_lower = alias.lower()
                if alias_lower in self._commands or alias_lower in self._aliases:
                    raise ValueError(f"Alias '{alias}' conflicts with existing command")
                self._aliases[alias_lower] = name_lower

            # Track category
            if command.category not in self._categories:
                self._categories[command.category] = set()
            self._categories[command.category].add(name_lower)

    def unregister(self, name: str) -> Optional[Command]:
        """Unregister a command.

        Args:
            name: Command name or alias

        Returns:
            The removed command, or None
        """
        with self._lock:
            name_lower = name.lower()

            # Resolve alias
            if name_lower in self._aliases:
                name_lower = self._aliases[name_lower]

            command = self._commands.pop(name_lower, None)
            if command:
                # Remove aliases
                for alias in command.aliases:
                    self._aliases.pop(alias.lower(), None)

                # Remove from category
                if command.category in self._categories:
                    self._categories[command.category].discard(name_lower)

            return command

    def get(self, name: str) -> Optional[Command]:
        """Get a command by name or alias.

        Args:
            name: Command name or alias

        Returns:
            The command if found, None otherwise
        """
        with self._lock:
            name_lower = name.lower()

            if name_lower in self._aliases:
                name_lower = self._aliases[name_lower]

            return self._commands.get(name_lower)

    def has_command(self, name: str) -> bool:
        """Check if a command exists.

        Args:
            name: Command name or alias

        Returns:
            True if command exists
        """
        return self.get(name) is not None

    def all_commands(self) -> list[Command]:
        """Get all registered commands.

        Returns:
            List of all commands
        """
        with self._lock:
            return list(self._commands.values())

    def by_category(self, category: str) -> list[Command]:
        """Get commands in a category.

        Args:
            category: The category name

        Returns:
            List of commands in category
        """
        with self._lock:
            names = self._categories.get(category, set())
            return [self._commands[n] for n in names if n in self._commands]

    def categories(self) -> list[str]:
        """Get all category names.

        Returns:
            List of category names
        """
        with self._lock:
            return list(self._categories.keys())

    def find(self, pattern: str) -> list[Command]:
        """Find commands matching a pattern.

        Args:
            pattern: Glob-style pattern (supports * wildcard)

        Returns:
            List of matching commands
        """
        import fnmatch

        with self._lock:
            return [
                cmd for name, cmd in self._commands.items()
                if fnmatch.fnmatch(name, pattern.lower())
            ]

    def parse_command_line(self, command_line: str) -> tuple[str, list[str]]:
        """Parse a command line into command and arguments.

        Args:
            command_line: The full command line

        Returns:
            Tuple of (command_name, arguments)
        """
        try:
            parts = shlex.split(command_line)
        except ValueError:
            # Handle unmatched quotes
            parts = command_line.split()

        if not parts:
            return "", []

        return parts[0], parts[1:]

    def execute(
        self,
        command_line: str,
        context: Optional[CommandContext] = None
    ) -> CommandResult:
        """Execute a command line.

        Args:
            command_line: The command line to execute
            context: Execution context

        Returns:
            Result of command execution
        """
        import time

        if context is None:
            context = CommandContext()

        command_name, raw_args = self.parse_command_line(command_line)
        if not command_name:
            return CommandResult.error("Empty command")

        command = self.get(command_name)
        if command is None:
            return CommandResult.not_found(command_name)

        # Check permissions
        if not self._check_permission(command, context):
            return CommandResult.permission_denied(command_name)

        # Parse and validate arguments
        parsed_args, error = self._parse_arguments(command, raw_args)
        if error:
            return CommandResult.invalid_args(error)

        # Execute
        start_time = time.perf_counter()
        try:
            # Check if handler expects context
            sig = inspect.signature(command.handler)
            params = list(sig.parameters.keys())

            if params and params[0] in ('ctx', 'context'):
                result = command.handler(context, *parsed_args)
            else:
                result = command.handler(*parsed_args)

            execution_time = time.perf_counter() - start_time

            if isinstance(result, CommandResult):
                result.execution_time = execution_time
                return result

            return CommandResult(
                status=CommandStatus.SUCCESS,
                message=str(result) if result is not None else "",
                return_value=result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return CommandResult(
                status=CommandStatus.ERROR,
                message=str(e),
                execution_time=execution_time
            )

    def _check_permission(self, command: Command, context: CommandContext) -> bool:
        """Check if context has permission to execute command.

        Args:
            command: The command to check
            context: Execution context

        Returns:
            True if permitted
        """
        if command.permission == PermissionLevel.USER:
            return True

        if command.permission == PermissionLevel.DEVELOPER:
            return context.permission_level.value >= PermissionLevel.DEVELOPER.value

        if command.permission == PermissionLevel.CHEAT:
            if not context.cheats_enabled:
                return False
            return context.permission_level.value >= PermissionLevel.DEVELOPER.value

        if command.permission == PermissionLevel.ADMIN:
            return context.permission_level.value >= PermissionLevel.ADMIN.value

        return False

    def _parse_arguments(
        self,
        command: Command,
        raw_args: list[str]
    ) -> tuple[list, Optional[str]]:
        """Parse and validate command arguments.

        Args:
            command: The command
            raw_args: Raw string arguments

        Returns:
            Tuple of (parsed_args, error_message)
        """
        parsed = []

        for i, arg_def in enumerate(command.args):
            raw_value = raw_args[i] if i < len(raw_args) else None

            is_valid, value, error = arg_def.validate(raw_value)
            if not is_valid:
                return [], error

            parsed.append(value)

        return parsed, None

    def get_completions(
        self,
        partial: str,
        context: Optional[CommandContext] = None
    ) -> list[str]:
        """Get command completions for partial input.

        Args:
            partial: Partial command input
            context: Execution context for filtering

        Returns:
            List of completion suggestions
        """
        if context is None:
            context = CommandContext()

        partial_lower = partial.lower()
        completions = []

        with self._lock:
            # Complete command names
            for name, cmd in self._commands.items():
                if cmd.hidden:
                    continue

                if not self._check_permission(cmd, context):
                    continue

                if name.startswith(partial_lower):
                    completions.append(cmd.name)

            # Complete aliases
            for alias, name in self._aliases.items():
                cmd = self._commands.get(name)
                if cmd and not cmd.hidden and self._check_permission(cmd, context):
                    if alias.startswith(partial_lower):
                        completions.append(alias)

        return sorted(set(completions))

    def clear(self) -> None:
        """Clear all registered commands (for testing)."""
        with self._lock:
            self._commands.clear()
            self._aliases.clear()
            self._categories.clear()


# Decorator functions for command registration

def command(
    name: Optional[str] = None,
    description: str = "",
    permission: PermissionLevel = PermissionLevel.USER,
    category: str = "general",
    aliases: tuple[str, ...] = (),
    hidden: bool = False,
    requires_confirmation: bool = False
) -> Callable:
    """Decorator to register a function as a console command.

    Args:
        name: Command name (defaults to function name)
        description: Command description
        permission: Required permission level
        category: Command category
        aliases: Alternative command names
        hidden: Hide from autocomplete
        requires_confirmation: Require user confirmation

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cmd_name = name or func.__name__
        cmd_description = description or func.__doc__ or ""

        cmd = Command(
            name=cmd_name,
            handler=func,
            description=cmd_description,
            permission=permission,
            category=category,
            aliases=aliases,
            hidden=hidden,
            requires_confirmation=requires_confirmation
        )

        # Mark the function with command info
        func._console_command = cmd

        # Auto-register with global registry
        try:
            CommandRegistry.get_instance().register(cmd)
        except ValueError:
            pass  # Already registered

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._console_command = cmd
        return wrapper

    return decorator


def cheat(
    name: Optional[str] = None,
    description: str = "",
    category: str = "cheats",
    aliases: tuple[str, ...] = (),
    requires_confirmation: bool = False
) -> Callable:
    """Decorator to register a cheat command.

    Cheat commands require sv_cheats to be enabled.

    Args:
        name: Command name (defaults to function name)
        description: Command description
        category: Command category
        aliases: Alternative command names
        requires_confirmation: Require user confirmation

    Returns:
        Decorator function
    """
    return command(
        name=name,
        description=description,
        permission=PermissionLevel.CHEAT,
        category=category,
        aliases=aliases,
        hidden=False,
        requires_confirmation=requires_confirmation
    )


def admin(
    name: Optional[str] = None,
    description: str = "",
    category: str = "admin",
    aliases: tuple[str, ...] = (),
    requires_confirmation: bool = True
) -> Callable:
    """Decorator to register an admin command.

    Admin commands require admin permission level.

    Args:
        name: Command name (defaults to function name)
        description: Command description
        category: Command category
        aliases: Alternative command names
        requires_confirmation: Require user confirmation

    Returns:
        Decorator function
    """
    return command(
        name=name,
        description=description,
        permission=PermissionLevel.ADMIN,
        category=category,
        aliases=aliases,
        hidden=False,
        requires_confirmation=requires_confirmation
    )


def developer(
    name: Optional[str] = None,
    description: str = "",
    category: str = "debug",
    aliases: tuple[str, ...] = (),
    hidden: bool = True
) -> Callable:
    """Decorator to register a developer command.

    Developer commands require developer permission level.

    Args:
        name: Command name (defaults to function name)
        description: Command description
        category: Command category
        aliases: Alternative command names
        hidden: Hide from autocomplete

    Returns:
        Decorator function
    """
    return command(
        name=name,
        description=description,
        permission=PermissionLevel.DEVELOPER,
        category=category,
        aliases=aliases,
        hidden=hidden,
        requires_confirmation=False
    )
