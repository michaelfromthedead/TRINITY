"""Command System - Console command registration and execution.

This module provides a command system for the debug console with:
- Command registration with handlers
- Built-in commands (help, list, exec, alias, clear)
- Command flags for access control
- Command argument parsing

Example:
    >>> registry = CommandRegistry()
    >>> registry.register("teleport",
    ...     handler=lambda x, y, z: player.set_position(x, y, z),
    ...     description="Teleport to position")
    >>> registry.execute("teleport", ["100", "200", "50"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .cvar import CVarFlags, CVarRegistry


class CommandFlags(Flag):
    """Flags that control command behavior and access.

    Attributes:
        NONE: No special flags.
        CHEAT: Command is only accessible when cheats are enabled.
        HIDDEN: Command is not shown in help/list.
        BUILTIN: Command is a built-in system command.
    """
    NONE = 0
    CHEAT = auto()
    HIDDEN = auto()
    BUILTIN = auto()


# Type for command handlers
CommandHandler = Callable[..., Optional[str]]


class CommandError(Exception):
    """Base exception for command errors."""
    pass


class CommandNotFoundError(CommandError):
    """Raised when a command is not found."""
    pass


class CommandAccessError(CommandError):
    """Raised when access to a command is denied."""
    pass


class CommandExecutionError(CommandError):
    """Raised when command execution fails."""
    pass


@dataclass
class Command:
    """A console command with handler and metadata.

    Attributes:
        name: Unique identifier for the command.
        handler: Callable that executes the command.
        description: Human-readable description.
        flags: Bitflags controlling access and visibility.
        usage: Optional usage string (e.g., "teleport <x> <y> <z>").
        min_args: Minimum number of required arguments.
        max_args: Maximum number of arguments (-1 for unlimited).
    """
    name: str
    handler: CommandHandler
    description: str = ""
    flags: CommandFlags = CommandFlags.NONE
    usage: str = ""
    min_args: int = 0
    max_args: int = -1

    def __post_init__(self) -> None:
        """Validate command configuration."""
        if not self.name:
            raise ValueError("Command name cannot be empty")
        if not callable(self.handler):
            raise ValueError(f"Command '{self.name}' handler must be callable")

    def validate_args(self, args: List[str]) -> None:
        """Validate argument count.

        Args:
            args: List of argument strings.

        Raises:
            CommandExecutionError: If argument count is invalid.
        """
        arg_count = len(args)
        if arg_count < self.min_args:
            raise CommandExecutionError(
                f"Command '{self.name}' requires at least {self.min_args} "
                f"argument(s), got {arg_count}"
            )
        if self.max_args >= 0 and arg_count > self.max_args:
            raise CommandExecutionError(
                f"Command '{self.name}' accepts at most {self.max_args} "
                f"argument(s), got {arg_count}"
            )

    def get_help(self) -> str:
        """Get formatted help text for this command.

        Returns:
            Formatted help string.
        """
        lines = [f"Command: {self.name}"]
        if self.description:
            lines.append(f"  {self.description}")
        if self.usage:
            lines.append(f"  Usage: {self.usage}")
        if self.flags != CommandFlags.NONE:
            lines.append(f"  Flags: {self.flags.name}")
        return "\n".join(lines)


class CommandRegistry:
    """Registry for console commands.

    Provides command registration, lookup, and execution with built-in
    commands for common operations.

    Example:
        >>> registry = CommandRegistry()
        >>> registry.register("ping", lambda: "pong")
        >>> result = registry.execute("ping", [])
        >>> print(result)
        pong
    """

    def __init__(self, register_builtins: bool = True) -> None:
        """Initialize the command registry.

        Args:
            register_builtins: If True, register built-in commands.
        """
        self._commands: Dict[str, Command] = {}
        self._cheats_enabled: bool = False

        if register_builtins:
            self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in commands."""
        # Help command
        self.register(
            name="help",
            handler=self._cmd_help,
            description="Show help for a command or list all commands",
            flags=CommandFlags.BUILTIN,
            usage="help [command]",
            max_args=1,
        )

        # List command
        self.register(
            name="list",
            handler=self._cmd_list,
            description="List all available commands",
            flags=CommandFlags.BUILTIN,
            usage="list [filter]",
            max_args=1,
        )

        # Clear command
        self.register(
            name="clear",
            handler=self._cmd_clear,
            description="Clear the console output",
            flags=CommandFlags.BUILTIN,
            max_args=0,
        )

        # Echo command
        self.register(
            name="echo",
            handler=self._cmd_echo,
            description="Print a message to the console",
            flags=CommandFlags.BUILTIN,
            usage="echo <message>",
        )

        # CVar list command
        self.register(
            name="cvarlist",
            handler=self._cmd_cvarlist,
            description="List all console variables",
            flags=CommandFlags.BUILTIN,
            usage="cvarlist [filter]",
            max_args=1,
        )

        # Reset command
        self.register(
            name="reset",
            handler=self._cmd_reset,
            description="Reset a CVar to its default value",
            flags=CommandFlags.BUILTIN,
            usage="reset <cvar>",
            min_args=1,
            max_args=1,
        )

        # Version command
        self.register(
            name="version",
            handler=self._cmd_version,
            description="Show engine version",
            flags=CommandFlags.BUILTIN,
            max_args=0,
        )

        # Quit command
        self.register(
            name="quit",
            handler=self._cmd_quit,
            description="Quit the application",
            flags=CommandFlags.BUILTIN,
            max_args=0,
        )

    def _cmd_help(self, command_name: Optional[str] = None) -> str:
        """Built-in help command."""
        if command_name:
            cmd = self.get(command_name)
            if cmd:
                return cmd.get_help()
            else:
                return f"Unknown command: {command_name}"
        else:
            # List all commands
            lines = ["Available commands:"]
            for cmd in sorted(self._commands.values(), key=lambda c: c.name):
                if CommandFlags.HIDDEN not in cmd.flags:
                    desc = cmd.description[:50] + "..." if len(cmd.description) > 50 else cmd.description
                    lines.append(f"  {cmd.name:<20} - {desc}")
            lines.append("\nType 'help <command>' for more information")
            return "\n".join(lines)

    def _cmd_list(self, pattern: Optional[str] = None) -> str:
        """Built-in list command."""
        import fnmatch

        commands = []
        for cmd in sorted(self._commands.values(), key=lambda c: c.name):
            if CommandFlags.HIDDEN not in cmd.flags:
                if pattern is None or fnmatch.fnmatch(cmd.name, pattern):
                    commands.append(cmd.name)

        if commands:
            return "Commands:\n  " + "\n  ".join(commands)
        else:
            return "No commands found"

    def _cmd_clear(self) -> str:
        """Built-in clear command."""
        # Returns special marker that console interprets as clear
        return "__CLEAR__"

    def _cmd_echo(self, *args: str) -> str:
        """Built-in echo command."""
        return " ".join(args)

    def _cmd_cvarlist(self, pattern: Optional[str] = None) -> str:
        """Built-in cvarlist command."""
        registry = CVarRegistry.instance()

        if pattern:
            cvars = registry.find(pattern)
        else:
            cvars = registry.all()

        if cvars:
            lines = []
            for cvar in cvars:
                lines.append(str(cvar))
            return "\n".join(lines)
        else:
            return "No CVars found"

    def _cmd_reset(self, cvar_name: str) -> str:
        """Built-in reset command."""
        registry = CVarRegistry.instance()
        cvar = registry.get(cvar_name)

        if cvar:
            try:
                cvar.reset()
                return f"Reset {cvar_name} to {cvar.default}"
            except Exception as e:
                return f"Error resetting {cvar_name}: {e}"
        else:
            return f"Unknown CVar: {cvar_name}"

    def _cmd_version(self) -> str:
        """Built-in version command."""
        return "AI Game Engine v1.0.0-dev"

    def _cmd_quit(self) -> str:
        """Built-in quit command."""
        # Returns special marker that console interprets as quit request
        return "__QUIT__"

    @property
    def cheats_enabled(self) -> bool:
        """Check if cheats are currently enabled."""
        return self._cheats_enabled

    @cheats_enabled.setter
    def cheats_enabled(self, value: bool) -> None:
        """Enable or disable cheats."""
        self._cheats_enabled = value

    def register(
        self,
        name: str,
        handler: CommandHandler,
        description: str = "",
        flags: CommandFlags = CommandFlags.NONE,
        usage: str = "",
        min_args: int = 0,
        max_args: int = -1,
    ) -> Command:
        """Register a new command.

        Args:
            name: Unique command identifier.
            handler: Callable that executes the command.
            description: Human-readable description.
            flags: Bitflags controlling access and visibility.
            usage: Optional usage string.
            min_args: Minimum required arguments.
            max_args: Maximum arguments (-1 for unlimited).

        Returns:
            The registered Command object.

        Raises:
            ValueError: If a command with the same name already exists.
        """
        if name in self._commands:
            raise ValueError(f"Command '{name}' is already registered")

        cmd = Command(
            name=name,
            handler=handler,
            description=description,
            flags=flags,
            usage=usage,
            min_args=min_args,
            max_args=max_args,
        )
        self._commands[name] = cmd
        return cmd

    def unregister(self, name: str) -> bool:
        """Unregister a command by name.

        Args:
            name: The name of the command to unregister.

        Returns:
            True if the command was found and removed, False otherwise.
        """
        if name in self._commands:
            del self._commands[name]
            return True
        return False

    def get(self, name: str) -> Optional[Command]:
        """Get a command by name.

        Args:
            name: The name of the command to retrieve.

        Returns:
            The Command if found, None otherwise.
        """
        return self._commands.get(name)

    def all(self) -> List[Command]:
        """Get all registered commands.

        Returns:
            List of all commands, sorted by name.
        """
        return sorted(self._commands.values(), key=lambda c: c.name)

    def visible(self) -> List[Command]:
        """Get all visible (non-hidden) commands.

        Returns:
            List of visible commands, sorted by name.
        """
        return [
            cmd for cmd in self.all()
            if CommandFlags.HIDDEN not in cmd.flags
        ]

    def execute(self, name: str, args: List[str]) -> Optional[str]:
        """Execute a command with arguments.

        Args:
            name: The command name to execute.
            args: List of argument strings.

        Returns:
            Command output string, or None.

        Raises:
            CommandNotFoundError: If the command doesn't exist.
            CommandAccessError: If access to the command is denied.
            CommandExecutionError: If command execution fails.
        """
        cmd = self.get(name)

        if cmd is None:
            raise CommandNotFoundError(f"Unknown command: {name}")

        # Check cheat access
        if CommandFlags.CHEAT in cmd.flags and not self._cheats_enabled:
            raise CommandAccessError(
                f"Command '{name}' requires cheats to be enabled"
            )

        # Validate argument count
        cmd.validate_args(args)

        # Execute the command
        try:
            return cmd.handler(*args)
        except TypeError as e:
            raise CommandExecutionError(
                f"Invalid arguments for command '{name}': {e}"
            )
        except Exception as e:
            raise CommandExecutionError(
                f"Error executing command '{name}': {e}"
            )

    def get_completions(self, partial: str) -> List[str]:
        """Get command name completions for partial input.

        Args:
            partial: Partial command name to complete.

        Returns:
            List of matching command names.
        """
        partial_lower = partial.lower()
        matches = []

        for name, cmd in self._commands.items():
            if CommandFlags.HIDDEN not in cmd.flags:
                if name.lower().startswith(partial_lower):
                    matches.append(name)

        return sorted(matches)

    def __len__(self) -> int:
        """Return the number of registered commands."""
        return len(self._commands)

    def __contains__(self, name: str) -> bool:
        """Check if a command is registered."""
        return name in self._commands

    def __iter__(self):
        """Iterate over command names."""
        return iter(self._commands)
