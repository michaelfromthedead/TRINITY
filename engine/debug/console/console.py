"""Main Console - In-game debug console with command execution and history.

This module provides the main Console class with:
- Multiple display modes (overlay, fullscreen, mini)
- Command execution with output
- Command history navigation
- CVar get/set support
- Toggle visibility

Example:
    >>> console = Console()
    >>> console.execute("help")
    >>> console.toggle()  # Show/hide console
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .aliases import AliasRegistry
from .autocomplete import Autocomplete
from .commands import (
    CommandError,
    CommandExecutionError,
    CommandNotFoundError,
    CommandRegistry,
)
from .config import CONSOLE as CONSOLE_CONFIG
from .cvar import CVar, CVarRegistry, CVarTypeError


class ConsoleMode(Enum):
    """Console display modes.

    Attributes:
        OVERLAY: Transparent overlay on top of game.
        FULLSCREEN: Dedicated fullscreen console view.
        MINI: Small input bar at bottom of screen.
    """
    OVERLAY = auto()
    FULLSCREEN = auto()
    MINI = auto()


@dataclass
class ConsoleOutput:
    """A single line of console output.

    Attributes:
        text: The output text.
        level: Output level (info, warning, error).
        timestamp: When the output was generated.
    """
    text: str
    level: str = "info"
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class Console:
    """Main debug console with command execution and display.

    The console provides an interactive interface for executing commands,
    modifying CVars, and viewing output. It supports multiple display modes
    and maintains command history for navigation.

    Attributes:
        mode: Current display mode.
        visible: Whether the console is currently shown.
        max_history: Maximum number of commands to remember.
        max_output: Maximum number of output lines to keep.

    Example:
        >>> console = Console()
        >>> console.execute("r.VSync 0")
        >>> console.execute("teleport 100 200 50")
        >>> console.visible = True
    """

    def __init__(
        self,
        command_registry: Optional[CommandRegistry] = None,
        alias_registry: Optional[AliasRegistry] = None,
        max_history: Optional[int] = None,
        max_output: Optional[int] = None,
    ) -> None:
        """Initialize the console.

        Args:
            command_registry: Optional command registry (creates default if None).
            alias_registry: Optional alias registry (creates default if None).
            max_history: Maximum number of commands in history (default from config).
            max_output: Maximum number of output lines to keep (default from config).
        """
        if max_history is None:
            max_history = CONSOLE_CONFIG.MAX_HISTORY
        if max_output is None:
            max_output = CONSOLE_CONFIG.MAX_OUTPUT
        self._command_registry = command_registry or CommandRegistry()
        self._alias_registry = alias_registry or AliasRegistry()
        self._autocomplete = Autocomplete(self._command_registry)

        self._mode: ConsoleMode = ConsoleMode.OVERLAY
        self._visible: bool = False

        self._history: List[str] = []
        self._history_index: int = -1
        self._max_history: int = max_history

        self._output: List[ConsoleOutput] = []
        self._max_output: int = max_output

        self._current_input: str = ""

        # Callbacks for UI integration
        self._on_output: List[Callable[[ConsoleOutput], None]] = []
        self._on_clear: List[Callable[[], None]] = []
        self._on_visibility_change: List[Callable[[bool], None]] = []

        # Add welcome message
        self.add_output("Console initialized. Type 'help' for commands.", "info")

    @property
    def mode(self) -> ConsoleMode:
        """Get the current console display mode."""
        return self._mode

    @mode.setter
    def mode(self, value: ConsoleMode) -> None:
        """Set the console display mode."""
        self._mode = value

    @property
    def visible(self) -> bool:
        """Check if the console is currently visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set console visibility."""
        if value != self._visible:
            self._visible = value
            for callback in self._on_visibility_change:
                callback(value)

    def toggle(self) -> bool:
        """Toggle console visibility.

        Returns:
            The new visibility state.
        """
        self.visible = not self._visible
        return self._visible

    def cycle_mode(self) -> ConsoleMode:
        """Cycle through console modes.

        Returns:
            The new console mode.
        """
        modes = list(ConsoleMode)
        current_index = modes.index(self._mode)
        self._mode = modes[(current_index + 1) % len(modes)]
        return self._mode

    def execute(self, input_text: str) -> Optional[str]:
        """Execute a console command or CVar operation.

        Supports:
        - Commands: "help", "teleport 100 200 50"
        - CVar get: "r.VSync" (returns current value)
        - CVar set: "r.VSync 0" (sets value)
        - Aliases: expands aliases before execution

        Args:
            input_text: The raw input string to execute.

        Returns:
            Output from command execution, or None.
        """
        input_text = input_text.strip()
        if not input_text:
            return None

        # Add to history
        self._add_to_history(input_text)

        # Echo the input
        self.add_output(f"> {input_text}", "input")

        try:
            # Expand aliases
            expanded = self._alias_registry.expand(input_text)

            # Handle multiple commands separated by semicolons
            results = []
            for cmd_line in expanded.split(";"):
                cmd_line = cmd_line.strip()
                if cmd_line:
                    result = self._execute_single(cmd_line)
                    if result:
                        results.append(result)

            output = "\n".join(results) if results else None
            if output:
                self.add_output(output, "info")
            return output

        except Exception as e:
            error_msg = str(e)
            self.add_output(error_msg, "error")
            return error_msg

    def _execute_single(self, input_text: str) -> Optional[str]:
        """Execute a single command or CVar operation.

        Args:
            input_text: A single command/CVar string.

        Returns:
            Output string, or None.
        """
        parts = self._parse_input(input_text)
        if not parts:
            return None

        name = parts[0]
        args = parts[1:]

        # First, check if it's a CVar
        cvar = CVarRegistry.instance().get(name)
        if cvar is not None:
            return self._handle_cvar(cvar, args)

        # Check if it's a command
        try:
            result = self._command_registry.execute(name, args)

            # Handle special results
            if result == "__CLEAR__":
                self.clear()
                return None
            elif result == "__QUIT__":
                self.add_output("Quit requested", "info")
                # In a real implementation, this would trigger application quit
                return "Quit requested"

            return result

        except CommandNotFoundError:
            # Neither command nor CVar
            return f"Unknown command or CVar: {name}"
        except CommandError as e:
            raise e

    def _handle_cvar(self, cvar: CVar, args: List[str]) -> str:
        """Handle CVar get or set operation.

        Args:
            cvar: The CVar to operate on.
            args: Arguments (empty for get, one value for set).

        Returns:
            Result string.
        """
        if not args:
            # Get CVar value
            return str(cvar)
        else:
            # Set CVar value
            try:
                cvar.value = args[0]
                return f"{cvar.name} = {cvar.value}"
            except Exception as e:
                return f"Error setting {cvar.name}: {e}"

    def _parse_input(self, input_text: str) -> List[str]:
        """Parse input into command name and arguments.

        Handles quoted strings and escapes.

        Args:
            input_text: Raw input string.

        Returns:
            List of parsed tokens.
        """
        tokens = []
        current = ""
        in_quotes = False
        quote_char = None
        escape_next = False

        for char in input_text:
            if escape_next:
                current += char
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif char.isspace() and not in_quotes:
                if current:
                    tokens.append(current)
                    current = ""
            else:
                current += char

        if current:
            tokens.append(current)

        return tokens

    def _add_to_history(self, command: str) -> None:
        """Add a command to the history.

        Args:
            command: The command string to add.
        """
        # Don't add duplicates of the most recent command
        if self._history and self._history[-1] == command:
            return

        self._history.append(command)

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Reset history navigation
        self._history_index = -1

    def history_up(self) -> Optional[str]:
        """Navigate up in command history.

        Returns:
            The previous command, or None if at the beginning.
        """
        if not self._history:
            return None

        if self._history_index == -1:
            # Start from the end
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1

        return self._history[self._history_index]

    def history_down(self) -> Optional[str]:
        """Navigate down in command history.

        Returns:
            The next command, or None if at the end.
        """
        if not self._history or self._history_index == -1:
            return None

        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            return self._history[self._history_index]
        else:
            # At the end, reset
            self._history_index = -1
            return None

    def get_history(self) -> List[str]:
        """Get the command history.

        Returns:
            List of previous commands (oldest first).
        """
        return list(self._history)

    def clear_history(self) -> None:
        """Clear the command history."""
        self._history.clear()
        self._history_index = -1

    def add_output(self, text: str, level: str = "info") -> None:
        """Add a line to the console output.

        Args:
            text: The text to add.
            level: Output level (info, warning, error, input).
        """
        output = ConsoleOutput(text=text, level=level)
        self._output.append(output)

        # Trim output if needed
        if len(self._output) > self._max_output:
            self._output = self._output[-self._max_output:]

        # Notify callbacks
        for callback in self._on_output:
            callback(output)

    def get_output(self, limit: Optional[int] = None) -> List[ConsoleOutput]:
        """Get console output lines.

        Args:
            limit: Maximum number of lines to return (None for all).

        Returns:
            List of output lines (oldest first).
        """
        if limit is None:
            return list(self._output)
        return self._output[-limit:]

    def clear(self) -> None:
        """Clear the console output."""
        self._output.clear()
        for callback in self._on_clear:
            callback()

    def complete(self, partial: str) -> List[str]:
        """Get completions for partial input.

        Args:
            partial: Partial input to complete.

        Returns:
            List of possible completions.
        """
        return self._autocomplete.get_completions(partial)

    def register_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        **kwargs
    ) -> None:
        """Register a console command.

        Convenience method that delegates to CommandRegistry.

        Args:
            name: Command name.
            handler: Command handler function.
            description: Command description.
            **kwargs: Additional Command parameters.
        """
        self._command_registry.register(
            name=name,
            handler=handler,
            description=description,
            **kwargs
        )

    def register_alias(self, name: str, expansion: str) -> None:
        """Register a command alias.

        Args:
            name: Alias name.
            expansion: Command(s) the alias expands to.
        """
        self._alias_registry.register(name, expansion)

    def on_output(self, callback: Callable[[ConsoleOutput], None]) -> None:
        """Register a callback for new output.

        Args:
            callback: Function called with each new output line.
        """
        self._on_output.append(callback)

    def on_clear(self, callback: Callable[[], None]) -> None:
        """Register a callback for clear events.

        Args:
            callback: Function called when console is cleared.
        """
        self._on_clear.append(callback)

    def on_visibility_change(self, callback: Callable[[bool], None]) -> None:
        """Register a callback for visibility changes.

        Args:
            callback: Function called with new visibility state.
        """
        self._on_visibility_change.append(callback)

    @property
    def cheats_enabled(self) -> bool:
        """Check if cheats are enabled."""
        return self._command_registry.cheats_enabled

    @cheats_enabled.setter
    def cheats_enabled(self, value: bool) -> None:
        """Enable or disable cheats."""
        self._command_registry.cheats_enabled = value
        CVarRegistry.instance().cheats_enabled = value
        level = "warning" if value else "info"
        status = "enabled" if value else "disabled"
        self.add_output(f"Cheats {status}", level)

    def get_stats(self) -> Dict[str, Any]:
        """Get console statistics.

        Returns:
            Dictionary with console statistics.
        """
        return {
            "mode": self._mode.name,
            "visible": self._visible,
            "history_count": len(self._history),
            "output_count": len(self._output),
            "command_count": len(self._command_registry),
            "alias_count": len(self._alias_registry),
            "cheats_enabled": self.cheats_enabled,
        }
