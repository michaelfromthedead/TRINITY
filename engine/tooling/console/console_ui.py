"""Console UI system with input, output, scrollback, and autocomplete.

Provides a comprehensive developer console interface for the game engine.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Optional


class ConsoleMode(Enum):
    """Console operation modes with different permission levels."""
    USER = auto()  # Basic commands only
    DEVELOPER = auto()  # Developer commands enabled
    CHEAT = auto()  # Cheat commands enabled
    ADMIN = auto()  # Full admin access


class OutputType(Enum):
    """Types of console output for styling."""
    NORMAL = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    SUCCESS = auto()
    COMMAND = auto()  # Echoed commands
    SYSTEM = auto()  # System messages


@dataclass
class ConsoleConfig:
    """Configuration for the console UI."""
    max_scrollback: int = 1000
    max_input_length: int = 4096
    echo_commands: bool = True
    timestamp_format: Optional[str] = None  # None = no timestamps
    autocomplete_enabled: bool = True
    autocomplete_min_chars: int = 1
    autocomplete_max_suggestions: int = 10
    history_enabled: bool = True
    history_capacity: int = 500
    filter_duplicates: bool = True


@dataclass
class OutputLine:
    """A single line of console output."""
    text: str
    output_type: OutputType = OutputType.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    category: Optional[str] = None

    def formatted(self, config: ConsoleConfig) -> str:
        """Get formatted output string.

        Args:
            config: Console configuration

        Returns:
            Formatted string
        """
        parts = []

        if config.timestamp_format:
            parts.append(self.timestamp.strftime(config.timestamp_format))

        if self.category:
            parts.append(f"[{self.category}]")

        parts.append(self.text)

        return " ".join(parts)


@dataclass
class AutocompleteResult:
    """Result of autocomplete operation."""
    suggestions: list[str]
    common_prefix: str
    total_matches: int


class ConsoleUI:
    """Developer console with input/output management.

    Provides:
    - Input handling with history navigation
    - Output with scrollback buffer
    - Autocomplete functionality
    - Mode-based permission control
    """
    __slots__ = (
        '_config', '_scrollback', '_input_buffer', '_cursor_pos',
        '_mode', '_history', '_lock', '_output_callbacks',
        '_command_handler', '_autocomplete_handler', '_visible',
        '_filter_pattern', '_filter_types'
    )

    def __init__(
        self,
        config: Optional[ConsoleConfig] = None,
        history: Optional['CommandHistory'] = None
    ):
        """Initialize the console UI.

        Args:
            config: Console configuration
            history: Command history instance
        """
        from .command_history import CommandHistory

        self._config = config or ConsoleConfig()
        self._scrollback: deque[OutputLine] = deque(maxlen=self._config.max_scrollback)
        self._input_buffer: str = ""
        self._cursor_pos: int = 0
        self._mode = ConsoleMode.USER
        self._history = history if history is not None else CommandHistory(
            capacity=self._config.history_capacity
        )
        self._lock = threading.RLock()
        self._output_callbacks: list[Callable[[OutputLine], None]] = []
        self._command_handler: Optional[Callable[[str], None]] = None
        self._autocomplete_handler: Optional[Callable[[str], list[str]]] = None
        self._visible = False
        self._filter_pattern: Optional[str] = None
        self._filter_types: Optional[set[OutputType]] = None

    @property
    def config(self) -> ConsoleConfig:
        """Get console configuration."""
        return self._config

    @property
    def mode(self) -> ConsoleMode:
        """Get current console mode."""
        return self._mode

    @mode.setter
    def mode(self, value: ConsoleMode) -> None:
        """Set console mode."""
        self._mode = value
        self.write(f"Console mode: {value.name}", OutputType.SYSTEM)

    @property
    def visible(self) -> bool:
        """Check if console is visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set console visibility."""
        self._visible = value

    @property
    def input_buffer(self) -> str:
        """Get current input buffer."""
        return self._input_buffer

    @property
    def cursor_position(self) -> int:
        """Get cursor position in input buffer."""
        return self._cursor_pos

    @property
    def scrollback_count(self) -> int:
        """Get number of lines in scrollback."""
        return len(self._scrollback)

    @property
    def history(self) -> 'CommandHistory':
        """Get command history."""
        return self._history

    def set_command_handler(self, handler: Callable[[str], None]) -> None:
        """Set handler for command execution.

        Args:
            handler: Function to call with command string
        """
        self._command_handler = handler

    def set_autocomplete_handler(self, handler: Callable[[str], list[str]]) -> None:
        """Set handler for autocomplete suggestions.

        Args:
            handler: Function that returns suggestions for input
        """
        self._autocomplete_handler = handler

    def add_output_callback(self, callback: Callable[[OutputLine], None]) -> None:
        """Add callback for output events.

        Args:
            callback: Function called when output is written
        """
        with self._lock:
            if callback not in self._output_callbacks:
                self._output_callbacks.append(callback)

    def remove_output_callback(self, callback: Callable[[OutputLine], None]) -> None:
        """Remove output callback.

        Args:
            callback: Callback to remove
        """
        with self._lock:
            try:
                self._output_callbacks.remove(callback)
            except ValueError:
                pass

    def toggle(self) -> bool:
        """Toggle console visibility.

        Returns:
            New visibility state
        """
        self._visible = not self._visible
        return self._visible

    def write(
        self,
        text: str,
        output_type: OutputType = OutputType.NORMAL,
        category: Optional[str] = None
    ) -> None:
        """Write text to console output.

        Args:
            text: Text to write
            output_type: Output type for styling
            category: Optional category tag
        """
        line = OutputLine(
            text=text,
            output_type=output_type,
            category=category
        )

        # Apply filters
        if self._filter_types and output_type not in self._filter_types:
            return

        if self._filter_pattern:
            if self._filter_pattern.lower() not in text.lower():
                return

        with self._lock:
            self._scrollback.append(line)

            # Notify callbacks
            for callback in self._output_callbacks:
                try:
                    callback(line)
                except Exception:
                    pass

    def write_info(self, text: str, category: Optional[str] = None) -> None:
        """Write info message."""
        self.write(text, OutputType.INFO, category)

    def write_warning(self, text: str, category: Optional[str] = None) -> None:
        """Write warning message."""
        self.write(text, OutputType.WARNING, category)

    def write_error(self, text: str, category: Optional[str] = None) -> None:
        """Write error message."""
        self.write(text, OutputType.ERROR, category)

    def write_success(self, text: str, category: Optional[str] = None) -> None:
        """Write success message."""
        self.write(text, OutputType.SUCCESS, category)

    def get_scrollback(
        self,
        count: Optional[int] = None,
        offset: int = 0
    ) -> list[OutputLine]:
        """Get scrollback buffer contents.

        Args:
            count: Number of lines (None = all)
            offset: Offset from end

        Returns:
            List of output lines
        """
        with self._lock:
            lines = list(self._scrollback)

            if offset > 0:
                lines = lines[:-offset] if offset < len(lines) else []

            if count is not None:
                lines = lines[-count:]

            return lines

    def clear_scrollback(self) -> None:
        """Clear the scrollback buffer."""
        with self._lock:
            self._scrollback.clear()
        self.write("Console cleared", OutputType.SYSTEM)

    def set_filter(
        self,
        pattern: Optional[str] = None,
        types: Optional[set[OutputType]] = None
    ) -> None:
        """Set output filter.

        Args:
            pattern: Text pattern to match
            types: Output types to show
        """
        self._filter_pattern = pattern
        self._filter_types = types

    def clear_filter(self) -> None:
        """Clear output filter."""
        self._filter_pattern = None
        self._filter_types = None

    # Input handling methods

    def set_input(self, text: str) -> None:
        """Set input buffer contents.

        Args:
            text: New input text
        """
        if len(text) > self._config.max_input_length:
            text = text[:self._config.max_input_length]

        with self._lock:
            self._input_buffer = text
            self._cursor_pos = len(text)

    def insert_char(self, char: str) -> None:
        """Insert character at cursor.

        Args:
            char: Character to insert
        """
        if len(self._input_buffer) >= self._config.max_input_length:
            return

        with self._lock:
            self._input_buffer = (
                self._input_buffer[:self._cursor_pos] +
                char +
                self._input_buffer[self._cursor_pos:]
            )
            self._cursor_pos += len(char)

    def delete_char(self) -> None:
        """Delete character before cursor (backspace)."""
        with self._lock:
            if self._cursor_pos > 0:
                self._input_buffer = (
                    self._input_buffer[:self._cursor_pos - 1] +
                    self._input_buffer[self._cursor_pos:]
                )
                self._cursor_pos -= 1

    def delete_forward(self) -> None:
        """Delete character at cursor (delete key)."""
        with self._lock:
            if self._cursor_pos < len(self._input_buffer):
                self._input_buffer = (
                    self._input_buffer[:self._cursor_pos] +
                    self._input_buffer[self._cursor_pos + 1:]
                )

    def delete_word(self) -> None:
        """Delete word before cursor."""
        with self._lock:
            if self._cursor_pos == 0:
                return

            # Find word boundary
            pos = self._cursor_pos - 1
            while pos > 0 and self._input_buffer[pos - 1].isspace():
                pos -= 1
            while pos > 0 and not self._input_buffer[pos - 1].isspace():
                pos -= 1

            self._input_buffer = (
                self._input_buffer[:pos] +
                self._input_buffer[self._cursor_pos:]
            )
            self._cursor_pos = pos

    def clear_input(self) -> None:
        """Clear input buffer."""
        with self._lock:
            self._input_buffer = ""
            self._cursor_pos = 0

    def move_cursor_left(self) -> None:
        """Move cursor left one position."""
        with self._lock:
            if self._cursor_pos > 0:
                self._cursor_pos -= 1

    def move_cursor_right(self) -> None:
        """Move cursor right one position."""
        with self._lock:
            if self._cursor_pos < len(self._input_buffer):
                self._cursor_pos += 1

    def move_cursor_home(self) -> None:
        """Move cursor to start of input."""
        self._cursor_pos = 0

    def move_cursor_end(self) -> None:
        """Move cursor to end of input."""
        self._cursor_pos = len(self._input_buffer)

    def move_cursor_word_left(self) -> None:
        """Move cursor to previous word boundary."""
        with self._lock:
            if self._cursor_pos == 0:
                return

            pos = self._cursor_pos - 1
            while pos > 0 and self._input_buffer[pos - 1].isspace():
                pos -= 1
            while pos > 0 and not self._input_buffer[pos - 1].isspace():
                pos -= 1

            self._cursor_pos = pos

    def move_cursor_word_right(self) -> None:
        """Move cursor to next word boundary."""
        with self._lock:
            length = len(self._input_buffer)
            if self._cursor_pos >= length:
                return

            pos = self._cursor_pos
            while pos < length and not self._input_buffer[pos].isspace():
                pos += 1
            while pos < length and self._input_buffer[pos].isspace():
                pos += 1

            self._cursor_pos = pos

    # History navigation

    def history_previous(self) -> Optional[str]:
        """Navigate to previous command in history.

        Returns:
            Previous command, or None
        """
        if not self._config.history_enabled:
            return None

        command = self._history.previous()
        if command:
            self.set_input(command)
        return command

    def history_next(self) -> Optional[str]:
        """Navigate to next command in history.

        Returns:
            Next command, or None
        """
        if not self._config.history_enabled:
            return None

        command = self._history.next()
        if command:
            self.set_input(command)
        else:
            self.clear_input()
        return command

    def history_search(self, query: str) -> list[str]:
        """Search command history.

        Args:
            query: Search query

        Returns:
            Matching commands
        """
        if not self._config.history_enabled:
            return []

        entries = self._history.search(query)
        return [e.command for e in entries]

    # Autocomplete

    def get_completions(self) -> AutocompleteResult:
        """Get autocomplete suggestions for current input.

        Returns:
            Autocomplete result with suggestions
        """
        if not self._config.autocomplete_enabled:
            return AutocompleteResult([], "", 0)

        input_text = self._input_buffer.strip()

        if len(input_text) < self._config.autocomplete_min_chars:
            return AutocompleteResult([], "", 0)

        # Get suggestions from handler
        if self._autocomplete_handler:
            suggestions = self._autocomplete_handler(input_text)
        else:
            suggestions = []

        # Limit results
        total = len(suggestions)
        suggestions = suggestions[:self._config.autocomplete_max_suggestions]

        # Find common prefix
        common_prefix = ""
        if suggestions:
            common_prefix = suggestions[0]
            for s in suggestions[1:]:
                while not s.startswith(common_prefix):
                    common_prefix = common_prefix[:-1]
                    if not common_prefix:
                        break

        return AutocompleteResult(
            suggestions=suggestions,
            common_prefix=common_prefix,
            total_matches=total
        )

    def apply_completion(self, completion: str) -> None:
        """Apply an autocomplete suggestion.

        Args:
            completion: Completion to apply
        """
        self.set_input(completion)

    def tab_complete(self) -> bool:
        """Perform tab completion on current input.

        Returns:
            True if completion was applied
        """
        result = self.get_completions()

        if not result.suggestions:
            return False

        if len(result.suggestions) == 1:
            # Single match - apply it
            self.apply_completion(result.suggestions[0])
            return True

        if result.common_prefix and len(result.common_prefix) > len(self._input_buffer.strip()):
            # Extend to common prefix
            self.apply_completion(result.common_prefix)
            return True

        # Multiple matches - show them
        self.write("Completions:", OutputType.INFO)
        for suggestion in result.suggestions:
            self.write(f"  {suggestion}", OutputType.NORMAL)

        if result.total_matches > len(result.suggestions):
            self.write(
                f"  ... and {result.total_matches - len(result.suggestions)} more",
                OutputType.INFO
            )

        return False

    # Command execution

    def execute(self) -> bool:
        """Execute current input as command.

        Returns:
            True if command was executed
        """
        command = self._input_buffer.strip()
        if not command:
            return False

        # Echo command
        if self._config.echo_commands:
            self.write(f"> {command}", OutputType.COMMAND)

        # Add to history
        if self._config.history_enabled:
            self._history.add(command)

        # Clear input
        self.clear_input()

        # Execute
        if self._command_handler:
            try:
                self._command_handler(command)
            except Exception as e:
                self.write_error(f"Error: {e}")
                return False

        return True

    def execute_command(self, command: str) -> bool:
        """Execute a command directly.

        Args:
            command: Command string to execute

        Returns:
            True if command was executed
        """
        self.set_input(command)
        return self.execute()
