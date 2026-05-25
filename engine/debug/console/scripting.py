"""Script Execution - Execute console scripts from files.

This module provides script file execution with:
- .cfg file parsing and execution
- Comment support (// and #)
- Variable substitution
- Conditional execution

Example:
    >>> executor = ScriptExecutor(console)
    >>> executor.exec_file("debug.cfg")
    # Executes all commands in debug.cfg
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .config import SCRIPT as SCRIPT_CONFIG

if TYPE_CHECKING:
    from .console import Console


class ScriptError(Exception):
    """Base exception for script errors."""
    pass


class ScriptFileNotFoundError(ScriptError):
    """Raised when a script file is not found."""
    pass


class ScriptSyntaxError(ScriptError):
    """Raised when a script contains syntax errors."""
    pass


class ScriptLimitError(ScriptError):
    """Raised when a script exceeds configured limits."""
    pass


@dataclass
class ScriptLine:
    """A single line from a script file.

    Attributes:
        line_number: 1-indexed line number.
        raw_text: Original line text.
        command: Parsed command (or None for comments/empty).
    """
    line_number: int
    raw_text: str
    command: Optional[str] = None


@dataclass
class ScriptResult:
    """Result of executing a script.

    Attributes:
        path: Path to the script file.
        lines_executed: Number of lines executed.
        lines_skipped: Number of lines skipped (comments/empty).
        errors: List of (line_number, error_message) tuples.
        success: True if no errors occurred.
    """
    path: str
    lines_executed: int = 0
    lines_skipped: int = 0
    errors: List[tuple] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if script executed without errors."""
        return len(self.errors) == 0


class ScriptExecutor:
    """Executes console scripts from files.

    Supports:
    - Line-by-line execution
    - Comment lines (// and #)
    - Empty lines
    - Variable substitution

    Example:
        >>> executor = ScriptExecutor(console)
        >>> result = executor.exec_file("startup.cfg")
        >>> if result.success:
        ...     print(f"Executed {result.lines_executed} commands")
    """

    def __init__(
        self,
        console: "Console",
        base_path: Optional[str] = None,
        max_lines: Optional[int] = None,
        max_include_depth: Optional[int] = None,
        max_line_length: Optional[int] = None,
    ) -> None:
        """Initialize the script executor.

        Args:
            console: The console to execute commands on.
            base_path: Base path for relative script paths.
            max_lines: Maximum lines per script (default from config).
            max_include_depth: Maximum nested include depth (default from config).
            max_line_length: Maximum line length (default from config).
        """
        self._console = console
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._variables: Dict[str, str] = {}
        self._include_stack: List[str] = []  # For recursion detection
        self._max_lines = max_lines if max_lines is not None else SCRIPT_CONFIG.MAX_LINES
        self._max_include_depth = max_include_depth if max_include_depth is not None else SCRIPT_CONFIG.MAX_INCLUDE_DEPTH
        self._max_line_length = max_line_length if max_line_length is not None else SCRIPT_CONFIG.MAX_LINE_LENGTH

    def exec_file(
        self,
        path: str,
        stop_on_error: bool = False,
    ) -> ScriptResult:
        """Execute a script file.

        Args:
            path: Path to the script file.
            stop_on_error: If True, stop on first error.

        Returns:
            ScriptResult with execution details.

        Raises:
            ScriptFileNotFoundError: If the file doesn't exist.
        """
        # Resolve path
        resolved_path = self._resolve_path(path)

        if not resolved_path.exists():
            raise ScriptFileNotFoundError(f"Script not found: {path}")

        # Check for recursive includes
        path_str = str(resolved_path.resolve())
        if path_str in self._include_stack:
            raise ScriptError(f"Recursive include detected: {path}")

        # Check include depth limit
        if len(self._include_stack) >= self._max_include_depth:
            raise ScriptLimitError(
                f"Maximum include depth ({self._max_include_depth}) exceeded"
            )

        self._include_stack.append(path_str)

        try:
            # Read and parse the file
            lines = self._parse_file(resolved_path)

            # Execute lines
            result = ScriptResult(path=str(resolved_path))

            for script_line in lines:
                if script_line.command is None:
                    result.lines_skipped += 1
                    continue

                try:
                    # Handle special commands
                    if self._handle_special_command(script_line.command):
                        result.lines_executed += 1
                        continue

                    # Substitute variables
                    command = self._substitute_variables(script_line.command)

                    # Execute the command
                    self._console.execute(command)
                    result.lines_executed += 1

                except Exception as e:
                    result.errors.append((script_line.line_number, str(e)))
                    if stop_on_error:
                        break

            return result

        finally:
            self._include_stack.pop()

    def exec_string(self, script: str) -> ScriptResult:
        """Execute a script string.

        Args:
            script: Multi-line script string.

        Returns:
            ScriptResult with execution details.
        """
        lines = self._parse_string(script)
        result = ScriptResult(path="<string>")

        for script_line in lines:
            if script_line.command is None:
                result.lines_skipped += 1
                continue

            try:
                # Handle special commands
                if self._handle_special_command(script_line.command):
                    result.lines_executed += 1
                    continue

                # Substitute variables
                command = self._substitute_variables(script_line.command)

                # Execute the command
                self._console.execute(command)
                result.lines_executed += 1

            except Exception as e:
                result.errors.append((script_line.line_number, str(e)))

        return result

    def _resolve_path(self, path: str) -> Path:
        """Resolve a script path.

        Args:
            path: Relative or absolute path.

        Returns:
            Resolved Path object.
        """
        script_path = Path(path)

        if script_path.is_absolute():
            return script_path

        # Try base path first
        resolved = self._base_path / script_path
        if resolved.exists():
            return resolved

        # Try common script directories
        for subdir in ["scripts", "config", "cfg"]:
            resolved = self._base_path / subdir / script_path
            if resolved.exists():
                return resolved

        # Return as-is for better error message
        return self._base_path / script_path

    def _parse_file(self, path: Path) -> List[ScriptLine]:
        """Parse a script file into lines.

        Args:
            path: Path to the script file.

        Returns:
            List of ScriptLine objects.
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return self._parse_string(content)

    def _parse_string(self, content: str) -> List[ScriptLine]:
        """Parse a script string into lines.

        Args:
            content: Script content string.

        Returns:
            List of ScriptLine objects.
        """
        lines = []
        content_lines = content.splitlines()

        # Check line count limit
        if len(content_lines) > self._max_lines:
            raise ScriptLimitError(
                f"Script exceeds maximum line count ({self._max_lines})"
            )

        for i, raw_line in enumerate(content_lines, 1):
            # Check line length limit
            if len(raw_line) > self._max_line_length:
                raise ScriptLimitError(
                    f"Line {i} exceeds maximum length ({self._max_line_length})"
                )
            line = raw_line.strip()

            # Skip empty lines
            if not line:
                lines.append(ScriptLine(line_number=i, raw_text=raw_line))
                continue

            # Skip comment lines
            if line.startswith("//") or line.startswith("#"):
                lines.append(ScriptLine(line_number=i, raw_text=raw_line))
                continue

            # Remove inline comments
            for comment_char in ["//", "#"]:
                if comment_char in line:
                    # Only remove if not inside quotes
                    comment_pos = self._find_comment(line, comment_char)
                    if comment_pos >= 0:
                        line = line[:comment_pos].strip()

            if line:
                lines.append(ScriptLine(
                    line_number=i,
                    raw_text=raw_line,
                    command=line,
                ))
            else:
                lines.append(ScriptLine(line_number=i, raw_text=raw_line))

        return lines

    def _find_comment(self, line: str, comment_char: str) -> int:
        """Find the position of a comment marker, respecting quotes.

        Args:
            line: The line to search.
            comment_char: The comment marker to find.

        Returns:
            Position of comment marker, or -1 if not found.
        """
        in_quotes = False
        quote_char = None

        for i, char in enumerate(line):
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif not in_quotes and line[i:].startswith(comment_char):
                return i

        return -1

    def _handle_special_command(self, command: str) -> bool:
        """Handle special script commands.

        Args:
            command: The command to check.

        Returns:
            True if the command was handled, False otherwise.
        """
        parts = command.split(None, 2)
        if not parts:
            return False

        cmd = parts[0].lower()

        # Include command
        if cmd in ("exec", "include"):
            if len(parts) < 2:
                raise ScriptSyntaxError("exec/include requires a file path")
            self.exec_file(parts[1])
            return True

        # Set variable
        if cmd == "set":
            if len(parts) < 3:
                raise ScriptSyntaxError("set requires name and value")
            self._variables[parts[1]] = parts[2]
            return True

        # Unset variable
        if cmd == "unset":
            if len(parts) < 2:
                raise ScriptSyntaxError("unset requires a variable name")
            self._variables.pop(parts[1], None)
            return True

        return False

    def _substitute_variables(self, command: str) -> str:
        """Substitute variables in a command.

        Variables use the syntax ${name} or $name.

        Args:
            command: Command with variable references.

        Returns:
            Command with variables substituted.
        """
        import re

        # Substitute ${name}
        def replace_braced(match):
            name = match.group(1)
            return self._variables.get(name, match.group(0))

        result = re.sub(r'\$\{(\w+)\}', replace_braced, command)

        # Substitute $name (word boundary)
        def replace_simple(match):
            name = match.group(1)
            return self._variables.get(name, match.group(0))

        result = re.sub(r'\$(\w+)(?=\s|$|[^\w])', replace_simple, result)

        return result

    def set_variable(self, name: str, value: str) -> None:
        """Set a script variable.

        Args:
            name: Variable name.
            value: Variable value.
        """
        self._variables[name] = value

    def get_variable(self, name: str) -> Optional[str]:
        """Get a script variable value.

        Args:
            name: Variable name.

        Returns:
            Variable value, or None if not set.
        """
        return self._variables.get(name)

    def clear_variables(self) -> None:
        """Clear all script variables."""
        self._variables.clear()

    @property
    def base_path(self) -> Path:
        """Get the base path for script resolution."""
        return self._base_path

    @base_path.setter
    def base_path(self, value: str) -> None:
        """Set the base path for script resolution."""
        self._base_path = Path(value)


def exec_file(console: "Console", path: str) -> ScriptResult:
    """Execute a script file.

    Convenience function for simple script execution.

    Args:
        console: The console to execute commands on.
        path: Path to the script file.

    Returns:
        ScriptResult with execution details.
    """
    executor = ScriptExecutor(console)
    return executor.exec_file(path)
