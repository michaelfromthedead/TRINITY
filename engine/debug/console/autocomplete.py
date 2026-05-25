"""Autocomplete - Tab completion for console input.

This module provides autocomplete functionality with:
- Command name completion
- CVar name completion
- History-based suggestions
- Partial matching

Example:
    >>> autocomplete = Autocomplete(command_registry)
    >>> autocomplete.get_completions("tel")
    ['teleport']
    >>> autocomplete.get_completions("r.V")
    ['r.VSync', 'r.ViewDistance']
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

from .commands import CommandFlags, CommandRegistry
from .config import AUTOCOMPLETE as AUTOCOMPLETE_CONFIG
from .cvar import CVarRegistry


@dataclass
class CompletionResult:
    """A single completion suggestion.

    Attributes:
        text: The completion text.
        kind: Type of completion (command, cvar, alias, history).
        description: Optional description for display.
    """
    text: str
    kind: str
    description: str = ""


class Autocomplete:
    """Provides autocomplete suggestions for console input.

    Supports completion of:
    - Command names
    - CVar names
    - Alias names
    - History entries

    Example:
        >>> autocomplete = Autocomplete(command_registry)
        >>> results = autocomplete.get_completions("he")
        >>> [r.text for r in results]
        ['help']
    """

    def __init__(
        self,
        command_registry: Optional[CommandRegistry] = None,
        max_suggestions: Optional[int] = None,
    ) -> None:
        """Initialize the autocomplete system.

        Args:
            command_registry: The command registry to use.
            max_suggestions: Maximum number of suggestions to return (default from config).
        """
        self._command_registry = command_registry or CommandRegistry()
        self._max_suggestions = max_suggestions if max_suggestions is not None else AUTOCOMPLETE_CONFIG.MAX_SUGGESTIONS
        self._history: List[str] = []
        self._max_history = AUTOCOMPLETE_CONFIG.MAX_HISTORY

    def get_completions(self, partial: str) -> List[str]:
        """Get completion strings for partial input.

        This is the simple interface that returns just the completion strings.

        Args:
            partial: Partial input to complete.

        Returns:
            List of matching completion strings.
        """
        results = self.get_completion_results(partial)
        return [r.text for r in results]

    def get_completion_results(self, partial: str) -> List[CompletionResult]:
        """Get detailed completion results for partial input.

        Args:
            partial: Partial input to complete.

        Returns:
            List of CompletionResult objects with metadata.
        """
        if not partial:
            return []

        partial = partial.strip()
        parts = partial.split()

        if len(parts) <= 1:
            # Complete command or CVar name
            return self._complete_name(partial)
        else:
            # Complete argument based on command
            command_name = parts[0]
            arg_partial = parts[-1] if len(parts) > 1 else ""
            return self._complete_argument(command_name, arg_partial, len(parts) - 1)

    def _complete_name(self, partial: str) -> List[CompletionResult]:
        """Complete a command or CVar name.

        Args:
            partial: Partial name to complete.

        Returns:
            List of matching completions.
        """
        results: List[CompletionResult] = []
        partial_lower = partial.lower()
        seen: Set[str] = set()

        # Complete commands
        for cmd in self._command_registry.all():
            if CommandFlags.HIDDEN not in cmd.flags:
                if cmd.name.lower().startswith(partial_lower):
                    if cmd.name not in seen:
                        results.append(CompletionResult(
                            text=cmd.name,
                            kind="command",
                            description=cmd.description[:50] if cmd.description else "",
                        ))
                        seen.add(cmd.name)

        # Complete CVars
        cvar_registry = CVarRegistry.instance()
        for cvar in cvar_registry.all():
            if cvar.name.lower().startswith(partial_lower):
                if cvar.name not in seen:
                    results.append(CompletionResult(
                        text=cvar.name,
                        kind="cvar",
                        description=f"= {cvar._value}",
                    ))
                    seen.add(cvar.name)

        # Complete from history
        for entry in reversed(self._history):
            if entry.lower().startswith(partial_lower):
                if entry not in seen:
                    results.append(CompletionResult(
                        text=entry,
                        kind="history",
                        description="",
                    ))
                    seen.add(entry)

        # Sort: exact prefix matches first, then alphabetically
        results.sort(key=lambda r: (not r.text.startswith(partial), r.text.lower()))

        return results[:self._max_suggestions]

    def _complete_argument(
        self,
        command_name: str,
        partial: str,
        arg_index: int,
    ) -> List[CompletionResult]:
        """Complete a command argument.

        Args:
            command_name: The command being typed.
            partial: Partial argument text.
            arg_index: Which argument (0-indexed).

        Returns:
            List of matching completions.
        """
        # For now, return empty list
        # In a full implementation, commands could register argument completers
        return []

    def add_to_history(self, command: str) -> None:
        """Add a command to completion history.

        Args:
            command: The command to add.
        """
        # Remove duplicates
        if command in self._history:
            self._history.remove(command)

        self._history.append(command)

        # Limit history size
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def clear_history(self) -> None:
        """Clear the completion history."""
        self._history.clear()

    def get_common_prefix(self, partial: str) -> str:
        """Get the longest common prefix of all completions.

        Useful for extending input to the common prefix on tab.

        Args:
            partial: Partial input to complete.

        Returns:
            The longest common prefix of all matches.
        """
        completions = self.get_completions(partial)

        if not completions:
            return partial

        if len(completions) == 1:
            return completions[0]

        # Find common prefix
        prefix = completions[0]
        for completion in completions[1:]:
            while not completion.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return partial

        return prefix if len(prefix) > len(partial) else partial


class ArgumentCompleter:
    """Base class for command argument completers.

    Commands can register custom argument completers to provide
    context-aware completions for their arguments.
    """

    def get_completions(
        self,
        partial: str,
        arg_index: int,
        previous_args: List[str],
    ) -> List[CompletionResult]:
        """Get completions for an argument.

        Args:
            partial: Partial argument text.
            arg_index: Which argument (0-indexed).
            previous_args: Previously typed arguments.

        Returns:
            List of matching completions.
        """
        return []


class FileCompleter(ArgumentCompleter):
    """Completes file paths."""

    def __init__(self, extensions: Optional[List[str]] = None) -> None:
        """Initialize file completer.

        Args:
            extensions: Optional list of file extensions to filter by.
        """
        self._extensions = extensions

    def get_completions(
        self,
        partial: str,
        arg_index: int,
        previous_args: List[str],
    ) -> List[CompletionResult]:
        """Get file path completions."""
        import os

        results = []

        # Determine directory to search
        if os.path.sep in partial:
            directory = os.path.dirname(partial)
            prefix = os.path.basename(partial)
        else:
            directory = "."
            prefix = partial

        try:
            for entry in os.listdir(directory):
                if entry.lower().startswith(prefix.lower()):
                    full_path = os.path.join(directory, entry)

                    # Filter by extension if specified
                    if self._extensions and os.path.isfile(full_path):
                        if not any(entry.endswith(ext) for ext in self._extensions):
                            continue

                    kind = "directory" if os.path.isdir(full_path) else "file"
                    results.append(CompletionResult(
                        text=full_path,
                        kind=kind,
                    ))
        except OSError:
            pass

        return results


class EnumCompleter(ArgumentCompleter):
    """Completes from a list of fixed values."""

    def __init__(self, values: List[str]) -> None:
        """Initialize enum completer.

        Args:
            values: List of valid values.
        """
        self._values = values

    def get_completions(
        self,
        partial: str,
        arg_index: int,
        previous_args: List[str],
    ) -> List[CompletionResult]:
        """Get completions from the value list."""
        partial_lower = partial.lower()
        results = []

        for value in self._values:
            if value.lower().startswith(partial_lower):
                results.append(CompletionResult(
                    text=value,
                    kind="value",
                ))

        return results
