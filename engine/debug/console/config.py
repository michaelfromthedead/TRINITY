"""
Console Configuration constants for the AI Game Engine Debug Console.

Centralizes magic numbers, default values, and configurable settings
for the debug console system to improve maintainability and allow customization.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConsoleDefaults:
    """Default console configuration values."""

    MAX_HISTORY: int = 100
    """Maximum number of commands to remember in history."""

    MAX_OUTPUT: int = 1000
    """Maximum number of output lines to keep in the console buffer."""


@dataclass(frozen=True)
class AutocompleteConfig:
    """Autocomplete configuration values."""

    MAX_SUGGESTIONS: int = 20
    """Maximum number of autocomplete suggestions to return."""

    MAX_HISTORY: int = 100
    """Maximum number of entries to keep in autocomplete history."""


@dataclass(frozen=True)
class AliasConfig:
    """Alias system configuration values."""

    MAX_RECURSION_DEPTH: int = 10
    """Maximum recursion depth for alias expansion to prevent infinite loops."""

    MAX_ALIASES: int = 1000
    """Maximum number of aliases that can be registered."""


@dataclass(frozen=True)
class ScriptConfig:
    """Script execution configuration values."""

    MAX_LINES: int = 10000
    """Maximum number of lines allowed in a script file."""

    MAX_INCLUDE_DEPTH: int = 10
    """Maximum depth of nested script includes."""

    MAX_LINE_LENGTH: int = 4096
    """Maximum length of a single script line."""


@dataclass(frozen=True)
class CVarConfig:
    """CVar system configuration values."""

    MAX_CALLBACKS: int = 100
    """Maximum number of change callbacks per CVar."""

    MAX_NAME_LENGTH: int = 256
    """Maximum length of a CVar name."""


# Singleton instances for easy access
CONSOLE = ConsoleDefaults()
AUTOCOMPLETE = AutocompleteConfig()
ALIAS = AliasConfig()
SCRIPT = ScriptConfig()
CVAR = CVarConfig()


__all__ = [
    "ConsoleDefaults",
    "AutocompleteConfig",
    "AliasConfig",
    "ScriptConfig",
    "CVarConfig",
    "CONSOLE",
    "AUTOCOMPLETE",
    "ALIAS",
    "SCRIPT",
    "CVAR",
]
