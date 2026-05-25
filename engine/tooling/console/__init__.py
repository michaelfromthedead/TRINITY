"""Console subsystem for the AI Game Engine.

This module provides a comprehensive developer console with:
- Command registration and execution
- Configuration variables (CVars)
- Command history with persistence
- Autocomplete functionality
- Permission levels for commands
"""

from .console_ui import ConsoleUI, ConsoleMode, ConsoleConfig
from .console_commands import (
    CommandRegistry,
    Command,
    CommandResult,
    CommandContext,
    cheat,
    admin,
    developer,
)
from .cvar_system import (
    CVar,
    CVarRegistry,
    CVarType,
    CVarFlags,
    IntCVar,
    FloatCVar,
    BoolCVar,
    StringCVar,
    EnumCVar,
)
from .command_history import CommandHistory, HistoryEntry

__all__ = [
    # Console UI
    "ConsoleUI",
    "ConsoleMode",
    "ConsoleConfig",
    # Commands
    "CommandRegistry",
    "Command",
    "CommandResult",
    "CommandContext",
    "cheat",
    "admin",
    "developer",
    # CVars
    "CVar",
    "CVarRegistry",
    "CVarType",
    "CVarFlags",
    "IntCVar",
    "FloatCVar",
    "BoolCVar",
    "StringCVar",
    "EnumCVar",
    # History
    "CommandHistory",
    "HistoryEntry",
]
