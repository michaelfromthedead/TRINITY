"""Debug Console System - In-game console with CVars, commands, and scripting.

This module provides a comprehensive debug console system with:

- **CVars**: Typed console variables with change notifications
- **Commands**: Command registration and execution with arguments
- **Console**: Main console with modes, history, and output
- **Autocomplete**: Tab completion for commands and CVars
- **Aliases**: Command alias mapping and expansion
- **Scripting**: Script file execution (.cfg files)

Example:
    >>> from engine.debug.console import Console, CVar, CVarFlags
    >>>
    >>> # Register a CVar
    >>> r_vsync = CVar("r.VSync", default=1, flags=CVarFlags.CONFIG)
    >>> r_vsync.on_change(lambda old, new: print(f"VSync: {old} -> {new}"))
    >>>
    >>> # Create console
    >>> console = Console()
    >>> console.execute("r.VSync 0")
    VSync: 1 -> 0
    >>>
    >>> # Register custom command
    >>> console.register_command("greet", lambda name: f"Hello, {name}!")
    >>> console.execute("greet World")
    'Hello, World!'
"""

from .aliases import Alias, AliasError, AliasLimitError, AliasRecursionError, AliasRegistry
from .config import (
    ALIAS,
    AUTOCOMPLETE,
    CONSOLE,
    CVAR,
    SCRIPT,
    AliasConfig,
    AutocompleteConfig,
    ConsoleDefaults,
    CVarConfig,
    ScriptConfig,
)
from .autocomplete import (
    ArgumentCompleter,
    Autocomplete,
    CompletionResult,
    EnumCompleter,
    FileCompleter,
)
from .commands import (
    Command,
    CommandAccessError,
    CommandError,
    CommandExecutionError,
    CommandFlags,
    CommandNotFoundError,
    CommandRegistry,
)
from .console import Console, ConsoleMode, ConsoleOutput
from .cvar import (
    CVar,
    CVarBoundsError,
    CVarCheatError,
    CVarFlags,
    CVarReadOnlyError,
    CVarRegistry,
    CVarTypeError,
)
from .scripting import (
    ScriptError,
    ScriptExecutor,
    ScriptFileNotFoundError,
    ScriptLimitError,
    ScriptLine,
    ScriptResult,
    ScriptSyntaxError,
    exec_file,
)

__all__ = [
    # CVar system
    "CVar",
    "CVarFlags",
    "CVarRegistry",
    "CVarTypeError",
    "CVarReadOnlyError",
    "CVarCheatError",
    "CVarBoundsError",
    # Command system
    "Command",
    "CommandFlags",
    "CommandRegistry",
    "CommandError",
    "CommandNotFoundError",
    "CommandAccessError",
    "CommandExecutionError",
    # Console
    "Console",
    "ConsoleMode",
    "ConsoleOutput",
    # Autocomplete
    "Autocomplete",
    "CompletionResult",
    "ArgumentCompleter",
    "FileCompleter",
    "EnumCompleter",
    # Aliases
    "Alias",
    "AliasRegistry",
    "AliasError",
    "AliasRecursionError",
    "AliasLimitError",
    # Scripting
    "ScriptExecutor",
    "ScriptResult",
    "ScriptLine",
    "ScriptError",
    "ScriptFileNotFoundError",
    "ScriptSyntaxError",
    "ScriptLimitError",
    "exec_file",
    # Config
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
