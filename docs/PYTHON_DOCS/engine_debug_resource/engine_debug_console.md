# Engine Debug Console Investigation

**Status**: REAL (Production-Ready Implementation)  
**Files Analyzed**: 8 files, 2,923 lines total  
**Date**: 2026-05-22

## Summary

The debug console system is a **fully implemented, production-quality** in-game console providing:

- **CVars**: Typed console variables with flags, bounds, and change notifications
- **Commands**: Command registration/execution with validation and built-in commands
- **Console**: Main console with modes, history, input parsing, and output management
- **Autocomplete**: Tab completion for commands, CVars, and history
- **Aliases**: Command alias mapping with argument substitution
- **Scripting**: .cfg file execution with includes and variable substitution

## Classification: REAL

All 8 modules contain complete, functional implementations:

| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `cvar.py` | 594 | REAL | Typed CVars with flags, bounds, callbacks |
| `console.py` | 518 | REAL | Main console with modes, history, execution |
| `commands.py` | 483 | REAL | Command registry with 8 built-in commands |
| `scripting.py` | 468 | REAL | Script file parsing and execution |
| `autocomplete.py` | 339 | REAL | Tab completion with multiple completers |
| `aliases.py` | 300 | REAL | Alias registry with argument substitution |
| `__init__.py` | 133 | REAL | Public API exports |
| `config.py` | 88 | REAL | Configuration constants |

## Architecture

```
Console (central hub)
    |
    +-- CommandRegistry (built-in + custom commands)
    +-- CVarRegistry (singleton, typed variables)
    +-- AliasRegistry (command shortcuts)
    +-- Autocomplete (tab completion)
    +-- ScriptExecutor (cfg file execution)
```

## CVar System (cvar.py)

### CVarFlags

```python
class CVarFlags(Flag):
    NONE = 0
    READONLY = auto()    # Cannot be modified at runtime
    CHEAT = auto()       # Requires cheats enabled
    CONFIG = auto()      # Saved to/loaded from config files
    SCALABILITY = auto() # Affected by scalability settings
```

### CVar Class

Generic typed console variable supporting `int`, `float`, `bool`, `str`:

```python
CVar(
    name: str,              # e.g., "r.VSync"
    default: CVarValue,     # Type inferred from default
    flags: CVarFlags,
    description: str,
    min_value: Optional,    # For numeric types
    max_value: Optional
)
```

**Features**:
- Type validation and coercion (string parsing for console input)
- Bounds checking for numeric types
- Change callbacks via `on_change(callback)`
- Auto-registration with singleton `CVarRegistry`
- Boolean parsing: "true"/"1"/"yes"/"on" and "false"/"0"/"no"/"off"

### CVarRegistry

Singleton registry providing:
- `get(name)`, `find(pattern)`, `all()` - Lookup
- `categories()`, `by_category(cat)` - Category-based organization
- `with_flags(flags)` - Filter by flags
- `reset_all()` - Bulk reset
- `export_config()`, `import_config()` - Persistence

### Exceptions

- `CVarTypeError` - Value type mismatch
- `CVarReadOnlyError` - Modifying READONLY cvar
- `CVarCheatError` - Accessing CHEAT cvar without cheats
- `CVarBoundsError` - Value outside min/max

## Command System (commands.py)

### CommandFlags

```python
class CommandFlags(Flag):
    NONE = 0
    CHEAT = auto()   # Requires cheats enabled
    HIDDEN = auto()  # Not shown in help/list
    BUILTIN = auto() # System command
```

### Command Class

```python
Command(
    name: str,
    handler: Callable[..., Optional[str]],
    description: str,
    flags: CommandFlags,
    usage: str,           # e.g., "teleport <x> <y> <z>"
    min_args: int,
    max_args: int         # -1 for unlimited
)
```

### Built-in Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `help` | Show help or list commands | `help [command]` |
| `list` | List available commands | `list [filter]` |
| `clear` | Clear console output | `clear` |
| `echo` | Print message | `echo <message>` |
| `cvarlist` | List console variables | `cvarlist [filter]` |
| `reset` | Reset CVar to default | `reset <cvar>` |
| `version` | Show engine version | `version` |
| `quit` | Quit application | `quit` |

### CommandRegistry

- `register(name, handler, ...)` - Register command
- `execute(name, args)` - Execute with validation
- `get_completions(partial)` - Tab completion
- `visible()` - Non-hidden commands
- Cheat access control via `cheats_enabled`

### Exceptions

- `CommandNotFoundError`
- `CommandAccessError` - Cheat required
- `CommandExecutionError` - Argument or runtime error

## Main Console (console.py)

### Console Modes

```python
class ConsoleMode(Enum):
    OVERLAY = auto()     # Transparent overlay
    FULLSCREEN = auto()  # Dedicated fullscreen
    MINI = auto()        # Small input bar
```

### Console Class

Central hub integrating all subsystems:

```python
Console(
    command_registry: Optional[CommandRegistry],
    alias_registry: Optional[AliasRegistry],
    max_history: int = 100,
    max_output: int = 1000
)
```

**Key Methods**:

| Method | Description |
|--------|-------------|
| `execute(input)` | Execute command/CVar with alias expansion |
| `toggle()` | Toggle visibility |
| `cycle_mode()` | Cycle through display modes |
| `complete(partial)` | Get tab completions |
| `register_command(...)` | Register custom command |
| `register_alias(...)` | Register alias |
| `history_up()`, `history_down()` | Navigate history |
| `clear()` | Clear output |

**Features**:
- Input parsing with quote handling and escapes
- Semicolon-separated multi-command execution
- Alias expansion before execution
- CVar get (name only) and set (name value) support
- History with duplicate deduplication
- Output with levels (info, warning, error, input)
- Callbacks: `on_output`, `on_clear`, `on_visibility_change`
- Special markers: `__CLEAR__`, `__QUIT__`

### ConsoleOutput

```python
@dataclass
class ConsoleOutput:
    text: str
    level: str = "info"  # info, warning, error, input
    timestamp: float
```

## Autocomplete System (autocomplete.py)

### Autocomplete Class

Tab completion for:
- Command names (non-hidden)
- CVar names
- History entries

```python
Autocomplete(
    command_registry: CommandRegistry,
    max_suggestions: int = 20
)
```

**Methods**:
- `get_completions(partial)` - Simple string list
- `get_completion_results(partial)` - Detailed `CompletionResult` objects
- `get_common_prefix(partial)` - For extending to longest match
- `add_to_history(command)` - Track entered commands

### CompletionResult

```python
@dataclass
class CompletionResult:
    text: str
    kind: str          # "command", "cvar", "alias", "history"
    description: str
```

### ArgumentCompleters

Extensible argument completion:

| Completer | Purpose |
|-----------|---------|
| `ArgumentCompleter` | Base class |
| `FileCompleter` | File path completion with extension filtering |
| `EnumCompleter` | Fixed value list completion |

## Alias System (aliases.py)

### Alias Class

```python
@dataclass
class Alias:
    name: str
    expansion: str      # Command(s) to expand to
    description: str
```

### AliasRegistry

```python
AliasRegistry(
    max_aliases: int = 1000,
    max_recursion_depth: int = 10
)
```

**Features**:
- Recursive alias expansion with depth limit
- Argument substitution: `$1`, `$2`, ..., `$*`, `$@`
- Multi-command expansion (semicolon-separated)
- Import/export for persistence

**Example**:
```python
registry.register("godmode", "god; infinite_ammo $1; fly")
registry.expand("godmode pistol")
# -> "god; infinite_ammo pistol; fly"
```

### Exceptions

- `AliasRecursionError` - Infinite loop detected
- `AliasLimitError` - Max aliases reached

## Scripting System (scripting.py)

### ScriptExecutor

Executes .cfg files with:

```python
ScriptExecutor(
    console: Console,
    base_path: Optional[str],
    max_lines: int = 10000,
    max_include_depth: int = 10,
    max_line_length: int = 4096
)
```

**Methods**:
- `exec_file(path, stop_on_error)` - Execute file
- `exec_string(script)` - Execute string
- `set_variable(name, value)` - Set script variable
- `get_variable(name)` - Get script variable

### Script Features

| Feature | Syntax |
|---------|--------|
| Comments | `//` or `#` (line start or inline) |
| Variables | `${name}` or `$name` |
| Set variable | `set varname value` |
| Unset variable | `unset varname` |
| Include file | `exec path.cfg` or `include path.cfg` |

### Script Search Paths

Files resolved in order:
1. Absolute path
2. Base path
3. `scripts/`, `config/`, `cfg/` subdirectories

### ScriptResult

```python
@dataclass
class ScriptResult:
    path: str
    lines_executed: int
    lines_skipped: int      # Comments and empty lines
    errors: List[tuple]     # (line_number, error_message)
    
    @property
    def success(self) -> bool
```

### Exceptions

- `ScriptFileNotFoundError`
- `ScriptSyntaxError`
- `ScriptLimitError` - Line count, include depth, or line length exceeded

## Configuration (config.py)

Frozen dataclasses with defaults:

```python
ConsoleDefaults:
    MAX_HISTORY = 100
    MAX_OUTPUT = 1000

AutocompleteConfig:
    MAX_SUGGESTIONS = 20
    MAX_HISTORY = 100

AliasConfig:
    MAX_RECURSION_DEPTH = 10
    MAX_ALIASES = 1000

ScriptConfig:
    MAX_LINES = 10000
    MAX_INCLUDE_DEPTH = 10
    MAX_LINE_LENGTH = 4096

CVarConfig:
    MAX_CALLBACKS = 100
    MAX_NAME_LENGTH = 256
```

Singleton instances: `CONSOLE`, `AUTOCOMPLETE`, `ALIAS`, `SCRIPT`, `CVAR`

## Public API (__init__.py)

Exports 33 symbols across all subsystems:

```python
# CVar
CVar, CVarFlags, CVarRegistry, CVarTypeError, CVarReadOnlyError, CVarCheatError, CVarBoundsError

# Commands
Command, CommandFlags, CommandRegistry, CommandError, CommandNotFoundError, CommandAccessError, CommandExecutionError

# Console
Console, ConsoleMode, ConsoleOutput

# Autocomplete
Autocomplete, CompletionResult, ArgumentCompleter, FileCompleter, EnumCompleter

# Aliases
Alias, AliasRegistry, AliasError, AliasRecursionError, AliasLimitError

# Scripting
ScriptExecutor, ScriptResult, ScriptLine, ScriptError, ScriptFileNotFoundError, ScriptSyntaxError, ScriptLimitError, exec_file

# Config
ConsoleDefaults, AutocompleteConfig, AliasConfig, ScriptConfig, CVarConfig, CONSOLE, AUTOCOMPLETE, ALIAS, SCRIPT, CVAR
```

## Usage Examples

### Basic Console Usage

```python
from engine.debug.console import Console, CVar, CVarFlags

# Register CVar
r_vsync = CVar("r.VSync", default=1, flags=CVarFlags.CONFIG)
r_vsync.on_change(lambda old, new: print(f"VSync: {old} -> {new}"))

# Create console
console = Console()

# Execute commands
console.execute("r.VSync 0")        # Set CVar
console.execute("r.VSync")          # Get CVar
console.execute("help")             # Built-in command
console.execute("cvarlist r.*")     # Filter CVars

# Register custom command
console.register_command(
    "teleport",
    lambda x, y, z: f"Teleported to ({x}, {y}, {z})",
    description="Teleport player",
    usage="teleport <x> <y> <z>",
    min_args=3,
    max_args=3
)

# Register alias
console.register_alias("tp", "teleport $1 $2 $3")
console.execute("tp 100 200 50")
```

### Script Execution

```python
from engine.debug.console import Console, ScriptExecutor

console = Console()
executor = ScriptExecutor(console, base_path="/game/config")

# Execute startup script
result = executor.exec_file("startup.cfg")
if result.success:
    print(f"Executed {result.lines_executed} commands")
else:
    for line_num, error in result.errors:
        print(f"Line {line_num}: {error}")
```

### Example .cfg File

```cfg
// startup.cfg - Game initialization

# Graphics settings
r.VSync 1
r.MaxFPS 60
r.ShadowQuality 3

# Set variables
set PLAYER_NAME "DefaultPlayer"

# Use variable
echo Welcome, ${PLAYER_NAME}!

# Include other configs
exec graphics_low.cfg
```

## Quality Indicators

### Strengths

1. **Type Safety**: Generic CVar with type inference and validation
2. **Robust Parsing**: Quote handling, escapes, inline comment stripping
3. **Security**: Cheat flags, readonly protection, recursion limits
4. **Extensibility**: Custom commands, argument completers, aliases
5. **Configuration**: All limits configurable via frozen dataclasses
6. **Error Handling**: Rich exception hierarchy with descriptive messages
7. **Callbacks**: Change notifications for UI integration
8. **Documentation**: Comprehensive docstrings with examples

### Design Patterns

- **Singleton**: CVarRegistry.instance() for global access
- **Dataclass**: Immutable configuration, structured data
- **Flag Enum**: Composable flags for commands/CVars
- **Registry Pattern**: Centralized command/alias/cvar management
- **Callback Pattern**: Change notifications throughout

### Integration Points

- UI callbacks for output, clear, visibility changes
- Script executor delegates to console for command execution
- Autocomplete integrates with command and CVar registries
- Console bridges commands, CVars, and aliases

## Dependencies

Internal only - no external package dependencies:
- `dataclasses`, `enum`, `typing` (stdlib)
- `fnmatch`, `re`, `os`, `pathlib`, `logging`, `time` (stdlib)

## Recommendations

1. **Tests**: Add comprehensive unit tests for edge cases
2. **Persistence**: Add CVar save/load to JSON/INI
3. **History Persistence**: Save command history between sessions
4. **Command Argument Completers**: Wire up `_complete_argument` in Autocomplete
5. **Batch Commands**: Consider `{ }` grouping for atomic execution
6. **Remote Console**: Network RCON support for dedicated servers

## Files Referenced

- `/home/user/dev/USER/PROJECTS_VOID/TRINITY/engine/debug/console/__init__.py`
- `/home/user/dev/USER/PROJECTS_VOID/TRINITY/engine/debug/console/config.py`
- `/home/user/dev/USER/PROJECTS_VOID/TRINITY/engine/debug/console/cvar.py`
- `/home/user/dev/USER/PROJECTS_VOID/TRINITY/engine/debug/console/console.py`
- `/home/user/dev/USER/PROJECTS_VOID/TRINITY/engine/debug/console/commands.py`
- `/home/user/dev/USER/PROJECTS_VOID/TRINITY/engine/debug/console/scripting.py`
- `/home/user/dev/USER/PROJECTS_VOID/TRINITY/engine/debug/console/autocomplete.py`
- `/home/user/dev/USER/PROJECTS_VOID/TRINITY/engine/debug/console/aliases.py`
