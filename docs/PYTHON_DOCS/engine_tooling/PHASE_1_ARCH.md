# PHASE 1 ARCHITECTURE: Core Editor Infrastructure

## Phase Overview

Phase 1 establishes the foundational editor infrastructure upon which all other tooling is built. This includes the application shell, undo system, console, logging, crash reporting, and hot reload capabilities.

## Components

### 1. Editor Application Shell (engine/tooling/editor/)

**Purpose**: Core editor framework with docking, panels, and extensibility

**Architecture**:
```
EditorApplication
    |
    +-- DockingManager
    |       +-- Panel (LEFT, RIGHT, TOP, BOTTOM, CENTER, FLOATING)
    |       +-- TabGroup
    |               +-- Tab (closable, dirty state)
    |
    +-- MenuBar
    |       +-- MenuItem (shortcuts, checkable, submenus)
    |
    +-- ToolBar
    |       +-- ToolButton (enabled, checked)
    |
    +-- StatusBar
            +-- StatusSection
```

**Key Patterns**:
- **Weak References**: Parent tracking uses `weakref` to prevent retain cycles
- **Layout Persistence**: `save_layout()` / `load_layout()` for session restore
- **Decorator Registration**: `@editor(category=)` marks editor-only classes
- **Reloadable Decorator**: `@reloadable(preserve=[], reinitialize=[])` for hot reload

**Integration Points**:
- Foundation Tracker for undo operations
- Foundation Mirror for property inspection
- Scene system for viewport rendering

### 2. Undo System (engine/tooling/undo/)

**Purpose**: Transaction-based undo/redo with branching history

**Architecture**:
```
UndoSystem (singleton)
    |
    +-- UndoStack / RedoStack
    |       +-- UndoEntry (name, timestamp, changes, metadata)
    |
    +-- TransactionManager
    |       +-- Transaction (state machine, savepoints)
    |               +-- CompositeCommand
    |                       +-- Command (execute, unexecute)
    |
    +-- HistoryView (branching)
    |       +-- HistoryNode (parent, children, current)
    |       +-- HistoryBranch (head, node_count)
    |
    +-- DirtyTracker
            +-- DirtyInfo (per document)
```

**Command Pattern Implementation**:
```python
Command (ABC)
    |-- SetFieldCommand      # Field mutations via mirror()
    |-- CallMethodCommand    # Method invocations with do/undo pairs
    |-- CreateObjectCommand  # Object creation with factory/deleter
    |-- DeleteObjectCommand  # Object deletion with restorer
    |-- CompositeCommand     # Atomic command groups
```

**Key Features**:
- Command merging for consecutive edits on same field
- Savepoint support within transactions
- Branching history (divergent paths preserved)
- Auto-grouping based on time window
- Thread-safe with RLock

### 3. Console System (engine/tooling/console/)

**Purpose**: Developer console with CVars and command execution

**Architecture**:
```
ConsoleUI
    |
    +-- CommandHistory
    |       +-- HistoryEntry (command, timestamp, success)
    |
    +-- CommandRegistry (singleton)
    |       +-- Command
    |               +-- CommandArg (typed arguments)
    |
    +-- CVarRegistry (singleton)
            +-- CVar (generic)
                    |-- IntCVar (min/max range)
                    |-- FloatCVar (precision)
                    |-- BoolCVar (flexible parsing)
                    |-- StringCVar (patterns, allowed values)
                    |-- EnumCVar (type-safe)
```

**CVar System Design**:
- Thread-safe with RLock per CVar and registry
- Change callbacks with `CVarChangeEvent`
- Flags: READONLY, CHEAT, ARCHIVE, REPLICATED, HIDDEN
- JSON persistence for ARCHIVE-flagged CVars
- Pattern matching with `fnmatch`

**Command System Design**:
- Permission levels: USER, DEVELOPER, CHEAT, ADMIN
- Decorators: `@command()`, `@cheat()`, `@admin()`, `@developer()`
- Introspection-based argument extraction from handler signatures
- `shlex.split` for proper quoted argument parsing
- Autocomplete with permission filtering

### 4. Logging System (engine/tooling/logging/)

**Purpose**: Comprehensive logging with multiple targets and filters

**Architecture**:
```
LogSystem (singleton)
    |
    +-- LogConfig (min_level, categories, async, buffer_size)
    |
    +-- LogTarget[]
    |       |-- ConsoleTarget (stdout/stderr, colors)
    |       |-- FileTarget (rotation, max_size, max_files)
    |       |-- NetworkTarget (UDP/TCP, JSON)
    |       |-- RingBufferTarget (crash dump)
    |       +-- CompositeTarget (fanout)
    |
    +-- LogFilter[]
    |       |-- LevelFilter (min/max)
    |       |-- CategoryFilter (include/exclude)
    |       |-- PatternFilter (regex)
    |       |-- RateLimitFilter (sliding window)
    |       |-- SamplingFilter (statistical)
    |       +-- DeduplicationFilter
    |
    +-- LogFormatter
            |-- DefaultFormatter (human-readable)
            |-- CompactFormatter (single-line)
            |-- JsonFormatter (structured)
            |-- ColorFormatter (ANSI)
            +-- SyslogFormatter (RFC 5424)
```

**Structured Logging**:
```
StructuredLogger
    +-- LogContext (thread-local)
            +-- Span (distributed tracing)
                    +-- SpanContext (trace_id, span_id, parent_span_id)
```

**Key Features**:
- 15 predefined categories (ENGINE, GAME, RENDER, PHYSICS, etc.)
- Async logging with background flush thread
- Rate limiting with sliding windows per category:level
- OpenTelemetry-compatible span model

### 5. Crash Reporting (engine/tooling/crash/)

**Purpose**: Exception capture, reporting, and analytics

**Architecture**:
```
CrashReporter
    |
    +-- CrashReport
    |       +-- ExceptionInfo (chained exceptions)
    |       +-- StackFrame (filename, line, function, locals)
    |       +-- SystemInfo (OS, CPU, memory, versions)
    |       +-- CrashContext (user actions, breadcrumbs)
    |
    +-- CrashUploader
    |       +-- UploadConfig (server, API key, compression)
    |       +-- AsyncCrashUploader (concurrent uploads)
    |
    +-- CrashAnalytics
    |       +-- CrashGroup (by fingerprint)
    |       +-- CrashPattern (stack, exception, temporal)
    |       +-- CrashTrend (time-series)
    |
    +-- SymbolServer
            +-- SymbolInfo (address, name, module, location)
            +-- ModuleInfo (base address, build ID)
            +-- SymbolCache (LRU with TTL)
```

**Assertions System**:
```python
@invariant(condition, message)      # Class invariant
@precondition(condition, message)   # Function precondition
@postcondition(condition, result_check)  # Function postcondition
@with_contracts(invariant_func)     # Full contract support
```

**Key Features**:
- MD5 fingerprinting for crash grouping
- Gzip compression for uploads
- Exponential backoff retry logic
- Pattern detection (stack patterns, version patterns, time patterns)

### 6. Hot Reload (engine/tooling/hotreload/)

**Purpose**: Live code reloading with state preservation

**Architecture**:
```
HotReloader
    |
    +-- ModuleWatcher
    |       +-- FileWatcher (platform-specific)
    |       +-- ModuleChangeEvent (type, path, timestamp)
    |
    +-- StatePreserver
    |       +-- StateSnapshot (schema hash, timestamp, state)
    |       +-- PreservationStrategy (7 strategies)
    |
    +-- SchemaHasher
    |       +-- SchemaComparison
    |       +-- SchemaChange (11 change types)
    |
    +-- ReloadCallbacks
    |       +-- CallbackRegistration
    |       +-- ReloadPhase (7 phases)
    |
    +-- DependencyTracker
            +-- DependencyGraph
            +-- ModuleNode (imports, imported_by)
```

**Preservation Strategies**:
1. SERIALIZER - Foundation's to_dict/from_dict
2. MIRROR - Foundation's reflection API
3. PICKLE_PROTOCOL - Python pickle
4. CUSTOM - User-defined
5. NONE - No preservation
6. SHALLOW_COPY - copy.copy
7. DEEP_COPY - copy.deepcopy

**Schema Change Detection**:
- Breaking: FIELD_REMOVED, FIELD_ADDED_WITHOUT_DEFAULT, FIELD_TYPE_CHANGED
- Non-breaking: FIELD_ADDED_WITH_DEFAULT, METHOD_ADDED, DEFAULT_VALUE_CHANGED
- Type widening detection (int->float safe, float->int breaking)

## Data Flow

### Application Startup
```
1. LogSystem.initialize(config)
2. CrashReporter.install_exception_handler()
3. ConsoleUI.initialize()
4. CVarRegistry.load(settings_path)
5. UndoSystem.initialize()
6. HotReloader.start_watching()
7. EditorApplication.initialize()
8. DockingManager.load_layout()
```

### Command Execution Flow
```
User Input -> ConsoleUI.execute()
    -> CommandRegistry.find(command_name)
    -> Command.check_permission(context)
    -> Command.parse_args(shlex.split(input))
    -> Command.handler(*args, **kwargs)
    -> CommandResult (status, message, return_value)
    -> CommandHistory.record(entry)
```

### Undo/Redo Flow
```
User Action -> UndoSystem.record(changes)
    -> Create UndoEntry
    -> Clear redo_stack
    -> Notify callbacks

User Undo -> UndoSystem.undo()
    -> Pop from undo_stack
    -> Execute unexecute() on each change
    -> Push to redo_stack
    -> Notify callbacks
```

## Integration Requirements

### Foundation Dependencies
- `foundation.tracker`: Change, Transaction, dirty flags
- `foundation.mirror`: ObjectMirror.get/set for field access

### Platform Dependencies
- `engine.platform.os.file_watcher`: File system monitoring

## Thread Safety Requirements

| Component | Lock Type | Scope |
|-----------|-----------|-------|
| UndoSystem | RLock | Instance-level |
| CVarRegistry | RLock | Global singleton |
| CommandRegistry | RLock | Global singleton |
| LogSystem | RLock | Instance + target-level |
| CrashReporter | None | Immutable after setup |
| HotReloader | RLock | Registry + watcher |

## Configuration

### Logging Configuration
```python
LogConfig(
    min_level=LogLevel.INFO,
    enabled_categories={LogCategory.ENGINE, LogCategory.GAME},
    async_logging=True,
    buffer_size=1000,
    flush_interval=0.1,
)
```

### Undo Configuration
```python
UndoSystemConfig(
    max_undo_levels=1000,
    max_redo_levels=1000,
    group_timeout_ms=500,
    enable_branching=False,
)
```

### Hot Reload Configuration
```python
ReloadConfig(
    debounce_seconds=0.1,
    exclude_patterns=["__pycache__", "*.pyc"],
    preserve_state=True,
    validate_schema=True,
)
```

## Testing Strategy

### Unit Tests
- Command argument parsing
- CVar validation and coercion
- Log filter matching
- Schema comparison logic
- Undo command execute/unexecute symmetry

### Integration Tests
- Full undo/redo cycle with Foundation Tracker
- Hot reload with state preservation
- Crash report generation and upload
- Console command execution with CVars

### Stress Tests
- 1000+ undo levels
- High-frequency logging (10,000 msgs/sec)
- Rapid module changes for hot reload
