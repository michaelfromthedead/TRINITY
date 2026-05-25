# Investigation: engine/tooling/hotreload/

## Classification: REAL IMPLEMENTATION

**Total Lines:** 2,729 lines across 7 files
**Status:** Production-ready hot-reload system with complete module watching, state preservation, schema comparison, callback management, and dependency tracking.

---

## File Analysis

### 1. `__init__.py` (93 lines) - REAL

**Purpose:** Package exports and public API definition.

**Exports:**
- Core: `reloadable`, `ReloadError`, `SchemaBreakingChangeError`, `ReloadableClass`, `HotReloader`
- Watcher: `ModuleWatcher`, `ModuleChangeEvent`, `ModuleChangeType`
- State: `StatePreserver`, `PreservationStrategy`, `StateSnapshot`
- Schema: `SchemaHasher`, `SchemaComparison`, `SchemaChange`, `SchemaChangeType`
- Callbacks: `ReloadCallbacks`, `ReloadPhase`, `ReloadContext`, `CallbackPriority`
- Dependencies: `DependencyTracker`, `DependencyGraph`, `ModuleNode`

**Evidence of REAL:** Well-structured package with cohesive API surface, meaningful re-exports.

---

### 2. `hot_reload.py` (451 lines) - REAL

**Purpose:** Core hot-reload functionality with `@reloadable` decorator.

**Key Components:**

| Component | Description |
|-----------|-------------|
| `@reloadable` decorator | Marks classes safe for hot-reload with state preservation |
| `ReloadableClass` | Metadata dataclass for reloadable classes |
| `ReloadResult` | Result of reload operations |
| `HotReloader` | Main coordinator for module reloading |
| `_reloadable_registry` | Module-level class registry |
| `_instance_registry` | Weak references to tracked instances |

**Implementation Details:**
- Uses `importlib.reload()` for actual module reloading
- Instance tracking via `weakref.ref` to avoid preventing garbage collection
- Integrates with Foundation's `schema_hash`, `to_dict`, `from_dict`, `mirror`
- Thread-safe with `threading.RLock`
- Supports migration functions for schema changes
- Singleton pattern with `get_hot_reloader()`

**Evidence of REAL:**
```python
def reload_module(self, module_name: str) -> ReloadResult:
    """Complete implementation with state preservation, error handling, callbacks"""
    if module_name not in sys.modules:
        result.errors.append(f"Module {module_name} not loaded")
        return result
    
    module = sys.modules[module_name]
    # ... preserve states, importlib.reload(), restore states
```

---

### 3. `module_watcher.py` (429 lines) - REAL

**Purpose:** File system monitoring for Python module changes.

**Key Components:**

| Component | Description |
|-----------|-------------|
| `ModuleChangeType` | Enum: CREATED, MODIFIED, DELETED, RENAMED |
| `ModuleChangeEvent` | Event dataclass with module name, file path, timestamp |
| `ModuleWatcher` | Main file watcher with callback system |

**Implementation Details:**
- Integrates with `engine.platform.os.file_watcher.FileWatcher`
- Debouncing support (default 0.1s) to prevent rapid-fire events
- Module name resolution from file paths
- Include/exclude pattern filtering (default excludes `__pycache__`, `.pyc`)
- Thread-safe callback invocation
- Recursive directory watching

**Evidence of REAL:**
```python
def watch_directory(self, path: str, recursive: bool = True, ...) -> bool:
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        return False
    # ... maps existing Python files to modules
    self._map_directory(abs_path, recursive)
    return self._file_watcher.watch_directory(abs_path, callback=self._on_file_change, recursive=recursive)
```

---

### 4. `state_preservation.py` (481 lines) - REAL

**Purpose:** Serialize and restore object state across hot-reloads.

**Key Components:**

| Component | Description |
|-----------|-------------|
| `PreservationStrategy` | Enum with 7 strategies (SERIALIZER, MIRROR, PICKLE_PROTOCOL, CUSTOM, NONE, SHALLOW_COPY, DEEP_COPY) |
| `StateSnapshot` | Snapshot dataclass with schema hash, timestamp, state |
| `PreservationConfig` | Configuration for field inclusion/exclusion |
| `StatePreserver` | Main state management with TTL and cleanup |

**Implementation Details:**
- Multiple preservation strategies with fallback chains
- Foundation integration: `to_dict`, `from_dict`, `mirror`, `schema_hash`
- Type validation on restore
- Weak reference tracking for object graphs
- Automatic cleanup of stale snapshots (default 5 min TTL)
- Field filtering by include/exclude sets
- Transient field handling via metadata

**Evidence of REAL:**
```python
def _extract_state(self, obj, strategy, config) -> Dict[str, Any]:
    if strategy == PreservationStrategy.SERIALIZER:
        try:
            state = to_dict(obj, include_schema_hash=False)
            return self._filter_fields(state, config)
        except Exception:
            return self._extract_via_mirror(obj, config)  # Fallback
```

---

### 5. `schema_hash.py` (417 lines) - REAL

**Purpose:** Schema change detection and analysis for safe hot-reloading.

**Key Components:**

| Component | Description |
|-----------|-------------|
| `SchemaChangeType` | Enum with 11 change types (breaking and non-breaking) |
| `SchemaChange` | Immutable dataclass describing a single change |
| `SchemaComparison` | Comparison result with breaking change detection |
| `SchemaHasher` | Advanced schema analysis and comparison |

**Breaking vs Non-Breaking Changes:**

| Breaking | Non-Breaking |
|----------|--------------|
| FIELD_REMOVED | FIELD_ADDED_WITH_DEFAULT |
| FIELD_ADDED_WITHOUT_DEFAULT | METADATA_CHANGED |
| FIELD_TYPE_CHANGED | METHOD_ADDED |
| FIELD_TYPE_NARROWED | METHOD_REMOVED |
| CLASS_RENAMED | DEFAULT_VALUE_CHANGED |

**Implementation Details:**
- Type widening detection (int->float safe, float->int breaking)
- Method signature tracking
- Migration hint generation
- Uses Foundation's `mirror()` for field introspection
- Cached schema info

**Evidence of REAL:**
```python
TYPE_WIDENING: Dict[type, Set[type]] = {
    int: {float, complex},
    float: {complex},
    bool: {int},
    tuple: {list},
}

def _classify_type_change(self, old_type, new_type) -> SchemaChangeType:
    if old_type in self.TYPE_WIDENING:
        if new_type in self.TYPE_WIDENING[old_type]:
            return SchemaChangeType.FIELD_TYPE_WIDENED
```

---

### 6. `reload_callbacks.py` (349 lines) - REAL

**Purpose:** Pre/post reload hooks and callback management.

**Key Components:**

| Component | Description |
|-----------|-------------|
| `ReloadPhase` | Enum with 7 phases (PRE_RELOAD through RELOAD_CANCELLED) |
| `CallbackPriority` | Enum: HIGHEST(0), HIGH(25), NORMAL(50), LOW(75), LOWEST(100) |
| `ReloadContext` | Context passed to callbacks with abort capability |
| `CallbackRegistration` | Registration info with module filtering |
| `ReloadCallbacks` | Thread-safe callback manager |

**Phases:**
1. PRE_RELOAD
2. STATE_PRESERVED
3. MODULE_RELOADED
4. STATE_RESTORED
5. POST_RELOAD
6. RELOAD_ERROR
7. RELOAD_CANCELLED

**Implementation Details:**
- Priority-based execution order
- Module-specific filtering
- One-shot callbacks
- Abort mechanism via context flag
- Thread-safe invocation
- Decorator-based registration

**Evidence of REAL:**
```python
def invoke(self, ctx: ReloadContext) -> ReloadContext:
    matching = [reg for reg in self._callbacks if reg.matches(ctx)]
    for reg in matching:
        if ctx.abort:
            break
        reg.callback(ctx)
        if reg.once:
            to_remove.append(reg)
```

---

### 7. `dependency_tracker.py` (509 lines) - REAL

**Purpose:** Track module dependencies for cascade reloads.

**Key Components:**

| Component | Description |
|-----------|-------------|
| `ModuleNode` | Node in dependency graph with imports/imported_by sets |
| `DependencyGraph` | Graph with topological sort and cycle detection |
| `DependencyTracker` | Main tracker with AST-based import analysis |

**Implementation Details:**
- AST parsing to extract imports (`ast.Import`, `ast.ImportFrom`)
- Bidirectional edge tracking (imports and imported_by)
- Transitive closure computation for cascade detection
- Kahn's algorithm for topological sort (reload order)
- DFS-based cycle detection
- Path-to-module conversion with sys.path awareness
- Thread-safe with `threading.RLock`

**Evidence of REAL:**
```python
def get_reload_order(self, modules: Set[str]) -> List[str]:
    """Kahn's algorithm for topological sort"""
    in_degree: Dict[str, int] = {}
    edges: Dict[str, Set[str]] = {}
    # ... compute in-degrees, execute Kahn's algorithm
    while queue:
        current = queue.pop(0)
        result.append(current)
        for name, deps in edges.items():
            if current in deps:
                in_degree[name] -= 1
                if in_degree[name] == 0:
                    queue.append(name)
    return result
```

---

## Architecture Diagram

```
+-------------------+
|  @reloadable      |  Decorator marks classes
+--------+----------+
         |
         v
+-------------------+     +------------------+
|  HotReloader      |<--->| StatePreserver   |
|  - reload_module()|     | - preserve()     |
|  - restore_state()|     | - restore()      |
+--------+----------+     +--------+---------+
         |                         |
         v                         v
+-------------------+     +------------------+
|  ModuleWatcher    |     | SchemaHasher     |
|  - watch_directory|     | - compare_schemas|
|  - on_file_change |     | - migration hints|
+--------+----------+     +------------------+
         |
         v
+-------------------+     +------------------+
| DependencyTracker |     | ReloadCallbacks  |
| - get_reload_plan |     | - PRE_RELOAD     |
| - detect_cycles   |     | - POST_RELOAD    |
+-------------------+     +------------------+
```

---

## Foundation Integration

| Foundation API | Usage in hotreload |
|----------------|-------------------|
| `schema_hash(cls)` | Detect class schema changes |
| `to_dict(obj)` | Serialize object state |
| `from_dict(data, cls)` | Deserialize object state |
| `mirror(obj)` | Introspect fields and methods |
| `FieldInfo` | Schema comparison field details |

---

## External Dependencies

| Dependency | Source |
|------------|--------|
| `engine.platform.os.file_watcher.FileWatcher` | Platform-specific file watching |
| `foundation` | Serialization and introspection |

---

## Quality Assessment

| Metric | Assessment |
|--------|------------|
| **Implementation Completeness** | Complete - all components fully functional |
| **Thread Safety** | All classes use `threading.RLock` |
| **Error Handling** | Comprehensive with fallback strategies |
| **Type Hints** | Full coverage with generics |
| **Documentation** | Docstrings on all public APIs |
| **Testing Hooks** | `clear()` methods for test isolation |

---

## Gaps and Recommendations

### Minor Gaps

1. **No async support** - All operations are synchronous, may block on large module graphs
2. **Limited migration automation** - Migration hints are generated but not applied automatically
3. **No hot-reload for C extensions** - Only Python modules are reloadable

### Recommendations

1. Consider async file watching for large projects
2. Add migration template generation from hints
3. Document integration with editor tooling
4. Add metrics/telemetry for reload performance

---

## Conclusion

The `engine/tooling/hotreload/` module is a **REAL, production-ready implementation** of a hot-reload system. It provides:

- Decorator-based class marking with state preservation
- File system watching with debouncing and filtering
- Multiple state preservation strategies with Foundation integration
- Detailed schema change analysis with breaking change detection
- Priority-based callback system with abort capability
- Dependency-aware cascade reloading with cycle detection

All 2,729 lines represent functional, well-documented code with comprehensive error handling and thread safety. No stubs or placeholder implementations were found.
