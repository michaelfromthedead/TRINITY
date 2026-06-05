"""T-CC-3.3: Python script hot-reload with state preservation.

Provides a unified script reloading system for the TRINITY editor with:
- ScriptReloader: Automatic detection and reload of Python module changes
- ScriptState: Captures and restores script instance state
- ModuleSwapper: Safely swaps modules with rollback on failure
- ExecutionContext: Tracks execution state for resume after reload

This builds on the hotreload infrastructure but provides editor-specific
integration with the serialization system (@serializable decorator).
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib
import importlib.util
import inspect
import os
import sys
import threading
import time
import traceback
import types
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from engine.core.serialization import (
    Serializable,
    SerializationContext,
    SerializationFormat,
    SchemaVersion,
    SchemaInfo,
    SerializationError,
    serializable,
)
from engine.tooling.hotreload import (
    ModuleWatcher,
    ModuleChangeEvent,
    ModuleChangeType,
    StatePreserver,
    PreservationStrategy,
    StateSnapshot,
    SchemaHasher,
    ReloadCallbacks,
    ReloadPhase,
    ReloadContext,
    DependencyTracker,
)


T = TypeVar("T")


class ReloadState(Enum):
    """State of the script reload process."""
    IDLE = auto()
    DETECTING = auto()
    PRESERVING_STATE = auto()
    UNLOADING = auto()
    LOADING = auto()
    RESTORING_STATE = auto()
    RESUMING = auto()
    ROLLING_BACK = auto()
    COMPLETED = auto()
    FAILED = auto()


class ReloadStrategy(Enum):
    """Strategy for handling module reload."""
    IMMEDIATE = auto()      # Reload immediately on change detection
    DEBOUNCED = auto()      # Wait for changes to settle
    MANUAL = auto()         # Only reload on explicit request
    SCHEDULED = auto()      # Reload at next safe point (e.g., between frames)


class ReloadErrorType(Enum):
    """Types of reload errors."""
    DETECTION_FAILED = auto()
    SERIALIZATION_FAILED = auto()
    UNLOAD_FAILED = auto()
    IMPORT_ERROR = auto()
    SYNTAX_ERROR = auto()
    SCHEMA_MISMATCH = auto()
    RESTORE_FAILED = auto()
    RESUME_FAILED = auto()
    ROLLBACK_FAILED = auto()


@dataclass
class ReloadError(Exception):
    """Exception raised during script reload."""
    error_type: ReloadErrorType
    message: str
    module_name: str = ""
    original_error: Optional[Exception] = None
    traceback_str: Optional[str] = None

    def __str__(self) -> str:
        result = f"{self.error_type.name}: {self.message}"
        if self.module_name:
            result += f" (module: {self.module_name})"
        return result


@dataclass
class ExecutionCheckpoint:
    """Checkpoint for resuming execution after reload."""
    function_name: str
    locals_snapshot: Dict[str, Any]
    line_number: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScriptState:
    """Captured state of a script/module for preservation across reloads."""

    module_name: str
    file_path: str
    schema_hash: str
    instances: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    globals_snapshot: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[ExecutionCheckpoint] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    version: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age(self) -> float:
        """Get age of this state in seconds."""
        return time.time() - self.timestamp

    def is_stale(self, max_age: float = 60.0) -> bool:
        """Check if this state is older than max_age seconds."""
        return self.age() > max_age

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "module_name": self.module_name,
            "file_path": self.file_path,
            "schema_hash": self.schema_hash,
            "instances": self.instances,
            "globals_snapshot": self.globals_snapshot,
            "checkpoints": [
                {
                    "function_name": cp.function_name,
                    "locals_snapshot": cp.locals_snapshot,
                    "line_number": cp.line_number,
                    "timestamp": cp.timestamp,
                }
                for cp in self.checkpoints
            ],
            "timestamp": self.timestamp,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScriptState":
        """Deserialize state from dictionary."""
        checkpoints = [
            ExecutionCheckpoint(
                function_name=cp["function_name"],
                locals_snapshot=cp["locals_snapshot"],
                line_number=cp["line_number"],
                timestamp=cp.get("timestamp", time.time()),
            )
            for cp in data.get("checkpoints", [])
        ]
        return cls(
            module_name=data["module_name"],
            file_path=data["file_path"],
            schema_hash=data["schema_hash"],
            instances=data.get("instances", {}),
            globals_snapshot=data.get("globals_snapshot", {}),
            checkpoints=checkpoints,
            timestamp=data.get("timestamp", time.time()),
            version=data.get("version", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ModuleBackup:
    """Backup of a module for rollback purposes."""
    module_name: str
    module_object: types.ModuleType
    spec: Optional[importlib.machinery.ModuleSpec]
    source_code: Optional[str]
    bytecode: Optional[bytes]
    file_path: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReloadResult:
    """Result of a reload operation."""
    success: bool
    module_name: str
    reload_time: float = 0.0
    instances_preserved: int = 0
    instances_restored: int = 0
    errors: List[ReloadError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rolled_back: bool = False
    state: ReloadState = ReloadState.IDLE

    def add_error(self, error: ReloadError) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> None:
        """Add a warning to the result."""
        self.warnings.append(warning)


class ModuleSwapper:
    """Handles safe module swapping with rollback capability."""

    def __init__(self):
        """Initialize the module swapper."""
        self._backups: Dict[str, ModuleBackup] = {}
        self._lock = threading.RLock()
        self._max_backups = 10
        self._schema_hasher = SchemaHasher()

    def backup_module(self, module_name: str) -> Optional[ModuleBackup]:
        """
        Create a backup of a module before reload.

        Args:
            module_name: Name of module to backup.

        Returns:
            ModuleBackup or None if module not loaded.
        """
        if module_name not in sys.modules:
            return None

        module = sys.modules[module_name]

        with self._lock:
            # Read source code if available
            source_code = None
            file_path = getattr(module, "__file__", "") or ""

            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source_code = f.read()
                except (IOError, OSError):
                    pass

            # Get module spec
            spec = getattr(module, "__spec__", None)

            # Create shallow copy of module
            backup_module = types.ModuleType(module_name)
            backup_module.__dict__.update(module.__dict__)

            backup = ModuleBackup(
                module_name=module_name,
                module_object=backup_module,
                spec=spec,
                source_code=source_code,
                bytecode=None,  # Could cache compiled bytecode
                file_path=file_path,
            )

            self._backups[module_name] = backup
            self._cleanup_old_backups()

            return backup

    def restore_backup(self, module_name: str) -> bool:
        """
        Restore a module from backup (rollback).

        Args:
            module_name: Name of module to restore.

        Returns:
            True if restore was successful.
        """
        with self._lock:
            if module_name not in self._backups:
                return False

            backup = self._backups[module_name]

            try:
                # Restore module to sys.modules
                sys.modules[module_name] = backup.module_object

                # Optionally restore source file (dangerous in production)
                # We don't do this by default, just restore the in-memory state

                return True
            except Exception:
                return False

    def discard_backup(self, module_name: str) -> bool:
        """
        Discard a module backup (after successful reload).

        Args:
            module_name: Name of module backup to discard.

        Returns:
            True if backup existed and was discarded.
        """
        with self._lock:
            if module_name in self._backups:
                del self._backups[module_name]
                return True
            return False

    def has_backup(self, module_name: str) -> bool:
        """Check if a backup exists for a module."""
        with self._lock:
            return module_name in self._backups

    def get_backup(self, module_name: str) -> Optional[ModuleBackup]:
        """Get the backup for a module."""
        with self._lock:
            return self._backups.get(module_name)

    def swap_module(
        self,
        module_name: str,
        new_source_path: Optional[str] = None,
    ) -> Tuple[bool, Optional[types.ModuleType], Optional[Exception]]:
        """
        Perform the actual module swap.

        Args:
            module_name: Name of module to swap.
            new_source_path: Optional new source file path.

        Returns:
            Tuple of (success, new_module, error).
        """
        try:
            # First backup the current module
            self.backup_module(module_name)

            # Remove from sys.modules to force reimport
            if module_name in sys.modules:
                old_module = sys.modules.pop(module_name)

                # Also remove any cached sub-modules
                to_remove = [
                    name for name in sys.modules
                    if name.startswith(module_name + ".")
                ]
                for name in to_remove:
                    sys.modules.pop(name, None)

            # Invalidate import caches
            importlib.invalidate_caches()

            # Reimport the module
            new_module = importlib.import_module(module_name)

            return True, new_module, None

        except SyntaxError as e:
            error = e
            # Restore from backup
            self.restore_backup(module_name)
            return False, None, error

        except ImportError as e:
            error = e
            self.restore_backup(module_name)
            return False, None, error

        except Exception as e:
            error = e
            self.restore_backup(module_name)
            return False, None, error

    def _cleanup_old_backups(self) -> None:
        """Remove old backups to prevent memory growth."""
        with self._lock:
            if len(self._backups) > self._max_backups:
                # Remove oldest backups
                sorted_backups = sorted(
                    self._backups.items(),
                    key=lambda x: x[1].timestamp,
                )
                to_remove = len(self._backups) - self._max_backups
                for module_name, _ in sorted_backups[:to_remove]:
                    del self._backups[module_name]

    def clear_all_backups(self) -> int:
        """Clear all backups."""
        with self._lock:
            count = len(self._backups)
            self._backups.clear()
            return count


class StateSerializer:
    """Handles serialization of script state using @serializable classes."""

    def __init__(self):
        """Initialize the state serializer."""
        self._ctx = SerializationContext(
            format=SerializationFormat.JSON,
            include_schema=True,
            include_defaults=False,
        )

    def serialize_instance(
        self,
        obj: Any,
        deep: bool = True,
    ) -> Dict[str, Any]:
        """
        Serialize an object instance to a dictionary.

        Args:
            obj: Object to serialize.
            deep: If True, serialize nested objects.

        Returns:
            Dictionary representation of the object.
        """
        # If object has serialize method (from @serializable), use it
        if hasattr(obj, "serialize"):
            try:
                return obj.serialize(self._ctx)
            except Exception:
                pass

        # Fallback to manual serialization
        result = {
            "__class__": type(obj).__name__,
            "__module__": type(obj).__module__,
        }

        # Get instance __dict__
        if hasattr(obj, "__dict__"):
            for key, value in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                try:
                    result[key] = self._serialize_value(value, deep)
                except Exception:
                    result[key] = str(value)

        return result

    def deserialize_instance(
        self,
        data: Dict[str, Any],
        target_class: Optional[Type] = None,
    ) -> Any:
        """
        Deserialize a dictionary back to an object instance.

        Args:
            data: Dictionary representation.
            target_class: Optional target class (uses data hints if not provided).

        Returns:
            Reconstructed object or dictionary if class not found.
        """
        if target_class is None:
            # Try to find class from data
            class_name = data.get("__class__")
            module_name = data.get("__module__")

            if class_name and module_name:
                try:
                    module = sys.modules.get(module_name)
                    if module:
                        target_class = getattr(module, class_name, None)
                except Exception:
                    pass

        # If target class has deserialize method, use it
        if target_class and hasattr(target_class, "deserialize"):
            try:
                return target_class.deserialize(data, self._ctx)
            except Exception:
                pass

        # Fallback: create instance and set attributes
        if target_class:
            try:
                obj = object.__new__(target_class)
                for key, value in data.items():
                    if not key.startswith("__"):
                        setattr(obj, key, self._deserialize_value(value))
                return obj
            except Exception:
                pass

        # Return raw data if deserialization fails
        return data

    def _serialize_value(self, value: Any, deep: bool) -> Any:
        """Serialize a single value."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v, deep) for v in value]

        if isinstance(value, dict):
            return {
                str(k): self._serialize_value(v, deep)
                for k, v in value.items()
            }

        if isinstance(value, set):
            return {"__set__": list(value)}

        if deep and hasattr(value, "serialize"):
            try:
                return value.serialize(self._ctx)
            except Exception:
                return str(value)

        if deep and hasattr(value, "__dict__"):
            return self.serialize_instance(value, deep=False)

        return str(value)

    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize a single value."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, list):
            return [self._deserialize_value(v) for v in value]

        if isinstance(value, dict):
            if "__set__" in value:
                return set(value["__set__"])
            if "__class__" in value:
                return self.deserialize_instance(value)
            return {k: self._deserialize_value(v) for k, v in value.items()}

        return value


# Type alias for reload callbacks
ReloadCallback = Callable[[str, ReloadResult], None]


class ScriptReloader:
    """
    Main script hot-reload manager for the TRINITY editor.

    Provides automatic detection and reload of Python scripts with
    state preservation across reloads. Integrates with the editor's
    serialization system for robust state management.

    Features:
    - Automatic module change detection via file watching
    - State serialization before reload using @serializable
    - Safe module swapping with rollback on failure
    - State restoration after successful reload
    - Execution checkpoint resumption
    - Configurable reload strategies
    """

    # Class-level constants
    DEFAULT_DEBOUNCE_TIME = 0.5  # seconds
    DEFAULT_MAX_STATES = 50
    DEFAULT_STATE_TTL = 300.0  # 5 minutes

    def __init__(
        self,
        strategy: ReloadStrategy = ReloadStrategy.DEBOUNCED,
        debounce_time: float = DEFAULT_DEBOUNCE_TIME,
        auto_start: bool = False,
    ):
        """
        Initialize the script reloader.

        Args:
            strategy: Reload strategy to use.
            debounce_time: Time to wait for changes to settle (debounced mode).
            auto_start: If True, start watching automatically.
        """
        self._strategy = strategy
        self._debounce_time = debounce_time

        # Core components
        self._module_watcher = ModuleWatcher(
            debounce_time=debounce_time,
        )
        self._module_swapper = ModuleSwapper()
        self._state_serializer = StateSerializer()
        self._state_preserver = StatePreserver(
            max_snapshots=self.DEFAULT_MAX_STATES,
            snapshot_ttl=self.DEFAULT_STATE_TTL,
        )
        self._dependency_tracker = DependencyTracker()
        self._schema_hasher = SchemaHasher()

        # State tracking
        self._states: Dict[str, ScriptState] = {}
        self._instance_refs: Dict[str, List[weakref.ref]] = {}
        self._pending_reloads: Dict[str, float] = {}  # module -> timestamp
        self._reload_history: List[ReloadResult] = []

        # Current state
        self._current_state = ReloadState.IDLE
        self._enabled = True
        self._lock = threading.RLock()

        # Callbacks
        self._on_reload_start: List[ReloadCallback] = []
        self._on_reload_complete: List[ReloadCallback] = []
        self._on_reload_error: List[ReloadCallback] = []
        self._on_state_change: List[Callable[[ReloadState], None]] = []

        # Watched modules
        self._watched_modules: Set[str] = set()
        self._watched_directories: Set[str] = set()

        # Execution checkpoints
        self._checkpoints: Dict[str, List[ExecutionCheckpoint]] = {}

        # Register watcher callback
        self._module_watcher.add_callback(self._on_module_change)

        # Statistics
        self._stats = {
            "total_reloads": 0,
            "successful_reloads": 0,
            "failed_reloads": 0,
            "rollbacks": 0,
            "states_preserved": 0,
            "states_restored": 0,
        }

        if auto_start:
            self.start()

    # Properties

    @property
    def enabled(self) -> bool:
        """Check if reload is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable reload."""
        self._enabled = value

    @property
    def state(self) -> ReloadState:
        """Get current reload state."""
        return self._current_state

    @property
    def strategy(self) -> ReloadStrategy:
        """Get reload strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: ReloadStrategy) -> None:
        """Set reload strategy."""
        self._strategy = value

    @property
    def stats(self) -> Dict[str, int]:
        """Get reload statistics."""
        with self._lock:
            return dict(self._stats)

    @property
    def is_running(self) -> bool:
        """Check if the reloader is running."""
        return self._module_watcher.is_running

    @property
    def watched_modules(self) -> List[str]:
        """Get list of watched module names."""
        with self._lock:
            return list(self._watched_modules)

    # Lifecycle methods

    def start(self) -> None:
        """Start watching for module changes."""
        self._module_watcher.start()

    def stop(self) -> None:
        """Stop watching for module changes."""
        self._module_watcher.stop()

    # Watch management

    def watch_module(self, module_name: str) -> bool:
        """
        Start watching a specific module for changes.

        Args:
            module_name: Name of the module to watch.

        Returns:
            True if module was found and is now being watched.
        """
        with self._lock:
            result = self._module_watcher.watch_module(module_name)
            if result:
                self._watched_modules.add(module_name)
            return result

    def watch_directory(
        self,
        path: str,
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> bool:
        """
        Watch a directory for Python module changes.

        Args:
            path: Directory path to watch.
            recursive: Watch subdirectories recursively.
            include_patterns: Optional patterns to include.
            exclude_patterns: Optional patterns to exclude.

        Returns:
            True if directory was added successfully.
        """
        result = self._module_watcher.watch_directory(
            path,
            recursive=recursive,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        if result:
            with self._lock:
                self._watched_directories.add(os.path.abspath(path))
        return result

    def unwatch_module(self, module_name: str) -> bool:
        """
        Stop watching a specific module.

        Args:
            module_name: Name of the module.

        Returns:
            True if module was being watched.
        """
        with self._lock:
            if module_name in self._watched_modules:
                self._watched_modules.discard(module_name)
                file_path = self._module_watcher.get_module_file(module_name)
                if file_path:
                    return self._module_watcher.unwatch(file_path)
            return False

    def unwatch_directory(self, path: str) -> bool:
        """
        Stop watching a directory.

        Args:
            path: Directory path.

        Returns:
            True if directory was being watched.
        """
        with self._lock:
            abs_path = os.path.abspath(path)
            if abs_path in self._watched_directories:
                self._watched_directories.discard(abs_path)
                return self._module_watcher.unwatch_directory(abs_path)
            return False

    # Instance tracking

    def register_instance(self, obj: Any, module_name: Optional[str] = None) -> bool:
        """
        Register an instance for state preservation.

        Args:
            obj: Object instance to track.
            module_name: Optional module name (inferred from type if not provided).

        Returns:
            True if instance was registered.
        """
        if module_name is None:
            module_name = type(obj).__module__

        with self._lock:
            if module_name not in self._instance_refs:
                self._instance_refs[module_name] = []

            # Add weak reference
            ref = weakref.ref(obj)
            self._instance_refs[module_name].append(ref)
            return True

    def unregister_instance(self, obj: Any) -> bool:
        """
        Unregister an instance from tracking.

        Args:
            obj: Object instance to unregister.

        Returns:
            True if instance was found and unregistered.
        """
        module_name = type(obj).__module__

        with self._lock:
            if module_name not in self._instance_refs:
                return False

            refs = self._instance_refs[module_name]
            obj_id = id(obj)

            for i, ref in enumerate(refs):
                if ref() is obj:
                    refs.pop(i)
                    return True

            return False

    def get_instances(self, module_name: str) -> List[Any]:
        """
        Get all tracked instances for a module.

        Args:
            module_name: Module name.

        Returns:
            List of live instances.
        """
        with self._lock:
            if module_name not in self._instance_refs:
                return []

            instances = []
            live_refs = []

            for ref in self._instance_refs[module_name]:
                obj = ref()
                if obj is not None:
                    instances.append(obj)
                    live_refs.append(ref)

            # Clean up dead references
            self._instance_refs[module_name] = live_refs

            return instances

    # Reload operations

    def reload_module(
        self,
        module_name: str,
        force: bool = False,
    ) -> ReloadResult:
        """
        Reload a Python module with state preservation.

        This is the main entry point for script hot-reload. It:
        1. Serializes state of all tracked instances
        2. Backs up the current module
        3. Swaps in the new module
        4. Restores state to new instances
        5. Rolls back on any failure

        Args:
            module_name: Name of the module to reload.
            force: If True, reload even if disabled.

        Returns:
            ReloadResult with details of the operation.
        """
        if not self._enabled and not force:
            return ReloadResult(
                success=False,
                module_name=module_name,
                errors=[ReloadError(
                    error_type=ReloadErrorType.DETECTION_FAILED,
                    message="Reload is disabled",
                    module_name=module_name,
                )],
            )

        start_time = time.time()
        result = ReloadResult(success=True, module_name=module_name)

        with self._lock:
            try:
                # Phase 1: Notify start
                self._set_state(ReloadState.DETECTING)
                self._notify_reload_start(module_name, result)

                # Phase 2: Check if module is loaded
                if module_name not in sys.modules:
                    result.add_error(ReloadError(
                        error_type=ReloadErrorType.DETECTION_FAILED,
                        message=f"Module {module_name} is not loaded",
                        module_name=module_name,
                    ))
                    self._set_state(ReloadState.FAILED)
                    return result

                # Phase 3: Preserve state
                self._set_state(ReloadState.PRESERVING_STATE)
                script_state = self._preserve_module_state(module_name)
                result.instances_preserved = len(script_state.instances)
                self._stats["states_preserved"] += result.instances_preserved

                # Phase 4: Backup and swap module
                self._set_state(ReloadState.UNLOADING)
                self._module_swapper.backup_module(module_name)

                self._set_state(ReloadState.LOADING)
                success, new_module, error = self._module_swapper.swap_module(module_name)

                if not success or error:
                    result.add_error(ReloadError(
                        error_type=ReloadErrorType.IMPORT_ERROR,
                        message=str(error),
                        module_name=module_name,
                        original_error=error,
                        traceback_str=traceback.format_exc() if error else None,
                    ))

                    # Rollback
                    self._set_state(ReloadState.ROLLING_BACK)
                    if self._module_swapper.restore_backup(module_name):
                        result.rolled_back = True
                        self._stats["rollbacks"] += 1

                    self._set_state(ReloadState.FAILED)
                    self._stats["failed_reloads"] += 1
                    return result

                # Phase 5: Restore state
                self._set_state(ReloadState.RESTORING_STATE)
                restored = self._restore_module_state(module_name, script_state)
                result.instances_restored = restored
                self._stats["states_restored"] += restored

                # Phase 6: Resume execution (if checkpoints exist)
                self._set_state(ReloadState.RESUMING)
                self._resume_execution(module_name, script_state)

                # Phase 7: Cleanup
                self._module_swapper.discard_backup(module_name)
                self._set_state(ReloadState.COMPLETED)

                self._stats["total_reloads"] += 1
                self._stats["successful_reloads"] += 1

            except Exception as e:
                result.add_error(ReloadError(
                    error_type=ReloadErrorType.RESUME_FAILED,
                    message=str(e),
                    module_name=module_name,
                    original_error=e,
                    traceback_str=traceback.format_exc(),
                ))

                # Attempt rollback
                self._set_state(ReloadState.ROLLING_BACK)
                if self._module_swapper.restore_backup(module_name):
                    result.rolled_back = True
                    self._stats["rollbacks"] += 1
                else:
                    result.add_error(ReloadError(
                        error_type=ReloadErrorType.ROLLBACK_FAILED,
                        message="Failed to restore module backup",
                        module_name=module_name,
                    ))

                self._set_state(ReloadState.FAILED)
                self._stats["failed_reloads"] += 1

            finally:
                result.reload_time = time.time() - start_time
                result.state = self._current_state
                self._reload_history.append(result)
                self._notify_reload_complete(module_name, result)

                # Return to idle
                self._set_state(ReloadState.IDLE)

        return result

    def reload_all_watched(self) -> Dict[str, ReloadResult]:
        """
        Reload all watched modules.

        Returns:
            Dictionary mapping module names to their reload results.
        """
        results = {}

        # Get sorted list by dependency order
        modules = list(self._watched_modules)

        for module_name in modules:
            results[module_name] = self.reload_module(module_name)

        return results

    # State management

    def _preserve_module_state(self, module_name: str) -> ScriptState:
        """
        Preserve state of all tracked instances in a module.

        Args:
            module_name: Module name.

        Returns:
            ScriptState containing all preserved state.
        """
        module = sys.modules.get(module_name)
        file_path = getattr(module, "__file__", "") if module else ""

        # Compute schema hash
        schema_hash = ""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                schema_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            except (IOError, OSError):
                pass

        state = ScriptState(
            module_name=module_name,
            file_path=file_path,
            schema_hash=schema_hash,
        )

        # Preserve instance states
        instances = self.get_instances(module_name)
        for obj in instances:
            obj_id = id(obj)
            try:
                state.instances[obj_id] = self._state_serializer.serialize_instance(obj)
            except Exception:
                state.instances[obj_id] = {"__error__": "Failed to serialize"}

        # Preserve module-level globals that are serializable
        if module:
            for name, value in vars(module).items():
                if name.startswith("_"):
                    continue
                if callable(value) or isinstance(value, type):
                    continue
                if hasattr(value, "serialize") or isinstance(value, (
                    bool, int, float, str, list, dict, tuple
                )):
                    try:
                        state.globals_snapshot[name] = self._state_serializer._serialize_value(
                            value, deep=True
                        )
                    except Exception:
                        pass

        # Preserve execution checkpoints
        if module_name in self._checkpoints:
            state.checkpoints = list(self._checkpoints[module_name])

        # Store state
        self._states[module_name] = state

        return state

    def _restore_module_state(
        self,
        module_name: str,
        state: ScriptState,
    ) -> int:
        """
        Restore state to instances in the reloaded module.

        Args:
            module_name: Module name.
            state: ScriptState to restore from.

        Returns:
            Number of instances restored.
        """
        restored_count = 0
        new_module = sys.modules.get(module_name)

        if not new_module:
            return 0

        # Get new instances that need state restoration
        new_instances = self.get_instances(module_name)

        # Try to match instances by class name and restore state
        for obj in new_instances:
            class_name = type(obj).__name__

            # Find matching state
            for old_id, old_state in state.instances.items():
                if old_state.get("__class__") == class_name:
                    try:
                        # Restore attributes
                        for key, value in old_state.items():
                            if key.startswith("__"):
                                continue
                            if hasattr(obj, key):
                                setattr(obj, key, self._state_serializer._deserialize_value(value))
                        restored_count += 1
                    except Exception:
                        pass
                    break

        # Restore module-level globals
        for name, value in state.globals_snapshot.items():
            if hasattr(new_module, name):
                try:
                    setattr(new_module, name, self._state_serializer._deserialize_value(value))
                except Exception:
                    pass

        return restored_count

    def _resume_execution(
        self,
        module_name: str,
        state: ScriptState,
    ) -> None:
        """
        Resume execution from checkpoints after reload.

        Args:
            module_name: Module name.
            state: ScriptState with checkpoints.
        """
        # Currently, full execution resumption requires more complex
        # machinery (e.g., generator-based coroutines or explicit
        # continuation passing). For now, we just record that
        # checkpoints were available.

        if state.checkpoints:
            # Store checkpoints for potential manual resumption
            self._checkpoints[module_name] = state.checkpoints

    # Checkpoint management

    def create_checkpoint(
        self,
        module_name: str,
        function_name: str,
        locals_dict: Dict[str, Any],
        line_number: int = 0,
    ) -> ExecutionCheckpoint:
        """
        Create an execution checkpoint for later resumption.

        Args:
            module_name: Module name.
            function_name: Name of the function.
            locals_dict: Local variables to preserve.
            line_number: Current line number.

        Returns:
            Created ExecutionCheckpoint.
        """
        # Filter locals to only serializable values
        filtered_locals = {}
        for key, value in locals_dict.items():
            if key.startswith("_"):
                continue
            try:
                filtered_locals[key] = self._state_serializer._serialize_value(
                    value, deep=True
                )
            except Exception:
                pass

        checkpoint = ExecutionCheckpoint(
            function_name=function_name,
            locals_snapshot=filtered_locals,
            line_number=line_number,
        )

        with self._lock:
            if module_name not in self._checkpoints:
                self._checkpoints[module_name] = []
            self._checkpoints[module_name].append(checkpoint)

        return checkpoint

    def get_checkpoints(self, module_name: str) -> List[ExecutionCheckpoint]:
        """Get all checkpoints for a module."""
        with self._lock:
            return list(self._checkpoints.get(module_name, []))

    def clear_checkpoints(self, module_name: str) -> int:
        """Clear checkpoints for a module."""
        with self._lock:
            if module_name in self._checkpoints:
                count = len(self._checkpoints[module_name])
                del self._checkpoints[module_name]
                return count
            return 0

    # Callback management

    def on_reload_start(self, callback: ReloadCallback) -> None:
        """Register callback for reload start event."""
        with self._lock:
            self._on_reload_start.append(callback)

    def on_reload_complete(self, callback: ReloadCallback) -> None:
        """Register callback for reload complete event."""
        with self._lock:
            self._on_reload_complete.append(callback)

    def on_reload_error(self, callback: ReloadCallback) -> None:
        """Register callback for reload error event."""
        with self._lock:
            self._on_reload_error.append(callback)

    def on_state_change(self, callback: Callable[[ReloadState], None]) -> None:
        """Register callback for state changes."""
        with self._lock:
            self._on_state_change.append(callback)

    def remove_callback(self, callback: Callable) -> bool:
        """Remove a callback from all lists."""
        with self._lock:
            removed = False
            for callback_list in [
                self._on_reload_start,
                self._on_reload_complete,
                self._on_reload_error,
                self._on_state_change,
            ]:
                if callback in callback_list:
                    callback_list.remove(callback)
                    removed = True
            return removed

    # Internal methods

    def _on_module_change(self, event: ModuleChangeEvent) -> None:
        """Handle module change events from the watcher."""
        if not self._enabled:
            return

        module_name = event.module_name

        # Skip if not in our watched set
        if module_name not in self._watched_modules:
            # Check if it matches watched directories
            matched = False
            for dir_path in self._watched_directories:
                if event.file_path.startswith(dir_path):
                    matched = True
                    self._watched_modules.add(module_name)
                    break
            if not matched:
                return

        # Handle based on strategy
        if self._strategy == ReloadStrategy.IMMEDIATE:
            self.reload_module(module_name)

        elif self._strategy == ReloadStrategy.DEBOUNCED:
            # Schedule reload after debounce time
            with self._lock:
                self._pending_reloads[module_name] = time.time() + self._debounce_time
            # Start timer thread
            threading.Thread(
                target=self._process_pending_reload,
                args=(module_name,),
                daemon=True,
            ).start()

        elif self._strategy == ReloadStrategy.SCHEDULED:
            # Just mark as pending, actual reload happens at next safe point
            with self._lock:
                self._pending_reloads[module_name] = time.time()

        # MANUAL strategy does nothing on change

    def _process_pending_reload(self, module_name: str) -> None:
        """Process a pending debounced reload."""
        time.sleep(self._debounce_time)

        with self._lock:
            if module_name not in self._pending_reloads:
                return

            # Check if we should still reload (no newer changes)
            scheduled_time = self._pending_reloads.get(module_name, 0)
            if time.time() < scheduled_time:
                return

            del self._pending_reloads[module_name]

        self.reload_module(module_name)

    def process_scheduled_reloads(self) -> Dict[str, ReloadResult]:
        """
        Process all scheduled reloads (for SCHEDULED strategy).

        Call this at a safe point (e.g., between frames) to reload
        all pending modules.

        Returns:
            Dictionary of module names to reload results.
        """
        results = {}

        with self._lock:
            pending = list(self._pending_reloads.keys())
            self._pending_reloads.clear()

        for module_name in pending:
            results[module_name] = self.reload_module(module_name)

        return results

    def _set_state(self, new_state: ReloadState) -> None:
        """Set the current reload state and notify listeners."""
        self._current_state = new_state

        # Notify callbacks
        for callback in self._on_state_change:
            try:
                callback(new_state)
            except Exception:
                pass

    def _notify_reload_start(self, module_name: str, result: ReloadResult) -> None:
        """Notify callbacks of reload start."""
        for callback in self._on_reload_start:
            try:
                callback(module_name, result)
            except Exception:
                pass

    def _notify_reload_complete(self, module_name: str, result: ReloadResult) -> None:
        """Notify callbacks of reload completion."""
        callbacks = self._on_reload_complete if result.success else self._on_reload_error

        for callback in callbacks:
            try:
                callback(module_name, result)
            except Exception:
                pass

        # Also notify complete callbacks for error cases
        if not result.success:
            for callback in self._on_reload_complete:
                try:
                    callback(module_name, result)
                except Exception:
                    pass

    # Query methods

    def get_state(self, module_name: str) -> Optional[ScriptState]:
        """Get the preserved state for a module."""
        with self._lock:
            return self._states.get(module_name)

    def get_reload_history(
        self,
        module_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[ReloadResult]:
        """
        Get reload history.

        Args:
            module_name: Optional filter by module name.
            limit: Maximum number of results.

        Returns:
            List of ReloadResults, most recent first.
        """
        with self._lock:
            history = self._reload_history[-limit:]

            if module_name:
                history = [r for r in history if r.module_name == module_name]

            return list(reversed(history))

    def has_pending_reloads(self) -> bool:
        """Check if there are pending reloads."""
        with self._lock:
            return len(self._pending_reloads) > 0

    def get_pending_reloads(self) -> List[str]:
        """Get list of pending reload module names."""
        with self._lock:
            return list(self._pending_reloads.keys())

    # Cleanup

    def clear_history(self) -> int:
        """Clear reload history."""
        with self._lock:
            count = len(self._reload_history)
            self._reload_history.clear()
            return count

    def clear_states(self) -> int:
        """Clear all preserved states."""
        with self._lock:
            count = len(self._states)
            self._states.clear()
            return count

    def clear_all(self) -> None:
        """Clear all state and stop watching."""
        self.stop()

        with self._lock:
            self._states.clear()
            self._instance_refs.clear()
            self._pending_reloads.clear()
            self._reload_history.clear()
            self._checkpoints.clear()
            self._watched_modules.clear()
            self._watched_directories.clear()
            self._module_swapper.clear_all_backups()
            self._module_watcher.clear()

        self._stats = {
            "total_reloads": 0,
            "successful_reloads": 0,
            "failed_reloads": 0,
            "rollbacks": 0,
            "states_preserved": 0,
            "states_restored": 0,
        }


# Convenience function for context-manager based checkpointing
@contextmanager
def reloadable_section(
    reloader: ScriptReloader,
    module_name: str,
    function_name: str,
) -> Iterator[Dict[str, Any]]:
    """
    Context manager for creating reloadable code sections.

    Usage:
        with reloadable_section(reloader, __name__, "my_function") as checkpoint:
            # Your code here
            checkpoint["progress"] = 50
            # If reload happens, checkpoint data is preserved

    Args:
        reloader: ScriptReloader instance.
        module_name: Module name.
        function_name: Function/section name.

    Yields:
        Dictionary to store checkpoint data.
    """
    checkpoint_data: Dict[str, Any] = {}

    try:
        yield checkpoint_data
    finally:
        if checkpoint_data:
            reloader.create_checkpoint(
                module_name=module_name,
                function_name=function_name,
                locals_dict=checkpoint_data,
            )


# Singleton instance
_script_reloader: Optional[ScriptReloader] = None


def get_script_reloader() -> ScriptReloader:
    """Get the global ScriptReloader instance."""
    global _script_reloader
    if _script_reloader is None:
        _script_reloader = ScriptReloader()
    return _script_reloader


__all__ = [
    # Enums
    "ReloadState",
    "ReloadStrategy",
    "ReloadErrorType",
    # Exceptions
    "ReloadError",
    # Data classes
    "ExecutionCheckpoint",
    "ScriptState",
    "ModuleBackup",
    "ReloadResult",
    # Core classes
    "ModuleSwapper",
    "StateSerializer",
    "ScriptReloader",
    # Utilities
    "reloadable_section",
    "get_script_reloader",
]
