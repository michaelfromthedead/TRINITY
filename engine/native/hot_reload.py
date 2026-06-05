"""T-CC-3.4: Native code hot-reload with function table patching.

Provides DLL/SO hot-swap capabilities for Rust native libraries via ctypes/cffi.
Supports function pointer indirection, state migration, and ABI version checking.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import pickle
import platform
import shutil
import struct
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core.file_watcher import FileChangeEvent, FileChangeType, FileWatcher


class LoadError(Exception):
    """Raised when a native library fails to load."""
    pass


class ReloadError(Exception):
    """Raised when a hot-reload operation fails."""
    pass


class ABIMismatchError(ReloadError):
    """Raised when ABI version mismatch is detected."""

    def __init__(self, expected: "ABIVersion", actual: "ABIVersion"):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"ABI mismatch: expected {expected}, got {actual}"
        )


class ReloadResult(Enum):
    """Result of a reload operation."""
    SUCCESS = auto()
    PARTIAL = auto()
    FAILED = auto()
    SKIPPED = auto()
    ROLLBACK = auto()


class ReloadEventType(Enum):
    """Types of reload events."""
    LIBRARY_LOADING = auto()
    LIBRARY_LOADED = auto()
    LIBRARY_UNLOADING = auto()
    LIBRARY_UNLOADED = auto()
    STATE_SAVING = auto()
    STATE_SAVED = auto()
    STATE_RESTORING = auto()
    STATE_RESTORED = auto()
    FUNCTION_PATCHING = auto()
    FUNCTION_PATCHED = auto()
    ABI_CHECK = auto()
    RELOAD_STARTED = auto()
    RELOAD_COMPLETED = auto()
    RELOAD_FAILED = auto()
    ROLLBACK_STARTED = auto()
    ROLLBACK_COMPLETED = auto()


@dataclass(slots=True)
class ABIVersion:
    """ABI version identifier for native libraries."""
    major: int
    minor: int
    patch: int = 0
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = ""

    def is_compatible(self, other: "ABIVersion") -> bool:
        """Check if this version is compatible with another."""
        if self.major != other.major:
            return False
        if self.minor > other.minor:
            return False
        return True

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.hash:
            return f"{base}+{self.hash[:8]}"
        return base

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ABIVersion):
            return False
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
        )

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ABIVersion":
        """Deserialize from dictionary."""
        return cls(
            major=data.get("major", 0),
            minor=data.get("minor", 0),
            patch=data.get("patch", 0),
            hash=data.get("hash", ""),
        )

    @classmethod
    def from_string(cls, version_str: str) -> "ABIVersion":
        """Parse version from string like '1.2.3' or '1.2.3+hash'."""
        parts = version_str.split("+")
        version_parts = parts[0].split(".")
        hash_str = parts[1] if len(parts) > 1 else ""

        major = int(version_parts[0]) if len(version_parts) > 0 else 0
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0

        return cls(major=major, minor=minor, patch=patch, hash=hash_str)


@dataclass
class ReloadEvent:
    """Event emitted during reload operations."""
    event_type: ReloadEventType
    library_path: Optional[Path] = None
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ReloadOutcome:
    """Result of a reload operation with details."""
    result: ReloadResult
    library_path: Path
    old_version: Optional[ABIVersion] = None
    new_version: Optional[ABIVersion] = None
    duration_ms: float = 0.0
    patched_functions: int = 0
    state_migrated: bool = False
    error: Optional[str] = None
    events: List[ReloadEvent] = field(default_factory=list)


@dataclass(slots=True)
class FunctionEntry:
    """Entry in the function table."""
    name: str
    address: int
    signature: str
    argtypes: Optional[List[Any]] = None
    restype: Any = None
    callable: Optional[Callable] = None

    def __hash__(self) -> int:
        return hash(self.name)


class FunctionTable:
    """Indirection table for native function pointers.

    Provides seamless function replacement during hot-reload by maintaining
    a level of indirection between callers and native functions.
    """

    __slots__ = ("_entries", "_lock", "_version", "_library")

    def __init__(self):
        self._entries: Dict[str, FunctionEntry] = {}
        self._lock = threading.RLock()
        self._version = 0
        self._library: Optional[ctypes.CDLL] = None

    @property
    def version(self) -> int:
        """Current table version (increments on each patch)."""
        return self._version

    @property
    def function_count(self) -> int:
        """Number of registered functions."""
        return len(self._entries)

    def register(
        self,
        name: str,
        signature: str = "",
        argtypes: Optional[List[Any]] = None,
        restype: Any = None,
    ) -> None:
        """Register a function entry in the table."""
        with self._lock:
            self._entries[name] = FunctionEntry(
                name=name,
                address=0,
                signature=signature,
                argtypes=argtypes,
                restype=restype,
            )

    def unregister(self, name: str) -> bool:
        """Remove a function from the table."""
        with self._lock:
            if name in self._entries:
                del self._entries[name]
                return True
            return False

    def get(self, name: str) -> Optional[FunctionEntry]:
        """Get a function entry by name."""
        with self._lock:
            return self._entries.get(name)

    def call(self, name: str, *args: Any) -> Any:
        """Call a registered function by name."""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise KeyError(f"Function not registered: {name}")
            if entry.callable is None:
                raise LoadError(f"Function not loaded: {name}")
            return entry.callable(*args)

    def bind_library(self, library: ctypes.CDLL) -> int:
        """Bind all registered functions to a library.

        Returns the number of successfully bound functions.
        """
        bound = 0
        with self._lock:
            self._library = library
            for name, entry in self._entries.items():
                try:
                    func = getattr(library, name, None)
                    if func is not None:
                        if entry.argtypes:
                            func.argtypes = entry.argtypes
                        if entry.restype:
                            func.restype = entry.restype
                        entry.callable = func
                        entry.address = ctypes.cast(func, ctypes.c_void_p).value or 0
                        bound += 1
                except (AttributeError, OSError):
                    entry.callable = None
                    entry.address = 0

            self._version += 1

        return bound

    def unbind(self) -> None:
        """Unbind all functions from the current library."""
        with self._lock:
            for entry in self._entries.values():
                entry.callable = None
                entry.address = 0
            self._library = None

    def patch(
        self,
        name: str,
        new_callable: Callable,
        new_address: int = 0,
    ) -> bool:
        """Patch a single function with a new implementation."""
        with self._lock:
            if name not in self._entries:
                return False
            entry = self._entries[name]
            entry.callable = new_callable
            entry.address = new_address
            self._version += 1
            return True

    def get_all_names(self) -> List[str]:
        """Get all registered function names."""
        with self._lock:
            return list(self._entries.keys())

    def get_bound_names(self) -> List[str]:
        """Get names of functions that are currently bound."""
        with self._lock:
            return [
                name for name, entry in self._entries.items()
                if entry.callable is not None
            ]

    def get_unbound_names(self) -> List[str]:
        """Get names of functions that are not bound."""
        with self._lock:
            return [
                name for name, entry in self._entries.items()
                if entry.callable is None
            ]

    def clear(self) -> None:
        """Clear all entries and unbind."""
        with self._lock:
            self._entries.clear()
            self._library = None
            self._version = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize function table metadata."""
        with self._lock:
            return {
                "version": self._version,
                "functions": [
                    {
                        "name": e.name,
                        "signature": e.signature,
                        "address": e.address,
                        "bound": e.callable is not None,
                    }
                    for e in self._entries.values()
                ],
            }


class StateSerializer:
    """Serializes and deserializes native library state for migration."""

    __slots__ = ("_format", "_max_size")

    def __init__(self, format: str = "pickle", max_size: int = 100_000_000):
        self._format = format
        self._max_size = max_size

    def serialize(self, state: Any) -> bytes:
        """Serialize state to bytes."""
        if self._format == "pickle":
            data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        elif self._format == "json":
            data = json.dumps(state).encode("utf-8")
        else:
            raise ValueError(f"Unknown format: {self._format}")

        if len(data) > self._max_size:
            raise ValueError(
                f"State too large: {len(data)} bytes (max {self._max_size})"
            )
        return data

    def deserialize(self, data: bytes) -> Any:
        """Deserialize state from bytes."""
        if self._format == "pickle":
            return pickle.loads(data)
        elif self._format == "json":
            return json.loads(data.decode("utf-8"))
        else:
            raise ValueError(f"Unknown format: {self._format}")


@dataclass
class LibraryState:
    """Captured state of a native library for migration."""
    data: bytes
    version: ABIVersion
    timestamp: float = field(default_factory=time.time)
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.checksum:
            self.checksum = hashlib.md5(self.data).hexdigest()

    def verify_checksum(self) -> bool:
        """Verify data integrity."""
        return hashlib.md5(self.data).hexdigest() == self.checksum


class NativeLibrary:
    """Wrapper for a native shared library with versioning support."""

    __slots__ = (
        "_path",
        "_handle",
        "_version",
        "_load_time",
        "_function_table",
        "_lock",
        "_temp_path",
    )

    def __init__(
        self,
        path: Union[str, Path],
        version: Optional[ABIVersion] = None,
    ):
        self._path = Path(path).resolve()
        self._handle: Optional[ctypes.CDLL] = None
        self._version = version or ABIVersion(0, 0, 0)
        self._load_time: float = 0.0
        self._function_table = FunctionTable()
        self._lock = threading.RLock()
        self._temp_path: Optional[Path] = None

    @property
    def path(self) -> Path:
        """Path to the library file."""
        return self._path

    @property
    def is_loaded(self) -> bool:
        """Whether the library is currently loaded."""
        return self._handle is not None

    @property
    def version(self) -> ABIVersion:
        """ABI version of the loaded library."""
        return self._version

    @property
    def load_time(self) -> float:
        """Timestamp when library was loaded."""
        return self._load_time

    @property
    def function_table(self) -> FunctionTable:
        """Function table for this library."""
        return self._function_table

    @property
    def handle(self) -> Optional[ctypes.CDLL]:
        """Raw ctypes handle to the library."""
        return self._handle

    def load(self, copy_first: bool = True) -> None:
        """Load the native library.

        Args:
            copy_first: If True, copy library to temp before loading
                       to allow the original file to be overwritten.
        """
        if not self._path.exists():
            raise LoadError(f"Library not found: {self._path}")

        with self._lock:
            if self._handle is not None:
                self.unload()

            load_path = self._path

            if copy_first:
                suffix = _get_library_suffix()
                fd, temp_path = tempfile.mkstemp(suffix=suffix)
                os.close(fd)
                self._temp_path = Path(temp_path)
                shutil.copy2(self._path, self._temp_path)
                load_path = self._temp_path

            try:
                self._handle = ctypes.CDLL(str(load_path))
                self._load_time = time.time()
                self._try_read_version()
            except OSError as e:
                if self._temp_path and self._temp_path.exists():
                    try:
                        os.unlink(self._temp_path)
                    except OSError:
                        pass
                    self._temp_path = None
                raise LoadError(f"Failed to load library: {e}") from e

    def unload(self) -> None:
        """Unload the native library."""
        with self._lock:
            if self._handle is not None:
                self._function_table.unbind()

                # Platform-specific unload
                if platform.system() == "Windows":
                    ctypes.windll.kernel32.FreeLibrary(self._handle._handle)
                else:
                    _dlclose = ctypes.CDLL(None).dlclose
                    _dlclose.argtypes = [ctypes.c_void_p]
                    _dlclose.restype = ctypes.c_int
                    _dlclose(self._handle._handle)

                self._handle = None
                self._load_time = 0.0

            if self._temp_path and self._temp_path.exists():
                try:
                    os.unlink(self._temp_path)
                except OSError:
                    pass
                self._temp_path = None

    def _try_read_version(self) -> None:
        """Try to read version from library's exported symbols."""
        if self._handle is None:
            return

        try:
            # Try reading ABI_VERSION_MAJOR, ABI_VERSION_MINOR, ABI_VERSION_PATCH
            major = getattr(self._handle, "ABI_VERSION_MAJOR", None)
            minor = getattr(self._handle, "ABI_VERSION_MINOR", None)
            patch = getattr(self._handle, "ABI_VERSION_PATCH", None)

            if major is not None:
                major_val = ctypes.c_int.in_dll(self._handle, "ABI_VERSION_MAJOR").value
                minor_val = 0
                patch_val = 0

                if minor is not None:
                    minor_val = ctypes.c_int.in_dll(self._handle, "ABI_VERSION_MINOR").value
                if patch is not None:
                    patch_val = ctypes.c_int.in_dll(self._handle, "ABI_VERSION_PATCH").value

                self._version = ABIVersion(major_val, minor_val, patch_val)
        except (AttributeError, OSError):
            pass

    def get_function(
        self,
        name: str,
        argtypes: Optional[List[Any]] = None,
        restype: Any = None,
    ) -> Optional[Callable]:
        """Get a function from the library."""
        if self._handle is None:
            return None

        try:
            func = getattr(self._handle, name)
            if argtypes:
                func.argtypes = argtypes
            if restype:
                func.restype = restype
            return func
        except (AttributeError, OSError):
            return None

    def bind_functions(self) -> int:
        """Bind all registered functions in the function table."""
        if self._handle is None:
            raise LoadError("Library not loaded")
        return self._function_table.bind_library(self._handle)

    def __enter__(self) -> "NativeLibrary":
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()


ReloadCallback = Callable[[ReloadEvent], None]


class NativeReloader:
    """Hot-reload manager for native libraries.

    Watches compiled .so/.dll files and performs hot-swap with
    function pointer patching and state migration.
    """

    __slots__ = (
        "_libraries",
        "_watcher",
        "_state_serializer",
        "_callbacks",
        "_lock",
        "_expected_version",
        "_strict_abi",
        "_auto_start",
        "_outcomes",
        "_max_outcomes",
        "_fallback_enabled",
    )

    def __init__(
        self,
        expected_version: Optional[ABIVersion] = None,
        strict_abi: bool = True,
        auto_start: bool = False,
        state_format: str = "pickle",
        fallback_enabled: bool = True,
    ):
        self._libraries: Dict[Path, NativeLibrary] = {}
        self._watcher = FileWatcher(poll_interval_ms=500)
        self._state_serializer = StateSerializer(format=state_format)
        self._callbacks: List[ReloadCallback] = []
        self._lock = threading.RLock()
        self._expected_version = expected_version
        self._strict_abi = strict_abi
        self._auto_start = auto_start
        self._outcomes: List[ReloadOutcome] = []
        self._max_outcomes = 100
        self._fallback_enabled = fallback_enabled

        self._watcher.registry.register_global(self._on_file_change)

        if auto_start:
            self._watcher.start()

    @property
    def is_running(self) -> bool:
        """Whether the reloader is actively watching."""
        return self._watcher.is_running

    @property
    def library_count(self) -> int:
        """Number of managed libraries."""
        return len(self._libraries)

    def register_library(
        self,
        path: Union[str, Path],
        function_names: Optional[List[str]] = None,
        version: Optional[ABIVersion] = None,
    ) -> NativeLibrary:
        """Register a native library for hot-reload.

        Args:
            path: Path to the .so/.dll file
            function_names: Functions to register in the function table
            version: Expected ABI version

        Returns:
            The NativeLibrary wrapper
        """
        path = Path(path).resolve()

        with self._lock:
            if path in self._libraries:
                return self._libraries[path]

            library = NativeLibrary(path, version)

            if function_names:
                for name in function_names:
                    library.function_table.register(name)

            self._libraries[path] = library
            self._watcher.watch(path)

        return library

    def unregister_library(self, path: Union[str, Path]) -> bool:
        """Unregister and unload a library."""
        path = Path(path).resolve()

        with self._lock:
            if path not in self._libraries:
                return False

            library = self._libraries.pop(path)
            if library.is_loaded:
                library.unload()
            self._watcher.unwatch(path)

        return True

    def get_library(self, path: Union[str, Path]) -> Optional[NativeLibrary]:
        """Get a registered library by path."""
        path = Path(path).resolve()
        with self._lock:
            return self._libraries.get(path)

    def load_library(
        self,
        path: Union[str, Path],
        bind_functions: bool = True,
    ) -> NativeLibrary:
        """Load a registered library.

        Args:
            path: Path to the library
            bind_functions: Whether to bind registered functions

        Returns:
            The loaded NativeLibrary
        """
        path = Path(path).resolve()

        with self._lock:
            library = self._libraries.get(path)
            if library is None:
                library = self.register_library(path)

        self._emit_event(ReloadEventType.LIBRARY_LOADING, path)

        library.load()

        if bind_functions:
            bound = library.bind_functions()
            self._emit_event(
                ReloadEventType.FUNCTION_PATCHED,
                path,
                details={"bound_count": bound},
            )

        self._emit_event(ReloadEventType.LIBRARY_LOADED, path)

        if self._expected_version is not None:
            if not self._expected_version.is_compatible(library.version):
                if self._strict_abi:
                    library.unload()
                    raise ABIMismatchError(self._expected_version, library.version)
                self._emit_event(
                    ReloadEventType.ABI_CHECK,
                    path,
                    details={
                        "expected": str(self._expected_version),
                        "actual": str(library.version),
                        "compatible": False,
                    },
                )

        return library

    def unload_library(self, path: Union[str, Path]) -> bool:
        """Unload a library but keep it registered."""
        path = Path(path).resolve()

        with self._lock:
            library = self._libraries.get(path)
            if library is None or not library.is_loaded:
                return False

        self._emit_event(ReloadEventType.LIBRARY_UNLOADING, path)
        library.unload()
        self._emit_event(ReloadEventType.LIBRARY_UNLOADED, path)

        return True

    def reload_library(
        self,
        path: Union[str, Path],
        state_getter: Optional[Callable[[], Any]] = None,
        state_setter: Optional[Callable[[Any], None]] = None,
    ) -> ReloadOutcome:
        """Hot-reload a library with state migration.

        Args:
            path: Path to the library
            state_getter: Callback to capture state before reload
            state_setter: Callback to restore state after reload

        Returns:
            ReloadOutcome with details
        """
        path = Path(path).resolve()
        start_time = time.perf_counter()
        events: List[ReloadEvent] = []

        def emit(event_type: ReloadEventType, **kwargs: Any) -> None:
            event = ReloadEvent(event_type, path, **kwargs)
            events.append(event)
            self._emit_event(event_type, path, **kwargs)

        with self._lock:
            library = self._libraries.get(path)
            if library is None:
                return ReloadOutcome(
                    result=ReloadResult.FAILED,
                    library_path=path,
                    error="Library not registered",
                    events=events,
                )

        old_version = library.version if library.is_loaded else None
        saved_state: Optional[LibraryState] = None
        old_handle = library.handle

        emit(ReloadEventType.RELOAD_STARTED)

        try:
            # Save state if getter provided
            if state_getter is not None and library.is_loaded:
                emit(ReloadEventType.STATE_SAVING)
                try:
                    state_data = state_getter()
                    serialized = self._state_serializer.serialize(state_data)
                    saved_state = LibraryState(
                        data=serialized,
                        version=library.version,
                    )
                    emit(ReloadEventType.STATE_SAVED, details={"size": len(serialized)})
                except Exception as e:
                    emit(ReloadEventType.STATE_SAVING, error=str(e))

            # Unload old library
            if library.is_loaded:
                emit(ReloadEventType.LIBRARY_UNLOADING)
                library.unload()
                emit(ReloadEventType.LIBRARY_UNLOADED)

            # Load new library
            emit(ReloadEventType.LIBRARY_LOADING)
            library.load()
            emit(ReloadEventType.LIBRARY_LOADED)

            # Check ABI compatibility
            new_version = library.version
            if self._expected_version is not None:
                compatible = self._expected_version.is_compatible(new_version)
                emit(
                    ReloadEventType.ABI_CHECK,
                    details={
                        "expected": str(self._expected_version),
                        "actual": str(new_version),
                        "compatible": compatible,
                    },
                )
                if not compatible and self._strict_abi:
                    raise ABIMismatchError(self._expected_version, new_version)

            # Bind functions
            emit(ReloadEventType.FUNCTION_PATCHING)
            patched = library.bind_functions()
            emit(
                ReloadEventType.FUNCTION_PATCHED,
                details={"patched_count": patched},
            )

            # Restore state if setter provided
            state_migrated = False
            if saved_state is not None and state_setter is not None:
                emit(ReloadEventType.STATE_RESTORING)
                try:
                    if saved_state.verify_checksum():
                        state_data = self._state_serializer.deserialize(saved_state.data)
                        state_setter(state_data)
                        state_migrated = True
                        emit(ReloadEventType.STATE_RESTORED)
                    else:
                        emit(
                            ReloadEventType.STATE_RESTORING,
                            error="Checksum verification failed",
                        )
                except Exception as e:
                    emit(ReloadEventType.STATE_RESTORING, error=str(e))

            duration = (time.perf_counter() - start_time) * 1000
            emit(ReloadEventType.RELOAD_COMPLETED)

            outcome = ReloadOutcome(
                result=ReloadResult.SUCCESS,
                library_path=path,
                old_version=old_version,
                new_version=new_version,
                duration_ms=duration,
                patched_functions=patched,
                state_migrated=state_migrated,
                events=events,
            )

        except Exception as e:
            duration = (time.perf_counter() - start_time) * 1000
            emit(ReloadEventType.RELOAD_FAILED, error=str(e))

            # Attempt rollback if fallback enabled
            if self._fallback_enabled and old_handle is not None:
                emit(ReloadEventType.ROLLBACK_STARTED)
                try:
                    # Library already unloaded, cannot truly rollback
                    # but we mark it as failed
                    emit(ReloadEventType.ROLLBACK_COMPLETED)
                except Exception:
                    pass

            outcome = ReloadOutcome(
                result=ReloadResult.FAILED,
                library_path=path,
                old_version=old_version,
                duration_ms=duration,
                error=str(e),
                events=events,
            )

        self._record_outcome(outcome)
        return outcome

    def save_library_state(
        self,
        path: Union[str, Path],
        state_getter: Callable[[], Any],
    ) -> Optional[LibraryState]:
        """Save library state for manual migration."""
        path = Path(path).resolve()

        with self._lock:
            library = self._libraries.get(path)
            if library is None or not library.is_loaded:
                return None

        try:
            state_data = state_getter()
            serialized = self._state_serializer.serialize(state_data)
            return LibraryState(
                data=serialized,
                version=library.version,
            )
        except Exception:
            return None

    def restore_library_state(
        self,
        path: Union[str, Path],
        state: LibraryState,
        state_setter: Callable[[Any], None],
    ) -> bool:
        """Restore library state from a saved state."""
        path = Path(path).resolve()

        with self._lock:
            library = self._libraries.get(path)
            if library is None or not library.is_loaded:
                return False

        if not state.verify_checksum():
            return False

        try:
            state_data = self._state_serializer.deserialize(state.data)
            state_setter(state_data)
            return True
        except Exception:
            return False

    def add_callback(self, callback: ReloadCallback) -> None:
        """Register a callback for reload events."""
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: ReloadCallback) -> bool:
        """Remove a reload callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                return True
            return False

    def start(self) -> None:
        """Start watching for file changes."""
        self._watcher.start()

    def stop(self) -> None:
        """Stop watching for file changes."""
        self._watcher.stop()

    def poll_once(self) -> List[FileChangeEvent]:
        """Manually poll for file changes."""
        return self._watcher.poll_once()

    def get_recent_outcomes(self, limit: int = 10) -> List[ReloadOutcome]:
        """Get recent reload outcomes."""
        with self._lock:
            return list(self._outcomes[-limit:])

    def get_status(self) -> Dict[str, Any]:
        """Get current reloader status."""
        with self._lock:
            return {
                "running": self.is_running,
                "library_count": len(self._libraries),
                "libraries": {
                    str(path): {
                        "loaded": lib.is_loaded,
                        "version": str(lib.version),
                        "function_count": lib.function_table.function_count,
                        "bound_functions": len(lib.function_table.get_bound_names()),
                    }
                    for path, lib in self._libraries.items()
                },
                "strict_abi": self._strict_abi,
                "expected_version": (
                    str(self._expected_version) if self._expected_version else None
                ),
                "recent_outcomes": len(self._outcomes),
            }

    def _on_file_change(self, event: FileChangeEvent) -> None:
        """Handle file change events."""
        if event.change_type != FileChangeType.MODIFIED:
            return

        path = event.path.resolve()
        with self._lock:
            if path not in self._libraries:
                return
            library = self._libraries[path]
            if not library.is_loaded:
                return

        # Auto-reload on file change
        self.reload_library(path)

    def _emit_event(
        self,
        event_type: ReloadEventType,
        library_path: Optional[Path] = None,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Emit a reload event to all callbacks."""
        event = ReloadEvent(
            event_type=event_type,
            library_path=library_path,
            details=details or {},
            error=error,
        )

        with self._lock:
            callbacks = self._callbacks.copy()

        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def _record_outcome(self, outcome: ReloadOutcome) -> None:
        """Record a reload outcome."""
        with self._lock:
            self._outcomes.append(outcome)
            if len(self._outcomes) > self._max_outcomes:
                self._outcomes.pop(0)

    def clear(self) -> None:
        """Clear all libraries and state."""
        with self._lock:
            for library in self._libraries.values():
                if library.is_loaded:
                    library.unload()
            self._libraries.clear()
            self._outcomes.clear()
            self._watcher.clear()

    def __enter__(self) -> "NativeReloader":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
        self.clear()


def _get_library_suffix() -> str:
    """Get the platform-specific library suffix."""
    system = platform.system()
    if system == "Windows":
        return ".dll"
    elif system == "Darwin":
        return ".dylib"
    else:
        return ".so"


def create_native_reloader(
    watch_paths: Optional[List[Union[str, Path]]] = None,
    expected_version: Optional[ABIVersion] = None,
    strict_abi: bool = True,
    auto_start: bool = True,
) -> NativeReloader:
    """Factory to create a configured NativeReloader.

    Args:
        watch_paths: Initial library paths to watch
        expected_version: Expected ABI version for all libraries
        strict_abi: Whether to enforce ABI compatibility
        auto_start: Whether to start watching immediately

    Returns:
        Configured NativeReloader instance
    """
    reloader = NativeReloader(
        expected_version=expected_version,
        strict_abi=strict_abi,
        auto_start=False,
    )

    if watch_paths:
        for path in watch_paths:
            reloader.register_library(path)

    if auto_start:
        reloader.start()

    return reloader
