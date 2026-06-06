"""Asset hot-reload system with handle indirection (T-CC-3.1 Level 2).

Provides:
- IndirectHandle[T]: Lightweight reference that remains valid across reloads
- AssetHandleTable: Maps handles to current asset instances
- AssetReloader: Watches files and triggers reloads via handle table
- Support for textures, meshes, audio assets
"""
from __future__ import annotations

import enum
import logging
import os
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from engine.core.file_watcher import FileChangeEvent, FileChangeType, FileWatcher
from engine.resource.constants import HOT_RELOAD_POLL_INTERVAL, THREAD_JOIN_TIMEOUT_MULTIPLIER

__all__ = [
    "HotReloadWatcher",
    "IndirectHandle",
    "AssetHandleTable",
    "AssetReloader",
    "ReloadStrategy",
    "ReloadEvent",
    "ReloadCallback",
    "AssetProcessor",
    "TextureProcessor",
    "MeshProcessor",
    "AudioProcessor",
    "ReloadError",
]

logger = logging.getLogger(__name__)
T = TypeVar("T")


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class ReloadError(Exception):
    """Error during asset reload."""

    def __init__(self, path: str, message: str, cause: Optional[Exception] = None):
        self.path = path
        self.message = message
        self.cause = cause
        super().__init__(f"Failed to reload {path}: {message}")


# -----------------------------------------------------------------------------
# Basic mtime-based watcher (existing interface preserved)
# -----------------------------------------------------------------------------


class HotReloadWatcher:
    """Monitors files for changes using mtime polling (legacy API)."""

    __slots__ = ("_watches", "_running", "_thread", "_interval", "_lock")

    def __init__(self, interval: float = HOT_RELOAD_POLL_INTERVAL) -> None:
        self._watches: dict[str, tuple[float, Callable[[str], None]]] = {}
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._interval: float = interval
        self._lock: threading.Lock = threading.Lock()

    def register(self, path: str, callback: Callable[[str], None]) -> None:
        """Register a file to watch with a callback."""
        mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
        with self._lock:
            self._watches[path] = (mtime, callback)

    def unregister(self, path: str) -> None:
        """Unregister a watched file."""
        with self._lock:
            self._watches.pop(path, None)

    def poll(self) -> list[str]:
        """Check all watched paths; fire callbacks for changed files."""
        changed: list[str] = []
        with self._lock:
            snapshot = list(self._watches.items())
        for path, (old_mtime, callback) in snapshot:
            try:
                current_mtime = os.path.getmtime(path)
            except OSError:
                continue
            if current_mtime != old_mtime:
                with self._lock:
                    if path in self._watches:
                        self._watches[path] = (current_mtime, callback)
                callback(path)
                changed.append(path)
        return changed

    def start(self) -> None:
        """Start background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._interval * THREAD_JOIN_TIMEOUT_MULTIPLIER)
            self._thread = None

    def _poll_loop(self) -> None:
        while self._running:
            self.poll()
            time.sleep(self._interval)


# -----------------------------------------------------------------------------
# Handle Indirection System
# -----------------------------------------------------------------------------


class HandleState(enum.Enum):
    """State of an indirect handle."""

    EMPTY = 0
    LOADING = 1
    READY = 2
    RELOADING = 3
    FAILED = 4
    DISPOSED = 5


@dataclass(slots=True)
class HandleEntry(Generic[T]):
    """Internal entry in the handle table."""

    handle_id: int
    generation: int
    asset_path: str
    asset_type: Type[T]
    state: HandleState
    data: Optional[T]
    error: Optional[str]
    ref_count: int
    version: int  # Incremented on each reload
    metadata: Dict[str, Any]


class IndirectHandle(Generic[T]):
    """Lightweight indirect reference to an asset.

    The handle itself doesn't hold the actual asset data. Instead, it holds
    a reference to a slot in the AssetHandleTable. When the asset is reloaded,
    the table entry is updated but handles remain valid.

    Attributes:
        handle_id: Unique identifier for this handle slot
        generation: Generation counter to detect stale handles
        asset_type: The expected type of the asset
    """

    __slots__ = ("_handle_id", "_generation", "_asset_type", "_table_ref")

    def __init__(
        self,
        handle_id: int,
        generation: int,
        asset_type: Type[T],
        table: "AssetHandleTable",
    ) -> None:
        self._handle_id = handle_id
        self._generation = generation
        self._asset_type = asset_type
        self._table_ref: weakref.ref[AssetHandleTable] = weakref.ref(table)

    @property
    def handle_id(self) -> int:
        """Unique slot identifier."""
        return self._handle_id

    @property
    def generation(self) -> int:
        """Generation at time of handle creation."""
        return self._generation

    @property
    def asset_type(self) -> Type[T]:
        """Expected asset type."""
        return self._asset_type

    def is_valid(self) -> bool:
        """Check if handle points to a valid, current entry."""
        table = self._table_ref()
        if table is None:
            return False
        return table.is_handle_valid(self)

    def get(self) -> Optional[T]:
        """Get the current asset data, or None if unavailable."""
        table = self._table_ref()
        if table is None:
            return None
        return table.get(self)

    def get_state(self) -> HandleState:
        """Get the current state of the asset."""
        table = self._table_ref()
        if table is None:
            return HandleState.DISPOSED
        return table.get_state(self)

    def get_version(self) -> int:
        """Get the current version (reload count)."""
        table = self._table_ref()
        if table is None:
            return -1
        return table.get_version(self)

    def get_path(self) -> Optional[str]:
        """Get the asset path."""
        table = self._table_ref()
        if table is None:
            return None
        return table.get_path(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndirectHandle):
            return NotImplemented
        return (
            self._handle_id == other._handle_id
            and self._generation == other._generation
        )

    def __hash__(self) -> int:
        return hash((self._handle_id, self._generation))

    def __repr__(self) -> str:
        state = self.get_state().name if self.is_valid() else "INVALID"
        type_name = self._asset_type.__name__ if self._asset_type else "?"
        return f"IndirectHandle(id={self._handle_id}, gen={self._generation}, type={type_name}, state={state})"


class AssetHandleTable:
    """Maps handles to current asset instances.

    The table manages the lifecycle of assets and provides stable handles
    that remain valid across reloads. When an asset is reloaded, only the
    table entry's data field is updated.
    """

    __slots__ = (
        "_entries",
        "_path_to_id",
        "_free_list",
        "_next_id",
        "_lock",
        "_reload_callbacks",
        "_disposed",
        "__weakref__",  # Allow weak references
    )

    def __init__(self) -> None:
        self._entries: Dict[int, HandleEntry[Any]] = {}
        self._path_to_id: Dict[str, int] = {}
        self._free_list: List[int] = []
        self._next_id: int = 0
        self._lock = threading.RLock()
        self._reload_callbacks: List[Callable[[str, int], None]] = []
        self._disposed = False

    def _allocate_id(self) -> Tuple[int, int]:
        """Allocate a handle ID, reusing from free list if possible."""
        if self._free_list:
            handle_id = self._free_list.pop()
            entry = self._entries.get(handle_id)
            gen = entry.generation if entry else 0
            return handle_id, gen
        handle_id = self._next_id
        self._next_id += 1
        return handle_id, 0

    def register(
        self,
        path: str,
        asset_type: Type[T],
        initial_data: Optional[T] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IndirectHandle[T]:
        """Register an asset and return a handle.

        If an asset with the same path already exists, returns an existing handle
        with incremented ref count.
        """
        with self._lock:
            if self._disposed:
                raise RuntimeError("AssetHandleTable has been disposed")

            # Deduplicate by path
            if path in self._path_to_id:
                handle_id = self._path_to_id[path]
                entry = self._entries[handle_id]
                entry.ref_count += 1
                return IndirectHandle(
                    handle_id, entry.generation, asset_type, self
                )

            handle_id, gen = self._allocate_id()
            state = HandleState.READY if initial_data is not None else HandleState.EMPTY
            entry: HandleEntry[T] = HandleEntry(
                handle_id=handle_id,
                generation=gen,
                asset_path=path,
                asset_type=asset_type,
                state=state,
                data=initial_data,
                error=None,
                ref_count=1,
                version=0,
                metadata=metadata or {},
            )
            self._entries[handle_id] = entry
            self._path_to_id[path] = handle_id
            return IndirectHandle(handle_id, gen, asset_type, self)

    def unregister(self, handle: IndirectHandle[Any]) -> bool:
        """Decrement ref count; free slot when it reaches zero."""
        with self._lock:
            if not self._is_handle_valid_internal(handle):
                return False

            entry = self._entries[handle.handle_id]
            entry.ref_count -= 1

            if entry.ref_count <= 0:
                self._path_to_id.pop(entry.asset_path, None)
                entry.state = HandleState.DISPOSED
                entry.data = None
                entry.generation += 1
                self._free_list.append(handle.handle_id)
                return True

            return True

    def get(self, handle: IndirectHandle[T]) -> Optional[T]:
        """Get the current asset data for a handle."""
        with self._lock:
            if not self._is_handle_valid_internal(handle):
                return None
            entry = self._entries[handle.handle_id]
            if entry.state != HandleState.READY:
                return None
            return cast(T, entry.data)

    def get_state(self, handle: IndirectHandle[Any]) -> HandleState:
        """Get the state of the asset."""
        with self._lock:
            if not self._is_handle_valid_internal(handle):
                return HandleState.DISPOSED
            return self._entries[handle.handle_id].state

    def get_version(self, handle: IndirectHandle[Any]) -> int:
        """Get the reload version counter."""
        with self._lock:
            if not self._is_handle_valid_internal(handle):
                return -1
            return self._entries[handle.handle_id].version

    def get_path(self, handle: IndirectHandle[Any]) -> Optional[str]:
        """Get the asset path for a handle."""
        with self._lock:
            if not self._is_handle_valid_internal(handle):
                return None
            return self._entries[handle.handle_id].asset_path

    def get_error(self, handle: IndirectHandle[Any]) -> Optional[str]:
        """Get error message if asset failed to load."""
        with self._lock:
            if not self._is_handle_valid_internal(handle):
                return None
            return self._entries[handle.handle_id].error

    def get_metadata(self, handle: IndirectHandle[Any]) -> Dict[str, Any]:
        """Get metadata for asset."""
        with self._lock:
            if not self._is_handle_valid_internal(handle):
                return {}
            return dict(self._entries[handle.handle_id].metadata)

    def set_metadata(
        self, handle: IndirectHandle[Any], key: str, value: Any
    ) -> bool:
        """Set a metadata value."""
        with self._lock:
            if not self._is_handle_valid_internal(handle):
                return False
            self._entries[handle.handle_id].metadata[key] = value
            return True

    def is_handle_valid(self, handle: IndirectHandle[Any]) -> bool:
        """Check if a handle is still valid."""
        with self._lock:
            return self._is_handle_valid_internal(handle)

    def _is_handle_valid_internal(self, handle: IndirectHandle[Any]) -> bool:
        """Internal validity check (caller must hold lock)."""
        if handle.handle_id not in self._entries:
            return False
        entry = self._entries[handle.handle_id]
        if entry.generation != handle.generation:
            return False
        if entry.state == HandleState.DISPOSED:
            return False
        return True

    def update_data(
        self,
        path: str,
        new_data: Any,
        increment_version: bool = True,
    ) -> bool:
        """Update the data for an asset by path (used during reload)."""
        with self._lock:
            if path not in self._path_to_id:
                return False
            handle_id = self._path_to_id[path]
            entry = self._entries[handle_id]
            entry.data = new_data
            entry.state = HandleState.READY
            entry.error = None
            if increment_version:
                entry.version += 1
            # Notify reload callbacks
            version = entry.version
        # Callbacks outside lock to avoid deadlock
        for cb in self._reload_callbacks:
            try:
                cb(path, version)
            except Exception as e:
                logger.warning("Reload callback error for %s: %s", path, e)
        return True

    def set_state(self, path: str, state: HandleState) -> bool:
        """Set the state of an asset by path."""
        with self._lock:
            if path not in self._path_to_id:
                return False
            handle_id = self._path_to_id[path]
            self._entries[handle_id].state = state
            return True

    def set_error(self, path: str, error: str) -> bool:
        """Set error for an asset."""
        with self._lock:
            if path not in self._path_to_id:
                return False
            handle_id = self._path_to_id[path]
            entry = self._entries[handle_id]
            entry.state = HandleState.FAILED
            entry.error = error
            return True

    def get_handle_by_path(self, path: str) -> Optional[IndirectHandle[Any]]:
        """Get a handle by asset path."""
        with self._lock:
            if path not in self._path_to_id:
                return None
            handle_id = self._path_to_id[path]
            entry = self._entries[handle_id]
            return IndirectHandle(
                handle_id, entry.generation, entry.asset_type, self
            )

    def get_all_paths(self) -> List[str]:
        """Get all registered asset paths."""
        with self._lock:
            return list(self._path_to_id.keys())

    def get_paths_by_type(self, asset_type: Type[Any]) -> List[str]:
        """Get all paths for a specific asset type."""
        with self._lock:
            result = []
            for path, handle_id in self._path_to_id.items():
                entry = self._entries[handle_id]
                if entry.asset_type == asset_type or issubclass(
                    entry.asset_type, asset_type
                ):
                    result.append(path)
            return result

    def add_reload_callback(
        self, callback: Callable[[str, int], None]
    ) -> None:
        """Register a callback to be called on reload (path, version)."""
        with self._lock:
            if callback not in self._reload_callbacks:
                self._reload_callbacks.append(callback)

    def remove_reload_callback(
        self, callback: Callable[[str, int], None]
    ) -> bool:
        """Unregister a reload callback."""
        with self._lock:
            if callback in self._reload_callbacks:
                self._reload_callbacks.remove(callback)
                return True
            return False

    @property
    def entry_count(self) -> int:
        """Number of registered entries."""
        with self._lock:
            return len(self._path_to_id)

    @property
    def disposed(self) -> bool:
        """Whether the table has been disposed."""
        return self._disposed

    def dispose(self) -> None:
        """Dispose of all entries and mark table as disposed."""
        with self._lock:
            for entry in self._entries.values():
                entry.data = None
                entry.state = HandleState.DISPOSED
            self._entries.clear()
            self._path_to_id.clear()
            self._free_list.clear()
            self._reload_callbacks.clear()
            self._disposed = True


# -----------------------------------------------------------------------------
# Asset Processors
# -----------------------------------------------------------------------------


class AssetProcessor(ABC, Generic[T]):
    """Abstract base for processing specific asset types."""

    @property
    @abstractmethod
    def supported_extensions(self) -> Set[str]:
        """File extensions this processor handles (e.g., {'.png', '.jpg'})."""
        ...

    @abstractmethod
    def load(self, path: str) -> T:
        """Load asset from path."""
        ...

    @abstractmethod
    def unload(self, asset: T) -> None:
        """Unload/cleanup asset."""
        ...

    def can_process(self, path: str) -> bool:
        """Check if this processor can handle the file."""
        ext = os.path.splitext(path)[1].lower()
        return ext in self.supported_extensions


@dataclass
class TextureData:
    """Representation of texture data."""

    path: str
    width: int
    height: int
    channels: int
    data: bytes
    format: str = "rgba8"


class TextureProcessor(AssetProcessor[TextureData]):
    """Processor for texture assets."""

    @property
    def supported_extensions(self) -> Set[str]:
        return {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".dds"}

    def load(self, path: str) -> TextureData:
        """Load texture from file."""
        with open(path, "rb") as f:
            data = f.read()

        # Parse basic info from PNG header if available
        width, height, channels = 0, 0, 4
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            # PNG: IHDR chunk starts at byte 8
            if len(data) >= 24:
                import struct

                width = struct.unpack(">I", data[16:20])[0]
                height = struct.unpack(">I", data[20:24])[0]

        return TextureData(
            path=path,
            width=width,
            height=height,
            channels=channels,
            data=data,
        )

    def unload(self, asset: TextureData) -> None:
        """Release texture resources."""
        pass  # Data is garbage collected


@dataclass
class MeshData:
    """Representation of mesh data."""

    path: str
    vertex_count: int
    index_count: int
    vertices: bytes
    indices: bytes
    format: str = "position_normal_uv"


class MeshProcessor(AssetProcessor[MeshData]):
    """Processor for mesh assets."""

    @property
    def supported_extensions(self) -> Set[str]:
        return {".obj", ".fbx", ".gltf", ".glb"}

    def load(self, path: str) -> MeshData:
        """Load mesh from file."""
        with open(path, "rb") as f:
            data = f.read()

        # Basic OBJ vertex counting
        vertex_count = 0
        index_count = 0
        if path.endswith(".obj"):
            for line in data.decode("utf-8", errors="ignore").split("\n"):
                if line.startswith("v "):
                    vertex_count += 1
                elif line.startswith("f "):
                    # Count face indices
                    parts = line.split()[1:]
                    index_count += max(0, len(parts) - 2) * 3  # Triangulate

        return MeshData(
            path=path,
            vertex_count=vertex_count,
            index_count=index_count,
            vertices=data,
            indices=b"",
        )

    def unload(self, asset: MeshData) -> None:
        """Release mesh resources."""
        pass


@dataclass
class AudioData:
    """Representation of audio data."""

    path: str
    sample_rate: int
    channels: int
    duration_ms: int
    data: bytes
    format: str = "pcm16"


class AudioProcessor(AssetProcessor[AudioData]):
    """Processor for audio assets."""

    @property
    def supported_extensions(self) -> Set[str]:
        return {".wav", ".mp3", ".ogg", ".flac"}

    def load(self, path: str) -> AudioData:
        """Load audio from file."""
        with open(path, "rb") as f:
            data = f.read()

        # Parse WAV header if present
        sample_rate = 44100
        channels = 2
        duration_ms = 0
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            import struct

            # Parse fmt chunk
            fmt_offset = data.find(b"fmt ")
            if fmt_offset >= 0:
                channels = struct.unpack("<H", data[fmt_offset + 10 : fmt_offset + 12])[0]
                sample_rate = struct.unpack("<I", data[fmt_offset + 12 : fmt_offset + 16])[0]
                # Calculate duration from data chunk
                data_offset = data.find(b"data")
                if data_offset >= 0:
                    data_size = struct.unpack("<I", data[data_offset + 4 : data_offset + 8])[0]
                    bytes_per_sample = 2  # 16-bit PCM
                    if sample_rate > 0 and channels > 0:
                        duration_ms = int(
                            (data_size / (sample_rate * channels * bytes_per_sample)) * 1000
                        )

        return AudioData(
            path=path,
            sample_rate=sample_rate,
            channels=channels,
            duration_ms=duration_ms,
            data=data,
        )

    def unload(self, asset: AudioData) -> None:
        """Release audio resources."""
        pass


# -----------------------------------------------------------------------------
# Reload Events and Callbacks
# -----------------------------------------------------------------------------


class ReloadStrategy(enum.Enum):
    """Strategy for handling reload."""

    IMMEDIATE = 0  # Reload immediately on file change
    DEFERRED = 1  # Queue reload for batch processing
    ON_ACCESS = 2  # Reload only when asset is next accessed


@dataclass
class ReloadEvent:
    """Information about a reload event."""

    path: str
    change_type: FileChangeType
    timestamp: float
    old_version: int
    new_version: int
    success: bool
    error: Optional[str] = None
    reload_time_ms: float = 0.0


ReloadCallback = Callable[[ReloadEvent], None]


# -----------------------------------------------------------------------------
# Asset Reloader
# -----------------------------------------------------------------------------


class AssetReloader:
    """Watches asset files and triggers reloads via handle table.

    Integrates with FileWatcher for file change detection and automatically
    reloads modified assets, updating the handle table so existing handles
    continue to work with the new data.
    """

    __slots__ = (
        "_handle_table",
        "_file_watcher",
        "_processors",
        "_strategy",
        "_reload_queue",
        "_reload_callbacks",
        "_running",
        "_worker_thread",
        "_lock",
        "_watched_dirs",
        "_stats",
    )

    def __init__(
        self,
        handle_table: AssetHandleTable,
        file_watcher: Optional[FileWatcher] = None,
        strategy: ReloadStrategy = ReloadStrategy.IMMEDIATE,
    ) -> None:
        self._handle_table = handle_table
        self._file_watcher = file_watcher or FileWatcher(poll_interval_ms=500)
        self._processors: Dict[str, AssetProcessor[Any]] = {}
        self._strategy = strategy
        self._reload_queue: List[Tuple[str, FileChangeType]] = []
        self._reload_callbacks: List[ReloadCallback] = []
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._watched_dirs: Set[str] = set()
        self._stats = {
            "reloads_attempted": 0,
            "reloads_succeeded": 0,
            "reloads_failed": 0,
            "total_reload_time_ms": 0.0,
        }

        # Register default processors
        self._register_default_processors()
        # Set up file watcher callback
        self._file_watcher.registry.register_global(self._on_file_change)

    def _register_default_processors(self) -> None:
        """Register built-in asset processors."""
        self.register_processor(TextureProcessor())
        self.register_processor(MeshProcessor())
        self.register_processor(AudioProcessor())

    def register_processor(self, processor: AssetProcessor[Any]) -> None:
        """Register an asset processor for specific extensions."""
        for ext in processor.supported_extensions:
            self._processors[ext.lower()] = processor

    def unregister_processor(self, extension: str) -> bool:
        """Unregister processor for an extension."""
        ext = extension.lower() if extension.startswith(".") else f".{extension}".lower()
        with self._lock:
            if ext in self._processors:
                del self._processors[ext]
                return True
            return False

    def get_processor(self, path: str) -> Optional[AssetProcessor[Any]]:
        """Get the processor for a file path."""
        ext = os.path.splitext(path)[1].lower()
        return self._processors.get(ext)

    def watch_directory(
        self,
        directory: str,
        recursive: bool = True,
        patterns: Optional[Set[str]] = None,
    ) -> bool:
        """Add a directory to watch for asset changes."""
        dir_path = os.path.abspath(directory)
        if not os.path.isdir(dir_path):
            return False

        # Default patterns for all supported extensions
        if patterns is None:
            patterns = set()
            for ext in self._processors:
                patterns.add(f"*{ext}")

        success = self._file_watcher.watch(
            dir_path, recursive=recursive, patterns=patterns
        )
        if success:
            with self._lock:
                self._watched_dirs.add(dir_path)
        return success

    def unwatch_directory(self, directory: str) -> bool:
        """Remove a directory from watching."""
        dir_path = os.path.abspath(directory)
        success = self._file_watcher.unwatch(dir_path)
        if success:
            with self._lock:
                self._watched_dirs.discard(dir_path)
        return success

    def watch_asset(self, path: str) -> bool:
        """Watch a specific asset file."""
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            return False
        return self._file_watcher.watch(abs_path)

    def _on_file_change(self, event: FileChangeEvent) -> None:
        """Handle file change event from FileWatcher."""
        path = str(event.path)
        processor = self.get_processor(path)
        if processor is None:
            return  # Not a supported asset type

        # Check if we're tracking this asset
        handle = self._handle_table.get_handle_by_path(path)
        if handle is None:
            return  # Asset not registered

        if self._strategy == ReloadStrategy.IMMEDIATE:
            self._reload_asset(path, event.change_type)
        elif self._strategy == ReloadStrategy.DEFERRED:
            with self._lock:
                self._reload_queue.append((path, event.change_type))
        # ON_ACCESS is handled lazily in get()

    def _reload_asset(self, path: str, change_type: FileChangeType) -> ReloadEvent:
        """Perform the actual reload of an asset."""
        start_time = time.perf_counter()
        old_version = self._handle_table.get_version(
            self._handle_table.get_handle_by_path(path)  # type: ignore
        )

        self._handle_table.set_state(path, HandleState.RELOADING)
        with self._lock:
            self._stats["reloads_attempted"] += 1

        success = False
        error: Optional[str] = None
        new_version = old_version

        try:
            if change_type == FileChangeType.DELETED:
                # Asset was deleted - mark as failed
                self._handle_table.set_error(path, "Asset file deleted")
                error = "Asset file deleted"
            else:
                processor = self.get_processor(path)
                if processor is None:
                    raise ReloadError(path, "No processor for asset type")

                new_data = processor.load(path)
                self._handle_table.update_data(path, new_data)
                success = True
                new_version = old_version + 1
                with self._lock:
                    self._stats["reloads_succeeded"] += 1

        except Exception as e:
            error = str(e)
            self._handle_table.set_error(path, error)
            logger.error("Failed to reload asset %s: %s", path, e)
            with self._lock:
                self._stats["reloads_failed"] += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        with self._lock:
            self._stats["total_reload_time_ms"] += elapsed_ms

        event = ReloadEvent(
            path=path,
            change_type=change_type,
            timestamp=time.time(),
            old_version=old_version,
            new_version=new_version,
            success=success,
            error=error,
            reload_time_ms=elapsed_ms,
        )

        # Notify callbacks
        for cb in self._reload_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning("Reload callback error: %s", e)

        return event

    def process_queue(self) -> List[ReloadEvent]:
        """Process pending reload queue (for DEFERRED strategy)."""
        events: List[ReloadEvent] = []
        with self._lock:
            queue_snapshot = list(self._reload_queue)
            self._reload_queue.clear()

        for path, change_type in queue_snapshot:
            event = self._reload_asset(path, change_type)
            events.append(event)

        return events

    def reload(self, path: str) -> ReloadEvent:
        """Manually trigger a reload of an asset."""
        return self._reload_asset(path, FileChangeType.MODIFIED)

    def reload_all(self) -> List[ReloadEvent]:
        """Reload all registered assets."""
        events: List[ReloadEvent] = []
        for path in self._handle_table.get_all_paths():
            if os.path.exists(path):
                event = self._reload_asset(path, FileChangeType.MODIFIED)
                events.append(event)
        return events

    def add_reload_callback(self, callback: ReloadCallback) -> None:
        """Register a callback for reload events."""
        with self._lock:
            if callback not in self._reload_callbacks:
                self._reload_callbacks.append(callback)

    def remove_reload_callback(self, callback: ReloadCallback) -> bool:
        """Unregister a reload callback."""
        with self._lock:
            if callback in self._reload_callbacks:
                self._reload_callbacks.remove(callback)
                return True
            return False

    def start(self) -> None:
        """Start the file watcher and reloader."""
        if self._running:
            return
        self._running = True
        self._file_watcher.start()

    def stop(self) -> None:
        """Stop the file watcher and reloader."""
        self._running = False
        self._file_watcher.stop()

    def poll(self) -> List[FileChangeEvent]:
        """Manually poll for file changes."""
        return self._file_watcher.poll_once()

    @property
    def is_running(self) -> bool:
        """Whether the reloader is running."""
        return self._running

    @property
    def strategy(self) -> ReloadStrategy:
        """Current reload strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: ReloadStrategy) -> None:
        """Set reload strategy."""
        self._strategy = value

    @property
    def stats(self) -> Dict[str, Any]:
        """Get reload statistics."""
        with self._lock:
            return dict(self._stats)

    @property
    def watched_directories(self) -> List[str]:
        """Get list of watched directories."""
        with self._lock:
            return list(self._watched_dirs)

    @property
    def pending_reload_count(self) -> int:
        """Number of pending reloads in queue."""
        with self._lock:
            return len(self._reload_queue)

    def dispose(self) -> None:
        """Dispose of reloader resources."""
        self.stop()
        self._file_watcher.clear()
        with self._lock:
            self._reload_queue.clear()
            self._reload_callbacks.clear()
            self._processors.clear()
