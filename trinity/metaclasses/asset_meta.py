"""
AssetMeta - Metaclass for asset handle types.

Handles asset registration and loading pipeline setup.
Assets are external resources like textures, models, sounds, etc.
"""

from __future__ import annotations

import heapq
import os
import threading
from typing import Any, Callable, ClassVar, Optional

from trinity.constants import ASSET_QUEUE_PROCESS_BATCH, ASSET_TYPE_CODE_LENGTH
from trinity.decorators.ops import Op, Step
from trinity.metaclasses.engine_meta import EngineMeta
from trinity.types import CachePolicy


class AssetMeta(EngineMeta):
    """
    Metaclass for asset types.

    Created classes will:
    - Be registered in the asset registry
    - Have loading pipeline configured
    - Support async loading
    - Have caching behavior defined

    Required class attributes (set by decorators or class definition):
    - _asset_extensions: tuple[str, ...] (file extensions this type handles)

    Optional class attributes (set by decorators):
    - _asset_loader: type (custom loader class)
    - _asset_cache_policy: CachePolicy
    - _asset_dependencies: tuple[type, ...] (other asset types needed)
    - _asset_hot_reload: bool (support hot reloading in dev)
    - _asset_priority: int (loading priority)

    Attached attributes:
    - _asset_id: int (unique identifier)
    - _asset_name: str (qualified name)
    - _asset_type_code: str (short type code for serialization)
    """

    _registry: ClassVar[dict[int, type]] = {}
    _extension_map: ClassVar[dict[str, type]] = {}  # extension -> asset type
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _load_queue: ClassVar[list] = []  # heap of (priority, counter, asset_cls, path, callback)
    _load_counter: ClassVar[int] = 0  # for stable heap ordering
    _watched_paths: ClassVar[dict[str, tuple[type, float]]] = {}  # path -> (asset_cls, mtime)

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> AssetMeta:
        """Create a new asset type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base Asset class
        if name == "Asset":
            return cls

        with mcs._lock:
            # === 1. GENERATE UNIQUE ID ===
            cls._asset_id = mcs._next_id
            mcs._next_id += 1
            cls._asset_name = f"{cls.__module__}.{name}"
            cls._asset_type_code = name[:ASSET_TYPE_CODE_LENGTH].upper()  # Short code for serialization

            # 3.7.2 — TAG steps for asset_id, asset_type_code
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "asset_id", "value": cls._asset_id}))
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "asset_type_code", "value": cls._asset_type_code}))

            # === 2. VALIDATE EXTENSIONS ===
            extensions = getattr(cls, "_asset_extensions", None)
            if not extensions:
                raise TypeError(
                    f"{cls.__name__}: Asset types must define _asset_extensions. "
                    f"Example: _asset_extensions = ('.png', '.jpg')"
                )

            # Normalize and validate extensions
            normalized_extensions = []
            for ext in extensions:
                ext = ext.lower()
                if not ext.startswith("."):
                    ext = "." + ext
                normalized_extensions.append(ext)
            cls._asset_extensions = tuple(normalized_extensions)

            # 3.7.3 — VALIDATE + TAG for extensions
            cls._metaclass_steps.append(Step(Op.VALIDATE, {"constraint": "asset_extensions_required"}))
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "extensions", "value": cls._asset_extensions}))

            # === 3. CHECK FOR EXTENSION CONFLICTS ===
            for ext in cls._asset_extensions:
                if ext in mcs._extension_map:
                    existing = mcs._extension_map[ext]
                    raise TypeError(
                        f"{cls.__name__}: Extension '{ext}' is already registered "
                        f"to {existing.__name__}"
                    )

            # 3.7.4 — VALIDATE extension uniqueness
            cls._metaclass_steps.append(Step(Op.VALIDATE, {"constraint": "extension_uniqueness"}))

            # === 4. SET DEFAULTS ===
            if not hasattr(cls, "_asset_loader"):
                cls._asset_loader = None
            if not hasattr(cls, "_asset_cache_policy"):
                cls._asset_cache_policy = CachePolicy()
            if not hasattr(cls, "_asset_dependencies"):
                cls._asset_dependencies = ()
            if not hasattr(cls, "_asset_hot_reload"):
                cls._asset_hot_reload = False
            if not hasattr(cls, "_asset_priority"):
                cls._asset_priority = 0

            # 3.7.5 — TAG steps for defaults
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "cache_policy", "value": cls._asset_cache_policy}))
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "hot_reload", "value": cls._asset_hot_reload}))
            cls._metaclass_steps.append(Step(Op.TAG, {"key": "asset_priority", "value": cls._asset_priority}))

            # === 5. REGISTER EXTENSIONS ===
            for ext in cls._asset_extensions:
                mcs._extension_map[ext] = cls
                # 3.7.6 — REGISTER step per extension
                cls._metaclass_steps.append(Step(Op.REGISTER, {"registry": "asset_extension_map"}))

            # === 6. REGISTER ===
            mcs._registry[cls._asset_id] = cls

            # 3.7.7 — REGISTER in asset_registry
            cls._metaclass_steps.append(Step(Op.REGISTER, {"registry": "asset_registry"}))

        return cls

    # =========================================================================
    # REGISTRY ACCESS CLASS METHODS
    # =========================================================================

    @classmethod
    def get_by_id(mcs, asset_id: int) -> Optional[type]:
        """Get asset class by ID."""
        return mcs._registry.get(asset_id)

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]:
        """Get asset class by qualified name."""
        for asset_cls in mcs._registry.values():
            if asset_cls._asset_name == name:
                return asset_cls
        return None

    @classmethod
    def all_assets(mcs) -> list[type]:
        """Get all registered asset classes."""
        return list(mcs._registry.values())

    @classmethod
    def get_for_extension(mcs, extension: str) -> Optional[type]:
        """
        Get asset type for a file extension.

        Args:
            extension: File extension (with or without leading dot).

        Returns:
            Asset type that handles this extension, or None.
        """
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        return mcs._extension_map.get(ext)

    @classmethod
    def get_for_path(mcs, path: str) -> Optional[type]:
        """
        Get asset type for a file path.

        Args:
            path: File path to check.

        Returns:
            Asset type that handles this file, or None.
        """
        # Extract extension from path
        dot_pos = path.rfind(".")
        if dot_pos == -1:
            return None

        extension = path[dot_pos:].lower()
        return mcs._extension_map.get(extension)

    @classmethod
    def get_loader(mcs, asset_type: type) -> Optional[type]:
        """
        Get the loader class for an asset type.

        Args:
            asset_type: Asset type to get loader for.

        Returns:
            Loader class, or None if no custom loader.
        """
        return getattr(asset_type, "_asset_loader", None)

    @classmethod
    def get_supported_extensions(mcs) -> list[str]:
        """Get all registered file extensions."""
        return list(mcs._extension_map.keys())

    @classmethod
    def get_hot_reloadable(mcs) -> list[type]:
        """Get all asset types that support hot reloading."""
        return [
            cls
            for cls in mcs._registry.values()
            if getattr(cls, "_asset_hot_reload", False)
        ]

    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the asset registry. Useful for testing."""
        with mcs._lock:
            mcs._registry.clear()
            mcs._extension_map.clear()
            mcs._next_id = 1
            mcs._load_queue.clear()
            mcs._load_counter = 0
            mcs._watched_paths.clear()
        super().clear_registry()

    # =========================================================================
    # ASYNC LOADING PIPELINE
    # =========================================================================

    @classmethod
    def queue_load(
        mcs,
        asset_cls: type,
        path: str,
        priority: int = 0,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Queue an asset for asynchronous loading.

        Args:
            asset_cls: Asset class to load.
            path: Path to the asset file.
            priority: Loading priority (higher = earlier). Default 0.
            callback: Optional callback to invoke when loaded.

        Raises:
            ValueError: If path is None or empty.
        """
        if not path:
            raise ValueError("path cannot be None or empty")

        with mcs._lock:
            # Use negative priority for max-heap behavior (higher priority first)
            # Include counter for stable ordering when priorities are equal
            heapq.heappush(
                mcs._load_queue,
                (-priority, mcs._load_counter, asset_cls, path, callback),
            )
            mcs._load_counter += 1

    @classmethod
    def process_queue(mcs, max_items: int = ASSET_QUEUE_PROCESS_BATCH) -> int:
        """
        Process queued asset loads.

        Args:
            max_items: Maximum number of assets to process.

        Returns:
            Number of assets processed.
        """
        processed = 0
        # Collect work items without holding lock
        work_items = []
        with mcs._lock:
            while mcs._load_queue and processed < max_items:
                work_items.append(heapq.heappop(mcs._load_queue))
                processed += 1

        # Process callbacks outside lock to prevent deadlock
        for _, _, asset_cls, path, callback in work_items:
            # TODO: Actual loading implementation would go here
            # For now, just invoke callback if provided
            if callback:
                try:
                    callback(asset_cls, path)
                except Exception:
                    pass  # Swallow callback exceptions

        return processed

    @classmethod
    def get_queue_status(mcs) -> dict[str, Any]:
        """
        Get the status of the load queue.

        Returns:
            Dict with 'pending' count and 'total_queued'.
        """
        with mcs._lock:
            return {
                "pending": len(mcs._load_queue),
                "total_queued": mcs._load_counter,
            }

    # =========================================================================
    # HOT-RELOAD WATCHER
    # =========================================================================

    @classmethod
    def watch(mcs, asset_cls: type, path: str) -> None:
        """
        Register a file path for hot-reload watching.

        Args:
            asset_cls: Asset class associated with this path.
            path: File path to watch.
        """
        with mcs._lock:
            try:
                mtime = os.path.getmtime(path)
                mcs._watched_paths[path] = (asset_cls, mtime)
            except (OSError, FileNotFoundError):
                pass  # Ignore if file doesn't exist yet

    @classmethod
    def unwatch(mcs, path: str) -> None:
        """
        Unregister a file path from hot-reload watching.

        Args:
            path: File path to stop watching.
        """
        with mcs._lock:
            mcs._watched_paths.pop(path, None)

    @classmethod
    def check_changes(mcs) -> list[tuple[type, str]]:
        """
        Check for changes in watched files.

        Returns:
            List of (asset_cls, path) tuples for files that have changed.
            Deleted files are automatically unwatched.
        """
        changes = []
        deleted_paths = []
        with mcs._lock:
            for path, (asset_cls, old_mtime) in list(mcs._watched_paths.items()):
                try:
                    new_mtime = os.path.getmtime(path)
                    if new_mtime > old_mtime:
                        mcs._watched_paths[path] = (asset_cls, new_mtime)
                        changes.append((asset_cls, path))
                except (OSError, FileNotFoundError):
                    # File deleted or inaccessible - stop watching it
                    deleted_paths.append(path)

            # Remove deleted paths from watch list
            for path in deleted_paths:
                mcs._watched_paths.pop(path, None)

        return changes

    # =========================================================================
    # DEPENDENCY-ORDERED LOADING
    # =========================================================================

    @classmethod
    def get_load_order(mcs, asset_cls: type) -> list[type]:
        """
        Get dependency-ordered load sequence for an asset class.

        Performs topological sort based on _asset_dependencies.
        Dependencies are loaded before dependents.

        Args:
            asset_cls: Asset class to get load order for.

        Returns:
            List of asset classes in load order (dependencies first).

        Raises:
            ValueError: If circular dependencies are detected.
        """
        visited = set()
        visiting = set()  # Track current path for cycle detection
        result = []

        def visit(cls: type, path: list[type]) -> None:
            if cls in visiting:
                # Circular dependency detected
                cycle = " -> ".join(c.__name__ for c in path + [cls])
                raise ValueError(f"Circular dependency detected: {cycle}")

            if cls in visited:
                return

            visiting.add(cls)

            # Visit dependencies first
            dependencies = getattr(cls, "_asset_dependencies", ())
            for dep_cls in dependencies:
                visit(dep_cls, path + [cls])

            visiting.remove(cls)
            visited.add(cls)
            result.append(cls)

        visit(asset_cls, [])
        return result
