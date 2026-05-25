"""Central asset coordinator managing lifecycles and ref counts."""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, TypeVar

from engine.resource.constants import ASSET_GENERATION_MASK
from engine.resource.asset.asset_handle import AssetHandle, AssetId, AssetState
from engine.resource.asset.asset_loader import AssetLoader, LoadRequest, LoadResult, SyncLoader

__all__ = ["AssetManager"]

logger = logging.getLogger(__name__)
T = TypeVar("T")


class _AssetEntry:
    """Internal bookkeeping for a single asset slot."""
    __slots__ = ("state", "data", "ref_count", "path", "asset_type", "generation")

    def __init__(self) -> None:
        self.state: AssetState = AssetState.UNLOADED
        self.data: Any = None
        self.ref_count: int = 0
        self.path: str = ""
        self.asset_type: type | None = None
        self.generation: int = 0


class AssetManager:
    """Coordinates asset loading, unloading, and access."""
    __slots__ = ("_entries", "_path_to_index", "_free_list", "_next_index",
                 "_loader", "_load_queue")

    def __init__(self, loader: AssetLoader | None = None) -> None:
        self._entries: list[_AssetEntry] = []
        self._path_to_index: dict[str, int] = {}
        self._free_list: list[int] = []
        self._next_index: int = 0
        self._loader: AssetLoader = loader or SyncLoader()
        self._load_queue: deque[tuple[int, LoadRequest]] = deque()

    def _allocate_slot(self) -> tuple[int, int]:
        if self._free_list:
            index = self._free_list.pop()
            entry = self._entries[index]
            return index, entry.generation
        index = self._next_index
        self._next_index += 1
        self._entries.append(_AssetEntry())
        return index, 0

    def load(self, path: str, asset_type: type[T] | None = None) -> AssetHandle[T]:
        """Request an asset to be loaded. Returns a handle immediately."""
        # Deduplicate by path
        if path in self._path_to_index:
            index = self._path_to_index[path]
            entry = self._entries[index]
            entry.ref_count += 1
            return AssetHandle(index, entry.generation, asset_type or entry.asset_type)

        index, gen = self._allocate_slot()
        entry = self._entries[index]
        entry.state = AssetState.QUEUED
        entry.path = path
        entry.asset_type = asset_type
        entry.ref_count = 1
        entry.data = None
        self._path_to_index[path] = index

        request = LoadRequest(path=path, asset_type=asset_type or object)
        self._load_queue.append((index, request))

        return AssetHandle(index, gen, asset_type)

    def unload(self, handle: AssetHandle[Any]) -> None:
        """Decrement ref count; free when it reaches zero."""
        if not handle.is_valid():
            return
        index = handle.index
        if index >= len(self._entries):
            return
        entry = self._entries[index]
        if entry.generation != handle.generation:
            return
        if entry.state == AssetState.UNLOADED or entry.ref_count <= 0:
            logger.warning("Attempted to unload already-unloaded asset at index %d", index)
            return
        entry.ref_count -= 1
        if entry.ref_count <= 0:
            self._loader.unload(entry.data)
            old_path = entry.path
            entry.state = AssetState.UNLOADED
            entry.data = None
            entry.ref_count = 0
            entry.path = ""
            entry.asset_type = None
            entry.generation = (entry.generation + 1) & ASSET_GENERATION_MASK
            self._path_to_index.pop(old_path, None)
            self._free_list.append(index)

    def get(self, handle: AssetHandle[T]) -> T | None:
        """Return the loaded asset data, or None."""
        if not handle.is_valid():
            return None
        index = handle.index
        if index >= len(self._entries):
            return None
        entry = self._entries[index]
        if entry.generation != handle.generation:
            return None
        if entry.state != AssetState.READY:
            return None
        return entry.data

    def is_loaded(self, handle: AssetHandle[Any]) -> bool:
        return self.get_state(handle) == AssetState.READY

    def get_state(self, handle: AssetHandle[Any]) -> AssetState:
        if not handle.is_valid():
            return AssetState.UNLOADED
        index = handle.index
        if index >= len(self._entries):
            return AssetState.UNLOADED
        entry = self._entries[index]
        if entry.generation != handle.generation:
            return AssetState.UNLOADED
        return entry.state

    def update(self) -> None:
        """Process pending load queue (call each frame)."""
        while self._load_queue:
            index, request = self._load_queue.popleft()
            entry = self._entries[index]
            if entry.state == AssetState.UNLOADED:
                continue  # was unloaded before we got to it
            entry.state = AssetState.LOADING
            result = self._loader.load(request.path, request.asset_type)
            if result.success:
                entry.data = result.data
                entry.state = AssetState.READY
            else:
                entry.state = AssetState.FAILED
                logger.error("Failed to load %s: %s", request.path, result.error)
