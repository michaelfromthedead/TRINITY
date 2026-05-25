"""Asset subsystem: handles, loaders, registry, hot reload, and dependency tracking."""

from engine.resource.asset.asset_handle import AssetHandle, AssetId, AssetState
from engine.resource.asset.asset_loader import (
    AssetLoader,
    AsyncLoader,
    LoadRequest,
    LoadResult,
    SyncLoader,
)
from engine.resource.asset.asset_manager import AssetManager
from engine.resource.asset.asset_registry import AssetRegistry, AssetType
from engine.resource.asset.dependency_graph import DependencyGraph
from engine.resource.asset.hot_reload import HotReloadWatcher

__all__ = [
    "AssetHandle",
    "AssetId",
    "AssetLoader",
    "AssetManager",
    "AssetRegistry",
    "AssetState",
    "AssetType",
    "AsyncLoader",
    "DependencyGraph",
    "HotReloadWatcher",
    "LoadRequest",
    "LoadResult",
    "SyncLoader",
]
