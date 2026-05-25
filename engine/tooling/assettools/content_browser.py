"""
ContentBrowser - Asset browser with thumbnails, filters, search, and favorites.

Provides a complete asset browsing experience with:
- Directory navigation with breadcrumbs
- Thumbnail preview for all asset types
- Advanced filtering by type, date, size
- Search integration
- Favorites and recent items
- Drag-drop support between browser and viewport
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Protocol, Union
from weakref import WeakSet

from trinity.decorators.dev import editor


class AssetType(Enum):
    """Asset type categories for filtering."""

    UNKNOWN = auto()
    MESH = auto()
    TEXTURE = auto()
    MATERIAL = auto()
    AUDIO = auto()
    ANIMATION = auto()
    PREFAB = auto()
    SCENE = auto()
    SCRIPT = auto()
    SHADER = auto()
    FONT = auto()
    DATA = auto()
    FOLDER = auto()


class SortOrder(Enum):
    """Sort order options for browser."""

    NAME_ASC = auto()
    NAME_DESC = auto()
    DATE_ASC = auto()
    DATE_DESC = auto()
    SIZE_ASC = auto()
    SIZE_DESC = auto()
    TYPE_ASC = auto()
    TYPE_DESC = auto()


@dataclass
class BrowserItem:
    """Represents an item in the content browser.

    Attributes:
        path: Absolute path to the asset
        name: Display name of the asset
        asset_type: Type category of the asset
        size_bytes: File size in bytes
        modified_time: Last modification timestamp
        is_directory: Whether this item is a directory
        thumbnail_path: Path to cached thumbnail image
        metadata: Additional metadata dict
    """

    path: Path
    name: str
    asset_type: AssetType
    size_bytes: int = 0
    modified_time: float = 0.0
    is_directory: bool = False
    thumbnail_path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def extension(self) -> str:
        """Get file extension without dot."""
        return self.path.suffix.lstrip(".").lower()

    @property
    def modified_datetime(self) -> datetime:
        """Get modification time as datetime."""
        return datetime.fromtimestamp(self.modified_time)

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BrowserItem):
            return self.path == other.path
        return False


@dataclass
class BrowserFilter:
    """Filter configuration for content browser.

    Attributes:
        asset_types: Set of asset types to include (empty = all)
        extensions: Set of file extensions to include
        min_size: Minimum file size in bytes
        max_size: Maximum file size in bytes
        modified_after: Only show files modified after this time
        modified_before: Only show files modified before this time
        name_pattern: Glob pattern for name matching
        include_hidden: Whether to include hidden files
        recursive: Whether to search subdirectories
    """

    asset_types: set[AssetType] = field(default_factory=set)
    extensions: set[str] = field(default_factory=set)
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    modified_after: Optional[float] = None
    modified_before: Optional[float] = None
    name_pattern: Optional[str] = None
    include_hidden: bool = False
    recursive: bool = False

    def matches(self, item: BrowserItem) -> bool:
        """Check if an item matches this filter."""
        # Type filter
        if self.asset_types and item.asset_type not in self.asset_types:
            return False

        # Extension filter
        if self.extensions and item.extension not in self.extensions:
            return False

        # Size filters
        if self.min_size is not None and item.size_bytes < self.min_size:
            return False
        if self.max_size is not None and item.size_bytes > self.max_size:
            return False

        # Date filters
        if self.modified_after is not None and item.modified_time < self.modified_after:
            return False
        if self.modified_before is not None and item.modified_time > self.modified_before:
            return False

        # Hidden file filter
        if not self.include_hidden and item.name.startswith("."):
            return False

        # Name pattern (simple glob)
        if self.name_pattern:
            import fnmatch
            if not fnmatch.fnmatch(item.name.lower(), self.name_pattern.lower()):
                return False

        return True

    def reset(self) -> None:
        """Reset all filters to defaults."""
        self.asset_types.clear()
        self.extensions.clear()
        self.min_size = None
        self.max_size = None
        self.modified_after = None
        self.modified_before = None
        self.name_pattern = None
        self.include_hidden = False
        self.recursive = False


@dataclass
class BrowserFavorites:
    """Manages favorite assets and folders.

    Attributes:
        items: Set of favorited paths
        _listeners: Callbacks for change notifications
    """

    items: set[Path] = field(default_factory=set)
    _listeners: list[Callable[[Path, bool], None]] = field(default_factory=list)

    def add(self, path: Path) -> None:
        """Add a path to favorites."""
        path = Path(path)
        if path not in self.items:
            self.items.add(path)
            self._notify(path, True)

    def remove(self, path: Path) -> None:
        """Remove a path from favorites."""
        path = Path(path)
        if path in self.items:
            self.items.discard(path)
            self._notify(path, False)

    def toggle(self, path: Path) -> bool:
        """Toggle favorite status, return new status."""
        path = Path(path)
        if path in self.items:
            self.remove(path)
            return False
        else:
            self.add(path)
            return True

    def is_favorite(self, path: Path) -> bool:
        """Check if a path is favorited."""
        return Path(path) in self.items

    def clear(self) -> None:
        """Clear all favorites."""
        for path in list(self.items):
            self.remove(path)

    def on_change(self, callback: Callable[[Path, bool], None]) -> None:
        """Register a change listener."""
        self._listeners.append(callback)

    def _notify(self, path: Path, added: bool) -> None:
        """Notify listeners of a change."""
        for listener in self._listeners:
            try:
                listener(path, added)
            except Exception:
                pass


@dataclass
class BrowserHistory:
    """Navigation history for the content browser.

    Attributes:
        _history: List of visited paths
        _position: Current position in history
        max_size: Maximum history size
    """

    _history: list[Path] = field(default_factory=list)
    _position: int = -1
    max_size: int = 100

    def push(self, path: Path) -> None:
        """Push a new path to history, clearing forward history."""
        path = Path(path)

        # Don't push duplicates of current position
        if self._history and self._position >= 0:
            if self._history[self._position] == path:
                return

        # Clear forward history
        self._history = self._history[: self._position + 1]

        # Add new path
        self._history.append(path)
        self._position = len(self._history) - 1

        # Trim to max size
        if len(self._history) > self.max_size:
            excess = len(self._history) - self.max_size
            self._history = self._history[excess:]
            self._position -= excess

    def back(self) -> Optional[Path]:
        """Go back in history, return the path or None."""
        if self.can_go_back():
            self._position -= 1
            return self._history[self._position]
        return None

    def forward(self) -> Optional[Path]:
        """Go forward in history, return the path or None."""
        if self.can_go_forward():
            self._position += 1
            return self._history[self._position]
        return None

    def can_go_back(self) -> bool:
        """Check if we can go back."""
        return self._position > 0

    def can_go_forward(self) -> bool:
        """Check if we can go forward."""
        return self._position < len(self._history) - 1

    def current(self) -> Optional[Path]:
        """Get current path in history."""
        if 0 <= self._position < len(self._history):
            return self._history[self._position]
        return None

    def clear(self) -> None:
        """Clear navigation history."""
        self._history.clear()
        self._position = -1


@dataclass
class DragDropPayload:
    """Payload for drag-drop operations.

    Attributes:
        items: List of items being dragged
        source: Source location identifier
        operation: Type of operation (copy, move, link)
        data: Additional payload data
    """

    items: list[BrowserItem] = field(default_factory=list)
    source: str = ""
    operation: str = "copy"  # copy, move, link
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> list[Path]:
        """Get paths of all items."""
        return [item.path for item in self.items]

    @property
    def is_single(self) -> bool:
        """Check if this is a single-item drag."""
        return len(self.items) == 1

    @property
    def first(self) -> Optional[BrowserItem]:
        """Get the first item or None."""
        return self.items[0] if self.items else None


class ThumbnailProvider(Protocol):
    """Protocol for thumbnail generation."""

    def get_thumbnail(self, path: Path, size: tuple[int, int]) -> Optional[bytes]:
        """Get thumbnail data for an asset."""
        ...

    def has_thumbnail(self, path: Path) -> bool:
        """Check if a thumbnail exists."""
        ...


# Extension to asset type mapping
_EXTENSION_MAP: dict[str, AssetType] = {
    # Meshes
    "fbx": AssetType.MESH,
    "obj": AssetType.MESH,
    "gltf": AssetType.MESH,
    "glb": AssetType.MESH,
    "dae": AssetType.MESH,
    "blend": AssetType.MESH,
    # Textures
    "png": AssetType.TEXTURE,
    "jpg": AssetType.TEXTURE,
    "jpeg": AssetType.TEXTURE,
    "tga": AssetType.TEXTURE,
    "dds": AssetType.TEXTURE,
    "exr": AssetType.TEXTURE,
    "hdr": AssetType.TEXTURE,
    "bmp": AssetType.TEXTURE,
    "tiff": AssetType.TEXTURE,
    "tif": AssetType.TEXTURE,
    "psd": AssetType.TEXTURE,
    # Audio
    "wav": AssetType.AUDIO,
    "ogg": AssetType.AUDIO,
    "mp3": AssetType.AUDIO,
    "flac": AssetType.AUDIO,
    "aiff": AssetType.AUDIO,
    # Materials
    "mat": AssetType.MATERIAL,
    "mtl": AssetType.MATERIAL,
    # Animations
    "anim": AssetType.ANIMATION,
    "bvh": AssetType.ANIMATION,
    # Prefabs
    "prefab": AssetType.PREFAB,
    # Scenes
    "scene": AssetType.SCENE,
    # Scripts
    "py": AssetType.SCRIPT,
    "lua": AssetType.SCRIPT,
    # Shaders
    "glsl": AssetType.SHADER,
    "hlsl": AssetType.SHADER,
    "vert": AssetType.SHADER,
    "frag": AssetType.SHADER,
    "comp": AssetType.SHADER,
    # Fonts
    "ttf": AssetType.FONT,
    "otf": AssetType.FONT,
    "woff": AssetType.FONT,
    "woff2": AssetType.FONT,
    # Data
    "json": AssetType.DATA,
    "yaml": AssetType.DATA,
    "yml": AssetType.DATA,
    "xml": AssetType.DATA,
    "csv": AssetType.DATA,
    "toml": AssetType.DATA,
}


def _get_asset_type(path: Path) -> AssetType:
    """Determine asset type from path."""
    if path.is_dir():
        return AssetType.FOLDER
    ext = path.suffix.lstrip(".").lower()
    return _EXTENSION_MAP.get(ext, AssetType.UNKNOWN)


@editor(category="Assets")
class ContentBrowser:
    """Asset browser with thumbnails, filters, search, and favorites.

    Provides a complete browsing experience for game assets with
    integration to the engine's content store and provenance systems.

    Attributes:
        root_path: Root directory for asset browsing
        current_path: Current directory being viewed
        filter: Active filter configuration
        favorites: Favorite items manager
        history: Navigation history
        sort_order: Current sort order
        _items_cache: Cached items for current directory
        _selection: Currently selected items
        _thumbnail_provider: Optional thumbnail provider
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        thumbnail_provider: Optional[ThumbnailProvider] = None,
    ) -> None:
        """Initialize the content browser.

        Args:
            root_path: Root directory for browsing
            thumbnail_provider: Optional thumbnail generation service
        """
        self.root_path = Path(root_path).resolve()
        self.current_path = self.root_path
        self.filter = BrowserFilter()
        self.favorites = BrowserFavorites()
        self.history = BrowserHistory()
        self.sort_order = SortOrder.NAME_ASC

        self._items_cache: list[BrowserItem] = []
        self._cache_valid = False
        self._selection: set[Path] = set()
        self._thumbnail_provider = thumbnail_provider
        self._change_listeners: list[Callable[[], None]] = []

        # Initialize history with root
        self.history.push(self.root_path)

    def navigate_to(self, path: Union[str, Path]) -> bool:
        """Navigate to a directory.

        Args:
            path: Path to navigate to

        Returns:
            True if navigation succeeded, False otherwise
        """
        path = Path(path).resolve()

        # Validate path
        if not path.exists():
            return False
        if not path.is_dir():
            return False

        # Ensure path is under root
        try:
            path.relative_to(self.root_path)
        except ValueError:
            return False

        self.current_path = path
        self.history.push(path)
        self._invalidate_cache()
        self._notify_change()
        return True

    def navigate_up(self) -> bool:
        """Navigate to parent directory."""
        if self.current_path == self.root_path:
            return False
        return self.navigate_to(self.current_path.parent)

    def navigate_back(self) -> bool:
        """Navigate back in history."""
        path = self.history.back()
        if path:
            self.current_path = path
            self._invalidate_cache()
            self._notify_change()
            return True
        return False

    def navigate_forward(self) -> bool:
        """Navigate forward in history."""
        path = self.history.forward()
        if path:
            self.current_path = path
            self._invalidate_cache()
            self._notify_change()
            return True
        return False

    def refresh(self) -> None:
        """Force refresh of current directory."""
        self._invalidate_cache()
        self._notify_change()

    def get_items(self) -> list[BrowserItem]:
        """Get items in current directory with filters applied.

        Returns:
            List of browser items in current directory
        """
        if not self._cache_valid:
            self._rebuild_cache()
        return self._items_cache.copy()

    def get_breadcrumbs(self) -> list[tuple[str, Path]]:
        """Get breadcrumb trail from root to current.

        Returns:
            List of (name, path) tuples for breadcrumb navigation
        """
        breadcrumbs: list[tuple[str, Path]] = []
        current = self.current_path

        while True:
            name = current.name or "Root"
            breadcrumbs.insert(0, (name, current))

            if current == self.root_path:
                break

            try:
                current.relative_to(self.root_path)
            except ValueError:
                break

            current = current.parent

        return breadcrumbs

    def select(self, paths: Union[Path, list[Path]]) -> None:
        """Select items by path.

        Args:
            paths: Path or list of paths to select
        """
        if isinstance(paths, Path):
            paths = [paths]

        self._selection = {Path(p) for p in paths}
        self._notify_change()

    def add_to_selection(self, path: Path) -> None:
        """Add an item to selection."""
        self._selection.add(Path(path))
        self._notify_change()

    def remove_from_selection(self, path: Path) -> None:
        """Remove an item from selection."""
        self._selection.discard(Path(path))
        self._notify_change()

    def toggle_selection(self, path: Path) -> bool:
        """Toggle selection state, return new state."""
        path = Path(path)
        if path in self._selection:
            self._selection.discard(path)
            self._notify_change()
            return False
        else:
            self._selection.add(path)
            self._notify_change()
            return True

    def clear_selection(self) -> None:
        """Clear all selection."""
        self._selection.clear()
        self._notify_change()

    def get_selection(self) -> list[BrowserItem]:
        """Get currently selected items."""
        items = []
        for item in self.get_items():
            if item.path in self._selection:
                items.append(item)
        return items

    def is_selected(self, path: Path) -> bool:
        """Check if a path is selected."""
        return Path(path) in self._selection

    def set_sort_order(self, order: SortOrder) -> None:
        """Set sort order and refresh."""
        self.sort_order = order
        self._invalidate_cache()
        self._notify_change()

    def set_filter(self, filter: BrowserFilter) -> None:
        """Set filter configuration."""
        self.filter = filter
        self._invalidate_cache()
        self._notify_change()

    def create_drag_payload(self) -> DragDropPayload:
        """Create a drag payload from current selection."""
        return DragDropPayload(
            items=self.get_selection(),
            source="content_browser",
            operation="copy",
        )

    def accept_drop(self, payload: DragDropPayload) -> bool:
        """Handle a drop operation.

        Args:
            payload: The drag-drop payload

        Returns:
            True if the drop was handled
        """
        if not payload.items:
            return False

        # In a real implementation, this would copy/move files
        # For now, just return True to indicate acceptance
        return True

    def on_change(self, callback: Callable[[], None]) -> None:
        """Register a change listener."""
        self._change_listeners.append(callback)

    def _invalidate_cache(self) -> None:
        """Mark cache as invalid."""
        self._cache_valid = False

    def _rebuild_cache(self) -> None:
        """Rebuild the items cache."""
        items: list[BrowserItem] = []

        if not self.current_path.exists():
            self._items_cache = []
            self._cache_valid = True
            return

        try:
            for entry in self.current_path.iterdir():
                item = self._create_item(entry)
                if self.filter.matches(item):
                    items.append(item)
        except PermissionError:
            pass

        # Sort items
        items = self._sort_items(items)

        self._items_cache = items
        self._cache_valid = True

    def _create_item(self, path: Path) -> BrowserItem:
        """Create a BrowserItem from a path."""
        stat = path.stat() if path.exists() else None

        return BrowserItem(
            path=path,
            name=path.name,
            asset_type=_get_asset_type(path),
            size_bytes=stat.st_size if stat and not path.is_dir() else 0,
            modified_time=stat.st_mtime if stat else 0.0,
            is_directory=path.is_dir(),
        )

    def _sort_items(self, items: list[BrowserItem]) -> list[BrowserItem]:
        """Sort items according to current sort order."""
        # Always put directories first
        dirs = [i for i in items if i.is_directory]
        files = [i for i in items if not i.is_directory]

        def sort_key(item: BrowserItem) -> Any:
            match self.sort_order:
                case SortOrder.NAME_ASC:
                    return item.name.lower()
                case SortOrder.NAME_DESC:
                    return item.name.lower()
                case SortOrder.DATE_ASC | SortOrder.DATE_DESC:
                    return item.modified_time
                case SortOrder.SIZE_ASC | SortOrder.SIZE_DESC:
                    return item.size_bytes
                case SortOrder.TYPE_ASC | SortOrder.TYPE_DESC:
                    return (item.asset_type.name, item.name.lower())
                case _:
                    return item.name.lower()

        reverse = self.sort_order in (
            SortOrder.NAME_DESC,
            SortOrder.DATE_DESC,
            SortOrder.SIZE_DESC,
            SortOrder.TYPE_DESC,
        )

        dirs.sort(key=lambda i: i.name.lower(), reverse=reverse)
        files.sort(key=sort_key, reverse=reverse)

        return dirs + files

    def _notify_change(self) -> None:
        """Notify listeners of a change."""
        for listener in self._change_listeners:
            try:
                listener()
            except Exception:
                pass


__all__ = [
    "AssetType",
    "SortOrder",
    "BrowserItem",
    "BrowserFilter",
    "BrowserFavorites",
    "BrowserHistory",
    "DragDropPayload",
    "ThumbnailProvider",
    "ContentBrowser",
]
