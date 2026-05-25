"""
Camera Bookmarks - Save and restore camera positions for quick navigation.

Provides:
- Named camera bookmarks with position, rotation, and settings
- Bookmark categories for organization
- Quick navigation with smooth transitions
- Bookmark thumbnails
- Import/export functionality

All bookmark operations integrate with Foundation Tracker for undo/redo.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from .placement import Vector3, Quaternion, editor, track_changes
from foundation.tracker import tracker


# =============================================================================
# Enums
# =============================================================================

class TransitionType(Enum):
    """Type of camera transition."""
    INSTANT = auto()  # Immediate jump
    LINEAR = auto()  # Linear interpolation
    EASE_IN = auto()  # Smooth start
    EASE_OUT = auto()  # Smooth end
    EASE_IN_OUT = auto()  # Smooth start and end


class BookmarkIcon(Enum):
    """Icons for bookmark categorization."""
    CAMERA = auto()
    STAR = auto()
    FLAG = auto()
    MARKER = auto()
    EYE = auto()
    TARGET = auto()
    HOME = auto()
    BOOKMARK = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(slots=True)
class CameraSettings:
    """Camera settings stored with bookmark."""
    fov: float = 60.0
    near_plane: float = 0.1
    far_plane: float = 1000.0
    orthographic: bool = False
    ortho_size: float = 10.0
    depth_of_field: bool = False
    focus_distance: float = 10.0
    aperture: float = 5.6


@dataclass(slots=True)
class TransitionSettings:
    """Settings for camera transitions."""
    transition_type: TransitionType = TransitionType.EASE_IN_OUT
    duration: float = 0.5  # Seconds
    animate_fov: bool = True


# =============================================================================
# Bookmark Category
# =============================================================================

@editor
class BookmarkCategory:
    """
    A category for organizing bookmarks.
    """

    __slots__ = (
        "_id",
        "_name",
        "_icon",
        "_color",
        "_expanded",
        "_order",
        "__weakref__",
    )

    def __init__(self, name: str, icon: BookmarkIcon = BookmarkIcon.BOOKMARK):
        """
        Initialize a bookmark category.

        Args:
            name: Category name
            icon: Category icon
        """
        self._id = str(uuid.uuid4())
        self._name = name
        self._icon = icon
        self._color: tuple[float, float, float] = (0.7, 0.7, 0.7)
        self._expanded = True
        self._order = 0

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        old_name = self._name
        self._name = value
        tracker.mark_dirty(self, "_name", old_name, value)

    @property
    def icon(self) -> BookmarkIcon:
        return self._icon

    @icon.setter
    def icon(self, value: BookmarkIcon) -> None:
        old_icon = self._icon
        self._icon = value
        tracker.mark_dirty(self, "_icon", old_icon, value)

    @property
    def color(self) -> tuple[float, float, float]:
        return self._color

    def set_color(self, r: float, g: float, b: float) -> None:
        """Set category color."""
        old_color = self._color
        self._color = (
            max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)),
        )
        tracker.mark_dirty(self, "_color", old_color, self._color)

    @property
    def expanded(self) -> bool:
        return self._expanded

    @expanded.setter
    def expanded(self, value: bool) -> None:
        self._expanded = value

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, value: int) -> None:
        self._order = value


# =============================================================================
# Camera Bookmark
# =============================================================================

@editor
class CameraBookmark:
    """
    A camera bookmark storing position, rotation, and settings.
    """

    __slots__ = (
        "_id",
        "_name",
        "_position",
        "_rotation",
        "_camera_settings",
        "_category_id",
        "_icon",
        "_thumbnail_path",
        "_created_at",
        "_modified_at",
        "_shortcut_key",
        "_description",
        "_tags",
        "__weakref__",
    )

    def __init__(
        self,
        name: str,
        position: Optional[Vector3] = None,
        rotation: Optional[Quaternion] = None,
    ):
        """
        Initialize a camera bookmark.

        Args:
            name: Bookmark name
            position: Camera position
            rotation: Camera rotation
        """
        self._id = str(uuid.uuid4())
        self._name = name
        self._position = position or Vector3()
        self._rotation = rotation or Quaternion.identity()
        self._camera_settings = CameraSettings()
        self._category_id: Optional[str] = None
        self._icon = BookmarkIcon.CAMERA
        self._thumbnail_path: Optional[str] = None
        self._created_at = time.time()
        self._modified_at = time.time()
        self._shortcut_key: Optional[str] = None
        self._description: str = ""
        self._tags: list[str] = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        old_name = self._name
        self._name = value
        self._modified_at = time.time()
        tracker.mark_dirty(self, "_name", old_name, value)

    @property
    def position(self) -> Vector3:
        return self._position

    @position.setter
    def position(self, value: Vector3) -> None:
        old_position = self._position
        self._position = value
        self._modified_at = time.time()
        tracker.mark_dirty(self, "_position", old_position, value)

    @property
    def rotation(self) -> Quaternion:
        return self._rotation

    @rotation.setter
    def rotation(self, value: Quaternion) -> None:
        old_rotation = self._rotation
        self._rotation = value
        self._modified_at = time.time()
        tracker.mark_dirty(self, "_rotation", old_rotation, value)

    @property
    def camera_settings(self) -> CameraSettings:
        return self._camera_settings

    @property
    def category_id(self) -> Optional[str]:
        return self._category_id

    @category_id.setter
    def category_id(self, value: Optional[str]) -> None:
        old_id = self._category_id
        self._category_id = value
        tracker.mark_dirty(self, "_category_id", old_id, value)

    @property
    def icon(self) -> BookmarkIcon:
        return self._icon

    @icon.setter
    def icon(self, value: BookmarkIcon) -> None:
        old_icon = self._icon
        self._icon = value
        tracker.mark_dirty(self, "_icon", old_icon, value)

    @property
    def thumbnail_path(self) -> Optional[str]:
        return self._thumbnail_path

    @thumbnail_path.setter
    def thumbnail_path(self, value: Optional[str]) -> None:
        self._thumbnail_path = value

    @property
    def shortcut_key(self) -> Optional[str]:
        return self._shortcut_key

    @shortcut_key.setter
    def shortcut_key(self, value: Optional[str]) -> None:
        old_key = self._shortcut_key
        self._shortcut_key = value
        tracker.mark_dirty(self, "_shortcut_key", old_key, value)

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        old_desc = self._description
        self._description = value
        tracker.mark_dirty(self, "_description", old_desc, value)

    @property
    def tags(self) -> list[str]:
        return self._tags.copy()

    @property
    def created_at(self) -> float:
        return self._created_at

    @property
    def modified_at(self) -> float:
        return self._modified_at

    def add_tag(self, tag: str) -> None:
        """Add a tag to the bookmark."""
        if tag not in self._tags:
            old_tags = self._tags.copy()
            self._tags.append(tag)
            tracker.mark_dirty(self, "_tags", old_tags, self._tags.copy())

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the bookmark."""
        if tag in self._tags:
            old_tags = self._tags.copy()
            self._tags.remove(tag)
            tracker.mark_dirty(self, "_tags", old_tags, self._tags.copy())
            return True
        return False

    @track_changes
    def update_from_camera(
        self,
        position: Vector3,
        rotation: Quaternion,
        settings: Optional[CameraSettings] = None
    ) -> None:
        """
        Update bookmark from current camera.

        Args:
            position: New position
            rotation: New rotation
            settings: Optional new settings
        """
        self._position = position
        self._rotation = rotation
        if settings:
            self._camera_settings = settings
        self._modified_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert bookmark to dictionary for serialization."""
        return {
            "id": self._id,
            "name": self._name,
            "position": {
                "x": self._position.x,
                "y": self._position.y,
                "z": self._position.z,
            },
            "rotation": {
                "x": self._rotation.x,
                "y": self._rotation.y,
                "z": self._rotation.z,
                "w": self._rotation.w,
            },
            "camera_settings": {
                "fov": self._camera_settings.fov,
                "near_plane": self._camera_settings.near_plane,
                "far_plane": self._camera_settings.far_plane,
                "orthographic": self._camera_settings.orthographic,
                "ortho_size": self._camera_settings.ortho_size,
            },
            "category_id": self._category_id,
            "icon": self._icon.name,
            "shortcut_key": self._shortcut_key,
            "description": self._description,
            "tags": self._tags,
            "created_at": self._created_at,
            "modified_at": self._modified_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CameraBookmark":
        """Create bookmark from dictionary."""
        bookmark = CameraBookmark(
            name=data["name"],
            position=Vector3(
                data["position"]["x"],
                data["position"]["y"],
                data["position"]["z"],
            ),
            rotation=Quaternion(
                data["rotation"]["x"],
                data["rotation"]["y"],
                data["rotation"]["z"],
                data["rotation"]["w"],
            ),
        )
        bookmark._id = data.get("id", bookmark._id)
        bookmark._category_id = data.get("category_id")
        bookmark._icon = BookmarkIcon[data.get("icon", "CAMERA")]
        bookmark._shortcut_key = data.get("shortcut_key")
        bookmark._description = data.get("description", "")
        bookmark._tags = data.get("tags", [])
        bookmark._created_at = data.get("created_at", time.time())
        bookmark._modified_at = data.get("modified_at", time.time())

        if "camera_settings" in data:
            cs = data["camera_settings"]
            bookmark._camera_settings = CameraSettings(
                fov=cs.get("fov", 60.0),
                near_plane=cs.get("near_plane", 0.1),
                far_plane=cs.get("far_plane", 1000.0),
                orthographic=cs.get("orthographic", False),
                ortho_size=cs.get("ortho_size", 10.0),
            )

        return bookmark


# =============================================================================
# Bookmark Manager
# =============================================================================

@editor
class BookmarkManager:
    """
    Central manager for camera bookmarks.

    Handles bookmark creation, navigation, and organization.
    """

    __slots__ = (
        "_bookmarks",
        "_categories",
        "_category_order",
        "_transition_settings",
        "_callbacks",
        "_history",
        "_history_index",
        "__weakref__",
    )

    MAX_HISTORY = 50

    def __init__(self):
        """Initialize bookmark manager."""
        self._bookmarks: dict[str, CameraBookmark] = {}
        self._categories: dict[str, BookmarkCategory] = {}
        self._category_order: list[str] = []
        self._transition_settings = TransitionSettings()
        self._callbacks: dict[str, list[Callable]] = {
            "on_bookmark_create": [],
            "on_bookmark_delete": [],
            "on_bookmark_update": [],
            "on_navigate": [],
            "on_category_change": [],
        }
        self._history: list[str] = []
        self._history_index = -1

        # Create default category
        default_cat = self.create_category("General", BookmarkIcon.BOOKMARK)
        self._category_order.insert(0, default_cat.id)

    @property
    def transition_settings(self) -> TransitionSettings:
        return self._transition_settings

    @transition_settings.setter
    def transition_settings(self, value: TransitionSettings) -> None:
        self._transition_settings = value

    @property
    def bookmark_count(self) -> int:
        return len(self._bookmarks)

    @property
    def category_count(self) -> int:
        return len(self._categories)

    def on(self, event: str, callback: Callable) -> None:
        """Register callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    # Category management
    @track_changes
    def create_category(
        self,
        name: str,
        icon: BookmarkIcon = BookmarkIcon.BOOKMARK
    ) -> BookmarkCategory:
        """
        Create a new category.

        Args:
            name: Category name
            icon: Category icon

        Returns:
            Created category
        """
        category = BookmarkCategory(name, icon)
        category.order = len(self._category_order)
        self._categories[category.id] = category
        self._category_order.append(category.id)

        for callback in self._callbacks["on_category_change"]:
            callback(category, "create")

        return category

    @track_changes
    def delete_category(self, category_id: str) -> bool:
        """
        Delete a category.

        Args:
            category_id: ID of category to delete

        Returns:
            True if deleted
        """
        if category_id not in self._categories:
            return False

        # Move bookmarks to first category or uncategorized
        default_id = self._category_order[0] if self._category_order else None
        for bookmark in self._bookmarks.values():
            if bookmark.category_id == category_id:
                bookmark.category_id = default_id if default_id != category_id else None

        category = self._categories.pop(category_id)
        self._category_order.remove(category_id)

        for callback in self._callbacks["on_category_change"]:
            callback(category, "delete")

        return True

    def get_category(self, category_id: str) -> Optional[BookmarkCategory]:
        """Get category by ID."""
        return self._categories.get(category_id)

    def get_all_categories(self) -> list[BookmarkCategory]:
        """Get all categories in order."""
        return [
            self._categories[cid]
            for cid in self._category_order
            if cid in self._categories
        ]

    # Bookmark management
    @track_changes
    def create_bookmark(
        self,
        name: str,
        position: Vector3,
        rotation: Quaternion,
        category_id: Optional[str] = None,
        settings: Optional[CameraSettings] = None
    ) -> CameraBookmark:
        """
        Create a new bookmark.

        Args:
            name: Bookmark name
            position: Camera position
            rotation: Camera rotation
            category_id: Optional category
            settings: Optional camera settings

        Returns:
            Created bookmark
        """
        bookmark = CameraBookmark(name, position, rotation)
        bookmark._category_id = category_id or (
            self._category_order[0] if self._category_order else None
        )
        if settings:
            bookmark._camera_settings = settings

        self._bookmarks[bookmark.id] = bookmark

        for callback in self._callbacks["on_bookmark_create"]:
            callback(bookmark)

        return bookmark

    @track_changes
    def delete_bookmark(self, bookmark_id: str) -> bool:
        """
        Delete a bookmark.

        Args:
            bookmark_id: ID of bookmark to delete

        Returns:
            True if deleted
        """
        if bookmark_id not in self._bookmarks:
            return False

        bookmark = self._bookmarks.pop(bookmark_id)

        # Remove from history
        self._history = [h for h in self._history if h != bookmark_id]
        if self._history_index >= len(self._history):
            self._history_index = len(self._history) - 1

        for callback in self._callbacks["on_bookmark_delete"]:
            callback(bookmark)

        return True

    def get_bookmark(self, bookmark_id: str) -> Optional[CameraBookmark]:
        """Get bookmark by ID."""
        return self._bookmarks.get(bookmark_id)

    def get_bookmark_by_name(self, name: str) -> Optional[CameraBookmark]:
        """Get bookmark by name."""
        for bookmark in self._bookmarks.values():
            if bookmark.name == name:
                return bookmark
        return None

    def get_bookmark_by_shortcut(self, shortcut: str) -> Optional[CameraBookmark]:
        """Get bookmark by shortcut key."""
        for bookmark in self._bookmarks.values():
            if bookmark.shortcut_key == shortcut:
                return bookmark
        return None

    def get_all_bookmarks(self) -> list[CameraBookmark]:
        """Get all bookmarks."""
        return list(self._bookmarks.values())

    def get_bookmarks_in_category(self, category_id: str) -> list[CameraBookmark]:
        """Get all bookmarks in a category."""
        return [
            b for b in self._bookmarks.values()
            if b.category_id == category_id
        ]

    def get_uncategorized_bookmarks(self) -> list[CameraBookmark]:
        """Get bookmarks without a category."""
        return [b for b in self._bookmarks.values() if b.category_id is None]

    def find_bookmarks_by_tag(self, tag: str) -> list[CameraBookmark]:
        """Find bookmarks with a specific tag."""
        return [b for b in self._bookmarks.values() if tag in b.tags]

    def search_bookmarks(self, query: str) -> list[CameraBookmark]:
        """Search bookmarks by name or description."""
        query_lower = query.lower()
        return [
            b for b in self._bookmarks.values()
            if query_lower in b.name.lower() or query_lower in b.description.lower()
        ]

    # Navigation
    def navigate_to(
        self,
        bookmark_id: str,
        instant: bool = False
    ) -> Optional[tuple[Vector3, Quaternion, CameraSettings, TransitionSettings]]:
        """
        Navigate to a bookmark.

        Args:
            bookmark_id: ID of bookmark to navigate to
            instant: Skip transition if True

        Returns:
            Tuple of (position, rotation, settings, transition) or None
        """
        bookmark = self._bookmarks.get(bookmark_id)
        if not bookmark:
            return None

        # Add to history
        if self._history_index < len(self._history) - 1:
            # Truncate forward history
            self._history = self._history[:self._history_index + 1]

        self._history.append(bookmark_id)
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)
        self._history_index = len(self._history) - 1

        transition = TransitionSettings(
            transition_type=TransitionType.INSTANT if instant else self._transition_settings.transition_type,
            duration=0 if instant else self._transition_settings.duration,
            animate_fov=self._transition_settings.animate_fov,
        )

        for callback in self._callbacks["on_navigate"]:
            callback(bookmark)

        return (
            bookmark.position,
            bookmark.rotation,
            bookmark.camera_settings,
            transition,
        )

    def navigate_back(self) -> Optional[tuple[Vector3, Quaternion, CameraSettings, TransitionSettings]]:
        """Navigate to previous bookmark in history."""
        if self._history_index <= 0:
            return None

        self._history_index -= 1
        bookmark_id = self._history[self._history_index]
        return self.navigate_to_without_history(bookmark_id)

    def navigate_forward(self) -> Optional[tuple[Vector3, Quaternion, CameraSettings, TransitionSettings]]:
        """Navigate to next bookmark in history."""
        if self._history_index >= len(self._history) - 1:
            return None

        self._history_index += 1
        bookmark_id = self._history[self._history_index]
        return self.navigate_to_without_history(bookmark_id)

    def navigate_to_without_history(
        self,
        bookmark_id: str
    ) -> Optional[tuple[Vector3, Quaternion, CameraSettings, TransitionSettings]]:
        """Navigate without adding to history."""
        bookmark = self._bookmarks.get(bookmark_id)
        if not bookmark:
            return None

        for callback in self._callbacks["on_navigate"]:
            callback(bookmark)

        return (
            bookmark.position,
            bookmark.rotation,
            bookmark.camera_settings,
            self._transition_settings,
        )

    def can_go_back(self) -> bool:
        """Check if can navigate back."""
        return self._history_index > 0

    def can_go_forward(self) -> bool:
        """Check if can navigate forward."""
        return self._history_index < len(self._history) - 1

    # Import/Export
    def export_bookmarks(self) -> dict[str, Any]:
        """Export all bookmarks and categories to dictionary."""
        return {
            "version": 1,
            "categories": [
                {
                    "id": cat.id,
                    "name": cat.name,
                    "icon": cat.icon.name,
                    "color": cat.color,
                    "order": cat.order,
                }
                for cat in self.get_all_categories()
            ],
            "bookmarks": [b.to_dict() for b in self._bookmarks.values()],
        }

    @track_changes
    def import_bookmarks(
        self,
        data: dict[str, Any],
        merge: bool = True
    ) -> tuple[int, int]:
        """
        Import bookmarks from dictionary.

        Args:
            data: Exported bookmark data
            merge: Merge with existing if True, replace if False

        Returns:
            Tuple of (categories_imported, bookmarks_imported)
        """
        if not merge:
            self._bookmarks.clear()
            self._categories.clear()
            self._category_order.clear()

        cat_count = 0
        bm_count = 0

        # Import categories
        for cat_data in data.get("categories", []):
            if merge and cat_data["id"] in self._categories:
                continue
            category = BookmarkCategory(cat_data["name"])
            category._id = cat_data["id"]
            category._icon = BookmarkIcon[cat_data.get("icon", "BOOKMARK")]
            category._color = tuple(cat_data.get("color", (0.7, 0.7, 0.7)))
            category._order = cat_data.get("order", 0)
            self._categories[category.id] = category
            if category.id not in self._category_order:
                self._category_order.append(category.id)
            cat_count += 1

        # Sort category order
        self._category_order.sort(
            key=lambda cid: self._categories[cid].order if cid in self._categories else 999
        )

        # Import bookmarks
        for bm_data in data.get("bookmarks", []):
            if merge and bm_data["id"] in self._bookmarks:
                continue
            bookmark = CameraBookmark.from_dict(bm_data)
            self._bookmarks[bookmark.id] = bookmark
            bm_count += 1

        return cat_count, bm_count

    def get_statistics(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "total_bookmarks": len(self._bookmarks),
            "total_categories": len(self._categories),
            "bookmarks_with_shortcuts": sum(
                1 for b in self._bookmarks.values() if b.shortcut_key
            ),
            "history_size": len(self._history),
            "history_position": self._history_index,
        }


__all__ = [
    "CameraBookmark",
    "BookmarkManager",
    "BookmarkCategory",
    "CameraSettings",
    "TransitionSettings",
    "TransitionType",
    "BookmarkIcon",
]
