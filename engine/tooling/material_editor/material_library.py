"""Material library - Material library with categories, search, and favorites."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid
import json
import os
from datetime import datetime

from .material_graph import MaterialGraph
from .material_instances import MaterialDefinition, MaterialInstance, MaterialInstanceManager


class LibraryItemType(Enum):
    """Type of library item."""
    MATERIAL = auto()
    INSTANCE = auto()
    FOLDER = auto()
    NODE_PRESET = auto()


@dataclass
class LibraryMetadata:
    """Metadata for a library item."""
    created_time: str = ""
    modified_time: str = ""
    author: str = ""
    description: str = ""
    version: str = "1.0"
    thumbnail_path: str = ""
    preview_image: str = ""
    file_size: int = 0


@dataclass
class LibraryItem:
    """Item in the material library."""
    id: str
    name: str
    item_type: LibraryItemType
    category: str = "Uncategorized"
    tags: List[str] = field(default_factory=list)
    metadata: LibraryMetadata = field(default_factory=LibraryMetadata)
    favorite: bool = False
    rating: int = 0  # 0-5 stars
    usage_count: int = 0
    parent_folder_id: Optional[str] = None
    data_path: str = ""  # Path to actual data file

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "item_type": self.item_type.name,
            "category": self.category,
            "tags": self.tags,
            "metadata": {
                "created_time": self.metadata.created_time,
                "modified_time": self.metadata.modified_time,
                "author": self.metadata.author,
                "description": self.metadata.description,
                "version": self.metadata.version,
                "thumbnail_path": self.metadata.thumbnail_path,
                "preview_image": self.metadata.preview_image,
                "file_size": self.metadata.file_size,
            },
            "favorite": self.favorite,
            "rating": self.rating,
            "usage_count": self.usage_count,
            "parent_folder_id": self.parent_folder_id,
            "data_path": self.data_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LibraryItem':
        metadata = LibraryMetadata(
            created_time=data.get("metadata", {}).get("created_time", ""),
            modified_time=data.get("metadata", {}).get("modified_time", ""),
            author=data.get("metadata", {}).get("author", ""),
            description=data.get("metadata", {}).get("description", ""),
            version=data.get("metadata", {}).get("version", "1.0"),
            thumbnail_path=data.get("metadata", {}).get("thumbnail_path", ""),
            preview_image=data.get("metadata", {}).get("preview_image", ""),
            file_size=data.get("metadata", {}).get("file_size", 0),
        )
        return cls(
            id=data["id"],
            name=data["name"],
            item_type=LibraryItemType[data["item_type"]],
            category=data.get("category", "Uncategorized"),
            tags=data.get("tags", []),
            metadata=metadata,
            favorite=data.get("favorite", False),
            rating=data.get("rating", 0),
            usage_count=data.get("usage_count", 0),
            parent_folder_id=data.get("parent_folder_id"),
            data_path=data.get("data_path", "")
        )


class SearchFilter:
    """Filter for library search."""

    def __init__(self):
        self.query: str = ""
        self.categories: List[str] = []
        self.tags: List[str] = []
        self.item_types: List[LibraryItemType] = []
        self.favorites_only: bool = False
        self.min_rating: int = 0
        self.author: str = ""
        self.date_from: Optional[str] = None
        self.date_to: Optional[str] = None

    def matches(self, item: LibraryItem) -> bool:
        """Check if item matches filter criteria."""
        # Query match (name, description, tags)
        if self.query:
            query_lower = self.query.lower()
            if not (
                query_lower in item.name.lower() or
                query_lower in item.metadata.description.lower() or
                any(query_lower in tag.lower() for tag in item.tags)
            ):
                return False

        # Category filter
        if self.categories and item.category not in self.categories:
            return False

        # Tag filter (any tag matches)
        if self.tags and not any(tag in item.tags for tag in self.tags):
            return False

        # Item type filter
        if self.item_types and item.item_type not in self.item_types:
            return False

        # Favorites filter
        if self.favorites_only and not item.favorite:
            return False

        # Rating filter
        if self.min_rating > 0 and item.rating < self.min_rating:
            return False

        # Author filter
        if self.author and self.author.lower() not in item.metadata.author.lower():
            return False

        return True


class SortOrder(Enum):
    """Sort order for library items."""
    NAME_ASC = auto()
    NAME_DESC = auto()
    DATE_CREATED_ASC = auto()
    DATE_CREATED_DESC = auto()
    DATE_MODIFIED_ASC = auto()
    DATE_MODIFIED_DESC = auto()
    RATING_ASC = auto()
    RATING_DESC = auto()
    USAGE_ASC = auto()
    USAGE_DESC = auto()


class MaterialLibrary:
    """
    Material library with categories, search, and favorites.

    Provides organization, browsing, and management of materials,
    instances, and related assets.
    """

    DEFAULT_CATEGORIES = [
        "Metals",
        "Woods",
        "Stones",
        "Fabrics",
        "Plastics",
        "Glass",
        "Organics",
        "Effects",
        "Terrain",
        "Water",
        "Stylized",
        "Uncategorized"
    ]

    def __init__(self, library_path: str = ""):
        self._path = library_path
        self._items: Dict[str, LibraryItem] = {}
        self._categories: Set[str] = set(self.DEFAULT_CATEGORIES)
        self._tags: Set[str] = set()
        self._recent_items: List[str] = []
        self._max_recent = 20

        # Callbacks
        self._on_item_added: List[Callable[[LibraryItem], None]] = []
        self._on_item_removed: List[Callable[[str], None]] = []
        self._on_item_updated: List[Callable[[LibraryItem], None]] = []

    @property
    def path(self) -> str:
        return self._path

    @property
    def item_count(self) -> int:
        return len(self._items)

    @property
    def categories(self) -> List[str]:
        return sorted(list(self._categories))

    @property
    def tags(self) -> List[str]:
        return sorted(list(self._tags))

    @property
    def recent_items(self) -> List[LibraryItem]:
        return [self._items[id] for id in self._recent_items if id in self._items]

    @property
    def favorites(self) -> List[LibraryItem]:
        return [item for item in self._items.values() if item.favorite]

    # ========================================================================
    # Item Management
    # ========================================================================

    def add_item(self, item: LibraryItem) -> bool:
        """Add an item to the library."""
        if item.id in self._items:
            return False

        # Set creation time if not set
        if not item.metadata.created_time:
            item.metadata.created_time = datetime.now().isoformat()
        item.metadata.modified_time = datetime.now().isoformat()

        self._items[item.id] = item

        # Track category and tags
        self._categories.add(item.category)
        self._tags.update(item.tags)

        for callback in self._on_item_added:
            callback(item)

        return True

    def remove_item(self, id: str) -> bool:
        """Remove an item from the library."""
        if id not in self._items:
            return False

        del self._items[id]

        # Remove from recent
        if id in self._recent_items:
            self._recent_items.remove(id)

        for callback in self._on_item_removed:
            callback(id)

        return True

    def get_item(self, id: str) -> Optional[LibraryItem]:
        """Get an item by ID."""
        item = self._items.get(id)
        if item:
            self._add_to_recent(id)
        return item

    def get_item_by_name(self, name: str) -> Optional[LibraryItem]:
        """Get an item by name."""
        for item in self._items.values():
            if item.name == name:
                self._add_to_recent(item.id)
                return item
        return None

    def update_item(self, item: LibraryItem) -> bool:
        """Update an existing item."""
        if item.id not in self._items:
            return False

        item.metadata.modified_time = datetime.now().isoformat()
        self._items[item.id] = item

        # Update category and tags
        self._categories.add(item.category)
        self._tags.update(item.tags)

        for callback in self._on_item_updated:
            callback(item)

        return True

    def _add_to_recent(self, id: str) -> None:
        """Add item to recent list."""
        if id in self._recent_items:
            self._recent_items.remove(id)
        self._recent_items.insert(0, id)
        if len(self._recent_items) > self._max_recent:
            self._recent_items = self._recent_items[:self._max_recent]

    # ========================================================================
    # Material/Instance Creation
    # ========================================================================

    def add_material(
        self,
        name: str,
        graph: MaterialGraph,
        category: str = "Uncategorized",
        tags: List[str] = None,
        description: str = "",
        author: str = ""
    ) -> LibraryItem:
        """Add a material graph to the library."""
        item = LibraryItem(
            id=str(uuid.uuid4()),
            name=name,
            item_type=LibraryItemType.MATERIAL,
            category=category,
            tags=tags or [],
            metadata=LibraryMetadata(
                description=description,
                author=author
            )
        )
        self.add_item(item)

        # Store graph data
        if self._path:
            data_path = os.path.join(self._path, f"{item.id}.material")
            item.data_path = data_path
            # In a real implementation, we'd save the graph here

        return item

    def add_material_instance(
        self,
        name: str,
        instance: MaterialInstance,
        category: str = "Uncategorized",
        tags: List[str] = None
    ) -> LibraryItem:
        """Add a material instance to the library."""
        item = LibraryItem(
            id=str(uuid.uuid4()),
            name=name,
            item_type=LibraryItemType.INSTANCE,
            category=category,
            tags=tags or []
        )
        self.add_item(item)
        return item

    # ========================================================================
    # Categories
    # ========================================================================

    def add_category(self, category: str) -> None:
        """Add a category."""
        self._categories.add(category)

    def remove_category(self, category: str) -> bool:
        """Remove a category (move items to Uncategorized)."""
        if category in self.DEFAULT_CATEGORIES:
            return False

        # Move items to Uncategorized
        for item in self._items.values():
            if item.category == category:
                item.category = "Uncategorized"

        self._categories.discard(category)
        return True

    def rename_category(self, old_name: str, new_name: str) -> bool:
        """Rename a category."""
        if old_name not in self._categories or old_name in self.DEFAULT_CATEGORIES:
            return False

        for item in self._items.values():
            if item.category == old_name:
                item.category = new_name

        self._categories.discard(old_name)
        self._categories.add(new_name)
        return True

    def get_items_in_category(self, category: str) -> List[LibraryItem]:
        """Get all items in a category."""
        return [item for item in self._items.values() if item.category == category]

    # ========================================================================
    # Tags
    # ========================================================================

    def add_tag_to_item(self, item_id: str, tag: str) -> bool:
        """Add a tag to an item."""
        item = self._items.get(item_id)
        if item is None:
            return False

        if tag not in item.tags:
            item.tags.append(tag)
            self._tags.add(tag)
        return True

    def remove_tag_from_item(self, item_id: str, tag: str) -> bool:
        """Remove a tag from an item."""
        item = self._items.get(item_id)
        if item is None or tag not in item.tags:
            return False

        item.tags.remove(tag)
        return True

    def get_items_with_tag(self, tag: str) -> List[LibraryItem]:
        """Get all items with a tag."""
        return [item for item in self._items.values() if tag in item.tags]

    # ========================================================================
    # Favorites
    # ========================================================================

    def set_favorite(self, item_id: str, favorite: bool) -> bool:
        """Set favorite status for an item."""
        item = self._items.get(item_id)
        if item is None:
            return False

        item.favorite = favorite
        return True

    def toggle_favorite(self, item_id: str) -> bool:
        """Toggle favorite status for an item."""
        item = self._items.get(item_id)
        if item is None:
            return False

        item.favorite = not item.favorite
        return True

    # ========================================================================
    # Rating
    # ========================================================================

    def set_rating(self, item_id: str, rating: int) -> bool:
        """Set rating for an item (0-5)."""
        item = self._items.get(item_id)
        if item is None:
            return False

        item.rating = max(0, min(5, rating))
        return True

    # ========================================================================
    # Search
    # ========================================================================

    def search(
        self,
        filter: SearchFilter,
        sort_order: SortOrder = SortOrder.NAME_ASC
    ) -> List[LibraryItem]:
        """Search the library with filter and sort."""
        results = [item for item in self._items.values() if filter.matches(item)]
        return self._sort_items(results, sort_order)

    def quick_search(self, query: str) -> List[LibraryItem]:
        """Quick search by name/description/tags."""
        filter = SearchFilter()
        filter.query = query
        return self.search(filter)

    def get_all_items(
        self,
        item_type: Optional[LibraryItemType] = None,
        sort_order: SortOrder = SortOrder.NAME_ASC
    ) -> List[LibraryItem]:
        """Get all items, optionally filtered by type."""
        if item_type:
            items = [item for item in self._items.values() if item.item_type == item_type]
        else:
            items = list(self._items.values())
        return self._sort_items(items, sort_order)

    def _sort_items(self, items: List[LibraryItem], sort_order: SortOrder) -> List[LibraryItem]:
        """Sort items by specified order."""
        if sort_order == SortOrder.NAME_ASC:
            return sorted(items, key=lambda x: x.name.lower())
        elif sort_order == SortOrder.NAME_DESC:
            return sorted(items, key=lambda x: x.name.lower(), reverse=True)
        elif sort_order == SortOrder.DATE_CREATED_ASC:
            return sorted(items, key=lambda x: x.metadata.created_time)
        elif sort_order == SortOrder.DATE_CREATED_DESC:
            return sorted(items, key=lambda x: x.metadata.created_time, reverse=True)
        elif sort_order == SortOrder.DATE_MODIFIED_ASC:
            return sorted(items, key=lambda x: x.metadata.modified_time)
        elif sort_order == SortOrder.DATE_MODIFIED_DESC:
            return sorted(items, key=lambda x: x.metadata.modified_time, reverse=True)
        elif sort_order == SortOrder.RATING_ASC:
            return sorted(items, key=lambda x: x.rating)
        elif sort_order == SortOrder.RATING_DESC:
            return sorted(items, key=lambda x: x.rating, reverse=True)
        elif sort_order == SortOrder.USAGE_ASC:
            return sorted(items, key=lambda x: x.usage_count)
        elif sort_order == SortOrder.USAGE_DESC:
            return sorted(items, key=lambda x: x.usage_count, reverse=True)
        return items

    # ========================================================================
    # Usage Tracking
    # ========================================================================

    def record_usage(self, item_id: str) -> None:
        """Record that an item was used."""
        item = self._items.get(item_id)
        if item:
            item.usage_count += 1
            self._add_to_recent(item_id)

    def get_most_used(self, limit: int = 10) -> List[LibraryItem]:
        """Get most frequently used items."""
        items = sorted(self._items.values(), key=lambda x: x.usage_count, reverse=True)
        return items[:limit]

    # ========================================================================
    # Folder Structure
    # ========================================================================

    def create_folder(self, name: str, parent_id: Optional[str] = None) -> LibraryItem:
        """Create a folder in the library."""
        folder = LibraryItem(
            id=str(uuid.uuid4()),
            name=name,
            item_type=LibraryItemType.FOLDER,
            parent_folder_id=parent_id
        )
        self.add_item(folder)
        return folder

    def get_folder_contents(self, folder_id: Optional[str] = None) -> List[LibraryItem]:
        """Get contents of a folder (None for root)."""
        return [item for item in self._items.values() if item.parent_folder_id == folder_id]

    def move_to_folder(self, item_id: str, folder_id: Optional[str]) -> bool:
        """Move an item to a folder."""
        item = self._items.get(item_id)
        if item is None:
            return False

        # Verify folder exists (if not moving to root)
        if folder_id and folder_id not in self._items:
            return False

        item.parent_folder_id = folder_id
        return True

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize library to dictionary."""
        return {
            "path": self._path,
            "items": {id: item.to_dict() for id, item in self._items.items()},
            "categories": list(self._categories),
            "tags": list(self._tags),
            "recent_items": self._recent_items
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaterialLibrary':
        """Deserialize from dictionary."""
        library = cls(data.get("path", ""))
        library._categories = set(data.get("categories", cls.DEFAULT_CATEGORIES))
        library._tags = set(data.get("tags", []))
        library._recent_items = data.get("recent_items", [])

        for id, item_data in data.get("items", {}).items():
            item = LibraryItem.from_dict(item_data)
            library._items[id] = item

        return library

    def save(self, path: Optional[str] = None) -> bool:
        """Save library to file."""
        save_path = path or self._path
        if not save_path:
            return False

        try:
            with open(os.path.join(save_path, "library.json"), "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception:
            return False

    def load(self, path: Optional[str] = None) -> bool:
        """Load library from file."""
        load_path = path or self._path
        if not load_path:
            return False

        try:
            with open(os.path.join(load_path, "library.json"), "r") as f:
                data = json.load(f)
            loaded = MaterialLibrary.from_dict(data)
            self._items = loaded._items
            self._categories = loaded._categories
            self._tags = loaded._tags
            self._recent_items = loaded._recent_items
            self._path = load_path
            return True
        except Exception:
            return False

    # ========================================================================
    # Callbacks
    # ========================================================================

    def on_item_added(self, callback: Callable[[LibraryItem], None]) -> None:
        """Register callback for item added events."""
        self._on_item_added.append(callback)

    def on_item_removed(self, callback: Callable[[str], None]) -> None:
        """Register callback for item removed events."""
        self._on_item_removed.append(callback)

    def on_item_updated(self, callback: Callable[[LibraryItem], None]) -> None:
        """Register callback for item updated events."""
        self._on_item_updated.append(callback)

    # ========================================================================
    # Utilities
    # ========================================================================

    def clear(self) -> None:
        """Clear all items from library."""
        self._items.clear()
        self._recent_items.clear()
        self._categories = set(self.DEFAULT_CATEGORIES)
        self._tags.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        by_type = {}
        by_category = {}

        for item in self._items.values():
            type_name = item.item_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1
            by_category[item.category] = by_category.get(item.category, 0) + 1

        return {
            "total_items": len(self._items),
            "by_type": by_type,
            "by_category": by_category,
            "total_categories": len(self._categories),
            "total_tags": len(self._tags),
            "favorites_count": len(self.favorites)
        }
