"""
AssetCollection - Asset collections for organization.

Provides collection-based asset organization:
- Manual collections (user-curated)
- Smart collections (query-based)
- Nested collections (hierarchies)
- Collection persistence
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Protocol, Union

from trinity.decorators.dev import editor


class CollectionType(Enum):
    """Types of asset collections."""

    MANUAL = auto()  # User-curated collection
    SMART = auto()  # Query-based dynamic collection
    FOLDER = auto()  # Folder-based collection


@dataclass
class CollectionQuery:
    """Query definition for smart collections.

    Attributes:
        filters: List of filter conditions
        sort_by: Sort field
        sort_ascending: Sort direction
        limit: Maximum results
    """

    filters: list[dict[str, Any]] = field(default_factory=list)
    sort_by: str = "name"
    sort_ascending: bool = True
    limit: Optional[int] = None

    def add_filter(
        self,
        field: str,
        operator: str,
        value: Any,
    ) -> "CollectionQuery":
        """Add a filter condition.

        Args:
            field: Field to filter on
            operator: Operator (eq, ne, gt, lt, contains, in, etc)
            value: Value to compare

        Returns:
            Self for chaining
        """
        self.filters.append({
            "field": field,
            "operator": operator,
            "value": value,
        })
        return self

    def matches(self, item: dict[str, Any]) -> bool:
        """Check if an item matches the query.

        Args:
            item: Item to check (dict with field values)

        Returns:
            True if matches all filters
        """
        for f in self.filters:
            field_value = item.get(f["field"])
            op = f["operator"]
            value = f["value"]

            if op == "eq" and field_value != value:
                return False
            elif op == "ne" and field_value == value:
                return False
            elif op == "gt" and not (field_value is not None and field_value > value):
                return False
            elif op == "lt" and not (field_value is not None and field_value < value):
                return False
            elif op == "gte" and not (field_value is not None and field_value >= value):
                return False
            elif op == "lte" and not (field_value is not None and field_value <= value):
                return False
            elif op == "contains":
                if not isinstance(field_value, str) or value not in field_value:
                    return False
            elif op == "in":
                if field_value not in value:
                    return False
            elif op == "startswith":
                if not isinstance(field_value, str) or not field_value.startswith(value):
                    return False
            elif op == "endswith":
                if not isinstance(field_value, str) or not field_value.endswith(value):
                    return False
            elif op == "regex":
                import re
                if not isinstance(field_value, str) or not re.search(value, field_value):
                    return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filters": self.filters,
            "sort_by": self.sort_by,
            "sort_ascending": self.sort_ascending,
            "limit": self.limit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollectionQuery":
        """Create from dictionary."""
        return cls(
            filters=data.get("filters", []),
            sort_by=data.get("sort_by", "name"),
            sort_ascending=data.get("sort_ascending", True),
            limit=data.get("limit"),
        )


@dataclass
class AssetCollection:
    """A collection of assets.

    Attributes:
        id: Unique collection identifier
        name: Collection name
        description: Collection description
        collection_type: Type of collection
        color: Display color (hex)
        icon: Icon identifier
        assets: Set of asset paths (for manual collections)
        query: Query for smart collections
        parent_id: Parent collection ID (for nesting)
        children_ids: Child collection IDs
        created_at: Creation timestamp
        modified_at: Last modification timestamp
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    collection_type: CollectionType = CollectionType.MANUAL
    color: str = "#4a90d9"
    icon: str = "folder"
    assets: set[Path] = field(default_factory=set)
    query: Optional[CollectionQuery] = None
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_smart(self) -> bool:
        """Check if this is a smart collection."""
        return self.collection_type == CollectionType.SMART

    @property
    def is_nested(self) -> bool:
        """Check if this collection is nested."""
        return self.parent_id is not None

    @property
    def has_children(self) -> bool:
        """Check if this collection has children."""
        return len(self.children_ids) > 0

    @property
    def asset_count(self) -> int:
        """Get number of assets in collection."""
        return len(self.assets)

    def add_asset(self, path: Union[str, Path]) -> bool:
        """Add an asset to the collection.

        Args:
            path: Path to the asset

        Returns:
            True if added (manual collection only)
        """
        if self.is_smart:
            return False

        path = Path(path)
        if path not in self.assets:
            self.assets.add(path)
            self.modified_at = time.time()
            return True
        return False

    def remove_asset(self, path: Union[str, Path]) -> bool:
        """Remove an asset from the collection.

        Args:
            path: Path to the asset

        Returns:
            True if removed
        """
        if self.is_smart:
            return False

        path = Path(path)
        if path in self.assets:
            self.assets.discard(path)
            self.modified_at = time.time()
            return True
        return False

    def contains(self, path: Union[str, Path]) -> bool:
        """Check if collection contains an asset.

        Args:
            path: Path to check

        Returns:
            True if asset is in collection
        """
        return Path(path) in self.assets

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "collection_type": self.collection_type.name,
            "color": self.color,
            "icon": self.icon,
            "assets": [str(p) for p in self.assets],
            "query": self.query.to_dict() if self.query else None,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetCollection":
        """Create from dictionary."""
        query_data = data.get("query")
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            collection_type=CollectionType[data.get("collection_type", "MANUAL")],
            color=data.get("color", "#4a90d9"),
            icon=data.get("icon", "folder"),
            assets={Path(p) for p in data.get("assets", [])},
            query=CollectionQuery.from_dict(query_data) if query_data else None,
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            created_at=data.get("created_at", time.time()),
            modified_at=data.get("modified_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SmartCollection(AssetCollection):
    """A smart collection with query-based membership.

    Smart collections automatically include assets matching their query.
    """

    def __init__(
        self,
        name: str,
        query: CollectionQuery,
        **kwargs: Any,
    ) -> None:
        """Create a smart collection.

        Args:
            name: Collection name
            query: Query defining membership
            **kwargs: Additional AssetCollection arguments
        """
        super().__init__(
            name=name,
            query=query,
            collection_type=CollectionType.SMART,
            **kwargs,
        )

    def evaluate(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Evaluate the query against items.

        Args:
            items: List of items to filter

        Returns:
            Matching items
        """
        if not self.query:
            return items

        # Filter
        matching = [item for item in items if self.query.matches(item)]

        # Sort
        if self.query.sort_by:
            matching.sort(
                key=lambda x: x.get(self.query.sort_by, ""),
                reverse=not self.query.sort_ascending,
            )

        # Limit
        if self.query.limit:
            matching = matching[:self.query.limit]

        return matching


class AssetProvider(Protocol):
    """Protocol for asset data providers."""

    def get_all_assets(self) -> list[dict[str, Any]]:
        """Get all assets as dicts with metadata."""
        ...

    def get_asset_data(self, path: Path) -> Optional[dict[str, Any]]:
        """Get data for a single asset."""
        ...


@editor(category="Assets")
class CollectionManager:
    """Manages asset collections.

    Provides:
    - Collection CRUD operations
    - Nested collections
    - Smart collection evaluation
    - Persistence
    - Change notifications

    Attributes:
        storage_path: Path for collection storage
        asset_provider: Provider for asset data
        _collections: All collections indexed by ID
        _root_collections: IDs of root-level collections
        _change_listeners: Change notification callbacks
    """

    def __init__(
        self,
        storage_path: Union[str, Path],
        asset_provider: Optional[AssetProvider] = None,
    ) -> None:
        """Initialize the collection manager.

        Args:
            storage_path: Path for persistent storage
            asset_provider: Provider for asset data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.asset_provider = asset_provider

        self._collections: dict[str, AssetCollection] = {}
        self._root_collections: list[str] = []
        self._change_listeners: list[Callable[[AssetCollection, str], None]] = []

        # Load existing collections
        self._load_collections()

    def create_collection(
        self,
        name: str,
        collection_type: CollectionType = CollectionType.MANUAL,
        parent_id: Optional[str] = None,
        query: Optional[CollectionQuery] = None,
        **kwargs: Any,
    ) -> AssetCollection:
        """Create a new collection.

        Args:
            name: Collection name
            collection_type: Type of collection
            parent_id: Parent collection ID
            query: Query for smart collections
            **kwargs: Additional collection arguments

        Returns:
            Created collection
        """
        collection = AssetCollection(
            name=name,
            collection_type=collection_type,
            query=query,
            parent_id=parent_id,
            **kwargs,
        )

        self._collections[collection.id] = collection

        # Update parent if nested
        if parent_id:
            parent = self._collections.get(parent_id)
            if parent:
                parent.children_ids.append(collection.id)
        else:
            self._root_collections.append(collection.id)

        self._save_collections()
        self._notify_change(collection, "created")

        return collection

    def create_smart_collection(
        self,
        name: str,
        query: CollectionQuery,
        parent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SmartCollection:
        """Create a smart collection.

        Args:
            name: Collection name
            query: Query defining membership
            parent_id: Parent collection ID
            **kwargs: Additional collection arguments

        Returns:
            Created smart collection
        """
        collection = SmartCollection(name=name, query=query, **kwargs)
        collection.parent_id = parent_id

        self._collections[collection.id] = collection

        if parent_id:
            parent = self._collections.get(parent_id)
            if parent:
                parent.children_ids.append(collection.id)
        else:
            self._root_collections.append(collection.id)

        self._save_collections()
        self._notify_change(collection, "created")

        return collection

    def get_collection(self, collection_id: str) -> Optional[AssetCollection]:
        """Get a collection by ID."""
        return self._collections.get(collection_id)

    def get_collection_by_name(self, name: str) -> Optional[AssetCollection]:
        """Get a collection by name."""
        for collection in self._collections.values():
            if collection.name == name:
                return collection
        return None

    def update_collection(self, collection: AssetCollection) -> bool:
        """Update a collection.

        Args:
            collection: Collection with updates

        Returns:
            True if updated
        """
        if collection.id not in self._collections:
            return False

        collection.modified_at = time.time()
        self._collections[collection.id] = collection
        self._save_collections()
        self._notify_change(collection, "updated")

        return True

    def delete_collection(self, collection_id: str, recursive: bool = False) -> bool:
        """Delete a collection.

        Args:
            collection_id: ID of collection to delete
            recursive: Delete children recursively

        Returns:
            True if deleted
        """
        collection = self._collections.get(collection_id)
        if not collection:
            return False

        # Handle children
        if collection.has_children:
            if recursive:
                for child_id in collection.children_ids.copy():
                    self.delete_collection(child_id, recursive=True)
            else:
                # Move children to root
                for child_id in collection.children_ids:
                    child = self._collections.get(child_id)
                    if child:
                        child.parent_id = None
                        self._root_collections.append(child_id)

        # Remove from parent
        if collection.parent_id:
            parent = self._collections.get(collection.parent_id)
            if parent:
                parent.children_ids.remove(collection_id)
        else:
            if collection_id in self._root_collections:
                self._root_collections.remove(collection_id)

        # Delete
        del self._collections[collection_id]
        self._save_collections()
        self._notify_change(collection, "deleted")

        return True

    def get_root_collections(self) -> list[AssetCollection]:
        """Get all root-level collections."""
        return [
            self._collections[cid]
            for cid in self._root_collections
            if cid in self._collections
        ]

    def get_all_collections(self) -> list[AssetCollection]:
        """Get all collections."""
        return list(self._collections.values())

    def get_children(self, collection_id: str) -> list[AssetCollection]:
        """Get child collections of a collection."""
        collection = self._collections.get(collection_id)
        if not collection:
            return []

        return [
            self._collections[cid]
            for cid in collection.children_ids
            if cid in self._collections
        ]

    def get_ancestors(self, collection_id: str) -> list[AssetCollection]:
        """Get all ancestor collections."""
        ancestors = []
        current = self._collections.get(collection_id)

        while current and current.parent_id:
            parent = self._collections.get(current.parent_id)
            if parent:
                ancestors.insert(0, parent)
                current = parent
            else:
                break

        return ancestors

    def move_collection(
        self,
        collection_id: str,
        new_parent_id: Optional[str],
    ) -> bool:
        """Move a collection to a new parent.

        Args:
            collection_id: Collection to move
            new_parent_id: New parent ID (None for root)

        Returns:
            True if moved
        """
        collection = self._collections.get(collection_id)
        if not collection:
            return False

        # Prevent circular references
        if new_parent_id:
            ancestors = self.get_ancestors(new_parent_id)
            if collection in ancestors or collection.id == new_parent_id:
                return False

        # Remove from old parent
        if collection.parent_id:
            old_parent = self._collections.get(collection.parent_id)
            if old_parent:
                old_parent.children_ids.remove(collection_id)
        else:
            if collection_id in self._root_collections:
                self._root_collections.remove(collection_id)

        # Add to new parent
        if new_parent_id:
            new_parent = self._collections.get(new_parent_id)
            if new_parent:
                new_parent.children_ids.append(collection_id)
        else:
            self._root_collections.append(collection_id)

        collection.parent_id = new_parent_id
        collection.modified_at = time.time()

        self._save_collections()
        self._notify_change(collection, "moved")

        return True

    def add_asset_to_collection(
        self,
        collection_id: str,
        asset_path: Union[str, Path],
    ) -> bool:
        """Add an asset to a collection.

        Args:
            collection_id: Target collection ID
            asset_path: Asset to add

        Returns:
            True if added
        """
        collection = self._collections.get(collection_id)
        if not collection:
            return False

        if collection.add_asset(asset_path):
            self._save_collections()
            self._notify_change(collection, "asset_added")
            return True

        return False

    def remove_asset_from_collection(
        self,
        collection_id: str,
        asset_path: Union[str, Path],
    ) -> bool:
        """Remove an asset from a collection.

        Args:
            collection_id: Target collection ID
            asset_path: Asset to remove

        Returns:
            True if removed
        """
        collection = self._collections.get(collection_id)
        if not collection:
            return False

        if collection.remove_asset(asset_path):
            self._save_collections()
            self._notify_change(collection, "asset_removed")
            return True

        return False

    def get_assets_in_collection(
        self,
        collection_id: str,
        include_children: bool = False,
    ) -> set[Path]:
        """Get assets in a collection.

        Args:
            collection_id: Collection ID
            include_children: Include assets from child collections

        Returns:
            Set of asset paths
        """
        collection = self._collections.get(collection_id)
        if not collection:
            return set()

        # For smart collections, evaluate query
        if collection.is_smart and self.asset_provider:
            all_assets = self.asset_provider.get_all_assets()
            matching = collection.evaluate(all_assets) if isinstance(collection, SmartCollection) else []
            result = {Path(item.get("path", "")) for item in matching}
        else:
            result = collection.assets.copy()

        # Include children
        if include_children:
            for child_id in collection.children_ids:
                result.update(self.get_assets_in_collection(child_id, include_children=True))

        return result

    def get_collections_for_asset(self, asset_path: Union[str, Path]) -> list[AssetCollection]:
        """Get all collections containing an asset.

        Args:
            asset_path: Asset path

        Returns:
            List of collections
        """
        asset_path = Path(asset_path)
        result = []

        for collection in self._collections.values():
            if collection.contains(asset_path):
                result.append(collection)

        return result

    def search_collections(self, query: str) -> list[AssetCollection]:
        """Search collections by name or description.

        Args:
            query: Search query

        Returns:
            Matching collections
        """
        query_lower = query.lower()
        return [
            c for c in self._collections.values()
            if query_lower in c.name.lower() or query_lower in c.description.lower()
        ]

    def on_change(self, callback: Callable[[AssetCollection, str], None]) -> None:
        """Register a change callback."""
        self._change_listeners.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        manual = sum(1 for c in self._collections.values() if c.collection_type == CollectionType.MANUAL)
        smart = sum(1 for c in self._collections.values() if c.collection_type == CollectionType.SMART)
        nested = sum(1 for c in self._collections.values() if c.is_nested)

        return {
            "total_collections": len(self._collections),
            "manual_collections": manual,
            "smart_collections": smart,
            "nested_collections": nested,
            "root_collections": len(self._root_collections),
        }

    def _load_collections(self) -> None:
        """Load collections from disk."""
        collections_file = self.storage_path / "collections.json"
        if not collections_file.exists():
            return

        try:
            with open(collections_file, "r") as f:
                data = json.load(f)

            for coll_data in data.get("collections", []):
                collection = AssetCollection.from_dict(coll_data)
                self._collections[collection.id] = collection

            self._root_collections = data.get("root_collections", [])

        except Exception:
            pass

    def _save_collections(self) -> None:
        """Save collections to disk."""
        collections_file = self.storage_path / "collections.json"

        data = {
            "collections": [c.to_dict() for c in self._collections.values()],
            "root_collections": self._root_collections,
        }

        with open(collections_file, "w") as f:
            json.dump(data, f, indent=2)

    def _notify_change(self, collection: AssetCollection, action: str) -> None:
        """Notify listeners of a change."""
        for listener in self._change_listeners:
            try:
                listener(collection, action)
            except Exception:
                pass


__all__ = [
    "CollectionType",
    "CollectionQuery",
    "AssetCollection",
    "SmartCollection",
    "CollectionManager",
]
