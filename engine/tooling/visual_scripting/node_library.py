"""
FlowForge Node Library - Node library with categories, search, favorites, and custom nodes.

Provides node organization and discovery:
- Hierarchical category system
- Full-text search with fuzzy matching
- Favorites management
- Recent nodes tracking
- Custom node registration
- Node templates and presets
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from .node_types import (
    Node,
    NodeCategory,
    NodeMetadata,
    NODE_REGISTRY,
    get_node_class,
    search_nodes,
)


@dataclass
class CategoryInfo:
    """Information about a node category."""
    category: NodeCategory
    display_name: str
    description: str = ""
    icon: str = ""
    color: Tuple[int, int, int] = (100, 100, 100)
    parent: Optional[NodeCategory] = None
    sort_order: int = 0
    is_expanded: bool = True


# Category hierarchy and metadata
CATEGORY_INFO: Dict[NodeCategory, CategoryInfo] = {
    NodeCategory.EVENT: CategoryInfo(
        category=NodeCategory.EVENT,
        display_name="Events",
        description="Entry points and event handlers",
        icon="event",
        color=(180, 0, 0),
        sort_order=0
    ),
    NodeCategory.FLOW_CONTROL: CategoryInfo(
        category=NodeCategory.FLOW_CONTROL,
        display_name="Flow Control",
        description="Control execution flow",
        icon="flow",
        color=(255, 255, 255),
        sort_order=1
    ),
    NodeCategory.FUNCTION: CategoryInfo(
        category=NodeCategory.FUNCTION,
        display_name="Functions",
        description="Function calls and definitions",
        icon="function",
        color=(0, 100, 200),
        sort_order=2
    ),
    NodeCategory.VARIABLE: CategoryInfo(
        category=NodeCategory.VARIABLE,
        display_name="Variables",
        description="Variable access and modification",
        icon="variable",
        color=(0, 160, 0),
        sort_order=3
    ),
    NodeCategory.MATH: CategoryInfo(
        category=NodeCategory.MATH,
        display_name="Math",
        description="Mathematical operations",
        icon="math",
        color=(100, 200, 100),
        sort_order=4
    ),
    NodeCategory.STRING: CategoryInfo(
        category=NodeCategory.STRING,
        display_name="String",
        description="String manipulation",
        icon="string",
        color=(255, 0, 200),
        sort_order=5
    ),
    NodeCategory.VECTOR: CategoryInfo(
        category=NodeCategory.VECTOR,
        display_name="Vector",
        description="Vector operations",
        icon="vector",
        color=(255, 200, 0),
        sort_order=6
    ),
    NodeCategory.TRANSFORM: CategoryInfo(
        category=NodeCategory.TRANSFORM,
        display_name="Transform",
        description="Transform operations",
        icon="transform",
        color=(255, 140, 0),
        sort_order=7
    ),
    NodeCategory.OBJECT: CategoryInfo(
        category=NodeCategory.OBJECT,
        display_name="Object",
        description="Object and actor operations",
        icon="object",
        color=(0, 160, 255),
        sort_order=8
    ),
    NodeCategory.ARRAY: CategoryInfo(
        category=NodeCategory.ARRAY,
        display_name="Array",
        description="Array/list operations",
        icon="array",
        color=(100, 100, 100),
        sort_order=9
    ),
    NodeCategory.UTILITY: CategoryInfo(
        category=NodeCategory.UTILITY,
        display_name="Utilities",
        description="Utility nodes",
        icon="utility",
        color=(150, 150, 150),
        sort_order=10
    ),
    NodeCategory.MACRO: CategoryInfo(
        category=NodeCategory.MACRO,
        display_name="Macros",
        description="Reusable node groups",
        icon="macro",
        color=(100, 100, 200),
        sort_order=11
    ),
    NodeCategory.CUSTOM: CategoryInfo(
        category=NodeCategory.CUSTOM,
        display_name="Custom",
        description="Custom/user-defined nodes",
        icon="custom",
        color=(200, 100, 100),
        sort_order=12
    ),
    NodeCategory.DEBUG: CategoryInfo(
        category=NodeCategory.DEBUG,
        display_name="Debug",
        description="Debug and development nodes",
        icon="debug",
        color=(255, 200, 0),
        sort_order=13
    ),
    NodeCategory.AI: CategoryInfo(
        category=NodeCategory.AI,
        display_name="AI",
        description="AI and behavior nodes",
        icon="ai",
        color=(200, 100, 200),
        sort_order=14
    ),
    NodeCategory.PHYSICS: CategoryInfo(
        category=NodeCategory.PHYSICS,
        display_name="Physics",
        description="Physics operations",
        icon="physics",
        color=(100, 200, 200),
        sort_order=15
    ),
    NodeCategory.AUDIO: CategoryInfo(
        category=NodeCategory.AUDIO,
        display_name="Audio",
        description="Audio operations",
        icon="audio",
        color=(200, 150, 100),
        sort_order=16
    ),
    NodeCategory.UI: CategoryInfo(
        category=NodeCategory.UI,
        display_name="UI",
        description="User interface nodes",
        icon="ui",
        color=(100, 150, 200),
        sort_order=17
    ),
}


@dataclass
class NodeEntry:
    """An entry in the node library."""
    node_class: Type[Node]
    metadata: NodeMetadata
    use_count: int = 0
    last_used: Optional[datetime] = None
    is_favorite: bool = False
    custom_tags: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """A search result with relevance score."""
    entry: NodeEntry
    score: float
    match_type: str  # "exact", "prefix", "contains", "keyword", "fuzzy"


class NodeLibrary:
    """Library of available nodes with search and organization."""

    def __init__(self):
        self._entries: Dict[str, NodeEntry] = {}
        self._favorites: Set[str] = set()
        self._recent: List[str] = []
        self._max_recent = 20

        # Custom categories
        self._custom_categories: Dict[str, CategoryInfo] = {}

        # Populate from registry
        self._load_from_registry()

    def _load_from_registry(self) -> None:
        """Load nodes from the global registry."""
        for name, node_class in NODE_REGISTRY.items():
            self.register_node(node_class)

    def register_node(
        self,
        node_class: Type[Node],
        custom_tags: Optional[List[str]] = None
    ) -> NodeEntry:
        """Register a node type in the library."""
        metadata = node_class.get_metadata()
        entry = NodeEntry(
            node_class=node_class,
            metadata=metadata,
            custom_tags=custom_tags or []
        )
        self._entries[metadata.display_name] = entry
        return entry

    def unregister_node(self, name: str) -> bool:
        """Unregister a node type."""
        if name in self._entries:
            del self._entries[name]
            self._favorites.discard(name)
            return True
        return False

    def get_entry(self, name: str) -> Optional[NodeEntry]:
        """Get a node entry by name."""
        return self._entries.get(name)

    def get_node_class(self, name: str) -> Optional[Type[Node]]:
        """Get a node class by name."""
        entry = self._entries.get(name)
        return entry.node_class if entry else None

    # =========================================================================
    # CATEGORIES
    # =========================================================================

    def get_categories(self) -> List[CategoryInfo]:
        """Get all categories sorted by order."""
        all_categories = list(CATEGORY_INFO.values()) + list(self._custom_categories.values())
        return sorted(all_categories, key=lambda c: c.sort_order)

    def get_category_info(self, category: NodeCategory) -> Optional[CategoryInfo]:
        """Get information about a category."""
        return CATEGORY_INFO.get(category) or self._custom_categories.get(category.name)

    def get_nodes_in_category(self, category: NodeCategory) -> List[NodeEntry]:
        """Get all nodes in a category."""
        return [
            entry for entry in self._entries.values()
            if entry.metadata.category == category
        ]

    def add_custom_category(
        self,
        name: str,
        display_name: str,
        description: str = "",
        color: Tuple[int, int, int] = (100, 100, 100),
        parent: Optional[NodeCategory] = None
    ) -> CategoryInfo:
        """Add a custom category."""
        # Create a dynamic category
        info = CategoryInfo(
            category=NodeCategory.CUSTOM,  # Use CUSTOM as base
            display_name=display_name,
            description=description,
            color=color,
            parent=parent,
            sort_order=100 + len(self._custom_categories)
        )
        self._custom_categories[name] = info
        return info

    def get_category_tree(self) -> Dict[str, Any]:
        """Get the category hierarchy as a tree."""
        tree = {}

        for category in self.get_categories():
            nodes = self.get_nodes_in_category(category.category)
            if nodes or category.parent is None:
                tree[category.category.name] = {
                    "info": category,
                    "nodes": [
                        {
                            "name": entry.metadata.display_name,
                            "is_favorite": entry.is_favorite,
                            "is_deprecated": entry.metadata.is_deprecated
                        }
                        for entry in sorted(nodes, key=lambda e: e.metadata.display_name)
                    ],
                    "children": {}
                }

        return tree

    # =========================================================================
    # SEARCH
    # =========================================================================

    def search(
        self,
        query: str,
        categories: Optional[List[NodeCategory]] = None,
        include_deprecated: bool = False,
        max_results: int = 50
    ) -> List[SearchResult]:
        """Search for nodes by query string."""
        if not query:
            return []

        query = query.lower().strip()
        results: List[SearchResult] = []

        for name, entry in self._entries.items():
            if not include_deprecated and entry.metadata.is_deprecated:
                continue

            if categories and entry.metadata.category not in categories:
                continue

            score, match_type = self._calculate_match_score(query, entry)

            if score > 0:
                results.append(SearchResult(
                    entry=entry,
                    score=score,
                    match_type=match_type
                ))

        # Sort by score (descending), then by name
        results.sort(key=lambda r: (-r.score, r.entry.metadata.display_name))

        return results[:max_results]

    def _calculate_match_score(
        self,
        query: str,
        entry: NodeEntry
    ) -> Tuple[float, str]:
        """Calculate relevance score for a search query."""
        name = entry.metadata.display_name.lower()
        keywords = [kw.lower() for kw in entry.metadata.keywords]
        description = entry.metadata.description.lower()
        tags = [t.lower() for t in entry.custom_tags]

        # Exact match (highest priority)
        if query == name:
            return (100.0, "exact")

        # Prefix match
        if name.startswith(query):
            return (80.0, "prefix")

        # Contains match
        if query in name:
            return (60.0, "contains")

        # Keyword exact match
        if query in keywords:
            return (70.0, "keyword")

        # Keyword partial match
        for kw in keywords:
            if query in kw or kw in query:
                return (50.0, "keyword")

        # Tag match
        if query in tags:
            return (65.0, "keyword")

        # Description match
        if query in description:
            return (30.0, "contains")

        # Fuzzy match (simple character matching)
        fuzzy_score = self._fuzzy_match_score(query, name)
        if fuzzy_score > 0.5:
            return (fuzzy_score * 40, "fuzzy")

        return (0.0, "none")

    def _fuzzy_match_score(self, query: str, target: str) -> float:
        """Simple fuzzy matching score (0.0 to 1.0)."""
        if not query or not target:
            return 0.0

        # Check if all query characters appear in order
        query_idx = 0
        for char in target:
            if query_idx < len(query) and char == query[query_idx]:
                query_idx += 1

        if query_idx == len(query):
            # All characters found in order
            # Score based on how "spread out" they are
            return len(query) / len(target)

        return 0.0

    def search_by_context(
        self,
        output_type: Optional[str] = None,
        input_type: Optional[str] = None,
        is_pure: Optional[bool] = None
    ) -> List[NodeEntry]:
        """Search for nodes that can connect to specific types."""
        results = []

        for entry in self._entries.values():
            if is_pure is not None and entry.metadata.is_pure != is_pure:
                continue

            # Would need to check pin types - simplified version
            results.append(entry)

        return results

    # =========================================================================
    # FAVORITES
    # =========================================================================

    def add_favorite(self, name: str) -> bool:
        """Add a node to favorites."""
        if name in self._entries:
            self._favorites.add(name)
            self._entries[name].is_favorite = True
            return True
        return False

    def remove_favorite(self, name: str) -> bool:
        """Remove a node from favorites."""
        if name in self._favorites:
            self._favorites.discard(name)
            if name in self._entries:
                self._entries[name].is_favorite = False
            return True
        return False

    def toggle_favorite(self, name: str) -> bool:
        """Toggle favorite status."""
        if name in self._favorites:
            self.remove_favorite(name)
            return False
        else:
            self.add_favorite(name)
            return True

    def get_favorites(self) -> List[NodeEntry]:
        """Get all favorite nodes."""
        return [
            self._entries[name]
            for name in self._favorites
            if name in self._entries
        ]

    def is_favorite(self, name: str) -> bool:
        """Check if a node is a favorite."""
        return name in self._favorites

    # =========================================================================
    # RECENT NODES
    # =========================================================================

    def mark_used(self, name: str) -> None:
        """Mark a node as recently used."""
        if name not in self._entries:
            return

        # Update entry
        entry = self._entries[name]
        entry.use_count += 1
        entry.last_used = datetime.now()

        # Update recent list
        if name in self._recent:
            self._recent.remove(name)
        self._recent.insert(0, name)

        # Trim to max size
        while len(self._recent) > self._max_recent:
            self._recent.pop()

    def get_recent(self, limit: Optional[int] = None) -> List[NodeEntry]:
        """Get recently used nodes."""
        limit = limit or self._max_recent
        return [
            self._entries[name]
            for name in self._recent[:limit]
            if name in self._entries
        ]

    def get_frequently_used(self, limit: int = 10) -> List[NodeEntry]:
        """Get most frequently used nodes."""
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.use_count,
            reverse=True
        )
        return sorted_entries[:limit]

    def clear_recent(self) -> None:
        """Clear recent nodes list."""
        self._recent.clear()

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save_preferences(self) -> Dict[str, Any]:
        """Save library preferences (favorites, recent, etc.)."""
        return {
            "favorites": list(self._favorites),
            "recent": self._recent[:],
            "usage": {
                name: {
                    "use_count": entry.use_count,
                    "last_used": entry.last_used.isoformat() if entry.last_used else None
                }
                for name, entry in self._entries.items()
                if entry.use_count > 0
            }
        }

    def load_preferences(self, data: Dict[str, Any]) -> None:
        """Load library preferences."""
        # Favorites
        self._favorites.clear()
        for name in data.get("favorites", []):
            if name in self._entries:
                self._favorites.add(name)
                self._entries[name].is_favorite = True

        # Recent
        self._recent.clear()
        for name in data.get("recent", []):
            if name in self._entries:
                self._recent.append(name)

        # Usage stats
        for name, usage in data.get("usage", {}).items():
            if name in self._entries:
                self._entries[name].use_count = usage.get("use_count", 0)
                if usage.get("last_used"):
                    try:
                        self._entries[name].last_used = datetime.fromisoformat(usage["last_used"])
                    except (ValueError, TypeError):
                        pass

    # =========================================================================
    # NODE TEMPLATES
    # =========================================================================

    def get_node_palette(self) -> List[Dict[str, Any]]:
        """Get a palette of common nodes for quick access."""
        return [
            {"name": "Branch", "category": "Flow Control"},
            {"name": "Sequence", "category": "Flow Control"},
            {"name": "For Loop", "category": "Flow Control"},
            {"name": "Print String", "category": "Debug"},
            {"name": "Delay", "category": "Flow Control"},
            {"name": "Event BeginPlay", "category": "Events"},
            {"name": "Event Tick", "category": "Events"},
        ]

    def get_context_menu_items(
        self,
        context_type: Optional[str] = None,
        position: Tuple[float, float] = (0, 0)
    ) -> List[Dict[str, Any]]:
        """Get items for a context menu."""
        items = []

        # Add favorites section
        favorites = self.get_favorites()
        if favorites:
            items.append({"type": "separator", "label": "Favorites"})
            for entry in favorites[:5]:
                items.append({
                    "type": "node",
                    "name": entry.metadata.display_name,
                    "category": entry.metadata.category.name
                })

        # Add recent section
        recent = self.get_recent(5)
        if recent:
            items.append({"type": "separator", "label": "Recent"})
            for entry in recent:
                items.append({
                    "type": "node",
                    "name": entry.metadata.display_name,
                    "category": entry.metadata.category.name
                })

        # Add category submenus
        items.append({"type": "separator", "label": "Categories"})
        for category_info in self.get_categories():
            nodes = self.get_nodes_in_category(category_info.category)
            if nodes:
                items.append({
                    "type": "submenu",
                    "label": category_info.display_name,
                    "items": [
                        {
                            "type": "node",
                            "name": entry.metadata.display_name
                        }
                        for entry in sorted(nodes, key=lambda e: e.metadata.display_name)
                    ]
                })

        return items

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        categories_count = {}
        for entry in self._entries.values():
            cat = entry.metadata.category.name
            categories_count[cat] = categories_count.get(cat, 0) + 1

        deprecated_count = sum(
            1 for e in self._entries.values()
            if e.metadata.is_deprecated
        )

        pure_count = sum(
            1 for e in self._entries.values()
            if e.metadata.is_pure
        )

        return {
            "total_nodes": len(self._entries),
            "favorites_count": len(self._favorites),
            "recent_count": len(self._recent),
            "categories_count": categories_count,
            "deprecated_count": deprecated_count,
            "pure_function_count": pure_count,
            "custom_categories": len(self._custom_categories)
        }


# Global library instance
_library: Optional[NodeLibrary] = None


def get_node_library() -> NodeLibrary:
    """Get the global node library instance."""
    global _library
    if _library is None:
        _library = NodeLibrary()
    return _library


def register_custom_node(
    node_class: Type[Node],
    tags: Optional[List[str]] = None
) -> NodeEntry:
    """Register a custom node in the global library."""
    return get_node_library().register_node(node_class, tags)


def search_library(query: str, **kwargs) -> List[SearchResult]:
    """Search the global node library."""
    return get_node_library().search(query, **kwargs)
