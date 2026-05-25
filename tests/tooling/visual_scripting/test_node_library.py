"""
Tests for FlowForge node library.

Tests categories, search, favorites, and custom node registration.
"""

import pytest

from engine.tooling.visual_scripting.node_library import (
    CategoryInfo,
    CATEGORY_INFO,
    NodeEntry,
    SearchResult,
    NodeLibrary,
    get_node_library,
    register_custom_node,
    search_library,
)
from engine.tooling.visual_scripting.node_types import (
    Node,
    NodeCategory,
    NodeMetadata,
    BeginPlayNode,
    BranchNode,
    PrintStringNode,
)


class TestCategoryInfo:
    """Tests for CategoryInfo."""

    def test_default_categories_exist(self):
        assert NodeCategory.EVENT in CATEGORY_INFO
        assert NodeCategory.FLOW_CONTROL in CATEGORY_INFO
        assert NodeCategory.FUNCTION in CATEGORY_INFO

    def test_category_has_display_name(self):
        info = CATEGORY_INFO[NodeCategory.EVENT]
        assert info.display_name == "Events"

    def test_category_has_color(self):
        info = CATEGORY_INFO[NodeCategory.EVENT]
        assert len(info.color) == 3


class TestNodeEntry:
    """Tests for NodeEntry."""

    def test_create_entry(self):
        entry = NodeEntry(
            node_class=BeginPlayNode,
            metadata=BeginPlayNode.get_metadata()
        )
        assert entry.node_class == BeginPlayNode
        assert entry.use_count == 0
        assert entry.is_favorite is False

    def test_entry_with_custom_tags(self):
        entry = NodeEntry(
            node_class=BeginPlayNode,
            metadata=BeginPlayNode.get_metadata(),
            custom_tags=["initialization", "startup"]
        )
        assert "initialization" in entry.custom_tags


class TestNodeLibrary:
    """Tests for NodeLibrary."""

    def test_create_library(self):
        library = NodeLibrary()
        assert len(library._entries) > 0

    def test_register_node(self):
        library = NodeLibrary()
        initial_count = len(library._entries)

        class CustomNode(Node):
            def _setup_pins(self):
                pass

            @classmethod
            def get_metadata(cls):
                return NodeMetadata(
                    display_name="CustomTestNode",
                    category=NodeCategory.CUSTOM
                )

            def execute(self, context):
                return None

        entry = library.register_node(CustomNode)
        assert entry.node_class == CustomNode
        assert len(library._entries) == initial_count + 1

    def test_unregister_node(self):
        library = NodeLibrary()

        result = library.unregister_node("Event BeginPlay")
        assert result is True

        result = library.unregister_node("NonExistent")
        assert result is False

    def test_get_entry(self):
        library = NodeLibrary()
        entry = library.get_entry("Branch")

        assert entry is not None
        assert entry.node_class == BranchNode

    def test_get_node_class(self):
        library = NodeLibrary()
        node_class = library.get_node_class("Branch")

        assert node_class == BranchNode

    def test_get_categories(self):
        library = NodeLibrary()
        categories = library.get_categories()

        assert len(categories) > 0
        assert all(isinstance(c, CategoryInfo) for c in categories)

    def test_get_nodes_in_category(self):
        library = NodeLibrary()
        events = library.get_nodes_in_category(NodeCategory.EVENT)

        assert len(events) > 0
        assert all(e.metadata.category == NodeCategory.EVENT for e in events)

    def test_add_custom_category(self):
        library = NodeLibrary()
        info = library.add_custom_category(
            name="AI",
            display_name="Artificial Intelligence",
            description="AI-related nodes",
            color=(200, 100, 200)
        )

        assert info.display_name == "Artificial Intelligence"


class TestNodeLibrarySearch:
    """Tests for node library search."""

    def test_search_by_name(self):
        library = NodeLibrary()
        results = library.search("Branch")

        assert len(results) > 0
        assert any(r.entry.metadata.display_name == "Branch" for r in results)

    def test_search_by_keyword(self):
        library = NodeLibrary()
        results = library.search("if")

        assert len(results) > 0
        # Branch should be found via "if" keyword
        assert any("Branch" in r.entry.metadata.display_name for r in results)

    def test_search_case_insensitive(self):
        library = NodeLibrary()
        results1 = library.search("branch")
        results2 = library.search("BRANCH")

        assert len(results1) == len(results2)

    def test_search_with_category_filter(self):
        library = NodeLibrary()
        results = library.search("e", categories=[NodeCategory.EVENT])

        assert all(r.entry.metadata.category == NodeCategory.EVENT for r in results)

    def test_search_max_results(self):
        library = NodeLibrary()
        results = library.search("e", max_results=3)

        assert len(results) <= 3

    def test_search_exclude_deprecated(self):
        library = NodeLibrary()

        # Add a deprecated node
        class DeprecatedNode(Node):
            def _setup_pins(self):
                pass

            @classmethod
            def get_metadata(cls):
                return NodeMetadata(
                    display_name="OldSearchNode",
                    category=NodeCategory.CUSTOM,
                    is_deprecated=True,
                    keywords=["old", "deprecated"]
                )

            def execute(self, context):
                return None

        library.register_node(DeprecatedNode)

        results = library.search("OldSearchNode", include_deprecated=False)
        assert len(results) == 0

        results = library.search("OldSearchNode", include_deprecated=True)
        assert len(results) > 0

    def test_search_empty_query(self):
        library = NodeLibrary()
        results = library.search("")

        assert len(results) == 0

    def test_search_result_has_score(self):
        library = NodeLibrary()
        results = library.search("Branch")

        assert len(results) > 0
        assert results[0].score > 0

    def test_search_exact_match_highest_score(self):
        library = NodeLibrary()
        results = library.search("Branch")

        # Exact match should have highest score
        assert results[0].entry.metadata.display_name == "Branch"

    def test_fuzzy_search(self):
        library = NodeLibrary()
        results = library.search("brnch")  # Missing 'a'

        # Should still find Branch with fuzzy matching
        assert any("Branch" in r.entry.metadata.display_name for r in results)


class TestNodeLibraryFavorites:
    """Tests for favorites management."""

    def test_add_favorite(self):
        library = NodeLibrary()
        result = library.add_favorite("Branch")

        assert result is True
        assert library.is_favorite("Branch") is True

    def test_add_favorite_not_found(self):
        library = NodeLibrary()
        result = library.add_favorite("NonExistentNode")

        assert result is False

    def test_remove_favorite(self):
        library = NodeLibrary()
        library.add_favorite("Branch")
        result = library.remove_favorite("Branch")

        assert result is True
        assert library.is_favorite("Branch") is False

    def test_toggle_favorite(self):
        library = NodeLibrary()

        result = library.toggle_favorite("Branch")
        assert result is True  # Added

        result = library.toggle_favorite("Branch")
        assert result is False  # Removed

    def test_get_favorites(self):
        library = NodeLibrary()
        library.add_favorite("Branch")
        library.add_favorite("Print String")

        favorites = library.get_favorites()

        assert len(favorites) == 2


class TestNodeLibraryRecent:
    """Tests for recent nodes tracking."""

    def test_mark_used(self):
        library = NodeLibrary()
        library.mark_used("Branch")

        entry = library.get_entry("Branch")
        assert entry.use_count == 1
        assert entry.last_used is not None

    def test_mark_used_updates_count(self):
        library = NodeLibrary()
        library.mark_used("Branch")
        library.mark_used("Branch")

        entry = library.get_entry("Branch")
        assert entry.use_count == 2

    def test_get_recent(self):
        library = NodeLibrary()
        library.mark_used("Branch")
        library.mark_used("Print String")

        recent = library.get_recent()

        assert len(recent) == 2
        # Most recent first
        assert recent[0].metadata.display_name == "Print String"

    def test_get_recent_limit(self):
        library = NodeLibrary()
        library.mark_used("Branch")
        library.mark_used("Print String")
        library.mark_used("Sequence")

        recent = library.get_recent(limit=2)

        assert len(recent) == 2

    def test_get_frequently_used(self):
        library = NodeLibrary()
        library.mark_used("Branch")
        library.mark_used("Branch")
        library.mark_used("Branch")
        library.mark_used("Print String")

        frequent = library.get_frequently_used(limit=2)

        assert frequent[0].metadata.display_name == "Branch"

    def test_clear_recent(self):
        library = NodeLibrary()
        library.mark_used("Branch")
        library.clear_recent()

        recent = library.get_recent()
        assert len(recent) == 0


class TestNodeLibraryPersistence:
    """Tests for preferences save/load."""

    def test_save_preferences(self):
        library = NodeLibrary()
        library.add_favorite("Branch")
        library.mark_used("Print String")

        prefs = library.save_preferences()

        assert "favorites" in prefs
        assert "recent" in prefs
        assert "usage" in prefs
        assert "Branch" in prefs["favorites"]

    def test_load_preferences(self):
        library1 = NodeLibrary()
        library1.add_favorite("Branch")
        library1.mark_used("Branch")
        library1.mark_used("Branch")
        prefs = library1.save_preferences()

        library2 = NodeLibrary()
        library2.load_preferences(prefs)

        assert library2.is_favorite("Branch") is True
        assert library2.get_entry("Branch").use_count == 2


class TestNodeLibraryPalette:
    """Tests for node palette and context menu."""

    def test_get_node_palette(self):
        library = NodeLibrary()
        palette = library.get_node_palette()

        assert len(palette) > 0
        assert all("name" in item for item in palette)

    def test_get_context_menu_items(self):
        library = NodeLibrary()
        library.add_favorite("Branch")
        library.mark_used("Print String")

        items = library.get_context_menu_items()

        assert len(items) > 0
        # Should have separators and items
        assert any(item.get("type") == "separator" for item in items)


class TestNodeLibraryStatistics:
    """Tests for library statistics."""

    def test_get_statistics(self):
        library = NodeLibrary()
        library.add_favorite("Branch")
        library.mark_used("Branch")

        stats = library.get_statistics()

        assert "total_nodes" in stats
        assert "favorites_count" in stats
        assert "categories_count" in stats
        assert stats["favorites_count"] == 1


class TestGlobalNodeLibrary:
    """Tests for global library functions."""

    def test_get_node_library(self):
        library = get_node_library()
        assert isinstance(library, NodeLibrary)

    def test_register_custom_node(self):
        class GlobalCustomNode(Node):
            def _setup_pins(self):
                pass

            @classmethod
            def get_metadata(cls):
                return NodeMetadata(
                    display_name="GlobalCustomTestNode",
                    category=NodeCategory.CUSTOM
                )

            def execute(self, context):
                return None

        entry = register_custom_node(GlobalCustomNode)
        assert entry is not None

    def test_search_library(self):
        results = search_library("Branch")
        assert len(results) > 0


class TestCategoryTree:
    """Tests for category tree structure."""

    def test_get_category_tree(self):
        library = NodeLibrary()
        tree = library.get_category_tree()

        assert isinstance(tree, dict)
        assert len(tree) > 0

    def test_category_tree_has_nodes(self):
        library = NodeLibrary()
        tree = library.get_category_tree()

        # Event category should have nodes
        if "EVENT" in tree:
            assert "nodes" in tree["EVENT"]
            assert len(tree["EVENT"]["nodes"]) > 0
