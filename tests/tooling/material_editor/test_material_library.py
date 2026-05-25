"""Tests for material library."""
import pytest
import tempfile
import os
from engine.tooling.material_editor.material_library import (
    LibraryItemType, LibraryMetadata, LibraryItem, SearchFilter, SortOrder,
    MaterialLibrary
)
from engine.tooling.material_editor.material_graph import MaterialGraph


class TestLibraryItem:
    """Tests for LibraryItem."""

    def test_create_item(self):
        """Test creating library item."""
        item = LibraryItem(
            id="test-id",
            name="Test Material",
            item_type=LibraryItemType.MATERIAL,
            category="Metals",
            tags=["metal", "shiny"]
        )
        assert item.id == "test-id"
        assert item.name == "Test Material"
        assert item.item_type == LibraryItemType.MATERIAL
        assert "metal" in item.tags

    def test_to_dict(self):
        """Test serialization to dict."""
        item = LibraryItem(
            id="test-id",
            name="Test",
            item_type=LibraryItemType.MATERIAL,
            favorite=True,
            rating=5
        )
        data = item.to_dict()
        assert data["id"] == "test-id"
        assert data["favorite"] is True
        assert data["rating"] == 5

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "id": "test-id",
            "name": "Test",
            "item_type": "MATERIAL",
            "category": "Metals",
            "tags": ["metal"],
            "metadata": {"author": "Test"},
            "favorite": True,
            "rating": 4,
            "usage_count": 10
        }
        item = LibraryItem.from_dict(data)
        assert item.id == "test-id"
        assert item.item_type == LibraryItemType.MATERIAL
        assert item.favorite is True


class TestSearchFilter:
    """Tests for SearchFilter."""

    def test_empty_filter_matches_all(self):
        """Test empty filter matches all items."""
        filter = SearchFilter()
        item = LibraryItem(
            id="1",
            name="Test",
            item_type=LibraryItemType.MATERIAL
        )
        assert filter.matches(item) is True

    def test_query_filter(self):
        """Test query filter matches name."""
        filter = SearchFilter()
        filter.query = "metal"

        metal_item = LibraryItem(id="1", name="Brushed Metal", item_type=LibraryItemType.MATERIAL)
        wood_item = LibraryItem(id="2", name="Oak Wood", item_type=LibraryItemType.MATERIAL)

        assert filter.matches(metal_item) is True
        assert filter.matches(wood_item) is False

    def test_category_filter(self):
        """Test category filter."""
        filter = SearchFilter()
        filter.categories = ["Metals"]

        metal = LibraryItem(id="1", name="Steel", item_type=LibraryItemType.MATERIAL, category="Metals")
        wood = LibraryItem(id="2", name="Oak", item_type=LibraryItemType.MATERIAL, category="Woods")

        assert filter.matches(metal) is True
        assert filter.matches(wood) is False

    def test_tag_filter(self):
        """Test tag filter."""
        filter = SearchFilter()
        filter.tags = ["shiny"]

        shiny = LibraryItem(id="1", name="Chrome", item_type=LibraryItemType.MATERIAL, tags=["shiny", "reflective"])
        matte = LibraryItem(id="2", name="Matte", item_type=LibraryItemType.MATERIAL, tags=["matte"])

        assert filter.matches(shiny) is True
        assert filter.matches(matte) is False

    def test_favorites_filter(self):
        """Test favorites filter."""
        filter = SearchFilter()
        filter.favorites_only = True

        fav = LibraryItem(id="1", name="Favorite", item_type=LibraryItemType.MATERIAL, favorite=True)
        not_fav = LibraryItem(id="2", name="Not Favorite", item_type=LibraryItemType.MATERIAL, favorite=False)

        assert filter.matches(fav) is True
        assert filter.matches(not_fav) is False

    def test_rating_filter(self):
        """Test minimum rating filter."""
        filter = SearchFilter()
        filter.min_rating = 4

        high_rated = LibraryItem(id="1", name="High", item_type=LibraryItemType.MATERIAL, rating=5)
        low_rated = LibraryItem(id="2", name="Low", item_type=LibraryItemType.MATERIAL, rating=2)

        assert filter.matches(high_rated) is True
        assert filter.matches(low_rated) is False


class TestMaterialLibrary:
    """Tests for MaterialLibrary."""

    @pytest.fixture
    def library(self):
        """Create a test library."""
        return MaterialLibrary()

    def test_create_library(self, library):
        """Test creating library."""
        assert library.item_count == 0
        assert len(library.categories) > 0  # Default categories

    def test_add_item(self, library):
        """Test adding item."""
        item = LibraryItem(
            id="test-id",
            name="Test",
            item_type=LibraryItemType.MATERIAL
        )
        result = library.add_item(item)
        assert result is True
        assert library.item_count == 1

    def test_add_duplicate_item(self, library):
        """Test adding duplicate item fails."""
        item = LibraryItem(id="test-id", name="Test", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)
        result = library.add_item(item)
        assert result is False

    def test_remove_item(self, library):
        """Test removing item."""
        item = LibraryItem(id="test-id", name="Test", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)
        result = library.remove_item("test-id")
        assert result is True
        assert library.item_count == 0

    def test_get_item(self, library):
        """Test getting item by ID."""
        item = LibraryItem(id="test-id", name="Test", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)
        retrieved = library.get_item("test-id")
        assert retrieved == item

    def test_get_item_by_name(self, library):
        """Test getting item by name."""
        item = LibraryItem(id="test-id", name="Test Material", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)
        retrieved = library.get_item_by_name("Test Material")
        assert retrieved == item

    def test_update_item(self, library):
        """Test updating item."""
        item = LibraryItem(id="test-id", name="Test", item_type=LibraryItemType.MATERIAL, rating=3)
        library.add_item(item)
        item.rating = 5
        library.update_item(item)
        retrieved = library.get_item("test-id")
        assert retrieved.rating == 5


class TestMaterialLibraryCategories:
    """Tests for library categories."""

    @pytest.fixture
    def library(self):
        return MaterialLibrary()

    def test_default_categories(self, library):
        """Test default categories exist."""
        categories = library.categories
        assert "Metals" in categories
        assert "Woods" in categories
        assert "Uncategorized" in categories

    def test_add_category(self, library):
        """Test adding category."""
        library.add_category("Custom")
        assert "Custom" in library.categories

    def test_remove_category(self, library):
        """Test removing category."""
        library.add_category("Custom")
        result = library.remove_category("Custom")
        assert result is True
        assert "Custom" not in library.categories

    def test_cannot_remove_default_category(self, library):
        """Test cannot remove default categories."""
        result = library.remove_category("Metals")
        assert result is False
        assert "Metals" in library.categories

    def test_rename_category(self, library):
        """Test renaming category."""
        library.add_category("OldName")
        result = library.rename_category("OldName", "NewName")
        assert result is True
        assert "NewName" in library.categories
        assert "OldName" not in library.categories

    def test_get_items_in_category(self, library):
        """Test getting items in category."""
        item1 = LibraryItem(id="1", name="Steel", item_type=LibraryItemType.MATERIAL, category="Metals")
        item2 = LibraryItem(id="2", name="Oak", item_type=LibraryItemType.MATERIAL, category="Woods")
        library.add_item(item1)
        library.add_item(item2)

        metals = library.get_items_in_category("Metals")
        assert len(metals) == 1
        assert metals[0].name == "Steel"


class TestMaterialLibraryTags:
    """Tests for library tags."""

    @pytest.fixture
    def library(self):
        return MaterialLibrary()

    def test_add_tag_to_item(self, library):
        """Test adding tag to item."""
        item = LibraryItem(id="1", name="Test", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)
        library.add_tag_to_item("1", "shiny")

        retrieved = library.get_item("1")
        assert "shiny" in retrieved.tags

    def test_remove_tag_from_item(self, library):
        """Test removing tag from item."""
        item = LibraryItem(id="1", name="Test", item_type=LibraryItemType.MATERIAL, tags=["shiny"])
        library.add_item(item)
        library.remove_tag_from_item("1", "shiny")

        retrieved = library.get_item("1")
        assert "shiny" not in retrieved.tags

    def test_get_items_with_tag(self, library):
        """Test getting items with tag."""
        item1 = LibraryItem(id="1", name="Chrome", item_type=LibraryItemType.MATERIAL, tags=["shiny"])
        item2 = LibraryItem(id="2", name="Matte", item_type=LibraryItemType.MATERIAL, tags=["matte"])
        library.add_item(item1)
        library.add_item(item2)

        shiny_items = library.get_items_with_tag("shiny")
        assert len(shiny_items) == 1
        assert shiny_items[0].name == "Chrome"


class TestMaterialLibraryFavorites:
    """Tests for library favorites."""

    @pytest.fixture
    def library(self):
        return MaterialLibrary()

    def test_set_favorite(self, library):
        """Test setting favorite status."""
        item = LibraryItem(id="1", name="Test", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)
        library.set_favorite("1", True)

        assert library.get_item("1").favorite is True

    def test_toggle_favorite(self, library):
        """Test toggling favorite status."""
        item = LibraryItem(id="1", name="Test", item_type=LibraryItemType.MATERIAL, favorite=False)
        library.add_item(item)
        library.toggle_favorite("1")

        assert library.get_item("1").favorite is True

        library.toggle_favorite("1")
        assert library.get_item("1").favorite is False

    def test_get_favorites(self, library):
        """Test getting favorites."""
        item1 = LibraryItem(id="1", name="Fav", item_type=LibraryItemType.MATERIAL, favorite=True)
        item2 = LibraryItem(id="2", name="Not Fav", item_type=LibraryItemType.MATERIAL, favorite=False)
        library.add_item(item1)
        library.add_item(item2)

        favorites = library.favorites
        assert len(favorites) == 1
        assert favorites[0].name == "Fav"


class TestMaterialLibraryRating:
    """Tests for library rating."""

    @pytest.fixture
    def library(self):
        return MaterialLibrary()

    def test_set_rating(self, library):
        """Test setting rating."""
        item = LibraryItem(id="1", name="Test", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)
        library.set_rating("1", 5)

        assert library.get_item("1").rating == 5

    def test_rating_clamped(self, library):
        """Test rating is clamped to 0-5."""
        item = LibraryItem(id="1", name="Test", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)

        library.set_rating("1", 10)
        assert library.get_item("1").rating == 5

        library.set_rating("1", -5)
        assert library.get_item("1").rating == 0


class TestMaterialLibrarySearch:
    """Tests for library search."""

    @pytest.fixture
    def populated_library(self):
        """Create library with test items."""
        library = MaterialLibrary()

        items = [
            LibraryItem(id="1", name="Brushed Steel", item_type=LibraryItemType.MATERIAL,
                       category="Metals", tags=["metal", "brushed"], rating=5),
            LibraryItem(id="2", name="Oak Wood", item_type=LibraryItemType.MATERIAL,
                       category="Woods", tags=["wood", "natural"], rating=4),
            LibraryItem(id="3", name="Chrome", item_type=LibraryItemType.MATERIAL,
                       category="Metals", tags=["metal", "shiny"], rating=5, favorite=True),
            LibraryItem(id="4", name="Marble", item_type=LibraryItemType.MATERIAL,
                       category="Stones", tags=["stone", "natural"], rating=3),
        ]

        for item in items:
            library.add_item(item)

        return library

    def test_search_by_query(self, populated_library):
        """Test searching by query."""
        filter = SearchFilter()
        filter.query = "steel"
        results = populated_library.search(filter)

        assert len(results) == 1
        assert results[0].name == "Brushed Steel"

    def test_search_with_multiple_filters(self, populated_library):
        """Test searching with multiple filters."""
        filter = SearchFilter()
        filter.categories = ["Metals"]
        filter.min_rating = 5
        results = populated_library.search(filter)

        assert len(results) == 2  # Steel and Chrome

    def test_quick_search(self, populated_library):
        """Test quick search."""
        results = populated_library.quick_search("metal")

        assert len(results) == 2  # Items with "metal" tag

    def test_search_sort_by_name(self, populated_library):
        """Test search with name sorting."""
        filter = SearchFilter()
        results = populated_library.search(filter, SortOrder.NAME_ASC)

        # Should be alphabetically sorted
        names = [r.name for r in results]
        assert names == sorted(names)

    def test_search_sort_by_rating(self, populated_library):
        """Test search with rating sorting."""
        filter = SearchFilter()
        results = populated_library.search(filter, SortOrder.RATING_DESC)

        # First items should have highest ratings
        assert results[0].rating >= results[-1].rating


class TestMaterialLibraryUsageTracking:
    """Tests for usage tracking."""

    @pytest.fixture
    def library(self):
        return MaterialLibrary()

    def test_record_usage(self, library):
        """Test recording usage."""
        item = LibraryItem(id="1", name="Test", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)

        library.record_usage("1")
        library.record_usage("1")
        library.record_usage("1")

        assert library.get_item("1").usage_count == 3

    def test_get_most_used(self, library):
        """Test getting most used items."""
        item1 = LibraryItem(id="1", name="Popular", item_type=LibraryItemType.MATERIAL, usage_count=100)
        item2 = LibraryItem(id="2", name="Less Popular", item_type=LibraryItemType.MATERIAL, usage_count=10)
        library.add_item(item1)
        library.add_item(item2)

        most_used = library.get_most_used(1)
        assert len(most_used) == 1
        assert most_used[0].name == "Popular"

    def test_recent_items(self, library):
        """Test recent items tracking."""
        for i in range(5):
            item = LibraryItem(id=str(i), name=f"Item{i}", item_type=LibraryItemType.MATERIAL)
            library.add_item(item)
            library.get_item(str(i))  # Access to add to recent

        recent = library.recent_items
        assert len(recent) > 0


class TestMaterialLibraryFolders:
    """Tests for folder structure."""

    @pytest.fixture
    def library(self):
        return MaterialLibrary()

    def test_create_folder(self, library):
        """Test creating folder."""
        folder = library.create_folder("My Folder")
        assert folder is not None
        assert folder.item_type == LibraryItemType.FOLDER

    def test_create_nested_folder(self, library):
        """Test creating nested folder."""
        parent = library.create_folder("Parent")
        child = library.create_folder("Child", parent.id)

        assert child.parent_folder_id == parent.id

    def test_get_folder_contents(self, library):
        """Test getting folder contents."""
        folder = library.create_folder("Folder")
        item = LibraryItem(id="1", name="Item", item_type=LibraryItemType.MATERIAL)
        item.parent_folder_id = folder.id
        library.add_item(item)

        contents = library.get_folder_contents(folder.id)
        assert len(contents) == 1

    def test_get_root_contents(self, library):
        """Test getting root level contents."""
        folder = library.create_folder("Folder")
        item = LibraryItem(id="1", name="Item", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)

        root_contents = library.get_folder_contents(None)
        # Both folder and item at root
        assert len(root_contents) == 2

    def test_move_to_folder(self, library):
        """Test moving item to folder."""
        folder = library.create_folder("Folder")
        item = LibraryItem(id="1", name="Item", item_type=LibraryItemType.MATERIAL)
        library.add_item(item)

        library.move_to_folder("1", folder.id)

        assert library.get_item("1").parent_folder_id == folder.id


class TestMaterialLibrarySerialization:
    """Tests for library serialization."""

    @pytest.fixture
    def library(self):
        lib = MaterialLibrary()
        lib.add_item(LibraryItem(id="1", name="Test", item_type=LibraryItemType.MATERIAL,
                                category="Metals", tags=["test"], favorite=True))
        return lib

    def test_to_dict(self, library):
        """Test serialization to dict."""
        data = library.to_dict()
        assert "items" in data
        assert len(data["items"]) == 1

    def test_from_dict(self, library):
        """Test deserialization from dict."""
        data = library.to_dict()
        restored = MaterialLibrary.from_dict(data)

        assert restored.item_count == 1
        item = restored.get_item("1")
        assert item.name == "Test"
        assert item.favorite is True


class TestMaterialLibraryCallbacks:
    """Tests for library callbacks."""

    def test_on_item_added(self):
        """Test item added callback."""
        library = MaterialLibrary()
        added_items = []

        library.on_item_added(lambda i: added_items.append(i))
        library.add_item(LibraryItem(id="1", name="Test", item_type=LibraryItemType.MATERIAL))

        assert len(added_items) == 1

    def test_on_item_removed(self):
        """Test item removed callback."""
        library = MaterialLibrary()
        removed_ids = []

        library.add_item(LibraryItem(id="1", name="Test", item_type=LibraryItemType.MATERIAL))
        library.on_item_removed(lambda id: removed_ids.append(id))
        library.remove_item("1")

        assert len(removed_ids) == 1
        assert removed_ids[0] == "1"


class TestMaterialLibraryStatistics:
    """Tests for library statistics."""

    def test_get_statistics(self):
        """Test getting library statistics."""
        library = MaterialLibrary()
        library.add_item(LibraryItem(id="1", name="Steel", item_type=LibraryItemType.MATERIAL,
                                    category="Metals", favorite=True))
        library.add_item(LibraryItem(id="2", name="Oak", item_type=LibraryItemType.MATERIAL,
                                    category="Woods"))
        library.add_item(LibraryItem(id="3", name="Folder", item_type=LibraryItemType.FOLDER))

        stats = library.get_statistics()

        assert stats["total_items"] == 3
        assert stats["by_type"]["MATERIAL"] == 2
        assert stats["by_type"]["FOLDER"] == 1
        assert stats["favorites_count"] == 1
