"""
Comprehensive tests for AssetSearch functionality.

Tests search queries, filters, indexing, and saved searches.
"""

import pytest
import sys
import tempfile
import shutil
import time
from pathlib import Path

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.assettools.search import (
    SearchOperator,
    SearchFieldType,
    SearchFilter,
    SearchQuery,
    SearchResult,
    SavedSearch,
    AssetSearch,
)


@pytest.fixture
def temp_search_dir():
    """Create a temporary directory for search tests."""
    path = Path(tempfile.mkdtemp())
    (path / "storage").mkdir()
    yield path
    shutil.rmtree(path)


@pytest.fixture
def sample_assets():
    """Create sample asset data for testing."""
    return [
        {
            "path": "/assets/hero_diffuse.png",
            "name": "hero_diffuse",
            "description": "Hero character diffuse texture",
            "asset_type": "TEXTURE",
            "extension": "png",
            "size": 2048,
            "tags": ["character", "hero", "diffuse"],
        },
        {
            "path": "/assets/hero_normal.png",
            "name": "hero_normal",
            "description": "Hero character normal map",
            "asset_type": "TEXTURE",
            "extension": "png",
            "size": 1024,
            "tags": ["character", "hero", "normal"],
        },
        {
            "path": "/assets/villain_model.fbx",
            "name": "villain_model",
            "description": "Villain character 3D model",
            "asset_type": "MESH",
            "extension": "fbx",
            "size": 5000,
            "tags": ["character", "villain", "mesh"],
        },
        {
            "path": "/assets/sword.fbx",
            "name": "sword",
            "description": "Weapon sword model",
            "asset_type": "MESH",
            "extension": "fbx",
            "size": 800,
            "tags": ["weapon", "prop"],
        },
        {
            "path": "/assets/ambient_music.ogg",
            "name": "ambient_music",
            "description": "Background ambient music",
            "asset_type": "AUDIO",
            "extension": "ogg",
            "size": 3500,
            "tags": ["audio", "music", "ambient"],
        },
    ]


class TestSearchOperator:
    """Test SearchOperator enum."""

    def test_operator_values(self):
        """All operators should be defined."""
        assert SearchOperator.EQUALS
        assert SearchOperator.NOT_EQUALS
        assert SearchOperator.CONTAINS
        assert SearchOperator.NOT_CONTAINS
        assert SearchOperator.STARTS_WITH
        assert SearchOperator.ENDS_WITH
        assert SearchOperator.GREATER_THAN
        assert SearchOperator.LESS_THAN
        assert SearchOperator.GREATER_EQUAL
        assert SearchOperator.LESS_EQUAL
        assert SearchOperator.IN
        assert SearchOperator.NOT_IN
        assert SearchOperator.MATCHES
        assert SearchOperator.EXISTS
        assert SearchOperator.NOT_EXISTS


class TestSearchFieldType:
    """Test SearchFieldType enum."""

    def test_field_types(self):
        """All field types should be defined."""
        assert SearchFieldType.STRING
        assert SearchFieldType.INTEGER
        assert SearchFieldType.FLOAT
        assert SearchFieldType.BOOLEAN
        assert SearchFieldType.DATE
        assert SearchFieldType.PATH
        assert SearchFieldType.TAG
        assert SearchFieldType.SIZE


class TestSearchFilter:
    """Test SearchFilter dataclass."""

    def test_filter_creation(self):
        """Filter should store all attributes."""
        filter = SearchFilter(
            field="name",
            operator=SearchOperator.CONTAINS,
            value="hero",
        )

        assert filter.field == "name"
        assert filter.operator == SearchOperator.CONTAINS
        assert filter.value == "hero"

    def test_filter_defaults(self):
        """Filter should have sensible defaults."""
        filter = SearchFilter(
            field="name",
            operator=SearchOperator.EQUALS,
            value="test",
        )

        assert filter.field_type == SearchFieldType.STRING
        assert filter.case_sensitive is False

    def test_matches_equals(self):
        """matches() should handle EQUALS operator."""
        filter = SearchFilter("name", SearchOperator.EQUALS, "hero")

        assert filter.matches({"name": "hero"}) is True
        assert filter.matches({"name": "Hero"}) is True  # case-insensitive
        assert filter.matches({"name": "villain"}) is False

    def test_matches_equals_case_sensitive(self):
        """matches() should respect case_sensitive flag."""
        filter = SearchFilter("name", SearchOperator.EQUALS, "hero", case_sensitive=True)

        assert filter.matches({"name": "hero"}) is True
        assert filter.matches({"name": "Hero"}) is False

    def test_matches_not_equals(self):
        """matches() should handle NOT_EQUALS operator."""
        filter = SearchFilter("type", SearchOperator.NOT_EQUALS, "texture")

        assert filter.matches({"type": "mesh"}) is True
        assert filter.matches({"type": "texture"}) is False

    def test_matches_contains(self):
        """matches() should handle CONTAINS operator."""
        filter = SearchFilter("name", SearchOperator.CONTAINS, "hero")

        assert filter.matches({"name": "hero_diffuse"}) is True
        assert filter.matches({"name": "my_hero_model"}) is True
        assert filter.matches({"name": "villain"}) is False

    def test_matches_contains_list(self):
        """matches() should handle CONTAINS for lists."""
        filter = SearchFilter("tags", SearchOperator.CONTAINS, "hero")

        assert filter.matches({"tags": ["character", "hero"]}) is True
        assert filter.matches({"tags": ["villain"]}) is False

    def test_matches_not_contains(self):
        """matches() should handle NOT_CONTAINS operator."""
        filter = SearchFilter("name", SearchOperator.NOT_CONTAINS, "temp")

        assert filter.matches({"name": "hero"}) is True
        assert filter.matches({"name": "temp_file"}) is False

    def test_matches_starts_with(self):
        """matches() should handle STARTS_WITH operator."""
        filter = SearchFilter("name", SearchOperator.STARTS_WITH, "hero")

        assert filter.matches({"name": "hero_diffuse"}) is True
        assert filter.matches({"name": "my_hero"}) is False

    def test_matches_ends_with(self):
        """matches() should handle ENDS_WITH operator."""
        filter = SearchFilter("name", SearchOperator.ENDS_WITH, "_diffuse")

        assert filter.matches({"name": "hero_diffuse"}) is True
        assert filter.matches({"name": "diffuse_map"}) is False

    def test_matches_greater_than(self):
        """matches() should handle GREATER_THAN operator."""
        filter = SearchFilter("size", SearchOperator.GREATER_THAN, 1000)

        assert filter.matches({"size": 2000}) is True
        assert filter.matches({"size": 1000}) is False
        assert filter.matches({"size": 500}) is False

    def test_matches_less_than(self):
        """matches() should handle LESS_THAN operator."""
        filter = SearchFilter("size", SearchOperator.LESS_THAN, 1000)

        assert filter.matches({"size": 500}) is True
        assert filter.matches({"size": 1000}) is False

    def test_matches_greater_equal(self):
        """matches() should handle GREATER_EQUAL operator."""
        filter = SearchFilter("size", SearchOperator.GREATER_EQUAL, 1000)

        assert filter.matches({"size": 1000}) is True
        assert filter.matches({"size": 1500}) is True
        assert filter.matches({"size": 500}) is False

    def test_matches_less_equal(self):
        """matches() should handle LESS_EQUAL operator."""
        filter = SearchFilter("size", SearchOperator.LESS_EQUAL, 1000)

        assert filter.matches({"size": 1000}) is True
        assert filter.matches({"size": 500}) is True
        assert filter.matches({"size": 1500}) is False

    def test_matches_in(self):
        """matches() should handle IN operator."""
        filter = SearchFilter("type", SearchOperator.IN, ["texture", "mesh"])

        assert filter.matches({"type": "texture"}) is True
        assert filter.matches({"type": "mesh"}) is True
        assert filter.matches({"type": "audio"}) is False

    def test_matches_not_in(self):
        """matches() should handle NOT_IN operator."""
        filter = SearchFilter("type", SearchOperator.NOT_IN, ["temp", "backup"])

        assert filter.matches({"type": "texture"}) is True
        assert filter.matches({"type": "temp"}) is False

    def test_matches_regex(self):
        """matches() should handle MATCHES (regex) operator."""
        filter = SearchFilter("name", SearchOperator.MATCHES, r"hero_\d+")

        assert filter.matches({"name": "hero_001"}) is True
        assert filter.matches({"name": "hero_abc"}) is False

    def test_matches_exists(self):
        """matches() should handle EXISTS operator."""
        filter = SearchFilter("description", SearchOperator.EXISTS, None)

        assert filter.matches({"description": "some text"}) is True
        assert filter.matches({"name": "no_desc"}) is False
        assert filter.matches({"description": None}) is False

    def test_matches_not_exists(self):
        """matches() should handle NOT_EXISTS operator."""
        filter = SearchFilter("description", SearchOperator.NOT_EXISTS, None)

        assert filter.matches({"name": "no_desc"}) is True
        assert filter.matches({"description": None}) is True
        assert filter.matches({"description": "has desc"}) is False

    def test_nested_field(self):
        """matches() should handle nested fields."""
        filter = SearchFilter("metadata.author", SearchOperator.EQUALS, "john")

        assert filter.matches({"metadata": {"author": "john"}}) is True
        assert filter.matches({"metadata": {"author": "jane"}}) is False

    def test_to_dict(self):
        """to_dict() should serialize filter."""
        filter = SearchFilter(
            field="name",
            operator=SearchOperator.CONTAINS,
            value="hero",
            field_type=SearchFieldType.STRING,
        )

        data = filter.to_dict()

        assert data["field"] == "name"
        assert data["operator"] == "CONTAINS"
        assert data["value"] == "hero"

    def test_from_dict(self):
        """from_dict() should deserialize filter."""
        data = {
            "field": "size",
            "operator": "GREATER_THAN",
            "value": 1000,
            "field_type": "INTEGER",
        }

        filter = SearchFilter.from_dict(data)

        assert filter.field == "size"
        assert filter.operator == SearchOperator.GREATER_THAN
        assert filter.value == 1000


class TestSearchQuery:
    """Test SearchQuery dataclass."""

    def test_query_creation(self):
        """Query should store all attributes."""
        query = SearchQuery(
            text="hero",
            sort_field="name",
            limit=10,
        )

        assert query.text == "hero"
        assert query.sort_field == "name"
        assert query.limit == 10

    def test_query_defaults(self):
        """Query should have sensible defaults."""
        query = SearchQuery()

        assert query.text == ""
        assert len(query.filters) == 0
        assert query.sort_ascending is True
        assert query.offset == 0

    def test_add_filter(self):
        """add_filter() should add filter and return self."""
        query = SearchQuery()
        result = query.add_filter("type", SearchOperator.EQUALS, "texture")

        assert result is query
        assert len(query.filters) == 1

    def test_matches_text_search(self):
        """matches() should search in name, description, tags."""
        query = SearchQuery(text="hero")

        assert query.matches({"name": "hero_model"}) is True
        assert query.matches({"description": "The hero character"}) is True
        assert query.matches({"tags": ["character", "hero"]}) is True
        assert query.matches({"name": "villain", "description": "enemy"}) is False

    def test_matches_with_filters(self):
        """matches() should apply all filters."""
        query = SearchQuery(text="hero")
        query.add_filter("size", SearchOperator.GREATER_THAN, 1000)

        assert query.matches({"name": "hero", "size": 2000}) is True
        assert query.matches({"name": "hero", "size": 500}) is False

    def test_matches_include_paths(self):
        """matches() should filter by include paths."""
        query = SearchQuery(include_paths=[Path("/assets/characters")])

        assert query.matches({"path": "/assets/characters/hero.fbx"}) is True
        assert query.matches({"path": "/assets/props/sword.fbx"}) is False

    def test_matches_exclude_paths(self):
        """matches() should filter by exclude paths."""
        query = SearchQuery(exclude_paths=[Path("/assets/temp")])

        assert query.matches({"path": "/assets/characters/hero.fbx"}) is True
        assert query.matches({"path": "/assets/temp/backup.fbx"}) is False

    def test_to_dict(self):
        """to_dict() should serialize query."""
        query = SearchQuery(text="hero", limit=10)
        query.add_filter("type", SearchOperator.EQUALS, "texture")

        data = query.to_dict()

        assert data["text"] == "hero"
        assert data["limit"] == 10
        assert len(data["filters"]) == 1

    def test_from_dict(self):
        """from_dict() should deserialize query."""
        data = {
            "text": "hero",
            "filters": [{"field": "type", "operator": "EQUALS", "value": "texture"}],
            "limit": 10,
        }

        query = SearchQuery.from_dict(data)

        assert query.text == "hero"
        assert query.limit == 10
        assert len(query.filters) == 1

    def test_parse_simple_text(self):
        """parse() should handle simple text search."""
        query = SearchQuery.parse("hero character")

        assert "hero" in query.text
        assert "character" in query.text

    def test_parse_field_colon(self):
        """parse() should handle field:value syntax."""
        query = SearchQuery.parse("name:hero")

        assert len(query.filters) == 1
        assert query.filters[0].field == "name"

    def test_parse_tag_filter(self):
        """parse() should handle tag: syntax."""
        query = SearchQuery.parse("tag:character")

        assert len(query.filters) == 1
        assert query.filters[0].field == "tags"
        assert query.filters[0].operator == SearchOperator.CONTAINS

    def test_parse_type_filter(self):
        """parse() should handle type: syntax."""
        query = SearchQuery.parse("type:mesh")

        assert len(query.filters) == 1
        assert query.filters[0].field == "asset_type"
        assert query.filters[0].value == "MESH"

    def test_parse_ext_filter(self):
        """parse() should handle ext: syntax."""
        query = SearchQuery.parse("ext:fbx")

        assert len(query.filters) == 1
        assert query.filters[0].field == "extension"
        assert query.filters[0].value == "fbx"

    def test_parse_comparison_operators(self):
        """parse() should handle comparison operators."""
        query = SearchQuery.parse("size>1000")

        assert len(query.filters) == 1
        assert query.filters[0].operator == SearchOperator.GREATER_THAN
        assert query.filters[0].value == 1000

    def test_parse_mixed(self):
        """parse() should handle mixed query."""
        query = SearchQuery.parse("hero type:texture size>500")

        assert query.text == "hero"
        assert len(query.filters) == 2

    def test_parse_quoted_strings(self):
        """parse() should handle quoted strings."""
        query = SearchQuery.parse('name:"hero character"')

        assert len(query.filters) == 1
        assert query.filters[0].value == "hero character"


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_result_creation(self):
        """Result should store all attributes."""
        query = SearchQuery(text="hero", limit=10)
        result = SearchResult(
            query=query,
            items=[{"name": "hero1"}, {"name": "hero2"}],
            total_count=5,
            search_time_ms=10.5,
        )

        assert result.query == query
        assert len(result.items) == 2
        assert result.total_count == 5
        assert result.search_time_ms == 10.5

    def test_page_count(self):
        """page_count should calculate pages."""
        query = SearchQuery(limit=10)
        result = SearchResult(query=query, total_count=25)

        assert result.page_count == 3

    def test_page_count_no_limit(self):
        """page_count should be 1 with no limit."""
        query = SearchQuery()
        result = SearchResult(query=query, total_count=100)

        assert result.page_count == 1

    def test_current_page(self):
        """current_page should calculate current page."""
        query = SearchQuery(limit=10, offset=20)
        result = SearchResult(query=query)

        assert result.current_page == 3

    def test_has_more(self):
        """has_more should check for remaining results."""
        query = SearchQuery(limit=10)
        result = SearchResult(
            query=query,
            items=[{"name": str(i)} for i in range(10)],
            total_count=25,
        )

        assert result.has_more is True

        query2 = SearchQuery(limit=10, offset=20)
        result2 = SearchResult(
            query=query2,
            items=[{"name": str(i)} for i in range(5)],
            total_count=25,
        )

        assert result2.has_more is False


class TestSavedSearch:
    """Test SavedSearch dataclass."""

    def test_saved_search_creation(self):
        """Saved search should store all attributes."""
        query = SearchQuery(text="hero")
        saved = SavedSearch(
            id="search1",
            name="Hero Assets",
            description="All hero-related assets",
            query=query,
        )

        assert saved.id == "search1"
        assert saved.name == "Hero Assets"
        assert saved.query == query

    def test_saved_search_defaults(self):
        """Saved search should have sensible defaults."""
        saved = SavedSearch(id="s1", name="Test")

        assert saved.use_count == 0
        assert saved.is_favorite is False
        assert saved.last_used is None

    def test_to_dict(self):
        """to_dict() should serialize saved search."""
        saved = SavedSearch(
            id="search1",
            name="Test",
            query=SearchQuery(text="hero"),
        )

        data = saved.to_dict()

        assert data["id"] == "search1"
        assert data["name"] == "Test"
        assert data["query"]["text"] == "hero"

    def test_from_dict(self):
        """from_dict() should deserialize saved search."""
        data = {
            "id": "search1",
            "name": "Test",
            "query": {"text": "hero"},
            "use_count": 5,
        }

        saved = SavedSearch.from_dict(data)

        assert saved.id == "search1"
        assert saved.name == "Test"
        assert saved.query.text == "hero"
        assert saved.use_count == 5


class TestAssetSearch:
    """Test AssetSearch main class."""

    def test_search_creation(self, temp_search_dir):
        """Search engine should initialize correctly."""
        search = AssetSearch(storage_path=temp_search_dir / "storage")

        assert search.storage_path == temp_search_dir / "storage"

    def test_search_with_provider(self, temp_search_dir, sample_assets):
        """search() should use data provider."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider())

        result = search.search("hero")

        assert result.total_count == 2  # hero_diffuse and hero_normal

    def test_search_text(self, temp_search_dir, sample_assets):
        """search() should perform text search."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider())

        result = search.search("villain")

        assert result.total_count == 1
        assert result.items[0]["name"] == "villain_model"

    def test_search_with_query_object(self, temp_search_dir, sample_assets):
        """search() should accept SearchQuery object."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider())

        query = SearchQuery(text="hero")
        query.add_filter("size", SearchOperator.GREATER_THAN, 1500)

        result = search.search(query)

        assert result.total_count == 1
        assert result.items[0]["name"] == "hero_diffuse"

    def test_search_pagination(self, temp_search_dir, sample_assets):
        """search() should paginate results."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider())

        query = SearchQuery(limit=2)
        result = search.search(query)

        assert len(result.items) == 2
        assert result.total_count == 5
        assert result.has_more is True

    def test_search_sorting(self, temp_search_dir, sample_assets):
        """search() should sort results."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider())

        query = SearchQuery(sort_field="size", sort_ascending=False)
        result = search.search(query)

        # Largest first
        assert result.items[0]["name"] == "villain_model"

    def test_quick_search(self, temp_search_dir, sample_assets):
        """quick_search() should return limited results."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider())

        results = search.quick_search("hero", limit=1)

        assert len(results) == 1

    def test_index_asset(self, temp_search_dir):
        """index_asset() should add to search index."""
        search = AssetSearch()

        search.index_asset({
            "path": "/assets/test.png",
            "name": "test_texture",
            "description": "A test texture",
            "tags": ["test"],
        })

        stats = search.get_stats()
        assert stats["indexed_assets"] == 1
        assert stats["index_terms"] > 0

    def test_remove_from_index(self, temp_search_dir):
        """remove_from_index() should remove from index."""
        search = AssetSearch()

        search.index_asset({"path": "/assets/test.png", "name": "test"})
        search.remove_from_index("/assets/test.png")

        stats = search.get_stats()
        assert stats["indexed_assets"] == 0

    def test_clear_index(self, temp_search_dir):
        """clear_index() should clear all indexed data."""
        search = AssetSearch()

        search.index_asset({"path": "/assets/test1.png", "name": "test1"})
        search.index_asset({"path": "/assets/test2.png", "name": "test2"})

        search.clear_index()

        stats = search.get_stats()
        assert stats["indexed_assets"] == 0
        assert stats["index_terms"] == 0

    def test_rebuild_index(self, temp_search_dir, sample_assets):
        """rebuild_index() should reindex all assets."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider())

        count = search.rebuild_index()

        assert count == 5
        assert search.get_stats()["indexed_assets"] == 5

    def test_save_search(self, temp_search_dir):
        """save_search() should persist search."""
        search = AssetSearch(storage_path=temp_search_dir / "storage")

        query = SearchQuery(text="hero")
        saved = search.save_search("Hero Search", query, description="Find heroes")

        assert saved.id is not None
        assert saved.name == "Hero Search"

    def test_delete_saved_search(self, temp_search_dir):
        """delete_saved_search() should remove saved search."""
        search = AssetSearch(storage_path=temp_search_dir / "storage")

        query = SearchQuery(text="hero")
        saved = search.save_search("Hero Search", query)

        success = search.delete_saved_search(saved.id)
        assert success is True

        assert search.get_saved_search(saved.id) is None

    def test_get_saved_searches(self, temp_search_dir):
        """get_saved_searches() should list all saved searches."""
        search = AssetSearch(storage_path=temp_search_dir / "storage")

        search.save_search("Search1", SearchQuery(text="a"))
        search.save_search("Search2", SearchQuery(text="b"))

        saved = search.get_saved_searches()

        assert len(saved) == 2

    def test_get_saved_searches_favorites(self, temp_search_dir):
        """get_saved_searches(favorites_only=True) should filter."""
        search = AssetSearch(storage_path=temp_search_dir / "storage")

        s1 = search.save_search("Search1", SearchQuery())
        s1.is_favorite = True
        search.save_search("Search2", SearchQuery())

        favorites = search.get_saved_searches(favorites_only=True)

        assert len(favorites) == 1

    def test_run_saved_search(self, temp_search_dir, sample_assets):
        """run_saved_search() should execute saved search."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(
            data_provider=MockProvider(),
            storage_path=temp_search_dir / "storage",
        )

        query = SearchQuery(text="hero")
        saved = search.save_search("Hero Search", query)

        result = search.run_saved_search(saved.id)

        assert result is not None
        assert result.total_count == 2

        # Should update use count
        saved = search.get_saved_search(saved.id)
        assert saved.use_count == 1

    def test_search_history(self, temp_search_dir, sample_assets):
        """search() should track history."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider(), max_history=5)

        search.search("hero")
        search.search("villain")
        search.search("sword")

        history = search.get_search_history()

        assert len(history) == 3

    def test_search_history_limit(self, temp_search_dir, sample_assets):
        """Search history should respect max_history."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider(), max_history=2)

        search.search("a")
        search.search("b")
        search.search("c")

        history = search.get_search_history()

        assert len(history) == 2

    def test_clear_history(self, temp_search_dir, sample_assets):
        """clear_history() should clear search history."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider())

        search.search("hero")
        search.clear_history()

        assert len(search.get_search_history()) == 0

    def test_get_stats(self, temp_search_dir):
        """get_stats() should return statistics."""
        search = AssetSearch(storage_path=temp_search_dir / "storage")

        search.index_asset({"path": "/test.png", "name": "test"})
        search.save_search("Test", SearchQuery())

        stats = search.get_stats()

        assert stats["indexed_assets"] == 1
        assert stats["saved_searches"] == 1

    def test_suggestions(self, temp_search_dir, sample_assets):
        """search() should generate suggestions."""
        class MockProvider:
            def get_all_assets(self):
                return sample_assets

        search = AssetSearch(data_provider=MockProvider())

        result = search.search("hero")

        assert len(result.suggestions) > 0

    def test_persistence(self, temp_search_dir):
        """Saved searches should persist across instances."""
        storage_path = temp_search_dir / "storage"

        # Save in first instance
        search1 = AssetSearch(storage_path=storage_path)
        search1.save_search("Persistent", SearchQuery(text="test"))

        # Load in new instance
        search2 = AssetSearch(storage_path=storage_path)
        saved = search2.get_saved_searches()

        assert len(saved) == 1
        assert saved[0].name == "Persistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
