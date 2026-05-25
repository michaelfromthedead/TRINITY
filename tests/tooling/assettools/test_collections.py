"""
Comprehensive tests for AssetCollection functionality.

Tests collections, smart collections, nesting, and CollectionManager.
"""

import pytest
import sys
import tempfile
import shutil
import time
from pathlib import Path

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.assettools.collections import (
    CollectionType,
    CollectionQuery,
    AssetCollection,
    SmartCollection,
    CollectionManager,
)


@pytest.fixture
def temp_collections_dir():
    """Create a temporary directory for collection tests."""
    path = Path(tempfile.mkdtemp())

    # Create test asset paths
    (path / "assets").mkdir()
    (path / "assets" / "texture.png").write_bytes(b"png data")
    (path / "assets" / "texture2.png").write_bytes(b"png data 2")
    (path / "assets" / "model.fbx").write_bytes(b"fbx data")
    (path / "assets" / "audio.wav").write_bytes(b"wav data")

    (path / "storage").mkdir()

    yield path
    shutil.rmtree(path)


class TestCollectionType:
    """Test CollectionType enum."""

    def test_collection_types(self):
        """All collection types should be defined."""
        assert CollectionType.MANUAL
        assert CollectionType.SMART
        assert CollectionType.FOLDER


class TestCollectionQuery:
    """Test CollectionQuery dataclass."""

    def test_query_creation(self):
        """Query should store all attributes."""
        query = CollectionQuery(
            sort_by="name",
            sort_ascending=False,
            limit=10,
        )

        assert query.sort_by == "name"
        assert query.sort_ascending is False
        assert query.limit == 10

    def test_query_defaults(self):
        """Query should have sensible defaults."""
        query = CollectionQuery()

        assert len(query.filters) == 0
        assert query.sort_by == "name"
        assert query.sort_ascending is True
        assert query.limit is None

    def test_add_filter(self):
        """add_filter() should add filter and return self."""
        query = CollectionQuery()
        result = query.add_filter("type", "eq", "texture")

        assert result is query
        assert len(query.filters) == 1
        assert query.filters[0]["field"] == "type"
        assert query.filters[0]["operator"] == "eq"
        assert query.filters[0]["value"] == "texture"

    def test_add_multiple_filters(self):
        """add_filter() should support chaining."""
        query = (
            CollectionQuery()
            .add_filter("type", "eq", "texture")
            .add_filter("size", "gt", 1000)
        )

        assert len(query.filters) == 2

    def test_matches_eq(self):
        """matches() should handle eq operator."""
        query = CollectionQuery().add_filter("type", "eq", "texture")

        assert query.matches({"type": "texture"}) is True
        assert query.matches({"type": "mesh"}) is False

    def test_matches_ne(self):
        """matches() should handle ne operator."""
        query = CollectionQuery().add_filter("type", "ne", "texture")

        assert query.matches({"type": "mesh"}) is True
        assert query.matches({"type": "texture"}) is False

    def test_matches_gt(self):
        """matches() should handle gt operator."""
        query = CollectionQuery().add_filter("size", "gt", 100)

        assert query.matches({"size": 150}) is True
        assert query.matches({"size": 100}) is False
        assert query.matches({"size": 50}) is False

    def test_matches_lt(self):
        """matches() should handle lt operator."""
        query = CollectionQuery().add_filter("size", "lt", 100)

        assert query.matches({"size": 50}) is True
        assert query.matches({"size": 100}) is False

    def test_matches_gte(self):
        """matches() should handle gte operator."""
        query = CollectionQuery().add_filter("size", "gte", 100)

        assert query.matches({"size": 100}) is True
        assert query.matches({"size": 150}) is True
        assert query.matches({"size": 50}) is False

    def test_matches_lte(self):
        """matches() should handle lte operator."""
        query = CollectionQuery().add_filter("size", "lte", 100)

        assert query.matches({"size": 100}) is True
        assert query.matches({"size": 50}) is True
        assert query.matches({"size": 150}) is False

    def test_matches_contains(self):
        """matches() should handle contains operator."""
        query = CollectionQuery().add_filter("name", "contains", "hero")

        assert query.matches({"name": "hero_sword"}) is True
        assert query.matches({"name": "villain"}) is False

    def test_matches_in(self):
        """matches() should handle in operator."""
        query = CollectionQuery().add_filter("type", "in", ["texture", "mesh"])

        assert query.matches({"type": "texture"}) is True
        assert query.matches({"type": "mesh"}) is True
        assert query.matches({"type": "audio"}) is False

    def test_matches_startswith(self):
        """matches() should handle startswith operator."""
        query = CollectionQuery().add_filter("name", "startswith", "hero")

        assert query.matches({"name": "hero_sword"}) is True
        assert query.matches({"name": "sword_hero"}) is False

    def test_matches_endswith(self):
        """matches() should handle endswith operator."""
        query = CollectionQuery().add_filter("name", "endswith", "_diffuse")

        assert query.matches({"name": "hero_diffuse"}) is True
        assert query.matches({"name": "hero_normal"}) is False

    def test_matches_regex(self):
        """matches() should handle regex operator."""
        query = CollectionQuery().add_filter("name", "regex", r"hero_\d+")

        assert query.matches({"name": "hero_001"}) is True
        assert query.matches({"name": "hero_abc"}) is False

    def test_matches_multiple_filters(self):
        """matches() should require all filters to match."""
        query = (
            CollectionQuery()
            .add_filter("type", "eq", "texture")
            .add_filter("size", "gt", 100)
        )

        assert query.matches({"type": "texture", "size": 150}) is True
        assert query.matches({"type": "texture", "size": 50}) is False
        assert query.matches({"type": "mesh", "size": 150}) is False

    def test_to_dict(self):
        """to_dict() should serialize query."""
        query = CollectionQuery(
            sort_by="size",
            limit=10,
        ).add_filter("type", "eq", "texture")

        data = query.to_dict()

        assert data["sort_by"] == "size"
        assert data["limit"] == 10
        assert len(data["filters"]) == 1

    def test_from_dict(self):
        """from_dict() should deserialize query."""
        data = {
            "filters": [{"field": "type", "operator": "eq", "value": "texture"}],
            "sort_by": "size",
            "limit": 10,
        }

        query = CollectionQuery.from_dict(data)

        assert query.sort_by == "size"
        assert query.limit == 10
        assert len(query.filters) == 1


class TestAssetCollection:
    """Test AssetCollection dataclass."""

    def test_collection_creation(self):
        """Collection should store all attributes."""
        collection = AssetCollection(
            name="Characters",
            description="Character assets",
            color="#FF0000",
        )

        assert collection.name == "Characters"
        assert collection.description == "Character assets"
        assert collection.color == "#FF0000"

    def test_collection_defaults(self):
        """Collection should have sensible defaults."""
        collection = AssetCollection()

        assert len(collection.id) > 0
        assert collection.collection_type == CollectionType.MANUAL
        assert len(collection.assets) == 0
        assert collection.parent_id is None

    def test_is_smart(self):
        """is_smart should check collection type."""
        manual = AssetCollection(collection_type=CollectionType.MANUAL)
        smart = AssetCollection(collection_type=CollectionType.SMART)

        assert manual.is_smart is False
        assert smart.is_smart is True

    def test_is_nested(self):
        """is_nested should check parent."""
        root = AssetCollection()
        child = AssetCollection(parent_id="parent123")

        assert root.is_nested is False
        assert child.is_nested is True

    def test_has_children(self):
        """has_children should check children list."""
        parent = AssetCollection(children_ids=["child1", "child2"])
        leaf = AssetCollection()

        assert parent.has_children is True
        assert leaf.has_children is False

    def test_asset_count(self):
        """asset_count should return number of assets."""
        collection = AssetCollection()
        collection.assets = {Path("/a.png"), Path("/b.png")}

        assert collection.asset_count == 2

    def test_add_asset(self):
        """add_asset() should add asset paths."""
        collection = AssetCollection()

        success = collection.add_asset("/asset.png")
        assert success is True
        assert Path("/asset.png") in collection.assets

        # Adding again should fail
        success = collection.add_asset("/asset.png")
        assert success is False

    def test_add_asset_smart_fails(self):
        """add_asset() should fail for smart collections."""
        collection = AssetCollection(collection_type=CollectionType.SMART)

        success = collection.add_asset("/asset.png")
        assert success is False

    def test_remove_asset(self):
        """remove_asset() should remove asset paths."""
        collection = AssetCollection()
        collection.add_asset("/asset.png")

        success = collection.remove_asset("/asset.png")
        assert success is True
        assert Path("/asset.png") not in collection.assets

        # Removing again should fail
        success = collection.remove_asset("/asset.png")
        assert success is False

    def test_contains(self):
        """contains() should check asset membership."""
        collection = AssetCollection()
        collection.add_asset("/asset.png")

        assert collection.contains("/asset.png") is True
        assert collection.contains("/other.png") is False

    def test_to_dict(self):
        """to_dict() should serialize collection."""
        collection = AssetCollection(
            name="Test",
            description="Test collection",
        )
        collection.add_asset("/asset.png")

        data = collection.to_dict()

        assert data["name"] == "Test"
        assert "/asset.png" in data["assets"]

    def test_from_dict(self):
        """from_dict() should deserialize collection."""
        data = {
            "id": "coll123",
            "name": "Test",
            "collection_type": "MANUAL",
            "assets": ["/asset.png", "/model.fbx"],
        }

        collection = AssetCollection.from_dict(data)

        assert collection.id == "coll123"
        assert collection.name == "Test"
        assert collection.asset_count == 2


class TestSmartCollection:
    """Test SmartCollection functionality."""

    def test_smart_collection_creation(self):
        """Smart collection should require query."""
        query = CollectionQuery().add_filter("type", "eq", "texture")
        collection = SmartCollection(name="Textures", query=query)

        assert collection.is_smart is True
        assert collection.query == query

    def test_evaluate(self):
        """evaluate() should filter items by query."""
        query = CollectionQuery().add_filter("type", "eq", "texture")
        collection = SmartCollection(name="Textures", query=query)

        items = [
            {"name": "tex1", "type": "texture"},
            {"name": "mesh1", "type": "mesh"},
            {"name": "tex2", "type": "texture"},
        ]

        matching = collection.evaluate(items)

        assert len(matching) == 2
        assert all(item["type"] == "texture" for item in matching)

    def test_evaluate_with_sort(self):
        """evaluate() should sort results."""
        query = CollectionQuery(sort_by="name", sort_ascending=True)
        collection = SmartCollection(name="All", query=query)

        items = [
            {"name": "charlie"},
            {"name": "alpha"},
            {"name": "bravo"},
        ]

        matching = collection.evaluate(items)

        assert matching[0]["name"] == "alpha"
        assert matching[1]["name"] == "bravo"
        assert matching[2]["name"] == "charlie"

    def test_evaluate_with_limit(self):
        """evaluate() should limit results."""
        query = CollectionQuery(limit=2)
        collection = SmartCollection(name="Limited", query=query)

        items = [{"name": str(i)} for i in range(10)]

        matching = collection.evaluate(items)

        assert len(matching) == 2


class TestCollectionManager:
    """Test CollectionManager functionality."""

    def test_manager_creation(self, temp_collections_dir):
        """Manager should initialize correctly."""
        manager = CollectionManager(temp_collections_dir / "storage")

        assert manager.storage_path == temp_collections_dir / "storage"

    def test_create_collection(self, temp_collections_dir):
        """create_collection() should create and store collection."""
        manager = CollectionManager(temp_collections_dir / "storage")

        collection = manager.create_collection(
            name="Characters",
            description="Character assets",
        )

        assert collection.name == "Characters"
        assert collection.id in [c.id for c in manager.get_all_collections()]

    def test_create_smart_collection(self, temp_collections_dir):
        """create_smart_collection() should create smart collection."""
        manager = CollectionManager(temp_collections_dir / "storage")

        query = CollectionQuery().add_filter("type", "eq", "texture")
        collection = manager.create_smart_collection(
            name="Textures",
            query=query,
        )

        assert collection.is_smart is True
        assert collection.query == query

    def test_get_collection(self, temp_collections_dir):
        """get_collection() should retrieve by ID."""
        manager = CollectionManager(temp_collections_dir / "storage")

        created = manager.create_collection(name="Test")
        retrieved = manager.get_collection(created.id)

        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_get_collection_by_name(self, temp_collections_dir):
        """get_collection_by_name() should retrieve by name."""
        manager = CollectionManager(temp_collections_dir / "storage")

        manager.create_collection(name="Characters")
        retrieved = manager.get_collection_by_name("Characters")

        assert retrieved is not None
        assert retrieved.name == "Characters"

    def test_update_collection(self, temp_collections_dir):
        """update_collection() should update and persist."""
        manager = CollectionManager(temp_collections_dir / "storage")

        collection = manager.create_collection(name="Original")
        collection.name = "Updated"

        success = manager.update_collection(collection)
        assert success is True

        retrieved = manager.get_collection(collection.id)
        assert retrieved.name == "Updated"

    def test_delete_collection(self, temp_collections_dir):
        """delete_collection() should remove collection."""
        manager = CollectionManager(temp_collections_dir / "storage")

        collection = manager.create_collection(name="ToDelete")
        collection_id = collection.id

        success = manager.delete_collection(collection_id)
        assert success is True

        assert manager.get_collection(collection_id) is None

    def test_nested_collections(self, temp_collections_dir):
        """Collections should support nesting."""
        manager = CollectionManager(temp_collections_dir / "storage")

        parent = manager.create_collection(name="Parent")
        child = manager.create_collection(name="Child", parent_id=parent.id)

        assert child.parent_id == parent.id
        assert child.id in parent.children_ids

    def test_get_root_collections(self, temp_collections_dir):
        """get_root_collections() should return only root collections."""
        manager = CollectionManager(temp_collections_dir / "storage")

        root1 = manager.create_collection(name="Root1")
        root2 = manager.create_collection(name="Root2")
        child = manager.create_collection(name="Child", parent_id=root1.id)

        roots = manager.get_root_collections()

        assert len(roots) == 2
        assert all(c.id in [root1.id, root2.id] for c in roots)

    def test_get_children(self, temp_collections_dir):
        """get_children() should return child collections."""
        manager = CollectionManager(temp_collections_dir / "storage")

        parent = manager.create_collection(name="Parent")
        child1 = manager.create_collection(name="Child1", parent_id=parent.id)
        child2 = manager.create_collection(name="Child2", parent_id=parent.id)

        children = manager.get_children(parent.id)

        assert len(children) == 2

    def test_get_ancestors(self, temp_collections_dir):
        """get_ancestors() should return ancestor chain."""
        manager = CollectionManager(temp_collections_dir / "storage")

        grandparent = manager.create_collection(name="Grandparent")
        parent = manager.create_collection(name="Parent", parent_id=grandparent.id)
        child = manager.create_collection(name="Child", parent_id=parent.id)

        ancestors = manager.get_ancestors(child.id)

        assert len(ancestors) == 2
        assert ancestors[0].id == grandparent.id
        assert ancestors[1].id == parent.id

    def test_move_collection(self, temp_collections_dir):
        """move_collection() should reparent collection."""
        manager = CollectionManager(temp_collections_dir / "storage")

        parent1 = manager.create_collection(name="Parent1")
        parent2 = manager.create_collection(name="Parent2")
        child = manager.create_collection(name="Child", parent_id=parent1.id)

        success = manager.move_collection(child.id, parent2.id)
        assert success is True

        child = manager.get_collection(child.id)
        assert child.parent_id == parent2.id
        assert child.id in parent2.children_ids
        assert child.id not in manager.get_collection(parent1.id).children_ids

    def test_move_collection_to_root(self, temp_collections_dir):
        """move_collection() should support moving to root."""
        manager = CollectionManager(temp_collections_dir / "storage")

        parent = manager.create_collection(name="Parent")
        child = manager.create_collection(name="Child", parent_id=parent.id)

        success = manager.move_collection(child.id, None)
        assert success is True

        child = manager.get_collection(child.id)
        assert child.parent_id is None

    def test_move_prevents_circular(self, temp_collections_dir):
        """move_collection() should prevent circular references."""
        manager = CollectionManager(temp_collections_dir / "storage")

        parent = manager.create_collection(name="Parent")
        child = manager.create_collection(name="Child", parent_id=parent.id)

        # Try to move parent under child
        success = manager.move_collection(parent.id, child.id)
        assert success is False

    def test_add_asset_to_collection(self, temp_collections_dir):
        """add_asset_to_collection() should add assets."""
        manager = CollectionManager(temp_collections_dir / "storage")
        asset_path = temp_collections_dir / "assets" / "texture.png"

        collection = manager.create_collection(name="Test")
        success = manager.add_asset_to_collection(collection.id, asset_path)

        assert success is True

        collection = manager.get_collection(collection.id)
        assert collection.contains(asset_path)

    def test_remove_asset_from_collection(self, temp_collections_dir):
        """remove_asset_from_collection() should remove assets."""
        manager = CollectionManager(temp_collections_dir / "storage")
        asset_path = temp_collections_dir / "assets" / "texture.png"

        collection = manager.create_collection(name="Test")
        manager.add_asset_to_collection(collection.id, asset_path)

        success = manager.remove_asset_from_collection(collection.id, asset_path)
        assert success is True

        collection = manager.get_collection(collection.id)
        assert not collection.contains(asset_path)

    def test_get_assets_in_collection(self, temp_collections_dir):
        """get_assets_in_collection() should return assets."""
        manager = CollectionManager(temp_collections_dir / "storage")

        collection = manager.create_collection(name="Test")
        manager.add_asset_to_collection(collection.id, "/asset1.png")
        manager.add_asset_to_collection(collection.id, "/asset2.png")

        assets = manager.get_assets_in_collection(collection.id)

        assert len(assets) == 2

    def test_get_assets_include_children(self, temp_collections_dir):
        """get_assets_in_collection() should include children assets."""
        manager = CollectionManager(temp_collections_dir / "storage")

        parent = manager.create_collection(name="Parent")
        child = manager.create_collection(name="Child", parent_id=parent.id)

        manager.add_asset_to_collection(parent.id, "/parent.png")
        manager.add_asset_to_collection(child.id, "/child.png")

        assets = manager.get_assets_in_collection(parent.id, include_children=True)

        assert len(assets) == 2

    def test_get_collections_for_asset(self, temp_collections_dir):
        """get_collections_for_asset() should find containing collections."""
        manager = CollectionManager(temp_collections_dir / "storage")
        asset_path = temp_collections_dir / "assets" / "texture.png"

        coll1 = manager.create_collection(name="Coll1")
        coll2 = manager.create_collection(name="Coll2")
        coll3 = manager.create_collection(name="Coll3")

        manager.add_asset_to_collection(coll1.id, asset_path)
        manager.add_asset_to_collection(coll2.id, asset_path)

        collections = manager.get_collections_for_asset(asset_path)

        assert len(collections) == 2
        assert all(c.id in [coll1.id, coll2.id] for c in collections)

    def test_search_collections(self, temp_collections_dir):
        """search_collections() should search by name/description."""
        manager = CollectionManager(temp_collections_dir / "storage")

        manager.create_collection(name="Hero Characters")
        manager.create_collection(name="Villain Characters")
        manager.create_collection(name="Environments")

        results = manager.search_collections("character")

        assert len(results) == 2

    def test_on_change_callback(self, temp_collections_dir):
        """on_change() should notify on changes."""
        manager = CollectionManager(temp_collections_dir / "storage")
        changes = []

        manager.on_change(lambda c, a: changes.append((c, a)))

        collection = manager.create_collection(name="Test")

        assert len(changes) == 1
        assert changes[0][1] == "created"

    def test_get_stats(self, temp_collections_dir):
        """get_stats() should return statistics."""
        manager = CollectionManager(temp_collections_dir / "storage")

        manager.create_collection(name="Manual1")
        manager.create_collection(name="Manual2")
        manager.create_smart_collection(name="Smart1", query=CollectionQuery())

        stats = manager.get_stats()

        assert stats["total_collections"] == 3
        assert stats["manual_collections"] == 2
        assert stats["smart_collections"] == 1

    def test_delete_recursive(self, temp_collections_dir):
        """delete_collection(recursive=True) should delete children."""
        manager = CollectionManager(temp_collections_dir / "storage")

        parent = manager.create_collection(name="Parent")
        child1 = manager.create_collection(name="Child1", parent_id=parent.id)
        child2 = manager.create_collection(name="Child2", parent_id=parent.id)

        success = manager.delete_collection(parent.id, recursive=True)
        assert success is True

        assert manager.get_collection(parent.id) is None
        assert manager.get_collection(child1.id) is None
        assert manager.get_collection(child2.id) is None

    def test_delete_moves_children_to_root(self, temp_collections_dir):
        """delete_collection(recursive=False) should move children to root."""
        manager = CollectionManager(temp_collections_dir / "storage")

        parent = manager.create_collection(name="Parent")
        child = manager.create_collection(name="Child", parent_id=parent.id)
        child_id = child.id

        success = manager.delete_collection(parent.id, recursive=False)
        assert success is True

        child = manager.get_collection(child_id)
        assert child is not None
        assert child.parent_id is None

    def test_persistence(self, temp_collections_dir):
        """Collections should persist across manager instances."""
        storage_path = temp_collections_dir / "storage"

        # Create collections
        manager1 = CollectionManager(storage_path)
        collection = manager1.create_collection(name="Persistent")
        manager1.add_asset_to_collection(collection.id, "/asset.png")
        collection_id = collection.id

        # Load in new manager
        manager2 = CollectionManager(storage_path)
        loaded = manager2.get_collection(collection_id)

        assert loaded is not None
        assert loaded.name == "Persistent"
        assert loaded.contains("/asset.png")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
