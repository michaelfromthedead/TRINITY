"""
Comprehensive tests for AssetMetadata functionality.

Tests metadata properties, tags, schemas, and the MetadataEditor.
"""

import pytest
import sys
import tempfile
import shutil
import time
from pathlib import Path

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.assettools.metadata import (
    PropertyType,
    MetadataProperty,
    MetadataTag,
    MetadataSchema,
    AssetMetadata,
    MetadataEditor,
)


@pytest.fixture
def temp_metadata_dir():
    """Create a temporary directory for metadata tests."""
    path = Path(tempfile.mkdtemp())

    # Create test asset files
    (path / "assets").mkdir()
    (path / "assets" / "texture.png").write_bytes(b"png data")
    (path / "assets" / "model.fbx").write_bytes(b"fbx data")
    (path / "assets" / "audio.wav").write_bytes(b"wav data")

    (path / "metadata").mkdir()

    yield path
    shutil.rmtree(path)


class TestPropertyType:
    """Test PropertyType enum."""

    def test_property_types(self):
        """All property types should be defined."""
        assert PropertyType.STRING
        assert PropertyType.INTEGER
        assert PropertyType.FLOAT
        assert PropertyType.BOOLEAN
        assert PropertyType.DATE
        assert PropertyType.COLOR
        assert PropertyType.VECTOR2
        assert PropertyType.VECTOR3
        assert PropertyType.PATH
        assert PropertyType.ENUM
        assert PropertyType.LIST
        assert PropertyType.DICT


class TestMetadataProperty:
    """Test MetadataProperty dataclass."""

    def test_property_creation(self):
        """Property should store all attributes."""
        prop = MetadataProperty(
            name="quality",
            property_type=PropertyType.INTEGER,
            value=100,
            default_value=50,
            description="Asset quality level",
        )

        assert prop.name == "quality"
        assert prop.property_type == PropertyType.INTEGER
        assert prop.value == 100
        assert prop.default_value == 50

    def test_property_defaults(self):
        """Property should have sensible defaults."""
        prop = MetadataProperty(
            name="test",
            property_type=PropertyType.STRING,
        )

        assert prop.value is None
        assert prop.default_value is None
        assert prop.required is False
        assert prop.readonly is False

    def test_validate_string(self):
        """Validation should check string type."""
        prop = MetadataProperty(
            name="name",
            property_type=PropertyType.STRING,
            value="test",
        )

        valid, error = prop.validate()
        assert valid is True
        assert error is None

    def test_validate_string_wrong_type(self):
        """Validation should fail for wrong type."""
        prop = MetadataProperty(
            name="name",
            property_type=PropertyType.STRING,
            value=123,  # Wrong type
        )

        valid, error = prop.validate()
        assert valid is False
        assert "invalid type" in error.lower()

    def test_validate_integer(self):
        """Validation should check integer type."""
        prop = MetadataProperty(
            name="count",
            property_type=PropertyType.INTEGER,
            value=42,
        )

        valid, error = prop.validate()
        assert valid is True

    def test_validate_float(self):
        """Validation should accept both int and float for FLOAT type."""
        prop = MetadataProperty(
            name="scale",
            property_type=PropertyType.FLOAT,
            value=1.5,
        )

        valid, _ = prop.validate()
        assert valid is True

        prop.value = 2  # int is also valid for FLOAT
        valid, _ = prop.validate()
        assert valid is True

    def test_validate_boolean(self):
        """Validation should check boolean type."""
        prop = MetadataProperty(
            name="enabled",
            property_type=PropertyType.BOOLEAN,
            value=True,
        )

        valid, _ = prop.validate()
        assert valid is True

    def test_validate_required(self):
        """Validation should fail for missing required values."""
        prop = MetadataProperty(
            name="required_field",
            property_type=PropertyType.STRING,
            value=None,
            required=True,
        )

        valid, error = prop.validate()
        assert valid is False
        assert "required" in error.lower()

    def test_validate_enum(self):
        """Validation should check enum values."""
        prop = MetadataProperty(
            name="status",
            property_type=PropertyType.ENUM,
            value="active",
            enum_values=["active", "inactive", "pending"],
        )

        valid, _ = prop.validate()
        assert valid is True

        prop.value = "invalid"
        valid, error = prop.validate()
        assert valid is False
        assert "not in allowed" in error.lower()

    def test_validate_vector2(self):
        """Validation should check vector2 type."""
        prop = MetadataProperty(
            name="position",
            property_type=PropertyType.VECTOR2,
            value=[1.0, 2.0],
        )

        valid, _ = prop.validate()
        assert valid is True

        prop.value = [1.0, 2.0, 3.0]  # Wrong length
        valid, _ = prop.validate()
        assert valid is False

    def test_validate_vector3(self):
        """Validation should check vector3 type."""
        prop = MetadataProperty(
            name="position",
            property_type=PropertyType.VECTOR3,
            value=[1.0, 2.0, 3.0],
        )

        valid, _ = prop.validate()
        assert valid is True

    def test_validate_color(self):
        """Validation should check color type."""
        prop = MetadataProperty(
            name="color",
            property_type=PropertyType.COLOR,
            value=[255, 128, 64],  # RGB
        )

        valid, _ = prop.validate()
        assert valid is True

        prop.value = [255, 128, 64, 200]  # RGBA
        valid, _ = prop.validate()
        assert valid is True

    def test_validate_custom_validator(self):
        """Validation should run custom validators."""
        prop = MetadataProperty(
            name="age",
            property_type=PropertyType.INTEGER,
            value=25,
            validators=[lambda x: x >= 0, lambda x: x <= 150],
        )

        valid, _ = prop.validate()
        assert valid is True

        prop.value = -5
        valid, error = prop.validate()
        assert valid is False
        assert "failed validation" in error.lower()

    def test_set_value(self):
        """set_value() should validate before setting."""
        prop = MetadataProperty(
            name="name",
            property_type=PropertyType.STRING,
            value="old",
        )

        success = prop.set_value("new")
        assert success is True
        assert prop.value == "new"

    def test_set_value_readonly(self):
        """set_value() should fail for readonly properties."""
        prop = MetadataProperty(
            name="id",
            property_type=PropertyType.STRING,
            value="original",
            readonly=True,
        )

        success = prop.set_value("modified")
        assert success is False
        assert prop.value == "original"

    def test_set_value_invalid(self):
        """set_value() should revert on validation failure."""
        prop = MetadataProperty(
            name="count",
            property_type=PropertyType.INTEGER,
            value=10,
        )

        success = prop.set_value("not an integer")
        assert success is False
        assert prop.value == 10

    def test_reset(self):
        """reset() should restore default value."""
        prop = MetadataProperty(
            name="quality",
            property_type=PropertyType.INTEGER,
            value=100,
            default_value=50,
        )

        prop.reset()
        assert prop.value == 50


class TestMetadataTag:
    """Test MetadataTag dataclass."""

    def test_tag_creation(self):
        """Tag should store all attributes."""
        tag = MetadataTag(
            name="character",
            color="#FF0000",
            category="Assets",
            description="Character assets",
        )

        assert tag.name == "character"
        assert tag.color == "#FF0000"
        assert tag.category == "Assets"

    def test_tag_defaults(self):
        """Tag should have sensible defaults."""
        tag = MetadataTag(name="test")

        assert tag.color == "#808080"
        assert tag.category == "General"
        assert tag.count == 0

    def test_tag_hash(self):
        """Tags should be hashable by name."""
        tag1 = MetadataTag(name="test")
        tag2 = MetadataTag(name="test")
        tag3 = MetadataTag(name="other")

        assert hash(tag1) == hash(tag2)
        assert hash(tag1) != hash(tag3)

    def test_tag_equality(self):
        """Tags should compare by name."""
        tag1 = MetadataTag(name="test", color="#FF0000")
        tag2 = MetadataTag(name="test", color="#00FF00")
        tag3 = MetadataTag(name="other")

        assert tag1 == tag2
        assert tag1 != tag3

    def test_tag_string_equality(self):
        """Tags should equal string of same name."""
        tag = MetadataTag(name="test")

        assert tag == "test"
        assert tag != "other"


class TestMetadataSchema:
    """Test MetadataSchema dataclass."""

    def test_schema_creation(self):
        """Schema should store all attributes."""
        schema = MetadataSchema(
            name="texture_schema",
            version="1.0",
            category="Textures",
        )

        assert schema.name == "texture_schema"
        assert schema.version == "1.0"

    def test_schema_properties(self):
        """Schema should store property definitions."""
        schema = MetadataSchema(
            name="asset_schema",
            properties=[
                MetadataProperty(name="quality", property_type=PropertyType.INTEGER),
                MetadataProperty(name="description", property_type=PropertyType.STRING),
            ],
        )

        assert len(schema.properties) == 2

    def test_get_property(self):
        """get_property() should find property by name."""
        schema = MetadataSchema(
            name="test",
            properties=[
                MetadataProperty(name="quality", property_type=PropertyType.INTEGER),
            ],
        )

        prop = schema.get_property("quality")
        assert prop is not None
        assert prop.name == "quality"

        missing = schema.get_property("missing")
        assert missing is None

    def test_validate_required_properties(self):
        """validate() should check required properties."""
        schema = MetadataSchema(
            name="test",
            properties=[
                MetadataProperty(name="required_field", property_type=PropertyType.STRING, required=True),
            ],
        )

        metadata = AssetMetadata(asset_path=Path("/test.png"))
        errors = schema.validate(metadata)

        assert len(errors) == 1
        assert "required_field" in errors[0]

    def test_validate_required_tags(self):
        """validate() should check required tags."""
        schema = MetadataSchema(
            name="test",
            required_tags=["character"],
        )

        metadata = AssetMetadata(asset_path=Path("/test.png"))
        errors = schema.validate(metadata)

        assert len(errors) == 1
        assert "character" in errors[0]

        metadata.add_tag("character")
        errors = schema.validate(metadata)
        assert len(errors) == 0


class TestAssetMetadata:
    """Test AssetMetadata dataclass."""

    def test_metadata_creation(self):
        """Metadata should store all attributes."""
        metadata = AssetMetadata(
            asset_path=Path("/assets/texture.png"),
            description="A texture asset",
        )

        assert metadata.asset_path == Path("/assets/texture.png")
        assert metadata.description == "A texture asset"

    def test_metadata_defaults(self):
        """Metadata should have sensible defaults."""
        metadata = AssetMetadata(asset_path=Path("/test.png"))

        assert len(metadata.properties) == 0
        assert len(metadata.tags) == 0
        assert metadata.created_at > 0
        assert metadata.modified_at > 0

    def test_get_set_property(self):
        """Properties should be gettable and settable."""
        metadata = AssetMetadata(asset_path=Path("/test.png"))

        metadata.set_property("quality", 100)
        assert metadata.get_property("quality") == 100
        assert metadata.get_property("missing", "default") == "default"

    def test_set_property_updates_modified(self):
        """Setting property should update modified_at."""
        metadata = AssetMetadata(asset_path=Path("/test.png"))
        original_modified = metadata.modified_at

        time.sleep(0.01)
        metadata.set_property("key", "value")

        assert metadata.modified_at > original_modified

    def test_remove_property(self):
        """remove_property() should remove and return success."""
        metadata = AssetMetadata(asset_path=Path("/test.png"))
        metadata.set_property("key", "value")

        success = metadata.remove_property("key")
        assert success is True
        assert metadata.get_property("key") is None

        success = metadata.remove_property("missing")
        assert success is False

    def test_add_tag(self):
        """add_tag() should add tags."""
        metadata = AssetMetadata(asset_path=Path("/test.png"))

        success = metadata.add_tag("character")
        assert success is True
        assert "character" in metadata.tags

        # Adding again should fail
        success = metadata.add_tag("character")
        assert success is False

    def test_add_tag_object(self):
        """add_tag() should accept MetadataTag objects."""
        metadata = AssetMetadata(asset_path=Path("/test.png"))
        tag = MetadataTag(name="environment")

        success = metadata.add_tag(tag)
        assert success is True
        assert "environment" in metadata.tags

    def test_remove_tag(self):
        """remove_tag() should remove tags."""
        metadata = AssetMetadata(asset_path=Path("/test.png"))
        metadata.add_tag("character")

        success = metadata.remove_tag("character")
        assert success is True
        assert "character" not in metadata.tags

        success = metadata.remove_tag("missing")
        assert success is False

    def test_has_tag(self):
        """has_tag() should check tag presence."""
        metadata = AssetMetadata(asset_path=Path("/test.png"))
        metadata.add_tag("character")

        assert metadata.has_tag("character") is True
        assert metadata.has_tag("missing") is False

    def test_to_dict(self):
        """to_dict() should serialize metadata."""
        metadata = AssetMetadata(
            asset_path=Path("/test.png"),
            description="Test asset",
        )
        metadata.set_property("quality", 100)
        metadata.add_tag("character")

        data = metadata.to_dict()

        assert data["asset_path"] == "/test.png"
        assert data["description"] == "Test asset"
        assert data["properties"]["quality"] == 100
        assert "character" in data["tags"]

    def test_from_dict(self):
        """from_dict() should deserialize metadata."""
        data = {
            "asset_path": "/test.png",
            "description": "Test asset",
            "properties": {"quality": 100},
            "tags": ["character", "hero"],
        }

        metadata = AssetMetadata.from_dict(data)

        assert metadata.asset_path == Path("/test.png")
        assert metadata.description == "Test asset"
        assert metadata.get_property("quality") == 100
        assert metadata.has_tag("character")


class TestMetadataEditor:
    """Test MetadataEditor functionality."""

    def test_editor_creation(self, temp_metadata_dir):
        """Editor should initialize correctly."""
        editor = MetadataEditor(
            root_path=temp_metadata_dir / "assets",
            metadata_directory=temp_metadata_dir / "metadata",
        )

        assert editor.root_path == temp_metadata_dir / "assets"
        assert editor.metadata_directory == temp_metadata_dir / "metadata"

    def test_get_metadata_create(self, temp_metadata_dir):
        """get_metadata() should create new metadata."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        metadata = editor.get_metadata(asset_path)

        assert metadata is not None
        assert metadata.asset_path == asset_path

    def test_get_metadata_no_create(self, temp_metadata_dir):
        """get_metadata(create=False) should return None if not exists."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        metadata = editor.get_metadata(asset_path, create=False)

        assert metadata is None

    def test_save_metadata(self, temp_metadata_dir):
        """save_metadata() should persist to disk."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        metadata = editor.get_metadata(asset_path)
        metadata.set_property("quality", 100)

        success = editor.save_metadata(metadata)
        assert success is True

        # Clear cache and reload
        editor.clear_cache()
        reloaded = editor.get_metadata(asset_path)

        assert reloaded.get_property("quality") == 100

    def test_delete_metadata(self, temp_metadata_dir):
        """delete_metadata() should remove metadata."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        metadata = editor.get_metadata(asset_path)
        editor.save_metadata(metadata)

        success = editor.delete_metadata(asset_path)
        assert success is True

        assert editor.get_metadata(asset_path, create=False) is None

    def test_set_property(self, temp_metadata_dir):
        """set_property() should set and save."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        success = editor.set_property(asset_path, "quality", 100)
        assert success is True

        assert editor.get_property(asset_path, "quality") == 100

    def test_get_property(self, temp_metadata_dir):
        """get_property() should retrieve property values."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        editor.set_property(asset_path, "quality", 100)

        assert editor.get_property(asset_path, "quality") == 100
        assert editor.get_property(asset_path, "missing", "default") == "default"

    def test_add_tag(self, temp_metadata_dir):
        """add_tag() should add tag to asset."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        success = editor.add_tag(asset_path, "character")
        assert success is True

        metadata = editor.get_metadata(asset_path)
        assert metadata.has_tag("character")

    def test_remove_tag(self, temp_metadata_dir):
        """remove_tag() should remove tag from asset."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        editor.add_tag(asset_path, "character")
        success = editor.remove_tag(asset_path, "character")

        assert success is True

        metadata = editor.get_metadata(asset_path)
        assert not metadata.has_tag("character")

    def test_register_tag(self, temp_metadata_dir):
        """register_tag() should add tag to registry."""
        editor = MetadataEditor(temp_metadata_dir / "assets")

        tag = MetadataTag(name="hero", color="#FF0000", category="Characters")
        editor.register_tag(tag)

        tags = editor.get_all_tags()
        assert any(t.name == "hero" for t in tags)

    def test_unregister_tag(self, temp_metadata_dir):
        """unregister_tag() should remove from registry."""
        editor = MetadataEditor(temp_metadata_dir / "assets")

        tag = MetadataTag(name="hero")
        editor.register_tag(tag)

        success = editor.unregister_tag("hero")
        assert success is True

        tags = editor.get_all_tags()
        assert not any(t.name == "hero" for t in tags)

    def test_get_tags_by_category(self, temp_metadata_dir):
        """get_tags_by_category() should filter tags."""
        editor = MetadataEditor(temp_metadata_dir / "assets")

        editor.register_tag(MetadataTag(name="hero", category="Characters"))
        editor.register_tag(MetadataTag(name="tree", category="Environment"))
        editor.register_tag(MetadataTag(name="villain", category="Characters"))

        char_tags = editor.get_tags_by_category("Characters")
        assert len(char_tags) == 2
        assert all(t.category == "Characters" for t in char_tags)

    def test_get_assets_by_tag(self, temp_metadata_dir):
        """get_assets_by_tag() should find assets with tag."""
        editor = MetadataEditor(temp_metadata_dir / "assets")

        asset1 = temp_metadata_dir / "assets" / "texture.png"
        asset2 = temp_metadata_dir / "assets" / "model.fbx"

        editor.add_tag(asset1, "character")
        editor.add_tag(asset2, "character")

        assets = editor.get_assets_by_tag("character")
        assert len(assets) == 2

    def test_register_schema(self, temp_metadata_dir):
        """register_schema() should add schema."""
        editor = MetadataEditor(temp_metadata_dir / "assets")

        schema = MetadataSchema(
            name="texture_schema",
            properties=[
                MetadataProperty(name="resolution", property_type=PropertyType.INTEGER, required=True),
            ],
        )

        editor.register_schema(schema)

    def test_validate_against_schema(self, temp_metadata_dir):
        """validate_against_schema() should validate metadata."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        schema = MetadataSchema(
            name="texture_schema",
            properties=[
                MetadataProperty(name="resolution", property_type=PropertyType.INTEGER, required=True),
            ],
        )
        editor.register_schema(schema)

        # Create metadata without required property
        editor.get_metadata(asset_path)

        errors = editor.validate_against_schema(asset_path, "texture_schema")
        assert len(errors) > 0

    def test_on_change_callback(self, temp_metadata_dir):
        """on_change() should notify on changes."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"
        changes = []

        editor.on_change(lambda m, a: changes.append((m, a)))

        metadata = editor.get_metadata(asset_path)
        editor.save_metadata(metadata)

        assert len(changes) == 1
        assert changes[0][1] == "saved"

    def test_get_stats(self, temp_metadata_dir):
        """get_stats() should return statistics."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        editor.get_metadata(asset_path)

        stats = editor.get_stats()

        assert "total_metadata" in stats
        assert "cached_metadata" in stats
        assert "registered_tags" in stats
        assert "registered_schemas" in stats

    def test_clear_cache(self, temp_metadata_dir):
        """clear_cache() should clear cached metadata."""
        editor = MetadataEditor(temp_metadata_dir / "assets")
        asset_path = temp_metadata_dir / "assets" / "texture.png"

        editor.get_metadata(asset_path)
        assert editor.get_stats()["cached_metadata"] == 1

        editor.clear_cache()
        assert editor.get_stats()["cached_metadata"] == 0

    def test_content_store_integration(self, temp_metadata_dir):
        """Editor should accept content store."""
        # Mock content store
        class MockContentStore:
            def put(self, obj):
                return "hash123"

            def get(self, hash):
                return None

            def has(self, hash):
                return False

        editor = MetadataEditor(
            temp_metadata_dir / "assets",
            content_store=MockContentStore(),
        )

        assert editor.content_store is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
