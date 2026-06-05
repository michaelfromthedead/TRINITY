"""
AssetMetadata - Asset metadata editing, tagging, and custom properties.

Provides comprehensive metadata management:
- Tag-based organization
- Custom property support
- Schema validation
- Metadata persistence
- Integration with ContentStore
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union

from trinity.decorators.dev import editor


class PropertyType(Enum):
    """Types of metadata properties."""

    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    DATE = auto()
    COLOR = auto()
    VECTOR2 = auto()
    VECTOR3 = auto()
    PATH = auto()
    ENUM = auto()
    LIST = auto()
    DICT = auto()


@dataclass
class MetadataProperty:
    """Definition of a metadata property.

    Attributes:
        name: Property name
        property_type: Type of the property
        value: Current value
        default_value: Default value
        description: Property description
        required: Whether property is required
        readonly: Whether property is readonly
        validators: List of validator functions
        enum_values: Valid values for ENUM type
        metadata: Additional property metadata
    """

    name: str
    property_type: PropertyType
    value: Any = None
    default_value: Any = None
    description: str = ""
    required: bool = False
    readonly: bool = False
    validators: list[Callable[[Any], bool]] = field(default_factory=list)
    enum_values: Optional[list[Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate the property value.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required
        if self.required and self.value is None:
            return False, f"Property '{self.name}' is required"

        # Type validation
        if self.value is not None:
            if not self._validate_type():
                return False, f"Property '{self.name}' has invalid type"

            # Enum validation
            if self.property_type == PropertyType.ENUM:
                if self.enum_values and self.value not in self.enum_values:
                    return False, f"Property '{self.name}' value not in allowed values"

            # Custom validators
            for validator in self.validators:
                try:
                    if not validator(self.value):
                        return False, f"Property '{self.name}' failed validation"
                except Exception as e:
                    return False, f"Property '{self.name}' validation error: {e}"

        return True, None

    def _validate_type(self) -> bool:
        """Validate value matches property type."""
        type_map = {
            PropertyType.STRING: str,
            PropertyType.INTEGER: int,
            PropertyType.FLOAT: (int, float),
            PropertyType.BOOLEAN: bool,
            PropertyType.LIST: list,
            PropertyType.DICT: dict,
        }

        expected = type_map.get(self.property_type)
        if expected:
            return isinstance(self.value, expected)

        # Special types
        if self.property_type == PropertyType.DATE:
            return isinstance(self.value, (str, float, datetime))

        if self.property_type == PropertyType.PATH:
            return isinstance(self.value, (str, Path))

        if self.property_type in (PropertyType.VECTOR2, PropertyType.VECTOR3):
            if not isinstance(self.value, (list, tuple)):
                return False
            expected_len = 2 if self.property_type == PropertyType.VECTOR2 else 3
            return len(self.value) == expected_len

        if self.property_type == PropertyType.COLOR:
            if not isinstance(self.value, (list, tuple)):
                return False
            return len(self.value) in (3, 4)  # RGB or RGBA

        return True

    def set_value(self, value: Any) -> bool:
        """Set the property value.

        Args:
            value: New value

        Returns:
            True if value was set successfully
        """
        if self.readonly:
            return False

        old_value = self.value
        self.value = value

        valid, error = self.validate()
        if not valid:
            self.value = old_value
            return False

        return True

    def reset(self) -> None:
        """Reset to default value."""
        if not self.readonly:
            self.value = self.default_value


@dataclass
class MetadataTag:
    """A tag for organizing assets.

    Attributes:
        name: Tag name
        color: Display color (hex string)
        category: Tag category
        description: Tag description
        count: Number of assets with this tag
    """

    name: str
    color: str = "#808080"
    category: str = "General"
    description: str = ""
    count: int = 0

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MetadataTag):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False


@dataclass
class MetadataSchema:
    """Schema defining valid metadata structure.

    Attributes:
        name: Schema name
        version: Schema version
        properties: List of property definitions
        required_tags: Tags that must be present
        category: Schema category
    """

    name: str
    version: str = "1.0"
    properties: list[MetadataProperty] = field(default_factory=list)
    required_tags: list[str] = field(default_factory=list)
    category: str = "General"

    def get_property(self, name: str) -> Optional[MetadataProperty]:
        """Get a property by name."""
        for prop in self.properties:
            if prop.name == name:
                return prop
        return None

    def validate(self, metadata: "AssetMetadata") -> list[str]:
        """Validate metadata against this schema.

        Args:
            metadata: Metadata to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check required properties
        for prop in self.properties:
            if prop.required:
                value = metadata.get_property(prop.name)
                if value is None:
                    errors.append(f"Missing required property: {prop.name}")

        # Check required tags
        for tag in self.required_tags:
            if not metadata.has_tag(tag):
                errors.append(f"Missing required tag: {tag}")

        return errors


@dataclass
class AssetMetadata:
    """Metadata for an asset.

    Attributes:
        asset_path: Path to the asset
        properties: Custom properties
        tags: Applied tags
        created_at: Creation timestamp
        modified_at: Last modification timestamp
        created_by: Creator identifier
        description: Asset description
        custom_data: Additional custom data
    """

    asset_path: Path
    properties: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    created_by: str = ""
    description: str = ""
    custom_data: dict[str, Any] = field(default_factory=dict)

    def get_property(self, name: str, default: Any = None) -> Any:
        """Get a property value."""
        return self.properties.get(name, default)

    def set_property(self, name: str, value: Any) -> None:
        """Set a property value."""
        self.properties[name] = value
        self.modified_at = time.time()

    def remove_property(self, name: str) -> bool:
        """Remove a property."""
        if name in self.properties:
            del self.properties[name]
            self.modified_at = time.time()
            return True
        return False

    def add_tag(self, tag: Union[str, MetadataTag]) -> bool:
        """Add a tag."""
        tag_name = tag.name if isinstance(tag, MetadataTag) else tag
        if tag_name not in self.tags:
            self.tags.add(tag_name)
            self.modified_at = time.time()
            return True
        return False

    def remove_tag(self, tag: Union[str, MetadataTag]) -> bool:
        """Remove a tag."""
        tag_name = tag.name if isinstance(tag, MetadataTag) else tag
        if tag_name in self.tags:
            self.tags.discard(tag_name)
            self.modified_at = time.time()
            return True
        return False

    def has_tag(self, tag: Union[str, MetadataTag]) -> bool:
        """Check if asset has a tag."""
        tag_name = tag.name if isinstance(tag, MetadataTag) else tag
        return tag_name in self.tags

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "asset_path": str(self.asset_path),
            "properties": self.properties.copy(),
            "tags": list(self.tags),
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "created_by": self.created_by,
            "description": self.description,
            "custom_data": self.custom_data.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetMetadata":
        """Create from dictionary."""
        return cls(
            asset_path=Path(data.get("asset_path", "")),
            properties=data.get("properties", {}),
            tags=set(data.get("tags", [])),
            created_at=data.get("created_at", time.time()),
            modified_at=data.get("modified_at", time.time()),
            created_by=data.get("created_by", ""),
            description=data.get("description", ""),
            custom_data=data.get("custom_data", {}),
        )


class ContentStoreProtocol(Protocol):
    """Protocol for ContentStore integration."""

    def put(self, obj: Any) -> Any:
        """Store object, return content hash."""
        ...

    def get(self, hash: Any) -> Any:
        """Retrieve object by hash."""
        ...

    def has(self, hash: Any) -> bool:
        """Check if hash exists."""
        ...


@editor(category="Assets")
class MetadataEditor:
    """Editor for asset metadata.

    Provides:
    - Metadata CRUD operations
    - Tag management
    - Schema validation
    - Persistence
    - ContentStore integration

    Attributes:
        root_path: Root directory for assets
        metadata_directory: Directory for metadata files
        content_store: ContentStore for deduplication
        _metadata_cache: Cache of loaded metadata
        _tag_registry: Registry of all tags
        _schemas: Registered schemas
        _change_listeners: Change notification callbacks
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        metadata_directory: Optional[Union[str, Path]] = None,
        content_store: Optional[ContentStoreProtocol] = None,
    ) -> None:
        """Initialize the metadata editor.

        Args:
            root_path: Root directory for assets
            metadata_directory: Directory for metadata files
            content_store: ContentStore for deduplication
        """
        self.root_path = Path(root_path).resolve()
        self.metadata_directory = Path(metadata_directory) if metadata_directory else self.root_path / ".metadata"
        self.metadata_directory.mkdir(parents=True, exist_ok=True)
        self.content_store = content_store

        self._metadata_cache: dict[Path, AssetMetadata] = {}
        self._tag_registry: dict[str, MetadataTag] = {}
        self._schemas: dict[str, MetadataSchema] = {}
        self._change_listeners: list[Callable[[AssetMetadata, str], None]] = []

        # Load existing metadata
        self._load_tag_registry()

    def get_metadata(self, asset_path: Union[str, Path], create: bool = True) -> Optional[AssetMetadata]:
        """Get metadata for an asset.

        Args:
            asset_path: Path to the asset
            create: Create new metadata if not exists

        Returns:
            AssetMetadata or None
        """
        asset_path = Path(asset_path)

        # Check cache
        if asset_path in self._metadata_cache:
            return self._metadata_cache[asset_path]

        # Try to load from disk
        metadata = self._load_metadata(asset_path)

        if metadata is None and create:
            metadata = AssetMetadata(asset_path=asset_path)

        if metadata:
            self._metadata_cache[asset_path] = metadata

        return metadata

    def save_metadata(self, metadata: AssetMetadata) -> bool:
        """Save metadata to disk.

        Args:
            metadata: Metadata to save

        Returns:
            True if saved successfully
        """
        try:
            metadata.modified_at = time.time()
            self._save_metadata(metadata)
            self._metadata_cache[metadata.asset_path] = metadata
            self._notify_change(metadata, "saved")
            return True
        except Exception:
            return False

    def delete_metadata(self, asset_path: Union[str, Path]) -> bool:
        """Delete metadata for an asset.

        Args:
            asset_path: Path to the asset

        Returns:
            True if deleted
        """
        asset_path = Path(asset_path)
        metadata_path = self._metadata_path(asset_path)

        if metadata_path.exists():
            metadata_path.unlink()

        if asset_path in self._metadata_cache:
            metadata = self._metadata_cache.pop(asset_path)
            self._notify_change(metadata, "deleted")
            return True

        return False

    def set_property(
        self,
        asset_path: Union[str, Path],
        name: str,
        value: Any,
    ) -> bool:
        """Set a property on an asset.

        Args:
            asset_path: Path to the asset
            name: Property name
            value: Property value

        Returns:
            True if set successfully
        """
        metadata = self.get_metadata(asset_path)
        if metadata:
            metadata.set_property(name, value)
            return self.save_metadata(metadata)
        return False

    def get_property(
        self,
        asset_path: Union[str, Path],
        name: str,
        default: Any = None,
    ) -> Any:
        """Get a property from an asset.

        Args:
            asset_path: Path to the asset
            name: Property name
            default: Default value if not found

        Returns:
            Property value or default
        """
        metadata = self.get_metadata(asset_path, create=False)
        if metadata:
            return metadata.get_property(name, default)
        return default

    def add_tag(self, asset_path: Union[str, Path], tag: Union[str, MetadataTag]) -> bool:
        """Add a tag to an asset.

        Args:
            asset_path: Path to the asset
            tag: Tag to add

        Returns:
            True if added
        """
        metadata = self.get_metadata(asset_path)
        if metadata and metadata.add_tag(tag):
            tag_name = tag.name if isinstance(tag, MetadataTag) else tag
            if tag_name in self._tag_registry:
                self._tag_registry[tag_name].count += 1
            return self.save_metadata(metadata)
        return False

    def remove_tag(self, asset_path: Union[str, Path], tag: Union[str, MetadataTag]) -> bool:
        """Remove a tag from an asset.

        Args:
            asset_path: Path to the asset
            tag: Tag to remove

        Returns:
            True if removed
        """
        metadata = self.get_metadata(asset_path, create=False)
        if metadata and metadata.remove_tag(tag):
            tag_name = tag.name if isinstance(tag, MetadataTag) else tag
            if tag_name in self._tag_registry:
                self._tag_registry[tag_name].count = max(0, self._tag_registry[tag_name].count - 1)
            return self.save_metadata(metadata)
        return False

    def get_assets_by_tag(self, tag: Union[str, MetadataTag]) -> list[Path]:
        """Get all assets with a tag.

        Args:
            tag: Tag to search for

        Returns:
            List of asset paths
        """
        tag_name = tag.name if isinstance(tag, MetadataTag) else tag
        result = []

        for metadata_path in self._iterate_metadata_files():
            # Load metadata directly from the file since we're iterating metadata files
            try:
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                metadata = AssetMetadata.from_dict(data)
                if metadata.has_tag(tag_name):
                    result.append(metadata.asset_path)
            except Exception:
                continue

        return result

    def register_tag(self, tag: MetadataTag) -> None:
        """Register a tag in the tag registry.

        Args:
            tag: Tag to register
        """
        self._tag_registry[tag.name] = tag
        self._save_tag_registry()

    def unregister_tag(self, tag_name: str) -> bool:
        """Unregister a tag.

        Args:
            tag_name: Tag name to unregister

        Returns:
            True if unregistered
        """
        if tag_name in self._tag_registry:
            del self._tag_registry[tag_name]
            self._save_tag_registry()
            return True
        return False

    def get_all_tags(self) -> list[MetadataTag]:
        """Get all registered tags."""
        return list(self._tag_registry.values())

    def get_tags_by_category(self, category: str) -> list[MetadataTag]:
        """Get tags in a category."""
        return [t for t in self._tag_registry.values() if t.category == category]

    def register_schema(self, schema: MetadataSchema) -> None:
        """Register a metadata schema.

        Args:
            schema: Schema to register
        """
        self._schemas[schema.name] = schema

    def validate_against_schema(
        self,
        asset_path: Union[str, Path],
        schema_name: str,
    ) -> list[str]:
        """Validate asset metadata against a schema.

        Args:
            asset_path: Path to the asset
            schema_name: Schema to validate against

        Returns:
            List of validation errors
        """
        metadata = self.get_metadata(asset_path, create=False)
        if not metadata:
            return ["Metadata not found"]

        schema = self._schemas.get(schema_name)
        if not schema:
            return [f"Schema not found: {schema_name}"]

        return schema.validate(metadata)

    def on_change(self, callback: Callable[[AssetMetadata, str], None]) -> None:
        """Register a change callback.

        Args:
            callback: Function receiving (metadata, action)
        """
        self._change_listeners.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """Get metadata statistics."""
        return {
            "total_metadata": len(list(self._iterate_metadata_files())),
            "cached_metadata": len(self._metadata_cache),
            "registered_tags": len(self._tag_registry),
            "registered_schemas": len(self._schemas),
        }

    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self._metadata_cache.clear()

    def _metadata_path(self, asset_path: Path) -> Path:
        """Get metadata file path for an asset."""
        relative = asset_path.relative_to(self.root_path) if self.root_path in asset_path.parents else asset_path
        return self.metadata_directory / f"{relative}.meta.json"

    def _load_metadata(self, asset_path: Path) -> Optional[AssetMetadata]:
        """Load metadata from disk."""
        metadata_path = self._metadata_path(asset_path)
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
            return AssetMetadata.from_dict(data)
        except Exception:
            return None

    def _save_metadata(self, metadata: AssetMetadata) -> None:
        """Save metadata to disk."""
        metadata_path = self._metadata_path(metadata.asset_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _iterate_metadata_files(self) -> list[Path]:
        """Iterate over all metadata files."""
        return list(self.metadata_directory.rglob("*.meta.json"))

    def _load_tag_registry(self) -> None:
        """Load tag registry from disk."""
        registry_path = self.metadata_directory / "tag_registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    data = json.load(f)
                for tag_data in data.get("tags", []):
                    tag = MetadataTag(**tag_data)
                    self._tag_registry[tag.name] = tag
            except Exception:
                pass

    def _save_tag_registry(self) -> None:
        """Save tag registry to disk."""
        registry_path = self.metadata_directory / "tag_registry.json"

        data = {
            "tags": [
                {
                    "name": t.name,
                    "color": t.color,
                    "category": t.category,
                    "description": t.description,
                    "count": t.count,
                }
                for t in self._tag_registry.values()
            ]
        }

        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _notify_change(self, metadata: AssetMetadata, action: str) -> None:
        """Notify listeners of a change."""
        for listener in self._change_listeners:
            try:
                listener(metadata, action)
            except Exception:
                pass


__all__ = [
    "PropertyType",
    "MetadataProperty",
    "MetadataTag",
    "MetadataSchema",
    "AssetMetadata",
    "MetadataEditor",
]
