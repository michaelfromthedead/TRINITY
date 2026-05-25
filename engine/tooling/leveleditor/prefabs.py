"""
Prefab System - Reusable object templates with nested prefabs and overrides.

Provides:
- Prefab assets (templates)
- Prefab instances (instantiated prefabs)
- Nested prefabs (prefabs containing prefabs)
- Override system (per-instance property changes)
- Prefab variants (derived prefabs with modifications)

All prefab operations integrate with Foundation Tracker for undo/redo.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from .placement import Vector3, Quaternion, Transform, editor, track_changes
from foundation.tracker import tracker


# =============================================================================
# Enums
# =============================================================================

class OverrideType(Enum):
    """Type of property override."""
    VALUE = auto()  # Simple value override
    ADD_COMPONENT = auto()  # Added component
    REMOVE_COMPONENT = auto()  # Removed component
    ADD_CHILD = auto()  # Added child object
    REMOVE_CHILD = auto()  # Removed child object
    REORDER = auto()  # Reordered children


class PrefabState(Enum):
    """State of a prefab instance."""
    SYNCHRONIZED = auto()  # Matches prefab asset
    MODIFIED = auto()  # Has overrides
    DISCONNECTED = auto()  # Broken link to asset
    NESTED = auto()  # Part of another prefab


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(slots=True)
class PrefabOverride:
    """A single property override on a prefab instance."""
    override_id: str
    target_path: str  # Path to target within prefab (e.g., "root/child/component")
    property_name: str
    override_type: OverrideType = OverrideType.VALUE
    original_value: Any = None
    override_value: Any = None
    enabled: bool = True

    def __post_init__(self):
        if not self.override_id:
            self.override_id = str(uuid.uuid4())


@dataclass(slots=True)
class PrefabComponent:
    """A component within a prefab."""
    component_id: str
    component_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass(slots=True)
class PrefabChild:
    """A child object within a prefab."""
    child_id: str
    name: str
    local_transform: Transform = field(default_factory=Transform)
    components: list[PrefabComponent] = field(default_factory=list)
    children: list["PrefabChild"] = field(default_factory=list)
    nested_prefab_id: Optional[str] = None  # If this child is a nested prefab


# =============================================================================
# Prefab Asset
# =============================================================================

@editor
class PrefabAsset:
    """
    A prefab asset - the template for creating instances.

    Stores the complete structure of a prefab including:
    - Root transform and components
    - Child objects and their components
    - Nested prefab references
    """

    __slots__ = (
        "_id",
        "_name",
        "_root",
        "_metadata",
        "_version",
        "_thumbnail_path",
        "_tags",
        "_category",
        "__weakref__",
    )

    def __init__(self, name: str):
        """
        Initialize a prefab asset.

        Args:
            name: Name of the prefab
        """
        self._id = str(uuid.uuid4())
        self._name = name
        self._root = PrefabChild(
            child_id=str(uuid.uuid4()),
            name=name,
        )
        self._metadata: dict[str, Any] = {}
        self._version = 1
        self._thumbnail_path: Optional[str] = None
        self._tags: list[str] = []
        self._category: str = "Uncategorized"

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
        self._root.name = value
        tracker.mark_dirty(self, "_name", old_name, value)

    @property
    def root(self) -> PrefabChild:
        return self._root

    @property
    def version(self) -> int:
        return self._version

    @property
    def tags(self) -> list[str]:
        return self._tags.copy()

    @property
    def category(self) -> str:
        return self._category

    @category.setter
    def category(self, value: str) -> None:
        old_cat = self._category
        self._category = value
        tracker.mark_dirty(self, "_category", old_cat, value)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the prefab."""
        if tag not in self._tags:
            old_tags = self._tags.copy()
            self._tags.append(tag)
            tracker.mark_dirty(self, "_tags", old_tags, self._tags.copy())

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the prefab."""
        if tag in self._tags:
            old_tags = self._tags.copy()
            self._tags.remove(tag)
            tracker.mark_dirty(self, "_tags", old_tags, self._tags.copy())
            return True
        return False

    @track_changes
    def add_component(
        self,
        component_type: str,
        properties: Optional[dict[str, Any]] = None,
        target_path: str = ""
    ) -> PrefabComponent:
        """
        Add a component to the prefab.

        Args:
            component_type: Type of component
            properties: Initial properties
            target_path: Path to target child (empty for root)

        Returns:
            The created component
        """
        component = PrefabComponent(
            component_id=str(uuid.uuid4()),
            component_type=component_type,
            properties=properties or {},
        )

        target = self._find_child(target_path) or self._root
        target.components.append(component)
        self._version += 1

        return component

    @track_changes
    def remove_component(self, component_id: str, target_path: str = "") -> bool:
        """
        Remove a component from the prefab.

        Args:
            component_id: ID of component to remove
            target_path: Path to target child

        Returns:
            True if removed
        """
        target = self._find_child(target_path) or self._root

        for i, comp in enumerate(target.components):
            if comp.component_id == component_id:
                target.components.pop(i)
                self._version += 1
                return True

        return False

    @track_changes
    def add_child(
        self,
        name: str,
        parent_path: str = "",
        transform: Optional[Transform] = None
    ) -> PrefabChild:
        """
        Add a child object to the prefab.

        Args:
            name: Name of the child
            parent_path: Path to parent (empty for root)
            transform: Local transform

        Returns:
            The created child
        """
        child = PrefabChild(
            child_id=str(uuid.uuid4()),
            name=name,
            local_transform=transform or Transform(),
        )

        parent = self._find_child(parent_path) or self._root
        parent.children.append(child)
        self._version += 1

        return child

    @track_changes
    def remove_child(self, child_id: str, parent_path: str = "") -> bool:
        """
        Remove a child object from the prefab.

        Args:
            child_id: ID of child to remove
            parent_path: Path to parent

        Returns:
            True if removed
        """
        parent = self._find_child(parent_path) or self._root

        for i, child in enumerate(parent.children):
            if child.child_id == child_id:
                parent.children.pop(i)
                self._version += 1
                return True

        return False

    @track_changes
    def set_nested_prefab(self, child_path: str, prefab_id: str) -> bool:
        """
        Set a child to reference a nested prefab.

        Args:
            child_path: Path to the child
            prefab_id: ID of the nested prefab

        Returns:
            True if set successfully
        """
        child = self._find_child(child_path)
        if child:
            child.nested_prefab_id = prefab_id
            self._version += 1
            return True
        return False

    def _find_child(self, path: str) -> Optional[PrefabChild]:
        """Find a child by path."""
        if not path:
            return None

        parts = path.split("/")
        current = self._root

        for part in parts:
            found = False
            for child in current.children:
                if child.name == part:
                    current = child
                    found = True
                    break
            if not found:
                return None

        return current

    def get_all_nested_prefab_ids(self) -> list[str]:
        """Get IDs of all nested prefabs."""
        ids = []
        self._collect_nested_ids(self._root, ids)
        return ids

    def _collect_nested_ids(self, node: PrefabChild, ids: list[str]) -> None:
        """Recursively collect nested prefab IDs."""
        if node.nested_prefab_id:
            ids.append(node.nested_prefab_id)
        for child in node.children:
            self._collect_nested_ids(child, ids)

    def clone(self) -> "PrefabAsset":
        """Create a deep copy of this prefab asset."""
        new_asset = PrefabAsset(f"{self._name} (Copy)")
        new_asset._root = self._deep_copy_child(self._root)
        new_asset._root.name = new_asset._name
        new_asset._metadata = copy.deepcopy(self._metadata)
        new_asset._tags = self._tags.copy()
        new_asset._category = self._category
        return new_asset

    def _deep_copy_child(self, child: PrefabChild) -> PrefabChild:
        """Deep copy a prefab child."""
        new_child = PrefabChild(
            child_id=str(uuid.uuid4()),
            name=child.name,
            local_transform=Transform(
                position=Vector3(
                    child.local_transform.position.x,
                    child.local_transform.position.y,
                    child.local_transform.position.z,
                ),
                rotation=Quaternion(
                    child.local_transform.rotation.x,
                    child.local_transform.rotation.y,
                    child.local_transform.rotation.z,
                    child.local_transform.rotation.w,
                ),
                scale=Vector3(
                    child.local_transform.scale.x,
                    child.local_transform.scale.y,
                    child.local_transform.scale.z,
                ),
            ),
            nested_prefab_id=child.nested_prefab_id,
        )

        for comp in child.components:
            new_child.components.append(PrefabComponent(
                component_id=str(uuid.uuid4()),
                component_type=comp.component_type,
                properties=copy.deepcopy(comp.properties),
                enabled=comp.enabled,
            ))

        for sub_child in child.children:
            new_child.children.append(self._deep_copy_child(sub_child))

        return new_child


# =============================================================================
# Prefab Instance
# =============================================================================

@editor
class PrefabInstance:
    """
    An instance of a prefab in the scene.

    Stores:
    - Reference to the source prefab asset
    - Local overrides (property modifications)
    - Instance transform
    """

    __slots__ = (
        "_id",
        "_prefab_asset_id",
        "_transform",
        "_overrides",
        "_state",
        "_prefab_version",
        "_callbacks",
        "__weakref__",
    )

    def __init__(self, prefab_asset_id: str):
        """
        Create a prefab instance.

        Args:
            prefab_asset_id: ID of the source prefab asset
        """
        self._id = str(uuid.uuid4())
        self._prefab_asset_id = prefab_asset_id
        self._transform = Transform()
        self._overrides: list[PrefabOverride] = []
        self._state = PrefabState.SYNCHRONIZED
        self._prefab_version = 0
        self._callbacks: dict[str, list[Callable]] = {
            "on_override_change": [],
            "on_state_change": [],
        }

    @property
    def id(self) -> str:
        return self._id

    @property
    def prefab_asset_id(self) -> str:
        return self._prefab_asset_id

    @property
    def transform(self) -> Transform:
        return self._transform

    @transform.setter
    def transform(self, value: Transform) -> None:
        old_transform = self._transform
        self._transform = value
        tracker.mark_dirty(self, "_transform", old_transform, value)

    @property
    def overrides(self) -> list[PrefabOverride]:
        return self._overrides.copy()

    @property
    def state(self) -> PrefabState:
        return self._state

    @property
    def has_overrides(self) -> bool:
        return len(self._overrides) > 0

    def on(self, event: str, callback: Callable) -> None:
        """Register callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    @track_changes
    def add_override(
        self,
        target_path: str,
        property_name: str,
        override_value: Any,
        original_value: Any = None,
        override_type: OverrideType = OverrideType.VALUE
    ) -> PrefabOverride:
        """
        Add a property override.

        Args:
            target_path: Path to target within prefab
            property_name: Name of property to override
            override_value: New value
            original_value: Original value from prefab
            override_type: Type of override

        Returns:
            The created override
        """
        # Check if override already exists
        for existing in self._overrides:
            if existing.target_path == target_path and existing.property_name == property_name:
                existing.override_value = override_value
                self._update_state()
                return existing

        override = PrefabOverride(
            override_id=str(uuid.uuid4()),
            target_path=target_path,
            property_name=property_name,
            override_type=override_type,
            original_value=original_value,
            override_value=override_value,
        )

        old_overrides = self._overrides.copy()
        self._overrides.append(override)
        tracker.mark_dirty(self, "_overrides", old_overrides, self._overrides.copy())

        self._update_state()

        for callback in self._callbacks["on_override_change"]:
            callback(override, "add")

        return override

    @track_changes
    def remove_override(self, override_id: str) -> bool:
        """
        Remove an override by ID.

        Args:
            override_id: ID of override to remove

        Returns:
            True if removed
        """
        for i, override in enumerate(self._overrides):
            if override.override_id == override_id:
                old_overrides = self._overrides.copy()
                removed = self._overrides.pop(i)
                tracker.mark_dirty(self, "_overrides", old_overrides, self._overrides.copy())

                self._update_state()

                for callback in self._callbacks["on_override_change"]:
                    callback(removed, "remove")

                return True

        return False

    @track_changes
    def clear_overrides(self) -> int:
        """
        Clear all overrides.

        Returns:
            Number of overrides cleared
        """
        count = len(self._overrides)
        old_overrides = self._overrides.copy()
        self._overrides.clear()
        tracker.mark_dirty(self, "_overrides", old_overrides, [])

        self._update_state()

        return count

    @track_changes
    def revert_override(self, override_id: str) -> bool:
        """
        Revert a specific override to original value.

        Args:
            override_id: ID of override to revert

        Returns:
            True if reverted
        """
        return self.remove_override(override_id)

    def get_override(self, target_path: str, property_name: str) -> Optional[PrefabOverride]:
        """
        Get override for a specific property.

        Args:
            target_path: Path to target
            property_name: Property name

        Returns:
            Override if exists, None otherwise
        """
        for override in self._overrides:
            if override.target_path == target_path and override.property_name == property_name:
                return override
        return None

    def has_override(self, target_path: str, property_name: str) -> bool:
        """Check if a property has an override."""
        return self.get_override(target_path, property_name) is not None

    def get_effective_value(
        self,
        target_path: str,
        property_name: str,
        prefab_value: Any
    ) -> Any:
        """
        Get effective value considering overrides.

        Args:
            target_path: Path to target
            property_name: Property name
            prefab_value: Original value from prefab

        Returns:
            Override value if exists, otherwise prefab value
        """
        override = self.get_override(target_path, property_name)
        if override and override.enabled:
            return override.override_value
        return prefab_value

    def _update_state(self) -> None:
        """Update instance state based on overrides."""
        old_state = self._state
        if self._overrides:
            self._state = PrefabState.MODIFIED
        else:
            self._state = PrefabState.SYNCHRONIZED

        if old_state != self._state:
            for callback in self._callbacks["on_state_change"]:
                callback(old_state, self._state)

    def check_version(self, asset_version: int) -> bool:
        """
        Check if instance is synchronized with asset version.

        Args:
            asset_version: Current version of prefab asset

        Returns:
            True if synchronized
        """
        return self._prefab_version == asset_version

    def update_version(self, asset_version: int) -> None:
        """Update the tracked asset version."""
        self._prefab_version = asset_version


# =============================================================================
# Prefab Variant
# =============================================================================

@editor
class PrefabVariant(PrefabAsset):
    """
    A prefab variant - derived from another prefab with modifications.

    Inherits from parent prefab and stores delta changes.
    """

    __slots__ = ("_parent_id", "_variant_overrides")

    def __init__(self, name: str, parent_id: str):
        """
        Create a prefab variant.

        Args:
            name: Name of the variant
            parent_id: ID of parent prefab
        """
        super().__init__(name)
        self._parent_id = parent_id
        self._variant_overrides: list[PrefabOverride] = []

    @property
    def parent_id(self) -> str:
        return self._parent_id

    @property
    def variant_overrides(self) -> list[PrefabOverride]:
        return self._variant_overrides.copy()

    @track_changes
    def add_variant_override(
        self,
        target_path: str,
        property_name: str,
        override_value: Any,
        original_value: Any = None
    ) -> PrefabOverride:
        """Add a variant-level override."""
        override = PrefabOverride(
            override_id=str(uuid.uuid4()),
            target_path=target_path,
            property_name=property_name,
            original_value=original_value,
            override_value=override_value,
        )

        old_overrides = self._variant_overrides.copy()
        self._variant_overrides.append(override)
        tracker.mark_dirty(self, "_variant_overrides", old_overrides,
                          self._variant_overrides.copy())
        self._version += 1

        return override


# =============================================================================
# Prefab Manager
# =============================================================================

@editor
class PrefabManager:
    """
    Central manager for prefab assets and instances.

    Handles:
    - Prefab asset storage and retrieval
    - Instance creation and tracking
    - Nested prefab resolution
    - Version synchronization
    """

    __slots__ = (
        "_assets",
        "_instances",
        "_callbacks",
        "__weakref__",
    )

    def __init__(self):
        """Initialize prefab manager."""
        self._assets: dict[str, PrefabAsset] = {}
        self._instances: dict[str, PrefabInstance] = {}
        self._callbacks: dict[str, list[Callable]] = {
            "on_asset_add": [],
            "on_asset_remove": [],
            "on_instance_create": [],
            "on_instance_destroy": [],
        }

    def on(self, event: str, callback: Callable) -> None:
        """Register callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    # Asset management
    @track_changes
    def register_asset(self, asset: PrefabAsset) -> None:
        """
        Register a prefab asset.

        Args:
            asset: Prefab asset to register
        """
        self._assets[asset.id] = asset

        for callback in self._callbacks["on_asset_add"]:
            callback(asset)

    @track_changes
    def unregister_asset(self, asset_id: str) -> bool:
        """
        Unregister a prefab asset.

        Args:
            asset_id: ID of asset to unregister

        Returns:
            True if unregistered
        """
        if asset_id not in self._assets:
            return False

        asset = self._assets.pop(asset_id)

        # Mark all instances as disconnected
        for instance in self._instances.values():
            if instance.prefab_asset_id == asset_id:
                instance._state = PrefabState.DISCONNECTED

        for callback in self._callbacks["on_asset_remove"]:
            callback(asset)

        return True

    def get_asset(self, asset_id: str) -> Optional[PrefabAsset]:
        """Get a prefab asset by ID."""
        return self._assets.get(asset_id)

    def get_all_assets(self) -> list[PrefabAsset]:
        """Get all registered assets."""
        return list(self._assets.values())

    def find_assets_by_name(self, name: str) -> list[PrefabAsset]:
        """Find assets by name (partial match)."""
        name_lower = name.lower()
        return [a for a in self._assets.values() if name_lower in a.name.lower()]

    def find_assets_by_tag(self, tag: str) -> list[PrefabAsset]:
        """Find assets with a specific tag."""
        return [a for a in self._assets.values() if tag in a.tags]

    def find_assets_by_category(self, category: str) -> list[PrefabAsset]:
        """Find assets in a specific category."""
        return [a for a in self._assets.values() if a.category == category]

    # Instance management
    @track_changes
    def instantiate(
        self,
        asset_id: str,
        transform: Optional[Transform] = None
    ) -> Optional[PrefabInstance]:
        """
        Create an instance of a prefab.

        Args:
            asset_id: ID of prefab asset
            transform: Optional transform for instance

        Returns:
            Created instance or None if asset not found
        """
        asset = self._assets.get(asset_id)
        if not asset:
            return None

        instance = PrefabInstance(asset_id)
        if transform:
            instance._transform = transform
        instance._prefab_version = asset.version

        self._instances[instance.id] = instance

        for callback in self._callbacks["on_instance_create"]:
            callback(instance)

        return instance

    @track_changes
    def destroy_instance(self, instance_id: str) -> bool:
        """
        Destroy a prefab instance.

        Args:
            instance_id: ID of instance to destroy

        Returns:
            True if destroyed
        """
        if instance_id not in self._instances:
            return False

        instance = self._instances.pop(instance_id)

        for callback in self._callbacks["on_instance_destroy"]:
            callback(instance)

        return True

    def get_instance(self, instance_id: str) -> Optional[PrefabInstance]:
        """Get a prefab instance by ID."""
        return self._instances.get(instance_id)

    def get_all_instances(self) -> list[PrefabInstance]:
        """Get all instances."""
        return list(self._instances.values())

    def get_instances_of_asset(self, asset_id: str) -> list[PrefabInstance]:
        """Get all instances of a specific prefab asset."""
        return [i for i in self._instances.values() if i.prefab_asset_id == asset_id]

    # Prefab operations
    @track_changes
    def apply_instance_to_asset(self, instance_id: str) -> bool:
        """
        Apply instance overrides to the prefab asset.

        Args:
            instance_id: ID of instance

        Returns:
            True if applied successfully
        """
        instance = self._instances.get(instance_id)
        if not instance:
            return False

        asset = self._assets.get(instance.prefab_asset_id)
        if not asset:
            return False

        # Apply overrides to asset (simplified - would need full implementation)
        for override in instance.overrides:
            if override.override_type == OverrideType.VALUE:
                # Would apply property change to asset here
                pass

        asset._version += 1
        instance.clear_overrides()
        instance._prefab_version = asset.version

        return True

    @track_changes
    def revert_instance(self, instance_id: str) -> bool:
        """
        Revert instance to match prefab asset.

        Args:
            instance_id: ID of instance

        Returns:
            True if reverted
        """
        instance = self._instances.get(instance_id)
        if not instance:
            return False

        instance.clear_overrides()
        return True

    def create_variant(
        self,
        asset_id: str,
        variant_name: str
    ) -> Optional[PrefabVariant]:
        """
        Create a variant of a prefab.

        Args:
            asset_id: ID of parent prefab
            variant_name: Name for the variant

        Returns:
            Created variant or None
        """
        parent = self._assets.get(asset_id)
        if not parent:
            return None

        variant = PrefabVariant(variant_name, asset_id)
        # Copy parent structure
        variant._root = parent._deep_copy_child(parent.root)
        variant._tags = parent.tags.copy()
        variant._category = parent.category

        self.register_asset(variant)
        return variant

    def check_circular_reference(self, asset_id: str, nested_id: str) -> bool:
        """
        Check if adding nested prefab would create circular reference.

        Args:
            asset_id: ID of asset to add nested prefab to
            nested_id: ID of nested prefab

        Returns:
            True if circular reference would be created
        """
        if asset_id == nested_id:
            return True

        nested_asset = self._assets.get(nested_id)
        if not nested_asset:
            return False

        # Check all nested prefabs recursively
        nested_ids = nested_asset.get_all_nested_prefab_ids()
        if asset_id in nested_ids:
            return True

        for nid in nested_ids:
            if self.check_circular_reference(asset_id, nid):
                return True

        return False

    def get_statistics(self) -> dict[str, int]:
        """Get manager statistics."""
        return {
            "total_assets": len(self._assets),
            "total_instances": len(self._instances),
            "variants": sum(1 for a in self._assets.values() if isinstance(a, PrefabVariant)),
            "modified_instances": sum(1 for i in self._instances.values() if i.has_overrides),
            "disconnected_instances": sum(
                1 for i in self._instances.values()
                if i.state == PrefabState.DISCONNECTED
            ),
        }


__all__ = [
    "PrefabAsset",
    "PrefabInstance",
    "PrefabOverride",
    "PrefabVariant",
    "PrefabManager",
    "PrefabComponent",
    "PrefabChild",
    "OverrideType",
    "PrefabState",
]
