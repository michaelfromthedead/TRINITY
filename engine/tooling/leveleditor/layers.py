"""
Layer Management System - Organize objects with visibility, locking, and color coding.

Provides:
- Layer creation and management
- Visibility toggling
- Lock/unlock functionality
- Color coding for visual organization
- Layer filtering and isolation

All layer operations integrate with Foundation Tracker for undo/redo.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from .placement import editor, track_changes
from foundation.tracker import tracker


# =============================================================================
# Enums
# =============================================================================

class LayerColor(Enum):
    """Predefined layer colors."""
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)
    YELLOW = (1.0, 1.0, 0.0)
    CYAN = (0.0, 1.0, 1.0)
    MAGENTA = (1.0, 0.0, 1.0)
    ORANGE = (1.0, 0.5, 0.0)
    PURPLE = (0.5, 0.0, 1.0)
    PINK = (1.0, 0.5, 0.5)
    LIME = (0.5, 1.0, 0.0)
    TEAL = (0.0, 0.5, 0.5)
    BROWN = (0.5, 0.25, 0.0)
    GRAY = (0.5, 0.5, 0.5)
    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0)


class LayerBlendMode(Enum):
    """How layer objects blend with others."""
    NORMAL = auto()
    ADDITIVE = auto()
    MULTIPLY = auto()
    OVERLAY = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(slots=True)
class LayerSettings:
    """Settings for a layer."""
    visible: bool = True
    locked: bool = False
    selectable: bool = True
    renderable: bool = True
    collidable: bool = True
    cast_shadows: bool = True
    receive_shadows: bool = True
    outline_on_select: bool = True


@dataclass(slots=True)
class LayerMask:
    """Bit mask for layer filtering."""
    value: int = 0xFFFFFFFF  # All layers by default

    def includes(self, layer_index: int) -> bool:
        """Check if layer is included in mask."""
        return bool(self.value & (1 << layer_index))

    def include(self, layer_index: int) -> None:
        """Include a layer in mask."""
        self.value |= (1 << layer_index)

    def exclude(self, layer_index: int) -> None:
        """Exclude a layer from mask."""
        self.value &= ~(1 << layer_index)

    def toggle(self, layer_index: int) -> None:
        """Toggle a layer in mask."""
        self.value ^= (1 << layer_index)

    @staticmethod
    def all_layers() -> "LayerMask":
        """Create mask including all layers."""
        return LayerMask(0xFFFFFFFF)

    @staticmethod
    def no_layers() -> "LayerMask":
        """Create mask excluding all layers."""
        return LayerMask(0)


# =============================================================================
# Layer
# =============================================================================

@editor
class Layer:
    """
    A layer for organizing scene objects.

    Provides visibility, locking, and color coding for objects.
    """

    __slots__ = (
        "_id",
        "_name",
        "_index",
        "_color",
        "_custom_color",
        "_settings",
        "_object_ids",
        "_parent_id",
        "_children_ids",
        "_metadata",
        "__weakref__",
    )

    def __init__(
        self,
        name: str,
        index: int = 0,
        color: LayerColor = LayerColor.GRAY
    ):
        """
        Initialize a layer.

        Args:
            name: Layer name
            index: Layer index (0-31)
            color: Layer color
        """
        self._id = str(uuid.uuid4())
        self._name = name
        self._index = min(31, max(0, index))  # Clamp to 0-31
        self._color = color
        self._custom_color: Optional[tuple[float, float, float]] = None
        self._settings = LayerSettings()
        self._object_ids: list[str] = []
        self._parent_id: Optional[str] = None
        self._children_ids: list[str] = []
        self._metadata: dict[str, Any] = {}

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
        tracker.mark_dirty(self, "_name", old_name, value)

    @property
    def index(self) -> int:
        return self._index

    @property
    def color(self) -> LayerColor:
        return self._color

    @color.setter
    def color(self, value: LayerColor) -> None:
        old_color = self._color
        self._color = value
        self._custom_color = None
        tracker.mark_dirty(self, "_color", old_color, value)

    @property
    def rgb(self) -> tuple[float, float, float]:
        """Get RGB color tuple."""
        if self._custom_color:
            return self._custom_color
        return self._color.value

    def set_custom_color(self, r: float, g: float, b: float) -> None:
        """Set a custom RGB color."""
        old_color = self._custom_color
        self._custom_color = (
            max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)),
        )
        tracker.mark_dirty(self, "_custom_color", old_color, self._custom_color)

    @property
    def settings(self) -> LayerSettings:
        return self._settings

    @property
    def visible(self) -> bool:
        return self._settings.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        old_visible = self._settings.visible
        self._settings.visible = value
        tracker.mark_dirty(self, "visible", old_visible, value)

    @property
    def locked(self) -> bool:
        return self._settings.locked

    @locked.setter
    def locked(self, value: bool) -> None:
        old_locked = self._settings.locked
        self._settings.locked = value
        tracker.mark_dirty(self, "locked", old_locked, value)

    @property
    def object_count(self) -> int:
        return len(self._object_ids)

    @property
    def object_ids(self) -> list[str]:
        return self._object_ids.copy()

    @property
    def bit_mask(self) -> int:
        """Get this layer's bit mask value."""
        return 1 << self._index

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self._metadata.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        old_metadata = self._metadata.copy()
        self._metadata[key] = value
        tracker.mark_dirty(self, "_metadata", old_metadata, self._metadata.copy())

    @track_changes
    def add_object(self, object_id: str) -> bool:
        """
        Add an object to this layer.

        Args:
            object_id: ID of object to add

        Returns:
            True if added (wasn't already in layer)
        """
        if object_id in self._object_ids:
            return False

        old_ids = self._object_ids.copy()
        self._object_ids.append(object_id)
        tracker.mark_dirty(self, "_object_ids", old_ids, self._object_ids.copy())
        return True

    @track_changes
    def remove_object(self, object_id: str) -> bool:
        """
        Remove an object from this layer.

        Args:
            object_id: ID of object to remove

        Returns:
            True if removed
        """
        if object_id not in self._object_ids:
            return False

        old_ids = self._object_ids.copy()
        self._object_ids.remove(object_id)
        tracker.mark_dirty(self, "_object_ids", old_ids, self._object_ids.copy())
        return True

    def contains(self, object_id: str) -> bool:
        """Check if object is in this layer."""
        return object_id in self._object_ids

    @track_changes
    def clear_objects(self) -> int:
        """
        Remove all objects from this layer.

        Returns:
            Number of objects removed
        """
        count = len(self._object_ids)
        old_ids = self._object_ids.copy()
        self._object_ids.clear()
        tracker.mark_dirty(self, "_object_ids", old_ids, [])
        return count

    def toggle_visibility(self) -> bool:
        """Toggle visibility and return new state."""
        self.visible = not self._settings.visible
        return self._settings.visible

    def toggle_lock(self) -> bool:
        """Toggle lock state and return new state."""
        self.locked = not self._settings.locked
        return self._settings.locked


# =============================================================================
# Layer Manager
# =============================================================================

@editor
class LayerManager:
    """
    Central manager for all layers.

    Handles layer creation, management, and object assignment.
    """

    __slots__ = (
        "_layers",
        "_layer_order",
        "_default_layer_id",
        "_active_layer_id",
        "_isolated_layer_id",
        "_callbacks",
        "_next_index",
        "__weakref__",
    )

    MAX_LAYERS = 32  # Standard layer limit

    def __init__(self):
        """Initialize layer manager with default layer."""
        self._layers: dict[str, Layer] = {}
        self._layer_order: list[str] = []
        self._next_index = 0
        self._isolated_layer_id: Optional[str] = None

        # Initialize callbacks before create_layer is called
        self._callbacks: dict[str, list[Callable]] = {
            "on_layer_create": [],
            "on_layer_delete": [],
            "on_layer_change": [],
            "on_active_change": [],
            "on_isolation_change": [],
        }

        # Create default layer
        default_layer = self.create_layer("Default", LayerColor.GRAY)
        self._default_layer_id = default_layer.id
        self._active_layer_id = default_layer.id

    @property
    def default_layer(self) -> Layer:
        return self._layers[self._default_layer_id]

    @property
    def active_layer(self) -> Optional[Layer]:
        return self._layers.get(self._active_layer_id)

    @property
    def isolated_layer(self) -> Optional[Layer]:
        if self._isolated_layer_id:
            return self._layers.get(self._isolated_layer_id)
        return None

    @property
    def layer_count(self) -> int:
        return len(self._layers)

    def on(self, event: str, callback: Callable) -> None:
        """Register callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    @track_changes
    def create_layer(
        self,
        name: str,
        color: LayerColor = LayerColor.GRAY
    ) -> Layer:
        """
        Create a new layer.

        Args:
            name: Layer name
            color: Layer color

        Returns:
            Created layer
        """
        if self._next_index >= self.MAX_LAYERS:
            raise ValueError(f"Maximum layer count ({self.MAX_LAYERS}) reached")

        layer = Layer(name, self._next_index, color)
        self._layers[layer.id] = layer
        self._layer_order.append(layer.id)
        self._next_index += 1

        for callback in self._callbacks["on_layer_create"]:
            callback(layer)

        return layer

    @track_changes
    def delete_layer(self, layer_id: str) -> bool:
        """
        Delete a layer.

        Args:
            layer_id: ID of layer to delete

        Returns:
            True if deleted
        """
        if layer_id == self._default_layer_id:
            return False  # Cannot delete default layer

        layer = self._layers.get(layer_id)
        if not layer:
            return False

        # Move objects to default layer
        for obj_id in layer.object_ids:
            self._layers[self._default_layer_id].add_object(obj_id)

        # Remove layer
        del self._layers[layer_id]
        self._layer_order.remove(layer_id)

        # Update active layer if needed
        if self._active_layer_id == layer_id:
            self._active_layer_id = self._default_layer_id

        # Clear isolation if needed
        if self._isolated_layer_id == layer_id:
            self._isolated_layer_id = None

        for callback in self._callbacks["on_layer_delete"]:
            callback(layer)

        return True

    def get_layer(self, layer_id: str) -> Optional[Layer]:
        """Get layer by ID."""
        return self._layers.get(layer_id)

    def get_layer_by_name(self, name: str) -> Optional[Layer]:
        """Get layer by name."""
        for layer in self._layers.values():
            if layer.name == name:
                return layer
        return None

    def get_layer_by_index(self, index: int) -> Optional[Layer]:
        """Get layer by index."""
        for layer in self._layers.values():
            if layer.index == index:
                return layer
        return None

    def get_all_layers(self) -> list[Layer]:
        """Get all layers in order."""
        return [self._layers[lid] for lid in self._layer_order if lid in self._layers]

    def get_layers_by_mask(self, mask: LayerMask) -> list[Layer]:
        """Get layers included in mask."""
        return [l for l in self._layers.values() if mask.includes(l.index)]

    @track_changes
    def set_active_layer(self, layer_id: str) -> bool:
        """
        Set the active layer.

        Args:
            layer_id: ID of layer to make active

        Returns:
            True if set successfully
        """
        if layer_id not in self._layers:
            return False

        old_active = self._active_layer_id
        self._active_layer_id = layer_id

        for callback in self._callbacks["on_active_change"]:
            callback(old_active, layer_id)

        return True

    @track_changes
    def assign_object_to_layer(self, object_id: str, layer_id: str) -> bool:
        """
        Assign an object to a layer.

        Args:
            object_id: ID of object
            layer_id: ID of target layer

        Returns:
            True if assigned
        """
        if layer_id not in self._layers:
            return False

        # Remove from current layer
        for layer in self._layers.values():
            layer.remove_object(object_id)

        # Add to new layer
        self._layers[layer_id].add_object(object_id)

        for callback in self._callbacks["on_layer_change"]:
            callback(object_id, layer_id)

        return True

    def get_object_layer(self, object_id: str) -> Optional[Layer]:
        """Get the layer containing an object."""
        for layer in self._layers.values():
            if layer.contains(object_id):
                return layer
        return None

    @track_changes
    def isolate_layer(self, layer_id: str) -> bool:
        """
        Isolate a layer (hide all others).

        Args:
            layer_id: ID of layer to isolate

        Returns:
            True if isolated
        """
        if layer_id not in self._layers:
            return False

        # Store previous visibility states if not already isolated
        if not self._isolated_layer_id:
            for layer in self._layers.values():
                layer.set_metadata("_pre_isolation_visible", layer.visible)

        self._isolated_layer_id = layer_id

        # Hide all layers except isolated one
        for lid, layer in self._layers.items():
            layer.visible = (lid == layer_id)

        for callback in self._callbacks["on_isolation_change"]:
            callback(layer_id, True)

        return True

    @track_changes
    def exit_isolation(self) -> bool:
        """
        Exit layer isolation mode.

        Returns:
            True if was isolated
        """
        if not self._isolated_layer_id:
            return False

        # Restore previous visibility states
        for layer in self._layers.values():
            prev_visible = layer.get_metadata("_pre_isolation_visible", True)
            layer.visible = prev_visible

        old_isolated = self._isolated_layer_id
        self._isolated_layer_id = None

        for callback in self._callbacks["on_isolation_change"]:
            callback(old_isolated, False)

        return True

    # Batch operations
    @track_changes
    def set_all_visible(self, visible: bool) -> None:
        """Set visibility for all layers."""
        for layer in self._layers.values():
            layer.visible = visible

    @track_changes
    def set_all_locked(self, locked: bool) -> None:
        """Set lock state for all layers."""
        for layer in self._layers.values():
            layer.locked = locked

    @track_changes
    def invert_visibility(self) -> None:
        """Invert visibility of all layers."""
        for layer in self._layers.values():
            layer.visible = not layer.visible

    @track_changes
    def reorder_layer(self, layer_id: str, new_position: int) -> bool:
        """
        Change layer order.

        Args:
            layer_id: ID of layer to move
            new_position: New position in order

        Returns:
            True if reordered
        """
        if layer_id not in self._layer_order:
            return False

        old_order = self._layer_order.copy()
        self._layer_order.remove(layer_id)
        new_position = max(0, min(len(self._layer_order), new_position))
        self._layer_order.insert(new_position, layer_id)

        tracker.mark_dirty(self, "_layer_order", old_order, self._layer_order.copy())
        return True

    @track_changes
    def merge_layers(self, source_id: str, target_id: str) -> bool:
        """
        Merge one layer into another.

        Args:
            source_id: Layer to merge from
            target_id: Layer to merge into

        Returns:
            True if merged
        """
        source = self._layers.get(source_id)
        target = self._layers.get(target_id)

        if not source or not target:
            return False

        if source_id == self._default_layer_id:
            return False  # Cannot merge default layer

        # Move all objects
        for obj_id in source.object_ids:
            target.add_object(obj_id)

        # Delete source layer
        return self.delete_layer(source_id)

    def create_visibility_mask(self) -> LayerMask:
        """Create mask of currently visible layers."""
        mask = LayerMask(0)
        for layer in self._layers.values():
            if layer.visible:
                mask.include(layer.index)
        return mask

    def create_selection_mask(self) -> LayerMask:
        """Create mask of selectable layers."""
        mask = LayerMask(0)
        for layer in self._layers.values():
            if layer.settings.selectable and not layer.locked:
                mask.include(layer.index)
        return mask

    def get_statistics(self) -> dict[str, Any]:
        """Get layer statistics."""
        visible_count = sum(1 for l in self._layers.values() if l.visible)
        locked_count = sum(1 for l in self._layers.values() if l.locked)
        total_objects = sum(l.object_count for l in self._layers.values())

        return {
            "total_layers": len(self._layers),
            "visible_layers": visible_count,
            "hidden_layers": len(self._layers) - visible_count,
            "locked_layers": locked_count,
            "total_objects": total_objects,
            "is_isolated": self._isolated_layer_id is not None,
            "available_slots": self.MAX_LAYERS - len(self._layers),
        }


__all__ = [
    "Layer",
    "LayerSettings",
    "LayerManager",
    "LayerColor",
    "LayerBlendMode",
    "LayerMask",
]
