"""
Tests for the layers module.

Tests visibility, locking, and filtering.
"""

import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.leveleditor.layers import (
    Layer,
    LayerSettings,
    LayerManager,
    LayerColor,
    LayerBlendMode,
    LayerMask,
)
from foundation.tracker import tracker


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset tracker state before each test."""
    tracker._dirty.clear()
    tracker._cb_global.clear()
    tracker._cb_type.clear()
    tracker._cb_obj.clear()
    tracker._undo.clear()
    tracker._redo.clear()
    tracker._txn = None
    yield


class TestLayerMask:
    """Tests for LayerMask class."""

    def test_all_layers(self):
        """All layers mask should include everything."""
        mask = LayerMask.all_layers()
        assert mask.includes(0) is True
        assert mask.includes(15) is True
        assert mask.includes(31) is True

    def test_no_layers(self):
        """No layers mask should include nothing."""
        mask = LayerMask.no_layers()
        assert mask.includes(0) is False
        assert mask.includes(15) is False

    def test_include(self):
        """Should include specific layer."""
        mask = LayerMask(0)
        mask.include(5)
        assert mask.includes(5) is True
        assert mask.includes(4) is False

    def test_exclude(self):
        """Should exclude specific layer."""
        mask = LayerMask.all_layers()
        mask.exclude(5)
        assert mask.includes(5) is False
        assert mask.includes(4) is True

    def test_toggle(self):
        """Should toggle layer inclusion."""
        mask = LayerMask(0)
        mask.toggle(3)
        assert mask.includes(3) is True
        mask.toggle(3)
        assert mask.includes(3) is False


class TestLayer:
    """Tests for Layer class."""

    def test_creation(self):
        """Layer should initialize with name and defaults."""
        layer = Layer("TestLayer", 0)
        assert layer.name == "TestLayer"
        assert layer.index == 0
        assert layer.visible is True
        assert layer.locked is False

    def test_unique_id(self):
        """Each layer should have unique ID."""
        layer1 = Layer("A", 0)
        layer2 = Layer("B", 1)
        assert layer1.id != layer2.id

    def test_index_clamped(self):
        """Index should be clamped to 0-31."""
        layer1 = Layer("Test", -5)
        layer2 = Layer("Test", 50)
        assert layer1.index == 0
        assert layer2.index == 31

    def test_color_preset(self):
        """Should use color preset."""
        layer = Layer("Test", 0, LayerColor.RED)
        assert layer.color == LayerColor.RED
        assert layer.rgb == (1.0, 0.0, 0.0)

    def test_custom_color(self):
        """Should use custom color."""
        layer = Layer("Test", 0)
        layer.set_custom_color(0.5, 0.5, 0.5)
        assert layer.rgb == (0.5, 0.5, 0.5)

    def test_custom_color_clamped(self):
        """Custom color should be clamped to 0-1."""
        layer = Layer("Test", 0)
        layer.set_custom_color(1.5, -0.5, 0.5)
        r, g, b = layer.rgb
        assert r == 1.0
        assert g == 0.0
        assert b == 0.5

    def test_visibility(self):
        """Should toggle visibility."""
        layer = Layer("Test", 0)
        assert layer.visible is True

        layer.visible = False
        assert layer.visible is False

    def test_toggle_visibility(self):
        """Toggle should return new state."""
        layer = Layer("Test", 0)
        result = layer.toggle_visibility()
        assert result is False
        assert layer.visible is False

    def test_locked(self):
        """Should toggle lock state."""
        layer = Layer("Test", 0)
        assert layer.locked is False

        layer.locked = True
        assert layer.locked is True

    def test_toggle_lock(self):
        """Toggle should return new state."""
        layer = Layer("Test", 0)
        result = layer.toggle_lock()
        assert result is True
        assert layer.locked is True

    def test_add_object(self):
        """Should add object to layer."""
        layer = Layer("Test", 0)
        result = layer.add_object("obj-1")

        assert result is True
        assert layer.contains("obj-1") is True
        assert layer.object_count == 1

    def test_add_duplicate_object(self):
        """Adding duplicate should return False."""
        layer = Layer("Test", 0)
        layer.add_object("obj-1")
        result = layer.add_object("obj-1")

        assert result is False
        assert layer.object_count == 1

    def test_remove_object(self):
        """Should remove object from layer."""
        layer = Layer("Test", 0)
        layer.add_object("obj-1")

        result = layer.remove_object("obj-1")

        assert result is True
        assert layer.contains("obj-1") is False

    def test_remove_nonexistent_object(self):
        """Removing nonexistent should return False."""
        layer = Layer("Test", 0)
        result = layer.remove_object("nonexistent")

        assert result is False

    def test_clear_objects(self):
        """Should clear all objects."""
        layer = Layer("Test", 0)
        layer.add_object("obj-1")
        layer.add_object("obj-2")

        count = layer.clear_objects()

        assert count == 2
        assert layer.object_count == 0

    def test_bit_mask(self):
        """Should return correct bit mask."""
        layer0 = Layer("L0", 0)
        layer5 = Layer("L5", 5)

        assert layer0.bit_mask == 1
        assert layer5.bit_mask == 32

    def test_metadata(self):
        """Should store metadata."""
        layer = Layer("Test", 0)
        layer.set_metadata("custom", "value")

        assert layer.get_metadata("custom") == "value"
        assert layer.get_metadata("missing", "default") == "default"

    def test_object_ids(self):
        """Should return copy of object IDs."""
        layer = Layer("Test", 0)
        layer.add_object("obj-1")
        layer.add_object("obj-2")

        ids = layer.object_ids
        assert "obj-1" in ids
        assert "obj-2" in ids

        # Should be a copy
        ids.append("obj-3")
        assert layer.object_count == 2


class TestLayerManager:
    """Tests for LayerManager class."""

    def test_creation(self):
        """Manager should initialize with default layer."""
        manager = LayerManager()
        assert manager.layer_count == 1
        assert manager.default_layer is not None
        assert manager.active_layer is manager.default_layer

    def test_create_layer(self):
        """Should create new layer."""
        manager = LayerManager()
        layer = manager.create_layer("Custom", LayerColor.BLUE)

        assert layer.name == "Custom"
        assert layer.color == LayerColor.BLUE
        assert manager.get_layer(layer.id) is layer

    def test_create_layer_max_limit(self):
        """Should raise error at max layers."""
        manager = LayerManager()
        for i in range(31):  # Already have default, add 31 more = 32 total
            manager.create_layer(f"Layer{i}")

        with pytest.raises(ValueError):
            manager.create_layer("OneMore")

    def test_delete_layer(self):
        """Should delete layer."""
        manager = LayerManager()
        layer = manager.create_layer("ToDelete")

        result = manager.delete_layer(layer.id)

        assert result is True
        assert manager.get_layer(layer.id) is None

    def test_delete_default_layer_fails(self):
        """Should not delete default layer."""
        manager = LayerManager()
        result = manager.delete_layer(manager.default_layer.id)

        assert result is False
        assert manager.default_layer is not None

    def test_delete_moves_objects(self):
        """Deleting layer should move objects to default."""
        manager = LayerManager()
        layer = manager.create_layer("ToDelete")
        layer.add_object("obj-1")

        manager.delete_layer(layer.id)

        assert manager.default_layer.contains("obj-1") is True

    def test_get_layer_by_name(self):
        """Should get layer by name."""
        manager = LayerManager()
        layer = manager.create_layer("Custom")

        found = manager.get_layer_by_name("Custom")

        assert found is layer

    def test_get_layer_by_index(self):
        """Should get layer by index."""
        manager = LayerManager()
        layer = manager.create_layer("Custom")

        found = manager.get_layer_by_index(layer.index)

        assert found is layer

    def test_get_all_layers(self):
        """Should get all layers in order."""
        manager = LayerManager()
        manager.create_layer("Layer1")
        manager.create_layer("Layer2")

        layers = manager.get_all_layers()

        assert len(layers) == 3

    def test_get_layers_by_mask(self):
        """Should filter layers by mask."""
        manager = LayerManager()
        layer1 = manager.create_layer("L1")
        layer2 = manager.create_layer("L2")

        mask = LayerMask(0)
        mask.include(layer1.index)

        layers = manager.get_layers_by_mask(mask)

        assert layer1 in layers
        assert layer2 not in layers

    def test_set_active_layer(self):
        """Should change active layer."""
        manager = LayerManager()
        layer = manager.create_layer("Custom")

        result = manager.set_active_layer(layer.id)

        assert result is True
        assert manager.active_layer is layer

    def test_assign_object_to_layer(self):
        """Should assign object to layer."""
        manager = LayerManager()
        layer = manager.create_layer("Custom")

        result = manager.assign_object_to_layer("obj-1", layer.id)

        assert result is True
        assert layer.contains("obj-1") is True

    def test_assign_moves_from_previous(self):
        """Assigning should remove from previous layer."""
        manager = LayerManager()
        layer1 = manager.create_layer("L1")
        layer2 = manager.create_layer("L2")

        manager.assign_object_to_layer("obj-1", layer1.id)
        manager.assign_object_to_layer("obj-1", layer2.id)

        assert layer1.contains("obj-1") is False
        assert layer2.contains("obj-1") is True

    def test_get_object_layer(self):
        """Should get layer containing object."""
        manager = LayerManager()
        layer = manager.create_layer("Custom")
        manager.assign_object_to_layer("obj-1", layer.id)

        found = manager.get_object_layer("obj-1")

        assert found is layer

    def test_isolate_layer(self):
        """Should isolate single layer."""
        manager = LayerManager()
        layer1 = manager.create_layer("L1")
        layer2 = manager.create_layer("L2")

        result = manager.isolate_layer(layer1.id)

        assert result is True
        assert manager.isolated_layer is layer1
        assert layer1.visible is True
        assert layer2.visible is False

    def test_exit_isolation(self):
        """Should restore visibility after isolation."""
        manager = LayerManager()
        layer1 = manager.create_layer("L1")
        layer2 = manager.create_layer("L2")

        layer2.visible = False  # Pre-isolation state

        manager.isolate_layer(layer1.id)
        result = manager.exit_isolation()

        assert result is True
        assert manager.isolated_layer is None
        # Should restore pre-isolation state
        assert layer2.visible is False

    def test_set_all_visible(self):
        """Should set all layers visible."""
        manager = LayerManager()
        layer = manager.create_layer("Custom")
        layer.visible = False

        manager.set_all_visible(True)

        assert layer.visible is True

    def test_set_all_locked(self):
        """Should lock all layers."""
        manager = LayerManager()
        layer = manager.create_layer("Custom")

        manager.set_all_locked(True)

        assert layer.locked is True
        assert manager.default_layer.locked is True

    def test_invert_visibility(self):
        """Should invert all layer visibility."""
        manager = LayerManager()
        layer = manager.create_layer("Custom")

        initial_visible = layer.visible
        manager.invert_visibility()

        assert layer.visible != initial_visible

    def test_reorder_layer(self):
        """Should change layer order."""
        manager = LayerManager()
        layer1 = manager.create_layer("L1")
        layer2 = manager.create_layer("L2")

        manager.reorder_layer(layer2.id, 0)

        layers = manager.get_all_layers()
        # Should be: L2, Default, L1 (or similar reordering)
        assert layers[0].id == layer2.id

    def test_merge_layers(self):
        """Should merge layers."""
        manager = LayerManager()
        layer1 = manager.create_layer("L1")
        layer2 = manager.create_layer("L2")

        layer1.add_object("obj-1")
        layer1.add_object("obj-2")

        result = manager.merge_layers(layer1.id, layer2.id)

        assert result is True
        assert layer2.contains("obj-1") is True
        assert layer2.contains("obj-2") is True
        assert manager.get_layer(layer1.id) is None

    def test_merge_default_fails(self):
        """Should not merge default layer."""
        manager = LayerManager()
        layer = manager.create_layer("Custom")

        result = manager.merge_layers(manager.default_layer.id, layer.id)

        assert result is False

    def test_create_visibility_mask(self):
        """Should create mask of visible layers."""
        manager = LayerManager()
        layer1 = manager.create_layer("L1")
        layer2 = manager.create_layer("L2")
        layer2.visible = False

        mask = manager.create_visibility_mask()

        assert mask.includes(manager.default_layer.index) is True
        assert mask.includes(layer1.index) is True
        assert mask.includes(layer2.index) is False

    def test_create_selection_mask(self):
        """Should create mask of selectable layers."""
        manager = LayerManager()
        layer1 = manager.create_layer("L1")
        layer2 = manager.create_layer("L2")
        layer2.locked = True

        mask = manager.create_selection_mask()

        assert mask.includes(layer1.index) is True
        assert mask.includes(layer2.index) is False

    def test_callbacks(self):
        """Should trigger callbacks."""
        manager = LayerManager()
        events = []

        manager.on("on_layer_create", lambda l: events.append(("create", l)))
        manager.on("on_layer_delete", lambda l: events.append(("delete", l)))

        layer = manager.create_layer("Test")
        manager.delete_layer(layer.id)

        assert len(events) == 2

    def test_get_statistics(self):
        """Should return statistics."""
        manager = LayerManager()
        layer1 = manager.create_layer("L1")
        layer2 = manager.create_layer("L2")
        layer2.locked = True
        layer1.add_object("obj-1")

        stats = manager.get_statistics()

        assert stats["total_layers"] == 3
        assert stats["locked_layers"] == 1
        assert stats["total_objects"] == 1
