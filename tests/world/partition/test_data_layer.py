"""
Tests for the DataLayer module.

Tests layer types, loading modes, and cell layer management.
"""

import pytest
from engine.world.partition.cell import CellCoord, CellState, StreamingCell
from engine.world.partition.data_layer import (
    DataLayer,
    DataLayerCellData,
    DataLayerLoadMode,
    DataLayerManager,
    DataLayerState,
    DataLayerType,
)


# =============================================================================
# DataLayerCellData Tests
# =============================================================================

class TestDataLayerCellData:
    """Tests for DataLayerCellData class."""

    def test_cell_data_creation_default(self):
        """Test default cell data creation."""
        data = DataLayerCellData()
        assert data.coord == CellCoord(0, 0)
        assert data.state == DataLayerState.UNLOADED
        assert data.load_progress == 0.0

    def test_cell_data_creation_custom(self):
        """Test custom cell data creation."""
        data = DataLayerCellData(
            coord=CellCoord(5, 10),
            state=DataLayerState.LOADED,
            memory_bytes=1024,
        )
        assert data.coord.x == 5
        assert data.state == DataLayerState.LOADED
        assert data.memory_bytes == 1024

    def test_cell_data_is_loaded(self):
        """Test is_loaded property."""
        data = DataLayerCellData()
        assert data.is_loaded is False

        data.state = DataLayerState.LOADED
        assert data.is_loaded is True

    def test_cell_data_actor_ids(self):
        """Test actor ID list."""
        data = DataLayerCellData()
        data.actor_ids.append("actor_1")
        data.actor_ids.append("actor_2")
        assert len(data.actor_ids) == 2

    def test_cell_data_asset_refs(self):
        """Test asset reference list."""
        data = DataLayerCellData()
        data.asset_refs.append("mesh_001")
        data.asset_refs.append("texture_002")
        assert len(data.asset_refs) == 2


# =============================================================================
# DataLayer Tests
# =============================================================================

class TestDataLayerCreation:
    """Tests for DataLayer creation."""

    def test_layer_creation_default(self):
        """Test default layer creation."""
        layer = DataLayer()
        assert layer.name == ""
        assert layer.layer_type == DataLayerType.DEFAULT
        assert layer.load_mode == DataLayerLoadMode.STREAMED
        assert layer.is_enabled is True

    def test_layer_creation_custom(self):
        """Test custom layer creation."""
        layer = DataLayer(
            name="gameplay",
            layer_type=DataLayerType.GAMEPLAY,
            load_mode=DataLayerLoadMode.ALWAYS_LOADED,
            priority=10,
        )
        assert layer.name == "gameplay"
        assert layer.layer_type == DataLayerType.GAMEPLAY
        assert layer.load_mode == DataLayerLoadMode.ALWAYS_LOADED
        assert layer.priority == 10


class TestDataLayerCellManagement:
    """Tests for layer cell data management."""

    def test_get_cell_data_nonexistent(self):
        """Test getting non-existent cell data."""
        layer = DataLayer()
        data = layer.get_cell_data(CellCoord(5, 5))
        assert data is None

    def test_get_or_create_cell_data(self):
        """Test getting or creating cell data."""
        layer = DataLayer()
        data = layer.get_or_create_cell_data(CellCoord(5, 5))
        assert data is not None
        assert data.coord.x == 5
        assert data.coord.y == 5

    def test_get_or_create_returns_same(self):
        """Test get_or_create returns same instance."""
        layer = DataLayer()
        data1 = layer.get_or_create_cell_data(CellCoord(5, 5))
        data2 = layer.get_or_create_cell_data(CellCoord(5, 5))
        assert data1 is data2

    def test_has_cell_data(self):
        """Test checking for cell data existence."""
        layer = DataLayer()
        assert layer.has_cell_data(CellCoord(5, 5)) is False
        layer.get_or_create_cell_data(CellCoord(5, 5))
        assert layer.has_cell_data(CellCoord(5, 5)) is True


class TestDataLayerLoading:
    """Tests for layer loading operations."""

    def test_load_cell(self):
        """Test loading cell data."""
        layer = DataLayer()
        result = layer.load_cell(CellCoord(5, 5))
        assert result is True
        assert layer.is_cell_loaded(CellCoord(5, 5)) is True

    def test_load_cell_already_loaded(self):
        """Test loading already loaded cell."""
        layer = DataLayer()
        layer.load_cell(CellCoord(5, 5))
        result = layer.load_cell(CellCoord(5, 5))
        assert result is False

    def test_load_cell_disabled_layer(self):
        """Test loading cell on disabled layer."""
        layer = DataLayer()
        layer.is_enabled = False
        result = layer.load_cell(CellCoord(5, 5))
        assert result is False

    def test_unload_cell(self):
        """Test unloading cell data."""
        layer = DataLayer()
        layer.load_cell(CellCoord(5, 5))
        result = layer.unload_cell(CellCoord(5, 5))
        assert result is True
        assert layer.is_cell_loaded(CellCoord(5, 5)) is False

    def test_unload_cell_not_loaded(self):
        """Test unloading non-loaded cell."""
        layer = DataLayer()
        result = layer.unload_cell(CellCoord(5, 5))
        assert result is False

    def test_unload_clears_data(self):
        """Test unloading clears cell data."""
        layer = DataLayer()
        layer.load_cell(CellCoord(5, 5))
        layer.set_cell_data(CellCoord(5, 5), {"test": "data"}, 1024)
        layer.add_actor_to_cell(CellCoord(5, 5), "actor_1")
        layer.unload_cell(CellCoord(5, 5))

        data = layer.get_cell_data(CellCoord(5, 5))
        assert data.data is None
        assert len(data.actor_ids) == 0
        assert data.memory_bytes == 0


class TestDataLayerCellQueries:
    """Tests for layer cell queries."""

    def test_is_cell_loaded_true(self):
        """Test is_cell_loaded returns True for loaded cell."""
        layer = DataLayer()
        layer.load_cell(CellCoord(5, 5))
        assert layer.is_cell_loaded(CellCoord(5, 5)) is True

    def test_is_cell_loaded_false(self):
        """Test is_cell_loaded returns False for non-loaded cell."""
        layer = DataLayer()
        assert layer.is_cell_loaded(CellCoord(5, 5)) is False

    def test_get_loaded_cells(self):
        """Test getting all loaded cells."""
        layer = DataLayer()
        layer.load_cell(CellCoord(1, 1))
        layer.load_cell(CellCoord(2, 2))
        layer.load_cell(CellCoord(3, 3))

        loaded = layer.get_loaded_cells()
        assert len(loaded) == 3


class TestDataLayerDataManagement:
    """Tests for layer data management."""

    def test_set_cell_data(self):
        """Test setting cell data."""
        layer = DataLayer()
        layer.set_cell_data(CellCoord(5, 5), {"key": "value"}, 512)
        data = layer.get_cell_data(CellCoord(5, 5))
        assert data.data == {"key": "value"}
        assert data.memory_bytes == 512

    def test_add_actor_to_cell(self):
        """Test adding actor to cell."""
        layer = DataLayer()
        layer.add_actor_to_cell(CellCoord(5, 5), "actor_1")
        layer.add_actor_to_cell(CellCoord(5, 5), "actor_2")
        data = layer.get_cell_data(CellCoord(5, 5))
        assert len(data.actor_ids) == 2

    def test_add_actor_duplicate(self):
        """Test adding duplicate actor is prevented."""
        layer = DataLayer()
        layer.add_actor_to_cell(CellCoord(5, 5), "actor_1")
        layer.add_actor_to_cell(CellCoord(5, 5), "actor_1")
        data = layer.get_cell_data(CellCoord(5, 5))
        assert len(data.actor_ids) == 1

    def test_remove_actor_from_cell(self):
        """Test removing actor from cell."""
        layer = DataLayer()
        layer.add_actor_to_cell(CellCoord(5, 5), "actor_1")
        result = layer.remove_actor_from_cell(CellCoord(5, 5), "actor_1")
        assert result is True
        data = layer.get_cell_data(CellCoord(5, 5))
        assert len(data.actor_ids) == 0

    def test_remove_actor_not_found(self):
        """Test removing non-existent actor."""
        layer = DataLayer()
        result = layer.remove_actor_from_cell(CellCoord(5, 5), "actor_1")
        assert result is False

    def test_add_asset_ref(self):
        """Test adding asset reference."""
        layer = DataLayer()
        layer.add_asset_ref(CellCoord(5, 5), "mesh_001")
        data = layer.get_cell_data(CellCoord(5, 5))
        assert "mesh_001" in data.asset_refs


class TestDataLayerCallbacks:
    """Tests for layer callbacks."""

    def test_on_load_callback(self):
        """Test load callback is invoked."""
        layer = DataLayer()
        loaded = []
        layer.on_load(lambda l, c: loaded.append((l.name, c)))
        layer.name = "test_layer"
        layer.load_cell(CellCoord(5, 5))
        assert len(loaded) == 1
        assert loaded[0][0] == "test_layer"

    def test_on_unload_callback(self):
        """Test unload callback is invoked."""
        layer = DataLayer()
        unloaded = []
        layer.on_unload(lambda l, c: unloaded.append((l.name, c)))
        layer.name = "test_layer"
        layer.load_cell(CellCoord(5, 5))
        layer.unload_cell(CellCoord(5, 5))
        assert len(unloaded) == 1


class TestDataLayerMemory:
    """Tests for layer memory tracking."""

    def test_get_memory_usage(self):
        """Test getting total memory usage."""
        layer = DataLayer()
        layer.set_cell_data(CellCoord(1, 1), None, 1024)
        layer.set_cell_data(CellCoord(2, 2), None, 2048)
        assert layer.get_memory_usage() == 3072

    def test_clear_layer(self):
        """Test clearing all cell data."""
        layer = DataLayer()
        layer.load_cell(CellCoord(1, 1))
        layer.load_cell(CellCoord(2, 2))
        layer.clear()
        assert len(layer.cell_data) == 0


# =============================================================================
# DataLayerManager Tests
# =============================================================================

class TestDataLayerManagerCreation:
    """Tests for DataLayerManager creation."""

    def test_manager_creation(self):
        """Test manager creation."""
        manager = DataLayerManager()
        assert len(manager.layers) == 0

    def test_register_layer(self):
        """Test registering a layer."""
        manager = DataLayerManager()
        layer = DataLayer(name="gameplay")
        manager.register_layer(layer)
        assert len(manager.layers) == 1
        assert "gameplay" in manager.layers

    def test_unregister_layer(self):
        """Test unregistering a layer."""
        manager = DataLayerManager()
        layer = DataLayer(name="gameplay")
        manager.register_layer(layer)
        result = manager.unregister_layer("gameplay")
        assert result is True
        assert len(manager.layers) == 0

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent layer."""
        manager = DataLayerManager()
        result = manager.unregister_layer("nonexistent")
        assert result is False


class TestDataLayerManagerAccess:
    """Tests for layer access methods."""

    def test_get_layer(self):
        """Test getting layer by name."""
        manager = DataLayerManager()
        layer = DataLayer(name="gameplay", layer_type=DataLayerType.GAMEPLAY)
        manager.register_layer(layer)
        found = manager.get_layer("gameplay")
        assert found is layer

    def test_get_layer_not_found(self):
        """Test getting non-existent layer."""
        manager = DataLayerManager()
        found = manager.get_layer("nonexistent")
        assert found is None

    def test_get_layers_by_type(self):
        """Test getting layers by type."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="gameplay1", layer_type=DataLayerType.GAMEPLAY))
        manager.register_layer(DataLayer(name="gameplay2", layer_type=DataLayerType.GAMEPLAY))
        manager.register_layer(DataLayer(name="audio", layer_type=DataLayerType.AUDIO))

        gameplay_layers = manager.get_layers_by_type(DataLayerType.GAMEPLAY)
        assert len(gameplay_layers) == 2


class TestDataLayerManagerCellOperations:
    """Tests for cell loading operations through manager."""

    def test_load_layers_for_cell(self):
        """Test loading all layers for a cell."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="layer1"))
        manager.register_layer(DataLayer(name="layer2"))

        cell = StreamingCell(coord=CellCoord(5, 5))
        count = manager.load_layers_for_cell(cell)
        assert count == 2

    def test_load_layers_filtered_by_type(self):
        """Test loading layers filtered by type."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="gameplay", layer_type=DataLayerType.GAMEPLAY))
        manager.register_layer(DataLayer(name="audio", layer_type=DataLayerType.AUDIO))

        cell = StreamingCell(coord=CellCoord(5, 5))
        count = manager.load_layers_for_cell(cell, layer_types=[DataLayerType.GAMEPLAY])
        assert count == 1

    def test_load_layers_skips_manual(self):
        """Test loading skips MANUAL mode layers."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="auto", load_mode=DataLayerLoadMode.STREAMED))
        manager.register_layer(DataLayer(name="manual", load_mode=DataLayerLoadMode.MANUAL))

        cell = StreamingCell(coord=CellCoord(5, 5))
        count = manager.load_layers_for_cell(cell)
        assert count == 1

    def test_unload_layers_for_cell(self):
        """Test unloading all layers for a cell."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="layer1"))
        manager.register_layer(DataLayer(name="layer2"))

        cell = StreamingCell(coord=CellCoord(5, 5))
        manager.load_layers_for_cell(cell)
        count = manager.unload_layers_for_cell(cell)
        assert count == 2

    def test_unload_layers_skips_always_loaded(self):
        """Test unloading skips ALWAYS_LOADED layers."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="stream", load_mode=DataLayerLoadMode.STREAMED))
        manager.register_layer(DataLayer(name="always", load_mode=DataLayerLoadMode.ALWAYS_LOADED))

        cell = StreamingCell(coord=CellCoord(5, 5))
        manager.load_layers_for_cell(cell)
        count = manager.unload_layers_for_cell(cell)
        assert count == 1  # Only streamed layer unloaded


class TestDataLayerManagerSingleLayerOps:
    """Tests for single layer operations."""

    def test_load_single_layer(self):
        """Test loading a specific layer."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="gameplay"))

        result = manager.load_layer("gameplay", CellCoord(5, 5))
        assert result is True

    def test_unload_single_layer(self):
        """Test unloading a specific layer."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="gameplay"))
        manager.load_layer("gameplay", CellCoord(5, 5))

        result = manager.unload_layer("gameplay", CellCoord(5, 5))
        assert result is True

    def test_is_layer_loaded(self):
        """Test checking if layer is loaded for cell."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="gameplay"))

        assert manager.is_layer_loaded("gameplay", CellCoord(5, 5)) is False
        manager.load_layer("gameplay", CellCoord(5, 5))
        assert manager.is_layer_loaded("gameplay", CellCoord(5, 5)) is True


class TestDataLayerManagerUtilities:
    """Tests for manager utility methods."""

    def test_set_layer_enabled(self):
        """Test enabling/disabling layer."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="gameplay"))

        result = manager.set_layer_enabled("gameplay", False)
        assert result is True
        assert manager.get_layer("gameplay").is_enabled is False

    def test_set_layer_enabled_not_found(self):
        """Test enabling non-existent layer."""
        manager = DataLayerManager()
        result = manager.set_layer_enabled("nonexistent", False)
        assert result is False

    def test_get_loaded_layers_for_cell(self):
        """Test getting loaded layers for a cell."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="layer1"))
        manager.register_layer(DataLayer(name="layer2"))

        manager.load_layer("layer1", CellCoord(5, 5))
        loaded = manager.get_loaded_layers_for_cell(CellCoord(5, 5))
        assert len(loaded) == 1

    def test_get_total_memory_usage(self):
        """Test getting total memory across all layers."""
        manager = DataLayerManager()
        layer1 = DataLayer(name="layer1")
        layer2 = DataLayer(name="layer2")
        manager.register_layer(layer1)
        manager.register_layer(layer2)

        layer1.set_cell_data(CellCoord(1, 1), None, 1024)
        layer2.set_cell_data(CellCoord(1, 1), None, 2048)

        assert manager.get_total_memory_usage() == 3072

    def test_get_layer_stats(self):
        """Test getting layer statistics."""
        manager = DataLayerManager()
        layer = DataLayer(name="gameplay", layer_type=DataLayerType.GAMEPLAY)
        manager.register_layer(layer)
        layer.load_cell(CellCoord(5, 5))

        stats = manager.get_layer_stats()
        assert "gameplay" in stats
        assert stats["gameplay"]["type"] == "GAMEPLAY"
        assert stats["gameplay"]["loaded_cells"] == 1

    def test_create_default_layers(self):
        """Test creating default layer set."""
        manager = DataLayerManager()
        manager.create_default_layers()
        assert len(manager.layers) == 8
        assert "runtime" in manager.layers
        assert "gameplay" in manager.layers
        assert "foliage" in manager.layers

    def test_clear_all(self):
        """Test clearing all layers."""
        manager = DataLayerManager()
        manager.create_default_layers()
        manager.clear_all()
        assert len(manager.layers) == 0

    def test_manager_len(self):
        """Test manager length."""
        manager = DataLayerManager()
        manager.register_layer(DataLayer(name="layer1"))
        manager.register_layer(DataLayer(name="layer2"))
        assert len(manager) == 2

    def test_manager_iteration(self):
        """Test iterating over manager."""
        manager = DataLayerManager()
        layer1 = DataLayer(name="layer1", priority=10)
        layer2 = DataLayer(name="layer2", priority=5)
        manager.register_layer(layer1)
        manager.register_layer(layer2)

        # Should iterate in priority order (highest first)
        layers = list(manager)
        assert layers[0].name == "layer1"  # Higher priority
        assert layers[1].name == "layer2"
