"""
Data layers for world partition.

Provides a system for organizing cell content into layers that can be
independently loaded based on gameplay needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from engine.world.partition.cell import CellCoord, StreamingCell
from engine.world.partition.constants import (
    DEFAULT_LAYER_LOAD_DISTANCE,
    LAYER_PRIORITY_RUNTIME,
    LAYER_PRIORITY_LANDSCAPE,
    LAYER_PRIORITY_GAMEPLAY,
    LAYER_PRIORITY_LIGHTING,
    LAYER_PRIORITY_NAVIGATION,
    LAYER_PRIORITY_FOLIAGE,
    LAYER_PRIORITY_AUDIO,
    LAYER_PRIORITY_VFX,
    LAYER_DISTANCE_GAMEPLAY,
    LAYER_DISTANCE_LIGHTING,
    LAYER_DISTANCE_FOLIAGE,
    LAYER_DISTANCE_VFX,
    LAYER_DISTANCE_AUDIO,
)


class DataLayerType(Enum):
    """Types of data layers for organizing cell content."""
    RUNTIME = auto()     # Always loaded with the cell
    DEFAULT = auto()     # Standard content layer
    GAMEPLAY = auto()    # Interactive objects, triggers
    LANDSCAPE = auto()   # Terrain data
    FOLIAGE = auto()     # Vegetation, grass
    AUDIO = auto()       # Sound actors, ambient audio
    NAVIGATION = auto()  # NavMesh data
    LIGHTING = auto()    # Light sources, probes
    VFX = auto()         # Particles, decals


class DataLayerLoadMode(Enum):
    """How a data layer should be loaded."""
    ALWAYS_LOADED = auto()  # Load with the cell
    STREAMED = auto()       # Stream based on distance
    BLUEPRINT = auto()      # Only load when explicitly requested
    MANUAL = auto()         # Manual load/unload control


class DataLayerState(Enum):
    """Current state of a data layer."""
    UNLOADED = auto()
    LOADING = auto()
    LOADED = auto()
    UNLOADING = auto()


@dataclass
class DataLayerCellData:
    """Data for a specific cell within a layer."""
    coord: CellCoord = field(default_factory=lambda: CellCoord(0, 0))
    state: DataLayerState = DataLayerState.UNLOADED
    load_progress: float = 0.0
    data: Any = None
    memory_bytes: int = 0

    # References to actual content
    actor_ids: List[str] = field(default_factory=list)
    asset_refs: List[str] = field(default_factory=list)

    @property
    def is_loaded(self) -> bool:
        """Check if the cell data is loaded."""
        return self.state == DataLayerState.LOADED


@dataclass
class DataLayer:
    """
    A data layer organizing a specific type of content across cells.

    Layers allow selective loading of content types (e.g., load gameplay
    but not audio for distant cells).
    """
    name: str = ""
    layer_type: DataLayerType = DataLayerType.DEFAULT
    load_mode: DataLayerLoadMode = DataLayerLoadMode.STREAMED

    # Per-cell data
    cell_data: Dict[Tuple[int, int], DataLayerCellData] = field(default_factory=dict)

    # Layer state
    is_enabled: bool = True
    is_loaded: bool = False  # Global layer state (for ALWAYS_LOADED mode)

    # Load configuration
    load_distance: float = DEFAULT_LAYER_LOAD_DISTANCE  # Distance for STREAMED mode
    priority: int = 0  # Higher priority loads first

    # Callbacks
    _on_load_callbacks: List[Callable[["DataLayer", CellCoord], None]] = field(
        default_factory=list, repr=False
    )
    _on_unload_callbacks: List[Callable[["DataLayer", CellCoord], None]] = field(
        default_factory=list, repr=False
    )

    def get_cell_data(self, coord: CellCoord) -> Optional[DataLayerCellData]:
        """
        Get data for a specific cell.

        Args:
            coord: Cell coordinates.

        Returns:
            Cell data if exists, None otherwise.
        """
        return self.cell_data.get((coord.x, coord.y))

    def get_or_create_cell_data(self, coord: CellCoord) -> DataLayerCellData:
        """
        Get or create data for a specific cell.

        Args:
            coord: Cell coordinates.

        Returns:
            Cell data.
        """
        key = (coord.x, coord.y)
        if key not in self.cell_data:
            self.cell_data[key] = DataLayerCellData(coord=coord)
        return self.cell_data[key]

    def has_cell_data(self, coord: CellCoord) -> bool:
        """Check if data exists for a cell."""
        return (coord.x, coord.y) in self.cell_data

    def load_cell(self, coord: CellCoord) -> bool:
        """
        Load data for a specific cell.

        Args:
            coord: Cell coordinates.

        Returns:
            True if load started successfully.
        """
        if not self.is_enabled:
            return False

        cell_data = self.get_or_create_cell_data(coord)

        if cell_data.state != DataLayerState.UNLOADED:
            return False

        cell_data.state = DataLayerState.LOADING
        cell_data.load_progress = 0.0

        # Simulate loading (actual implementation would be async)
        cell_data.state = DataLayerState.LOADED
        cell_data.load_progress = 1.0

        for callback in self._on_load_callbacks:
            callback(self, coord)

        return True

    def unload_cell(self, coord: CellCoord) -> bool:
        """
        Unload data for a specific cell.

        Args:
            coord: Cell coordinates.

        Returns:
            True if unload started successfully.
        """
        cell_data = self.get_cell_data(coord)

        if cell_data is None or cell_data.state != DataLayerState.LOADED:
            return False

        cell_data.state = DataLayerState.UNLOADING

        for callback in self._on_unload_callbacks:
            callback(self, coord)

        # Clear data
        cell_data.data = None
        cell_data.actor_ids.clear()
        cell_data.asset_refs.clear()
        cell_data.memory_bytes = 0
        cell_data.state = DataLayerState.UNLOADED
        cell_data.load_progress = 0.0

        return True

    def is_cell_loaded(self, coord: CellCoord) -> bool:
        """Check if a cell's data is loaded."""
        cell_data = self.get_cell_data(coord)
        return cell_data is not None and cell_data.is_loaded

    def get_loaded_cells(self) -> List[CellCoord]:
        """Get all loaded cell coordinates."""
        return [
            CellCoord(x, y)
            for (x, y), data in self.cell_data.items()
            if data.state == DataLayerState.LOADED
        ]

    def get_memory_usage(self) -> int:
        """Get total memory usage across all cells."""
        return sum(data.memory_bytes for data in self.cell_data.values())

    def set_cell_data(self, coord: CellCoord, data: Any, memory_bytes: int = 0) -> None:
        """
        Set the data for a cell.

        Args:
            coord: Cell coordinates.
            data: Data to store.
            memory_bytes: Estimated memory usage.
        """
        cell_data = self.get_or_create_cell_data(coord)
        cell_data.data = data
        cell_data.memory_bytes = memory_bytes

    def add_actor_to_cell(self, coord: CellCoord, actor_id: str) -> None:
        """Add an actor reference to a cell."""
        cell_data = self.get_or_create_cell_data(coord)
        if actor_id not in cell_data.actor_ids:
            cell_data.actor_ids.append(actor_id)

    def remove_actor_from_cell(self, coord: CellCoord, actor_id: str) -> bool:
        """Remove an actor reference from a cell."""
        cell_data = self.get_cell_data(coord)
        if cell_data and actor_id in cell_data.actor_ids:
            cell_data.actor_ids.remove(actor_id)
            return True
        return False

    def add_asset_ref(self, coord: CellCoord, asset_ref: str) -> None:
        """Add an asset reference to a cell."""
        cell_data = self.get_or_create_cell_data(coord)
        if asset_ref not in cell_data.asset_refs:
            cell_data.asset_refs.append(asset_ref)

    def on_load(self, callback: Callable[["DataLayer", CellCoord], None]) -> None:
        """Register a callback for cell load."""
        self._on_load_callbacks.append(callback)

    def on_unload(self, callback: Callable[["DataLayer", CellCoord], None]) -> None:
        """Register a callback for cell unload."""
        self._on_unload_callbacks.append(callback)

    def clear(self) -> None:
        """Clear all cell data."""
        self.cell_data.clear()
        self.is_loaded = False


class DataLayerManager:
    """
    Manages multiple data layers and their loading.

    Coordinates which layers are loaded for each cell based on
    distance, priority, and load mode.
    """

    def __init__(self) -> None:
        """Initialize the data layer manager."""
        self.layers: Dict[str, DataLayer] = {}
        self._layer_order: List[str] = []  # For priority ordering

    def register_layer(self, layer: DataLayer) -> None:
        """
        Register a data layer.

        Args:
            layer: Layer to register.
        """
        self.layers[layer.name] = layer
        self._update_layer_order()

    def unregister_layer(self, name: str) -> bool:
        """
        Unregister a data layer.

        Args:
            name: Name of the layer to remove.

        Returns:
            True if found and removed.
        """
        if name in self.layers:
            del self.layers[name]
            self._update_layer_order()
            return True
        return False

    def _update_layer_order(self) -> None:
        """Update the layer loading order based on priority."""
        self._layer_order = sorted(
            self.layers.keys(),
            key=lambda n: self.layers[n].priority,
            reverse=True,
        )

    def get_layer(self, name: str) -> Optional[DataLayer]:
        """Get a layer by name."""
        return self.layers.get(name)

    def get_layers_by_type(self, layer_type: DataLayerType) -> List[DataLayer]:
        """Get all layers of a specific type."""
        return [
            layer for layer in self.layers.values()
            if layer.layer_type == layer_type
        ]

    def load_layers_for_cell(
        self,
        cell: StreamingCell,
        layer_types: Optional[List[DataLayerType]] = None,
    ) -> int:
        """
        Load data layers for a cell.

        Args:
            cell: Cell to load layers for.
            layer_types: Optional filter for specific layer types.

        Returns:
            Number of layers loaded.
        """
        loaded_count = 0

        for name in self._layer_order:
            layer = self.layers[name]

            if not layer.is_enabled:
                continue

            if layer_types and layer.layer_type not in layer_types:
                continue

            if layer.load_mode == DataLayerLoadMode.MANUAL:
                continue

            if layer.load_cell(cell.coord):
                loaded_count += 1

        return loaded_count

    def unload_layers_for_cell(
        self,
        cell: StreamingCell,
        layer_types: Optional[List[DataLayerType]] = None,
    ) -> int:
        """
        Unload data layers for a cell.

        Args:
            cell: Cell to unload layers for.
            layer_types: Optional filter for specific layer types.

        Returns:
            Number of layers unloaded.
        """
        unloaded_count = 0

        # Unload in reverse priority order
        for name in reversed(self._layer_order):
            layer = self.layers[name]

            if layer_types and layer.layer_type not in layer_types:
                continue

            # Don't unload ALWAYS_LOADED layers
            if layer.load_mode == DataLayerLoadMode.ALWAYS_LOADED:
                continue

            if layer.unload_cell(cell.coord):
                unloaded_count += 1

        return unloaded_count

    def load_layer(self, layer_name: str, coord: CellCoord) -> bool:
        """
        Load a specific layer for a cell.

        Args:
            layer_name: Name of the layer.
            coord: Cell coordinates.

        Returns:
            True if loaded successfully.
        """
        layer = self.get_layer(layer_name)
        if layer:
            return layer.load_cell(coord)
        return False

    def unload_layer(self, layer_name: str, coord: CellCoord) -> bool:
        """
        Unload a specific layer for a cell.

        Args:
            layer_name: Name of the layer.
            coord: Cell coordinates.

        Returns:
            True if unloaded successfully.
        """
        layer = self.get_layer(layer_name)
        if layer:
            return layer.unload_cell(coord)
        return False

    def is_layer_loaded(self, layer_name: str, coord: CellCoord) -> bool:
        """Check if a layer is loaded for a cell."""
        layer = self.get_layer(layer_name)
        if layer:
            return layer.is_cell_loaded(coord)
        return False

    def get_loaded_layers_for_cell(self, coord: CellCoord) -> List[DataLayer]:
        """Get all layers that are loaded for a cell."""
        return [
            layer for layer in self.layers.values()
            if layer.is_cell_loaded(coord)
        ]

    def set_layer_enabled(self, layer_name: str, enabled: bool) -> bool:
        """
        Enable or disable a layer.

        Args:
            layer_name: Name of the layer.
            enabled: Whether to enable.

        Returns:
            True if layer found and updated.
        """
        layer = self.get_layer(layer_name)
        if layer:
            layer.is_enabled = enabled
            return True
        return False

    def get_total_memory_usage(self) -> int:
        """Get total memory usage across all layers."""
        return sum(layer.get_memory_usage() for layer in self.layers.values())

    def get_layer_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all layers."""
        return {
            name: {
                "type": layer.layer_type.name,
                "load_mode": layer.load_mode.name,
                "enabled": layer.is_enabled,
                "loaded_cells": len(layer.get_loaded_cells()),
                "total_cells": len(layer.cell_data),
                "memory_bytes": layer.get_memory_usage(),
            }
            for name, layer in self.layers.items()
        }

    def create_default_layers(self) -> None:
        """Create a set of default data layers."""
        default_layers = [
            DataLayer(
                name="runtime",
                layer_type=DataLayerType.RUNTIME,
                load_mode=DataLayerLoadMode.ALWAYS_LOADED,
                priority=LAYER_PRIORITY_RUNTIME,
            ),
            DataLayer(
                name="landscape",
                layer_type=DataLayerType.LANDSCAPE,
                load_mode=DataLayerLoadMode.ALWAYS_LOADED,
                priority=LAYER_PRIORITY_LANDSCAPE,
            ),
            DataLayer(
                name="gameplay",
                layer_type=DataLayerType.GAMEPLAY,
                load_mode=DataLayerLoadMode.STREAMED,
                load_distance=LAYER_DISTANCE_GAMEPLAY,
                priority=LAYER_PRIORITY_GAMEPLAY,
            ),
            DataLayer(
                name="foliage",
                layer_type=DataLayerType.FOLIAGE,
                load_mode=DataLayerLoadMode.STREAMED,
                load_distance=LAYER_DISTANCE_FOLIAGE,
                priority=LAYER_PRIORITY_FOLIAGE,
            ),
            DataLayer(
                name="lighting",
                layer_type=DataLayerType.LIGHTING,
                load_mode=DataLayerLoadMode.STREAMED,
                load_distance=LAYER_DISTANCE_LIGHTING,
                priority=LAYER_PRIORITY_LIGHTING,
            ),
            DataLayer(
                name="audio",
                layer_type=DataLayerType.AUDIO,
                load_mode=DataLayerLoadMode.STREAMED,
                load_distance=LAYER_DISTANCE_AUDIO,
                priority=LAYER_PRIORITY_AUDIO,
            ),
            DataLayer(
                name="navigation",
                layer_type=DataLayerType.NAVIGATION,
                load_mode=DataLayerLoadMode.BLUEPRINT,
                priority=LAYER_PRIORITY_NAVIGATION,
            ),
            DataLayer(
                name="vfx",
                layer_type=DataLayerType.VFX,
                load_mode=DataLayerLoadMode.STREAMED,
                load_distance=LAYER_DISTANCE_VFX,
                priority=LAYER_PRIORITY_VFX,
            ),
        ]

        for layer in default_layers:
            self.register_layer(layer)

    def clear_all(self) -> None:
        """Clear all layers and their data."""
        for layer in self.layers.values():
            layer.clear()
        self.layers.clear()
        self._layer_order.clear()

    def __len__(self) -> int:
        """Get the number of layers."""
        return len(self.layers)

    def __iter__(self):
        """Iterate over layers in priority order."""
        for name in self._layer_order:
            yield self.layers[name]
