"""
HLOD Layer Management.

Manages HLOD layers for cells in the world partition system.
Each cell can have multiple HLOD layers at different detail levels,
with automatic selection based on camera distance.

References:
- WORLD_CONTEXT.md Section 7 HLOD System
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .generator import (
    AABB,
    HLODGenerationMethod,
    HLODGenerator,
    HLODMeshData,
    MeshData,
    MergeSettings,
    SimplificationSettings,
    Vec3,
)


# =============================================================================
# HLOD LAYER CONSTANTS
# =============================================================================


class HLODLayerConstants:
    """Constants for HLOD layer management."""
    # Default distance thresholds (in world units)
    DEFAULT_LOD0_DISTANCE: float = 500.0
    DEFAULT_LOD1_DISTANCE: float = 1000.0
    DEFAULT_LOD2_DISTANCE: float = 2000.0
    DEFAULT_LOD3_DISTANCE: float = 4000.0

    # Default simplification ratios per layer
    DEFAULT_LOD0_RATIO: float = 1.0      # Original quality
    DEFAULT_LOD1_RATIO: float = 0.5      # 50% triangles
    DEFAULT_LOD2_RATIO: float = 0.25     # 25% triangles
    DEFAULT_LOD3_RATIO: float = 0.1      # 10% triangles

    # Maximum layers supported
    MAX_LAYERS: int = 8

    # Minimum distance between thresholds
    MIN_THRESHOLD_GAP: float = 100.0


# =============================================================================
# HLOD CELL STATE
# =============================================================================


class HLODCellState(Enum):
    """State of an HLOD cell."""
    UNINITIALIZED = auto()    # Cell exists but no HLOD generated
    GENERATING = auto()        # HLOD generation in progress
    READY = auto()             # HLOD layers ready to use
    INVALID = auto()           # HLOD needs regeneration
    ERROR = auto()             # Generation failed


# =============================================================================
# HLOD LAYER CONFIGURATION
# =============================================================================


@dataclass
class HLODLayerConfig:
    """Configuration for a single HLOD layer."""
    layer_index: int = 0
    distance_threshold: float = HLODLayerConstants.DEFAULT_LOD0_DISTANCE
    generation_method: HLODGenerationMethod = HLODGenerationMethod.SIMPLIFICATION
    simplification_ratio: float = HLODLayerConstants.DEFAULT_LOD0_RATIO
    max_triangles: int = 0  # 0 = no limit
    use_impostor: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if self.distance_threshold < 0:
            raise ValueError("distance_threshold must be non-negative")
        if not 0.0 < self.simplification_ratio <= 1.0:
            raise ValueError("simplification_ratio must be in (0, 1]")

    @classmethod
    def create_default_configs(cls, layer_count: int = 4) -> List["HLODLayerConfig"]:
        """
        Create a default set of layer configurations.

        Args:
            layer_count: Number of layers to create

        Returns:
            List of layer configurations with increasing distance/simplification
        """
        configs: List[HLODLayerConfig] = []

        # Default thresholds and ratios
        distances = [500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0, 64000.0]
        ratios = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.01]

        for i in range(min(layer_count, HLODLayerConstants.MAX_LAYERS)):
            method = HLODGenerationMethod.SIMPLIFICATION
            use_impostor = False

            # Use impostor for very distant layers
            if i >= layer_count - 1 and layer_count > 2:
                method = HLODGenerationMethod.IMPOSTOR
                use_impostor = True

            configs.append(
                HLODLayerConfig(
                    layer_index=i,
                    distance_threshold=distances[i],
                    generation_method=method,
                    simplification_ratio=ratios[i],
                    use_impostor=use_impostor,
                )
            )

        return configs


# =============================================================================
# HLOD LAYER
# =============================================================================


@dataclass
class HLODLayer:
    """A single HLOD layer with mesh data."""
    config: HLODLayerConfig = field(default_factory=HLODLayerConfig)
    mesh_data: Optional[MeshData] = None
    bounds: Optional[AABB] = None
    source_actor_ids: List[int] = field(default_factory=list)
    is_generated: bool = False
    generation_error: Optional[str] = None

    @property
    def triangle_count(self) -> int:
        """Get triangle count for this layer."""
        if self.mesh_data:
            return self.mesh_data.get_triangle_count()
        return 0

    @property
    def vertex_count(self) -> int:
        """Get vertex count for this layer."""
        if self.mesh_data:
            return self.mesh_data.get_vertex_count()
        return 0

    def clear(self) -> None:
        """Clear layer data."""
        self.mesh_data = None
        self.is_generated = False
        self.generation_error = None


# =============================================================================
# HLOD CELL
# =============================================================================


@dataclass
class HLODCell:
    """
    An HLOD cell containing multiple detail layers.

    Each cell represents a region of the world that can be rendered
    at different detail levels based on camera distance.
    """
    cell_id: Tuple[int, int] = (0, 0)
    bounds: AABB = field(default_factory=AABB)
    layers: List[HLODLayer] = field(default_factory=list)
    source_meshes: List[MeshData] = field(default_factory=list)
    active_layer_index: int = 0
    state: HLODCellState = HLODCellState.UNINITIALIZED
    _is_dirty: bool = True

    @property
    def is_generated(self) -> bool:
        """Check if all layers are generated."""
        return len(self.layers) > 0 and all(layer.is_generated for layer in self.layers)

    @property
    def is_dirty(self) -> bool:
        """Check if cell needs regeneration."""
        return self._is_dirty

    @property
    def layer_count(self) -> int:
        """Get number of layers."""
        return len(self.layers)

    def generate_layers(
        self,
        layer_configs: List[HLODLayerConfig],
        generator: Optional[HLODGenerator] = None,
    ) -> None:
        """
        Generate HLOD layers from source meshes.

        Args:
            layer_configs: Configuration for each layer
            generator: HLOD generator to use (creates default if None)
        """
        if not self.source_meshes:
            return

        if generator is None:
            generator = HLODGenerator()

        self.state = HLODCellState.GENERATING
        self.layers = []

        try:
            for config in layer_configs:
                layer = HLODLayer(config=config, bounds=self.bounds)

                # Configure generator for this layer
                simplification_settings = SimplificationSettings(
                    target_ratio=config.simplification_ratio,
                )
                generator.configure(simplification_settings=simplification_settings)
                generator.method = config.generation_method

                # Generate HLOD mesh
                hlod_data = generator.generate(
                    source_meshes=self.source_meshes,
                    bounds=self.bounds,
                )

                layer.mesh_data = hlod_data.mesh
                layer.is_generated = True
                layer.source_actor_ids = list(range(len(self.source_meshes)))

                self.layers.append(layer)

            self.state = HLODCellState.READY
            self._is_dirty = False

        except Exception as e:
            self.state = HLODCellState.ERROR
            # Store error on first layer if it exists
            if self.layers:
                self.layers[-1].generation_error = str(e)

    def get_active_layer(self, camera_distance: float) -> Optional[HLODLayer]:
        """
        Get the appropriate layer for a given camera distance.

        Args:
            camera_distance: Distance from camera to cell center

        Returns:
            The layer to use, or None if no layers available
        """
        if not self.layers:
            return None

        # Validate camera distance
        if camera_distance < 0.0:
            camera_distance = 0.0

        # Find the highest-index layer whose threshold is less than camera distance
        selected_index = 0

        for i, layer in enumerate(self.layers):
            if camera_distance >= layer.config.distance_threshold:
                selected_index = i

        # Validate selected_index is within bounds (defensive check)
        if selected_index < 0:
            selected_index = 0
        elif selected_index >= len(self.layers):
            selected_index = len(self.layers) - 1

        self.active_layer_index = selected_index
        return self.layers[selected_index]

    def get_layer(self, index: int) -> Optional[HLODLayer]:
        """Get layer by index."""
        if 0 <= index < len(self.layers):
            return self.layers[index]
        return None

    def invalidate(self) -> None:
        """Mark cell as needing regeneration."""
        self._is_dirty = True
        self.state = HLODCellState.INVALID

    def clear_layers(self) -> None:
        """Clear all layer data."""
        for layer in self.layers:
            layer.clear()
        self.layers = []
        self.state = HLODCellState.UNINITIALIZED
        self._is_dirty = True

    def add_source_mesh(self, mesh: MeshData) -> None:
        """Add a source mesh and invalidate."""
        self.source_meshes.append(mesh)
        self.invalidate()

    def remove_source_mesh(self, index: int) -> bool:
        """Remove a source mesh by index and invalidate."""
        if 0 <= index < len(self.source_meshes):
            del self.source_meshes[index]
            self.invalidate()
            return True
        return False

    def clear_source_meshes(self) -> None:
        """Clear all source meshes and invalidate."""
        self.source_meshes = []
        self.invalidate()


# =============================================================================
# HLOD LAYER MANAGER
# =============================================================================


class HLODLayerManager:
    """
    Manages HLOD layers across all cells in the world.
    """

    def __init__(
        self,
        layer_configs: Optional[List[HLODLayerConfig]] = None,
    ) -> None:
        """
        Initialize the layer manager.

        Args:
            layer_configs: Default layer configurations
        """
        self._layer_configs = layer_configs or HLODLayerConfig.create_default_configs()
        self._cells: Dict[Tuple[int, int], HLODCell] = {}
        self._generator = HLODGenerator()
        self._generation_queue: List[Tuple[int, int]] = []

    @property
    def layer_configs(self) -> List[HLODLayerConfig]:
        return self._layer_configs

    @layer_configs.setter
    def layer_configs(self, configs: List[HLODLayerConfig]) -> None:
        self._layer_configs = configs
        # Invalidate all cells when configs change
        for cell in self._cells.values():
            cell.invalidate()

    @property
    def cell_count(self) -> int:
        """Get number of managed cells."""
        return len(self._cells)

    @property
    def cells(self) -> Dict[Tuple[int, int], HLODCell]:
        """Get all cells (read-only view)."""
        return self._cells

    def add_cell(
        self,
        cell_id: Tuple[int, int],
        bounds: AABB,
        source_meshes: List[MeshData],
    ) -> HLODCell:
        """
        Add a new cell to the manager.

        Args:
            cell_id: Grid coordinates (x, y) of the cell
            bounds: Bounding box of the cell
            source_meshes: Source meshes for HLOD generation

        Returns:
            The created cell
        """
        cell = HLODCell(
            cell_id=cell_id,
            bounds=bounds,
            source_meshes=source_meshes,
        )

        self._cells[cell_id] = cell

        return cell

    def remove_cell(self, cell_id: Tuple[int, int]) -> bool:
        """
        Remove a cell from the manager.

        Args:
            cell_id: Grid coordinates of the cell

        Returns:
            True if cell was removed
        """
        if cell_id in self._cells:
            del self._cells[cell_id]
            # Remove from generation queue if present
            if cell_id in self._generation_queue:
                self._generation_queue.remove(cell_id)
            return True
        return False

    def get_cell(self, cell_id: Tuple[int, int]) -> Optional[HLODCell]:
        """
        Get a cell by its ID.

        Args:
            cell_id: Grid coordinates of the cell

        Returns:
            The cell, or None if not found
        """
        return self._cells.get(cell_id)

    def generate_cell_hlod(
        self,
        cell_id: Tuple[int, int],
        force: bool = False,
    ) -> bool:
        """
        Generate HLOD for a specific cell.

        Args:
            cell_id: Grid coordinates of the cell
            force: Force regeneration even if not dirty

        Returns:
            True if generation was performed
        """
        cell = self._cells.get(cell_id)
        if not cell:
            return False

        if not force and not cell.is_dirty:
            return False

        cell.generate_layers(self._layer_configs, self._generator)
        return True

    def invalidate_cell(self, cell_id: Tuple[int, int]) -> bool:
        """
        Mark a cell as needing regeneration.

        Args:
            cell_id: Grid coordinates of the cell

        Returns:
            True if cell was invalidated
        """
        cell = self._cells.get(cell_id)
        if cell:
            cell.invalidate()
            if cell_id not in self._generation_queue:
                self._generation_queue.append(cell_id)
            return True
        return False

    def rebuild_all(self, force: bool = False) -> int:
        """
        Rebuild HLOD for all cells.

        Args:
            force: Force regeneration even if not dirty

        Returns:
            Number of cells rebuilt
        """
        rebuilt = 0
        for cell_id in self._cells:
            if self.generate_cell_hlod(cell_id, force):
                rebuilt += 1
        return rebuilt

    def rebuild_dirty(self) -> int:
        """
        Rebuild HLOD for all dirty cells.

        Returns:
            Number of cells rebuilt
        """
        rebuilt = 0
        for cell_id in list(self._generation_queue):
            if self.generate_cell_hlod(cell_id):
                rebuilt += 1
                self._generation_queue.remove(cell_id)
        return rebuilt

    def get_dirty_cells(self) -> List[Tuple[int, int]]:
        """Get list of cells that need regeneration."""
        return [
            cell_id for cell_id, cell in self._cells.items()
            if cell.is_dirty
        ]

    def get_cells_in_range(
        self,
        center: Vec3,
        radius: float,
    ) -> List[HLODCell]:
        """
        Get all cells within a distance from a point.

        Args:
            center: Center point
            radius: Search radius

        Returns:
            List of cells within range
        """
        result: List[HLODCell] = []

        for cell in self._cells.values():
            cell_center = cell.bounds.center
            distance = center.distance_to(cell_center)

            if distance <= radius:
                result.append(cell)

        return result

    def update_cell_source_meshes(
        self,
        cell_id: Tuple[int, int],
        source_meshes: List[MeshData],
    ) -> bool:
        """
        Update source meshes for a cell.

        Args:
            cell_id: Grid coordinates of the cell
            source_meshes: New source meshes

        Returns:
            True if cell was updated
        """
        cell = self._cells.get(cell_id)
        if not cell:
            return False

        cell.source_meshes = source_meshes
        cell.invalidate()

        if cell_id not in self._generation_queue:
            self._generation_queue.append(cell_id)

        return True

    def configure_generator(
        self,
        merge_settings: Optional[MergeSettings] = None,
        simplification_settings: Optional[SimplificationSettings] = None,
    ) -> None:
        """Configure the underlying HLOD generator."""
        self._generator.configure(
            merge_settings=merge_settings,
            simplification_settings=simplification_settings,
        )


# =============================================================================
# HLOD CLUSTER
# =============================================================================


@dataclass
class HLODCluster:
    """
    A cluster of HLOD cells that can be combined for distant rendering.

    Used for hierarchical HLOD where multiple cells are merged into
    a single representation at very far distances.
    """
    cluster_id: int = 0
    cells: List[HLODCell] = field(default_factory=list)
    combined_bounds: AABB = field(default_factory=AABB)
    combined_layers: List[HLODLayer] = field(default_factory=list)
    is_generated: bool = False

    @property
    def cell_count(self) -> int:
        """Get number of cells in cluster."""
        return len(self.cells)

    @property
    def total_triangle_count(self) -> int:
        """Get total triangles across all cells."""
        total = 0
        for cell in self.cells:
            for layer in cell.layers:
                total += layer.triangle_count
        return total

    def add_cell(self, cell: HLODCell) -> None:
        """Add a cell to the cluster."""
        self.cells.append(cell)
        self._update_bounds()
        self.is_generated = False

    def remove_cell(self, cell: HLODCell) -> bool:
        """Remove a cell from the cluster."""
        if cell in self.cells:
            self.cells.remove(cell)
            self._update_bounds()
            self.is_generated = False
            return True
        return False

    def _update_bounds(self) -> None:
        """Update combined bounds from cells."""
        self.combined_bounds = AABB()
        for cell in self.cells:
            self.combined_bounds = self.combined_bounds.merge(cell.bounds)

    def get_combined_layer(
        self,
        layer_index: int,
        generator: Optional[HLODGenerator] = None,
    ) -> Optional[MeshData]:
        """
        Get a combined mesh for a specific layer across all cells.

        Args:
            layer_index: Layer index to combine
            generator: Generator for mesh merging

        Returns:
            Combined mesh data, or None if not available
        """
        if not self.cells:
            return None

        # Collect meshes from all cells at this layer
        meshes: List[MeshData] = []

        for cell in self.cells:
            layer = cell.get_layer(layer_index)
            if layer and layer.mesh_data:
                meshes.append(layer.mesh_data)

        if not meshes:
            return None

        # Merge meshes
        if generator is None:
            generator = HLODGenerator(HLODGenerationMethod.MESH_MERGING)

        result = generator.generate(
            source_meshes=meshes,
            bounds=self.combined_bounds,
            method=HLODGenerationMethod.MESH_MERGING,
        )

        return result.mesh

    def generate_combined_layers(
        self,
        layer_configs: List[HLODLayerConfig],
        generator: Optional[HLODGenerator] = None,
    ) -> None:
        """
        Generate combined layers for the cluster.

        Args:
            layer_configs: Layer configurations
            generator: HLOD generator
        """
        if generator is None:
            generator = HLODGenerator()

        self.combined_layers = []

        for config in layer_configs:
            combined_mesh = self.get_combined_layer(
                config.layer_index,
                generator,
            )

            layer = HLODLayer(
                config=config,
                mesh_data=combined_mesh,
                bounds=self.combined_bounds,
                is_generated=combined_mesh is not None,
            )

            self.combined_layers.append(layer)

        self.is_generated = True

    def invalidate(self) -> None:
        """Mark cluster as needing regeneration."""
        self.is_generated = False
        self.combined_layers = []


# =============================================================================
# HLOD HIERARCHY MANAGER
# =============================================================================


class HLODHierarchyManager:
    """
    Manages hierarchical HLOD with nested clusters.

    Supports multiple levels of detail:
    - Level 0: Individual cells
    - Level 1: Clusters of cells
    - Level 2+: Clusters of clusters
    """

    def __init__(
        self,
        layer_manager: HLODLayerManager,
        cells_per_cluster: int = 4,
        max_hierarchy_levels: int = 3,
    ) -> None:
        """
        Initialize hierarchy manager.

        Args:
            layer_manager: Underlying layer manager
            cells_per_cluster: Number of cells per cluster
            max_hierarchy_levels: Maximum nesting levels
        """
        self._layer_manager = layer_manager
        self._cells_per_cluster = cells_per_cluster
        self._max_levels = max_hierarchy_levels
        self._clusters: Dict[int, Dict[int, HLODCluster]] = {}  # level -> id -> cluster

    @property
    def layer_manager(self) -> HLODLayerManager:
        return self._layer_manager

    @property
    def cluster_levels(self) -> int:
        """Get number of cluster levels."""
        return len(self._clusters)

    def build_hierarchy(self) -> None:
        """Build the complete HLOD hierarchy from cells."""
        self._clusters = {}

        # Level 1: Group cells into clusters
        cells = list(self._layer_manager.cells.values())
        if not cells:
            return

        # Simple grid-based clustering
        cluster_id = 0
        self._clusters[1] = {}

        for i in range(0, len(cells), self._cells_per_cluster):
            cluster = HLODCluster(cluster_id=cluster_id)

            for j in range(i, min(i + self._cells_per_cluster, len(cells))):
                cluster.add_cell(cells[j])

            self._clusters[1][cluster_id] = cluster
            cluster_id += 1

        # Higher levels: cluster the clusters
        for level in range(2, self._max_levels + 1):
            prev_level = level - 1
            if prev_level not in self._clusters:
                break

            prev_clusters = list(self._clusters[prev_level].values())
            if len(prev_clusters) <= 1:
                break

            self._clusters[level] = {}
            cluster_id = 0

            # Create meta-clusters (simplified - just pairs for now)
            for i in range(0, len(prev_clusters), 2):
                meta_cluster = HLODCluster(cluster_id=cluster_id)

                # Add cells from sub-clusters
                for j in range(i, min(i + 2, len(prev_clusters))):
                    for cell in prev_clusters[j].cells:
                        meta_cluster.add_cell(cell)

                self._clusters[level][cluster_id] = meta_cluster
                cluster_id += 1

    def get_cluster(self, level: int, cluster_id: int) -> Optional[HLODCluster]:
        """Get a specific cluster."""
        if level in self._clusters:
            return self._clusters[level].get(cluster_id)
        return None

    def get_clusters_at_level(self, level: int) -> List[HLODCluster]:
        """Get all clusters at a hierarchy level."""
        if level in self._clusters:
            return list(self._clusters[level].values())
        return []

    def invalidate_hierarchy(self) -> None:
        """Invalidate the entire hierarchy."""
        for level_clusters in self._clusters.values():
            for cluster in level_clusters.values():
                cluster.invalidate()


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Constants
    "HLODLayerConstants",
    # Enums
    "HLODCellState",
    # Configuration
    "HLODLayerConfig",
    # Core classes
    "HLODLayer",
    "HLODCell",
    "HLODLayerManager",
    "HLODCluster",
    "HLODHierarchyManager",
]
