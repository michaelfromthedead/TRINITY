"""
Tests for HLOD layer management.

Tests HLOD layers, cells, layer manager, clusters, and hierarchy.
"""

import pytest

from engine.world.hlod.layers import (
    HLODLayerConstants,
    HLODCellState,
    HLODLayerConfig,
    HLODLayer,
    HLODCell,
    HLODLayerManager,
    HLODCluster,
    HLODHierarchyManager,
)
from engine.world.hlod.generator import (
    AABB,
    Vec3,
    Vec2,
    MeshData,
    HLODGenerationMethod,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_mesh() -> MeshData:
    """Create a simple triangle mesh."""
    mesh = MeshData(
        vertices=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.5, 1.0, 0.0),
        ],
        normals=[Vec3(0.0, 0.0, 1.0)] * 3,
        uvs=[Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(0.5, 1.0)],
        indices=[0, 1, 2],
    )
    mesh.compute_bounds()
    return mesh


@pytest.fixture
def source_meshes() -> list:
    """Create multiple source meshes."""
    meshes = []
    for i in range(3):
        mesh = MeshData(
            vertices=[
                Vec3(i * 2, 0, 0),
                Vec3(i * 2 + 1, 0, 0),
                Vec3(i * 2 + 0.5, 1, 0),
            ],
            normals=[Vec3(0, 0, 1)] * 3,
            uvs=[Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1)],
            indices=[0, 1, 2],
        )
        mesh.compute_bounds()
        meshes.append(mesh)
    return meshes


@pytest.fixture
def default_layer_configs() -> list:
    """Create default layer configurations."""
    return HLODLayerConfig.create_default_configs(3)


# =============================================================================
# HLOD LAYER CONFIG TESTS
# =============================================================================


class TestHLODLayerConfig:
    """Tests for HLODLayerConfig."""

    def test_default_creation(self) -> None:
        """Test default config creation."""
        config = HLODLayerConfig()
        assert config.layer_index == 0
        assert config.distance_threshold == HLODLayerConstants.DEFAULT_LOD0_DISTANCE
        assert config.simplification_ratio == HLODLayerConstants.DEFAULT_LOD0_RATIO

    def test_custom_creation(self) -> None:
        """Test custom config creation."""
        config = HLODLayerConfig(
            layer_index=2,
            distance_threshold=2000.0,
            generation_method=HLODGenerationMethod.IMPOSTOR,
            simplification_ratio=0.25,
        )
        assert config.layer_index == 2
        assert config.distance_threshold == 2000.0
        assert config.generation_method == HLODGenerationMethod.IMPOSTOR
        assert config.simplification_ratio == 0.25

    def test_invalid_layer_index(self) -> None:
        """Test invalid layer index."""
        with pytest.raises(ValueError):
            HLODLayerConfig(layer_index=-1)

    def test_invalid_distance_threshold(self) -> None:
        """Test invalid distance threshold."""
        with pytest.raises(ValueError):
            HLODLayerConfig(distance_threshold=-100.0)

    def test_invalid_simplification_ratio(self) -> None:
        """Test invalid simplification ratio."""
        with pytest.raises(ValueError):
            HLODLayerConfig(simplification_ratio=0.0)

        with pytest.raises(ValueError):
            HLODLayerConfig(simplification_ratio=1.5)

    def test_create_default_configs(self) -> None:
        """Test creating default configs."""
        configs = HLODLayerConfig.create_default_configs(4)

        assert len(configs) == 4
        assert configs[0].layer_index == 0
        assert configs[1].layer_index == 1
        assert configs[2].layer_index == 2
        assert configs[3].layer_index == 3

        # Distance thresholds should increase
        for i in range(1, len(configs)):
            assert configs[i].distance_threshold > configs[i - 1].distance_threshold

        # Simplification ratios should decrease
        for i in range(1, len(configs)):
            assert configs[i].simplification_ratio < configs[i - 1].simplification_ratio

    def test_create_default_configs_max_layers(self) -> None:
        """Test that default configs respects max layers."""
        configs = HLODLayerConfig.create_default_configs(100)
        assert len(configs) == HLODLayerConstants.MAX_LAYERS


# =============================================================================
# HLOD LAYER TESTS
# =============================================================================


class TestHLODLayer:
    """Tests for HLODLayer."""

    def test_default_creation(self) -> None:
        """Test default layer creation."""
        layer = HLODLayer()
        assert layer.mesh_data is None
        assert not layer.is_generated
        assert layer.triangle_count == 0
        assert layer.vertex_count == 0

    def test_with_mesh_data(self, simple_mesh: MeshData) -> None:
        """Test layer with mesh data."""
        layer = HLODLayer(
            config=HLODLayerConfig(),
            mesh_data=simple_mesh,
            is_generated=True,
        )
        assert layer.is_generated
        assert layer.triangle_count == 1
        assert layer.vertex_count == 3

    def test_clear(self, simple_mesh: MeshData) -> None:
        """Test layer clear."""
        layer = HLODLayer(
            mesh_data=simple_mesh,
            is_generated=True,
        )
        layer.clear()

        assert layer.mesh_data is None
        assert not layer.is_generated

    def test_generation_error(self) -> None:
        """Test storing generation error."""
        layer = HLODLayer()
        layer.generation_error = "Test error"
        assert layer.generation_error == "Test error"


# =============================================================================
# HLOD CELL TESTS
# =============================================================================


class TestHLODCell:
    """Tests for HLODCell."""

    def test_default_creation(self) -> None:
        """Test default cell creation."""
        cell = HLODCell()
        assert cell.cell_id == (0, 0)
        assert cell.state == HLODCellState.UNINITIALIZED
        assert not cell.is_generated
        assert cell.is_dirty
        assert cell.layer_count == 0

    def test_custom_creation(self, source_meshes: list) -> None:
        """Test custom cell creation."""
        bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        cell = HLODCell(
            cell_id=(5, 10),
            bounds=bounds,
            source_meshes=source_meshes,
        )

        assert cell.cell_id == (5, 10)
        assert len(cell.source_meshes) == 3

    def test_generate_layers(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> None:
        """Test layer generation."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(6, 1, 0))
        cell = HLODCell(
            cell_id=(0, 0),
            bounds=bounds,
            source_meshes=source_meshes,
        )

        cell.generate_layers(default_layer_configs)

        assert cell.state == HLODCellState.READY
        assert cell.layer_count == 3
        assert cell.is_generated
        assert not cell.is_dirty

    def test_get_active_layer_by_distance(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> None:
        """Test getting active layer based on distance."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(6, 1, 0))
        cell = HLODCell(
            cell_id=(0, 0),
            bounds=bounds,
            source_meshes=source_meshes,
        )
        cell.generate_layers(default_layer_configs)

        # Close distance should get LOD 0
        layer = cell.get_active_layer(100.0)
        assert layer is not None
        assert cell.active_layer_index == 0

        # Far distance should get higher LOD
        layer = cell.get_active_layer(1500.0)
        assert layer is not None
        assert cell.active_layer_index > 0

    def test_get_layer_by_index(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> None:
        """Test getting layer by index."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(6, 1, 0))
        cell = HLODCell(
            cell_id=(0, 0),
            bounds=bounds,
            source_meshes=source_meshes,
        )
        cell.generate_layers(default_layer_configs)

        layer0 = cell.get_layer(0)
        assert layer0 is not None

        layer1 = cell.get_layer(1)
        assert layer1 is not None

        # Invalid index returns None
        invalid = cell.get_layer(100)
        assert invalid is None

    def test_invalidate(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> None:
        """Test cell invalidation."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(6, 1, 0))
        cell = HLODCell(
            cell_id=(0, 0),
            bounds=bounds,
            source_meshes=source_meshes,
        )
        cell.generate_layers(default_layer_configs)
        assert not cell.is_dirty

        cell.invalidate()

        assert cell.is_dirty
        assert cell.state == HLODCellState.INVALID

    def test_clear_layers(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> None:
        """Test clearing layers."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(6, 1, 0))
        cell = HLODCell(
            cell_id=(0, 0),
            bounds=bounds,
            source_meshes=source_meshes,
        )
        cell.generate_layers(default_layer_configs)

        cell.clear_layers()

        assert cell.layer_count == 0
        assert cell.state == HLODCellState.UNINITIALIZED
        assert cell.is_dirty

    def test_add_source_mesh(self) -> None:
        """Test adding source mesh."""
        cell = HLODCell()
        mesh = MeshData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 1, 0)],
            indices=[0, 1, 2],
        )

        cell.add_source_mesh(mesh)

        assert len(cell.source_meshes) == 1
        assert cell.is_dirty

    def test_remove_source_mesh(self, source_meshes: list) -> None:
        """Test removing source mesh."""
        cell = HLODCell(source_meshes=source_meshes)

        result = cell.remove_source_mesh(0)

        assert result
        assert len(cell.source_meshes) == 2
        assert cell.is_dirty

    def test_remove_source_mesh_invalid_index(self) -> None:
        """Test removing source mesh with invalid index."""
        cell = HLODCell()
        result = cell.remove_source_mesh(0)
        assert not result

    def test_clear_source_meshes(self, source_meshes: list) -> None:
        """Test clearing source meshes."""
        cell = HLODCell(source_meshes=source_meshes)

        cell.clear_source_meshes()

        assert len(cell.source_meshes) == 0
        assert cell.is_dirty


# =============================================================================
# HLOD LAYER MANAGER TESTS
# =============================================================================


class TestHLODLayerManager:
    """Tests for HLODLayerManager."""

    @pytest.fixture
    def manager(self, default_layer_configs: list) -> HLODLayerManager:
        """Create a layer manager."""
        return HLODLayerManager(layer_configs=default_layer_configs)

    def test_creation(self, manager: HLODLayerManager) -> None:
        """Test manager creation."""
        assert manager.cell_count == 0
        assert len(manager.layer_configs) == 3

    def test_add_cell(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test adding cell."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        cell = manager.add_cell((0, 0), bounds, source_meshes)

        assert manager.cell_count == 1
        assert cell.cell_id == (0, 0)

    def test_remove_cell(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test removing cell."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        manager.add_cell((0, 0), bounds, source_meshes)

        result = manager.remove_cell((0, 0))

        assert result
        assert manager.cell_count == 0

    def test_remove_nonexistent_cell(self, manager: HLODLayerManager) -> None:
        """Test removing nonexistent cell."""
        result = manager.remove_cell((99, 99))
        assert not result

    def test_get_cell(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test getting cell."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        manager.add_cell((0, 0), bounds, source_meshes)

        cell = manager.get_cell((0, 0))
        assert cell is not None
        assert cell.cell_id == (0, 0)

    def test_get_nonexistent_cell(self, manager: HLODLayerManager) -> None:
        """Test getting nonexistent cell."""
        cell = manager.get_cell((99, 99))
        assert cell is None

    def test_generate_cell_hlod(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test generating HLOD for cell."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        manager.add_cell((0, 0), bounds, source_meshes)

        result = manager.generate_cell_hlod((0, 0))

        assert result
        cell = manager.get_cell((0, 0))
        assert cell.state == HLODCellState.READY

    def test_generate_cell_hlod_force(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test force regeneration."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        manager.add_cell((0, 0), bounds, source_meshes)
        manager.generate_cell_hlod((0, 0))

        # Without force, should not regenerate
        result = manager.generate_cell_hlod((0, 0), force=False)
        assert not result

        # With force, should regenerate
        result = manager.generate_cell_hlod((0, 0), force=True)
        assert result

    def test_invalidate_cell(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test invalidating cell."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        manager.add_cell((0, 0), bounds, source_meshes)
        manager.generate_cell_hlod((0, 0))

        result = manager.invalidate_cell((0, 0))

        assert result
        cell = manager.get_cell((0, 0))
        assert cell.is_dirty

    def test_rebuild_all(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test rebuilding all cells."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        manager.add_cell((0, 0), bounds, source_meshes)
        manager.add_cell((1, 0), bounds, source_meshes)

        rebuilt = manager.rebuild_all()

        assert rebuilt == 2

    def test_rebuild_dirty(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test rebuilding dirty cells."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        manager.add_cell((0, 0), bounds, source_meshes)
        manager.add_cell((1, 0), bounds, source_meshes)

        # Generate first, then invalidate one
        manager.rebuild_all()
        manager.invalidate_cell((0, 0))

        rebuilt = manager.rebuild_dirty()

        assert rebuilt == 1

    def test_get_dirty_cells(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test getting dirty cells."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        manager.add_cell((0, 0), bounds, source_meshes)
        manager.add_cell((1, 0), bounds, source_meshes)

        # All cells start dirty
        dirty = manager.get_dirty_cells()
        assert len(dirty) == 2

        # Generate all
        manager.rebuild_all()
        dirty = manager.get_dirty_cells()
        assert len(dirty) == 0

    def test_get_cells_in_range(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test getting cells in range."""
        # Add cells at different positions
        manager.add_cell(
            (0, 0),
            AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
            source_meshes,
        )
        manager.add_cell(
            (1, 0),
            AABB(Vec3(100, 0, 0), Vec3(110, 10, 10)),
            source_meshes,
        )

        # Query from origin
        cells = manager.get_cells_in_range(Vec3(0, 0, 0), 50.0)

        # Only first cell should be in range
        assert len(cells) == 1
        assert cells[0].cell_id == (0, 0)

    def test_update_cell_source_meshes(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test updating cell source meshes."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        manager.add_cell((0, 0), bounds, source_meshes)
        manager.generate_cell_hlod((0, 0))

        # Update with new meshes
        new_meshes = source_meshes[:1]  # Just first mesh
        result = manager.update_cell_source_meshes((0, 0), new_meshes)

        assert result
        cell = manager.get_cell((0, 0))
        assert len(cell.source_meshes) == 1
        assert cell.is_dirty

    def test_layer_config_update_invalidates_cells(
        self,
        manager: HLODLayerManager,
        source_meshes: list,
    ) -> None:
        """Test that changing layer configs invalidates cells."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        manager.add_cell((0, 0), bounds, source_meshes)
        manager.generate_cell_hlod((0, 0))

        # Change configs
        new_configs = HLODLayerConfig.create_default_configs(2)
        manager.layer_configs = new_configs

        cell = manager.get_cell((0, 0))
        assert cell.is_dirty


# =============================================================================
# HLOD CLUSTER TESTS
# =============================================================================


class TestHLODCluster:
    """Tests for HLODCluster."""

    @pytest.fixture
    def populated_cell(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> HLODCell:
        """Create a populated cell with generated layers."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        cell = HLODCell(
            cell_id=(0, 0),
            bounds=bounds,
            source_meshes=source_meshes,
        )
        cell.generate_layers(default_layer_configs)
        return cell

    def test_creation(self) -> None:
        """Test cluster creation."""
        cluster = HLODCluster(cluster_id=0)
        assert cluster.cluster_id == 0
        assert cluster.cell_count == 0
        assert not cluster.is_generated

    def test_add_cell(self, populated_cell: HLODCell) -> None:
        """Test adding cell to cluster."""
        cluster = HLODCluster(cluster_id=0)
        cluster.add_cell(populated_cell)

        assert cluster.cell_count == 1
        assert not cluster.is_generated  # Adding cell invalidates

    def test_remove_cell(self, populated_cell: HLODCell) -> None:
        """Test removing cell from cluster."""
        cluster = HLODCluster(cluster_id=0)
        cluster.add_cell(populated_cell)

        result = cluster.remove_cell(populated_cell)

        assert result
        assert cluster.cell_count == 0

    def test_remove_nonexistent_cell(self) -> None:
        """Test removing nonexistent cell."""
        cluster = HLODCluster()
        cell = HLODCell()

        result = cluster.remove_cell(cell)

        assert not result

    def test_combined_bounds(self, source_meshes: list) -> None:
        """Test combined bounds computation."""
        cluster = HLODCluster(cluster_id=0)

        cell1 = HLODCell(
            cell_id=(0, 0),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
            source_meshes=source_meshes,
        )
        cell2 = HLODCell(
            cell_id=(1, 0),
            bounds=AABB(Vec3(20, 0, 0), Vec3(30, 10, 10)),
            source_meshes=source_meshes,
        )

        cluster.add_cell(cell1)
        cluster.add_cell(cell2)

        # Combined bounds should span both cells
        assert cluster.combined_bounds.min_point.x == pytest.approx(0.0)
        assert cluster.combined_bounds.max_point.x == pytest.approx(30.0)

    def test_total_triangle_count(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> None:
        """Test total triangle count across cells."""
        cluster = HLODCluster(cluster_id=0)

        cell1 = HLODCell(
            cell_id=(0, 0),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
            source_meshes=source_meshes,
        )
        cell1.generate_layers(default_layer_configs)

        cluster.add_cell(cell1)

        # Should have triangles from generated layers
        assert cluster.total_triangle_count > 0

    def test_get_combined_layer(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> None:
        """Test getting combined layer mesh."""
        cluster = HLODCluster(cluster_id=0)

        cell = HLODCell(
            cell_id=(0, 0),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
            source_meshes=source_meshes,
        )
        cell.generate_layers(default_layer_configs)
        cluster.add_cell(cell)

        combined = cluster.get_combined_layer(0)

        assert combined is not None
        assert combined.get_triangle_count() > 0

    def test_generate_combined_layers(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> None:
        """Test generating combined layers."""
        cluster = HLODCluster(cluster_id=0)

        cell = HLODCell(
            cell_id=(0, 0),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
            source_meshes=source_meshes,
        )
        cell.generate_layers(default_layer_configs)
        cluster.add_cell(cell)

        cluster.generate_combined_layers(default_layer_configs)

        assert cluster.is_generated
        assert len(cluster.combined_layers) == len(default_layer_configs)

    def test_invalidate(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> None:
        """Test cluster invalidation."""
        cluster = HLODCluster(cluster_id=0)

        cell = HLODCell(
            cell_id=(0, 0),
            bounds=AABB(Vec3(0, 0, 0), Vec3(10, 10, 10)),
            source_meshes=source_meshes,
        )
        cell.generate_layers(default_layer_configs)
        cluster.add_cell(cell)
        cluster.generate_combined_layers(default_layer_configs)

        cluster.invalidate()

        assert not cluster.is_generated
        assert len(cluster.combined_layers) == 0


# =============================================================================
# HLOD HIERARCHY MANAGER TESTS
# =============================================================================


class TestHLODHierarchyManager:
    """Tests for HLODHierarchyManager."""

    @pytest.fixture
    def populated_manager(
        self,
        source_meshes: list,
        default_layer_configs: list,
    ) -> HLODLayerManager:
        """Create a layer manager with populated cells."""
        manager = HLODLayerManager(layer_configs=default_layer_configs)

        # Add multiple cells
        for i in range(4):
            bounds = AABB(
                Vec3(i * 20, 0, 0),
                Vec3(i * 20 + 10, 10, 10),
            )
            manager.add_cell((i, 0), bounds, source_meshes)
            manager.generate_cell_hlod((i, 0))

        return manager

    def test_creation(
        self,
        populated_manager: HLODLayerManager,
    ) -> None:
        """Test hierarchy manager creation."""
        hierarchy = HLODHierarchyManager(populated_manager)

        assert hierarchy.layer_manager is populated_manager
        assert hierarchy.cluster_levels == 0

    def test_build_hierarchy(
        self,
        populated_manager: HLODLayerManager,
    ) -> None:
        """Test building hierarchy."""
        hierarchy = HLODHierarchyManager(populated_manager)
        hierarchy.build_hierarchy()

        # Should have at least one level of clusters
        assert hierarchy.cluster_levels >= 1

    def test_get_cluster(
        self,
        populated_manager: HLODLayerManager,
    ) -> None:
        """Test getting cluster by level and ID."""
        hierarchy = HLODHierarchyManager(populated_manager)
        hierarchy.build_hierarchy()

        cluster = hierarchy.get_cluster(1, 0)
        assert cluster is not None

    def test_get_nonexistent_cluster(
        self,
        populated_manager: HLODLayerManager,
    ) -> None:
        """Test getting nonexistent cluster."""
        hierarchy = HLODHierarchyManager(populated_manager)

        cluster = hierarchy.get_cluster(99, 99)
        assert cluster is None

    def test_get_clusters_at_level(
        self,
        populated_manager: HLODLayerManager,
    ) -> None:
        """Test getting all clusters at a level."""
        hierarchy = HLODHierarchyManager(populated_manager)
        hierarchy.build_hierarchy()

        clusters = hierarchy.get_clusters_at_level(1)
        assert len(clusters) > 0

    def test_invalidate_hierarchy(
        self,
        populated_manager: HLODLayerManager,
    ) -> None:
        """Test invalidating hierarchy."""
        hierarchy = HLODHierarchyManager(populated_manager)
        hierarchy.build_hierarchy()

        # Generate combined layers
        for cluster in hierarchy.get_clusters_at_level(1):
            cluster.generate_combined_layers(populated_manager.layer_configs)

        hierarchy.invalidate_hierarchy()

        # All clusters should be invalidated
        for cluster in hierarchy.get_clusters_at_level(1):
            assert not cluster.is_generated


# =============================================================================
# CELL STATE TESTS
# =============================================================================


class TestHLODCellState:
    """Tests for HLODCellState."""

    def test_state_values(self) -> None:
        """Test that all state values exist."""
        assert HLODCellState.UNINITIALIZED is not None
        assert HLODCellState.GENERATING is not None
        assert HLODCellState.READY is not None
        assert HLODCellState.INVALID is not None
        assert HLODCellState.ERROR is not None

    def test_state_transitions(self, source_meshes: list) -> None:
        """Test state transitions during generation."""
        configs = HLODLayerConfig.create_default_configs(2)
        bounds = AABB(Vec3(0, 0, 0), Vec3(10, 10, 10))
        cell = HLODCell(
            cell_id=(0, 0),
            bounds=bounds,
            source_meshes=source_meshes,
        )

        # Initial state
        assert cell.state == HLODCellState.UNINITIALIZED

        # After generation
        cell.generate_layers(configs)
        assert cell.state == HLODCellState.READY

        # After invalidation
        cell.invalidate()
        assert cell.state == HLODCellState.INVALID
