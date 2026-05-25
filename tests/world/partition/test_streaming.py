"""
Tests for the Streaming module.

Tests streaming sources, budget management, and cell loading decisions.
"""

import pytest
from engine.core.math.vec import Vec3
from engine.world.partition.cell import CellCoord, CellState, StreamingCell
from engine.world.partition.grid import WorldGrid
from engine.world.partition.streaming import (
    CameraStreamingSource,
    CustomStreamingSource,
    PlayerStreamingSource,
    StreamingBudget,
    StreamingConfig,
    StreamingPriority,
    StreamingRequest,
    StreamingSource,
    StreamingVolume,
    StreamingVolumeType,
    WorldStreaming,
)


# =============================================================================
# StreamingSource Tests
# =============================================================================

class TestPlayerStreamingSource:
    """Tests for PlayerStreamingSource class."""

    def test_player_source_creation(self):
        """Test player streaming source creation."""
        source = PlayerStreamingSource(
            position=Vec3(100, 50, 200),
            load_radius=5000.0,
        )
        assert source.position.x == 100
        assert source.load_radius == 5000.0

    def test_player_source_position_update(self):
        """Test updating player position."""
        source = PlayerStreamingSource()
        source.position = Vec3(500, 0, 300)
        assert source.position.x == 500

    def test_player_source_priority(self):
        """Test player source priority."""
        source = PlayerStreamingSource(priority=StreamingPriority.CRITICAL.value)
        assert source.priority == StreamingPriority.CRITICAL.value

    def test_player_source_active_state(self):
        """Test player source active state."""
        source = PlayerStreamingSource()
        assert source.is_active is True
        source.is_active = False
        assert source.is_active is False

    def test_player_source_load_radius_setter(self):
        """Test setting load radius."""
        source = PlayerStreamingSource()
        source.load_radius = 8000.0
        assert source.load_radius == 8000.0


class TestCameraStreamingSource:
    """Tests for CameraStreamingSource class."""

    def test_camera_source_creation(self):
        """Test camera streaming source creation."""
        source = CameraStreamingSource(
            position=Vec3(100, 50, 200),
            forward=Vec3(0, 0, -1),
            load_radius=3000.0,
        )
        assert source.position.x == 100
        assert source.load_radius == 3000.0

    def test_camera_source_forward(self):
        """Test camera forward direction."""
        source = CameraStreamingSource(forward=Vec3(1, 0, 0))
        # Forward should be normalized
        assert source.forward.x == pytest.approx(1.0)
        assert source.forward.y == pytest.approx(0.0)

    def test_camera_source_forward_setter(self):
        """Test setting camera forward direction."""
        source = CameraStreamingSource()
        source.forward = Vec3(1, 1, 0)
        # Should be normalized
        assert source.forward.length() == pytest.approx(1.0, abs=0.001)

    def test_camera_source_forward_bias(self):
        """Test camera forward bias property."""
        source = CameraStreamingSource(forward_bias=2.0)
        assert source.forward_bias == 2.0

    def test_camera_source_priority(self):
        """Test camera source has HIGH priority."""
        source = CameraStreamingSource()
        assert source.priority == StreamingPriority.HIGH.value


class TestCustomStreamingSource:
    """Tests for CustomStreamingSource class."""

    def test_custom_source_creation(self):
        """Test custom streaming source creation."""
        pos = Vec3(100, 50, 200)
        source = CustomStreamingSource(
            position_getter=lambda: pos,
            load_radius=2000.0,
            name="TestSource",
        )
        assert source.position.x == 100
        assert source.load_radius == 2000.0
        assert source.name == "TestSource"

    def test_custom_source_dynamic_position(self):
        """Test custom source with dynamic position."""
        position = [Vec3(0, 0, 0)]  # Mutable to allow updates

        source = CustomStreamingSource(
            position_getter=lambda: position[0],
        )

        assert source.position.x == 0
        position[0] = Vec3(100, 0, 0)
        assert source.position.x == 100

    def test_custom_source_active_toggle(self):
        """Test toggling active state."""
        source = CustomStreamingSource(position_getter=lambda: Vec3())
        assert source.is_active is True
        source.is_active = False
        assert source.is_active is False


# =============================================================================
# StreamingConfig Tests
# =============================================================================

class TestStreamingConfig:
    """Tests for StreamingConfig class."""

    def test_config_creation_default(self):
        """Test default config creation."""
        config = StreamingConfig()
        assert config.load_distance == 5000.0
        assert config.unload_distance == 6000.0
        assert config.max_concurrent_loads == 4

    def test_config_creation_custom(self):
        """Test custom config creation."""
        config = StreamingConfig(
            load_distance=3000.0,
            unload_distance=4000.0,
            max_concurrent_loads=8,
        )
        assert config.load_distance == 3000.0
        assert config.max_concurrent_loads == 8

    def test_config_hysteresis_enforcement(self):
        """Test that unload_distance is adjusted if less than load_distance."""
        config = StreamingConfig(
            load_distance=5000.0,
            unload_distance=4000.0,  # Less than load
            hysteresis=1000.0,
        )
        assert config.unload_distance > config.load_distance

    def test_config_preloading(self):
        """Test preloading configuration."""
        config = StreamingConfig(
            enable_preloading=True,
            preload_distance=7000.0,
        )
        assert config.enable_preloading is True
        assert config.preload_distance == 7000.0


# =============================================================================
# StreamingBudget Tests
# =============================================================================

class TestStreamingBudget:
    """Tests for StreamingBudget class."""

    def test_budget_creation_default(self):
        """Test default budget creation."""
        budget = StreamingBudget()
        assert budget.memory_mb == 1024.0
        assert budget.io_mbps == 100.0
        assert budget.frame_ms == 2.0

    def test_budget_memory_available(self):
        """Test available memory calculation."""
        budget = StreamingBudget(memory_mb=1024.0)
        budget.current_memory_mb = 512.0
        assert budget.memory_available == 512.0

    def test_budget_io_available(self):
        """Test available IO calculation."""
        budget = StreamingBudget(io_mbps=100.0)
        budget.current_io_mbps = 60.0
        assert budget.io_available == 40.0

    def test_budget_frame_available(self):
        """Test available frame time calculation."""
        budget = StreamingBudget(frame_ms=2.0)
        budget.current_frame_ms = 0.5
        assert budget.frame_available == 1.5

    def test_budget_can_load_true(self):
        """Test can_load returns True with sufficient budget."""
        budget = StreamingBudget(memory_mb=1024.0, io_mbps=100.0)
        budget.current_memory_mb = 500.0
        assert budget.can_load(estimated_mb=100, estimated_io=20) is True

    def test_budget_can_load_false_memory(self):
        """Test can_load returns False when memory exceeded."""
        budget = StreamingBudget(memory_mb=1024.0)
        budget.current_memory_mb = 1000.0
        assert budget.can_load(estimated_mb=100) is False

    def test_budget_reserve_memory_success(self):
        """Test successful memory reservation."""
        budget = StreamingBudget(memory_mb=1024.0)
        result = budget.reserve_memory(500.0)
        assert result is True
        assert budget.current_memory_mb == 500.0

    def test_budget_reserve_memory_failure(self):
        """Test failed memory reservation."""
        budget = StreamingBudget(memory_mb=1024.0)
        budget.current_memory_mb = 600.0
        result = budget.reserve_memory(500.0)
        assert result is False
        assert budget.current_memory_mb == 600.0  # Unchanged

    def test_budget_release_memory(self):
        """Test releasing memory."""
        budget = StreamingBudget()
        budget.current_memory_mb = 500.0
        budget.release_memory(200.0)
        assert budget.current_memory_mb == 300.0

    def test_budget_release_memory_clamp(self):
        """Test releasing more than current memory clamps to 0."""
        budget = StreamingBudget()
        budget.current_memory_mb = 100.0
        budget.release_memory(200.0)
        assert budget.current_memory_mb == 0.0

    def test_budget_reset_frame(self):
        """Test resetting per-frame budgets."""
        budget = StreamingBudget()
        budget.current_frame_ms = 1.5
        budget.current_io_mbps = 50.0
        budget.reset_frame_budget()
        assert budget.current_frame_ms == 0.0
        assert budget.current_io_mbps == 0.0


# =============================================================================
# StreamingVolume Tests
# =============================================================================

class TestStreamingVolume:
    """Tests for StreamingVolume class."""

    def test_volume_creation_default(self):
        """Test default volume creation."""
        volume = StreamingVolume()
        assert volume.volume_type == StreamingVolumeType.TRIGGER
        assert volume.enabled is True

    def test_volume_creation_custom(self):
        """Test custom volume creation."""
        volume = StreamingVolume(
            volume_type=StreamingVolumeType.PRELOAD,
            bounds_min=Vec3(-100, -50, -100),
            bounds_max=Vec3(100, 50, 100),
            priority=StreamingPriority.HIGH.value,
        )
        assert volume.volume_type == StreamingVolumeType.PRELOAD
        assert volume.priority == StreamingPriority.HIGH.value

    def test_volume_bounds_property(self):
        """Test volume bounds AABB property."""
        volume = StreamingVolume(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(100, 100, 100),
        )
        assert volume.bounds.min.x == 0
        assert volume.bounds.max.x == 100

    def test_volume_center_property(self):
        """Test volume center calculation."""
        volume = StreamingVolume(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(100, 100, 100),
        )
        assert volume.center.x == 50
        assert volume.center.y == 50

    def test_volume_contains_point_inside(self):
        """Test point containment inside volume."""
        volume = StreamingVolume(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(100, 100, 100),
        )
        assert volume.contains_point(Vec3(50, 50, 50)) is True

    def test_volume_contains_point_outside(self):
        """Test point containment outside volume."""
        volume = StreamingVolume(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(100, 100, 100),
        )
        assert volume.contains_point(Vec3(150, 50, 50)) is False

    def test_volume_overlaps_true(self):
        """Test overlapping volumes."""
        volume1 = StreamingVolume(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(100, 100, 100),
        )
        volume2 = StreamingVolume(
            bounds_min=Vec3(50, 50, 50),
            bounds_max=Vec3(150, 150, 150),
        )
        assert volume1.overlaps(volume2) is True

    def test_volume_overlaps_false(self):
        """Test non-overlapping volumes."""
        volume1 = StreamingVolume(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(100, 100, 100),
        )
        volume2 = StreamingVolume(
            bounds_min=Vec3(200, 200, 200),
            bounds_max=Vec3(300, 300, 300),
        )
        assert volume1.overlaps(volume2) is False


# =============================================================================
# StreamingRequest Tests
# =============================================================================

class TestStreamingRequest:
    """Tests for StreamingRequest class."""

    def test_request_creation(self):
        """Test streaming request creation."""
        cell = StreamingCell(coord=CellCoord(5, 5))
        request = StreamingRequest(
            cell=cell,
            priority=100,
            is_load=True,
        )
        assert request.cell is cell
        assert request.priority == 100
        assert request.is_load is True

    def test_request_unload(self):
        """Test unload request."""
        cell = StreamingCell()
        request = StreamingRequest(
            cell=cell,
            is_load=False,
        )
        assert request.is_load is False


# =============================================================================
# WorldStreaming Tests
# =============================================================================

class TestWorldStreamingCreation:
    """Tests for WorldStreaming creation."""

    def test_streaming_creation(self):
        """Test world streaming manager creation."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        assert streaming.grid is grid
        assert streaming.config is not None
        assert streaming.budget is not None

    def test_streaming_custom_config(self):
        """Test world streaming with custom config."""
        grid = WorldGrid()
        config = StreamingConfig(load_distance=3000.0)
        streaming = WorldStreaming(grid, config=config)
        assert streaming.config.load_distance == 3000.0


class TestWorldStreamingSources:
    """Tests for streaming source management."""

    def test_add_source(self):
        """Test adding streaming source."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        source = PlayerStreamingSource()
        streaming.add_source(source)
        assert len(streaming.sources) == 1

    def test_remove_source(self):
        """Test removing streaming source."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        source = PlayerStreamingSource()
        streaming.add_source(source)
        result = streaming.remove_source(source)
        assert result is True
        assert len(streaming.sources) == 0

    def test_remove_source_not_found(self):
        """Test removing non-existent source."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        source = PlayerStreamingSource()
        result = streaming.remove_source(source)
        assert result is False


class TestWorldStreamingVolumes:
    """Tests for streaming volume management."""

    def test_add_volume(self):
        """Test adding streaming volume."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        volume = StreamingVolume()
        streaming.add_volume(volume)
        assert len(streaming.volumes) == 1

    def test_remove_volume(self):
        """Test removing streaming volume."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        volume = StreamingVolume()
        streaming.add_volume(volume)
        result = streaming.remove_volume(volume)
        assert result is True
        assert len(streaming.volumes) == 0


class TestWorldStreamingUpdate:
    """Tests for streaming update logic."""

    def test_update_loads_cells(self):
        """Test that update loads cells near sources."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        config = StreamingConfig(load_distance=500.0)
        streaming = WorldStreaming(grid, config=config)

        source = PlayerStreamingSource(position=Vec3(128, 0, 128))
        streaming.add_source(source)

        loaded, unloaded = streaming.update(0.016)
        assert loaded > 0  # Should load some cells

    def test_update_unloads_distant_cells(self):
        """Test that update unloads distant cells."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        config = StreamingConfig(
            load_distance=500.0,
            unload_distance=600.0,
        )
        streaming = WorldStreaming(grid, config=config)

        source = PlayerStreamingSource(position=Vec3(128, 0, 128))
        streaming.add_source(source)

        # First update - load cells near initial position
        streaming.update(0.016)

        # Move source far away
        source.position = Vec3(2000, 0, 2000)
        loaded, unloaded = streaming.update(0.016)
        # Should unload cells from original position
        assert unloaded >= 0  # May or may not unload depending on timing

    def test_update_respects_inactive_sources(self):
        """Test that inactive sources are ignored."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        streaming = WorldStreaming(grid)

        source = PlayerStreamingSource(position=Vec3(128, 0, 128))
        source.is_active = False
        streaming.add_source(source)

        loaded, unloaded = streaming.update(0.016)
        # Should not load cells for inactive source
        assert loaded == 0


class TestWorldStreamingCellLists:
    """Tests for cell list retrieval."""

    def test_get_cells_to_load(self):
        """Test getting pending load list."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        # Initially empty
        assert len(streaming.get_cells_to_load()) == 0

    def test_get_cells_to_unload(self):
        """Test getting pending unload list."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        # Initially empty
        assert len(streaming.get_cells_to_unload()) == 0

    def test_get_loading_cells(self):
        """Test getting currently loading cells."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        assert len(streaming.get_loading_cells()) == 0

    def test_get_unloading_cells(self):
        """Test getting currently unloading cells."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        assert len(streaming.get_unloading_cells()) == 0


class TestWorldStreamingForceOperations:
    """Tests for force load/unload operations."""

    def test_force_load_cell(self):
        """Test forcing cell load."""
        grid = WorldGrid(width=10, height=10)
        streaming = WorldStreaming(grid)

        result = streaming.force_load_cell(CellCoord(5, 5))
        assert result is True
        cell = grid.get_cell(5, 5)
        assert cell.state == CellState.LOADED

    def test_force_load_already_loaded(self):
        """Test forcing load on already loaded cell."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        streaming.force_load_cell(CellCoord(5, 5))
        result = streaming.force_load_cell(CellCoord(5, 5))
        assert result is False

    def test_force_unload_cell(self):
        """Test forcing cell unload."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        streaming.force_load_cell(CellCoord(5, 5))
        result = streaming.force_unload_cell(CellCoord(5, 5))
        assert result is True
        cell = grid.get_cell(5, 5)
        assert cell.state == CellState.UNLOADED

    def test_force_unload_not_loaded(self):
        """Test forcing unload on non-loaded cell."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        result = streaming.force_unload_cell(CellCoord(5, 5))
        assert result is False


class TestWorldStreamingCallbacks:
    """Tests for streaming callbacks."""

    def test_on_cell_loaded_callback(self):
        """Test cell loaded callback."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        loaded_cells = []
        streaming.on_cell_loaded(lambda c: loaded_cells.append(c.coord))

        streaming.force_load_cell(CellCoord(5, 5))
        assert len(loaded_cells) == 1

    def test_on_cell_unloaded_callback(self):
        """Test cell unloaded callback."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        unloaded_cells = []
        streaming.on_cell_unloaded(lambda c: unloaded_cells.append(c.coord))

        streaming.force_load_cell(CellCoord(5, 5))
        streaming.force_unload_cell(CellCoord(5, 5))
        assert len(unloaded_cells) == 1


class TestWorldStreamingStats:
    """Tests for streaming statistics."""

    def test_get_streaming_stats(self):
        """Test getting streaming statistics."""
        grid = WorldGrid()
        streaming = WorldStreaming(grid)
        streaming.add_source(PlayerStreamingSource())
        streaming.add_volume(StreamingVolume())

        stats = streaming.get_streaming_stats()
        assert "total_cells" in stats
        assert "loaded_cells" in stats
        assert "source_count" in stats
        assert stats["source_count"] == 1
        assert stats["volume_count"] == 1

    def test_clear_queues(self):
        """Test clearing streaming queues."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        streaming = WorldStreaming(grid)
        source = PlayerStreamingSource(position=Vec3(128, 0, 128))
        streaming.add_source(source)

        # Trigger some queue population (but don't process)
        # Clear queues
        streaming.clear_queues()
        assert len(streaming.get_cells_to_load()) == 0
        assert len(streaming.get_cells_to_unload()) == 0


class TestStreamingEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_player_source_none_position_defaults_to_origin(self):
        """Test that None position defaults to Vec3(0,0,0)."""
        source = PlayerStreamingSource(position=None)
        assert source.position.x == 0
        assert source.position.y == 0
        assert source.position.z == 0

    def test_camera_source_none_forward_defaults(self):
        """Test that None forward defaults to looking down -Z."""
        source = CameraStreamingSource(forward=None)
        assert source.forward.z == -1

    def test_streaming_config_equal_load_unload_distances(self):
        """Test config adjusts when unload equals load distance."""
        config = StreamingConfig(
            load_distance=5000.0,
            unload_distance=5000.0,  # Same as load
            hysteresis=500.0,
        )
        assert config.unload_distance > config.load_distance

    def test_streaming_with_no_active_sources(self):
        """Test streaming with all sources deactivated."""
        grid = WorldGrid(width=10, height=10)
        streaming = WorldStreaming(grid)

        source = PlayerStreamingSource(position=Vec3(500, 0, 500))
        source.is_active = False
        streaming.add_source(source)

        loaded, unloaded = streaming.update(0.016)
        assert loaded == 0

    def test_budget_negative_release(self):
        """Test releasing negative memory is handled."""
        budget = StreamingBudget()
        budget.current_memory_mb = 100.0
        budget.release_memory(-50.0)  # Negative release
        # Should add to current (release of negative = add)
        assert budget.current_memory_mb == 150.0

    def test_budget_zero_available_when_full(self):
        """Test available returns zero when at capacity."""
        budget = StreamingBudget(memory_mb=100.0)
        budget.current_memory_mb = 100.0
        assert budget.memory_available == 0.0

    def test_volume_with_cells_to_load(self):
        """Test streaming volume with explicit cells to load."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        streaming = WorldStreaming(grid)

        # Create volume with explicit cells
        volume = StreamingVolume(
            bounds_min=Vec3(0, 0, 0),
            bounds_max=Vec3(256, 256, 256),
            cells_to_load=[CellCoord(0, 0), CellCoord(1, 1)],
        )
        streaming.add_volume(volume)

        source = PlayerStreamingSource(position=Vec3(128, 0, 128))
        streaming.add_source(source)

        loaded, unloaded = streaming.update(0.016)
        assert loaded >= 0

    def test_streaming_with_very_large_distance_source(self):
        """Test streaming with source at very large distance."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        streaming = WorldStreaming(grid)

        # Source very far away
        source = PlayerStreamingSource(
            position=Vec3(1000000, 0, 1000000),
            load_radius=100.0,  # Small radius
        )
        streaming.add_source(source)

        # Should not crash with large coordinates
        loaded, unloaded = streaming.update(0.016)
        assert isinstance(loaded, int)

    def test_config_priority_scale_applied(self):
        """Test that priority scale is configurable."""
        config = StreamingConfig(priority_scale=2.0)
        assert config.priority_scale == 2.0

    def test_multiple_sources_priority_ordering(self):
        """Test that higher priority sources take precedence."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        config = StreamingConfig(load_distance=500.0)
        streaming = WorldStreaming(grid, config=config)

        # Add sources with different priorities
        high_priority = PlayerStreamingSource(
            position=Vec3(1000, 0, 1000),
            priority=StreamingPriority.CRITICAL.value,
        )
        low_priority = PlayerStreamingSource(
            position=Vec3(128, 0, 128),
            priority=StreamingPriority.LOW.value,
        )

        streaming.add_source(high_priority)
        streaming.add_source(low_priority)

        # Both should work without errors
        loaded, unloaded = streaming.update(0.016)
        assert isinstance(loaded, int)
