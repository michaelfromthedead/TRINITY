"""
Tests for the WorldGrid module.

Tests WorldGrid coordinate mapping, cell retrieval, and streaming radius.
"""

import pytest
from engine.core.math.vec import Vec3
from engine.world.partition.grid import WorldGrid
from engine.world.partition.cell import CellCoord, CellState, StreamingCell


class TestWorldGridCreation:
    """Tests for WorldGrid creation and initialization."""

    def test_grid_creation_default(self):
        """Test default grid creation."""
        grid = WorldGrid()
        assert grid.width == 100
        assert grid.height == 100
        assert grid.cell_size == 256.0

    def test_grid_creation_custom(self):
        """Test custom grid creation."""
        grid = WorldGrid(
            width=50,
            height=75,
            cell_size=512.0,
        )
        assert grid.width == 50
        assert grid.height == 75
        assert grid.cell_size == 512.0

    def test_grid_origin(self):
        """Test grid origin setting."""
        grid = WorldGrid(origin=Vec3(1000, 0, 2000))
        assert grid.origin.x == 1000
        assert grid.origin.z == 2000


class TestWorldGridCellAccess:
    """Tests for cell access methods."""

    def test_get_cell_nonexistent(self):
        """Test getting non-existent cell returns None."""
        grid = WorldGrid()
        cell = grid.get_cell(5, 5)
        assert cell is None

    def test_get_or_create_cell(self):
        """Test getting or creating a cell."""
        grid = WorldGrid()
        cell = grid.get_or_create_cell(5, 5)
        assert cell is not None
        assert cell.coord.x == 5
        assert cell.coord.y == 5

    def test_get_or_create_cell_cached(self):
        """Test that get_or_create returns same cell on subsequent calls."""
        grid = WorldGrid()
        cell1 = grid.get_or_create_cell(5, 5)
        cell2 = grid.get_or_create_cell(5, 5)
        assert cell1 is cell2

    def test_cell_bounds_calculation(self):
        """Test cell bounds are calculated correctly."""
        grid = WorldGrid(cell_size=256.0, origin=Vec3(0, 0, 0))
        cell = grid.get_or_create_cell(2, 3)
        assert cell.bounds_min.x == 512  # 2 * 256
        assert cell.bounds_min.z == 768  # 3 * 256
        assert cell.bounds_max.x == 768  # (2+1) * 256
        assert cell.bounds_max.z == 1024  # (3+1) * 256


class TestWorldGridCoordinateConversion:
    """Tests for coordinate conversion methods."""

    def test_world_to_cell_coord(self):
        """Test world position to cell coordinate conversion."""
        grid = WorldGrid(cell_size=256.0, origin=Vec3(0, 0, 0))
        coord = grid.world_to_cell_coord(Vec3(300, 100, 600))
        assert coord is not None
        assert coord.x == 1  # 300 // 256
        assert coord.y == 2  # 600 // 256

    def test_world_to_cell_coord_with_origin(self):
        """Test conversion with non-zero origin."""
        grid = WorldGrid(cell_size=256.0, origin=Vec3(1000, 0, 2000))
        coord = grid.world_to_cell_coord(Vec3(1300, 0, 2600))
        assert coord is not None
        assert coord.x == 1  # (1300 - 1000) // 256
        assert coord.y == 2  # (2600 - 2000) // 256

    def test_world_to_cell_coord_outside_bounds(self):
        """Test conversion for position outside grid bounds."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        coord = grid.world_to_cell_coord(Vec3(5000, 0, 5000))
        assert coord is None  # Outside 10x10 grid

    def test_world_to_cell_coord_negative(self):
        """Test conversion for negative coordinates."""
        grid = WorldGrid(cell_size=256.0)
        coord = grid.world_to_cell_coord(Vec3(-100, 0, -200))
        assert coord is None  # Negative is outside default grid

    def test_cell_coord_to_world_center(self):
        """Test cell coordinate to world center conversion."""
        grid = WorldGrid(cell_size=256.0, origin=Vec3(0, 0, 0))
        center = grid.cell_coord_to_world_center(CellCoord(2, 3))
        assert center.x == 640  # (2 + 0.5) * 256
        assert center.z == 896  # (3 + 0.5) * 256

    def test_cell_coord_to_world_min(self):
        """Test cell coordinate to world minimum corner conversion."""
        grid = WorldGrid(cell_size=256.0, origin=Vec3(100, 0, 200))
        min_pos = grid.cell_coord_to_world_min(CellCoord(1, 2))
        assert min_pos.x == 356  # 100 + 1 * 256
        assert min_pos.z == 712  # 200 + 2 * 256


class TestWorldGridValidation:
    """Tests for coordinate validation."""

    def test_is_valid_coord_inside(self):
        """Test valid coordinate inside grid."""
        grid = WorldGrid(width=10, height=10)
        assert grid.is_valid_coord(5, 5) is True

    def test_is_valid_coord_edge(self):
        """Test valid coordinate on edge."""
        grid = WorldGrid(width=10, height=10)
        assert grid.is_valid_coord(0, 0) is True
        assert grid.is_valid_coord(9, 9) is True

    def test_is_valid_coord_outside(self):
        """Test invalid coordinate outside grid."""
        grid = WorldGrid(width=10, height=10)
        assert grid.is_valid_coord(10, 5) is False
        assert grid.is_valid_coord(5, 10) is False
        assert grid.is_valid_coord(-1, 5) is False

    def test_contains_world_position_inside(self):
        """Test world position inside grid."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        assert grid.contains_world_position(Vec3(500, 0, 500)) is True

    def test_contains_world_position_outside(self):
        """Test world position outside grid."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        assert grid.contains_world_position(Vec3(5000, 0, 5000)) is False


class TestWorldGridCellRetrieval:
    """Tests for cell retrieval methods."""

    def test_get_cell_at_world_pos(self):
        """Test getting cell at world position."""
        grid = WorldGrid(cell_size=256.0)
        grid.get_or_create_cell(1, 2)
        cell = grid.get_cell_at_world_pos(Vec3(300, 0, 600))
        assert cell is not None
        assert cell.coord.x == 1
        assert cell.coord.y == 2

    def test_get_cell_at_world_pos_nonexistent(self):
        """Test getting cell at position where none exists."""
        grid = WorldGrid()
        cell = grid.get_cell_at_world_pos(Vec3(300, 0, 600))
        assert cell is None

    def test_get_or_create_cell_at_world_pos(self):
        """Test getting or creating cell at world position."""
        grid = WorldGrid(cell_size=256.0)
        cell = grid.get_or_create_cell_at_world_pos(Vec3(300, 0, 600))
        assert cell is not None
        assert cell.coord.x == 1
        assert cell.coord.y == 2

    def test_get_cells_in_radius(self):
        """Test getting cells within a radius."""
        grid = WorldGrid(cell_size=256.0)
        # Create cells
        for x in range(5):
            for y in range(5):
                grid.get_or_create_cell(x, y)

        cells = grid.get_cells_in_radius(Vec3(640, 0, 640), 500)  # ~2 cells radius
        assert len(cells) > 0
        assert len(cells) <= 25  # At most all cells

    def test_get_cells_in_radius_empty_grid(self):
        """Test getting cells in radius from empty grid."""
        grid = WorldGrid()
        cells = grid.get_cells_in_radius(Vec3(500, 0, 500), 1000)
        # Should create cells within radius
        assert len(cells) > 0

    def test_get_cells_in_radius_center_only(self):
        """Test getting cells in small radius."""
        grid = WorldGrid(cell_size=256.0)
        cells = grid.get_cells_in_radius(Vec3(128, 0, 128), 50, include_partial=False)
        # Small radius should only include center cell
        assert len(cells) >= 1

    def test_get_cells_in_rect(self):
        """Test getting cells in rectangular region."""
        grid = WorldGrid()
        for x in range(5):
            for y in range(5):
                grid.get_or_create_cell(x, y)

        cells = grid.get_cells_in_rect(CellCoord(1, 1), CellCoord(3, 3))
        assert len(cells) == 9  # 3x3 region


class TestWorldGridStreaming:
    """Tests for streaming center updates."""

    def test_update_streaming_center_initial(self):
        """Test initial streaming center update."""
        grid = WorldGrid(cell_size=256.0, load_radius=2.0)
        to_load, to_unload = grid.update_streaming_center(Vec3(640, 0, 640))
        assert len(to_load) > 0
        assert len(to_unload) == 0

    def test_update_streaming_center_movement(self):
        """Test streaming center movement loads new cells."""
        grid = WorldGrid(cell_size=256.0, load_radius=2.0)
        # First update
        grid.update_streaming_center(Vec3(256, 0, 256))
        # Move to new position
        to_load, to_unload = grid.update_streaming_center(Vec3(1024, 0, 1024))
        # Should have cells to load and potentially unload
        assert isinstance(to_load, list)
        assert isinstance(to_unload, list)

    def test_streaming_center_no_change(self):
        """Test streaming center with no movement."""
        grid = WorldGrid(cell_size=256.0, load_radius=2.0)
        grid.update_streaming_center(Vec3(500, 0, 500))
        to_load, to_unload = grid.update_streaming_center(Vec3(500, 0, 500))
        assert len(to_load) == 0  # No new cells needed


class TestWorldGridNeighbors:
    """Tests for neighbor cell retrieval."""

    def test_get_neighboring_cells(self):
        """Test getting all neighboring cells."""
        grid = WorldGrid()
        for x in range(3):
            for y in range(3):
                grid.get_or_create_cell(x, y)

        center = grid.get_or_create_cell(1, 1)
        neighbors = grid.get_neighboring_cells(center.coord, include_diagonal=True)
        assert len(neighbors) == 8  # All 8 neighbors

    def test_get_neighboring_cells_cardinal_only(self):
        """Test getting only cardinal neighbors."""
        grid = WorldGrid()
        for x in range(3):
            for y in range(3):
                grid.get_or_create_cell(x, y)

        center = grid.get_or_create_cell(1, 1)
        neighbors = grid.get_neighboring_cells(center.coord, include_diagonal=False)
        assert len(neighbors) == 4  # Only cardinal directions

    def test_get_neighboring_cells_edge(self):
        """Test getting neighbors for edge cell."""
        grid = WorldGrid(width=5, height=5)
        for x in range(2):
            for y in range(2):
                grid.get_or_create_cell(x, y)

        corner = grid.get_or_create_cell(0, 0)
        neighbors = grid.get_neighboring_cells(corner.coord, include_diagonal=True)
        assert len(neighbors) == 3  # Only 3 neighbors at corner


class TestWorldGridCellManagement:
    """Tests for cell state management."""

    def test_get_loaded_cells(self):
        """Test getting loaded cells."""
        grid = WorldGrid()
        cell1 = grid.get_or_create_cell(0, 0)
        cell2 = grid.get_or_create_cell(1, 1)
        cell1.state = CellState.LOADED

        loaded = grid.get_loaded_cells()
        assert len(loaded) == 1
        assert loaded[0].coord == cell1.coord

    def test_get_active_cells(self):
        """Test getting active cells."""
        grid = WorldGrid()
        cell1 = grid.get_or_create_cell(0, 0)
        cell2 = grid.get_or_create_cell(1, 1)
        cell1.state = CellState.ACTIVATED

        active = grid.get_active_cells()
        assert len(active) == 1

    def test_get_all_cells(self):
        """Test getting all cells."""
        grid = WorldGrid()
        grid.get_or_create_cell(0, 0)
        grid.get_or_create_cell(1, 1)
        grid.get_or_create_cell(2, 2)

        all_cells = grid.get_all_cells()
        assert len(all_cells) == 3

    def test_clear_cells(self):
        """Test clearing all cells."""
        grid = WorldGrid()
        grid.get_or_create_cell(0, 0)
        grid.get_or_create_cell(1, 1)
        grid.clear()
        assert grid.get_cell_count() == 0

    def test_set_cell_state(self):
        """Test setting cell state."""
        grid = WorldGrid()
        grid.get_or_create_cell(5, 5)
        result = grid.set_cell_state(CellCoord(5, 5), CellState.LOADED)
        assert result is True
        assert grid.get_cell(5, 5).state == CellState.LOADED

    def test_set_cell_state_nonexistent(self):
        """Test setting state for non-existent cell."""
        grid = WorldGrid()
        result = grid.set_cell_state(CellCoord(5, 5), CellState.LOADED)
        assert result is False


class TestWorldGridUtilities:
    """Tests for grid utility methods."""

    def test_get_cell_count(self):
        """Test getting cell count."""
        grid = WorldGrid()
        assert grid.get_cell_count() == 0
        grid.get_or_create_cell(0, 0)
        grid.get_or_create_cell(1, 1)
        assert grid.get_cell_count() == 2

    def test_get_total_possible_cells(self):
        """Test getting total possible cells."""
        grid = WorldGrid(width=10, height=20)
        assert grid.get_total_possible_cells() == 200

    def test_get_grid_bounds_world(self):
        """Test getting world bounds of grid."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0, origin=Vec3(100, 0, 200))
        min_pt, max_pt = grid.get_grid_bounds_world()
        assert min_pt.x == 100
        assert min_pt.z == 200
        assert max_pt.x == 100 + 10 * 256
        assert max_pt.z == 200 + 10 * 256

    def test_grid_iteration(self):
        """Test iterating over grid cells."""
        grid = WorldGrid()
        grid.get_or_create_cell(0, 0)
        grid.get_or_create_cell(1, 1)

        cells = list(grid)
        assert len(cells) == 2

    def test_grid_len(self):
        """Test grid length."""
        grid = WorldGrid()
        grid.get_or_create_cell(0, 0)
        grid.get_or_create_cell(1, 1)
        assert len(grid) == 2

    def test_grid_contains(self):
        """Test checking if coordinate exists in grid."""
        grid = WorldGrid()
        grid.get_or_create_cell(5, 5)
        assert (5, 5) in grid
        assert (6, 6) not in grid


class TestWorldGridEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_get_cells_in_radius_zero_cell_size_protection(self):
        """Test that zero cell size is protected against division by zero."""
        grid = WorldGrid(cell_size=0.0)
        # Should not raise division by zero - protected by MIN_CELL_SIZE
        cells = grid.get_cells_in_radius(Vec3(128, 0, 128), 500)
        assert isinstance(cells, list)

    def test_get_cells_in_radius_very_small_cell_size(self):
        """Test with very small cell size."""
        grid = WorldGrid(cell_size=0.001)
        cells = grid.get_cells_in_radius(Vec3(0, 0, 0), 0.01)
        assert isinstance(cells, list)

    def test_grid_boundary_coordinate_handling(self):
        """Test coordinate handling at grid boundaries."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)

        # Test at exact boundaries
        assert grid.is_valid_coord(0, 0) is True
        assert grid.is_valid_coord(9, 9) is True
        assert grid.is_valid_coord(10, 0) is False
        assert grid.is_valid_coord(0, 10) is False

    def test_world_to_cell_coord_at_origin(self):
        """Test coordinate conversion at origin."""
        grid = WorldGrid(cell_size=256.0, origin=Vec3(0, 0, 0))
        coord = grid.world_to_cell_coord(Vec3(0, 0, 0))
        assert coord is not None
        assert coord.x == 0
        assert coord.y == 0

    def test_clamp_to_grid_bounds_far_outside(self):
        """Test clamping position very far outside grid."""
        grid = WorldGrid(width=10, height=10, cell_size=256.0)
        cell = grid.get_or_create_cell_at_world_pos(Vec3(1000000, 0, -1000000))
        assert cell is not None
        # Should be clamped to valid range
        assert 0 <= cell.coord.x < 10
        assert 0 <= cell.coord.y < 10

    def test_streaming_center_update_empty_grid(self):
        """Test streaming center update on empty grid."""
        grid = WorldGrid(width=5, height=5, cell_size=256.0, load_radius=1.0)
        to_load, to_unload = grid.update_streaming_center(Vec3(128, 0, 128))
        assert isinstance(to_load, list)
        assert isinstance(to_unload, list)

    def test_neighboring_cells_at_corner(self):
        """Test getting neighbors at grid corner."""
        grid = WorldGrid(width=10, height=10)
        corner_cell = grid.get_or_create_cell(0, 0)
        grid.get_or_create_cell(1, 0)
        grid.get_or_create_cell(0, 1)
        grid.get_or_create_cell(1, 1)

        neighbors = grid.get_neighboring_cells(corner_cell.coord, include_diagonal=True)
        # Corner should only have 3 neighbors (no negative coordinates)
        assert len(neighbors) == 3
