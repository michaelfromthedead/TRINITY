"""
World partition grid system.

Provides a 2D grid structure for organizing the world into cells
for efficient streaming and spatial queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from engine.core.math.vec import Vec2, Vec3
from engine.world.partition.cell import CellCoord, CellState, StreamingCell
from engine.world.partition.constants import (
    DEFAULT_GRID_WIDTH,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_CELL_SIZE,
    DEFAULT_GRID_LOAD_RADIUS,
    CELL_VERTICAL_MIN,
    CELL_VERTICAL_MAX,
    SQRT2_OVER_2,
    MIN_CELL_SIZE,
)


@dataclass
class WorldGrid:
    """
    2D grid partitioning of the world into streaming cells.

    The grid divides world space into uniform cells that can be
    independently loaded and unloaded based on distance from
    streaming sources.
    """
    width: int = DEFAULT_GRID_WIDTH  # Number of cells in X direction
    height: int = DEFAULT_GRID_HEIGHT  # Number of cells in Y/Z direction
    cell_size: float = DEFAULT_CELL_SIZE  # World units per cell

    # Grid origin (world position of cell 0,0)
    origin: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))

    # Cells dictionary indexed by coordinates
    cells: Dict[Tuple[int, int], StreamingCell] = field(default_factory=dict)

    # Streaming state
    load_center: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    load_radius: float = DEFAULT_GRID_LOAD_RADIUS  # In cells, not world units

    # Performance tracking
    _active_cells: Set[Tuple[int, int]] = field(default_factory=set, repr=False)
    _loaded_cells: Set[Tuple[int, int]] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        """Initialize grid if empty."""
        pass

    def get_cell(self, x: int, y: int) -> Optional[StreamingCell]:
        """
        Get a cell by grid coordinates.

        Args:
            x: Grid X coordinate.
            y: Grid Y coordinate.

        Returns:
            The cell at the coordinates, or None if not found.
        """
        return self.cells.get((x, y))

    def get_or_create_cell(self, x: int, y: int) -> StreamingCell:
        """
        Get a cell by coordinates, creating it if it doesn't exist.

        Args:
            x: Grid X coordinate.
            y: Grid Y coordinate.

        Returns:
            The cell at the coordinates.
        """
        key = (x, y)
        if key not in self.cells:
            self.cells[key] = self._create_cell(x, y)
        return self.cells[key]

    def _create_cell(self, x: int, y: int) -> StreamingCell:
        """Create a new cell at the given coordinates."""
        world_x = self.origin.x + x * self.cell_size
        world_z = self.origin.z + y * self.cell_size

        return StreamingCell(
            coord=CellCoord(x, y),
            bounds_min=Vec3(world_x, CELL_VERTICAL_MIN, world_z),
            bounds_max=Vec3(world_x + self.cell_size, CELL_VERTICAL_MAX, world_z + self.cell_size),
        )

    def get_cell_at_world_pos(self, position: Vec3) -> Optional[StreamingCell]:
        """
        Get the cell containing a world position.

        Args:
            position: World position.

        Returns:
            The cell containing the position, or None if outside grid.
        """
        coord = self.world_to_cell_coord(position)
        if coord is None:
            return None
        return self.get_cell(coord.x, coord.y)

    def get_or_create_cell_at_world_pos(self, position: Vec3) -> StreamingCell:
        """
        Get or create the cell containing a world position.

        Args:
            position: World position.

        Returns:
            The cell containing the position.
        """
        coord = self.world_to_cell_coord(position)
        if coord is None:
            # Position outside grid bounds, clamp to nearest edge
            coord = self._clamp_to_grid_bounds(position)
        return self.get_or_create_cell(coord.x, coord.y)

    def world_to_cell_coord(self, position: Vec3) -> Optional[CellCoord]:
        """
        Convert a world position to cell coordinates.

        Args:
            position: World position.

        Returns:
            Cell coordinates, or None if outside grid bounds.
        """
        # Protect against division by zero
        effective_cell_size = max(self.cell_size, MIN_CELL_SIZE)
        x = int((position.x - self.origin.x) // effective_cell_size)
        y = int((position.z - self.origin.z) // effective_cell_size)

        if not self.is_valid_coord(x, y):
            return None

        return CellCoord(x, y)

    def cell_coord_to_world_center(self, coord: CellCoord) -> Vec3:
        """
        Get the world-space center of a cell.

        Args:
            coord: Cell coordinates.

        Returns:
            World position of cell center.
        """
        world_x = self.origin.x + (coord.x + 0.5) * self.cell_size
        world_z = self.origin.z + (coord.y + 0.5) * self.cell_size
        return Vec3(world_x, 0, world_z)

    def cell_coord_to_world_min(self, coord: CellCoord) -> Vec3:
        """
        Get the minimum corner of a cell in world space.

        Args:
            coord: Cell coordinates.

        Returns:
            World position of cell minimum corner.
        """
        world_x = self.origin.x + coord.x * self.cell_size
        world_z = self.origin.z + coord.y * self.cell_size
        return Vec3(world_x, 0, world_z)

    def is_valid_coord(self, x: int, y: int) -> bool:
        """
        Check if coordinates are within grid bounds.

        Args:
            x: Grid X coordinate.
            y: Grid Y coordinate.

        Returns:
            True if coordinates are valid.
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def _clamp_to_grid_bounds(self, position: Vec3) -> CellCoord:
        """Clamp a world position to valid grid coordinates."""
        # Protect against division by zero
        effective_cell_size = max(self.cell_size, MIN_CELL_SIZE)
        x = int((position.x - self.origin.x) // effective_cell_size)
        y = int((position.z - self.origin.z) // effective_cell_size)

        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))

        return CellCoord(x, y)

    def get_cells_in_radius(
        self,
        center: Vec3,
        radius: float,
        include_partial: bool = True,
    ) -> List[StreamingCell]:
        """
        Get all cells within a radius of a world position.

        Args:
            center: Center position in world space.
            radius: Radius in world units.
            include_partial: If True, include cells that partially overlap.

        Returns:
            List of cells within the radius.
        """
        result = []

        # Convert radius to cell units (protect against division by zero)
        effective_cell_size = max(self.cell_size, MIN_CELL_SIZE)
        cell_radius = int(radius / effective_cell_size) + 1

        center_coord = self.world_to_cell_coord(center)
        if center_coord is None:
            center_coord = self._clamp_to_grid_bounds(center)

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                x = center_coord.x + dx
                y = center_coord.y + dy

                if not self.is_valid_coord(x, y):
                    continue

                cell = self.get_or_create_cell(x, y)

                if include_partial:
                    # Check if any part of the cell is within radius
                    cell_center = self.cell_coord_to_world_center(cell.coord)
                    # Use simple distance check (actual implementation would use proper AABB-sphere test)
                    dist = center.distance(cell_center)
                    half_diag = self.cell_size * SQRT2_OVER_2
                    if dist <= radius + half_diag:
                        result.append(cell)
                else:
                    # Check if cell center is within radius
                    cell_center = self.cell_coord_to_world_center(cell.coord)
                    if center.distance(cell_center) <= radius:
                        result.append(cell)

        return result

    def get_cells_in_rect(
        self,
        min_coord: CellCoord,
        max_coord: CellCoord,
    ) -> List[StreamingCell]:
        """
        Get all cells in a rectangular region.

        Args:
            min_coord: Minimum cell coordinates (inclusive).
            max_coord: Maximum cell coordinates (inclusive).

        Returns:
            List of cells in the rectangle.
        """
        result = []

        for x in range(min_coord.x, max_coord.x + 1):
            for y in range(min_coord.y, max_coord.y + 1):
                if self.is_valid_coord(x, y):
                    cell = self.get_or_create_cell(x, y)
                    result.append(cell)

        return result

    def update_streaming_center(self, position: Vec3) -> Tuple[List[StreamingCell], List[StreamingCell]]:
        """
        Update the streaming center and determine cells to load/unload.

        Args:
            position: New streaming center position.

        Returns:
            Tuple of (cells_to_load, cells_to_unload).
        """
        self.load_center = position

        # Get cells that should be loaded
        new_active = set()
        cells_in_range = self.get_cells_in_radius(position, self.load_radius * self.cell_size)

        for cell in cells_in_range:
            new_active.add((cell.coord.x, cell.coord.y))

        # Determine changes
        to_load_coords = new_active - self._active_cells
        to_unload_coords = self._active_cells - new_active

        # Get actual cells
        to_load = [self.get_or_create_cell(x, y) for x, y in to_load_coords]
        to_unload = [self.get_cell(x, y) for x, y in to_unload_coords if self.get_cell(x, y)]

        # Update active set
        self._active_cells = new_active

        return to_load, to_unload

    def get_neighboring_cells(self, coord: CellCoord, include_diagonal: bool = True) -> List[StreamingCell]:
        """
        Get all neighboring cells of a coordinate.

        Args:
            coord: Center cell coordinates.
            include_diagonal: If True, include diagonal neighbors.

        Returns:
            List of neighboring cells.
        """
        neighbors = []

        # Cardinal directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        if include_diagonal:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

        for dx, dy in directions:
            x = coord.x + dx
            y = coord.y + dy

            if self.is_valid_coord(x, y):
                cell = self.get_cell(x, y)
                if cell:
                    neighbors.append(cell)

        return neighbors

    def get_loaded_cells(self) -> List[StreamingCell]:
        """Get all currently loaded cells."""
        return [
            cell for cell in self.cells.values()
            if cell.state in (CellState.LOADED, CellState.ACTIVATED)
        ]

    def get_active_cells(self) -> List[StreamingCell]:
        """Get all currently active (activated) cells."""
        return [
            cell for cell in self.cells.values()
            if cell.state == CellState.ACTIVATED
        ]

    def get_all_cells(self) -> List[StreamingCell]:
        """Get all cells in the grid."""
        return list(self.cells.values())

    def clear(self) -> None:
        """Clear all cells from the grid."""
        self.cells.clear()
        self._active_cells.clear()
        self._loaded_cells.clear()

    def get_cell_count(self) -> int:
        """Get the number of cells in the grid."""
        return len(self.cells)

    def get_total_possible_cells(self) -> int:
        """Get the maximum number of cells the grid can contain."""
        return self.width * self.height

    def get_grid_bounds_world(self) -> Tuple[Vec3, Vec3]:
        """
        Get the world-space bounds of the entire grid.

        Returns:
            Tuple of (min_point, max_point).
        """
        min_point = self.origin
        max_point = Vec3(
            self.origin.x + self.width * self.cell_size,
            0,
            self.origin.z + self.height * self.cell_size,
        )
        return min_point, max_point

    def contains_world_position(self, position: Vec3) -> bool:
        """
        Check if a world position is within the grid bounds.

        Args:
            position: World position to check.

        Returns:
            True if the position is within the grid.
        """
        return self.world_to_cell_coord(position) is not None

    def set_cell_state(self, coord: CellCoord, state: CellState) -> bool:
        """
        Set the state of a cell.

        Args:
            coord: Cell coordinates.
            state: New state.

        Returns:
            True if the state was set, False if cell not found.
        """
        cell = self.get_cell(coord.x, coord.y)
        if cell:
            cell.state = state

            # Update tracking sets
            key = (coord.x, coord.y)
            if state in (CellState.LOADED, CellState.ACTIVATED):
                self._loaded_cells.add(key)
            else:
                self._loaded_cells.discard(key)

            return True
        return False

    def __iter__(self) -> Iterator[StreamingCell]:
        """Iterate over all cells."""
        return iter(self.cells.values())

    def __len__(self) -> int:
        """Get the number of cells."""
        return len(self.cells)

    def __contains__(self, coord: Tuple[int, int]) -> bool:
        """Check if a coordinate exists in the grid."""
        return coord in self.cells
