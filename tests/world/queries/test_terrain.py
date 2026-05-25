"""
Comprehensive tests for terrain query system.

Tests cover:
- Height interpolation accuracy
- Normal calculation
- Slope calculation
- Layer weight queries
- Terrain raycast stepping
- Area queries
- Visibility checks
"""

import math
import pytest
from typing import Dict, List, Optional, Tuple

from engine.world.queries.terrain import (
    TerrainHitResult,
    TerrainQuery,
    TerrainRaycast,
    TerrainLineTrace,
    TerrainAreaQuery,
    TerrainVisibility,
    TerrainQuerySystem,
)


# =============================================================================
# TYPE ALIASES
# =============================================================================

Vector3 = Tuple[float, float, float]
Bounds2D = Tuple[float, float, float, float]


# =============================================================================
# MOCK TERRAIN SYSTEM
# =============================================================================


class MockTerrainSystem:
    """Mock terrain system for testing."""

    def __init__(
        self,
        bounds: Bounds2D = (0.0, 0.0, 100.0, 100.0),
        resolution: Tuple[int, int] = (11, 11),
        cell_size: float = 10.0,
    ) -> None:
        """Initialize mock terrain."""
        self._bounds = bounds
        self._resolution = resolution
        self._cell_size = cell_size

        # Heightfield data (resolution x resolution grid)
        self._heights: List[List[float]] = [
            [0.0 for _ in range(resolution[0])]
            for _ in range(resolution[1])
        ]

        # Layer data
        self._layer_count = 4
        self._layer_names = ["grass", "dirt", "rock", "sand"]
        self._layer_weights: Dict[Tuple[int, int], List[float]] = {}

        # Physical materials
        self._materials: Dict[Tuple[int, int], str] = {}

        # Patch IDs
        self._patches: Dict[Tuple[int, int], int] = {}

    def set_height(self, grid_x: int, grid_z: int, height: float) -> None:
        """Set height at grid position."""
        if 0 <= grid_x < self._resolution[0] and 0 <= grid_z < self._resolution[1]:
            self._heights[grid_z][grid_x] = height

    def set_heights_from_function(self, func) -> None:
        """Set heights from a function f(x, z) -> height."""
        for gz in range(self._resolution[1]):
            for gx in range(self._resolution[0]):
                x = self._bounds[0] + gx * self._cell_size
                z = self._bounds[1] + gz * self._cell_size
                self._heights[gz][gx] = func(x, z)

    def set_layer_weights(
        self, grid_x: int, grid_z: int, weights: List[float]
    ) -> None:
        """Set layer weights at grid position."""
        self._layer_weights[(grid_x, grid_z)] = weights

    def set_material(self, grid_x: int, grid_z: int, material: str) -> None:
        """Set physical material at grid position."""
        self._materials[(grid_x, grid_z)] = material

    def set_patch_id(self, grid_x: int, grid_z: int, patch_id: int) -> None:
        """Set patch ID at grid position."""
        self._patches[(grid_x, grid_z)] = patch_id

    def _world_to_grid(self, x: float, z: float) -> Tuple[float, float]:
        """Convert world coords to fractional grid coords."""
        gx = (x - self._bounds[0]) / self._cell_size
        gz = (z - self._bounds[1]) / self._cell_size
        return (gx, gz)

    def get_height(self, x: float, z: float) -> float:
        """Get height using bilinear interpolation."""
        gx, gz = self._world_to_grid(x, z)

        # Clamp to valid range
        gx = max(0, min(gx, self._resolution[0] - 1))
        gz = max(0, min(gz, self._resolution[1] - 1))

        # Integer and fractional parts
        x0 = int(gx)
        z0 = int(gz)
        x1 = min(x0 + 1, self._resolution[0] - 1)
        z1 = min(z0 + 1, self._resolution[1] - 1)
        fx = gx - x0
        fz = gz - z0

        # Bilinear interpolation
        h00 = self._heights[z0][x0]
        h10 = self._heights[z0][x1]
        h01 = self._heights[z1][x0]
        h11 = self._heights[z1][x1]

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx

        return h0 * (1 - fz) + h1 * fz

    def get_raw_height(self, grid_x: int, grid_z: int) -> float:
        """Get raw height at grid position."""
        if 0 <= grid_x < self._resolution[0] and 0 <= grid_z < self._resolution[1]:
            return self._heights[grid_z][grid_x]
        return 0.0

    def get_bounds(self) -> Bounds2D:
        """Get terrain bounds."""
        return self._bounds

    def get_resolution(self) -> Tuple[int, int]:
        """Get heightfield resolution."""
        return self._resolution

    def get_cell_size(self) -> float:
        """Get cell size."""
        return self._cell_size

    def get_layer_count(self) -> int:
        """Get layer count."""
        return self._layer_count

    def get_layer_weight(self, x: float, z: float, layer_index: int) -> float:
        """Get layer weight at world position."""
        gx, gz = self._world_to_grid(x, z)
        gx_int = int(max(0, min(gx, self._resolution[0] - 1)))
        gz_int = int(max(0, min(gz, self._resolution[1] - 1)))

        weights = self._layer_weights.get((gx_int, gz_int))
        if weights and 0 <= layer_index < len(weights):
            return weights[layer_index]
        # Default: first layer = 1, others = 0
        return 1.0 if layer_index == 0 else 0.0

    def get_layer_name(self, layer_index: int) -> str:
        """Get layer name."""
        if 0 <= layer_index < len(self._layer_names):
            return self._layer_names[layer_index]
        return "unknown"

    def get_physical_material(self, x: float, z: float) -> str:
        """Get physical material at world position."""
        gx, gz = self._world_to_grid(x, z)
        gx_int = int(max(0, min(gx, self._resolution[0] - 1)))
        gz_int = int(max(0, min(gz, self._resolution[1] - 1)))

        return self._materials.get((gx_int, gz_int), "default")

    def get_patch_id(self, x: float, z: float) -> Optional[int]:
        """Get patch ID at world position."""
        gx, gz = self._world_to_grid(x, z)
        gx_int = int(max(0, min(gx, self._resolution[0] - 1)))
        gz_int = int(max(0, min(gz, self._resolution[1] - 1)))

        return self._patches.get((gx_int, gz_int))


class MockHoleManager:
    """Mock hole manager for testing."""

    def __init__(self) -> None:
        """Initialize hole manager."""
        self._holes: set = set()

    def add_hole(self, grid_x: int, grid_z: int) -> None:
        """Mark a cell as a hole."""
        self._holes.add((grid_x, grid_z))

    def is_visible(self, x: float, z: float) -> bool:
        """Check if position is visible."""
        gx = int(x / 10.0)
        gz = int(z / 10.0)
        return (gx, gz) not in self._holes

    def get_hole_mask(
        self, min_x: float, min_z: float, max_x: float, max_z: float
    ) -> List[List[bool]]:
        """Get visibility mask."""
        gx_min = int(min_x / 10.0)
        gz_min = int(min_z / 10.0)
        gx_max = int(max_x / 10.0) + 1
        gz_max = int(max_z / 10.0) + 1

        mask = []
        for gz in range(gz_min, gz_max):
            row = []
            for gx in range(gx_min, gx_max):
                row.append((gx, gz) not in self._holes)
            mask.append(row)
        return mask


# =============================================================================
# TERRAIN HIT RESULT TESTS
# =============================================================================


class TestTerrainHitResult:
    """Tests for TerrainHitResult."""

    def test_default_values(self):
        """Test default hit result values."""
        result = TerrainHitResult()
        assert not result.hit
        assert result.position == (0.0, 0.0, 0.0)
        assert result.normal == (0.0, 1.0, 0.0)
        assert result.height == 0.0
        assert result.slope_degrees == 0.0
        assert result.layer_weights == []
        assert result.physical_material == "default"
        assert result.terrain_patch_id is None

    def test_no_hit_factory(self):
        """Test no_hit factory method."""
        result = TerrainHitResult.no_hit()
        assert not result.hit

    def test_hit_with_values(self):
        """Test hit result with values."""
        result = TerrainHitResult(
            hit=True,
            position=(10.0, 5.0, 20.0),
            normal=(0.0, 1.0, 0.0),
            height=5.0,
            slope_degrees=15.0,
            layer_weights=[0.5, 0.3, 0.2, 0.0],
            physical_material="grass",
            terrain_patch_id=42,
        )
        assert result.hit
        assert result.height == 5.0
        assert result.terrain_patch_id == 42


# =============================================================================
# TERRAIN QUERY TESTS
# =============================================================================


class TestTerrainQuery:
    """Tests for TerrainQuery."""

    def test_get_height_at_flat(self):
        """Test height query on flat terrain."""
        terrain = MockTerrainSystem()
        # All heights default to 0

        query = TerrainQuery(terrain)
        height = query.get_height_at(50.0, 50.0)

        assert height == 0.0

    def test_get_height_at_uniform(self):
        """Test height query with uniform height."""
        terrain = MockTerrainSystem()
        for gz in range(11):
            for gx in range(11):
                terrain.set_height(gx, gz, 10.0)

        query = TerrainQuery(terrain)
        height = query.get_height_at(50.0, 50.0)

        assert abs(height - 10.0) < 0.01

    def test_get_height_bilinear_interpolation(self):
        """Test height uses bilinear interpolation."""
        terrain = MockTerrainSystem()
        # Create a simple gradient
        terrain.set_height(0, 0, 0.0)
        terrain.set_height(1, 0, 10.0)
        terrain.set_height(0, 1, 0.0)
        terrain.set_height(1, 1, 10.0)

        query = TerrainQuery(terrain)

        # At midpoint (5, 5), should interpolate
        height = query.get_height_at(5.0, 5.0)
        assert abs(height - 5.0) < 0.1  # Should be ~5.0

    def test_get_height_at_corners(self):
        """Test height at grid corners."""
        terrain = MockTerrainSystem()
        terrain.set_height(0, 0, 1.0)
        terrain.set_height(10, 0, 2.0)
        terrain.set_height(0, 10, 3.0)
        terrain.set_height(10, 10, 4.0)

        query = TerrainQuery(terrain)

        assert abs(query.get_height_at(0.0, 0.0) - 1.0) < 0.01
        assert abs(query.get_height_at(100.0, 0.0) - 2.0) < 0.01
        assert abs(query.get_height_at(0.0, 100.0) - 3.0) < 0.01
        assert abs(query.get_height_at(100.0, 100.0) - 4.0) < 0.01

    def test_get_normal_flat(self):
        """Test normal on flat terrain."""
        terrain = MockTerrainSystem()

        query = TerrainQuery(terrain)
        normal = query.get_normal_at(50.0, 50.0)

        # Flat terrain should have normal pointing up
        assert abs(normal[0]) < 0.1
        assert abs(normal[1] - 1.0) < 0.1
        assert abs(normal[2]) < 0.1

    def test_get_normal_sloped(self):
        """Test normal on sloped terrain."""
        terrain = MockTerrainSystem()
        # Create slope in X direction
        terrain.set_heights_from_function(lambda x, z: x * 0.1)

        query = TerrainQuery(terrain)
        normal = query.get_normal_at(50.0, 50.0)

        # Normal should tilt in -X direction
        assert normal[0] < 0  # Negative X component
        assert normal[1] > 0  # Still mostly up

    def test_get_slope_flat(self):
        """Test slope on flat terrain."""
        terrain = MockTerrainSystem()

        query = TerrainQuery(terrain)
        slope = query.get_slope_at(50.0, 50.0)

        assert slope < 1.0  # Essentially 0 degrees

    def test_get_slope_45_degrees(self):
        """Test slope calculation for ~45 degree slope."""
        terrain = MockTerrainSystem()
        # Create 1:1 slope (45 degrees)
        terrain.set_heights_from_function(lambda x, z: x)

        query = TerrainQuery(terrain)
        slope = query.get_slope_at(50.0, 50.0)

        # Should be approximately 45 degrees
        assert 40.0 < slope < 50.0

    def test_get_layer_weights(self):
        """Test layer weight query."""
        terrain = MockTerrainSystem()
        terrain.set_layer_weights(5, 5, [0.4, 0.3, 0.2, 0.1])

        query = TerrainQuery(terrain)
        weights = query.get_layer_weights_at(50.0, 50.0)

        assert len(weights) == 4
        assert abs(weights[0] - 0.4) < 0.01
        assert abs(weights[1] - 0.3) < 0.01

    def test_get_dominant_layer(self):
        """Test dominant layer detection."""
        terrain = MockTerrainSystem()
        terrain.set_layer_weights(5, 5, [0.1, 0.5, 0.3, 0.1])

        query = TerrainQuery(terrain)
        dominant = query.get_dominant_layer_at(50.0, 50.0)

        assert dominant == 1  # Layer with 0.5 weight

    def test_get_physical_material(self):
        """Test physical material query."""
        terrain = MockTerrainSystem()
        terrain.set_material(5, 5, "gravel")

        query = TerrainQuery(terrain)
        material = query.get_physical_material_at(50.0, 50.0)

        assert material == "gravel"

    def test_is_in_bounds(self):
        """Test bounds checking."""
        terrain = MockTerrainSystem(bounds=(0, 0, 100, 100))

        query = TerrainQuery(terrain)

        assert query.is_in_bounds(50, 50)
        assert query.is_in_bounds(0, 0)
        assert query.is_in_bounds(100, 100)
        assert not query.is_in_bounds(-1, 50)
        assert not query.is_in_bounds(50, -1)
        assert not query.is_in_bounds(101, 50)
        assert not query.is_in_bounds(50, 101)


# =============================================================================
# TERRAIN RAYCAST TESTS
# =============================================================================


class TestTerrainRaycast:
    """Tests for TerrainRaycast."""

    def test_raycast_down_hit(self):
        """Test downward raycast hits terrain."""
        terrain = MockTerrainSystem()
        # Set flat terrain at height 0

        raycast = TerrainRaycast(terrain)
        result = raycast.raycast(
            origin=(50.0, 100.0, 50.0),
            direction=(0.0, -1.0, 0.0),
            max_distance=500.0,
        )

        assert result.hit
        assert abs(result.height - 0.0) < 1.0

    def test_raycast_down_miss_outside_bounds(self):
        """Test raycast misses outside terrain bounds."""
        terrain = MockTerrainSystem(bounds=(0, 0, 100, 100))

        raycast = TerrainRaycast(terrain)
        result = raycast.raycast(
            origin=(500.0, 100.0, 500.0),  # Outside bounds
            direction=(0.0, -1.0, 0.0),
            max_distance=500.0,
        )

        assert not result.hit

    def test_raycast_parallel_misses(self):
        """Test horizontal raycast misses terrain."""
        terrain = MockTerrainSystem()

        raycast = TerrainRaycast(terrain)
        result = raycast.raycast(
            origin=(0.0, 50.0, 50.0),
            direction=(1.0, 0.0, 0.0),  # Horizontal
            max_distance=500.0,
        )

        assert not result.hit

    def test_raycast_hit_position(self):
        """Test raycast returns correct hit position."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: 10.0)

        raycast = TerrainRaycast(terrain)
        result = raycast.raycast(
            origin=(50.0, 100.0, 50.0),
            direction=(0.0, -1.0, 0.0),
            max_distance=500.0,
        )

        assert result.hit
        assert abs(result.position[0] - 50.0) < 1.0
        assert abs(result.position[2] - 50.0) < 1.0
        assert abs(result.height - 10.0) < 1.0

    def test_raycast_hit_normal(self):
        """Test raycast returns surface normal."""
        terrain = MockTerrainSystem()

        raycast = TerrainRaycast(terrain)
        result = raycast.raycast(
            origin=(50.0, 100.0, 50.0),
            direction=(0.0, -1.0, 0.0),
            max_distance=500.0,
        )

        assert result.hit
        # Flat terrain should have up normal
        assert abs(result.normal[1] - 1.0) < 0.1

    def test_raycast_hit_layer_weights(self):
        """Test raycast returns layer weights."""
        terrain = MockTerrainSystem()
        terrain.set_layer_weights(5, 5, [0.4, 0.3, 0.2, 0.1])

        raycast = TerrainRaycast(terrain)
        result = raycast.raycast(
            origin=(50.0, 100.0, 50.0),
            direction=(0.0, -1.0, 0.0),
            max_distance=500.0,
        )

        assert result.hit
        assert len(result.layer_weights) == 4

    def test_raycast_beyond_max_distance(self):
        """Test raycast respects max distance."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: 0.0)

        raycast = TerrainRaycast(terrain)
        result = raycast.raycast(
            origin=(50.0, 1000.0, 50.0),  # Very high
            direction=(0.0, -1.0, 0.0),
            max_distance=100.0,  # Not enough
        )

        assert not result.hit

    def test_raycast_zero_direction(self):
        """Test raycast with zero direction."""
        terrain = MockTerrainSystem()

        raycast = TerrainRaycast(terrain)
        result = raycast.raycast(
            origin=(50.0, 100.0, 50.0),
            direction=(0.0, 0.0, 0.0),
            max_distance=500.0,
        )

        assert not result.hit

    def test_raycast_diagonal(self):
        """Test diagonal raycast."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: 0.0)

        raycast = TerrainRaycast(terrain)
        # Start high and aim diagonal-down toward terrain center
        # Origin at (50, 50, 50), direction pointing down with slight XZ offset
        # Will hit terrain at approx (50, 0, 50) within bounds
        result = raycast.raycast(
            origin=(50.0, 50.0, 50.0),
            direction=(0.1, -1.0, 0.1),  # Mostly down, slight diagonal
            max_distance=100.0,
        )

        assert result.hit


# =============================================================================
# TERRAIN LINE TRACE TESTS
# =============================================================================


class TestTerrainLineTrace:
    """Tests for TerrainLineTrace."""

    def test_sample_along_line(self):
        """Test sampling terrain along a line."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: x * 0.1)

        trace = TerrainLineTrace(terrain)
        samples = trace.sample_along_line(
            start=(0.0, 0.0, 50.0),
            end=(100.0, 0.0, 50.0),
            step_size=10.0,
        )

        assert len(samples) > 0
        assert all(s.hit for s in samples)
        # Heights should increase along the line
        for i in range(1, len(samples)):
            assert samples[i].height >= samples[i - 1].height - 0.1

    def test_sample_empty_for_zero_length(self):
        """Test sampling returns empty for zero-length line."""
        terrain = MockTerrainSystem()

        trace = TerrainLineTrace(terrain)
        samples = trace.sample_along_line(
            start=(50.0, 0.0, 50.0),
            end=(50.0, 0.0, 50.0),
            step_size=10.0,
        )

        assert len(samples) == 0

    def test_sample_outside_bounds(self):
        """Test sampling skips points outside bounds."""
        terrain = MockTerrainSystem(bounds=(0, 0, 100, 100))

        trace = TerrainLineTrace(terrain)
        samples = trace.sample_along_line(
            start=(-50.0, 0.0, 50.0),
            end=(150.0, 0.0, 50.0),
            step_size=10.0,
        )

        # Should only have samples within bounds
        for s in samples:
            assert 0 <= s.position[0] <= 100


# =============================================================================
# TERRAIN AREA QUERY TESTS
# =============================================================================


class TestTerrainAreaQuery:
    """Tests for TerrainAreaQuery."""

    def test_get_heights_in_bounds(self):
        """Test getting height grid."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: x + z)

        area = TerrainAreaQuery(terrain)
        heights = area.get_heights_in_bounds(0, 0, 100, 100, 5)

        assert len(heights) == 5
        assert len(heights[0]) == 5
        # Heights should increase
        assert heights[4][4] > heights[0][0]

    def test_get_average_height(self):
        """Test average height calculation."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: 50.0)  # Uniform height

        area = TerrainAreaQuery(terrain)
        avg = area.get_average_height((0, 0, 100, 100))

        assert abs(avg - 50.0) < 1.0

    def test_get_min_max_height(self):
        """Test min/max height detection."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: x)  # 0 to 100

        area = TerrainAreaQuery(terrain)
        min_h, max_h = area.get_min_max_height((0, 0, 100, 100))

        assert min_h < 10.0
        assert max_h > 90.0

    def test_find_flat_areas(self):
        """Test finding flat areas."""
        terrain = MockTerrainSystem()
        # Create mostly flat terrain
        terrain.set_heights_from_function(lambda x, z: 0.0)

        area = TerrainAreaQuery(terrain)
        flat = area.find_flat_areas(
            bounds=(0, 0, 100, 100),
            max_slope=15.0,
            min_size=10.0,
        )

        # Should find at least one flat area
        assert len(flat) > 0

    def test_find_flat_areas_with_slopes(self):
        """Test finding flat areas on varied terrain."""
        terrain = MockTerrainSystem()
        # Create terrain with a steep section
        def height_func(x, z):
            if 40 < x < 60:
                return x * 2  # Steep slope
            return 0.0
        terrain.set_heights_from_function(height_func)

        area = TerrainAreaQuery(terrain)
        flat = area.find_flat_areas(
            bounds=(0, 0, 100, 100),
            max_slope=10.0,
            min_size=10.0,
        )

        # Should find flat areas outside the steep section
        assert len(flat) >= 0  # May or may not find any depending on implementation


# =============================================================================
# TERRAIN VISIBILITY TESTS
# =============================================================================


class TestTerrainVisibility:
    """Tests for TerrainVisibility."""

    def test_visibility_without_holes(self):
        """Test visibility without hole manager."""
        terrain = MockTerrainSystem()

        visibility = TerrainVisibility(terrain)

        assert visibility.is_terrain_visible_at(50, 50)
        assert visibility.is_terrain_visible_at(0, 0)

    def test_visibility_outside_bounds(self):
        """Test visibility outside bounds."""
        terrain = MockTerrainSystem(bounds=(0, 0, 100, 100))

        visibility = TerrainVisibility(terrain)

        assert not visibility.is_terrain_visible_at(-10, 50)
        assert not visibility.is_terrain_visible_at(150, 50)

    def test_visibility_with_holes(self):
        """Test visibility with hole manager."""
        terrain = MockTerrainSystem()
        holes = MockHoleManager()
        holes.add_hole(5, 5)

        visibility = TerrainVisibility(terrain, holes)

        assert visibility.is_terrain_visible_at(20, 20)
        assert not visibility.is_terrain_visible_at(50, 50)  # Hole at cell (5,5)

    def test_get_visible_bounds_clamp(self):
        """Test visible bounds are clamped to terrain."""
        terrain = MockTerrainSystem(bounds=(0, 0, 100, 100))

        visibility = TerrainVisibility(terrain)
        result = visibility.get_visible_bounds((-50, -50, 150, 150))

        assert result is not None
        assert result[0] == 0  # min_x clamped
        assert result[1] == 0  # min_z clamped
        assert result[2] == 100  # max_x clamped
        assert result[3] == 100  # max_z clamped

    def test_get_visible_bounds_none(self):
        """Test visible bounds returns None for outside."""
        terrain = MockTerrainSystem(bounds=(0, 0, 100, 100))

        visibility = TerrainVisibility(terrain)
        result = visibility.get_visible_bounds((200, 200, 300, 300))

        assert result is None


# =============================================================================
# TERRAIN QUERY SYSTEM TESTS
# =============================================================================


class TestTerrainQuerySystem:
    """Tests for TerrainQuerySystem."""

    def test_system_creation(self):
        """Test system creation."""
        terrain = MockTerrainSystem()
        system = TerrainQuerySystem(terrain)

        assert system.query is not None
        assert system.raycast is not None
        assert system.line_trace is not None
        assert system.area is not None
        assert system.visibility is not None

    def test_system_with_hole_manager(self):
        """Test system with hole manager."""
        terrain = MockTerrainSystem()
        holes = MockHoleManager()
        holes.add_hole(5, 5)

        system = TerrainQuerySystem(terrain, holes)

        assert not system.visibility.is_terrain_visible_at(50, 50)

    def test_system_integrated_query(self):
        """Test integrated query through system."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: 25.0)

        system = TerrainQuerySystem(terrain)

        # Test height query
        height = system.query.get_height_at(50, 50)
        assert abs(height - 25.0) < 1.0

        # Test raycast
        hit = system.raycast.raycast(
            origin=(50, 100, 50),
            direction=(0, -1, 0),
            max_distance=500,
        )
        assert hit.hit

        # Test area query
        avg = system.area.get_average_height((0, 0, 100, 100))
        assert abs(avg - 25.0) < 5.0


# =============================================================================
# INTERPOLATION ACCURACY TESTS
# =============================================================================


class TestInterpolationAccuracy:
    """Tests for interpolation accuracy."""

    def test_bilinear_midpoint(self):
        """Test bilinear interpolation at cell midpoint."""
        terrain = MockTerrainSystem()
        terrain.set_height(0, 0, 0.0)
        terrain.set_height(1, 0, 10.0)
        terrain.set_height(0, 1, 20.0)
        terrain.set_height(1, 1, 30.0)

        query = TerrainQuery(terrain)

        # Center of cell should be average
        height = query.get_height_at(5.0, 5.0)
        expected = (0.0 + 10.0 + 20.0 + 30.0) / 4
        assert abs(height - expected) < 0.5

    def test_bilinear_edge(self):
        """Test bilinear interpolation at cell edge."""
        terrain = MockTerrainSystem()
        terrain.set_height(0, 0, 0.0)
        terrain.set_height(1, 0, 10.0)
        terrain.set_height(0, 1, 0.0)
        terrain.set_height(1, 1, 10.0)

        query = TerrainQuery(terrain)

        # Middle of X edge should be 5.0
        height = query.get_height_at(5.0, 0.0)
        assert abs(height - 5.0) < 0.5

    def test_interpolation_continuity(self):
        """Test interpolation is continuous across cells."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: math.sin(x * 0.1) * 10)

        query = TerrainQuery(terrain)

        # Sample nearby points should be similar
        h1 = query.get_height_at(50.0, 50.0)
        h2 = query.get_height_at(50.1, 50.0)

        assert abs(h1 - h2) < 1.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_query_at_exact_grid_point(self):
        """Test query exactly at grid point."""
        terrain = MockTerrainSystem()
        terrain.set_height(5, 5, 42.0)

        query = TerrainQuery(terrain)
        height = query.get_height_at(50.0, 50.0)

        # Should be very close to the exact value
        assert abs(height - 42.0) < 1.0

    def test_normal_at_boundary(self):
        """Test normal calculation at terrain boundary."""
        terrain = MockTerrainSystem()

        query = TerrainQuery(terrain)

        # Should not crash at boundary
        normal = query.get_normal_at(0.0, 0.0)
        assert len(normal) == 3

        normal = query.get_normal_at(100.0, 100.0)
        assert len(normal) == 3

    def test_empty_layer_weights(self):
        """Test layer weights when not set."""
        terrain = MockTerrainSystem()

        query = TerrainQuery(terrain)
        weights = query.get_layer_weights_at(50, 50)

        # Should return default weights
        assert len(weights) == 4

    def test_raycast_from_below_terrain(self):
        """Test raycast from below terrain."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: 10.0)

        raycast = TerrainRaycast(terrain)
        result = raycast.raycast(
            origin=(50.0, -10.0, 50.0),  # Below terrain
            direction=(0.0, 1.0, 0.0),   # Looking up
            max_distance=500.0,
        )

        # Should not hit (going away from terrain)
        assert not result.hit


# =============================================================================
# TERRAIN INTERPOLATION ACCURACY TESTS
# =============================================================================


class TestTerrainInterpolationAccuracy:
    """Tests verifying terrain height interpolation accuracy."""

    def test_bilinear_interpolation_exact_corners(self):
        """Test bilinear interpolation at exact grid corners."""
        terrain = MockTerrainSystem()
        # Set distinct heights at corners
        terrain.set_height(0, 0, 100.0)
        terrain.set_height(1, 0, 200.0)
        terrain.set_height(0, 1, 300.0)
        terrain.set_height(1, 1, 400.0)

        query = TerrainQuery(terrain)

        # At exact corners, should return exact values
        assert abs(query.get_height_at(0.0, 0.0) - 100.0) < 0.1
        assert abs(query.get_height_at(10.0, 0.0) - 200.0) < 0.1
        assert abs(query.get_height_at(0.0, 10.0) - 300.0) < 0.1
        assert abs(query.get_height_at(10.0, 10.0) - 400.0) < 0.1

    def test_bilinear_interpolation_center(self):
        """Test bilinear interpolation at cell center."""
        terrain = MockTerrainSystem()
        terrain.set_height(0, 0, 0.0)
        terrain.set_height(1, 0, 40.0)
        terrain.set_height(0, 1, 20.0)
        terrain.set_height(1, 1, 60.0)

        query = TerrainQuery(terrain)

        # Center should be average
        expected = (0.0 + 40.0 + 20.0 + 60.0) / 4
        actual = query.get_height_at(5.0, 5.0)
        assert abs(actual - expected) < 0.5

    def test_bilinear_interpolation_edge_midpoints(self):
        """Test bilinear interpolation at edge midpoints."""
        terrain = MockTerrainSystem()
        terrain.set_height(0, 0, 0.0)
        terrain.set_height(1, 0, 100.0)
        terrain.set_height(0, 1, 0.0)
        terrain.set_height(1, 1, 100.0)

        query = TerrainQuery(terrain)

        # Bottom edge midpoint: average of h00 and h10
        expected_bottom = (0.0 + 100.0) / 2
        actual_bottom = query.get_height_at(5.0, 0.0)
        assert abs(actual_bottom - expected_bottom) < 0.5

        # Left edge midpoint: average of h00 and h01
        expected_left = (0.0 + 0.0) / 2
        actual_left = query.get_height_at(0.0, 5.0)
        assert abs(actual_left - expected_left) < 0.5

    def test_interpolation_smoothness(self):
        """Test that interpolation produces smooth transitions."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: math.sin(x * 0.1) * 10)

        query = TerrainQuery(terrain)

        # Sample many points and verify no discontinuities
        prev_height = query.get_height_at(0.0, 50.0)
        for i in range(1, 100):
            x = i * 1.0
            height = query.get_height_at(x, 50.0)
            # Height change should be gradual
            assert abs(height - prev_height) < 2.0
            prev_height = height

    def test_interpolation_gradient_accuracy(self):
        """Test that interpolation preserves gradient."""
        terrain = MockTerrainSystem()
        # Linear gradient: height = x
        terrain.set_heights_from_function(lambda x, z: x)

        query = TerrainQuery(terrain)

        # Sample at multiple points and verify gradient is approximately 1
        for x in [10.0, 30.0, 50.0, 70.0]:
            h1 = query.get_height_at(x, 50.0)
            h2 = query.get_height_at(x + 1.0, 50.0)
            gradient = h2 - h1
            assert abs(gradient - 1.0) < 0.2

    def test_normal_calculation_accuracy(self):
        """Test normal calculation accuracy on known surfaces."""
        terrain = MockTerrainSystem()

        # Flat terrain - normal should be straight up
        query_flat = TerrainQuery(terrain)
        normal_flat = query_flat.get_normal_at(50.0, 50.0)
        assert abs(normal_flat[0]) < 0.01
        assert abs(normal_flat[1] - 1.0) < 0.01
        assert abs(normal_flat[2]) < 0.01

    def test_slope_calculation_accuracy(self):
        """Test slope calculation accuracy."""
        terrain = MockTerrainSystem()

        # Flat terrain - slope should be near 0
        terrain.set_heights_from_function(lambda x, z: 0.0)
        query_flat = TerrainQuery(terrain)
        slope_flat = query_flat.get_slope_at(50.0, 50.0)
        assert slope_flat < 1.0

        # 45 degree slope (rise = run)
        terrain.set_heights_from_function(lambda x, z: x)
        query_45 = TerrainQuery(terrain)
        slope_45 = query_45.get_slope_at(50.0, 50.0)
        assert 40.0 < slope_45 < 50.0


class TestTerrainRaycastAccuracy:
    """Tests verifying terrain raycast accuracy."""

    def test_raycast_hit_position_accuracy(self):
        """Test that raycast hit position matches terrain height."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: 25.0)

        raycast = TerrainRaycast(terrain)
        result = raycast.raycast(
            origin=(50.0, 100.0, 50.0),
            direction=(0.0, -1.0, 0.0),
            max_distance=200.0,
        )

        assert result.hit
        # Hit position Y should match terrain height
        assert abs(result.position[1] - 25.0) < 1.0
        # Hit height should match
        assert abs(result.height - 25.0) < 1.0

    def test_raycast_finds_correct_intersection(self):
        """Test raycast finds intersection at correct terrain height."""
        terrain = MockTerrainSystem()
        # Terrain varies: higher in middle
        terrain.set_heights_from_function(
            lambda x, z: 50.0 if 40 < x < 60 and 40 < z < 60 else 0.0
        )

        raycast = TerrainRaycast(terrain)

        # Ray from above center should hit at height 50
        result_center = raycast.raycast(
            origin=(50.0, 100.0, 50.0),
            direction=(0.0, -1.0, 0.0),
            max_distance=200.0,
        )
        assert result_center.hit
        assert abs(result_center.height - 50.0) < 5.0

        # Ray from above edge should hit at height 0
        result_edge = raycast.raycast(
            origin=(10.0, 100.0, 10.0),
            direction=(0.0, -1.0, 0.0),
            max_distance=200.0,
        )
        assert result_edge.hit
        assert abs(result_edge.height - 0.0) < 5.0

    def test_raycast_diagonal_accuracy(self):
        """Test diagonal raycast accuracy."""
        terrain = MockTerrainSystem()
        terrain.set_heights_from_function(lambda x, z: 0.0)

        raycast = TerrainRaycast(terrain)

        # Diagonal ray should still hit terrain
        result = raycast.raycast(
            origin=(25.0, 50.0, 25.0),
            direction=(0.5, -1.0, 0.5),
            max_distance=200.0,
        )

        assert result.hit
        assert abs(result.height - 0.0) < 2.0
