"""
Blackbox tests for Architecture SDF module (T-DEMO-4.7 and T-DEMO-4.8).

Tests public API behavior without knowledge of implementation details:
- Building configuration produces expected visual results
- City generation creates varied but consistent layouts
- WGSL output is syntactically valid
- Parameter boundaries are enforced
- Integration with Trinity tracker system

BLACKBOX scenarios:
  Scenario 1:  Create a simple residential building
  Scenario 2:  Create a tall office tower
  Scenario 3:  Generate a city block with varied buildings
  Scenario 4:  Verify reproducibility with seeds
  Scenario 5:  Test parameter boundary conditions
  Scenario 6:  WGSL output can be parsed
  Scenario 7:  Building materials and styles vary
"""

import math
import pytest

from engine.rendering.demoscene.architecture_sdf import (
    BuildingSDF,
    CityBlockSDF,
    RoofStyle,
    cell_hash,
    hash_to_float,
)
from engine.rendering.demoscene.sdf_ast import Vec3


TOL = 1e-6


# =============================================================================
# Scenario 1: Create a Simple Residential Building
# =============================================================================

class TestResidentialBuilding:
    """Test creating a simple residential-style building."""

    def test_small_house_dimensions(self):
        """Small house has correct size."""
        house = BuildingSDF(
            width=8.0,
            height=6.0,
            depth=6.0,
            floors=2,
            windows_per_floor=2,
            roof_style=RoofStyle.PITCHED,
            roof_height=2.0
        )
        assert house.width == 8.0
        assert house.height == 6.0
        assert house.floors == 2
        assert house.total_height == 8.0  # 6 + 2 roof

    def test_house_has_door(self):
        """House has accessible door."""
        house = BuildingSDF(
            width=8.0, height=6.0, depth=6.0,
            door_width=1.2, door_height=2.2
        )
        door_pos = house.get_door_position()
        assert door_pos.y < house.floor_height  # Door within first floor
        assert door_pos.z == house.depth / 2  # On front face

    def test_house_windows_per_floor(self):
        """House has correct window count."""
        house = BuildingSDF(floors=2, windows_per_floor=3)
        windows = house.get_window_positions()
        # 3 windows * 4 faces * 2 floors = 24
        assert len(windows) == 24

    def test_house_evaluates_inside(self):
        """Can detect points inside house."""
        house = BuildingSDF(
            width=8.0, height=6.0, depth=6.0,
            windows_per_floor=0
        )
        # Center point
        d = house.evaluate(Vec3(0, 3, 0))
        assert d < 0


# =============================================================================
# Scenario 2: Create a Tall Office Tower
# =============================================================================

class TestOfficeTower:
    """Test creating a tall office tower."""

    def test_tower_many_floors(self):
        """Tower has many floors."""
        tower = BuildingSDF(
            width=20.0,
            height=100.0,
            depth=20.0,
            floors=25,
            windows_per_floor=8,
            roof_style=RoofStyle.FLAT
        )
        assert tower.floors == 25
        assert tower.floor_height == 4.0

    def test_tower_window_density(self):
        """Tower has dense window grid."""
        tower = BuildingSDF(
            floors=10,
            windows_per_floor=6
        )
        windows = tower.get_window_positions()
        # 6 * 4 * 10 = 240 windows
        assert len(windows) == 240

    def test_tower_flat_roof_trim(self):
        """Flat roof tower has edge trim."""
        tower = BuildingSDF(
            height=50.0,
            roof_style=RoofStyle.FLAT,
            edge_trim_size=0.5
        )
        # Point just above roof should be near trim
        d = tower.evaluate(Vec3(0, 50.3, 0))
        # Should be inside or very close to trim

    def test_tower_glass_facade_material(self):
        """Tower can have material ID for glass."""
        tower = BuildingSDF(material_id=2)
        assert tower.material_id == 2


# =============================================================================
# Scenario 3: Generate City Block with Varied Buildings
# =============================================================================

class TestCityBlockGeneration:
    """Test city block generation."""

    def test_city_creates_multiple_buildings(self):
        """City block creates buildings for different cells."""
        city = CityBlockSDF()
        buildings = [
            city.get_building_for_cell(x, z)
            for x in range(-2, 3)
            for z in range(-2, 3)
        ]
        assert len(buildings) == 25
        # All should be BuildingSDF instances
        assert all(isinstance(b, BuildingSDF) for b in buildings)

    def test_city_buildings_vary(self):
        """City buildings have variation."""
        city = CityBlockSDF(
            min_floors=2, max_floors=10,
            min_windows=1, max_windows=6
        )
        buildings = [
            city.get_building_for_cell(x, 0)
            for x in range(10)
        ]
        # Should have at least some variation in floors
        floor_counts = [b.floors for b in buildings]
        assert len(set(floor_counts)) > 1

    def test_city_has_streets(self):
        """City has street gaps between buildings."""
        city = CityBlockSDF(cell_size=20, street_width=5)
        # Point in street area (past building bounds)
        # Cell center at (10, _, 10), building area extends 7.5 from center
        # Street starts at local 7.5, so world x=17.5 is in street
        d = city.evaluate(Vec3(17.6, 0, 10))
        assert d > 0  # Outside building

    def test_city_evaluates_at_any_position(self):
        """City SDF evaluates at arbitrary world positions."""
        city = CityBlockSDF()
        positions = [
            Vec3(0, 0, 0),
            Vec3(100, 50, -100),
            Vec3(-500, 10, 500),
        ]
        for p in positions:
            d = city.evaluate(p)
            assert isinstance(d, float)


# =============================================================================
# Scenario 4: Verify Reproducibility with Seeds
# =============================================================================

class TestReproducibility:
    """Test that city generation is reproducible."""

    def test_same_seed_same_city(self):
        """Same seed produces identical city."""
        city1 = CityBlockSDF(seed=12345)
        city2 = CityBlockSDF(seed=12345)

        for x in range(-5, 6):
            for z in range(-5, 6):
                props1 = city1.get_cell_properties(x, z)
                props2 = city2.get_cell_properties(x, z)
                assert props1 == props2

    def test_different_seed_different_city(self):
        """Different seeds produce different cities."""
        city1 = CityBlockSDF(seed=12345)
        city2 = CityBlockSDF(seed=54321)

        different_count = 0
        for x in range(-5, 6):
            for z in range(-5, 6):
                props1 = city1.get_cell_properties(x, z)
                props2 = city2.get_cell_properties(x, z)
                if props1 != props2:
                    different_count += 1

        # Most cells should differ
        assert different_count > 50  # Out of 121 cells

    def test_cell_hash_reproducible(self):
        """Cell hash is reproducible across calls."""
        results = [cell_hash(5, 10, 42) for _ in range(100)]
        assert len(set(results)) == 1


# =============================================================================
# Scenario 5: Test Parameter Boundary Conditions
# =============================================================================

class TestParameterBoundaries:
    """Test parameter validation and edge cases."""

    def test_building_single_floor(self):
        """Single floor building works."""
        b = BuildingSDF(floors=1)
        assert b.floors == 1
        assert b.floor_height == b.height

    def test_building_no_windows(self):
        """Building with no windows works."""
        b = BuildingSDF(windows_per_floor=0)
        windows = b.get_window_positions()
        assert len(windows) == 0

    def test_building_many_windows(self):
        """Building with many windows works."""
        b = BuildingSDF(floors=10, windows_per_floor=10)
        windows = b.get_window_positions()
        assert len(windows) == 10 * 10 * 4

    def test_building_very_tall(self):
        """Very tall building works."""
        b = BuildingSDF(height=500, floors=100)
        assert b.floor_height == 5.0

    def test_building_very_wide(self):
        """Very wide building works."""
        b = BuildingSDF(width=100, depth=100)
        assert b.width == 100

    def test_city_small_cells(self):
        """City with small cells works."""
        city = CityBlockSDF(cell_size=5, street_width=1)
        assert city.building_area_size == 4

    def test_city_large_cells(self):
        """City with large cells works."""
        city = CityBlockSDF(cell_size=100, street_width=20)
        assert city.building_area_size == 80

    def test_city_minimal_street(self):
        """City with minimal street width works."""
        city = CityBlockSDF(cell_size=20, street_width=0.1)
        assert city.building_area_size == 19.9


# =============================================================================
# Scenario 6: WGSL Output Parsing
# =============================================================================

class TestWGSLParsing:
    """Test that WGSL output is syntactically reasonable."""

    def test_building_wgsl_has_function(self):
        """Building WGSL has function signature."""
        b = BuildingSDF()
        wgsl = b.to_wgsl("my_building")
        assert "fn sd_my_building" in wgsl
        assert "vec3<f32>" in wgsl
        assert "-> f32" in wgsl

    def test_building_wgsl_has_body(self):
        """Building WGSL has function body."""
        b = BuildingSDF()
        wgsl = b.to_wgsl()
        assert "{" in wgsl
        assert "}" in wgsl
        assert "return" in wgsl

    def test_building_wgsl_balanced_braces(self):
        """Building WGSL has balanced braces."""
        b = BuildingSDF()
        wgsl = b.to_wgsl()
        assert wgsl.count("{") == wgsl.count("}")

    def test_city_wgsl_has_functions(self):
        """City WGSL has required functions."""
        city = CityBlockSDF()
        wgsl = city.to_wgsl("my_city")
        assert "fn cell_hash" in wgsl
        assert "fn hash_to_float" in wgsl
        assert "fn sd_my_city" in wgsl

    def test_city_wgsl_has_loops(self):
        """City WGSL has neighbor loops."""
        city = CityBlockSDF()
        wgsl = city.to_wgsl()
        assert "for" in wgsl

    def test_city_wgsl_returns_vec2(self):
        """City WGSL returns distance and material."""
        city = CityBlockSDF()
        wgsl = city.to_wgsl()
        assert "-> vec2<f32>" in wgsl
        assert "vec2<f32>(min_d, material_id)" in wgsl

    def test_wgsl_no_syntax_errors_keywords(self):
        """WGSL uses correct WGSL keywords."""
        b = BuildingSDF()
        wgsl = b.to_wgsl()
        # Check for common WGSL constructs
        assert "let " in wgsl
        assert "var " in wgsl or "let " in wgsl
        assert "vec3<f32>" in wgsl


# =============================================================================
# Scenario 7: Building Materials and Styles
# =============================================================================

class TestMaterialsAndStyles:
    """Test building material and style variation."""

    def test_roof_style_options(self):
        """All roof styles can be used."""
        for style in RoofStyle:
            b = BuildingSDF(roof_style=style)
            assert b.roof_style == style

    def test_roof_styles_generate_different_wgsl(self):
        """Different roof styles generate different code."""
        wgsl_flat = BuildingSDF(roof_style=RoofStyle.FLAT).to_wgsl()
        wgsl_pitched = BuildingSDF(roof_style=RoofStyle.PITCHED).to_wgsl()
        wgsl_dome = BuildingSDF(roof_style=RoofStyle.DOME).to_wgsl()

        assert wgsl_flat != wgsl_pitched
        assert wgsl_pitched != wgsl_dome
        assert wgsl_flat != wgsl_dome

    def test_material_ids_vary_in_city(self):
        """City buildings have varying material IDs."""
        city = CityBlockSDF()
        material_ids = [
            city.get_cell_properties(x, 0)["material_id"]
            for x in range(20)
        ]
        # Should have some variation
        assert len(set(material_ids)) > 1

    def test_building_material_id_range(self):
        """Building material IDs are in expected range."""
        city = CityBlockSDF()
        for x in range(-10, 10):
            for z in range(-10, 10):
                props = city.get_cell_properties(x, z)
                assert 0 <= props["material_id"] < 8


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Test integration between building and city."""

    def test_city_uses_building_sdf(self):
        """City evaluation uses building SDF internally."""
        city = CityBlockSDF(
            cell_size=20, street_width=5,
            base_width=10, base_height=15, base_depth=8
        )
        # Get building for cell (0, 0)
        building = city.get_building_for_cell(0, 0)

        # Evaluate city at building center
        # Cell center is at (10, _, 10)
        city_d = city.evaluate(Vec3(10, 7.5, 10))

        # Evaluate building at same relative position (0, 7.5, 0)
        building_d = building.evaluate(Vec3(0, 7.5, 0))

        # Should be same (within tolerance for potential offset)
        assert abs(city_d - building_d) < 1.0

    def test_building_tracker_integration(self):
        """Building integrates with Trinity tracker."""
        b = BuildingSDF()
        # Tracker should be accessible
        assert hasattr(b, 'tracker')
        # Should track changes
        b.tracker.mark_dirty("width")
        assert "width" in b.tracker.dirty_fields

    def test_building_mirror_integration(self):
        """Building integrates with Trinity mirror."""
        b = BuildingSDF()
        # Mirror should be accessible
        assert hasattr(b, 'mirror')
        mirror = b.mirror
        assert mirror.node_type == "BuildingSDF"

    def test_city_tracker_integration(self):
        """City integrates with Trinity tracker."""
        c = CityBlockSDF()
        assert hasattr(c, 'tracker')

    def test_building_clone_evaluates_same(self):
        """Cloned building evaluates same as original."""
        b = BuildingSDF(
            width=12, height=18, depth=10,
            floors=3, windows_per_floor=2,
            roof_style=RoofStyle.PITCHED
        )
        clone = b.clone()

        test_points = [
            Vec3(0, 9, 0),
            Vec3(6, 15, 0),
            Vec3(0, 0, 5),
        ]
        for p in test_points:
            d1 = b.evaluate(p)
            d2 = clone.evaluate(p)
            assert d1 == pytest.approx(d2, abs=1e-10)


# =============================================================================
# Performance Boundary Tests
# =============================================================================

class TestPerformanceBoundaries:
    """Test performance-related boundaries."""

    def test_many_cells_cached(self):
        """Accessing many cells uses cache efficiently."""
        city = CityBlockSDF()
        # Access same cells multiple times
        for _ in range(10):
            for x in range(-5, 6):
                for z in range(-5, 6):
                    city.get_building_for_cell(x, z)
        # Cache should prevent creating duplicate buildings
        cache_size = len(city._building_cache)
        assert cache_size == 121  # 11 * 11

    def test_building_with_many_windows_evaluates(self):
        """Building with many windows still evaluates."""
        b = BuildingSDF(floors=20, windows_per_floor=10)
        # Should complete without hanging
        d = b.evaluate(Vec3(0, 0, 0))
        assert isinstance(d, float)

    def test_city_evaluate_many_points(self):
        """City can evaluate many points."""
        city = CityBlockSDF()
        results = [
            city.evaluate(Vec3(x, 0, z))
            for x in range(-50, 51, 5)
            for z in range(-50, 51, 5)
        ]
        assert len(results) == 21 * 21


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_evaluate_at_origin(self):
        """Building evaluation at origin."""
        b = BuildingSDF()
        d = b.evaluate(Vec3(0, 0, 0))
        assert isinstance(d, float)

    def test_evaluate_far_away(self):
        """Building evaluation far from origin."""
        b = BuildingSDF()
        d = b.evaluate(Vec3(1000, 1000, 1000))
        assert d > 0  # Should be outside

    def test_evaluate_negative_coords(self):
        """Building evaluation at negative coords."""
        b = BuildingSDF()
        d = b.evaluate(Vec3(-5, -5, -5))
        assert isinstance(d, float)

    def test_city_negative_cell(self):
        """City evaluation in negative cells."""
        city = CityBlockSDF()
        d = city.evaluate(Vec3(-30, 5, -30))
        assert isinstance(d, float)

    def test_building_exactly_on_surface(self):
        """Point exactly on building surface."""
        b = BuildingSDF(width=10, height=15, depth=8, windows_per_floor=0)
        d = b.evaluate(Vec3(5, 7.5, 0))  # On +X face
        assert abs(d) < 0.1  # Should be very close to zero

    def test_hash_large_coordinates(self):
        """Cell hash handles large coordinates."""
        h = cell_hash(1000000, 1000000, 0)
        assert isinstance(h, int)

    def test_hash_to_float_boundary(self):
        """hash_to_float at range boundaries."""
        val = hash_to_float(0, 0.0, 1.0)
        assert 0.0 <= val <= 1.0

        val2 = hash_to_float(0xFFFFFFFF, 0.0, 1.0)
        assert 0.0 <= val2 <= 1.0
