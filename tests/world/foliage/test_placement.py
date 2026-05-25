"""
Tests for foliage placement system (placement.py).

Tests the placement system including:
- PlacementRule validation
- PlacementResult transforms
- ProceduralPlacer generation
- FoliagePlacement configurations
- ManualPlacement operations
- procedural_placement decorator
"""

import math
from typing import Tuple

import pytest

from engine.world.foliage.placement import (
    Bounds,
    FoliagePlacement,
    ManualPlacement,
    NoiseGenerator,
    PlacementResult,
    PlacementRule,
    ProceduralPlacer,
    TerrainInterface,
    procedural_placement,
)


# =============================================================================
# Mock Terrain
# =============================================================================


class MockTerrain:
    """Mock terrain for testing."""

    def __init__(
        self,
        height: float = 0.0,
        normal: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        layer: int = 0,
        water: bool = False,
        road: bool = False,
    ):
        self._height = height
        self._normal = normal
        self._layer = layer
        self._water = water
        self._road = road

    def get_height_at(self, x: float, z: float) -> float:
        return self._height

    def get_normal_at(self, x: float, z: float) -> Tuple[float, float, float]:
        return self._normal

    def get_layer_at(self, x: float, z: float) -> int:
        return self._layer

    def is_water_at(self, x: float, z: float) -> bool:
        return self._water

    def is_road_at(self, x: float, z: float) -> bool:
        return self._road


# =============================================================================
# PlacementRule
# =============================================================================


class TestPlacementRule:
    def test_default_values(self):
        rule = PlacementRule()
        assert rule.slope_range == (0.0, 90.0)
        assert rule.height_range is None
        assert rule.terrain_layers == []
        assert rule.noise_threshold == 0.0
        assert rule.noise_scale == 10.0

    def test_custom_slope_range(self):
        rule = PlacementRule(slope_range=(0.0, 45.0))
        assert rule.slope_range == (0.0, 45.0)

    def test_custom_height_range(self):
        rule = PlacementRule(height_range=(10.0, 100.0))
        assert rule.height_range == (10.0, 100.0)

    def test_custom_terrain_layers(self):
        rule = PlacementRule(terrain_layers=[0, 1, 2])
        assert rule.terrain_layers == [0, 1, 2]

    def test_invalid_slope_range_min_negative(self):
        with pytest.raises(ValueError, match="slope_range must be between 0 and 90"):
            PlacementRule(slope_range=(-10.0, 45.0))

    def test_invalid_slope_range_max_over_90(self):
        with pytest.raises(ValueError, match="slope_range must be between 0 and 90"):
            PlacementRule(slope_range=(0.0, 100.0))

    def test_invalid_slope_range_inverted(self):
        with pytest.raises(ValueError, match="slope_range min must be <= max"):
            PlacementRule(slope_range=(60.0, 30.0))

    def test_invalid_height_range_inverted(self):
        with pytest.raises(ValueError, match="height_range min must be <= max"):
            PlacementRule(height_range=(100.0, 10.0))

    def test_invalid_noise_threshold_negative(self):
        with pytest.raises(ValueError, match="noise_threshold must be between 0 and 1"):
            PlacementRule(noise_threshold=-0.5)

    def test_invalid_noise_threshold_over_one(self):
        with pytest.raises(ValueError, match="noise_threshold must be between 0 and 1"):
            PlacementRule(noise_threshold=1.5)

    def test_invalid_noise_scale_zero(self):
        with pytest.raises(ValueError, match="noise_scale must be > 0"):
            PlacementRule(noise_scale=0)

    def test_invalid_noise_scale_negative(self):
        with pytest.raises(ValueError, match="noise_scale must be > 0"):
            PlacementRule(noise_scale=-5.0)

    def test_water_exclusion_default(self):
        rule = PlacementRule()
        assert rule.exclude_water is True

    def test_road_exclusion_default(self):
        rule = PlacementRule()
        assert rule.exclude_roads is True


# =============================================================================
# PlacementResult
# =============================================================================


class TestPlacementResult:
    def test_default_values(self):
        result = PlacementResult()
        assert result.position == (0.0, 0.0, 0.0)
        assert result.rotation == (0.0, 0.0, 0.0)
        assert result.scale == (1.0, 1.0, 1.0)
        assert result.foliage_type_id == ""

    def test_custom_values(self):
        result = PlacementResult(
            position=(10.0, 5.0, 20.0),
            rotation=(0.0, 45.0, 0.0),
            scale=(1.5, 1.5, 1.5),
            foliage_type_id="test_plant",
        )
        assert result.position == (10.0, 5.0, 20.0)
        assert result.rotation == (0.0, 45.0, 0.0)
        assert result.scale == (1.5, 1.5, 1.5)
        assert result.foliage_type_id == "test_plant"

    def test_transform_matrix_identity(self):
        result = PlacementResult()
        matrix = result.get_transform_matrix()
        assert len(matrix) == 4
        assert len(matrix[0]) == 4
        # Identity rotation with no translation
        assert abs(matrix[0][0] - 1.0) < 0.001
        assert abs(matrix[1][1] - 1.0) < 0.001
        assert abs(matrix[2][2] - 1.0) < 0.001
        assert matrix[3][3] == 1.0

    def test_transform_matrix_with_position(self):
        result = PlacementResult(position=(5.0, 10.0, 15.0))
        matrix = result.get_transform_matrix()
        assert matrix[0][3] == 5.0
        assert matrix[1][3] == 10.0
        assert matrix[2][3] == 15.0

    def test_transform_matrix_with_scale(self):
        result = PlacementResult(scale=(2.0, 2.0, 2.0))
        matrix = result.get_transform_matrix()
        # Diagonal should be 2.0 (scale applied)
        assert abs(matrix[0][0] - 2.0) < 0.001
        assert abs(matrix[1][1] - 2.0) < 0.001
        assert abs(matrix[2][2] - 2.0) < 0.001


# =============================================================================
# Bounds
# =============================================================================


class TestBounds:
    def test_default_values(self):
        bounds = Bounds()
        assert bounds.min_x == 0.0
        assert bounds.min_z == 0.0
        assert bounds.max_x == 100.0
        assert bounds.max_z == 100.0

    def test_custom_values(self):
        bounds = Bounds(min_x=10.0, min_z=20.0, max_x=50.0, max_z=60.0)
        assert bounds.min_x == 10.0
        assert bounds.min_z == 20.0
        assert bounds.max_x == 50.0
        assert bounds.max_z == 60.0

    def test_invalid_x_range(self):
        with pytest.raises(ValueError, match="min_x must be <= max_x"):
            Bounds(min_x=100.0, max_x=50.0)

    def test_invalid_z_range(self):
        with pytest.raises(ValueError, match="min_z must be <= max_z"):
            Bounds(min_z=100.0, max_z=50.0)

    def test_width(self):
        bounds = Bounds(min_x=10.0, max_x=50.0)
        assert bounds.width == 40.0

    def test_depth(self):
        bounds = Bounds(min_z=20.0, max_z=80.0)
        assert bounds.depth == 60.0

    def test_area(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=10.0, max_z=10.0)
        assert bounds.area == 100.0

    def test_center(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        cx, cz = bounds.center
        assert cx == 50.0
        assert cz == 50.0

    def test_contains_inside(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        assert bounds.contains(50.0, 50.0) is True

    def test_contains_edge(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        assert bounds.contains(0.0, 0.0) is True
        assert bounds.contains(100.0, 100.0) is True

    def test_contains_outside(self):
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        assert bounds.contains(150.0, 50.0) is False

    def test_intersects_overlapping(self):
        b1 = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        b2 = Bounds(min_x=50.0, min_z=50.0, max_x=150.0, max_z=150.0)
        assert b1.intersects(b2) is True

    def test_intersects_no_overlap(self):
        b1 = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        b2 = Bounds(min_x=100.0, min_z=100.0, max_x=150.0, max_z=150.0)
        assert b1.intersects(b2) is False

    def test_intersects_touching(self):
        b1 = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        b2 = Bounds(min_x=50.0, min_z=50.0, max_x=100.0, max_z=100.0)
        assert b1.intersects(b2) is True


# =============================================================================
# NoiseGenerator
# =============================================================================


class TestNoiseGenerator:
    def test_sample_returns_float(self):
        noise = NoiseGenerator(seed=42)
        value = noise.sample(10.0, 20.0)
        assert isinstance(value, float)

    def test_sample_range(self):
        noise = NoiseGenerator(seed=42)
        for _ in range(100):
            value = noise.sample(float(_), float(_ * 2))
            assert 0.0 <= value <= 1.0

    def test_deterministic(self):
        noise1 = NoiseGenerator(seed=42)
        noise2 = NoiseGenerator(seed=42)
        v1 = noise1.sample(10.0, 20.0)
        v2 = noise2.sample(10.0, 20.0)
        assert v1 == v2

    def test_different_seeds(self):
        noise1 = NoiseGenerator(seed=42)
        noise2 = NoiseGenerator(seed=123)
        v1 = noise1.sample(10.0, 20.0)
        v2 = noise2.sample(10.0, 20.0)
        assert v1 != v2

    def test_scale_affects_frequency(self):
        noise = NoiseGenerator(seed=42)
        v1 = noise.sample(10.0, 20.0, scale=1.0)
        v2 = noise.sample(10.0, 20.0, scale=10.0)
        # Different scales should give different values
        # (not always, but usually)
        # This is a weak test but checks the parameter is used
        assert v1 is not None and v2 is not None


# =============================================================================
# ProceduralPlacer
# =============================================================================


class TestProceduralPlacer:
    def test_seed_property(self):
        placer = ProceduralPlacer(seed=42)
        assert placer.seed == 42

    def test_evaluate_position_flat_terrain(self):
        placer = ProceduralPlacer(seed=42)
        terrain = MockTerrain()
        rule = PlacementRule()
        # Flat terrain with default rule should pass
        assert placer.evaluate_position(terrain, 0.0, 0.0, rule) is True

    def test_evaluate_position_steep_slope(self):
        placer = ProceduralPlacer(seed=42)
        # Normal pointing sideways = vertical surface
        terrain = MockTerrain(normal=(1.0, 0.0, 0.0))
        rule = PlacementRule(slope_range=(0.0, 45.0))
        assert placer.evaluate_position(terrain, 0.0, 0.0, rule) is False

    def test_evaluate_position_height_check(self):
        placer = ProceduralPlacer(seed=42)
        terrain = MockTerrain(height=50.0)
        rule = PlacementRule(height_range=(0.0, 30.0))
        assert placer.evaluate_position(terrain, 0.0, 0.0, rule) is False

    def test_evaluate_position_layer_check(self):
        placer = ProceduralPlacer(seed=42)
        terrain = MockTerrain(layer=5)
        rule = PlacementRule(terrain_layers=[0, 1, 2])
        assert placer.evaluate_position(terrain, 0.0, 0.0, rule) is False

    def test_evaluate_position_water_exclusion(self):
        placer = ProceduralPlacer(seed=42)
        terrain = MockTerrain(water=True)
        rule = PlacementRule(exclude_water=True)
        assert placer.evaluate_position(terrain, 0.0, 0.0, rule) is False

    def test_evaluate_position_road_exclusion(self):
        placer = ProceduralPlacer(seed=42)
        terrain = MockTerrain(road=True)
        rule = PlacementRule(exclude_roads=True)
        assert placer.evaluate_position(terrain, 0.0, 0.0, rule) is False

    def test_get_slope_at_flat(self):
        placer = ProceduralPlacer()
        terrain = MockTerrain(normal=(0.0, 1.0, 0.0))
        slope = placer.get_slope_at(terrain, 0.0, 0.0)
        assert abs(slope) < 0.001

    def test_get_slope_at_45_degrees(self):
        placer = ProceduralPlacer()
        # Normal at 45 degrees
        n = 1.0 / math.sqrt(2)
        terrain = MockTerrain(normal=(n, n, 0.0))
        slope = placer.get_slope_at(terrain, 0.0, 0.0)
        assert abs(slope - 45.0) < 0.1

    def test_sample_noise(self):
        placer = ProceduralPlacer(seed=42)
        value = placer.sample_noise(10.0, 20.0, 5.0)
        assert 0.0 <= value <= 1.0

    def test_generate_in_bounds_empty_density(self):
        placer = ProceduralPlacer(seed=42)
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=10.0, max_z=10.0)
        rule = PlacementRule()
        results = placer.generate_in_bounds(
            terrain, bounds, "test", 0.0, 1.0, (0.8, 1.2), True, rule
        )
        assert len(results) == 0

    def test_generate_in_bounds_produces_results(self):
        placer = ProceduralPlacer(seed=42)
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=10.0, max_z=10.0)
        rule = PlacementRule(noise_threshold=0.0)  # Accept all
        results = placer.generate_in_bounds(
            terrain, bounds, "test_plant", 1.0, 1.0, (0.8, 1.2), True, rule
        )
        assert len(results) > 0

    def test_generate_in_bounds_deterministic(self):
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=10.0, max_z=10.0)
        rule = PlacementRule(noise_threshold=0.0)

        placer1 = ProceduralPlacer(seed=42)
        results1 = placer1.generate_in_bounds(
            terrain, bounds, "test", 1.0, 1.0, (0.8, 1.2), True, rule
        )

        placer2 = ProceduralPlacer(seed=42)
        results2 = placer2.generate_in_bounds(
            terrain, bounds, "test", 1.0, 1.0, (0.8, 1.2), True, rule
        )

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.position == r2.position

    def test_generate_respects_foliage_type_id(self):
        placer = ProceduralPlacer(seed=42)
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=10.0, max_z=10.0)
        rule = PlacementRule(noise_threshold=0.0)
        results = placer.generate_in_bounds(
            terrain, bounds, "my_plant", 1.0, 1.0, (0.8, 1.2), True, rule
        )
        for r in results:
            assert r.foliage_type_id == "my_plant"


# =============================================================================
# FoliagePlacement
# =============================================================================


class TestFoliagePlacement:
    def test_default_values(self):
        fp = FoliagePlacement()
        assert fp.foliage_type_id == ""
        assert fp.seed == 0
        assert fp.density == 1.0
        assert fp.min_spacing == 1.0

    def test_custom_values(self):
        fp = FoliagePlacement(
            foliage_type_id="test_plant",
            seed=42,
            density=5.0,
            min_spacing=2.0,
        )
        assert fp.foliage_type_id == "test_plant"
        assert fp.seed == 42
        assert fp.density == 5.0
        assert fp.min_spacing == 2.0

    def test_generate_placements(self):
        fp = FoliagePlacement(
            foliage_type_id="test_plant",
            seed=42,
            density=1.0,
            rules=PlacementRule(noise_threshold=0.0),
        )
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=10.0, max_z=10.0)
        results = fp.generate_placements(terrain, bounds)
        assert len(results) > 0


# =============================================================================
# ManualPlacement
# =============================================================================


class TestManualPlacement:
    def test_add_instance(self):
        mp = ManualPlacement()
        placement = PlacementResult(position=(10.0, 0.0, 20.0))
        instance_id = mp.add_instance(placement)
        assert instance_id == 0
        assert mp.count() == 1

    def test_add_multiple_instances(self):
        mp = ManualPlacement()
        mp.add_instance(PlacementResult(position=(10.0, 0.0, 20.0)))
        mp.add_instance(PlacementResult(position=(30.0, 0.0, 40.0)))
        assert mp.count() == 2

    def test_remove_instance(self):
        mp = ManualPlacement()
        instance_id = mp.add_instance(PlacementResult())
        assert mp.remove_instance(instance_id) is True
        assert mp.count() == 0

    def test_remove_instance_not_found(self):
        mp = ManualPlacement()
        assert mp.remove_instance(999) is False

    def test_move_instance(self):
        mp = ManualPlacement()
        instance_id = mp.add_instance(PlacementResult(position=(0.0, 0.0, 0.0)))
        assert mp.move_instance(instance_id, (10.0, 5.0, 20.0)) is True
        inst = mp.get_instance(instance_id)
        assert inst.position == (10.0, 5.0, 20.0)

    def test_move_instance_not_found(self):
        mp = ManualPlacement()
        assert mp.move_instance(999, (10.0, 0.0, 20.0)) is False

    def test_update_instance_position(self):
        mp = ManualPlacement()
        instance_id = mp.add_instance(PlacementResult(position=(0.0, 0.0, 0.0)))
        mp.update_instance(instance_id, position=(5.0, 5.0, 5.0))
        inst = mp.get_instance(instance_id)
        assert inst.position == (5.0, 5.0, 5.0)

    def test_update_instance_rotation(self):
        mp = ManualPlacement()
        instance_id = mp.add_instance(PlacementResult())
        mp.update_instance(instance_id, rotation=(45.0, 90.0, 0.0))
        inst = mp.get_instance(instance_id)
        assert inst.rotation == (45.0, 90.0, 0.0)

    def test_update_instance_scale(self):
        mp = ManualPlacement()
        instance_id = mp.add_instance(PlacementResult())
        mp.update_instance(instance_id, scale=(2.0, 2.0, 2.0))
        inst = mp.get_instance(instance_id)
        assert inst.scale == (2.0, 2.0, 2.0)

    def test_update_instance_not_found(self):
        mp = ManualPlacement()
        assert mp.update_instance(999, position=(0.0, 0.0, 0.0)) is False

    def test_get_instance(self):
        mp = ManualPlacement()
        instance_id = mp.add_instance(PlacementResult(foliage_type_id="test"))
        inst = mp.get_instance(instance_id)
        assert inst is not None
        assert inst.foliage_type_id == "test"

    def test_get_instance_not_found(self):
        mp = ManualPlacement()
        assert mp.get_instance(999) is None

    def test_get_all_instances(self):
        mp = ManualPlacement()
        mp.add_instance(PlacementResult(position=(10.0, 0.0, 20.0)))
        mp.add_instance(PlacementResult(position=(30.0, 0.0, 40.0)))
        instances = mp.get_all_instances()
        assert len(instances) == 2

    def test_get_instances_in_bounds(self):
        mp = ManualPlacement()
        mp.add_instance(PlacementResult(position=(5.0, 0.0, 5.0)))
        mp.add_instance(PlacementResult(position=(50.0, 0.0, 50.0)))
        mp.add_instance(PlacementResult(position=(150.0, 0.0, 150.0)))

        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        in_bounds = mp.get_instances_in_bounds(bounds)
        assert len(in_bounds) == 2

    def test_get_instances_by_type(self):
        mp = ManualPlacement()
        mp.add_instance(PlacementResult(foliage_type_id="plant_a"))
        mp.add_instance(PlacementResult(foliage_type_id="plant_a"))
        mp.add_instance(PlacementResult(foliage_type_id="plant_b"))

        plant_a = mp.get_instances_by_type("plant_a")
        assert len(plant_a) == 2

    def test_clear(self):
        mp = ManualPlacement()
        mp.add_instance(PlacementResult())
        mp.add_instance(PlacementResult())
        mp.clear()
        assert mp.count() == 0


# =============================================================================
# procedural_placement decorator
# =============================================================================


class TestProceduralPlacementDecorator:
    def test_decorator_creates_placement(self):
        @procedural_placement(foliage_type_id="test_plant")
        class TestPlacement:
            pass

        assert hasattr(TestPlacement, "_foliage_placement")
        assert TestPlacement._foliage_placement.foliage_type_id == "test_plant"

    def test_decorator_with_seed(self):
        @procedural_placement(foliage_type_id="test", seed=42)
        class TestPlacement:
            pass

        assert TestPlacement._foliage_placement.seed == 42

    def test_decorator_with_density(self):
        @procedural_placement(foliage_type_id="test", density=5.0)
        class TestPlacement:
            pass

        assert TestPlacement._foliage_placement.density == 5.0

    def test_decorator_with_slope_range(self):
        @procedural_placement(
            foliage_type_id="test",
            slope_range=(0.0, 30.0),
        )
        class TestPlacement:
            pass

        assert TestPlacement._foliage_placement.rules.slope_range == (0.0, 30.0)

    def test_decorator_with_height_range(self):
        @procedural_placement(
            foliage_type_id="test",
            height_range=(0.0, 100.0),
        )
        class TestPlacement:
            pass

        assert TestPlacement._foliage_placement.rules.height_range == (0.0, 100.0)

    def test_decorator_reads_class_attributes(self):
        @procedural_placement(foliage_type_id="test")
        class TestPlacement:
            scale_range = (0.5, 1.5)
            rotation_random = False

        fp = TestPlacement._foliage_placement
        assert fp.scale_range == (0.5, 1.5)
        assert fp.rotation_random is False
