"""Tests for terrain special features."""

import math
from typing import List, Tuple

import pytest

from engine.world.terrain.features import (
    DeformationSettings,
    HitResult,
    PhysicalMaterialMapping,
    RiverSpline,
    RoadSpline,
    SplinePoint,
    TerrainCollision,
    TerrainDeformer,
    TerrainHole,
    TerrainHoleManager,
    TerrainSpline,
)


class MockHeightfield:
    """Mock heightfield for testing."""

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        sample_spacing: float = 1.0,
        initial_height: float = 10.0,
    ):
        self._width = width
        self._height = height
        self._sample_spacing = sample_spacing
        self._heights = [
            [initial_height for _ in range(width)] for _ in range(height)
        ]

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def sample_spacing(self) -> float:
        return self._sample_spacing

    def get_height_at(self, x: int, z: int) -> float:
        return self._heights[z][x]

    def set_height_at(self, x: int, z: int, height: float) -> None:
        self._heights[z][x] = height

    def world_to_sample(self, world_x: float, world_z: float) -> Tuple[int, int]:
        return int(world_x / self._sample_spacing), int(world_z / self._sample_spacing)

    def sample_to_world(self, sample_x: int, sample_z: int) -> Tuple[float, float]:
        return sample_x * self._sample_spacing, sample_z * self._sample_spacing


class MockWeightMap:
    """Mock weight map for testing."""

    def __init__(self, dominant_layer: int = 0):
        self._dominant_layer = dominant_layer

    def get_dominant_layer_at(self, x: int, z: int) -> int:
        return self._dominant_layer


# ============================================================================
# TerrainHole tests
# ============================================================================


class TestTerrainHole:
    """Tests for TerrainHole class."""

    def test_default_values(self):
        """Test default hole values."""
        hole = TerrainHole()
        assert hole.center_x == 0.0
        assert hole.center_z == 0.0
        assert hole.radius == 10.0
        assert hole.mask_resolution == 32

    def test_custom_values(self):
        """Test custom hole values."""
        hole = TerrainHole(
            center_x=50.0,
            center_z=50.0,
            radius=20.0,
            mask_resolution=64,
        )
        assert hole.center_x == 50.0
        assert hole.center_z == 50.0
        assert hole.radius == 20.0
        assert hole.mask_resolution == 64

    def test_invalid_radius(self):
        """Test that invalid radius raises error."""
        with pytest.raises(ValueError, match="radius must be > 0"):
            TerrainHole(radius=0)

        with pytest.raises(ValueError, match="radius must be > 0"):
            TerrainHole(radius=-5.0)

    def test_invalid_mask_resolution(self):
        """Test that invalid mask_resolution raises error."""
        with pytest.raises(ValueError, match="mask_resolution must be >= 2"):
            TerrainHole(mask_resolution=1)

    def test_is_visible_at_center(self):
        """Test visibility at hole center."""
        hole = TerrainHole(center_x=50.0, center_z=50.0, radius=10.0)

        # Center should not be visible (in hole)
        assert not hole.is_visible_at(50.0, 50.0)

    def test_is_visible_outside(self):
        """Test visibility outside hole."""
        hole = TerrainHole(center_x=50.0, center_z=50.0, radius=10.0)

        # Far outside should be visible
        assert hole.is_visible_at(100.0, 100.0)

    def test_is_visible_at_edge(self):
        """Test visibility at hole edge."""
        hole = TerrainHole(center_x=50.0, center_z=50.0, radius=10.0)

        # Just outside radius should be visible
        assert hole.is_visible_at(50.0 + 11.0, 50.0)

    def test_bounds_property(self):
        """Test bounds property."""
        hole = TerrainHole(center_x=50.0, center_z=50.0, radius=10.0)

        bounds = hole.bounds
        assert bounds == (40.0, 40.0, 60.0, 60.0)

    def test_set_custom_mask(self):
        """Test setting custom visibility mask."""
        hole = TerrainHole(mask_resolution=4)

        # Create a mask with a specific pattern
        custom_mask = [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
        hole.set_custom_mask(custom_mask)

    def test_set_custom_mask_wrong_size(self):
        """Test that wrong mask size raises error."""
        hole = TerrainHole(mask_resolution=4)

        wrong_mask = [[True, True], [True, True]]

        with pytest.raises(ValueError, match="mask height must be 4"):
            hole.set_custom_mask(wrong_mask)


# ============================================================================
# TerrainHoleManager tests
# ============================================================================


class TestTerrainHoleManager:
    """Tests for TerrainHoleManager class."""

    def test_initial_state(self):
        """Test initial manager state."""
        manager = TerrainHoleManager()
        assert manager.hole_count == 0
        assert len(manager.holes) == 0

    def test_add_hole(self):
        """Test adding a hole."""
        manager = TerrainHoleManager()
        hole = TerrainHole(center_x=50.0, center_z=50.0, radius=10.0)

        index = manager.add_hole(hole)

        assert index == 0
        assert manager.hole_count == 1

    def test_remove_hole(self):
        """Test removing a hole."""
        manager = TerrainHoleManager()
        manager.add_hole(TerrainHole(center_x=50.0, center_z=50.0))
        manager.add_hole(TerrainHole(center_x=100.0, center_z=100.0))

        manager.remove_hole(0)

        assert manager.hole_count == 1

    def test_remove_hole_invalid_index(self):
        """Test removing hole with invalid index."""
        manager = TerrainHoleManager()

        with pytest.raises(ValueError, match="index must be in range"):
            manager.remove_hole(0)

    def test_clear_holes(self):
        """Test clearing all holes."""
        manager = TerrainHoleManager()
        manager.add_hole(TerrainHole())
        manager.add_hole(TerrainHole())

        manager.clear_holes()

        assert manager.hole_count == 0

    def test_is_visible_at_no_holes(self):
        """Test visibility with no holes."""
        manager = TerrainHoleManager()

        assert manager.is_visible_at(50.0, 50.0)

    def test_is_visible_at_with_holes(self):
        """Test visibility with holes."""
        manager = TerrainHoleManager()
        manager.add_hole(TerrainHole(center_x=50.0, center_z=50.0, radius=10.0))

        # In hole
        assert not manager.is_visible_at(50.0, 50.0)
        # Outside hole
        assert manager.is_visible_at(100.0, 100.0)

    def test_is_visible_multiple_holes(self):
        """Test visibility with multiple holes."""
        manager = TerrainHoleManager()
        manager.add_hole(TerrainHole(center_x=50.0, center_z=50.0, radius=10.0))
        manager.add_hole(TerrainHole(center_x=100.0, center_z=100.0, radius=10.0))

        # In first hole
        assert not manager.is_visible_at(50.0, 50.0)
        # In second hole
        assert not manager.is_visible_at(100.0, 100.0)
        # Between holes
        assert manager.is_visible_at(75.0, 75.0)

    def test_get_visibility_mask(self):
        """Test getting visibility mask for a region."""
        manager = TerrainHoleManager()
        manager.add_hole(TerrainHole(center_x=50.0, center_z=50.0, radius=10.0))

        mask = manager.get_visibility_mask(0.0, 0.0, 100.0, 100.0, 10)

        assert len(mask) == 10
        assert len(mask[0]) == 10

    def test_get_visibility_mask_invalid_resolution(self):
        """Test that invalid resolution raises error."""
        manager = TerrainHoleManager()

        with pytest.raises(ValueError, match="resolution must be >= 1"):
            manager.get_visibility_mask(0.0, 0.0, 100.0, 100.0, 0)


# ============================================================================
# SplinePoint tests
# ============================================================================


class TestSplinePoint:
    """Tests for SplinePoint class."""

    def test_default_values(self):
        """Test default point values."""
        point = SplinePoint()
        assert point.position == (0.0, 0.0, 0.0)
        assert point.tangent == (1.0, 0.0, 0.0)
        assert point.width == 10.0

    def test_custom_values(self):
        """Test custom point values."""
        point = SplinePoint(
            position=(10.0, 5.0, 20.0),
            tangent=(0.0, 0.0, 1.0),
            width=15.0,
        )
        assert point.position == (10.0, 5.0, 20.0)
        assert point.tangent == (0.0, 0.0, 1.0)
        assert point.width == 15.0

    def test_invalid_width(self):
        """Test that invalid width raises error."""
        with pytest.raises(ValueError, match="width must be > 0"):
            SplinePoint(width=0)


# ============================================================================
# TerrainSpline tests
# ============================================================================


class TestTerrainSpline:
    """Tests for TerrainSpline class."""

    def test_empty_spline(self):
        """Test empty spline."""
        spline = TerrainSpline()
        assert spline.point_count == 0

    def test_add_point(self):
        """Test adding points."""
        spline = TerrainSpline()
        spline.add_point(SplinePoint(position=(0.0, 0.0, 0.0)))
        spline.add_point(SplinePoint(position=(10.0, 0.0, 10.0)))

        assert spline.point_count == 2

    def test_insert_point(self):
        """Test inserting points."""
        spline = TerrainSpline()
        spline.add_point(SplinePoint(position=(0.0, 0.0, 0.0)))
        spline.add_point(SplinePoint(position=(20.0, 0.0, 20.0)))

        spline.insert_point(1, SplinePoint(position=(10.0, 0.0, 10.0)))

        assert spline.point_count == 3
        assert spline.points[1].position == (10.0, 0.0, 10.0)

    def test_remove_point(self):
        """Test removing points."""
        spline = TerrainSpline()
        spline.add_point(SplinePoint(position=(0.0, 0.0, 0.0)))
        spline.add_point(SplinePoint(position=(10.0, 0.0, 10.0)))
        spline.add_point(SplinePoint(position=(20.0, 0.0, 20.0)))

        spline.remove_point(1)

        assert spline.point_count == 2

    def test_remove_point_invalid_index(self):
        """Test removing point with invalid index."""
        spline = TerrainSpline()
        spline.add_point(SplinePoint())

        with pytest.raises(ValueError, match="index must be in range"):
            spline.remove_point(5)

    def test_set_point(self):
        """Test setting a point."""
        spline = TerrainSpline()
        spline.add_point(SplinePoint(position=(0.0, 0.0, 0.0)))

        new_point = SplinePoint(position=(5.0, 5.0, 5.0))
        spline.set_point(0, new_point)

        assert spline.points[0].position == (5.0, 5.0, 5.0)

    def test_evaluate_requires_minimum_points(self):
        """Test that evaluate requires at least 2 points."""
        spline = TerrainSpline()
        spline.add_point(SplinePoint())

        with pytest.raises(ValueError, match="Spline must have at least 2 points"):
            spline.evaluate(0.5)

    def test_evaluate_at_endpoints(self):
        """Test evaluation at spline endpoints."""
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(100.0, 50.0, 100.0)),
        ])

        start = spline.evaluate(0.0)
        end = spline.evaluate(1.0)

        # Start should be near first point
        assert abs(start[0] - 0.0) < 0.1
        # End should be near last point
        assert abs(end[0] - 100.0) < 0.1

    def test_evaluate_midpoint(self):
        """Test evaluation at midpoint."""
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(100.0, 0.0, 0.0)),
        ])

        mid = spline.evaluate(0.5)

        # Midpoint should be near 50.0
        assert 40.0 < mid[0] < 60.0

    def test_evaluate_tangent(self):
        """Test tangent evaluation."""
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(100.0, 0.0, 0.0)),
        ])

        tangent = spline.evaluate_tangent(0.5)

        # Tangent should point in X direction
        assert tangent[0] > 0.9
        length = math.sqrt(sum(t * t for t in tangent))
        assert abs(length - 1.0) < 0.001

    def test_evaluate_width(self):
        """Test width interpolation."""
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0), width=10.0),
            SplinePoint(position=(100.0, 0.0, 0.0), width=20.0),
        ])

        width_start = spline.evaluate_width(0.0)
        width_mid = spline.evaluate_width(0.5)
        width_end = spline.evaluate_width(1.0)

        assert abs(width_start - 10.0) < 0.1
        assert abs(width_end - 20.0) < 0.1
        assert 14.0 < width_mid < 16.0

    def test_get_length(self):
        """Test length calculation."""
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(100.0, 0.0, 0.0)),
        ])

        length = spline.get_length()

        # Should be approximately 100
        assert 95.0 < length < 105.0

    def test_get_point_at_distance(self):
        """Test getting point at distance."""
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(100.0, 0.0, 0.0)),
        ])

        point = spline.get_point_at_distance(50.0)

        # Should be near midpoint
        assert 40.0 < point[0] < 60.0

    def test_points_property_returns_copy(self):
        """Test that points property returns a copy."""
        spline = TerrainSpline()
        spline.add_point(SplinePoint())

        points = spline.points
        points.append(SplinePoint())

        assert spline.point_count == 1


# ============================================================================
# RoadSpline tests
# ============================================================================


class TestRoadSpline:
    """Tests for RoadSpline class."""

    def test_default_values(self):
        """Test default road values."""
        road = RoadSpline()
        assert road.surface_material == "mat_asphalt"
        assert road.bank_angle == 5.0

    def test_invalid_bank_angle(self):
        """Test that negative bank_angle raises error."""
        with pytest.raises(ValueError, match="bank_angle must be >= 0"):
            RoadSpline(bank_angle=-5.0)

    def test_deform_terrain(self):
        """Test terrain deformation for road."""
        road = RoadSpline([
            SplinePoint(position=(10.0, 10.0, 32.0), width=10.0),
            SplinePoint(position=(54.0, 10.0, 32.0), width=10.0),
        ])

        heightfield = MockHeightfield(width=64, height=64, initial_height=15.0)

        road.deform_terrain(heightfield, depth=0.1, blend_width=5.0)

        # Road area should be lower
        assert heightfield.get_height_at(32, 32) < 15.0

    def test_deform_terrain_invalid_depth(self):
        """Test that invalid depth raises error."""
        road = RoadSpline([
            SplinePoint(position=(0.0, 10.0, 0.0)),
            SplinePoint(position=(100.0, 10.0, 0.0)),
        ])

        heightfield = MockHeightfield()

        with pytest.raises(ValueError, match="depth must be >= 0"):
            road.deform_terrain(heightfield, depth=-1.0)

    def test_deform_terrain_invalid_blend_width(self):
        """Test that invalid blend_width raises error."""
        road = RoadSpline([
            SplinePoint(position=(0.0, 10.0, 0.0)),
            SplinePoint(position=(100.0, 10.0, 0.0)),
        ])

        heightfield = MockHeightfield()

        with pytest.raises(ValueError, match="blend_width must be >= 0"):
            road.deform_terrain(heightfield, blend_width=-1.0)

    def test_get_surface_height_at_on_road(self):
        """Test getting surface height on road."""
        road = RoadSpline([
            SplinePoint(position=(0.0, 10.0, 50.0), width=20.0),
            SplinePoint(position=(100.0, 10.0, 50.0), width=20.0),
        ])

        heightfield = MockHeightfield(width=128, height=128)

        height = road.get_surface_height_at(50.0, 50.0, heightfield)

        assert height is not None
        assert abs(height - 10.0) < 1.0

    def test_get_surface_height_at_off_road(self):
        """Test getting surface height off road."""
        road = RoadSpline([
            SplinePoint(position=(0.0, 10.0, 50.0), width=10.0),
            SplinePoint(position=(100.0, 10.0, 50.0), width=10.0),
        ])

        heightfield = MockHeightfield()

        height = road.get_surface_height_at(50.0, 100.0, heightfield)

        assert height is None


# ============================================================================
# RiverSpline tests
# ============================================================================


class TestRiverSpline:
    """Tests for RiverSpline class."""

    def test_default_values(self):
        """Test default river values."""
        river = RiverSpline()
        assert river.water_level == 0.0
        assert river.flow_speed == 1.0

    def test_invalid_flow_speed(self):
        """Test that negative flow_speed raises error."""
        with pytest.raises(ValueError, match="flow_speed must be >= 0"):
            RiverSpline(flow_speed=-1.0)

    def test_carve_channel(self):
        """Test channel carving."""
        river = RiverSpline([
            SplinePoint(position=(10.0, 5.0, 32.0), width=10.0),
            SplinePoint(position=(54.0, 5.0, 32.0), width=10.0),
        ], water_level=5.0)

        heightfield = MockHeightfield(width=64, height=64, initial_height=10.0)

        river.carve_channel(heightfield, depth=2.0, bank_slope=45.0)

        # River area should be carved out
        center_height = heightfield.get_height_at(32, 32)
        assert center_height < 10.0

    def test_carve_channel_invalid_depth(self):
        """Test that invalid depth raises error."""
        river = RiverSpline([
            SplinePoint(position=(0.0, 5.0, 0.0)),
            SplinePoint(position=(100.0, 5.0, 0.0)),
        ])

        heightfield = MockHeightfield()

        with pytest.raises(ValueError, match="depth must be >= 0"):
            river.carve_channel(heightfield, depth=-1.0)

    def test_carve_channel_invalid_bank_slope(self):
        """Test that invalid bank_slope raises error."""
        river = RiverSpline([
            SplinePoint(position=(0.0, 5.0, 0.0)),
            SplinePoint(position=(100.0, 5.0, 0.0)),
        ])

        heightfield = MockHeightfield()

        with pytest.raises(ValueError, match="bank_slope must be in range"):
            river.carve_channel(heightfield, bank_slope=0)

        with pytest.raises(ValueError, match="bank_slope must be in range"):
            river.carve_channel(heightfield, bank_slope=91.0)

    def test_get_flow_direction_in_river(self):
        """Test getting flow direction in river."""
        river = RiverSpline([
            SplinePoint(position=(0.0, 5.0, 50.0), width=20.0),
            SplinePoint(position=(100.0, 5.0, 50.0), width=20.0),
        ], flow_speed=2.0)

        flow = river.get_flow_direction_at(50.0, 50.0)

        assert flow is not None
        assert flow[0] > 0  # Flow should be in positive X direction

    def test_get_flow_direction_outside_river(self):
        """Test getting flow direction outside river."""
        river = RiverSpline([
            SplinePoint(position=(0.0, 5.0, 50.0), width=10.0),
            SplinePoint(position=(100.0, 5.0, 50.0), width=10.0),
        ])

        flow = river.get_flow_direction_at(50.0, 100.0)

        assert flow is None


# ============================================================================
# DeformationSettings tests
# ============================================================================


class TestDeformationSettings:
    """Tests for DeformationSettings class."""

    def test_default_values(self):
        """Test default settings values."""
        settings = DeformationSettings()
        assert settings.blend_mode == "replace"
        assert settings.smoothing_passes == 0
        assert settings.smoothing_strength == 0.5

    def test_invalid_blend_mode(self):
        """Test that invalid blend_mode raises error."""
        with pytest.raises(ValueError, match="blend_mode must be one of"):
            DeformationSettings(blend_mode="invalid")

    def test_invalid_smoothing_passes(self):
        """Test that negative smoothing_passes raises error."""
        with pytest.raises(ValueError, match="smoothing_passes must be >= 0"):
            DeformationSettings(smoothing_passes=-1)

    def test_invalid_smoothing_strength(self):
        """Test that invalid smoothing_strength raises error."""
        with pytest.raises(ValueError, match="smoothing_strength must be in range"):
            DeformationSettings(smoothing_strength=1.5)


# ============================================================================
# TerrainDeformer tests
# ============================================================================


class TestTerrainDeformer:
    """Tests for TerrainDeformer class."""

    def test_apply_spline_deformation_road(self):
        """Test applying road spline deformation."""
        deformer = TerrainDeformer()
        road = RoadSpline([
            SplinePoint(position=(10.0, 10.0, 32.0)),
            SplinePoint(position=(54.0, 10.0, 32.0)),
        ])
        heightfield = MockHeightfield(width=64, height=64, initial_height=15.0)
        settings = DeformationSettings()

        deformer.apply_spline_deformation(heightfield, road, settings)

        # Road area should be modified
        assert heightfield.get_height_at(32, 32) != 15.0

    def test_apply_spline_deformation_river(self):
        """Test applying river spline deformation."""
        deformer = TerrainDeformer()
        river = RiverSpline([
            SplinePoint(position=(10.0, 5.0, 32.0)),
            SplinePoint(position=(54.0, 5.0, 32.0)),
        ])
        heightfield = MockHeightfield(width=64, height=64, initial_height=10.0)
        settings = DeformationSettings()

        deformer.apply_spline_deformation(heightfield, river, settings)

        # River area should be carved
        assert heightfield.get_height_at(32, 32) < 10.0

    def test_create_ramp(self):
        """Test creating a ramp."""
        deformer = TerrainDeformer()
        heightfield = MockHeightfield(width=64, height=64, initial_height=0.0)

        deformer.create_ramp(
            heightfield,
            start=(10.0, 32.0),
            end=(54.0, 32.0),
            width=10.0,
            start_height=0.0,
            end_height=20.0,
        )

        # Heights should increase along ramp
        h_start = heightfield.get_height_at(10, 32)
        h_end = heightfield.get_height_at(54, 32)

        assert h_start < h_end

    def test_create_ramp_invalid_width(self):
        """Test that invalid ramp width raises error."""
        deformer = TerrainDeformer()
        heightfield = MockHeightfield()

        with pytest.raises(ValueError, match="width must be > 0"):
            deformer.create_ramp(
                heightfield,
                start=(0.0, 0.0),
                end=(100.0, 0.0),
                width=0,
            )

    def test_create_plateau(self):
        """Test creating a plateau."""
        deformer = TerrainDeformer()
        heightfield = MockHeightfield(width=64, height=64, initial_height=0.0)

        deformer.create_plateau(
            heightfield,
            center_x=32.0,
            center_z=32.0,
            radius=10.0,
            height=20.0,
            blend_width=5.0,
        )

        # Center should be at plateau height
        assert abs(heightfield.get_height_at(32, 32) - 20.0) < 0.1

    def test_create_plateau_invalid_radius(self):
        """Test that invalid plateau radius raises error."""
        deformer = TerrainDeformer()
        heightfield = MockHeightfield()

        with pytest.raises(ValueError, match="radius must be > 0"):
            deformer.create_plateau(heightfield, 32.0, 32.0, 0, 20.0)

    def test_create_plateau_invalid_blend_width(self):
        """Test that invalid blend_width raises error."""
        deformer = TerrainDeformer()
        heightfield = MockHeightfield()

        with pytest.raises(ValueError, match="blend_width must be >= 0"):
            deformer.create_plateau(heightfield, 32.0, 32.0, 10.0, 20.0, -1.0)


# ============================================================================
# PhysicalMaterialMapping tests
# ============================================================================


class TestPhysicalMaterialMapping:
    """Tests for PhysicalMaterialMapping class."""

    def test_default_material(self):
        """Test default material."""
        mapping = PhysicalMaterialMapping()
        assert mapping.default_material == "default"

    def test_set_default_material(self):
        """Test setting default material."""
        mapping = PhysicalMaterialMapping()
        mapping.default_material = "concrete"
        assert mapping.default_material == "concrete"

    def test_set_mapping(self):
        """Test setting material mapping."""
        mapping = PhysicalMaterialMapping()
        mapping.set_mapping(0, "grass")
        mapping.set_mapping(1, "rock")

        assert mapping.get_material_for_layer(0) == "grass"
        assert mapping.get_material_for_layer(1) == "rock"

    def test_set_mapping_invalid_layer_index(self):
        """Test that negative layer index raises error."""
        mapping = PhysicalMaterialMapping()

        with pytest.raises(ValueError, match="layer_index must be >= 0"):
            mapping.set_mapping(-1, "grass")

    def test_get_material_for_unmapped_layer(self):
        """Test getting material for unmapped layer returns default."""
        mapping = PhysicalMaterialMapping()

        assert mapping.get_material_for_layer(99) == "default"

    def test_get_material_at(self):
        """Test getting material at terrain position."""
        mapping = PhysicalMaterialMapping()
        mapping.set_mapping(0, "grass")
        mapping.set_mapping(1, "rock")

        weight_map = MockWeightMap(dominant_layer=1)

        material = mapping.get_material_at(weight_map, 0, 0)
        assert material == "rock"

    def test_clear_mappings(self):
        """Test clearing mappings."""
        mapping = PhysicalMaterialMapping()
        mapping.set_mapping(0, "grass")

        mapping.clear_mappings()

        assert mapping.get_material_for_layer(0) == "default"


# ============================================================================
# HitResult tests
# ============================================================================


class TestHitResult:
    """Tests for HitResult class."""

    def test_default_values(self):
        """Test default hit result values."""
        result = HitResult()
        assert not result.hit
        assert result.position == (0.0, 0.0, 0.0)
        assert result.normal == (0.0, 1.0, 0.0)
        assert result.distance == 0.0
        assert result.material == "default"


# ============================================================================
# TerrainCollision tests
# ============================================================================


class TestTerrainCollision:
    """Tests for TerrainCollision class."""

    def test_get_height_at(self):
        """Test getting height at position."""
        heightfield = MockHeightfield(width=64, height=64, initial_height=10.0)
        collision = TerrainCollision(heightfield)

        height = collision.get_height_at(32.0, 32.0)

        assert abs(height - 10.0) < 0.1

    def test_get_height_at_with_holes(self):
        """Test getting height at position with holes."""
        heightfield = MockHeightfield(width=64, height=64, initial_height=10.0)
        hole_manager = TerrainHoleManager()
        hole_manager.add_hole(TerrainHole(center_x=32.0, center_z=32.0, radius=5.0))

        collision = TerrainCollision(heightfield, hole_manager)

        # In hole - should return None
        height = collision.get_height_at(32.0, 32.0)
        assert height is None

        # Outside hole - should return height
        height_outside = collision.get_height_at(50.0, 50.0)
        assert height_outside is not None

    def test_get_normal_at(self):
        """Test getting surface normal."""
        heightfield = MockHeightfield(width=64, height=64, initial_height=10.0)
        collision = TerrainCollision(heightfield)

        normal = collision.get_normal_at(32.0, 32.0)

        # Flat terrain should have upward normal
        assert normal[1] > 0.9

    def test_get_physical_material_at(self):
        """Test getting physical material at position."""
        heightfield = MockHeightfield(width=64, height=64)
        weight_map = MockWeightMap(dominant_layer=1)
        material_mapping = PhysicalMaterialMapping()
        material_mapping.set_mapping(1, "rock")

        collision = TerrainCollision(
            heightfield,
            material_mapping=material_mapping,
            weight_map=weight_map,
        )

        material = collision.get_physical_material_at(32.0, 32.0)
        assert material == "rock"

    def test_raycast_hit(self):
        """Test raycast that hits terrain."""
        heightfield = MockHeightfield(width=64, height=64, initial_height=10.0)
        collision = TerrainCollision(heightfield)

        # Ray from above pointing down
        result = collision.raycast(
            origin_x=32.0,
            origin_y=50.0,
            origin_z=32.0,
            direction_x=0.0,
            direction_y=-1.0,
            direction_z=0.0,
        )

        assert result.hit
        assert abs(result.position[1] - 10.0) < 1.0

    def test_raycast_miss(self):
        """Test raycast that misses terrain."""
        heightfield = MockHeightfield(width=64, height=64, initial_height=10.0)
        collision = TerrainCollision(heightfield)

        # Ray pointing up
        result = collision.raycast(
            origin_x=32.0,
            origin_y=50.0,
            origin_z=32.0,
            direction_x=0.0,
            direction_y=1.0,
            direction_z=0.0,
        )

        assert not result.hit

    def test_sphere_cast(self):
        """Test sphere cast."""
        heightfield = MockHeightfield(width=64, height=64, initial_height=10.0)
        collision = TerrainCollision(heightfield)

        result = collision.sphere_cast(
            origin_x=32.0,
            origin_y=50.0,
            origin_z=32.0,
            radius=5.0,
            direction_x=0.0,
            direction_y=-1.0,
            direction_z=0.0,
        )

        assert result.hit
        # Hit position should account for sphere radius
        assert result.position[1] > 10.0

    def test_sphere_cast_invalid_radius(self):
        """Test that invalid radius raises error."""
        heightfield = MockHeightfield()
        collision = TerrainCollision(heightfield)

        with pytest.raises(ValueError, match="radius must be > 0"):
            collision.sphere_cast(32.0, 50.0, 32.0, 0, 0.0, -1.0, 0.0)


# ============================================================================
# Integration tests
# ============================================================================


class TestFeaturesIntegration:
    """Integration tests for terrain features."""

    def test_full_terrain_setup(self):
        """Test complete terrain feature setup."""
        heightfield = MockHeightfield(width=128, height=128, initial_height=20.0)

        # Add holes
        hole_manager = TerrainHoleManager()
        hole_manager.add_hole(TerrainHole(center_x=100.0, center_z=64.0, radius=10.0))

        # Add a road
        road = RoadSpline([
            SplinePoint(position=(10.0, 18.0, 64.0), width=8.0),
            SplinePoint(position=(64.0, 18.0, 64.0), width=8.0),
            SplinePoint(position=(118.0, 18.0, 64.0), width=8.0),
        ])
        road.deform_terrain(heightfield, depth=0.2, blend_width=3.0)

        # Add a river
        river = RiverSpline([
            SplinePoint(position=(64.0, 15.0, 10.0), width=6.0),
            SplinePoint(position=(64.0, 15.0, 54.0), width=8.0),
            SplinePoint(position=(64.0, 15.0, 118.0), width=10.0),
        ], water_level=16.0)
        river.carve_channel(heightfield, depth=3.0, bank_slope=45.0)

        # Set up materials
        weight_map = MockWeightMap(dominant_layer=0)
        material_mapping = PhysicalMaterialMapping()
        material_mapping.set_mapping(0, "grass")
        material_mapping.set_mapping(1, "asphalt")
        material_mapping.set_mapping(2, "mud")

        # Create collision
        collision = TerrainCollision(
            heightfield,
            hole_manager,
            material_mapping,
            weight_map,
        )

        # Test collision queries
        # On road
        h_road = collision.get_height_at(32.0, 64.0)
        assert h_road is not None

        # In hole
        h_hole = collision.get_height_at(100.0, 64.0)
        assert h_hole is None

        # Raycast
        result = collision.raycast(64.0, 100.0, 32.0, 0.0, -1.0, 0.0)
        assert result.hit


# ============================================================================
# Enhanced spline tests for edge cases
# ============================================================================


class TestEnhancedSplineValidation:
    """Enhanced tests for spline edge cases and continuity."""

    def test_tangent_at_exact_endpoints(self):
        """Verify tangent is valid at exact endpoints t=0 and t=1."""
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(100.0, 0.0, 0.0)),
        ])

        tangent_start = spline.evaluate_tangent(0.0)
        tangent_end = spline.evaluate_tangent(1.0)

        # Both should be valid unit vectors
        length_start = math.sqrt(sum(t * t for t in tangent_start))
        length_end = math.sqrt(sum(t * t for t in tangent_end))

        assert abs(length_start - 1.0) < 0.001, f"Start tangent not normalized: {length_start}"
        assert abs(length_end - 1.0) < 0.001, f"End tangent not normalized: {length_end}"

        # Both should point in positive X direction
        assert tangent_start[0] > 0.9, f"Start tangent wrong direction: {tangent_start}"
        assert tangent_end[0] > 0.9, f"End tangent wrong direction: {tangent_end}"

    def test_spline_c1_continuity(self):
        """Verify spline maintains C1 continuity (continuous first derivative)."""
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(50.0, 20.0, 0.0)),
            SplinePoint(position=(100.0, 0.0, 0.0)),
        ])

        # Sample tangents at close intervals
        t_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        tangents = [spline.evaluate_tangent(t) for t in t_values]

        # Verify tangents change smoothly (no sudden jumps)
        for i in range(len(tangents) - 1):
            t1 = tangents[i]
            t2 = tangents[i + 1]

            # Calculate angle change
            dot = t1[0] * t2[0] + t1[1] * t2[1] + t1[2] * t2[2]
            dot = max(-1.0, min(1.0, dot))
            angle_change = math.acos(dot)

            # Angle change should be small for smooth curves
            assert angle_change < 0.5, f"Large tangent jump at t={t_values[i]}: {math.degrees(angle_change)} degrees"

    def test_short_spline_length_calculation(self):
        """Verify length calculation for very short splines."""
        # Create a very short spline
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(0.1, 0.0, 0.0)),
        ])

        length = spline.get_length()

        # Should be approximately 0.1
        assert 0.05 < length < 0.15, f"Expected ~0.1, got {length}"

    def test_degenerate_spline_tangent(self):
        """Verify tangent handling for near-degenerate spline."""
        # Create spline with very close points
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(0.0001, 0.0, 0.0001)),
        ])

        tangent = spline.evaluate_tangent(0.5)

        # Should return a valid unit vector (fallback behavior)
        length = math.sqrt(sum(t * t for t in tangent))
        assert abs(length - 1.0) < 0.001, f"Tangent not normalized: {length}"

    def test_get_point_at_distance_accuracy(self):
        """Verify point_at_distance returns accurate position."""
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(100.0, 0.0, 0.0)),
        ])

        total_length = spline.get_length()
        midpoint_by_distance = spline.get_point_at_distance(total_length / 2)
        midpoint_by_t = spline.evaluate(0.5)

        # Should be very close to evaluate(0.5)
        dx = abs(midpoint_by_distance[0] - midpoint_by_t[0])
        dy = abs(midpoint_by_distance[1] - midpoint_by_t[1])
        dz = abs(midpoint_by_distance[2] - midpoint_by_t[2])

        assert dx < 5.0, f"X difference too large: {dx}"
        assert dy < 5.0, f"Y difference too large: {dy}"
        assert dz < 5.0, f"Z difference too large: {dz}"

    def test_curved_spline_length(self):
        """Verify length calculation for curved splines."""
        # Create a curved spline
        spline = TerrainSpline([
            SplinePoint(position=(0.0, 0.0, 0.0)),
            SplinePoint(position=(50.0, 50.0, 0.0)),
            SplinePoint(position=(100.0, 0.0, 0.0)),
        ])

        length = spline.get_length()

        # Curved length should be more than straight-line distance (100)
        assert length > 100.0, f"Curved length should exceed straight-line: {length}"
        # But shouldn't be absurdly long
        assert length < 200.0, f"Length seems too long: {length}"
