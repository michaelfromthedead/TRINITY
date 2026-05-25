"""
Tests for the placement module.

Tests all placement modes, constraints, and settings.
"""

import math
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.leveleditor.placement import (
    PlacementMode,
    PlacementTool,
    PlacementResult,
    ScatterSettings,
    FoliageSettings,
    SplineSettings,
    BrushSettings,
    SplinePoint,
    AxisConstraint,
    SurfaceAlignment,
    ScatterPattern,
    Vector3,
    Quaternion,
    Transform,
)
from foundation.tracker import tracker


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset tracker state before each test."""
    tracker._dirty.clear()
    tracker._cb_global.clear()
    tracker._cb_type.clear()
    tracker._cb_obj.clear()
    tracker._undo.clear()
    tracker._redo.clear()
    tracker._txn = None
    yield


class TestVector3:
    """Tests for Vector3 class."""

    def test_creation_default(self):
        """Vector3 default should be origin."""
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_creation_with_values(self):
        """Vector3 should store provided values."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_addition(self):
        """Vector3 addition should work correctly."""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1 + v2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_subtraction(self):
        """Vector3 subtraction should work correctly."""
        v1 = Vector3(4, 5, 6)
        v2 = Vector3(1, 2, 3)
        result = v1 - v2
        assert result.x == 3
        assert result.y == 3
        assert result.z == 3

    def test_scalar_multiplication(self):
        """Vector3 scalar multiplication should work correctly."""
        v = Vector3(1, 2, 3)
        result = v * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_length(self):
        """Vector3 length should calculate correctly."""
        v = Vector3(3, 4, 0)
        assert v.length() == 5.0

    def test_normalized(self):
        """Vector3 normalized should return unit vector."""
        v = Vector3(3, 0, 4)
        n = v.normalized()
        assert abs(n.length() - 1.0) < 0.0001

    def test_normalized_zero_vector(self):
        """Zero vector normalized should return zero vector."""
        v = Vector3(0, 0, 0)
        n = v.normalized()
        assert n.x == 0
        assert n.y == 0
        assert n.z == 0

    def test_dot_product(self):
        """Dot product should calculate correctly."""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        assert v1.dot(v2) == 32

    def test_cross_product(self):
        """Cross product should calculate correctly."""
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        result = v1.cross(v2)
        assert result.x == 0
        assert result.y == 0
        assert result.z == 1


class TestQuaternion:
    """Tests for Quaternion class."""

    def test_identity(self):
        """Identity quaternion should have w=1, x=y=z=0."""
        q = Quaternion.identity()
        assert q.w == 1
        assert q.x == 0
        assert q.y == 0
        assert q.z == 0

    def test_from_axis_angle(self):
        """Quaternion from axis-angle should be normalized."""
        axis = Vector3(0, 1, 0)
        angle = math.pi / 2
        q = Quaternion.from_axis_angle(axis, angle)
        length = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        assert abs(length - 1.0) < 0.0001

    def test_from_euler(self):
        """Quaternion from Euler angles should create valid rotation."""
        q = Quaternion.from_euler(0, math.pi/2, 0)
        length = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        assert abs(length - 1.0) < 0.0001


class TestPlacementTool:
    """Tests for PlacementTool class."""

    def test_creation(self):
        """PlacementTool should initialize with default settings."""
        tool = PlacementTool()
        assert tool.mode == PlacementMode.SINGLE
        assert tool.axis_constraint == AxisConstraint.NONE
        assert tool.surface_alignment == SurfaceAlignment.NONE

    def test_mode_change(self):
        """Mode change should trigger callback."""
        tool = PlacementTool()
        modes_changed = []

        def on_mode_change(old, new):
            modes_changed.append((old, new))

        tool.on("on_mode_change", on_mode_change)
        tool.mode = PlacementMode.PAINT_BRUSH

        assert len(modes_changed) == 1
        assert modes_changed[0] == (PlacementMode.SINGLE, PlacementMode.PAINT_BRUSH)

    def test_axis_constraint_x(self):
        """X axis constraint should only allow X movement."""
        tool = PlacementTool()
        tool.axis_constraint = AxisConstraint.X
        pos = Vector3(5, 10, 15)
        ref = Vector3(0, 0, 0)
        result = tool.apply_axis_constraint(pos, ref)
        assert result.x == 5
        assert result.y == 0
        assert result.z == 0

    def test_axis_constraint_y(self):
        """Y axis constraint should only allow Y movement."""
        tool = PlacementTool()
        tool.axis_constraint = AxisConstraint.Y
        pos = Vector3(5, 10, 15)
        ref = Vector3(0, 0, 0)
        result = tool.apply_axis_constraint(pos, ref)
        assert result.x == 0
        assert result.y == 10
        assert result.z == 0

    def test_axis_constraint_z(self):
        """Z axis constraint should only allow Z movement."""
        tool = PlacementTool()
        tool.axis_constraint = AxisConstraint.Z
        pos = Vector3(5, 10, 15)
        ref = Vector3(0, 0, 0)
        result = tool.apply_axis_constraint(pos, ref)
        assert result.x == 0
        assert result.y == 0
        assert result.z == 15

    def test_axis_constraint_xy(self):
        """XY plane constraint should allow X and Y movement."""
        tool = PlacementTool()
        tool.axis_constraint = AxisConstraint.XY
        pos = Vector3(5, 10, 15)
        ref = Vector3(0, 0, 0)
        result = tool.apply_axis_constraint(pos, ref)
        assert result.x == 5
        assert result.y == 10
        assert result.z == 0

    def test_axis_constraint_xz(self):
        """XZ plane constraint should allow X and Z movement."""
        tool = PlacementTool()
        tool.axis_constraint = AxisConstraint.XZ
        pos = Vector3(5, 10, 15)
        ref = Vector3(0, 0, 0)
        result = tool.apply_axis_constraint(pos, ref)
        assert result.x == 5
        assert result.y == 0
        assert result.z == 15

    def test_axis_constraint_yz(self):
        """YZ plane constraint should allow Y and Z movement."""
        tool = PlacementTool()
        tool.axis_constraint = AxisConstraint.YZ
        pos = Vector3(5, 10, 15)
        ref = Vector3(0, 0, 0)
        result = tool.apply_axis_constraint(pos, ref)
        assert result.x == 0
        assert result.y == 10
        assert result.z == 15

    def test_place_single(self):
        """Single placement should create one object."""
        tool = PlacementTool()
        pos = Vector3(10, 5, 20)
        result = tool.place_single(pos)

        assert result.success is True
        assert result.object_id is not None
        assert result.position.x == 10
        assert result.position.y == 5
        assert result.position.z == 20

    def test_place_single_with_rotation(self):
        """Single placement with rotation should use provided rotation."""
        tool = PlacementTool()
        pos = Vector3(0, 0, 0)
        rot = Quaternion.from_axis_angle(Vector3(0, 1, 0), math.pi / 4)
        result = tool.place_single(pos, rotation=rot)

        assert result.success is True
        assert result.rotation.y == rot.y

    def test_place_single_with_scale(self):
        """Single placement with scale should use provided scale."""
        tool = PlacementTool()
        pos = Vector3(0, 0, 0)
        scale = Vector3(2, 2, 2)
        result = tool.place_single(pos, scale=scale)

        assert result.success is True
        assert result.scale.x == 2
        assert result.scale.y == 2
        assert result.scale.z == 2

    def test_place_single_callback(self):
        """Single placement should trigger on_place callback."""
        tool = PlacementTool()
        placements = []

        def on_place(result):
            placements.append(result)

        tool.on("on_place", on_place)
        tool.place_single(Vector3(0, 0, 0))

        assert len(placements) == 1

    def test_place_single_undo(self):
        """Single placement should be undoable."""
        tool = PlacementTool()
        tool.place_single(Vector3(0, 0, 0))

        assert tracker.can_undo

    def test_brush_settings_default(self):
        """BrushSettings should have sensible defaults."""
        settings = BrushSettings()
        assert settings.radius == 5.0
        assert settings.density == 1.0
        assert settings.random_rotation is True

    def test_place_with_brush(self):
        """Brush placement should create multiple objects."""
        tool = PlacementTool()
        tool.brush_settings.radius = 10.0
        tool.brush_settings.density = 0.5
        results = tool.place_with_brush(Vector3(0, 0, 0))

        assert len(results) >= 1
        for r in results:
            assert r.success is True

    def test_brush_placement_within_radius(self):
        """All brush placements should be within brush radius."""
        tool = PlacementTool()
        tool.brush_settings.radius = 5.0
        tool.brush_settings.density = 2.0
        center = Vector3(10, 0, 10)
        results = tool.place_with_brush(center)

        for r in results:
            dx = r.position.x - center.x
            dz = r.position.z - center.z
            dist = math.sqrt(dx*dx + dz*dz)
            assert dist <= tool.brush_settings.radius

    def test_scatter_settings_default(self):
        """ScatterSettings should have sensible defaults."""
        settings = ScatterSettings()
        assert settings.count == 10
        assert settings.pattern == ScatterPattern.RANDOM

    def test_place_scatter(self):
        """Scatter placement should create specified count of objects."""
        tool = PlacementTool()
        tool.scatter_settings.count = 5
        tool.scatter_settings.region_min = Vector3(0, 0, 0)
        tool.scatter_settings.region_max = Vector3(10, 0, 10)
        results = tool.place_scatter()

        assert len(results) == 5

    def test_scatter_within_region(self):
        """Scatter placements should be within defined region."""
        tool = PlacementTool()
        tool.scatter_settings.count = 20
        tool.scatter_settings.region_min = Vector3(0, 0, 0)
        tool.scatter_settings.region_max = Vector3(10, 0, 10)
        results = tool.place_scatter()

        for r in results:
            assert 0 <= r.position.x <= 10
            assert 0 <= r.position.z <= 10

    def test_scatter_poisson_pattern(self):
        """Poisson disk scatter should maintain minimum distance."""
        tool = PlacementTool()
        tool.scatter_settings.count = 10
        tool.scatter_settings.pattern = ScatterPattern.POISSON_DISK
        tool.scatter_settings.min_distance = 2.0
        tool.scatter_settings.region_min = Vector3(0, 0, 0)
        tool.scatter_settings.region_max = Vector3(20, 0, 20)
        results = tool.place_scatter()

        # Check minimum distance between all pairs
        for i, r1 in enumerate(results):
            for j, r2 in enumerate(results):
                if i != j:
                    dx = r1.position.x - r2.position.x
                    dz = r1.position.z - r2.position.z
                    dist = math.sqrt(dx*dx + dz*dz)
                    # Allow small tolerance for numerical precision
                    assert dist >= tool.scatter_settings.min_distance - 0.1

    def test_scatter_grid_jitter(self):
        """Grid jitter pattern should create grid-like distribution."""
        tool = PlacementTool()
        tool.scatter_settings.count = 9
        tool.scatter_settings.pattern = ScatterPattern.GRID_JITTER
        tool.scatter_settings.region_min = Vector3(0, 0, 0)
        tool.scatter_settings.region_max = Vector3(10, 0, 10)
        results = tool.place_scatter()

        assert len(results) == 9

    def test_scatter_cluster(self):
        """Cluster pattern should create grouped distribution."""
        tool = PlacementTool()
        tool.scatter_settings.count = 15
        tool.scatter_settings.pattern = ScatterPattern.CLUSTER
        tool.scatter_settings.region_min = Vector3(0, 0, 0)
        tool.scatter_settings.region_max = Vector3(50, 0, 50)
        results = tool.place_scatter()

        assert len(results) >= 1

    def test_scatter_with_seed(self):
        """Scatter with seed should produce reproducible results."""
        tool1 = PlacementTool()
        tool1.scatter_settings.count = 5
        tool1.scatter_settings.seed = 42
        results1 = tool1.place_scatter()

        tool2 = PlacementTool()
        tool2.scatter_settings.count = 5
        tool2.scatter_settings.seed = 42
        results2 = tool2.place_scatter()

        for r1, r2 in zip(results1, results2):
            assert r1.position.x == r2.position.x
            assert r1.position.z == r2.position.z

    def test_foliage_settings_default(self):
        """FoliageSettings should have sensible defaults."""
        settings = FoliageSettings()
        assert settings.brush_radius == 10.0
        assert settings.density == 100.0
        assert settings.align_to_normal is True

    def test_place_foliage(self):
        """Foliage placement should create multiple instances."""
        tool = PlacementTool()
        tool.foliage_settings.brush_radius = 5.0
        tool.foliage_settings.density = 50.0
        results = tool.place_foliage(Vector3(0, 0, 0))

        assert len(results) >= 1

    def test_spline_settings_default(self):
        """SplineSettings should have sensible defaults."""
        settings = SplineSettings()
        assert settings.spacing == 2.0
        assert settings.closed_loop is False

    def test_place_along_spline(self):
        """Spline placement should place objects along path."""
        tool = PlacementTool()
        tool.spline_settings.points = [
            SplinePoint(position=Vector3(0, 0, 0)),
            SplinePoint(position=Vector3(10, 0, 0)),
        ]
        tool.spline_settings.spacing = 2.0
        results = tool.place_along_spline()

        assert len(results) >= 1
        # Objects should be along X axis
        for r in results:
            assert abs(r.position.z) < 0.1

    def test_spline_insufficient_points(self):
        """Spline with less than 2 points should return empty."""
        tool = PlacementTool()
        tool.spline_settings.points = [
            SplinePoint(position=Vector3(0, 0, 0)),
        ]
        results = tool.place_along_spline()

        assert len(results) == 0

    def test_update_preview(self):
        """Preview update should trigger callback."""
        tool = PlacementTool()
        previews = []

        def on_preview(transform):
            previews.append(transform)

        tool.on("on_preview", on_preview)
        tool.update_preview(Vector3(5, 5, 5))

        assert len(previews) == 1
        assert previews[0].position.x == 5

    def test_get_placed_objects(self):
        """Should track all placed object IDs."""
        tool = PlacementTool()
        tool.place_single(Vector3(0, 0, 0))
        tool.place_single(Vector3(1, 0, 0))

        objects = tool.get_placed_objects()
        assert len(objects) == 2

    def test_clear_placed_objects(self):
        """Should clear placed objects list."""
        tool = PlacementTool()
        tool.place_single(Vector3(0, 0, 0))
        tool.clear_placed_objects()

        objects = tool.get_placed_objects()
        assert len(objects) == 0

    def test_callback_registration(self):
        """Callbacks should register and unregister properly."""
        tool = PlacementTool()
        count = [0]

        def callback(result):
            count[0] += 1

        tool.on("on_place", callback)
        tool.place_single(Vector3(0, 0, 0))
        assert count[0] == 1

        tool.off("on_place", callback)
        tool.place_single(Vector3(1, 0, 0))
        assert count[0] == 1  # Should not have incremented

    def test_surface_alignment_none(self):
        """No alignment should return identity rotation."""
        tool = PlacementTool()
        tool.surface_alignment = SurfaceAlignment.NONE
        rot = tool.compute_alignment_rotation()
        assert rot.w == 1

    def test_surface_alignment_world_up(self):
        """World up alignment should return identity rotation."""
        tool = PlacementTool()
        tool.surface_alignment = SurfaceAlignment.WORLD_UP
        rot = tool.compute_alignment_rotation()
        assert rot.w == 1

    def test_surface_alignment_normal(self):
        """Normal alignment should rotate to match surface normal."""
        tool = PlacementTool()
        tool.surface_alignment = SurfaceAlignment.NORMAL
        normal = Vector3(1, 0, 0)  # Pointing along X
        rot = tool.compute_alignment_rotation(normal)
        # Rotation should not be identity
        assert not (rot.x == 0 and rot.y == 0 and rot.z == 0 and rot.w == 1)
