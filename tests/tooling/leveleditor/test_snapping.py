"""
Tests for the snapping module.

Tests all snap types and combinations.
"""

import math
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.leveleditor.snapping import (
    SnapMode,
    SnapSettings,
    SnapResult,
    GridSnap,
    GridSnapSettings,
    SurfaceSnap,
    SurfaceSnapSettings,
    VertexSnap,
    VertexSnapSettings,
    VertexInfo,
    EdgeSnap,
    EdgeSnapSettings,
    EdgeInfo,
    PivotSnap,
    PivotSnapSettings,
    PivotInfo,
    SnapManager,
    SnapPriority,
    GridType,
)
from engine.tooling.leveleditor.placement import Vector3, Quaternion, Transform
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


class TestSnapMode:
    """Tests for SnapMode flags."""

    def test_mode_none(self):
        """NONE mode should be zero."""
        assert SnapMode.NONE.value == 0

    def test_mode_combination(self):
        """Modes should combine with bitwise OR."""
        combined = SnapMode.GRID | SnapMode.SURFACE
        assert SnapMode.GRID in combined
        assert SnapMode.SURFACE in combined
        assert SnapMode.VERTEX not in combined

    def test_mode_all(self):
        """ALL mode should include all individual modes."""
        assert SnapMode.GRID in SnapMode.ALL
        assert SnapMode.SURFACE in SnapMode.ALL
        assert SnapMode.VERTEX in SnapMode.ALL
        assert SnapMode.EDGE in SnapMode.ALL
        assert SnapMode.PIVOT in SnapMode.ALL


class TestGridSnap:
    """Tests for GridSnap class."""

    def test_creation_default(self):
        """GridSnap should initialize with default settings."""
        snap = GridSnap()
        assert snap.settings.enabled is True
        assert snap.settings.size_x == 1.0

    def test_creation_with_settings(self):
        """GridSnap should use provided settings."""
        settings = GridSnapSettings(size_x=2.0, size_y=2.0, size_z=2.0)
        snap = GridSnap(settings)
        assert snap.settings.size_x == 2.0

    def test_snap_to_grid(self):
        """Snap should align to grid positions."""
        snap = GridSnap(GridSnapSettings(size_x=1.0, size_y=1.0, size_z=1.0))
        result = snap.snap(Vector3(0.3, 0.7, 1.4))

        assert result.snapped is True
        assert result.position.x == 0.0
        assert result.position.y == 1.0
        assert result.position.z == 1.0

    def test_snap_large_grid(self):
        """Snap should work with larger grid sizes."""
        snap = GridSnap(GridSnapSettings(size_x=5.0, size_y=5.0, size_z=5.0))
        result = snap.snap(Vector3(7.4, 3.2, 12.8))

        assert result.position.x == 5.0
        assert result.position.y == 5.0
        assert result.position.z == 15.0

    def test_snap_negative_positions(self):
        """Snap should work with negative positions."""
        snap = GridSnap(GridSnapSettings(size_x=1.0, size_y=1.0, size_z=1.0))
        result = snap.snap(Vector3(-0.3, -1.7, -2.4))

        assert result.position.x == 0.0
        assert result.position.y == -2.0
        assert result.position.z == -2.0

    def test_snap_disabled(self):
        """Disabled snap should return original position."""
        snap = GridSnap(GridSnapSettings(enabled=False))
        pos = Vector3(0.3, 0.7, 1.4)
        result = snap.snap(pos)

        assert result.snapped is False
        assert result.position.x == 0.3

    def test_snap_with_subdivisions(self):
        """Subdivisions should create finer grid."""
        snap = GridSnap(GridSnapSettings(size_x=1.0, size_y=1.0, size_z=1.0, subdivisions=2))
        result = snap.snap(Vector3(0.3, 0.7, 1.4))

        assert result.position.x == 0.5
        assert result.position.y == 0.5
        assert result.position.z == 1.5

    def test_snap_with_origin(self):
        """Custom origin should offset the grid."""
        settings = GridSnapSettings(
            size_x=1.0, size_y=1.0, size_z=1.0,
            grid_type=GridType.CUSTOM,
            origin=Vector3(0.5, 0.5, 0.5)
        )
        snap = GridSnap(settings)
        result = snap.snap(Vector3(1.3, 0.7, 1.4))

        # Should snap to 1.5, 0.5, 1.5 (origin + grid)
        assert result.position.x == 1.5
        assert result.position.y == 0.5
        assert result.position.z == 1.5

    def test_snap_to_integer(self):
        """Snap to integer should round to whole numbers."""
        snap = GridSnap(GridSnapSettings(size_x=2.0, snap_to_integer=True))
        result = snap.snap(Vector3(3.7, 5.2, 8.9))

        assert result.position.x == 4.0
        assert result.position.y == 5.0
        assert result.position.z == 9.0

    def test_snap_type_set(self):
        """Result should indicate GRID snap type."""
        snap = GridSnap()
        result = snap.snap(Vector3(0, 0, 0))
        assert result.snap_type == SnapMode.GRID

    def test_snap_distance_calculated(self):
        """Distance should be calculated from original position."""
        snap = GridSnap(GridSnapSettings(size_x=1.0, size_y=1.0, size_z=1.0))
        result = snap.snap(Vector3(0.4, 0, 0))

        assert abs(result.distance - 0.4) < 0.0001

    def test_get_grid_lines(self):
        """Grid lines should be generated for visualization."""
        snap = GridSnap(GridSnapSettings(size_x=1.0, size_y=1.0, size_z=1.0))
        lines = snap.get_grid_lines(Vector3(0, 0, 0), 10)

        assert len(lines) > 0
        for start, end in lines:
            assert isinstance(start, Vector3)
            assert isinstance(end, Vector3)


class TestVertexSnap:
    """Tests for VertexSnap class."""

    def test_creation_default(self):
        """VertexSnap should initialize with default settings."""
        snap = VertexSnap()
        assert snap.settings.enabled is True
        assert snap.settings.snap_radius == 5.0

    def test_snap_to_nearest_vertex(self):
        """Should snap to nearest vertex within radius."""
        snap = VertexSnap()
        snap._cached_vertices = [
            VertexInfo(position=Vector3(0, 0, 0), object_id="obj1"),
            VertexInfo(position=Vector3(10, 0, 0), object_id="obj2"),
        ]

        result = snap.snap(Vector3(2, 0, 0))

        assert result.snapped is True
        assert result.position.x == 0
        assert result.snap_target_id == "obj1"

    def test_snap_outside_radius(self):
        """Should not snap if outside radius."""
        snap = VertexSnap(VertexSnapSettings(snap_radius=1.0))
        snap._cached_vertices = [
            VertexInfo(position=Vector3(0, 0, 0), object_id="obj1"),
        ]

        result = snap.snap(Vector3(5, 0, 0))

        assert result.snapped is False

    def test_snap_disabled(self):
        """Disabled snap should return original position."""
        snap = VertexSnap(VertexSnapSettings(enabled=False))
        snap._cached_vertices = [
            VertexInfo(position=Vector3(0, 0, 0), object_id="obj1"),
        ]

        result = snap.snap(Vector3(0.1, 0, 0))

        assert result.snapped is False

    def test_get_snap_candidates(self):
        """Should return vertices within radius."""
        snap = VertexSnap(VertexSnapSettings(snap_radius=5.0))
        snap._cached_vertices = [
            VertexInfo(position=Vector3(0, 0, 0), object_id="obj1"),
            VertexInfo(position=Vector3(3, 0, 0), object_id="obj2"),
            VertexInfo(position=Vector3(10, 0, 0), object_id="obj3"),
        ]

        candidates = snap.get_snap_candidates(Vector3(1, 0, 0))

        assert len(candidates) == 2

    def test_snap_preserves_normal(self):
        """Should preserve vertex normal in result."""
        snap = VertexSnap()
        snap._cached_vertices = [
            VertexInfo(
                position=Vector3(0, 0, 0),
                normal=Vector3(0, 1, 0),
                object_id="obj1"
            ),
        ]

        result = snap.snap(Vector3(1, 0, 0))

        assert result.normal is not None
        assert result.normal.y == 1

    def test_snap_type_set(self):
        """Result should indicate VERTEX snap type."""
        snap = VertexSnap()
        snap._cached_vertices = [
            VertexInfo(position=Vector3(0, 0, 0), object_id="obj1"),
        ]

        result = snap.snap(Vector3(0, 0, 0))
        assert result.snap_type == SnapMode.VERTEX


class TestEdgeSnap:
    """Tests for EdgeSnap class."""

    def test_creation_default(self):
        """EdgeSnap should initialize with default settings."""
        snap = EdgeSnap()
        assert snap.settings.enabled is True
        assert snap.settings.snap_to_midpoint is True

    def test_snap_to_edge(self):
        """Should snap to nearest point on edge."""
        snap = EdgeSnap()
        snap._cached_edges = [
            EdgeInfo(start=Vector3(0, 0, 0), end=Vector3(10, 0, 0), object_id="obj1"),
        ]

        result = snap.snap(Vector3(5, 2, 0))

        assert result.snapped is True
        assert result.position.x == 5
        assert result.position.y == 0

    def test_snap_to_midpoint(self):
        """Should prefer midpoint when close."""
        snap = EdgeSnap(EdgeSnapSettings(snap_to_midpoint=True, snap_radius=10.0))
        snap._cached_edges = [
            EdgeInfo(start=Vector3(0, 0, 0), end=Vector3(10, 0, 0), object_id="obj1"),
        ]

        result = snap.snap(Vector3(5, 0.5, 0))

        assert result.position.x == 5  # Midpoint

    def test_edge_info_midpoint(self):
        """EdgeInfo should calculate midpoint correctly."""
        edge = EdgeInfo(start=Vector3(0, 0, 0), end=Vector3(10, 0, 0))
        mid = edge.midpoint

        assert mid.x == 5
        assert mid.y == 0
        assert mid.z == 0

    def test_edge_info_direction(self):
        """EdgeInfo should calculate direction correctly."""
        edge = EdgeInfo(start=Vector3(0, 0, 0), end=Vector3(10, 0, 0))
        direction = edge.direction

        assert abs(direction.x - 1.0) < 0.0001
        assert direction.y == 0
        assert direction.z == 0

    def test_edge_info_length(self):
        """EdgeInfo should calculate length correctly."""
        edge = EdgeInfo(start=Vector3(0, 0, 0), end=Vector3(3, 4, 0))

        assert edge.length == 5.0

    def test_snap_outside_radius(self):
        """Should not snap if outside radius."""
        snap = EdgeSnap(EdgeSnapSettings(snap_radius=1.0))
        snap._cached_edges = [
            EdgeInfo(start=Vector3(0, 0, 0), end=Vector3(10, 0, 0), object_id="obj1"),
        ]

        result = snap.snap(Vector3(5, 10, 0))

        assert result.snapped is False

    def test_snap_type_set(self):
        """Result should indicate EDGE snap type."""
        snap = EdgeSnap()
        snap._cached_edges = [
            EdgeInfo(start=Vector3(0, 0, 0), end=Vector3(10, 0, 0), object_id="obj1"),
        ]

        result = snap.snap(Vector3(5, 0, 0))
        assert result.snap_type == SnapMode.EDGE


class TestPivotSnap:
    """Tests for PivotSnap class."""

    def test_creation_default(self):
        """PivotSnap should initialize with default settings."""
        snap = PivotSnap()
        assert snap.settings.enabled is True
        assert snap.settings.snap_radius == 10.0

    def test_snap_to_pivot(self):
        """Should snap to nearest pivot."""
        snap = PivotSnap()
        snap._cached_pivots = [
            PivotInfo(position=Vector3(0, 0, 0), object_id="obj1"),
            PivotInfo(position=Vector3(10, 0, 0), object_id="obj2"),
        ]

        result = snap.snap(Vector3(2, 0, 0))

        assert result.snapped is True
        assert result.position.x == 0
        assert result.snap_target_id == "obj1"

    def test_snap_outside_radius(self):
        """Should not snap if outside radius."""
        snap = PivotSnap(PivotSnapSettings(snap_radius=1.0))
        snap._cached_pivots = [
            PivotInfo(position=Vector3(0, 0, 0), object_id="obj1"),
        ]

        result = snap.snap(Vector3(5, 0, 0))

        assert result.snapped is False

    def test_snap_type_set(self):
        """Result should indicate PIVOT snap type."""
        snap = PivotSnap()
        snap._cached_pivots = [
            PivotInfo(position=Vector3(0, 0, 0), object_id="obj1"),
        ]

        result = snap.snap(Vector3(0, 0, 0))
        assert result.snap_type == SnapMode.PIVOT


class TestSnapManager:
    """Tests for SnapManager class."""

    def test_creation(self):
        """SnapManager should initialize with default settings."""
        manager = SnapManager()
        assert manager.settings.enabled is True
        assert SnapMode.GRID in manager.settings.mode

    def test_set_grid_size(self):
        """Should set uniform grid size."""
        manager = SnapManager()
        manager.set_grid_size(5.0)

        assert manager.grid.settings.size_x == 5.0
        assert manager.grid.settings.size_y == 5.0
        assert manager.grid.settings.size_z == 5.0

    def test_toggle_mode(self):
        """Should toggle snap modes on and off."""
        manager = SnapManager()
        initial_has_surface = SnapMode.SURFACE in manager.settings.mode

        manager.toggle_mode(SnapMode.SURFACE)

        if initial_has_surface:
            assert SnapMode.SURFACE not in manager.settings.mode
        else:
            assert SnapMode.SURFACE in manager.settings.mode

    def test_snap_with_grid_only(self):
        """Should use grid snap when only grid is enabled."""
        manager = SnapManager()
        manager.settings.mode = SnapMode.GRID

        result = manager.snap(Vector3(0.3, 0.7, 1.4))

        assert result.snapped is True
        assert result.snap_type == SnapMode.GRID

    def test_snap_disabled(self):
        """Disabled manager should not snap."""
        manager = SnapManager()
        manager.settings.enabled = False

        result = manager.snap(Vector3(0.3, 0.7, 1.4))

        assert result.snapped is False

    def test_snap_priority_nearest(self):
        """NEAREST priority should choose closest snap."""
        manager = SnapManager()
        manager.settings.priority = SnapPriority.NEAREST
        manager.settings.mode = SnapMode.GRID | SnapMode.VERTEX

        # Setup vertex very close
        manager.vertex._cached_vertices = [
            VertexInfo(position=Vector3(0.1, 0, 0), object_id="obj1"),
        ]

        result = manager.snap(Vector3(0.15, 0, 0))

        # Vertex should be closer than grid at 0
        assert result.snap_type == SnapMode.VERTEX

    def test_snap_priority_vertex_first(self):
        """VERTEX_FIRST priority should prefer vertex snaps."""
        manager = SnapManager()
        manager.settings.priority = SnapPriority.VERTEX_FIRST
        manager.settings.mode = SnapMode.GRID | SnapMode.VERTEX

        # Setup vertex within range
        manager.vertex._cached_vertices = [
            VertexInfo(position=Vector3(2, 0, 0), object_id="obj1"),
        ]

        result = manager.snap(Vector3(1, 0, 0))

        assert result.snap_type == SnapMode.VERTEX

    def test_snap_priority_grid_first(self):
        """GRID_FIRST priority should prefer grid snaps."""
        manager = SnapManager()
        manager.settings.priority = SnapPriority.GRID_FIRST
        manager.settings.mode = SnapMode.GRID | SnapMode.VERTEX

        manager.vertex._cached_vertices = [
            VertexInfo(position=Vector3(2, 0, 0), object_id="obj1"),
        ]

        result = manager.snap(Vector3(1, 0, 0))

        assert result.snap_type == SnapMode.GRID

    def test_last_result_stored(self):
        """Should store last snap result."""
        manager = SnapManager()
        result = manager.snap(Vector3(0, 0, 0))

        assert manager.last_result is result

    def test_snap_combines_multiple_modes(self):
        """Should evaluate all enabled modes."""
        manager = SnapManager()
        manager.settings.mode = SnapMode.GRID | SnapMode.VERTEX | SnapMode.EDGE

        result = manager.snap(Vector3(0.3, 0.7, 1.4))

        assert result.snapped is True
