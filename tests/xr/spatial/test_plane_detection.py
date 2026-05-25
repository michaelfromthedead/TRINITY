"""
Tests for plane detection module.

Tests the plane detection system including:
    - Plane classification (floor, ceiling, wall, table, seat)
    - Plane geometry and bounds
    - Plane tracking states
    - Raycast and placement queries
    - PlaneDetector operations
"""

import pytest

from engine.core.math.vec import Vec2, Vec3
from engine.core.math.quat import Quat
from engine.xr.spatial.plane_detection import (
    DetectedPlane,
    PlaneBounds,
    PlaneAlignment,
    PlaneDetectionConfig,
    PlaneDetector,
    PlaneGeometry,
    PlaneOrientation,
    PlaneTrackingState,
    PlaneType,
)


class TestPlaneBounds:
    """Tests for PlaneBounds class."""

    def test_empty_bounds(self):
        bounds = PlaneBounds()
        assert bounds.vertex_count == 0
        assert bounds.compute_area() == 0.0

    def test_vertex_count(self):
        bounds = PlaneBounds(vertices=[
            Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)
        ])
        assert bounds.vertex_count == 4

    def test_compute_area_square(self):
        bounds = PlaneBounds(vertices=[
            Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)
        ])
        area = bounds.compute_area()
        assert abs(area - 1.0) < 0.001

    def test_compute_area_rectangle(self):
        bounds = PlaneBounds(vertices=[
            Vec2(0, 0), Vec2(2, 0), Vec2(2, 3), Vec2(0, 3)
        ])
        area = bounds.compute_area()
        assert abs(area - 6.0) < 0.001

    def test_contains_point_inside(self):
        bounds = PlaneBounds(vertices=[
            Vec2(0, 0), Vec2(2, 0), Vec2(2, 2), Vec2(0, 2)
        ])
        assert bounds.contains_point(Vec2(1, 1)) is True

    def test_contains_point_outside(self):
        bounds = PlaneBounds(vertices=[
            Vec2(0, 0), Vec2(2, 0), Vec2(2, 2), Vec2(0, 2)
        ])
        assert bounds.contains_point(Vec2(5, 5)) is False

    def test_contains_point_on_edge(self):
        bounds = PlaneBounds(vertices=[
            Vec2(0, 0), Vec2(2, 0), Vec2(2, 2), Vec2(0, 2)
        ])
        # Points on edge may or may not be included depending on algorithm
        result = bounds.contains_point(Vec2(0, 1))
        assert isinstance(result, bool)


class TestPlaneGeometry:
    """Tests for PlaneGeometry class."""

    def test_default_geometry(self):
        geo = PlaneGeometry()
        assert geo.center == Vec3.zero()
        assert geo.normal == Vec3.up()
        assert geo.width == 0.0
        assert geo.height == 0.0

    def test_area_from_bounds(self):
        geo = PlaneGeometry(
            width=2.0,
            height=3.0,
            bounds=PlaneBounds(vertices=[
                Vec2(-1, -1.5), Vec2(1, -1.5), Vec2(1, 1.5), Vec2(-1, 1.5)
            ])
        )
        assert abs(geo.area - 6.0) < 0.001

    def test_extents(self):
        geo = PlaneGeometry(width=4.0, height=6.0)
        extents = geo.extents
        assert extents.x == 2.0
        assert extents.y == 3.0


class TestDetectedPlane:
    """Tests for DetectedPlane class."""

    def test_create_floor_plane(self):
        plane = DetectedPlane(
            plane_type=PlaneType.FLOOR,
            center=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
        )
        assert plane.plane_type == PlaneType.FLOOR
        assert plane.is_horizontal is True
        assert plane.is_vertical is False

    def test_create_wall_plane(self):
        plane = DetectedPlane(
            plane_type=PlaneType.WALL,
            center=Vec3(2, 1, 0),
            normal=Vec3(1, 0, 0),
        )
        assert plane.plane_type == PlaneType.WALL
        assert plane.is_horizontal is False
        assert plane.is_vertical is True

    def test_plane_orientation_detection(self):
        # Floor (up normal)
        floor = DetectedPlane(normal=Vec3(0, 1, 0))
        assert floor.plane_orientation == PlaneOrientation.HORIZONTAL_UP

        # Ceiling (down normal)
        ceiling = DetectedPlane(normal=Vec3(0, -1, 0))
        assert ceiling.plane_orientation == PlaneOrientation.HORIZONTAL_DOWN

        # Wall (horizontal normal)
        wall = DetectedPlane(normal=Vec3(1, 0, 0))
        assert wall.plane_orientation == PlaneOrientation.VERTICAL

    def test_initial_tracking_state(self):
        plane = DetectedPlane()
        assert plane.tracking_state == PlaneTrackingState.NONE
        assert plane.is_tracked is False
        assert plane.is_valid is False

    def test_update_geometry(self):
        plane = DetectedPlane()
        plane.update_geometry(
            center=Vec3(1, 0, 2),
            normal=Vec3(0, 1, 0),
            orientation=Quat.identity(),
            width=2.0,
            height=3.0,
            boundary_vertices=[
                Vec2(-1, -1.5), Vec2(1, -1.5), Vec2(1, 1.5), Vec2(-1, 1.5)
            ],
            timestamp=1.0,
        )

        assert plane.center == Vec3(1, 0, 2)
        assert plane.width == 2.0
        assert plane.height == 3.0
        assert abs(plane.area - 6.0) < 0.001

    def test_update_tracking_state(self):
        plane = DetectedPlane()
        plane.update_tracking_state(PlaneTrackingState.TRACKED, confidence=0.95)

        assert plane.tracking_state == PlaneTrackingState.TRACKED
        assert plane.is_tracked is True
        assert plane.confidence == 0.95

    def test_mark_subsumed(self):
        plane = DetectedPlane()
        plane.mark_subsumed("other-plane-id")

        assert plane.is_subsumed is True
        assert plane.subsumed_by == "other-plane-id"
        assert plane.tracking_state == PlaneTrackingState.STOPPED

    def test_world_to_local(self):
        plane = DetectedPlane(center=Vec3(5, 0, 5))
        plane.update_geometry(
            center=Vec3(5, 0, 5),
            normal=Vec3(0, 1, 0),
            orientation=Quat.identity(),
            width=2.0,
            height=2.0,
            boundary_vertices=[],
            timestamp=1.0,
        )

        local = plane.world_to_local(Vec3(6, 0, 6))
        assert abs(local.x - 1.0) < 0.01
        assert abs(local.y - 1.0) < 0.01

    def test_local_to_world(self):
        plane = DetectedPlane(center=Vec3(5, 0, 5))
        plane.update_geometry(
            center=Vec3(5, 0, 5),
            normal=Vec3(0, 1, 0),
            orientation=Quat.identity(),
            width=2.0,
            height=2.0,
            boundary_vertices=[],
            timestamp=1.0,
        )

        world = plane.local_to_world(Vec2(1, 1))
        assert abs(world.x - 6.0) < 0.01
        assert abs(world.z - 6.0) < 0.01

    def test_project_point(self):
        plane = DetectedPlane(center=Vec3(0, 0, 0), normal=Vec3(0, 1, 0))
        plane.update_geometry(
            center=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            orientation=Quat.identity(),
            width=10.0,
            height=10.0,
            boundary_vertices=[],
            timestamp=1.0,
        )

        projected = plane.project_point(Vec3(3, 5, 2))
        assert abs(projected.x - 3.0) < 0.01
        assert abs(projected.y - 0.0) < 0.01
        assert abs(projected.z - 2.0) < 0.01

    def test_distance_to_point(self):
        plane = DetectedPlane(center=Vec3(0, 0, 0), normal=Vec3(0, 1, 0))
        plane.update_geometry(
            center=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            orientation=Quat.identity(),
            width=10.0,
            height=10.0,
            boundary_vertices=[],
            timestamp=1.0,
        )

        distance = plane.distance_to_point(Vec3(0, 5, 0))
        assert abs(distance - 5.0) < 0.01

    def test_callbacks(self):
        plane = DetectedPlane()
        callback_results = []

        def on_geometry_updated(p):
            callback_results.append("geometry")

        plane.add_callback("geometry_updated", on_geometry_updated)
        plane.update_geometry(
            Vec3.zero(), Vec3.up(), Quat.identity(), 1.0, 1.0, [], 1.0
        )

        assert "geometry" in callback_results


class TestPlaneDetector:
    """Tests for PlaneDetector class."""

    def test_default_config(self):
        detector = PlaneDetector()
        assert detector.config.alignment == PlaneAlignment.ANY
        assert detector.config.min_area == 0.25
        assert detector.config.max_planes == 100

    def test_custom_config(self):
        config = PlaneDetectionConfig(
            alignment=PlaneAlignment.HORIZONTAL,
            min_area=0.5,
            max_planes=50,
        )
        detector = PlaneDetector(config)
        assert detector.config.alignment == PlaneAlignment.HORIZONTAL
        assert detector.config.min_area == 0.5

    def test_start_stop(self):
        detector = PlaneDetector()
        assert detector.is_running is False

        detector.start()
        assert detector.is_running is True

        detector.stop()
        assert detector.is_running is False

    def test_double_start(self):
        detector = PlaneDetector()
        detector.start()
        result = detector.start()
        assert result is False

    def test_add_plane(self):
        detector = PlaneDetector()
        plane = DetectedPlane(plane_type=PlaneType.FLOOR)
        detector.add_plane(plane)

        assert detector.plane_count == 1
        assert detector.get_plane(plane.plane_id) is plane

    def test_get_all_planes(self):
        detector = PlaneDetector()
        for _ in range(3):
            detector.add_plane(DetectedPlane())

        assert len(detector.get_all_planes()) == 3

    def test_get_planes_by_type(self):
        detector = PlaneDetector()
        detector.add_plane(DetectedPlane(plane_type=PlaneType.FLOOR))
        detector.add_plane(DetectedPlane(plane_type=PlaneType.WALL))
        detector.add_plane(DetectedPlane(plane_type=PlaneType.FLOOR))

        floors = detector.get_planes_by_type(PlaneType.FLOOR)
        assert len(floors) == 2

    def test_get_horizontal_planes(self):
        detector = PlaneDetector()
        detector.add_plane(DetectedPlane(
            plane_type=PlaneType.FLOOR, normal=Vec3(0, 1, 0)
        ))
        detector.add_plane(DetectedPlane(
            plane_type=PlaneType.WALL, normal=Vec3(1, 0, 0)
        ))
        detector.add_plane(DetectedPlane(
            plane_type=PlaneType.TABLE, normal=Vec3(0, 1, 0)
        ))

        horizontal = detector.get_horizontal_planes()
        assert len(horizontal) == 2

    def test_get_vertical_planes(self):
        detector = PlaneDetector()
        detector.add_plane(DetectedPlane(
            plane_type=PlaneType.FLOOR, normal=Vec3(0, 1, 0)
        ))
        detector.add_plane(DetectedPlane(
            plane_type=PlaneType.WALL, normal=Vec3(1, 0, 0)
        ))

        vertical = detector.get_vertical_planes()
        assert len(vertical) == 1

    def test_get_floor_planes(self):
        detector = PlaneDetector()
        detector.add_plane(DetectedPlane(plane_type=PlaneType.FLOOR))
        detector.add_plane(DetectedPlane(plane_type=PlaneType.WALL))

        floors = detector.get_floor_planes()
        assert len(floors) == 1

    def test_raycast_hit(self):
        detector = PlaneDetector()
        plane = DetectedPlane(
            plane_type=PlaneType.FLOOR,
            center=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
        )
        plane.update_geometry(
            center=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            orientation=Quat.identity(),
            width=10.0,
            height=10.0,
            boundary_vertices=[
                Vec2(-5, -5), Vec2(5, -5), Vec2(5, 5), Vec2(-5, 5)
            ],
            timestamp=1.0,
        )
        plane.update_tracking_state(PlaneTrackingState.TRACKED)
        detector.add_plane(plane)

        result = detector.raycast(
            origin=Vec3(0, 5, 0),
            direction=Vec3(0, -1, 0),
        )

        assert result is not None
        hit_plane, hit_point, distance = result
        assert hit_plane is plane
        assert abs(distance - 5.0) < 0.1

    def test_raycast_miss(self):
        detector = PlaneDetector()
        plane = DetectedPlane(
            plane_type=PlaneType.FLOOR,
            center=Vec3(100, 0, 100),  # Far away
            normal=Vec3(0, 1, 0),
        )
        plane.update_geometry(
            center=Vec3(100, 0, 100),
            normal=Vec3(0, 1, 0),
            orientation=Quat.identity(),
            width=1.0,
            height=1.0,
            boundary_vertices=[
                Vec2(-0.5, -0.5), Vec2(0.5, -0.5), Vec2(0.5, 0.5), Vec2(-0.5, 0.5)
            ],
            timestamp=1.0,
        )
        plane.update_tracking_state(PlaneTrackingState.TRACKED)
        detector.add_plane(plane)

        result = detector.raycast(
            origin=Vec3(0, 5, 0),
            direction=Vec3(0, -1, 0),
        )

        assert result is None

    def test_find_placement_surface(self):
        detector = PlaneDetector()
        plane = DetectedPlane(
            plane_type=PlaneType.FLOOR,
            center=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
        )
        plane.update_geometry(
            center=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            orientation=Quat.identity(),
            width=10.0,
            height=10.0,
            boundary_vertices=[
                Vec2(-5, -5), Vec2(5, -5), Vec2(5, 5), Vec2(-5, 5)
            ],
            timestamp=1.0,
        )
        plane.update_tracking_state(PlaneTrackingState.TRACKED)
        detector.add_plane(plane)

        result = detector.find_placement_surface(
            position=Vec3(1, 2, 1),
            prefer_horizontal=True,
        )

        assert result is not None
        found_plane, snap_point = result
        assert found_plane is plane

    def test_clear_all(self):
        detector = PlaneDetector()
        detector.add_plane(DetectedPlane())
        detector.add_plane(DetectedPlane())

        detector.clear_all()
        assert detector.plane_count == 0

    def test_callbacks(self):
        detector = PlaneDetector()
        callback_results = []

        def on_plane_added(plane):
            callback_results.append("added")

        detector.add_callback("plane_added", on_plane_added)
        detector.add_plane(DetectedPlane())

        assert "added" in callback_results
