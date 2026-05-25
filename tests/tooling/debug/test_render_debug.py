"""Tests for render debug - wireframe, bounds, overdraw."""

import pytest
from engine.tooling.debug.render_debug import (
    RenderDebugger,
    WireframeMode,
    BoundingBoxDisplay,
    LODVisualization,
    OverdrawHeatmap,
    BoundingBox,
    BoundingBoxType,
    LODObject,
    LODLevel,
    Vector3,
    Quaternion,
)


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_bbox_creation(self):
        bbox = BoundingBox(
            box_id="box_001",
            box_type=BoundingBoxType.AABB,
            center=Vector3(0, 0, 0),
            extents=Vector3(1, 1, 1),
        )
        assert bbox.box_id == "box_001"
        assert bbox.box_type == BoundingBoxType.AABB

    def test_bbox_obb(self):
        bbox = BoundingBox(
            box_id="obb",
            box_type=BoundingBoxType.OBB,
            center=Vector3(0, 0, 0),
            extents=Vector3(1, 2, 3),
            rotation=Quaternion(0, 0, 0, 1),
        )
        assert bbox.box_type == BoundingBoxType.OBB

    def test_bbox_sphere(self):
        bbox = BoundingBox(
            box_id="sphere",
            box_type=BoundingBoxType.SPHERE,
            center=Vector3(0, 0, 0),
            radius=5.0,
        )
        assert bbox.radius == 5.0


class TestBoundingBoxDisplay:
    """Tests for BoundingBoxDisplay class."""

    def test_display_creation(self):
        display = BoundingBoxDisplay()
        assert display.is_enabled is True
        assert display.box_count == 0

    def test_add_box(self):
        display = BoundingBoxDisplay()
        bbox = BoundingBox("b1", BoundingBoxType.AABB, Vector3(0, 0, 0))
        display.add_box(bbox)
        assert display.box_count == 1
        assert display.get_box("b1") is bbox

    def test_remove_box(self):
        display = BoundingBoxDisplay()
        bbox = BoundingBox("b1", BoundingBoxType.AABB, Vector3(0, 0, 0))
        display.add_box(bbox)
        removed = display.remove_box("b1")
        assert removed is bbox
        assert display.box_count == 0

    def test_update_box(self):
        display = BoundingBoxDisplay()
        bbox = BoundingBox("b1", BoundingBoxType.AABB, Vector3(0, 0, 0))
        display.add_box(bbox)

        result = display.update_box(
            "b1",
            center=Vector3(10, 10, 10),
            extents=Vector3(2, 2, 2),
        )
        assert result is True
        assert bbox.center.x == 10
        assert bbox.extents.x == 2

    def test_update_box_not_found(self):
        display = BoundingBoxDisplay()
        result = display.update_box("nonexistent", center=Vector3(0, 0, 0))
        assert result is False

    def test_show_options(self):
        display = BoundingBoxDisplay()
        display.set_show_aabb(False)
        display.set_show_obb(False)
        display.set_show_sphere(False)

    def test_generate_draw_commands_aabb(self):
        display = BoundingBoxDisplay()
        bbox = BoundingBox("b1", BoundingBoxType.AABB, Vector3(0, 0, 0), extents=Vector3(1, 1, 1))
        display.add_box(bbox)

        commands = display.generate_draw_commands()
        assert len(commands) == 1
        assert commands[0]["type"] == "box"

    def test_generate_draw_commands_obb(self):
        display = BoundingBoxDisplay()
        bbox = BoundingBox(
            "b1",
            BoundingBoxType.OBB,
            Vector3(0, 0, 0),
            extents=Vector3(1, 1, 1),
            rotation=Quaternion.identity(),
        )
        display.add_box(bbox)

        commands = display.generate_draw_commands()
        assert len(commands) == 1
        assert commands[0]["type"] == "box"
        assert "rotation" in commands[0]

    def test_generate_draw_commands_sphere(self):
        display = BoundingBoxDisplay()
        bbox = BoundingBox("b1", BoundingBoxType.SPHERE, Vector3(0, 0, 0), radius=5.0)
        display.add_box(bbox)

        commands = display.generate_draw_commands()
        assert len(commands) == 1
        assert commands[0]["type"] == "sphere"

    def test_generate_draw_commands_disabled(self):
        display = BoundingBoxDisplay()
        bbox = BoundingBox("b1", BoundingBoxType.AABB, Vector3(0, 0, 0))
        display.add_box(bbox)
        display.disable()

        commands = display.generate_draw_commands()
        assert len(commands) == 0

    def test_filter_by_type(self):
        display = BoundingBoxDisplay()
        display.add_box(BoundingBox("aabb", BoundingBoxType.AABB, Vector3(0, 0, 0)))
        display.add_box(BoundingBox("obb", BoundingBoxType.OBB, Vector3(0, 0, 0)))

        display.set_show_obb(False)
        commands = display.generate_draw_commands()
        assert len(commands) == 1

    def test_clear_all_boxes(self):
        display = BoundingBoxDisplay()
        display.add_box(BoundingBox("b1", BoundingBoxType.AABB, Vector3(0, 0, 0)))
        display.add_box(BoundingBox("b2", BoundingBoxType.AABB, Vector3(0, 0, 0)))
        display.clear_all_boxes()
        assert display.box_count == 0


class TestLODObject:
    """Tests for LODObject class."""

    def test_lod_object_creation(self):
        obj = LODObject(
            object_id="obj_001",
            position=Vector3(0, 0, 0),
            current_lod=LODLevel.LOD0,
        )
        assert obj.object_id == "obj_001"
        assert obj.current_lod == LODLevel.LOD0

    def test_lod_object_triangle_counts(self):
        obj = LODObject(
            object_id="obj",
            position=Vector3(0, 0, 0),
            triangle_counts=[10000, 5000, 2000, 500, 100],
        )
        assert obj.triangle_counts[0] == 10000
        assert obj.triangle_counts[4] == 100


class TestLODVisualization:
    """Tests for LODVisualization class."""

    def test_visualization_creation(self):
        viz = LODVisualization()
        assert viz.is_enabled is True
        assert viz.object_count == 0

    def test_add_object(self):
        viz = LODVisualization()
        obj = LODObject("o1", Vector3(0, 0, 0))
        viz.add_object(obj)
        assert viz.object_count == 1
        assert viz.get_object("o1") is obj

    def test_remove_object(self):
        viz = LODVisualization()
        obj = LODObject("o1", Vector3(0, 0, 0))
        viz.add_object(obj)
        removed = viz.remove_object("o1")
        assert removed is obj
        assert viz.object_count == 0

    def test_update_object(self):
        viz = LODVisualization()
        obj = LODObject("o1", Vector3(0, 0, 0))
        viz.add_object(obj)

        result = viz.update_object(
            "o1",
            position=Vector3(10, 0, 0),
            current_lod=LODLevel.LOD2,
            is_culled=True,
            screen_size=5.0,
        )
        assert result is True
        assert obj.position.x == 10
        assert obj.current_lod == LODLevel.LOD2
        assert obj.is_culled is True
        assert obj.screen_size == 5.0

    def test_update_object_not_found(self):
        viz = LODVisualization()
        result = viz.update_object("nonexistent", position=Vector3(0, 0, 0))
        assert result is False

    def test_show_options(self):
        viz = LODVisualization()
        viz.set_show_lod_levels(False)
        viz.set_show_triangle_counts(True)
        viz.set_show_screen_size(True)

    def test_get_lod_color(self):
        viz = LODVisualization()
        color_lod0 = viz.get_lod_color(LODLevel.LOD0)
        color_lod4 = viz.get_lod_color(LODLevel.LOD4)
        # LOD0 should be green, LOD4 should be red
        assert color_lod0[1] == 1.0  # Green
        assert color_lod4[0] == 1.0  # Red

    def test_generate_draw_commands(self):
        viz = LODVisualization()
        obj = LODObject("o1", Vector3(0, 0, 0), current_lod=LODLevel.LOD1)
        viz.add_object(obj)

        commands = viz.generate_draw_commands()
        assert len(commands) > 0

    def test_generate_draw_commands_culled(self):
        viz = LODVisualization()
        obj = LODObject("o1", Vector3(0, 0, 0), is_culled=True)
        viz.add_object(obj)

        commands = viz.generate_draw_commands()
        # Should still generate commands for culled objects
        assert len(commands) > 0

    def test_generate_draw_commands_disabled(self):
        viz = LODVisualization()
        obj = LODObject("o1", Vector3(0, 0, 0))
        viz.add_object(obj)
        viz.disable()

        commands = viz.generate_draw_commands()
        assert len(commands) == 0

    def test_get_objects_by_lod(self):
        viz = LODVisualization()
        viz.add_object(LODObject("o1", Vector3(0, 0, 0), current_lod=LODLevel.LOD0))
        viz.add_object(LODObject("o2", Vector3(0, 0, 0), current_lod=LODLevel.LOD2))
        viz.add_object(LODObject("o3", Vector3(0, 0, 0), current_lod=LODLevel.LOD0))

        lod0_objects = viz.get_objects_by_lod(LODLevel.LOD0)
        assert len(lod0_objects) == 2

    def test_get_culled_objects(self):
        viz = LODVisualization()
        viz.add_object(LODObject("o1", Vector3(0, 0, 0), is_culled=False))
        viz.add_object(LODObject("o2", Vector3(0, 0, 0), is_culled=True))
        viz.add_object(LODObject("o3", Vector3(0, 0, 0), is_culled=True))

        culled = viz.get_culled_objects()
        assert len(culled) == 2

    def test_get_stats(self):
        viz = LODVisualization()
        viz.add_object(LODObject(
            "o1",
            Vector3(0, 0, 0),
            current_lod=LODLevel.LOD0,
            triangle_counts=[1000, 500, 200, 100, 50],
        ))
        viz.add_object(LODObject(
            "o2",
            Vector3(0, 0, 0),
            current_lod=LODLevel.LOD1,
            triangle_counts=[2000, 1000, 400, 200, 100],
        ))

        stats = viz.get_stats()
        assert stats["total_objects"] == 2
        assert stats["lod0_count"] == 1
        assert stats["lod1_count"] == 1
        assert stats["total_triangles"] == 1000 + 1000  # LOD0 + LOD1

    def test_clear_all_objects(self):
        viz = LODVisualization()
        viz.add_object(LODObject("o1", Vector3(0, 0, 0)))
        viz.add_object(LODObject("o2", Vector3(0, 0, 0)))
        viz.clear_all_objects()
        assert viz.object_count == 0


class TestOverdrawHeatmap:
    """Tests for OverdrawHeatmap class."""

    def test_heatmap_creation(self):
        heatmap = OverdrawHeatmap(width=100, height=100)
        assert heatmap.width == 100
        assert heatmap.height == 100

    def test_record_overdraw(self):
        heatmap = OverdrawHeatmap(width=100, height=100)
        heatmap.record_overdraw(50, 50, 3)
        assert heatmap.get_overdraw(50, 50) == 3

    def test_record_overdraw_accumulates(self):
        heatmap = OverdrawHeatmap(width=100, height=100)
        heatmap.record_overdraw(50, 50, 2)
        heatmap.record_overdraw(50, 50, 3)
        assert heatmap.get_overdraw(50, 50) == 5

    def test_set_overdraw(self):
        heatmap = OverdrawHeatmap(width=100, height=100)
        heatmap.set_overdraw(10, 10, 5)
        assert heatmap.get_overdraw(10, 10) == 5

    def test_get_overdraw_out_of_bounds(self):
        heatmap = OverdrawHeatmap(width=100, height=100)
        assert heatmap.get_overdraw(-1, 0) == 0
        assert heatmap.get_overdraw(0, -1) == 0
        assert heatmap.get_overdraw(100, 0) == 0
        assert heatmap.get_overdraw(0, 100) == 0

    def test_max_overdraw(self):
        heatmap = OverdrawHeatmap(width=100, height=100)
        heatmap.set_overdraw(10, 10, 5)
        heatmap.set_overdraw(20, 20, 10)
        assert heatmap._max_overdraw == 10

    def test_clear(self):
        heatmap = OverdrawHeatmap(width=100, height=100)
        heatmap.set_overdraw(50, 50, 10)
        heatmap.clear()
        assert heatmap.get_overdraw(50, 50) == 0
        assert heatmap._max_overdraw == 0

    def test_resize(self):
        heatmap = OverdrawHeatmap(width=100, height=100)
        heatmap.set_overdraw(50, 50, 10)
        heatmap.resize(200, 200)
        assert heatmap.width == 200
        assert heatmap.height == 200
        assert heatmap.get_overdraw(50, 50) == 0  # Data cleared on resize

    def test_get_color_for_overdraw(self):
        heatmap = OverdrawHeatmap()
        color0 = heatmap.get_color_for_overdraw(0)
        color1 = heatmap.get_color_for_overdraw(1)
        color5 = heatmap.get_color_for_overdraw(5)
        color10 = heatmap.get_color_for_overdraw(10)

        # Different overdraw counts should have different colors
        assert color0 != color1
        assert color5 == color10  # Both should be max color (red)

    def test_show_legend(self):
        heatmap = OverdrawHeatmap()
        heatmap.set_show_legend(True)
        heatmap.set_show_legend(False)

    def test_generate_render_data(self):
        heatmap = OverdrawHeatmap(width=100, height=100)
        heatmap.set_overdraw(50, 50, 3)

        data = heatmap.generate_render_data()
        assert data["type"] == "overdraw_heatmap"
        assert data["width"] == 100
        assert data["height"] == 100
        assert data["max_overdraw"] == 3

    def test_generate_render_data_disabled(self):
        heatmap = OverdrawHeatmap()
        heatmap.disable()
        data = heatmap.generate_render_data()
        assert data == {}

    def test_get_stats(self):
        heatmap = OverdrawHeatmap(width=10, height=10)
        heatmap.set_overdraw(0, 0, 2)
        heatmap.set_overdraw(1, 1, 4)

        stats = heatmap.get_stats()
        assert stats["total_pixels"] == 100
        assert stats["max_overdraw"] == 4
        assert stats["non_zero_pixels"] == 2

    def test_get_overdraw_histogram(self):
        heatmap = OverdrawHeatmap(width=10, height=10)
        heatmap.set_overdraw(0, 0, 1)
        heatmap.set_overdraw(1, 0, 1)
        heatmap.set_overdraw(2, 0, 2)

        histogram = heatmap.get_overdraw_histogram()
        assert histogram[0] == 97  # 100 - 3 non-zero
        assert histogram[1] == 2
        assert histogram[2] == 1


class TestRenderDebugger:
    """Tests for RenderDebugger singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        RenderDebugger.reset_instance()
        yield
        RenderDebugger.reset_instance()

    def test_singleton(self):
        d1 = RenderDebugger.get_instance()
        d2 = RenderDebugger.get_instance()
        assert d1 is d2

    def test_enable_disable(self):
        debugger = RenderDebugger.get_instance()
        debugger.enable()
        assert debugger.is_enabled
        debugger.disable()
        assert not debugger.is_enabled

    def test_wireframe_mode(self):
        debugger = RenderDebugger.get_instance()
        debugger.wireframe_mode = WireframeMode.OVERLAY
        assert debugger.wireframe_mode == WireframeMode.OVERLAY

    def test_cycle_wireframe_mode(self):
        debugger = RenderDebugger.get_instance()
        initial = debugger.wireframe_mode
        next_mode = debugger.cycle_wireframe_mode()
        assert next_mode != initial

    def test_subsystems_accessible(self):
        debugger = RenderDebugger.get_instance()
        assert isinstance(debugger.bounding_box_display, BoundingBoxDisplay)
        assert isinstance(debugger.lod_visualization, LODVisualization)
        assert isinstance(debugger.overdraw_heatmap, OverdrawHeatmap)

    def test_show_options(self):
        debugger = RenderDebugger.get_instance()
        debugger.set_show_normals(True)
        debugger.set_show_tangents(True)
        debugger.set_show_uvs(True)
        debugger.set_show_vertex_colors(True)

    def test_force_lod(self):
        debugger = RenderDebugger.get_instance()
        debugger.force_lod(LODLevel.LOD2)
        assert debugger.forced_lod_level == LODLevel.LOD2
        debugger.force_lod(None)
        assert debugger.forced_lod_level is None

    def test_generate_all_draw_commands(self):
        debugger = RenderDebugger.get_instance()
        debugger.bounding_box_display.add_box(
            BoundingBox("b1", BoundingBoxType.AABB, Vector3(0, 0, 0))
        )
        debugger.lod_visualization.add_object(
            LODObject("o1", Vector3(0, 0, 0))
        )

        commands = debugger.generate_all_draw_commands()
        assert len(commands) > 0

    def test_generate_all_disabled(self):
        debugger = RenderDebugger.get_instance()
        debugger.bounding_box_display.add_box(
            BoundingBox("b1", BoundingBoxType.AABB, Vector3(0, 0, 0))
        )
        debugger.disable()

        commands = debugger.generate_all_draw_commands()
        assert len(commands) == 0

    def test_get_render_settings(self):
        debugger = RenderDebugger.get_instance()
        debugger.wireframe_mode = WireframeMode.XRAY
        debugger.set_show_normals(True)
        debugger.force_lod(LODLevel.LOD1)

        settings = debugger.get_render_settings()
        assert settings["wireframe_mode"] == "XRAY"
        assert settings["show_normals"] is True
        assert settings["force_lod"] == "LOD1"

    def test_clear_all(self):
        debugger = RenderDebugger.get_instance()
        debugger.bounding_box_display.add_box(
            BoundingBox("b1", BoundingBoxType.AABB, Vector3(0, 0, 0))
        )
        debugger.lod_visualization.add_object(
            LODObject("o1", Vector3(0, 0, 0))
        )
        debugger.overdraw_heatmap.set_overdraw(0, 0, 5)

        debugger.clear_all()
        assert debugger.bounding_box_display.box_count == 0
        assert debugger.lod_visualization.object_count == 0

    def test_get_stats(self):
        debugger = RenderDebugger.get_instance()
        debugger.bounding_box_display.add_box(
            BoundingBox("b1", BoundingBoxType.AABB, Vector3(0, 0, 0))
        )
        debugger.lod_visualization.add_object(
            LODObject("o1", Vector3(0, 0, 0))
        )

        stats = debugger.get_stats()
        assert stats["bounding_boxes"] == 1
        assert stats["lod_objects"] == 1
        assert "lod_stats" in stats
        assert "overdraw_stats" in stats
