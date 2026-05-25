"""Tests for debug draw primitives, batching, and persistence."""

import pytest
import time
from engine.tooling.debug.debug_draw import (
    DebugDraw,
    DebugCategory,
    DebugColor,
    DepthTestMode,
    DrawPrimitive,
    DrawCommand,
    DebugDrawBatch,
    Vector3,
    Quaternion,
    debug_draw,
)


class TestVector3:
    """Tests for Vector3 class."""

    def test_vector3_creation(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vector3_default(self):
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vector3_addition(self):
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1 + v2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_vector3_subtraction(self):
        v1 = Vector3(4, 5, 6)
        v2 = Vector3(1, 2, 3)
        result = v1 - v2
        assert result.x == 3
        assert result.y == 3
        assert result.z == 3

    def test_vector3_multiplication(self):
        v = Vector3(1, 2, 3)
        result = v * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_vector3_length(self):
        v = Vector3(3, 4, 0)
        assert v.length() == 5.0

    def test_vector3_normalized(self):
        v = Vector3(0, 10, 0)
        n = v.normalized()
        assert abs(n.x) < 0.001
        assert abs(n.y - 1.0) < 0.001
        assert abs(n.z) < 0.001

    def test_vector3_normalized_zero(self):
        v = Vector3(0, 0, 0)
        n = v.normalized()
        assert n.x == 0
        assert n.y == 0
        assert n.z == 0

    def test_vector3_cross(self):
        x = Vector3(1, 0, 0)
        y = Vector3(0, 1, 0)
        z = x.cross(y)
        assert abs(z.z - 1.0) < 0.001

    def test_vector3_to_tuple(self):
        v = Vector3(1, 2, 3)
        assert v.to_tuple() == (1, 2, 3)


class TestQuaternion:
    """Tests for Quaternion class."""

    def test_quaternion_identity(self):
        q = Quaternion.identity()
        assert q.x == 0
        assert q.y == 0
        assert q.z == 0
        assert q.w == 1

    def test_quaternion_from_axis_angle(self):
        axis = Vector3(0, 1, 0)
        q = Quaternion.from_axis_angle(axis, 0)
        assert abs(q.w - 1.0) < 0.001


class TestDebugColor:
    """Tests for DebugColor class."""

    def test_predefined_colors(self):
        assert DebugColor.RED == (1.0, 0.0, 0.0, 1.0)
        assert DebugColor.GREEN == (0.0, 1.0, 0.0, 1.0)
        assert DebugColor.BLUE == (0.0, 0.0, 1.0, 1.0)

    def test_from_category(self):
        color = DebugColor.from_category(DebugCategory.PHYSICS)
        assert color == DebugColor.PHYSICS_COLOR

    def test_from_category_unknown(self):
        # Custom category should return white
        color = DebugColor.from_category(DebugCategory.CUSTOM)
        assert color == DebugColor.GRAY


class TestDebugDrawBatch:
    """Tests for DebugDrawBatch class."""

    def test_batch_creation(self):
        batch = DebugDrawBatch()
        assert batch.count == 0

    def test_batch_add(self):
        batch = DebugDrawBatch()
        cmd = DrawCommand(
            primitive=DrawPrimitive.LINE,
            category=DebugCategory.PHYSICS,
            color=DebugColor.RED,
            depth_mode=DepthTestMode.DISABLED,
            persistent=False,
            lifetime=0,
            creation_time=time.time(),
        )
        batch.add(cmd)
        assert batch.count == 1

    def test_batch_clear(self):
        batch = DebugDrawBatch()
        cmd = DrawCommand(
            primitive=DrawPrimitive.LINE,
            category=DebugCategory.PHYSICS,
            color=DebugColor.RED,
            depth_mode=DepthTestMode.DISABLED,
            persistent=False,
            lifetime=0,
            creation_time=time.time(),
        )
        batch.add(cmd)
        batch.clear()
        assert batch.count == 0

    def test_batch_remove_expired(self):
        batch = DebugDrawBatch()
        # Add expired command
        cmd = DrawCommand(
            primitive=DrawPrimitive.LINE,
            category=DebugCategory.PHYSICS,
            color=DebugColor.RED,
            depth_mode=DepthTestMode.DISABLED,
            persistent=False,
            lifetime=0.1,
            creation_time=time.time() - 1.0,  # Created 1 second ago
        )
        batch.add(cmd)
        removed = batch.remove_expired(time.time())
        assert removed == 1
        assert batch.count == 0


class TestDebugDraw:
    """Tests for DebugDraw singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        DebugDraw.reset_instance()
        yield
        DebugDraw.reset_instance()

    def test_singleton(self):
        debug1 = DebugDraw.get_instance()
        debug2 = DebugDraw.get_instance()
        assert debug1 is debug2

    def test_enable_disable(self):
        debug = DebugDraw.get_instance()
        debug.enable()
        assert debug.is_enabled
        debug.disable()
        assert not debug.is_enabled

    def test_category_enable_disable(self):
        debug = DebugDraw.get_instance()
        debug.enable_category(DebugCategory.PHYSICS)
        assert debug.is_category_enabled(DebugCategory.PHYSICS)
        debug.disable_category(DebugCategory.PHYSICS)
        assert not debug.is_category_enabled(DebugCategory.PHYSICS)

    def test_draw_line(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_line(
            start=Vector3(0, 0, 0),
            end=Vector3(1, 1, 1),
            color=DebugColor.RED,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.LINE

    def test_draw_line_disabled(self):
        debug = DebugDraw.get_instance()
        debug.disable()
        cmd = debug.draw_line(
            start=Vector3(0, 0, 0),
            end=Vector3(1, 1, 1),
        )
        assert cmd is None

    def test_draw_line_category_disabled(self):
        debug = DebugDraw.get_instance()
        debug.disable_category(DebugCategory.PHYSICS)
        cmd = debug.draw_line(
            start=Vector3(0, 0, 0),
            end=Vector3(1, 1, 1),
            category=DebugCategory.PHYSICS,
        )
        assert cmd is None

    def test_draw_sphere(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_sphere(
            center=Vector3(0, 0, 0),
            radius=1.0,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.SPHERE

    def test_draw_box(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_box(
            center=Vector3(0, 0, 0),
            extents=Vector3(1, 1, 1),
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.BOX

    def test_draw_arrow(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_arrow(
            start=Vector3(0, 0, 0),
            end=Vector3(0, 1, 0),
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.ARROW

    def test_draw_text(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_text(
            position=Vector3(0, 0, 0),
            text="Hello",
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.TEXT
        assert cmd.data["text"] == "Hello"

    def test_draw_text_2d(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_text_2d(
            x=100,
            y=100,
            text="Screen Text",
        )
        assert cmd is not None
        assert cmd.data["is_2d"] is True

    def test_draw_plane(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_plane(
            center=Vector3(0, 0, 0),
            normal=Vector3(0, 1, 0),
            size=5.0,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.PLANE

    def test_draw_circle(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_circle(
            center=Vector3(0, 0, 0),
            radius=1.0,
            normal=Vector3(0, 1, 0),
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.CIRCLE

    def test_draw_cylinder(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_cylinder(
            start=Vector3(0, 0, 0),
            end=Vector3(0, 2, 0),
            radius=0.5,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.CYLINDER

    def test_draw_capsule(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_capsule(
            start=Vector3(0, 0, 0),
            end=Vector3(0, 2, 0),
            radius=0.5,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.CAPSULE

    def test_draw_aabb(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_aabb(
            min_point=Vector3(-1, -1, -1),
            max_point=Vector3(1, 1, 1),
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.BOX

    def test_draw_ray(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_ray(
            origin=Vector3(0, 0, 0),
            direction=Vector3(0, 1, 0),
            length=10.0,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.LINE

    def test_draw_lines(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        points = [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0)]
        cmds = debug.draw_lines(points)
        assert len(cmds) == 2  # 3 points = 2 lines

    def test_draw_lines_closed(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        points = [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0)]
        cmds = debug.draw_lines(points, closed=True)
        assert len(cmds) == 3  # Closed polygon = 3 lines

    def test_draw_direction(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_direction(
            position=Vector3(0, 0, 0),
            direction=Vector3(0, 1, 0),
            length=2.0,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.ARROW

    def test_draw_axis(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_axis(
            position=Vector3(0, 0, 0),
            size=1.0,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.AXIS

    def test_draw_grid(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_grid(
            center=Vector3(0, 0, 0),
            size=10.0,
            divisions=10,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.GRID

    def test_draw_frustum(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_frustum(
            position=Vector3(0, 0, 0),
            forward=Vector3(0, 0, 1),
            up=Vector3(0, 1, 0),
            fov=60.0,
            aspect=16/9,
            near=0.1,
            far=100.0,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.FRUSTUM

    def test_draw_polygon(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        vertices = [Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0.5, 1, 0)]
        cmd = debug.draw_polygon(vertices)
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.POLYGON

    def test_draw_triangle(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_triangle(
            v0=Vector3(0, 0, 0),
            v1=Vector3(1, 0, 0),
            v2=Vector3(0.5, 1, 0),
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.TRIANGLE

    def test_draw_point(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_point(
            position=Vector3(0, 0, 0),
            size=5.0,
        )
        assert cmd is not None
        assert cmd.primitive == DrawPrimitive.POINT

    def test_persistent_draw(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_line(
            start=Vector3(0, 0, 0),
            end=Vector3(1, 1, 1),
            persistent=True,
        )
        assert cmd is not None
        assert cmd.persistent is True
        # Should be in persistent batch
        assert len(debug.get_persistent_commands()) == 1

    def test_lifetime_draw(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        cmd = debug.draw_sphere(
            center=Vector3(0, 0, 0),
            radius=1.0,
            lifetime=5.0,
        )
        assert cmd is not None
        assert cmd.lifetime == 5.0

    def test_clear_all(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        debug.draw_line(Vector3(0, 0, 0), Vector3(1, 1, 1))
        debug.draw_sphere(Vector3(0, 0, 0), 1.0, persistent=True)
        debug.clear_all()
        assert debug.total_command_count == 0

    def test_clear_category(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        debug.draw_line(Vector3(0, 0, 0), Vector3(1, 1, 1), category=DebugCategory.PHYSICS)
        debug.draw_sphere(Vector3(0, 0, 0), 1.0, category=DebugCategory.AI)
        debug.clear_category(DebugCategory.PHYSICS)
        # Should only have AI command left
        commands = debug.get_commands_by_category(DebugCategory.AI)
        assert len(commands) == 1

    def test_begin_frame_clears_immediate(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        debug.draw_line(Vector3(0, 0, 0), Vector3(1, 1, 1))
        assert len(debug.get_immediate_commands()) == 1
        debug.begin_frame()
        assert len(debug.get_immediate_commands()) == 0

    def test_depth_modes(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()

        debug.set_default_depth_mode(DepthTestMode.ENABLED)
        cmd = debug.draw_line(Vector3(0, 0, 0), Vector3(1, 1, 1))
        assert cmd.depth_mode == DepthTestMode.ENABLED

        cmd = debug.draw_line(
            Vector3(0, 0, 0),
            Vector3(1, 1, 1),
            depth_mode=DepthTestMode.XRAY,
        )
        assert cmd.depth_mode == DepthTestMode.XRAY

    def test_frame_count(self):
        debug = DebugDraw.get_instance()
        debug.begin_frame()
        frame1 = debug.frame_count
        debug.begin_frame()
        frame2 = debug.frame_count
        assert frame2 == frame1 + 1


class TestDebugDrawDecorator:
    """Tests for debug_draw decorator."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        DebugDraw.reset_instance()
        yield
        DebugDraw.reset_instance()

    def test_decorator_basic(self):
        @debug_draw(category=DebugCategory.PHYSICS)
        def draw_physics_viz(debug: DebugDraw):
            debug.draw_sphere(Vector3(0, 0, 0), 1.0)
            return True

        result = draw_physics_viz()
        assert result is True

    def test_decorator_disabled(self):
        @debug_draw(category=DebugCategory.PHYSICS, enabled=False)
        def draw_disabled(debug: DebugDraw):
            return True

        result = draw_disabled()
        assert result is None

    def test_decorator_category_disabled(self):
        debug = DebugDraw.get_instance()
        debug.disable_category(DebugCategory.AI)

        @debug_draw(category=DebugCategory.AI)
        def draw_ai_viz(debug: DebugDraw):
            return True

        result = draw_ai_viz()
        assert result is None

    def test_decorator_preserves_attributes(self):
        @debug_draw(category=DebugCategory.PHYSICS)
        def my_draw_func(debug: DebugDraw):
            """My documentation."""
            pass

        assert my_draw_func.__name__ == "my_draw_func"
        assert "documentation" in my_draw_func.__doc__


class TestDrawCommandExpiration:
    """Tests for draw command lifetime and expiration."""

    def test_immediate_command_expires(self):
        cmd = DrawCommand(
            primitive=DrawPrimitive.LINE,
            category=DebugCategory.PHYSICS,
            color=DebugColor.RED,
            depth_mode=DepthTestMode.DISABLED,
            persistent=False,
            lifetime=0,
            creation_time=time.time(),
        )
        # Immediate commands with 0 lifetime should expire
        assert cmd.is_expired(time.time()) is True

    def test_persistent_command_no_expire(self):
        cmd = DrawCommand(
            primitive=DrawPrimitive.LINE,
            category=DebugCategory.PHYSICS,
            color=DebugColor.RED,
            depth_mode=DepthTestMode.DISABLED,
            persistent=True,
            lifetime=0,
            creation_time=time.time(),
        )
        # Persistent with no lifetime should not expire
        assert cmd.is_expired(time.time()) is False

    def test_timed_command_expires(self):
        creation = time.time() - 2.0  # Created 2 seconds ago
        cmd = DrawCommand(
            primitive=DrawPrimitive.LINE,
            category=DebugCategory.PHYSICS,
            color=DebugColor.RED,
            depth_mode=DepthTestMode.DISABLED,
            persistent=False,
            lifetime=1.0,  # 1 second lifetime
            creation_time=creation,
        )
        # Should be expired
        assert cmd.is_expired(time.time()) is True

    def test_timed_command_not_expired(self):
        cmd = DrawCommand(
            primitive=DrawPrimitive.LINE,
            category=DebugCategory.PHYSICS,
            color=DebugColor.RED,
            depth_mode=DepthTestMode.DISABLED,
            persistent=False,
            lifetime=10.0,  # 10 second lifetime
            creation_time=time.time(),
        )
        # Should not be expired yet
        assert cmd.is_expired(time.time()) is False
