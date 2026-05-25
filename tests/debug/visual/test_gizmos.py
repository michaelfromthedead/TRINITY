"""
Tests for the gizmo system.

Tests cover:
- GizmoType enum
- TransformGizmo for translate/rotate/scale
- BoundsGizmo for AABB/OBB visualization
- LightGizmo for light source visualization
- CameraGizmo for frustum visualization
"""

import math

import pytest


class TestGizmoEnums:
    """Tests for gizmo enumerations."""

    def test_gizmo_types(self):
        """Test GizmoType enum values."""
        from engine.debug.visual import GizmoType

        assert GizmoType.TRANSLATE
        assert GizmoType.ROTATE
        assert GizmoType.SCALE
        assert GizmoType.UNIVERSAL

    def test_gizmo_space(self):
        """Test GizmoSpace enum values."""
        from engine.debug.visual import GizmoSpace

        assert GizmoSpace.WORLD
        assert GizmoSpace.LOCAL
        assert GizmoSpace.VIEW

    def test_gizmo_axis(self):
        """Test GizmoAxis enum values."""
        from engine.debug.visual import GizmoAxis

        assert GizmoAxis.NONE
        assert GizmoAxis.X
        assert GizmoAxis.Y
        assert GizmoAxis.Z
        assert GizmoAxis.XY
        assert GizmoAxis.XZ
        assert GizmoAxis.YZ
        assert GizmoAxis.XYZ
        assert GizmoAxis.VIEW


class TestGizmoStyle:
    """Tests for GizmoStyle configuration."""

    def test_default_style(self):
        """Test default GizmoStyle values."""
        from engine.debug.visual import Color, GizmoStyle

        style = GizmoStyle()
        assert style.x_color == Color.RED
        assert style.y_color == Color.GREEN
        assert style.z_color == Color.BLUE
        assert style.highlight_color == Color.YELLOW
        assert style.inactive_opacity == 0.5
        assert style.axis_thickness == 2.0
        assert style.handle_size == 0.15

    def test_custom_style(self):
        """Test custom GizmoStyle values."""
        from engine.debug.visual import Color, GizmoStyle

        style = GizmoStyle(
            x_color=Color.ORANGE,
            axis_thickness=3.0,
            handle_size=0.2
        )
        assert style.x_color == Color.ORANGE
        assert style.axis_thickness == 3.0
        assert style.handle_size == 0.2


class TestGizmoState:
    """Tests for GizmoState."""

    def test_default_state(self):
        """Test default GizmoState values."""
        from engine.debug.visual import GizmoAxis, GizmoState

        state = GizmoState()
        assert state.active is False
        assert state.hovered_axis == GizmoAxis.NONE
        assert state.dragging is False
        assert state.start_position == (0.0, 0.0, 0.0)
        assert state.current_position == (0.0, 0.0, 0.0)
        assert state.delta == (0.0, 0.0, 0.0)


class TestTransformGizmo:
    """Tests for TransformGizmo."""

    def test_creation(self):
        """Test TransformGizmo creation."""
        from engine.debug.visual import GizmoType, TransformGizmo

        gizmo = TransformGizmo()
        assert gizmo.mode == GizmoType.TRANSLATE
        assert gizmo.enabled is True

    def test_creation_with_mode(self):
        """Test TransformGizmo creation with specific mode."""
        from engine.debug.visual import GizmoType, TransformGizmo

        gizmo = TransformGizmo(mode=GizmoType.ROTATE)
        assert gizmo.mode == GizmoType.ROTATE

    def test_creation_with_space(self):
        """Test TransformGizmo creation with specific space."""
        from engine.debug.visual import GizmoSpace, TransformGizmo

        gizmo = TransformGizmo(space=GizmoSpace.LOCAL)
        assert gizmo.space == GizmoSpace.LOCAL

    def test_set_mode(self):
        """Test changing gizmo mode."""
        from engine.debug.visual import GizmoType, TransformGizmo

        gizmo = TransformGizmo()
        gizmo.mode = GizmoType.SCALE
        assert gizmo.mode == GizmoType.SCALE

    def test_set_target(self):
        """Test setting target transform."""
        from engine.debug.visual import TransformGizmo

        gizmo = TransformGizmo()
        gizmo.set_target(
            position=(1.0, 2.0, 3.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0)
        )

        assert gizmo.position == (1.0, 2.0, 3.0)
        assert gizmo.rotation == (0.0, 0.0, 0.0, 1.0)
        assert gizmo.scale == (1.0, 1.0, 1.0)

    def test_get_result(self):
        """Test getting transform result."""
        from engine.debug.visual import TransformGizmo

        gizmo = TransformGizmo()
        gizmo.set_target(
            position=(5.0, 5.0, 5.0),
            rotation=(0.0, 0.707, 0.0, 0.707),
            scale=(2.0, 2.0, 2.0)
        )

        pos, rot, scale = gizmo.get_result()
        assert pos == (5.0, 5.0, 5.0)
        assert rot == (0.0, 0.707, 0.0, 0.707)
        assert scale == (2.0, 2.0, 2.0)

    def test_enabled_property(self):
        """Test enabled property."""
        from engine.debug.visual import TransformGizmo

        gizmo = TransformGizmo(enabled=True)
        assert gizmo.enabled is True

        gizmo.enabled = False
        assert gizmo.enabled is False

    def test_visible_property(self):
        """Test visible property."""
        from engine.debug.visual import TransformGizmo

        gizmo = TransformGizmo()
        assert gizmo.visible is True

        gizmo.visible = False
        assert gizmo.visible is False

    def test_size_property(self):
        """Test size property."""
        from engine.debug.visual import TransformGizmo

        gizmo = TransformGizmo()
        gizmo.size = 2.0
        assert gizmo.size == 2.0

    def test_size_validation(self):
        """Test size property validation."""
        from engine.debug.visual import TransformGizmo

        gizmo = TransformGizmo()
        with pytest.raises(ValueError):
            gizmo.size = 0

        with pytest.raises(ValueError):
            gizmo.size = -1.0

    def test_set_snap(self):
        """Test setting snap values affects translation behavior."""
        from engine.debug.visual import GizmoAxis, TransformGizmo

        gizmo = TransformGizmo()
        gizmo.set_target(position=(0.0, 0.0, 0.0))
        gizmo.set_snap(translation=1.0, rotation=15.0, scale=0.1)

        # Start a drag operation on X axis
        gizmo.begin_drag(GizmoAxis.X, (0, 0, 10), (0, 0, -1))

        # Update with a small movement (less than snap threshold)
        gizmo.update_drag((0.3, 0, 10), (0, 0, -1))

        # Position should snap to 0 (not 0.3)
        pos, _, _ = gizmo.get_result()
        assert pos[0] == 0.0  # Snapped to nearest 1.0 increment

        # Update with larger movement
        gizmo.update_drag((0.6, 0, 10), (0, 0, -1))

        # Position should snap to 1.0
        pos, _, _ = gizmo.get_result()
        assert pos[0] == 1.0  # Snapped to nearest 1.0 increment

        gizmo.end_drag()

    def test_is_active(self):
        """Test is_active property."""
        from engine.debug.visual import TransformGizmo

        gizmo = TransformGizmo()
        assert gizmo.is_active is False

    def test_hovered_axis(self):
        """Test hovered_axis property."""
        from engine.debug.visual import GizmoAxis, TransformGizmo

        gizmo = TransformGizmo()
        assert gizmo.hovered_axis == GizmoAxis.NONE

    def test_hit_test_disabled(self):
        """Test hit test returns NONE when disabled."""
        from engine.debug.visual import GizmoAxis, TransformGizmo

        gizmo = TransformGizmo(enabled=False)
        result = gizmo.hit_test((0, 0, 0), (0, 0, 1))
        assert result == GizmoAxis.NONE

    def test_callback_registration(self):
        """Test callback registration and invocation."""
        from engine.debug.visual import GizmoAxis, TransformGizmo

        gizmo = TransformGizmo()
        gizmo.set_target(position=(0.0, 0.0, 0.0))
        calls = []

        def callback(value):
            calls.append(value)

        # Register callback and verify it gets called on drag
        gizmo.register_callback(callback)
        gizmo.begin_drag(GizmoAxis.X, (0, 0, 10), (0, 0, -1))
        gizmo.update_drag((5, 0, 10), (0, 0, -1))

        assert len(calls) > 0, "Callback should have been invoked during drag"
        assert isinstance(calls[0], tuple), "Callback should receive position tuple"

        # Unregister and verify no more calls
        initial_call_count = len(calls)
        gizmo.unregister_callback(callback)
        gizmo.update_drag((10, 0, 10), (0, 0, -1))

        assert len(calls) == initial_call_count, "Callback should not be called after unregister"

        gizmo.end_drag()

    def test_render_translate_mode(self):
        """Test rendering in translate mode."""
        from engine.debug.visual import DebugDraw, GizmoType, TransformGizmo

        DebugDraw.clear()
        gizmo = TransformGizmo(mode=GizmoType.TRANSLATE)
        gizmo.set_target(position=(0, 0, 0))
        gizmo.render(None)

        # Should create primitives for arrows and plane handles
        assert DebugDraw.get_primitive_count() > 0
        DebugDraw.clear()

    def test_render_rotate_mode(self):
        """Test rendering in rotate mode."""
        from engine.debug.visual import DebugDraw, GizmoType, TransformGizmo

        DebugDraw.clear()
        gizmo = TransformGizmo(mode=GizmoType.ROTATE)
        gizmo.set_target(position=(0, 0, 0))
        gizmo.render(None)

        # Should create primitives for rotation circles
        assert DebugDraw.get_primitive_count() > 0
        DebugDraw.clear()

    def test_render_scale_mode(self):
        """Test rendering in scale mode."""
        from engine.debug.visual import DebugDraw, GizmoType, TransformGizmo

        DebugDraw.clear()
        gizmo = TransformGizmo(mode=GizmoType.SCALE)
        gizmo.set_target(position=(0, 0, 0))
        gizmo.render(None)

        # Should create primitives for scale handles
        assert DebugDraw.get_primitive_count() > 0
        DebugDraw.clear()

    def test_render_when_disabled(self):
        """Test that rendering is skipped when disabled."""
        from engine.debug.visual import DebugDraw, TransformGizmo

        DebugDraw.clear()
        gizmo = TransformGizmo(enabled=False)
        gizmo.render(None)

        assert DebugDraw.get_primitive_count() == 0
        DebugDraw.clear()

    def test_render_when_invisible(self):
        """Test that rendering is skipped when invisible."""
        from engine.debug.visual import DebugDraw, TransformGizmo

        DebugDraw.clear()
        gizmo = TransformGizmo()
        gizmo.visible = False
        gizmo.render(None)

        assert DebugDraw.get_primitive_count() == 0
        DebugDraw.clear()


class TestBoundsGizmo:
    """Tests for BoundsGizmo."""

    def test_creation(self):
        """Test BoundsGizmo creation."""
        from engine.debug.visual import BoundsGizmo

        gizmo = BoundsGizmo()
        assert gizmo.enabled is True

    def test_set_bounds(self):
        """Test setting bounds."""
        from engine.debug.visual import BoundsGizmo

        gizmo = BoundsGizmo()
        gizmo.set_bounds(
            center=(1.0, 2.0, 3.0),
            extent=(0.5, 1.0, 1.5)
        )

        assert gizmo.center == (1.0, 2.0, 3.0)
        assert gizmo.extent == (0.5, 1.0, 1.5)
        assert gizmo.rotation is None  # AABB

    def test_set_bounds_with_rotation(self):
        """Test setting bounds with rotation (OBB)."""
        from engine.debug.visual import BoundsGizmo

        gizmo = BoundsGizmo()
        rotation = (0.0, 0.707, 0.0, 0.707)
        gizmo.set_bounds(
            center=(0.0, 0.0, 0.0),
            extent=(1.0, 1.0, 1.0),
            rotation=rotation
        )

        assert gizmo.rotation == rotation

    def test_set_from_min_max(self):
        """Test setting bounds from min/max corners."""
        from engine.debug.visual import BoundsGizmo

        gizmo = BoundsGizmo()
        gizmo.set_from_min_max(
            min_point=(-1.0, -2.0, -3.0),
            max_point=(1.0, 2.0, 3.0)
        )

        assert gizmo.center == (0.0, 0.0, 0.0)
        assert gizmo.extent == (1.0, 2.0, 3.0)

    def test_color_property(self):
        """Test color property."""
        from engine.debug.visual import BoundsGizmo, Color

        gizmo = BoundsGizmo()
        gizmo.color = Color.CYAN
        assert gizmo.color == Color.CYAN

    def test_show_axes(self):
        """Test show_axes property."""
        from engine.debug.visual import BoundsGizmo

        gizmo = BoundsGizmo()
        gizmo.show_axes = True
        assert gizmo.show_axes is True

    def test_show_size_labels(self):
        """Test show_size_labels property."""
        from engine.debug.visual import BoundsGizmo

        gizmo = BoundsGizmo()
        gizmo.show_size_labels = True
        assert gizmo.show_size_labels is True

    def test_render(self):
        """Test rendering bounds gizmo."""
        from engine.debug.visual import BoundsGizmo, DebugDraw

        DebugDraw.clear()
        gizmo = BoundsGizmo()
        gizmo.set_bounds(center=(0, 0, 0), extent=(1, 1, 1))
        gizmo.render(None)

        # Should create box primitive
        assert DebugDraw.get_primitive_count() > 0
        DebugDraw.clear()

    def test_render_with_axes_and_labels(self):
        """Test rendering with axes and size labels."""
        from engine.debug.visual import BoundsGizmo, DebugDraw

        DebugDraw.clear()
        gizmo = BoundsGizmo()
        gizmo.set_bounds(center=(0, 0, 0), extent=(1, 1, 1))
        gizmo.show_axes = True
        gizmo.show_size_labels = True
        gizmo.render(None)

        # Should create box, axes (3 arrows), and text
        assert DebugDraw.get_primitive_count() > 3
        DebugDraw.clear()

    def test_hit_test_returns_none(self):
        """Test that BoundsGizmo is not interactive."""
        from engine.debug.visual import BoundsGizmo, GizmoAxis

        gizmo = BoundsGizmo()
        result = gizmo.hit_test((0, 0, 0), (0, 0, 1))
        assert result == GizmoAxis.NONE


class TestLightGizmo:
    """Tests for LightGizmo."""

    def test_creation(self):
        """Test LightGizmo creation."""
        from engine.debug.visual import LightGizmo

        gizmo = LightGizmo()
        assert gizmo.enabled is True
        assert gizmo.light_type == LightGizmo.LightType.POINT

    def test_creation_with_type(self):
        """Test LightGizmo creation with specific type."""
        from engine.debug.visual import LightGizmo

        gizmo = LightGizmo(light_type=LightGizmo.LightType.SPOT)
        assert gizmo.light_type == LightGizmo.LightType.SPOT

    def test_set_point_light(self):
        """Test configuring as point light."""
        from engine.debug.visual import LightGizmo

        gizmo = LightGizmo()
        gizmo.set_point_light(
            position=(5.0, 5.0, 5.0),
            radius=10.0
        )

        assert gizmo.light_type == LightGizmo.LightType.POINT
        assert gizmo.position == (5.0, 5.0, 5.0)
        assert gizmo.radius == 10.0

    def test_set_spot_light(self):
        """Test configuring as spot light."""
        from engine.debug.visual import LightGizmo

        gizmo = LightGizmo()
        gizmo.set_spot_light(
            position=(0.0, 10.0, 0.0),
            direction=(0.0, -1.0, 0.0),
            radius=15.0,
            inner_angle=0.3,
            outer_angle=0.5
        )

        assert gizmo.light_type == LightGizmo.LightType.SPOT
        assert gizmo.position == (0.0, 10.0, 0.0)
        assert gizmo.radius == 15.0
        assert gizmo.inner_angle == 0.3
        assert gizmo.outer_angle == 0.5

    def test_set_directional_light(self):
        """Test configuring as directional light."""
        from engine.debug.visual import LightGizmo

        gizmo = LightGizmo()
        gizmo.set_directional_light(direction=(0.5, -1.0, 0.5))

        assert gizmo.light_type == LightGizmo.LightType.DIRECTIONAL

    def test_radius_validation(self):
        """Test radius property validation."""
        from engine.debug.visual import LightGizmo

        gizmo = LightGizmo()
        with pytest.raises(ValueError):
            gizmo.radius = 0

        with pytest.raises(ValueError):
            gizmo.radius = -5.0

    def test_render_point_light(self):
        """Test rendering point light gizmo."""
        from engine.debug.visual import DebugDraw, LightGizmo

        DebugDraw.clear()
        gizmo = LightGizmo(light_type=LightGizmo.LightType.POINT)
        gizmo.set_point_light(position=(0, 0, 0), radius=5.0)
        gizmo.render(None)

        # Should create sphere and possibly radius indicator
        assert DebugDraw.get_primitive_count() > 0
        DebugDraw.clear()

    def test_render_spot_light(self):
        """Test rendering spot light gizmo."""
        from engine.debug.visual import DebugDraw, LightGizmo

        DebugDraw.clear()
        gizmo = LightGizmo(light_type=LightGizmo.LightType.SPOT)
        gizmo.set_spot_light(
            position=(0, 10, 0),
            direction=(0, -1, 0),
            radius=10.0,
            inner_angle=0.3,
            outer_angle=0.5
        )
        gizmo.render(None)

        # Should create cones and direction arrow
        assert DebugDraw.get_primitive_count() > 0
        DebugDraw.clear()

    def test_render_directional_light(self):
        """Test rendering directional light gizmo."""
        from engine.debug.visual import DebugDraw, LightGizmo

        DebugDraw.clear()
        gizmo = LightGizmo(light_type=LightGizmo.LightType.DIRECTIONAL)
        gizmo.set_directional_light(direction=(0.5, -1.0, 0.0))
        gizmo.render(None)

        # Should create parallel arrows
        assert DebugDraw.get_primitive_count() > 0
        DebugDraw.clear()


class TestCameraGizmo:
    """Tests for CameraGizmo."""

    def test_creation(self):
        """Test CameraGizmo creation."""
        from engine.debug.visual.gizmos import CameraGizmo

        gizmo = CameraGizmo()
        assert gizmo.enabled is True

    def test_set_camera(self):
        """Test setting camera parameters affects rendering."""
        from engine.debug.visual import DebugDraw
        from engine.debug.visual.gizmos import CameraGizmo

        DebugDraw.clear()
        gizmo = CameraGizmo()

        # Render with default settings
        gizmo.render(None)
        default_count = DebugDraw.get_primitive_count()
        DebugDraw.clear()

        # Set custom camera parameters
        gizmo.set_camera(
            position=(0.0, 5.0, 10.0),
            direction=(0.0, 0.0, -1.0),
            up=(0.0, 1.0, 0.0),
            fov_y=math.radians(60),
            aspect=16 / 9,
            near=0.1,
            far=100.0
        )

        # Render with custom settings - should still produce primitives
        gizmo.render(None)
        custom_count = DebugDraw.get_primitive_count()

        # Both renders should produce the same number of frustum primitives
        # (12 lines for frustum + 1 sphere for camera icon = 13 primitives)
        assert custom_count > 0, "Camera gizmo should render primitives after set_camera"
        assert custom_count == default_count, "Primitive count should be consistent"

        DebugDraw.clear()

    def test_render(self):
        """Test rendering camera gizmo."""
        from engine.debug.visual import DebugDraw
        from engine.debug.visual.gizmos import CameraGizmo

        DebugDraw.clear()
        gizmo = CameraGizmo()
        gizmo.set_camera(
            position=(0, 0, 0),
            direction=(0, 0, -1),
            up=(0, 1, 0),
            fov_y=math.radians(60),
            aspect=16 / 9,
            near=0.1,
            far=10.0
        )
        gizmo.render(None)

        # Should create frustum lines and camera icon
        assert DebugDraw.get_primitive_count() > 0
        DebugDraw.clear()


class TestGizmoCallbacks:
    """Tests for gizmo callback system."""

    def test_register_and_trigger_callback(self):
        """Test registering and triggering callbacks."""
        from engine.debug.visual import GizmoAxis, TransformGizmo

        gizmo = TransformGizmo()
        calls = []

        def callback(value):
            calls.append(value)

        gizmo.register_callback(callback)

        # Simulate drag operation
        gizmo.begin_drag(GizmoAxis.X, (0, 0, 10), (0, 0, -1))
        gizmo.update_drag((5, 0, 10), (0, 0, -1))

        # Callback should have been called
        assert len(calls) > 0

    def test_unregister_callback(self):
        """Test unregistering callbacks."""
        from engine.debug.visual import TransformGizmo

        gizmo = TransformGizmo()
        calls = []

        def callback(value):
            calls.append(value)

        gizmo.register_callback(callback)
        result = gizmo.unregister_callback(callback)
        assert result is True

        # Trying to unregister again should return False
        result = gizmo.unregister_callback(callback)
        assert result is False

    def test_callback_error_handling(self):
        """Test that callback errors don't crash the gizmo."""
        from engine.debug.visual import GizmoAxis, TransformGizmo

        gizmo = TransformGizmo()
        calls = []

        def bad_callback(value):
            raise RuntimeError("Callback error")

        def good_callback(value):
            calls.append(value)

        gizmo.register_callback(bad_callback)
        gizmo.register_callback(good_callback)

        # Should not raise, should continue to good callback
        gizmo.begin_drag(GizmoAxis.X, (0, 0, 10), (0, 0, -1))
        gizmo.update_drag((5, 0, 10), (0, 0, -1))

        # Good callback should still have been called
        assert len(calls) > 0
