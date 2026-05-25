"""
Tests for the debug draw primitives system.

Tests cover:
- Color dataclass and predefined colors
- DrawOptions validation
- DebugDraw static methods for all primitive types
- Internal batch storage and expiration
- Enable/disable functionality
"""

import math
import time

import pytest


class TestColor:
    """Tests for the Color dataclass."""

    def test_color_creation(self):
        """Test basic color creation with RGBA values."""
        from engine.debug.visual import Color

        color = Color(0.5, 0.6, 0.7, 0.8)
        assert color.r == 0.5
        assert color.g == 0.6
        assert color.b == 0.7
        assert color.a == 0.8

    def test_color_default_alpha(self):
        """Test that alpha defaults to 1.0."""
        from engine.debug.visual import Color

        color = Color(0.5, 0.5, 0.5)
        assert color.a == 1.0

    def test_color_validation_out_of_range(self):
        """Test that color components must be in [0, 1] range."""
        from engine.debug.visual import Color

        with pytest.raises(ValueError):
            Color(1.5, 0.5, 0.5)  # r > 1.0

        with pytest.raises(ValueError):
            Color(0.5, -0.1, 0.5)  # g < 0

        with pytest.raises(ValueError):
            Color(0.5, 0.5, 0.5, 1.5)  # a > 1.0

    def test_predefined_colors(self):
        """Test predefined color constants."""
        from engine.debug.visual import Color

        assert Color.RED.r == 1.0
        assert Color.RED.g == 0.0
        assert Color.RED.b == 0.0

        assert Color.GREEN.r == 0.0
        assert Color.GREEN.g == 1.0
        assert Color.GREEN.b == 0.0

        assert Color.BLUE.r == 0.0
        assert Color.BLUE.g == 0.0
        assert Color.BLUE.b == 1.0

        assert Color.WHITE.r == 1.0
        assert Color.WHITE.g == 1.0
        assert Color.WHITE.b == 1.0

        assert Color.BLACK.r == 0.0
        assert Color.BLACK.g == 0.0
        assert Color.BLACK.b == 0.0

        assert Color.YELLOW.r == 1.0
        assert Color.YELLOW.g == 1.0
        assert Color.YELLOW.b == 0.0

    def test_color_with_alpha(self):
        """Test creating new color with modified alpha."""
        from engine.debug.visual import Color

        original = Color.RED
        modified = original.with_alpha(0.5)

        assert modified.r == 1.0
        assert modified.g == 0.0
        assert modified.b == 0.0
        assert modified.a == 0.5
        # Original should be unchanged
        assert original.a == 1.0

    def test_color_to_tuple(self):
        """Test converting color to tuple."""
        from engine.debug.visual import Color

        color = Color(0.1, 0.2, 0.3, 0.4)
        t = color.to_tuple()
        assert t == (0.1, 0.2, 0.3, 0.4)

    def test_color_to_hex(self):
        """Test converting color to hex string."""
        from engine.debug.visual import Color

        assert Color.RED.to_hex() == "#FF0000FF"
        assert Color.GREEN.to_hex() == "#00FF00FF"
        assert Color.BLUE.to_hex() == "#0000FFFF"
        assert Color.WHITE.to_hex() == "#FFFFFFFF"

    def test_color_from_hex(self):
        """Test creating color from hex string."""
        from engine.debug.visual import Color

        # 6-digit hex
        color = Color.from_hex("#FF0000")
        assert color.r == 1.0
        assert color.g == 0.0
        assert color.b == 0.0
        assert color.a == 1.0

        # 8-digit hex with alpha
        color = Color.from_hex("#FF000080")
        assert color.r == 1.0
        assert abs(color.a - 0.502) < 0.01  # ~50% alpha

        # Without hash
        color = Color.from_hex("00FF00")
        assert color.g == 1.0

    def test_color_from_hex_short_format(self):
        """Test creating color from short hex format."""
        from engine.debug.visual import Color

        # 3-digit hex
        color = Color.from_hex("#F00")
        assert color.r == 1.0
        assert color.g == 0.0
        assert color.b == 0.0

        # 4-digit hex with alpha
        color = Color.from_hex("#F008")
        assert color.r == 1.0
        assert abs(color.a - 0.533) < 0.01

    def test_color_immutability(self):
        """Test that Color is immutable (frozen dataclass)."""
        from engine.debug.visual import Color

        color = Color(0.5, 0.5, 0.5)
        with pytest.raises(AttributeError):
            color.r = 0.8


class TestDrawOptions:
    """Tests for DrawOptions dataclass."""

    def test_draw_options_defaults(self):
        """Test default DrawOptions values."""
        from engine.debug.visual import Color, DrawOptions

        opts = DrawOptions()
        assert opts.color == Color.WHITE
        assert opts.duration == 0.0
        assert opts.thickness == 1.0
        assert opts.depth_test is True
        assert opts.wireframe is True

    def test_draw_options_custom_values(self):
        """Test DrawOptions with custom values."""
        from engine.debug.visual import Color, DrawOptions

        opts = DrawOptions(
            color=Color.RED,
            duration=5.0,
            thickness=3.0,
            depth_test=False,
            wireframe=False
        )
        assert opts.color == Color.RED
        assert opts.duration == 5.0
        assert opts.thickness == 3.0
        assert opts.depth_test is False
        assert opts.wireframe is False

    def test_draw_options_validation(self):
        """Test DrawOptions validation."""
        from engine.debug.visual import DrawOptions

        with pytest.raises(ValueError):
            DrawOptions(duration=-1.0)

        with pytest.raises(ValueError):
            DrawOptions(thickness=0.0)

        with pytest.raises(ValueError):
            DrawOptions(thickness=-1.0)


class TestDebugDraw:
    """Tests for the DebugDraw static class."""

    @pytest.fixture(autouse=True)
    def reset_debug_draw(self):
        """Reset DebugDraw state before and after each test."""
        from engine.debug.visual import DebugDraw

        DebugDraw.clear()
        DebugDraw.set_enabled(True)
        yield
        DebugDraw.clear()
        DebugDraw.set_enabled(True)

    def test_line_primitive(self):
        """Test drawing a line."""
        from engine.debug.visual import Color, DebugDraw

        primitive = DebugDraw.line(
            start=(0, 0, 0),
            end=(10, 0, 0),
            color=Color.RED
        )

        assert primitive is not None
        assert primitive.data["start"] == (0, 0, 0)
        assert primitive.data["end"] == (10, 0, 0)
        assert primitive.options.color == Color.RED
        assert DebugDraw.get_primitive_count() == 1

    def test_arrow_primitive(self):
        """Test drawing an arrow."""
        from engine.debug.visual import Color, DebugDraw

        primitive = DebugDraw.arrow(
            origin=(0, 0, 0),
            direction=(1, 0, 0),
            color=Color.GREEN,
            length=5.0
        )

        assert primitive is not None
        assert primitive.data["origin"] == (0, 0, 0)
        assert primitive.data["length"] == 5.0
        # Direction should be normalized
        dir_vec = primitive.data["direction"]
        length = math.sqrt(sum(c**2 for c in dir_vec))
        assert abs(length - 1.0) < 1e-6

    def test_point_primitive(self):
        """Test drawing a point."""
        from engine.debug.visual import Color, DebugDraw

        primitive = DebugDraw.point(
            position=(5, 5, 5),
            size=10.0,
            color=Color.YELLOW
        )

        assert primitive is not None
        assert primitive.data["position"] == (5, 5, 5)
        assert primitive.data["size"] == 10.0

    def test_sphere_primitive(self):
        """Test drawing a sphere."""
        from engine.debug.visual import Color, DebugDraw

        primitive = DebugDraw.sphere(
            center=(0, 5, 0),
            radius=2.0,
            color=Color.BLUE,
            segments=24
        )

        assert primitive is not None
        assert primitive.data["center"] == (0, 5, 0)
        assert primitive.data["radius"] == 2.0
        assert primitive.data["segments"] == 24

    def test_box_primitive(self):
        """Test drawing a box."""
        from engine.debug.visual import Color, DebugDraw

        primitive = DebugDraw.box(
            center=(0, 0, 0),
            extent=(1, 2, 3),
            color=Color.CYAN
        )

        assert primitive is not None
        assert primitive.data["center"] == (0, 0, 0)
        assert primitive.data["extent"] == (1, 2, 3)
        # Default identity rotation
        assert primitive.data["rotation"] == (0.0, 0.0, 0.0, 1.0)

    def test_box_with_rotation(self):
        """Test drawing a rotated box."""
        from engine.debug.visual import Color, DebugDraw

        rotation = (0.0, 0.707, 0.0, 0.707)  # 90 degrees around Y
        primitive = DebugDraw.box(
            center=(0, 0, 0),
            extent=(1, 1, 1),
            color=Color.MAGENTA,
            rotation=rotation
        )

        assert primitive.data["rotation"] == rotation

    def test_capsule_primitive(self):
        """Test drawing a capsule."""
        from engine.debug.visual import Color, DebugDraw

        primitive = DebugDraw.capsule(
            start=(0, 0, 0),
            end=(0, 5, 0),
            radius=1.0,
            color=Color.ORANGE
        )

        assert primitive is not None
        assert primitive.data["start"] == (0, 0, 0)
        assert primitive.data["end"] == (0, 5, 0)
        assert primitive.data["radius"] == 1.0

    def test_cylinder_primitive(self):
        """Test drawing a cylinder."""
        from engine.debug.visual import DebugDraw

        primitive = DebugDraw.cylinder(
            start=(0, 0, 0),
            end=(0, 10, 0),
            radius=2.0
        )

        assert primitive is not None
        assert primitive.data["start"] == (0, 0, 0)
        assert primitive.data["end"] == (0, 10, 0)
        assert primitive.data["radius"] == 2.0

    def test_cone_primitive(self):
        """Test drawing a cone."""
        from engine.debug.visual import DebugDraw

        primitive = DebugDraw.cone(
            apex=(0, 0, 0),
            direction=(0, 1, 0),
            height=5.0,
            angle=0.5  # radians
        )

        assert primitive is not None
        assert primitive.data["apex"] == (0, 0, 0)
        assert primitive.data["height"] == 5.0
        assert primitive.data["angle"] == 0.5

    def test_screen_text(self):
        """Test drawing screen-space text."""
        from engine.debug.visual import Color, DebugDraw

        primitive = DebugDraw.screen_text(
            text="Debug Info",
            x=10,
            y=20,
            color=Color.WHITE,
            scale=1.5
        )

        assert primitive is not None
        assert primitive.data["text"] == "Debug Info"
        assert primitive.data["x"] == 10
        assert primitive.data["y"] == 20
        assert primitive.data["scale"] == 1.5

    def test_world_text(self):
        """Test drawing world-space text."""
        from engine.debug.visual import Color, DebugDraw

        primitive = DebugDraw.world_text(
            text="Entity Name",
            position=(5, 10, 5),
            color=Color.GREEN,
            face_camera=True
        )

        assert primitive is not None
        assert primitive.data["text"] == "Entity Name"
        assert primitive.data["position"] == (5, 10, 5)
        assert primitive.data["face_camera"] == 1

    def test_coordinate_axes(self):
        """Test drawing coordinate axes helper."""
        from engine.debug.visual import DebugDraw

        DebugDraw.coordinate_axes(origin=(0, 0, 0), size=2.0)

        # Should create 3 arrows (X, Y, Z)
        assert DebugDraw.get_primitive_count() == 3

    def test_circle_primitive(self):
        """Test drawing a circle."""
        from engine.debug.visual import DebugDraw

        primitive = DebugDraw.circle(
            center=(0, 0, 0),
            normal=(0, 1, 0),
            radius=3.0,
            segments=32
        )

        assert primitive is not None
        assert primitive.data["center"] == (0, 0, 0)
        assert primitive.data["radius"] == 3.0
        assert primitive.data["segments"] == 32

    def test_triangle_primitive(self):
        """Test drawing a triangle."""
        from engine.debug.visual import DebugDraw

        primitive = DebugDraw.triangle(
            v0=(0, 0, 0),
            v1=(1, 0, 0),
            v2=(0.5, 1, 0)
        )

        assert primitive is not None
        assert primitive.data["v0"] == (0, 0, 0)
        assert primitive.data["v1"] == (1, 0, 0)
        assert primitive.data["v2"] == (0.5, 1, 0)

    def test_plane_primitive(self):
        """Test drawing a plane."""
        from engine.debug.visual import DebugDraw

        primitive = DebugDraw.plane(
            center=(0, 0, 0),
            normal=(0, 1, 0),
            size=5.0
        )

        assert primitive is not None
        assert primitive.data["center"] == (0, 0, 0)
        assert primitive.data["size"] == 5.0


class TestDebugDrawBatch:
    """Tests for the DebugDrawBatch internal storage."""

    @pytest.fixture(autouse=True)
    def reset_debug_draw(self):
        """Reset DebugDraw state before and after each test."""
        from engine.debug.visual import DebugDraw

        DebugDraw.clear()
        DebugDraw.set_enabled(True)
        yield
        DebugDraw.clear()

    def test_single_frame_primitives(self):
        """Test that primitives with duration=0 last one frame."""
        from engine.debug.visual import Color, DebugDraw

        DebugDraw.line((0, 0, 0), (1, 0, 0), Color.RED, duration=0.0)
        assert DebugDraw.get_primitive_count() == 1

        batch = DebugDraw.get_batch()
        assert batch.frame_count == 1
        assert batch.persistent_count == 0

        # End frame should clear single-frame primitives
        DebugDraw.end_frame()
        assert DebugDraw.get_primitive_count() == 0

    def test_persistent_primitives(self):
        """Test that primitives with duration > 0 persist."""
        from engine.debug.visual import Color, DebugDraw

        # Use a fixed time for testing
        test_time = 1000.0
        DebugDraw.set_time_provider(lambda: test_time)

        DebugDraw.line((0, 0, 0), (1, 0, 0), Color.RED, duration=5.0)
        assert DebugDraw.get_primitive_count() == 1

        batch = DebugDraw.get_batch()
        assert batch.frame_count == 0
        assert batch.persistent_count == 1

        # End frame should NOT clear persistent primitives
        DebugDraw.end_frame()
        assert DebugDraw.get_primitive_count() == 1

        # Update should remove expired primitives
        DebugDraw.set_time_provider(lambda: test_time + 6.0)  # 6 seconds later
        DebugDraw.update()
        assert DebugDraw.get_primitive_count() == 0

    def test_batch_get_all(self):
        """Test getting all primitives from batch."""
        from engine.debug.visual import Color, DebugDraw

        DebugDraw.set_time_provider(lambda: 0.0)

        # Add both frame and persistent primitives
        DebugDraw.line((0, 0, 0), (1, 0, 0), Color.RED, duration=0.0)
        DebugDraw.line((0, 0, 0), (0, 1, 0), Color.GREEN, duration=5.0)

        batch = DebugDraw.get_batch()
        all_primitives = batch.get_all()
        assert len(all_primitives) == 2

    def test_batch_clear(self):
        """Test clearing all primitives from batch."""
        from engine.debug.visual import Color, DebugDraw

        DebugDraw.line((0, 0, 0), (1, 0, 0), Color.RED)
        DebugDraw.line((0, 0, 0), (0, 1, 0), Color.GREEN, duration=5.0)
        assert DebugDraw.get_primitive_count() == 2

        DebugDraw.clear()
        assert DebugDraw.get_primitive_count() == 0


class TestDebugDrawEnableDisable:
    """Tests for DebugDraw enable/disable functionality."""

    @pytest.fixture(autouse=True)
    def reset_debug_draw(self):
        """Reset DebugDraw state before and after each test."""
        from engine.debug.visual import DebugDraw

        DebugDraw.clear()
        DebugDraw.set_enabled(True)
        yield
        DebugDraw.clear()
        DebugDraw.set_enabled(True)

    def test_disable_prevents_drawing(self):
        """Test that disabling prevents new primitives."""
        from engine.debug.visual import Color, DebugDraw

        DebugDraw.set_enabled(False)
        assert not DebugDraw.is_enabled()

        result = DebugDraw.line((0, 0, 0), (1, 0, 0), Color.RED)
        assert result is None
        assert DebugDraw.get_primitive_count() == 0

    def test_reenable_allows_drawing(self):
        """Test that re-enabling allows drawing."""
        from engine.debug.visual import Color, DebugDraw

        DebugDraw.set_enabled(False)
        DebugDraw.set_enabled(True)
        assert DebugDraw.is_enabled()

        result = DebugDraw.line((0, 0, 0), (1, 0, 0), Color.RED)
        assert result is not None
        assert DebugDraw.get_primitive_count() == 1


class TestDebugDrawFrustum:
    """Tests for frustum drawing."""

    @pytest.fixture(autouse=True)
    def reset_debug_draw(self):
        """Reset DebugDraw state before and after each test."""
        from engine.debug.visual import DebugDraw

        DebugDraw.clear()
        DebugDraw.set_enabled(True)
        yield
        DebugDraw.clear()

    def test_frustum_draws_lines(self):
        """Test that frustum creates expected number of lines."""
        from engine.debug.visual import Color, DebugDraw

        DebugDraw.frustum(
            origin=(0, 0, 0),
            direction=(0, 0, -1),
            up=(0, 1, 0),
            fov_y=math.radians(60),
            aspect=16 / 9,
            near=0.1,
            far=100.0,
            color=Color.CYAN
        )

        # Frustum should create 12 lines (4 near, 4 far, 4 connecting)
        assert DebugDraw.get_primitive_count() == 12


class TestDebugDrawConfig:
    """Tests for debug draw configuration."""

    @pytest.fixture(autouse=True)
    def reset_debug_draw(self):
        """Reset DebugDraw state before and after each test."""
        from engine.debug.visual import DebugDraw

        DebugDraw.clear()
        DebugDraw.set_enabled(True)
        yield
        DebugDraw.clear()

    def test_config_exists(self):
        """Test that configuration module is accessible."""
        from engine.debug.visual import DEBUG_DRAW_CONFIG, DebugDrawConfig

        assert isinstance(DEBUG_DRAW_CONFIG, DebugDrawConfig)

    def test_config_has_expected_fields(self):
        """Test that config has all expected fields."""
        from engine.debug.visual import DEBUG_DRAW_CONFIG

        assert hasattr(DEBUG_DRAW_CONFIG, 'vector_normalize_epsilon')
        assert hasattr(DEBUG_DRAW_CONFIG, 'max_primitives')
        assert hasattr(DEBUG_DRAW_CONFIG, 'primitive_warning_threshold')
        assert hasattr(DEBUG_DRAW_CONFIG, 'default_arrow_head_size')
        assert hasattr(DEBUG_DRAW_CONFIG, 'coordinate_axes_head_size')

    def test_config_values_are_reasonable(self):
        """Test that config values are within reasonable bounds."""
        from engine.debug.visual import DEBUG_DRAW_CONFIG

        assert DEBUG_DRAW_CONFIG.vector_normalize_epsilon > 0
        assert DEBUG_DRAW_CONFIG.max_primitives >= 0
        assert DEBUG_DRAW_CONFIG.primitive_warning_threshold >= 0
        assert 0 < DEBUG_DRAW_CONFIG.default_arrow_head_size <= 1.0
        assert DEBUG_DRAW_CONFIG.default_line_thickness > 0

    def test_gizmo_config_exists(self):
        """Test that gizmo configuration is accessible."""
        from engine.debug.visual import GIZMO_CONFIG, GizmoConfig

        assert isinstance(GIZMO_CONFIG, GizmoConfig)

    def test_gizmo_config_has_expected_fields(self):
        """Test that gizmo config has all expected fields."""
        from engine.debug.visual import GIZMO_CONFIG

        assert hasattr(GIZMO_CONFIG, 'translate_plane_size_ratio')
        assert hasattr(GIZMO_CONFIG, 'scale_handle_box_ratio')
        assert hasattr(GIZMO_CONFIG, 'hit_test_radius_ratio')
        assert hasattr(GIZMO_CONFIG, 'light_center_sphere_radius')
        assert hasattr(GIZMO_CONFIG, 'camera_default_aspect_ratio')
