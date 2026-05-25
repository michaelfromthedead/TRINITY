"""
Comprehensive tests for the Border widget.

Tests cover:
- Border style configuration
- Corner radius settings
- Border width and color
- Serialization/deserialization

Note: The border.py source file may not exist yet. These tests are written
based on the expected API from the primitives __init__.py exports:
- Border
- BorderStyle
- CornerRadius
"""

import pytest
from unittest.mock import MagicMock, patch


class TestCornerRadius:
    """Tests for CornerRadius configuration."""

    def test_corner_radius_uniform(self):
        """Test uniform corner radius creation."""
        # Expected API based on __init__.py exports
        try:
            from engine.ui.widgets.primitives.border import CornerRadius
        except ImportError:
            pytest.skip("border.py not yet implemented")

        radius = CornerRadius.uniform(8.0)
        assert radius.top_left == 8.0
        assert radius.top_right == 8.0
        assert radius.bottom_left == 8.0
        assert radius.bottom_right == 8.0

    def test_corner_radius_individual(self):
        """Test individual corner radius values."""
        try:
            from engine.ui.widgets.primitives.border import CornerRadius
        except ImportError:
            pytest.skip("border.py not yet implemented")

        radius = CornerRadius(
            top_left=4.0,
            top_right=8.0,
            bottom_left=12.0,
            bottom_right=16.0
        )
        assert radius.top_left == 4.0
        assert radius.top_right == 8.0
        assert radius.bottom_left == 12.0
        assert radius.bottom_right == 16.0

    def test_corner_radius_zero(self):
        """Test zero corner radius (sharp corners)."""
        try:
            from engine.ui.widgets.primitives.border import CornerRadius
        except ImportError:
            pytest.skip("border.py not yet implemented")

        radius = CornerRadius.uniform(0.0)
        assert radius.top_left == 0.0
        assert radius.is_zero

    def test_corner_radius_negative_fails(self):
        """Test negative corner radius fails validation."""
        try:
            from engine.ui.widgets.primitives.border import CornerRadius
        except ImportError:
            pytest.skip("border.py not yet implemented")

        with pytest.raises(ValueError, match="must be >= 0"):
            CornerRadius(top_left=-1.0)

    def test_corner_radius_is_uniform(self):
        """Test is_uniform property."""
        try:
            from engine.ui.widgets.primitives.border import CornerRadius
        except ImportError:
            pytest.skip("border.py not yet implemented")

        uniform = CornerRadius.uniform(8.0)
        assert uniform.is_uniform is True

        non_uniform = CornerRadius(top_left=4.0, top_right=8.0)
        assert non_uniform.is_uniform is False

    def test_corner_radius_max_radius(self):
        """Test getting maximum radius value."""
        try:
            from engine.ui.widgets.primitives.border import CornerRadius
        except ImportError:
            pytest.skip("border.py not yet implemented")

        radius = CornerRadius(
            top_left=4.0,
            top_right=8.0,
            bottom_left=6.0,
            bottom_right=10.0
        )
        assert radius.max_radius == 10.0


class TestBorderStyle:
    """Tests for BorderStyle configuration."""

    def test_border_style_default(self):
        """Test default BorderStyle values."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle()
        assert style.width == 1.0
        assert style.color is not None
        assert style.style == "solid"

    def test_border_style_solid(self):
        """Test solid border style."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(style="solid", width=2.0)
        assert style.style == "solid"
        assert style.width == 2.0

    def test_border_style_dashed(self):
        """Test dashed border style."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(style="dashed", dash_length=5.0, gap_length=3.0)
        assert style.style == "dashed"
        assert style.dash_length == 5.0
        assert style.gap_length == 3.0

    def test_border_style_dotted(self):
        """Test dotted border style."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(style="dotted")
        assert style.style == "dotted"

    def test_border_style_double(self):
        """Test double border style."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(style="double", width=4.0)
        assert style.style == "double"

    def test_border_style_groove(self):
        """Test groove border style."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(style="groove")
        assert style.style == "groove"

    def test_border_style_ridge(self):
        """Test ridge border style."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(style="ridge")
        assert style.style == "ridge"

    def test_border_style_inset(self):
        """Test inset border style."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(style="inset")
        assert style.style == "inset"

    def test_border_style_outset(self):
        """Test outset border style."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(style="outset")
        assert style.style == "outset"

    def test_border_style_none(self):
        """Test none border style."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(style="none")
        assert style.style == "none"
        assert style.is_visible is False

    def test_border_style_invalid_type(self):
        """Test invalid border style type."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        with pytest.raises(ValueError, match="Invalid border style"):
            BorderStyle(style="wavy")

    def test_border_style_width_validation(self):
        """Test border width must be non-negative."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        with pytest.raises(ValueError, match="must be >= 0"):
            BorderStyle(width=-1.0)

    def test_border_style_color_hex(self):
        """Test border color with hex string."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(color="#FF0000")
        # Color should be parsed to RGBA tuple
        assert style.color[0] == pytest.approx(1.0)
        assert style.color[1] == pytest.approx(0.0)
        assert style.color[2] == pytest.approx(0.0)

    def test_border_style_color_tuple(self):
        """Test border color with RGBA tuple."""
        try:
            from engine.ui.widgets.primitives.border import BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(color=(0.5, 0.6, 0.7, 0.8))
        assert style.color == (0.5, 0.6, 0.7, 0.8)


class TestBorderWidget:
    """Tests for the Border widget class."""

    def test_border_default_initialization(self):
        """Test Border initializes with correct defaults."""
        try:
            from engine.ui.widgets.primitives.border import Border
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border()
        assert border.width == 0.0
        assert border.height == 0.0
        assert border.style is not None

    def test_border_with_dimensions(self):
        """Test Border with explicit dimensions."""
        try:
            from engine.ui.widgets.primitives.border import Border
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border(width=200.0, height=100.0)
        assert border.width == 200.0
        assert border.height == 100.0

    def test_border_with_style(self):
        """Test Border with custom style."""
        try:
            from engine.ui.widgets.primitives.border import Border, BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        style = BorderStyle(width=3.0, style="dashed")
        border = Border(style=style)
        assert border.style.width == 3.0
        assert border.style.style == "dashed"

    def test_border_with_corner_radius(self):
        """Test Border with corner radius."""
        try:
            from engine.ui.widgets.primitives.border import Border, CornerRadius
        except ImportError:
            pytest.skip("border.py not yet implemented")

        radius = CornerRadius.uniform(12.0)
        border = Border(corner_radius=radius)
        assert border.corner_radius.top_left == 12.0

    def test_border_style_setter(self):
        """Test setting border style."""
        try:
            from engine.ui.widgets.primitives.border import Border, BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border()
        new_style = BorderStyle(width=5.0, color="#00FF00")
        border.style = new_style
        assert border.style.width == 5.0

    def test_border_corner_radius_setter(self):
        """Test setting corner radius."""
        try:
            from engine.ui.widgets.primitives.border import Border, CornerRadius
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border()
        radius = CornerRadius(top_left=10, top_right=20, bottom_left=30, bottom_right=40)
        border.corner_radius = radius
        assert border.corner_radius.top_left == 10
        assert border.corner_radius.bottom_right == 40

    def test_border_width_setter(self):
        """Test setting border widget width."""
        try:
            from engine.ui.widgets.primitives.border import Border
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border()
        border.width = 150.0
        assert border.width == 150.0

    def test_border_height_setter(self):
        """Test setting border widget height."""
        try:
            from engine.ui.widgets.primitives.border import Border
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border()
        border.height = 75.0
        assert border.height == 75.0


class TestBorderBounds:
    """Tests for Border bounds and geometry."""

    def test_border_bounds(self):
        """Test getting border bounds."""
        try:
            from engine.ui.widgets.primitives.border import Border
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border(x=10, y=20, width=100, height=50)
        assert border.bounds == (10, 20, 100, 50)

    def test_border_inner_bounds(self):
        """Test getting inner bounds (content area)."""
        try:
            from engine.ui.widgets.primitives.border import Border, BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border(
            x=0, y=0, width=100, height=100,
            style=BorderStyle(width=5.0)
        )
        inner = border.inner_bounds
        # Inner should be inset by border width
        assert inner[0] == 5.0  # x
        assert inner[1] == 5.0  # y
        assert inner[2] == 90.0  # width
        assert inner[3] == 90.0  # height

    def test_border_contains_point(self):
        """Test point containment check."""
        try:
            from engine.ui.widgets.primitives.border import Border
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border(x=0, y=0, width=100, height=100)
        assert border.contains_point(50, 50) is True
        assert border.contains_point(0, 0) is True
        assert border.contains_point(100, 100) is True
        assert border.contains_point(-1, 50) is False
        assert border.contains_point(101, 50) is False


class TestBorderDirtyState:
    """Tests for dirty state tracking."""

    def test_border_dirty_after_style_change(self):
        """Test border is dirty after style changes."""
        try:
            from engine.ui.widgets.primitives.border import Border, BorderStyle
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border()
        border.mark_clean()
        border.style = BorderStyle(width=3.0)
        assert border.is_dirty

    def test_border_dirty_after_corner_radius_change(self):
        """Test border is dirty after corner radius changes."""
        try:
            from engine.ui.widgets.primitives.border import Border, CornerRadius
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border()
        border.mark_clean()
        border.corner_radius = CornerRadius.uniform(8.0)
        assert border.is_dirty

    def test_border_mark_clean(self):
        """Test mark_clean clears dirty state."""
        try:
            from engine.ui.widgets.primitives.border import Border
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border()
        border.mark_clean()
        assert border.is_dirty is False


class TestBorderSerialization:
    """Tests for Border serialization and deserialization."""

    def test_border_to_dict(self):
        """Test serialization to dictionary."""
        try:
            from engine.ui.widgets.primitives.border import (
                Border, BorderStyle, CornerRadius
            )
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border(
            width=200, height=100,
            style=BorderStyle(width=2.0, style="solid", color="#FF0000"),
            corner_radius=CornerRadius.uniform(8.0)
        )

        data = border.to_dict()
        assert data["width"] == 200
        assert data["height"] == 100
        assert data["style"]["width"] == 2.0
        assert data["style"]["style"] == "solid"
        assert data["corner_radius"]["top_left"] == 8.0

    def test_border_from_dict(self):
        """Test deserialization from dictionary."""
        try:
            from engine.ui.widgets.primitives.border import Border
        except ImportError:
            pytest.skip("border.py not yet implemented")

        data = {
            "width": 150,
            "height": 75,
            "style": {
                "width": 3.0,
                "style": "dashed",
                "color": (0.0, 0.0, 1.0, 1.0),
            },
            "corner_radius": {
                "top_left": 4.0,
                "top_right": 8.0,
                "bottom_left": 12.0,
                "bottom_right": 16.0,
            }
        }

        border = Border.from_dict(data)
        assert border.width == 150
        assert border.height == 75
        assert border.style.width == 3.0
        assert border.style.style == "dashed"
        assert border.corner_radius.top_left == 4.0
        assert border.corner_radius.bottom_right == 16.0

    def test_border_roundtrip_serialization(self):
        """Test serialization roundtrip preserves data."""
        try:
            from engine.ui.widgets.primitives.border import (
                Border, BorderStyle, CornerRadius
            )
        except ImportError:
            pytest.skip("border.py not yet implemented")

        original = Border(
            x=10, y=20,
            width=200, height=150,
            style=BorderStyle(
                width=4.0,
                style="double",
                color=(0.5, 0.5, 0.5, 1.0)
            ),
            corner_radius=CornerRadius(5, 10, 15, 20)
        )

        data = original.to_dict()
        restored = Border.from_dict(data)

        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.style.width == original.style.width
        assert restored.style.style == original.style.style
        assert restored.corner_radius.top_left == original.corner_radius.top_left


class TestBorderRendering:
    """Tests for Border rendering-related functionality."""

    def test_border_get_path_points(self):
        """Test getting border path points for rendering."""
        try:
            from engine.ui.widgets.primitives.border import Border, CornerRadius
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border(
            width=100, height=100,
            corner_radius=CornerRadius.uniform(10.0)
        )

        points = border.get_path_points()
        # Should have points for the border path
        assert len(points) > 0

    def test_border_get_vertices(self):
        """Test getting border vertices for mesh rendering."""
        try:
            from engine.ui.widgets.primitives.border import Border
        except ImportError:
            pytest.skip("border.py not yet implemented")

        border = Border(width=100, height=100)

        vertices = border.get_vertices()
        # Should return vertex data for rendering
        assert vertices is not None
