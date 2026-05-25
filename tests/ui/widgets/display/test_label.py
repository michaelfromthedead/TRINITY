"""
Comprehensive tests for Label widget.

Tests cover:
- Text content and display
- Typography settings (font, size, weight, color)
- Text alignment and overflow handling
- Icon support (leading/trailing)
- Auto-sizing and constraints
- Transform properties (position, dimensions)
- Visibility and opacity
- Dirty tracking and layout invalidation
- Accessibility features
- Serialization/deserialization
"""

import pytest
import sys
from pathlib import Path

# Add engine to path
engine_path = Path(__file__).parent.parent.parent.parent.parent / "engine"
sys.path.insert(0, str(engine_path.parent))

from engine.ui.widgets.display.label import (
    Label,
    IconPosition,
    TextAlign,
    TextOverflow,
)


class TestLabelInitialization:
    """Test Label initialization and default values."""

    def test_default_initialization(self):
        """Test label initializes with correct defaults."""
        label = Label()
        assert label.text == ""
        assert label.font_family == "default"
        assert label.font_size == 14.0
        assert label.font_weight == "normal"
        assert label.text_color == "#000000"
        assert label.auto_size is True
        assert label.visible is True
        assert label.enabled is True

    def test_text_initialization(self):
        """Test label can be initialized with text."""
        label = Label()
        label.text = "Hello World"
        assert label.text == "Hello World"

    def test_default_text_align(self):
        """Test default text alignment is LEFT."""
        label = Label()
        assert label.text_align == TextAlign.LEFT

    def test_default_text_overflow(self):
        """Test default text overflow is ELLIPSIS."""
        label = Label()
        assert label.text_overflow == TextOverflow.ELLIPSIS

    def test_default_icon_position(self):
        """Test default icon position is LEADING."""
        label = Label()
        assert label.icon_position == IconPosition.LEADING


class TestLabelTextContent:
    """Test Label text content handling."""

    def test_set_text_content(self):
        """Test setting text content."""
        label = Label()
        label.text = "Player Score: 1000"
        assert label.text == "Player Score: 1000"

    def test_empty_text(self):
        """Test empty text handling."""
        label = Label()
        label.text = ""
        assert label.text == ""

    def test_text_with_special_characters(self):
        """Test text with special characters."""
        label = Label()
        label.text = "Health: 100/100 (Max!)"
        assert label.text == "Health: 100/100 (Max!)"

    def test_text_with_unicode(self):
        """Test text with unicode characters."""
        label = Label()
        label.text = "Player: \u2764 100"  # Heart symbol
        assert label.text == "Player: \u2764 100"

    def test_text_change_marks_dirty(self):
        """Test that changing text marks the label as dirty."""
        label = Label()
        label.clear_dirty()
        label.text = "New Text"
        assert label.is_dirty("text")


class TestLabelTypography:
    """Test Label typography settings."""

    def test_set_font_family(self):
        """Test setting font family."""
        label = Label()
        label.font_family = "Arial"
        assert label.font_family == "Arial"

    def test_set_font_size(self):
        """Test setting font size."""
        label = Label()
        label.font_size = 18.0
        assert label.font_size == 18.0

    def test_font_size_clamped_minimum(self):
        """Test font size is clamped to minimum."""
        label = Label()
        label.font_size = 0.5
        assert label.font_size >= 1.0

    def test_font_size_clamped_maximum(self):
        """Test font size is clamped to maximum."""
        label = Label()
        label.font_size = 250.0
        assert label.font_size <= 200.0

    def test_set_font_weight_normal(self):
        """Test setting font weight to normal."""
        label = Label()
        label.font_weight = "normal"
        assert label.font_weight == "normal"

    def test_set_font_weight_bold(self):
        """Test setting font weight to bold."""
        label = Label()
        label.font_weight = "bold"
        assert label.font_weight == "bold"

    def test_set_font_weight_light(self):
        """Test setting font weight to light."""
        label = Label()
        label.font_weight = "light"
        assert label.font_weight == "light"

    def test_invalid_font_weight_raises(self):
        """Test invalid font weight raises ValueError."""
        label = Label()
        with pytest.raises(ValueError):
            label.font_weight = "invalid"

    def test_set_text_color(self):
        """Test setting text color."""
        label = Label()
        label.text_color = "#FF0000"
        assert label.text_color == "#FF0000"


class TestLabelTextAlignment:
    """Test Label text alignment settings."""

    def test_set_align_left(self):
        """Test setting text align to LEFT."""
        label = Label()
        label.text_align = TextAlign.LEFT
        assert label.text_align == TextAlign.LEFT

    def test_set_align_center(self):
        """Test setting text align to CENTER."""
        label = Label()
        label.text_align = TextAlign.CENTER
        assert label.text_align == TextAlign.CENTER

    def test_set_align_right(self):
        """Test setting text align to RIGHT."""
        label = Label()
        label.text_align = TextAlign.RIGHT
        assert label.text_align == TextAlign.RIGHT


class TestLabelTextOverflow:
    """Test Label text overflow handling."""

    def test_set_overflow_clip(self):
        """Test setting overflow to CLIP."""
        label = Label()
        label.text_overflow = TextOverflow.CLIP
        assert label.text_overflow == TextOverflow.CLIP

    def test_set_overflow_ellipsis(self):
        """Test setting overflow to ELLIPSIS."""
        label = Label()
        label.text_overflow = TextOverflow.ELLIPSIS
        assert label.text_overflow == TextOverflow.ELLIPSIS

    def test_set_overflow_fade(self):
        """Test setting overflow to FADE."""
        label = Label()
        label.text_overflow = TextOverflow.FADE
        assert label.text_overflow == TextOverflow.FADE


class TestLabelIconSupport:
    """Test Label icon support."""

    def test_no_icon_by_default(self):
        """Test no icon is set by default."""
        label = Label()
        assert label.icon is None
        assert label.has_icon is False

    def test_set_icon(self):
        """Test setting an icon."""
        label = Label()
        label.icon = "heart"
        assert label.icon == "heart"
        assert label.has_icon is True

    def test_set_icon_position_leading(self):
        """Test setting icon position to LEADING."""
        label = Label()
        label.icon = "star"
        label.icon_position = IconPosition.LEADING
        assert label.icon_position == IconPosition.LEADING

    def test_set_icon_position_trailing(self):
        """Test setting icon position to TRAILING."""
        label = Label()
        label.icon = "star"
        label.icon_position = IconPosition.TRAILING
        assert label.icon_position == IconPosition.TRAILING

    def test_set_icon_spacing(self):
        """Test setting icon spacing."""
        label = Label()
        label.icon_spacing = 8.0
        assert label.icon_spacing == 8.0

    def test_icon_spacing_non_negative(self):
        """Test icon spacing cannot be negative."""
        label = Label()
        label.icon_spacing = -5.0
        assert label.icon_spacing >= 0.0

    def test_set_icon_color(self):
        """Test setting icon color."""
        label = Label()
        label.icon_color = "#00FF00"
        assert label.icon_color == "#00FF00"

    def test_effective_icon_color_inherits(self):
        """Test effective icon color inherits from text color."""
        label = Label()
        label.text_color = "#FF0000"
        label.icon_color = None
        assert label.effective_icon_color == "#FF0000"

    def test_effective_icon_color_override(self):
        """Test effective icon color with explicit setting."""
        label = Label()
        label.text_color = "#FF0000"
        label.icon_color = "#00FF00"
        assert label.effective_icon_color == "#00FF00"


class TestLabelSizing:
    """Test Label sizing and constraints."""

    def test_auto_size_enabled_by_default(self):
        """Test auto-size is enabled by default."""
        label = Label()
        assert label.auto_size is True

    def test_disable_auto_size(self):
        """Test disabling auto-size."""
        label = Label()
        label.auto_size = False
        assert label.auto_size is False

    def test_set_min_width(self):
        """Test setting minimum width."""
        label = Label()
        label.min_width = 50.0
        assert label.min_width == 50.0

    def test_min_width_non_negative(self):
        """Test minimum width cannot be negative."""
        label = Label()
        label.min_width = -10.0
        assert label.min_width >= 0.0

    def test_set_max_width(self):
        """Test setting maximum width."""
        label = Label()
        label.max_width = 200.0
        assert label.max_width == 200.0

    def test_max_width_non_negative(self):
        """Test maximum width cannot be negative."""
        label = Label()
        label.max_width = -10.0
        assert label.max_width >= 0.0

    def test_measure_returns_tuple(self):
        """Test measure returns width and height tuple."""
        label = Label()
        label.text = "Test"
        result = label.measure()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_measure_respects_min_width(self):
        """Test measure respects minimum width constraint."""
        label = Label()
        label.text = "A"  # Short text
        label.min_width = 100.0
        width, height = label.measure()
        assert width >= 100.0


class TestLabelTransform:
    """Test Label transform properties."""

    def test_set_x_position(self):
        """Test setting X position."""
        label = Label()
        label.x = 100.0
        assert label.x == 100.0

    def test_set_y_position(self):
        """Test setting Y position."""
        label = Label()
        label.y = 50.0
        assert label.y == 50.0

    def test_set_width(self):
        """Test setting width."""
        label = Label()
        label.auto_size = False
        label.width = 150.0
        assert label.width == 150.0

    def test_set_height(self):
        """Test setting height."""
        label = Label()
        label.auto_size = False
        label.height = 30.0
        assert label.height == 30.0


class TestLabelVisibility:
    """Test Label visibility and state."""

    def test_visible_by_default(self):
        """Test label is visible by default."""
        label = Label()
        assert label.visible is True

    def test_set_invisible(self):
        """Test setting label invisible."""
        label = Label()
        label.visible = False
        assert label.visible is False

    def test_enabled_by_default(self):
        """Test label is enabled by default."""
        label = Label()
        assert label.enabled is True

    def test_set_disabled(self):
        """Test disabling label."""
        label = Label()
        label.enabled = False
        assert label.enabled is False

    def test_default_opacity(self):
        """Test default opacity is 1.0."""
        label = Label()
        assert label.opacity == 1.0

    def test_set_opacity(self):
        """Test setting opacity."""
        label = Label()
        label.opacity = 0.5
        assert label.opacity == 0.5

    def test_opacity_clamped_minimum(self):
        """Test opacity is clamped to minimum 0.0."""
        label = Label()
        label.opacity = -0.5
        assert label.opacity >= 0.0

    def test_opacity_clamped_maximum(self):
        """Test opacity is clamped to maximum 1.0."""
        label = Label()
        label.opacity = 1.5
        assert label.opacity <= 1.0


class TestLabelDirtyTracking:
    """Test Label dirty tracking."""

    def test_initially_dirty(self):
        """Test label starts with dirty state."""
        label = Label()
        # After initialization, dirty tracking is set up
        assert hasattr(label, '_dirty_fields')

    def test_clear_dirty(self):
        """Test clearing dirty flags."""
        label = Label()
        label.text = "Test"
        label.clear_dirty()
        assert not label.is_dirty()

    def test_text_change_marks_dirty(self):
        """Test text change marks as dirty."""
        label = Label()
        label.clear_dirty()
        label.text = "New"
        assert label.is_dirty("text")

    def test_font_change_marks_dirty(self):
        """Test font change marks as dirty."""
        label = Label()
        label.clear_dirty()
        label.font_size = 20.0
        assert label.is_dirty("font_size")


class TestLabelLayout:
    """Test Label layout calculations."""

    def test_get_text_bounds_no_icon(self):
        """Test getting text bounds without icon."""
        label = Label()
        label.text = "Test"
        label._width = 100.0
        label._height = 20.0
        bounds = label.get_text_bounds()
        assert isinstance(bounds, tuple)
        assert len(bounds) == 4

    def test_get_text_bounds_with_leading_icon(self):
        """Test getting text bounds with leading icon."""
        label = Label()
        label.text = "Test"
        label.icon = "star"
        label.icon_position = IconPosition.LEADING
        label._width = 100.0
        label._height = 20.0
        bounds = label.get_text_bounds()
        x, y, width, height = bounds
        assert x > label._x  # Text starts after icon

    def test_get_icon_bounds_no_icon(self):
        """Test getting icon bounds when no icon."""
        label = Label()
        label.text = "Test"
        bounds = label.get_icon_bounds()
        assert bounds is None

    def test_get_icon_bounds_with_icon(self):
        """Test getting icon bounds with icon."""
        label = Label()
        label.icon = "star"
        label._width = 100.0
        label._height = 20.0
        bounds = label.get_icon_bounds()
        assert bounds is not None
        assert len(bounds) == 4


class TestLabelAccessibility:
    """Test Label accessibility features."""

    def test_get_accessible_text(self):
        """Test getting accessible text."""
        label = Label()
        label.text = "Player Score"
        assert label.get_accessible_text() == "Player Score"

    def test_get_accessible_role(self):
        """Test getting accessible role."""
        label = Label()
        role = label.get_accessible_role()
        assert role == "text"


class TestLabelSerialization:
    """Test Label serialization and deserialization."""

    def test_to_dict(self):
        """Test serializing label to dictionary."""
        label = Label()
        label.text = "Test Label"
        label.font_size = 16.0
        data = label.to_dict()
        assert data["text"] == "Test Label"
        assert data["font_size"] == 16.0

    def test_from_dict(self):
        """Test deserializing label from dictionary."""
        data = {
            "text": "Loaded Label",
            "font_size": 18.0,
            "font_weight": "bold",
            "text_color": "#FF0000",
        }
        label = Label.from_dict(data)
        assert label.text == "Loaded Label"
        assert label.font_size == 18.0
        assert label.font_weight == "bold"
        assert label.text_color == "#FF0000"

    def test_round_trip_serialization(self):
        """Test round-trip serialization preserves data."""
        label1 = Label()
        label1.text = "Round Trip"
        label1.font_family = "Roboto"
        label1.font_size = 20.0
        label1.icon = "info"
        label1.opacity = 0.8

        data = label1.to_dict()
        label2 = Label.from_dict(data)

        assert label2.text == label1.text
        assert label2.font_family == label1.font_family
        assert label2.font_size == label1.font_size
        assert label2.icon == label1.icon
        assert label2.opacity == label1.opacity


class TestLabelRepr:
    """Test Label string representation."""

    def test_repr_without_icon(self):
        """Test repr without icon."""
        label = Label()
        label.text = "Test"
        repr_str = repr(label)
        assert "Label" in repr_str
        assert "Test" in repr_str

    def test_repr_with_icon(self):
        """Test repr with icon."""
        label = Label()
        label.text = "Test"
        label.icon = "star"
        repr_str = repr(label)
        assert "Label" in repr_str
        assert "star" in repr_str
