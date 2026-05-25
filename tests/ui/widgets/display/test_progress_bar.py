"""
Comprehensive tests for ProgressBar widget.

Tests cover:
- Value management (value, min, max, clamping)
- Styles (horizontal, vertical, circular)
- Modes (determinate, indeterminate)
- Animation (animated transitions, duration)
- Segments (segmented display, gaps)
- Colors (fill, background, border)
- Value display (text formatting)
- Transform properties
- Computed properties (normalized, percent, complete)
- Update and animation methods
- Accessibility
- Serialization
"""

import pytest
import time
import sys
from pathlib import Path

# Add engine to path
engine_path = Path(__file__).parent.parent.parent.parent.parent / "engine"
sys.path.insert(0, str(engine_path.parent))

from engine.ui.widgets.display.progress_bar import (
    ProgressBar,
    ProgressBarStyle,
    ProgressBarMode,
    ProgressBarDirection,
)


class TestProgressBarInitialization:
    """Test ProgressBar initialization and defaults."""

    def test_default_initialization(self):
        """Test progress bar initializes with correct defaults."""
        pb = ProgressBar()
        assert pb.value == 0.0
        assert pb.min_value == 0.0
        assert pb.max_value == 1.0
        assert pb.style == ProgressBarStyle.HORIZONTAL
        assert pb.mode == ProgressBarMode.DETERMINATE

    def test_default_animation_settings(self):
        """Test default animation settings."""
        pb = ProgressBar()
        assert pb.animated is True
        assert pb.animation_duration == 0.2

    def test_default_colors(self):
        """Test default colors."""
        pb = ProgressBar()
        assert pb.fill_color == "#4CAF50"
        assert pb.background_color == "#E0E0E0"
        assert pb.border_color == "#BDBDBD"

    def test_default_dimensions(self):
        """Test default dimensions."""
        pb = ProgressBar()
        assert pb.width == 200.0
        assert pb.height == 20.0


class TestProgressBarValue:
    """Test ProgressBar value management."""

    def test_set_value(self):
        """Test setting value."""
        pb = ProgressBar()
        pb._animated = False  # Disable animation for immediate update
        pb.value = 0.5
        assert pb.value == 0.5

    def test_value_clamped_to_max(self):
        """Test value is clamped to max."""
        pb = ProgressBar()
        pb._animated = False
        pb.value = 2.0
        assert pb.value <= pb.max_value

    def test_value_clamped_to_min(self):
        """Test value is clamped to min."""
        pb = ProgressBar()
        pb._animated = False
        pb.value = -1.0
        assert pb.value >= pb.min_value

    def test_set_min_value(self):
        """Test setting min value."""
        pb = ProgressBar()
        pb.min_value = 10.0
        assert pb.min_value == 10.0

    def test_min_value_cannot_exceed_max(self):
        """Test min value cannot exceed max value."""
        pb = ProgressBar()
        with pytest.raises(ValueError):
            pb.min_value = 2.0  # max is 1.0 by default

    def test_set_max_value(self):
        """Test setting max value."""
        pb = ProgressBar()
        pb.max_value = 100.0
        assert pb.max_value == 100.0

    def test_max_value_cannot_be_less_than_min(self):
        """Test max value cannot be less than min value."""
        pb = ProgressBar()
        with pytest.raises(ValueError):
            pb.max_value = -1.0

    def test_value_reclamped_when_max_changes(self):
        """Test value is reclamped when max changes."""
        pb = ProgressBar()
        pb._animated = False
        pb._value = 0.8
        pb.max_value = 0.5
        assert pb.value <= pb.max_value

    def test_custom_range(self):
        """Test progress bar with custom range."""
        pb = ProgressBar()
        pb._min_value = 0.0
        pb._max_value = 100.0
        pb._animated = False
        pb.set_value_immediate(75.0)
        assert pb.value == 75.0


class TestProgressBarStyles:
    """Test ProgressBar visual styles."""

    def test_set_horizontal_style(self):
        """Test setting horizontal style."""
        pb = ProgressBar()
        pb.style = ProgressBarStyle.HORIZONTAL
        assert pb.style == ProgressBarStyle.HORIZONTAL

    def test_set_vertical_style(self):
        """Test setting vertical style."""
        pb = ProgressBar()
        pb.style = ProgressBarStyle.VERTICAL
        assert pb.style == ProgressBarStyle.VERTICAL

    def test_set_circular_style(self):
        """Test setting circular style."""
        pb = ProgressBar()
        pb.style = ProgressBarStyle.CIRCULAR
        assert pb.style == ProgressBarStyle.CIRCULAR


class TestProgressBarModes:
    """Test ProgressBar modes."""

    def test_set_determinate_mode(self):
        """Test setting determinate mode."""
        pb = ProgressBar()
        pb.mode = ProgressBarMode.DETERMINATE
        assert pb.mode == ProgressBarMode.DETERMINATE

    def test_set_indeterminate_mode(self):
        """Test setting indeterminate mode."""
        pb = ProgressBar()
        pb.mode = ProgressBarMode.INDETERMINATE
        assert pb.mode == ProgressBarMode.INDETERMINATE

    def test_indeterminate_animation_updates(self):
        """Test indeterminate animation updates position."""
        pb = ProgressBar()
        pb.mode = ProgressBarMode.INDETERMINATE
        initial_pos = pb._indeterminate_position
        pb.update(0.5)
        assert pb._indeterminate_position != initial_pos


class TestProgressBarDirection:
    """Test ProgressBar fill direction."""

    def test_set_forward_direction(self):
        """Test setting forward direction."""
        pb = ProgressBar()
        pb.direction = ProgressBarDirection.FORWARD
        assert pb.direction == ProgressBarDirection.FORWARD

    def test_set_reverse_direction(self):
        """Test setting reverse direction."""
        pb = ProgressBar()
        pb.direction = ProgressBarDirection.REVERSE
        assert pb.direction == ProgressBarDirection.REVERSE


class TestProgressBarAnimation:
    """Test ProgressBar animation features."""

    def test_animation_enabled_by_default(self):
        """Test animation is enabled by default."""
        pb = ProgressBar()
        assert pb.animated is True

    def test_disable_animation(self):
        """Test disabling animation."""
        pb = ProgressBar()
        pb.animated = False
        assert pb.animated is False

    def test_set_animation_duration(self):
        """Test setting animation duration."""
        pb = ProgressBar()
        pb.animation_duration = 0.5
        assert pb.animation_duration == 0.5

    def test_animation_duration_non_negative(self):
        """Test animation duration cannot be negative."""
        pb = ProgressBar()
        pb.animation_duration = -1.0
        assert pb.animation_duration >= 0.0

    def test_set_value_immediate(self):
        """Test set_value_immediate bypasses animation."""
        pb = ProgressBar()
        pb.set_value_immediate(0.7)
        assert pb.value == 0.7
        assert pb.is_animating is False

    def test_animation_starts_on_value_change(self):
        """Test animation starts when value changes."""
        pb = ProgressBar()
        pb.animated = True
        pb.animation_duration = 0.2
        pb.value = 0.5
        assert pb._is_animating is True

    def test_is_animating_property(self):
        """Test is_animating property."""
        pb = ProgressBar()
        pb.set_value_immediate(0.5)
        assert pb.is_animating is False

    def test_display_value_during_animation(self):
        """Test display_value returns interpolated value during animation."""
        pb = ProgressBar()
        pb._value = 0.0
        pb._animation_target_value = 1.0
        pb._animation_start_value = 0.0
        pb._is_animating = True
        pb._animation_start_time = time.time()
        # Display value should be between start and target
        display = pb.display_value
        # It's running so could be any value in range
        assert 0.0 <= display <= 1.0

    def test_indeterminate_speed(self):
        """Test setting indeterminate animation speed."""
        pb = ProgressBar()
        pb.indeterminate_speed = 2.0
        assert pb.indeterminate_speed == 2.0

    def test_indeterminate_speed_minimum(self):
        """Test indeterminate speed has minimum."""
        pb = ProgressBar()
        pb.indeterminate_speed = 0.0
        assert pb.indeterminate_speed >= 0.1


class TestProgressBarSegments:
    """Test ProgressBar segmented display."""

    def test_default_no_segments(self):
        """Test default is no segments (smooth)."""
        pb = ProgressBar()
        assert pb.segments == 0

    def test_set_segments(self):
        """Test setting number of segments."""
        pb = ProgressBar()
        pb.segments = 10
        assert pb.segments == 10

    def test_segments_non_negative(self):
        """Test segments cannot be negative."""
        pb = ProgressBar()
        pb.segments = -5
        assert pb.segments >= 0

    def test_set_segment_gap(self):
        """Test setting segment gap."""
        pb = ProgressBar()
        pb.segment_gap = 4.0
        assert pb.segment_gap == 4.0

    def test_segment_gap_non_negative(self):
        """Test segment gap cannot be negative."""
        pb = ProgressBar()
        pb.segment_gap = -2.0
        assert pb.segment_gap >= 0.0

    def test_get_segment_rects_no_segments(self):
        """Test get_segment_rects returns empty for no segments."""
        pb = ProgressBar()
        pb.segments = 0
        rects = pb.get_segment_rects()
        assert rects == []

    def test_get_segment_rects_with_segments(self):
        """Test get_segment_rects returns correct number of segments."""
        pb = ProgressBar()
        pb.segments = 5
        pb._value = 0.5
        rects = pb.get_segment_rects()
        assert len(rects) == 5


class TestProgressBarColors:
    """Test ProgressBar color settings."""

    def test_set_fill_color(self):
        """Test setting fill color."""
        pb = ProgressBar()
        pb.fill_color = "#FF0000"
        assert pb.fill_color == "#FF0000"

    def test_set_background_color(self):
        """Test setting background color."""
        pb = ProgressBar()
        pb.background_color = "#333333"
        assert pb.background_color == "#333333"

    def test_set_border_color(self):
        """Test setting border color."""
        pb = ProgressBar()
        pb.border_color = "#000000"
        assert pb.border_color == "#000000"

    def test_set_border_width(self):
        """Test setting border width."""
        pb = ProgressBar()
        pb.border_width = 2.0
        assert pb.border_width == 2.0

    def test_border_width_non_negative(self):
        """Test border width cannot be negative."""
        pb = ProgressBar()
        pb.border_width = -1.0
        assert pb.border_width >= 0.0

    def test_set_corner_radius(self):
        """Test setting corner radius."""
        pb = ProgressBar()
        pb.corner_radius = 8.0
        assert pb.corner_radius == 8.0


class TestProgressBarValueDisplay:
    """Test ProgressBar value text display."""

    def test_show_value_disabled_by_default(self):
        """Test value display is disabled by default."""
        pb = ProgressBar()
        assert pb.show_value is False

    def test_enable_show_value(self):
        """Test enabling value display."""
        pb = ProgressBar()
        pb.show_value = True
        assert pb.show_value is True

    def test_set_value_format(self):
        """Test setting value format string."""
        pb = ProgressBar()
        pb.value_format = "{:.1%}"
        assert pb.value_format == "{:.1%}"

    def test_set_value_color(self):
        """Test setting value text color."""
        pb = ProgressBar()
        pb.value_color = "#FFFFFF"
        assert pb.value_color == "#FFFFFF"

    def test_set_value_font_size(self):
        """Test setting value font size."""
        pb = ProgressBar()
        pb.value_font_size = 14.0
        assert pb.value_font_size == 14.0

    def test_value_font_size_minimum(self):
        """Test value font size has minimum."""
        pb = ProgressBar()
        pb.value_font_size = 0.5
        assert pb.value_font_size >= 1.0

    def test_formatted_value(self):
        """Test formatted value string."""
        pb = ProgressBar()
        pb._value = 0.5
        formatted = pb.formatted_value
        assert "50" in formatted or "0.5" in formatted


class TestProgressBarComputedProperties:
    """Test ProgressBar computed properties."""

    def test_normalized_value(self):
        """Test normalized value (0-1)."""
        pb = ProgressBar()
        pb._min_value = 0.0
        pb._max_value = 100.0
        pb._value = 50.0
        assert pb.normalized_value == 0.5

    def test_normalized_value_zero_range(self):
        """Test normalized value with zero range."""
        pb = ProgressBar()
        pb._min_value = 50.0
        pb._max_value = 50.0
        assert pb.normalized_value == 0.0

    def test_percent(self):
        """Test percent value."""
        pb = ProgressBar()
        pb._value = 0.5
        assert pb.percent == 50.0

    def test_is_complete_true(self):
        """Test is_complete when at max."""
        pb = ProgressBar()
        pb._value = 1.0
        assert pb.is_complete is True

    def test_is_complete_false(self):
        """Test is_complete when not at max."""
        pb = ProgressBar()
        pb._value = 0.5
        assert pb.is_complete is False


class TestProgressBarMethods:
    """Test ProgressBar methods."""

    def test_reset(self):
        """Test reset method."""
        pb = ProgressBar()
        pb._value = 0.5
        pb.reset()
        assert pb.value == pb.min_value

    def test_complete(self):
        """Test complete method."""
        pb = ProgressBar()
        pb.complete()
        # Animation may be in progress
        assert pb._animation_target_value == pb.max_value or pb.value == pb.max_value

    def test_increment(self):
        """Test increment method."""
        pb = ProgressBar()
        pb._animated = False
        pb._value = 0.5
        pb.increment(0.1)
        assert pb.value == 0.6

    def test_decrement(self):
        """Test decrement method."""
        pb = ProgressBar()
        pb._animated = False
        pb._value = 0.5
        pb.decrement(0.1)
        assert pb.value == 0.4


class TestProgressBarUpdate:
    """Test ProgressBar update method."""

    def test_update_processes_animation(self):
        """Test update processes animation."""
        pb = ProgressBar()
        pb._is_animating = True
        pb._animation_start_time = time.time() - 1.0  # 1 second ago
        pb._animation_duration = 0.2
        pb.update(0.016)  # ~60fps frame
        # Animation should be complete
        assert pb._is_animating is False

    def test_update_indeterminate_mode(self):
        """Test update in indeterminate mode."""
        pb = ProgressBar()
        pb.mode = ProgressBarMode.INDETERMINATE
        initial = pb._indeterminate_position
        pb.update(0.5)
        assert pb._indeterminate_position != initial


class TestProgressBarTransform:
    """Test ProgressBar transform properties."""

    def test_set_x(self):
        """Test setting X position."""
        pb = ProgressBar()
        pb.x = 100.0
        assert pb.x == 100.0

    def test_set_y(self):
        """Test setting Y position."""
        pb = ProgressBar()
        pb.y = 50.0
        assert pb.y == 50.0

    def test_set_width(self):
        """Test setting width."""
        pb = ProgressBar()
        pb.width = 300.0
        assert pb.width == 300.0

    def test_width_non_negative(self):
        """Test width cannot be negative."""
        pb = ProgressBar()
        pb.width = -100.0
        assert pb.width >= 0.0

    def test_set_height(self):
        """Test setting height."""
        pb = ProgressBar()
        pb.height = 30.0
        assert pb.height == 30.0


class TestProgressBarVisibility:
    """Test ProgressBar visibility properties."""

    def test_visible_by_default(self):
        """Test visible by default."""
        pb = ProgressBar()
        assert pb.visible is True

    def test_set_invisible(self):
        """Test setting invisible."""
        pb = ProgressBar()
        pb.visible = False
        assert pb.visible is False

    def test_enabled_by_default(self):
        """Test enabled by default."""
        pb = ProgressBar()
        assert pb.enabled is True

    def test_set_disabled(self):
        """Test setting disabled."""
        pb = ProgressBar()
        pb.enabled = False
        assert pb.enabled is False

    def test_set_opacity(self):
        """Test setting opacity."""
        pb = ProgressBar()
        pb.opacity = 0.7
        assert pb.opacity == 0.7


class TestProgressBarCallbacks:
    """Test ProgressBar callbacks."""

    def test_on_complete_callback(self):
        """Test on_complete callback is called."""
        pb = ProgressBar()
        pb._animated = False
        callback_called = []

        def on_complete():
            callback_called.append(True)

        pb.on_complete = on_complete
        pb._value = 0.5
        pb.value = 1.0  # Should trigger callback
        assert len(callback_called) == 1


class TestProgressBarRenderingHelpers:
    """Test ProgressBar rendering helper methods."""

    def test_get_fill_rect_horizontal(self):
        """Test get_fill_rect for horizontal style."""
        pb = ProgressBar()
        pb.style = ProgressBarStyle.HORIZONTAL
        pb._value = 0.5
        rect = pb.get_fill_rect()
        assert len(rect) == 4
        x, y, w, h = rect
        assert w == pb.width * 0.5

    def test_get_fill_rect_vertical(self):
        """Test get_fill_rect for vertical style."""
        pb = ProgressBar()
        pb.style = ProgressBarStyle.VERTICAL
        pb._value = 0.5
        rect = pb.get_fill_rect()
        x, y, w, h = rect
        assert h == pb.height * 0.5

    def test_get_fill_rect_reverse_direction(self):
        """Test get_fill_rect with reverse direction."""
        pb = ProgressBar()
        pb.style = ProgressBarStyle.HORIZONTAL
        pb.direction = ProgressBarDirection.REVERSE
        pb._value = 0.5
        rect = pb.get_fill_rect()
        x, y, w, h = rect
        assert x > pb.x

    def test_get_circular_arc(self):
        """Test get_circular_arc method."""
        pb = ProgressBar()
        pb.style = ProgressBarStyle.CIRCULAR
        pb._value = 0.5
        arc = pb.get_circular_arc()
        assert len(arc) == 5  # cx, cy, radius, start_angle, end_angle


class TestProgressBarAccessibility:
    """Test ProgressBar accessibility features."""

    def test_get_accessible_text_determinate(self):
        """Test accessible text for determinate mode."""
        pb = ProgressBar()
        pb._value = 0.5
        text = pb.get_accessible_text()
        assert "50" in text

    def test_get_accessible_text_indeterminate(self):
        """Test accessible text for indeterminate mode."""
        pb = ProgressBar()
        pb.mode = ProgressBarMode.INDETERMINATE
        text = pb.get_accessible_text()
        assert "Loading" in text

    def test_get_accessible_role(self):
        """Test accessible role."""
        pb = ProgressBar()
        role = pb.get_accessible_role()
        assert role == "progressbar"

    def test_get_accessible_value(self):
        """Test accessible value info."""
        pb = ProgressBar()
        pb._min_value = 0.0
        pb._max_value = 100.0
        pb._value = 50.0
        info = pb.get_accessible_value()
        assert info["min"] == 0.0
        assert info["max"] == 100.0
        assert info["now"] == 50.0


class TestProgressBarSerialization:
    """Test ProgressBar serialization."""

    def test_to_dict(self):
        """Test serializing to dictionary."""
        pb = ProgressBar()
        pb._value = 0.75
        pb.fill_color = "#FF0000"
        pb.segments = 10
        data = pb.to_dict()
        assert data["value"] == 0.75
        assert data["fill_color"] == "#FF0000"
        assert data["segments"] == 10

    def test_from_dict(self):
        """Test deserializing from dictionary."""
        data = {
            "value": 0.6,
            "min_value": 0.0,
            "max_value": 1.0,
            "style": "VERTICAL",
            "fill_color": "#00FF00",
            "segments": 5,
        }
        pb = ProgressBar.from_dict(data)
        assert pb.value == 0.6
        assert pb.style == ProgressBarStyle.VERTICAL
        assert pb.fill_color == "#00FF00"
        assert pb.segments == 5

    def test_round_trip_serialization(self):
        """Test round-trip serialization preserves data."""
        pb1 = ProgressBar()
        pb1._value = 0.8
        pb1.style = ProgressBarStyle.CIRCULAR
        pb1.mode = ProgressBarMode.DETERMINATE
        pb1.segments = 8
        pb1.fill_color = "#123456"

        data = pb1.to_dict()
        pb2 = ProgressBar.from_dict(data)

        assert pb2.value == pb1.value
        assert pb2.style == pb1.style
        assert pb2.segments == pb1.segments
        assert pb2.fill_color == pb1.fill_color


class TestProgressBarDirtyTracking:
    """Test ProgressBar dirty tracking."""

    def test_is_dirty_after_change(self):
        """Test is_dirty returns True after change."""
        pb = ProgressBar()
        pb.clear_dirty()
        pb._animated = False
        pb.value = 0.5
        assert pb.is_dirty("value")

    def test_clear_dirty(self):
        """Test clearing dirty flags."""
        pb = ProgressBar()
        pb.value = 0.5
        pb.clear_dirty()
        assert not pb.is_dirty()


class TestProgressBarRepr:
    """Test ProgressBar string representation."""

    def test_repr(self):
        """Test repr includes key info."""
        pb = ProgressBar()
        pb._value = 0.5
        pb._min_value = 0.0
        pb._max_value = 1.0
        repr_str = repr(pb)
        assert "ProgressBar" in repr_str
        assert "0.5" in repr_str
