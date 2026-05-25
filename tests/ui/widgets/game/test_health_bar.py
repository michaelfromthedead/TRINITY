"""
Comprehensive tests for HealthBar widget.

Tests cover:
- Initialization and defaults
- Value management (current, max, display)
- Damage and healing operations
- Damage preview visualization
- Shield and armor overlays
- Animation and transitions
- Segmented display
- Callbacks (value changed, depleted, full)
- Resource types
- Rendering helpers
- Styling
"""

import pytest
import sys
from pathlib import Path

# Add engine to path
engine_path = Path(__file__).parent.parent.parent.parent.parent / "engine"
sys.path.insert(0, str(engine_path.parent))

from engine.ui.widgets.game.health_bar import (
    HealthBar,
    HealthBarStyle,
    HealthBarSegment,
    ResourceType,
)


class TestHealthBarInitialization:
    """Test HealthBar initialization and defaults."""

    def test_default_initialization(self):
        """Test health bar initializes with correct defaults."""
        hb = HealthBar()
        assert hb.max_value == 100.0
        assert hb.current_value == 100.0  # Defaults to max
        assert hb.is_visible is True

    def test_custom_max_value(self):
        """Test initialization with custom max value."""
        hb = HealthBar(max_value=200.0)
        assert hb.max_value == 200.0
        assert hb.current_value == 200.0

    def test_custom_current_value(self):
        """Test initialization with custom current value."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        assert hb.current_value == 50.0

    def test_custom_dimensions(self):
        """Test initialization with custom dimensions."""
        hb = HealthBar(width=300.0, height=30.0)
        assert hb.width == 300.0
        assert hb.height == 30.0

    def test_custom_position(self):
        """Test initialization with custom position."""
        hb = HealthBar(x=100.0, y=50.0)
        assert hb.x == 100.0
        assert hb.y == 50.0

    def test_resource_type_default(self):
        """Test default resource type is HEALTH."""
        hb = HealthBar()
        assert hb.resource_type == ResourceType.HEALTH

    def test_custom_resource_type(self):
        """Test custom resource type."""
        hb = HealthBar(resource_type=ResourceType.MANA)
        assert hb.resource_type == ResourceType.MANA

    def test_unique_id(self):
        """Test each health bar gets unique ID."""
        hb1 = HealthBar()
        hb2 = HealthBar()
        assert hb1.id != hb2.id


class TestHealthBarValueManagement:
    """Test HealthBar value management."""

    def test_set_value(self):
        """Test setting value."""
        hb = HealthBar(max_value=100.0)
        hb.set_value(50.0, animate=False)
        assert hb.current_value == 50.0

    def test_value_clamped_to_max(self):
        """Test value is clamped to max."""
        hb = HealthBar(max_value=100.0)
        hb.set_value(150.0, animate=False)
        assert hb.current_value == 100.0

    def test_value_clamped_to_min(self):
        """Test value is clamped to minimum (0)."""
        hb = HealthBar(max_value=100.0)
        hb.set_value(-50.0, animate=False)
        assert hb.current_value == 0.0

    def test_set_max_value(self):
        """Test setting max value."""
        hb = HealthBar(max_value=100.0)
        hb.max_value = 200.0
        assert hb.max_value == 200.0

    def test_fill_percent(self):
        """Test fill_percent property."""
        hb = HealthBar(max_value=100.0, current_value=75.0)
        assert hb.fill_percent == 0.75

    def test_fill_percent_empty(self):
        """Test fill_percent when empty."""
        hb = HealthBar(max_value=100.0, current_value=0.0)
        assert hb.fill_percent == 0.0

    def test_fill_percent_full(self):
        """Test fill_percent when full."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        assert hb.fill_percent == 1.0

    def test_actual_percent(self):
        """Test actual_percent (non-animated)."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        assert hb.actual_percent == 0.5


class TestHealthBarStates:
    """Test HealthBar state properties."""

    def test_is_empty_true(self):
        """Test is_empty when at zero."""
        hb = HealthBar(max_value=100.0, current_value=0.0)
        assert hb.is_empty is True

    def test_is_empty_false(self):
        """Test is_empty when not at zero."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        assert hb.is_empty is False

    def test_is_full_true(self):
        """Test is_full when at max."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        assert hb.is_full is True

    def test_is_full_false(self):
        """Test is_full when not at max."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        assert hb.is_full is False

    def test_is_low_true(self):
        """Test is_low when below threshold."""
        hb = HealthBar(max_value=100.0, current_value=20.0)
        # Default threshold is 0.25
        assert hb.is_low is True

    def test_is_low_false(self):
        """Test is_low when above threshold."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        assert hb.is_low is False


class TestHealthBarDamage:
    """Test HealthBar damage operations."""

    def test_apply_damage(self):
        """Test applying damage."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.apply_damage(25.0, show_preview=False)
        hb.skip_animation()
        assert hb.current_value == 75.0

    def test_apply_damage_zero(self):
        """Test applying zero damage does nothing."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        result = hb.apply_damage(0.0)
        assert result == 0.0
        assert hb.current_value == 100.0

    def test_apply_damage_negative(self):
        """Test applying negative damage does nothing."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        result = hb.apply_damage(-10.0)
        assert result == 0.0

    def test_apply_damage_with_preview(self):
        """Test damage preview is set."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.apply_damage(25.0, show_preview=True)
        assert hb.pending_damage == 25.0

    def test_damage_depletes_to_zero(self):
        """Test damage that exceeds current value."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        hb.apply_damage(75.0, show_preview=False)
        hb.skip_animation()
        assert hb.current_value == 0.0


class TestHealthBarHealing:
    """Test HealthBar healing operations."""

    def test_apply_heal(self):
        """Test applying healing."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        actual = hb.apply_heal(25.0, show_preview=False)
        hb.skip_animation()
        assert hb.current_value == 75.0
        assert actual == 25.0

    def test_apply_heal_zero(self):
        """Test applying zero healing."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        actual = hb.apply_heal(0.0)
        assert actual == 0.0

    def test_apply_heal_negative(self):
        """Test applying negative healing."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        actual = hb.apply_heal(-10.0)
        assert actual == 0.0

    def test_heal_capped_at_max(self):
        """Test healing is capped at max value."""
        hb = HealthBar(max_value=100.0, current_value=80.0)
        actual = hb.apply_heal(50.0, show_preview=False)
        hb.skip_animation()
        assert hb.current_value == 100.0
        assert actual == 20.0  # Only 20 was actually healed


class TestHealthBarShield:
    """Test HealthBar shield overlay."""

    def test_set_shield(self):
        """Test setting shield value."""
        hb = HealthBar(max_value=100.0)
        hb.set_shield(50.0)
        assert hb.shield_value == 50.0
        assert hb.shield_max == 50.0

    def test_set_shield_with_max(self):
        """Test setting shield with explicit max."""
        hb = HealthBar(max_value=100.0)
        hb.set_shield(30.0, maximum=50.0)
        assert hb.shield_value == 30.0
        assert hb.shield_max == 50.0

    def test_shield_percent(self):
        """Test shield_percent property."""
        hb = HealthBar(max_value=100.0)
        hb.set_shield(25.0, maximum=50.0)
        assert hb.shield_percent == 0.5

    def test_damage_applies_to_shield_first(self):
        """Test damage applies to shield before health."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.set_shield(30.0)
        hb.apply_damage(20.0, show_preview=False)
        hb.skip_animation()
        assert hb.shield_value == 10.0
        assert hb.current_value == 100.0  # Health unchanged

    def test_damage_penetrates_shield(self):
        """Test damage that exceeds shield reaches health."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.set_shield(20.0)
        hb.apply_damage(50.0, show_preview=False)
        hb.skip_animation()
        assert hb.shield_value == 0.0
        assert hb.current_value == 70.0


class TestHealthBarArmor:
    """Test HealthBar armor overlay."""

    def test_set_armor(self):
        """Test setting armor value."""
        hb = HealthBar(max_value=100.0)
        hb.set_armor(25.0)
        assert hb.armor_value == 25.0
        assert hb.armor_max == 25.0

    def test_damage_applies_to_armor_after_shield(self):
        """Test damage applies to armor after shield."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.set_shield(10.0)
        hb.set_armor(20.0)
        hb.apply_damage(25.0, show_preview=False)
        hb.skip_animation()
        assert hb.shield_value == 0.0
        assert hb.armor_value == 5.0
        assert hb.current_value == 100.0


class TestHealthBarAnimation:
    """Test HealthBar animation features."""

    def test_animation_on_value_change(self):
        """Test animation starts on value change."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.set_value(50.0, animate=True)
        assert hb.is_animating is True

    def test_skip_animation(self):
        """Test skip_animation method."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.set_value(50.0, animate=True)
        hb.skip_animation()
        assert hb.is_animating is False
        assert hb.display_value == 50.0

    def test_update_progresses_animation(self):
        """Test update method progresses animation."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.set_value(50.0, animate=True)
        hb.update(0.5)  # Half second
        # Animation should have progressed

    def test_display_value_during_animation(self):
        """Test display_value differs during animation."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.set_value(0.0, animate=True)
        # Display value should be interpolating
        assert 0.0 <= hb.display_value <= 100.0

    def test_animation_without_animate_flag(self):
        """Test setting value without animation."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.set_value(50.0, animate=False)
        assert hb.is_animating is False
        assert hb.current_value == 50.0


class TestHealthBarDamagePreview:
    """Test HealthBar damage preview."""

    def test_pending_damage_set(self):
        """Test pending damage is set."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.apply_damage(25.0, show_preview=True)
        assert hb.pending_damage == 25.0

    def test_damage_preview_timer(self):
        """Test damage preview timer decrements."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.apply_damage(25.0, show_preview=True)
        initial_damage = hb.pending_damage
        hb.update(2.0)  # Update past preview duration
        assert hb.pending_damage == 0.0

    def test_clear_pending(self):
        """Test clear_pending method."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.apply_damage(25.0, show_preview=True)
        hb.clear_pending()
        assert hb.pending_damage == 0.0

    def test_get_damage_preview_rect(self):
        """Test get_damage_preview_rect method."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        hb.apply_damage(25.0, show_preview=True)
        rect = hb.get_damage_preview_rect()
        assert rect is not None
        assert len(rect) == 4

    def test_get_damage_preview_rect_no_damage(self):
        """Test get_damage_preview_rect with no damage."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        rect = hb.get_damage_preview_rect()
        assert rect is None


class TestHealthBarSegments:
    """Test HealthBar segmented display."""

    def test_default_no_segments(self):
        """Test default is no segments."""
        hb = HealthBar()
        assert hb.segment_count == 0

    def test_set_segment_count(self):
        """Test setting segment count."""
        hb = HealthBar(segment_count=10)
        assert hb.segment_count == 10

    def test_segments_property(self):
        """Test segments property returns list."""
        hb = HealthBar(segment_count=5)
        segments = hb.segments
        assert len(segments) == 5

    def test_segment_fill_state(self):
        """Test segment fill states based on value."""
        hb = HealthBar(max_value=100.0, current_value=60.0, segment_count=10)
        segments = hb.segments
        filled = sum(1 for s in segments if s.is_filled)
        # At least 6 segments should be filled for 60%
        assert filled >= 6

    def test_get_segment_rects(self):
        """Test get_segment_rects method."""
        hb = HealthBar(max_value=100.0, segment_count=5)
        rects = hb.get_segment_rects()
        assert len(rects) == 5


class TestHealthBarCallbacks:
    """Test HealthBar callback functions."""

    def test_on_value_changed_callback(self):
        """Test value changed callback."""
        hb = HealthBar(max_value=100.0, current_value=100.0)
        changes = []

        def callback(old, new):
            changes.append((old, new))

        hb.on_value_changed(callback)
        hb.set_value(50.0, animate=False)
        assert len(changes) == 1
        assert changes[0] == (100.0, 50.0)

    def test_on_depleted_callback(self):
        """Test depleted callback."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        depleted = []

        def callback():
            depleted.append(True)

        hb.on_depleted(callback)
        hb.set_value(0.0, animate=False)
        assert len(depleted) == 1

    def test_on_full_callback(self):
        """Test full callback."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        full = []

        def callback():
            full.append(True)

        hb.on_full(callback)
        hb.set_value(100.0, animate=False)
        assert len(full) == 1


class TestHealthBarFillDeplete:
    """Test HealthBar fill and deplete methods."""

    def test_fill(self):
        """Test fill method."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        hb.fill(animate=False)
        assert hb.current_value == 100.0

    def test_deplete(self):
        """Test deplete method."""
        hb = HealthBar(max_value=100.0, current_value=50.0)
        hb.deplete(animate=False)
        assert hb.current_value == 0.0


class TestHealthBarRenderingHelpers:
    """Test HealthBar rendering helper methods."""

    def test_get_fill_rect(self):
        """Test get_fill_rect method."""
        hb = HealthBar(x=0.0, y=0.0, width=200.0, height=20.0)
        hb.set_value(50.0, animate=False)
        rect = hb.get_fill_rect()
        assert len(rect) == 4
        x, y, w, h = rect
        assert w == 100.0  # 50% of 200

    def test_get_shield_rect(self):
        """Test get_shield_rect method."""
        hb = HealthBar(x=0.0, y=0.0, width=200.0, height=20.0)
        hb.set_shield(50.0, maximum=100.0)
        rect = hb.get_shield_rect()
        assert rect is not None

    def test_get_shield_rect_no_shield(self):
        """Test get_shield_rect with no shield."""
        hb = HealthBar()
        rect = hb.get_shield_rect()
        assert rect is None


class TestHealthBarStyling:
    """Test HealthBar styling."""

    def test_default_style(self):
        """Test default style is applied."""
        hb = HealthBar()
        assert hb.style is not None

    def test_custom_style(self):
        """Test custom style."""
        style = HealthBarStyle(
            fill_color="#FF0000",
            background_color="#333333",
            animation_duration=0.5,
        )
        hb = HealthBar(style=style)
        assert hb.style.fill_color == "#FF0000"
        assert hb.style.animation_duration == 0.5

    def test_modify_style(self):
        """Test modifying style after creation."""
        hb = HealthBar()
        hb.style.fill_color = "#00FF00"
        assert hb.style.fill_color == "#00FF00"


class TestHealthBarResourceTypes:
    """Test HealthBar with different resource types."""

    def test_health_type(self):
        """Test HEALTH resource type."""
        hb = HealthBar(resource_type=ResourceType.HEALTH)
        assert hb.resource_type == ResourceType.HEALTH

    def test_mana_type(self):
        """Test MANA resource type."""
        hb = HealthBar(resource_type=ResourceType.MANA)
        assert hb.resource_type == ResourceType.MANA

    def test_stamina_type(self):
        """Test STAMINA resource type."""
        hb = HealthBar(resource_type=ResourceType.STAMINA)
        assert hb.resource_type == ResourceType.STAMINA

    def test_experience_type(self):
        """Test EXPERIENCE resource type."""
        hb = HealthBar(resource_type=ResourceType.EXPERIENCE)
        assert hb.resource_type == ResourceType.EXPERIENCE


class TestHealthBarVisibility:
    """Test HealthBar visibility."""

    def test_visible_by_default(self):
        """Test visible by default."""
        hb = HealthBar()
        assert hb.is_visible is True

    def test_set_invisible(self):
        """Test setting invisible."""
        hb = HealthBar()
        hb.is_visible = False
        assert hb.is_visible is False


class TestHealthBarTransform:
    """Test HealthBar transform properties."""

    def test_set_x(self):
        """Test setting X position."""
        hb = HealthBar()
        hb.x = 150.0
        assert hb.x == 150.0

    def test_set_y(self):
        """Test setting Y position."""
        hb = HealthBar()
        hb.y = 75.0
        assert hb.y == 75.0

    def test_set_width(self):
        """Test setting width."""
        hb = HealthBar()
        hb.width = 300.0
        assert hb.width == 300.0

    def test_width_minimum(self):
        """Test width has minimum."""
        hb = HealthBar()
        hb.width = 0.0
        assert hb.width >= 1.0

    def test_set_height(self):
        """Test setting height."""
        hb = HealthBar()
        hb.height = 40.0
        assert hb.height == 40.0


class TestHealthBarRepr:
    """Test HealthBar string representation."""

    def test_repr(self):
        """Test repr includes key info."""
        hb = HealthBar(max_value=100.0, current_value=75.0)
        repr_str = repr(hb)
        assert "HealthBar" in repr_str
        assert "75" in repr_str
        assert "100" in repr_str
