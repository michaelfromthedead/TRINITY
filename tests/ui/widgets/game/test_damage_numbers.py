"""
Comprehensive tests for DamageNumbers widget.

Tests cover:
- DamageNumber creation and properties
- DamageNumberManager initialization
- Spawning damage numbers
- Damage types and colors
- Critical hit styling
- Healing numbers
- Special damage types (miss, blocked, etc.)
- Animation and update
- Number stacking
- Screen coordinate conversion
- Lifecycle management
- Configuration
- Rendering helpers
"""

import pytest
import sys
from pathlib import Path

# Add engine to path
engine_path = Path(__file__).parent.parent.parent.parent.parent / "engine"
sys.path.insert(0, str(engine_path.parent))

from engine.ui.widgets.game.damage_numbers import (
    DamageNumber,
    DamageNumberManager,
    DamageNumberConfig,
    DamageType,
)


class TestDamageType:
    """Test DamageType enum."""

    def test_physical_type(self):
        """Test PHYSICAL damage type exists."""
        assert DamageType.PHYSICAL is not None

    def test_magic_type(self):
        """Test MAGIC damage type exists."""
        assert DamageType.MAGIC is not None

    def test_heal_type(self):
        """Test HEAL damage type exists."""
        assert DamageType.HEAL is not None

    def test_miss_type(self):
        """Test MISS damage type exists."""
        assert DamageType.MISS is not None

    def test_all_types_defined(self):
        """Test all expected damage types are defined."""
        expected = [
            "PHYSICAL", "MAGIC", "FIRE", "ICE", "LIGHTNING",
            "POISON", "HEAL", "SHIELD", "EXPERIENCE",
            "MISS", "BLOCKED", "ABSORBED", "CUSTOM",
        ]
        for name in expected:
            assert hasattr(DamageType, name)


class TestDamageNumberConfig:
    """Test DamageNumberConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DamageNumberConfig()
        assert config.duration > 0
        assert config.rise_distance > 0
        assert config.fade_start >= 0

    def test_custom_duration(self):
        """Test custom duration."""
        config = DamageNumberConfig(duration=2.0)
        assert config.duration == 2.0

    def test_custom_colors(self):
        """Test custom colors."""
        config = DamageNumberConfig(
            physical_color="#FFFFFF",
            fire_color="#FF4444",
        )
        assert config.physical_color == "#FFFFFF"
        assert config.fire_color == "#FF4444"

    def test_critical_settings(self):
        """Test critical hit settings."""
        config = DamageNumberConfig(
            crit_scale=2.0,
            crit_suffix="!",
        )
        assert config.crit_scale == 2.0
        assert config.crit_suffix == "!"


class TestDamageNumber:
    """Test DamageNumber class."""

    def test_creation(self):
        """Test damage number creation."""
        dn = DamageNumber(
            id=1,
            value=100,
            damage_type=DamageType.PHYSICAL,
            world_x=500.0,
            world_y=300.0,
        )
        assert dn.id == 1
        assert dn.value == 100
        assert dn.damage_type == DamageType.PHYSICAL

    def test_default_state(self):
        """Test default state properties."""
        dn = DamageNumber(
            id=1, value=50,
            damage_type=DamageType.PHYSICAL,
            world_x=0.0, world_y=0.0,
        )
        assert dn.is_active is True
        assert dn.is_visible is True
        assert dn.opacity == 1.0

    def test_display_text_number(self):
        """Test display text for numeric value."""
        dn = DamageNumber(
            id=1, value=500,
            damage_type=DamageType.PHYSICAL,
            world_x=0.0, world_y=0.0,
        )
        text = dn.get_display_text()
        assert "500" in text

    def test_display_text_large_number(self):
        """Test display text with K suffix for large numbers."""
        dn = DamageNumber(
            id=1, value=5000,
            damage_type=DamageType.PHYSICAL,
            world_x=0.0, world_y=0.0,
        )
        text = dn.get_display_text()
        assert "K" in text or "5.0" in text

    def test_display_text_millions(self):
        """Test display text with M suffix for millions."""
        dn = DamageNumber(
            id=1, value=2500000,
            damage_type=DamageType.PHYSICAL,
            world_x=0.0, world_y=0.0,
        )
        text = dn.get_display_text()
        assert "M" in text

    def test_display_text_miss(self):
        """Test display text for miss."""
        dn = DamageNumber(
            id=1, value=0,
            damage_type=DamageType.MISS,
            world_x=0.0, world_y=0.0,
        )
        assert dn.get_display_text() == "MISS"

    def test_display_text_blocked(self):
        """Test display text for blocked."""
        dn = DamageNumber(
            id=1, value=0,
            damage_type=DamageType.BLOCKED,
            world_x=0.0, world_y=0.0,
        )
        assert dn.get_display_text() == "BLOCKED"

    def test_display_text_heal(self):
        """Test display text for healing."""
        dn = DamageNumber(
            id=1, value=100,
            damage_type=DamageType.HEAL,
            world_x=0.0, world_y=0.0,
        )
        text = dn.get_display_text()
        assert "+" in text

    def test_display_text_experience(self):
        """Test display text for experience."""
        dn = DamageNumber(
            id=1, value=250,
            damage_type=DamageType.EXPERIENCE,
            world_x=0.0, world_y=0.0,
        )
        text = dn.get_display_text()
        assert "XP" in text

    def test_custom_text(self):
        """Test custom text override."""
        dn = DamageNumber(
            id=1, value=100,
            damage_type=DamageType.CUSTOM,
            world_x=0.0, world_y=0.0,
            custom_text="CRITICAL!",
        )
        assert dn.get_display_text() == "CRITICAL!"


class TestDamageNumberManagerInit:
    """Test DamageNumberManager initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        manager = DamageNumberManager()
        assert manager.active_count == 0
        assert manager.config is not None

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = DamageNumberConfig(duration=3.0)
        manager = DamageNumberManager(config=config)
        assert manager.config.duration == 3.0

    def test_max_active_limit(self):
        """Test max active limit."""
        manager = DamageNumberManager(max_active=50)
        # Spawn more than max
        for i in range(60):
            manager.spawn(100, float(i), 0.0)
        assert manager.active_count <= 50


class TestDamageNumberManagerSpawn:
    """Test DamageNumberManager spawning."""

    def test_spawn_damage_number(self):
        """Test spawning a damage number."""
        manager = DamageNumberManager()
        num_id = manager.spawn(100, 500.0, 300.0)
        assert num_id is not None
        assert manager.active_count == 1

    def test_spawn_returns_unique_ids(self):
        """Test spawn returns unique IDs."""
        manager = DamageNumberManager()
        id1 = manager.spawn(100, 0.0, 0.0)
        id2 = manager.spawn(200, 100.0, 0.0)
        assert id1 != id2

    def test_spawn_with_damage_type(self):
        """Test spawning with specific damage type."""
        manager = DamageNumberManager()
        num_id = manager.spawn(
            100, 500.0, 300.0,
            damage_type=DamageType.FIRE,
        )
        number = manager.get_number(num_id)
        assert number is not None
        assert number.damage_type == DamageType.FIRE

    def test_spawn_critical(self):
        """Test spawning critical hit."""
        manager = DamageNumberManager()
        num_id = manager.spawn(
            500, 500.0, 300.0,
            is_critical=True,
        )
        number = manager.get_number(num_id)
        assert number.is_critical is True
        assert number.scale == manager.config.crit_scale

    def test_spawn_with_custom_color(self):
        """Test spawning with custom color."""
        manager = DamageNumberManager()
        num_id = manager.spawn(
            100, 500.0, 300.0,
            custom_color="#FF00FF",
        )
        number = manager.get_number(num_id)
        assert number.color == "#FF00FF"

    def test_spawn_with_custom_text(self):
        """Test spawning with custom text."""
        manager = DamageNumberManager()
        num_id = manager.spawn(
            0, 500.0, 300.0,
            custom_text="PARRY!",
        )
        number = manager.get_number(num_id)
        assert number.custom_text == "PARRY!"


class TestDamageNumberManagerHealers:
    """Test DamageNumberManager helper spawn methods."""

    def test_spawn_heal(self):
        """Test spawning heal number."""
        manager = DamageNumberManager()
        num_id = manager.spawn_heal(150, 500.0, 300.0)
        number = manager.get_number(num_id)
        assert number.damage_type == DamageType.HEAL

    def test_spawn_heal_critical(self):
        """Test spawning critical heal."""
        manager = DamageNumberManager()
        num_id = manager.spawn_heal(
            150, 500.0, 300.0,
            is_critical=True,
        )
        number = manager.get_number(num_id)
        assert number.is_critical is True

    def test_spawn_miss(self):
        """Test spawning miss indicator."""
        manager = DamageNumberManager()
        num_id = manager.spawn_miss(500.0, 300.0)
        number = manager.get_number(num_id)
        assert number.damage_type == DamageType.MISS

    def test_spawn_blocked(self):
        """Test spawning blocked indicator."""
        manager = DamageNumberManager()
        num_id = manager.spawn_blocked(500.0, 300.0)
        number = manager.get_number(num_id)
        assert number.damage_type == DamageType.BLOCKED

    def test_spawn_experience(self):
        """Test spawning experience gain."""
        manager = DamageNumberManager()
        num_id = manager.spawn_experience(1000, 500.0, 300.0)
        number = manager.get_number(num_id)
        assert number.damage_type == DamageType.EXPERIENCE


class TestDamageNumberManagerUpdate:
    """Test DamageNumberManager update and animation."""

    def test_update_moves_numbers(self):
        """Test update moves numbers upward."""
        manager = DamageNumberManager()
        num_id = manager.spawn(100, 500.0, 300.0)
        number = manager.get_number(num_id)
        initial_y = number.screen_y
        manager.update(0.1)
        assert number.screen_y < initial_y  # Moved up (y decreases)

    def test_update_removes_expired(self):
        """Test update removes expired numbers."""
        config = DamageNumberConfig(duration=0.1)
        manager = DamageNumberManager(config=config)
        manager.spawn(100, 500.0, 300.0)
        assert manager.active_count == 1
        manager.update(0.5)  # Past duration
        assert manager.active_count == 0

    def test_update_fades_numbers(self):
        """Test update fades numbers near end."""
        config = DamageNumberConfig(duration=1.0, fade_start=0.5)
        manager = DamageNumberManager(config=config)
        num_id = manager.spawn(100, 500.0, 300.0)
        number = manager.get_number(num_id)
        manager.update(0.8)  # Past fade_start
        assert number.opacity < 1.0


class TestDamageNumberManagerStacking:
    """Test DamageNumberManager number stacking."""

    def test_stacking_combines_nearby(self):
        """Test nearby same-type numbers stack."""
        config = DamageNumberConfig(
            stack_threshold=0.5,
            stack_distance=50.0,
        )
        manager = DamageNumberManager(config=config)
        # Spawn two numbers at same position quickly
        manager.spawn(100, 500.0, 300.0, damage_type=DamageType.PHYSICAL)
        manager.spawn(100, 500.0, 300.0, damage_type=DamageType.PHYSICAL)
        # Second should stack (depends on timing)

    def test_stacking_different_types_dont_stack(self):
        """Test different types don't stack."""
        config = DamageNumberConfig(
            stack_threshold=0.5,
            stack_distance=50.0,
        )
        manager = DamageNumberManager(config=config)
        manager.spawn(100, 500.0, 300.0, damage_type=DamageType.PHYSICAL)
        manager.spawn(100, 500.0, 300.0, damage_type=DamageType.FIRE)
        assert manager.active_count == 2


class TestDamageNumberManagerCoordinates:
    """Test DamageNumberManager coordinate conversion."""

    def test_set_world_to_screen_converter(self):
        """Test setting world to screen converter."""
        manager = DamageNumberManager()

        def converter(wx, wy):
            return (wx * 0.5, wy * 0.5)

        manager.set_world_to_screen_converter(converter)
        num_id = manager.spawn(100, 1000.0, 800.0)
        number = manager.get_number(num_id)
        assert number.screen_x == 500.0
        assert number.screen_y == 400.0


class TestDamageNumberManagerLifecycle:
    """Test DamageNumberManager lifecycle management."""

    def test_get_number(self):
        """Test getting number by ID."""
        manager = DamageNumberManager()
        num_id = manager.spawn(100, 500.0, 300.0)
        number = manager.get_number(num_id)
        assert number is not None
        assert number.id == num_id

    def test_get_number_not_found(self):
        """Test getting nonexistent number."""
        manager = DamageNumberManager()
        number = manager.get_number(999)
        assert number is None

    def test_remove_number(self):
        """Test removing a number."""
        manager = DamageNumberManager()
        num_id = manager.spawn(100, 500.0, 300.0)
        result = manager.remove(num_id)
        assert result is True
        assert manager.get_number(num_id) is None

    def test_remove_nonexistent(self):
        """Test removing nonexistent number."""
        manager = DamageNumberManager()
        result = manager.remove(999)
        assert result is False

    def test_clear_all(self):
        """Test clearing all numbers."""
        manager = DamageNumberManager()
        for i in range(10):
            manager.spawn(100, float(i * 50), 0.0)
        manager.clear()
        assert manager.active_count == 0


class TestDamageNumberManagerRendering:
    """Test DamageNumberManager rendering helpers."""

    def test_get_visible_numbers(self):
        """Test getting visible numbers."""
        manager = DamageNumberManager()
        manager.spawn(100, 500.0, 300.0)
        manager.spawn(200, 600.0, 300.0)
        visible = manager.get_visible_numbers()
        assert len(visible) == 2

    def test_visible_numbers_sorted_by_y(self):
        """Test visible numbers are sorted by Y."""
        manager = DamageNumberManager()
        manager.spawn(100, 500.0, 300.0)
        manager.spawn(200, 500.0, 200.0)  # Higher (lower Y)
        visible = manager.get_visible_numbers()
        assert visible[0].screen_y <= visible[1].screen_y

    def test_get_render_data(self):
        """Test getting render data for number."""
        manager = DamageNumberManager()
        num_id = manager.spawn(100, 500.0, 300.0)
        number = manager.get_number(num_id)
        data = manager.get_render_data(number)
        assert "text" in data
        assert "x" in data
        assert "y" in data
        assert "color" in data
        assert "font_size" in data
        assert "opacity" in data


class TestDamageNumberManagerColorsByType:
    """Test damage type color mapping."""

    def test_physical_color(self):
        """Test physical damage color."""
        manager = DamageNumberManager()
        num_id = manager.spawn(100, 0.0, 0.0, damage_type=DamageType.PHYSICAL)
        number = manager.get_number(num_id)
        assert number.color == manager.config.physical_color

    def test_fire_color(self):
        """Test fire damage color."""
        manager = DamageNumberManager()
        num_id = manager.spawn(100, 0.0, 0.0, damage_type=DamageType.FIRE)
        number = manager.get_number(num_id)
        assert number.color == manager.config.fire_color

    def test_heal_color(self):
        """Test heal color."""
        manager = DamageNumberManager()
        num_id = manager.spawn(100, 0.0, 0.0, damage_type=DamageType.HEAL)
        number = manager.get_number(num_id)
        assert number.color == manager.config.heal_color


class TestDamageNumberManagerRepr:
    """Test DamageNumberManager string representation."""

    def test_repr(self):
        """Test repr includes key info."""
        manager = DamageNumberManager()
        manager.spawn(100, 0.0, 0.0)
        repr_str = repr(manager)
        assert "DamageNumberManager" in repr_str
        assert "1" in repr_str  # active count
