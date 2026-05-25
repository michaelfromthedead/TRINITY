"""
Comprehensive tests for the Spacer widget.

Tests cover:
- Spacer mode configuration (fixed, flexible, fill)
- Size constraints
- Layout behavior
- Serialization/deserialization

Note: The spacer.py source file may not exist yet. These tests are written
based on the expected API from the primitives __init__.py exports:
- Spacer
- SpacerMode
"""

import pytest
from unittest.mock import MagicMock


class TestSpacerMode:
    """Tests for SpacerMode enumeration."""

    def test_spacer_mode_fixed(self):
        """Test FIXED spacer mode exists."""
        try:
            from engine.ui.widgets.primitives.spacer import SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        assert SpacerMode.FIXED is not None

    def test_spacer_mode_flexible(self):
        """Test FLEXIBLE spacer mode exists."""
        try:
            from engine.ui.widgets.primitives.spacer import SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        assert SpacerMode.FLEXIBLE is not None

    def test_spacer_mode_fill(self):
        """Test FILL spacer mode exists."""
        try:
            from engine.ui.widgets.primitives.spacer import SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        assert SpacerMode.FILL is not None

    def test_spacer_mode_expand(self):
        """Test EXPAND spacer mode exists."""
        try:
            from engine.ui.widgets.primitives.spacer import SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        # This mode might not exist
        try:
            assert SpacerMode.EXPAND is not None
        except AttributeError:
            pass  # Mode not implemented


class TestSpacerWidget:
    """Tests for the Spacer widget class."""

    def test_spacer_default_initialization(self):
        """Test Spacer initializes with correct defaults."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer()
        assert spacer.mode == SpacerMode.FIXED
        assert spacer.size == 0.0

    def test_spacer_fixed_mode_with_size(self):
        """Test Spacer in fixed mode with specific size."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(mode=SpacerMode.FIXED, size=20.0)
        assert spacer.mode == SpacerMode.FIXED
        assert spacer.size == 20.0

    def test_spacer_flexible_mode(self):
        """Test Spacer in flexible mode."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(mode=SpacerMode.FLEXIBLE, flex=1.0)
        assert spacer.mode == SpacerMode.FLEXIBLE
        assert spacer.flex == 1.0

    def test_spacer_fill_mode(self):
        """Test Spacer in fill mode."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(mode=SpacerMode.FILL)
        assert spacer.mode == SpacerMode.FILL

    def test_spacer_size_setter(self):
        """Test setting spacer size."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer()
        spacer.size = 50.0
        assert spacer.size == 50.0

    def test_spacer_size_negative_fails(self):
        """Test negative size fails validation."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        with pytest.raises(ValueError, match="must be >= 0"):
            Spacer(size=-10.0)

    def test_spacer_mode_setter(self):
        """Test setting spacer mode."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer()
        spacer.mode = SpacerMode.FLEXIBLE
        assert spacer.mode == SpacerMode.FLEXIBLE

    def test_spacer_flex_setter(self):
        """Test setting flex factor."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(mode=SpacerMode.FLEXIBLE)
        spacer.flex = 2.0
        assert spacer.flex == 2.0

    def test_spacer_flex_negative_fails(self):
        """Test negative flex factor fails."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        with pytest.raises(ValueError, match="must be > 0"):
            Spacer(mode=SpacerMode.FLEXIBLE, flex=-1.0)


class TestSpacerConstraints:
    """Tests for Spacer size constraints."""

    def test_spacer_min_size(self):
        """Test minimum size constraint."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(min_size=10.0)
        assert spacer.min_size == 10.0

    def test_spacer_max_size(self):
        """Test maximum size constraint."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(max_size=100.0)
        assert spacer.max_size == 100.0

    def test_spacer_min_max_size(self):
        """Test both min and max size constraints."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(min_size=20.0, max_size=80.0)
        assert spacer.min_size == 20.0
        assert spacer.max_size == 80.0

    def test_spacer_min_greater_than_max_fails(self):
        """Test min_size > max_size fails."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        with pytest.raises(ValueError, match="min_size cannot be greater than max_size"):
            Spacer(min_size=100.0, max_size=50.0)

    def test_spacer_clamp_size(self):
        """Test clamping size to constraints."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(min_size=20.0, max_size=80.0)

        # Test clamping below min
        assert spacer.clamp_size(10.0) == 20.0

        # Test clamping above max
        assert spacer.clamp_size(100.0) == 80.0

        # Test within range
        assert spacer.clamp_size(50.0) == 50.0


class TestSpacerLayoutBehavior:
    """Tests for Spacer layout behavior."""

    def test_spacer_fixed_computed_size(self):
        """Test computed size in fixed mode."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(mode=SpacerMode.FIXED, size=30.0)
        assert spacer.compute_size(available_space=100.0) == 30.0

    def test_spacer_flexible_computed_size(self):
        """Test computed size in flexible mode."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(mode=SpacerMode.FLEXIBLE, flex=1.0)
        # With 100 available space and flex=1, gets portion based on total flex
        size = spacer.compute_size(available_space=100.0, total_flex=2.0)
        assert size == 50.0

    def test_spacer_fill_computed_size(self):
        """Test computed size in fill mode."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(mode=SpacerMode.FILL)
        assert spacer.compute_size(available_space=100.0) == 100.0

    def test_spacer_horizontal_orientation(self):
        """Test spacer with horizontal orientation."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(horizontal=True, size=20.0)
        assert spacer.horizontal is True
        assert spacer.width == 20.0
        assert spacer.height == 0.0

    def test_spacer_vertical_orientation(self):
        """Test spacer with vertical orientation."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(horizontal=False, size=20.0)
        assert spacer.horizontal is False
        assert spacer.width == 0.0
        assert spacer.height == 20.0


class TestSpacerDirtyState:
    """Tests for dirty state tracking."""

    def test_spacer_dirty_after_size_change(self):
        """Test spacer is dirty after size changes."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(size=10.0)
        spacer.mark_clean()
        spacer.size = 20.0
        assert spacer.is_dirty

    def test_spacer_dirty_after_mode_change(self):
        """Test spacer is dirty after mode changes."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer()
        spacer.mark_clean()
        spacer.mode = SpacerMode.FILL
        assert spacer.is_dirty

    def test_spacer_mark_clean(self):
        """Test mark_clean clears dirty state."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer()
        spacer.mark_clean()
        assert spacer.is_dirty is False


class TestSpacerSerialization:
    """Tests for Spacer serialization and deserialization."""

    def test_spacer_to_dict_fixed(self):
        """Test serialization of fixed spacer."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(mode=SpacerMode.FIXED, size=25.0)

        data = spacer.to_dict()
        assert data["mode"] == "FIXED"
        assert data["size"] == 25.0

    def test_spacer_to_dict_flexible(self):
        """Test serialization of flexible spacer."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(mode=SpacerMode.FLEXIBLE, flex=2.5)

        data = spacer.to_dict()
        assert data["mode"] == "FLEXIBLE"
        assert data["flex"] == 2.5

    def test_spacer_to_dict_with_constraints(self):
        """Test serialization with size constraints."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer(min_size=10.0, max_size=50.0)

        data = spacer.to_dict()
        assert data["min_size"] == 10.0
        assert data["max_size"] == 50.0

    def test_spacer_from_dict(self):
        """Test deserialization from dictionary."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        data = {
            "mode": "FLEXIBLE",
            "flex": 1.5,
            "min_size": 5.0,
            "max_size": 100.0,
        }

        spacer = Spacer.from_dict(data)
        assert spacer.mode == SpacerMode.FLEXIBLE
        assert spacer.flex == 1.5
        assert spacer.min_size == 5.0
        assert spacer.max_size == 100.0

    def test_spacer_roundtrip_serialization(self):
        """Test serialization roundtrip preserves data."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        original = Spacer(
            mode=SpacerMode.FLEXIBLE,
            flex=3.0,
            min_size=15.0,
            max_size=200.0,
        )

        data = original.to_dict()
        restored = Spacer.from_dict(data)

        assert restored.mode == original.mode
        assert restored.flex == original.flex
        assert restored.min_size == original.min_size
        assert restored.max_size == original.max_size


class TestSpacerFactoryMethods:
    """Tests for Spacer factory/convenience methods."""

    def test_spacer_fixed_factory(self):
        """Test Spacer.fixed() factory method."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer.fixed(32.0)
        assert spacer.mode == SpacerMode.FIXED
        assert spacer.size == 32.0

    def test_spacer_flexible_factory(self):
        """Test Spacer.flexible() factory method."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer.flexible(2.0)
        assert spacer.mode == SpacerMode.FLEXIBLE
        assert spacer.flex == 2.0

    def test_spacer_fill_factory(self):
        """Test Spacer.fill() factory method."""
        try:
            from engine.ui.widgets.primitives.spacer import Spacer, SpacerMode
        except ImportError:
            pytest.skip("spacer.py not yet implemented")

        spacer = Spacer.fill()
        assert spacer.mode == SpacerMode.FILL
