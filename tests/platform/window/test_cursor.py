"""Tests for cursor management."""

import pytest

from engine.platform.window import CursorManager, CursorType


class TestCursorManager:
    """Tests for CursorManager class."""

    def test_cursor_manager_creation(self):
        """Test creating a cursor manager with default state."""
        manager = CursorManager()
        assert manager is not None

        # Verify default cursor type
        assert manager.current_type == CursorType.ARROW

        # Verify default visibility
        assert manager.visible is True

        # Verify default confinement
        assert manager.confined is False

        # Verify set_cursor changes current_type
        manager.set_cursor(CursorType.HAND)
        assert manager.current_type == CursorType.HAND

        manager.set_cursor(CursorType.CROSSHAIR)
        assert manager.current_type == CursorType.CROSSHAIR

    def test_default_cursor_type(self):
        """Test default cursor type is arrow."""
        manager = CursorManager()
        assert manager.current_type == CursorType.ARROW


class TestCursorType:
    """Tests for cursor type management."""

    def test_set_cursor_arrow(self):
        """Test setting arrow cursor."""
        manager = CursorManager()
        manager.set_cursor(CursorType.ARROW)
        assert manager.current_type == CursorType.ARROW

    def test_set_cursor_hand(self):
        """Test setting hand cursor."""
        manager = CursorManager()
        manager.set_cursor(CursorType.HAND)
        assert manager.current_type == CursorType.HAND

    def test_set_cursor_ibeam(self):
        """Test setting I-beam cursor."""
        manager = CursorManager()
        manager.set_cursor(CursorType.IBEAM)
        assert manager.current_type == CursorType.IBEAM

    def test_set_cursor_crosshair(self):
        """Test setting crosshair cursor."""
        manager = CursorManager()
        manager.set_cursor(CursorType.CROSSHAIR)
        assert manager.current_type == CursorType.CROSSHAIR

    def test_set_cursor_resize(self):
        """Test setting resize cursors."""
        manager = CursorManager()

        manager.set_cursor(CursorType.RESIZE_NS)
        assert manager.current_type == CursorType.RESIZE_NS

        manager.set_cursor(CursorType.RESIZE_EW)
        assert manager.current_type == CursorType.RESIZE_EW

        manager.set_cursor(CursorType.RESIZE_NESW)
        assert manager.current_type == CursorType.RESIZE_NESW

        manager.set_cursor(CursorType.RESIZE_NWSE)
        assert manager.current_type == CursorType.RESIZE_NWSE

        manager.set_cursor(CursorType.RESIZE_ALL)
        assert manager.current_type == CursorType.RESIZE_ALL

    def test_set_cursor_special_types(self):
        """Test setting special cursor types."""
        manager = CursorManager()

        manager.set_cursor(CursorType.NOT_ALLOWED)
        assert manager.current_type == CursorType.NOT_ALLOWED

        manager.set_cursor(CursorType.WAIT)
        assert manager.current_type == CursorType.WAIT

        manager.set_cursor(CursorType.WAIT_ARROW)
        assert manager.current_type == CursorType.WAIT_ARROW


class TestCursorVisibility:
    """Tests for cursor visibility management."""

    def test_default_visibility(self):
        """Test default cursor visibility."""
        manager = CursorManager()
        assert manager.visible is True

    def test_hide_cursor(self):
        """Test hiding cursor."""
        manager = CursorManager()
        manager.set_visible(False)
        assert manager.visible is False

    def test_show_cursor(self):
        """Test showing cursor."""
        manager = CursorManager()
        manager.set_visible(False)
        manager.set_visible(True)
        assert manager.visible is True

    def test_toggle_visibility(self):
        """Test toggling cursor visibility."""
        manager = CursorManager()
        original = manager.visible
        manager.set_visible(not original)
        assert manager.visible != original


class TestCursorConfinement:
    """Tests for cursor confinement."""

    def test_default_confinement(self):
        """Test default cursor confinement."""
        manager = CursorManager()
        assert manager.confined is False

    def test_confine_cursor(self):
        """Test confining cursor to window."""
        manager = CursorManager()
        manager.confine(True)
        assert manager.confined is True

    def test_release_cursor(self):
        """Test releasing confined cursor."""
        manager = CursorManager()
        manager.confine(True)
        manager.confine(False)
        assert manager.confined is False

    def test_toggle_confinement(self):
        """Test toggling cursor confinement."""
        manager = CursorManager()
        original = manager.confined
        manager.confine(not original)
        assert manager.confined != original


class TestCustomCursor:
    """Tests for custom cursor management."""

    def test_set_custom_cursor(self):
        """Test setting custom cursor."""
        manager = CursorManager()
        image_data = b'\x00\x01\x02\x03' * 64  # Mock image data
        manager.set_custom_cursor(image_data, 0, 0)
        assert manager.has_custom_cursor

    def test_custom_cursor_clears_type(self):
        """Test that setting custom cursor clears cursor type."""
        manager = CursorManager()
        manager.set_cursor(CursorType.HAND)
        image_data = b'\x00\x01\x02\x03' * 64
        manager.set_custom_cursor(image_data, 0, 0)
        assert manager.has_custom_cursor
        assert manager.current_type == CursorType.ARROW

    def test_set_cursor_clears_custom(self):
        """Test that setting cursor type clears custom cursor."""
        manager = CursorManager()
        image_data = b'\x00\x01\x02\x03' * 64
        manager.set_custom_cursor(image_data, 0, 0)
        assert manager.has_custom_cursor
        manager.set_cursor(CursorType.HAND)
        assert not manager.has_custom_cursor

    def test_custom_cursor_hotspot(self):
        """Test custom cursor with different hotspot coordinates."""
        manager = CursorManager()
        image_data = b'\x00\x01\x02\x03' * 64
        # Just verify it doesn't crash with different hotspot values
        manager.set_custom_cursor(image_data, 16, 16)
        assert manager.has_custom_cursor


class TestCursorState:
    """Tests for cursor state management."""

    def test_independent_properties(self):
        """Test that cursor properties are independent."""
        manager = CursorManager()
        manager.set_cursor(CursorType.CROSSHAIR)
        manager.set_visible(False)
        manager.confine(True)

        assert manager.current_type == CursorType.CROSSHAIR
        assert manager.visible is False
        assert manager.confined is True

    def test_state_persistence(self):
        """Test that cursor state persists across operations."""
        manager = CursorManager()
        manager.set_visible(False)
        manager.set_cursor(CursorType.HAND)
        assert manager.visible is False  # Should still be hidden

        manager.confine(True)
        assert manager.visible is False  # Should still be hidden
        assert manager.current_type == CursorType.HAND  # Should still be hand
