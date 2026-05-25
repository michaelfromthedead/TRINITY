"""Tests for display management."""

import pytest

from engine.platform.window import Display, DisplayMode, DisplayInfo, Rect


class TestDisplayEnumeration:
    """Tests for display enumeration."""

    def test_enumerate_displays(self):
        """Test enumerating displays."""
        displays = Display.enumerate()
        assert len(displays) > 0
        assert all(isinstance(d, Display) for d in displays)

    def test_primary_display(self):
        """Test getting primary display."""
        primary = Display.primary()
        assert primary is not None
        assert primary.is_primary

    def test_multiple_displays(self):
        """Test that multiple displays are available."""
        displays = Display.enumerate()
        assert len(displays) >= 2  # Primary + Secondary in headless mode

    def test_primary_in_enumeration(self):
        """Test that primary display is in enumeration."""
        displays = Display.enumerate()
        primary = Display.primary()
        # Check that primary display is in the list
        assert any(d.is_primary for d in displays)


class TestDisplayProperties:
    """Tests for display properties."""

    def test_display_name(self):
        """Test display name property."""
        display = Display.primary()
        assert isinstance(display.name, str)
        assert len(display.name) > 0
        # In headless mode, should be "Headless Primary Display"
        assert "Headless" in display.name or "Primary" in display.name

    def test_display_bounds(self):
        """Test display bounds property."""
        display = Display.primary()
        bounds = display.bounds
        assert isinstance(bounds, Rect)
        assert bounds.width > 0
        assert bounds.height > 0

    def test_display_work_area(self):
        """Test display work area property."""
        display = Display.primary()
        work_area = display.work_area
        assert isinstance(work_area, Rect)
        assert work_area.width > 0
        assert work_area.height > 0
        # Work area should be <= bounds
        assert work_area.width <= display.bounds.width
        assert work_area.height <= display.bounds.height

    def test_display_dpi_scale(self):
        """Test display DPI scale property."""
        display = Display.primary()
        dpi_scale = display.dpi_scale
        assert isinstance(dpi_scale, float)
        assert dpi_scale > 0.0
        # Headless mode should use default DPI scale of 1.0
        from engine.platform.window.display import DEFAULT_DPI_SCALE
        assert dpi_scale == DEFAULT_DPI_SCALE

    def test_display_is_primary(self):
        """Test display is_primary property."""
        displays = Display.enumerate()
        primary_count = sum(1 for d in displays if d.is_primary)
        assert primary_count == 1  # Exactly one primary display


class TestDisplayModes:
    """Tests for display modes."""

    def test_supported_modes(self):
        """Test getting supported display modes."""
        display = Display.primary()
        modes = display.supported_modes()
        assert len(modes) > 0
        assert all(isinstance(m, DisplayMode) for m in modes)

    def test_current_mode(self):
        """Test getting current display mode."""
        display = Display.primary()
        current = display.current_mode()
        assert isinstance(current, DisplayMode)
        assert current.width > 0
        assert current.height > 0
        assert current.refresh_rate > 0

    def test_mode_in_supported_list(self):
        """Test that current mode is in supported modes."""
        display = Display.primary()
        current = display.current_mode()
        supported = display.supported_modes()
        # Current mode should match one of the supported modes
        matches = [
            m for m in supported
            if m.width == current.width
            and m.height == current.height
            and m.refresh_rate == current.refresh_rate
        ]
        assert len(matches) > 0

    def test_multiple_refresh_rates(self):
        """Test that displays support multiple refresh rates."""
        display = Display.primary()
        modes = display.supported_modes()
        refresh_rates = {m.refresh_rate for m in modes}
        assert len(refresh_rates) > 1  # Should have multiple refresh rates


class TestDisplayMode:
    """Tests for DisplayMode dataclass."""

    def test_display_mode_creation(self):
        """Test creating a display mode."""
        mode = DisplayMode(1920, 1080, 60)
        assert mode.width == 1920
        assert mode.height == 1080
        assert mode.refresh_rate == 60
        assert mode.format == "RGBA8888"

    def test_display_mode_with_format(self):
        """Test creating display mode with custom format."""
        mode = DisplayMode(3840, 2160, 120, format="RGB10A2")
        assert mode.width == 3840
        assert mode.height == 2160
        assert mode.refresh_rate == 120
        assert mode.format == "RGB10A2"


class TestDisplayInfo:
    """Tests for DisplayInfo dataclass."""

    def test_display_info_creation(self):
        """Test creating display info."""
        info = DisplayInfo(
            name="Test Display",
            bounds=Rect(0, 0, 1920, 1080),
            work_area=Rect(0, 0, 1920, 1040),
            dpi_scale=1.0,
            is_primary=True
        )
        assert info.name == "Test Display"
        assert info.bounds.width == 1920
        assert info.bounds.height == 1080
        assert info.work_area.height == 1040
        assert info.dpi_scale == 1.0
        assert info.is_primary is True


class TestSecondaryDisplay:
    """Tests for secondary display."""

    def test_secondary_display_exists(self):
        """Test that a secondary display exists."""
        displays = Display.enumerate()
        secondary = [d for d in displays if not d.is_primary]
        assert len(secondary) > 0

    def test_secondary_display_properties(self):
        """Test secondary display properties."""
        displays = Display.enumerate()
        secondary = [d for d in displays if not d.is_primary][0]
        assert secondary.name is not None
        assert secondary.bounds.width > 0
        assert secondary.bounds.height > 0
        assert not secondary.is_primary

    def test_secondary_display_offset(self):
        """Test that secondary display has different bounds."""
        primary = Display.primary()
        displays = Display.enumerate()
        secondary = [d for d in displays if not d.is_primary][0]
        # Secondary should have different position
        assert (secondary.bounds.x != primary.bounds.x or
                secondary.bounds.y != primary.bounds.y)
