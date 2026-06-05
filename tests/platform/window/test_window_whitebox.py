"""
Whitebox tests for the window subsystem.

Tests window management, display enumeration, cursor management,
HDR support, VRR capabilities, and thread safety.
"""

import pytest
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, '/home/user/dev/USER/PROJECTS_VOID/TRINITY')

from engine.platform.window import (
    Window,
    WindowConfig,
    WindowStyle,
    WindowState,
    WindowEvent,
    WindowEventType,
    FullscreenMode,
    Rect,
    Display,
    DisplayMode,
    DisplayInfo,
    CursorManager,
    CursorType,
    DisplayHDR,
    HDRCapabilities,
    ColorSpace,
    VariableRefresh,
    VRRType,
    RefreshRange,
)
from engine.platform.constants import (
    DEFAULT_WINDOW_X,
    DEFAULT_WINDOW_Y,
    DEFAULT_WINDOW_WIDTH,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_DPI_SCALE,
    STANDARD_RESOLUTIONS,
    STANDARD_REFRESH_RATES,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def window():
    """Provide fresh Window for each test."""
    config = WindowConfig(title="Test Window")
    return Window.create(config)


@pytest.fixture
def display():
    """Provide primary Display for each test."""
    # Reset display state
    Display._displays = []
    Display._initialized = False
    return Display.primary()


@pytest.fixture
def cursor_manager():
    """Provide fresh CursorManager for each test."""
    return CursorManager()


@pytest.fixture
def display_hdr():
    """Provide fresh DisplayHDR for each test."""
    return DisplayHDR()


@pytest.fixture
def variable_refresh():
    """Provide fresh VariableRefresh for each test."""
    return VariableRefresh()


# ============================================================================
# Rect Tests
# ============================================================================

class TestRect:
    """Tests for Rect dataclass."""

    def test_creation(self):
        """Test rect creation."""
        rect = Rect(10, 20, 800, 600)
        assert rect.x == 10
        assert rect.y == 20
        assert rect.width == 800
        assert rect.height == 600

    def test_zero_rect(self):
        """Test zero-sized rect."""
        rect = Rect(0, 0, 0, 0)
        assert rect.x == 0
        assert rect.width == 0

    def test_negative_values(self):
        """Test rect with negative position."""
        rect = Rect(-100, -50, 400, 300)
        assert rect.x == -100
        assert rect.y == -50


# ============================================================================
# WindowConfig Tests
# ============================================================================

class TestWindowConfig:
    """Tests for WindowConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = WindowConfig()
        assert config.title == "Game Window"
        assert config.x == DEFAULT_WINDOW_X
        assert config.y == DEFAULT_WINDOW_Y
        assert config.width == DEFAULT_WINDOW_WIDTH
        assert config.height == DEFAULT_WINDOW_HEIGHT
        assert config.resizable is True
        assert config.fullscreen is False
        assert config.style == WindowStyle.WINDOWED

    def test_custom_values(self):
        """Test custom config values."""
        config = WindowConfig(
            title="Custom",
            x=200,
            y=200,
            width=1920,
            height=1080,
            resizable=False,
            fullscreen=True,
            style=WindowStyle.FULLSCREEN_EXCLUSIVE
        )
        assert config.title == "Custom"
        assert config.width == 1920
        assert config.fullscreen is True


# ============================================================================
# WindowStyle Tests
# ============================================================================

class TestWindowStyle:
    """Tests for WindowStyle enum."""

    def test_styles_exist(self):
        """Verify all styles exist."""
        assert WindowStyle.WINDOWED is not None
        assert WindowStyle.BORDERLESS is not None
        assert WindowStyle.FULLSCREEN_EXCLUSIVE is not None


# ============================================================================
# WindowState Tests
# ============================================================================

class TestWindowState:
    """Tests for WindowState enum."""

    def test_states_exist(self):
        """Verify all states exist."""
        assert WindowState.NORMAL is not None
        assert WindowState.MINIMIZED is not None
        assert WindowState.MAXIMIZED is not None
        assert WindowState.HIDDEN is not None


# ============================================================================
# WindowEventType Tests
# ============================================================================

class TestWindowEventType:
    """Tests for WindowEventType enum."""

    def test_event_types_exist(self):
        """Verify all event types exist."""
        assert WindowEventType.CLOSE is not None
        assert WindowEventType.RESIZE is not None
        assert WindowEventType.MOVE is not None
        assert WindowEventType.FOCUS is not None
        assert WindowEventType.BLUR is not None
        assert WindowEventType.MINIMIZE is not None
        assert WindowEventType.MAXIMIZE is not None
        assert WindowEventType.RESTORE is not None


# ============================================================================
# Window Creation Tests
# ============================================================================

class TestWindowCreation:
    """Tests for window creation."""

    def test_create_default(self):
        """Test creating window with defaults."""
        window = Window.create(WindowConfig())
        assert window is not None
        assert window.is_open

    def test_create_custom(self):
        """Test creating window with custom config."""
        config = WindowConfig(
            title="Custom Window",
            width=1920,
            height=1080
        )
        window = Window.create(config)
        assert window.config.title == "Custom Window"
        assert window.config.width == 1920

    def test_unique_handles(self):
        """Test windows get unique handles."""
        windows = [Window.create(WindowConfig()) for _ in range(10)]
        handles = [w.native_handle() for w in windows]
        assert len(handles) == len(set(handles))

    def test_starts_hidden_unless_fullscreen(self):
        """Test window starts hidden unless fullscreen."""
        window = Window.create(WindowConfig())
        assert window.state == WindowState.HIDDEN

        fullscreen_window = Window.create(WindowConfig(fullscreen=True))
        assert fullscreen_window.state == WindowState.NORMAL


# ============================================================================
# Window Lifecycle Tests
# ============================================================================

class TestWindowLifecycle:
    """Tests for window lifecycle."""

    def test_show(self, window):
        """Test showing window."""
        window.show()
        assert window.state == WindowState.NORMAL

    def test_show_idempotent(self, window):
        """Test showing already shown window."""
        window.show()
        window.show()
        events = window.poll_events()
        # Should only have one restore event
        restore_events = [e for e in events if e.type == WindowEventType.RESTORE]
        assert len(restore_events) == 1

    def test_hide(self, window):
        """Test hiding window."""
        window.show()
        window.poll_events()
        window.hide()
        assert window.state == WindowState.HIDDEN

    def test_minimize(self, window):
        """Test minimizing window."""
        window.show()
        window.minimize()
        assert window.state == WindowState.MINIMIZED

    def test_maximize(self, window):
        """Test maximizing window."""
        window.show()
        window.maximize()
        assert window.state == WindowState.MAXIMIZED

    def test_restore_from_minimized(self, window):
        """Test restoring from minimized."""
        window.show()
        window.minimize()
        window.restore()
        assert window.state == WindowState.NORMAL

    def test_restore_from_maximized(self, window):
        """Test restoring from maximized."""
        window.show()
        window.maximize()
        window.restore()
        assert window.state == WindowState.NORMAL

    def test_close(self, window):
        """Test closing window."""
        assert window.is_open
        window.close()
        assert not window.is_open

    def test_close_generates_event(self, window):
        """Test close generates event."""
        window.close()
        events = window.poll_events()
        close_events = [e for e in events if e.type == WindowEventType.CLOSE]
        assert len(close_events) == 1


# ============================================================================
# Window Properties Tests
# ============================================================================

class TestWindowProperties:
    """Tests for window properties."""

    def test_set_title(self, window):
        """Test setting title."""
        window.set_title("New Title")
        assert window.config.title == "New Title"

    def test_set_size(self, window):
        """Test setting size."""
        window.set_size(1920, 1080)
        assert window.config.width == 1920
        assert window.config.height == 1080

    def test_set_size_generates_event(self, window):
        """Test set size generates resize event."""
        window.set_size(800, 600)
        events = window.poll_events()
        resize_events = [e for e in events if e.type == WindowEventType.RESIZE]
        assert len(resize_events) == 1
        assert resize_events[0].data["width"] == 800
        assert resize_events[0].data["height"] == 600

    def test_set_size_no_change_no_event(self, window):
        """Test no event when size unchanged."""
        original_width = window.config.width
        original_height = window.config.height
        window.poll_events()
        window.set_size(original_width, original_height)
        events = window.poll_events()
        resize_events = [e for e in events if e.type == WindowEventType.RESIZE]
        assert len(resize_events) == 0

    def test_set_position(self, window):
        """Test setting position."""
        window.set_position(200, 300)
        assert window.config.x == 200
        assert window.config.y == 300

    def test_set_position_generates_event(self, window):
        """Test set position generates move event."""
        window.set_position(500, 400)
        events = window.poll_events()
        move_events = [e for e in events if e.type == WindowEventType.MOVE]
        assert len(move_events) == 1
        assert move_events[0].data["x"] == 500
        assert move_events[0].data["y"] == 400

    def test_client_rect(self, window):
        """Test client rect."""
        rect = window.client_rect()
        assert rect.width == window.config.width
        assert rect.height == window.config.height

    def test_dpi_scale(self, window):
        """Test DPI scale."""
        scale = window.dpi_scale()
        assert scale == DEFAULT_DPI_SCALE

    def test_native_handle(self, window):
        """Test native handle."""
        handle = window.native_handle()
        assert isinstance(handle, int)
        assert handle > 0


# ============================================================================
# Window Fullscreen Tests
# ============================================================================

class TestWindowFullscreen:
    """Tests for fullscreen functionality."""

    def test_set_fullscreen_exclusive(self, window):
        """Test exclusive fullscreen."""
        window.set_fullscreen(True, FullscreenMode.EXCLUSIVE)
        assert window.config.fullscreen is True
        assert window.config.style == WindowStyle.FULLSCREEN_EXCLUSIVE

    def test_set_fullscreen_borderless(self, window):
        """Test borderless fullscreen."""
        window.set_fullscreen(True, FullscreenMode.BORDERLESS)
        assert window.config.fullscreen is True
        assert window.config.style == WindowStyle.BORDERLESS

    def test_exit_fullscreen(self, window):
        """Test exiting fullscreen."""
        window.set_fullscreen(True)
        window.set_fullscreen(False)
        assert window.config.fullscreen is False
        assert window.config.style == WindowStyle.WINDOWED

    def test_fullscreen_generates_resize_event(self, window):
        """Test fullscreen generates resize event."""
        window.set_fullscreen(True)
        events = window.poll_events()
        resize_events = [e for e in events if e.type == WindowEventType.RESIZE]
        assert len(resize_events) == 1
        assert resize_events[0].data.get("fullscreen") is True


# ============================================================================
# Window Event Tests
# ============================================================================

class TestWindowEvents:
    """Tests for window event system."""

    def test_poll_clears_queue(self, window):
        """Test polling clears event queue."""
        window.show()
        events1 = window.poll_events()
        assert len(events1) > 0
        events2 = window.poll_events()
        assert len(events2) == 0

    def test_event_data_types(self, window):
        """Test event data types are correct."""
        window.set_size(1024, 768)
        events = window.poll_events()
        resize_event = next(e for e in events if e.type == WindowEventType.RESIZE)
        assert isinstance(resize_event.data["width"], int)
        assert isinstance(resize_event.data["height"], int)


# ============================================================================
# Display Tests
# ============================================================================

class TestDisplay:
    """Tests for Display class."""

    def test_enumerate_returns_displays(self, display):
        """Test display enumeration."""
        displays = Display.enumerate()
        assert len(displays) >= 1

    def test_primary_display(self, display):
        """Test getting primary display."""
        primary = Display.primary()
        assert primary is not None
        assert primary.is_primary

    def test_display_name(self, display):
        """Test display name."""
        assert isinstance(display.name, str)
        assert len(display.name) > 0

    def test_display_bounds(self, display):
        """Test display bounds."""
        bounds = display.bounds
        assert isinstance(bounds, Rect)
        assert bounds.width > 0
        assert bounds.height > 0

    def test_display_work_area(self, display):
        """Test display work area."""
        work_area = display.work_area
        assert isinstance(work_area, Rect)
        # Work area should be same or smaller than bounds
        assert work_area.width <= display.bounds.width
        assert work_area.height <= display.bounds.height

    def test_display_dpi_scale(self, display):
        """Test display DPI scale."""
        scale = display.dpi_scale
        assert scale > 0

    def test_display_supported_modes(self, display):
        """Test supported display modes."""
        modes = display.supported_modes()
        assert len(modes) > 0

    def test_display_current_mode(self, display):
        """Test current display mode."""
        mode = display.current_mode()
        assert isinstance(mode, DisplayMode)
        assert mode.width > 0
        assert mode.height > 0
        assert mode.refresh_rate > 0


# ============================================================================
# DisplayMode Tests
# ============================================================================

class TestDisplayMode:
    """Tests for DisplayMode dataclass."""

    def test_creation(self):
        """Test creating display mode."""
        mode = DisplayMode(1920, 1080, 60)
        assert mode.width == 1920
        assert mode.height == 1080
        assert mode.refresh_rate == 60
        assert mode.format == "RGBA8888"

    def test_custom_format(self):
        """Test custom format."""
        mode = DisplayMode(1920, 1080, 60, "HDR10")
        assert mode.format == "HDR10"


# ============================================================================
# DisplayInfo Tests
# ============================================================================

class TestDisplayInfo:
    """Tests for DisplayInfo dataclass."""

    def test_creation(self):
        """Test creating display info."""
        info = DisplayInfo(
            name="Test Display",
            bounds=Rect(0, 0, 1920, 1080),
            work_area=Rect(0, 0, 1920, 1040),
            dpi_scale=1.0,
            is_primary=True
        )
        assert info.name == "Test Display"
        assert info.is_primary is True


# ============================================================================
# CursorType Tests
# ============================================================================

class TestCursorType:
    """Tests for CursorType enum."""

    def test_cursor_types_exist(self):
        """Verify cursor types exist."""
        assert CursorType.ARROW is not None
        assert CursorType.IBEAM is not None
        assert CursorType.HAND is not None
        assert CursorType.CROSSHAIR is not None
        assert CursorType.RESIZE_NS is not None
        assert CursorType.RESIZE_EW is not None
        assert CursorType.RESIZE_NWSE is not None
        assert CursorType.RESIZE_NESW is not None
        assert CursorType.WAIT is not None
        assert CursorType.NOT_ALLOWED is not None
        assert CursorType.RESIZE_ALL is not None
        assert CursorType.WAIT_ARROW is not None


# ============================================================================
# CursorManager Tests
# ============================================================================

class TestCursorManager:
    """Tests for CursorManager class."""

    def test_initial_cursor(self, cursor_manager):
        """Test initial cursor type."""
        assert cursor_manager.current_type == CursorType.ARROW

    def test_initial_visible(self, cursor_manager):
        """Test cursor is initially visible."""
        assert cursor_manager.visible is True

    def test_initial_not_confined(self, cursor_manager):
        """Test cursor is not initially confined."""
        assert cursor_manager.confined is False

    def test_set_cursor(self, cursor_manager):
        """Test setting cursor type."""
        cursor_manager.set_cursor(CursorType.HAND)
        assert cursor_manager.current_type == CursorType.HAND

    def test_set_visibility(self, cursor_manager):
        """Test setting cursor visibility."""
        cursor_manager.set_visible(False)
        assert cursor_manager.visible is False
        cursor_manager.set_visible(True)
        assert cursor_manager.visible is True

    def test_confine_cursor(self, cursor_manager):
        """Test confining cursor."""
        cursor_manager.confine(True)
        assert cursor_manager.confined is True

    def test_release_cursor(self, cursor_manager):
        """Test releasing confined cursor."""
        cursor_manager.confine(True)
        cursor_manager.confine(False)
        assert cursor_manager.confined is False

    def test_set_custom_cursor(self, cursor_manager):
        """Test setting custom cursor."""
        image_data = b"\x00\x00\x00\x00" * 16  # 16 pixels
        cursor_manager.set_custom_cursor(image_data, 0, 0)
        assert cursor_manager.has_custom_cursor is True

    def test_custom_cursor_clears_on_set_cursor(self, cursor_manager):
        """Test custom cursor cleared when setting standard cursor."""
        cursor_manager.set_custom_cursor(b"\x00", 0, 0)
        cursor_manager.set_cursor(CursorType.CROSSHAIR)
        assert cursor_manager.has_custom_cursor is False


# ============================================================================
# ColorSpace Tests
# ============================================================================

class TestColorSpace:
    """Tests for ColorSpace enum."""

    def test_color_spaces_exist(self):
        """Verify color spaces exist."""
        assert ColorSpace.SRGB is not None
        assert ColorSpace.SCRGB is not None
        assert ColorSpace.HDR10 is not None
        assert ColorSpace.DOLBY_VISION is not None


# ============================================================================
# HDRCapabilities Tests
# ============================================================================

class TestHDRCapabilities:
    """Tests for HDRCapabilities dataclass."""

    def test_creation(self):
        """Test creating HDR capabilities."""
        caps = HDRCapabilities(
            supported=True,
            min_luminance=0.001,
            max_luminance=1000.0,
            max_full_frame_luminance=400.0,
            color_space=ColorSpace.HDR10
        )
        assert caps.supported is True
        assert caps.max_luminance == 1000.0
        assert caps.color_space == ColorSpace.HDR10


# ============================================================================
# DisplayHDR Tests
# ============================================================================

class TestDisplayHDR:
    """Tests for DisplayHDR class."""

    def test_default_not_supported(self, display_hdr):
        """Test HDR not supported by default."""
        assert display_hdr.is_supported() is False

    def test_simulated_hdr_supported(self):
        """Test simulated HDR is supported."""
        hdr = DisplayHDR(simulate_hdr=True)
        assert hdr.is_supported() is True

    def test_get_capabilities(self, display_hdr):
        """Test getting HDR capabilities."""
        caps = display_hdr.get_capabilities()
        assert isinstance(caps, HDRCapabilities)

    def test_simulated_capabilities(self):
        """Test simulated HDR capabilities."""
        hdr = DisplayHDR(simulate_hdr=True)
        caps = hdr.get_capabilities()
        assert caps.supported is True
        assert caps.color_space == ColorSpace.HDR10

    def test_set_color_space_srgb(self, display_hdr):
        """Test setting sRGB color space always succeeds."""
        result = display_hdr.set_color_space(ColorSpace.SRGB)
        assert result is True

    def test_set_color_space_hdr_unsupported(self, display_hdr):
        """Test setting HDR color space fails when unsupported."""
        result = display_hdr.set_color_space(ColorSpace.HDR10)
        assert result is False

    def test_set_color_space_hdr_supported(self):
        """Test setting HDR color space when supported."""
        hdr = DisplayHDR(simulate_hdr=True)
        result = hdr.set_color_space(ColorSpace.HDR10)
        assert result is True

    def test_current_color_space(self, display_hdr):
        """Test getting current color space."""
        space = display_hdr.current_color_space
        assert isinstance(space, ColorSpace)

    def test_set_metadata(self):
        """Test setting HDR metadata."""
        hdr = DisplayHDR(simulate_hdr=True)
        hdr.set_metadata(
            max_content_light_level=1000.0,
            max_frame_average_light_level=400.0
        )
        metadata = hdr.metadata
        assert metadata["max_cll"] == 1000.0
        assert metadata["max_fall"] == 400.0

    def test_metadata_empty_by_default(self, display_hdr):
        """Test metadata is empty by default."""
        assert len(display_hdr.metadata) == 0


# ============================================================================
# VRRType Tests
# ============================================================================

class TestVRRType:
    """Tests for VRRType enum."""

    def test_vrr_types_exist(self):
        """Verify VRR types exist."""
        assert VRRType.NONE is not None
        assert VRRType.FREESYNC is not None
        assert VRRType.GSYNC is not None
        assert VRRType.ADAPTIVE_SYNC is not None


# ============================================================================
# RefreshRange Tests
# ============================================================================

class TestRefreshRange:
    """Tests for RefreshRange dataclass."""

    def test_creation(self):
        """Test creating refresh range."""
        range_ = RefreshRange(min_hz=48, max_hz=144)
        assert range_.min_hz == 48
        assert range_.max_hz == 144


# ============================================================================
# VariableRefresh Tests
# ============================================================================

class TestVariableRefresh:
    """Tests for VariableRefresh class."""

    def test_default_not_supported(self, variable_refresh):
        """Test VRR not supported by default."""
        assert variable_refresh.is_supported() is False
        assert variable_refresh.supported is False

    def test_simulated_vrr_supported(self):
        """Test simulated VRR is supported."""
        vrr = VariableRefresh(simulate_vrr=True)
        assert vrr.is_supported() is True

    def test_vrr_type(self, variable_refresh):
        """Test getting VRR type."""
        vrr_type = variable_refresh.vrr_type
        assert isinstance(vrr_type, VRRType)

    def test_default_vrr_type_none(self, variable_refresh):
        """Test default VRR type is NONE."""
        assert variable_refresh.vrr_type == VRRType.NONE

    def test_simulated_vrr_type(self):
        """Test simulated VRR type."""
        vrr = VariableRefresh(simulate_vrr=True)
        assert vrr.vrr_type == VRRType.FREESYNC

    def test_get_range(self, variable_refresh):
        """Test getting refresh range."""
        range_ = variable_refresh.get_range()
        assert isinstance(range_, RefreshRange)

    def test_enable_unsupported(self, variable_refresh):
        """Test enabling VRR when unsupported."""
        result = variable_refresh.enable(True)
        assert result is False

    def test_enable_supported(self):
        """Test enabling VRR when supported."""
        vrr = VariableRefresh(simulate_vrr=True)
        result = vrr.enable(True)
        assert result is True
        assert vrr.enabled is True

    def test_disable(self):
        """Test disabling VRR."""
        vrr = VariableRefresh(simulate_vrr=True)
        vrr.enable(True)
        result = vrr.enable(False)
        assert result is True
        assert vrr.enabled is False

    def test_enabled_property(self, variable_refresh):
        """Test enabled property."""
        assert variable_refresh.enabled is False


# ============================================================================
# Thread Safety Tests
# ============================================================================

class TestWindowThreadSafety:
    """Tests for window subsystem thread safety."""

    def test_concurrent_window_creation(self):
        """Test concurrent window creation."""
        windows = []
        lock = threading.Lock()
        errors = []

        def create_window():
            try:
                w = Window.create(WindowConfig())
                with lock:
                    windows.append(w)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_window) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(windows) == 10

    def test_concurrent_window_operations(self, window):
        """Test concurrent window operations."""
        errors = []

        def operate_window():
            try:
                for _ in range(50):
                    window.set_size(800, 600)
                    window.set_position(100, 100)
                    window.poll_events()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=operate_window) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestWindowEdgeCases:
    """Tests for window edge cases."""

    def test_zero_size_window(self):
        """Test creating zero-sized window."""
        config = WindowConfig(width=0, height=0)
        window = Window.create(config)
        assert window.config.width == 0
        assert window.config.height == 0

    def test_very_large_window(self):
        """Test creating very large window."""
        config = WindowConfig(width=10000, height=10000)
        window = Window.create(config)
        assert window.config.width == 10000

    def test_negative_position(self):
        """Test window with negative position."""
        config = WindowConfig(x=-100, y=-100)
        window = Window.create(config)
        assert window.config.x == -100

    def test_empty_title(self):
        """Test window with empty title."""
        config = WindowConfig(title="")
        window = Window.create(config)
        assert window.config.title == ""

    def test_unicode_title(self):
        """Test window with unicode title."""
        config = WindowConfig(title="Test Window unicode")
        window = Window.create(config)
        assert "unicode" in window.config.title

    def test_rapid_state_changes(self, window):
        """Test rapid state changes."""
        for _ in range(100):
            window.show()
            window.minimize()
            window.restore()
            window.maximize()
            window.restore()
            window.hide()
        # Should not raise


# ============================================================================
# Performance Tests
# ============================================================================

class TestWindowPerformance:
    """Performance tests for window subsystem."""

    def test_window_creation_performance(self):
        """Test window creation performance."""
        num_windows = 100

        start = time.time()
        windows = [Window.create(WindowConfig()) for _ in range(num_windows)]
        elapsed = time.time() - start

        assert len(windows) == num_windows
        assert elapsed < 1.0, f"Creation too slow: {elapsed:.2f}s"

    def test_event_polling_performance(self, window):
        """Test event polling performance."""
        # Generate many events
        for _ in range(100):
            window.set_size(800, 600)
            window.set_position(100, 100)

        num_polls = 10000
        start = time.time()
        for _ in range(num_polls):
            window.poll_events()
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Polling too slow: {elapsed:.2f}s"


# ============================================================================
# Multiple Window Tests
# ============================================================================

class TestMultipleWindows:
    """Tests for multiple window scenarios."""

    def test_multiple_windows_independent(self):
        """Test multiple windows are independent."""
        w1 = Window.create(WindowConfig(title="Window 1"))
        w2 = Window.create(WindowConfig(title="Window 2"))

        w1.set_size(800, 600)
        w2.set_size(1024, 768)

        assert w1.config.width == 800
        assert w2.config.width == 1024

    def test_close_one_window(self):
        """Test closing one window doesn't affect others."""
        w1 = Window.create(WindowConfig())
        w2 = Window.create(WindowConfig())

        w1.close()

        assert not w1.is_open
        assert w2.is_open

    def test_many_windows(self):
        """Test creating many windows."""
        windows = [Window.create(WindowConfig()) for _ in range(50)]
        assert len(windows) == 50

        # All should be open
        assert all(w.is_open for w in windows)

        # Close all
        for w in windows:
            w.close()

        assert all(not w.is_open for w in windows)


# ============================================================================
# FullscreenMode Tests
# ============================================================================

class TestFullscreenMode:
    """Tests for FullscreenMode enum."""

    def test_modes_exist(self):
        """Verify fullscreen modes exist."""
        assert FullscreenMode.WINDOWED is not None
        assert FullscreenMode.BORDERLESS is not None
        assert FullscreenMode.EXCLUSIVE is not None
