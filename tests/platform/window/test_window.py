"""Tests for window management."""

import pytest

from engine.platform.window import (
    Window,
    WindowConfig,
    WindowStyle,
    WindowState,
    WindowEvent,
    WindowEventType,
    FullscreenMode,
    Rect,
)


class TestWindowConfig:
    """Tests for WindowConfig dataclass."""

    def test_default_config(self):
        """Test default window configuration."""
        config = WindowConfig()
        assert config.title == "Game Window"
        assert config.x == 100
        assert config.y == 100
        assert config.width == 1280
        assert config.height == 720
        assert config.resizable is True
        assert config.fullscreen is False
        assert config.style == WindowStyle.WINDOWED

    def test_custom_config(self):
        """Test custom window configuration."""
        config = WindowConfig(
            title="Custom Window",
            x=200,
            y=200,
            width=1920,
            height=1080,
            resizable=False,
            fullscreen=True,
            style=WindowStyle.FULLSCREEN_EXCLUSIVE
        )
        assert config.title == "Custom Window"
        assert config.x == 200
        assert config.y == 200
        assert config.width == 1920
        assert config.height == 1080
        assert config.resizable is False
        assert config.fullscreen is True
        assert config.style == WindowStyle.FULLSCREEN_EXCLUSIVE


class TestWindowCreation:
    """Tests for window creation."""

    def test_create_window(self):
        """Test basic window creation."""
        config = WindowConfig(title="Test Window", width=800, height=600, x=50, y=50)
        window = Window.create(config)
        assert window is not None
        assert window.is_open
        assert window.state == WindowState.HIDDEN
        # Verify config values propagated correctly
        assert window.config.title == "Test Window"
        assert window.config.width == 800
        assert window.config.height == 600
        assert window.config.x == 50
        assert window.config.y == 50

    def test_unique_handles(self):
        """Test that windows get unique handles."""
        window1 = Window.create(WindowConfig())
        window2 = Window.create(WindowConfig())
        assert window1.native_handle() != window2.native_handle()

    def test_window_starts_hidden(self):
        """Test that windows start in hidden state."""
        window = Window.create(WindowConfig())
        assert window.state == WindowState.HIDDEN


class TestWindowLifecycle:
    """Tests for window lifecycle operations."""

    def test_show_window(self):
        """Test showing a window."""
        window = Window.create(WindowConfig())
        window.show()
        assert window.state == WindowState.NORMAL
        events = window.poll_events()
        assert len(events) == 1
        assert events[0].type == WindowEventType.RESTORE

    def test_hide_window(self):
        """Test hiding a window."""
        window = Window.create(WindowConfig())
        window.show()
        window.poll_events()  # Clear show event
        window.hide()
        assert window.state == WindowState.HIDDEN
        events = window.poll_events()
        assert any(e.type == WindowEventType.BLUR for e in events)

    def test_minimize_window(self):
        """Test minimizing a window."""
        window = Window.create(WindowConfig())
        window.show()
        window.poll_events()
        window.minimize()
        assert window.state == WindowState.MINIMIZED
        events = window.poll_events()
        assert any(e.type == WindowEventType.MINIMIZE for e in events)

    def test_maximize_window(self):
        """Test maximizing a window."""
        window = Window.create(WindowConfig())
        window.show()
        window.poll_events()
        window.maximize()
        assert window.state == WindowState.MAXIMIZED
        events = window.poll_events()
        assert any(e.type == WindowEventType.MAXIMIZE for e in events)

    def test_restore_window(self):
        """Test restoring a window."""
        window = Window.create(WindowConfig())
        window.show()
        window.minimize()
        window.poll_events()
        window.restore()
        assert window.state == WindowState.NORMAL
        events = window.poll_events()
        assert any(e.type == WindowEventType.RESTORE for e in events)

    def test_close_window(self):
        """Test closing a window."""
        window = Window.create(WindowConfig())
        assert window.is_open
        window.close()
        assert not window.is_open
        events = window.poll_events()
        assert any(e.type == WindowEventType.CLOSE for e in events)


class TestWindowProperties:
    """Tests for window property operations."""

    def test_set_title(self):
        """Test setting window title."""
        window = Window.create(WindowConfig())
        window.set_title("New Title")
        assert window.config.title == "New Title"

    def test_set_size(self):
        """Test setting window size."""
        window = Window.create(WindowConfig())
        window.set_size(1920, 1080)
        assert window.config.width == 1920
        assert window.config.height == 1080
        events = window.poll_events()
        assert any(e.type == WindowEventType.RESIZE for e in events)

    def test_set_position(self):
        """Test setting window position."""
        window = Window.create(WindowConfig())
        window.set_position(200, 300)
        assert window.config.x == 200
        assert window.config.y == 300
        events = window.poll_events()
        assert any(e.type == WindowEventType.MOVE for e in events)

    def test_client_rect(self):
        """Test getting client rectangle."""
        config = WindowConfig(x=100, y=100, width=800, height=600)
        window = Window.create(config)
        rect = window.client_rect()
        assert rect.x == 100
        assert rect.y == 100
        assert rect.width == 800
        assert rect.height == 600

    def test_dpi_scale(self):
        """Test getting DPI scale."""
        window = Window.create(WindowConfig())
        scale = window.dpi_scale()
        assert scale == 1.0

    def test_native_handle(self):
        """Test getting native handle."""
        window = Window.create(WindowConfig())
        handle = window.native_handle()
        assert isinstance(handle, int)
        assert handle > 0


class TestWindowFullscreen:
    """Tests for fullscreen operations."""

    def test_set_fullscreen_exclusive(self):
        """Test setting exclusive fullscreen."""
        window = Window.create(WindowConfig())
        window.set_fullscreen(True, FullscreenMode.EXCLUSIVE)
        assert window.config.fullscreen is True
        assert window.config.style == WindowStyle.FULLSCREEN_EXCLUSIVE
        events = window.poll_events()
        resize_events = [e for e in events if e.type == WindowEventType.RESIZE]
        assert len(resize_events) > 0
        assert resize_events[0].data.get("fullscreen") is True

    def test_set_fullscreen_borderless(self):
        """Test setting borderless fullscreen."""
        window = Window.create(WindowConfig())
        window.set_fullscreen(True, FullscreenMode.BORDERLESS)
        assert window.config.fullscreen is True
        assert window.config.style == WindowStyle.BORDERLESS

    def test_exit_fullscreen(self):
        """Test exiting fullscreen."""
        window = Window.create(WindowConfig())
        window.set_fullscreen(True, FullscreenMode.EXCLUSIVE)
        window.poll_events()
        window.set_fullscreen(False)
        assert window.config.fullscreen is False
        assert window.config.style == WindowStyle.WINDOWED
        events = window.poll_events()
        resize_events = [e for e in events if e.type == WindowEventType.RESIZE]
        assert len(resize_events) > 0
        assert resize_events[0].data.get("fullscreen") is False


class TestWindowEvents:
    """Tests for window event system."""

    def test_poll_events_clears_queue(self):
        """Test that polling events clears the queue."""
        window = Window.create(WindowConfig())
        window.show()
        events1 = window.poll_events()
        assert len(events1) > 0
        events2 = window.poll_events()
        assert len(events2) == 0

    def test_resize_event_data(self):
        """Test resize event contains correct data."""
        window = Window.create(WindowConfig())
        window.set_size(1920, 1080)
        events = window.poll_events()
        resize_events = [e for e in events if e.type == WindowEventType.RESIZE]
        assert len(resize_events) > 0
        assert resize_events[0].data["width"] == 1920
        assert resize_events[0].data["height"] == 1080

    def test_move_event_data(self):
        """Test move event contains correct data."""
        window = Window.create(WindowConfig())
        window.set_position(500, 600)
        events = window.poll_events()
        move_events = [e for e in events if e.type == WindowEventType.MOVE]
        assert len(move_events) > 0
        assert move_events[0].data["x"] == 500
        assert move_events[0].data["y"] == 600

    def test_no_duplicate_state_changes(self):
        """Test that duplicate state changes don't create duplicate events."""
        window = Window.create(WindowConfig())
        window.show()
        window.poll_events()
        window.show()  # Show again
        events = window.poll_events()
        # Should not generate another event
        assert len(events) == 0


class TestRect:
    """Tests for Rect dataclass."""

    def test_rect_creation(self):
        """Test creating a rectangle."""
        rect = Rect(10, 20, 800, 600)
        assert rect.x == 10
        assert rect.y == 20
        assert rect.width == 800
        assert rect.height == 600
