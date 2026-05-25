"""
Comprehensive tests for the Viewport system.

Tests cover:
- Camera positioning and rotation
- Camera control modes (orbit, fly, pan)
- Render modes switching
- Viewport overlays
- Grid settings
- Input handling
- Screen/world coordinate conversion
"""
import pytest
import sys
import math

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.editor.viewport import (
    Viewport,
    Camera,
    CameraMode,
    RenderMode,
    ViewportOverlay,
    ViewportOverlayFlags,
    GridSettings,
    ViewportInput,
)


class TestCamera:
    """Tests for Camera class."""

    def test_camera_creation_defaults(self):
        """Camera should be created with default values."""
        camera = Camera()
        assert camera.fov == 60.0
        assert camera.near_clip == 0.1
        assert camera.far_clip == 10000.0
        assert camera.orthographic is False

    def test_camera_creation_custom(self):
        """Camera can be created with custom values."""
        camera = Camera(position=(10, 20, 30), rotation=(45, 90, 0), fov=75.0)
        assert camera.position == (10, 20, 30)
        assert camera.rotation == (45, 90, 0)
        assert camera.fov == 75.0

    def test_camera_set_position(self):
        """Camera position can be set."""
        camera = Camera()
        camera.set_position(5, 10, 15)
        assert camera.position == (5, 10, 15)

    def test_camera_set_rotation(self):
        """Camera rotation can be set."""
        camera = Camera()
        camera.set_rotation(30, 45, 10)
        assert camera.rotation == (30, 45, 10)

    def test_camera_look_at(self):
        """Camera can look at a target."""
        camera = Camera(position=(0, 0, 10))
        camera.look_at((0, 0, 0))

        assert camera.focus_point == (0, 0, 0)
        assert camera.focus_distance == pytest.approx(10.0, rel=0.1)

    def test_camera_orbit(self):
        """Camera can orbit around focus point."""
        camera = Camera(position=(0, 0, 10))
        camera.focus_point = (0, 0, 0)
        camera.focus_distance = 10.0
        camera.rotation = (0, 0, 0)

        # Orbit yaw
        camera.orbit(0, 45)
        assert camera.rotation[1] == pytest.approx(45.0)

    def test_camera_orbit_pitch_clamped(self):
        """Camera pitch should be clamped to avoid gimbal lock."""
        camera = Camera()
        camera.rotation = (80, 0, 0)

        camera.orbit(20, 0)  # Try to exceed 90 degrees
        assert camera.rotation[0] <= 89.0

    def test_camera_pan(self):
        """Camera can pan."""
        camera = Camera(position=(0, 0, 10))
        camera.focus_point = (0, 0, 0)
        camera.rotation = (0, 0, 0)

        original_pos = camera.position
        camera.pan(5, 2)

        # Position should have changed
        assert camera.position != original_pos

    def test_camera_zoom(self):
        """Camera can zoom."""
        camera = Camera(position=(0, 0, 10))
        camera.focus_distance = 10.0
        camera.focus_point = (0, 0, 0)
        camera.rotation = (0, 0, 0)

        original_distance = camera.focus_distance
        camera.zoom(1.0)  # Zoom in

        assert camera.focus_distance < original_distance

    def test_camera_zoom_minimum(self):
        """Camera zoom has minimum distance."""
        camera = Camera()
        camera.focus_distance = 0.5
        camera.rotation = (0, 0, 0)
        camera.focus_point = (0, 0, 0)

        camera.zoom(100)  # Try to zoom way in
        assert camera.focus_distance >= 0.1

    def test_camera_fly_forward(self):
        """Camera can fly forward."""
        camera = Camera(position=(0, 0, 0))
        camera.rotation = (0, 0, 0)  # Looking down -Z

        camera.fly_forward(10)

        # Should have moved forward (in -Z direction based on rotation)
        assert camera.position[2] != 0

    def test_camera_fly_strafe(self):
        """Camera can strafe."""
        camera = Camera(position=(0, 0, 0))
        camera.rotation = (0, 0, 0)

        camera.fly_strafe(10)

        # Should have moved on X axis
        assert camera.position[0] == pytest.approx(10.0)

    def test_camera_fly_up(self):
        """Camera can fly up/down."""
        camera = Camera(position=(0, 0, 0))

        camera.fly_up(5)
        assert camera.position[1] == 5

        camera.fly_up(-3)
        assert camera.position[1] == 2

    def test_camera_frame_bounds(self):
        """Camera can frame bounds."""
        camera = Camera()
        min_b = (-5, -5, -5)
        max_b = (5, 5, 5)

        camera.frame_bounds(min_b, max_b)

        assert camera.focus_point == (0, 0, 0)
        assert camera.focus_distance > 0

    def test_camera_orthographic_mode(self):
        """Camera can switch to orthographic."""
        camera = Camera()

        camera.set_orthographic(True, 20.0)
        assert camera.orthographic is True
        assert camera.ortho_size == 20.0

        camera.set_orthographic(False)
        assert camera.orthographic is False


class TestGridSettings:
    """Tests for GridSettings class."""

    def test_grid_creation(self):
        """GridSettings should have default values."""
        grid = GridSettings()
        assert grid.enabled is True
        assert grid.size == 100.0
        assert grid.divisions == 10

    def test_grid_snap_settings(self):
        """Grid snap settings can be configured."""
        grid = GridSettings()
        grid.snap_enabled = True
        grid.snap_size = 2.5

        assert grid.snap_enabled is True
        assert grid.snap_size == 2.5


class TestViewportOverlay:
    """Tests for ViewportOverlay class."""

    def test_overlay_creation(self):
        """ViewportOverlay should have default flags."""
        overlay = ViewportOverlay()
        assert ViewportOverlayFlags.GRID in overlay.flags
        assert ViewportOverlayFlags.GIZMOS in overlay.flags

    def test_overlay_toggle_flag(self):
        """Overlay flags can be toggled."""
        overlay = ViewportOverlay()

        overlay.toggle_flag(ViewportOverlayFlags.GRID)
        assert ViewportOverlayFlags.GRID not in overlay.flags

        overlay.toggle_flag(ViewportOverlayFlags.GRID)
        assert ViewportOverlayFlags.GRID in overlay.flags

    def test_overlay_has_flag(self):
        """Can check for overlay flags."""
        overlay = ViewportOverlay()

        assert overlay.has_flag(ViewportOverlayFlags.GRID) is True
        assert overlay.has_flag(ViewportOverlayFlags.NAMES) is False


class TestViewportInput:
    """Tests for ViewportInput class."""

    def test_input_creation(self):
        """ViewportInput should be created with defaults."""
        viewport = Viewport(id="main")
        input_handler = ViewportInput(viewport)

        assert input_handler.orbit_sensitivity == 0.5
        assert input_handler.fly_speed == 10.0
        assert input_handler.invert_y is False

    def test_input_mouse_orbit(self):
        """Middle mouse should start orbit."""
        viewport = Viewport(id="main")
        input_handler = viewport.input

        # Middle button down
        assert input_handler.on_mouse_down(100, 100, 1) is True
        assert input_handler._is_orbiting is True

        # Middle button up
        assert input_handler.on_mouse_up(100, 100, 1) is True
        assert input_handler._is_orbiting is False

    def test_input_mouse_pan(self):
        """Right mouse should start pan."""
        viewport = Viewport(id="main")
        input_handler = viewport.input

        # Right button down
        assert input_handler.on_mouse_down(100, 100, 2) is True
        assert input_handler._is_panning is True

        # Right button up
        assert input_handler.on_mouse_up(100, 100, 2) is True
        assert input_handler._is_panning is False

    def test_input_mouse_wheel_zoom(self):
        """Mouse wheel should zoom."""
        viewport = Viewport(id="main")
        input_handler = viewport.input

        original_distance = viewport.camera.focus_distance
        input_handler.on_mouse_wheel(1.0)

        assert viewport.camera.focus_distance != original_distance

    def test_input_fly_mode_keys(self):
        """WASD keys work in fly mode."""
        viewport = Viewport(id="main")
        viewport.camera_mode = CameraMode.FLY
        input_handler = viewport.input

        original_pos = viewport.camera.position
        input_handler.on_key_down("w")

        assert viewport.camera.position != original_pos

    def test_input_fly_mode_key_not_fly_mode(self):
        """WASD keys don't work in orbit mode."""
        viewport = Viewport(id="main")
        viewport.camera_mode = CameraMode.ORBIT
        input_handler = viewport.input

        original_pos = viewport.camera.position
        result = input_handler.on_key_down("w")

        assert result is False
        assert viewport.camera.position == original_pos


class TestViewport:
    """Tests for Viewport class."""

    def test_viewport_creation(self):
        """Viewport should be created with defaults."""
        viewport = Viewport(id="main", name="Main Viewport")
        assert viewport.id == "main"
        assert viewport.name == "Main Viewport"
        assert viewport.width == 800
        assert viewport.height == 600
        assert viewport.active is True

    def test_viewport_aspect_ratio(self):
        """Viewport aspect ratio calculation."""
        viewport = Viewport(id="main", width=1920, height=1080)
        assert viewport.aspect_ratio == pytest.approx(1920 / 1080)

    def test_viewport_set_size(self):
        """Viewport size can be changed."""
        viewport = Viewport(id="main")
        resized = []
        viewport.on_resize = lambda w, h: resized.append((w, h))

        viewport.set_size(1920, 1080)
        assert viewport.width == 1920
        assert viewport.height == 1080
        assert len(resized) == 1

    def test_viewport_set_size_minimum(self):
        """Viewport size has minimum."""
        viewport = Viewport(id="main")
        viewport.set_size(0, 0)

        assert viewport.width >= 1
        assert viewport.height >= 1

    def test_viewport_camera_mode(self):
        """Camera mode can be changed."""
        viewport = Viewport(id="main")
        assert viewport.camera_mode == CameraMode.ORBIT

        viewport.set_camera_mode(CameraMode.FLY)
        assert viewport.camera_mode == CameraMode.FLY

    def test_viewport_render_mode(self):
        """Render mode can be changed."""
        viewport = Viewport(id="main")
        assert viewport.render_mode == RenderMode.LIT

        viewport.set_render_mode(RenderMode.WIREFRAME)
        assert viewport.render_mode == RenderMode.WIREFRAME

    def test_viewport_cycle_render_mode(self):
        """Render mode can be cycled."""
        viewport = Viewport(id="main")
        viewport.render_mode = RenderMode.LIT

        next_mode = viewport.cycle_render_mode()
        assert next_mode == RenderMode.UNLIT
        assert viewport.render_mode == RenderMode.UNLIT

    def test_viewport_render_callback(self):
        """Render callback is called."""
        viewport = Viewport(id="main")
        rendered = []
        viewport.on_render = lambda: rendered.append(True)

        viewport.render()
        assert len(rendered) == 1

    def test_viewport_render_inactive(self):
        """Inactive viewport doesn't render."""
        viewport = Viewport(id="main")
        viewport.active = False
        rendered = []
        viewport.on_render = lambda: rendered.append(True)

        viewport.render()
        assert len(rendered) == 0

    def test_viewport_screen_to_world(self):
        """Screen coordinates can be converted to world."""
        viewport = Viewport(id="main", width=800, height=600)
        viewport.camera.position = (0, 0, 10)
        viewport.camera.rotation = (0, 0, 0)

        world_pos = viewport.screen_to_world(400, 300, 10.0)

        # Center of screen at depth 10 should be roughly at origin
        assert isinstance(world_pos, tuple)
        assert len(world_pos) == 3

    def test_viewport_world_to_screen(self):
        """World coordinates can be converted to screen."""
        viewport = Viewport(id="main", width=800, height=600)
        viewport.camera.position = (0, 0, 10)
        viewport.camera.rotation = (0, 0, 0)

        # Point in front of camera
        screen_pos = viewport.world_to_screen(0, 0, 0)

        assert screen_pos is not None
        # Should be roughly center of screen
        assert 200 < screen_pos[0] < 600
        assert 100 < screen_pos[1] < 500

    def test_viewport_world_to_screen_behind_camera(self):
        """Points behind camera return None."""
        viewport = Viewport(id="main")
        viewport.camera.position = (0, 0, 10)
        viewport.camera.rotation = (0, 0, 0)

        # Point behind camera
        screen_pos = viewport.world_to_screen(0, 0, 20)

        assert screen_pos is None

    def test_viewport_save_load_state(self):
        """Viewport state can be saved and loaded."""
        viewport = Viewport(id="main")
        viewport.camera.position = (10, 20, 30)
        viewport.camera.rotation = (15, 45, 0)
        viewport.camera.fov = 75.0
        viewport.camera_mode = CameraMode.FLY
        viewport.render_mode = RenderMode.WIREFRAME

        # Save state
        state = viewport.save_state()

        # Modify viewport
        viewport.camera.position = (0, 0, 0)
        viewport.render_mode = RenderMode.LIT

        # Load state
        viewport.load_state(state)

        assert viewport.camera.position == (10, 20, 30)
        assert viewport.camera_mode == CameraMode.FLY
        assert viewport.render_mode == RenderMode.WIREFRAME


class TestRenderModes:
    """Tests for render mode enumeration."""

    def test_all_render_modes_exist(self):
        """All required render modes exist."""
        modes = [
            RenderMode.LIT,
            RenderMode.UNLIT,
            RenderMode.WIREFRAME,
            RenderMode.NORMALS,
            RenderMode.OVERDRAW,
            RenderMode.LOD_COLORING,
            RenderMode.COLLISION,
            RenderMode.NAVMESH,
        ]

        for mode in modes:
            assert isinstance(mode, RenderMode)

    def test_render_mode_values_unique(self):
        """All render mode values are unique."""
        values = [mode.value for mode in RenderMode]
        assert len(values) == len(set(values))


class TestCameraModes:
    """Tests for camera mode enumeration."""

    def test_all_camera_modes_exist(self):
        """All required camera modes exist."""
        modes = [
            CameraMode.ORBIT,
            CameraMode.FLY,
            CameraMode.PAN,
            CameraMode.ZOOM,
        ]

        for mode in modes:
            assert isinstance(mode, CameraMode)


class TestViewportOverlayFlags:
    """Tests for overlay flag enumeration."""

    def test_overlay_flags_combinable(self):
        """Overlay flags can be combined."""
        combined = ViewportOverlayFlags.GRID | ViewportOverlayFlags.GIZMOS

        assert ViewportOverlayFlags.GRID in combined
        assert ViewportOverlayFlags.GIZMOS in combined
        assert ViewportOverlayFlags.NAMES not in combined

    def test_overlay_flags_removable(self):
        """Overlay flags can be removed."""
        combined = ViewportOverlayFlags.GRID | ViewportOverlayFlags.GIZMOS | ViewportOverlayFlags.NAMES
        reduced = combined & ~ViewportOverlayFlags.NAMES

        assert ViewportOverlayFlags.GRID in reduced
        assert ViewportOverlayFlags.NAMES not in reduced
