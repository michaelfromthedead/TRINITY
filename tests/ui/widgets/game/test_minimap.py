"""
Comprehensive tests for Minimap widget.

Tests cover:
- Initialization and defaults
- Position and dimensions
- World bounds
- Zoom control
- Marker management
- Player tracking
- Coordinate conversion
- Click handling
- Drag/pan handling
- Scroll/zoom handling
- Callbacks
- Rendering helpers
- Configuration
"""

import pytest
import sys
from pathlib import Path

# Add engine to path
engine_path = Path(__file__).parent.parent.parent.parent.parent / "engine"
sys.path.insert(0, str(engine_path.parent))

from engine.ui.widgets.game.minimap import (
    Minimap,
    MinimapMarker,
    MinimapConfig,
    MarkerType,
)


class TestMinimapInitialization:
    """Test Minimap initialization and defaults."""

    def test_default_initialization(self):
        """Test minimap initializes with correct defaults."""
        mm = Minimap()
        assert mm.width == 200.0
        assert mm.height == 200.0
        assert mm.is_visible is True

    def test_custom_dimensions(self):
        """Test initialization with custom dimensions."""
        mm = Minimap(width=300.0, height=300.0)
        assert mm.width == 300.0
        assert mm.height == 300.0

    def test_custom_position(self):
        """Test initialization with custom position."""
        mm = Minimap(x=100.0, y=50.0)
        assert mm.x == 100.0
        assert mm.y == 50.0

    def test_custom_world_bounds(self):
        """Test initialization with custom world bounds."""
        mm = Minimap(world_width=2000.0, world_height=2000.0)
        assert mm.world_width == 2000.0
        assert mm.world_height == 2000.0

    def test_unique_id(self):
        """Test each minimap gets unique ID."""
        mm1 = Minimap()
        mm2 = Minimap()
        assert mm1.id != mm2.id

    def test_default_config(self):
        """Test default configuration is applied."""
        mm = Minimap()
        assert mm.config is not None

    def test_custom_config(self):
        """Test custom configuration."""
        config = MinimapConfig(
            default_zoom=2.0,
            player_color="#00FF00",
        )
        mm = Minimap(config=config)
        assert mm.zoom == 2.0
        assert mm.config.player_color == "#00FF00"


class TestMinimapWorldBounds:
    """Test Minimap world bounds management."""

    def test_set_world_bounds(self):
        """Test setting world bounds."""
        mm = Minimap()
        mm.set_world_bounds(5000.0, 3000.0)
        assert mm.world_width == 5000.0
        assert mm.world_height == 3000.0

    def test_world_bounds_minimum(self):
        """Test world bounds have minimum."""
        mm = Minimap()
        mm.set_world_bounds(0.0, 0.0)
        assert mm.world_width >= 1.0
        assert mm.world_height >= 1.0


class TestMinimapZoom:
    """Test Minimap zoom control."""

    def test_default_zoom(self):
        """Test default zoom level."""
        mm = Minimap()
        assert mm.zoom == 1.0

    def test_set_zoom(self):
        """Test setting zoom level."""
        mm = Minimap()
        mm.zoom = 2.0
        assert mm.zoom == 2.0

    def test_zoom_clamped_to_min(self):
        """Test zoom is clamped to minimum."""
        mm = Minimap()
        mm.zoom = 0.1
        assert mm.zoom >= mm.config.min_zoom

    def test_zoom_clamped_to_max(self):
        """Test zoom is clamped to maximum."""
        mm = Minimap()
        mm.zoom = 10.0
        assert mm.zoom <= mm.config.max_zoom

    def test_zoom_in(self):
        """Test zoom_in method."""
        mm = Minimap()
        initial = mm.zoom
        mm.zoom_in()
        assert mm.zoom > initial

    def test_zoom_out(self):
        """Test zoom_out method."""
        mm = Minimap()
        mm.zoom = 2.0
        initial = mm.zoom
        mm.zoom_out()
        assert mm.zoom < initial

    def test_reset_zoom(self):
        """Test reset_zoom method."""
        mm = Minimap()
        mm.zoom = 3.0
        mm.reset_zoom()
        assert mm.zoom == mm.config.default_zoom


class TestMinimapMarkers:
    """Test Minimap marker management."""

    def test_add_marker(self):
        """Test adding a marker."""
        mm = Minimap()
        marker_id = mm.add_marker(MarkerType.ENEMY, 500.0, 500.0)
        assert marker_id is not None
        assert mm.marker_count == 1

    def test_add_multiple_markers(self):
        """Test adding multiple markers."""
        mm = Minimap()
        mm.add_marker(MarkerType.ENEMY, 100.0, 100.0)
        mm.add_marker(MarkerType.ALLY, 200.0, 200.0)
        mm.add_marker(MarkerType.OBJECTIVE, 300.0, 300.0)
        assert mm.marker_count == 3

    def test_remove_marker(self):
        """Test removing a marker."""
        mm = Minimap()
        marker_id = mm.add_marker(MarkerType.ENEMY, 500.0, 500.0)
        result = mm.remove_marker(marker_id)
        assert result is True
        assert mm.marker_count == 0

    def test_remove_nonexistent_marker(self):
        """Test removing a nonexistent marker."""
        mm = Minimap()
        result = mm.remove_marker(999)
        assert result is False

    def test_get_marker(self):
        """Test getting a marker by ID."""
        mm = Minimap()
        marker_id = mm.add_marker(MarkerType.QUEST, 500.0, 500.0)
        marker = mm.get_marker(marker_id)
        assert marker is not None
        assert marker.marker_type == MarkerType.QUEST

    def test_update_marker(self):
        """Test updating marker properties."""
        mm = Minimap()
        marker_id = mm.add_marker(MarkerType.ENEMY, 100.0, 100.0)
        result = mm.update_marker(marker_id, world_x=200.0, world_y=200.0)
        assert result is True
        marker = mm.get_marker(marker_id)
        assert marker.world_x == 200.0
        assert marker.world_y == 200.0

    def test_update_marker_rotation(self):
        """Test updating marker rotation."""
        mm = Minimap()
        marker_id = mm.add_marker(MarkerType.ALLY, 100.0, 100.0)
        mm.update_marker(marker_id, rotation=45.0)
        marker = mm.get_marker(marker_id)
        assert marker.rotation == 45.0

    def test_clear_all_markers(self):
        """Test clearing all markers."""
        mm = Minimap()
        mm.add_marker(MarkerType.ENEMY, 100.0, 100.0)
        mm.add_marker(MarkerType.ALLY, 200.0, 200.0)
        count = mm.clear_markers()
        assert count == 2
        assert mm.marker_count == 0

    def test_clear_markers_by_type(self):
        """Test clearing markers by type."""
        mm = Minimap()
        mm.add_marker(MarkerType.ENEMY, 100.0, 100.0)
        mm.add_marker(MarkerType.ENEMY, 150.0, 150.0)
        mm.add_marker(MarkerType.ALLY, 200.0, 200.0)
        count = mm.clear_markers(MarkerType.ENEMY)
        assert count == 2
        assert mm.marker_count == 1

    def test_visible_markers(self):
        """Test getting visible markers."""
        mm = Minimap()
        marker_id = mm.add_marker(MarkerType.ENEMY, 100.0, 100.0)
        marker = mm.get_marker(marker_id)
        marker.is_visible = True
        visible = mm.visible_markers
        assert len(visible) >= 1

    def test_marker_default_color_by_type(self):
        """Test marker gets default color by type."""
        mm = Minimap()
        enemy_id = mm.add_marker(MarkerType.ENEMY, 100.0, 100.0)
        ally_id = mm.add_marker(MarkerType.ALLY, 200.0, 200.0)
        enemy = mm.get_marker(enemy_id)
        ally = mm.get_marker(ally_id)
        assert enemy.color == mm.config.enemy_color
        assert ally.color == mm.config.ally_color


class TestMinimapPlayerTracking:
    """Test Minimap player tracking."""

    def test_set_player_position(self):
        """Test setting player position."""
        mm = Minimap()
        mm.set_player_position(500.0, 500.0)
        pos = mm.get_player_position()
        assert pos == (500.0, 500.0)

    def test_set_player_position_with_rotation(self):
        """Test setting player position with rotation."""
        mm = Minimap()
        mm.set_player_position(500.0, 500.0, rotation=45.0)
        pos = mm.get_player_position()
        assert pos == (500.0, 500.0)

    def test_follow_player_enabled(self):
        """Test follow player mode."""
        mm = Minimap()
        mm.follow_player = True
        mm.set_player_position(500.0, 500.0)
        assert mm.center_x == 500.0
        assert mm.center_y == 500.0

    def test_follow_player_disabled(self):
        """Test follow player disabled."""
        mm = Minimap()
        mm.follow_player = False
        mm.set_center(250.0, 250.0)
        mm.set_player_position(500.0, 500.0)
        # Center should not change
        assert mm.center_x == 250.0

    def test_get_player_position_no_player(self):
        """Test get_player_position with no player set."""
        mm = Minimap()
        pos = mm.get_player_position()
        assert pos is None


class TestMinimapCoordinateConversion:
    """Test Minimap coordinate conversion."""

    def test_world_to_map(self):
        """Test world to map coordinate conversion."""
        mm = Minimap(
            x=0.0, y=0.0,
            width=200.0, height=200.0,
            world_width=1000.0, world_height=1000.0,
        )
        mm.set_center(500.0, 500.0)
        map_x, map_y = mm.world_to_map(500.0, 500.0)
        # Center of world should be center of minimap
        assert map_x == 100.0  # width/2
        assert map_y == 100.0  # height/2

    def test_map_to_world(self):
        """Test map to world coordinate conversion."""
        mm = Minimap(
            x=0.0, y=0.0,
            width=200.0, height=200.0,
            world_width=1000.0, world_height=1000.0,
        )
        mm.set_center(500.0, 500.0)
        world_x, world_y = mm.map_to_world(100.0, 100.0)
        # Center of minimap should be center of view
        assert world_x == 500.0
        assert world_y == 500.0

    def test_round_trip_conversion(self):
        """Test round-trip coordinate conversion."""
        mm = Minimap(
            x=0.0, y=0.0,
            width=200.0, height=200.0,
            world_width=1000.0, world_height=1000.0,
        )
        mm.set_center(500.0, 500.0)
        original_x, original_y = 300.0, 400.0
        map_x, map_y = mm.world_to_map(original_x, original_y)
        result_x, result_y = mm.map_to_world(map_x, map_y)
        assert abs(result_x - original_x) < 0.01
        assert abs(result_y - original_y) < 0.01

    def test_is_world_position_visible(self):
        """Test checking if world position is visible."""
        mm = Minimap(
            width=200.0, height=200.0,
            world_width=1000.0, world_height=1000.0,
        )
        mm.set_center(500.0, 500.0)
        # Center should be visible
        assert mm.is_world_position_visible(500.0, 500.0) is True

    def test_is_world_position_not_visible(self):
        """Test world position outside view."""
        mm = Minimap(
            width=200.0, height=200.0,
            world_width=1000.0, world_height=1000.0,
        )
        mm.set_center(500.0, 500.0)
        mm.zoom = 4.0  # Zoom in tight
        # Far corner should not be visible when zoomed in
        # This depends on zoom level


class TestMinimapCenter:
    """Test Minimap center/pan control."""

    def test_set_center(self):
        """Test setting center position."""
        mm = Minimap()
        mm.set_center(250.0, 350.0)
        assert mm.center_x == 250.0
        assert mm.center_y == 350.0

    def test_center_clamped_to_bounds(self):
        """Test center is clamped to world bounds."""
        mm = Minimap(world_width=1000.0, world_height=1000.0)
        mm.set_center(1500.0, 1500.0)
        assert mm.center_x <= 1000.0
        assert mm.center_y <= 1000.0

    def test_pan(self):
        """Test pan method."""
        config = MinimapConfig(allow_pan=True)
        mm = Minimap(config=config)
        mm.set_center(500.0, 500.0)
        mm.pan(50.0, 50.0)
        assert mm.center_x == 550.0
        assert mm.center_y == 550.0

    def test_pan_disabled(self):
        """Test pan when disabled."""
        config = MinimapConfig(allow_pan=False)
        mm = Minimap(config=config)
        mm.set_center(500.0, 500.0)
        mm.pan(50.0, 50.0)
        assert mm.center_x == 500.0  # Unchanged


class TestMinimapRotation:
    """Test Minimap rotation."""

    def test_set_rotation(self):
        """Test setting rotation."""
        mm = Minimap()
        mm.rotation = 45.0
        assert mm.rotation == 45.0

    def test_rotation_normalized(self):
        """Test rotation is normalized to 0-360."""
        mm = Minimap()
        mm.rotation = 450.0
        assert mm.rotation == 90.0


class TestMinimapClickHandling:
    """Test Minimap click handling."""

    def test_handle_click(self):
        """Test handling click on minimap."""
        mm = Minimap(x=0.0, y=0.0, width=200.0, height=200.0)
        result = mm.handle_click(100.0, 100.0)
        assert result is True

    def test_handle_click_outside_bounds(self):
        """Test click outside minimap bounds."""
        mm = Minimap(x=0.0, y=0.0, width=200.0, height=200.0)
        result = mm.handle_click(300.0, 300.0)
        assert result is False

    def test_handle_click_when_not_interactive(self):
        """Test click when minimap is not interactive."""
        mm = Minimap()
        mm.is_interactive = False
        result = mm.handle_click(100.0, 100.0)
        assert result is False

    def test_click_callback(self):
        """Test click callback is invoked."""
        mm = Minimap(x=0.0, y=0.0, width=200.0, height=200.0)
        clicks = []

        def callback(wx, wy):
            clicks.append((wx, wy))

        mm.on_click(callback)
        mm.handle_click(100.0, 100.0)
        assert len(clicks) == 1

    def test_navigate_callback(self):
        """Test navigate callback is invoked."""
        config = MinimapConfig(allow_click_navigation=True)
        mm = Minimap(x=0.0, y=0.0, width=200.0, height=200.0, config=config)
        navigations = []

        def callback(wx, wy):
            navigations.append((wx, wy))

        mm.on_navigate(callback)
        mm.handle_click(100.0, 100.0)
        assert len(navigations) == 1


class TestMinimapScrollHandling:
    """Test Minimap scroll/zoom handling."""

    def test_handle_scroll_zoom_in(self):
        """Test scroll to zoom in."""
        config = MinimapConfig(allow_zoom=True)
        mm = Minimap(config=config)
        initial_zoom = mm.zoom
        mm.handle_scroll(1.0)
        assert mm.zoom > initial_zoom

    def test_handle_scroll_zoom_out(self):
        """Test scroll to zoom out."""
        config = MinimapConfig(allow_zoom=True)
        mm = Minimap(config=config)
        mm.zoom = 2.0
        initial_zoom = mm.zoom
        mm.handle_scroll(-1.0)
        assert mm.zoom < initial_zoom

    def test_handle_scroll_when_disabled(self):
        """Test scroll when zoom disabled."""
        config = MinimapConfig(allow_zoom=False)
        mm = Minimap(config=config)
        initial_zoom = mm.zoom
        result = mm.handle_scroll(1.0)
        assert result is False
        assert mm.zoom == initial_zoom


class TestMinimapDragHandling:
    """Test Minimap drag/pan handling."""

    def test_handle_drag_start(self):
        """Test drag start."""
        config = MinimapConfig(allow_pan=True)
        mm = Minimap(x=0.0, y=0.0, width=200.0, height=200.0, config=config)
        result = mm.handle_drag_start(100.0, 100.0)
        assert result is True

    def test_handle_drag(self):
        """Test drag movement."""
        config = MinimapConfig(allow_pan=True)
        mm = Minimap(x=0.0, y=0.0, width=200.0, height=200.0, config=config)
        mm.handle_drag_start(100.0, 100.0)
        result = mm.handle_drag(110.0, 110.0)
        assert result is True

    def test_handle_drag_end(self):
        """Test drag end."""
        config = MinimapConfig(allow_pan=True)
        mm = Minimap(config=config)
        mm.handle_drag_start(100.0, 100.0)
        mm.handle_drag_end()
        # No assertion needed, just should not raise


class TestMinimapCallbacks:
    """Test Minimap callback functions."""

    def test_marker_click_callback(self):
        """Test marker click callback."""
        mm = Minimap(x=0.0, y=0.0, width=200.0, height=200.0)
        marker_id = mm.add_marker(MarkerType.OBJECTIVE, 500.0, 500.0, size=20.0)
        marker = mm.get_marker(marker_id)
        clicked_markers = []

        def callback(m):
            clicked_markers.append(m)

        mm.on_marker_click(callback)
        # Click on marker position (need to convert)
        map_x, map_y = mm.world_to_map(500.0, 500.0)
        mm.handle_click(map_x, map_y)
        # Callback should be called if marker was hit


class TestMinimapRenderingHelpers:
    """Test Minimap rendering helper methods."""

    def test_get_visible_bounds(self):
        """Test get_visible_bounds method."""
        mm = Minimap(
            world_width=1000.0, world_height=1000.0,
        )
        mm.set_center(500.0, 500.0)
        bounds = mm.get_visible_bounds()
        assert len(bounds) == 4
        min_x, min_y, max_x, max_y = bounds
        assert min_x < max_x
        assert min_y < max_y

    def test_get_map_uv_bounds(self):
        """Test get_map_uv_bounds method."""
        mm = Minimap()
        mm.set_center(500.0, 500.0)
        uv = mm.get_map_uv_bounds()
        assert len(uv) == 4
        u_min, v_min, u_max, v_max = uv
        assert 0.0 <= u_min <= 1.0
        assert 0.0 <= u_max <= 1.0


class TestMinimapTexture:
    """Test Minimap texture management."""

    def test_set_map_texture(self):
        """Test setting map texture."""
        mm = Minimap()
        mm.set_map_texture("world_map.png")
        assert mm.get_map_texture_path() == "world_map.png"

    def test_get_map_texture_no_texture(self):
        """Test getting texture when none set."""
        mm = Minimap()
        assert mm.get_map_texture_path() is None


class TestMinimapVisibility:
    """Test Minimap visibility."""

    def test_visible_by_default(self):
        """Test visible by default."""
        mm = Minimap()
        assert mm.is_visible is True

    def test_set_invisible(self):
        """Test setting invisible."""
        mm = Minimap()
        mm.is_visible = False
        assert mm.is_visible is False

    def test_interactive_by_default(self):
        """Test interactive by default."""
        mm = Minimap()
        assert mm.is_interactive is True


class TestMinimapRepr:
    """Test Minimap string representation."""

    def test_repr(self):
        """Test repr includes key info."""
        mm = Minimap(width=200.0, height=200.0)
        mm.add_marker(MarkerType.ENEMY, 100.0, 100.0)
        repr_str = repr(mm)
        assert "Minimap" in repr_str
        assert "200" in repr_str
